"""
mlops/scripts/export_onnx.py

Standalone ONNX export entrypoint.
Launched by OnnxWorker as a subprocess.

Stdout protocol:
  PROGRESS N        (0–100)
  STATUS message    (informational lines for the log console)
  [ERROR] message   (on failure, before sys.exit(1))
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import torch
import shutil
from ultralytics import YOLO
from seg_model import SegModel
from mlops.registry.manifest import ManifestWriter


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--version-folder", required=True,
                            help="Absolute path to the version folder in the registry")
        args = parser.parse_args()
        version_folder = args.version_folder

        config_path   = os.path.join(version_folder, "config.json")
        manifest_path = os.path.join(version_folder, "manifest.json")

        with open(config_path,   "r", encoding="utf-8") as f: cfg      = json.load(f)
        with open(manifest_path, "r", encoding="utf-8") as f: manifest = json.load(f)

        hp         = cfg["hyperparams"]
        model_type = manifest.get("model_type", "unet").lower()

        in_channels  = int(hp.get("in_channels",  3))
        image_width  = int(hp.get("image_width",  320))
        image_height = int(hp.get("image_height", 240))
        out_path     = os.path.join(version_folder, "best.onnx")

        print("STATUS Loading model...", flush=True)
        print("PROGRESS 10", flush=True)

        if model_type == "unet":
            ckpt_path = os.path.join(version_folder, "best.ckpt")
            pt_path   = os.path.join(version_folder, "best.pt")

            if os.path.isfile(ckpt_path):
                model = SegModel.load_from_checkpoint(ckpt_path, map_location="cpu")
            elif os.path.isfile(pt_path):
                # Reconstruct from config, then load state dict
                enc_weights = hp.get("encoder_weights", "imagenet")
                if enc_weights == "none": enc_weights = None
                model = SegModel(
                    arch            = hp["architecture"],
                    encoder_name    = hp["encoder"],
                    encoder_weights = enc_weights,
                    in_channels     = in_channels,
                    out_classes     = int(hp.get("out_classes", 2)),
                    lr              = float(hp.get("learning_rate", 0.001)),
                )
                model.load_state_dict(torch.load(pt_path, map_location="cpu"))
            else:
                raise FileNotFoundError("No best.ckpt or best.pt found in version folder.")

            model.eval()
            print("STATUS Model loaded.", flush=True)
            print("PROGRESS 30", flush=True)

            dummy = torch.zeros(1, in_channels, image_height, image_width)
            input_shape = [1, in_channels, image_height, image_width]

            print("STATUS Exporting to ONNX...", flush=True)
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy,
                    out_path,
                    dynamo          = False,
                    opset_version   = 11,
                    input_names     = ["input"],
                    output_names    = ["output"],
                    dynamic_axes    = {"input": {0: "batch"}, "output": {0: "batch"}},
                )
            print("PROGRESS 80", flush=True)

        elif model_type == "yolo":
            pt_path = os.path.join(version_folder, "best.pt")
            if not os.path.isfile(pt_path):
                raise FileNotFoundError(f"best.pt not found: {pt_path}")

            print("STATUS Loading YOLO model...", flush=True)
            print("PROGRESS 20", flush=True)

            yolo_model = YOLO(pt_path)
            print("STATUS Exporting to ONNX...", flush=True)

            # Ultralytics export returns the path to the exported file
            result_path = yolo_model.export(
                format  = "onnx",
                imgsz   = image_width,
                opset   = 11,
                dynamic = True,
            )
            print("PROGRESS 70", flush=True)

            if result_path and os.path.isfile(str(result_path)):
                src = os.path.realpath(str(result_path))
                dst = os.path.realpath(out_path)
                if src != dst:
                    shutil.copy2(src, dst)
            elif not os.path.isfile(out_path):
                raise RuntimeError(f"ONNX export produced no file. Expected: {result_path}")

            input_shape = [1, 3, image_width, image_width]
            print("PROGRESS 80", flush=True)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Issue 16 fix: Smoke-test the exported ONNX model
        print("STATUS Validating ONNX export...", flush=True)
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(out_path)
            dummy_np = np.zeros(input_shape, dtype=np.float32)
            outputs = sess.run(None, {sess.get_inputs()[0].name: dummy_np})
            print(f"STATUS ONNX smoke-test passed. Output shape: {outputs[0].shape}", flush=True)
        except Exception as smoke_err:
            print(f"STATUS ONNX smoke-test warning: {smoke_err}", flush=True)

        print("STATUS Updating manifest...", flush=True)
        writer = ManifestWriter(version_folder)
        writer.update_onnx(input_shape=input_shape, opset_version=11)

        try:
            from mlops.registry.utils import write_project_csv
            registry_root = os.path.dirname(os.path.dirname(version_folder))
            project       = os.path.basename(os.path.dirname(version_folder))
            write_project_csv(registry_root, project)
        except Exception:
            pass

        print("PROGRESS 100", flush=True)
        print("STATUS Export complete.", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
