"""
mlops/scripts/infer_unet.py

ONNX / PyTorch inference entrypoint for UNet segmentation.
Launched by InferWorker as a subprocess.

Stdout protocol:
  PROGRESS N
  RESULT {name}|{orig_path}|{mask_path}|{overlay_path}
  DONE total=N
  [ERROR] message
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import cv2
import numpy as np

_VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def preprocess(img_bgr, in_channels, image_width, image_height):
    if in_channels == 1:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_AREA)
        img = img[..., None]
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_AREA)

    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - 0.5) / 0.5
    return img_f.transpose(2, 0, 1)[None]  # 1, C, H, W


def colorise_mask(mask_hw, out_classes):
    scale = max(255 // max(out_classes - 1, 1), 1)
    return cv2.applyColorMap((mask_hw * scale).astype(np.uint8), cv2.COLORMAP_JET)


def _build_torch_runner(onnx_path, cfg):
    """Load SegModel from a .pt checkpoint. Returns callable: batch_np -> logits_np."""
    import torch
    from seg_model import SegModel

    hp          = cfg["hyperparams"]
    arch        = hp.get("architecture", "unet")
    encoder     = hp.get("encoder", "resnet34")
    enc_weights = hp.get("encoder_weights", "none")
    in_ch       = int(hp.get("in_channels", 3))
    out_cl      = int(hp.get("out_classes", 2))

    model = SegModel(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=enc_weights,
        in_channels=in_ch,
        out_classes=out_cl,
    )
    ckpt  = torch.load(onnx_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    def run(batch_np):
        with torch.no_grad():
            return model(torch.from_numpy(batch_np)).numpy()

    return run


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--onnx",        required=True)
        parser.add_argument("--config",      required=True)
        parser.add_argument("--test-folder", required=True)
        parser.add_argument("--out",         required=True)
        args = parser.parse_args()

        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        hp = cfg["hyperparams"]

        in_channels  = int(hp.get("in_channels",  3))
        image_width  = int(hp.get("image_width",  320))
        image_height = int(hp.get("image_height", 240))
        out_classes  = int(hp.get("out_classes",  2))

        # Build inference callable — ONNX or PyTorch
        if args.onnx.lower().endswith(".onnx"):
            try:
                import onnxruntime as ort
            except ImportError:
                print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime", flush=True)
                sys.exit(1)
            sess       = ort.InferenceSession(args.onnx)
            input_name = sess.get_inputs()[0].name
            def run_inference(batch):
                return sess.run(None, {input_name: batch})[0]
            print(f"[INFO] Using ONNX model: {args.onnx}", flush=True)
        else:
            run_inference = _build_torch_runner(args.onnx, cfg)
            print(f"[INFO] Using PyTorch model: {args.onnx}", flush=True)

        # Auto-detect dataset layout: test/ or test/images/
        test_folder = args.test_folder
        images_sub  = os.path.join(test_folder, "images")
        if os.path.isdir(images_sub):
            test_folder = images_sub

        images = sorted(
            f for f in os.listdir(test_folder)
            if os.path.splitext(f)[1].lower() in _VALID_EXTS
        )
        if not images:
            raise ValueError(f"No images found in: {test_folder}")

        os.makedirs(args.out, exist_ok=True)
        print("PROGRESS 0", flush=True)

        total = len(images)

        for i, img_name in enumerate(images):
            img_path = os.path.join(test_folder, img_name)
            img_bgr  = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[WARNING] Skipping unreadable: {img_path}", flush=True)
                continue

            orig_w, orig_h = img_bgr.shape[1], img_bgr.shape[0]

            batch  = preprocess(img_bgr, in_channels, image_width, image_height)
            logits = run_inference(batch)  # 1, C, H, W

            if out_classes == 1:
                pred = (logits[0, 0] > 0).astype(np.uint8)
            else:
                pred = np.argmax(logits[0], axis=0).astype(np.uint8)

            pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            colored      = colorise_mask(pred_resized, out_classes)
            orig_display = cv2.resize(img_bgr, (orig_w, orig_h))
            overlay      = cv2.addWeighted(orig_display, 0.6, colored, 0.4, 0)

            # Save three separate output images
            stem, ext    = os.path.splitext(img_name)
            orig_path    = os.path.join(args.out, f"{stem}_orig{ext}")
            mask_path    = os.path.join(args.out, f"{stem}_mask{ext}")
            overlay_path = os.path.join(args.out, f"{stem}_overlay{ext}")
            cv2.imwrite(orig_path,    orig_display)
            cv2.imwrite(mask_path,    colored)
            cv2.imwrite(overlay_path, overlay)

            pct = int((i + 1) / total * 100)
            print(f"PROGRESS {pct}", flush=True)
            print(f"RESULT {img_name}|{orig_path}|{mask_path}|{overlay_path}", flush=True)

        print(f"DONE total={total}", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
