"""
mlops/scripts/infer_yolo.py

Inference entrypoint for YOLO detection/segmentation.
Accepts a .pt, .onnx, or .torchscript model path via --onnx.
Prefers best.pt over exported formats (more reliable for seg models).

Stdout protocol:
  PROGRESS N
  RESULT {name}|{abs_path_to_annotated}|{abs_path_to_original}
  DONE total=N
  [ERROR] message
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import time
import cv2

_VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--onnx",        required=True)
        parser.add_argument("--config",      required=True)
        parser.add_argument("--test-folder", required=True)
        parser.add_argument("--out",         required=True)
        parser.add_argument("--conf",        type=float, default=0.25)
        args = parser.parse_args()

        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        hp = cfg["hyperparams"]

        image_width = int(hp.get("image_width", 640))
        arch        = str(hp.get("architecture", "yolov8n"))
        yolo_task   = "segment" if arch.endswith("-seg") else "detect"

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

        from ultralytics import YOLO

        # Prefer .pt over exported formats — exported seg models can silently
        # produce near-zero confidence scores due to output format differences.
        # Ultralytics' YOLO() auto-detects .onnx vs .torchscript from the extension.
        pt_candidate = os.path.splitext(args.onnx)[0] + ".pt"
        if os.path.isfile(pt_candidate):
            model = YOLO(pt_candidate)
            print(f"[INFO] Using .pt model: {pt_candidate}", flush=True)
        else:
            model = YOLO(args.onnx, task=yolo_task)
            print(f"[INFO] Using exported model: {args.onnx}", flush=True)

        print(f"[INFO] Task: {yolo_task}  arch: {arch}  imgsz: {image_width}  conf: {args.conf}", flush=True)
        total = len(images)
        total_infer_time = 0.0

        for i, img_name in enumerate(images):
            img_path = os.path.join(test_folder, img_name)
            if not os.path.isfile(img_path):
                print(f"[WARNING] Skipping missing file: {img_path}", flush=True)
                continue

            t0 = time.perf_counter()
            results = model(img_path, imgsz=image_width, conf=args.conf, verbose=False)
            dt = (time.perf_counter() - t0) * 1000  # milliseconds
            total_infer_time += dt

            boxes = results[0].boxes
            n_det = len(boxes) if boxes is not None else 0
            if n_det > 0:
                max_conf = float(boxes.conf.max())
                print(f"[INFO] {img_name}: {n_det} det(s), max_conf={max_conf:.3f}, {dt:.1f}ms", flush=True)
            else:
                # Probe at near-zero conf to diagnose threshold vs. no-detection
                probe       = model(img_path, imgsz=image_width, conf=0.001, verbose=False)
                probe_boxes = probe[0].boxes
                if probe_boxes is not None and len(probe_boxes) > 0:
                    peak = float(probe_boxes.conf.max())
                    print(f"[WARN] {img_name}: 0 det at conf={args.conf}, "
                          f"peak_conf={peak:.4f} — lower threshold below {peak:.2f}", flush=True)
                else:
                    print(f"[WARN] {img_name}: 0 detections even at conf=0.001", flush=True)

            annotated = results[0].plot()
            out_path  = os.path.join(args.out, img_name)
            cv2.imwrite(out_path, annotated)

            pct = int((i + 1) / total * 100)
            print(f"PROGRESS {pct}", flush=True)
            # Protocol: name | original_path | annotated_path
            print(f"RESULT {img_name}|{img_path}|{out_path}", flush=True)

        if total > 0:
            avg_ms = total_infer_time / total
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
            print(f"[INFO] Inference time: {avg_ms:.1f}ms/image avg ({fps:.1f} FPS)", flush=True)

        print(f"DONE total={total}", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
