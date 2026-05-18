"""
mlops/scripts/infer_yolo.py

ONNX inference entrypoint for YOLO detection.
Uses Ultralytics to load and run inference (handles NMS internally).

Stdout protocol:
  PROGRESS N
  RESULT {name}|{abs_path_to_annotated_image}
  DONE total=N
  [ERROR] message
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import cv2

_VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


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

        image_width = int(hp.get("image_width", 640))

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
        model = YOLO(args.onnx)   # Ultralytics can load ONNX directly
        total = len(images)

        for i, img_name in enumerate(images):
            img_path = os.path.join(args.test_folder, img_name)
            if not os.path.isfile(img_path):
                print(f"[WARNING] Skipping missing file: {img_path}", flush=True)
                continue

            results  = model(img_path, imgsz=image_width, verbose=False)
            annotated = results[0].plot()    # BGR numpy array with boxes drawn

            out_path = os.path.join(args.out, img_name)
            cv2.imwrite(out_path, annotated)

            pct = int((i + 1) / total * 100)
            print(f"PROGRESS {pct}", flush=True)
            print(f"RESULT {img_name}|{out_path}", flush=True)

        print(f"DONE total={total}", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
