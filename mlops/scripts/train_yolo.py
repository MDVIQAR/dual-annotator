"""
mlops/scripts/train_yolo.py

Entrypoint for YOLOv8 detection training via Ultralytics.
Launched by TrainWorker as a subprocess.

Stdout protocol (same as UNet):
  METRIC epoch=N train_loss=X val_loss=Y
  PROGRESS N
  METRICS_FINAL best_val_loss=X best_epoch=N final_train_loss=Y
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import shutil
from ultralytics import YOLO


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--out", required=True)
        args = parser.parse_args()

        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        hp       = cfg["hyperparams"]
        out      = args.out
        arch     = hp["architecture"]
        epochs   = hp["epochs"]
        batch    = hp["batch_size"]
        img_w    = hp["image_width"]
        device   = hp.get("device", "cpu")
        yaml_path = os.path.join(cfg["dataset_folder"], "data.yaml")

        model = YOLO(f"{arch}.pt")

        best_val_loss   = float("inf")
        best_epoch      = 1
        last_train_loss = 0.0

        def on_fit_epoch_end(trainer):
            nonlocal best_val_loss, best_epoch, last_train_loss
            epoch = trainer.epoch + 1
            train_loss = float(trainer.loss) if hasattr(trainer, "loss") else 0.0
            
            val_metrics = trainer.metrics or {}
            val_loss = float(val_metrics.get("val/box_loss", val_metrics.get("metrics/mAP50(B)", 0.0)))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
            last_train_loss = train_loss

            pct = int(epoch / epochs * 90)
            print(f"METRIC epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
            print(f"PROGRESS {min(pct, 90)}", flush=True)

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(
            data    = yaml_path,
            epochs  = epochs,
            batch   = batch,
            imgsz   = img_w,
            device  = device,
            project = out,
            name    = "run",
            exist_ok= True,
            verbose = False,
        )

        yolo_best = os.path.join(out, "run", "weights", "best.pt")
        if os.path.isfile(yolo_best):
            shutil.copy2(yolo_best, os.path.join(out, "best.pt"))

        print(
            f"METRICS_FINAL "
            f"best_val_loss={best_val_loss:.6f} "
            f"best_epoch={best_epoch} "
            f"final_train_loss={last_train_loss:.6f}",
            flush=True,
        )
        print("PROGRESS 100", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
