"""
mlops/scripts/train_yolo.py

YOLOv8 detection / segmentation training entrypoint.
Launched by TrainWorker as a subprocess.

Stdout protocol:
  METRIC epoch=N train_loss=X val_loss=Y train_iou=0.0 val_iou=Z map50=Z [mask_map50=W]
  PROGRESS N
  METRICS_FINAL best_val_loss=X best_epoch=N final_train_loss=Y
  [INFO]  informational
  [WARN]  non-fatal warning
  [ERROR] fatal — script exits 1
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import shutil


# ── Label validator ────────────────────────────────────────────────────────────

def _validate_labels(dataset_folder: str, nc: int, is_seg: bool) -> None:
    """
    Scan up to 100 .txt label files and verify:
      1. All class indices are in [0, nc-1]
      2. For seg models: labels contain polygon data (>5 values per line)

    Prints [WARN] or raises ValueError on hard errors.
    """
    # Support both dataset layouts
    candidates = [
        os.path.join(dataset_folder, "labels", "train"),
        os.path.join(dataset_folder, "train",  "labels"),
    ]
    labels_dir = next((d for d in candidates if os.path.isdir(d)), None)
    if labels_dir is None:
        print(f"[WARN] Labels folder not found — skipping validation.", flush=True)
        return

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    if not txt_files:
        print("[WARN] No .txt label files found — skipping validation.", flush=True)
        return

    sample = txt_files[:100]
    max_class = -1
    has_polygon = False
    total_lines = 0

    for fname in sample:
        path = os.path.join(labels_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_idx = int(float(parts[0]))
                    except (ValueError, IndexError):
                        continue
                    max_class = max(max_class, cls_idx)
                    if len(parts) > 5:
                        has_polygon = True
                    total_lines += 1
        except OSError:
            continue

    if max_class < 0:
        print("[WARN] Label files appear empty — skipping validation.", flush=True)
        return

    # Hard error: class index out of range
    if max_class >= nc:
        raise ValueError(
            f"Label class index {max_class} found, but data.yaml has nc={nc} "
            f"(valid indices: 0 – {nc - 1}).\n"
            f"  Fix: re-export your dataset from Roboflow as 'YOLOv8' format, "
            f"or update the Class Names field in Data Prep to match all classes in your labels."
        )

    print(f"[INFO] Labels OK — {len(sample)} files checked, "
          f"max class index: {max_class}, nc: {nc}", flush=True)

    # Soft warning: seg model but bbox-only labels
    if is_seg and not has_polygon:
        print(
            f"[WARN] Architecture ends in '-seg' but all labels have exactly 5 values "
            f"(bounding-box format). Segmentation masks will NOT be trained.\n"
            f"  Fix: re-export from Roboflow as 'YOLOv8 Segmentation (OBB or Instance Seg)' "
            f"so label files contain polygon coordinates.",
            flush=True,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_nc_from_yaml(yaml_path: str) -> int:
    """Read nc field from data.yaml. Returns 1 if not found."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("nc:"):
                    return int(stripped.split(":", 1)[1].strip())
    except Exception:
        pass
    return 1


def _save_best_pt(out: str) -> None:
    """Copy run/weights/best.pt → {out}/best.pt. Raises if neither exists."""
    src = os.path.join(out, "run", "weights", "best.pt")
    dst = os.path.join(out, "best.pt")
    if os.path.isfile(src):
        if os.path.realpath(src) != os.path.realpath(dst):
            shutil.copy2(src, dst)
        print(f"[INFO] best.pt saved → {dst}", flush=True)
    elif not os.path.isfile(dst):
        raise FileNotFoundError(
            f"YOLO did not produce best.pt at: {src}\n"
            "Training may have completed 0 epochs. "
            "Check for data.yaml or label errors above."
        )


def _emit_final(best_val_loss: float, best_epoch: int, final_train_loss: float) -> None:
    print(
        f"METRICS_FINAL"
        f" best_val_loss={best_val_loss:.6f}"
        f" best_epoch={best_epoch}"
        f" final_train_loss={final_train_loss:.6f}",
        flush=True,
    )
    print("PROGRESS 100", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--out",    required=True)
        args = parser.parse_args()

        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

        hp  = cfg["hyperparams"]
        out = args.out

        arch   = str(hp["architecture"])
        epochs = int(hp["epochs"])
        batch  = int(hp["batch_size"])
        img_w  = int(hp["image_width"])
        device = str(hp.get("device", "cpu"))
        is_seg = arch.endswith("-seg")

        yaml_path = os.path.join(cfg["dataset_folder"], "data.yaml")
        stop_file = os.path.join(out, "stop_requested")

        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

        nc = _read_nc_from_yaml(yaml_path)

        # ── Validate labels before wasting GPU time ────────────────────
        _validate_labels(cfg["dataset_folder"], nc, is_seg)

        # ── Hyperparams ────────────────────────────────────────────────
        lr0           = float(hp.get("learning_rate",           0.002))
        lrf           = float(hp.get("yolo_lrf",                0.01))
        optimizer     = str(hp.get("yolo_optimizer",            "AdamW"))
        momentum      = float(hp.get("yolo_momentum",           0.937))
        weight_decay  = float(hp.get("yolo_weight_decay",       0.0005))
        warmup_epochs = float(hp.get("yolo_warmup_epochs",      5.0))
        cos_lr        = bool(hp.get("yolo_cos_lr",              False))
        patience      = int(hp.get("early_stopping_patience",   40))
        workers       = int(hp.get("num_workers",               0))
        box           = float(hp.get("yolo_box",                7.5))
        cls           = float(hp.get("yolo_cls",                3.0))
        dfl           = float(hp.get("yolo_dfl",                1.5))
        dropout       = float(hp.get("yolo_dropout",            0.1))
        overlap_mask  = bool(hp.get("yolo_overlap_mask",        True))
        mask_ratio    = int(hp.get("yolo_mask_ratio",           4))
        hsv_h         = float(hp.get("yolo_hsv_h",              0.01))
        hsv_s         = float(hp.get("yolo_hsv_s",              0.30))
        hsv_v         = float(hp.get("yolo_hsv_v",              0.20))
        fliplr        = float(hp.get("yolo_fliplr",             0.5))
        flipud        = float(hp.get("yolo_flipud",             0.0))
        degrees       = float(hp.get("yolo_degrees",            5.0))
        translate     = float(hp.get("yolo_translate",          0.05))
        scale         = float(hp.get("yolo_scale",              0.25))
        mosaic        = float(hp.get("yolo_mosaic",             0.3))
        mixup         = float(hp.get("yolo_mixup",              0.0))
        copy_paste    = float(hp.get("yolo_copy_paste",         0.1))
        close_mosaic  = int(hp.get("yolo_close_mosaic",         20))
        amp           = bool(hp.get("yolo_amp",                 True))
        cache         = bool(hp.get("yolo_cache",               False))
        save_period   = int(hp.get("yolo_save_period",          -1))
        plots         = bool(hp.get("yolo_plots",               True))

        # Clamp optimizer to valid Ultralytics values
        if optimizer not in {"AdamW", "Adam", "SGD", "RMSProp", "auto"}:
            optimizer = "AdamW"

        # ── Load model ─────────────────────────────────────────────────
        from ultralytics import YOLO

        pretrained = str(cfg.get("pretrained_weights", "")).strip()
        if pretrained and os.path.isfile(pretrained):
            model = YOLO(pretrained)
            print(f"[INFO] Fine-tuning from: {pretrained}", flush=True)
        else:
            model = YOLO(f"{arch}.pt")
            print(f"[INFO] Loaded pretrained: {arch}.pt", flush=True)

        print(
            f"[INFO] Task: {'segment' if is_seg else 'detect'} | "
            f"arch: {arch} | nc: {nc} | epochs: {epochs} | "
            f"batch: {batch} | imgsz: {img_w} | device: {device}",
            flush=True,
        )

        # ── Per-epoch callback ─────────────────────────────────────────
        best_val_loss   = float("inf")
        best_epoch      = 1
        last_train_loss = 0.0

        def on_fit_epoch_end(trainer):
            nonlocal best_val_loss, best_epoch, last_train_loss

            epoch      = trainer.epoch + 1
            train_loss = float(trainer.loss) if hasattr(trainer, "loss") else 0.0
            metrics    = trainer.metrics or {}

            # Primary metrics
            box_loss   = float(metrics.get("val/box_loss",        0.0))
            seg_loss   = float(metrics.get("val/seg_loss",        0.0))
            map50_b    = float(metrics.get("metrics/mAP50(B)",    0.0))
            mask_map50 = float(metrics.get("metrics/mAP50(M)",    0.0))

            # Track the most meaningful validation loss
            if is_seg and seg_loss > 0:
                val_loss = seg_loss
            else:
                val_loss = box_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
            last_train_loss = train_loss

            pct = min(int(epoch / epochs * 90), 90)

            metric_line = (
                f"METRIC epoch={epoch}"
                f" train_loss={train_loss:.6f}"
                f" val_loss={val_loss:.6f}"
                f" train_iou=0.0"
                f" val_iou={map50_b:.4f}"
                f" map50={map50_b:.4f}"
            )
            if is_seg:
                metric_line += f" mask_map50={mask_map50:.4f}"

            print(metric_line, flush=True)
            print(f"PROGRESS {pct}", flush=True)

            # Graceful stop via sentinel file
            if os.path.isfile(stop_file):
                print("[INFO] Stop requested — saving checkpoint and exiting.", flush=True)
                _save_best_pt(out)
                _emit_final(best_val_loss, best_epoch, last_train_loss)
                sys.exit(0)

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # ── Build training kwargs ──────────────────────────────────────
        train_kwargs = dict(
            data          = yaml_path,
            epochs        = epochs,
            patience      = patience,
            batch         = batch,
            imgsz         = img_w,
            device        = device,
            workers       = workers,
            optimizer     = optimizer,
            lr0           = lr0,
            lrf           = lrf,
            momentum      = momentum,
            weight_decay  = weight_decay,
            warmup_epochs = warmup_epochs,
            cos_lr        = cos_lr,
            box           = box,
            cls           = cls,
            dfl           = dfl,
            hsv_h         = hsv_h,
            hsv_s         = hsv_s,
            hsv_v         = hsv_v,
            fliplr        = fliplr,
            flipud        = flipud,
            degrees       = degrees,
            translate     = translate,
            scale         = scale,
            mosaic        = mosaic,
            mixup         = mixup,
            copy_paste    = copy_paste,
            close_mosaic  = close_mosaic,
            amp           = amp,
            cache         = cache,
            save          = True,
            save_period   = save_period,
            plots         = plots,
            project       = out,
            name          = "run",
            exist_ok      = True,
            verbose       = False,
        )

        # Seg-only kwargs
        if is_seg:
            train_kwargs["overlap_mask"] = overlap_mask
            train_kwargs["mask_ratio"]   = mask_ratio
            train_kwargs["dropout"]      = dropout

        # ── Train ──────────────────────────────────────────────────────
        model.train(**train_kwargs)

        _save_best_pt(out)
        _emit_final(best_val_loss, best_epoch, last_train_loss)

    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
