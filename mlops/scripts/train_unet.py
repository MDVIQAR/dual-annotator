"""
mlops/scripts/train_unet.py

Entrypoint for UNet segmentation training.
Launched by TrainWorker as a subprocess.
Reads config from --config JSON, saves weights to --out folder.

Stdout protocol:
  METRIC epoch=N train_loss=X val_loss=Y train_iou=A val_iou=B  (after each epoch)
  PROGRESS N                                                     (0-100)
  METRICS_FINAL best_val_loss=X best_epoch=N final_train_loss=Y best_val_iou=Z
"""

import sys
import os

# Allow importing sibling scripts and mlops package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from seg_model import SegModel
from dataloader_unet import UNetDataModule


class MetricCallback(pl.Callback):
    def __init__(self, total_epochs):
        self._total              = total_epochs
        self._best_val           = float("inf")
        self._best_epoch         = 1
        self._last_train_loss    = 0.0
        self._last_train_iou     = 0.0
        self._last_train_per_iou = 0.0
        self._best_val_iou       = 0.0

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self._last_train_loss    = float(m.get("train_loss",          self._last_train_loss))
        self._last_train_iou     = float(m.get("train_iou",           self._last_train_iou))
        self._last_train_per_iou = float(m.get("train_per_image_iou", self._last_train_per_iou))

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch       = trainer.current_epoch + 1
        m           = trainer.callback_metrics
        val_loss    = float(m.get("val_loss",          0.0))
        val_iou     = float(m.get("val_iou",           0.0))
        val_per_iou = float(m.get("val_per_image_iou", 0.0))

        if val_loss < self._best_val:
            self._best_val     = val_loss
            self._best_epoch   = epoch
            self._best_val_iou = val_iou

        pct = int(epoch / self._total * 90)
        print(
            f"METRIC epoch={epoch} "
            f"train_loss={self._last_train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"train_iou={self._last_train_iou:.4f} "
            f"val_iou={val_iou:.4f} "
            f"train_per_image_iou={self._last_train_per_iou:.4f} "
            f"val_per_image_iou={val_per_iou:.4f}",
            flush=True,
        )
        print(f"PROGRESS {min(pct, 90)}", flush=True)


class GracefulStopCallback(pl.Callback):
    """Checks for a stop sentinel file at the end of each validation epoch.
    If found, signals Lightning to stop after the current epoch completes."""

    def __init__(self, stop_file: str):
        self._stop_file = stop_file

    def on_validation_epoch_end(self, trainer, pl_module):
        if os.path.isfile(self._stop_file):
            print("[INFO] Stop requested — finishing after this epoch.", flush=True)
            trainer.should_stop = True


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--out", required=True)
        args = parser.parse_args()

        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        hp  = cfg["hyperparams"]
        out = args.out

        aug_config = cfg.get("augmentations") or None  # None → default pipeline
        dm = UNetDataModule(
            dataset_folder = cfg["dataset_folder"],
            batch_size     = hp["batch_size"],
            image_width    = hp["image_width"],
            image_height   = hp["image_height"],
            in_channels    = hp["in_channels"],
            num_workers    = int(hp.get("num_workers", 0)),
            repeat_factor  = int(hp.get("repeat_factor", 2)),
            aug_config     = aug_config,
        )

        enc_weights = hp["encoder_weights"] if hp["encoder_weights"] != "none" else None
        model = SegModel(
            arch            = hp["architecture"],
            encoder_name    = hp["encoder"],
            encoder_weights = enc_weights,
            in_channels     = hp["in_channels"],
            out_classes     = hp["out_classes"],
            lr              = hp["learning_rate"],
        )

        # Fine-tune: load pre-trained weights if provided
        pretrained = cfg.get("pretrained_weights", "").strip()
        if pretrained and os.path.isfile(pretrained):
            print(f"[INFO] Loading pre-trained weights: {pretrained}", flush=True)
            state = torch.load(pretrained, map_location="cpu")
            # Support both raw state_dict and Lightning checkpoint formats
            if "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[INFO] Missing keys (expected for different out_classes): {len(missing)}", flush=True)
            print("[INFO] Pre-trained weights loaded — fine-tuning enabled.", flush=True)

        metric_cb = MetricCallback(total_epochs=hp["epochs"])

        stop_file = os.path.join(out, "stop_requested")
        graceful_stop_cb = GracefulStopCallback(stop_file)

        ckpt_cb = ModelCheckpoint(
            dirpath    = out,
            filename   = "best",
            monitor    = "val_loss",
            mode       = "min",
            save_top_k = 1,
            save_last  = True,
        )

        early_stop_cb = EarlyStopping(
            monitor  = "val_loss",
            patience = 15,
            mode     = "min",
            verbose  = False,
        )

        device      = hp.get("device", "cpu")
        accelerator = "gpu" if device.startswith("cuda") else "cpu"

        trainer = pl.Trainer(
            max_epochs          = hp["epochs"],
            callbacks           = [ckpt_cb, metric_cb, graceful_stop_cb, early_stop_cb],
            accelerator         = accelerator,
            devices             = 1,
            log_every_n_steps   = 1,
            enable_progress_bar = False,
            logger              = False,
        )

        trainer.fit(model, datamodule=dm)

        best_ckpt = ckpt_cb.best_model_path
        if best_ckpt and os.path.isfile(best_ckpt):
            best_model = SegModel.load_from_checkpoint(best_ckpt)
            torch.save(best_model.state_dict(), os.path.join(out, "best.pt"))

        print(
            f"METRICS_FINAL "
            f"best_val_loss={metric_cb._best_val:.6f} "
            f"best_epoch={metric_cb._best_epoch} "
            f"final_train_loss={metric_cb._last_train_loss:.6f} "
            f"best_val_iou={metric_cb._best_val_iou:.4f}",
            flush=True,
        )
        print("PROGRESS 100", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
