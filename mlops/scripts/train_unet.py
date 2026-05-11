"""
mlops/scripts/train_unet.py

Entrypoint for UNet segmentation training.
Launched by TrainWorker as a subprocess.
Reads config from --config JSON, saves weights to --out folder.

Stdout protocol:
  METRIC epoch=N train_loss=X val_loss=Y       (after each epoch)
  PROGRESS N                                    (0-100)
  METRICS_FINAL best_val_loss=X best_epoch=N final_train_loss=Y
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
from pytorch_lightning.callbacks import ModelCheckpoint

from seg_model import SegModel
from dataloader_unet import UNetDataModule


class MetricCallback(pl.Callback):
    def __init__(self, total_epochs):
        self._total = total_epochs
        self._best_val = float("inf")
        self._best_epoch = 1
        self._last_train = 0.0

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self._last_train = float(metrics.get("train_loss", self._last_train))

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch    = trainer.current_epoch + 1
        metrics  = trainer.callback_metrics
        val_loss = float(metrics.get("val_loss", 0.0))

        if val_loss < self._best_val:
            self._best_val   = val_loss
            self._best_epoch = epoch

        pct = int(epoch / self._total * 90)
        print(f"METRIC epoch={epoch} train_loss={self._last_train:.6f} val_loss={val_loss:.6f}", flush=True)
        print(f"PROGRESS {min(pct, 90)}", flush=True)


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

        dm = UNetDataModule(
            dataset_folder = cfg["dataset_folder"],
            batch_size     = hp["batch_size"],
            image_width    = hp["image_width"],
            image_height   = hp["image_height"],
            in_channels    = hp["in_channels"],
            num_workers    = 0,
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

        metric_cb = MetricCallback(total_epochs=hp["epochs"])
        
        ckpt_cb = ModelCheckpoint(
            dirpath    = out,
            filename   = "best",
            monitor    = "val_loss",
            mode       = "min",
            save_top_k = 1,
            save_last  = True,
        )

        device      = hp.get("device", "cpu")
        accelerator = "gpu" if device.startswith("cuda") else "cpu"

        trainer = pl.Trainer(
            max_epochs          = hp["epochs"],
            callbacks           = [ckpt_cb, metric_cb],
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
            f"final_train_loss={metric_cb._last_train:.6f}",
            flush=True,
        )
        print("PROGRESS 100", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
