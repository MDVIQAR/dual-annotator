"""
mlops/scripts/seg_model.py

PyTorch Lightning module for UNet-family segmentation models.
Uses segmentation_models_pytorch (smp) as the model backend.
Loss: smp.losses.DiceLoss (multi-class safe) + CrossEntropyLoss combined.
Metrics: IoU always tracked; Precision/Recall/F1 tracked if enabled via extra_metrics.
"""

import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
from torchmetrics.classification import (
    BinaryJaccardIndex, MulticlassJaccardIndex,
    BinaryPrecision, MulticlassPrecision,
    BinaryRecall, MulticlassRecall,
    BinaryF1Score, MulticlassF1Score,
)


class SegModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes,
                 lr=0.001, extra_metrics=None):
        super().__init__()
        self.save_hyperparameters()

        weights = None if encoder_weights == "none" else encoder_weights

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=out_classes,
        )

        _em = set(extra_metrics or [])

        if out_classes == 1:
            self.dice_loss = smp.losses.DiceLoss(mode="binary")
            self.ce_loss   = torch.nn.BCEWithLogitsLoss()
            self.train_iou = BinaryJaccardIndex()
            self.val_iou   = BinaryJaccardIndex()
            if "precision" in _em:
                self.train_precision = BinaryPrecision()
                self.val_precision   = BinaryPrecision()
            if "recall" in _em:
                self.train_recall = BinaryRecall()
                self.val_recall   = BinaryRecall()
            if "f1" in _em:
                self.train_f1 = BinaryF1Score()
                self.val_f1   = BinaryF1Score()
        else:
            self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
            self.ce_loss   = torch.nn.CrossEntropyLoss()
            self.train_iou = MulticlassJaccardIndex(num_classes=out_classes)
            self.val_iou   = MulticlassJaccardIndex(num_classes=out_classes)
            if "precision" in _em:
                self.train_precision = MulticlassPrecision(num_classes=out_classes, average="macro")
                self.val_precision   = MulticlassPrecision(num_classes=out_classes, average="macro")
            if "recall" in _em:
                self.train_recall = MulticlassRecall(num_classes=out_classes, average="macro")
                self.val_recall   = MulticlassRecall(num_classes=out_classes, average="macro")
            if "f1" in _em:
                self.train_f1 = MulticlassF1Score(num_classes=out_classes, average="macro")
                self.val_f1   = MulticlassF1Score(num_classes=out_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, logits, masks):
        if self.hparams.out_classes == 1:
            ce = self.ce_loss(logits, masks.float().unsqueeze(1))
        else:
            ce = self.ce_loss(logits, masks.long())
        dice = self.dice_loss(logits, masks)
        return 0.5 * dice + 0.5 * ce

    def _get_preds(self, logits):
        if self.hparams.out_classes == 1:
            return torch.sigmoid(logits).squeeze(1)
        return torch.softmax(logits, dim=1)

    def _per_image_iou(self, logits, masks) -> torch.Tensor:
        """IoU computed independently per image in the batch, then averaged."""
        preds = self._get_preds(logits)

        if self.hparams.out_classes == 1:
            pred_bin = (preds > 0.5).long()
            tgt      = masks.long()
            ious = []
            for i in range(pred_bin.shape[0]):
                p     = pred_bin[i].flatten()
                t     = tgt[i].flatten()
                inter = (p & t).sum().float()
                union = (p | t).sum().float()
                ious.append(inter / (union + 1e-6))
        else:
            pred_cls = preds.argmax(dim=1)
            tgt      = masks.long()
            n_cls    = self.hparams.out_classes
            ious = []
            for i in range(pred_cls.shape[0]):
                p        = pred_cls[i]
                t        = tgt[i]
                cls_ious = []
                for c in range(n_cls):
                    inter = ((p == c) & (t == c)).sum().float()
                    union = ((p == c) | (t == c)).sum().float()
                    if union > 0:
                        cls_ious.append(inter / (union + 1e-6))
                if cls_ious:
                    ious.append(torch.stack(cls_ious).mean())

        if not ious:
            return torch.zeros(1, device=logits.device).squeeze()
        return torch.stack(ious).mean()

    def _per_image_prf(self, logits, masks):
        """Per-image Precision, Recall, F1 averaged over the batch.
        Returns a 3-tuple (precision, recall, f1) as scalar tensors."""
        eps = 1e-6
        precisions, recalls, f1s = [], [], []

        if self.hparams.out_classes == 1:
            preds = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
            tgt   = masks.long()
            for i in range(preds.shape[0]):
                p  = preds[i].flatten()
                t  = tgt[i].flatten()
                tp = ((p == 1) & (t == 1)).sum().float()
                fp = ((p == 1) & (t == 0)).sum().float()
                fn = ((p == 0) & (t == 1)).sum().float()
                pr = tp / (tp + fp + eps)
                rc = tp / (tp + fn + eps)
                f1 = 2 * pr * rc / (pr + rc + eps)
                precisions.append(pr)
                recalls.append(rc)
                f1s.append(f1)
        else:
            pred_cls = torch.softmax(logits, dim=1).argmax(dim=1)
            tgt      = masks.long()
            n_cls    = self.hparams.out_classes
            for i in range(pred_cls.shape[0]):
                p = pred_cls[i]
                t = tgt[i]
                cls_p, cls_r, cls_f = [], [], []
                for c in range(n_cls):
                    tp = ((p == c) & (t == c)).sum().float()
                    fp = ((p == c) & (t != c)).sum().float()
                    fn = ((p != c) & (t == c)).sum().float()
                    if (tp + fp + fn) > 0:
                        pr = tp / (tp + fp + eps)
                        rc = tp / (tp + fn + eps)
                        f1 = 2 * pr * rc / (pr + rc + eps)
                        cls_p.append(pr)
                        cls_r.append(rc)
                        cls_f.append(f1)
                if cls_p:
                    precisions.append(torch.stack(cls_p).mean())
                    recalls.append(torch.stack(cls_r).mean())
                    f1s.append(torch.stack(cls_f).mean())

        z = torch.zeros(1, device=logits.device).squeeze()
        if not precisions:
            return z, z, z
        return (torch.stack(precisions).mean(),
                torch.stack(recalls).mean(),
                torch.stack(f1s).mean())

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits      = self(images)
        loss        = self._compute_loss(logits, masks)
        preds       = self._get_preds(logits)
        iou         = self.train_iou(preds, masks)
        per_img_iou = self._per_image_iou(logits, masks)
        self.log("train_loss",          loss,        on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_iou",           iou,         on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_per_image_iou", per_img_iou, on_step=False, on_epoch=True, prog_bar=False)

        _em = set(self.hparams.extra_metrics or [])
        if _em:
            pi_prec, pi_rec, pi_f1 = self._per_image_prf(logits, masks)
            if "precision" in _em:
                self.log("train_precision",           self.train_precision(preds, masks), on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_per_image_precision", pi_prec,                            on_step=False, on_epoch=True, prog_bar=False)
            if "recall" in _em:
                self.log("train_recall",              self.train_recall(preds, masks),    on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_per_image_recall",    pi_rec,                             on_step=False, on_epoch=True, prog_bar=False)
            if "f1" in _em:
                self.log("train_f1",                  self.train_f1(preds, masks),        on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_per_image_f1",        pi_f1,                              on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits      = self(images)
        loss        = self._compute_loss(logits, masks)
        preds       = self._get_preds(logits)
        iou         = self.val_iou(preds, masks)
        per_img_iou = self._per_image_iou(logits, masks)
        self.log("val_loss",          loss,        on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_iou",           iou,         on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_per_image_iou", per_img_iou, on_step=False, on_epoch=True, prog_bar=False)

        _em = set(self.hparams.extra_metrics or [])
        if _em:
            pi_prec, pi_rec, pi_f1 = self._per_image_prf(logits, masks)
            if "precision" in _em:
                self.log("val_precision",           self.val_precision(preds, masks), on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_per_image_precision", pi_prec,                          on_step=False, on_epoch=True, prog_bar=False)
            if "recall" in _em:
                self.log("val_recall",              self.val_recall(preds, masks),    on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_per_image_recall",    pi_rec,                           on_step=False, on_epoch=True, prog_bar=False)
            if "f1" in _em:
                self.log("val_f1",                  self.val_f1(preds, masks),        on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_per_image_f1",        pi_f1,                            on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
