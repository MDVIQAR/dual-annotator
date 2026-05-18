"""
mlops/scripts/seg_model.py

PyTorch Lightning module for UNet-family segmentation models.
Uses segmentation_models_pytorch (smp) as the model backend.
Loss: smp.losses.DiceLoss (multi-class safe) + CrossEntropyLoss combined.
Metrics: IoU (Jaccard Index) tracked per epoch for train and validation.
"""

import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex


class SegModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Convert string 'none' to actual None
        weights = None if encoder_weights == "none" else encoder_weights
        
        # Create model dynamically based on architecture string
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=out_classes,
        )
        
        # Determine losses based on binary vs multiclass
        if out_classes == 1:
            self.dice_loss = smp.losses.DiceLoss(mode="binary")
            self.ce_loss = torch.nn.BCEWithLogitsLoss()
            # Issue 2 fix: Add IoU metrics
            self.train_iou = BinaryJaccardIndex()
            self.val_iou = BinaryJaccardIndex()
        else:
            self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
            self.ce_loss = torch.nn.CrossEntropyLoss()
            # Issue 2 fix: Add IoU metrics
            self.train_iou = MulticlassJaccardIndex(num_classes=out_classes)
            self.val_iou = MulticlassJaccardIndex(num_classes=out_classes)

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, logits, masks):
        # BCEWithLogitsLoss expects float target for binary
        if self.hparams.out_classes == 1:
            ce = self.ce_loss(logits, masks.float().unsqueeze(1))
        else:
            ce = self.ce_loss(logits, masks.long())
            
        dice = self.dice_loss(logits, masks)
        return 0.5 * dice + 0.5 * ce

    def _get_preds(self, logits):
        """Convert raw logits to predictions suitable for metric calculation."""
        if self.hparams.out_classes == 1:
            return torch.sigmoid(logits).squeeze(1)  # (B,1,H,W) → (B,H,W)
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

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits      = self(images)
        loss        = self._compute_loss(logits, masks)
        iou         = self.train_iou(self._get_preds(logits), masks)
        per_img_iou = self._per_image_iou(logits, masks)
        self.log("train_loss",          loss,        on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_iou",           iou,         on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_per_image_iou", per_img_iou, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits      = self(images)
        loss        = self._compute_loss(logits, masks)
        iou         = self.val_iou(self._get_preds(logits), masks)
        per_img_iou = self._per_image_iou(logits, masks)
        self.log("val_loss",          loss,        on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_iou",           iou,         on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_per_image_iou", per_img_iou, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        # Issue 8 fix: Add ReduceLROnPlateau scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
