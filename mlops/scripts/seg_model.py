"""
mlops/scripts/seg_model.py

PyTorch Lightning module for UNet-family segmentation models.
Uses segmentation_models_pytorch (smp) as the model backend.
Loss: smp.losses.DiceLoss (multi-class safe) + CrossEntropyLoss combined.
"""

import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp


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
        else:
            self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
            self.ce_loss = torch.nn.CrossEntropyLoss()

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

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self._compute_loss(logits, masks)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self._compute_loss(logits, masks)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
