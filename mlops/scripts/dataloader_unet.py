"""
mlops/scripts/dataloader_unet.py

Dataset and LightningDataModule for UNet training.
Expects dataset layout produced by Phase 2 DataPreparator.export_unet():
  {root}/train/images/, {root}/train/masks/
  {root}/val/images/,   {root}/val/masks/

aug_config dict format (from training config / UI):
  {
    "horizontal_flip": {"enabled": true, "p": 0.5},
    "rotation":        {"enabled": true, "limit": 30, "p": 0.4},
    ...
  }
If aug_config is None → use the default HorizontalFlip + ShiftScaleRotate pipeline.
If aug_config is {}   → no augmentation.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A


# ---------------------------------------------------------------------------
# Dynamic augmentation pipeline builder
# ---------------------------------------------------------------------------

def _single_aug(key: str, vals: dict, img_w: int, img_h: int):
    """Build one albumentations transform from key + parameter dict. Returns None on failure."""
    try:
        p = float(vals.get("p", 0.5))
        if key == "horizontal_flip":
            return A.HorizontalFlip(p=p)
        if key == "vertical_flip":
            return A.VerticalFlip(p=p)
        if key == "rotation":
            return A.Rotate(limit=int(vals.get("limit", 30)), p=p, border_mode=0)
        if key == "random_crop":
            pct = int(vals.get("crop_pct", 80))
            ch = max(32, min(int(img_h * pct / 100), img_h))
            cw = max(32, min(int(img_w * pct / 100), img_w))
            return A.RandomCrop(height=ch, width=cw, p=p)
        if key == "scale":
            return A.RandomScale(scale_limit=float(vals.get("scale_limit", 0.1)), p=p)
        if key == "shift":
            lim = float(vals.get("shift_limit", 0.05))
            return A.Affine(translate_percent={"x": (-lim, lim), "y": (-lim, lim)},
                            scale=1.0, rotate=0, p=p, border_mode=0)
        if key == "shear":
            lim = int(vals.get("shear_limit", 15))
            return A.Affine(shear=(-lim, lim), p=p)
        if key == "perspective":
            sc = max(0.01, float(vals.get("scale", 0.05)))
            return A.Perspective(scale=(0.01, sc), p=p)
        if key == "elastic":
            return A.ElasticTransform(
                alpha=float(vals.get("alpha", 1.0)),
                sigma=int(vals.get("sigma", 50)), p=p,
            )
        if key == "padding":
            pct = int(vals.get("pad_pct", 10))
            ph = max(1, int(img_h * pct / 100))
            pw = max(1, int(img_w * pct / 100))
            return A.PadIfNeeded(min_height=img_h + 2*ph, min_width=img_w + 2*pw,
                                 border_mode=4, p=p)
        if key == "grid_distortion":
            return A.GridDistortion(
                num_steps=int(vals.get("num_steps", 5)),
                distort_limit=float(vals.get("distort_limit", 0.3)), p=p,
            )
        if key == "brightness":
            return A.RandomBrightnessContrast(
                brightness_limit=float(vals.get("limit", 0.2)), contrast_limit=0.0, p=p)
        if key == "contrast":
            return A.RandomBrightnessContrast(
                brightness_limit=0.0, contrast_limit=float(vals.get("limit", 0.2)), p=p)
        if key == "saturation":
            return A.HueSaturationValue(
                hue_shift_limit=0, sat_shift_limit=int(vals.get("sat_limit", 30)),
                val_shift_limit=0, p=p)
        if key == "hue_shift":
            return A.HueSaturationValue(
                hue_shift_limit=int(vals.get("hue_limit", 20)),
                sat_shift_limit=0, val_shift_limit=0, p=p)
        if key == "gaussian_noise":
            v = float(vals.get("var_limit", 10.0))
            return A.GaussNoise(std_range=(max(0.001, v / 1000), v / 500), p=p)
        if key == "gaussian_blur":
            lim = int(vals.get("blur_limit", 7))
            lim = lim if lim % 2 == 1 else lim + 1
            return A.GaussianBlur(blur_limit=(3, lim), p=p)
        if key == "grayscale":
            return A.ToGray(p=p)
        if key == "color_jitter":
            return A.ColorJitter(
                brightness=float(vals.get("brightness", 0.2)),
                contrast=float(vals.get("contrast", 0.2)),
                saturation=float(vals.get("saturation", 0.2)),
                hue=0.0, p=p,
            )
        if key == "gamma":
            return A.RandomGamma(
                gamma_limit=(int(vals.get("gamma_low", 80)), int(vals.get("gamma_high", 120))),
                p=p,
            )
        if key == "channel_shuffle":
            return A.ChannelShuffle(p=p)
        if key == "cutout":
            n = max(1, int(vals.get("num_holes", 4)))
            sz = max(4, int(vals.get("max_size", 32)))
            return A.CoarseDropout(
                num_holes_range=(1, n),
                hole_height_range=(sz, sz),
                hole_width_range=(sz, sz), p=p)
        if key == "jpeg_compression":
            return A.ImageCompression(
                quality_range=(max(1, int(vals.get("quality_lower", 70))), 100), p=p)
    except Exception:
        pass
    return None


def _build_aug_transforms(aug_config, img_w: int, img_h: int):
    """
    Build an A.Compose pipeline from aug_config dict.
    aug_config=None  → default pipeline (flip + shift-scale-rotate).
    aug_config={}    → no augmentation (returns None).
    """
    if aug_config is None:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                     scale=(0.9, 1.1), rotate=(-15, 15), p=0.3, border_mode=0),
        ], additional_targets={"mask": "mask"})

    transforms = []
    needs_resize = False
    for key, vals in aug_config.items():
        if not vals.get("enabled", False):
            continue
        t = _single_aug(key, vals, img_w, img_h)
        if t is not None:
            transforms.append(t)
        if key in ("random_crop", "padding"):
            needs_resize = True

    if needs_resize:
        transforms.append(A.Resize(img_h, img_w))

    if not transforms:
        return None
    return A.Compose(transforms, additional_targets={"mask": "mask"})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UNetDataset(Dataset):
    def __init__(self, root: str, split: str, in_channels: int, img_w: int, img_h: int,
                 augment: bool = False, repeat_factor: int = 1, aug_config=None):
        self.root = root
        self.split = split
        self.in_channels = in_channels
        self.img_w = img_w
        self.img_h = img_h
        self.repeat_factor = max(1, repeat_factor)

        self.images_dir = os.path.join(root, split, "images")
        self.masks_dir = os.path.join(root, split, "masks")

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.pairs = []
        valid_exts = {".png", ".jpg", ".jpeg"}
        for img_name in sorted(os.listdir(self.images_dir)):
            stem, ext = os.path.splitext(img_name)
            if ext.lower() in valid_exts:
                img_path = os.path.join(self.images_dir, img_name)
                mask_path = os.path.join(self.masks_dir, stem + ".png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(self.masks_dir, img_name)
                if os.path.exists(mask_path):
                    self.pairs.append((img_path, mask_path))

        if len(self.pairs) == 0:
            raise ValueError(
                f"[CRITICAL] No matching Image/Mask pairs found in "
                f"'{self.images_dir}' and '{self.masks_dir}'. "
                f"Check that filenames match (stem-based matching)."
            )

        # aug_config takes precedence over the legacy augment flag
        if aug_config is not None:
            self.transform = _build_aug_transforms(aug_config, img_w, img_h)
        elif augment:
            self.transform = _build_aug_transforms(None, img_w, img_h)  # default pipeline
        else:
            self.transform = None

    def __len__(self):
        return len(self.pairs) * self.repeat_factor

    def __getitem__(self, idx):
        real_idx = idx % len(self.pairs)
        img_path, mask_path = self.pairs[real_idx]

        if self.in_channels == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to load image: {img_path}")
            img = img[..., None]
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")

        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        if self.in_channels == 1 and len(img.shape) == 2:
            img = img[..., None]

        if self.transform is not None:
            try:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
                # Ensure dimensions match after augmentation (e.g. crop + resize)
                if img.shape[0] != self.img_h or img.shape[1] != self.img_w:
                    img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, (self.img_w, self.img_h),
                                      interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass  # Fall back to unaugmented image silently

        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, dataset_folder: str, batch_size: int, image_width: int, image_height: int,
                 in_channels: int, num_workers: int = 0, repeat_factor: int = 2,
                 aug_config=None):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.img_w = image_width
        self.img_h = image_height
        self.in_channels = in_channels
        self.num_workers = num_workers
        self.repeat_factor = repeat_factor
        self.aug_config = aug_config  # None = default, {} = none, dict = custom
        self.pin_memory = torch.cuda.is_available()

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        self.train_ds = UNetDataset(
            root=self.dataset_folder,
            split="train",
            in_channels=self.in_channels,
            img_w=self.img_w,
            img_h=self.img_h,
            aug_config=self.aug_config,  # None = use default pipeline
            repeat_factor=self.repeat_factor,
        )
        self.val_ds = UNetDataset(
            root=self.dataset_folder,
            split="val",
            in_channels=self.in_channels,
            img_w=self.img_w,
            img_h=self.img_h,
            aug_config={},  # no augmentation for validation
            repeat_factor=1,
        )

        train_stems = {os.path.splitext(os.path.basename(p))[0] for p, _ in self.train_ds.pairs}
        val_stems   = {os.path.splitext(os.path.basename(p))[0] for p, _ in self.val_ds.pairs}
        overlap = train_stems & val_stems
        if overlap:
            raise ValueError(
                f"Train/Val overlap detected ({len(overlap)} files): {list(overlap)[:5]}"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
