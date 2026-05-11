"""
mlops/scripts/dataloader_unet.py

Dataset and LightningDataModule for UNet training.
Expects dataset layout produced by Phase 2 DataPreparator.export_unet():
  {root}/train/images/, {root}/train/masks/
  {root}/val/images/,   {root}/val/masks/
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A


class UNetDataset(Dataset):
    def __init__(self, root: str, split: str, in_channels: int, img_w: int, img_h: int, augment: bool = False):
        self.root = root
        self.split = split
        self.in_channels = in_channels
        self.img_w = img_w
        self.img_h = img_h
        self.augment = augment
        
        self.images_dir = os.path.join(root, split, "images")
        self.masks_dir = os.path.join(root, split, "masks")
        
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            
        self.pairs = []
        valid_exts = {".png", ".jpg", ".jpeg"}
        
        for img_name in sorted(os.listdir(self.images_dir)):
            stem, ext = os.path.splitext(img_name)
            if ext.lower() in valid_exts:
                img_path = os.path.join(self.images_dir, img_name)
                # Masks usually end with .png
                mask_path = os.path.join(self.masks_dir, stem + ".png")
                # Fallback to whatever extension the image had if .png doesn't exist
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(self.masks_dir, img_name)
                    
                if os.path.exists(mask_path):
                    self.pairs.append((img_path, mask_path))
                    
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3, border_mode=0),
            ], additional_targets={"mask": "mask"})
        else:
            self.transform = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # Load image
        if self.in_channels == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and len(img.shape) == 2:
                img = img[..., None] # H, W, 1
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        if img is None:
            # Fallback
            img = torch.zeros((self.img_h, self.img_w, self.in_channels), dtype=torch.uint8).numpy()
            
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = torch.zeros((self.img_h, self.img_w), dtype=torch.uint8).numpy()
            
        # Resize
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        
        if self.in_channels == 1 and len(img.shape) == 2:
            img = img[..., None]
            
        # Augment
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
            
        # To tensor
        if self.in_channels == 1:
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        else:
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            
        # Normalize
        img_tensor = (img_tensor - 0.5) / 0.5
        
        mask_tensor = torch.from_numpy(mask).long()
        
        return img_tensor, mask_tensor


class UNetDataModule(pl.LightningDataModule):
    def __init__(self, dataset_folder: str, batch_size: int, image_width: int, image_height: int, in_channels: int, num_workers: int = 0):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.img_w = image_width
        self.img_h = image_height
        self.in_channels = in_channels
        self.num_workers = num_workers
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
            augment=True
        )
        self.val_ds = UNetDataset(
            root=self.dataset_folder,
            split="val",
            in_channels=self.in_channels,
            img_w=self.img_w,
            img_h=self.img_h,
            augment=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
