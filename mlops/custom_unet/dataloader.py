import os
import random
import numpy as np
import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def joint_transform(image_np, mask_np):
    """
    Applies joint Albumentations transform (flip, shift, noise) on NumPy arrays.
    Args:
        image_np: RGB image as NumPy array (H, W, 3)
        mask_np:  mask as NumPy array (H, W)
    Returns:
        Transformed image and mask (NumPy arrays)
    """
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.3,
            border_mode=0  # constant padding
        ),
        # A.ISONoise(color_shift=(0.02, 0.04), intensity=(0.15, 0.4), p=0.7)
    ], additional_targets={'mask': 'mask'})

    augmented = aug(image=image_np, mask=mask_np)
    return augmented['image'], augmented['mask']


class CustomImageMaskDataset(Dataset):
    def __init__(self, root, split="train", transform=None, mask_transform=None, apply_joint_transform=True, load_mask=True,imgWidth=400,imgHeight=300):
        self.image_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.load_mask = load_mask

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if self.load_mask and not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.filenames = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.mask_transform = mask_transform
        self.apply_joint_transform = apply_joint_transform
        self.imgWidth=imgWidth
        self.imgHeight=imgHeight
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Load image and convert BGR to RGB
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.load_mask:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # mask =(mask / 255).astype('uint8')
            if self.apply_joint_transform:
                image, mask = joint_transform(image, mask)

            # Resize after augmentation
            image = cv2.resize(image, (self.imgWidth, self.imgHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.imgWidth, self.imgHeight), interpolation=cv2.INTER_NEAREST)
      
            if self.transform:
                image = self.transform(image)  # Must support NumPy
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(mask).long()

            return image, mask
        else:
            # No mask loading
            image = cv2.resize(image, (self.imgWidth, self.imgHeight), interpolation=cv2.INTER_AREA)

            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            return image


class AugmentedDataset(CustomImageMaskDataset):
    def __init__(self, *args, repeat_factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat_factor = repeat_factor

    def __len__(self):
        return len(self.filenames) * self.repeat_factor

    def __getitem__(self, idx):
        real_idx = idx % len(self.filenames)
        return super().__getitem__(real_idx)


def mask_transform_fn(mask_np):
    #mask_np = (mask_np > 0).astype(np.uint8)
    mask_tensor = torch.from_numpy(mask_np).long()
    # mask_tensor = (mask_tensor > 0).long()
    return mask_tensor

def get_dataloaders(root, batch_size=1, num_workers=None, imgwidth=400,imgHeight=300):
    if num_workers is None:
        num_workers = 1  # or os.cpu_count()

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = AugmentedDataset(
        root, "train",
        transform=image_transform,
        mask_transform=mask_transform_fn,
        repeat_factor=2,
        imgWidth=imgwidth,
        imgHeight=imgHeight
    )

    valid_dataset = CustomImageMaskDataset(
        root, "valid",
        transform=image_transform,
        mask_transform=mask_transform_fn,
        imgWidth=imgwidth,
        imgHeight=imgHeight
    )

    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames)), "Train and Valid sets overlap!"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    return train_loader, valid_loader
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames)), "Train and Valid sets overlap!"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    return train_loader, valid_loader


def get_test_dataloader(root, batch_size=1, num_workers=None,imgwidth=400,imgHeight=300):
    if num_workers is None:
        num_workers = os.cpu_count()

    test_dataset = CustomImageMaskDataset(
        root, "test",
        apply_joint_transform=False,
        transform=transforms.ToTensor(),  # or custom transform if needed
        mask_transform=None,
        load_mask=False,
        imgWidth=imgwidth,
        imgHeight=imgHeight
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Test size: {len(test_dataset)}")
    return test_loader
