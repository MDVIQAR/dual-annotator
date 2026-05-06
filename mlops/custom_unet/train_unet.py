import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pytorch_lightning as pl
from dataloader import get_dataloaders
from model import SegModel
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint

def visualize_sample(image, mask, title, idx):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # (H, W, C)

        if image.shape[2] == 1:
            image = image.squeeze(axis=2)

        # Undo normalization
        image = image * 0.5 + 0.5
        image = np.clip(image, 0, 1)

    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().numpy()  # (H, W)

    plt.figure(figsize=(10, 5))
    plt.suptitle(f"{title} - Sample {idx}")

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="tab20")
    plt.axis("off")

    save_dir = "D:/SPOOKFISH/NestleS/trainimg"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}_sample_{idx}.png")

    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train loader and sample visualization")
    parser.add_argument('--root', type=str, default="X:/spookfish/NestleS/data", help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')
    parser.add_argument('--visualise_images', type=bool,default=False, help='View images after data loading')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run on (cpu or cuda)")
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b3', help='Encoder name')
    parser.add_argument('--arch', type=str, default='unet', help='Model architecture')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--width', type=int, default=320, help='Width of image to be resized')
    parser.add_argument('--height', type=int, default=240, help='Height of image to be resized')


    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(args.root, batch_size=args.batch_size,imgwidth=args.width,imgHeight=args.height)

    train_dataset = train_loader.dataset
    valid_dataset = val_loader.dataset
    

    img, msk = train_dataset[0]
    print(f"Image min: {img.min().item()}, max: {img.max().item()}")
    print(f"Mask min: {msk.min().item()}, max: {msk.max().item()}")

    if args.visualise_images:
       for idx in range(len(train_dataset)):
            image, mask = train_dataset[idx]
            visualize_sample(image, mask, "Train Sample", idx)
    

    model = SegModel(args.arch, args.encoder_name, in_channels=args.in_channels, out_classes=args.out_classes)

    # Move model to device before printing
    torch_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    print(f"Model is on device: {next(model.parameters()).device}")




    # Define checkpoint path
    checkpoint_path = "X:/multiclass/lightning_logs/version_28_1w3j/checkpoints/epoch=38-step=234l.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        # Option 1: Load PyTorch Lightning checkpoint
        model = SegModel.load_from_checkpoint(
            checkpoint_path,
            arch=args.arch,
            encoder_name=args.encoder_name,
            in_channels=args.in_channels,
            out_classes=args.out_classes
        )
        model = model.to(torch_device)
        print("Model loaded from checkpoint successfully!")
    else:
        print("No checkpoint found. Training from scratch.")


        # Setup model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=False
    )

    # Trainer configuration
    trainer_kwargs = {
        "max_epochs": args.epochs,
        "log_every_n_steps": 1,
        "devices": 1,
        "accelerator": args.device
    }

    trainer = pl.Trainer(**trainer_kwargs)

    # Train the model
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path if os.path.exists(checkpoint_path) else None)

    # trainer_kwargs = {
    #     "max_epochs": args.epochs,
    #     "log_every_n_steps": 1,
    #     "devices": 1,
    #     "accelerator": args.device
    # }

    # trainer = pl.Trainer(**trainer_kwargs)
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()


# v1,v2- 5 classes
#v1-old data 53 epochs
#v2-new data(new+old) 15 epochs 
#v3-lactalis - cam0 cold
#v4- lactalis - cam1 cold