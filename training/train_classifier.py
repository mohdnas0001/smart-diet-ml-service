"""
Train EfficientNet-B4 food classifier.
Requires: requirements-training.txt, a labelled image dataset.

Usage:
    python training/train_classifier.py --data_dir ./datasets/food --epochs 50 --batch_size 32
"""
import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 food classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=380)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    import timm
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from training.dataset import FoodDataset
    from training.augmentation import get_train_transforms, get_val_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Dataset
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    train_dataset = FoodDataset(str(train_dir), transform=get_train_transforms(args.img_size))
    val_dataset = FoodDataset(str(val_dir), transform=get_val_transforms(args.img_size))
    num_classes = len(train_dataset.classes)
    print(f"Classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = timm.create_model("efficientnet_b4", pretrained=args.pretrained, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.wandb:
        import wandb
        wandb.init(project="smart-diet-classifier", config=vars(args))

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

        if args.wandb:
            wandb.log({"train_acc": train_acc, "val_acc": val_acc, "epoch": epoch+1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.output_dir}/classifier.pt")
            print(f"  → Saved best model (val_acc={val_acc:.2f}%)")

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
