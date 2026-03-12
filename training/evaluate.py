"""
Evaluate trained models.

Usage:
    python training/evaluate.py --model_path ./models/classifier.pt --data_dir ./datasets/food/test
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate food classifier")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=274)
    parser.add_argument("--img_size", type=int, default=380)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import timm
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, top_k_accuracy_score
    import numpy as np
    from training.dataset import FoodDataset
    from training.augmentation import get_val_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    dataset = FoodDataset(args.data_dir, transform=get_val_transforms(args.img_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_labels = []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = all_probs.argmax(axis=1)

    top1_acc = (preds == all_labels).mean()
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5)
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(classification_report(all_labels, preds, target_names=dataset.classes))


if __name__ == "__main__":
    main()
