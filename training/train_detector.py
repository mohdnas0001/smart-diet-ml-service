"""
Train YOLOv8 food detector.
Requires: requirements-training.txt, a YOLO-format dataset.

Usage:
    python training/train_detector.py --data ./datasets/food.yaml --epochs 100
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 food detector")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--name", type=str, default="food_detector")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    if args.wandb:
        import wandb
        wandb.init(project="smart-diet-detector", config=vars(args))

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=args.output_dir,
        name=args.name,
        device="0" if __import__("torch").cuda.is_available() else "cpu",
    )
    print(f"Training complete. Results: {results}")


if __name__ == "__main__":
    main()
