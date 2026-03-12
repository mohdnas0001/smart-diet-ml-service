"""
Export trained PyTorch model to ONNX / TFLite.

Usage:
    python training/export_model.py --model_path ./models/classifier.pt --format onnx
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to deployment format")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./models/classifier.onnx")
    parser.add_argument("--format", choices=["onnx", "tflite"], default="onnx")
    parser.add_argument("--num_classes", type=int, default=274)
    parser.add_argument("--img_size", type=int, default=380)
    return parser.parse_args()


def export_onnx(model, output_path, img_size):
    import torch
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, output_path,
        export_params=True, opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {output_path}")


def main():
    args = parse_args()
    import torch
    import timm

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    model.eval()

    if args.format == "onnx":
        export_onnx(model, args.output_path, args.img_size)
    else:
        print("TFLite export requires ONNX intermediate and onnx-tf conversion")


if __name__ == "__main__":
    main()
