import io
import numpy as np
from PIL import Image
from app.utils.image_utils import fix_exif_rotation


def resize_image(image: Image.Image, size: tuple = (640, 640)) -> Image.Image:
    """Resize image to target size maintaining aspect ratio with padding."""
    image = image.convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    background = Image.new("RGB", size, (114, 114, 114))
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    background.paste(image, offset)
    return background


def normalize_image(image: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized float32 numpy array (C, H, W)."""
    arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)


def preprocess_for_detection(image: Image.Image) -> np.ndarray:
    """Preprocess image for YOLOv8 detection."""
    image = fix_exif_rotation(image)
    resized = resize_image(image, (640, 640))
    return normalize_image(resized)


def preprocess_for_classification(image: Image.Image) -> np.ndarray:
    """Preprocess image for EfficientNet-B4 classification."""
    image = fix_exif_rotation(image)
    resized = resize_image(image, (380, 380))
    return normalize_image(resized)
