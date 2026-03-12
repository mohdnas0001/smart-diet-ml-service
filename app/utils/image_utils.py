import io
from PIL import Image, ExifTags
import numpy as np


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL Image from raw bytes."""
    return Image.open(io.BytesIO(data))


def strip_exif(image: Image.Image) -> Image.Image:
    """Return a copy of the image without EXIF metadata."""
    data = list(image.getdata())
    clean = Image.new(image.mode, image.size)
    clean.putdata(data)
    return clean


def fix_exif_rotation(image: Image.Image) -> Image.Image:
    """Rotate image according to EXIF orientation tag."""
    try:
        exif = image._getexif()
        if exif is None:
            return image
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None or orientation_key not in exif:
            return image
        orientation = exif[orientation_key]
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            image = image.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return image


def validate_image_file(data: bytes, max_size_bytes: int = 10 * 1024 * 1024) -> None:
    """Raise ValueError if data is not a valid JPEG/PNG under max_size_bytes."""
    if len(data) > max_size_bytes:
        raise ValueError(f"File too large: {len(data)} bytes (max {max_size_bytes})")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ("JPEG", "PNG"):
            raise ValueError(f"Unsupported image format: {img.format}")
    except Exception as exc:
        raise ValueError(f"Invalid image: {exc}") from exc
