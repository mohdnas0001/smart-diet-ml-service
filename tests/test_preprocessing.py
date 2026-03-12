import io
import numpy as np
from PIL import Image
from app.services.preprocessing import resize_image, normalize_image, preprocess_for_detection


def make_test_image(w=200, h=150):
    return Image.new("RGB", (w, h), color=(100, 150, 200))


def test_resize_image_output_shape():
    img = make_test_image(200, 150)
    result = resize_image(img, (640, 640))
    assert result.size == (640, 640)


def test_resize_image_mode():
    img = make_test_image()
    result = resize_image(img, (100, 100))
    assert result.mode == "RGB"


def test_normalize_image_shape():
    img = make_test_image(64, 64)
    arr = normalize_image(img)
    assert arr.shape == (3, 64, 64)


def test_normalize_image_dtype():
    img = make_test_image(32, 32)
    arr = normalize_image(img)
    assert arr.dtype == np.float32


def test_preprocess_for_detection_shape():
    img = make_test_image(300, 200)
    arr = preprocess_for_detection(img)
    assert arr.shape == (3, 640, 640)
