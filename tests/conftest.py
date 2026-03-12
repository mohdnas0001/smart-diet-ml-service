import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """Create a 100x100 red JPEG image in memory."""
    img = Image.new("RGB", (100, 100), color=(220, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
