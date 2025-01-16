"""Shared test fixtures for all tests"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scripts.download_model import download_files
from src.api.main import app
from src.classifier.classifier import ImageClassifier
from src.core.config import settings


def pytest_configure():
    """Download model files if they don't exist"""
    download_files()


@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture(scope="session")
def classifier():
    """Create a classifier instance"""
    return ImageClassifier(settings.MODEL_PATH, settings.LABELS_PATH)


@pytest.fixture(scope="session")
def test_image():
    """Create a test image"""
    img = Image.new("RGB", (224, 224), color="red")
    return img


@pytest.fixture(scope="session")
def test_image_bytes(test_image):
    """Convert test image to bytes"""
    import io

    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()
