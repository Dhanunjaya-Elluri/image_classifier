"""Test preprocessing utilities"""

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from src.utils.preprocessing import preprocess_image, validate_image


@pytest.mark.unit
def test_preprocess_image(test_image):
    """Test image preprocessing"""
    processed = preprocess_image(test_image, (224, 224))
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 3, 224, 224)
    assert processed.dtype == np.float32
    assert np.all((processed >= 0) & (processed <= 1))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_image(test_image_bytes):
    """Test image validation"""
    image = await validate_image(test_image_bytes)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_invalid_image():
    """Test validation with invalid image"""
    with pytest.raises(HTTPException):
        await validate_image(b"invalid image data")
