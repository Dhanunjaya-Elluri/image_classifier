"""Preprocessing utilities"""

import io

import numpy as np
from fastapi import HTTPException, status
from PIL import Image


def preprocess_image(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Preprocess image for classification

    Args:
        image (PIL.Image.Image): The image to preprocess
        size (tuple[int, int]): The size to resize the image to

    Returns:
        np.ndarray: The preprocessed image
    """
    # Resize and preprocess image
    image = image.resize(size)
    image_array = np.array(image)
    # Normalize to [0,1] and convert to NCHW format
    image_array = image_array.transpose(2, 0, 1)
    image_array = image_array / 255.0
    image_array = image_array.astype(np.float32)
    return np.expand_dims(image_array, axis=0)


async def validate_image(contents: bytes) -> Image.Image:
    """Validates and converts uploaded bytes to PIL Image

    Args:
        contents (bytes): The image bytes to validate

    Returns:
        PIL.Image.Image: The validated image

    Raises:
        HTTPException: If the image is invalid
    """
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file"
        ) from e
