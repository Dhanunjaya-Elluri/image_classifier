from fastapi import HTTPException, status
from PIL import Image
import io


async def validate_image(contents: bytes) -> Image.Image:
    """
    Validates and converts uploaded bytes to PIL Image.
    Raises HTTPException if image is invalid.
    """
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file"
        )
