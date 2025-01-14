from PIL import Image, ImageDraw, ImageFont
import numpy as np


def visualize_prediction(
    image: Image.Image, class_name: str, confidence: float
) -> Image.Image:
    # Create a copy of the image
    result = image.copy()
    draw = ImageDraw.Draw(result)

    # Add prediction text
    text = f"{class_name} ({confidence:.2%})"

    # Calculate text position and size
    img_width = result.width
    img_height = result.height
    padding = 10
    text_position = (padding, img_height - 40)

    # Add semi-transparent background strip
    strip_height = 50
    overlay = Image.new("RGBA", (img_width, strip_height), (255, 255, 255, 180))
    result.paste(overlay, (0, img_height - strip_height), overlay)

    # Draw text with shadow effect
    shadow_offset = 1
    # Draw shadow
    draw.text(
        (text_position[0] + shadow_offset, text_position[1] + shadow_offset),
        text,
        fill=(0, 0, 0, 128),
    )
    # Draw main text
    draw.text(text_position, text, fill=(0, 0, 0, 255))

    return result
