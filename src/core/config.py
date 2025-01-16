"""Configuration settings for the image classification service"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    PROJECT_NAME: str = "Image Classification Service"

    MODEL_PATH: Path = (
        Path(__file__).parent.parent.parent / "models/squeezenet1.1-7.onnx"
    )
    LABELS_PATH: Path = (
        Path(__file__).parent.parent.parent / "models/imagenet_labels.txt"
    )

    IMAGE_SIZE: tuple[int, int] = (224, 224)

    API_V1_STR: str = "/api/v1"  # API version prefix
    BASE_URL: str = "http://localhost:8000"

    PROMETHEUS_URL: str = "http://localhost:9090"

    model_config = SettingsConfigDict(case_sensitive=True)


settings = Settings()
