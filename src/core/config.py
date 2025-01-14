from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Image Classification Service"

    # Model Settings
    MODEL_PATH: Path = (
        Path(__file__).parent.parent.parent / "models/squeezenet1.1-7.onnx"
    )
    LABELS_PATH: Path = (
        Path(__file__).parent.parent.parent / "models/imagenet_labels.txt"
    )

    # Image Settings
    IMAGE_SIZE: tuple[int, int] = (224, 224)

    # Monitoring Settings
    PROMETHEUS_URL: str = "http://localhost:9090"
    ALLOWED_ORIGINS: list[str] = ["*"]  # For CORS

    class Config:
        case_sensitive = True


settings = Settings()
