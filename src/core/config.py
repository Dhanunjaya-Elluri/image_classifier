from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
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

    BASE_URL: str = "http://localhost:8000"
    API_V1_STR: str = "/api/v1"

    # Monitoring Settings
    PROMETHEUS_URL: str = "http://localhost:9090"

    class Config:
        case_sensitive = True


settings = Settings()
