"""API service layer"""

from typing import Optional

import requests

from ..api.schemas import ModelInfo, PredictionResponse
from ..core.config import settings


class APIService:
    def __init__(self, base_url: str = f"http://localhost:8000{settings.API_V1_STR}"):
        self.base_url = base_url
        self.timeout = 5  # seconds

    async def predict(self, image_bytes: bytes) -> Optional[PredictionResponse]:
        """Make prediction API call"""
        try:
            files = {"file": image_bytes}
            response = requests.post(
                f"{self.base_url}/predict", files=files, timeout=self.timeout
            )
            response.raise_for_status()
            return PredictionResponse(**response.json())
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to API: {str(e)}")

    async def get_model_info(self) -> Optional[ModelInfo]:
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model-info", timeout=self.timeout)
            response.raise_for_status()
            return ModelInfo(**response.json())
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch model info: {str(e)}")

    async def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch metrics: {str(e)}")
