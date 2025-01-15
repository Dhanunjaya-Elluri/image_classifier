"""API service layer"""

from typing import Optional

import requests

from ..api.schemas import ModelInfo, PredictionResponse
from ..core.config import settings
from ..core.exceptions import APIConnectionError


class APIService:
    def __init__(self):
        self.base_url = settings.BASE_URL
        self.api_v1_str = settings.API_V1_STR
        self.timeout = 5  # seconds

    async def predict(self, image_bytes: bytes) -> Optional[PredictionResponse]:
        """Make prediction API call"""
        try:
            files = {"file": image_bytes}
            response = requests.post(
                f"{self.base_url}{self.api_v1_str}/predict",
                files=files,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return PredictionResponse(**response.json())
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Failed to connect to prediction API: {str(e)}")

    async def get_model_info(self) -> Optional[ModelInfo]:
        """Get model information"""
        try:
            response = requests.get(
                f"{self.base_url}{self.api_v1_str}/model-info", timeout=self.timeout
            )
            response.raise_for_status()
            return ModelInfo(**response.json())
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Failed to fetch model info: {str(e)}")
