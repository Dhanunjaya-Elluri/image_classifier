from pydantic import BaseModel
from typing import List


class PredictionItem(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]


class ModelInfo(BaseModel):
    name: str
    description: str
    input_shape: List[int]
    output_shape: List[int]
