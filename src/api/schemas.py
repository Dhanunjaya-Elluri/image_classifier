from typing import List

from pydantic import BaseModel


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
