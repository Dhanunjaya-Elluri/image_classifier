"""API endpoints for the image classification model"""

from fastapi import APIRouter, HTTPException, UploadFile

from ..core.config import settings
from ..services.inference import get_classifier
from ..utils.preprocessing import validate_image
from .schemas import HealthCheckResponse, ModelInfo, PredictionItem, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    """Predict endpoint"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    contents = await file.read()
    image = await validate_image(contents)

    classifier = get_classifier()
    top_predictions = classifier.predict(image, settings.IMAGE_SIZE)

    predictions = [
        PredictionItem(class_name=class_name, confidence=confidence)
        for class_name, confidence in top_predictions
    ]

    return PredictionResponse(predictions=predictions)


@router.get("/model-info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    """Get model information including architecture and input/output shapes"""
    classifier = get_classifier()
    session = classifier.session

    input_details = session.get_inputs()[0]
    output_details = session.get_outputs()[0]

    return ModelInfo(
        name="SqueezeNet 1.1",
        description="A lightweight CNN model for image classification, offering a smaller architecture with reduced computational requirements",
        input_shape=list(input_details.shape),
        output_shape=list(output_details.shape),
    )


@router.get("/health", response_model=HealthCheckResponse)
def health_check() -> HealthCheckResponse:
    """Health check endpoint"""
    return HealthCheckResponse(status="OK")
