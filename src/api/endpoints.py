from fastapi import APIRouter, File, UploadFile

from ..classifier.preprocessing import validate_image
from ..core.config import settings
from ..services.inference import get_classifier
from .schemas import ModelInfo, PredictionItem, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = await validate_image(contents)

    classifier = get_classifier()
    top_predictions = classifier.predict(image, settings.IMAGE_SIZE)

    # Convert the list of tuples to list of PredictionItem
    predictions = [
        PredictionItem(class_name=class_name, confidence=confidence)
        for class_name, confidence in top_predictions
    ]

    return PredictionResponse(predictions=predictions)


@router.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information including architecture and input/output shapes"""
    classifier = get_classifier()
    session = classifier.session

    # Get input and output details
    input_details = session.get_inputs()[0]
    output_details = session.get_outputs()[0]

    return ModelInfo(
        name="SqueezeNet 1.1",
        description="A lightweight convolutional neural network for image classification",
        input_shape=list(input_details.shape),
        output_shape=list(output_details.shape),
    )
