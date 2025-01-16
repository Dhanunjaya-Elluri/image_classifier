"""Integration tests for API endpoints"""

import pytest

from src.api.schemas import PredictionResponse


@pytest.mark.integration
def test_health_check(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


@pytest.mark.integration
def test_model_info(test_client):
    """Test model info endpoint"""
    response = test_client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SqueezeNet 1.1"
    assert "input_shape" in data
    assert "output_shape" in data


@pytest.mark.integration
def test_predict_valid_image(test_client, test_image_bytes):
    """Test prediction with valid image"""
    response = test_client.post(
        "/api/v1/predict", files={"file": ("test.png", test_image_bytes, "image/png")}
    )
    assert response.status_code == 200

    # Validate response matches our schema
    prediction_response = PredictionResponse(**response.json())
    assert len(prediction_response.predictions) > 0

    # Validate first prediction
    top_prediction = prediction_response.predictions[0]
    assert isinstance(top_prediction.class_name, str)
    assert isinstance(top_prediction.confidence, float)
    assert 0 <= top_prediction.confidence <= 1


@pytest.mark.integration
def test_predict_invalid_image(test_client):
    """Test prediction with invalid image"""
    response = test_client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", b"invalid image data", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]


@pytest.mark.integration
def test_predict_no_file(test_client):
    """Test prediction without file"""
    response = test_client.post("/api/v1/predict")
    assert response.status_code == 422
    assert "detail" in response.json()
