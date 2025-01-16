"""Integration tests for monitoring"""

import pytest


@pytest.mark.integration
def test_metrics_endpoint(test_client):
    """Test metrics endpoint is accessible and returns Prometheus format"""
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "image_classifier_predictions_total" in response.text
    assert "image_classifier_prediction_seconds" in response.text


@pytest.mark.integration
def test_metrics_after_prediction(test_client, test_image_bytes):
    """Test metrics are updated after making predictions"""
    # Make initial prediction
    test_client.post(
        "/api/v1/predict",
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )

    # Get metrics
    response = test_client.get("/metrics")
    assert response.status_code == 200
    metrics_text = response.text

    # Verify prediction counters
    assert 'image_classifier_predictions_total{status="200"}' in metrics_text
    assert "image_classifier_prediction_seconds_count" in metrics_text
