"""Test classifier functionality"""

import numpy as np
import pytest

from src.core.exceptions import ModelError


def test_classifier_initialization(classifier):
    """Test classifier initialization"""
    assert classifier.session is not None
    assert len(classifier.labels) > 0


def test_classifier_predict(classifier, test_image):
    """Test prediction functionality"""
    predictions = classifier.predict(test_image, (224, 224))

    assert len(predictions) == 10  # Top 10 predictions
    for class_name, confidence in predictions:
        assert isinstance(class_name, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


def test_classifier_invalid_image(classifier):
    """Test prediction with invalid image"""
    invalid_image = np.zeros((224, 224))  # Wrong format

    with pytest.raises(ModelError):
        classifier.predict(invalid_image, (224, 224))
