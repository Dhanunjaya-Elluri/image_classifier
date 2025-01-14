from functools import lru_cache
from ..img_classifier.classifier import ImageClassifier
from ..core.config import settings


@lru_cache()
def get_classifier() -> ImageClassifier:
    """
    Creates or returns a cached instance of ImageClassifier.
    Uses lru_cache to ensure only one instance is created (singleton pattern).
    """
    return ImageClassifier(
        model_path=settings.MODEL_PATH, labels_path=settings.LABELS_PATH
    )
