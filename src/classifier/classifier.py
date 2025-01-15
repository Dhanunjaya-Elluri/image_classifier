from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from loguru import logger
from PIL import Image
from scipy.special import softmax

from ..core.exceptions import ModelError
from ..utils.preprocessing import preprocess_image


class ImageClassifier:
    def __init__(self, model_path: Path, labels_path: Path):
        """ImageClassifier for image classification.

        Args:
            model_path: Path to the ONNX model file
            labels_path: Path to the labels file

        Attributes:
            session: ONNX Runtime session for inference
            labels: List of class labels
        """
        try:
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self.labels = self._load_labels(labels_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            raise ModelError(f"Failed to initialize model: {str(e)}")

    def _load_labels(self, labels_path: Path) -> list[str]:
        """Load the labels from the given file path.

        Args:
            labels_path: Path to the labels file

        Returns:
            List of class labels
        """
        try:
            with open(labels_path) as f:
                labels = [line.strip() for line in f.readlines()]
            return labels
        except Exception as e:
            raise ModelError(f"Failed to load labels: {str(e)}")

    def predict(
        self, image: Image.Image, size: tuple[int, int]
    ) -> List[tuple[str, float]]:
        """Predict the class of the given image.

        Args:
            image: PIL Image object
            size: Tuple of image width and height

        Returns:
            List of tuples containing class name and confidence
        """
        try:
            input_array = preprocess_image(image, size)
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            predictions = self.session.run([output_name], {input_name: input_array})[0]

            # Apply softmax to convert logits to probabilities
            probabilities = softmax(predictions[0])
            top_indices = np.argsort(probabilities)[-10:][::-1]

            return [
                (
                    self.labels[idx],
                    float(probabilities[idx]),
                )
                for idx in top_indices
            ]
        except Exception as e:
            raise ModelError(f"Prediction failed: {str(e)}")
