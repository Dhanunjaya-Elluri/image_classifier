from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from loguru import logger
from PIL import Image
from scipy.special import softmax

from ..core.exceptions import ModelError
from .preprocessing import preprocess_image


class ImageClassifier:
    def __init__(self, model_path: Path, labels_path: Path):
        try:
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self.labels = self._load_labels(labels_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            raise ModelError(f"Failed to initialize model: {str(e)}")

    def _load_labels(self, labels_path: Path) -> list[str]:
        try:
            with open(labels_path) as f:
                labels = [line.strip() for line in f.readlines()]
            return labels
        except Exception as e:
            raise ModelError(f"Failed to load labels: {str(e)}")

    def predict(
        self, image: Image.Image, size: tuple[int, int]
    ) -> List[tuple[str, float]]:
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
                )  # Return raw probabilities
                for idx in top_indices
            ]
        except Exception as e:
            raise ModelError(f"Prediction failed: {str(e)}")
