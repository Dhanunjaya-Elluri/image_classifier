from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from loguru import logger
from scipy.special import softmax
from typing import List


class ImageClassifier:
    def __init__(self, model_path: Path, labels_path: Path):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.labels = self._load_labels(labels_path)
        logger.info(f"Model loaded from {model_path}")

    def _load_labels(self, labels_path: Path) -> list[str]:
        with open(labels_path) as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def preprocess_image(self, image: Image.Image, size: tuple[int, int]) -> np.ndarray:
        # Resize and preprocess image
        image = image.resize(size)
        image_array = np.array(image)
        # Normalize to [0,1] and convert to NCHW format
        image_array = image_array.transpose(2, 0, 1)
        image_array = image_array / 255.0
        image_array = image_array.astype(np.float32)
        return np.expand_dims(image_array, axis=0)

    def predict(
        self, image: Image.Image, size: tuple[int, int]
    ) -> List[tuple[str, float]]:
        input_array = self.preprocess_image(image, size)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        predictions = self.session.run([output_name], {input_name: input_array})[0]

        # Apply softmax to convert logits to probabilities
        probabilities = softmax(predictions[0])
        top_indices = np.argsort(probabilities)[-10:][::-1]

        return [
            (self.labels[idx], float(probabilities[idx]))  # Return raw probabilities
            for idx in top_indices
        ]
