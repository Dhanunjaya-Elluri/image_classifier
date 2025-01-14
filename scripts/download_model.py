import urllib.request
import os
from pathlib import Path


def download_files():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Download SqueezeNet model
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
    model_path = models_dir / "squeezenet1.1-7.onnx"

    if not model_path.exists():
        print("Downloading SqueezeNet model...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model saved to {model_path}")

    # Download ImageNet labels
    labels_url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    labels_path = models_dir / "imagenet_labels.txt"

    if not labels_path.exists():
        print("Downloading ImageNet labels...")
        urllib.request.urlretrieve(labels_url, labels_path)
        print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    download_files()
