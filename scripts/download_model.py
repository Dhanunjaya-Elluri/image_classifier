"""Script to download model and label files"""

import urllib.request
from pathlib import Path
from urllib.parse import urlparse

SQUEEZENET_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
SQUEEZENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)


def is_safe_url(url: str) -> bool:
    """Validate URL scheme and domain"""
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.netloc in {
            "github.com",
            "raw.githubusercontent.com",
        }
    except Exception:
        return False


def download_file(url: str, path: Path) -> None:
    """Safely download a file from a validated URL"""
    if not is_safe_url(url):
        raise ValueError(f"Unsafe or invalid URL: {url}")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        req = urllib.request.Request(url, headers=headers)
        with (
            urllib.request.urlopen(req) as response,  # nosec B310
            open(path, "wb") as out_file,
        ):
            out_file.write(response.read())
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {str(e)}")


def download_files():
    """Download model and label files"""
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Download SqueezeNet model
    model_path = models_dir / "squeezenet1.1-7.onnx"

    if not model_path.exists():
        print("Downloading SqueezeNet model...")
        download_file(SQUEEZENET_MODEL_URL, model_path)
        print(f"Model saved to {model_path}")

    # Download ImageNet labels
    labels_path = models_dir / "imagenet_labels.txt"

    if not labels_path.exists():
        print("Downloading ImageNet labels...")
        download_file(SQUEEZENET_LABELS_URL, labels_path)
        print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    download_files()
