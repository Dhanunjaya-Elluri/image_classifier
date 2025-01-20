# Image Classifier

## Table of Contents

- [Image Classifier](#image-classifier)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Project Structure](#project-structure)
  - [Project Setup](#project-setup)
    - [Local Setup](#local-setup)
    - [Using Docker](#using-docker)
  - [Testing](#testing)
  - [Code Quality](#code-quality)
  - [Monitoring with Prometheus](#monitoring-with-prometheus)

## Description

This project is a simple image classifier built using FastAPI, Streamlit, and Prometheus. It allows users to upload an image and get a prediction of the image's class. It also provides a monitoring dashboard to view the request metrics.

## Project Structure

- `scripts/`: Contains the scripts to download the model and labels files.
- `models/`: Contains the SqueezeNet1.1 model and labels files.
- `src/`: Source code for the project.
  - `api/`: Contains the main application with endpoints and schemas.
  - `classifier/`: Contains the classifier model to predict the image class.
  - `core/`: Core configuration, settings, exceptions and middleware.
  - `services/`: Service modules for API and Monitoring.
  - `ui/`: Streamlit pages for Classification, Model Info and Monitoring. Also, plot to show top 10 predictions.
  - `utils/`: Utility functions for the project.
- `tests/`: Contains the unit and integration tests.
- `streamlit_app.py`: Entry point for the Streamlit app.
- `pyproject.toml`: Configuration file for the project.
- `Dockerfile`: Dockerfile for the project.
- `docker-compose.yml`: Docker compose file for the project.
- `prometheus.local.yml`: Prometheus configuration file.

## Project Setup

### Local Setup

1. Clone this repository.

   ```bash
   git clone https://github.com/your-username/image-classifier.git
   ```

2. For this project, use the `uv` package manager. So, first install the `uv` package manager by running the following command:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create a virtual environment and install the dependencies:

   ```bash
   uv venv -p 3.12 --seed
   ```

   Activate the virtual environment:

   | Linux/MacOS                 | Windows                  |
   | --------------------------- | ------------------------ |
   | `source .venv/bin/activate` | `.venv\Scripts\activate` |

   Install the dependencies:

   ```bash
   uv pip install -e .
   ```

4. Download the model and labels files:

   ```bash
   python scripts/download_model.py
   ```

5. Install and run Prometheus:

   | Linux/MacOS               | Windows                    |
   | ------------------------- | -------------------------- |
   | `brew install prometheus` | `choco install prometheus` |

   Start Prometheus:

   ```bash
   prometheus --config.file=prometheus.local.yml
   ```

6. Run the FastAPI server:

   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

7. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

8. Open the Streamlit app in your browser:

   ```bash
   http://localhost:8501
   ```

### Using Docker

1. Build and run the Docker image:

   Make sure you have Docker installed and running.

   ```bash
   docker compose up --build
   ```

## Testing

First, install the dependencies:

```bash
uv pip install -e ".[test,lint]"
```

To run unit tests:

```bash
pytest -v -m unit
```

To run integration tests:

```bash
pytest -v -m integration
```

To run all tests:

```bash
pytest -v
```

## Code Quality

To check the code quality, use the pre-commit hooks.

First, install the pre-commit hooks:

```bash
uv pip install pre-commit
pre-commit install
```

Then, run the pre-commit hooks:

```bash
pre-commit run --all-files
```

## Monitoring with Prometheus

To view the Prometheus dashboard, open the following URL in your browser (make sure Prometheus is running):

```bash
http://localhost:9090
```

Click on the `Graph` tab and enter the following queries to view the request metrics:

1. For total number of predictions:

   ```bash
   image_classifier_predictions_total
   ```

2. For successful predictions:

   ```bash
   image_classifier_predictions_total{status="200"}
   ```

3. For error predictions:

   ```bash
   image_classifier_predictions_total{status!="200"}
   ```

4. For prediction latency (response time) histogram:

   ```bash
   rate(image_classifier_prediction_seconds_bucket[5m])
   ```

5. For average response time:

   ```bash
   rate(image_classifier_prediction_seconds_sum[5m]) / rate(image_classifier_prediction_seconds_count[5m])
   ```

6. For success rate percentage:

   ```bash
   sum(image_classifier_predictions_total{status="200"}) / sum(image_classifier_predictions_total) * 100
   ```
