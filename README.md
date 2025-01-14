# Image Classifier

## Description

This project is a simple image classifier built using FastAPI and Streamlit. It allows users to upload an image and get a prediction of the image's class.

## Project Structure

- `src/`: Source code for the project.
  - `core/`: Core configuration and settings.
  - `utils/`: Utility functions for the project.
  - `web/`: Web application module.
- `streamlit_app.py`: Main Streamlit app.
- `pyproject.toml`: Configuration file for the project.

## Project Setup

1. For this project, use the `uv` package manager. Install the `uv` package manager by running the following command:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install the dependencies:

```bash
$ uv venv -p 3.12 --seed
$ uv pip install -e .
```

3. Run the FastAPI server:

```bash
$ uvicorn src.api.main:app --reload --port 8000
```

4. Run the Streamlit app:

```bash
$ streamlit run streamlit_app.py
```

## Monitoring Setup

1. Install Prometheus:
   ```bash
   # For macOS
   brew install prometheus

   # For Linux
   wget https://github.com/prometheus/prometheus/releases/download/v2.49.1/prometheus-2.49.1.linux-amd64.tar.gz
   tar xvfz prometheus-*.tar.gz
   cd prometheus-*
   ```

2. Start Prometheus with the configuration:
   ```bash
   prometheus --config.file=prometheus.yml
   ```

3. Start the FastAPI server:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```
