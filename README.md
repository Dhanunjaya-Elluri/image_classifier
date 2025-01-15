# Image Classifier

## Table of Contents

- [Image Classifier](#image-classifier)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Project Structure](#project-structure)
  - [Project Setup](#project-setup)
    - [Local Setup](#local-setup)
    - [Using Docker](#using-docker)

## Description

This project is a simple image classifier built using FastAPI, Streamlit, and Prometheus. It allows users to upload an image and get a prediction of the image's class. It also provides a monitoring dashboard to view the request metrics.

## Project Structure

- `src/`: Source code for the project.
  - `api/`: FastAPI application module.
  - `core/`: Core configuration and settings.
  - `services/`: Service modules for the project.
  - `ui/`: Streamlit application module.
  - `utils/`: Utility functions for the project.
- `streamlit_app.py`: Main Streamlit app.
- `pyproject.toml`: Configuration file for the project.
- `Dockerfile`: Dockerfile for the project.
- `docker-compose.yml`: Docker compose file for the project.


## Project Setup

### Local Setup

1. For this project, use the `uv` package manager. So, first install the `uv` package manager by running the following command:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and install the dependencies:

   ```bash
   uv venv -p 3.12 --seed
   ```

   Activate the virtual environment:

   | Linux/MacOS                   | Windows                   |
   |-------------------------------|---------------------------|
   | `source .venv/bin/activate`   | `.venv\Scripts\activate`  |

   Install the dependencies:

   ```bash
   uv pip install -e .
   ```

3. Install and run Prometheus:

   | Linux/MacOS                   | Windows                      |
   |-------------------------------|------------------------------|
   | `brew install prometheus`     | `choco install prometheus`   |

   Start Prometheus:

   ```bash
   prometheus --config.file=prometheus.local.yml
   ```

4. Run the FastAPI server:

   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

5. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

6. Open the Streamlit app in your browser:

   ```bash
   http://localhost:8501
   ```

### Using Docker

1. Build and run the Docker image:

   ```bash
   docker compose up --build
   ```
