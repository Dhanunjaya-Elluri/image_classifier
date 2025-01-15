# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Install Python dependencies
COPY pyproject.toml .
RUN uv venv -p 3.12 --seed /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache-dir -e .

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml .

ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY streamlit_app.py .

# Create models directory and download model files
RUN mkdir -p models && \
    python scripts/download_model.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command (will be overridden by docker-compose)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
