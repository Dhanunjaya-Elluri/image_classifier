"""Monitoring utilities for the application"""

from prometheus_client import Counter, Histogram
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Define Prometheus metrics
PREDICTION_REQUESTS = Counter(
    "image_classifier_predictions_total",
    "Total number of prediction requests",
    ["status"],
)

PREDICTION_LATENCY = Histogram(
    "image_classifier_prediction_seconds",
    "Time spent processing prediction requests",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor prediction requests and latency"""

    async def dispatch(self, request: Request, call_next):
        # Only monitor the prediction endpoint
        if not request.url.path.endswith("/predict"):
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Record metrics only for prediction requests
        PREDICTION_REQUESTS.labels(status=response.status_code).inc()

        PREDICTION_LATENCY.observe(duration)

        return response
