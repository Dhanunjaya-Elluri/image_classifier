"""Middleware for monitoring requests"""

import time
from typing import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

REQUESTS_TOTAL = Counter(
    "image_classifier_predictions_total",
    "Total number of predictions",
    ["status"],
)

PREDICTION_LATENCY = Histogram(
    "image_classifier_prediction_seconds",
    "Time spent processing prediction",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring requests"""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not request.url.path.endswith("/predict"):
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        REQUESTS_TOTAL.labels(status=str(response.status_code)).inc()
        PREDICTION_LATENCY.observe(duration)

        return response
