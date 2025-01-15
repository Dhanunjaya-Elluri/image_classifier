"""Main FastAPI application module"""

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..core.config import settings
from ..core.middleware import MonitoringMiddleware
from .endpoints import health_check, router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )

    # Add monitoring middleware
    app.add_middleware(MonitoringMiddleware)

    # Add health check at root level
    app.get("/health")(health_check)

    # Add metrics endpoint at root level for Prometheus
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Include API routes under version prefix
    app.include_router(router, prefix=settings.API_V1_STR)

    return app


app = create_app()
