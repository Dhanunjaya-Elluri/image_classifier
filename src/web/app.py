"""
Web application module for the image classifier service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from ..core.config import settings
from ..api.endpoints import router
from ..core.monitoring import MonitoringMiddleware


def create_app() -> FastAPI:
    """Factory pattern for creating FastAPI application"""
    app = FastAPI(
        title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add monitoring middleware
    app.middleware("http")(MonitoringMiddleware())

    # Add metrics endpoint BEFORE including API routes
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Include API routes
    app.include_router(router, prefix=settings.API_V1_STR)

    return app


app = create_app()
