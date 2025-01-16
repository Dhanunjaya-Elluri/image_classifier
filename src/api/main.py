"""Main FastAPI application module"""

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.endpoints import health_check, router
from src.core.config import settings
from src.core.middleware import MonitoringMiddleware


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )

    app.add_middleware(MonitoringMiddleware)

    app.get("/health")(health_check)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router, prefix=settings.API_V1_STR)

    return app


app = create_app()
