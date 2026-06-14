"""
FastAPI application entry point for the Market Predictor.

Wires together:
  - Application settings & structured logging (from backend.config.settings)
  - Async database lifecycle (engine init on startup, dispose on shutdown)
  - CORS middleware
  - Health endpoints
  - The /predictions router (historical OHLCV + live inference)

Run (development):
    uvicorn backend.api.main:app --reload

Run (production, settings-driven host/port/workers handled by your process
manager or by main.py at the repo entry point):
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import predictions
from backend.config.settings import configure_logging, get_settings
from backend.database.session import check_connection, dispose_engine, init_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialise resources on startup, clean up on shutdown."""
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "Starting %s v%s (env=%s)",
        settings.app_name, settings.app_version, settings.environment.value,
    )

    # Initialise the DB engine eagerly so startup fails fast on misconfiguration.
    init_engine()
    healthy = await check_connection()
    if not healthy:
        logger.error("Database connection check FAILED during startup.")
    else:
        logger.info("Database connection check passed.")

    try:
        yield
    finally:
        await dispose_engine()
        logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.api.api_v1_prefix}/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(
        predictions.router,
        prefix=settings.api.api_v1_prefix,
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe — returns OK if the process is up."""
        return {"status": "ok", "app": settings.app_name, "version": settings.app_version}

    @app.get("/health/db", tags=["health"])
    async def health_db() -> dict[str, str]:
        """Readiness probe — verifies database connectivity."""
        ok = await check_connection()
        return {"status": "ok" if ok else "unavailable", "component": "database"}

    return app


app = create_app()
