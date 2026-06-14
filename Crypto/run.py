"""
Market Predictor — master orchestration entrypoint.

Responsibilities (in order):
  1. Load settings & configure logging.
  2. Initialise the async DB engine and create the raw schema tables
     (Base.metadata.create_all) if they do not already exist.
  3. Start an APScheduler AsyncIOScheduler with the platform's periodic jobs
     (crypto fetch, stock fetch) on the same event loop as the web server.
  4. Launch the Uvicorn FastAPI server on API_HOST:API_PORT (0.0.0.0:8000 by
     default), serving until shutdown, then stop the scheduler and dispose the
     engine cleanly.

Run:
    python run.py

Note on schema creation: create_all() builds plain PostgreSQL tables. The
composite (symbol/coin_id, timestamp) primary keys already satisfy
TimescaleDB's hypertable constraint, but converting the tables into
hypertables (SELECT create_hypertable(...)) is left to an explicit migration
step and is NOT performed here.
"""

from __future__ import annotations

import asyncio
import logging

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from backend.api.main import app
from backend.config.settings import configure_logging, get_settings
from backend.database.models import Base
from backend.database.session import dispose_engine, get_engine, init_engine

logger = logging.getLogger(__name__)


async def create_schema() -> None:
    """Create all ORM-defined tables if they do not already exist."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema ensured (create_all complete).")


def build_scheduler() -> AsyncIOScheduler:
    """
    Construct the AsyncIOScheduler and register the platform's periodic jobs.

    Jobs call the fetchers' convenience functions and persist via the
    repository. They are wrapped so a failure in one cycle is logged and does
    not kill the scheduler.
    """
    settings = get_settings()
    scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": settings.scheduler.misfire_grace_time,
        }
    )

    async def crypto_fetch_job() -> None:
        try:
            from backend.data_ingestion.crypto_fetcher import fetch_watchlist_market_data
            from backend.database.repository import upsert_crypto_market_data

            result = await fetch_watchlist_market_data()
            count = await upsert_crypto_market_data(result)
            logger.info("crypto_fetch_job: upserted %d crypto snapshots", count)
        except Exception:
            logger.exception("crypto_fetch_job failed for this cycle.")

    async def stock_fetch_job() -> None:
        try:
            from backend.data_ingestion.stock_fetcher import fetch_watchlist_daily
            from backend.database.repository import upsert_stock_batch_daily

            results = await fetch_watchlist_daily()
            counts = await upsert_stock_batch_daily(results, interval="daily")
            logger.info("stock_fetch_job: upserted %s", counts)
        except Exception:
            logger.exception("stock_fetch_job failed for this cycle.")

    scheduler.add_job(
        crypto_fetch_job,
        trigger="interval",
        seconds=settings.scheduler.crypto_fetch_interval_seconds,
        id="crypto_fetch",
        next_run_time=None,
    )
    scheduler.add_job(
        stock_fetch_job,
        trigger="interval",
        seconds=settings.scheduler.stock_fetch_interval_seconds,
        id="stock_fetch",
        next_run_time=None,
    )
    logger.info(
        "Scheduler configured | crypto=%ds stock=%ds",
        settings.scheduler.crypto_fetch_interval_seconds,
        settings.scheduler.stock_fetch_interval_seconds,
    )
    return scheduler


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "Booting %s v%s (env=%s)",
        settings.app_name, settings.app_version, settings.environment.value,
    )

    # 1) DB engine + schema
    init_engine()
    await create_schema()

    # 2) Scheduler on the current event loop
    scheduler = build_scheduler()
    scheduler.start()

    # 3) Uvicorn server on the same loop
    config = uvicorn.Config(
        app=app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.log_level.value.lower(),
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    try:
        logger.info("Starting Uvicorn on %s:%d", settings.api.host, settings.api.port)
        await server.serve()
    finally:
        logger.info("Shutting down: stopping scheduler and disposing DB engine.")
        scheduler.shutdown(wait=False)
        await dispose_engine()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Interrupted — exiting.")
