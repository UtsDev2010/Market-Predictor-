"""
Async database engine and session management for the Market Predictor.

Provides:
  - A lazily-created module-level async engine built from settings.db.async_dsn
    (asyncpg driver) with a tuned connection pool.
  - An async_sessionmaker factory bound to that engine.
  - `get_session()` : an async context manager that safely opens a session,
    yields it, commits on success, rolls back on error, and always closes.
  - `get_db()`      : a FastAPI-style async dependency generator.
  - Lifecycle helpers (`init_engine`, `dispose_engine`, `check_connection`).

The engine is created once per process and reused. Call `dispose_engine()` on
application shutdown to release pooled connections cleanly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (lazily initialised)
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine() -> AsyncEngine:
    """
    Create (once) and return the global async engine.

    The engine is configured from the DatabaseSettings group:
      - async DSN via the asyncpg driver
      - pool sizing, overflow, timeout, and recycle from settings
      - pre-ping enabled to transparently recover stale connections
    Subsequent calls return the existing engine.
    """
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    db = get_settings().db

    _engine = create_async_engine(
        db.async_dsn,
        echo=db.echo_sql,
        pool_size=db.pool_size,
        max_overflow=db.max_overflow,
        pool_timeout=db.pool_timeout,
        pool_recycle=db.pool_recycle,
        pool_pre_ping=True,
        future=True,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    logger.info(
        "Async DB engine initialised | host=%s port=%d db=%s pool_size=%d",
        db.host, db.port, db.name, db.pool_size,
    )
    return _engine


def get_engine() -> AsyncEngine:
    """Return the global async engine, initialising it on first use."""
    if _engine is None:
        return init_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the global async_sessionmaker, initialising the engine if needed."""
    if _session_factory is None:
        init_engine()
    assert _session_factory is not None  # set by init_engine()
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager yielding a database session.

    Usage:
        async with get_session() as session:
            session.add(obj)

    On clean exit the transaction is committed; on any exception it is rolled
    back and the exception re-raised. The session is always closed.
    """
    factory = get_session_factory()
    session: AsyncSession = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        logger.exception("DB session rolled back due to an exception.")
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a session.

    Usage:
        @app.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with get_session() as session:
        yield session


async def check_connection() -> bool:
    """
    Lightweight health check: run `SELECT 1` against the database.
    Returns True on success, False on any failure.
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database connection check failed.")
        return False


async def dispose_engine() -> None:
    """
    Dispose the engine and release all pooled connections.
    Call this on application shutdown.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Async DB engine disposed.")
    _engine = None
    _session_factory = None
