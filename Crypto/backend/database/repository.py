"""
Async persistence repository for the Market Predictor.

Implements high-performance batch upserts using PostgreSQL's native
`INSERT ... ON CONFLICT (...) DO UPDATE` (via SQLAlchemy's
`sqlalchemy.dialects.postgresql.insert`).

The repository accepts the typed `FetchResult` payloads produced by the fetcher
scripts:
  - crypto_fetcher.OHLCVCandle      -> crypto_market_data
  - crypto_fetcher.MarketData       -> crypto_market_snapshot
  - stock_fetcher.StockBar          -> stock_data
  - stock_fetcher.DailyAdjustedBar  -> stock_daily_adjusted

All write methods are idempotent: re-ingesting the same time range updates the
existing rows rather than failing on the composite primary key.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.data_ingestion import crypto_fetcher as cf
from backend.data_ingestion import stock_fetcher as sf
from backend.database.models import (
    CryptoMarketSnapshot,
    MarketData,
    StockData,
    StockDailyAdjusted,
)
from backend.database.session import get_session

logger = logging.getLogger(__name__)

# Default batch size for chunked inserts. PostgreSQL has a hard limit of 65535
# bind parameters per statement; chunking keeps us well under it regardless of
# column count.
DEFAULT_CHUNK_SIZE = 500


def _chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Split a list of row-dicts into chunks of at most `size` items."""
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [rows[i : i + size] for i in range(0, len(rows), size)]


async def _bulk_upsert(
    session: AsyncSession,
    model: type,
    rows: list[dict[str, Any]],
    conflict_cols: list[str],
    update_cols: list[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """
    Execute a chunked `INSERT ... ON CONFLICT DO UPDATE` for the given model.

    Args:
        session: An open AsyncSession (transaction managed by the caller).
        model: The ORM model class to insert into.
        rows: List of column->value dicts to upsert.
        conflict_cols: Columns forming the unique/primary key conflict target.
        update_cols: Columns to overwrite with the incoming values on conflict.
        chunk_size: Max rows per INSERT statement.

    Returns:
        The number of rows submitted for upsert.
    """
    if not rows:
        return 0

    total = 0
    for chunk in _chunked(rows, chunk_size):
        stmt = pg_insert(model).values(chunk)
        set_clause = {col: stmt.excluded[col] for col in update_cols}
        stmt = stmt.on_conflict_do_update(
            index_elements=conflict_cols,
            set_=set_clause,
        )
        await session.execute(stmt)
        total += len(chunk)

    logger.debug(
        "Upserted %d rows into %s (conflict on %s)",
        total, model.__tablename__, conflict_cols,
    )
    return total


# ---------------------------------------------------------------------------
# Row builders: typed fetcher dataclasses -> ORM column dicts
# ---------------------------------------------------------------------------

def _crypto_ohlcv_to_row(
    candle: cf.OHLCVCandle,
    coin_id: str,
    symbol: str,
    vs_currency: str,
) -> dict[str, Any]:
    return {
        "coin_id": coin_id,
        "timestamp": candle.timestamp,
        "symbol": symbol,
        "vs_currency": vs_currency,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": None,  # CoinGecko OHLC endpoint carries no per-candle volume
    }


def _crypto_market_to_row(md: cf.MarketData, vs_currency: str) -> dict[str, Any]:
    return {
        "coin_id": md.coin_id,
        "timestamp": md.last_updated,
        "symbol": md.symbol,
        "name": md.name,
        "vs_currency": vs_currency,
        "current_price": md.current_price,
        "market_cap": md.market_cap,
        "market_cap_rank": md.market_cap_rank,
        "total_volume": md.total_volume,
        "high_24h": md.high_24h,
        "low_24h": md.low_24h,
        "price_change_24h": md.price_change_24h,
        "price_change_pct_24h": md.price_change_pct_24h,
        "circulating_supply": md.circulating_supply,
        "total_supply": md.total_supply,
        "max_supply": md.max_supply,
    }


def _stock_bar_to_row(bar: sf.StockBar, symbol: str, interval: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "interval": interval,
        "timestamp": bar.timestamp,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


def _stock_adjusted_to_row(bar: sf.DailyAdjustedBar, symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timestamp": bar.timestamp,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "adjusted_close": bar.adjusted_close,
        "volume": bar.volume,
        "dividend_amount": bar.dividend_amount,
        "split_coefficient": bar.split_coefficient,
    }


# ---------------------------------------------------------------------------
# Public repository API
# ---------------------------------------------------------------------------

async def upsert_crypto_ohlcv(
    coin_id: str,
    result: cf.FetchResult,
    symbol: str | None = None,
    vs_currency: str = "usd",
    session: AsyncSession | None = None,
) -> int:
    """
    Upsert a crypto OHLCV FetchResult (list[OHLCVCandle]) into crypto_market_data.

    Args:
        coin_id: CoinGecko coin id the candles belong to.
        result: FetchResult whose `.data` is list[OHLCVCandle].
        symbol: Ticker symbol; falls back to coin_id when unknown.
        vs_currency: Quote currency.
        session: Optional externally-managed session. If omitted, a session is
            opened and committed internally.

    Returns:
        Number of rows upserted (0 if the FetchResult was unsuccessful/empty).
    """
    if not result.success or not result.data:
        logger.warning(
            "upsert_crypto_ohlcv: skipping coin_id=%s (success=%s, error=%s)",
            coin_id, result.success, result.error,
        )
        return 0

    candles: Sequence[cf.OHLCVCandle] = result.data
    rows = [
        _crypto_ohlcv_to_row(c, coin_id, symbol or coin_id, vs_currency)
        for c in candles
    ]
    update_cols = ["symbol", "vs_currency", "open", "high", "low", "close", "volume"]

    async def _run(s: AsyncSession) -> int:
        return await _bulk_upsert(
            s, MarketData, rows,
            conflict_cols=["coin_id", "timestamp"],
            update_cols=update_cols,
        )

    if session is not None:
        count = await _run(session)
    else:
        async with get_session() as s:
            count = await _run(s)
    logger.info("upsert_crypto_ohlcv: %d candles for coin_id=%s", count, coin_id)
    return count


async def upsert_crypto_market_data(
    result: cf.FetchResult,
    vs_currency: str = "usd",
    session: AsyncSession | None = None,
) -> int:
    """
    Upsert a crypto market FetchResult (list[MarketData]) into
    crypto_market_snapshot.

    Returns:
        Number of rows upserted.
    """
    if not result.success or not result.data:
        logger.warning(
            "upsert_crypto_market_data: skipping (success=%s, error=%s)",
            result.success, result.error,
        )
        return 0

    items: Sequence[cf.MarketData] = result.data
    rows = [_crypto_market_to_row(md, vs_currency) for md in items]
    update_cols = [
        "symbol", "name", "vs_currency", "current_price", "market_cap",
        "market_cap_rank", "total_volume", "high_24h", "low_24h",
        "price_change_24h", "price_change_pct_24h", "circulating_supply",
        "total_supply", "max_supply",
    ]

    async def _run(s: AsyncSession) -> int:
        return await _bulk_upsert(
            s, CryptoMarketSnapshot, rows,
            conflict_cols=["coin_id", "timestamp"],
            update_cols=update_cols,
        )

    if session is not None:
        count = await _run(session)
    else:
        async with get_session() as s:
            count = await _run(s)
    logger.info("upsert_crypto_market_data: %d snapshots upserted", count)
    return count


async def upsert_stock_bars(
    symbol: str,
    result: sf.FetchResult,
    interval: str = "daily",
    session: AsyncSession | None = None,
) -> int:
    """
    Upsert a stock OHLCV FetchResult (list[StockBar]) into stock_data.

    Args:
        symbol: Equity ticker.
        result: FetchResult whose `.data` is list[StockBar].
        interval: Series interval label ('1min', '5min', 'daily', 'weekly', ...).
            Part of the composite key so multiple intervals coexist.
        session: Optional externally-managed session.

    Returns:
        Number of rows upserted.
    """
    if not result.success or not result.data:
        logger.warning(
            "upsert_stock_bars: skipping symbol=%s interval=%s (success=%s, error=%s)",
            symbol, interval, result.success, result.error,
        )
        return 0

    bars: Sequence[sf.StockBar] = result.data
    rows = [_stock_bar_to_row(b, symbol, interval) for b in bars]
    update_cols = ["open", "high", "low", "close", "volume"]

    async def _run(s: AsyncSession) -> int:
        return await _bulk_upsert(
            s, StockData, rows,
            conflict_cols=["symbol", "interval", "timestamp"],
            update_cols=update_cols,
        )

    if session is not None:
        count = await _run(session)
    else:
        async with get_session() as s:
            count = await _run(s)
    logger.info(
        "upsert_stock_bars: %d bars for symbol=%s interval=%s", count, symbol, interval
    )
    return count


async def upsert_stock_daily_adjusted(
    symbol: str,
    result: sf.FetchResult,
    session: AsyncSession | None = None,
) -> int:
    """
    Upsert a daily-adjusted FetchResult (list[DailyAdjustedBar]) into
    stock_daily_adjusted.

    Returns:
        Number of rows upserted.
    """
    if not result.success or not result.data:
        logger.warning(
            "upsert_stock_daily_adjusted: skipping symbol=%s (success=%s, error=%s)",
            symbol, result.success, result.error,
        )
        return 0

    bars: Sequence[sf.DailyAdjustedBar] = result.data
    rows = [_stock_adjusted_to_row(b, symbol) for b in bars]
    update_cols = [
        "open", "high", "low", "close", "adjusted_close", "volume",
        "dividend_amount", "split_coefficient",
    ]

    async def _run(s: AsyncSession) -> int:
        return await _bulk_upsert(
            s, StockDailyAdjusted, rows,
            conflict_cols=["symbol", "timestamp"],
            update_cols=update_cols,
        )

    if session is not None:
        count = await _run(session)
    else:
        async with get_session() as s:
            count = await _run(s)
    logger.info("upsert_stock_daily_adjusted: %d bars for symbol=%s", count, symbol)
    return count


async def upsert_stock_batch_daily(
    results: dict[str, sf.FetchResult],
    interval: str = "daily",
) -> dict[str, int]:
    """
    Upsert a batch of per-symbol stock FetchResults (as returned by
    stock_fetcher.fetch_batch_daily) within a single transaction.

    Returns:
        Dict mapping symbol -> rows upserted.
    """
    counts: dict[str, int] = {}
    async with get_session() as s:
        for symbol, result in results.items():
            counts[symbol] = await upsert_stock_bars(
                symbol, result, interval=interval, session=s
            )
    return counts


async def upsert_crypto_batch_ohlcv(
    results: dict[str, sf.FetchResult],
    symbols: dict[str, str] | None = None,
    vs_currency: str = "usd",
) -> dict[str, int]:
    """
    Upsert a batch of per-coin OHLCV FetchResults (as returned by
    crypto_fetcher.fetch_batch_ohlcv) within a single transaction.

    Args:
        results: Dict mapping coin_id -> FetchResult(list[OHLCVCandle]).
        symbols: Optional mapping coin_id -> ticker symbol.
        vs_currency: Quote currency.

    Returns:
        Dict mapping coin_id -> rows upserted.
    """
    symbols = symbols or {}
    counts: dict[str, int] = {}
    async with get_session() as s:
        for coin_id, result in results.items():
            counts[coin_id] = await upsert_crypto_ohlcv(
                coin_id,
                result,
                symbol=symbols.get(coin_id),
                vs_currency=vs_currency,
                session=s,
            )
    return counts
