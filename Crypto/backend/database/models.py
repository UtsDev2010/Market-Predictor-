"""
SQLAlchemy 2.0 async declarative models for the Market Predictor persistence layer.

Designed for TimescaleDB (PostgreSQL extension):
  - Every time-series table uses a COMPOSITE PRIMARY KEY of (symbol/coin_id, timestamp).
    This is required because TimescaleDB hypertables must include the partitioning
    (time) column in every unique constraint / primary key.
  - The `timestamp` column is the intended hypertable partitioning dimension.
    After table creation, convert each table to a hypertable, e.g.:
        SELECT create_hypertable('crypto_market_data', 'timestamp',
                                 chunk_time_interval => INTERVAL '7 days');
  - Secondary indexes are declared on (symbol, timestamp DESC) to accelerate the
    most common query pattern: "latest N bars for a given symbol".

Models:
  - MarketData            : crypto OHLCV price bars (CoinGecko)
  - CryptoMarketSnapshot  : crypto market-cap / volume snapshots (CoinGecko /coins/markets)
  - StockData             : equity OHLCV bars (Alpha Vantage TIME_SERIES_*)
  - StockDailyAdjusted    : equity daily-adjusted bars incl. dividends & splits

All models use SQLAlchemy 2.0 typed `Mapped[...]` / `mapped_column(...)` declarative
style and are fully compatible with the async engine configured in session.py.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    MetaData,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ---------------------------------------------------------------------------
# Naming convention for constraints/indexes (keeps Alembic migrations stable)
# ---------------------------------------------------------------------------

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""

    metadata = metadata


# ---------------------------------------------------------------------------
# Reusable column type aliases
# ---------------------------------------------------------------------------
#
# Numeric(20, 8) gives 12 integer digits + 8 decimal places: enough headroom for
# both high-priced equities and low-priced / high-precision crypto assets while
# avoiding floating-point rounding error in financial data.

_PRICE = Numeric(20, 8)
_LARGE = Numeric(30, 8)  # market cap / total volume (can be very large)


# ---------------------------------------------------------------------------
# Crypto OHLCV price bars
# ---------------------------------------------------------------------------

class MarketData(Base):
    """
    Crypto OHLCV price bars sourced from CoinGecko.

    Composite PK (coin_id, timestamp) prepares this table for conversion into a
    TimescaleDB hypertable partitioned on `timestamp`.
    """

    __tablename__ = "crypto_market_data"

    coin_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )

    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    vs_currency: Mapped[str] = mapped_column(String(16), nullable=False, default="usd")

    open: Mapped[float] = mapped_column(_PRICE, nullable=False)
    high: Mapped[float] = mapped_column(_PRICE, nullable=False)
    low: Mapped[float] = mapped_column(_PRICE, nullable=False)
    close: Mapped[float] = mapped_column(_PRICE, nullable=False)
    volume: Mapped[float | None] = mapped_column(_LARGE, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_crypto_market_data_coin_ts", "coin_id", "timestamp"),
        Index("ix_crypto_market_data_symbol_ts", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<MarketData coin_id={self.coin_id!r} ts={self.timestamp!r} "
            f"close={self.close!r}>"
        )


class CryptoMarketSnapshot(Base):
    """
    Crypto market snapshot from CoinGecko /coins/markets.

    Captures point-in-time market metrics (price, market cap, 24h stats) rather
    than candlestick bars. Composite PK (coin_id, timestamp) for hypertable use.
    """

    __tablename__ = "crypto_market_snapshot"

    coin_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )

    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    vs_currency: Mapped[str] = mapped_column(String(16), nullable=False, default="usd")

    current_price: Mapped[float] = mapped_column(_PRICE, nullable=False)
    market_cap: Mapped[float] = mapped_column(_LARGE, nullable=False)
    market_cap_rank: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_volume: Mapped[float] = mapped_column(_LARGE, nullable=False)
    high_24h: Mapped[float | None] = mapped_column(_PRICE, nullable=True)
    low_24h: Mapped[float | None] = mapped_column(_PRICE, nullable=True)
    price_change_24h: Mapped[float | None] = mapped_column(_PRICE, nullable=True)
    price_change_pct_24h: Mapped[float | None] = mapped_column(Numeric(12, 6), nullable=True)
    circulating_supply: Mapped[float | None] = mapped_column(_LARGE, nullable=True)
    total_supply: Mapped[float | None] = mapped_column(_LARGE, nullable=True)
    max_supply: Mapped[float | None] = mapped_column(_LARGE, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_crypto_market_snapshot_symbol_ts", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<CryptoMarketSnapshot coin_id={self.coin_id!r} ts={self.timestamp!r} "
            f"price={self.current_price!r}>"
        )


# ---------------------------------------------------------------------------
# Equity OHLCV bars (Alpha Vantage)
# ---------------------------------------------------------------------------

class StockData(Base):
    """
    Equity OHLCV bars sourced from Alpha Vantage TIME_SERIES_* endpoints.

    Composite PK (symbol, interval, timestamp): the `interval` is part of the key
    so intraday (5min), daily, weekly, and monthly series for the same symbol can
    coexist without collision. `timestamp` remains the hypertable partition column.
    """

    __tablename__ = "stock_data"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    interval: Mapped[str] = mapped_column(String(16), primary_key=True, default="daily")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )

    open: Mapped[float] = mapped_column(_PRICE, nullable=False)
    high: Mapped[float] = mapped_column(_PRICE, nullable=False)
    low: Mapped[float] = mapped_column(_PRICE, nullable=False)
    close: Mapped[float] = mapped_column(_PRICE, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_stock_data_symbol_ts", "symbol", "timestamp"),
        Index("ix_stock_data_symbol_interval_ts", "symbol", "interval", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<StockData symbol={self.symbol!r} interval={self.interval!r} "
            f"ts={self.timestamp!r} close={self.close!r}>"
        )


class StockDailyAdjusted(Base):
    """
    Equity daily-adjusted bars from Alpha Vantage TIME_SERIES_DAILY_ADJUSTED,
    including dividend amount and split coefficient for corporate-action handling.

    Composite PK (symbol, timestamp) for hypertable conversion on `timestamp`.
    """

    __tablename__ = "stock_daily_adjusted"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )

    open: Mapped[float] = mapped_column(_PRICE, nullable=False)
    high: Mapped[float] = mapped_column(_PRICE, nullable=False)
    low: Mapped[float] = mapped_column(_PRICE, nullable=False)
    close: Mapped[float] = mapped_column(_PRICE, nullable=False)
    adjusted_close: Mapped[float] = mapped_column(_PRICE, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    dividend_amount: Mapped[float] = mapped_column(_PRICE, nullable=False, default=0)
    split_coefficient: Mapped[float] = mapped_column(Numeric(12, 6), nullable=False, default=1)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_stock_daily_adjusted_symbol_ts", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<StockDailyAdjusted symbol={self.symbol!r} ts={self.timestamp!r} "
            f"adj_close={self.adjusted_close!r}>"
        )
