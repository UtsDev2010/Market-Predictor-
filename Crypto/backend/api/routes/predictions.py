"""
REST endpoints exposing historical OHLCV data and live model inference.

Routes (mounted under the configured API v1 prefix):
  GET  /crypto/{coin_id}/ohlcv        -> historical crypto OHLCV bars
  GET  /stocks/{symbol}/ohlcv         -> historical stock OHLCV bars
  POST /crypto/{coin_id}/predict      -> live crypto forecast
  POST /stocks/{symbol}/predict       -> live stock forecast

The OHLCV endpoints read from the database via the async session dependency.
The predict endpoints accept a JSON body of recent OHLCV history and run the
cached XGBoost inference service.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import MarketData, StockData
from backend.database.session import get_db
from backend.prediction_engine.model_inference import (
    InferenceError,
    PredictionResult,
    predict_crypto,
    predict_stock,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OHLCVBar(BaseModel):
    """A single OHLCV bar returned by the history endpoints."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class OHLCVResponse(BaseModel):
    """Envelope for a historical OHLCV series."""

    asset: str
    asset_class: str
    interval: str
    count: int
    bars: list[OHLCVBar]


class HistoryBar(BaseModel):
    """A single OHLCV bar supplied in a prediction request body."""

    timestamp: datetime | None = None
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class PredictRequest(BaseModel):
    """Request body for a live prediction: recent OHLCV history."""

    history: list[HistoryBar] = Field(..., min_length=1)
    interval: str | None = None


class PredictResponse(BaseModel):
    """Forecast response surfaced to API clients."""

    asset: str
    asset_class: str
    interval: str
    horizon: int
    last_close: float
    predicted_close: float
    predicted_return_pct: float
    direction: str
    n_history: int
    padded: bool
    predicted_at: datetime

    @classmethod
    def from_result(cls, result: PredictionResult) -> "PredictResponse":
        return cls(
            asset=result.asset,
            asset_class=result.asset_class,
            interval=result.interval,
            horizon=result.horizon,
            last_close=result.last_close,
            predicted_close=result.predicted_close,
            predicted_return_pct=result.predicted_return_pct,
            direction=result.direction,
            n_history=result.n_history,
            padded=result.padded,
            predicted_at=result.predicted_at,
        )


# ---------------------------------------------------------------------------
# Historical OHLCV endpoints
# ---------------------------------------------------------------------------

@router.get("/crypto/{coin_id}/ohlcv", response_model=OHLCVResponse)
async def get_crypto_ohlcv(
    coin_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    db: AsyncSession = Depends(get_db),
) -> OHLCVResponse:
    """Return the most recent `limit` crypto OHLCV bars for a coin, oldest-first."""
    stmt = (
        select(MarketData)
        .where(MarketData.coin_id == coin_id)
        .order_by(MarketData.timestamp.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No OHLCV data found for crypto '{coin_id}'.",
        )

    bars = [
        OHLCVBar(
            timestamp=r.timestamp,
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=float(r.volume) if r.volume is not None else None,
        )
        for r in reversed(rows)
    ]
    return OHLCVResponse(
        asset=coin_id, asset_class="crypto", interval="ohlc",
        count=len(bars), bars=bars,
    )


@router.get("/stocks/{symbol}/ohlcv", response_model=OHLCVResponse)
async def get_stock_ohlcv(
    symbol: str,
    interval: str = Query(default="daily"),
    limit: int = Query(default=500, ge=1, le=5000),
    db: AsyncSession = Depends(get_db),
) -> OHLCVResponse:
    """Return the most recent `limit` stock OHLCV bars for a symbol, oldest-first."""
    stmt = (
        select(StockData)
        .where(StockData.symbol == symbol, StockData.interval == interval)
        .order_by(StockData.timestamp.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No OHLCV data found for stock '{symbol}' (interval={interval}).",
        )

    bars = [
        OHLCVBar(
            timestamp=r.timestamp,
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=float(r.volume),
        )
        for r in reversed(rows)
    ]
    return OHLCVResponse(
        asset=symbol, asset_class="stock", interval=interval,
        count=len(bars), bars=bars,
    )


# ---------------------------------------------------------------------------
# Live prediction endpoints
# ---------------------------------------------------------------------------

def _history_payload(req: PredictRequest) -> list[dict[str, Any]]:
    """Convert the request body history into the dict rows the service expects."""
    return [bar.model_dump() for bar in req.history]


@router.post("/crypto/{coin_id}/predict", response_model=PredictResponse)
async def predict_crypto_endpoint(
    coin_id: str,
    req: PredictRequest,
) -> PredictResponse:
    """Run a live forecast for a crypto asset from supplied recent history."""
    try:
        result = await predict_crypto(
            coin_id,
            _history_payload(req),
            interval=req.interval or "ohlc",
        )
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return PredictResponse.from_result(result)


@router.post("/stocks/{symbol}/predict", response_model=PredictResponse)
async def predict_stock_endpoint(
    symbol: str,
    req: PredictRequest,
) -> PredictResponse:
    """Run a live forecast for an equity from supplied recent history."""
    try:
        result = await predict_stock(
            symbol,
            _history_payload(req),
            interval=req.interval or "daily",
        )
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return PredictResponse.from_result(result)
