"""
Prediction-engine inference service for the Market Predictor.

Responsibilities:
  1. Load trained XGBoost model artifacts (.json booster) and their metadata
     sidecars (.meta.json) produced by model_trainer.py.
  2. Run feature engineering on live OHLCV history using the EXACT same math
     the trainer used — it imports and reuses `engineer_features` from
     model_trainer, then aligns the computed columns to the persisted
     `feature_names`, guaranteeing train/inference parity (RSI, volatilities,
     moving averages, momentum, returns, volume features).
  3. Cleanly handle short / missing history via forward/zero padding so a
     prediction can still be produced (with a clear `padded` warning flag)
     when fewer bars are available than the longest feature window needs.
  4. Produce a typed forecast for the next `horizon` period.

An in-process LRU-style cache keeps loaded boosters warm so the scheduler /
API can call predictions repeatedly without re-reading disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.prediction_engine.model_trainer import (
    MA_WINDOWS,
    VOL_WINDOWS,
    engineer_features,
    get_models_dir,
)

logger = logging.getLogger(__name__)

# The longest look-back window any feature needs; used to size padding and to
# warn when supplied history is shorter than the model was trained to expect.
MIN_RECOMMENDED_HISTORY: int = max((*MA_WINDOWS, *VOL_WINDOWS, 20)) + 1

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# Result / error containers
# ---------------------------------------------------------------------------

class InferenceError(Exception):
    """Raised when a prediction cannot be produced."""


@dataclass
class LoadedModel:
    """A loaded booster plus its training metadata."""

    booster: Any                      # xgboost.XGBRegressor
    feature_names: list[str]
    asset: str
    asset_class: str
    interval: str
    horizon: int
    model_path: str
    metadata: dict[str, Any]


@dataclass
class PredictionResult:
    """Typed forecast returned by the inference service."""

    asset: str
    asset_class: str
    interval: str
    horizon: int
    last_close: float
    predicted_close: float
    predicted_return_pct: float
    direction: str                    # "up" | "down" | "flat"
    n_history: int
    padded: bool
    feature_vector: dict[str, float]
    predicted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Artifact loading (with in-process cache)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, LoadedModel] = {}
_CACHE_LOCK = asyncio.Lock()


def _artifact_stem(asset_class: str, asset: str, interval: str) -> str:
    """Recreate the artifact filename stem used by the trainer."""
    safe_asset = asset.replace("/", "_").replace(" ", "_")
    return f"{asset_class}_{safe_asset}_{interval}"


def _load_model_from_disk(asset_class: str, asset: str, interval: str) -> LoadedModel:
    """
    Load an XGBoost booster and its metadata sidecar from the models/ directory.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - environment guard
        raise InferenceError(
            "xgboost is not installed. It is pinned in backend/requirements.txt."
        ) from exc

    models_dir = get_models_dir()
    stem = _artifact_stem(asset_class, asset, interval)
    model_path = models_dir / f"{stem}.json"
    metadata_path = models_dir / f"{stem}.meta.json"

    if not model_path.exists():
        raise InferenceError(f"Model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise InferenceError(f"Model metadata not found: {metadata_path}")

    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names: list[str] = metadata.get("feature_names", [])
    if not feature_names:
        raise InferenceError(f"Metadata {metadata_path} has no feature_names.")

    horizon = int(metadata.get("config", {}).get("horizon", 1))

    booster = XGBRegressor()
    booster.load_model(str(model_path))

    logger.info(
        "Loaded model %s (%d features, horizon=%d)", stem, len(feature_names), horizon
    )
    return LoadedModel(
        booster=booster,
        feature_names=feature_names,
        asset=metadata.get("asset", asset),
        asset_class=metadata.get("asset_class", asset_class),
        interval=metadata.get("interval", interval),
        horizon=horizon,
        model_path=str(model_path),
        metadata=metadata,
    )


async def load_model(
    asset_class: str,
    asset: str,
    interval: str = "daily",
    *,
    force_reload: bool = False,
) -> LoadedModel:
    """
    Async-safe cached model loader. Disk I/O runs in a thread to avoid blocking
    the event loop.
    """
    key = _artifact_stem(asset_class, asset, interval)
    if not force_reload and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    async with _CACHE_LOCK:
        # Re-check inside the lock to avoid a double load race.
        if not force_reload and key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        loaded = await asyncio.to_thread(
            _load_model_from_disk, asset_class, asset, interval
        )
        _MODEL_CACHE[key] = loaded
        return loaded


def clear_model_cache() -> None:
    """Drop all cached models (e.g. after retraining)."""
    _MODEL_CACHE.clear()
    logger.info("Inference model cache cleared.")


# ---------------------------------------------------------------------------
# Input normalisation & padding
# ---------------------------------------------------------------------------

def _coerce_history_to_frame(history: Any) -> pd.DataFrame:
    """
    Accept several live-stream shapes and normalise them to an OHLCV DataFrame.

    Supported inputs:
      - pandas.DataFrame with open/high/low/close[/volume] columns
      - list[dict] rows with those keys
      - dict of column-arrays {"open": [...], "high": [...], ...}
    """
    if isinstance(history, pd.DataFrame):
        df = history.copy()
    elif isinstance(history, dict):
        df = pd.DataFrame(history)
    elif isinstance(history, (list, tuple)):
        df = pd.DataFrame(list(history))
    else:
        raise InferenceError(
            f"Unsupported history type: {type(history)!r}. "
            "Provide a DataFrame, list[dict], or dict of arrays."
        )

    df.columns = [str(c).lower() for c in df.columns]

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise InferenceError(f"History is missing required columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    for col in REQUIRED_OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the core price columns are entirely unusable.
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def _pad_history(df: pd.DataFrame, min_rows: int) -> tuple[pd.DataFrame, bool]:
    """
    Ensure at least `min_rows` rows by back-padding with the earliest bar.

    Padding replicates the oldest available bar at the front of the series so
    that warm-up windows (SMA/volatility/RSI) can compute without NaNs. Volume
    on padded rows is set to 0 so it does not distort volume-change features.

    Returns (padded_frame, was_padded).
    """
    n = len(df)
    if n == 0:
        raise InferenceError("Cannot pad an empty history; no bars supplied.")
    if n >= min_rows:
        return df, False

    pad_count = min_rows - n
    first_row = df.iloc[0]
    pad_block = pd.DataFrame(
        [{
            "open": first_row["open"],
            "high": first_row["high"],
            "low": first_row["low"],
            "close": first_row["close"],
            "volume": 0.0,
        } for _ in range(pad_count)]
    )
    if "timestamp" in df.columns:
        pad_block["timestamp"] = pd.NaT

    padded = pd.concat([pad_block, df], ignore_index=True)
    logger.warning(
        "History padded: supplied %d bars, padded to %d (warm-up requires %d).",
        n, len(padded), min_rows,
    )
    return padded, True


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def _build_inference_vector(
    df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Run the trainer's feature engineering and return a single-row DataFrame of
    the most recent observation, columns ordered exactly as `feature_names`.

    Any feature column that is missing or NaN at the final row is filled with a
    neutral 0.0 so the booster always receives a complete, ordered vector.
    """
    featured, computed_names = engineer_features(df)

    # Reindex to the model's exact, ordered feature set. Columns the live data
    # could not produce become NaN here, then are neutral-filled below.
    last = featured.iloc[[-1]].copy()
    aligned = last.reindex(columns=feature_names)

    # Replace inf/-inf then neutral-fill remaining NaNs.
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    nan_cols = [c for c in feature_names if pd.isna(aligned.iloc[0][c])]
    if nan_cols:
        logger.debug("Neutral-filling NaN inference features: %s", nan_cols)
    aligned = aligned.fillna(0.0).astype(float)

    return aligned[feature_names]


# ---------------------------------------------------------------------------
# Prediction service
# ---------------------------------------------------------------------------

async def predict(
    asset_class: str,
    asset: str,
    history: Any,
    interval: str = "daily",
    *,
    force_reload: bool = False,
) -> PredictionResult:
    """
    Produce a next-`horizon` close forecast for an asset from live OHLCV history.

    Args:
        asset_class: "crypto" | "stock".
        asset: coin_id or ticker symbol (must match the trained artifact).
        history: live OHLCV history (DataFrame, list[dict], or dict of arrays).
        interval: series interval label matching the trained artifact.
        force_reload: bypass the model cache and re-read from disk.

    Returns:
        PredictionResult with the forecast, direction, and the feature vector.
    """
    model = await load_model(asset_class, asset, interval, force_reload=force_reload)

    df = _coerce_history_to_frame(history)
    df, padded = _pad_history(df, MIN_RECOMMENDED_HISTORY)

    last_close = float(df["close"].iloc[-1])

    feature_row = _build_inference_vector(df, model.feature_names)

    # XGBoost prediction runs in a thread (CPU-bound) to keep the loop free.
    pred_arr = await asyncio.to_thread(model.booster.predict, feature_row)
    predicted_close = float(np.asarray(pred_arr, dtype=float).ravel()[0])

    if last_close > 0:
        predicted_return_pct = (predicted_close - last_close) / last_close * 100.0
    else:
        predicted_return_pct = 0.0

    if predicted_close > last_close:
        direction = "up"
    elif predicted_close < last_close:
        direction = "down"
    else:
        direction = "flat"

    result = PredictionResult(
        asset=model.asset,
        asset_class=model.asset_class,
        interval=model.interval,
        horizon=model.horizon,
        last_close=last_close,
        predicted_close=predicted_close,
        predicted_return_pct=predicted_return_pct,
        direction=direction,
        n_history=int(len(df)),
        padded=padded,
        feature_vector={k: float(feature_row.iloc[0][k]) for k in model.feature_names},
    )

    logger.info(
        "Prediction %s:%s (%s) | last_close=%.6f pred=%.6f (%+.3f%%) dir=%s padded=%s",
        model.asset_class, model.asset, model.interval,
        last_close, predicted_close, predicted_return_pct, direction, padded,
    )
    return result


async def predict_crypto(
    coin_id: str,
    history: Any,
    interval: str = "ohlc",
    *,
    force_reload: bool = False,
) -> PredictionResult:
    """Convenience wrapper: forecast a crypto asset."""
    return await predict("crypto", coin_id, history, interval=interval, force_reload=force_reload)


async def predict_stock(
    symbol: str,
    history: Any,
    interval: str = "daily",
    *,
    force_reload: bool = False,
) -> PredictionResult:
    """Convenience wrapper: forecast an equity."""
    return await predict("stock", symbol, history, interval=interval, force_reload=force_reload)


async def predict_batch(
    requests: list[dict[str, Any]],
    concurrency: int = 4,
) -> list[PredictionResult | dict[str, Any]]:
    """
    Run multiple predictions concurrently.

    Each request dict must contain: asset_class, asset, history, and optionally
    interval. Failed predictions are returned as error dicts so one bad asset
    does not abort the batch.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(req: dict[str, Any]) -> PredictionResult | dict[str, Any]:
        async with semaphore:
            try:
                return await predict(
                    asset_class=req["asset_class"],
                    asset=req["asset"],
                    history=req["history"],
                    interval=req.get("interval", "daily"),
                )
            except (InferenceError, KeyError) as exc:
                logger.error("Batch prediction failed for %s: %s", req.get("asset"), exc)
                return {"asset": req.get("asset"), "error": str(exc), "success": False}

    tasks = [asyncio.create_task(_one(r)) for r in requests]
    return list(await asyncio.gather(*tasks))
