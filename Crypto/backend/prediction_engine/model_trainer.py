"""
Prediction-engine model trainer for the Market Predictor.

Pipeline:
  1. Load historical OHLCV bars from the database (crypto_market_data /
     stock_data) into a pandas DataFrame.
  2. Engineer features: simple/exponential moving averages, rolling
     volatilities, normalized log returns, momentum, RSI, and volume changes.
  3. Build a supervised regression target (next-period close) and train an
     XGBoost gradient-boosted ensemble with an explicit chronological
     train/test split (no shuffling — this is time-series data).
  4. Evaluate with MAE, RMSE, and directional accuracy.
  5. Persist the trained model + feature metadata as artifacts under the
     configured models/ directory.

The heavy numeric work uses NumPy/Pandas; the model is XGBoost (already pinned
in requirements.txt). All DB access is async via the project's session layer;
a synchronous `train_symbol_sync` wrapper is provided for CLI / scheduler use.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.config.settings import get_settings
from backend.database.models import MarketData, StockData
from backend.database.session import get_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters and split configuration for a training run."""

    test_size: float = 0.2          # fraction of (chronologically) latest rows held out
    horizon: int = 1                # predict the close `horizon` periods ahead
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    early_stopping_rounds: int = 30
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in the open interval (0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


@dataclass
class EvaluationMetrics:
    """Hold-out evaluation metrics for a trained model."""

    mae: float
    rmse: float
    directional_accuracy: float
    n_train: int
    n_test: int
    r2: float


@dataclass
class TrainingArtifacts:
    """Everything produced by a successful training run."""

    asset: str
    asset_class: str               # "crypto" | "stock"
    interval: str
    metrics: EvaluationMetrics
    feature_names: list[str]
    model_path: str
    metadata_path: str
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TrainingError(Exception):
    """Raised when training cannot proceed (e.g. insufficient data)."""


# ---------------------------------------------------------------------------
# Artifact directory resolution
# ---------------------------------------------------------------------------

def get_models_dir() -> Path:
    """
    Resolve and create the models artifact directory.

    Derived from settings.base_dir (the backend root) as `<base_dir>/models`.
    """
    base = get_settings().base_dir
    models_dir = Path(base) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def load_crypto_ohlcv(coin_id: str) -> pd.DataFrame:
    """Load all OHLCV bars for a crypto coin_id, ordered ascending by time."""
    async with get_session() as session:
        stmt = (
            select(
                MarketData.timestamp,
                MarketData.open,
                MarketData.high,
                MarketData.low,
                MarketData.close,
                MarketData.volume,
            )
            .where(MarketData.coin_id == coin_id)
            .order_by(MarketData.timestamp.asc())
        )
        rows = (await session.execute(stmt)).all()
    return _rows_to_frame(rows)


async def load_stock_ohlcv(symbol: str, interval: str = "daily") -> pd.DataFrame:
    """Load all OHLCV bars for a stock symbol+interval, ordered ascending by time."""
    async with get_session() as session:
        stmt = (
            select(
                StockData.timestamp,
                StockData.open,
                StockData.high,
                StockData.low,
                StockData.close,
                StockData.volume,
            )
            .where(StockData.symbol == symbol, StockData.interval == interval)
            .order_by(StockData.timestamp.asc())
        )
        rows = (await session.execute(stmt)).all()
    return _rows_to_frame(rows)


def _rows_to_frame(rows: list[Any]) -> pd.DataFrame:
    """Convert SQLAlchemy result rows into a typed, time-indexed DataFrame."""
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

MA_WINDOWS: tuple[int, ...] = (5, 10, 20, 50)
VOL_WINDOWS: tuple[int, ...] = (5, 10, 20)
MOMENTUM_WINDOWS: tuple[int, ...] = (3, 7, 14)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate technical features from an OHLCV DataFrame.

    Produces:
      - log returns and normalized (z-scored) returns
      - simple & exponential moving averages and price/SMA ratios
      - rolling volatility (std of log returns) over several windows
      - momentum (n-period percentage change)
      - RSI (14)
      - volume change and rolling-mean volume ratio

    Returns the feature-augmented frame (NaNs from warm-up dropped) and the
    ordered list of feature column names.
    """
    if df.empty:
        raise TrainingError("Cannot engineer features on an empty DataFrame.")

    out = df.copy()
    close = out["close"].astype(float)

    feature_names: list[str] = []

    # --- Returns ---
    out["log_return"] = np.log(close / close.shift(1))
    feature_names.append("log_return")

    roll_mean_ret = out["log_return"].rolling(window=20, min_periods=20).mean()
    roll_std_ret = out["log_return"].rolling(window=20, min_periods=20).std()
    out["norm_return"] = (out["log_return"] - roll_mean_ret) / roll_std_ret.replace(0, np.nan)
    feature_names.append("norm_return")

    # --- Moving averages & price/SMA ratios ---
    for w in MA_WINDOWS:
        sma_col = f"sma_{w}"
        ema_col = f"ema_{w}"
        ratio_col = f"close_sma_{w}_ratio"
        out[sma_col] = close.rolling(window=w, min_periods=w).mean()
        out[ema_col] = close.ewm(span=w, adjust=False, min_periods=w).mean()
        out[ratio_col] = close / out[sma_col].replace(0, np.nan)
        feature_names.extend([sma_col, ema_col, ratio_col])

    # --- Rolling volatility ---
    for w in VOL_WINDOWS:
        vol_col = f"volatility_{w}"
        out[vol_col] = out["log_return"].rolling(window=w, min_periods=w).std()
        feature_names.append(vol_col)

    # --- Momentum ---
    for w in MOMENTUM_WINDOWS:
        mom_col = f"momentum_{w}"
        out[mom_col] = close.pct_change(periods=w)
        feature_names.append(mom_col)

    # --- RSI (14) ---
    out["rsi_14"] = _compute_rsi(close, period=14)
    feature_names.append("rsi_14")

    # --- High/Low range & volume features ---
    out["hl_range"] = (out["high"] - out["low"]) / close.replace(0, np.nan)
    feature_names.append("hl_range")

    volume = out["volume"].astype(float).fillna(0.0)
    out["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    vol_ma = volume.rolling(window=20, min_periods=20).mean()
    out["volume_ratio"] = volume / vol_ma.replace(0, np.nan)
    feature_names.extend(["volume_change", "volume_ratio"])

    return out, feature_names


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def build_supervised_dataset(
    df: pd.DataFrame,
    feature_names: list[str],
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the supervised (X, y) dataset.

    Target is the close price `horizon` periods in the future. Rows with NaNs
    in any feature (warm-up period) or in the target (final rows) are dropped.
    """
    data = df.copy()
    data["target"] = data["close"].shift(-horizon)

    cols = feature_names + ["target"]
    data = data[cols + ["timestamp"]].replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise TrainingError(
            "No rows remain after feature warm-up and target alignment. "
            "More historical data is required."
        )

    X = data[feature_names].astype(float)
    y = data["target"].astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Train / test split (chronological — NO shuffling for time series)
# ---------------------------------------------------------------------------

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X/y by time order: the most recent `test_size` fraction is the test set."""
    n = len(X)
    if n < 10:
        raise TrainingError(f"Need at least 10 samples to train; got {n}.")
    split_idx = int(n * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= n:
        raise TrainingError("test_size produces an empty train or test set.")
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    last_known_close: np.ndarray,
    n_train: int,
) -> EvaluationMetrics:
    """
    Compute regression and directional metrics.

    Directional accuracy compares the predicted move direction (vs. the last
    known close at prediction time) against the actual move direction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    last_known_close = np.asarray(last_known_close, dtype=float)

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    actual_dir = np.sign(y_true - last_known_close)
    pred_dir = np.sign(y_pred - last_known_close)
    # Treat flat (0) predictions as matching only exact flats.
    directional_accuracy = float(np.mean(actual_dir == pred_dir)) if len(y_true) else 0.0

    return EvaluationMetrics(
        mae=mae,
        rmse=rmse,
        directional_accuracy=directional_accuracy,
        n_train=n_train,
        n_test=int(len(y_true)),
        r2=r2,
    )


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------

def _train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    last_close_full: pd.Series,
    config: TrainingConfig,
):
    """
    Train an XGBoost regressor with a chronological split and early stopping.

    Returns:
        (fitted_model, EvaluationMetrics)
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - environment guard
        raise TrainingError(
            "xgboost is not installed. Add it to your environment (it is pinned "
            "in backend/requirements.txt)."
        ) from exc

    X_train, X_test, y_train, y_test = chronological_split(X, y, config.test_size)
    # Align the 'last known close' rows with the test target rows for direction.
    last_close_test = last_close_full.loc[X_test.index]

    model = XGBRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        random_state=config.random_state,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=config.early_stopping_rounds,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    metrics = evaluate(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        last_known_close=last_close_test.to_numpy(),
        n_train=len(X_train),
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def _save_artifacts(
    model: Any,
    asset: str,
    asset_class: str,
    interval: str,
    feature_names: list[str],
    metrics: EvaluationMetrics,
    config: TrainingConfig,
) -> tuple[str, str]:
    """
    Persist the model and a JSON metadata sidecar to the models/ directory.

    Model is saved natively via XGBoost's `save_model` (.json booster format),
    which is portable and version-stable.

    Returns:
        (model_path, metadata_path)
    """
    models_dir = get_models_dir()
    safe_asset = asset.replace("/", "_").replace(" ", "_")
    stem = f"{asset_class}_{safe_asset}_{interval}"

    model_path = models_dir / f"{stem}.json"
    metadata_path = models_dir / f"{stem}.meta.json"

    model.save_model(str(model_path))

    metadata = {
        "asset": asset,
        "asset_class": asset_class,
        "interval": interval,
        "feature_names": feature_names,
        "metrics": asdict(metrics),
        "config": asdict(config),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_format": "xgboost-json",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Saved model artifact -> %s", model_path)
    logger.info("Saved model metadata -> %s", metadata_path)
    return str(model_path), str(metadata_path)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def train_from_frame(
    df: pd.DataFrame,
    asset: str,
    asset_class: str,
    interval: str,
    config: TrainingConfig | None = None,
) -> TrainingArtifacts:
    """
    Run the full training pipeline on an already-loaded OHLCV DataFrame.

    This is the synchronous core; DB-loading wrappers call it after fetching.
    """
    config = config or TrainingConfig()

    if df.empty:
        raise TrainingError(f"No data available to train {asset_class}:{asset}.")

    featured, feature_names = engineer_features(df)
    X, y = build_supervised_dataset(featured, feature_names, config.horizon)

    # 'last known close' aligned to X's surviving index for directional accuracy.
    last_close_full = featured.loc[X.index, "close"].astype(float)

    model, metrics = _train_xgboost(X, y, last_close_full, config)

    model_path, metadata_path = _save_artifacts(
        model, asset, asset_class, interval, feature_names, metrics, config
    )

    logger.info(
        "Trained %s:%s (%s) | MAE=%.6f RMSE=%.6f DirAcc=%.3f R2=%.3f (train=%d test=%d)",
        asset_class, asset, interval,
        metrics.mae, metrics.rmse, metrics.directional_accuracy, metrics.r2,
        metrics.n_train, metrics.n_test,
    )

    return TrainingArtifacts(
        asset=asset,
        asset_class=asset_class,
        interval=interval,
        metrics=metrics,
        feature_names=feature_names,
        model_path=model_path,
        metadata_path=metadata_path,
    )


async def train_crypto(
    coin_id: str,
    config: TrainingConfig | None = None,
) -> TrainingArtifacts:
    """Load crypto OHLCV from the DB and train a model for `coin_id`."""
    df = await load_crypto_ohlcv(coin_id)
    return train_from_frame(df, asset=coin_id, asset_class="crypto", interval="ohlc", config=config)


async def train_stock(
    symbol: str,
    interval: str = "daily",
    config: TrainingConfig | None = None,
) -> TrainingArtifacts:
    """Load stock OHLCV from the DB and train a model for `symbol`."""
    df = await load_stock_ohlcv(symbol, interval=interval)
    return train_from_frame(df, asset=symbol, asset_class="stock", interval=interval, config=config)


def train_stock_sync(
    symbol: str,
    interval: str = "daily",
    config: TrainingConfig | None = None,
) -> TrainingArtifacts:
    """Synchronous convenience wrapper around `train_stock` for CLI/scheduler use."""
    return asyncio.run(train_stock(symbol, interval=interval, config=config))


def train_crypto_sync(
    coin_id: str,
    config: TrainingConfig | None = None,
) -> TrainingArtifacts:
    """Synchronous convenience wrapper around `train_crypto` for CLI/scheduler use."""
    return asyncio.run(train_crypto(coin_id, config=config))


if __name__ == "__main__":  # pragma: no cover - manual smoke entry point
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m backend.prediction_engine.model_trainer <crypto|stock> <asset> [interval]")
        sys.exit(1)

    asset_class_arg = sys.argv[1].lower()
    asset_arg = sys.argv[2]
    interval_arg = sys.argv[3] if len(sys.argv) > 3 else "daily"

    try:
        if asset_class_arg == "crypto":
            artifacts = train_crypto_sync(asset_arg)
        elif asset_class_arg == "stock":
            artifacts = train_stock_sync(asset_arg, interval=interval_arg)
        else:
            print(f"Unknown asset class: {asset_class_arg!r}")
            sys.exit(1)
        print(json.dumps(asdict(artifacts.metrics), indent=2))
        print(f"Model saved to: {artifacts.model_path}")
    except TrainingError as exc:
        print(f"[TRAINING ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
