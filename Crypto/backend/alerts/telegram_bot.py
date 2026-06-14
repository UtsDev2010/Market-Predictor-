"""
Asynchronous Telegram alert engine for the Market Predictor.

Dispatches high-confidence price-trend forecast alerts to a configured Telegram
chat/channel using the Bot API over httpx (consistent with the rest of the
codebase's async/httpx style), driven by the `TelegramSettings` config group.

Features:
  - Async send via httpx.AsyncClient
  - Per-asset cooldown so duplicate alerts are suppressed within a window
  - Confidence gating: only forecasts whose absolute predicted move exceeds a
    threshold (and which are not based on padded/low-quality history) are sent
  - MarkdownV2-safe message formatting with full special-character escaping
  - Structured logging; the bot token is never logged
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from backend.config.settings import TelegramSettings, get_settings
from backend.prediction_engine.model_inference import PredictionResult

logger = logging.getLogger(__name__)

# Telegram MarkdownV2 reserved characters that must be backslash-escaped.
_MARKDOWN_V2_SPECIALS = r"_*[]()~`>#+-=|{}.!"


def escape_markdown_v2(text: str) -> str:
    """Escape all MarkdownV2 reserved characters in a string."""
    return "".join(f"\\{ch}" if ch in _MARKDOWN_V2_SPECIALS else ch for ch in text)


@dataclass
class AlertDispatchResult:
    """Outcome of an attempt to dispatch an alert."""

    sent: bool
    skipped_reason: str | None = None
    status_code: int | None = None
    error: str | None = None


class TelegramAlertEngine:
    """
    Async Telegram alert dispatcher.

    Usage:
        async with TelegramAlertEngine() as engine:
            await engine.dispatch_forecast(prediction_result)
    """

    def __init__(self, settings: TelegramSettings | None = None) -> None:
        self._cfg: TelegramSettings = settings or get_settings().telegram
        self._client: httpx.AsyncClient | None = None
        self._base_url = f"https://api.telegram.org/bot{self._cfg.bot_token_value}"
        # Tracks the last send time per dedup-key for cooldown enforcement.
        self._last_sent: dict[str, float] = {}
        self._lock = asyncio.Lock()
        logger.info(
            "TelegramAlertEngine initialised | chat_id=%s cooldown=%ds parse_mode=%s",
            self._cfg.chat_id, self._cfg.alert_cooldown_seconds, self._cfg.parse_mode,
        )

    # -- lifecycle --

    async def open(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(float(self._cfg.request_timeout)),
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "TelegramAlertEngine":
        await self.open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # -- cooldown --

    def _cooldown_active(self, key: str) -> bool:
        last = self._last_sent.get(key)
        if last is None:
            return False
        return (time.monotonic() - last) < self._cfg.alert_cooldown_seconds

    def _mark_sent(self, key: str) -> None:
        self._last_sent[key] = time.monotonic()

    # -- core send --

    async def send_message(self, text: str) -> AlertDispatchResult:
        """
        Send a raw (already-escaped) message to the configured chat.
        """
        if self._client is None:
            raise RuntimeError(
                "TelegramAlertEngine is not open. Use 'async with TelegramAlertEngine()'."
            )

        payload = {
            "chat_id": self._cfg.chat_id,
            "text": text,
            "parse_mode": self._cfg.parse_mode,
            "disable_web_page_preview": self._cfg.disable_web_page_preview,
        }
        try:
            response = await self._client.post(f"{self._base_url}/sendMessage", json=payload)
        except httpx.HTTPError as exc:
            logger.error("Telegram send failed (transport): %s", exc)
            return AlertDispatchResult(sent=False, error=str(exc))

        if response.status_code == 200 and response.json().get("ok"):
            return AlertDispatchResult(sent=True, status_code=200)

        detail = response.text[:300]
        logger.error("Telegram send failed: HTTP %d %s", response.status_code, detail)
        return AlertDispatchResult(
            sent=False, status_code=response.status_code, error=detail
        )

    # -- forecast alerting --

    def _format_forecast(self, result: PredictionResult) -> str:
        """Build a MarkdownV2-escaped alert message from a PredictionResult."""
        arrow = {"up": "\U0001F4C8", "down": "\U0001F4C9", "flat": "\u27A1\uFE0F"}.get(
            result.direction, "\u27A1\uFE0F"
        )
        title = f"{arrow} Forecast Alert: {result.asset.upper()} ({result.asset_class})"
        lines = [
            title,
            "",
            f"Interval: {result.interval} | Horizon: {result.horizon}",
            f"Last close: {result.last_close:.6f}",
            f"Predicted: {result.predicted_close:.6f} ({result.predicted_return_pct:+.2f}%)",
            f"Direction: {result.direction.upper()}",
        ]
        if result.padded:
            lines.append("\u26A0\uFE0F Low-confidence: history was padded.")
        raw = "\n".join(lines)
        return escape_markdown_v2(raw)

    async def dispatch_forecast(
        self,
        result: PredictionResult,
        min_abs_return_pct: float = 2.0,
    ) -> AlertDispatchResult:
        """
        Conditionally dispatch a forecast alert.

        An alert is sent only when:
          - the forecast is not based on padded (low-quality) history, AND
          - the absolute predicted move >= `min_abs_return_pct`, AND
          - the per-asset cooldown window has elapsed.

        Returns an AlertDispatchResult describing what happened.
        """
        if result.padded:
            return AlertDispatchResult(sent=False, skipped_reason="padded_history")

        if abs(result.predicted_return_pct) < min_abs_return_pct:
            return AlertDispatchResult(sent=False, skipped_reason="below_threshold")

        dedup_key = f"{result.asset_class}:{result.asset}:{result.interval}:{result.direction}"

        async with self._lock:
            if self._cooldown_active(dedup_key):
                return AlertDispatchResult(sent=False, skipped_reason="cooldown")
            self._mark_sent(dedup_key)

        message = self._format_forecast(result)
        dispatch = await self.send_message(message)
        if dispatch.sent:
            logger.info("Dispatched forecast alert for %s", dedup_key)
        return dispatch


async def send_forecast_alert(
    result: PredictionResult,
    min_abs_return_pct: float = 2.0,
) -> AlertDispatchResult:
    """
    Convenience one-shot: open an engine, dispatch a single forecast alert,
    and close. Intended for direct use by the scheduler.
    """
    async with TelegramAlertEngine() as engine:
        return await engine.dispatch_forecast(result, min_abs_return_pct=min_abs_return_pct)
