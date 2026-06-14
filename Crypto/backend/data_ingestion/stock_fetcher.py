"""
Production-grade asynchronous Alpha Vantage API client.

Features:
  - httpx.AsyncClient with connection pooling and configurable timeouts
  - Token-bucket rate limiter (asyncio-native), tuned for Alpha Vantage's
    strict free-tier limit of 5 requests/minute
  - Exponential-backoff retry with jitter for transient errors
  - Alpha Vantage-specific soft-error handling (the API returns HTTP 200 with
    a JSON body containing 'Note', 'Information', or 'Error Message' keys when
    rate limited or on bad input) -- these are detected and handled correctly
  - Structured logging on every request, retry, and failure
  - Typed dataclasses for all returned payloads
  - Async context-manager lifecycle (open / close)
  - Convenience top-level fetch functions for the scheduler

Mirrors the structural patterns established in crypto_fetcher.py.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import httpx

from backend.config.settings import AlphaVantageSettings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AlphaVantageError(Exception):
    """Base exception for all Alpha Vantage client errors."""


class AlphaVantageRateLimitError(AlphaVantageError):
    """Raised when the API rate limit is hit and all retries are exhausted.

    Alpha Vantage signals rate limiting in two ways:
      - HTTP 429 (rare, on the premium gateway)
      - HTTP 200 with a JSON 'Note' / 'Information' message (free tier)
    """

    def __init__(self, retry_after: float | None = None, detail: str | None = None) -> None:
        self.retry_after = retry_after
        self.detail = detail
        msg = "Alpha Vantage rate limit exceeded."
        if detail:
            msg = f"{msg} {detail}"
        if retry_after:
            msg = f"{msg} Retry after: {retry_after}s"
        super().__init__(msg)


class AlphaVantageHTTPError(AlphaVantageError):
    """Raised for non-retryable HTTP errors (4xx except 429)."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class AlphaVantageAPIError(AlphaVantageError):
    """Raised when the API returns an explicit 'Error Message' in a 200 body
    (e.g. invalid symbol or malformed parameters). Not retryable."""


class AlphaVantageTimeoutError(AlphaVantageError):
    """Raised when a request times out after all retries."""


class AlphaVantageMaxRetriesError(AlphaVantageError):
    """Raised when max retry attempts are exhausted."""


# ---------------------------------------------------------------------------
# Typed response dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StockBar:
    """A single OHLCV bar from a TIME_SERIES_* endpoint."""

    timestamp: datetime          # UTC
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True, slots=True)
class DailyAdjustedBar:
    """A single bar from TIME_SERIES_DAILY_ADJUSTED, including corporate actions."""

    timestamp: datetime          # UTC (date at midnight)
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    dividend_amount: float
    split_coefficient: float


@dataclass(frozen=True, slots=True)
class GlobalQuote:
    """A real-time-ish snapshot from the GLOBAL_QUOTE endpoint."""

    symbol: str
    open: float
    high: float
    low: float
    price: float
    volume: int
    latest_trading_day: date
    previous_close: float
    change: float
    change_percent: float


@dataclass(frozen=True, slots=True)
class SymbolMatch:
    """A single result from the SYMBOL_SEARCH endpoint."""

    symbol: str
    name: str
    type: str
    region: str
    currency: str
    match_score: float


@dataclass(frozen=True, slots=True)
class CompanyOverview:
    """Fundamental company data from the OVERVIEW endpoint."""

    symbol: str
    name: str
    description: str
    exchange: str
    currency: str
    sector: str
    industry: str
    market_capitalization: float | None
    pe_ratio: float | None
    peg_ratio: float | None
    dividend_yield: float | None
    eps: float | None
    beta: float | None
    week_52_high: float | None
    week_52_low: float | None


@dataclass
class FetchResult:
    """Wrapper returned by every public fetch method."""

    success: bool
    data: Any = None
    error: str | None = None
    status_code: int | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------

class AsyncTokenBucket:
    """
    Asyncio-native token-bucket rate limiter.

    Allows `capacity` tokens per `period` seconds. Each call to ``acquire()``
    consumes one token, blocking until one is available. For Alpha Vantage's
    free tier this is configured as 5 tokens / 60 seconds, guaranteeing we
    never exceed 5 requests per minute regardless of concurrency.
    """

    def __init__(self, capacity: int, period: float) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if period <= 0:
            raise ValueError("period must be > 0")

        self._capacity = capacity
        self._period = period
        self._tokens: float = float(capacity)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def refill_rate(self) -> float:
        """Tokens added per second."""
        return self._capacity / self._period

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self.refill_rate,
        )
        self._last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        """Block until ``tokens`` tokens are available, then consume them."""
        if tokens > self._capacity:
            raise ValueError(
                f"Requested {tokens} tokens but bucket capacity is {self._capacity}"
            )
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_time = deficit / self.refill_rate

            logger.debug(
                "Rate limiter: waiting %.3fs for token availability", wait_time
            )
            await asyncio.sleep(wait_time)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _backoff_delay(attempt: int, factor: float) -> float:
    """Compute jittered exponential backoff: factor * 2^attempt + random jitter."""
    base = factor * (2 ** attempt)
    jitter = random.uniform(0, base * 0.25)  # up to 25% jitter
    return round(base + jitter, 3)


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Extract Retry-After header value in seconds, if present."""
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


async def _execute_with_retry(
    coro_factory,          # callable() -> coroutine returning httpx.Response
    max_retries: int,
    backoff_factor: float,
    operation_name: str,
) -> httpx.Response:
    """
    Execute an async coroutine factory with exponential-backoff retry.

    Retries on:
      - httpx.TimeoutException
      - httpx.NetworkError
      - HTTP 429 (rate limited by server)
      - HTTP 500, 502, 503, 504 (transient server errors)

    Does NOT retry on:
      - HTTP 4xx (except 429)

    Note: Alpha Vantage soft rate-limit notices (HTTP 200 + JSON 'Note') are
    handled by the caller (_get), not here, because they require body parsing.
    """
    retryable_status_codes = {429, 500, 502, 503, 504}
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):  # attempt 0 is the initial try
        try:
            response: httpx.Response = await coro_factory()

            if response.status_code == 429:
                retry_after = _parse_retry_after(response)
                if attempt == max_retries:
                    raise AlphaVantageRateLimitError(retry_after=retry_after)
                wait = retry_after if retry_after else _backoff_delay(
                    attempt, backoff_factor
                )
                logger.warning(
                    "[%s] HTTP 429 received. Waiting %.2fs before retry %d/%d.",
                    operation_name, wait, attempt + 1, max_retries,
                )
                await asyncio.sleep(wait)
                continue

            if response.status_code in retryable_status_codes:
                if attempt == max_retries:
                    response.raise_for_status()
                wait = _backoff_delay(attempt, backoff_factor)
                logger.warning(
                    "[%s] HTTP %d received. Waiting %.2fs before retry %d/%d.",
                    operation_name, response.status_code, wait,
                    attempt + 1, max_retries,
                )
                await asyncio.sleep(wait)
                last_exc = httpx.HTTPStatusError(
                    message=f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
                continue

            # Non-retryable 4xx
            if 400 <= response.status_code < 500:
                raise AlphaVantageHTTPError(
                    status_code=response.status_code,
                    message=response.text[:500],
                )

            return response

        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt == max_retries:
                raise AlphaVantageTimeoutError(
                    f"[{operation_name}] Request failed after {max_retries} retries: {exc}"
                ) from exc
            wait = _backoff_delay(attempt, backoff_factor)
            logger.warning(
                "[%s] Network/timeout error: %s. Waiting %.2fs before retry %d/%d.",
                operation_name, exc, wait, attempt + 1, max_retries,
            )
            await asyncio.sleep(wait)

    raise AlphaVantageMaxRetriesError(
        f"[{operation_name}] Exhausted {max_retries} retries."
    ) from last_exc


# ---------------------------------------------------------------------------
# Alpha Vantage soft-error detection
# ---------------------------------------------------------------------------

def _inspect_payload_for_soft_errors(payload: dict[str, Any]) -> None:
    """
    Alpha Vantage returns HTTP 200 even for errors and rate limits, embedding
    the condition in the JSON body. Inspect and raise the appropriate typed
    exception so the retry/handling logic can react correctly.

    - 'Error Message'  -> invalid symbol / bad params (non-retryable)
    - 'Note'           -> rate limit reached (retryable)
    - 'Information'    -> rate limit / premium-endpoint notice (retryable)
    """
    if not isinstance(payload, dict):
        return

    error_message = payload.get("Error Message")
    if error_message:
        raise AlphaVantageAPIError(str(error_message))

    note = payload.get("Note")
    if note:
        raise AlphaVantageRateLimitError(detail=str(note))

    information = payload.get("Information")
    if information:
        raise AlphaVantageRateLimitError(detail=str(information))


# ---------------------------------------------------------------------------
# Alpha Vantage async client
# ---------------------------------------------------------------------------

class AlphaVantageClient:
    """
    Production-grade async Alpha Vantage API client.

    Usage (preferred -- async context manager):

        async with AlphaVantageClient() as client:
            result = await client.fetch_global_quote("AAPL")

    Usage (manual lifecycle):

        client = AlphaVantageClient()
        await client.open()
        try:
            result = await client.fetch_daily("AAPL")
        finally:
            await client.close()
    """

    def __init__(self, settings: AlphaVantageSettings | None = None) -> None:
        self._cfg: AlphaVantageSettings = settings or get_settings().alphavantage
        self._base_url: str = str(self._cfg.base_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        # Alpha Vantage free tier: 5 requests per 60 seconds.
        self._rate_limiter = AsyncTokenBucket(
            capacity=self._cfg.rate_limit_calls,
            period=float(self._cfg.rate_limit_period),
        )
        logger.info(
            "AlphaVantageClient initialised | base_url=%s rate=%d/%ds",
            self._base_url,
            self._cfg.rate_limit_calls,
            self._cfg.rate_limit_period,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Create and configure the underlying httpx.AsyncClient."""
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": "MarketPredictor/1.0 (+https://github.com/your-org/market-predictor)",
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(self._cfg.request_timeout),
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )
        logger.debug("AlphaVantageClient: httpx.AsyncClient opened.")

    async def close(self) -> None:
        """Gracefully close the underlying httpx.AsyncClient."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("AlphaVantageClient: httpx.AsyncClient closed.")

    async def __aenter__(self) -> "AlphaVantageClient":
        await self.open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal request dispatcher
    # ------------------------------------------------------------------

    async def _get(
        self,
        params: dict[str, Any],
        operation_name: str,
    ) -> dict[str, Any]:
        """
        Rate-limited, retrying GET request against the single Alpha Vantage
        query endpoint.

        1. Injects the API key.
        2. Acquires a token from the bucket (blocks if needed).
        3. Delegates transport-level retries to _execute_with_retry.
        4. Parses JSON and inspects the body for Alpha Vantage soft errors,
           applying the same jittered backoff retry loop on soft rate limits.
        5. Returns the parsed JSON dict on success.
        """
        if self._client is None:
            raise RuntimeError(
                "AlphaVantageClient is not open. "
                "Use 'async with AlphaVantageClient()' or call await client.open() first."
            )

        request_params = {**params, **self._cfg.auth_params}
        # Avoid leaking the API key in logs.
        log_params = {k: v for k, v in params.items()}

        max_retries = self._cfg.max_retries
        backoff_factor = self._cfg.retry_backoff_factor
        last_soft_error: AlphaVantageRateLimitError | None = None

        for attempt in range(max_retries + 1):
            await self._rate_limiter.acquire()

            logger.debug("[%s] GET /query params=%s", operation_name, log_params)
            start = time.monotonic()

            async def _coro() -> httpx.Response:
                return await self._client.get("", params=request_params)  # type: ignore[union-attr]

            response = await _execute_with_retry(
                coro_factory=_coro,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                operation_name=operation_name,
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                "[%s] GET /query -> HTTP %d (%.1fms)",
                operation_name, response.status_code, elapsed_ms,
            )

            try:
                payload: dict[str, Any] = response.json()
            except ValueError as exc:
                raise AlphaVantageError(
                    f"[{operation_name}] Response was not valid JSON: {exc}"
                ) from exc

            try:
                _inspect_payload_for_soft_errors(payload)
            except AlphaVantageRateLimitError as soft_exc:
                # Soft rate limit (HTTP 200 + Note/Information): retry with backoff.
                last_soft_error = soft_exc
                if attempt == max_retries:
                    raise
                wait = _backoff_delay(attempt, backoff_factor)
                logger.warning(
                    "[%s] Soft rate-limit notice. Waiting %.2fs before retry %d/%d. detail=%s",
                    operation_name, wait, attempt + 1, max_retries, soft_exc.detail,
                )
                await asyncio.sleep(wait)
                continue

            return payload

        # Should be unreachable, but guards against logic drift.
        if last_soft_error is not None:
            raise last_soft_error
        raise AlphaVantageMaxRetriesError(
            f"[{operation_name}] Exhausted {max_retries} retries."
        )

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def fetch_global_quote(self, symbol: str) -> FetchResult:
        """
        Fetch the latest price and volume snapshot for a symbol.
        Function: GLOBAL_QUOTE
        """
        start = time.monotonic()
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol}
        try:
            payload = await self._get(params, operation_name=f"global_quote:{symbol}")
            quote_raw = payload.get("Global Quote") or payload.get("Global Quote - DATA") or {}
            if not quote_raw:
                return FetchResult(
                    success=False,
                    error=f"No quote data returned for symbol '{symbol}'",
                )
            parsed = _parse_global_quote(quote_raw, symbol)
            logger.info("fetch_global_quote: retrieved quote for %s", symbol)
            return FetchResult(
                success=True,
                data=parsed,
                status_code=200,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except AlphaVantageError as exc:
            logger.error("fetch_global_quote failed for %s: %s", symbol, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_intraday(
        self,
        symbol: str,
        interval: str = "5min",
        output_size: str | None = None,
        adjusted: bool = True,
        extended_hours: bool = True,
    ) -> FetchResult:
        """
        Fetch intraday OHLCV bars for a symbol.
        Function: TIME_SERIES_INTRADAY

        Args:
            symbol: Equity ticker (e.g. "AAPL").
            interval: One of 1min, 5min, 15min, 30min, 60min.
            output_size: 'compact' (latest 100) or 'full'. Defaults to config.
            adjusted: Whether to adjust for splits/dividends.
            extended_hours: Include pre/post market data.

        Returns:
            FetchResult with data as list[StockBar].
        """
        start = time.monotonic()
        allowed_intervals = {"1min", "5min", "15min", "30min", "60min"}
        if interval not in allowed_intervals:
            return FetchResult(
                success=False,
                error=f"interval must be one of {sorted(allowed_intervals)}, got '{interval}'",
            )
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": output_size or self._cfg.output_size,
            "adjusted": str(adjusted).lower(),
            "extended_hours": str(extended_hours).lower(),
            "datatype": "json",
        }
        try:
            payload = await self._get(params, operation_name=f"intraday:{symbol}")
            series_key = f"Time Series ({interval})"
            series_raw: dict[str, Any] = payload.get(series_key, {})
            bars = _parse_time_series(series_raw)
            logger.info(
                "fetch_intraday: %d bars for %s interval=%s", len(bars), symbol, interval
            )
            return FetchResult(
                success=True,
                data=bars,
                status_code=200,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except AlphaVantageError as exc:
            logger.error("fetch_intraday failed for %s: %s", symbol, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_daily(
        self,
        symbol: str,
        output_size: str | None = None,
    ) -> FetchResult:
        """
        Fetch daily OHLCV bars for a symbol.
        Function: TIME_SERIES_DAILY

        Returns:
            FetchResult with data as list[StockBar].
        """
        start = time.monotonic()
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": output_size or self._cfg.output_size,
            "datatype": "json",
        }
        try:
            payload = await self._get(params, operation_name=f"daily:{symbol}")
            series_raw: dict[str, Any] = payload.get("Time Series (Daily)", {})
            bars = _parse_time_series(series_raw)
            logger.info("fetch_daily: %d bars for %s", len(bars), symbol)
            return FetchResult(
                success=True,
                data=bars,
                status_code=200,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except AlphaVantageError as exc:
            logger.error("fetch_daily failed for %s: %s", symbol, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_daily_adjusted(
        self,
        symbol: str,
        output_size: str | None = None,
    ) -> FetchResult:
        """
        Fetch daily adjusted OHLCV bars including dividends and split coefficients.
        Function: TIME_SERIES_DAILY_ADJUSTED (premium endpoint)

        Returns:
            FetchResult with data as list[DailyAdjustedBar].
        """
        start = time.monotonic()
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": output_size or self._cfg.output_size,
            "datatype": "json",
        }
        try:
            payload = await self._get(params, operation_name=f"daily_adjusted:{symbol}")
            series_raw: dict[str, Any] = payload.get("Time Series (Daily)", {})
            bars = _parse_daily_adjusted_series(series_raw)
            logger.info("fetch_daily_adjusted: %d bars for %s", len(bars), symbol)
            return FetchResult(
                success=True,
                data=bars,
                status_code=200,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except AlphaVantageError as exc:
            logger.error("fetch_daily_adjusted failed for %s: %s", symbol, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_weekly(self, symbol: str) -> FetchResult:
        """
        Fetch weekly OHLCV bars for a symbol.
        Function: TIME_SERIES_WEEKLY

        Returns:
            FetchResult with data as list[StockBar].
        """
        start = time.monotonic()
        params = {
            "function": "TIME_SERIES_WEEKLY",
            "symbol": symbol,
            "datatype": "json",
        }
        try:
            payload = await self._get(params, operation_name=f"weekly:{symbol}")
            series_raw: dict[str, Any] = payload.get("Weekly Time Series", {})
            bars = _parse_time_series(series_raw)
            logger.info("fetch_weekly: %d bars for %s", len(bars), symbol)
            return FetchResult(
                success=True,
                data=bars,
                status_code=200,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except AlphaVantageError as exc:
            logger.error("fetch_weekly failed for %s: %s", symbol, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_monthly(self, symbol: str) -> FetchResult:
        """
        Fetch monthly OHLCV bars for a symbol.
        Function: TIME_SERIES_MONTHLY

        Returns:
            FetchResult with data as list[StockBar].
        """
        start = time.monotonic()
        params = {
            "function": "TIME_SERIES_MONTHLY",
            "symbol": symbol,
            "datatype": "json",
        }
        try:
            payload = await self._get(params, operation_name=f"monthly:{symbol}")
            series_raw: dict[str, Any] = payload.get("Monthly Time Series", {})
            bars = _parse_time_series(series_raw)
            logger.info("fetch_monthly: %d bars for %s", len(bars), symbol)
            return FetchResult(
                success=True,
                data=bars,
                status_code=200,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except AlphaVantageError as exc:
            logger.error("fetch_monthly failed for %s: %s", symbol, exc)
            return FetchResult(success=False, error=str(exc))

    async def search_symbol(self, keywords: str) -> FetchResult:
        """
        Search for symbols matching a keyword string.
        Function: SYMBOL_SEARCH

        Returns:
            FetchResult with data as list[SymbolMatch].
        """
        start = time.monotonic()
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
            "datatype": "json",
        }
        try:
            payload = await self._get(params, operation_name=f"symbol_search:{keywords}")
            matches_raw: list[dict[str, Any]] = payload.get("bestMatches", [])
            parsed = [_parse_symbol_match(item) for item in matches_raw]
            logger.info("search_symbol: %d matches for '%s'", len(parsed), keywords)
            return FetchResult(
                success=True,
                data=parsed,
                status_code=200,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except AlphaVantageError as exc:
            logger.error("search_symbol failed for '%s': %s", keywords, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_company_overview(self, symbol: str) -> FetchResult:
        """
        Fetch fundamental company data for a symbol.
        Function: OVERVIEW

        Returns:
            FetchResult with data as CompanyOverview.
        """
        start = time.monotonic()
        params = {"function": "OVERVIEW", "symbol": symbol}
        try:
            payload = await self._get(params, operation_name=f"overview:{symbol}")
            if not payload or not payload.get("Symbol"):
                return FetchResult(
                    success=False,
                    error=f"No overview data returned for symbol '{symbol}'",
                )
            parsed = _parse_company_overview(payload)
            logger.info("fetch_company_overview: retrieved overview for %s", symbol)
            return FetchResult(
                success=True,
                data=parsed,
                status_code=200,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except AlphaVantageError as exc:
            logger.error("fetch_company_overview failed for %s: %s", symbol, exc)
            return FetchResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    async def fetch_batch_daily(
        self,
        symbols: list[str],
        output_size: str | None = None,
        concurrency: int = 2,
    ) -> dict[str, FetchResult]:
        """
        Fetch daily bars for multiple symbols concurrently.

        The shared AsyncTokenBucket enforces the 5/min ceiling, so concurrency
        here only controls how many requests queue against the limiter at once.
        A low concurrency (2) is recommended on the free tier to keep latency
        predictable.

        Returns:
            Dict mapping symbol -> FetchResult.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch_one(symbol: str) -> tuple[str, FetchResult]:
            async with semaphore:
                result = await self.fetch_daily(symbol, output_size=output_size)
                return symbol, result

        tasks = [asyncio.create_task(_fetch_one(s)) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return dict(results)

    async def fetch_batch_global_quote(
        self,
        symbols: list[str],
        concurrency: int = 2,
    ) -> dict[str, FetchResult]:
        """
        Fetch global quotes for multiple symbols concurrently.

        Returns:
            Dict mapping symbol -> FetchResult.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch_one(symbol: str) -> tuple[str, FetchResult]:
            async with semaphore:
                result = await self.fetch_global_quote(symbol)
                return symbol, result

        tasks = [asyncio.create_task(_fetch_one(s)) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return dict(results)


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None if not possible."""
    if value is None or value == "None" or value == "-":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    """Safely convert a value to int, returning 0 if not possible."""
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _parse_timestamp(key: str) -> datetime:
    """
    Parse an Alpha Vantage time-series key into a UTC-aware datetime.
    Daily/weekly/monthly keys are 'YYYY-MM-DD'; intraday keys are
    'YYYY-MM-DD HH:MM:SS'.
    """
    fmt = "%Y-%m-%d %H:%M:%S" if " " in key else "%Y-%m-%d"
    try:
        naive = datetime.strptime(key, fmt)
        return naive.replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("Could not parse time-series key: %r", key)
        return datetime.now(timezone.utc)


def _parse_time_series(series: dict[str, Any]) -> list[StockBar]:
    """
    Map a standard (non-adjusted) Alpha Vantage time series dict to StockBar list,
    sorted ascending by timestamp.
    """
    bars: list[StockBar] = []
    for key, values in series.items():
        bars.append(
            StockBar(
                timestamp=_parse_timestamp(key),
                open=_safe_float(values.get("1. open")) or 0.0,
                high=_safe_float(values.get("2. high")) or 0.0,
                low=_safe_float(values.get("3. low")) or 0.0,
                close=_safe_float(values.get("4. close")) or 0.0,
                volume=_safe_int(values.get("5. volume")),
            )
        )
    bars.sort(key=lambda b: b.timestamp)
    return bars


def _parse_daily_adjusted_series(series: dict[str, Any]) -> list[DailyAdjustedBar]:
    """
    Map a TIME_SERIES_DAILY_ADJUSTED series dict to DailyAdjustedBar list,
    sorted ascending by timestamp.
    """
    bars: list[DailyAdjustedBar] = []
    for key, values in series.items():
        bars.append(
            DailyAdjustedBar(
                timestamp=_parse_timestamp(key),
                open=_safe_float(values.get("1. open")) or 0.0,
                high=_safe_float(values.get("2. high")) or 0.0,
                low=_safe_float(values.get("3. low")) or 0.0,
                close=_safe_float(values.get("4. close")) or 0.0,
                adjusted_close=_safe_float(values.get("5. adjusted close")) or 0.0,
                volume=_safe_int(values.get("6. volume")),
                dividend_amount=_safe_float(values.get("7. dividend amount")) or 0.0,
                split_coefficient=_safe_float(values.get("8. split coefficient")) or 1.0,
            )
        )
    bars.sort(key=lambda b: b.timestamp)
    return bars


def _parse_global_quote(raw: dict[str, Any], fallback_symbol: str) -> GlobalQuote:
    """Map a raw 'Global Quote' dict to a typed GlobalQuote dataclass."""
    change_pct_raw = raw.get("10. change percent", "0")
    change_pct = _safe_float(str(change_pct_raw).rstrip("%")) or 0.0

    trading_day_raw = raw.get("07. latest trading day", "")
    try:
        trading_day = datetime.strptime(trading_day_raw, "%Y-%m-%d").date()
    except ValueError:
        trading_day = datetime.now(timezone.utc).date()

    return GlobalQuote(
        symbol=raw.get("01. symbol", fallback_symbol),
        open=_safe_float(raw.get("02. open")) or 0.0,
        high=_safe_float(raw.get("03. high")) or 0.0,
        low=_safe_float(raw.get("04. low")) or 0.0,
        price=_safe_float(raw.get("05. price")) or 0.0,
        volume=_safe_int(raw.get("06. volume")),
        latest_trading_day=trading_day,
        previous_close=_safe_float(raw.get("08. previous close")) or 0.0,
        change=_safe_float(raw.get("09. change")) or 0.0,
        change_percent=change_pct,
    )


def _parse_symbol_match(raw: dict[str, Any]) -> SymbolMatch:
    """Map a raw 'bestMatches' entry to a typed SymbolMatch dataclass."""
    return SymbolMatch(
        symbol=raw.get("1. symbol", ""),
        name=raw.get("2. name", ""),
        type=raw.get("3. type", ""),
        region=raw.get("4. region", ""),
        currency=raw.get("8. currency", ""),
        match_score=_safe_float(raw.get("9. matchScore")) or 0.0,
    )


def _parse_company_overview(raw: dict[str, Any]) -> CompanyOverview:
    """Map a raw OVERVIEW response to a typed CompanyOverview dataclass."""
    return CompanyOverview(
        symbol=raw.get("Symbol", ""),
        name=raw.get("Name", ""),
        description=raw.get("Description", ""),
        exchange=raw.get("Exchange", ""),
        currency=raw.get("Currency", ""),
        sector=raw.get("Sector", ""),
        industry=raw.get("Industry", ""),
        market_capitalization=_safe_float(raw.get("MarketCapitalization")),
        pe_ratio=_safe_float(raw.get("PERatio")),
        peg_ratio=_safe_float(raw.get("PEGRatio")),
        dividend_yield=_safe_float(raw.get("DividendYield")),
        eps=_safe_float(raw.get("EPS")),
        beta=_safe_float(raw.get("Beta")),
        week_52_high=_safe_float(raw.get("52WeekHigh")),
        week_52_low=_safe_float(raw.get("52WeekLow")),
    )


# ---------------------------------------------------------------------------
# Convenience top-level functions (used by the scheduler)
# ---------------------------------------------------------------------------

# Default watchlist -- override via scheduler job arguments
DEFAULT_STOCK_WATCHLIST: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "V",
    "WMT",
]


async def fetch_watchlist_quotes(
    symbols: list[str] | None = None,
    concurrency: int = 2,
) -> dict[str, FetchResult]:
    """
    Convenience function: fetch latest global quotes for the default watchlist.
    Intended to be called directly by the APScheduler job.

    Returns:
        Dict mapping symbol -> FetchResult.
    """
    tickers = symbols or DEFAULT_STOCK_WATCHLIST
    async with AlphaVantageClient() as client:
        return await client.fetch_batch_global_quote(tickers, concurrency=concurrency)


async def fetch_watchlist_daily(
    symbols: list[str] | None = None,
    output_size: str | None = None,
    concurrency: int = 2,
) -> dict[str, FetchResult]:
    """
    Convenience function: fetch daily OHLCV bars for the default watchlist.
    Intended to be called directly by the APScheduler job.

    Returns:
        Dict mapping symbol -> FetchResult.
    """
    tickers = symbols or DEFAULT_STOCK_WATCHLIST
    async with AlphaVantageClient() as client:
        return await client.fetch_batch_daily(
            tickers, output_size=output_size, concurrency=concurrency
        )


async def fetch_single_daily(
    symbol: str,
    output_size: str | None = None,
) -> FetchResult:
    """
    Convenience function: fetch daily OHLCV bars for a single symbol.

    Returns:
        FetchResult with data as list[StockBar].
    """
    async with AlphaVantageClient() as client:
        return await client.fetch_daily(symbol, output_size=output_size)
