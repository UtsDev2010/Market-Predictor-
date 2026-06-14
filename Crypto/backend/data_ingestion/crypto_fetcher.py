"""
Production-grade asynchronous CoinGecko API client.

Features:
  - httpx.AsyncClient with connection pooling and configurable timeouts
  - Token-bucket rate limiter (asyncio-native, no third-party rate-limit lib)
  - Exponential-backoff retry with jitter for transient errors and HTTP 429
  - Structured logging on every request, retry, and failure
  - Typed dataclasses for all returned payloads
  - Async context-manager lifecycle (open / close)
  - Convenience top-level fetch functions for the scheduler
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.config.settings import CoinGeckoSettings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class CoinGeckoError(Exception):
    """Base exception for all CoinGecko client errors."""


class CoinGeckoRateLimitError(CoinGeckoError):
    """Raised when the API returns HTTP 429 and all retries are exhausted."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(
            f"CoinGecko rate limit exceeded. "
            f"Retry after: {retry_after}s" if retry_after else "CoinGecko rate limit exceeded."
        )


class CoinGeckoHTTPError(CoinGeckoError):
    """Raised for non-retryable HTTP errors (4xx except 429)."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class CoinGeckoTimeoutError(CoinGeckoError):
    """Raised when a request times out after all retries."""


class CoinGeckoMaxRetriesError(CoinGeckoError):
    """Raised when max retry attempts are exhausted."""


# ---------------------------------------------------------------------------
# Typed response dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OHLCVCandle:
    """A single OHLCV candle returned by the CoinGecko OHLC endpoint."""

    timestamp: datetime          # UTC
    open: float
    high: float
    low: float
    close: float
    # CoinGecko OHLC does not include volume per candle; volume comes separately


@dataclass(frozen=True, slots=True)
class MarketData:
    """Snapshot of a coin's market data from /coins/markets."""

    coin_id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int | None
    total_volume: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_pct_24h: float
    circulating_supply: float
    total_supply: float | None
    max_supply: float | None
    ath: float
    ath_change_pct: float
    atl: float
    atl_change_pct: float
    last_updated: datetime       # UTC


@dataclass(frozen=True, slots=True)
class CoinInfo:
    """Detailed coin metadata from /coins/{id}."""

    coin_id: str
    symbol: str
    name: str
    description: str
    homepage: str
    genesis_date: str | None
    sentiment_votes_up_pct: float | None
    sentiment_votes_down_pct: float | None
    coingecko_score: float | None
    developer_score: float | None
    community_score: float | None
    liquidity_score: float | None
    public_interest_score: float | None


@dataclass(frozen=True, slots=True)
class HistoricalPrice:
    """A single price point from /coins/{id}/market_chart."""

    timestamp: datetime          # UTC
    price: float
    market_cap: float
    total_volume: float


@dataclass(frozen=True, slots=True)
class TrendingCoin:
    """A single entry from /search/trending."""

    coin_id: str
    symbol: str
    name: str
    market_cap_rank: int | None
    score: int                   # trending rank (0 = most trending)


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

    Allows `capacity` tokens per `period` seconds.  Each call to
    ``acquire()`` consumes one token, blocking until one is available.
    This ensures we never exceed the CoinGecko Pro rate limit regardless
    of how many concurrent coroutines are running.
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
        """
        Block until ``tokens`` tokens are available, then consume them.
        """
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
                # Calculate how long to wait for enough tokens
                deficit = tokens - self._tokens
                wait_time = deficit / self.refill_rate

            # Release lock while sleeping so other coroutines can check
            logger.debug(
                "Rate limiter: waiting %.3fs for token availability", wait_time
            )
            await asyncio.sleep(wait_time)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _execute_with_retry(
    coro_factory,          # callable() -> coroutine
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
      - HTTP 4xx (except 429) — client errors are not transient
      - httpx.HTTPStatusError for non-retryable codes

    Jitter is added to each backoff delay to avoid thundering-herd.
    """
    import random

    retryable_status_codes = {429, 500, 502, 503, 504}
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):  # attempt 0 is the initial try
        try:
            response: httpx.Response = await coro_factory()

            if response.status_code == 429:
                retry_after = _parse_retry_after(response)
                if attempt == max_retries:
                    raise CoinGeckoRateLimitError(retry_after=retry_after)
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
                raise CoinGeckoHTTPError(
                    status_code=response.status_code,
                    message=response.text[:500],
                )

            return response

        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt == max_retries:
                raise CoinGeckoTimeoutError(
                    f"[{operation_name}] Request failed after {max_retries} retries: {exc}"
                ) from exc
            wait = _backoff_delay(attempt, backoff_factor)
            logger.warning(
                "[%s] Network/timeout error: %s. Waiting %.2fs before retry %d/%d.",
                operation_name, exc, wait, attempt + 1, max_retries,
            )
            await asyncio.sleep(wait)

    raise CoinGeckoMaxRetriesError(
        f"[{operation_name}] Exhausted {max_retries} retries."
    ) from last_exc


def _backoff_delay(attempt: int, factor: float) -> float:
    """Compute jittered exponential backoff: factor * 2^attempt + random jitter."""
    import random
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


# ---------------------------------------------------------------------------
# CoinGecko async client
# ---------------------------------------------------------------------------

class CoinGeckoClient:
    """
    Production-grade async CoinGecko Pro API client.

    Usage (preferred — async context manager):

        async with CoinGeckoClient() as client:
            result = await client.fetch_market_data(["bitcoin", "ethereum"])

    Usage (manual lifecycle):

        client = CoinGeckoClient()
        await client.open()
        try:
            result = await client.fetch_ohlcv("bitcoin", days=30)
        finally:
            await client.close()
    """

    # CoinGecko Pro supported vs_currencies
    DEFAULT_VS_CURRENCY = "usd"

    def __init__(self, settings: CoinGeckoSettings | None = None) -> None:
        self._cfg: CoinGeckoSettings = settings or get_settings().coingecko
        self._base_url: str = str(self._cfg.base_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = AsyncTokenBucket(
            capacity=self._cfg.rate_limit_calls,
            period=float(self._cfg.rate_limit_period),
        )
        logger.info(
            "CoinGeckoClient initialised | base_url=%s rate=%d/%ds",
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
                **self._cfg.auth_header,
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
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )
        logger.debug("CoinGeckoClient: httpx.AsyncClient opened.")

    async def close(self) -> None:
        """Gracefully close the underlying httpx.AsyncClient."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("CoinGeckoClient: httpx.AsyncClient closed.")

    async def __aenter__(self) -> "CoinGeckoClient":
        await self.open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal request dispatcher
    # ------------------------------------------------------------------

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        operation_name: str = "GET",
    ) -> httpx.Response:
        """
        Rate-limited, retrying GET request.

        1. Acquires a token from the bucket (blocks if needed).
        2. Delegates to _execute_with_retry for backoff logic.
        3. Returns the raw httpx.Response on success.
        """
        if self._client is None:
            raise RuntimeError(
                "CoinGeckoClient is not open. "
                "Use 'async with CoinGeckoClient()' or call await client.open() first."
            )

        await self._rate_limiter.acquire()

        logger.debug(
            "[%s] GET %s params=%s",
            operation_name, endpoint, params,
        )

        start = time.monotonic()

        async def _coro() -> httpx.Response:
            return await self._client.get(endpoint, params=params)  # type: ignore[union-attr]

        response = await _execute_with_retry(
            coro_factory=_coro,
            max_retries=self._cfg.max_retries,
            backoff_factor=self._cfg.retry_backoff_factor,
            operation_name=operation_name,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "[%s] %s %s -> HTTP %d (%.1fms)",
            operation_name, "GET", endpoint,
            response.status_code, elapsed_ms,
        )
        return response

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def ping(self) -> FetchResult:
        """
        Health-check the CoinGecko API.
        Endpoint: GET /ping
        """
        start = time.monotonic()
        try:
            response = await self._get("/ping", operation_name="ping")
            data = response.json()
            return FetchResult(
                success=True,
                data=data,
                status_code=response.status_code,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except CoinGeckoError as exc:
            logger.error("ping failed: %s", exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_market_data(
        self,
        coin_ids: list[str],
        vs_currency: str = DEFAULT_VS_CURRENCY,
        per_page: int = 250,
        page: int = 1,
        sparkline: bool = False,
        price_change_percentage: str = "24h,7d,30d",
    ) -> FetchResult:
        """
        Fetch market data for a list of coins.
        Endpoint: GET /coins/markets

        Args:
            coin_ids: List of CoinGecko coin IDs (e.g. ["bitcoin", "ethereum"]).
            vs_currency: Target currency (default "usd").
            per_page: Results per page (max 250).
            page: Page number.
            sparkline: Include 7-day sparkline data.
            price_change_percentage: Comma-separated intervals for price change.

        Returns:
            FetchResult with data as list[MarketData].
        """
        start = time.monotonic()
        params: dict[str, Any] = {
            "vs_currency": vs_currency,
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": min(per_page, 250),
            "page": page,
            "sparkline": str(sparkline).lower(),
            "price_change_percentage": price_change_percentage,
            "locale": "en",
        }
        try:
            response = await self._get(
                "/coins/markets", params=params, operation_name="fetch_market_data"
            )
            raw: list[dict[str, Any]] = response.json()
            parsed = [_parse_market_data(item) for item in raw]
            logger.info(
                "fetch_market_data: retrieved %d coins for ids=%s",
                len(parsed), coin_ids,
            )
            return FetchResult(
                success=True,
                data=parsed,
                status_code=response.status_code,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except CoinGeckoError as exc:
            logger.error("fetch_market_data failed for ids=%s: %s", coin_ids, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_ohlcv(
        self,
        coin_id: str,
        vs_currency: str = DEFAULT_VS_CURRENCY,
        days: int = 30,
    ) -> FetchResult:
        """
        Fetch OHLC candlestick data for a coin.
        Endpoint: GET /coins/{id}/ohlc

        Granularity is determined automatically by CoinGecko:
          - 1-2 days   -> 30-minute candles
          - 3-30 days  -> 4-hour candles
          - 31-90 days -> 4-hour candles
          - 91+ days   -> 4-day candles

        Args:
            coin_id: CoinGecko coin ID (e.g. "bitcoin").
            vs_currency: Target currency (default "usd").
            days: Number of days of data (1, 7, 14, 30, 90, 180, 365, max).

        Returns:
            FetchResult with data as list[OHLCVCandle].
        """
        start = time.monotonic()
        params: dict[str, Any] = {
            "vs_currency": vs_currency,
            "days": days,
        }
        try:
            response = await self._get(
                f"/coins/{coin_id}/ohlc",
                params=params,
                operation_name=f"fetch_ohlcv:{coin_id}",
            )
            raw: list[list[float]] = response.json()
            candles = [_parse_ohlcv_candle(row) for row in raw]
            logger.info(
                "fetch_ohlcv: %d candles for coin=%s days=%d",
                len(candles), coin_id, days,
            )
            return FetchResult(
                success=True,
                data=candles,
                status_code=response.status_code,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except CoinGeckoError as exc:
            logger.error("fetch_ohlcv failed for coin=%s: %s", coin_id, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_historical_market_chart(
        self,
        coin_id: str,
        vs_currency: str = DEFAULT_VS_CURRENCY,
        days: int | str = 30,
        interval: str | None = None,
    ) -> FetchResult:
        """
        Fetch historical price, market cap, and volume data.
        Endpoint: GET /coins/{id}/market_chart

        Args:
            coin_id: CoinGecko coin ID.
            vs_currency: Target currency.
            days: Number of days back, or "max" for full history.
            interval: Data interval override ("daily" forces daily granularity).

        Returns:
            FetchResult with data as list[HistoricalPrice].
        """
        start = time.monotonic()
        params: dict[str, Any] = {
            "vs_currency": vs_currency,
            "days": days,
        }
        if interval:
            params["interval"] = interval

        try:
            response = await self._get(
                f"/coins/{coin_id}/market_chart",
                params=params,
                operation_name=f"fetch_market_chart:{coin_id}",
            )
            raw: dict[str, list[list[float]]] = response.json()
            prices = raw.get("prices", [])
            market_caps = raw.get("market_caps", [])
            volumes = raw.get("total_volumes", [])

            # Zip all three series by index (they are always the same length)
            parsed: list[HistoricalPrice] = []
            for i, price_point in enumerate(prices):
                ts_ms, price = price_point
                _, mkt_cap = market_caps[i] if i < len(market_caps) else (ts_ms, 0.0)
                _, volume = volumes[i] if i < len(volumes) else (ts_ms, 0.0)
                parsed.append(
                    HistoricalPrice(
                        timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                        price=float(price),
                        market_cap=float(mkt_cap),
                        total_volume=float(volume),
                    )
                )

            logger.info(
                "fetch_historical_market_chart: %d data points for coin=%s days=%s",
                len(parsed), coin_id, days,
            )
            return FetchResult(
                success=True,
                data=parsed,
                status_code=response.status_code,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except CoinGeckoError as exc:
            logger.error(
                "fetch_historical_market_chart failed for coin=%s: %s", coin_id, exc
            )
            return FetchResult(success=False, error=str(exc))

    async def fetch_coin_info(
        self,
        coin_id: str,
        localization: bool = False,
        tickers: bool = False,
        market_data: bool = False,
        community_data: bool = True,
        developer_data: bool = False,
    ) -> FetchResult:
        """
        Fetch detailed metadata for a single coin.
        Endpoint: GET /coins/{id}

        Args:
            coin_id: CoinGecko coin ID.
            localization: Include localized language data.
            tickers: Include exchange ticker data.
            market_data: Include full market data block.
            community_data: Include community stats.
            developer_data: Include GitHub/developer stats.

        Returns:
            FetchResult with data as CoinInfo.
        """
        start = time.monotonic()
        params: dict[str, Any] = {
            "localization": str(localization).lower(),
            "tickers": str(tickers).lower(),
            "market_data": str(market_data).lower(),
            "community_data": str(community_data).lower(),
            "developer_data": str(developer_data).lower(),
            "sparkline": "false",
        }
        try:
            response = await self._get(
                f"/coins/{coin_id}",
                params=params,
                operation_name=f"fetch_coin_info:{coin_id}",
            )
            raw: dict[str, Any] = response.json()
            parsed = _parse_coin_info(raw)
            logger.info("fetch_coin_info: retrieved metadata for coin=%s", coin_id)
            return FetchResult(
                success=True,
                data=parsed,
                status_code=response.status_code,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except CoinGeckoError as exc:
            logger.error("fetch_coin_info failed for coin=%s: %s", coin_id, exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_trending(
        self,
    ) -> FetchResult:
        """
        Fetch the top-7 trending coins on CoinGecko in the last 24 hours.
        Endpoint: GET /search/trending

        Returns:
            FetchResult with data as list[TrendingCoin].
        """
        start = time.monotonic()
        try:
            response = await self._get(
                "/search/trending", operation_name="fetch_trending"
            )
            raw: dict[str, Any] = response.json()
            coins_raw: list[dict[str, Any]] = raw.get("coins", [])
            parsed = [_parse_trending_coin(entry, rank) for rank, entry in enumerate(coins_raw)]
            logger.info("fetch_trending: retrieved %d trending coins", len(parsed))
            return FetchResult(
                success=True,
                data=parsed,
                status_code=response.status_code,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except CoinGeckoError as exc:
            logger.error("fetch_trending failed: %s", exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_global_market_data(self) -> FetchResult:
        """
        Fetch global cryptocurrency market statistics.
        Endpoint: GET /global

        Returns:
            FetchResult with data as raw dict (fields vary by API version).
        """
        start = time.monotonic()
        try:
            response = await self._get("/global", operation_name="fetch_global")
            raw: dict[str, Any] = response.json()
            data = raw.get("data", {})
            logger.info(
                "fetch_global_market_data: total_market_cap_usd=%.2f",
                data.get("total_market_cap", {}).get("usd", 0.0),
            )
            return FetchResult(
                success=True,
                data=data,
                status_code=response.status_code,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except CoinGeckoError as exc:
            logger.error("fetch_global_market_data failed: %s", exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_supported_coins(self) -> FetchResult:
        """
        Fetch the full list of supported coins with their IDs, symbols, and names.
        Endpoint: GET /coins/list

        Returns:
            FetchResult with data as list[dict] (id, symbol, name).
        """
        start = time.monotonic()
        try:
            response = await self._get(
                "/coins/list",
                params={"include_platform": "false"},
                operation_name="fetch_supported_coins",
            )
            data: list[dict[str, str]] = response.json()
            logger.info("fetch_supported_coins: %d coins available", len(data))
            return FetchResult(
                success=True,
                data=data,
                status_code=response.status_code,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except CoinGeckoError as exc:
            logger.error("fetch_supported_coins failed: %s", exc)
            return FetchResult(success=False, error=str(exc))

    async def fetch_batch_ohlcv(
        self,
        coin_ids: list[str],
        vs_currency: str = DEFAULT_VS_CURRENCY,
        days: int = 30,
        concurrency: int = 5,
    ) -> dict[str, FetchResult]:
        """
        Fetch OHLCV data for multiple coins concurrently, respecting the rate limiter.

        Args:
            coin_ids: List of CoinGecko coin IDs.
            vs_currency: Target currency.
            days: Number of days of OHLCV data.
            concurrency: Max simultaneous in-flight requests.

        Returns:
            Dict mapping coin_id -> FetchResult.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch_one(coin_id: str) -> tuple[str, FetchResult]:
            async with semaphore:
                result = await self.fetch_ohlcv(coin_id, vs_currency=vs_currency, days=days)
                return coin_id, result

        tasks = [asyncio.create_task(_fetch_one(cid)) for cid in coin_ids]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return dict(results)

    async def fetch_batch_market_chart(
        self,
        coin_ids: list[str],
        vs_currency: str = DEFAULT_VS_CURRENCY,
        days: int = 30,
        interval: str | None = None,
        concurrency: int = 5,
    ) -> dict[str, FetchResult]:
        """
        Fetch historical market chart data for multiple coins concurrently.

        Args:
            coin_ids: List of CoinGecko coin IDs.
            vs_currency: Target currency.
            days: Number of days of history.
            interval: Optional granularity override.
            concurrency: Max simultaneous in-flight requests.

        Returns:
            Dict mapping coin_id -> FetchResult.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch_one(coin_id: str) -> tuple[str, FetchResult]:
            async with semaphore:
                result = await self.fetch_historical_market_chart(
                    coin_id,
                    vs_currency=vs_currency,
                    days=days,
                    interval=interval,
                )
                return coin_id, result

        tasks = [asyncio.create_task(_fetch_one(cid)) for cid in coin_ids]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return dict(results)


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

def _parse_market_data(raw: dict[str, Any]) -> MarketData:
    """Map a raw /coins/markets item dict to a typed MarketData dataclass."""
    return MarketData(
        coin_id=raw["id"],
        symbol=raw["symbol"],
        name=raw["name"],
        current_price=float(raw.get("current_price") or 0.0),
        market_cap=float(raw.get("market_cap") or 0.0),
        market_cap_rank=raw.get("market_cap_rank"),
        total_volume=float(raw.get("total_volume") or 0.0),
        high_24h=float(raw.get("high_24h") or 0.0),
        low_24h=float(raw.get("low_24h") or 0.0),
        price_change_24h=float(raw.get("price_change_24h") or 0.0),
        price_change_pct_24h=float(raw.get("price_change_percentage_24h") or 0.0),
        circulating_supply=float(raw.get("circulating_supply") or 0.0),
        total_supply=float(raw["total_supply"]) if raw.get("total_supply") is not None else None,
        max_supply=float(raw["max_supply"]) if raw.get("max_supply") is not None else None,
        ath=float(raw.get("ath") or 0.0),
        ath_change_pct=float(raw.get("ath_change_percentage") or 0.0),
        atl=float(raw.get("atl") or 0.0),
        atl_change_pct=float(raw.get("atl_change_percentage") or 0.0),
        last_updated=_parse_iso_datetime(raw.get("last_updated", "")),
    )


def _parse_ohlcv_candle(row: list[float]) -> OHLCVCandle:
    """
    Map a raw OHLC row [timestamp_ms, open, high, low, close] to OHLCVCandle.
    CoinGecko does not include volume in the OHLC endpoint.
    """
    ts_ms, open_, high, low, close = row
    return OHLCVCandle(
        timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
        open=float(open_),
        high=float(high),
        low=float(low),
        close=float(close),
    )


def _parse_coin_info(raw: dict[str, Any]) -> CoinInfo:
    """Map a raw /coins/{id} response to a typed CoinInfo dataclass."""
    description_block = raw.get("description") or {}
    description = description_block.get("en", "") if isinstance(description_block, dict) else ""

    links = raw.get("links") or {}
    homepage_list = links.get("homepage") or []
    homepage = homepage_list[0] if homepage_list else ""

    scores = raw.get("scores") or {}

    return CoinInfo(
        coin_id=raw.get("id", ""),
        symbol=raw.get("symbol", ""),
        name=raw.get("name", ""),
        description=description,
        homepage=homepage,
        genesis_date=raw.get("genesis_date"),
        sentiment_votes_up_pct=_safe_float(raw.get("sentiment_votes_up_percentage")),
        sentiment_votes_down_pct=_safe_float(raw.get("sentiment_votes_down_percentage")),
        coingecko_score=_safe_float(raw.get("coingecko_score")),
        developer_score=_safe_float(raw.get("developer_score")),
        community_score=_safe_float(raw.get("community_score")),
        liquidity_score=_safe_float(raw.get("liquidity_score")),
        public_interest_score=_safe_float(raw.get("public_interest_score")),
    )


def _parse_trending_coin(entry: dict[str, Any], rank: int) -> TrendingCoin:
    """Map a raw trending coins entry to a typed TrendingCoin dataclass."""
    item = entry.get("item") or {}
    return TrendingCoin(
        coin_id=item.get("id", ""),
        symbol=item.get("symbol", ""),
        name=item.get("name", ""),
        market_cap_rank=item.get("market_cap_rank"),
        score=rank,
    )


def _parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO 8601 datetime string to a UTC-aware datetime."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        # Python 3.11+ handles Z suffix natively; for 3.10 compatibility replace it
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Could not parse datetime string: %r", value)
        return datetime.now(timezone.utc)


def _safe_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Convenience top-level functions (used by the scheduler)
# ---------------------------------------------------------------------------

# Default watchlist — override via scheduler job arguments
DEFAULT_CRYPTO_WATCHLIST: list[str] = [
    "bitcoin",
    "ethereum",
    "binancecoin",
    "solana",
    "ripple",
    "cardano",
    "avalanche-2",
    "polkadot",
    "chainlink",
    "uniswap",
]


async def fetch_watchlist_market_data(
    coin_ids: list[str] | None = None,
    vs_currency: str = "usd",
) -> FetchResult:
    """
    Convenience function: fetch market data for the default watchlist.
    Intended to be called directly by the APScheduler job.

    Args:
        coin_ids: Override the default watchlist.
        vs_currency: Target currency.

    Returns:
        FetchResult with data as list[MarketData].
    """
    ids = coin_ids or DEFAULT_CRYPTO_WATCHLIST
    async with CoinGeckoClient() as client:
        return await client.fetch_market_data(ids, vs_currency=vs_currency)


async def fetch_watchlist_ohlcv(
    coin_ids: list[str] | None = None,
    vs_currency: str = "usd",
    days: int = 30,
    concurrency: int = 5,
) -> dict[str, FetchResult]:
    """
    Convenience function: fetch OHLCV data for the default watchlist concurrently.
    Intended to be called directly by the APScheduler job.

    Args:
        coin_ids: Override the default watchlist.
        vs_currency: Target currency.
        days: Number of days of OHLCV history.
        concurrency: Max simultaneous requests.

    Returns:
        Dict mapping coin_id -> FetchResult.
    """
    ids = coin_ids or DEFAULT_CRYPTO_WATCHLIST
    async with CoinGeckoClient() as client:
        return await client.fetch_batch_ohlcv(
            ids, vs_currency=vs_currency, days=days, concurrency=concurrency
        )


async def fetch_watchlist_history(
    coin_ids: list[str] | None = None,
    vs_currency: str = "usd",
    days: int = 90,
    concurrency: int = 5,
) -> dict[str, FetchResult]:
    """
    Convenience function: fetch historical market chart data for the default watchlist.
    Intended to be called directly by the APScheduler job.

    Args:
        coin_ids: Override the default watchlist.
        vs_currency: Target currency.
        days: Number of days of history.
        concurrency: Max simultaneous requests.

    Returns:
        Dict mapping coin_id -> FetchResult.
    """
    ids = coin_ids or DEFAULT_CRYPTO_WATCHLIST
    async with CoinGeckoClient() as client:
        return await client.fetch_batch_market_chart(
            ids, vs_currency=vs_currency, days=days, concurrency=concurrency
        )
