"""
Production-grade application settings using Pydantic V2 BaseSettings.
All values are parsed and validated from environment variables or a .env file.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Sub-settings groups
# ---------------------------------------------------------------------------

class DatabaseSettings(BaseSettings):
    """TimescaleDB / PostgreSQL connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: SecretStr = Field(..., description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, description="Max pool overflow connections")
    pool_timeout: int = Field(default=30, ge=1, description="Pool connection timeout (seconds)")
    pool_recycle: int = Field(default=1800, ge=60, description="Recycle connections after N seconds")
    echo_sql: bool = Field(default=False, description="Echo raw SQL to stdout (dev only)")

    @property
    def async_dsn(self) -> str:
        """Async SQLAlchemy DSN using asyncpg driver."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def sync_dsn(self) -> str:
        """Sync SQLAlchemy DSN using psycopg2 driver (for Alembic migrations)."""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class CoinGeckoSettings(BaseSettings):
    """CoinGecko API settings."""

    model_config = SettingsConfigDict(
        env_prefix="COINGECKO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(..., description="CoinGecko Pro API key")
    base_url: AnyHttpUrl = Field(
        default="https://pro-api.coingecko.com/api/v3",
        description="CoinGecko base URL",
    )
    request_timeout: int = Field(default=30, ge=5, le=120, description="HTTP timeout (seconds)")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts on failure")
    retry_backoff_factor: float = Field(
        default=1.5, ge=0.1, le=10.0, description="Exponential backoff multiplier"
    )
    rate_limit_calls: int = Field(default=30, ge=1, description="Max API calls per rate window")
    rate_limit_period: int = Field(default=60, ge=1, description="Rate limit window in seconds")

    @property
    def auth_header(self) -> dict[str, str]:
        return {"x-cg-pro-api-key": self.api_key.get_secret_value()}


class AlphaVantageSettings(BaseSettings):
    """Alpha Vantage API settings."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHAVANTAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(..., description="Alpha Vantage API key")
    base_url: AnyHttpUrl = Field(
        default="https://www.alphavantage.co/query",
        description="Alpha Vantage base URL",
    )
    request_timeout: int = Field(default=30, ge=5, le=120, description="HTTP timeout (seconds)")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts on failure")
    retry_backoff_factor: float = Field(
        default=2.0, ge=0.1, le=10.0, description="Exponential backoff multiplier"
    )
    # Free tier: 5 calls/min, 500 calls/day
    rate_limit_calls: int = Field(default=5, ge=1, description="Max API calls per rate window")
    rate_limit_period: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    output_size: str = Field(
        default="compact",
        description="'compact' (100 data points) or 'full' (20+ years)",
    )

    @field_validator("output_size")
    @classmethod
    def validate_output_size(cls, v: str) -> str:
        allowed = {"compact", "full"}
        if v not in allowed:
            raise ValueError(f"output_size must be one of {allowed}, got '{v}'")
        return v

    @property
    def auth_params(self) -> dict[str, str]:
        return {"apikey": self.api_key.get_secret_value()}


class TelegramSettings(BaseSettings):
    """Telegram Bot alerting settings."""

    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: SecretStr = Field(..., description="Telegram Bot API token from @BotFather")
    chat_id: str = Field(..., description="Target chat/channel ID for alerts")
    alert_cooldown_seconds: int = Field(
        default=300, ge=0, description="Minimum seconds between duplicate alerts"
    )
    max_message_length: int = Field(
        default=4096, ge=1, le=4096, description="Telegram max message length"
    )
    parse_mode: str = Field(default="MarkdownV2", description="Telegram message parse mode")
    disable_web_page_preview: bool = Field(default=True)
    request_timeout: int = Field(default=30, ge=5, le=120, description="HTTP timeout (seconds)")

    @field_validator("parse_mode")
    @classmethod
    def validate_parse_mode(cls, v: str) -> str:
        allowed = {"MarkdownV2", "HTML", "Markdown"}
        if v not in allowed:
            raise ValueError(f"parse_mode must be one of {allowed}, got '{v}'")
        return v

    @property
    def bot_token_value(self) -> str:
        return self.bot_token.get_secret_value()


class SchedulerSettings(BaseSettings):
    """APScheduler job interval settings."""

    model_config = SettingsConfigDict(
        env_prefix="SCHEDULER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    crypto_fetch_interval_seconds: int = Field(
        default=300, ge=60, description="How often to fetch crypto data (seconds)"
    )
    stock_fetch_interval_seconds: int = Field(
        default=900, ge=60, description="How often to fetch stock data (seconds)"
    )
    prediction_interval_seconds: int = Field(
        default=600, ge=60, description="How often to run prediction inference (seconds)"
    )
    alert_check_interval_seconds: int = Field(
        default=120, ge=30, description="How often to evaluate alert rules (seconds)"
    )
    max_concurrent_jobs: int = Field(
        default=5, ge=1, le=50, description="Max concurrent scheduler jobs"
    )
    misfire_grace_time: int = Field(
        default=60, ge=0, description="Seconds a job can be late before being skipped"
    )


class APISettings(BaseSettings):
    """FastAPI server settings."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")
    workers: int = Field(default=4, ge=1, le=64, description="Uvicorn worker count")
    reload: bool = Field(default=False, description="Enable hot-reload (dev only)")
    cors_origins: list[str] = Field(
        default=["http://localhost:5173"],
        description="Allowed CORS origins",
    )
    secret_key: SecretStr = Field(..., description="JWT / session secret key (min 32 chars)")
    access_token_expire_minutes: int = Field(
        default=60, ge=5, description="JWT access token TTL in minutes"
    )
    api_v1_prefix: str = Field(default="/api/v1", description="API route prefix")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError("API_SECRET_KEY must be at least 32 characters long")
        return v


# ---------------------------------------------------------------------------
# Root application settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Root settings object. Composes all sub-settings groups.
    Loaded once at startup via get_settings().
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Application meta --
    app_name: str = Field(
        default="Market Predictor", description="Human-readable application name"
    )
    app_version: str = Field(default="1.0.0", description="Semantic version string")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Runtime environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Root log level")
    log_json: bool = Field(
        default=True, description="Emit logs as JSON (True for prod, False for dev)"
    )
    base_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent,
        description="Absolute path to the backend root directory",
    )

    # -- Sub-settings (composed via model_validator) --
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    coingecko: CoinGeckoSettings = Field(default_factory=CoinGeckoSettings)
    alphavantage: AlphaVantageSettings = Field(default_factory=AlphaVantageSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    api: APISettings = Field(default_factory=APISettings)

    @model_validator(mode="after")
    def enforce_production_constraints(self) -> "Settings":
        """Apply strict validation rules when running in production."""
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("debug must be False in production")
            if self.api.reload:
                raise ValueError("API hot-reload must be disabled in production")
            if self.db.echo_sql:
                raise ValueError("DB SQL echo must be disabled in production")
            if self.log_level == LogLevel.DEBUG:
                raise ValueError("Log level DEBUG is not allowed in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def build_logging_config(settings: Settings) -> dict[str, Any]:
    """
    Build a logging.config.dictConfig-compatible dictionary.
    Uses JSON formatter in production and a human-readable formatter in development.
    """
    log_level = settings.log_level.value

    formatters: dict[str, Any] = {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    }

    # Use python-json-logger if available, fall back to standard in dev
    if settings.log_json:
        try:
            import pythonjsonlogger  # noqa: F401 – presence check only
            formatters["json"] = {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(lineno)d %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
            active_formatter = "json"
        except ImportError:
            active_formatter = "standard"
    else:
        active_formatter = "standard"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": active_formatter,
                "level": log_level,
            },
            "error_console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": active_formatter,
                "level": "ERROR",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "error_console"],
        },
        "loggers": {
            # Silence noisy third-party loggers
            "httpx": {"level": "WARNING", "propagate": True},
            "httpcore": {"level": "WARNING", "propagate": True},
            "asyncio": {"level": "WARNING", "propagate": True},
            "sqlalchemy.engine": {
                "level": "DEBUG" if settings.db.echo_sql else "WARNING",
                "propagate": True,
            },
            "apscheduler": {"level": "INFO", "propagate": True},
            "telegram": {"level": "INFO", "propagate": True},
            # Application namespaces
            "backend": {"level": log_level, "propagate": True},
        },
    }


def configure_logging(settings: Settings) -> None:
    """
    Apply the logging configuration derived from application settings.
    Call this once at application startup before any loggers are used.
    """
    config = build_logging_config(settings)
    logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "environment": settings.environment.value,
            "log_level": settings.log_level.value,
            "log_json": settings.log_json,
        },
    )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application Settings instance.

    Usage:
        from backend.config.settings import get_settings
        settings = get_settings()

    The @lru_cache ensures Settings is instantiated exactly once per process,
    making it safe to call get_settings() anywhere without performance overhead.
    To override in tests, call get_settings.cache_clear() before patching env vars.
    """
    settings = Settings()
    configure_logging(settings)
    return settings


# ---------------------------------------------------------------------------
# Module-level startup guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick validation: python -m backend.config.settings
    try:
        s = get_settings()
        print(f"Settings loaded successfully for environment: {s.environment.value}")
        print(f"  App         : {s.app_name} v{s.app_version}")
        print(f"  DB host     : {s.db.host}:{s.db.port}/{s.db.name}")
        print(f"  API         : {s.api.host}:{s.api.port}")
        print(f"  Log level   : {s.log_level.value}")
        print(f"  Environment : {s.environment.value}")
    except Exception as exc:
        print(f"[FATAL] Settings validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
