"""Configuration management using pydantic-settings"""

from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.constants import DEFAULT_REGION_ID

# APScheduler's CronTrigger only accepts three-letter day abbreviations (or
# 0-6). Full day names such as "sunday" raise ValueError at startup, which used
# to crash-loop the collector container.
_DAY_ALIASES = {
    "monday": "mon",
    "tuesday": "tue",
    "tues": "tue",
    "wednesday": "wed",
    "thursday": "thu",
    "thur": "thu",
    "thurs": "thu",
    "friday": "fri",
    "saturday": "sat",
    "sunday": "sun",
}
_VALID_DAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}


def _split_csv(value):
    """Accept a comma-separated string or an already-parsed list."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Tolerate retired keys still present in a local .env
        protected_namespaces=()  # Allow "model_" prefix in field names
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://eve_user:password@localhost:5432/eve_markets",
        description="PostgreSQL connection URL"
    )
    db_pool_size: int = Field(default=20, description="Database connection pool size")
    db_max_overflow: int = Field(default=10, description="Max connections above pool size")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    esi_cache_enabled: bool = Field(
        default=True,
        description="Cache ESI responses (and ETags) in Redis when reachable"
    )

    # ESI API
    esi_base_url: str = Field(
        default="https://esi.evetech.net/latest",
        description="ESI API base URL"
    )
    esi_datasource: str = Field(default="tranquility", description="ESI datasource")
    esi_rate_limit: int = Field(default=100, description="Max requests per period")
    esi_rate_period: int = Field(default=60, description="Rate limit period in seconds")
    esi_user_agent: str = Field(
        default="eve-discovery/0.1 (market analytics; +https://github.com/TargetedEntropy/eve-discovery)",
        description="User-Agent sent to ESI, as required by CCP's third-party guidelines"
    )
    esi_max_retries: int = Field(default=4, description="Retry attempts for 5xx/timeout errors")
    esi_error_limit_floor: int = Field(
        default=20,
        description="Pause until the error window resets when remaining errors drop below this"
    )

    # Application
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")

    # Collection scope
    # Stored as a raw comma-separated string: pydantic-settings JSON-decodes
    # list-typed fields before validators run, which makes "a,b" unusable.
    collection_region_ids: str = Field(
        default=str(DEFAULT_REGION_ID),
        description="Regions to collect and expose in the dashboard (comma separated)"
    )

    # Model
    model_dir: str = Field(
        default="data/models",
        description="Directory holding per-item model checkpoints and metadata"
    )
    lookback_window: int = Field(default=30, description="Lookback window for LSTM")
    prediction_horizon_days: int = Field(
        default=30,
        description="How many days ahead the recursive forecast runs"
    )

    # API
    api_cors_origins: str = Field(
        default="http://localhost:8050",
        description="Allowed CORS origins (comma separated, or '*' for any)"
    )
    api_cors_allow_credentials: bool = Field(
        default=False,
        description="Send credentials with CORS requests (invalid together with origin '*')"
    )
    api_store_predictions: bool = Field(
        default=True,
        description="Persist /predict results to the predictions table for later scoring"
    )

    # Dashboard
    dashboard_debug: bool = Field(
        default=False,
        description="Run Dash with the Werkzeug debugger (never enable in a container)"
    )
    dashboard_refresh_seconds: int = Field(
        default=900,
        description="Dashboard auto-refresh interval; daily data does not need 60s polling"
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = Field(
        default="eve-price-prediction",
        description="MLflow experiment name"
    )
    mlflow_registered_model_prefix: str = Field(
        default="eve-price",
        description="Prefix for MLflow registered model names (one per type_id)"
    )

    # Scheduler
    collection_interval_minutes: int = Field(
        default=5,
        description="Market order snapshot collection interval"
    )
    feature_computation_interval_hours: int = Field(
        default=1,
        description="Feature computation interval"
    )
    history_collection_hour: int = Field(
        default=1,
        description="Hour (UTC) of the daily market-history collection job"
    )
    prediction_hour: int = Field(
        default=3,
        description="Hour (UTC) of the daily prediction generation job"
    )
    drift_check_hour: int = Field(
        default=4,
        description="Hour (UTC) of the daily drift detection job"
    )
    retraining_day: str = Field(default="sun", description="Model retraining day (mon-sun or 0-6)")
    retraining_hour: int = Field(default=2, description="Model retraining hour")
    retrain_on_drift: bool = Field(
        default=True,
        description="Queue a retraining run when the daily drift check fires"
    )

    # Monitoring / alerting
    drift_threshold: float = Field(default=0.5, description="Share of drifted features that alerts")
    alert_webhook_url: Optional[str] = Field(
        default=None,
        description="Optional webhook (Slack-compatible JSON payload) for drift and job failures"
    )

    # Data retention (TimescaleDB policies)
    order_snapshot_retention_days: int = Field(
        default=30,
        description="Drop raw order snapshot chunks older than this many days"
    )
    order_snapshot_compress_after_days: int = Field(
        default=2,
        description="Compress order snapshot chunks older than this many days"
    )

    @field_validator("collection_region_ids")
    @classmethod
    def _validate_region_ids(cls, value):
        parts = _split_csv(value)
        if not parts:
            raise ValueError("collection_region_ids must list at least one region id")
        for part in parts:
            if not part.isdigit():
                raise ValueError(f"collection_region_ids entries must be numeric; got {part!r}")
        return value

    @field_validator("retraining_day", mode="before")
    @classmethod
    def _normalize_retraining_day(cls, value):
        """Accept full day names and normalize to what CronTrigger understands."""
        if value is None:
            return value
        raw = str(value).strip().lower()
        if raw.isdigit():
            if not 0 <= int(raw) <= 6:
                raise ValueError("retraining_day as a number must be 0-6 (mon-sun)")
            return raw
        normalized = _DAY_ALIASES.get(raw, raw)
        if normalized not in _VALID_DAYS:
            raise ValueError(
                f"retraining_day must be one of {sorted(_VALID_DAYS)}, a full day name, "
                f"or 0-6; got {value!r}"
            )
        return normalized

    @field_validator("api_cors_allow_credentials")
    @classmethod
    def _reject_wildcard_with_credentials(cls, value, info):
        origins = _split_csv(info.data.get("api_cors_origins") or "")
        if value and "*" in origins:
            raise ValueError(
                "api_cors_allow_credentials cannot be true while api_cors_origins contains '*' "
                "-- browsers reject that combination and it exposes the API to any site"
            )
        return value

    @property
    def region_ids(self) -> List[int]:
        """Parsed ``COLLECTION_REGION_IDS``."""
        return [int(part) for part in _split_csv(self.collection_region_ids)]

    @property
    def cors_origins(self) -> List[str]:
        """Parsed ``API_CORS_ORIGINS``."""
        return _split_csv(self.api_cors_origins)


# Singleton settings instance
settings = Settings()
