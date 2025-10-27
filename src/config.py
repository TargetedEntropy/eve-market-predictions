"""Configuration management using pydantic-settings"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
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

    # ESI API
    esi_base_url: str = Field(
        default="https://esi.evetech.net/latest",
        description="ESI API base URL"
    )
    esi_datasource: str = Field(default="tranquility", description="ESI datasource")
    esi_rate_limit: int = Field(default=100, description="Max requests per period")
    esi_rate_period: int = Field(default=60, description="Rate limit period in seconds")

    # Application
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")

    # Model
    model_path: str = Field(default="data/models/best_model.pth", description="Model file path")
    lookback_window: int = Field(default=30, description="Lookback window for LSTM")
    prediction_horizon: int = Field(default=24, description="Prediction horizon in hours")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = Field(
        default="eve-price-prediction",
        description="MLflow experiment name"
    )

    # Scheduler
    collection_interval_minutes: int = Field(
        default=5,
        description="Market data collection interval"
    )
    feature_computation_interval_hours: int = Field(
        default=1,
        description="Feature computation interval"
    )
    retraining_day: str = Field(default="sun", description="Model retraining day (mon-sun or 0-6)")
    retraining_hour: int = Field(default=2, description="Model retraining hour")


# Singleton settings instance
settings = Settings()
