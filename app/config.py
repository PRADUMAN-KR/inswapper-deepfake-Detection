from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    app_name: str = "inswapper-detector"
    app_version: str = "0.1.0"
    environment: str = "local"
    model_path: Path = Path("checkpoints/best_model.pt")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    device: str = "auto"
    max_upload_mb: int = Field(default=100, ge=1)
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="INSWAPPER_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
