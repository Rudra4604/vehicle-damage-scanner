"""Application configuration management using pydantic-settings.
Supports loading settings from environment variables and .env file.
"""
from pathlib import Path
from typing import List, Union
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings with environment variable support."""
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    MODEL_PATH: Path = Path("models/best.pt")
    OUTPUT_DIR: Path = Field(
        default=Path("outputs"),
        validation_alias=AliasChoices("OUTPUT_DIR", "OUTPUT_DIRECTORY"),
    )
    OUTPUT_RETENTION_HOURS: int = 24
    MAX_OUTPUT_FILES: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def OUTPUT_DIRECTORY(self) -> Path:
        """Alias property for OUTPUT_DIR."""
        return self.OUTPUT_DIR

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()
