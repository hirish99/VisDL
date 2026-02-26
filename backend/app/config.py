"""Application configuration via environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "VisDL"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    upload_dir: Path = Path("uploads")
    max_upload_size_mb: int = 100
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_prefix": "VISDL_"}


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
