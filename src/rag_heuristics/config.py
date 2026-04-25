from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RHD_", env_file=".env", extra="ignore")

    model_provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "all-MiniLM-L6-v2"
    hf_token: Optional[str] = None

    vector_db_path: Path = Field(default=Path("data/vector_db"))
    normalized_docs_path: Path = Field(default=Path("data/corpus/normalized_docs.jsonl"))
    program_db_path: Path = Field(default=Path("data/program_db/programs.sqlite"))
    reports_dir: Path = Field(default=Path("reports"))

    default_top_k: int = 8
    generation_timeout_seconds: int = 6
    random_seed: int = 42


def get_settings() -> Settings:
    return Settings()
