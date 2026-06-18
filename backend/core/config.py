"""
ORBITAL Configuration Module
Loads settings from .env file using Pydantic BaseSettings.
"""

import os
from functools import lru_cache
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    class BaseSettings:
        """Minimal fallback settings loader when pydantic-settings is unavailable."""

        model_config = {}

        def __init__(self, **kwargs):
            env_file = self.model_config.get("env_file")
            if env_file:
                _load_env_file(env_file)

            for field_name, default_value in self.__class__.__dict__.items():
                if field_name.startswith("_") or callable(default_value):
                    continue
                env_value = os.getenv(field_name, kwargs.get(field_name, default_value))
                setattr(self, field_name, _coerce_value(default_value, env_value))


def _load_env_file(env_file: str) -> None:
    path = Path(env_file)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _coerce_value(default_value, env_value):
    if isinstance(default_value, bool):
        return str(env_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(env_value)
    if isinstance(default_value, float):
        return float(env_value)
    return env_value


class OrbitalConfig(BaseSettings):
    """Central configuration for the ORBITAL system."""

    # LLM Provider
    LLM_PROVIDER: str = "groq"

    # Groq API
    GROQ_API_KEY: str = ""
    GROQ_MODEL_EXTRACTION: str = "llama-3.1-8b-instant"
    GROQ_MODEL_GAP_ANALYSIS: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_EVIDENCE: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_CHAT: str = "llama-3.1-8b-instant"
    GROQ_MODEL_CLASSIFICATION: str = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS: int = 4096
    GROQ_TEMPERATURE: float = 0.1
    GROQ_TIMEOUT_SECONDS: int = 60

    # Ollama / local model
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3:latest"
    OLLAMA_MAX_TOKENS: int = 4096
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TIMEOUT_SECONDS: int = 60

    # Data paths
    RAW_DATA_PATH: str = "data/raw"
    EXTRACTED_DATA_PATH: str = "data/extracted"
    STRUCTURED_DATA_PATH: str = "data/structured"
    FINETUNE_DATA_PATH: str = "data/finetune"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MIN_OBLIGATION_CONFIDENCE: float = 0.65
    
    # Feature Flags
    GENERATE_FINETUNE_PAIRS: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_config() -> OrbitalConfig:
    """Return a cached singleton instance of OrbitalConfig."""
    return OrbitalConfig()
