"""
PRISM application settings
==========================

``Settings`` is populated at import time from environment variables (see
``env_template.txt``). Defaults favor local development (``DEBUG=true``).

**Important fields**

- ``OPENAI_*``: LLM pages and scripts; leave key empty only for offline demos.
- ``ML_RANDOM_STATE``: Seeds sklearn/XGBoost training for reproducibility.
- ``LLM_*``: Cache directory under ``.cache/llm_responses`` and retry policy.
- ``STREAMLIT_*``: Host/port when not using Streamlit CLI defaults.

Call ``Settings.validate()`` before production runs if you require an API key.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """
    Application settings loaded from environment variables.

    Attributes mirror keys in ``env_template.txt``. Paths are absolute under
    :attr:`PROJECT_ROOT`.
    """

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_RAW_PATH: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_PATH: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_PATH: Path = PROJECT_ROOT / "models"
    CONFIG_PATH: Path = PROJECT_ROOT / "config"

    # Application settings
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

    # ML settings
    ML_MODEL_VERSION: str = os.getenv("ML_MODEL_VERSION", "v1")
    ML_RANDOM_STATE: int = int(os.getenv("ML_RANDOM_STATE", "42"))

    # LLM cache settings
    LLM_CACHE_ENABLED: bool = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
    LLM_CACHE_PATH: Path = PROJECT_ROOT / ".cache" / "llm_responses"
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))

    # Dashboard settings
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

    @classmethod
    def validate(cls) -> list[str]:
        """Validate required settings and return list of errors."""
        errors = []

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")

        return errors

    @classmethod
    def ensure_directories(cls) -> None:
        """Create required directories if they don't exist."""
        directories = [
            cls.DATA_RAW_PATH,
            cls.DATA_PROCESSED_PATH,
            cls.MODELS_PATH,
            cls.LLM_CACHE_PATH,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
