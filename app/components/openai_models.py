"""
Shared OpenAI model list for LLM Insights and Chat Assistant (single source of truth).

The selected model is stored in ``st.session_state["openai_model"]`` so both pages stay aligned.
"""

from __future__ import annotations

import os

OPENAI_MODEL_OPTIONS: list[str] = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]

DEFAULT_OPENAI_MODEL = "gpt-4.1"


def openai_model_index_from_env() -> int:
    """Index into :data:`OPENAI_MODEL_OPTIONS` for ``OPENAI_MODEL`` (fallback: 0)."""
    m = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return OPENAI_MODEL_OPTIONS.index(m) if m in OPENAI_MODEL_OPTIONS else 0


def normalize_openai_model(value: str | None) -> str:
    """Return ``value`` if it is a known option, else :data:`DEFAULT_OPENAI_MODEL`."""
    if value and value in OPENAI_MODEL_OPTIONS:
        return value
    return DEFAULT_OPENAI_MODEL
