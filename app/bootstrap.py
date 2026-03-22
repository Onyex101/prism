"""
Streamlit bootstrap: ``sys.path``, optional ``.env``, and shared UI setup.

Call :func:`bootstrap_env` before importing ``src`` from entry scripts.
For multipage scripts, call :func:`init_page` once at the top (after ``sys.path`` fix)
so ``.env`` and sidebar styles apply consistently.

Session state contract (Streamlit)
----------------------------------
The app uses these keys (values are typically ``pandas.DataFrame`` or chart data):

- ``projects_df``: loaded / generated project rows (required for analysis pages).
- ``rankings_df``: MCDA output from Rankings (optional; Dashboard prefers MCDA labels).
- ``ml_*`` / ``llm_*``: caches from ML and LLM pages where applicable.
- ``messages``, ``chat_assistant``, ``pending_question``: Chat Assistant state.
- ``openai_model``: OpenAI model id shared by **LLM Insights** and **Chat** (set on LLM page).

Keys are created on demand; pages guard with :func:`app.components.require_projects_df`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.errors


def ensure_project_root() -> Path:
    """
    Add repository root (parent of the ``app`` package) to ``sys.path``.

    :return: Absolute path to project root.
    """
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _load_dotenv(root: Path) -> None:
    """Load ``.env`` from project root if ``python-dotenv`` is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def bootstrap_env() -> Path:
    """
    Put repo root on ``sys.path`` and load ``.env`` when present.

    Use this from ``app/main.py`` **before** ``st.set_page_config`` so API keys are
    available; then call :func:`app.streamlit_theme.inject_sidebar_styles` after config.
    """
    root = ensure_project_root()
    _load_dotenv(root)
    return root


def init_page() -> Path:
    """
    Full setup for multipage scripts: env, **wide layout**, and sidebar styles.

    Call once at the top of each page (after the usual ``sys.path`` shim that
    allows ``import app``).

    ``st.set_page_config(layout="wide")`` is applied here (not only in ``main.py``)
    so **refreshing or deep-linking a subpage** still uses full-width layout.
    When the home page already configured the session, the duplicate call is
    ignored.
    """
    root = bootstrap_env()
    try:
        st.set_page_config(
            page_title="PRISM - Project Risk Intelligence",
            page_icon="🔮",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except streamlit.errors.StreamlitAPIException:
        pass
    from app.streamlit_theme import inject_sidebar_styles

    inject_sidebar_styles()

    from app.components.openai_models import (
        DEFAULT_OPENAI_MODEL,
        OPENAI_MODEL_OPTIONS,
        normalize_openai_model,
    )

    if "openai_model" not in st.session_state:
        raw = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        st.session_state.openai_model = normalize_openai_model(raw)

    return root
