"""
Reusable Streamlit UI helpers (guards, radar normalization, shared constants).

Import from ``app.components`` after :func:`app.bootstrap.init_page` so ``src`` and
``streamlit`` are available.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.streamlit_theme import RADAR_METRICS_JIRA, RISK_COLORS


def require_projects_df(key: str = "projects_df") -> pd.DataFrame:
    """
    Ensure session state has project data; otherwise show upload prompt and stop.

    :param key: Session key for the projects table (default ``projects_df``).
    :return: The loaded DataFrame.
    """
    if key not in st.session_state:
        st.warning("⚠️ No data loaded. Please upload data first.")
        if st.button("Go to Upload Page"):
            st.switch_page("pages/2_📁_Upload_Data.py")
        st.stop()
    return st.session_state[key]


def minmax_normalize_columns(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Min–max scale each column in ``metrics`` to [0, 1]; constant columns become 0.5.

    Used for portfolio radar (mean of normalized rows).
    """
    norm_df = df[metrics].copy()
    for m in metrics:
        lo, hi = norm_df[m].min(), norm_df[m].max()
        if hi > lo:
            norm_df[m] = (norm_df[m] - lo) / (hi - lo)
        else:
            norm_df[m] = 0.5
    return norm_df


def normalize_row_against_frame(
    row: pd.Series,
    df: pd.DataFrame,
    metrics: list[str],
    name_col: str,
) -> dict[str, Any]:
    """
    Min–max each metric using **column** min/max in ``df`` (compare two projects fairly).
    """
    out: dict[str, Any] = {"project_name": str(row[name_col])}
    for m in metrics:
        col_min = float(df[m].min())
        col_max = float(df[m].max())
        rng = col_max - col_min if col_max != col_min else 1.0
        val = float(row[m])
        out[m] = (val - col_min) / rng
    return out


def portfolio_radar_row(df: pd.DataFrame, metrics: list[str]) -> dict[str, Any]:
    """Mean of per-column min–max normalized values (Dashboard portfolio radar)."""
    norm_df = minmax_normalize_columns(df, metrics)
    vals = {m: float(norm_df[m].mean()) for m in metrics}
    return {"project_name": "Portfolio (mean normalized)", **vals}


__all__ = [
    "RADAR_METRICS_JIRA",
    "RISK_COLORS",
    "minmax_normalize_columns",
    "normalize_row_against_frame",
    "portfolio_radar_row",
    "require_projects_df",
]
