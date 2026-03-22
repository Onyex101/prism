"""Shared Streamlit CSS and UI color tokens. Multipage apps run each page independently."""

from __future__ import annotations

import streamlit as st

# Plotly / CSS fallbacks — keep in sync with ``main.py`` header styles
ACCENT_BLUE = "#1E88E5"
RISK_COLOR_HIGH = "#FF4B4B"
RISK_COLOR_MEDIUM = "#FFA500"
RISK_COLOR_LOW = "#00CC66"
RISK_COLORS = {"High": RISK_COLOR_HIGH, "Medium": RISK_COLOR_MEDIUM, "Low": RISK_COLOR_LOW}

# JIRA-style dimensions used by Dashboard and Compare (notebook 07)
RADAR_METRICS_JIRA = [
    "defect_rate",
    "blocker_ratio",
    "reopen_rate",
    "churn_rate",
    "completion_rate",
]


def inject_sidebar_styles() -> None:
    """Larger type for built-in multipage navigation links in the sidebar."""
    st.markdown(
        """
<style>
    /* Built-in page list (streamlit multipage navigation) */
    [data-testid="stSidebarNav"] {
        font-size: 1.125rem;
    }
    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebarNav"] p,
    [data-testid="stSidebarNav"] span {
        font-size: 1.125rem !important;
    }
    [data-testid="stSidebarNav"] li {
        font-size: 1.125rem;
    }
</style>
""",
        unsafe_allow_html=True,
    )
