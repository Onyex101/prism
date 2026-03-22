"""
Compare Projects Page

Side-by-side comparison using JIRA-aligned metrics and RiskCharts.comparison_radar (notebook 07).
"""

from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st

from app.bootstrap import init_page
from app.components import RADAR_METRICS_JIRA, normalize_row_against_frame, require_projects_df

init_page()

from src.visualization.risk_charts import RiskCharts

st.title("🔍 Compare Projects")
st.markdown("Side-by-side comparison of project metrics and risk factors")

df = require_projects_df()

st.markdown("### Select Projects to Compare")

name_col = "project_name" if "project_name" in df.columns else "project_id"
project_list = df[name_col].tolist()

col1, col2 = st.columns(2)

with col1:
    project_a = st.selectbox("Project A", project_list, index=0)

with col2:
    project_b = st.selectbox(
        "Project B",
        project_list,
        index=min(1, len(project_list) - 1),
    )

if project_a == project_b:
    st.warning("Please select two different projects to compare.")
    st.stop()

row_a = df[df[name_col] == project_a].iloc[0]
row_b = df[df[name_col] == project_b].iloc[0]

st.markdown("---")
st.markdown("### Side-by-Side Comparison")

metrics_to_compare = [
    ("completion_rate", "Completion Rate", "%"),
    ("budget", "Budget", "$"),
    ("spent", "Spent", "$"),
    ("team_size", "Team Size", ""),
    ("defect_rate", "Defect rate", ""),
    ("blocker_ratio", "Blocker ratio", ""),
    ("reopen_rate", "Reopen rate", ""),
    ("churn_rate", "Churn rate", ""),
    ("risk_score", "ML Risk Score", ""),
    ("mcda_score", "MCDA Score", ""),
]


def _fmt_val(val, unit):
    if pd.isna(val):
        return "N/A"
    if unit == "$":
        return f"${val:,.0f}"
    if unit == "%":
        return f"{val:.1f}%"
    return f"{val:.4f}" if isinstance(val, (int, float)) else str(val)


for metric, label, unit in metrics_to_compare:
    if metric not in df.columns:
        continue
    c1, c2, c3 = st.columns([2, 1, 2])
    val_a = row_a[metric]
    val_b = row_b[metric]
    with c1:
        st.write(_fmt_val(val_a, unit))
    with c2:
        st.write(f"**{label}**")
    with c3:
        st.write(_fmt_val(val_b, unit))

st.markdown("---")
st.markdown("### Visual Comparison (JIRA metrics, notebook 07)")

available_metrics = [
    m
    for m in RADAR_METRICS_JIRA
    if m in df.columns
    and pd.notna(row_a.get(m))
    and pd.notna(row_b.get(m))
    and pd.notna(df[m].min())
    and pd.notna(df[m].max())
]


if len(available_metrics) >= 3:
    try:
        projects_payload = [
            normalize_row_against_frame(row_a, df, available_metrics, name_col),
            normalize_row_against_frame(row_b, df, available_metrics, name_col),
        ]
        fig = RiskCharts.comparison_radar(projects_payload, available_metrics)
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.warning(f"Could not build radar chart: {e}")
else:
    st.info(
        "Not enough JIRA metrics for radar (need at least 3 of: defect_rate, blocker_ratio, "
        "reopen_rate, churn_rate, completion_rate)."
    )

if "status_comments" in df.columns:
    st.markdown("---")
    st.markdown("### Status Comments Comparison")

    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown(f"**{project_a}**")
        comments_a = "" if pd.isna(row_a["status_comments"]) else str(row_a["status_comments"])
        st.text_area(
            f"Status comments for {project_a}",
            comments_a,
            height=200,
            disabled=True,
            key="comments_a",
            label_visibility="collapsed",
        )

    with cc2:
        st.markdown(f"**{project_b}**")
        comments_b = "" if pd.isna(row_b["status_comments"]) else str(row_b["status_comments"])
        st.text_area(
            f"Status comments for {project_b}",
            comments_b,
            height=200,
            disabled=True,
            key="comments_b",
            label_visibility="collapsed",
        )
