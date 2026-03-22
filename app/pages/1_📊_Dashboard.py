"""
Dashboard Page

Portfolio overview and risk summary using PRISM src modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.bootstrap import init_page
from app.components import RADAR_METRICS_JIRA, RISK_COLORS, portfolio_radar_row, require_projects_df

init_page()

from src.utils.metrics import calculate_metrics, calculate_portfolio_health
from src.visualization.risk_charts import RiskCharts

st.title("📊 Dashboard")
st.markdown("Portfolio overview and risk summary")

df = require_projects_df()

# Prefer MCDA risk labels when present (notebook / Rankings), else ML / source risk_level
risk_view = (
    df["mcda_risk_level"]
    if "mcda_risk_level" in df.columns
    else df["risk_level"] if "risk_level" in df.columns else None
)
df_risk = df.copy()
if risk_view is not None:
    df_risk = df.assign(risk_level=risk_view)

try:
    metrics = calculate_metrics(df_risk)
    health = calculate_portfolio_health(metrics)
except (ValueError, KeyError, TypeError, AttributeError):
    metrics = {"total_projects": len(df)}
    health = {"health_score": 0.5, "health_level": "Unknown", "components": {}}

st.markdown("### Portfolio Overview")
if "mcda_risk_level" in df.columns:
    st.caption(
        "Risk counts and distribution use **MCDA** labels (`mcda_risk_level`) when available."
    )

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Projects", metrics.get("total_projects", len(df)))

with col2:
    high_risk = metrics.get(
        "high_risk_count",
        (df_risk["risk_level"] == "High").sum() if "risk_level" in df_risk.columns else "N/A",
    )
    st.metric("High Risk", high_risk)

with col3:
    medium_risk = metrics.get(
        "medium_risk_count",
        (df_risk["risk_level"] == "Medium").sum() if "risk_level" in df_risk.columns else "N/A",
    )
    st.metric("Medium Risk", medium_risk)

with col4:
    low_risk = metrics.get(
        "low_risk_count",
        (df_risk["risk_level"] == "Low").sum() if "risk_level" in df_risk.columns else "N/A",
    )
    st.metric("Low Risk", low_risk)

with col5:
    if "completion_rate" in df.columns:
        avg_completion = df["completion_rate"].mean()
        st.metric("Avg Completion", f"{avg_completion:.1f}%")
    else:
        st.metric("Avg Completion", metrics.get("avg_completion", "N/A"))

with col6:
    st.metric(
        "Portfolio Health",
        f"{health.get('health_level', 'N/A')} ({health.get('health_score', 0):.0%})",
    )

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Risk Distribution")
    if "risk_level" in df_risk.columns:
        try:
            fig = RiskCharts.risk_distribution_pie(df_risk)
            st.plotly_chart(fig, width="stretch")
        except Exception:
            risk_counts = df_risk["risk_level"].value_counts()
            colors = RISK_COLORS
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        hole=0.4,
                        marker_colors=[colors.get(str(l), "#808080") for l in risk_counts.index],
                    )
                ]
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Risk level not calculated yet. Run ML Analysis or Rankings first.")

with col2:
    st.markdown("### Status Distribution")
    if "status" in df.columns:
        status_counts = df["status"].value_counts()
        fig = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            color=status_counts.index,
        )
        fig.update_layout(
            height=350,
            xaxis_title="Status",
            yaxis_title="Count",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No status data available.")

# Portfolio JIRA radar (notebook 07 — five dimensions, mean-normalized)
st.markdown("---")
st.markdown("### Portfolio risk profile (JIRA metrics)")
st.caption(
    "Mean of min–max normalized metrics across loaded projects — same dimensions as notebook 07."
)

avail_r = [m for m in RADAR_METRICS_JIRA if m in df.columns]
if len(avail_r) >= 3:
    portfolio_project = portfolio_radar_row(df, avail_r)
    try:
        fig_pr = RiskCharts.comparison_radar([portfolio_project], avail_r)
        st.plotly_chart(fig_pr, width="stretch")
    except Exception as e:
        st.info(f"Could not render portfolio radar: {e}")
else:
    st.info(
        "Load JIRA-style data with defect_rate, blocker_ratio, reopen_rate, churn_rate, completion_rate for the portfolio radar."
    )

# Top risk projects
st.markdown("---")
st.markdown("### Top Risk Projects")

score_col = "mcda_score" if "mcda_score" in df.columns else "risk_score"
if score_col in df.columns:
    try:
        rankings_df = st.session_state.get("rankings_df", df)
        chart_df = rankings_df
        if "mcda_risk_level" in chart_df.columns:
            chart_df = chart_df.rename(columns={"mcda_risk_level": "risk_level"})
        if "risk_level" in chart_df.columns:
            fig = RiskCharts.risk_score_bar(chart_df, top_n=5)
            st.plotly_chart(fig, width="stretch")
    except Exception:
        pass

    if score_col == "risk_score":
        top_risk = df.nlargest(5, score_col)
    else:
        top_risk = df.nsmallest(5, score_col)

    level_col = "mcda_risk_level" if "mcda_risk_level" in df.columns else "risk_level"
    top_risk = top_risk[
        [
            col
            for col in [
                "project_name",
                "project_id",
                score_col,
                level_col,
                "completion_rate",
                "status",
            ]
            if col in df.columns
        ]
    ]
    st.dataframe(top_risk, width="stretch", hide_index=True)
else:
    st.info("Risk analysis not yet performed. Run ML Analysis or Rankings first.")
    display_cols = [
        col
        for col in ["project_name", "completion_rate", "status", "priority"]
        if col in df.columns
    ]
    if display_cols:
        st.dataframe(df[display_cols].head(10), width="stretch", hide_index=True)

st.markdown("---")
st.markdown("### Budget Overview")

if "budget" in df.columns and "spent" in df.columns:
    b1, b2, b3 = st.columns(3)

    with b1:
        total_budget = df["budget"].sum()
        st.metric("Total Budget", f"${total_budget:,.0f}")

    with b2:
        total_spent = df["spent"].sum()
        st.metric("Total Spent", f"${total_spent:,.0f}")

    with b3:
        variance = ((total_spent - total_budget) / total_budget * 100) if total_budget > 0 else 0
        st.metric("Budget Variance", f"{variance:+.1f}%")

    budget_data = df[["project_name", "budget", "spent"]].head(10)
    figb = go.Figure()
    figb.add_trace(go.Bar(name="Budget", x=budget_data["project_name"], y=budget_data["budget"]))
    figb.add_trace(go.Bar(name="Spent", x=budget_data["project_name"], y=budget_data["spent"]))
    figb.update_layout(barmode="group", height=350)
    st.plotly_chart(figb, width="stretch")
else:
    st.info("No budget data available.")
