"""
Dashboard Page

Portfolio overview and risk summary using PRISM src modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard - PRISM", page_icon="📊", layout="wide")

st.title("📊 Dashboard")
st.markdown("Portfolio overview and risk summary")

# Check if data is loaded
if "projects_df" not in st.session_state:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/2_📁_Upload_Data.py")
    st.stop()

df = st.session_state["projects_df"]

# Overview metrics using src.utils.metrics
try:
    from src.utils.metrics import calculate_metrics, calculate_portfolio_health

    metrics = calculate_metrics(df)
    health = calculate_portfolio_health(metrics)
except Exception:
    metrics = {"total_projects": len(df)}
    health = {"health_score": 0.5, "health_level": "Unknown", "components": {}}

st.markdown("### Portfolio Overview")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Projects", metrics.get("total_projects", len(df)))

with col2:
    high_risk = metrics.get("high_risk_count", (df["risk_level"] == "High").sum() if "risk_level" in df.columns else "N/A")
    st.metric("High Risk", high_risk)

with col3:
    medium_risk = metrics.get("medium_risk_count", (df["risk_level"] == "Medium").sum() if "risk_level" in df.columns else "N/A")
    st.metric("Medium Risk", medium_risk)

with col4:
    low_risk = metrics.get("low_risk_count", (df["risk_level"] == "Low").sum() if "risk_level" in df.columns else "N/A")
    st.metric("Low Risk", low_risk)

with col5:
    if "completion_rate" in df.columns:
        avg_completion = df["completion_rate"].mean()
        st.metric("Avg Completion", f"{avg_completion:.1f}%")
    else:
        st.metric("Avg Completion", metrics.get("avg_completion", "N/A"))

with col6:
    st.metric("Portfolio Health", f"{health.get('health_level', 'N/A')} ({health.get('health_score', 0):.0%})")

st.markdown("---")

# Charts using RiskCharts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Risk Distribution")
    if "risk_level" in df.columns:
        try:
            from src.visualization.risk_charts import RiskCharts

            fig = RiskCharts.risk_distribution_pie(df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            import plotly.graph_objects as go

            risk_counts = df["risk_level"].value_counts()
            colors = {"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#00CC66"}
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        hole=0.4,
                        marker_colors=[colors.get(l, "#808080") for l in risk_counts.index],
                    )
                ]
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Risk level not calculated yet. Run ML Analysis or Rankings first.")

with col2:
    st.markdown("### Status Distribution")
    if "status" in df.columns:
        import plotly.express as px

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
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No status data available.")

# Top risk projects
st.markdown("---")
st.markdown("### Top Risk Projects")

score_col = "mcda_score" if "mcda_score" in df.columns else "risk_score"
if score_col in df.columns:
    try:
        from src.visualization.risk_charts import RiskCharts

        rankings_df = st.session_state.get("rankings_df", df)
        if "risk_level" in rankings_df.columns:
            fig = RiskCharts.risk_score_bar(rankings_df, top_n=5)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    if score_col == "risk_score":
        top_risk = df.nlargest(5, score_col)
    else:
        top_risk = df.nsmallest(5, score_col)
    top_risk = top_risk[
        [
            col
            for col in [
                "project_name",
                "project_id",
                score_col,
                "risk_level",
                "completion_rate",
                "status",
            ]
            if col in df.columns
        ]
    ]
    st.dataframe(top_risk, use_container_width=True, hide_index=True)
else:
    st.info("Risk analysis not yet performed. Run ML Analysis or Rankings first.")
    display_cols = [
        col
        for col in ["project_name", "completion_rate", "status", "priority"]
        if col in df.columns
    ]
    if display_cols:
        st.dataframe(df[display_cols].head(10), use_container_width=True, hide_index=True)

# Budget overview
st.markdown("---")
st.markdown("### Budget Overview")

if "budget" in df.columns and "spent" in df.columns:
    col1, col2, col3 = st.columns(3)

    with col1:
        total_budget = df["budget"].sum()
        st.metric("Total Budget", f"${total_budget:,.0f}")

    with col2:
        total_spent = df["spent"].sum()
        st.metric("Total Spent", f"${total_spent:,.0f}")

    with col3:
        variance = ((total_spent - total_budget) / total_budget * 100) if total_budget > 0 else 0
        st.metric("Budget Variance", f"{variance:+.1f}%")

    import plotly.graph_objects as go

    budget_data = df[["project_name", "budget", "spent"]].head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Budget", x=budget_data["project_name"], y=budget_data["budget"]))
    fig.add_trace(go.Bar(name="Spent", x=budget_data["project_name"], y=budget_data["spent"]))
    fig.update_layout(barmode="group", height=350)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No budget data available.")
