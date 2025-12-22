"""
Dashboard Page

Portfolio overview and risk summary.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# Overview metrics
st.markdown("### Portfolio Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Projects", len(df))

with col2:
    if "risk_level" in df.columns:
        high_risk = len(df[df["risk_level"] == "High"])
        st.metric("High Risk", high_risk)
    else:
        st.metric("High Risk", "N/A")

with col3:
    if "risk_level" in df.columns:
        medium_risk = len(df[df["risk_level"] == "Medium"])
        st.metric("Medium Risk", medium_risk)
    else:
        st.metric("Medium Risk", "N/A")

with col4:
    if "risk_level" in df.columns:
        low_risk = len(df[df["risk_level"] == "Low"])
        st.metric("Low Risk", low_risk)
    else:
        st.metric("Low Risk", "N/A")

with col5:
    if "completion_rate" in df.columns:
        avg_completion = df["completion_rate"].mean()
        st.metric("Avg Completion", f"{avg_completion:.1f}%")
    else:
        st.metric("Avg Completion", "N/A")

st.markdown("---")

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Risk Distribution")
    if "risk_level" in df.columns:
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
        st.info("Risk level not calculated yet. Run analysis first.")

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
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No status data available.")

# Top risk projects
st.markdown("---")
st.markdown("### Top Risk Projects")

if "risk_level" in df.columns or "mcda_score" in df.columns:
    score_col = "mcda_score" if "mcda_score" in df.columns else "risk_score"

    if score_col in df.columns:
        top_risk = df.nsmallest(5, score_col)[
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
        st.info("Run analysis to see risk rankings.")
else:
    # Show projects by completion rate as fallback
    st.info("Risk analysis not yet performed. Showing projects by completion rate.")
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

    # Budget chart
    budget_data = df[["project_name", "budget", "spent"]].head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Budget", x=budget_data["project_name"], y=budget_data["budget"]))
    fig.add_trace(go.Bar(name="Spent", x=budget_data["project_name"], y=budget_data["spent"]))
    fig.update_layout(barmode="group", height=350)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No budget data available.")
