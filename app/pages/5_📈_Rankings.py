"""
Rankings Page

MCDA-based project prioritization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Rankings - PRISM", page_icon="📈", layout="wide")

st.title("📈 Project Rankings")
st.markdown("MCDA-based project prioritization combining ML, LLM, and metrics")

# Check if data is loaded
if "projects_df" not in st.session_state:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/2_📁_Upload_Data.py")
    st.stop()

df = st.session_state["projects_df"]

# MCDA Configuration
st.markdown("### MCDA Configuration")

st.markdown("Adjust the weights for each criterion (must sum to 1.0)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    w_ml = st.slider("ML Risk Score", 0.0, 1.0, 0.40, 0.05)

with col2:
    w_llm = st.slider("LLM Sentiment", 0.0, 1.0, 0.25, 0.05)

with col3:
    w_spi = st.slider("Schedule Perf", 0.0, 1.0, 0.15, 0.05)

with col4:
    w_cpi = st.slider("Cost Perf", 0.0, 1.0, 0.10, 0.05)

with col5:
    w_team = st.slider("Team Stability", 0.0, 1.0, 0.10, 0.05)

total_weight = w_ml + w_llm + w_spi + w_cpi + w_team
if abs(total_weight - 1.0) > 0.01:
    st.warning(f"⚠️ Weights sum to {total_weight:.2f}. Please adjust to sum to 1.0")

# Run ranking
if st.button("🎯 Calculate Rankings", type="primary"):
    with st.spinner("Calculating MCDA rankings..."):
        import numpy as np

        # Calculate MCDA score (simplified version)
        # In production, this would use the actual MCDA ranker

        # Normalize available criteria
        scores = np.zeros(len(df))

        # ML risk score (cost - lower is better when inverted)
        if "risk_score" in df.columns:
            ml_normalized = 1 - df["risk_score"].values  # Invert: higher is better
        else:
            ml_normalized = np.full(len(df), 0.5)

        # Sentiment (benefit - higher is better)
        if "sentiment_score" in df.columns:
            sent_normalized = (df["sentiment_score"].values + 1) / 2  # Scale -1,1 to 0,1
        else:
            sent_normalized = np.full(len(df), 0.5)

        # Schedule Performance
        if "schedule_performance_index" in df.columns:
            spi = df["schedule_performance_index"].clip(0, 2) / 2
        elif "planned_hours" in df.columns and "actual_hours" in df.columns:
            spi = (df["planned_hours"] / df["actual_hours"].replace(0, 1)).clip(0, 2) / 2
        else:
            spi = np.full(len(df), 0.5)

        # Cost Performance
        if "cost_performance_index" in df.columns:
            cpi = df["cost_performance_index"].clip(0, 2) / 2
        elif "budget" in df.columns and "spent" in df.columns:
            cpi = (df["budget"] / df["spent"].replace(0, 1)).clip(0, 2) / 2
        else:
            cpi = np.full(len(df), 0.5)

        # Team Stability
        if "team_turnover" in df.columns:
            team = 1 - df["team_turnover"].clip(0, 1)
        else:
            team = np.full(len(df), 0.9)

        # Weighted sum
        mcda_score = (
            w_ml * ml_normalized
            + w_llm * sent_normalized
            + w_spi * spi
            + w_cpi * cpi
            + w_team * team
        )

        df["mcda_score"] = mcda_score
        df["rank"] = df["mcda_score"].rank(ascending=False).astype(int)
        df["risk_level"] = df["mcda_score"].apply(
            lambda x: "Low" if x >= 0.7 else ("Medium" if x >= 0.4 else "High")
        )

        st.session_state["projects_df"] = df
        st.success("✅ Rankings calculated!")

# Display rankings
if "mcda_score" in df.columns:
    st.markdown("---")
    st.markdown("### Project Rankings")

    # Summary
    col1, col2, col3 = st.columns(3)

    with col1:
        high_risk = (df["risk_level"] == "High").sum()
        st.metric("High Risk", high_risk)

    with col2:
        medium_risk = (df["risk_level"] == "Medium").sum()
        st.metric("Medium Risk", medium_risk)

    with col3:
        low_risk = (df["risk_level"] == "Low").sum()
        st.metric("Low Risk", low_risk)

    # Bar chart
    sorted_df = df.sort_values("mcda_score", ascending=True).head(15)

    colors = [
        "#FF4B4B" if l == "High" else "#FFA500" if l == "Medium" else "#00CC66"
        for l in sorted_df["risk_level"]
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=(
                    sorted_df["project_name"]
                    if "project_name" in sorted_df.columns
                    else sorted_df["project_id"]
                ),
                x=sorted_df["mcda_score"],
                orientation="h",
                marker_color=colors,
                text=sorted_df["mcda_score"].round(2),
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Project Rankings (Lower Score = Higher Risk)",
        xaxis_title="MCDA Score",
        yaxis_title="Project",
        height=max(400, len(sorted_df) * 30),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Rankings table
    st.markdown("### Full Rankings Table")

    display_cols = ["rank", "project_name", "mcda_score", "risk_level"]
    for col in ["completion_rate", "status", "priority"]:
        if col in df.columns:
            display_cols.append(col)

    rankings_df = df[[col for col in display_cols if col in df.columns]].sort_values("rank")

    st.dataframe(
        rankings_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "mcda_score": st.column_config.ProgressColumn(
                "MCDA Score",
                min_value=0,
                max_value=1,
                format="%.2f",
            ),
        },
    )

    # Export
    st.markdown("---")
    csv = rankings_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Rankings to CSV",
        data=csv,
        file_name="prism_rankings.csv",
        mime="text/csv",
    )

else:
    st.info("Click 'Calculate Rankings' to generate MCDA-based project prioritization.")
