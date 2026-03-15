"""
Rankings Page

MCDA-based project prioritization using PRISM src modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import yaml

st.set_page_config(page_title="Rankings - PRISM", page_icon="📈", layout="wide")

st.title("📈 Project Rankings")
st.markdown("MCDA-based project prioritization combining ML, LLM, and metrics")

# Check if data is loaded
if "projects_df" not in st.session_state:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/2_📁_Upload_Data.py")
    st.stop()

df = st.session_state["projects_df"].copy()

# Ensure project_id exists (ProjectRanker requires it)
if "project_id" not in df.columns and "project_name" in df.columns:
    df["project_id"] = df["project_name"]

# Run feature engineering if needed (for SPI, CPI, team_stability)
if "schedule_performance_index" not in df.columns or "cost_performance_index" not in df.columns:
    try:
        from src.data import FeatureEngineer

        fe = FeatureEngineer()
        df = fe.create_features(df)
    except Exception:
        pass

st.session_state["projects_df"] = df

# Load MCDA config for default weights
config_path = Path(__file__).parent.parent.parent / "config" / "mcda_config.yaml"
default_weights = {
    "ml_risk_score": 0.40,
    "llm_sentiment_score": 0.25,
    "schedule_performance_index": 0.15,
    "cost_performance_index": 0.10,
    "team_stability": 0.10,
}
if config_path.exists():
    try:
        with open(config_path) as f:
            mcda_config = yaml.safe_load(f)
        if mcda_config and "criteria" in mcda_config:
            for k, v in mcda_config["criteria"].items():
                if isinstance(v, dict) and "weight" in v:
                    default_weights[k] = v["weight"]
    except Exception:
        pass

# MCDA Configuration
st.markdown("### MCDA Configuration")

st.markdown("Adjust the weights for each criterion (must sum to 1.0)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    w_ml = st.slider("ML Risk Score", 0.0, 1.0, float(default_weights["ml_risk_score"]), 0.05)

with col2:
    w_llm = st.slider("LLM Sentiment", 0.0, 1.0, float(default_weights["llm_sentiment_score"]), 0.05)

with col3:
    w_spi = st.slider("Schedule Perf", 0.0, 1.0, float(default_weights["schedule_performance_index"]), 0.05)

with col4:
    w_cpi = st.slider("Cost Perf", 0.0, 1.0, float(default_weights["cost_performance_index"]), 0.05)

with col5:
    w_team = st.slider("Team Stability", 0.0, 1.0, float(default_weights["team_stability"]), 0.05)

total_weight = w_ml + w_llm + w_spi + w_cpi + w_team
if abs(total_weight - 1.0) > 0.01:
    st.warning(f"⚠️ Weights sum to {total_weight:.2f}. Please adjust to sum to 1.0")

# Run ranking
if st.button("🎯 Calculate Rankings", type="primary"):
    with st.spinner("Calculating MCDA rankings..."):
        try:
            from src.mcda import ProjectRanker

            criteria = {
                "ml_risk_score": {"weight": w_ml, "type": "cost"},
                "llm_sentiment_score": {"weight": w_llm, "type": "benefit"},
                "schedule_performance_index": {"weight": w_spi, "type": "benefit"},
                "cost_performance_index": {"weight": w_cpi, "type": "benefit"},
                "team_stability": {"weight": w_team, "type": "benefit"},
            }

            ranker = ProjectRanker(criteria=criteria)
            rankings_df = ranker.rank(df)

            # Merge rankings back into projects_df
            merged = df.merge(
                rankings_df[["project_id", "mcda_score", "rank", "risk_level"]],
                on="project_id",
                how="left",
            )
            st.session_state["projects_df"] = merged
            st.session_state["rankings_df"] = rankings_df

            st.success("✅ Rankings calculated!")

        except Exception as e:
            st.error(f"Ranking failed: {e}")
            import traceback

            st.code(traceback.format_exc())

# Display rankings
disp_df = st.session_state.get("projects_df", df)
if "mcda_score" in disp_df.columns:
    st.markdown("---")
    st.markdown("### Project Rankings")

    col1, col2, col3 = st.columns(3)

    with col1:
        high_risk = (disp_df["risk_level"] == "High").sum()
        st.metric("High Risk", high_risk)

    with col2:
        medium_risk = (disp_df["risk_level"] == "Medium").sum()
        st.metric("Medium Risk", medium_risk)

    with col3:
        low_risk = (disp_df["risk_level"] == "Low").sum()
        st.metric("Low Risk", low_risk)

    try:
        from src.visualization.risk_charts import RiskCharts

        rankings_for_chart = st.session_state.get("rankings_df", disp_df)
        if "risk_level" in rankings_for_chart.columns:
            fig = RiskCharts.risk_score_bar(
                rankings_for_chart,
                top_n=15,
                name_col="project_name" if "project_name" in rankings_for_chart.columns else "project_id",
                score_col="mcda_score",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        import plotly.graph_objects as go

        sorted_df = disp_df.sort_values("mcda_score", ascending=True).head(15)
        name_col = "project_name" if "project_name" in sorted_df.columns else "project_id"
        colors = [
            "#FF4B4B" if l == "High" else "#FFA500" if l == "Medium" else "#00CC66"
            for l in sorted_df["risk_level"]
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    y=sorted_df[name_col],
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

    st.markdown("### Full Rankings Table")

    display_cols = ["rank", "project_name", "mcda_score", "risk_level"]
    for col in ["completion_rate", "status", "priority"]:
        if col in disp_df.columns:
            display_cols.append(col)

    rankings_df = disp_df[[col for col in display_cols if col in disp_df.columns]].sort_values("rank")

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
