"""
ML Analysis Page

Machine learning predictions and explanations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ML Analysis - PRISM", page_icon="🤖", layout="wide")

st.title("🤖 ML Analysis")
st.markdown("Machine learning risk predictions and feature importance")

# Check if data is loaded
if "projects_df" not in st.session_state:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/2_📁_Upload_Data.py")
    st.stop()

df = st.session_state["projects_df"]

# Run analysis section
st.markdown("### Run ML Analysis")

col1, col2 = st.columns([1, 2])

with col1:
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "LightGBM"],
        help="Choose the ML algorithm for prediction",
    )

    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running ML analysis..."):
            # Placeholder for actual ML analysis
            # In production, this would call the ML predictor

            import numpy as np

            # Generate synthetic predictions for demo
            np.random.seed(42)
            n = len(df)

            # Calculate risk scores based on available metrics
            risk_scores = np.random.uniform(0.2, 0.9, n)

            # Adjust based on completion rate if available
            if "completion_rate" in df.columns:
                # Lower completion -> higher risk
                completion_factor = (100 - df["completion_rate"].values) / 100
                risk_scores = 0.5 * risk_scores + 0.5 * completion_factor

            # Classify risk levels
            risk_levels = []
            for score in risk_scores:
                if score >= 0.6:
                    risk_levels.append("High")
                elif score >= 0.3:
                    risk_levels.append("Medium")
                else:
                    risk_levels.append("Low")

            df["risk_score"] = risk_scores
            df["risk_level"] = risk_levels
            st.session_state["projects_df"] = df

            st.success("✅ ML analysis complete!")

with col2:
    if "risk_score" in df.columns:
        # Show score distribution
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=df["risk_score"],
                    nbinsx=20,
                    marker_color="#1E88E5",
                )
            ]
        )
        fig.update_layout(
            title="Risk Score Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Count",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

# Results section
if "risk_score" in df.columns:
    st.markdown("---")
    st.markdown("### Prediction Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_score = df["risk_score"].mean()
        st.metric("Average Risk Score", f"{avg_score:.2f}")

    with col2:
        high_count = (df["risk_level"] == "High").sum()
        st.metric("High Risk Projects", high_count)

    with col3:
        medium_count = (df["risk_level"] == "Medium").sum()
        st.metric("Medium Risk Projects", medium_count)

    with col4:
        low_count = (df["risk_level"] == "Low").sum()
        st.metric("Low Risk Projects", low_count)

    # Results table
    st.markdown("### Project Risk Scores")

    display_cols = ["project_name", "risk_score", "risk_level"]
    if "completion_rate" in df.columns:
        display_cols.append("completion_rate")
    if "status" in df.columns:
        display_cols.append("status")

    results_df = df[[col for col in display_cols if col in df.columns]].sort_values(
        "risk_score", ascending=False
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=1,
                format="%.2f",
            ),
        },
    )

    # Feature importance (placeholder)
    st.markdown("---")
    st.markdown("### Feature Importance")

    st.info(
        "Feature importance shows which factors contribute most to risk predictions. "
        "This will be populated when a trained model is available."
    )

    # Placeholder feature importance
    features = [
        "completion_rate",
        "budget_variance",
        "schedule_performance",
        "team_size",
        "complexity_score",
        "defect_rate",
    ]
    importance = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]

    fig = go.Figure(
        data=[
            go.Bar(
                y=features,
                x=importance,
                orientation="h",
                marker_color="#1E88E5",
            )
        ]
    )
    fig.update_layout(
        title="Feature Importance (Placeholder)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Click 'Run Analysis' to generate ML predictions for your projects.")
