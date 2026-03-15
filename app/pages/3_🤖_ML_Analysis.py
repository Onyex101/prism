"""
ML Analysis Page

Machine learning risk predictions and feature importance using PRISM src modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ML Analysis - PRISM", page_icon="🤖", layout="wide")

st.title("🤖 ML Analysis")
st.markdown("Machine learning risk predictions and feature importance")

# Check if data is loaded
if "projects_df" not in st.session_state:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/2_📁_Upload_Data.py")
    st.stop()

df = st.session_state["projects_df"].copy()

# Run analysis section
st.markdown("### Run ML Analysis")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running ML analysis..."):
            try:
                from src.data import FeatureEngineer
                from src.models.ml import MLPredictor, MLTrainer, ModelEvaluator

                # Feature engineering
                fe = FeatureEngineer()
                df_fe = fe.create_features(df)

                # Exclude non-feature columns
                exclude_cols = [
                    "project_id",
                    "project_name",
                    "risk_level",
                    "status_comments",
                    "project_description",
                    "team_feedback",
                    "start_date",
                    "planned_end_date",
                    "actual_end_date",
                    "technology_stack",
                    "stakeholder_notes",
                ]
                feature_cols = [
                    col
                    for col in df_fe.columns
                    if col not in exclude_cols
                    and df_fe[col].dtype in ["int64", "float64", "int32", "float32"]
                ]

                X = df_fe[feature_cols].fillna(0)

                model_path = Path(__file__).parent.parent.parent / "models" / "ml" / "best_model.pkl"
                predictor = None

                importance_df = None
                if model_path.exists():
                    predictor = MLPredictor(model_path=model_path)
                    if predictor.feature_names:
                        for col in predictor.feature_names:
                            if col not in X.columns:
                                X[col] = 0
                        X = X[predictor.feature_names]
                    scores_df = predictor.get_risk_scores(X)
                    importance_df = predictor.get_feature_importance()
                else:
                    # No trained model: train on the fly
                    if MLTrainer is None:
                        st.error(
                            "MLTrainer unavailable. Install xgboost: pip install xgboost. "
                            "Or run 'make train' to train a model first."
                        )
                        st.stop()

                    y = (
                        (df_fe["risk_level"] == "High").astype(int)
                        if "risk_level" in df_fe.columns
                        else (df_fe["completion_rate"] < 50).astype(int)
                        if "completion_rate" in df_fe.columns
                        else pd.Series([0] * len(df_fe))
                    )

                    trainer = MLTrainer(model_type="random_forest")
                    trainer.train(X, y)
                    probas = trainer.model.predict_proba(X)[:, -1]
                    risk_levels = [
                        "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"
                        for p in probas
                    ]
                    scores_df = pd.DataFrame(
                        {"risk_score": probas, "risk_level": risk_levels}
                    )
                    imp = getattr(trainer.model, "feature_importances_", None)
                    if imp is not None:
                        importance_df = pd.DataFrame(
                            {"feature": feature_cols[: len(imp)], "importance": imp}
                        ).sort_values("importance", ascending=False)

                df_fe["risk_score"] = scores_df["risk_score"].values
                df_fe["risk_level"] = scores_df["risk_level"].values

                st.session_state["projects_df"] = df_fe
                st.session_state["ml_importance_df"] = importance_df
                st.success("✅ ML analysis complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback

                st.code(traceback.format_exc())

with col2:
    if "risk_score" in df.columns or "risk_score" in st.session_state.get("projects_df", pd.DataFrame()).columns:
        disp_df = st.session_state["projects_df"]
        fig_data = disp_df["risk_score"] if "risk_score" in disp_df.columns else None
        if fig_data is not None:
            import plotly.graph_objects as go

            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=fig_data,
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
disp_df = st.session_state.get("projects_df", df)
if "risk_score" in disp_df.columns:
    st.markdown("---")
    st.markdown("### Prediction Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_score = disp_df["risk_score"].mean()
        st.metric("Average Risk Score", f"{avg_score:.2f}")

    with col2:
        high_count = (disp_df["risk_level"] == "High").sum()
        st.metric("High Risk Projects", high_count)

    with col3:
        medium_count = (disp_df["risk_level"] == "Medium").sum()
        st.metric("Medium Risk Projects", medium_count)

    with col4:
        low_count = (disp_df["risk_level"] == "Low").sum()
        st.metric("Low Risk Projects", low_count)

    st.markdown("### Project Risk Scores")

    display_cols = ["project_name", "risk_score", "risk_level"]
    if "completion_rate" in disp_df.columns:
        display_cols.append("completion_rate")
    if "status" in disp_df.columns:
        display_cols.append("status")

    results_df = disp_df[
        [col for col in display_cols if col in disp_df.columns]
    ].sort_values("risk_score", ascending=False)

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

    st.markdown("---")
    st.markdown("### Feature Importance")

    importance_df = st.session_state.get("ml_importance_df")

    if importance_df is not None and len(importance_df) > 0:
        try:
            from src.visualization.risk_charts import RiskCharts

            fig = RiskCharts.feature_importance_bar(importance_df, top_n=10)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            import plotly.graph_objects as go

            plot_df = importance_df.head(10)
            fig = go.Figure(
                data=[
                    go.Bar(
                        y=plot_df["feature"],
                        x=plot_df["importance"],
                        orientation="h",
                        marker_color="#1E88E5",
                    )
                ]
            )
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Feature importance shows which factors contribute most to risk predictions. "
            "Run 'make train' to train a model, then run analysis to see importance."
        )

else:
    st.info("👆 Click 'Run Analysis' to generate ML predictions for your projects.")
