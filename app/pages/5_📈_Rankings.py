"""
Rankings Page

MCDA-based project prioritization (TOPSIS) aligned with notebooks and ProjectRanker.
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
import yaml

from app.bootstrap import init_page
from app.components import require_projects_df

root = init_page()

from src.data import FeatureEngineer
from src.mcda.ranker import ProjectRanker

st.title("📈 Project Rankings")
st.markdown("MCDA-based project prioritization combining ML, LLM, and JIRA metrics")

df = require_projects_df().copy()

if "project_id" not in df.columns and "project_name" in df.columns:
    df["project_id"] = df["project_name"]

if "schedule_performance_index" not in df.columns or "blocker_ratio" not in df.columns:
    try:
        fe = FeatureEngineer()
        df = fe.create_features(df)
    except Exception:
        pass

st.session_state["projects_df"] = df

config_path = root / "config" / "mcda_config.yaml"
default_weights = {k: v["weight"] for k, v in ProjectRanker.DEFAULT_CRITERIA.items()}
weight_profiles: dict = {}
if config_path.exists():
    try:
        with open(config_path, encoding="utf-8") as f:
            mcda_config = yaml.safe_load(f)
        if mcda_config and "criteria" in mcda_config:
            for k, v in mcda_config["criteria"].items():
                if isinstance(v, dict) and "weight" in v:
                    default_weights[k] = v["weight"]
        weight_profiles = mcda_config.get("weight_profiles", {}) if mcda_config else {}
    except Exception:
        pass

profile_names = list(weight_profiles.keys())
selected_profile = st.selectbox(
    "Weight profile (notebook presets)",
    options=["Custom"] + profile_names,
    index=0,
    help="Choose a preset from config/mcda_config.yaml or adjust sliders manually.",
)

if selected_profile != "Custom" and selected_profile in weight_profiles:
    preset = weight_profiles[selected_profile]
    defaults_for_sliders = {k: float(preset.get(k, default_weights[k])) for k in default_weights}
else:
    defaults_for_sliders = {k: float(default_weights[k]) for k in default_weights}

st.markdown("### MCDA Configuration")
st.markdown(
    "Adjust weights for each criterion (must sum to **1.0**). Criteria match notebook 05 / `ProjectRanker`."
)

col1, col2, col3, col4, col5 = st.columns(5)
keys = [
    "ml_risk_score",
    "llm_sentiment_score",
    "schedule_performance_index",
    "blocker_ratio",
    "defect_rate",
]
sliders: dict[str, float] = {}
with col1:
    sliders["ml_risk_score"] = st.slider(
        "ML risk", 0.0, 1.0, defaults_for_sliders.get("ml_risk_score", 0.4), 0.05
    )
with col2:
    sliders["llm_sentiment_score"] = st.slider(
        "LLM sentiment", 0.0, 1.0, defaults_for_sliders.get("llm_sentiment_score", 0.25), 0.05
    )
with col3:
    sliders["schedule_performance_index"] = st.slider(
        "SPI", 0.0, 1.0, defaults_for_sliders.get("schedule_performance_index", 0.15), 0.05
    )
with col4:
    sliders["blocker_ratio"] = st.slider(
        "Blocker ratio", 0.0, 1.0, defaults_for_sliders.get("blocker_ratio", 0.10), 0.05
    )
with col5:
    sliders["defect_rate"] = st.slider(
        "Defect rate", 0.0, 1.0, defaults_for_sliders.get("defect_rate", 0.10), 0.05
    )

total_weight = sum(sliders.values())
if abs(total_weight - 1.0) > 0.01:
    st.warning(f"⚠️ Weights sum to {total_weight:.2f}. Please adjust to sum to 1.0")

criteria = {
    "ml_risk_score": {"weight": sliders["ml_risk_score"], "type": "cost"},
    "llm_sentiment_score": {"weight": sliders["llm_sentiment_score"], "type": "benefit"},
    "schedule_performance_index": {
        "weight": sliders["schedule_performance_index"],
        "type": "benefit",
    },
    "blocker_ratio": {"weight": sliders["blocker_ratio"], "type": "cost"},
    "defect_rate": {"weight": sliders["defect_rate"], "type": "cost"},
}

if st.button("🎯 Calculate Rankings", type="primary"):
    with st.spinner("Calculating MCDA rankings..."):
        try:
            ranker = ProjectRanker(criteria=criteria)
            rankings_df = ranker.rank(df)
            rankings_df = rankings_df.rename(columns={"risk_level": "mcda_risk_level"})

            merge_cols = ["project_id", "mcda_score", "rank", "mcda_risk_level"]
            merged = df.drop(
                columns=["mcda_score", "rank", "mcda_risk_level"], errors="ignore"
            ).merge(
                rankings_df[merge_cols],
                on="project_id",
                how="left",
            )
            st.session_state["projects_df"] = merged
            st.session_state["rankings_df"] = rankings_df
            st.session_state["last_ranker"] = ranker

            try:
                from src.data.snapshot_store import SnapshotStore

                SnapshotStore().save_snapshot(merged, source="mcda")
            except Exception:
                pass

            st.success("✅ Rankings calculated!")
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            import traceback

            st.code(traceback.format_exc())

disp_df = st.session_state.get("projects_df", df)
if "mcda_score" in disp_df.columns:
    st.markdown("---")
    st.markdown("### Project Rankings")

    col1, col2, col3 = st.columns(3)
    with col1:
        high_risk = (
            (disp_df["mcda_risk_level"] == "High").sum()
            if "mcda_risk_level" in disp_df.columns
            else 0
        )
        st.metric("High (MCDA)", int(high_risk))
    with col2:
        med = (
            (disp_df["mcda_risk_level"] == "Medium").sum()
            if "mcda_risk_level" in disp_df.columns
            else 0
        )
        st.metric("Medium (MCDA)", int(med))
    with col3:
        low = (
            (disp_df["mcda_risk_level"] == "Low").sum()
            if "mcda_risk_level" in disp_df.columns
            else 0
        )
        st.metric("Low (MCDA)", int(low))

    try:
        from src.visualization.risk_charts import RiskCharts

        rankings_for_chart = st.session_state.get("rankings_df", disp_df)
        if "mcda_risk_level" in rankings_for_chart.columns:
            tmp = rankings_for_chart.rename(columns={"mcda_risk_level": "risk_level"})
            fig = RiskCharts.risk_score_bar(
                tmp,
                top_n=15,
                name_col="project_name" if "project_name" in tmp.columns else "project_id",
                score_col="mcda_score",
            )
            st.plotly_chart(fig, width="stretch")
    except Exception:
        import plotly.graph_objects as go

        sorted_df = disp_df.sort_values("mcda_score", ascending=True).head(15)
        name_col = "project_name" if "project_name" in sorted_df.columns else "project_id"
        rl = "mcda_risk_level" if "mcda_risk_level" in sorted_df.columns else None
        colors = (
            [
                "#FF4B4B" if l == "High" else "#FFA500" if l == "Medium" else "#00CC66"
                for l in sorted_df[rl]
            ]
            if rl
            else ["#1E88E5"] * len(sorted_df)
        )
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
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Full Rankings Table")
    display_cols = ["rank", "project_name", "mcda_score", "mcda_risk_level"]
    for col in ["completion_rate", "status", "priority"]:
        if col in disp_df.columns:
            display_cols.append(col)
    rankings_table = disp_df[[c for c in display_cols if c in disp_df.columns]].sort_values("rank")

    st.dataframe(
        rankings_table,
        width="stretch",
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

    csv = rankings_table.to_csv(index=False)
    st.download_button(
        label="📥 Export Rankings to CSV",
        data=csv,
        file_name="prism_rankings.csv",
        mime="text/csv",
    )

    ranker = st.session_state.get("last_ranker")
    if ranker is not None:
        with st.expander("Sensitivity analysis (±10% weight perturbation)", expanded=False):
            try:
                sens = ranker.sensitivity_analysis(df, weight_variation=0.10)
                rows = []
                for crit, data in sens.get("criteria_sensitivity", {}).items():
                    rows.append(
                        {
                            "criterion": crit,
                            "weight": data.get("original_weight"),
                            "avg_rank_change": data.get("avg_rank_change"),
                        }
                    )
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    st.caption(
                        "Higher average rank change means rankings are more sensitive to that criterion."
                    )
            except Exception as e:
                st.info(f"Sensitivity analysis unavailable: {e}")

    # ── Risk Response Engine ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🛡️ Recommended Actions (Risk Response Engine)", expanded=True):
        st.markdown(
            "Rule-based PMBOK response strategies derived from project metrics. "
            "These recommendations are available offline — no API key required."
        )

        try:
            from src.risk_response.engine import RiskResponseEngine

            engine = RiskResponseEngine()

            _STRATEGY_COLOUR = {
                "Avoid": "#FF4B4B",
                "Transfer": "#FFA500",
                "Mitigate": "#1E88E5",
                "Accept": "#00CC66",
            }

            top_n = st.slider(
                "Projects to analyse",
                min_value=1,
                max_value=min(20, len(disp_df)),
                value=min(10, len(disp_df)),
                key="rre_top_n",
            )

            target_df = (
                disp_df.sort_values("rank").head(top_n)
                if "rank" in disp_df.columns
                else disp_df.head(top_n)
            )
            name_col = "project_name" if "project_name" in target_df.columns else "project_id"

            any_response = False
            for _, row in target_df.iterrows():
                responses = engine.get_responses(row.to_dict())
                if not responses:
                    continue
                any_response = True
                project_label = row.get(name_col, row.get("project_id", "Unknown"))
                risk_label = row.get("mcda_risk_level", row.get("risk_level", ""))
                colour = _STRATEGY_COLOUR.get("Avoid", "#ccc")
                if risk_label == "High":
                    colour = _STRATEGY_COLOUR["Avoid"]
                elif risk_label == "Medium":
                    colour = _STRATEGY_COLOUR["Mitigate"]

                st.markdown(
                    f"**{project_label}** "
                    f"<span style='background:{colour};color:#fff;padding:2px 8px;"
                    f"border-radius:4px;font-size:0.8em'>{risk_label or 'N/A'}</span>",
                    unsafe_allow_html=True,
                )
                for resp in responses:
                    badge_colour = _STRATEGY_COLOUR.get(resp.strategy, "#888")
                    st.markdown(
                        f"<span style='background:{badge_colour};color:#fff;padding:2px 8px;"
                        f"border-radius:4px;font-size:0.75em;margin-right:6px'>"
                        f"{resp.strategy}</span> "
                        f"**{resp.trigger}** — {resp.recommendation}",
                        unsafe_allow_html=True,
                    )
                st.markdown("")

            if not any_response:
                st.success(
                    "No risk response actions triggered for the selected projects. "
                    "All metrics are within acceptable thresholds."
                )

        except Exception as exc:
            st.warning(f"Risk Response Engine unavailable: {exc}")

else:
    st.info("Click **Calculate Rankings** to generate MCDA-based project prioritization.")
