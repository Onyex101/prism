"""
LLM Insights Page

AI-extracted risk indicators (aligned with notebook 04: stratified sampling, non-empty comments).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st

from app.bootstrap import init_page
from app.components import require_projects_df
from app.components.openai_models import OPENAI_MODEL_OPTIONS

init_page()

from src.models.llm import LLMAnalyzer
from src.models.llm.risk_extractor import RiskExtractor


def _select_llm_projects_subset(
    df: pd.DataFrame,
    max_projects: int,
    stratify: bool,
) -> pd.DataFrame:
    """
    Rows with non-empty ``status_comments``, capped by ``max_projects``.

    When ``stratify`` is True, sample up to ``max_projects // 3`` from each
    risk band (High / Medium / Low), then take at most ``max_projects`` total.
    """
    base = df.copy()
    sc = base["status_comments"]
    mask = sc.notna() & (sc.astype(str).str.strip() != "")
    base = base.loc[mask].copy()
    if len(base) == 0:
        return base
    if stratify and "risk_level" in base.columns:
        tiers = ["High", "Medium", "Low"]
        per_tier = max(1, max_projects // len(tiers))
        parts = []
        for t in tiers:
            rl = base["risk_level"].astype(str).str.strip().str.title()
            sub = base[rl == t]
            if len(sub) == 0:
                continue
            parts.append(sub.sample(n=min(per_tier, len(sub)), random_state=42))
        if not parts:
            return base.head(0)
        return pd.concat(parts, axis=0).head(max_projects)
    return base.head(max_projects)


st.title("💬 LLM Insights")
st.markdown("AI-extracted risk indicators from project comments and updates")

df = require_projects_df()

if "status_comments" not in df.columns:
    st.warning("⚠️ No 'status_comments' column found. LLM analysis requires text data.")
    st.stop()

st.markdown("### LLM Configuration")

col1, col2 = st.columns([1, 2])

with col1:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.warning(
            "OPENAI_API_KEY not set. Add it to .env or Streamlit secrets to run new LLM analyses. "
            "Cached results below remain visible."
        )

    _idx = OPENAI_MODEL_OPTIONS.index(st.session_state.openai_model)
    model = st.selectbox("Model", OPENAI_MODEL_OPTIONS, index=_idx)
    st.session_state.openai_model = model

    _cap = min(50, max(1, len(df)))
    max_projects = st.slider(
        "Max projects to analyze (for cost control)",
        min_value=1,
        max_value=_cap,
        value=_cap,
        help="By default this is **all** loaded projects (up to 50). Lower it to save API cost.",
    )

    stratify = st.checkbox(
        "Stratified sampling by risk_level (notebook 04)",
        value="risk_level" in df.columns,
        disabled="risk_level" not in df.columns,
    )

    _preview = _select_llm_projects_subset(df, max_projects, stratify)
    _per_tier = max(1, max_projects // 3) if stratify and "risk_level" in df.columns else None
    st.caption(
        f"**Preview:** {_preview.shape[0]} project(s) will be analyzed "
        f"(rows with non-empty comments, capped at {max_projects}). "
        + (
            f"Stratified: up to {_per_tier} per tier (High/Medium/Low), total ≤ {max_projects}. "
            if _per_tier is not None
            else ""
        )
        + "This is **not** hardcoded to 10 — increase **Max projects** if you see fewer than expected."
    )

    if st.button("🧠 Run LLM Analysis", type="primary", disabled=not api_key):
        try:
            projects_subset = _select_llm_projects_subset(df, max_projects, stratify)

            if len(projects_subset) == 0:
                st.error("No rows with non-empty status_comments.")
                st.stop()

            projects_list = projects_subset.to_dict(orient="records")

            progress_bar = st.progress(0.0)
            status_line = st.empty()
            log_exp = st.expander("Live progress (per project)", expanded=True)
            log_box = log_exp.empty()
            log_lines: list[str] = []

            def _on_llm_progress(done: int, total: int, name: str, result: dict) -> None:
                progress_bar.progress(min(1.0, done / total) if total else 1.0)
                rl = str(result.get("risk_level", "?"))
                line = f"Analyzed **{name}**: `{rl}`"
                log_lines.append(line)
                log_box.markdown("\n\n".join(log_lines[-30:]))
                status_line.markdown(f"**{done} / {total}** — `{name}` → **{rl}**")

            analyzer = LLMAnalyzer(api_key=api_key, model=model)
            extractor = RiskExtractor()

            results = analyzer.analyze_batch(
                projects_list,
                text_field="status_comments",
                name_field="project_name",
                on_progress=_on_llm_progress,
            )

            status_line.markdown("Merging risk extraction…")
            extractor.extract(results)
            llm_df = extractor.to_dataframe()

            merged = df.copy()
            llm_subset = llm_df[
                ["project_name", "sentiment_score", "sentiment_label"]
            ].drop_duplicates("project_name")
            merged = merged.merge(llm_subset, on="project_name", how="left")

            st.session_state["projects_df"] = merged
            st.session_state["llm_insights"] = results
            st.session_state["llm_analyses_df"] = llm_df

            progress_bar.progress(1.0)
            status_line.markdown(f"**Done.** Processed **{len(results)}** project(s).")
            st.success(f"✅ LLM analysis complete for {len(results)} projects!")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"LLM analysis failed: {e}")
            import traceback

            st.code(traceback.format_exc())

with col2:
    st.markdown("#### About LLM Analysis")
    st.markdown(
        """
    The LLM analysis uses OpenAI to:
    - **Sentiment Analysis**: Detects overall tone (positive/negative/neutral)
    - **Risk Indicators**: Extracts specific concerns from text
    - **Risk Categories**: Classifies into technical, resource, schedule, scope, budget
    - **Key Quotes**: Highlights relevant text snippets

    Rows with empty `status_comments` are excluded before batching (notebook 04).
    """
    )

st.markdown("---")
st.markdown("### Analysis Results")

disp_df = st.session_state.get("projects_df", df)
if "sentiment_score" in disp_df.columns:
    disp_df = disp_df.dropna(subset=["sentiment_score"])
if "sentiment_score" in disp_df.columns and len(disp_df) > 0:
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_sentiment = disp_df["sentiment_score"].mean()
        color = "🟢" if avg_sentiment > 0.1 else ("🔴" if avg_sentiment < -0.1 else "🟡")
        st.metric("Avg Sentiment", f"{color} {avg_sentiment:.2f}")

    with col2:
        negative_count = (disp_df["sentiment_label"] == "negative").sum()
        st.metric("Negative Sentiment", int(negative_count))

    with col3:
        positive_count = (disp_df["sentiment_label"] == "positive").sum()
        st.metric("Positive Sentiment", int(positive_count))

    display_cols = ["project_name", "sentiment_score", "sentiment_label"]
    if "status_comments" in disp_df.columns:
        display_cols.append("status_comments")

    st.dataframe(
        disp_df[[col for col in display_cols if col in disp_df.columns]].sort_values(
            "sentiment_score", ascending=True
        ),
        width="stretch",
        hide_index=True,
    )

    if "llm_analyses_df" in st.session_state:
        llm_df = st.session_state["llm_analyses_df"]
        if "risk_categories" in llm_df.columns or "risk_indicators_str" in llm_df.columns:
            st.markdown("### Risk Categories & Indicators")
            cat_cols = [
                c
                for c in [
                    "project_name",
                    "risk_level",
                    "risk_categories_str",
                    "risk_indicators_str",
                    "summary",
                ]
                if c in llm_df.columns
            ]
            if cat_cols:
                st.dataframe(llm_df[cat_cols], width="stretch", hide_index=True)

    try:
        from src.visualization.risk_charts import RiskCharts

        fig = RiskCharts.sentiment_distribution(disp_df)
        st.plotly_chart(fig, width="stretch")
    except Exception:
        pass

else:
    st.info(
        "Configure your OpenAI API key (environment secrets recommended), then click "
        "**Run LLM Analysis** to extract insights from project comments."
    )

st.markdown("---")
st.markdown("### Project Detail View")

name_col = "project_name" if "project_name" in df.columns else "project_id"
options = df[name_col].tolist() if name_col in df.columns else []
selected_project = st.selectbox("Select a project to view details", options) if options else None

if selected_project and options:
    project_row = df[(df[name_col] == selected_project)].iloc[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Status Comments")
        comments = project_row.get("status_comments", "No comments available")
        if pd.isna(comments):
            comments = "No comments available"
        st.text_area(
            "Status comments",
            str(comments),
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("#### LLM Analysis")
        if "sentiment_score" in project_row and pd.notna(project_row.get("sentiment_score")):
            st.write(f"**Sentiment Score:** {project_row['sentiment_score']:.2f}")
            st.write(f"**Sentiment Label:** {project_row.get('sentiment_label', 'N/A')}")
            if "llm_insights" in st.session_state:
                insight = next(
                    (
                        i
                        for i in st.session_state["llm_insights"]
                        if i.get("project_name") == selected_project
                        or i.get("project_id") == selected_project
                    ),
                    None,
                )
                if insight:
                    st.write("**Risk Level:**", insight.get("risk_level", "N/A"))
                    if insight.get("risk_categories"):
                        st.write("**Categories:**", ", ".join(insight["risk_categories"]))
                    if insight.get("summary"):
                        st.write("**Summary:**", insight["summary"])
        else:
            st.info("Run LLM analysis to see insights.")
