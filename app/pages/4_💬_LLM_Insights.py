"""
LLM Insights Page

AI-extracted risk indicators from project text using PRISM src modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="LLM Insights - PRISM", page_icon="💬", layout="wide")

st.title("💬 LLM Insights")
st.markdown("AI-extracted risk indicators from project comments and updates")

# Check if data is loaded
if "projects_df" not in st.session_state:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/2_📁_Upload_Data.py")
    st.stop()

df = st.session_state["projects_df"]

# Check for text data
if "status_comments" not in df.columns:
    st.warning("⚠️ No 'status_comments' column found. LLM analysis requires text data.")
    st.stop()

# Configuration
st.markdown("### LLM Configuration")

col1, col2 = st.columns([1, 2])

with col1:
    import os

    default_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        value=default_key,
        type="password",
        help="Enter your OpenAI API key. Or set OPENAI_API_KEY in .env",
    )

    model = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    )

    max_projects = st.slider(
        "Max projects to analyze (for cost control)",
        min_value=1,
        max_value=min(50, len(df)),
        value=min(10, len(df)),
    )

    if st.button("🧠 Run LLM Analysis", type="primary", disabled=not api_key):
        with st.spinner("Running LLM analysis (this may take a minute)..."):
            try:
                from src.models.llm import LLMAnalyzer
                from src.models.llm.risk_extractor import RiskExtractor

                analyzer = LLMAnalyzer(api_key=api_key, model=model)
                extractor = RiskExtractor()

                projects_subset = df.head(max_projects)
                projects_list = projects_subset.to_dict(orient="records")

                results = analyzer.analyze_batch(
                    projects_list,
                    text_field="status_comments",
                    name_field="project_name",
                )

                extractor.extract(results)
                llm_df = extractor.to_dataframe()

                # Merge LLM results back into projects_df (use project_name - in both)
                merged = df.copy()
                llm_subset = llm_df[["project_name", "sentiment_score", "sentiment_label"]].drop_duplicates("project_name")
                merged = merged.merge(llm_subset, on="project_name", how="left")

                st.session_state["projects_df"] = merged
                st.session_state["llm_insights"] = results
                st.session_state["llm_analyses_df"] = llm_df

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
    """
    )

# Results section
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
        st.metric("Negative Sentiment", negative_count)

    with col3:
        positive_count = (disp_df["sentiment_label"] == "positive").sum()
        st.metric("Positive Sentiment", positive_count)

    display_cols = ["project_name", "sentiment_score", "sentiment_label"]
    if "status_comments" in disp_df.columns:
        display_cols.append("status_comments")

    st.dataframe(
        disp_df[[col for col in display_cols if col in disp_df.columns]].sort_values(
            "sentiment_score", ascending=True
        ),
        use_container_width=True,
        hide_index=True,
    )

    if "llm_analyses_df" in st.session_state:
        llm_df = st.session_state["llm_analyses_df"]
        if "risk_categories" in llm_df.columns or "risk_indicators_str" in llm_df.columns:
            st.markdown("### Risk Categories & Indicators")
            cat_cols = [c for c in ["project_name", "risk_level", "risk_categories_str", "risk_indicators_str", "summary"] if c in llm_df.columns]
            if cat_cols:
                st.dataframe(llm_df[cat_cols], use_container_width=True, hide_index=True)

    try:
        from src.visualization.risk_charts import RiskCharts

        fig = RiskCharts.sentiment_distribution(disp_df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

else:
    st.info(
        "Enter your OpenAI API key and click 'Run LLM Analysis' to extract insights from project comments."
    )

# Project detail viewer
st.markdown("---")
st.markdown("### Project Detail View")

name_col = "project_name" if "project_name" in df.columns else "project_id"
options = df[name_col].tolist() if name_col in df.columns else []
selected_project = st.selectbox("Select a project to view details", options) if options else None

if selected_project and options:
    project_row = df[
        (df[name_col] == selected_project)
    ].iloc[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Status Comments")
        comments = project_row.get("status_comments", "No comments available")
        if pd.isna(comments):
            comments = "No comments available"
        st.text_area("", str(comments), height=200, disabled=True)

    with col2:
        st.markdown("#### LLM Analysis")
        if "sentiment_score" in project_row and pd.notna(project_row.get("sentiment_score")):
            st.write(f"**Sentiment Score:** {project_row['sentiment_score']:.2f}")
            st.write(f"**Sentiment Label:** {project_row.get('sentiment_label', 'N/A')}")
            if "llm_insights" in st.session_state:
                insight = next(
                    (i for i in st.session_state["llm_insights"] if i.get("project_name") == selected_project or i.get("project_id") == selected_project),
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
