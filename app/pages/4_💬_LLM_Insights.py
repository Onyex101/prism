"""
LLM Insights Page

AI-extracted risk indicators from project text.
"""

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
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to enable LLM analysis",
    )

    model = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    )

    if st.button("🧠 Run LLM Analysis", type="primary", disabled=not api_key):
        st.info("LLM analysis would run here with the provided API key.")
        st.warning("Demo mode: Showing placeholder results.")

        # Placeholder sentiment scores
        import numpy as np

        np.random.seed(42)

        df["sentiment_score"] = np.random.uniform(-0.5, 0.5, len(df))
        df["sentiment_label"] = df["sentiment_score"].apply(
            lambda x: "positive" if x > 0.1 else ("negative" if x < -0.1 else "neutral")
        )
        st.session_state["projects_df"] = df
        st.success("✅ LLM analysis complete (demo mode)")
        st.rerun()

with col2:
    st.markdown("#### About LLM Analysis")
    st.markdown(
        """
    The LLM analysis:
    - **Sentiment Analysis**: Detects overall tone (positive/negative/neutral)
    - **Risk Indicators**: Extracts specific concerns from text
    - **Risk Categories**: Classifies into technical, resource, schedule, scope, budget
    - **Key Quotes**: Highlights relevant text snippets
    """
    )

# Results section
st.markdown("---")
st.markdown("### Analysis Results")

if "sentiment_score" in df.columns:
    # Summary
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_sentiment = df["sentiment_score"].mean()
        color = "🟢" if avg_sentiment > 0.1 else ("🔴" if avg_sentiment < -0.1 else "🟡")
        st.metric("Avg Sentiment", f"{color} {avg_sentiment:.2f}")

    with col2:
        negative_count = (df["sentiment_label"] == "negative").sum()
        st.metric("Negative Sentiment", negative_count)

    with col3:
        positive_count = (df["sentiment_label"] == "positive").sum()
        st.metric("Positive Sentiment", positive_count)

    # Results table
    display_cols = ["project_name", "sentiment_score", "sentiment_label"]
    if "status_comments" in df.columns:
        display_cols.append("status_comments")

    st.dataframe(
        df[[col for col in display_cols if col in df.columns]].sort_values(
            "sentiment_score", ascending=True
        ),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "Enter your OpenAI API key and click 'Run LLM Analysis' to extract insights from project comments."
    )

# Project detail viewer
st.markdown("---")
st.markdown("### Project Detail View")

selected_project = st.selectbox(
    "Select a project to view details",
    df["project_name"].tolist() if "project_name" in df.columns else df["project_id"].tolist(),
)

if selected_project:
    project_row = df[
        (
            (df["project_name"] == selected_project)
            if "project_name" in df.columns
            else (df["project_id"] == selected_project)
        )
    ].iloc[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Status Comments")
        comments = project_row.get("status_comments", "No comments available")
        st.text_area("", comments, height=200, disabled=True)

    with col2:
        st.markdown("#### LLM Analysis")
        if "sentiment_score" in project_row:
            st.write(f"**Sentiment Score:** {project_row['sentiment_score']:.2f}")
            st.write(f"**Sentiment Label:** {project_row['sentiment_label']}")
        else:
            st.info("Run LLM analysis to see insights.")
