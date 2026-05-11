"""
Chat Assistant Page

AI-powered Q&A about project risks using PRISM ChatAssistant.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_p = Path(__file__).resolve().parent
_repo_root = _p.parent.parent if _p.name == "pages" else _p.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from app.bootstrap import init_page

init_page()

# Same model as LLM Insights (session_state.openai_model, set in init_page / LLM page).
chat_model = st.session_state.openai_model

st.title("💭 Chat Assistant")
st.markdown("Ask questions about your project portfolio and risk analysis")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """👋 Hello! I'm PRISM, your AI assistant for project risk analysis.

I can help you understand:
- Why a project is flagged as high risk
- What factors contribute to risk scores
- Recommendations to reduce risk
- Comparisons between projects

Load project data to get started.
Try asking: "Which projects are highest risk?" or "Why is Project X high risk?"
""",
        }
    ]

with st.sidebar:
    st.markdown("### Chat Configuration")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        st.success("OpenAI API key configured via environment.")
    else:
        st.warning("OPENAI_API_KEY not set. Chat will fall back to keyword-only responses.")

    st.caption(f"**LLM model:** `{chat_model}` — same as **LLM Insights** (change it there).")

    if "projects_df" in st.session_state:
        df_ctx = st.session_state["projects_df"]
        st.markdown("---")
        st.markdown("### Current Context")
        st.write(f"📊 {len(df_ctx)} projects loaded")

        level_col = "mcda_risk_level" if "mcda_risk_level" in df_ctx.columns else "risk_level"
        if level_col in df_ctx.columns:
            high = (df_ctx[level_col] == "High").sum()
            st.write(f"🔴 {high} high-risk projects (by **{level_col}**)")

    st.markdown("---")
    st.markdown("### Suggested Questions")

    suggestions = [
        "Which projects are highest risk?",
        "Why is the top-ranked project considered high risk?",
        "What can I do to reduce risk in the portfolio?",
        "Are there any projects with team morale issues?",
        "Show me projects that are over budget.",
        "Compare the top 3 risk projects.",
    ]

    for suggestion in suggestions:
        if st.button(suggestion, width="stretch"):
            st.session_state.pending_question = suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def generate_response(
    prompt: str,
    api_key: str | None,
    *,
    chat_model: str | None = None,
) -> str:
    """Delegate to :mod:`app.chat_helpers` (LLM vs keyword fallback)."""
    if "projects_df" not in st.session_state:
        return "⚠️ No project data loaded. Please upload data first to get meaningful answers."

    from app.chat_helpers import build_chat_response

    df = st.session_state["projects_df"]
    return build_chat_response(
        prompt,
        api_key if api_key else None,
        df,
        rankings_df=st.session_state.get("rankings_df"),
        llm_insights=st.session_state.get("llm_insights"),
        session_state=st.session_state,
        chat_model=chat_model,
    )


if "pending_question" in st.session_state:
    prompt = st.session_state.pending_question
    del st.session_state.pending_question

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(
                prompt,
                api_key if api_key else None,
                chat_model=chat_model,
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if prompt := st.chat_input("Ask about your projects..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(
                prompt,
                api_key if api_key else None,
                chat_model=chat_model,
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
