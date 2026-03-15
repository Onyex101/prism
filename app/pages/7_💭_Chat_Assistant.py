"""
Chat Assistant Page

AI-powered Q&A about project risks using PRISM ChatAssistant.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Chat Assistant - PRISM", page_icon="💭", layout="wide")

st.title("💭 Chat Assistant")
st.markdown("Ask questions about your project portfolio and risk analysis")

# Initialize chat history
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

Enter your OpenAI API key in the sidebar and load project data to get started.
Try asking: "Which projects are highest risk?" or "Why is Project X high risk?"
""",
        }
    ]

# Sidebar with context
with st.sidebar:
    st.markdown("### Chat Configuration")

    default_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        value=default_key,
        type="password",
        help="Required for AI responses. Or set OPENAI_API_KEY in .env",
    )

    if "projects_df" in st.session_state:
        df = st.session_state["projects_df"]
        st.markdown("---")
        st.markdown("### Current Context")
        st.write(f"📊 {len(df)} projects loaded")

        if "risk_level" in df.columns:
            high = (df["risk_level"] == "High").sum()
            st.write(f"🔴 {high} high-risk projects")

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
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def generate_response(prompt: str, api_key: str | None) -> str:
    """Generate a response using ChatAssistant when API key is available."""
    if "projects_df" not in st.session_state:
        return "⚠️ No project data loaded. Please upload data first to get meaningful answers."

    df = st.session_state["projects_df"]
    rankings_df = st.session_state.get("rankings_df")
    llm_insights = st.session_state.get("llm_insights")

    if api_key:
        try:
            from src.chat import ChatAssistant

            if "chat_assistant" not in st.session_state or st.session_state.get("chat_api_key") != api_key:
                st.session_state.chat_assistant = ChatAssistant(api_key=api_key)
                st.session_state.chat_api_key = api_key

            assistant = st.session_state.chat_assistant
            assistant.set_context(
                projects_df=df,
                rankings_df=rankings_df,
                llm_insights=llm_insights,
            )
            return assistant.chat(prompt)

        except ImportError as e:
            return f"OpenAI package not installed: {e}"
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."

    # Fallback when no API key - simple keyword-based responses
    prompt_lower = prompt.lower()

    if "highest risk" in prompt_lower or "top risk" in prompt_lower:
        if "mcda_score" in df.columns or "risk_score" in df.columns:
            score_col = "mcda_score" if "mcda_score" in df.columns else "risk_score"
            top = df.nlargest(3, score_col) if score_col == "risk_score" else df.nsmallest(3, score_col)
            response = "**Top 3 High-Risk Projects:**\n\n"
            for _, row in top.iterrows():
                name = row.get("project_name", row.get("project_id", "Unknown"))
                score = row[score_col]
                response += f"- **{name}**: Score {score:.2f}\n"
            return response
        else:
            return "Risk analysis hasn't been run yet. Go to ML Analysis or Rankings to generate scores."

    elif "over budget" in prompt_lower:
        if "budget" in df.columns and "spent" in df.columns:
            over = df[df["spent"] > df["budget"]]
            if len(over) > 0:
                response = f"**{len(over)} projects are over budget:**\n\n"
                for _, row in over.iterrows():
                    name = row.get("project_name", row.get("project_id", "Unknown"))
                    variance = (row["spent"] - row["budget"]) / row["budget"] * 100
                    response += f"- **{name}**: {variance:+.1f}% over\n"
                return response
            else:
                return "✅ No projects are currently over budget."
        else:
            return "Budget data not available."

    elif "how many" in prompt_lower and "project" in prompt_lower:
        return f"You have **{len(df)} projects** loaded in the current analysis."

    elif "risk factor" in prompt_lower or "main risk" in prompt_lower:
        return """**Key risk factors analyzed by PRISM:**

1. **Completion Rate** - Projects falling behind schedule
2. **Budget Variance** - Cost overruns or underutilization
3. **Team Turnover** - Instability in project teams
4. **Schedule Performance** - Velocity and timeline adherence
5. **Sentiment Analysis** - Team morale from comments

Run the ML Analysis for detailed feature importance."""

    else:
        return "For AI-powered responses, please enter your OpenAI API key in the sidebar.\n\nTry asking:\n- 'Which projects are highest risk?'\n- 'Show me projects over budget'\n- 'How many projects do I have?'"


# Handle pending question from sidebar (api_key from sidebar block above)
if "pending_question" in st.session_state:
    prompt = st.session_state.pending_question
    del st.session_state.pending_question

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, api_key if api_key else None)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about your projects..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, api_key if api_key else None)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
