"""
Chat Assistant Page

AI-powered Q&A about project risks.
"""

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

Try asking: "Which projects are highest risk?" or "Why is Project X high risk?"
""",
        }
    ]

# Sidebar with context
with st.sidebar:
    st.markdown("### Chat Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for AI responses",
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
        "What are the main risk factors?",
        "Show me projects over budget",
        "Compare the top 3 risky projects",
    ]

    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle pending question from sidebar
if "pending_question" in st.session_state:
    prompt = st.session_state.pending_question
    del st.session_state.pending_question

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
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
        response = generate_response(prompt, api_key if api_key else None)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


def generate_response(prompt: str, api_key: str | None) -> str:
    """Generate a response to the user's question."""

    # Check if data is loaded
    if "projects_df" not in st.session_state:
        return "⚠️ No project data loaded. Please upload data first to get meaningful answers."

    df = st.session_state["projects_df"]

    # Simple keyword-based responses for demo
    prompt_lower = prompt.lower()

    if "highest risk" in prompt_lower or "top risk" in prompt_lower:
        if "mcda_score" in df.columns or "risk_score" in df.columns:
            score_col = "mcda_score" if "mcda_score" in df.columns else "risk_score"
            top = df.nsmallest(3, score_col)
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
        if api_key:
            return f"I understand you're asking about: '{prompt}'\n\nFor detailed AI-powered responses, the OpenAI integration would provide contextual answers based on your project data."
        else:
            return f"I can help with that! For detailed AI-powered responses, please enter your OpenAI API key in the sidebar.\n\nIn the meantime, try asking:\n- 'Which projects are highest risk?'\n- 'Show me projects over budget'\n- 'How many projects do I have?'"
