"""
Chat Assistant response logic (LLM path vs keyword heuristics).

Separated from the Streamlit page for single-responsibility and easier testing.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd

from app.components.openai_models import DEFAULT_OPENAI_MODEL


def run_llm_chat(
    prompt: str,
    api_key: str,
    df: pd.DataFrame,
    rankings_df: Optional[pd.DataFrame],
    llm_insights: Any,
    session_state: Any,
    *,
    model: Optional[str] = None,
) -> str:
    """Call :class:`src.chat.ChatAssistant` with portfolio context."""
    from src.chat import ChatAssistant

    resolved_model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    cache_ok = (
        "chat_assistant" in session_state
        and session_state.get("chat_api_key") == api_key
        and session_state.get("chat_model") == resolved_model
    )
    if not cache_ok:
        session_state.chat_assistant = ChatAssistant(api_key=api_key, model=resolved_model)
        session_state.chat_api_key = api_key
        session_state.chat_model = resolved_model

    assistant = session_state.chat_assistant
    assistant.set_context(
        projects_df=df,
        rankings_df=rankings_df,
        llm_insights=llm_insights,
    )
    return assistant.chat(prompt)


def keyword_fallback_response(prompt: str, df: pd.DataFrame) -> str:
    """Offline answers when no API key is set (simple keyword routing)."""
    prompt_lower = prompt.lower()

    if "highest risk" in prompt_lower or "top risk" in prompt_lower:
        if "mcda_score" in df.columns or "risk_score" in df.columns:
            score_col = "mcda_score" if "mcda_score" in df.columns else "risk_score"
            top = (
                df.nlargest(3, score_col)
                if score_col == "risk_score"
                else df.nsmallest(3, score_col)
            )
            response = "**Top 3 High-Risk Projects:**\n\n"
            for _, row in top.iterrows():
                name = row.get("project_name", row.get("project_id", "Unknown"))
                score = row[score_col]
                response += f"- **{name}**: Score {score:.2f}\n"
            return response
        return (
            "Risk analysis hasn't been run yet. Go to ML Analysis or Rankings to generate scores."
        )

    if "over budget" in prompt_lower:
        if "budget" in df.columns and "spent" in df.columns:
            over = df[df["spent"] > df["budget"]]
            if len(over) > 0:
                response = f"**{len(over)} projects are over budget:**\n\n"
                for _, row in over.iterrows():
                    name = row.get("project_name", row.get("project_id", "Unknown"))
                    variance = (row["spent"] - row["budget"]) / row["budget"] * 100
                    response += f"- **{name}**: {variance:+.1f}% over\n"
                return response
            return "✅ No projects are currently over budget."
        return "Budget data not available."

    if "how many" in prompt_lower and "project" in prompt_lower:
        return f"You have **{len(df)} projects** loaded in the current analysis."

    if "risk factor" in prompt_lower or "main risk" in prompt_lower:
        return """**Key risk factors analyzed by PRISM:**

1. **Completion Rate** - Projects falling behind schedule
2. **Budget Variance** - Cost overruns or underutilization
3. **Team Turnover** - Instability in project teams
4. **Schedule Performance** - Velocity and timeline adherence
5. **Sentiment Analysis** - Team morale from comments
6. **JIRA quality metrics** - defect_rate, blocker_ratio (MCDA)

Run the ML Analysis for detailed feature importance."""

    return (
        "For AI-powered responses, please enter your OpenAI API key in the sidebar.\n\n"
        "Try asking:\n- 'Which projects are highest risk?'\n"
        "- 'Show me projects over budget'\n"
        "- 'How many projects do I have?'"
    )


def build_chat_response(
    prompt: str,
    api_key: Optional[str],
    df: pd.DataFrame,
    *,
    rankings_df: Optional[pd.DataFrame],
    llm_insights: Any,
    session_state: Any,
    chat_model: Optional[str] = None,
) -> str:
    """
    Full orchestration: LLM when key is present, else keyword fallback.

    Call from the Chat page after confirming ``projects_df`` exists.
    """
    if not api_key:
        return keyword_fallback_response(prompt, df)

    try:
        return run_llm_chat(
            prompt,
            api_key,
            df,
            rankings_df,
            llm_insights,
            session_state,
            model=chat_model,
        )
    except ImportError as e:
        return f"OpenAI package not installed: {e}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try again."
