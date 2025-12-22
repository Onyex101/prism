"""
Chat Assistant Module

Provides conversational interface for querying project risk data.
"""

from typing import Any, Optional

import pandas as pd
from loguru import logger

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ChatAssistant:
    """Conversational assistant for project risk analysis."""

    SYSTEM_PROMPT = """You are PRISM, an AI assistant for software project risk analysis. 
You help project managers understand risk predictions and make data-driven decisions.

You have access to:
- ML model predictions and feature importance
- LLM-extracted risk indicators from project comments
- MCDA rankings comparing multiple projects

Be helpful, concise, and always back up your answers with data from the analysis.
If you don't have information about something, say so clearly.

When discussing risk:
- High risk (score > 0.6): Immediate attention needed
- Medium risk (0.3-0.6): Monitor closely
- Low risk (< 0.3): On track

Always be constructive and suggest actionable steps when discussing problems."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
    ):
        """
        Initialize the chat assistant.

        Args:
            api_key: OpenAI API key
            model: Model to use for chat
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.api_key = api_key
        self.model = model
        self.client: Optional[OpenAI] = None
        self.conversation_history: list[dict] = []
        self.context_data: dict = {}

        if api_key:
            self.client = OpenAI(api_key=api_key)

    def set_context(
        self,
        projects_df: Optional[pd.DataFrame] = None,
        rankings_df: Optional[pd.DataFrame] = None,
        llm_insights: Optional[list[dict]] = None,
    ) -> None:
        """
        Set the context data for the assistant.

        Args:
            projects_df: Project data
            rankings_df: MCDA rankings
            llm_insights: LLM analysis results
        """
        if projects_df is not None:
            self.context_data["projects"] = projects_df.to_dict(orient="records")
            self.context_data["project_count"] = len(projects_df)

        if rankings_df is not None:
            self.context_data["rankings"] = rankings_df.to_dict(orient="records")
            self.context_data["high_risk_count"] = len(
                rankings_df[rankings_df["risk_level"] == "High"]
            )

        if llm_insights is not None:
            self.context_data["insights"] = llm_insights

        logger.info("Chat context updated")

    def chat(self, user_message: str) -> str:
        """
        Process a chat message and return response.

        Args:
            user_message: User's question/message

        Returns:
            Assistant's response
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Provide API key.")

        # Build context message
        context_str = self._build_context_message()

        # Add user message to history
        self.conversation_history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # Prepare messages for API
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT + "\n\n" + context_str},
        ]
        messages.extend(self.conversation_history[-10:])  # Last 10 messages

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )

            assistant_message = response.choices[0].message.content

            # Add to history
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                }
            )

            return assistant_message

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"I encountered an error: {str(e)}. Please try again."

    def _build_context_message(self) -> str:
        """Build context message from available data."""
        parts = ["Current Analysis Context:"]

        if "project_count" in self.context_data:
            parts.append(f"- Total projects analyzed: {self.context_data['project_count']}")

        if "high_risk_count" in self.context_data:
            parts.append(f"- High-risk projects: {self.context_data['high_risk_count']}")

        if "rankings" in self.context_data:
            top_5 = self.context_data["rankings"][:5]
            parts.append("\nTop 5 Risk Projects:")
            for proj in top_5:
                name = proj.get("project_name", proj.get("project_id", "Unknown"))
                score = proj.get("mcda_score", 0)
                level = proj.get("risk_level", "Unknown")
                parts.append(f"  - {name}: Score {score:.2f} ({level})")

        if "insights" in self.context_data:
            parts.append(
                f"\n- LLM insights available for {len(self.context_data['insights'])} projects"
            )

        return "\n".join(parts)

    def get_project_info(self, project_name: str) -> str:
        """
        Get information about a specific project.

        Args:
            project_name: Name or ID of project

        Returns:
            Formatted project information
        """
        if "rankings" not in self.context_data:
            return "No project data available."

        project_name_lower = project_name.lower()

        for proj in self.context_data.get("rankings", []):
            name = proj.get("project_name", proj.get("project_id", ""))
            if project_name_lower in name.lower():
                info = [f"**{name}**"]
                info.append(f"- MCDA Score: {proj.get('mcda_score', 'N/A'):.2f}")
                info.append(f"- Risk Level: {proj.get('risk_level', 'N/A')}")
                info.append(f"- Rank: {proj.get('rank', 'N/A')}")
                return "\n".join(info)

        return f"Project '{project_name}' not found in current analysis."

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_suggested_questions(self) -> list[str]:
        """Get list of suggested questions."""
        return [
            "Which projects are highest risk?",
            "Why is the top-ranked project considered high risk?",
            "What can I do to reduce risk in the portfolio?",
            "Are there any projects with team morale issues?",
            "Show me projects that are over budget.",
            "Compare the top 3 risk projects.",
        ]
