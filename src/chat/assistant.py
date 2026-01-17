"""
Chat Assistant Module
=====================

This module provides a conversational interface for querying project risk data.

The assistant can answer questions about projects, explain risk predictions,
and provide recommendations based on the analysis results.

Example:
    >>> from src.chat.assistant import ChatAssistant
    >>> assistant = ChatAssistant(api_key="sk-...")
    >>> assistant.set_context(projects_df, rankings_df)
    >>> response = assistant.chat("Which projects are highest risk?")
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
    """
    Conversational assistant for project risk analysis.

    This class provides a natural language interface for exploring
    risk analysis results. It maintains conversation history for
    multi-turn dialogues.

    :cvar SYSTEM_PROMPT: Default system prompt for the assistant.
    :vartype SYSTEM_PROMPT: str

    :ivar api_key: OpenAI API key.
    :vartype api_key: Optional[str]
    :ivar model: Model name to use.
    :vartype model: str
    :ivar client: OpenAI client instance.
    :vartype client: Optional[OpenAI]
    :ivar conversation_history: List of conversation messages.
    :vartype conversation_history: list[dict]
    :ivar context_data: Data context for answering questions.
    :vartype context_data: dict

    Example:
        >>> assistant = ChatAssistant(api_key="sk-...")
        >>> assistant.set_context(projects_df, rankings_df)
        >>> response = assistant.chat("Why is Project X high risk?")
    """

    SYSTEM_PROMPT: str = """You are PRISM, an AI assistant for software project risk analysis. 
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
    ) -> None:
        """
        Initialize the chat assistant.

        :param api_key: OpenAI API key.
        :type api_key: Optional[str]
        :param model: Model to use for chat.
        :type model: str
        :raises ImportError: If OpenAI package not installed.
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

        :param projects_df: Project data.
        :type projects_df: Optional[pd.DataFrame]
        :param rankings_df: MCDA rankings.
        :type rankings_df: Optional[pd.DataFrame]
        :param llm_insights: LLM analysis results.
        :type llm_insights: Optional[list[dict]]

        Example:
            >>> assistant.set_context(
            ...     projects_df=df,
            ...     rankings_df=rankings,
            ...     llm_insights=llm_results
            ... )
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

        :param user_message: User's question/message.
        :type user_message: str
        :return: Assistant's response.
        :rtype: str
        :raises ValueError: If OpenAI client not initialized.

        Example:
            >>> response = assistant.chat("What are the top 3 risk factors?")
            >>> print(response)
        """
        self._validate_client()

        context_str = self._build_context_message()
        self._add_to_history("user", user_message)

        try:
            response = self._call_api(context_str)
            self._add_to_history("assistant", response)
            return response
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"I encountered an error: {str(e)}. Please try again."

    def _validate_client(self) -> None:
        """
        Validate that OpenAI client is initialized.

        :raises ValueError: If client not initialized.
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Provide API key.")

    def _add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        :param role: Message role ("user" or "assistant").
        :type role: str
        :param content: Message content.
        :type content: str
        """
        self.conversation_history.append({"role": role, "content": content})

    def _call_api(self, context_str: str) -> str:
        """
        Call the OpenAI API.

        :param context_str: Context string to include.
        :type context_str: str
        :return: Assistant response.
        :rtype: str
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT + "\n\n" + context_str},
        ]
        messages.extend(self.conversation_history[-10:])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )

        return response.choices[0].message.content

    def _build_context_message(self) -> str:
        """
        Build context message from available data.

        :return: Formatted context string.
        :rtype: str
        """
        parts = ["Current Analysis Context:"]

        if "project_count" in self.context_data:
            parts.append(f"- Total projects analyzed: {self.context_data['project_count']}")

        if "high_risk_count" in self.context_data:
            parts.append(f"- High-risk projects: {self.context_data['high_risk_count']}")

        if "rankings" in self.context_data:
            parts.append(self._format_top_projects())

        if "insights" in self.context_data:
            parts.append(
                f"\n- LLM insights available for " f"{len(self.context_data['insights'])} projects"
            )

        return "\n".join(parts)

    def _format_top_projects(self) -> str:
        """
        Format top risk projects for context.

        :return: Formatted string.
        :rtype: str
        """
        top_5 = self.context_data["rankings"][:5]
        lines = ["\nTop 5 Risk Projects:"]

        for proj in top_5:
            name = proj.get("project_name", proj.get("project_id", "Unknown"))
            score = proj.get("mcda_score", 0)
            level = proj.get("risk_level", "Unknown")
            lines.append(f"  - {name}: Score {score:.2f} ({level})")

        return "\n".join(lines)

    def get_project_info(self, project_name: str) -> str:
        """
        Get information about a specific project.

        :param project_name: Name or ID of project.
        :type project_name: str
        :return: Formatted project information.
        :rtype: str

        Example:
            >>> info = assistant.get_project_info("Mobile App")
            >>> print(info)
        """
        if "rankings" not in self.context_data:
            return "No project data available."

        project_name_lower = project_name.lower()

        for proj in self.context_data.get("rankings", []):
            name = proj.get("project_name", proj.get("project_id", ""))
            if project_name_lower in name.lower():
                return self._format_project_info(name, proj)

        return f"Project '{project_name}' not found in current analysis."

    def _format_project_info(self, name: str, proj: dict) -> str:
        """
        Format project information.

        :param name: Project name.
        :type name: str
        :param proj: Project data.
        :type proj: dict
        :return: Formatted string.
        :rtype: str
        """
        return "\n".join(
            [
                f"**{name}**",
                f"- MCDA Score: {proj.get('mcda_score', 'N/A'):.2f}",
                f"- Risk Level: {proj.get('risk_level', 'N/A')}",
                f"- Rank: {proj.get('rank', 'N/A')}",
            ]
        )

    def clear_history(self) -> None:
        """
        Clear conversation history.

        Example:
            >>> assistant.clear_history()
        """
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_suggested_questions(self) -> list[str]:
        """
        Get list of suggested questions.

        :return: List of suggested question strings.
        :rtype: list[str]

        Example:
            >>> for question in assistant.get_suggested_questions():
            ...     print(f"- {question}")
        """
        return [
            "Which projects are highest risk?",
            "Why is the top-ranked project considered high risk?",
            "What can I do to reduce risk in the portfolio?",
            "Are there any projects with team morale issues?",
            "Show me projects that are over budget.",
            "Compare the top 3 risk projects.",
        ]
