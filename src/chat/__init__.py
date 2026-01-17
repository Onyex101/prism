"""
Chat Module
===========

This module provides conversational AI functionality for querying
risk analysis results.

Classes:
    ChatAssistant: Conversational assistant for project risk analysis.

Example:
    >>> from src.chat import ChatAssistant
    >>> assistant = ChatAssistant(api_key="sk-...")
    >>> assistant.set_context(projects_df, rankings_df)
    >>> response = assistant.chat("Which projects need attention?")
"""

from src.chat.assistant import ChatAssistant

__all__ = [
    "ChatAssistant",
]
