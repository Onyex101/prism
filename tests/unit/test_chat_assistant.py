"""
Tests for the ChatAssistant module.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_rankings_df():
    """Sample rankings for ChatAssistant context."""
    return pd.DataFrame(
        {
            "project_id": ["P1", "P2", "P3"],
            "project_name": ["Alpha", "Beta", "Gamma"],
            "mcda_score": [0.35, 0.55, 0.80],
            "rank": [1, 2, 3],
            "risk_level": ["High", "Medium", "Low"],
        }
    )


@pytest.fixture
def sample_projects_df():
    """Sample projects for context."""
    return pd.DataFrame(
        {
            "project_id": ["P1", "P2", "P3"],
            "project_name": ["Alpha", "Beta", "Gamma"],
        }
    )


class TestChatAssistant:
    """Test suite for ChatAssistant."""

    @pytest.fixture
    def skip_if_no_openai(self):
        """Skip if OpenAI not available."""
        try:
            from openai import OpenAI
        except ImportError:
            pytest.skip("OpenAI not installed")

    def test_set_context_with_missing_risk_level(self):
        """Test set_context handles rankings without risk_level column."""
        from unittest.mock import patch

        with patch("src.chat.assistant.OPENAI_AVAILABLE", True):
            with patch("src.chat.assistant.OpenAI"):
                try:
                    from src.chat.assistant import ChatAssistant

                    assistant = ChatAssistant(api_key="dummy-key")
                    rankings_no_risk = pd.DataFrame(
                        {
                            "project_id": ["P1"],
                            "project_name": ["A"],
                            "mcda_score": [0.5],
                            "rank": [1],
                        }
                    )
                    assistant.set_context(rankings_df=rankings_no_risk)
                    assert assistant.context_data.get("high_risk_count", 0) == 0
                except ImportError:
                    pytest.skip("OpenAI not installed")

    def test_set_context_with_risk_level(self, sample_rankings_df, sample_projects_df):
        """Test set_context with full rankings."""
        try:
            from unittest.mock import patch

            with patch("src.chat.assistant.OPENAI_AVAILABLE", True):
                with patch("src.chat.assistant.OpenAI"):
                    from src.chat.assistant import ChatAssistant

                    assistant = ChatAssistant(api_key="dummy-key")
                    assistant.set_context(
                        projects_df=sample_projects_df,
                        rankings_df=sample_rankings_df,
                    )
                    assert assistant.context_data["project_count"] == 3
                    assert assistant.context_data["high_risk_count"] == 1
                    assert len(assistant.context_data["rankings"]) == 3
        except ImportError:
            pytest.skip("OpenAI not installed")

    def test_get_suggested_questions(self):
        """Test suggested questions."""
        try:
            from unittest.mock import patch

            with patch("src.chat.assistant.OPENAI_AVAILABLE", True):
                with patch("src.chat.assistant.OpenAI"):
                    from src.chat.assistant import ChatAssistant

                    assistant = ChatAssistant(api_key="dummy-key")
                    questions = assistant.get_suggested_questions()
                    assert len(questions) > 0
                    assert "highest risk" in questions[0].lower() or "risk" in questions[0].lower()
        except ImportError:
            pytest.skip("OpenAI not installed")

    def test_format_top_projects_shows_riskiest(self):
        """Test _format_top_projects shows lowest MCDA (riskiest) first."""
        try:
            from unittest.mock import patch

            with patch("src.chat.assistant.OPENAI_AVAILABLE", True):
                with patch("src.chat.assistant.OpenAI"):
                    from src.chat.assistant import ChatAssistant

                    assistant = ChatAssistant(api_key="dummy-key")
                    assistant.context_data["rankings"] = [
                        {"project_name": "Safe", "mcda_score": 0.9, "risk_level": "Low"},
                        {"project_name": "Risky", "mcda_score": 0.2, "risk_level": "High"},
                    ]
                    text = assistant._format_top_projects()
                    assert "Risky" in text
                    assert "Safe" in text
                    assert "0.20" in text or "0.2" in text
        except ImportError:
            pytest.skip("OpenAI not installed")

    def test_format_project_info_handles_missing_score(self):
        """Test _format_project_info with missing mcda_score."""
        try:
            from unittest.mock import patch

            with patch("src.chat.assistant.OPENAI_AVAILABLE", True):
                with patch("src.chat.assistant.OpenAI"):
                    from src.chat.assistant import ChatAssistant

                    assistant = ChatAssistant(api_key="dummy-key")
                    result = assistant._format_project_info("Test", {"project_name": "Test"})
                    assert "N/A" in result or "Test" in result
        except ImportError:
            pytest.skip("OpenAI not installed")
