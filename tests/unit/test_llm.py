"""
Tests for the LLM modules (analyzer, risk_extractor).
"""

import pytest


@pytest.fixture
def sample_llm_results():
    """Sample LLM analysis results from conftest."""
    return [
        {
            "project_id": "PROJ-001",
            "project_name": "Project Alpha",
            "sentiment_score": -0.3,
            "sentiment_label": "negative",
            "risk_level": "high",
            "risk_categories": ["technical", "schedule"],
            "risk_indicators": ["code quality issues", "deadline pressure"],
            "key_quotes": ["facing significant challenges"],
            "confidence": 0.8,
            "summary": "Project has technical and schedule risks.",
        },
        {
            "project_id": "PROJ-002",
            "project_name": "Project Beta",
            "sentiment_score": 0.5,
            "sentiment_label": "positive",
            "risk_level": "low",
            "risk_categories": [],
            "risk_indicators": [],
            "key_quotes": ["on track"],
            "confidence": 0.9,
            "summary": "Project is on track.",
        },
    ]


class TestRiskExtractor:
    """Test suite for RiskExtractor class."""

    def test_extract(self, sample_llm_results):
        """Test extracting risk analyses from LLM results."""
        from src.models.llm.risk_extractor import RiskExtractor

        extractor = RiskExtractor()
        analyses = extractor.extract(sample_llm_results)

        assert len(analyses) == 2
        assert analyses[0].project_name == "Project Alpha"
        assert analyses[0].risk_level == "high"
        assert analyses[0].sentiment_score == -0.3
        assert "technical" in analyses[0].risk_categories

    def test_to_dataframe(self, sample_llm_results):
        """Test converting analyses to DataFrame."""
        from src.models.llm.risk_extractor import RiskExtractor

        extractor = RiskExtractor()
        extractor.extract(sample_llm_results)
        df = extractor.to_dataframe()

        assert len(df) == 2
        assert "sentiment_score" in df.columns
        assert "risk_level" in df.columns
        assert df["risk_level"].iloc[0] == "high"

    def test_get_summary_stats(self, sample_llm_results):
        """Test summary statistics."""
        from src.models.llm.risk_extractor import RiskExtractor

        extractor = RiskExtractor()
        extractor.extract(sample_llm_results)
        stats = extractor.get_summary_stats()

        assert stats["total_projects"] == 2
        assert "risk_distribution" in stats
        assert "avg_sentiment" in stats

    def test_get_high_risk_projects(self, sample_llm_results):
        """Test getting high-risk projects."""
        from src.models.llm.risk_extractor import RiskExtractor

        extractor = RiskExtractor()
        extractor.extract(sample_llm_results)
        high_risk = extractor.get_high_risk_projects()

        assert len(high_risk) == 1
        assert high_risk[0].project_name == "Project Alpha"


class TestLLMAnalyzer:
    """Test suite for LLMAnalyzer with mocked OpenAI."""

    def test_empty_result_for_short_text(self):
        """Test that short text returns empty result."""
        try:
            from unittest.mock import MagicMock, patch

            with patch("src.models.llm.analyzer.OPENAI_AVAILABLE", True):
                with patch("src.models.llm.analyzer.OpenAI"):
                    from src.models.llm.analyzer import LLMAnalyzer

                    analyzer = LLMAnalyzer(api_key="test-key")
                    result = analyzer.analyze_project("Test", "Short")

                    assert result["sentiment_score"] == 0.0
                    assert "Insufficient" in result["summary"]
        except ImportError:
            pytest.skip("OpenAI not installed")

    def test_validate_client_raises_without_key(self):
        """Test that analyze_project raises without API key."""
        try:
            from unittest.mock import patch

            with patch("src.models.llm.analyzer.OPENAI_AVAILABLE", True):
                with patch("src.models.llm.analyzer.OpenAI"):
                    from src.models.llm.analyzer import LLMAnalyzer

                    analyzer = LLMAnalyzer(api_key=None)
                    with pytest.raises(ValueError, match="API key"):
                        analyzer.analyze_project("P", "Valid status comment here for analysis.")
        except ImportError:
            pytest.skip("OpenAI not installed")

    def test_analyze_batch(self):
        """Test batch analysis with mocked API."""
        try:
            from unittest.mock import MagicMock, patch

            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(
                        content='{"sentiment_score": 0.1, "sentiment_label": "neutral", "risk_level": "medium", '
                        '"risk_categories": [], "risk_indicators": [], "key_quotes": [], '
                        '"confidence": 0.7, "summary": "Test summary"}'
                    )
                )
            ]
            mock_response.usage.total_tokens = 100

            with patch("src.models.llm.analyzer.OPENAI_AVAILABLE", True):
                with patch("src.models.llm.analyzer.OpenAI") as MockOpenAI:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_response
                    MockOpenAI.return_value = mock_client

                    from src.models.llm.analyzer import LLMAnalyzer

                    analyzer = LLMAnalyzer(api_key="test-key")
                    projects = [
                        {"project_name": "P1", "status_comments": "Good progress on the project."},
                        {"project_name": "P2", "status_comments": "Some delays observed."},
                    ]
                    progress_calls: list[tuple[int, int, str]] = []

                    def _cb(done: int, total: int, name: str, result: dict) -> None:
                        progress_calls.append((done, total, name))

                    results = analyzer.analyze_batch(projects, on_progress=_cb)

                    assert len(results) == 2
                    assert progress_calls == [(1, 2, "P1"), (2, 2, "P2")]
                    assert results[0]["project_name"] == "P1"
                    assert "sentiment_score" in results[0]
        except ImportError:
            pytest.skip("OpenAI not installed")
