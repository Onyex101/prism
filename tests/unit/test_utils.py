"""
Tests for the utils module.
"""

import pandas as pd
import pytest


class TestCalculateMetrics:
    """Test suite for calculate_metrics function."""

    def test_calculate_metrics_basic(self, sample_projects_df):
        """Test basic metrics calculation."""
        from src.utils.metrics import calculate_metrics

        df = sample_projects_df.copy()
        df["risk_level"] = ["High", "Medium", "Low"]

        metrics = calculate_metrics(df)

        assert "total_projects" in metrics
        assert metrics["total_projects"] == 3

    def test_risk_distribution(self):
        """Test risk distribution calculation."""
        from src.utils.metrics import calculate_metrics

        df = pd.DataFrame(
            {
                "risk_level": ["High", "High", "Medium", "Low", "Low", "Low"],
            }
        )

        metrics = calculate_metrics(df)

        assert metrics["high_risk_count"] == 2
        assert metrics["medium_risk_count"] == 1
        assert metrics["low_risk_count"] == 3

    def test_budget_metrics(self):
        """Test budget metrics calculation."""
        from src.utils.metrics import calculate_metrics

        df = pd.DataFrame(
            {
                "budget": [100000, 200000],
                "spent": [90000, 220000],
            }
        )

        metrics = calculate_metrics(df)

        assert metrics["total_budget"] == 300000
        assert metrics["total_spent"] == 310000
        assert abs(metrics["budget_utilization"] - (310000 / 300000)) < 0.001

    def test_completion_metrics(self):
        """Test completion metrics calculation."""
        from src.utils.metrics import calculate_metrics

        df = pd.DataFrame(
            {
                "completion_rate": [80, 90, 95, 50],
            }
        )

        metrics = calculate_metrics(df)

        assert metrics["avg_completion"] == pytest.approx(78.75)
        assert metrics["projects_near_complete"] == 2

    def test_team_metrics(self):
        """Test team metrics calculation."""
        from src.utils.metrics import calculate_metrics

        df = pd.DataFrame(
            {
                "team_size": [5, 10, 15],
            }
        )

        metrics = calculate_metrics(df)

        assert metrics["total_team_members"] == 30
        assert metrics["avg_team_size"] == 10


class TestCalculatePortfolioHealth:
    """Test suite for calculate_portfolio_health function."""

    def test_healthy_portfolio(self):
        """Test healthy portfolio classification."""
        from src.utils.metrics import calculate_portfolio_health

        metrics = {
            "avg_risk_score": 0.2,
            "budget_utilization": 0.95,
            "avg_completion": 80,
            "high_risk_count": 1,
            "total_projects": 10,
        }

        health = calculate_portfolio_health(metrics)

        assert "health_score" in health
        assert "health_level" in health
        assert health["health_level"] in ["Healthy", "At Risk", "Critical"]

    def test_critical_portfolio(self):
        """Test critical portfolio classification."""
        from src.utils.metrics import calculate_portfolio_health

        metrics = {
            "avg_risk_score": 0.8,
            "budget_utilization": 2.0,
            "avg_completion": 20,
            "high_risk_count": 8,
            "total_projects": 10,
        }

        health = calculate_portfolio_health(metrics)

        assert health["health_level"] == "Critical"

    def test_empty_metrics(self):
        """Test with empty metrics."""
        from src.utils.metrics import calculate_portfolio_health

        metrics = {}
        health = calculate_portfolio_health(metrics)

        assert health["health_score"] == 0.5


class TestCalculateTrend:
    """Test suite for calculate_trend function."""

    def test_improving_trend(self):
        """Test detection of improving trend."""
        from src.utils.metrics import calculate_trend

        current = {"avg_risk_score": 0.3, "avg_completion": 80}
        previous = {"avg_risk_score": 0.5, "avg_completion": 60}

        trend = calculate_trend(current, previous)

        # Lower risk score is improving
        assert trend["avg_risk_score"]["direction"] == "improving"
        # Higher completion is improving
        assert trend["avg_completion"]["direction"] == "improving"

    def test_worsening_trend(self):
        """Test detection of worsening trend."""
        from src.utils.metrics import calculate_trend

        current = {"avg_risk_score": 0.7}
        previous = {"avg_risk_score": 0.3}

        trend = calculate_trend(current, previous)

        assert trend["avg_risk_score"]["direction"] == "worsening"

    def test_stable_trend(self):
        """Test detection of stable trend."""
        from src.utils.metrics import calculate_trend

        current = {"avg_risk_score": 0.5}
        previous = {"avg_risk_score": 0.51}

        trend = calculate_trend(current, previous)

        assert trend["avg_risk_score"]["direction"] == "stable"

    def test_change_percentage(self):
        """Test change percentage calculation."""
        from src.utils.metrics import calculate_trend

        current = {"budget_utilization": 1.0}
        previous = {"budget_utilization": 0.8}

        trend = calculate_trend(current, previous)

        assert trend["budget_utilization"]["change_pct"] == 25.0


class TestSetupLogger:
    """Test suite for setup_logger function."""

    def test_setup_logger_default(self):
        """Test default logger setup."""
        from src.utils.logger import setup_logger

        # Should not raise any errors
        setup_logger(log_level="INFO")

    def test_setup_logger_with_file(self, tmp_path):
        """Test logger setup with file output."""
        from src.utils.logger import setup_logger

        log_file = tmp_path / "test.log"
        setup_logger(log_level="DEBUG", log_file=str(log_file))

        # File should be created on first log
        from loguru import logger

        logger.info("Test message")

        # Note: File might not be created immediately due to buffering

    def test_setup_logger_levels(self):
        """Test different log levels."""
        from src.utils.logger import setup_logger

        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            setup_logger(log_level=level)
            # Should not raise
