"""
Tests for the FeatureEngineer module.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    def test_create_features_basic(self, sample_projects_df):
        """Test basic feature creation."""
        from src.data.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        result = engineer.create_features(sample_projects_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_projects_df)
        assert len(engineer.get_feature_names()) > 0

    def test_schedule_performance_index(self):
        """Test SPI calculation."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "planned_hours": [100, 100, 100],
                "actual_hours": [100, 50, 200],
            }
        )

        engineer = FeatureEngineer()
        result = engineer.create_features(df)

        assert "schedule_performance_index" in result.columns
        # SPI = planned/actual
        assert result["schedule_performance_index"].iloc[0] == 1.0
        assert result["schedule_performance_index"].iloc[1] == 2.0
        assert result["schedule_performance_index"].iloc[2] == 0.5

    def test_cost_performance_index(self):
        """Test CPI calculation."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "budget": [100000, 100000, 100000],
                "spent": [100000, 50000, 200000],
            }
        )

        engineer = FeatureEngineer()
        result = engineer.create_features(df)

        assert "cost_performance_index" in result.columns
        # CPI = budget/spent
        assert result["cost_performance_index"].iloc[0] == 1.0
        assert result["cost_performance_index"].iloc[1] == 2.0
        assert result["cost_performance_index"].iloc[2] == 0.5

    def test_budget_variance_pct(self):
        """Test budget variance calculation."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "budget": [100000, 100000, 100000],
                "spent": [100000, 110000, 80000],
            }
        )

        engineer = FeatureEngineer()
        result = engineer.create_features(df)

        assert "budget_variance_pct" in result.columns
        assert result["budget_variance_pct"].iloc[0] == 0
        assert result["budget_variance_pct"].iloc[1] == 10
        assert result["budget_variance_pct"].iloc[2] == -20

    def test_team_stability(self):
        """Test team stability calculation."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "team_turnover": [0.0, 0.3, 1.0],
            }
        )

        engineer = FeatureEngineer()
        result = engineer.create_features(df)

        assert "team_stability" in result.columns
        assert result["team_stability"].iloc[0] == 1.0
        assert result["team_stability"].iloc[1] == 0.7
        assert result["team_stability"].iloc[2] == 0.0

    def test_binary_flags(self):
        """Test binary flag creation."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "budget": [100000, 100000],
                "spent": [90000, 120000],
                "planned_hours": [100, 100],
                "actual_hours": [100, 100],
                "completion_rate": [50, 50],
                "start_date": ["2024-01-01", "2024-01-01"],
                "planned_end_date": ["2024-06-01", "2024-06-01"],
            }
        )

        engineer = FeatureEngineer()
        result = engineer.create_features(df)

        if "is_over_budget" in result.columns:
            assert result["is_over_budget"].iloc[0] == 0
            assert result["is_over_budget"].iloc[1] == 1

    def test_get_feature_names(self):
        """Test feature names retrieval."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "budget": [100000],
                "spent": [90000],
                "planned_hours": [100],
                "actual_hours": [100],
            }
        )

        engineer = FeatureEngineer()
        result = engineer.create_features(df)

        feature_names = engineer.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "cost_performance_index" in feature_names

    def test_temporal_features_with_reference_date(self):
        """Test temporal features with custom reference date."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "start_date": ["2024-01-01"],
                "planned_end_date": ["2024-06-01"],
            }
        )

        reference_date = datetime(2024, 3, 15)
        engineer = FeatureEngineer()
        result = engineer.create_features(df, reference_date=reference_date)

        assert "days_since_start" in result.columns
        assert "days_remaining" in result.columns
        assert "planned_duration_days" in result.columns

    def test_handles_missing_columns(self):
        """Test graceful handling of missing columns."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "project_id": ["P1", "P2"],
                "project_name": ["A", "B"],
            }
        )

        engineer = FeatureEngineer()
        result = engineer.create_features(df)

        # Should not raise error with missing columns
        assert isinstance(result, pd.DataFrame)

    def test_clipping_values(self):
        """Test that extreme values are clipped."""
        from src.data.feature_engineer import FeatureEngineer

        df = pd.DataFrame(
            {
                "planned_hours": [100],
                "actual_hours": [10],  # Very efficient, would give SPI of 10
            }
        )

        engineer = FeatureEngineer()
        result = engineer.create_features(df)

        # SPI should be clipped to max 3
        assert result["schedule_performance_index"].iloc[0] <= 3.0
