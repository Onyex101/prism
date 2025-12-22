"""
Tests for the DataValidator module.
"""

import pytest
import pandas as pd


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_validate_valid_data(self, sample_projects_df):
        """Test validation of valid project data."""
        from src.data.validator import DataValidator

        validator = DataValidator()
        result = validator.validate(sample_projects_df)

        assert result.is_valid
        assert result.error_count == 0

    def test_validate_missing_columns(self):
        """Test validation catches missing required columns."""
        from src.data.validator import DataValidator

        df = pd.DataFrame({"project_id": ["P1"], "project_name": ["Test"]})

        validator = DataValidator()
        result = validator.validate(df)

        assert not result.is_valid
        assert result.error_count > 0
        assert any("missing" in str(e).lower() for e in result.errors)

    def test_validate_duplicate_ids(self, sample_projects_df):
        """Test validation catches duplicate project IDs."""
        from src.data.validator import DataValidator

        df = sample_projects_df.copy()
        df = pd.concat([df, df.iloc[[0]]])  # Duplicate first row

        validator = DataValidator()
        result = validator.validate(df)

        assert any("duplicate" in str(e).lower() for e in result.errors)

    def test_calculate_stats(self, sample_projects_df):
        """Test statistics calculation."""
        from src.data.validator import DataValidator

        validator = DataValidator()
        result = validator.validate(sample_projects_df)

        assert "total_projects" in result.stats
        assert result.stats["total_projects"] == 3
        assert "overall_completeness_pct" in result.stats
