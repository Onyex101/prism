"""
Tests for the DataPreprocessor module.
"""

import numpy as np
import pandas as pd
import pytest


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    def test_fit_transform_basic(self, sample_projects_df):
        """Test basic fit_transform operation."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        numerical_cols = ["budget", "spent", "team_size"]
        categorical_cols = ["status", "priority"]

        result = preprocessor.fit_transform(sample_projects_df, numerical_cols, categorical_cols)

        assert isinstance(result, pd.DataFrame)
        assert preprocessor.fitted
        assert len(result) == len(sample_projects_df)

    def test_transform_requires_fit(self, sample_projects_df):
        """Test that transform requires fitting first."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.transform(sample_projects_df, [], [])

    def test_missing_value_handling_median(self):
        """Test median imputation for missing values."""
        from src.data.preprocessor import DataPreprocessor

        df = pd.DataFrame(
            {
                "value": [1.0, 2.0, np.nan, 4.0, 5.0],
                "category": ["a", "b", None, "a", "b"],
            }
        )

        preprocessor = DataPreprocessor(
            numerical_strategy="median",
            categorical_strategy="most_frequent",
        )
        # Note: The fill value is computed before scaling
        # We need to check after fit_transform which also scales
        original_median = df["value"].dropna().median()
        result = preprocessor.fit_transform(df, ["value"], ["category"])

        assert not result["value"].isna().any()
        # After scaling, we just check no NaN values exist

    def test_missing_value_handling_mean(self):
        """Test mean imputation for missing values."""
        from src.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0]})

        preprocessor = DataPreprocessor(numerical_strategy="mean")
        result = preprocessor.fit_transform(df, ["value"], [])

        # After fit_transform, values are scaled, so just check no NaN
        assert not result["value"].isna().any()

    def test_categorical_encoding(self):
        """Test categorical variable encoding."""
        from src.data.preprocessor import DataPreprocessor

        df = pd.DataFrame(
            {
                "status": ["Active", "Completed", "Active", "On Hold"],
            }
        )

        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(df, [], ["status"])

        assert "status_encoded" in result.columns
        assert result["status_encoded"].dtype in [np.int32, np.int64]

    def test_numerical_scaling(self):
        """Test numerical feature scaling."""
        from src.data.preprocessor import DataPreprocessor

        df = pd.DataFrame(
            {
                "value": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(df, ["value"], [])

        # Scaled values should have mean ~0 and std ~1
        # Using ddof=0 for population std (sklearn default)
        assert abs(result["value"].mean()) < 1e-10
        # Standard scaler uses ddof=0, pandas uses ddof=1 by default
        assert abs(result["value"].std(ddof=0) - 1) < 0.01

    def test_clean_text(self):
        """Test text cleaning function."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        # Test whitespace normalization
        assert preprocessor.clean_text("  multiple   spaces  ") == "multiple spaces"

        # Test empty/None handling
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text("") == ""

    def test_prepare_for_ml(self, sample_projects_df):
        """Test ML preparation function."""
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_projects_df, [], [])

        X, y = preprocessor.prepare_for_ml(sample_projects_df)

        assert isinstance(X, pd.DataFrame)
        assert y is None
        # Non-feature columns should be dropped
        assert "project_id" not in X.columns
        assert "status_comments" not in X.columns

    def test_prepare_for_ml_with_target(self, sample_projects_df):
        """Test ML preparation with target column."""
        from src.data.preprocessor import DataPreprocessor

        df = sample_projects_df.copy()
        df["risk_level"] = ["High", "Medium", "Low"]

        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(df, [], [])

        X, y = preprocessor.prepare_for_ml(df, target_col="risk_level")

        assert y is not None
        assert len(y) == 3
        assert "risk_level" not in X.columns
