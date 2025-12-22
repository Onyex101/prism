"""
Tests for the MCDA module.
"""

import pytest
import numpy as np
import pandas as pd


class TestTOPSIS:
    """Test suite for TOPSIS algorithm."""

    def test_topsis_basic(self, sample_decision_matrix):
        """Test basic TOPSIS calculation."""
        from src.mcda.topsis import TOPSIS

        weights = [0.25, 0.25, 0.25, 0.25]
        types = ["cost", "benefit", "cost", "benefit"]

        topsis = TOPSIS(weights=weights, criteria_types=types)
        topsis.fit(sample_decision_matrix)

        scores = topsis.get_scores()

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_topsis_ranking(self, sample_decision_matrix):
        """Test TOPSIS ranking order."""
        from src.mcda.topsis import TOPSIS

        topsis = TOPSIS()
        topsis.fit(sample_decision_matrix)

        ranks = topsis.get_ranking()

        assert len(ranks) == 3
        assert set(ranks) == {1, 2, 3}

    def test_topsis_equal_weights(self):
        """Test TOPSIS with equal weights."""
        from src.mcda.topsis import TOPSIS

        matrix = np.array([[1, 2], [2, 1], [1.5, 1.5]])

        topsis = TOPSIS()
        topsis.fit(matrix)

        scores = topsis.get_scores()

        assert len(scores) == 3

    def test_topsis_not_fitted_error(self):
        """Test error when accessing scores before fitting."""
        from src.mcda.topsis import TOPSIS

        topsis = TOPSIS()

        with pytest.raises(ValueError):
            topsis.get_scores()


class TestProjectRanker:
    """Test suite for ProjectRanker class."""

    def test_ranker_basic(self, sample_projects_df):
        """Test basic project ranking."""
        from src.mcda.ranker import ProjectRanker

        ranker = ProjectRanker()
        rankings = ranker.rank(sample_projects_df)

        assert "project_id" in rankings.columns
        assert "mcda_score" in rankings.columns
        assert "rank" in rankings.columns
        assert len(rankings) == len(sample_projects_df)

    def test_ranker_risk_classification(self, sample_projects_df):
        """Test risk level classification."""
        from src.mcda.ranker import ProjectRanker

        ranker = ProjectRanker()
        rankings = ranker.rank(sample_projects_df)

        assert "risk_level" in rankings.columns
        assert all(rankings["risk_level"].isin(["High", "Medium", "Low"]))
