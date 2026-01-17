"""
Tests for the MCDA module.
"""

import numpy as np
import pandas as pd
import pytest


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

    def test_topsis_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        from src.mcda.topsis import TOPSIS

        matrix = np.array([[1, 2], [2, 1]])
        weights = [2, 2]  # Sum to 4, should be normalized

        topsis = TOPSIS(weights=weights)
        topsis.fit(matrix)

        # Should not raise error
        scores = topsis.get_scores()
        assert len(scores) == 2

    def test_topsis_invalid_weights_length(self):
        """Test error with mismatched weights length."""
        from src.mcda.topsis import TOPSIS

        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        weights = [0.5, 0.5]  # 2 weights for 3 criteria

        topsis = TOPSIS(weights=weights)

        with pytest.raises(ValueError, match="Weights length"):
            topsis.fit(matrix)

    def test_topsis_cost_vs_benefit(self):
        """Test that cost and benefit criteria are handled differently."""
        from src.mcda.topsis import TOPSIS

        # Alternative 1: Low cost (good), High benefit (good)
        # Alternative 2: High cost (bad), Low benefit (bad)
        matrix = np.array(
            [
                [10, 100],  # Low cost, high benefit (should rank best)
                [100, 10],  # High cost, low benefit (should rank worst)
            ]
        )

        # First column is cost (lower is better)
        # Second column is benefit (higher is better)
        topsis = TOPSIS(weights=[0.5, 0.5], criteria_types=["cost", "benefit"])
        topsis.fit(matrix)

        ranks = topsis.get_ranking()

        # First alternative should rank better (lower rank number)
        assert ranks[0] < ranks[1]

    def test_topsis_rank_alternatives(self):
        """Test rank_alternatives method."""
        from src.mcda.topsis import TOPSIS

        matrix = np.array([[1, 2], [2, 1], [1.5, 1.5]])
        names = ["Option A", "Option B", "Option C"]

        topsis = TOPSIS()
        result = topsis.rank_alternatives(matrix, alternative_names=names)

        assert isinstance(result, pd.DataFrame)
        assert "alternative" in result.columns
        assert "score" in result.columns
        assert "rank" in result.columns
        assert len(result) == 3


class TestProjectRanker:
    """Test suite for ProjectRanker class."""

    def test_ranker_basic(self, sample_projects_df):
        """Test basic project ranking."""
        from src.mcda.ranker import ProjectRanker
        from src.data.feature_engineer import FeatureEngineer

        # Need features for ranking
        engineer = FeatureEngineer()
        df = engineer.create_features(sample_projects_df)

        ranker = ProjectRanker()
        rankings = ranker.rank(df)

        assert "project_id" in rankings.columns
        assert "mcda_score" in rankings.columns
        assert "rank" in rankings.columns
        assert len(rankings) == len(sample_projects_df)

    def test_ranker_risk_classification(self, sample_projects_df):
        """Test risk level classification."""
        from src.mcda.ranker import ProjectRanker
        from src.data.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        df = engineer.create_features(sample_projects_df)

        ranker = ProjectRanker()
        rankings = ranker.rank(df)

        assert "risk_level" in rankings.columns
        assert all(rankings["risk_level"].isin(["High", "Medium", "Low"]))

    def test_ranker_with_ml_scores(self, sample_projects_df):
        """Test ranking with ML scores."""
        from src.mcda.ranker import ProjectRanker
        from src.data.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        df = engineer.create_features(sample_projects_df)

        ml_scores = pd.Series([0.8, 0.5, 0.2])

        ranker = ProjectRanker()
        rankings = ranker.rank(df, ml_scores=ml_scores)

        assert len(rankings) == 3

    def test_ranker_with_llm_scores(self, sample_projects_df):
        """Test ranking with LLM sentiment scores."""
        from src.mcda.ranker import ProjectRanker
        from src.data.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        df = engineer.create_features(sample_projects_df)

        llm_scores = pd.Series([-0.5, 0.0, 0.5])  # -1 to 1 range

        ranker = ProjectRanker()
        rankings = ranker.rank(df, llm_scores=llm_scores)

        assert len(rankings) == 3

    def test_ranker_get_top_risk_projects(self, sample_projects_df):
        """Test getting top risk projects."""
        from src.mcda.ranker import ProjectRanker
        from src.data.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        df = engineer.create_features(sample_projects_df)

        ranker = ProjectRanker()
        ranker.rank(df)

        top_risk = ranker.get_top_risk_projects(n=2)

        assert len(top_risk) == 2

    def test_ranker_get_score_breakdown(self, sample_projects_df):
        """Test score breakdown for a project."""
        from src.mcda.ranker import ProjectRanker
        from src.data.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        df = engineer.create_features(sample_projects_df)

        ranker = ProjectRanker()
        ranker.rank(df)

        breakdown = ranker.get_score_breakdown("PROJ-001")

        assert "project_id" in breakdown
        assert "mcda_score" in breakdown
        assert "rank" in breakdown
        assert "criteria_weights" in breakdown

    def test_ranker_custom_criteria(self, sample_projects_df):
        """Test ranking with custom criteria."""
        from src.mcda.ranker import ProjectRanker
        from src.data.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        df = engineer.create_features(sample_projects_df)

        custom_criteria = {
            "ml_risk_score": {"weight": 0.60, "type": "cost"},
            "llm_sentiment_score": {"weight": 0.40, "type": "benefit"},
        }

        ranker = ProjectRanker(criteria=custom_criteria)
        rankings = ranker.rank(df)

        assert len(rankings) == 3

    def test_ranker_not_ranked_error(self):
        """Test error when accessing rankings before ranking."""
        from src.mcda.ranker import ProjectRanker

        ranker = ProjectRanker()

        with pytest.raises(ValueError, match="No rankings available"):
            ranker.get_rankings()

    def test_ranker_sensitivity_analysis(self, sample_projects_df):
        """Test sensitivity analysis."""
        from src.mcda.ranker import ProjectRanker
        from src.data.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer()
        df = engineer.create_features(sample_projects_df)

        ranker = ProjectRanker()
        ranker.rank(df)

        sensitivity = ranker.sensitivity_analysis(df, weight_variation=0.10)

        assert "weight_variation" in sensitivity
        assert "criteria_sensitivity" in sensitivity
