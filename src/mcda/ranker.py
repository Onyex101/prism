"""
Project Ranker Module
=====================

This module ranks projects using MCDA combining ML, LLM, and metric scores.

It integrates risk scores from ML models, sentiment scores from LLM analysis,
and performance metrics to produce a unified project risk ranking.

Example:
    >>> from src.mcda.ranker import ProjectRanker
    >>> ranker = ProjectRanker()
    >>> rankings = ranker.rank(projects_df, ml_scores, llm_scores)
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.mcda.topsis import TOPSIS


class ProjectRanker:
    """
    Rank projects using multi-criteria decision analysis.

    This class combines ML risk predictions, LLM sentiment analysis,
    and performance metrics using TOPSIS to produce a unified ranking.

    :cvar DEFAULT_CRITERIA: Default criteria configuration with weights and types.
    :vartype DEFAULT_CRITERIA: dict

    :ivar criteria: Criteria configuration.
    :vartype criteria: dict
    :ivar topsis: TOPSIS algorithm instance.
    :vartype topsis: TOPSIS
    :ivar rankings: Current rankings DataFrame.
    :vartype rankings: Optional[pd.DataFrame]

    Example:
        >>> ranker = ProjectRanker()
        >>> rankings = ranker.rank(projects_df)
        >>> top_risk = ranker.get_top_risk_projects(n=5)
    """

    DEFAULT_CRITERIA: dict = {
        "ml_risk_score": {"weight": 0.40, "type": "cost"},
        "llm_sentiment_score": {"weight": 0.25, "type": "benefit"},
        "schedule_performance_index": {"weight": 0.15, "type": "benefit"},
        "cost_performance_index": {"weight": 0.10, "type": "benefit"},
        "team_stability": {"weight": 0.10, "type": "benefit"},
    }

    def __init__(self, criteria: Optional[dict] = None) -> None:
        """
        Initialize the ranker.

        :param criteria: Dict defining criteria with weights and types.
            Each criterion should have "weight" and "type" keys.
            Type can be "benefit" (higher is better) or "cost" (lower is better).
        :type criteria: Optional[dict]

        Example:
            >>> custom_criteria = {
            ...     "ml_risk_score": {"weight": 0.50, "type": "cost"},
            ...     "llm_sentiment_score": {"weight": 0.30, "type": "benefit"},
            ...     "team_stability": {"weight": 0.20, "type": "benefit"},
            ... }
            >>> ranker = ProjectRanker(criteria=custom_criteria)
        """
        self.criteria = criteria or self.DEFAULT_CRITERIA.copy()
        self.topsis = TOPSIS()
        self.rankings: Optional[pd.DataFrame] = None

    def rank(
        self,
        projects_df: pd.DataFrame,
        ml_scores: Optional[pd.Series] = None,
        llm_scores: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Rank projects using MCDA.

        :param projects_df: DataFrame with project data.
        :type projects_df: pd.DataFrame
        :param ml_scores: ML risk scores (0-1, higher = riskier).
        :type ml_scores: Optional[pd.Series]
        :param llm_scores: LLM sentiment scores (-1 to 1).
        :type llm_scores: Optional[pd.Series]
        :return: DataFrame with rankings.
        :rtype: pd.DataFrame

        Example:
            >>> rankings = ranker.rank(projects_df, ml_scores=risk_predictions)
            >>> print(rankings[["project_name", "rank", "risk_level"]])
        """
        decision_matrix, project_ids = self._prepare_matrix(projects_df, ml_scores, llm_scores)

        weights = [self.criteria[c]["weight"] for c in self.criteria]
        types = [self.criteria[c]["type"] for c in self.criteria]

        self.topsis = TOPSIS(weights=weights, criteria_types=types)
        self.topsis.fit(decision_matrix)

        self.rankings = self._build_rankings_df(project_ids, projects_df)

        logger.info(f"Ranked {len(self.rankings)} projects")

        return self.rankings

    def _prepare_matrix(
        self,
        df: pd.DataFrame,
        ml_scores: Optional[pd.Series],
        llm_scores: Optional[pd.Series],
    ) -> tuple[np.ndarray, list]:
        """
        Prepare the decision matrix for TOPSIS.

        :param df: Project data.
        :type df: pd.DataFrame
        :param ml_scores: ML risk scores.
        :type ml_scores: Optional[pd.Series]
        :param llm_scores: LLM sentiment scores.
        :type llm_scores: Optional[pd.Series]
        :return: Tuple of (decision_matrix, project_ids).
        :rtype: tuple[np.ndarray, list]
        """
        project_ids = df["project_id"].tolist()
        n_projects = len(project_ids)
        n_criteria = len(self.criteria)

        matrix = np.zeros((n_projects, n_criteria))

        for i, criterion in enumerate(self.criteria.keys()):
            matrix[:, i] = self._get_criterion_values(
                criterion, df, ml_scores, llm_scores, n_projects
            )

        return matrix, project_ids

    def _get_criterion_values(
        self,
        criterion: str,
        df: pd.DataFrame,
        ml_scores: Optional[pd.Series],
        llm_scores: Optional[pd.Series],
        n_projects: int,
    ) -> np.ndarray:
        """
        Get values for a single criterion.

        :param criterion: Criterion name.
        :type criterion: str
        :param df: Project data.
        :type df: pd.DataFrame
        :param ml_scores: ML risk scores.
        :type ml_scores: Optional[pd.Series]
        :param llm_scores: LLM sentiment scores.
        :type llm_scores: Optional[pd.Series]
        :param n_projects: Number of projects.
        :type n_projects: int
        :return: Array of criterion values.
        :rtype: np.ndarray
        """
        if criterion == "ml_risk_score":
            return self._get_ml_scores(df, ml_scores, n_projects)

        if criterion == "llm_sentiment_score":
            return self._get_llm_scores(df, llm_scores, n_projects)

        if criterion == "schedule_performance_index":
            return self._get_spi(df, n_projects)

        if criterion == "cost_performance_index":
            return self._get_cpi(df, n_projects)

        if criterion == "team_stability":
            return self._get_team_stability(df, n_projects)

        return np.full(n_projects, 0.5)

    def _get_ml_scores(
        self,
        df: pd.DataFrame,
        ml_scores: Optional[pd.Series],
        n_projects: int,
    ) -> np.ndarray:
        """Get ML risk scores."""
        if ml_scores is not None:
            return ml_scores.values
        if "risk_score" in df.columns:
            return df["risk_score"].values
        return np.full(n_projects, 0.5)

    def _get_llm_scores(
        self,
        df: pd.DataFrame,
        llm_scores: Optional[pd.Series],
        n_projects: int,
    ) -> np.ndarray:
        """Get LLM sentiment scores (normalized to 0-1)."""
        if llm_scores is not None:
            return (llm_scores.values + 1) / 2
        if "sentiment_score" in df.columns:
            return (df["sentiment_score"].values + 1) / 2
        return np.full(n_projects, 0.5)

    def _get_spi(self, df: pd.DataFrame, n_projects: int) -> np.ndarray:
        """Get Schedule Performance Index (normalized)."""
        if "schedule_performance_index" in df.columns:
            return df["schedule_performance_index"].clip(0, 2).values / 2
        return np.full(n_projects, 0.5)

    def _get_cpi(self, df: pd.DataFrame, n_projects: int) -> np.ndarray:
        """Get Cost Performance Index (normalized)."""
        if "cost_performance_index" in df.columns:
            return df["cost_performance_index"].clip(0, 2).values / 2
        return np.full(n_projects, 0.5)

    def _get_team_stability(self, df: pd.DataFrame, n_projects: int) -> np.ndarray:
        """Get team stability score."""
        if "team_stability" in df.columns:
            return df["team_stability"].values
        if "team_turnover" in df.columns:
            return 1 - df["team_turnover"].clip(0, 1).values
        return np.full(n_projects, 0.9)

    def _build_rankings_df(self, project_ids: list, projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the rankings DataFrame.

        :param project_ids: List of project IDs.
        :type project_ids: list
        :param projects_df: Original projects DataFrame.
        :type projects_df: pd.DataFrame
        :return: Rankings DataFrame.
        :rtype: pd.DataFrame
        """
        rankings = pd.DataFrame(
            {
                "project_id": project_ids,
                "mcda_score": self.topsis.get_scores(),
                "rank": self.topsis.get_ranking(),
            }
        )

        rankings["risk_level"] = rankings["mcda_score"].apply(self._classify_risk)

        if "project_name" in projects_df.columns:
            name_map = projects_df.set_index("project_id")["project_name"].to_dict()
            rankings["project_name"] = rankings["project_id"].map(name_map)

        return rankings.sort_values("rank")

    def _classify_risk(self, score: float) -> str:
        """
        Classify risk level based on MCDA score.

        :param score: MCDA score (0-1, higher is better).
        :type score: float
        :return: Risk level string.
        :rtype: str
        """
        if score >= 0.70:
            return "Low"
        if score >= 0.40:
            return "Medium"
        return "High"

    def get_rankings(self) -> pd.DataFrame:
        """
        Get the current rankings.

        :return: Rankings DataFrame.
        :rtype: pd.DataFrame
        :raises ValueError: If no rankings available.

        Example:
            >>> rankings = ranker.get_rankings()
        """
        if self.rankings is None:
            raise ValueError("No rankings available. Call rank() first.")
        return self.rankings

    def get_top_risk_projects(self, n: int = 5) -> pd.DataFrame:
        """
        Get top N high-risk projects.

        :param n: Number of projects to return.
        :type n: int
        :return: DataFrame with top risk projects.
        :rtype: pd.DataFrame
        :raises ValueError: If no rankings available.

        Example:
            >>> top_5 = ranker.get_top_risk_projects(n=5)
            >>> for _, row in top_5.iterrows():
            ...     print(f"{row['project_name']}: {row['risk_level']}")
        """
        if self.rankings is None:
            raise ValueError("No rankings available. Call rank() first.")
        return self.rankings.nsmallest(n, "mcda_score")

    def get_score_breakdown(self, project_id: str) -> dict:
        """
        Get detailed score breakdown for a project.

        :param project_id: Project identifier.
        :type project_id: str
        :return: Dict with score breakdown.
        :rtype: dict
        :raises ValueError: If no rankings available.

        Example:
            >>> breakdown = ranker.get_score_breakdown("PROJ-001")
            >>> print(f"MCDA Score: {breakdown['mcda_score']:.3f}")
        """
        if self.rankings is None:
            raise ValueError("No rankings available. Call rank() first.")

        row = self.rankings[self.rankings["project_id"] == project_id]
        if row.empty:
            return {}

        return {
            "project_id": project_id,
            "mcda_score": float(row["mcda_score"].values[0]),
            "rank": int(row["rank"].values[0]),
            "risk_level": row["risk_level"].values[0],
            "criteria_weights": {k: v["weight"] for k, v in self.criteria.items()},
        }

    def sensitivity_analysis(
        self,
        projects_df: pd.DataFrame,
        weight_variation: float = 0.10,
    ) -> dict:
        """
        Perform sensitivity analysis on weights.

        Tests how rankings change when each criterion's weight is varied
        by the specified amount.

        :param projects_df: Project data.
        :type projects_df: pd.DataFrame
        :param weight_variation: Amount to vary weights (±).
        :type weight_variation: float
        :return: Dict with sensitivity results.
        :rtype: dict

        Example:
            >>> sensitivity = ranker.sensitivity_analysis(projects_df, 0.15)
            >>> for criterion, data in sensitivity["criteria_sensitivity"].items():
            ...     print(f"{criterion}: avg rank change = {data['avg_rank_change']:.2f}")
        """
        original_rankings = self.rankings.copy() if self.rankings is not None else None

        results = {
            "weight_variation": weight_variation,
            "criteria_sensitivity": {},
        }

        for criterion in self.criteria:
            avg_change = self._analyze_criterion_sensitivity(
                criterion, projects_df, weight_variation, original_rankings
            )
            results["criteria_sensitivity"][criterion] = {
                "original_weight": self.criteria[criterion]["weight"],
                "avg_rank_change": float(avg_change),
            }

        # Restore original rankings
        if original_rankings is not None:
            self.rankings = original_rankings

        return results

    def _analyze_criterion_sensitivity(
        self,
        criterion: str,
        projects_df: pd.DataFrame,
        weight_variation: float,
        original_rankings: Optional[pd.DataFrame],
    ) -> float:
        """
        Analyze sensitivity for a single criterion.

        :param criterion: Criterion name.
        :type criterion: str
        :param projects_df: Project data.
        :type projects_df: pd.DataFrame
        :param weight_variation: Weight variation amount.
        :type weight_variation: float
        :param original_rankings: Original rankings for comparison.
        :type original_rankings: Optional[pd.DataFrame]
        :return: Average rank change.
        :rtype: float
        """
        original_weight = self.criteria[criterion]["weight"]

        # Test with increased weight
        self.criteria[criterion]["weight"] = min(1.0, original_weight + weight_variation)
        self.rank(projects_df)
        increased_rankings = self.rankings["rank"].values

        # Test with decreased weight
        self.criteria[criterion]["weight"] = max(0.0, original_weight - weight_variation)
        self.rank(projects_df)
        decreased_rankings = self.rankings["rank"].values

        # Calculate average rank change
        avg_change = 0.0
        if original_rankings is not None:
            original_ranks = original_rankings["rank"].values
            avg_change = (
                np.mean(
                    np.abs(increased_rankings - original_ranks)
                    + np.abs(decreased_rankings - original_ranks)
                )
                / 2
            )

        # Restore original weight
        self.criteria[criterion]["weight"] = original_weight

        return avg_change
