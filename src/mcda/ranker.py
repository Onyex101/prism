"""
Project Ranker Module

Ranks projects using MCDA combining ML, LLM, and metric scores.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .topsis import TOPSIS


class ProjectRanker:
    """Rank projects using multi-criteria decision analysis."""

    DEFAULT_CRITERIA = {
        "ml_risk_score": {"weight": 0.40, "type": "cost"},
        "llm_sentiment_score": {"weight": 0.25, "type": "benefit"},
        "schedule_performance_index": {"weight": 0.15, "type": "benefit"},
        "cost_performance_index": {"weight": 0.10, "type": "benefit"},
        "team_stability": {"weight": 0.10, "type": "benefit"},
    }

    def __init__(
        self,
        criteria: Optional[dict] = None,
    ):
        """
        Initialize the ranker.

        Args:
            criteria: Dict defining criteria with weights and types
        """
        self.criteria = criteria or self.DEFAULT_CRITERIA
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

        Args:
            projects_df: DataFrame with project data
            ml_scores: ML risk scores (0-1)
            llm_scores: LLM sentiment scores (-1 to 1)

        Returns:
            DataFrame with rankings
        """
        # Prepare decision matrix
        decision_matrix, project_ids = self._prepare_matrix(projects_df, ml_scores, llm_scores)

        # Get weights and types from criteria config
        weights = [self.criteria[c]["weight"] for c in self.criteria]
        types = [self.criteria[c]["type"] for c in self.criteria]

        # Run TOPSIS
        self.topsis = TOPSIS(weights=weights, criteria_types=types)
        self.topsis.fit(decision_matrix)

        # Build results DataFrame
        self.rankings = pd.DataFrame(
            {
                "project_id": project_ids,
                "mcda_score": self.topsis.get_scores(),
                "rank": self.topsis.get_ranking(),
            }
        )

        # Add risk level classification
        self.rankings["risk_level"] = self.rankings["mcda_score"].apply(self._classify_risk)

        # Merge with original project data
        if "project_name" in projects_df.columns:
            name_map = projects_df.set_index("project_id")["project_name"].to_dict()
            self.rankings["project_name"] = self.rankings["project_id"].map(name_map)

        self.rankings = self.rankings.sort_values("rank")

        logger.info(f"Ranked {len(self.rankings)} projects")

        return self.rankings

    def _prepare_matrix(
        self,
        df: pd.DataFrame,
        ml_scores: Optional[pd.Series],
        llm_scores: Optional[pd.Series],
    ) -> tuple[np.ndarray, list]:
        """Prepare the decision matrix for TOPSIS."""
        project_ids = df["project_id"].tolist()
        n_projects = len(project_ids)

        # Initialize matrix
        matrix = np.zeros((n_projects, len(self.criteria)))

        criteria_list = list(self.criteria.keys())

        for i, criterion in enumerate(criteria_list):
            if criterion == "ml_risk_score":
                if ml_scores is not None:
                    matrix[:, i] = ml_scores.values
                elif "risk_score" in df.columns:
                    matrix[:, i] = df["risk_score"].values
                else:
                    matrix[:, i] = 0.5  # Default medium risk

            elif criterion == "llm_sentiment_score":
                if llm_scores is not None:
                    # Normalize from -1,1 to 0,1
                    matrix[:, i] = (llm_scores.values + 1) / 2
                elif "sentiment_score" in df.columns:
                    matrix[:, i] = (df["sentiment_score"].values + 1) / 2
                else:
                    matrix[:, i] = 0.5  # Neutral

            elif criterion == "schedule_performance_index":
                if "schedule_performance_index" in df.columns:
                    # Cap at 2.0 and normalize
                    spi = df["schedule_performance_index"].clip(0, 2).values
                    matrix[:, i] = spi / 2
                else:
                    matrix[:, i] = 0.5

            elif criterion == "cost_performance_index":
                if "cost_performance_index" in df.columns:
                    cpi = df["cost_performance_index"].clip(0, 2).values
                    matrix[:, i] = cpi / 2
                else:
                    matrix[:, i] = 0.5

            elif criterion == "team_stability":
                if "team_stability" in df.columns:
                    matrix[:, i] = df["team_stability"].values
                elif "team_turnover" in df.columns:
                    matrix[:, i] = 1 - df["team_turnover"].clip(0, 1).values
                else:
                    matrix[:, i] = 0.9  # Default stable

        return matrix, project_ids

    def _classify_risk(self, score: float) -> str:
        """Classify risk level based on MCDA score."""
        if score >= 0.70:
            return "Low"
        elif score >= 0.40:
            return "Medium"
        else:
            return "High"

    def get_rankings(self) -> pd.DataFrame:
        """Get the current rankings."""
        if self.rankings is None:
            raise ValueError("No rankings available. Call rank() first.")
        return self.rankings

    def get_top_risk_projects(self, n: int = 5) -> pd.DataFrame:
        """Get top N high-risk projects."""
        if self.rankings is None:
            raise ValueError("No rankings available. Call rank() first.")
        return self.rankings.nsmallest(n, "mcda_score")

    def get_score_breakdown(self, project_id: str) -> dict:
        """Get detailed score breakdown for a project."""
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

        Args:
            projects_df: Project data
            weight_variation: Amount to vary weights (±)

        Returns:
            Dict with sensitivity results
        """
        original_rankings = self.rankings.copy() if self.rankings is not None else None

        results = {"weight_variation": weight_variation, "criteria_sensitivity": {}}

        for criterion in self.criteria:
            original_weight = self.criteria[criterion]["weight"]

            # Test with increased weight
            self.criteria[criterion]["weight"] = min(1.0, original_weight + weight_variation)
            self.rank(projects_df)
            increased_rankings = self.rankings["rank"].values

            # Test with decreased weight
            self.criteria[criterion]["weight"] = max(0.0, original_weight - weight_variation)
            self.rank(projects_df)
            decreased_rankings = self.rankings["rank"].values

            # Calculate rank changes
            if original_rankings is not None:
                original_ranks = original_rankings["rank"].values
                avg_change = (
                    np.mean(
                        np.abs(increased_rankings - original_ranks)
                        + np.abs(decreased_rankings - original_ranks)
                    )
                    / 2
                )
            else:
                avg_change = 0

            results["criteria_sensitivity"][criterion] = {
                "original_weight": original_weight,
                "avg_rank_change": float(avg_change),
            }

            # Restore original weight
            self.criteria[criterion]["weight"] = original_weight

        # Restore original rankings
        if original_rankings is not None:
            self.rankings = original_rankings

        return results
