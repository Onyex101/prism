"""
TOPSIS Algorithm Implementation

Technique for Order of Preference by Similarity to Ideal Solution.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class TOPSIS:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

    A multi-criteria decision analysis method that ranks alternatives based on
    their distance from ideal and negative-ideal solutions.
    """

    def __init__(
        self,
        weights: Optional[list[float]] = None,
        criteria_types: Optional[list[str]] = None,
    ):
        """
        Initialize TOPSIS.

        Args:
            weights: List of criteria weights (must sum to 1)
            criteria_types: List of 'benefit' or 'cost' for each criterion
        """
        self.weights = weights
        self.criteria_types = criteria_types
        self.normalized_matrix: Optional[np.ndarray] = None
        self.weighted_matrix: Optional[np.ndarray] = None
        self.ideal_solution: Optional[np.ndarray] = None
        self.negative_ideal: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None

    def fit(
        self,
        decision_matrix: np.ndarray,
        weights: Optional[list[float]] = None,
        criteria_types: Optional[list[str]] = None,
    ) -> "TOPSIS":
        """
        Fit TOPSIS model and calculate scores.

        Args:
            decision_matrix: Matrix of alternatives x criteria
            weights: Criteria weights (optional, overrides init weights)
            criteria_types: Criteria types (optional, overrides init types)

        Returns:
            Self for method chaining
        """
        if weights is not None:
            self.weights = weights
        if criteria_types is not None:
            self.criteria_types = criteria_types

        n_alternatives, n_criteria = decision_matrix.shape

        # Validate weights
        if self.weights is None:
            self.weights = [1.0 / n_criteria] * n_criteria
        else:
            if len(self.weights) != n_criteria:
                raise ValueError(
                    f"Weights length ({len(self.weights)}) != criteria count ({n_criteria})"
                )
            # Normalize weights to sum to 1
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

        # Validate criteria types
        if self.criteria_types is None:
            self.criteria_types = ["benefit"] * n_criteria
        elif len(self.criteria_types) != n_criteria:
            raise ValueError("Criteria types length must match number of criteria")

        # Step 1: Normalize the decision matrix (vector normalization)
        self.normalized_matrix = self._normalize(decision_matrix)

        # Step 2: Apply weights
        self.weighted_matrix = self.normalized_matrix * np.array(self.weights)

        # Step 3: Determine ideal and negative-ideal solutions
        self.ideal_solution, self.negative_ideal = self._get_ideal_solutions()

        # Step 4: Calculate distances and scores
        self.scores = self._calculate_scores()

        logger.info(f"TOPSIS fitted on {n_alternatives} alternatives with {n_criteria} criteria")

        return self

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize using vector normalization."""
        # Avoid division by zero
        norms = np.sqrt((matrix**2).sum(axis=0))
        norms[norms == 0] = 1  # Avoid division by zero
        return matrix / norms

    def _get_ideal_solutions(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate ideal and negative-ideal solutions."""
        n_criteria = self.weighted_matrix.shape[1]

        ideal = np.zeros(n_criteria)
        negative_ideal = np.zeros(n_criteria)

        for j in range(n_criteria):
            if self.criteria_types[j] == "benefit":
                ideal[j] = self.weighted_matrix[:, j].max()
                negative_ideal[j] = self.weighted_matrix[:, j].min()
            else:  # cost
                ideal[j] = self.weighted_matrix[:, j].min()
                negative_ideal[j] = self.weighted_matrix[:, j].max()

        return ideal, negative_ideal

    def _calculate_scores(self) -> np.ndarray:
        """Calculate TOPSIS scores (closeness coefficient)."""
        # Euclidean distance to ideal solution
        d_plus = np.sqrt(((self.weighted_matrix - self.ideal_solution) ** 2).sum(axis=1))

        # Euclidean distance to negative-ideal solution
        d_minus = np.sqrt(((self.weighted_matrix - self.negative_ideal) ** 2).sum(axis=1))

        # Closeness coefficient (0-1, higher is better)
        # Avoid division by zero
        denominator = d_plus + d_minus
        denominator[denominator == 0] = 1

        scores = d_minus / denominator

        return scores

    def get_scores(self) -> np.ndarray:
        """Get the calculated TOPSIS scores."""
        if self.scores is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.scores

    def get_ranking(self) -> np.ndarray:
        """Get ranking (1 = best)."""
        if self.scores is None:
            raise ValueError("Model not fitted. Call fit() first.")
        # Higher score = better, so rank in descending order
        return np.argsort(np.argsort(-self.scores)) + 1

    def rank_alternatives(
        self,
        decision_matrix: np.ndarray,
        alternative_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Rank alternatives and return as DataFrame.

        Args:
            decision_matrix: Matrix of alternatives x criteria
            alternative_names: Names for each alternative

        Returns:
            DataFrame with rankings
        """
        self.fit(decision_matrix)

        n_alternatives = len(self.scores)

        if alternative_names is None:
            alternative_names = [f"Alt_{i+1}" for i in range(n_alternatives)]

        result = pd.DataFrame(
            {
                "alternative": alternative_names,
                "score": self.scores,
                "rank": self.get_ranking(),
            }
        )

        return result.sort_values("rank")
