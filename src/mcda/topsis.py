"""
TOPSIS Algorithm Implementation
===============================

This module implements the TOPSIS (Technique for Order of Preference by
Similarity to Ideal Solution) algorithm for multi-criteria decision analysis.

TOPSIS ranks alternatives based on their geometric distance from the ideal
and negative-ideal solutions.

Example:
    >>> from src.mcda.topsis import TOPSIS
    >>> topsis = TOPSIS(weights=[0.4, 0.3, 0.3], criteria_types=["cost", "benefit", "benefit"])
    >>> topsis.fit(decision_matrix)
    >>> scores = topsis.get_scores()

Reference:
    Hwang, C.L.; Yoon, K. (1981). Multiple Attribute Decision Making:
    Methods and Applications. New York: Springer-Verlag.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class TOPSIS:
    """
    TOPSIS multi-criteria decision analysis method.

    TOPSIS ranks alternatives based on their distance from ideal and
    negative-ideal solutions. It follows the principle that the chosen
    alternative should have the shortest distance from the ideal solution
    and the farthest distance from the negative-ideal solution.

    :ivar weights: Criteria weights (sum to 1).
    :vartype weights: Optional[list[float]]
    :ivar criteria_types: Type of each criterion ("benefit" or "cost").
    :vartype criteria_types: Optional[list[str]]
    :ivar normalized_matrix: Vector-normalized decision matrix.
    :vartype normalized_matrix: Optional[np.ndarray]
    :ivar weighted_matrix: Weighted normalized matrix.
    :vartype weighted_matrix: Optional[np.ndarray]
    :ivar ideal_solution: Ideal solution vector.
    :vartype ideal_solution: Optional[np.ndarray]
    :ivar negative_ideal: Negative-ideal solution vector.
    :vartype negative_ideal: Optional[np.ndarray]
    :ivar scores: TOPSIS closeness coefficients.
    :vartype scores: Optional[np.ndarray]

    Example:
        >>> topsis = TOPSIS()
        >>> result_df = topsis.rank_alternatives(
        ...     decision_matrix,
        ...     alternative_names=["Project A", "Project B", "Project C"]
        ... )
    """

    def __init__(
        self,
        weights: Optional[list[float]] = None,
        criteria_types: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize TOPSIS.

        :param weights: List of criteria weights. Must sum to 1 or will
            be normalized. If None, equal weights are used.
        :type weights: Optional[list[float]]
        :param criteria_types: List of "benefit" or "cost" for each criterion.
            Benefit criteria: higher is better.
            Cost criteria: lower is better.
        :type criteria_types: Optional[list[str]]
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

        :param decision_matrix: Matrix of shape (n_alternatives, n_criteria).
        :type decision_matrix: np.ndarray
        :param weights: Criteria weights (overrides init weights).
        :type weights: Optional[list[float]]
        :param criteria_types: Criteria types (overrides init types).
        :type criteria_types: Optional[list[str]]
        :return: Self for method chaining.
        :rtype: TOPSIS
        :raises ValueError: If weights length doesn't match criteria count.

        Example:
            >>> topsis = TOPSIS()
            >>> topsis.fit(matrix, weights=[0.4, 0.3, 0.3])
        """
        self._update_parameters(weights, criteria_types)

        n_alternatives, n_criteria = decision_matrix.shape
        self._validate_and_normalize_weights(n_criteria)
        self._validate_criteria_types(n_criteria)

        # TOPSIS algorithm steps
        self.normalized_matrix = self._normalize(decision_matrix)
        self.weighted_matrix = self.normalized_matrix * np.array(self.weights)
        self.ideal_solution, self.negative_ideal = self._get_ideal_solutions()
        self.scores = self._calculate_scores()

        logger.info(f"TOPSIS fitted on {n_alternatives} alternatives with {n_criteria} criteria")

        return self

    def _update_parameters(
        self,
        weights: Optional[list[float]],
        criteria_types: Optional[list[str]],
    ) -> None:
        """
        Update parameters if provided.

        :param weights: New weights.
        :type weights: Optional[list[float]]
        :param criteria_types: New criteria types.
        :type criteria_types: Optional[list[str]]
        """
        if weights is not None:
            self.weights = weights
        if criteria_types is not None:
            self.criteria_types = criteria_types

    def _validate_and_normalize_weights(self, n_criteria: int) -> None:
        """
        Validate and normalize weights.

        :param n_criteria: Number of criteria.
        :type n_criteria: int
        :raises ValueError: If weights length doesn't match.
        """
        if self.weights is None:
            self.weights = [1.0 / n_criteria] * n_criteria
            return

        if len(self.weights) != n_criteria:
            raise ValueError(
                f"Weights length ({len(self.weights)}) != criteria count ({n_criteria})"
            )

        # Normalize weights to sum to 1
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def _validate_criteria_types(self, n_criteria: int) -> None:
        """
        Validate criteria types.

        :param n_criteria: Number of criteria.
        :type n_criteria: int
        :raises ValueError: If criteria types length doesn't match.
        """
        if self.criteria_types is None:
            self.criteria_types = ["benefit"] * n_criteria
        elif len(self.criteria_types) != n_criteria:
            raise ValueError("Criteria types length must match number of criteria")

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize using vector normalization.

        :param matrix: Decision matrix.
        :type matrix: np.ndarray
        :return: Normalized matrix.
        :rtype: np.ndarray
        """
        norms = np.sqrt((matrix**2).sum(axis=0))
        norms[norms == 0] = 1  # Avoid division by zero
        return matrix / norms

    def _get_ideal_solutions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate ideal and negative-ideal solutions.

        :return: Tuple of (ideal_solution, negative_ideal).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        n_criteria = self.weighted_matrix.shape[1]

        ideal = np.zeros(n_criteria)
        negative_ideal = np.zeros(n_criteria)

        for j in range(n_criteria):
            column = self.weighted_matrix[:, j]
            if self.criteria_types[j] == "benefit":
                ideal[j] = column.max()
                negative_ideal[j] = column.min()
            else:  # cost
                ideal[j] = column.min()
                negative_ideal[j] = column.max()

        return ideal, negative_ideal

    def _calculate_scores(self) -> np.ndarray:
        """
        Calculate TOPSIS closeness coefficients.

        :return: Array of scores (0-1, higher is better).
        :rtype: np.ndarray
        """
        # Euclidean distance to ideal solution
        d_plus = np.sqrt(((self.weighted_matrix - self.ideal_solution) ** 2).sum(axis=1))

        # Euclidean distance to negative-ideal solution
        d_minus = np.sqrt(((self.weighted_matrix - self.negative_ideal) ** 2).sum(axis=1))

        # Closeness coefficient
        denominator = d_plus + d_minus
        denominator[denominator == 0] = 1  # Avoid division by zero

        return d_minus / denominator

    def get_scores(self) -> np.ndarray:
        """
        Get the calculated TOPSIS scores.

        :return: Array of closeness coefficients.
        :rtype: np.ndarray
        :raises ValueError: If model not fitted.

        Example:
            >>> scores = topsis.get_scores()
            >>> best_alternative = np.argmax(scores)
        """
        if self.scores is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.scores

    def get_ranking(self) -> np.ndarray:
        """
        Get ranking (1 = best).

        :return: Array of ranks.
        :rtype: np.ndarray
        :raises ValueError: If model not fitted.

        Example:
            >>> ranks = topsis.get_ranking()
            >>> print(f"Best alternative: {np.where(ranks == 1)[0][0]}")
        """
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

        :param decision_matrix: Matrix of alternatives x criteria.
        :type decision_matrix: np.ndarray
        :param alternative_names: Names for each alternative.
        :type alternative_names: Optional[list[str]]
        :return: DataFrame with rankings.
        :rtype: pd.DataFrame

        Example:
            >>> result = topsis.rank_alternatives(
            ...     matrix,
            ...     alternative_names=["Option A", "Option B"]
            ... )
            >>> print(result.head())
        """
        self.fit(decision_matrix)

        n_alternatives = len(self.scores)

        if alternative_names is None:
            alternative_names = [f"Alt_{i + 1}" for i in range(n_alternatives)]

        result = pd.DataFrame(
            {
                "alternative": alternative_names,
                "score": self.scores,
                "rank": self.get_ranking(),
            }
        )

        return result.sort_values("rank")
