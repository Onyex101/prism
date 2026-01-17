"""
MCDA Module
===========

This module provides Multi-Criteria Decision Analysis functionality
for ranking projects based on multiple criteria.

Classes:
    TOPSIS: TOPSIS algorithm implementation.
    ProjectRanker: Project ranking using MCDA.

Example:
    >>> from src.mcda import ProjectRanker, TOPSIS
    >>> ranker = ProjectRanker()
    >>> rankings = ranker.rank(projects_df, ml_scores, llm_scores)
    >>>
    >>> # Or use TOPSIS directly
    >>> topsis = TOPSIS(weights=[0.4, 0.3, 0.3])
    >>> topsis.fit(decision_matrix)
"""

from src.mcda.ranker import ProjectRanker
from src.mcda.topsis import TOPSIS

__all__ = [
    "TOPSIS",
    "ProjectRanker",
]
