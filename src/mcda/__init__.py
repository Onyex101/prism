"""
MCDA Module

Multi-Criteria Decision Analysis for project ranking.
"""

from .topsis import TOPSIS
from .ranker import ProjectRanker

__all__ = ["TOPSIS", "ProjectRanker"]
