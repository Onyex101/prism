"""
Utils Module
============

This module provides utility functions for logging and metrics.

Functions:
    setup_logger: Configure application logging.
    calculate_metrics: Calculate portfolio summary metrics.
    calculate_portfolio_health: Calculate portfolio health score.
    calculate_trend: Calculate metric trends.

Example:
    >>> from src.utils import setup_logger, calculate_metrics
    >>> setup_logger(log_level="INFO")
    >>> metrics = calculate_metrics(projects_df)
"""

from src.utils.logger import setup_logger
from src.utils.metrics import (
    calculate_metrics,
    calculate_portfolio_health,
    calculate_trend,
)

__all__ = [
    "setup_logger",
    "calculate_metrics",
    "calculate_portfolio_health",
    "calculate_trend",
]
