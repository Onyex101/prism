"""
Visualization Module
====================

This module provides chart generation for risk visualization.

Classes:
    RiskCharts: Generate various chart types for risk data.

Example:
    >>> from src.visualization import RiskCharts
    >>> fig = RiskCharts.risk_distribution_pie(rankings_df)
    >>> fig.show()
"""

from src.visualization.risk_charts import RiskCharts

__all__ = [
    "RiskCharts",
]
