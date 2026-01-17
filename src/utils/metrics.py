"""
Metrics Utilities
=================

This module provides common metric calculations for project analysis.

It includes functions for calculating summary metrics, portfolio health
scores, and trend analysis.

Example:
    >>> from src.utils.metrics import calculate_metrics, calculate_portfolio_health
    >>> metrics = calculate_metrics(projects_df)
    >>> health = calculate_portfolio_health(metrics)
"""

from typing import Any

import numpy as np
import pandas as pd


def calculate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate summary metrics for a project portfolio.

    :param df: DataFrame with project data.
    :type df: pd.DataFrame
    :return: Dict with calculated metrics including risk distribution,
        budget metrics, completion metrics, and team metrics.
    :rtype: dict[str, Any]

    Example:
        >>> metrics = calculate_metrics(projects_df)
        >>> print(f"High risk projects: {metrics['high_risk_count']}")
    """
    metrics: dict[str, Any] = {
        "total_projects": len(df),
    }

    _add_risk_metrics(metrics, df)
    _add_score_metrics(metrics, df)
    _add_budget_metrics(metrics, df)
    _add_completion_metrics(metrics, df)
    _add_team_metrics(metrics, df)

    return metrics


def _add_risk_metrics(metrics: dict[str, Any], df: pd.DataFrame) -> None:
    """
    Add risk distribution metrics.

    :param metrics: Metrics dict to update.
    :type metrics: dict[str, Any]
    :param df: Project data.
    :type df: pd.DataFrame
    """
    if "risk_level" not in df.columns:
        return

    risk_counts = df["risk_level"].value_counts().to_dict()
    metrics["risk_distribution"] = risk_counts
    metrics["high_risk_count"] = risk_counts.get("High", 0)
    metrics["medium_risk_count"] = risk_counts.get("Medium", 0)
    metrics["low_risk_count"] = risk_counts.get("Low", 0)


def _add_score_metrics(metrics: dict[str, Any], df: pd.DataFrame) -> None:
    """
    Add risk score metrics.

    :param metrics: Metrics dict to update.
    :type metrics: dict[str, Any]
    :param df: Project data.
    :type df: pd.DataFrame
    """
    score_col = "risk_score" if "risk_score" in df.columns else "mcda_score"

    if score_col not in df.columns:
        return

    metrics["avg_risk_score"] = float(df[score_col].mean())
    metrics["min_risk_score"] = float(df[score_col].min())
    metrics["max_risk_score"] = float(df[score_col].max())


def _add_budget_metrics(metrics: dict[str, Any], df: pd.DataFrame) -> None:
    """
    Add budget-related metrics.

    :param metrics: Metrics dict to update.
    :type metrics: dict[str, Any]
    :param df: Project data.
    :type df: pd.DataFrame
    """
    if "budget" not in df.columns or "spent" not in df.columns:
        return

    total_budget = df["budget"].sum()
    total_spent = df["spent"].sum()

    metrics["total_budget"] = float(total_budget)
    metrics["total_spent"] = float(total_spent)
    metrics["budget_utilization"] = float(total_spent / total_budget) if total_budget > 0 else 0


def _add_completion_metrics(metrics: dict[str, Any], df: pd.DataFrame) -> None:
    """
    Add completion-related metrics.

    :param metrics: Metrics dict to update.
    :type metrics: dict[str, Any]
    :param df: Project data.
    :type df: pd.DataFrame
    """
    if "completion_rate" not in df.columns:
        return

    metrics["avg_completion"] = float(df["completion_rate"].mean())
    metrics["projects_near_complete"] = int((df["completion_rate"] >= 90).sum())


def _add_team_metrics(metrics: dict[str, Any], df: pd.DataFrame) -> None:
    """
    Add team-related metrics.

    :param metrics: Metrics dict to update.
    :type metrics: dict[str, Any]
    :param df: Project data.
    :type df: pd.DataFrame
    """
    if "team_size" not in df.columns:
        return

    metrics["total_team_members"] = int(df["team_size"].sum())
    metrics["avg_team_size"] = float(df["team_size"].mean())


def calculate_portfolio_health(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate overall portfolio health score.

    Combines multiple metrics into a single health score with
    component breakdown.

    :param metrics: Dict of calculated metrics from calculate_metrics().
    :type metrics: dict[str, Any]
    :return: Dict with health_score (0-1), health_level, and components.
    :rtype: dict[str, Any]

    Example:
        >>> metrics = calculate_metrics(df)
        >>> health = calculate_portfolio_health(metrics)
        >>> print(f"Portfolio health: {health['health_level']}")
    """
    scores: list[tuple[str, float, float]] = []

    _add_risk_score_component(scores, metrics)
    _add_budget_component(scores, metrics)
    _add_completion_component(scores, metrics)
    _add_low_risk_ratio_component(scores, metrics)

    health_score = _calculate_weighted_score(scores)
    health_level = _classify_health_level(health_score)

    return {
        "health_score": round(health_score, 2),
        "health_level": health_level,
        "components": {s[0]: round(s[1], 2) for s in scores},
    }


def _add_risk_score_component(
    scores: list[tuple[str, float, float]], metrics: dict[str, Any]
) -> None:
    """Add risk score component."""
    if "avg_risk_score" in metrics:
        risk_component = 1 - metrics["avg_risk_score"]
        scores.append(("risk", risk_component, 0.4))


def _add_budget_component(scores: list[tuple[str, float, float]], metrics: dict[str, Any]) -> None:
    """Add budget component."""
    if "budget_utilization" not in metrics:
        return

    util = metrics["budget_utilization"]
    budget_component = 1 - abs(1 - util) if util <= 2 else 0
    scores.append(("budget", budget_component, 0.2))


def _add_completion_component(
    scores: list[tuple[str, float, float]], metrics: dict[str, Any]
) -> None:
    """Add completion component."""
    if "avg_completion" in metrics:
        completion_component = metrics["avg_completion"] / 100
        scores.append(("completion", completion_component, 0.2))


def _add_low_risk_ratio_component(
    scores: list[tuple[str, float, float]], metrics: dict[str, Any]
) -> None:
    """Add low risk ratio component."""
    required_keys = ["high_risk_count", "total_projects"]
    if not all(k in metrics for k in required_keys):
        return

    total = metrics["total_projects"]
    if total > 0:
        high_risk_ratio = metrics["high_risk_count"] / total
        ratio_component = 1 - high_risk_ratio
        scores.append(("low_risk_ratio", ratio_component, 0.2))


def _calculate_weighted_score(scores: list[tuple[str, float, float]]) -> float:
    """Calculate weighted average of scores."""
    if not scores:
        return 0.5

    total_weight = sum(s[2] for s in scores)
    return sum(s[1] * s[2] for s in scores) / total_weight


def _classify_health_level(health_score: float) -> str:
    """Classify health level based on score."""
    if health_score >= 0.7:
        return "Healthy"
    if health_score >= 0.4:
        return "At Risk"
    return "Critical"


def calculate_trend(
    current_metrics: dict[str, Any],
    previous_metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate trend between two periods.

    :param current_metrics: Current period metrics.
    :type current_metrics: dict[str, Any]
    :param previous_metrics: Previous period metrics.
    :type previous_metrics: dict[str, Any]
    :return: Dict with trend indicators for each comparable metric.
    :rtype: dict[str, Any]

    Example:
        >>> current = calculate_metrics(current_df)
        >>> previous = calculate_metrics(previous_df)
        >>> trend = calculate_trend(current, previous)
        >>> print(f"Risk trend: {trend['avg_risk_score']['direction']}")
    """
    trends: dict[str, Any] = {}

    comparable_metrics = [
        "avg_risk_score",
        "budget_utilization",
        "avg_completion",
        "high_risk_count",
    ]

    for metric in comparable_metrics:
        if metric in current_metrics and metric in previous_metrics:
            trends[metric] = _calculate_metric_trend(
                metric, current_metrics[metric], previous_metrics[metric]
            )

    return trends


def _calculate_metric_trend(metric: str, current: float, previous: float) -> dict[str, Any]:
    """
    Calculate trend for a single metric.

    :param metric: Metric name.
    :type metric: str
    :param current: Current value.
    :type current: float
    :param previous: Previous value.
    :type previous: float
    :return: Trend dictionary.
    :rtype: dict[str, Any]
    """
    if previous != 0:
        change_pct = ((current - previous) / abs(previous)) * 100
    else:
        change_pct = 0

    direction = _determine_trend_direction(metric, change_pct)

    return {
        "current": current,
        "previous": previous,
        "change_pct": round(change_pct, 1),
        "direction": direction,
    }


def _determine_trend_direction(metric: str, change_pct: float) -> str:
    """Determine trend direction based on metric type and change."""
    if abs(change_pct) < 5:
        return "stable"

    # Lower is better for these metrics
    lower_is_better = ["avg_risk_score", "high_risk_count"]

    if metric in lower_is_better:
        return "improving" if change_pct < 0 else "worsening"
    return "improving" if change_pct > 0 else "worsening"
