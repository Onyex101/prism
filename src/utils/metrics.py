"""
Metrics Utilities

Common metric calculations for project analysis.
"""

from typing import Any

import numpy as np
import pandas as pd


def calculate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate summary metrics for a project portfolio.

    Args:
        df: DataFrame with project data

    Returns:
        Dict with calculated metrics
    """
    metrics = {
        "total_projects": len(df),
    }

    # Risk distribution
    if "risk_level" in df.columns:
        risk_counts = df["risk_level"].value_counts().to_dict()
        metrics["risk_distribution"] = risk_counts
        metrics["high_risk_count"] = risk_counts.get("High", 0)
        metrics["medium_risk_count"] = risk_counts.get("Medium", 0)
        metrics["low_risk_count"] = risk_counts.get("Low", 0)

    # Average metrics
    if "risk_score" in df.columns or "mcda_score" in df.columns:
        score_col = "risk_score" if "risk_score" in df.columns else "mcda_score"
        metrics["avg_risk_score"] = float(df[score_col].mean())
        metrics["min_risk_score"] = float(df[score_col].min())
        metrics["max_risk_score"] = float(df[score_col].max())

    # Budget metrics
    if "budget" in df.columns and "spent" in df.columns:
        total_budget = df["budget"].sum()
        total_spent = df["spent"].sum()
        metrics["total_budget"] = float(total_budget)
        metrics["total_spent"] = float(total_spent)
        metrics["budget_utilization"] = float(total_spent / total_budget) if total_budget > 0 else 0

    # Completion metrics
    if "completion_rate" in df.columns:
        metrics["avg_completion"] = float(df["completion_rate"].mean())
        metrics["projects_near_complete"] = int((df["completion_rate"] >= 90).sum())

    # Team metrics
    if "team_size" in df.columns:
        metrics["total_team_members"] = int(df["team_size"].sum())
        metrics["avg_team_size"] = float(df["team_size"].mean())

    return metrics


def calculate_portfolio_health(metrics: dict) -> dict:
    """
    Calculate overall portfolio health score.

    Args:
        metrics: Dict of calculated metrics

    Returns:
        Dict with health assessment
    """
    scores = []

    # Risk score component (inverted - lower risk is better)
    if "avg_risk_score" in metrics:
        risk_component = 1 - metrics["avg_risk_score"]
        scores.append(("risk", risk_component, 0.4))

    # Budget component
    if "budget_utilization" in metrics:
        util = metrics["budget_utilization"]
        # Score is high if utilization is close to 1 (on budget)
        budget_component = 1 - abs(1 - util) if util <= 2 else 0
        scores.append(("budget", budget_component, 0.2))

    # Completion component
    if "avg_completion" in metrics:
        completion_component = metrics["avg_completion"] / 100
        scores.append(("completion", completion_component, 0.2))

    # Low risk ratio
    if all(k in metrics for k in ["high_risk_count", "total_projects"]):
        total = metrics["total_projects"]
        if total > 0:
            high_risk_ratio = metrics["high_risk_count"] / total
            ratio_component = 1 - high_risk_ratio
            scores.append(("low_risk_ratio", ratio_component, 0.2))

    # Calculate weighted average
    if scores:
        total_weight = sum(s[2] for s in scores)
        health_score = sum(s[1] * s[2] for s in scores) / total_weight
    else:
        health_score = 0.5

    # Determine health level
    if health_score >= 0.7:
        health_level = "Healthy"
    elif health_score >= 0.4:
        health_level = "At Risk"
    else:
        health_level = "Critical"

    return {
        "health_score": round(health_score, 2),
        "health_level": health_level,
        "components": {s[0]: round(s[1], 2) for s in scores},
    }


def calculate_trend(
    current_metrics: dict,
    previous_metrics: dict,
) -> dict:
    """
    Calculate trend between two periods.

    Args:
        current_metrics: Current period metrics
        previous_metrics: Previous period metrics

    Returns:
        Dict with trend indicators
    """
    trends = {}

    comparable_metrics = [
        "avg_risk_score",
        "budget_utilization",
        "avg_completion",
        "high_risk_count",
    ]

    for metric in comparable_metrics:
        if metric in current_metrics and metric in previous_metrics:
            current = current_metrics[metric]
            previous = previous_metrics[metric]

            if previous != 0:
                change_pct = ((current - previous) / abs(previous)) * 100
            else:
                change_pct = 0

            # Determine direction
            if abs(change_pct) < 5:
                direction = "stable"
            elif metric in ["avg_risk_score", "high_risk_count"]:
                # Lower is better for risk
                direction = "improving" if change_pct < 0 else "worsening"
            else:
                direction = "improving" if change_pct > 0 else "worsening"

            trends[metric] = {
                "current": current,
                "previous": previous,
                "change_pct": round(change_pct, 1),
                "direction": direction,
            }

    return trends
