"""
Risk Response Engine
====================

A pure-Python, offline rule-based module that maps project risk metrics to
PMBOK risk response strategies (Avoid, Transfer, Mitigate, Accept).

This module supplements — but does not replace — the LLM analysis.  It
operates without an API key and produces deterministic, auditable
recommendations from structured project metrics.

PMBOK strategies used
---------------------
- **Avoid**    – Eliminate the threat by changing the project plan.
- **Transfer** – Shift the impact to a third party (e.g., outsourcing).
- **Mitigate** – Reduce the probability or impact of the threat.
- **Accept**   – Acknowledge the risk and either do nothing or prepare a
  contingency reserve.

Usage
-----
    from src.risk_response.engine import RiskResponseEngine

    engine = RiskResponseEngine()
    responses = engine.get_responses(project_row_dict)
    for r in responses:
        print(r.strategy, r.recommendation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RiskResponse:
    """A single actionable risk response recommendation.

    Attributes
    ----------
    strategy:
        PMBOK strategy label: ``"Avoid"``, ``"Transfer"``,
        ``"Mitigate"``, or ``"Accept"``.
    trigger:
        The metric name or condition that fired this rule.
    recommendation:
        Plain-language action for the project manager.
    """

    strategy: str
    trigger: str
    recommendation: str


# ---------------------------------------------------------------------------
# Rule definitions
# Each entry is a tuple:
#   (condition_fn, strategy, trigger_label, recommendation)
# Rules are evaluated in order; all matching rules fire (not first-match only).
# ---------------------------------------------------------------------------


def _val(project: dict, key: str, default: float = float("nan")) -> float:
    """Safely coerce a project metric to float."""
    raw = project.get(key)
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


_RULES: list[tuple[Any, str, str, str]] = [
    # Critical risk — activate contingency immediately
    (
        lambda p: _val(p, "risk_score") >= 0.8,
        "Accept",
        "risk_score ≥ 0.8",
        "Activate contingency reserve immediately; escalate to senior stakeholders and prepare a crisis response plan.",
    ),
    # High blocker concentration
    (
        lambda p: _val(p, "blocker_ratio") > 0.15,
        "Mitigate",
        "blocker_ratio > 0.15",
        "Escalate blockers to senior management; establish a daily unblock stand-up until ratio drops below 10%.",
    ),
    # Very low completion rate — scope is out of control
    (
        lambda p: _val(p, "completion_rate") < 30,
        "Avoid",
        "completion_rate < 30 %",
        "Reduce scope immediately; defer non-critical features to a future release and re-baseline the schedule.",
    ),
    # High defect density
    (
        lambda p: _val(p, "defect_rate") > 0.4,
        "Mitigate",
        "defect_rate > 0.40",
        "Enforce mandatory peer code review; introduce or expand regression test suite before next release.",
    ),
    # Significant team churn
    (
        lambda p: _val(p, "team_turnover") > 0.3,
        "Transfer",
        "team_turnover > 0.30",
        "Engage external specialists to fill gaps; capture critical knowledge in documentation and knowledge-base articles.",
    ),
    # Frequent issue re-opening suggests root causes unresolved
    (
        lambda p: _val(p, "reopen_rate") > 0.2,
        "Mitigate",
        "reopen_rate > 0.20",
        "Conduct root-cause analysis on repeatedly re-opened issues; implement definition-of-done checklist.",
    ),
    # Runaway scope churn
    (
        lambda p: _val(p, "churn_rate") > 0.5,
        "Avoid",
        "churn_rate > 0.50",
        "Freeze scope immediately; suspend new change requests until backlog stabilises.",
    ),
    # Poor schedule performance
    (
        lambda p: _val(p, "schedule_performance_index") < 0.7,
        "Mitigate",
        "SPI < 0.70",
        "Re-plan remaining work; apply schedule compression (fast-tracking or crashing) on the critical path.",
    ),
    # Moderate risk — monitor only
    (
        lambda p: 0.3 <= _val(p, "risk_score") < 0.6,
        "Accept",
        "risk_score 0.30–0.60",
        "Risk is within acceptable tolerance; review weekly and update risk register. No immediate action required.",
    ),
]


class RiskResponseEngine:
    """Evaluate all rules against a project metrics dictionary and return
    the list of triggered :class:`RiskResponse` objects.

    Multiple rules can fire for a single project — they are returned in
    priority order (highest-severity rules first).
    """

    def get_responses(self, project: dict) -> list[RiskResponse]:
        """Return all applicable risk responses for *project*.

        Parameters
        ----------
        project:
            A dictionary of project metrics (typically one row from a
            Pandas DataFrame converted via ``row.to_dict()`` or
            ``df.iterrows()``).

        Returns
        -------
        list[RiskResponse]
            Zero or more responses, ordered by rule priority.
        """
        responses: list[RiskResponse] = []
        for condition, strategy, trigger, recommendation in _RULES:
            try:
                if condition(project):
                    responses.append(
                        RiskResponse(
                            strategy=strategy,
                            trigger=trigger,
                            recommendation=recommendation,
                        )
                    )
            except Exception:
                continue
        return responses
