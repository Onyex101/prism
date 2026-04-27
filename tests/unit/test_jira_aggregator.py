"""Regression tests for :class:`src.data.jira_aggregator.JiraAggregator`."""

from __future__ import annotations

import pandas as pd

from src.data.jira_aggregator import JiraAggregator


def test_jira_aggregator_single_project_shape():
    """One project row with expected columns and risk labels."""
    issues = pd.DataFrame(
        {
            "key": ["P-1", "P-2"],
            "resolution.name": ["Fixed", None],
            "priority.name": ["Medium", "Medium"],
            "status.name": ["Closed", "Open"],
            "issuetype.name": ["Bug", "Task"],
            "project.key": ["P", "P"],
            "project.name": ["Proj", "Proj"],
            "created": ["2020-01-01", "2020-02-01"],
            "updated": ["2020-03-01", "2020-04-01"],
            "resolutiondate": ["2020-03-15", None],
            "assignee": ["a", "b"],
            "creator": ["c", "c"],
            "reporter": ["d", "d"],
            "votes.votes": [1, 0],
            "watches.watchCount": [2, 1],
        }
    )
    agg = JiraAggregator()
    out = agg.aggregate(
        issues,
        {"P": ["comment one", "comment two"]},
        {"P": {"status_changes": 3, "reopens": 1}},
        {"P": ["desc"]},
        show_progress=False,
    )
    assert len(out) == 1
    assert out.iloc[0]["project_id"] == "P"
    assert "risk_level" in out.columns
    assert "risk_score_composite" in out.columns


def test_jira_aggregator_empty_issues():
    """Empty issues frame yields empty output with risk columns."""
    issues = pd.DataFrame(
        columns=[
            "key",
            "resolution.name",
            "priority.name",
            "status.name",
            "issuetype.name",
            "project.key",
            "project.name",
            "created",
            "updated",
            "resolutiondate",
            "assignee",
            "creator",
            "reporter",
            "votes.votes",
            "watches.watchCount",
        ]
    )
    agg = JiraAggregator()
    out = agg.aggregate(issues, {}, {}, {}, show_progress=False)
    assert len(out) == 0
