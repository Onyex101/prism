"""Tests for :class:`src.integrations.jira_sync_service.JiraSyncService`."""

from __future__ import annotations

import pandas as pd

from src.data.jira_aggregator import JiraAggregator
from src.integrations.base import ProjectSummary, RawProjectData
from src.integrations.jira_sync_service import JiraSyncService


class FakeConnector:
    """Minimal connector returning canned raw data."""

    def test_connection(self) -> dict[str, str]:
        return {"display_name": "x"}

    def list_projects(self):
        return [ProjectSummary(key="P", name="P")]

    def fetch_project_data(self, project_key, max_issues, jql_extra="", progress=None):
        issues = pd.DataFrame(
            {
                "key": ["P-1"],
                "resolution.name": [None],
                "priority.name": ["Medium"],
                "status.name": ["Open"],
                "issuetype.name": ["Task"],
                "project.key": ["P"],
                "project.name": ["Proj"],
                "created": ["2020-01-01"],
                "updated": ["2020-02-01"],
                "resolutiondate": [None],
                "assignee": [""],
                "creator": [""],
                "reporter": [""],
                "votes.votes": [0],
                "watches.watchCount": [0],
            }
        )
        return RawProjectData(
            project_key=project_key,
            issues=issues,
            comments={project_key: []},
            changelog_stats={project_key: {"status_changes": 0, "reopens": 0}},
            descriptions={project_key: []},
        )


def test_sync_produces_prism_dataframe():
    svc = JiraSyncService(FakeConnector(), JiraAggregator())
    df = svc.sync(["P"], max_issues_per_project=10, progress=None)
    assert "project_id" in df.columns
    assert "department" in df.columns
    assert df.iloc[0]["department"] == "Jira Cloud"
