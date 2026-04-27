"""Contract smoke tests for :class:`src.integrations.base.ProjectManagementConnector`."""

from __future__ import annotations

import inspect

import pandas as pd

from src.integrations.base import ProjectManagementConnector, ProjectSummary, RawProjectData
from src.integrations.jira_cloud import JiraCloudConnector


class _MinimalFake(ProjectManagementConnector):
    def test_connection(self) -> dict[str, str]:
        return {}

    def list_projects(self):
        return [ProjectSummary(key="K", name="N")]

    def fetch_project_data(self, project_key, max_issues, jql_extra="", progress=None):
        df = pd.DataFrame(
            {
                "key": [],
                "resolution.name": [],
                "priority.name": [],
                "status.name": [],
                "issuetype.name": [],
                "project.key": [],
                "project.name": [],
                "created": [],
                "updated": [],
                "resolutiondate": [],
                "assignee": [],
                "creator": [],
                "reporter": [],
                "votes.votes": [],
                "watches.watchCount": [],
            }
        )
        return RawProjectData(
            project_key=project_key,
            issues=df,
            comments={},
            changelog_stats={},
            descriptions={},
        )


def test_jira_cloud_connector_is_subclass():
    assert issubclass(JiraCloudConnector, ProjectManagementConnector)


def test_abstract_methods_implemented():
    for name, method in inspect.getmembers(ProjectManagementConnector, predicate=inspect.isfunction):
        if getattr(method, "__isabstractmethod__", False):
            assert hasattr(_MinimalFake, name)
