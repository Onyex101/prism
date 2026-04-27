"""
Orchestrate multi-project Jira fetch + PRISM aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger

from src.data.jira_aggregator import JiraAggregator
from src.integrations.base import ProjectManagementConnector


class JiraSyncService:
    """
    Fetch selected projects from a connector and produce a PRISM DataFrame.

    :param connector: Injected PM connector (typically :class:`~src.integrations.jira_cloud.JiraCloudConnector`).
    :type connector: ProjectManagementConnector
    :param aggregator: Injected :class:`~src.data.jira_aggregator.JiraAggregator`.
    :type aggregator: JiraAggregator

    Example:
        >>> JiraSyncService(connector, JiraAggregator())
    """

    def __init__(
        self,
        connector: ProjectManagementConnector,
        aggregator: JiraAggregator,
    ) -> None:
        self._connector = connector
        self._aggregator = aggregator

    def test_connection(self) -> dict[str, str]:
        """
        Delegate to the connector's ``test_connection``.

        :return: Identity payload.
        :rtype: dict[str, str]
        """
        return self._connector.test_connection()

    def list_projects(self):
        """
        Delegate to the connector's ``list_projects``.

        :return: Project summaries.
        :rtype: list[src.integrations.base.ProjectSummary]
        """
        return self._connector.list_projects()

    def sync(
        self,
        project_keys: list[str],
        max_issues_per_project: int,
        jql_extra: str = "",
        progress: Optional[Callable[[str, int, int], None]] = None,
        *,
        save_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Fetch and aggregate one or more Jira projects into PRISM rows.

        :param project_keys: Jira project keys to include.
        :type project_keys: list[str]
        :param max_issues_per_project: Cap issues fetched per project.
        :type max_issues_per_project: int
        :param jql_extra: Extra JQL ``AND`` clauses.
        :type jql_extra: str
        :param progress: Optional ``(phase, current, total)`` callback.
        :type progress: collections.abc.Callable[[str, int, int], None] | None
        :param save_path: If set, write CSV to this path.
        :type save_path: pathlib.Path | None
        :return: Project-level PRISM DataFrame.
        :rtype: pandas.DataFrame
        """
        all_issues: list[pd.DataFrame] = []
        comments: dict[str, list[str]] = {}
        changelog: dict[str, dict[str, int]] = {}
        descriptions: dict[str, list[str]] = {}

        for i, pkey in enumerate(project_keys):
            if progress:
                progress("project", i + 1, len(project_keys))
            raw = self._connector.fetch_project_data(
                pkey,
                max_issues_per_project,
                jql_extra=jql_extra,
                progress=progress,
            )
            if len(raw.issues) > 0:
                all_issues.append(raw.issues)
            comments.update(raw.comments)
            for ck, cv in raw.changelog_stats.items():
                if ck in changelog:
                    changelog[ck]["status_changes"] += cv.get("status_changes", 0)
                    changelog[ck]["reopens"] += cv.get("reopens", 0)
                else:
                    changelog[ck] = dict(cv)
            descriptions.update(raw.descriptions)

        if not all_issues:
            issues_df = pd.DataFrame(
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
        else:
            issues_df = pd.concat(all_issues, ignore_index=True)

        df = self._aggregator.aggregate(
            issues_df,
            comments,
            changelog,
            descriptions,
            show_progress=False,
        )
        df = df.copy()
        df["department"] = "Jira Cloud"

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info("Wrote synced data to {}", save_path)

        return df
