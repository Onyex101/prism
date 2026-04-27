"""
Abstract project-management connector and shared domain types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


class ConnectorError(Exception):
    """Base class for integration failures."""


class AuthError(ConnectorError):
    """Authentication or authorization failed."""


class TokenExpiredError(AuthError):
    """Access token could not be refreshed or is invalid."""


class NoSiteSelectedError(ConnectorError):
    """OAuth token has no ``selected_cloud_id``; user must pick a Jira site."""


class RateLimitError(ConnectorError):
    """Remote API rate limit exceeded."""


class ConnectionFailure(ConnectorError):
    """Network or server error after retries."""


@dataclass(frozen=True)
class ProjectSummary:
    """
    Minimal metadata for a selectable project in the UI.

    :param key: Project key (e.g. ``PROJ``).
    :type key: str
    :param name: Human-readable project name.
    :type name: str
    :param id: Optional Jira project id.
    :type id: str | None

    Example:
        >>> ProjectSummary(key="DEMO", name="Demo", id="10000")
    """

    key: str
    name: str
    id: Optional[str] = None


@dataclass(frozen=True)
class RawProjectData:
    """
    Issue-level inputs for :class:`~src.data.jira_aggregator.JiraAggregator`.

    :param project_key: Jira project key this batch belongs to.
    :type project_key: str
    :param issues: Issue metric rows (Apache / API column names).
    :type issues: pandas.DataFrame
    :param comments: Comment bodies keyed by project key.
    :type comments: dict[str, list[str]]
    :param changelog_stats: Status-change / reopen counters per project key.
    :type changelog_stats: dict[str, dict[str, int]]
    :param descriptions: Sampled issue descriptions per project key.
    :type descriptions: dict[str, list[str]]
    """

    project_key: str
    issues: pd.DataFrame
    comments: dict[str, list[str]]
    changelog_stats: dict[str, dict[str, int]]
    descriptions: dict[str, list[str]]


class ProjectManagementConnector(ABC):
    """
    Contract for ingesting project data from a PM tool into PRISM shapes.

    Implementations must not depend on Streamlit; progress is reported via an
    optional callback.
    """

    @abstractmethod
    def test_connection(self) -> dict[str, str]:
        """
        Verify credentials and return a small identity payload.

        :return: Keys such as ``display_name``, ``email``.
        :rtype: dict[str, str]
        :raises AuthError: If the session is not authenticated.
        """

    @abstractmethod
    def list_projects(self) -> list[ProjectSummary]:
        """
        List projects visible to the current user.

        :return: Non-empty or empty list of projects.
        :rtype: list[ProjectSummary]
        """

    @abstractmethod
    def fetch_project_data(
        self,
        project_key: str,
        max_issues: int,
        jql_extra: str = "",
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> RawProjectData:
        """
        Fetch issues, comments, and changelog-derived stats for one project.

        :param project_key: Jira project key.
        :type project_key: str
        :param max_issues: Upper bound on issues to retrieve (pagination cap).
        :type max_issues: int
        :param jql_extra: Additional JQL AND clauses (e.g. date filter).
        :type jql_extra: str
        :param progress: Optional ``(message, current, total)`` callback.
        :type progress: collections.abc.Callable[[str, int, int], None] | None
        :return: Structured data for aggregation.
        :rtype: RawProjectData
        """
