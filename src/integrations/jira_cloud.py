"""
Jira Cloud REST API v3 connector (OAuth-backed).
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from urllib.parse import quote as url_quote

import pandas as pd
import requests
from loguru import logger

from src.integrations.base import AuthError, ProjectManagementConnector, ProjectSummary, RawProjectData
from src.integrations.http import JiraHttpClient
from src.data.jira_aggregator import JiraAggregator


def _display_name(user: Any) -> str:
    if user is None:
        return ""
    if isinstance(user, dict):
        return str(user.get("displayName") or user.get("emailAddress") or "")
    return str(user)


def _adf_to_plain_text(node: Any) -> str:
    """Extract plain text from Atlassian Document Format (best-effort)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if node.get("type") == "text" and "text" in node:
            return str(node["text"])
        parts: list[str] = []
        for c in node.get("content") or []:
            t = _adf_to_plain_text(c)
            if t:
                parts.append(t)
        return "\n".join(parts)
    if isinstance(node, list):
        return "\n".join(_adf_to_plain_text(x) for x in node)
    return ""


class JiraCloudConnector(ProjectManagementConnector):
    """
    Fetch issues and comments from Jira Cloud using :class:`JiraHttpClient`.

    :param http: Authenticated HTTP client (injected).
    :type http: JiraHttpClient

    Example:
        >>> JiraCloudConnector(http_client)
    """

    PAGE_SIZE = 100

    def __init__(self, http: JiraHttpClient) -> None:
        self._http = http

    def test_connection(self) -> dict[str, str]:
        """
        Call ``GET /rest/api/3/myself``.

        :return: ``display_name`` and ``email`` keys when present.
        :rtype: dict[str, str]
        :raises AuthError: On HTTP failure.
        """
        resp = self._http.request("GET", "/rest/api/3/myself")
        if resp.status_code >= 400:
            raise AuthError(f"myself failed: HTTP {resp.status_code}")
        data = resp.json()
        return {
            "display_name": str(data.get("displayName", "")),
            "email": str(data.get("emailAddress", "")),
        }

    def list_projects(self) -> list[ProjectSummary]:
        """
        Paginate ``GET /rest/api/3/project/search``.

        :return: Project summaries.
        :rtype: list[ProjectSummary]
        """
        out: list[ProjectSummary] = []
        start = 0
        while True:
            resp = self._http.request(
                "GET",
                "/rest/api/3/project/search",
                params={"startAt": start, "maxResults": 50},
            )
            if resp.status_code >= 400:
                raise AuthError(f"project/search failed: HTTP {resp.status_code}")
            data = resp.json()
            values = data.get("values") or []
            for p in values:
                if not isinstance(p, dict):
                    continue
                key = p.get("key")
                name = p.get("name", "")
                pid = p.get("id")
                if isinstance(key, str):
                    out.append(ProjectSummary(key=key, name=str(name), id=str(pid) if pid else None))
            total = int(data.get("total", len(out)))
            start += len(values)
            if start >= total or not values:
                break
        return out

    def fetch_project_data(
        self,
        project_key: str,
        max_issues: int,
        jql_extra: str = "",
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> RawProjectData:
        """
        Search issues with changelog; fetch sampled comments for LLM fields.

        :param project_key: Jira project key.
        :type project_key: str
        :param max_issues: Cap total issues pulled from search.
        :type max_issues: int
        :param jql_extra: Extra ``AND`` clauses for JQL.
        :type jql_extra: str
        :param progress: Optional progress callback.
        :type progress: collections.abc.Callable[[str, int, int], None] | None
        :return: Data for :class:`~src.data.jira_aggregator.JiraAggregator`.
        :rtype: RawProjectData
        """
        jql = f"project = {project_key} ORDER BY created ASC"
        if jql_extra.strip():
            jql = f"{jql} AND ({jql_extra.strip()})"

        issues_rows: list[dict[str, Any]] = []
        changelog_by_project: dict[str, dict[str, int]] = {project_key: {"status_changes": 0, "reopens": 0}}
        descriptions: list[str] = []
        fetched = 0
        start_at = 0

        while fetched < max_issues:
            page_size = min(self.PAGE_SIZE, max_issues - fetched)
            body: dict[str, Any] = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": page_size,
                "fields": [
                    "summary",
                    "description",
                    "status",
                    "priority",
                    "issuetype",
                    "created",
                    "updated",
                    "resolutiondate",
                    "assignee",
                    "creator",
                    "reporter",
                    "votes",
                    "watches",
                    "project",
                    "resolution",
                ],
                "expand": ["changelog"],
            }
            resp = self._http.request("POST", "/rest/api/3/search", json=body)
            if resp.status_code >= 400:
                raise AuthError(f"search failed: HTTP {resp.status_code} {resp.text[:300]}")
            data = resp.json()
            issues = data.get("issues") or []
            total = int(data.get("total", 0))
            if progress:
                progress("search", min(start_at + len(issues), total), max(total, 1))

            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                row = self._issue_to_row(issue)
                issues_rows.append(row)
                self._accumulate_changelog(issue, changelog_by_project)
                desc = row.get("_description_adf")
                if isinstance(desc, dict) or isinstance(desc, list):
                    t = _adf_to_plain_text(desc)
                    if t:
                        descriptions.append(t)
                elif isinstance(desc, str) and desc:
                    descriptions.append(desc)
                fetched += 1
                if fetched >= max_issues:
                    break

            start_at += len(issues)
            if not issues or start_at >= total:
                break

        issues_df = pd.DataFrame(issues_rows)
        if len(issues_df) == 0:
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
            issues_df = issues_df.drop(columns=[c for c in issues_df.columns if c.startswith("_")], errors="ignore")

        comments_list = self._fetch_comments_for_project(
            project_key,
            list(issues_rows),
            JiraAggregator.MAX_COMMENTS_PER_PROJECT,
            progress,
        )
        comments_by_project = {project_key: comments_list}

        desc_limit = JiraAggregator.MAX_DESCRIPTIONS_PER_PROJECT
        descriptions_by_project = {project_key: descriptions[:desc_limit]}

        return RawProjectData(
            project_key=project_key,
            issues=issues_df,
            comments=comments_by_project,
            changelog_stats=changelog_by_project,
            descriptions=descriptions_by_project,
        )

    def _issue_to_row(self, issue: dict[str, Any]) -> dict[str, Any]:
        fields = issue.get("fields") or {}
        project = fields.get("project") or {}
        status = fields.get("status") or {}
        priority = fields.get("priority") or {}
        issuetype = fields.get("issuetype") or {}
        resolution = fields.get("resolution") or {}
        votes = fields.get("votes") or {}
        watches = fields.get("watches") or {}
        return {
            "key": issue.get("key", ""),
            "resolution.name": resolution.get("name"),
            "priority.name": (priority or {}).get("name"),
            "status.name": (status or {}).get("name"),
            "issuetype.name": (issuetype or {}).get("name"),
            "project.key": project.get("key"),
            "project.name": project.get("name"),
            "created": fields.get("created"),
            "updated": fields.get("updated"),
            "resolutiondate": fields.get("resolutiondate"),
            "assignee": _display_name(fields.get("assignee")),
            "creator": _display_name(fields.get("creator")),
            "reporter": _display_name(fields.get("reporter")),
            "votes.votes": int(votes.get("total", 0) or 0),
            "watches.watchCount": int(watches.get("watchCount", 0) or 0),
            "_description_adf": fields.get("description"),
        }

    def _accumulate_changelog(self, issue: dict[str, Any], stats: dict[str, dict[str, int]]) -> None:
        changelog = issue.get("changelog") or {}
        histories = changelog.get("histories") or []
        pkey = None
        fields = issue.get("fields") or {}
        proj = fields.get("project") or {}
        pkey = str(proj.get("key") or "")
        if not pkey:
            return
        bucket = stats.setdefault(pkey, {"status_changes": 0, "reopens": 0})
        for h in histories:
            if not isinstance(h, dict):
                continue
            items = h.get("items") or []
            for it in items:
                if not isinstance(it, dict):
                    continue
                if it.get("field") != "status":
                    continue
                bucket["status_changes"] += 1
                to_s = it.get("toString") or ""
                if to_s == "Reopened":
                    bucket["reopens"] += 1

    def _fetch_comments_for_project(
        self,
        project_key: str,
        issue_rows: list[dict[str, Any]],
        limit: int,
        progress: Optional[Callable[[str, int, int], None]],
    ) -> list[str]:
        bodies: list[str] = []
        for idx, row in enumerate(issue_rows):
            if len(bodies) >= limit:
                break
            key = row.get("key")
            if not isinstance(key, str) or not key.startswith(project_key):
                continue
            resp = self._http.request(
                "GET",
                f"/rest/api/3/issue/{url_quote(key, safe='')}/comment",
                params={"maxResults": 50, "startAt": 0},
            )
            if resp.status_code >= 400:
                logger.warning("comments for {}: HTTP {}", key, resp.status_code)
                continue
            data = resp.json()
            comments = data.get("comments") or []
            for c in comments:
                if len(bodies) >= limit:
                    break
                if not isinstance(c, dict):
                    continue
                body = c.get("body")
                text = _adf_to_plain_text(body) if body is not None else ""
                if text:
                    bodies.append(text)
            if progress:
                progress("comments", idx + 1, len(issue_rows))
        return bodies
