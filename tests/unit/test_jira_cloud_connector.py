"""Tests for :class:`src.integrations.jira_cloud.JiraCloudConnector` with a fake HTTP client."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pandas as pd

from src.integrations.jira_cloud import JiraCloudConnector


class FakeHttp:
    """Minimal fake matching :class:`src.integrations.http.JiraHttpClient` ``request`` API."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Any = None,
        timeout: float = 60.0,
    ) -> Any:
        self.calls.append((method, path))
        if not self._responses:
            raise RuntimeError("No more fake responses")
        return self._responses.pop(0)


def _ok_json(data: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = data
    r.text = ""
    return r


def test_test_connection():
    fake = FakeHttp([_ok_json({"displayName": "Alice", "emailAddress": "a@x.com"})])
    c = JiraCloudConnector(fake)  # type: ignore[arg-type]
    out = c.test_connection()
    assert out["display_name"] == "Alice"


def test_list_projects_pagination():
    page1 = _ok_json(
        {
            "values": [{"key": "A", "name": "Alpha", "id": "1"}],
            "total": 2,
        }
    )
    page2 = _ok_json(
        {
            "values": [{"key": "B", "name": "Beta", "id": "2"}],
            "total": 2,
        }
    )
    fake = FakeHttp([page1, page2])
    c = JiraCloudConnector(fake)  # type: ignore[arg-type]
    ps = c.list_projects()
    assert [p.key for p in ps] == ["A", "B"]


def test_fetch_project_data_maps_dataframe():
    search_body = {
        "issues": [
            {
                "key": "DEMO-1",
                "changelog": {"histories": []},
                "fields": {
                    "project": {"key": "DEMO", "name": "Demo"},
                    "status": {"name": "Open"},
                    "priority": {"name": "Medium"},
                    "issuetype": {"name": "Task"},
                    "resolution": None,
                    "created": "2024-01-01T00:00:00.000+0000",
                    "updated": "2024-01-02T00:00:00.000+0000",
                    "resolutiondate": None,
                    "assignee": {"displayName": "Bob"},
                    "creator": {"displayName": "Bob"},
                    "reporter": {"displayName": "Bob"},
                    "votes": {"total": 0},
                    "watches": {"watchCount": 1},
                    "description": None,
                },
            }
        ],
        "total": 1,
    }
    comment_resp = _ok_json({"comments": []})
    fake = FakeHttp(
        [
            _ok_json(search_body),
            comment_resp,
        ]
    )
    c = JiraCloudConnector(fake)  # type: ignore[arg-type]
    raw = c.fetch_project_data("DEMO", max_issues=10)
    assert raw.project_key == "DEMO"
    assert isinstance(raw.issues, pd.DataFrame)
    assert len(raw.issues) == 1
    assert raw.issues.iloc[0]["key"] == "DEMO-1"


def test_progress_callback_invoked():
    search_body = {
        "issues": [],
        "total": 0,
    }
    fake = FakeHttp([_ok_json(search_body)])
    calls: list[tuple[str, int, int]] = []

    def prog(phase: str, cur: int, tot: int) -> None:
        calls.append((phase, cur, tot))

    c = JiraCloudConnector(fake)  # type: ignore[arg-type]
    c.fetch_project_data("X", max_issues=10, progress=prog)
    assert any(t[0] == "search" for t in calls)
