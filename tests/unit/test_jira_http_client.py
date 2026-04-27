"""Tests for :class:`src.integrations.http.JiraHttpClient`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.integrations.base import NoSiteSelectedError, TokenExpiredError
from src.integrations.http import JiraHttpClient
from src.integrations.token_store import JiraSite, OAuthToken, SessionTokenStore


def _token(selected: str | None) -> OAuthToken:
    sites = (JiraSite("c1", "N", "https://x.atlassian.net", (), None),)
    return OAuthToken(
        access_token="at",
        refresh_token="rt",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        scope="s",
        available_sites=sites,
        selected_cloud_id=selected,
    )


def test_raises_no_site_selected():
    store = SessionTokenStore({})
    store.save(_token(None))
    oauth = MagicMock()
    client = JiraHttpClient(store, oauth, session=MagicMock())
    with pytest.raises(NoSiteSelectedError):
        client.request("GET", "/rest/api/3/myself")


def test_success_returns_response():
    store = SessionTokenStore({})
    store.save(_token("c1"))
    oauth = MagicMock()
    sess = MagicMock()
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"displayName": "Me"}
    sess.request.return_value = ok
    client = JiraHttpClient(store, oauth, session=sess)
    r = client.request("GET", "/rest/api/3/myself")
    assert r.status_code == 200
    assert "ex/jira/c1" in sess.request.call_args[0][1]


def test_401_refreshes_and_retries():
    store = SessionTokenStore({})
    store.save(_token("c1"))
    oauth = MagicMock()

    def _refresh(t: OAuthToken) -> OAuthToken:
        nt = OAuthToken(
            "new_at",
            t.refresh_token,
            t.expires_at,
            t.scope,
            t.available_sites,
            t.selected_cloud_id,
        )
        store.save(nt)
        return nt

    oauth.refresh.side_effect = _refresh

    sess = MagicMock()
    unauthorized = MagicMock(status_code=401)
    ok = MagicMock(status_code=200)
    sess.request.side_effect = [unauthorized, ok]

    client = JiraHttpClient(store, oauth, session=sess)
    r = client.request("GET", "/rest/api/3/myself")
    assert r.status_code == 200
    oauth.refresh.assert_called_once()


def test_401_after_refresh_raises():
    store = SessionTokenStore({})
    store.save(_token("c1"))
    oauth = MagicMock()
    oauth.refresh.return_value = _token("c1")

    sess = MagicMock()
    sess.request.return_value = MagicMock(status_code=401)

    client = JiraHttpClient(store, oauth, session=sess)
    with pytest.raises(TokenExpiredError):
        client.request("GET", "/rest/api/3/myself")
