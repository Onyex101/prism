"""Tests for :class:`src.integrations.jira_oauth.JiraOAuthFlow`."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.integrations.base import AuthError
from src.integrations.jira_oauth import JiraOAuthFlow
from src.integrations.token_store import JiraSite, OAuthToken, SessionTokenStore


def _flow(store: SessionTokenStore, http: MagicMock | None = None) -> JiraOAuthFlow:
    return JiraOAuthFlow(
        client_id="cid",
        client_secret="csec",
        redirect_uri="http://localhost/cb",
        scopes="read:jira-work read:jira-user offline_access",
        store=store,
        http=http,
    )


def test_build_authorize_url_pkce_and_state():
    store = SessionTokenStore({})
    http = MagicMock()
    flow = _flow(store, http)
    url = flow.build_authorize_url()
    assert "https://auth.atlassian.com/authorize" in url
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert "audience=api.atlassian.com" in url
    assert "state=" in url
    pending = store.load_oauth_pending()
    assert pending is not None
    assert len(pending[0]) > 20


def test_exchange_code_state_mismatch():
    store = SessionTokenStore({})
    http = MagicMock()
    flow = _flow(store, http)
    store.save_oauth_pending("v", "good")
    with pytest.raises(AuthError):
        flow.exchange_code("c", "bad")


def test_exchange_code_single_site_auto_selects():
    store = SessionTokenStore({})
    http = MagicMock()
    flow = _flow(store, http)
    store.save_oauth_pending("verifier", "st")

    post_r = MagicMock()
    post_r.status_code = 200
    post_r.json.return_value = {
        "access_token": "at",
        "refresh_token": "rt",
        "expires_in": 3600,
        "scope": "read:jira-work",
    }
    http.post.return_value = post_r

    get_r = MagicMock()
    get_r.status_code = 200
    get_r.json.return_value = [
        {
            "id": "cloud1",
            "name": "N",
            "url": "https://x.atlassian.net",
            "scopes": ["read:jira-work"],
        }
    ]
    http.get.return_value = get_r

    tok = flow.exchange_code("code", "st")
    assert tok.selected_cloud_id == "cloud1"
    assert len(tok.available_sites) == 1
    loaded = store.load()
    assert loaded is not None
    assert loaded.access_token == "at"


def test_exchange_code_multi_site_no_auto_select():
    store = SessionTokenStore({})
    http = MagicMock()
    flow = _flow(store, http)
    store.save_oauth_pending("verifier", "st")

    post_r = MagicMock(status_code=200)
    post_r.json.return_value = {
        "access_token": "at",
        "refresh_token": "rt",
        "expires_in": 3600,
        "scope": "read:jira-work",
    }
    http.post.return_value = post_r

    get_r = MagicMock(status_code=200)
    get_r.json.return_value = [
        {"id": "a", "name": "A", "url": "u1", "scopes": []},
        {"id": "b", "name": "B", "url": "u2", "scopes": []},
    ]
    http.get.return_value = get_r

    tok = flow.exchange_code("code", "st")
    assert tok.selected_cloud_id is None
    assert len(tok.available_sites) == 2


def test_select_site_unknown_raises():
    store = SessionTokenStore({})
    http = MagicMock()
    flow = _flow(store, http)
    store.save(
        OAuthToken(
            "a",
            "r",
            datetime(2030, 1, 1, tzinfo=timezone.utc),
            "s",
            (JiraSite("x", "N", "u", (), None),),
            None,
        )
    )
    with pytest.raises(ValueError):
        flow.select_site("bad")


def test_refresh_preserves_sites():
    store = SessionTokenStore({})
    http = MagicMock()
    flow = _flow(store, http)
    sites = (JiraSite("x", "N", "u", (), None),)
    tok = OAuthToken(
        "old",
        "r",
        datetime(2030, 1, 1, tzinfo=timezone.utc),
        "s",
        sites,
        "x",
    )
    store.save(tok)

    http.post.return_value = MagicMock(
        status_code=200,
    )
    http.post.return_value.json.return_value = {
        "access_token": "new",
        "refresh_token": "r2",
        "expires_in": 3600,
        "scope": "read:jira-work",
    }

    out = flow.refresh(tok)
    assert out.access_token == "new"
    assert out.selected_cloud_id == "x"
    assert out.available_sites == sites
