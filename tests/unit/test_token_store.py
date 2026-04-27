"""Tests for :mod:`src.integrations.token_store`."""

from __future__ import annotations

from datetime import datetime, timezone

from src.integrations.token_store import (
    JiraSite,
    OAuthToken,
    SessionTokenStore,
    TOKEN_KEY,
)


def test_session_token_store_round_trip():
    """Save and load OAuthToken via an in-memory dict."""
    session: dict = {}
    store = SessionTokenStore(session)
    assert store.load() is None

    tok = OAuthToken(
        access_token="a",
        refresh_token="r",
        expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        scope="read:jira-work",
        available_sites=(
            JiraSite("c1", "Site", "https://x.atlassian.net", ("read:jira-work",), None),
        ),
        selected_cloud_id="c1",
    )
    store.save(tok)
    assert TOKEN_KEY in session
    loaded = store.load()
    assert loaded is not None
    assert loaded.access_token == "a"
    assert loaded.selected_cloud_id == "c1"

    store.clear()
    assert store.load() is None


def test_oauth_token_with_selected_site():
    """with_selected_site updates cloud id."""
    sites = (JiraSite("x", "A", "u", (), None), JiraSite("y", "B", "u2", (), None))
    t = OAuthToken(
        "a",
        "r",
        datetime(2030, 1, 1, tzinfo=timezone.utc),
        "s",
        sites,
        None,
    )
    u = t.with_selected_site("y")
    assert u.selected_cloud_id == "y"
    assert t.selected_cloud_id is None
