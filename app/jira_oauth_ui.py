"""
Streamlit helpers: build Jira OAuth + sync stack from :class:`config.settings.Settings`.
"""

from __future__ import annotations

from typing import Any, MutableMapping

from config.settings import Settings
from src.data.jira_aggregator import JiraAggregator
from src.integrations.http import JiraHttpClient
from src.integrations.jira_cloud import JiraCloudConnector
from src.integrations.jira_oauth import JiraOAuthFlow
from src.integrations.jira_sync_service import JiraSyncService
from src.integrations.token_store import SessionTokenStore


def build_jira_stack(
    settings: Settings,
    session: MutableMapping[str, Any],
) -> tuple[JiraOAuthFlow, SessionTokenStore, JiraSyncService]:
    """
    Wire OAuth flow, token store, HTTP client, connector, and sync service.

    :param settings: Loaded application settings (must have OAuth fields set).
    :type settings: config.settings.Settings
    :param session: Typically ``st.session_state``.
    :type session: collections.abc.MutableMapping[str, typing.Any]
    :return: ``(flow, store, sync_service)``.
    :rtype: tuple[JiraOAuthFlow, SessionTokenStore, JiraSyncService]

    Example:
        >>> build_jira_stack(settings, st.session_state)
    """
    store = SessionTokenStore(session)
    flow = JiraOAuthFlow(
        client_id=settings.JIRA_OAUTH_CLIENT_ID or "",
        client_secret=settings.JIRA_OAUTH_CLIENT_SECRET or "",
        redirect_uri=settings.JIRA_OAUTH_REDIRECT_URI or "",
        scopes=settings.JIRA_OAUTH_SCOPES,
        store=store,
    )
    http = JiraHttpClient(store, flow)
    connector = JiraCloudConnector(http)
    sync = JiraSyncService(connector, JiraAggregator())
    return flow, store, sync


def query_param_first(params: Any, key: str) -> str | None:
    """
    Return a single query param value (Streamlit may return a list).

    :param params: ``st.query_params`` or similar mapping.
    :type params: typing.Any
    :param key: Parameter name.
    :type key: str
    :return: First value or ``None``.
    :rtype: str | None
    """
    if key not in params:
        return None
    v = params[key]
    if isinstance(v, list) and v:
        return str(v[0])
    if isinstance(v, str):
        return v
    return str(v) if v is not None else None
