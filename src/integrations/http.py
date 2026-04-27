"""
Authenticated HTTP client for Atlassian Cloud Jira REST (via OAuth bearer token).
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests
from loguru import logger

from src.integrations.base import AuthError, ConnectionFailure, NoSiteSelectedError, TokenExpiredError
from src.integrations.jira_oauth import JiraOAuthFlow
from src.integrations.token_store import OAuthToken, TokenStore


class JiraHttpClient:
    """
    ``requests``-based client: bearer auth, ``/ex/jira/{cloudId}`` base URL, retries.

    :param store: OAuth token storage.
    :type store: TokenStore
    :param oauth: OAuth flow used to refresh tokens on ``401``.
    :type oauth: JiraOAuthFlow
    :param session: Optional session for tests.
    :type session: requests.Session | None

    Example:
        >>> JiraHttpClient(store, oauth)
    """

    BASE_HOST = "https://api.atlassian.com"

    def __init__(
        self,
        store: TokenStore,
        oauth: JiraOAuthFlow,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._store = store
        self._oauth = oauth
        self._session = session if session is not None else requests.Session()

    def _load_token(self) -> OAuthToken:
        token = self._store.load()
        if token is None:
            raise AuthError("Not authenticated")
        return token

    def _ensure_fresh(self, token: OAuthToken) -> OAuthToken:
        """Refresh if expiring within 60 seconds."""
        now = datetime.now(timezone.utc)
        if token.expires_at <= now + timedelta(seconds=60):
            if not token.refresh_token:
                raise TokenExpiredError("Access token expired and no refresh token")
            return self._oauth.refresh(token)
        return token

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Any = None,
        timeout: float = 60.0,
    ) -> requests.Response:
        """
        Perform an HTTP request against Jira REST under the selected cloud id.

        :param method: HTTP method (``GET``, ``POST``, ...).
        :type method: str
        :param path: Path beginning with ``/rest/...``.
        :type path: str
        :param params: Optional query parameters.
        :type params: dict[str, typing.Any] | None
        :param json: Optional JSON body (for ``POST``).
        :type json: typing.Any | None
        :param timeout: Request timeout in seconds.
        :type timeout: float
        :return: ``requests.Response`` (may have ``status_code >= 400``).
        :rtype: requests.Response
        :raises NoSiteSelectedError: If no site is selected.
        :raises AuthError: If not logged in.
        :raises TokenExpiredError: If refresh fails after ``401``.
        :raises ConnectionFailure: On repeated network or ``5xx`` failures.
        """
        token = self._load_token()
        if token.selected_cloud_id is None:
            raise NoSiteSelectedError("Select a Jira Cloud site before calling the API")
        token = self._ensure_fresh(token)

        url = f"{self.BASE_HOST}/ex/jira/{token.selected_cloud_id}{path}"
        headers = {"Authorization": f"Bearer {token.access_token}", "Accept": "application/json"}

        resp = self._request_with_retries(method, url, headers, params=params, json=json, timeout=timeout)
        if resp.status_code == 401:
            try:
                self._oauth.refresh(self._load_token())
                tok = self._ensure_fresh(self._load_token())
                headers["Authorization"] = f"Bearer {tok.access_token}"
                resp = self._request_with_retries(
                    method, url, headers, params=params, json=json, timeout=timeout
                )
            except AuthError as exc:
                raise TokenExpiredError("Session expired; log in again") from exc
            if resp.status_code == 401:
                raise TokenExpiredError("Session expired; log in again")
        return resp

    def _request_with_retries(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        *,
        params: Optional[dict[str, Any]],
        json: Any,
        timeout: float,
    ) -> requests.Response:
        max_attempts = 5
        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt < max_attempts:
            attempt += 1
            try:
                resp = self._session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                self._backoff_sleep(attempt)
                continue

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after and retry_after.isdigit() else min(
                    2.0**attempt, 60.0
                )
                logger.warning("Rate limited (429), sleeping {:.1f}s", wait_s)
                time.sleep(wait_s)
                continue

            if 500 <= resp.status_code < 600:
                self._backoff_sleep(attempt)
                continue

            return resp

        if last_exc:
            raise ConnectionFailure(str(last_exc)) from last_exc
        raise ConnectionFailure("HTTP request failed after retries")

    @staticmethod
    def _backoff_sleep(attempt: int) -> None:
        base = min(2.0**attempt, 30.0)
        jitter = random.uniform(0, base * 0.2)
        time.sleep(base + jitter)
