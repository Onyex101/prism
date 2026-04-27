"""
Atlassian OAuth 2.0 (3LO) authorization code + PKCE flow for Jira Cloud.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from loguru import logger

from src.integrations.base import AuthError
from src.integrations.token_store import JiraSite, OAuthToken, TokenStore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for S256 PKCE."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


class JiraOAuthFlow:
    """
    OAuth 2.0 (3LO) against ``auth.atlassian.com`` with PKCE.

    :param client_id: OAuth app client id.
    :type client_id: str
    :param client_secret: OAuth app client secret.
    :type client_secret: str
    :param redirect_uri: Registered callback URL (must match console exactly).
    :type redirect_uri: str
    :param scopes: Space-separated scopes (e.g. ``read:jira-work``).
    :type scopes: str
    :param store: Token and transient PKCE storage.
    :type store: TokenStore
    :param http: Optional ``requests.Session`` for tests.
    :type http: requests.Session | None

    Example:
        >>> from src.integrations.token_store import SessionTokenStore
        >>> flow = JiraOAuthFlow("id", "sec", "http://localhost:8501/cb", "read:jira-work", SessionTokenStore({}))
    """

    AUTH_URL = "https://auth.atlassian.com/authorize"
    TOKEN_URL = "https://auth.atlassian.com/oauth/token"
    ACCESSIBLE_RESOURCES_URL = "https://api.atlassian.com/oauth/token/accessible-resources"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: str,
        store: TokenStore,
        http: Optional[requests.Session] = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._scopes = scopes
        self._store = store
        self._http = http if http is not None else requests.Session()

    def build_authorize_url(self) -> str:
        """
        Build the user-consent URL and stash PKCE verifier + state.

        :return: Full ``https://auth.atlassian.com/authorize?...`` URL.
        :rtype: str
        """
        verifier, challenge = _pkce_pair()
        state = secrets.token_urlsafe(32)
        self._store.save_oauth_pending(verifier, state)

        params: dict[str, str] = {
            "audience": "api.atlassian.com",
            "client_id": self._client_id,
            "scope": self._scopes,
            "redirect_uri": self._redirect_uri,
            "state": state,
            "response_type": "code",
            "prompt": "consent",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        url = f"{self.AUTH_URL}?{urlencode(params)}"
        logger.info("Built Atlassian OAuth authorize URL (PKCE)")
        return url

    def exchange_code(self, code: str, state: str) -> OAuthToken:
        """
        Exchange the authorization code for tokens and load accessible sites.

        :param code: Authorization code from the redirect query string.
        :type code: str
        :param state: ``state`` query param from the redirect.
        :type state: str
        :return: Persisted :class:`OAuthToken`.
        :rtype: OAuthToken
        :raises AuthError: If ``state`` does not match or token exchange fails.
        """
        pending = self._store.load_oauth_pending()
        if pending is None:
            raise AuthError("Missing OAuth PKCE session; start login again.")
        code_verifier, expected_state = pending
        if state != expected_state:
            self._store.clear_oauth_pending()
            raise AuthError("OAuth state mismatch; start login again.")

        payload: dict[str, Any] = {
            "grant_type": "authorization_code",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "code": code,
            "redirect_uri": self._redirect_uri,
            "code_verifier": code_verifier,
        }
        resp = self._http.post(
            self.TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        self._store.clear_oauth_pending()
        if resp.status_code >= 400:
            logger.warning("Token exchange failed: {} {}", resp.status_code, resp.text[:500])
            raise AuthError(f"Token exchange failed: HTTP {resp.status_code}")

        data = resp.json()
        access_token = str(data["access_token"])
        refresh_token = str(data.get("refresh_token", "") or "")
        expires_in = int(data.get("expires_in", 3600))
        scope = str(data.get("scope", self._scopes))
        expires_at = _utcnow() + timedelta(seconds=expires_in)

        sites = self._fetch_accessible_sites(access_token)
        selected: Optional[str] = None
        if len(sites) == 1:
            selected = sites[0].id
        elif len(sites) == 0:
            logger.warning("accessible-resources returned no Jira sites")

        token = OAuthToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scope=scope,
            available_sites=tuple(sites),
            selected_cloud_id=selected,
        )
        self._store.save(token)
        logger.info("OAuth token stored (sites: {})", len(sites))
        return token

    def _fetch_accessible_sites(self, access_token: str) -> list[JiraSite]:
        resp = self._http.get(
            self.ACCESSIBLE_RESOURCES_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=60,
        )
        if resp.status_code >= 400:
            logger.warning("accessible-resources failed: {} {}", resp.status_code, resp.text[:300])
            raise AuthError(f"Could not list Jira sites: HTTP {resp.status_code}")
        raw_list = resp.json()
        if not isinstance(raw_list, list):
            raise AuthError("Unexpected accessible-resources response shape")
        sites: list[JiraSite] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            cid = item.get("id")
            name = item.get("name", "")
            url = item.get("url", "")
            if not isinstance(cid, str):
                continue
            scopes_raw = item.get("scopes", [])
            if isinstance(scopes_raw, list):
                scopes_t = tuple(str(s) for s in scopes_raw)
            else:
                scopes_t = tuple()
            avatar = item.get("avatarUrl")
            sites.append(
                JiraSite(
                    id=cid,
                    name=str(name),
                    url=str(url),
                    scopes=scopes_t,
                    avatar_url=str(avatar) if avatar else None,
                )
            )
        return sites

    def refresh(self, token: OAuthToken) -> OAuthToken:
        """
        Refresh the access token using the refresh token.

        :param token: Current token (must include ``refresh_token`` if rotated).
        :type token: OAuthToken
        :return: New token with updated credentials and preserved site metadata.
        :rtype: OAuthToken
        :raises AuthError: If refresh fails.
        """
        if not token.refresh_token:
            raise AuthError("No refresh token available; log in again.")
        payload: dict[str, Any] = {
            "grant_type": "refresh_token",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": token.refresh_token,
        }
        resp = self._http.post(
            self.TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if resp.status_code >= 400:
            logger.warning("Token refresh failed: {} {}", resp.status_code, resp.text[:500])
            raise AuthError(f"Token refresh failed: HTTP {resp.status_code}")

        data = resp.json()
        access_token = str(data["access_token"])
        new_refresh = str(data.get("refresh_token", token.refresh_token) or token.refresh_token)
        expires_in = int(data.get("expires_in", 3600))
        scope = str(data.get("scope", token.scope))
        expires_at = _utcnow() + timedelta(seconds=expires_in)

        new_token = OAuthToken(
            access_token=access_token,
            refresh_token=new_refresh,
            expires_at=expires_at,
            scope=scope,
            available_sites=token.available_sites,
            selected_cloud_id=token.selected_cloud_id,
        )
        self._store.save(new_token)
        return new_token

    def select_site(self, cloud_id: str) -> OAuthToken:
        """
        Persist the chosen Jira Cloud site id.

        :param cloud_id: Must match one of :attr:`OAuthToken.available_sites` ``id`` values.
        :type cloud_id: str
        :return: Updated token.
        :rtype: OAuthToken
        :raises ValueError: If ``cloud_id`` is unknown.
        """
        current = self._store.load()
        if current is None:
            raise ValueError("Not logged in")
        valid = {s.id for s in current.available_sites}
        if cloud_id not in valid:
            raise ValueError(f"Unknown cloud id: {cloud_id!r}")
        updated = current.with_selected_site(cloud_id)
        self._store.save(updated)
        return updated

    def clear_selected_site(self) -> OAuthToken:
        """
        Clear ``selected_cloud_id`` so the user can pick another site.

        :return: Updated token.
        :rtype: OAuthToken
        :raises ValueError: If not logged in.
        """
        current = self._store.load()
        if current is None:
            raise ValueError("Not logged in")
        updated = current.with_cleared_site_selection()
        self._store.save(updated)
        return updated

    def logout(self) -> None:
        """Remove token and OAuth pending fields."""
        self._store.clear()
