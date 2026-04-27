"""
OAuth token persistence behind a small protocol (session-backed for Streamlit).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, MutableMapping, Optional

# Keys used inside the injected mapping (e.g. st.session_state)
TOKEN_KEY = "_prism_jira_oauth_token"
PENDING_VERIFIER_KEY = "_prism_oauth_code_verifier"
PENDING_STATE_KEY = "_prism_oauth_expected_state"


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class JiraSite:
    """
    One Atlassian Cloud site from ``accessible-resources``.

    :param id: Cloud id (``cloudid``) used in Jira REST URLs.
    :type id: str
    :param name: Site name.
    :type name: str
    :param url: Site base URL.
    :type url: str
    :param scopes: Granted OAuth scopes for this resource.
    :type scopes: tuple[str, ...]
    :param avatar_url: Optional avatar URL.
    :type avatar_url: str | None

    Example:
        >>> JiraSite("abc", "Acme", "https://acme.atlassian.net", ("read:jira-work",), None)
    """

    __slots__ = ("id", "name", "url", "scopes", "avatar_url")

    def __init__(
        self,
        id: str,
        name: str,
        url: str,
        scopes: tuple[str, ...],
        avatar_url: Optional[str],
    ) -> None:
        self.id = id
        self.name = name
        self.url = url
        self.scopes = scopes
        self.avatar_url = avatar_url

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JiraSite):
            return NotImplemented
        return (
            self.id == other.id
            and self.name == other.name
            and self.url == other.url
            and self.scopes == other.scopes
            and self.avatar_url == other.avatar_url
        )

    def __hash__(self) -> int:
        return hash((self.id, self.name, self.url, self.scopes, self.avatar_url))


class OAuthToken:
    """
    OAuth 2.0 tokens plus multi-site metadata for Jira Cloud.

    :param access_token: Bearer access token.
    :type access_token: str
    :param refresh_token: Refresh token (may be empty if not returned).
    :type refresh_token: str
    :param expires_at: Absolute expiry of the access token (UTC).
    :type expires_at: datetime
    :param scope: Space-separated scope string from the token response.
    :type scope: str
    :param available_sites: Sites returned by ``accessible-resources``.
    :type available_sites: tuple[JiraSite, ...]
    :param selected_cloud_id: Chosen site id, or ``None`` if multi-site and not chosen.
    :type selected_cloud_id: str | None

    Example:
        >>> OAuthToken("a", "r", _utcnow(), "read:jira-work", tuple(), "x")
    """

    __slots__ = (
        "access_token",
        "refresh_token",
        "expires_at",
        "scope",
        "available_sites",
        "selected_cloud_id",
    )

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        expires_at: datetime,
        scope: str,
        available_sites: tuple[JiraSite, ...],
        selected_cloud_id: Optional[str],
    ) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
        self.scope = scope
        self.available_sites = available_sites
        self.selected_cloud_id = selected_cloud_id

    def with_selected_site(self, cloud_id: str) -> OAuthToken:
        """
        Return a copy with ``selected_cloud_id`` set.

        :param cloud_id: Must exist in :attr:`available_sites`.
        :type cloud_id: str
        :return: New token instance.
        :rtype: OAuthToken
        """
        return OAuthToken(
            self.access_token,
            self.refresh_token,
            self.expires_at,
            self.scope,
            self.available_sites,
            cloud_id,
        )

    def with_cleared_site_selection(self) -> OAuthToken:
        """
        Return a copy with ``selected_cloud_id`` cleared (for \"Switch site\").

        :return: New token instance.
        :rtype: OAuthToken
        """
        return OAuthToken(
            self.access_token,
            self.refresh_token,
            self.expires_at,
            self.scope,
            self.available_sites,
            None,
        )


class TokenStore(ABC):
    """
    Persistence for :class:`OAuthToken` and transient OAuth handshake fields.
    """

    @abstractmethod
    def load(self) -> Optional[OAuthToken]:
        """Load the current token, or ``None`` if logged out."""

    @abstractmethod
    def save(self, token: OAuthToken) -> None:
        """Persist ``token`` (replaces any previous token)."""

    @abstractmethod
    def clear(self) -> None:
        """Remove token and OAuth pending fields."""

    @abstractmethod
    def save_oauth_pending(self, code_verifier: str, state: str) -> None:
        """Store PKCE verifier and expected state for the authorize round-trip."""

    @abstractmethod
    def load_oauth_pending(self) -> Optional[tuple[str, str]]:
        """Return ``(code_verifier, expected_state)`` or ``None``."""

    @abstractmethod
    def clear_oauth_pending(self) -> None:
        """Clear PKCE/state after a successful or failed exchange."""


class SessionTokenStore(TokenStore):
    """
    Store tokens in an injected mutable mapping (e.g. ``st.session_state``).

    :param session: Backing mapping.
    :type session: collections.abc.MutableMapping[str, Any]

    Example:
        >>> store = SessionTokenStore({})
        >>> store.save(OAuthToken("a", "r", _utcnow(), "", tuple(), None))
    """

    def __init__(self, session: MutableMapping[str, Any]) -> None:
        self._session = session

    def load(self) -> Optional[OAuthToken]:
        raw = self._session.get(TOKEN_KEY)
        if raw is None:
            return None
        if isinstance(raw, OAuthToken):
            return raw
        raise TypeError(f"Unexpected token payload type: {type(raw)!r}")

    def save(self, token: OAuthToken) -> None:
        self._session[TOKEN_KEY] = token

    def clear(self) -> None:
        self._session.pop(TOKEN_KEY, None)
        self.clear_oauth_pending()

    def save_oauth_pending(self, code_verifier: str, state: str) -> None:
        self._session[PENDING_VERIFIER_KEY] = code_verifier
        self._session[PENDING_STATE_KEY] = state

    def load_oauth_pending(self) -> Optional[tuple[str, str]]:
        v = self._session.get(PENDING_VERIFIER_KEY)
        s = self._session.get(PENDING_STATE_KEY)
        if v is None or s is None:
            return None
        if not isinstance(v, str) or not isinstance(s, str):
            return None
        return (v, s)

    def clear_oauth_pending(self) -> None:
        self._session.pop(PENDING_VERIFIER_KEY, None)
        self._session.pop(PENDING_STATE_KEY, None)
