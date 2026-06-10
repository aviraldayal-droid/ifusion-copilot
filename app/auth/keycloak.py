"""
Keycloak OIDC client for the TBG Copilot dashboard.

Implements the bits the auth-code (PKCE) login flow needs:

  - lazy-loaded OIDC discovery doc + JWKS, cached in-process
  - authorize-URL builder (state + nonce + PKCE S256)
  - token-endpoint code exchange
  - id_token signature/claims verification (identity)
  - access_token signature verification + role extraction (RBAC)
  - Keycloak-role → app-role mapping driven by KEYCLOAK_ROLE_MAP

Notes (per advisor 2026-06-09):
  - By default Keycloak puts realm_access.roles on the access_token, NOT the
    id_token. We verify both and read roles from the access_token's claims.
  - The access_token's `aud` is often "account", not our client_id. We turn
    off audience verification for the access_token and instead require `azp`
    to equal our client_id.
  - The token issuer is read verbatim from the discovery doc — never
    hand-constructed — because Keycloak's frontendUrl may rewrite it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import threading
from urllib.parse import urlencode

import httpx
from jose import jwt

from app.config.settings import settings

log = logging.getLogger("tbg.auth.keycloak")


# ---------------------------------------------------------------------------
# Discovery + JWKS cache
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_discovery: dict | None = None
_jwks: dict | None = None
_role_map_cache: tuple[str, dict[str, str]] | None = None


def _client() -> httpx.Client:
    """httpx client honoring KEYCLOAK_TLS_VERIFY for self-signed dev certs."""
    return httpx.Client(verify=settings.KEYCLOAK_TLS_VERIFY, timeout=10.0)


def _discovery_url() -> str:
    base = settings.KEYCLOAK_BASE_URL.rstrip("/")
    realm = settings.KEYCLOAK_REALM
    if not base or not realm:
        raise RuntimeError(
            "Keycloak is enabled but KEYCLOAK_BASE_URL / KEYCLOAK_REALM are not configured."
        )
    return f"{base}/realms/{realm}/.well-known/openid-configuration"


def get_discovery() -> dict:
    """Return the OIDC discovery document, fetched once and cached."""
    global _discovery
    if _discovery is not None:
        return _discovery
    with _lock:
        if _discovery is not None:
            return _discovery
        url = _discovery_url()
        log.info("keycloak: fetching discovery doc from %s", url)
        with _client() as c:
            r = c.get(url)
            r.raise_for_status()
            _discovery = r.json()
    return _discovery


def get_jwks() -> dict:
    """Return the realm JWKS, fetched once and cached."""
    global _jwks
    if _jwks is not None:
        return _jwks
    with _lock:
        if _jwks is not None:
            return _jwks
        jwks_uri = get_discovery()["jwks_uri"]
        log.info("keycloak: fetching JWKS from %s", jwks_uri)
        with _client() as c:
            r = c.get(jwks_uri)
            r.raise_for_status()
            _jwks = r.json()
    return _jwks


def _refresh_jwks() -> dict:
    """Force a JWKS reload — used when a `kid` lookup misses (key rotation)."""
    global _jwks
    with _lock:
        _jwks = None
    return get_jwks()


def _jwk_for_kid(kid: str) -> dict:
    jwks = get_jwks()
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return k
    jwks = _refresh_jwks()
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return k
    raise ValueError(f"No JWK matches kid={kid!r} in realm JWKS")


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------
def new_pkce_pair() -> tuple[str, str]:
    """Generate a (verifier, S256-challenge) pair per RFC 7636."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def new_state() -> str:
    return secrets.token_urlsafe(32)


def new_nonce() -> str:
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Authorize URL + token exchange
# ---------------------------------------------------------------------------
def build_authorize_url(*, state: str, nonce: str, code_challenge: str) -> str:
    """Assemble the redirect URL the browser navigates to for login."""
    disc = get_discovery()
    params = {
        "response_type":         "code",
        "client_id":             settings.KEYCLOAK_CLIENT_ID,
        "redirect_uri":          settings.KEYCLOAK_REDIRECT_URI,
        "scope":                 "openid email profile",
        "state":                 state,
        "nonce":                 nonce,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{disc['authorization_endpoint']}?{urlencode(params)}"


def exchange_code(code: str, code_verifier: str) -> dict:
    """Exchange an auth code for tokens. Returns the raw /token response body."""
    disc = get_discovery()
    data = {
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  settings.KEYCLOAK_REDIRECT_URI,
        "client_id":     settings.KEYCLOAK_CLIENT_ID,
        "client_secret": settings.KEYCLOAK_CLIENT_SECRET,
        "code_verifier": code_verifier,
    }
    with _client() as c:
        r = c.post(disc["token_endpoint"], data=data,
                   headers={"Accept": "application/json"})
        if r.status_code != 200:
            log.warning("keycloak token exchange failed: status=%s body=%s",
                        r.status_code, r.text[:300])
            r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------
def verify_id_token(token: str, *, expected_nonce: str | None = None) -> dict:
    """Verify the id_token's signature and standard claims; return its claims."""
    disc = get_discovery()
    header = jwt.get_unverified_header(token)
    jwk    = _jwk_for_kid(header["kid"])
    claims = jwt.decode(
        token, jwk,
        algorithms=[header.get("alg", "RS256")],
        audience=settings.KEYCLOAK_CLIENT_ID,
        issuer=disc["issuer"],
    )
    if expected_nonce is not None and claims.get("nonce") != expected_nonce:
        raise ValueError("id_token nonce mismatch")
    return claims


def verify_access_token(token: str) -> dict:
    """
    Verify the access_token's signature and issuer; return its claims.

    Audience is intentionally NOT checked (Keycloak access tokens commonly have
    aud="account"). We instead require azp to equal our client_id.
    """
    disc = get_discovery()
    header = jwt.get_unverified_header(token)
    jwk    = _jwk_for_kid(header["kid"])
    claims = jwt.decode(
        token, jwk,
        algorithms=[header.get("alg", "RS256")],
        issuer=disc["issuer"],
        options={"verify_aud": False},
    )
    azp = claims.get("azp")
    if azp != settings.KEYCLOAK_CLIENT_ID:
        raise ValueError(f"access_token azp={azp!r} does not match client_id")
    return claims


# ---------------------------------------------------------------------------
# Role mapping
# ---------------------------------------------------------------------------
def _load_role_map() -> dict[str, str]:
    """Parse KEYCLOAK_ROLE_MAP (JSON) once and cache."""
    global _role_map_cache
    raw = (settings.KEYCLOAK_ROLE_MAP or "").strip()
    if _role_map_cache is not None and _role_map_cache[0] == raw:
        return _role_map_cache[1]
    mapping: dict[str, str] = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                mapping = {str(k): str(v) for k, v in parsed.items()}
        except Exception as exc:
            log.warning("KEYCLOAK_ROLE_MAP is not valid JSON: %s", exc)
    _role_map_cache = (raw, mapping)
    return mapping


def _collect_token_roles(access_claims: dict) -> list[str]:
    """Pull roles from realm_access and from any resource_access client roles."""
    roles: list[str] = []
    realm = access_claims.get("realm_access") or {}
    if isinstance(realm.get("roles"), list):
        roles.extend(str(r) for r in realm["roles"])
    resource = access_claims.get("resource_access") or {}
    if isinstance(resource, dict):
        for client_block in resource.values():
            if isinstance(client_block, dict) and isinstance(client_block.get("roles"), list):
                roles.extend(str(r) for r in client_block["roles"])
    return roles


def map_keycloak_role(access_claims: dict) -> str:
    """
    Reduce the Keycloak roles on an access token to a single app role.

    Strategy: walk KEYCLOAK_ROLE_MAP in declaration order; the first
    Keycloak role present in the token wins. Falls back to KEYCLOAK_DEFAULT_ROLE.
    """
    mapping = _load_role_map()
    token_roles = set(_collect_token_roles(access_claims))
    for kc_role, app_role in mapping.items():
        if kc_role in token_roles:
            return app_role
    return settings.KEYCLOAK_DEFAULT_ROLE


def is_enabled() -> bool:
    """Single source of truth for whether Keycloak routes should be live."""
    return bool(
        settings.KEYCLOAK_ENABLED
        and settings.KEYCLOAK_BASE_URL
        and settings.KEYCLOAK_REALM
        and settings.KEYCLOAK_CLIENT_ID
    )
