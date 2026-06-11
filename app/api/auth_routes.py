"""
Authentication endpoints for TBG Copilot.

POST /api/v1/auth/register  — admin-only account creation (requires X-Admin-Key header)
POST /api/v1/auth/login     — verify credentials, return JWT
GET  /api/v1/auth/me        — return current user profile (requires auth)
"""
from __future__ import annotations

import asyncio
import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field

log = logging.getLogger("tbg.auth")

from app.auth import keycloak as kc
from app.auth.deps import get_current_user
from app.auth.jwt_utils import create_access_token, hash_password, verify_password
from app.config.settings import settings
from app.db.auth_store import (
    create_user,
    get_user_by_email,
    update_session_token,
    update_user_role,
    upsert_keycloak_user,
)
from app.middleware.rate_limit import rate_limit_by_ip
from app.utils.user_logger import log_user_activity

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str    = Field(..., min_length=3, max_length=320)
    name:  str    = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email:    str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(rate_limit_by_ip("login"))])
async def register(body: RegisterRequest, x_admin_key: str | None = Header(default=None)):
    """Admin-only: create a new account. Requires X-Admin-Key header matching ADMIN_KEY setting."""
    admin_key = settings.ADMIN_KEY.strip()
    if not admin_key or x_admin_key != admin_key:
        log.warning("register: rejected — invalid or missing admin key for email=%s", body.email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration is disabled. Contact the administrator.",
        )

    existing = await asyncio.to_thread(get_user_by_email, body.email)
    if existing:
        log.warning("register: duplicate email=%s", body.email)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    pw_hash = hash_password(body.password)
    user    = await asyncio.to_thread(create_user, body.email, body.name, pw_hash)
    log.info("register: created user id=%s email=%s", user["id"], user["email"])
    log_user_activity(user["id"], user["email"], "viewer", "USER_CREATE", f"account created by admin")

    token = create_access_token(user["id"], user["email"])
    return TokenResponse(
        access_token=token,
        user={"id": user["id"], "email": user["email"], "name": user["name"], "role": user.get("role", "viewer")},
    )


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(rate_limit_by_ip("login"))])
async def login(body: LoginRequest):
    """Verify credentials and return an access token."""
    user = await asyncio.to_thread(get_user_by_email, body.email)

    if not user or not verify_password(body.password, user["password_hash"]):
        log.warning("login: failed for email=%s", body.email)
        if user:
            log_user_activity(user["id"], user["email"], user.get("role", "viewer"), "LOGIN_FAIL", "bad password")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    session_token = secrets.token_hex(32)
    await asyncio.to_thread(update_session_token, user["id"], session_token)
    token = create_access_token(user["id"], user["email"], session_token)
    log.info("login: success user_id=%s email=%s role=%s", user["id"], user["email"], user.get("role", "viewer"))
    log_user_activity(user["id"], user["email"], user.get("role", "viewer"), "LOGIN")
    return TokenResponse(
        access_token=token,
        user={"id": user["id"], "email": user["email"], "name": user["name"], "role": user.get("role", "viewer")},
    )


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    log.debug("me: user_id=%s email=%s", current_user["id"], current_user["email"])
    return {
        "id":    current_user["id"],
        "email": current_user["email"],
        "name":  current_user["name"],
        "role":  current_user.get("role", "viewer"),
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh(current_user: dict = Depends(get_current_user)):
    """
    Silent-refresh endpoint: mint a fresh JWT for an already-authenticated user.

    The frontend calls this when the current token is close to expiry so the
    user's session doesn't die mid-conversation. Works for users from either
    provider (local password OR Keycloak) since the JWT shape is identical.

    NOTE: we deliberately do NOT rotate session_token here. Rotation happens
    on /login and /keycloak/callback (where it intentionally invalidates other
    devices, per current "one active session" behavior). A /refresh just
    extends the same session — otherwise refreshing in one tab would 401
    every other open tab via the sid check in get_current_user.
    """
    sid   = current_user.get("session_token") or ""
    token = create_access_token(current_user["id"], current_user["email"], sid)
    log.debug("refresh: user_id=%s email=%s", current_user["id"], current_user["email"])
    return TokenResponse(
        access_token=token,
        user={
            "id":    current_user["id"],
            "email": current_user["email"],
            "name":  current_user["name"],
            "role":  current_user.get("role", "viewer"),
        },
    )


class SetRoleRequest(BaseModel):
    email: str
    role:  str = Field(..., pattern="^(admin|executive|manager|viewer)$")


@router.post("/admin/set-role")
async def set_role(
    body: SetRoleRequest,
    x_admin_key: str | None = Header(default=None),
):
    """Admin-only: change a user's role. Requires X-Admin-Key matching ADMIN_KEY."""
    admin_key = settings.ADMIN_KEY.strip()
    if not admin_key or x_admin_key != admin_key:
        log.warning("set-role: rejected — invalid admin key for email=%s", body.email)
        raise HTTPException(status_code=403, detail="Admin key required.")
    user = await asyncio.to_thread(get_user_by_email, body.email)
    if not user:
        log.warning("set-role: user not found email=%s", body.email)
        raise HTTPException(status_code=404, detail="User not found.")
    await asyncio.to_thread(update_user_role, user["id"], body.role)
    log.info("set-role: user_id=%s email=%s new_role=%s", user["id"], body.email, body.role)
    return {"status": "ok", "email": body.email, "role": body.role}


# ---------------------------------------------------------------------------
# Keycloak SSO (OIDC auth-code flow with PKCE)
# ---------------------------------------------------------------------------
# Lightweight client-side config so the frontend knows whether to render the
# "Sign in with Keycloak" button. Unauthenticated — safe to expose.
@router.get("/config")
async def auth_config():
    return {"keycloak_enabled": kc.is_enabled()}


_KC_COOKIE = "tbg_kc_flow"
_KC_COOKIE_MAX_AGE = 600  # 10 minutes is plenty for a redirect round-trip


def _set_flow_cookie(resp: RedirectResponse, state: str, nonce: str, verifier: str) -> None:
    # State + nonce + PKCE verifier travel in one httpOnly cookie so they're
    # bound to the user's browser and never reach JS. SameSite=Lax is required
    # because Keycloak redirects back as a top-level GET.
    import base64 as _b64, json as _j
    payload = _b64.urlsafe_b64encode(
        _j.dumps({"s": state, "n": nonce, "v": verifier}).encode()
    ).decode()
    resp.set_cookie(
        _KC_COOKIE, payload,
        max_age=_KC_COOKIE_MAX_AGE,
        httponly=True,
        secure=settings.KEYCLOAK_COOKIE_SECURE,
        samesite="lax",
        path="/api/v1/auth",
    )


def _read_flow_cookie(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        import base64 as _b64, json as _j
        return _j.loads(_b64.urlsafe_b64decode(raw.encode()).decode())
    except Exception:
        return None


def _require_kc_enabled() -> None:
    if not kc.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Keycloak login is not enabled on this server.",
        )


@router.get("/keycloak/login", dependencies=[Depends(rate_limit_by_ip("login"))])
async def keycloak_login():
    """Kick off the OIDC auth-code flow — redirect the browser to Keycloak."""
    _require_kc_enabled()
    state = kc.new_state()
    nonce = kc.new_nonce()
    verifier, challenge = kc.new_pkce_pair()

    try:
        url = await asyncio.to_thread(
            kc.build_authorize_url,
            state=state, nonce=nonce, code_challenge=challenge,
        )
    except Exception as exc:
        log.error("keycloak_login: discovery failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach Keycloak. Check KEYCLOAK_BASE_URL / TLS settings.",
        ) from exc

    resp = RedirectResponse(url=url, status_code=302)
    _set_flow_cookie(resp, state, nonce, verifier)
    return resp


@router.get("/keycloak/callback")
async def keycloak_callback(
    request: Request,
    code:  str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    tbg_kc_flow: str | None = Cookie(default=None),
):
    """
    Handle the redirect back from Keycloak: validate state, exchange the code,
    verify both tokens, upsert the user, mint a local JWT, then bounce the
    browser back to the frontend with ?token=...&kc=1 in the URL.
    """
    _require_kc_enabled()

    if error:
        log.warning("keycloak_callback: provider error=%s desc=%s", error, error_description)
        raise HTTPException(status_code=400, detail=f"Keycloak error: {error}")

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state in callback.")

    flow = _read_flow_cookie(tbg_kc_flow)
    if not flow or flow.get("s") != state:
        log.warning("keycloak_callback: state/cookie mismatch")
        raise HTTPException(status_code=400, detail="Invalid or expired login flow. Try again.")

    # Exchange the auth code for tokens (network call → off the event loop)
    try:
        tokens = await asyncio.to_thread(kc.exchange_code, code, flow["v"])
    except Exception as exc:
        log.error("keycloak_callback: code exchange failed: %s", exc)
        raise HTTPException(status_code=502, detail="Keycloak code exchange failed.") from exc

    id_token = tokens.get("id_token")
    access_token = tokens.get("access_token")
    if not id_token or not access_token:
        raise HTTPException(status_code=502, detail="Keycloak did not return both id_token and access_token.")

    try:
        id_claims     = await asyncio.to_thread(kc.verify_id_token, id_token, expected_nonce=flow.get("n"))
        access_claims = await asyncio.to_thread(kc.verify_access_token, access_token)
    except Exception as exc:
        log.warning("keycloak_callback: token verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Token verification failed.") from exc

    sub   = id_claims.get("sub")
    email = id_claims.get("email")
    name  = id_claims.get("name") or id_claims.get("preferred_username") or (email or "Keycloak user")
    email_verified = bool(id_claims.get("email_verified", False))
    if not sub or not email:
        raise HTTPException(status_code=401, detail="Keycloak token is missing required claims (sub/email).")

    app_role = kc.map_keycloak_role(access_claims)

    # Upsert the user. If the email belongs to a local-password account we
    # require email_verified to prevent account takeover via SSO claim-jumping.
    try:
        user = await asyncio.to_thread(
            upsert_keycloak_user,
            subject=sub, email=email, name=name, role=app_role,
            email_verified=email_verified,
        )
    except PermissionError as exc:
        log.warning("keycloak_callback: blocked unverified link: %s", exc)
        raise HTTPException(
            status_code=403,
            detail=("Your Keycloak email is not verified. Verify it in Keycloak "
                    "before signing in to the dashboard."),
        ) from exc

    # Mirror the local /login JWT-mint sequence exactly so deps.get_current_user
    # (which compares the token's `sid` to copilot_users.session_token) is happy.
    session_token = secrets.token_hex(32)
    await asyncio.to_thread(update_session_token, user["id"], session_token)
    token = create_access_token(user["id"], user["email"], session_token)

    log.info("keycloak_callback: login user_id=%s email=%s role=%s sub=%s",
             user["id"], user["email"], user["role"], sub)
    log_user_activity(user["id"], user["email"], user["role"], "LOGIN", "keycloak")

    # Bounce back to the frontend with the token in the query string. The
    # frontend reads it, stores it in localStorage, and scrubs the URL.
    target = settings.KEYCLOAK_POST_LOGIN_REDIRECT or "/"
    sep = "&" if "?" in target else "?"
    redirect_url = f"{target}{sep}{urlencode({'token': token, 'kc': '1'})}"
    resp = RedirectResponse(url=redirect_url, status_code=302)
    resp.delete_cookie(_KC_COOKIE, path="/api/v1/auth")
    return resp

