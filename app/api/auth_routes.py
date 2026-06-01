"""
Authentication endpoints for TBG Copilot.

POST /api/v1/auth/register  — admin-only account creation (requires X-Admin-Key header)
POST /api/v1/auth/login     — verify credentials, return JWT
GET  /api/v1/auth/me        — return current user profile (requires auth)
"""
from __future__ import annotations

import asyncio

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.auth.deps import get_current_user
from app.auth.jwt_utils import create_access_token, hash_password, verify_password
from app.config.settings import settings
from app.db.auth_store import create_user, get_user_by_email, update_session_token

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

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, x_admin_key: str | None = Header(default=None)):
    """Admin-only: create a new account. Requires X-Admin-Key header matching ADMIN_KEY setting."""
    admin_key = settings.ADMIN_KEY.strip()
    if not admin_key or x_admin_key != admin_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration is disabled. Contact the administrator.",
        )

    existing = await asyncio.to_thread(get_user_by_email, body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    pw_hash = hash_password(body.password)
    user    = await asyncio.to_thread(create_user, body.email, body.name, pw_hash)

    token = create_access_token(user["id"], user["email"])
    return TokenResponse(
        access_token=token,
        user={"id": user["id"], "email": user["email"], "name": user["name"], "role": user.get("role", "viewer")},
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Verify credentials and return an access token."""
    user = await asyncio.to_thread(get_user_by_email, body.email)

    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    session_token = secrets.token_hex(32)
    await asyncio.to_thread(update_session_token, user["id"], session_token)
    token = create_access_token(user["id"], user["email"], session_token)
    return TokenResponse(
        access_token=token,
        user={"id": user["id"], "email": user["email"], "name": user["name"], "role": user.get("role", "viewer")},
    )


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return {
        "id":    current_user["id"],
        "email": current_user["email"],
        "name":  current_user["name"],
        "role":  current_user.get("role", "viewer"),
    }


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
        raise HTTPException(status_code=403, detail="Admin key required.")
    from app.db.auth_store import update_user_role
    user = await asyncio.to_thread(get_user_by_email, body.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    await asyncio.to_thread(update_user_role, user["id"], body.role)
    return {"status": "ok", "email": body.email, "role": body.role}
