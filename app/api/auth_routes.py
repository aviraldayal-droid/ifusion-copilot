"""
Authentication endpoints for TBG Copilot.

POST /api/v1/auth/register  — create account, return JWT
POST /api/v1/auth/login     — verify credentials, return JWT
GET  /api/v1/auth/me        — return current user profile (requires auth)
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.auth.deps import get_current_user
from app.auth.jwt_utils import create_access_token, hash_password, verify_password
from app.db.auth_store import create_user, get_user_by_email

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
async def register(body: RegisterRequest):
    """Create a new account and return an access token."""
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
        user={"id": user["id"], "email": user["email"], "name": user["name"]},
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

    token = create_access_token(user["id"], user["email"])
    return TokenResponse(
        access_token=token,
        user={"id": user["id"], "email": user["email"], "name": user["name"]},
    )


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return {
        "id":    current_user["id"],
        "email": current_user["email"],
        "name":  current_user["name"],
    }
