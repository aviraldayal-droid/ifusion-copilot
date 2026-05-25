"""
FastAPI dependency functions for authentication.

Usage:
    current_user = Depends(get_current_user)   # raises 401 if not authenticated
    opt_user     = Depends(get_optional_user)  # returns None if not authenticated
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.auth.jwt_utils import decode_token
from app.db.auth_store import get_user_by_id

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False,
)


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict:
    """
    Resolve the bearer token to a user dict.
    Raises HTTP 401 if the token is absent, expired, or invalid.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    import asyncio
    user = await asyncio.to_thread(get_user_by_id, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Reject if another session has logged in since this token was issued
    sid = payload.get("sid", "")
    if sid and user.get("session_token") != sid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_optional_user(token: str | None = Depends(oauth2_scheme)) -> dict | None:
    """
    Like get_current_user but returns None instead of raising on auth failure.
    Use this for endpoints that work in both guest and authenticated modes.
    """
    if not token:
        return None

    payload = decode_token(token)
    if payload is None:
        return None

    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError):
        return None

    import asyncio
    return await asyncio.to_thread(get_user_by_id, user_id)
