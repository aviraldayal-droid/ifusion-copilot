"""
JWT utilities for TBG Copilot authentication.

Uses python-jose for JWT and the bcrypt package directly for password hashing.
(passlib is not used — it is unmaintained and incompatible with bcrypt ≥ 4 on Python 3.13)
"""
from __future__ import annotations

import base64
import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

_SECRET: str = os.getenv("JWT_SECRET_KEY", "tbg-copilot-secret-change-in-prod")
_ALGO: str   = "HS256"
_EXPIRE_DAYS: int = 7


def _prepare(password: str) -> bytes:
    # SHA-256 → base64 keeps the input to bcrypt at exactly 44 ASCII bytes,
    # safely below bcrypt's 72-byte limit for any password length.
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_prepare(plain), hashed.encode("utf-8"))


def create_access_token(user_id: int, email: str, session_token: str = "") -> str:
    """Create a signed JWT containing user_id, email and session_token, valid for 7 days."""
    now     = datetime.now(timezone.utc)
    payload = {
        "sub":   str(user_id),
        "email": email,
        "sid":   session_token,
        "iat":   now,
        "exp":   now + timedelta(days=_EXPIRE_DAYS),
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGO)


def decode_token(token: str) -> dict[str, Any] | None:
    """
    Decode and verify a JWT.  Returns the payload dict on success, or None if
    the token is missing, expired, or has an invalid signature.
    """
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGO])
        return payload
    except JWTError:
        return None
