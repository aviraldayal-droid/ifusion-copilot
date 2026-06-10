"""
Per-user / per-IP rate limiting for FastAPI endpoints.

Uses a sliding-window counter held in memory — fine for a single-instance
deployment. For multi-worker or multi-host deployments, swap the in-memory
dict for Redis (or use `slowapi` with a Redis backend).

Usage:
    from app.middleware.rate_limit import rate_limit

    @router.post("/db/chat", dependencies=[Depends(rate_limit("chat"))])
    async def db_chat(...): ...

Limits are configured in RATE_LIMITS below.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Depends, HTTPException, Request, status

from app.auth.deps import get_optional_user

log = logging.getLogger("tbg.ratelimit")


# ---------------------------------------------------------------------------
# Limits (requests, window seconds)
# ---------------------------------------------------------------------------
RATE_LIMITS: dict[str, tuple[int, int]] = {
    # Expensive LLM endpoints — protects against cost runaway / stuck UI loops
    "chat":   (30, 60),     # 30 chat requests per minute per user
    # File upload + parse — heavy CPU
    "upload": (10, 60),     # 10 uploads per minute per user
    # Login — tight to prevent brute force
    "login":  (5, 60),      # 5 attempts per minute per IP
    # Admin endpoints — generous but bounded
    "admin":  (60, 60),     # 60 admin actions per minute per admin
}


# ---------------------------------------------------------------------------
# Sliding-window store
# ---------------------------------------------------------------------------
_buckets: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()


def _check_and_record(key: str, limit: int, window: int) -> tuple[bool, int]:
    """
    Return (allowed, retry_after_seconds).
    Drops timestamps older than `window` seconds, then checks count.
    """
    now = time.monotonic()
    cutoff = now - window
    with _lock:
        bucket = _buckets[key]
        # Drop expired timestamps from the left
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(bucket[0] + window - now))
            return False, retry_after
        bucket.append(now)
        return True, 0


# ---------------------------------------------------------------------------
# Dependency factory
# ---------------------------------------------------------------------------
def rate_limit(bucket_name: str):
    """
    FastAPI dependency that enforces the named rate-limit bucket.
    Identifies the caller by authenticated user_id when available, else by IP.
    """
    if bucket_name not in RATE_LIMITS:
        raise ValueError(f"Unknown rate-limit bucket: {bucket_name}")
    limit, window = RATE_LIMITS[bucket_name]

    async def _dep(
        request: Request,
        optional_user: dict | None = Depends(get_optional_user),
    ) -> None:
        # Identify caller: prefer authenticated user, fall back to client IP
        if optional_user:
            caller = f"user:{optional_user['id']}"
        else:
            caller = f"ip:{request.client.host if request.client else 'unknown'}"

        key = f"{bucket_name}:{caller}"
        allowed, retry_after = _check_and_record(key, limit, window)
        if not allowed:
            log.warning(
                "rate-limit BLOCK: bucket=%s caller=%s limit=%d/%ds retry_after=%ds",
                bucket_name, caller, limit, window, retry_after,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({limit} requests per {window}s). Retry in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )

    return _dep


# ---------------------------------------------------------------------------
# IP-only variant (for login, before any user identity exists)
# ---------------------------------------------------------------------------
def rate_limit_by_ip(bucket_name: str):
    """Like rate_limit() but never looks at the auth header — pure IP-based."""
    if bucket_name not in RATE_LIMITS:
        raise ValueError(f"Unknown rate-limit bucket: {bucket_name}")
    limit, window = RATE_LIMITS[bucket_name]

    async def _dep(request: Request) -> None:
        caller = f"ip:{request.client.host if request.client else 'unknown'}"
        key = f"{bucket_name}:{caller}"
        allowed, retry_after = _check_and_record(key, limit, window)
        if not allowed:
            log.warning(
                "rate-limit BLOCK (ip): bucket=%s caller=%s limit=%d/%ds retry_after=%ds",
                bucket_name, caller, limit, window, retry_after,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({limit} requests per {window}s). Retry in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )

    return _dep
