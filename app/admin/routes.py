"""
Admin API routes — served on port 8001 only.
All endpoints require a valid JWT with role == 'admin'.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

log = logging.getLogger("tbg.admin")
from pydantic import BaseModel, Field

from app.auth.deps import get_current_user, require_admin_user
from app.auth.jwt_utils import hash_password
from app.middleware.rate_limit import rate_limit, rate_limit_by_ip
from app.db.auth_store import (
    create_user,
    delete_user,
    get_audit_log,
    get_user_by_email,
    get_user_by_id,
    list_users,
    update_user_role,
)
from app.utils.user_logger import log_user_activity

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Auth passthrough — admin app needs its own login endpoint
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email:    str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


@router.post("/login", dependencies=[Depends(rate_limit_by_ip("login"))])
async def admin_login(body: LoginRequest):
    """Verify credentials and confirm admin role."""
    import secrets
    from app.auth.jwt_utils import create_access_token, verify_password
    from app.db.auth_store import update_session_token

    user = await asyncio.to_thread(get_user_by_email, body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        log.warning("admin_login: failed for email=%s", body.email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    if user.get("role") != "admin":
        log.warning("admin_login: non-admin access attempt email=%s role=%s", body.email, user.get("role"))
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")

    session_token = secrets.token_hex(32)
    await asyncio.to_thread(update_session_token, user["id"], session_token)
    token = create_access_token(user["id"], user["email"], session_token)
    log.info("admin_login: success user_id=%s email=%s", user["id"], user["email"])
    log_user_activity(user["id"], user["email"], "admin", "LOGIN", "admin dashboard")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "name": user["name"], "role": user["role"]},
    }


@router.get("/me")
async def admin_me(current_user: dict = Depends(require_admin_user)):
    return {"id": current_user["id"], "email": current_user["email"],
            "name": current_user["name"], "role": current_user["role"]}


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

class CreateUserRequest(BaseModel):
    email:    str = Field(..., min_length=3, max_length=320)
    name:     str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=6, max_length=128)
    role:     str = Field(default="viewer", pattern="^(admin|executive|manager|viewer)$")


class UpdateUserRequest(BaseModel):
    role: str = Field(..., pattern="^(admin|executive|manager|viewer)$")


@router.get("/users")
async def list_users_endpoint(admin: dict = Depends(require_admin_user)):
    users = await asyncio.to_thread(list_users)
    log.debug("list_users: admin_id=%s count=%d", admin["id"], len(users))
    return {"users": users}


@router.post("/users", status_code=status.HTTP_201_CREATED, dependencies=[Depends(rate_limit("admin"))])
async def create_user_endpoint(body: CreateUserRequest, admin: dict = Depends(require_admin_user)):
    existing = await asyncio.to_thread(get_user_by_email, body.email)
    if existing:
        log.warning("create_user: duplicate email=%s admin_id=%s", body.email, admin["id"])
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    pw_hash = hash_password(body.password)
    user = await asyncio.to_thread(create_user, body.email, body.name, pw_hash)
    await asyncio.to_thread(update_user_role, user["id"], body.role)
    log.info("create_user: id=%s email=%s role=%s by admin_id=%s", user["id"], body.email, body.role, admin["id"])
    log_user_activity(user["id"], body.email, body.role, "USER_CREATE", f"created by admin_id={admin['id']}")
    return {"status": "created", "user": {**user, "role": body.role}}


@router.patch("/users/{user_id}", dependencies=[Depends(rate_limit("admin"))])
async def update_user_endpoint(
    user_id: int, body: UpdateUserRequest, admin: dict = Depends(require_admin_user)
):
    if admin["id"] == user_id and body.role != "admin":
        log.warning("update_user: self-demotion attempt admin_id=%s", admin["id"])
        raise HTTPException(status_code=400, detail="Cannot demote yourself.")
    user = await asyncio.to_thread(get_user_by_id, user_id)
    if not user:
        log.warning("update_user: user not found user_id=%s admin_id=%s", user_id, admin["id"])
        raise HTTPException(status_code=404, detail="User not found.")
    await asyncio.to_thread(update_user_role, user_id, body.role)
    log.info("update_user: user_id=%s new_role=%s by admin_id=%s", user_id, body.role, admin["id"])
    log_user_activity(user_id, user.get("email", ""), body.role, "ROLE_CHANGE",
                      f"new_role={body.role} | changed by admin_id={admin['id']}")
    return {"status": "ok", "user_id": user_id, "role": body.role}


@router.delete("/users/{user_id}", dependencies=[Depends(rate_limit("admin"))])
async def delete_user_endpoint(user_id: int, admin: dict = Depends(require_admin_user)):
    if admin["id"] == user_id:
        log.warning("delete_user: self-delete attempt admin_id=%s", admin["id"])
        raise HTTPException(status_code=400, detail="Cannot delete yourself.")
    deleted = await asyncio.to_thread(delete_user, user_id)
    if not deleted:
        log.warning("delete_user: not found user_id=%s admin_id=%s", user_id, admin["id"])
        raise HTTPException(status_code=404, detail="User not found.")
    log.info("delete_user: user_id=%s by admin_id=%s", user_id, admin["id"])
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@router.get("/audit")
async def audit_log_endpoint(limit: int = 100, admin: dict = Depends(require_admin_user)):
    logs = await asyncio.to_thread(get_audit_log, min(limit, 200))
    log.debug("audit_log: admin_id=%s returned %d entries", admin["id"], len(logs))
    return {"logs": logs}
