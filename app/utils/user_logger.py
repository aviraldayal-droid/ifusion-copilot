"""
Per-user activity logger.

Each user gets a dedicated log file at logs/users/<user_id>_<email>.log.
Entries are plain text, one line per action — easy to open and read.

Usage:
    from app.utils.user_logger import log_user_activity
    log_user_activity(user_id=1, email="aviral@ksolves.com", role="admin",
                      action="DB_CHAT", detail="Which months had the biggest gap vs budget?")
"""
from __future__ import annotations

import logging
import logging.handlers
import re
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "users"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_user_loggers: dict[int, logging.Logger] = {}


def _safe_filename(email: str) -> str:
    """Strip characters not safe for filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", email)


def _get_logger(user_id: int, email: str) -> logging.Logger:
    """Return (and cache) a RotatingFileHandler logger for this user."""
    if user_id in _user_loggers:
        return _user_loggers[user_id]

    name = f"tbg.user.{user_id}"
    logger = logging.getLogger(name)
    logger.propagate = False          # don't bubble up to root/app.log
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        log_path = _LOG_DIR / f"{user_id}_{_safe_filename(email)}.log"
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,   # 5 MB per user file
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)

    _user_loggers[user_id] = logger
    return logger


def log_user_activity(
    user_id: int | None,
    email: str,
    role: str,
    action: str,
    detail: str = "",
) -> None:
    """
    Write one line to the user's personal log file.

    Actions (convention):
        LOGIN           successful sign-in
        LOGIN_FAIL      bad credentials
        FILE_UPLOAD     TBG Excel file uploaded and parsed
        FILE_CHAT       question sent against uploaded file
        DB_CHAT         question sent against PostgreSQL
        DB_SCHEMA       tables selected by RAG for the query
        DB_SQL          final SQL generated and executed
        DB_RESULT       rows returned by the query
        LLM_THINKING    reasoning block extracted from LLM response
        RBAC_BLOCK      question blocked by policy
        ROLE_CHANGE     user's role was changed by admin
        USER_CREATE     new user account created
        USER_DELETE     user account deleted
    """
    if user_id is None:
        return

    logger = _get_logger(user_id, email)
    msg = f"[{action:<14}] role={role:<10} | {detail}" if detail else f"[{action:<14}] role={role}"
    logger.info(msg)


def log_pipeline_event(action: str, detail: str = "") -> None:
    """
    Write a pipeline event to the current request's user log file.
    Reads user identity from the request context variables — safe to call
    from anywhere in the pipeline (graph.py, tools.py, etc.) without
    passing user objects around.
    """
    from app.config.settings import request_user_id, request_user_email, request_user_role
    uid   = request_user_id.get()
    email = request_user_email.get()
    role  = request_user_role.get()
    log_user_activity(uid, email, role, action, detail)
