"""
DB CRUD layer for copilot authentication and per-user conversation history.

Tables live in the public schema (same DB as Digiwise).
All functions call ensure_schema() lazily so the tables are created on first use.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2
import psycopg2.extras

from app.db.connection import get_pool

log = logging.getLogger("tbg.auth_store")

_SCHEMA_CREATED = False


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS public.copilot_users (
    id            SERIAL PRIMARY KEY,
    email         VARCHAR(320) UNIQUE NOT NULL,
    name          VARCHAR(200) NOT NULL,
    password_hash VARCHAR(200) NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.copilot_conversations (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES public.copilot_users(id) ON DELETE CASCADE,
    title      VARCHAR(500) NOT NULL DEFAULT 'New Chat',
    mode       VARCHAR(20)  NOT NULL DEFAULT 'db',
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_copilot_conv_user ON public.copilot_conversations(user_id);

CREATE TABLE IF NOT EXISTS public.copilot_messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES public.copilot_conversations(id) ON DELETE CASCADE,
    role            VARCHAR(10)  NOT NULL,
    content         TEXT        NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_copilot_msg_conv ON public.copilot_messages(conversation_id);
"""


def ensure_schema() -> None:
    """Create auth/history tables if they don't exist yet. Idempotent."""
    global _SCHEMA_CREATED
    if _SCHEMA_CREATED:
        return
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        _SCHEMA_CREATED = True
        log.info("copilot auth schema ensured")
    except Exception as exc:
        conn.rollback()
        log.error("ensure_schema failed: %s", exc)
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _conn_cursor(pool):
    conn = pool.getconn()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    return conn, cur


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(email: str, name: str, password_hash: str) -> dict:
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO public.copilot_users (email, name, password_hash)
                   VALUES (%s, %s, %s)
                   RETURNING id, email, name, created_at""",
                (email.lower().strip(), name.strip(), password_hash),
            )
            row = dict(cur.fetchone())
        conn.commit()
        return row
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def get_user_by_email(email: str) -> dict | None:
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email, name, password_hash, created_at FROM public.copilot_users WHERE email = %s",
                (email.lower().strip(),),
            )
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        pool.putconn(conn)


def get_user_by_id(user_id: int) -> dict | None:
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email, name, created_at FROM public.copilot_users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(user_id: int, title: str = "New Chat", mode: str = "db") -> dict:
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO public.copilot_conversations (user_id, title, mode)
                   VALUES (%s, %s, %s)
                   RETURNING id, user_id, title, mode, created_at, updated_at""",
                (user_id, title[:500], mode),
            )
            row = dict(cur.fetchone())
        conn.commit()
        return row
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def list_conversations(user_id: int) -> list[dict]:
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT c.id, c.user_id, c.title, c.mode, c.created_at, c.updated_at,
                          COUNT(m.id)::int AS msg_count
                   FROM public.copilot_conversations c
                   LEFT JOIN public.copilot_messages m ON m.conversation_id = c.id
                   WHERE c.user_id = %s
                   GROUP BY c.id
                   ORDER BY c.updated_at DESC
                   LIMIT 100""",
                (user_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]
        return rows
    finally:
        pool.putconn(conn)


def get_conversation(conv_id: int) -> dict | None:
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, user_id, title, mode, created_at, updated_at FROM public.copilot_conversations WHERE id = %s",
                (conv_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        pool.putconn(conn)


def update_conversation_title(conv_id: int, title: str) -> None:
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.copilot_conversations SET title = %s, updated_at = NOW() WHERE id = %s",
                (title[:500], conv_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def touch_conversation(conv_id: int) -> None:
    """Set updated_at = NOW() to push it to the top of the list."""
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.copilot_conversations SET updated_at = NOW() WHERE id = %s",
                (conv_id,),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def delete_conversation(conv_id: int, user_id: int) -> bool:
    """Delete a conversation owned by user_id. Returns True if a row was deleted."""
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.copilot_conversations WHERE id = %s AND user_id = %s",
                (conv_id, user_id),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def save_message(conv_id: int, role: str, content: str, metadata: dict | None = None) -> int:
    """Insert a message and return its id."""
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    meta_json = json.dumps(metadata or {})
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO public.copilot_messages (conversation_id, role, content, metadata)
                   VALUES (%s, %s, %s, %s)
                   RETURNING id""",
                (conv_id, role, content, meta_json),
            )
            msg_id = cur.fetchone()[0]
        conn.commit()
        return msg_id
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def list_messages(conv_id: int) -> list[dict]:
    ensure_schema()
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT id, conversation_id, role, content, metadata, created_at
                   FROM public.copilot_messages
                   WHERE conversation_id = %s
                   ORDER BY created_at ASC, id ASC""",
                (conv_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]
        return rows
    finally:
        pool.putconn(conn)
