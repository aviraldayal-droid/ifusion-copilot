"""
SQLite-backed CRUD layer for copilot authentication and per-user conversation history.

Data is stored in a local `copilot.db` file at the project root.
No changes are made to the PostgreSQL/Digiwise database.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("tbg.auth_store")

# DB file sits at the repo root (next to requirements.txt)
_DB_PATH = Path(__file__).resolve().parents[2] / "copilot.db"

_init_lock = threading.Lock()
_initialized = False

_DDL = """
CREATE TABLE IF NOT EXISTS copilot_users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    UNIQUE NOT NULL,
    name          TEXT    NOT NULL,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS copilot_conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES copilot_users(id),
    title      TEXT    NOT NULL DEFAULT 'New Chat',
    mode       TEXT    NOT NULL DEFAULT 'db',
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON copilot_conversations(user_id);

CREATE TABLE IF NOT EXISTS copilot_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES copilot_conversations(id),
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    metadata        TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON copilot_messages(conversation_id);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_schema() -> None:
    global _initialized
    with _init_lock:
        if _initialized:
            return
        conn = _connect()
        try:
            conn.executescript(_DDL)
            conn.commit()
            _initialized = True
            log.info("SQLite copilot schema ensured at %s", _DB_PATH)
        finally:
            conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(r: sqlite3.Row | None) -> dict | None:
    return dict(r) if r else None


def _rows(rs) -> list[dict]:
    return [dict(r) for r in rs]


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(email: str, name: str, password_hash: str) -> dict:
    ensure_schema()
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO copilot_users (email, name, password_hash) VALUES (?, ?, ?)",
            (email.lower().strip(), name.strip(), password_hash),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, email, name, created_at FROM copilot_users WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict | None:
    ensure_schema()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, email, name, password_hash, created_at FROM copilot_users WHERE email = ?",
            (email.lower().strip(),),
        ).fetchone()
        return _row(row)
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    ensure_schema()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, email, name, created_at FROM copilot_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return _row(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(user_id: int, title: str = "New Chat", mode: str = "db") -> dict:
    ensure_schema()
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO copilot_conversations (user_id, title, mode) VALUES (?, ?, ?)",
            (user_id, title[:500], mode),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, user_id, title, mode, created_at, updated_at FROM copilot_conversations WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_conversations(user_id: int) -> list[dict]:
    ensure_schema()
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT c.id, c.user_id, c.title, c.mode, c.created_at, c.updated_at,
                      COUNT(m.id) AS msg_count
               FROM copilot_conversations c
               LEFT JOIN copilot_messages m ON m.conversation_id = c.id
               WHERE c.user_id = ?
               GROUP BY c.id
               ORDER BY c.updated_at DESC
               LIMIT 100""",
            (user_id,),
        ).fetchall()
        return _rows(rows)
    finally:
        conn.close()


def get_conversation(conv_id: int) -> dict | None:
    ensure_schema()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, user_id, title, mode, created_at, updated_at FROM copilot_conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        return _row(row)
    finally:
        conn.close()


def update_conversation_title(conv_id: int, title: str) -> None:
    ensure_schema()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE copilot_conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title[:500], _now(), conv_id),
        )
        conn.commit()
    finally:
        conn.close()


def touch_conversation(conv_id: int) -> None:
    ensure_schema()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE copilot_conversations SET updated_at = ? WHERE id = ?",
            (_now(), conv_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conv_id: int, user_id: int) -> bool:
    ensure_schema()
    conn = _connect()
    try:
        cur = conn.execute(
            "DELETE FROM copilot_conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def save_message(conv_id: int, role: str, content: str, metadata: dict | None = None) -> int:
    ensure_schema()
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO copilot_messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, json.dumps(metadata or {})),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_messages(conv_id: int) -> list[dict]:
    ensure_schema()
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT id, conversation_id, role, content, metadata, created_at
               FROM copilot_messages
               WHERE conversation_id = ?
               ORDER BY created_at ASC, id ASC""",
            (conv_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result
    finally:
        conn.close()
