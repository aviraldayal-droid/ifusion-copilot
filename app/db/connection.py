"""
Thread-safe psycopg2 connection pool for the TBG database.

All queries run inside the `tbg` schema automatically.
"""
from __future__ import annotations

import logging
import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from app.config.settings import settings

log = logging.getLogger("tbg.db")

_pool: pg_pool.ThreadedConnectionPool | None = None


def get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        dsn = settings.DATABASE_URL
        if 'sslmode' not in dsn:
            sep = '&' if '?' in dsn else '?'
            dsn += f'{sep}sslmode=disable'
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=dsn,
            options="-c search_path=public,digiwise_schema",
        )
    return _pool


def explain(sql: str) -> None:
    """
    Run EXPLAIN on a SELECT statement to validate it without executing it.
    Raises an exception if the SQL is syntactically or semantically invalid.
    """
    # Ensure SQL is properly trimmed and doesn't have stray characters
    sql = sql.strip()
    if not sql:
        raise ValueError("SQL statement is empty")
    
    # If SQL ends with semicolon, remove it (EXPLAIN doesn't expect it)
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    
    p = get_pool()
    conn = p.getconn()
    try:
        explain_cmd = f"EXPLAIN {sql}"
        log.debug("Running EXPLAIN: %s", explain_cmd[:200])
        with conn.cursor() as cur:
            cur.execute(explain_cmd)
        log.debug("EXPLAIN validation passed")
    except psycopg2.Error as exc:
        # Log the exact SQL that failed for debugging
        log.error("EXPLAIN failed on SQL: %s", sql[:500])
        log.error("Error details: %s", str(exc))
        raise
    finally:
        p.putconn(conn)


def execute(sql: str, params: tuple = ()) -> tuple[list[dict], list[str]]:
    """
    Run a single SQL statement and return (rows_as_dicts, column_names).
    Always operates inside the tbg schema.
    Only SELECT / WITH statements are allowed.
    """
    normalised = sql.strip().upper().lstrip("(")
    if not (normalised.startswith("SELECT") or normalised.startswith("WITH")):
        raise ValueError("Only SELECT queries are permitted.")

    p = get_pool()
    conn = p.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or None)
            cols = [d.name for d in cur.description] if cur.description else []
            rows = [dict(r) for r in cur.fetchall()]
        return rows, cols
    finally:
        p.putconn(conn)


def execute_on(sql: str, dbname: str, params: tuple = ()) -> tuple[list[dict], list[str]]:
    """
    Run a SELECT against any database on the same server (for snapshot queries).
    Creates a short-lived connection — not pooled.
    """
    normalised = sql.strip().upper().lstrip("(")
    if not (normalised.startswith("SELECT") or normalised.startswith("WITH")):
        raise ValueError("Only SELECT queries are permitted.")

    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=int(settings.DB_PORT),
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        dbname=dbname,
        options="-c search_path=public",
        sslmode="disable",
    )
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or None)
            cols = [d.name for d in cur.description] if cur.description else []
            rows = [dict(r) for r in cur.fetchall()]
        return rows, cols
    finally:
        conn.close()


def explain_on(sql: str, dbname: str) -> None:
    """Run EXPLAIN against a snapshot database (for validate_semantic)."""
    sql = sql.strip().rstrip(";")
    if not sql:
        raise ValueError("SQL statement is empty")
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=int(settings.DB_PORT),
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        dbname=dbname,
        sslmode="disable",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN {sql}")
    finally:
        conn.close()


def ping() -> bool:
    """Return True if the DB is reachable."""
    try:
        execute("SELECT 1 AS ok")
        return True
    except Exception:
        return False
