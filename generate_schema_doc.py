"""
Standalone script: connects to the Digiwise PostgreSQL database and writes
a full schema reference as schema_doc.md.

Usage:
    python generate_schema_doc.py
    python generate_schema_doc.py --schema digiwise_schema
    python generate_schema_doc.py --schema public
"""
from __future__ import annotations

import argparse
import sys
from urllib.parse import quote_plus

import psycopg2
import psycopg2.extras

# ── Connection ──────────────────────────────────────────────────────────────
DB_USER     = "digiwise_rw"
DB_PASSWORD = "Digi@3456rw$"
DB_HOST     = "197.230.47.51"
DB_PORT     = "5432"
DB_NAME     = "digiwise"

DSN = (
    f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"
)


def connect():
    return psycopg2.connect(DSN, options="-c search_path=public,digiwise_schema")


def q(conn, sql: str, params=()) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params or None)
        return [dict(r) for r in cur.fetchall()]


# ── Introspection ────────────────────────────────────────────────────────────

def get_schemas(conn) -> list[str]:
    rows = q(conn, """
        SELECT schema_name
        FROM   information_schema.schemata
        WHERE  schema_name NOT IN ('information_schema','pg_catalog','pg_toast',
                                   'pg_temp_1','pg_toast_temp_1')
        ORDER  BY schema_name
    """)
    return [r["schema_name"] for r in rows]


def get_tables(conn, schema: str) -> list[str]:
    rows = q(conn, """
        SELECT table_name
        FROM   information_schema.tables
        WHERE  table_schema = %s AND table_type = 'BASE TABLE'
        ORDER  BY table_name
    """, (schema,))
    return [r["table_name"] for r in rows]


def get_views(conn, schema: str) -> list[str]:
    rows = q(conn, """
        SELECT table_name
        FROM   information_schema.views
        WHERE  table_schema = %s
        ORDER  BY table_name
    """, (schema,))
    return [r["table_name"] for r in rows]


def get_columns(conn, schema: str, table: str) -> list[dict]:
    return q(conn, """
        SELECT c.column_name,
               c.data_type,
               c.character_maximum_length,
               c.numeric_precision,
               c.is_nullable,
               c.column_default,
               CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END AS is_pk
        FROM   information_schema.columns c
        LEFT JOIN (
            SELECT ku.column_name
            FROM   information_schema.table_constraints tc
            JOIN   information_schema.key_column_usage   ku
                   ON  ku.constraint_name = tc.constraint_name
                   AND ku.table_schema    = tc.table_schema
                   AND ku.table_name      = tc.table_name
            WHERE  tc.constraint_type = 'PRIMARY KEY'
              AND  tc.table_schema    = %s
              AND  tc.table_name      = %s
        ) pk ON pk.column_name = c.column_name
        WHERE  c.table_schema = %s AND c.table_name = %s
        ORDER  BY c.ordinal_position
    """, (schema, table, schema, table))


def get_fks(conn, schema: str, table: str) -> list[dict]:
    return q(conn, """
        SELECT kcu.column_name,
               ccu.table_name  AS ref_table,
               ccu.column_name AS ref_column
        FROM   information_schema.table_constraints        tc
        JOIN   information_schema.key_column_usage         kcu
               ON  kcu.constraint_name = tc.constraint_name
               AND kcu.table_schema    = tc.table_schema
        JOIN   information_schema.constraint_column_usage  ccu
               ON  ccu.constraint_name = tc.constraint_name
               AND ccu.table_schema    = tc.table_schema
        WHERE  tc.constraint_type = 'FOREIGN KEY'
          AND  tc.table_schema    = %s
          AND  tc.table_name      = %s
        ORDER  BY kcu.column_name
    """, (schema, table))


def get_indexes(conn, schema: str, table: str) -> list[dict]:
    return q(conn, """
        SELECT i.relname AS index_name,
               ix.indisunique AS is_unique,
               array_to_string(
                   ARRAY(SELECT a.attname
                         FROM   pg_attribute a
                         WHERE  a.attrelid = t.oid
                           AND  a.attnum = ANY(ix.indkey)
                         ORDER  BY array_position(ix.indkey, a.attnum)),
                   ', '
               ) AS columns
        FROM   pg_class t
        JOIN   pg_namespace n  ON n.oid = t.relnamespace
        JOIN   pg_index ix     ON ix.indrelid = t.oid
        JOIN   pg_class i      ON i.oid = ix.indexrelid
        WHERE  n.nspname = %s
          AND  t.relname = %s
          AND  NOT ix.indisprimary
        ORDER  BY i.relname
    """, (schema, table))


def get_row_estimates(conn, schema: str) -> dict[str, int]:
    rows = q(conn, """
        SELECT c.relname AS table_name, c.reltuples::bigint AS estimate
        FROM   pg_class c
        JOIN   pg_namespace n ON n.oid = c.relnamespace
        WHERE  n.nspname = %s AND c.relkind = 'r'
    """, (schema,))
    return {r["table_name"]: int(r["estimate"]) for r in rows if int(r["estimate"]) >= 0}


def _type_str(col: dict) -> str:
    t = col["data_type"]
    if col.get("character_maximum_length"):
        return f"{t}({col['character_maximum_length']})"
    if t in ("numeric", "decimal") and col.get("numeric_precision"):
        return f"{t}({col['numeric_precision']})"
    return t


# ── Markdown builder ─────────────────────────────────────────────────────────

def build_markdown(conn, schema: str) -> str:
    tables     = get_tables(conn, schema)
    views      = get_views(conn, schema)
    row_counts = get_row_estimates(conn, schema)

    lines: list[str] = []
    lines += [
        f"# Database Schema — `{DB_NAME}` / schema `{schema}`",
        "",
        f"> Generated from `{DB_HOST}:{DB_PORT}`.  "
        f"**{len(tables)} tables**, **{len(views)} views**.",
        "",
    ]

    # ── Table of Contents ────────────────────────────────────────────────
    lines += ["## Table of Contents", ""]
    lines.append("### Tables")
    for t in tables:
        anchor = t.lower().replace("_", "-")
        count_hint = f" (~{row_counts[t]:,} rows)" if row_counts.get(t) else ""
        lines.append(f"- [{t}](#{anchor}){count_hint}")
    lines.append("")

    if views:
        lines.append("### Views")
        for v in views:
            lines.append(f"- {v}")
        lines.append("")

    lines += ["---", ""]

    # ── Per-table sections ───────────────────────────────────────────────
    all_fks: list[tuple[str, str, str, str]] = []   # (table, col, ref_table, ref_col)

    for table in tables:
        anchor = table.lower().replace("_", "-")
        count  = row_counts.get(table)
        count_str = f"  _(~{count:,} rows)_" if count else ""

        lines += [f"## `{table}`{count_str}", ""]

        cols = get_columns(conn, schema, table)
        fks  = get_fks(conn, schema, table)
        idxs = get_indexes(conn, schema, table)

        fk_map = {fk["column_name"]: fk for fk in fks}

        # Column table
        lines.append("| Column | Type | Nullable | Default | Notes |")
        lines.append("|--------|------|----------|---------|-------|")
        for col in cols:
            name     = col["column_name"]
            typ      = _type_str(col)
            nullable = "✓" if col["is_nullable"] == "YES" else "✗"
            default  = col["column_default"] or ""
            if len(default) > 40:
                default = default[:37] + "…"

            notes = []
            if col["is_pk"]:
                notes.append("**PK**")
            if name in fk_map:
                fk = fk_map[name]
                notes.append(f"FK → `{fk['ref_table']}.{fk['ref_column']}`")
                all_fks.append((table, name, fk["ref_table"], fk["ref_column"]))

            lines.append(
                f"| `{name}` | `{typ}` | {nullable} | `{default}` | {', '.join(notes)} |"
            )

        lines.append("")

        # Indexes
        if idxs:
            lines.append("**Indexes**")
            for idx in idxs:
                uniq = " _(unique)_" if idx["is_unique"] else ""
                lines.append(f"- `{idx['index_name']}` on `{idx['columns']}`{uniq}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── Views ────────────────────────────────────────────────────────────
    if views:
        lines += ["## Views", ""]
        for v in views:
            lines.append(f"- `{v}`")
        lines.append("")
        lines += ["---", ""]

    # ── Relationships summary ────────────────────────────────────────────
    if all_fks:
        lines += ["## Relationships", ""]
        lines.append("| Table | Column | References |")
        lines.append("|-------|--------|------------|")
        for (tbl, col, ref_t, ref_c) in sorted(all_fks):
            lines.append(f"| `{tbl}` | `{col}` | `{ref_t}.{ref_c}` |")
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate schema_doc.md from Digiwise DB")
    parser.add_argument("--schema", default=None,
                        help="Postgres schema to document (default: auto-detect)")
    parser.add_argument("--out", default="schema_doc.md",
                        help="Output file path (default: schema_doc.md)")
    args = parser.parse_args()

    print(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME} …")
    try:
        conn = connect()
    except Exception as exc:
        print(f"ERROR: could not connect — {exc}", file=sys.stderr)
        sys.exit(1)

    if args.schema:
        schemas = [args.schema]
    else:
        # Auto: use the non-public schemas that have tables
        all_schemas = get_schemas(conn)
        schemas = []
        for s in all_schemas:
            if get_tables(conn, s):
                schemas.append(s)
        if not schemas:
            schemas = ["public"]

    print(f"Documenting schemas: {schemas}")

    sections: list[str] = []
    for schema in schemas:
        print(f"  → schema '{schema}' …")
        sections.append(build_markdown(conn, schema))

    conn.close()

    output = "\n\n".join(sections)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nDone → {args.out}")


if __name__ == "__main__":
    main()
