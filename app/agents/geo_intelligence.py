"""
Geographic Revenue Intelligence — TBG AI Copilot

Aggregates MoMo reactivation and data/voice auto-recharge records
by department, providing a geographic view of revenue activity.
Column names are discovered dynamically via INFORMATION_SCHEMA so
the module is resilient to schema changes.
"""
from __future__ import annotations

import logging
from typing import Any

from app.db.connection import execute

log = logging.getLogger("tbg.geo_intel")


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _columns(table: str) -> list[str]:
    rows, _ = execute("""
        SELECT column_name
        FROM   information_schema.columns
        WHERE  table_schema = 'public' AND table_name = %s
        ORDER  BY ordinal_position
    """, (table,))
    return [r["column_name"] for r in rows]


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


# ---------------------------------------------------------------------------
# Per-table aggregation
# ---------------------------------------------------------------------------

def _agg_momo(year: int | None, month: int | None) -> list[dict]:
    cols = _columns("moov_money_reactivation_data")
    if not cols:
        return []

    dept_col   = _pick(cols, ["department", "departement", "dept"])
    comm_col   = _pick(cols, ["commune", "commun"])
    dist_col   = _pick(cols, ["district"])
    amount_col = _pick(cols, ["amount", "montant", "montant_depot", "depot", "value"])
    year_col   = _pick(cols, ["year", "annee"])
    month_col  = _pick(cols, ["month", "mois"])
    date_col   = _pick(cols, ["date", "created_at", "transaction_date", "upload_date"])

    if not dept_col:
        log.warning("moov_money_reactivation_data: no department column found (cols: %s)", cols)
        return []

    selects = [f"{dept_col} AS department"]
    if comm_col:
        selects.append(f"COUNT(DISTINCT {comm_col}) AS commune_count")
    if dist_col:
        selects.append(f"COUNT(DISTINCT {dist_col}) AS district_count")
    selects.append("COUNT(*) AS record_count")
    if amount_col:
        selects.append(f"ROUND(SUM({amount_col})::numeric, 0) AS total_amount")
        selects.append(f"ROUND(AVG({amount_col})::numeric, 0) AS avg_amount")

    where, params = _build_where(year, month, year_col, month_col, date_col)
    sql = f"""
        SELECT {', '.join(selects)}
        FROM   moov_money_reactivation_data
        {where}
        GROUP  BY {dept_col}
        ORDER  BY record_count DESC
    """
    try:
        rows, _ = execute(sql, tuple(params))
        return [_safe_row(r) for r in rows]
    except Exception as exc:
        log.warning("MoMo geo aggregation failed: %s", exc)
        return []


def _agg_recharge(year: int | None, month: int | None) -> list[dict]:
    cols = _columns("data_voix_auto_rechargement_data")
    if not cols:
        return []

    dept_col = _pick(cols, ["department", "departement", "dept"])
    comm_col = _pick(cols, ["commune", "commun", "quartier", "neighborhood"])
    dep_col  = _pick(cols, ["deposit_amount", "montant_depot", "depot", "montant", "amount"])
    act_col  = _pick(cols, ["activation_amount", "montant_activation", "activation"])
    year_col = _pick(cols, ["year", "annee"])
    month_col= _pick(cols, ["month", "mois"])
    date_col = _pick(cols, ["date", "created_at", "transaction_date", "upload_date"])

    if not dept_col:
        log.warning("data_voix_auto_rechargement_data: no department column found")
        return []

    selects = [f"{dept_col} AS department"]
    if comm_col:
        selects.append(f"COUNT(DISTINCT {comm_col}) AS commune_count")
    selects.append("COUNT(*) AS record_count")
    if dep_col:
        selects.append(f"ROUND(SUM({dep_col})::numeric, 0) AS total_deposit")
    if act_col:
        selects.append(f"ROUND(SUM({act_col})::numeric, 0) AS total_activation")
    if dep_col and act_col:
        selects.append(
            f"ROUND((SUM({dep_col}) + SUM({act_col}))::numeric, 0) AS total_combined"
        )

    where, params = _build_where(year, month, year_col, month_col, date_col)
    sql = f"""
        SELECT {', '.join(selects)}
        FROM   data_voix_auto_rechargement_data
        {where}
        GROUP  BY {dept_col}
        ORDER  BY record_count DESC
    """
    try:
        rows, _ = execute(sql, tuple(params))
        return [_safe_row(r) for r in rows]
    except Exception as exc:
        log.warning("Recharge geo aggregation failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# WHERE clause builder
# ---------------------------------------------------------------------------

def _build_where(
    year: int | None, month: int | None,
    year_col: str | None, month_col: str | None, date_col: str | None,
) -> tuple[str, list]:
    parts, params = [], []
    if year:
        if year_col:
            parts.append(f"{year_col} = %s"); params.append(year)
        elif date_col:
            parts.append(f"EXTRACT(YEAR FROM {date_col}) = %s"); params.append(year)
    if month:
        if month_col:
            parts.append(f"{month_col} = %s"); params.append(month)
        elif date_col and not year_col:
            parts.append(f"EXTRACT(MONTH FROM {date_col}) = %s"); params.append(month)
    where = ("WHERE " + " AND ".join(parts)) if parts else ""
    return where, params


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _safe_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v) if v is not None and k != "department" else v
        except (TypeError, ValueError):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Merge into combined department view
# ---------------------------------------------------------------------------

def _combine(momo: list[dict], recharge: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for r in momo:
        d = r.get("department") or "Unknown"
        merged.setdefault(d, {
            "department": d,
            "momo_records": 0, "momo_total_amount": 0,
            "recharge_records": 0, "recharge_total": 0,
        })
        merged[d]["momo_records"]      = r.get("record_count", 0)
        merged[d]["momo_total_amount"] = r.get("total_amount", 0) or 0
    for r in recharge:
        d = r.get("department") or "Unknown"
        merged.setdefault(d, {
            "department": d,
            "momo_records": 0, "momo_total_amount": 0,
            "recharge_records": 0, "recharge_total": 0,
        })
        merged[d]["recharge_records"] = r.get("record_count", 0)
        merged[d]["recharge_total"]   = r.get("total_combined") or r.get("total_deposit", 0) or 0
    return sorted(merged.values(),
                  key=lambda x: x["momo_records"] + x["recharge_records"],
                  reverse=True)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def get_geo_intelligence(year: int | None = None, month: int | None = None) -> dict[str, Any]:
    """
    Return department-level MoMo + recharge aggregates.
    If year/month are None, defaults to the latest period in financial_metrics_data.
    """
    # Resolve default period from financial data
    if year is None:
        period_rows, _ = execute("""
            SELECT EXTRACT(YEAR  FROM MAX(date))::int AS year,
                   EXTRACT(MONTH FROM MAX(date))::int AS month
            FROM   financial_metrics_data
            WHERE  real_value IS NOT NULL
        """)
        if period_rows:
            year  = period_rows[0].get("year")
            month = month or period_rows[0].get("month")

    momo_data     = _agg_momo(year, month)
    recharge_data = _agg_recharge(year, month)
    combined      = _combine(momo_data, recharge_data)

    # Top department highlight
    top = combined[0] if combined else None

    return {
        "year":                    year,
        "month":                   month,
        "momo_by_department":      momo_data,
        "recharge_by_department":  recharge_data,
        "combined_by_department":  combined,
        "top_department":          top,
        "total_departments":       len(combined),
        "total_momo_records":      sum(r.get("record_count", 0) for r in momo_data),
        "total_recharge_records":  sum(r.get("record_count", 0) for r in recharge_data),
    }
