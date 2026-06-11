"""
Automated Monthly Narrative Generator — TBG AI Copilot

Fetches the latest month's KPIs, CAPEX, cashflow, and commissions,
computes red/yellow/green budget signals, then calls the LLM to write
a structured bilingual management report.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ollama import Client

from app.config.settings import settings
from app.db.connection import execute

log = logging.getLogger("tbg.monthly_report")

_GREEN_ABOVE = -5     # ≥ -5 % vs budget  → on track
_RED_BELOW   = -15    # < -15 % vs budget  → at risk

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",    4: "April",
    5: "May",     6: "June",     7: "July",      8: "August",
    9: "September",10:"October",11:"November",  12:"December",
}


def _signal(pct: float | None) -> str:
    if pct is None:
        return "neutral"
    return "green" if pct >= _GREEN_ABOVE else ("yellow" if pct >= _RED_BELOW else "red")


def _fmt(val: Any) -> str:
    """Format a numeric value as Bn / M / K FCFA."""
    try:
        v = float(val)
        if abs(v) >= 1e9:
            return f"{v / 1e9:.2f} Bn FCFA"
        elif abs(v) >= 1e6:
            return f"{v / 1e6:.1f} M FCFA"
        elif abs(v) >= 1e3:
            return f"{v:,.0f} FCFA"
        return f"{v:.0f} FCFA"
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _fetch_data() -> dict[str, Any]:
    # 1. Latest period
    period_rows, _ = execute("""
        SELECT EXTRACT(YEAR  FROM MAX(date))::int        AS year,
               EXTRACT(MONTH FROM MAX(date))::int        AS month,
               TO_CHAR(MAX(date), 'FMMonth YYYY')        AS period_label
        FROM   financial_metrics_data
        WHERE  real_value IS NOT NULL
    """)
    p     = period_rows[0] if period_rows else {}
    year  = p.get("year")  or 2025
    month = p.get("month") or 1
    label = p.get("period_label") or f"{_MONTH_NAMES.get(month, '')} {year}"

    # 2. All KPIs for that month (latest version only)
    kpi_rows, _ = execute("""
        WITH dedup AS (
            SELECT DISTINCT ON (financial_metric_id, date) *
            FROM   financial_metrics_data
            ORDER  BY financial_metric_id, date, version_id DESC NULLS LAST
        )
        SELECT  fc.name                                  AS category,
                fm.name                                  AS metric,
                ROUND(d.real_value::numeric,    0)       AS actual,
                ROUND(d.budget_value::numeric,  0)       AS budget,
                ROUND(d.last_year_real_value::numeric, 0) AS prior_year,
                CASE WHEN d.budget_value IS NOT NULL AND d.budget_value <> 0
                     THEN ROUND(((d.real_value - d.budget_value) * 100.0
                                 / ABS(d.budget_value))::numeric, 1)
                     ELSE NULL END                       AS vs_budget_pct,
                CASE WHEN d.last_year_real_value IS NOT NULL
                          AND d.last_year_real_value <> 0
                     THEN ROUND(((d.real_value - d.last_year_real_value) * 100.0
                                 / ABS(d.last_year_real_value))::numeric, 1)
                     ELSE NULL END                       AS yoy_pct
        FROM    dedup d
        JOIN    financial_metric     fm ON fm.id = d.financial_metric_id
        JOIN    financial_types      ft ON ft.id = fm.financial_type_id
        JOIN    financial_categories fc ON fc.id = ft.financial_category_id
        WHERE   d.date = (
                    SELECT MAX(date) FROM financial_metrics_data
                    WHERE  real_value IS NOT NULL
                )
          AND   d.real_value IS NOT NULL
        ORDER   BY fc.name, ft.sequence_id, fm.sequence_id
    """)

    # 3. CAPEX YTD
    capex_total_rows, _ = execute("""
        SELECT ROUND(SUM(equipment + services + additional_costs)::numeric, 0) AS total
        FROM   capex_data
        WHERE  year = %s
    """, (year,))
    capex_total = float(capex_total_rows[0].get("total") or 0) if capex_total_rows else 0.0

    capex_suppliers, _ = execute("""
        SELECT cp.supplier_name,
               ROUND(SUM(cd.equipment + cd.services + cd.additional_costs)::numeric, 0) AS spend
        FROM   capex_data cd
        JOIN   capex_projects cp ON cp.id = cd.capex_projects_id
        WHERE  cd.year = %s
        GROUP  BY cp.supplier_name
        ORDER  BY spend DESC
        LIMIT  5
    """, (year,))

    capex_dirs, _ = execute("""
        SELECT cp.direction_name,
               ROUND(SUM(cd.equipment + cd.services + cd.additional_costs)::numeric, 0) AS spend
        FROM   capex_data cd
        JOIN   capex_projects cp ON cp.id = cd.capex_projects_id
        WHERE  cd.year = %s
        GROUP  BY cp.direction_name
        ORDER  BY spend DESC
        LIMIT  5
    """, (year,))

    # 4. Cashflow (current + prior year)
    cf_rows, _ = execute("""
        SELECT year, current_year_total,
               jan, feb, mar, apr, may, jun,
               jul, aug, sep, oct, nov, dec
        FROM   cashflow_data
        WHERE  year IN (%s, %s)
        ORDER  BY year
    """, (year - 1, year))

    # 5. Commissions YTD
    comm_rows, _ = execute("""
        SELECT ROUND(SUM(prodium + linarcels + easycom + somac
                         + d_commercial + aftel + senaniminde)::numeric, 0) AS total
        FROM   commission_enlevements
        WHERE  year = %s
    """, (year,))
    commissions_total = float(comm_rows[0].get("total") or 0) if comm_rows else 0.0

    # 6. Monthly CA evolution
    evo_rows, _ = execute("""
        SELECT month, ca_global, ca_voix_globale, ca_data, ca_forfaits_voix, rechargement
        FROM   monthly_evolution
        WHERE  current_year = %s
        ORDER  BY month
    """, (year,))

    return {
        "year": year, "month": month, "period_label": label,
        "kpi_rows": kpi_rows,
        "capex_total": capex_total, "capex_suppliers": capex_suppliers, "capex_dirs": capex_dirs,
        "cashflow": cf_rows,
        "commissions_total": commissions_total,
        "monthly_evolution": evo_rows,
    }


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _compute_signals(kpi_rows: list[dict]) -> dict:
    enriched = []
    counts = {"green": 0, "yellow": 0, "red": 0, "neutral": 0}
    for row in kpi_rows:
        sig = _signal(row.get("vs_budget_pct"))
        counts[sig] += 1
        enriched.append({**row, "signal": sig})
    return {"rows": enriched, "counts": counts}


# ---------------------------------------------------------------------------
# Data → compact text for LLM prompt
# ---------------------------------------------------------------------------

def _build_data_summary(data: dict, signals: dict) -> str:
    year, month, label = data["year"], data["month"], data["period_label"]
    month_cols = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    lines: list[str] = []

    lines.append(f"REPORT PERIOD: {label}")
    c = signals["counts"]
    lines.append(
        f"KPI SIGNALS: {c['green']} On Track | {c['yellow']} Watch | "
        f"{c['red']} At Risk | {c['neutral']} No Target"
    )
    lines.append("")

    # KPIs grouped by category
    cats: dict[str, list] = {}
    for row in signals["rows"]:
        cats.setdefault(row["category"], []).append(row)

    lines.append("=== KPI DETAILS ===")
    for cat, rows in cats.items():
        lines.append(f"\n[{cat}]")
        for r in rows:
            pct_b = f"{r['vs_budget_pct']:+.1f}%" if r["vs_budget_pct"] is not None else "n/a"
            pct_y = f"{r['yoy_pct']:+.1f}%" if r["yoy_pct"] is not None else "n/a"
            lines.append(
                f"  {r['metric']}: actual={_fmt(r['actual'])} "
                f"| vs budget={pct_b} | YoY={pct_y} [{r['signal'].upper()}]"
            )

    # Monthly CA evolution
    if data["monthly_evolution"]:
        lines.append("\n=== MONTHLY CA EVOLUTION ===")
        for row in data["monthly_evolution"]:
            mn = _MONTH_NAMES.get(row.get("month", 0), "?")
            lines.append(
                f"  {mn}: CA={_fmt(row.get('ca_global'))} "
                f"| Voix={_fmt(row.get('ca_voix_globale'))} "
                f"| Data={_fmt(row.get('ca_data'))}"
            )

    # Cashflow
    if data["cashflow"]:
        lines.append("\n=== CASHFLOW ===")
        for cf in data["cashflow"]:
            yr = cf.get("year", "")
            lines.append(f"  {yr}: YTD Total = {_fmt(cf.get('current_year_total'))}")
            monthly = [
                f"{m.upper()}={_fmt(cf[m]) if cf.get(m) is not None else 'n/a'}"
                for m in month_cols[:month]
            ]
            lines.append(f"  Monthly: {' | '.join(monthly)}")

    # CAPEX
    lines.append(f"\n=== CAPEX YTD {year} ===")
    lines.append(f"  Total: {_fmt(data['capex_total'])}")
    if data["capex_suppliers"]:
        lines.append("  Top suppliers:")
        for s in data["capex_suppliers"]:
            lines.append(f"    {s.get('supplier_name','?')}: {_fmt(s.get('spend',0))}")
    if data["capex_dirs"]:
        lines.append("  By direction:")
        for d in data["capex_dirs"]:
            lines.append(f"    {d.get('direction_name','?')}: {_fmt(d.get('spend',0))}")

    # Commissions
    lines.append(f"\n=== COMMISSIONS YTD {year} ===")
    lines.append(f"  Total paid: {_fmt(data['commissions_total'])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM narrative call
# ---------------------------------------------------------------------------

def _call_llm(data_summary: str, language: str, model: str | None) -> str:
    model_name = model or settings.OLLAMA_MODEL
    kwargs: dict = {"host": settings.OLLAMA_BASE_URL}
    if settings.OLLAMA_API_KEY:
        kwargs["headers"] = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
    client = Client(**kwargs)

    lang_instr = (
        "Écris entièrement en français (registre formel d'entreprise, Afrique de l'Ouest)."
        if language == "fr" else
        "Write entirely in English (formal business register)."
    )

    system = (
        "You are the Chief Financial Officer of Moov Benin preparing the monthly board report. "
        "Write clear, data-driven executive commentary. Explain WHY numbers are what they are — "
        "seasonal effects, market dynamics, regulatory context, West African telecom trends. "
        f"{lang_instr}\n\n"
        "Use EXACTLY these markdown section headers:\n"
        "# Monthly Management Report — [PERIOD]\n"
        "## Executive Summary\n"
        "## Revenue Performance\n"
        "## Profitability\n"
        "## Capital Expenditure\n"
        "## Cash Flow\n"
        "## Commercial KPIs\n"
        "## Risk Flags\n"
        "## Highlights\n\n"
        "Rules:\n"
        "- Use exact numbers from the data (do not invent figures)\n"
        "- Each section: 3–6 sentences or 3–5 bullets\n"
        "- Risk Flags and Highlights must name the specific metric and its variance %\n"
        "- Always explain business reasoning, not just figures"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Financial data:\n\n{data_summary}\n\nWrite the management report."},
    ]

    try:
        resp = client.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0.3, "num_predict": 4096},
        )
        return resp.message.content or ""
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        return f"*(Narrative generation failed: {exc})*"


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def generate_monthly_report(language: str = "fr", model: str | None = None) -> dict:
    """
    Fetch DB data, compute signals, generate LLM narrative.
    Returns a JSON-serialisable dict.
    """
    t0      = time.monotonic()
    data    = _fetch_data()
    signals = _compute_signals(data["kpi_rows"])
    summary = _build_data_summary(data, signals)
    narrative = _call_llm(summary, language, model)

    # Group enriched KPIs by category for the response
    cats: dict[str, list[dict]] = {}
    for row in signals["rows"]:
        cats.setdefault(row["category"], []).append(row)

    # Convert numeric values to float for JSON serialisation
    def _safe(v: Any) -> Any:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return v

    serialised_cats = {
        cat: [
            {**r,
             "actual":        _safe(r.get("actual")),
             "budget":        _safe(r.get("budget")),
             "prior_year":    _safe(r.get("prior_year")),
             "vs_budget_pct": _safe(r.get("vs_budget_pct")),
             "yoy_pct":       _safe(r.get("yoy_pct")),
             }
            for r in rows
        ]
        for cat, rows in cats.items()
    }

    return {
        "period":             data["period_label"],
        "year":               data["year"],
        "month":              data["month"],
        "signals_summary":    signals["counts"],
        "kpis_by_category":   serialised_cats,
        "capex_total":        data["capex_total"],
        "capex_top_suppliers": [
            {**s, "spend": _safe(s.get("spend"))} for s in data["capex_suppliers"]
        ],
        "capex_by_direction": [
            {**d, "spend": _safe(d.get("spend"))} for d in data["capex_dirs"]
        ],
        "cashflow":           [
            {k: _safe(v) if k != "year" else v for k, v in cf.items()}
            for cf in data["cashflow"]
        ],
        "commissions_total":  data["commissions_total"],
        "monthly_evolution":  [
            {k: _safe(v) if k != "month" else v for k, v in r.items()}
            for r in data["monthly_evolution"]
        ],
        "narrative":          narrative,
        "generated_in_s":     round(time.monotonic() - t0, 1),
    }
