"""
LangGraph tools for the TBG AI Copilot.

Tools are created as closures over the session's parsed data so the LLM
never needs to pass a session_id — it's baked in at graph-creation time.

Scenarios covered:
  1. Explain this report  -> query_metric, get_report_summary, list_metrics
  2. Why did X change?    -> analyze_variance, get_metric_trend
  3. Compare two periods  -> compare_periods
  4. Generate charts      -> generate_chart_spec
  5. Flag concerns        -> check_all_alerts
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from app.parsers.excel_parser import (
    compute_vs_budget_pct,
    compute_yoy_pct,
    find_metric_by_label,
)


def _fmt(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val:,.{decimals}f}"


def _pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}%"


def _severity(change_pct: float, warning: float, critical: float, higher_is_worse: bool) -> str:
    if higher_is_worse:
        if change_pct >= critical:
            return "CRITICAL"
        if change_pct >= warning:
            return "WARNING"
        return "OK"
    else:
        if change_pct <= critical:
            return "CRITICAL"
        if change_pct <= warning:
            return "WARNING"
        return "OK"


def run_alerts_check(parsed_data: dict, thresholds: dict, period: str, role: str | None = None) -> list[dict]:
    """
    Scan all metrics against threshold rules for a given period.
    Returns a list of alert dicts for CRITICAL and WARNING severities.
    When `role` is given, filter out alerts the role cannot access.
    """
    from app.auth.policies import check_metric
    sheets = parsed_data.get("sheets", {})
    threshold_rules = thresholds.get("rules", [])
    alerts: list[dict] = []

    for rule in threshold_rules:
        sheet_key = rule["sheet"]
        metric_code = rule["metric_code"]
        comparison = rule["comparison"]
        warning_pct = rule["warning_pct"]
        critical_pct = rule["critical_pct"]
        higher_is_worse = rule["direction"] == "higher_is_worse"

        sheet = sheets.get(sheet_key)
        if not sheet:
            continue

        metric = sheet["metrics"].get(metric_code)
        if not metric:
            continue

        # Role-based filter: drop alerts for sheets/sections this role can't see
        if role:
            allowed, _ = check_metric(role, sheet_key, metric.get("section"))
            if not allowed:
                continue

        vals = metric["values"].get(period)
        if not vals:
            continue

        reel = vals.get("reel")
        budget = vals.get("budget")
        n1 = vals.get("n1_reel")

        if comparison == "yoy":
            change_pct = compute_yoy_pct(reel, n1)
            comp_val = n1
            comp_label = "N-1"
        else:
            change_pct = compute_vs_budget_pct(reel, budget)
            comp_val = budget
            comp_label = "Budget"

        if change_pct is None:
            continue

        sev = _severity(change_pct, warning_pct, critical_pct, higher_is_worse)
        if sev == "OK":
            continue

        alerts.append({
            "severity": sev,
            "sheet": sheet_key,
            "metric": metric["label"],
            "period": period,
            "actual": reel,
            "comparison": comp_val,
            "comparison_label": comp_label,
            "change_pct": change_pct,
            "warning_pct": warning_pct,
            "critical_pct": critical_pct,
            "message": (
                f"[{sev}] {metric['label']} ({sheet_key}) | {period} | "
                f"Réel: {_fmt(reel)} | {comp_label}: {_fmt(comp_val)} | "
                f"Change: {_pct(change_pct)}"
            ),
        })

    return alerts


def build_tools(parsed_data: dict, thresholds: dict) -> list:
    """Return a list of LangChain tools bound to the session's parsed data."""

    sheets = parsed_data.get("sheets", {})
    all_periods = parsed_data.get("all_periods", [])
    threshold_rules = thresholds.get("rules", [])

    # ------------------------------------------------------------------
    # Helper: resolve a sheet alias to the canonical sheet key
    # ------------------------------------------------------------------
    SHEET_ALIASES: dict[str, str] = {
        "p&l": "pnl_conso",
        "pnl": "pnl_conso",
        "p&l conso": "pnl_conso",
        "pnl_conso": "pnl_conso",
        "ca mobile": "ca_mobile",
        "ca_mobile": "ca_mobile",
        "opex": "opex_consolides",
        "opex_consolides": "opex_consolides",
        "capex": "capex_consolides",
        "capex_consolides": "capex_consolides",
        "mobile money": "mobile_money",
        "mobile_money": "mobile_money",
        "parc": "parc_mobile",
        "parc mobile": "parc_mobile",
        "parc_mobile": "parc_mobile",
        "marge": "marge_mobile",
        "marge mobile": "marge_mobile",
        "marge_mobile": "marge_mobile",
        "trafic": "trafic_mobile",
        "trafic_mobile": "trafic_mobile",
        "data": "data_mobile",
        "data_mobile": "data_mobile",
        "cash": "cash_conso",
        "cash_conso": "cash_conso",
    }

    def _resolve_sheet(name: str) -> str | None:
        key = name.lower().strip()
        return SHEET_ALIASES.get(key, key if key in sheets else None)

    def _get_metric(sheet_key: str, metric_key: str) -> dict | None:
        sheet = sheets.get(sheet_key)
        if not sheet:
            return None
        metrics = sheet["metrics"]
        if metric_key in metrics:
            return metrics[metric_key]
        # Fuzzy match on label or code
        lower = metric_key.lower()
        for m in metrics.values():
            if lower in m["label"].lower() or (m["code"] and lower in m["code"].lower()):
                return m
        return None

    # ------------------------------------------------------------------
    # Role-based access helpers (read user_role from request ContextVar)
    # ------------------------------------------------------------------
    def _role_ctx() -> str:
        try:
            from app.config.settings import request_user_role
            return request_user_role.get() or "viewer"
        except Exception:
            return "viewer"

    def _refusal(sheet_key: str | None, section: str | None, reason: str) -> str:
        from app.auth.policies import get_policy
        label = get_policy(_role_ctx()).get("label", _role_ctx())
        scope = f"sheet '{sheet_key}'" + (f" / section '{section}'" if section else "")
        return (
            f"Access denied for your role ({label}). "
            f"This data ({scope}) is restricted: {reason}. "
            f"Contact the administrator if you need access."
        )

    def _sheet_allowed(sheet_key: str) -> tuple[bool, str]:
        from app.auth.policies import check_metric
        return check_metric(_role_ctx(), sheet_key, None)

    def _metric_allowed(sheet_key: str, metric: dict | None) -> tuple[bool, str]:
        from app.auth.policies import check_metric
        section = (metric or {}).get("section") if metric else None
        return check_metric(_role_ctx(), sheet_key, section)

    def _filter_sheets_for_role() -> list[str]:
        """Return only the sheet keys the current role can access."""
        from app.auth.policies import check_metric
        role = _role_ctx()
        return [sk for sk in sheets.keys() if check_metric(role, sk, None)[0]]

    def _filter_metrics_for_role(sheet_key: str, metrics: dict) -> dict:
        """Return only the metrics whose section the current role can access."""
        from app.auth.policies import check_metric
        role = _role_ctx()
        out: dict = {}
        for k, m in metrics.items():
            ok, _ = check_metric(role, sheet_key, m.get("section"))
            if ok:
                out[k] = m
        return out

    # ===================================================================
    # Tool 1 – list_available_sheets
    # ===================================================================
    @tool
    def list_available_sheets() -> str:
        """
        List all available TBG report sheets parsed from the uploaded file.
        Use this to discover what data is available before querying metrics.
        """
        allowed = _filter_sheets_for_role()
        lines = ["Available TBG report sheets:"]
        for key in allowed:
            sheet = sheets[key]
            allowed_metrics = _filter_metrics_for_role(key, sheet.get("metrics", {}))
            periods = sheet.get("periods", [])
            period_range = f"{periods[0]} to {periods[-1]}" if periods else "N/A"
            lines.append(f"  • {key}: {len(allowed_metrics)} metrics | {period_range}")
        lines.append(f"\nAll available periods: {', '.join(all_periods)}")
        return "\n".join(lines)

    # ===================================================================
    # Tool 2 – list_metrics
    # ===================================================================
    @tool
    def list_metrics(sheet_name: str) -> str:
        """
        List all metrics available in a given TBG sheet.

        Args:
            sheet_name: Sheet key such as 'pnl_conso', 'ca_mobile', 'opex_consolides',
                        'capex_consolides', 'mobile_money', 'parc_mobile', 'marge_mobile'.
        """
        key = _resolve_sheet(sheet_name)
        if not key:
            return f"Sheet '{sheet_name}' not found. Available: {_filter_sheets_for_role()}"
        ok, reason = _sheet_allowed(key)
        if not ok:
            return _refusal(key, None, reason)
        sheet = sheets[key]
        filtered_metrics = _filter_metrics_for_role(key, sheet["metrics"])
        lines = [f"Metrics in '{key}':"]
        for mkey, m in filtered_metrics.items():
            code_str = f" [{m['code']}]" if m.get("code") else ""
            section_str = f" (section: {m['section']})" if m.get("section") else ""
            lines.append(f"  {mkey}{code_str}{section_str}: {m['label']}")
        return "\n".join(lines)

    # ===================================================================
    # Tool 3 – query_metric
    # ===================================================================
    @tool
    def query_metric(sheet_name: str, metric_key: str, period: str) -> str:
        """
        Retrieve all value types (Réel, Budget, Écart, N-1, Évolution%) for a
        specific metric in a given month.

        Args:
            sheet_name: e.g. 'pnl_conso', 'ca_mobile', 'opex_consolides' …
            metric_key: metric code (e.g. 'PL1', 'CA1') or label substring
                        (e.g. 'Chiffre d affaires', 'CA Global')
            period:     month in YYYY-MM format, e.g. '2025-01'
        """
        key = _resolve_sheet(sheet_name)
        if not key:
            return f"Sheet '{sheet_name}' not found."
        ok, reason = _sheet_allowed(key)
        if not ok:
            return _refusal(key, None, reason)

        metric = _get_metric(key, metric_key)
        if not metric:
            # Try fuzzy search and suggest
            matches = find_metric_by_label(parsed_data, key, metric_key)
            if matches:
                suggestions = ", ".join(f"{k} ({lbl})" for k, lbl in matches[:5])
                return f"Metric '{metric_key}' not found. Did you mean: {suggestions}?"
            return f"Metric '{metric_key}' not found in sheet '{key}'."

        ok, reason = _metric_allowed(key, metric)
        if not ok:
            return _refusal(key, metric.get("section"), reason)

        vals = metric["values"].get(period)
        if not vals:
            available = sorted(metric["values"].keys())
            return (
                f"No data for period '{period}'. "
                f"Available periods: {', '.join(available)}"
            )

        reel = vals.get("reel")
        budget = vals.get("budget")
        n1 = vals.get("n1_reel")
        ecart = vals.get("ecart_budget")
        evol = vals.get("evol_pct")

        vs_bud_pct = compute_vs_budget_pct(reel, budget)
        yoy_pct = compute_yoy_pct(reel, n1)

        lines = [
            f"{metric['label']} ({period}):",
            f"  Réel (Actual) : {_fmt(reel)} M CFA",
            f"  Budget        : {_fmt(budget)} M CFA",
            f"  Écart/Budget  : {_fmt(ecart)} M CFA  ({_pct(vs_bud_pct)})",
            f"  N-1 Réel      : {_fmt(n1)} M CFA",
            f"  Évolution YoY : {_pct(yoy_pct)}",
        ]
        return "\n".join(lines)

    # ===================================================================
    # Tool 4 – get_report_summary
    # ===================================================================
    @tool
    def get_report_summary(sheet_name: str, period: str, top_n: int = 15) -> str:
        """
        Return a formatted summary of the most important metrics in a sheet
        for a given period, ranked by absolute Réel value.

        Args:
            sheet_name: Sheet key, e.g. 'pnl_conso'.
            period:     Month in YYYY-MM format.
            top_n:      Maximum number of metrics to show (default 15).
        """
        key = _resolve_sheet(sheet_name)
        if not key:
            return f"Sheet '{sheet_name}' not found."
        ok, reason = _sheet_allowed(key)
        if not ok:
            return _refusal(key, None, reason)

        sheet = sheets[key]
        filtered = _filter_metrics_for_role(key, sheet["metrics"])
        rows: list[tuple[float, str]] = []

        for mkey, m in filtered.items():
            vals = m["values"].get(period)
            if not vals:
                continue
            reel = vals.get("reel")
            budget = vals.get("budget")
            n1 = vals.get("n1_reel")
            if reel is None:
                continue
            vs_bud = compute_vs_budget_pct(reel, budget)
            yoy = compute_yoy_pct(reel, n1)
            rows.append((
                abs(reel),
                f"{m['label']}: Réel={_fmt(reel)} | Budget={_fmt(budget)} "
                f"({_pct(vs_bud)} vs Bgt) | N-1={_fmt(n1)} ({_pct(yoy)} YoY)"
            ))

        rows.sort(reverse=True)
        lines = [f"Summary of '{key}' — Period: {period}", ""]
        for _, line in rows[:top_n]:
            lines.append(f"  • {line}")

        if not rows:
            return f"No data found for sheet '{key}' in period '{period}'."
        return "\n".join(lines)

    # ===================================================================
    # Tool 5 – analyze_variance
    # ===================================================================
    @tool
    def analyze_variance(sheet_name: str, parent_metric_key: str, period: str) -> str:
        """
        Decompose the year-over-year change of a metric by ranking all sub-metrics
        in the same sheet by their absolute YoY impact.
        Use this for Scenario 2 (Why did X change?).

        Args:
            sheet_name:        Sheet key, e.g. 'pnl_conso' or 'opex_consolides'.
            parent_metric_key: The metric to analyse, e.g. 'PL1' or 'Opex1'.
            period:            Month in YYYY-MM format.
        """
        key = _resolve_sheet(sheet_name)
        if not key:
            return f"Sheet '{sheet_name}' not found."
        ok, reason = _sheet_allowed(key)
        if not ok:
            return _refusal(key, None, reason)

        sheet = sheets[key]
        parent = _get_metric(key, parent_metric_key)
        if parent:
            ok, reason = _metric_allowed(key, parent)
            if not ok:
                return _refusal(key, parent.get("section"), reason)
        filtered = _filter_metrics_for_role(key, sheet["metrics"])

        decomp: list[tuple[float, str]] = []

        for mkey, m in filtered.items():
            vals = m["values"].get(period)
            if not vals:
                continue
            reel = vals.get("reel")
            n1 = vals.get("n1_reel")
            budget = vals.get("budget")
            if reel is None or n1 is None:
                continue
            delta = reel - n1
            yoy_pct = compute_yoy_pct(reel, n1)
            vs_bud_pct = compute_vs_budget_pct(reel, budget)
            decomp.append((abs(delta), f"{m['label']}: Δ={_fmt(delta)} M CFA "
                f"({_pct(yoy_pct)} YoY) | vs Budget {_pct(vs_bud_pct)}"))

        decomp.sort(reverse=True)

        header = f"Variance decomposition — '{key}' — {period}"
        if parent:
            p_vals = parent["values"].get(period, {})
            p_reel = p_vals.get("reel")
            p_n1 = p_vals.get("n1_reel")
            p_yoy = compute_yoy_pct(p_reel, p_n1)
            header += (
                f"\n{parent['label']}: Réel={_fmt(p_reel)} | N-1={_fmt(p_n1)} "
                f"| YoY change: {_pct(p_yoy)}"
            )

        lines = [header, "", "Contributors ranked by absolute YoY impact:"]
        for rank, (_, desc) in enumerate(decomp[:20], 1):
            lines.append(f"  {rank:2}. {desc}")

        return "\n".join(lines)

    # ===================================================================
    # Tool 6 – get_metric_trend
    # ===================================================================
    @tool
    def get_metric_trend(sheet_name: str, metric_key: str, num_periods: int = 12) -> str:
        """
        Return the month-by-month trend of a metric (Réel, Budget, N-1) across
        the most recent N periods. Use this to identify trends and acceleration.

        Args:
            sheet_name:  Sheet key, e.g. 'ca_mobile'.
            metric_key:  Metric code or label substring.
            num_periods: How many months to show (default 12).
        """
        key = _resolve_sheet(sheet_name)
        if not key:
            return f"Sheet '{sheet_name}' not found."
        ok, reason = _sheet_allowed(key)
        if not ok:
            return _refusal(key, None, reason)

        metric = _get_metric(key, metric_key)
        if not metric:
            return f"Metric '{metric_key}' not found in sheet '{key}'."
        ok, reason = _metric_allowed(key, metric)
        if not ok:
            return _refusal(key, metric.get("section"), reason)

        periods_with_data = sorted(metric["values"].keys())[-num_periods:]
        lines = [f"Trend: {metric['label']} ({key})", ""]
        lines.append(f"{'Period':<12} {'Réel':>12} {'Budget':>12} {'N-1':>12} {'YoY %':>10} {'vs Bgt %':>10}")
        lines.append("-" * 72)

        for p in periods_with_data:
            vals = metric["values"][p]
            reel = vals.get("reel")
            budget = vals.get("budget")
            n1 = vals.get("n1_reel")
            yoy = compute_yoy_pct(reel, n1)
            vs_bud = compute_vs_budget_pct(reel, budget)
            lines.append(
                f"{p:<12} {_fmt(reel):>12} {_fmt(budget):>12} {_fmt(n1):>12} "
                f"{_pct(yoy):>10} {_pct(vs_bud):>10}"
            )

        return "\n".join(lines)

    # ===================================================================
    # Tool 7 – compare_periods
    # ===================================================================
    @tool
    def compare_periods(sheet_name: str, period1: str, period2: str, top_n: int = 20) -> str:
        """
        Compare all metrics in a sheet between two periods and rank them by the
        absolute magnitude of change. Use this for Scenario 3.

        Args:
            sheet_name: Sheet key, e.g. 'pnl_conso'.
            period1:    Earlier period in YYYY-MM format, e.g. '2025-01'.
            period2:    Later period in YYYY-MM format, e.g. '2025-02'.
            top_n:      Maximum rows to return (default 20).
        """
        key = _resolve_sheet(sheet_name)
        if not key:
            return f"Sheet '{sheet_name}' not found."
        ok, reason = _sheet_allowed(key)
        if not ok:
            return _refusal(key, None, reason)

        sheet = sheets[key]
        filtered = _filter_metrics_for_role(key, sheet["metrics"])
        changes: list[tuple[float, str]] = []

        for mkey, m in filtered.items():
            v1 = m["values"].get(period1, {})
            v2 = m["values"].get(period2, {})
            r1 = v1.get("reel")
            r2 = v2.get("reel")
            if r1 is None or r2 is None:
                continue
            delta = r2 - r1
            pct_change = round((r2 - r1) / abs(r1) * 100, 2) if r1 != 0 else None
            sign = "▲" if delta > 0 else "▼"
            changes.append((
                abs(delta),
                f"{sign} {m['label']}: {_fmt(r1)} → {_fmt(r2)} "
                f"(Δ {_fmt(delta)}, {_pct(pct_change)})"
            ))

        changes.sort(reverse=True)
        lines = [
            f"Period Comparison — '{key}'",
            f"  {period1}  vs  {period2}",
            f"  (ranked by absolute change)",
            "",
        ]
        for _, desc in changes[:top_n]:
            lines.append(f"  • {desc}")

        if not changes:
            return f"No comparable data found for '{key}' between {period1} and {period2}."
        return "\n".join(lines)

    # ===================================================================
    # Tool 8 – generate_chart_spec
    # ===================================================================
    @tool
    def generate_chart_spec(
        chart_type: str,
        sheet_name: str,
        metric_keys: str,
        periods: str,
        value_types: str = "reel,budget,n1_reel",
    ) -> str:
        """
        Generate a JSON chart specification for visualisation (Scenario 4).
        The spec can be consumed by any chart library (Recharts, Chart.js, etc.).

        Args:
            chart_type:   One of 'bar', 'line', 'pie', 'waterfall'.
            sheet_name:   Sheet key, e.g. 'pnl_conso'.
            metric_keys:  Comma-separated metric codes/labels, e.g. 'CA1,CA10,CA16'.
                          Use 'top5' to auto-select the 5 largest metrics.
            periods:      Comma-separated YYYY-MM values, e.g. '2025-01,2025-02,2025-03'.
                          Use 'all' for all available periods.
            value_types:  Comma-separated value types to include.
                          Options: reel, budget, n1_reel, ecart_budget, evol_pct.
                          Default: 'reel,budget,n1_reel'.
        """
        key = _resolve_sheet(sheet_name)
        if not key:
            return f"Sheet '{sheet_name}' not found."
        ok, reason = _sheet_allowed(key)
        if not ok:
            return _refusal(key, None, reason)

        sheet = sheets[key]
        requested_periods = (
            all_periods if periods.strip().lower() == "all"
            else [p.strip() for p in periods.split(",")]
        )
        requested_vtypes = [v.strip() for v in value_types.split(",")]

        # Resolve metric keys
        if metric_keys.strip().lower() == "top5":
            # Pick the 5 metrics with the largest average Réel value
            scored: list[tuple[float, str]] = []
            for mkey, m in sheet["metrics"].items():
                vals = [
                    m["values"][p].get("reel")
                    for p in requested_periods
                    if p in m["values"] and m["values"][p].get("reel") is not None
                ]
                if vals:
                    scored.append((sum(abs(v) for v in vals) / len(vals), mkey))
            scored.sort(reverse=True)
            requested_metric_keys = [mk for _, mk in scored[:5]]
        else:
            requested_metric_keys = [k.strip() for k in metric_keys.split(",")]

        # Build chart data
        chart_data: list[dict[str, Any]] = []
        for p in requested_periods:
            row: dict[str, Any] = {"period": p}
            for mkey in requested_metric_keys:
                metric = _get_metric(key, mkey)
                if not metric:
                    continue
                m_label = metric["label"][:20]  # truncate for readability
                m_vals = metric["values"].get(p, {})
                for vtype in requested_vtypes:
                    col_name = f"{m_label}_{vtype}"
                    row[col_name] = m_vals.get(vtype)
            chart_data.append(row)

        y_keys = []
        for mkey in requested_metric_keys:
            metric = _get_metric(key, mkey)
            if metric:
                m_label = metric["label"][:20]
                for vtype in requested_vtypes:
                    y_keys.append(f"{m_label}_{vtype}")

        spec = {
            "chart_type": chart_type,
            "title": f"{key.replace('_', ' ').title()} — {chart_type.title()} Chart",
            "sheet": key,
            "periods": requested_periods,
            "x_key": "period",
            "y_keys": y_keys,
            "data": chart_data,
            "unit": "M CFA",
            "colors": ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"],
        }

        return json.dumps(spec, ensure_ascii=False, indent=2)

    # ===================================================================
    # Tool 9 – check_all_alerts
    # ===================================================================
    @tool
    def check_all_alerts(period: str) -> str:
        """
        Scan ALL metrics in ALL sheets against threshold rules for a given period
        and return a prioritised list of anomalies. Use this for Scenario 5.

        Args:
            period: Month to check in YYYY-MM format, e.g. '2025-12'.
        """
        critical_alerts: list[str] = []
        warning_alerts: list[str] = []
        ok_items: list[str] = []

        for rule in threshold_rules:
            sheet_key = rule["sheet"]
            metric_code = rule["metric_code"]
            comparison = rule["comparison"]
            warning_pct = rule["warning_pct"]
            critical_pct = rule["critical_pct"]
            higher_is_worse = rule["direction"] == "higher_is_worse"
            unit = rule.get("unit", "M CFA")

            sheet = sheets.get(sheet_key)
            if not sheet:
                continue

            metric = sheet["metrics"].get(metric_code)
            if not metric:
                continue

            # Skip metrics the user's role can't see
            allowed_metric, _ = _metric_allowed(sheet_key, metric)
            if not allowed_metric:
                continue

            vals = metric["values"].get(period)
            if not vals:
                continue

            reel = vals.get("reel")
            budget = vals.get("budget")
            n1 = vals.get("n1_reel")

            if comparison == "yoy":
                change_pct = compute_yoy_pct(reel, n1)
                comp_val = n1
                comp_label = "N-1"
            else:  # vs_budget
                change_pct = compute_vs_budget_pct(reel, budget)
                comp_val = budget
                comp_label = "Budget"

            if change_pct is None:
                continue

            sev = _severity(change_pct, warning_pct, critical_pct, higher_is_worse)
            msg = (
                f"[{sev}] {metric['label']} ({sheet_key}) | {period} | "
                f"Réel: {_fmt(reel)} {unit} | {comp_label}: {_fmt(comp_val)} {unit} | "
                f"Change: {_pct(change_pct)} (threshold: {_pct(warning_pct)} / {_pct(critical_pct)})"
            )

            if sev == "CRITICAL":
                critical_alerts.append(msg)
            elif sev == "WARNING":
                warning_alerts.append(msg)
            else:
                ok_items.append(msg)

        lines = [f"=== ALERT SCAN — Period: {period} ===", ""]

        if critical_alerts:
            lines.append(f"🔴 CRITICAL ({len(critical_alerts)} alerts):")
            lines.extend(f"   {a}" for a in critical_alerts)
            lines.append("")

        if warning_alerts:
            lines.append(f"🟡 WARNING ({len(warning_alerts)} alerts):")
            lines.extend(f"   {a}" for a in warning_alerts)
            lines.append("")

        if ok_items:
            lines.append(f"✅ OK ({len(ok_items)} metrics within thresholds):")
            lines.extend(f"   {a}" for a in ok_items)

        if not critical_alerts and not warning_alerts:
            lines.append("No threshold breaches detected for this period.")

        return "\n".join(lines)

    # ===================================================================
    # Tool 10 – compare_two_files_period
    # ===================================================================
    @tool
    def compare_across_all_sheets(period1: str, period2: str) -> str:
        """
        Compare two periods across ALL sheets and highlight the most significant
        changes. Useful when two files have been uploaded (Scenario 3 full mode).

        Args:
            period1: Earlier period YYYY-MM.
            period2: Later period YYYY-MM.
        """
        all_changes: list[tuple[float, str]] = []
        allowed_sheets = set(_filter_sheets_for_role())

        for sheet_key, sheet in sheets.items():
            if sheet_key not in allowed_sheets:
                continue
            filtered = _filter_metrics_for_role(sheet_key, sheet["metrics"])
            for mkey, m in filtered.items():
                v1 = m["values"].get(period1, {})
                v2 = m["values"].get(period2, {})
                r1 = v1.get("reel")
                r2 = v2.get("reel")
                if r1 is None or r2 is None or r1 == 0:
                    continue
                delta = r2 - r1
                pct = round((r2 - r1) / abs(r1) * 100, 2)
                all_changes.append((
                    abs(pct),
                    f"{sheet_key}/{m['label']}: {_fmt(r1)} → {_fmt(r2)} "
                    f"(Δ {_fmt(delta)}, {_pct(pct)})"
                ))

        all_changes.sort(reverse=True)
        lines = [
            f"Cross-sheet comparison: {period1} vs {period2}",
            f"Top movements by % change:",
            "",
        ]
        for _, desc in all_changes[:25]:
            lines.append(f"  • {desc}")

        return "\n".join(lines)

    return [
        list_available_sheets,
        list_metrics,
        query_metric,
        get_report_summary,
        analyze_variance,
        get_metric_trend,
        compare_periods,
        generate_chart_spec,
        check_all_alerts,
        compare_across_all_sheets,
    ]
