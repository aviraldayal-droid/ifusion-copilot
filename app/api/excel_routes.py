"""
EDA (Exploratory Data Analysis) endpoint for Excel / CSV files.

POST /api/v1/excel/analyze
  Accepts any .xlsx / .xls / .csv file plus an optional focus question.
  For TBG-format files: domain-specific KPI cards, réel/budget/N-1 trend charts,
  and a chat interface backed by the full TBG ReAct agent.
  For generic files: column metadata, descriptive stats, Plotly charts, LLM narrative.

POST /api/v1/excel/chat
  Chat against a previously analysed file (session_id from /analyze).
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_json
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.agents.graph import _make_llm, run_agent, run_agent_stream
from app.parsers.excel_parser import SHEETS_OF_INTEREST, parse_tbg_file

log = logging.getLogger("tbg.eda")
router = APIRouter(prefix="/api/v1/excel", tags=["eda"])

_PALETTE = [
    "#4F46E5", "#06B6D4", "#10B981", "#F59E0B",
    "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6",
    "#F97316", "#84CC16",
]
_ORANGE       = "#F97316"
_MAX_CHART_ROWS = 60
_MAX_SERIES     = 5
_FONT_FAMILY    = '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif'

# In-memory EDA sessions { session_id -> {is_tbg, parsed_data | context} }
_eda_sessions: dict[str, dict] = {}

# TBG metric search targets for KPI cards: (sheet_key, partial_labels, display_name)
_TBG_KPI_TARGETS = [
    ("pnl_conso",  ["CA Global", "Total Revenu", "Chiffre d'Affaires", "Revenus"],  "CA Global"),
    ("pnl_conso",  ["EBITDA"],                                                        "EBITDA"),
    ("pnl_conso",  ["Total Opex", "Opex Total", "Total des Opex"],                   "Total Opex"),
    ("pnl_conso",  ["Capex", "Total Capex", "Capex Total"],                          "Total Capex"),
    ("pnl_conso",  ["Résultat net", "Résultat Net", "EBIT", "Net result"],           "Résultat Net"),
    ("cash_conso", ["Cash Flow", "Net Cash", "Flux de trésorerie"],                  "Cash Flow"),
]

# TBG chart targets: (sheet_key, partial_labels, chart_title)
_TBG_CHART_TARGETS = [
    ("pnl_conso",  ["CA Global", "Total Revenu", "Chiffre d'Affaires"],      "CA Global"),
    ("pnl_conso",  ["EBITDA"],                                                 "EBITDA"),
    ("pnl_conso",  ["Total Opex", "Opex Total"],                              "Total Opex"),
    ("pnl_conso",  ["Capex", "Total Capex"],                                  "Total Capex"),
    ("ca_mobile",  ["CA Mobile", "Chiffre d'Affaires Mobile", "CA Global"],  "CA Mobile"),
    ("cash_conso", ["Cash Flow", "Net Cash"],                                  "Cash Flow"),
]


# ── file reading ─────────────────────────────────────────────────────────────

def _read_excel(content: bytes, filename: str) -> dict[str, pd.DataFrame]:
    buf = io.BytesIO(content)
    if filename.lower().endswith(".csv"):
        return {"Sheet1": pd.read_csv(buf)}
    return pd.read_excel(buf, sheet_name=None, engine="openpyxl")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)


# ── column classification ─────────────────────────────────────────────────────

def _classify_col(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.dtype == object:
        sample = series.dropna().head(20)
        try:
            pd.to_datetime(sample, errors="raise")
            return "date"
        except Exception:
            pass
        if series.nunique() <= max(20, len(series) * 0.3):
            return "categorical"
        return "text"
    return "other"


def _detect_x_col(df: pd.DataFrame, col_types: dict[str, str]) -> str | None:
    for col, t in col_types.items():
        if t == "date":
            return col
    for col, t in col_types.items():
        if t == "categorical" and df[col].nunique() <= 50:
            return col
    return None


def _coerce_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        pass
    return df


# ── per-column metadata ───────────────────────────────────────────────────────

def _column_meta(df: pd.DataFrame, col_types: dict[str, str]) -> list[dict]:
    n = len(df)
    meta = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        sample_val = df[col].dropna().iloc[0] if df[col].dropna().size else ""
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            sample_str = str(pd.Timestamp(sample_val).date())
        else:
            sample_str = str(sample_val)[:40]
        meta.append({
            "name":        col,
            "dtype":       col_types.get(col, "other"),
            "missing":     missing,
            "missing_pct": round(missing / n * 100, 1) if n else 0,
            "unique":      int(df[col].nunique()),
            "sample":      sample_str,
        })
    return meta


# ── descriptive statistics ────────────────────────────────────────────────────

def _describe(df: pd.DataFrame, num_cols: list[str]) -> dict[str, dict]:
    if not num_cols:
        return {}
    desc = df[num_cols].describe().round(2)
    return {col: {k: float(v) for k, v in desc[col].items()} for col in num_cols}


# ── Plotly helpers ────────────────────────────────────────────────────────────

def _base_layout(title: str) -> dict[str, Any]:
    return dict(
        title=dict(
            text=title,
            font=dict(size=12, color="#374151", family=_FONT_FAMILY),
            x=0.01, pad=dict(l=0, t=4),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT_FAMILY, size=11, color="#6B7280"),
        margin=dict(l=50, r=20, t=46, b=44),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            zeroline=False, linecolor="rgba(0,0,0,0.1)",
            ticks="outside", ticklen=3,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            zeroline=False, linecolor="rgba(0,0,0,0.1)",
            ticks="outside", ticklen=3,
        ),
        hoverlabel=dict(
            bgcolor="#09090E", font_color="#F5F5F5",
            font_size=11, font_family=_FONT_FAMILY,
        ),
    )


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= _MAX_CHART_ROWS:
        return df
    step = max(1, len(df) // _MAX_CHART_ROWS)
    return df.iloc[::step].reset_index(drop=True)


def _fig_to_dict(fig: go.Figure) -> dict:
    return json.loads(to_json(fig))


# ── chart builders ─────────────────────────────────────────────────────────────

def _trend_fig(
    df: pd.DataFrame,
    x_col: str,
    num_cols: list[str],
    col_types: dict[str, str],
) -> dict:
    df_s  = _sample(df)
    series = num_cols[:_MAX_SERIES]
    title  = f"Overview — {x_col}"

    if len(df_s) > 10:
        fig = px.line(
            df_s, x=x_col, y=series,
            color_discrete_sequence=_PALETTE,
            markers=len(df_s) <= 30,
        )
        fig.update_traces(line_width=2)
    else:
        fig = px.bar(
            df_s, x=x_col, y=series,
            barmode="group",
            color_discrete_sequence=_PALETTE,
        )
        fig.update_traces(marker_line_width=0)

    layout = _base_layout(title)
    if col_types.get(x_col) == "date":
        layout["xaxis"]["tickformat"] = "%b %Y"
    fig.update_layout(**layout)
    return _fig_to_dict(fig)


def _hist_fig(series: pd.Series, col_name: str) -> dict | None:
    clean = series.dropna()
    if len(clean) < 4:
        return None
    nbins = min(20, max(5, len(clean) // 5))
    fig = px.histogram(
        clean.to_frame(), x=col_name, nbins=nbins,
        color_discrete_sequence=[_ORANGE],
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.5)")
    layout = _base_layout(f"Distribution — {col_name}")
    layout["bargap"] = 0.05
    fig.update_layout(**layout)
    return _fig_to_dict(fig)


def _category_fig(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str | None,
) -> dict:
    if num_col:
        grp   = df.groupby(cat_col)[num_col].sum().nlargest(12).reset_index()
        x_col = num_col
        title = f"{num_col} by {cat_col}"
    else:
        vc    = df[cat_col].value_counts().head(12).reset_index()
        vc.columns = [cat_col, "count"]
        grp   = vc
        x_col = "count"
        title = f"Distribution — {cat_col}"

    fig = px.bar(
        grp, x=x_col, y=cat_col,
        orientation="h",
        color_discrete_sequence=["#06B6D4"],
    )
    fig.update_traces(marker_line_width=0)
    layout = _base_layout(title)
    layout["yaxis"]["showgrid"] = False
    layout["xaxis"]["showgrid"] = True
    fig.update_layout(**layout)
    return _fig_to_dict(fig)


def _missing_fig(df: pd.DataFrame) -> dict | None:
    miss = df.isnull().mean().mul(100).round(1)
    miss = miss[miss > 0].sort_values(ascending=True)
    if miss.empty:
        return None
    miss_df = miss.reset_index()
    miss_df.columns = ["column", "missing_pct"]
    fig = px.bar(
        miss_df, x="missing_pct", y="column",
        orientation="h",
        color_discrete_sequence=["#F59E0B"],
    )
    fig.update_traces(marker_line_width=0)
    layout = _base_layout("Missing Values (%)")
    layout["xaxis"]["ticksuffix"] = "%"
    layout["xaxis"]["range"] = [0, 100]
    layout["yaxis"]["showgrid"] = False
    fig.update_layout(**layout)
    return _fig_to_dict(fig)


def _corr_fig(df: pd.DataFrame, num_cols: list[str]) -> dict | None:
    cols = num_cols[:10]
    if len(cols) < 3:
        return None
    corr = df[cols].corr().round(2)
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        text_auto=".2f",
    )
    fig.update_traces(textfont_size=9)
    layout = _base_layout("Correlation Matrix")
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    layout["coloraxis_colorbar"] = dict(
        thickness=12, len=0.8, tickfont=dict(size=9),
    )
    fig.update_layout(**layout)
    return _fig_to_dict(fig)


def _generate_charts(
    df: pd.DataFrame,
    x_col: str | None,
    num_cols: list[str],
    cat_cols: list[str],
    col_types: dict[str, str],
) -> list[dict]:
    charts: list[dict] = []

    if x_col and num_cols:
        charts.append(_trend_fig(df, x_col, num_cols, col_types))
    elif num_cols:
        df2 = df.copy()
        df2["#"] = range(len(df2))
        charts.append(_trend_fig(df2, "#", num_cols[:3], col_types))

    if num_cols:
        col = max(
            num_cols,
            key=lambda c: df[c].std() / (abs(df[c].mean()) + 1e-9) if df[c].mean() != 0 else 0,
        )
        h = _hist_fig(df[col], col)
        if h:
            charts.append(h)

    if cat_cols:
        best_cat = min(cat_cols, key=lambda c: df[c].nunique())
        charts.append(_category_fig(df, best_cat, num_cols[0] if num_cols else None))

    corr = _corr_fig(df, num_cols)
    if corr:
        charts.append(corr)
    elif len(num_cols) >= 2 and len(charts) < 4:
        already = col if num_cols else None
        rest = [c for c in num_cols if c != already]
        if rest:
            h2 = _hist_fig(df[rest[0]], rest[0])
            if h2:
                charts.append(h2)

    miss = _missing_fig(df)
    if miss:
        charts.append(miss)

    return charts[:5]


# ── TBG-specific helpers ──────────────────────────────────────────────────────

def _is_tbg_file(sheet_names: list[str]) -> bool:
    return len(set(sheet_names) & set(SHEETS_OF_INTEREST.keys())) >= 3


def _find_metric(sheet_data: dict, partial_labels: list[str]) -> dict | None:
    """Return first metric whose label contains any of the partial_labels (case-insensitive)."""
    for _, m in sheet_data.get("metrics", {}).items():
        label_lower = m["label"].lower()
        for p in partial_labels:
            if p.lower() in label_lower:
                return m
    return None


def _extract_tbg_kpis(tbg_data: dict) -> list[dict]:
    all_periods = tbg_data.get("all_periods", [])
    if not all_periods:
        return []
    latest = all_periods[-1]
    kpis: list[dict] = []

    for sheet_key, partial_labels, display_name in _TBG_KPI_TARGETS:
        sheet = tbg_data["sheets"].get(sheet_key)
        if not sheet:
            continue
        metric = _find_metric(sheet, partial_labels)
        if not metric:
            continue
        vals = metric["values"].get(latest, {})
        reel   = vals.get("reel")
        budget = vals.get("budget")
        n1     = vals.get("n1_reel")
        if reel is None:
            continue

        vs_budget_pct: float | None = None
        if budget and budget != 0:
            vs_budget_pct = round((reel - budget) / abs(budget) * 100, 1)

        vs_n1_pct: float | None = None
        if n1 and n1 != 0:
            vs_n1_pct = round((reel - n1) / abs(n1) * 100, 1)

        kpis.append({
            "name":          display_name,
            "value":         reel,
            "budget":        budget,
            "n1":            n1,
            "period":        latest,
            "vs_budget_pct": vs_budget_pct,
            "vs_n1_pct":     vs_n1_pct,
        })

    return kpis


def _generate_tbg_charts(tbg_data: dict) -> list[dict]:
    all_periods = sorted(tbg_data.get("all_periods", []))
    charts: list[dict] = []

    for sheet_key, partial_labels, chart_title in _TBG_CHART_TARGETS:
        sheet = tbg_data["sheets"].get(sheet_key)
        if not sheet:
            continue
        metric = _find_metric(sheet, partial_labels)
        if not metric:
            continue

        reels   = [metric["values"].get(p, {}).get("reel")    for p in all_periods]
        budgets = [metric["values"].get(p, {}).get("budget")   for p in all_periods]
        n1s     = [metric["values"].get(p, {}).get("n1_reel")  for p in all_periods]

        if not any(v is not None for v in reels):
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=all_periods, y=reels,
            name="Réel", mode="lines+markers",
            line=dict(color="#4F46E5", width=2.5),
            marker=dict(size=5),
            connectgaps=True,
        ))
        if any(v is not None for v in budgets):
            fig.add_trace(go.Scatter(
                x=all_periods, y=budgets,
                name="Budget", mode="lines",
                line=dict(color="#F59E0B", width=1.5, dash="dot"),
                connectgaps=True,
            ))
        if any(v is not None for v in n1s):
            fig.add_trace(go.Scatter(
                x=all_periods, y=n1s,
                name="N-1", mode="lines",
                line=dict(color="#9CA3AF", width=1.5, dash="dash"),
                connectgaps=True,
            ))

        layout = _base_layout(f"{chart_title} (M FCFA)")
        layout["xaxis"]["tickangle"] = -35
        fig.update_layout(**layout)
        charts.append(_fig_to_dict(fig))

        if len(charts) >= 6:
            break

    return charts


# ── LLM narrative ─────────────────────────────────────────────────────────────

def _llm_prompt(
    filename: str,
    sheet: str,
    shape: tuple[int, int],
    col_meta: list[dict],
    stats: dict,
    question: str,
    language: str,
) -> str:
    lang_instr = "Respond in French." if language == "fr" else "Respond in English."
    col_lines = "\n".join(
        f"  • {m['name']} [{m['dtype']}]  missing={m['missing_pct']}%  unique={m['unique']}"
        for m in col_meta
    )
    stat_lines = ""
    for col, s in list(stats.items())[:6]:
        stat_lines += (
            f"  {col}: mean={s.get('mean','?')}, std={s.get('std','?')}, "
            f"min={s.get('min','?')}, max={s.get('max','?')}\n"
        )
    focus = f"\n\nUser's focus: {question.strip()}" if question.strip() else ""

    return f"""\
You are a senior data analyst performing exploratory data analysis. {lang_instr}

File: {filename}  |  Sheet: {sheet}  |  Shape: {shape[0]} rows × {shape[1]} columns
{focus}

Columns:
{col_lines}

Descriptive statistics:
{stat_lines or "  (no numeric columns)"}

Write a structured EDA narrative with exactly these 5 sections, each starting with the bold header:

**Data Quality** — comment on completeness, missing values, and data types.
**Key Metrics** — highlight the most important values, ranges, or totals.
**Trends & Patterns** — describe any visible trends, seasonality, or patterns.
**Anomalies** — flag outliers, unexpected values, or anything suspicious.
**Recommendations** — 2–3 concrete, actionable next steps for the analyst.

Be specific — reference actual column names and numbers from the stats above. Use **bold** for key values.
"""


def _tbg_llm_prompt(tbg_data: dict, kpis: list[dict], language: str) -> str:
    lang_instr = "Respond in French." if language == "fr" else "Respond in English."
    periods    = tbg_data.get("all_periods", [])
    sheets     = list(tbg_data.get("sheets", {}).keys())
    latest     = periods[-1] if periods else "N/A"

    kpi_lines = "\n".join(
        f"  • {k['name']}: {k['value']:.1f} M FCFA  "
        f"({'vs budget: {:+.1f}%'.format(k['vs_budget_pct']) if k['vs_budget_pct'] is not None else 'budget N/A'})"
        f"  ({'vs N-1: {:+.1f}%'.format(k['vs_n1_pct']) if k['vs_n1_pct'] is not None else 'N-1 N/A'})"
        for k in kpis
    ) or "  (no KPI data extracted)"

    return f"""\
You are a senior telecom finance analyst reviewing a TBG (Tableau de Bord de Gestion) report. {lang_instr}

File: {tbg_data.get('file', 'TBG')}
Sheets parsed: {', '.join(sheets)}
Periods available: {', '.join(periods[:6])}{'...' if len(periods) > 6 else ''}
Latest period: {latest}

Key KPIs for {latest}:
{kpi_lines}

Write a concise executive summary with exactly these 4 sections:

**Performance Overview** — summarise the latest-period results vs budget and N-1.
**Strengths** — highlight metrics performing ahead of plan or prior year.
**Concerns** — flag metrics below budget or declining vs N-1.
**Watch Points** — 2–3 items to monitor closely in the coming months.

Be specific — reference actual KPI names and percentages above. Use **bold** for key values.
"""


def _call_llm(prompt: str) -> str:
    try:
        llm    = _make_llm()
        result = llm.invoke([prompt])
        return result.content if hasattr(result, "content") else str(result)
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        return (
            "**Data Quality** — LLM analysis unavailable. Review the statistics and charts below manually.\n\n"
            "**Key Metrics** — See the Descriptive Statistics section.\n\n"
            "**Trends & Patterns** — Inspect the trend chart above.\n\n"
            "**Anomalies** — Check the distribution charts for outliers.\n\n"
            "**Recommendations** — Investigate missing values and verify data types before further analysis."
        )


def _generic_chat_prompt(context: str, question: str, language: str) -> str:
    lang_instr = "Respond in French." if language == "fr" else "Respond in English."
    return f"""\
You are a data analyst. {lang_instr}
Answer the user's question based strictly on the dataset summary below.

Dataset summary:
{context}

User question: {question}

Give a concise, specific answer referencing actual column names and values from the summary.
"""


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_excel(
    file:       UploadFile = File(...),
    question:   str        = Form(default=""),
    language:   str        = Form(default="en"),
    sheet_name: str        = Form(default=""),
):
    t0 = time.perf_counter()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"xlsx", "xls", "csv"}:
        raise HTTPException(status_code=400, detail="Only .xlsx, .xls, or .csv files are supported.")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")

    # ── TBG path ──────────────────────────────────────────────────────────────
    if ext in {"xlsx", "xls"}:
        try:
            raw_sheets = _read_excel(content, file.filename)
            sheet_names = list(raw_sheets.keys())
        except Exception:
            sheet_names = []

        if _is_tbg_file(sheet_names):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                tbg_data   = parse_tbg_file(tmp_path)
                kpis       = _extract_tbg_kpis(tbg_data)
                tbg_charts = _generate_tbg_charts(tbg_data)

                session_id = str(uuid.uuid4())
                _eda_sessions[session_id] = {
                    "is_tbg":      True,
                    "parsed_data": tbg_data,
                }

                prompt    = _tbg_llm_prompt(tbg_data, kpis, language)
                narrative = _call_llm(prompt)
                elapsed   = round(time.perf_counter() - t0, 2)

                parsed_sheets = tbg_data.get("sheets", {})
                display_sheets = [
                    name for name, key in SHEETS_OF_INTEREST.items()
                    if key in parsed_sheets
                ]

                return {
                    "is_tbg":         True,
                    "session_id":     session_id,
                    "file_name":      file.filename,
                    "all_sheets":     display_sheets,
                    "sheet_analyzed": "TBG — Multi-sheet",
                    "all_periods":    tbg_data.get("all_periods", []),
                    "shape":          [len(tbg_data.get("all_periods", [])), len(parsed_sheets)],
                    "columns":        [],
                    "stats":          {},
                    "missing_pct":    0,
                    "num_col_count":  0,
                    "charts":         [],
                    "tbg_kpis":       kpis,
                    "tbg_charts":     tbg_charts,
                    "ai_insights":    narrative,
                    "inference_time": elapsed,
                }
            except Exception as exc:
                log.error("TBG parse failed, falling through to generic: %s", exc)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # ── Generic path ──────────────────────────────────────────────────────────
    try:
        sheets = _read_excel(content, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read file: {exc}")

    if not sheets:
        raise HTTPException(status_code=422, detail="No data found in file.")

    if sheet_name and sheet_name in sheets:
        chosen = sheet_name
        df = sheets[chosen]
    else:
        chosen, df = max(
            sheets.items(),
            key=lambda kv: len(kv[1].select_dtypes(include="number").columns) * len(kv[1]),
        )

    df = _clean(df)
    if df.empty:
        raise HTTPException(status_code=422, detail="The selected sheet has no data after cleaning.")

    col_types: dict[str, str] = {col: _classify_col(df[col]) for col in df.columns}
    x_col = _detect_x_col(df, col_types)
    if x_col and col_types[x_col] == "date":
        df = _coerce_dates(df, x_col)

    num_cols = [c for c in df.columns if col_types[c] == "numeric" and c != x_col][:8]
    cat_cols = [c for c in df.columns if col_types[c] == "categorical" and c != x_col]

    if not num_cols and not cat_cols:
        raise HTTPException(
            status_code=422,
            detail="No numeric or categorical columns found to analyse.",
        )

    col_meta = _column_meta(df, col_types)
    stats    = _describe(df, num_cols)
    charts   = _generate_charts(df, x_col, num_cols, cat_cols, col_types)

    total_cells   = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    missing_pct   = round(missing_cells / total_cells * 100, 1) if total_cells else 0

    # Build a text context for future chat
    context_lines = [f"File: {file.filename}, Sheet: {chosen}, {df.shape[0]} rows × {df.shape[1]} cols"]
    for m in col_meta:
        context_lines.append(f"  {m['name']} [{m['dtype']}]: {m['unique']} unique, {m['missing_pct']}% missing")
    for col, s in list(stats.items())[:8]:
        context_lines.append(f"  {col}: mean={s.get('mean')}, std={s.get('std')}, min={s.get('min')}, max={s.get('max')}")
    context_str = "\n".join(context_lines)

    session_id = str(uuid.uuid4())
    _eda_sessions[session_id] = {
        "is_tbg":  False,
        "context": context_str,
    }

    prompt    = _llm_prompt(file.filename, chosen, df.shape, col_meta, stats, question, language)
    narrative = _call_llm(prompt)
    elapsed   = round(time.perf_counter() - t0, 2)

    return {
        "is_tbg":         False,
        "session_id":     session_id,
        "file_name":      file.filename,
        "all_sheets":     list(sheets.keys()),
        "sheet_analyzed": chosen,
        "shape":          list(df.shape),
        "columns":        col_meta,
        "stats":          stats,
        "missing_pct":    missing_pct,
        "num_col_count":  len(num_cols),
        "charts":         charts,
        "tbg_kpis":       [],
        "tbg_charts":     [],
        "ai_insights":    narrative,
        "inference_time": elapsed,
    }


class EdaChatRequest(BaseModel):
    session_id:      str
    question:        str
    language:        str = "en"
    conversation_id: str = "default"


@router.post("/chat")
async def eda_chat(req: EdaChatRequest):
    session = _eda_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="EDA session not found or expired.")

    if session["is_tbg"]:
        result = await run_agent(
            session_id=req.session_id,
            parsed_data=session["parsed_data"],
            message=req.question,
            conversation_id=req.conversation_id,
            language=req.language,
        )
        return {"answer": result["answer"], "inference_time": result.get("inference_time", 0)}
    else:
        prompt = _generic_chat_prompt(session["context"], req.question, req.language)
        answer = _call_llm(prompt)
        return {"answer": answer, "inference_time": 0}


# ── New TBG upload + streaming chat endpoints ─────────────────────────────────

import json as _json
from datetime import datetime, timezone
from fastapi.responses import StreamingResponse
from app.api.routes import _sessions, _THRESHOLDS_PATH
from app.agents.tools import run_alerts_check
from app.models.schemas import UploadResponse


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def excel_upload(file: UploadFile = File(...)):
    """
    Upload a TBG Excel file and create a main session (stores in shared _sessions).
    Equivalent to POST /api/v1/sessions.
    """
    upload_filename = file.filename or ""
    if not upload_filename:
        raise HTTPException(status_code=400, detail="File has no filename.")

    suffix = "." + upload_filename.rsplit(".", 1)[-1] if "." in upload_filename else ".xlsx"
    content = await file.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        parsed = parse_tbg_file(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not parsed.get("sheets"):
        raise HTTPException(
            status_code=422,
            detail="Could not parse any TBG sheets from the uploaded file. "
                   "Ensure it is a valid TBG Excel export.",
        )

    all_periods = parsed.get("all_periods", [])
    latest_period = all_periods[-1] if all_periods else ""
    alerts: list[dict] = []
    if latest_period:
        try:
            with open(_THRESHOLDS_PATH) as _tf:
                thresholds = _json.load(_tf)
            alerts = run_alerts_check(parsed, thresholds, latest_period)
        except Exception:
            pass

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id": session_id,
        "files": [upload_filename],
        "parsed_data": parsed,
        "alerts": alerts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return UploadResponse(
        session_id=session_id,
        file_name=upload_filename,
        sheets_parsed=list(parsed["sheets"].keys()),
        periods_available=all_periods,
        message=f"Session created. {len(parsed['sheets'])} reports parsed, "
                f"{len(all_periods)} periods available.",
        alerts=alerts,
    )


class ExcelStreamChatRequest(BaseModel):
    session_id: str
    message: str
    conversation_id: str = "default"
    model: str | None = None
    language: str = "en"
    metric_hints: list[dict] = []


@router.post("/chat/stream")
async def excel_chat_stream(
    req: ExcelStreamChatRequest,
    http_request: Request,
):
    """
    Streaming chat against a session created by POST /api/v1/excel/upload.
    Returns Server-Sent Events.
    """
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found or expired.")

    # Resolve user (may be None for anonymous) and check role-based access
    from app.auth.deps import get_optional_user
    from app.auth.policies import check_question, policy_refusal_text
    from app.db.auth_store import log_policy_block
    from app.config.settings import request_api_key

    # Extract Ollama API key and pin into request context
    per_user_key = http_request.headers.get("X-Ollama-Api-Key", "").strip()
    if not per_user_key:
        async def _no_key():
            yield f"event: error\ndata: {_json.dumps({'detail': 'NO_API_KEY'})}\n\n"
        return StreamingResponse(_no_key(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    request_api_key.set(per_user_key)

    # Optional user lookup from bearer token
    auth_header = http_request.headers.get("Authorization", "")
    token = auth_header.split(" ", 1)[1].strip() if auth_header.lower().startswith("bearer ") else None
    user = None
    if token:
        try:
            user = await get_optional_user(token=token)
        except Exception:
            user = None
    user_role = (user or {}).get("role", "viewer")
    from app.config.settings import request_user_role
    request_user_role.set(user_role)

    # Policy keyword scan
    allowed, blocked_term = check_question(user_role, req.message)
    if not allowed:
        if user:
            import asyncio as _asyncio
            await _asyncio.to_thread(
                log_policy_block, user["id"], user_role,
                req.message, blocked_term, "keyword block (excel_chat_stream)"
            )
        refusal = policy_refusal_text(blocked_term, user_role, language=req.language)
        async def _refusal_gen():
            yield f"event: token\ndata: {_json.dumps({'text': refusal})}\n\n"
            yield f"event: done\ndata: {_json.dumps({'inference_time': 0.0, 'charts': []})}\n\n"
        return StreamingResponse(_refusal_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    parsed_data = session["parsed_data"]
    hints = req.metric_hints if req.metric_hints else None

    async def event_generator():
        try:
            async for chunk in run_agent_stream(
                session_id=req.session_id,
                parsed_data=parsed_data,
                message=req.message,
                conversation_id=req.conversation_id,
                model=req.model,
                language=req.language,
                metric_hints=hints,
            ):
                yield chunk
        except Exception as exc:
            yield f"event: error\ndata: {_json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
