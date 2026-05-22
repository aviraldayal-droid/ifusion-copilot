"""
LangGraph agents for the Digiwise AI Copilot.

DB pipeline:
  0. retrieve_schema    — RAG: pick 2–4 relevant tables from embeddings
  1. write_sql          — Writer LLM generates SQL (strict: ONLY SQL output)
  2. validate_syntax    — sqlglot checks syntax (no LLM, no DB)
  3. validate_tables    — whitelist check: blocks hallucinated table names
  4. validate_semantic  — EXPLAIN checks real table/col names against DB
  4. critique_sql       — Critic LLM repairs SQL with full error history
     └── loops back to validate_syntax (max _MAX_RETRIES times)
  5. execute_sql        — runs the validated SQL
  6. format_answer      — formats rows into natural language (no SQL leaked)

All nodes emit INFO/WARNING/ERROR log lines for full traceability.
On retry exhaustion the pipeline returns a structured SQL_GENERATION_FAILED error.
"""
from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Iterator, TypedDict

import sqlglot
import sqlglot.expressions as sexp
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolCall
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from app.agents.db_tools import build_db_tools
from app.agents.tools import build_tools
from app.config.settings import settings
from app.db.connection import execute, explain

_THRESHOLDS_PATH     = Path(__file__).parent.parent / "thresholds.json"
_FINANCIAL_TERMS_PATH = Path(__file__).resolve().parents[2] / "financial_terms.json"
_DEFAULT_EXCEL_PATH  = Path(__file__).resolve().parents[2] / "TBG Moov_Africa_Bénin DEC 2025 DF SANS LIEN.xlsx"

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

log = logging.getLogger("tbg.pipeline")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_h)
log.setLevel(logging.DEBUG)


def _clip(text: str, n: int = 140) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


# ---------------------------------------------------------------------------
# Shared LLM factory
# ---------------------------------------------------------------------------
from ollama import Client

class OllamaLLM:
    """Wrapper for Ollama client that implements invoke() interface for LangGraph."""
    def __init__(self, client: Client, model: str):
        self.client = client
        self.model = model
        self.tools = None

    def bind_tools(self, tools, **kwargs):
        """Allow LangGraph to attach tools to the model."""
        self.tools = tools
        return self

    def _prepare_messages(self, messages) -> list[dict]:
        """Convert LangChain or raw messages to Ollama format."""
        msgs = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role, content = "system", m.content
            elif isinstance(m, HumanMessage):
                role, content = "user", m.content
            elif isinstance(m, str):
                role, content = "user", m
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content") or m.get("text") or ""
            elif hasattr(m, "content"):
                role = getattr(m, "role", "user")
                content = m.content
            else:
                raise ValueError(f"Unsupported message type: {type(m)}")
            if content:
                msgs.append({"role": role, "content": content})
        return msgs

    def invoke(self, messages):
        """Convert LangChain or raw messages to Ollama format and get response."""
        msgs = self._prepare_messages(messages)
        try:
            kwargs: dict = {"model": self.model, "messages": msgs}
            if self.tools:
                ollama_tools = []
                for t in self.tools:
                    try:
                        schema = t.args_schema.schema()
                    except Exception:
                        schema = {}
                    ollama_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": schema,
                        },
                    })
                kwargs["tools"] = ollama_tools
            response = self.client.chat(**kwargs)
            content = response.get('message', {}).get('content', '') or ''
            tool_calls_raw = response.get('message', {}).get('tool_calls') or []
            lc_tc = []
            for i, tc in enumerate(tool_calls_raw):
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                lc_tc.append(ToolCall(
                    name=fn.get("name", ""),
                    args=args,
                    id=f"call_{fn.get('name', '')}_{i}",
                ))
            return AIMessage(content=content, tool_calls=lc_tc) if lc_tc else AIMessage(content=content)
        except Exception as e:
            log.error("Ollama API call failed: %s", str(e))
            raise

    def __call__(self, messages, *args, **kwargs):
        return self.invoke(messages)

    def stream(self, messages, **kwargs):
        """LangChain-compatible sync streaming."""
        msgs = self._prepare_messages(messages)
        for chunk in self.client.chat(model=self.model, messages=msgs, stream=True):
            token = chunk.get('message', {}).get('content', '')
            if token:
                yield AIMessageChunk(content=token)

    def invoke_streaming(self, messages) -> AIMessage:
        """Like invoke() but uses the streaming API so the read-timeout is per-token,
        not per-full-response. Avoids timeouts with slow/large models (e.g. 123B)."""
        msgs = self._prepare_messages(messages)
        try:
            content = ""
            for chunk in self.client.chat(model=self.model, messages=msgs, stream=True):
                token = chunk.get("message", {}).get("content", "")
                if token:
                    content += token
            return AIMessage(content=content)
        except Exception as e:
            log.error("Ollama API call failed: %s", str(e))
            raise

    def stream_tokens(self, messages) -> Iterator[str]:
        """Stream response tokens from Ollama."""
        msgs = self._prepare_messages(messages)
        try:
            for chunk in self.client.chat(model=self.model, messages=msgs, stream=True):
                token = chunk.get('message', {}).get('content', '')
                if token:
                    yield token
        except Exception as e:
            log.error("Ollama stream failed: %s", str(e))
            raise


_OLLAMA_TIMEOUT = 300  # seconds per-token read timeout (streaming resets this per chunk)


def _make_llm(model: str | None = None) -> OllamaLLM:
    """Create Ollama client for local or cloud Ollama."""
    resolved = model or settings.OLLAMA_MODEL
    if settings.is_ollama_cloud:
        headers = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
        client = Client(host=settings.OLLAMA_BASE_URL, headers=headers, timeout=_OLLAMA_TIMEOUT)
        log.info("LLM initialized — API: OLLAMA CLOUD | model: %s | timeout: %ds", resolved, _OLLAMA_TIMEOUT)
    else:
        client = Client(host=settings.OLLAMA_BASE_URL, timeout=_OLLAMA_TIMEOUT)
        log.info("LLM initialized — API: LOCAL OLLAMA | model: %s | base_url: %s", resolved, settings.OLLAMA_BASE_URL)
    return OllamaLLM(client, resolved)


# ---------------------------------------------------------------------------
# File-upload session graph (unchanged)
# ---------------------------------------------------------------------------

_graph_cache: dict[str, object] = {}


def _load_thresholds() -> dict:
    with open(_THRESHOLDS_PATH) as f:
        return json.load(f)


def _build_system_prompt(parsed_data: dict) -> str:
    sheets     = list(parsed_data.get("sheets", {}).keys())
    periods    = parsed_data.get("all_periods", [])
    period_range = f"{periods[0]} to {periods[-1]}" if periods else "unknown"
    file_name  = parsed_data.get("file", "uploaded file")
    return f"""You are the TBG AI Copilot — an expert financial analyst assistant for Moov Benin.

You have access to data parsed from the TBG report: {file_name}
Available sheets: {', '.join(sheets)}
Available periods: {period_range}

Rules:
- Always use tools to retrieve real data; do NOT invent numbers.
- For monetary values: thousands separators, one decimal place, unit M CFA.
- For percentages: always show the sign (+/-).
- After answering, offer the next logical follow-up question.
- After retrieving numeric data spanning multiple periods or entities, call generate_chart_spec to visualise it. Use 'line' for time-series, 'bar' for category rankings.
"""


def get_or_create_graph(session_id: str, parsed_data: dict, model: str | None = None):
    key = f"{session_id}:{model or settings.OLLAMA_MODEL}"
    if key in _graph_cache:
        return _graph_cache[key]
    thresholds = _load_thresholds()
    tools      = build_tools(parsed_data, thresholds)
    llm        = _make_llm(model)
    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=_build_system_prompt(parsed_data)),
        checkpointer=MemorySaver(),
    )
    _graph_cache[key] = graph
    return graph


# ---------------------------------------------------------------------------
# DB pipeline state
# ---------------------------------------------------------------------------

class DbPipelineState(TypedDict):
    question:         str
    history:          str
    language:         str          # "en" or "fr" — controls answer language
    # Snapshot routing
    target_db:        str          # DB name to execute against; "" = main DB
    snapshot_label:   str          # human-readable snapshot description for the answer
    # RAG output
    retrieved_schema: str          # compact schema for relevant tables only
    allowed_tables:   list[str]    # whitelist — blocks hallucinated table names
    # SQL under construction
    sql:              str
    # Validation errors (one set at a time)
    syntax_error:     str          # from sqlglot
    table_error:      str          # table not in whitelist
    semantic_error:   str          # from EXPLAIN
    # Repair loop
    error_history:    list[str]    # accumulated across all retries
    column_facts:     str          # verified columns for tables in SQL (injected on semantic fail)
    critic_feedback:  str
    retry_count:      int
    # Execution
    sql_error:        str
    rows:             list[dict]
    cols:             list[str]
    answer:           str
    chart_specs:      list[dict]


_MAX_RETRIES = 3      # writer attempt + up to 3 critic repairs = 4 total SQL generations


# ---------------------------------------------------------------------------
# SQL extraction — strict SELECT/WITH guard
# ---------------------------------------------------------------------------

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)
_SQL_BARE_RE  = re.compile(
    r"((?:SELECT|WITH)\b[\s\S]*?)(?:;|\Z)",
    re.IGNORECASE,
)


def _extract_sql(raw: str) -> tuple[str, str]:
    """
    Returns (sql, error).
    Strips markdown fences and prose; validates the result starts with SELECT/WITH.
    """
    # 1. Try fenced block first
    m = _SQL_FENCE_RE.search(raw)
    if m:
        sql = m.group(1).strip().rstrip(";")
    else:
        # 2. Find first SELECT/WITH … to end-of-string or first semicolon
        m2 = _SQL_BARE_RE.search(raw)
        sql = m2.group(1).strip().rstrip(";") if m2 else raw.strip().rstrip(";")

    upper = sql.lstrip().upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        snippet = _clip(raw, 120)
        return "", f"LLM output does not start with SELECT or WITH. Raw: {snippet!r}"
    
    log.debug("_extract_sql: extracted %d chars from %d char input", len(sql), len(raw))
    return sql, ""


# ---------------------------------------------------------------------------
# Node 0 — Snapshot resolver (pure Python, no LLM, no DB)
# ---------------------------------------------------------------------------

def resolve_snapshot(state: DbPipelineState) -> dict:
    """
    Detect explicit snapshot requests and set target_db in state.
    All normal questions pass through unchanged (target_db stays "").
    """
    from app.db.snapshots import resolve_snapshot as _resolve
    snap = _resolve(state["question"])
    if snap:
        log.info("resolve_snapshot → %s (%s)", snap.dbname, snap.label)
        return {"target_db": snap.dbname, "snapshot_label": snap.label}
    return {"target_db": "", "snapshot_label": ""}


# ---------------------------------------------------------------------------
# Node 1 — Retrieve schema (RAG)
# ---------------------------------------------------------------------------

def retrieve_schema(state: DbPipelineState) -> dict:
    from app.agents.schema_retriever import get_schema_retriever
    question = state["question"]
    try:
        retriever = get_schema_retriever()
        result    = retriever.retrieve(question)
        log.info(
            "retrieve_schema: %d tables selected (%d chars) — %s",
            len(result.tables), len(result.schema_text),
            [t.name for t in result.tables],
        )
        return {
            "retrieved_schema": result.schema_text,
            "allowed_tables":   sorted(result.allowed_tables),
        }
    except Exception as exc:
        log.error("retrieve_schema failed: %s — falling back to full schema", exc)
        try:
            from app.db.schema_inspector import build_schema_context, get_tables
            schema     = build_schema_context()
            all_tables = get_tables()
            log.info("Fallback full schema: %d tables", len(all_tables))
            return {"retrieved_schema": schema, "allowed_tables": all_tables}
        except Exception as exc2:
            log.error("Full schema fallback also failed: %s", exc2)
            return {"retrieved_schema": "(schema unavailable)", "allowed_tables": []}


# ---------------------------------------------------------------------------
# Node 1 — Writer LLM: generate SQL
# ---------------------------------------------------------------------------

_WRITER_SYSTEM = """\
You are a PostgreSQL 15 expert connected to a financial database for Moov Benin.

╔══ OUTPUT FORMAT — STRICTLY ENFORCED ══╗
║  Output ONLY a raw SQL SELECT statement  ║
║  • First token must be SELECT or WITH     ║
║  • Last character must be ;               ║
║  • ZERO prose, ZERO comments, ZERO markdown ║
║  Any text outside the SQL = REJECTED       ║
╚═══════════════════════════════════════════╝

LANGUAGE RULE:
  The user question may be in French or English — this does NOT change your output.
  You ALWAYS output SQL only. Never write a French or English sentence before or after the SQL.
  SQL column aliases must be in English (e.g. AS total_revenue, AS month, AS variance_pct).
  ✗ "Voici la requête SQL :" ← REJECTED
  ✗ "Here is the query:"     ← REJECTED
  ✓ SELECT ... FROM ...;     ← CORRECT

COLUMN ALIAS RULE (mandatory):
  Every computed expression MUST have an AS alias so column headers are readable.
  ✓  SUM(jan + feb + mar) AS q1_total
  ✓  ROUND(real_value / budget_value * 100, 2) AS pct_budget
  ✓  prodium + linarcels + easycom AS total_commission
  ✗  SUM(jan + feb)          ← produces ugly "?column?" header — FORBIDDEN

MONTH FORMATTING RULE (mandatory):
  Whenever you SELECT a month integer column, convert it to a full month name.
  ✓  to_char(to_date(cd.month::text, 'MM'), 'Month') AS month
  ✓  to_char(to_date(EXTRACT(MONTH FROM d.date)::int::text, 'MM'), 'Month') AS month
  This applies to any column named month, mois, month_number, month_no, etc.
  ✗  SELECT cd.month …        ← returns "9" — FORBIDDEN
  ✗  SELECT EXTRACT(MONTH …)  ← returns 9.0 — FORBIDDEN
  Always ORDER BY the original integer expression, not the name alias.
  ✓  ORDER BY cd.month        ← sorts correctly as 1,2,…12
  ✓  ORDER BY EXTRACT(MONTH FROM d.date)

⚠ DEDUPLICATION RULE — MANDATORY FOR financial_metrics_data:
  The financial_metrics_data table contains DUPLICATE rows for 2025 and 2026 data
  (multiple uploads created multiple version_id entries per metric per month).
  The same (financial_metric_id, date) pair can have version_id=1 (original) AND version_id=2
  (corrected later). Without dedup you return stale values alongside the current ones.

  YOU MUST ALWAYS wrap financial_metrics_data in a deduplication CTE — for EVERY query,
  including simple point lookups, not only aggregations.

  ✓ CORRECT pattern (always use this as the base CTE):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT ... FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    ...

  ✗ WRONG — inflates all values by 2–3x (aggregations):
    SELECT SUM(real_value) FROM financial_metrics_data WHERE ...

  ✗ WRONG — returns stale version alongside latest (point lookup without dedup):
    SELECT real_value FROM financial_metrics_data
    JOIN financial_metric fm ON fm.id = financial_metric_id
    WHERE fm.name = 'EBITDA' AND date = '2025-06-01';
    -- if version_id=1 (890) and version_id=2 (912) both exist → returns 2 rows, one wrong

  ✗ WRONG — using LIMIT 1 to "fix" the duplicate (silently drops valid rows):
    SELECT real_value FROM financial_metrics_data WHERE ...
    ORDER BY version_id DESC LIMIT 1;
    -- FORBIDDEN: hides other metrics that legitimately match the same filter

  ✓ CORRECT point lookup (dedup CTE, then filter):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fm.name, fmd.real_value
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    WHERE fm.name ILIKE '%EBITDA%'
      AND fmd.date = '2025-06-01';
    -- dedup picks version_id=2 (912) per metric; returns one row per matching metric

  LIMIT RULE: NEVER add LIMIT to handle versioning or to "get one result".
    ✓ Use LIMIT only when the question explicitly asks for "top N", "highest N", or "most recent N".
    ✗ LIMIT 1 as a shortcut for dedup is FORBIDDEN — it silently drops valid matching rows.

  This rule applies to ALL years including 2024. Always use the dedup CTE.

EBITDA METRIC NAME RULE:
  Use fm.name = 'EBITDA' (exact match) for all EBITDA queries.
  ✗ NEVER use ILIKE '%EBITDA%' — it also matches 'Neutralisation de la var. de provisions
    incluses dans l\'Ebitda (-)' which is a sub-adjustment entry, not the main EBITDA.
  ✓ Correct:  WHERE fm.name = 'EBITDA'
  ✗ Wrong:    WHERE fm.name ILIKE '%EBITDA%'

QUARTERLY / SINGLE-PERIOD QUERIES — for any question about a SPECIFIC named metric (EBITDA, ARPU, OPEX, etc.):
  ⚠ MANDATORY: When querying a specific metric by name, ALWAYS include budget_value and last_year_real_value.
    Omitting these columns forces the formatter to write "N/A" for budget and YoY — FORBIDDEN.
    ✓ Always SELECT: actual_m_fcfa, budget_m_fcfa, prior_year_m_fcfa, variance
    ✗ NEVER return only a single aggregate column (e.g. AS ebitda_q1_2025) — this strips budget/YoY context.

  EXCEPTION — "list ALL metrics" queries (no specific metric named, e.g. "show all metrics in January"):
    Select ONLY the columns the question asks for (typically name, real_value).
    Do NOT add budget_value or last_year_real_value unless the question explicitly requests them.
    ✓ SELECT fm.name, fmd.real_value FROM fmd ... WHERE date = '2025-01-01' ORDER BY fm.name
    ✗ Adding budget_value here bloats output and breaks comparisons with expected schema.

  QUARTER MONTH RANGES: Q1=1–3, Q2=4–6, Q3=7–9, Q4=10–12

  ✓ EBITDA for a single quarter — monthly breakdown + quarterly total (use this for any quarter query):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT
      to_char(fmd.date, 'Month YYYY')                          AS period,
      ROUND(fmd.real_value::numeric, 0)                        AS actual_m_fcfa,
      ROUND(fmd.budget_value::numeric, 0)                      AS budget_m_fcfa,
      ROUND(fmd.last_year_real_value::numeric, 0)              AS prior_year_m_fcfa,
      ROUND((fmd.real_value - fmd.budget_value)::numeric, 0)   AS variance,
      CASE WHEN fmd.budget_value != 0
           THEN ROUND(((fmd.real_value - fmd.budget_value) / fmd.budget_value * 100)::numeric, 1)
           ELSE NULL END                                        AS variance_pct
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    WHERE fm.name = 'EBITDA'
      AND EXTRACT(YEAR  FROM fmd.date) = 2025
      AND EXTRACT(MONTH FROM fmd.date) BETWEEN 1 AND 3
    ORDER BY fmd.date;

  This pattern (per-row with period label, actual, budget, prior_year, variance) applies to ALL
  single-quarter or single-month queries on financial_metrics_data — not just EBITDA.

CRITICAL — TWO TABLE STRUCTURES EXIST:

DENORMALIZED TABLES (months stored as separate columns):
  • cashflow_data: columns are [year, jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec, current_year_total]
    ✓ Query: SELECT SUM(jan)+SUM(feb)+SUM(mar)... FROM cashflow_data WHERE year=2024
    ✗ Do NOT use: EXTRACT(), date column, month column (they don't exist)

  • commission_enlevements: DISTRIBUTORS ARE COLUMNS, NOT ROWS.
    Columns: [year, month, prodium, linarcels, easycom, somac, d_commercial, aftel, senaniminde]
    ✗ WRONG — causes "must appear in GROUP BY" error:
        SELECT distributor, SUM(amount) FROM commission_enlevements GROUP BY distributor
    ✓ CORRECT — use UNION ALL to unpivot distributors into rows, then rank:
        WITH totals AS (
            SELECT 'Prodium'      AS distributor, SUM(prodium)      AS total FROM commission_enlevements WHERE year = 2025
            UNION ALL
            SELECT 'Linarcels',                   SUM(linarcels)             FROM commission_enlevements WHERE year = 2025
            UNION ALL
            SELECT 'Easycom',                     SUM(easycom)               FROM commission_enlevements WHERE year = 2025
            UNION ALL
            SELECT 'Somac',                       SUM(somac)                 FROM commission_enlevements WHERE year = 2025
            UNION ALL
            SELECT 'D-Commercial',                SUM(d_commercial)          FROM commission_enlevements WHERE year = 2025
            UNION ALL
            SELECT 'Aftel',                       SUM(aftel)                 FROM commission_enlevements WHERE year = 2025
            UNION ALL
            SELECT 'Senaniminde',                 SUM(senaniminde)           FROM commission_enlevements WHERE year = 2025
        )
        SELECT distributor, total AS total_commission
        FROM totals
        WHERE total IS NOT NULL
        ORDER BY total DESC
        LIMIT 5;

NORMALIZED TABLES (date column):
  • financial_metrics_data: has [date, real_value, budget_value, financial_metric_id, financial_submetric_id]
    ✓ Query: WHERE EXTRACT(YEAR FROM date)=2025 AND real_value IS NOT NULL
  • capex_data: has [year, month, equipment, services, additional_costs, capex_projects_id]
    ✓ Query: WHERE year=2025 AND month=9, GROUP BY month

BEFORE WRITING SQL:
  1. Identify which table(s) you'll query
  2. Check if it's denormalized (monthly columns) or normalized (date column)
  3. Use appropriate filter syntax:
     - Denormalized: WHERE year = 2024 (direct comparison)
     - Normalized: WHERE EXTRACT(YEAR FROM date) = 2024 (date extraction)

POSTGRESQL RULES:
  ✗ MONTH() YEAR() ISNULL() IFNULL()   ← MySQL functions — FORBIDDEN
  ✗ backtick quoting                    ← use double-quotes only
  ✗ CONCAT() with 2 args                ← use || operator
  ✓ NULLIF(col, 0) to prevent division by zero
  ✓ COALESCE(col, 0) for NULL handling

ROUND CASTING RULE (mandatory):
  PostgreSQL ROUND(expr, n) only works when expr is NUMERIC, not FLOAT/DOUBLE PRECISION.
  Financial columns (real_value, budget_value, last_year_real_value, division results) are
  double precision — you MUST cast to ::numeric before rounding with decimal places.
  ✗ ROUND(real_value / budget_value * 100, 2)         ← FAILS: function does not exist
  ✗ ROUND(AVG(real_value), 2)                         ← FAILS: function does not exist
  ✓ ROUND((real_value / budget_value * 100)::numeric, 2)   ← CORRECT
  ✓ ROUND(AVG(real_value)::numeric, 2)                     ← CORRECT
  Rule: always write ROUND((expr)::numeric, n) — cast the whole expression, not the column.

NULL HANDLING:
  • "List all" / "show every" / "for all metrics" queries: NEVER add WHERE real_value IS NOT NULL
    or WHERE budget_value IS NOT NULL. These questions must return every row including NULL rows.
    ✗ FORBIDDEN: WHERE real_value IS NOT NULL (drops rows the user asked to see)
    ✓ CORRECT: no IS NOT NULL filter — let NULLs appear in the result
  • Aggregations (SUM/AVG) on a SPECIFIC metric: Add WHERE value IS NOT NULL only when the
    question asks for a specific named metric trend (not "all metrics").
  • Division: Use NULLIF(denominator, 0) to guard against division by zero.
  • Variance pct: Use CASE WHEN budget_value != 0 THEN ROUND((...) * 100)::numeric, 2) ELSE NULL END
  • CAPEX aggregations only: COALESCE(cd.equipment,0) + COALESCE(cd.services,0) +
    COALESCE(cd.additional_costs,0) — only for SUM of capex spend, not for COUNT queries.

REVENUE / KPI HIERARCHY — use this for ANY question about revenue, ARPU, EBITDA, budgets, categories:
  financial_categories   ← top-level groupings (CA Mobile, Data Mobile, Mobile Money, Capex, etc.)
       └── financial_types     ← report sections within each category
             └── financial_metric   ← individual KPI metric names
                   └── financial_metrics_data   ← monthly values: real_value, budget_value, last_year_real_value

  Categories in the DB: 'CA Mobile', 'Data Mobile', 'Mobile Money', 'Parc Mobile',
                        'Capex Consolidés', 'Cash Conso', 'P&L conso', 'Opex Consolidés',
                        'Marge brute Mobile', 'Trafic mobile', 'Indicateurs Mobile'

  ✓ Revenue by category (always use dedup CTE + join via financial_metric):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fc.name AS category, ROUND(SUM(fmd.real_value)::numeric, 0) AS total
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE EXTRACT(YEAR FROM fmd.date) = 2025 AND fmd.real_value IS NOT NULL
    GROUP BY fc.name ORDER BY total DESC;

  ✗ NEVER use cashflow_data for revenue questions — cashflow_data stores treasury cash flow,
    NOT operational revenue. "Category" in a revenue question means financial_categories, not cashflow_categories.

REVENUE BY SEGMENT — revenue_raw_data (DAILY DATA):
  Use revenue_raw_data (NOT financial_metrics_data, NOT monthly_evolution) when the question
  asks about: revenue breakdown by segment, CA Voix vs CA Data vs CA Forfaits, rechargement
  ratio, YoY CA Global growth, segment share of total revenue, monthly revenue aggregation.

  revenue_raw_data has one row per DAY. To get monthly totals, always GROUP BY DATE_TRUNC.
  Columns: date (DATE), ca_global, ca_voix_classique, ca_forfaits_voix, ca_pass_bonus,
           ca_data, moov_sayaa, autres, rechargement, parc_abonnes_global,
           gross_add, churn, net_add, trafic_voix, trafic_data_ko.

  IMPORTANT: monthly_evolution is SPARSE (only 1-2 rows) and stores YoY growth RATES
  (decimal form like -0.20 = -20%), NOT absolute revenue. Do NOT use monthly_evolution
  for revenue values or segment breakdowns.

  ✓ Revenue breakdown by segment with share % for a quarter:
    SELECT DATE_TRUNC('month', date) AS month,
           SUM(ca_global) AS ca_global,
           SUM(ca_voix_classique) AS ca_voix,
           SUM(ca_data) AS ca_data,
           SUM(ca_forfaits_voix) AS ca_forfaits,
           ROUND((SUM(ca_voix_classique) / NULLIF(SUM(ca_global), 0) * 100)::numeric, 2) AS voix_share_pct,
           ROUND((SUM(ca_data)           / NULLIF(SUM(ca_global), 0) * 100)::numeric, 2) AS data_share_pct,
           ROUND((SUM(ca_forfaits_voix)  / NULLIF(SUM(ca_global), 0) * 100)::numeric, 2) AS forfaits_share_pct
    FROM revenue_raw_data
    WHERE EXTRACT(YEAR FROM date) = 2025 AND EXTRACT(MONTH FROM date) BETWEEN 1 AND 3
    GROUP BY DATE_TRUNC('month', date)
    ORDER BY month;

  ✓ Year-over-year CA Global growth per month (2025 vs 2024):
    WITH monthly AS (
      SELECT EXTRACT(YEAR FROM date)::int AS yr, EXTRACT(MONTH FROM date)::int AS mo,
             SUM(ca_global) AS ca_global
      FROM revenue_raw_data GROUP BY yr, mo
    )
    SELECT m25.mo AS month,
           m25.ca_global AS ca_2025,
           m24.ca_global AS ca_2024,
           ROUND(((m25.ca_global - m24.ca_global) / NULLIF(m24.ca_global, 0) * 100)::numeric, 2) AS yoy_growth_pct
    FROM monthly m25
    LEFT JOIN monthly m24 ON m24.yr = 2024 AND m24.mo = m25.mo
    WHERE m25.yr = 2025 ORDER BY m25.mo;

  ✓ Rechargement as % of CA Global per month:
    SELECT EXTRACT(MONTH FROM date)::int AS month,
           SUM(rechargement) AS rechargement,
           SUM(ca_global) AS ca_global,
           ROUND((SUM(rechargement) / NULLIF(SUM(ca_global), 0) * 100)::numeric, 2) AS rechargement_ratio_pct
    FROM revenue_raw_data WHERE EXTRACT(YEAR FROM date) = 2025
    GROUP BY EXTRACT(MONTH FROM date) ORDER BY month;

  ✗ Do NOT use financial_metrics_data for segment breakdown — it does not have ca_voix/ca_data columns.
  ✗ Do NOT use monthly_evolution for revenue values — it stores YoY evolution rates, not amounts.

CAPEX HIERARCHY — for questions about CAPEX suppliers, projects, spend by month:
  capex_projects: id [PK], supplier_name, direction_name, project_title, contract_no
  capex_data:     id [PK], capex_projects_id (FK → capex_projects.id), month, year,
                  equipment, services, additional_costs

  ✓ Monthly trend (total spend per month — NO supplier breakdown):
    SELECT cd.month,
           SUM(COALESCE(cd.equipment,0) + COALESCE(cd.services,0) + COALESCE(cd.additional_costs,0)) AS total_capex
    FROM capex_data cd
    WHERE cd.year = 2025
    GROUP BY cd.month
    ORDER BY cd.month;

  CAPEX COALESCE RULE — applies only when computing total spend (SUM of cost columns):
    equipment, services, additional_costs can be NULL. When summing them:
    ✓ COALESCE(cd.equipment,0) + COALESCE(cd.services,0) + COALESCE(cd.additional_costs,0)
    ✗ cd.equipment + cd.services + cd.additional_costs  ← NULL in any column = NULL total
    Do NOT apply COALESCE for COUNT queries or when not aggregating spend.

    Always qualify capex columns with the table alias cd. — never reference them bare.
    ✗ SUM(additional_costs)        ← ambiguous, will error
    ✓ SUM(cd.additional_costs)     ← correct

  ✓ Total capex PER DIRECTION in a year (group by cp.direction_name):
    SELECT cp.direction_name,
           SUM(COALESCE(cd.equipment,0) + COALESCE(cd.services,0) + COALESCE(cd.additional_costs,0)) AS total_capex
    FROM capex_data cd
    JOIN capex_projects cp ON cp.id = cd.capex_projects_id
    WHERE cd.year = 2025
    GROUP BY cp.direction_name
    ORDER BY total_capex DESC;

  ✓ All projects ranked by total cost:
    SELECT cp.project_title, cp.supplier_name, cp.direction_name,
           SUM(COALESCE(cd.equipment,0) + COALESCE(cd.services,0) + COALESCE(cd.additional_costs,0)) AS total_cost
    FROM capex_data cd
    JOIN capex_projects cp ON cp.id = cd.capex_projects_id
    WHERE cd.year = 2025
    GROUP BY cp.project_title, cp.supplier_name, cp.direction_name
    ORDER BY total_cost DESC;

  ⚠ CAPEX PROJECT GROUPING RULE — CRITICAL:
    When grouping by project, ALWAYS use (cp.project_title, cp.supplier_name, cp.direction_name).
    ✗ NEVER include cp.id in GROUP BY — it splits the same project into multiple rows.
    ✓ GROUP BY cp.project_title, cp.supplier_name, cp.direction_name
    ✗ GROUP BY cp.id, cp.project_title, cp.supplier_name, cp.direction_name  ← extra rows, FORBIDDEN

  ✓ Total additional_costs per direction (use SUM(cd.additional_costs) — no COALESCE needed here):
    SELECT cp.direction_name,
           SUM(cd.additional_costs) AS total_additional_costs
    FROM capex_data cd
    JOIN capex_projects cp ON cp.id = cd.capex_projects_id
    WHERE cd.year = 2025
    GROUP BY cp.direction_name
    ORDER BY total_additional_costs DESC NULLS LAST;

  ✓ Suppliers for a specific month (requires JOIN to capex_projects):
    SELECT cp.supplier_name,
           SUM(COALESCE(cd.equipment,0) + COALESCE(cd.services,0) + COALESCE(cd.additional_costs,0)) AS total_spend
    FROM capex_data cd
    JOIN capex_projects cp ON cp.id = cd.capex_projects_id
    WHERE cd.year = 2025 AND cd.month = 9
    GROUP BY cp.supplier_name
    ORDER BY total_spend DESC;

  ✓ Supplier spend breakdown by cost type (equipment vs services vs additional_costs):
    SELECT SUM(COALESCE(cd.equipment,0))         AS equipment_spend,
           SUM(COALESCE(cd.services,0))          AS services_spend,
           SUM(COALESCE(cd.additional_costs,0))  AS additional_costs_spend,
           SUM(COALESCE(cd.equipment,0)+COALESCE(cd.services,0)+COALESCE(cd.additional_costs,0)) AS total_spend,
           ROUND(SUM(COALESCE(cd.equipment,0)) * 100.0
                 / NULLIF(SUM(COALESCE(cd.equipment,0)+COALESCE(cd.services,0)+COALESCE(cd.additional_costs,0)),0), 2)
                 AS equipment_pct
    FROM capex_data cd
    JOIN capex_projects cp ON cp.id = cd.capex_projects_id
    WHERE UPPER(cp.supplier_name) LIKE '%ERICSSON%';

  SUPPLIER NAME RULES:
    ✓ Always use UPPER(cp.supplier_name) LIKE '%NAME%' for fuzzy supplier matching
      (supplier names have variants: 'ERICSSON AB', 'ERICSSON BENIN', etc.)
    ✗ Never use exact equality cp.supplier_name = 'X' — variants will be missed

  CAPEX COST TYPES (columns in capex_data):
    equipment        = hardware / network infrastructure purchases
    services         = implementation, installation, maintenance services
    additional_costs = other project costs (transport, duties, etc.)

  ✗ NEVER mix supplier columns into a monthly trend query — if the question
    asks "monthly trend" or "per month", GROUP BY cd.month only, no JOIN needed.
  ✗ NEVER reference capex_projects columns without the JOIN above.

  ✓ CAPEX intensity ratio (total CAPEX as % of total revenue) — cross-table via CTE:
    WITH capex_total AS (
      SELECT SUM(COALESCE(equipment,0) + COALESCE(services,0) + COALESCE(additional_costs,0)) AS total_capex
      FROM capex_data WHERE year = 2025
    ),
    revenue_total AS (
      SELECT SUM(ca_global) AS total_revenue
      FROM revenue_raw_data WHERE EXTRACT(YEAR FROM date) = 2025
    )
    SELECT total_capex, total_revenue,
           ROUND(((total_capex / NULLIF(total_revenue, 0)) * 100)::numeric, 2) AS capex_intensity_pct
    FROM capex_total, revenue_total;

  ✓ CAPEX YoY comparison per month (self-join on capex_data, LEFT JOIN for missing 2024 months):
    SELECT cd25.month,
           to_char(to_date(cd25.month::text, 'MM'), 'Month') AS month_name,
           SUM(COALESCE(cd25.equipment,0) + COALESCE(cd25.services,0) + COALESCE(cd25.additional_costs,0)) AS capex_2025,
           COALESCE(SUM(COALESCE(cd24.equipment,0) + COALESCE(cd24.services,0) + COALESCE(cd24.additional_costs,0)), 0) AS capex_2024,
           ROUND((((SUM(COALESCE(cd25.equipment,0)+COALESCE(cd25.services,0)+COALESCE(cd25.additional_costs,0))
                  - COALESCE(SUM(COALESCE(cd24.equipment,0)+COALESCE(cd24.services,0)+COALESCE(cd24.additional_costs,0)),0))
                  / NULLIF(COALESCE(SUM(COALESCE(cd24.equipment,0)+COALESCE(cd24.services,0)+COALESCE(cd24.additional_costs,0)),0), 0)
                 ) * 100)::numeric, 2) AS yoy_change_pct
    FROM capex_data cd25
    LEFT JOIN capex_data cd24 ON cd24.capex_projects_id = cd25.capex_projects_id
      AND cd24.month = cd25.month AND cd24.year = 2024
    WHERE cd25.year = 2025
    GROUP BY cd25.month ORDER BY cd25.month;

  ✓ CAPEX breakdown by type as percentage:
    SELECT
      SUM(cd.equipment)         AS equipment_spend,
      SUM(cd.services)          AS services_spend,
      SUM(cd.additional_costs)  AS additional_costs_spend,
      ROUND(SUM(cd.equipment) * 100.0
            / NULLIF(SUM(cd.equipment + cd.services + cd.additional_costs), 0)::numeric, 2) AS equipment_pct,
      ROUND(SUM(cd.services)  * 100.0
            / NULLIF(SUM(cd.equipment + cd.services + cd.additional_costs), 0)::numeric, 2) AS services_pct
    FROM capex_data cd
    WHERE cd.year = 2025;

CASH FLOW HIERARCHY — use ONLY for questions about flux de trésorerie / treasury / liquidity:
  realised_cashflow → cashflow_sections → cashflow_categories → cashflow_subcategories
  Data in cashflow_data: [year, jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]
  Each row's entity_type is 'section', 'category', or 'subcategory'.

JOIN PATH RULE — CRITICAL:
  financial_metrics_data has TWO foreign keys:
    fmd.financial_metric_id  → financial_metric.id       (ALWAYS populated)
    fmd.financial_type_id    → financial_types.id        (sometimes NULL — do NOT rely on it)

  When filtering by metric name (fm.name) or category name (fc.name), ALWAYS use:
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id   ← join data → metric
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id      ← join metric → type  (NOT fmd.financial_type_id)
    JOIN financial_categories fc ON fc.id = ft.financial_category_id

  ✗ WRONG (fmd.financial_type_id is NULL for many metrics → returns 0 rows):
      JOIN financial_types ft ON ft.id = fmd.financial_type_id
  ✓ CORRECT:
      JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
      JOIN financial_types ft  ON ft.id = fm.financial_type_id

MARGINS / PROFITABILITY QUERIES:
  "Business segments" = financial_categories (CA Mobile, Data Mobile, Mobile Money, P&L conso, etc.)
  There is NO single "margins" column — margins are stored as named metrics inside financial_metrics_data.

  Available margin metrics (financial_metric.name):
    '% CA'                — net margin % of revenue             (category='P&L conso',     type='RESULTAT NET')
    '% Marge Brute/CA'    — MoMo gross margin %                 (category='Mobile Money',   type='Marge Brute (en monnaie locale)')
    'Marge Brute'         — absolute gross margin in FCFA        (category='P&L conso',     type="Chiffre d'affaires")
    'EBITDA'              — EBITDA in FCFA                       (category='P&L conso',     type="Chiffre d'affaires")
    'EBITA'               — EBITA in FCFA                        (category='P&L conso',     type="Chiffre d'affaires")

  ✓ Compare available margin % metrics across segments:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fc.name AS segment, fm.name AS margin_type,
           ROUND(AVG(fmd.real_value)::numeric, 2) AS avg_margin_pct
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fm.name IN ('% CA', '% Marge Brute/CA')
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL
    GROUP BY fc.name, fm.name
    ORDER BY avg_margin_pct DESC;

  ✓ P&L summary (revenue, gross margin, EBITDA, net result) for a year:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fm.name AS kpi,
           ROUND(SUM(fmd.real_value)::numeric, 0) AS total_fcfa
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'P&L conso'
      AND fm.name IN ('Mobile', 'Marge Brute', 'EBITDA', 'RESULTAT NET')
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL
    GROUP BY fm.name
    ORDER BY total_fcfa DESC;

  ✗ Do NOT invent a "margin" column — it does not exist. Always JOIN to financial_metric.name.
  ✗ Do NOT use fc.name = 'Marge brute Mobile' for margin % — that category stores costs, not %.

  ⚠ UNIT WARNING: financial_metrics_data stores values in MILLIONS of FCFA.
    revenue_raw_data stores values in actual FCFA (raw amounts).
    When joining these tables for ratio calculations, convert revenue_raw_data to millions:
    SUM(ca_global) / 1000000.0 AS ca_global_millions
    COLUMN ALIAS UNIT HINT: When selecting financial_metrics_data monetary columns, suffix
    aliases with _m_fcfa so the formatter can display the correct unit:
    ✓ ROUND(SUM(fmd.real_value)::numeric, 0) AS actual_m_fcfa
    ✓ ROUND(SUM(fmd.real_value)::numeric, 0) AS total_opex_m_fcfa
    This applies to all real_value, budget_value, last_year_real_value aggregations.

  ✓ EBITDA margin % (EBITDA / CA Global) — join financial_metrics_data with revenue_raw_data:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    ),
    monthly_ca AS (
      SELECT DATE_TRUNC('month', date) AS month,
             SUM(ca_global) / 1000000.0 AS ca_global_millions
      FROM revenue_raw_data GROUP BY DATE_TRUNC('month', date)
    )
    SELECT
      to_char(e.date, 'Month YYYY')  AS period,
      ROUND(e.real_value::numeric, 2)                                                AS ebitda_millions,
      ROUND(mc.ca_global_millions::numeric, 2)                                       AS ca_global_millions,
      ROUND((e.real_value / NULLIF(mc.ca_global_millions, 0) * 100)::numeric, 2)    AS ebitda_margin_pct
    FROM fmd e
    JOIN financial_metric fm ON fm.id = e.financial_metric_id
    JOIN monthly_ca mc ON DATE_TRUNC('month', e.date) = mc.month
    WHERE fm.name = 'EBITDA'
      AND EXTRACT(YEAR FROM e.date) = 2025
    ORDER BY e.date;

  JOIN KEY: financial_metrics_data → revenue_raw_data (for CA Global):
    DATE_TRUNC('month', fmd.date) = DATE_TRUNC('month', rrd.date) — aggregate rrd by month first.
    ALWAYS divide SUM(revenue_raw_data.ca_global) by 1,000,000 to convert to millions before dividing.
  Use this join whenever computing a ratio of a financial_metrics_data metric against CA Global.

CHURN RATE QUERIES — for questions about churn, taux de churn, attrition, résiliation:
  ⚠ There is NO metric called 'churn' or 'taux de churn' directly in financial_metric.
  Churn must be COMPUTED from prepaid subscriber counts in category 'Parc Mobile':
    fm.name = 'Parc prépayé actif début Période'  ← subscribers at start of month
    fm.name = 'Parc prépayé actif fin de période'  ← subscribers at end of month
  Monthly churn rate = (début - fin) / début * 100

  ✓ Monthly churn rate 2025 vs prior year (YoY):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    ),
    debut AS (
      SELECT fmd.date,
             fmd.real_value          AS debut_2025,
             fmd.last_year_real_value AS debut_2024
      FROM fmd
      JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
      WHERE fm.name = 'Parc prépayé actif début Période'
        AND EXTRACT(YEAR FROM fmd.date) = 2025
        AND fmd.real_value IS NOT NULL
    ),
    fin AS (
      SELECT fmd.date,
             fmd.real_value          AS fin_2025,
             fmd.last_year_real_value AS fin_2024
      FROM fmd
      JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
      WHERE fm.name = 'Parc prépayé actif fin de période'
        AND EXTRACT(YEAR FROM fmd.date) = 2025
        AND fmd.real_value IS NOT NULL
    )
    SELECT EXTRACT(MONTH FROM d.date)::int AS month,
           ROUND(((d.debut_2025 - f.fin_2025) * 100.0 / NULLIF(d.debut_2025, 0))::numeric, 2) AS churn_pct_2025,
           ROUND(((d.debut_2024 - f.fin_2024) * 100.0 / NULLIF(d.debut_2024, 0))::numeric, 2) AS churn_pct_2024
    FROM debut d
    JOIN fin f ON f.date = d.date
    ORDER BY month;

  ✗ Do NOT search for metric names like 'churn', 'taux de churn', 'attrition' — they don't exist.
  ✗ Do NOT query financial_metric with ILIKE '%churn%' — returns nothing.

ROAMING REVENUE QUERIES — for questions about roaming, itinérance, roaming in/out:
  ⚠ IMPORTANT: The DB does NOT have metrics called 'Roaming', 'Revenu Roaming', or 'Sortant voix roaming'.
  The ONLY roaming metrics in financial_metric are:
    fm.name = 'Dont Roaming National'  → roaming-IN revenue  (category='CA Mobile', type='Roaming in')
    fm.name = 'Roaming out'            → roaming-OUT costs   (category='Marge brute Mobile') — stored as negative values

  ✓ Monthly roaming-in revenue trend for 2025 with YoY comparison:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT EXTRACT(MONTH FROM fmd.date)::int AS month,
           ROUND(SUM(fmd.real_value)::numeric, 2)            AS roaming_in_revenue,
           ROUND(SUM(fmd.last_year_real_value)::numeric, 2)  AS prior_year,
           ROUND(SUM(fmd.budget_value)::numeric, 2)          AS budget
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    WHERE fm.name = 'Dont Roaming National'
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL
    GROUP BY EXTRACT(MONTH FROM fmd.date)
    ORDER BY month;

  ✓ Roaming-out cost trend (costs are negative — show as positive for readability):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT EXTRACT(MONTH FROM fmd.date)::int AS month,
           ROUND(ABS(SUM(fmd.real_value))::numeric, 2) AS roaming_out_cost
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    WHERE fm.name = 'Roaming out'
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL
    GROUP BY EXTRACT(MONTH FROM fmd.date)
    ORDER BY month;

  ✗ NEVER search for: 'Roaming', 'Revenu Roaming', 'roaming revenue', 'Sortant voix roaming' — these names do not exist.

MOOV MONEY TRANSACTION QUERIES — for questions about MoMo transaction volume, count, amount, activity:
  Use moov_money_data directly — NOT financial_metrics_data.
  Columns: transaction_date (timestamp), msisdn, transaction_type, amount (bigint), amount_status,
           unique_trans_status, within_7_days_status.

  ✓ Total transaction volume (sum of amounts) in 2025:
    SELECT SUM(amount) AS total_volume
    FROM moov_money_data
    WHERE EXTRACT(YEAR FROM transaction_date) = 2025;

  ✓ Monthly transaction count in 2025:
    SELECT EXTRACT(MONTH FROM transaction_date)::int AS month,
           COUNT(*) AS transaction_count,
           SUM(amount) AS total_amount
    FROM moov_money_data
    WHERE EXTRACT(YEAR FROM transaction_date) = 2025
    GROUP BY EXTRACT(MONTH FROM transaction_date)
    ORDER BY month;

  ✗ Do NOT use financial_metrics_data for MoMo transaction counts or amounts.
  ✗ The date column is transaction_date (timestamp) — NOT date, NOT created_at.

SUBSCRIBER & TRAFFIC METRICS — for questions about subscribers, churn, ARPU, MoU, traffic, parc:
  Categories: 'Parc Mobile', 'Trafic mobile', 'Indicateurs Mobile', 'CA Mobile', 'Data Mobile'
  All stored in financial_metrics_data — same JOIN path as revenue queries.

  ✓ Monthly subscriber base trend:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT EXTRACT(MONTH FROM fmd.date)::int AS month,
           ROUND(SUM(fmd.real_value)::numeric, 0) AS total_subscribers
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'Parc Mobile'
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL
    GROUP BY EXTRACT(MONTH FROM fmd.date)
    ORDER BY month;

  ✓ ARPU monthly trend (fuzzy match on metric name):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT EXTRACT(MONTH FROM fmd.date)::int AS month,
           ROUND(AVG(fmd.real_value)::numeric, 2) AS avg_arpu
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    WHERE UPPER(fm.name) LIKE '%ARPU%'
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL
    GROUP BY EXTRACT(MONTH FROM fmd.date) ORDER BY month;

  ✓ List all available metric names in a category (for exploration):
    SELECT DISTINCT fm.name AS metric, ft.name AS type
    FROM financial_metric fm
    JOIN financial_types ft ON ft.id = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'Indicateurs Mobile'
    ORDER BY ft.name, fm.name;

PRODUCT / SERVICE NAME LOOKUP — for questions about a specific named product, service, or line item
  (e.g. "Clé Internet Mobile", "Data Bundle", "Forfait Voix", "3G", "4G", "Prépayé"):

  ⚠ NEVER assume the metric name is spelled exactly as in the user's question.
  ⚠ NEVER query capex_projects for sales, revenue, or product performance data.
  ALL product/service revenue metrics are stored in financial_metrics_data + financial_metric.

  STRATEGY — always use ILIKE with a broad pattern so spelling variants are matched:
    fm.name ILIKE '%Clé Internet%'      -- matches "Clé Internet Mobile", "Clé Internet" etc.
    fm.name ILIKE '%3G%'               -- matches any 3G metric
    fm.name ILIKE '%Forfait%'          -- matches any forfait/bundle metric

  ✓ Sales/revenue trend for a named product (e.g. "Clé Internet Mobile") in 2025 with YoY:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT EXTRACT(MONTH FROM fmd.date)::int AS month,
           fm.name                                                    AS metric_name,
           ROUND(SUM(fmd.real_value)::numeric, 0)                    AS actual_2025,
           ROUND(SUM(fmd.last_year_real_value)::numeric, 0)          AS prior_2024,
           ROUND((SUM(fmd.real_value) - SUM(fmd.last_year_real_value)) * 100.0
                 / NULLIF(ABS(SUM(fmd.last_year_real_value)), 0), 1) AS yoy_pct
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    WHERE fm.name ILIKE '%Clé Internet%'
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL
    GROUP BY EXTRACT(MONTH FROM fmd.date), fm.name
    ORDER BY month;

  ✓ If unsure of the exact metric name, first explore what names exist:
    SELECT DISTINCT fm.name, ft.name AS type, fc.name AS category
    FROM financial_metric fm
    JOIN financial_types ft ON ft.id = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fm.name ILIKE '%Internet%' OR fm.name ILIKE '%Clé%' OR fm.name ILIKE '%Dongle%'
    ORDER BY fc.name, fm.name;

  ✗ NEVER query capex_projects for sales or revenue data — it has no sales columns.
  ✗ NEVER use fm.name = 'exact name' without ILIKE — the spelling in the DB may differ.

BUDGET vs ACTUAL VARIANCE — for questions about budget, forecast, écart, vs target, over/under:
  Use: real_value (actual), budget_value (plan/budget).
  variance_pct formula — use CASE WHEN to handle zero budget cleanly:
    CASE WHEN budget_value != 0
         THEN ROUND((((real_value - budget_value) / budget_value) * 100)::numeric, 2)
         ELSE NULL
    END AS variance_pct
  ✗ Do NOT use ABS(budget_value) in the denominator for simple variance — it changes the sign direction.

  ✓ Variance by metric for a year (over/under budget):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fm.name AS metric,
           ROUND(SUM(fmd.real_value)::numeric, 0)      AS actual,
           ROUND(SUM(fmd.budget_value)::numeric, 0)    AS budget,
           ROUND((SUM(fmd.real_value) - SUM(fmd.budget_value))::numeric, 0) AS variance,
           ROUND((SUM(fmd.real_value) - SUM(fmd.budget_value)) * 100.0
                 / NULLIF(ABS(SUM(fmd.budget_value)), 0), 2)               AS variance_pct
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'P&L conso'
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL AND fmd.budget_value IS NOT NULL
    GROUP BY fm.name ORDER BY variance_pct DESC;

  ✓ Monthly actual vs budget for a single metric:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT EXTRACT(MONTH FROM fmd.date)::int AS month,
           ROUND(SUM(fmd.real_value)::numeric, 0)   AS actual,
           ROUND(SUM(fmd.budget_value)::numeric, 0) AS budget
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    WHERE fm.name = 'EBITDA'
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL AND fmd.budget_value IS NOT NULL
    GROUP BY EXTRACT(MONTH FROM fmd.date) ORDER BY month;

QUARTERLY VARIANCE ANALYSIS — for questions about budget overrun, underperformance, écart budgétaire,
  "which metrics exceeded/missed budget", "largest overruns", "budget tracking", "vs budget":

  ⚠ ALWAYS include quarterly breakdown alongside the annual total.
  This is MANDATORY — it reveals whether performance is improving or worsening across the year.
  Columns q1_variance … q4_variance are absolute FCFA variances (positive = overrun, negative = miss).

  ✓ Top budget overrunners with quarterly breakdown:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT
      fc.name AS category,
      fm.name AS metric,
      ROUND(SUM(fmd.real_value)::numeric, 0)                                AS annual_actual,
      ROUND(SUM(fmd.budget_value)::numeric, 0)                              AS annual_budget,
      ROUND((SUM(fmd.real_value) - SUM(fmd.budget_value))::numeric, 0)     AS total_variance,
      ROUND((SUM(fmd.real_value) - SUM(fmd.budget_value)) * 100.0
            / NULLIF(ABS(SUM(fmd.budget_value)), 0), 1)                     AS variance_pct,
      ROUND(SUM(CASE WHEN EXTRACT(MONTH FROM fmd.date) BETWEEN 1  AND 3
                     THEN fmd.real_value - fmd.budget_value END)::numeric, 0) AS q1_variance,
      ROUND(SUM(CASE WHEN EXTRACT(MONTH FROM fmd.date) BETWEEN 4  AND 6
                     THEN fmd.real_value - fmd.budget_value END)::numeric, 0) AS q2_variance,
      ROUND(SUM(CASE WHEN EXTRACT(MONTH FROM fmd.date) BETWEEN 7  AND 9
                     THEN fmd.real_value - fmd.budget_value END)::numeric, 0) AS q3_variance,
      ROUND(SUM(CASE WHEN EXTRACT(MONTH FROM fmd.date) BETWEEN 10 AND 12
                     THEN fmd.real_value - fmd.budget_value END)::numeric, 0) AS q4_variance
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL AND fmd.budget_value IS NOT NULL
    GROUP BY fc.name, fm.name
    HAVING ABS(SUM(fmd.real_value) - SUM(fmd.budget_value)) > 0
    ORDER BY ABS(SUM(fmd.real_value) - SUM(fmd.budget_value)) DESC
    LIMIT 15;

  ✓ For underperformers only (missed budget), change HAVING to:
      HAVING SUM(fmd.real_value) < SUM(fmd.budget_value)
    and ORDER BY total_variance ASC

  ✓ For overrunners only (exceeded budget), change HAVING to:
      HAVING SUM(fmd.real_value) > SUM(fmd.budget_value)
    and ORDER BY total_variance DESC

  ✗ NEVER return just an annual total for variance/overrun questions — always include q1…q4 columns.
  ✗ NEVER omit q1_variance … q4_variance — the format layer uses them for trend analysis.

UNDERPERFORMING REVENUE LINES — for questions like "which lines are underperforming vs last year",
  "quelles lignes sous-performent", "revenue below last year", "en baisse par rapport à N-1":
  Use HAVING SUM(real_value) < SUM(last_year_real_value) to filter only declining lines.

  ⚠ MANDATORY: You MUST include ALL FOUR categories in the WHERE clause:
    'CA Mobile', 'Data Mobile', 'Mobile Money', 'P&L conso'
  ✗ NEVER remove 'P&L conso' — it contains EBITA and EBITDA which are known underperformers.
  ✗ NEVER use LIMIT smaller than 20 — there are typically 10–15 underperforming lines.

  Expected results include metrics such as:
    EBITA (P&L conso), EBITDA (P&L conso), Prépayé (CA Mobile),
    Achat de forfaits (CA Mobile), 3G (Data Mobile), 4G (Data Mobile),
    Recharge Mobile (CA Mobile), Transfert d'argent international (Mobile Money).

  ✓ Revenue lines underperforming vs prior year (ranked worst first):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fc.name AS category,
           fm.name AS revenue_line,
           ROUND(SUM(fmd.real_value)::numeric, 0)           AS actual_2025,
           ROUND(SUM(fmd.last_year_real_value)::numeric, 0) AS prior_2024,
           ROUND(((SUM(fmd.real_value) - SUM(fmd.last_year_real_value)) * 100.0
                 / NULLIF(ABS(SUM(fmd.last_year_real_value)), 0))::numeric, 1) AS yoy_pct
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name IN ('CA Mobile', 'Data Mobile', 'Mobile Money', 'P&L conso')
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL AND fmd.last_year_real_value IS NOT NULL
    GROUP BY fc.name, fm.name
    HAVING SUM(fmd.real_value) < SUM(fmd.last_year_real_value)
    ORDER BY yoy_pct ASC
    LIMIT 20;

YEAR-ON-YEAR (YoY) COMPARISON — for questions about growth, YoY, vs last year, evolution, year-over-year:
  Use last_year_real_value for the prior-year figure. Do NOT query two separate years.

  ✓ YoY comparison for key P&L metrics:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fm.name AS metric,
           ROUND(SUM(fmd.last_year_real_value)::numeric, 0) AS prior_year,
           ROUND(SUM(fmd.real_value)::numeric, 0)           AS current_year,
           ROUND((SUM(fmd.real_value) - SUM(fmd.last_year_real_value)) * 100.0
                 / NULLIF(ABS(SUM(fmd.last_year_real_value)), 0), 2)       AS yoy_pct
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'P&L conso'
      AND fm.name IN ('Mobile', 'EBITDA', 'RESULTAT NET')
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL AND fmd.last_year_real_value IS NOT NULL
    GROUP BY fm.name ORDER BY current_year DESC;

  ✓ YoY monthly trend for a single metric:
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT EXTRACT(MONTH FROM fmd.date)::int AS month,
           ROUND(SUM(fmd.last_year_real_value)::numeric, 0) AS prior_year,
           ROUND(SUM(fmd.real_value)::numeric, 0)           AS current_year
    FROM fmd
    JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
    WHERE UPPER(fm.name) LIKE '%EBITDA%'
      AND EXTRACT(YEAR FROM fmd.date) = 2025
      AND fmd.real_value IS NOT NULL AND fmd.last_year_real_value IS NOT NULL
    GROUP BY EXTRACT(MONTH FROM fmd.date) ORDER BY month;

OPEX QUERIES — for questions about operating expenses, charges, coûts opérationnels:
  Category: 'Opex Consolidés'. Same JOIN path as revenue.

  ⚠ QUERY SELECTION RULE — choose the pattern based on what the question asks:
    • Contains "monthly trend", "month by month", "evolution mensuelle", "par mois"
      AND no specific month named → MONTHLY TREND (GROUP BY month, no ft.name)
    • Contains "breakdown", "by type", "by category", "par type", "which types"
      AND a specific month named (e.g. "May 2024", "March", "Q1") → SINGLE MONTH BREAKDOWN
      (GROUP BY ft.name WITH month + year filter) — returns one row per opex type
    • Contains "breakdown", "by type", "which categories" AND no specific month
      → YEARLY BREAKDOWN (GROUP BY ft.name, full year) — returns one row per opex type
    • Simple "what is opex for [month]?" with no "breakdown" keyword
      → also use SINGLE MONTH BREAKDOWN (GROUP BY ft.name + MONTH filter)
    ✗ NEVER return a single-row ungrouped total — ALWAYS group by ft.name or by month.
    ✗ NEVER apply GROUP BY ft.name to a monthly trend question — it produces wrong results.
    ✓ A breakdown query MUST return multiple rows (3–8 rows, one per opex type).
      If your query has no GROUP BY ft.name, it is wrong for any breakdown question.

  ✓ Monthly OpEx trend — use for "monthly trend", "month by month", "evolution mensuelle" (e.g. "opex 2024 monthly trend"):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT EXTRACT(MONTH FROM fmd.date)::int        AS month,
           ROUND(SUM(fmd.real_value)::numeric, 0)         AS total_opex_m_fcfa,
           ROUND(SUM(fmd.budget_value)::numeric, 0)        AS budget_m_fcfa,
           ROUND(SUM(fmd.last_year_real_value)::numeric, 0) AS prior_year_m_fcfa
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'Opex Consolidés'
      AND EXTRACT(YEAR FROM fmd.date) = 2024
      AND fmd.real_value IS NOT NULL
    GROUP BY EXTRACT(MONTH FROM fmd.date)
    ORDER BY month;

  ✓ OpEx breakdown by type for a year — use for "breakdown", "by category", "which types", or "total opex for the year":
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT ft.name AS opex_type,
           ROUND(SUM(fmd.real_value)::numeric, 0)            AS actual_m_fcfa,
           ROUND(SUM(fmd.budget_value)::numeric, 0)          AS budget_m_fcfa,
           ROUND(SUM(fmd.last_year_real_value)::numeric, 0)  AS prior_year_m_fcfa,
           ROUND((SUM(fmd.real_value) - SUM(fmd.budget_value))::numeric, 0) AS variance
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'Opex Consolidés'
      AND EXTRACT(YEAR FROM fmd.date) = 2024
      AND fmd.real_value IS NOT NULL
    GROUP BY ft.name ORDER BY actual_m_fcfa DESC;

  ✓ Top N OpEx drivers (individual metrics) for a year — use this for "top drivers", "largest costs", "biggest expenses":
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fm.name AS driver,
           ft.name AS opex_type,
           ROUND(SUM(fmd.real_value)::numeric, 0)   AS actual,
           ROUND(SUM(fmd.budget_value)::numeric, 0) AS budget,
           ROUND((SUM(fmd.real_value) - SUM(fmd.budget_value))::numeric, 0) AS variance
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'Opex Consolidés'
      AND EXTRACT(YEAR FROM fmd.date) = 2024
      AND fmd.real_value IS NOT NULL
    GROUP BY fm.name, ft.name
    ORDER BY actual DESC
    LIMIT 5;

  ✓ OpEx breakdown by type for a SINGLE MONTH — use for "breakdown for May 2024", "opex May 2024",
    "what was opex for March 2025", or any question naming a specific month (returns 3–8 rows, one per type):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT ft.name AS opex_type,
           ROUND(SUM(fmd.real_value)::numeric, 0)              AS actual_m_fcfa,
           ROUND(SUM(fmd.budget_value)::numeric, 0)            AS budget_m_fcfa,
           ROUND(SUM(fmd.last_year_real_value)::numeric, 0)    AS prior_year_m_fcfa,
           ROUND((SUM(fmd.real_value) - SUM(fmd.budget_value))::numeric, 0) AS variance
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'Opex Consolidés'
      AND EXTRACT(YEAR FROM fmd.date) = 2024
      AND EXTRACT(MONTH FROM fmd.date) = 5
      AND fmd.real_value IS NOT NULL
    GROUP BY ft.name
    ORDER BY actual_m_fcfa DESC;

BFR / WORKING CAPITAL QUERIES — for questions about BFR, "Besoin en Fonds de Roulement", working capital,
  "variation de BFR", "BFR opérationnel", or "sub-components of BFR":

  ⚠ CRITICAL: BFR is NOT in cashflow_data. It lives in financial_metrics_data.
    metric name: 'Variation de BFR opérationnel (+/-)'
    financial_category: 'Cash Conso'
    financial_type: 'CFFO'
    Use the standard dedup CTE — NEVER query cashflow_data for BFR.

  Related CFFO sub-components (other metrics under type='CFFO', category='Cash Conso'):
    'Neutralisation de la var. de provisions incluses dans l''Ebitda (-)'
    'Investissements bruts (y compris acquisitions de société) (-)'
    'Investissements nets (Capex brutes - cession d''immo.) (-)'
    'Cession d''immobilisations'
    'Dividendes reçus des participations non consolidées (+)'
    'Produit de cession des immobilisations corporelles et incorporelles (+)'
    'Plan de restructuration'

  ✓ BFR and CFFO sub-components for a specific quarter/year (e.g. Q2 2024):
    WITH fmd AS (
      SELECT DISTINCT ON (financial_metric_id, date) *
      FROM financial_metrics_data
      ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
    )
    SELECT fm.name AS metric,
           ROUND(SUM(fmd.real_value)::numeric, 0)          AS actual,
           ROUND(SUM(fmd.budget_value)::numeric, 0)        AS budget,
           ROUND((SUM(fmd.real_value) - SUM(fmd.budget_value))::numeric, 0) AS variance
    FROM fmd
    JOIN financial_metric fm  ON fm.id  = fmd.financial_metric_id
    JOIN financial_types ft   ON ft.id  = fm.financial_type_id
    JOIN financial_categories fc ON fc.id = ft.financial_category_id
    WHERE fc.name = 'Cash Conso'
      AND ft.name = 'CFFO'
      AND EXTRACT(YEAR FROM fmd.date) = 2024
      AND EXTRACT(MONTH FROM fmd.date) BETWEEN 4 AND 6   -- Q2: Apr–Jun
      AND fmd.real_value IS NOT NULL
    GROUP BY fm.name
    ORDER BY ABS(SUM(fmd.real_value) - SUM(fmd.budget_value)) DESC NULLS LAST;

  Quarter month ranges: Q1=1–3, Q2=4–6, Q3=7–9, Q4=10–12
  ✗ NEVER JOIN cashflow_data or cashflow_sections for BFR questions.
  ✗ NEVER filter WHERE fm.name = 'Variation de BFR...' alone — include all CFFO metrics to show sub-components.

CASH FLOW QUERIES — for questions about cash flow, trésorerie, FCF, liquidity, flux, cash position:

  UNIT: cashflow_data values are in MILLIONS of FCFA (M CFA).

  VERSION FILTER (CRITICAL for 2025 and 2026 data):
    The table stores multiple upload versions per year. ALWAYS filter with:
      AND cd.version_id = (SELECT MAX(version_id) FROM cashflow_data WHERE year = <year>)
    For year 2024: version_id IS NULL — use: AND cd.version_id IS NULL

  CORRUPTED VALUES: Some monthly cells for 2025 contain near-zero floats (7.21e-321) instead of NULL.
    Always wrap monthly columns with: NULLIF(CASE WHEN ABS(COALESCE(cd.jan,0)) < 0.01 THEN NULL ELSE cd.jan END, 0)
    Shorthand helper: use CASE WHEN ABS(COALESCE(col,0)) < 0.01 THEN NULL ELSE col END for each month.

  CASH FLOW LINES (tbg_key reference — use in WHERE cs.tbg_key = '...'):
    RL01  (A) TOTAL ENCAISSEMENTS               — total cash inflows
    RL03  (B) TOTAL DECAISSEMENTS                — total cash outflows
    RL05  (C) FLUX MENSUEL GENERE PAR EXPLOITAT  — operating cash flow (CFFO)
    RL07  (D) FLUX MENSUEL GENERE PAR H. EXPLOITAT — investment/non-operating cash flow
    RL09  (E) FLUX MENSUEL                       — financing cash flow
    RL11  FLUX NET MENSUEL (C+D+E)               — NET CASH FLOW (best proxy for FCF)
    RL13  CASH & CASH EQUIVALENT                 — cash position (balance)
    RL14  DETTE BRUTE                            — gross debt

  entity_type: always use 'section' to avoid double-counting categories and subcategories.

  ✓ Net cash flow (FCF proxy) for a year — works for 2025:
    WITH latest AS (SELECT MAX(version_id) AS vid FROM cashflow_data WHERE year = 2025)
    SELECT cs.name AS line_item,
           ROUND(cd.current_year_total::numeric, 0) AS annual_total_m_cfa
    FROM cashflow_data cd
    JOIN cashflow_sections cs ON cs.id = cd.entity_id
    JOIN latest ON cd.version_id = latest.vid
    WHERE cd.entity_type = 'section'
      AND cd.year = 2025
      AND cs.tbg_key IN ('RL01', 'RL03', 'RL05', 'RL11')
    ORDER BY cs.sequence_id;

  ✓ Monthly cash position (CASH & CASH EQUIVALENT) for 2024:
    SELECT cs.name AS line_item,
           cd.jan, cd.feb, cd.mar, cd.apr, cd.may, cd.jun,
           cd.jul, cd.aug, cd.sep, cd.oct, cd.nov, cd.dec
    FROM cashflow_data cd
    JOIN cashflow_sections cs ON cs.id = cd.entity_id
    WHERE cd.entity_type = 'section'
      AND cd.year = 2024
      AND cd.version_id IS NULL
      AND cs.tbg_key = 'RL13';

  ✓ Q4 operating cash flow for 2025:
    WITH latest AS (SELECT MAX(version_id) AS vid FROM cashflow_data WHERE year = 2025)
    SELECT cs.name AS line_item,
           CASE WHEN ABS(COALESCE(cd.oct,0)) < 0.01 THEN NULL ELSE ROUND(cd.oct::numeric,0) END AS oct,
           CASE WHEN ABS(COALESCE(cd.nov,0)) < 0.01 THEN NULL ELSE ROUND(cd.nov::numeric,0) END AS nov,
           CASE WHEN ABS(COALESCE(cd.dec,0)) < 0.01 THEN NULL ELSE ROUND(cd.dec::numeric,0) END AS dec,
           ROUND((COALESCE(CASE WHEN ABS(COALESCE(cd.oct,0)) < 0.01 THEN NULL ELSE cd.oct END, 0)
                + COALESCE(CASE WHEN ABS(COALESCE(cd.nov,0)) < 0.01 THEN NULL ELSE cd.nov END, 0)
                + COALESCE(CASE WHEN ABS(COALESCE(cd.dec,0)) < 0.01 THEN NULL ELSE cd.dec END, 0))::numeric, 0) AS q4_total
    FROM cashflow_data cd
    JOIN cashflow_sections cs ON cs.id = cd.entity_id
    JOIN latest ON cd.version_id = latest.vid
    WHERE cd.entity_type = 'section'
      AND cd.year = 2025
      AND cs.tbg_key = 'RL05';

  ✓ Cash inflows vs outflows comparison for 2025:
    WITH latest AS (SELECT MAX(version_id) AS vid FROM cashflow_data WHERE year = 2025)
    SELECT cs.tbg_key AS code, cs.name AS line_item,
           ROUND(cd.current_year_total::numeric, 0) AS total_m_cfa
    FROM cashflow_data cd
    JOIN cashflow_sections cs ON cs.id = cd.entity_id
    JOIN latest ON cd.version_id = latest.vid
    WHERE cd.entity_type = 'section'
      AND cd.year = 2025
      AND cs.tbg_key IN ('RL01', 'RL03')
    ORDER BY cs.sequence_id;

  ✓ Which months had negative cashflow in 2025 (ALWAYS use this exact pattern):
    WITH latest AS (SELECT MAX(version_id) AS vid FROM cashflow_data WHERE year = 2025)
    SELECT month_name AS month FROM (
        SELECT 'January'   AS month_name, 1  AS mo, cd.jan  AS val FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'February',  2,  cd.feb FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'March',     3,  cd.mar FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'April',     4,  cd.apr FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'May',       5,  cd.may FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'June',      6,  cd.jun FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'July',      7,  cd.jul FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'August',    8,  cd.aug FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'September', 9,  cd.sep FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'October',   10, cd.oct FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'November',  11, cd.nov FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 'December',  12, cd.dec FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
    ) m WHERE val IS NOT NULL AND val < 0 ORDER BY mo;

  ✓ Running cumulative cashflow by month for 2025 (ALWAYS use this exact pattern):
    WITH latest AS (SELECT MAX(version_id) AS vid FROM cashflow_data WHERE year = 2025),
    monthly AS (
        SELECT 1 AS mo, SUM(cd.jan) AS val FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 2,  SUM(cd.feb) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 3,  SUM(cd.mar) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 4,  SUM(cd.apr) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 5,  SUM(cd.may) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 6,  SUM(cd.jun) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 7,  SUM(cd.jul) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 8,  SUM(cd.aug) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 9,  SUM(cd.sep) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 10, SUM(cd.oct) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 11, SUM(cd.nov) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
        UNION ALL SELECT 12, SUM(cd.dec) FROM cashflow_data cd JOIN cashflow_sections cs ON cs.id = cd.entity_id JOIN latest ON cd.version_id = latest.vid WHERE cd.year = 2025 AND cd.entity_type = 'section' AND cs.tbg_key = 'RL11'
    )
    SELECT mo AS month_num, val AS monthly_cashflow,
           SUM(val) OVER (ORDER BY mo ROWS UNBOUNDED PRECEDING) AS cumulative_cashflow
    FROM monthly ORDER BY mo;

  ✗ Do NOT use EXTRACT() or date columns on cashflow_data — they don't exist.
  ✗ NEVER reference tbg_key on cashflow_data — tbg_key is a column on cashflow_sections only; always JOIN cashflow_sections to use it.
  ✗ Do NOT SUM across multiple rows without version_id filter — you will get inflated totals.
  ✗ Do NOT use cashflow_data for revenue questions — it is treasury/liquidity only.
  ✗ FCF is not a named column — use RL11 (Net Cash Flow) as the best available proxy.

FOLLOW-UP / CONTINUATION QUERIES:
  When the conversation history contains a previous SQL query, USE IT AS THE BASE.
  Preserve the FROM / JOIN / WHERE structure; only change what the follow-up asks for.
  • "now show Q4"            → change WHERE month filter to IN (10,11,12)
  • "compare with last year" → add last_year_real_value to SELECT
  • "break down by category" → add JOIN financial_categories, GROUP BY fc.name
  • "top 5 only"             → add ORDER BY <metric> DESC LIMIT 5
  • "filter for EBITDA"      → add WHERE fm.name = 'EBITDA' to existing query
  Do NOT start from scratch when the previous SQL is a valid starting point.

USE ONLY the tables and columns listed below.
Do NOT invent table names or column names.

{schema}
"""


def _make_write_sql_node(llm: ChatOllama):
    def write_sql(state: DbPipelineState) -> dict:
        retry  = state.get("retry_count", 0)
        schema = state.get("retrieved_schema", "(no schema)")
        system = _WRITER_SYSTEM.format(schema=schema)

        messages = [SystemMessage(content=system)]
        if state.get("history"):
            messages.append(HumanMessage(
                content=f"Conversation history:\n{state['history']}"
            ))
        messages.append(HumanMessage(content=state["question"]))

        log.info("[attempt %d/%d] Writer invoked", retry + 1, _MAX_RETRIES + 1)
        t0 = time.monotonic()
        response = llm.invoke_streaming(messages)
        elapsed  = time.monotonic() - t0

        # Log raw LLM output for debugging
        raw_output = response.content
        log.debug("[attempt %d/%d] Raw LLM output: %s", retry + 1, _MAX_RETRIES + 1, raw_output[:300])

        sql, err = _extract_sql(raw_output)
        if err:
            log.warning("Writer output rejected in %.1fs: %s", elapsed, err)
            log.debug("Raw output that failed extraction: %s", raw_output[:500])
            return {
                "sql": "", "syntax_error": err,
                "table_error": "", "semantic_error": "", "critic_feedback": "",
            }

        log.info(
            "[attempt %d/%d] Writer OK in %.1fs:\n%s",
            retry + 1, _MAX_RETRIES + 1, elapsed,
            textwrap.indent(sql, "    "),
        )
        return {
            "sql": sql,
            "syntax_error": "", "table_error": "", "semantic_error": "",
            "critic_feedback": "",
        }

    return write_sql


# ---------------------------------------------------------------------------
# Node 2 — sqlglot syntax validation (no LLM, no DB)
# ---------------------------------------------------------------------------

# All AST node types that can mutate or destroy data.
_DESTRUCTIVE_TYPES = (
    sexp.Delete,
    sexp.Insert,
    sexp.Update,
    sexp.Drop,
    sexp.TruncateTable,
    sexp.Alter,
    sexp.Create,
    sexp.Command,   # catches raw COPY, VACUUM, SET, etc.
)

# Regex safety net used in execute_sql as a last-resort guard.
_DESTRUCTIVE_RE = re.compile(
    r"\b(DELETE|INSERT|UPDATE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE|GRANT|REVOKE|COPY|VACUUM|LOCK)\b",
    re.IGNORECASE,
)


def validate_syntax(state: DbPipelineState) -> dict:
    sql = state.get("sql", "").strip()
    if not sql:
        err = "No SQL was generated."
        log.warning("validate_syntax FAIL: %s", err)
        return {"syntax_error": err}
    try:
        stmts = sqlglot.parse(sql, dialect="postgres", error_level=sqlglot.ErrorLevel.RAISE)
        if not stmts:
            err = "Could not parse the SQL statement."
            log.warning("validate_syntax FAIL: %s", err)
            return {"syntax_error": err}
        stmt = stmts[0]

        # Layer 1 — top-level statement must be SELECT or WITH…SELECT
        if not isinstance(stmt, (sexp.Select, sexp.With)):
            err = f"Only SELECT/WITH allowed. Got: {type(stmt).__name__}"
            log.warning("validate_syntax FAIL: %s", err)
            return {"syntax_error": err}

        # Layer 2 — walk every AST node; reject any DML/DDL hiding inside CTEs
        for node in stmt.walk():
            if isinstance(node, _DESTRUCTIVE_TYPES):
                err = f"Destructive statement forbidden inside query: {type(node).__name__}"
                log.warning("validate_syntax FAIL: %s", err)
                return {"syntax_error": err}

        log.info("validate_syntax PASS")
        return {"syntax_error": ""}
    except sqlglot.errors.ParseError as exc:
        err = str(exc).split("\n")[0]
        log.warning("validate_syntax FAIL: %s", err)
        return {"syntax_error": err}


# ---------------------------------------------------------------------------
# Node 3 — Table whitelist validation (no LLM, no DB)
# ---------------------------------------------------------------------------

def _referenced_tables(sql: str) -> set[str]:
    """Return all real table names in SQL, excluding CTE aliases."""
    try:
        stmt = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return set()
    cte_aliases = {cte.alias.lower() for cte in stmt.find_all(sexp.CTE)}
    return {
        t.name.lower()
        for t in stmt.find_all(sexp.Table)
        if t.name and t.name.lower() not in cte_aliases
    }


def validate_tables(state: DbPipelineState) -> dict:
    if state.get("syntax_error"):
        return {"table_error": ""}      # wait for syntax to pass first

    allowed = {t.lower() for t in state.get("allowed_tables", [])}
    if not allowed:
        log.warning("validate_tables: no whitelist set — skipping check")
        return {"table_error": ""}

    sql = state.get("sql", "").strip()
    if not sql:
        return {"table_error": "No SQL to validate."}

    referenced = _referenced_tables(sql)
    disallowed  = referenced - allowed
    if disallowed:
        err = (
            f"Query references unknown table(s): {', '.join(sorted(disallowed))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
        log.warning("validate_tables FAIL: %s", err)
        return {"table_error": err}

    log.info("validate_tables PASS (referenced: %s)", sorted(referenced))
    return {"table_error": ""}


# ---------------------------------------------------------------------------
# Node 4 — EXPLAIN semantic validation (no LLM, checks real DB names)
# ---------------------------------------------------------------------------

def _get_column_facts(sql: str) -> str:
    """
    For every table referenced in sql, fetch its real column list from DB.
    Returns a compact string injected into the critic prompt so the LLM
    cannot hallucinate column names on the next attempt.
    """
    try:
        from app.db.schema_inspector import get_columns
        referenced = _referenced_tables(sql)
        if not referenced:
            return ""
        lines = ["VERIFIED COLUMNS (exact names + types from live DB — use ONLY these):"]
        for tname in sorted(referenced):
            try:
                cols = get_columns(tname)
                col_list = ", ".join(
                    f"{c.name}({c.data_type}{'PK' if c.is_pk else ''})" for c in cols
                )
                lines.append(f"  {tname}: {col_list}")
            except Exception:
                pass
        return "\n".join(lines)
    except Exception:
        return ""


def validate_semantic(state: DbPipelineState) -> dict:
    if state.get("syntax_error") or state.get("table_error"):
        return {"semantic_error": ""}   # earlier check must pass first

    sql = state.get("sql", "").strip()
    if not sql:
        return {"semantic_error": "No SQL to validate."}

    target_db = state.get("target_db", "")
    try:
        if target_db:
            from app.db.connection import explain_on
            explain_on(sql, target_db)
        else:
            explain(sql)
        log.info("validate_semantic PASS (db=%s)", target_db or "main")
        return {"semantic_error": "", "column_facts": ""}
    except Exception as exc:
        err = str(exc).split("\n")[0]
        log.warning("validate_semantic FAIL: %s", err)
        log.warning("  SQL: %s", sql[:300])
        facts = _get_column_facts(sql)
        if facts:
            log.info("column_facts populated:\n%s", facts)
        return {"semantic_error": err, "column_facts": facts}


# ---------------------------------------------------------------------------
# Node 5 — Critic LLM: repair SQL using full error history
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = """\
You are a PostgreSQL SQL repair agent. The database contains financial data for Moov Benin.

╔══ OUTPUT FORMAT — STRICTLY ENFORCED ══╗
║  Output ONLY the corrected SQL SELECT   ║
║  • Start with SELECT or WITH            ║
║  • End with ;                           ║
║  • ZERO explanations, ZERO markdown     ║
╚═════════════════════════════════════════╝

⚠️  ABSOLUTE RULE — COLUMN NAMES:
If error says "column X does not exist", then X DOES NOT EXIST. Do NOT invent names.
Use ONLY the exact column names from VERIFIED COLUMNS below.

KEY INSIGHT — TWO SCHEMA TYPES:

DENORMALIZED (months as columns):
  • cashflow_data: [year, jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec, current_year_total]
    ✓ Query: SELECT SUM(jan), SUM(feb), SUM(mar), ... FROM cashflow_data WHERE year = 2024
    ✗ Do NOT use: date_column, month, EXTRACT(MONTH FROM ...), month column reference
  
  • commission_enlevements: [year, month, prodium, linarcels, somac, easycom, d_commercial, aftel, senaniminde]
    ✓ Query: SELECT prodium, linarcels, somac, ... FROM commission_enlevements WHERE year = 2025

NORMALIZED (date column):
  • financial_metrics_data: has date column → use WHERE EXTRACT(YEAR FROM date) = 2025
  • capex_data: has year and month columns → use WHERE year = 2025 AND month = 9

REPAIR PROCESS:
1. Look at VERIFIED COLUMNS
2. Identify if table is denormalized or normalized
3. If denormalized and error mentions non-existent columns → use the actual month columns (jan, feb, mar, ...)
4. If normalized and error is "can't use EXTRACT on integer" → use direct column comparison

STEP 1 — READ VERIFIED COLUMNS FIRST:
{column_facts}

STEP 2 — TABLE CONSTRAINT:
  Query ONLY these tables:
    {tables_in_sql}

STEP 3 — ERROR DIAGNOSIS:
  • "column X does not exist" → X is wrong. Use exact name from VERIFIED COLUMNS.
  • "relation X does not exist" → X is not a valid table.
  • "function pg_catalog.extract(unknown, integer)" → EXTRACT() on integer column. Use: WHERE year = 2024
  • "must appear in GROUP BY or aggregate" on commission_enlevements →
      Distributors are COLUMNS not rows. Fix by using UNION ALL unpivot:
        WITH totals AS (
            SELECT 'Prodium' AS distributor, SUM(prodium) AS total FROM commission_enlevements WHERE year = 2025
            UNION ALL SELECT 'Linarcels', SUM(linarcels) FROM commission_enlevements WHERE year = 2025
            UNION ALL SELECT 'Easycom',   SUM(easycom)   FROM commission_enlevements WHERE year = 2025
            UNION ALL SELECT 'Somac',     SUM(somac)     FROM commission_enlevements WHERE year = 2025
            UNION ALL SELECT 'D-Commercial', SUM(d_commercial) FROM commission_enlevements WHERE year = 2025
            UNION ALL SELECT 'Aftel',     SUM(aftel)     FROM commission_enlevements WHERE year = 2025
            UNION ALL SELECT 'Senaniminde', SUM(senaniminde) FROM commission_enlevements WHERE year = 2025
        )
        SELECT distributor, total AS total_commission FROM totals
        WHERE total IS NOT NULL ORDER BY total DESC LIMIT 5;
  • Missing columns in denormalized table → use the actual month/distributor columns shown in VERIFIED COLUMNS

STEP 4 — POSTGRESQL RULES:
  • EXTRACT() works ONLY on date/timestamp columns, NOT integers or text
  • Integer columns: WHERE year = 2024 (NOT EXTRACT(YEAR FROM year))
  • ✗ MONTH() YEAR() ISNULL() IFNULL() ← MySQL syntax — FORBIDDEN
  • ✓ NULLIF(amount, 0) to prevent division by zero
  • ✓ COALESCE(col, 0) for NULL defaults

Allowed tables: {allowed_tables}

SCHEMA (VERIFIED COLUMNS takes priority if there's conflict):
{schema}
"""

_CRITIC_USER = """\
USER QUESTION:
{question}

REPAIR HISTORY (all previous attempts, oldest first):
{history}

CURRENT BROKEN SQL:
{sql}

CURRENT ERROR:
{error}

Output ONLY the corrected SQL:
"""


def _make_critique_sql_node(llm: ChatOllama):
    def critique_sql(state: DbPipelineState) -> dict:
        retry    = state.get("retry_count", 0)
        schema   = state.get("retrieved_schema", "(no schema)")
        allowed  = state.get("allowed_tables", [])
        error    = (
            state.get("syntax_error")
            or state.get("table_error")
            or state.get("semantic_error")
            or "unknown error"
        )
        question = state["question"]
        sql      = state.get("sql", "")

        # Accumulate error history
        prev     = state.get("error_history", [])
        entry    = f"  Attempt {retry + 1}: {_clip(error, 200)}\n  SQL was: {_clip(sql, 160)}"
        history  = prev + [entry]

        column_facts = state.get("column_facts", "")
        facts_block    = column_facts if column_facts else (
            "VERIFIED COLUMNS: not yet available — use column names from SCHEMA below."
        )
        tables_in_sql  = sorted(_referenced_tables(sql)) or ["(no tables parsed)"]

        system_msg = _CRITIC_SYSTEM.format(
            column_facts=facts_block,
            tables_in_sql=", ".join(tables_in_sql),
            schema=schema,
            allowed_tables=", ".join(sorted(allowed)),
        )
        user_msg = _CRITIC_USER.format(
            question=question,
            history="\n".join(history),
            sql=sql,
            error=error,
        )

        log.warning(
            "[repair %d/%d] Critic invoked. Error: %s",
            retry + 1, _MAX_RETRIES, _clip(error, 160),
        )
        t0 = time.monotonic()
        response  = llm.invoke_streaming([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])
        elapsed   = time.monotonic() - t0

        fixed_sql, extract_err = _extract_sql(response.content)
        if extract_err:
            # Critic also produced non-SQL — count as a failure and let routing decide
            log.warning("Critic output rejected in %.1fs: %s", elapsed, extract_err)
            return {
                "sql": "",
                "error_history":   history,
                "critic_feedback": f"Repair {retry + 1} failed ({error}).",
                "syntax_error":    extract_err,
                "table_error":     "",
                "semantic_error":  "",
                "retry_count":     retry + 1,
            }

        log.info(
            "Critic fix in %.1fs:\n%s",
            elapsed,
            textwrap.indent(fixed_sql, "    "),
        )
        return {
            "sql":             fixed_sql,
            "error_history":   history,
            "critic_feedback": f"Repair {retry + 1}: fixed '{_clip(error, 80)}'.",
            "syntax_error":    "",
            "table_error":     "",
            "semantic_error":  "",
            "retry_count":     retry + 1,
        }

    return critique_sql


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _any_validation_error(state: DbPipelineState) -> str:
    return (
        state.get("syntax_error")
        or state.get("table_error")
        or state.get("semantic_error")
        or ""
    )


def _route_after_syntax(state: DbPipelineState) -> str:
    if state.get("syntax_error"):
        if state.get("retry_count", 0) < _MAX_RETRIES:
            return "critique_sql"
        log.error("Syntax retries exhausted — routing to format_answer")
        return "format_answer"
    return "validate_tables"


def _route_after_tables(state: DbPipelineState) -> str:
    if state.get("table_error"):
        if state.get("retry_count", 0) < _MAX_RETRIES:
            return "critique_sql"
        log.error("Table retries exhausted — routing to format_answer")
        return "format_answer"
    return "validate_semantic"


def _route_after_semantic(state: DbPipelineState) -> str:
    if state.get("semantic_error"):
        if state.get("retry_count", 0) < _MAX_RETRIES:
            return "critique_sql"
        log.error("Semantic retries exhausted — routing to format_answer")
        return "format_answer"
    return "execute_sql"


def _route_after_critique(state: DbPipelineState) -> str:
    return "validate_syntax"   # always re-validate from scratch after a fix


_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",    4: "April",
    5: "May",     6: "June",     7: "July",      8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

_MONTH_ORDER: dict[str, int] = {
    name.lower(): idx for idx, name in _MONTH_NAMES.items()
}

# Column names that commonly store month numbers (1-12)
_MONTH_COL_NAMES = {
    "month", "mois", "month_number", "month_no", "month_num",
    "month_index", "m", "mo", "periode_mois",
}


def _resolve_month_names(rows: list[dict], cols: list[str]) -> list[dict]:
    """
    Convert integer month values (1-12) to month names in any column whose
    name looks like a month-index column.  Mutates nothing — returns new rows.
    Also strips trailing spaces from to_char-style padded month strings.
    """
    month_cols = []
    for col in cols:
        if col.lower() not in _MONTH_COL_NAMES:
            continue
        # Only convert if all non-null values are integers 1-12
        vals = [r.get(col) for r in rows if r.get(col) is not None]
        if vals and all(isinstance(v, int) and 1 <= v <= 12 for v in vals):
            month_cols.append(col)

    if not month_cols:
        # Strip trailing spaces from to_char-padded strings (e.g. "January  ")
        str_month_cols = [
            col for col in cols
            if col.lower() in _MONTH_COL_NAMES
            and rows
            and isinstance(rows[0].get(col), str)
            and rows[0].get(col, "").strip().lower() in _MONTH_ORDER
        ]
        if not str_month_cols:
            return rows
        return [{**r, **{col: r[col].strip() for col in str_month_cols if isinstance(r.get(col), str)}} for r in rows]

    new_rows = []
    for r in rows:
        nr = dict(r)
        for col in month_cols:
            v = nr.get(col)
            if isinstance(v, int) and 1 <= v <= 12:
                nr[col] = _MONTH_NAMES[v]
        new_rows.append(nr)
    return new_rows


def _sort_by_month_order(rows: list[dict], cols: list[str]) -> list[dict]:
    """
    If any column contains month names, sort rows into chronological month order.
    Handles both integer-converted names and to_char-padded strings.
    """
    for col in cols:
        if col.lower() not in _MONTH_COL_NAMES:
            continue
        vals = [str(r.get(col, "")).strip().lower() for r in rows if r.get(col) is not None]
        if vals and all(v in _MONTH_ORDER for v in vals):
            return sorted(rows, key=lambda r: _MONTH_ORDER.get(str(r.get(col, "")).strip().lower(), 99))
    return rows


# ---------------------------------------------------------------------------
# Node 6 — Execute SQL
# ---------------------------------------------------------------------------

def execute_sql(state: DbPipelineState) -> dict:
    sql = state.get("sql", "").strip()
    if not sql:
        return {"sql_error": "No SQL available for execution.", "rows": [], "cols": []}

    # Last-resort guard — catches anything that slipped past AST validation
    m = _DESTRUCTIVE_RE.search(sql)
    if m:
        err = f"Execution blocked: destructive keyword '{m.group().upper()}' detected in SQL."
        log.error("execute_sql BLOCKED: %s", err)
        return {"sql_error": err, "rows": [], "cols": []}

    target_db = state.get("target_db", "")
    log.info("execute_sql (db=%s):\n%s", target_db or "main", textwrap.indent(sql, "    "))
    t0 = time.monotonic()
    try:
        if target_db:
            from app.db.connection import execute_on
            rows, cols = execute_on(sql, target_db)
        else:
            rows, cols = execute(sql)
        rows = _resolve_month_names(rows, cols)
        rows = _sort_by_month_order(rows, cols)
        log.info("execute_sql OK: %d rows, %d cols in %.1fs", len(rows), len(cols), time.monotonic() - t0)
        return {"rows": rows, "cols": cols, "sql_error": ""}
    except Exception as exc:
        log.error("execute_sql FAIL in %.1fs: %s", time.monotonic() - t0, exc)
        return {"sql_error": str(exc), "rows": [], "cols": []}


# ---------------------------------------------------------------------------
# Node 7 — Format answer (never leaks SQL)
# ---------------------------------------------------------------------------

_FMT_SYSTEM = """\
You are a senior financial analyst at a management consulting firm presenting database query results to the executive committee of Moov Benin (a West African telecom operator).

CURRENCY: All monetary values are in FCFA (West African CFA franc). NEVER write $ or USD.
UNIT — CRITICAL: financial_metrics_data, cashflow_data, and capex_data values are stored in MILLIONS of FCFA.
  A value of 4,683 means 4,683 M FCFA (i.e. 4.683 billion FCFA), NOT 4,683 FCFA.
  In tables: column names ending in _m_fcfa or containing real_value/budget_value are in M FCFA.
  In prose: write "4,683 M FCFA" or "4.7 billion FCFA" — NEVER "4,683 FCFA".
  revenue_raw_data values (ca_global, ca_voix, etc.) are in actual FCFA (not millions).
NUMBERS:  Use thousands separators in tables — 4,683. In prose, abbreviate: 4.7 billion FCFA, 150 M FCFA.
  Rule: if the primary metric column ends with _m_fcfa, append " (M FCFA)" to that column header in the rendered table.

CRITICAL — DATA FACTS are pre-computed in Python and are 100% accurate.
You MUST copy numbers from DATA FACTS verbatim into your Summary and Key Insights.
⚠ PEAK/TROUGH RULE: The PEAK is whatever DATA FACTS says PEAK is. Never scan the table to find a different row.
⚠ SINGLE-ROW RULE: If DATA FACTS says "Row count: 1", NEVER write "peak equals trough" or "single recorded figure".
  Instead, describe the total magnitude, state what portion of budget it represents (if budget column exists),
  compare to prior year (if prior_year column exists), and comment on business implications.
Month labels are always full names (January … December) — never write "Month N".
Do NOT re-derive totals, peaks, shares, or trend direction from the table.

OPEX ANALYSIS RULE:
  • When the result contains a `month` column (monthly trend): state the annual total, identify the peak and trough months,
    comment on seasonality. If budget_m_fcfa is present, note the worst budget miss month.
  • When the result contains an `opex_type` column (breakdown by type): state the grand total of actual_m_fcfa
    across all types, then name the largest cost category. Discuss cost structure, budget adherence per type,
    and YoY change if available.
  Never state "no comparative data available" if a prior_year_m_fcfa column is present in the result.
  Key Insights must include one bullet on the largest OpEx driver and one on budget vs actual (if those columns exist).

OUTPUT FORMAT — use EXACTLY this structure (no deviations):

## [Short descriptive title — 4–8 words, e.g. "Monthly CapEx Trend — 2025", "Top Suppliers by Spend — 2024", "EBITDA vs Budget Variance — Q1 2025"]

**Summary**
[2–3 sentences. State the single most important finding first, with the exact number from DATA FACTS. Then state trend direction and period. Be direct — no preamble like "Based on" or "The data shows".]

[Reproduce the data table exactly as provided — do not reformat, reorder, or omit rows. Keep all pipe characters.]

**Analysis**
[3–5 sentences of business reasoning. Go beyond restating facts: explain the WHY. What drives the peak or trough? Are there seasonal or cyclical patterns typical in West African telecom (e.g. budget cycles, infrastructure rollout seasons, rainy season subscriber churn)? What does the trend imply for next quarter? If this is a supplier/vendor ranking, comment on concentration risk or dependency. If it is a P&L or EBITDA metric, link the movement to likely operational causes. Draw inferences — do not re-list numbers already in the table.]

**Key Insights**
- **[Label]**: [value + % share or YoY context] — [one-sentence business implication or risk/opportunity]
- **[Label]**: [second finding with magnitude] — [operational or strategic implication]
- **[Label]**: [trend or pattern observed] — [what it signals for the next period]
- **[Label]**: [anomaly, risk concentration, or standout outlier] — [recommended action or watch item]

⚠ KEY INSIGHTS RULE: Write ONLY bullets that are supported by data in the result table.
  If budget_m_fcfa is present → write a budget variance bullet.
  If prior_year_m_fcfa is present → write a YoY comparison bullet.
  If those columns are ABSENT → skip those bullets entirely. NEVER write "N/A — No budget figure provided"
  or "N/A — Single quarter precludes YoY comparison". Silence is better than a meaningless N/A bullet.
  Replace any missing bullet with an observation that IS supported by the data (magnitude, trend, share).

**Data Source**: [table(s) queried] | [key columns: list from "Columns in result"] | [period: infer from data or query context] | [N rows returned]

**Follow-up**: [One specific drill-down question that would identify root cause or support a decision — not generic like "Would you like more detail?"]

QUARTERLY VARIANCE RULE — applies when result columns include q1_variance, q2_variance, q3_variance, q4_variance:
  The prompt will include a line "QUARTERLY DATA AVAILABILITY: Data available: Q1 (Jan–Mar), Q2 (Apr–Jun) | No data yet: Q3 (Jul–Sep), Q4 (Oct–Dec)" (exact quarters vary).

  After the main table, insert a "**Quarterly Trend**" section (before Analysis):

  0. DATA SCOPE — first sentence MUST state which quarters are covered and which are pending:
     "Analysis covers [available quarters] ([months]). [missing quarters] data not yet available."
     Example: "Analysis covers Q1 (Jan–Mar) and Q2 (Apr–Jun). Q3 and Q4 data not yet available."

  1. TREND DIRECTION — compare the absolute quarterly variances across available quarters only:
     - IMPROVING: absolute variance shrinking quarter-over-quarter
     - WORSENING: absolute variance growing quarter-over-quarter
     - MIXED or STABLE: no consistent direction
     State this in one sentence for the top 2–3 metrics.

  2. WORST QUARTER — name the quarter with the largest absolute deviation for the top metric.

  3. YEAR-END PROJECTION — based only on available quarters:
     Sum available quarterly variances, divide by count of available quarters, multiply by 4.
     State: "Based on [N] quarters of data, full-year variance is projected at ~X Bn FCFA."
     If only 1 quarter is available, caveat: "projection based on a single quarter — treat as indicative."

  4. ONE management action tied to the trend:
     - IMPROVING → "Corrective controls appear to be working; maintain current oversight cadence."
     - WORSENING → "Escalation recommended: variance is accelerating — review cost authorisation thresholds."
     - MIXED     → "Investigate Q[N] spike before drawing conclusions on structural trend."

  Key Insights must include one bullet labelled **Trend Signal** that states: direction + quarters covered + projection.

STRICT RULES:
- The ## title MUST reflect the actual question topic and period.
- The **Analysis** section is MANDATORY — minimum 3 sentences. Never skip it, never replace it with bullets.
- Key Insights: write exactly 4 bullets. Each must end with a business implication after the em dash (—).
- Reproduce the table AS-IS including all | characters.
- NEVER output SQL, column types, or any technical internals.
- NEVER invent numbers not present in the results or DATA FACTS.
- If a cell shows "(null)" → write "no data available". NEVER write "None" or "null".
- Do NOT start with "I'm sorry", "Based on", "The query returned", or "Here is".
- {language_instruction}
"""


def _cell(value) -> str:
    return "(null)" if value is None else str(value)


# ---------------------------------------------------------------------------
# Chart spec builder — pure Python, no LLM
# ---------------------------------------------------------------------------

_TIME_COL_NAMES = {
    "month", "year", "date", "period", "quarter",
    "month_number", "month_no", "month_num",
    "week", "week_number", "week_no",
    "quarter_number", "quarter_no",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
}

_RANK_KEYWORDS = {
    "top", "rank", "best", "worst", "highest", "lowest",
    "distributor", "supplier", "fournisseur", "category", "categorie",
}

_TREND_KEYWORDS = {
    "trend", "monthly", "evolution", "mensuel", "par mois",
    "over time", "breakdown", "month by month",
}

_PALETTE = ["#6c63ff", "#4ecca3", "#f6ad55", "#fc8181", "#63b3ed", "#f687b3"]


def _to_float(v) -> float | None:
    from decimal import Decimal
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _compute_trend_analysis(rows: list[dict], x_key: str, y_key: str | None) -> str:
    """Compute a short plain-English trend summary from chart data."""
    if not y_key or len(rows) < 2:
        return ""
    vals = [(_to_float(r.get(y_key)), str(r.get(x_key, ""))) for r in rows]
    vals = [(v, lbl) for v, lbl in vals if v is not None]
    if len(vals) < 2:
        return ""

    numbers = [v for v, _ in vals]
    total   = sum(numbers)
    avg     = total / len(numbers)
    max_v, max_lbl = max(vals, key=lambda x: x[0])
    min_v, min_lbl = min(vals, key=lambda x: x[0])
    first, last = numbers[0], numbers[-1]
    overall_change_pct = ((last - first) / abs(first) * 100) if first else 0

    increases = sum(1 for i in range(1, len(numbers)) if numbers[i] > numbers[i - 1])
    decreases = len(numbers) - 1 - increases
    if increases > decreases:
        direction = "an overall upward trend"
    elif decreases > increases:
        direction = "an overall downward trend"
    else:
        direction = "mixed movement with no clear direction"

    parts = [
        f"The data shows {direction} across {len(vals)} periods.",
        f"Peak: {max_lbl} at {max_v:,.0f} FCFA.",
        f"Trough: {min_lbl} at {min_v:,.0f} FCFA.",
        f"Average: {avg:,.0f} FCFA.",
    ]
    if abs(overall_change_pct) >= 1:
        sign = "+" if overall_change_pct > 0 else ""
        parts.append(f"Net change from first to last period: {sign}{overall_change_pct:.1f}%.")
    return " ".join(parts)


def _build_chart_spec(cols: list[str], rows: list[dict], question: str) -> dict | None:
    """Infer a chart spec from the result shape and question keywords. Returns None if not chartable."""
    if len(rows) < 2 or len(cols) < 2:
        return None

    # Classify columns
    numeric_cols: list[str] = []
    label_cols:   list[str] = []
    for c in cols:
        samples = [r.get(c) for r in rows[:10] if r.get(c) is not None]
        if samples and all(_to_float(v) is not None for v in samples):
            numeric_cols.append(c)
        else:
            label_cols.append(c)

    if not numeric_cols:
        return None

    # Pick x-axis: prefer time column, then first label col
    time_col = next((c for c in cols if c.lower() in _TIME_COL_NAMES), None)
    x_key    = time_col or (label_cols[0] if label_cols else None)
    if x_key is None:
        return None

    y_keys = [c for c in numeric_cols if c != x_key][:4]
    if not y_keys:
        # All columns are numeric — use row index as x
        y_keys = numeric_cols[:4]
        x_key  = cols[0]

    # Detect chart type
    q = question.lower()
    is_time  = time_col in ("month", "date", "period") or any(k in q for k in _TREND_KEYWORDS)
    is_rank  = any(k in q for k in _RANK_KEYWORDS) and len(rows) <= 20
    is_multi = len(y_keys) > 1

    if is_time:
        chart_type = "line"
    elif is_rank:
        chart_type = "bar_horizontal"
    else:
        chart_type = "bar"

    # Serialize — convert Decimal/None to float/null
    data = []
    for r in rows[:60]:
        entry: dict = {}
        entry[x_key] = str(r.get(x_key) or "")
        for y in y_keys:
            v = _to_float(r.get(y))
            entry[y] = v  # None serialises to null in JSON
        data.append(entry)

    # Peak / trough indices and stats (needed for axis label unit detection)
    primary_vals = [(_to_float(r.get(y_keys[0])), i) for i, r in enumerate(data)]
    primary_vals_nn = [(v, i) for v, i in primary_vals if v is not None]
    peak_idx   = max(primary_vals_nn, key=lambda x: x[0])[1] if primary_vals_nn else None
    trough_idx = min(primary_vals_nn, key=lambda x: x[0])[1] if primary_vals_nn else None
    nums_nn    = [v for v, _ in primary_vals_nn]
    avg_value  = sum(nums_nn) / len(nums_nn) if nums_nn else None

    # Detect unit from column name and value magnitude
    _pct_keywords = ("pct", "rate", "ratio", "percent", "evol", "change", "growth", "margin", "share", "taux")
    y_col_lower = y_keys[0].lower() if y_keys else ""
    if any(k in y_col_lower for k in _pct_keywords):
        unit = "%"
        y_suffix = " (%)"
    else:
        unit = "FCFA"
        avg_abs = sum(abs(v) for v in nums_nn) / len(nums_nn) if nums_nn else 0
        if avg_abs >= 1e9:
            y_suffix = " (Bn FCFA)"
        elif avg_abs >= 1e6:
            y_suffix = " (M FCFA)"
        else:
            y_suffix = " (FCFA)"

    # Axis labels — humanised column names with unit context
    x_label = x_key.replace("_", " ").title()
    y_label = (y_keys[0].replace("_", " ").title() + y_suffix) if len(y_keys) == 1 else f"Value ({unit})"

    # Chart title derived from column names
    y_human = ", ".join(k.replace("_", " ").title() for k in y_keys[:2])
    x_human = x_key.replace("_", " ").title()
    if is_time:
        raw_title = f"{y_human} — Monthly Trend"
    elif is_rank:
        raw_title = f"{y_human} — Ranking"
    else:
        raw_title = f"{y_human} by {x_human}"
    title = raw_title[:80]

    # Trend analysis from the full row set
    trend_analysis = _compute_trend_analysis(rows, x_key, y_keys[0] if y_keys else None)

    return {
        "chart_type":       chart_type,
        "title":            title,
        "data":             data,
        "x_key":            x_key,
        "y_keys":           y_keys,
        "colors":           _PALETTE[: len(y_keys)],
        "unit":             unit,
        "x_label":          x_label,
        "y_label":          y_label,
        "trend_analysis":   trend_analysis,
        "peak_idx":         peak_idx,
        "trough_idx":       trough_idx,
        "avg_value":        avg_value,
        "y_begin_at_zero":  chart_type in ("bar", "bar_horizontal"),
    }


def _fmt_number(s: str) -> str:
    """Add thousands separators to bare numeric strings."""
    try:
        f = float(s)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (ValueError, TypeError):
        return s


# Max rows shown to the LLM — keeps prompts short and avoids cloud timeouts.
# DATA FACTS already captures totals/peaks for the full result set.
_LLM_TABLE_ROW_CAP = 30


def _build_ascii_table(rows: list[dict], cols: list[str], cap: int = _LLM_TABLE_ROW_CAP) -> str:
    # Rename ugly PostgreSQL default headers
    display_cols = ["value" if c in ("?column?", "?column?") else c for c in cols]
    col_rename   = dict(zip(cols, display_cols))

    sample = rows[:cap]
    col_w  = {
        dc: max(len(dc), max((len(_fmt_number(_cell(r.get(c)))) for r in sample), default=0))
        for c, dc in col_rename.items()
    }
    header = "| " + " | ".join(dc.ljust(col_w[dc]) for dc in display_cols) + " |"
    sep    = "| " + " | ".join("-" * col_w[dc] for dc in display_cols) + " |"
    lines  = [header, sep]
    for r in sample:
        lines.append("| " + " | ".join(
            _fmt_number(_cell(r.get(c))).ljust(col_w[dc])
            for c, dc in col_rename.items()
        ) + " |")
    if len(rows) > cap:
        lines.append(f"... ({len(rows) - cap} more rows — totals in DATA FACTS above)")
    return "\n".join(lines)


def _compute_data_facts(cols: list[str], rows: list[dict]) -> str:
    """
    Pre-compute key statistics from the result set in Python (ground truth).
    Injected into the LLM prompt so it cannot misidentify peaks, totals, or shares.
    """
    from decimal import Decimal

    if not rows or not cols:
        return ""

    # Columns that look like dimension/index, not metrics
    _DIM_EXACT = {
        "month", "year", "id", "sequence", "rank", "quarter", "week",
        "month_number", "week_number", "quarter_number",
        "month_no", "week_no", "quarter_no", "num", "number", "m",
    }
    _DIM_SUFFIXES = ("_id", "_number", "_no", "_num", "_rank", "_seq")

    def _is_dim_col(name: str, vals: list) -> bool:
        n = name.lower()
        if n in _DIM_EXACT:
            return True
        if any(n.endswith(s) for s in _DIM_SUFFIXES):
            return True
        # Heuristic: if all values are small integers (≤ 366), likely an index/period column
        nums = [_to_float(v) for v in vals if v is not None]
        if nums and all(v is not None and v == int(v) and abs(v) <= 366 for v in nums):
            return True
        return False

    # Percentage/rate column names — when present, prefer these over raw value columns
    # so peak/trough reflect the best/worst % change, not highest absolute value.
    _PCT_SUFFIXES = ("_pct", "_rate", "_share", "_ratio", "_variance", "pct", "rate")

    numeric_col = label_col = pct_col = None
    for c in cols:
        vals = [r.get(c) for r in rows if r.get(c) is not None]
        if not vals:
            continue
        is_numeric = all(isinstance(v, (int, float, Decimal)) for v in vals)
        is_dim     = _is_dim_col(c, vals)
        if is_numeric and not is_dim:
            if numeric_col is None:
                numeric_col = c
            # Track first pct/rate column separately
            if pct_col is None and any(c.lower().endswith(s) for s in _PCT_SUFFIXES):
                pct_col = c
        else:
            if label_col is None:
                label_col = c

    # For comparison/variance queries, use the % column so peak = least decline,
    # trough = worst decline — more meaningful than picking by absolute value.
    if pct_col and pct_col != numeric_col:
        numeric_col = pct_col

    if not numeric_col:
        return ""

    float_vals = [(_to_float(r.get(numeric_col)), r) for r in rows]
    float_vals = [(v, r) for v, r in float_vals if v is not None]
    if not float_vals:
        return ""

    total      = sum(v for v, _ in float_vals)
    max_val, max_row = max(float_vals, key=lambda x: x[0])
    min_val, min_row = min(float_vals, key=lambda x: x[0])
    max_label  = str(max_row.get(label_col or cols[0], ""))
    min_label  = str(min_row.get(label_col or cols[0], ""))

    # For pct/rate columns: summing percentages is meaningless — show range instead.
    is_pct_col = any(numeric_col.lower().endswith(s) for s in _PCT_SUFFIXES)

    # Single-row result: skip peak/trough entirely, just state the value
    if len(float_vals) == 1:
        unit_hint = " M FCFA" if "_m_fcfa" in numeric_col.lower() else ""
        facts = [
            f"DATA FACTS — AUTHORITATIVE. Copy these numbers verbatim. DO NOT re-derive from the table:",
            f"  • Value: {max_val:,.2f}{unit_hint}",
            f"  • Row count: 1 (single data point — do NOT write 'peak = trough')",
            f"  • Metric column: {numeric_col}  |  Label column: {label_col or cols[0]}",
            f"  NOTE: This is a single aggregate value. Do NOT say 'peak equals trough'. Focus analysis on magnitude, budget comparison, and business context.",
        ]
        return "\n".join(facts)

    if is_pct_col:
        # PEAK = least decline (closest to 0 / highest), TROUGH = worst decline (most negative)
        peak_label  = "BEST (least decline)" if max_val < 0 else "BEST"
        trough_label = "WORST (most decline)" if min_val < 0 else "WORST"
        facts = [
            f"DATA FACTS — AUTHORITATIVE. Copy these numbers verbatim. DO NOT re-derive from the table:",
            f"  • {peak_label}: label={max_label!r}  value={max_val:,.2f}%",
            f"  • {trough_label}: label={min_label!r}  value={min_val:,.2f}%",
            f"  • Row count  : {len(rows)}",
            f"  • Metric column: {numeric_col} (% change)  |  Label column: {label_col or cols[0]}",
        ]
    else:
        max_share = (max_val / total * 100) if total else 0
        facts = [
            f"DATA FACTS — AUTHORITATIVE. Copy these numbers verbatim. DO NOT re-derive from the table:",
            f"  • HIGHEST: label={max_label!r}  value={max_val:,.2f}  share={max_share:.1f}% of total",
            f"  • LOWEST : label={min_label!r}  value={min_val:,.2f}",
            f"  • Grand total    : {total:,.2f}",
            f"  • Row count      : {len(rows)}",
            f"  • Metric column  : {numeric_col}  |  Label column: {label_col or cols[0]}",
        ]
        # Trend direction only makes sense for time-series numeric values
        if len(float_vals) >= 3:
            ordered_vals = [v for v, _ in float_vals]
            increases = sum(1 for i in range(1, len(ordered_vals)) if ordered_vals[i] > ordered_vals[i-1])
            direction = "mostly increasing" if increases > len(ordered_vals) // 2 else "mostly decreasing" if increases < len(ordered_vals) // 2 else "mixed"
            facts.append(f"  • Trend direction: {direction}")

    return "\n".join(facts)


def _format_fallback(rows: list[dict], cols: list[str], question: str,
                     facts: str, source: str, chart_spec: dict | None,
                     lang: str) -> dict:
    """
    Return a structured answer built entirely from pre-computed Python values,
    used when the LLM call times out or fails.
    """
    table = _build_ascii_table(rows, cols)

    if lang == "fr":
        source_label   = "Source"
        followup_label = "Suggestion"
        followup_text  = "Souhaitez-vous filtrer ou approfondir ces résultats ?"
        note           = f"(Réponse générée sans IA — {len(rows)} ligne(s) récupérée(s))"
        insights_label = "Points clés"
    else:
        source_label   = "Source"
        followup_label = "Follow-up"
        followup_text  = "Would you like to filter or drill down into these results?"
        note           = f"(Response generated without AI narration — {len(rows)} row(s) returned)"
        insights_label = "Key Insights"

    # Extract DATA FACTS bullets for Key Insights
    bullets: list[str] = []
    if facts:
        for line in facts.splitlines():
            line = line.strip().lstrip("•").strip()
            if line and not line.startswith("DATA FACTS"):
                bullets.append(f"- **{line.split(':')[0].strip()}**: {':'.join(line.split(':')[1:]).strip()}" if ":" in line else f"- {line}")
    bullets_text = "\n".join(bullets[:4]) if bullets else f"- {len(rows)} row(s) returned"

    summary_text = bullets[0].lstrip("- ").strip() if bullets else f"{len(rows)} row(s) returned."
    title = question[:70]

    answer = (
        f"## {title}\n\n"
        f"**Summary**\n{summary_text}\n\n"
        f"{table}\n\n"
        f"**{insights_label}**\n{bullets_text}\n\n"
        f"**{source_label}**: {source}\n\n"
        f"**{followup_label}**: {followup_text}\n\n"
        f"_{note}_"
    )
    log.warning("format_answer: used fallback (no LLM) for %d rows", len(rows))
    return {"answer": answer, "chart_specs": [chart_spec] if chart_spec else []}


def _source_context(sql: str, rows: list[dict], cols: list[str]) -> str:
    """Derive a validation/provenance line from the SQL, result size, and column list."""
    tables = sorted(_referenced_tables(sql))
    table_str = ", ".join(tables) if tables else "database"
    n = len(rows)
    row_str = f"{n} row" if n == 1 else f"{n} rows"

    # Period from SQL year filter
    year_m = re.search(r"\b(20\d{2})\b", sql)
    period = f" — {year_m.group(1)}" if year_m else ""

    # Key columns (exclude FK/id suffixes and trivial keys)
    key_cols = [c for c in cols if not c.endswith("_id") and c not in ("id",)][:5]
    col_str = f" | columns: {', '.join(key_cols)}" if key_cols else ""

    return f"{table_str}{period} | {row_str} returned{col_str}"


_LANG_INSTRUCTIONS = {
    "fr": "Répondez en français. Toutes les réponses, analyses et libellés doivent être en français.",
    "en": "Respond in English.",
}

_QUARTER_MONTHS = {
    "q1_variance": ("Q1", "Jan–Mar"),
    "q2_variance": ("Q2", "Apr–Jun"),
    "q3_variance": ("Q3", "Jul–Sep"),
    "q4_variance": ("Q4", "Oct–Dec"),
}


def _detect_quarterly_availability(rows: list[dict], cols: list[str]) -> str:
    """
    If the result contains q1_variance…q4_variance columns, return a one-line
    data-availability summary so the LLM knows which quarters are populated.

    Returns empty string when no quarterly columns are present.
    """
    present = [c for c in ["q1_variance", "q2_variance", "q3_variance", "q4_variance"] if c in cols]
    if not present:
        return ""

    available, missing = [], []
    for col in ["q1_variance", "q2_variance", "q3_variance", "q4_variance"]:
        label, months = _QUARTER_MONTHS[col]
        has_data = any(r.get(col) is not None for r in rows)
        if has_data:
            available.append(f"{label} ({months})")
        else:
            missing.append(f"{label} ({months})")

    parts = []
    if available:
        parts.append(f"Data available: {', '.join(available)}")
    if missing:
        parts.append(f"No data yet: {', '.join(missing)}")

    return "QUARTERLY DATA AVAILABILITY: " + " | ".join(parts)


def _make_format_answer_node(llm: ChatOllama):
    def format_answer(state: DbPipelineState) -> dict:
        retry     = state.get("retry_count", 0)
        sql_error = state.get("sql_error", "")
        rows      = state.get("rows", [])
        cols      = state.get("cols", [])
        question  = state["question"]
        sql       = state.get("sql", "")
        lang      = state.get("language", "en")

        # ── Validation exhausted ─────────────────────────────────────────
        remaining = _any_validation_error(state)
        if remaining and not rows:
            log.warning("format_answer: SQL_GENERATION_FAILED after %d retries", retry)
            return {"answer": (
                f"SQL_GENERATION_FAILED: I was unable to generate a valid query "
                f"after {retry} repair attempt(s).\n"
                f"Last error: {remaining[:300]}"
            )}

        # ── Runtime execution error ──────────────────────────────────────
        if sql_error:
            reason = sql_error.split("\n")[0][:250]
            log.warning("format_answer: execution error → %s", reason)
            return {"answer": f"I could not retrieve the data. Reason: {reason}"}

        # ── Empty result set ─────────────────────────────────────────────
        if not rows:
            log.info("format_answer: 0 rows returned")
            return {"answer": (
                "The query ran successfully but returned no results. "
                "Try broadening your filters or checking the date range."
            )}

        # ── Build table, facts, chart spec, and source context ───────────
        snapshot_label = state.get("snapshot_label", "")
        table      = _build_ascii_table(rows, cols)
        facts      = _compute_data_facts(cols, rows)
        source     = _source_context(sql, rows, cols)
        chart_spec = _build_chart_spec(cols, rows, question)
        if chart_spec:
            log.info("format_answer: chart_type=%s x=%s y=%s",
                     chart_spec["chart_type"], chart_spec["x_key"], chart_spec["y_keys"])
        if facts:
            log.info("format_answer: data facts injected:\n%s", facts)

        lang_instr = _LANG_INSTRUCTIONS.get(lang, _LANG_INSTRUCTIONS["en"])
        fmt_system = _FMT_SYSTEM.format(language_instruction=lang_instr)

        snapshot_ctx  = f"Database snapshot: {snapshot_label}\n" if snapshot_label else ""
        col_ctx       = f"Columns in result: {', '.join(cols)}\n" if cols else ""
        quarterly_ctx = _detect_quarterly_availability(rows, cols)
        quarterly_line = f"{quarterly_ctx}\n" if quarterly_ctx else ""

        log.info("format_answer: narrating %d rows via LLM (lang=%s, showing up to %d)",
                 len(rows), lang, _LLM_TABLE_ROW_CAP)
        if quarterly_ctx:
            log.info("format_answer: %s", quarterly_ctx)
        t0 = time.monotonic()
        try:
            response = llm.invoke_streaming([
                SystemMessage(content=fmt_system),
                HumanMessage(content=(
                    f"User question: {question}\n\n"
                    f"{col_ctx}"
                    f"{quarterly_line}"
                    f"{snapshot_ctx}"
                    f"{facts}\n\n"
                    f"Query source (for the **Data Source** line): {source}\n\n"
                    f"Result ({len(rows)} row(s), first {_LLM_TABLE_ROW_CAP} shown):\n{table}"
                )),
            ])
            log.info("format_answer LLM done in %.1fs", time.monotonic() - t0)
            return {
                "answer":      response.content,
                "chart_specs": [chart_spec] if chart_spec else [],
            }
        except Exception as llm_err:
            log.error("format_answer LLM failed after %.1fs: %s — using fallback",
                      time.monotonic() - t0, llm_err)
            return _format_fallback(rows, cols, question, facts, source, chart_spec, lang)

    return format_answer


# ---------------------------------------------------------------------------
# Build the pipeline graph
# ---------------------------------------------------------------------------

def _build_db_pipeline(model: str | None = None) -> object:
    sql_llm       = _make_llm(model)
    narration_model = settings.OLLAMA_NARRATION_MODEL or model or settings.OLLAMA_MODEL
    narration_llm = _make_llm(narration_model)
    g   = StateGraph(DbPipelineState)

    g.add_node("resolve_snapshot",  resolve_snapshot)
    g.add_node("retrieve_schema",   retrieve_schema)
    g.add_node("write_sql",         _make_write_sql_node(sql_llm))
    g.add_node("validate_syntax",   validate_syntax)
    g.add_node("validate_tables",   validate_tables)
    g.add_node("validate_semantic", validate_semantic)
    g.add_node("critique_sql",      _make_critique_sql_node(sql_llm))
    g.add_node("execute_sql",       execute_sql)
    g.add_node("format_answer",     _make_format_answer_node(narration_llm))

    g.set_entry_point("resolve_snapshot")
    g.add_edge("resolve_snapshot",  "retrieve_schema")
    g.add_edge("retrieve_schema",   "write_sql")
    g.add_edge("write_sql",       "validate_syntax")

    g.add_conditional_edges(
        "validate_syntax", _route_after_syntax,
        {"critique_sql": "critique_sql", "validate_tables": "validate_tables", "format_answer": "format_answer"},
    )
    g.add_conditional_edges(
        "validate_tables", _route_after_tables,
        {"critique_sql": "critique_sql", "validate_semantic": "validate_semantic", "format_answer": "format_answer"},
    )
    g.add_conditional_edges(
        "validate_semantic", _route_after_semantic,
        {"critique_sql": "critique_sql", "execute_sql": "execute_sql", "format_answer": "format_answer"},
    )
    g.add_conditional_edges(
        "critique_sql", _route_after_critique,
        {"validate_syntax": "validate_syntax"},
    )
    g.add_edge("execute_sql",   "format_answer")
    g.add_edge("format_answer", END)

    return g.compile()


_db_pipeline_cache: dict[str, object] = {}


def _get_db_pipeline(model: str | None = None) -> object:
    key = model or settings.OLLAMA_MODEL
    if key not in _db_pipeline_cache:
        _db_pipeline_cache[key] = _build_db_pipeline(model=key)
    return _db_pipeline_cache[key]


# ---------------------------------------------------------------------------
# Question intent classification
# ---------------------------------------------------------------------------

_INTENT_CLASSIFIER_PROMPT = """\
Classify the user's question as exactly one of these five labels:
  definition   — ONLY asking for an explanation or definition of a financial term
  data_query   — asking for data, numbers, trends, or analysis from the database
  both         — asking BOTH a definition AND data in the same message
  out_of_scope — asking about something the financial database cannot answer:
                 external regulatory decrees, tax law amendments, court rulings,
                 government policy documents, news events, competitor information,
                 or any causal "why" question that requires legal/regulatory documents
                 not stored in the database
  other        — greetings, small talk, or completely off-topic

The question may be in English OR French. Classify based on meaning, not language.

English examples:
  "What is CAPEX?"                                          → definition
  "Show me monthly revenue for 2025"                        → data_query
  "What is EBITDA and what was it in 2024?"                 → both
  "Hello, how are you?"                                     → other
  "Which decree triggered the variance in Redevances?"      → out_of_scope
  "What tax law caused the OPEX increase?"                  → out_of_scope
  "Which regulatory amendment changed the fee structure?"   → out_of_scope
  "Why did the government change roaming fees?"             → out_of_scope

French examples:
  "Qu'est-ce que l'EBITDA ?"                                → definition
  "Montre-moi l'évolution du chiffre d'affaires"            → data_query
  "Qu'est-ce que le churn et quel est son taux ?"           → both
  "Quel décret a causé la variance des redevances ?"        → out_of_scope
  "Quelle ordonnance fiscale a modifié les charges ?"       → out_of_scope
  "Bonjour, comment ça va ?"                                → other

Respond with ONLY one word — the label itself. No punctuation, no explanation.

Question: {question}
Classification:"""


_OUT_OF_SCOPE_TEMPLATE_EN = """\
This question asks about external regulatory or legal information (decrees, tax ordinances, policy amendments) that is not stored in this database.

The TBG Copilot database contains **financial metrics only** — actual vs. budget values, monthly trends, OpEx, CapEx, revenue, and KPIs for Moov Benin. It does not hold regulatory documents, legal texts, or government decree archives.

**What I can tell you from the database:**
- The exact variance in Redevances régulateur (actual vs. budget, vs. prior year) for any month/year
- Whether the variance is isolated to one month or part of a sustained trend
- How Redevances compares to other OpEx categories

To identify the specific regulatory trigger, consult:
- ARCEP Bénin (the regulator) published decree logs
- Moov Benin's legal/regulatory affairs team
- The company's annual report regulatory section

Would you like me to pull the Redevances régulateur data so you can cross-reference it with the regulation timeline?
"""

_OUT_OF_SCOPE_TEMPLATE_FR = """\
Cette question porte sur des informations réglementaires ou juridiques externes (décrets, ordonnances fiscales, amendements) qui ne sont pas stockées dans cette base de données.

La base TBG Copilot contient uniquement des **métriques financières** — valeurs réalisées vs. budget, tendances mensuelles, OpEx, CapEx, revenus et KPIs de Moov Bénin. Elle ne contient pas de documents réglementaires, de textes juridiques ni d'archives de décrets gouvernementaux.

**Ce que je peux vous fournir depuis la base :**
- L'écart exact des Redevances régulateur (réalisé vs. budget, vs. N-1) pour n'importe quel mois/année
- Si l'écart est isolé à un mois ou s'inscrit dans une tendance durable
- La comparaison des Redevances avec les autres catégories OpEx

Pour identifier le déclencheur réglementaire précis, consultez :
- Les journaux de décrets publiés par l'ARCEP Bénin
- L'équipe affaires juridiques/réglementaires de Moov Bénin
- La section réglementaire du rapport annuel de l'entreprise

Souhaitez-vous que je récupère les données Redevances régulateur pour les croiser avec la chronologie réglementaire ?
"""


def _get_out_of_scope_answer(question: str, language: str = "en") -> str:
    """Return a clear out-of-scope message when the question requires external regulatory/legal data."""
    q_lower = question.lower()
    # Try to detect what metric the user is asking about
    metric_hint = ""
    for kw in ("redevance", "redevances", "opex", "capex", "revenue", "charge", "frais"):
        if kw in q_lower:
            metric_hint = kw
            break

    if language == "fr":
        return _OUT_OF_SCOPE_TEMPLATE_FR
    return _OUT_OF_SCOPE_TEMPLATE_EN


def _classify_question_intent(llm: OllamaLLM, question: str) -> str:
    """Use LLM to classify if question is asking for definition, data, both, out_of_scope, or other."""
    # Fast keyword pre-check for obvious out-of-scope patterns before calling the LLM
    _OOS_KEYWORDS = (
        "decree", "décret", "ordonnance", "ordinance", "regulation amendment",
        "tax law", "loi fiscale", "loi de finance", "amendement", "amendment",
        "court ruling", "jugement", "tribunal", "politique gouvernementale",
        "government policy", "legislative", "législatif", "arcep ruling",
        "which law", "quelle loi", "quel texte", "what regulation caused",
        "quelle réglementation", "triggered by", "déclenchée par",
    )
    q_lower = question.lower()
    if any(kw in q_lower for kw in _OOS_KEYWORDS):
        log.info("Question pre-classified as out_of_scope via keyword match")
        return "out_of_scope"

    try:
        prompt = _INTENT_CLASSIFIER_PROMPT.format(question=question)
        response = llm.invoke_streaming([
            SystemMessage(content="You are a question classifier."),
            HumanMessage(content=prompt),
        ])
        classification = response.content.strip().lower()

        valid = ("definition", "data_query", "both", "out_of_scope", "other")
        if classification not in valid:
            log.warning("Unexpected classification output: %s — defaulting to data_query", classification)
            return "data_query"

        log.info("Question classified as: %s", classification)
        return classification
    except Exception as exc:
        log.warning("Classification failed (%s) — defaulting to data_query", exc)
        return "data_query"


def _load_financial_terms() -> list[dict]:
    """Load financial_terms.json once; return empty list on failure."""
    try:
        with open(_FINANCIAL_TERMS_PATH, encoding="utf-8") as fh:
            return json.load(fh).get("terms", [])
    except Exception as exc:
        log.warning("Could not load financial_terms.json: %s", exc)
        return []

_FINANCIAL_TERMS: list[dict] = _load_financial_terms()


def _lookup_term(question: str) -> dict | None:
    """
    Find the best matching term entry for a question.
    Matches against the canonical term name and all aliases (case-insensitive).
    Returns the term dict, or None if no match found.
    """
    q_lower = question.lower()
    best: dict | None = None
    best_len = 0
    for entry in _FINANCIAL_TERMS:
        candidates = [entry["term"].lower()] + [a.lower() for a in entry.get("aliases", [])]
        for candidate in candidates:
            # Short aliases (≤4 chars) must match on a word boundary to avoid
            # false positives like "ca" matching inside "categories".
            if len(candidate) <= 4:
                pattern = r"(?<![a-z])" + re.escape(candidate) + r"(?![a-z])"
                matched = bool(re.search(pattern, q_lower))
            else:
                matched = candidate in q_lower
            if matched and len(candidate) > best_len:
                best = entry
                best_len = len(candidate)
    return best


def _format_term_reference(entry: dict, language: str) -> str:
    """Format a financial_terms.json entry into readable reference text."""
    lang = language if language in ("en", "fr") else "en"
    definition_key = f"definition_{lang}"
    definition = entry.get(definition_key) or entry.get("definition_en", "")

    lines = [
        f"**{entry['term']}**",
        f"*Category: {entry.get('category', '')}*",
        "",
        definition,
    ]
    if formula := entry.get("formula"):
        lines += ["", f"**Formula:** `{formula}`"]
    if unit := entry.get("unit"):
        lines += [f"**Unit:** {unit}"]
    if context := entry.get("context"):
        lines += ["", f"**Context:** {context}"]
    return "\n".join(lines)


def _get_definition_answer(question: str, llm: OllamaLLM, language: str = "en", definition_only: bool = False) -> str:
    """
    Answer a definition question.
    1. Check financial_terms.json for a matching entry — if found, inject the
       reference text into the LLM prompt so it gives an authoritative answer.
    2. Fall back to pure LLM knowledge if no entry matches.

    definition_only=True: strictly define the term; do not acknowledge or attempt
    to answer any data/trend parts of the question (those will be handled separately).
    """
    lang_instr = _LANG_INSTRUCTIONS.get(language, _LANG_INSTRUCTIONS["en"])
    term_entry = _lookup_term(question)

    scope_instruction = (
        "Your ONLY task is to define the financial term mentioned in the question. "
        "Stop after the definition. "
        "NEVER mention data, trends, monthly figures, or numerical analysis. "
        "NEVER say you cannot provide something. "
        "NEVER add any disclaimer, caveat, or note about data availability."
    ) if definition_only else (
        "Answer the question precisely and concisely."
    )

    if term_entry:
        log.info("Term reference found: %s", term_entry["term"])
        reference_block = _format_term_reference(term_entry, language)
        system_content = (
            "You are a financial analyst expert for Moov Benin. "
            f"{scope_instruction} "
            "Use the following authoritative reference. "
            "You may expand with telecom/Benin context but do not contradict the reference. "
            f"{lang_instr}\n\n"
            "--- REFERENCE ---\n"
            f"{reference_block}\n"
            "--- END REFERENCE ---"
        )
    else:
        log.info("No term reference found — using LLM knowledge only")
        system_content = (
            "You are a financial analyst expert for Moov Benin. "
            f"{scope_instruction} "
            f"Include practical examples relevant to telecom/financial services in Benin. "
            f"{lang_instr}"
        )

    # When definition_only, extract just the term name to avoid the LLM seeing
    # the data/trend part of the question and feeling compelled to address it.
    if definition_only and term_entry:
        user_prompt = f"Define {term_entry['term']}."
    else:
        user_prompt = question

    try:
        response = llm.invoke_streaming([
            SystemMessage(content=system_content),
            HumanMessage(content=user_prompt),
        ])
        answer = response.content.strip()
        if not answer:
            answer = "I was unable to generate an explanation. Please try rephrasing your question."
        return answer
    except Exception as exc:
        log.error("Definition generation failed: %s", exc)
        if term_entry:
            return _format_term_reference(term_entry, language)
        return f"I encountered an error while generating the explanation: {str(exc)}"


def _get_definition_tokens(question: str, llm: OllamaLLM, language: str = "en", definition_only: bool = False):
    """Stream tokens for a definition answer."""
    lang_instr = _LANG_INSTRUCTIONS.get(language, _LANG_INSTRUCTIONS["en"])
    term_entry = _lookup_term(question)

    scope_instruction = (
        "Your ONLY task is to define the financial term mentioned in the question. "
        "Stop after the definition. "
        "NEVER mention data, trends, monthly figures, or numerical analysis. "
        "NEVER say you cannot provide something. "
        "NEVER add any disclaimer, caveat, or note about data availability."
    ) if definition_only else (
        "Answer the question precisely and concisely."
    )

    if term_entry:
        reference_block = _format_term_reference(term_entry, language)
        system_content = (
            "You are a financial analyst expert for Moov Benin. "
            f"{scope_instruction} "
            "Use the following authoritative reference. "
            "You may expand with telecom/Benin context but do not contradict the reference. "
            f"{lang_instr}\n\n"
            "--- REFERENCE ---\n"
            f"{reference_block}\n"
            "--- END REFERENCE ---"
        )
    else:
        system_content = (
            "You are a financial analyst expert for Moov Benin. "
            f"{scope_instruction} "
            f"Include practical examples relevant to telecom/financial services in Benin. "
            f"{lang_instr}"
        )

    if definition_only and term_entry:
        user_prompt = f"Define {term_entry['term']}."
    else:
        user_prompt = question

    yield from llm.stream_tokens([
        SystemMessage(content=system_content),
        HumanMessage(content=user_prompt),
    ])


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

_conversation_history: dict[str, list[str]] = {}


def evict_graph(session_id: str) -> None:
    _graph_cache.pop(session_id, None)
    for key in list(_conversation_history.keys()):
        if key.startswith(f"db:{session_id}:"):
            _conversation_history.pop(key, None)


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

async def run_db_agent(
    session_id: str,
    message: str,
    conversation_id: str = "default",
    model: str | None = None,
    language: str = "en",
) -> dict:
    import asyncio
    from app.agents.semantic_cache import semantic_cache
    from app.agents.schema_retriever import embed_text

    t_start       = time.monotonic()
    thread_id     = f"db:{session_id}:{conversation_id}"
    history_turns = _conversation_history.get(thread_id, [])
    history_text  = "\n".join(history_turns[-6:])

    # ─────────────────────────────────────────────────────────────
    # Pre-check: Classify question intent using LLM
    # ─────────────────────────────────────────────────────────────
    llm = _make_llm(model)
    classification = _classify_question_intent(llm, message)

    if classification == "definition":
        answer = _get_definition_answer(message, llm, language=language)
        log.info("Definition question detected — returning explanation")
        history_turns.append(f"Q: {message}\nA: {answer}")
        _conversation_history[thread_id] = history_turns
        inference_time = round(time.monotonic() - t_start, 2)
        return {"answer": answer, "charts": [], "inference_time": inference_time}

    if classification == "out_of_scope":
        answer = _get_out_of_scope_answer(message, language=language)
        log.info("Out-of-scope question detected — returning scope clarification")
        history_turns.append(f"Q: {message}\nA: {answer}")
        _conversation_history[thread_id] = history_turns
        inference_time = round(time.monotonic() - t_start, 2)
        return {"answer": answer, "charts": [], "inference_time": inference_time}

    definition_prefix = ""
    if classification == "both":
        log.info("Combined question detected — answering definition then querying data")
        definition_prefix = _get_definition_answer(message, llm, language=language, definition_only=True)

    # ─────────────────────────────────────────────────────────────
    # Semantic cache check (skip for definition-only questions)
    # ─────────────────────────────────────────────────────────────
    q_embedding = await asyncio.to_thread(embed_text, message)
    cached = semantic_cache.get(message, q_embedding)
    if cached is not None:
            cached_answer = cached["answer"]
            if definition_prefix:
                cached_answer = definition_prefix + "\n\n---\n\n" + cached_answer
            history_turns.append(f"User: {message}\nAnswer (cached): {cached_answer[:300]}")
            _conversation_history[thread_id] = history_turns
            return {
                "answer":         cached_answer,
                "charts":         [],
                "inference_time": round(time.monotonic() - t_start, 2),
                "sql":            cached["sql"],
                "tables_queried": sorted(_referenced_tables(cached["sql"])) if cached["sql"] else [],
                "row_count":      len(cached["rows"]),
                "cache_hit":      True,
            }

    # ─────────────────────────────────────────────────────────────
    # Proceed with normal SQL pipeline for data queries
    # ─────────────────────────────────────────────────────────────
    pipeline = _get_db_pipeline(model)

    log.info("=== run_db_agent  session=%s  conv=%s  model=%s  lang=%s ===", session_id, conversation_id, model or settings.OLLAMA_MODEL, language)
    log.info("Question: %s", _clip(message))

    state: DbPipelineState = {
        "question":         message,
        "history":          history_text,
        "language":         language,
        "target_db":        "",
        "snapshot_label":   "",
        "retrieved_schema": "",
        "allowed_tables":   [],
        "sql":              "",
        "syntax_error":     "",
        "table_error":      "",
        "semantic_error":   "",
        "error_history":    [],
        "column_facts":     "",
        "critic_feedback":  "",
        "retry_count":      0,
        "sql_error":        "",
        "rows":             [],
        "cols":             [],
        "answer":           "",
        "chart_specs":      [],
    }

    result = await asyncio.to_thread(pipeline.invoke, state)
    inference_time = round(time.monotonic() - t_start, 2)
    log.info("Pipeline finished in %.1fs", inference_time)

    answer      = result.get("answer") or "I was unable to generate a response."
    chart_specs = result.get("chart_specs", [])
    sql         = result.get("sql", "")
    tables_queried = sorted(_referenced_tables(sql)) if sql else []
    rows        = result.get("rows", [])
    cols        = result.get("cols", [])
    row_count   = len(rows)

    # Store in semantic cache if pipeline succeeded (has SQL + non-error answer)
    error_phrases = ("unable to generate", "could not", "error", "failed")
    pipeline_ok = bool(sql) and not any(p in answer.lower() for p in error_phrases)
    if pipeline_ok:
        semantic_cache.set(message, sql, rows, cols, answer, embedding=q_embedding)

    if definition_prefix:
        answer = definition_prefix + "\n\n---\n\n" + answer

    history_entry = f"User: {message}"
    if sql:
        history_entry += f"\nSQL:\n{sql}"
    history_entry += f"\nAnswer: {answer[:300]}"
    history_turns.append(history_entry)
    _conversation_history[thread_id] = history_turns
    return {
        "answer": answer,
        "charts": chart_specs,
        "inference_time": inference_time,
        "sql": sql,
        "tables_queried": tables_queried,
        "row_count": row_count,
        "cache_hit": False,
    }


async def run_db_agent_stream(
    session_id: str,
    message: str,
    conversation_id: str = "default",
    model: str | None = None,
    language: str = "en",
):
    """
    Async generator that yields SSE-formatted strings for streaming chat.
    Events: progress, token, charts, done, error
    """
    import asyncio, json as _json
    from app.agents.semantic_cache import semantic_cache
    from app.agents.schema_retriever import embed_text

    def _sse(event: str, data) -> str:
        return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

    t_start   = time.monotonic()
    thread_id = f"db:{session_id}:{conversation_id}"
    history_turns = _conversation_history.get(thread_id, [])
    history_text  = "\n".join(history_turns[-6:])

    # ── Semantic cache check ──────────────────────────────────────────────────
    q_embedding = await asyncio.to_thread(embed_text, message)
    cached = semantic_cache.get(message, q_embedding)
    if cached is not None:
            yield _sse("progress", {"message": "Cache hit — returning cached result", "step": "cache"})
            cached_answer = cached["answer"]
            yield _sse("token", {"text": cached_answer})
            history_turns.append(f"User: {message}\nAnswer (cached): {cached_answer[:300]}")
            _conversation_history[thread_id] = history_turns
            yield _sse("done", {
                "inference_time": round(time.monotonic() - t_start, 2),
                "charts":         [],
                "sql":            cached["sql"],
                "tables_queried": sorted(_referenced_tables(cached["sql"])) if cached["sql"] else [],
                "row_count":      len(cached["rows"]),
                "cache_hit":      True,
            })
            return

    yield _sse("progress", {"message": "Classifying question…", "step": "classify"})

    llm = _make_llm(model)
    classification = await asyncio.to_thread(_classify_question_intent, llm, message)

    definition_prefix = ""
    if classification == "definition":
        yield _sse("progress", {"message": "Generating definition…", "step": "define"})
        full_answer = ""
        try:
            for token in _get_definition_tokens(message, llm, language=language):
                full_answer += token
                yield _sse("token", {"text": token})
        except Exception as exc:
            log.error("Definition stream failed: %s", exc)
            full_answer = _get_definition_answer(message, llm, language=language)
            yield _sse("token", {"text": full_answer})
        history_turns.append(f"Q: {message}\nA: {full_answer}")
        _conversation_history[thread_id] = history_turns
        inference_time = round(time.monotonic() - t_start, 2)
        yield _sse("done", {"inference_time": inference_time, "charts": []})
        return

    if classification == "out_of_scope":
        yield _sse("progress", {"message": "Checking scope…", "step": "scope"})
        full_answer = _get_out_of_scope_answer(message, language=language)
        yield _sse("token", {"text": full_answer})
        history_turns.append(f"Q: {message}\nA: {full_answer}")
        _conversation_history[thread_id] = history_turns
        inference_time = round(time.monotonic() - t_start, 2)
        yield _sse("done", {"inference_time": inference_time, "charts": []})
        return

    if classification == "both":
        yield _sse("progress", {"message": "Generating definition…", "step": "define"})
        try:
            for token in _get_definition_tokens(message, llm, language=language, definition_only=True):
                definition_prefix += token
                yield _sse("token", {"text": token})
        except Exception as exc:
            log.error("Definition stream failed: %s", exc)
            definition_prefix = _get_definition_answer(message, llm, language=language, definition_only=True)
            yield _sse("token", {"text": definition_prefix})
        yield _sse("token", {"text": "\n\n---\n\n"})

    elif classification == "data_query":
        # If the question mentions a known financial term, add a brief intro to prefix the answer.
        # We do NOT emit this as an early token — it will be streamed with the answer.
        term_entry = _lookup_term(message)
        if term_entry:
            lang_key = f"definition_{language}" if language in ("en", "fr") else "definition_en"
            full_def = term_entry.get(lang_key) or term_entry.get("definition_en", "")
            first_sentence = full_def.split(".")[0].strip() + "." if "." in full_def else full_def
            definition_prefix = f"**{term_entry['term']}**: {first_sentence}\n\n---\n\n"
            log.info("data_query: will prepend term intro for %s", term_entry["term"])

    yield _sse("progress", {"message": "Retrieving schema…", "step": "schema"})
    schema_state = await asyncio.to_thread(retrieve_schema, {
        "question": message, "history": history_text, "language": language,
        "retrieved_schema": "", "allowed_tables": [], "sql": "",
        "syntax_error": "", "table_error": "", "semantic_error": "",
        "error_history": [], "column_facts": "", "critic_feedback": "",
        "retry_count": 0, "sql_error": "", "rows": [], "cols": [],
        "answer": "", "chart_specs": [],
    })
    allowed_tables = schema_state.get("allowed_tables", [])
    yield _sse("progress", {
        "message": f"Found {len(allowed_tables)} relevant table(s)",
        "step": "schema", "status": "done",
        "tables": allowed_tables,
    })

    yield _sse("progress", {"message": "Writing SQL…", "step": "write_sql"})
    write_node = _make_write_sql_node(llm)
    sql_state = await asyncio.to_thread(write_node, {**schema_state, "question": message,
        "history": history_text, "language": language,
        "syntax_error": "", "table_error": "", "semantic_error": "",
        "error_history": [], "column_facts": "", "critic_feedback": "",
        "retry_count": 0, "sql_error": "", "rows": [], "cols": [],
        "answer": "", "chart_specs": [],
    })
    current_state = {
        "question": message, "history": history_text, "language": language,
        "retry_count": 0, "error_history": [], "chart_specs": [], "answer": "",
        **schema_state, **sql_state,
    }
    draft_sql = (current_state.get("sql") or "").strip()
    if draft_sql:
        yield _sse("progress", {
            "message": "SQL drafted", "step": "write_sql", "status": "done",
            "sql": draft_sql,
        })

    # Validate + repair loop (synchronous, same as pipeline)
    yield _sse("progress", {"message": "Validating query…", "step": "validate"})
    for _ in range(_MAX_RETRIES + 2):
        vs = await asyncio.to_thread(validate_syntax, current_state)
        current_state = {**current_state, **vs}
        if not current_state.get("syntax_error"):
            vt = await asyncio.to_thread(validate_tables, current_state)
            current_state = {**current_state, **vt}
            if not current_state.get("table_error"):
                vsem = await asyncio.to_thread(validate_semantic, current_state)
                current_state = {**current_state, **vsem}
                if not current_state.get("semantic_error"):
                    yield _sse("progress", {"message": "Query validated", "step": "validate", "status": "done"})
                    break
        err = _any_validation_error(current_state)
        if not err or current_state.get("retry_count", 0) >= _MAX_RETRIES:
            break
        attempt = current_state.get("retry_count", 0) + 1
        yield _sse("progress", {
            "message": f"Repairing SQL (attempt {attempt})…",
            "step": "repair", "attempt": attempt,
            "error": (err or "")[:200],
        })
        critique_node = _make_critique_sql_node(llm)
        cr = await asyncio.to_thread(critique_node, current_state)
        current_state = {**current_state, **cr}
        repaired_sql = (current_state.get("sql") or "").strip()
        if repaired_sql:
            yield _sse("progress", {
                "message": f"SQL repaired (attempt {attempt})",
                "step": "repair", "status": "done",
                "sql": repaired_sql,
            })

    yield _sse("progress", {"message": "Executing query…", "step": "execute"})
    exec_state = await asyncio.to_thread(execute_sql, current_state)
    current_state = {**current_state, **exec_state}
    exec_rows = len(current_state.get("rows", []))
    yield _sse("progress", {
        "message": f"Query returned {exec_rows} row(s)",
        "step": "execute", "status": "done",
    })

    rows = current_state.get("rows", [])
    cols = current_state.get("cols", [])

    # Build chart spec and facts
    chart_spec = _build_chart_spec(cols, rows, message) if rows else None
    facts      = _compute_data_facts(cols, rows) if rows else ""
    source     = _source_context(current_state.get("sql",""), rows, cols)
    lang_instr = _LANG_INSTRUCTIONS.get(language, _LANG_INSTRUCTIONS["en"])
    fmt_system = _FMT_SYSTEM.format(language_instruction=lang_instr)
    table      = _build_ascii_table(rows, cols) if rows else ""

    snapshot_label = current_state.get("snapshot_label", "")
    snapshot_ctx   = f"Database snapshot: {snapshot_label}\n" if snapshot_label else ""
    col_ctx        = f"Columns in result: {', '.join(cols)}\n" if cols else ""
    quarterly_ctx  = _detect_quarterly_availability(rows, cols)
    quarterly_line = f"{quarterly_ctx}\n" if quarterly_ctx else ""

    yield _sse("progress", {"message": "Formatting answer…", "step": "format"})

    full_answer = ""
    # Stream the definition prefix first so the user sees it immediately
    if definition_prefix:
        yield _sse("token", {"text": definition_prefix})
    try:
        fmt_messages = [
            SystemMessage(content=fmt_system),
            HumanMessage(content=(
                f"User question: {message}\n\n"
                f"{col_ctx}"
                f"{quarterly_line}"
                f"{snapshot_ctx}"
                f"{facts}\n\n"
                f"Query source (for the **Data Source** line): {source}\n\n"
                f"Result ({len(rows)} row(s), first {_LLM_TABLE_ROW_CAP} shown):\n{table}"
            )),
        ]
        for token in llm.stream_tokens(fmt_messages):
            full_answer += token
            yield _sse("token", {"text": token})
    except Exception as exc:
        log.error("Streaming format_answer failed: %s", exc)
        fallback = _format_fallback(rows, cols, message, facts, source, chart_spec, language)
        fallback_text = fallback["answer"]
        full_answer = fallback_text
        yield _sse("token", {"text": fallback_text})

    full_answer = (definition_prefix + full_answer) if definition_prefix else full_answer
    sql = current_state.get("sql", "")
    tables_queried = sorted(_referenced_tables(sql)) if sql else []
    row_count = len(rows)

    # Store in semantic cache if pipeline succeeded
    error_phrases = ("unable to generate", "could not", "error", "failed")
    pipeline_ok = bool(sql) and not any(p in full_answer.lower() for p in error_phrases)
    if pipeline_ok:
        semantic_cache.set(message, sql, rows, cols, full_answer, embedding=q_embedding)

    history_entry = f"User: {message}"
    if sql:
        history_entry += f"\nSQL:\n{sql}"
    history_entry += f"\nAnswer: {full_answer[:300]}"
    history_turns.append(history_entry)
    _conversation_history[thread_id] = history_turns
    inference_time = round(time.monotonic() - t_start, 2)
    yield _sse("done", {
        "inference_time": inference_time,
        "charts": [chart_spec] if chart_spec else [],
        "sql": sql,
        "tables_queried": tables_queried,
        "row_count": row_count,
        "cache_hit": False,
    })


async def run_agent(
    session_id: str,
    parsed_data: dict,
    message: str,
    conversation_id: str = "default",
    model: str | None = None,
    language: str = "en",
) -> dict:
    t_start   = time.monotonic()
    graph     = get_or_create_graph(session_id, parsed_data, model=model)
    thread_id = f"{session_id}:{conversation_id}"
    config    = {"configurable": {"thread_id": thread_id}}

    lang_instr = _LANG_INSTRUCTIONS.get(language, _LANG_INSTRUCTIONS["en"])
    augmented  = f"[{lang_instr}]\n{message}"

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=augmented)]},
        config=config,
    )

    inference_time = round(time.monotonic() - t_start, 2)
    messages = result.get("messages", [])

    chart_specs = []
    for msg in messages:
        if getattr(msg, "name", None) == "generate_chart_spec":
            try:
                chart_specs.append(json.loads(msg.content))
            except Exception:
                pass

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return {"answer": msg.content, "inference_time": inference_time, "charts": chart_specs}

    return {"answer": "I was unable to generate a response. Please try again.", "inference_time": inference_time, "charts": chart_specs}


async def run_agent_stream(
    session_id: str,
    parsed_data: dict,
    message: str,
    conversation_id: str = "default",
    model: str | None = None,
    language: str = "en",
):
    """
    Async generator that yields SSE-formatted strings for streaming file-session chat.
    Events: progress, token, done, error
    """
    import json as _j

    def _sse(event, data):
        return f"event: {event}\ndata: {_j.dumps(data, ensure_ascii=False)}\n\n"

    t_start = time.monotonic()
    try:
        graph = get_or_create_graph(session_id, parsed_data, model=model)
        thread_id = f"{session_id}:{conversation_id}"
        config = {"configurable": {"thread_id": thread_id}}

        lang_instr = _LANG_INSTRUCTIONS.get(language, _LANG_INSTRUCTIONS["en"])
        augmented = f"[{lang_instr}]\n{message}"

        chart_specs = []
        final_answer = ""

        async for delta in graph.astream(
            {"messages": [HumanMessage(content=augmented)]},
            config=config,
            stream_mode="updates",
        ):
            for node_msgs in delta.values():
                msgs = node_msgs.get("messages", []) if isinstance(node_msgs, dict) else []
                for msg in msgs:
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                            yield _sse("progress", {"message": f"Calling {tool_name.replace('_', ' ')}…"})
                    elif getattr(msg, "name", None) == "generate_chart_spec":
                        try:
                            chart_specs.append(_j.loads(msg.content))
                        except Exception:
                            pass
                    elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                        final_answer = msg.content

        yield _sse("token", {"text": final_answer})
        inference_time = round(time.monotonic() - t_start, 2)
        yield _sse("done", {"inference_time": inference_time, "charts": chart_specs})
    except Exception as exc:
        log.error("run_agent_stream failed: %s", exc)
        yield _sse("error", {"detail": str(exc)})
