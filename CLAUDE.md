# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

TBG AI Copilot — an agentic FastAPI backend for the **Tableau de Bord de Gestion (TBG)** financial reports of Moov Benin. It supports two modes:

1. **File-upload mode**: Users upload a TBG Excel file; the agent answers questions about parsed metrics.
2. **DB mode**: The agent queries a PostgreSQL database (Digiwise) directly using LLM-generated SQL through a multi-step validation pipeline.

The LLM layer uses **Ollama** (local or cloud) via `langchain-ollama`. The agent orchestration uses **LangGraph**.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Check DB health (requires running Postgres)
curl http://localhost:8000/api/v1/db/health

# Spin up local Postgres from SQL dump
./docker_postgres.sh up       # create container + import dump
./docker_postgres.sh connect  # open psql shell
./docker_postgres.sh stop     # stop container
./docker_postgres.sh reset    # destroy and recreate
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|---|---|
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | PostgreSQL connection (defaults point to remote Digiwise instance) |
| `OLLAMA_BASE_URL` | Ollama endpoint (`http://localhost:11434` for local, `https://api.ollama.com` for cloud) |
| `OLLAMA_MODEL` | Model to use (e.g. `devstral-2:123b`, `llama3`) |
| `OLLAMA_API_KEY` | Set for Ollama Cloud; leave empty for local |
| `OLLAMA_EMBEDDING_MODEL` | Optional dedicated embedding model (e.g. `nomic-embed-text`); falls back to `OLLAMA_MODEL` |
| `LANGSMITH_API_KEY` | Optional LangSmith tracing |

Settings are loaded by `app/config/settings.py` via `pydantic-settings`. The loader looks for `.env` at the repo root first, then falls back to `tb/.env`.

## Architecture

### Request Flow

```
HTTP request
    └─ FastAPI router (app/api/routes.py)
         ├─ File-upload endpoints → run_agent()
         │       └─ LangGraph ReAct agent (LangChain tools in app/agents/tools.py)
         └─ DB endpoints → run_db_agent() / run_db_agent_stream()
                 └─ Custom LangGraph pipeline (app/agents/graph.py)
```

### DB Agent Pipeline (app/agents/graph.py)

The DB pipeline is a sequential LangGraph `StateGraph` with retry loops:

1. `retrieve_schema` — RAG: picks 2–4 relevant tables via cosine similarity (falls back to keyword overlap)
2. `write_sql` — Writer LLM generates SQL (strict: SQL output only)
3. `validate_syntax` — `sqlglot` parses SQL without hitting the DB
4. `validate_tables` — whitelists table names against the retrieved schema; blocks hallucinations
5. `validate_semantic` — runs `EXPLAIN` against the real DB to catch column/type errors
6. `critique_sql` — Critic LLM repairs SQL using the full error history; loops back to `validate_syntax` (max 3 retries)
7. `execute_sql` — runs the validated `SELECT`
8. `format_answer` — formats rows into natural language; SQL is never exposed to the user

The `OllamaLLM` class (in `graph.py`) is a custom wrapper around the `ollama.Client` that implements the LangChain `invoke()` interface.

### Schema RAG (app/agents/schema_retriever.py)

`SchemaRetriever` is a thread-safe singleton that:
- Embeds table descriptions (from `schema_descriptions.yaml` + column names) using Ollama
- Caches vectors in memory; falls back to keyword scoring when embeddings fail
- Applies domain-keyword hints (e.g. "ARPU", "capex", "cashflow") to force-include relevant tables
- Expands top-K results by one FK hop to ensure JOIN-connected tables are included
- Blocklists system/operational tables (auth, config, upload tracking)

### File-Upload Agent (app/agents/tools.py)

10 LangChain `@tool` functions are built as closures over the session's parsed data (`build_tools()`). These cover the 5 business scenarios:

| Scenario | Tools |
|---|---|
| 1. Metric Q&A | `query_metric`, `get_report_summary`, `list_metrics`, `list_available_sheets` |
| 2. Root-cause variance | `analyze_variance`, `get_metric_trend` |
| 3. Period comparison | `compare_periods`, `compare_across_all_sheets` |
| 4. Chart generation | `generate_chart_spec` (returns JSON spec for Recharts/Chart.js) |
| 5. Anomaly alerts | `check_all_alerts` (uses rules from `app/thresholds.json`) |

Sheet aliases (e.g. `"p&l"` → `"pnl_conso"`) are resolved in `_resolve_sheet()`.

### Session Management

Sessions are stored in-memory in `routes.py` as `_sessions: dict[str, dict]`. The DB agent uses a fixed `"db-global"` session ID; individual conversations are isolated by `conversation_id` through LangGraph's `MemorySaver` checkpointer.

### Excel Parser (app/parsers/excel_parser.py)

Parses TBG `.xlsx` files whose column layout varies by month (later months insert `ACTU1/2/3` columns). The parser reads the actual date-header row to build a `value_type → column_index` map, extracting: `reel`, `budget`, `ecart_budget`, `n1_reel`, `evol_pct`.

Target sheets are defined in `SHEETS_OF_INTEREST`; metric codes match `_CODE_RE` pattern (`PL1`, `CA10`, etc.).

## Key Data Files

- `app/thresholds.json` — alert threshold rules per metric (warning/critical %, direction)
- `financial_terms.json` — bilingual (EN/FR) glossary of telecom/financial KPIs
- `model_list.json` — available Ollama model list served at `GET /api/v1/models`
- `app/agents/schema_descriptions.yaml` — human-readable table descriptions + French aliases for RAG embedding
- `digiwise_schema.sql` — PostgreSQL schema DDL (reference only; use `docker_postgres.sh` to load the full dump)

## API Endpoints Summary

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/sessions` | Upload single TBG Excel, create session |
| `POST` | `/api/v1/sessions/compare` | Upload two Excel files for period comparison |
| `POST` | `/api/v1/sessions/{id}/chat` | Chat against uploaded file data |
| `POST` | `/api/v1/db/chat` | Chat against PostgreSQL (streaming: `/db/chat/stream`) |
| `GET` | `/api/v1/db/health` | Check DB connectivity + row counts |
| `GET` | `/api/v1/db/schema` | Full schema introspection as JSON |
| `GET` | `/api/v1/db/schema/text` | Schema as plain text (same text injected into agent) |
| `GET` | `/api/v1/models` | List available Ollama models |
| `GET` | `/api/v1/health` | App health + active session count |

The `ChatRequest` model accepts `language: "en" | "fr"` to control response language, and an optional `model` override per request.
