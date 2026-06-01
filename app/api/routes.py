"""
FastAPI route definitions for the TBG AI Copilot.

File-based endpoints (Excel upload)
------------------------------------
POST   /api/v1/sessions                      Upload one TBG Excel file, create session
POST   /api/v1/sessions/compare              Upload two files for period comparison
POST   /api/v1/sessions/{id}/chat            Chat with parsed data
GET    /api/v1/sessions/{id}                 Session metadata
GET    /api/v1/sessions/{id}/sheets          List available sheets
GET    /api/v1/sessions/{id}/metrics/{sheet} List metrics for a sheet
DELETE /api/v1/sessions/{id}                 Delete session
GET    /api/v1/health                        Health check

Database-backed endpoints (PostgreSQL)
---------------------------------------
POST   /api/v1/db/chat                       Chat directly against the Postgres DB
GET    /api/v1/db/health                     Check DB connectivity
GET    /api/v1/db/sheets                     List sheets available in the DB
GET    /api/v1/db/metrics/{sheet}            List metrics for a sheet (from DB)
"""
from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import asyncio as _asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from app.auth.deps import get_optional_user
from app.db.auth_store import create_conversation, save_message, touch_conversation

_MODEL_LIST_PATH = Path(__file__).resolve().parent.parent.parent / "model_list.json"
_THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "app" / "thresholds.json"

from app.agents.graph import evict_graph, run_agent, run_agent_stream, run_db_agent, run_db_agent_stream
from app.agents.tools import run_alerts_check
from app.db.connection import execute as db_execute, ping as db_ping
from app.db.schema_inspector import build_schema_context, inspect_schema, get_views
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    CompareUploadResponse,
    FieldMapResponse,
    FieldMapSheet,
    FieldMapSection,
    FieldMapMetric,
    HealthResponse,
    MetricListItem,
    MetricListResponse,
    SessionInfo,
    UploadResponse,
)
from app.parsers.excel_parser import parse_tbg_file

router = APIRouter(prefix="/api/v1")

# ---------------------------------------------------------------------------
# In-memory session store  { session_id -> SessionData }
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}


def _require_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found or expired.",
        )
    return session


async def _save_and_parse(upload: UploadFile) -> tuple[str, dict]:
    """Save uploaded file to a temp path and parse it. Returns (path, parsed_data)."""
    suffix = Path(upload.filename or "file.xlsx").suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload.read()
        tmp.write(content)
        tmp_path = tmp.name
    parsed = parse_tbg_file(tmp_path)
    return tmp_path, parsed


# ---------------------------------------------------------------------------
# POST /api/v1/sessions  — single file upload
# ---------------------------------------------------------------------------
@router.post("/sessions", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def create_session(file: Annotated[UploadFile, File(description="TBG Excel file (.xlsx)")]):
    """
    Upload a single TBG Excel file.  Returns a session_id used for all
    subsequent chat and metadata requests.
    """
    if not (upload_filename := file.filename or ""):
        raise HTTPException(status_code=400, detail="File has no filename.")

    _, parsed = await _save_and_parse(file)

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
                thresholds = json.load(_tf)
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


# ---------------------------------------------------------------------------
# POST /api/v1/sessions/compare  — two-file upload for period comparison
# ---------------------------------------------------------------------------
@router.post(
    "/sessions/compare",
    response_model=CompareUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_comparison_session(
    file1: Annotated[UploadFile, File(description="First TBG Excel file (earlier period)")],
    file2: Annotated[UploadFile, File(description="Second TBG Excel file (later period)")],
):
    """
    Upload two TBG Excel files for cross-period comparison (Scenario 3).
    Both files are merged into a single parsed dataset keyed by period.
    """
    _, parsed1 = await _save_and_parse(file1)
    _, parsed2 = await _save_and_parse(file2)

    if not parsed1.get("sheets") or not parsed2.get("sheets"):
        raise HTTPException(status_code=422, detail="One or both files could not be parsed.")

    # Merge sheets: for each sheet key, merge the metrics dicts
    merged_sheets: dict = {}
    all_sheet_keys = set(parsed1["sheets"]) | set(parsed2["sheets"])

    for sk in all_sheet_keys:
        s1 = parsed1["sheets"].get(sk, {"periods": [], "metrics": {}})
        s2 = parsed2["sheets"].get(sk, {"periods": [], "metrics": {}})

        merged_metrics: dict = {}
        all_metric_keys = set(s1["metrics"]) | set(s2["metrics"])

        for mk in all_metric_keys:
            m1 = s1["metrics"].get(mk)
            m2 = s2["metrics"].get(mk)
            if m1 and m2:
                merged_values = {**m1["values"], **m2["values"]}
                merged_metrics[mk] = {**m1, "values": merged_values}
            else:
                merged_metrics[mk] = (m1 or m2)

        all_periods = sorted(set(s1.get("periods", [])) | set(s2.get("periods", [])))
        merged_sheets[sk] = {"periods": all_periods, "metrics": merged_metrics}

    all_periods_global = sorted(
        set(parsed1.get("all_periods", [])) | set(parsed2.get("all_periods", []))
    )

    merged_data = {
        "file": f"{file1.filename} + {file2.filename}",
        "files": [file1.filename, file2.filename],
        "sheets": merged_sheets,
        "all_periods": all_periods_global,
    }

    latest_period_global = all_periods_global[-1] if all_periods_global else ""
    compare_alerts: list[dict] = []
    if latest_period_global:
        try:
            with open(_THRESHOLDS_PATH) as _tf:
                thresholds = json.load(_tf)
            compare_alerts = run_alerts_check(merged_data, thresholds, latest_period_global)
        except Exception:
            pass

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id": session_id,
        "files": [file1.filename, file2.filename],
        "parsed_data": merged_data,
        "alerts": compare_alerts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return CompareUploadResponse(
        session_id=session_id,
        file1_name=file1.filename or "",
        file2_name=file2.filename or "",
        sheets_parsed=list(merged_sheets.keys()),
        periods_available=all_periods_global,
        message=f"Comparison session created. {len(merged_sheets)} sheets, "
                f"{len(all_periods_global)} periods merged.",
        alerts=compare_alerts,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/sessions/{session_id}/chat
# ---------------------------------------------------------------------------
@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, request: ChatRequest):
    """
    Send a message to the TBG AI Copilot for the given session.

    Covers all five scenarios:
    1. Natural language Q&A about metrics
    2. Root-cause variance analysis
    3. Period comparison
    4. Chart specification generation
    5. Anomaly / alert detection
    """
    session = _require_session(session_id)
    parsed_data = session["parsed_data"]

    hints = [h.model_dump() for h in request.metric_hints] if request.metric_hints else None
    try:
        agent_result = await run_agent(
            session_id=session_id,
            parsed_data=parsed_data,
            message=request.message,
            conversation_id=request.conversation_id,
            model=request.model,
            language=request.language,
            metric_hints=hints,
        )
        print(f"Agent response for session {session_id}:\n{agent_result['answer']}\n")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(exc)}",
        ) from exc

    return ChatResponse(
        response=agent_result["answer"],
        conversation_id=request.conversation_id,
        session_id=session_id,
        inference_time=agent_result.get("inference_time"),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}
# ---------------------------------------------------------------------------
@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Return metadata for the given session."""
    session = _require_session(session_id)
    parsed = session["parsed_data"]
    return SessionInfo(
        session_id=session_id,
        files=session.get("files", []),
        sheets=list(parsed.get("sheets", {}).keys()),
        periods=parsed.get("all_periods", []),
        created_at=session.get("created_at", ""),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}/sheets
# ---------------------------------------------------------------------------
@router.get("/sessions/{session_id}/sheets")
async def list_sheets(session_id: str):
    """List all parsed report sheets and their period coverage."""
    session = _require_session(session_id)
    parsed = session["parsed_data"]
    result = {}
    for sk, sd in parsed.get("sheets", {}).items():
        result[sk] = {
            "periods": sd.get("periods", []),
            "metric_count": len(sd.get("metrics", {})),
        }
    return {"session_id": session_id, "sheets": result}


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}/metrics/{sheet_name}
# ---------------------------------------------------------------------------
@router.get("/sessions/{session_id}/metrics/{sheet_name}", response_model=MetricListResponse)
async def list_metrics(session_id: str, sheet_name: str):
    """List all metrics available in a specific sheet for this session."""
    session = _require_session(session_id)
    parsed = session["parsed_data"]
    sheet = parsed.get("sheets", {}).get(sheet_name)
    if not sheet:
        available = list(parsed.get("sheets", {}).keys())
        raise HTTPException(
            status_code=404,
            detail=f"Sheet '{sheet_name}' not found. Available: {available}",
        )

    items = [
        MetricListItem(
            code=m.get("code"),
            label=m["label"],
            section=m.get("section"),
            periods_available=sorted(m["values"].keys()),
        )
        for m in sheet["metrics"].values()
    ]
    return MetricListResponse(sheet=sheet_name, metrics=items)


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}/field-map
# ---------------------------------------------------------------------------

_SHEET_DISPLAY_NAMES: dict[str, str] = {
    "pnl_conso":        "P&L Consolidé",
    "ca_mobile":        "CA Mobile",
    "opex_consolides":  "Opex Consolidés",
    "capex_consolides": "Capex Consolidés",
    "mobile_money":     "Mobile Money",
    "parc_mobile":      "Parc Mobile",
    "marge_mobile":     "Marge Mobile",
    "trafic_mobile":    "Trafic Mobile",
    "data_mobile":      "Data Mobile",
    "cash_conso":       "Cash Consolidé",
    "marge_fixe":       "Marge Fixe",
    "ca_fixe":          "CA Fixe",
}


@router.get("/sessions/{session_id}/field-map", response_model=FieldMapResponse)
async def get_field_map(
    session_id: str,
    optional_user: dict | None = Depends(get_optional_user),
):
    """
    Return a structured mapping of all sheets → sub-sections → metrics,
    annotated with which labels appear more than once in a sheet.
    Used by the UI to let users select specific metrics before asking a question.
    Filtered by the authenticated user's role.
    """
    from collections import Counter, defaultdict
    from app.auth.policies import check_metric

    session = _require_session(session_id)
    parsed  = session["parsed_data"]
    user_role = (optional_user or {}).get("role", "viewer")

    sheets_out: list[FieldMapSheet] = []

    for sheet_key, sheet_data in parsed.get("sheets", {}).items():
        # Drop the entire sheet if the role can't see it
        allowed, _ = check_metric(user_role, sheet_key, None)
        if not allowed:
            continue
        metrics = sheet_data.get("metrics", {})

        # Find duplicate labels within this sheet
        label_counts = Counter(m["label"] for m in metrics.values())
        duplicate_labels = [lbl for lbl, cnt in label_counts.items() if cnt > 1]

        # Group metrics by section, preserving insertion order — and drop
        # any section the user's role can't see
        sections_map: dict[str, list[FieldMapMetric]] = defaultdict(list)
        for m in metrics.values():
            sec = m.get("section") or ""
            ok, _ = check_metric(user_role, sheet_key, sec)
            if not ok:
                continue
            sections_map[sec].append(FieldMapMetric(
                code=m.get("code"),
                label=m["label"],
                section=sec or None,
            ))

        sections_out = [
            FieldMapSection(
                section=sec_name or "General",
                metrics=sec_metrics,
            )
            for sec_name, sec_metrics in sections_map.items()
        ]

        sheets_out.append(FieldMapSheet(
            sheet_key=sheet_key,
            display_name=_SHEET_DISPLAY_NAMES.get(sheet_key, sheet_key.replace("_", " ").title()),
            sections=sections_out,
            duplicate_labels=duplicate_labels,
        ))

    return FieldMapResponse(session_id=session_id, sheets=sheets_out)


# ---------------------------------------------------------------------------
# DELETE /api/v1/sessions/{session_id}
# ---------------------------------------------------------------------------
@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """Delete a session and free its memory."""
    _require_session(session_id)
    _sessions.pop(session_id, None)
    evict_graph(session_id)


# ---------------------------------------------------------------------------
# GET /api/v1/models
# ---------------------------------------------------------------------------
@router.get("/models")
async def list_models():
    """Return the list of available inference models from model_list.json."""
    try:
        with open(_MODEL_LIST_PATH) as f:
            data = json.load(f)
        return {"models": data.get("models", [])}
    except FileNotFoundError:
        return {"models": []}


# ---------------------------------------------------------------------------
# GET /api/v1/settings  — return key presence (never the full key)
# POST /api/v1/settings — update runtime settings (API key, etc.)
# ---------------------------------------------------------------------------
@router.get("/settings")
async def get_settings():
    """Return current runtime settings (key presence only — never the full key)."""
    from app.config.settings import settings
    return {"has_api_key": bool(settings.OLLAMA_API_KEY)}


@router.post("/settings")
async def update_settings(body: dict):
    """Update runtime settings in-memory. Changes apply immediately to all new requests."""
    from app.config.settings import settings
    if "ollama_api_key" in body:
        settings.OLLAMA_API_KEY = body["ollama_api_key"] or ""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    from app.config.settings import settings
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        active_sessions=len(_sessions),
    )


# ===========================================================================
#  DATABASE-BACKED ROUTES  /api/v1/db/*
# ===========================================================================

# ---------------------------------------------------------------------------
# POST /api/v1/db/chat
# ---------------------------------------------------------------------------
@router.post("/db/chat", response_model=ChatResponse)
async def db_chat(
    request: ChatRequest,
    http_request: Request,
    optional_user: dict | None = Depends(get_optional_user),
):
    """
    Chat with the TBG AI Copilot backed by PostgreSQL.
    No file upload needed — the agent queries the database directly.

    Use conversation_id to maintain multi-turn history.
    When authenticated, messages are persisted to the database.
    """
    from app.config.settings import request_api_key
    per_user_key = http_request.headers.get("X-Ollama-Api-Key", "").strip()
    if not per_user_key:
        raise HTTPException(status_code=403, detail="NO_API_KEY")
    _token = request_api_key.set(per_user_key)

    # ── Policy check (role-based access control) ───────────────────────────
    from app.auth.policies import check_question, policy_refusal_text
    from app.config.settings import request_user_role
    from app.db.auth_store import log_policy_block
    user_role = (optional_user or {}).get("role", "viewer")
    request_user_role.set(user_role)
    allowed, blocked_term = check_question(user_role, request.message)
    if not allowed:
        if optional_user:
            await _asyncio.to_thread(
                log_policy_block, optional_user["id"], user_role,
                request.message, blocked_term, "keyword block (db_chat)"
            )
        refusal = policy_refusal_text(blocked_term, user_role, language=request.language)
        return ChatResponse(
            response=refusal, conversation_id=request.conversation_id,
            charts=[], alerts=[], session_id="db-global",
            inference_time=0.0, conv_id=request.conv_id,
        )

    # Use a fixed "db" session so the graph is shared across all DB chats
    # but each conversation_id gets its own thread in MemorySaver.
    session_id = "db-global"

    # ── Persist user message if authenticated ──────────────────────────────
    conv_id: int | None = request.conv_id
    if optional_user:
        if conv_id is None:
            # Auto-create a conversation named after the first question
            title = request.message[:80]
            conv  = await _asyncio.to_thread(
                create_conversation, optional_user["id"], title, "db"
            )
            conv_id = conv["id"]
        await _asyncio.to_thread(save_message, conv_id, "user", request.message, {})

    try:
        result = await run_db_agent(
            session_id=session_id,
            message=request.message,
            conversation_id=request.conversation_id,
            model=request.model,
            language=request.language,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(exc)}",
        ) from exc

    response_text = result["answer"]
    charts_raw    = result.get("charts", [])
    charts        = []
    for c in charts_raw:
        try:
            from app.models.schemas import ChartSpec
            charts.append(ChartSpec(**c))
        except Exception:
            pass

    # ── Persist bot reply if authenticated ─────────────────────────────────
    if optional_user and conv_id is not None:
        bot_meta = {
            "sql":            result.get("sql"),
            "tables":         result.get("tables_queried"),
            "inference_time": result.get("inference_time"),
            "cache_hit":      result.get("cache_hit"),
        }
        await _asyncio.to_thread(save_message, conv_id, "bot", response_text, bot_meta)
        await _asyncio.to_thread(touch_conversation, conv_id)

    return ChatResponse(
        response=response_text,
        charts=charts,
        conversation_id=request.conversation_id,
        session_id=session_id,
        inference_time=result.get("inference_time"),
        sql=result.get("sql") or None,
        tables_queried=result.get("tables_queried", []),
        row_count=result.get("row_count"),
        cache_hit=result.get("cache_hit", False),
        conv_id=conv_id,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/sessions/{session_id}/chat/stream
# ---------------------------------------------------------------------------
@router.post("/sessions/{session_id}/chat/stream")
async def session_chat_stream(session_id: str, request: ChatRequest):
    """
    Streaming version of /sessions/{id}/chat using Server-Sent Events.
    """
    session = _require_session(session_id)
    parsed_data = session["parsed_data"]

    stream_hints = [h.model_dump() for h in request.metric_hints] if request.metric_hints else None

    async def event_generator():
        try:
            async for chunk in run_agent_stream(
                session_id=session_id,
                parsed_data=parsed_data,
                message=request.message,
                conversation_id=request.conversation_id,
                model=request.model,
                language=request.language,
                metric_hints=stream_hints,
            ):
                yield chunk
        except Exception as exc:
            import json as _j
            yield f"event: error\ndata: {_j.dumps({'detail': str(exc)})}\n\n"

    from fastapi.responses import StreamingResponse as _SR
    return _SR(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /api/v1/db/chat/stream
# ---------------------------------------------------------------------------
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

@router.post("/db/chat/stream")
async def db_chat_stream(
    request: ChatRequest,
    http_request: Request,
    optional_user: dict | None = Depends(get_optional_user),
):
    """
    Streaming version of /db/chat using Server-Sent Events.
    When authenticated, user and bot messages are persisted to the database
    and the conv_id is injected into the final 'done' SSE event.
    """
    import json as _json

    from app.config.settings import request_api_key
    per_user_key = http_request.headers.get("X-Ollama-Api-Key", "").strip()
    if not per_user_key:
        raise HTTPException(status_code=403, detail="NO_API_KEY")
    request_api_key.set(per_user_key)

    # ── Policy check ───────────────────────────────────────────────────────
    from app.auth.policies import check_question, policy_refusal_text
    from app.config.settings import request_user_role
    from app.db.auth_store import log_policy_block
    user_role = (optional_user or {}).get("role", "viewer")
    request_user_role.set(user_role)
    allowed, blocked_term = check_question(user_role, request.message)
    if not allowed:
        if optional_user:
            await _asyncio.to_thread(
                log_policy_block, optional_user["id"], user_role,
                request.message, blocked_term, "keyword block (db_chat_stream)"
            )
        refusal = policy_refusal_text(blocked_term, user_role, language=request.language)
        async def _refusal_gen():
            yield f"event: token\ndata: {_json.dumps({'text': refusal})}\n\n"
            yield f"event: done\ndata: {_json.dumps({'inference_time': 0.0, 'charts': []})}\n\n"
        return FastAPIStreamingResponse(_refusal_gen(), media_type="text/event-stream",
                                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    session_id = "db-global"

    # Resolve / create conversation before streaming starts
    conv_id: int | None = request.conv_id
    if optional_user:
        if conv_id is None:
            title = request.message[:80]
            conv  = await _asyncio.to_thread(
                create_conversation, optional_user["id"], title, "db"
            )
            conv_id = conv["id"]
        await _asyncio.to_thread(save_message, conv_id, "user", request.message, {})

    async def event_generator():
        stream_answer = []
        try:
            async for chunk in run_db_agent_stream(
                session_id=session_id,
                message=request.message,
                conversation_id=request.conversation_id,
                model=request.model,
                language=request.language,
            ):
                # Intercept token events to collect the full answer
                if chunk.startswith("event: token\n"):
                    try:
                        data_line = [l for l in chunk.split("\n") if l.startswith("data: ")][0]
                        tok_payload = _json.loads(data_line[6:])
                        stream_answer.append(tok_payload.get("text", ""))
                    except Exception:
                        pass
                    yield chunk

                elif chunk.startswith("event: done\n"):
                    # Inject conv_id into the done payload and persist bot message
                    try:
                        data_line = [l for l in chunk.split("\n") if l.startswith("data: ")][0]
                        done_payload = _json.loads(data_line[6:])
                    except Exception:
                        done_payload = {}

                    if optional_user and conv_id is not None:
                        full_answer = "".join(stream_answer)
                        bot_meta = {
                            "sql":            done_payload.get("sql"),
                            "tables":         done_payload.get("tables_queried"),
                            "inference_time": done_payload.get("inference_time"),
                            "cache_hit":      done_payload.get("cache_hit"),
                        }
                        try:
                            await _asyncio.to_thread(save_message, conv_id, "bot", full_answer, bot_meta)
                            await _asyncio.to_thread(touch_conversation, conv_id)
                        except Exception:
                            pass

                    done_payload["conv_id"] = conv_id
                    yield f"event: done\ndata: {_json.dumps(done_payload)}\n\n"

                else:
                    yield chunk

        except Exception as exc:
            yield f"event: error\ndata: {_json.dumps({'detail': str(exc)})}\n\n"

    return FastAPIStreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/v1/db/health
# ---------------------------------------------------------------------------
@router.get("/db/health")
async def db_health():
    """Check that the PostgreSQL database is reachable."""
    import asyncio
    reachable = await asyncio.to_thread(db_ping)
    if not reachable:
        raise HTTPException(
            status_code=503,
            detail="Database unreachable. Check DATABASE_URL and that the Postgres container is running.",
        )
    rows, _ = await asyncio.to_thread(
        db_execute,
        """SELECT
             (SELECT COUNT(*) FROM financial_metrics_data)  AS monthly_data_rows,
             (SELECT COUNT(*) FROM financial_metric)        AS metrics,
             (SELECT COUNT(*) FROM financial_categories)    AS categories""",
    )
    return {"status": "ok", "database": "connected", **rows[0]}


# ---------------------------------------------------------------------------
# GET /api/v1/db/alerts
# ---------------------------------------------------------------------------
@router.get("/db/alerts")
async def db_alerts():
    """Return metrics significantly under budget (>10% below) for the most recent month."""
    import asyncio
    _SQL_ALERTS = """
WITH fmd AS (
  SELECT DISTINCT ON (financial_metric_id, date) *
  FROM financial_metrics_data
  ORDER BY financial_metric_id, date, version_id DESC NULLS LAST
),
latest_month AS (
  SELECT MAX(date) AS max_date FROM fmd WHERE real_value IS NOT NULL
)
SELECT
  fc.name  AS category,
  fm.name  AS metric,
  TO_CHAR(fmd.date, 'YYYY-MM') AS period,
  ROUND(fmd.real_value::numeric, 0)      AS actual,
  ROUND(fmd.budget_value::numeric, 0)    AS budget,
  ROUND(fmd.last_year_real_value::numeric, 0) AS prior_year,
  ROUND((fmd.real_value - fmd.budget_value) * 100.0 /
        NULLIF(ABS(fmd.budget_value), 0), 1)            AS vs_budget_pct,
  ROUND((fmd.real_value - fmd.last_year_real_value) * 100.0 /
        NULLIF(ABS(fmd.last_year_real_value), 0), 1)    AS yoy_pct
FROM fmd
JOIN latest_month lm ON fmd.date = lm.max_date
JOIN financial_metric fm ON fm.id = fmd.financial_metric_id
JOIN financial_types ft   ON ft.id = fm.financial_type_id
JOIN financial_categories fc ON fc.id = ft.financial_category_id
WHERE fmd.real_value IS NOT NULL AND fmd.budget_value IS NOT NULL
  AND fc.name IN ('CA Mobile','Data Mobile','Mobile Money','P&L conso','Opex Consolidés')
  AND (fmd.real_value - fmd.budget_value) * 100.0 /
      NULLIF(ABS(fmd.budget_value), 0) < -10
ORDER BY vs_budget_pct ASC
LIMIT 25
"""
    rows, _ = await asyncio.to_thread(db_execute, _SQL_ALERTS)
    return {"alerts": rows, "period": rows[0]["period"] if rows else None}


# ---------------------------------------------------------------------------
# GET /api/v1/db/sheets
# ---------------------------------------------------------------------------
@router.get("/db/financial-types")
async def db_list_financial_types():
    """List all financial categories and their types from the Digiwise database."""
    import asyncio
    rows, _ = await asyncio.to_thread(db_execute, """
        SELECT fc.name  AS category,
               ft.name  AS type,
               ft.id    AS type_id,
               COUNT(DISTINCT fm.id) AS metric_count
        FROM   financial_categories fc
        JOIN   financial_types ft    ON ft.financial_category_id = fc.id
        LEFT JOIN financial_metric fm ON fm.financial_type_id = ft.id
        GROUP  BY fc.name, ft.name, ft.id
        ORDER  BY fc.name, ft.sequence_id
    """)
    return {"financial_types": rows}


# ---------------------------------------------------------------------------
# GET /api/v1/db/metrics/{sheet_name}
# ---------------------------------------------------------------------------
@router.get("/db/metrics/{financial_type_id}")
async def db_list_metrics(financial_type_id: int):
    """List all metrics and submetrics for a given financial_type_id."""
    import asyncio
    rows, _ = await asyncio.to_thread(db_execute, """
        SELECT fm.id        AS metric_id,
               fm.name      AS metric_name,
               fm.sequence_id,
               COUNT(fs.id) AS submetric_count
        FROM   financial_metric fm
        LEFT JOIN financial_submetric fs ON fs.financial_metric_id = fm.id
        WHERE  fm.financial_type_id = %s
        GROUP  BY fm.id, fm.name, fm.sequence_id
        ORDER  BY fm.sequence_id
    """, (financial_type_id,))
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for financial_type_id={financial_type_id}."
        )
    return {"financial_type_id": financial_type_id, "metrics": rows}


# ---------------------------------------------------------------------------
# GET /api/v1/db/schema
# ---------------------------------------------------------------------------
@router.get("/db/schema")
async def db_schema():
    """
    Return the full database schema: every table with its columns (name,
    type, PK flag, nullability), foreign-key relationships, views, and
    approximate row counts.
    """
    import asyncio

    table_infos = await asyncio.to_thread(inspect_schema)
    views = await asyncio.to_thread(get_views)

    tables_out = []
    for ti in table_infos:
        tables_out.append({
            "name": ti.name,
            "schema": ti.schema,
            "row_count_estimate": ti.row_count_estimate,
            "columns": [
                {
                    "name": c.name,
                    "data_type": c.data_type,
                    "is_pk": c.is_pk,
                    "nullable": c.nullable,
                    "default": c.default,
                }
                for c in ti.columns
            ],
            "foreign_keys": [
                {
                    "column": fk.column,
                    "ref_table": fk.ref_table,
                    "ref_column": fk.ref_column,
                }
                for fk in ti.foreign_keys
            ],
        })

    return {"tables": tables_out, "views": views}


# ---------------------------------------------------------------------------
# GET /api/v1/db/monthly-report
# ---------------------------------------------------------------------------
@router.get("/db/monthly-report")
async def db_monthly_report(language: str = "fr", model: str = ""):
    """
    Generate an automated monthly management narrative.
    Fetches the latest month's KPIs, CAPEX, cashflow, and commissions,
    assigns red/yellow/green signals, then calls the LLM to write a
    structured board-level report.

    Query params:
      language  fr | en  (default: fr)
      model     override model name (default: settings.OLLAMA_MODEL)
    """
    import asyncio
    from app.agents.monthly_report import generate_monthly_report

    model_arg = model.strip() or None
    try:
        result = await asyncio.to_thread(generate_monthly_report, language, model_arg)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Monthly report generation failed: {exc}",
        ) from exc
    return result


# ---------------------------------------------------------------------------
# GET /api/v1/db/geo-intelligence
# ---------------------------------------------------------------------------
@router.get("/db/geo-intelligence")
async def db_geo_intelligence(year: int | None = None, month: int | None = None):
    """
    Return geographic revenue intelligence — MoMo reactivation and
    data/voice auto-recharge aggregated by department/commune.

    Query params:
      year   e.g. 2025 (default: latest period in financial_metrics_data)
      month  1–12     (default: latest period's month)
    """
    import asyncio
    from app.agents.geo_intelligence import get_geo_intelligence

    try:
        result = await asyncio.to_thread(get_geo_intelligence, year, month)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Geo intelligence query failed: {exc}",
        ) from exc
    return result


# ---------------------------------------------------------------------------
# GET /api/v1/db/schema/text
# ---------------------------------------------------------------------------
@router.get("/db/schema/text")
async def db_schema_text():
    """
    Return the schema as a plain-text, human-readable string — the same
    text that is injected into the agent system prompt.
    """
    import asyncio
    from fastapi.responses import PlainTextResponse

    text = await asyncio.to_thread(build_schema_context)
    return PlainTextResponse(content=text)


# ---------------------------------------------------------------------------
# GET  /api/v1/cache/stats   — semantic cache statistics
# POST /api/v1/cache/clear   — flush the semantic cache
# ---------------------------------------------------------------------------
@router.get("/cache/stats")
async def cache_stats():
    """Return semantic cache statistics (hit rate, entry count, TTL)."""
    from app.agents.semantic_cache import semantic_cache
    return semantic_cache.stats()


@router.post("/cache/clear", status_code=status.HTTP_200_OK)
async def cache_clear():
    """Flush all semantic cache entries."""
    from app.agents.semantic_cache import semantic_cache
    n = semantic_cache.clear()
    return {"cleared": n, "message": f"Semantic cache cleared ({n} entries removed)"}
