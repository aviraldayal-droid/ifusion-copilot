"""
Smoke-test for all pure-Python components of the DB pipeline.
No LLM and no DB connection required.

Run BEFORE and AFTER any change to app/agents/graph.py to catch regressions.

Usage:
    python check_pipeline.py
"""
import sys
import traceback

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, str]] = []


def test(name: str, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as exc:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}: {exc}")
        traceback.print_exc()


# ── Month resolution ─────────────────────────────────────────────────────────

def test_month_int_resolution():
    from app.agents.graph import _resolve_month_names
    rows = [{"month": 1, "v": 100}, {"month": 12, "v": 50}]
    out = _resolve_month_names(rows, ["month", "v"])
    assert out[0]["month"] == "January",  f"got {out[0]['month']}"
    assert out[1]["month"] == "December", f"got {out[1]['month']}"


def test_month_padded_string():
    from app.agents.graph import _resolve_month_names
    rows = [{"month": "September  ", "v": 100}]
    out = _resolve_month_names(rows, ["month", "v"])
    assert out[0]["month"] == "September", f"got {repr(out[0]['month'])}"


def test_month_sort_order():
    from app.agents.graph import _sort_by_month_order
    rows = [{"month": "September"}, {"month": "January"}, {"month": "March"}]
    out = _sort_by_month_order(rows, ["month"])
    assert [r["month"] for r in out] == ["January", "March", "September"]


def test_month_sort_noop_when_no_month_col():
    from app.agents.graph import _sort_by_month_order
    rows = [{"supplier": "B", "total": 2}, {"supplier": "A", "total": 1}]
    out = _sort_by_month_order(rows, ["supplier", "total"])
    assert [r["supplier"] for r in out] == ["B", "A"], "non-month cols should not be reordered"


def test_month_already_ordered():
    from app.agents.graph import _resolve_month_names, _sort_by_month_order
    rows = [{"month": i, "capex": i * 10} for i in range(1, 13)]
    resolved = _resolve_month_names(rows, ["month", "capex"])
    sorted_rows = _sort_by_month_order(resolved, ["month", "capex"])
    names = [r["month"] for r in sorted_rows]
    from app.agents.graph import _MONTH_NAMES
    expected = [_MONTH_NAMES[i] for i in range(1, 13)]
    assert names == expected, f"order wrong: {names}"


# ── SQL extraction ────────────────────────────────────────────────────────────

def test_sql_extraction_clean():
    from app.agents.graph import _extract_sql
    sql, err = _extract_sql("SELECT id FROM foo WHERE year = 2025")
    assert not err and sql.startswith("SELECT"), f"err={err}"


def test_sql_extraction_fenced():
    from app.agents.graph import _extract_sql
    sql, err = _extract_sql("```sql\nSELECT id FROM foo;\n```")
    assert not err and sql.startswith("SELECT"), f"err={err}"


def test_sql_extraction_with_prose():
    from app.agents.graph import _extract_sql
    sql, err = _extract_sql("Here is the query:\nSELECT id FROM foo WHERE year = 2025;")
    assert not err and sql.startswith("SELECT"), f"err={err}, sql={sql!r}"


def test_sql_extraction_prose_only_rejected():
    from app.agents.graph import _extract_sql
    _, err = _extract_sql("Here is the answer.")
    assert err, "should have rejected non-SQL output"


def test_sql_extraction_with_cte():
    from app.agents.graph import _extract_sql
    raw = "WITH fmd AS (SELECT * FROM financial_metrics_data) SELECT * FROM fmd;"
    sql, err = _extract_sql(raw)
    assert not err and sql.upper().startswith("WITH"), f"err={err}"


# ── SQL syntax validation ─────────────────────────────────────────────────────

def test_syntax_valid_simple():
    from app.agents.graph import validate_syntax
    out = validate_syntax({"sql": "SELECT id, name FROM capex_data WHERE year = 2025"})
    assert not out.get("syntax_error"), out.get("syntax_error")


def test_syntax_valid_cte():
    from app.agents.graph import validate_syntax
    sql = "WITH fmd AS (SELECT * FROM financial_metrics_data ORDER BY id DESC) SELECT * FROM fmd"
    out = validate_syntax({"sql": sql})
    assert not out.get("syntax_error"), out.get("syntax_error")


def test_syntax_invalid():
    from app.agents.graph import validate_syntax
    out = validate_syntax({"sql": "SELECT FROM WHERE NOTHING"})
    assert out.get("syntax_error"), "should have caught bad SQL"


def test_syntax_empty_sql():
    from app.agents.graph import validate_syntax
    out = validate_syntax({"sql": ""})
    assert out.get("syntax_error"), "empty SQL should fail"


def test_syntax_blocks_delete():
    from app.agents.graph import validate_syntax
    out = validate_syntax({"sql": "DELETE FROM capex_data WHERE year = 2024"})
    assert out.get("syntax_error"), "DELETE should be blocked"


def test_syntax_blocks_drop():
    from app.agents.graph import validate_syntax
    out = validate_syntax({"sql": "DROP TABLE capex_data"})
    assert out.get("syntax_error"), "DROP should be blocked"


def test_syntax_blocks_update():
    from app.agents.graph import validate_syntax
    out = validate_syntax({"sql": "UPDATE capex_data SET equipment = 0 WHERE year = 2024"})
    assert out.get("syntax_error"), "UPDATE should be blocked"


def test_syntax_blocks_cte_with_delete():
    from app.agents.graph import validate_syntax
    sql = "WITH d AS (DELETE FROM capex_data RETURNING *) SELECT * FROM d"
    out = validate_syntax({"sql": sql})
    assert out.get("syntax_error"), "DELETE inside CTE should be blocked"


def test_execute_blocks_destructive_keyword():
    from app.agents.graph import execute_sql
    out = execute_sql({"sql": "DELETE FROM capex_data WHERE year = 2024"})
    assert out.get("sql_error"), "execute_sql should block destructive SQL"
    assert "blocked" in out["sql_error"].lower(), f"unexpected error: {out['sql_error']}"


# ── Table whitelist ───────────────────────────────────────────────────────────

def test_referenced_tables_basic():
    from app.agents.graph import _referenced_tables
    sql = "SELECT * FROM capex_data cd JOIN capex_projects cp ON cp.id = cd.capex_projects_id"
    tables = _referenced_tables(sql)
    assert "capex_data" in tables
    assert "capex_projects" in tables


def test_referenced_tables_cte_excluded():
    from app.agents.graph import _referenced_tables
    sql = (
        "WITH fmd AS (SELECT * FROM financial_metrics_data) "
        "SELECT * FROM fmd JOIN financial_metric fm ON fm.id = fmd.financial_metric_id"
    )
    tables = _referenced_tables(sql)
    assert "financial_metrics_data" in tables
    assert "financial_metric" in tables
    assert "fmd" not in tables, f"CTE alias should be excluded, got: {tables}"


def test_validate_tables_blocks_hallucination():
    from app.agents.graph import validate_tables
    state = {
        "sql": "SELECT * FROM nonexistent_table",
        "allowed_tables": ["capex_data", "capex_projects"],
        "syntax_error": "",
    }
    out = validate_tables(state)
    assert out.get("table_error"), "should block unknown table"


def test_validate_tables_passes_known():
    from app.agents.graph import validate_tables
    state = {
        "sql": "SELECT * FROM capex_data WHERE year = 2025",
        "allowed_tables": ["capex_data"],
        "syntax_error": "",
    }
    out = validate_tables(state)
    assert not out.get("table_error"), out.get("table_error")


# ── Chart spec builder ────────────────────────────────────────────────────────

def test_chart_line_monthly():
    from app.agents.graph import _build_chart_spec
    rows = [
        {"month": "January",  "total_capex": 150},
        {"month": "February", "total_capex": 200},
        {"month": "March",    "total_capex": 180},
    ]
    spec = _build_chart_spec(["month", "total_capex"], rows, "monthly capex trend 2025")
    assert spec is not None
    assert spec["chart_type"] == "line"
    assert spec["peak_idx"]   == 1,   f"peak wrong: {spec['peak_idx']}"
    assert spec["trough_idx"] == 0,   f"trough wrong: {spec['trough_idx']}"
    assert spec["avg_value"]  is not None


def test_chart_bar_rank():
    from app.agents.graph import _build_chart_spec
    rows = [{"supplier": "ERICSSON", "total": 500}, {"supplier": "HUAWEI", "total": 300}]
    spec = _build_chart_spec(["supplier", "total"], rows, "top suppliers by spend")
    assert spec is not None
    assert spec["chart_type"] == "bar_horizontal"


def test_chart_none_for_single_row():
    from app.agents.graph import _build_chart_spec
    rows = [{"metric": "EBITDA", "value": 1000}]
    spec = _build_chart_spec(["metric", "value"], rows, "what is ebitda")
    assert spec is None, f"single row should return None, got {spec}"


def test_chart_title_derived_from_columns():
    from app.agents.graph import _build_chart_spec
    rows = [{"month": "January", "total_capex": 100}, {"month": "February", "total_capex": 200}]
    spec = _build_chart_spec(["month", "total_capex"], rows, "monthly capex trend 2025")
    assert "Total Capex" in spec["title"], f"title should mention column: {spec['title']}"
    assert "Monthly Trend" in spec["title"] or "Trend" in spec["title"]


# ── Data facts ────────────────────────────────────────────────────────────────

def test_data_facts_peak_trough():
    from app.agents.graph import _compute_data_facts
    rows = [
        {"month": "January",  "total_capex": 150_000_000},
        {"month": "March",    "total_capex": 300_000_000},
        {"month": "February", "total_capex": 200_000_000},
    ]
    facts = _compute_data_facts(["month", "total_capex"], rows)
    assert "March"   in facts, f"peak (March=300M) not found: {facts}"
    assert "January" in facts, f"trough (Jan=150M) not found: {facts}"


def test_data_facts_empty():
    from app.agents.graph import _compute_data_facts
    facts = _compute_data_facts([], [])
    assert facts == "", "empty input should return empty string"


# ── ASCII table ───────────────────────────────────────────────────────────────

def test_ascii_table_basic():
    from app.agents.graph import _build_ascii_table
    rows = [{"month": "January", "capex": 100}, {"month": "February", "capex": 200}]
    table = _build_ascii_table(rows, ["month", "capex"])
    assert "January"  in table
    assert "February" in table
    assert "|" in table


# ── Routing logic ─────────────────────────────────────────────────────────────

def test_route_after_syntax_error_retries():
    from app.agents.graph import _route_after_syntax
    state = {"syntax_error": "bad sql", "retry_count": 0}
    assert _route_after_syntax(state) == "critique_sql"


def test_route_after_syntax_exhausted():
    from app.agents.graph import _route_after_syntax, _MAX_RETRIES
    state = {"syntax_error": "still bad", "retry_count": _MAX_RETRIES}
    assert _route_after_syntax(state) == "format_answer"


def test_route_after_syntax_passes():
    from app.agents.graph import _route_after_syntax
    state = {"syntax_error": "", "retry_count": 0}
    assert _route_after_syntax(state) == "validate_tables"


# ── Run all ───────────────────────────────────────────────────────────────────

print("Pipeline smoke test")
print("─" * 50)

tests = [
    ("month int → name resolution",         test_month_int_resolution),
    ("month padded string stripping",        test_month_padded_string),
    ("month chronological sort",             test_month_sort_order),
    ("month sort noop for non-month cols",   test_month_sort_noop_when_no_month_col),
    ("month full 1-12 round-trip",           test_month_already_ordered),
    ("SQL extract — bare SELECT",            test_sql_extraction_clean),
    ("SQL extract — fenced block",           test_sql_extraction_fenced),
    ("SQL extract — SELECT after prose",     test_sql_extraction_with_prose),
    ("SQL extract — prose-only rejected",    test_sql_extraction_prose_only_rejected),
    ("SQL extract — WITH CTE",               test_sql_extraction_with_cte),
    ("syntax valid — simple",               test_syntax_valid_simple),
    ("syntax valid — CTE",                  test_syntax_valid_cte),
    ("syntax invalid",                      test_syntax_invalid),
    ("syntax empty SQL",                    test_syntax_empty_sql),
    ("syntax blocks DELETE",                test_syntax_blocks_delete),
    ("syntax blocks DROP",                  test_syntax_blocks_drop),
    ("syntax blocks UPDATE",                test_syntax_blocks_update),
    ("syntax blocks CTE with DELETE",       test_syntax_blocks_cte_with_delete),
    ("execute blocks destructive keyword",  test_execute_blocks_destructive_keyword),
    ("tables — basic join",                 test_referenced_tables_basic),
    ("tables — CTE alias excluded",         test_referenced_tables_cte_excluded),
    ("tables — blocks hallucination",       test_validate_tables_blocks_hallucination),
    ("tables — passes known table",         test_validate_tables_passes_known),
    ("chart — line monthly trend",          test_chart_line_monthly),
    ("chart — bar_horizontal rank",         test_chart_bar_rank),
    ("chart — None for single row",         test_chart_none_for_single_row),
    ("chart — title from columns",          test_chart_title_derived_from_columns),
    ("data facts — peak/trough",            test_data_facts_peak_trough),
    ("data facts — empty input",            test_data_facts_empty),
    ("ascii table — renders rows",          test_ascii_table_basic),
    ("routing — retries on syntax error",   test_route_after_syntax_error_retries),
    ("routing — exhausted → format_answer", test_route_after_syntax_exhausted),
    ("routing — passes to validate_tables", test_route_after_syntax_passes),
]

for name, fn in tests:
    test(name, fn)

passed = sum(1 for r, _ in results if r == PASS)
failed = sum(1 for r, _ in results if r == FAIL)
total  = passed + failed

print("─" * 50)
print(f"{passed}/{total} passed", end="")
if failed:
    print(f"  ← {failed} FAILED")
    sys.exit(1)
else:
    print("  — all clear")
