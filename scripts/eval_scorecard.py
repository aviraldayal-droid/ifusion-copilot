"""
LLM Evaluation Scorecard — TBG AI Copilot
==========================================
Runs 50 benchmark Q+SQL pairs through the DB pipeline for 3 models
and produces a per-category + overall scorecard.

Usage:
    python eval_scorecard.py                     # full run (150 LLM calls)
    python eval_scorecard.py --questions 10      # first 10 questions only
    python eval_scorecard.py --category 3        # category 3 only
    python eval_scorecard.py --no-retry          # skip critic repair loop
    python eval_scorecard.py --out results.csv   # custom output path

Results are written to eval_results.csv after every question so the run
is safe to interrupt and resume (already-done rows are skipped).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Any

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = [
    "nemotron-3-nano:30b",
    "gemma4:31b",
    "devstral-2:123b",
    "qwen3-coder:480b",
    "qwen3-next:80b",
    # "deepseek-v3.2",
]

BENCHMARK_FILE = Path(__file__).parent / "benchmark_questions.txt"
DEFAULT_OUT    = Path(__file__).parent / "new_eval_results_new.csv"

CATEGORY_NAMES = {
    1: "Simple Lookups",
    2: "Aggregations",
    3: "Date Ranges",
    4: "Joins",
    5: "Edge Cases",
}

# Weights for the composite score
W_EXEC    = 0.35   # SQL executes without error
W_MATCH   = 0.45   # result matches gold within 1%
W_FIRSTRY = 0.20   # succeeded on first attempt (no critic repair)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BenchmarkItem:
    q_id:     int
    category: int
    question: str
    gold_sql: str


@dataclass
class QuestionResult:
    q_id:           int
    category:       int
    model:          str
    # Validation pipeline
    syntax_pass:    bool = False
    tables_pass:    bool = False
    semantic_pass:  bool = False
    # Execution
    executed:       bool = False
    # Gold comparison
    gold_executed:  bool = False
    result_match:   bool | None = None   # None = gold SQL failed, skip
    row_count_gold: int  = 0
    row_count_model:int  = 0
    # Performance
    retries:        int  = 0
    latency_s:      float = 0.0   # end-to-end
    sql_gen_s:      float = 0.0   # write_sql LLM call only
    # Retrieval quality
    retrieved_tables: str  = ""   # pipe-separated list of tables the retriever returned
    gold_tables:      str  = ""   # pipe-separated tables needed by gold SQL
    retrieval_recall: bool = False  # True if all gold tables were in retrieved set
    error:          str  = ""


# ── Benchmark parser ──────────────────────────────────────────────────────────

def parse_benchmark(path: Path) -> list[BenchmarkItem]:
    text  = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Build a map: line_number → category (from CATEGORY N: headers)
    cat_by_line: list[tuple[int, int]] = []  # (line_no, cat)
    for i, line in enumerate(lines):
        m = re.match(r"CATEGORY (\d+):", line.strip())
        if m:
            cat_by_line.append((i, int(m.group(1))))

    def cat_for_line(line_no: int) -> int:
        """Return the most recent category header that appeared before this line."""
        cat = 0
        for ln, c in cat_by_line:
            if ln <= line_no:
                cat = c
            else:
                break
        return cat

    items: list[BenchmarkItem] = []
    blocks = re.split(r"\n(?=Q\d+\.)", text)
    char_pos = 0  # track position in original text

    for block in blocks:
        # Locate this block's starting line number
        block_line = text[:char_pos].count("\n")
        char_pos  += len(block) + 1   # +1 for the \n consumed by split

        q_m = re.match(r"Q(\d+)\.\s+(.+?)(?:\n|$)", block.strip())
        if not q_m:
            continue
        q_id  = int(q_m.group(1))
        q_txt = q_m.group(2).strip()

        sql_m = re.search(r"SQL:\n([\s\S]+?)(?:\n---|\Z)", block)
        if not sql_m:
            continue
        gold_sql = sql_m.group(1).strip().rstrip(";")

        category = cat_for_line(block_line)
        items.append(BenchmarkItem(q_id=q_id, category=category, question=q_txt, gold_sql=gold_sql))

    return sorted(items, key=lambda x: x.q_id)


# ── Result comparison ─────────────────────────────────────────────────────────

def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sort_key(row: dict, numeric_cols: list[str]) -> tuple:
    """Sort key for result rows — by numeric values so ordering doesn't matter."""
    return tuple(_to_float(row.get(c)) or 0.0 for c in numeric_cols)


def compare_results(
    gold_rows: list[dict], gold_cols: list[str],
    model_rows: list[dict], model_cols: list[str],
    tolerance: float = 0.01,
) -> bool:
    """
    Returns True if:
    - Same number of rows
    - For every numeric column in gold, the model has the same column
      with values within `tolerance` (default 1%) of the gold value.
    Rows are sorted by their numeric values before comparison so
    different ordering doesn't cause false negatives.
    Non-numeric columns are not compared.
    """
    if len(gold_rows) != len(model_rows):
        return False
    if not gold_rows:
        return True  # both empty = match

    # Find numeric columns in gold that also exist in model
    numeric_cols = []
    for col in gold_cols:
        if col not in model_cols:
            continue
        sample = [gold_rows[i].get(col) for i in range(min(3, len(gold_rows)))]
        sample = [s for s in sample if s is not None]
        if sample and all(isinstance(s, (int, float, Decimal)) for s in sample):
            numeric_cols.append(col)

    if not numeric_cols:
        return True  # no numeric columns to compare — row count match is enough

    # Sort both result sets by numeric values to handle ordering differences
    try:
        gold_sorted  = sorted(gold_rows,  key=lambda r: _sort_key(r, numeric_cols))
        model_sorted = sorted(model_rows, key=lambda r: _sort_key(r, numeric_cols))
    except Exception:
        gold_sorted, model_sorted = gold_rows, model_rows

    for g_row, m_row in zip(gold_sorted, model_sorted):
        for col in numeric_cols:
            g_val = _to_float(g_row.get(col))
            m_val = _to_float(m_row.get(col))
            if g_val is None and m_val is None:
                continue
            if g_val is None or m_val is None:
                return False
            if g_val == 0 and m_val == 0:
                continue
            if g_val == 0:
                return False
            if abs((m_val - g_val) / abs(g_val)) > tolerance:
                return False
    return True


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_one(
    item: BenchmarkItem,
    model: str,
    llm,
    gold_rows: list[dict],
    gold_cols: list[str],
    gold_ok: bool,
    schema_state: dict,
    allow_retry: bool = True,
) -> QuestionResult:
    """Run the write→validate→execute pipeline for one question + model."""
    from app.agents.graph import (
        _make_write_sql_node,
        _make_critique_sql_node,
        validate_syntax,
        validate_tables,
        validate_semantic,
        execute_sql,
        _MAX_RETRIES,
    )

    from app.agents.graph import _referenced_tables
    res = QuestionResult(q_id=item.q_id, category=item.category, model=model)
    res.gold_executed  = gold_ok
    res.row_count_gold = len(gold_rows)

    # Retrieval quality — compare retrieved tables against gold SQL's required tables
    retrieved = schema_state.get("allowed_tables", [])
    gold_tbls = sorted(_referenced_tables(item.gold_sql))
    res.retrieved_tables = "|".join(sorted(retrieved))
    res.gold_tables      = "|".join(gold_tbls)
    res.retrieval_recall = all(t in retrieved for t in gold_tbls) if gold_tbls else True

    t0 = time.monotonic()

    write_sql    = _make_write_sql_node(llm)
    critique_sql = _make_critique_sql_node(llm)

    state = {
        "question":        item.question,
        "history":         "",
        "language":        "en",
        "target_db":       "",
        "snapshot_label":  "",
        "retrieved_schema": schema_state.get("retrieved_schema", ""),
        "allowed_tables":   schema_state.get("allowed_tables", []),
        "sql":             "",
        "syntax_error":    "",
        "table_error":     "",
        "semantic_error":  "",
        "error_history":   [],
        "column_facts":    "",
        "critic_feedback": "",
        "retry_count":     0,
        "sql_error":       "",
        "rows":            [],
        "cols":            [],
        "answer":          "",
        "chart_specs":     [],
    }

    # Write SQL (timed separately — this is the LLM inference cost)
    t_sql = time.monotonic()
    state.update(write_sql(state))
    res.sql_gen_s = round(time.monotonic() - t_sql, 2)

    # Validate + repair loop
    for attempt in range(_MAX_RETRIES + 1):
        syn = validate_syntax(state)
        state.update(syn)
        if state.get("syntax_error"):
            if allow_retry and attempt < _MAX_RETRIES:
                state.update(critique_sql(state))
                state["retry_count"] = attempt + 1
                res.retries += 1
                continue
            break

        res.syntax_pass = True

        tbl = validate_tables(state)
        state.update(tbl)
        if state.get("table_error"):
            if allow_retry and attempt < _MAX_RETRIES:
                state.update(critique_sql(state))
                state["retry_count"] = attempt + 1
                res.retries += 1
                continue
            break

        res.tables_pass = True

        sem = validate_semantic(state)
        state.update(sem)
        if state.get("semantic_error"):
            if allow_retry and attempt < _MAX_RETRIES:
                state.update(critique_sql(state))
                state["retry_count"] = attempt + 1
                res.retries += 1
                continue
            break

        res.semantic_pass = True
        break

    # Execute if all validations passed
    if res.syntax_pass and res.tables_pass and res.semantic_pass:
        exec_out = execute_sql(state)
        state.update(exec_out)
        if not state.get("sql_error"):
            res.executed = True
            res.row_count_model = len(state.get("rows", []))
            if gold_ok:
                res.result_match = compare_results(
                    gold_rows, gold_cols,
                    state.get("rows", []), state.get("cols", []),
                )
            else:
                res.result_match = None
        else:
            res.error = state.get("sql_error", "")[:200]
    else:
        err_parts = [state.get("syntax_error",""), state.get("table_error",""), state.get("semantic_error","")]
        res.error = next((e for e in err_parts if e), "validation failed")[:200]

    res.latency_s = round(time.monotonic() - t0, 2)
    return res


# ── Gold SQL runner ───────────────────────────────────────────────────────────

def run_gold(item: BenchmarkItem) -> tuple[list[dict], list[str], bool]:
    from app.db.connection import execute
    try:
        rows, cols = execute(item.gold_sql)
        return rows, cols, True
    except Exception:
        return [], [], False


# ── Schema retrieval (once per question, shared across models) ────────────────

def retrieve_schema_for(question: str) -> dict:
    from app.agents.graph import retrieve_schema
    return retrieve_schema({
        "question":        question,
        "history":         "",
        "language":        "en",
        "target_db":       "",
        "snapshot_label":  "",
        "retrieved_schema": "",
        "allowed_tables":   [],
        "sql":             "",
        "syntax_error":    "",
        "table_error":     "",
        "semantic_error":  "",
        "error_history":   [],
        "column_facts":    "",
        "critic_feedback": "",
        "retry_count":     0,
        "sql_error":       "",
        "rows":            [],
        "cols":            [],
        "answer":          "",
        "chart_specs":     [],
    })


# ── CSV I/O ───────────────────────────────────────────────────────────────────

CSV_FIELDS = [f.name for f in fields(QuestionResult)]


def load_done(path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not path.exists():
        return done
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row["q_id"]), row["model"]))
            except (KeyError, ValueError):
                pass
    return done


def append_result(path: Path, res: QuestionResult):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            f.name: getattr(res, f.name)
            for f in fields(QuestionResult)
        })


# ── Scorecard printer ─────────────────────────────────────────────────────────

def _pct(n: int, d: int) -> str:
    if d == 0:
        return "  N/A "
    return f"{n:3d}/{d} ({n/d*100:5.1f}%)"


def _score(res_list: list[QuestionResult]) -> float:
    """Composite score 0–100."""
    if not res_list:
        return 0.0
    n = len(res_list)
    exec_rate  = sum(r.executed for r in res_list) / n
    match_list = [r for r in res_list if r.result_match is not None]
    match_rate = sum(r.result_match for r in match_list) / len(match_list) if match_list else exec_rate
    first_rate = sum(r.executed and r.retries == 0 for r in res_list) / n
    return round((W_EXEC * exec_rate + W_MATCH * match_rate + W_FIRSTRY * first_rate) * 100, 1)


def print_scorecard(all_results: list[QuestionResult], models: list[str]):
    by_model: dict[str, list[QuestionResult]] = {m: [] for m in models}
    for r in all_results:
        if r.model in by_model:
            by_model[r.model].append(r)

    # Short model names for column headers
    short = {m: m.split(":")[0][:14] for m in models}

    W = 16  # column width
    sep  = "+" + ("-" * 28) + "+" + (("+" + "-" * (W + 2)) * len(models)) + "+"
    head = "| " + "Metric".ljust(27) + "|"
    for m in models:
        head += f" {short[m].center(W)} |"

    print("\n" + "=" * (30 + (W + 3) * len(models)))
    print(" LLM EVALUATION SCORECARD — TBG AI Copilot DB Pipeline")
    print("=" * (30 + (W + 3) * len(models)))
    print(sep)
    print(head)
    print(sep)

    def _p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    metrics = [
        ("Retrieval Recall",      lambda rs: _pct(sum(r.retrieval_recall for r in rs), len(rs))),
        ("Syntax Pass Rate",      lambda rs: _pct(sum(r.syntax_pass   for r in rs), len(rs))),
        ("Table Pass Rate",       lambda rs: _pct(sum(r.tables_pass   for r in rs), len(rs))),
        ("Semantic Pass Rate",    lambda rs: _pct(sum(r.semantic_pass  for r in rs), len(rs))),
        ("Execution Rate",        lambda rs: _pct(sum(r.executed       for r in rs), len(rs))),
        ("Result Accuracy (±1%)", lambda rs: _pct(sum(r.result_match   for r in rs if r.result_match is not None),
                                                   sum(1 for r in rs if r.result_match is not None))),
        ("First-Try Success",     lambda rs: _pct(sum(r.executed and r.retries == 0 for r in rs), len(rs))),
        ("Avg Retries",           lambda rs: f"{sum(r.retries for r in rs)/len(rs):.2f}".center(W) if rs else "N/A".center(W)),
        ("Avg Latency (s)",       lambda rs: f"{sum(r.latency_s for r in rs)/len(rs):.1f}s".center(W) if rs else "N/A".center(W)),
        ("P95 Latency (s)",       lambda rs: f"{_p95([r.latency_s for r in rs]):.1f}s".center(W) if rs else "N/A".center(W)),
        ("Min / Max Latency (s)", lambda rs: f"{min(r.latency_s for r in rs):.1f}s / {max(r.latency_s for r in rs):.1f}s".center(W) if rs else "N/A".center(W)),
        ("Avg SQL Gen Time (s)",  lambda rs: f"{sum(r.sql_gen_s for r in rs)/len(rs):.1f}s".center(W) if rs else "N/A".center(W)),
    ]

    for label, fn in metrics:
        row = "| " + label.ljust(27) + "|"
        for m in models:
            val = fn(by_model[m]) if by_model[m] else "N/A".center(W)
            row += f" {str(val).center(W)} |"
        print(row)

    print(sep)
    score_row = "| " + "COMPOSITE SCORE".ljust(27) + "|"
    for m in models:
        s = _score(by_model[m])
        score_row += f" {'★ ' + str(s) + ' / 100':^{W}} |"
    print(score_row)
    print(sep)

    # Per-category breakdown
    all_cats = sorted(set(r.category for r in all_results if r.category))
    if len(all_cats) > 1:
        print("\n  Category Breakdown — Execution Rate")
        cat_sep = "  +" + "-" * 22 + "+" + (("+" + "-" * (W + 2)) * len(models)) + "+"
        print(cat_sep)
        cat_head = "  | " + "Category".ljust(21) + "|"
        for m in models:
            cat_head += f" {short[m].center(W)} |"
        print(cat_head)
        print(cat_sep)
        for cat in all_cats:
            cname = CATEGORY_NAMES.get(cat, f"Cat {cat}")[:20]
            row   = f"  | {cname.ljust(21)}|"
            for m in models:
                cat_rs = [r for r in by_model[m] if r.category == cat]
                exec_n = sum(r.executed for r in cat_rs)
                row   += f" {_pct(exec_n, len(cat_rs)).center(W)} |"
            print(row)
        print(cat_sep)
    print()


# ── Progress line ─────────────────────────────────────────────────────────────

def _status(r: QuestionResult) -> str:
    checks = [
        ("syn", r.syntax_pass),
        ("tbl", r.tables_pass),
        ("sem", r.semantic_pass),
        ("exe", r.executed),
    ]
    parts = [f"{'✓' if ok else '✗'}{label}" for label, ok in checks]
    if r.result_match is True:
        match = " match✓"
    elif r.result_match is False:
        match = " match✗"
    elif not r.gold_executed:
        match = " (gold failed)"
    elif not r.executed:
        match = " (exec failed)"
    else:
        match = ""
    retry = f" r{r.retries}" if r.retries else ""
    # Retrieval annotation — show missed tables if recall failed
    if not r.retrieval_recall and r.gold_tables:
        retrieved = set(r.retrieved_tables.split("|")) if r.retrieved_tables else set()
        missed = [t for t in r.gold_tables.split("|") if t and t not in retrieved]
        retr = f" ⚠ retr missing: {', '.join(missed)}"
    else:
        retr = " ✓retr"
    return " ".join(parts) + match + retry + retr + f" {r.latency_s:.1f}s sql={r.sql_gen_s:.1f}s"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Scorecard")
    parser.add_argument("--questions", type=int, default=None,
                        help="Run only the first N questions")
    parser.add_argument("--category",  type=int, default=None,
                        help="Run only this category (1-5)")
    parser.add_argument("--models",    nargs="+", default=MODELS,
                        help="Models to compare (space-separated)")
    parser.add_argument("--no-retry",  action="store_true",
                        help="Skip critic repair loop (faster, stricter)")
    parser.add_argument("--out",       default=str(DEFAULT_OUT),
                        help="Output CSV path")
    args = parser.parse_args()

    out_path    = Path(args.out)
    allow_retry = not args.no_retry
    models      = args.models

    # Parse benchmark
    items = parse_benchmark(BENCHMARK_FILE)
    if args.category:
        items = [i for i in items if i.category == args.category]
    if args.questions:
        items = items[: args.questions]

    print(f"Benchmark: {len(items)} question(s)  |  Models: {len(models)}")
    print(f"Output:    {out_path}")
    print(f"Retry:     {'enabled' if allow_retry else 'disabled'}")
    print("─" * 60)

    # Load already-done pairs so we can resume
    done = load_done(out_path)
    all_results: list[QuestionResult] = []

    # Load existing results for the final scorecard
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    all_results.append(QuestionResult(
                        q_id=int(row["q_id"]),
                        category=int(row["category"]),
                        model=row["model"],
                        syntax_pass=row["syntax_pass"] == "True",
                        tables_pass=row["tables_pass"] == "True",
                        semantic_pass=row["semantic_pass"] == "True",
                        executed=row["executed"] == "True",
                        gold_executed=row["gold_executed"] == "True",
                        result_match=None if row["result_match"] == "None" else row["result_match"] == "True",
                        row_count_gold=int(row.get("row_count_gold", 0) or 0),
                        row_count_model=int(row.get("row_count_model", 0) or 0),
                        retries=int(row.get("retries", 0) or 0),
                        latency_s=float(row.get("latency_s", 0) or 0),
                        sql_gen_s=float(row.get("sql_gen_s", 0) or 0),
                        retrieved_tables=row.get("retrieved_tables", ""),
                        gold_tables=row.get("gold_tables", ""),
                        retrieval_recall=row.get("retrieval_recall", "True") == "True",
                        error=row.get("error", ""),
                    ))
                except Exception:
                    pass

    total = len(items) * len(models)
    completed = len(done)

    from app.agents.graph import _make_llm

    for item in items:
        print(f"\n[Q{item.q_id:02d} Cat{item.category}] {item.question[:72]}")

        # Run gold SQL once per question
        gold_rows, gold_cols, gold_ok = run_gold(item)
        gold_label = f"gold: {len(gold_rows)} row(s)" if gold_ok else "gold: FAILED"
        print(f"  {gold_label}")

        # Retrieve schema once per question (shared across models)
        schema_state = retrieve_schema_for(item.question)

        for model in models:
            short_m = model.split(":")[0]
            if (item.q_id, model) in done:
                print(f"  [{short_m}] skipped (already done)")
                continue

            try:
                llm = _make_llm(model)
                res = run_one(
                    item, model, llm,
                    gold_rows, gold_cols, gold_ok,
                    schema_state,
                    allow_retry=allow_retry,
                )
            except Exception as exc:
                res = QuestionResult(
                    q_id=item.q_id, category=item.category, model=model,
                    gold_executed=gold_ok,
                    row_count_gold=len(gold_rows),
                    error=str(exc)[:200],
                )

            all_results.append(res)
            append_result(out_path, res)
            completed += 1
            print(f"  [{short_m}] {_status(res)}  [{completed}/{total}]")

    print("\n")
    print_scorecard(all_results, models)
    print(f"Raw results → {out_path}")


if __name__ == "__main__":
    main()
