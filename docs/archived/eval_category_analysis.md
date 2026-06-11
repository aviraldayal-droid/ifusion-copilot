# TBG AI Copilot — LLM Evaluation: Category Failure Analysis

**Date:** 2026-05-19
**Eval Framework:** `eval_scorecard.py` | **Benchmark:** `benchmark_questions.txt` (73 Q+SQL pairs)
**Models Tested:** devstral-2:123b · gemma4:31b · nemotron-3-nano:30b · qwen3-coder:480b · qwen3-next:80b

---

## What "Gold SQL" Means

Each benchmark question has a **Gold SQL** — a human-written, manually verified SQL query that produces the correct answer.

During evaluation:
- **`gold_executed`** — did the gold SQL run without error against the live DB? (validates the benchmark itself)
- **`result_match`** — does the AI-generated SQL produce the **exact same result set** as the gold SQL?

A question is counted as **passed** only when `result_match = True`.

> `result_match = False` can mean wrong rows, wrong columns, wrong aggregation, or a runtime error. It does **not** mean the answer was wrong in natural language — only that the SQL output didn't match precisely.

---

## Summary Table

| Cat | Topic | Questions | Avg Accuracy | Status |
|-----|-------|-----------|-------------|--------|
| 1 | Simple Lookups | 10 | **52%** | Partial — dedup mismatch |
| 2 | Aggregations | 10 | **56%** | Partial — retrieval gaps |
| 3 | Date Ranges | 10 | **32%** | Poor — multi-row ordering |
| 4 | Joins | 10 | **30%** | Poor — ROUND cast error |
| 5 | Edge Cases | 10 | **50%** | Partial — NULL handling |
| 6 | Revenue & Growth | 4 | **20%** | Critical — schema not retrieved |
| 7 | Profitability & Margins | 4 | **25%** | Poor — cross-table math |
| 8 | Operating Expenses | 3 | **33%** | Incomplete — quota blocked |
| 9 | Capex Analysis | 3 | **0%** | Blocked — Ollama quota |
| 10 | Cash Flow & Liquidity | 3 | **0%** | Blocked — Ollama quota |
| 11 | Cost Analysis & KPI | 6 | **0%** | Blocked — Ollama quota |

---

## Category 1 — Simple Lookups

**Accuracy: 52%** (26/50 across 5 models)

| Model | Score |
|-------|-------|
| devstral-2:123b | 5/10 (50%) |
| gemma4:31b | 5/10 (50%) |
| nemotron-3-nano:30b | 6/10 (60%) |
| qwen3-coder:480b | 4/10 (40%) |
| qwen3-next:80b | 6/10 (60%) |

**Example Questions:**
- _"What is the EBITDA real value for the most recent available date?"_
- _"What is the total CA Global for January 2025?"_

**Root Cause:**
The gold SQL uses `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY version_id DESC)` to deduplicate versioned rows and return exactly 1 row. AI models often skip the deduplication window and return all versions, producing 2–3 rows instead of 1. This causes `result_match = False` even when the latest value is correct.

**Fix (DA-40):**
Add a deduplication note to the writer system prompt explaining that `financial_metrics_data` is versioned and always requires dedup via `ROW_NUMBER() OVER (PARTITION BY financial_metric_id, date ORDER BY version_id DESC)`.

---

## Category 2 — Aggregations

**Accuracy: 56%** (28/50 across 5 models)

| Model | Score |
|-------|-------|
| devstral-2:123b | 5/10 (50%) |
| gemma4:31b | 5/10 (50%) |
| nemotron-3-nano:30b | 6/10 (60%) |
| qwen3-coder:480b | 7/10 (70%) |
| qwen3-next:80b | 5/10 (50%) |

**Example Questions:**
- _"What is the total CA Global across all months in 2025?"_
- _"What is the average EBITDA real value across all available months?"_

**Root Cause:**
Schema retrieval recall drops to 35/50 (vs 45/50 in Cat 1). When the schema retriever misses a required table, the model is forced to hallucinate joins or use a fallback table — producing structurally valid but semantically wrong SQL. `SUM`/`AVG` over the wrong table or wrong column yields a non-matching result.

**Fix (DA-38):**
Improve schema retriever aliases for aggregation-heavy tables (`financial_metrics_data`, `monthly_evolution`). Add domain hint keywords: `"total"`, `"average"`, `"aggregate"`.

---

## Category 3 — Date Ranges

**Accuracy: 32%** (16/50 across 5 models)

| Model | Score |
|-------|-------|
| devstral-2:123b | 2/10 (20%) |
| gemma4:31b | 3/10 (30%) |
| nemotron-3-nano:30b | 4/10 (40%) |
| qwen3-coder:480b | 3/10 (30%) |
| qwen3-next:80b | 4/10 (40%) |

**Example Questions:**
- _"What was the EBITDA real value for each month in Q1 2025 (January–March)?"_
- _"Show CA Global trend from January 2025 to June 2025."_

**Root Cause:**
Gold SQL for date-range queries returns ordered, multi-row result sets. Models frequently:
1. Use wrong date filter syntax (`BETWEEN` with wrong bounds, or `EXTRACT(YEAR)` only without month)
2. Return rows in a different sort order — causing result set comparison to fail even if values are correct
3. Use `date_trunc('month', ...)` inconsistently, producing mismatched date formats

**Fix:**
Add to writer prompt: always `ORDER BY date ASC` for time-series queries; use `date >= 'YYYY-01-01' AND date < 'YYYY+1-01-01'` pattern for year filters; use `date BETWEEN 'YYYY-MM-01' AND 'YYYY-MM-01'::date + INTERVAL '1 month' - INTERVAL '1 day'` for month ranges.

---

## Category 4 — Joins

**Accuracy: 30%** (15/50 across 5 models)

| Model | Score |
|-------|-------|
| devstral-2:123b | 3/10 (30%) |
| gemma4:31b | 3/10 (30%) |
| nemotron-3-nano:30b | 3/10 (30%) |
| qwen3-coder:480b | 3/10 (30%) |
| qwen3-next:80b | 3/10 (30%) |

**Example Questions:**
- _"What is the total capex per direction in 2025?"_
- _"List all capex projects for ERICSSON AB with their monthly spend in 2025."_

**Root Cause:**
Two compounding issues:
1. **PostgreSQL ROUND type error** — `ROUND(double_precision, int)` is not supported; requires `ROUND((expr)::numeric, n)`. Multi-table join queries frequently compute percentages that hit this bug.
2. **Join path hallucination** — models invent FK paths between tables that aren't directly connected, producing invalid SQL that fails at `validate_semantic` and exhausts retries.

**Fix (implemented):**
`ROUND CASTING RULE` added to writer system prompt. All 29 gold SQL ROUND calls in benchmark fixed with `::numeric` cast.
Remaining fix: add join-path hints to schema descriptions for capex tables.

---

## Category 5 — Edge Cases

**Accuracy: 50%** (25/50 across 5 models)

| Model | Score |
|-------|-------|
| devstral-2:123b | 5/10 (50%) |
| gemma4:31b | 5/10 (50%) |
| nemotron-3-nano:30b | 5/10 (50%) |
| qwen3-coder:480b | 5/10 (50%) |
| qwen3-next:80b | 5/10 (50%) |

Example Questions:
- Which metrics have a real value below their budget in any month of 2025? (underperforming metrics)
- Are there any months where cashflow data is missing (NULL) in 2025?

Root Cause:
Edge case queries require correct NULL handling (`IS NULL`, `COALESCE`, `NULLIF`). Models handle basic NULL checks correctly 50% of the time but fail on:
- `NULLIF(denominator, 0)` for safe division
- Queries requiring `WHERE real_value IS NOT NULL AND budget_value IS NOT NULL` before comparison
- Threshold comparisons that differ in boundary inclusion (`<` vs `<=`)

Fix:
Add NULL handling rules to writer prompt: always use `NULLIF(expr, 0)` in division denominators; add `WHERE ... IS NOT NULL` guards before comparisons.

---

## Category 6 — Revenue & Growth

**Accuracy: 20%** (4/20 across 5 models)

| Model | Score |
|-------|-------|
| devstral-2:123b | 1/4 (25%) |
| gemma4:31b | 2/4 (50%) |
| nemotron-3-nano:30b | 0/4 (0%) |
| qwen3-coder:480b | 0/4 (0%) |
| qwen3-next:80b | 1/4 (25%) |

**Example Questions:**
- _"What is the year-over-year CA Global growth rate per month in 2025 vs 2024?"_
- _"What is the revenue breakdown by segment (Voix, Data, Forfaits) and each segment's share of total CA Global in Q1 2025?"_

**Root Cause — Critical:**
Schema retrieval recall = **0/20**. The `monthly_evolution` table (which stores YoY revenue, segment breakdowns, and growth rates) is **never retrieved** by the schema RAG. The table exists in `schema_descriptions.yaml` but has no YoY/growth/evolution aliases, so cosine similarity and keyword scoring both miss it when questions use terms like "growth rate", "year-over-year", or "trend".

**Fix (DA-38 — high priority):**
Add the following aliases to `schema_descriptions.yaml` for `monthly_evolution`:
```
year-over-year growth, YoY, revenue evolution, monthly trend, CA evolution,
growth rate, revenue by segment, segment breakdown, Voix Data Forfaits
```

---

## Category 7 — Profitability & Margins

**Accuracy: 25%** (5/20 across 5 models)

| Model | Score |
|-------|-------|
| devstral-2:123b | 1/4 (25%) |
| gemma4:31b | 1/4 (25%) |
| nemotron-3-nano:30b | 1/4 (25%) |
| qwen3-coder:480b | 1/4 (25%) |
| qwen3-next:80b | 1/4 (25%) |

**Example Questions:**
- _"What is the EBITDA margin (EBITDA as % of CA Global) for each month in 2025?"_
- _"Show EBITDA budget vs actual variance for each month in 2025."_

**Root Cause:**
Margin calculations require joining `financial_metrics_data` for EBITDA with `monthly_evolution` for CA Global — two structurally different tables with different key columns. Models either:
1. Use only one table and approximate the margin with wrong data
2. Attempt the join but connect on wrong columns (no direct FK between the two)

Retrieval recall is 15/20 — tables are found, but the join logic fails.

**Fix:**
Add explicit cross-table calculation hints to schema for EBITDA margin. Consider adding a computed view (`v_monthly_margin`) to simplify this join for the agent.

---

## Category 8 — Operating Expenses

**Accuracy: 33%** (5/15 across 5 models) — **Eval Incomplete**

| Model | Score |
|-------|-------|
| All 5 models | 1/3 (33%) each |

**Example Questions:**
- _"Which financial metrics had the largest budget overrun (real > budget) in 2025 — top 5?"_
- _"What is the average monthly value for each financial metric in 2025?"_

**Root Cause:**
Evaluation was interrupted mid-run by Ollama Cloud session quota limits (HTTP 429). Only 3 questions per model were evaluated (15/15 questions total, partial). The 33% pass rate on completed rows is likely inaccurate due to small sample.

Budget overrun queries require the same dedup + version handling as Cat 1 but with a `WHERE real_value > budget_value` filter — models that don't dedup return multiple versions and inflate the overrun count.

**Fix:**
Complete eval run once Ollama Cloud quota is resolved or local GPU is available. Apply dedup fix from Cat 1.

---

## Category 9 — Capex Analysis

**Accuracy: 0%** (0/15) — **Eval Blocked by Quota**

**Example Questions:**
- _"What is the CAPEX spend breakdown by type (equipment vs services vs additional costs) as a percentage in 2025?"_
- _"What is the CAPEX intensity ratio (total CAPEX as % of total revenue) in 2025?"_

**Root Cause:**
All 15 rows failed due to Ollama Cloud HTTP 429 (session usage limit). The agent never executed — `executed = False`, `result_match = False`. No SQL was generated.

**Expected failure mode (pre-quota):** CAPEX intensity requires dividing total CAPEX by total revenue — again a cross-table calculation between `capex_data` and `monthly_evolution`, plus the ROUND cast issue.

**Fix:**
Unblock eval (GPU/quota). Apply `::numeric` ROUND fix. Add CAPEX-to-revenue join hint in schema.

---

## Category 10 — Cash Flow & Liquidity

**Accuracy: 0%** (0/15) — **Eval Blocked by Quota**

**Example Questions:**
- _"Which months in 2025 had negative cashflow?"_
- _"Show the running cumulative cashflow by month in 2025."_

**Root Cause:**
Blocked by Ollama Cloud quota (same as Cat 9). No SQL generated.

**Expected failure mode:** Cumulative cashflow requires a window function `SUM(cashflow) OVER (ORDER BY month)` — a pattern models tend to handle correctly, so this category may have higher pass rate once unblocked.

**Fix:**
Unblock eval. No additional prompt fixes expected to be needed.

---

## Category 11 — Cost Analysis & KPI Monitoring

**Accuracy: 0%** (0/30) — **Eval Blocked by Quota**

**Example Questions:**
- _"What is the total commission cost as a percentage of total revenue in 2025?"_
- _"Which commission agent had the highest growth in payout from 2024 to 2025?"_

**Root Cause:**
Blocked by Ollama Cloud quota. No SQL generated.

**Expected failure mode:** Commission % of revenue again requires joining `commission_enlevements` with `monthly_evolution` — same cross-table pattern as Cat 6/7/9.

**Fix:**
Unblock eval. Address cross-table join hints in schema for commission tables.

---

## Root Cause Summary

| Root Cause | Affects | Fix |
|------------|---------|-----|
| `ROUND(double_precision, int)` PostgreSQL error | Cat 4, 6, 7, 9 | ✅ Fixed — `::numeric` cast in prompt + benchmark |
| Deduplication missing (`ROW_NUMBER` window) | Cat 1, 2, 8 | Pending — DA-40 |
| `monthly_evolution` not retrieved by RAG | Cat 6 (0% retrieval) | Pending — DA-38 |
| Cross-table join hallucination | Cat 7, 9, 10, 11 | Pending — schema join hints |
| Date ordering / boundary mismatch | Cat 3 | Pending — writer prompt date rules |
| NULL / NULLIF handling | Cat 5 | Pending — writer prompt NULL rules |
| Ollama Cloud quota exhaustion (HTTP 429) | Cat 8–11 | Pending — DA-42 GPU/cloud upgrade |

---

## Recommended Fix Priority

1. **DA-38** — Add `monthly_evolution` YoY/growth aliases to `schema_descriptions.yaml` _(unblocks Cat 6, highest ROI)_
2. **DA-40** — Add dedup rule to writer system prompt _(improves Cat 1, 2, 8)_
3. **DA-42** — Resolve Ollama Cloud quota / provision local/cloud GPU _(unblocks Cat 9–11 eval)_
4. Cross-table join hints in schema descriptions _(improves Cat 7, 9, 10, 11)_
5. Date range and NULL handling rules in writer prompt _(improves Cat 3, 5)_

---

*Generated by TBG AI Copilot eval pipeline — eval_scorecard.py · eval_results_v2.csv*
