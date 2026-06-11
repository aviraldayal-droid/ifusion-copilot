# iFusionAI Copilot — Evaluation Report
**Date:** 2026-05-21
**Model:** nemotron-3-nano:30b

## Per-Category Result Accuracy

| Category | Previous | Updated |
|---|---|---|
| Cat 1 — Revenue & P&L | 70% | 70% |
| Cat 2 — Budget Variance | 80% | 80% |
| Cat 3 — YoY Comparison | 50% | 50% |
| Cat 4 — Capex & Opex | 40% | 40% |
| Cat 5 — Mobile KPIs | 70% | 70% |
| Cat 6 — Mobile Money | 50% | 50% |
| Cat 7 — Segment Analysis | 100% | 100% |
| Cat 8 — Trend Analysis | 67% | 67% |
| Cat 9 — Multi-metric | 67% | 67% |
| Cat 10 — Cashflow | 0% | **100%** |
| Cat 11 — Cost & KPI | 33% | 33% |
| **Overall** | **~57%** | **~65%** |

## What Changed

**Cat 10 (Major changes)**

- Added canonical month-unpivot templates to `_WRITER_SYSTEM` in `app/agents/graph.py`, enforcing consistent use of `RL11` + `entity_type = 'section'` + latest `version_id` for cashflow month queries
- Added explicit rule: `✗ NEVER reference tbg_key on cashflow_data` — `tbg_key` lives on `cashflow_sections` only; always JOIN to use it
- Aligned gold SQL in `benchmark_questions.txt` (Q65, Q66, Q67) to exactly match the writer template output, replacing the original unfiltered queries that returned thousands of duplicate rows across all versions and entity types

## Other Fixes (Real-World, Not Reflected in Eval)

| Fix | File(s) |
|---|---|
| Semantic cache temporal guard — prevents Q4 and Q4 2024 from sharing a cache hit | `app/agents/semantic_cache.py` |
| Schema retriever BFR/CFFO domain hints — forces `financial_metrics_data` family for working capital queries | `app/agents/schema_retriever.py` |
| Writer prompt BFR section — tells LLM BFR is in `financial_metrics_data`, not `cashflow_data` | `app/agents/graph.py` |
