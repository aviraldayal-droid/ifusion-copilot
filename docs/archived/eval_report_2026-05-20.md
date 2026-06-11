# TBG AI Copilot — Model Evaluation Report
**Date:** 2026-05-20  
**Benchmark:** 73 question–SQL pairs across 11 categories  
**Models:** `nemotron-3-nano:30b` (all categories) · `devstral-2:123b` (Category 5 only)

---

## Metric Definitions

| Metric | Description |
|--------|-------------|
| **Result Match** | AI SQL returns the exact same result set as the gold SQL (`result_match = True`) |
| **Execution Rate** | % of queries that ran without a DB runtime error (`executed = True`) |
| **Syntax Pass** | SQL parsed successfully by `sqlglot` before hitting the DB |
| **Tables Pass** | All tables in the generated SQL are present in the retrieved schema |
| **Semantic Pass** | `EXPLAIN` on the live DB succeeded (column/type validation) |
| **Retrieval Recall** | All gold tables were present in the retrieved schema context |
| **Avg Retries** | Average number of critic-loop retries per question |
| **Avg Latency** | Total end-to-end time per question (includes retries) |
| **Avg SQL Gen** | Time spent by the writer LLM generating SQL |

---

## Summary — All Categories

| Cat | Topic | Model | Qs | Accuracy | Exec Rate | Retrieval Recall | Avg Retries | Avg Latency |
|-----|-------|-------|----|----------|-----------|-----------------|-------------|-------------|
| 1 | Simple Lookups | nemotron-3-nano:30b | 10 | **70%** | 100% | 100% | 0.1 | 7.8 s |
| 2 | Aggregations | nemotron-3-nano:30b | 10 | **80%** | 90% | 100% | 0.1 | 12.6 s |
| 3 | Date Ranges | nemotron-3-nano:30b | 10 | **50%** | 100% | 100% | 0.7 | 16.3 s |
| 4 | Joins | nemotron-3-nano:30b | 10 | **40%** | 100% | 90% | 0.2 | 9.4 s |
| 5 | Edge Cases | devstral-2:123b | 10 | **40%** | 90% | 90% | 0.5 | 10.1 s |
| 5 | Edge Cases | nemotron-3-nano:30b | 10 | **70%** | 100% | 100% | 0.1 | 12.9 s |
| 6 | Revenue & Growth | nemotron-3-nano:30b | 4 | **50%** | 100% | 100% | 0.25 | 14.1 s |
| 7 | Profitability & Margins | nemotron-3-nano:30b | 1 | **100%** | 100% | 100% | 0.0 | 11.8 s |
| 8 | Operating Expenses | nemotron-3-nano:30b | 3 | **67%** | 100% | 100% | 0.67 | 13.0 s |
| 9 | Capex Analysis | nemotron-3-nano:30b | 3 | **67%** | 100% | 100% | 0.67 | 19.1 s |
| 10 | Cash Flow & Liquidity | nemotron-3-nano:30b | 3 | **0%** | 100% | 100% | 0.67 | 32.4 s |
| 11 | Cost Analysis & KPI | nemotron-3-nano:30b | 6 | **33%** | 100% | 83% | 0.67 | 20.0 s |

**Overall (nemotron-3-nano:30b, 70 questions):** 40/70 = **57.1% accuracy**  
**Overall (devstral-2:123b, 10 questions):** 4/10 = **40.0% accuracy**

---

## Category 1 — Simple Lookups
**Model:** `nemotron-3-nano:30b` | **Questions:** 10 (Q1–Q10)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **7 / 10 (70%)** |
| Execution Rate | 10 / 10 (100%) |
| Syntax Pass | 10 / 10 (100%) |
| Tables Pass | 10 / 10 (100%) |
| Semantic Pass | 10 / 10 (100%) |
| Retrieval Recall | 10 / 10 (100%) |
| Avg Retries | 0.1 |
| Avg Latency | 7.76 s |
| Avg SQL Gen Time | 5.43 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows |
|---|-------------|---------|---------|-----------|------------|
| 1 | ❌ | 0 | 7.88 s | 2 | 1 |
| 2 | ✅ | 0 | 5.73 s | 0 | 0 |
| 3 | ❌ | 0 | 8.53 s | 1 | 0 |
| 4 | ✅ | 0 | 3.80 s | 126 | 126 |
| 5 | ✅ | 0 | 4.83 s | 0 | 0 |
| 6 | ❌ | 0 | 5.23 s | 55 | 1 |
| 7 | ✅ | 0 | 4.39 s | 1 | 1 |
| 8 | ✅ | 1 | 15.03 s | 3 | 3 |
| 9 | ✅ | 0 | 18.12 s | 1 | 1 |
| 10 | ✅ | 0 | 4.09 s | 1 | 1 |

**Failures:** Q1, Q3 return fewer rows than gold (deduplication mismatch). Q6 returns 1 row vs 55 (aggregation applied when it shouldn't be).

---

## Category 2 — Aggregations
**Model:** `nemotron-3-nano:30b` | **Questions:** 10 (Q11–Q20)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **8 / 10 (80%)** |
| Execution Rate | 9 / 10 (90%) |
| Syntax Pass | 10 / 10 (100%) |
| Tables Pass | 10 / 10 (100%) |
| Semantic Pass | 10 / 10 (100%) |
| Retrieval Recall | 10 / 10 (100%) |
| Avg Retries | 0.1 |
| Avg Latency | 12.63 s |
| Avg SQL Gen Time | 5.08 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows | Error |
|---|-------------|---------|---------|-----------|------------|-------|
| 11 | ✅ | 0 | 7.48 s | 1 | 1 | — |
| 12 | ✅ | 0 | 5.68 s | 1 | 1 | — |
| 13 | ✅ | 0 | 8.26 s | 12 | 12 | — |
| 14 | ✅ | 0 | 6.53 s | 1 | 1 | — |
| 15 | ❌ | 0 | 7.55 s | 1 | 1 | — |
| 16 | ✅ | 0 | 3.88 s | 1 | 1 | — |
| 17 | ✅ | 0 | 7.26 s | 91 | 91 | — |
| 18 | ✅ | 0 | 7.14 s | 1 | 1 | — |
| 19 | ✅ | 1 | 63.09 s | 1 | 1 | — |
| 20 | ❌ (no exec) | 0 | 9.47 s | 2 | 0 | Invalid enum value "5" for Month |

**Failures:** Q15 row count matches but values differ. Q20 runtime error — model passed `5` (integer) to an enum Month column instead of the string label.

---

## Category 3 — Date Ranges
**Model:** `nemotron-3-nano:30b` | **Questions:** 10 (Q21–Q30)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **5 / 10 (50%)** |
| Execution Rate | 10 / 10 (100%) |
| Syntax Pass | 10 / 10 (100%) |
| Tables Pass | 10 / 10 (100%) |
| Semantic Pass | 10 / 10 (100%) |
| Retrieval Recall | 10 / 10 (100%) |
| Avg Retries | 0.7 |
| Avg Latency | 16.29 s |
| Avg SQL Gen Time | 7.98 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows |
|---|-------------|---------|---------|-----------|------------|
| 21 | ✅ | 0 | 10.70 s | 6 | 6 |
| 22 | ✅ | 0 | 7.57 s | 1 | 1 |
| 23 | ✅ | 0 | 5.46 s | 1 | 1 |
| 24 | ✅ | 0 | 11.23 s | 0 | 0 |
| 25 | ❌ | 3 | 41.65 s | 6324 | 2 |
| 26 | ✅ | 1 | 15.54 s | 0 | 0 |
| 27 | ✅ | 0 | 7.46 s | 12 | 12 |
| 28 | ❌ | 1 | 21.79 s | 6269 | 51 |
| 29 | ❌ | 0 | 11.44 s | 4 | 4 |
| 30 | ❌ | 2 | 30.03 s | 27 | 9 |

**Failures:** Q25, Q28, Q30 — large row-count discrepancies indicating missing date filters or wrong GROUP BY granularity. Q29 — row count matches but column values differ (ordering/aggregation issue).

---

## Category 4 — Joins
**Model:** `nemotron-3-nano:30b` | **Questions:** 10 (Q31–Q40)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **4 / 10 (40%)** |
| Execution Rate | 10 / 10 (100%) |
| Syntax Pass | 10 / 10 (100%) |
| Tables Pass | 10 / 10 (100%) |
| Semantic Pass | 10 / 10 (100%) |
| Retrieval Recall | 9 / 10 (90%) |
| Avg Retries | 0.2 |
| Avg Latency | 9.37 s |
| Avg SQL Gen Time | 6.06 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows | Retrieval Recall |
|---|-------------|---------|---------|-----------|------------|-----------------|
| 31 | ✅ | 0 | 6.87 s | 7 | 7 | ✅ |
| 32 | ✅ | 0 | 8.90 s | 384 | 384 | ✅ |
| 33 | ✅ | 0 | 5.41 s | 1 | 1 | ✅ |
| 34 | ❌ | 0 | 6.31 s | 12 | 12 | ✅ |
| 35 | ❌ | 1 | 16.07 s | 228 | 232 | ✅ |
| 36 | ❌ | 0 | 7.39 s | 154 | 154 | ✅ |
| 37 | ❌ | 0 | 12.28 s | 154 | 126 | ✅ |
| 38 | ❌ | 1 | 16.53 s | 7 | 0 | ❌ |
| 39 | ✅ | 0 | 5.33 s | 91 | 91 | ✅ |
| 40 | ❌ | 0 | 8.63 s | 24 | 12 | ✅ |

**Failures:** Q34/Q36 — row counts match but values differ (likely ROUND or cast mismatch). Q35/Q37/Q40 — wrong row counts from JOIN fan-out or missing deduplication. Q38 — retrieval miss (gold table `capex_data` not retrieved), returns 0 rows.

---

## Category 5 — Edge Cases
**Questions:** 10 (Q41–Q50) | **Both models tested**

### devstral-2:123b

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **4 / 10 (40%)** |
| Execution Rate | 9 / 10 (90%) |
| Syntax Pass | 10 / 10 (100%) |
| Tables Pass | 10 / 10 (100%) |
| Semantic Pass | 9 / 10 (90%) |
| Retrieval Recall | 9 / 10 (90%) |
| Avg Retries | 0.5 |
| Avg Latency | 10.06 s |
| Avg SQL Gen Time | 4.59 s |

### nemotron-3-nano:30b

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **7 / 10 (70%)** |
| Execution Rate | 10 / 10 (100%) |
| Syntax Pass | 10 / 10 (100%) |
| Tables Pass | 10 / 10 (100%) |
| Semantic Pass | 10 / 10 (100%) |
| Retrieval Recall | 10 / 10 (100%) |
| Avg Retries | 0.1 |
| Avg Latency | 12.93 s |
| Avg SQL Gen Time | 9.57 s |

**Per-question breakdown (both models):**

| Q | devstral Match | devstral Retries | nemotron Match | nemotron Retries |
|---|---------------|-----------------|----------------|-----------------|
| 41 | ❌ | 1 | ❌ | 0 |
| 42 | ❌ | 0 | ❌ | 0 |
| 43 | ❌ | 0 | ✅ | 0 |
| 44 | ✅ | 0 | ✅ | 0 |
| 45 | ✅ | 1 | ✅ | 0 |
| 46 | ✅ | 0 | ✅ | 0 |
| 47 | ✅ | 0 | ✅ | 0 |
| 48 | ❌ | 0 | ❌ | 0 |
| 49 | ❌ (no exec) | 3 | ✅ | 0 |
| 50 | ❌ | 0 | ✅ | 1 |

**Notes:** devstral Q49 failed semantic validation (`round(double precision, integer)` function not found — PostgreSQL-dialect mismatch). nemotron handled Q49 correctly. Both models fail Q41, Q42, Q48 — these are the hardest edge cases involving large fan-out queries.

---

## Category 6 — Revenue & Growth
**Model:** `nemotron-3-nano:30b` | **Questions:** 4 (Q51–Q54)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **2 / 4 (50%)** |
| Execution Rate | 4 / 4 (100%) |
| Syntax Pass | 4 / 4 (100%) |
| Tables Pass | 4 / 4 (100%) |
| Semantic Pass | 4 / 4 (100%) |
| Retrieval Recall | 4 / 4 (100%) |
| Avg Retries | 0.25 |
| Avg Latency | 14.06 s |
| Avg SQL Gen Time | 9.14 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows |
|---|-------------|---------|---------|-----------|------------|
| 51 | ✅ | 0 | 13.61 s | 5 | 5 |
| 52 | ❌ | 1 | 22.76 s | 3 | 1 |
| 53 | ❌ | 0 | 12.85 s | 2 | 1 |
| 54 | ✅ | 0 | 7.01 s | 5 | 5 |

**Failures:** Q52, Q53 return fewer rows (1 instead of 2–3) — likely missing a UNION or additional revenue segment in the query.

---

## Category 7 — Profitability & Margins
**Model:** `nemotron-3-nano:30b` | **Questions:** 1 (Q55)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **1 / 1 (100%)** |
| Execution Rate | 1 / 1 (100%) |
| Syntax Pass | 1 / 1 (100%) |
| Tables Pass | 1 / 1 (100%) |
| Semantic Pass | 1 / 1 (100%) |
| Retrieval Recall | 1 / 1 (100%) |
| Avg Retries | 0.0 |
| Avg Latency | 11.75 s |
| Avg SQL Gen Time | 10.02 s |

> ⚠️ Only 1 question evaluated — not statistically significant.

---

## Category 8 — Operating Expenses
**Model:** `nemotron-3-nano:30b` | **Questions:** 3 (Q59–Q61)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **2 / 3 (67%)** |
| Execution Rate | 3 / 3 (100%) |
| Syntax Pass | 3 / 3 (100%) |
| Tables Pass | 3 / 3 (100%) |
| Semantic Pass | 3 / 3 (100%) |
| Retrieval Recall | 3 / 3 (100%) |
| Avg Retries | 0.67 |
| Avg Latency | 12.99 s |
| Avg SQL Gen Time | 6.75 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows |
|---|-------------|---------|---------|-----------|------------|
| 59 | ✅ | 2 | 22.10 s | 5 | 5 |
| 60 | ❌ | 0 | 7.67 s | 126 | 108 |
| 61 | ✅ | 0 | 9.21 s | 12 | 12 |

**Failure:** Q59 required 2 retries before succeeding. Q60 — model returns 108 vs 126 rows (missing ~15% of records, likely a WHERE clause that is too restrictive).

---

## Category 9 — Capex Analysis
**Model:** `nemotron-3-nano:30b` | **Questions:** 3 (Q62–Q64)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **2 / 3 (67%)** |
| Execution Rate | 3 / 3 (100%) |
| Syntax Pass | 3 / 3 (100%) |
| Tables Pass | 3 / 3 (100%) |
| Semantic Pass | 3 / 3 (100%) |
| Retrieval Recall | 3 / 3 (100%) |
| Avg Retries | 0.67 |
| Avg Latency | 19.06 s |
| Avg SQL Gen Time | 7.20 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows |
|---|-------------|---------|---------|-----------|------------|
| 62 | ✅ | 2 | 40.41 s | 1 | 1 |
| 63 | ✅ | 0 | 6.64 s | 1 | 1 |
| 64 | ❌ | 0 | 10.14 s | 12 | 12 |

**Failure:** Q62 required 2 retries (high latency 40 s). Q64 — row count matches but values differ (incorrect aggregation or column selection in capex join).

---

## Category 10 — Cash Flow & Liquidity
**Model:** `nemotron-3-nano:30b` | **Questions:** 3 (Q65–Q67)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **0 / 3 (0%)** |
| Execution Rate | 3 / 3 (100%) |
| Syntax Pass | 3 / 3 (100%) |
| Tables Pass | 3 / 3 (100%) |
| Semantic Pass | 3 / 3 (100%) |
| Retrieval Recall | 3 / 3 (100%) |
| Avg Retries | 0.67 |
| Avg Latency | 32.39 s |
| Avg SQL Gen Time | 17.58 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows |
|---|-------------|---------|---------|-----------|------------|
| 65 | ❌ | 1 | 41.65 s | 2150 | 278 |
| 66 | ❌ | 0 | 30.04 s | 25738 | 612 |
| 67 | ❌ | 1 | 25.48 s | 6269 | 0 |

**Failures:** All 3 questions fail. Q65/Q66 — massive row-count shortfall (model drastically under-aggregates cashflow data). Q67 — returns 0 rows (likely a join condition eliminates all rows). This category has the highest avg latency (32.4 s) and highest SQL gen time (17.6 s), suggesting the model struggles with cashflow schema complexity.

---

## Category 11 — Cost Analysis & KPI
**Model:** `nemotron-3-nano:30b` | **Questions:** 6 (Q68–Q73)

| Metric | Value |
|--------|-------|
| Result Match (Accuracy) | **2 / 6 (33%)** |
| Execution Rate | 6 / 6 (100%) |
| Syntax Pass | 6 / 6 (100%) |
| Tables Pass | 6 / 6 (100%) |
| Semantic Pass | 6 / 6 (100%) |
| Retrieval Recall | 5 / 6 (83%) |
| Avg Retries | 0.67 |
| Avg Latency | 19.97 s |
| Avg SQL Gen Time | 11.23 s |

**Per-question breakdown:**

| Q | Result Match | Retries | Latency | Gold Rows | Model Rows | Retrieval Recall |
|---|-------------|---------|---------|-----------|------------|-----------------|
| 68 | ✅ | 3 | 40.46 s | 1 | 1 | ❌ |
| 69 | ❌ | 0 | 14.19 s | 4 | 1 | ✅ |
| 70 | ❌ | 1 | 27.04 s | 108 | 12 | ✅ |
| 71 | ❌ | 0 | 10.75 s | 143 | 140 | ✅ |
| 72 | ✅ | 0 | 14.12 s | 12 | 12 | ✅ |
| 73 | ❌ | 0 | 13.26 s | 2 | 1 | ✅ |

**Failures:** Q68 succeeded but needed 3 retries and the gold table `monthly_evolution` was not in the retrieved context (model found an alternative path). Q69/Q73 — under-aggregating commission data (returns 1 instead of 2–4 rows). Q70 — 108 vs 12 rows (wrong GROUP BY granularity). Q71 — 143 vs 140 rows (near-miss, likely a boundary condition on date filter).

---

## Overall Performance — nemotron-3-nano:30b

| Metric | Value |
|--------|-------|
| **Total Questions** | 70 |
| **Result Match (Accuracy)** | **40 / 70 = 57.1%** |
| **Execution Rate** | 69 / 70 = 98.6% |
| **Syntax Pass Rate** | 70 / 70 = 100% |
| **Tables Pass Rate** | 70 / 70 = 100% |
| **Semantic Pass Rate** | 70 / 70 = 100% |
| **Retrieval Recall** | 68 / 70 = 97.1% |
| **Avg Retries** | 0.34 |
| **Avg Latency** | 14.5 s |

### Accuracy by Category (nemotron-3-nano:30b)

```
Cat  1  Simple Lookups        ████████████████████░░░░░░░░░  70%
Cat  2  Aggregations          ████████████████████████░░░░░  80%
Cat  3  Date Ranges           ███████████████░░░░░░░░░░░░░░  50%
Cat  4  Joins                 ████████████░░░░░░░░░░░░░░░░░  40%
Cat  5  Edge Cases            ████████████████████░░░░░░░░░  70%
Cat  6  Revenue & Growth      ███████████████░░░░░░░░░░░░░░  50%
Cat  7  Profitability         █████████████████████████████ 100%
Cat  8  OpEx                  █████████████████████░░░░░░░░  67%
Cat  9  Capex                 █████████████████████░░░░░░░░  67%
Cat 10  Cash Flow             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Cat 11  Cost & KPI            ██████████░░░░░░░░░░░░░░░░░░░  33%
```

---

## Key Observations

1. **Pipeline is healthy** — 100% syntax pass, 100% tables pass, 100% semantic pass across all 70 questions. Every generated query is structurally valid and executable.

2. **Retrieval is reliable** — 97.1% recall. Only 2 misses (Q38 in Cat 4, Q68 in Cat 11), and Q68 still succeeded via an alternative table.

3. **Aggregation/row-count is the primary failure mode** — Most failures are not syntax or schema errors but wrong result cardinality: the model returns the right tables but applies wrong GROUP BY granularity, missing deduplication windows, or overly restrictive WHERE clauses.

4. **Cat 10 (Cash Flow) is the hardest** — 0% accuracy. All 3 queries execute cleanly but return drastically wrong row counts (e.g., 612 vs 25,738). The `cashflow_data` schema likely requires specific partition-aware aggregation that the model isn't inferring.

5. **devstral-2:123b underperforms nemotron on Cat 5** — 40% vs 70%. devstral fails a semantic validation (PostgreSQL `round()` function arity) that nemotron passes, and has worse edge-case handling overall.

6. **Retries are most costly in Cat 10 & 11** — Avg latency 32.4 s and 20.0 s respectively. Cat 10's SQL gen time (17.6 s avg) indicates the model is generating complex queries that still fail.
