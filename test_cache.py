"""
Demonstrates semantic cache improvements:
  1. Temporal guard — queries differing only by year/quarter never share a cache hit
  2. Jaccard threshold raised to 0.72 — reduced false positives on rephrased queries
  3. "for" / "in" removed from stop words — time scopes are no longer stripped
"""
import sys
sys.path.insert(0, ".")

from app.agents.semantic_cache import SemanticCache, _temporal_tokens, _content_tokens, _jaccard

SEP = "─" * 62


def show(label, value):
    print(f"  {label:<40} {value}")


# ── 1. Temporal token extraction ─────────────────────────────────
print(SEP)
print("1. TEMPORAL TOKEN EXTRACTION")
print(SEP)
pairs = [
    "Which sub-components of BFR spike in Q4?",
    "Which sub-components of BFR spike in Q4 for 2024?",
    "What was EBITDA in Q3 2025?",
    "What was EBITDA in Q3 2024?",
    "Show revenue for January",
    "Show revenue for January 2025",
]
for q in pairs:
    show(q[:50], str(_temporal_tokens(q)))
print()


# ── 2. Temporal compatibility check ──────────────────────────────
print(SEP)
print("2. TEMPORAL COMPATIBILITY (must be equal for cache hit)")
print(SEP)
from app.agents.semantic_cache import _temporal_compatible
cases = [
    ("BFR variance spike in Q4?",           "BFR variance spike in Q4 for 2024?"),
    ("EBITDA in Q3 2025",                    "EBITDA in Q3 2024"),
    ("Revenue in January",                   "Revenue in January 2025"),
    ("What is the ARPU for Q2?",             "What is the ARPU for Q2?"),
    ("Show cashflow for 2024",               "Show cashflow for 2025"),
]
for a, b in cases:
    compat = _temporal_compatible(a, b)
    flag = "✓ same period" if compat else "✗ different periods — NO cache hit"
    print(f"  Q: {a[:45]}")
    print(f"  C: {b[:45]}")
    show("→ compatible?", flag)
    print()


# ── 3. Live cache hit/miss demo ───────────────────────────────────
print(SEP)
print("3. LIVE CACHE HIT / MISS DEMO")
print(SEP)

cache = SemanticCache()

# Store an answer for Q4 (no year)
cache.set(
    question="Which sub-components of BFR spike in Q4?",
    sql="SELECT ...",
    rows=[{"metric": "Investissements bruts", "variance": -2695336762}],
    cols=["metric", "variance"],
    answer="Investissements bruts had the largest Q4 variance.",
)

print("  Stored: 'Which sub-components of BFR spike in Q4?'")
print()

tests = [
    "Which sub-components of BFR spike in Q4?",           # exact → Tier 3 HIT
    "Which BFR sub-components spike most in Q4?",          # rephrase, same period → Tier 2 HIT (Jaccard)
    "Which sub-components of BFR spike in Q4 for 2024?",  # adds year → temporal MISS
    "Which sub-components of BFR spike in Q4 for 2025?",  # different year → temporal MISS
]

for q in tests:
    result = cache.get(q)
    hit = result is not None
    print(f"  Query : {q}")
    print(f"  Result: {'HIT  ✓' if hit else 'MISS ✗'}  (tokens={_content_tokens(q)})")
    print()

stats = cache.stats()
print(f"  Cache stats — hits: {stats['hits']}, misses: {stats['misses']}, "
      f"hit rate: {stats['hit_rate']}%")
print()


# ── 4. Jaccard threshold demo ─────────────────────────────────────
print(SEP)
print("4. JACCARD TOKEN SIMILARITY (threshold = 0.72)")
print(SEP)
stored = "Which sub-components of BFR spike in Q4?"
queries = [
    "Which BFR sub-components spike most in Q4?",           # minor rephrase → HIT
    "BFR sub-components Q4 spike breakdown",                # borderline
    "What BFR elements caused the Q4 overrun?",             # more different → MISS
    "Which sub-components of BFR spike in Q4 for 2024?",   # temporal mismatch → MISS
]
s_tok = _content_tokens(stored)
for q in queries:
    q_tok = _content_tokens(q)
    jac = round(_jaccard(q_tok, s_tok), 3)
    would_hit = jac >= 0.72 and _temporal_compatible(stored, q)
    print(f"  {q[:55]}")
    show(f"  Jaccard={jac}  temporal_ok={_temporal_compatible(stored,q)}", "HIT ✓" if would_hit else "MISS ✗")
    print()
