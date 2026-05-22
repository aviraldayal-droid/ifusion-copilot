"""
Global in-memory semantic cache for DB pipeline queries.

Three-tier cache (checked in order):
  1. Semantic (embedding-based, cosine ≥ 0.92) — when Ollama embeddings work
  2. Token / Jaccard (stop-word-filtered word overlap ≥ 0.72) — handles rephrasing
  3. Exact (normalized-string hash) — always works, zero-cost fallback

All tiers enforce a temporal guard: queries that reference different years,
quarters, months, or numeric values never hit each other in the cache.

TTL: 24 hours. Thread-safe.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

import numpy as np

log = logging.getLogger("tbg.cache")

_TTL_SECONDS          = 86_400   # 24 hours
_SIMILARITY_THRESHOLD = 0.92
_TOKEN_THRESHOLD      = 0.72

# Common question words that carry no semantic weight for matching purposes
# NOTE: "for", "in" intentionally kept OUT of stop words — they matter in
# phrases like "revenue for 2024" or "in Q3".  Numbers are never stripped.
_STOP_WORDS: frozenset[str] = frozenset({
    "what", "is", "the", "of", "me", "show", "give", "tell",
    "a", "an", "how", "much", "many", "do", "does", "has", "have", "can",
    "will", "would", "list", "display", "get", "find", "are", "was", "were",
    "be", "at", "its", "my", "their", "our", "this", "that", "and", "or",
    "by", "to", "from", "with", "all", "any", "on", "about", "which",
    "please", "hi", "hello", "hey",
})

# Month names / abbreviations
_MONTH_NAMES: frozenset[str] = frozenset({
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
})


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for exact-match fallback."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)        # collapse whitespace
    return text


def _content_tokens(text: str) -> frozenset[str]:
    """Return meaningful words after normalizing and removing stop words."""
    return frozenset(_normalize(text).split()) - _STOP_WORDS


def _temporal_tokens(text: str) -> frozenset[str]:
    """
    Extract tokens that represent specific time periods.
    These MUST match exactly between a query and a cached entry —
    "Q4 2024" and "Q4" are different questions even if everything else is identical.
    """
    out: set[str] = set()
    for tok in _normalize(text).split():
        if re.fullmatch(r'\d{4}', tok):             # 4-digit year: 2024, 2025
            out.add(tok)
        elif re.fullmatch(r'q[1-4]', tok):          # quarter: q1–q4
            out.add(tok)
        elif re.fullmatch(r'h[12]', tok):           # half: h1, h2
            out.add(tok)
        elif tok in _MONTH_NAMES:                   # month names / abbrevs
            out.add(tok)
        elif re.fullmatch(r'\d{1,3}', tok):         # 1–3 digit numbers (day, month, etc.)
            out.add(tok)
    return frozenset(out)


def _temporal_compatible(q_text: str, stored_text: str) -> bool:
    """
    Return True only when both queries reference the same set of time periods.
    A query with no temporal tokens is compatible only with other no-temporal queries.
    """
    return _temporal_tokens(q_text) == _temporal_tokens(stored_text)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


class SemanticCache:
    def __init__(self) -> None:
        # Tier 1 (semantic): entries with float embeddings
        self._entries: list[dict[str, Any]] = []
        # Tier 3 (exact): normalized_question → entry
        self._exact: dict[str, dict[str, Any]] = {}
        self._lock   = threading.Lock()
        self._hits   = 0
        self._misses = 0

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom > 1e-8 else 0.0

    def _evict(self) -> None:
        cutoff = time.time() - _TTL_SECONDS
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["ts"] >= cutoff]
        self._exact   = {k: v for k, v in self._exact.items() if v["ts"] >= cutoff}
        removed = before - len(self._entries)
        if removed:
            log.debug("Cache evicted %d expired entries", removed)

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, question: str, embedding: list[float] | None = None) -> dict[str, Any] | None:
        """
        Return the best matching cached entry, or None on miss.

        Tier 1: cosine similarity ≥ 0.92 on embeddings (when available)
        Tier 2: Jaccard ≥ 0.72 on content tokens (handles rephrasing without embeddings)
        Tier 3: exact normalized-string hash (zero-cost fallback)

        All tiers apply a temporal guard: entries whose time-period tokens
        (years, quarters, months, numbers) differ from the query are skipped.
        """
        with self._lock:
            self._evict()

            all_entries = list(self._exact.values())

            # Tier 1: semantic similarity (only when embedding available)
            if embedding is not None and self._entries:
                candidates = [
                    e for e in self._entries
                    if e.get("embedding") is not None
                    and _temporal_compatible(question, e["question"])
                ]
                if candidates:
                    sims = [(self._cosine(embedding, e["embedding"]), e)
                            for e in candidates]
                    best_sim, best_entry = max(sims, key=lambda x: x[0])
                    if best_sim >= _SIMILARITY_THRESHOLD:
                        self._hits += 1
                        log.info("Cache HIT (semantic) sim=%.3f  q=%s",
                                 best_sim, best_entry["question"][:70])
                        return best_entry

            # Tier 2: Jaccard token similarity (bridges rephrasing gap)
            q_tokens = _content_tokens(question)
            temporal_filtered = [
                e for e in all_entries
                if _temporal_compatible(question, e["question"])
            ]
            if q_tokens and temporal_filtered:
                best_jac, best_entry = max(
                    ((_jaccard(q_tokens, _content_tokens(e["question"])), e)
                     for e in temporal_filtered),
                    key=lambda x: x[0],
                )
                if best_jac >= _TOKEN_THRESHOLD:
                    self._hits += 1
                    log.info("Cache HIT (token)  jac=%.2f  stored=%s  q=%s",
                             best_jac, best_entry["question"][:60], question[:60])
                    return best_entry

            # Tier 3: exact normalized-string match (temporal already guaranteed equal)
            key = _normalize(question)
            if key in self._exact:
                self._hits += 1
                log.info("Cache HIT (exact)  q=%s", question[:70])
                return self._exact[key]

            self._misses += 1
            log.debug("Cache MISS  q=%s", question[:70])
            return None

    def set(
        self,
        question: str,
        sql: str,
        rows: list[dict],
        cols: list[str],
        answer: str,
        embedding: list[float] | None = None,
    ) -> None:
        """Store a new cache entry in both tiers."""
        entry = dict(
            embedding=embedding,
            question=question,
            sql=sql,
            rows=rows,
            cols=cols,
            answer=answer,
            ts=time.time(),
        )
        with self._lock:
            self._evict()

            # Tier 1: update or append semantic entry
            if embedding is not None:
                for e in self._entries:
                    if e.get("embedding") and self._cosine(embedding, e["embedding"]) > 0.999:
                        e.update(entry)
                        log.debug("Cache UPDATE semantic: %s", question[:70])
                        break
                else:
                    self._entries.append(entry)

            # Tier 3: always store exact-match entry (also feeds Tier 2 scan)
            key = _normalize(question)
            self._exact[key] = entry

            log.info("Cache STORE semantic=%d exact=%d  q=%s",
                     len(self._entries), len(self._exact), question[:70])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._evict()
            return {
                "semantic_entries": len(self._entries),
                "exact_entries":    len(self._exact),
                "hits":             self._hits,
                "misses":           self._misses,
                "hit_rate":         round(self._hits / max(self._hits + self._misses, 1) * 100, 1),
                "ttl_hours":        _TTL_SECONDS // 3600,
                "thresholds": {
                    "semantic_cosine": _SIMILARITY_THRESHOLD,
                    "token_jaccard":   _TOKEN_THRESHOLD,
                },
            }

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries) + len(self._exact)
            self._entries.clear()
            self._exact.clear()
            self._hits = self._misses = 0
            log.info("Cache cleared (%d entries removed)", n)
            return n


# Global singleton
semantic_cache = SemanticCache()
