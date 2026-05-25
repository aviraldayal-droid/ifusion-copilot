"""
RAG-based schema retriever — v2

Schema source: schema_doc.md (parsed once at startup; falls back to live DB
               inspection if the file is missing).
Embeddings:    cached to disk in app/agents/schema_cache.{npz,json}.
               Cache is invalidated automatically when schema_doc.md or
               schema_descriptions.yaml changes, or when the embedding model
               changes.  On a cache hit startup is instant (no Ollama calls).
               On a cache miss all tables are embedded then saved to disk.

Public API is unchanged:
    retriever = get_schema_retriever()
    result    = retriever.retrieve(question)   → RetrievedSchema
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app.config.settings import settings
from app.db.schema_inspector import ColumnInfo, ForeignKey, TableInfo, inspect_schema

log = logging.getLogger("tbg.retriever")

# ── File paths ────────────────────────────────────────────────────────────────

_AGENTS_DIR       = Path(__file__).parent
_SCHEMA_DOC       = _AGENTS_DIR.parent.parent / "schema_doc.md"
_DESCRIPTIONS_FILE = _AGENTS_DIR / "schema_descriptions.yaml"
_CACHE_NPZ        = _AGENTS_DIR / "schema_cache.npz"
_CACHE_META       = _AGENTS_DIR / "schema_cache.json"

# ── Retrieval config ──────────────────────────────────────────────────────────

_TOP_K_DEFAULT = 4
_MAX_EXPANDED  = 10

# ── Domain hints ──────────────────────────────────────────────────────────────

_FMT = ["financial_metrics_data", "financial_metric", "financial_types", "financial_categories"]
_RRD = ["revenue_raw_data"]

_DOMAIN_HINTS: dict[str, list[str]] = {
    # ── Financial metrics (core) ──────────────────────────────────────────────
    "arpu":           _FMT,
    "revenue":        _RRD + _FMT,
    "recette":        _RRD + _FMT,
    "breakdown":      _RRD + _FMT,
    "repartition":    _RRD + _FMT,
    "category":       _FMT,
    "categorie":      _FMT,
    "p&l":            _FMT,
    "ebitda":         _FMT,
    "ebita":          _FMT,
    "sortant":        ["financial_metrics_data", "financial_metric", "financial_types"],
    "entrant":        ["financial_metrics_data", "financial_metric", "financial_types"],
    "budget":         _FMT,
    "kpi":            ["financial_metrics_data", "financial_metric", "financial_types"],
    "metric":         ["financial_metrics_data", "financial_metric"],
    "metrique":       ["financial_metrics_data", "financial_metric"],
    "financial":      _FMT,
    # ── Sales / product / trend — route unknown product queries to financial metrics
    "sales":          _FMT,
    "ventes":         _FMT,
    "vente":          _FMT,
    "decline":        _FMT,
    "declin":         _FMT,
    "baisse":         _FMT,
    "hausse":         _FMT,
    "growth":         _RRD + _FMT,
    "croissance":     _RRD + _FMT,
    "trend":          _RRD + _FMT,
    "tendance":       _RRD + _FMT,
    "performance":    _FMT,
    "product":        _FMT,
    "produit":        _FMT,
    "service":        _FMT,
    "chiffre":        _FMT,
    "lifecycle":      _FMT,
    "cycle de vie":   _FMT,
    "market":         _FMT,
    "marché":         _FMT,
    "parc":           _FMT,
    "subscriber":     _FMT,
    "abonné":         _FMT,
    "abonnés":        _FMT,
    "internet":       _FMT,
    "data":           _FMT,
    "voix":           _FMT,
    "sms":            _FMT,
    "clé":            _FMT,
    "cle":            _FMT,
    "dongle":         _FMT,
    "modem":          _FMT,
    # ── BFR / Working capital ─────────────────────────────────────────────────
    "bfr":                  _FMT,
    "cffo":                 _FMT,
    "working capital":      _FMT,
    "besoin en fonds":      _FMT,
    "fonds de roulement":   _FMT,
    "variation de bfr":     _FMT,
    "sub-component":        _FMT,
    "sub component":        _FMT,
    "composant":            _FMT,
    "composants":           _FMT,
    "variance spike":       _FMT,
    "overrun":              _FMT,
    "ecart budgetaire":     _FMT,
    "écart budgétaire":     _FMT,
    # ── Cashflow ──────────────────────────────────────────────────────────────
    "flux":           ["cashflow_data", "cashflow_sections", "cashflow_categories", "realised_cashflow"],
    "cashflow":       ["cashflow_data", "cashflow_sections", "cashflow_categories", "realised_cashflow"],
    "cash flow":      ["cashflow_data", "cashflow_sections", "cashflow_categories", "realised_cashflow"],
    "tresorerie":     ["cashflow_data", "cashflow_sections", "cashflow_categories"],
    # ── OpEx ──────────────────────────────────────────────────────────────────
    "opex":           _FMT,
    "operating expense": _FMT,
    "operating cost": _FMT,
    "opex driver":    _FMT,
    "opex drivers":   _FMT,
    "coût opérationnel": _FMT,
    "charges":        _FMT,
    "redevance":      _FMT,
    "frais":          _FMT,
    "exploitation":   _FMT,
    # ── Capex ─────────────────────────────────────────────────────────────────
    "capex":              ["capex_data", "capex_projects"],
    "investissement":     ["capex_data", "capex_projects"],
    "additional_costs":   ["capex_data", "capex_projects"],
    "additional costs":   ["capex_data", "capex_projects"],
    "cout supplementaire":["capex_data", "capex_projects"],
    "frais additionnels": ["capex_data", "capex_projects"],
    "direction":          ["capex_data", "capex_projects"],
    "supplier":           ["capex_data", "capex_projects"],
    "fournisseur":        ["capex_data", "capex_projects"],
    "equipment":          ["capex_data", "capex_projects"],
    "equipement":         ["capex_data", "capex_projects"],
    "project_title":      ["capex_data", "capex_projects"],
    "project title":      ["capex_data", "capex_projects"],
    # ── Commissions ───────────────────────────────────────────────────────────
    "enlevement":     ["commission_enlevements"],
    "enlevements":    ["commission_enlevements"],
    "commission":     ["commission_enlevements", "commission_calculation_rules", "commission_types"],
    "distributor":    ["commission_enlevements"],
    "distributeur":   ["commission_enlevements"],
    # ── Mobile Money ──────────────────────────────────────────────────────────
    "momo":           ["moov_money_data"],
    "mobile money":   ["moov_money_data"],
    "moov money":     ["moov_money_data"],
    "adoption rate":  ["moov_money_data"] + _FMT,
    "adoption":       ["moov_money_data"] + _FMT,
    # ── Revenue by segment (daily) ───────────────────────────────────────────
    "ca global":      _RRD,
    "ca voix":        _RRD,
    "ca data":        _RRD + _FMT,
    "ca forfait":     _RRD,
    "forfaits":       _RRD + _FMT,
    "rechargement":   _RRD,
    "segment":        _RRD + _FMT,
    "year-over-year": _RRD + _FMT,
    "yoy":            _RRD + _FMT,
    "evolution":      ["monthly_evolution"] + _FMT,
    "growth rate":    _RRD + _FMT,
    "revenue growth": _RRD,
    "daily revenue":  _RRD,
    "gross add":      _RRD,
    "net add":        _RRD,
    "churn":          _RRD + _FMT,
    "trafic voix":    _RRD + _FMT,
    "trafic data":    _RRD + _FMT,
}

# ── Blocklist ─────────────────────────────────────────────────────────────────

_BLOCKLISTED_TABLES: frozenset[str] = frozenset({
    "DownloadRequests", "SequelizeMeta",
    "activity_events", "alert_notifications", "app_icons",
    "archive_registry", "archive_storage_kpis",
    "configuration", "configuration_archive",
    "country_module_configs", "data_lineage_notification",
    "decoder_configuration", "feature_prerequisites", "feature_section",
    "health_apps", "health_checks", "import_sources",
    "login_events", "mail_recipient_lists",
    "modules", "navigation_items",
    "network_registry", "nifi_connection_ids",
    "notification_tracking", "notifications",
    "portal_settings", "si_registry", "smtp_details",
    "superset_dashboard", "theme_configurations",
    "upload_component_types", "upload_sheet_types",
    "upload_tbg_file_types", "vendor_registry",
    "sage_yexptdb", "tbg_capex_project_details",
    "revenue_uploaded_files", "data_cormat_upload_files",
    "data_dormant_upload_files", "upload_data",
})


# ── schema_doc.md parser ──────────────────────────────────────────────────────

_TABLE_HEADER_RE = re.compile(
    r"^##\s+`([^`]+)`\s*(?:_\(~([\d,]+)\s*rows?\)_)?",
    re.MULTILINE,
)
# Matches a data row (not the header/separator rows) in the column table.
# Group 1: col name, 2: type, 3: nullable (✓/✗), 4: notes
_COL_ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([^`]*)`\s*\|\s*([✓✗])\s*\|[^|]*\|\s*(.*?)\s*\|",
    re.MULTILINE,
)
_FK_NOTE_RE = re.compile(r"FK\s*→\s*`([^.`]+)\.([^`]+)`")


def _parse_schema_doc(path: Path) -> list[TableInfo]:
    """Parse schema_doc.md into TableInfo objects (same type as inspect_schema())."""
    text   = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n---\n", text)

    result: list[TableInfo] = []
    for block in blocks:
        m = _TABLE_HEADER_RE.search(block)
        if not m:
            continue

        table_name    = m.group(1)
        row_count_str = m.group(2)
        row_count     = int(row_count_str.replace(",", "")) if row_count_str else None

        columns:      list[ColumnInfo]  = []
        foreign_keys: list[ForeignKey]  = []

        for cm in _COL_ROW_RE.finditer(block):
            col_name = cm.group(1)
            col_type = cm.group(2)
            nullable = cm.group(3) == "✓"
            notes    = cm.group(4)

            is_pk = "**PK**" in notes
            columns.append(ColumnInfo(
                name=col_name,
                data_type=col_type,
                nullable=nullable,
                default=None,
                is_pk=is_pk,
            ))

            fk_m = _FK_NOTE_RE.search(notes)
            if fk_m:
                foreign_keys.append(ForeignKey(
                    column=col_name,
                    ref_table=fk_m.group(1),
                    ref_column=fk_m.group(2),
                    constraint_name="",
                ))

        if columns:
            result.append(TableInfo(
                schema="public",
                name=table_name,
                columns=columns,
                foreign_keys=foreign_keys,
                row_count_estimate=row_count,
            ))

    log.info("schema_doc.md parsed: %d tables", len(result))
    return result


# ── YAML descriptions loader ──────────────────────────────────────────────────

def _load_descriptions() -> dict[str, dict]:
    if not _DESCRIPTIONS_FILE.exists():
        log.warning("schema_descriptions.yaml not found — using column-only embeddings")
        return {}
    with _DESCRIPTIONS_FILE.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    log.info("Loaded human descriptions for %d tables", len(data))
    return data


# ── Disk cache ────────────────────────────────────────────────────────────────

def _source_hash(model_name: str) -> str:
    """SHA-256 of schema_doc.md + schema_descriptions.yaml + model name."""
    h = hashlib.sha256()
    for p in (_SCHEMA_DOC, _DESCRIPTIONS_FILE):
        if p.exists():
            h.update(p.read_bytes())
    h.update(model_name.encode())
    return h.hexdigest()[:24]


def _load_embed_cache(expected_hash: str) -> dict[str, np.ndarray] | None:
    if not _CACHE_NPZ.exists() or not _CACHE_META.exists():
        return None
    try:
        meta = json.loads(_CACHE_META.read_text())
        if meta.get("hash") != expected_hash:
            log.info("Embedding cache invalid (source changed) — will re-embed")
            return None
        data  = np.load(_CACHE_NPZ)
        names = meta["names"]
        vecs  = data["vectors"]
        log.info("Embedding cache hit: %d tables loaded from disk", len(names))
        return {name: vecs[i] for i, name in enumerate(names)}
    except Exception as exc:
        log.warning("Could not load embedding cache (%s) — will re-embed", exc)
        return None


def _save_embed_cache(vecs: dict[str, np.ndarray], source_hash: str) -> None:
    try:
        names   = list(vecs.keys())
        vectors = np.stack([vecs[n] for n in names]).astype(np.float32)
        np.savez_compressed(str(_CACHE_NPZ), vectors=vectors)
        _CACHE_META.write_text(json.dumps({"hash": source_hash, "names": names}))
        log.info("Saved embedding cache: %d tables → %s", len(names), _CACHE_NPZ.name)
    except Exception as exc:
        log.warning("Could not save embedding cache: %s", exc)


# ── Public result type ────────────────────────────────────────────────────────

@dataclass
class RetrievedSchema:
    tables:         list[TableInfo]
    schema_text:    str
    allowed_tables: set[str]


# ── Retriever ─────────────────────────────────────────────────────────────────

class SchemaRetriever:
    """Thread-safe singleton.  Call build() once; retrieve() is thread-safe."""

    def __init__(self) -> None:
        self._lock         = threading.Lock()
        self._ready        = False
        self._infos:       list[TableInfo]       = []
        self._texts:       dict[str, str]        = {}
        self._vecs:        dict[str, np.ndarray] = {}
        self._fk_out:      dict[str, set[str]]   = {}
        self._embed_model  = None
        self._descriptions: dict[str, dict]      = {}

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self) -> None:
        with self._lock:
            if self._ready:
                return

            self._descriptions = _load_descriptions()

            # Load schema structure — prefer schema_doc.md, fall back to live DB
            if _SCHEMA_DOC.exists():
                all_infos = _parse_schema_doc(_SCHEMA_DOC)
            else:
                log.warning("schema_doc.md not found — falling back to live DB inspection")
                all_infos = inspect_schema()

            self._infos = [t for t in all_infos if t.name not in _BLOCKLISTED_TABLES]
            log.info(
                "SchemaRetriever ready: %d analytical tables (%d blocklisted)",
                len(self._infos), len(all_infos) - len(self._infos),
            )

            for ti in self._infos:
                self._texts[ti.name]  = self._describe(ti)
                self._fk_out[ti.name] = {fk.ref_table for fk in ti.foreign_keys}

            self._try_embed()
            self._ready = True

    def _describe(self, ti: TableInfo) -> str:
        """
        Rich embedding text combining:
          1. Human description + aliases from schema_descriptions.yaml
          2. Column names + types from schema_doc.md (structural signal)
          3. FK relationship targets
        """
        meta   = self._descriptions.get(ti.name, {})
        parts: list[str] = [ti.name.replace("_", " ")]

        if desc := meta.get("description", ""):
            parts.append(desc.strip())
        if aliases := meta.get("aliases", ""):
            parts.append(aliases.strip())

        for col in ti.columns:
            # "real value double precision" is more informative than "real_value"
            parts.append(f"{col.name.replace('_', ' ')} {col.data_type}")

        for fk in ti.foreign_keys:
            parts.append(f"related to {fk.ref_table.replace('_', ' ')}")

        return " ".join(parts)

    def _try_embed(self) -> None:
        model_name  = settings.OLLAMA_EMBEDDING_MODEL or settings.OLLAMA_MODEL
        source_hash = _source_hash(model_name)

        # Always initialise the embedding model (needed for query-time embed_query)
        try:
            from langchain_ollama import OllamaEmbeddings
            self._embed_model = OllamaEmbeddings(
                model=model_name, base_url=settings.OLLAMA_BASE_URL
            )
        except Exception as exc:
            log.warning("Could not initialise embedding model (%s) — keyword fallback active", exc)
            self._embed_model = None
            return

        # Try loading vectors from disk cache
        cached = _load_embed_cache(source_hash)
        if cached is not None:
            self._vecs = cached
            return

        # Cache miss — embed all tables and persist
        texts = [self._texts[ti.name] for ti in self._infos]
        names = [ti.name for ti in self._infos]
        try:
            vecs = self._embed_model.embed_documents(texts)
            for name, vec in zip(names, vecs):
                self._vecs[name] = np.array(vec, dtype=np.float32)
            log.info("Embedded %d tables with model '%s'", len(self._vecs), model_name)
            _save_embed_cache(self._vecs, source_hash)
        except Exception as exc:
            log.warning("Embedding documents failed (%s) — keyword fallback active", exc)
            self._embed_model = None

    # ── Retrieve ───────────────────────────────────────────────────────────

    def retrieve(self, question: str, top_k: int = _TOP_K_DEFAULT) -> RetrievedSchema:
        if not self._ready:
            self.build()

        names  = [ti.name for ti in self._infos]
        ti_map = {ti.name: ti for ti in self._infos}

        scores = (
            self._embedding_scores(question, names)
            if self._vecs
            else self._keyword_scores(question, names)
        )

        for name in names:
            scores[name] = scores.get(name, 0.0) + self._name_boost(question, name)

        hinted       = self._domain_hint_tables(question)
        known_hinted = hinted & {ti.name for ti in self._infos}
        for name in known_hinted:
            scores[name] = max(scores.get(name, 0.0), 0.9)

        ranked  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_set = {n for n, _ in ranked[:top_k]} | known_hinted

        # 1-hop FK expansion
        expanded: set[str] = set(top_set)
        for tname in top_set:
            expanded |= self._fk_out.get(tname, set())

        if len(expanded) > _MAX_EXPANDED:
            fk_only    = expanded - top_set
            ranked_fk  = sorted(fk_only, key=lambda n: scores.get(n, 0), reverse=True)
            expanded   = top_set | set(ranked_fk[: _MAX_EXPANDED - len(top_set)])

        ordered  = [n for n, _ in ranked if n in expanded]
        ordered += [n for n in expanded if n not in {x for x, _ in ranked}]
        selected = [ti_map[n] for n in ordered if n in ti_map]

        log.info(
            "Retrieved %d/%d tables for '%s': %s",
            len(selected), len(self._infos),
            question[:60],
            [t.name for t in selected],
        )

        schema_text = _format_schema(selected)
        return RetrievedSchema(
            tables=selected,
            schema_text=schema_text,
            allowed_tables={ti.name for ti in selected},
        )

    def _name_boost(self, question: str, table_name: str) -> float:
        q_lower = question.lower()
        parts   = table_name.lower().split("_")
        return sum(0.35 for p in parts if len(p) > 3 and p in q_lower)

    def _domain_hint_tables(self, question: str) -> set[str]:
        q_lower = question.lower()
        forced: set[str] = set()
        for keyword, tables in _DOMAIN_HINTS.items():
            if keyword in q_lower:
                forced.update(tables)
        return forced

    def _embedding_scores(self, question: str, names: list[str]) -> dict[str, float]:
        try:
            q_vec = np.array(self._embed_model.embed_query(question), dtype=np.float32)
            out: dict[str, float] = {}
            for name in names:
                if name not in self._vecs:
                    out[name] = 0.0
                    continue
                v = self._vecs[name]
                out[name] = float(
                    np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-8)
                )
            return out
        except Exception as exc:
            log.warning("Embedding query failed (%s) — keyword fallback", exc)
            return self._keyword_scores(question, names)

    def _keyword_scores(self, question: str, names: list[str]) -> dict[str, float]:
        q_words = set(question.lower().split())
        ti_map  = {ti.name: ti for ti in self._infos}
        out: dict[str, float] = {}
        for name in names:
            ti = ti_map.get(name)
            if not ti:
                out[name] = 0.0
                continue
            t_words: set[str] = set()
            for part in name.split("_"):
                t_words.add(part.lower())
            for col in ti.columns:
                for part in col.name.split("_"):
                    t_words.add(part.lower())
            out[name] = len(q_words & t_words) / (len(q_words) + 1)
        return out

    def invalidate(self) -> None:
        """Force a full rebuild on the next call (e.g. after adding new tables)."""
        with self._lock:
            self._ready = False
            self._vecs.clear()
            self._infos.clear()
            self._texts.clear()
            self._embed_model = None


# ── Schema text formatter ─────────────────────────────────────────────────────

def _format_schema(tables: list[TableInfo]) -> str:
    lines: list[str] = []
    for ti in tables:
        row_hint = f"  (~{ti.row_count_estimate:,} rows)" if ti.row_count_estimate else ""
        lines.append(f"Table: {ti.name}{row_hint}")
        for col in ti.columns:
            pk = " [PK]" if col.is_pk else ""
            nn = " NOT NULL" if not col.nullable else ""
            lines.append(f"  {col.name}: {col.data_type}{pk}{nn}")
        for fk in ti.foreign_keys:
            lines.append(f"  FK: {fk.column} -> {fk.ref_table}.{fk.ref_column}")
        lines.append("")
    return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────

_retriever: SchemaRetriever | None = None


def get_schema_retriever() -> SchemaRetriever:
    global _retriever
    if _retriever is None:
        _retriever = SchemaRetriever()
    return _retriever


def embed_text(text: str) -> list[float] | None:
    """Return a float embedding for `text` using the schema retriever's embed model.
    Returns None if the embedding model is unavailable (keyword-fallback mode)."""
    retriever = get_schema_retriever()
    if retriever._embed_model is None:
        return None
    try:
        return retriever._embed_model.embed_query(text)
    except Exception as exc:
        log.warning("embed_text failed (%s)", exc)
        return None
