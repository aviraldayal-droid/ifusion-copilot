"""
Snapshot database registry.

Snapshot databases are dated copies of the main `digiwise` DB, each
representing the state of the data at the time the snapshot was taken.
They share the same schema so any validated SQL runs unchanged on them.

Naming convention:
    digiwise_{day}_{month_abbr}      e.g. digiwise_23_feb
    digiwise_ca_{day}_{month_abbr}   e.g. digiwise_ca_30_apr  (second entity)

Routing: resolve_snapshot(question) returns a db name only when the question
explicitly requests a snapshot (keywords: "snapshot", "as of", "version",
"database [month]"). All other questions stay on the main DB.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import date

log = logging.getLogger("tbg.snapshots")

# ── Month mappings ────────────────────────────────────────────────────────────

_ABBR_TO_NUM: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Full month names (EN + FR) → month number
_NAME_TO_NUM: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}

_NUM_TO_NAME: dict[int, str] = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

# ── Snapshot registry ─────────────────────────────────────────────────────────

@dataclass
class SnapshotInfo:
    dbname:        str
    month:         int          # 1-12
    day:           int          # day in the snapshot name (used to pick "latest" per month)
    is_ca:         bool         # True for digiwise_ca_* variant
    max_data_date: date         # max date with real_value data (queried once)

    @property
    def label(self) -> str:
        variant = " (ca)" if self.is_ca else ""
        return f"{_NUM_TO_NAME[self.month]}{variant} — data through {self.max_data_date.strftime('%b %Y')}"


# Hard-coded from the discovery query run on 2025-05-14.
# Re-run discover_snapshots() after new snapshots are added.
_ALL_SNAPSHOTS: list[SnapshotInfo] = [
    # ── non-ca ──────────────────────────────────────────────────────────────
    SnapshotInfo("digiwise_11_feb",  month=2,  day=11, is_ca=False, max_data_date=date(2026, 1, 1)),
    SnapshotInfo("digiwise_19_feb",  month=2,  day=19, is_ca=False, max_data_date=date(2026, 1, 1)),
    SnapshotInfo("digiwise_23_feb",  month=2,  day=23, is_ca=False, max_data_date=date(2026, 1, 1)),
    SnapshotInfo("digiwise_5_mar",   month=3,  day=5,  is_ca=False, max_data_date=date(2026, 1, 1)),
    SnapshotInfo("digiwise_6_mar",   month=3,  day=6,  is_ca=False, max_data_date=date(2026, 1, 1)),
    SnapshotInfo("digiwise_9_mar",   month=3,  day=9,  is_ca=False, max_data_date=date(2026, 1, 1)),
    SnapshotInfo("digiwise_11_mar",  month=3,  day=11, is_ca=False, max_data_date=date(2026, 1, 1)),
    SnapshotInfo("digiwise_12_mar",  month=3,  day=12, is_ca=False, max_data_date=date(2026, 1, 1)),
    SnapshotInfo("digiwise_7_apr",   month=4,  day=7,  is_ca=False, max_data_date=date(2026, 2, 1)),
    SnapshotInfo("digiwise_20_apr",  month=4,  day=20, is_ca=False, max_data_date=date(2026, 3, 1)),
    SnapshotInfo("digiwise_5_may",   month=5,  day=5,  is_ca=False, max_data_date=date(2026, 3, 1)),
    SnapshotInfo("digiwise_16_oct",  month=10, day=16, is_ca=False, max_data_date=date(2025, 9, 1)),
    # ── ca variant ──────────────────────────────────────────────────────────
    SnapshotInfo("digiwise_ca_26_dec", month=12, day=26, is_ca=True, max_data_date=date(2026, 2, 1)),
    SnapshotInfo("digiwise_ca_30_apr", month=4,  day=30, is_ca=True, max_data_date=date(2026, 3, 1)),
]

# Best non-ca snapshot per month: highest day number = most complete for that month
_BEST_BY_MONTH: dict[int, SnapshotInfo] = {}
for _s in _ALL_SNAPSHOTS:
    if _s.is_ca:
        continue
    if _s.month not in _BEST_BY_MONTH or _s.day > _BEST_BY_MONTH[_s.month].day:
        _BEST_BY_MONTH[_s.month] = _s

# name → SnapshotInfo lookup
SNAPSHOTS: dict[str, SnapshotInfo] = {s.dbname: s for s in _ALL_SNAPSHOTS}


# ── Snapshot resolver ─────────────────────────────────────────────────────────

# Explicit snapshot-intent keywords
_SNAPSHOT_TRIGGER = re.compile(
    r"\b(snapshot|as[\s\-]of|version|historique|base\s+de|ancienne?\s+base|old\s+db|archived?)\b",
    re.IGNORECASE,
)

# Month name pattern for extraction
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[eé]cembre"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)


def resolve_snapshot(question: str) -> SnapshotInfo | None:
    """
    Return the best matching snapshot if the question explicitly requests one.
    Returns None for all normal questions (stays on main DB).
    """
    q = question.lower()

    # Only trigger on explicit snapshot intent
    if not _SNAPSHOT_TRIGGER.search(q):
        return None

    # Extract month references
    months_found: list[int] = []
    for m in _MONTH_RE.finditer(q):
        token = m.group(1).lower()
        num = _NAME_TO_NUM.get(token) or _ABBR_TO_NUM.get(token[:3])
        if num and num not in months_found:
            months_found.append(num)

    if not months_found:
        # No specific month — return the most recent non-ca snapshot
        if _BEST_BY_MONTH:
            latest = max(_BEST_BY_MONTH.values(), key=lambda s: s.max_data_date)
            log.info("Snapshot resolved (no month): %s", latest.dbname)
            return latest
        return None

    # Return the best snapshot for the first mentioned month
    month = months_found[0]
    snap = _BEST_BY_MONTH.get(month)
    if snap:
        log.info("Snapshot resolved month=%d → %s", month, snap.dbname)
        return snap

    log.info("No snapshot found for month=%d", month)
    return None


def list_snapshots() -> str:
    """Human-readable summary of all available snapshots (for /db/snapshots endpoint)."""
    lines = ["Available database snapshots:\n"]
    seen_months: set[int] = set()
    for snap in sorted(_BEST_BY_MONTH.values(), key=lambda s: s.month):
        lines.append(f"  {snap.label}")
        lines.append(f"    Database: {snap.dbname}")
        seen_months.add(snap.month)
    ca_snaps = [s for s in _ALL_SNAPSHOTS if s.is_ca]
    if ca_snaps:
        lines.append("\n  CA variant:")
        for snap in ca_snaps:
            lines.append(f"  {snap.label}")
            lines.append(f"    Database: {snap.dbname}")
    return "\n".join(lines)


def discover_snapshots() -> list[dict]:
    """
    Live discovery: query the server for all digiwise_* databases and their
    max real_value date. Use this to refresh _ALL_SNAPSHOTS after new snapshots
    are added.  Not called at runtime — run manually when needed.
    """
    import re as _re
    import psycopg2
    import psycopg2.extras
    from app.config.settings import settings

    _PAT = _re.compile(r"digiwise(?:_(ca))?_(\d+)_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$", _re.I)

    base_url = settings.DATABASE_URL.replace(f"/{settings.DB_NAME}", "/postgres")
    conn = psycopg2.connect(dsn=base_url, sslmode="disable")
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT datname FROM pg_database WHERE datistemplate=false AND datname LIKE 'digiwise_%' ORDER BY datname")
    db_names = [r["datname"] for r in cur.fetchall()]
    cur.close(); conn.close()

    result = []
    for dbname in db_names:
        m = _PAT.match(dbname)
        if not m:
            continue
        is_ca = bool(m.group(1))
        day   = int(m.group(2))
        month = _ABBR_TO_NUM[m.group(3).lower()]
        try:
            c2  = psycopg2.connect(host=settings.DB_HOST, port=int(settings.DB_PORT),
                                   user=settings.DB_USER, password=settings.DB_PASSWORD,
                                   dbname=dbname, sslmode="disable", connect_timeout=5)
            cur2 = c2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur2.execute("SELECT MAX(date) AS mx FROM financial_metrics_data WHERE real_value IS NOT NULL")
            mx = cur2.fetchone()["mx"]
            cur2.close(); c2.close()
            result.append({"dbname": dbname, "month": month, "day": day, "is_ca": is_ca, "max_data_date": mx})
            print(f"SnapshotInfo({dbname!r}, month={month}, day={day}, is_ca={is_ca}, max_data_date=date{mx.timetuple()[:3]}),")
        except Exception as exc:
            print(f"  skip {dbname}: {exc}")
    return result
