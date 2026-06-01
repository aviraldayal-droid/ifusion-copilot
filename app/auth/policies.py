"""
Role-based access control for the iFusion Copilot.

Loads role definitions from policies.yaml and provides:
  • get_policy(role)        — look up role definition
  • check_question(...)     — pre-query keyword scan; returns (allowed, reason)
  • check_metric(...)       — per-metric check used when filtering metric_hints
  • policy_refusal_text(...) — user-facing refusal message
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("tbg.policies")

_POLICIES_PATH = Path(__file__).parent / "policies.yaml"
_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(_POLICIES_PATH) as f:
            _cache = yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("policies.yaml not found at %s — defaulting to permissive admin role", _POLICIES_PATH)
        _cache = {"roles": {}}
    return _cache


def get_policy(role: str | None) -> dict[str, Any]:
    """Return the policy entry for a role. Falls back to 'viewer' for unknown roles."""
    roles = _load().get("roles", {})
    if role and role in roles:
        return roles[role]
    # Unknown / missing role → fall back to the most restrictive
    return roles.get("viewer", {
        "label": "Viewer", "can_manage_users": False,
        "allowed_sheets": [], "blocked_sheets": [],
        "blocked_sections": [], "blocked_keywords": [],
    })


def check_question(role: str | None, question: str) -> tuple[bool, str]:
    """
    Quick keyword-level check against the user's question.
    Returns (allowed, blocking_keyword_or_empty).
    Designed to catch the obvious cases — full enforcement happens at metric_hint
    and tool-call layers.
    """
    pol = get_policy(role)
    blocked = pol.get("blocked_keywords") or []
    q = (question or "").lower()
    for kw in blocked:
        if kw.lower() in q:
            return False, kw
    return True, ""


def check_metric(role: str | None, sheet_key: str, section: str | None) -> tuple[bool, str]:
    """
    Per-metric scoping. Returns (allowed, reason_or_empty).
    Used to filter the ambiguity map, field-map endpoint, and metric_hints.
    """
    pol = get_policy(role)
    allowed_sheets = pol.get("allowed_sheets") or []
    blocked_sheets = pol.get("blocked_sheets") or []
    blocked_sections = [s.lower() for s in (pol.get("blocked_sections") or [])]

    if sheet_key in blocked_sheets:
        return False, f"sheet '{sheet_key}'"
    if "*" not in allowed_sheets and allowed_sheets and sheet_key not in allowed_sheets:
        return False, f"sheet '{sheet_key}' not in allow-list"
    sec_lower = (section or "").lower()
    for b in blocked_sections:
        if b in sec_lower:
            return False, f"section '{section}'"
    return True, ""


def filter_field_map(role: str | None, field_map: dict) -> dict:
    """Strip blocked sheets/sections from a parsed field-map before sending to the UI."""
    pol = get_policy(role)
    allowed = pol.get("allowed_sheets") or []
    blocked_sheets = set(pol.get("blocked_sheets") or [])
    blocked_secs_l = [s.lower() for s in (pol.get("blocked_sections") or [])]

    out_sheets = []
    for sheet in field_map.get("sheets", []):
        if sheet["sheet_key"] in blocked_sheets:
            continue
        if "*" not in allowed and allowed and sheet["sheet_key"] not in allowed:
            continue
        new_sections = []
        for sec in sheet.get("sections", []):
            sec_name = (sec.get("section") or "").lower()
            if any(b in sec_name for b in blocked_secs_l):
                continue
            new_sections.append(sec)
        if new_sections:
            sheet = {**sheet, "sections": new_sections}
            out_sheets.append(sheet)
    return {**field_map, "sheets": out_sheets}


def policy_refusal_text(blocked_by: str, role: str | None, language: str = "en") -> str:
    """User-facing refusal message."""
    label = get_policy(role).get("label", role or "viewer")
    if language == "fr":
        return (
            f"Cette demande touche des données restreintes pour votre rôle ({label}). "
            f"Élément bloqué : « {blocked_by} ». Si vous avez besoin d'y accéder, "
            f"contactez l'administrateur."
        )
    return (
        f"This request touches data restricted for your role ({label}). "
        f"Blocked term: \"{blocked_by}\". If you need access, please contact the administrator."
    )


def policy_prompt_block(role: str | None) -> str:
    """Build the system-prompt block that lists the role's restrictions for the LLM."""
    pol = get_policy(role)
    label = pol.get("label", role or "viewer")
    if pol.get("can_manage_users") and not (pol.get("blocked_sections") or pol.get("blocked_keywords")):
        return f"User role: {label}. No data restrictions.\n"
    lines = [f"User role: {label}."]
    if pol.get("blocked_sections"):
        lines.append("BLOCKED sub-sections (refuse any question that targets these): " +
                     ", ".join(pol["blocked_sections"]))
    if pol.get("blocked_sheets"):
        lines.append("BLOCKED sheets (refuse any question about these): " +
                     ", ".join(pol["blocked_sheets"]))
    if pol.get("blocked_keywords"):
        lines.append("BLOCKED keywords (refuse if the question contains any): " +
                     ", ".join(pol["blocked_keywords"]))
    lines.append("If the user asks about a blocked item, respond with a brief refusal mentioning their role and the blocked term — do NOT call any tool, do NOT answer the underlying question.")
    return "\n".join(lines) + "\n"
