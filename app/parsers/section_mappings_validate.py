"""
Validate section_mappings.yaml against a parsed TBG file.

Reports:
  • OK     — section caught N metrics
  • WARN   — codes that fall outside every range (unsectioned)
  • ERROR  — overlap between ranges, or start/end code missing from workbook

Usage:
    python -m app.parsers.section_mappings_validate <path/to/TBG.xlsx>
"""
from __future__ import annotations

import sys
from collections import defaultdict

from app.parsers.excel_parser import (
    _code_in_range,
    _load_section_overrides,
    _split_code,
    parse_tbg_file,
)


def validate(xlsx_path: str) -> int:
    data = parse_tbg_file(xlsx_path)
    overrides = _load_section_overrides()
    error_count = 0

    print(f"\nValidating section_mappings.yaml against:\n  {xlsx_path}\n")

    for sheet_key, entries in overrides.items():
        sheet_data = data["sheets"].get(sheet_key)
        if not sheet_data:
            print(f"[SKIP] sheet {sheet_key} not present in workbook")
            continue
        metrics = sheet_data.get("metrics", {})
        all_codes = [m["code"] for m in metrics.values() if m.get("code")]

        print(f"\n=== {sheet_key.upper()} ({len(all_codes)} coded metrics) ===")

        # 1. Per-range hit count
        section_to_codes: dict[str, list[str]] = defaultdict(list)
        unsectioned: list[str] = []
        for code in all_codes:
            assigned = False
            for entry in entries:
                rng = entry["range"]
                if _code_in_range(code, rng[0], rng[1]):
                    section_to_codes[entry["section"]].append(code)
                    assigned = True
                    break
            if not assigned:
                unsectioned.append(code)

        for entry in entries:
            sec = entry["section"]
            rng = entry["range"]
            count = len(section_to_codes.get(sec, []))
            tag = "OK   " if count > 0 else "WARN "
            print(f"  [{tag}] {rng[0]:>8} → {rng[1]:<8}  {count:>3} metrics  {sec}")
            if count == 0:
                error_count += 1

        # 2. Range overlap check
        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                if _ranges_overlap(a["range"], b["range"]):
                    print(f"  [ERROR] overlap: {a['range']} ({a['section']}) and {b['range']} ({b['section']})")
                    error_count += 1

        # 3. Start/end codes must exist
        coded_set = set(all_codes)
        for entry in entries:
            for endpoint in entry["range"]:
                if endpoint not in coded_set:
                    print(f"  [ERROR] endpoint {endpoint} not in workbook  ({entry['section']})")
                    error_count += 1

        # 4. Unsectioned metrics
        if unsectioned:
            print(f"  [WARN ] {len(unsectioned)} metrics fall outside every range:")
            for code in unsectioned[:10]:
                lbl = next((m["label"] for m in metrics.values() if m.get("code") == code), "")
                print(f"            {code:<8}  {lbl[:55]}")
            if len(unsectioned) > 10:
                print(f"            … and {len(unsectioned) - 10} more")

    print(f"\n{'=' * 60}")
    if error_count == 0:
        print("Validation passed — no errors.")
    else:
        print(f"Validation found {error_count} issue(s) — see WARN/ERROR lines above.")
    return error_count


def _ranges_overlap(a: list[str], b: list[str]) -> bool:
    """Both ranges must share a code prefix; return True if they overlap numerically."""
    sa = _split_code(a[0])
    ea = _split_code(a[1])
    sb = _split_code(b[0])
    eb = _split_code(b[1])
    if not (sa and ea and sb and eb):
        return False
    if sa[0] != sb[0]:
        return False
    return not (ea[1] < sb[1] or eb[1] < sa[1])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "schema_files/TBG Moov_Africa_Bénin DEC 2025 DF SANS LIEN.xlsx"
    sys.exit(0 if validate(path) == 0 else 1)
