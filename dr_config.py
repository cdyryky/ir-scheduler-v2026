from __future__ import annotations

from typing import Any, Optional, Tuple

DR_CONFIG_TYPE = "DR"
DR_SCHEMA_VERSION = 1
DR_YEAR_LABELS = ["Y1", "Y2", "Y3", "Y4"]
DR_RESIDENT_TRACKS = ["DR", "IR", "NM"]

DEFAULT_DR_YEAR_COUNTS = {
    "Y1": 10,
    "Y2": 10,
    "Y3": 9,
    "Y4": 8,
}

DEFAULT_DR_YEAR_NAMES = {
    "Y1": [
        "Bansal",
        "Barr",
        "Gadepally",
        "Jmaileh",
        "Patel",
        "Qi",
        "Raghupathy",
        "Veras",
        "Verst",
        "Vorobchevici",
    ],
    "Y2": [
        "Byju",
        "Choudhury",
        "Ekunseitan",
        "Knight",
        "Lan",
        "Madsen",
        "Mahmoud",
        "Mayo",
        "Rosen",
        "Seto",
    ],
    "Y3": [
        "Choi",
        "Escudero",
        "Govardhan",
        "Griffith",
        "Javan",
        "Kwan",
        "Lim",
        "Sachdev",
        "Virk",
    ],
    "Y4": [
        "Aeilts",
        "Brock",
        "Chen",
        "Dhingra",
        "Getz",
        "DiSanti",
        "Katz",
        "Lee",
    ],
}

DEFAULT_DR_TRACK_BY_NAME = {
    "Bansal": "DR",
    "Barr": "NM",
    "Gadepally": "DR",
    "Jmaileh": "DR",
    "Patel": "DR",
    "Qi": "IR",
    "Raghupathy": "DR",
    "Veras": "DR",
    "Verst": "IR",
    "Vorobchevici": "DR",
    "Byju": "DR",
    "Choudhury": "DR",
    "Ekunseitan": "DR",
    "Knight": "DR",
    "Lan": "DR",
    "Madsen": "IR",
    "Mahmoud": "IR",
    "Mayo": "DR",
    "Rosen": "DR",
    "Seto": "DR",
    "Choi": "DR",
    "Escudero": "DR",
    "Govardhan": "DR",
    "Griffith": "DR",
    "Javan": "IR",
    "Kwan": "DR",
    "Lim": "DR",
    "Sachdev": "DR",
    "Virk": "IR",
    "Aeilts": "DR",
    "Brock": "IR",
    "Chen": "DR",
    "Dhingra": "DR",
    "Getz": "DR",
    "DiSanti": "DR",
    "Katz": "IR",
    "Lee": "NM",
}

DEFAULT_DR_CALENDAR_START_DATE = "06/29/26"
DEFAULT_DR_BLOCKS = 13
DEFAULT_DR_MAX_VACATION_REQUESTS = 6
DEFAULT_DR_AWARDS_PER_RESIDENT = 4


def _to_non_negative_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        return int(default)
    return max(0, out)


def _normalize_blocks(value: Any) -> int | list[str]:
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, list):
        labels = [str(v).strip() for v in value if str(v).strip()]
        return labels if labels else DEFAULT_DR_BLOCKS
    return DEFAULT_DR_BLOCKS


def _block_labels(blocks: int | list[str]) -> list[str]:
    if isinstance(blocks, int):
        return [f"B{i}" for i in range(blocks)]
    return [str(b) for b in blocks]


def _week_labels(blocks: int | list[str]) -> list[str]:
    labels: list[str] = []
    for block in _block_labels(blocks):
        for week_idx in range(4):
            labels.append(f"{block}.{week_idx}")
    return labels


def _default_year_rows(year_label: str, count: Optional[int] = None) -> list[dict]:
    names = list(DEFAULT_DR_YEAR_NAMES.get(year_label, []))
    target = len(names) if count is None else int(count)
    rows = [
        {"name": name, "track": str(DEFAULT_DR_TRACK_BY_NAME.get(name, "DR"))}
        for name in names
    ]
    if target < len(rows):
        return rows[:target]

    used = {str(r["name"]).strip() for r in rows if str(r["name"]).strip()}
    idx = len(rows) + 1
    while len(rows) < target:
        candidate = f"{year_label}-{idx}"
        idx += 1
        if candidate in used:
            continue
        used.add(candidate)
        rows.append({"name": candidate, "track": "DR"})
    return rows


def _normalize_year_rows(year_label: str, rows: Any, target_count: int) -> list[dict]:
    out: list[dict] = []
    if isinstance(rows, list):
        for row in rows:
            if len(out) >= target_count:
                break
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "") or "").strip()
            track = str(row.get("track", "DR") or "DR").strip().upper()
            if track not in DR_RESIDENT_TRACKS:
                track = "DR"
            if not name:
                continue
            out.append({"name": name, "track": track})

    if len(out) < target_count:
        defaults = _default_year_rows(year_label, count=target_count)
        for default_row in defaults:
            if len(out) >= target_count:
                break
            if default_row["name"] in {r["name"] for r in out}:
                continue
            out.append(default_row)

    used = {str(r["name"]).strip() for r in out if str(r["name"]).strip()}
    idx = 1
    while len(out) < target_count:
        candidate = f"{year_label}-{idx}"
        idx += 1
        if candidate in used:
            continue
        used.add(candidate)
        out.append({"name": candidate, "track": "DR"})

    return out[:target_count]


def _normalize_residents(gui: dict) -> dict:
    residents = gui.get("residents")
    if not isinstance(residents, dict):
        residents = {}

    raw_year_counts = residents.get("year_counts")
    raw_years = residents.get("years") if isinstance(residents.get("years"), dict) else {}
    year_counts: dict[str, int] = {}
    for year in DR_YEAR_LABELS:
        if isinstance(raw_year_counts, dict) and year in raw_year_counts:
            count = _to_non_negative_int(raw_year_counts.get(year), DEFAULT_DR_YEAR_COUNTS[year])
        elif isinstance(raw_years.get(year), list):
            count = _to_non_negative_int(len(raw_years.get(year, [])), DEFAULT_DR_YEAR_COUNTS[year])
        else:
            count = DEFAULT_DR_YEAR_COUNTS[year]
        year_counts[year] = count

    years: dict[str, list[dict]] = {}
    for year in DR_YEAR_LABELS:
        years[year] = _normalize_year_rows(year, raw_years.get(year), year_counts[year])

    return {"year_counts": year_counts, "years": years}


def _normalize_vacation_requests(gui: dict, blocks: int | list[str], resident_names: list[str]) -> dict:
    requests = gui.get("requests")
    if not isinstance(requests, dict):
        requests = {}
    vacation = requests.get("vacation")
    if not isinstance(vacation, dict):
        vacation = {}

    max_requests = _to_non_negative_int(
        vacation.get("max_requests_per_resident"), DEFAULT_DR_MAX_VACATION_REQUESTS
    )
    by_resident_raw = vacation.get("by_resident")
    if not isinstance(by_resident_raw, dict):
        by_resident_raw = {}

    valid_weeks = set(_week_labels(blocks))
    by_resident_out: dict[str, list[dict]] = {}

    for name in resident_names:
        raw_entries = by_resident_raw.get(name, [])
        if not isinstance(raw_entries, list):
            continue

        seen_ranks: set[int] = set()
        seen_weeks: set[str] = set()
        normalized: list[dict] = []

        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            week = str(entry.get("week", "") or "").strip()
            rank = _to_non_negative_int(entry.get("rank"), 0)
            if not week or week not in valid_weeks:
                continue
            if rank <= 0:
                continue
            if week in seen_weeks or rank in seen_ranks:
                continue
            seen_weeks.add(week)
            seen_ranks.add(rank)
            normalized.append({"week": week, "rank": rank})

        normalized.sort(key=lambda item: item["rank"])
        if max_requests >= 0:
            normalized = normalized[:max_requests]
        if normalized:
            by_resident_out[name] = normalized

    return {
        "max_requests_per_resident": max_requests,
        "awards_per_resident": DEFAULT_DR_AWARDS_PER_RESIDENT,
        "by_resident": by_resident_out,
    }


def _resident_names_in_order(cfg: dict) -> list[str]:
    gui = cfg.get("gui") if isinstance(cfg.get("gui"), dict) else {}
    residents = gui.get("residents") if isinstance(gui.get("residents"), dict) else {}
    years = residents.get("years") if isinstance(residents.get("years"), dict) else {}

    out: list[str] = []
    for year in DR_YEAR_LABELS:
        rows = years.get(year)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "") or "").strip()
            if name:
                out.append(name)
    return out


def default_dr_config() -> dict:
    cfg = {
        "config_type": DR_CONFIG_TYPE,
        "schema_version": DR_SCHEMA_VERSION,
        "blocks": DEFAULT_DR_BLOCKS,
        "gui": {
            "calendar": {"start_date": DEFAULT_DR_CALENDAR_START_DATE},
            "residents": {
                "year_counts": dict(DEFAULT_DR_YEAR_COUNTS),
                "years": {
                    year: _default_year_rows(year, count=DEFAULT_DR_YEAR_COUNTS[year])
                    for year in DR_YEAR_LABELS
                },
            },
            "class_year_assignments": {},
            "constraints": {},
            "prioritization": {},
            "solve": {},
            "instructions": {},
            "requests": {
                "vacation": {
                    "max_requests_per_resident": DEFAULT_DR_MAX_VACATION_REQUESTS,
                    "awards_per_resident": DEFAULT_DR_AWARDS_PER_RESIDENT,
                    "by_resident": {},
                }
            },
        },
    }
    return cfg


def validate_dr_resident_names(cfg: dict) -> Optional[str]:
    names: list[str] = []
    gui = cfg.get("gui") if isinstance(cfg.get("gui"), dict) else {}
    residents = gui.get("residents") if isinstance(gui.get("residents"), dict) else {}
    years = residents.get("years") if isinstance(residents.get("years"), dict) else {}

    for year in DR_YEAR_LABELS:
        rows = years.get(year)
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                return f"{year} resident #{idx} is not a valid row."
            name = str(row.get("name", "") or "").strip()
            if not name:
                return f"{year} resident #{idx} has an empty name."
            names.append(name)

    seen: set[str] = set()
    dupes: list[str] = []
    for name in names:
        if name in seen and name not in dupes:
            dupes.append(name)
        seen.add(name)

    if dupes:
        shown = ", ".join(dupes[:5])
        suffix = "" if len(dupes) <= 5 else ", ..."
        return f"Resident names must be unique across all years. Duplicate(s): {shown}{suffix}"
    return None


def prepare_dr_config(cfg: Any) -> Tuple[dict, bool]:
    if not isinstance(cfg, dict):
        cfg = {}

    out = dict(cfg)
    out["config_type"] = DR_CONFIG_TYPE
    out["schema_version"] = DR_SCHEMA_VERSION
    out["blocks"] = _normalize_blocks(out.get("blocks"))

    gui = out.get("gui")
    if not isinstance(gui, dict):
        gui = {}
    out["gui"] = gui

    calendar = gui.get("calendar")
    if not isinstance(calendar, dict):
        calendar = {}
    start_date = str(calendar.get("start_date", "") or "").strip()
    if not start_date:
        start_date = DEFAULT_DR_CALENDAR_START_DATE
    gui["calendar"] = {"start_date": start_date}

    gui["residents"] = _normalize_residents(gui)
    gui.setdefault("class_year_assignments", {})
    gui.setdefault("constraints", {})
    gui.setdefault("prioritization", {})
    gui.setdefault("solve", {})
    gui.setdefault("instructions", {})

    resident_names = _resident_names_in_order(out)
    requests = gui.get("requests")
    if not isinstance(requests, dict):
        requests = {}
    requests["vacation"] = _normalize_vacation_requests(
        gui,
        out["blocks"],
        resident_names,
    )
    gui["requests"] = requests

    return out, True
