from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import yaml


IR_TRACKS = ["IR1", "IR2", "IR3", "IR4", "IR5"]
DR_TRACKS = ["DR1", "DR2", "DR3"]
CLASS_TRACKS = IR_TRACKS + DR_TRACKS

ROTATION_COLUMNS = ["MH-IR", "MH-CT/US", "48X-IR", "48X-CT/US", "KIR"]

DEFAULT_IR_NAMES = {
    "IR1": ["Gaburak", "Miller"],
    "IR2": ["Qi", "Verst"],
    "IR3": ["Madsen", "Mahmoud"],
    "IR4": ["Javan", "Virk"],
    "IR5": ["Brock", "Katz"],
}

DEFAULT_DR_COUNTS = {"DR1": 8, "DR2": 7, "DR3": 8}
DEFAULT_CLASS_REQUIREMENTS = {
    "IR1": {"MH-IR": 1, "MH-CT/US": 0, "48X-IR": 1, "48X-CT/US": 1, "KIR": 0},
    "IR2": {"MH-IR": 2, "MH-CT/US": 0, "48X-IR": 1, "48X-CT/US": 0, "KIR": 0},
    "IR3": {"MH-IR": 1, "MH-CT/US": 0, "48X-IR": 1, "48X-CT/US": 1, "KIR": 0},
    "IR4": {"MH-IR": 3, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0, "KIR": 3},
    "IR5": {"MH-IR": 8, "MH-CT/US": 0, "48X-IR": 2, "48X-CT/US": 0, "KIR": 3},
    "DR1": {"MH-IR": 1, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0, "KIR": 0},
    "DR2": {"MH-IR": 0, "MH-CT/US": 1, "48X-IR": 0, "48X-CT/US": 0, "KIR": 0},
    "DR3": {"MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 1, "KIR": 0},
}

CURRENT_SCHEMA_VERSION = 2

DEFAULT_CALENDAR_START_DATE = "06/29/26"


def default_gui_residents() -> dict:
    return {
        "IR": {track: list(DEFAULT_IR_NAMES[track]) for track in IR_TRACKS},
        "DR_counts": dict(DEFAULT_DR_COUNTS),
    }


def default_config() -> dict:
    default_path = Path(__file__).with_name("schedule-config-2026-01-31.yml")
    try:
        if default_path.exists():
            loaded = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                return loaded
    except Exception:
        pass

    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "blocks": 13,
        "blocked": [],
        "forced": {},
        "weights": {"consec": 100, "first_timer": 30, "adj": 1},
        "num_solutions": 1,
        "gui": {
            "residents": default_gui_residents(),
            "calendar": {"start_date": DEFAULT_CALENDAR_START_DATE},
            "class_year_requirements": {
                track: dict(DEFAULT_CLASS_REQUIREMENTS[track]) for track in CLASS_TRACKS
            },
            "constraints": {
                "modes": {},
                "soft_priority": [],
                "params": {
                    "coverage_48x_ir": {"op": "==", "target_units": 2},
                    "coverage_48x_ctus": {"op": "==", "target_units": 2},
                    "mh_total_minmax": {"min_fte": 3, "max_fte": 4},
                    "mh_ctus_cap": {"max_fte": 1},
                    "kir_cap": {"max_fte": 2},
                    "ir4_plus_mh_cap": {"ir_min_year": 4, "max_fte": 2},
                    "dr1_early_block": {"first_n_blocks": 4},
                    "ir3_late_block": {"after_block": 7, "rotations": ["MH-IR", "48X-IR"]},
                    "consec_full_mh": {"max_consecutive": 3},
                },
            },
        },
    }


def migrate_config(cfg: Any) -> dict:
    if not isinstance(cfg, dict):
        cfg = {}

    version_raw = cfg.get("schema_version", 0)
    try:
        version = int(version_raw)
    except (TypeError, ValueError):
        version = 0

    # If the file is from a newer schema than this code knows about, keep it as-is.
    if version > CURRENT_SCHEMA_VERSION:
        cfg["schema_version"] = version
        return cfg

    # v0 -> v1: introduce schema_version field. (All other normalization happens in normalize_config.)
    if version < 1:
        cfg["schema_version"] = 1
        version = 1

    # v1 -> v2: Calendar start date now lives under gui.calendar.start_date.
    if version < 2:
        cfg["schema_version"] = 2
        version = 2

    cfg["schema_version"] = CURRENT_SCHEMA_VERSION
    return cfg


def _infer_gui_residents(residents: Any) -> Tuple[dict, bool]:
    ir_map = {track: [] for track in IR_TRACKS}
    dr_counts = {track: 0 for track in DR_TRACKS}
    for entry in residents or []:
        track = entry.get("track") if isinstance(entry, dict) else None
        resident_id = entry.get("id") if isinstance(entry, dict) else None
        if track in ir_map and resident_id:
            ir_map[track].append(str(resident_id))
        elif track in dr_counts:
            dr_counts[track] += 1

    ok = all(len(ir_map[track]) == 2 for track in IR_TRACKS)
    gui_ir = {track: ir_map[track] if ok else list(DEFAULT_IR_NAMES[track]) for track in IR_TRACKS}
    if not ok:
        dr_counts = dict(DEFAULT_DR_COUNTS)
    return {"IR": gui_ir, "DR_counts": dr_counts}, ok


def _normalize_class_year_requirements(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}

    out: dict = {}
    for track in CLASS_TRACKS:
        row = value.get(track)
        if not isinstance(row, dict):
            row = {}
        normalized_row = {}
        defaults = DEFAULT_CLASS_REQUIREMENTS[track]
        for rot in ROTATION_COLUMNS:
            raw = row.get(rot, defaults.get(rot, 0))
            if track in IR_TRACKS:
                try:
                    if rot == "KIR":
                        normalized_value = max(0.0, float(int(round(float(raw)))))
                    else:
                        normalized_value = max(0.0, round(float(raw) * 2) / 2)
                    normalized_row[rot] = (
                        int(normalized_value) if float(normalized_value).is_integer() else float(normalized_value)
                    )
                except (TypeError, ValueError):
                    normalized_row[rot] = float(defaults.get(rot, 0))
            else:
                try:
                    normalized_row[rot] = int(raw)
                except (TypeError, ValueError):
                    normalized_row[rot] = int(defaults.get(rot, 0))
        out[track] = normalized_row
    return out


def _requirements_from_class_year_requirements(class_year_requirements: dict) -> dict:
    requirements: dict = {}
    for track in CLASS_TRACKS:
        requirements[track] = dict(class_year_requirements.get(track, {}))
    return requirements


def normalize_config(cfg: Any) -> Tuple[dict, bool]:
    cfg = migrate_config(cfg)

    input_gui = cfg.get("gui") if isinstance(cfg.get("gui"), dict) else None
    input_gui_had_class_year_requirements = (
        isinstance(input_gui, dict) and "class_year_requirements" in input_gui
    )

    if "blocks" not in cfg and "num_blocks" not in cfg:
        cfg["blocks"] = 13
    cfg.setdefault("blocked", [])
    cfg.setdefault("forced", {})
    cfg.setdefault("weights", {"consec": 100, "first_timer": 30, "adj": 1})
    cfg.setdefault("num_solutions", 1)

    gui = cfg.setdefault("gui", {})
    if not isinstance(gui, dict):
        gui = {}
        cfg["gui"] = gui

    gui_constraints = gui.setdefault("constraints", {})
    if not isinstance(gui_constraints, dict):
        gui_constraints = {}
        gui["constraints"] = gui_constraints
    gui_constraints.setdefault("modes", {})
    gui_constraints.setdefault("soft_priority", [])
    gui_constraints.setdefault("params", {})
    if not isinstance(gui_constraints.get("params"), dict):
        gui_constraints["params"] = {}

    gui_calendar = gui.setdefault("calendar", {})
    if not isinstance(gui_calendar, dict):
        gui_calendar = {}
        gui["calendar"] = gui_calendar
    start_date = gui_calendar.get("start_date")
    gui_calendar["start_date"] = str(start_date) if isinstance(start_date, str) and start_date.strip() else DEFAULT_CALENDAR_START_DATE

    # Ensure GUI requirements exist (used for the editable table), even if YAML only has solver-level requirements.
    if "class_year_requirements" not in gui:
        if isinstance(cfg.get("requirements"), dict):
            gui["class_year_requirements"] = _normalize_class_year_requirements(cfg["requirements"])
        else:
            gui["class_year_requirements"] = _normalize_class_year_requirements(None)
    else:
        gui["class_year_requirements"] = _normalize_class_year_requirements(gui["class_year_requirements"])

    # Ensure solver-level requirements exist if the GUI requirements exist (typical Streamlit path).
    if (
        "requirements" not in cfg
        and input_gui_had_class_year_requirements
        and isinstance(gui.get("class_year_requirements"), dict)
    ):
        cfg["requirements"] = _requirements_from_class_year_requirements(gui["class_year_requirements"])

    if "residents" not in gui:
        gui_residents, ok = _infer_gui_residents(cfg.get("residents", []))
        gui["residents"] = gui_residents
        return cfg, ok

    return cfg, True


def prepare_config(cfg: Any) -> Tuple[dict, bool]:
    return normalize_config(cfg)
