from typing import Optional

import os

import streamlit as st
import yaml

from ir_scheduler import CONSTRAINT_SPECS, ScheduleError, expand_residents


IR_TRACKS = ["IR1", "IR2", "IR3", "IR4", "IR5"]
DR_TRACKS = ["DR1", "DR2", "DR3"]

DEFAULT_IR_NAMES = {
    "IR1": ["Gaburak", "Miller"],
    "IR2": ["Qi", "Verst"],
    "IR3": ["Madsen", "Mahmoud"],
    "IR4": ["Javan", "Virk"],
    "IR5": ["Brock", "Katz"],
}

DEFAULT_DR_COUNTS = {"DR1": 8, "DR2": 7, "DR3": 8}


def _default_gui_residents() -> dict:
    return {
        "IR": {track: list(DEFAULT_IR_NAMES[track]) for track in IR_TRACKS},
        "DR_counts": dict(DEFAULT_DR_COUNTS),
    }


def _default_config() -> dict:
    return {
        "blocks": 13,
        "blocked": [],
        "weights": {"consec": 100, "first_timer": 30, "adj": 1},
        "num_solutions": 1,
        "gui": {
            "residents": _default_gui_residents(),
            "constraints": {"modes": {}, "soft_priority": []},
        },
    }


def _infer_gui_residents(residents: list) -> tuple[dict, bool]:
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


def _normalize_config(cfg: dict) -> tuple[dict, bool]:
    if not isinstance(cfg, dict):
        cfg = {}

    if "blocks" not in cfg and "num_blocks" not in cfg:
        cfg["blocks"] = 13
    cfg.setdefault("blocked", [])
    cfg.setdefault("weights", {"consec": 100, "first_timer": 30, "adj": 1})
    cfg.setdefault("num_solutions", 1)

    gui = cfg.setdefault("gui", {})
    gui_constraints = gui.setdefault("constraints", {})
    gui_constraints.setdefault("modes", {})
    gui_constraints.setdefault("soft_priority", [])

    if "residents" not in gui:
        gui_residents, ok = _infer_gui_residents(cfg.get("residents", []))
        gui["residents"] = gui_residents
        return cfg, ok

    return cfg, True


def _ensure_cfg_state():
    if "cfg" not in st.session_state:
        cfg, ok = _normalize_config(_default_config())
        st.session_state["cfg"] = cfg
        st.session_state["infer_ok"] = ok


def _sync_residents(cfg: dict) -> Optional[str]:
    try:
        cfg["residents"] = expand_residents(cfg["gui"]["residents"])
    except ScheduleError as exc:
        return str(exc)
    return None


def _spec_label(spec) -> str:
    label = f"{spec.label} [{spec.id}]"
    if not spec.softenable:
        label += " (not softenable)"
    return label


st.set_page_config(page_title="IR/DR Scheduler Config", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stDownloadButton"] { display: flex; justify-content: flex-end; }
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stFileUploader"] button {
        height: 2.5rem;
        padding: 0 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("IR/DR Scheduler Configurator")

_ensure_cfg_state()

top_cols = st.columns([3, 1])
with top_cols[0]:
    uploaded = st.file_uploader("Load YAML", type=["yml", "yaml"])
download_slot = top_cols[1].empty()
notice_slot = st.empty()
if uploaded:
    try:
        loaded = yaml.safe_load(uploaded) or {}
    except yaml.YAMLError as exc:
        st.error(f"Failed to parse YAML: {exc}")
    else:
        cfg, ok = _normalize_config(loaded)
        st.session_state["cfg"] = cfg
        st.session_state["infer_ok"] = ok

cfg = st.session_state["cfg"]

if not st.session_state.get("infer_ok", True):
    st.warning("Could not infer IR1-IR5 names from residents; using defaults.")

st.divider()

resident_error = None
tabs = st.tabs(["Resident management", "Constraint enforcement", "Soft constraint prioritization"])

with tabs[0]:
    st.subheader("Residents")
    ir_inputs: dict = {}
    for track in IR_TRACKS:
        cols = st.columns(2)
        default_names = cfg["gui"]["residents"]["IR"].get(track, list(DEFAULT_IR_NAMES[track]))
        name1 = cols[0].text_input(f"{track} name 1", value=default_names[0], key=f"ir_{track}_1")
        name2 = cols[1].text_input(f"{track} name 2", value=default_names[1], key=f"ir_{track}_2")
        ir_inputs[track] = [name1.strip(), name2.strip()]

    st.subheader("DR counts")
    dr_counts: dict = {}
    cols = st.columns(3)
    for idx, track in enumerate(DR_TRACKS):
        default_count = int(cfg["gui"]["residents"]["DR_counts"].get(track, DEFAULT_DR_COUNTS[track]))
        dr_counts[track] = int(
            cols[idx].number_input(
                f"{track} count", min_value=0, max_value=50, value=default_count, step=1, key=f"dr_{track}"
            )
        )

    cfg["gui"]["residents"] = {"IR": ir_inputs, "DR_counts": dr_counts}
    resident_error = _sync_residents(cfg)
    if resident_error:
        st.error(resident_error)

with tabs[1]:
    st.subheader("Constraint modes")
    modes = cfg["gui"]["constraints"].get("modes", {})
    for spec in CONSTRAINT_SPECS:
        if spec.softenable:
            options = ["always", "if_able", "disabled"]
        else:
            options = ["always", "disabled"]
        default_mode = modes.get(spec.id, "if_able" if spec.softenable else "always")
        if default_mode not in options:
            default_mode = "if_able" if spec.softenable else "always"
        selection = st.radio(
            _spec_label(spec),
            options=options,
            horizontal=True,
            index=options.index(default_mode),
            key=f"mode_{spec.id}",
        )
        modes[spec.id] = selection
    cfg["gui"]["constraints"]["modes"] = modes

with tabs[2]:
    st.subheader("Soft constraint priority")
    modes = cfg["gui"]["constraints"].get("modes", {})
    if_able_ids = [
        spec.id
        for spec in CONSTRAINT_SPECS
        if modes.get(spec.id, "if_able" if spec.softenable else "always") == "if_able"
    ]
    if not if_able_ids:
        st.info("No constraints currently set to if_able.")
    else:
        priority = cfg["gui"]["constraints"].get("soft_priority", [])
        priority = [cid for cid in priority if cid in if_able_ids]
        for cid in if_able_ids:
            if cid not in priority:
                priority.append(cid)
        cfg["gui"]["constraints"]["soft_priority"] = priority

        for idx, cid in enumerate(priority):
            spec = next(spec for spec in CONSTRAINT_SPECS if spec.id == cid)
            cols = st.columns([6, 1, 1])
            cols[0].write(f"{idx + 1}. {_spec_label(spec)}")
            if cols[1].button("Up", key=f"prio_up_{cid}") and idx > 0:
                priority[idx - 1], priority[idx] = priority[idx], priority[idx - 1]
                cfg["gui"]["constraints"]["soft_priority"] = priority
                st.rerun()
            if cols[2].button("Down", key=f"prio_dn_{cid}") and idx < len(priority) - 1:
                priority[idx + 1], priority[idx] = priority[idx], priority[idx + 1]
                cfg["gui"]["constraints"]["soft_priority"] = priority
                st.rerun()

st.divider()

yaml_text = yaml.safe_dump(cfg, sort_keys=False)
saved = download_slot.download_button(
    "Download YAML",
    data=yaml_text,
    file_name="ir-scheduler.yml",
    mime="text/yaml",
)
if saved:
    notice_slot.info(
        f"Move the downloaded file 'ir-scheduler.yml' into:\n{os.getcwd()}",
        icon="📌",
    )

st.subheader("Current YAML")
st.code(yaml_text, language="yaml")
