from typing import Optional

import os

import streamlit as st
import yaml
import pandas as pd

from ir_scheduler import (
    CONSTRAINT_SPECS,
    ScheduleError,
    expand_residents,
    load_schedule_input_from_data,
    result_to_csv,
    solve_schedule,
)


IR_TRACKS = ["IR1", "IR2", "IR3", "IR4", "IR5"]
DR_TRACKS = ["DR1", "DR2", "DR3"]
CLASS_TRACKS = ["IR1", "IR2", "IR3", "IR4", "IR5", "DR1", "DR2", "DR3"]
ROTATION_COLUMNS = ["MH-IR", "MH-CT/US", "48X-IR", "48X-CT/US", "KIR"]
DISPLAY_COLUMNS = ROTATION_COLUMNS + ["Total Blocks"]

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
            "class_year_requirements": {
                track: dict(DEFAULT_CLASS_REQUIREMENTS[track]) for track in CLASS_TRACKS
            },
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
    gui.setdefault(
        "class_year_requirements",
        {track: dict(DEFAULT_CLASS_REQUIREMENTS[track]) for track in CLASS_TRACKS},
    )

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


def _num_blocks(cfg: dict) -> int:
    blocks = cfg.get("blocks")
    if isinstance(blocks, int):
        return blocks
    if isinstance(blocks, list):
        return len(blocks)
    num_blocks = cfg.get("num_blocks")
    if isinstance(num_blocks, int):
        return num_blocks
    return 0


def _track_counts(cfg: dict) -> dict:
    counts = {track: 2 for track in IR_TRACKS}
    dr_counts = cfg["gui"]["residents"]["DR_counts"]
    for track in DR_TRACKS:
        counts[track] = int(dr_counts.get(track, 0))
    return counts


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
    /* Make the drag/drop area taller so it visually aligns with the Save column. */
    section[data-testid="stFileUploaderDropzone"] {
        min-height: clamp(7.5rem, 12vh, 10.5rem);
        display: flex;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("IR/DR Scheduler Configurator")

_ensure_cfg_state()

cfg = st.session_state["cfg"]

if not st.session_state.get("infer_ok", True):
    st.warning("Could not infer IR1-IR5 names from residents; using defaults.")

st.divider()

resident_error = None
tabs = st.tabs(
    [
        "Residents",
        "Class/Year Assignments",
        "Constraints",
        "Prioritization",
        "Solve",
        "Save/Load Configuration",
    ]
)

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
    st.subheader("Class/Year Assignments")
    num_blocks = _num_blocks(cfg)
    req = cfg["gui"]["class_year_requirements"]
    prev_req = {track: dict(req.get(track, {})) for track in CLASS_TRACKS}
    rows = []
    for track in CLASS_TRACKS:
        row = {"Track": track}
        for rot in ROTATION_COLUMNS:
            row[rot] = int(req.get(track, {}).get(rot, 0))
        row["Total Blocks"] = sum(row[rot] for rot in ROTATION_COLUMNS)
        rows.append(row)

    edited = st.data_editor(
        rows,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Track": st.column_config.TextColumn(disabled=True),
            "MH-IR": st.column_config.NumberColumn(min_value=0, step=1),
            "MH-CT/US": st.column_config.NumberColumn(min_value=0, step=1),
            "48X-IR": st.column_config.NumberColumn(min_value=0, step=1),
            "48X-CT/US": st.column_config.NumberColumn(min_value=0, step=1),
            "KIR": st.column_config.NumberColumn(min_value=0, step=1),
            "Total Blocks": st.column_config.NumberColumn(disabled=True),
        },
        key="class_year_table",
    )

    updated_req = {}
    for row in edited:
        track = row["Track"]
        updated_req[track] = {rot: int(row.get(rot, 0)) for rot in ROTATION_COLUMNS}

    cfg["gui"]["class_year_requirements"] = updated_req
    if updated_req != prev_req:
        # Trigger a second rerun so computed cells (e.g., Total Blocks) update immediately.
        st.rerun()

    requirements = {}
    for track in CLASS_TRACKS:
        requirements[track] = dict(updated_req[track])
    cfg["requirements"] = requirements

    counts = _track_counts(cfg)
    avail = {rot: 0 for rot in ROTATION_COLUMNS}
    for track in CLASS_TRACKS:
        for rot in ROTATION_COLUMNS:
            avail[rot] += counts[track] * requirements[track][rot]

    # Read-only display table that includes a shaded Total Blocks column and a Total FTE row.
    display_rows = []
    for track in CLASS_TRACKS:
        r = {"Track": track}
        for rot in ROTATION_COLUMNS:
            r[rot] = requirements[track][rot]
        r["Total Blocks"] = sum(requirements[track][rot] for rot in ROTATION_COLUMNS)
        display_rows.append(r)
    total_fte_row = {"Track": "Total FTE"}
    for rot in ROTATION_COLUMNS:
        total_fte_row[rot] = avail[rot]
    total_fte_row["Total Blocks"] = None
    display_rows.append(total_fte_row)

    display_df = pd.DataFrame(display_rows, columns=["Track"] + DISPLAY_COLUMNS)
    for col in ROTATION_COLUMNS + ["Total Blocks"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").astype("Int64")

    def _style_totals(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        if "Total Blocks" in df.columns:
            styles["Total Blocks"] = "background-color: #f4f4f4; font-weight: 600;"
        total_idx = df.index[df["Track"] == "Total FTE"]
        if len(total_idx) == 1:
            styles.loc[total_idx[0], :] = (
                styles.loc[total_idx[0], :] + "background-color: #eef6ff; font-weight: 700;"
            )
        return styles

    st.dataframe(
        display_df.style.apply(_style_totals, axis=None).format(na_rep=""),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Rotation FTE availability vs. coverage**")
    rotation_rows = [
        {
            "Rotation": "48X-IR",
            "Available": avail["48X-IR"],
            "ReqMin": num_blocks,
            "ReqMax": num_blocks,
        },
        {
            "Rotation": "48X-CT/US",
            "Available": avail["48X-CT/US"],
            "ReqMin": num_blocks,
            "ReqMax": num_blocks,
        },
        {"Rotation": "MH-IR", "Available": avail["MH-IR"], "ReqMin": None, "ReqMax": None},
        {
            "Rotation": "MH-CT/US",
            "Available": avail["MH-CT/US"],
            "ReqMin": None,
            "ReqMax": num_blocks,
        },
        {
            "Rotation": "KIR",
            "Available": avail["KIR"],
            "ReqMin": None,
            "ReqMax": num_blocks * 2,
        },
    ]
    rotation_df = pd.DataFrame(rotation_rows)
    rotation_df[["ReqMin", "ReqMax"]] = rotation_df[["ReqMin", "ReqMax"]].astype("Int64")

    def _fmt_required(req_min, req_max) -> str:
        if pd.isna(req_min) and pd.isna(req_max):
            return "-"
        if pd.notna(req_min) and pd.notna(req_max):
            req_min_i = int(req_min)
            req_max_i = int(req_max)
            if req_min_i == req_max_i:
                return str(req_min_i)
            return f"{req_min_i} - {req_max_i}"
        if pd.isna(req_min) and pd.notna(req_max):
            return f"<= {int(req_max)}"
        if pd.notna(req_min) and pd.isna(req_max):
            return f">= {int(req_min)}"
        return "-"

    rotation_df["Required"] = rotation_df.apply(
        lambda row: _fmt_required(row["ReqMin"], row["ReqMax"]),
        axis=1,
    )

    def _make_row_style(source_df: pd.DataFrame):
        def _row_style(row):
            src = source_df.loc[row.name]
            req_min = src["ReqMin"]
            req_max = src["ReqMax"]
            if pd.isna(req_min) and pd.isna(req_max):
                return [""] * len(row)
            available = src["Available"]
            ok = True
            if pd.notna(req_min) and available < req_min:
                ok = False
            if pd.notna(req_max) and available > req_max:
                ok = False
            color = "#e9f7ef" if ok else "#fdecea"
            return [f"background-color: {color}"] * len(row)

        return _row_style

    rotation_display = rotation_df[["Rotation", "Available", "Required"]]
    st.dataframe(
        rotation_display.style.apply(_make_row_style(rotation_df), axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Location totals (per year)**")
    mh_total = avail["MH-IR"] + avail["MH-CT/US"]
    location_rows = [
        {
            "Location": "MH total",
            "Available": mh_total,
            "ReqMin": num_blocks * 3,
            "ReqMax": num_blocks * 4,
        },
        {
            "Location": "48X total",
            "Available": avail["48X-IR"] + avail["48X-CT/US"],
            "ReqMin": num_blocks * 2,
            "ReqMax": num_blocks * 2,
        },
    ]
    location_df = pd.DataFrame(location_rows)
    location_df[["ReqMin", "ReqMax"]] = location_df[["ReqMin", "ReqMax"]].astype("Int64")
    location_df["Required"] = location_df.apply(
        lambda row: _fmt_required(row["ReqMin"], row["ReqMax"]),
        axis=1,
    )
    location_display = location_df[["Location", "Available", "Required"]]
    st.dataframe(
        location_display.style.apply(_make_row_style(location_df), axis=1),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
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

with tabs[3]:
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

with tabs[4]:
    st.subheader("Solve")
    st.caption("Runs the solver using the current in-app configuration (no YAML download required).")

    cfg["num_solutions"] = int(
        st.number_input(
            "Number of solutions",
            min_value=1,
            max_value=50,
            value=int(cfg.get("num_solutions", 1)),
            step=1,
            key="num_solutions_input",
        )
    )

    col_run, col_dl = st.columns([1, 1])
    run = col_run.button("Solve", type="primary", use_container_width=True)

    if run:
        st.session_state.pop("solve_result", None)
        st.session_state.pop("solve_csv", None)
        try:
            schedule_input = load_schedule_input_from_data(cfg)
        except Exception as exc:
            st.error(f"Invalid configuration: {exc}")
        else:
            with st.spinner("Solving..."):
                result = solve_schedule(schedule_input)
            st.session_state["solve_result"] = result
            st.session_state["solve_csv"] = result_to_csv(result)

    result = st.session_state.get("solve_result")
    if result is None:
        st.info("Click Solve to run the scheduler.")
    else:
        if result.diagnostic:
            st.error("Model infeasible.")
            st.markdown("**Conflicting constraints**")
            st.dataframe(
                pd.DataFrame(result.diagnostic.conflicting_constraints),
                use_container_width=True,
                hide_index=True,
            )
            if result.diagnostic.suggestions:
                st.markdown("**Fast suggestions**")
                st.dataframe(
                    pd.DataFrame(result.diagnostic.suggestions),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.success(f"Found {len(result.solutions)} solution(s).")
            if not result.solutions:
                st.warning("No solutions returned (unexpected without infeasibility diagnostic).")
            else:
                idx = st.selectbox(
                    "Solution",
                    options=list(range(len(result.solutions))),
                    format_func=lambda i: f"Solution {i}",
                    key="solution_select",
                )
                sol = result.solutions[int(idx)]
                st.markdown("**Objective**")
                st.json(sol.objective)

                st.markdown("**Assignments by Track (blocks as columns)**")
                track_order = ["DR1", "DR2", "DR3", "IR1", "IR2", "IR3", "IR4", "IR5"]
                blocks = list(next(iter(sol.assignments.values())).keys()) if sol.assignments else []

                # Map resident -> track from the current config (cfg["residents"] is generated by the app).
                resident_to_track = {}
                for entry in cfg.get("residents", []):
                    if isinstance(entry, dict) and entry.get("id") and entry.get("track"):
                        resident_to_track[str(entry["id"])] = str(entry["track"])

                table_rows = []
                for track in track_order:
                    row = {"Track": track}
                    for block in blocks:
                        names = []
                        for resident, r_blocks in sol.assignments.items():
                            if resident_to_track.get(resident) != track:
                                continue
                            if r_blocks.get(block):  # any non-empty assignment in this block
                                names.append(resident)
                        row[block] = ", ".join(sorted(names))
                    table_rows.append(row)

                grid_df = pd.DataFrame(table_rows, columns=["Track"] + blocks)
                st.dataframe(grid_df, use_container_width=True, hide_index=True)

        csv_text = st.session_state.get("solve_csv", "")
        if csv_text:
            col_dl.download_button(
                "Download CSV",
                data=csv_text,
                file_name="schedule-output.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tabs[5]:
    st.subheader("Save/Load Configuration")

    def _reset_widget_state() -> None:
        prefixes = (
            "ir_",
            "dr_",
            "mode_",
            "prio_",
            "class_year_table",
            "num_solutions_input",
            "solution_select",
        )
        for key in list(st.session_state.keys()):
            if key.startswith(prefixes):
                del st.session_state[key]

    load_col, save_col = st.columns(2, gap="large")

    with load_col:
        st.markdown("### Load")
        st.caption("Drag & drop a `.yml`/`.yaml` to preview, then apply it to the current session.")
        uploaded = st.file_uploader(
            "Drop YAML here",
            type=["yml", "yaml"],
            label_visibility="collapsed",
            key="config_uploader",
        )

        if not uploaded:
            # If the user clears the uploader (clicks the X), also clear any pending preview state.
            st.session_state.pop("pending_cfg", None)
            st.session_state.pop("pending_infer_ok", None)
        else:
            try:
                loaded = yaml.safe_load(uploaded) or {}
            except yaml.YAMLError as exc:
                st.error(f"Failed to parse YAML: {exc}")
            else:
                try:
                    cfg_loaded, ok = _normalize_config(loaded)
                except Exception as exc:
                    st.error(f"Invalid configuration: {exc}")
                else:
                    st.session_state["pending_cfg"] = cfg_loaded
                    st.session_state["pending_infer_ok"] = ok

        pending = st.session_state.get("pending_cfg")
        if pending and uploaded:
            st.success("Loaded file parsed successfully. Review and apply when ready.")
            btn_apply, btn_discard = st.columns([1, 1])
            if btn_apply.button("Apply loaded configuration", type="primary", use_container_width=True):
                st.session_state["cfg"] = st.session_state.pop("pending_cfg")
                st.session_state["infer_ok"] = st.session_state.pop("pending_infer_ok", True)
                st.session_state["config_uploader"] = None
                _reset_widget_state()
                st.rerun()
            if btn_discard.button("Discard", use_container_width=True):
                st.session_state.pop("pending_cfg", None)
                st.session_state.pop("pending_infer_ok", None)
                st.session_state["config_uploader"] = None
                st.rerun()

    with save_col:
        st.markdown("### Save")
        st.caption("Download the current in-app configuration as a YAML file.")
        filename = st.text_input(
            "Filename",
            value="ir-scheduler.yml",
            key="config_filename",
        )
        yaml_text = yaml.safe_dump(cfg, sort_keys=False)
        saved = st.download_button(
            "Download configuration",
            data=yaml_text,
            file_name=filename or "ir-scheduler.yml",
            mime="text/yaml",
            use_container_width=True,
        )
        if saved:
            st.info(f"Downloaded. If you want the CLI default to pick it up, move it into:\n{os.getcwd()}")

    with st.expander("Current YAML", expanded=False):
        st.code(yaml.safe_dump(st.session_state["cfg"], sort_keys=False), language="yaml")
        if st.session_state.get("pending_cfg"):
            st.markdown("---")
            st.caption("Pending YAML (not applied yet)")
            st.code(yaml.safe_dump(st.session_state["pending_cfg"], sort_keys=False), language="yaml")
