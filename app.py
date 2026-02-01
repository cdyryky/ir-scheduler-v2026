from typing import Optional

import os
from datetime import date
import math

import streamlit as st
import yaml
import pandas as pd
from matplotlib import cm, colors

from ir_config import (
    CLASS_TRACKS,
    DEFAULT_CLASS_REQUIREMENTS,
    DEFAULT_DR_COUNTS,
    DEFAULT_IR_NAMES,
    DR_TRACKS,
    IR_TRACKS,
    ROTATION_COLUMNS,
    default_config,
    prepare_config,
)
from ir_scheduler import (
    CONSTRAINT_SPECS,
    ScheduleError,
    expand_residents,
    load_schedule_input_from_data,
    result_to_csv,
    solve_schedule,
)

APP_TITLE = "IR Schedulator 5000"


DISPLAY_COLUMNS = ROTATION_COLUMNS + ["Total Blocks"]

def _ensure_cfg_state():
    if "cfg" not in st.session_state:
        cfg, ok = prepare_config(default_config())
        st.session_state["cfg"] = cfg
        st.session_state["infer_ok"] = ok


def _block_labels(cfg: dict) -> list[str]:
    blocks = cfg.get("blocks")
    if isinstance(blocks, int):
        return [f"B{idx}" for idx in range(blocks)]
    if isinstance(blocks, list):
        return [str(b) for b in blocks]
    num_blocks = cfg.get("num_blocks")
    if isinstance(num_blocks, int):
        return [f"B{idx}" for idx in range(num_blocks)]
    return []


def _ir_resident_rows(cfg: dict) -> list[dict]:
    gui = cfg.get("gui") if isinstance(cfg.get("gui"), dict) else {}
    residents = gui.get("residents") if isinstance(gui.get("residents"), dict) else {}
    ir_section = residents.get("IR") if isinstance(residents.get("IR"), dict) else {}

    rows: list[dict] = []
    for track in IR_TRACKS:
        names = ir_section.get(track, [])
        if not isinstance(names, list):
            continue
        for name in names:
            if not isinstance(name, str) or not name.strip():
                continue
            rows.append({"Track": track, "Resident": name.strip()})

    track_order = {t: idx for idx, t in enumerate(IR_TRACKS)}
    rows.sort(key=lambda r: (track_order.get(r["Track"], 999), r["Resident"].casefold()))
    return rows


def _parse_block_label(block, block_labels: list[str]) -> Optional[str]:
    if isinstance(block, int):
        if 0 <= block < len(block_labels):
            return block_labels[block]
        return None
    if isinstance(block, str):
        if block in block_labels:
            return block
        return None
    return None


def _blocked_set(cfg: dict, block_labels: list[str]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    raw = cfg.get("blocked", [])
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            resident = entry.get("resident")
            rotation = entry.get("rotation")
            block_label = _parse_block_label(entry.get("block"), block_labels)
            if resident and rotation and block_label:
                out.add((str(resident), str(block_label), str(rotation)))
    elif isinstance(raw, dict):
        for resident, blocks in raw.items():
            if not isinstance(blocks, dict):
                continue
            for block_key, rotations in blocks.items():
                block_label = _parse_block_label(block_key, block_labels)
                if not block_label or not isinstance(rotations, list):
                    continue
                for rotation in rotations:
                    if rotation:
                        out.add((str(resident), str(block_label), str(rotation)))
    return out


def _forced_set(cfg: dict, block_labels: list[str]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    raw = cfg.get("forced", {})
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            resident = entry.get("resident")
            rotation = entry.get("rotation")
            block_label = _parse_block_label(entry.get("block"), block_labels)
            if resident and rotation and block_label:
                out.add((str(resident), str(block_label), str(rotation)))
    elif isinstance(raw, dict):
        for resident, blocks in raw.items():
            if not isinstance(blocks, dict):
                continue
            for block_key, rotation in blocks.items():
                block_label = _parse_block_label(block_key, block_labels)
                if not block_label:
                    continue
                if isinstance(rotation, list):
                    if len(rotation) != 1:
                        continue
                    rotation = rotation[0]
                if rotation:
                    out.add((str(resident), str(block_label), str(rotation)))
    return out


def _blocked_dict(blocked: set[tuple[str, str, str]]) -> dict:
    out: dict = {}
    rot_order = {r: idx for idx, r in enumerate(ROTATION_COLUMNS)}
    for resident, block, rotation in blocked:
        out.setdefault(resident, {}).setdefault(block, []).append(rotation)
    for resident, blocks in out.items():
        for block, rotations in blocks.items():
            blocks[block] = sorted(set(rotations), key=lambda r: rot_order.get(r, 999))
    return out


def _forced_dict(forced: set[tuple[str, str, str]]) -> dict:
    out: dict = {}
    for resident, block, rotation in forced:
        out.setdefault(resident, {})[block] = rotation
    return out


def _keyify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


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


st.set_page_config(page_title=APP_TITLE, layout="wide")

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

    /* Modern header */
    .hero {
        position: relative;
        border-radius: 16px;
        padding: 1.15rem 1.35rem;
        margin: 0 0 0.9rem 0;
        background:
          radial-gradient(900px 260px at 10% 0%, rgba(59, 130, 246, 0.16), rgba(0,0,0,0) 60%),
          radial-gradient(900px 260px at 90% 10%, rgba(168, 85, 247, 0.14), rgba(0,0,0,0) 60%),
          var(--secondary-background-color);
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow:
          0 18px 40px rgba(0,0,0,0.10);
        overflow: hidden;
    }
    @media (prefers-color-scheme: dark) {
        .hero { border-color: rgba(255,255,255,0.10); }
    }
    .hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0));
        opacity: 0.35;
        pointer-events: none;
    }
    .hero-row {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .hero-title {
        position: relative;
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
        font-weight: 800;
        letter-spacing: -0.02em;
        font-size: clamp(24px, 2.6vw, 40px);
        line-height: 1.1;
        color: var(--text-color);
    }
    .hero-badge {
        position: relative;
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 12px;
        line-height: 1;
        font-weight: 600;
        color: var(--text-color);
        padding: 0.45rem 0.65rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(59, 130, 246, 0.18);
        white-space: nowrap;
    }
    .hero-sub {
        position: relative;
        margin: 0.55rem 0 0 0;
        color: var(--text-color);
        opacity: 0.78;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-row">
        <h1 class="hero-title">{APP_TITLE}</h1>
        <div class="hero-badge">CONFIG</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        "Requests",
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
            row[rot] = float(req.get(track, {}).get(rot, 0))
        row["Total Blocks"] = sum(row[rot] for rot in ROTATION_COLUMNS)
        rows.append(row)

    rows_df = pd.DataFrame(rows, columns=["Track"] + DISPLAY_COLUMNS)
    with st.expander("Edit class/year requirements", expanded=False):
        edited_df = st.data_editor(
            rows_df,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Track": st.column_config.TextColumn(disabled=True),
                "MH-IR": st.column_config.NumberColumn(min_value=0, step=0.5),
                "MH-CT/US": st.column_config.NumberColumn(min_value=0, step=0.5),
                "48X-IR": st.column_config.NumberColumn(min_value=0, step=0.5),
                "48X-CT/US": st.column_config.NumberColumn(min_value=0, step=0.5),
                "KIR": st.column_config.NumberColumn(min_value=0, step=1.0),
                "Total Blocks": st.column_config.NumberColumn(disabled=True),
            },
            key="class_year_table",
        )

    updated_req = {}
    for row in edited_df.to_dict("records"):
        track = row["Track"]
        updated_req[track] = {rot: float(row.get(rot, 0) or 0) for rot in ROTATION_COLUMNS}

    cfg["gui"]["class_year_requirements"] = updated_req
    if updated_req != prev_req:
        # Trigger a second rerun so computed cells (e.g., Total Blocks) update immediately.
        st.rerun()

    non_integer_tracks = []
    for track in CLASS_TRACKS:
        total_blocks = sum(updated_req[track][rot] for rot in ROTATION_COLUMNS)
        if not math.isclose(total_blocks, round(total_blocks)):
            non_integer_tracks.append(f"{track} ({total_blocks:.1f})")
    if non_integer_tracks:
        st.warning(
            "Total blocks must be a whole number before solving. "
            f"Check: {', '.join(non_integer_tracks)}"
        )

    non_whole_kir_tracks = []
    for track in CLASS_TRACKS:
        if not track.startswith("IR"):
            continue
        kir = float(updated_req.get(track, {}).get("KIR", 0) or 0)
        if not math.isclose(kir, round(kir)):
            non_whole_kir_tracks.append(f"{track} (KIR={kir:.1f})")
    if non_whole_kir_tracks:
        st.warning(
            "KIR requirements must be a whole number (no 0.5 blocks). "
            f"Check: {', '.join(non_whole_kir_tracks)}"
        )

    requirements = {}
    for track in CLASS_TRACKS:
        requirements[track] = dict(updated_req[track])
    cfg["requirements"] = requirements

    counts = _track_counts(cfg)
    avail = {rot: 0 for rot in ROTATION_COLUMNS}
    for track in CLASS_TRACKS:
        for rot in ROTATION_COLUMNS:
            avail[rot] += counts[track] * requirements[track][rot]

    display_rows = []
    for track in CLASS_TRACKS:
        r = {"Track": track}
        for rot in ROTATION_COLUMNS:
            r[rot] = requirements[track][rot]
        r["Total Blocks"] = sum(requirements[track][rot] for rot in ROTATION_COLUMNS)
        display_rows.append(r)

    display_df = pd.DataFrame(display_rows, columns=["Track"] + DISPLAY_COLUMNS)
    for col in ROTATION_COLUMNS + ["Total Blocks"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

    def _shade_zero(value: int | float | None) -> str:
        if value is None or pd.isna(value):
            return ""
        if abs(float(value)) < 1e-9:
            return "background-color: #f4f4f4; color: #8a8a8a;"
        return "background-color: #e9f7ef; font-weight: 600;"

    st.dataframe(
        display_df.style.map(_shade_zero, subset=ROTATION_COLUMNS + ["Total Blocks"]).format(
            "{:.1f}",
            subset=ROTATION_COLUMNS + ["Total Blocks"],
            na_rep="",
        ),
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
        rotation_display.style.apply(_make_row_style(rotation_df), axis=1).format(
            "{:.1f}",
            subset=["Available"],
            na_rep="",
        ),
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
        location_display.style.apply(_make_row_style(location_df), axis=1).format(
            "{:.1f}",
            subset=["Available"],
            na_rep="",
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    st.subheader("Requests")
    st.caption(
        "Use Off to forbid assignments. Use On to force a specific assignment. "
        "Both are hard constraints and may make the model infeasible."
    )

    block_labels = _block_labels(cfg)
    ir_rows = _ir_resident_rows(cfg)
    forced_set = _forced_set(cfg, block_labels)
    blocked_set = _blocked_set(cfg, block_labels)

    if not block_labels:
        st.error("No blocks configured.")
        st.stop()
    if not ir_rows:
        st.info("Add IR residents first (Residents tab).")
        st.stop()

    left, main = st.columns([1, 4], gap="large")

    with left:
        st.markdown("### Resident")
        if "requests_selected_resident_idx" not in st.session_state:
            st.session_state["requests_selected_resident_idx"] = 0

        # Clamp in case residents change (e.g., after loading a new config).
        selected_idx = int(st.session_state.get("requests_selected_resident_idx", 0) or 0)
        selected_idx = max(0, min(selected_idx, len(ir_rows) - 1))
        st.session_state["requests_selected_resident_idx"] = selected_idx

        for i, row in enumerate(ir_rows):
            label = f"{row['Track']} — {row['Resident']}"
            clicked = st.button(
                label,
                key=f"requests_pick_{i}",
                type="primary" if i == selected_idx else "secondary",
                use_container_width=True,
            )
            if clicked and i != selected_idx:
                st.session_state["requests_selected_resident_idx"] = i
                st.rerun()

        selected = ir_rows[selected_idx]
        selected_resident = selected["Resident"]
        st.caption(f"{selected['Track']}")

        resident_off = {(blk, rot) for (res, blk, rot) in blocked_set if res == selected_resident}
        resident_on = {(blk, rot) for (res, blk, rot) in forced_set if res == selected_resident}
        st.caption(f"Off checks: {len(resident_off)}")
        st.caption(f"On checks: {len(resident_on)}")

    with main:
        sub_off, sub_on = st.tabs(["Off", "On"])

        with sub_off:
            st.caption(
                "Checked = resident cannot be assigned to that rotation in that block. "
                "You may check multiple rotations per block."
            )
            current_on_by_block: dict[str, str] = {}
            for res, blk, rot in forced_set:
                if res == selected_resident and blk not in current_on_by_block:
                    current_on_by_block[blk] = rot

            current_off = {(blk, rot) for (res, blk, rot) in blocked_set if res == selected_resident}

            header = st.columns([1.2, 0.9, 0.9] + [1.0] * len(ROTATION_COLUMNS) + [1.1])
            header[0].markdown("**Block**")
            header[1].markdown("**Select**")
            header[2].markdown("**Clear**")
            for idx, rot in enumerate(ROTATION_COLUMNS):
                header[3 + idx].markdown(f"**{rot}**")
            header[-1].markdown("**On**")

            for block in block_labels:
                k_res = _keyify(selected_resident)
                k_blk = _keyify(block)
                forced_rot = current_on_by_block.get(block, "")

                cols = st.columns([1.2, 0.9, 0.9] + [1.0] * len(ROTATION_COLUMNS) + [1.1])
                cols[0].write(block)

                if cols[1].button("All", key=f"requests_off_all_{k_res}_{k_blk}", use_container_width=True):
                    for rot in ROTATION_COLUMNS:
                        if forced_rot and rot == forced_rot:
                            st.session_state[f"requests_off_{k_res}_{k_blk}_{_keyify(rot)}"] = False
                        else:
                            st.session_state[f"requests_off_{k_res}_{k_blk}_{_keyify(rot)}"] = True
                    st.rerun()

                if cols[2].button("None", key=f"requests_off_none_{k_res}_{k_blk}", use_container_width=True):
                    for rot in ROTATION_COLUMNS:
                        st.session_state[f"requests_off_{k_res}_{k_blk}_{_keyify(rot)}"] = False
                    st.rerun()

                for idx, rot in enumerate(ROTATION_COLUMNS):
                    k_rot = _keyify(rot)
                    key = f"requests_off_{k_res}_{k_blk}_{k_rot}"
                    checked = cols[3 + idx].checkbox(
                        "off",
                        value=(block, rot) in current_off,
                        key=key,
                        label_visibility="collapsed",
                        disabled=bool(forced_rot and rot == forced_rot),
                    )
                    if forced_rot and rot == forced_rot and checked:
                        st.session_state[key] = False

                cols[-1].write(forced_rot or "—")

            next_blocked = {t for t in blocked_set if t[0] != selected_resident}
            for block in block_labels:
                k_res = _keyify(selected_resident)
                k_blk = _keyify(block)
                forced_rot = current_on_by_block.get(block, "")
                for rot in ROTATION_COLUMNS:
                    if forced_rot and rot == forced_rot:
                        continue
                    key = f"requests_off_{k_res}_{k_blk}_{_keyify(rot)}"
                    if bool(st.session_state.get(key, (block, rot) in current_off)):
                        next_blocked.add((selected_resident, block, rot))

            # If something is forced ON, it cannot also be blocked OFF for that exact rotation.
            next_blocked = {t for t in next_blocked if t not in forced_set}
            cfg["blocked"] = _blocked_dict(next_blocked)

        with sub_on:
            st.caption("Select a single rotation per block (or blank for no On request).")
            current_on_by_block: dict[str, str] = {}
            for res, blk, rot in forced_set:
                if res == selected_resident and blk not in current_on_by_block:
                    current_on_by_block[blk] = rot

            blocked_now = _blocked_set(cfg, block_labels)
            blocked_by_block: dict[str, set[str]] = {}
            for res, blk, rot in blocked_now:
                if res != selected_resident:
                    continue
                blocked_by_block.setdefault(blk, set()).add(rot)

            for block in block_labels:
                k_res = _keyify(selected_resident)
                k_blk = _keyify(block)
                key = f"requests_on_{k_res}_{k_blk}"

                blocked_rots = blocked_by_block.get(block, set())
                options = [""] + [rot for rot in ROTATION_COLUMNS if rot not in blocked_rots]
                current = current_on_by_block.get(block, "")
                if current not in options:
                    current = ""

                cols = st.columns([1.2, 4.0])
                cols[0].write(block)
                cols[1].selectbox(
                    "Rotation",
                    options=options,
                    index=options.index(current),
                    key=key,
                    label_visibility="collapsed",
                    help="Set blank to clear the On request for this block.",
                )

            next_forced = {t for t in forced_set if t[0] != selected_resident}
            for block in block_labels:
                k_res = _keyify(selected_resident)
                k_blk = _keyify(block)
                rot = str(st.session_state.get(f"requests_on_{k_res}_{k_blk}", "") or "").strip()
                if rot:
                    next_forced.add((selected_resident, block, rot))

            cfg["forced"] = _forced_dict(next_forced)

            # Remove any exact blocked conflicts now that forced is updated.
            blocked_now = _blocked_set(cfg, block_labels)
            blocked_now = {t for t in blocked_now if t not in next_forced}
            cfg["blocked"] = _blocked_dict(blocked_now)


with tabs[3]:
    st.subheader("Constraints")
    modes = cfg["gui"]["constraints"].get("modes", {})
    if not isinstance(modes, dict):
        modes = {}
    params = cfg["gui"]["constraints"].get("params", {})
    if not isinstance(params, dict):
        params = {}
    cfg["gui"]["constraints"]["params"] = params

    def _constraint_title_and_description(spec, spec_params: dict, num_blocks: int) -> tuple[str, str]:
        hard_or_pref = "Preference" if spec.softenable else "Hard constraint"
        if spec.id == "one_place":
            return (
                "One rotation per resident per block",
                f"{hard_or_pref}. Prevents a resident from being scheduled in multiple places in the same block.",
            )
        if spec.id == "blocked":
            return (
                "Honor Off requests",
                f"{hard_or_pref}. Enforces the Off selections from the Requests tab.",
            )
        if spec.id == "forced":
            return (
                "Honor On requests",
                f"{hard_or_pref}. Enforces the On selections from the Requests tab.",
            )
        if spec.id == "block_total_zero_or_full":
            return (
                "Block totals must be 0 or 1.0 FTE",
                f"{hard_or_pref}. If a resident is scheduled in a block, they must total 1.0 FTE; otherwise 0.",
            )
        if spec.id == "no_half_non_ir5":
            return (
                "No half-block assignments for DR",
                f"{hard_or_pref}. DR residents must take full 1.0 FTE rotations within a block.",
            )
        if spec.id == "coverage_48x_ir":
            op = str(spec_params.get("op", "=="))
            target_units = spec_params.get("target_units", 2)
            try:
                target_fte = int(round(int(target_units) / 2))
            except Exception:
                target_fte = 1
            op_text = {"<=": "at most", "==": "exactly", ">=": "at least"}.get(op, "exactly")
            return (
                "48X-IR coverage per block",
                f"{hard_or_pref}. Requires {op_text} {target_fte}.0 FTE of 48X-IR each block.",
            )
        if spec.id == "coverage_48x_ctus":
            op = str(spec_params.get("op", "=="))
            target_units = spec_params.get("target_units", 2)
            try:
                target_fte = int(round(int(target_units) / 2))
            except Exception:
                target_fte = 1
            op_text = {"<=": "at most", "==": "exactly", ">=": "at least"}.get(op, "exactly")
            return (
                "48X-CT/US coverage per block",
                f"{hard_or_pref}. Requires {op_text} {target_fte}.0 FTE of 48X-CT/US each block.",
            )
        if spec.id == "mh_total_minmax":
            min_fte = spec_params.get("min_fte", 3)
            max_fte = spec_params.get("max_fte", 4)
            return (
                "MH total coverage per block",
                f"{hard_or_pref}. Combined MH-IR + MH-CT/US must be between {min_fte}.0 and {max_fte}.0 FTE per block.",
            )
        if spec.id == "mh_ctus_cap":
            max_fte = spec_params.get("max_fte", 1)
            return (
                "MH-CT/US cap per block",
                f"{hard_or_pref}. Limits MH-CT/US to at most {max_fte}.0 FTE per block.",
            )
        if spec.id == "kir_cap":
            max_fte = spec_params.get("max_fte", 2)
            return (
                "KIR cap per block",
                f"{hard_or_pref}. Limits KIR to at most {max_fte}.0 FTE per block.",
            )
        if spec.id == "track_requirements":
            return (
                "Per-class rotation totals",
                f"{hard_or_pref}. Uses the Class/Year Assignments tab to enforce total blocks per resident (IR/DR class).",
            )
        if spec.id == "ir5_mh_min_per_block":
            return (
                "IR5 MH-IR minimum per block",
                f"{hard_or_pref}. Ensures at least 1.0 FTE of IR5 MH-IR coverage per block.",
            )
        if spec.id == "ir4_plus_mh_cap":
            ir_min_year = spec_params.get("ir_min_year", 4)
            max_fte = spec_params.get("max_fte", 2)
            return (
                "Limit senior residents on MH-IR per block",
                f"{hard_or_pref}. 'Senior' means IR{ir_min_year}+; cap is {max_fte}.0 FTE per block.",
            )
        if spec.id == "dr1_early_block":
            first_n = spec_params.get("first_n_blocks", 4)
            return (
                "Keep DR1 off MH-IR early",
                f"{hard_or_pref}. DR1 cannot do MH-IR in the first {first_n} block(s).",
            )
        if spec.id == "ir3_late_block":
            after_block = spec_params.get("after_block", 7)
            after_label = f"B{after_block}" if after_block <= num_blocks else "the end"
            return (
                "Keep IR3 off MH-IR / 48X-IR late",
                f"{hard_or_pref}. Starting at {after_label}, IR3 cannot do MH-IR or 48X-IR.",
            )
        if spec.id == "first_timer":
            return (
                "First-timer MH-IR limit",
                f"{hard_or_pref}. Limits first-time MH-IR residents (DR1/IR1) to 1 per block.",
            )
        if spec.id == "consec_full_mh":
            max_consecutive = spec_params.get("max_consecutive", 3)
            return (
                "Avoid consecutive full MH-IR blocks",
                f"{hard_or_pref}. Prevents (or discourages) runs of {max_consecutive} full MH-IR blocks in a row.",
            )
        if spec.id == "no_sequential_year1_3":
            return (
                "No back-to-back blocks for years 1–3",
                f"{hard_or_pref}. Prevents residents in years 1–3 (DR1–3, IR1–3) from being scheduled in consecutive blocks.",
            )

        # Fallback
        return (spec.label, f"{hard_or_pref}.")

    def _render_spec_mode(spec, title: str, description: str, *, allow_try: bool) -> None:
        mode_display = {
            "always": "On (hard)",
            "if_able": "Try (soft)",
            "disabled": "Off",
        }
        if allow_try:
            options = ["always", "if_able", "disabled"]
        else:
            options = ["always", "disabled"]

        default_mode = modes.get(spec.id, "if_able" if spec.softenable else "always")
        if default_mode not in options:
            default_mode = "always"

        selection = st.radio(
            title,
            options=options,
            format_func=lambda v: mode_display.get(v, v),
            horizontal=True,
            index=options.index(default_mode),
            key=f"mode_{spec.id}",
        )
        modes[spec.id] = selection
        st.caption(description)

    def _params_for(spec_id: str) -> dict:
        raw = params.get(spec_id)
        if not isinstance(raw, dict):
            raw = {}
            params[spec_id] = raw
        return raw

    def _render_spec_params(spec, num_blocks: int) -> None:
        if spec.id == "coverage_48x_ir" or spec.id == "coverage_48x_ctus":
            p = _params_for(spec.id)
            op = str(p.get("op", "=="))
            if op not in {"<=", "==", ">="}:
                op = "=="
            target_units = p.get("target_units", 2)
            if not isinstance(target_units, int) or target_units < 0:
                target_units = 2

            c1, c2 = st.columns([1.3, 2.0])
            op_sel = c1.selectbox(
                "Inequality",
                options=["<=", "==", ">="],
                index=["<=", "==", ">="].index(op),
                format_func={"<=": "≤", "==": "Exactly", ">=": "≥"}.get,
                key=f"cparam_{spec.id}_op",
            )
            fte_options = list(range(0, 6))  # whole-number FTE only
            current_fte = int(round(target_units / 2))
            current_fte = max(0, min(current_fte, fte_options[-1]))
            target_fte_sel = c2.selectbox(
                "Target (FTE)",
                options=fte_options,
                index=fte_options.index(current_fte),
                key=f"cparam_{spec.id}_target_units",
            )
            p["op"] = op_sel
            p["target_units"] = int(target_fte_sel) * 2

        elif spec.id == "mh_total_minmax":
            p = _params_for(spec.id)
            min_fte = p.get("min_fte", 3)
            max_fte = p.get("max_fte", 4)
            if not isinstance(min_fte, int) or min_fte < 0:
                min_fte = 3
            if not isinstance(max_fte, int) or max_fte < 0:
                max_fte = 4

            fte_options = list(range(0, 11))
            c1, c2 = st.columns(2)
            min_sel = c1.selectbox(
                "Min (FTE)",
                options=fte_options,
                index=fte_options.index(min_fte) if min_fte in fte_options else fte_options.index(3),
                key="cparam_mh_total_minmax_min_fte",
            )
            max_sel = c2.selectbox(
                "Max (FTE)",
                options=fte_options,
                index=fte_options.index(max_fte) if max_fte in fte_options else fte_options.index(4),
                key="cparam_mh_total_minmax_max_fte",
            )
            p["min_fte"] = int(min_sel)
            p["max_fte"] = int(max_sel)

        elif spec.id == "mh_ctus_cap" or spec.id == "kir_cap":
            p = _params_for(spec.id)
            max_fte = p.get("max_fte", 1 if spec.id == "mh_ctus_cap" else 2)
            if not isinstance(max_fte, int) or max_fte < 0:
                max_fte = 1 if spec.id == "mh_ctus_cap" else 2
            options = list(range(0, 6))
            fallback = 1 if spec.id == "mh_ctus_cap" else 2
            max_sel = st.selectbox(
                "Max per block (FTE)",
                options=options,
                index=options.index(max_fte) if max_fte in options else options.index(fallback),
                key=f"cparam_{spec.id}_max_fte",
            )
            p["max_fte"] = int(max_sel)

        elif spec.id == "ir4_plus_mh_cap":
            p = _params_for(spec.id)
            ir_min_year = p.get("ir_min_year", 4)
            max_fte = p.get("max_fte", 2)
            if not isinstance(ir_min_year, int) or ir_min_year < 1 or ir_min_year > 5:
                ir_min_year = 4
            if not isinstance(max_fte, int) or max_fte < 0:
                max_fte = 2
            c1, c2 = st.columns([1.3, 2.0])
            ir_sel = c1.selectbox(
                "IR year+",
                options=[1, 2, 3, 4, 5],
                index=[1, 2, 3, 4, 5].index(ir_min_year),
                key="cparam_ir4_plus_mh_cap_ir_min_year",
            )
            max_sel = c2.selectbox(
                "Max per block (FTE)",
                options=list(range(0, 6)),
                index=list(range(0, 6)).index(max_fte) if max_fte in range(0, 6) else 2,
                key="cparam_ir4_plus_mh_cap_max_fte",
            )
            p["ir_min_year"] = int(ir_sel)
            p["max_fte"] = int(max_sel)

        elif spec.id == "dr1_early_block":
            p = _params_for(spec.id)
            first_n = p.get("first_n_blocks", 4)
            if not isinstance(first_n, int) or first_n < 0:
                first_n = 4
            options = list(range(0, num_blocks + 1))
            sel = st.selectbox(
                "First N blocks",
                options=options,
                index=options.index(first_n) if first_n in options else options.index(min(4, num_blocks)),
                key="cparam_dr1_early_block_first_n_blocks",
            )
            p["first_n_blocks"] = int(sel)

        elif spec.id == "ir3_late_block":
            p = _params_for(spec.id)
            after_block = p.get("after_block", 7)
            if not isinstance(after_block, int) or after_block < 0:
                after_block = 7
            options = list(range(0, num_blocks + 1))
            sel = st.selectbox(
                "After block #",
                options=options,
                index=options.index(after_block) if after_block in options else options.index(min(7, num_blocks)),
                help="Restrictions apply starting at the next block (e.g., 7 means starting at block 8).",
                key="cparam_ir3_late_block_after_block",
            )
            p["after_block"] = int(sel)

        elif spec.id == "consec_full_mh":
            p = _params_for(spec.id)
            max_consecutive = p.get("max_consecutive", 3)
            if not isinstance(max_consecutive, int) or max_consecutive < 2:
                max_consecutive = 3
            upper = max(2, min(num_blocks, 8))
            options = list(range(2, upper + 1))
            sel = st.selectbox(
                "Avoid N full MH-IR blocks in a row",
                options=options,
                index=options.index(max_consecutive) if max_consecutive in options else options.index(min(3, upper)),
                key="cparam_consec_full_mh_max_consecutive",
            )
            p["max_consecutive"] = int(sel)

    categories = {
        "Core Rules": {"one_place", "block_total_zero_or_full", "no_half_non_ir5"},
        "Requests": {"blocked", "forced"},
        "Coverage & Caps": {
            "coverage_48x_ir",
            "coverage_48x_ctus",
            "mh_total_minmax",
            "mh_ctus_cap",
            "kir_cap",
            "ir5_mh_min_per_block",
            "ir4_plus_mh_cap",
            "track_requirements",
        },
        "Track Rules": {"dr1_early_block", "ir3_late_block"},
        "Preferences": {"first_timer", "consec_full_mh", "no_sequential_year1_3"},
    }

    used_ids = set().union(*categories.values()) if categories else set()
    other_ids = [spec.id for spec in CONSTRAINT_SPECS if spec.id not in used_ids]
    tab_names = list(categories.keys()) + (["Other"] if other_ids else [])
    sub_tabs = st.tabs(tab_names)

    num_blocks = _num_blocks(cfg)
    for tab_name, tab in zip(tab_names, sub_tabs):
        with tab:
            ids = categories.get(tab_name, set())
            allow_try = tab_name != "Core Rules"
            for spec in CONSTRAINT_SPECS:
                if tab_name == "Other":
                    if spec.id not in other_ids:
                        continue
                else:
                    if spec.id not in ids:
                        continue
                if tab_name in {"Coverage & Caps", "Track Rules", "Preferences"}:
                    spec_params = _params_for(spec.id)
                    title, description = _constraint_title_and_description(spec, spec_params, num_blocks=num_blocks)
                    _render_spec_mode(spec, title=title, description=description, allow_try=allow_try)
                    _render_spec_params(spec, num_blocks=num_blocks)
                    st.divider()
                else:
                    spec_params = _params_for(spec.id)
                    title, description = _constraint_title_and_description(spec, spec_params, num_blocks=num_blocks)
                    _render_spec_mode(spec, title=title, description=description, allow_try=allow_try)

    cfg["gui"]["constraints"]["modes"] = modes

with tabs[4]:
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

if False:  # Checks tab removed
    """
    st.subheader("Checks")
    st.caption("Quick preflight checks for common conflicts and settings that often cause infeasibility or relaxations.")

    block_labels = _block_labels(cfg)
    if not block_labels:
        st.error("No blocks configured.")
        st.stop()

    forced_set = _forced_set(cfg, block_labels)
    blocked_set = _blocked_set(cfg, block_labels)
    block_idx = {b: i for i, b in enumerate(block_labels)}

    modes = cfg.get("gui", {}).get("constraints", {}).get("modes", {})
    if not isinstance(modes, dict):
        modes = {}
    params = cfg.get("gui", {}).get("constraints", {}).get("params", {})
    if not isinstance(params, dict):
        params = {}

    spec_by_id = {spec.id: spec for spec in CONSTRAINT_SPECS}

    def _mode_for(constraint_id: str) -> str:
        spec = spec_by_id.get(constraint_id)
        default = "if_able" if (spec and spec.softenable) else "always"
        mode = str(modes.get(constraint_id, default)).lower()
        if mode not in {"always", "if_able", "disabled"}:
            return default
        if constraint_id in {"one_place", "block_total_zero_or_full", "no_half_non_ir5"} and mode == "if_able":
            return "always"
        return mode

    def _params_for(constraint_id: str) -> dict:
        raw = params.get(constraint_id, {})
        return raw if isinstance(raw, dict) else {}

    one_place_active = _mode_for("one_place") != "disabled"

    def _residents_from_cfg() -> list[dict]:
        raw = cfg.get("residents")
        if isinstance(raw, list) and raw:
            return raw
        gui_res = cfg.get("gui", {}).get("residents")
        if isinstance(gui_res, dict):
            try:
                return expand_residents(gui_res)
            except Exception:
                return []
        return []

    residents_list = _residents_from_cfg()
    if not residents_list:
        st.warning("Residents list missing. Visit the Residents tab or load a config that includes residents.")

    resident_to_track: dict[str, str] = {}
    for entry in residents_list:
        if isinstance(entry, dict) and entry.get("id") and entry.get("track"):
            resident_to_track[str(entry["id"])] = str(entry["track"])

    forced_by_res_block: dict[tuple[str, str], str] = {}
    forced_units_by_block_rot: dict[tuple[str, str], int] = {}
    for resident, block, rotation in forced_set:
        forced_by_res_block[(resident, block)] = rotation
        forced_units_by_block_rot[(block, rotation)] = forced_units_by_block_rot.get((block, rotation), 0) + 2

    results: list[dict] = []

    def _add(level: str, issue: str, details: str, suggestion: str = "") -> None:
        results.append({"Level": level, "Issue": issue, "Details": details, "Suggestion": suggestion})

    def _level_for(mode: str) -> Optional[str]:
        if mode == "always":
            return "Error"
        if mode == "if_able":
            return "Warning"
        return None

    def _resident_can_take(resident: str, block: str, rotation: str, *, strict_track_rules: bool) -> bool:
        if (resident, block, rotation) in blocked_set:
            return False

        forced_rot = forced_by_res_block.get((resident, block))
        if one_place_active and forced_rot and forced_rot != rotation:
            return False

        if not strict_track_rules:
            return True

        b_idx = block_idx.get(block, -1)
        track = resident_to_track.get(resident, "")

        dr1_mode = _mode_for("dr1_early_block")
        if dr1_mode == "always" and track == "DR1" and rotation == "MH-IR":
            first_n = int(_params_for("dr1_early_block").get("first_n_blocks", 4) or 0)
            if b_idx < first_n:
                return False

        ir3_mode = _mode_for("ir3_late_block")
        if ir3_mode == "always" and track == "IR3" and rotation in {"MH-IR", "48X-IR"}:
            after_block = int(_params_for("ir3_late_block").get("after_block", 7) or 0)
            if b_idx >= after_block:
                return False

        return True

    # Basic conflicts
    conflicts = forced_set.intersection(blocked_set)
    if conflicts:
        samples = ", ".join(sorted({f"{r} {b} {rot}" for (r, b, rot) in list(conflicts)[:5]}))
        _add(
            "Error",
            "On/Off conflict",
            f"Same resident/block/rotation is both forced On and blocked Off (e.g. {samples}).",
            "Fix in Requests tab: clear either the On or Off selection for those cells.",
        )

    forced_multi: dict[tuple[str, str], list[str]] = {}
    for resident, block, rotation in forced_set:
        forced_multi.setdefault((resident, block), []).append(rotation)
    bad_multi = {k: v for k, v in forced_multi.items() if len(set(v)) > 1}
    if bad_multi:
        (resident, block), rots = next(iter(bad_multi.items()))
        _add(
            "Error",
            "Multiple On rotations in one block",
            f"{resident} has multiple forced On rotations in {block}: {sorted(set(rots))}.",
            "Fix in Requests → On: each block must have at most one forced rotation per resident.",
        )

    # Coverage checks
    def _check_coverage(spec_id: str, rotation: str) -> None:
        mode = _mode_for(spec_id)
        level = _level_for(mode)
        if level is None:
            return
        p = _params_for(spec_id)
        op = str(p.get("op", "==")).strip()
        if op not in {"<=", "==", ">="}:
            op = "=="
        try:
            target_units = int(p.get("target_units", 2))
        except Exception:
            target_units = 2
        target_units = 2 * int(round(target_units / 2))  # whole-number FTE only

        for block in block_labels:
            forced_units = forced_units_by_block_rot.get((block, rotation), 0)
            max_units = 0
            for resident in resident_to_track.keys():
                if _resident_can_take(resident, block, rotation, strict_track_rules=True):
                    max_units += 2
            if op == "<=" and forced_units > target_units:
                _add(
                    level,
                    f"{rotation} coverage exceeds target",
                    f"{block}: forced {forced_units/2:.1f} FTE > target {target_units/2:.1f} FTE.",
                    "Lower the target, change the inequality, or clear conflicting On requests.",
                )
            if op in {"==", ">="} and max_units < target_units:
                _add(
                    level,
                    f"{rotation} coverage target unreachable",
                    f"{block}: max possible {max_units/2:.1f} FTE < target {target_units/2:.1f} FTE.",
                    "Lower the target, change to ≤, or clear Off requests that remove eligible residents.",
                )
            if op == "==" and forced_units > target_units:
                _add(
                    level,
                    f"{rotation} forced coverage exceeds exact target",
                    f"{block}: forced {forced_units/2:.1f} FTE > exact target {target_units/2:.1f} FTE.",
                    "Clear conflicting On requests or change the target/inequality.",
                )

    _check_coverage("coverage_48x_ir", "48X-IR")
    _check_coverage("coverage_48x_ctus", "48X-CT/US")

    # MH total range + caps
    mh_mode = _mode_for("mh_total_minmax")
    mh_level = _level_for(mh_mode)
    if mh_level is not None:
        p = _params_for("mh_total_minmax")
        try:
            min_fte = int(p.get("min_fte", 3))
            max_fte = int(p.get("max_fte", 4))
        except Exception:
            min_fte, max_fte = 3, 4
        if min_fte > max_fte:
            _add("Error", "MH total range invalid", f"Min {min_fte}.0 FTE is greater than max {max_fte}.0 FTE.", "Fix the min/max values.")
        for block in block_labels:
            forced_mh_units = forced_units_by_block_rot.get((block, "MH-IR"), 0) + forced_units_by_block_rot.get(
                (block, "MH-CT/US"), 0
            )
            if forced_mh_units > 2 * max_fte:
                _add(
                    mh_level,
                    "MH total exceeds max",
                    f"{block}: forced MH total {forced_mh_units/2:.1f} FTE > max {max_fte}.0 FTE.",
                    "Raise the max or clear conflicting On requests.",
                )
            max_mh_units = 0
            for resident in resident_to_track.keys():
                can_mh_ir = _resident_can_take(resident, block, "MH-IR", strict_track_rules=True)
                can_mh_ct = _resident_can_take(resident, block, "MH-CT/US", strict_track_rules=True)
                if can_mh_ir or can_mh_ct:
                    max_mh_units += 2
            if forced_mh_units < 2 * min_fte and max_mh_units < 2 * min_fte:
                _add(
                    mh_level,
                    "MH total min unreachable",
                    f"{block}: max possible MH total {max_mh_units/2:.1f} FTE < min {min_fte}.0 FTE.",
                    "Lower the min or clear Off requests on MH rotations.",
                )

    def _check_cap(spec_id: str, rotation: str) -> None:
        mode = _mode_for(spec_id)
        level = _level_for(mode)
        if level is None:
            return
        p = _params_for(spec_id)
        try:
            max_fte = int(p.get("max_fte", 0))
        except Exception:
            max_fte = 0
        for block in block_labels:
            forced_units = forced_units_by_block_rot.get((block, rotation), 0)
            if forced_units > 2 * max_fte:
                _add(
                    level,
                    f"{rotation} cap exceeded",
                    f"{block}: forced {forced_units/2:.1f} FTE > cap {max_fte}.0 FTE.",
                    "Raise the cap or clear conflicting On requests.",
                )

    _check_cap("mh_ctus_cap", "MH-CT/US")
    _check_cap("kir_cap", "KIR")

    # Senior MH-IR cap
    senior_mode = _mode_for("ir4_plus_mh_cap")
    senior_level = _level_for(senior_mode)
    if senior_level is not None:
        p = _params_for("ir4_plus_mh_cap")
        try:
            ir_min_year = int(p.get("ir_min_year", 4))
            max_fte = int(p.get("max_fte", 2))
        except Exception:
            ir_min_year, max_fte = 4, 2

        def _is_senior(resident: str) -> bool:
            track = resident_to_track.get(resident, "")
            if not track.startswith("IR"):
                return False
            try:
                year = int(track[2:])
            except ValueError:
                return False
            return year >= ir_min_year

        for block in block_labels:
            forced_senior_units = 0
            for resident in resident_to_track.keys():
                if not _is_senior(resident):
                    continue
                if forced_by_res_block.get((resident, block)) == "MH-IR":
                    forced_senior_units += 2
            if forced_senior_units > 2 * max_fte:
                _add(
                    senior_level,
                    "Senior MH-IR cap exceeded",
                    f"{block}: forced senior MH-IR {forced_senior_units/2:.1f} FTE > cap {max_fte}.0 FTE (senior = IR{ir_min_year}+).",
                    "Raise the cap, increase the senior threshold, or clear conflicting On requests.",
                )

    # Track requirements sanity
    req_mode = _mode_for("track_requirements")
    req_level = _level_for(req_mode)
    requirements = cfg.get("requirements")
    if req_level is not None and isinstance(requirements, dict) and one_place_active:
        per_track_sum = {}
        for track, row in requirements.items():
            if not isinstance(row, dict):
                continue
            total = 0.0
            for rot in ROTATION_COLUMNS:
                try:
                    total += float(row.get(rot, 0))
                except Exception:
                    continue
            per_track_sum[str(track)] = total
        num_blocks = len(block_labels)
        too_high = {t: n for t, n in per_track_sum.items() if n > num_blocks + 1e-6}
        if too_high:
            track, total = next(iter(too_high.items()))
            _add(
                req_level,
                "Per-class requirements exceed available blocks",
                f"{track}: total required blocks across rotations is {total}, but there are only {num_blocks} blocks.",
                "Lower the per-class requirements totals or disable the 'one rotation per resident per block' core rule.",
            )

    # Try-mode warnings for track rules (informational)
    for constraint_id, label in [("dr1_early_block", "DR1 early MH-IR rule"), ("ir3_late_block", "IR3 late rule")]:
        if _mode_for(constraint_id) == "if_able":
            _add(
                "Info",
                f"{label} set to Try",
                "This rule can be relaxed by the solver if needed.",
                "If you want it strictly enforced, switch it to On (hard).",
            )

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    else:
        st.success("No issues found by quick checks.")

    st.divider()
    if st.button("Run quick feasibility probe (no schedule)", key="checks_probe_btn", type="primary"):
        try:
            schedule_input = load_schedule_input_from_data(cfg)
        except Exception as exc:
            st.error(f"Invalid configuration: {exc}")
        else:
            ok = _is_feasible(schedule_input)
            if ok:
                st.success("Feasibility probe: model is feasible with current settings (Try constraints may be relaxed).")
            else:
                st.error("Feasibility probe: model is infeasible with current settings.")
    """

with tabs[5]:
    st.subheader("Solve")
    st.caption("Runs the solver using the current in-app configuration (no YAML download required).")

    def _fmt_fte(value: float) -> str:
        text = f"{float(value):.1f}"
        if text.endswith(".0"):
            return text[:-2]
        return text

    def _objective_breakdown(sol_objective: dict, weights: dict) -> tuple[pd.DataFrame, int]:
        label_by_key = {
            "consec": "Consecutive full MH blocks",
            "first_timer": "First-timer assignments",
            "adj": "Adjacent assignments (Y1–Y3)",
        }
        weight_by_key = {
            "consec": int(weights.get("consec", 0)),
            "first_timer": int(weights.get("first_timer", 0)),
            "adj": int(weights.get("adj", 0)),
        }
        rows = []
        total = 0
        for key, label in label_by_key.items():
            count = int(sol_objective.get(key, 0) or 0)
            weight = int(weight_by_key.get(key, 0))
            weighted = count * weight
            if count:
                rows.append({"Penalty": label, "Count": count, "Weight": weight, "Weighted": weighted})
            total += weighted

        extra_keys = sorted(
            [k for k in sol_objective.keys() if k not in label_by_key],
            key=lambda s: str(s).casefold(),
        )
        for key in extra_keys:
            count = int(sol_objective.get(key, 0) or 0)
            if not count:
                continue
            rows.append({"Penalty": str(key), "Count": count, "Weight": "-", "Weighted": "-"})

        return pd.DataFrame(rows, columns=["Penalty", "Count", "Weight", "Weighted"]), total

    def _ir_totals_table(sol, schedule_input) -> pd.DataFrame:
        track_order = {t: idx for idx, t in enumerate(IR_TRACKS)}
        rows = []
        for resident in schedule_input.residents:
            if not isinstance(resident.track, str) or not resident.track.startswith("IR"):
                continue
            totals: dict[str, float] = {rot: 0.0 for rot in ROTATION_COLUMNS}
            blocks_map = sol.assignments.get(resident.resident_id, {})
            if not isinstance(blocks_map, dict):
                continue
            for block_assignments in blocks_map.values():
                if not isinstance(block_assignments, dict):
                    continue
                for rot in ROTATION_COLUMNS:
                    try:
                        totals[rot] += float(block_assignments.get(rot, 0) or 0.0)
                    except (TypeError, ValueError):
                        continue
            total = float(sum(totals.values()))
            if abs(total) < 1e-9:
                continue
            rows.append(
                {
                    "Track": resident.track,
                    "Resident": resident.resident_id,
                    **totals,
                    "Total": total,
                }
            )

        rows.sort(
            key=lambda r: (
                track_order.get(str(r["Track"]), 999),
                str(r["Resident"]).casefold(),
            )
        )
        return pd.DataFrame(rows, columns=["Track", "Resident"] + ROTATION_COLUMNS + ["Total"])

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
        st.session_state.pop("solve_input", None)
        try:
            schedule_input = load_schedule_input_from_data(cfg)
        except Exception as exc:
            st.error(f"Invalid configuration: {exc}")
        else:
            with st.spinner("Solving..."):
                result = solve_schedule(schedule_input)
            st.session_state["solve_result"] = result
            st.session_state["solve_csv"] = result_to_csv(result)
            st.session_state["solve_input"] = schedule_input

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
                schedule_input = st.session_state.get("solve_input")
                if schedule_input is not None:
                    st.markdown("**Objective (lower is better)**")
                    obj_df, obj_total = _objective_breakdown(sol.objective or {}, schedule_input.weights)
                    st.metric("Weighted objective score", obj_total)
                    if not obj_df.empty:
                        st.dataframe(obj_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No soft penalties in this solution.")

                st.markdown("**Assignments by Rotation**")
                blocks = _block_labels(cfg)
                if not blocks:
                    blocks = list(next(iter(sol.assignments.values())).keys()) if sol.assignments else []

                rot_rows = []
                for rot in ROTATION_COLUMNS:
                    row = {"Rotation": rot}
                    for block in blocks:
                        assigned = []
                        for resident, r_blocks in sol.assignments.items():
                            r_block = r_blocks.get(block) if isinstance(r_blocks, dict) else None
                            if not isinstance(r_block, dict):
                                continue
                            fte = r_block.get(rot)
                            if not fte:
                                continue
                            name = str(resident)
                            if float(fte) != 1.0:
                                name = f"{name} ({_fmt_fte(float(fte))})"
                            assigned.append(name)
                        row[block] = "\n".join(sorted(assigned, key=lambda s: s.casefold()))
                    rot_rows.append(row)

                rot_df = pd.DataFrame(rot_rows, columns=["Rotation"] + blocks)
                st.dataframe(
                    rot_df.style.set_properties(**{"white-space": "pre-line"}),
                    use_container_width=True,
                    hide_index=True,
                )

                if schedule_input is not None:
                    ir_totals_df = _ir_totals_table(sol, schedule_input)
                    if not ir_totals_df.empty:
                        numeric_cols = ROTATION_COLUMNS + ["Total"]
                        nonzero = ir_totals_df[numeric_cols].to_numpy()
                        nonzero = nonzero[nonzero > 0]
                        vmin = float(nonzero.min()) if nonzero.size else 0.0
                        vmax = float(nonzero.max()) if nonzero.size else 0.0
                        cmap = cm.get_cmap("viridis")

                        def _bg(value: int | float | None) -> str:
                            if value is None or pd.isna(value):
                                return ""
                            v = float(value)
                            if abs(v) < 1e-9:
                                return ""
                            t = 0.5 if math.isclose(vmin, vmax) else (v - vmin) / (vmax - vmin)
                            t = max(0.0, min(1.0, float(t)))
                            rgba = cmap(t)
                            hex_color = colors.to_hex(rgba, keep_alpha=False)
                            r, g, b = colors.to_rgb(hex_color)
                            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            text = "#111111" if luminance > 0.55 else "#ffffff"
                            return f"background-color: {hex_color}; color: {text}; font-weight: 600;"

                        def _fmt_cell(value: int | float | None) -> str:
                            if value is None or pd.isna(value):
                                return ""
                            v = float(value)
                            if abs(v) < 1e-9:
                                return ""
                            return _fmt_fte(v)

                        st.markdown("**IR resident rotation totals**")
                        st.dataframe(
                            ir_totals_df.style.map(_bg, subset=numeric_cols).format(
                                _fmt_cell,
                                subset=numeric_cols,
                                na_rep="",
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

        csv_text = st.session_state.get("solve_csv", "")
        if csv_text:
            col_dl.download_button(
                "Download CSV",
                data=csv_text,
                file_name="schedule-output.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tabs[6]:
    st.subheader("Save/Load Configuration")

    def _reset_widget_state() -> None:
        prefixes = (
            "ir_",
            "dr_",
            "requests_",
            "cparam_",
            "mode_",
            "prio_",
            "class_year_table",
            "num_solutions_input",
            "solution_select",
        )
        for key in list(st.session_state.keys()):
            if key.startswith(prefixes):
                del st.session_state[key]

    def _clear_pending_and_uploader() -> None:
        st.session_state.pop("pending_cfg", None)
        st.session_state.pop("pending_infer_ok", None)
        st.session_state["uploader_nonce"] = int(st.session_state.get("uploader_nonce", 0)) + 1

    load_col, save_col = st.columns(2, gap="large")

    with load_col:
        st.markdown("### Load")
        st.caption("Drag & drop a `.yml`/`.yaml` to preview, then apply it to the current session.")
        if "uploader_nonce" not in st.session_state:
            st.session_state["uploader_nonce"] = 0
        uploader_key = f"config_uploader_{st.session_state['uploader_nonce']}"
        uploaded = st.file_uploader(
            "Drop YAML here",
            type=["yml", "yaml"],
            label_visibility="collapsed",
            key=uploader_key,
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
                    cfg_loaded, ok = prepare_config(loaded)
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
                _clear_pending_and_uploader()
                _reset_widget_state()
                st.rerun()
            if btn_discard.button("Discard", use_container_width=True):
                _clear_pending_and_uploader()
                st.rerun()

    with save_col:
        st.markdown("### Save")
        st.caption("Download the current in-app configuration as a YAML file.")
        default_filename = f"schedule-config-{date.today().isoformat()}.yml"
        filename = st.text_input(
            "Filename",
            value=default_filename,
            key="config_filename",
        )
        yaml_text = yaml.safe_dump(cfg, sort_keys=False)
        saved = st.download_button(
            "Download configuration",
            data=yaml_text,
            file_name=filename or default_filename,
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
