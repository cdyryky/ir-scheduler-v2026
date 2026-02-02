from typing import Any, Callable, Optional

import os
from datetime import date, timedelta
import math
import html
import re

import streamlit as st
import yaml
import pandas as pd

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

APP_TITLE_TEXT = "SCHEDULIZER 5000"
APP_TITLE_DISPLAY_HTML = f"<strong><em>{html.escape(APP_TITLE_TEXT)}</em></strong>"


DISPLAY_COLUMNS = ROTATION_COLUMNS + ["Total Blocks"]

CONSTRAINT_CATEGORY_IDS: dict[str, set[str]] = {
    "Core Rules": {"one_place", "block_total_zero_or_full", "no_half_non_ir5", "no_half_kir"},
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
    "Track Rules": {"dr1_early_block", "ir3_late_block", "ir4_off_sicu"},
    "Special Blocks": {"holiday_block_staffing", "viva_block_staffing"},
    "Preferences": {"first_timer", "consec_full_mh", "no_sequential_year1_3"},
}


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _render_html_dataframe(
    df: pd.DataFrame,
    *,
    table_class: str,
    escape_cells: bool = True,
    format_cell: Optional[Callable[[str, Any], str]] = None,
    cell_style: Optional[Callable[[int, str, Any], str]] = None,
    row_style: Optional[Callable[[int, pd.Series], str]] = None,
) -> None:
    cols = list(df.columns)

    def _default_format(col: str, value: Any) -> str:
        if _is_na(value):
            return ""
        return str(value)

    fmt = format_cell or _default_format

    parts: list[str] = [
        f"""
<style>
table.{table_class} {{
  width: 100%;
  border-collapse: collapse;
}}
table.{table_class} th, table.{table_class} td {{
  border: 1px solid rgba(128, 128, 128, 0.35);
  padding: 0.35rem 0.5rem;
  vertical-align: top;
  font-size: 0.92rem;
}}
table.{table_class} th {{
  background: rgba(127, 127, 127, 0.10);
  font-weight: 700;
  text-align: left;
}}
</style>
<table class="{html.escape(table_class)}">
  <thead><tr>
"""
    ]

    for col in cols:
        parts.append(f"<th>{html.escape(str(col))}</th>")
    parts.append("</tr></thead><tbody>")

    for row_idx, row in df.iterrows():
        tr_style = row_style(row_idx, row) if row_style else ""
        tr_style_attr = f' style="{html.escape(tr_style)}"' if tr_style else ""
        parts.append(f"<tr{tr_style_attr}>")
        for col in cols:
            raw = row[col]
            text = fmt(str(col), raw)
            if escape_cells:
                text = html.escape(str(text))
            td_style = cell_style(row_idx, str(col), raw) if cell_style else ""
            td_style_attr = f' style="{html.escape(td_style)}"' if td_style else ""
            parts.append(f"<td{td_style_attr}>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    st.markdown("\n".join(parts), unsafe_allow_html=True)


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


def _parse_mmddyy(value: str) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None

    m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2})\s*$", text)
    if m:
        mm, dd, yy = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) != 6:
            return None
        mm, dd, yy = int(digits[:2]), int(digits[2:4]), int(digits[4:6])

    year = 2000 + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def _format_mmddyy(d: date) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year % 100:02d}"


def _calendar_start_date(cfg: dict) -> Optional[date]:
    gui = cfg.get("gui") if isinstance(cfg.get("gui"), dict) else {}
    cal = gui.get("calendar") if isinstance(gui.get("calendar"), dict) else {}
    return _parse_mmddyy(str(cal.get("start_date", "") or ""))


def _block_date_ranges(cfg: dict, block_labels: list[str]) -> dict[str, tuple[date, date]]:
    start = _calendar_start_date(cfg)
    if not start:
        return {}
    out: dict[str, tuple[date, date]] = {}
    for idx, block in enumerate(block_labels):
        b_start = start + timedelta(days=28 * idx)
        b_end = b_start + timedelta(days=27)
        out[str(block)] = (b_start, b_end)
    return out


def _block_display_labels(cfg: dict, block_labels: list[str]) -> dict[str, str]:
    ranges = _block_date_ranges(cfg, block_labels)
    out: dict[str, str] = {}
    for block in block_labels:
        r = ranges.get(str(block))
        if not r:
            out[str(block)] = str(block)
        else:
            out[str(block)] = f"{block} ({_format_mmddyy(r[0])} - {_format_mmddyy(r[1])})"
    return out


def _render_block_label_cell(container, block: str, r: Optional[tuple[date, date]]) -> None:
    safe_block = html.escape(str(block))
    if not r:
        container.markdown(f"**{safe_block}**")
        return
    safe_dates = html.escape(f"{_format_mmddyy(r[0])} - {_format_mmddyy(r[1])}")
    container.markdown(
        f"""
<div style="line-height: 1.15;">
  <div><strong>{safe_block}</strong></div>
  <div style="font-size: 0.85em; color: rgba(120, 120, 120, 0.95);">{safe_dates}</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title=APP_TITLE_TEXT, layout="wide")

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
        padding: 0.55rem 1.0rem;
        margin: 0 0 0.45rem 0;
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
        font-size: clamp(18px, 1.75vw, 28px);
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

    /* Small UI primitives */
    .pill-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin: 0.25rem 0 0.5rem 0; }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.28rem 0.55rem;
        font-size: 12px;
        line-height: 1;
        border: 1px solid rgba(127,127,127,0.25);
        background: rgba(127,127,127,0.08);
        color: var(--text-color);
        opacity: 0.92;
        white-space: nowrap;
    }
    .pill.good { border-color: rgba(34, 197, 94, 0.35); background: rgba(34, 197, 94, 0.10); }
    .pill.warn { border-color: rgba(245, 158, 11, 0.35); background: rgba(245, 158, 11, 0.10); }
    .pill.info { border-color: rgba(59, 130, 246, 0.35); background: rgba(59, 130, 246, 0.10); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-row">
        <h1 class="hero-title">{APP_TITLE_DISPLAY_HTML}</h1>
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

st.markdown('<hr style="margin: 0.35rem 0 0.45rem 0; opacity: 0.35;">', unsafe_allow_html=True)

resident_error = None
tabs = st.tabs(
    [
        "Residents",
        "Class/Year Assignments",
        "Requests",
        "Constraints",
        "Prioritization",
        "Solve",
        "Calendar",
        "Save/Load Configuration",
        "Instructions",
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
    st.markdown(
        """
<div class="pill-row">
  <span class="pill info">IR (non-KIR): 0.5 steps</span>
  <span class="pill good">KIR: whole blocks</span>
  <span class="pill good">DR (all rotations): whole blocks</span>
</div>
""",
        unsafe_allow_html=True,
    )
    rows = []
    ir_rows = []
    dr_rows = []
    for track in CLASS_TRACKS:
        row = {"Track": track}
        for rot in ROTATION_COLUMNS:
            raw = req.get(track, {}).get(rot, 0)
            try:
                value = float(raw)
            except Exception:
                value = 0.0
            row[rot] = value
        row["Total Blocks"] = sum(row[rot] for rot in ROTATION_COLUMNS)
        rows.append(row)
        (dr_rows if track.startswith("DR") else ir_rows).append(dict(row))

    rows_df = pd.DataFrame(rows, columns=["Track"] + list(ROTATION_COLUMNS))
    ir_df = pd.DataFrame(ir_rows, columns=["Track"] + list(ROTATION_COLUMNS))
    dr_df = pd.DataFrame(dr_rows, columns=["Track"] + list(ROTATION_COLUMNS))
    if not dr_df.empty:
        for col in ROTATION_COLUMNS:
            dr_df[col] = dr_df[col].fillna(0).apply(lambda v: int(round(float(v))) if not _is_na(v) else 0)
    if "class_year_editor_open" not in st.session_state:
        st.session_state["class_year_editor_open"] = False

    left_btn, _ = st.columns([1, 3])
    if left_btn.button(
        "Edit class/year requirements",
        type="primary",
        key="class_year_editor_toggle_btn",
        use_container_width=True,
    ):
        next_open = not bool(st.session_state.get("class_year_editor_open"))
        st.session_state["class_year_editor_open"] = next_open
        if next_open:
            st.session_state.pop("class_year_table_ir", None)
            st.session_state.pop("class_year_table_dr", None)
            st.session_state.pop("class_year_req_rounding_warnings", None)

    if st.session_state.get("class_year_editor_open"):
        with st.container(border=True):
            with st.form("class_year_editor_form", clear_on_submit=False):
                t_ir, t_dr = st.tabs(["IR tracks", "DR tracks"])
                with t_ir:
                    st.caption("Non-KIR rotations accept 0.5 increments. KIR is whole blocks.")
                    edited_ir_df = st.data_editor(
                        ir_df,
                        hide_index=True,
                        num_rows="fixed",
                        use_container_width=True,
                        column_config={
                            "Track": st.column_config.TextColumn(disabled=True),
                            "MH-IR": st.column_config.NumberColumn(min_value=0.0, max_value=float(num_blocks), step=0.5),
                            "MH-CT/US": st.column_config.NumberColumn(min_value=0.0, max_value=float(num_blocks), step=0.5),
                            "48X-IR": st.column_config.NumberColumn(min_value=0.0, max_value=float(num_blocks), step=0.5),
                            "48X-CT/US": st.column_config.NumberColumn(min_value=0.0, max_value=float(num_blocks), step=0.5),
                            "KIR": st.column_config.NumberColumn(min_value=0, max_value=int(num_blocks), step=1),
                        },
                        key="class_year_table_ir",
                    )
                with t_dr:
                    st.caption("All DR requirements are whole blocks.")
                    edited_dr_df = st.data_editor(
                        dr_df,
                        hide_index=True,
                        num_rows="fixed",
                        use_container_width=True,
                        column_config={
                            "Track": st.column_config.TextColumn(disabled=True),
                            "MH-IR": st.column_config.NumberColumn(min_value=0, max_value=int(num_blocks), step=1),
                            "MH-CT/US": st.column_config.NumberColumn(min_value=0, max_value=int(num_blocks), step=1),
                            "48X-IR": st.column_config.NumberColumn(min_value=0, max_value=int(num_blocks), step=1),
                            "48X-CT/US": st.column_config.NumberColumn(min_value=0, max_value=int(num_blocks), step=1),
                            "KIR": st.column_config.NumberColumn(min_value=0, max_value=int(num_blocks), step=1),
                        },
                        key="class_year_table_dr",
                    )

                edited_df = (
                    pd.concat([edited_ir_df, edited_dr_df], ignore_index=True)
                    if not edited_dr_df.empty
                    else edited_ir_df
                )
                st.caption("Edits are applied when you click Apply. Values outside the allowed increments will be rounded on apply.")
                b_apply, b_close = st.columns(2)
                apply_clicked = b_apply.form_submit_button("Apply edits", type="primary", use_container_width=True)
                apply_close_clicked = b_close.form_submit_button("Apply + close", use_container_width=True)
    else:
        edited_df = rows_df
        apply_clicked = True
        apply_close_clicked = False

    updated_req = {
        track: {rot: float(req.get(track, {}).get(rot, 0) or 0) for rot in ROTATION_COLUMNS}
        for track in CLASS_TRACKS
    }
    if bool(apply_clicked) or bool(apply_close_clicked):
        round_warnings: list[str] = []
        updated_req = {}
        rows_by_track = {row.get("Track"): row for row in edited_df.to_dict("records")}
        for track in CLASS_TRACKS:
            row = rows_by_track.get(track, {"Track": track})
            row_req: dict[str, float] = {}
            for rot in ROTATION_COLUMNS:
                raw = row.get(rot, 0) or 0
                try:
                    value = float(raw)
                except Exception:
                    value = 0.0
                if value < 0:
                    round_warnings.append(f"{track}.{rot}: {value} clamped to 0")
                    value = 0.0

                if track.startswith("DR") or rot == "KIR":
                    snapped = float(int(round(value)))
                    if not math.isclose(value, snapped, abs_tol=1e-9):
                        round_warnings.append(f"{track}.{rot}: {value} rounded to {int(snapped)}")
                    value = snapped
                else:
                    # IR track, non-KIR rotations: snap to 0.5 increments.
                    snapped = round(value * 2.0) / 2.0
                    if not math.isclose(value, snapped, abs_tol=1e-9):
                        round_warnings.append(f"{track}.{rot}: {value} rounded to {snapped:.1f}")
                    value = float(snapped)

                row_req[rot] = value
            updated_req[track] = row_req

        st.session_state["class_year_req_rounding_warnings"] = tuple(round_warnings)

    cfg["gui"]["class_year_requirements"] = updated_req
    if bool(apply_close_clicked):
        st.session_state["class_year_editor_open"] = False
        st.rerun()

    ui_round_warnings = list(st.session_state.get("class_year_req_rounding_warnings", ()))
    if ui_round_warnings:
        with st.expander("Class/year requirements were normalized with rounding", expanded=False):
            st.markdown("\n".join(f"- {w}" for w in ui_round_warnings))

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

    numeric_cols = set(ROTATION_COLUMNS + ["Total Blocks"])

    def _display_cell_style(_row: int, col: str, value: Any) -> str:
        if col not in numeric_cols:
            return ""
        if _is_na(value):
            return ""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return ""
        if abs(v) < 1e-9:
            return "background-color: #f4f4f4; color: #8a8a8a;"
        return "background-color: #e9f7ef; font-weight: 600;"

    def _display_cell_format(col: str, value: Any) -> str:
        if col in numeric_cols:
            if _is_na(value):
                return ""
            try:
                return f"{float(value):.1f}"
            except (TypeError, ValueError):
                return ""
        return "" if _is_na(value) else str(value)

    _render_html_dataframe(
        display_df,
        table_class="class-year-summary",
        cell_style=_display_cell_style,
        format_cell=_display_cell_format,
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

    rotation_display = rotation_df[["Rotation", "Available", "Required"]]

    def _req_row_style(source: pd.DataFrame) -> Callable[[int, pd.Series], str]:
        def _style(row_idx: int, _row: pd.Series) -> str:
            src = source.loc[row_idx]
            req_min = src["ReqMin"]
            req_max = src["ReqMax"]
            if pd.isna(req_min) and pd.isna(req_max):
                return ""
            available = src["Available"]
            ok = True
            if pd.notna(req_min) and available < req_min:
                ok = False
            if pd.notna(req_max) and available > req_max:
                ok = False
            color = "#e9f7ef" if ok else "#fdecea"
            return f"background-color: {color};"

        return _style

    def _rotation_cell_format(col: str, value: Any) -> str:
        if col == "Available":
            if _is_na(value):
                return ""
            try:
                return f"{float(value):.1f}"
            except (TypeError, ValueError):
                return ""
        return "" if _is_na(value) else str(value)

    _render_html_dataframe(
        rotation_display,
        table_class="rotation-availability",
        row_style=_req_row_style(rotation_df),
        format_cell=_rotation_cell_format,
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

    _render_html_dataframe(
        location_display,
        table_class="location-availability",
        row_style=_req_row_style(location_df),
        format_cell=_rotation_cell_format,
    )

with tabs[2]:
    st.subheader("Requests")
    st.caption(
        "Use Off to forbid assignments. Use On to force a specific assignment. "
        "Both are hard constraints and may make the model infeasible."
    )

    block_labels = _block_labels(cfg)
    block_ranges = _block_date_ranges(cfg, block_labels)
    ir_rows = _ir_resident_rows(cfg)
    forced_set = _forced_set(cfg, block_labels)
    blocked_set = _blocked_set(cfg, block_labels)

    if not block_labels:
        st.error("No blocks configured.")
    elif not ir_rows:
        st.info("Add IR residents first (Residents tab).")
    else:
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
                    _render_block_label_cell(cols[0], str(block), block_ranges.get(str(block)))

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
                    _render_block_label_cell(cols[0], str(block), block_ranges.get(str(block)))
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

    def _block_label_with_dates_idx(block_idx: int, num_blocks: int) -> str:
        block_labels = _block_labels(cfg)
        if not block_labels:
            block_labels = [f"B{i}" for i in range(num_blocks)]
        if block_idx < 0 or block_idx >= len(block_labels):
            return "None"
        block = str(block_labels[block_idx])
        ranges = _block_date_ranges(cfg, block_labels)
        r = ranges.get(block)
        if not r:
            return block
        return f"{block} ({_format_mmddyy(r[0])} - {_format_mmddyy(r[1])})"

    def _viva_relaxation_for_ui(num_blocks: int) -> tuple[int, str] | None:
        mode = str(st.session_state.get("mode_viva_block_staffing", modes.get("viva_block_staffing", "always")) or "").lower()
        if mode == "disabled":
            return None

        raw_params = params.get("viva_block_staffing")
        if not isinstance(raw_params, dict):
            raw_params = {}

        block = st.session_state.get("cparam_viva_block_staffing_block", raw_params.get("block", 4))
        try:
            block_idx = int(block)
        except Exception:
            block_idx = 4
        block_idx = min(max(0, block_idx), max(0, num_blocks - 1))

        raw_min_dr = st.session_state.get(
            "cparam_viva_block_staffing_min_dr_residents",
            raw_params.get("min_dr_residents", 3),
        )
        try:
            min_dr = int(raw_min_dr)
        except Exception:
            min_dr = 3
        if min_dr < 3:
            return None

        raw_choice = st.session_state.get(
            "cparam_viva_block_staffing_relaxation",
            raw_params.get("relaxation", "mh_ctus_cap"),
        )
        choice = str(raw_choice or "").strip().lower()
        if choice not in {"mh_ctus_cap", "first_timer"}:
            return None
        return block_idx, choice

    def _constraint_title_and_description(spec, spec_params: dict, num_blocks: int) -> tuple[str, str]:
        def _block_label_with_dates(block_idx: int) -> str:
            if block_idx < 0:
                return "None"
            return _block_label_with_dates_idx(block_idx, num_blocks)

        if spec.id == "one_place":
            return (
                "One rotation per resident per block",
                "Prevents a resident from being scheduled in multiple places in the same block.",
            )
        if spec.id == "blocked":
            return (
                "Honor Off requests",
                "Enforces the Off selections from the Requests tab.",
            )
        if spec.id == "forced":
            return (
                "Honor On requests",
                "Enforces the On selections from the Requests tab.",
            )
        if spec.id == "block_total_zero_or_full":
            return (
                "Block totals must be 0 or 1.0 FTE",
                "If a resident is scheduled in a block, they must total 1.0 FTE; otherwise 0.",
            )
        if spec.id == "no_half_non_ir5":
            return (
                "No half-block assignments for DR",
                "DR residents must take full 1.0 FTE rotations within a block.",
            )
        if spec.id == "no_half_kir":
            return (
                "No half assignments on KIR",
                "Prevents 0.5-block KIR assignments (KIR must be 0 or 1.0 FTE).",
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
                f"Requires {op_text} {target_fte}.0 FTE of 48X-IR each block.",
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
                f"Requires {op_text} {target_fte}.0 FTE of 48X-CT/US each block.",
            )
        if spec.id == "mh_total_minmax":
            min_fte = spec_params.get("min_fte", 3)
            max_fte = spec_params.get("max_fte", 4)
            return (
                "MH total coverage per block",
                f"Combined MH-IR + MH-CT/US must be between {min_fte}.0 and {max_fte}.0 FTE per block.",
            )
        if spec.id == "mh_ctus_cap":
            max_fte = spec_params.get("max_fte", 1)
            return (
                "MH-CT/US cap per block",
                f"Limits MH-CT/US to at most {max_fte}.0 FTE per block.",
            )
        if spec.id == "kir_cap":
            max_fte = spec_params.get("max_fte", 2)
            return (
                "KIR cap per block",
                f"Limits KIR to at most {max_fte}.0 FTE per block.",
            )
        if spec.id == "track_requirements":
            return (
                "Per-class rotation totals",
                "Uses the Class/Year Assignments tab to enforce total blocks per resident (IR/DR class).",
            )
        if spec.id == "ir5_mh_min_per_block":
            return (
                "IR5 MH-IR minimum per block",
                "Ensures at least 1.0 FTE of IR5 MH-IR coverage per block.",
            )
        if spec.id == "ir4_plus_mh_cap":
            ir_min_year = spec_params.get("ir_min_year", 4)
            max_fte = spec_params.get("max_fte", 2)
            return (
                "Limit senior residents on MH-IR per block",
                f"'Senior' means IR{ir_min_year}+; cap is {max_fte}.0 FTE per block.",
            )
        if spec.id == "ir4_off_sicu":
            return (
                "IR4 off for SICU",
                "During the first N blocks (N = number of IR4 residents), exactly one IR4 is off per block, rotating so each IR4 is off once.",
            )
        if spec.id == "dr1_early_block":
            first_n = spec_params.get("first_n_blocks", 4)
            return (
                "Keep DR1 off MH-IR early",
                f"DR1 cannot do MH-IR in the first {first_n} block(s).",
            )
        if spec.id == "ir3_late_block":
            after_block = spec_params.get("after_block", 7)
            rots = spec_params.get("rotations", ["MH-IR", "48X-IR"])
            if not isinstance(rots, list):
                rots = ["MH-IR", "48X-IR"]
            rots = [str(r) for r in rots if str(r)]
            rots_text = ", ".join(rots) if rots else "specified rotations"
            if isinstance(after_block, int) and after_block >= num_blocks:
                return (
                    "IR-3 core studying",
                    f"No restrictions (Block N set to None).",
                )
            after_label = f"B{after_block}" if isinstance(after_block, int) else "the selected block"
            return (
                "IR-3 core studying",
                f"IR-3s not assigned to {rots_text} after block {after_label} (block {after_label} allowed).",
            )
        if spec.id == "holiday_block_staffing":
            try:
                block_idx = int(spec_params.get("block", 6) or 0)
            except Exception:
                block_idx = 6
            try:
                min_residents = int(spec_params.get("min_residents", 4) or 0)
            except Exception:
                min_residents = 4
            block_label = _block_label_with_dates(block_idx)
            return (
                "Holiday block staffing",
                f"Ensure block {block_label} MH (IR + CT/US) has at least {min_residents} resident(s).",
            )
        if spec.id == "viva_block_staffing":
            try:
                block_idx = int(spec_params.get("block", 4) or 0)
            except Exception:
                block_idx = 4
            try:
                min_residents = int(spec_params.get("min_residents", 4) or 0)
            except Exception:
                min_residents = 4
            try:
                min_dr = int(spec_params.get("min_dr_residents", 3) or 0)
            except Exception:
                min_dr = 3
            block_label = _block_label_with_dates(block_idx)
            return (
                "VIVA block staffing",
                f"Ensure block {block_label} MH (IR + CT/US) has at least {min_residents} resident(s), {min_dr} of which are DR.",
            )
        if spec.id == "first_timer":
            return (
                "First-timer MH-IR limit",
                "Limits first-time MH-IR residents (DR1/IR1) to 1 per block.",
            )
        if spec.id == "consec_full_mh":
            window_blocks = spec_params.get("window_blocks", spec_params.get("max_consecutive", 4))
            try:
                window_blocks = int(window_blocks)
            except Exception:
                window_blocks = 4
            window_blocks = min(max(1, window_blocks), max(1, num_blocks))

            raw_min_off = spec_params.get("min_off_mh_fte")
            if raw_min_off is None:
                raw_min_off = 1.5
            try:
                min_off = float(raw_min_off)
            except Exception:
                min_off = 1.5
            min_off = min(max(0.0, min_off), float(window_blocks))
            min_off = round(min_off * 2) / 2.0

            return (
                "Minimum time off MH (rolling window)",
                f"All residents must have at least {min_off:.1f} FTE off MH rotations in every {window_blocks}-block window.",
            )
        if spec.id == "no_sequential_year1_3":
            return (
                "No back-to-back blocks for years 1–3",
                "Prevents residents in years 1–3 (DR1–3, IR1–3) from being scheduled in consecutive blocks.",
            )

        # Fallback
        return (spec.label, "")

    def _render_spec_mode(spec, title: str, *, allow_try: bool):
        preference_default_on = {"first_timer", "consec_full_mh", "no_sequential_year1_3"}
        mode_display = {
            "always": "On (hard)",
            "if_able": "Try (soft)",
            "disabled": "Off",
        }
        if allow_try:
            options = ["always", "if_able", "disabled"]
        else:
            options = ["always", "disabled"]

        if spec.id in {"holiday_block_staffing", "viva_block_staffing"} and spec.id not in modes:
            default_mode = "always"
        elif spec.id in preference_default_on and spec.id not in modes:
            default_mode = "always"
        else:
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
        return selection, st.empty()

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
            if spec.id == "mh_ctus_cap":
                viva = _viva_relaxation_for_ui(num_blocks)
                mode = str(st.session_state.get("mode_mh_ctus_cap", modes.get("mh_ctus_cap", "always")) or "").lower()
                if viva and viva[1] == "mh_ctus_cap" and mode != "disabled":
                    block_label = _block_label_with_dates_idx(viva[0], num_blocks)
                    st.info(
                        f"Note: MH-CT/US cap is relaxed during block {block_label} due to VIVA block staffing enforcement."
                    )

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

        elif spec.id == "first_timer":
            viva = _viva_relaxation_for_ui(num_blocks)
            mode = str(st.session_state.get("mode_first_timer", modes.get("first_timer", "always")) or "").lower()
            if viva and viva[1] == "first_timer" and mode != "disabled":
                block_label = _block_label_with_dates_idx(viva[0], num_blocks)
                st.info(
                    f"Note: First-timer MH-IR limit is relaxed during block {block_label} due to VIVA block staffing enforcement."
                )

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

        elif spec.id == "holiday_block_staffing":
            p = _params_for(spec.id)
            block_labels = _block_labels(cfg) or [f"B{i}" for i in range(num_blocks)]
            block_ranges = _block_date_ranges(cfg, block_labels)
            options = list(range(len(block_labels)))

            block_idx = p.get("block", 6)
            if not isinstance(block_idx, int):
                block_idx = 6
            block_idx = min(max(0, block_idx), max(0, len(block_labels) - 1))

            def _fmt_block(i: int) -> str:
                blk = str(block_labels[i])
                r = block_ranges.get(blk)
                if not r:
                    return blk
                return f"{blk} ({_format_mmddyy(r[0])} - {_format_mmddyy(r[1])})"

            sel_block = st.selectbox(
                "Block",
                options=options,
                index=options.index(block_idx) if block_idx in options else 0,
                format_func=_fmt_block,
                key="cparam_holiday_block_staffing_block",
            )
            p["block"] = int(sel_block)

            min_residents = p.get("min_residents", 4)
            if not isinstance(min_residents, int) or min_residents < 0:
                min_residents = 4
            p["min_residents"] = int(
                st.number_input(
                    "Min residents at MH (MH-IR + MH-CT/US)",
                    min_value=0,
                    max_value=50,
                    value=int(min_residents),
                    step=1,
                    key="cparam_holiday_block_staffing_min_residents",
                )
            )

        elif spec.id == "viva_block_staffing":
            p = _params_for(spec.id)
            block_labels = _block_labels(cfg) or [f"B{i}" for i in range(num_blocks)]
            block_ranges = _block_date_ranges(cfg, block_labels)
            options = list(range(len(block_labels)))

            block_idx = p.get("block", 4)
            if not isinstance(block_idx, int):
                block_idx = 4
            block_idx = min(max(0, block_idx), max(0, len(block_labels) - 1))

            def _fmt_block(i: int) -> str:
                blk = str(block_labels[i])
                r = block_ranges.get(blk)
                if not r:
                    return blk
                return f"{blk} ({_format_mmddyy(r[0])} - {_format_mmddyy(r[1])})"

            sel_block = st.selectbox(
                "Block",
                options=options,
                index=options.index(block_idx) if block_idx in options else 0,
                format_func=_fmt_block,
                key="cparam_viva_block_staffing_block",
            )
            p["block"] = int(sel_block)

            min_residents = p.get("min_residents", 4)
            if not isinstance(min_residents, int) or min_residents < 0:
                min_residents = 4
            p["min_residents"] = int(
                st.number_input(
                    "Min residents at MH (MH-IR + MH-CT/US)",
                    min_value=0,
                    max_value=50,
                    value=int(min_residents),
                    step=1,
                    key="cparam_viva_block_staffing_min_residents",
                )
            )

            min_dr = p.get("min_dr_residents", 3)
            if not isinstance(min_dr, int) or min_dr < 0:
                min_dr = 3
            min_dr_val = int(
                st.number_input(
                    "Min DR residents at MH (MH-IR + MH-CT/US)",
                    min_value=0,
                    max_value=50,
                    value=int(min_dr),
                    step=1,
                    key="cparam_viva_block_staffing_min_dr_residents",
                )
            )
            p["min_dr_residents"] = min_dr_val

            viva_mode = str(st.session_state.get("mode_viva_block_staffing", "") or "").strip().lower()
            if viva_mode != "disabled" and min_dr_val >= 3:
                current = str(p.get("relaxation", "mh_ctus_cap") or "").strip().lower()
                relax_options = ["mh_ctus_cap", "first_timer"]
                if current not in relax_options:
                    current = "mh_ctus_cap"

                selection = st.radio(
                    "In order to have 3+ DR residents at MH, you must relax a constraint for this VIVA block:",
                    options=relax_options,
                    index=relax_options.index(current),
                    format_func={
                        "mh_ctus_cap": "Allow up to 2 residents on MH-CT/US for this block (relaxes MH-CT/US cap per block)",
                        "first_timer": "Allow two IR-1 residents on MH-IR for this block (relaxes First-timer MH-IR limit)",
                    }.get,
                    key="cparam_viva_block_staffing_relaxation",
                )
                p["relaxation"] = str(selection)

        elif spec.id == "ir3_late_block":
            p = _params_for(spec.id)
            after_block = p.get("after_block", 7)
            if not isinstance(after_block, int) or after_block < 0:
                after_block = 7

            block_labels = _block_labels(cfg)
            if not block_labels:
                block_labels = [f"B{i}" for i in range(num_blocks)]

            options = list(range(0, len(block_labels) + 1))
            none_label = "None (no restriction)"

            def _fmt_after(idx: int) -> str:
                return block_labels[idx] if idx < len(block_labels) else none_label

            after_block = min(after_block, len(block_labels))
            after_key = "cparam_ir3_late_block_after_block"
            stored_after = st.session_state.get(after_key, after_block)
            if not isinstance(stored_after, int):
                stored_after = after_block
            stored_after = min(max(0, stored_after), len(block_labels))
            st.session_state[after_key] = stored_after

            sel = st.selectbox(
                "Block N (last allowed)",
                options=options,
                index=options.index(stored_after) if stored_after in options else options.index(after_block),
                format_func=_fmt_after,
                help="IR-3s may still be assigned in block N. Restrictions apply starting at block N+1.",
                key=after_key,
            )
            p["after_block"] = int(sel)

            if "rotations" in p:
                raw_rots = p.get("rotations", [])
            else:
                raw_rots = ["MH-IR", "48X-IR"]
            if not isinstance(raw_rots, list):
                raw_rots = ["MH-IR", "48X-IR"]
            current = [str(r) for r in raw_rots if str(r) in ROTATION_COLUMNS]

            rot_key = "cparam_ir3_late_block_rotations"
            stored_rots = st.session_state.get(rot_key, current)
            if not isinstance(stored_rots, list):
                stored_rots = current
            stored_rots = [str(r) for r in stored_rots if str(r) in ROTATION_COLUMNS]
            if rot_key not in st.session_state:
                st.session_state[rot_key] = stored_rots

            p["rotations"] = st.multiselect(
                "Rotations blocked after N",
                options=ROTATION_COLUMNS,
                key=rot_key,
                help="Defaults to MH-IR and 48X-IR.",
            )

        elif spec.id == "consec_full_mh":
            p = _params_for(spec.id)
            raw_window = p.get("window_blocks", p.get("max_consecutive", 4))
            if not isinstance(raw_window, int) or raw_window < 1:
                raw_window = 4
            raw_window = min(max(1, raw_window), max(1, num_blocks))

            window_key = "cparam_consec_full_mh_window_blocks"
            stored_window = st.session_state.get(window_key, raw_window)
            if not isinstance(stored_window, int) or stored_window < 1:
                stored_window = raw_window
            stored_window = min(max(1, stored_window), max(1, num_blocks))
            st.session_state[window_key] = stored_window

            window_options = list(range(1, max(1, num_blocks) + 1))
            window_sel = st.selectbox(
                "Window size (blocks)",
                options=window_options,
                index=window_options.index(stored_window) if stored_window in window_options else window_options.index(raw_window),
                key=window_key,
            )
            window_blocks = int(window_sel)

            raw_min_off = p.get("min_off_mh_fte", None)
            if raw_min_off is None:
                raw_min_off = 1.5
            try:
                min_off = float(raw_min_off)
            except Exception:
                min_off = 1.5
            min_off = min(max(0.0, min_off), float(window_blocks))
            min_off = round(min_off * 2) / 2.0

            off_options = [i / 2.0 for i in range(0, (2 * window_blocks) + 1)]
            if min_off not in off_options:
                min_off = max(0.0, min(float(window_blocks), min_off))
                min_off = round(min_off * 2) / 2.0
            off_key = "cparam_consec_full_mh_min_off_mh_fte"
            stored_off = st.session_state.get(off_key, min_off)
            try:
                stored_off = float(stored_off)
            except Exception:
                stored_off = min_off
            if stored_off not in off_options:
                stored_off = min_off
            st.session_state[off_key] = stored_off

            off_sel = st.selectbox(
                "Min FTE off MH in each window",
                options=off_options,
                index=off_options.index(stored_off),
                key=off_key,
                help="Counts all time not on MH rotations (unassigned + 48X/KIR), not 'vacation'.",
                format_func=lambda v: f"{float(v):.1f}",
            )

            p["window_blocks"] = int(window_blocks)
            p["min_off_mh_fte"] = float(off_sel)
            if "max_consecutive" in p:
                del p["max_consecutive"]

    categories = CONSTRAINT_CATEGORY_IDS

    tab_names = list(categories.keys())
    sub_tabs = st.tabs(tab_names)

    num_blocks = _num_blocks(cfg)
    for tab_name, tab in zip(tab_names, sub_tabs):
        with tab:
            ids = categories.get(tab_name, set())
            allow_try = tab_name != "Core Rules"
            for spec in CONSTRAINT_SPECS:
                if spec.id not in ids:
                    continue

                spec_params = _params_for(spec.id)
                title, description = _constraint_title_and_description(spec, spec_params, num_blocks=num_blocks)
                spec_allow_try = allow_try or spec.id == "no_half_kir"
                _selection, desc_slot = _render_spec_mode(spec, title=title, allow_try=spec_allow_try)

                if tab_name in {"Coverage & Caps", "Track Rules", "Special Blocks", "Preferences"}:
                    _render_spec_params(spec, num_blocks=num_blocks)
                    spec_params = _params_for(spec.id)
                    _title2, description = _constraint_title_and_description(spec, spec_params, num_blocks=num_blocks)
                    if description:
                        desc_slot.caption(description)
                    st.divider()
                else:
                    if description:
                        desc_slot.caption(description)

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
        st.info("No constraints currently set to Try (soft).")
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
            if getattr(schedule_input, "warnings", None):
                st.warning("Input normalized with rounding:")
                st.markdown("\n".join(f"- {w}" for w in schedule_input.warnings))
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
        # Solver stores objective counts under each constraint's objective_key
        # (e.g., "consec_excess"), while weights are keyed separately (e.g., "consec").
        label_by_objective_key = {
            "consec_excess": "Consecutive full MH blocks",
            "first_timer_excess": "First-timer assignments",
            "adj": "Adjacent assignments (Y1–Y3)",
        }
        weight_by_objective_key = {
            "consec_excess": int(weights.get("consec", 0)),
            "first_timer_excess": int(weights.get("first_timer", 0)),
            "adj": int(weights.get("adj", 0)),
        }
        rows = []
        total = 0
        for objective_key, label in label_by_objective_key.items():
            count = int(sol_objective.get(objective_key, 0) or 0)
            weight = int(weight_by_objective_key.get(objective_key, 0))
            weighted = count * weight
            if count:
                rows.append({"Penalty": label, "Count": count, "Weight": weight, "Weighted": weighted})
            total += weighted

        extra_keys = sorted(
            [
                k
                for k in sol_objective.keys()
                if k not in label_by_objective_key
                and k != "try_relaxations"
                and not str(k).startswith("relax_")
            ],
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

    def _render_html_table(df: pd.DataFrame, table_class: str = "ir-table", escape_cells: bool = True) -> None:
        safe_df = df.copy()
        if escape_cells:
            for col in safe_df.columns:
                safe_df[col] = safe_df[col].apply(lambda v: html.escape(str(v)) if v is not None else "")
        table_html = safe_df.to_html(index=False, escape=False, classes=table_class)
        st.markdown(
            f"""
<style>
table.{table_class} {{
  width: 100%;
  border-collapse: collapse;
}}
table.{table_class} th, table.{table_class} td {{
  border: 1px solid rgba(128, 128, 128, 0.35);
  padding: 0.35rem 0.5rem;
  vertical-align: top;
  font-size: 0.92rem;
}}
table.{table_class} th {{
  background: rgba(127, 127, 127, 0.10);
  font-weight: 700;
}}
</style>
{table_html}
""",
            unsafe_allow_html=True,
        )

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

    run = st.button("Solve", type="primary", use_container_width=True, key="solve_btn")

    if run:
        st.session_state.pop("solve_result", None)
        st.session_state.pop("solve_csv", None)
        st.session_state.pop("solve_input", None)
        try:
            schedule_input = load_schedule_input_from_data(cfg)
        except Exception as exc:
            st.error(f"Invalid configuration: {exc}")
        else:
            if getattr(schedule_input, "warnings", None):
                st.warning("Input normalized with rounding:")
                st.markdown("\n".join(f"- {w}" for w in schedule_input.warnings))
            with st.spinner("Solving..."):
                result = solve_schedule(schedule_input)
            st.session_state["solve_result"] = result
            st.session_state["solve_csv"] = result_to_csv(result)
            st.session_state["solve_input"] = schedule_input

    result = st.session_state.get("solve_result")
    if result is None:
        st.info("Click Solve to run the scheduler.")
    else:
        if getattr(result, "warnings", None):
            st.warning("Input normalized with rounding:")
            st.markdown("\n".join(f"- {w}" for w in result.warnings))
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
                pick_col, dl_col = st.columns([2.3, 2.4], gap="small")
                with pick_col:
                    idx = st.selectbox(
                        "Solution",
                        options=list(range(len(result.solutions))),
                        format_func=lambda i: f"Solution {i}",
                        key="solution_select",
                        label_visibility="collapsed",
                    )
                sol = result.solutions[int(idx)]
                schedule_input = st.session_state.get("solve_input")

                csv_text = st.session_state.get("solve_csv", "")
                blocks = _block_labels(cfg)
                if not blocks:
                    blocks = list(next(iter(sol.assignments.values())).keys()) if sol.assignments else []
                block_display = _block_display_labels(cfg, blocks)
                display_blocks = [block_display.get(b, b) for b in blocks]

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
                        row[block_display.get(block, block)] = "\n".join(sorted(assigned, key=lambda s: s.casefold()))
                    rot_rows.append(row)

                rot_df = pd.DataFrame(rot_rows, columns=["Rotation"] + display_blocks)
                table_csv_text = rot_df.to_csv(index=False)

                with dl_col:
                    b1, b2 = st.columns(2, gap="small")
                    b1.download_button(
                        "Download CSV (Table)",
                        data=table_csv_text,
                        file_name=f"schedule-table-solution-{int(idx)}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    b2.download_button(
                        "Download CSV (Long)",
                        data=csv_text or "",
                        file_name="schedule-output.csv",
                        mime="text/csv",
                        use_container_width=True,
                        disabled=not bool(csv_text),
                    )
                if schedule_input is not None:
                    obj_df, obj_total = _objective_breakdown(sol.objective or {}, schedule_input.weights)
                    st.markdown(f"**Weighted objective score (lower is better): {obj_total}**")
                    relaxed_ids = sorted(
                        [k[len("relax_") :] for k in (sol.objective or {}).keys() if str(k).startswith("relax_")],
                        key=lambda s: str(s).casefold(),
                    )
                    if not obj_df.empty:
                        st.dataframe(obj_df, use_container_width=True, hide_index=True)
                    else:
                        if relaxed_ids:
                            st.caption("No weighted soft penalties in this solution (some Try-mode hard constraints were relaxed below).")
                        else:
                            st.caption("No soft penalties in this solution.")
                    if relaxed_ids:
                        label_by_id = {spec.id: spec.label for spec in CONSTRAINT_SPECS}
                        st.warning(
                            f"Relaxed {len(relaxed_ids)} Try-mode hard constraint(s) to find a feasible schedule."
                        )
                        st.dataframe(
                            pd.DataFrame(
                                [{"Constraint": label_by_id.get(cid, cid), "Id": cid} for cid in relaxed_ids],
                                columns=["Constraint", "Id"],
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                st.markdown("**Assignments by Rotation**")
                html_df = rot_df.copy()
                for col in html_df.columns:
                    html_df[col] = html_df[col].apply(
                        lambda v: "<br>".join(html.escape(part) for part in str(v).split("\n"))
                        if isinstance(v, str) and v
                        else ""
                    )
                _render_html_table(html_df, table_class="assignments-table", escape_cells=False)

                if schedule_input is not None:
                    ir_totals_df = _ir_totals_table(sol, schedule_input)
                    if not ir_totals_df.empty:
                        _viridis = [
                            "#440154",
                            "#482878",
                            "#3e4989",
                            "#31688e",
                            "#26828e",
                            "#1f9e89",
                            "#35b779",
                            "#6ece58",
                            "#b5de2b",
                            "#fde725",
                        ]
                        _greys = [
                            "#000000",
                            "#1f1f1f",
                            "#3b3b3b",
                            "#585858",
                            "#767676",
                            "#969696",
                            "#b8b8b8",
                            "#d9d9d9",
                            "#f0f0f0",
                            "#ffffff",
                        ]

                        def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
                            h = hex_color.lstrip("#")
                            r = int(h[0:2], 16) / 255.0
                            g = int(h[2:4], 16) / 255.0
                            b = int(h[4:6], 16) / 255.0
                            return r, g, b

                        def _rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
                            r, g, b = rgb
                            return "#{:02x}{:02x}{:02x}".format(
                                int(max(0, min(1, r)) * 255),
                                int(max(0, min(1, g)) * 255),
                                int(max(0, min(1, b)) * 255),
                            )

                        def _lerp(a: float, b: float, t: float) -> float:
                            return a + (b - a) * t

                        def _palette_hex(palette: list[str], t: float) -> str:
                            if not palette:
                                return "#ffffff"
                            if len(palette) == 1:
                                return palette[0]
                            t = max(0.0, min(1.0, float(t)))
                            pos = t * (len(palette) - 1)
                            idx = int(pos)
                            if idx >= len(palette) - 1:
                                return palette[-1]
                            frac = pos - idx
                            r1, g1, b1 = _hex_to_rgb01(palette[idx])
                            r2, g2, b2 = _hex_to_rgb01(palette[idx + 1])
                            return _rgb01_to_hex((_lerp(r1, r2, frac), _lerp(g1, g2, frac), _lerp(b1, b2, frac)))

                        numeric_cols = ROTATION_COLUMNS + ["Total"]
                        rot_cols = ROTATION_COLUMNS
                        total_col = ["Total"]

                        rot_values = ir_totals_df[rot_cols].to_numpy()
                        rot_nonzero = rot_values[rot_values > 0]
                        rot_vmin = float(rot_nonzero.min()) if rot_nonzero.size else 0.0
                        rot_vmax = float(rot_nonzero.max()) if rot_nonzero.size else 0.0

                        total_values = ir_totals_df[total_col].to_numpy()
                        total_nonzero = total_values[total_values > 0]
                        total_vmin = float(total_nonzero.min()) if total_nonzero.size else 0.0
                        total_vmax = float(total_nonzero.max()) if total_nonzero.size else 0.0

                        def _bg_from(value: int | float | None, vmin: float, vmax: float, palette: list[str]) -> str:
                            if value is None or pd.isna(value):
                                return ""
                            v = float(value)
                            if abs(v) < 1e-9:
                                return ""
                            t = 0.5 if math.isclose(vmin, vmax) else (v - vmin) / (vmax - vmin)
                            t = max(0.0, min(1.0, float(t)))
                            hex_color = _palette_hex(palette, t)
                            r, g, b = _hex_to_rgb01(hex_color)
                            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            text = "#111111" if luminance > 0.55 else "#ffffff"
                            return f"background-color: {hex_color}; color: {text}; font-weight: 600;"

                        def _bg_rot(value: int | float | None) -> str:
                            return _bg_from(value, rot_vmin, rot_vmax, _viridis)

                        def _bg_total(value: int | float | None) -> str:
                            return _bg_from(value, total_vmin, total_vmax, _greys)

                        def _fmt_cell(value: int | float | None) -> str:
                            if value is None or pd.isna(value):
                                return ""
                            v = float(value)
                            if abs(v) < 1e-9:
                                return ""
                            return _fmt_fte(v)

                        st.markdown("**IR resident rotation totals**")

                        def _ir_totals_cell_style(_row: int, col: str, value: Any) -> str:
                            if col in rot_cols:
                                return _bg_rot(value)
                            if col in total_col:
                                return _bg_total(value)
                            return ""

                        def _ir_totals_cell_format(col: str, value: Any) -> str:
                            if col in numeric_cols:
                                return _fmt_cell(value)
                            return "" if _is_na(value) else str(value)

                        _render_html_dataframe(
                            ir_totals_df,
                            table_class="ir-totals",
                            cell_style=_ir_totals_cell_style,
                            format_cell=_ir_totals_cell_format,
                        )

with tabs[6]:
    st.subheader("Calendar")
    st.caption("Choose the start date for block B0. Each block is 4 weeks (28 days).")

    gui = cfg.get("gui") if isinstance(cfg.get("gui"), dict) else {}
    cal = gui.get("calendar") if isinstance(gui.get("calendar"), dict) else {}
    cfg.setdefault("gui", {}).setdefault("calendar", {})

    default_start = _calendar_start_date(cfg) or date(2026, 6, 29)
    if "calendar_start_date_picker" not in st.session_state:
        st.session_state["calendar_start_date_picker"] = default_start

    st.date_input(
        "Pick start date",
        key="calendar_start_date_picker",
    )

    picked = st.session_state.get("calendar_start_date_picker")
    if not isinstance(picked, date):
        picked = default_start
        st.session_state["calendar_start_date_picker"] = picked

    cfg["gui"]["calendar"]["start_date"] = _format_mmddyy(picked)

    block_labels = _block_labels(cfg)
    block_ranges = _block_date_ranges(cfg, block_labels)
    start_row = {"Date": "Start"}
    end_row = {"Date": "End"}
    for blk in block_labels:
        r = block_ranges.get(blk)
        if not r:
            start_row[blk] = ""
            end_row[blk] = ""
        else:
            start_row[blk] = _format_mmddyy(r[0])
            end_row[blk] = _format_mmddyy(r[1])
    st.dataframe(
        pd.DataFrame([start_row, end_row], columns=["Date"] + block_labels),
        use_container_width=True,
        hide_index=True,
    )

with tabs[7]:
    st.subheader("Save/Load Configuration")

    def _reset_widget_state() -> None:
        prefixes = (
            "ir_",
            "dr_",
            "requests_",
            "calendar_",
            "cparam_",
            "mode_",
            "prio_",
            "class_year_editor_",
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

with tabs[8]:
    st.subheader("Instructions")
    st.caption("A quick guide for using the scheduler GUI (no YAML editing required).")

    with st.expander("Quick start (recommended order)", expanded=True):
        st.markdown(
            """
            1. **Residents**: enter IR names and DR counts.
            2. **Class/Year Assignments**: set per-class rotation totals (make totals whole numbers).
            3. **Calendar**: pick the start date for `B0` so blocks show dates throughout the app.
            4. **Requests**: add resident-specific Off/On requests.
            5. **Constraints**: choose which rules are hard (**Always**), soft (**Try**), or disabled.
            6. **Prioritization**: order Try constraints to tell the solver what to prefer.
            7. **Solve**: run the solver and download a CSV.
            8. **Save/Load Configuration**: download a YAML snapshot to preserve your residents + constraints (otherwise they reset on page reload).
            """
        )

    with st.expander("Key concepts (blocks, rotations, FTE)", expanded=False):
        st.markdown(
            """
            - **Blocks**: scheduling periods labeled `B0`, `B1`, ... (each block is 4 weeks / 28 days).
            - **Rotations**: the columns you can assign within a block (e.g., `MH-IR`, `48X-CT/US`, `KIR`).
            - **FTE values**: most rotations allow `0`, `0.5`, or `1.0` per block (some are full-block only).
            - **Requests vs. Constraints**:
              - **Requests** are resident-specific (“this resident must/must not do X in block Y”).
              - **Constraints** are global rules (coverage, caps, sequencing rules, preferences, etc.).
            - **Session state**: the app keeps your configuration while the page is open; download YAML to save it permanently.
            """
        )

    with st.expander("Residents tab", expanded=False):
        st.markdown(
            """
            - Enter **two IR names per track** (IR1–IR5). These become the residents you can schedule.
            - Set **DR counts** per DR track. DRs are included in coverage/caps and class/year totals.
            """
        )

    with st.expander("Class/Year Assignments tab", expanded=False):
        st.markdown(
            """
            - Use **Edit class/year requirements** to set how many blocks each class/track should do for each rotation.
            - **Totals must be whole numbers** before solving (the app warns if they are not).
            - The summary tables help you sanity-check:
              - **Per-class totals** (what each resident is expected to complete)
              - **Availability vs. required coverage/caps** (whether totals can satisfy your coverage rules)
            - If you see an error (**red**), fix it before solving (many other tabs depend on valid residents and totals).
            """
        )

    with st.expander("Class/Year Assignments: common pitfalls", expanded=False):
        st.markdown(
            """
            - Non-integer totals (e.g., `11.5` blocks) will block feasibility.
            - `KIR` totals must be whole numbers (no `0.5` KIR).
            - If availability is below minimums (or above caps), the solver may be infeasible unless those constraints are set to Try/Off.
            """
        )

    with st.expander("Requests tab (Off / On)", expanded=False):
        st.markdown(
            """
            - Pick a resident on the left.
            - **Off**: check one or more rotations in a block to forbid them for that resident/block.
              - Use **All** / **None** to quickly set a full row.
              - If a rotation is forced **On** for a block, the matching **Off** checkbox is disabled.
            - **On**: select *one* rotation per block to force that assignment (or blank to clear).
            - Requests are **hard** rules; too many On/Off selections can make the model infeasible.
            """
        )

    with st.expander("Constraints tab (Always / Try / Off)", expanded=False):
        st.markdown(
            """
            Constraints are grouped into subtabs (Core Rules, Requests, Coverage & Caps, etc.).

            - **Always**: hard requirement; must be satisfied.
            - **Try**: soft requirement; the solver may relax it to regain feasibility and/or improve other priorities.
            - **Off**: not enforced.

            Some constraints have parameters (caps, block ranges, specific rotations) that appear directly under the constraint.

            A short explanation of each constraint is available at the bottom of this Instructions tab.
            """
        )

    with st.expander("Prioritization tab (Try constraint ordering)", expanded=False):
        st.markdown(
            """
            - This tab only lists constraints currently set to **Try**.
            - Move items **Up/Down** to tell the solver which Try constraints matter most when tradeoffs are necessary.
            - Lower objective score is better; the Solve tab shows a breakdown per solution.
            """
        )

    with st.expander("Solve tab (run + interpret results)", expanded=False):
        st.markdown(
            """
            - Set **Number of solutions** to explore multiple feasible options.
            - Click **Solve** to run the scheduler using the current in-app configuration.
            - If infeasible, the app shows **conflicting constraints** and **fast suggestions**.
            - If feasible, pick a solution and download:
              - **CSV (Table)**: rotation × block table with assigned residents
              - **CSV (Long)**: long-format schedule (`schedule-output.csv`)
            """
        )

    with st.expander("Solve: troubleshooting infeasibility", expanded=False):
        st.markdown(
            """
            Common causes:
            - Too many **On** requests, or On requests conflicting with Off selections.
            - Coverage minimums that exceed available residents/FTE.
            - Caps that are too tight (e.g., `MH-CT/US` cap + required MH total).

            Quick fixes:
            - Set the hardest constraints to **Try**, then re-solve; the Solve tab will list which Try-mode constraints were relaxed to reach feasibility.
            - Reduce strict caps/minimums, then re-tighten once you have a baseline feasible schedule.
            """
        )

    with st.expander("Calendar tab", expanded=False):
        st.markdown(
            """
            - Pick the start date for block `B0`.
            - The app treats each block as 28 days and shows a start/end table.
            - This is used to display “block + date range” labels throughout the UI.
            """
        )

    with st.expander("Save/Load Configuration tab", expanded=False):
        st.markdown(
            """
            - **Load**: drag & drop a YAML file to preview it, then click **Apply** to replace the current session configuration.
            - **Save**: download the current in-app configuration as YAML (a snapshot you can reload later).
            - **Current YAML** shows the exact configuration the solver will use.
            """
        )

    st.markdown("### Constraints reference")
    st.caption("Expand a constraint to see its mode (Always/Try/Off) and a short description.")

    gui_constraints = cfg.get("gui", {}).get("constraints", {})
    modes = gui_constraints.get("modes", {}) if isinstance(gui_constraints, dict) else {}
    params = gui_constraints.get("params", {}) if isinstance(gui_constraints, dict) else {}
    if not isinstance(modes, dict):
        modes = {}
    if not isinstance(params, dict):
        params = {}

    num_blocks = _num_blocks(cfg)
    tab_names = list(CONSTRAINT_CATEGORY_IDS.keys())
    ref_tabs = st.tabs(tab_names)

    def _mode_text(spec) -> str:
        raw = modes.get(spec.id, "if_able" if getattr(spec, "softenable", False) else "always")
        m = str(raw or "").lower()
        return {"always": "Always (hard)", "if_able": "Try (soft)", "disabled": "Off (disabled)"}.get(m, str(raw))

    for tab_name, tab in zip(tab_names, ref_tabs):
        with tab:
            ids = CONSTRAINT_CATEGORY_IDS.get(tab_name, set())
            for spec in CONSTRAINT_SPECS:
                if spec.id not in ids:
                    continue
                spec_params = params.get(spec.id, {})
                if not isinstance(spec_params, dict):
                    spec_params = {}
                title, description = _constraint_title_and_description(spec, spec_params, num_blocks=num_blocks)
                with st.expander(title, expanded=False):
                    st.markdown(f"**Id**: `{spec.id}`")
                    st.markdown(f"**Mode**: {_mode_text(spec)}")
                    if description:
                        st.markdown(description)
                    if spec.id == "viva_block_staffing":
                        st.markdown(
                            """
                            **VIVA “relaxations” (when `min_dr_residents` ≥ 3)**

                            The VIVA block staffing rule can optionally loosen *one other rule for the VIVA block only* to help
                            the solver place enough DRs on MH that block:

                            - `mh_ctus_cap`: raises the MH-CT/US cap to **2.0 FTE** *in the VIVA block* (instead of the usual cap).
                            - `first_timer`: allows **2 first-timers** *in the VIVA block* (instead of 1).

                            This is a targeted exception (not a Try-mode relaxation): the schedule still satisfies the affected
                            constraint, but with a looser per-block limit for the VIVA block.
                            """
                        )
