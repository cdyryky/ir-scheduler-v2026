from __future__ import annotations

from typing import Dict, List, Tuple

import yaml

from ir_config import prepare_config
from ir_scheduler_constants import ROTATIONS
from ir_scheduler_types import Resident, ScheduleError, ScheduleInput


def expand_residents(gui_residents: dict) -> List[dict]:
    if not isinstance(gui_residents, dict):
        raise ScheduleError("gui.residents must be a mapping.")

    ir_section = gui_residents.get("IR")
    dr_counts = gui_residents.get("DR_counts")
    if not isinstance(ir_section, dict):
        raise ScheduleError("gui.residents.IR must be a mapping of IR tracks to names.")
    if not isinstance(dr_counts, dict):
        raise ScheduleError("gui.residents.DR_counts must be a mapping of DR tracks to counts.")

    residents: List[dict] = []
    for track in ["IR1", "IR2", "IR3", "IR4", "IR5"]:
        names = ir_section.get(track)
        if not isinstance(names, list) or len(names) != 2:
            raise ScheduleError(f"gui.residents.IR.{track} must be a list of exactly 2 names.")
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ScheduleError(f"gui.residents.IR.{track} names must be non-empty strings.")
            residents.append({"id": name.strip(), "track": track})

    for track in ["DR1", "DR2", "DR3"]:
        count = dr_counts.get(track)
        if not isinstance(count, int) or count < 0:
            raise ScheduleError(f"gui.residents.DR_counts.{track} must be a non-negative integer.")
        for idx in range(1, count + 1):
            residents.append({"id": f"{track}-{idx}", "track": track})
    return residents


def _parse_block_labels(data: dict) -> List[str]:
    if "blocks" in data:
        blocks = data["blocks"]
        if isinstance(blocks, int):
            return [f"B{idx}" for idx in range(blocks)]
        if isinstance(blocks, list):
            return [str(b) for b in blocks]
    if "num_blocks" in data:
        num_blocks = data["num_blocks"]
        return [f"B{idx}" for idx in range(num_blocks)]
    raise ScheduleError("Input must include 'blocks' (list or int) or 'num_blocks'.")


def _parse_residents(data: dict) -> List[Resident]:
    raw = data.get("residents", [])
    if not raw:
        gui = data.get("gui") or {}
        if not isinstance(gui, dict):
            raise ScheduleError("gui must be a mapping when provided.")
        gui_residents = gui.get("residents")
        if gui_residents:
            raw = expand_residents(gui_residents)
        else:
            raise ScheduleError("Input must include at least one resident or gui.residents.")

    residents = []
    if not isinstance(raw, list):
        raise ScheduleError("Residents must be a list when provided.")
    for entry in raw:
        resident_id = entry.get("id") or entry.get("resident")
        if not resident_id:
            raise ScheduleError("Resident entries require 'id'.")
        track = entry.get("track")
        if track not in {
            "DR1",
            "DR2",
            "DR3",
            "IR1",
            "IR2",
            "IR3",
            "IR4",
            "IR5",
        }:
            raise ScheduleError(f"Unknown track for resident {resident_id}: {track}")
        residents.append(Resident(resident_id=resident_id, track=track))
    return residents


def _parse_block_index(block_labels: List[str], block) -> int:
    if isinstance(block, int):
        return block
    if isinstance(block, str):
        if block in block_labels:
            return block_labels.index(block)
    raise ScheduleError(f"Unknown block reference: {block}")


def _parse_blocked(data: dict, block_labels: List[str]) -> Dict[Tuple[str, int, str], bool]:
    blocked: Dict[Tuple[str, int, str], bool] = {}
    raw = data.get("blocked", [])
    if isinstance(raw, list):
        for entry in raw:
            resident_id = entry.get("resident")
            block = _parse_block_index(block_labels, entry.get("block"))
            rotation = entry.get("rotation")
            if rotation not in ROTATIONS:
                raise ScheduleError(f"Unknown rotation in blocked entry: {rotation}")
            blocked[(resident_id, block, rotation)] = True
    elif isinstance(raw, dict):
        for resident_id, blocks in raw.items():
            for block_key, rotations in blocks.items():
                block = _parse_block_index(block_labels, block_key)
                for rotation in rotations:
                    if rotation not in ROTATIONS:
                        raise ScheduleError(f"Unknown rotation in blocked entry: {rotation}")
                    blocked[(resident_id, block, rotation)] = True
    elif raw:
        raise ScheduleError("Blocked must be a list or dictionary when provided.")
    return blocked


def _parse_forced(data: dict, block_labels: List[str]) -> Dict[Tuple[str, int, str], bool]:
    forced: Dict[Tuple[str, int, str], bool] = {}
    per_resident_block: Dict[Tuple[str, int], str] = {}
    raw = data.get("forced", {})

    def _set(resident_id: str, block: int, rotation: str) -> None:
        if not resident_id:
            raise ScheduleError("Forced entries require 'resident'.")
        if not rotation:
            raise ScheduleError("Forced entries require 'rotation'.")
        if rotation not in ROTATIONS:
            raise ScheduleError(f"Unknown rotation in forced entry: {rotation}")
        key = (str(resident_id), int(block))
        if key in per_resident_block and per_resident_block[key] != rotation:
            raise ScheduleError(
                f"Forced assignments must have at most one rotation per resident/block; "
                f"{resident_id} block {block_labels[int(block)]} has both "
                f"{per_resident_block[key]} and {rotation}."
            )
        per_resident_block[key] = rotation
        forced[(str(resident_id), int(block), str(rotation))] = True

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                raise ScheduleError("Forced entries must be mappings when provided as a list.")
            resident_id = entry.get("resident")
            block = _parse_block_index(block_labels, entry.get("block"))
            rotation = entry.get("rotation")
            _set(resident_id, block, rotation)
    elif isinstance(raw, dict):
        for resident_id, blocks in raw.items():
            if not isinstance(blocks, dict):
                raise ScheduleError("Forced must map residents to a mapping of blocks.")
            for block_key, rotation in blocks.items():
                block = _parse_block_index(block_labels, block_key)
                if isinstance(rotation, list):
                    if len(rotation) != 1:
                        raise ScheduleError(
                            f"Forced assignments must have exactly one rotation per resident/block; "
                            f"{resident_id} block {block_labels[int(block)]} has {len(rotation)}."
                        )
                    rotation = rotation[0]
                _set(resident_id, block, rotation)
    elif raw:
        raise ScheduleError("Forced must be a list or dictionary when provided.")

    return forced


def _parse_weights(data: dict) -> Dict[str, int]:
    weights = data.get("weights", {})
    return {
        "consec": int(weights.get("consec", 100)),
        "first_timer": int(weights.get("first_timer", 30)),
        "adj": int(weights.get("adj", 1)),
    }


def _parse_gui_constraints(data: dict) -> tuple[Dict[str, str], List[str], Dict[str, dict]]:
    gui = data.get("gui") or {}
    if not isinstance(gui, dict):
        raise ScheduleError("gui must be a mapping when provided.")
    constraints = gui.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise ScheduleError("gui.constraints must be a mapping when provided.")
    modes = constraints.get("modes") or {}
    soft_priority = constraints.get("soft_priority") or []
    params = constraints.get("params") or {}

    if not isinstance(modes, dict):
        raise ScheduleError("gui.constraints.modes must be a mapping when provided.")
    if not isinstance(soft_priority, list):
        raise ScheduleError("gui.constraints.soft_priority must be a list when provided.")
    if not isinstance(params, dict):
        raise ScheduleError("gui.constraints.params must be a mapping when provided.")

    normalized_modes: Dict[str, str] = {}
    for key, value in modes.items():
        mode = str(value).lower()
        if mode not in {"always", "if_able", "disabled"}:
            raise ScheduleError(f"Unknown constraint mode for {key}: {value}")
        normalized_modes[str(key)] = mode

    normalized_params: Dict[str, dict] = {}
    for key, value in params.items():
        normalized_params[str(key)] = value if isinstance(value, dict) else {}

    return normalized_modes, [str(item) for item in soft_priority], normalized_params


def _parse_requirements(data: dict, num_blocks: int) -> Dict[str, Dict[str, int]]:
    raw = data.get("requirements")
    if not isinstance(raw, dict):
        raise ScheduleError("Input must include 'requirements' as a mapping.")
    requirements: Dict[str, Dict[str, int]] = {}
    expected_tracks = {"DR1", "DR2", "DR3", "IR1", "IR2", "IR3", "IR4", "IR5"}
    missing_tracks = expected_tracks.difference(raw.keys())
    if missing_tracks:
        raise ScheduleError(f"Missing requirements for tracks: {sorted(missing_tracks)}")
    for track, rotations in raw.items():
        if track not in expected_tracks:
            raise ScheduleError(f"Unknown track in requirements: {track}")
        if not isinstance(rotations, dict):
            raise ScheduleError(f"Requirements for {track} must be a mapping.")
        entry: Dict[str, int] = {}
        for rot in ROTATIONS:
            if rot not in rotations:
                raise ScheduleError(f"Requirements for {track} missing rotation {rot}.")
            value = rotations[rot]
            if track.startswith("IR"):
                try:
                    fte_value = float(value)
                except (TypeError, ValueError):
                    raise ScheduleError(f"Requirements for {track} {rot} must be a non-negative number.")
                if fte_value < 0:
                    raise ScheduleError(f"Requirements for {track} {rot} must be a non-negative number.")
                if rot == "KIR" and not fte_value.is_integer():
                    raise ScheduleError(
                        f"Requirements for {track} {rot} must be a non-negative whole number."
                    )
                units = int(round(fte_value * 2))
            else:
                blocks: int | None = None
                # YAML emitted by the GUI may serialize whole-number DR requirements as floats (e.g., 0.0).
                if isinstance(value, bool):
                    blocks = None
                elif isinstance(value, int):
                    blocks = value
                elif isinstance(value, float) and value.is_integer():
                    blocks = int(value)
                elif isinstance(value, str):
                    try:
                        blocks = int(value)
                    except ValueError:
                        blocks = None

                if blocks is None:
                    raise ScheduleError(f"Requirements for {track} {rot} must be a non-negative integer.")
                if blocks < 0:
                    raise ScheduleError(f"Requirements for {track} {rot} must be a non-negative integer.")
                units = 2 * blocks
            if units > 2 * num_blocks:
                raise ScheduleError(
                    f"Requirements for {track} {rot} exceed number of blocks ({num_blocks})."
                )
            entry[rot] = units
        total_units = sum(entry.values())
        if total_units % 2 != 0:
            raise ScheduleError(f"Total blocks for {track} must be a whole number before solving.")
        requirements[track] = entry
    return requirements


def load_schedule_input(path: str) -> ScheduleInput:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return load_schedule_input_from_data(data)


def load_schedule_input_from_data(data: dict) -> ScheduleInput:
    data, _ = prepare_config(data)
    block_labels = _parse_block_labels(data)
    residents = _parse_residents(data)
    blocked = _parse_blocked(data, block_labels)
    forced = _parse_forced(data, block_labels)
    for key in forced.keys():
        if blocked.get(key, False):
            resident_id, block, rotation = key
            raise ScheduleError(
                f"Forced assignment conflicts with blocked: {resident_id} {block_labels[int(block)]} {rotation}."
            )
    weights = _parse_weights(data)
    num_solutions = int(data.get("num_solutions", 1))
    constraint_modes, soft_priority, constraint_params = _parse_gui_constraints(data)
    requirements = _parse_requirements(data, len(block_labels))
    return ScheduleInput(
        block_labels=block_labels,
        residents=residents,
        blocked=blocked,
        forced=forced,
        weights=weights,
        num_solutions=num_solutions,
        constraint_modes=constraint_modes,
        soft_priority=soft_priority,
        constraint_params=constraint_params,
        requirements=requirements,
    )

