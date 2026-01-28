from __future__ import annotations

import argparse
import csv
import os
import sys
from io import StringIO
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from ortools.sat.python import cp_model


ROTATIONS = ["KIR", "MH-IR", "MH-CT/US", "48X-IR", "48X-CT/US"]


@dataclass(frozen=True)
class Resident:
    resident_id: str
    track: str


@dataclass(frozen=True)
class ScheduleInput:
    block_labels: List[str]
    residents: List[Resident]
    blocked: Dict[Tuple[str, int, str], bool]
    weights: Dict[str, int]
    num_solutions: int
    constraint_modes: Dict[str, str]
    soft_priority: List[str]


@dataclass
class Solution:
    assignments: Dict[str, Dict[str, Dict[str, float]]]
    objective: Dict[str, int]


@dataclass
class Diagnostic:
    status: str
    conflicting_constraints: List[Dict[str, str]]
    suggestions: List[Dict[str, str]]


@dataclass
class SolveResult:
    solutions: List[Solution]
    diagnostic: Optional[Diagnostic] = None


class ScheduleError(Exception):
    pass


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


def _parse_weights(data: dict) -> Dict[str, int]:
    weights = data.get("weights", {})
    return {
        "consec": int(weights.get("consec", 100)),
        "first_timer": int(weights.get("first_timer", 30)),
        "adj": int(weights.get("adj", 1)),
    }


def _parse_gui_constraints(data: dict) -> Tuple[Dict[str, str], List[str]]:
    gui = data.get("gui") or {}
    if not isinstance(gui, dict):
        raise ScheduleError("gui must be a mapping when provided.")
    constraints = gui.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise ScheduleError("gui.constraints must be a mapping when provided.")
    modes = constraints.get("modes") or {}
    soft_priority = constraints.get("soft_priority") or []

    if not isinstance(modes, dict):
        raise ScheduleError("gui.constraints.modes must be a mapping when provided.")
    if not isinstance(soft_priority, list):
        raise ScheduleError("gui.constraints.soft_priority must be a list when provided.")

    normalized_modes: Dict[str, str] = {}
    for key, value in modes.items():
        mode = str(value).lower()
        if mode not in {"always", "if_able", "disabled"}:
            raise ScheduleError(f"Unknown constraint mode for {key}: {value}")
        normalized_modes[str(key)] = mode

    return normalized_modes, [str(item) for item in soft_priority]


def load_schedule_input(path: str) -> ScheduleInput:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    block_labels = _parse_block_labels(data)
    residents = _parse_residents(data)
    blocked = _parse_blocked(data, block_labels)
    weights = _parse_weights(data)
    num_solutions = int(data.get("num_solutions", 1))
    constraint_modes, soft_priority = _parse_gui_constraints(data)
    return ScheduleInput(
        block_labels=block_labels,
        residents=residents,
        blocked=blocked,
        weights=weights,
        num_solutions=num_solutions,
        constraint_modes=constraint_modes,
        soft_priority=soft_priority,
    )


def _build_variables(
    model: cp_model.CpModel,
    residents: List[Resident],
    block_labels: List[str],
) -> Tuple[Dict[Tuple[str, int, str], cp_model.IntVar], Dict[Tuple[str, int, str], cp_model.IntVar]]:
    u: Dict[Tuple[str, int, str], cp_model.IntVar] = {}
    p: Dict[Tuple[str, int, str], cp_model.IntVar] = {}
    for resident in residents:
        for b, _ in enumerate(block_labels):
            for rot in ROTATIONS:
                u_var = model.NewIntVar(0, 2, f"u_{resident.resident_id}_{b}_{rot}")
                p_var = model.NewBoolVar(f"p_{resident.resident_id}_{b}_{rot}")
                model.Add(u_var >= 1).OnlyEnforceIf(p_var)
                model.Add(u_var == 0).OnlyEnforceIf(p_var.Not())
                u[(resident.resident_id, b, rot)] = u_var
                p[(resident.resident_id, b, rot)] = p_var
    return u, p


def _resident_groups(residents: List[Resident]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {
        "DR1": [],
        "DR2": [],
        "DR3": [],
        "IR1": [],
        "IR2": [],
        "IR3": [],
        "IR4": [],
        "IR5": [],
    }
    for resident in residents:
        groups[resident.track].append(resident.resident_id)
    groups["IR4_PLUS"] = groups["IR4"] + groups["IR5"]
    groups["FIRST_TIMER_CANDIDATES"] = groups["DR1"] + groups["IR1"]
    groups["YEAR1_3"] = (
        groups["DR1"]
        + groups["DR2"]
        + groups["DR3"]
        + groups["IR1"]
        + groups["IR2"]
        + groups["IR3"]
    )
    return groups


def _total_units(
    u: Dict[Tuple[str, int, str], cp_model.IntVar], resident_id: str, rot: str, num_blocks: int
):
    return sum(u[(resident_id, b, rot)] for b in range(num_blocks))


def _enforce(constraint: cp_model.Constraint, assumption: Optional[cp_model.BoolVar]):
    if assumption is not None:
        constraint.OnlyEnforceIf(assumption)
    return constraint


@dataclass
class ConstraintContext:
    model: cp_model.CpModel
    schedule_input: ScheduleInput
    u: Dict[Tuple[str, int, str], cp_model.IntVar]
    p: Dict[Tuple[str, int, str], cp_model.IntVar]
    groups: Dict[str, List[str]]
    num_blocks: int
    _full_mh: Optional[Dict[Tuple[str, int], cp_model.BoolVar]] = None
    _first_timer: Optional[Dict[Tuple[str, int], cp_model.BoolVar]] = None

    def full_mh(self) -> Dict[Tuple[str, int], cp_model.BoolVar]:
        if self._full_mh is not None:
            return self._full_mh
        full_mh: Dict[Tuple[str, int], cp_model.BoolVar] = {}
        for resident in self.schedule_input.residents:
            for b in range(self.num_blocks):
                fm = self.model.NewBoolVar(f"full_mh_{resident.resident_id}_{b}")
                self.model.Add(self.u[(resident.resident_id, b, "MH-IR")] == 2).OnlyEnforceIf(fm)
                self.model.Add(self.u[(resident.resident_id, b, "MH-IR")] <= 1).OnlyEnforceIf(fm.Not())
                full_mh[(resident.resident_id, b)] = fm
        self._full_mh = full_mh
        return full_mh

    def first_timer(self) -> Dict[Tuple[str, int], cp_model.BoolVar]:
        if self._first_timer is not None:
            return self._first_timer
        first_timer: Dict[Tuple[str, int], cp_model.BoolVar] = {}
        for resident_id in self.groups["FIRST_TIMER_CANDIDATES"]:
            for b in range(self.num_blocks):
                prior_units = self.model.NewIntVar(0, self.num_blocks, f"prior_mh_{resident_id}_{b}")
                _enforce(
                    self.model.Add(prior_units == sum(self.p[(resident_id, k, "MH-IR")] for k in range(b))),
                    None,
                )
                prior_zero = self.model.NewBoolVar(f"prior_zero_{resident_id}_{b}")
                _enforce(self.model.Add(prior_units == 0), None).OnlyEnforceIf(prior_zero)
                _enforce(self.model.Add(prior_units >= 1), None).OnlyEnforceIf(prior_zero.Not())
                ft = self.model.NewBoolVar(f"first_timer_{resident_id}_{b}")
                self.model.AddBoolAnd([self.p[(resident_id, b, "MH-IR")], prior_zero]).OnlyEnforceIf(ft)
                self.model.AddBoolOr(
                    [self.p[(resident_id, b, "MH-IR")].Not(), prior_zero.Not()]
                ).OnlyEnforceIf(ft.Not())
                first_timer[(resident_id, b)] = ft
        self._first_timer = first_timer
        return first_timer


@dataclass(frozen=True)
class ConstraintSpec:
    id: str
    label: str
    softenable: bool
    impact: int
    add_hard: Callable[[ConstraintContext, Optional[cp_model.BoolVar]], None]
    add_soft: Optional[Callable[[ConstraintContext], List[cp_model.IntVar]]] = None
    weight_key: Optional[str] = None
    objective_key: Optional[str] = None


def _add_one_place_at_time(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident in ctx.schedule_input.residents:
        for b in range(ctx.num_blocks):
            _enforce(
                ctx.model.Add(sum(ctx.u[(resident.resident_id, b, rot)] for rot in ROTATIONS) <= 2),
                assumption,
            )


def _add_blocked(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for (resident_id, block, rotation), is_blocked in ctx.schedule_input.blocked.items():
        if is_blocked:
            _enforce(ctx.model.Add(ctx.u[(resident_id, block, rotation)] == 0), assumption)


def _add_no_half_assignments(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident in ctx.schedule_input.residents:
        is_ir5 = resident.track == "IR5"
        for b in range(ctx.num_blocks):
            for rot in ROTATIONS:
                if is_ir5 and rot in {"MH-IR", "48X-IR"}:
                    continue
                _enforce(ctx.model.Add(ctx.u[(resident.resident_id, b, rot)] != 1), assumption)


def _add_ir5_split_coupling(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["IR5"]:
        for b in range(ctx.num_blocks):
            mh = ctx.u[(resident_id, b, "MH-IR")]
            x48 = ctx.u[(resident_id, b, "48X-IR")]
            is_half_mh = ctx.model.NewBoolVar(f"is_half_mh_{resident_id}_{b}")
            is_half_x48 = ctx.model.NewBoolVar(f"is_half_x48_{resident_id}_{b}")
            _enforce(ctx.model.Add(mh == 1), assumption).OnlyEnforceIf(is_half_mh)
            _enforce(ctx.model.Add(mh != 1), assumption).OnlyEnforceIf(is_half_mh.Not())
            _enforce(ctx.model.Add(x48 == 1), assumption).OnlyEnforceIf(is_half_x48)
            _enforce(ctx.model.Add(x48 != 1), assumption).OnlyEnforceIf(is_half_x48.Not())
            _enforce(ctx.model.Add(is_half_mh == is_half_x48), assumption)


def _add_coverage_48x_ir(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for b in range(ctx.num_blocks):
        _enforce(
            ctx.model.Add(sum(ctx.u[(resident.resident_id, b, "48X-IR")] for resident in ctx.schedule_input.residents)
                          == 2),
            assumption,
        )


def _add_coverage_48x_ctus(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for b in range(ctx.num_blocks):
        _enforce(
            ctx.model.Add(
                sum(ctx.u[(resident.resident_id, b, "48X-CT/US")] for resident in ctx.schedule_input.residents)
                == 2
            ),
            assumption,
        )


def _add_mh_total_minmax(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for b in range(ctx.num_blocks):
        mh_total = sum(ctx.u[(resident.resident_id, b, "MH-IR")] for resident in ctx.schedule_input.residents) + sum(
            ctx.u[(resident.resident_id, b, "MH-CT/US")] for resident in ctx.schedule_input.residents
        )
        _enforce(ctx.model.Add(mh_total >= 6), assumption)
        _enforce(ctx.model.Add(mh_total <= 8), assumption)


def _add_mh_ctus_cap(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for b in range(ctx.num_blocks):
        _enforce(
            ctx.model.Add(sum(ctx.u[(resident.resident_id, b, "MH-CT/US")] for resident in ctx.schedule_input.residents)
                          <= 2),
            assumption,
        )


def _add_kir_cap(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for b in range(ctx.num_blocks):
        _enforce(
            ctx.model.Add(sum(ctx.u[(resident.resident_id, b, "KIR")] for resident in ctx.schedule_input.residents) <= 4),
            assumption,
        )


def _add_dr1_totals(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["DR1"]:
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-IR", ctx.num_blocks) == 2), assumption)
        for rot in ["KIR", "MH-CT/US", "48X-IR", "48X-CT/US"]:
            _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, rot, ctx.num_blocks) == 0), assumption)


def _add_dr2_totals(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["DR2"]:
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-CT/US", ctx.num_blocks) == 2), assumption)
        for rot in ["KIR", "MH-IR", "48X-IR", "48X-CT/US"]:
            _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, rot, ctx.num_blocks) == 0), assumption)


def _add_dr3_totals(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["DR3"]:
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-CT/US", ctx.num_blocks) == 2), assumption)
        for rot in ["KIR", "MH-IR", "MH-CT/US", "48X-IR"]:
            _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, rot, ctx.num_blocks) == 0), assumption)


def _add_ir1_totals(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["IR1"]:
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-IR", ctx.num_blocks) == 2), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-IR", ctx.num_blocks) == 2), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-CT/US", ctx.num_blocks) == 2), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "KIR", ctx.num_blocks) == 0), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-CT/US", ctx.num_blocks) == 0), assumption)


def _add_ir2_totals(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["IR2"]:
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-IR", ctx.num_blocks) == 4), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-IR", ctx.num_blocks) == 2), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "KIR", ctx.num_blocks) == 0), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-CT/US", ctx.num_blocks) == 0), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-CT/US", ctx.num_blocks) == 0), assumption)


def _add_ir3_totals(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["IR3"]:
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-IR", ctx.num_blocks) == 2), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-IR", ctx.num_blocks) == 2), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-CT/US", ctx.num_blocks) == 2), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "KIR", ctx.num_blocks) == 0), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-CT/US", ctx.num_blocks) == 0), assumption)


def _add_ir4_totals(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["IR4"]:
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "KIR", ctx.num_blocks) == 6), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-IR", ctx.num_blocks) == 6), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-IR", ctx.num_blocks) == 0), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-CT/US", ctx.num_blocks) == 0), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-CT/US", ctx.num_blocks) == 0), assumption)


def _add_ir5_totals(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["IR5"]:
        _enforce(
            ctx.model.Add(sum(_total_units(ctx.u, resident_id, rot, ctx.num_blocks) for rot in ROTATIONS) == 2 * ctx.num_blocks),
            assumption,
        )
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "KIR", ctx.num_blocks) == 6), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "48X-IR", ctx.num_blocks) >= 4), assumption)
        _enforce(ctx.model.Add(_total_units(ctx.u, resident_id, "MH-CT/US", ctx.num_blocks) == 0), assumption)


def _add_ir5_mh_min_per_block(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    if not ctx.groups["IR5"]:
        return
    for b in range(ctx.num_blocks):
        _enforce(ctx.model.Add(sum(ctx.u[(resident_id, b, "MH-IR")] for resident_id in ctx.groups["IR5"]) >= 2), assumption)


def _add_ir4_plus_mh_cap(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for b in range(ctx.num_blocks):
        _enforce(
            ctx.model.Add(sum(ctx.u[(resident_id, b, "MH-IR")] for resident_id in ctx.groups["IR4_PLUS"]) <= 4),
            assumption,
        )


def _add_dr1_early_block_mh_prohibited(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    max_block = min(4, ctx.num_blocks)
    for resident_id in ctx.groups["DR1"]:
        for b in range(max_block):
            _enforce(ctx.model.Add(ctx.u[(resident_id, b, "MH-IR")] == 0), assumption)


def _add_ir3_late_block_restrictions(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    if ctx.num_blocks >= 13:
        for resident_id in ctx.groups["IR3"]:
            for b in range(7, min(13, ctx.num_blocks)):
                _enforce(ctx.model.Add(ctx.u[(resident_id, b, "MH-IR")] == 0), assumption)
                _enforce(ctx.model.Add(ctx.u[(resident_id, b, "48X-IR")] == 0), assumption)


def _add_first_timer_hard(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    first_timer = ctx.first_timer()
    for b in range(ctx.num_blocks):
        count_b = sum(first_timer[(resident_id, b)] for resident_id in ctx.groups["FIRST_TIMER_CANDIDATES"])
        _enforce(ctx.model.Add(count_b <= 1), assumption)


def _add_first_timer_soft(ctx: ConstraintContext) -> List[cp_model.IntVar]:
    first_timer = ctx.first_timer()
    excess_vars: List[cp_model.IntVar] = []
    for b in range(ctx.num_blocks):
        count_b = sum(first_timer[(resident_id, b)] for resident_id in ctx.groups["FIRST_TIMER_CANDIDATES"])
        excess = ctx.model.NewIntVar(0, len(ctx.groups["FIRST_TIMER_CANDIDATES"]), f"ft_excess_{b}")
        ctx.model.Add(count_b <= 1 + excess)
        excess_vars.append(excess)
    return excess_vars


def _add_consec_full_mh_hard(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    full_mh = ctx.full_mh()
    for resident in ctx.schedule_input.residents:
        for b in range(0, ctx.num_blocks - 2):
            window = full_mh[(resident.resident_id, b)]
            window += full_mh[(resident.resident_id, b + 1)]
            window += full_mh[(resident.resident_id, b + 2)]
            _enforce(ctx.model.Add(window <= 2), assumption)


def _add_consec_full_mh_soft(ctx: ConstraintContext) -> List[cp_model.IntVar]:
    full_mh = ctx.full_mh()
    excess_vars: List[cp_model.IntVar] = []
    for resident in ctx.schedule_input.residents:
        for b in range(0, ctx.num_blocks - 2):
            window = full_mh[(resident.resident_id, b)]
            window += full_mh[(resident.resident_id, b + 1)]
            window += full_mh[(resident.resident_id, b + 2)]
            excess = ctx.model.NewIntVar(0, 1, f"mh3_excess_{resident.resident_id}_{b}")
            ctx.model.Add(window <= 2 + excess)
            excess_vars.append(excess)
    return excess_vars


def _add_no_sequential_hard(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident_id in ctx.groups["YEAR1_3"]:
        for b in range(ctx.num_blocks - 1):
            sum_b = sum(ctx.p[(resident_id, b, rot)] for rot in ROTATIONS)
            sum_next = sum(ctx.p[(resident_id, b + 1, rot)] for rot in ROTATIONS)
            _enforce(ctx.model.Add(sum_b + sum_next <= 1), assumption)


def _add_no_sequential_soft(ctx: ConstraintContext) -> List[cp_model.IntVar]:
    penalties: List[cp_model.IntVar] = []
    for resident_id in ctx.groups["YEAR1_3"]:
        for b in range(ctx.num_blocks - 1):
            any_b = ctx.model.NewBoolVar(f"any_assign_{resident_id}_{b}")
            any_next = ctx.model.NewBoolVar(f"any_assign_{resident_id}_{b+1}")
            sum_b = sum(ctx.p[(resident_id, b, rot)] for rot in ROTATIONS)
            sum_next = sum(ctx.p[(resident_id, b + 1, rot)] for rot in ROTATIONS)
            ctx.model.Add(sum_b >= 1).OnlyEnforceIf(any_b)
            ctx.model.Add(sum_b == 0).OnlyEnforceIf(any_b.Not())
            ctx.model.Add(sum_next >= 1).OnlyEnforceIf(any_next)
            ctx.model.Add(sum_next == 0).OnlyEnforceIf(any_next.Not())

            z = ctx.model.NewBoolVar(f"adj_any_{resident_id}_{b}")
            ctx.model.AddBoolAnd([any_b, any_next]).OnlyEnforceIf(z)
            ctx.model.AddBoolOr([any_b.Not(), any_next.Not()]).OnlyEnforceIf(z.Not())
            penalties.append(z)
    return penalties


CONSTRAINT_SPECS: List[ConstraintSpec] = [
    ConstraintSpec(
        id="one_place",
        label="One place at a time (<=1.0 FTE per block)",
        softenable=False,
        impact=70,
        add_hard=_add_one_place_at_time,
    ),
    ConstraintSpec(
        id="blocked",
        label="Blocked assignments",
        softenable=False,
        impact=20,
        add_hard=_add_blocked,
    ),
    ConstraintSpec(
        id="no_half_non_ir5",
        label="No half assignments outside IR5 split",
        softenable=False,
        impact=30,
        add_hard=_add_no_half_assignments,
    ),
    ConstraintSpec(
        id="ir5_split_coupling",
        label="IR5 MH-IR/48X-IR split coupling",
        softenable=False,
        impact=40,
        add_hard=_add_ir5_split_coupling,
    ),
    ConstraintSpec(
        id="coverage_48x_ir",
        label="48X-IR coverage == 1.0 FTE",
        softenable=False,
        impact=100,
        add_hard=_add_coverage_48x_ir,
    ),
    ConstraintSpec(
        id="coverage_48x_ctus",
        label="48X-CT/US coverage == 1.0 FTE",
        softenable=False,
        impact=100,
        add_hard=_add_coverage_48x_ctus,
    ),
    ConstraintSpec(
        id="mh_total_minmax",
        label="MH total coverage between 3.0 and 4.0 FTE",
        softenable=False,
        impact=90,
        add_hard=_add_mh_total_minmax,
    ),
    ConstraintSpec(
        id="mh_ctus_cap",
        label="MH-CT/US cap <= 1.0 FTE",
        softenable=False,
        impact=50,
        add_hard=_add_mh_ctus_cap,
    ),
    ConstraintSpec(
        id="kir_cap",
        label="KIR cap <= 2.0 FTE",
        softenable=False,
        impact=50,
        add_hard=_add_kir_cap,
    ),
    ConstraintSpec(
        id="dr1_totals",
        label="DR1 totals (MH-IR only)",
        softenable=False,
        impact=80,
        add_hard=_add_dr1_totals,
    ),
    ConstraintSpec(
        id="dr2_totals",
        label="DR2 totals (MH-CT/US only)",
        softenable=False,
        impact=80,
        add_hard=_add_dr2_totals,
    ),
    ConstraintSpec(
        id="dr3_totals",
        label="DR3 totals (48X-CT/US only)",
        softenable=False,
        impact=80,
        add_hard=_add_dr3_totals,
    ),
    ConstraintSpec(
        id="ir1_totals",
        label="IR1 totals",
        softenable=False,
        impact=80,
        add_hard=_add_ir1_totals,
    ),
    ConstraintSpec(
        id="ir2_totals",
        label="IR2 totals",
        softenable=False,
        impact=80,
        add_hard=_add_ir2_totals,
    ),
    ConstraintSpec(
        id="ir3_totals",
        label="IR3 totals",
        softenable=False,
        impact=80,
        add_hard=_add_ir3_totals,
    ),
    ConstraintSpec(
        id="ir4_totals",
        label="IR4 totals",
        softenable=False,
        impact=80,
        add_hard=_add_ir4_totals,
    ),
    ConstraintSpec(
        id="ir5_totals",
        label="IR5 totals",
        softenable=False,
        impact=90,
        add_hard=_add_ir5_totals,
    ),
    ConstraintSpec(
        id="ir5_mh_min_per_block",
        label="IR5 MH-IR min 1.0 FTE per block",
        softenable=False,
        impact=70,
        add_hard=_add_ir5_mh_min_per_block,
    ),
    ConstraintSpec(
        id="ir4_plus_mh_cap",
        label="IR4+ MH-IR max 2.0 FTE per block",
        softenable=False,
        impact=60,
        add_hard=_add_ir4_plus_mh_cap,
    ),
    ConstraintSpec(
        id="dr1_early_block",
        label="DR1 no MH-IR in first four blocks",
        softenable=False,
        impact=15,
        add_hard=_add_dr1_early_block_mh_prohibited,
    ),
    ConstraintSpec(
        id="ir3_late_block",
        label="IR3 no MH-IR or 48X-IR in blocks 8-13",
        softenable=False,
        impact=15,
        add_hard=_add_ir3_late_block_restrictions,
    ),
    ConstraintSpec(
        id="first_timer",
        label="Limit first-timer MH-IR to 1 per block",
        softenable=True,
        impact=25,
        add_hard=_add_first_timer_hard,
        add_soft=_add_first_timer_soft,
        weight_key="first_timer",
        objective_key="first_timer_excess",
    ),
    ConstraintSpec(
        id="consec_full_mh",
        label="Avoid 3 full MH-IR blocks in any 3-block window",
        softenable=True,
        impact=25,
        add_hard=_add_consec_full_mh_hard,
        add_soft=_add_consec_full_mh_soft,
        weight_key="consec",
        objective_key="consec_excess",
    ),
    ConstraintSpec(
        id="no_sequential_year1_3",
        label="No back-to-back blocks for year 1-3 residents",
        softenable=True,
        impact=35,
        add_hard=_add_no_sequential_hard,
        add_soft=_add_no_sequential_soft,
        weight_key="adj",
        objective_key="adj",
    ),
]


SPEC_BY_ID = {spec.id: spec for spec in CONSTRAINT_SPECS}


def _default_mode_for_spec(spec: ConstraintSpec) -> str:
    return "if_able" if spec.softenable else "always"


def _constraint_mode(spec: ConstraintSpec, mode_map: Dict[str, str]) -> str:
    return mode_map.get(spec.id, _default_mode_for_spec(spec))


def _apply_constraints(
    model: cp_model.CpModel,
    schedule_input: ScheduleInput,
    u: Dict[Tuple[str, int, str], cp_model.IntVar],
    p: Dict[Tuple[str, int, str], cp_model.IntVar],
) -> Tuple[
    Dict[str, List[cp_model.IntVar]],
    Dict[int, ConstraintSpec],
    Dict[int, cp_model.BoolVar],
]:
    groups = _resident_groups(schedule_input.residents)
    ctx = ConstraintContext(
        model=model,
        schedule_input=schedule_input,
        u=u,
        p=p,
        groups=groups,
        num_blocks=len(schedule_input.block_labels),
    )

    penalties: Dict[str, List[cp_model.IntVar]] = {spec.id: [] for spec in CONSTRAINT_SPECS}
    assumption_index_map: Dict[int, ConstraintSpec] = {}
    assumption_var_map: Dict[int, cp_model.BoolVar] = {}

    for spec in CONSTRAINT_SPECS:
        mode = _constraint_mode(spec, schedule_input.constraint_modes)
        if mode == "disabled":
            continue
        if mode == "if_able" and spec.softenable and spec.add_soft is not None:
            penalties[spec.id] = spec.add_soft(ctx)
            continue

        assumption = model.NewBoolVar(f"a_{spec.id}")
        model.AddAssumption(assumption)
        assumption_index_map[assumption.Index()] = spec
        assumption_var_map[assumption.Index()] = assumption
        spec.add_hard(ctx, assumption)

    return penalties, assumption_index_map, assumption_var_map


def _linear_terms_expr(terms: Iterable[Tuple[cp_model.IntVar, int]]):
    expr = 0
    for var, coeff in terms:
        expr += coeff * var
    return expr


def _linear_terms_value(solver: cp_model.CpSolver, terms: Iterable[Tuple[cp_model.IntVar, int]]) -> int:
    return int(sum(coeff * solver.Value(var) for var, coeff in terms))


def _optimize_and_fix(
    model: cp_model.CpModel, solver: cp_model.CpSolver, terms: Iterable[Tuple[cp_model.IntVar, int]]
) -> Tuple[int, Optional[int]]:
    terms = list(terms)
    if not terms:
        return cp_model.OPTIMAL, 0
    expr = _linear_terms_expr(terms)
    model.Minimize(expr)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return status, None
    best = _linear_terms_value(solver, terms)
    model.Add(expr == best)
    return status, best


def _build_weighted_terms(
    penalties: Dict[str, List[cp_model.IntVar]],
    weights: Dict[str, int],
    ignore_ids: Optional[set] = None,
) -> List[Tuple[cp_model.IntVar, int]]:
    if ignore_ids is None:
        ignore_ids = set()
    terms: List[Tuple[cp_model.IntVar, int]] = []
    for spec in CONSTRAINT_SPECS:
        if spec.weight_key is None or spec.id in ignore_ids:
            continue
        weight = int(weights.get(spec.weight_key, 0))
        if weight == 0:
            continue
        for var in penalties.get(spec.id, []):
            terms.append((var, weight))
    return terms


def _core_indices(core: Sequence) -> List[int]:
    indices: List[int] = []
    for lit in core:
        if hasattr(lit, "Index"):
            idx = lit.Index()
        else:
            idx = abs(int(lit))
        indices.append(idx)
    return indices


def _shrink_core(
    solver: cp_model.CpSolver, model: cp_model.CpModel, assumptions: List[cp_model.BoolVar]
) -> List[cp_model.BoolVar]:
    if not hasattr(solver, "SolveWithAssumptions"):
        return assumptions
    core = list(assumptions)
    idx = 0
    while idx < len(core):
        trial = core[:idx] + core[idx + 1 :]
        status = solver.SolveWithAssumptions(model, trial)
        if status == cp_model.INFEASIBLE:
            core = trial
        else:
            idx += 1
    return core


def _with_constraint_modes(schedule_input: ScheduleInput, overrides: Dict[str, str]) -> ScheduleInput:
    modes = dict(schedule_input.constraint_modes)
    modes.update(overrides)
    return ScheduleInput(
        block_labels=schedule_input.block_labels,
        residents=schedule_input.residents,
        blocked=schedule_input.blocked,
        weights=schedule_input.weights,
        num_solutions=schedule_input.num_solutions,
        constraint_modes=modes,
        soft_priority=schedule_input.soft_priority,
    )


def _is_feasible(schedule_input: ScheduleInput) -> bool:
    model = cp_model.CpModel()
    u, p = _build_variables(model, schedule_input.residents, schedule_input.block_labels)
    _apply_constraints(model, schedule_input, u, p)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def _suggest_relaxations_fast(
    schedule_input: ScheduleInput,
    specs: List[ConstraintSpec],
    max_suggestions: int = 3,
    verify_top_n: int = 1,
    progress_cb: Optional[Callable[[int, int, ConstraintSpec, str], None]] = None,
) -> List[Dict[str, str]]:
    candidates: List[Tuple[ConstraintSpec, str]] = []
    for spec in specs:
        current_mode = _constraint_mode(spec, schedule_input.constraint_modes)
        mode = "if_able" if spec.softenable else "disabled"
        if mode != current_mode:
            candidates.append((spec, mode))

    candidates.sort(key=lambda item: item[0].impact, reverse=True)
    suggestions: List[Dict[str, str]] = []
    total = min(len(candidates), max_suggestions)
    total_verify = min(verify_top_n, total)
    for idx, (spec, mode) in enumerate(candidates[:max_suggestions], start=1):
        verified = False
        if idx <= verify_top_n:
            if progress_cb:
                progress_cb(idx, total_verify, spec, mode)
            candidate_input = _with_constraint_modes(schedule_input, {spec.id: mode})
            verified = _is_feasible(candidate_input)
        suggestions.append(
            {
                "id": spec.id,
                "label": spec.label,
                "mode": mode,
                "verified": verified,
            }
        )
    return suggestions


def _build_diagnostic(
    schedule_input: ScheduleInput,
    model: cp_model.CpModel,
    solver: cp_model.CpSolver,
    assumption_index_map: Dict[int, ConstraintSpec],
    assumption_var_map: Dict[int, cp_model.BoolVar],
    shrink_core: bool = False,
    suggest_relaxations: bool = False,
    progress_cb: Optional[Callable[[int, int, ConstraintSpec, str], None]] = None,
) -> Diagnostic:
    core = solver.SufficientAssumptionsForInfeasibility()
    core_indices = _core_indices(core)
    if shrink_core and assumption_var_map:
        core_vars = [assumption_var_map[idx] for idx in core_indices if idx in assumption_var_map]
        shrunk = _shrink_core(solver, model, core_vars)
        core_indices = [var.Index() for var in shrunk]

    seen: set = set()
    specs: List[ConstraintSpec] = []
    conflicts: List[Dict[str, str]] = []
    for idx in core_indices:
        spec = assumption_index_map.get(idx)
        if spec and spec.id not in seen:
            specs.append(spec)
            conflicts.append({"id": spec.id, "label": spec.label})
            seen.add(spec.id)
    suggestions = (
        _suggest_relaxations_fast(schedule_input, specs, progress_cb=progress_cb)
        if suggest_relaxations
        else []
    )
    return Diagnostic(
        status="INFEASIBLE",
        conflicting_constraints=conflicts,
        suggestions=suggestions,
    )


def _extract_solution(
    solver: cp_model.CpSolver,
    schedule_input: ScheduleInput,
    u: Dict[Tuple[str, int, str], cp_model.IntVar],
    penalties: Dict[str, List[cp_model.IntVar]],
) -> Solution:
    assignments: Dict[str, Dict[str, Dict[str, float]]] = {}
    for resident in schedule_input.residents:
        resident_assignments: Dict[str, Dict[str, float]] = {}
        for b, label in enumerate(schedule_input.block_labels):
            block_assignments: Dict[str, float] = {}
            for rot in ROTATIONS:
                units = solver.Value(u[(resident.resident_id, b, rot)])
                if units > 0:
                    block_assignments[rot] = units / 2.0
            resident_assignments[label] = block_assignments
        assignments[resident.resident_id] = resident_assignments

    objective: Dict[str, int] = {}
    for spec in CONSTRAINT_SPECS:
        if spec.objective_key is None:
            continue
        objective[spec.objective_key] = int(sum(solver.Value(var) for var in penalties.get(spec.id, [])))

    return Solution(assignments=assignments, objective=objective)


def solve_schedule(
    schedule_input: ScheduleInput,
    suggest_relaxations: bool = False,
    shrink_core: bool = False,
    progress_cb: Optional[Callable[[int, int, ConstraintSpec, str], None]] = None,
) -> SolveResult:
    model = cp_model.CpModel()
    u, p = _build_variables(model, schedule_input.residents, schedule_input.block_labels)
    penalties, assumption_index_map, assumption_var_map = _apply_constraints(model, schedule_input, u, p)

    solver = cp_model.CpSolver()

    groups = _resident_groups(schedule_input.residents)
    ir5_ids = groups["IR5"]
    if len(ir5_ids) == 2:
        num_blocks = len(schedule_input.block_labels)
        t0 = _total_units(u, ir5_ids[0], "48X-IR", num_blocks)
        t1 = _total_units(u, ir5_ids[1], "48X-IR", num_blocks)
        diff = model.NewIntVar(0, 2 * num_blocks, "ir5_48x_ir_diff")
        model.AddAbsEquality(diff, t0 - t1)
        status, _ = _optimize_and_fix(model, solver, [(diff, 1)])
        if status == cp_model.INFEASIBLE:
            return SolveResult(
                [],
                _build_diagnostic(
                    schedule_input,
                    model,
                    solver,
                    assumption_index_map,
                    assumption_var_map,
                    shrink_core=shrink_core,
                    suggest_relaxations=suggest_relaxations,
                    progress_cb=progress_cb,
                ),
            )
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return SolveResult([], None)

    soft_priority = [cid for cid in schedule_input.soft_priority if cid in SPEC_BY_ID]
    if soft_priority:
        for cid in soft_priority:
            spec = SPEC_BY_ID[cid]
            if _constraint_mode(spec, schedule_input.constraint_modes) != "if_able":
                continue
            penalty_vars = penalties.get(cid, [])
            if not penalty_vars:
                continue
            status, _ = _optimize_and_fix(model, solver, [(var, 1) for var in penalty_vars])
            if status == cp_model.INFEASIBLE:
                return SolveResult(
                    [],
                    _build_diagnostic(
                        schedule_input,
                        model,
                        solver,
                        assumption_index_map,
                        assumption_var_map,
                        shrink_core=shrink_core,
                        suggest_relaxations=suggest_relaxations,
                        progress_cb=progress_cb,
                    ),
                )
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return SolveResult([], None)
        final_terms = _build_weighted_terms(penalties, schedule_input.weights, ignore_ids=set(soft_priority))
    else:
        final_terms = _build_weighted_terms(penalties, schedule_input.weights)

    if final_terms:
        model.Minimize(_linear_terms_expr(final_terms))

    solutions: List[Solution] = []
    last_status = None
    while len(solutions) < schedule_input.num_solutions:
        status = solver.Solve(model)
        last_status = status
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        solutions.append(_extract_solution(solver, schedule_input, u, penalties))
        lits = []
        for resident in schedule_input.residents:
            for b in range(len(schedule_input.block_labels)):
                for rot in ROTATIONS:
                    if solver.Value(p[(resident.resident_id, b, rot)]) == 1:
                        lits.append(p[(resident.resident_id, b, rot)])
                    else:
                        lits.append(p[(resident.resident_id, b, rot)].Not())
        model.AddBoolOr([lit.Not() for lit in lits])

    if not solutions and last_status == cp_model.INFEASIBLE:
        return SolveResult(
            [],
            _build_diagnostic(
                schedule_input,
                model,
                solver,
                assumption_index_map,
                assumption_var_map,
                shrink_core=shrink_core,
                suggest_relaxations=suggest_relaxations,
                progress_cb=progress_cb,
            ),
        )

    return SolveResult(solutions, None)


def _solutions_to_yaml(result: SolveResult) -> str:
    payload: Dict[str, object] = {"solutions": []}
    if result.solutions:
        payload["solutions"] = [
            {"objective": solution.objective, "assignments": solution.assignments}
            for solution in result.solutions
        ]
    if result.diagnostic:
        payload["diagnostic"] = {
            "status": result.diagnostic.status,
            "conflicting_constraints": result.diagnostic.conflicting_constraints,
            "suggestions": result.diagnostic.suggestions,
        }
    return yaml.safe_dump(payload, sort_keys=False)


def _solutions_to_csv(result: SolveResult) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["solution_idx", "resident", "block", "rotation", "fte"])
    for idx, solution in enumerate(result.solutions):
        for resident, blocks in solution.assignments.items():
            for block, rotations in blocks.items():
                for rotation, fte in rotations.items():
                    if fte:
                        writer.writerow([idx, resident, block, rotation, fte])
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="IR/DR scheduling solver using CP-SAT.")
    parser.add_argument(
        "input",
        nargs="?",
        default="ir-scheduler.yml",
        help="Path to YAML input file (default: ir-scheduler.yml).",
    )
    parser.add_argument("-o", "--output", help="Output YAML file path. Defaults to stdout.")
    parser.add_argument(
        "--csv-output",
        default="schedule-output.csv",
        help="CSV output path (default: schedule-output.csv).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(
            f"Input file not found: {args.input}\n"
            "Tip: run the Streamlit app and download to the current directory, "
            "or pass a path explicitly (e.g., python ir_scheduler.py path/to/file.yml).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    schedule_input = load_schedule_input(args.input)
    result = solve_schedule(schedule_input)
    if result.diagnostic and sys.stdin.isatty():
        response = input(
            "Model infeasible. Run fast diagnostic suggestions? [y/N]: "
        ).strip().lower()
        if response.startswith("y"):
            progress_state = {"printed": False}

            def _progress(current: int, total: int, spec: ConstraintSpec, mode: str) -> None:
                if not sys.stdout.isatty():
                    return
                progress_state["printed"] = True
                label = f"{spec.id}->{mode}"
                print(
                    f"Diagnostic suggestions: {current}/{total} {label}",
                    end="\r",
                    file=sys.stderr,
                    flush=True,
                )

            result = solve_schedule(
                schedule_input,
                suggest_relaxations=True,
                shrink_core=False,
                progress_cb=_progress,
            )
            if progress_state["printed"]:
                print("", file=sys.stderr)
    output = _solutions_to_yaml(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)

    csv_text = _solutions_to_csv(result)
    with open(args.csv_output, "w", encoding="utf-8", newline="") as handle:
        handle.write(csv_text)


if __name__ == "__main__":
    main()
