from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from ir_scheduler_constants import ROTATIONS
from ir_scheduler_model import _coerce_int, _coerce_units, _enforce, _resident_groups, _total_units
from ir_scheduler_types import ScheduleInput


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
                self.model.Add(prior_units == sum(self.p[(resident_id, k, "MH-IR")] for k in range(b)))
                prior_zero = self.model.NewBoolVar(f"prior_zero_{resident_id}_{b}")
                self.model.Add(prior_units == 0).OnlyEnforceIf(prior_zero)
                self.model.Add(prior_units >= 1).OnlyEnforceIf(prior_zero.Not())
                ft = self.model.NewBoolVar(f"first_timer_{resident_id}_{b}")
                self.model.AddBoolAnd([self.p[(resident_id, b, "MH-IR")], prior_zero]).OnlyEnforceIf(ft)
                self.model.AddBoolOr(
                    [self.p[(resident_id, b, "MH-IR")].Not(), prior_zero.Not()]
                ).OnlyEnforceIf(ft.Not())
                first_timer[(resident_id, b)] = ft
        self._first_timer = first_timer
        return first_timer

    def constraint_param(self, spec_id: str, key: str, default):
        params = self.schedule_input.constraint_params.get(spec_id, {})
        if not isinstance(params, dict):
            return default
        return params.get(key, default)


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


def _add_forced(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for (resident_id, block, rotation), is_forced in ctx.schedule_input.forced.items():
        if is_forced:
            _enforce(ctx.model.Add(ctx.u[(resident_id, block, rotation)] == 2), assumption)


def _add_no_half_assignments(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident in ctx.schedule_input.residents:
        is_dr = resident.track.startswith("DR")
        for b in range(ctx.num_blocks):
            for rot in ROTATIONS:
                if not is_dr:
                    continue
                _enforce(ctx.model.Add(ctx.u[(resident.resident_id, b, rot)] != 1), assumption)


def _add_no_half_kir_assignments(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident in ctx.schedule_input.residents:
        for b in range(ctx.num_blocks):
            _enforce(ctx.model.Add(ctx.u[(resident.resident_id, b, "KIR")] != 1), assumption)


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


def _add_block_total_zero_or_full(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident in ctx.schedule_input.residents:
        for b in range(ctx.num_blocks):
            total_units = ctx.model.NewIntVar(0, 2, f"total_units_{resident.resident_id}_{b}")
            _enforce(
                ctx.model.Add(
                    total_units == sum(ctx.u[(resident.resident_id, b, rot)] for rot in ROTATIONS)
                ),
                assumption,
            )
            _enforce(ctx.model.AddAllowedAssignments([total_units], [[0], [2]]), assumption)


def _add_coverage_48x_ir(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    op = str(ctx.constraint_param("coverage_48x_ir", "op", "==")).strip()
    target_units = _coerce_units(ctx.constraint_param("coverage_48x_ir", "target_units", 2), 2)
    # Coverage targets are whole-number FTE, represented in half-FTE units.
    target_units = 2 * int(round(target_units / 2))
    for b in range(ctx.num_blocks):
        total = sum(
            ctx.u[(resident.resident_id, b, "48X-IR")] for resident in ctx.schedule_input.residents
        )
        if op == "<=":
            _enforce(ctx.model.Add(total <= target_units), assumption)
        elif op == ">=":
            _enforce(ctx.model.Add(total >= target_units), assumption)
        else:
            _enforce(ctx.model.Add(total == target_units), assumption)


def _add_coverage_48x_ctus(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    op = str(ctx.constraint_param("coverage_48x_ctus", "op", "==")).strip()
    target_units = _coerce_units(ctx.constraint_param("coverage_48x_ctus", "target_units", 2), 2)
    # Coverage targets are whole-number FTE, represented in half-FTE units.
    target_units = 2 * int(round(target_units / 2))
    for b in range(ctx.num_blocks):
        total = sum(
            ctx.u[(resident.resident_id, b, "48X-CT/US")] for resident in ctx.schedule_input.residents
        )
        if op == "<=":
            _enforce(ctx.model.Add(total <= target_units), assumption)
        elif op == ">=":
            _enforce(ctx.model.Add(total >= target_units), assumption)
        else:
            _enforce(ctx.model.Add(total == target_units), assumption)


def _add_mh_total_minmax(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    min_fte = _coerce_int(ctx.constraint_param("mh_total_minmax", "min_fte", 3), 3, min_value=0)
    max_fte = _coerce_int(ctx.constraint_param("mh_total_minmax", "max_fte", 4), 4, min_value=0)
    if min_fte > max_fte:
        min_fte, max_fte = max_fte, min_fte
    for b in range(ctx.num_blocks):
        mh_total = sum(
            ctx.u[(resident.resident_id, b, "MH-IR")] for resident in ctx.schedule_input.residents
        ) + sum(
            ctx.u[(resident.resident_id, b, "MH-CT/US")] for resident in ctx.schedule_input.residents
        )
        _enforce(ctx.model.Add(mh_total >= 2 * min_fte), assumption)
        _enforce(ctx.model.Add(mh_total <= 2 * max_fte), assumption)


def _viva_block_relaxation(ctx: ConstraintContext) -> tuple[int, str] | None:
    mode = str(ctx.schedule_input.constraint_modes.get("viva_block_staffing", "always") or "").lower()
    if mode == "disabled":
        return None
    params = ctx.schedule_input.constraint_params.get("viva_block_staffing", {})
    if not isinstance(params, dict):
        return None
    min_dr = _coerce_int(params.get("min_dr_residents", 3), 3, min_value=0)
    if min_dr < 3:
        return None
    block = _coerce_int(params.get("block", 4), 4, min_value=0, max_value=max(0, ctx.num_blocks - 1))
    choice = str(params.get("relaxation", "") or "").strip().lower()
    if choice not in {"mh_ctus_cap", "first_timer"}:
        return None
    return block, choice


def _add_mh_ctus_cap(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    max_fte = _coerce_int(ctx.constraint_param("mh_ctus_cap", "max_fte", 1), 1, min_value=0)
    viva = _viva_block_relaxation(ctx)
    viva_block = viva[0] if viva and viva[1] == "mh_ctus_cap" else None
    for b in range(ctx.num_blocks):
        cap_fte = max_fte
        if viva_block is not None and b == viva_block:
            cap_fte = max(cap_fte, 2)
        _enforce(
            ctx.model.Add(
                sum(
                    ctx.u[(resident.resident_id, b, "MH-CT/US")]
                    for resident in ctx.schedule_input.residents
                )
                <= 2 * cap_fte
            ),
            assumption,
        )


def _add_kir_cap(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    max_fte = _coerce_int(ctx.constraint_param("kir_cap", "max_fte", 2), 2, min_value=0)
    for b in range(ctx.num_blocks):
        _enforce(
            ctx.model.Add(
                sum(ctx.u[(resident.resident_id, b, "KIR")] for resident in ctx.schedule_input.residents)
                <= 2 * max_fte
            ),
            assumption,
        )


def _add_track_requirements(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    for resident in ctx.schedule_input.residents:
        req = ctx.schedule_input.requirements[resident.track]
        for rot in ROTATIONS:
            units = req[rot]
            _enforce(
                ctx.model.Add(_total_units(ctx.u, resident.resident_id, rot, ctx.num_blocks) == units),
                assumption,
            )


def _add_ir5_mh_min_per_block(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    if not ctx.groups["IR5"]:
        return
    for b in range(ctx.num_blocks):
        _enforce(
            ctx.model.Add(sum(ctx.u[(resident_id, b, "MH-IR")] for resident_id in ctx.groups["IR5"]) >= 2),
            assumption,
        )


def _add_ir4_plus_mh_cap(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    ir_min_year = _coerce_int(
        ctx.constraint_param("ir4_plus_mh_cap", "ir_min_year", 4), 4, min_value=1, max_value=5
    )
    max_fte = _coerce_int(ctx.constraint_param("ir4_plus_mh_cap", "max_fte", 2), 2, min_value=0)
    group: List[str] = []
    for resident in ctx.schedule_input.residents:
        if not isinstance(resident.track, str) or not resident.track.startswith("IR"):
            continue
        try:
            year = int(resident.track[2:])
        except ValueError:
            continue
        if year >= ir_min_year:
            group.append(resident.resident_id)
    for b in range(ctx.num_blocks):
        _enforce(
            ctx.model.Add(sum(ctx.u[(resident_id, b, "MH-IR")] for resident_id in group) <= 2 * max_fte),
            assumption,
        )


def _add_dr1_early_block_mh_prohibited(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    max_block = _coerce_int(ctx.constraint_param("dr1_early_block", "first_n_blocks", 4), 4, min_value=0)
    max_block = min(max_block, ctx.num_blocks)
    for resident_id in ctx.groups["DR1"]:
        for b in range(max_block):
            _enforce(ctx.model.Add(ctx.u[(resident_id, b, "MH-IR")] == 0), assumption)


def _add_ir3_late_block_restrictions(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    after_block = _coerce_int(
        ctx.constraint_param("ir3_late_block", "after_block", 7),
        7,
        min_value=0,
        max_value=ctx.num_blocks,
    )
    raw_rots = ctx.constraint_param("ir3_late_block", "rotations", ["MH-IR", "48X-IR"])
    if isinstance(raw_rots, list):
        rotations = [str(r) for r in raw_rots if str(r) in ROTATIONS]
    elif isinstance(raw_rots, str):
        rotations = [raw_rots] if raw_rots in ROTATIONS else []
    else:
        rotations = ["MH-IR", "48X-IR"]

    # "After block N (block N allowed)" => restrictions apply starting at block N+1.
    start_b = min(after_block + 1, ctx.num_blocks)
    for resident_id in ctx.groups["IR3"]:
        for b in range(start_b, ctx.num_blocks):
            for rot in rotations:
                _enforce(ctx.model.Add(ctx.u[(resident_id, b, rot)] == 0), assumption)


def _add_ir4_off_sicu_rotation(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    ir4_ids = list(ctx.groups.get("IR4") or [])
    if not ir4_ids:
        return

    # Let N be the number of IR4 residents. In blocks 0..N-1 (or 0..num_blocks-1 if fewer blocks),
    # exactly one IR4 is OFF each block, and no IR4 is OFF more than once.
    k = min(len(ir4_ids), ctx.num_blocks)
    if k <= 0:
        return

    is_off: Dict[Tuple[str, int], cp_model.BoolVar] = {}
    for resident_id in ir4_ids:
        for b in range(k):
            off = ctx.model.NewBoolVar(f"ir4_sicu_off_{resident_id}_{b}")
            any_assigned = sum(ctx.p[(resident_id, b, rot)] for rot in ROTATIONS)
            _enforce(ctx.model.Add(any_assigned == 0), assumption).OnlyEnforceIf(off)
            _enforce(ctx.model.Add(any_assigned >= 1), assumption).OnlyEnforceIf(off.Not())
            is_off[(resident_id, b)] = off

    for b in range(k):
        _enforce(ctx.model.Add(sum(is_off[(resident_id, b)] for resident_id in ir4_ids) == 1), assumption)

    for resident_id in ir4_ids:
        _enforce(ctx.model.Add(sum(is_off[(resident_id, b)] for b in range(k)) <= 1), assumption)


def _mh_any_in_block(
    ctx: ConstraintContext,
    *,
    resident_id: str,
    block: int,
    spec_id: str,
    assumption: Optional[cp_model.BoolVar],
) -> cp_model.BoolVar:
    mh_any = ctx.model.NewBoolVar(f"{spec_id}_mh_any_{resident_id}_{block}")
    mh_units = ctx.u[(resident_id, block, "MH-IR")] + ctx.u[(resident_id, block, "MH-CT/US")]
    _enforce(ctx.model.Add(mh_units >= 1), assumption).OnlyEnforceIf(mh_any)
    _enforce(ctx.model.Add(mh_units == 0), assumption).OnlyEnforceIf(mh_any.Not())
    return mh_any


def _add_holiday_block_staffing(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    block = _coerce_int(
        ctx.constraint_param("holiday_block_staffing", "block", 6),
        6,
        min_value=0,
        max_value=max(0, ctx.num_blocks - 1),
    )
    min_residents = _coerce_int(
        ctx.constraint_param("holiday_block_staffing", "min_residents", 4),
        4,
        min_value=0,
    )
    mh_any_vars = [
        _mh_any_in_block(ctx, resident_id=res.resident_id, block=block, spec_id="holiday_block_staffing", assumption=assumption)
        for res in ctx.schedule_input.residents
    ]
    _enforce(ctx.model.Add(sum(mh_any_vars) >= min_residents), assumption)


def _add_viva_block_staffing(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    block = _coerce_int(
        ctx.constraint_param("viva_block_staffing", "block", 4),
        4,
        min_value=0,
        max_value=max(0, ctx.num_blocks - 1),
    )
    min_residents = _coerce_int(
        ctx.constraint_param("viva_block_staffing", "min_residents", 4),
        4,
        min_value=0,
    )
    min_dr_residents = _coerce_int(
        ctx.constraint_param("viva_block_staffing", "min_dr_residents", 3),
        3,
        min_value=0,
    )
    mh_any_by_resident = {
        res.resident_id: _mh_any_in_block(
            ctx,
            resident_id=res.resident_id,
            block=block,
            spec_id="viva_block_staffing",
            assumption=assumption,
        )
        for res in ctx.schedule_input.residents
    }

    dr_ids = list(ctx.groups.get("DR1", [])) + list(ctx.groups.get("DR2", [])) + list(ctx.groups.get("DR3", []))
    _enforce(ctx.model.Add(sum(mh_any_by_resident.values()) >= min_residents), assumption)
    _enforce(ctx.model.Add(sum(mh_any_by_resident[rid] for rid in dr_ids if rid in mh_any_by_resident) >= min_dr_residents), assumption)


def _add_first_timer_hard(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    first_timer = ctx.first_timer()
    viva = _viva_block_relaxation(ctx)
    viva_block = viva[0] if viva and viva[1] == "first_timer" else None
    for b in range(ctx.num_blocks):
        count_b = sum(first_timer[(resident_id, b)] for resident_id in ctx.groups["FIRST_TIMER_CANDIDATES"])
        limit = 2 if viva_block is not None and b == viva_block else 1
        _enforce(ctx.model.Add(count_b <= limit), assumption)


def _add_first_timer_soft(ctx: ConstraintContext) -> List[cp_model.IntVar]:
    first_timer = ctx.first_timer()
    viva = _viva_block_relaxation(ctx)
    viva_block = viva[0] if viva and viva[1] == "first_timer" else None
    excess_vars: List[cp_model.IntVar] = []
    for b in range(ctx.num_blocks):
        count_b = sum(first_timer[(resident_id, b)] for resident_id in ctx.groups["FIRST_TIMER_CANDIDATES"])
        excess = ctx.model.NewIntVar(0, len(ctx.groups["FIRST_TIMER_CANDIDATES"]), f"ft_excess_{b}")
        limit = 2 if viva_block is not None and b == viva_block else 1
        ctx.model.Add(count_b <= limit + excess)
        excess_vars.append(excess)
    return excess_vars


def _add_consec_full_mh_hard(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    max_consecutive = _coerce_int(
        ctx.constraint_param("consec_full_mh", "max_consecutive", 3), 3, min_value=2, max_value=ctx.num_blocks
    )
    full_mh = ctx.full_mh()
    for resident in ctx.schedule_input.residents:
        for b in range(0, ctx.num_blocks - (max_consecutive - 1)):
            window = sum(full_mh[(resident.resident_id, b + k)] for k in range(max_consecutive))
            _enforce(ctx.model.Add(window <= max_consecutive - 1), assumption)


def _add_consec_full_mh_soft(ctx: ConstraintContext) -> List[cp_model.IntVar]:
    max_consecutive = _coerce_int(
        ctx.constraint_param("consec_full_mh", "max_consecutive", 3), 3, min_value=2, max_value=ctx.num_blocks
    )
    full_mh = ctx.full_mh()
    excess_vars: List[cp_model.IntVar] = []
    for resident in ctx.schedule_input.residents:
        for b in range(0, ctx.num_blocks - (max_consecutive - 1)):
            window = sum(full_mh[(resident.resident_id, b + k)] for k in range(max_consecutive))
            excess = ctx.model.NewIntVar(0, 1, f"mh_consec_excess_{resident.resident_id}_{b}")
            ctx.model.Add(window <= (max_consecutive - 1) + excess)
            excess_vars.append(excess)
    return excess_vars


def _add_no_sequential_hard(ctx: ConstraintContext, assumption: Optional[cp_model.BoolVar]):
    # This is about *blocks*, not rotations within a block. A resident may have multiple rotations in a
    # single block (e.g., 0.5 + 0.5 split), and that should still count as "assigned in the block".
    for resident_id in ctx.groups["YEAR1_3"]:
        for b in range(ctx.num_blocks - 1):
            any_b = ctx.model.NewBoolVar(f"any_assign_{resident_id}_{b}")
            any_next = ctx.model.NewBoolVar(f"any_assign_{resident_id}_{b+1}")
            sum_b = sum(ctx.p[(resident_id, b, rot)] for rot in ROTATIONS)
            sum_next = sum(ctx.p[(resident_id, b + 1, rot)] for rot in ROTATIONS)
            _enforce(ctx.model.Add(sum_b >= 1), assumption).OnlyEnforceIf(any_b)
            _enforce(ctx.model.Add(sum_b == 0), assumption).OnlyEnforceIf(any_b.Not())
            _enforce(ctx.model.Add(sum_next >= 1), assumption).OnlyEnforceIf(any_next)
            _enforce(ctx.model.Add(sum_next == 0), assumption).OnlyEnforceIf(any_next.Not())
            _enforce(ctx.model.Add(any_b + any_next <= 1), assumption)


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
        id="block_total_zero_or_full",
        label="Per-block assignment is 0 or 1.0 FTE",
        softenable=False,
        impact=75,
        add_hard=_add_block_total_zero_or_full,
    ),
    ConstraintSpec(
        id="blocked",
        label="Blocked assignments",
        softenable=False,
        impact=20,
        add_hard=_add_blocked,
    ),
    ConstraintSpec(
        id="forced",
        label="Forced assignments",
        softenable=False,
        impact=25,
        add_hard=_add_forced,
    ),
    ConstraintSpec(
        id="no_half_non_ir5",
        label="No half assignments for DR residents",
        softenable=False,
        impact=30,
        add_hard=_add_no_half_assignments,
    ),
    ConstraintSpec(
        id="no_half_kir",
        label="No half assignments on KIR",
        softenable=False,
        impact=35,
        add_hard=_add_no_half_kir_assignments,
    ),
    ConstraintSpec(
        id="coverage_48x_ir",
        label="48X-IR coverage",
        softenable=False,
        impact=100,
        add_hard=_add_coverage_48x_ir,
    ),
    ConstraintSpec(
        id="coverage_48x_ctus",
        label="48X-CT/US coverage",
        softenable=False,
        impact=100,
        add_hard=_add_coverage_48x_ctus,
    ),
    ConstraintSpec(
        id="mh_total_minmax",
        label="MH total coverage range",
        softenable=False,
        impact=90,
        add_hard=_add_mh_total_minmax,
    ),
    ConstraintSpec(
        id="mh_ctus_cap",
        label="MH-CT/US cap",
        softenable=False,
        impact=50,
        add_hard=_add_mh_ctus_cap,
    ),
    ConstraintSpec(
        id="kir_cap",
        label="KIR cap",
        softenable=False,
        impact=50,
        add_hard=_add_kir_cap,
    ),
    ConstraintSpec(
        id="track_requirements",
        label="Per-class (IR/DR) rotation requirements",
        softenable=False,
        impact=90,
        add_hard=_add_track_requirements,
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
        label="Limit senior residents on MH-IR per block",
        softenable=False,
        impact=60,
        add_hard=_add_ir4_plus_mh_cap,
    ),
    ConstraintSpec(
        id="ir4_off_sicu",
        label="IR4 off for SICU",
        softenable=False,
        impact=15,
        add_hard=_add_ir4_off_sicu_rotation,
    ),
    ConstraintSpec(
        id="dr1_early_block",
        label="DR1 no MH-IR in early blocks",
        softenable=False,
        impact=15,
        add_hard=_add_dr1_early_block_mh_prohibited,
    ),
    ConstraintSpec(
        id="ir3_late_block",
        label="IR-3 core studying",
        softenable=False,
        impact=15,
        add_hard=_add_ir3_late_block_restrictions,
    ),
    ConstraintSpec(
        id="holiday_block_staffing",
        label="Holiday block staffing",
        softenable=False,
        impact=45,
        add_hard=_add_holiday_block_staffing,
    ),
    ConstraintSpec(
        id="viva_block_staffing",
        label="VIVA block staffing",
        softenable=False,
        impact=45,
        add_hard=_add_viva_block_staffing,
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
        label="Avoid consecutive full MH-IR blocks",
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

CORE_RULE_IDS = {"one_place", "block_total_zero_or_full", "no_half_non_ir5"}


def _default_mode_for_spec(spec: ConstraintSpec) -> str:
    if spec.id in {"first_timer", "consec_full_mh", "no_sequential_year1_3"}:
        return "always"
    return "if_able" if spec.softenable else "always"


def _constraint_mode(spec: ConstraintSpec, mode_map: Dict[str, str]) -> str:
    return mode_map.get(spec.id, _default_mode_for_spec(spec))


def _apply_constraints(
    model: cp_model.CpModel,
    schedule_input: ScheduleInput,
    u: Dict[Tuple[str, int, str], cp_model.IntVar],
    p: Dict[Tuple[str, int, str], cp_model.IntVar],
) -> tuple[
    Dict[str, List[cp_model.IntVar]],
    Dict[int, ConstraintSpec],
    Dict[int, cp_model.BoolVar],
    List[tuple[cp_model.IntVar, int]],
    Dict[str, cp_model.BoolVar],
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
    relax_terms: List[tuple[cp_model.IntVar, int]] = []
    relax_var_by_id: Dict[str, cp_model.BoolVar] = {}

    for spec in CONSTRAINT_SPECS:
        mode = _constraint_mode(spec, schedule_input.constraint_modes)
        if mode == "if_able" and spec.id in CORE_RULE_IDS:
            mode = "always"
        if mode == "disabled":
            continue
        if mode == "if_able" and spec.softenable and spec.add_soft is not None:
            penalties[spec.id] = spec.add_soft(ctx)
            continue
        if mode == "if_able":
            # "Try" mode for hard-only constraints: enforce unless the solver needs to relax it.
            # We model this by gating the hard constraints with an assumption var that is NOT required,
            # then minimizing how many such constraints are relaxed.
            assumption = model.NewBoolVar(f"a_{spec.id}")
            spec.add_hard(ctx, assumption)
            relaxed = model.NewBoolVar(f"relax_{spec.id}")
            model.Add(assumption + relaxed == 1)
            relax_terms.append((relaxed, 1))
            relax_var_by_id[spec.id] = relaxed
            continue

        assumption = model.NewBoolVar(f"a_{spec.id}")
        model.AddAssumption(assumption)
        assumption_index_map[assumption.Index()] = spec
        assumption_var_map[assumption.Index()] = assumption
        spec.add_hard(ctx, assumption)

    return penalties, assumption_index_map, assumption_var_map, relax_terms, relax_var_by_id
