from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from ir_scheduler_constants import ROTATIONS
from ir_scheduler_types import Resident


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


def _coerce_int(value, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), out)
    if max_value is not None:
        out = min(int(max_value), out)
    return out


def _coerce_units(value, default_units: int, *, min_units: int = 0, max_units: int = 9999) -> int:
    # Units are half-FTE integers: 1 unit = 0.5 FTE.
    if isinstance(value, int):
        units = value
    elif isinstance(value, float):
        units = int(round(value * 2))
    else:
        try:
            units = int(value)
        except (TypeError, ValueError):
            try:
                units = int(round(float(value) * 2))
            except (TypeError, ValueError):
                units = default_units
    units = max(min_units, min(max_units, int(units)))
    return units

