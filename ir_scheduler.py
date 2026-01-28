import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


@dataclass
class Solution:
    assignments: Dict[str, Dict[str, Dict[str, float]]]
    objective: Dict[str, int]


class ScheduleError(Exception):
    pass


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
    residents = []
    raw = data.get("residents", [])
    if not raw:
        raise ScheduleError("Input must include at least one resident.")
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


def load_schedule_input(path: str) -> ScheduleInput:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    block_labels = _parse_block_labels(data)
    residents = _parse_residents(data)
    blocked = _parse_blocked(data, block_labels)
    weights = _parse_weights(data)
    num_solutions = int(data.get("num_solutions", 1))
    return ScheduleInput(
        block_labels=block_labels,
        residents=residents,
        blocked=blocked,
        weights=weights,
        num_solutions=num_solutions,
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
    return groups


def _total_units(u: Dict[Tuple[str, int, str], cp_model.IntVar], resident_id: str, rot: str, num_blocks: int):
    return sum(u[(resident_id, b, rot)] for b in range(num_blocks))


def _add_core_constraints(
    model: cp_model.CpModel,
    schedule_input: ScheduleInput,
    u: Dict[Tuple[str, int, str], cp_model.IntVar],
):
    num_blocks = len(schedule_input.block_labels)
    residents = schedule_input.residents
    groups = _resident_groups(residents)

    for resident in residents:
        for b in range(num_blocks):
            model.Add(sum(u[(resident.resident_id, b, rot)] for rot in ROTATIONS) <= 2)

    for (resident_id, block, rotation), is_blocked in schedule_input.blocked.items():
        if is_blocked:
            model.Add(u[(resident_id, block, rotation)] == 0)

    for resident in residents:
        is_ir5 = resident.track == "IR5"
        for b in range(num_blocks):
            for rot in ROTATIONS:
                if is_ir5 and rot in {"MH-IR", "48X-IR"}:
                    continue
                model.Add(u[(resident.resident_id, b, rot)] != 1)

    for resident_id in groups["IR5"]:
        for b in range(num_blocks):
            mh = u[(resident_id, b, "MH-IR")]
            x48 = u[(resident_id, b, "48X-IR")]
            is_half_mh = model.NewBoolVar(f"is_half_mh_{resident_id}_{b}")
            is_half_x48 = model.NewBoolVar(f"is_half_x48_{resident_id}_{b}")
            model.Add(mh == 1).OnlyEnforceIf(is_half_mh)
            model.Add(mh != 1).OnlyEnforceIf(is_half_mh.Not())
            model.Add(x48 == 1).OnlyEnforceIf(is_half_x48)
            model.Add(x48 != 1).OnlyEnforceIf(is_half_x48.Not())
            model.Add(is_half_mh == is_half_x48)

    for b in range(num_blocks):
        model.Add(sum(u[(resident.resident_id, b, "48X-IR")] for resident in residents) == 2)
        model.Add(sum(u[(resident.resident_id, b, "48X-CT/US")] for resident in residents) == 2)

        mh_total = sum(u[(resident.resident_id, b, "MH-IR")] for resident in residents) + sum(
            u[(resident.resident_id, b, "MH-CT/US")] for resident in residents
        )
        model.Add(mh_total >= 6)
        model.Add(mh_total <= 8)

        model.Add(sum(u[(resident.resident_id, b, "MH-CT/US")] for resident in residents) <= 2)
        model.Add(sum(u[(resident.resident_id, b, "KIR")] for resident in residents) <= 4)

    for resident_id in groups["DR1"]:
        model.Add(_total_units(u, resident_id, "MH-IR", num_blocks) == 2)
        for rot in ["KIR", "MH-CT/US", "48X-IR", "48X-CT/US"]:
            model.Add(_total_units(u, resident_id, rot, num_blocks) == 0)

    for resident_id in groups["DR2"]:
        model.Add(_total_units(u, resident_id, "MH-CT/US", num_blocks) == 2)
        for rot in ["KIR", "MH-IR", "48X-IR", "48X-CT/US"]:
            model.Add(_total_units(u, resident_id, rot, num_blocks) == 0)

    for resident_id in groups["DR3"]:
        model.Add(_total_units(u, resident_id, "48X-CT/US", num_blocks) == 2)
        for rot in ["KIR", "MH-IR", "MH-CT/US", "48X-IR"]:
            model.Add(_total_units(u, resident_id, rot, num_blocks) == 0)

    for resident_id in groups["IR1"]:
        model.Add(_total_units(u, resident_id, "MH-IR", num_blocks) == 2)
        model.Add(_total_units(u, resident_id, "48X-IR", num_blocks) == 2)
        model.Add(_total_units(u, resident_id, "48X-CT/US", num_blocks) == 2)
        model.Add(_total_units(u, resident_id, "KIR", num_blocks) == 0)
        model.Add(_total_units(u, resident_id, "MH-CT/US", num_blocks) == 0)

    for resident_id in groups["IR2"]:
        model.Add(_total_units(u, resident_id, "MH-IR", num_blocks) == 4)
        model.Add(_total_units(u, resident_id, "48X-IR", num_blocks) == 2)
        model.Add(_total_units(u, resident_id, "KIR", num_blocks) == 0)
        model.Add(_total_units(u, resident_id, "MH-CT/US", num_blocks) == 0)
        model.Add(_total_units(u, resident_id, "48X-CT/US", num_blocks) == 0)

    for resident_id in groups["IR3"]:
        model.Add(_total_units(u, resident_id, "MH-IR", num_blocks) == 2)
        model.Add(_total_units(u, resident_id, "48X-IR", num_blocks) == 2)
        model.Add(_total_units(u, resident_id, "48X-CT/US", num_blocks) == 2)
        model.Add(_total_units(u, resident_id, "KIR", num_blocks) == 0)
        model.Add(_total_units(u, resident_id, "MH-CT/US", num_blocks) == 0)

    for resident_id in groups["IR4"]:
        model.Add(_total_units(u, resident_id, "KIR", num_blocks) == 6)
        model.Add(_total_units(u, resident_id, "MH-IR", num_blocks) == 6)
        model.Add(_total_units(u, resident_id, "48X-IR", num_blocks) == 0)
        model.Add(_total_units(u, resident_id, "48X-CT/US", num_blocks) == 0)
        model.Add(_total_units(u, resident_id, "MH-CT/US", num_blocks) == 0)

    for resident_id in groups["IR5"]:
        model.Add(
            sum(_total_units(u, resident_id, rot, num_blocks) for rot in ROTATIONS) == 2 * num_blocks
        )
        model.Add(_total_units(u, resident_id, "KIR", num_blocks) == 6)
        model.Add(_total_units(u, resident_id, "48X-IR", num_blocks) >= 4)
        model.Add(_total_units(u, resident_id, "MH-CT/US", num_blocks) == 0)

    for b in range(num_blocks):
        model.Add(sum(u[(resident_id, b, "MH-IR")] for resident_id in groups["IR5"]) >= 2)
        model.Add(sum(u[(resident_id, b, "MH-IR")] for resident_id in groups["IR4_PLUS"]) <= 4)

    max_block = min(4, num_blocks)
    for resident_id in groups["DR1"]:
        for b in range(max_block):
            model.Add(u[(resident_id, b, "MH-IR")] == 0)

    if num_blocks >= 13:
        for resident_id in groups["IR3"]:
            for b in range(7, min(13, num_blocks)):
                model.Add(u[(resident_id, b, "MH-IR")] == 0)
                model.Add(u[(resident_id, b, "48X-IR")] == 0)


def _add_soft_constraints(
    model: cp_model.CpModel,
    schedule_input: ScheduleInput,
    u: Dict[Tuple[str, int, str], cp_model.IntVar],
    p: Dict[Tuple[str, int, str], cp_model.IntVar],
):
    num_blocks = len(schedule_input.block_labels)
    residents = schedule_input.residents
    groups = _resident_groups(residents)
    first_timer_excess: Dict[int, cp_model.IntVar] = {}

    first_timer: Dict[Tuple[str, int], cp_model.IntVar] = {}
    for resident_id in groups["FIRST_TIMER_CANDIDATES"]:
        for b in range(num_blocks):
            prior_units = model.NewIntVar(0, num_blocks, f"prior_mh_{resident_id}_{b}")
            model.Add(
                prior_units == sum(p[(resident_id, k, "MH-IR")] for k in range(b))
            )
            prior_zero = model.NewBoolVar(f"prior_zero_{resident_id}_{b}")
            model.Add(prior_units == 0).OnlyEnforceIf(prior_zero)
            model.Add(prior_units >= 1).OnlyEnforceIf(prior_zero.Not())
            ft = model.NewBoolVar(f"first_timer_{resident_id}_{b}")
            model.AddBoolAnd([p[(resident_id, b, "MH-IR")], prior_zero]).OnlyEnforceIf(ft)
            model.AddBoolOr([p[(resident_id, b, "MH-IR")].Not(), prior_zero.Not()]).OnlyEnforceIf(
                ft.Not()
            )
            first_timer[(resident_id, b)] = ft

    for b in range(num_blocks):
        count_b = sum(first_timer[(resident_id, b)] for resident_id in groups["FIRST_TIMER_CANDIDATES"])
        excess = model.NewIntVar(0, len(groups["FIRST_TIMER_CANDIDATES"]), f"ft_excess_{b}")
        model.Add(count_b <= 1 + excess)
        first_timer_excess[b] = excess

    full_mh: Dict[Tuple[str, int], cp_model.IntVar] = {}
    for resident in residents:
        for b in range(num_blocks):
            fm = model.NewBoolVar(f"full_mh_{resident.resident_id}_{b}")
            model.Add(u[(resident.resident_id, b, "MH-IR")] == 2).OnlyEnforceIf(fm)
            model.Add(u[(resident.resident_id, b, "MH-IR")] <= 1).OnlyEnforceIf(fm.Not())
            full_mh[(resident.resident_id, b)] = fm

    consec_excess: List[cp_model.IntVar] = []
    for resident in residents:
        for b in range(0, num_blocks - 2):
            window = full_mh[(resident.resident_id, b)]
            window += full_mh[(resident.resident_id, b + 1)]
            window += full_mh[(resident.resident_id, b + 2)]
            excess = model.NewIntVar(0, 1, f"mh3_excess_{resident.resident_id}_{b}")
            model.Add(window <= 2 + excess)
            consec_excess.append(excess)

    adj: List[cp_model.IntVar] = []
    for resident in residents:
        for b in range(num_blocks - 1):
            z = model.NewBoolVar(f"adj_full_mh_{resident.resident_id}_{b}")
            model.AddBoolAnd([full_mh[(resident.resident_id, b)], full_mh[(resident.resident_id, b + 1)]]).OnlyEnforceIf(z)
            model.AddBoolOr([
                full_mh[(resident.resident_id, b)].Not(),
                full_mh[(resident.resident_id, b + 1)].Not(),
            ]).OnlyEnforceIf(z.Not())
            adj.append(z)

    penalties = {
        "consec_excess": consec_excess,
        "first_timer_excess": list(first_timer_excess.values()),
        "adj": adj,
    }
    return penalties


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

    objective = {
        "consec_excess": int(sum(solver.Value(var) for var in penalties["consec_excess"])),
        "first_timer_excess": int(
            sum(solver.Value(var) for var in penalties["first_timer_excess"])
        ),
        "adj": int(sum(solver.Value(var) for var in penalties["adj"])),
    }
    return Solution(assignments=assignments, objective=objective)


def solve_schedule(schedule_input: ScheduleInput) -> List[Solution]:
    model = cp_model.CpModel()
    u, p = _build_variables(model, schedule_input.residents, schedule_input.block_labels)
    _add_core_constraints(model, schedule_input, u)
    penalties = _add_soft_constraints(model, schedule_input, u, p)

    groups = _resident_groups(schedule_input.residents)
    ir5_ids = groups["IR5"]
    diff = None
    if len(ir5_ids) == 2:
        num_blocks = len(schedule_input.block_labels)
        t0 = _total_units(u, ir5_ids[0], "48X-IR", num_blocks)
        t1 = _total_units(u, ir5_ids[1], "48X-IR", num_blocks)
        diff = model.NewIntVar(0, 2 * num_blocks, "ir5_48x_ir_diff")
        model.AddAbsEquality(diff, t0 - t1)
        model.Minimize(diff)
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []
        best_diff = solver.Value(diff)
        model.Add(diff == best_diff)

    weights = schedule_input.weights
    model.Minimize(
        weights["consec"] * sum(penalties["consec_excess"])
        + weights["first_timer"] * sum(penalties["first_timer_excess"])
        + weights["adj"] * sum(penalties["adj"])
    )

    solutions: List[Solution] = []
    solver = cp_model.CpSolver()
    while len(solutions) < schedule_input.num_solutions:
        status = solver.Solve(model)
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
    return solutions


def _solutions_to_yaml(solutions: List[Solution]) -> str:
    payload = []
    for solution in solutions:
        payload.append(
            {
                "objective": solution.objective,
                "assignments": solution.assignments,
            }
        )
    return yaml.safe_dump({"solutions": payload}, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="IR/DR scheduling solver using CP-SAT.")
    parser.add_argument("input", help="Path to YAML input file.")
    parser.add_argument("-o", "--output", help="Output YAML file path. Defaults to stdout.")
    args = parser.parse_args()

    schedule_input = load_schedule_input(args.input)
    solutions = solve_schedule(schedule_input)
    output = _solutions_to_yaml(solutions)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
