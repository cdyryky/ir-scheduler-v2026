from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ortools.sat.python import cp_model

from ir_scheduler_constants import ROTATIONS
from ir_scheduler_constraints import (
    CONSTRAINT_SPECS,
    CORE_RULE_IDS,
    SPEC_BY_ID,
    ConstraintSpec,
    _apply_constraints,
    _constraint_mode,
)
from ir_scheduler_model import _build_variables, _resident_groups, _total_units
from ir_scheduler_types import Diagnostic, ScheduleInput, Solution, SolveResult


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
        forced=schedule_input.forced,
        weights=schedule_input.weights,
        num_solutions=schedule_input.num_solutions,
        constraint_modes=modes,
        soft_priority=schedule_input.soft_priority,
        constraint_params=schedule_input.constraint_params,
        requirements=schedule_input.requirements,
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
        mode = "disabled" if spec.id in CORE_RULE_IDS else "if_able"
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
        suggestions.append({"id": spec.id, "label": spec.label, "mode": mode, "verified": verified})
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
    return Diagnostic(status="INFEASIBLE", conflicting_constraints=conflicts, suggestions=suggestions)


def _extract_solution(
    solver: cp_model.CpSolver,
    schedule_input: ScheduleInput,
    u: Dict[Tuple[str, int, str], cp_model.IntVar],
    penalties: Dict[str, List[cp_model.IntVar]],
    relax_var_by_id: Dict[str, cp_model.BoolVar],
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

    relaxed_ids = [cid for cid, var in relax_var_by_id.items() if int(solver.Value(var)) == 1]
    if relaxed_ids:
        objective["try_relaxations"] = int(len(relaxed_ids))
        for cid in sorted(relaxed_ids, key=lambda s: str(s).casefold()):
            objective[f"relax_{cid}"] = 1

    return Solution(assignments=assignments, objective=objective)


def solve_schedule(
    schedule_input: ScheduleInput,
    suggest_relaxations: bool = False,
    shrink_core: bool = False,
    progress_cb: Optional[Callable[[int, int, ConstraintSpec, str], None]] = None,
) -> SolveResult:
    model = cp_model.CpModel()
    u, p = _build_variables(model, schedule_input.residents, schedule_input.block_labels)
    penalties, assumption_index_map, assumption_var_map, relax_terms, relax_var_by_id = _apply_constraints(
        model, schedule_input, u, p
    )

    solver = cp_model.CpSolver()

    if relax_terms:
        status, _ = _optimize_and_fix(model, solver, relax_terms)
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
        solutions.append(_extract_solution(solver, schedule_input, u, penalties, relax_var_by_id))
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
