import tempfile
import unittest

import yaml
from ortools.sat.python import cp_model

from ir_scheduler import (
    CONSTRAINT_SPECS,
    ScheduleError,
    _optimize_and_fix,
    expand_residents,
    load_schedule_input,
    solve_schedule,
)


class SchedulerTests(unittest.TestCase):
    def test_no_sequential_year1_3_allows_split_within_block(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["one_place"] = "always"
        modes["block_total_zero_or_full"] = "always"
        modes["track_requirements"] = "always"
        modes["no_sequential_year1_3"] = "always"

        # IR2 has half-block requirements, which forces a 0.5/0.5 split within a single block.
        data = {
            "blocks": 2,
            "residents": [{"id": "ir2a", "track": "IR2"}],
            "requirements": {
                "DR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0.5, "48X-CT/US": 0.5},
                "IR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR4": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR5": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
            },
            "gui": {"constraints": {"modes": modes}},
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            path = handle.name

        schedule_input = load_schedule_input(path)
        result = solve_schedule(schedule_input)
        self.assertTrue(result.solutions)

    def test_expand_residents_defaults(self):
        gui = {
            "IR": {
                "IR1": ["Gaburak", "Miller"],
                "IR2": ["Qi", "Verst"],
                "IR3": ["Madsen", "Mahmoud"],
                "IR4": ["Javan", "Virk"],
                "IR5": ["Brock", "Katz"],
            },
            "DR_counts": {"DR1": 2, "DR2": 1, "DR3": 0},
        }
        residents = expand_residents(gui)
        ir1_ids = [r["id"] for r in residents if r["track"] == "IR1"]
        self.assertEqual(ir1_ids, ["Gaburak", "Miller"])
        dr1_ids = [r for r in residents if r["track"] == "DR1"]
        self.assertEqual(len(dr1_ids), 2)
        self.assertEqual({r["id"] for r in dr1_ids}, {"DR1-1", "DR1-2"})

    def test_modes_disable_blocked(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["track_requirements"] = "always"
        modes["blocked"] = "always"

        data = {
            "blocks": 1,
            "residents": [{"id": "dr1a", "track": "DR1"}],
            "blocked": [{"resident": "dr1a", "block": 0, "rotation": "MH-IR"}],
            "requirements": {
                "DR1": {"KIR": 0, "MH-IR": 1, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR4": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR5": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
            },
            "gui": {"constraints": {"modes": modes}},
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            path = handle.name

        schedule_input = load_schedule_input(path)
        result = solve_schedule(schedule_input)
        self.assertFalse(result.solutions)
        self.assertIsNotNone(result.diagnostic)

        data["gui"]["constraints"]["modes"]["blocked"] = "disabled"
        with tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            path = handle.name

        schedule_input = load_schedule_input(path)
        result = solve_schedule(schedule_input)
        self.assertTrue(result.solutions)

    def test_priority_ordering_changes_solution(self):
        model = cp_model.CpModel()
        x = model.NewIntVar(0, 1, "x")
        y = model.NewIntVar(0, 1, "y")
        model.Add(x + y >= 1)
        solver = cp_model.CpSolver()

        status, _ = _optimize_and_fix(model, solver, [(x, 1)])
        self.assertIn(status, (cp_model.OPTIMAL, cp_model.FEASIBLE))
        status, _ = _optimize_and_fix(model, solver, [(y, 1)])
        self.assertIn(status, (cp_model.OPTIMAL, cp_model.FEASIBLE))
        self.assertEqual(solver.Value(x), 0)
        self.assertEqual(solver.Value(y), 1)

        model = cp_model.CpModel()
        x = model.NewIntVar(0, 1, "x")
        y = model.NewIntVar(0, 1, "y")
        model.Add(x + y >= 1)
        solver = cp_model.CpSolver()

        status, _ = _optimize_and_fix(model, solver, [(y, 1)])
        self.assertIn(status, (cp_model.OPTIMAL, cp_model.FEASIBLE))
        status, _ = _optimize_and_fix(model, solver, [(x, 1)])
        self.assertIn(status, (cp_model.OPTIMAL, cp_model.FEASIBLE))
        self.assertEqual(solver.Value(x), 1)
        self.assertEqual(solver.Value(y), 0)

    def test_infeasibility_diagnostic_nonempty(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["track_requirements"] = "always"
        modes["blocked"] = "always"

        data = {
            "blocks": 1,
            "residents": [{"id": "dr1a", "track": "DR1"}],
            "blocked": [{"resident": "dr1a", "block": 0, "rotation": "MH-IR"}],
            "requirements": {
                "DR1": {"KIR": 0, "MH-IR": 1, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR4": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR5": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
            },
            "gui": {"constraints": {"modes": modes}},
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            path = handle.name

        schedule_input = load_schedule_input(path)
        result = solve_schedule(schedule_input)
        self.assertIsNotNone(result.diagnostic)
        self.assertTrue(result.diagnostic.conflicting_constraints)

    def test_forced_enforces_assignment_even_without_requirements(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["one_place"] = "always"
        modes["forced"] = "always"

        data = {
            "blocks": 1,
            "residents": [{"id": "dr1a", "track": "DR1"}],
            "forced": [{"resident": "dr1a", "block": 0, "rotation": "KIR"}],
            "requirements": {
                "DR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR4": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR5": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
            },
            "gui": {"constraints": {"modes": modes}},
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            path = handle.name

        schedule_input = load_schedule_input(path)
        result = solve_schedule(schedule_input)
        self.assertTrue(result.solutions)

        sol = result.solutions[0]
        self.assertEqual(sol.assignments["dr1a"]["B0"].get("KIR"), 1.0)

    def test_forced_conflicts_with_blocked_raises(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["one_place"] = "always"
        modes["forced"] = "always"
        modes["blocked"] = "always"

        data = {
            "blocks": 1,
            "residents": [{"id": "dr1a", "track": "DR1"}],
            "forced": [{"resident": "dr1a", "block": 0, "rotation": "KIR"}],
            "blocked": [{"resident": "dr1a", "block": 0, "rotation": "KIR"}],
            "requirements": {
                "DR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR4": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR5": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
            },
            "gui": {"constraints": {"modes": modes}},
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            path = handle.name

        with self.assertRaises(ScheduleError):
            load_schedule_input(path)

    def test_ir_kir_requirement_cannot_be_half(self):
        data = {
            "blocks": 1,
            "residents": [{"id": "ir1a", "track": "IR1"}],
            "requirements": {
                "DR1": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "DR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR1": {"KIR": 0.5, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR2": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR3": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR4": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
                "IR5": {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0},
            },
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            path = handle.name

        with self.assertRaises(ScheduleError):
            load_schedule_input(path)


if __name__ == "__main__":
    unittest.main()
