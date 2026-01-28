import tempfile
import unittest

import yaml
from ortools.sat.python import cp_model

from ir_scheduler import (
    CONSTRAINT_SPECS,
    _optimize_and_fix,
    expand_residents,
    load_schedule_input,
    solve_schedule,
)


class SchedulerTests(unittest.TestCase):
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
        modes["dr1_totals"] = "always"
        modes["blocked"] = "always"

        data = {
            "blocks": 1,
            "residents": [{"id": "dr1a", "track": "DR1"}],
            "blocked": [{"resident": "dr1a", "block": 0, "rotation": "MH-IR"}],
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
        modes["dr1_totals"] = "always"
        modes["blocked"] = "always"

        data = {
            "blocks": 1,
            "residents": [{"id": "dr1a", "track": "DR1"}],
            "blocked": [{"resident": "dr1a", "block": 0, "rotation": "MH-IR"}],
            "gui": {"constraints": {"modes": modes}},
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".yml", delete=False) as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            path = handle.name

        schedule_input = load_schedule_input(path)
        result = solve_schedule(schedule_input)
        self.assertIsNotNone(result.diagnostic)
        self.assertTrue(result.diagnostic.conflicting_constraints)


if __name__ == "__main__":
    unittest.main()
