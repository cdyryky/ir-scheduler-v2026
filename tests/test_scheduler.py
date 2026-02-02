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
    load_schedule_input_from_data,
    solve_schedule,
)


class SchedulerTests(unittest.TestCase):
    def test_viva_relaxation_allows_two_first_timers_on_mh_ir(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["forced"] = "always"
        modes["first_timer"] = "always"
        modes["viva_block_staffing"] = "always"

        requirements = {
            track: {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0}
            for track in ["DR1", "DR2", "DR3", "IR1", "IR2", "IR3", "IR4", "IR5"]
        }

        data = {
            "blocks": 1,
            "residents": [
                {"id": "ir1a", "track": "IR1"},
                {"id": "ir1b", "track": "IR1"},
                {"id": "dr2a", "track": "DR2"},
                {"id": "dr2b", "track": "DR2"},
                {"id": "dr2c", "track": "DR2"},
            ],
            "forced": [
                {"resident": "ir1a", "block": 0, "rotation": "MH-IR"},
                {"resident": "ir1b", "block": 0, "rotation": "MH-IR"},
                {"resident": "dr2a", "block": 0, "rotation": "MH-IR"},
                {"resident": "dr2b", "block": 0, "rotation": "MH-IR"},
                {"resident": "dr2c", "block": 0, "rotation": "MH-IR"},
            ],
            "requirements": requirements,
            "gui": {
                "constraints": {
                    "modes": modes,
                    "params": {
                        "viva_block_staffing": {
                            "block": 0,
                            "min_residents": 4,
                            "min_dr_residents": 3,
                            "relaxation": "first_timer",
                        }
                    },
                }
            },
        }

        schedule_input = load_schedule_input_from_data(data)
        result = solve_schedule(schedule_input)
        self.assertTrue(result.solutions)

        # Without the VIVA relaxation, first_timer hard-limit (<=1) makes this infeasible.
        data["gui"]["constraints"]["params"]["viva_block_staffing"]["relaxation"] = "mh_ctus_cap"
        schedule_input = load_schedule_input_from_data(data)
        result = solve_schedule(schedule_input)
        self.assertFalse(result.solutions)
        self.assertIsNotNone(result.diagnostic)

    def test_viva_relaxation_does_not_allow_three_first_timers_on_mh_ir(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["forced"] = "always"
        modes["first_timer"] = "always"
        modes["viva_block_staffing"] = "always"

        requirements = {
            track: {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0}
            for track in ["DR1", "DR2", "DR3", "IR1", "IR2", "IR3", "IR4", "IR5"]
        }

        data = {
            "blocks": 1,
            "residents": [
                {"id": "ir1a", "track": "IR1"},
                {"id": "ir1b", "track": "IR1"},
                {"id": "ir1c", "track": "IR1"},
                {"id": "dr2a", "track": "DR2"},
                {"id": "dr2b", "track": "DR2"},
                {"id": "dr2c", "track": "DR2"},
            ],
            "forced": [
                {"resident": "ir1a", "block": 0, "rotation": "MH-IR"},
                {"resident": "ir1b", "block": 0, "rotation": "MH-IR"},
                {"resident": "ir1c", "block": 0, "rotation": "MH-IR"},
                {"resident": "dr2a", "block": 0, "rotation": "MH-IR"},
                {"resident": "dr2b", "block": 0, "rotation": "MH-IR"},
                {"resident": "dr2c", "block": 0, "rotation": "MH-IR"},
            ],
            "requirements": requirements,
            "gui": {
                "constraints": {
                    "modes": modes,
                    "params": {
                        "viva_block_staffing": {
                            "block": 0,
                            "min_residents": 4,
                            "min_dr_residents": 3,
                            "relaxation": "first_timer",
                        }
                    },
                }
            },
        }

        # Even with VIVA relaxation, first_timer should only relax from <=1 to <=2 (not unlimited).
        schedule_input = load_schedule_input_from_data(data)
        result = solve_schedule(schedule_input)
        self.assertFalse(result.solutions)
        self.assertIsNotNone(result.diagnostic)

    def test_viva_relaxation_allows_two_mh_ctus(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["forced"] = "always"
        modes["mh_ctus_cap"] = "always"
        modes["viva_block_staffing"] = "always"

        requirements = {
            track: {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0}
            for track in ["DR1", "DR2", "DR3", "IR1", "IR2", "IR3", "IR4", "IR5"]
        }

        data = {
            "blocks": 1,
            "residents": [
                {"id": "dr2a", "track": "DR2"},
                {"id": "dr2b", "track": "DR2"},
                {"id": "dr2c", "track": "DR2"},
                {"id": "ir4a", "track": "IR4"},
            ],
            "forced": [
                {"resident": "dr2a", "block": 0, "rotation": "MH-CT/US"},
                {"resident": "dr2b", "block": 0, "rotation": "MH-CT/US"},
                {"resident": "dr2c", "block": 0, "rotation": "MH-IR"},
                {"resident": "ir4a", "block": 0, "rotation": "MH-IR"},
            ],
            "requirements": requirements,
            "gui": {
                "constraints": {
                    "modes": modes,
                    "params": {
                        "mh_ctus_cap": {"max_fte": 1},
                        "viva_block_staffing": {
                            "block": 0,
                            "min_residents": 4,
                            "min_dr_residents": 3,
                            "relaxation": "mh_ctus_cap",
                        },
                    },
                }
            },
        }

        schedule_input = load_schedule_input_from_data(data)
        result = solve_schedule(schedule_input)
        self.assertTrue(result.solutions)

        # Without the VIVA relaxation, mh_ctus_cap (<=1.0 FTE) makes this infeasible.
        data["gui"]["constraints"]["params"]["viva_block_staffing"]["relaxation"] = "first_timer"
        schedule_input = load_schedule_input_from_data(data)
        result = solve_schedule(schedule_input)
        self.assertFalse(result.solutions)
        self.assertIsNotNone(result.diagnostic)

    def test_viva_relaxation_does_not_allow_three_mh_ctus(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["forced"] = "always"
        modes["mh_ctus_cap"] = "always"
        modes["viva_block_staffing"] = "always"

        requirements = {
            track: {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0}
            for track in ["DR1", "DR2", "DR3", "IR1", "IR2", "IR3", "IR4", "IR5"]
        }

        data = {
            "blocks": 1,
            "residents": [
                {"id": "dr2a", "track": "DR2"},
                {"id": "dr2b", "track": "DR2"},
                {"id": "dr2c", "track": "DR2"},
                {"id": "dr2d", "track": "DR2"},
                {"id": "ir4a", "track": "IR4"},
            ],
            "forced": [
                {"resident": "dr2a", "block": 0, "rotation": "MH-CT/US"},
                {"resident": "dr2b", "block": 0, "rotation": "MH-CT/US"},
                {"resident": "dr2c", "block": 0, "rotation": "MH-CT/US"},
                {"resident": "dr2d", "block": 0, "rotation": "MH-IR"},
                {"resident": "ir4a", "block": 0, "rotation": "MH-IR"},
            ],
            "requirements": requirements,
            "gui": {
                "constraints": {
                    "modes": modes,
                    "params": {
                        "mh_ctus_cap": {"max_fte": 1},
                        "viva_block_staffing": {
                            "block": 0,
                            "min_residents": 4,
                            "min_dr_residents": 3,
                            "relaxation": "mh_ctus_cap",
                        },
                    },
                }
            },
        }

        # Even with VIVA relaxation, mh_ctus_cap should only relax from <=1.0 FTE to <=2.0 FTE.
        schedule_input = load_schedule_input_from_data(data)
        result = solve_schedule(schedule_input)
        self.assertFalse(result.solutions)
        self.assertIsNotNone(result.diagnostic)

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

    def test_ir4_off_sicu_rotates_one_off_per_block(self):
        modes = {spec.id: "disabled" for spec in CONSTRAINT_SPECS}
        modes["one_place"] = "always"
        modes["block_total_zero_or_full"] = "always"
        modes["ir4_off_sicu"] = "always"

        requirements = {
            track: {"KIR": 0, "MH-IR": 0, "MH-CT/US": 0, "48X-IR": 0, "48X-CT/US": 0}
            for track in ["DR1", "DR2", "DR3", "IR1", "IR2", "IR3", "IR4", "IR5"]
        }

        data = {
            "blocks": 3,
            "residents": [
                {"id": "ir4a", "track": "IR4"},
                {"id": "ir4b", "track": "IR4"},
                {"id": "ir4c", "track": "IR4"},
            ],
            "requirements": requirements,
            "gui": {"constraints": {"modes": modes}},
        }

        schedule_input = load_schedule_input_from_data(data)
        result = solve_schedule(schedule_input)
        self.assertTrue(result.solutions)

        sol = result.solutions[0]

        def _is_off(resident_id: str, block_label: str) -> bool:
            rotations = sol.assignments.get(resident_id, {}).get(block_label, {})
            return not any(rotations.values())

        for b in range(3):
            block = f"B{b}"
            off_ids = [rid for rid in ["ir4a", "ir4b", "ir4c"] if _is_off(rid, block)]
            self.assertEqual(len(off_ids), 1, msg=f"Expected exactly one IR4 off in {block}: {off_ids}")

        for rid in ["ir4a", "ir4b", "ir4c"]:
            off_count = sum(_is_off(rid, f"B{b}") for b in range(3))
            self.assertEqual(off_count, 1, msg=f"Expected {rid} off exactly once in B0..B2")

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
