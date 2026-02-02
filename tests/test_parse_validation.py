import unittest

from ir_scheduler import ScheduleError, load_schedule_input_from_data


def _zero_requirements() -> dict:
    tracks = ["DR1", "DR2", "DR3", "IR1", "IR2", "IR3", "IR4", "IR5"]
    rotations = ["KIR", "MH-IR", "MH-CT/US", "48X-IR", "48X-CT/US"]
    return {track: {rot: 0 for rot in rotations} for track in tracks}


class ParseValidationTests(unittest.TestCase):
    def test_block_index_out_of_range_in_blocked_raises_schedule_error(self):
        data = {
            "blocks": 1,
            "residents": [{"id": "ir1a", "track": "IR1"}],
            "requirements": _zero_requirements(),
            "blocked": [{"resident": "ir1a", "block": 3, "rotation": "MH-IR"}],
        }
        with self.assertRaisesRegex(ScheduleError, "Block index out of range"):
            load_schedule_input_from_data(data)

    def test_block_index_out_of_range_in_forced_raises_schedule_error(self):
        data = {
            "blocks": 1,
            "residents": [{"id": "ir1a", "track": "IR1"}],
            "requirements": _zero_requirements(),
            "forced": [{"resident": "ir1a", "block": 3, "rotation": "MH-IR"}],
        }
        with self.assertRaisesRegex(ScheduleError, "Block index out of range"):
            load_schedule_input_from_data(data)

    def test_blocked_list_non_mapping_entry_raises_schedule_error(self):
        data = {
            "blocks": 1,
            "residents": [{"id": "ir1a", "track": "IR1"}],
            "requirements": _zero_requirements(),
            "blocked": ["not-a-dict"],
        }
        with self.assertRaisesRegex(ScheduleError, "Blocked entries must be mappings"):
            load_schedule_input_from_data(data)

    def test_unknown_resident_in_blocked_raises_schedule_error(self):
        data = {
            "blocks": 1,
            "residents": [{"id": "ir1a", "track": "IR1"}],
            "requirements": _zero_requirements(),
            "blocked": [{"resident": "missing", "block": 0, "rotation": "MH-IR"}],
        }
        with self.assertRaisesRegex(ScheduleError, "Unknown resident in blocked"):
            load_schedule_input_from_data(data)

    def test_unknown_resident_in_forced_raises_schedule_error(self):
        data = {
            "blocks": 1,
            "residents": [{"id": "ir1a", "track": "IR1"}],
            "requirements": _zero_requirements(),
            "forced": [{"resident": "missing", "block": 0, "rotation": "MH-IR"}],
        }
        with self.assertRaisesRegex(ScheduleError, "Unknown resident in forced"):
            load_schedule_input_from_data(data)

    def test_requirements_non_half_increment_rounds_and_emits_warning(self):
        requirements = _zero_requirements()
        requirements["IR2"]["48X-IR"] = 0.74
        requirements["IR2"]["48X-CT/US"] = 0.74
        data = {
            "blocks": 2,
            "residents": [{"id": "ir2a", "track": "IR2"}],
            "requirements": requirements,
        }

        schedule_input = load_schedule_input_from_data(data)
        warnings = list(getattr(schedule_input, "warnings", ()))
        self.assertTrue(warnings)
        joined = "\n".join(warnings)
        self.assertIn("requirements.IR2.48X-IR", joined)
        self.assertIn("rounded to", joined)


if __name__ == "__main__":
    unittest.main()

