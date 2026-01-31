import unittest

from ir_config import (
    CLASS_TRACKS,
    CURRENT_SCHEMA_VERSION,
    DEFAULT_DR_COUNTS,
    DEFAULT_IR_NAMES,
    ROTATION_COLUMNS,
    prepare_config,
)


class ConfigSchemaTests(unittest.TestCase):
    def test_prepare_config_sets_schema_version_and_defaults(self):
        cfg, ok = prepare_config({})
        self.assertEqual(cfg.get("schema_version"), CURRENT_SCHEMA_VERSION)
        self.assertIn("gui", cfg)
        self.assertIn("residents", cfg["gui"])
        self.assertIn("forced", cfg)
        self.assertFalse(ok)

        gui_res = cfg["gui"]["residents"]
        for track, names in DEFAULT_IR_NAMES.items():
            self.assertEqual(gui_res["IR"][track], names)
        self.assertEqual(gui_res["DR_counts"], DEFAULT_DR_COUNTS)

    def test_prepare_config_infers_gui_residents_from_residents_list(self):
        residents = []
        for track in ["IR1", "IR2", "IR3", "IR4", "IR5"]:
            residents.append({"id": f"{track}A", "track": track})
            residents.append({"id": f"{track}B", "track": track})
        for idx in range(3):
            residents.append({"id": f"DR1-{idx+1}", "track": "DR1"})
        residents.append({"id": "DR2-1", "track": "DR2"})

        cfg, ok = prepare_config({"residents": residents})
        self.assertTrue(ok)
        gui_res = cfg["gui"]["residents"]
        for track in ["IR1", "IR2", "IR3", "IR4", "IR5"]:
            self.assertEqual(gui_res["IR"][track], [f"{track}A", f"{track}B"])
        self.assertEqual(gui_res["DR_counts"]["DR1"], 3)
        self.assertEqual(gui_res["DR_counts"]["DR2"], 1)
        self.assertEqual(gui_res["DR_counts"]["DR3"], 0)

    def test_prepare_config_populates_gui_requirements_from_requirements(self):
        requirements = {
            track: {rot: 0 for rot in ROTATION_COLUMNS} for track in CLASS_TRACKS
        }
        requirements["IR1"]["MH-IR"] = 7

        cfg, _ = prepare_config({"requirements": requirements})
        gui_req = cfg["gui"]["class_year_requirements"]
        self.assertEqual(gui_req["IR1"]["MH-IR"], 7)
        for rot in ROTATION_COLUMNS:
            self.assertIsInstance(gui_req["IR1"][rot], int)

    def test_prepare_config_derives_requirements_from_gui_requirements(self):
        gui_req = {
            track: {rot: 0 for rot in ROTATION_COLUMNS} for track in CLASS_TRACKS
        }
        gui_req["DR3"]["48X-CT/US"] = 2

        cfg, _ = prepare_config({"gui": {"class_year_requirements": gui_req}})
        self.assertIn("requirements", cfg)
        self.assertEqual(cfg["requirements"]["DR3"]["48X-CT/US"], 2)


if __name__ == "__main__":
    unittest.main()
