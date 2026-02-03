import unittest

from dr_config import (
    DEFAULT_DR_AWARDS_PER_RESIDENT,
    DEFAULT_DR_MAX_VACATION_REQUESTS,
    DEFAULT_DR_YEAR_COUNTS,
    DEFAULT_DR_YEAR_NAMES,
    DR_CONFIG_TYPE,
    DR_SCHEMA_VERSION,
    default_dr_config,
    prepare_dr_config,
    validate_dr_resident_names,
)


class DRConfigSchemaTests(unittest.TestCase):
    def test_default_dr_config_has_expected_defaults(self):
        cfg = default_dr_config()
        self.assertEqual(cfg["config_type"], DR_CONFIG_TYPE)
        self.assertEqual(cfg["schema_version"], DR_SCHEMA_VERSION)

        residents = cfg["gui"]["residents"]
        self.assertEqual(residents["year_counts"], DEFAULT_DR_YEAR_COUNTS)
        self.assertEqual(len(residents["years"]["Y1"]), 10)
        self.assertEqual(
            [row["name"] for row in residents["years"]["Y1"]],
            DEFAULT_DR_YEAR_NAMES["Y1"],
        )
        self.assertNotIn("Sausalito", [row["name"] for row in residents["years"]["Y1"]])

        vacation = cfg["gui"]["requests"]["vacation"]
        self.assertEqual(vacation["max_requests_per_resident"], DEFAULT_DR_MAX_VACATION_REQUESTS)
        self.assertEqual(vacation["awards_per_resident"], DEFAULT_DR_AWARDS_PER_RESIDENT)

    def test_prepare_dr_config_normalizes_missing_and_malformed_sections(self):
        cfg, ok = prepare_dr_config(
            {
                "config_type": "IR",
                "blocks": "bad",
                "gui": {
                    "calendar": {"start_date": ""},
                    "residents": {
                        "year_counts": {"Y1": "not-int"},
                        "years": {
                            "Y1": [
                                {"name": "  Alpha  ", "track": "nm"},
                                {"name": "", "track": "DR"},
                                "junk",
                            ]
                        },
                    },
                },
            }
        )

        self.assertTrue(ok)
        self.assertEqual(cfg["config_type"], DR_CONFIG_TYPE)
        self.assertEqual(cfg["schema_version"], DR_SCHEMA_VERSION)
        self.assertEqual(cfg["blocks"], 13)

        self.assertEqual(cfg["gui"]["calendar"]["start_date"], "06/29/26")
        self.assertIn("class_year_assignments", cfg["gui"])
        self.assertIn("constraints", cfg["gui"])
        self.assertIn("prioritization", cfg["gui"])
        self.assertIn("solve", cfg["gui"])
        self.assertIn("instructions", cfg["gui"])

        y1_rows = cfg["gui"]["residents"]["years"]["Y1"]
        self.assertEqual(len(y1_rows), 10)
        self.assertEqual(y1_rows[0]["name"], "Alpha")
        self.assertEqual(y1_rows[0]["track"], "NM")

    def test_prepare_dr_config_normalizes_vacation_requests(self):
        cfg, _ = prepare_dr_config(
            {
                "blocks": 2,
                "gui": {
                    "residents": {
                        "year_counts": {"Y1": 2, "Y2": 0, "Y3": 0, "Y4": 0},
                        "years": {
                            "Y1": [
                                {"name": "A", "track": "DR"},
                                {"name": "B", "track": "IR"},
                            ]
                        },
                    },
                    "requests": {
                        "vacation": {
                            "max_requests_per_resident": 2,
                            "awards_per_resident": 99,
                            "by_resident": {
                                "A": [
                                    {"week": "B0.0", "rank": 2},
                                    {"week": "B0.0", "rank": 1},
                                    {"week": "B0.1", "rank": 2},
                                    {"week": "B9.0", "rank": 3},
                                ],
                                "missing": [{"week": "B0.0", "rank": 1}],
                            },
                        }
                    },
                },
            }
        )

        vacation = cfg["gui"]["requests"]["vacation"]
        self.assertEqual(vacation["max_requests_per_resident"], 2)
        self.assertEqual(vacation["awards_per_resident"], DEFAULT_DR_AWARDS_PER_RESIDENT)

        self.assertIn("A", vacation["by_resident"])
        self.assertNotIn("missing", vacation["by_resident"])
        self.assertEqual(
            vacation["by_resident"]["A"],
            [
                {"week": "B0.0", "rank": 2},
            ],
        )

    def test_validate_dr_resident_names_flags_duplicates(self):
        cfg, _ = prepare_dr_config(
            {
                "gui": {
                    "residents": {
                        "year_counts": {"Y1": 2, "Y2": 0, "Y3": 0, "Y4": 0},
                        "years": {
                            "Y1": [
                                {"name": "Dup", "track": "DR"},
                                {"name": "Dup", "track": "IR"},
                            ]
                        },
                    }
                }
            }
        )
        error = validate_dr_resident_names(cfg)
        self.assertIsNotNone(error)
        self.assertIn("unique", str(error).lower())


if __name__ == "__main__":
    unittest.main()
