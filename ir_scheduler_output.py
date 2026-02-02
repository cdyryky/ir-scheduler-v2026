from __future__ import annotations

import csv
from io import StringIO
from typing import Dict

import yaml

from ir_scheduler_types import SolveResult


def result_to_yaml(result: SolveResult) -> str:
    payload: Dict[str, object] = {"solutions": []}
    if result.solutions:
        payload["solutions"] = [
            {"objective": solution.objective, "assignments": solution.assignments}
            for solution in result.solutions
        ]
    if getattr(result, "warnings", None):
        payload["warnings"] = list(result.warnings)
    if result.diagnostic:
        payload["diagnostic"] = {
            "status": result.diagnostic.status,
            "conflicting_constraints": result.diagnostic.conflicting_constraints,
            "suggestions": result.diagnostic.suggestions,
        }
    return yaml.safe_dump(payload, sort_keys=False)


def result_to_csv(result: SolveResult) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["solution_idx", "resident", "block", "rotation", "fte"])
    for idx, solution in enumerate(result.solutions):
        for resident, blocks in solution.assignments.items():
            for block, rotations in blocks.items():
                for rotation, fte in rotations.items():
                    if fte:
                        writer.writerow([idx, resident, block, rotation, fte])
    return output.getvalue()
