from __future__ import annotations

import argparse
import os
import sys

from ir_scheduler_constants import ROTATIONS
from ir_scheduler_constraints import CONSTRAINT_SPECS, ConstraintSpec
from ir_scheduler_output import result_to_csv, result_to_yaml
from ir_scheduler_parse import expand_residents, load_schedule_input, load_schedule_input_from_data
from ir_scheduler_solve import _optimize_and_fix, solve_schedule
from ir_scheduler_types import (
    Diagnostic,
    Resident,
    ScheduleError,
    ScheduleInput,
    Solution,
    SolveResult,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="IR/DR scheduling solver using CP-SAT.")
    parser.add_argument(
        "input",
        nargs="?",
        default="ir-scheduler.yml",
        help="Path to YAML input file (default: ir-scheduler.yml).",
    )
    parser.add_argument("-o", "--output", help="Output YAML file path. Defaults to stdout.")
    parser.add_argument(
        "--csv-output",
        default="schedule-output.csv",
        help="CSV output path (default: schedule-output.csv).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(
            f"Input file not found: {args.input}\n"
            "Tip: run the Streamlit app and download to the current directory, "
            "or pass a path explicitly (e.g., python ir_scheduler.py path/to/file.yml).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    schedule_input = load_schedule_input(args.input)
    result = solve_schedule(schedule_input)
    if result.diagnostic and sys.stdin.isatty():
        response = input("Model infeasible. Run fast diagnostic suggestions? [y/N]: ").strip().lower()
        if response.startswith("y"):
            progress_state = {"printed": False}

            def _progress(current: int, total: int, spec: ConstraintSpec, mode: str) -> None:
                if not sys.stdout.isatty():
                    return
                progress_state["printed"] = True
                label = f"{spec.id}->{mode}"
                print(
                    f"Diagnostic suggestions: {current}/{total} {label}",
                    end="\r",
                    file=sys.stderr,
                    flush=True,
                )

            result = solve_schedule(
                schedule_input,
                suggest_relaxations=True,
                shrink_core=False,
                progress_cb=_progress,
            )
            if progress_state["printed"]:
                print("", file=sys.stderr)
    output = result_to_yaml(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)

    csv_text = result_to_csv(result)
    with open(args.csv_output, "w", encoding="utf-8", newline="") as handle:
        handle.write(csv_text)


if __name__ == "__main__":
    main()

