# IR/DR Scheduler (CP-SAT)

This repository contains a Python-based CP-SAT solver for IR/DR service scheduling. The solver reads a YAML input file, applies hard/soft constraints, and outputs one or more feasible schedules with objective penalties plus diagnostics when infeasible.

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

1. Create an input YAML file (example below).
2. Run the solver (defaults to `ir-scheduler.yml` in the current directory):

```bash
python ir_scheduler.py
```

To write output to a file:

```bash
python ir_scheduler.py -o output.yml
```

The solver also always writes a CSV summary to `schedule-output.csv` in the current directory (override with `--csv-output`).

To launch the GUI:

```bash
streamlit run app.py
```

## GUI Modes and Config Files

- The app now has two page modes in the header: `IR` (default) and `DR`.
- IR and DR use separate in-memory state and separate YAML files.
- Use IR mode to load/save IR scheduler configs (recommended filename: `ir-config-YYYY-MM-DD.yml`).
- Use DR mode to load/save DR page configs (recommended filename: `dr-config-YYYY-MM-DD.yml`).
- Do not mix IR/DR YAML files across modes.

### DR config schema (GUI-only, solver not wired yet)

DR mode currently supports residents, vacation requests, calendar, and save/load. DR solving/constraints are placeholders for future work.

```yaml
config_type: DR
schema_version: 1
blocks: 13
gui:
  calendar:
    start_date: 06/29/26
  residents:
    year_counts: {Y1: 10, Y2: 10, Y3: 9, Y4: 8}
    years:
      Y1:
        - {name: Bansal, track: DR}
  requests:
    vacation:
      max_requests_per_resident: 6
      awards_per_resident: 4
      by_resident:
        Bansal:
          - {week: B0.1, rank: 1}
          - {week: B3.2, rank: 2}
```

## Input Format

The solver expects a YAML file with the following keys:

- `blocks`: Either an integer (number of blocks) or a list of block labels.
- `residents`: List of residents with `id` and `track`.
- `blocked` (optional): Per-resident, per-block, per-rotation hard blocks.
- `forced` (optional): Per-resident, per-block forced rotation assignment (hard).
- `weights` (optional): Objective weights for soft constraints.
- `num_solutions` (optional): Number of solutions to enumerate (default: 1).
- `gui` (optional): GUI-specific config used by Streamlit (see below).

### Resident tracks

Valid tracks are: `DR1`, `DR2`, `DR3`, `IR1`, `IR2`, `IR3`, `IR4`, `IR5`.

### Rotations

Valid rotations are: `KIR`, `MH-IR`, `MH-CT/US`, `48X-IR`, `48X-CT/US`.

### Example Input

```yaml
blocks: [B1, B2, B3, B4]
residents:
  - id: alice
    track: IR1
  - id: bob
    track: IR2
  - id: carol
    track: IR5
blocked:
  - resident: alice
    block: B2
    rotation: MH-IR
weights:
  consec: 100
  first_timer: 30
  adj: 1
num_solutions: 2
```

### GUI config (`gui`)

When using the Streamlit app, the file includes a `gui` section. The CLI still uses `residents`, so the app writes both.

```yaml
gui:
  residents:
    IR:
      IR1: ["Gaburak","Miller"]
      IR2: ["Qi","Verst"]
      IR3: ["Madsen","Mahmoud"]
      IR4: ["Javan","Virk"]
      IR5: ["Brock","Katz"]
    DR_counts: {DR1: 8, DR2: 7, DR3: 8}
  constraints:
    modes:
      one_place: always
      first_timer: if_able
    soft_priority:
      - first_timer
```

### `blocked` alternatives

You can also provide `blocked` as a mapping of resident IDs to blocks and rotations:

```yaml
blocked:
  alice:
    B2: [MH-IR]
    B4: [KIR, 48X-IR]
```

### `weights`

All weight values are integers. Defaults are:

```yaml
weights:
  consec: 100
  first_timer: 30
  adj: 1
```

## Output Format (YAML)

The solver outputs YAML with a `solutions` list. Each solution includes:

- `objective`: penalty totals for soft constraints.
- `assignments`: per-resident assignments by block with FTE values.

Example output structure:

```yaml
solutions:
  - objective:
      consec_excess: 0
      first_timer_excess: 1
      adj: 2
    assignments:
      alice:
        B1:
          MH-IR: 1.0
        B2: {}
      bob:
        B1:
          48X-IR: 1.0
```

If infeasible, the output includes a `diagnostic` with conflicting constraints.

## Output Format (CSV)

The solver writes `schedule-output.csv` in the current directory on every run (overwritten).

Columns:

```
solution_idx,resident,block,rotation,fte
```

Only non-zero assignments are included.

## Notes

- The full constraint mapping is documented in `implementation-guide.md`.
- Half-block (0.5) assignments are supported for IR tracks (except `KIR`), which can produce within-block 0.5/0.5 splits across two rotations. DR residents are always full-block.
- If the model is infeasible, the solver returns an empty `solutions` list and a `diagnostic` section.
