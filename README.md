# IR/DR Scheduler (CP-SAT)

This repository contains a Python-based CP-SAT solver for IR/DR service scheduling. The solver reads a YAML input file, applies hard/soft constraints, and outputs one or more feasible schedules with objective penalties.

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
2. Run the solver:

```bash
python ir_scheduler.py path/to/input.yml
```

To write output to a file:

```bash
python ir_scheduler.py path/to/input.yml -o output.yml
```

## Input Format

The solver expects a YAML file with the following keys:

- `blocks`: Either an integer (number of blocks) or a list of block labels.
- `residents`: List of residents with `id` and `track`.
- `blocked` (optional): Per-resident, per-block, per-rotation hard blocks.
- `weights` (optional): Objective weights for soft constraints.
- `num_solutions` (optional): Number of solutions to enumerate (default: 1).

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

## Output Format

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

## Notes

- The full constraint mapping is documented in `implementation-guide.md`.
- IR5 residents may split 0.5/0.5 between `MH-IR` and `48X-IR` within a block.
- If the model is infeasible, the solver returns an empty `solutions` list.
