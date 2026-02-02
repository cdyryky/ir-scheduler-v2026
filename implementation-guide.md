# IR/DR Scheduler — CP-SAT Constraint Implementation Guide (v9)

This guide maps IR-service scheduling rules into Google OR-Tools CP-SAT constraints.

---

## 1. Scope

This solver schedules **IR-service rotations only**:

- `KIR`
- `MH-IR`
- `MH-CT/US`
- `48X-IR`
- `48X-CT/US`

Residents may be **unassigned** in many blocks.

Whether a resident is assigned every block is determined by the per-track `requirements` totals (e.g., if a track’s total required blocks equals `num_blocks`, then assignments must fill every block under the “0 or 1.0 FTE per block” rule).

---

## 2. Core Variables

### 2.1 Half-unit assignment variables

Use half-unit integers to support 0.5 assignments and within-block splits:

- `u[r,b,rot] ∈ {0,1,2}` where:
  - `0` = 0.0 FTE
  - `1` = 0.5 FTE
  - `2` = 1.0 FTE

### 2.2 Presence variables

Derive presence booleans:

- `p[r,b,rot] ∈ {0,1}`

Reify `p` from `u`:

```python
# u in {0,1,2}
model.Add(u[r,b,rot] >= 1).OnlyEnforceIf(p[r,b,rot])
model.Add(u[r,b,rot] == 0).OnlyEnforceIf(p[r,b,rot].Not())
```

(With `u` restricted to {0,1,2}, `u>=1` and `u==0` is sufficient.)

---

## 3. Global Hard Constraints

### 3.1 One place at a time (per resident per block)

No resident may exceed 1.0 FTE total in a block:

```python
model.Add(sum(u[r,b,rot] for rot in Rot) <= 2)
```

### 3.2 Strict block-level forbids (`blocked`)

If `blocked[r,b,rot] == 1`, then that assignment is forbidden:

```python
model.Add(u[r,b,rot] == 0).OnlyEnforceIf(blocked[r,b,rot])
```

`blocked` is **strict** (hard).

### 3.3 Strict block-level requires (`forced`)

If `forced[r,b,rot] == 1`, then that assignment is required as a full 1.0 FTE:

```python
model.Add(u[r,b,rot] == 2).OnlyEnforceIf(forced[r,b,rot])
```

`forced` is **strict** (hard) and conflicts with `blocked` for the same `(r,b,rot)`.

---

## 4. Within-block Splitting (Supported)

Residents may satisfy a single block’s 1.0 FTE via a 0.5/0.5 split across two rotations (driven by requirements and other constraints).

- DR residents are prohibited from half assignments (see `no_half_non_ir5`).
- Half assignments on `KIR` are prohibited (see `no_half_kir`).
- There is **no special coupling** that restricts which two non-`KIR` rotations may be split, and there is **no IR5-only split rule**.

---

## 5. Coverage Constraints (Hard, per block)

All coverage constraints apply for every block `b ∈ B`.

### 5.1 48X coverage (exact)

Always exactly 1.0 FTE per block:

```python
# Exactly 1.0 FTE => sum half-units == 2
model.Add(sum(u[r,b,"48X-IR"] for r in R) == 2)
model.Add(sum(u[r,b,"48X-CT/US"] for r in R) == 2)
```

### 5.2 MH aggregate coverage (min/max)

MH total is **3.0 to 4.0 FTE** inclusive. `MH-CT/US` is optional.

```python
mh_total = (
    sum(u[r,b,"MH-IR"] for r in R) +
    sum(u[r,b,"MH-CT/US"] for r in R)
)
model.Add(mh_total >= 6)  # 3.0 FTE
model.Add(mh_total <= 8)  # 4.0 FTE
```

### 5.3 MH-CT/US cap (optional, no minimum)

```python
model.Add(sum(u[r,b,"MH-CT/US"] for r in R) <= 2)  # <= 1.0 FTE
```

### 5.4 KIR capacity

```python
model.Add(sum(u[r,b,"KIR"] for r in R) <= 4)  # <= 2.0 FTE
```

---

## 6. Per-Resident Requirements (Hard totals)

Define:

```python
def total_units(r, rot):
    return sum(u[r,b,rot] for b in B)
```

All totals are in half-units (2 per block).

### 6.1 DR-track totals (and exclusivity)

DR residents cannot be assigned to two IR-service rotations.

- DR1: exactly 1 block `MH-IR`, nothing else
- DR2: exactly 1 block `MH-CT/US`, nothing else
- DR3: exactly 1 block `48X-CT/US`, nothing else

```python
# DR1
model.Add(total_units(r,"MH-IR") == 2)
for rot in ["KIR","MH-CT/US","48X-IR","48X-CT/US"]:
    model.Add(total_units(r,rot) == 0)

# DR2
model.Add(total_units(r,"MH-CT/US") == 2)
for rot in ["KIR","MH-IR","48X-IR","48X-CT/US"]:
    model.Add(total_units(r,rot) == 0)

# DR3
model.Add(total_units(r,"48X-CT/US") == 2)
for rot in ["KIR","MH-IR","MH-CT/US","48X-IR"]:
    model.Add(total_units(r,rot) == 0)
```

### 6.2 IR-track totals

IR1:
- 1 `MH-IR`, 1 `48X-IR`, 1 `48X-CT/US`

IR2:
- 2 `MH-IR`, 1 `48X-IR`

IR3:
- 1 `MH-IR`, 1 `48X-IR`, 1 `48X-CT/US`

IR4:
- 3 `KIR`, 3 `MH-IR`

IR5:
- Assigned every block: total IR-service units = 26
- `KIR == 6` units (3 blocks)
- `48X-IR >= 4` units (>=2 blocks)
- `MH-CT/US == 0`
- Remaining may be `MH-IR`, `48X-IR`, `48X-CT/US`

(Implement as in v8; unchanged except reference corrections below.)

---

## 7. Distribution / Safety Rules

### 7.1 IR5 leadership on MH-IR (Hard, per block)

At least **1.0 FTE** of IR5 coverage on MH-IR each block:

```python
model.Add(sum(u[r,b,"MH-IR"] for r in IR5s) >= 2)  # >= 1.0 FTE
```

### 7.2 Max 2.0 FTE seniors (IR4+IR5) on MH-IR (Hard, per block)

This is an **FTE cap** (not headcount).

```python
model.Add(sum(u[r,b,"MH-IR"] for r in IR4s_plus_IR5s) <= 4)  # <= 2.0 FTE
```

### 7.3 Max 1 “first-timer” on MH-IR per block (Soft)

Applies to (DR1 + IR1). “First-timer” means present on MH-IR in block `b` and never present on MH-IR in any earlier block.

#### Construct per-resident first-timer indicators

```python
first_timer = {}  # first_timer[(r,b)] -> BoolVar

for r in FIRST_TIMER_CANDIDATES:  # DR1s + IR1s
    for b in B:
        prior_units = model.NewIntVar(0, len(B), f"prior_mh_{r}_{b}")
        model.Add(prior_units == sum(p[r,k,"MH-IR"] for k in range(b)))

        prior_zero = model.NewBoolVar(f"prior_zero_{r}_{b}")
        model.Add(prior_units == 0).OnlyEnforceIf(prior_zero)
        model.Add(prior_units >= 1).OnlyEnforceIf(prior_zero.Not())

        ft = model.NewBoolVar(f"first_timer_{r}_{b}")
        # ft <-> (p[r,b,"MH-IR"] AND prior_zero)
        model.AddBoolAnd([p[r,b,"MH-IR"], prior_zero]).OnlyEnforceIf(ft)
        model.AddBoolOr([p[r,b,"MH-IR"].Not(), prior_zero.Not()]).OnlyEnforceIf(ft.Not())
        first_timer[(r,b)] = ft
```

#### Soft cap with “excess” penalty

For each block:

```python
first_timer_excess = {}  # IntVar >= max(0, count-1)

for b in B:
    count_b = sum(first_timer[(r,b)] for r in FIRST_TIMER_CANDIDATES)
    excess = model.NewIntVar(0, len(FIRST_TIMER_CANDIDATES), f"ft_excess_{b}")
    model.Add(count_b <= 1 + excess)  # excess = 0 when count<=1
    first_timer_excess[b] = excess
```

Add `sum(first_timer_excess[b])` to the objective with a configured weight.

### 7.4 No DR1 on MH-IR in first 4 blocks (Hard)

Blocks `0..3`:

```python
for r in DR1s:
    for b in range(4):
        model.Add(u[r,b,"MH-IR"] == 0)
```

### 7.5 IR3 study protection: last 6 blocks no MH-IR or 48X-IR (Hard)

Blocks `7..12`:

```python
for r in IR3s:
    for b in range(7,13):
        model.Add(u[r,b,"MH-IR"] == 0)
        model.Add(u[r,b,"48X-IR"] == 0)
```

### 7.6 IR4 off for SICU (Hard)

Let `N = len(IR4s)`. In each of the first `min(N, num_blocks)` blocks, exactly one IR4 is OFF (no assignment),
and no IR4 is OFF more than once across those blocks.

```python
K = min(len(IR4s), len(B))
off = {}  # BoolVar off[(r,b)] <-> (sum(p[r,b,*]) == 0)

for r in IR4s:
    for b in range(K):
        off[(r,b)] = model.NewBoolVar(f"ir4_sicu_off_{r}_{b}")
        any_assigned = sum(p[r,b,rot] for rot in ROTATIONS)
        model.Add(any_assigned == 0).OnlyEnforceIf(off[(r,b)])
        model.Add(any_assigned >= 1).OnlyEnforceIf(off[(r,b)].Not())

for b in range(K):
    model.Add(sum(off[(r,b)] for r in IR4s) == 1)

for r in IR4s:
    model.Add(sum(off[(r,b)] for b in range(K)) <= 1)
```

### 7.7 Minimum time off MH in rolling windows (Soft)

Interpretation: “off MH” means **not on MH** (MH-IR or MH-CT/US). This includes both unassigned time and time
scheduled on non-MH rotations (48X/KIR).

Let `N` be the window size, and `X` be the minimum “off MH” time (in FTE) required in every `N`-block window.
For each resident and each window, compute:

```
off_units = sum((2 - (u[r,b,"MH-IR"] + u[r,b,"MH-CT/US"])) for b in window)  # half-FTE units
```

Add a per-window deficit and penalize it:

```python
deficit = []  # IntVar (0..2X) for each window

target_units = int(round(X * 2))
for r in R:
    for b0 in range(0, len(B) - (N - 1)):
        off_units = sum((2 - (u[r,b0+k,"MH-IR"] + u[r,b0+k,"MH-CT/US"])) for k in range(N))
        d = model.NewIntVar(0, target_units, f"mh_off_deficit_{r}_{b0}")
        model.Add(off_units + d >= target_units)
        deficit.append(d)
```

Add `sum(deficit)` to the objective with a configured weight.

---

## 8. Objective (Recommended)

Lexicographic two-stage optimization:

### Stage 1: balance IR5 48X-IR totals

```python
t0 = total_units(IR5_1, "48X-IR")
t1 = total_units(IR5_2, "48X-IR")
diff = model.NewIntVar(0, 26, "ir5_48x_ir_diff")
model.AddAbsEquality(diff, t0 - t1)

model.Minimize(diff)
```

Solve, record `best_diff`, then add:

```python
model.Add(diff == best_diff)
```

### Stage 2: minimize soft penalties (and optionally adjacency)

Recommended components:

- `pen_consec = sum(consec_excess)`  (soft §7.6)
- `pen_first_timer = sum(first_timer_excess[b])` (soft §7.3)
- Optional: adjacency (still useful even if §7.6 exists)

Adjacency:

```python
adj = []
for r in R:
    for b in range(len(B) - 1):
        z = model.NewBoolVar(f"adj_full_mh_{r}_{b}")
        model.AddBoolAnd([full_mh[(r,b)], full_mh[(r,b+1)]]).OnlyEnforceIf(z)
        model.AddBoolOr([full_mh[(r,b)].Not(), full_mh[(r,b+1)].Not()]).OnlyEnforceIf(z.Not())
        adj.append(z)
```

Weighted minimize:

```python
W_CONSEC_3 = 100
W_FIRST_TIMER = 30
W_ADJ = 1

model.Minimize(
    W_CONSEC_3 * sum(consec_excess) +
    W_FIRST_TIMER * sum(first_timer_excess[b] for b in B) +
    W_ADJ * sum(adj)
)
```

---

## 9. Multiple Schedules (N solutions)

Simplest implementation (recommended for now):

- Enumerate solutions by adding a **no-good cut** on the full assignment for each found solution.
- Residents are treated as distinct. This can produce DR-year “symmetry duplicates” (same schedule up to swapping DR residents).

No-good cut pattern:

```python
lits = []
for r in R:
    for b in B:
        for rot in Rot:
            if solver.Value(p[r,b,rot]) == 1:
                lits.append(p[r,b,rot])
            else:
                lits.append(p[r,b,rot].Not())
model.AddBoolOr([lit.Not() for lit in lits])  # forbid exact same solution
```

Optional later enhancement:
- Post-process solutions to canonicalize DR-year swaps and de-duplicate by canonical signature.

---

## 10. Corrections vs v8

- The IR5 “assigned every block” statement is enforced by **§6 IR5 totals** (sum units == 26), not §7.2.
- §7.1 and §7.2 are explicitly **FTE-based** and **hard**.
- §7.3 and §7.6 are **soft** and implemented via “excess” penalty variables.
- `full_mh.Not()` is implemented as `u <= 1` (not `u != 2`).
- Updated the splitting description: within-block 0.5/0.5 splits are supported, but there is no IR5-only coupling rule.
