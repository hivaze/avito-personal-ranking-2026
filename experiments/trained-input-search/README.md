# Searching for a missing input

In brief:

- After distillation, the stored data were searched for information the distilled students do not use yet, and the best find was trained into the ranker as an input.
- Trained members used that input, a co-visitation walk, but gained nothing: the all-data file with it read -0.00054 ± 0.00050 on public against FD16P20.
- An automatic search over 11,617 generated features behaved like its noise controls.

## Why it was tried

After distillation the leader was still 0.0032 ahead of the team's best public score, and capacity changes had failed. Terms are defined in the [glossary](../../docs/glossary.md).

About 300 candidate columns had been closed by a nested residual gate on top of the pool. That gate tests a column as a correction to the pool's score, with the weight fitted in user folds on validation and checked on the holdout.

On 2026-11-14 an audit showed that gate to be nearly blind. The block of the 8 popularity inputs is worth +0.00852 ± 0.00066 on the holdout as a trained input. In the gate it read +0.00086 (linear) and +0.00222 (monotone GBDT) on the holdout, and nothing on validation.

Those closures therefore covered the residual form only (coverage map, row "Feature rows closed by a nested residual gate"). A column could still help when trained into the ranker, a form the earlier gates never tested.

## What ran

### Automatic feature search

On 2026-11-14 a CPU search (`screen.py`, `blocks.py`) tested 11,617 generated features in 12 operator families, plus OpenFE, an open-source feature generator. Each ran on top of the 8-student file W8 (validation 0.40165), with its weight fitted on one half of the validation users and scored on the other.

Two controls went through the same code. A set of 397 label-free noise features showed how often chance passes, and a planted interaction checked that the harness could see a real effect.

### Calibrated conditional score test

On 2026-11-16 `condlib.py` fitted a null model on the pool's score and 84 of the ranker's own input columns. Each stored column was then scored against the residual of that model. Under "nothing new" the statistic is close to N(0,1): placebo columns gave an sd of 0.86 to 1.06.

### The walk input, code name nx3

The strongest column of that test was a two-step co-visitation walk. It goes from the candidate through other users' whole histories to the feed's own history items.

`nx3_build.py` built it for every split. Train rows use a lag view that reads only sources 1 to 8 days old, to match test rows. Rows of Sep 6 to 11 have no admissible source.

CPU gates (`nx3_gates.py`) had to pass before any GPU run. They read two witnesses, samples on which a change is judged:

- the lagged holdout, the 15,225 holdout feeds whose popularity inputs can be rebuilt at the test lag;
- stale validation, validation re-predicted with private-like stale inputs.

The GPU stages followed:

1. Stage A: 8 click-label members with the input against 6 stored and 2 fresh members without it (width 128, 10 epochs, 180 to 182 min each).
2. Stage B: 8 distilled width-192 students against the 16 stored ones.
3. A full all-data file.

Every rule and forecast was written in the decision ledger before the first build.

### Second CPU round

A second CPU round (2026-11-16) tried six new walk constructions, neighbour density, counts by search surface, and a click-strength input repaired for exposure and train/test parity.

## Results

A pair "a / b" gives the two samples of the Sample column in order. ± is one standard error of a paired per-feed difference, and z is the difference divided by it.

Pooled z is (z on validation + z on the lagged holdout) / sqrt(2). Unlike the standing rule, which pools validation with the full holdout of 30,000 random-order train feeds, this line used the lagged holdout as its second witness. Nested z is the z of a feature's gain when its weight is fitted on one half of the validation users and scored on the other.

| Test | Sample | Result |
|---|---|---|
| automatic search, share of features with nested z ≥ 2 | validation | 1.02% (linear) / 0.38% (binned), noise features 1.01% / 0.25% |
| automatic search, best block | validation | +0.00006 ± 0.00016 |
| planted interaction worth +0.0009 on validation / +0.0019 on the holdout | validation / holdout | recovered at +0.0006 (z 1.1) / +0.0012 (z 4.5) |
| standardising inputs with validation statistics at test time | validation | -0.0024 to -0.0057 on single members |
| score test over 284 stored columns | validation and holdout | 25 found on both witnesses; one survived parity and lag checks |
| walk input, CPU gates | lagged holdout / stale validation | score statistic +4.98 / +6.46: pass |
| walk input, stage A (click labels) | validation / lagged holdout | member +0.00037 ± 0.00056 / +0.00006 ± 0.00033, pooled z +0.59 (bar 2.0) |
| walk input, stage B (students) | validation / lagged holdout | member +0.00005 ± 0.00029 / +0.00016 ± 0.00024, pooled z +0.62 |
| walk input, stage A, natural minus permuted values | validation | +0.00179 ± 0.00034 (8 of 8 members) |
| file with walk students (FDN16P20) | validation / public | +0.00011 ± 0.00038 / 0.3943559, -0.00054 ± 0.00050 against 0.394892 |
| file mixing plain and walk students (FDM32P20) | validation / public | -0.00007 ± 0.00026 / 0.394578 |
| repaired click strength | lagged holdout / validation at private lag | score statistic +1.22 to +2.17 against a bar of 2.74; NDCG gain ≤ 0 on validation |

The two files are compared with FD16P20: 16 all-data full-view students plus 4 popularity-light students at weight 0.2, public 0.394892. Full-view students read all 34 pre-computed inputs. Popularity-light students have the 8 popularity inputs replaced by one exposure count.

Both stages read positive and neither came near its bar. The design had said beforehand that the test could see a member-level +0.0009 to +0.0011 with power 0.8, not the +0.0003 the information predicted (`results/walk_input_rule_power.txt`).

The members did use the walk: permuting its values within the feed cost the stage A members 0.00179 on validation. The net gain stayed at noise level, so the walk replaced information the ranker already had.

Walk members were not a different view either. Their correlation with plain members (0.9487) equals that among plain members (0.9490), and a size-matched mix of the two read -0.00017 ± 0.00019 on validation.

## What it taught

A column can carry information about the click on two witnesses, be used by the trained model, and still add nothing, because the model substitutes it for information it already had. Only a trained arm read at pool level can show a gain, and at 8 seeds it sees effects of about +0.001.

The automatic search behaved exactly like its noise controls. Nothing of the size of the 0.0032 gap to the leader was found in the stored data.

Standardising inputs with validation statistics hurt, because the shift of input levels between train and validation is itself information for the model.

Not run: the walk together with the search-surface count, and a teacher rebuilt from walk members.

## Files

Scripts in `code/`:

- `screen.py`, `blocks.py`: stages 1 and 2 of the automatic search.
- `condlib.py`: the conditional score test.
- `nx3_build.py`, `nx3_gates.py`: build and CPU gates of the walk input.
- `launch_nx3_hl.sh`: stage A launcher.
- `nx3_read_a.py`, `mix_read.py`: stage A read-out, mixed-pool read.

Read-outs in `results/`:

- `autofe_per_family_table.txt`, `autofe_test_time_standardisation.json`: the search per family; test-time standardisation per member.
- `walk_input_rule_power.txt`, `walk_input_step0_gates.txt`, `walk_input_stage_a_harness.txt`, `walk_input_stage_a_read.txt`, `walk_input_stage_b_read.txt`, `walk_input_mix_read.txt`: the walk input from design to mix.
- `click_strength_conditional_test.txt`, `click_strength_ndcg.txt`: the click-strength input.

### Raw material and records

- Lab folders: `lab/nn_x/w12/autofe/`, `lab/nn_x/w12/nx3/`, `lab/reports/INPUTS_1116/` (inventories, `round2/`, `residual_rows/`), `lab/reports/NEXT_1116/`.
- Automatic search report: `lab/reports/AUTOFE_1114.md`.
- Walk input reports, in `lab/reports/INPUTS_1116/`: `DESIGN.md`, `CRITIC.md`, `nx3_step0.md`, `nx3_verify.md`.
- Click-strength reports: `lab/reports/NEXT_1116/clickstrength/REPORT.md`, `lab/reports/NEXT_1116/CRITIC.md`.
- Records: the decision ledger (a working record, not published).
