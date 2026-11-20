# Training waves 5 to 11

In brief:

- From the evening of 2026-11-12 to the morning of 2026-11-14, waves 5 to 11 tested 28 changes to the click-label ranker: its cooldown, regularisation, training length, batch size and structure.
- No arm was adopted. The cooldown recipe and 10 epochs stayed.
- Arms that differed from the base were 0.001 to 0.003 too weak on validation, and the strong ones were near copies. Member gains did not carry to the pool.

## Why it was tried

By the evening of 2026-11-12 the click-label recipe was settled: Muon with a constant-then-cooldown schedule, 10 epochs, and a cooldown that mixes in 2% random-order feeds. The waves asked whether a change to the cooldown, the regularisation, the training length or batch, or the model's structure would give a stronger member, or one different enough to help the pool.

A member is one trained ranker, and the pool is the mean of the members' within-feed z-scores. The cooldown is the last 20% of training steps, in which the learning rate falls to 3% of its peak. The [glossary](../../docs/glossary.md) defines these and the other terms.

A pool gains from members that are strong and different. On validation, the gain over the better of two families followed 0.019 (1 - ρ) - 0.38 × gap, where ρ is their within-feed correlation and gap their difference in strength.

Seeds of the recipe correlate 0.93 to 0.94. A new family therefore had to differ more than a seed does while staying almost as strong.

Cooldown changes were cheap to test. The last 2 epochs of a finished member can be re-run from its stored checkpoint. This cooldown re-run came from [rounds-5-7](../rounds-5-7/).

## What ran

The runner `lane.sh` ran job lists on shared GPUs in three stages.

| Stage | Job | Runs per arm | Time per run |
|---|---|---|---|
| A | 2 epochs from scratch | seeds 23 and 24 | about 37 minutes |
| B | cooldown re-run of a stored member, against the same re-run without the change | four stored members | 38 to 48 minutes |
| E | 10 epochs from scratch | seeds 951 and 952 | about an hour |

Every arm was the base command plus one flag that is off by default (`cd_arms8.py` shows two). Each wave's rule was written to `lab/nn_x/w5/logs/*_predeclared.txt` before its first run.

The read-outs (`w5_eval.py`, `w9_joint_s_readout.py`) compare an arm, feed by feed, with its paired reference on two samples:

- validation: 14,423 labelled feeds;
- the holdout: 30,000 random-order train feeds kept out of training.

The read-outs from wave 9 on, and separate read-outs for some stage-E doses, also report a diversity margin. It is the correlation of two base seeds minus that of an arm seed with a base seed, so a positive margin means the arm differs from the base more than another seed does.

What each wave tested:

- Wave 5: an offset for the truncation of production feeds (`TRUNC-PROD`).
- Waves 6 to 8: the cooldown. Wave 6 changed the auxiliary weight and the random-order share, wave 7 added R-Drop or sharpness-aware minimisation, and wave 8 changed the batch size.
- Between waves 7 and 8: doses of training length and batch size, run from scratch in stage E.
- Waves 9 and 10: eight arms, mostly new model families.
- Wave 11: cheap gates in front of two costly ideas, width 192 and a next-item auxiliary task.

R-Drop adds a second dropout pass and a symmetric KL penalty between the two passes. Sharpness-aware minimisation takes its gradient at a point nudged uphill from the current weights.

## Results

Each row is a member difference against the paired reference, ± one paired standard error (SE). An arm has 4 cooldown re-runs or 2 seeds unless the row says otherwise. A pair a / b in the verdict column is validation / holdout. `results/summary.csv` lists all 28 arms with their sources.

| Wave | Arm | Validation | Holdout | Diversity margin | Verdict |
|---|---|---|---|---|---|
| 5 | truncation offset, 10 epochs | -0.0005 ± 0.0005 | -0.0001 | | fail; a tree proxy had read +0.0022 to +0.0031 |
| 6 | auxiliary weight 1.0 in the cooldown | +0.00032 ± 0.00020 | +0.00011 | | stop; the 10-epoch pair read +0.00049 ± 0.00071 |
| 7 | R-Drop in the cooldown | -0.00012 ± 0.00032 | +0.00059 (z 3.3) | | stop; pool holdout z 1.5 |
| 7 | sharpness-aware minimisation in the cooldown | +0.00004 | +0.00041 (z 2.4) | | stop; pool holdout z 1.1 |
| E | 15 epochs | -0.00055 ± 0.00071 | -0.00021 ± 0.00043 | | fail |
| E | 20 epochs (4 seeds) | -0.00078 ± 0.00053 | -0.00058 ± 0.00030 | +0.0038 | fail |
| E | 30 epochs | -0.00155 ± 0.00076 | -0.00069 ± 0.00043 | +0.0066 | fail |
| E | 4x batch | -0.00262 ± 0.00080 | -0.00209 ± 0.00048 | +0.0098 | fail |
| E | 4x batch with 2x learning rate | -0.00232 ± 0.00078 | -0.00163 ± 0.00045 | +0.0053 | fail |
| 9 | category-factorised output | -0.00097 ± 0.00061 | -0.00120 ± 0.00034 | +0.0113 | closed |
| 9 | history of feed-surface events only | -0.00311 ± 0.00074 | -0.00266 ± 0.00043 | +0.0159 | closed |
| 9 | one transformer over user, history and candidates | -0.00073 ± 0.00051 | -0.00027 ± 0.00029 | +0.0011 | closed |
| 10 | examination offset on random-order train rows | +0.00002 ± 0.00028 | -0.00008 ± 0.00015 | -0.0001 | closed |
| 11 | width 192, best of three cooldown re-run arms | +0.00021 ± 0.00048 | -0.00010 | | closed; needed +0.0008 / +0.0004 |
| 11 | next-item auxiliary task, trigger test | | | | not run; partial correlation -0.0012 ± 0.0139 against a bar of +0.05 |

### Re-test on the distilled students, 2026-11-17

On 2026-11-17 the same runner re-ran the 2-epoch cooldown of the 16 held-out distilled students (W16), which never trained on the holdout feeds. Some arms were aimed at the final model's failure map ([error-analysis](../error-analysis/)). Three were settings prepared for students on 2026-11-14 and never run.

Each arm was read as a 16-pool against reference re-runs from the same checkpoints, with a randomization test (`lab/nn_x/w12/gap1117/perm_read.py`).

| Setting | Validation | Holdout | p |
|---|---|---|---|
| learning rate 4e-3 | -0.00062 ± 0.00032 | -0.00030 ± 0.00018 | 0.98 |
| 4 epochs | -0.00046 ± 0.00033 | +0.00043 ± 0.00018 | 0.26 |

The third setting, teacher weight 0.3, is in [distillation](../distillation/). No arm of the round passed, including the failure-map arms.

### Pseudo-feeds, from wave 11 to pick 2

One idea built in wave 11 is not in the table. Pseudo-feeds are extra training lists built from a user's history (`lab/nn_x/w11/pseudo/train.py`). As a family of click-label members, 2 seeds, they read +0.00108 ± 0.00072 / +0.00024 per member on 2026-11-14. The holdout gain was below one SE and the diversity margin +0.0024 against a bar of +0.005, so the idea was closed.

On 2026-11-17 the same flags were added to the distilled students. Their 8-seed pool failed its rule (randomization p 0.19), yet the all-data file whose 16 full-view students train this way scored 0.39526061 on public and became pick 2 on 2026-11-19 ([distillation](../distillation/)).

## What it taught

No arm of waves 5 to 11 was adopted as tested. The cooldown recipe and 10 epochs stayed.

Longer training lost in proportion to the dose. The constant-rate phase (the plateau) saturates, and the gain comes from the cooldown, which needs its steps. At 4x batch the cooldown added nothing on the holdout at the same learning rate and +0.0012 with a doubled one, against +0.0035 at the normal batch.

Arms that differed from the base were 0.001 to 0.003 too weak on validation. The strong ones were near copies: R-Drop and sharpness-aware minimisation correlated 0.990 with the reference re-run of the same stored member. No arm passed both the strength bar and its diversity bar of +0.003 to +0.008. Under distillation only an input view stayed different ([popularity-views](../popularity-views/)).

Member gains did not carry to the pool. The doubled auxiliary weight gained +0.00038 ± 0.00017 over 7 cooldown re-runs, while their pool lost 0.00023 on the holdout. R-Drop's member holdout gain fell from z 3.3 to z 1.5 in the pool.

Gates before GPU time paid off. The trigger test closed the next-item task at zero GPU-hours. The gap probe showed that width 192 memorises at the plateau: the gap between its loss on the holdout and on random-order train feeds it had trained on was larger by 0.0079 ± 0.0015 than at width 128. Under distillation width 192 became the adopted width ([distillation](../distillation/)).

A gate can also pass while the arm fails. In random-order train feeds the first slot is clicked at 3.15 times the base rate, yet the examination offset built on that changed nothing.

## Files

`code/`

- `lane.sh`: the runner; its header lists every wave's arms.
- `cd_arms8.py`: how an arm was added (R-Drop, sharpness-aware minimisation).
- `w5_eval.py`: paired read-out and stop rule.
- `w9_joint_model.py`: the model with the single-stream arm.
- `w9_joint_s_readout.py`: 2-epoch read-out, diversity margin.
- `w10_r2_premise_check.py`: CPU check of the examination-offset premise.
- `w11_width_gap_probe.py`: generalisation gap by width.
- `w11_ssl_e2_trigger.py`: the zero-GPU trigger test.

`results/`

- `summary.csv`: all arms.
- `wave5_*` to `wave8_*`, `wave10_cooldown_*`: read-outs of the cooldown re-runs.
- `epochs*`, `batch4x*`: 10-epoch doses.
- `wave9_*`, `wave10_r2_*`, `wave10_r3_*`: 2-epoch screens, premise check.
- `wave11_*`: width gate, gap probe, trigger test.

### Raw material and records

- Lab folders: `lab/nn_x/w5/` (runner, trainer, read-outs, JSON logs and wave rules), `lab/nn_x/w9/`, `lab/nn_x/w10/`, `lab/nn_x/w11/`.
- The proposals: `lab/nn_x/w9/fresh_eye_ideas_1113.json`.

The same runner later trained every distilled student. The student arms of 14 to 16 November (width, depth, heads, dropout, weight decay, auxiliary-loss weight, teacher temperature, sub-lists, GRU students) are in [distillation, student recipe arms](../distillation/#student-recipe-arms-14-to-16-november).

Records: the experiment registry and the decision ledger (working records, not published).
