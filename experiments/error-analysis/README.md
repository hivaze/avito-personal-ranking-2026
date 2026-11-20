# Error analysis

In brief:

- Four CPU passes (2026-11-11 to 2026-11-13) and later checks on the final model asked where the best pool loses NDCG on validation, and whether any available signal separates the positives it misses from the negatives ranked above them.
- No signal did. Every residual model, gate and router gained on feeds with cold positives and lost more on feeds with warm ones, and pick 1 (FD16P20@m0.4_0.2) is calibrated. The one gain was a rule: pool members of different recipes at equal weight.

A positive is a candidate with target 1, one the user acted on. An item is cold if it was never a train candidate before the feed's day, warm otherwise. Other terms are in the [glossary](../../docs/glossary.md).

## Why it was tried

If the misses share a cue, a feature or a residual model can use it. If they do not, stackers (second-stage models on top of the pool's score), routers (rules that choose a model per feed) and post-processing can be closed without GPU time.

Four passes ran, each on the best pool of its day. Checks on the final model asked the same question again on 2026-11-17 and 2026-11-18.

## What ran

All passes were CPU only, on validation (14,423 feeds). Passes 1 and 3 drop the 117 feeds whose user has a later validation or test feed. Slices defined by the label, such as cold or warm positive or the rank of the positive, show where the loss sits and say nothing about its cause.

1. 2026-11-11, morning (`lab/eda/errors/`). Four lenses (segments, category intent, missed positives, feed structure) on the second neural ensemble of the day (`nn_final2`). A synthesis followed on the third (`nn_final3`, train-only 0.38662 on leak-free feeds), with a gate for new columns (`02_screen.py`). The pass used 16 threads per process and up to 8 processes.
2. 2026-11-11, evening (`lab/solvers/errors/`). The best file of that hour, v11: the fifth neural ensemble, fine-tuned and train-only halves at 0.5 each, 0.39793 on validation. Loss decomposition, local AUC around each positive, nested residual models, oracles with realistic gates, and routing between models.
3. 2026-11-12 (`lab/nn_x/insights/errors/`). G3, the pool of 20 click-label rankers (0.40032 on validation, public 0.392820). Exact per-positive split of 1 - NDCG, oracles, the response of each recent lever by positive type, and calibration by listing age.
4. 2026-11-13 (`lab/nn_x/insights/tail_1113/tail.py`, 21 s at 12 threads). Tail feeds: feeds where G3 has no positive in its top 8.

An oracle is a rule allowed to read the label. A realistic gate tries to predict the same thing without it.

Later checks on the final model:

- 2026-11-17 (`lab/nn_x/errfix_1117/`): a second error analysis, which looked for code rules (hand-written heuristics that correct wrong predictions). Nothing passed the sealed half of validation. Its failure map led to two loss rounds on GPU: cooldown re-runs of the 16 students (`lab/nn_x/w12/gap1117/`) and, that evening, a loss aimed at the hardest feeds (`lab/nn_x/w5/cd_arms10.py`).
- 2026-11-18 (`lab/nn_x/w12/noise/`, CPU): a split of the remaining error by kind of noise.
- 2026-11-18 (`lab/nn_x/w12/fixmiss/`, CPU only): a search of pick 1, FD16P20@m0.4_0.2, for systematic misses that a correction could fix. It ran three lines of analysis, each re-checked by a separate pass.

## Results

### The four passes

| Pass | Finding | Number (validation) |
|---|---|---|
| 1 | best of 11 new columns, nested blend on the train-only base / on the fine-tuned base | +0.00093 ± 0.00038 / -0.00003 ± 0.00027 |
| 1 | `nn_final3` against `nn_final2`, train-only: warm-positive feeds / all-cold-positive feeds | +0.0111 ± 0.0012 / -0.0206 ± 0.0022 |
| 2 | share of the gap to 1 from positives at ranks 3 to 99 | 83% |
| 2 | local AUC of 70 attributes, positive against negatives within ±5 ranks | 0.492 to 0.504 |
| 2 | nested per-feed softmax residual on 32 signals: all / all-cold-positive feeds | -0.00058 ± 0.00049 / +0.00466 ± 0.00114 |
| 2 | nested LightGBM residuals | -0.0015 to -0.011 |
| 2 | oracle "the feed's positives are all cold", then a gate that predicts it (AUC 0.661) | +0.0253 / -0.00003 ± 0.00021 |
| 2 | equal-member pool of three recipes (S3) against v11 | +0.00115 ± 0.00044; holdout +0.00128 ± 0.00027 |
| 3 | loss outside the pool's top category / inside it with 3+ same-category items above | 53% / 37% |
| 3 | members left to add: limit of the a - b/k fit minus G3 | +0.0002 |
| 3 | calibration of the train-only half, listings 1 to 3 days old / older than 90 days | actual/expected 1.136 ± 0.028 / 0.957 ± 0.011 |
| 4 | tail feeds (no positive in the top 8): share of feeds / share of loss | 51.2% / 66.0% |
| 4 | AUC of "is this a tail feed": feed size alone / all label-free statistics | 0.766 / 0.773 |

What each pass found:

- Pass 1 found the cold-for-warm trade of the newer ensemble, and a residual stacker that no longer added anything (-0.0001 ± 0.0006).
- Pass 2 found nothing to add to v11 from any attribute: low-ranked positives look like average negatives on every cue. Its one positive result was pooling members of different recipes at equal weight. That rule is behind the first pool of three recipes, built the same night (Cd, public 0.39208).
- Pass 3 found the pool saturated in members, and member disagreement close to noise. A best-member-per-feed oracle over real members reads 0.85 of the same oracle over noisy copies of the pool.
- Pass 3 also explained the fine-tuned half. Train-only members over-trust popularity on young listings: on train rows a young listing is counted in weeks of other users' later snapshots, on validation and test rows in few. The fine-tune re-calibrates this. A nested age offset gave only +0.0003 ± 0.0003 on G3.
- Pass 4 found that in tail feeds the positive loses to the negatives above it on every cue the ranker has (pairwise probability 0.37 to 0.47). A perfect re-order of the top 8 would recover only 29% of the validation headroom.

### The failure map of the final model (2026-11-17)

The failure map (`lab/nn_x/errfix_1117/failure_map.txt`) gave the same picture as the passes:

| Feeds | Share of feeds | Share of loss |
|---|---|---|
| 101 to 300 candidates | 37% | 43% |
| best positive outside the top item's category | half | 59% |

Sorting each category's items perfectly inside the model's category slots would lift validation from 0.40 to 0.69.

Cooldown re-runs of the 16 students aimed the loss at these parts:

- more weight on the order inside categories;
- more weight on the choice of category;
- a per-feed weight by feed size;
- up to 128 older history events for feeds at the 256-event cap.

None passed its randomization read (smallest p 0.16). None moved the part of the error it targeted beyond the spread of two reference runs (`lab/nn_x/w12/gap1117/mechanism_check.py`).

The loss aimed at the hardest feeds went the same way that evening. It multiplied a per-feed weight from the student's gap to the teacher by a lower weight on clicks the model finds surprising. It read -0.00058 ± 0.00047 on validation and -0.00074 ± 0.00028 on the holdout (`lab/nn_x/w5/cd_arms10.py`).

The probe behind that loss explains the result. The hardest 20% of feeds hold 27.5% of the loss, and 27.6% when the labels are redrawn from the model's own softmax. Most of the loss in the hardest feeds is therefore the draw of the label itself.

### How much of the error is noise (2026-11-18)

The remaining error, split by kind of noise:

- The click draw in the test labels is 67% of the per-feed variance of NDCG. On its own it moves a file's mean by ±0.0017 on validation, ±0.0021 on public and ±0.0014 on private. It is part of what is scored and cannot be removed.
- Knowing the hidden display slot of test feeds would be worth about +0.001 to +0.002, but test positions are not published.
- In the training labels, distillation already removed most of the click-draw noise (+0.0026 per member). Perfect soft labels would add at most +0.0001 to +0.0008 per file.
- Position bias in train labels does not reach the models in aggregate. Slot 0 holds 4.75% of the non-last click mass in production clicks, 2.3 to 2.7% under flat examination, 2.55% in the teacher and 2.25 to 2.57% in a student.
- The label-draw test checks calibration and sets no ceiling: a boosted-tree model at 0.370 passes it too.

One denoiser was left untested: inverse-propensity weights on the click half of the target, which weight each click by the inverse of its chance of being seen. Its CPU proxy reads -0.00017 ± 0.00025, and its prior is about 0.

### Fixable misses in pick 1 (2026-11-18)

The search of FD16P20@m0.4_0.2 found no systematic miss that a correction can fix:

- Its score behaves as a calibrated log-odds: the best temperature is 1.004 and 1.011 on two halves of the users.
- Actual over expected clicks lies within about 0.93 to 1.07 on about 660 slices.
- A perfect correction of every detected bias would be worth +0.00035 on validation.
- Corrections estimated from half of the users cost 0.0004 to 0.0009 in placebo runs, and about 50 cross-fitted corrections read 0 to -0.0013.

## What it taught

Every residual learner, gate, router and weaker model moved along one axis: it gained on feeds whose positives are cold items and lost more on warm ones. The cell of warm positives in recently visited categories (46% of positives, 42% of the loss in pass 3) responded to no lever.

The misses are atypical clicks that no available cue separates, so the limit was information. Re-weighting the loss does not add the information the misses lack. Cleaner labels would not close the gap of 0.003 on public to the leader; that needs a sharper model.

Oracles that read the label say nothing about what a predictor can do. "If the model knew which feeds have only cold positives" read +0.0253; the realistic gate read -0.00003. The same trap returned later with the privileged teacher ([distillation](../distillation/)).

On public, only new information or a new input view has paid.

Stackers stopped after pass 1, routers and post-processing after pass 2, head re-ranking after pass 4.

Not tested: a model of why users click outside their recent categories. Positives in a category new to the user are 32% of positives and 37% of the loss in pass 3. No input on record described it, and pick 1 is calibrated on those positives (model mass 0.329 against 0.322 actual, 2026-11-18).

## Files

`code/`

- `02_screen.py`: pass 1, the cheap gate for new columns.
- `common_e.py`: pass 2 loaders. The intermediate arrays were not kept, so the pass-2 scripts do not run as copied.
- `s01_decomp.py`, `s02b_local.py`, `s04b_clogit.py`, `s11_oracles.py`: pass 2 decomposition, local AUC, residual model, oracles.
- `05_mass.py`: pass 3, exact per-positive loss split.
- `tail.py`: pass 4.

`results/`

- `pass1_feature_screen.txt`: the gate on two bases.
- `pass2_loss_decomposition.txt`, `pass2_local_window_auc.csv`, `pass2_residual_models.txt`, `pass2_oracles.txt`: pass 2.
- `pass3_loss_mass.txt`, `pass3_oracles.txt`, `pass3_levers_cold_warm.txt`: pass 3.
- `pass4_tail_anatomy.txt`: pass 4.

### Raw material and records

- Passes 1 to 4: `lab/eda/errors/` (four lenses and synthesis), `lab/solvers/errors/` with its CSV read-outs in `lab/solvers/out/errors/`, `lab/nn_x/insights/errors/` (`findings.txt`, `run_all.sh`) and `lab/nn_x/insights/tail_1113/`. The raw logs and the intermediate arrays are not in this repository.
- The second error analysis of the final model (2026-11-17): `lab/nn_x/errfix_1117/`, with `RESULT.txt`, `BRIEF.txt`, the analysts' notes and rule files.
- The loss rounds aimed at its failure map: `lab/nn_x/w12/gap1117/` and `lab/nn_x/w5/cd_arms10.py`.
- The search of pick 1 for fixable misses (2026-11-18): `lab/nn_x/w12/fixmiss/`.
- Reports under `lab/reports/`: `err_segments.md`, `err_category_intent.md`, `err_missed_positives.md`, `err_feed_structure.md`, `err_SYNTHESIS.md`, `sol_errors.md`, and `INSIGHTS_1112.md`.
- In context: [what the data showed](../../docs/02-what-the-data-told-us.md#where-the-ranking-error-sits), [what failed](../../docs/07-what-failed.md).
