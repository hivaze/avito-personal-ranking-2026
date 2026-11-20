# Fine-tuning on the validation days

In brief:

- Each member is trained further on the 14,423 validation feeds, in 5 folds grouped by user, and the final file mixes the fine-tuned and train-only halves at 0.5 / 0.5.
- On public, W8 (a pool of 8 distilled students) scored 0.394486 with that mix, 0.394009 with the train-only half alone and 0.393960 with the fine-tuned half alone. Validation reads the fine-tuned half low, and every refinement of the recipe was null at pool level.

Terms such as member, pool and out of fold are defined in the [glossary](../../docs/glossary.md).

## Why it was tried

Validation is the only labelled data shown in random order, and it comes from just before the test period. The question was how a ranker trained on production-ordered train feeds should learn from the 14,423 validation feeds, and what the fine-tuned half is worth in the final file.

A ranker trained from scratch on validation alone was worse than one fine-tuned from the train-fitted checkpoint, so the train stage stays (`lab/reports/nn_report.md`):

| Validation fold | From scratch on validation | Fine-tuned from the train checkpoint |
|---|---|---|
| 0 | about 0.352 | 0.365 |
| 1 | about 0.366 | 0.372 |

The fine-tune can also use inputs that exist only for validation and test rows, such as co-visitation over the histories of the scoring period.

## What ran

`code/finetune.py` starts from a train-only checkpoint and adds a zero-initialised branch for 42 extra inputs. The branch trains at 10 times the learning rate of the rest of the network.

| Setting | Value |
|---|---|
| Folds | 5, grouped by user; each of the 658 feeds of user 0, the anonymous bucket, is its own group |
| Optimiser | AdamW at 5e-5, cosine decay over 4 epochs |
| Weights that predict | an exponential moving average (EMA) of the weights from epoch 0.5 |
| Seeds | 3 per fold |
| Prediction for a validation row | the 3 models that did not see it |
| Prediction for a test row | all 15 models |
| A pool's file | 0.5 z(fine-tuned pool) + 0.5 z(train-only pool), z taken per feed |
| Cost | 15 to 22 minutes per member on one GPU; the 20 fine-tunes of the scored all-data file logged 398 GPU-minutes |

Round 1 on 2026-11-11 shaped the recipe. Each change below was read out of fold on validation, on a different checkpoint:

| Change | Before | After |
|---|---|---|
| average the predictions of epochs 1 to 4 instead of picking the best epoch | 0.37530 | 0.37704 |
| learning rate 5e-5 instead of 1e-4 | 0.37995 | 0.38068 |
| the larger extra branch (co-visitation, ALS item profiles, near-duplicate context) | 0.38162 | 0.38535 |

Cosine decay with a weight average came from [optimiser-and-modern-tricks](../optimiser-and-modern-tricks/).

Six studies of the recipe followed, each read against the recipe as it stood. They use these terms:

- The stale view of validation replays it with the popularity inputs set to test-like stale values.
- The private-mix view re-weights validation feeds to the feed-size and cold-share mix of public or private.
- W8 is the first file of 8 full-view distilled students (width 192). W8P20 adds 4 popularity-light students at weight 0.2 (public 0.3950094). It was pick 2 from 18 to 19 November.

## Results

| Study | Date | Sample | Result | Decision |
|---|---|---|---|---|
| weight w of the fine-tuned half, nested over 5 user folds, 19 members | 11-11 | validation | all folds chose 0.67; nested +0.00024 ± 0.00025; with the fine-tune's folds -0.00021 ± 0.00024 | kept 0.5 |
| Optuna re-tune, 35 trials scored on half the validation users | 11-12 | the other half | best trial +0.00055 ± 0.00041; on new members -0.00009 ± 0.00027 | kept the recipe |
| 3 seeds per fold instead of 1, 28 members | 11-13 | validation | per member -0.00000 ± 0.00004; 20-member file +0.00018 ± 0.00012 | kept 3 |
| fine-tune on the stale view of validation, 8 members | 11-15 | validation / stale view | file +0.00022 ± 0.00026 / +0.00054 ± 0.00028; in the form of W8P20 +0.00012 ± 0.00025 / +0.00015 ± 0.00024 | not adopted |
| weight curve on the final pools | 11-15 | validation, private-mix view | optimum 0.32 to 0.57, gain over 0.5 at most +0.00006 | kept 0.5 |
| W8 at w = 0, 0.5, 1 | 11-14 to 11-16 | public | 0.394009, 0.394486, 0.393960 | kept 0.5 |

A parabola through the three public points peaks at w = 0.49. On test the two halves are equally strong, and their equal mix is worth about +0.0005 over either (paired SE 0.00052 to 0.00055).

## What it taught

Validation under-reads the fine-tuned half. Out of fold the half reads +0.00009 ± 0.00043 for the 8-student pool, against +0.000477 on public.

The ledger of 2026-11-15 02:15 UTC blamed fold averaging: 3 models predict a validation row, 15 a test row. The review of that evening is the newer record (`results/weight_review_synthesis.txt`, 21:25 UTC). It measured the fold term at only +0.00006 to +0.00008.

The same review found most of the gap in the feed mix. Validation re-weighted to the public mix reads +0.00066 ± 0.00054, and validation and public together give +0.00051 ± 0.00030 for the half on public.

Every refinement was null at pool level:

- The re-tune's best trial dropped the extra branch altogether. Its gain shrank from +0.0009 on the users it was tuned on to +0.0005 on the others and zero on new members.
- Three seeds per fold left each member unchanged. They were kept because they cost minutes.
- The stale-view fine-tune traded fresh validation for stale-view accuracy at member level (-0.00033 ± 0.00020 and +0.00026 ± 0.00023). It added nothing once the popularity-light students were in the file, since they supply the same kind of difference ([popularity-views](../popularity-views/)).
- Test pseudo-labels (-0.0004 on validation) and one model refitted on all of validation (-0.00025 ± 0.00018) also failed.

The fine-tune does not recover what stale inputs lose, because the loss is information they do not carry. Stale inputs cost the final file -0.0017 ± 0.0005 on the stale view, and four epochs on validation recovered at most +0.0005 of it.

## Files

`code/`

- `finetune.py`: the fine-tune of every full-view final member; command line in the header.
- `weight_nested_check.py`: nested weight choice on user folds.
- `retune_optuna_driver.py`, `retune_confirm.py`: the re-tune and its read-out on held-back users.
- `three_seeds_readout.py`: 3 seeds against 1.
- `stale_finetune_patch.py`, `stale_finetune_readout.py`: builds the stale-view fine-tune by exact text replacements; its read-out.
- `weight_curve_review.py`: weight curve on the final pools.

`results/`

- `weight_nested_check_rule.txt`, `weight_nested_check.txt`: the check of 2026-11-11.
- `retune_findings.txt`, `retune_trials.csv`, `retune_tune_vs_confirm.txt`, `retune_confirm_trial21.txt`: the re-tune.
- `three_seeds_result.txt`, `three_seeds_readout.txt`: 3 seeds.
- `stale_finetune_w8.txt`, `stale_finetune_best_file_form_fresh.txt`, `stale_finetune_best_file_form_stale.txt`: the stale view.
- `weight_curve_review.txt`, `weight_review_synthesis.txt`, `weight_review_seed_averaging.txt`: the review of 2026-11-15.

### Raw material and records

- `lab/nn/finetune.py`, with byte-identical copies in `lab/nn_x/w5/` and `lab/nn_x/r6/`. The popularity-light members ran their own copy, `lab/nn_x/abl/popfree/finetune.py`.
- `lab/solvers/r5_wft.py` with `lab/solvers/out/r5_wft/`.
- `lab/nn_x/ftopt/`, `lab/nn_x/abl/ftbag/`, `lab/nn_x/w12/stale/staleft/`, `lab/nn_x/w12/ft_1115/`.
- Reports: `lab/reports/r5_wft.md`, `lab/reports/STALE_1115/staleft.md`, `lab/reports/nn_report.md` (round 1).
- Records: the decision ledger (a working record, not published).

In context: [the solution, fine-tune](../../docs/03-solution.md), [validation](../../docs/04-validation.md).
