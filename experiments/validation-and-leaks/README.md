# Validation and leaks

In brief:

- In train, a user's next history snapshot holds 92.1% of the positives of the earlier feed. An overlay built on that gained +0.00454 on validation and nothing on public (0.392819 against 0.392820). Since 2026-11-11 no input reads later snapshots.
- Validation predicts a step between model generations one for one but cannot order near copies. The holdout predicted public differences until it gave the wrong sign twice for members with lighter popularity inputs.

## Why it was tried

The line had two aims: to find which signals in the data are leaks, and to learn how far each of three samples can be trusted when two files differ by less than 0.001 NDCG.

- Validation: 14,423 labelled feeds.
- The holdout: 30,000 random-order train feeds kept out of training, all with at least 50 candidates.
- The public board: 8,562 test feeds.

The first reason was the way histories arrive. A user's history is a snapshot taken at feed time, and 18% of train feeds have a later feed of the same user. If the later snapshot contains what the user clicked in the earlier feed, every count built from histories can carry the label.

The second reason came on 2026-11-12. The pool with the best validation score, M2 (validation 0.40094), scored 0.392434 on public. That is below the pool kept, G3 (20 click-label rankers, validation 0.40032, public 0.392820). The line had to find out which of the three samples to believe.

A pool is the mean of its members' within-feed z-scores; the [glossary](../../docs/glossary.md) defines this and the other terms.

## What ran

All of it was CPU work on the data and on stored predictions. Two public submissions went to the leak.

2026-11-11:

- Alignment of 571,923 pairs of consecutive train feeds of one user (`02_snapshot.py`), and cross-feed flags for every validation and test candidate (`03_leak.py`).
- An audit of every validation feature by its single-feature NDCG on the 117 validation feeds whose user has a later feed, against the other 14,306.
- Seven written reviews: metric, leakage, overfitting, network maths, shift, integrity and ideas. A second reader re-checked their 24 findings.

2026-11-12:

- The leak measured on G3 and projected to test (`01_sameuser_leak.py`).
- The overlay sent to the board (`later_overlay.py`). The overlay is a post-hoc boost of items found in the same user's later history.
- A noise model for public differences (`noise.py`).
- A bug hunt. It recomputed four members through both training code paths, bit-identical to the stored predictions, and checked the 152 input files they read.

2026-11-13:

- A validation-to-public study: 40 pre-specified re-weightings of validation, transfer models over 7 scored pools (`c3_transfer.py`) and 400 random half-splits of validation (`s05_dispersion.py`).
- An audit of the input pipeline: a CPU rebuild of the inputs, 94 label-free constructions and 78 second-level combiners read as residuals on G3.

## Results

"±" is one paired standard error (SE) over feeds.

### Leak measurements

| Reading | Sample | Value |
|---|---|---|
| Positives of a feed found in the same user's next history (negatives) | train, 571,923 feed pairs | 92.1% (0.40%) |
| Flagged candidates that are positive | validation, 184 candidates in 110 feeds | 93.5% |
| Flagged items pushed to the top of G3 | validation, paired | +0.00454 ± 0.00047 |
| The same overlay | public | 0.392819, against 0.392820 without it |
| Boost on the day-one blend (`v5`) | public | 0.3802 with and without |
| Test feeds with a later feed of the same user | test | 59 of 8,562 public, 49 of 20,033 private |
| ALS factors trained on later-snapshot pairs: the 117 feeds, the rest | validation, single-feature NDCG | 0.583, 0.329 |

### Validation, holdout and public

| Reading | Sample | Value |
|---|---|---|
| Expected paired SE of a public difference between two pools | noise model | 0.0003 to 0.0007 |
| Public against validation, 11 scored files (4 early blends, 7 pools), early fit | public | 0.0071 + 0.964 (± 0.033) × validation, residual sd 0.0008 |
| Spread of 7 near-copy pools (rank correlation of the two orders +0.04) | validation; public | 0.00198; 0.00074 |
| M2 and S2 (best pool on validation, stacker trained on validation) against their validation prediction | public | -0.0010 ± 0.0005; -0.0009 ± 0.0004 |
| Holdout difference taken 1:1 as the public difference | 4 contrasts | chi2(4) 0.27 |
| Popularity-light students added at 0.2 (W8P20 against W8) | validation; holdout; public | +0.00049; -0.00063; +0.00052 |

The level line in the table is the early fit on 11 files. The line in use from 2026-11-14, fitted on 18 files, is public = 0.0074 + 0.963 × validation with residual sd 0.0006 ([docs/04](../../docs/04-validation.md#the-level-line)).

W8 is the pool of 8 held-out full-view distilled students of width 192. W8P20 adds 4 narrow popularity-light students at weight 0.2 (public 0.3950094). It was pick 2 from 18 to 19 November, until FDX16P20:ps (public 0.39526061) replaced it on 2026-11-19. A popularity-light student reads one clipped exposure count in place of the 8 popularity inputs.

## What it taught

### The leak

The leak is real in the data and worth nothing on the board. The overlay that gained 0.00454 on validation moved public by 0.000001. The full account is in [docs/02](../../docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak).

Since 2026-11-11 nothing from a user's later snapshots enters any input. Popularity counts leave the same user out, and leaky features and predictions sit in quarantine folders.

On day one the leak had entered through ALS factors and through trees trained on popularity counts that included later snapshots. It inflated the best validation score of that day by about 0.002 (0.3869 read, about 0.385 clean).

The [pre-submission gate](../../docs/04-validation.md#the-pre-submission-gate) repeats the 117-feed probe on every file. The probe has little power. It failed the first pool of three recipes, Cd, at z 2.0 (+0.0123 ± 0.0061), while a controlled pair that differed only in the suspect input read -0.0009 ± 0.0039. After that trigger the fail threshold moved to z > 3, and the change was logged as such.

### What each sample can be trusted for

Validation predicts a step between model generations (0.002 or more on validation, see the [glossary](../../docs/glossary.md)) one for one. It cannot order near copies, pools with a within-feed correlation of at least 0.995 to the reference and a validation difference of at most 0.002.

The expected excess of the best of K near copies on validation is +0.00045 for K = 5 and +0.00087 for K = 50, the size of the shortfall of M2 and S2. None of the 40 re-weightings reproduced that shortfall. Selection on validation explains it.

The holdout predicted the public differences with no fitted parameter: chi2(4) 0.27, while the calibration script reads chi2(5) 0.72 on five contrasts. From 2026-11-13 near copies were ranked on it.

The holdout then gave the wrong sign twice for members with lighter popularity inputs:

- C3, the click-label pool with 8 such members: holdout -0.00073 against G3, public +0.000634;
- W8P20 against W8, in the last row of the table under [Validation, holdout and public](#validation-holdout-and-public).

The holdout's popularity inputs are 1.75 days stale on average, against 5.7 days on validation, 9.3 on public and 11.2 on private. Staleness counts the days since a warm item was last a train candidate.

A witness, a sample on which a change is judged, can be biased for a whole family of ideas. Later lines therefore read validation and the holdout together with a stale-input replay of validation ([popularity-staleness](../popularity-staleness/)). The policy that followed is in [docs/04](../../docs/04-validation.md#rules-written-before-the-read).

The validation-to-public study rests on 7 files that share one reference. That is enough for one statement: near-copy validation differences do not transfer 1:1.

### The pipeline audit

The pipeline audit found nothing to fix.

- The CPU rebuild matches the stored predictions at correlation 0.999999.
- 6 of the 94 constructions reached z > 2, where chance gives 2.1.
- 0 of 78 combiners passed on both validation and the holdout.

## Files

`code/`

- `common.py`: loaders and the paired NDCG read used by the two snapshot scripts.
- `02_snapshot.py`: alignment of consecutive histories of one user.
- `03_leak.py`: cross-feed flags for validation and test, their target rates, validation effect and test coverage.
- `01_sameuser_leak.py`: the leak measured on G3 and projected to public and private.
- `later_overlay.py`: the overlay scored on the board.
- `noise.py`: noise model for public differences and the 8-file level line.
- `c3_transfer.py`: one-parameter models for the gap between validation and public differences.
- `s05_dispersion.py`: spread of 7 pools on validation and public; half-split null of the transfer slope.

`results/`

- `snapshot_pairs_train.txt`, `crossfeed_flags_val_test.txt`: read-outs of the two snapshot scripts.
- `later_leak_on_best_pool.txt`: validation gain, coverage and projection of the overlay.
- `leak_boost_public_scores.csv`: the board readings with and without the boost.
- `public_noise_model.txt`, `val_public_transfer.txt`, `val_public_dispersion.txt`: read-outs of the three validation scripts.
- `pools_val_holdout_public.csv`: validation, holdout and public score of each pool in the study (copy of `lab/nn_x/audit/valpub/landscape/out/s1_matrix.csv`).

### Raw material and records

- Scripts and read-outs: `lab/eda/history_leak/`, `lab/nn_x/audit/` (15 audit folders, most with a `findings.txt`), `lab/nn_x/pubdiag/` (noise model and bug hunt). The raw logs and the arrays are not in this repository.
- The gate: `lab/src/presub_gate.py`. Its port is `avito_ranker/validation/gate.py`, which `scripts/09_check_submission.py` runs.
- Reports: `lab/reports/eda_history_leak.md`, `lab/reports/review_*.md` and `lab/reports/REVIEW_SYNTHESIS.md`, `lab/reports/AUDIT_1112.md`, `lab/reports/VAL_PUBLIC_1113.md`, `lab/reports/DATA_PIPELINE_1113.md`.
- Records: the status log (a working record, not published).
