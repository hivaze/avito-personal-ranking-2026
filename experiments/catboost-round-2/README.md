# Gradient-boosted trees, round 2 (CatBoost)

In brief:

- The round asked whether a CatBoost model could join the final neural pool once levers never used before were switched on. These were teacher targets in its native ranking losses, its native embedding, text and target-statistic features, and hand-made label statistics from all train feeds.
- No arm passed. The best single arm, B3, read +0.00147 / +0.00080 nats on the screen against a bar of 0.0016 on both, and no member-rule read reached a pooled z of 2. The highest were +0.65 for the B3 bundle and +0.67 for a reduced-scale family N run.
- The stop rule closed the tree track at 2026-11-15 07:03 UTC. The round ran on CPU during the night of 2026-11-14 to 15.

Scores are NDCG on validation (14,423 feeds) and on the holdout (30,000 random-order train feeds that neither the pool nor the trees trained on). A pair "a / b" is validation / holdout, and ± is one paired standard error (SE). Pool, member, arm, student, teacher, witness and pooled z are defined in the [glossary](../../docs/glossary.md).

## Why it was tried

No tree had helped a neural pool in the first round ([gbdt-and-stacking](../gbdt-and-stacking/)). The audit at the start of this round found all 13 tree models with holdout predictions below the member rule defined below (pooled z -1.40 to +0.64).

The levels to beat, before the round:

- The best tree on record, CatBoost YetiRank (one of CatBoost's native ranking losses) on 134,118 random-order train feeds with 177 tabular features: 0.38201 / 0.30424.
- A LightGBM student fitted to the teacher's scores: 0.3847 on validation.
- The reference pool W16, 16 full-view distilled students of width 192 with train-only predictions: 0.40144 / 0.32460.

The CatBoost documentation listed feature processing that had never been tried here, and a CPU round cost no GPU time.

### Rules written before the runs

The rules were fixed before any new tree was trained (`results/predeclared_rules_extract.txt`). Here z(x) is the within-feed z-score of a score x.

- The member rule was the only decision for a file. It compares 0.9 z(W16) + 0.1 z(tree) with z(W16). The difference had to read at or above 0 on validation and on the holdout, and the pooled z = (z_val + z_holdout) / sqrt(2) had to be at or above 2.
- The screen (`nats.py`) is the gain in per-feed conditional-logit loss (a softmax over the feed's candidates), in nats, when the tree's score is added to the pool's score. It counts click information the tree has and the pool lacks.
- An arm was promising at 0.0016 nats or more on both witnesses. The bar came from the audit's what-if, where a borderline partner read 0.0020 / 0.0016.
- If no arm and no declared bundle reached the screen, the tree track closed (the stop rule).
- Tree counts were chosen on a split of train feeds, never on validation.

## What ran

Everything ran on CPU with 12 threads per process. The arms fall into three groups, called families B, S and N after their lab folders (`famB`, `famS`, `famN`). Here a family is a group of arms, not a seed family in the glossary sense.

1. Preparation: two maps of the CatBoost levers (one of them with 632 synthetic checks on version 1.2.10), a CPU baseline with train-side selection (`cbx.py`), and an audit of why trees lose, with a what-if read of hypothetical partners (`au_paths.py`).
2. Family B, click labels plus label information the pool lacks, in seven fits of 45 to 101 minutes each:
   - CatBoost's native target statistics (built-in label means per category value) on cluster keys and on the item ID. A noise-label test (`b1_noise3.py`) first checked them for leaks.
   - 70 hand-made columns from all 3.12M non-holdout train feeds (`b3_build_a.py`, `b3_arm.py`), such as a user's click rate on the candidate's category in strictly earlier feeds.
3. Family S, CatBoost students of the neural teacher (`run_s.py`): seven recipes (RMSE, QuerySoftMax or YetiRank on teacher targets, depth 10, a raw-embedding block), the best at 3 seeds. It took 605 minutes of 12-thread fitting.
4. Family N, CatBoost's own feature processing (`famN_run.py`): LDA and KNN embedding calcers (CatBoost's feature builders for vector columns), text token sets, finer borders, one-hot encoding and feature selection. It made 34 runs at 30,000 or 6,000 training feeds against references of the same size.
5. Three adversarial reviews re-derived every number with independent code.

## Results

NDCG on validation and on the holdout. Screen and member-rule reads are against W16, ± one paired SE.

| Model | Validation | Holdout | Screen in nats, validation / holdout (bar 0.0016) | Member rule, pooled z (needs 2) |
|---|---|---|---|---|
| B0, CatBoost on click labels | 0.38150 | 0.30462 | +0.00117 ± 0.00038 / +0.00051 ± 0.00017 | -0.36 |
| B3, B0 + 70 label statistics | 0.38362 | 0.30480 | +0.00147 ± 0.00043 / +0.00080 ± 0.00021 | +0.30 |
| B3, two seeds as one bundle | 0.38476 | 0.30575 | +0.00151 ± 0.00043 / +0.00083 ± 0.00021 | +0.65 |
| B1, native statistics on cluster keys | 0.38148 | 0.30480 | +0.00096 / +0.00054 | -0.57 |
| B2, native statistics on the item ID | 0.37947 | 0.30159 | +0.00112 / +0.00051 | -0.75 |
| S, declared 3-seed student bundle | 0.38461 | 0.30797 | +0.00056 ± 0.00026 / +0.00040 ± 0.00015 | -0.52 |
| N, 34 runs (level of the best single run) | 0.38346 | 0.30686 | at most +0.00111 / +0.00050 | -1.42 to +0.67 |

The audit's what-if read two kinds of hypothetical partner:

- A tree that absorbs part of what the pool already knows never passes, at any level up to 0.4006 on validation (pooled z -0.5 to +0.4).
- A tree that adds label signal the pool lacks passes at about 0.3867 on validation (pooled z +3.5).

## What it taught

Trees on this data are limited by their inputs. A tree fitted to the W16 score itself from the 177 features plateaued at 0.3865 on validation, with correlation 0.909 to the pool. So no loss or optimiser on these inputs closes the gap.

The students show the same limit. They sit 0.0031 above the click-label CatBoost on validation and still carry less of what the pool lacks. Given the pool and the click-label tree, they add +0.00007 ± 0.00010 nats on validation.

Family B found real extra information, but too little of it. B3's holdout screen is 3.8 SE below the bar, and in a linear read its 51 click statistics add +0.00004 ± 0.00007 nats on validation. Those statistics exist for 18 to 30% of feeds and are 10 to 15 days old on validation and test.

No native lever of family N had an effect. The LDA direction is the cosine the tree already has (correlation 0.67), and the KNN teacher means are a noisy copy of what the pool knows.

### Facts about CatBoost itself

- Native target statistics with the default 15 borders cannot resolve click rates near 1%. The round set 255.
- A native statistic on a user-level key leaks on CPU. On noise labels it reached a learn NDCG of 0.558 against 0.266 for a leak-free twin, while candidate-level keys showed no excess (+0.002 ± 0.008). User-level native keys were never used.

### The closure and what was not run

The stop rule closed the tree track at 2026-11-15 07:03 UTC. Validation alone was null for every arm, so the closure does not rest on the holdout.

Not run:

- Feeding an out-of-fold tree score to the neural students. It needed a promising arm.
- Native KNN with k = 200, at over 4.4 ms per row on one thread.
- B3 refitted on statistics lagged like test. Only a review ran it, at reduced scale: +0.00024 ± 0.00028 on validation and -0.00023 ± 0.00012 on the holdout against fresh statistics.

## Files

`code/`

- `nats.py`: the shared screen and member-rule read.
- `cbx.py`: data, features, train-side selection, leak guards and the evaluator used by every arm.
- `au_paths.py`: the what-if read of hypothetical partners.
- `b3_build_a.py`, `b3_arm.py`: one block of hand-made label statistics and the best arm of the round.
- `b1_noise3.py`: the noise-label leak test of native target statistics.
- `run_s.py`: the student runner.
- `famN_run.py`: the runner of the native feature levers (lab file `run.py`).

`results/`

- `predeclared_rules_extract.txt`: the rule lines written before the runs.
- `audit_level_table.txt`, `audit_ceiling.txt`, `audit_whatif_paths.txt`: level of every tree on disk, the student run to its plateau, the what-if paths.
- `famB_final_lines.txt`, `famB_native_ctr_noise_test.txt`: family B screen lines and the leak test.
- `famS_final_reads.txt`, `famN_nats_lines.txt`: screen lines of families S and N.

### Raw material and records

- Scripts and read-outs (JSON and text): `lab/nn_x/w12/cb/` (`audit/`, `infra/`, `famB/`, `famS/`, `famN/`, the three review folders, `PREDECLARED.txt`). The raw logs and the prediction arrays are not in this repository.
- Reports: `lab/reports/CATBOOST_1115/` (`audit.md`, `infra.md`, `docs_rank.md`, `docs_feat.md`, `famB.md`, `famS.md`, `famN.md`, and the reviews `famB_review.md`, `famS_review.md`, `famSN_review.md`).
- Records: the decision ledger (a working record, not published).
- In context: [what failed](../../docs/07-what-failed.md).
