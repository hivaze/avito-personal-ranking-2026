# Gradient-boosted trees and stackers

In brief:

- Gradient-boosted trees (GBDT) were tried as rankers of their own, as a second stage over the neural scores and as extra members of the neural pool. Neural stackers were tried against the plain mean of the pool.
- Trees stopped about 0.012 below the clean neural ensemble on validation. The tree stacker added nothing to the next neural generation, and no CatBoost role helped the neural pool. Every tree role was closed on 2026-11-12.
- The neural gate stacker read +0.00056 on nested validation and lost 0.00035 on public. From then on every pool was a member-equal mean.

A stacker is a second-stage model fitted on top of the scores of first-stage models. Other terms (pool, holdout, G3, train-only) are in the [glossary](../../docs/glossary.md).

## Why it was tried

Trees are the usual first model for tabular ranking. The winning solution of the 2025 Avito cup fed item-item scorers and target statistics to a tree ranker, and [external-solutions](../external-solutions/) has the team's transfer of it.

A stacker fitted on validation can use labels that train-only members never see.

## What ran

### Tree rankers and the residual stacker (2026-11-11)

- Tree rankers were fitted on 300,000 to 1,000,000 train feeds and early-stopped on validation (`train_lgb.py`, `train_xgb.py`, `train_cb.py`; 6 to 55 minutes each).
- A separate study fitted 12 tree models 5-fold on validation to compare objectives, pair settings and averaging (`lab/reports/deep_gbdt_ranking.md`). Its scripts ran in a scratch folder and are not in the tree.
- `stack2.py` is a residual XGBoost stacker. The neural train-only score is a fixed margin, and trees fitted 5-fold on validation by user learn a correction from about 40 features.

### CatBoost in every role (2026-11-12)

CatBoost was tested in every role against G3, the pool of 20 networks trained on click labels, under a rule written before the runs (`cbrule.py`):

- a residual trained on the 30,000 holdout feeds (23 GPU-minutes);
- the same residual nested inside validation (26 CPU-minutes);
- a YetiRank member trained on 134,118 random-order train feeds (51 GPU-minutes);
- the nine tree and stacker files on disk, added to the pool.

### Neural stackers (2026-11-12)

The same day, `nnstack_train.py` fitted neural stackers over G3's member scores, in forms from linear to a per-feed gate. They were nested inside the fine-tune's five user folds: 3 replicates, 5 to 8 minutes each.

## Results

Validation is 14,423 feeds. The holdout is 30,000 random-order train feeds the pool members never saw. Differences are paired per feed, ± one standard error.

| Model or change | Sample | Result |
|---|---|---|
| best tree fitted on train (CatBoost, 300,000 feeds, leaky counts) | validation | 0.36997 |
| 12 trees fitted on validation, z-score mean, over the best single | validation, out of fold | 0.3705, +0.0018 ± 0.0008 |
| best blend of the first neural ensemble (0.38186) with a tree | validation | 0.3820 |
| trees with counts that include the users' later snapshots, on feeds whose user has a later feed | validation | +0.025 to +0.052 |
| residual stacker on leak-free inputs; on the next neural generation | validation | 0.38491; -0.0001 ± 0.0006 |
| submissions v3, v4, v5 (network with stackers and trees) | validation / public | 0.3816 / 0.3739, 0.3839 / 0.3772, 0.3869 / 0.3802 |
| CatBoost residual trained on the holdout, on G3 | validation / holdout | -0.00031 ± 0.00029 / -0.00042 ± 0.00018 |
| CatBoost residual nested on validation | validation | -0.00009 ± 0.00034 |
| CatBoost member on random-order feeds: alone; added at weight 0.05 | validation / holdout | 0.38201 / 0.30424; +0.00015 ± 0.00024 / -0.00016 ± 0.00014 |
| neural gate stacker on G3, mean of 3 replicates | validation, nested / holdout | +0.00056 ± 0.00028 / -0.00006 ± 0.00012 |
| the gate stacker as a file (S2), against G3 | public | 0.392473 against 0.392820 |

## What it taught

Trees stopped about 0.012 below the clean train-only neural ensemble of the same morning, which read 0.3825 on validation. The CatBoost member correlated 0.856 with the pool on validation. At that correlation it would have needed a validation score of about 0.393 to break even.

Every tree role was closed on 2026-11-12. A second CatBoost round on 2026-11-15 confirmed the closure ([catboost-round-2](../catboost-round-2/)).

The objective made little difference on this data. Logistic, pairwise and listwise objectives landed within ±0.001 of each other on out-of-fold validation, at a paired SE of about 0.0011.

Library defaults moved the score more. XGBoost's ranking objective with its defaults, 32 pairs per row on feeds of up to 300 candidates, cost 0.0052 ± 0.0013 on out-of-fold validation against a logistic objective.

Averaging integer ranks creates exact ties, and the metric is pessimistic about ties: it orders tied candidates with the negatives first. Two CatBoost models read 0.3619 by rank mean and 0.3675 with a tie-break, on out-of-fold validation. Per-feed z-scores, the form kept for every pool, have no such ties.

The trees also learned the [same-user later-snapshot leak](../../docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak). Counts that included the users' later snapshots taught them that a later increment means a click. With the leaky ALS factors, this made up about 0.002 of the third submission's validation score of 0.3869. Afterwards only leave-user-out counts were used.

The leak-free stacker read 0.38491 on validation, below the fine-tuned network of the same day (0.38634), and added nothing to the next neural generation. The stacker line stopped that morning.

The neural gate stacker read +0.00056 on nested validation and lost 0.00035 on the public board. For a combiner of near-identical members, nested validation did not predict the board. From then on every pool was a member-equal mean.

A display-position term gave LightGBM +0.0022 ± 0.0011 on validation (one seed), while XGBoost's inverse-propensity weights cost 0.0022 ± 0.0010. In the network a position tower was neutral: 0.3736 on validation with it, 0.3739 without.

## Files

`code/`

- `train_lgb.py`, `train_xgb.py`, `train_cb.py`: tree rankers fitted on train.
- `stack2.py`: residual XGBoost stacker.
- `cbrule.py`: the written rule of the CatBoost tests and its check.
- `cb_resid.py`: CatBoost residual on the pool.
- `cb_member.py`: CatBoost member on random-order feeds.
- `nnstack_train.py`: neural stackers, including the submitted gate.

`results/`

- `summary.csv`: the table with more rows, each with its source.
- `early_trees.csv`, `residual_stacker_runs.csv`: every early tree and stacker run with its arguments and score.
- `cb26_ana_existing.txt`, `cb26_eval_e1e2.txt`, `cb26_eval_e3.txt`: read-outs of the CatBoost roles.
- `stack_gate_combine.txt`, `stack_holdout_fit_compare.txt`, `stack_holdout_fit_stability.txt`, `stack_pool_of_pools.txt`: gate stacker, family weights fitted on the holdout, pools of pools.

### Raw material and records

- Code and read-outs: `lab/src/` (the trainers, `stack2.py`, `cv_val.py`), `lab/logs/`, `lab/nn_x/cb26/`, `lab/nn_x/stack/`.
- Reports: `lab/reports/deep_gbdt_ranking.md`; the complementarity section of `lab/reports/nn_report.md`.
- Records: the status log and the experiment registry (working records, not published).
- In context: [what failed](../../docs/07-what-failed.md), [the timeline](../../docs/05-timeline.md).
