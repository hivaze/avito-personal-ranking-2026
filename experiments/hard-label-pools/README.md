# Pools of click-label rankers

In brief:

- Before distillation, the team's files were pools of 14 to 38 rankers trained on click labels. From 2026-11-11 to 2026-11-13 this line asked which rankers go into a pool and how their scores are combined.
- No fitted combiner beat an equal mix of three families declared in advance, so the member-equal mean of per-feed z-scores stayed. The gain came from pooling families trained with different recipes.
- The pool best on validation (M2) and a stacker fitted on validation (S2) both scored below G3 on public. G3 (public 0.392820) stayed the pick of this generation.

A pool is the mean over members of their within-feed z-scores, with equal weights. A family is a set of members that share a recipe and differ only by seed. These and the lab code names (F5, F5cd, Cd, Pall, PF7x, G3, M2, S2) are in the [glossary](../../docs/glossary.md).

## Why it was tried

More seeds of one recipe stopped paying early. Six train-only members of one recipe (F5) lifted validation from 0.39247, the mean single member, to 0.39712. A fitted curve put unlimited seeds at 0.39774 on validation and 0.32232 on the holdout.

The 12 predictions of such a family (6 members, fine-tuned and train-only) behaved like 1.1 independent models, and train-only members correlated 0.936 per feed. Further gain had to come from families trained differently.

Pooling was also the easiest place to overfit. Any weight or member choice fitted on the 14,423 validation feeds is read on the same feeds.

## What ran

Everything here was CPU work on stored member predictions. The members are 10-epoch rankers whose GPU cost belongs to the training lines.

A pool's z-mean is taken separately over the fine-tuned and the train-only predictions of its members. The file is 0.5 z(fine-tuned mean) + 0.5 z(train-only mean).

1. 2026-11-11, design study (`ens_common.py`, `exp1_combiners.py`, `exp3_diversity.py`, `exp7_pairs.py`). Its 11 experiments took 10 s to about 25 minutes each. They covered combiners, the fine-tune weight, stackers in nested user folds, per-feed correlations between families, and a law for when a second family helps.
2. 2026-11-11, first cross-recipe pools (`CPU1_xrecipe_member_equal_pool_v3.py`, 100 s). The families, a strength gate and the acceptance rule were written before the run. The gate drops a family more than 0.002 below the best on validation or on the holdout. The pool was chosen on the holdout only.
3. 2026-11-12 and 13, decision rules: a series of scripts (`decide4.py`, `decide5.py`), each stamped before any number of its new members existed. Pools were written with `build_pool.py`.

## Results

NDCG of the pools of this generation, with the lab code name in brackets. The holdout is 30,000 random-order train feeds that no member trained on.

| Pool | Members | Validation | Holdout | Public |
|---|---|---|---|---|
| First pool of three recipes (Cd) | 19 | 0.39901 | 0.32319 | 0.39208 |
| Six families (Pall) | 38 | 0.39896 | 0.32335 | 0.392436 |
| Six families, one with a cooldown re-run (PF7x) | 38 | 0.39974 | 0.32360 | 0.392723 |
| GRU family and two families with a cooldown re-run (G3) | 20 | 0.40032 | 0.32379 | 0.392820 |
| Only members with a cooldown re-run, best on validation (M2) | 14 | 0.40094 | 0.32361 | 0.392434 |
| Neural gate stacker on G3 (S2) | 20 | 0.40088 | none | 0.392473 |
| G3 without its fine-tuned half | 20 | 0.40035 | 0.32379 | 0.392346 |

A cooldown re-run means that the last training phase was run again from a stored checkpoint with 2% random-order feeds.

Against G3 the validation differences have paired SEs of 0.00028 to 0.00049. The expected paired SE of a public difference is 0.00035 to 0.00062.

Combiners on one 6-member family, as the paired validation difference against the z-mean (± one standard error):

| Combiner | Validation difference |
|---|---|
| Raw-logit mean; softmax-probability mean | +0.00004 ± 0.00015; +0.00005 ± 0.00021 |
| Rank mean; reciprocal rank fusion; Gaussianised ranks | -0.0020 ± 0.00035; -0.0007 ± 0.0003; -0.0008 ± 0.0003 |
| Fine-tuned weight 0.67 instead of 0.5 | -0.00032 ± 0.00024 |
| LightGBM stackers (narrow, wide, residual), nested | -0.0020 ± 0.0006; +0.0004 ± 0.0007; -0.0008 ± 0.0004 |
| Listwise linear stacker; greedy member selection, nested | +0.00095 ± 0.00064; +0.0010 ± 0.0005 with 0.0014 optimism |
| Equal mix of three families, declared in advance | +0.00117 ± 0.00048 |

Over pairs of families with a strength gap below 0.01, the gain of an equal mix over its better part followed gain = a (1 - correlation) - b × gap:

| Sample | a | b | R² |
|---|---|---|---|
| validation | 0.019 | 0.38 | 0.95 |
| holdout | 0.061 | 0.62 | 0.96 |

## What it taught

The combiner stayed as it was: the member-equal z-mean, with the branches mixed 0.5 to 0.5. No fitted combiner beat an equal mix with no parameter. The final distilled files use the same combiner, and [tricks-sweeps](../tricks-sweeps/) re-checked it on 2026-11-15.

Pooling families trained with different recipes helped. Six cooldown continuations (F5cd) added to the six members of F5 gave +0.0008 on validation and +0.0011 on the holdout. The holdout score of the 12 (0.32272) passed the fitted limit of unlimited seeds of the F5 recipe (0.32232).

The first such pool, Cd, read +0.00157 ± 0.00034 on the holdout and +0.00083 ± 0.00054 on validation over the one-family file. It scored 0.39208 on public, against 0.38786 for the previous scored file.

The fitted law also explains why trees, the encoder score and older rankers never helped a pool. A weaker family pays only if its gap is below about 0.05 (1 - correlation) on validation.

The failure is in the first table. M2, the pool that was best on validation, read +0.00062 ± 0.00034 over G3 on validation and landed 0.00039 below it on public. S2, the stacker trained on validation, read +0.00056 ± 0.00028 and landed 0.00035 below. Near copies cannot be ranked on validation ([validation-and-leaks](../validation-and-leaks/)).

After that reading, pools were chosen by a rule that discounts validation gains the holdout does not confirm, and more members stopped helping. G3 plus 12 new from-scratch members scored +0.00007 on that rule, against +0.00027 for G3. G3 plus 4 new roots read +0.00003 on validation and -0.00003 on the holdout.

G3 stayed the pick of this generation. Its public score was passed only once members with lighter popularity inputs were added. C3, G3 plus 8 popularity-light members with a 3-seed fine-tune, scored 0.393454 ([popularity-views](../popularity-views/)).

Limits:

- The rule in `decide5.py` was written after the first public readings of these pools, and the records flag it as not blind.
- The seven scored pools lie within 0.00074 of each other on public, inside sampling noise, so the board does not confirm the choice of G3 over its neighbours.

## Files

`code/`

- `ens_common.py`: loaders, per-feed z-scores, slices.
- `exp1_combiners.py`: eight combiners and segment-specific fine-tune weights.
- `exp3_diversity.py`: correlations between families and NDCG against the number of members.
- `exp7_pairs.py`: the strength-against-diversity law.
- `CPU1_xrecipe_member_equal_pool_v3.py`: the first cross-recipe pools under a pre-declared rule.
- `build_pool.py`: writes a pool as a submission file.
- `decide4.py`, `decide5.py`: two of the dated decision rules.

`results/`

- `combiners.txt`, `stackers.txt`, `diversity.txt`, `strength_diversity_law.txt`: read-outs of the design study.
- `cross_recipe_pool_phaseA.txt`: the first cross-recipe pools.
- `decide3_1130.txt`, `decide4_interim_1350.txt`, `decide5_out.txt`: outputs of three decision runs.
- `gate_best_pool_G3.txt`: pre-submission gate output for G3.
- `pools_val_holdout_public.csv`, `pools_deltas_vs_G3.csv`: every pool of the generation on validation, the holdout and the public board (copies of `lab/nn_x/audit/valpub/landscape/out/s1_*.csv`).
- `public_scores_hard_label_files.csv`, `pools_vs_best_pool_G3.csv`: public scores and paired differences.

### Raw material and records

- Lab folders: `lab/solvers/ensemble/` and `lab/solvers/out/ensemble/` (design study, including the stacker script `exp2_stacker.py`), `lab/solvers/out/CPU1_xrecipe_member_equal_pool_v3/`, `lab/nn_x/picks/` (builders, decision scripts, gate outputs), `lab/nn_x/stack/` (the neural gate stacker).
- Reports: `lab/reports/sol_ensemble.md`, `lab/reports/sol_cpu_CPU1_xrecipe_member_equal_pool_v3.md` and `lab/reports/VAL_PUBLIC_1113.md`.
- Records: the status log (a working record, not published).
- In context: [the timeline](../../docs/05-timeline.md), [validation](../../docs/04-validation.md).
