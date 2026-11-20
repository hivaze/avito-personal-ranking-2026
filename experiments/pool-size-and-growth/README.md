# Pool size and growth

In brief:

- A pool of students of one recipe follows NDCG(n) = a - b / n. Eight full-view students sit within 0.00024 of the limit a on validation. This member-pool law, fitted per branch, forecasts +0.00007 from 16 to 32 students, below the paired public SE of 0.0003 to 0.0006.
- Other kinds of students, forward selection and learned weights added nothing beyond more seeds of one recipe.
- Pools of 24, 32 and 48 all-data full-view students read below FD16P20 on validation, and the two scored on public were level with it. Growth was closed on 2026-11-16 after about 120 to 140 GPU-hours.

## Why it was tried

The final files average the per-feed z-scores of distilled students. A student is a ranker trained toward half clicks and half the averaged scores of a teacher ensemble. A pool is the equal-weight mean of its members' within-feed z-scores ([glossary](../../docs/glossary.md)).

The questions were how much a pool gains from more seeds of the same recipe, from other kinds of students mixed in, or from learned weights. A second question was whether growing the all-data pool from 16 to 32 or 48 full-view students was worth its GPU time.

All-data students trained on every train feed, the holdout included. Full-view students read all 34 pre-computed inputs, and popularity-light students have the 8 popularity inputs replaced by one exposure count. FD16P20 is 16 all-data full-view students plus 4 popularity-light students at weight 0.2.

More members average out seed noise, and by 2026-11-15 a larger pool was the one change left with a positive forecast. Its size had to be weighed against the cost. One all-data full-view student took 220.8 to 258.8 minutes on a shared GPU, so 16 more students cost about 60 to 70 GPU-hours.

## What ran

The reads ran on CPU over stored predictions; only the new students needed GPUs.

### 2026-11-14: the law, unions and selection

W8 is the pool of the 8 held-out full-view students, which never trained on the 30,000 holdout feeds.

- All 255 subsets of W8, to fit the pool-size law NDCG(n) = a - b / n, where a is the level of an infinitely large pool.
- 32 unions of W8 with other students.
- Greedy forward selection fitted on one witness and read on the other. A witness is a sample a change is judged on, here validation or the holdout.

The ordering rule was written before any pool number (`results/pool_composition_ordering_rule.txt`).

### 2026-11-15: larger structures and learned weights

- A cache of 633 members.
- The law on 16 and 32 students.
- Two large structures: 68 members (32 full-view plus 36 popularity-light) and 60 all-data members.
- Ten learned combiners, estimated nested inside the fine-tune's five user folds, so that no weight is read on the feeds it was fitted on.
- The 495 possible draws of four popularity-light students out of 12.

### 2026-11-15 to 16: growing the all-data pool

The all-data full-view pool grew to 24, 32 and 48 students (seeds 959 to 974). A larger file could replace FD16P20 only if all three conditions held:

- every new student passed the structure checks
- the file passed the pre-submission gate, the mandatory check of every file
- its validation score was no more than one paired SE below FD16P20

## Results

Validation is 14,423 feeds, the holdout 30,000 random-order train feeds. A "±" is one paired SE, and values split by "/" follow the order of the Sample column. The train-only branch is the members' prediction before the fine-tune on validation, and narrow students have width 128.

The law, unions and selection:

| Pool | Sample | Result |
|---|---|---|
| law fitted to W8, asymptote a | validation / holdout | 0.40181 / 0.32472; W8 below it by 0.00024 ± 0.00012 / 0.00017 ± 0.00007 |
| 16 held-out full-view students (W16) against W8; the law forecast +0.00016 on validation | public | 0.39448337 against 0.394486 |
| W8 plus 8 narrow students against W8 | validation / holdout / public | +0.00008 ± 0.00034 / -0.00028 ± 0.00019 / 0.394078 against 0.394486 |
| forward selection fitted on validation, read on the holdout | holdout | -0.00095 ± 0.00028 |
| law on 16 students: 16 to 32 / 16 to infinity, train-only branch | validation | +0.00007 ± 0.00001 / +0.00015 ± 0.00003 |
| rank pooling against z-score pooling, 4 disjoint member sets | validation | -0.00007 pooled (z -0.5); not replicated |

Large structures, learned weights and growth, read against FD16P20 (validation 0.40271, public 0.394892) unless stated:

| Pool | Sample | Result |
|---|---|---|
| 68 members against FD16P20 | validation / public | -0.00024 ± 0.00028 / 0.3946427 against 0.394892 |
| 60 all-data members against FD16P20 | validation | -0.00056 ± 0.00027 |
| learned combiners, nested, against the 60-member structure | validation | -0.00083 ± 0.00042 to +0.00020 ± 0.00034 (in sample up to +0.00177) |
| 24 / 32 / 48 all-data full-view students against FD16P20 | validation | -0.00013 ± 0.00022 / -0.00034 ± 0.00025 / -0.00049 ± 0.00028 |
| 24 / 48 all-data full-view students, scored 2026-11-17 | public | 0.394905 / 0.39484072 against 0.394892 for FD16P20 |
| students of seeds 959-966 / 967-974, each as a 16-student pool, against the first 16, train-only branch | validation | -0.00086 ± 0.00039 / -0.00027 ± 0.00038 |

All-data students trained on the holdout feeds and have no holdout reading. FD32P20, the 32-student file of one recipe, was not scored on the public board. FDXU32P20:ps, 32 full-view students of two recipes (the 16 of FD16P20 plus 16 pseudo-feed students, [distillation](../distillation/)), read 0.40301 on validation and scored 0.39524613 on public on 2026-11-19, 0.00035 above FD16P20, inside the paired noise.

A second final built from members disjoint from FD16P20 was tried on 2026-11-16. FDB16P20 holds the 16 all-data full-view students of seeds 959 to 966 and the 4 all-data popularity-light students of seeds 955 to 958. It read -0.00052 ± 0.00039 against FD16P20 on validation and missed the one-SE bar for pick 2. Before it was built, two disjoint 8-student all-data files had shown test correlation 0.99726, against 0.99753 between W8P20 and FD16P20 (`pick2_decorr.py`, [tricks-sweeps](../tricks-sweeps/)).

## What it taught

The law fits. It holds to 1e-4 over all 255 subsets, and 8 full-view students sit within 0.00024 of the asymptote on validation.

Pool size cannot be read on the board. From 16 students the member-pool law gives +0.00007 for 32, while the paired public SE between such files is 0.0003 to 0.0006. A second law fitted on whole files (final validation = 0.40254 - 0.00177 / n_full - 0.00014 / n_light, `lab/reports/TRICKS_1115/SYNTHESIS.md`, section 2.4) gives +0.000055; [docs/07](../../docs/07-what-failed.md) quotes that one. The 68-member file showed this, and the 24- and 48-student files confirmed it (+0.000013 and -0.00005 against FD16P20 on public).

Under distillation, the seed was the only source of diversity that showed. Students with another teacher, generation or width correlate with the reference as closely as another seed does (mean correlation 0.9718 against 0.9722 between seeds 951 to 954 of the reference). Of 24 other student arms whose pool level is within 0.001 of the full-view students, none added more than extra seeds.

Removing weight decay did decorrelate students (mean correlation 0.9563 against 0.9723 over the reference's 8 seeds). It cost more per member than the pool regained: an 8-student pool without weight decay read -0.00134 ± 0.00056 on validation against the reference 8-student pool.

Forward selection and learned weights did not transfer. Forward selection fitted on validation lost 0.00095 on the holdout, and combiners that read up to +0.00177 in sample were worth nothing when estimated nested.

The growth failed its own rule for a reason seen only afterwards: FD16P20 was a favourable draw, a pool that sits above the mean of same-size pools of its recipe by seed luck ([failure 18](../../docs/07-what-failed.md#18-growing-the-ensemble-past-16-members) in the docs).

- Its four popularity-light students put it above 472 of the 495 possible draws. That excess does not carry to new feeds: its split-half correlation was -0.156 on validation and -0.005 on the private-mix view (validation re-weighted to match the private part of test).
- Its first 16 full-view students were lucky too. The next 16 of the same recipe read -0.00086 against them.

Pick 1 (FD16P20@m0.4_0.2, public 0.3952092) raises the popularity-light weight to 0.4 on feeds of fewer than 50 candidates. That weight was read on this draw. Under the same rule, 8 or 12 popularity-light students read 0.00057 and 0.00039 below the 4 on validation.

The unbiased level of the family is about 0.4023 to 0.4024 on validation, against FD16P20's 0.40271. A non-inferiority bar set against FD16P20, such as the one-SE condition above, therefore fails by noise more often than its nominal rate.

Growth was closed at 48 on 2026-11-16, after about 120 to 140 GPU-hours on 32 extra students (derived from the time per student above).

One ledger block (2026-11-16 03:26Z) proposed the 48-student file as a second final for its low draw variance. The newer block of 05:59Z did not promote it, because validation read it 0.00049 below FD16P20.

## Files

`code/`:

- `pc_common.py`, `ens_common.py`: loaders, pooling, fast NDCG.
- `pc_saturation.py`: the pool-size law on all subsets of W8.
- `pc_unions.py`: unions of student groups against W8.
- `pc_forward.py`: cross-witness forward selection.
- `e2_law.py`: the law on 16 and 32 students.
- `e3_nested.py`: learned combiners, estimated nested.
- `e6_draws.py`: transfer of a file's draw excess.

`results/`:

- `summary.csv`: these numbers and more, each with its source.
- `pool_composition_*`: the rule, saturation, unions, forward selection, final files.
- `ens_e*`: law, combiners, draws, rank pooling, the 68- and 60-member builds.
- `gate_32_wide_students.txt`, `gate_48_wide_students.txt`: gate outputs of the grown files.

### Raw material and records

- Working folders: `lab/nn_x/w12/recheck/pool_composition/` and `lab/nn_x/w12/ens_1115/`.
- The growth watcher `lab/nn_x/w12/kd/fd/watch_growth2.sh` and the file builder `lab/nn_x/w12/kd/build_r4.py`.
- Reports: `lab/reports/RECHECK_1114/pool_composition.md`, `lab/reports/TRICKS_1115/int_mega.md`, and sections 0 and 2.4 of `lab/reports/TRICKS_1115/SYNTHESIS.md`.
- Records: the decision ledger (a working record, not published).
- The all-data retrains themselves are in [full-data-retrains](../full-data-retrains/).
