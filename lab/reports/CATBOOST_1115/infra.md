# CatBoost round 2, KEY = infra: data and baseline infrastructure

2026-11-15, track `nn_x/w12/cb` (rules: `nn_x/w12/cb/PREDECLARED.txt`). CPU only, 12 threads per process.
Marks: [M] measured here, [D] read in a project file (path given), [I] inferred. No web access was used.

## 1. Verdict

DRAFT (runs still training)



## 2. Tables

### 2.1 Reproduction of the CatBoost of record and the train-side variant

| model | selection | fit feeds | trees used (built) | val NDCG ± SE (n = 14,423) | holdout NDCG ± SE (n = 30,000) | val half A | val half B | rho val / hold | law record / refit | member rule 0.9 z(W16) + 0.1 z(X): val, holdout, pooled z |
|---|---|---:|---:|---|---|---:|---:|---|---|---|
| record: GPU YetiRank `e3_xm_yeti`, lr 0.08 (predictions on record, re-scored here) | val half A (record) | - | - | 0.38201 ± 0.00199 | 0.30424 ± 0.00108 | - | - | 0.837 / 0.840 | -0.0043 / -0.0042 | -0.00019 ± 0.00040, +0.00005 ± 0.00022, -0.16 FAIL |
| CPU `repro_valA`: record configuration, lr 0.08 | val half A (record rule) | not finished | | | | | | | | |
| CPU `ts_lr020_s0`: lr 0.2, od_wait 100, otherwise the record configuration | train-side split (no val label) | not finished | | | | | | | | |

Reading notes:
- "val half A" / "val half B" are the record's val user halves (GroupKFold(2) by user, user-0 feeds as singletons; my half
  B equals `nn_x/cb26/data/e3_xm_yeti_halfB_feeds.npy`, asserted in the script) [M].
- Every row is one seed. Single-seed tree arms differ by 0.001 to 0.002 from the seed alone ([D]
  `nn_x/w25/out/results.txt`); no seed repeat was run here.
- rho, the law and the member rule are computed against W16 (val 0.40144, holdout 0.32460 [M]); conventions in 3.3.

### 2.2 CPU curves next to the GPU record (selection-set NDCG by number of trees)

| trees | GPU record, val half A | `repro_valA`: selection-set NDCG (minutes of fit) | `ts_lr020_s0`: selection-set NDCG (minutes of fit) |
|---:|---:|---:|---:|
| 1 | 0.33386 | 0.32788 (0) | 0.24355 (0) |
| 26 |  | 0.37135 (4) | 0.29703 (4) |
| 51 |  | 0.37565 (8) | 0.30079 (8) |
| 101 | 0.37286 | 0.37825 (15) | 0.30304 (15) |
| 151 |  | 0.38019 (23) | 0.30441 (22) |
| 201 | 0.37597 | 0.38130 (30) | 0.30502 (29) |
| 250 |  | 0.38153 (38) | 0.30550 (36) |
| 251 |  | 0.38139 (38) |  |
| 301 | 0.37782 | 0.38231 (46) |  |
| 351 |  | 0.38241 (54) |  |
| 400 |  | 0.38326 (62) |  |
| 401 | 0.37899 |  |  |
| 501 | 0.38000 |  |  |
| 601 | 0.38032 |  |  |
| 701 | 0.38103 |  |  |
| 801 | 0.38135 |  |  |
| 901 | 0.38197 |  |  |
| 1001 | 0.38213 |  |  |
| 1101 | 0.38234 |  |  |
| 1201 | 0.38286 |  |  |
| 1301 | 0.38343 |  |  |
| 1401 | 0.38377 |  |  |
| 1501 | 0.38393 |  |  |
| 1601 | 0.38427 |  |  |
| 1701 | 0.38447 |  |  |
| 1801 | 0.38459 |  |  |
| 1901 | 0.38520 |  |  |

The GPU column is the record's own log (`nn_x/cb26/logs/e3_xm_yeti.log` [D]): val half A, 7,212 feeds. The `repro_valA`
column is the same metric on the same feeds from the CPU run, so those two columns are directly comparable [M]. The
`ts_lr020_s0` column is NDCG on its train-side selection split (13,527 exploration train feeds with M >= 50), a different
and harder set: read its shape, not its level.

How many CPU trees reach the level of the GPU record at a given GPU tree count (same feeds, same metric) [M]:

| GPU trees | GPU half-A NDCG | CPU trees to reach the same half-A NDCG | CPU minutes |
|---:|---:|---:|---:|
| 101 | 0.37286 | 34 | 5 |
| 301 | 0.37782 | 98 | 15 |
| 501 | 0.38000 | 150 | 23 |
| 1001 | 0.38213 | 288 | 44 |
| 1501 | 0.38393 | not reached in 400 trees (best 0.38326) |  |



### 2.3 Timing on this machine (12 threads, CatBoost 1.2.10 CPU, 134,118 feeds = 19,741,607 rows x 177 features)

All rows: learning rate 0.08, l2_leaf_reg 5, border_count 254, category_id native categorical, pool quantized once (12 s,
pinned to 12 cores) and reused, 10 iterations per row (7 at depth 10), median seconds per iteration after dropping the
first two [M]. Measured while one other 12-thread job of mine (the reproduction) was running and with a machine load of
70 to 110 on a shared machine, so absolute numbers carry roughly +-10% [I]. The reproduction run itself, which also evaluates
NDCG and PFound on 691,143 selection rows at every iteration, ran at 9.3 s per iteration [M].

| loss | depth 6: s / iteration (M rows / s) | depth 8: s / iteration (M rows / s) | depth 10: s / iteration (M rows / s) |
|---|---:|---:|---:|
| YetiRank | 7.82 (2.52) | 8.31 (2.37) | 12.86 (1.53) |
| QuerySoftMax | 13.67 (1.44) | 14.37 (1.37) | 19.13 (1.03) |
| RMSE | 2.12 (9.32) | 2.34 (8.42) | 8.01 (2.46) |

Speed levers at depth 8 [M]:

| configuration | s / iteration | relative to the default of the same loss |
|---|---:|---:|
| YetiRank default | 8.31 | 1.00x |
| YetiRank, rsm 0.3 | 7.33 | 0.88x |
| YetiRank, rsm 0.1 | 6.42 | 0.77x |
| YetiRank, Bernoulli subsample 0.5 (objects) | 7.98 | 0.96x |
| YetiRank, Bernoulli subsample 0.5, sampling_unit Group | 7.39 | 0.89x |
| YetiRank:permutations=3 | 6.50 | 0.78x |
| YetiRank, rsm 0.3 + Bernoulli 0.5 | 7.25 | 0.87x |
| RMSE default | 2.34 | 1.00x |
| RMSE, rsm 0.3 | 1.59 | 0.68x |
| RMSE, MVS subsample 0.5 | 2.15 | 0.92x |

Memory [M]: the float32 matrix is a 14.0 GB file-backed memmap; after `Pool` construction RSS is 13.8 GB, after
quantization 18.5 GB, during a fit 25 to 30 GB, peak 31.8 GB. The reproduction script (unquantized training pool plus a
selection pool) sits at 36 to 37 GB.

What fits into 2 hours per arm on CPU (7,000 s net of pool building, prediction and evaluation). The 134K column is
measured [M]; the other columns scale it linearly in rows [I]:

| loss, depth 8 | 134,118 feeds (19.7M rows): iterations in 2 h | xm + xs = 485,023 feeds (25.1M rows) | xm + prod200k (36.4M rows) | 1M production feeds (83M rows): s / iteration, iterations in 2 h, RSS |
|---|---:|---:|---:|---|
| YetiRank | 841 | 662 | 457 | 35 s, 200 iterations, about 134 GB |
| QuerySoftMax | 487 | 383 | 264 | 60 s, 115 iterations, about 134 GB |
| RMSE | 2985 | 2347 | 1620 | 10 s, 710 iterations, about 134 GB |

Reference on record [D] `nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118.json`: LightGBM L2 on the same 19.7M rows, 255
leaves, feature and row fraction 0.5, 10 threads: 600 rounds in 13.7 minutes including data loading and six evaluations,
about 1.2 s per round.

### 2.4 Data sets and what is in them

| set | feeds | rows | mean M | definition | status |
|---|---:|---:|---:|---|---|
| `xm` | 134,118 | 19,741,607 | 147.2 | nn/train.py p_random > 0.5 and M >= 50, minus the holdout | the training set of record [M] |
| `xs` | 350,905 | 5,361,945 | 15.3 | raw p_random > 0.5 and M < 50 | "exploration-LIKE" natural short feeds; p_random is documented as unreliable for M < 50 [D] `reports/eda_labels_position.md:211` |
| `prod` (pool) | 2,236,185 | 185,699,480 | 83.0 | raw p_random <= 0.5, not NaN | production feeds; `prod200k` = 200,000 feeds, 16,622,487 rows (uniform, seed 20261115) [M] |
| unknown regime | 400,000 | not counted | not counted | raw p_random is NaN (feeds used to fit the proxy [D] same file) | in no cbx training set |
| `hold` | 30,000 | 4,630,966 | 154.4 | rng(12345) draw from p_random > 0.9 and M >= 50 | evaluation only; identical to `cbdata.hold_index()` [M] |
| `val` | 14,423 | 1,389,971 | 96.4 | flat order | row order of `preds/` |
| `test` | 28,595 | 2,904,806 | 101.6 | flat order | no labels; row order of `preds/` |

Total train: 3,151,208 feeds. Exploration with M >= 50 and p_random > 0.5: 164,118 = 134,118 + the 30,000 holdout [M].
On record [D]: adding the short feeds hurt a CatBoost (W25 A1 - A0, 485,023 against 134,118 feeds: val -0.00125 +- 0.00074,
holdout -0.00187 +- 0.00044, `nn_x/w25/out/results.txt`) and the NN cooldown pool (-0.00053 +- 0.00016,
`reports/sol_solvers.md` TL;DR item 5).
The holdout is a feed-level split. Known users who also have holdout feeds: `xm` 792, `xs` 1,269, `prod200k` 1,011 [M].

Teacher targets (aligned to `data/flat/train` rows; 246,745,836 float32 values per file) [M unless marked]:

| file | what it is | alignment check | scale |
|---|---|---|---|
| `nn_x/w12/kd/out/teacher_mean11.npy` | (sum of `teacher_a..f.npy`) / 11, then centred per feed. The 11 models are `r6_cd_m02_r101-107` and `w5_e10_base_s951-954` ([D] `nn_x/w12/kd/out/dump_{a..f}.log`): hard-label, train-only, none saw the holdout | equals the recomputed centred mean to 3e-5 on 3M rows; on the holdout rows it equals the mean of the 11 members' holdout prediction files (correlation 1.000000, max abs difference below 5e-5) | feed mean 0 (max abs 8e-8); within-feed sd 1.14 on the first 3M rows, 1.21 on `xm`, 1.22 on `hold`, 1.01 on `xs`, 1.15 on `prod200k` |
| `nn_x/w12/kd/out/teacher_oof.npy` | per user fold k = `cbx.xfold_of(user_id)` (the formula of `nn_x/w12/kd/assemble_oof.py`): mean RAW logit of the 3 models `w5_e10_xf<k>_s951-953` that excluded fold k | equals `teacher_xf<k>.npy / 3` on 1,000 rows of every train set | NOT centred: per-feed offset +114 (sd 17); within-feed sd 1.12 (1.18 on `xm`); within-feed correlation to mean11 0.987 |

The in-sample teacher scores 0.3465 on training feeds and 0.3237 on the unseen holdout ([D] `reports/CURRENT_STATUS.md:715`);
my evaluator gives 0.32370 on the holdout and 0.40072 on val for the mean of the same 11 members, and 0.32100 on the
holdout for the cross-fitted teacher [M]. The soft label of the NN students is `softmax(teacher_mean11 / 1.0)` inside the
feed ([D] `nn_x/w5/train.py:500`) = `ds.teacher("mean11", "soft")`.

Raw embeddings [M unless marked]:

| object | where | shape and dtype | missing-value convention |
|---|---|---|---|
| item table | `data/competition/<split>/item_embeddings/data.memmap`, one table per split | train (81,557,944, 128), val (2,516,711, 128), test (4,710,948, 128), int8, row norm about 256 | index -1 in `cand_emb_idx` / `hist_emb_idx` = no embedding ([D] `data/competition/readme.md`). Feature `emb_missing` = (`cand_emb_idx` < 0): 0.004% of the first 5M train rows, 0.015% of val rows. 16.8% of history events have no embedding |
| user embedding | `data/flat/<split>/user_emb.npy` | (n_feeds, 128) int8, one per feed, norm about 250 | an all-zero vector in 2.3% of the first 200K train feeds [I: no user embedding] |
| `ds.emb_cand_i8` | cache | (n_rows, 128) int8 memmap, 2.5 GB for `xm` | zero row when the candidate has no embedding |
| `ds.emb_feed("user" / "hmean" / "h5" / "h20")` | cache | (n_feeds, 128) float32 | zero vector when there is nothing to average; `ds.emb_feed_n(kind)` = number of events averaged |
| `ds.emb_rows(kind, dtype)` | RAM | (n_rows, 128); on `xm` 5.05 GB float16 or 10.1 GB float32 per kind, 25 / 51 GB for all five kinds | as above |

History is stored newest first (`hist_hpos` = 0..H-1, checked). `h5` / `h20` average the 5 / 20 most recent events that
HAVE an embedding, `hmean` all of them. Values are raw int8 units, not normalised. float16 is exact for `cand` and `user`.

### 2.5 The evaluator on the tree models of record (their saved predictions, re-scored against W16)

| model | selection | fit feeds | trees used (built) | val NDCG ± SE (n = 14,423) | holdout NDCG ± SE (n = 30,000) | val half A | val half B | rho val / hold | law record / refit | member rule 0.9 z(W16) + 0.1 z(X): val, holdout, pooled z |
|---|---|---:|---:|---|---|---:|---:|---|---|---|
| GPU CatBoost YetiRank of record (hard label) | val half A | - | - | 0.38201 ± 0.00199 | 0.30424 ± 0.00108 | - | - | 0.837 / 0.840 | -0.0043 / -0.0042 | -0.00019 ± 0.00040, +0.00005 ± 0.00022, -0.16 FAIL |
| LightGBM lambdarank, hard label, round 600 (long_bets) | fixed rounds | - | - | 0.37677 ± 0.00194 | 0.30092 ± 0.00105 | - | - | 0.852 / 0.859 | -0.0066 / -0.0060 | +0.00007 ± 0.00037, +0.00014 ± 0.00020, +0.64 FAIL |
| LightGBM L2 on teacher_mean11, 134K feeds, round 600 (long_bets) | fixed rounds | - | - | 0.38484 ± 0.00199 | 0.30761 ± 0.00108 | - | - | 0.895 / 0.902 | -0.0043 / -0.0040 | -0.00035 ± 0.00037, -0.00021 ± 0.00020, -1.40 FAIL |
| LightGBM L2 on teacher_oof, 60K feeds, round 600 (long_bets) | fixed rounds | - | - | 0.38444 ± 0.00198 | 0.30768 ± 0.00108 | - | - | 0.894 / 0.899 | -0.0044 / -0.0041 | -0.00012 ± 0.00037, -0.00019 ± 0.00020, -0.92 FAIL |
| bundle: z-mean of the GPU CatBoost and the LightGBM KD student | - | - | - | 0.38470 ± 0.00199 | 0.30754 ± 0.00109 | - | - | 0.879 / 0.885 | -0.0041 / -0.0038 | -0.00031 ± 0.00038, -0.00010 ± 0.00020, -0.92 FAIL |

Paired per feed [M]: LightGBM KD student minus the GPU CatBoost of record: val +0.00282 +- 0.00094 (n = 14,423), holdout
+0.00337 +- 0.00055 (n = 30,000). The SE of the member-rule difference is about 0.00038 on val and 0.00020 on the
holdout, so a pooled z of 2 needs roughly +0.0005 on val together with +0.0003 on the holdout at weight 0.1 [I].

### 2.6 Smoke test (`smoke.py --big`, last run on the final module)

85 PASS, 0 FAIL (ALL PASS  (145s)); log `nn_x/w12/cb/infra/logs/smoke_big.log` [M].

| check | what | result |
|---|---|---|
| 6 | training path opens no val / test / holdout label file; `ds.y` raises for val, hold, test | PASS (4) |
| 1 | 177 features == cb26 `member_features()` == record JSON | PASS (1) |
| 2 | shapes of X / gid / item_id / cat for xm, hold, val, test, xs, prod200k | PASS (6) |
| 3a | f1, f5, category_id, M, H, emb_missing, item_id, fidx recomputed from raw for 1,000 random rows per set | PASS (6) |
| 3b | the same 1,000 rows x 177 columns equal `cbdata.load_tab` of cb26 | PASS (6) |
| 3c | dot_u, cos_u, item_norm recomputed from the cbx raw embeddings (max abs error 0 / 6e-8 / 0) | PASS (6) |
| 3d | hmean / h5 / h20 and their counts recomputed in numpy for 100 feeds per set | PASS (6) |
| 3e | history order: hist_hpos == 0..H-1 | PASS (6) |
| 3f | user embedding rows == user_emb.npy | PASS (6) |
| 4a | teacher value of a row == the stored file (1,000 rows per train set) | PASS (4) |
| 4b | teacher_oof == teacher_xf<k> / 3 with k = cbx.xfold_of(user) | PASS (4) |
| 4c | soft labels sum to 1 per feed; centred forms have zero feed mean | PASS (4) |
| 5 | no holdout feed in xm, xs, prod200k, concat(xm, xs); holdout == cb26 `hold_index()`; fit / selection share no known user | PASS (6) |
| 7a | NDCG kernel is a verbatim copy of `pc_common._ndcg` | PASS (1) |
| 7b | fast NDCG == `src/metrics.ndcg_per_feed` on val (W16 pool, heavy ties, constant, random): max abs diff 0 | PASS (4) |
| 7c | the same on the holdout | PASS (1) |
| 7d | feed z == `stack_val.feed_rank_z[:, 1]` | PASS (1) |
| 7e | W16 pool = 0.40144 / 0.32460 | PASS (1) |
| 9a | evaluator reproduces the record's 0.38201 / 0.30424 from its saved predictions (row-order check for val and holdout) | PASS (1) |
| 11a | tiny YetiRank fit on `make_pool`, prediction on val | PASS (1) |
| 11b | embedding-feature pools train with YetiRank, RMSE (teacher logit), QuerySoftMax (soft teacher) | PASS (3) |
| 11c | baseline pool (YetiRank on top of the centred OOF teacher logit) | PASS (1) |
| 11d | item_id / user_id as native categorical features + OOF NN logit as a column | PASS (1) |
| 12a | holdout: mean of the 11 members' prediction files == teacher_mean11 on the holdout rows | PASS (1) |
| 12b | NN inputs on val / holdout score as expected | PASS (2) |
| 12c | NN input on test has the right shape | PASS (1) |
| 12d | OOF teacher agrees with the xf15 input on the holdout | PASS (1) |

`smoke_arm.py` (the `cbx.run_arm` protocol on 3,000 feeds: an RMSE-on-teacher arm selected by NDCG on clicks, a YetiRank
arm, and `Evaluator.compare`) also ran clean [M], log `nn_x/w12/cb/infra/logs/smoke_arm.log`.

## 3. Method

### 3.1 Loader (`nn_x/w12/cb/infra/cbx.py`)
- One handle per set (`cbx.load(name)`). Per-row arrays are in the flat file order of the split restricted to the set's
  feeds (feeds ascending), so val / test predictions are already in the row order of `preds/` and holdout predictions in
  the row order of `nn_x/w5/preds/holdout` (asserted against `holdout_meta.npy` and `holdout_y.npy` in every `Evaluator()`).
- The 177-feature matrix is gathered from `data/feats/<split>` (175 features) and `data/feats_new/<split>` (`pre2_fe_s`,
  `pre2_new_s`), never from `data/feats_quarantine`. The name list is the record's JSON and equals
  `nn_x/cb26/cb_member.member_features()` today (smoke check 1).
- Labels of `val` and `hold` raise `PermissionError` outside the evaluator; the training path was run under an `np.load`
  spy and opened no evaluation label file (smoke check 6).
- Train-side selection: "selection users" = teacher fold 0, sub-half 0 by user hash (user-0 feeds by feed index).
  `cbx.trainside_split(ds)` returns fit = feeds of all other users and sel = the `xm` feeds of the selection users:
  13,527 selection feeds (1,983,163 rows) and 120,591 fit feeds on `xm`, no known user on both sides. The selection set is
  the same for every training set (`concat(xm, xs)`: 436,320 fit feeds, 35,176 short feeds of selection users left
  unused; `prod200k`: 180,124 fit feeds). The folds are those of the cross-fitted teacher, so `teacher_oof` on selection
  feeds comes from models that never saw those users.
- `cbx.run_arm(...)` is the standard arm: fit on the fit split, early stopping on the selection split with NDCG on
  clicks whatever the training target is, prediction at the selected iteration, evaluation, saving.
- `cbx.nn_input(ds, which)` gives NN scores that are legal as INPUTS: the cross-fitted teacher on train rows, and the
  fold models' (or the 11 members') train-only predictions on val / holdout / test.

### 3.2 Baseline runs (`nn_x/w12/cb/infra/cb_member_cpu.py`)
A copy of `nn_x/cb26/cb_member.py` with the GPU arguments removed (`task_type`, `devices`, `gpu_ram_part`),
`thread_count=12`, data through `cbx`, a wall-clock / STOP-file callback and the evaluation through `cbx.Evaluator`.
The model definition is unchanged: YetiRank, depth 8, l2_leaf_reg 5, border_count 254, seed 0,
`eval_metric="NDCG:type=Base"`, od Iter. `--sel valA` is the record's rule (early stopping on val half A: val labels
select the iteration count). `--sel trainside` reads no val label before the final evaluation. `use_best_model` is off
and prediction uses `ntree_end = best iteration + 1`, which gives the same predictions and keeps later trees for the
staged curves.

### 3.3 Evaluator conventions (fixed for the whole track)
- z(.) = per-feed z-score, identical to `src/stack_val.feed_rank_z[:, 1]` (checked).
- W16 = member-equal mean of the 16 members' z (train-only predictions), standardised again per feed.
- rho = mean over feeds of the within-feed Pearson correlation of z(X) and z(W16). The project's other convention
  ("wcorr": one Pearson over all rows of the z arrays, used by the long-bets read-out and by
  `reports/CATBOOST_1115/audit.md`) reads about 0.003 higher (record CatBoost: 0.837 / 0.840 here, 0.840 / 0.841 there
  [M, D]); the evaluator prints both, and the law uses the first.
- law value = 0.0186 (1 - rho_val) - 0.378 (NDCG_val(W16) - NDCG_val(X)); the refit pair 0.0103 / 0.304 is printed next
  to it.
- member rule = NDCG(0.9 z(W16) + 0.1 z(X)) - NDCG(z(W16)) per feed on val and on the holdout, SE = sd / sqrt(feeds),
  pooled z = (z_val + z_hold) / sqrt(2); PASS iff both differences >= 0 and pooled z >= 2. Weights 0.05 and 0.2 are
  printed, never chosen.
- NDCG = a verbatim copy of `pc_common._ndcg` (body compared as text), equal to `src/metrics.ndcg_per_feed` with a
  maximum absolute difference of 0 on val and holdout for real, tied, constant and random scores.

## 4. Hazards found while building this (for every arm agent)

1. `Pool.quantize()` has no thread argument and used many cores during a 0.9 s call here [M]. Use
   `cbx.quantize(pool, border_count=254)` (pins the call to 12 idle cores; 12 s for the 19.7M-row pool) or let `fit`
   quantize, which honours `thread_count`.
2. On CPU, YetiRank costs 3.5 times and QuerySoftMax 6 times an RMSE iteration, at depth 6 and 8 alike, and feature or
   row subsampling barely helps (rsm 0.1: 6.4 s against 8.3 s). The time goes into the loss, not into the trees [M].
   Hard-label ranking arms are the expensive ones; teacher-regression arms are cheap.
3. CPU YetiRank is not the GPU YetiRank of the record: at equal iteration counts the CPU curve is far ahead (table 2.2)
   [M]. Iteration counts read from GPU logs do not transfer to CPU.
4. CatBoost's progress lines are block-buffered when stdout is a file: start with `python -u`. In a callback
   `info.iteration` is 1-based [M].
5. `get_feature_importance()` without `type=` fails for ranking losses unless the training pool is passed; use
   `type="PredictionValuesChange"` [M].
6. Native embedding features (`embedding_features_data`) do train on CPU with YetiRank, RMSE and QuerySoftMax, but the
   default processing (`["LDA", "KNN"]` according to the fitted model's parameters) costs 30 s for 44,480 rows and 162 s
   for 179,354 rows for ONE 128-d embedding at 4 threads, against 0.4 s and 1.5 s without it, and the cost is paid before
   the first tree [M]. At the measured growth (4.0 times the rows, 5.4 times the time) the 17.8M fit rows would take tens
   of hours [I]. `CatBoostRanker(...)` rejects the keywords `embedding_processing` and `embedding_calcers` [M]; the
   supported route is the generic `CatBoost(params)` constructor with `embedding_calcers=[...]`, and KNN is the
   expensive calcer while LDA is cheap ([D] `reports/CATBOOST_1115/docs_feat.md` section 3.1 and its cost table).
   `cbx.run_arm` accepts that generic estimator. An embedding arm on the full set needs LDA only, a row subsample, or
   vectors reduced beforehand and passed as numeric columns.
7. `teacher_oof.npy` is not centred (offset +114) while `teacher_mean11.npy` is: use `form="centred"` for regression
   targets and baselines.
8. The 177-feature set of record contains 12 `tte_*` target-statistic features and 5 `pre2_*` pretrained-encoder scores.
   `tte_*` is time-aware (train rows: statistics from earlier train days only; val / test rows: all of train, [D]
   `src/features_tte.py:1-4`), so the holdout feeds' labels are inside the statistics of later rows. A holdout feed never
   sees its own labels, but the files were not rebuilt without the holdout. This is inherited from cb26 and was not
   audited here.

## 5. Limits: what was NOT checked

- DRAFT

## 6. Exact commands

Environment prefix of every command (`ENV`):
`cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12`
(the light checks used 4 instead of 12 plus `CBX_THREADS=4`).

```bash
# caches (27 s + 8 s + 36 s + 2 s)
.venv/bin/python -u nn_x/w12/cb/infra/build_cache.py core aux big nn
# record configuration on CPU, the record's selection rule (val half A)
setsid nohup .venv/bin/python nn_x/w12/cb/infra/cb_member_cpu.py --tag repro_valA --sel valA \
  > nn_x/w12/cb/infra/logs/repro_valA.log 2>&1 < /dev/null &
# train-side selection variant (no val label), CPU-feasible learning rate
setsid nohup .venv/bin/python -u nn_x/w12/cb/infra/cb_member_cpu.py --tag ts_lr020_s0 --sel trainside --lr 0.2 \
  --od_wait 100 --iters 1500 --budget_min 68 --curve_every 25 --seed 0 > nn_x/w12/cb/infra/logs/ts_lr020_s0.log 2>&1 < /dev/null &
# clean early stop of a run (model, predictions and evaluation are still written)
touch nn_x/w12/cb/infra/out/<tag>.STOP
# timing table (stopped by its exact PID before the YetiRankPairwise row)
setsid nohup .venv/bin/python -u nn_x/w12/cb/infra/cb_timing.py --iters 10 > nn_x/w12/cb/infra/logs/timing.log 2>&1 < /dev/null &
# smoke tests and probes (4 threads)
.venv/bin/python -u nn_x/w12/cb/infra/smoke.py --big > nn_x/w12/cb/infra/logs/smoke_big.log 2>&1
.venv/bin/python -u nn_x/w12/cb/infra/smoke_arm.py > nn_x/w12/cb/infra/logs/smoke_arm.log 2>&1
.venv/bin/python -u nn_x/w12/cb/infra/emb_cost.py > nn_x/w12/cb/infra/logs/emb_cost.log 2>&1
# this report
.venv/bin/python nn_x/w12/cb/infra/mk_report.py nn_x/w12/cb/infra/verdict.txt nn_x/w12/cb/infra/notes.txt
```

## 7. Usage block for the arm agents (full text: `nn_x/w12/cb/infra/README.md`)

```python
# start: cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
#        NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 && setsid nohup .venv/bin/python -u nn_x/w12/cb/<KEY>/arm.py > nn_x/w12/cb/<KEY>/logs/arm.log 2>&1 < /dev/null &
import sys; sys.path.insert(0, "<repo>/nn_x/w12/cb/infra"); import cbx
from catboost import CatBoostRanker
tr = cbx.load("xm")                                    # 134,118 exploration feeds; also "xs", "prod200k", cbx.concat(a, b)
tr.X; tr.names; tr.y; tr.gid; tr.M; tr.uid; tr.item_id; tr.cat       # 177 features (memmap), labels, groups, sizes, ids
tr.teacher("mean11", "centred"); tr.teacher("mean11", "soft"); tr.teacher("oof", "centred")   # targets (mean11) / OOF (oof)
tr.emb_rows("cand"); tr.emb_feed("h5")                 # raw embeddings: cand | user | hmean | h5 | h20
m = CatBoostRanker(loss_function="YetiRank", depth=8, learning_rate=0.2, l2_leaf_reg=5, border_count=254, random_seed=0,
                   iterations=1500, od_type="Iter", od_wait=100, eval_metric="NDCG:type=Base", allow_writing_files=False,
                   thread_count=cbx.THREADS)
r = cbx.run_arm("my_arm_s0", m, tr, tr.y, "<repo>/nn_x/w12/cb/<KEY>/preds", budget_min=100,
                pool_kw=dict(cat="native"))            # add max_feeds=3000 for a 20-second dry run first
# r = evaluation dict; files <tag>_{val,hold,test}.npy (float32, row order of preds/ and nn_x/w5/preds/holdout), .eval.json, .curve.json, .cbm
ev = cbx.Evaluator(); ev.compare(pv_arm, ph_arm, pv_base, ph_base, label="arm - baseline")     # progress rule, paired per feed
```

## 8. Files created

All under `nn_x/w12/cb/infra/` unless a full path is given.

| file | content |
|---|---|
| `cbx.py` | loader, pools, train-side split, `run_arm`, `nn_input`, `Evaluator`, thread-safe `quantize` |
| `README.md` | usage lines for the arm agents |
| `build_cache.py`, `cache/` (42 GB) | cache builder and caches: per set `X177.npy`, `y.npy`, `item_id.npy`, `cat.npy`, `emb_idx.npy`, `pos.npy`, `teacher_{mean11,oof}.npy`, `emb_cand_i8.npy`, `emb_{user,hmean,h5,h20}.npy`, `emb_hn_*.npy`, `nn_*.npy`; `w16_z_{val,hold,test}.npy`; `features_177.json` |
| `cb_member_cpu.py` | CPU copy of the CatBoost of record with the two selection modes |
| `cb_timing.py`, `out/timing.jsonl`, `logs/timing.log` | timing table |
| `smoke.py`, `smoke_arm.py`, `logs/smoke_big.log`, `logs/smoke_core.log`, `logs/smoke_arm.log` | smoke tests |
| `emb_cost.py`, `emb_cost2.py`, `logs/emb_cost*.log` | cost of the default embedding calcers |
| `curve.py`, `tables.py`, `mk_report.py`, `report_template.txt`, `verdict.txt`, `notes.txt` | report tooling |
| `out/repro_valA*`, `out/ts_lr020_s0*` | baseline runs: `_val.npy`, `_hold.npy`, `_test.npy` (float32), `.eval.json`, `.curve.json`, `.cbm`; logs in `logs/` |
| `out/record_*.eval.json` | the tree models of record re-scored with the evaluator |
| `<repo>/reports/CATBOOST_1115/infra.md` | this report |
