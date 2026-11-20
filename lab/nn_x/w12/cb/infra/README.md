# cbx: data + evaluation infrastructure for the round-2 CatBoost / tree-member track

Everything here is CPU only. Module: `nn_x/w12/cb/infra/cbx.py`. Caches: `nn_x/w12/cb/infra/cache/` (42 GB; every array
listed below is pre-built for `xm`, `xs`, `prod200k`, `hold`, `val`, `test`, so normal use only READS there). A set that is
not pre-built (for example `prod500k`) is built on first use: export `CBX_CACHE_EXTRA=<repo>/nn_x/w12/cb/<KEY>/cache`
and it is written into your own directory (atomic writes; reads look in `infra/cache` first).
Rules it implements: `nn_x/w12/cb/PREDECLARED.txt` (member rule, progress rule, law value, stop rule).
Report with numbers: `reports/CATBOOST_1115/infra.md`.

## 1. Start a process (copy these lines)

```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
  NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 && \
  setsid nohup .venv/bin/python -u nn_x/w12/cb/<KEY>/my_arm.py > nn_x/w12/cb/<KEY>/logs/my_arm.log 2>&1 < /dev/null &
```

`cbx` refuses to import when `CUDA_VISIBLE_DEVICES` is not the empty string. Use `python -u` (CatBoost's progress lines are
block-buffered otherwise). `CBX_THREADS=4` lowers every thread cap for light checks.

## 2. Data

```python
import sys; sys.path.insert(0, "<repo>/nn_x/w12/cb/infra"); import cbx
tr = cbx.load("xm")            # 134,118 exploration feeds, 19,741,607 rows (the training set of record)
tr.X                           # float32 memmap (19741607, 177); column order = tr.names = cbx.feature_names()
tr.y                           # float32 click labels (train sets only; guarded for 'val' and 'hold')
tr.gid                         # int32 group id per row (feed position inside the set, contiguous, file order)
tr.M, tr.off, tr.fidx          # per feed: size, row offset inside the set, index into data/flat/train/feeds.parquet
tr.uid_feed, tr.uid            # user id per feed / per row (0 = unknown user)
tr.item_id, tr.cat             # per row: item id (int32), native category id (int8, 0..53, 0 = unknown)
tr.rows                        # flat row indices into data/flat/train/cand_*.npy (None for val / test = identity)
tr.pos                         # train only: production position, int16 (does not exist for val / test: never a model input)
tr.p_random, tr.xfold          # per feed: raw p_random; user fold 0..4 of the cross-fitted teacher
tr.teacher("mean11")           # teacher logit per row; forms: "logit" | "centred" | "soft" (within-feed softmax, T=1)
tr.teacher("oof", "centred")   # cross-fitted teacher (out-of-fold by user fold)
tr.emb_cand_i8                 # int8 memmap (n_rows, 128): candidate item embedding, zero row if missing
tr.emb_feed("user")            # per FEED float32 (n_feeds, 128); kinds: user | hmean | h5 | h20
tr.emb_rows("h5", np.float16)  # per ROW (n_rows, 128) in RAM; kinds: cand | user | hmean | h5 | h20
va, te, ho = cbx.load("val"), cbx.load("test"), cbx.load("hold")     # prediction sets (labels are guarded)
big = cbx.concat(cbx.load("xm"), cbx.load("xs"))                       # 485,023 feeds: all exploration-like feeds incl. M < 50
prod = cbx.load("prod200k")                                            # 200,000 uniformly sampled production feeds (any N: "prod<N>k")
```

| set | split | feeds | rows | what |
|---|---|---:|---:|---|
| `xm` | train | 134,118 | 19,741,607 | exploration: p_random > 0.5 and M >= 50, minus the 30K holdout |
| `xs` | train | 350,905 | 5,361,945 | natural short exploration-like feeds: raw p_random > 0.5 and M < 50 |
| `prod<N>k` | train | N x 1000 | about 83 rows / feed | uniform sample (seed 20261115) of the 2,236,185 production feeds = raw p_random <= 0.5 and not NaN |
| `hold` | train | 30,000 | 4,630,966 | the 30K holdout: evaluation only, row order of `nn_x/w5/preds/holdout` |
| `val` | val | 14,423 | 1,389,971 | row order of `preds/*_val.npy` |
| `test` | test | 28,595 | 2,904,806 | row order of `preds/*_test.npy` |

Teacher files (train rows only, aligned to `data/flat/train` rows; `ds.teacher()` gathers the set's rows):

| name | file | producer | scale |
|---|---|---|---|
| `mean11` | `nn_x/w12/kd/out/teacher_mean11.npy` | mean logit of 11 hard-label train-only NN members (`r6_cd_m02_r101-107`, `w5_e10_base_s951-954`), none trained on the holdout | already centred per feed (feed mean 0); within-feed sd 1.14 (1.21 on `xm`); IN-SAMPLE on training feeds |
| `oof` | `nn_x/w12/kd/out/teacher_oof.npy` | per user fold k: mean logit of the 3 models `w5_e10_xf<k>_s951-953` that excluded fold k (`cbx.xfold_of`) | RAW logits, per-feed offset about +114, within-feed sd 1.12 (1.18 on `xm`): use `form="centred"` or `"soft"` |

The NN students' soft label is `softmax(teacher_mean11 / 1.0)` inside the feed = `tr.teacher("mean11", "soft")`.
Use `mean11` as a TARGET only. Any NN-derived INPUT for train rows must be `oof`.

Raw embeddings: the item table is `data/competition/<split>/item_embeddings/data.memmap` (int8, 128-d, one table per split,
indexed by `cand_emb_idx` / `hist_emb_idx`; index -1 = no embedding). `user_emb.npy` is int8 (n_feeds, 128), one row per feed.
Feature `emb_missing` = 1 iff the CANDIDATE has no embedding (`cand_emb_idx` < 0): 0.004% to 0.015% of rows; 16.8% of history
events have no embedding; 2.3% of feeds have an all-zero user embedding. History is stored newest first; `h5` / `h20` = mean
over the 5 / 20 most recent events THAT HAVE an embedding, `hmean` = over all of them; `ds.emb_feed_n(kind)` = how many were
averaged (0 -> zero vector). Scale is raw int8 units (item norm about 256, user norm about 250), not normalised.
Memory per embedding kind on `xm`: 2.5 GB int8 (cand cache on disk), 5.05 GB float16, 10.1 GB float32 per-row matrix.

## 3. Pools, the train-side selection split, thread caps

```python
fit, sel = cbx.trainside_split(tr)                      # per-feed masks, user-grouped: 120,591 fit / 13,527 selection feeds on xm
ptr = cbx.make_pool(tr, feeds=fit, label=tr.y)          # 177 features, category_id native categorical (as the record)
pes = cbx.make_pool(tr, feeds=sel, label=tr.y)          # early stopping set: NO val label anywhere
m = CatBoostRanker(loss_function="YetiRank", depth=8, learning_rate=0.08, l2_leaf_reg=5, border_count=254, random_seed=0,
                   iterations=2500, od_type="Iter", od_wait=200, eval_metric="NDCG:type=Base", use_best_model=True,
                   allow_writing_files=False, thread_count=cbx.THREADS).fit(ptr, eval_set=pes, verbose=50)
pv, ph, pt = (m.predict(cbx.make_pool(d), thread_count=cbx.THREADS) for d in (va, ho, te))
# other pool options
cbx.make_pool(tr, feeds=fit, label=tr.teacher("mean11", "centred"))                 # KD regression target (RMSE)
cbx.make_pool(tr, feeds=fit, label=tr.y, embeddings=["cand", "user", "h5"])         # CatBoost embedding features
cbx.make_pool(tr, feeds=fit, label=tr.y, cat="numeric", cols=[...])                 # column subset / category as a number
cbx.make_pool(tr, feeds=fit, label=tr.y, extra={"item_id": None, "user_id": None}, extra_cat=["item_id", "user_id"])  # CTRs on ids
cbx.make_pool(tr, feeds=fit, label=tr.y, extra={"nn": cbx.nn_input})                # OOF NN logit as an input column
cbx.make_pool(tr, feeds=fit, label=tr.y, baseline=cbx.nn_input(tr))                 # boosting on top of the OOF NN logit
```

NN scores as INPUTS (never `teacher("mean11")` on training feeds: it is in-sample there):

| call | rows | what |
|---|---|---|
| `cbx.nn_input(tr)` = `"oof"` | any train set | cross-fitted teacher, centred per feed (3 models that never saw the feed's user fold) |
| `cbx.nn_input(va, "xf<k>")` | val / hold / test | mean of the 3 fold-k models `w5_e10_xf<k>_s951-953` (like-for-like partner of `"oof"`), k = 0..4 |
| `cbx.nn_input(va)` = `"xf15"` | val / hold / test | mean of all 15 fold models (less noisy than the 3-model train input: note the mismatch) |
| `cbx.nn_input(va, "mean11")` | val / hold / test | the 11 members behind `teacher_mean11` (all out-of-sample on val, test and the holdout) |

An `extra` value may be an array aligned to the set's rows or a callable `f(ds)`; with callables the same `pool_kw` builds the
train, val, holdout and test pools inside `cbx.run_arm`. For a baseline arm give `pred_pool_fn=lambda d: cbx.make_pool(d,
baseline=cbx.nn_input(d, "xf15"))` to `run_arm`.

Thread caps: pass `thread_count=cbx.THREADS` to EVERY CatBoost constructor, `Pool`, `predict`, `staged_predict`,
`get_feature_importance`, `eval_metrics`. `Pool.quantize()` has no thread argument and uses every core of the machine:
call `cbx.quantize(pool, border_count=254)` (pins the call to 12 cores) or do not quantize by hand.

The whole arm protocol in one call (train-side fit / selection split, early stopping without val labels, prediction at the
selected iteration, evaluation, saving). Dry-run every arm first with `max_feeds=3000` (about 20 s):

```python
from catboost import CatBoostRanker
m = CatBoostRanker(loss_function="RMSE", depth=8, learning_rate=0.1, iterations=3000, od_type="Iter", od_wait=150,
                   eval_metric="NDCG:type=Base", l2_leaf_reg=5, random_seed=0, allow_writing_files=False, thread_count=cbx.THREADS)
r = cbx.run_arm("kd_rmse_s0", m, tr, tr.teacher("mean11", "centred"), "<repo>/nn_x/w12/cb/<KEY>/preds",
                pool_kw=dict(cat="native"), budget_min=100)          # selection metric = NDCG on CLICKS of the train-side split
```

`run_arm` writes `<tag>_{val,hold,test}.npy`, `<tag>.eval.json`, `<tag>.curve.json`, `<tag>.cbm`; `touch <out_dir>/<tag>.STOP`
ends the fit cleanly and still evaluates. Example: `smoke_arm.py`.

The selection set is the same for every arm: the 13,527 `xm` feeds of the "selection users" (teacher fold 0, sub-half 0).
`cbx.trainside_split(ds)` returns `fit` = feeds of all other users and `sel` = those `xm` feeds; with a larger training set
(`cbx.concat(xm, xs)`: 436,320 fit feeds; `prod200k`: 180,124) the other feeds of the selection users are simply left out,
and `run_arm` always builds the selection pool from `xm`. In that case pass `baseline` / `weight` as callables `f(ds)`.

## 4. Evaluation (one function for every arm)

```python
ev = cbx.Evaluator()
r = ev.evaluate("my_arm", pv, ph, pt, out_dir="<repo>/nn_x/w12/cb/<KEY>/preds")   # prints + saves + returns dict
ev.compare(pv_arm, ph_arm, pv_base, ph_base, label="arm - baseline")                             # progress rule (paired per feed)
```

`evaluate` reports val and holdout NDCG (mean, SE over feeds, feed count), rho = mean within-feed Pearson of z(X) to z(W16),
the law value `0.0186 (1 - rho_val) - 0.378 gap_val` (and the refit 0.0103 / 0.304), the member-rule statistic
`0.9 z(W16) + 0.1 z(X)` vs `z(W16)` on both witnesses with paired SEs and the pooled z, and the 0.05 / 0.2 weights (reported,
never chosen). It saves `<tag>_{val,hold,test}.npy` (float32; val / test in the row order of `preds/`, holdout in the row
order of `nn_x/w5/preds/holdout`) and `<tag>.eval.json`. NaN / wrong-length predictions raise.
W16 = `w5_e10_kd05w192_s951-958` + `w5_e10_kd05t25w192_s951-958`, train-only predictions, member-equal mean of per-feed z,
re-standardised per feed: val 0.40144, holdout 0.32460.

## 5. Scripts

| file | what | command |
|---|---|---|
| `build_cache.py` | builds caches | `.venv/bin/python -u nn_x/w12/cb/infra/build_cache.py core aux big nn` |
| `smoke_arm.py` | dry run of `cbx.run_arm` on 3,000 feeds (KD-RMSE arm and hard-label arm) | `CBX_THREADS=4 .venv/bin/python -u nn_x/w12/cb/infra/smoke_arm.py` |
| `curve.py`, `tables.py` | read the eval curve of a run next to the GPU record; markdown tables of the report | `.venv/bin/python nn_x/w12/cb/infra/curve.py <tag>` |
| `emb_cost.py` | cost of CatBoost's default embedding calcers on small subsets | `CBX_THREADS=4 .venv/bin/python -u nn_x/w12/cb/infra/emb_cost.py` |
| `smoke.py` | every data path, alignment, leak guards, metric, evaluator | `.venv/bin/python -u nn_x/w12/cb/infra/smoke.py --big` |
| `cb_member_cpu.py` | CPU copy of the CatBoost of record; `--sel valA` (record rule) or `--sel trainside` (no val label) | `.venv/bin/python -u nn_x/w12/cb/infra/cb_member_cpu.py --tag T --sel trainside --lr 0.08 --budget_min 100` |
| `cb_timing.py` | timing table | `.venv/bin/python -u nn_x/w12/cb/infra/cb_timing.py --iters 10` |

`cb_member_cpu.py` stops cleanly (model, predictions and evaluation are still written) when `--budget_min` is used up or
when the file `nn_x/w12/cb/infra/out/<tag>.STOP` is created.
