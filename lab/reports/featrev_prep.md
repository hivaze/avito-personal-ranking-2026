# Feature review: preprocessing, scaling and raw-input parity for each model (featrev_prep)

Author: featrev prep agent, 2026-11-11 (~15:00-16:10 UTC). CPU only (`CUDA_VISIBLE_DEVICES=""`, 16 threads). No live file was modified.
- New code: `feat_new/prep_segments.py` (segment samples) and `feat_new/prew_wrk.py` (one new feature set).
- New arrays: `data/feats_new/{train,val,test}/prew_{fe,new,hmax32}_wrk.npy`.
- Analysis scripts, logs and full tables: `/tmp/scratchpad/featrev/prep/` (`s01`…`s17`, `out/*.log`).
- Full per-input table: `out/nn_trainstage_inputs_posttransform.csv` (also Appendix A/B below).

**Method.** I replicated the NN's input path on CPU, using `nn/model.py:dense_features` imported read-only, plus a copy of `SplitData.gather` / `attach_xt` that includes the fp16 round-trip. Inputs were then standardised with the checkpoint's train-fitted `f_mean` / `f_std` and clipped at ±6 (`nn/model.py:419`).

| item | detail |
|---|---|
| checkpoints | `v12_allprew_s83` (cvnd_prew, 104 inputs, a live nn_final5 candidate) and `v9_e10` (cvnd, 95 inputs, an nn_final4 member) |
| segments (`feat_new/prep_segments.py`, seed 7) | **trR** = 6,000 train random-like feeds (p_random > 0.9, M ≥ 50, 932K rows)<br>**trP** = 6,000 train production feeds (p_random < 0.05, 433K rows)<br>**val** = all 14,423 feeds<br>**pub** = test ts < 1790497920 (8,562 feeds)<br>**priv** = the rest of test (20,033 feeds) |
| "SMD" | standardised mean difference in the reference segment's sd |
| "z-mean" | mean of the model-standardised input, in train-sd units |
| AUC | mean per-feed within-feed AUC |

**Overlap with concurrent reports.** I cite these instead of repeating them: review_shift SHIFT-1..7, review_leakage LEAKAGE-2/6/7, review_integrity INTEGRITY-1/4/6, review_nnmath NNMATH-4/9, featrev_research RESEARCH-1/2.
- The prew clip bug (PREP-1) was found independently by featrev_research (RESEARCH-1) and the inventory agent (`feat_new/inventory_prew_rescale.py`). This report adds the train/val spread mismatch it causes.
- f4 = listing age is already established (feat_life_ §1), and f3 / f1 = user × (sub)category affinity is in STRATEGY.md. Section 3 only adds verification and the parts not covered before.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **The NN's input scaling is in good shape overall.**
   - Of 104 train-stage inputs, only 8 have z-sd outside 0.75-1.25 on val after the model's transform. All 8 are explained below:
     - two rare indicators (`cmiss`, `log1p_inorm`);
     - four zero-inflated co-vis/popularity/ctx columns (`seq_cv_invd20`, `seq_cv_norm`, `it_hist_et4_luo`, `ctx_hn_top1w`);
     - two prew rank channels broken by PREP-1.
   - 12 inputs have more than 0.5% of val rows at the ±6 clip; the largest share is 1.6%:
     - by design: the H = 0 sentinel of the 5 similarity maxima (1.4%), `user0_log1pH` (1.6%) and `log1p_ccnt_et10` (1.4%);
     - heavy tails, harmless (PREP-6): the four ctx_hs sums (0.5-1.0%) and `seq_cv_invd20@fz` (0.9%).
   - fp16 storage costs nothing measurable (AUC identical to 4 decimals).
   - No input drifts val→private by more than 0.16 SMD; the largest are `has:prew_fe` −0.157 and `it_hist_all_luo` −0.119.
   - f1..f5 are stable across all train days, both val days and all 4 test days. The daily means of random-like train, val and test fall within a 0.04 band (f2) or less (others), i.e. ≤ 0.13 train sd.
2. **PREP-1 (medium, verified, live in the cvnd_prew members of nn_final5).**
   - The raw `prew_fe` / `prew_new` saturate at the ±10 clip of `attach_xt`: 85-94% of warm rows sit at 10 on every split.
   - Their only route into the model, the in-model within-feed rank/z (`XT_RELONLY`), therefore runs on ties. Val AUC is 0.562 vs 0.608 unclipped, and only 19% of warm values within a feed are distinct.
   - The saturation share differs between train production (94%) and val (86%). Together with the Sep 6-8 all-NaN train feeds, this makes the standardised spread of the prew rank channel **1.55x larger on val than in the train-fitted stats**, a train/val parity break on top of the information loss.
   - The ordering survives in `prew_fe_z`, so the loss is probably small.
   - Fixes exist: `prew_fe_c` (research agent), `prew_fe_s` (inventory agent), and `prew_*_wrk` (this report: warm-only within-feed rank of the unclipped value, val AUC 0.6096 vs 0.5623).
3. **PREP-2 (medium, verified, a new localisation of SHIFT-1).** Using f4 (exact listing age), the train/val mismatch of the main train-stage popularity input `it_hist_*_luo` is **concentrated in young listings** (partly mid-age; old listings match).
   - Young (f4 < 1/3): zero-popularity share is 36% on train random-like vs 77% val / 87% public / 91% private, and P(pos | zero)/P(pos) is 0.62 on train vs 0.88 on val.
   - Mid-age: 16% vs 22%, ratio 0.73 vs 0.90.
   - Old (f4 ≥ 2/3): the input matches (15% vs 19-20%; ratio 0.92 vs 0.91).
   - After the model's standardisation, it_hist sits at z-mean −0.32 (trR), −0.64 (val), −0.71 (pub) and −0.75 (priv).
   - This explains why the fine-tune's cold re-calibration matters more for private, and why a plain causal recount lost on val (LEAKAGE-7: it also changes the old-item counts that already match).
4. **PREP-3 (low): fine-tune extra branch (n2c_pre, 51 inputs) hygiene.**
   - NaN is imputed to the val mean with no indicator. `pre_idmax32` is 55% / 60% NaN on val / private.
   - The `max > 20` log rule misses zero-inflated heavy tails: `seq_cv_norm` has skew 24.8 and 88.7% zeros.
   - `hour` is log1p-transformed (non-cyclic) and shifts −0.33 / +0.18 sd on public / private.
   - Raw staleness-driven counts shift by up to −0.22 sd (`it_cand_cnt_train`, private) under val-fit stats.
   - Expected value ≤ +0.0003.
5. **Embedding parity is fine.**
   - Each split's own item table is the best for that split: swapping in an earlier or later snapshot lowers cos_u AUC by 0.0002-0.0005 (val ↔ test is neutral at +0.00002).
   - The later tables therefore do not encode later interactions. The pretraining content table's newest-split-first priority is harmless.
   - The user embedding is a per-request real-time vector. Its AUC is flat across train days (0.633-0.650 on random-like M ≥ 50) and matches val (0.641-0.651), relative to the causal maxsim32 control.
   - Its per-dimension distribution is stable: max |SMD| vs val is 0.089 (trR), 0.046 (pub) and 0.038 (priv). Nothing suggests future information.
6. **Feature ideas are small.**
   - The fixed prew rel channels (built).
   - A density-/horizon-matched popularity input judged on the young-item parity diagnostic (uncertain, 0 to +0.001; prior evidence against nearby variants).
   - A fine-tune hygiene bundle (≤ +0.0003).
   - A causal user-interest-drift feature: +0.00038 ± 0.00018 post hoc at w = 0.05 (one of three weights tried), so ≤ +0.0003.
   - An organiser f5 trend was tested and is null (+0.00035 ± 0.00037).

---------------------------------------------------------------------------------------------------------------------

## 1. Findings (most severe first)

### PREP-1 (medium; verified; independently confirmed by RESEARCH-1): in cvnd_prew models, the raw prew_fe / prew_new enter only through in-model rank/z of a column saturated by the ±10 clip, and the saturation differs between train and val

**Code path**

1. `nn/train.py:48-51`: `cvnd_prew` lists `prew_fe`, `prew_new`, `prew_hmax32` in `XT_REL` and in `XT_RELONLY`. Their raw columns are dropped, and only the within-feed pct rank and z computed in the model are kept (`nn/model.py:123-131`).
2. `nn/data.py:199`: the column passed to the model is `np.clip(x, -10, 10)`, because these names match none of the `_rk` / `_z` / `seq_cv` / `it_hist` rules. NaN then becomes 0 and the value is cast to fp16 (`nn/data.py:200`).
3. The live cvnd_prew runs `v12_allprew_s83/s84/s85` and `v12_prew3` all use this path (`nn/logs/v12_allprew_s83.log`: `'xt': 'cvnd_prew'`; the checkpoint cfg has `xt_rel [0, 3, 30, 31, 32]` and `xt_relonly [30, 31, 32]`).

**Evidence** (`s04_xt_checks.py`, `s03_dense_stats.py`, `out/s04.log`, `out/s03_v12.log`)

Share of warm (non-NaN) rows ≥ 10:

| segment | prew_fe | prew_new |
|---|---|---|
| trR | 85.7% | 78.3% |
| trP | 94.1% | 90.3% |
| val | 86.1% | 78.7% |
| pub | 86.8% | 79.5% |
| priv | 87.1% | 80.1% |

`prew_hmax32` lies in [−2, 1] and is unaffected.

Per-feed AUC, all rows, as the model sees them (NaN → 0):

| input | val | trR |
|---|---|---|
| in-model rank of the clipped prew_fe | 0.5623 | 0.5487 |
| rank of the unclipped prew_fe | 0.6083 | 0.6022 |
| prew_fe_z (separate input, carries the ordering) | 0.6204 | 0.6045 |

- On warm rows only: clipped 0.5419 vs unclipped 0.6457 (val).
- Distinct values per feed among warm rows: 19% clipped vs 100% unclipped (val); 16% on trR.
- Spread mismatch: with the checkpoint's train-fitted stats, the standardised sd on val is **1.55** (`prew_fe@fpct`), 1.48 (`prew_new@fpct`), 1.19 (`prew_fe@fz`) and 1.18 (`prew_new@fz`). It has two parts.
  - (i) All prew rank/z channels share a ~1.15-1.2x part. It also appears for the unclipped `prew_hmax32` (1.14 / 1.18). Its cause: 25-28% of train feeds (trR / trP), essentially all from Sep 6-8, are all-NaN for prew, so every candidate in them gets rank 0.5 and z = 0, which compresses the train-fitted sd. On val/test, 0.02-0.09% of feeds are all-NaN. This is the window gap of INTEGRITY-4 / LEAKAGE-6, seen after the transform.
  - (ii) Only the rank channels of the clipped columns get a further 1.3x (1.55 / 1.48 vs 1.14), because the clip saturates 94% of warm train-production rows vs 86% on val.
  - The other within-feed rank/z inputs sit at 0.80-1.01.

**Impact**
- The information is redundant with `prew_fe_z` / `prew_new_z`, so the loss is probably small. It is unmeasured, because measuring it needs a paired GPU run.
- The prew acceptance (+0.0011 ± 0.0007 val, SHIFT-8) was obtained with the bug, so the accepted gain does not rest on these channels. A fixed channel should be neutral to positive, but it still needs the paired test.
- Do not change `attach_xt` in place: existing checkpoints were trained on the clipped column (RESEARCH-1 makes the same point).

**Fix (for the next train-stage run)**
- Use one of the three clip-safe inputs that now exist in `data/feats_new/`:
  - `prew_fe_c` / `prew_new_c` (research agent) and `prew_fe_s` / `prew_new_s` (inventory agent) are fixed affine maps into (0, 10). They are drop-in replacements in `XT_REL` / `XT_RELONLY`, and the in-model rank equals the unclipped rank.
  - `prew_{fe,new,hmax32}_wrk` (this report) is the within-feed mid-rank percentile among warm candidates only, NaN for cold. Pass it through raw with `has:prew_fe` and remove prew from `XT_REL`. Val per-feed AUC with NaN → 0 is 0.6096, vs 0.5623 for the in-model clipped rank.
- I recommend the affine copies as the primary fix, because they keep both the rank and the z semantics of the accepted recipe.

### PREP-2 (medium; verified; extends SHIFT-1 / NNMATH-9 / INTEGRITY-4 / LEAKAGE-7 / RESEARCH-2): the it_hist_*_luo train/val mismatch is concentrated in young listings

**Evidence** (`s14_age_cov.py`, `out/s14.log`; f4 = exact listing age per feat_life_ §1, re-verified in section 3)

| segment | zero-pop share: all / young / mid / old (f4 terciles) | P(pos \| zero, tercile) / P(pos \| tercile): young / mid / old | f4 mean, zero-pop vs pop > 0 |
|---|---|---|---|
| train random-like | 0.222 / **0.364** / 0.160 / 0.147 | **0.62** / 0.73 / 0.92 | 0.370 vs 0.531 |
| val | 0.379 / **0.771** / 0.219 / 0.187 | **0.88** / 0.90 / 0.91 | 0.332 vs 0.611 |
| public | 0.428 / 0.873 / 0.261 / 0.200 | – | 0.342 vs 0.634 |
| private | 0.456 / **0.912** / 0.298 / 0.204 | – | 0.338 vs 0.650 |

**After the model's transform** (`out/s03_v12.log`): the train-fitted standardisation is dominated by production feeds.

| input | z-mean trP / trR / val / pub / priv | SMD trR vs val (M ≥ 50 only) |
|---|---|---|
| it_hist_all_luo | +0.04 / −0.32 / −0.64 / −0.71 / −0.75 | +0.33 |
| it_hist_et1_luo | +0.04 / −0.32 / −0.61 / −0.67 / −0.71 | +0.32 |

- These are the largest train(random)→val shifts of all 104 inputs; et2/et3 follow at +0.22.
- The per-feed AUC of `it_hist_all_luo` is 0.571 on trR vs 0.556 on val (LEAKAGE-7 reports the same gap on the holdout: 0.5716 vs 0.5561).

**Reading**
- Train rows count other users' *future* dense production snapshots. For a young listing these are most of its count. For an old listing, the past dominates and the count matches val/test.
- So on train, "young and never seen" is rare and strongly negative. On val/test it is the typical cold listing and nearly neutral.
- This is the it_hist counterpart of RESEARCH-2 (co-visitation censoring). It explains two earlier observations:
  - the fine-tune gains most on cold-positive feeds (SHIFT-3);
  - a plain causal recount lost on the train-fit LightGBM (LEAKAGE-7: −0.0018 ± 0.0008). That recount also shrinks the old-item counts on early train days, and those counts already match.

**Fix / test**
- Any popularity variant for train rows should be judged on this young-item diagnostic (target: train young zero-share → ~0.77, ratio → ~0.88) and on train-only val.
- A density-matched variant is described in Idea 2.
- Until then, keep the val fine-tuned half in the final mix (SHIFT-3), because it is the component that repairs young-cold calibration.

### PREP-3 (low; verified): fine-tune extra branch (n2c_pre, 51 inputs): NaN imputation, missed logs, non-cyclic hour, val-fit stats with test drift

**Code.** `nn/data.py:203-220` (`attach_extra`):
- log1p if min ≥ 0 and max > 20 on **val**;
- then (x − mean_val) / sd_val, clip ±6, NaN → 0, i.e. NaN takes the val mean;
- stats fitted on all of val and reused for test (`nn/finetune.py:166-167`).

**Evidence** (`s02_ft_extras.py`, `out/s02.log`, Appendix B)

(a) Mean imputation without an indicator
- `pre_idmax32`: NaN 55.3% (val) / 60.1% (private), row-level (98% of feeds partially NaN).
- NaN rows have positive rate 0.0140 vs 0.0181. NaN is imputed to 0.59, about the median of present values (p10/p50/p90 = 0.26 / 0.65 / 0.83).
- 69% of NaN rows have it_hist_all_luo = 0, so the cold signal largely reaches the branch through other inputs.
- `seq_max_all` / `seq_top3` / `seq_argmax_hpos` / `ctx_nd_*` are NaN on 1.4% (val) and 0.9% (private) of rows. These are whole feeds with no usable history (mostly H = 0).

(b) Heavy tails left unlogged (the max > 20 rule misses them)
- `seq_cv_norm`: skew 24.8, 88.7% zeros, max 5.69; z-sd on private 0.50.
- `it_hist_all_luo_z`: skew 5.9.
- `u_tr_ctr`: skew 5.8.
- `pre_hmax32` / `pre_hmax8`: skew −6.1 / −5.5 because of a −2 sentinel (1.4% val / 0.9% test rows), which ends up at the −6 clip and so acts like an indicator.

(c) Val-fit stats with population drift (z-mean on public / private)

| input | public | private |
|---|---|---|
| `hour` | −0.325 | +0.180 |
| `it_cand_cnt_train` | −0.157 | −0.219 |
| `it_tr_coec_gap` | +0.066 | +0.144 |
| `it_hist_all_luo` | −0.082 | −0.121 |
| `it_tr_pos` | −0.074 | −0.104 |

- These are staleness and calendar effects, not construction differences (LEAKAGE "checked": PSI ≤ 0.07 except hour and it_tr_coec_rk).
- The within-feed `_rk` / `_z` versions of the same signals stay at z-mean +0.000.

(d) Other
- `hour` goes through log1p because its max of 23 is > 20. That puts hour 0 far from hour 23 and compresses the evening (SHIFT-7 covers the val/test hour mix).
- `u_tr_feeds` treats user 0 as one user with 123,904 feeds (SHIFT-6).

**Impact.** Small. The branch is a zero-initialised MLP (`nn/model.py:367-373`) trained for 4 epochs on about 11.5K feeds, and it already gets correlated clean inputs (`_rk` / `_z`, it_hist = 0). Expected ≤ +0.0003 for the whole bundle (Idea 3).

### PREP-4 (low; verified): train-stage xt columns map NaN → 0 without an indicator, colliding with legitimate zeros

**Evidence** (`out/s03_v12.log`, `out/s04.log`)

| input | legitimate exact zeros on val | NaN rate trR / val / priv |
|---|---|---|
| `ctx_hn_top1w` | 87.7% | 3.1% / 1.4% / 0.9% |
| `ctx_hs_mutual` | 89.9% | same |
| `ctx_hn_bestrk` | 11.5% (lower is better; AUC 0.37) | same |

- There is no `has:` column for any ctx input; `has:prew_fe` is the only indicator in the live presets.
- **Mitigation that already exists:** nearly all NaN rows sit in feeds that are entirely NaN (3.5% of trR feeds, 1.4% of val feeds, 0.9% of private feeds).
  - These are mostly H = 0 feeds (val H = 0 share 1.35%), which `log1pH` = 0 identifies.
  - The rest are feeds with H > 0 but no valid history embedding.
- The partial-NaN feeds (0.7% trR, 1.4% val) are candidates without an embedding, which `cmiss` flags. Within-feed ranking is therefore unaffected.
- The "0 = best rank" value given to the H = 0 feeds of `ctx_hn_bestrk` acts only through interactions.

**Impact.** About 0. Hygiene for future presets: add `has:ctx_nd_hcent` (one indicator covers the group), or encode NaN as an out-of-range value.

### PREP-5 (info; verified): no explicit "user vector missing" input, and the zero-vector regime changes composition across splits

**Evidence** (`s09_useremb.py`)

| segment | zero user vector | P(H = 0 \| zero) | P(user 0 \| zero) |
|---|---|---|---|
| trR | 4.17% | 0.42 | 0.16 |
| trP | 2.75% | 0.07 | 0.02 |
| val | 1.42% | 0.36 | 0.08 |
| pub | 0.15% | 1.00 | 0.69 |
| priv | 0.33% | 0.66 | 0.33 |

- The zero share falls over the train days on random-like feeds: d03 7.4% → d13 1.8%.
- In the NN a zero vector gives `un` = 0, so cos_u = dot_u = 0 for every candidate, `ce*un` = 0, and the ctx_nd_*_u inputs are NaN → 0. The model can infer the regime from a constant 0, but nothing flags it.
- **Impact:** negligible on test (0.15-0.33% of feeds). INTEGRITY-6 shows val is slightly pessimistic here.

### PREP-6 (info; verified): heavy-tailed unlogged ctx_hs sums are clipped at +10, with no measurable loss

**Evidence** (`s10_clip10.py`, `out/s10.log`)

`ctx_hn_soft05`, `ctx_hs_k128`, `ctx_hs_e1` and `ctx_hs_t10` are attention-mass sums: val median 0.16-0.54, p99 6-10.6, max 58-125.

| measure | value |
|---|---|
| rows > 10 | 0.18-1.2% (trR / val / priv alike) |
| positives above the clip | enriched 2.4-3.2x |
| per-feed AUC, clipped vs unclipped | identical to 4 decimals (e.g. 0.6489 vs 0.6489) |
| standardised sd on val | 1.12-1.16 |

- PLE, a quantile-like re-representation of every dense input, was neutral on val (nnx_exp_feat B1).
- featrev_research reaches the same conclusion.
- **Action:** none. Optionally use log1p for hygiene at the next from-scratch preset.

### PREP-7 (info; verified): f1..f5 need no extra preprocessing, but two semantics matter for feature work

- Daily drift and identity are covered in section 3.
- Within-feed relation to production position is weak, so none of f1..f5 is the production score. Within-feed correlation with pos on production feeds: f1 −0.072, f3 −0.062, f4 +0.068, f2 / f5 ≤ 0.022. On random-like feeds all are ≤ 0.009.
- **f3 is constant within every (feed, category) group** (val: 100% of groups with n > 1; f1: 30%). So f3's within-feed z/pct rank whole categories, not items.
- The NN feeds raw + within-feed z + within-feed pct for all five (`nn/model.py:65-68`). That fits quantile-coded [0, 1] inputs with point masses at the bounds.

---------------------------------------------------------------------------------------------------------------------

## 2. Checked and correct

**Overall train-stage scaling.** Full table in Appendix A.

- **f1..f5, embeddings and similarity inputs.**
  - The 15 f-inputs have val z-sd 0.95-1.04. Raw z-means on val are −0.12..+0.01, the expected production-vs-exploration difference (trR −0.17..−0.03).
  - SMD val→private for f1..f5: −0.016..+0.010.
- **Within-feed rank and z inputs.** The 10 in-model rank/z channels of cos_u / smax / lse / top3 / log1p_ccnt, plus the f z/pct channels, have z-mean 0.00 on every segment and val z-sd 0.995-1.008 (they are split-invariant by construction).
- **Near-constant inputs are harmless.**
  - `cmiss` (train sd 0.0054) and `log1p_inorm` (sd 0.03) carry the 0.003% missing-embedding bit. The item norm itself is quantisation noise (norm 256 ± 0.3; eda_embeddings 1.1), which becomes ±0.04 standardised units.
  - `dot_u` equals 6.55 × `cos_u` (`nn/model.py:72`; every vector has norm 256), a redundant but harmless copy.
- **fp16 storage of xt** (`nn/data.py:200`; `s04`).
  - Max relative error 4.9e-4.
  - Per-feed AUC is identical to 4 decimals for every column tested (seq_cv_invd20, seq_cv_norm, it_hist_all_luo, ctx_nd_max, ctx_nd_knn_h, ctx_hn_rrw, ctx_hs_k128, ctx_hn_soft05, ctx_nd_top5h, prew_fe_z).
  - Within-feed distinct values drop for some similarity columns (ctx_nd_knn_h 0.987 → 0.904 of M on val), with no AUC effect.
  - bf16 inference of the in-model features is covered by NNMATH-4 (noise level).
- **Count logs in the train stage.** `seq_cv*` and `it_hist*` get log1p (`nn/data.py:197`). The in-model category counts use log1p (`nn/model.py:98-100`), and M and H are logged.
- **Val→private drift of train-stage inputs is small.** The largest |SMD| are `has:prew_fe` −0.157 (warm share 62% → 54%), `it_hist_all_luo` −0.119, `it_hist_et1_luo` −0.112, `seq_cv_pop` −0.094, `ctx_nd_knn_u` +0.061 and `cos_u` +0.054. Everything else is below 0.05 (`out/s15.log`).
- **Normalisation stats source.** `feature_stats` (`nn/common.py:91-104`) uses 40 random M-bucketed train batches (about 1.3M candidates, eval mode, no xt dropout). The affine map is identical for every split, so its sampling noise is a fixed reparametrisation, not a parity issue. My CPU replica's trP z-means are 0.00 ± 0.06 for all inputs, which confirms the replica.
- **GBDT harness** (`src/eval_family.py:40-50`, `src/common.py:load_feats`). Raw float32 with NaN kept as missing, `category_id` as categorical, and no scaling. This is correct for LightGBM, and it handles the organisers' quantile codes natively.
- **f1..f5 per-day stability** (`s05_forg.py`, `out/s05.log`).
  - Means per day for random-like train / val / test: f1 0.376-0.395 / 0.397-0.402 / 0.391-0.399; f2 0.404-0.426 / 0.435-0.445 / 0.438-0.442; f3 0.449-0.462 / 0.450-0.451 / 0.444-0.449; f4 0.492-0.500 / 0.503-0.507 / 0.508-0.509; f5 0.443-0.463 / 0.439-0.445 / 0.442-0.446.
  - Boundary masses (values at the encoder's min / max, i.e. "≤ / ≥ train range") are stable. The one exception is f4's lower bound: 1.0-1.3% on train/val vs 0.5-0.8% on test.
  - Per-feed AUC: train random-like 0.627 / 0.491 / 0.605 / 0.548 / 0.569 vs val 0.623 / 0.494 / 0.601 / 0.545 / 0.561 for f1..f5.
- **Item-table snapshot parity** (`s08_embparity.py`, `out/s08.log`).

  | candidates of | found in table | of those, vector differs | median cos | p01 cos |
  |---|---|---|---|---|
  | train feeds | val 15.4% | 23.6% | 0.9996 | 0.909 |
  | train feeds | test 21.4% | 35.5% | 0.9996 | 0.908 |
  | val feeds | train 70.9% | 22.9% | 0.9996 | 0.894 |

  Swapping in the other snapshot for the differing items, paired per-feed AUC of cos(user, item):

  | feeds | swap | ΔAUC |
  |---|---|---|
  | train | → later val vector | −0.00027 ± 0.00010 |
  | train | → later test vector | −0.00016 ± 0.00015 |
  | val | → earlier train vector | −0.00053 ± 0.00019 |
  | val | ↔ test vector | +0.00002 ± 0.00011 |

  - Each split's own table is the most time-local. The later tables do **not** align better with train users' clicks, so they do not encode later interactions.
  - Consequences:
    - (i) The NN sim inputs, sim_bias, ctx_* and seq_* all use the split's own table (`nn/data.py:52-55`, `feats_v2/*/` builders load `competition/<split>/item_embeddings`). That is correct.
    - (ii) The pretraining content table takes the newest split first (`pretrain/build_corpus.py:139-150`), and item_tab2 adds train candidates (`pretrain/build_item_tab2.py`). Val rows then get test vectors for shared items (neutral, +0.00002). Train rows in pre2 get the val/test vector for items that also occur in those tables (15-21% of train candidate rows, of which 24-36% differ from the train vector), costing about −0.0002 (not a leak).
    - (iii) clu_ assignments can flip only for the ~1% of shared items with cos < 0.9 (not measured; negligible).
  - Leak-rule reminder: presence in a *later* split's table is a future-existence signal. Train candidates also present in the val table have positive rate 1.38-1.45% vs 1.11% for unshared ones. No live input encodes it, apart from the known test-snapshot horizon of val `it_hist` (SHIFT-2).
- **The user embedding is a real-time, per-request vector with no future information** (`s09_useremb.py`; eda_embeddings 1.4; `eda/history_leak/09_ue_leak.log`).
  - Per-feed AUC of cos(user, item) on random-like M ≥ 50 train feeds, by day: 0.6431, 0.6421, 0.6488, 0.6425, 0.6501, 0.6432, 0.6405, 0.6415, 0.6420, 0.6465, 0.6327, 0.6390 (d00..d13). Val: 0.6425 / 0.6513 (Tue / Thu, M ≥ 50). There is no early-day excess.
  - The cos_u − maxsim32 gap is −0.0011 on average on train vs −0.003..−0.008 on val; per-day sd is about 0.003, so this is within noise.
  - The own-feed embedding does not contain the own clicks: cos(ue_own) 0.3630, vs 0.3939 for the *next* feed's embedding (09_ue_leak).
  - Per-dimension SMD vs val, non-zero vectors: max 0.089 (trR), 0.058 (trP), 0.046 (pub), 0.038 (priv). The mean vectors agree to cos ≥ 0.9991.
- **History embeddings and missing values.** History events without an embedding (14-15.5%) are masked (`hemb_ok`, `nn/data.py:160-163`) and never pooled as zero vectors. Candidate missing embeddings (0.003-0.016%) are zero-filled with the `cmiss` flag.
- **f1 as category interest beyond the visible history** (`s07_f1cat.py`, `s12_f1beyond.py`).
  - In 46% of rows whose category is absent from the 256-event history, f1 is above its minimum. Those rows have positive rate 0.0084 vs 0.0051.
  - The current best mix already uses this: within mix-percentile bins, the f1 > min vs f1 = min positive rates do not differ consistently (`out/s12.log`). No residual feature is needed.

---------------------------------------------------------------------------------------------------------------------

## 3. Organiser features f1..f5: what they are, and verification

| | what it is | evidence (this report unless cited) | how the NN uses it | verdict |
|---|---|---|---|---|
| **f1** | user × sub-category affinity score (recency-weighted, partly beyond the visible history) | Spearman with history category count cat_cnt_all: 0.62 / 0.68 / 0.79 for H < 50 / 50-255 / 256 (val); with cat_cnt_20 0.64-0.81. Constant within (feed, category) in only 30% of groups. For H = 0 feeds, f1 mean 0.081, and 29% of rows are above the min. Global corr with maxsim32 +0.45 and cos_u +0.38. Category eta² 0.21, between-feed variance share 0.32 | raw + feed z + feed pct | fine |
| **f2** | unknown impression-level quantity, weak | AUC 0.494 val / 0.491 trR. Within-item variance share 0.45, between-feed share 0.31. 15% mass at 0. Weak corr with centrality (+0.25) and cos_u (+0.14). Candidate meanings such as user-item distance are speculation | same | fine (weak) |
| **f3** | user × category affinity | constant within every (feed, category) group (100%). Spearman with cat_cnt_all up to 0.885 (H = 256). f1–f3 correlation 0.82 | same; its within-feed z/pct rank whole categories | fine |
| **f4** | exact listing age (feat_life_ §1) | 97.5% of 16.0M multi-day items increase, none decrease. Δf4 over a gap is explained by (f4_last, gap) with R² 0.997 on val; velocity falls with level (+0.044 → +0.0001 per day by start decile). Corr with popularity +0.52 (it_hist), +0.54 (seq_cv_pop) on val | same | fine; used in PREP-2 |
| **f5** | dynamic item rate (popularity or quality-like), mean-reverting | item-level (within-item variance share 0.06). Over > 2 days, 44% of items go up and 51% go down; the start level explains only R² 0.28 of the change (low start rises, high start falls). Corr with it_hist +0.34 and seq_cv_pop +0.33 | same | fine; its trend is null (Idea 6) |

All five are quantile-type encodings fitted on train: f1 ∈ [1/56, 55/56], f2 ∈ [0, 63/64], f3 ∈ [1/62, 61/62], f4 ∈ [1/64, 63/64], f5 ∈ [0, 61/62] (`s05` value scan). The NN's train-fitted standardisation (sd 0.29-0.32) and within-feed z/pct suit them. No log, clip or quantile map is needed.

---------------------------------------------------------------------------------------------------------------------

## 4. Feature ideas and recommended transforms (model-specific; the expected value is my estimate, with the evidence given)

1. **Clip-safe prew rel channels** (NN train stage; built: `data/feats_new/*/prew_{fe,new,hmax32}_wrk.npy`, or the affine copies `prew_*_c` / `prew_*_s`).
   - Expected: 0 to +0.0005, unmeasured. Also removes the 1.55x val/train spread mismatch.
   - Cost: one paired 2-seed run when the next cvnd_prew batch is trained anyway.
2. **Density-matched popularity for train rows, judged on the young-item diagnostic** (NN train stage).
   - Train row at time t: other-user snapshots with ts < t (dense) + other-user snapshots with ts ≥ t, thinned by a deterministic hash of the snapshot's feed_id at rate ρ ≈ the exploration sampling rate (about 3%), + val/test snapshots as now.
   - Val/test rows keep the as-built count, which already has "dense past, sparse future" semantics. LUO and the user-0 exclusion apply unchanged.
   - Acceptance: young-item zero-share on train random-like moves from 0.36 toward 0.77, P(pos | zero, young) ratio toward 0.88, train-only val paired > 2 SE, and the trunc17 emulated val (SHIFT idea 5) not worse.
   - Caveat: on Sep 6-8 the dense past covers only 0-3 days, so even old items look colder there than on val (val sees 14 dense days). Judge the diagnostic on Sep 12+ rows, or add an explicit "days of dense past" input.
   - Expected: uncertain, 0 to +0.001, mostly on private. Prior evidence against nearby variants: the causal and lagged recounts lost on a train-fit LightGBM (LEAKAGE-7). Screen first with LEAKAGE's `lgb_causal.py` harness (about 9 min).
3. **Fine-tune extra-branch hygiene bundle** (val/test-only inputs; nested 5-fold, one pre-declared choice).
   - `has:pre_idmax32`.
   - Signed log `sign(x)·log1p(|x|)` for `seq_cv_norm`, `it_hist_all_luo_z`, `it_tr_coec_z` and `u_tr_ctr`.
   - `hour` replaced by 4 bins (0-5, 6-11, 12-17, 18-23) or dropped (SHIFT-7).
   - `u_tr_*` NaN for user 0 (SHIFT-6).
   - Drop raw `it_cand_cnt_train` and keep the within-feed versions.
   - Expected ≤ +0.0003 (PREP-3).
4. **Causal user-interest drift** (GBDT harness or fine-tune extra; not the train stage, where coverage differs).
   - x = cos(c, u_now) − cos(c, u_prev), where u_prev is the user embedding of the same non-zero user's latest strictly earlier feed in any split. NaN if there is none.
   - Measured on val (`s16_udrift.py`): coverage 29.4% of feeds (median gap 9.9 days), per-feed AUC 0.548 on covered feeds.
   - Fixed blend over the best mix: w = 0.05 gives +0.00038 ± 0.00018 (+0.00129 ± 0.00060 on covered feeds), w = 0.1 gives +0.00016 and w = 0.2 gives −0.00066. These are post-hoc picks among 3 weights, so treat it as ≤ +0.0003.
5. **Explicit regime flags in the next from-scratch train-stage preset**: `u_missing` (zero user vector) and `has:ctx_nd_hcent`. Expected about 0 (PREP-4/5). Hygiene only, because the regimes are rare on test.
6. **Not recommended (tested null or no headroom).**
   - Organiser f5 trend: d5 = f5_now − f5 at the item's last train appearance. Val coverage 65.4%, AUC 0.508; best-mix blend +0.00035 ± 0.00037 (w = 0.05) and −0.00041 (w = 0.1) (`s13_f5trend.py`).
   - log1p or quantile maps for the ctx_hs sums: no AUC loss from the clip, and PLE was neutral (PREP-6).
   - An "f1 beyond history" residual: the mix already exploits it (section 2).
   - An f4 velocity residual: AUC 0.498, because f4 is deterministic age.

Literature behind the transform choices:
- Qin et al., ICLR 2021, "Are neural rankers still outperformed by GBDTs?": a symmetric log1p transform of heavy-tailed LTR features is essential for neural rankers.
- Gorishniy et al. 2021/2022: quantile and PLE numeric representations.
- Holzmüller et al. 2024 (RealMLP): robust scaling plus smooth clipping.

In the train stage, our pipeline already logs the heavy-tailed count families (`seq_cv*`, `it_hist*`, the in-model category counts). The few unlogged tails (ctx_hs sums; `seq_cv_norm` in the fine-tune branch) lose nothing measurable. PLE / TabM were tested and rejected (nnx_VERDICT). The remaining problems are therefore parity of meaning (PREP-1, PREP-2), not scale.

---------------------------------------------------------------------------------------------------------------------

## Appendix A: train-stage NN inputs after the model's transform (checkpoint v12_allprew_s83, cvnd_prew; the cvnd model v9_e10 uses the first 91 rows plus the 4 rel channels of seq_cv_invd20 and it_hist_all_luo)

How to read the columns:
- "train sd" is the checkpoint's `f_std`.
- z-mean is in train-sd units after `(X − f_mean)/f_std`.
- "at ±6 clip" is the share of val rows at the model's clamp.
- "NaN→0" is the share of source NaNs (xt columns only).
- `@fpct` / `@fz` are the in-model within-feed rank / z of an xt column.
- `_fz` / `_fpct` are the in-model within-feed z / pct of a dense input.

Generated by `s17_appendix.py` from `out/nn_trainstage_inputs_posttransform.csv`.

| input | train sd | z-mean trR / trP / val / pub / priv | z-sd val | at ±6 clip (val) | exact 0 (val / priv) | NaN→0 (trR / val / priv) | SMD trR−valM50 | SMD priv−val | AUC trR / val |
|---|---|---|---|---|---|---|---|---|---|
| `f1` | 0.312 | -0.17 / +0.05 / -0.12 / -0.12 / -0.14 | 1.01 | 0.0000 | 0.000 / 0.000 | – | -0.050 | -0.013 | 0.627 / 0.623 |
| `f1_fz` | 0.991 | -0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.007 / 0.006 | – | +0.000 | -0.000 | 0.627 / 0.623 |
| `f1_fpct` | 0.287 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.001 / 0.001 | – | -0.000 | -0.000 | 0.627 / 0.623 |
| `f2` | 0.316 | -0.07 / +0.03 / -0.00 / -0.00 / -0.00 | 1.03 | 0.0000 | 0.154 / 0.151 | – | -0.070 | +0.001 | 0.491 / 0.494 |
| `f2_fz` | 0.995 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.007 / 0.003 | – | -0.000 | +0.000 | 0.491 / 0.494 |
| `f2_fpct` | 0.265 | -0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.001 / 0.001 | – | -0.000 | +0.000 | 0.491 / 0.494 |
| `f3` | 0.295 | -0.10 / +0.02 / -0.12 / -0.13 / -0.13 | 1.02 | 0.0000 | 0.000 / 0.000 | – | +0.013 | -0.016 | 0.605 / 0.601 |
| `f3_fz` | 0.997 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.004 / 0.002 | – | -0.000 | -0.000 | 0.605 / 0.601 |
| `f3_fpct` | 0.275 | -0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.01 | 0.0000 | 0.004 / 0.003 | – | -0.000 | +0.000 | 0.605 / 0.601 |
| `f4` | 0.289 | -0.03 / +0.00 / +0.01 / +0.02 / +0.02 | 0.95 | 0.0000 | 0.000 / 0.000 | – | -0.037 | +0.010 | 0.548 / 0.545 |
| `f4_fz` | 1 | -0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.000 / 0.000 | – | +0.000 | -0.000 | 0.548 / 0.545 |
| `f4_fpct` | 0.293 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.009 / 0.009 | – | -0.000 | -0.000 | 0.548 / 0.545 |
| `f5` | 0.3 | -0.08 / +0.02 / -0.12 / -0.10 / -0.12 | 1.04 | 0.0000 | 0.013 / 0.011 | – | +0.041 | +0.001 | 0.569 / 0.561 |
| `f5_fz` | 1 | -0.00 / -0.00 / +0.00 / -0.00 / -0.00 | 1.00 | 0.0000 | 0.000 / 0.000 | – | +0.000 | +0.000 | 0.569 / 0.561 |
| `f5_fpct` | 0.293 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 0.99 | 0.0000 | 0.004 / 0.004 | – | -0.000 | -0.000 | 0.569 / 0.561 |
| `cmiss` | 0.00541 | -0.01 / -0.01 / -0.00 / -0.00 / -0.00 | 0.07 | 0.0002 | 1.000 / 1.000 | – | -0.009 | +0.001 | 0.500 / 0.500 |
| `log1p_inorm` | 0.03 | +0.00 / +0.01 / +0.00 / +0.00 / +0.00 | 0.08 | 0.0002 | 0.000 / 0.000 | – | +0.009 | -0.001 | 0.494 / 0.501 |
| `cos_u` | 0.122 | -0.32 / +0.03 / -0.22 / -0.17 / -0.17 | 0.88 | 0.0000 | 0.014 / 0.003 | – | -0.116 | +0.054 | 0.637 / 0.641 |
| `dot_u` | 0.799 | -0.32 / +0.03 / -0.22 / -0.17 / -0.17 | 0.88 | 0.0000 | 0.014 / 0.003 | – | -0.116 | +0.054 | 0.637 / 0.641 |
| `smax` | 0.287 | -0.23 / +0.05 / -0.13 / -0.10 / -0.10 | 0.85 | 0.0142 | 0.000 / 0.000 | – | -0.122 | +0.040 | 0.635 / 0.645 |
| `smax8` | 0.303 | -0.29 / +0.04 / -0.18 / -0.15 / -0.15 | 0.90 | 0.0144 | 0.000 / 0.000 | – | -0.119 | +0.036 | 0.635 / 0.635 |
| `smax32` | 0.295 | -0.28 / +0.05 / -0.18 / -0.14 / -0.14 | 0.88 | 0.0142 | 0.000 / 0.000 | – | -0.123 | +0.039 | 0.641 / 0.648 |
| `smean` | 0.132 | -0.26 / +0.02 / -0.17 / -0.16 / -0.16 | 0.99 | 0.0000 | 0.014 / 0.009 | – | -0.084 | +0.012 | 0.600 / 0.605 |
| `lse` | 0.258 | -0.21 / +0.04 / -0.10 / -0.07 / -0.07 | 0.85 | 0.0142 | 0.000 / 0.000 | – | -0.131 | +0.035 | 0.619 / 0.624 |
| `top3` | 0.297 | -0.25 / +0.05 / -0.14 / -0.11 / -0.11 | 0.88 | 0.0142 | 0.000 / 0.000 | – | -0.132 | +0.037 | 0.637 / 0.645 |
| `smax_et1` | 0.41 | -0.21 / +0.02 / -0.12 / -0.06 / -0.08 | 1.02 | 0.0000 | 0.000 / 0.000 | – | -0.087 | +0.041 | 0.632 / 0.639 |
| `smax_et2` | 0.834 | -0.07 / +0.03 / -0.02 / +0.01 / -0.02 | 0.98 | 0.0000 | 0.000 / 0.000 | – | -0.045 | -0.002 | 0.563 / 0.566 |
| `smax_et3` | 0.666 | -0.14 / +0.03 / -0.06 / -0.02 / -0.05 | 1.00 | 0.0000 | 0.000 / 0.000 | – | -0.081 | +0.002 | 0.569 / 0.572 |
| `smax_et4` | 0.931 | -0.10 / +0.01 / -0.01 / +0.02 / +0.01 | 0.99 | 0.0000 | 0.000 / 0.000 | – | -0.080 | +0.020 | 0.538 / 0.547 |
| `smax_et5` | 1.03 | -0.02 / +0.05 / +0.01 / +0.01 / -0.00 | 0.99 | 0.0000 | 0.000 / 0.000 | – | -0.037 | -0.016 | 0.544 / 0.545 |
| `smax_et6` | 1.04 | -0.01 / +0.06 / +0.03 / +0.05 / +0.01 | 0.99 | 0.0000 | 0.000 / 0.000 | – | -0.040 | -0.015 | 0.538 / 0.541 |
| `smax_et7` | 1.03 | -0.11 / +0.01 / +0.04 / +0.06 / +0.06 | 0.98 | 0.0000 | 0.000 / 0.000 | – | -0.147 | +0.023 | 0.516 / 0.527 |
| `smax_et8` | 1.04 | +0.01 / +0.03 / +0.01 / +0.03 / -0.01 | 0.99 | 0.0000 | 0.000 / 0.000 | – | +0.001 | -0.025 | 0.535 / 0.535 |
| `smax_et9` | 1.01 | -0.08 / -0.01 / +0.03 / +0.06 / +0.04 | 1.00 | 0.0000 | 0.000 / 0.000 | – | -0.110 | +0.018 | 0.506 / 0.511 |
| `smax_et10` | 0.496 | -0.01 / +0.02 / +0.07 / +0.05 / +0.06 | 1.12 | 0.0000 | 0.000 / 0.000 | – | -0.077 | -0.010 | 0.506 / 0.508 |
| `log1p_ccnt` | 1.66 | -0.08 / +0.04 / -0.09 / -0.09 / -0.10 | 1.02 | 0.0000 | 0.158 / 0.157 | – | +0.011 | -0.011 | 0.579 / 0.580 |
| `ccnt_share` | 0.279 | -0.03 / +0.02 / -0.03 / -0.04 / -0.05 | 1.00 | 0.0000 | 0.158 / 0.157 | – | +0.002 | -0.018 | 0.579 / 0.580 |
| `log1p_ccnt_et1` | 1.56 | -0.08 / +0.03 / -0.14 / -0.13 / -0.14 | 0.98 | 0.0000 | 0.228 / 0.225 | – | +0.052 | -0.007 | 0.582 / 0.586 |
| `log1p_ccnt_et2` | 1.16 | -0.02 / +0.02 / +0.03 / +0.02 / +0.01 | 1.03 | 0.0000 | 0.495 / 0.502 | – | -0.036 | -0.017 | 0.538 / 0.536 |
| `log1p_ccnt_et3` | 1.16 | -0.05 / +0.03 / -0.03 / -0.03 / -0.05 | 1.00 | 0.0000 | 0.480 / 0.489 | – | -0.018 | -0.019 | 0.543 / 0.542 |
| `log1p_ccnt_et4` | 1.02 | -0.04 / +0.01 / +0.02 / +0.03 / +0.02 | 1.03 | 0.0000 | 0.616 / 0.617 | – | -0.060 | -0.006 | 0.527 / 0.531 |
| `log1p_ccnt_et5` | 0.961 | +0.04 / +0.03 / +0.03 / +0.03 / +0.00 | 1.03 | 0.0000 | 0.718 / 0.727 | – | +0.006 | -0.024 | 0.528 / 0.524 |
| `log1p_ccnt_et6` | 0.922 | +0.03 / +0.03 / +0.03 / +0.03 / +0.00 | 1.02 | 0.0000 | 0.742 / 0.749 | – | +0.002 | -0.022 | 0.524 / 0.522 |
| `log1p_ccnt_et7` | 0.734 | -0.06 / +0.02 / +0.03 / +0.03 / +0.03 | 1.02 | 0.0003 | 0.722 / 0.723 | – | -0.094 | -0.007 | 0.514 / 0.518 |
| `log1p_ccnt_et8` | 0.78 | +0.04 / +0.04 / +0.04 / +0.04 / +0.01 | 1.06 | 0.0000 | 0.785 / 0.796 | – | +0.001 | -0.028 | 0.522 / 0.519 |
| `log1p_ccnt_et9` | 0.446 | -0.06 / -0.00 / +0.02 / +0.03 / +0.04 | 1.03 | 0.0021 | 0.867 / 0.862 | – | -0.073 | +0.016 | 0.505 / 0.508 |
| `log1p_ccnt_et10` | 0.347 | -0.04 / -0.01 / +0.01 / -0.02 / +0.00 | 0.91 | 0.0144 | 0.963 / 0.964 | – | -0.060 | -0.011 | 0.505 / 0.506 |
| `cat_first_rr` | 0.43 | -0.10 / +0.02 / -0.11 / -0.11 / -0.12 | 0.98 | 0.0000 | 0.158 / 0.157 | – | +0.015 | -0.012 | 0.600 / 0.596 |
| `feed_cat_share` | 0.225 | -0.10 / +0.02 / -0.13 / -0.15 / -0.16 | 0.91 | 0.0000 | 0.000 / 0.000 | – | +0.047 | -0.029 | 0.584 / 0.579 |
| `logM` | 0.877 | +0.51 / +0.03 / +0.25 / +0.31 / +0.28 | 0.84 | 0.0000 | 0.000 / 0.000 | – | +0.110 | +0.043 | 0.500 / 0.500 |
| `log1pH` | 0.984 | -0.15 / +0.06 / -0.06 / -0.03 / -0.04 | 1.00 | 0.0000 | 0.014 / 0.009 | – | -0.082 | +0.021 | 0.500 / 0.500 |
| `cos_u_fz` | 0.984 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.01 | 0.0000 | 0.014 / 0.003 | – | +0.000 | -0.000 | 0.637 / 0.641 |
| `cos_u_fpct` | 0.289 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.010 / 0.010 | – | -0.000 | -0.000 | 0.637 / 0.641 |
| `smax_fz` | 0.993 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0001 | 0.014 / 0.009 | – | +0.000 | +0.000 | 0.635 / 0.645 |
| `smax_fpct` | 0.291 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 0.99 | 0.0000 | 0.010 / 0.010 | – | -0.000 | -0.000 | 0.635 / 0.645 |
| `lse_fz` | 0.993 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.014 / 0.009 | – | -0.000 | -0.000 | 0.619 / 0.624 |
| `lse_fpct` | 0.291 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 0.99 | 0.0000 | 0.010 / 0.010 | – | -0.000 | -0.000 | 0.619 / 0.624 |
| `top3_fz` | 0.993 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0001 | 0.014 / 0.009 | – | +0.000 | -0.000 | 0.637 / 0.645 |
| `top3_fpct` | 0.291 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 0.99 | 0.0000 | 0.010 / 0.010 | – | -0.000 | -0.000 | 0.637 / 0.645 |
| `log1p_ccnt_fz` | 0.993 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.00 | 0.0000 | 0.014 / 0.009 | – | +0.000 | +0.000 | 0.579 / 0.580 |
| `log1p_ccnt_fpct` | 0.272 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.01 | 0.0000 | 0.001 / 0.001 | – | -0.000 | +0.000 | 0.579 / 0.580 |
| `user0` | 0.205 | +0.06 / -0.01 / +0.03 / +0.00 / +0.02 | 1.07 | 0.0000 | 0.950 / 0.953 | – | +0.017 | -0.013 | 0.500 / 0.500 |
| `user0_log1pH` | 0.707 | -0.02 / -0.01 / +0.02 / -0.01 / +0.01 | 1.02 | 0.0163 | 0.957 / 0.959 | – | -0.051 | -0.014 | 0.500 / 0.500 |
| `xt:seq_cv_invd20` | 0.18 | -0.13 / -0.01 / -0.12 / -0.13 / -0.13 | 0.51 | 0.0023 | 0.887 / 0.902 | 0.000 / 0.000 / 0.000 | -0.032 | -0.031 | 0.542 / 0.552 |
| `xt:seq_cv_pop` | 1.52 | -0.15 / +0.00 / +0.02 / -0.05 / -0.07 | 1.01 | 0.0000 | 0.420 / 0.476 | 0.000 / 0.000 / 0.000 | -0.175 | -0.094 | 0.546 / 0.551 |
| `xt:seq_cv_norm` | 0.0578 | -0.12 / -0.02 / -0.12 / -0.13 / -0.13 | 0.43 | 0.0016 | 0.887 / 0.902 | 0.000 / 0.000 / 0.000 | -0.014 | -0.027 | 0.542 / 0.551 |
| `xt:it_hist_all_luo` | 1.7 | -0.32 / +0.04 / -0.64 / -0.71 / -0.75 | 0.95 | 0.0000 | 0.379 / 0.456 | 0.000 / 0.000 / 0.000 | +0.332 | -0.119 | 0.571 / 0.556 |
| `xt:it_hist_et1_luo` | 1.6 | -0.32 / +0.04 / -0.61 / -0.67 / -0.71 | 0.90 | 0.0000 | 0.411 / 0.485 | 0.000 / 0.000 / 0.000 | +0.320 | -0.112 | 0.574 / 0.557 |
| `xt:it_hist_et2_luo` | 1.02 | -0.21 / +0.02 / -0.38 / -0.41 / -0.43 | 0.78 | 0.0000 | 0.689 / 0.728 | 0.000 / 0.000 / 0.000 | +0.222 | -0.067 | 0.542 / 0.533 |
| `xt:it_hist_et3_luo` | 1.12 | -0.24 / +0.03 / -0.40 / -0.44 / -0.46 | 0.76 | 0.0000 | 0.713 / 0.751 | 0.000 / 0.000 / 0.000 | +0.222 | -0.070 | 0.542 / 0.531 |
| `xt:it_hist_et4_luo` | 1.18 | -0.20 / +0.03 / -0.32 / -0.35 / -0.36 | 0.75 | 0.0000 | 0.749 / 0.775 | 0.000 / 0.000 / 0.000 | +0.156 | -0.050 | 0.538 / 0.528 |
| `xt:ctx_nd_hcent` | 0.106 | -0.13 / +0.03 / -0.13 / -0.13 / -0.13 | 1.03 | 0.0000 | 0.014 / 0.009 | 0.031 / 0.014 / 0.009 | -0.012 | -0.003 | 0.597 / 0.606 |
| `xt:ctx_nd_knn_f1` | 0.29 | -0.21 / +0.05 / -0.15 / -0.15 / -0.16 | 1.01 | 0.0000 | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | -0.059 | -0.015 | 0.624 / 0.607 |
| `xt:ctx_nd_knn_h` | 0.162 | -0.42 / +0.05 / -0.33 / -0.31 / -0.30 | 1.03 | 0.0000 | 0.014 / 0.009 | 0.031 / 0.014 / 0.009 | -0.086 | +0.025 | 0.624 / 0.612 |
| `xt:ctx_nd_knn_h_dev` | 0.0664 | -0.01 / +0.00 / -0.04 / -0.04 / -0.04 | 1.01 | 0.0002 | 0.014 / 0.009 | 0.031 / 0.014 / 0.009 | +0.021 | +0.000 | 0.587 / 0.602 |
| `xt:ctx_nd_knn_u` | 0.115 | -0.33 / +0.03 / -0.20 / -0.15 / -0.15 | 0.84 | 0.0000 | 0.014 / 0.003 | 0.038 / 0.014 / 0.003 | -0.138 | +0.061 | 0.614 / 0.600 |
| `xt:ctx_nd_knn_u_dev` | 0.0417 | -0.04 / +0.00 / -0.08 / -0.08 / -0.08 | 1.08 | 0.0005 | 0.014 / 0.003 | 0.038 / 0.014 / 0.003 | +0.022 | -0.002 | 0.590 / 0.602 |
| `xt:ctx_nd_max` | 0.0974 | -0.14 / +0.04 / -0.26 / -0.24 / -0.25 | 1.04 | 0.0002 | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | +0.034 | +0.013 | 0.565 / 0.573 |
| `xt:ctx_nd_top5h` | 0.171 | -0.23 / +0.02 / -0.22 / -0.23 / -0.23 | 0.92 | 0.0000 | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | -0.007 | -0.013 | 0.593 / 0.598 |
| `xt:ctx_nd_top5u` | 0.201 | -0.20 / +0.02 / -0.14 / -0.11 / -0.11 | 0.88 | 0.0000 | 0.014 / 0.003 | 0.038 / 0.014 / 0.003 | -0.074 | +0.030 | 0.596 / 0.594 |
| `xt:ctx_nd_ucent` | 0.0881 | -0.05 / +0.01 / -0.10 / -0.12 / -0.11 | 0.94 | 0.0000 | 0.014 / 0.003 | 0.038 / 0.014 / 0.003 | +0.043 | -0.015 | 0.555 / 0.565 |
| `xt:ctx_hn_top1w` | 0.0587 | -0.14 / -0.03 / -0.09 / -0.09 / -0.09 | 0.72 | 0.0049 | 0.877 / 0.880 | 0.031 / 0.014 / 0.009 | -0.013 | -0.008 | 0.552 / 0.580 |
| `xt:ctx_hn_rrw` | 0.0885 | -0.31 / -0.03 / -0.15 / -0.18 / -0.17 | 0.82 | 0.0026 | 0.014 / 0.009 | 0.031 / 0.014 / 0.009 | -0.043 | -0.017 | 0.645 / 0.643 |
| `xt:ctx_hn_bestrk` | 0.214 | +0.08 / -0.01 / +0.10 / +0.10 / +0.09 | 1.05 | 0.0000 | 0.115 / 0.108 | 0.031 / 0.014 / 0.009 | -0.010 | -0.006 | 0.370 / 0.367 |
| `xt:ctx_hn_margin` | 0.153 | -0.30 / +0.02 / -0.26 / -0.29 / -0.28 | 1.05 | 0.0000 | 0.014 / 0.009 | 0.031 / 0.014 / 0.009 | +0.008 | -0.011 | 0.632 / 0.638 |
| `xt:ctx_hn_soft05` | 1.54 | -0.04 / +0.01 / -0.02 / -0.02 / -0.02 | 1.15 | 0.0000 | 0.014 / 0.009 | 0.031 / 0.014 / 0.009 | -0.009 | +0.003 | 0.649 / 0.649 |
| `xt:ctx_hn_e1_rrw` | 0.0837 | -0.32 / -0.03 / -0.16 / -0.18 / -0.17 | 0.82 | 0.0021 | 0.031 / 0.023 | 0.047 / 0.031 / 0.023 | -0.048 | -0.017 | 0.641 / 0.641 |
| `xt:ctx_hs_mutual` | 0.0237 | -0.12 / -0.01 / -0.08 / -0.08 / -0.08 | 0.76 | 0.0059 | 0.899 / 0.901 | 0.031 / 0.014 / 0.009 | -0.009 | -0.003 | 0.542 / 0.566 |
| `xt:ctx_hs_k128` | 1.31 | -0.03 / +0.00 / -0.02 / -0.01 / -0.01 | 1.16 | 0.0079 | 0.014 / 0.009 | 0.031 / 0.014 / 0.009 | -0.011 | +0.003 | 0.650 / 0.653 |
| `xt:ctx_hs_e1` | 1.46 | -0.03 / +0.00 / -0.02 / -0.01 / -0.01 | 1.16 | 0.0098 | 0.031 / 0.023 | 0.047 / 0.031 / 0.023 | -0.012 | +0.003 | 0.645 / 0.646 |
| `xt:ctx_hs_t10` | 1.06 | -0.03 / -0.00 / -0.01 / -0.01 / -0.01 | 1.12 | 0.0053 | 0.014 / 0.009 | 0.031 / 0.014 / 0.009 | -0.013 | +0.004 | 0.648 / 0.648 |
| `xt:prew_fe_z` | 0.787 | -0.00 / +0.00 / -0.00 / -0.00 / -0.00 | 1.00 | 0.0001 | 0.379 / 0.456 | 0.433 / 0.379 / 0.456 | +0.000 | -0.000 | 0.604 / 0.620 |
| `xt:prew_new_z` | 0.788 | -0.00 / +0.00 / -0.00 / -0.00 / -0.00 | 1.00 | 0.0001 | 0.379 / 0.456 | 0.433 / 0.379 / 0.456 | -0.000 | -0.000 | 0.605 / 0.619 |
| `xt:has:prew_fe` | 0.485 | -0.11 / -0.01 / -0.00 / -0.10 / -0.16 | 1.00 | 0.0000 | 0.379 / 0.456 | 0.000 / 0.000 / 0.000 | -0.111 | -0.157 | 0.528 / 0.544 |
| `xt:seq_cv_invd20@fpct` | 0.173 | +0.00 / +0.00 / +0.00 / -0.00 / +0.00 | 0.91 | 0.0000 | 0.000 / 0.000 | – | +0.000 | +0.000 | 0.542 / 0.552 |
| `xt:seq_cv_invd20@fz` | 0.927 | -0.03 / -0.02 / -0.03 / -0.03 / -0.03 | 0.80 | 0.0087 | 0.045 / 0.045 | – | +0.000 | +0.000 | 0.542 / 0.552 |
| `xt:it_hist_all_luo@fpct` | 0.292 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 0.96 | 0.0000 | 0.000 / 0.000 | – | -0.000 | -0.000 | 0.571 / 0.556 |
| `xt:it_hist_all_luo@fz` | 1 | +0.00 / +0.00 / -0.00 / -0.00 / -0.00 | 1.00 | 0.0000 | 0.000 / 0.000 | – | -0.000 | -0.000 | 0.571 / 0.556 |
| `xt:prew_fe@fpct` | 0.164 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.55 | 0.0000 | 0.001 / 0.000 | – | +0.000 | +0.000 | 0.549 / 0.562 |
| `xt:prew_fe@fz` | 0.843 | +0.00 / +0.00 / -0.00 / -0.00 / -0.00 | 1.19 | 0.0000 | 0.000 / 0.000 | – | +0.000 | -0.000 | 0.549 / 0.562 |
| `xt:prew_new@fpct` | 0.174 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.48 | 0.0000 | 0.001 / 0.001 | – | +0.000 | -0.000 | 0.557 / 0.570 |
| `xt:prew_new@fz` | 0.845 | +0.00 / +0.00 / -0.00 / -0.00 / -0.00 | 1.18 | 0.0000 | 0.000 / 0.000 | – | -0.000 | -0.000 | 0.557 / 0.570 |
| `xt:prew_hmax32@fpct` | 0.248 | +0.00 / +0.00 / +0.00 / +0.00 / +0.00 | 1.14 | 0.0000 | 0.000 / 0.000 | – | -0.000 | -0.000 | 0.589 / 0.601 |
| `xt:prew_hmax32@fz` | 0.851 | +0.00 / +0.00 / +0.00 / -0.00 / +0.00 | 1.17 | 0.0000 | 0.000 / 0.000 | – | +0.000 | -0.000 | 0.589 / 0.601 |

## Appendix B: fine-tune extra branch (n2c_pre, 51 inputs actually loaded; stats fitted on val, applied to test)

| extra | log1p | NaN val / priv | exact 0 val | raw p1 / p50 / p99 / max (val) | skew after log | at ±6 clip val / priv | z-mean pub / priv | z-sd priv |
|---|---|---|---|---|---|---|---|---|
| `it_cand_cnt_train` | yes | 0.000 / 0.000 | 0.346 | 0 / 3 / 513 / 9.37e+03 | +0.8 | 0.0000 / 0.0000 | -0.157 / -0.219 | 0.97 |
| `it_tr_coec` | no | 0.000 / 0.000 | 0.000 | 0.525 / 0.996 / 1.79 / 4.23 | +1.9 | 0.0014 / 0.0010 | -0.017 / -0.019 | 0.86 |
| `it_tr_coec_rk` | no | 0.000 / 0.000 | 0.010 | 0 / 0.567 / 0.997 / 1 | -0.1 | 0.0000 / 0.0000 | +0.000 / +0.000 | 0.96 |
| `it_tr_coec_z` | no | 0.000 / 0.000 | 0.000 | -2.36 / -0.076 / 3.76 / 14.1 | +1.6 | 0.0009 / 0.0015 | -0.001 / -0.001 | 0.99 |
| `it_tr_coec_gap` | no | 0.000 / 0.000 | 0.012 | -1.95 / -0.95 / 0 / 0 | -0.2 | 0.0000 / 0.0000 | +0.066 / +0.144 | 0.98 |
| `it_tr_ctr` | no | 0.000 / 0.000 | 0.000 | 0.00342 / 0.0159 / 0.0628 / 0.187 | +2.9 | 0.0018 / 0.0013 | -0.032 / -0.042 | 0.86 |
| `it_tr_pos` | yes | 0.000 / 0.000 | 0.803 | 0 / 0 / 11 / 207 | +2.9 | 0.0013 / 0.0008 | -0.074 / -0.104 | 0.87 |
| `u_tr_ctr` | no | 0.000 / 0.000 | 0.000 | 0.00619 / 0.0159 / 0.0593 / 0.352 | +5.8 | 0.0035 / 0.0030 | -0.025 / -0.031 | 0.83 |
| `u_tr_feeds` | yes | 0.000 / 0.000 | 0.640 | 0 / 0 / 1.24e+05 / 1.24e+05 | +3.9 | 0.0000 / 0.0000 | -0.028 / -0.023 | 0.97 |
| `hour` | yes | 0.000 / 0.000 | 0.028 | 0 / 13 / 23 / 23 | -2.1 | 0.0000 / 0.0000 | -0.325 / +0.180 | 0.93 |
| `centrality` | no | 0.000 / 0.000 | 0.000 | 0.435 / 0.713 / 0.893 / 0.983 | -0.5 | 0.0000 / 0.0000 | -0.011 / -0.006 | 0.99 |
| `it_hist_all_luo` | yes | 0.000 / 0.000 | 0.379 | 0 / 2 / 369 / 4.96e+03 | +0.8 | 0.0000 / 0.0000 | -0.082 / -0.121 | 0.99 |
| `it_hist_all_luo_rk` | no | 0.000 / 0.000 | 0.010 | 0 / 0.5 / 1 / 1 | -0.0 | 0.0000 / 0.0000 | +0.000 / +0.000 | 1.00 |
| `it_hist_all_luo_z` | no | 0.000 / 0.000 | 0.000 | -0.622 / -0.258 / 4.84 / 16.7 | +5.9 | 0.0062 / 0.0063 | -0.015 / -0.015 | 0.87 |
| `it_hist_all_luo_gap` | no | 0.000 / 0.000 | 0.010 | -2.85e+03 / -486 / 0 / 0 | -2.2 | 0.0019 / 0.0013 | -0.007 / +0.057 | 0.93 |
| `it_hist_et1_luo` | yes | 0.000 / 0.000 | 0.411 | 0 / 1 / 239 / 3.76e+03 | +1.0 | 0.0000 / 0.0000 | -0.077 / -0.115 | 0.98 |
| `it_hist_et2_luo` | yes | 0.000 / 0.000 | 0.689 | 0 / 0 / 26 / 896 | +2.0 | 0.0002 / 0.0002 | -0.044 / -0.067 | 0.96 |
| `it_hist_et3_luo` | yes | 0.000 / 0.000 | 0.713 | 0 / 0 / 29 / 1.24e+03 | +2.0 | 0.0001 / 0.0001 | -0.048 / -0.070 | 0.95 |
| `it_hist_et4_luo` | yes | 0.000 / 0.000 | 0.749 | 0 / 0 / 42 / 2.8e+03 | +2.4 | 0.0007 / 0.0005 | -0.035 / -0.050 | 0.95 |
| `it_hist_et5_luo` | yes | 0.000 / 0.000 | 0.905 | 0 / 0 / 17 / 642 | +4.6 | 0.0055 / 0.0050 | -0.030 / -0.045 | 0.89 |
| `it_hist_et6_luo` | yes | 0.000 / 0.000 | 0.910 | 0 / 0 / 16 / 655 | +4.7 | 0.0059 / 0.0053 | -0.030 / -0.045 | 0.89 |
| `it_hist_et7_luo` | yes | 0.000 / 0.000 | 0.875 | 0 / 0 / 10 / 287 | +3.5 | 0.0023 / 0.0022 | -0.023 / -0.030 | 0.94 |
| `it_hist_et8_luo` | yes | 0.000 / 0.000 | 0.944 | 0 / 0 / 9 / 319 | +5.9 | 0.0081 / 0.0075 | -0.028 / -0.037 | 0.85 |
| `it_hist_et9_luo` | yes | 0.000 / 0.000 | 0.964 | 0 / 0 / 4 / 121 | +6.8 | 0.0082 / 0.0076 | -0.027 / -0.030 | 0.82 |
| `it_hist_et10_luo` | yes | 0.000 / 0.000 | 0.913 | 0 / 0 / 4 / 40 | +4.2 | 0.0046 / 0.0037 | -0.033 / -0.046 | 0.89 |
| `seq_cv_invd20` | yes | 0.000 / 0.000 | 0.887 | 0 / 0 / 0.63 / 93.2 | +11.7 | 0.0064 / 0.0051 | -0.044 / -0.052 | 0.61 |
| `seq_cv_invd20_rk` | no | 0.000 / 0.000 | 0.000 | 0.341 / 0.459 / 0.996 / 1 | +2.3 | 0.0000 / 0.0000 | -0.005 / -0.005 | 0.94 |
| `seq_cv_pop` | yes | 0.000 / 0.000 | 0.420 | 0 / 1 / 279 / 2.97e+03 | +0.9 | 0.0000 / 0.0000 | -0.072 / -0.096 | 0.98 |
| `seq_cv_norm` | no | 0.000 / 0.000 | 0.887 | 0 / 0 / 0.117 / 5.69 | +24.8 | 0.0043 / 0.0033 | -0.039 / -0.044 | 0.50 |
| `seq_max_all` | no | 0.014 / 0.009 | 0.000 | 0.42 / 0.829 / 0.991 / 1 | -0.9 | 0.0000 / 0.0000 | -0.004 / +0.000 | 1.00 |
| `seq_top3` | no | 0.014 / 0.009 | 0.000 | 0.4 / 0.798 / 0.985 / 1 | -0.7 | 0.0000 / 0.0000 | -0.002 / -0.000 | 0.99 |
| `seq_argmax_hpos` | yes | 0.014 / 0.009 | 0.039 | 0 / 46 / 241 / 255 | -0.8 | 0.0000 / 0.0000 | +0.003 / -0.007 | 1.00 |
| `ctx_nd_hcent` | no | 0.014 / 0.009 | 0.000 | -0.364 / -0.0773 / 0.18 / 0.573 | -0.2 | 0.0000 / 0.0000 | +0.002 / +0.000 | 1.01 |
| `ctx_nd_knn_f1` | no | 0.000 / 0.000 | 0.000 | 0.0179 / 0.375 / 0.978 / 0.982 | +0.3 | 0.0000 / 0.0000 | -0.004 / -0.015 | 0.99 |
| `ctx_nd_knn_h` | no | 0.014 / 0.009 | 0.000 | 0.375 / 0.799 / 0.966 / 0.997 | -0.9 | 0.0000 / 0.0000 | -0.008 / +0.002 | 0.99 |
| `ctx_nd_knn_h_dev` | no | 0.014 / 0.009 | 0.000 | -0.244 / -0.0185 / 0.111 / 0.627 | -0.9 | 0.0002 / 0.0002 | +0.006 / +0.002 | 0.99 |
| `ctx_nd_knn_u` | no | 0.014 / 0.003 | 0.000 | 0.278 / 0.512 / 0.633 / 0.71 | -0.8 | 0.0000 / 0.0000 | -0.008 / +0.005 | 1.00 |
| `ctx_nd_knn_u_dev` | no | 0.014 / 0.003 | 0.000 | -0.166 / -0.0105 / 0.0678 / 0.287 | -1.1 | 0.0003 / 0.0003 | +0.005 / +0.002 | 0.99 |
| `ctx_nd_max` | no | 0.000 / 0.000 | 0.000 | 0.55 / 0.885 / 0.994 / 1 | -1.1 | 0.0000 / 0.0000 | +0.011 / +0.005 | 0.99 |
| `ctx_nd_top5h` | no | 0.000 / 0.000 | 0.000 | 0.218 / 0.514 / 0.904 / 0.999 | +0.3 | 0.0000 / 0.0000 | -0.015 / -0.013 | 1.00 |
| `ctx_nd_top5u` | no | 0.014 / 0.003 | 0.000 | 0.223 / 0.538 / 0.913 / 0.997 | +0.2 | 0.0000 / 0.0000 | -0.009 / -0.005 | 0.99 |
| `ctx_nd_ucent` | no | 0.014 / 0.003 | 0.000 | -0.381 / -0.229 / -0.00227 / 0.298 | +0.5 | 0.0000 / 0.0000 | +0.011 / +0.016 | 1.00 |
| `pre_fe` | no | 0.000 / 0.000 | 0.000 | 1.63 / 15 / 22.7 / 27.6 | -0.6 | 0.0000 / 0.0000 | +0.011 / +0.032 | 0.98 |
| `pre_fe_z` | no | 0.000 / 0.000 | 0.000 | -2.72 / 0.14 / 1.77 / 7.2 | -0.7 | 0.0000 / 0.0000 | +0.000 / +0.000 | 1.00 |
| `pre_new` | no | 0.000 / 0.000 | 0.000 | 0.701 / 13.4 / 21.5 / 26.7 | -0.5 | 0.0000 / 0.0000 | +0.016 / +0.039 | 0.98 |
| `pre_new_z` | no | 0.000 / 0.000 | 0.000 | -2.67 / 0.123 / 1.82 / 8.27 | -0.6 | 0.0000 / 0.0000 | +0.000 / +0.000 | 1.00 |
| `pre_hmax32` | no | 0.000 / 0.000 | 0.000 | -2 / 0.787 / 0.977 / 0.997 | -6.1 | 0.0140 / 0.0090 | +0.060 / +0.059 | 0.72 |
| `pre_hmax8` | no | 0.000 / 0.000 | 0.000 | -2 / 0.701 / 0.966 / 0.998 | -5.5 | 0.0140 / 0.0090 | +0.053 / +0.051 | 0.76 |
| `pre_fe_noid` | no | 0.000 / 0.000 | 0.000 | 1.53 / 14.6 / 22.4 / 26.1 | -0.6 | 0.0000 / 0.0000 | +0.020 / +0.047 | 0.98 |
| `pre_id_contrib` | no | 0.000 / 0.000 | 0.000 | -1.94 / -0.00229 / 4.83 / 16.2 | +2.2 | 0.0014 / 0.0010 | -0.041 / -0.064 | 0.91 |
| `pre_idmax32` | no | 0.553 / 0.601 | 0.000 | -0.0136 / 0.647 / 0.892 / 0.958 | -0.9 | 0.0000 / 0.0000 | -0.008 / -0.014 | 0.64 |
