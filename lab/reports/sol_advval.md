# Adversarial validation: train (random-like) vs val vs public vs private

Date 2026-11-11. CPU only (`CUDA_VISIBLE_DEVICES=""`, 16 threads). Nothing under nn/, src/, pretrain/, data/feats/, preds/ or subs/ was modified.

| what | where |
|---|---|
| code | `<repo>/solvers/advval/` (`common_av.py`, `adv_lib.py`, `build_feed.py`, `feed_adv.py`, `cand_adv.py`, `cand_adv2.py`, `cand_groups.py`, `weights.py`, `weights2.py`, `agree.py`, `holdout_w.py`) |
| outputs and logs | `<repo>/solvers/out/advval/` (`*.log`, `*.json`, OOF `*.npz`, `val_weights.npz`) |
| diagnostics | `/tmp/scratchpad/solvers/advval/diag{1..7}.py` |

**Setup**

- **Classifiers.** LightGBM binary. Feed level: 15 leaves, lr 0.03, 400 rounds. Candidate level: 63 leaves, lr 0.05, 300 rounds, max_bin 127.
- **CV.** 5-fold `GroupKFold` by `user_id` across the pooled rows of both sides. Every user-0 feed is its own group, so user identity cannot cross folds. AUC is the pooled out-of-fold (OOF) AUC; the fold spread is in the logs.
- **Rows.**
  - Feed level: all 14,423 val feeds, 8,562 public, 20,033 private, and 89,794 random-like train feeds. Random-like is the nn/train.py holdout pool definition: p_random > 0.9 and M ≥ 50.
  - Candidate level: up to 30 random candidates per feed. The train side is 30K random-like feeds from Sep 9 onward, where pre2 is defined.
  - Random-like train feeds all have M ≥ 50 by construction, so comparisons (d) and (e) restrict val and private to M ≥ 50.
- **Feature sets.**
  - `all` = everything, including the time-identifying features (ts, hour, weekday).
  - `notime` = all minus ts, hour, weekday.
  - `intrinsic` (feed level) / `nocov` (candidate level) = notime minus coverage/staleness features: cold-item shares, zero shares of popularity and co-visitation, prew availability, days since the user's last train feed, it_tr_*, it_cand_cnt_train, pre_idmax32, pre_id_contrib, u_tr_*.
- **Causal definitions for train rows.**
  - Cold item = not seen in train (as candidate or history event) on a day strictly before the feed's day. For val/test it means never seen in train.
  - Prior feeds = the user's train feeds strictly before the feed.
- **Candidate-level inputs are the live ones:**
  - `XT_PRESETS['cvnd']` (nn/train.py:32), the cvnd_prew extras (3 of the 6 nn_final5 members), cvnd_pre2s (nn_final7), f1..f5, category, M, H;
  - embedding proxies (maxsim32, cos_u, cos_h10, cat_cnt_20, emb_missing);
  - the 51 on-disk names of fine-tune preset `n2c_pre` (nn/finetune.py:153, confirmed in nn/logs/v12_allprew_s84_n2cpf_ft.log), val/test only.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **Val, public and private are the same population on intrinsic covariates.**
   - Feed-level AUC: val→private 0.541, val→public 0.546, public→private 0.535.
   - For scale, val Tue→Thu, a 2-day gap with the same pipeline, is 0.526.
   - Candidate level, without coverage features: 0.544 / 0.539 / 0.521.
2. **The whole val→test shift is staleness (item coverage), and it is ordered by time distance.**
   - Coverage-inclusive feed AUC: Tue→Thu 0.855, public→private 0.779, val→public 0.897, val→private 0.950.
   - Top drivers (PSI val→private): cold_item_share 1.07, days_since_prior 0.75, it_hist_all_luo zero share 0.46, prew availability 0.46, seq_cv_pop zero share 0.28, hist10_cold_share 0.24.
3. **A live construction-parity defect is the largest detectable candidate-level difference.**
   - `pre_id_contrib` is a fine-tune extra in v11's n2c_pre: PSI 3.57, 67% of the gain, mean |SHAP| 4.26.
   - Replacing it with `pre_id_contrib_v2` drops the candidate AUC from 0.956 to 0.793.
   - Known (P5 / SHIFT-5). The earlier LightGBM proxy measured its impact at about 0 (+0.00004 ± 0.00031), and W3 swaps it.
4. **With parity fixed, the remaining val→private drift sits in the fine-tune extras, not in the NN train-stage inputs.**
   - Extras alone: AUC 0.737. Train-stage inputs alone: 0.595.
   - `it_tr_coec_rk` has PSI 0.163 overall and **0.261 on cold candidates**: the rank of cold items, which tie on the default value, depends on how many cold items the feed has.
   - Next: `it_cand_cnt_train` 0.071, `it_tr_coec` 0.057, `it_tr_ctr` 0.053.
   - These are exactly the W3 hygiene drops, and val cannot see their private cost.
5. **Random-like train feeds vs val/private (M ≥ 50).**
   - Coverage-inclusive AUC 0.9987 / 0.9999. This is SHIFT-1 re-confirmed: `it_hist_all_luo_zero_share` has PSI 2.76 / 5.30 (train 0.22 vs val 0.38 vs private 0.46).
   - Intrinsic AUC 0.756 / 0.769 comes from history composition: H, and hist et1 share 0.49-0.50 on random-like train, stable over all train days, vs 0.46 on val/test. This is a population difference between random-like production feeds and exploration feeds, not time.
   - Candidate level by input family (train→val): popularity (it_hist/seq_cv) 0.80, prew 0.62, **pre2s 0.61**, f1-5 + category 0.57, M/H 0.57, ctx 0.56, embedding proxies 0.53.
6. **Importance-weighted val.**
   - **Intrinsic weights** (ESS 13,721): they reproduce calib.py's M-only weighting within the procedure's noise. v11 private 0.3931 vs 0.3924, while label-permutation nulls move the level by ±0.0022 (sd).
   - **Deltas are robust:** v11−v10 is +0.0041 ± 0.0009 (null sd 0.00015), nn_final5 fine-tuned − train-only is +0.0009 ± 0.0007, and cd12 − v11 is +0.0009 ± 0.0004.
   - **v11 > v10 survives every weighting** (+0.0033 to +0.0050). The fine-tuned ≥ train-only ranking only flips under the invalid coverage weights, and not significantly (−0.0007 ± 0.0014).
   - **Coverage-inclusive weights are invalid** (ESS 2,880; v11 "private" 0.442), the same failure mode as SHIFT-4.
7. **v11 per-feed NDCG vs the "looks like private" score.**
   - Raw Spearman: −0.116 (intrinsic score) and +0.164 (notime score), both M effects.
   - After removing M from the lift: +0.025 / −0.023 (|z| ≈ 3, practically nil).
   - v11−v10 and fine-tuned − train-only deltas are uncorrelated with the score (|ρ| ≤ 0.012).
8. **Public is a good proxy for private for ranking models.**
   - The same deltas come out under public and private weights (v11−v10: +0.00386 vs +0.00388 M-only, +0.00407 vs +0.00412 intrinsic).
   - Label-free model agreement is identical: fine-tuned~train-only Spearman 0.9847 on both; v11~v10 0.9551 vs 0.9549.
   - Public is about 1 day less stale (cold candidate share 37.1% vs 40.6%) and morning-heavy. A paired public delta has SE ≈ 0.0012, so the public LB resolves only differences ≥ ~0.0025.
9. **Expected val→private gap for v11: about −0.006, range −0.003 to −0.009.**
   - The private estimate is 0.392 ± 0.003, unchanged.
   - Breakdown: M-only 0.39244; hour −0.0006; extra staleness vs public −0.0002 to −0.0006; mean public residual +0.0012.
10. **Do not reweight the train holdout.** Intrinsic weights cut ESS to 15.5K and lower the across-model agreement with val (Spearman 0.975 → 0.961 over 59 models).

---------------------------------------------------------------------------------------------------------------------

## 1. AUC tables

### 1.1 Feed level (`solvers/out/advval/feed_adv.log`)

| comparison | n A / n B | all | notime | intrinsic |
|---|---|---|---|---|
| (a) val vs private | 14,423 / 20,033 | 1.0000 | 0.9499 | **0.5407** |
| (b) val vs public | 14,423 / 8,562 | 1.0000 | 0.8970 | **0.5464** |
| (c) public vs private | 8,562 / 20,033 | 1.0000 | 0.7792 | **0.5345** |
| (d) train random-like vs val M ≥ 50 | 89,794 / 8,530 | 1.0000 | 0.9987 | 0.7556 |
| (e) train random-like vs private M ≥ 50 | 89,794 / 12,186 | 1.0000 | 0.9999 | 0.7688 |
| (f) val Tue vs val Thu (reference) | 7,542 / 6,878 | 1.0000 | 0.8549 | 0.5255 |
| (g) val M ≥ 50 vs private M ≥ 50 | 8,530 / 12,186 | 1.0000 | 0.9773 | 0.5465 |

- Fold AUCs are within ±0.01 of the pooled value in every cell.
- `all` = 1.0 is trivial: the ts ranges are disjoint, and ts takes 89% of the gain.
- These numbers are consistent with the earlier EDA (`eda_shift_overlap.md` §5: val vs test 0.524 intrinsic, 0.909 with overlap features).

### 1.2 Candidate level (`cand_adv.log`, `cand_adv2.log`, `cand_groups.log`)

| comparison | rows A / B | all | notime | nocov |
|---|---|---|---|---|
| (a) val vs private | 377K / 530K | 1.0000 | 0.9562 | 0.5440 |
| (a') same, `pre_id_contrib` → `_v2` | | | **0.7927** | |
| (a') fine-tune extras only (fixed id_contrib, no hour) | | | 0.7374 | |
| (a') NN train-stage inputs only | | | **0.5947** | |
| (a') warm candidates only (fixed id_contrib) | | | 0.7883 | |
| (b) val vs public | 377K / 228K | 1.0000 | 0.9407 | 0.5390 |
| (c) public vs private | 228K / 530K | 1.0000 | 0.5922 | 0.5211 |
| (d) train random-like Sep 9+ vs val M ≥ 50 | 900K / 256K | 1.0000 | 0.8836 | 0.6771 |
| (e) train random-like Sep 9+ vs private M ≥ 50 | 900K / 366K | 1.0000 | 0.8977 | 0.6897 |

Candidate level, by NN input family (200 rounds; `cand_groups.log`):

| family | train → val50 | train → private50 | val50 → private50 |
|---|---|---|---|
| f1-5 + category | 0.573 | 0.585 | 0.515 |
| M, H | 0.568 | 0.574 | 0.506 |
| cvnd popularity (it_hist_*_luo, seq_cv_*) | **0.801** | **0.813** | **0.550** |
| cvnd context (ctx_*) | 0.562 | 0.569 | 0.515 |
| prew (5) | 0.621 | 0.657 | 0.547 |
| pre2s (5) | 0.609 | 0.614 | 0.526 |
| embedding proxies | 0.530 | 0.541 | 0.509 |

---------------------------------------------------------------------------------------------------------------------

## 2. Findings (most decision-relevant first)

### ADV-1 (verified): val→test drift is staleness only; intrinsic composition matches

Univariate PSI, with the means of the two sides in parentheses (`feed_adv.json`, set `all`). Binary features were recomputed with a categorical PSI.

| feature | val→private | val→public | public→private | Tue→Thu | trainRL→val50 | trainRL→private50 |
|---|---|---|---|---|---|---|
| cold_item_share | **1.074** (0.290→0.406) | 0.568 (→0.371) | 0.106 | 0.389 (0.258→0.324) | 0.673 | 3.208 |
| it_hist_all_luo zero share | 0.460 (0.381→0.459) | 0.184 | 0.072 | 0.131 | **2.764** (0.221→0.381) | **5.300** (→0.456) |
| days_since_prior (days) | 0.749 (9.8→15.2) | 0.643 (→13.0) | 0.318 | 0.601 | 0.790 (3.8→9.7) | 1.443 |
| prew availability share | 0.460 (0.619→0.541) | 0.184 | 0.072 | 0.133 | 1.182 | 2.691 |
| seq_cv_pop zero share | 0.277 (0.421→0.480) | 0.178 | 0.016 | 0.153 | 0.253 | 0.753 |
| hist10_cold_share | 0.243 (0.230→0.347) | 0.132 | 0.019 | 0.089 | 0.229 | 0.692 |
| it_cand_cnt_train (feed mean) | 0.132 (30.6→23.1) | 0.054 | 0.019 | 0.049 | - | - |
| hour | 0.083 (13.0→14.6) | 0.165 | 0.343 | 0.042 | 0.118 | 0.048 |
| M | 0.003 | 0.009 | 0.002 | 0.002 | 0.012 | 0.004 |
| H | 0.002 | 0.003 | 0.001 | 0.002 | 0.078 | 0.095 |
| hist et1 share | 0.003 | 0.004 | 0.001 | 0.004 | **0.147** (0.511→0.466) | 0.151 |
| cold_user | 0.002 | 0.000 | 0.001 | 0.000 | 0.080 | 0.049 |
| user embedding all-zero | 0.016 (1.42%→0.33%) | 0.029 | 0.001 | - | 0.029 | 0.094 |
| f4 mean / f5 sd | 0.002 / 0.004 | 0.006 / 0.004 | 0.003 / 0.001 | 0.005 / 0.002 | 0.021 / 0.030 | 0.034 / 0.062 |

- **Top drifting features per comparison, coverage-inclusive (notime) set, by mean |SHAP|:**

  | comparison | top features (mean abs SHAP) |
  |---|---|
  | (a) val vs private | cold_item_share 1.62, f4_mean 0.65, f4_sd 0.39, days_since_prior 0.37, seq_cv_pop_zero_share 0.34, f5_sd 0.29, hist10_cold_share 0.23 |
  | (b) val vs public | same order |
  | (c) public vs private | cold_item_share 0.47, seq_cv_pop_zero_share 0.46, it_hist_luo_zero_share 0.22, days_since_prior 0.20 |
  | (d) train random-like vs val50 | it_hist_luo_zero_share 1.00, seq_cv_pop_mlog 0.85, it_hist_luo_mlog 0.79, cold_item_share 0.54 |

- **f4 interaction.** f4 shows high SHAP but PSI ≈ 0.002. This is an interaction with coverage: f4 is the listing's exact age (`feat_life_.md` §1).
  - Cold share by val-f4 decile: val [1.00, 0.905, 0.252, 0.078, ...]; private [1.00, 1.00, 0.999, 0.295, ...] (`diag3.py`).
  - Among warm candidates, f4 has PSI 0.73 (0.599 → 0.652), because warm items are older on private by construction.
  - The joint (age, coverage) therefore encodes "time since train end". It is not an f4 problem: the f4 marginal PSI at candidate level is 0.0004.
- **Intrinsic tables.** The intrinsic tables (a)-(c) have no feature above mean |SHAP| 0.08. The top ones are pre_hmax32_mean, maxsim32_mean, pre2_fe_s_mean, f5, cat_n and ctx_nd_ucent_mean, all with PSI ≤ 0.02. This is consistent with AUC 0.54.

### ADV-2 (verified, live): `pre_id_contrib` parity defect is the #1 candidate-level val/test separator

- **Size.** Candidate (a) notime: pre_id_contrib has mean |SHAP| 4.26, gain share 0.669, PSI 3.57 and univariate AUC 0.645. With `pre_id_contrib_v2` the AUC falls 0.956 → 0.793, and the v2 PSI is 0.012.
- **Where it is used.** It is in the PRE list (nn/finetune.py:150) that makes up n2c_pre (line 153), so it is in every v11 fine-tune. The train-stage presets do not contain it (CURRENT_STATUS 16:50).
- **Prior impact estimate.** About 0 (SHIFT-5 LightGBM proxy +0.00004 ± 0.00031). The fine-tune standardises extras with val statistics and applies them to test (nn/finetune.py:180-181), so the test-only zero mass (59%) enters the fine-tuned model unchanged.
- **Status.** W3 (n2c_pre_v2) already swaps it.

### ADV-3 (verified): residual val→private candidate drift is in the fine-tune's train-period statistics

Fixed id_contrib, private vs val; PSI overall / cold candidates / warm candidates (`cand_adv2.log`):

| feature (fine-tune extra) | PSI all | PSI cold | PSI warm | mean val → private |
|---|---|---|---|---|
| it_tr_coec_rk | **0.163** | **0.261** | 0.068 | 0.4995 → 0.5000 |
| it_cand_cnt_train | 0.071 | 0 | 0.018 | 30.4 → 23.1 |
| it_tr_coec | 0.057 | 0 | 0.016 | |
| it_tr_ctr | 0.053 | 0 | 0.013 | |
| seq_cv_invd20_rk | 0.036 | 0.030 | 0.037 | |
| it_tr_coec_z | 0.035 | 0.010 | 0.011 | |
| prew_* (NaN 0.380 → 0.459) | 0.025-0.026 | ≤ 0.002 | ≤ 0.005 | |
| it_hist_all_luo / _et1 | 0.025 / 0.023 | 0 | 0.004 / 0.003 | |
| ctx_nd_{ucent,top5u,knn_u} (NaN 0.014 → 0.003) | 0.016 | | | |

- **Mechanism.** Raw counts drift only through the cold-share change: their PSI within the cold and warm strata is about 0.
- **The `_rk` versions of tied defaults** drift *within* the cold stratum (verified: PSI 0.261 on cold rows). A cold candidate's rank is the average rank of all tied cold candidates, which moves with the feed's cold share.
  - *Inferred, not measured:* on val the fine-tune learns a cold-share-specific meaning of `it_tr_coec_rk`, and on private (41% cold vs 29%) that meaning is off. Proposal 2 would measure it.
- **Out-of-range check.** No live input has more than 0.37% of private rows outside val's [q0.1%, q99.9%] range (max: it_tr_coec_z), so there is no extrapolation problem.
- **NN train-stage inputs.** Alone they give AUC 0.595. Drivers: f4 (the age × coverage interaction), prew availability, and it_hist_all_luo (PSI 0.025).

### ADV-4 (verified): random-like train ≠ exploration val even after time is removed

- **Coverage.** Coverage separates them almost perfectly (`it_hist_all_luo_zero_share` PSI 2.76 / 5.30). This is SHIFT-1 / FR-3: the non-causal LUO popularity for train rows.
- **Intrinsic separation (AUC 0.756).** It comes from H (mean |SHAP| 0.59), hist et1 share (0.56), hist emb-missing share (0.18), f5 and n_prior_feeds.
  - hist et1 share by train day (`diag4.py`) is 0.48-0.50 on random-like train every day Sep 6-19, and 0.51-0.52 on all M ≥ 50 train. On val/test M ≥ 50 it is 0.46/0.47.
  - So this is a population difference: random-like production feeds sit between production and exploration traffic. It is not a time trend.
  - hist emb-missing share does trend down over train days (0.18 → 0.149; val 0.153).
- **Candidate families (train→val50, from §1.2).** popularity 0.80 ≫ prew 0.62 ≈ pre2s 0.61 > f/category 0.57 ≈ M/H 0.57 ≈ ctx 0.56 > embedding proxies 0.53.
  - pre2s is the most train-shifted *non-popularity* family, relevant for nn_final7 (`cvnd_pre2s`).
  - Its SHAP comes from the joint (pre2_fe_s, pre2_fe_z), i.e. the per-feed level/spread of pre2. Feed-mean pre2_fe_s / within-feed sd: train Sep 9-14 5.82 / 0.58, val 5.73 / 0.56, private 5.76 / 0.55.
  - Cold-item pre2_fe_s: train 5.55-5.64, val 5.58, private 5.65 (`diag5.py`).
  - Similarity features drift upward with time: cos_u is 0.456 (train Sep 9-14) → 0.469 (Sep 17-19) → 0.475 (val) → 0.480 (private). This is small (PSI 0.02).

### ADV-5 (verified): importance-weighted val. M-only stays the right level calibrator, and conclusions are stable

Density ratio w = p/(1−p) · n_val/n_target from the OOF feed-level classifiers. Normalised weights are clipped to [0.1, 10] (pre-declared; [0.2, 5] and no-clip are in `weights.json`). Results from `weights.log` and `weights2.log`:

| weighting | ESS | v10 | v11 | f5 fine-tuned | f5 train-only | cd12 mix | v11−v10 | ft−to | cd12−v11 |
|---|---|---|---|---|---|---|---|---|---|
| uniform (val) | 14,423 | 0.39399 | 0.39793 | 0.39784 | 0.39712 | 0.39873 | +0.00394 ± 0.00090 | +0.00072 ± 0.00065 | +0.00080 ± 0.00043 |
| **M-only private (calib.py)** | 14,374 | 0.38856 | **0.39244** | 0.39240 | 0.39163 | 0.39328 | +0.00388 ± 0.00089 | +0.00077 ± 0.00064 | +0.00084 ± 0.00042 |
| adv private, intrinsic | 13,721 | 0.38896 | 0.39309 | 0.39311 | 0.39225 | 0.39399 | +0.00412 ± 0.00090 | +0.00086 ± 0.00067 | +0.00090 ± 0.00043 |
| adv private, intrinsic + hour | 12,279 | | 0.39452 | | | | +0.00491 ± 0.00100 | +0.00074 ± 0.00070 | +0.00083 ± 0.00043 |
| M × hour (4 bins) private | 13,268 | 0.38749 | 0.39186 | | | | | | |
| hour-4 only, private | | | | | | | +0.00436 ± 0.00093 | +0.00044 ± 0.00068 | |
| adv private, notime (coverage) | **2,880** | 0.43717 | **0.44221** | 0.44069 | 0.44139 | 0.44344 | +0.00504 ± 0.00231 | −0.00070 ± 0.00135 | +0.00122 ± 0.00122 |
| adv private, all (with time) | 11,161 | 0.40260 | 0.40736 | | | | +0.00477 ± 0.00105 | +0.00061 ± 0.00076 | +0.00091 ± 0.00052 |
| val Thu only | 6,878 | 0.39442 | 0.39772 | | | | +0.00330 ± 0.00130 | +0.00119 ± 0.00094 | +0.00069 ± 0.00064 |
| M-only public | 14,297 | 0.38592 | 0.38978 | 0.38979 | 0.38895 | 0.39066 | +0.00386 ± 0.00088 | +0.00084 ± 0.00064 | +0.00088 ± 0.00041 |
| adv public, intrinsic | 13,323 | 0.38501 | 0.38908 | 0.38922 | 0.38803 | 0.39001 | +0.00407 ± 0.00090 | +0.00119 ± 0.00068 | +0.00093 ± 0.00044 |
| M × hour (4 bins) public | 12,735 | 0.38632 | 0.39017 | | | | | | |

- **Procedure noise (label-permutation null; 5 permutations of the user-group labels, null AUC 0.492-0.505).**
  - Level shifts of v11 under null weights: sd 0.0022 (private) and 0.0017 (public), mean +0.0010.
  - v11−v10 under null weights: sd 0.00015-0.00022.
  - The intrinsic level differs from M-only by +0.00065 (private) and −0.0007 (public), which is inside the null spread.
  - The intrinsic + hour level (+0.0021 vs M-only on private) has the opposite sign from the hour-bucket term (M × hour: −0.0006 for v11, −0.0011 for v10), so it is noise too.
- **Conclusion: adversarial weights are too noisy at 14K feeds for levels, but fine for deltas.**
  - v11 > v10 holds under every weighting (+0.0033 to +0.0050). v11's edge is largest on evening feeds, which private over-weights: 18-24 h +0.0075 ± 0.0018 vs 12-18 h +0.0008 ± 0.0015 (`diag6.py`).
  - The fine-tuned vs train-only order holds everywhere except the coverage weights (−0.0007 ± 0.0014, n.s.), and those weights are invalid.
- **Why the coverage (notime) weights are invalid.**
  - They put the "private" level at 0.442, 0.05 above val.
  - The classifier's p(private) has Spearman −0.28 with M: cold shares are noisier in small feeds, so small val feeds look "private-like".
  - Small feeds have high NDCG, so the weights inflate the level. This is the SHIFT-4 mechanism (cold share confounded with M, user 0 and short history).
  - Staleness rises for every test feed; it is not a covariate shift that val feeds can be reweighted to.
- **Public LB check (single point).** v10's actual public is 0.38786.
  - M-only predicted 0.38592 (residual +0.0019), M × hour 0.38632 (+0.0015), intrinsic adversarial 0.38501 (+0.0029).
  - The single point does not favour the adversarial weights. The prediction-error SD is 0.0033.

### ADV-6 (verified): per-feed v11 NDCG vs the "looks like private" score

| score | ρ(p, v11 NDCG) | ρ(p, v11 lift) | ρ(p, M-residualised lift) | ρ(p, v11−v10) | ρ(p, ft−to) | ρ(p, M) |
|---|---|---|---|---|---|---|
| intrinsic p(private) | −0.116 | −0.006 | +0.025 | −0.004 | +0.004 | +0.214 |
| notime p(private) | +0.164 | +0.015 | −0.023 | +0.006 | −0.012 | −0.278 |

- Lift = NDCG − mean NDCG of 20 random permutations of the same feed.
- SE of ρ ≈ 0.008. The raw correlations are M effects.
- Decile tables in `weights.log`:
  - Intrinsic p deciles: v11 lift is 0.10-0.13 in every decile; v11−v10 is positive in all 10 (+0.0008 to +0.0068).
  - Notime p deciles: lift rises 0.098 → 0.123, following falling M (124 → 66). The Thu share rises 0.29 → 0.80.

### ADV-7 (verified): public is a good proxy for private; the difference is ~1 day of staleness plus the hour mix

- **Intrinsic AUC 0.535.** Coverage-inclusive AUC is 0.779 (cold_item_share, seq_cv_pop_zero_share, it_hist_luo_zero_share, days_since_prior); candidate level 0.592.
- **Weights.** M-only public and private weights have Spearman 0.90; the notime adversarial weights 0.88. All deltas agree within 0.0001-0.0003 (ADV-5 table).
- **Label-free agreement on test (`agree.log`; within-feed Spearman):**

  | segment | ft5~to5 | v11~v10 | v11~cd12 | cold z (ft5 / to5) | ft − to gap in cold z |
  |---|---|---|---|---|---|
  | val Tue | 0.9834 | 0.9544 | 0.9931 | −0.400 / −0.432 | +0.032 |
  | val Thu | 0.9835 | 0.9537 | 0.9932 | −0.325 / −0.364 | +0.039 |
  | public | 0.9847 | 0.9551 | 0.9934 | −0.274 / −0.317 | +0.043 |
  | private | 0.9847 | 0.9549 | 0.9933 | −0.252 / −0.295 | +0.043 |

  - The models disagree no more on private than on public or val.
  - The fine-tune lifts cold candidates by the same amount on public and private, slightly more than on val. This is consistent with SHIFT-3's direction, and small.
- **Resolution of the public LB.** The paired SE of a v11−v10-type delta on public is ≈ 0.0009 · √(14,423 / 8,562) ≈ 0.0012. So the public LB cannot rank candidates closer than about 0.0025, consistent with the frozen rule "never choose between candidates within 0.003 on public".

### ADV-8 (verified, trap): per-feed means of within-feed z-scores fingerprint the split

- Feed means of pre_fe_z, pre_new_z, pre2_fe_z, pre2_new_z and prew_fe_z are 0 up to float32 rounding (|mean| ~2e-7). Yet these 5 columns alone separate val from private with **AUC 0.997** (`diag1.py`); with them included, the "intrinsic" AUC was 0.94.
- The per-feed sd is identical across splits (0.99445 val / 0.99446 test; `diag2.py`). So the z construction matches, and the rounding pattern is a harmless numerical fingerprint (no model sees 1e-7 differences).
- They are excluded from all reported runs (`feed_adv.py` ZM list). **Any future AV must drop aggregates that are constant by construction.**

### ADV-9 (verified): reweighting the train holdout toward val does not help

- `holdout_w.log`: 59 models with both holdout and train-only val predictions.

  | holdout | ESS | Spearman with val | sign agreement on the 1,236 pairs with \|Δval\| > 2 SE |
  |---|---|---|---|
  | unweighted | 30,000 | 0.975 | 0.997 |
  | intrinsic density-ratio weights | 15,468 | 0.961 | 0.988 |

- The holdout already tracks val rankings. The remaining val/holdout disagreements come from coverage (P3) and the M < 50 blind spot, which weighting cannot fix.

---------------------------------------------------------------------------------------------------------------------

## 3. Checked and correct

- **Base numbers reproduce.**
  - v10 val 0.39399 and v11 val 0.39793.
  - calib.py M-only v10 0.38592 / 0.38856 and v11 0.38978 / 0.39244 (CURRENT_STATUS 3j: 0.3898 / 0.3924).
  - Public = ts < 1790497920 (8,562 feeds).
- **Holdout reconstruction** (rng 12345 over p_random > 0.9 & M ≥ 50) matches preds/holdout/holdout_meta.npy (assert in holdout_w.py).
- **Val↔test intrinsic covariates match.** M PSI 0.003, H 0.002, user 0 0.0004, cold user 0.002, category mix (top-15 shares PSI ≤ 0.007), history event/source shares ≤ 0.004, f1..f5 feed stats ≤ 0.006, user-embedding PCs ≤ 0.003.
- **pre_*_z are built identically on val and test** (ADV-8). pre2_id_contrib (fixed) and pre_id_contrib_v2 show no parity drift (PSI 0.012, from the cold-share change only).
- **Fine-tune extras are standardised with val statistics and reused for test** (nn/finetune.py:180-181), as intended.
- **NN train-stage inputs (cvnd, prew, pre2s, f1-5, embedding proxies) are stable val→private** apart from coverage: candidate AUC 0.595. Among non-coverage inputs, the largest PSI is 0.016 (ctx_nd_ucent / top5u / knn_u, from their NaN share; see the user-0 bullet); every other one is ≤ 0.0033 (ctx_hn_rrw 0.0033, M 0.0028, pre2_hmax32 0.0015).
- **No live input puts more than 0.4% of private rows outside val's 0.1-99.9% range**, so the fine-tune does not extrapolate.
- **The user-0 ctx fix leaves a small footprint.** ctx_nd_{ucent,top5u,knn_u} NaN share is 1.4% on val vs 0.3% on private (PSI 0.016). It follows the user-embedding all-zero rate (1.42% vs 0.33%, known), not a construction difference.
- **f4 is stable marginally** (PSI 0.0004). Its drift is the age × coverage interaction (ADV-1).
- **The intrinsic val→private shift is no larger than the Tue→Thu shift within val** (0.541 vs 0.526), so val's two days already span the composition variation that matters.

---------------------------------------------------------------------------------------------------------------------

## 4. Recommendations

**Features (question 5).**

- **Drop or replace in the fine-tune:**
  - `pre_id_contrib` → `pre_id_contrib_v2` (ADV-2);
  - `it_tr_coec_rk` (cold-share-dependent ties; PSI 0.26 on cold rows);
  - `it_cand_cnt_train`, `it_tr_coec`, `it_tr_ctr`, `it_tr_coec_z`;
  - `hour` (time-identifying, and val has 9 feeds at 20 h vs 1,475 on test).
  
  This is the W3 drop list, so W3 is the right bundle. Val cannot price the private cost of coverage-drifting extras, because the fine-tune is trained and scored in val's coverage regime. So the W3 rule (OOF delta ≥ −0.0003) is the correct asymmetric bar: accept small val losses for private safety.
- **NN train stage:** nothing new to drop. The drift is the known SHIFT-1/FR-3 popularity coverage (5 causal rebuilds already failed) and prew availability.
  - For nn_final7, pre2s is the second most train-shifted family (0.61). Watch its Tue-vs-Thu split and the emulated-coverage val.
  - Use within-feed z/rank channels rather than the raw level if the raw one turns out to carry the train/val gap (FR-1 already moved to `_s` copies).

**Yardsticks.**

- **Do not adopt an adversarially weighted val as a second acceptance yardstick.**
  - With intrinsic covariates it equals M-only within noise.
  - With coverage covariates it is biased (ESS 2.9K, level +0.05).
  - Keep M-only calib for levels with an explicit hour term (M × hour-4). Report the intrinsic-weighted *delta* only as a robustness column (null sd 0.0002).
  - For staleness, the right second yardstick is the coverage-*emulated* val (SHIFT-3 / review_shift Idea 5, `trunc17` / `notest` feature files), not reweighting.
- **Expected val→private gap:** about −0.006 for v11; private 0.392 ± 0.003 (CURRENT_STATUS estimate stands).

---------------------------------------------------------------------------------------------------------------------

## 5. Ranked proposals

1. **W3a: id_contrib-only parity swap**, only if W3/W3b fail their pre-declared rule.
   - Spec: GPU N, n2c_pre with pre_id_contrib → pre_id_contrib_v2, 6 nn_final5 members × 1 fta fine-tune; about 40 GPU-min (within the 3-process cap).
   - Acceptance: the same as W3 (6-member OOF delta ≥ −0.0003, no slice < −2 SE).
   - Expected: 0 ± 0.0003 on val. It removes the largest val/test construction asymmetry in v11.
2. **Staleness-emulated OOF for fine-tune feature decisions.**
   - Copy nn/finetune.py to nn_x/solvers and audit its output paths. After each fold, also predict the held-out fold with substituted extras:
     - it_hist_*_luo and seq_cv_* from the review_shift `out/cf/both_*` files;
     - it_tr_* and it_cand_cnt_train recomputed on CPU from train Sep 6-14 only (src/features.py logic).
   - Score n2c_pre vs n2c_pre_v2 on the 6 nn_final5 members (GPU N, about 85 min).
   - Pre-declare: if n2c_pre loses > 2 SE more than v2 under emulation, prefer v2 (or v3) even when the val-only delta is in [−0.0006, −0.0003].
3. **Coverage-invariant `_rk` for tied defaults** (CPU feature work, then an FT test via proposal 2).
   - Build it_tr_coec_rk_w (rank among warm candidates only, NaN for cold) plus a cold flag, under new names.
4. **Reporting conventions** (zero cost):
   - M × hour for levels;
   - an intrinsic-weighted delta column;
   - no coverage reweighting and no holdout reweighting;
   - drop constant-by-construction aggregates from any AV.
