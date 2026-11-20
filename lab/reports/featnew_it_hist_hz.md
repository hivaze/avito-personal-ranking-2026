# featnew: family `it_hist_hz` (age-gated, density-matched LUO popularity for train rows): REJECT

Date 2026-11-11. CPU only (CUDA_VISIBLE_DEVICES="", at most 16 threads, one job at a time).
- Code: `feat_new/it_hist_hz.py` (builder), `feat_new/it_hist_hz_gate.py` (train-fit gate, arms Rn/R0/R2), `feat_new/it_hist_hz_gate_g1.py` (gate_pre2-g1-style robustness gate).
- Shared definitions are imported from `feat_new/hz_common.py` (md5 a0ab1444…; written by the hzcv agent): the per-feed lag L, the 1/37 sparse sample, the train cutoff and the holdout (L = 0).
- Scratch scripts and logs: `/tmp/scratchpad/featrev/it_hist_hz/` (detect.py, decomp.py, resid_nn.py, eval_family_new.py, reweight*.py and their logs). `*_v1_ownhash*` files are from a first build with my own hash; section 5.4 explains them.

## 0. Verdict
**Reject. No pre-declared screen passes.**

| screen (pre-declared) | delta (SE) | rule | result |
|---|---|---|---|
| (c) train-fit LightGBM gate, R2 − R0, val, 2 seeds (primary run) | +0.00111 (0.00075), z 1.49 | val > 2 SE | FAIL |
| (c) same, deterministic re-run, 4 seeds (noise check) | −0.00027 (0.00058) | | FAIL |
| (c) gate_pre2-g1-style robustness gate, G2 − G0, val, 2 seeds | −0.00067 (0.00076) | | FAIL |
| (b) residual over the NN mix (nn_final4 OOF + train-only), GBM | −0.00061 (0.00032); noise control −0.00020 (0.00038) | > 2 SE | FAIL |
| (a) val-only user-fold LightGBM (`eval_family` copy, 3 seeds) | +0.00077 (0.00068); per seed +0.0009 / −0.0008 / +0.0022 | > 2 SE | FAIL |

- **Leak and parity detectors: no flags** (section 3). The construction is leak-free and does what it was designed to do:
  - the zero-popularity label lift on train random-like rows goes from 0.564 (as built) to 0.793, vs val 0.798;
  - within young listings, the count-to-lift curve now matches val (section 3.3).
- **Why it does not help a GBDT:** it is a cold/warm trade that nets to about zero at val's mix.
  - Feeds whose positives are all young listings gain +0.041 … +0.043 (SE 0.0023-0.0027).
  - Feeds with no young positive lose −0.005 … −0.007 (SE 0.0006-0.0008).
  - Reweighted to the private feed mix (more young candidates), it is +0.0008 ± 0.0009. That is diagnostic only and not significant.
- Screens (a) and (b) are degenerate by construction. Val/test values are byte-identical to the as-built `it_hist_*_luo`, which the NN and the val-only base already contain. Only gate (c) can see the change.
- **Recommendation:**
  - Do not spend a paired GPU NN run on it under the current rule.
  - Keep the files in `data/feats_new/` for reference.
  - This family supersedes W7 (×37 weighting) and answers it: density matching fixes the train/val semantics of the popularity input, but at the GBDT level it shows no net val gain.

## 1. Construction (`feat_new/it_hist_hz.py`, 32 s wall, 16 threads)
**Train row** (item j, feed time t, user u):
- t_post = t − age(f4), using the life_ knots (`hz_common.age_hours`). This is the identical arithmetic to `feat_new/research_expo.py` src_expo_h.
- c = day_cut(t) − L·86400, with day_cut = 00:00 MSK of the latest full train day ≤ own day. L ∈ {0, 4, 8} comes from `hz_common.lag_raw(feed_id)`, with shares 0.340/0.330/0.330. The 30K nn/train.py holdout feeds get L = 0.
- E = (c − t_post) in hours. Capped f4 (≥ 63/64) counts as old.
- **E > 0** (the listing existed before the cutoff): `it_hist_hz_X = it_hist_X_luo` as built.
- **E ≤ 0** (the listing was created at or after the cutoff): the number of occurrences of j with event type X in the histories of snapshots S, where S is either a val/test feed or a train feed with `hz_common.sparse_flag(feed_id)`. That flag selects 85,196 train feeds (1/36.99).
  - Snapshots S of the same non-zero user with ts(S) > t are excluded (the as-built LUO rule).
  - User 0 keeps the as-built handling: nothing is removed for user-0 rows, and user-0 snapshots count like any other user's.

**Val/test rows:** `it_hist_X_luo` as built. Byte-identity was checked for all 10 files.

**Files** (float32, aligned with `data/flat/<split>/cand_*`, no NaN): `data/feats_new/{train,val,test}/it_hist_hz_{all,et1,et2,et3,et4}.npy`.

**Scale of the change:**
- Young (E ≤ 0) rows are 22.39% of train rows (55.2M), 20.07% of val and 30.24% of test.
- Among young train rows, hz < luo for 74.6% of rows, so 16.7% of all train rows change.
- The young-row zero share goes from 0.243 to 0.678 (build.log).

## 2. Correctness and leak checks inside the builder (build.log)
| check | result |
|---|---|
| V1: val/test young rows recounted by this code with c = Sep 20 vs as-built luo | equal on 278,941/278,944 val and 878,495/878,502 test rows; the rest differ by 1 (t_post boundary rounding) |
| V2: train young rows, as-built luo − hz, per type | min 0 for all 5 types: every counted occurrence is an as-built LUO occurrence (subset) |
| V3: 1M random young train rows recounted WITHOUT thinning | exact match with as-built `it_hist_*_luo` on 100% of rows, all 5 types, so the LUO rule here equals `src/features_luo.py` |
| val/test output vs `data/feats/<split>/it_hist_*_luo.npy` | byte-identical (float32), all 10 files |

- No labels are read anywhere in the builder.
- The gate uses only the row's own f4, the feed timestamp, feed_id hashes and fixed constants.
- There is no split-specific missingness. Counts exist for every row of every split, and the same counting code reproduces the val/test values (V1).

## 3. Pre-declared leak / parity detectors (`detect.py`, `detect.log`)
Train random-like = all feeds with p_random > 0.5 (515,023 feeds, 29.7M rows, P(y) 0.0251), compared with all val (1.39M rows, P(y) 0.0158).

### 3.1 Detector table
| feature | AUC tr / val (diff) | zero share tr / val | lift(y\|0) tr / val (ratio) | z(pos) tr / val | skew vs maxsim32 | flag |
|---|---|---|---|---|---|---|
| it_hist_hz_all | 0.5336 / 0.5561 (−0.0226) | 0.278 / 0.379 | 0.793 / 0.798 (0.994) | +0.064 / +0.126 | +0.0194 | no |
| it_hist_hz_et1 | 0.5351 / 0.5571 (−0.0221) | 0.307 / 0.411 | 0.802 / 0.806 (0.994) | +0.069 / +0.132 | +0.0198 | no |
| it_hist_hz_et2 | 0.5191 / 0.5331 (−0.0140) | 0.597 / 0.689 | 0.929 / 0.929 (1.000) | +0.041 / +0.091 | +0.0279 | no |
| it_hist_hz_et3 | 0.5181 / 0.5308 (−0.0126) | 0.625 / 0.713 | 0.925 / 0.935 (0.990) | +0.043 / +0.094 | +0.0293 | no |
| it_hist_hz_et4 | 0.5173 / 0.5278 (−0.0105) | 0.682 / 0.749 | 0.940 / 0.943 (0.997) | +0.036 / +0.079 | +0.0314 | no |
| *ref.* it_hist_all_luo (as built) | 0.5464 / 0.5561 (−0.0097) | 0.194 / 0.379 | 0.564 / 0.798 (**0.707**) | +0.084 / +0.126 | +0.0322 | no |
| *control* maxsim32 | 0.6062 / 0.6481 (−0.0419) | – | – | – | – | (population) |

- Flag rules: |AUC diff| > 0.03, or zero-lift ratio outside [1/1.5, 1.5]. No hz feature is flagged.
- The p > 0.5 train population is itself less separable than val; maxsim32 loses 0.042 AUC.
- Relative to that control, hz moves *toward* val semantics: skew falls from +0.032 to +0.019.
- On M ≥ 50 feeds (holdout-like population): hz_all AUC is 0.5625 on train vs 0.5522 on val, as-built 0.5661 vs 0.5522, and maxsim32 0.6370 vs 0.6516.
- Label-conditional z(neg) is −0.002 everywhere.

### 3.2 PREP s14 parity (pre-declared targets; same rows as `featrev/prep/out/segments.npz`)
| segment | zero share young / mid / old | P(pos \| zero, tercile) / P(pos \| tercile): young / mid / old |
|---|---|---|
| train random-like, as built | 0.364 / 0.160 / 0.147 | 0.62 / 0.73 / 0.92 |
| **train random-like, hz** | **0.567** / 0.161 / 0.147 | **0.76** / 0.74 / 0.92 |
| val (target) | 0.771 / 0.219 / 0.187 | 0.88 / 0.90 / 0.91 |
| private (target) | 0.912 / 0.298 / 0.204 | – |

- **Young:** both statistics move about halfway to val.
- **Mid:** unchanged. The mid tercile (f4 1/3-2/3 = 8-100 days old) is almost never young under the gate, which reaches at most L + 1 ≤ 9 days.
- **Old:** unchanged, by construction.

### 3.3 Where the young-row signal comes from (`decomp.py`, `decomp.log`)
Young train random-like rows (6.10M) vs val young rows (0.28M). Lifts are relative to P(y | young) of the same split.

| | train random-like | val |
|---|---|---|
| hz > 0 share | 0.239 | 0.110 |
| lift hz = 0 / hz > 0 | 0.786 / **1.682** | 0.914 / **1.695** |
| lift count 1 / 2-3 / ≥ 4 | 1.455 / 1.686 / 2.071 | 1.576 / 1.750 / 2.046 |
| only val/test snapshots (VT) | share 0.082, lift 1.53 | share 0.110, lift 1.70 |
| only s=1 train snapshots before t / after t (other users) | 0.011 lift 0.99 / 0.075 lift 1.66 | – |
| same-user earlier occurrences | < 0.005% of rows | 0 |
| user-0 rows with hz > 0 | share 0.012, lift 1.27 | share 0.007, lift 1.60 |

- **This is ordinary popularity with val-matched calibration, not a label channel.** The count-to-lift mapping matches val within about 0.1 at every count level.
- The thinned future other-user part (TA) behaves like the val/test part (VT).
- User-0 rows are *less* predictive on train than on val, which is the opposite of a pooled-user leak signature.
- For comparison, as-built LUO on young rows with L = 4 has lift 0.40 / 1.31 relative to P(y | young) (D3 in `detect.log`).
- Remaining differences:
  - The young zero share is 0.76 on train vs 0.89 on val. A train young listing's sparse window covers every day up to Sep 29, while val/test snapshots exist only on 6 of the 10 days after Sep 20.
  - The base rate differs: P(y | young)/P(y) is 1.19 on train vs 0.68 on val. This is a population difference that the feature does not touch.

## 4. Screens on val values (a, b): degenerate by construction
- **(a)** `eval_family_new.py` is a copy of `src/eval_family.py` whose only change is that `data/feats_new` takes precedence.
  - Command: `--names it_hist_hz_all,…,it_hist_hz_et4 --tag it_hist_hz --threads 16`, 3 seeds, frozen base OOF cache.
  - Result: base 0.36340 → 0.36417, delta **+0.00077 ± 0.00068**, per seed +0.00092 / −0.00080 / +0.00220.
  - Seed 2 is the known low base draw, which inflates the 3-seed delta by about +0.0008 (CURRENT_STATUS 3c).
  - What this measures: adding the as-built LUO counts to a base that already has the non-LUO `it_hist_*`. It says nothing about the train-row change.
  - Appended to `reports/feature_evals.jsonl` and to scratch `eval_family.jsonl`.
- **(b)** `resid_nn.py`: folds exactly as nn/finetune.py:180-183 (GroupKFold(5) by feed user_id, no shuffle).
  - b = z(0.5 z(nn_final4_val_oof) + 0.5 z(nn_final4_trainonly_val)) = 0.39401; GBM control on [b] = 0.39370.
  - [b + 5 hz counts]: **−0.00061 ± 0.00032**. With z(hz_all) added: −0.00059 ± 0.00034. Noise control (4 N(0,1) columns): −0.00020 ± 0.00038.
  - The NN already consumes these exact val values.

## 5. Train-fit LightGBM gate (c): the only screen that sees the change
**Harness** (`feat_new/it_hist_hz_gate.py`): a copy of review LEAKAGE-7 `lgb_causal.py`, the same harness as the hzcv agent's `hz_gate.py`. Its R0 is identical to hzcv's R0.
- Training rows: 50K random-like non-holdout train feeds, the same list as review/leakage `train_sample_feeds.npy` (verified identical); 7.35M rows.
- Model: eval_family PARAMS, 350 rounds, 16 threads.

**Arms:**
- Rn = BASE (49 features).
- R0 = BASE + `it_hist_{all,et1..4}_luo` + within-feed rank/z of `it_hist_all_luo`.
- R2 = BASE + `it_hist_hz_*` + rank/z of `it_hist_hz_all`.
  - R2 changes 12.75% of training rows and 3.2% of holdout rows (holdout L = 0); val rows are identical.
  - The name collides with hzcv's capped-lag "R2" in `hz_gate.py`. Mine lives in a separate script, and `hz_gate.py` was not touched.

**Eval sets:**
- all val;
- the 30K holdout;
- stale-val trunc17 (review SHIFT-3: val `it_hist_*_luo` and `seq_cv_*` with train Sep 17-19 removed; diagnostic only).

### 5.1 Results (val / holdout / stale-val; paired per-feed delta, SE)
| run | R0 val | R2 − R0 val | R2 − R0 holdout | R2 − R0 stale-val | per-seed val |
|---|---|---|---|---|---|
| **primary**, shared hz_common hash, 2 seeds (`gate.log`) | 0.37114 | **+0.00111 (0.00075)** | −0.00099 (0.00044) | +0.00090 (0.00076) | +0.00086, +0.00137 |
| deterministic=True, 4 seeds (`gate_det4.log`) | 0.37189 | −0.00027 (0.00058) | −0.00135 (0.00034) | −0.00005 (0.00058) | −0.00031, −0.00014, −0.00131, +0.00066 |
| first build with my own hash (`gate_v1_ownhash.log`), 2 seeds | 0.37200 | −0.00020 (0.00075) | −0.00145 (0.00044) | +0.00100 (0.00076) | −0.00057, +0.00016 |
| g1-style (`gate_g1.log`): 75K p > 0.9 window feeds, 122-feature swap_luo base, 2 seeds | 0.36495 | −0.00067 (0.00076) | window holdout −0.00081 (0.00071) | – | −0.00191, +0.00057 |

- **Run-to-run noise is as large as the effect.** R0 alone scored val 0.37114 / 0.37200 / 0.37226 (lgb_causal) across identical non-deterministic runs, and the deterministic per-seed deltas span −0.0013 … +0.0007.
- The primary run's +0.0011 is therefore within noise. Every run is far below 2 SE.
- Pooled over the 6 seeds on the official files, R2 − R0 is about +0.0002.
- **Reference:** R0 − Rn = +0.00086 (0.00084) in the primary run. As-built popularity itself adds little on val in this harness; lgb_causal measured +0.0022.

### 5.2 Splits (primary run / deterministic run)
| val split | n feeds | R2 − R0 |
|---|---|---|
| Tue / Thu | 7542 / 6881 | +0.0023 (0.0011) / −0.0002 (0.0010); det: +0.0010 / −0.0016 |
| M < 50 / M ≥ 50 | 5893 / 8530 | +0.0008 (0.0014) / +0.0013 (0.0008); det: −0.0019 / +0.0008 |
| positives all young (E ≤ 0) | 1487 | **+0.0408 (0.0027)**; det +0.0433 (0.0023) |
| some young | 1165 | +0.0148 (0.0021); det +0.0143 (0.0015) |
| no young positive | 11771 | **−0.0053 (0.0008)**; det −0.0072 (0.0006) |

- R0 − Rn on all-young-positive feeds is −0.038: as-built popularity costs young-positive feeds 0.038 and pays it back on warm ones.
- R2 removes this cost entirely (R2 − Rn +0.003) and gives back most of the warm gain.

### 5.3 Diagnostics (not acceptance)
- **Holdout.** The holdout delta is negative in every run (−0.0010 … −0.0015; the task says to read this per-feed z as ~2.3x inflated). This is expected and does not measure what matters here:
  - holdout rows with as-built inputs carry the dense future counts, which are real information inside the train domain;
  - REVIEW_SYNTHESIS P3 already notes that the holdout's LUO popularity is inflated by future snapshots.
- **Private-mix reweighting** (`reweight*.log`): val feeds reweighted to the private distribution of (young-candidate share decile × M < 50), using test features only (ESS 5,557).
  - R2 − R0 = +0.00078 (0.00122) in the primary run, +0.00083 (0.00091) deterministic, +0.00048 (0.00120) own hash.
  - Public mix: +0.0006 … +0.0007.
  - The sign is consistent with "mostly on private", but it is not significant.

### 5.4 Shared-hash rebuild
- I first built the files with my own feed_id hashes (L and the sparse sample), because `feat_new/hz_common.py` did not exist when I started.
- The hzcv agent then published it (16:28). I switched the builder to import `hz_common` (lag_raw, sparse_flag, cutoff_per_feed, holdout_feeds), so every hz* family shares L and the sparse sample for each feed. Then I rebuilt and re-ran every train-side check.
- The first-build logs are kept as `*_v1_ownhash*`. All conclusions are the same.

## 6. NN scaling and integration notes (for a possible later run)
- **attach_xt** (nn/data.py:182-201): the `it_hist` prefix gives `log1p(max(x, 0))`, unclipped. These counts are ≥ 0 with no NaN, so no `has:` column is needed.
- **ID columns:** nn/train.py:178 puts them in `xt_idcols` (prefix `it_hist`).
- **Arm preset:** `cvnd` with `it_hist_*_luo` → `it_hist_hz_*` in the same slots, and `XT_REL` "it_hist_all_luo" → "it_hist_hz_all" (nn/train.py:38) for the in-model rank/z. Val/test inputs are then bit-identical to cvnd; only train rows differ.
- **File access:** the NN reads features only through `feat_path` → `data/feats/<split>` (nn/data.py:31-39). Using these files requires hard links `data/feats/{train,val,test}/it_hist_hz_*.npy` → `data/feats_new/...`. I did not create them (data/feats is off-limits for this workflow).
- **Fine-tune branch:** unchanged. Its `it_hist_*_luo` val/test values are the same numbers.
- **prew mask:** if this were ever accepted, recompute the train-row prew NaN mask (`has:prew_fe`, pretrain/make_prew.py: NaN where it_hist_all_luo == 0) from `it_hist_hz_all > 0`. The val/test masks would not change. Not done here.
- **LightGBM:** raw values (as in the gates).

## 7. Reading
- **The mechanism is confirmed.** On train, as-built LUO counts let young listings collect other users' dense future snapshots, so a young listing that is never seen is strongly negative. Under hz this goes away:
  - the zero-lift ratio moves from 0.71 to 0.99;
  - the young-row count-to-lift curve matches val (section 3.3);
  - a train-fit GBDT stops penalising young positives (+0.04 on all-young-positive feeds).
- **But the as-built semantics also carry real warm-item information** that the GBDT uses: −0.005 … −0.007 on feeds without young positives. At val's mix (10% all-young-positive feeds) the two cancel: −0.0003 … +0.0011 across runs, all < 1.5 SE.
- **Compared with the earlier variants:** this design is clearly better than the causal (−0.0018 ± 0.0008) and lagged (−0.0027 ± 0.0008) recounts of LEAKAGE-7. It removes their loss, but it does not turn into a gain.
- **What this implies for private:** test has more young candidates (30% vs 20%), so private would favour hz slightly (+0.0008 ± 0.0009 reweighted). The fine-tune half of the current mix already repairs young-cold calibration (SHIFT-3), which shrinks the room further.
- **Expected value of an NN arm:** 0 to +0.0008, mostly private, below the val MDE of about 0.002 for a seed variant. It should not get a GPU slot under the unified rule.
  - If a free slot appears anyway: run it paired (≥ 2 seeds), judge it on train-only val, and report stale-val and the all-young-positive split.
  - Do not accept on holdout, because the holdout's future dense counts bias it against this change.
