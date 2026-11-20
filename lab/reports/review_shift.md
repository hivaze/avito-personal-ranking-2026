# Critical review: distribution shift, calibration and error analysis

Reviewer dimension: shift / calibration / errors. Date 2026-11-11. CPU only (CUDA_VISIBLE_DEVICES="", 12 threads). I changed nothing in the project. Scripts and logs are in `/tmp/scratchpad/review/shift/` (`s01`..`s17`, `nncpu/`; logs in `out/s*.log`).

**Base used everywhere:** mix = 0.5 z(`nn_final4_val_oof`) + 0.5 z(`nn_final4_trainonly_val`) + 0.1 z(`pre_final_val`), with per-feed z-scores.
- Reproduced values: val **0.39399**, or 0.39401 without pre (pre adds −0.00002 ± 0.00042). calib.py gives public 0.38592 and private 0.38856.
- `subs/sub_v10_nnfinal4mix_pre.parquet` matches this test mix exactly (within-feed Spearman 1.000; 2,904,806 keys, no duplicates).
- "±" values are per-feed SEs; deltas are paired.

**Counterfactual NN runs.** I copied `nn/data.py` and `nn/model.py` into scratch, patched the copies for CPU, and ran the stage-1 checkpoints of the four nn_final4 train-only members on CPU (`v9_e10`, `_s61/_s62/_s63`).
- The CPU run reproduces the stored predictions: v9_e10 gives 0.39149 vs 0.39141 stored, corr 0.999999. The 4-member z-ensemble gives 0.39290 vs 0.39281 for `nn_final4_trainonly`.
- I then swapped in modified input features. These runs are read-only, and the fine-tuned fold models are not saved, so the fine-tuned half cannot be run this way.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **Val→private shift is mostly staleness, not covariates.** Intrinsic covariates barely move: M +4.5%, category JS 0.0002, user-0 share 4.6% → 4.1%, cold-user share 65% → 67%. Item coverage does move:
   - candidate rows unseen in train go from 25.5% (Tue) and 32.5% (Thu) to 36.5 / 38.5 / 40.1 / 41.5% on Sat / Sun / Mon / Tue;
   - among the 10 most recent history items, the share unseen in train goes from 0.20 / 0.26 (Tue / Thu) to 0.31 (public) and 0.35 (private).
2. **The train-stage popularity input `it_hist_*_luo` is not causal for train rows (SHIFT-1).** It counts every snapshot of all splits, so a train row "knows" about other users' future views (dense train histories up to Sep 19).
   - On random-like train rows its zero rate is 17.5-29.7%, vs 37.9% on val, 42.8% on public and 45.6% on private.
   - "Zero popularity" is a much stronger negative signal on train than on val: P(pos | zero) / P(pos) is 0.61-0.68 on train vs 0.80 on val.
   - The fine-tune's main job is repairing exactly this: +0.0140 ± 0.0018 on feeds whose positives are all cold, −0.0035 ± 0.0009 on feeds whose positives are all warm, +0.043 within-feed z for cold items.
   - This is the same mechanism class as the rejected train-stage pre_*.
3. **Val is optimistic for private by about 0.001-0.002 beyond the M-mix (SHIFT-2/3).**
   - Val's `it_hist_*_luo` include test-split snapshots taken 2-7 days *later*. Removing them costs the train-only ensemble **−0.00069 ± 0.00026**.
   - Emulating private-level item coverage costs **−0.0006 ± 0.0006** (coverage-matched emulation) to **−0.0017 ± 0.0006** (pessimistic: train ends Sep 14 *and* no future snapshots). The 3-epoch v8_nd_e3 loses −0.0021 under the pessimistic version.
   - The hour mix (private is evening-heavy) adds about −0.001.
   - Partial offset: the fine-tuned half is worth more on private (next point).
   - Public LB residuals (+0.0003 / +0.0013 / +0.0014 above the M-only prediction) show the net error has so far been within ±0.002.
4. **The fine-tuned half should matter more on private (SHIFT-3).** Its gain over train-only is +0.0004 on val overall but +0.0027 ± 0.0015 larger on Thu than on Tue. Reweighted to the emulated private mix of positive coldness it is **+0.0025 ± 0.0008**, and the equal mix's edge over fine-tune-only disappears (+0.0008 → −0.0003). Keep the fine-tuned half; do not drop it.
5. **Do not reweight val on cold share (SHIFT-4).** Inside val, cold share is confounded with user-0 and short-history feeds, which have higher lift. On test, cold share rises for every feed because of time. Reweighting on cold share therefore predicts private 0.3957 (M × cold) or 0.427 (logistic model with all covariates), which is absurd. The "test-mix" numbers in err_SYNTHESIS (M × cold-share) are inflated by about +0.006-0.007 in level.
6. **Small verified construction issues:**
   - `pre_id_contrib` is built differently on val and test (SHIFT-5). Impact is negligible: +0.00004 ± 0.0003 in a LightGBM proxy.
   - `u_tr_feeds` / `u_tr_ctr` treat user 0 as a user (SHIFT-6).
   - `hour` is a fine-tune input (SHIFT-7).
   - The prew acceptance was 1.6 SE on one seed, though its gain does survive the private-coverage emulation (SHIFT-8).
7. **Where the loss is (SHIFT-9).** Positives unrelated to the history are ranked essentially at random:
   - positives with maxsim32 < 0.7: 18% of positives, 21.6% of the loss, gain ratio 0.02-0.03;
   - positives whose category is not in the history: 6.6% of positives, 8.6% of the loss;
   - 14% of feeds contain only such positives, and the mix scores 0.271 on them, below random (0.286).
   - Cold positives are only mildly worse than warm ones (gain ratio 0.125 vs 0.154).
   - Users who have train feeds are *harder* than new users: lift 0.095 vs 0.108.

---------------------------------------------------------------------------------------------------------------------

## 1. Findings (most severe first)

### SHIFT-1 (high potential, asymmetry verified, impact inferred): `it_hist_*_luo` is non-causal for train rows, which creates a train/val/test popularity-coverage mismatch in the main train-stage input

- **Construction.** `src/features.py:231-262` (`global_item_stats`) counts `hist_item_id` over train + val + test and all snapshots. `src/features_luo.py` removes only the *same* user's later snapshots. A train row on day d therefore counts other users' histories from days d..19 (about 225K feeds per day, dense) plus val/test.
- **Contrast with co-visitation.** `seq_cv_*` is causal (`feats_v2/seq_/feat_covis.py`: train rows of day d use train sources strictly before d).

Zero rate and the conditional positive rate on random-like train feeds (p_random > 0.9, M ≥ 50; 40K-feed sample; `s17_train_cov.py`):

| rows | it_hist_all_luo = 0 | P(pos \| it_hist = 0) / P(pos) | seq_cv_pop = 0 | P(pos \| cv_pop = 0) / P(pos) |
|---|---|---|---|---|
| train Sep 6 | 17.5% | 0.61 | 100% (no source) | 1.00 |
| train Sep 10 | 20.5% | 0.67 | 42.1% | 0.82 |
| train Sep 14 | 22.7% | 0.65 | 37.5% | 0.79 |
| train Sep 19 | 29.7% | 0.68 | 36.7% | 0.78 |
| val | 37.9% | **0.80** | 42.0% | 0.84 |
| public | 42.8% | - | 46.4% | - |
| private | **45.6%** | - | 47.6% | - |

- **Reading.** For train rows, "never in anyone's history" is rare and means truly obscure. On val/test it is common and mostly means "fresh listing". The causal `seq_cv_pop` lines up across splits (ratio 0.78-0.89 on train vs 0.84 on val); `it_hist` does not.
- **The model depends on this coverage** (counterfactual, `s08`/`s09`). I set val coverage to private level (train Sep 17-19 removed, no future snapshots) and re-ran the train-only ensemble:
  - overall −0.0017 ± 0.0006;
  - feeds whose positives are all cold **+0.0133 ± 0.0014**;
  - feeds whose positives are all warm **−0.0058 ± 0.0007**.
  
  The train stage acts as a strong "known popularity = good" prior, and the val fine-tune repairs it: cold items +0.043 z, all-cold-positive feeds +0.0140 ± 0.0018 (`s02`, `s13`).
- **The same flag propagates into prew_\*.** `pretrain/make_prew.py` sets prew to NaN where `it_hist_all_luo == 0`, so `has:prew_fe` on train rows is this non-causal indicator: available for 70-82% of train rows vs 62% on val and 54% on private.
- **Impact.** This is not a label leak (LUO holds). It mis-calibrates cold items, and more so on private (45.6% zero rate vs 37.9% on val). It also probably explains why nn_final3's covis/LUO inputs traded −0.021 on cold-positive feeds for +0.011 on warm ones (err_SYNTHESIS §3.1).
- **Fix to test.** Use a time-consistent `it_hist` for train rows, either a day-cutoff expanding window like `seq_cv`, or `snapshot ts < feed ts` on all splits. Recompute `has:prew_fe` from it. Run a paired 2-seed train-stage test judged on train-only val. Also check the coverage-emulated val (Idea 5), where a gain on private should show first.

### SHIFT-2 (medium, verified, −0.0007): information-horizon asymmetry, because val features see future test snapshots that private feeds mostly do not

- **What happens.** `it_hist_*_luo` for val rows include every test-split snapshot (Sep 26-29), which is 2-7 days in the future. A private feed (Sun 11:32 to Tue) sees on average about 1.2 days of later test snapshots.
- The pretraining agent fixed this for pre_* with val encoders trained without test snapshots (pretrain_report §1). It was never fixed for `it_hist_*`, which also feed the fine-tune branch (`ITEM_CLEAN` in `nn/finetune.py:75-86`).
- **Counterfactual** (`s08_cf_feats.py` → `out/cf/notest_*`, `s09`). Val `it_hist_*_luo` recomputed without test-split snapshots (same-user overlap added back exactly; no negative counts), then the train-only ensemble re-scored:
  - overall **−0.00069 ± 0.00026**: Tue −0.00072 ± 0.00036, Thu −0.00066 ± 0.00038;
  - feeds with a positive that would be newly cold: −0.0029 ± 0.0006.
- **Impact.** Val overstates private by about +0.0007 through this channel on the train-only half, and likely similarly on the fine-tuned half (it uses the same features). It is not a leak, but it is a val-only advantage.
- **Fix.** Either use the "data up to the end of the feed's own split" rule for val (no test snapshots), or build the causal version from SHIFT-1 for all splits.

### SHIFT-3 (medium, verified): staleness, and how the fine-tune's value grows with it

Coverage decay (`s01`/`s03`; cold = never a train candidate nor in any train history):

| | val Tue 22 | val Thu 24 | Sat 26 | Sun 27 | Mon 28 | Tue 29 | public | private |
|---|---|---|---|---|---|---|---|---|
| candidate rows cold | 25.5% | 32.5% | 36.5% | 38.5% | 40.1% | 41.5% | 37.1% | 40.6% |
| not in train candidates | 30.9% | 38.6% | 43.1% | 45.4% | 47.1% | 48.6% | 43.9% | 47.7% |
| last-10 history items cold | 0.200 | 0.264 | 0.309 | 0.327 | 0.344 | 0.363 | 0.313 | 0.347 |
| days since the user's last train feed (median, warm users) | - | - | - | - | - | - | - | val 10.0 / private 15.5 |

- **Growth rate.** About +3.5 pp per day between Tue and Thu, then +1.5-2 pp per day over the test days (concave). History-side freshness grows in step, which reduces co-visitation coverage on the history side too: `seq_cv_invd20` non-zero share is 12.0% / 10.7% on Tue / Thu vs 9.8% on private.
- **Tue vs Thu**, with Thu reweighted to Tue on intrinsic covariates (logistic density ratio; AUC 0.54, ESS 6,716; `s06`):

  | | difference (Thu − Tue) |
  |---|---|
  | mix | +0.0005 ± 0.0041 |
  | train-only | −0.0005 ± 0.0041 |
  | fine-tuned | +0.0022 ± 0.0041 |
  | f1 | −0.0036 ± 0.0035 |
  | pre | −0.0041 ± 0.0038 |

  No staleness is detectable at a 2-day spacing: day-level noise is about ±0.004, and the within-day first-half vs second-half difference of lift is −0.008 ± 0.005. The fine-tune gain, however, is larger on Thu by **+0.0027 ± 0.0015**: Tue −0.0009 ± 0.0010, Thu +0.0018 ± 0.0011.
- **Emulated private coverage** (`s08`). Removing train Sep 17-19 from the val features (`it_hist` minus late counts; `seq_cv` zeroed for items seen only on Sep 17-19 and in no earlier val/test snapshot) gives val a cold share of **0.415**, the private level. `it_hist` zero rate becomes 0.451 (private 0.456) and `seq_cv_pop` zero rate 0.474 (private 0.476).

  | scenario (train-only, 4 members) | Δ val | Tue | Thu |
  |---|---|---|---|
  | trunc17 (coverage-matched) | −0.0006 ± 0.0006 | −0.0015 ± 0.0008 | +0.0004 ± 0.0008 |
  | notest (SHIFT-2) | −0.0007 ± 0.0003 | −0.0007 | −0.0007 |
  | both (pessimistic) | **−0.0017 ± 0.0006** | −0.0023 ± 0.0008 | −0.0010 ± 0.0008 |
  | both, 3-epoch v8_nd_e3 | −0.0021 | | |

  The emulation leaves history-side co-visitation staleness out: `seq_cv_invd20` zero rate reaches 0.890 vs private's 0.902. So "trunc17" is slightly optimistic.
- **Fine-tune value under the emulated private positive-coldness mix** (all-cold / some-cold / all-warm = 0.281 / 0.146 / 0.572, vs val 0.174 / 0.117 / 0.709; `s16`):

  | delta | val | emulated private |
  |---|---|---|
  | oof − tr | +0.0004 ± 0.0007 | **+0.0025 ± 0.0008** |
  | mix − tr | +0.0012 ± 0.0005 | +0.0022 ± 0.0005 |
  | mix − oof | +0.0008 ± 0.0005 | −0.0003 ± 0.0005 |

- **Caveat.** These are first-order composition estimates. The fine-tune's cold re-calibration was learned on val cold items, which are the same kind of item as test cold items (unseen in train, mostly new listings), so it should transfer.

### SHIFT-4 (medium, verified): calibration. M-only is right; cold-share reweighting is invalid, and "test-mix" figures are inflated

Predicted public / private for the mix under several reweighting schemes (`s06`):

| scheme | ESS | public | private | Tue 29 |
|---|---|---|---|---|
| **M buckets (calib.py)** | 14,297 / 14,374 | **0.3859** | **0.3886** | 0.3890 |
| M × hour (4 bins) | 12,735 / 13,268 | 0.3863 | 0.3875 | 0.3880 |
| logistic model, intrinsic covariates only (M, H, user0, cold user, number of categories, et1 share, hour) | 12,784 / 13,239 | 0.3861 | 0.3878 | 0.3880 |
| M × cold share | 8,786 / 5,972 | 0.3902 | 0.3957 | 0.3972 |
| logistic model, all covariates incl. cold share | 7,327 / 3,928 | 0.3999 | **0.4272** | 0.4179 |

- **Why cold-share reweighting fails.** Inside val, high-cold-share feeds are user-0 and short-history feeds with higher lift (`s07`):
  - cold share > 0.6: 17-88% user 0, H 20-147;
  - after controlling log M, the OLS coefficient of NDCG on cold share is +0.050.
  
  On test, cold share rises for *every* feed because of time, so it is not a covariate shift. Reweighting val on it multiplies the weight of user-0 and short-history feeds.
- **Consequence for earlier reports.** err_SYNTHESIS "test-mix" (M × cold-share bucket) reports nn_final3 OOF at 0.38865 test-mix vs 0.38797 val, while the M-only expectation is about −0.005. Level comparisons and any "test-mix" deltas there are biased; for example, the cold-offset "test-mix −0.00079" upweights user-0 and short-H feeds.
- **Hour effect.** It is consistent in sign but weak. Lift residual after M, by hour bucket (Tue / Thu):

  | hours | Tue | Thu |
  |---|---|---|
  | 18-23 h | −0.0099 ± 0.0042 | −0.0035 ± 0.0053 |
  | 12-17 h | +0.0054 ± 0.0043 | +0.0022 ± 0.0043 |

  Private is 36.9% evening vs 25.1% for val; public is 55.6% before noon. M × hour moves public +0.0004 and private −0.0011. The public LB residual has the same sign as the M × hour shift (actual − M-pred = +0.0003 / +0.0013 / +0.0014).
- **Calibration bookkeeping.** calib.py's boundary (ts < 1790497920) and buckets are correct (reproduced exactly). Recommended private estimate for v10:

  | component | amount |
  |---|---|
  | M-only prediction | 0.3886 |
  | hour | −0.0005 to −0.0011 |
  | horizon + staleness, train-only half | −0.0003 to −0.0009 (half of −0.0006 to −0.0017) |
  | fine-tune half gains | about +0.0005 |
  | **total** | **0.386-0.388** |

  None of this changes how models rank against each other.

### SHIFT-5 (low, verified, impact about 0): `pre_id_contrib` (and `pre2_id_contrib`) are built differently on val and test

| | exact zeros | zeros among no-ID rows |
|---|---|---|
| val | 0.02% | val no-ID rows: mean 0.003, sd 0.125 (overall sd 1.13) |
| test | **59.2%** (public 57.9%, private 59.8%) | 99.5% |

- **Cause.** `pretrain/assemble.py` (`build`, the `calib` branch) moment-matches `fe` and `fe_noid` with *different* affine maps (a_fe, b_fe) and (a_noid, b_noid), then sets `id_contrib = fe − noid`. On val, a no-ID row (raw fe == raw noid) gets (a_fe − a_noid)·fe + (b_fe − b_noid) instead of 0.
- **Not affected:** `pre_final`, the `_z` features and `pre_idmax32`.
- **Impact checks:**
  - LightGBM lambdarank proxy (5-fold by user, 2 seeds, z(train-only) + 9 pre_* features), predicting held-out folds with the test-style construction: **+0.00004 ± 0.00031**;
  - the fine-tune moves no-ID warm items by +0.003 z on val vs −0.015 / −0.019 on public / private, mostly explained by the larger cold share pushing warm items down (ID-warm items −0.030 → −0.042).
- **Fix.** Compute `id_contrib` from raw (fe − noid) × a_fe per config, or apply one affine to both. Do this before `pre2_*` enters any model.

### SHIFT-6 (low, verified): user 0 is treated as a user in fine-tune inputs `u_tr_feeds` / `u_tr_ctr`

- User-0 rows get `u_tr_feeds` = 123,904 and the pooled `u_tr_ctr` = 0.0211 on both splits (`s15`). Non-user-0 values: median 0, max 22.
- This breaks the established user-0 rule. In practice it acts as a constant user-0 indicator (log1p(123,904) = 11.7 after the fine-tune's count transform), so the impact is about 0. Set both to NaN/0 for user 0.

### SHIFT-7 (low): `hour` is a fine-tune input

- Val has 9 feeds at 20 h, against 1,475 on test (5.2% of test), because of outages on both val days. Private is evening-heavy.
- `hour` is constant within a feed, so it can only act through interactions learned from about 11.5K training feeds. `attach_extra` also applies log1p to it (min ≥ 0, max > 20).
- Recommend dropping it (pre-declared ablation) or replacing it with a coarse 4-bin version.

### SHIFT-8 (low, protocol): prew acceptance was marginal, but robust to staleness

- nn_report line 441: v12_prew3 − v8_nd_e3 gave val **+0.0011 ± 0.0007 (1.6 SE)** on a single seed; holdout +0.0008 ± 0.0004. The protocol requires > 2 SE over ≥ 2 seeds where applicable.
- I re-ran the pair on CPU (`nncpu/cf_prew.py`), reproducing +0.00116 ± 0.00073. Under the pessimistic private-coverage emulation, with prew masked accordingly and z recomputed, the gain is **+0.00137 ± 0.00073** (difference-in-differences +0.0002 ± 0.0006).
- So prew is not staleness-fragile, but its acceptance still rests on one seed. The pre-declared nn_final5 holdout-chosen member sets (S2 vs S3) are the right safeguard.

### SHIFT-9 (info): error analysis, i.e. where the loss of the mix is

Each positive is assigned its share of 1 − NDCG; these shares sum exactly to the mean of 1 − NDCG, 0.6060 vs random 0.7110 (`s11`). "Gain ratio" is the fraction of the random-ranking loss that the model removes.

| positive segment | share of positives | share of loss | mean pct rank | gain ratio |
|---|---|---|---|---|
| all | 100% | 100% | 0.282 | 0.148 |
| cold item | 22.8% | 23.1% | 0.324 | 0.125 |
| warm item | 77.2% | 76.9% | 0.270 | 0.154 |
| category not in history | 6.6% | 8.6% | 0.503 | **0.043** |
| category last seen > 49 events ago | 13.8% | 17.0% | 0.491 | **0.031** |
| category in the most recent event (cat_min_hpos = 0) | 40.7% | 36.6% | 0.192 | 0.229 |
| maxsim32 < 0.7 | 18.1% | 21.6% | 0.50 | **0.02-0.03** |
| maxsim32 ≥ 0.9 | 31.5% | 26.4% | 0.151 | 0.273 |
| pre_final pct > 0.6 | 19.2% | 22.6% | 0.533 | 0.011 |
| f1 pct > 0.6 | 23.4% | 26.8% | 0.453 | 0.045 |
| it_hist_luo = 0 | 30.3% | 31.1% | 0.334 | 0.106 |
| it_hist_luo > 100 | 7.0% | 6.3% | 0.208 | 0.240 |
| in other users' val histories | 11.0% | 10.8% | 0.239 | 0.194 |

Other facts from the same analysis:
- **History-unrelated positives** (maxsim32 < 0.75 and category last seen more than 9 events ago or absent) are 16.3% of positives. 13.9% of feeds contain *only* such positives. On those feeds:

  | score | NDCG |
  |---|---|
  | mix | 0.271 |
  | random | 0.286 |
  | f1 | 0.231 |
  | it_hist_luo | **0.332** |
  | −maxsim32 | 0.350 |

  This is selection-biased by construction. A feed-level gate is only weakly predictable from observables: logistic AUC 0.64 for "all positives unrelated" (`s12`).
- **Segments by user and history** (`s02`):
  - users with train feeds are harder: NDCG 0.3850, lift 0.0951 vs 0.1084 for new users; 2-4 train feeds give lift 0.0884;
  - H = 256 has the lowest lift (0.0905);
  - user 0 scores well: 0.4018, lift 0.126.
- **Segments by M and hour.** Lift falls with M, from 0.139 (M < 10) to 0.074 (M ≥ 250); normalised gain falls from 0.29 to 0.09. 18-23 h is the weakest hour band (0.3760).
- **Fine-tune deltas by segment** (oof − tr): only the positive-coldness split is clearly significant (+0.0140 / +0.0037 / −0.0035). By M, the largest are +0.0070 ± 0.0032 at M < 10 and −0.0050 ± 0.0024 at M 200-249. By dominant category, 13 cells are tested and none exceeds about 2.1 SE (consistent with noise).
- **Item-level vs impression-level features.** f4 and f5 are item attributes: their within-item sd is 0.016 and 0.10 of the total sd (train, items with ≥ 5 appearances in the first 20M rows). f1-f3 vary by impression (0.65-0.71). Any "price relative to history" feature would have to come from f4/f5 looked up on the history items' train-candidate appearances (untested).

---------------------------------------------------------------------------------------------------------------------

## 2. Checked and correct

- **calib.py:** boundary ts < 1790497920 (public = 8,562 feeds), M-bucket edges, and weight normalisation are correct. Reproduced 0.38592 / 0.38856 exactly. ESS is 14.3K of 14.4K, so the M-only reweighting is statistically cheap and the most defensible scheme.
- **Submission:** `sub_v10` equals 0.5 z(nn_final4_test) + 0.5 z(nn_final4_trainonly_test) + 0.1 z(pre_final_test). All test keys match, with no duplicates.
- **Mix numbers:** val 0.39401 (mix) and 0.39399 (+pre), as stated in CURRENT_STATUS. The fine-tune split by positive coldness reproduces nn_report exactly (+0.0140 / −0.0035).
- **Intrinsic covariates match between val and test.** Standardised mean difference val→private:
  - |SMD| < 0.06 for M, log M, H, user 0, cold user, n_cat, max_cat_share, history structure, et1 share and category mix (JS 0.00016);
  - the user-embedding all-zero flag is 1.42% → 0.33% (known);
  - among the model inputs, only `hour` (SMD 0.28) and the coverage-driven features differ materially;
  - all content, similarity and ctx inputs have |SMD| ≤ 0.06 overall and within cold/warm strata (`s04`).
- **`seq_cv_*` is causal and aligned** across train / val / test in both zero rate and conditional positive rate (`s17`).
- **pre_\* horizon handling** (val encoders trained without test snapshots) is right. `pre_final` and `pre_*_z` are invariant to the moment matching.
- **Leak-flag feeds do not drive results:** the 117 later-user feeds give mix 0.3763 vs 0.3941 on the clean 14,306.
- **Val has no feeds without positives** (min npos = 1, M ≥ 5 in val and test), so the "no positives → 1" rule never fires on val.
- **The history has no timestamps** (raw schema checked). "Time since the last history event" cannot be computed; I used days since the user's last train feed and the share of recent history items that are cold as proxies.
- **CPU replay of the train-only members** matches the stored predictions (corr 0.999999), so the counterfactual deltas are valid paired comparisons.
- **The `it_hist` horizon counterfactual** was built exactly: same-user later overlap added back (149 rows), and no negative counts before clipping.

---------------------------------------------------------------------------------------------------------------------

## 3. Ideas (testable, in priority order)

1. **Causal `it_hist_*` for train rows (SHIFT-1).** For train rows of day d, count only snapshots before d. For val and test, use snapshots before the feed timestamp, or "own split only, no later split". Recompute `has:prew_fe` from it. Run a paired 2-seed train-stage test on train-only val, and report the coverage-emulated val (idea 5) alongside it. Expected: better cold calibration, smaller fine-tune correction, and a larger gain on private than on val.
2. **Keep the fine-tuned half, and consider a pre-declared private-oriented weight.** The fine-tuned half's value grows with staleness: Thu − Tue +0.0027 ± 0.0015, emulated private +0.0025 vs +0.0004 on val. Under the emulated private composition, a single pre-declared coarse choice such as w_ft = 2/3 is justifiable; 0.5 remains the hedge. Do not pick w on val.
3. **Train-stage coverage augmentation done structurally.** For train rows, compute `it_hist` with a random per-feed cutoff age of 3-10 days, so that *fresh* items become zero and the zero rate matches val/test (38-46%). Zeroing random items, as the rejected xt-dropout did, does not achieve this.
4. **Recency-weighted transductive popularity from val/test histories** (legitimate inputs): causal, time-windowed (for example the last 3 days of snapshots before the feed), and normalised per feed-day. A window is required because whole-split coverage is not symmetric: candidates found in other same-split histories are 8.4% on val vs 13.5-14.7% on test, because test has twice as many feeds. This covers the fresh items whose share grows by 2-3.5 pp per day. Test it as a fine-tune feature first (cheap), then in the train stage.
5. **A staleness-robust second yardstick for train-stage changes.** Evaluate every train-only candidate on val *and* on the trunc17 / both-emulated val. This is cheap: 4 members × 25 s on CPU with the scratch `nncpu/` copy, and the feature files are in `out/cf/`. A change that helps val but loses on the emulated val is fragile for private.
6. **Calibration reporting.** Keep M-only for deltas. Report private as M-only − 0.001..0.002, with the hour and staleness terms listed separately. Retire the M × cold "test-mix".
7. **Small fixes:**
   - id_contrib construction before pre2_* is used (SHIFT-5);
   - u_tr_* for user 0 (SHIFT-6);
   - drop or coarsen `hour` in the fine-tune (SHIFT-7);
   - a second seed for prew if GPU time allows (SHIFT-8).
8. **Low-prior items from the error analysis:**
   - an exploration-propensity feed feature (history category entropy, share of candidates in history categories, recency of each candidate category) for the history-unrelated positives (16% of positives, ranked essentially at random);
   - "price relative to history" from the item-level f4/f5 looked up on history items via their train-candidate appearances.
   
   Expect ≤ +0.0005 from each; test only via nested CV or as pre-declared fine-tune features.
