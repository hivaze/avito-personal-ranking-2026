# featnew: family `ixh` (horizon-matched item exposure context from other users' production candidate lists): REJECT

Date 2026-11-11, CPU only (CUDA_VISIBLE_DEVICES="", one 16-thread job at a time). Code: `feat_new/ixh.py`. Scratch scripts and logs: `/tmp/scratchpad/featrev/ixh/`.

## 0. Verdict
**Reject.** Every screen was declared before it ran, and none passes. All point estimates fall between −0.0004 and +0.0010, inside 1.2 SE of zero:

| screen (pre-declared) | delta (SE) | rule | result |
|---|---|---|---|
| (a) val-only user-fold LightGBM (`eval_family` copy, 3 seeds) | +0.00064 (0.00069); per seed −0.0003 / −0.0009 / +0.0031 | > 2 SE | FAIL (0.9 SE) |
| (b) residual over the nn_final4 mix (fine-tune folds), group GBM | +0.00053 (0.00047) | > 2 SE | FAIL (1.1 SE) |
| (b) same, group LIN | −0.00002 (0.00034) | | |
| (b') residual over the nn_final5 mix (current best; informational), GBM / LIN | −0.00036 (0.00045) / +0.00023 (0.00034) | | |
| CPU screen: `hz_gate.py` arm **R3 = R0 + ixh_*** (copy, same R0 / sample / seeds) | val +0.00012 (0.00072); holdout +0.00007 (0.00042); stale-val −0.00109 (0.00073) | val > 2 SE and holdout ≥ 0 | FAIL |
| (c) train-fit gate in the `gate_pre2.py` g1 style (75K random-like feeds, frozen base, 2 seeds) | val +0.00097 (0.00079); NN holdout −0.00088 (0.00052, nominal); H_mix −0.00026 (0.00073) | val > 2 SE | FAIL (1.2 SE) |

Leak and parity:
- **Leak:** none found. The has-later excess is ≤ 0 for all 7 features (§3.3). Brute-force recomputation matches the builder (§1.2). By construction nothing later than the row's own cutoff is read.
- **Pre-declared detector:** fires on 37 ixh feature/population pairs, and fires just as much on the existing reference `tte_it_imp` (12 flags). That reference has the identical construction at L = 0.
  - The cause is the declared population. 68% of the "p_random > 0.5" train feeds have M < 50, with an 8.4% positive rate (val M < 50: 5.2%). On those feeds even f1 and tte_it_imp diverge from val.
  - On M-matched random-like feeds (p_random > 0.9, M ≥ 50), ixh agrees with val (§3.2).
- **Drift:** PSI val→private is 0.068-0.070, which is the about-0.07 the old ctx_ix had (feat_ctx_ l.188).

**Recommendation:** do not spend a paired GPU NN run on ixh. The old v5_cvix +0.0010 val / +0.0002 holdout (one seed, nn_report l.217-219) is consistent with what is seen here: a weak signal that is largely redundant. At L = 0 its count is exactly `tte_it_imp` (§1.2), and it decays with staleness (stale-val −0.0011). The files stay in `data/feats_new/` for reference.

## 1. Construction (`feat_new/ixh.py`)
- For a candidate row with cutoff day k, A = all train candidate rows with the same item_id from train feeds whose MSK day is < k.
  - The cumulative per-item sums are kept in float64 arrays of size 111.4M. Train days are walked in order. Every row with cutoff day k is answered before day k is added (`ixh.py` main loop, `_answer` / `_add_rows`).
  - The own feed can never be in A, because its day is ≥ k.
- **Cutoff, train rows:** c0 = start of the latest full train day (> 1000 feeds) ≤ own day, then c = c0 − L days. L is the **shared lag of `feat_new/hz_common.py`** (`cutoff_per_feed("train", lag=True)`): L ∈ {0, 4, 8} by a fixed feed_id hash, with the 30K `nn/train.py --holdout` feeds forced to L = 0.
  - `train_cutoffs()` asserts that c0 − L from the full-day rule equals hz_common's cutoff for every feed, and that every holdout feed has L = 0.
  - Lag counts: L0 1,070,496 / L4 1,040,518 / L8 1,040,194 feeds.
  - A first build used a placeholder hash, because hz_common did not exist yet. It was rebuilt with hz_common before any screening; `build_oldlag.log` and `det_oldlag.log` are kept for reference only.
- **Cutoff, val/test rows:** k = 2026-09-20 (all train days < Sep 20). The 34 train feeds stamped Sep 20 MSK are excluded for every split alike.
- **rpos:** within-feed pct rank of the production pos, using a copy of `build_ix._feed_pct_rank_asc` (0 = shown first, ties averaged).
- **Files** in `data/feats_new/{train,val,test}/`: float32, aligned with `data/flat/<split>/cand_*` rows, for all 246.7M train, 1.39M val and 2.90M test rows.
  - `ixh_good` = 1 − mean_A rpos
  - `ixh_lpos` = mean_A log1p(pos)
  - `ixh_mlogM` = mean_A log M; `ixh_dlogM` = log M_own − ixh_mlogM
  - `ixh_mf1` = mean_A f1; `ixh_df1` = f1_own − ixh_mf1
  - All of the above are NaN when n = 0.
  - `ixh_ln` = log1p(n)/2, never NaN.
  - `ixh_has` = 1[n > 0], the explicit has-file for the fine-tune branch.
  - `slogM` and the old `*_rk` columns are dropped, as specified.
- **Build:** `CUDA_VISIBLE_DEVICES="" .venv/bin/python feat_new/ixh.py --threads 16`, wall time 30 s (`build.log`).

### 1.2 Verification
- **Brute force** (`verify_ixh.py`, `verify.log`): an independent polars filter over all train rows of the sampled items, with an independent pure-python rpos.
  - Sample: 4,000 random train rows (at their shared-L cutoff), 2,000 val rows and 2,000 test rows.
  - All 8 files have an identical NaN pattern; max abs error ≤ 4.6e-7. **VERIFY PASS.**
- **Equivalence to an existing feature** (`chk_tte.py`): on every L = 0 train row, n == `tte_it_imp` (share 1.000 on the 30K holdout and on the L = 0 random-like feeds).
  - So ixh_ln at L = 0 is log1p(tte_it_imp)/2, which is already in the eval_family / g1 base.
  - On val, ixh_ln and tte_it_imp have identical within-feed AUC (0.5341).

## 2. Parity: NaN shares and drift (`det_ixh.py`, `det.log` / `det_ixh.json`)
NaN share of `ixh_good` on random-like train rows (p_random > 0.5) by L and own day, vs the share of rows never seen as a train candidate:

| rows | Sep 7-10 | Sep 11-14 | Sep 17 | Sep 18 | Sep 19 | all |
|---|---|---|---|---|---|---|
| L = 0 | 0.41 / 0.34 / 0.30 / 0.29 | 0.26 / 0.25 / 0.24 / 0.24 | 0.35 | 0.24 | 0.22 | **0.348** |
| L = 4 | 1.0 (all) | 0.62 / 0.55 / 0.51 / 0.49 | 0.45 | 0.43 | 0.42 | **0.700** |
| L = 8 | 1.0 (all) | 1.0 (all) | 0.61 | 0.58 | 0.55 | **0.883** |
| val / public / private | | | | | | **0.346 / 0.436 / 0.473** (task: 34.6 / 43.6 / 47.3%) |

Sep 6 is NaN for every L.

- **What lines up:** late L = 4 rows (0.42-0.45) match public (0.436). The NN holdout (L = 0) is 0.356 vs val 0.346.
- **What does not:** late L = 8 rows (0.55-0.61) overshoot private (0.473), because their window is only Sep 6-10, against 12 train days for val/test.
- **Coverage cost of the lag mix:** 63.9% of all train rows are NaN (58.5% of the g1 training sample) vs 35-47% on val/test. With the shared lag, the train stage sees ixh mostly missing.
- **PSI** (10 val-quantile bins + NaN bin):
  - val→public 0.035; val→private 0.068-0.070 (`ixh_lpos` 0.0701 is the max); public→private ≤ 0.0063.
  - val→late L0 / L4 train 0.027-0.053; val→late L8 0.23-0.26.
  - The val→private drift is entirely the coverage fall from 65.4% (val) through 56.4% (public) to 52.7% (private).
- **Ranges** (min / max over all rows of all splits): good [0, 1]; lpos [0, 5.704]; mlogM [1.609, 5.704]; dlogM [−4.094, 4.094]; mf1 [0.018, 0.982]; df1 [−0.964, 0.964]; ln [0, 4.573]. All are inside the ±10 clip, as specified.

## 3. Leak / parity detectors

### 3.1 Pre-declared: train random-like (p_random > 0.5, nothing excluded) vs val
- Flags: 37 on ixh, for example ixh_ln AUC 0.500 (train) vs 0.534 (val), and missing-lift 1.198 vs 0.808.
- **The reference `tte_it_imp` gets 12 flags under the same rule** (AUC 0.498 vs 0.534). The f1 reference gets 0 flags but also diverges (0.602 vs 0.623).
- Diagnosis (`chk_tte.py`): 350,905 of the 515,023 feeds in this population have M < 50.
  - Their positive rate is 8.36% (val M < 50: 5.22%). nn/train.py:152 calls the p_random detector "unreliable for short feeds".
  - On them, every item-exposure signal flattens or reverses: tte_it_imp AUC 0.487 vs 0.538 on val M < 50.
  - The flags therefore describe the population, not the feature. The same holds for any train-candidate count feature.

### 3.2 Supplementary (not pre-declared): M-matched populations vs val M ≥ 50

| population | feeds | pos rate | ixh_good AUC (imp / present) | ixh_df1 AUC (imp / present) | ixh_ln AUC | missing-lift | tte_it_imp AUC |
|---|---|---|---|---|---|---|---|
| val M ≥ 50 | 8.5K | 1.18% | 0.525 / 0.502 | 0.562 / 0.583 | 0.532 | 0.819 | 0.532 |
| NN holdout (L = 0) | 30,000 | 1.16% | 0.519 / 0.503 | 0.548 / 0.566 | 0.528 | 0.869 | 0.528 |
| p > 0.9, M ≥ 50, L = 0, Sep 17-19 | 5,598 | 1.15% | 0.525 / 0.502 | 0.568 / 0.577 | 0.539 | 0.782 | 0.539 |
| same, L = 4, Sep 17-19 | 5,576 | 1.16% | 0.522 / 0.500 | 0.551 / 0.578 | 0.529 | 0.835 | 0.534 |
| same, L = 8, Sep 17-19 | 5,539 | 1.13% | 0.522 / 0.505 | 0.537 / 0.576 | 0.524 | 0.847 | 0.533 |

- No flag on the M-matched populations, except ixh_df1 `auc_imp` over all days for L = 4 / 8 (0.525 / 0.510 vs 0.562). Those rows are 70% / 88% NaN (the early days have no window). The present-only AUC is 0.576 for both, against 0.583 on val.
- Label-conditional per-feed z means agree, e.g. ixh_df1 z(pos) +0.274 val M ≥ 50 vs +0.224 to +0.261 on train.

### 3.3 Has-later audit on val (CURRENT_STATUS 2b method; `haslater.log`)
Single-feature NDCG on the 117 val feeds whose user has a later val/test feed, minus all others:

| feature | excess (SE) |
|---|---|
| good | −0.015 (0.015) |
| lpos | −0.004 (0.016) |
| mlogM | −0.004 (0.016) |
| dlogM | −0.022 (0.013) |
| mf1 | −0.014 (0.015) |
| df1 | −0.004 (0.017) |
| ln | −0.018 (0.016) |
| reference: random | −0.011 |
| reference: f1 | −0.009 |

No excess anywhere.

### 3.4 Leak argument (re-checked in code)
- `_answer` reads only cumulative state built from train rows of days < k, where k ≤ own day. No target is loaded anywhere in `ixh.py`.
- pos is the production display position, decided before any interaction.
- Val/test read train only, all of it earlier.
- The only user dimension is implicit: earlier train feeds of the same user may be in A, which is allowed, and no later ones can be. No user-level statistic is formed, so user 0 needs no special case.
- The missingness mechanism ("never a train candidate before the cutoff") is the same on all splits. Only the horizon differs, by design (§2).

## 4. Screens (details)
- **(a) `eval_family_fn.py`:** a copy of src/eval_family.py whose only change is that it reads feats_new when a file is not in data/feats.
  - Result: base 0.36340 → 0.36404, delta **+0.00064 ± 0.00069**.
  - Seed 2 is the known low base draw (CURRENT_STATUS 3c), so the 3-seed mean is optimistic.
  - The result line was appended to reports/feature_evals.jsonl (tag `ixh`, src "featrev_copy(feats_new)").
- **(b) `resid_ixh.py`:** folds are exactly nn/finetune.py:180-183. The base is the within-feed z of 0.5 z(nn_final4_val_oof) + 0.5 z(nn_final4_trainonly_val) (control val 0.39401).
  - Group GBM +0.00053 ± 0.00047 (Tue +0.0004, Thu +0.0007, M < 50 +0.0009, M ≥ 50 +0.0003).
  - Single features (GBM): −0.00013 … +0.00033, all < 1 SE.
  - Noise controls span −0.00063 … +0.00021.
  - On the current nn_final5 mix (0.39793) the group is −0.00036 ± 0.00045.
- **hz_gate R3 (`hz_gate_ixh.py`):** a copy of feat_new/hz_gate.py (md5 9802ee14…) with arms reduced to R0 and R3. R0, the 50K train sample, PARAMS, the seeds and the splits are identical. Stale-val uses ixh rebuilt with c = Sep 18 (`ixh_stale.py`, val NaN share 0.424).
  - R0 val 0.37197 here vs 0.37156 in the sibling's identical-R0 run (featrev/hzcv/gate.log). That is an A/A spread of 0.0004 under load (LightGBM is not bit-exact).
  - R3 − R0:
    - val +0.00012 ± 0.00072 (Tue +0.0006, Thu −0.0004);
    - holdout +0.00007 ± 0.00042 (Sep 17-19 slice +0.00028 ± 0.00079);
    - stale-val −0.00109 ± 0.00073.
  - ixh gain shares are 0.004-0.006 each.
- **(c) `gate_ixh.py`:** 25K random-like (p > 0.9) non-holdout feeds per gate_pre2 window, base `eval_cache/base_features.json` → swap_luo, 2 seeds.
  - val +0.00097 ± 0.00079 (Tue +0.0016, Thu +0.0003).
  - NN-holdout feeds (L = 0, 21,898 in-window): −0.00088 ± 0.00052. That is nominal −1.7 SE, about −0.7 SE after the 2.3x inflation.
  - H_mix −0.00026 ± 0.00073.

## 5. NN scaling (if anyone revisits it)
- **Train stage:** every ixh_* name takes the generic `attach_xt` path (nn/data.py:197-199: not `_z` / `_rk`, not `seq_cv` / `it_hist`). The verified ranges in §2 never reach the ±10 clip.
- Add `has:ixh_good`. Without it, NaN → 0 reads as "always shown last" (good = 0).
- Add ixh_good and ixh_df1 to XT_REL.
- **Fine-tune branch:** `attach_extra` log rule (min ≥ 0 and max > 20) fires for none of them (max 5.7). Use `ixh_has` as the explicit availability column.
- **LightGBM:** raw values with NaN kept.

## 6. Side findings for the workflow
1. **The pre-declared detector population "p_random > 0.5, nothing excluded" is dominated by M < 50 feeds** (68% of feeds; 8.4% positive rate vs val M < 50 5.2%). There, established clean features fail the 0.03 AUC rule (tte_it_imp 0.487 vs 0.538; f1 0.591 vs 0.618).
   - Parity detectors should use p_random > 0.9 & M ≥ 50 (the nn/train.py holdout rule) against val M ≥ 50, or always run a reference feature through the same rule.
   - `feat_new/hz_detect.py` uses the same population.
2. **The shared lag makes 64% of train rows NaN** for any expanding-window train-candidate statistic: L = 8 rows have no window before Sep 15, and L = 4 rows none before Sep 11. Late L = 8 coverage (55-61% NaN) overshoots private (47%).
   - Any hz-family input built on train-candidate history inherits this. A capped lag or a later window start would match better, but no variant was tried here, to stay within ≤ 3 choices.

## 7. Files
- Code: `<repo>/feat_new/ixh.py`.
- Features: `<repo>/data/feats_new/{train,val,test}/ixh_{good,lpos,mlogM,dlogM,mf1,df1,ln,has}.npy`.
- Scratch (`/tmp/scratchpad/featrev/ixh/`):
  - `verify_ixh.py`, `det_ixh.py`, `chk_tte.py`, `eval_family_fn.py`, `resid_ixh.py`, `hz_gate_ixh.py`, `gate_ixh.py`, `ixh_stale.py` (+ `stale/`);
  - logs `build.log`, `verify.log`, `det.log`, `det_ixh.json`, `haslater.log`, `eval_family.log`, `resid.log`, `hz_gate_ixh.log`, `gate_ixh.log`;
  - per-train-feed lags `train_lag.npz`.
