# featnew_hzcv: horizon-matched co-visitation for train rows (lagged dense source, 1/37 sparse window, censoring offsets)

Author: feature-review workflow, family `hzcv`. CPU only (16 threads), no GPU. Code: `feat_new/hz_common.py` (shared
with ixh / it_hist_hz), `feat_new/hzcv.py`, screens `feat_new/hz_{detect,gate,lagcap,eval_family,resid,ae}.py`.
Logs: `scratchpad/featrev/hzcv/*.log` (full path below).

## 0. Verdict: **reject**

The step-4 rule needs a train-fit val delta above 2 SE, or a residual over the NN above 2 SE, with no leak flags. Neither condition is met:

| screen (pre-declared) | delta (SE) | z |
|---|---|---|
| (c) train-fit LightGBM gate, R1 − R0, **val** | **+0.00032 (0.00071)** | +0.45 |
| (c) same, train holdout (all L = 0) | −0.00057 (0.00041) | −1.40 (−0.6 if read as 2.3× inflated) |
| (c) same, stale-val (diagnostic) | +0.00024 (0.00071) | +0.34 |
| (b) residual over the NN mix, all 6 features, GBM / LIN | +0.00023 (0.00041) / −0.00063 (0.00034) | +0.58 / −1.83 |
| (b) residual over the NN mix, the 3 offsets only, GBM / LIN | +0.00020 (0.00037) / −0.00011 (0.00028) | +0.54 / −0.39 |
| (a) eval_family vs the frozen base | +0.00291 (0.00078) | +3.7, but this is the already-used `seq_cv_*` (val values are identical copies) |
| (a') offsets over base + `seq_cv_*` (val-only LightGBM) | −0.00064 (0.00065) | −0.99 |

- **The family's own soft gate** ("val > 0 and no split below −1 SE") is met on the val splits but not on the holdout:
  - val splits: Tue +0.00007 (0.00099), Thu +0.00060 (0.00101), M<50 +0.00108 (0.00136), M≥50 −0.00020 (0.00075);
  - the holdout is at −1.40 SE using face-value per-feed SEs. REVIEW_SYNTHESIS 3k found those SEs calibrated, so the soft gate is borderline at best.
- **The mechanism did not move.** Val actual/expected positives on cv = 0 & censored rows is 1.130 for R0 and 1.123 for R1 (±0.018).
- **The density-safe variant also gained nothing.** The capped-lag variant R2 gives val −0.00001 (0.00068).
- **finetune_only does not apply.** On val/test the family is bit-identical to `seq_cv_*` (already in the n2c_pre fine-tune branch) plus `src_expo_*` (already built). The new val-stage content, the 3 offsets, adds −0.0006 ± 0.0006 over base + seq_cv and +0.0002 ± 0.0004 over the NN.
- **Recommendation:** do not spend a joint paired NN run on hzcv in this form. Section 7 gives the reason, which is structural.

## 1. What was built (all splits, all candidate rows, float32, never NaN)

`data/feats_new/{train,val,test}/`:
- `seq_cvlag_invd20`, `seq_cvlag_pop`, `seq_cvlag_norm`;
- `hzcv_expo_slog`, `hzcv_censored`, `hzcv_stale_d`;
- `train/hz_lag.npy` (L in days, row-aligned).

**Shared helpers** (`feat_new/hz_common.py`):
- Lag L = [0, 4, 8][((feed_id·0x9E3779B97F4A7C15 mod 2⁶⁴) >> 40) % 3].
  - The 30K nn/train.py holdout feeds are forced to L = 0. `holdout_feeds()` matches the review's `common_rows.holdout_feeds()` exactly (verified equal).
  - Result: 1,070,496 / 1,040,518 / 1,040,194 train feeds at L = 0 / 4 / 8, and all holdout feeds have L = 0.
- Sparse flag s = ((feed_id·0xC2B2AE3D27D4EB4F) >> 40) % 37 == 0. It selects 85,196 train feeds (2.70%, about 7.1K per day).
- Cutoff c:
  - train: the `research_expo.cutoff_per_feed` rule (verified equal at L = 0) minus 86400·L;
  - val/test: 1789851600.
- t_post = t − 3600·age_hours(f4), using the life_ knots.

**Co-visitation**
- **Train rows:** the unchanged `feats_v2/seq_/feat_covis.py` `Index` + `_covis` kernel, imported rather than copied, over `[("train", source_fidx(day(c)), False), ("train", {s = 1, ts ≥ c}, True)]`.
  - Cutoffs at or before Sep 6 have an empty dense part and an identical sparse part, so they share one index. That leaves 13 indices (`train.log`).
  - Each feed is queried once, with its own L. Build time 907 s at 16 threads.
- **Pipeline check** (`check.log`): with L = 0 and no sparse part, the same calls reproduce `data/feats/train/seq_cv_{invd20,pop,norm}` exactly (max |diff| = 0 on 223,854 rows of 3,000 feeds from two days).
- **Val/test rows:** byte copies of `data/feats/{val,test}/seq_cv_*`.

**Offsets**
- They are computed by the same code path on every split. On val/test they are asserted bit-equal to `src_expo_h`, `src_expo_slog` and `src_censored` (`vt.log`).
- `hzcv_stale_d` ranges: train 0-0.1 / 0.4-0.5 / 0.8-0.9 by L; val 0.2-0.5; public 0.60-0.75; private 0.75-0.90.

## 2. Leak argument (verified in code)

- **No labels anywhere.** No target, pos or label is read by `hz_common.py` / `hzcv.py`. The only label-informed object is the holdout feed list, which is used solely to force L = 0 on rows that never enter training.
- **Every source sequence is strictly earlier than the feed:**
  - dense part: day < day(c) ≤ own day, i.e. ts < c ≤ t;
  - sparse part: the kernel skips constrained sequences with ts ≥ t (`feat_covis.py` `_covis`: `if scons[s] and sts[s] >= t: continue`).
- **The own user is excluded.** Every sequence of the same non-zero user is skipped (earlier and later). User-0 queries use only strictly earlier user-0 feeds, as in the as-built version.
- **The offsets** read only the row's f4, the feed timestamp and constants. Both hashes depend only on feed_id.
- **No split-specific missingness.** NaN share is 0 on every split, and no train-only availability mechanism exists. The sparse sample is a feed_id hash, independent of labels and candidates.
- **Val/test inputs are unchanged.** They are the already-audited `seq_cv_*` / `src_expo_*` (review_leakage "Checked and correct").

## 3. Parity checks (label-free unless noted)

| quantity | train L = 0 | L = 4 | L = 8 | val | public | private |
|---|---|---|---|---|---|---|
| censored share, all train rows | 8.56% | **26.12%** | **33.29%** | 20.07% | 27.88% | 31.28% |
| `cv_pop == 0` share, all train rows | 0.419 | **0.653** | **0.757** | 0.420 | 0.464 | 0.476 |
| feeds with an EMPTY dense part | 9.3% (Sep 6) | **45.5%** | **76.0%** | 0 | 0 | 0 |

Label-conditional parity uses the strict random-like population: p_random > 0.9, M ≥ 50, holdout excluded, 59,794 feeds. It is compared with val M ≥ 50 (`detect_strict.log`).

| lift P(y \| group)/P(y) | train as built | train lagged (all) | lagged L = 4 | lagged L = 8 | val (Tue / Thu) |
|---|---|---|---|---|---|
| cv = 0 & censored | **0.427** | **0.698** | 0.705 | 0.759 | **0.682** (0.638 / 0.717) |
| cv = 0 & exposed 0-72 h | 0.753 | 0.854 | 0.945 | 1.043 | 0.941 |
| cv = 0 & exposed > 72 h | 1.020 | 1.056 | 1.067 | 1.074 | 1.000 |
| cv > 0 | 1.125 | 1.146 | 1.146 | 1.196 | 1.113 |
| `cv_pop == 0` share | 0.473 | **0.648** | 0.683 | 0.791 | 0.419 |
| within-feed AUC of `cv_invd20` | 0.5416 | **0.5244** | 0.5210 | 0.5105 | **0.5525** |

**Reading**
- The lag does what RESEARCH-2 asked for: the censored-zero lift moves from 0.43 to 0.70, against 0.68 on val, and the censored shares hit the targets.
- But it over-shoots co-vis coverage (zero share 0.65 vs 0.42-0.48) and weakens the co-vis signal on train below val's (AUC 0.524 vs 0.5525; as built 0.5416 was closer).
- The cause is the 14-day train window. For 43.3% of all train feeds the lagged cutoff falls on or before Sep 6, so the dense source is empty and only the 1/37 sparse window remains. No val/test row has that.

## 4. Leak / parity detectors (pre-declared: train random-like p_random > 0.5 vs all val)

Source: `detect.log`, 515,023 train feeds and 29.7M rows. The table gives within-feed AUC and per-feed z(pos) − z(neg) (dz).

| feature | AUC train (all / L0 / L4 / L8) | AUC val | dz train / val | zero share train / val | flag |
|---|---|---|---|---|---|
| seq_cvlag_invd20 | 0.5187 / 0.5313 / 0.5148 / 0.5078 | 0.5515 | +0.103 / +0.268 | 0.935 / 0.887 | **AUC Δ 0.033** |
| seq_cvlag_pop | 0.5131 / 0.5201 / 0.5096 / 0.5083 | 0.5505 | +0.034 / +0.140 | 0.609 / 0.420 | **AUC Δ 0.037** |
| seq_cvlag_norm | 0.5188 / 0.5315 / 0.5149 / 0.5078 | 0.5515 | +0.104 / +0.270 | 0.935 / 0.887 | **AUC Δ 0.033** |
| hzcv_expo_slog | 0.5078 / 0.5121 / 0.5051 / 0.5055 | 0.5447 | +0.025 / +0.156 | 0 / 0 | **AUC Δ 0.037** |
| hzcv_censored | 0.4941 / 0.4923 / 0.4961 / 0.4941 | 0.4654 | −0.034 / −0.162 | 0.795 / 0.799 | - |
| *ref: as-built seq_cv_invd20 / src_expo_slog (train)* | 0.5300 / 0.5078 | 0.5515 / 0.5447 | | | (Δ 0.022 / **0.037**) |
| hzcv_stale_d | feed-constant (no within-feed AUC) | | | | - |

- **NaN share** is 0 on every split, so there is no missingness lift to compare and no 1.5× flag.
- **The flags are parity flags, not leak flags.** They are also partly an artefact of the p > 0.5 population, which is not label-comparable with val for item-age inputs:
  - the as-built `src_expo_slog` has exactly the same 0.5078 vs 0.5447 gap;
  - in this population P(y) is 2.51% vs 1.58% on val;
  - the as-built censored-zero lift is 1.15 here, vs 0.43 in the strict population. Feeds with p_random in (0.5, 0.9) are partly production-ranked.
- **Strict population** (p > 0.9, M ≥ 50 vs val M ≥ 50): **no flags.** The largest gap is seq_cvlag_invd20 at 0.5244 vs 0.5525 (Δ 0.028); at L = 8 alone it is 0.5105 vs 0.5525 (Δ 0.042).
- **The residual train/val AUC deficit of the lagged co-vis** is the empty-dense-part effect of section 3. A leak would push the train AUC the other way, above val.

## 5. Screens

### (c) Train-fit LightGBM gate (`feat_new/hz_gate.py`)

The harness is a copy of the review's `lgb_causal.py`:
- 50K random-like non-holdout train feeds (7.35M rows);
- same PARAMS, 350 rounds, 2 seeds, 16 threads, clean BASE.

Absolute scores:
- **R0:** val 0.37156, holdout 0.29633, stale-val 0.37094. `lgb_causal.log` had 0.37226 at 12 threads; LightGBM is not bit-exact across thread counts, and all deltas below are paired within this run.
- **R1:** val 0.37188, holdout 0.29577, stale-val 0.37118.

| arm − R0 | val | Tue | Thu | M<50 | M≥50 | holdout | holdout Sep 17-19 | stale-val |
|---|---|---|---|---|---|---|---|---|
| **R1** (spec: lagged co-vis + 3 offsets) | **+0.00032 (0.00071)** | +0.00007 (0.00099) | +0.00060 (0.00101) | +0.00108 (0.00136) | −0.00020 (0.00075) | −0.00057 (0.00041) | +0.00113 (0.00076) | +0.00024 (0.00071) |
| D1 (lagged co-vis only) | −0.00013 (0.00070) | +0.00043 | −0.00076 | −0.00041 | +0.00006 | −0.00070 (0.00041) | +0.00039 | −0.00067 (0.00071) |
| D2 (offsets at L = 0 only) | +0.00004 (0.00067) | −0.00041 | +0.00054 | −0.00053 | +0.00044 | −0.00015 (0.00038) | +0.00023 | −0.00003 (0.00068) |
| R2 (capped lag, dense ≥ 3 days) | −0.00001 (0.00068) | −0.00000 | −0.00001 | −0.00013 | +0.00008 | −0.00005 (0.00039) | −0.00035 | −0.00003 (0.00069) |

- D1, D2 and R2 are diagnostics only, not used for the verdict.
- **R2 setup** (`feat_new/hz_lagcap.py`): only the gate's training rows are rebuilt (41.8% of them change). The training sample then has censored share 13.3% and `cv_pop == 0` share 0.490.
- **Gain shares (seed 0):** the co-vis columns hold 0.040 of total gain in R0 and 0.021 in R1. The trees lean on the lagged co-vis about half as much. hzcv_expo_slog 0.009 and hzcv_stale_d 0.007 pick up part of the slack; hzcv_censored stays at 0.0002.
- **Stale-val** is val with `seq_cv` from `review/verify_SHIFT-3/out/cv_sep18_*` (train source cut at Sep 18) and offsets recomputed with c = 2026-09-18 00:00 MSK.

### (b) Residual over the NN mix (`feat_new/hz_resid.py`)

- Folds are exactly those of nn/finetune.py (`GroupKFold(5)` by val user, no shuffle).
- Base = z(0.5 z(nn_final4_val_oof) + 0.5 z(nn_final4_trainonly_val)), which scores 0.39401.
- GBM: 60 rounds, 7 leaves, init = β·b. LIN: listwise logit.

| group | GBM | LIN | splits (GBM): Tue / Thu / M<50 / M≥50 |
|---|---|---|---|
| ALL6 | +0.00023 (0.00041) | −0.00063 (0.00034) | −0.00019 / +0.00070 / +0.00037 / +0.00014 |
| OFF3 | +0.00020 (0.00037) | −0.00011 (0.00028) | −0.00050 / +0.00097 / +0.00048 / +0.00000 |

### (a) eval_family (`feat_new/hz_eval_family.py`)

This is a copy of `src/eval_family.py` that reads feats_new. Results are appended to `reports/feature_evals.jsonl` under tag `hzcv`.

| run | delta (SE) | per seed |
|---|---|---|
| tag `hzcv`, 6 names vs the frozen base | +0.00291 (0.00078) | +0.0028 / +0.0018 / +0.0041 |
| `--base_extra seq_cv_*`, the 3 offsets only (not logged) | −0.00064 (0.00065) | −0.0008 / +0.0003 / −0.0014 |

- The frozen base excludes `seq_*`, so the first line re-measures the existing co-vis inputs (compare `seq_grp_CV_s1`, +0.0016 ± 0.0012). It is not evidence for the lag.
- Seed 2's base is the known low draw.

## 6. Mechanism check (`feat_new/hz_ae.py`)

Val actual/expected positives, with one global temperature per model.

| model | cv = 0 & censored (19.8%) | cv = 0 & exposed 0-72 h (4.9%) | cv = 0 & exposed > 72 h | cv > 0 |
|---|---|---|---|---|
| R0 (as built) | 1.130 ± 0.018 | 1.092 ± 0.032 | 0.991 | 0.974 |
| R1 (hzcv) | 1.123 ± 0.018 | 1.090 ± 0.032 | 0.988 | 0.976 |

- The GBDT shows the same under-rating of censored zeros on val as the train-only NN (RESEARCH-2: 1.055 / 1.103).
- Training on lagged rows does not repair it. The censored-zero lift in the training rows did move to the val level (section 3), but the model compensates through the weaker co-vis signal and through f4 / offsets, so val calibration stays where it was.

## 7. Why it fails, and what it would take

- **Parity trade.** The train window starts on Sep 6, so a 4- or 8-day lag empties the dense source for 43% of train feeds (76% at L = 8). The lagged rows then fix the censoring mismatch (lift 0.43 → 0.70 vs val 0.68) but create a coverage mismatch (zero share 0.65 vs 0.42-0.48; AUC 0.524 vs val 0.5525). Val/test rows, which always have the full 13-day dense source, look unlike either.
- **A side effect on `hzcv_stale_d`.** At L = 8 (stale_d 0.8-0.9) three quarters of train feeds have no dense source, while private rows with the same stale_d have a full one. The offset therefore co-varies with "no dense source" on train only. This is a reason not to ship the spec version even if a NN run looked positive.
- **The density-safe variant** (R2, every lagged row keeps ≥ 3 dense days) restores coverage parity but gains exactly nothing (−0.00001 ± 0.00068). Its censoring shift is smaller (13.3%).
- **The val/test side cannot change.** The val/test inputs are identical by design, so the family can only act through the train stage. Its expected size (≤ +0.0008 val) is below the gate's MDE (~0.0014), and every estimate came out within ±0.5 SE of zero.
- **What would be needed:** a longer pre-period of train data (not available), or a synthetic dense source for the missing days (not leak-free to construct). The fine-tune already re-calibrates censored zeros (RESEARCH-2: 0.967), and v11's fine-tuned half carries that to test. **No further work is recommended on this family.**

## 8. NN scaling (only if someone runs it anyway)

- **`seq_cvlag_*`** match `attach_xt`'s `seq_cv` prefix, so they get log1p(max(x, 0)) and are never clipped.
  - Train ranges: invd20 0-248.7, pop 0-3000, norm 0-6.23.
  - Swap `seq_cv_invd20` → `seq_cvlag_invd20` in XT_REL. The rank is unchanged by log1p.
- **The offsets** take the generic ±10 clip, which never fires:
  - `hzcv_expo_slog` −2.30…8.34;
  - `hzcv_stale_d` 0…0.9;
  - `hzcv_censored` {0, 1}.
- **No NaN**, so no `has:` flags. No name ends in `_z` or `_rk`.
- **Preset `cvnd_hz`** = cvnd with `seq_cv_{invd20,pop,norm}` → `seq_cvlag_*`, plus the 3 offsets. The fine-tune branch is unchanged. For LightGBM, use raw floats.

## 9. Files and commands

- **Features:** `<repo>/data/feats_new/{train,val,test}/{seq_cvlag_invd20,seq_cvlag_pop,seq_cvlag_norm,hzcv_expo_slog,hzcv_censored,hzcv_stale_d}.npy` and `<repo>/data/feats_new/train/hz_lag.npy`.
- **Build:** `CUDA_VISIBLE_DEVICES="" python feat_new/hzcv.py {vt|check|train} 16`.
- **Screens:**
  - `feat_new/hz_detect.py [rl05|strict]`
  - `feat_new/hz_lagcap.py`
  - `feat_new/hz_gate.py`
  - `feat_new/hz_eval_family.py --names ... --tag hzcv --threads 16`
  - `feat_new/hz_resid.py`
  - `feat_new/hz_ae.py`
- **Runner:** `scratchpad/featrev/hzcv/run_screen.sh`.
- **Logs and JSON:** `/tmp/scratchpad/featrev/hzcv/`. Files: `{train,vt,check,detect,detect_strict,lagcap,gate,evalfam,evalfam_off,resid,ae}.log`, `gate_summary.json`, `resid_summary.json`, `detect.json` (pre-declared rl05 population), `detect_strict.json`.
