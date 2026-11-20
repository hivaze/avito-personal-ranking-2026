# featnew: family `ud` (user-vector drift since the user's strictly earlier snapshot): REJECT

Date 2026-11-11, CPU only (CUDA_VISIBLE_DEVICES="", at most 16 threads in total). Code: `feat_new/ud.py`. Scratch scripts and logs: `/tmp/scratchpad/featrev/ud/` (u01-u07).

## 0. Verdict
**Reject.** All four screens were declared before running and none passes. Each is within about 1 SE of zero and inside the matched noise-control band:

| screen | delta (SE) | rule | result |
|---|---|---|---|
| residual over the current NN mix (nn_final5, val 0.39793), group G_ud, LIN | −0.00008 (0.00020) | > 2 SE in LIN or GBM, ≥ 0 in the other | FAIL |
| same, GBM | +0.00027 (0.00028) | | (0.96 SE) |
| residual over the nn_final4 mix (task step 3b), LIN / GBM | −0.00012 (0.00021) / −0.00044 (0.00029) | | FAIL |
| val-only user-fold LightGBM (`eval_family` copy, 3 seeds) | −0.00039 (0.00067) | > 2 SE | FAIL |
| train-fit LightGBM gate, val / train holdout | −0.00056 (0.00069) / +0.00034 (0.00042) | val > 2 SE | FAIL |

Leak and parity detectors: **no flags** (section 3).

The "+0.00038 ± 0.00018" prior from `s16_udrift.py` does reproduce exactly on its own base. That base was nn_final4 mix + 0.1 z(pre_final). Without the pre term the same blend gives +0.00013 (nn_final4 mix) and +0.00009 ± 0.00020 on the current nn_final5 mix (section 4d). Most of the old number came from the pre-term base, which nn_final5 has dropped.

**Recommendation:** do not spend the fine-tune GPU pair (≥ 2 FT seeds) on `ud`. The files stay in `data/feats_new/` for reference only.

## 1. Construction (`feat_new/ud.py`)
For a feed g of a non-zero user at quantised ts t (every feeds.parquet timestamp is a multiple of 128 s; rounding is monotone, so stored ts_P < t means a truly earlier request):
- **P** = the same user's feed with the largest ts strictly below t, over train + val + test. If several feeds share that ts, the largest (split code, fidx) is taken.
  - The code asserts `t_P < t` for every pair (`ud.py` prev_feeds) and that no val feed gets a test P.
  - User 0 never gets a P.
- **Covered feed**: P exists and neither u_now nor u_prev is all zero.
- **Candidate vector**: the row of the own-split item table, through `cand_emb_idx`. It is NaN when the index is −1: 214 val rows and 455 test rows. No used item row is all zero.
- **Cosine**: `a·b / ((|a|+1e-6)(|b|+1e-6))`, the same as `src/features.py` cos_u.
- **Files** (float32, aligned with `data/flat/<split>/cand_*` rows, NaN on uncovered feeds), in `data/feats_new/{val,test}/`:
  - `ud_cos_prev` = cos(c, u_prev)
  - `ud_drift` = cos(c, u_now) − ud_cos_prev
  - `ud_drift_fz` = within-feed z of ud_drift over non-NaN rows
  - `ud_gap_l` = log1p(gap days), feed-level
  - `ud_has` = row-level non-NaN indicator of ud_drift
- **No train files are written**, as the family spec says (val/test-only fine-tune extra). The diagnostics build train values with the same `build()` code path, in memory only.
- **Build**: `CUDA_VISIBLE_DEVICES="" .venv/bin/python feat_new/ud.py --splits val,test --threads 16`, wall time 2.3 s.

**Verification** (`u01_verify.py`, `u01.log`):
- Brute-force P (polars filter over all feeds) against the builder: 0 mismatches in 400 random val feeds and 0 in 400 random test feeds.
- On covered rows, cos(c, u_now) = ud_drift + ud_cos_prev equals `data/feats/<split>/cos_u` to within 1.2e-7 on both splits.

## 2. Coverage and val/test parity (`u01.log`)
| segment | feeds | has P (any) | covered (non-zero vectors) | gap median (p10-p90), days | P from train / val / test |
|---|---|---|---|---|---|
| val | 14,423 | 30.5% | 29.45% | 9.9 (3.9-15.8) | 4222 / 25 / 0 |
| val Tue | 7,543 | 30.7% | 29.6% | 8.9 (3.1-14.8) | 2222 / 9 / 0 |
| val Thu | 6,880 | 30.2% | 29.3% | 11.0 (5.2-16.7) | 2000 / 16 / 0 |
| public | 8,562 | 30.4% | 29.6% | 13.0 (7.2-19.0) | 2488 / 28 / 17 |
| private | 20,033 | 29.0% | 28.2% | 15.2 (9.1-21.0) | 5505 / 62 / 88 |

The task's coverage figures (30.5 / 30.4 / 29.0%) are the "has P" column. The zero-vector rule removes about 1 point.

Covered-row moments, val vs test:

| feature | val mean (sd) | test mean (sd) |
|---|---|---|
| ud_cos_prev | 0.458 (0.092) | 0.455 (0.092) |
| ud_drift | 0.0155 (0.070) | 0.0187 (0.071) |
| ud_gap_l | 2.27 (0.48) | 2.66 (0.39) |

`ud_gap_l` shifts by **+0.83 val-sd** on test. If the family were ever used in the NN, `ud_gap_l` would be an extrapolating input under attach_extra's val-fitted standardisation.

## 3. Leak / parity detectors (pre-declared; `u02_detect.py`, `u02.log`, `u02_detect.json`)
Comparison set TRR = all 515,023 train feeds with p_random > 0.5, built with the same `build()`, against all val feeds.
- **AUC**: within-feed, over present rows, on feeds with both classes.
- **Lift**: P(y | present) / P(y | missing).
- **z_pos / z_neg**: mean per-feed z of positives / negatives.

| feature | AUC TRR | AUC val | diff | lift TRR | lift val | lift ratio | z_pos TRR / val | flag |
|---|---|---|---|---|---|---|---|---|
| ud_cos_prev | 0.5792 | 0.6068 | −0.028 | 0.996 | 1.086 | 1.09 | 0.272 / 0.362 | none |
| ud_drift | 0.5345 | 0.5484 | −0.014 | 0.996 | 1.086 | 1.09 | 0.124 / 0.154 | none |
| ud_drift_fz | 0.5345 | 0.5484 | −0.014 | (same) | | 1.09 | 0.124 / 0.154 | none |
| ud_gap_l | feed-constant (AUC 0.5 by definition) | | | 0.996 | 1.086 | 1.09 | | none |
| reference: cos(c, u_now) = cos_u | 0.6032 | 0.6414 | −0.038 | | | | 0.358 / 0.480 | AUC (reference only) |

- **The reference shows the set-composition effect.** The known-clean per-request cos_u itself differs by 0.038 on this TRR set (all M, p_random > 0.5). On M ≥ 50 random-like feeds it matches val (featrev_prep sec 2, s09). So the 0.03 AUC rule is loose for this set.
- **Relative strength matches.** (AUC_drift − 0.5)/(AUC_cos_u − 0.5) is 0.33 on TRR and 0.34 on val.
- **Gap-matched TRR** (gap ≥ 3 days, 42.7K feeds): drift AUC 0.5351, cos_prev 0.5682.
- **AUC by val gap tercile** (drift / cos_prev):
  - T1, < 6.8 d: 0.554 / 0.613
  - T2, 6.8-12.2 d: 0.533 / 0.607
  - T3, ≥ 12.2 d: 0.558 / 0.600
  - There is no clear decay with gap, so the longer private gaps should not change the signal much.
- **Leak by construction**: only a strictly earlier same-user snapshot is used (asserted), no labels, and user 0 is excluded. The missingness mechanism is identical on val and test (has P plus non-zero vectors). Coverage tracks returning users: val 29.5%, private 28.2%.

**Correction to the family spec's leak argument.** The spec says a train production user's previous feed is "minutes old". Measured (`u07_traincov.log`):

| train feeds | has P | gap median | gap < 1 h | gap ≥ 3 d |
|---|---|---|---|---|
| all train | 18.1% | 2.87 d | 3.0% | 48% |
| production (p ≤ 0.5) | 18.3% | 2.83 d | | |
| random-like (p > 0.5) | 17.4% | 3.02 d | | |

The real train/val mismatch is lower coverage and shorter gaps, because train starts on Sep 6. Feeds minutes apart are rare. The conclusion (no train-stage use) still holds: the train-fit gate below shows no transfer.

## 4. Screens (all pre-declared; no tuning on val)
### 4a. Residual over the NN on val (`u03_resid.py`, a copy of inventory c03b/c08 models with unchanged params)
- **Folds**: exactly `nn/finetune.py` GroupKFold(5) by user_id, no shuffle.
- **LIN**: listwise softmax with clip-5 z + pctrank per feature.
- **GBM**: LightGBM lambdarank with init score β·b, inputs raw + z, 60 rounds, 7 leaves, min_data 2000, l2 50.
- **G_ud** = [ud_drift, ud_cos_prev, ud_gap_l]. z(ud_drift) ≡ ud_drift_fz (asserted), and ud_has is the NaN pattern, so this group carries all five files.
- **Noise controls**: 4 groups of 3 Gaussian columns with the same NaN pattern (2 row-level columns, 1 feed-constant column like gap).

| base (control NDCG) | arm | LIN Δ (SE) | GBM Δ (SE) | covered feeds LIN / GBM |
|---|---|---|---|---|
| **mix5** = 0.5 z(nn_final5_val_oof) + 0.5 z(nn_final5_trainonly_val) (0.39793) | **G_ud** | **−0.00008 (0.00020)** | **+0.00027 (0.00028)** | −0.00026 / +0.00049 |
| | S_drift (ud_drift + isnan) | −0.00003 (0.00016) | −0.00018 (0.00026) | |
| | 4 noise groups | −0.00006 … +0.00006 | −0.00020 … +0.00006 | |
| mix4 = 0.5 z(nn_final4_val_oof) + 0.5 z(nn_final4_trainonly_val) (0.39401) | G_ud | −0.00012 (0.00021) | −0.00044 (0.00029) | −0.00040 / −0.00155 |
| | S_drift | −0.00017 (0.00014) | −0.00044 (0.00027) | |
| | 4 noise groups | −0.00022 … +0.00019 | −0.00062 … +0.00006 | |

PASS = False on both bases (`resid_mix5/summary.json`, `resid_mix4/summary.json`).

The fitted LIN weights give drift a negative pctrank weight (−0.07) and cos_prev a positive one (+0.10). Given the NN, which already sees cos_now, the model wants a little more "old-interest" match, not drift.

Segments of the G_ud delta on mix5 (`u06.log`):

| segment | LIN | GBM |
|---|---|---|
| Tue | +0.00007 (0.00028) | −0.00005 (0.00041) |
| Thu | −0.00023 (0.00028) | +0.00063 (0.00037) |
| M < 50 | −0.00002 | +0.00031 |
| M ≥ 50 | −0.00012 | +0.00025 |

No segment reaches 2 SE. The GBM moves uncovered feeds by +0.00019 (0.00020), even though they carry no ud information, which shows the GBM noise floor.

### 4b. Val-only user-fold LightGBM (`u04_evalfam.py`)
This is `src/eval_family.py` copied unchanged except that ud_* is read from `data/feats_new`. Frozen base with cached base OOF, 3 seeds, 8 threads.
- Names: ud_cos_prev, ud_drift, ud_drift_fz, ud_gap_l.
- Result: Δ = **−0.00039 ± 0.00067**; per seed −0.0007, −0.0023, +0.0018.
- Appended to `reports/feature_evals.jsonl` with tag `ud`.

### 4c. Train-fit LightGBM gate (`u05_gate.py`)
Same setup as inventory c06 and gate_pre2 g1:
- 25K random-like train feeds (p > 0.9, M ≥ 50, not in the NN holdout, rng 5).
- Base = swap_luo(base_features.json), 122 columns; diag_trainstage PARAMS, 350 rounds, 2 seeds.
- Train-row ud from the same builder: coverage 17.0%, gap median 3.0 d.

| | val | holdout (30K NN-holdout feeds; coverage 17.1%) |
|---|---|---|
| U − A, all feeds | **−0.00056 ± 0.00069** | +0.00034 ± 0.00042 |
| U − A, covered feeds | −0.00028 ± 0.00131 | +0.00189 ± 0.00105 |

- Gain share of the 4 ud columns: 0.25% / 0.24% / 0.19% / 0.16%.
- The base A val here is 0.36336, vs 0.36427 in the c06 log for the same setup. This is the known non-bit-exact LightGBM under load; the deltas are paired within one run.
- The holdout-positive / val-negative pattern is the familiar train→val asymmetry, and none of it reaches 2 SE.

### 4d. Diagnostic only, not used for the verdict: the s16 fixed blend b + 0.05·z(ud_drift_fz) (`u06.log`)

| base | Δ (SE) | covered feeds |
|---|---|---|
| s16 base: nn_final4 mix + 0.1 z(pre_final) | +0.00038 (0.00018) | +0.00130 |
| nn_final4 mix | +0.00013 (0.00018) | +0.00045 |
| nn_final5 mix | +0.00009 (0.00020) | +0.00031 |

The first row reproduces s16 exactly.

## 5. If it were ever used in the NN (record only)
- **attach_extra** (`nn/data.py:203-220`): no ud column triggers the log rule (min ≥ 0 & max > 20). ud_gap_l max is 3.2 and ud_has max is 1.
  - Values are standardised with val stats and clipped at ±6. NaN → 0 = the val mean, which is why `ud_has` exists.
  - Use ud_drift_fz rather than raw ud_drift.
  - Leave ud_gap_l out: its test values sit 0.83 sd above val.
- **Integration**: `finetune.py` only finds extras in `data/feats/` or `nn/feat_cache/` (`feat_exists`, `nn/data.py:45`). The NN owner would have to hard-link the files from `data/feats_new/{val,test}/`.
  - Do not use with `--add_random`: there are no train files.
- **LightGBM**: raw values, NaN as missing (as screened).

## 6. Files
- Code: `<repo>/feat_new/ud.py`
- Features: `<repo>/data/feats_new/{val,test}/{ud_cos_prev,ud_drift,ud_drift_fz,ud_gap_l,ud_has}.npy`
- Scratch scripts, logs and JSON in `/tmp/scratchpad/featrev/ud/`:
  - u01_verify.py / u01.log
  - u02_detect.py / u02.log / u02_detect.json
  - u03_resid.py / u03_mix5.log / u03_mix4.log / resid_{mix5,mix4}/
  - u04_evalfam.py / u04.log
  - u05_gate.py / u05.log / u05_gate.json
  - u06_diag.py / u06.log / u06_diag.json
  - u07_traincov.log
