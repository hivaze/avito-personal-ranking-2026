
## 6. Drop-in for the trainer

- `nn_x/w12/stale/hz2/cd_arms6b.py` = `nn_x/w5/cd_arms6.py` (md5 2801de7b…, byte-identical to `nn_x/abl/cd_arms6.py`) with three changes, each marked `# 6b` (`cd_arms6b_vs_cd_arms6.diff`, 84 lines):
  1. the view directory comes from the environment variable `HZCD_DIR` (default `data/feats_new` = the existing half-dose files);
  2. the column map also covers `seq_cv_invd20`, `seq_cv_pop`, `seq_cv_norm`, `xp_imp_c50`; a column is swapped only if its file exists in `<HZCD_DIR>/train/` (`seq_cv_hz_invd20.npy`, `seq_cv_hz_pop.npy`, `seq_cv_hz_norm.npy`, `xp_imp_hz_c50.npy`; none exists in `data/feats_new`);
  3. optional `HZ2_KEEP=<npy, bool per train candidate row>`: flagged rows keep their stored xt values.
- Same scaling as the stored columns: the files are raw-scale float32 like `it_hist_hz_*.npy`; `load_hz` applies the `attach_xt` branch of the name (log1p for `it_hist*` / `seq_cv*`, clip ±10 for `xp_imp*`) and stores fp16. The in-model rank / z of `seq_cv_invd20` and `it_hist_all_luo` follow the swap; `f_mean` / `f_std` stay the as-built statistics, as on test rows.
- Proof (`test_cd_arms6b.py`, `logs/test_cd_arms6b.log`, CPU MiniSplit of 200 random train feeds, 15,471 rows, preset `cvnd_pre2s`) [M]:

| test | result |
|---|---|
| T1 text | `HzTrainView`, `xt_transform`, `load_hz`, `hz_path`, `hz_stats`, `save_holdout_hz` AST-identical; changed: `luo_cols`, `setup_hzcd`; new: `apply_keep` |
| T2 `HZCD_DIR` unset | same 5 columns, bit-identical hz tensor, **8 batches x 22 keys bit-identical** between `cd_arms6` and `cd_arms6b`; `hz_stats` identical |
| T3 `HZCD_DIR` = `feats` | 8 columns swapped (xt 0-7); every swapped cell = fp16(attach_xt transform of the hz2 file); every other key / column = the plain gather |
| T3b popularity-light xt list (27 columns, `xp_imp_c50` last) | `cd_arms6` refuses the list (no LUO column); `cd_arms6b` swaps exactly `xp_imp_c50` |
| T4 `HZ2_KEEP` | 3,339 of 15,471 sample rows flagged: stored values in all 8 columns; the other rows hz2 |

- Change to `train.py` (NOT applied; `train_py_hz2.diff`; the same line exists once in `nn_x/abl/train.py`). Copy `cd_arms6b.py` next to `train.py` first (a new file, nothing overwritten):

```diff
--- nn_x/w5/train.py
+++ nn_x/w5/train.py (hz2)
@@ -54,5 +54,5 @@
 from cd_arms4 import setup_wave4
 from cd_arms5 import save_holdout, save_last, save_last_test, setup_wave5
-from cd_arms6 import save_holdout_hz, setup_hzcd  # wave-5 hzcd (INSIGHTS_1112 item 6; OFF by default)
+from cd_arms6b import save_holdout_hz, setup_hzcd  # wave-5 hzcd; 6b = cd_arms6 + env HZCD_DIR / HZ2_KEEP (hz2 view); identical when both are unset
 from cd_arms7 import setup_valcd  # wave-5 valcd (val-in-cooldown, D1; OFF by default)
 from cd_arms8 import setup_rdrop, setup_sam  # wave-7 R-Drop / SAM cooldown regularisers (OFF by default)
```

- One line. With both variables unset every queued `--hzcd` lane (arm hz) runs exactly as before (T2). An hz2 lane is the hz lane plus `HZCD_DIR=<repo>/nn_x/w12/stale/hz2/feats HZ2_KEEP=<repo>/nn_x/w12/stale/hz2/feats/train/hz2_keep_rows.npy` in its environment. `tests/strip_hzcd.py` expects the old import line and needs the same one-word change.
- Side effects to know: the view tensor is 3.95 GB on the GPU (8 x 246.7M x fp16) against 2.47 GB for hz. `preds/holdout/nn_<v>_holdout_hz.npy` becomes a causal lagged-holdout read of the selected checkpoint (every holdout feed has a lag in the view; with `HZ2_KEEP` its Sep 6-8 truncated rows are as stored).
- Per-feed lag for a later model input: `feats/train/hz2_lag.npy` (int8, 6-9), `hz2_part_pub.npy` (bool), `hz2_cut_slot.npy` (int16, lagged cutoff as day slot; Sep 6 = 0). For val / test rows the same quantity is day - Sep 20: val 2 / 4, test 6-9.

## 7. Pre-registered expectation for the hz2 arm (`PREREG_hz2_arm.txt`, written 03:14Z before any model saw an hz2 view and before the hz arm was read)

Proposed arm: the K: / L: re-cool with `--hzcd` and `HZCD_DIR=feats`, `HZ2_KEEP` on ("hz2k"); reference = the ref re-cools; comparison = hz, paired by snapshot. Central value and 80% interval, 16-pool, selected checkpoints:

| reading | expected | why |
|---|---|---|
| stale-val, hz2k - reference | +0.0018 [+0.0008, +0.0032] | hz (11-12, hard labels) gave +0.00107 ± 0.00036 with a third of the `it_hist` dose and no `seq_cv` dose |
| stale-val, hz2k - hz | +0.0007 [-0.0002, +0.0018] | 2.3-2.8 x the dose, concave response |
| val | 0.0000 [-0.0006, +0.0006] | val rows have lag 2 / 4 d; no feed of the view has a lag below 6 |
| in-period holdout (as built) | -0.0010 [-0.0020, -0.0003] | about twice the hz reading (-0.00052); reported, not a veto |
| causal lagged holdout, days >= Sep 13: hz2k model - reference model, both through the hz2 view | +0.0015 [+0.0005, +0.0030] | a model cooled on these inputs against one that never saw them |

Falsifiers of the mechanism "part of the staleness cost is a train / test mismatch that training removes" (any one): F1 stale-val of hz2k not above +1 SE although its dose is more than twice that of hz; F2 stale-val(hz2k) - stale-val(hz) < -1 SE of the paired difference; F3 the lagged-holdout reading <= 0 (then the cost is information loss, not mismatch); F4 in-period holdout >= 0 and stale-val <= 0 (no-op). Not a falsifier: val in [-0.0006, 0] with stale-val >= +2 SE; if "val >= 0" stays a hard condition, hz2 is closer to failing it than hz was.

## 8. Limits

- **Two thirds of the rows are an approximation.** Exact replay exists for 34.2% of rows (listings created on or after Sep 6 or after the cutoff). For the other 65.8% the test-like value is not observable; the view matches coverage by one calibrated parameter and a depth factor, like poplag's band g, and its parity on those rows is partly by construction.
- **Old listings are not stale in the primary view** (nearest dense day 1-2 d before the row). Their zero share matches private rows on days >= Sep 13; their level is 0.2-0.3 log1p units above. hz2L restores the lag on Sep 17-19 only (6-8 d), with 5 dense days instead of 12.
- **Sep 6-8 (27% of train feeds) cannot be made test-like causally.** The view is far colder than test there (dZ 0.23), the stored columns are far warmer in `it_hist` and equally shallow in `seq_cv` / `xp_imp`. `HZ2_KEEP` leaves the truncated rows of those feeds as stored; it is a choice between two wrong formats, made on the feed-level cold share.
- Strictly causal sparse days make young listings 0.014-0.047 colder than private rows; a test row also sees later sparse days.
- The depth factor makes values non-integer and caps at 4; on Sep 7-8 it under-corrects. A zero stays a zero, so shallow windows have too many zeros (hz2L: +0.05-0.06 for old listings).
- The gate reference is test feeds with M >= 50; the view rows are exploration-like train feeds of the same M range. Production feeds (trAll) have a more popular item mix and are reported for information only.
- Marginal and pairwise-joint parity by listing age is shown; parity of the full joint with the other 26 inputs is not. The conditional check of 4.5 is one input at a time.
- One draw of the lag hash and of the public / private part.
- The listing age comes from f4 knots (128-second grid); its accuracy on train against test rows is not tested. Exact rows do not depend on it; truncated rows do (through X_test).

## 9. What I did NOT check

- No model was trained on the view. No GPU was used.
- No val or test label was used for anything except the stored-val within-feed AUC of section 4.5 (descriptive, declared in ADDENDUM 4).
- The view for val / test rows was not built (the stored columns are the view there: V1).
- A transductive variant (sparse days after the row counted as on a test row) was measured (section 2) but not written.
- A variant with val-like lags (2 / 4 d) mixed in was not built.
- No sensitivity of the gate to the lag hash, to the reference (all-M test feeds) or to the bucket edges.
- `tests/test_hzcd.py` of nn_x/w5 was not re-run against the patched import (I do not edit train.py); the GPU end-to-end part of that test needs a GPU.
- Whether the stale-val inputs (`nn_x/r6/stale_cf`) resemble the hz2 view was not compared.

## 10. Commands to reproduce

```
cd <repo>/nn_x/w12/stale/hz2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
PY=<repo>/.venv/bin/python
$PY sizes.py                          # calendar and size facts                          -> logs/sizes.log
$PY prep.py                           # postings + per-day count matrices (3 min)         -> cache/
$PY views.py v1 ; $PY views.py v2     # code checks V1 / V2                               -> logs/v1.log, logs/v2.log
$PY views.py calib ; $PY views.py calib_late ; $PY diag_m.py    # calibration of m       -> logs/calib*.log, logs/diag_m.log
./chain.sh "" 12                      # PRIMARY view, all 246.7M rows x 9 columns (9 min) -> feats/train/
./chain.sh L 5                        # lag-first variant (7 min)                         -> feats_L/train/
$PY gate.py final ; $PY gate_summary.py final                   # gates                  -> logs/gate_final.txt, logs/gate_summary_final.md
$PY audit.py feats 12 ; $PY audit.py feats_L 5                  # causality audit V3     -> logs/audit_feats*.txt
$PY joint.py ; $PY cond.py            # joint parity, model-free conditional check        -> logs/joint.txt, logs/cond.txt
$PY test_cd_arms6b.py                 # drop-in proof T1-T4                               -> logs/test_cd_arms6b.log
$PY hold_infer.py feats all late ; $PY hold_infer.py feats_L all late   # 12 members on holdout feeds of days >= Sep 13 through the views
$PY hold_read.py feats feats_L        # labelled holdout read (train-holdout labels)      -> logs/hold_read_feats_feats_L.txt
# for the lead, after an hz2 run exists: reference checkpoint through the view on the late holdout feeds
$PY hold_infer.py feats w5_kcd_ref_s951,w5_kcd_hz2_s951 late
```

## 11. Files created (all under `nn_x/w12/stale/hz2/` unless stated)

- `DECLARATION.txt` (design, gates, calibration rule; ADDENDA 1-4, each stamped before its build or read), `PREREG_hz2_arm.txt`, `PROGRESS.md`.
- **View, primary:** `feats/train/it_hist_hz_{all,et1,et2,et3,et4}.npy`, `seq_cv_hz_{invd20,pop,norm}.npy`, `xp_imp_hz_c50.npy` (float32, raw scale, 246,745,836 rows each), `hz2_lag.npy`, `hz2_part_pub.npy`, `hz2_cut_slot.npy` (per feed), `hz2_keep_rows.npy` (bool per row). 8.6 GB.
- **View, lag-first:** `feats_L/train/` with the same file names. 8.6 GB.
- **Drop-in:** `cd_arms6b.py`, `cd_arms6b_vs_cd_arms6.diff`, `train_py_hz2.diff`, `test_cd_arms6b.py`.
- Code: `hz2_common.py`, `prep.py`, `kernels.py`, `views.py`, `chain.sh`, `gate.py`, `gate_summary.py`, `audit.py`, `joint.py`, `cond.py`, `diag_m.py`, `sizes.py`, `hold_infer.py`, `hold_read.py`.
- Logs: `logs/` (one per command above; `gate_final.txt` holds every population x part x day-group table).
- Cache: `cache/` 41 GB (postings, day matrices, row masks, seq parts, holdout predictions `hpred_*`). Only `hpred_*`, `gate_final.json` and `calib*.json` are needed again; the rest can be rebuilt in 12 minutes and may be deleted.
- `reports/STALE_1115/hz2.md` (this file).
