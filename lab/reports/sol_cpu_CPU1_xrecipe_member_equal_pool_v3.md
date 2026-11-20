# CPU1_xrecipe_member_equal_pool_v3: cross-recipe member-equal pools on the n2cpv3 basis

Date: 2026-11-11, 19:55-20:15 UTC. CPU only (`CUDA_VISIBLE_DEVICES=""`, 16 threads). No training. Inputs are read-only from `preds/`, `preds/holdout/` and `data/`.
* Code:
  * `<repo>/solvers/CPU1_xrecipe_member_equal_pool_v3.py` (main script; `--phase A|B`). `<repo>/solvers/ensemble/exp12_pools.py` is a thin wrapper around it.
  * `<repo>/solvers/CPU1_xrecipe_member_equal_pool_v3_info.py` (information-only decomposition, no selection).
* Outputs: `<repo>/solvers/out/CPU1_xrecipe_member_equal_pool_v3/`
  * `phaseA.{log,json}` and `info.{log,json}`;
  * `cand_A_{Q1,Q2,Q3,Q2h,Q3h}_{val,test}.npy` (float32, aligned with `data/flat/<split>` cand rows);
  * `gate_A_<pool>.parquet` (the rank file that presub_gate checks; NOT a submission) and `gate_A_<pool>_vs_{R,v11}.log`.
* Nothing was written under `preds/`, `subs/`, `nn/` or `src/`. presub_gate ran in-process with its `P` pointed at `gate_links/` (symlinks to the reference preds plus our vectors).

## 0. Verdict

**The holdout-chosen phase-A pool is Q3 = F5+F6+F5cd+H (21 members). It does NOT pass the frozen final-pick rule. R (sub_v12) stays pick 1.**

* **Paired val:**
  * Q3 vs R: +0.00061 ± 0.00046 (z 1.33).
  * Q3 vs v11: +0.00086 ± 0.00047 (z 1.82).
  * Both are below the > 2 SE threshold.
* **Every other criterion holds:**
  * no slice is below −2 SE (every slice is positive);
  * holdout TO half 0.32300 ≥ 0.32162;
  * presub_gate PASS with 21 WARN, all FR-2;
  * file checks pass.
* **Q3 is not final-eligible anyway.** Its F5cd and H fine-tune halves are n2cpf, because their n2cpv3 fine-tunes do not exist yet. So it is "info, not final".
* **Pick-2 hedge by the frozen rule:**
  * All three pools are within 1 SE of R (in fact above it). Q3 has the lowest test rank correlation with R (Spearman 0.9914), so Q3 is the rule's pick-2 hedge, once its FT halves are n2cpv3.
  * The only final-eligible pool today is Q1 = F5+F6 (12 members, all n2cpv3): +0.00028 ± 0.00043 vs R, test Spearman 0.9925.
* **The holdout result is the strong evidence for pooling:**
  * holdout: Q3 +0.00138 ± 0.00028 (z 4.9), late 20% +0.0014 ± 0.0006;
  * train-only half on val: +0.00102 ± 0.00050 (z 2.04).
  * The full mix gains less on val because the fine-tuned branch dilutes it (section 4).
* **Verdict label: "likely".** The gain is positive on every slice and robust on the holdout, but it is below the pre-declared val bar.

## 1. Pre-declared setup (applied exactly)

Families and file rules:

| family | members | TO | FT used | holdout |
|---|---|---|---|---|
| F5 | v12_all_{e10,s81,s82}, v12_allprew_{s83,s84,s85} | trainonly | **n2cpv3** | yes |
| F6 | v13_lr1_mix1, v14_s92..s96 | trainonly | **n2cpv3** | yes |
| F5cd | v15_cd_all_{e10,s81,s82}, v15_cd_allprew_{s83,s84,s85} | trainonly | n2cpf (n2cpv3 missing) | yes |
| H | v13_lr1_mix0, v13_lr2_mix1, v14_mix1_s91 | trainonly | n2cpf (n2cpv3 missing) | yes |
| F7 | v17_pre2s_s101..s106 | not yet produced (stage 1 at 47.5k/98k steps at 19:57) | | |

* **Combiner:** take the per-feed z of each member, average the members with equal weight inside each branch, then final = 0.5 z(FT mean) + 0.5 z(TO mean).
* **Pools:**
  * phase A: Q1 = F5+F6, Q2 = Q1+F5cd, Q3 = Q2+H;
  * phase B: Q1B = F5+F6+F7, Q3B = Q3+F7.
* **Choice:** the pool with the highest holdout TO-half NDCG inside the phase. That is 3 choices, made on the holdout, never on val.
* **References:**
  * R = 0.5 z(nn_final5v3_val_oof) + 0.5 z(nn_final5_trainonly_val) = **0.39818**. The script rebuilds it from the 6 F5 n2cpv3 member files and gets 0.39818 (`phaseA.log:2`).
  * v11 = **0.39793**.
  * R and v11 share the F5 TO half, so their holdout is the same: **0.32162**.
* **Strength gate:** a family whose mean single-member TO NDCG is more than 0.002 below the best family on val OR on the holdout is excluded.
* **Fold check:** nothing is fitted on val, so no nested CV is needed. Per-fold deltas over nn/finetune.py's GroupKFold(5)-by-user folds are reported as a stability check.

## 2. Members and strength gate (`phaseA.log:4-29`)

| family | n | mean TO val | gap | mean TO holdout | gap | mean FT val | gate |
|---|---|---|---|---|---|---|---|
| F5 | 6 | 0.39247 | 0.00180 | 0.31790 | 0.00151 | 0.39404 (n2cpv3) | pass |
| F6 | 6 | 0.39294 | 0.00132 | 0.31795 | 0.00146 | 0.39417 (n2cpv3) | pass |
| F5cd | 6 | **0.39426** | 0 | **0.31940** | 0 | 0.39589 (n2cpf) | pass |
| H | 3 | 0.39421 | 0.00005 | 0.31871 | 0.00069 | 0.39534 (n2cpf) | pass |

All four families pass the gate. F5 is the closest to the 0.002 limit (0.0018 on val), so F5 can drop out in phase B if F7 members are about 0.0002 stronger than F5cd.

## 3. Pools and the holdout choice (phase A)

| pool | members | FT suffixes | holdout TO half | Δ vs R/v11 | late 20% | val TO half | Δ TO half vs F5 (val) |
|---|---|---|---|---|---|---|---|
| R / v11 | 6 | n2cpv3 / n2cpf | 0.32162 | 0 | 0 | 0.39712 | 0 |
| Q1 | 12 | 12 v3 | 0.32239 | +0.00077 ± 0.00026 | +0.00105 ± 0.00056 | 0.39770 | +0.00057 ± 0.00046 |
| Q2 | 18 | 12 v3 + 6 f | 0.32292 | +0.00130 ± 0.00028 | +0.00151 ± 0.00060 | 0.39813 | +0.00101 ± 0.00049 |
| **Q3** | 21 | 12 v3 + 9 f | **0.32300** | **+0.00138 ± 0.00028** | +0.00140 ± 0.00061 | 0.39814 | **+0.00102 ± 0.00050** |

**Chosen: Q3** (highest holdout). Q3 and Q2 differ by only 0.00008 on the holdout, and the rule picks the maximum.

## 4. Val comparison and calibration (`phaseA.log`, `phaseA.json`)

Paired per-feed deltas, mean ± SE (n = 14,423). The weighted columns use se(d, w).
* priv M = calib M-only private weights; priv MxH4 = M bucket × MSK hour-4 cells.
* adv priv / adv pub = intrinsic adversarial weights `adv_{private,public}_intrinsic` from `solvers/out/advval/val_weights.npz`.
* Slice sizes: M<50 5,893; M≥50 8,530; Tue 7,543; Thu 6,880; all-cold-positive 2,510; all-warm-positive 10,230.
* The coldness flag is `eda/errors/synthesis/derived/val/cold_item.npy`, the same as sol_errors s14.

| pool vs ref | all | M<50 | M>=50 | Tue | Thu | all-cold | all-warm | priv M | priv MxH4 | adv priv | adv pub | z | folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Q3 vs R** | **+0.00061 ± 0.00046** | +0.00075 ± 0.00087 | +0.00052 ± 0.00050 | +0.00076 ± 0.00066 | +0.00045 ± 0.00065 | +0.00295 ± 0.00115 | +0.00013 ± 0.00057 | +0.00064 | +0.00057 | +0.00055 ± 0.00046 | +0.00071 ± 0.00047 | 1.33 | +.0014 +.0001 −.0002 +.0010 +.0008 |
| **Q3 vs v11** | **+0.00086 ± 0.00047** | +0.00145 ± 0.00090 | +0.00046 ± 0.00051 | +0.00087 ± 0.00066 | +0.00086 ± 0.00068 | +0.00175 ± 0.00115 | +0.00065 ± 0.00059 | +0.00088 | +0.00079 | +0.00086 ± 0.00047 | +0.00103 ± 0.00048 | 1.82 | +.0019 +.0002 +.0009 +.0006 +.0008 |
| Q2 vs R | +0.00060 ± 0.00046 | +0.00072 ± 0.00086 | +0.00051 ± 0.00049 | +0.00082 ± 0.00065 | +0.00035 ± 0.00064 | +0.00286 ± 0.00116 | +0.00020 ± 0.00056 | +0.00062 | +0.00064 | +0.00051 ± 0.00046 | +0.00061 ± 0.00046 | 1.30 | |
| Q2 vs v11 | +0.00085 ± 0.00047 | +0.00142 ± 0.00090 | +0.00045 ± 0.00050 | +0.00092 ± 0.00065 | +0.00076 ± 0.00068 | +0.00166 ± 0.00116 | +0.00072 ± 0.00058 | +0.00086 | +0.00086 | +0.00082 ± 0.00047 | +0.00093 ± 0.00048 | 1.80 | |
| Q1 vs R | +0.00028 ± 0.00043 | −0.00019 ± 0.00083 | +0.00059 ± 0.00045 | +0.00045 ± 0.00061 | +0.00008 ± 0.00061 | +0.00091 ± 0.00101 | +0.00017 ± 0.00054 | +0.00031 | +0.00028 | +0.00017 ± 0.00044 | +0.00033 ± 0.00044 | 0.64 | |
| Q1 vs v11 | +0.00053 ± 0.00045 | +0.00051 ± 0.00088 | +0.00053 ± 0.00047 | +0.00056 ± 0.00063 | +0.00049 ± 0.00065 | −0.00029 ± 0.00106 | +0.00069 ± 0.00057 | +0.00055 | +0.00050 | +0.00047 ± 0.00046 | +0.00065 ± 0.00046 | 1.16 | |

Calibrated levels, test correlation and gate results:

| | val | pub / priv (M) | pub / priv (M × hour-4) | test Pearson z vs v11 / v12(R) | test Spearman vs v11 / v12(R) | presub_gate |
|---|---|---|---|---|---|---|
| v11 | 0.39793 | 0.38978 / 0.39244 | 0.39017 / 0.39186 | 1 / 0.9998 | 1 / 0.9988 | |
| R = sub_v12 | 0.39818 | 0.39002 / 0.39268 | 0.39050 / 0.39208 | 0.9998 / 1 | 0.9988 / 1 | (PASS earlier) |
| Q1 | 0.39846 | 0.39035 / 0.39299 | 0.39078 / 0.39237 | 0.9965 / 0.9967 | 0.9922 / 0.9925 | PASS, 9 WARN |
| Q2 | 0.39878 | 0.39066 / 0.39330 | 0.39095 / 0.39272 | 0.9963 / 0.9964 | 0.9917 / 0.9919 | PASS, 18 WARN |
| **Q3** | **0.39879** | **0.39069 / 0.39332** | **0.39100 / 0.39266** | 0.9960 / 0.9960 | **0.9913 / 0.9914** | PASS, 21 WARN (vs R and vs v11) |

* M × hour-4 cells cover every test cell: missing target mass is 0. ESS is 13,268 (private) and 12,735 (public).
* My v11 M × hour private value of 0.39186 matches sol_advval's 0.39186.
* **presub_gate details** (`gate_A_Q3_vs_R.log`):
  * A: valid.
  * B: reproduction min per-feed Spearman 0.999995, which is float32 rounding.
  * C: 21 members audited, 0 FAIL.
  * D: no slice below −2 SE. The leak probe gives has_later minus no_later +0.0028 ± 0.0050 (117 feeds, n.s.).
  * E: predicted public 0.3907 / private 0.3933.
  * F: val/test Spearman parity 0.9909 / 0.9914.
* **All 21 WARNs are FR-2**, i.e. raw `prew_fe` / `prew_new` / `has:prew_fe` as train-stage inputs. They come from v12_allprew_s83-85, v15_cd_allprew_s83-85 and v14_mix1_s91. This is the same justification as sub_v12's 9 WARNs: a quality issue, not a leak.
* **File checks:** every test vector is float32, finite and 2,904,806 long (script asserts plus a re-check).

## 5. Frozen final-pick rule (`phaseA.log`, ACCEPT lines)

| criterion | Q3 vs R | Q3 vs v11 | Q1 vs R (final-eligible) |
|---|---|---|---|
| paired val > 2 SE | **no** (z 1.33) | **no** (z 1.82) | no (z 0.64) |
| no slice (M<50, M>=50, Tue, Thu) < −2 SE | yes | yes | yes |
| holdout TO half ≥ max(0.32162, R's holdout) | yes (0.32300) | yes | yes (0.32239) |
| presub_gate PASS (A-F) | yes (21 FR-2 WARN) | yes | yes (9 FR-2 WARN) |
| file checks | yes | yes | yes |
| FT halves all n2cpv3 | **no** (9 n2cpf) | **no** | yes |
| **result** | **not accepted** | **not accepted** | not accepted |

**Pick 2 by the frozen rule:**
* Pick 1 stays R. All pools are within 1 SE of R (all are above it).
* Lowest test rank correlation with R: Q3 0.9914 < Q2 0.9919 < Q1 0.9925. So the rule's pick-2 hedge is **Q3**.
* Q3 becomes a FINAL file only after the n2cpv3 fine-tunes of the 6 F5cd and 3 H members exist (a lead/NN-agent action; ~1.5 GPU-h):
  `EXTRA=n2c_pre_v3 SUF=n2cpv3 FTARGS='--decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton' bash nn/ft_list.sh v15_cd_all_e10 v15_cd_all_s81 v15_cd_all_s82 v15_cd_allprew_s83 v15_cd_allprew_s84 v15_cd_allprew_s85 v13_lr1_mix0 v13_lr2_mix1 v14_mix1_s91`
  Then rerun this script; it picks n2cpv3 up automatically. The lead copies the vector into preds/ and runs src/presub_gate.py.
* Public scores within 0.003 must never be used to choose.

## 6. Information-only decomposition (`info.log`; not used for any choice)

| comparison | val Δ | holdout TO Δ | reading |
|---|---|---|---|
| D4f (sol_ensemble D4, all n2cpf) − v11 | +0.00115 ± 0.00044 | +0.00128 ± 0.00027 | reproduces exp8 (0.39908) |
| D4v3 (same members, n2cpv3 where available) − v11 | +0.00093 ± 0.00044 | +0.00128 ± 0.00027 | switching F5's FT to n2cpv3 costs D4 −0.00022 |
| D4v3 − R | +0.00068 ± 0.00045 | +0.00128 ± 0.00027 | the honest pooling gain on the adopted basis |
| Q3 − (D4v3 + v14_mix1_s91), i.e. the effect of adding v14_s92..s96 | −0.00019 ± 0.00029 | +0.00007 ± 0.00016 | **F6 adds nothing on top of F5cd/H** |
| Q3 − Q3h (FT branch from the 12 n2cpv3 members only) | −0.00000 ± 0.00025 | 0 (same TO half) | **the F5cd/H fine-tuned halves add nothing at ensemble level** |
| Q3 − Q1 | +0.00034 ± 0.00035 | +0.00061 ± 0.00020 | F5cd+H add on top of F5+F6 |
| family-equal − member-equal: Q1 / Q2 / Q3 | +0.00003 ± 0.00004 / +0.00002 ± 0.00003 / +0.00027 ± 0.00018 | ~0 / ~0 / +0.00003 ± 0.00011 | identity verified at equal counts. For Q3, H gets 1/4 of the weight instead of 3/21; the holdout sees no difference |
| Q3 − nn_final5cd12 mix (frozen hedge) | +0.00006 ± 0.00039 | | |
| Q3 − D4 file (`solvers/out/ensemble/cand_D4_member_eq15_val.npy`) | −0.00029 ± 0.00032 | +0.00010 ± 0.00017 (vs D4v3) | |
| F5 FT branch alone, n2cpv3 − n2cpf | −0.00022 ± 0.00042 | | yet the R − v11 mix is +0.00025 |

Family-mix per-feed correlations on val are 0.984-0.989: F5|F6 0.9867, F5|F5cd 0.9886, F6|F5cd 0.9838, F5cd|H 0.9840. Family-mix val: F5 0.39818, F6 0.39848, F5cd 0.39828, H 0.39784.

What this shows:
1. **The earlier D4 z = 2.6 was partly fine-tune-preset noise.**
   * Swapping F5's fine-tune from n2cpf to n2cpv3 moves F5 alone by +0.00025 but D4 by −0.00022. The fine-tune A/A spread is 1.4x the per-feed SE (REVIEW_SYNTHESIS).
   * On the adopted n2cpv3 basis, cross-recipe pooling is worth about +0.0006..+0.0008 on val (z 1.3-1.8).
   * The train-only evidence does not depend on the fine-tune preset: holdout +0.0013..+0.0014 (z ≈ 5), TO-half val +0.0010 (z 2.0), and stale-val +0.00136 from sol_errors s17. That evidence is the robust part.
2. **F6 is neutral once F5cd/H are in.** This matches the strength-diversity law (sol_ensemble 2.5):
   * F6's member-mean gap to F5cd is 0.0013 on val and 0.0015 on the holdout, at ρ(F6, F5cd) 0.984.
   * The break-even gap is about 0.049(1−ρ) ≈ 0.0008, so F6 is past break-even.
   * The 0.002 gate is lenient enough to let F6 in.
   * (Suspected) F6 cooldown continuations (F6cd), like F5cd over F5, are what would add.
3. **Fine-tuning F5cd/H with n2cpv3 is expected to change val by ~0.** Q3h equals Q3 because the fine-tuned branch is saturated by the 12 n2cpv3 members. The ~1.5 GPU-h of fine-tunes buys formal eligibility, not score.
   * Zero-GPU alternative for the lead: Q3h is structurally all-n2cpv3, but it was **not pre-declared**. Declaring it now is a lead decision. Its val (0.39880) and holdout (0.32300) equal Q3's, so no val selection would be harvested.

## 7. Phase B (pending: F7 not produced yet)

* Run once every F7 member has `trainonly_{val,test}`, `n2cpv3_{val_oof,test}` and `preds/holdout/nn_<m>_holdout.npy` (run_final7b.sh writes all three):
  `cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 .venv/bin/python -u solvers/CPU1_xrecipe_member_equal_pool_v3.py --phase B > solvers/out/CPU1_xrecipe_member_equal_pool_v3/phaseB.log 2>&1`
* **What the script does:**
  * It stops with `phaseB_missing.json` if any F7 file is missing.
  * It applies the gate over F5/F6/F5cd/H/F7. Excluded families drop out of Q1B/Q3B; pools that become identical collapse.
  * It chooses between Q1B and Q3B on the holdout.
  * It reports everything above, plus paired val vs the phase-A choice.
* **Definition conflict the lead must resolve BEFORE running:**
  * This task says F7 = v17_pre2s_s101..s106 (the script default).
  * CURRENT_STATUS 19:50 pre-declares F7 = s101..s107; s107 runs on GPU N and is later, ~22:15-22:30.
  * If s107 belongs in F7, pass `--f7 v17_pre2s_s101,v17_pre2s_s102,v17_pre2s_s103,v17_pre2s_s104,v17_pre2s_s105,v17_pre2s_s106,v17_pre2s_s107`. Do not run both and pick one.
* **F5 gate risk:** F5 is 0.0018 behind F5cd on val. If F7's member-mean TO val beats 0.39447, F5 is excluded from both phase-B pools by the pre-declared gate.

## 8. Verified vs suspected

* **Verified (numbers above, files cited):**
  * R and v11 reproduce: 0.39818 / 0.39793, holdout 0.32162.
  * D4 reproduces 0.39908.
  * Member-equal equals family-equal at equal counts (Δ ≤ 0.00003).
  * All Q pools are positive on every slice vs R and vs v11, except Q1 M<50 (−0.0002) and Q1 all-cold vs v11 (−0.0003), both n.s.
  * The holdout gains are z 3.0-4.9.
  * No pool clears 2 SE on paired val.
  * The F5cd/H fine-tuned halves add 0.00000 ± 0.00025.
  * Gate PASS (FR-2 WARNs only).
  * Test vectors are finite and 2,904,806 long.
  * No writes to preds/ or subs/.
* **Suspected:**
  * n2cpv3 fine-tunes of F5cd/H would leave Q3's val within ±0.0003 (inferred from Q3 ≈ Q3h and the A/A spread).
  * Cooled F6/F7 continuations would add like F5cd did.
  * Phase B will be decided mostly by F7's strength relative to F5cd, given the gate and the law.
