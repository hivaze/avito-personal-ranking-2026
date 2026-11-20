# ROUND 5: staleness interventions and the fine-tune combiner (synthesis)

2026-11-12 ~00:40 UTC. Three pre-declared experiments: age-conditioned input dropout, full-val refit, and the w_ft nested check. Detail reports: `reports/r5_agedrop.md`, `reports/r5_refit.md`, `reports/r5_wft.md`.

## TL;DR

- **All three FAIL their pre-declared rules. Nothing changes:** picks, Cd, test files, w_ft and the overnight plan all stay as they are.
- **What you can submit right now to check:** `subs/sub_v12_nnfinal5v3mix.parquet` (pick 1). It is the only gated, ready file that has never been on public (details in §1).
- Round 5 produced no new prediction or submission file. Nothing was written to `preds/` or `subs/`.

## 1. Answer to "what can I submit just now to check?"

- **Correction:** the agedrop and wft agent answers call Cd "prepared, not yet submitted". That information is stale. `CURRENT_STATUS.md` 3p (23:20) records `sub_v13_Cd_memeq_F5cdF6F7` = **0.39208 public** (predicted 0.3909), so the check is already done and passed.
- **Submit: `subs/sub_v12_nnfinal5v3mix.parquet`**, if you have a slot.
  - Purpose: confirm that the pick-1 file is built correctly on test. It is not for choosing between v12 and Cd.
  - Gate: presub_gate PASS with 9 justified WARN (3m).
  - Expected public about 0.391. Two estimates agree:
    - Cd's public score minus the val gap: 0.39208 − 0.0008;
    - calib prediction 0.3900 plus the mean calibration residual +0.0012.
  - **Alarm only if it scores below 0.387** (0.3900 − 0.003). Then check row alignment and the blend spec before the final picks.
  - **Do not swap picks on the result.** The protocol forbids picking on public differences < 0.003. Public SE is about 0.0012, while the v12-to-Cd val gap is +0.00083 ± 0.00054.
- No newer candidate exists. The newest file in `subs/` is 23:02, and Cf / F9 / F10 / F11 / F12 are not built or gated yet.
- Verified: `cmp subs/sub_v13_Cd_memeq_F5cdF6F7.parquet subs/cand_Cd_memeq_F5cdF6F7.parquet` shows the two files are identical.

## 2. Experiments

### 2a. Age-conditioned ID-input dropout (review_ideas #6): FAIL

- **Rule** (frozen 23:15, before launch; `r5_agedrop.md` §0). Paired 2-epoch A/B on the cvnd_pre2s harness, seeds 23 and 24 pooled per feed.
  - Control: `sx_base_s23/s24`. Treatment: `--agedrop prv_rl13`.
  - PASS iff all three hold:
    - stale-val (CPU2 `both_pre2both`) > 2 SE;
    - val ≥ −1 SE;
    - holdout ≥ −1 SE.
- **Numbers:**

  | yardstick | delta | z | rule |
  |---|---|---|---|
  | stale-val (primary) | +0.00089 ± 0.00062 | 1.44 | FAIL |
  | val | +0.00005 ± 0.00061 | 0.08 | pass |
  | holdout | −0.00189 ± 0.00034 | −5.58 | FAIL |

  - Per seed:
    - stale-val: s23 +0.00092, s24 +0.00085;
    - holdout: s23 −0.00166, s24 −0.00212.
  - Slices:
    - stale-val: M<50 +0.00316 ± 0.00116, M≥50 −0.00068, Tue +0.00188, Thu −0.00020;
    - holdout late-20%: +0.00031 ± 0.00075.
- **Mechanism.** The dropout works as intended: stale-minus-val difference +0.00121 ± 0.00047 (z 2.6). But it trades warm accuracy for cold:
  - val feeds whose positives are all young: +0.0247 ± 0.0019 (n 2,418);
  - feeds with no young positive (72% of feeds): −0.0063 ± 0.0007 (n 10,392).
  - This is the same cold-for-warm trade seen before with it_hist_hz and with uniform xt_drop.
- **Adversarial checks (re-run by me):**
  - The numbers in `nn_x/solvers/logs/sx_eval_base_agedrop.txt` and `solvers/out/CPU2_stale_replay_pre2s_and_arms/phaseC_agedrop.log` match the table.
  - Treatment vs control checkpoints: `cfg` identical; `args` differ only in `agedrop` (None → prv_rl13) and `variant`. Checked by loading both .pt files.
  - Replay check (d): corr ≥ 0.999991 for all 4 checkpoints.
  - Unit test: `test_agedrop.log` ALL PASS. The realised drop rate per bucket equals p, e.g. 0-6h 0.961 / 0.961, 14-30d 0.162 / 0.164.
  - The probabilities (`measure.json`, 23:08) were fixed before the 23:15 launch.
  - **Caveat:** the CPU replay and the stored GPU predictions disagree by 0.00038 on the val delta (−0.00033 vs +0.00005), about 0.6 SE of tool noise.
    - On the stored-GPU basis, stale-val is +0.00126 ± 0.00063 (z 1.99), right at the threshold.
    - The verdict does not depend on this: the holdout fails on its own at z −5.6.
  - The treatment checkpoint is holdout-selected at step 19000, the control at 19670 (final). At the final step the deltas are the same (val −0.00045 / +0.00062; holdout −0.00165 / −0.00214), so step selection does not drive the result.
  - Post hoc, not actionable: an ensemble of 1 base + 1 agedrop model vs 2 base models gives holdout −0.00050 ± 0.00024 (late-20% +0.00107 ± 0.00054).
- **Lead action:**
  - There is no 10-epoch family spec, as pre-declared on a FAIL.
  - Do not add `sx_agedrop_*` to any pool.
  - Close #6 for this round. A young-only or milder variant would need a new pre-declaration and is not recommended.
  - The val fine-tune half remains the staleness lever.

### 2b. Full-val refit of the fine-tune (review_ideas #8): FAIL

- **Rule** (`scratchpad/round5/refit/PREREG.txt`, 23:06:39, before the first GPU call at 23:07:37).
  - Nested simulation over the production u0-singleton GroupKFold(5).
  - 3 members: v15_cd_all_e10, v14_s93, v17_pre2s_s102.
  - Variant x passes iff the pooled delta vs (i) > 2 SE **and** every member's delta ≥ 0.
  - Variants:
    - (i) mean of 5 inner fold models;
    - (ii) one refit on the whole fit set;
    - (iii) 0.5 (i) + 0.5 (ii).
- **Numbers** (n 14,423; `solvers/out/r5_refit/nested.json`, `decision.choice = "none (keep 5-fold mean)"`):

  | vs (i) | pooled | z | members v15cd / v14 / v17 | rule |
  |---|---|---|---|---|
  | (ii) refit | −0.00025 ± 0.00018 | −1.37 | −0.00028 / −0.00054 / +0.00007 | FAIL |
  | (iii) blend | −0.00002 ± 0.00012 | −0.18 | +0.00019 / −0.00018 / −0.00007 | FAIL |

  - No slice reverses the result. Slices for (ii): M<50 −0.00022, M≥50 −0.00027, Tue −0.00022, Thu −0.00029, cold tercile −0.00013.
- **Adversarial checks:**
  - Independent replication at no cost: the production fold-k model was trained on the same 80% as (ii). It is also below (i): −0.00052 ± 0.00017 (z −3.1), negative on all 3 members and every slice.
  - A/A test, (ii) minus the production fold-k model: +0.00027 ± 0.00019, consistent with seed noise.
  - Parity: the copied recipe reproduces production fold 0 (corr 1.000000, NDCG 0.40047 vs 0.40065; `nn_x/solvers/logs/r5_refit_check.log`).
  - Declared caveat: the simulation compares 64% vs 80% of val, while production would compare 80% vs 100%. With diminishing returns the production effect would be smaller, so the FAIL direction holds.
  - Post hoc: a 2-seed refit is still −0.00022 ± 0.00016.
  - `ls preds/ | grep -E "n2cpv3full|refit|agedrop"` returns nothing.
- **Lead action:**
  - Every member of Cd and of the next candidates keeps `preds/nn_<v>_n2cpv3_test.npy`, the 5-fold mean.
  - No `n2cpv3full` files exist, and none should be made.
  - Close #8, including the "refit with several seeds" version.
  - If anything, more fold or seed averaging of the fine-tune is the lever, but that is untested.

### 2c. w_ft nested check (SOLUTIONS_REVIEW roadmap #9): FAIL, keep 0.5

- **Rule** (`solvers/out/r5_wft/PREDECLARATION.txt`, 23:05:41, before outputs at 23:06:24).
  - Grid {0.33, 0.5, 0.67}; primary folds = GroupKFold(5) by user with user 0 as one group.
  - Adopt w ≠ 0.5 only if all 5 folds choose the same w **and** the nested z ≥ 2.5.
- **Numbers** (`r5_wft.json`):
  - Sanity: s(0.5) reproduces Cd val exactly (0.399013).
  - Primary folds: all 5 choose 0.67, but the nested gain is +0.00024 ± 0.00025 (z 0.96). `adopt = False`.
  - Robustness folds (the fine-tune folds): choices are 0.67 ×3 and 0.33 ×2; nested −0.00021 ± 0.00024 (z −0.86).
  - The in-sample curve over w ∈ [0, 1] is flat within ±0.00025.
- **Adversarial checks:**
  - Because all 5 primary folds chose 0.67, the "nested" gain equals the in-sample gain at 0.67 exactly (0.39925 − 0.39901). The nested procedure therefore adds no independent evidence.
  - The sign flips when 658 user-0 feeds are spread across folds.
  - The families disagree: F5cd leans fine-tune (+0.00026 at 0.67), F6 is best at 0.5, F7 leans train-only (−0.00018).
  - Staleness gives no reason to raise w:
    - Thu (the staler day) gains less than Tue;
    - by cold-share tercile, there is no gradient (+0.00055 / −0.00004 / +0.00020).
- **Lead action:** keep w_ft = 0.5 for Cd and for Cf / Cg / Ch and the overnight pools. Do not build a 0.67 file. Do not reopen w_ft or a w_ft per M segment.

## 3. Housekeeping

- No round-5 process is running: `ps` shows no agedrop, refit_ft, r5_ or wait_and_run processes, and `RUNNING_agedrop.txt` says it finished at 00:27.
- GPU N is back to the lead's jobs.
- Two incidents, both already handled:
  - the 23:15 agedrop launch hit an out-of-memory error during data load and was relaunched staggered;
  - the refit driver was killed by the harness after it was paused, and the lost member s102 was re-run with the same seeds.
- **Suggested CURRENT_STATUS line** (lead to paste):
  > 3q 00:40 ROUND 5 DONE (reports/ROUND5.md): agedrop FAIL (stale +0.0009 z 1.4, holdout −0.0019 z −5.6; warm/cold trade); full-val refit FAIL (5-fold mean beats a single refit, −0.00025 z −1.4; closes #8); w_ft nested FAIL (z 0.96, flips with fold split) -> keep 0.5. No new files; picks unchanged (v12, Cd). Optional public check: v12 (alarm < 0.387).
