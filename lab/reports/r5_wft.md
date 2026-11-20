# r5_wft: nested-CV check of the fine-tuned share w_ft for Cd (SOLUTIONS_REVIEW roadmap #9)

2026-11-11 ~23:10 UTC. CPU only. Read-only on preds/ and subs/. Nothing under nn/, src/, preds/ or subs/ was written.

## Verdict: keep w_ft = 0.5 (pre-declared rule NOT met)

- Primary folds: **all 5 folds choose w = 0.67**. But the nested delta vs w = 0.5 is **+0.00024 ± 0.00025 (z +0.96)**, below the required z ≥ 2.5. Result: **keep 0.5.**
- Robustness folds (the n2cpv3 fine-tune folds, user-0 singletons): the choices **flip**: 0.67 / 0.67 / 0.67 / 0.33 / 0.33. The nested delta is **−0.00021 ± 0.00024 (z −0.86)**.
  - Moving 658 user-0 feeds between folds is enough to change the sign. So the 5/5 agreement on the primary folds is not a stable signal.
- No change to Cd, sub_v13 or any pick.

## Setup (pre-declared before any w ≠ 0.5 score was computed)

- Pre-declaration: `solvers/out/r5_wft/PREDECLARATION.txt` (written 23:05, before the script ran).
- Code: `solvers/r5_wft.py`.
- Outputs: `solvers/out/r5_wft/r5_wft.{log,json}`.
- Command: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 POLARS_MAX_THREADS=16 .venv/bin/python solvers/r5_wft.py`.

**Members.** The 19 Cd members:
- F5cd: v15_cd_all_e10 / s81 / s82, v15_cd_allprew_s83 / s84 / s85;
- F6: v13_lr1_mix1, v14_s92..s96;
- F7: v17_pre2s_s101..s107.

**Branches** (`r5_wft.py:41`):
- ft = z(member-equal mean of per-feed z of `preds/nn_<m>_n2cpv3_val_oof.npy`);
- to = the same over `preds/nn_<m>_trainonly_val.npy`;
- per-feed z = `src/stack_val.feed_rank_z`, as in the decide.py that built Cd.

**Final score:** s(w) = w z(ft) + (1−w) z(to). Metric: `src/metrics.ndcg_per_feed` (pessimistic ties).

**Sanity check.** s(0.5) reproduces Cd val **0.39901** exactly (asserted at `r5_wft.py:117`). The single branches score:
- ft alone: 0.39889;
- to alone: 0.39877.

**Nested rule** (`r5_wft.py:84-92`):
- Grid {0.33, 0.5, 0.67}.
- For each outer fold, choose the w with the best mean per-feed NDCG on the other 4 folds. Ties go to 0.5.
- Score the held-out fold with the chosen w.
- Delta = nested score − s(0.5), paired per feed. SE = std/√n. The user-clustered SE is also given.

**Folds** (`r5_wft.py:73`):
- PRIMARY = GroupKFold(5) by user_id, with user 0 as one group (the ens_common fold definition; the task's wording).
- ROBUSTNESS (report only) = the same with user-0 feeds as singleton groups, i.e. the n2cpv3 fine-tune folds (`nn/finetune.py:197-201`).

**Adoption** (`r5_wft.py:135`): all 5 primary folds choose the same w ≠ 0.5 **and** the nested z ≥ 2.5.

## Results

### Per-fold choices (delta vs w = 0.5)

| fold | n | train-folds d(0.33) | train-folds d(0.67) | chosen | held-out d(0.33) | held-out d(0.67) |
|---|---|---|---|---|---|---|
| **PRIMARY** | | | | | | |
| 0 (all 658 user-0 feeds) | 2885 | −0.00009 | +0.00012 | 0.67 | +0.00066 | +0.00073 |
| 1 | 2885 | +0.00021 | +0.00029 | 0.67 | −0.00055 | +0.00005 |
| 2 | 2885 | +0.00006 | +0.00007 | 0.67 | +0.00006 | +0.00092 |
| 3 | 2884 | +0.00004 | +0.00025 | 0.67 | +0.00015 | +0.00018 |
| 4 | 2884 | +0.00008 | +0.00047 | 0.67 | −0.00002 | −0.00068 |
| **ROBUSTNESS (u0 singletons)** | | | | | | |
| 0 | 2885 | +0.00007 | +0.00040 | 0.67 | +0.00004 | −0.00042 |
| 1 | 2885 | −0.00006 | +0.00035 | 0.67 | +0.00053 | −0.00020 |
| 2 | 2885 | −0.00002 | +0.00024 | 0.67 | +0.00037 | +0.00023 |
| 3 | 2884 | +0.00021 | +0.00017 | **0.33** | −0.00052 | +0.00050 |
| 4 | 2884 | +0.00011 | +0.00003 | **0.33** | −0.00012 | +0.00109 |

All training-fold margins are ≤ 0.00047. The in-sample SE of a fixed-w delta is ~0.00022-0.00025, so every choice sits inside one SE.

### Nested delta vs w = 0.5, with slices (mean ± SE, z)

| folds | all | M<50 | M≥50 | Tue | Thu | M-private-weighted | user-clustered SE |
|---|---|---|---|---|---|---|---|
| PRIMARY (5 × 0.67) | **+0.00024 ± 0.00025 (+0.96)** | −0.00034 ± 0.00043 (−0.79) | +0.00064 ± 0.00030 (+2.14) | +0.00031 ± 0.00031 (+1.02) | +0.00016 ± 0.00040 (+0.39) | +0.00026 ± 0.00025 (+1.06) | 0.00025 |
| ROBUSTNESS | −0.00021 ± 0.00024 (−0.86) | −0.00068 ± 0.00040 (−1.69) | +0.00012 ± 0.00030 (+0.40) | −0.00003 ± 0.00030 (−0.09) | −0.00040 ± 0.00038 (−1.06) | −0.00019 ± 0.00024 (−0.80) | 0.00024 |

### Information only: in-sample curves (never used for the decision)

**Full val, fixed w** (`r5_wft.log`, "INFORMATION ONLY" block):

| w | 0 | 0.1 | 0.2 | 0.3 | 0.33 | 0.4 | **0.5** | 0.6 | 0.67 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val NDCG | 0.39877 | 0.39881 | 0.39879 | 0.39891 | 0.39907 | 0.39904 | **0.39901** | 0.39920 | 0.39925 | 0.39920 | 0.39911 | 0.39903 | 0.39889 |

- The curve is flat within ±0.00025 over w ∈ [0, 1]. It is also non-smooth (0.30 → 0.33 jumps by +0.00016), which looks like noise.
- The w = 0.67 gain is all on M≥50 (+0.00064, z 2.1). M<50 is −0.00034.
  - This is the same M split sol_ensemble.md found on F5. There, M-specific weights failed to replicate across pools (sol_ensemble.md:37).
- Staleness gives no reason to raise w_ft:
  - Thu, the staler val day: d(0.67) is +0.00016, smaller than Tue's +0.00031.
  - Feed cold-share terciles (label-free, it_hist_all_luo == 0; `scratchpad/round5/wft/cold_info.log`): d(0.67) is +0.00055 / −0.00004 / +0.00020 for the low / mid / high-cold terciles, so there is no gradient toward cold feeds.
  - Stale-val cannot test this: the CPU2 replay rebuilds train-only halves only, and no fine-tuned predictions exist at stale coverage.
- **Per family, d vs 0.5 (± ≈ 0.00023):**

  | family | d(w = 0.33) | d(w = 0.67) | d(w = 1.0) | direction |
  |---|---|---|---|---|
  | F5cd | −0.00030 | +0.00026 | +0.00054 | FT-leaning |
  | F6 | −0.00034 | −0.00006 | −0.00051 | 0.5 best |
  | F7 | +0.00002 | −0.00018 | −0.00025 | to-leaning |

  The families disagree, which is consistent with "FT adds less as members improve" (SOLUTIONS_REVIEW §3). But no single direction holds for the pool.

## Consequences

- w_ft = 0.5 stays for Cd and for all pre-declared candidates (Cf / Cg / Ch).
- Do not reopen w_ft or M-segment w_ft:
  - the earlier F5 check chose 0.5 in 5/5 folds (sol_ensemble.md:36);
  - this Cd check chose 0.67 in 5/5 folds on one fold assignment, but 3/2 on another;
  - both are inside noise (|z| < 1).
- Submission now: `subs/sub_v13_Cd_memeq_F5cdF6F7.parquet` (byte-identical to `subs/cand_Cd_memeq_F5cdF6F7.parquet`; `cmp` checked).
  - presub_gate PASS with 7 justified WARN (CURRENT_STATUS 3o/3p).
  - Predicted public 0.3909 ± 0.003 (SOLUTIONS_REVIEW §4.1).
  - It is the planned public sanity submission. This check gives no reason to build a w = 0.67 variant.
