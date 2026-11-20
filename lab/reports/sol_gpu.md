# sol_gpu: K9 production-order distillation arms (GPU N, nn_x/solvers)

Finished 2026-11-11 22:25 UTC. 8 train-stage runs at 2 epochs, paired by seed (23/24), on GPU N only. At most 3 of my
processes ran at once (rounds 1-2: 3; round 3: 2), next to the NN agent's v17_pre2s_s107.

## TL;DR
| arm | pooled train-only val (2 seeds) | pooled holdout | verdict |
|---|---|---|---|
| A `--aux_tau 30` | **-0.00091 +- 0.00049** (z -1.85); s23 -0.00099, s24 -0.00083 | **-0.00109 +- 0.00028** (z -3.86) | **WORSE**: Thu slice -2.45 SE; keep tau 10 |
| B `--aux_mode pl --aux_pl_k 20` | +0.00059 +- 0.00045 (z +1.31); s23 +0.00124, s24 -0.00006 | +0.00021 +- 0.00025 | **NULL** (fails the > 2 SE rule) |
| C = B + `--aux_pos 1.0` (pre-declared branch) | +0.00051 +- 0.00051 (z +0.99); s23 +0.00088, s24 +0.00013 | +0.00026 +- 0.00028 | **NULL**. vs B: -0.00008 +- 0.00048 |

* **No arm passes the unified rule.** The K9 recipe stays as it is: listwise, tau 10, aux_pos 0.5. Per the pre-declared
  plan, no 10-epoch confirmation pair is requested.
* None of the arms has ensemble value. The mixed pair (base + arm, other seed) vs 2x base gives val +0.00003..+0.00007,
  and the arms correlate more with the same-seed base than two base seeds correlate with each other.
* The CPU-2 stale-val replay vetoes nothing. It shows that B's small gain disappears under private-like staleness (below).
* Post-hoc only: B's gain is concentrated in M<20 feeds, where the top-20 PL target is the *full* production
  permutation. Long feeds are slightly negative. This is a hypothesis for a new mechanism, not a result (section 6).

## 1. Workspace and path audit (verified before any GPU use)
* Copy: `nn/*.py` + `nn/*.sh` -> `<repo>/nn_x/solvers/`. `feat_cache` is a symlink to
  `<repo>/nn/feat_cache`, used read-only.
* Redirects in the copy:
  - `common.py:24-26`: CKPT, LOGS and PREDS point to `nn_x/solvers/{ckpt,logs,preds}`.
  - `common.py:13-14`: the guard accepts only `CUDA_VISIBLE_DEVICES=N`.
  - `data.py:28`: FEAT_CACHE points to `nn_x/solvers/feat_cache`.
  - `data.py:36`: `feat_path` raises FileNotFoundError; the os.link/copy2 fallback is gone.
  - `data.py:40`: `feat_exists` checks the cache only.
  - `holdout_eval.py:17`: OUT = PREDS/holdout, i.e. `nn_x/solvers/preds/holdout`.
  - `pcmp.py:34,37`: reads from `nn_x/solvers/preds`.
  - `train.py:156`: variant names must start with `sx_`.
* Disabled: `ensemble.py`, `zmean.py`, `finetune.py`, `build_w3b_feats.py` and `make_cand.py` start with `raise SystemExit`,
  and every copied `.sh` exits at line 2.
* `grep -n "<repo>/preds\|nn/ckpt\|nn/logs" nn_x/solvers/*.py *.sh` leaves only docstrings and the disabled
  scripts. No write target remains.
* Dry check: all 34 names of `XT_PRESETS['cvnd_pre2s']` exist in `feat_cache/{train,val,test}`, and the logs list all 34
  as loaded.
* Post-run check:
  - 0 `sx_*` files exist in `preds/`, `preds/holdout`, `nn/ckpt` or `nn/logs`.
  - No file in `nn/feat_cache` is newer than the workspace setup.
  - The `nn/train.py`, `common.py` and `data.py` mtimes (17:37 / 16:27 / 06:30) predate this run.
* Deliberate deviation, inherited from the task: `--xt cvnd_pre2s` instead of cvnd, the nn_final7 family.

## 2. Arm B implementation and CPU unit test
* `nn_x/solvers/train.py` has three additions:
  - `pl_aux_loss` (`:61-70`), the task's formula: argsort by pos, logcumsumexp of the remaining set, top-K, divided by
    min(M, K);
  - `--aux_pl_k` (`:103`, default 20) and `pl` in the `--aux_mode` choices;
  - the branch at `:318-321`. It uses the same `aux_w(step)` anneal and the same `(1 - p_random)` feed weight as K9,
    and adds no RNG draws.
* The listwise (K9) and mse branches are textually unchanged (`diff nn/train.py nn_x/solvers/train.py` shows only
  additions).
* Data check: `pos` is unique within every one of 20,000 sampled train feeds. It has gaps (excluded viewed items), which
  PL ignores.
* `nn_x/solvers/test_pl_aux.py` passes (log: `nn_x/solvers/logs/test_pl_aux.txt`):
  1. Brute-force Python PL agreement on padded batches with M in {1, 2, 5, 19, 20, 21, 37, 64}, K in {1, 3, 20, 100}:
     max |loss diff| 4.8e-7 and max |grad diff| 1.5e-7. Padded slots get exactly zero gradient, and everything is
     finite.
  2. Invariance to extra padding and to a joint column permutation. Perfect-order loss is 0.
  3. Flag-off: the `if a.aux_pos > 0:` block executed from `nn/train.py` and from the copy gives **bit-identical**
     loss and gradients for listwise tau 10/30 (with and without p_random) and for mse.
  4. The `pl` branch equals the task's reference formula.
  - Tau 30 puts 29-35% of the target mass on slots 0-9 at M = 100-200, vs 63% at tau 10, as the task stated.
* Baseline and arms differ only in the appended treatment flags. Checked with `diff` of the logged args: only `variant`
  and `aux_tau` differ for A, `aux_mode` for B, and `aux_mode` + `aux_pos` for C.

## 3. Runs
Command: the task's baseline (`nn_x/solvers/run_sx_lane.sh`) plus the treatment flags:
`--muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.1 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2`.
That is 19,670 steps with the cooldown from step 15,736 (exploration pool 134,118 feeds).

| run | selected step | train-only val | holdout (holdout_eval) | final-step val / holdout (log) | max val in log |
|---|---|---|---|---|---|
| sx_base_s23 | 19670 (last) | 0.39375 | 0.31785 | 0.39375 / 0.31783 | 0.39375 |
| sx_base_s24 | 19670 (last) | 0.39288 | 0.31754 | 0.39288 / 0.31755 | 0.39356 |
| sx_tau30_s23 | 19670 (last) | 0.39276 | 0.31673 | 0.39276 / 0.31674 | 0.39285 |
| sx_tau30_s24 | 19670 (last) | 0.39206 | 0.31649 | 0.39206 / 0.31649 | 0.39227 |
| sx_pl20_s23 | 19670 (last) | 0.39500 | 0.31817 | 0.39500 / 0.31816 | 0.39500 |
| sx_pl20_s24 | 19670 (last) | 0.39282 | 0.31764 | 0.39282 / 0.31764 | 0.39299 |
| sx_pl20_w1_s23 | 19670 (last) | 0.39463 | 0.31813 | 0.39463 / 0.31813 | 0.39463 |
| sx_pl20_w1_s24 | 19670 (last) | 0.39302 | 0.31778 | 0.39302 / 0.31777 | 0.39302 |

* **Secondary readout (i), selection noise: none at this horizon.** All 8 runs selected the last step on the holdout,
  so the selected-step and final-step deltas are identical (logs `nn_x/solvers/logs/sx_*.json`).
* Side observation: the lr 1e-3 / mix 0.1 WSD cooldown improved monotonically to the end in all 8 runs, with no
  collapse. This agrees with CURRENT_STATUS 3k P1 (the collapse is caused by mix 0.3).
* Wall time: about 50 min per run with 4 processes on GPU N (rounds 1-2), 37 min with 3 (round 3).

## 4. Paired results (train-only val, pooled over seeds 23/24; per-feed SE; z in parentheses)
Source: `nn_x/solvers/logs/sx_eval_r12.txt`, `sx_eval_r3_vs_base.txt` and `sx_eval_r3_vs_pl20.txt`. They were
re-checked with the copied canonical `pcmp.py` (identical numbers). Commands:
`python nn_x/solvers/pcmp.py val sx_base_s23_trainonly_val:sx_X_s23_trainonly_val sx_base_s24_trainonly_val:sx_X_s24_trainonly_val`
and `... hold sx_base_s23:sx_X_s23 sx_base_s24:sx_X_s24`.

| slice | A tau30 | B pl20 | C pl20+w1 vs base | C vs B |
|---|---|---|---|---|
| all | -0.00091 +- 0.00049 (-1.85) | +0.00059 +- 0.00045 (+1.31) | +0.00051 +- 0.00051 (+0.99) | -0.00008 +- 0.00048 (-0.17) |
| M<50 | -0.00066 +- 0.00090 | **+0.00235 +- 0.00086 (+2.73)** | +0.00150 +- 0.00097 (+1.55) | -0.00085 +- 0.00092 |
| M>=50 | -0.00108 +- 0.00055 (-1.98) | -0.00063 +- 0.00047 (-1.32) | -0.00018 +- 0.00055 | +0.00045 +- 0.00051 |
| Tue | -0.00016 +- 0.00068 | +0.00096 +- 0.00062 | +0.00024 +- 0.00072 | -0.00071 +- 0.00067 |
| Thu | **-0.00173 +- 0.00071 (-2.45)** | +0.00019 +- 0.00066 | +0.00079 +- 0.00073 | +0.00061 +- 0.00070 |
| per-seed val all (s23 / s24) | -0.00099 / -0.00083 | +0.00124 / -0.00006 | +0.00088 / +0.00013 | -0.00036 / +0.00020 |
| holdout all | **-0.00109 +- 0.00028 (-3.86)** | +0.00021 +- 0.00025 | +0.00026 +- 0.00028 | +0.00005 +- 0.00027 |
| holdout late20% | -0.00080 +- 0.00060 | +0.00005 +- 0.00053 | +0.00141 +- 0.00059 (+2.38) | +0.00136 +- 0.00058 |
| per-seed holdout (s23 / s24) | -0.00112 / -0.00106 | +0.00033 / +0.00010 | +0.00029 / +0.00023 | -0.00004 / +0.00014 |

Notes:
* Per-seed slices below -2 SE: tau30 s23 Thu -0.00207 (z -2.16); pl20 s24 M>=50 -0.00161 (z -2.46). The pooled B
  M>=50 is -1.32 SE, so no pooled B or C slice is below -2 SE.
* The C late20% holdout z +2.38 is one slice among roughly 20 slice readouts. Holdout "all" is null, so I treat it as
  multiplicity, not a signal.
* Realised pooled SE: 0.00045-0.00051 on val (planned ~0.0006) and 0.00025-0.00028 on holdout.

### Acceptance (applied exactly as pre-declared)
* **A tau30: FAIL, verdict WORSE.**
  - Val is negative on both seeds; holdout is -3.9 SE and negative on both seeds; the Thu slice is -2.45 SE.
  - The stale-val veto is not triggered (-0.00077 +- 0.00049), but it points the same way.
  - Reading: moving target mass toward mid-list hurts, so the useful part of the production order is near the top.
    Tau 10 stays.
* **B pl20: FAIL the val rule (z 1.31 < 2), verdict NULL.**
  - Holdout >= 0 (+0.00021), and no pooled slice is below -2 SE.
  - Not adopted. No 10-epoch confirmation pair.
* **Round 3 branch (pre-declared, applied before C ran).**
  - Condition: max(z_A, z_B) = z_B = +1.31 >= 1.0, and B's pooled holdout is +0.00021 >= 0.
  - So C = B + `--aux_pos 1.0` (`sx_pl20_w1_s23/s24`), not the baseline-w1 branch.
* **C: FAIL vs base (z 0.99), verdict NULL.**
  - vs B: -0.00008 +- 0.00048, so a higher weight does not add on top of the PL target.
  - Not adopted.

## 5. Secondary readouts (information only)
**(ii) Ensemble level.** The comparison is z-mean(base_s23, X_s24) and z-mean(base_s24, X_s23) against
z-mean(base_s23, base_s24), with the 2x base pair at val 0.39433:

| arm | val | holdout |
|---|---|---|
| A tau30 | +0.00007 +- 0.00037 | -0.00049 +- 0.00021 |
| B pl20 | +0.00003 +- 0.00034 | +0.00009 +- 0.00019 |
| C | -0.00006 +- 0.00039 | +0.00004 +- 0.00021 |

* A 4-member ensemble (2 base + 2 arm) vs 2x base: A -0.00015, B -0.00005, C +0.00013 (all < 0.4 SE).
* Test z-correlation, same seed, arm vs base: tau30 0.985, pl20 0.989. Base s23 vs s24: 0.978.
* None of the arms adds diversity beyond a new seed.

**(iii) CPU-2 stale-val replay** (`solvers/out/CPU2_stale_replay_pre2s_and_arms/phaseC_{tau30,pl20}.stdout`; scenario
both_pre2both):

| arm | val delta in CPU replay | stale delta | diff-in-diff | veto |
|---|---|---|---|---|
| A tau30 | -0.00111 +- 0.00049 | -0.00077 +- 0.00049 | +0.00034 +- 0.00040 | none |
| B pl20 | +0.00027 +- 0.00045 | -0.00037 +- 0.00045 | -0.00064 +- 0.00039 | none |

* For B, the M<50 diff-in-diff is -0.00148 +- 0.00075. **B's M<50 gain mostly disappears under private-like coverage.**
* C was not replayed. The val rule already fails, so the veto is moot.
* **Measurement caveat (verified from the same files):** CPU-2 re-predicts the same checkpoints in fp32 on CPU. The
  stored predictions came from GPU bf16 autocast (corr 0.99999, max |logit diff| 0.54-0.68). Re-inference alone moves
  a per-seed delta by up to 0.0005: pl20_s24 is -0.00006 with GPU preds and -0.00060 in the replay; pooled B is
  +0.00059 vs +0.00027. Inference-precision noise is therefore not negligible next to 2-epoch deltas of ~0.0005. It
  affects all project comparisons equally (noise, not bias), but it argues against reading sub-0.001 single-seed
  deltas.

## 6. Post-hoc analysis (exploratory, NOT pre-declared, never used for acceptance)
Source: `nn_x/solvers/logs/sx_posthoc_r12.txt` and `sx_posthoc_r3.txt`.

| slice (val, pooled over 2 seeds) | B pl20 | C | PL family (avg of B and C per seed) | A tau30 |
|---|---|---|---|---|
| M<20 (n=2548) | +0.00464 +- 0.00148 | +0.00272 +- 0.00160 | +0.00368 +- 0.00132 | +0.00035 +- 0.00150 |
| 20<=M<50 (n=3345) | +0.00061 +- 0.00102 | +0.00057 +- 0.00119 | +0.00059 +- 0.00096 | -0.00143 +- 0.00111 |
| M>=100 (n=5348) | -0.00075 +- 0.00050 | -0.00027 +- 0.00061 | -0.00051 +- 0.00048 | -0.00097 +- 0.00064 |
| user 0 (n=658) | -0.00368 +- 0.00192 | n/a | n/a | -0.00078 +- 0.00223 |

* Reading: the PL gain sits where K=20 >= M, so the aux target there is the full production permutation (exact order
  of every candidate). On long feeds, top-20 PL is slightly worse than K9. One candidate *new mechanism* is an
  M-adaptive aux target: full-permutation PL for M <= 20 and K9 tau 10 otherwise.
* I do **not** recommend spending GPU on it now:
  1. The M<20 slice was found by looking. It is 1 of 7 slices, so it survives Bonferroni-7 only marginally
     (z 3.1, p ~0.014 after correction).
  2. M<50 is holdout-blind, because the holdout has M>=50 only.
  3. The stale-val replay shows the M<50 PL gain shrinking under private-like coverage (diff-in-diff -0.00148 +- 0.00075).
  4. The val fine-tune half of every family likely absorbs short-feed effects.
* If it is ever tried, it needs its own pre-declared 2-seed pair, evaluated on M<50 as a pre-declared primary slice
  together with stale-val.

## 7. Conclusions for the lead
1. The K9 channel appears locally optimal along all three tested dimensions:
   - tau: 10 beats 30 clearly (holdout -3.9 SE);
   - target form: PL top-20 ~ K9;
   - weight: PL at 0.5 ~ 1.0.
   Keep `--aux_mode listwise --aux_tau 10 --aux_pos 0.5` in all families. No 10-epoch confirmation pair is warranted.
2. None of these arms gives ensemble diversity, so they are not hedge members either.
3. The 2-epoch recipe (lr 1e-3, mix 0.1) cools down cleanly (8/8 runs select the last step). That makes it a usable
   cheap A/B harness for future train-stage probes.
4. Inference precision (GPU bf16 vs CPU fp32) moves per-seed deltas by up to ~0.0005. Treat sub-0.001 single-seed
   differences as noise.

## Files
* Code:
  - `<repo>/nn_x/solvers/train.py` (pl_aux_loss `:61`, flag `:103`, branch `:318`)
  - `<repo>/nn_x/solvers/test_pl_aux.py`
  - `<repo>/nn_x/solvers/run_sx_lane.sh`
  - `<repo>/nn_x/solvers/sx_eval.py`
* Checkpoints: `<repo>/nn_x/solvers/ckpt/sx_{base,tau30,pl20,pl20_w1}_s{23,24}.pt`
* Predictions:
  - `<repo>/nn_x/solvers/preds/nn_sx_*_trainonly_{val,test}.npy`
  - `<repo>/nn_x/solvers/preds/holdout/nn_sx_*_holdout.npy`
* Logs:
  - `<repo>/nn_x/solvers/logs/sx_*.{log,json,stdout}`
  - `sx_eval_r12.txt`, `sx_eval_r3_vs_base.txt`, `sx_eval_r3_vs_pl20.txt`, `sx_posthoc_r12.txt`, `sx_posthoc_r3.txt`,
    `test_pl_aux.txt`, `sx_eval_*.json`
