# r7 stage-1 screen: 2-epoch paired screen of the architecture-block and hyperparameter arms

Written 2026-11-12 by the r7 screen agent. Inputs: CURRENT_STATUS (to the 04:55 line), r7_research.md, r7_build.md,
R6_PLAN.md, nn_x/r7 (code frozen below), nn_x/r6/queue_morning.sh (read-only).
**§1 was written and frozen at 05:25 UTC, before any r7 screen run was launched.** §2-§5 were added afterwards.

---------------------------------------------------------------------------------------------------------------------

## 1. PRE-DECLARATION (frozen before launch)

### 1.1 Code under test (md5 at freeze)
| file | md5 |
|---|---|
| nn_x/r7/model.py | 23dddd1ecfd34c37a026df3aefa9df82 |
| nn_x/r7/train.py | 71c2decdb21b79fd9707720ed394c2d1 |
| nn_x/r7/data.py | 121301e948d5ee0894f61e241f76a6ac |
| nn_x/r7/common.py | 8e0a7319275bbf8624aa35ebe3f8344a |
| nn_x/r7/screen.sh | 723c845a1459f7a5f7eca83ca93ba0fa |
| nn_x/r7/eval_screen.py | 9ff4c7eb3c27f12a6a1047f9307215f4 |

Build, tests (321/321 GPU checks) and smoke runs: reports/r7_build.md. Every arm sits behind a flag whose default
reproduces nn/ bit for bit.

### 1.2 What runs
- **Stage-1 config** (`screen.sh flags base`, = r6 BASE2 / sx_base): `--no_pos --aux_pos 0.5 --aux_mode listwise
  --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0
  --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.1 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2
  --eval_every 1000 --no_test --save_holdout`. Treatment flags are appended last.
- **Seeds 23 and 24.** The baseline `r7_base_s23/s24` is re-run in this code copy, first in the same batch of launches.
- **Arms** (flags appended to the stage-1 config; specs in r7_research §5, implementation in r7_build §2):

  | arm | flags | kind |
  |---|---|---|
  | muon_in | `--muon_inputs adamw` | optimiser partition (never varied) |
  | batch2s | `--cand_budget 65536 --max_feeds 1024 --lr 1.414e-3 --warmup 250 --eval_every 500 --muon_momentum 0.90 --adam_betas 0.8,0.96` | batch size (never varied) |
  | resdrop0 | `--res_drop 0` | residual / FFN / retrieval dropout (never varied) |
  | pbld | `--pbld 16` | tabular input encoder (RealMLP PBLD) |
  | hc4 | `--hc 4` | residual scheme: dynamic hyper-connections |
  | dwa | `--dwa` | residual scheme: DenseFormer depth-weighted average |
  | diff2 | `--diff_attn cross` | attention: differential target attention |
  | swiglu | `--ffn swiglu` | FFN form |
  | derf | `--norm derf` | norm-free (Derf instead of LayerNorm) |
  | hl2 | `--hist_extra 1` | history depth (never varied under Muon) |
  | periln | `--periln` | norm placement: Peri-LN |
  | ftok | `--ftok 8` | tabular input encoder (feature-group tokens) |
  | postln (reserve) | `--postln` | norm placement: Post-LN + DeepNorm |

- **lr rule for the batch arm (pre-declared in r7_research §4.3): SDE square-root rule.** batch2s doubles the batch
  (65,536 candidates / 1,024 feeds), so lr × √2 (1e-3 → 1.414e-3) and (1 − β) × 2 for every momentum (Muon 0.95 → 0.90;
  AdamW (0.9, 0.98) → (0.8, 0.96)). Warmup (500 → 250 steps) and the eval interval (1000 → 500 steps) stay constant in
  samples. No other arm changes the batch size or the lr.
- **Launch order (29 runs):** r7_base s23, s24; then muon_in, batch2s, resdrop0, pbld, hc4, dwa, diff2, swiglu, derf, hl2,
  periln, ftok (s23 then s24 each); then the A/A rerun `r7_base_s23r` (R7_SUFFIX=r); then postln s23 / s24.
- **Not in this screen:**
  - the r6-owned arms (a4u, cset, typed, adrop0, a4w; r6 dropped wd01). nn_x/r6/queue_morning.sh has been running them
    on GPU N against r6's own baseline since 04:51 UTC, and CURRENT_STATUS (04:55) says r7 must not run them. They are
    not duplicated. Their result is read from nn_x/r6/logs/r6_eval_screen.txt and quoted in §4 as a cross-reference
    only; it is not an r7 decision.
  - drop0 (stage-2-only combination, needs r6 adrop0 to promote too).
  - cooldown_frac, aux-anneal target, pool mix, wd: horizon-dependent knobs owned by r6 Q1 / the continuation harness
    (r7_research §4.2).

### 1.3 Scheduling under the GPU rules
- Driver: `nn_x/r7/queue_r7.py` (detached; PID in nn_x/r7/logs/RUNNING.txt). It launches `screen.sh <gpu> <arm> <seed>`
  detached on a GPU only if both hold:
  - **process cap:** ≤ 3 compute processes already on the GPU, so ≤ 4 with ours. The count includes our own launched
    runs that are not yet visible to nvidia-smi and one slot per live r6 lane;
  - **memory headroom:** effective free − need ≥ 80 GB, where need = 62 GB (72 batch2s, 66 hc4 / ftok). Effective free
    also deducts the not-yet-claimed part of our own runs (need − current use) and 60 GB per live r6 lane that is
    between runs.
- screen.sh repeats its own check under a per-GPU flock and refuses (exit 3) if the GPU no longer qualifies; the job is
  then re-queued.
- GPU preference <ids>, then 0, 1, 2, 5, 3 (GPU N last because r6's screen lanes use it). The two seeds of an arm go
  to different GPUs where possible.
- **batch2s cost pairing:** batch2s s<seed> is held up to 55 min after r7_base_s<seed> starts, for that base run's GPU.
  After that it goes to any GPU, and its COST verdict is "undetermined" (eval_screen.py rule).
- **Crashes:** a failed run (no json / val preds / holdout preds) is relaunched once, with its old log moved to
  `nn_x/r7/logs/failed/`. A second failure is reported as "no verdict".
- Only our own jobs are ever stopped, by PID. Nothing under nn/, src/, preds/, subs/, nn_x/r6/ or nn_x/llm/ is written.

### 1.4 Metric and statistics (r7_build §4, unchanged)
- Per-feed full-list NDCG (src/metrics.ndcg_per_feed) of the train-only predictions at the holdout-selected checkpoint.
  - val: 14,423 val feeds;
  - holdout: the fixed 30K random-like M ≥ 50 train feeds (a selector, not independent evidence).
- Paired per-feed delta per seed = NDCG(arm_s) − NDCG(base_s); pooled = per-feed mean over the 2 seeds.
- **Calibrated SE** = sqrt(SE_feed² + σ_run² / n_seeds), with stage-1 σ_run = 0.00035 on val and 0.00015 on the holdout
  (same-seed GPU non-determinism of ±0.0002-0.0005 at 2 epochs). Slices use SE_feed.
- Slices: val M<50, M≥50, Tue, Thu (and coldpos / warmpos for information); holdout late 20%.
- **A/A step (before reading any arm):** `eval_screen.py aa` on r7_base_s23r vs r7_base_s23.
  - Fork check: |Δval| ≤ 0.0010 and |Δhold| ≤ 0.0005, else STOP and audit the copy.
  - Calibration: if |Δ| > 2·sqrt(SE_feed² + σ²) on val or holdout, re-run `eval_screen.py screen` with
    `--sigma_val` / `--sigma_hold` raised to the A/A |Δ| on that metric, and use those verdicts.

### 1.5 Stage-1 PROMOTION rule (applied exactly as coded in eval_screen.py `screen`)
PROMOTE iff all of:
1. pooled val Δ > 1.5 SE_cal;
2. pooled holdout Δ ≥ 0;
3. both seeds' val Δ > 0 **or** both seeds' holdout Δ > 0.

batch2s, additionally (a cost lever, not a promotion): **COST-PASS** iff pooled val Δ ≥ −0.0005, holdout Δ ≥ −0.0003, and
train minutes ≤ 0.75 × the paired baseline on the same GPU with start times within 60 min, for every comparable seed. No
comparable pair → "COST undetermined".

Information only (not part of the rule): WARN slices < −2 SE_feed; SCALE-SUSPECT (an arm's |final − init| growth >
1.25 × the baseline's on ≥ 2 of top sim-bias scale, c_ln gain, retr.a, cross0 W_O RMS); diversity (rho_bb, rho_x,
mixed 2-member ensemble); FAMILY-CANDIDATE; end − selected decline; train minutes and peak memory.

### 1.6 Combinations (stage 1; at most 2)
- Only if ≥ 2 arms PROMOTE individually. Rank the promoted arms by pooled val z.
  - **C1** = the two highest-ranked mutually compatible arms.
  - **C2** = the top arm with the third-ranked compatible promoted arm, if one exists. Otherwise there is no C2.
  - Incompatible (asserted in the code): hc4 / dwa / postln with each other; postln with swiglu / derf / periln. A
    batch2s COST-PASS is not a promotion and does not enter combinations.
- Each combination runs 2 seeds (23, 24) with the same stage-1 config. It is scored by the same promotion rule against
  r7_base_s23/s24, and against each component arm (`eval_screen.py pair`, information).
- Deviation, declared now: the combination runs start later than the baseline batch. Same-seed GPU non-determinism does
  not depend on launch time and is in σ_run, so the baseline is not re-run.
- With 0 or 1 promotion, no combination is run.

### 1.7 What is reported
For every arm: per-seed and pooled val / holdout Δ with SE_cal and z, the slices, seed signs, the verdict, train minutes
and cost ratio, peak memory, the scale growth flags, correlations and the 2-member ensemble readout. Then the list of
promoted arms, which goes to stage 2 (10-epoch pairs, unified rule; `screen.sh <gpu> <arm> <seed> e10`).

---------------------------------------------------------------------------------------------------------------------

## 2. Progress log (UTC)

(appended during the run)
