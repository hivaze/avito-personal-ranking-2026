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
- 05:24 Driver `nn_x/r7/queue_r7.py logs/queue_r7_s1.jobs` launched (PID 1542799). First wave, 7 runs, one per qualifying
  GPU: base s23 (GPU N), base s24 (6), muon_in s23 (7), muon_in s24 (0), resdrop0 s23 (1), resdrop0 s24 (2), pbld s23 (3).
  GPU N did not qualify ([n] GB free < 62 + 80). batch2s is held for the base GPUs (cost pairing, §1.3).
- 05:52-06:05 Wave 2 as lanes freed: pbld s24 (GPU N), hc4 s23 (1), batch2s s24 (6, right after r7_base_s24 on the same
  GPU, so the cost pair is comparable), hc4 s24 (4), dwa s23 / s24 (0 / 2), diff2 s23 / s24 (3 / 5).
- 06:16-06:33 swiglu s23 / s24 (6 / 2), batch2s s23 (7; the 55-min hold for GPU N expired because GPU N never had room
  for 72 + 80 GB beside the r6 lanes, so s23 has no comparable cost pair), derf s23 / s24 (2 / 1).
- 06:41-06:58 hl2 s23 / s24 (0 / 2), periln s23 / s24 (7 / 3), ftok s23 / s24 (6 / 4), A/A r7_base_s23r (2).
- 07:01-07:05 postln s23 / s24 (1 / 2).
- 07:46 Last run (ftok s24, GPU N) done. **29/29 runs completed, 0 crashes, 0 relaunches.** 16.7 lane-hours of training
  in 2 h 22 min of wall-clock, on 7-8 lanes. Driver exited by itself; END line written to RUNNING.txt.
- GPU-rule compliance: every launch satisfied ≤ 3 existing processes and effective free − need ≥ 80 GB (driver log
  `nn_x/r7/logs/queue_r7_s1.driver.log`). Other workflows later launched on two of our GPUs (the lead's two
  holdout_evals on GPU N at ~06:25, giving 4 processes; an LLM 0.6B job on GPU N after diff2 s24, leaving ~81 GB free).
  Both stayed within the caps. No process we did not start was touched.
- 07:47 A/A step, then `eval_screen.py screen` (all 13 arms) → `nn_x/r7/logs/eval_screen.{txt,json}`.

---------------------------------------------------------------------------------------------------------------------

## 3. Results (pre-declared rule)

### 3.1 A/A step (done first, per §1.4)
`eval_screen.py aa` → `nn_x/r7/logs/eval_aa.txt`:

| run | selected step | val | holdout |
|---|---|---|---|
| r7_base_s23 | 19670 | 0.39408 | 0.31779 |
| r7_base_s23r (same seed, rerun) | 19000 | 0.39411 | 0.31773 |
| Δ (r − original) | | +0.00003 (SE_feed 0.00045) | −0.00006 (SE_feed 0.00024) |

**Fork check OK; calibration consistent.** The pre-declared σ_run (val 0.00035, holdout 0.00015) is kept. Same-seed val
correlation 0.9967.

### 3.2 Stage-1 verdicts: **0 of 13 arms promote**

Baselines: r7_base_s23 val 0.39408 / holdout 0.31779; r7_base_s24 val 0.39334 / holdout 0.31749. Both select the
last or second-to-last eval, like every arm. 2-epoch runs never decline: end − selected is −0.0002..+0.0008 on val and
−0.0003..0 on the holdout across all 29 runs.

| arm | val Δ s23 / s24 | val pooled Δ ± SE_cal (z) | hold Δ s23 / s24 | hold pooled Δ ± SE_cal (z) | worst rule slice (z) | verdict (failed conditions) |
|---|---|---|---|---|---|---|
| muon_in | −0.00049 / −0.00154 | −0.00101 ± 0.00057 (−1.79) | −0.00001 / −0.00060 | −0.00031 ± 0.00031 (−0.98) | val M≥50 −2.51 | FAIL (1, 2, 3) |
| batch2s | −0.00164 / −0.00126 | **−0.00145 ± 0.00059 (−2.47)** | −0.00071 / −0.00148 | **−0.00109 ± 0.00032 (−3.44)** | val M≥50 −4.17 | FAIL (1, 2, 3); COST FAIL |
| resdrop0 | −0.00030 / +0.00091 | +0.00030 ± 0.00058 (+0.53) | +0.00038 / +0.00042 | +0.00040 ± 0.00031 (+1.29) | val M≥50 −0.65 | FAIL (1) |
| pbld | −0.00060 / −0.00054 | −0.00057 ± 0.00061 (−0.93) | +0.00026 / +0.00058 | +0.00042 ± 0.00034 (+1.22) | val Thu −2.22 | FAIL (1) |
| hc4 | −0.00032 / −0.00013 | −0.00022 ± 0.00054 (−0.42) | +0.00035 / −0.00018 | +0.00008 ± 0.00028 (+0.30) | val M≥50 −2.30 | FAIL (1, 3) |
| dwa | +0.00013 / +0.00114 | +0.00064 ± 0.00048 (+1.32) | −0.00040 / −0.00014 | −0.00027 ± 0.00025 (−1.07) | val M≥50 −0.85 | FAIL (1, 2) |
| diff2 | −0.00141 / −0.00046 | −0.00093 ± 0.00054 (−1.73) | −0.00006 / −0.00023 | −0.00014 ± 0.00029 (−0.49) | val Thu −3.19 | FAIL (1, 2, 3) |
| swiglu | −0.00159 / −0.00054 | −0.00107 ± 0.00055 (−1.95) | +0.00013 / +0.00024 | +0.00019 ± 0.00031 (+0.60) | val M≥50 −2.54 | FAIL (1) |
| derf | −0.00116 / −0.00132 | **−0.00124 ± 0.00058 (−2.15)** | −0.00069 / −0.00122 | **−0.00096 ± 0.00033 (−2.91)** | val M≥50 −3.67 | FAIL (1, 2, 3) |
| hl2 | −0.00020 / −0.00098 | −0.00059 ± 0.00053 (−1.11) | +0.00007 / +0.00007 | +0.00007 ± 0.00028 (+0.26) | val M≥50 −3.27 | FAIL (1) |
| periln | −0.00110 / −0.00111 | −0.00111 ± 0.00056 (−1.97) | −0.00083 / −0.00044 | **−0.00064 ± 0.00032 (−2.02)** | val M≥50 −2.91 | FAIL (1, 2, 3) |
| ftok | −0.00124 / −0.00104 | **−0.00114 ± 0.00054 (−2.12)** | −0.00025 / +0.00012 | −0.00007 ± 0.00030 (−0.22) | val Thu −3.42 | FAIL (1, 2, 3) |
| postln | −0.00130 / +0.00041 | −0.00045 ± 0.00055 (−0.82) | −0.00036 / +0.00023 | −0.00007 ± 0.00030 (−0.22) | val Thu −2.18 | FAIL (1, 2, 3) |

Conditions (§1.5): 1 = pooled val Δ > 1.5 SE_cal; 2 = pooled holdout Δ ≥ 0; 3 = both seeds positive on val or on holdout.
No arm meets condition 1. The nearest is dwa (z +1.32), and it also fails the holdout (−0.00027).

**Promoted arms: none.**
- **Combinations: none run.** §1.6 needs ≥ 2 individual promotions.
- **Stage 2: nothing to run.** The pre-declared drop0 combination is not triggered, because resdrop0 did not promote.
- **batch2s cost lever: COST FAIL.** It is not non-inferior (val −0.00145 < −0.0005; holdout −0.00109 < −0.0003). The
  one comparable pair (s24: GPU N, starts 29 min apart) gives a train-minute ratio of 0.815 > 0.75.
  The s23 pair is not comparable (GPU N vs 4).

### 3.3 Slices (pooled over both seeds; Δ and z with SE_feed)

| arm | val M<50 | val M≥50 | val Tue | val Thu | val coldpos | val warmpos | hold late 20% | 2 arm vs 2 base members, z-mean: val / hold |
|---|---|---|---|---|---|---|---|---|
| muon_in | −0.00046 (−0.5) | −0.00139 (−2.5) | −0.00064 (−0.9) | −0.00142 (−1.9) | +0.00021 (+0.2) | −0.00186 (−2.7) | +0.00024 (+0.4) | −0.00084 (−1.3) / −0.00058 (−1.7) |
| batch2s | −0.00004 (−0.0) | −0.00242 (−4.2) | −0.00103 (−1.4) | −0.00190 (−2.5) | −0.00042 (−0.4) | −0.00199 (−2.8) | −0.00178 (−2.8) | −0.00132 (−2.0) / −0.00121 (−3.4) |
| resdrop0 | +0.00127 (+1.3) | −0.00037 (−0.6) | +0.00072 (+1.0) | −0.00015 (−0.2) | +0.00123 (+1.2) | −0.00014 (−0.2) | +0.00092 (+1.4) | +0.00004 (+0.1) / +0.00041 (+1.2) |
| pbld | +0.00007 (+0.1) | −0.00101 (−1.7) | +0.00054 (+0.7) | −0.00179 (−2.2) | +0.00334 (+2.8) | −0.00234 (−3.2) | +0.00077 (+1.1) | +0.00013 (+0.2) / +0.00028 (+0.7) |
| hc4 | +0.00122 (+1.4) | −0.00122 (−2.3) | +0.00014 (+0.2) | −0.00062 (−0.9) | +0.00205 (+2.1) | −0.00088 (−1.4) | −0.00043 (−0.8) | +0.00000 (+0.0) / −0.00015 (−0.5) |
| dwa | +0.00208 (+2.6) | −0.00036 (−0.8) | +0.00070 (+1.2) | +0.00057 (+0.9) | +0.00126 (+1.5) | +0.00029 (+0.5) | −0.00023 (−0.5) | +0.00091 (+1.7) / −0.00045 (−1.5) |
| diff2 | −0.00022 (−0.2) | −0.00142 (−2.8) | +0.00027 (+0.4) | −0.00225 (−3.2) | +0.00014 (+0.2) | −0.00123 (−1.9) | +0.00007 (+0.1) | −0.00038 (−0.6) / −0.00014 (−0.4) |
| swiglu | −0.00064 (−0.7) | −0.00136 (−2.5) | −0.00082 (−1.2) | −0.00133 (−1.9) | −0.00145 (−1.5) | −0.00088 (−1.3) | −0.00030 (−0.5) | −0.00101 (−1.6) / +0.00029 (+0.8) |
| derf | −0.00001 (−0.0) | −0.00210 (−3.7) | −0.00096 (−1.4) | −0.00155 (−2.0) | −0.00350 (−3.3) | −0.00039 (−0.6) | −0.00138 (−2.1) | −0.00141 (−2.2) / −0.00103 (−2.8) |
| hl2 | +0.00105 (+1.2) | −0.00172 (−3.3) | −0.00029 (−0.5) | −0.00092 (−1.3) | +0.00023 (+0.2) | −0.00090 (−1.4) | +0.00016 (+0.3) | −0.00038 (−0.6) / −0.00001 (−0.0) |
| periln | −0.00042 (−0.4) | −0.00158 (−2.9) | −0.00049 (−0.7) | −0.00178 (−2.5) | −0.00093 (−0.9) | −0.00149 (−2.2) | −0.00015 (−0.2) | −0.00050 (−0.8) / −0.00074 (−2.0) |
| ftok | −0.00040 (−0.4) | −0.00166 (−3.2) | −0.00008 (−0.1) | −0.00231 (−3.4) | −0.00021 (−0.2) | −0.00157 (−2.5) | −0.00024 (−0.4) | −0.00119 (−2.0) / +0.00003 (+0.1) |
| postln | −0.00026 (−0.3) | −0.00058 (−1.1) | +0.00057 (+0.9) | −0.00157 (−2.2) | −0.00066 (−0.7) | −0.00029 (−0.5) | −0.00042 (−0.7) | +0.00005 (+0.1) / +0.00017 (+0.5) |

### 3.4 Other pre-declared readouts (information)

| arm | cost per step (r7_build T9, controlled) | params Δ | SCALE growth flags (arm > 1.25 × base) | SCALE-SUSPECT | rho_x vs rho_bb 0.9781 | mixed {base_sA, arm_sB} vs 2 × base: val / hold | FAMILY-CANDIDATE |
|---|---|---|---|---|---|---|---|
| muon_in | 1.00 | 0 | top sim-bias, retr.a | **yes** (moot: FAIL) | 0.9769 | +0.00011 / −0.00005 | no |
| batch2s | 0.82 per sample (measured 0.815 wall-clock, s24) | 0 | retr.a | no | 0.9789 | −0.00080 / −0.00045 | no |
| resdrop0 | 0.99 | 0 | retr.a | no | 0.9745 | +0.00008 / +0.00037 | no |
| pbld | 1.10 | +88,504 | - | no | 0.9748 | +0.00060 / +0.00044 | no (rho_x > 0.9731) |
| hc4 | 1.43 | +7,376 | - | no | 0.9777 | +0.00010 / +0.00001 | no |
| dwa | 1.04 | +14 | - | no | 0.9780 | +0.00059 / −0.00030 | no |
| diff2 | 1.11 | +34,064 | - | no | 0.9779 | −0.00012 / −0.00001 | no |
| swiglu | 1.05 | −860 | - | no | 0.9765 | −0.00023 / +0.00018 | no |
| derf | 1.06 | +28 | - | no | 0.9750 | −0.00037 / −0.00043 | no |
| hl2 | 1.15 | +149,248 | - | no | 0.9784 | −0.00018 / +0.00008 | no |
| periln | 1.06 | +2,560 | c_ln gain | no | 0.9772 | −0.00037 / −0.00030 | no |
| ftok | 1.47 | +181,248 | - | no | 0.9777 | −0.00032 / −0.00001 | no |
| postln | 1.00 | 0 | cross0 W_O RMS (its β-scaled init) | no | 0.9770 | +0.00030 / −0.00017 | no |

- Peak reserved memory was 55 GB for every run, including hc4, ftok and batch2s.
- Wall-clock train minutes (27-51 min) are confounded by GPU sharing. The 51-min runs (hc4 s24, ftok s24) were on
  GPU N next to the two r6 continuation lanes. Use the controlled T9 ratios above.
- **No arm is more diverse than a seed change.** Every rho_x is above rho_bb − 0.005 = 0.9731, so there are no
  same-strength structural variants for the ensemble.

---------------------------------------------------------------------------------------------------------------------

## 4. Information only (NOT the pre-declared rule)

### 4.1 Robustness to the baseline draw
Both r7 baselines landed on the high side of the same-seed replicates that exist for this exact flag-off config:

| seed | replicate (code copy) | val | holdout |
|---|---|---|---|
| 23 | r7_base_s23 / r7_base_s23r (nn_x/r7) | 0.39408 / 0.39411 | 0.31779 / 0.31773 |
| 23 | sx_base_s23 (nn_x/solvers) / r6_base_s23 (nn_x/r6) | 0.39375 / 0.39335 | 0.31785 / 0.31761 |
| 24 | r7_base_s24 / sx_base_s24 | 0.39334 / 0.39288 | 0.31751 / 0.31754 |

- Same-seed A/A val deltas across all replicate pairs: +0.00003, −0.00033, −0.00036, −0.00040, −0.00046, −0.00073,
  −0.00076 (RMS 0.00050; the pairs are not independent). The holdout is stable: |Δ| ≤ 0.00024.
- **Re-scored against the per-seed mean of all replicates** (4 for s23, 2 for s24; same SE_cal formula and the same
  3-part test): **still 0 / 13 promote.**
  - The val deltas move up by about 0.0003. Holdout deltas are unchanged.
  - Nearest arm: dwa, val +0.00088 (z +1.92, both seeds positive), but holdout −0.00025 (both seeds negative), so it
    still fails condition 2.
  - Next: resdrop0, val +0.00055 (z +0.99), holdout +0.00042 (z +1.41).
- Script: `scratchpad/r7/screen/sens.py`; output `scratchpad/r7/screen/sens.json`.

### 4.2 A consistent slice pattern: every perturbation costs warm / M≥50 feeds
- **val M≥50:** all 13 arms are negative (mean −0.00132). Against the multi-replicate baseline all 13 stay negative.
- **Thu:** 12 / 13 negative. **warmpos:** 12 / 13 negative. **M<50:** mean +0.00025 (5 positive, 8 negative).
- Gains sit in M<50 / cold feeds for dwa (M<50 +0.0021, z +2.6), resdrop0 (+0.0013), hc4 (+0.0012; coldpos +0.0021),
  hl2 (+0.0011) and pbld (coldpos +0.0033, z +2.8; warmpos −0.0023).
- Reading: the base recipe sits at a local optimum for warm, M≥50 feeds. Every past choice was selected on the M≥50
  random-like holdout and on val, where 59% of the feeds have M ≥ 50.
- R6_PLAN §0.5 notes the cold share rises from 0.29 on val to 0.41 on private. A cold-tilted arm could therefore be
  worth slightly more on private than on val. This is post hoc, with 13 arms × 6 slices of multiple comparisons, so it
  is not a basis for promotion.

### 4.3 What the nulls answer (the user's two questions)
**Modern blocks.** The 9 structural arms cover:
- norms: Derf (norm-free), Peri-LN, Post-LN + DeepNorm;
- residual schemes: dynamic hyper-connections n = 4, DenseFormer DWA;
- FFN form: SwiGLU;
- attention: differential target attention;
- tabular input encoders: RealMLP PBLD periodic embeddings, feature-group tokens.

None makes a stronger 2-epoch member, and none makes a more diverse one.
- Derf and Peri-LN are clearly harmful on both val and holdout (val z −2.15 / −1.97, holdout z −2.91 / −2.02).
- FT-style feature tokens, SwiGLU and DIFF attention cost about 0.001 on val with a flat holdout.
- hc4 is flat. PBLD trades val (−0.0006) for holdout (+0.0004).
- DWA is the only arm with a positive val in both seeds, but its holdout is negative in both.

This matches r7_research's prior (0 to +0.0005 at this scale, MDE about +0.0014). The model is shallow (5 blocks,
d = 128), so the depth-wise fixes these blocks bring have little to act on.

**Never-varied hyperparameters, now measured:**
- **Optimiser partition.** Input projections on AdamW instead of Muon: −0.0010 val, −0.0003 holdout. Keep Muon on
  every 2-D matrix.
- **Batch ×2 (square-root rule).** −0.0015 val, −0.0011 holdout. It is only 0.82× the cost per sample, so it is not a
  cost lever. Keep 512 feeds / 32,768 candidates.
  - Untested alternative: the linear rule (lr 2e-3). Under Muon the per-step update norm is fixed, so with half the
    steps the square-root rule moves the weights about 0.71× as far as the baseline. lr 2e-3 is also the lr that F7
    already found better at 10 epochs.
- **Residual / FFN / retrieval dropout 0.1 → 0.** +0.0003 val, +0.0004 holdout (holdout positive in both seeds): null
  at 2 epochs. Dropout should matter in the 10-epoch memorising cooldown, which this screen cannot see. That belongs to
  the continuation harness (r7_research §4.2; r6 Q1).
- **History depth 1 → 2** (zero-init extra layer): −0.0006 val, +0.0001 holdout. Keep 1.

The horizon-dependent knobs (cooldown length, pool mix, wd, aux-anneal target, dropout in the cooldown) were not part
of this screen by design. r6 Q1 is running the cooldown-continuation screen for pool mix and wd.

### 4.4 Cross-reference: the r6 structure screen (r6's decision, not r7's)
- r6's readout landed at 07:59: `nn_x/r6/logs/r6_eval_screen_a4u_wd01_cset_typed_adrop0_a4w.txt`. Its baseline is
  sx_base_s23/s24. r6's A/A gives r6_base_s23 − sx_base_s23 = val −0.00040, holdout −0.00024, which is OK.
- r6 verdicts:
  - **a4u PASS:** val +0.00117 (z +2.48), holdout −0.00004.
  - **cset FAMILY-CANDIDATE:** val +0.00028, holdout +0.00090 (z +2.69), rho_x 0.9729.
  - typed, adrop0 and a4w FAIL. wd01 was dropped by r6.
- **Information only:** the r6 arms re-scored against the same multi-replicate baseline as §4.1, with r7's 3-part test.

  | arm | val Δ (z), s23 / s24 | holdout Δ (z), s23 / s24 | val M≥50 | r7 test |
  |---|---|---|---|---|
  | a4u | +0.00102 (+2.01), +0.00073 / +0.00131 | +0.00002 (+0.08), +0.00021 / −0.00016 | +0.00056 | would pass |
  | cset | +0.00013 (+0.22) | +0.00096 (+2.81) | −0.00092 | fail on val |
  | typed | +0.00028 (+0.61) | +0.00026 (+1.07) | −0.00037 | fail |
  | adrop0 | −0.00038 (−0.73) | +0.00010 (+0.37) | −0.00146 | fail |
  | a4w | −0.00024 (−0.41) | +0.00003 (+0.11) | −0.00101 | fail |

- So a4u (unique retrieval slots + per-type counts) is the only structural change screened this morning, in r6 or r7,
  that lifts val in both seeds, and it is the only one that does not cost M≥50. It goes to r6's 10-epoch pair.
- **The pre-declared drop0 stage-2 combination is not triggered:** neither r7 resdrop0 nor r6 adrop0 promoted.

---------------------------------------------------------------------------------------------------------------------

## 5. Workflow status (08:05 UTC) and pointers

- **r7: research (04:15-05:00) → build (04:35-05:20; 321/321 GPU tests) → stage-1 screen (05:24-07:46): DONE.**
  - Outcome: 0 / 13 promotions, so there is no stage 2 or stage 3 to run from r7.
  - All r7 GPU jobs have ended. RUNNING.txt has END lines for all 29 runs and for the driver.
- **r6 Q2 structure screen** (nn_x/r6, GPU N; baseline sx_base_s23/s24): DONE at 07:59.
  - a4u PASS → 10-epoch pair; cset FAMILY-CANDIDATE → 10-epoch pair; typed, adrop0 and a4w FAIL (§4.4).
  - r6 owns that decision; r7 did not duplicate those arms.
- **r6 Q1 cooldown continuations** (GPU N): c0 / m02 / w10 done for F7 roots 101-103, c0 for 104; the rest is running.
- **Files:**
  - r7 evaluation: `nn_x/r7/logs/eval_screen.{txt,json}` and `nn_x/r7/logs/eval_aa.{txt,json}`;
  - driver log: `nn_x/r7/logs/queue_r7_s1.driver.log`;
  - per-run logs and json: `nn_x/r7/logs/r7_*_s{23,24}.{log,json}`;
  - predictions: `nn_x/r7/preds/nn_r7_*_trainonly_val.npy` and `nn_x/r7/preds/holdout/`;
  - pre-declaration snapshot: `nn_x/r7/logs/r7_screen_predeclaration_frozen.md`.
- **Suggestions for the lead** (not decisions; nothing was launched):
  1. Treat the "modern blocks" question as closed for this model at 2 epochs. Add derf, periln, ftok, swiglu, diff2,
     hc4, dwa, pbld, postln, muon_in, batch2s and hl2 to COVERAGE_MAP as closed (screened null or harmful).
  2. If one more knob is worth GPU time, the screen points to the horizon, not the block: dropout and pool dose in the
     10-epoch cooldown (r6 Q1 / continuation harness). Optionally a batch ×2 arm under the linear rule (lr 2e-3), as a
     cheap 2-seed check.
