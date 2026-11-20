# R6 PLAN: what to run next, and why

Written 2026-11-12 ~04:30-05:00 UTC by the r6 plan step. CPU only (`CUDA_VISIBLE_DEVICES=""`, 12 threads). No GPU job was launched.
Inputs: reports/r6_recap.md, r6_external.md, r6_tta.md, r6_design.md plus its review, CURRENT_STATUS (to the 04:15 line), COVERAGE_MAP, and the run logs in nn/logs (including the F11/F12 runs that finished overnight).
New files from this step:
- `nn_x/r6/cd_lane.sh`, the launcher for Q1 in §4. It was DRY-tested only, and every command it prints parses correctly with train.py's own parser.
- Scratch outputs in `scratchpad/r6/plan/`.

The user asked for this round: to think about hyper-parameters and model structure as well. §1.2 answers it.

---------------------------------------------------------------------------------------------------------------------

## 0. Decisions

1. **The binding constraint is late-cooldown memorisation of the exploration pool, not capacity, depth or encoder type.**
   - New evidence comes from the overnight families (§1.2).
   - Across the 5 pre2s families, the rank of selected-step strength matches the rank of late-cooldown decline: Spearman 1.0 on val and 0.9 on the holdout (n = 5).
   - d192 has 2.1x the parameters of F7. It peaks 0.0008 lower on val and 0.0007 lower on the holdout, and its end-of-run decline is 3-5x larger.
   - The levers that act on this were never varied: pool dose, weight decay and dropout during the cooldown. A 2-epoch screen cannot see them (29/29 two-epoch runs never decline).
2. **Morning queue, in priority order.**
   - **Q1: cooldown-continuation screen (new).** F7's own cooldown is re-run from the 7 saved @77.5k snapshots, one change per arm: dose-matched pool mix 0.02, and wd 0.1. No model code is needed.
   - **Q2: the reviewed 2-epoch structure screen,** with two pre-launch amendments: a4u gains the per-type count fix (`--retr_tcnt`), and wd01 is dropped because Q1 covers it.
   - Follow-ups are conditional and pre-declared (§4.4).
3. **TTA: do not use.** All three arms fail at ensemble level, and T3 is harmful: −0.0010 ± 0.00045. Details in §3.
4. **Do not add families for diversity alone.**
   - Cross-family correlation is 0.92-0.94, about seed level, and F10 GRU added nothing (CURRENT_STATUS 04:05).
   - Only a stronger member moves the pool now.
   - With 27 days left (deadline 2026-12-09), cost levers have low value: earlier cooldown (P1), batch ×2 (P3) and branch families (R1).
5. **Honest expectation.**
   - Q1 + Q2 together give about +0.0003..+0.0010 val over the current pool if anything passes. P(at least one adoption) ≈ 0.4. This is at or below the MDE for one model generation.
   - Arms whose gain sits in the cold slice transfer best to private (cold share 0.29 → 0.41).
6. **GPU scope.** r6 code accepts only some GPUs (`nn_x/r6/common.py:15`). CURRENT_STATUS §3 records the user's 03:40 authorisation of some GPUs, and the lead is using 0-3 now. Whether r6 may use them is the lead's call: edit common.py:15, the `check_gpus` case in queue_morning.sh, and the `case` line in cd_lane.sh.

---------------------------------------------------------------------------------------------------------------------

## 1. Recap

### 1.1 Coverage
Every idea and its status is in **reports/COVERAGE_MAP.md**; every run is in nn_x/r6/recap/runs.csv (285 train.py segments, 247 fine-tunes).

Never varied in any run: dropout 0.1, wd 0.02, batch 32768/512 (~317 feeds), ff 2, max_hist 256.
Never varied under Muon + WSD:
- blocks 2, hist_layers 1, heads 4 (d192/6 heads now run as F12);
- retr_m 16, cooldown_frac 0.2, aux_anneal 0.1;
- warmup, momentum, betas, clip;
- every fine-tune knob.

Source: r6_recap §0.2.

### 1.2 What the runs say about hyperparameters and structure

**Late-cooldown memorisation across families** (new, from the train logs). Values are holdout-selected step vs final step, train-only, GPU eval.
Command: `.venv/bin/python scratchpad/r6/plan/family_decline.py` → `scratchpad/r6/plan/family_decline.txt`.

| family (all pre2s, Muon/WSD, mix 0.1) | params | n | selected val | selected holdout | end − selected val | end − selected holdout |
|---|---|---|---|---|---|---|
| F10 GRU, lr 2e-3 | 1.08M | 4 | **0.39469** | 0.31880 | −0.0015 | −0.0017 |
| F7 tf d128, lr 2e-3 | 1.01M | 7 | 0.39460 | **0.31911** | −0.0020 | −0.0031 |
| F9 tf d128, lr 1e-3 | 1.01M | 6 | 0.39382 | 0.31845 | −0.0044 | −0.0053 |
| **F12 tf d192/6h, lr 2e-3 (new)** | 2.12M | 4 | 0.39377 | 0.31840 | **−0.0096** | **−0.0094** |
| F11 DIN + DCNv2, lr 2e-3 | 1.72M | 4 | 0.39097 | 0.31612 | −0.0130 | −0.0125 |

- **The faster a family memorises, the lower it peaks.** On val the two rankings are identical (Spearman 1.0); on the holdout Spearman is 0.9.
  - This is correlational and confounded by architecture and lr.
  - Within one architecture it holds too: lr 2e-3 vs 1e-3 halves the decline and peaks +0.0008 / +0.0007 higher (r6_recap §3.1, §3.3).
- **d192 shows that capacity is not binding.**
  - Like F7, it peaks at 80-85k, but then falls to 0.384 on val by the end (`nn/logs/v26_d192_pre2s_s702.log`).
  - The extra capacity goes into memorising the pool.
- **The pool dose explains why 10-epoch runs memorise and 2-epoch runs do not.**
  - The pool is 134,118 feeds (`nn/logs/v17_pre2s_s101.log:8`) and a step holds 317.4 feeds (3,121,208 / 9,835).
  - At mix 0.1, the 19,670-step 10-epoch cooldown replays the pool **4.65 extra times**. The 3,934-step 2-epoch cooldown replays it **0.93 times**.
  - The decline grows with the dose: mix 0 gives −0.0007, mix 0.1 gives −0.0033, mix 0.3 collapses at −0.027 (r6_recap §3.3).
  - So the mix of 0.1 chosen in short runs is 5x over-dosed at 10 epochs. **Mix 0.02 restores the 2-epoch dose.**
- **Caveat.** Holdout selection already protects the members from the decline. A lever pays only if it raises the peak itself, not just the end of the run, and Q1 measures exactly that.

**Knob-by-knob verdict:**

| knob / structure | evidence | verdict |
|---|---|---|
| lr (Muon) | 2e-3 vs 1e-3 at 10 epochs: +0.0008 val / +0.0007 holdout single; halves the decline (r6_recap §3.1) | keep 2e-3; > 2e-3 low prior |
| epochs / plateau length | 10 vs 2 epochs: single +0.0006, 6-seed ensemble +0.0040. Ensemble at 77.5k vs 60k: −0.0011 ± 0.0006 (r6_recap §0.3, §3.2) | keep 10; longer = no |
| cooldown (WSD 20%) | early cooldown adds +0.0031 ± 0.0006 at ensemble level. The peak comes 1-9k steps into the 19.7k-step cooldown; the rest declines | the second half is wasted → **Q1** |
| pool mix 0.1 | dose arithmetic above. Mix 0 vs 0.1 at the selected step: +0.00045 val (1 seed, noise) | **Q1 arm m02**; m0 in wave 2 |
| weight decay 0.02 | never varied. lr·wd doubling coincides with half the decline | **Q1 arm w10** (cooldown only); 10-epoch pair if w10 ≥ 0 |
| dropout 0.1 | never varied. Attention dropout drops the matched event in peaked heads | **Q2 adrop0**; residual dropout 0.2 in the cooldown → wave 2 (needs a flag) |
| batch 32768/512 | B_simple 4.9-9.4K feeds vs 317 per step (r6_recap §4.2) | cost lever only; not now |
| width / depth | d256 null, heads 8 −0.0011, blocks 3 −0.0006, deep ≈ (r6_recap §3.5); d192 table above | **closed under the current regulariser**; re-test d192 only after a Q1 winner |
| history encoder | GRU ≈ transformer and adds nothing to the pool; DIN fails the gates | closed |
| A4 retrieval (strongest component) | +0.0019 / +0.0033. 30-32% of its 16 slots repeat a higher slot's item. Predictions depend on batch composition by up to 0.0137 logits (review) | **Q2 a4u (+ tcnt fix)**; a4w last |
| candidate set attention | neutral, with no similarity prior (nn_report:68) | **Q2 cset** |
| history length / target actions | max_hist 256 is binding: crop TTA −0.0007..−0.0041 (r6_tta §5.1). Target action = et1 & es1, median 51 events per feed | **Q2 typed**; recency-band summary tokens later (r6_tta §5.1, untested) |
| inputs, losses, fine-tune knobs, ensembling method | saturated / closed (COVERAGE_MAP) | none |

**Found while planning (information).**
- The F6cd continuations (`v21_cd_v14_s92/94/96`, members of the interim leader F6*) trained their cooldown with `loss_norm 'batch'`. Their parents and F5cd used `'feed'` (`nn/logs/v21_cd_v14_s92.log:1` vs `v14_s92.log`), because `nn/run_f6cd.sh` omits `--loss_norm feed`.
- This is a recipe deviation, not a leak, and their measured strength already includes it.
- **Any continuation must pass `--loss_norm feed`.** The R1 spec in r6_external §3 copies the run_f6cd flags and would inherit the deviation. cd_lane.sh passes it.

---------------------------------------------------------------------------------------------------------------------

## 2. External tricks: new vs covered (r6_external §2)

Of 34 topics, 20 are covered, 10 partial and 4 new. Each is listed with where it goes now:

| item | status | where it goes |
|---|---|---|
| R1 cooldown branches from plateau snapshots | new | Q1's control c0 is an R1 branch, so R1 is read for free. Branch families are not run as a cost lever |
| R2 augmentation of pool batches (candidate subset + history-event drop) | new | wave-2 Q1 arm; needs a flag |
| R4 candidate-side retrieval + set similarity bias | new | Q2 `cset` |
| R6 target-action key view | new | Q2 `typed` |
| R3 wd / dropout | partial | Q1 w10, Q2 adrop0, wave-2 d20 |
| R5 retr_m 32 + product term | partial | Q2 `a4w` (lowest) |
| R7 hyper-branches | partial | the Q1 arms are hyper-branches; they become families only via the gates |
| R8 SwiGLU / RMSNorm / GDCN / dual-softmax | partial | owned by the r7 workflow (norm / residual / SwiGLU research), not r6 |

Closed by evidence:
- soups and WiSE-FT (our 0.5 FT + 0.5 TO is already the output-space version);
- slide-window augmentation (its gain comes from next-item targets);
- rank fusion, KD, and every loss family;
- larger batch for throughput (diversity risk).

---------------------------------------------------------------------------------------------------------------------

## 3. TTA: result, verification, decision

Source: reports/r6_tta.md; numbers re-read from `solvers/r6/out/tta_eval.json`. Six members (F7 s101-104, F6 s93/s95), 14,423 val feeds, paired per feed against the deterministic CPU replay.

| arm | 6-member ensemble Δ ± SE (z) | members > 0 | mean member Δ | cost per member (CPU passes) | verdict |
|---|---|---|---|---|---|
| T1 MC-dropout, K = 8 | −0.00009 ± 0.00024 (−0.36) | 4/6 | +0.00006 | 21.6× | FAIL |
| T2 history-event dropout 10%, K = 8 | −0.00008 ± 0.00034 (−0.23) | 5/6 | +0.00036 | 8.5× | FAIL (the member gain disappears in the z-mean) |
| T3 history crop {256, 192, 128, 64} | **−0.00100 ± 0.00045 (−2.22)** | 0/6 | −0.00087 | 4× | FAIL, harmful |

- **Verification.** The literal replay bar (per-feed Spearman ≥ 0.9999) was missed: values were 0.99921-0.99939. The miss is bf16 rounding.
  - CPU bf16 vs CPU fp32 gives 0.99934, the same as GPU vs CPU at 0.99929.
  - Pearson is 0.999998 or higher.
  - Member NDCG CPU − GPU is within ±0.00016; the ensemble is 0.39877 on CPU vs 0.39880 on GPU.
  - Every delta is paired against the same deterministic replay, so the verdicts stand.
  - For future replays, use the gate "Pearson plus NDCG within 1 SE" (r6_tta §5.5).
- **Decision: no TTA in any submission.** T1 + T2 at 8 passes each is also null (+0.00003 ± 0.00026). The cross-member z-mean already removes what TTA removes.
- By-products that were used:
  - history length is binding (keep 256);
  - duplicate and tie-ambiguous retrieval slots led to a4u;
  - fp32 inference gains nothing (−0.00003 ± 0.00016).

---------------------------------------------------------------------------------------------------------------------

## 4. Morning GPU queue (some GPUs; at most 2 r6 lanes per GPU; project cap 4 processes per GPU)

### 4.0 Pre-launch steps (CPU, about 10 minutes; record them in the CURRENT_STATUS Log before any r6 GPU job, so the pre-declaration stays valid)

1. **a4u amendment** (review: plain dedup drops older event types for 17.8% of distinct (feed, item) pairs, many of them type 2/3 contact-like events).
   ```
   cd <repo>/nn_x/r6 && patch -p0 < /tmp/scratchpad/r6/review/tcnt/retr_tcnt.patch
   ```
   - The patch applies cleanly (`--dry-run` checked).
   - Then in queue_morning.sh set `a4u) echo "--retr_uniq --retr_tcnt" ;;`.
2. **Drop wd01 from the 2-epoch screen.** Its mechanism is invisible at 2 epochs; Q1 w10 and the conditional 10-epoch pair cover it.
   ```
   SCREEN_JOBS="base:23 a4u:23 a4u:24 cset:23 cset:24 typed:23 typed:24 adrop0:23 adrop0:24 a4w:23 a4w:24"
   ```
   r6_eval.py skips missing arms.
3. **Re-run the tests:**
   ```
   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/tests/test_r6.py
   ```
   Expect 79/79. Also run `scratchpad/r6/review/tcnt/test_tcnt.py` (10/10).
4. **Preflight:** `bash nn_x/r6/queue_morning.sh check "<gpus>"`. It refuses a GPU with ≥ 3 processes or < [n] GB free.
5. **Ownership with r7.** Tell r7 not to run wd, attention dropout, retr_m, cooldown mix or cooldown_frac arms; r6 owns them. r7 keeps norms, residuals, SwiGLU and encoders.

### 4.1 Q1: cooldown-continuation (CD) screen (priority 1; no model code)

**Design.**
- Each job re-runs F7's own cooldown from `nn/ckpt/v17_pre2s_s<root>@77500.pt` (read-only; pre-cooldown, since the cooldown starts at 78,680).
- Cooldown settings: `--epochs 2 --cooldown_frac 1.0 --warmup 0 --lr 2e-3`, i.e. the same 19,670 steps, 1-sqrt decay and K9 anneal as the original.
- Other flags: `--loss_norm feed --holdout 30000 --select_on holdout --eval_every 1000 --seed 811`.
- The holdout is fixed by rng 12345, so it is the same set of feeds as in the original runs (train.py:178).
- Arm flags are appended last:

| arm | flag | mechanism |
|---|---|---|
| **c0** | none | Control: F7's cooldown with a new data order and a reset optimizer. It is also an R1 branch member |
| **m02** | `--cooldown_mix 0.02` | 0.93 extra pool passes, the 2-epoch dose, instead of 4.65 |
| **w10** | `--wd 0.1` | 5× decoupled decay in the cooldown. At wd 0.1, Σ lr·wd over the cooldown is ≈ 1.4, vs 0.28 at wd 0.02 |

**Stage 1 jobs:** roots 101-104 × {c0, m02, w10} = 12 jobs, ordered so that triples complete first.

**Launch:**
```
cd <repo>
J="c0:101 m02:101 w10:101 c0:102 m02:102 w10:102 c0:103 m02:103 w10:103 c0:104 m02:104 w10:104"
DRY=1 bash nn_x/r6/cd_lane.sh 4 "$J"          # print only
for g in <Q1 GPUs, e.g. 4>; do for k in 1 2; do
  setsid nohup bash nn_x/r6/cd_lane.sh $g "$J" > nn_x/r6/logs/lane_cd_gpu${g}_$k.out 2>&1 &
  echo "$! lane cd GPU $g #$k ($(date -u +%FT%TZ))" >> nn_x/r6/logs/RUNNING_r6.txt; done; done
```
- Lanes split the list through mkdir claims and share queue_morning.sh's per-GPU data-load lock.
- A lane started later on another GPU with the same `$J` takes the unclaimed jobs.
- Mirror the PIDs into nn/logs/RUNNING.txt.

**Read-out (CPU, ~15 s; tested on stand-in preds):**
```
cd <repo>/nn_x/r6 && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ../../.venv/bin/python -c "
import r6_eval as R; v,h=R.setup('val'),R.setup('hold'); roots=(101,102,103,104)
C=lambda r:f'r6:r6_cd_c0_r{r}'
for a in ('m02','w10'):
    A=lambda r,a=a:f'r6:r6_cd_{a}_r{r}'; R.P('== '+a+' vs c0'); R.arm_block(v,h,C,A,roots,rules=False)
    for nm,ctx in (('VAL',v),('HOLD',h)):
        e=R.compare(ctx,[('+'.join(C(r) for r in roots),'+'.join(A(r) for r in roots))]); R.P(f'  ENS {nm} arm vs c0: '+R.fmt({'all':e['pooled']['all']}))
R.P('== harness: c0 vs original F7 member'); R.arm_block(v,h,lambda r:f'nn:v17_pre2s_s{r}',C,roots,rules=False)
R.P('  same-root corr c0 vs F7 (R1 read): '+' '.join(f'{R.corr(v,C(r),f\"nn:v17_pre2s_s{r}\"):.4f}' for r in roots))
open('logs/r6_eval_cd.txt','w').write(chr(10).join(R.OUT)+chr(10))"
```
Also report each run's end − selected decline, using `R.log_readout`.

**Pre-declared rules.** Calibration: a 4-root pooled member SE is ≈ 0.00055 on val and 0.00032 on the holdout, measured with F9 vs F7 as stand-ins.
- **Stage 1 (futility only):** an arm continues iff its pooled member Δ vs c0 is ≥ 0 on val AND ≥ 0 on the holdout.
- **Stage 2:** add roots 105-107 for c0 and each continuing arm, e.g. `J2="c0:105 c0:106 c0:107 m02:105 m02:106 m02:107"`.
- **PASS-CD (7 roots) requires all of:**
  - pooled member val Δ > 2 SE;
  - holdout Δ ≥ 0;
  - the 7-member arm ensemble vs the 7-member c0 ensemble ≥ 0 on the holdout and ≥ −1 SE on val. This is the guard against de-noising-only gains (the avg3 lesson);
  - no val slice below −2 SE. A coldpos slice below −1 SE triggers the stale-val replay before adoption.
- **Diagnostic, not a gate:** an arm's end − selected decline relative to c0.
- **Harness check:** pooled c0 vs the original F7 member must satisfy |val Δ| ≤ 0.0014 and |holdout Δ| ≤ 0.0008 (about 2.5 SE).
  - If it fails, arms are still read relative to c0, but promotion by re-cooling F7 roots is suspended and a winner goes through the 10-epoch from-scratch pair instead.
- **R1 read:** if the same-root correlation of c0 with F7 is ≤ 0.955 on val, branches are seed-like. Then branching from `--save_steps` snapshots is a valid way to grow any new family (r6_external R1 bar).

**Wave 2** runs after Q1 stage 1, with flags built on CPU today (§5). Same 4-root futility stage and rules:
- m0 (`--cooldown_mix 0`);
- aa5 (`--aux_anneal 0.5`);
- d20 (dropout 0.2 set after `--init`);
- pp02 (pool threshold p > 0.2, +45% distinct pool feeds, cost-free in the proxy: sol_solvers §2.6);
- aug (R2: `--aug_cand 0.6 --aug_hist 0.15` on pool batches).

### 4.2 Q2: 2-epoch structure screen (priority 2; reviewed queue)

`bash nn_x/r6/queue_morning.sh screen "<Q2 GPUs>"`. After the §4.0 amendments the list is 11 jobs in priority order:
- the A/A job `base:23`;
- **a4u** (`--retr_uniq --retr_tcnt`);
- **cset** (`--cretr_m 8 --set_sim_bias`);
- **typed** (`--typed_cross`);
- **adrop0** (`--attn_drop 0`);
- **a4w** (`--retr_m 32 --retr_prod`).

Every arm uses seeds 23/24 against the existing sx_base_s23/s24. The CPU evaluation runs automatically in the last lane and writes `nn_x/r6/logs/r6_eval_{aa,screen}.txt`.

**Rules (frozen in the r6_eval.py header):**
- **A/A:** |Δ| ≤ 0.0010 val and ≤ 0.0005 holdout, otherwise STOP.
- **PASS:** pooled val Δ > 2 SE AND holdout ≥ −1 SE.
- **FAMILY-CANDIDATE**, all of:
  - val ≥ −0.0010 and holdout ≥ −0.0005;
  - ρ_x ≤ ρ_bb − 0.005;
  - mixed 2-member ensemble ≥ 2× base on the holdout.
- **Other WARN:** any slice below −2 SE. cset WARN on coldpos/warmpos is expected to matter most for private.

### 4.3 Wall time

Per job, about 1 lane-hour:
- a 2-epoch run: sx_base took 49.5 min, plus loading and holdout_eval;
- a 2-epoch continuation with 20 evals: 1-epoch continuations took 19-28 min.

Two lanes per GPU.

| free GPUs (from 4-7) | allocation | Q1 stage 1 + Q2 done after |
|---|---|---|
| 1 | Q1 on both lanes (≈ 6 h), then Q2 (≈ 5.5 h) | ≈ 11-12 h |
| 2 | GPU A: Q1, GPU B: Q2 | ≈ 6 h |
| 3 | GPU A: Q1; GPUs B, C: Q2 (≈ 3 h), then B, C start cd_lane with the same `$J` to take unclaimed Q1 jobs | ≈ 4 h |

Q1 stage 2 adds 3-9 jobs (≈ 1.5-2 h on one GPU). GPU N was idle at 04:31 UTC. GPUs N held F12 fine-tunes and other jobs (≈ 76-[n] GB used); `check` decides.

### 4.4 Conditional follow-ups and how winners become ensemble families

1. **Q2 PASS or FAMILY → 10-epoch pair:** `bash nn_x/r6/queue_morning.sh e10 <arm> "<2 GPUs>"`.
   - F7 recipe, seeds 101/102 against F7 s101/s102, `--save_steps 50000,60000,70000,77500`. About 2-2.5 h wall on 2 GPUs.
   - Rules from `r6_eval.py e10`:
     - a recipe winner proceeds iff val Δ ≥ 0 AND holdout ≥ −1 SE;
     - a family candidate proceeds iff |Δ| ≤ 0.002 on both.
   - If cset proceeds, split it into `--cretr_m 8` and `--set_sim_bias` pairs, for attribution.
2. **Q1 w10 ≥ 0 on val and holdout at stage 1 → wd 10-epoch pair:** `queue_morning.sh e10 wd01` (wd 0.05 at lr 2e-3, so lr·wd = 1e-4).
3. **Q1 PASS-CD → family F7x** = the arm on all 7 F7 roots (4 runs already exist; stage 2 adds 3). F7b, the 7 c0 runs, is a free R1 family candidate.
   - The winning arm becomes the default cooldown of every new 10-epoch launch.
   - Re-test capacity with it: a d192 10-epoch pair from scratch with the arm (≈ 4.5 GPU-h), since F12 saved no snapshots.
4. **Growing a structure family:** seeds 103-106 from scratch, with the same flags plus `--save_steps`. Use branches from snapshots only if the R1 read in §4.1 passes.
   - Each member then gets the fine-tune:
     ```
     CUDA_VISIBLE_DEVICES=<g> EXTRA=n2c_pre_v3 SUF=n2cpv3 FTARGS='--decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton' bash nn_x/r6/ft_list.sh <v>
     ```
     About 10 min per member on a GPU.
5. **Into the pool (lead only; unchanged project rules, CURRENT_STATUS §3):**
   1. Copy the member files into the live tree:
      - `nn_x/r6/preds/nn_<v>_{trainonly_val,trainonly_test,n2cpv3_val_oof,n2cpv3_test}.npy` → `preds/`;
      - `nn_x/r6/preds/holdout/nn_<v>_holdout.npy` → `preds/holdout/`;
      - `nn_x/r6/logs/<v>.log` and `<v>_n2cpv3_ft.log` → `nn/logs/`.

      This is needed because decide2.py reads only preds/, and presub_gate check C reads only nn/logs.
   2. **Gates:** `python scratchpad/decide2.py F7x:r6_cd_m02_r:101,102,103,104,105,106,107`.
      - (a) the mean single train-only score is within 0.002 of the best family on val AND the holdout;
      - (b) the 6-member family mix is within 0.002 of the best on val AND the holdout.
   3. **Candidate sets, pre-declared before reading:**
      - {current leader, i.e. P(F6\*) plus any family added by the overnight decision} + new family;
      - the same with the new family replacing its parent (F7x for F7).

      Choose among them on the holdout, using train-only members.
   4. **Picks.**
      - Pick 2 is replaced on a holdout win.
      - Pick 1 is replaced only with all of: val > 2 SE vs v12; no slice below −2 SE; holdout ≥ 0.32162; `src/presub_gate.py` PASS, with every WARN justified in the Log.

---------------------------------------------------------------------------------------------------------------------

## 5. Everything else, ranked by expected private gain per GPU-hour

| # | action | GPU-h | expected private gain | note |
|---|---|---|---|---|
| 1 | Finish the pre-declared overnight decision with F12 / F10 s505-506 (decide2.py) | 0 | ≈ 0 | F12 single 0.39377 / 0.31840 is within gate (a) of F7. The F10 precedent says the pool gain is ≈ 0, but the decision is pre-declared |
| 2 | Pre-launch steps §4.0 | 0 | enables Q1/Q2 | - |
| 3 | **Q1 CD screen** | ≈ 6 (+1.5 for stage 2) | +0.0003..+0.0008 if PASS; P ≈ 0.25 | the only test of the measured mechanism; shortest path to a family (re-cooling 7 roots) |
| 4 | **Q2 structure screen** | ≈ 5.5 (+4-5 per 10-epoch pair) | +0.0002..+0.0007 if an arm survives; P ≈ 0.3 for any | a4u is the best-founded arm |
| 5 | CPU builds today: wave-2 flags in nn_x/r6/train.py (`--drop_cd`, `--pool_p`, `--aug_cand/--aug_hist`), with flag-off bit-identity tests (T2-style) | 0 | enables wave 2 | `--pool_p` is a one-line change at train.py:284 |
| 6 | Add `--save_steps 50000,60000,70000,77500` to every future 10-epoch launch | 0 | keeps every family re-coolable | F10/F11/F12 launchers lack it |
| 7 | Wave-2 CD arms (m0, aa5, d20, pp02, aug), 4 roots each | ≈ 7-10 | as Q1 | run only the arms whose mechanism Q1 has not already refuted |
| 8 | Stale-val adapter for r6 checkpoints (CPU) | 0 | needed only for a cold/warm trade | r6_design §4.5.3 |
| 9 | LLM matched-data test | 2-6 | ≈ +0.00002..+0.00008 | already the lead's lowest-priority job; no new information channel |
| 10 | Cost levers: 7-epoch schedule (P1), batch ×2 (P3), branch families (R1) | - | ≈ 0 now | GPU is not the binding constraint with 27 days left; revisit only if many families must be built late |

### Do NOT
- **TTA of any kind.** All three arms fail; T3 is harmful.
- **Weight-average plateau snapshots or soup branches.** The 7-member ensemble lost −0.0025 val / −0.0016 holdout.
- **Run longer plateaus, more than 10 epochs, or lr above 2e-3.** The plateau mean function is saturated by 60k steps.
- **Add width, depth or encoder variants under the current regulariser.** d192's end-of-run decline is 3-5x larger and it peaks lower; DIN is closed; GRU adds nothing to the pool.
- **Treat 2-epoch nulls on wd or dropout as closed.** Log them as "inconclusive at 2 epochs".
- **Re-cool already-cooled checkpoints.** Use pre-cooldown snapshots only.
- **Run a continuation without `--loss_norm feed`.**
- **Use cooldown mix 0.3 at 10 epochs, or short random-like feeds in the pool.** The proxy loses −0.0005 with them.
- **Shorten max_hist.**
- **Increase the batch for throughput alone.**
- **Run the same hyperparameter arm in both r6 and r7.**
- **Choose between candidates less than 0.003 apart on public.**

---------------------------------------------------------------------------------------------------------------------

## 6. Evidence index
- Family table: `scratchpad/r6/plan/family_decline.{py,txt}`, which parses nn/logs/{v17,v22,v23,v24,v26,v14}_*.log.
- d192 curve: `nn/logs/v26_d192_pre2s_s70{2,4,6,7}.log`.
- Pool size and steps: `nn/logs/v17_pre2s_s101.log:7-8`.
- loss_norm deviation: `nn/logs/v21_cd_v14_s9{2,4,6}.log:1`.
- CD launcher: `nn_x/r6/cd_lane.sh`.
  - `bash -n` passes. DRY prints 12 commands.
  - All parse with train.py's own `parse()`: mix / wd / aux_anneal as intended, loss_norm feed, and the train.py:186 init assert satisfied.
  - A bad arm, root or GPU is refused.
- CD read-out: the exact §4.1 command was run on CPU with stand-in names (F7 s101-104 as c0, F9 s401-404 as the arm, output redirected to scratch: `scratchpad/r6/plan/readout_test.{sh,txt}`). It gave member val −0.00103 ± 0.00055 and holdout −0.00072 ± 0.00032, the SE scale behind the §4.1 rules; arm-control correlation was 0.932-0.942.
- TTA: `solvers/r6/out/tta_eval.json`. Queue and rules: `nn_x/r6/queue_morning.sh`, `nn_x/r6/r6_eval.py` header. Review patch: `scratchpad/r6/review/tcnt/`.
