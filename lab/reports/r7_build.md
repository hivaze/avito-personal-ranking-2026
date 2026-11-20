# r7 build: flag-gated architecture, tabular-encoder and hyperparameter arms in nn_x/r7 (GPU-tested)

Written 2026-11-12, 04:35-05:20 UTC, by the r7 build agent. Inputs: reports/r7_research.md (arm specs, §5-6),
CURRENT_STATUS, r6_recap, R6_PLAN, nn/model.py and nn/train.py (in full), nn_x/r6 (read-only).
Nothing under nn/, src/, preds/, subs/, nn_x/r6/ or nn_x/llm/ was written. Every GPU process was checked against the GPU
rules first (<= 4 processes, >= 80 GB headroom after it) and logged in nn_x/r7/logs/RUNNING.txt. No r7 process is running
now. **No screen run has been launched: the selection rules in §4 are pre-declared here, before any r7 screen result.**

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **All 13 r7 arms are implemented behind flags, and none was dropped.** That is the 12 selected arms plus the postln
   reserve. With every flag off, the code reproduces nn/ bit for bit: init, global RNG stream, cfg, eval logits, train
   loss with dropout, and gradients under deterministic kernels.
2. **Tests: 321/321 GPU unit checks pass** (`nn_x/r7/tests/test_r7.py`, final run on GPU N).
   - **5 end-to-end smoke runs** through `screen.sh` on real data (GPUs N): flag-off base (twice), a combination of
     every mutually compatible arm incl. batch2s and the r6 arms, dwa, and postln.
   - `train.py --save_holdout` writes holdout preds **bit-identical** to `holdout_eval.py`. This saves one train-split
     reload per run.
3. **The tests found two real bugs, both fixed:**
   - **periln was silently inactive.** A Module-valued class attribute (`Block.ln_ao = None`) shadowed the registered
     submodule, because `nn.Module.__getattr__` is only a fallback. The arm would have run as the baseline. A new "arm is
     active" check now guards every arm.
   - **ftok crashed at 65,536 rows.** It attends over B·M rows, and at 65,536 rows the fused SDPA kernels exceed the CUDA
     grid limit of 65,535. That size occurs in every `predict()` call and under batch2s, so every ftok run would have
     died at its first eval. Fixed by chunking at 32,768 rows. A predict-size (1,024 × 64) test now covers every arm.
4. **The r6-owned arms are NOT in the r7 batch.**
   - `nn_x/r6/queue_morning.sh` launched at 04:51 UTC on GPU N: `r6_base_s23` is done (val 0.39335, 23.8 min) and
     `r6_a4u_s23` is running. r6 also dropped wd01 and amended a4u (`--retr_tcnt`, added 04:48).
   - Per the task rule, read `nn_x/r6/logs/r6_eval_screen.txt` instead.
   - The flag-gated port (r6 code as of 04:20) stays in nn_x/r7 and is tested equal to nn_x/r6/model.py. It is
     pre-`--retr_tcnt`, so do not use it to replicate r6's a4u.
   - It is still needed for one pre-declared case: if r7 resdrop0 and r6 adrop0 both promote, stage 2 runs `drop0`
     (= `--drop 0`, all sites; arm `drop0` in screen.sh).
5. **Deviations from the spec (all measured; details in §2):**
   - **hc4 costs 1.43× per step** (spec ~1.1×). The cost is memory-bound fp32 stream arithmetic on [B, M, 4, d], which
     is needed for exactness.
   - **ftok costs 1.47× and +181K parameters** (spec ~1.2×, ~165K). It pushes 9 tokens × 32K candidates through a full
     block.
   - **batch2s costs 0.82× per sample.** The pre-declared COST-PASS needs ≤ 0.75× train minutes, so it may well fail on
     cost even if it is non-inferior.
   - derf replaces `lnc` only in the 2 cross blocks, where it is used. In self-attention blocks `lnc` is dead code in nn/
     too.
   - In combinations, muon_in also moves the other input-reading projections to AdamW: pbld.P, the ftok group inputs and
     cretr.wk.
6. **Ready to launch:** `bash nn_x/r7/screen.sh <gpu> <arm> <seed>`, then `eval_screen.py screen`.
   - 29 runs: r7_base s23/s24, the A/A rerun r7_base_s23r, and 13 arms × 2 seeds.
   - About 25-50 min per run (r6_base_s23 took 23.8 min; sx_base took 49.5 min at 3 processes per GPU) × the cost
     factor in §3.4.
   - **GPU capacity right now (05:15):** every GPU carries 1-2 other processes. Under the 80 GB headroom rule a GPU
     with ~[n] GB free takes **one** r7 run.
   - The batch2s COST rule needs base and batch2s of the same seed on the **same GPU in the same wave**: about [n] GB
     free, i.e. a GPU that is otherwise nearly empty.

---------------------------------------------------------------------------------------------------------------------

## 1. Code copy and output-path audit (nn_x/r7)

- **Copy:** nn/*.py and nn/*.sh copied, plus `feat_cache -> ../../nn/feat_cache` (symlink, read-only).
  - `data.feat_path` never links or copies into the shared cache; it raises instead.
  - `train.py` fails loudly on a missing xt feature instead of dropping it silently.
- **common.py:**
  - CKPT / LOGS / PREDS point to `nn_x/r7/{ckpt,logs,preds}`; holdout preds go to `nn_x/r7/preds/holdout`.
  - GPU guard: exactly one id, 0-7.
  - `torch.set_num_threads(min(16, OMP_NUM_THREADS))`.
- **Shared-path writers:**
  - Redirected to nn_x/r7: ensemble.py, zmean.py, pcmp.py, and every `.sh` (`../preds` → `preds`, absolute venv path).
  - Disabled with `SystemExit`: make_cand.py (writes subs/) and build_w3b_feats.py (writes data/feats_new and
    nn/feat_cache).
  - The legacy launchers (`run_*.sh`, `stage2_ft*.sh`, `mix_ft*.sh`, `post_final6.sh`) refuse to run unless
    `R7_ALLOW_LEGACY=1`.
- **Audit:** a grep of every `np.save`, `torch.save`, `open(...'w')`, `write_text` and `mkdir` in nn_x/r7/*.py finds only
  nn_x/r7 targets. The variant guard requires names that start with `r7_`.
- **Bytecode caches:** the tests import nn/model.py and nn_x/r6/model.py read-only, with `sys.dont_write_bytecode`. The
  existing `__pycache__` timestamps there predate this session.
- **New files:**
  - `nn_x/r7/screen.sh` (launcher);
  - `nn_x/r7/eval_screen.py` (paired evaluation);
  - `nn_x/r7/tests/test_r7.py` (GPU tests);
  - `nn_x/r7/tests/mini_split.py` (r6's memory-mapped mini split, plus a device argument).

## 2. Arms: flags and implementation

Shared mechanics:
- Every treatment module is built in `Ranker._r7_build` **after** all baseline modules (and after r6's cretr).
- Modules that draw random numbers are built under `extra_init` (forked CPU RNG), so the baseline `_EXTRA` counter
  sequence and the global RNG are untouched.
- cfg keys are stored only when non-default, so a flag-off checkpoint loads in nn/model.py.
- Optimiser: `train.build_opts` keeps nn/'s rule (Muon for every 2-D Linear with min dim > 1; AdamW for the rest; decay
  iff 2-D and no "emb" in the name). Two additions:
  - `R7_NO_DECAY_PREFIX = ("hc.", "pbld.w1", "pbld.b1", "pbld.b2")`;
  - `--muon_inputs`, `--muon_momentum` and `--adam_betas`, whose defaults are nn/'s hard-coded values.

| key | flag(s) | implementation (nn_x/r7/model.py, train.py) | exact at init | Δ params | partition (Muon / AdamW-decay / no-decay tensors) |
|---|---|---|---|---|---|
| base | none | nn/ | - | 1,008,512 | 46 / 12 / 93 |
| muon_in | `--muon_inputs adamw` | c_mlp.0, c_lin, h_lin, u_lin, retr.wk → AdamW decay (wd 0.02, same lr / schedule); logged | yes (same model) | 0 | **41** / 17 / 93 |
| batch2s | `--cand_budget 65536 --max_feeds 1024 --lr 1.414e-3 --warmup 250 --eval_every 500 --muon_momentum 0.90 --adam_betas 0.8,0.96`; e10: lr 2.83e-3, eval 1250, save 25k/30k/35k/38.75k | SDE square-root rule, pre-declared in r7_research §4.3 | yes | 0 | 46 / 12 / 93 (momentum 0.90, betas 0.8/0.96) |
| resdrop0 | `--res_drop 0` | `Block.dr`, the FFN-hidden dropout (`ff[2]` / `SwiGLU.dp`), `HistRetrieval.dp` (+ `CandRetrieval.dp`) → p 0; `Attn.drop` stays 0.1 (test checks every site) | yes (eval) | 0 | = base |
| pbld | `--pbld 16` | `PBLD`: w1 ~ N(0, 0.1²), b1 ~ U[−π, π], W2 [104, 16, 3] Linear-default init, b2 = 0, P = Linear(312, 256) zero-init; `c_mlp[2](GELU(c_mlp[0](X) + P(per)))` | yes | +88,504 | 47 / 13 / 96 (P Muon; W2 decay; w1 / b1 / b2 no-decay) |
| hc4 | `--hc 4` | `HCUnit` × 8 around cross.attn, cross.ff, set.attn, set.ff (×2 blocks): A_m = e_{k mod 4}, A_r = I, B = 1 (learnable), W_m / W_r / W_b = 0, RMSNorm gain, s_a = s_b = 0.01; stream mix in element-wise fp32 (no bf16 / TF32 matmul on the stream); readout = pairwise mean, so exact; then dvec | yes (bit-exact) | +7,376 | 46 / 12 / 165 (all 72 HC tensors no-decay) |
| dwa | `--dwa` | `ParameterList` of sizes 2, 3, 4, 5, one-hot on the last entry; acc = Σ_j α_j y_j in fp32 after each block | yes | +14 | 46 / 12 / 97 |
| diff2 | `--diff_attn cross` | in the 2 cross Attn modules: q2 = Linear(d, d), lam = Linear(d, 4) (extra_init), bias_scale2 = linspace(0, 8, 4); o = o1 − σ(lam x)·o2 with shared k / v, own sim-bias scale, attention dropout in both calls, no per-head norm; then the A1 gate and W_O; mean λ per head logged | no | +34,064 | 50 / 12 / 99 |
| swiglu | `--ffn swiglu` | original `ff` constructed first (baseline RNG draws), then `SwiGLU(128, 170)` built under extra_init replaces it in all 5 blocks | no | −860 | 51 / 12 / 98 (g, u, o separate) |
| derf | `--norm derf` | `Derf`: α = 0.5, s = 0 scalars, weight 1, bias 0; replaces ln1 and ln2 in all 5 blocks, lnc in the 2 cross blocks (the only ones that use it), head[0] and aux_head[0]. c_ln, h_ln and retr.v_ln stay LayerNorm | no | +28 | 46 / 12 / 121 |
| hl2 | `--hist_extra 1` | extra `Block` built under extra_init and appended to `hist_enc`; attn.o and the FFN output (weight + bias) are zeroed after any FFN swap | yes | +149,248 | 53 / 12 / 106 |
| periln | `--periln` | `ln_ao` and `ln_fo` LayerNorms on every sublayer output, before the dropout, in all blocks; gated by the bool `Block.periln` (bug fix, §0.3) | no | +2,560 | 46 / 12 / 113 |
| ftok | `--ftok 8` | `FTok`: the 8 groups of r7_research §5.12, asserted to partition 0..103 with n_dense = 104 and stored in cfg; per-group Linear(n_g, 128) + group embedding + CLS (N(0, 0.02)); one `Block` over the 9 tokens with no mask, in chunks of ≤ 32,768 rows; Zf zero-init; added before c_ln | yes | +181,248 | 62 / 12 / 117 |
| postln (reserve) | `--postln` | `x = ln1(α x + dr(attn(x, ·)))`, `x = ln2(α x + dr(ff(x)))`; v, o, ff[0], ff[3] scaled by β at init (candidate blocks N = 4: α 1.682, β 0.420; history N = 1: α 1.189, β 0.595); test checks exactly β × the nn/ init | no | 0 | = base |
| drop0 (stage-2 combo only) | `--drop 0` | existing flag | - | 0 | = base |
| r6 port (not in batch) | `--retr_uniq`, `--cretr_m 8 --set_sim_bias`, `--typed_cross`, `--attn_drop 0`, `--retr_m 32 --retr_prod`, `--wd 0.1` | nn_x/r6 diff as of 04:20 (no `--retr_tcnt`); state_dict, eval logits and train loss equal to nn_x/r6/model.py | as in r6 | as in r6 | - |

Mutually exclusive (asserted): hc / dwa / postln, and postln with swiglu / derf / periln. Everything else combines. Arms
join with `+` in screen.sh, e.g. `muon_in+pbld`.

Logged per run (in the log and in the run json `diag` / `diag_init`):
- the r7_research §2.4 scale quantities: sim-bias scales per cross block, c_ln gain, retr.a, cross0 W_O RMS and spectral
  norm, and the spectral norms of c_mlp.0 / c_lin / u_lin / h_lin / retr.wk;
- every arm's own scalars: HC A_m / A_r diagonal and off-diagonal / B / s_a / s_b / W RMS, the DWA α matrix, Derf α / s /
  gain per site, PBLD ‖P‖, ftok ‖Zf‖, hl2 ‖W_O‖, Peri-LN gains, diff2 bias_scale2 and mean λ per head;
- peak GPU memory, minutes per 1,000 steps, GPU id and start time (for the batch2s same-GPU / same-wave rule).

## 3. Tests (GPU; `CUDA_VISIBLE_DEVICES=<g> nn_x/r7/tests/test_r7.py`; log nn_x/r7/logs/test_r7.{log,json})

Base = the sx_base_s23 cfg and trained weights, i.e. the current 2-epoch screen recipe. Data = real train feeds through a
memory-mapped mini split moved to the GPU.

### 3.1 Flag-off identity (T0-T2)
- **CLI:** `screen.sh flags base` reproduces every sx_base_s23 arg (only variant and `--no_test` differ). Every r6 / r7
  model flag is at its OFF default.
- **Arm flags:** each arm's screen.sh flags parse to exactly the specified values and change nothing else.
- **Stage-2 line:** `screen.sh flags base e10` reproduces the v17_pre2s_s101 (F7) args, apart from variant / seed /
  save_steps.
- **Init (T1):**
  - cfg, state_dict (keys, order, values), parameter order and the global CPU RNG state after construction are equal to
    nn/'s.
  - Per arm: 0 baseline tensors change, the RNG state is the same, and cfg gains only the arm's keys.
  - Exceptions, as specified: swiglu removes the 20 replaced FFN tensors; postln changes exactly the 20 DeepNorm tensors
    to β × nn/.
- **Trained weights, training precision (T2):**
  - eval logits and train-mode loss (dropout on, same seed) are bit-identical;
  - all gradients are bit-identical under deterministic kernels (math SDPA, `use_deterministic_algorithms`);
  - with default kernels both nn/ vs nn/ and r7 vs nn/ differ by ~1e-5. That is the GPU's own non-deterministic
    backward (SDPA / index_put atomics), reported as information.
- **r6 port (T2b):** a4u, cset, typed, adrop0 and a4w are identical to nn_x/r6/model.py: init state_dict, warm eval
  logits and train loss.

### 3.2 Exactness at init (T3)
- **Bit-identical eval logits vs nn/,** at the fresh init and at a warm start from the trained weights: muon_in, batch2s,
  resdrop0, pbld, **hc4**, dwa, hl2, ftok (and the r6 adrop0 / wd01 / a4u / a4w zero-init parts at the fresh init).
- **Non-exact by design** (warm-start eval-logit change, mean |d|, information only):

  | arm | mean \|d\| |
  |---|---|
  | diff2 | 0.08 |
  | swiglu | 0.29 |
  | derf | 0.66 |
  | periln | 0.46 |
  | postln | 0.29 |

### 3.3 Structure (T4-T7, fp32 with TF32 off), every arm
- logits [B, Mx] are finite;
- valid logits do not depend on padded candidates or history (max |d| 0);
- they do not depend on the **amount** of padding (a feed alone vs next to a longer feed, ≤ 4e-6). Position-keyed jitter
  breaks exact ties between repeated items. Without it the gap is the same (≤ 4e-6) for every arm.
- no cross-feed leakage: perturbing all of feed 0's inputs changes no other feed (max |d| 0);
- permutation equivariance within a feed (≤ 2e-6), on the real batch and on a jittered one;
- every used parameter gets a finite gradient, and every new parameter a non-zero one after a small perturbation of the
  zero-init output layers;
- zero-init outputs (pbld.P, ftok.Zf, hl2 W_O / FFN out) and HC's static A_m / A_r / B get a gradient at the true init;
- **"arm is active":** the perturbed new parameters change the eval logits. For resdrop0 / adrop0 the train-mode outputs
  differ at the same seed, and every dropout site has the specified p.
- every arm runs a predict-size batch (1,024 feeds × 64 = 65,536 candidates) in eval and train mode. Exact arms give eval
  output equal to nn/.

### 3.4 Cost (T9)
Setup:
- Real full train batches: 512 feeds × Mpad 64 = 32,768 slots, history 256.
- Each step = dense features + forward + backward + clip + optimiser step, using train.py's own `build_opts`.
- Median of 20 steps after 5 warm-up steps, on GPU N, shared with one LLM process. The baseline re-run at the end is
  1.00×, so there is no drift.

| arm | ms/step | × base | Δ activation peak | 25-step train loss |
|---|---|---|---|---|
| base | 84.4 | 1.00 | - | 6.124 → 5.884 |
| muon_in | 84.0 | 1.00 | +0.0 GB | 6.124 → 5.838 |
| batch2s (1,024-feed step) | 139.2 | 1.65 per step = **0.82 per sample** | +3.3 GB | 6.166 → 5.838 |
| resdrop0 | 83.9 | 0.99 | −0.3 | 6.123 → 5.870 |
| pbld | 93.1 | 1.10 | +0.3 | 6.125 → 5.878 |
| hc4 | 120.9 | **1.43** | +2.0 | 6.124 → 5.877 |
| dwa | 87.9 | 1.04 | +0.1 | 6.124 → 5.881 |
| diff2 | 93.7 | 1.11 | +0.2 | 6.123 → 5.886 |
| swiglu | 88.7 | 1.05 | +0.1 | 6.122 → 5.879 |
| derf | 89.0 | 1.06 | +0.8 | 6.116 → 5.920 |
| hl2 | 97.1 | 1.15 | +0.7 | 6.122 → 5.883 |
| periln | 89.3 | 1.06 | +0.3 | 6.120 → 5.864 |
| ftok | 124.0 | **1.47** | +1.5 | 6.124 → 5.879 |
| postln | 84.6 | 1.00 | −0.0 | 6.127 → 5.900 |

Notes on the table:
- **Whole-process memory** (smoke runs, `peak_reserved`): flag-off trainer 54.9 GB; batch2s + every other arm together
  61.5 GB.
- screen.sh therefore reserves 62 GB per run (72 GB for batch2s, 66 GB for hc4 / ftok) plus the 80 GB headroom.
- On an idle GPU the base step is 37 ms. Fusing hc4's arithmetic (one coefficient matmul, one broadcast mix) took it from
  1.59× to 1.48× on the idle GPU. Unmasked SDPA took ftok from 1.59× to 1.46× on the shared GPU.
- Evaluation cost per run is the same for every arm (the same number of evals), so the run-level ratio is a little
  closer to 1.

### 3.5 Pipeline smokes (screen.sh with `R7_SMOKE=1`: 20K train feeds, 1 epoch, real data, GPUs N)
- `r7_smoke-base_s23`, the combination of every compatible arm (incl. batch2s and the r6 arms), `r7_smoke-dwa_s23` and
  `r7_smoke-postln_s23` all complete.
- Each writes ckpt, the json (with gpu / start / peak / diag / diag_init), val preds and holdout preds.
- The flock, the GPU check and the RUNNING.txt START/END lines work.
- holdout_meta / holdout_y are equal to the shared preds/holdout ones.
- **holdout preds:**
  - The first comparison against holdout_eval.py differed by mean |d| 6e-5 (NDCG Δ 6e-7), because train.py runs with
    TF32 and holdout_eval.py without.
  - `--save_holdout` now predicts with TF32 off and is **bit-identical** to holdout_eval.py, verified on a rerun.
- Smoke artefacts are named `r7_smoke-*`, so they cannot collide with screen variants.

## 4. PRE-DECLARED selection methodology (implemented in nn_x/r7/eval_screen.py; frozen before any r7 screen run)

**Pairing:**
- Seeds 23 and 24. Baseline = `r7_base_s23` / `r7_base_s24`, re-run in this code copy in the same batch of launches, with
  each arm's seed run next to its baseline seed where capacity allows.
- Optional A/A: `R7_SUFFIX=r bash screen.sh <g> base 23` gives `r7_base_s23r`; `eval_screen.py aa` checks it.
  - Fork check: |Δval| ≤ 0.0010 and |Δhold| ≤ 0.0005.
  - Calibration check: |Δ| ≤ 2·sqrt(SE_feed² + σ²). If this fails, raise `--sigma_*` to the A/A |Δ| before reading any
    arm.

**Metric:**
- Per-feed full NDCG (`src/metrics.ndcg_per_feed`) of the train-only preds at the holdout-selected checkpoint.
- Val = 14,423 feeds; holdout = the fixed 30K random-like M≥50 train feeds.
- Paired per-feed deltas are averaged over the seeds.
- **Calibrated SE** = sqrt(SE_feed² + σ_run² / n_seeds):
  - σ_run is the run-to-run SD of a same-config delta (GPU non-determinism);
  - stage 1: val 0.00035 (same-seed reruns differ by ±0.0002-0.0005 at 2 epochs), holdout 0.00015;
  - stage 2: val 0.0007, holdout 0.0003 (r6_recap §4.5, r7_research §6.3).
  - Slices use SE_feed. This is stricter for warnings.
- **Validation of the machinery:** on the old sx arms (pl20 / tau30 / agedrop) eval_screen.py reproduces r6_eval.py's
  deltas and slice SEs exactly. Only the "all" SE grows by the σ term (e.g. 0.00045 → 0.00051).

**Stage 1 (2-epoch screen) PROMOTE iff all of:**
- pooled val Δ > 1.5 SE_cal;
- pooled holdout Δ ≥ 0;
- both seeds' val Δ > 0 **or** both seeds' holdout Δ > 0.

**batch2s, additionally:**
- **COST-PASS** iff pooled val Δ ≥ −0.0005 AND holdout Δ ≥ −0.0003 AND train minutes ≤ 0.75× the paired baseline on the
  same GPU in the same wave (start within 60 min), for every comparable seed. No comparable pair means "COST
  undetermined".
- Stage 2 is the batch2s e10 flags.
- Stage-2 adoption also needs the 2-member TO ensemble ≥ the baseline pair on the holdout, and corr(arm s23, arm s24) ≤
  corr(base s23, base s24) + 0.005.

**Readouts (information only, not part of the rule):**
- WARN for slices < −2 SE_feed: val M<50, M≥50, Tue, Thu; holdout late 20%.
- **SCALE-SUSPECT** (r7_research §2.4): the arm's |final − init| growth exceeds 1.25× the paired baseline's on ≥ 2 of
  {top sim-bias scale, c_ln gain, retr.a, cross0 W_O RMS}. A promotion with this tag goes to stage 2 with no other
  weight.
- Diversity: rho_bb, rho_x, rho_aa and rho_same (within-feed z), and the mixed 2-member ensemble.
- FAMILY-CANDIDATE (r6 definition): a non-promoted arm with val ≥ −0.0010, holdout ≥ −0.0005, rho_x ≤ rho_bb − 0.005 and
  a mixed ensemble ≥ 2× base on the holdout.

**Stage 2 (10 epochs, `screen.sh <g> <arm> <seed> e10`, F7 recipe):**
- **ACCEPT** iff val Δ > 2 SE_cal AND holdout Δ ≥ 0 AND no slice < −2 SE.
- Pre-declared combinations:
  - `drop0` if resdrop0 and r6 adrop0 both promote;
  - all promoted arms together if ≥ 2 promote and they act on different parts (r7_research §6.4).

**Stage 3 (report only; the lead decides):**
- z-mean of the 2 treated members vs the 2 baseline members (val, holdout, slices, SE_cal with σ/√2);
- correlations;
- every run's end − selected decline (late-cooldown memorisation; required for resdrop0).

## 5. How to run

```
bash nn_x/r7/screen.sh check <gpu> <arm>                       # preflight (<= 3 processes, free >= need + 80 GB)
setsid nohup bash nn_x/r7/screen.sh <gpu> <arm> <seed> > nn_x/r7/logs/r7_<arm>_s<seed>.screen.out 2>&1 < /dev/null &
WAIT=1 ... screen.sh ...                                       # poll every 180 s instead of refusing
bash nn_x/r7/screen.sh flags <arm> [e10]                       # print the exact train.py arguments
CUDA_VISIBLE_DEVICES="" nn_x/r7/eval_screen.py screen [arm ...]   # stage 1 (default: the 13 r7 arms)
CUDA_VISIBLE_DEVICES="" nn_x/r7/eval_screen.py e10 <arm ...>      # stage 2 + 3
CUDA_VISIBLE_DEVICES="" nn_x/r7/eval_screen.py aa                 # A/A
```

**Launcher behaviour:**
- Stage 1 = the r6 BASE2 / sx_base line plus `--no_test --save_holdout`.
- The per-GPU flock covers check + launch + data load, so two r7 runs never load the train split on one GPU at once.
- PIDs go to `nn_x/r7/logs/RUNNING.txt`. Stop jobs only by PID.

**Job order:**
1. r7_base s23 / s24;
2. muon_in, batch2s (next to its baseline seed on the same GPU, for the cost rule), resdrop0, pbld, hc4, dwa, diff2,
   swiglu, derf, hl2, periln, ftok;
3. the A/A r7_base_s23r and postln on idle lanes.

## 6. Workflow snapshot (05:15 UTC)

| GPU | running |
|---|---|
| 0 | F10 GRU s505, s506 |
| 1 | LLM Qwen3.5-9B L1 s23 |
| 2 | F12 d192 s701, s703 |
| 3 | r6 Q2 screen: r6_a4u_s23 training; holdout_eval of r6_base_s23 |
| 4 | r6 Q1 cooldown continuations: r6_cd_m02_r101; holdout_eval of r6_cd_c0_r101 |
| 5 | LLM 0.6B s23, Qwen3.5-9B s24 |
| 6 | LLM Qwen3.5-4B s24 |
| 7 | LLM Qwen3.5-4B s23 |

## 7. Caveats

- **Screen power:** the expected effects of these blocks are 0 to +0.0008 (r7_research §0.5). Most arms will read null at
  2 epochs. A null is still the deliverable for the "modern blocks" question.
- **Scale regime:** arms with free scale knobs (hc4, dwa, derf, periln) can win at 2 epochs on speed of scale growth
  alone. The SCALE-SUSPECT readout exists for this. Only the 10-epoch pair counts.
- **batch2s** changes the batch composition, so its pairing is weaker than for the other arms.
- **hc4 and ftok** cost about 1.45× per step. Their 10-epoch pairs cost about 1.4× a normal pair.
- **Timing** numbers come from a shared GPU (±5%). The batch2s cost rule uses the real run minutes from the same GPU and
  wave, not this table.
