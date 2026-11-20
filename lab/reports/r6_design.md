# R6 design: six flag-gated arms and a pre-declared morning screen

Written 2026-11-12, 03:45-04:45 UTC, by the r6 design agent. **CPU only** (`CUDA_VISIBLE_DEVICES=""`, ≤ 12 threads, ≤ 2 processes). **No GPU job was launched**, and `nn_x/r6/queue_morning.sh` was only syntax-checked and dry-run (`DRY=1` prints the plan, starts no process and makes no GPU query).

Nothing under nn/, src/, pretrain/, data/feats/, preds/, subs/ or nn_x/solvers/ was written. `find -newer` over those trees shows no file of mine; the only new files in preds/ are the live F10 GRU outputs.

Inputs: reports/r6_recap.md, reports/r6_external.md, reports/r6_tta.md, CURRENT_STATUS (to the 03:45 line), COVERAGE_MAP, SOLUTIONS_REVIEW, sol_ensemble §2.5-2.6, sol_errors, nnx_exp_attn, nn_report, and the code (nn/model.py, nn/train.py, nn/data.py, nn_x/solvers/*).

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

**Six arms, ranked by expected ensemble value per GPU-hour.** Each arm is behind a flag whose default reproduces the current run exactly.

| rank | key | flag(s) | type | mechanism, one line |
|---|---|---|---|---|
| 1 | **a4u** | `--retr_uniq` | structure (A4 fix) | Retrieve over distinct items only (newest occurrence), with fp32 scores and a repeat-count value. In the baseline top-16, **30-32% of valid slots repeat a higher slot's item** (measured here), so A4, the strongest component, wastes a third of its slots. |
| 2 | **wd01** | `--wd 0.1` | hyperparameter, never varied (285/285 runs) | Decay timescale τ = 1/(lr·wd): 50k steps (2.5× the screen run) → 10k steps. Muon long runs want stronger decay. |
| 3 | **cset** | `--cretr_m 8 --set_sim_bias` | structure, new | TabR retrieval over each candidate's 8 most similar same-feed competitors (difference vectors plus the neighbours' tokens), and a cosine prior in the candidate set attention, which currently has no prior and is neutral. |
| 4 | **typed** | `--typed_cross` | structure, new | The last cross block attends only to the user token and target actions (event_type 1 from source 1): 28% of history events, median 51 per feed. This is Avito's production L3 filter. |
| 5 | **adrop0** | `--attn_drop 0` | hyperparameter, never varied | Remove attention-probability dropout, which drops the single matched event in peaked sim-bias heads. Residual, FFN and retrieval dropout stay at 0.1. It is 0.54-0.62× CPU cost per step. |
| 6 | **a4w** | `--retr_m 32 --retr_prod` | hyperparameter + structure | A4 width (only m=16 was ever run) plus DIN's product term T3(k_c ⊙ k_j). |

**Morning queue:** `bash nn_x/r6/queue_morning.sh screen "<GPUs from 4-7>"`.
- It runs 13 jobs: the 6 arms × seeds 23/24, plus one A/A fork check (`r6_base_s23`).
- The recipe is the 2-epoch current recipe on cvnd_pre2s (lr 1e-3, mix 0.1). Each GPU gets two lanes, so at most 2 processes per GPU.
- **Controls:** the existing `nn_x/solvers` runs `sx_base_s23` and `sx_base_s24` are reused. Their args are identical to the r6 command line (test T12).
- **Cost:** ≈ 5.5 GPU-h in total. Wall-clock is ≈ 1.7 h on 4 GPUs, ≈ 2.5 h on 3 GPUs, or ≈ 3.3 h on 2 GPUs.
- **Evaluation:** the last lane runs `r6_eval.py aa` and `r6_eval.py screen` on the CPU. Results go to `nn_x/r6/logs/r6_eval_*.txt`.

**Pre-declared acceptance.** The rules were frozen in the `nn_x/r6/r6_eval.py` header before any r6 GPU run.
- **PASS** (recipe change or stronger member): pooled train-only val delta > 2 SE AND pooled holdout delta ≥ −1 SE.
- **FAMILY-CANDIDATE** (same strength, different function): all of
  - val ≥ −0.0010 and holdout ≥ −0.0005;
  - cross-seed correlation ρ_x ≤ ρ_bb − 0.005;
  - the mixed 2-member ensemble ≥ the 2× base ensemble on the holdout.
- Slices are reported. Any slice below −2 SE is flagged WARN.
- Both PASS and FAMILY-CANDIDATE lead to a **10-epoch pair** (seeds 101/102 against F7 s101/s102, same init and data order) before any family: `queue_morning.sh e10 <arm> "<GPUs>"`.

**Tests:** 79/79 CPU checks pass (`nn_x/r6/logs/test_r6.log`).
- The flag-off r6 model is **bit-identical** to nn/model.py with the trained sx_base_s23 weights on real train batches: eval logits, train-mode loss with dropout, and every gradient.
- Every arm keeps every baseline init and the global RNG stream.
- Masks hold, permutation equivariance holds within feeds, there is no cross-feed leakage, and gradients reach every new parameter.

**Honest expectation.**
- The screen's pooled val SE is 0.00045-0.00061 (calibration, §4.4), so its MDE is ≈ +0.0014.
- My subjective probabilities: a4u P(PASS) ≈ 0.25, the others 0.10-0.20. P(at least one arm reaches the 10-epoch pair) ≈ 0.6.
- Expected pool value of the whole screen ≈ +0.0005, consistent with r6_external §6 (+0.0005..+0.0015 for everything proposed).

---------------------------------------------------------------------------------------------------------------------

## 1. Ground rules and state

- **GPUs.** At 03:40 CURRENT_STATUS logs a user authorisation of some GPUs, and nn/common.py now accepts 0-7. This design task's hard rules reserve 0-3, so **`nn_x/r6/common.py:12-16` accepts only `CUDA_VISIBLE_DEVICES` ∈ {<ids>}**, and the queue refuses other ids. If the lead wants 0-3, they edit those two lines and the `check_gpus` case in queue_morning.sh. I did not add a bypass.
- **Process caps.** The queue runs 2 lanes per GPU. Before launching, it prints the number of compute processes already on each GPU and warns at ≥ 3, because the project cap is 4 per GPU.
- **Process records.** Lane PIDs go to `nn_x/r6/logs/RUNNING_r6.txt`. The lead should mirror them into nn/logs/RUNNING.txt. Stop lanes by PID only.
- **Timing.** F12 occupies GPUs N until about 06:30-07:30 (CURRENT_STATUS §4). The screen fits after that, or on any GPU with ≤ 2 processes.

---------------------------------------------------------------------------------------------------------------------

## 2. Hyperparameter and model-structure audit (the user's request)

Each knob gets a decision against the coverage. Evidence tags: [R] = report:line, [M] = measured here.

### 2.1 Hyperparameters

| knob (current) | ever varied? | decision | why |
|---|---|---|---|
| **weight decay 0.02** | never (285/285 runs, r6_recap §2) | **arm wd01** | τ = 1/(lr·wd) = 50k steps at lr 1e-3, which is 2.5× the 2-epoch run and 0.51 of a 10-epoch run. lr 2e-3 (2× lr·wd) halves the late decline (−0.0020 vs −0.0043, r6_recap §3.3). Moonlight: wd is crucial for Muon in long runs (r6_external §3 R3). Visible at 2 epochs: τ drops from 2.5 runs to 0.5. |
| **attention dropout 0.1** | never | **arm adrop0** | Peaked sim-bias heads lose the matched event 10% of the time (r6_external R3b). Residual, FFN and retrieval dropout are untouched. |
| retr_m 16 | never (62/62 retrieval runs) | **arm a4w** (with the product term) | Low prior on its own. The repeat problem (a4u) is the better-founded A4 lever. |
| residual dropout 0.2, aux_anneal 0.5, mix 0, P8 pool-aux fix, pool augmentation | never | **not in a 2-epoch screen**: needs the continuation harness (§6) | The mechanism is late-cooldown memorisation after ≥ 8 passes. 2-epoch runs never decline: 29/29 select one of the last two evals (r6_recap §3.3), so a 2-epoch from-scratch screen is blind to it. |
| batch 32768 / 512 (~317 feeds) | never | deferred | A cost lever only: B_simple 4.9K-9.4K feeds (r6_recap §4.2). Its risk is lower member noise, i.e. lower ensemble diversity (r6_recap §5), which only a 10-epoch 4-seed confirm can show. GPU time is not the binding constraint with 27 days left (≈ 2 h per 10-epoch member, nn/logs/v17_*.log). |
| cooldown_frac 0.2, lr floor 3% | never in a full run | not in the screen | Only visible at 10 epochs. The cost side is covered by P1 (cooldown from @60000, r6_recap §7), which needs no code. |
| lr 1e-3 / 2e-3 | yes | keep | lr > 2e-3 blurs the mean function (r6_recap §5). |
| warmup 500, Muon momentum 0.95, AdamW betas, clip 1.0 | never | skip | No instability anywhere (r6_recap §6 rank 8). |
| max_hist 256 | n/a (data cap) | keep | Crop TTA costs −0.0007 (192) to −0.0041 (64) on the ensemble; history length is binding (r6_tta §5.1). |
| d / heads | old recipe; F12 d192/6h running | wait for F12 | Capacity flat in every recipe probed (r6_recap §3.5). |
| blocks 3, hist_layers 2, ff 4 | old recipe only (ff never) | skip | heads 8 −0.0011, blocks 3 −0.0006 at 2 epochs; deep ≈ tf at 3/5/10 epochs (r6_recap §3.5). |
| fine-tune lr / epochs / EMA | partly | skip | Mix-level effect refuted (OVERFIT-4, ±0.0004). |

### 2.2 Model structure

| component | measured | decision |
|---|---|---|
| history target attention + sim bias | largest component (sim bias +0.0024, nn_report:68) | **typed** adds a target-action view in the last cross block |
| A4 retrieval (top-16, difference vectors) | +0.0019 val / +0.0033 holdout (nnx_VERDICT §2.3). About 90% of the gain is in T(k_c − k_j) (nnx_exp_attn:99). No measurable GPU cost (nnx_exp_attn:27). | **a4u** (duplicate-free retrieval), **a4w** (width, product term) |
| candidate set attention (2 blocks) | neutral in the old recipe (nn_report:68); Avito production +0.6% | **cset** gives it a cosine prior and a retrieval channel over competitors |
| gated attention, Muon/WSD, K9, inputs | adopted / saturated | none |
| GRU (F10), DIN (F11), d192 (F12) | running | none |
| SwiGLU/RMSNorm, GDCN head, dual-softmax head (R8) | never | after this screen, only if a FAMILY-CANDIDATE shows that structural diversity pays |

---------------------------------------------------------------------------------------------------------------------

## 3. The six arms

Cost per arm: 2 runs × ≈ 50 min at 2 processes per GPU (sx_base took 49.5 min at 3 per GPU, nn_x/solvers/logs/sx_base_s23.log:29), ≈ 0.8 GPU-h.

Values come from the frontier law (sol_ensemble §2.5: equal-mix gain = 0.0186(1−ρ) − 0.378·|gap| on val):
- a confirmed stronger member worth +0.001 single carries ≈ +0.0007 into the pool;
- a same-strength family with family-level ρ ≈ 0.97 is worth ≈ +0.0005.

EV = P(screen verdict) × P(10-epoch confirms ≈ 0.5) × value. The probabilities are my judgement and are labelled as such.

| arm | P(PASS) | P(FAMILY) | EV (pool val) | CPU fwd+bwd cost vs base (3 runs) | params |
|---|---|---|---|---|---|
| a4u | 0.25 | 0.10 | +0.00011 | 0.99-1.57× (L batch); 1.02-1.06× (S) | +1,152 |
| wd01 | 0.20 | 0.10 | +0.00010 | 1.00× | +0 |
| cset | 0.15 | 0.20 | +0.00010 | 1.04-1.44× (S); 1.16-1.30× (L) | +66,058 |
| typed | 0.10 | 0.20 | +0.00009 | 1.01-1.15× | +0 |
| adrop0 | 0.15 | 0.05 | +0.00007 | 0.54-0.62× | +0 |
| a4w | 0.10 | 0.05 | +0.00005 | 1.45-1.84× | +8,192 |

CPU ratios are noisy across the three test runs and are not GPU ratios. A4 itself had no measurable GPU wall-clock cost.

### 3.1 a4u: duplicate-free A4 retrieval (`--retr_uniq`)

**Mechanism.** The user's history repeats items: 30% of events repeat an item already in the history (r6_tta §5.2). Repeats have identical embeddings, so they tie in S and fill several of the 16 retrieval slots with the same item. Measured on real train batches (test T10, `test_r6.log`):
- **30.1-31.8% of valid top-16 slots are repeats of a higher slot**;
- **67-75% of candidate rows retrieve at least one repeated item**.

Which copy is retrieved depends on top-k tie-breaking, which differs between bf16 on the GPU and fp32 on the CPU, and with padding (r6_tta §5.2: 50% boundary ties in bf16). a4u fixes this in three parts:
- (i) one slot per distinct item, the newest occurrence (`first` mask), so the 16 slots cover up to 16 different items;
- (ii) fp32 scores for the top-k. On the GPU these are TF32, because train.py sets `allow_tf32`. Exact duplicates are removed by (i) regardless;
- (iii) the lost multiplicity comes back as a zero-init embedding of floor(log2(repeat count)).

**Code.**
- `history_dedup` at model.py:213-221;
- `HistRetrieval(..., uniq=True)`: `e_cnt` created last inside the retr fork at :179-181, used at :205-207;
- forward branch at model.py:565-569;
- data.py:158 adds `hidx` (the history item-table row, already computed in gather) to every batch. It is read only by this flag.

**Expected:** +0.0003..+0.0010 single. It is also more deterministic across devices.

**Risk:** repeat-heavy users may rely on multiplicity in the softmax mass, which the count embedding has to relearn. Warm-start change is mean |Δlogit| 0.03 (T3, info).

### 3.2 wd01: weight decay 0.1 (`--wd 0.1`, existing flag, no code)

**Mechanism.** §2.1. At 2 epochs the baseline's τ (50k steps) exceeds the run (19,670 steps), so its weights are effectively undecayed. wd 0.1 gives τ = 10k steps, about 1 epoch. The arm tests whether decay matters at all under Muon at this scale.

**10-epoch translation, pre-declared:** constant lr·wd = 1e-4, i.e. **`--wd 0.05` at lr 2e-3** (τ = 10k steps ≈ 1 epoch, the Wang & Aitchison "τ constant in epochs" form). Caveat: the Power-Lines form (τ/T ∝ TPP^−0.5) maps the screen's τ/T = 0.5 at 2 epochs to τ/T ≈ 0.22 at 10 epochs, i.e. wd ≈ 0.023 at lr 2e-3. That is close to F7's current 0.02. The two scaling rules disagree, which is why the 10-epoch pair is mandatory for this arm.

**Note:** the decay group also includes the cat/et/es/rec/dom tables (train.py name filter, unchanged).

### 3.3 cset: candidate-side retrieval plus a set-attention prior (`--cretr_m 8 --set_sim_bias`)

**Mechanism.**
- The set blocks mix candidates with no similarity prior. They are neutral in the old recipe (nn_report:68), whereas the same prior on the history side was +0.0024.
- `CandRetrieval` (model.py:224-272) is a TabR copy over each candidate's 8 most similar same-feed candidates (self, padding and embedding-less candidates excluded):
  - logit −‖k_i − k_j‖² + a·S_ij, plus a sink;
  - values LN(P·c_j + T(k_i − k_j)), where c_j is the neighbour's current token (content, dense and collaborative inputs);
  - zero-init output Z. It is built **last** under its own RNG fork (model.py:451-454).
- `--set_sim_bias` builds the set blocks with `sim_bias=1`, a deterministic linspace parameter with no RNG draw. It feeds them the candidate cosine matrix with the diagonal zeroed (model.py:582-595).
- A cold candidate can borrow its content neighbours' popularity and co-visitation inputs through P·c_j. This is relevant to the private staleness shift (cold share 0.29 → 0.41).

**Evidence:**
- ctx_nd neighbour features: holdout +0.0014 (r6_external R4);
- twins: 19.8% of positives have a same-feed negative at cos ≥ 0.95, and perfect disambiguation is worth +0.0163, with v11 pair AUC 0.629 (sol_errors:152).
- Skeptic: neighbour-only signals add +0.00005 in a stacker (sol_errors:150).

**Expected:** 0..+0.0008 single. It has the highest chance of a FAMILY-CANDIDATE, because it changes how candidates interact.

**Leak-free:** same-feed inputs only, no labels. T7 shows exactly zero change in other feeds.

### 3.4 typed: target-action key view in the last cross block (`--typed_cross`)

**Mechanism.** The target action is "open the item from the feed surface": 84.7% of positives reappear as event_type 1, and 99.7% of those have source 1 (eda_history_leak.md:10). Measured in val history:
- et1 & es1 events are 28.3% of all events;
- median 51 per feed, against Avito production's cap of 50 target actions (deep_ru §2.1);
- 4.1% of feeds have none; the user token remains as the sink.

The last cross block's key mask becomes `kmask & (het==1 & hes==1)` (model.py:587-593). The history encoder and the first cross block still see everything. There are no new parameters.

**Expected:** 0..+0.0005 single, plus a different routing (diversity).

**Test:** T11 checks the masks. The typed view keeps 34.5% of valid events on the train batch.

### 3.5 adrop0: attention-probability dropout 0 (`--attn_drop 0`)

**Mechanism:** §2.1. Plumbed as `Attn.attn_drop` (model.py:278, :328) through every Block. `attn_drop=0.1` reproduces the baseline bit-for-bit in train mode (test).

**Side effect:** CPU fwd+bwd costs 0.54-0.62× of the baseline (SDPA without dropout takes a faster path). A GPU speed-up is plausible but not verified.

**Risk:** less regularisation could worsen the 10-epoch late-cooldown memorisation. The 10-epoch pair decides.

### 3.6 a4w: A4 width 32 plus product term (`--retr_m 32 --retr_prod`)

- `t3 = Linear(dk, d)` is zero-init and created last inside the retr fork (model.py:176-178, :203-204). `--retr_prod` alone is an exact no-op at warm start (test).
- retr_m 32 changes the retrieval set: warm-start mean |Δlogit| 0.025.
- **Expected:** 0..+0.0008, with the lowest prior.
- **Cost:** retrieval tensors double. The CPU cost is the highest of the six, and the GPU cost of A4 was unmeasurable.

---------------------------------------------------------------------------------------------------------------------

## 4. Morning screen (pre-declared)

### 4.1 What runs
```
bash nn_x/r6/queue_morning.sh screen "<ids>"     # GPU list from 4-7; 2 lanes per GPU
DRY=1 bash nn_x/r6/queue_morning.sh screen "4 6 7" # print the plan only
```

**Common flags** (= `nn_x/solvers/run_sx_lane.sh` = sx_base):
`--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.1 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 1000 --seed S` + arm flags (appended last).

**Job order (claim order):**
1. `base:23` (A/A);
2. a4u 23/24;
3. wd01 23/24;
4. cset 23/24;
5. typed 23/24;
6. adrop0 23/24;
7. a4w 23/24.

**Each job:**
- `train.py` (r6), writing to `nn_x/r6/ckpt/r6_<arm>_s<seed>.pt`, `nn_x/r6/preds/nn_<v>_trainonly_{val,test}.npy` and `nn_x/r6/logs/<v>.{log,json}`;
- then `holdout_eval.py`, writing to `nn_x/r6/preds/holdout/nn_<v>_holdout.npy`.
- A job with a finished checkpoint and holdout preds is never re-run.

### 4.2 Reusing sx_base as the control
- sx_base_s23/s24 were trained by nn_x/solvers/train.py. Its only differences from nn/train.py are flag-gated arms that were off for sx_base (agedrop, PL aux). nn_x/solvers/model.py is byte-identical to nn/model.py.
- T12 parses the r6 command line with r6 train.py's own parser and compares it with sx_base_s23's stored args: **identical**. The extra r6 keys are all at their OFF defaults.
- T2 shows that the flag-off r6 model equals nn/model.py bit-for-bit, with the same weights, on real batches.
- Same seed means the same init and the same data order, because new modules are created under forked RNGs (T1: global RNG state unchanged).
- The remaining difference is GPU non-determinism: same-seed reruns diverge by 0.0002-0.0005 in the first 10k steps (r6_recap §4.5).
- **A/A job** `r6_base_s23` measures this directly. Rule: |Δval| ≤ 0.0010 and |Δholdout| ≤ 0.0005 against sx_base_s23; otherwise STOP and audit the fork before reading any arm.

### 4.3 Acceptance (frozen; `nn_x/r6/r6_eval.py` header)

The screen uses train-only predictions at the holdout-selected checkpoint, which is the last step in 29/29 earlier 2-epoch runs. Paired per feed, pooled over seeds 23/24 (per-feed mean of the two paired deltas, SE = sd/√n_feeds):

- **PASS** (recipe change / stronger member): **val delta > 2 SE AND holdout delta ≥ −1 SE.**
- **FAMILY-CANDIDATE**, only if not PASS. All of:
  - val delta ≥ −0.0010 and holdout delta ≥ −0.0005 (non-inferior, about −2 SE each);
  - ρ_x ≤ ρ_bb − 0.005. Here ρ_x = mean(corr(arm_s23, base_s24), corr(arm_s24, base_s23)) and ρ_bb = corr(base_s23, base_s24), both as within-feed z Pearson correlations over all val candidates;
  - the mixed ensemble {base_sA, arm_sB} ≥ the 2× base ensemble on the holdout (pooled delta ≥ 0).
- Otherwise **FAIL**.
- **Slices reported:** val M<50 / M≥50 / Tue / Thu / all-cold-positive / all-warm-positive (cold = it_hist_all_luo == 0, the CPU2 definition), and holdout late 20%. Any slice below −2 SE is flagged **WARN** and must be re-checked in the 10-epoch pair, because the holdout cannot see M<50 (REVIEW_SYNTHESIS P3).

### 4.4 Why these thresholds (calibration on old arms, information only)

`r6_eval.py calib` (`nn_x/r6/logs/r6_eval_calib.txt`) was run after the rules were written, on the four existing 2-epoch sx arms:

| old arm | val Δ (z) | hold Δ | ρ_bb | ρ_x | ρ_x − ρ_bb | mixed-ens hold Δ |
|---|---|---|---|---|---|---|
| pl20 (aux loss) | +0.00059 (1.31) | +0.00021 | 0.9784 | 0.9777 | −0.0007 | +0.00009 |
| tau30 (aux loss) | −0.00091 (−1.85) | −0.00109 | 0.9784 | 0.9773 | −0.0011 | −0.00049 |
| pl20_w1 (aux loss) | +0.00051 (0.99) | +0.00026 | 0.9784 | 0.9764 | −0.0020 | +0.00004 |
| agedrop (input change, strong cold/warm trade) | +0.00005 (0.08) | −0.00189 | 0.9784 | **0.9715** | **−0.0069** | −0.00050 |

**Reading:**
- Loss tweaks lower ρ_x by only 0.0007-0.0020.
- A genuinely different function (agedrop: coldpos +0.020, warmpos −0.008) lowers it by 0.0069.
- So the −0.005 bar separates "different function" from "same function, re-seeded". agedrop would still FAIL on non-inferiority, as it should.
- Pooled val SE is 0.00045-0.00061, so 2 SE ≈ 0.0009-0.0012.

### 4.5 After the screen
1. **10-epoch pair** for each PASS or FAMILY-CANDIDATE arm: `bash nn_x/r6/queue_morning.sh e10 <arm> "<GPUs>"`.
   - F7 recipe (identical to nn/run_final7b.sh: lr 2e-3, mix 0.1, eval 2500) + arm flags. wd01 becomes `--wd 0.05` (§3.2).
   - Adds `--save_steps 50000,60000,70000,77500`, so the members can later be grown by cooldown branching (R1/P1).
   - Seeds 101/102, paired with v17_pre2s_s101/s102 (same init and data order). ≈ 2 × 2 h.
   - The lane runs `r6_eval.py e10 <arm>` automatically.
   - **Rules:**
     - a recipe winner proceeds to a 6-seed family iff pooled val Δ ≥ 0 AND holdout Δ ≥ −1 SE;
     - a family candidate proceeds iff |val Δ| ≤ 0.002 and |holdout Δ| ≤ 0.002 (gate (a) proxy).
2. **Family:** seeds 101-106 through gates (a)+(b) as usual.
   - Fine-tune with `EXTRA=n2c_pre_v3 SUF=n2cpv3 FTARGS='--decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton' bash nn_x/r6/ft_list.sh <v>`. The r6 copy has its paths redirected, and finetune.py is byte-identical to nn/.
   - A recipe change becomes the default only if its 6-seed TO ensemble beats F7's 6-seed ensemble on the holdout with val ≥ 0. Candidate sets are pre-declared and chosen on the holdout (CURRENT_STATUS §3).
   - Arm checkpoints carry new cfg keys. They load only with nn_x/r6 code (predict.py, holdout_eval.py, finetune.py). nn/ code cannot load them, by design.
3. **Stale-val replay** is required for any cold/warm trade (CURRENT_STATUS §3). The CPU2 harness (solvers/CPU2_stale_replay_pre2s_and_arms.py) imports its own model.py, so it needs a small adapter that points it at nn_x/r6/model.py before it can replay r6 arms. Not built here.

---------------------------------------------------------------------------------------------------------------------

## 5. Implementation and audit

### 5.1 nn_x/r6 contents
Copied from nn/: every *.py and *.sh. data.py and model.py were byte-identical copies made earlier by the recap agent.

| file | change vs nn/ (diff lines) |
|---|---|
| model.py | 144: the six options (§3), cfg stores r6 keys only when non-default (model.py:367-375), so a flag-off checkpoint has exactly the nn/ cfg |
| train.py | 23: flags (train.py:141-150), `r6_` variant prefix (:156), `--init /abs/path` read-only (:187), Ranker kwargs (:225-228). The loop, loss, schedule and optimizer are untouched. |
| data.py | 22: FEAT_CACHE = nn_x/r6/feat_cache (symlink → ../../nn/feat_cache), **read-only** feat_path (never links or copies into the shared cache), CPU-safe synchronize, `hidx` in the batch (:158) |
| common.py | 17: CKPT / LOGS / PREDS = nn_x/r6/{ckpt,logs,preds}; GPU guard 4-7 only |
| pcmp.py, zmean.py, ensemble.py | preds paths → nn_x/r6/preds |
| build_w3b_feats.py, make_cand.py | disabled (`raise SystemExit`): they write into data/feats_new, nn/feat_cache and subs/ |
| *.sh | `../.venv` → absolute venv, `../preds/` → `preds/` (nn_x/r6/preds); every legacy launcher (run_final*, stage2*, mix_ft*, post_final6, run_ftv3, run_f6cd) refuses to run unless `R6_ALLOW_LEGACY=1`; ft_list.sh is active |
| blend.py, finetune.py, holdout_eval.py, predict.py | byte-identical; they write only through common's r6 paths |

**New files:**
- `queue_morning.sh`;
- `r6_eval.py`;
- `tests/test_r6.py` and `tests/mini_split.py`. mini_split is a memory-mapped 288-feed miniature of SplitData that reuses the real gather, batches and scatter.

**Path audit:** every write in the r6 Python goes through `CKPT`, `PREDS` or `LOGS` (grep of np.save / torch.save / open / write_text). holdout_eval writes `PREDS/holdout`.

### 5.2 CPU tests (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/tests/test_r6.py`, 95 s, 79/79 PASS)

| test | result |
|---|---|
| T12 CLI | the sx_base command line leaves all r6 flags OFF; every sx_base_s23 arg is reproduced |
| T1 init | flag-off cfg and state_dict equal nn/'s. For every arm: 0 baseline tensors changed, global RNG state after construction unchanged. New tensors: a4u 1 (1,152 params), a4w 1 (8,192), cset 13 (66,058), typed / adrop0 / wd01 0 |
| T2 flag-off equality | real train batches S (B 96, Mx 64) and L (B 40, Mx 304), trained sx_base_s23 weights: eval logits max\|d\| = 0; train-mode loss 2.160182 = 2.160182 and 2.867377 = 2.867377 (dropout on, same seed); all gradients identical |
| T3 warm start | adrop0 eval and `--retr_prod` alone and `--cretr_m` alone are exact no-ops. Changes (info, mean \|Δlogit\|): a4u 0.03, a4w 0.025, cset 0.06, typed 0.13-0.15 |
| T4 masks | valid logits unchanged (max\|d\| = 0) when padded candidates and padded history events are overwritten with noise, for every arm |
| T5 gradients | every new tensor gets a non-zero, finite gradient. The zero-init output layers get a gradient at the true init (cretr.Z, retr.t3, retr.e_cnt) |
| T6 equivariance | permuting candidates within feeds: max\|d\| 1.9e-6 for every arm, on the real and the jittered batch |
| T7 leakage | perturbing all inputs of feed 0 changes feed 0 (max\|d\| 4.3-4.6) and **no** other feed (max\|d\| = 0), for every arm |
| T10 dedup | hand example correct. a4u's retrieved items are all distinct. Baseline top-16 repeat share 30.1-31.8% |
| T11 typed | cross[0] gets the full mask, cross[-1] the target-action mask |
| T13 optimizer | train.py's grouping and one step per arm: finite; new Linear matrices join Muon (cset +5, a4w +1); every new tensor updates |
| adrop0 semantics | attn_drop 0.1 is bit-identical to the baseline in train mode; attn_drop 0 changes train outputs only; 11 Dropout modules stay at 0.1; 5 Attn have attention dropout 0 |

**Not testable on CPU:**
- the bf16-autocast GPU path. The new modules use the same dtype pattern as HistRetrieval: fp32 scores and softmax, bf16 Linears, fp32 LayerNorm;
- GPU memory. The largest addition is a4w's doubled retrieval tensors, ≈ +2-3 GB at 32,768 candidates. Each process holds 21.2 GB of data (sx_base stdout), so 2 processes per GPU are safe;
- GPU speed.

The A/A job and the first 1000-step log line of each arm (minutes per 1000 steps vs sx_base's 2.5) are the GPU-side checks.

---------------------------------------------------------------------------------------------------------------------

## 6. Deliberately not in this queue (recommended separately)

1. **Cooldown branches (r6_external R1 / r6_recap P1).** No code. They are the cost multiplier for any family this screen produces, and the lead's lane decision. The r6 e10 pairs save plateau snapshots so they can be branched. r6 train.py also accepts `--init <repo>/nn/ckpt/v17_pre2s_s10x@60000.pt` (read-only), so branches can run from the r6 copy with outputs isolated, and with `--wd` as a hyper-branch (R7).
2. **Cooldown-phase regularisation and pool augmentation (P2 / R2 / R7).**
   - These are the only proposals aimed at the measured 10-epoch memorisation (end − selected: −0.0033 val in 19/21 mix-0.1 runs).
   - They need a continuation harness from shared plateau snapshots, plus flags this design did not implement: dropout override after `--init`, `--aux_pool_norm`, `--aug_cand` / `--aug_hist`.
   - They cannot be judged by a 2-epoch screen.
3. **Batch ×2 (P3):** §2.1. Cost lever only, and the diversity risk is invisible at 2 epochs.

---------------------------------------------------------------------------------------------------------------------

## 7. Caveats

- **Power.** 2 seeds per arm give MDE ≈ 0.0014 val. Real effects of +0.0005 will usually read as FAIL. The FAMILY route catches arms that do not raise strength but change the function.
- **Horizon.** A 2-epoch result predicts only the 2-epoch mean function. 2- and 10-epoch ensembles differ by +0.0040 (r6_recap §3.2), and regularisation knobs (wd01, adrop0) can reverse at 10 epochs. The e10 pair is mandatory, with no shortcut to a family.
- **Mapping ρ_x to families.** The FAMILY rule assumes the 2-epoch arm-vs-base correlation (single members ≈ mean function, ρ 0.978) tracks the 10-epoch family-ensemble correlation (0.978-0.986 between existing families, r6_recap §3.6). That is plausible but unverified. The e10 pair plus gates (a)+(b) decide.
- **a4u precision.** The "fp32" scores in a4u are TF32 on the GPU (train.py sets allow_tf32). Exact duplicates are removed regardless.
- **Subjectivity.** The P(PASS) / P(FAMILY) values in §3 are judgement, not measurement.
