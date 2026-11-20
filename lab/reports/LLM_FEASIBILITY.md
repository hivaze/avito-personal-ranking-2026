# Small LLMs / embedding models for the Avito ranker: feasibility verdict

Status: 2026-11-12, synthesis of the LLM workflow (research, build, independent review).
- **No GPU was used** by any step: every command ran with `CUDA_VISIBLE_DEVICES=""`. Nothing was trained beyond 1-6 optimiser steps on 8-32 feeds on CPU.
- **Detailed sources:** `reports/llm_research.md` (evidence, compute math, design) and `reports/llm_build.md` (env, code, smoke tests). The review scripts are in `/tmp/scratchpad/llm/review/`.
- **Evidence tags:** [P] project measurement, [V] primary source read, [S] snippet or secondary source, [inf] arithmetic or inference, [spec] subjective prior.

## TL;DR
1. **Verdict: LONG SHOT, but cheap to settle.**
   - Nothing in the data is text, so an LLM can bring only a pretrained *computation prior* over soft tokens that we project in. It adds capacity and a different inductive bias, but **no new information**. Our own probes say the limit is information, not capacity.
   - Every route that needs text is dead: Qwen3-Embedding, prompts, numbers serialised as text.
2. **What exists:** a working, CPU-verified **Qwen3-0.6B soft-token ranker**.
   - LoRA r=32 adapters, projectors aligned to the token-embedding RMS, an fp32 score head plus a K9 aux head.
   - The ranker's own inputs (`cvnd_pre2s`, 104 dense features) and its own loss.
   - An exact Qwen3.5 "branch" trunk for 2B/4B.
   - A pre-declared matched-data test (`nn_x/llm/run_matched.sh`, **never executed**).
   - The isolated env `.venv_llm`; `.venv` is untouched (re-verified at synthesis time).
3. **Recommendation: conditional GO for Stage A only**, at the **lowest priority** of the morning queue:
   - Start only after F10-F12 finish and after the r6 screens have been launched.
   - Use one lane, about **1.8-2 GPU-h exclusive** (3.5-6 h wall-clock on a shared lane), with a hard cap of **6 GPU-h**.
   - Before launch, implement and CPU-retest the 4 pre-launch amendments (§5.2).
   - **NO-GO** for any escalation (4B, Qwen3.5, full-data members) unless the pre-declared A1/A2 rules fire.
   - **NO-GO** for Qwen3.5-9B, Qwen3-Embedding-4B and text serialisation.
4. **Expected value is small** [spec].
   - P(an LLM family ends up in the final pool) ≈ 5-10%, worth about +0.0003..+0.0008 val if it does.
   - So the expected gain is ≈ +0.00002..+0.00008 val, far below the public SE of 0.0012.
   - The test's real value is a matched, reusable answer to "can small LLMs help here?". It costs ≈ 0.1-0.2% of the ≈ 2,600 GPU-h left on some GPUs before 2026-12-09.

---

## 1. What a small LLM can contribute when the data has no text

| route | status | why |
|---|---|---|
| (a) pretrained trunk over projected soft tokens (user, history, candidates), with LoRA and custom heads | **only live option, built** | Brings a computation prior and a different inductive bias. The projectors are learned from labels, like the ranker's first Linear |
| (b) Qwen3-Embedding-4B / text prompts / numbers as text | **dead** | No text exists. The int8[128] embedding would cost ≈ 287 tokens per item, ≈ 76K tokens per feed vs ≈ 265 soft tokens [inf]. **Qwen3-Embedding-4B weights are not local** (config only, 16 MB; re-checked) [P]. As a trunk it would be just another Qwen3-4B, and Qwen3-4B-Instruct-2507 is local |
| (c) LLM teacher → distil into the ranker | redundant | It needs (a) to work first. Blending is cheaper: 0.6B test inference is ≈ 25 s. KD lowers ensemble diversity (nnx_VERDICT C5) [P] |
| (d) LLM trunk for our pretrained history encoder | not worth it | Encoder scaling is already flat (v2→v4); its value sits in the ID table [P] |
| "LLM tech" pieces trained from scratch (gated attention, Muon, WSD) | **already paid** | gate+retr16 +0.0028 and Muon+WSD +0.0031 (nnx_VERDICT) [P] |

**Backbone choice** [P/V]:
- **Qwen3-0.6B** (pure softmax attention; custom 4D masks and position ids are honoured) goes first, with Qwen3-4B-Instruct-2507 as the only escalation.
- **Qwen3.5 (2B/4B)**: 3/4 of its layers are Gated DeltaNet, which **silently ignores custom attention masks**. A naive candidate append is order-dependent: scores move by up to 0.44 (2B) and 0.40 (4B) when candidates are shuffled. It works only through our exact branch trunk, and `.venv_llm` lacks the `flash-linear-attention` kernels, so GPU speed is unknown.
- **Qwen3.5-9B and Qwen3-8B**: weights not local (≈ 18 GB download); about 21 h per full-data epoch.
- The local MoE checkpoints (Qwen3-30B-A3B, Qwen3.6-35B-A3B; 57-67 GB) were not considered: memory, and no extra information.

## 2. Evidence

**External evidence on non-language inputs: no reliable gain against a matched model trained from scratch, except at small data.**
- **Tan et al., NeurIPS'24** [V]: LLM-free ablations beat Time-LLM in 26/26 cases, CALF in 22/26 and OneFitsAll in 19/26. Random init + fine-tune won 8 configurations vs 3 for pretrained + fine-tune. No LLM benefit at 10% data.
- **FPT** [V]: only parity with a tuned small transformer (C10-LRA 38.6 vs 42). It beats only a *frozen random* network.
- **Chronos** [V via review]: LM init "overlaps with or slightly underperforms" random init.
- **Positive cases come with caveats:**
  - HLLM's pretrained user-LLM gives +5-7% R@10 on embedding inputs, but only at 4M interactions (Pixel200K) and with item text upstream [V].
  - "Random Initialization Can't Catch Up" (2506.21570) reports a lasting gap in low-data forecasting [S].
  - E4SRec's +21% nDCG is against a weak, unmatched SASRec [V; that it is the BCE version is an inference].
- **From-scratch systems at scale:** large ID-only industrial recommenders (HSTU, OneRec-V1) are trained from scratch. The reported top solutions of anonymised-log competitions (RecSys Challenge 2025, OTTO) used no pretrained LLM [S/K]. LLM4Rec models also ignore item order ("Lost in Sequence", KDD'25) [S].
- **Implication** [inf]: the 150K-feed matched test sits nearer the low-data regime where a prior helps. So it leans *towards* the LLM: a STOP there is strong evidence, and a PASS still needs the 600K slope check.

**Project evidence: information, not capacity, is the limit** [P].
- d=256 (3M params), 3 blocks, 8 heads, MoE and TabM were null; GRU/DIN are "similar alone" (nn_report, nnx_VERDICT).
- Error analysis found no candidate signal carrying information beyond the model (sol_errors).
- Ensemble value comes from same-strength, different-recipe families. The strength-diversity law gives a break-even gap of 0.049(1−ρ) on val (sol_ensemble 2.5).
- So a full-data LLM member must reach train-only val ≈ 0.393 at ρ 0.96, or ≈ 0.390 at ρ 0.90. The best family's single members average 0.3946.

## 3. Compute math

Basis [inf] (times below use the measured padding efficiency 0.944):
- 6N·T per training feed (LoRA + gradient checkpointing; frozen weights get no weight gradient), with N = body parameters and T = 264.5 tokens per train feed.
- Overhead: +5% in the plan, but the review measured **+12-17%** (dense-mask SDPA attention 7-9% of 2N, LoRA ≈ 4.6%).
- GPU dense BF16 peak the vendor's peak rate. **MFU is assumed, not measured:** 15% for 0.6B, 25% for 2B+. Pessimistic = half the MFU.

| model (local?) | body params | TFLOP / train feed | × ranker (0.42 GFLOP) | 150K feeds × 2 ep | 1 full epoch (3.15M) | val + test inference |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (yes) | 0.44B | 0.73 | 1.7K | ≈ 11-13 min | 2.0-2.2 h (4.5 pess.) | ≈ 40 s |
| Qwen3.5-2B (yes) | 1.37B | 2.29 | 5.4K | ≈ 21 min | 4.2 h | ≈ 1.2 min |
| Qwen3-4B-Instr. / Qwen3.5-4B (yes) | 3.6B | ≈ 6.0 | 14K | ≈ 56 min | 11 h | ≈ 3 min |
| Qwen3.5-9B (**no**) | 6.9B | 11.5 | 27K | ≈ 2 h | 21 h | ≈ 6 min |

**Token counts are exact** [P, `nn_x/llm/estimate.py`]:
- Seed-23 subset: 39.2M real / 41.5M padded tokens per epoch (padding efficiency 0.944), 753 steps of 56K tokens.
- Holdout 9.9M, val 3.9M, test 7.9M, full train 833.5M real tokens per epoch.

**CPU sanity check** [P]: forward ran at 0.79 TFLOP/s on 12 threads, so the per-token FLOP model is in range. This says nothing about GPU MFU; the `probe` arm measures that.

**Cost of a full-data LLM family**, needed for gate (b) (6 seeds × 1 epoch) [inf]:
- Training: ≈ 12-13 GPU-h exclusive at 0.6B (≈ 26 pessimistic).
- Val fine-tune branch: 5 folds × 4 epochs ≈ 8-10 min per member.

## 4. What was built and smoke-tested (CPU only)

**Env.**
- `.venv_llm` (Python 3.12.3): transformers 5.22.2, peft 0.21.2, accelerate 1.15.0. torch 2.11.0+cu130 comes from `.venv` via a trailing `.pth`, and a pycache redirect keeps bytecode out of `.venv`.
- `.venv`'s `uv pip freeze` was identical (94 lines) at build, at review and **again at synthesis time**.
- One disclosed incident: 114 `.pyc` cache files were written into `.venv` before the redirect existed. They were removed, the redirect was added, and a re-test wrote 0 files.

**Code in `nn_x/llm/`:**

| file | role |
|---|---|
| `data_llm.py` | read-only data layer, bit-identical to `nn/data.SplitData` |
| `model_llm.py` | Qwen3 soft-token ranker, layouts L1/L2, LoRA, fp32 heads |
| `qwen35_branch.py` | exact Qwen3.5 branch trunk |
| `train_llm.py` | trainer |
| `run_nn_baseline.py` | runs the unmodified `nn/train.py` with redirected outputs |
| `compare.py` | A1/A2 evaluation; reproduces `nn/pcmp.py` exactly |
| `estimate.py` | exact token counts, FLOP and time estimates |
| `run_matched.sh` | launch script |
| `smoke_*.py` | CPU smoke tests |

**Model as built.**
- Token sequence: `[user, history old→new, candidates]`.
- **Layout L1:** user and history are causal. Each candidate attends to the user, all history and itself, and **every candidate gets position id H+1**. Each candidate is scored as "the next token", the model is permutation-equivariant, and the whole feed is one pass.
- Trainable parameters: 21.99M (20.19M LoRA on 196 modules, plus projectors, norms and heads) over a frozen 440.5M body.
- Loss: the ranker's listwise softmax per feed (`nn/model.listwise_loss`, imported read-only) + 0.5(1−p_random) K9 aux at τ 10.

**Smoke and review results** (32 real feeds unless stated) [P]:

| check | result |
|---|---|
| permutation equivariance (L1 / L2) | max Δscore 8.4e-6 |
| candidate alone `[u,H,c]` vs batched; feed alone vs padded batch | 1.1e-6; 4.5e-6 |
| independent check: batched L1 == HF native causal forward (H = 0 and 256, non-zero LoRA B) | 8.0e-6 |
| leakage: labels and positions never read; cross-feed; cross-candidate in L1 | Δ = 0.0 exactly |
| one optimiser step (pretrained / random-init control) | eval loss 3.950 → 3.891; 3.962 → 3.859 |
| Qwen3.5 branch == HF forward (2B / 4B); branch equivariance | relative error 8.9e-6 / 4.0e-6; 4.9e-5 / 1.3e-6 |
| end to end (data → train → holdout-best reload → preds), 0.6B and Qwen3.5-2B | PASS |
| bf16 vs fp32 scores (review) | corr 0.997, max \|Δ\| 0.098 at score std 0.72; 0/281 within-feed ties |

**Design flaw found by the review** (measured; cheap fix, §5.2 S1):
- In L1 the user token at position 0 becomes Qwen3's massive-activation **attention sink** from about layer 6 on: max \|h\| ≈ 5,600 vs ≈ 20 for other tokens.
- Its state becomes content-independent: cosine 0.99999 between two users at layer 13.
- User information still reaches the candidates through `Linear([e, e⊙u])` and the dense features, as in the ranker. So this is a handicap, not a leak.

## 5. Pre-declared matched-data GPU test (Stage A)

### 5.1 Arms
All arms are train-only, with checkpoints selected on the fixed 30K random-like holdout; val never selects anything.
- **Data:** subset per seed = `np.sort(default_rng(seed).choice(3151208, 150000))` minus the holdout, i.e. exactly `nn/train.py --train_feeds`. That gives 148,600 feeds (seed 23) and 148,557 (seed 24).
- **B23, B24:** the unmodified `nn/train.py` with the F7 stage-1 flags plus `--train_feeds 150000 --epochs 20 --eval_every 250`.
- **L23, L24:** Qwen3-0.6B, L1, LoRA r=32, AdamW (LoRA 2e-4, projectors/heads 1e-3), 56K-token batches, 2 epochs (1,506 steps), holdout every 200 steps.
- **R23:** the same architecture with random init, fully trained. A weak attribution control only: it is untuned.
- **Disclosed asymmetries:**
  - The recipes differ on purpose (Muon + WSD over 20 epochs vs LoRA/AdamW over 2).
  - B gets 32 holdout evaluations vs 8 for L. This favours B: conservative for A1, but harder for L on A2(i).
  - The K9 aux weight per feed is ≈ 0.53 in L vs 0.80 in B, and B anneals it in cooldown.

### 5.2 Pre-launch amendments
These are declared here, before any GPU look. They must be implemented in `nn_x/llm/` and re-tested on CPU (T4-T7, plus the review's native-causal check R1); about 1-2 agent-hours.
- **S1 (sink token):**
  - Prepend one frozen soft token at position 0: the checkpoint's own `<|endoftext|>` (id 151643) embedding row, captured before `embed_tokens` is dropped. The random-init control uses the same row of its own init.
  - The user moves to position 1, history to 2..H+1, and every candidate to H+2. All tokens may attend to the sink.
  - CPU acceptance: equivariance and leakage tests as before, plus user-token cosine across two users at layer 13 < 0.99.
- **E1 (under-training guard):**
  - After L23 and L24 finish, read **only their holdout lines**. If for either seed the holdout-best evaluation is the final one and it beats the previous evaluation, re-run **both** seeds once with `--epochs 4` (≈ +1 h exclusive, 2 h pessimistic) and judge A1/A2 on those runs.
  - Apply the same guard to R23 only if the P4B exception (§5.4) would otherwise fire.
  - At most one extension. `compare` runs only after this decision.
- **I3 (inference precision):** the probe scores the holdout with bf16 and with fp32 trunk weights (new flag). If \|ΔNDCG\| > 1e-4, all final holdout/val/test predictions are written in fp32 (< 1 min for 0.6B).
- **V0 (hygiene):** L and R arms run with `--no_val_log`, so val is first read by `compare.py`. This also cuts the per-arm evaluation time from 5.3 to 3.8 min.

### 5.3 Commands, gates and cost
```
cd <repo>/nn_x/llm && mkdir -p logs
bash run_matched.sh <GPU> probe        # 40 steps + 1 holdout eval (+ I3 check), ~5 min
setsid nohup bash run_matched.sh <gpuA> L23 L24 R23 > logs/lane_gpuA.out 2>&1 &   # PIDs -> RUNNING.txt
setsid nohup bash run_matched.sh <gpuB> B23 B24     > logs/lane_gpuB.out 2>&1 &   # (lead also logs them in nn/logs/RUNNING.txt)
# after E1 is decided:
bash run_matched.sh <any 4-7> compare     # CPU only; prints A1/A2 and the decision
```
- The script refuses some GPUs and refuses to start without free memory: 30 GB for L, 35 for R, 60 for B.
- It needs a lane with fewer than 4 training processes on that GPU.

**Probe gate** (pre-declared; the plan is ≈ 122K padded tokens/s at 15% MFU):
- Run the arms only if the probe measures **≥ 25K padded tokens/s** on the chosen lane (L-arm training ≤ 55 min) **and** peak memory ≤ 30 GB.
- Otherwise run no arms. Wait for a quieter lane, or profile once.

**Cost** [inf; exact tokens, assumed MFU]:

| arm | exclusive GPU | pessimistic |
|---|---|---|
| probe | 0.07 h | |
| L × 2 | 0.55 h | |
| R | 0.35 h | |
| B × 2 | 0.8 h | 40-45 GB, since it loads the full train split |
| **Stage A** | **≈ 1.8-2.0 GPU-h** | ≈ 3.5-4 h |

- Realistic wall-clock on a shared lane (75-95% utilised): **3.5-6 h** (review).
- With E1: +1-2 h. **Hard cap: 6 GPU-h.**

### 5.4 Acceptance and decision rules (`compare.py`)
- **A1, beats the ranker:**
  - pooled paired train-only val L−B > 2 SE, and both seeds positive;
  - holdout ≥ 0;
  - no slice below −2 SE (val M<50, M≥50, Tue, Thu, all-cold-positive; holdout late 20%).
- **A2, useful ensemble member:**
  - (i) L ≥ B − 0.002 on val **and** on the holdout;
  - (ii) the mean over seeds of eq(L_s, B_s) − eq(B23, B24) > 0 on val with z ≥ 1.5, and holdout ≥ 0;
  - also reported: ρ and the law-predicted gain 0.0186(1−ρ) − 0.378\|gap\|.
- **Both fail → STOP.**
  - One exception: L ≥ B − 1 SE **and** L − R > 2 SE (after E1 has been applied to R if R was still improving) → **one** Qwen3-4B-Instruct-2507 probe (P4B, ≈ 1.5-3 GPU-h). Stop if it fails A1/A2.
- **A1 passes:** a 600K-feed slope check (≈ 0.7-1.4 GPU-h). Continue only if the advantage keeps ≥ 50%.
- **A2 passes, or the slope check passes:** full-data 0.6B members (1 epoch each; 2 seeds ≈ 4-9 GPU-h, then 4 more), then gates (a)+(b) and the frozen pool protocol, including `presub_gate`.
- **Other rules:**
  - L2 (candidate↔candidate attention) runs only if L1 passes.
  - Qwen3.5 only through `qwen35_branch.py`, and only after installing `flash-linear-attention` into `.venv_llm`.
  - At most one divergence fallback (`--lora_r 64 --lr_lora 1e-4`). No search on val.

## 6. Go / no-go relative to the queued work

| item | GPU cost [inf] | P(useful) [spec] | payoff if useful |
|---|---|---|---|
| F9-F12 families (running or queued) | ≈ 1-1.3 GPU-h per member, 6-7 members per family | family passes gates (a)+(b): moderate (the one lever that keeps paying) | ≈ +0.0003-0.0009 val per accepted family |
| r6 P1 earlier cooldown (continuations from existing snapshots; r6_recap §7, R6_PLAN.md pending) | ≈ 2 GPU-h | moderate | ≈ 30% cheaper members for every future family |
| r6 structure/hyperparameter screens (batch, dropout, wd, depth/heads, retr_m, cooldown) | 2-epoch screens, a few GPU-h each | low-moderate each | direct NDCG or throughput |
| **LLM Stage A** | **≈ 2 GPU-h (cap 6)**, 2 of 16 slots for 4-6 h | A1 10-20%, A2 25-35% | Opens a 13-26 GPU-h family build. End to end: P(in final pool) 5-10%, worth +0.0003-0.0008 |

**Decision.**
- **GO for Stage A**, scheduled last: after F10-F12 complete and after the r6 P1-P4 screens are launched. One lane, probe gate first, hard cap 6 GPU-h. Never stop or displace an F-family seed or a screen for it.
- Opportunity cost: about 1-2 ranker members' worth of GPU time, ≈ 0.1-0.2% of the remaining budget.
- Why spend it at all:
  - The user asked for this route to be tried.
  - The answer is matched, pre-declared and reusable.
  - A pretrained trunk is one of the few untested sources of diversity at equal information.
  - The 2026-12-09 deadline leaves room for Stage C if it surprises.
- **NO-GO:** anything beyond Stage A unless the A1/A2 rules fire. Qwen3.5-9B, Qwen3-8B and Qwen3-Embedding (weights not local, and nothing to embed). Text serialisation. Distillation. Qwen3.5 on GPU before the `flash-linear-attention` kernels are installed and checked.

## 7. Evidence vs speculation; open risks
- **Measured** (CPU, this workflow): every smoke and review number in §4, token counts, parameter counts, CPU TFLOP/s, the sink finding, and `.venv` freeze identity.
- **Assumed:** GPU MFU, the B-arm run time (from earlier project runs), and memory figures (analytic).
- **Speculative:** P(A1), P(A2), P(pool) and the payoffs in §6.
- **Untested on CUDA:** bf16 autocast with bf16 frozen weights and fp32 LoRA/norms, SDPA with a boolean 4D mask, fused AdamW, and non-reentrant checkpointing. The probe tests these in about 5 minutes.
- **R control:** the random-init control is untuned, so L−R is weak attribution evidence. This is mitigated by E1 applied to R before the P4B exception.
- **One more pipeline cost:** an LLM family would also need the val fine-tune branch (≈ 10 min per member) to sit in both halves of the 0.5 FT + 0.5 train-only ensemble.
