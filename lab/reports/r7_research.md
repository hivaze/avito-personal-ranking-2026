# r7 research: modern blocks, tabular-DL encoders and never-varied hyperparameters (arm selection)

Written 2026-11-12, 04:15-05:00 UTC, by the r7 research agent. **CPU only.** No GPU job was launched, and nothing under nn/, src/, preds/, subs/, nn_x/r6/ or nn_x/llm/ was written. The only other files are an information-only diagnostic script and its log:
- `/tmp/scratchpad/r7/research/diag_cpu.py`
- `/tmp/scratchpad/r7/research/diag_cpu.log`

The script reads nn/ckpt read-only. No decision depends on a threshold in it.

**Evidence tags**
- [V]: primary source read this session (URL in §9).
- [S]: search snippet or secondary mirror.
- [K]: background knowledge.
- [R]: earlier project report.
- [D]: my own CPU measurement this session.

**Inputs read:** CURRENT_STATUS (to the 04:15 line), EXPERIMENT_REGISTRY, COVERAGE_MAP, r6_recap, r6_external, r6_design, nnx_VERDICT, nnx_research_tabular_dl, nnx_research_llm_arch, nnx_design §7-8, nn/model.py and nn/train.py (in full), nn/data.py (attach_xt, gather), and nn_x/r6/queue_morning.sh (current version).

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **Selected for the r7 2-epoch screen: 12 arms plus 1 reserve.** 8 are architecture / input-encoding arms (+1 reserve), and 4 are hyperparameter / training arms. Each sits behind a flag whose default reproduces nn/ exactly. They are ranked by expected gain per GPU-hour (exact specs in §5):

   | prio | key | kind | one-line mechanism | exact at init? | cost vs base | P(promote) (judgement) |
   |---|---|---|---|---|---|---|
   | 1 | **muon_in** | training | Put the 5 input projections (c_mlp.0, c_lin, h_lin, u_lin, retr.wk) on AdamW instead of Muon. Muon's authors say input layers belong on AdamW, and c_mlp.0 maps 104 heterogeneous tabular features, where Muon's orthogonalised update removes per-feature step sizes. | yes (same model) | 1.0× | 0.20 |
   | 2 | **batch2s** | hyperparameter | Batch ×2 (65,536 candidates / 1,024 feeds) under the **pre-declared square-root SDE rule**: lr ×√2, (1−β) ×2 for every momentum, warmup and eval interval constant in samples. A cost lever: B_simple 4.9-9.4K ≫ 317 feeds. | same init, different batching | ~0.6-0.7× | 0.6 (non-inferiority) |
   | 3 | **resdrop0** | hyperparameter | Residual, FFN and retrieval dropout 0.1 → 0. Attention dropout stays at 0.1, so this is disjoint from r6 adrop0. 2-epoch runs never over-fit (29/29 select one of the last two evals, 26/29 the last). | same init | ~0.9× | 0.30 at 2 ep (10-ep risk) |
   | 4 | **pbld** | input-encoding | RealMLP PBLD periodic embeddings of the 104 dense inputs, fed through a zero-init block into c_mlp's first layer. 21/104 inputs have a point mass > 50% and 22 have \|skew\| > 3 [D]. Distinct from the closed PLE: smooth, low-frequency features (σ = 0.1). | yes | ~1.05× | 0.15 |
   | 5 | **hc4** | architecture | Dynamic Hyper-Connections, n = 4 streams, wrapping the 8 candidate-stream sublayers. Static part initialised to pre-LN, dynamic part zero-init, readout = mean of streams. | yes | ~1.1× | 0.12 |
   | 6 | **dwa** | architecture | DenseFormer depth-weighted average after each of the 4 candidate blocks: 14 scalars, α initialised to the identity. | yes | 1.0× | 0.10 |
   | 7 | **diff2** | architecture | DIFF-Transformer-V2 target attention in the 2 cross blocks: query maps doubled, K/V shared, out = A1 − σ(W_λ x)·A2, each map with its own sim-bias scale, no per-head norm. | no | ~1.15× | 0.10 |
   | 8 | **swiglu** | architecture | SwiGLU FFN at matched parameters (hidden 170 vs GELU 256) in all 5 blocks, with gate and up as separate Linears so Muon orthogonalises each. | no | 1.0× | 0.10 |
   | 9 | **derf** | architecture | Normalisation-free: Derf γ·erf(αx+s)+β (CVPR 2026, successor to DyT) replaces the 15 block norms and the head norms. Input-embedding norms stay LayerNorm. | no | 1.0× | 0.08 |
   | 10 | **hl2** | hyperparameter | A second history-encoder layer, built under extra_init with zero-init output matrices. The history self-attention is the most-used sublayer (update 0.57× of its stream [D]). hist_layers has never been varied alone or under Muon. | yes | ~1.1× | 0.08 |
   | 11 | **periln** | architecture | Peri-LN (sandwich) output norms on all 10 sublayers. Gemma / OLMo-2 placement; +1.9 avg at 400M [V]. | no | 1.0× | 0.07 |
   | 12 | **ftok** | input-encoding | FT-Transformer-lite: 8 semantic feature-group tokens + CLS go through 1 transformer layer, and the CLS output is added through a zero-init projection. | yes | ~1.2× | 0.08 |
   | R | **postln** (reserve) | architecture | Post-LN with DeepNorm (α = (2N)^¼, β-scaled init). Run only if a lane is idle. | no | 1.0× | 0.05 |

2. **Arms owned by r6 (do not duplicate; listed separately so the screen agent can include them).** nn_x/r6/queue_morning.sh screens `a4u` (--retr_uniq), `wd01` (--wd 0.1), `cset` (--cretr_m 8 --set_sim_bias), `typed` (--typed_cross), `adrop0` (--attn_drop 0) and `a4w` (--retr_m 32 --retr_prod) on seeds 23/24.
   - **Workflow status at 04:26 UTC:** that queue has **not been launched**. There is no `nn_x/r6/logs/RUNNING_r6.txt` and no queue directory, `reports/R6_PLAN.md` does not exist yet, and `nn_x/r7/` does not exist yet.
   - The r6 code refuses some GPUs.
   - If r6 still has not started when r7 launches, the cleanest option is to port its flag-gated diff (79/79 tests; flag-off is bit-identical to nn/) into nn_x/r7 and score its arms against r7's fresh baseline (§6.5).
3. **New CPU evidence [D] that shapes the selection** (§2):
   - **(a) The 2-epoch screen sits in a different scale regime from the 10-epoch members.**
     - Top sim-bias scale: init 8 → 9.6 at 2 ep, 12.1 at F9 10 ep, 21.2 at F7 10 ep.
     - c_ln gain: 1.3 → 1.8 → 3.2.
     - Muon-matrix RMS: 0.085 → 0.118 → 0.18.
     - Arms that add free scale knobs (DWA, HC, Derf, Peri-LN, LayerScale) can therefore show 2-epoch gains that 10 epochs reach anyway. **Pre-declared: these arms log their scalars, and no 2-epoch result counts without the stage-2 pair.**
   - **(b) The residual stream is dominated by the input token.** c_ln gain is 3.2. Cross-attention updates are 0.11-0.12× the stream and FFN updates 0.06-0.23×, whereas set-block-1 attention is 0.50× and history attention 0.57×.
   - **(c) The dense inputs are already well conditioned.** The ±6 clamp bites on ≤ 1.6% of candidates per feature, and only at sentinels. RealMLP robust scaling / smooth clipping therefore has no headroom. Point masses and skew remain, which is the target of pbld / ftok.
   - **(d) Cross attention has no sink** (2.3-2.8% mass on the user token) and already has one anti-similarity head per block (sim-bias scale −2.6 / −4.3).
4. **Not selected, with reasons (§7):**
   - nGPT, mHC (n = 4 HC over 8 sublayers cannot reach the ~3000× gain mHC fixes, and its doubly-stochastic constraint forbids the amplification the model visibly wants);
   - value residual (both cross blocks read the same context);
   - LayerScale / ReZero / SkipInit (subsumed by DWA's per-update scalars and by hl2's zero-init block);
   - RMSNorm (≈ identity of function), DyT (Derf dominates it in its authors' own comparison);
   - robust scale + smooth clip ([D] no headroom), TabM (closed), PLE (closed), DCNv2;
   - heads 8 / blocks 3 (closed), ff ratio (FFNs are the least-used sublayers [D]), warmup, AdamW betas, clip;
   - Muon momentum alone (a de-noising knob that trades member noise for lag; r6 §5 says de-noising does not help the pool);
   - cooldown_frac and aux-anneal target. These are horizon-dependent: a 2-epoch screen would select the 2-epoch optimum, so they belong to the continuation harness (§4).
5. **Honest expectation.**
   - Screen power: the pooled val SE is 0.00045-0.00061 (r6 calibration), so promotion at 1.5 SE means about +0.0008 and the MDE at 80% power is about +0.0014.
   - Literature effects of these blocks at our scale are 0 to +0.0005, so most arms will read null. Expect 1-3 promotions, about 0.4 of them false (per-arm false-promotion rate ≈ 3%).
   - Stage 2 (2 seeds, 10 ep) can confirm only effects of about +0.0015 once run-level noise is counted (§6.3).
   - The likely deliverables:
     - **(i)** a cheaper recipe (batch2s; resdrop0 / adrop0 speed);
     - **(ii)** same-strength structural variants for the stage-3 diversity report (derf, hc4, diff2, ftok change the function more than a seed);
     - **(iii)** a definitive closure of the "modern blocks" question for this model.
   - Screen cost: about 27 runs × ~50-60 min, i.e. 25-30 lane-hours, or about 4-5 h of wall-clock on 6-8 lanes.

---------------------------------------------------------------------------------------------------------------------

## 1. State, constraints, and what the model is

**Model** (nn/model.py, F7 checkpoint cfg [D]):
- d = 128, 4 heads, 2 cross + 2 set blocks, 1 history layer, ff 2, dropout 0.1 (attention probabilities, residual, FFN hidden, retrieval T).
- sim_bias = 1, gated attention (elementwise), split k/v for Muon, retr_m 16.
- n_dense = 104 = 61 base + 2 user0 + 31 xt + 10 within-feed rel.
- 1.01M parameters, of which 46 Muon matrices hold 0.944M.

**Residual streams:**
- candidate stream: c0 = c_ln(c_lin([e, e⊙u]) + cat + c_mlp(X) + miss) + dom + Z(retr), then 8 sublayers (cross0.attn, cross0.ff, set0.attn, set0.ff, cross1.attn, cross1.ff, set1.attn, set1.ff), then head = LN → Linear → GELU → Linear, plus the wide linear on X.
- history stream: h_ln(...), then 2 sublayers (hist0.attn, hist0.ff).

**Optimiser:**
- Muon (momentum 0.95, Nesterov, match_rms_adamw) on **every** 2-D Linear with min dim > 1, which includes the input projections.
- AdamW (0.9, 0.98), wd 0.02, on the rest.
- clip 1.0, warmup 500, WSD 1-√ cooldown 20% to a 3% floor.

**Closed (do not redo without a new mechanism):**
- FoX, QK-norm + SSMax, MoE, PLE, TabM, propensity head, semantic IDs, EWMA, stage-1 EMA, KD, LP-FT, stage 2 after WSD;
- plateau weight averaging (ensemble −0.0025), surrogate losses, DIN, d256 (AdamW);
- heads 8 / blocks 3 at 2 ep (old recipe), deep 3 + 2 ≈ tf.
- F12 d192 so far: s707 selected val 0.39442 (F7 mean 0.39460); the other seeds are still running, which suggests width under Muon is flat as well.

**Workflow snapshot, 04:11-04:26 UTC [D]:**

| GPU | processes now | memory used / (GPU memory size removed) |
|---|---|---|
| 0 | F10 GRU s505, s506 | 114 GB |
| 1 | LLM matched-test baselines | 94 GB |
| 2 | F12 s701, s703 | 118 GB |
| 3 | F12 s705 | 59 GB |
| 4 | fine-tune v26 s702 | 59 GB |
| 5 | LLM Qwen 9B probe | 120 GB |
| 6 | fine-tune v26 s704 + LLM 4B s24 | 59 GB (+70 GB arriving) |
| 7 | LLM 4B s23 | 59-71 GB |

- A 10-epoch F7-type training process shows **57-59 GB** in nvidia-smi. Budget about 55-60 GB per r7 process.
- Under the ≥ 80 GB-headroom rule, a GPU with about [n] GB free takes **2** new r7 processes, not 3.
- Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` should lower the reserved memory (not verified).

---------------------------------------------------------------------------------------------------------------------

## 2. New CPU diagnostics [D] (information only)

Setup: F7 member `v17_pre2s_s101` (selected checkpoint), 12 random val batches (334,720 candidates), CPU fp32, eval mode. Log: `.../scratchpad/r7/research/diag_cpu.log`.

### 2.1 Dense input after the model's own standardisation (104 features)
- Clamp |z| > 6: mean share 0.15% per feature, maximum 1.64%.
- The clamped mass is sentinels and point masses:
  - user0 · log1p(H): 1.64%;
  - the 6 similarity summaries: 1.45%, which is the −1.5 "no history" sentinel;
  - per-type category count t = 10: 1.27%;
  - ctx_hs_e1: 1.05%;
  - seq_cv_invd20_z: 0.89%.
- No feature exceeds 2%.
- Point masses: **21/104 features have a mode share > 50%, and 11 have > 80%.** Examples: seq_cv_invd20 89% (log1p(0)), ctx_hn_top1w 90%, ctx_hs_mutual 92%, and the it_hist_et2-4 counts 69-75%.
- Skew: 22 features have |skew| > 3, and 4 have |skew| > 10.
- **Reading:**
  - RealMLP robust scaling + smooth clipping would change almost nothing. This confirms nnx_research_tabular_dl §1.1 on the base features and extends it to the 41 xt/rel features.
  - The open question is per-feature nonlinearity at point masses. The single GELU layer of c_mlp handles that only through mixing. This is the target of **pbld** (smooth per-feature functions) and **ftok** (group-wise interaction before mixing).

### 2.2 Residual-branch sizes (mean over batches)

| sublayer | rms(stream in) | attn branch (× stream) | ffn branch (× stream) | cos(attn update, stream) |
|---|---|---|---|---|
| hist0 | 2.10 | 1.20 (0.57×) | 0.30 (0.12×) | +0.13 |
| cross0 | 4.44 | 0.50 (0.11×) | 0.61 (0.13×) | +0.12 |
| set0 | 4.61 | 1.07 (0.23×) | 1.00 (0.21×) | +0.08 |
| cross1 | 5.02 | 0.61 (0.12×) | 1.22 (0.23×) | +0.24 |
| set1 | 5.59 | 2.82 (0.50×) | 0.44 (0.06×) | +0.27 |

- Retrieval Z(r): rms 1.34 against a stream of 4.4.
- LN gains: c_ln **3.22**, head LN 0.46, ln2 (pre-FFN) 0.33-0.57.
- **Reading:**
  - The model has amplified its input token 3.2× through the only scale knob that is free of weight decay and Muon (an AdamW no-decay gain).
  - Target-attention updates are small corrections on top of it.
  - The last set block does the most work.
  - FFNs are the least-used sublayers, so FFN capacity (the ff ratio) is not binding. SwiGLU tests the FFN *form*, not its size.
  - Learnable depth-wise mixing (DWA, HC) lets the model re-weight input vs block outputs per reader without pushing c_ln. That is the concrete mechanism behind hc4 and dwa here.

### 2.3 Attention
- **Cross attention, mass on the user token:** 2.3-2.8% per head. No sink forms, presumably because the A1 gate already supplies "attend to nothing".
- **Cross attention, per head:**
  - normalised entropy (H / log n_keys) 0.87 / 0.72 / 0.64 / 0.47 (block 0) and 0.82 / 0.77 / 0.67 / 0.53 (block 1);
  - top-1 mass 0.09-0.38.
  - Learned sim-bias scales: [−2.6, 3.5, 11.3, 21.2] and [−4.3, 4.8, 10.5, 16.5]. **One head per block attends to *dissimilar* history**, so a hand-made contrast exists already.
- **Set attention**, normalised entropy per head: 0.60 / 0.92 / 0.70 / 0.93 (block 0) and 0.75 / 0.81 / 0.66 / 0.71 (block 1). Two block-0 heads are near-uniform, i.e. feed-mean pooling.
- **Reading:**
  - DIFF attention's "cancel common-mode attention noise" has material to act on: diffuse heads at 0.8-0.9 of maximum entropy, and an emerging contrast head.
  - Sink-type fixes (registers, gpt-oss sinks) have nothing to fix. They are not selected.

### 2.4 Scale growth: 2-epoch vs 10-epoch checkpoints (state dicts, read-only)

| checkpoint | top sim-bias scale (cross0 / cross1) | c_ln gain | retr.a | cross0 W_O RMS / spectral norm | c_mlp.0 spectral norm |
|---|---|---|---|---|---|
| sx_base_s23 / s24, v20_e2_s301 (2 ep, lr 1e-3) | 9.6-9.7 / 9.5-9.6 | 1.30 | 4.6-4.7 | 0.085-0.087 / 2.9-3.0 | 5.0-5.2 |
| v22_lr1_s401 (F9, 10 ep, lr 1e-3) | 12.1 / 11.4 | 1.77 | 3.55 | 0.118 / 4.85 | 6.2 |
| v17_s101@60000 (F7 plateau) | 18.9 / 15.3 | 2.95 | 1.16 | 0.181 / 7.05 | 8.7 |
| v17_s101 / s102 (F7 selected) | 21.2 / 20.2 | 3.22-3.25 | 0.72-0.76 | 0.178-0.183 / 7.1-7.3 | 8.5-8.6 |

**Reading:**
- At 2 epochs the Muon weights are far from their decay equilibrium. τ = 1/(lr·wd) = 50k steps against a 19.7k-step run (r6_design §3.2).
- The model is still sharpening its retrieval: top head 9.6 → 21, and retrieval moves from raw cosine (a 4.6 → 0.7) to learned key distance.
- Consequence for the screen: any arm that speeds up scale growth (free scalars, gains, output norms, higher effective lr on scalars) gets a 2-epoch advantage that need not survive 10 epochs. This is the same horizon trap r6_design §7 flags for wd and dropout.
- **Pre-declared:** every arm logs the four baseline scale quantities above, plus its own scalars, at the end. A promotion whose arm shows markedly faster scale growth than the paired baseline is marked "SCALE-SUSPECT" and goes to stage 2 with no other weight in the decision.

---------------------------------------------------------------------------------------------------------------------

## 3. Survey mapped onto our model

### 3.1 Normalisation

| technique | what it is (source) | evidence | mapping onto our model | decision |
|---|---|---|---|---|
| Pre-LN (current) | `x + F(LN(x))` | - | 15 block LNs plus c_ln / h_ln / head / v_ln | baseline |
| RMSNorm | LN without mean-centring or bias [K] | LLM-standard; quality ≈ LN; speed gain only when kernels are LN-bound | the same function class at d = 128; no speed gain at our size | **not selected** (≈ identity) |
| DyT | `γ tanh(αx) + β`, α scalar (init 0.5 for non-LLMs; smaller for wide LLMs), replaces every norm including the final one [V 2503.10622] | "matches or exceeds" LN; fails as a BatchNorm replacement (ResNet-50 76.2 → 68.9); ViT-L diverges at α0 > 0.6; no speed-up once compiled [V] | at width 128 the default α0 = 0.5 is in the safe range | **superseded by Derf** |
| **Derf** | `γ erf(αx + s) + β`, α = 0.5, s = 0, γ = 1, β = 0; replaces the pre-attention, pre-FFN and final norms one-to-one [V 2512.10938, CVPR 2026] | ViT-B 82.3 (LN) / 82.5 (DyT) / **82.8**; DiT-XL FID 19.94 / 20.83 / **18.92**; wav2vec, DNA small gains; **GPT-2 124M ties LN (2.94)**. Higher training loss than LN, which the authors attribute to better generalisation [V] | our objective is a shifted-distribution ranking metric (production → exploration), where a "regulariser-like" norm could plausibly help; it also changes the function more than any other block swap (diversity) | **selected (derf)**. Input-embedding norms stay LN because they mix very different input scales |
| nGPT | everything on the hypersphere, `h ← Norm(h + α(h_A − h))`, normalised weights, no wd / warmup [V 2410.01131 via R] | 4-20× token efficiency at 0.5-1B, with 60-80% step overhead and α sensitivity; mixed replications [R, K] | requires weight normalisation that conflicts with Muon + decoupled wd and with the scale growth we rely on (§2.4); QK-norm (its attention piece) is closed | **not selected** |
| **Peri-LN / sandwich** | `x + LN_out(F(LN_in(x)))`, output-LN γ initialised to 1 [V 2502.02732] | 400M: avg 49.69 → 51.57, loss 3.43 → 3.34; 1.5B: 53.71 → 56.55; 3.2B: 56.69 → 58.56; seed std more than halved; Gemma and OLMo-2 use output norms [V] | gives each sublayer a free output gain, so the model need not inflate c_ln (§2.2) | **selected (periln)**, low prior (it fixes depth-wise variance growth, and we have 5 blocks) |
| Post-LN + DeepNorm | `x_{l+1} = LN(α x_l + G(x_l))`; encoder-only α = (2N)^¼, init β = (8N)^−¼ on v, o and FFN weights [K, DeepNet 2203.00555] | Post-LN is often better than Pre-LN in shallow nets when it trains; DeepNorm stabilises it [K]. In the Peri-LN paper, Post-LN at 400M was far worse (avg 42.45) [V] | candidate stream N = 4 blocks → α = 1.68, β = 0.42; history N = 1 → α = 1.19, β = 0.59 | **reserve (postln)** |

### 3.2 Residual schemes

| technique | what / source | evidence | mapping | decision |
|---|---|---|---|---|
| LayerScale / ReZero / SkipInit | per-branch learnable diag(λ) or scalar α (init 1e-4 … 0.1, or 0) [K] | helps very deep ViTs and post-norm stability; ~0 for shallow pre-LN [K] | under Muon the λ scalars (AdamW) reach O(0.1) in about 100 steps; in pre-LN, ln1's gain already scales the attention branch linearly (attention is linear in V) | **not selected**. DWA has per-update scalars and more; hl2 uses a ReZero-style zero-init block |
| **Hyper-Connections** | n parallel residual streams; per layer an (n+1)×(n+1) matrix [[0, B], [A_m, A_r]]: input h0 = A_mᵀH, write-back Ĥ = Bᵀ T(h0) + A_rᵀH. Static init A_m = e_{k mod n}, A_r = I, B = 1 (≡ Pre-LN). Dynamic (DHC): + s·tanh(norm(H) W) with W = 0, s = 0.01 [V 2409.19606] | OLMo-1B: V2 loss 2.811 → 2.781, n = 4 best (n = 1 worse than baseline; n = 2 +0.5, n = 4 +1.3 downstream avg); OLMo-7B 2.581 → 2.559; ViT-B 76.38 → 77.60 (SHC); +26% activation memory, +0.03% params [V]. No depth-controlled study; gains larger for ViT-L than ViT-B [V] | candidate stream, 8 sublayers, n = 4, dynamic, mean readout (exact at init). Lets the model keep an input-preserving stream next to working streams, and run set / cross sublayers in parallel when useful | **selected (hc4)** |
| mHC | HC with H_res projected to doubly stochastic by Sinkhorn-Knopp (20 iterations), H_pre = σ(·), H_post = 2σ(·), n = 4 [V 2512.24880] | fixes HC's composite gain of ~3000× at 27B / 30 layers; 27B BBH 43.8 / 48.9 (HC) / 51.0; +6.7% time [V] | at 8 sublayers the HC composite gain cannot blow up; the doubly-stochastic A_r forbids amplification, and the model visibly amplifies its input (c_ln 3.2) | **not selected**. Fallback if hc4 shows instability (spikes in the eval curve) |
| **DenseFormer (DWA)** | after block i: Y_i = Σ_{j≤i} α_{ij} X_j (X_0 = embedding), with α_ii = 1 and the rest 0 [V 2402.02622] | 48 layers: ppl 18.61 → 17.84 (≈ a 72-layer baseline); 24 layers (PG-19): 20.13 → 19.60. Learned α put large weight on X_0, positive early and negative late [V] | 4 candidate blocks → 14 scalars; exact no-op at init. The learned pattern "large weight on X_0" is what our model does with c_ln, and DWA does it per reader | **selected (dwa)** |
| Value residual (ResFormer) | V_n ← λ1 V_1 + λ2 V_n [V 2410.17897 via R] | −16% params at equal loss (468M); later layers benefit [R] | both cross blocks read the same encoded context (values = projections of the same ctx); the set stream has 2 blocks; ctx already contains h via the residual | **not selected** |

### 3.3 FFN
- **SwiGLU / GeGLU vs GELU at matched parameters** (Shazeer 2020 [V via R]): T5-base log-ppl 1.997 (ReLU) / 1.983 (GELU) / 1.942 (GEGLU) / 1.944 (SwiGLU) at 65K steps. Used by every modern LLM in scope [R].
  - Ours: ff 2 GELU = 65,920 params per block. SwiGLU hidden 170 with biases = 65,748 (−0.3%).
  - Our FFNs carry 0.06-0.23× updates [D], so the expected effect is small.
  - **Selected (swiglu).** One arm, GLU form only. GeGLU ≈ SwiGLU in the source, so it gets no separate arm.
- **Expansion ratio:** never varied. Capacity is flat everywhere and the FFNs are the least-used sublayers [D]. **Not selected.**

### 3.4 Attention
- **Differential attention.**
  - V1 [V 2410.05258 via R]: (softmax(Q1K1ᵀ) − λ softmax(Q2K2ᵀ))V, with a per-head GroupNorm (worse than baseline without it) and λ_init 0.8 − 0.6e^{−0.3(l−1)}.
  - **V2 (Jan 2026)** [S, author blog mirror]:
    - query heads doubled, shared K/V within the group;
    - λ = sigmoid(projection) per token and head;
    - per-head RMSNorm *removed* (instability), output bounded in (0, √2);
    - FlashAttention-compatible;
    - LM loss 0.02-0.03 below the Transformer at 1T tokens (preliminary).
  - Mapping: target attention over ≤ 256 history items, where heads are diffuse (0.8-0.9 normalised entropy) and a contrast head already exists (§2.3). V2 is simpler, needs no GroupNorm and keeps W_O's shape. Each of the 8 query maps gets its own sim-bias scale.
  - **Selected (diff2)**, cross blocks only.
- **Attention dropout:** owned by r6 (adrop0).
- **More or fewer heads:**
  - heads 8: closed (−0.0011, old recipe);
  - heads 2: no mechanism, and it would cut the sim-bias heads from 4 to 2;
  - diff2 adds query maps in a principled way.
  - **Not selected.**
- **Sinks / registers / SSMax / QK-norm:** closed, or nothing to fix (§2.3).

### 3.5 Tabular-DL input encoders (the dense path: 104 inputs → c_mlp, a 1-hidden-layer GELU MLP, plus the wide linear)

| technique | source / evidence | mapping | decision |
|---|---|---|---|
| RealMLP robust scaling + smooth clip `f(x) = x / √(1 + (x/3)²)` after median / IQR scaling [V 2407.04491] | ablation: robust + smooth clip is the best preprocessing (regression error +9.5% for robust only) [R] | [D] §2.1: the clamp bites ≤ 1.6%, only at sentinels; xt counts are already log1p'd (data.py attach_xt) | **not selected** (no headroom) |
| **RealMLP PBLD**: per feature `[x, W2 cos(2π w1 x + b1) + b2]` ∈ R⁴, 16 frequencies, w1 ~ N(0, 0.1²), b1 ~ U[−π, π] [V] | removing PBLD costs +2.3% (classification) / +20.6% (regression) error; PBLD vs PL +0.7 / +0.5; PLR (periodic) is the best of the 2022 embeddings on 11 datasets; tiny on MSLR (0.747 → 0.746) [V, R] | the closed PLE fits production step functions (B1). Low-frequency periodic features (σ 0.1, period ≈ 10 z-units) are smooth and cannot memorise steps. Zero-init "version-B" form: exact at init | **selected (pbld)** |
| RealMLP diagonal scaling layer (s_i = 1, lr ×6) [V] | removal costs +1.4 / +1.0% | per-feature step sizes for c_mlp's input, which **Muon removes** (orthogonalised update) | folded into **muon_in** (AdamW gives per-coordinate step sizes); the scaling layer is the stage-2 follow-up if muon_in passes |
| RealMLP parametric activations, NTP, β2 0.95, label smoothing | small, or harmful for ranking metrics [V, R] | - | not selected |
| **FT-Transformer feature tokens** [K 2106.11959] | ≈ MLP-PLR on MSLR; quadratic in the feature count [R] | 104 tokens × 32K candidates is too costly → **8 semantic group tokens + CLS** (RankMixer-style tokenisation, at the granularity of our feature families) | **selected (ftok)** |
| TabR (A4) | adopted (retr16) | r6 a4u / a4w vary it | owned by r6 |
| TabM | closed (B5, members 0.999-correlated) | - | closed |
| ModernNCA | cosine / L2 > dot; stochastic neighbour sampling → history-key dropout [R] | L2 keys are in A4; history dropout is a regulariser (blind at 2 ep, §4) | deferred to the continuation harness |
| PLE | closed (B1) | - | closed |

### 3.6 Regularisation
- **Stochastic depth / drop-path:** rejected earlier for depth 5 (nnx_design §8). At 2 epochs, where nothing over-fits, it can only look negative. Its mechanism (anti-memorisation, more member noise) acts in the 10-epoch cooldown. **Deferred** to the continuation harness, spec in §4.2.
- **Dropout:** see resdrop0 (r7) and adrop0 (r6). A higher dropout (0.2) is a cooldown-phase regulariser (r6_recap P2, lead).
- **Weight decay:** owned by r6 (wd01; at 10 epochs wd 0.05 keeps lr·wd = 1e-4).

---------------------------------------------------------------------------------------------------------------------

## 4. Never-varied hyperparameters: analysis and decisions

### 4.1 Decision table

| knob (current) | evidence | horizon where it acts | decision |
|---|---|---|---|
| **batch** 32,768 cands / 512 feeds (~317 feeds) | B_simple 4.9K (F9) - 9.4K (F7) feeds ≫ 317 (r6_recap §4.2), so a 2× batch costs about +4-7% examples (McCandlish). Muon keeps data efficiency past the critical batch, and Nesterov shifts it right [S 2505.02222, 2507.01598]. Muon momentum 0.95 already averages ~20 steps ≈ 6.3K feeds ≈ B_noise | both: strength at 2 ep; diversity at 10 ep | **r7 arm batch2s** with the SDE square-root rule (§4.3) |
| **dropout** 0.1 (all sites) | never varied (285/285 runs); 2-ep runs never over-fit; 10-ep cooldowns memorise the pool | 2 ep: less dropout fits more; 10 ep: risk | **r7 resdrop0** (residual / FFN / retrieval) + r6 adrop0 (attention). Dropout 0.2: continuation harness |
| **wd** 0.02 | τ = 50k steps > run; scale growth [D §2.4] | both | **r6 wd01** (not duplicated) |
| **optimiser partition** (Muon on input projections) | Muon's author: "input and output layers should be optimized by AdamW, even though these are typically 2D" [V]; our input layers include c_mlp.0 over 104 heterogeneous features; c_mlp.0 spectral norm grows 5.0 → 8.6 [D] | both | **r7 muon_in** (new knob; never varied) |
| **hist_layers** 1 | the history attention is the most-used sublayer (0.57× update) [D]; tested only bundled with blocks 3 (≈ tf, AdamW era) | both | **r7 hl2** (zero-init extra layer, low priority) |
| retr_m 16 | A4 is the strongest architecture item | 2 ep sees it (low-noise mean function) | **r6 a4w / a4u** |
| heads 4 | heads 8 −0.0011 (old recipe, closed) | - | not selected; diff2 adds query maps |
| blocks 2 | blocks 3 −0.0006 (closed); deep ≈ tf | - | not selected; hc4 / dwa test depth-wise composition instead of depth |
| ff 2 | FFN updates 0.06-0.23× [D]; capacity flat | - | not selected; swiglu tests the form |
| warmup 500 | no instability anywhere; 2.5% of a 2-ep run, 0.5% of a 10-ep run | - | not selected |
| Muon momentum 0.95 | standard; ≈ B_noise averaging window already; a de-noising knob (r6 §5: de-noising helps singles and hurts the pool) | 10 ep (diversity) | not selected alone; scaled inside batch2s per the SDE rule |
| AdamW betas (0.9, 0.98), clip 1.0, lr floor 3% | no instability; floor acts only in the memorising tail | - | not selected (betas scaled inside batch2s) |
| **cooldown_frac** 0.2 | 2-ep cooldowns never decline and are still rising at the end, so a 2-ep screen would pick a longer cooldown. At 10 ep the best point is 1-9k steps into the cooldown, and longer means more pool passes and more memorisation (r6_recap §3.3) | **10 ep only** | **not in the 2-ep screen**; proper test = continuations from saved plateau snapshots (§4.2) |
| **aux anneal target** 0.1 | NN-agent hypothesis for the late decline; 2-ep cooldown is 3.9k steps | **10 ep only** | continuation harness (§4.2) |

### 4.2 Proper selection of the horizon-dependent knobs (recommended to the lead; not part of the r7 screen)
- These knobs act only in the 10-epoch cooldown regime, which the 2-epoch screen cannot see: cooldown_frac / cooldown length, aux-anneal target, dropout 0.2, stochastic depth and history dropout.
- **The paired design:** continuations from shared plateau snapshots `nn/ckpt/v17_pre2s_s10{1..7}@{60000,77500}.pt`:
  - same start, same data order;
  - same-start divergence 0.0002-0.0005 in the first 10k steps (r6_recap §4.5), far below the 0.0007 late-plateau non-determinism;
  - r6_recap P1 / P2 and r6_external R1 / R2 give the harness;
  - each arm = 2 starts × 1 continuation of 0.5-2 epochs, scored at the holdout-selected step and at the end, with the declining tail as the mechanism readout.
- Grid, in priority order:
  - cooldown length {0.5, 1, 2} epochs from @77500;
  - aux_anneal {0.1, 0.5};
  - dropout {0.1, 0.2} (needs a dropout override after `--init`);
  - drop-path {0, 0.1} on the 8 candidate sublayers;
  - hist_drop {0, 0.15}.
- About 10 runs × 20-50 min.
- This is the proper selection method for these knobs. I recommend it as the lead's lane (it overlaps r6 P1 / P2).

### 4.3 lr scaling rules (pre-declared)
- **Batch (batch2s): SDE square-root rule** (Malladi et al. 2022, "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms" [K]). For batch ×κ:
  - lr ×√κ;
  - (1 − β) ×κ for every momentum-like coefficient;
  - warmup and evaluation interval held constant in samples.
- With κ = 2:
  - lr 1e-3 → 1.414e-3 (2 ep), and 2e-3 → 2.83e-3 at 10 ep;
  - Muon momentum 0.95 → 0.90;
  - AdamW betas (0.9, 0.98) → (0.8, 0.96);
  - warmup 500 → 250;
  - eval_every 1000 → 500 (10 ep: 2500 → 1250);
  - save_steps halved.
- Why sqrt and not linear:
  - Muon and AdamW are normalised-update optimisers. Their SDE scaling is square-root; linear scaling is the SGD rule.
  - Our regime (B ≪ B_noise) is exactly where SDE approximations hold, so the rule should preserve the plateau noise level, i.e. member diversity, which r6 identified as the ensemble's resource.
  - The momentum adjustment keeps Muon's averaging window at ~6.3K feeds.
- **Dropout / optimiser-partition / architecture arms:** lr unchanged. The A/A baseline sits in the same batch.
- **wd (r6):** lr·wd held constant across horizons (r6_design §3.2).

---------------------------------------------------------------------------------------------------------------------

## 5. Selected arms: exact specs

**Common rules** (house rules plus r6 practice):
- **Code copy:** code goes in `nn_x/r7/`, copied from nn/*.py and *.sh. Symlink `feat_cache → ../../nn/feat_cache`, read-only, never link into it. CKPT / LOGS / PREDS (and preds/holdout) point into nn_x/r7, and the GPU guard accepts 0-7.
- **Flag-off identity:** with every new flag off, the model and training are bit-identical to nn/. The test is the r6 T2-style check: same weights on real batches give identical eval logits, train-mode loss and gradients.
- **cfg keys** are stored only when non-default, so flag-off checkpoints keep the nn/ cfg.
- **New modules** are built under `extra_init()` and created **after** all baseline modules. Where an arm replaces a baseline module that consumes RNG (swiglu's FFN), the original module is still constructed and then discarded, so the global RNG stream, and with it every other module's init, is unchanged. Test T1: the global RNG state after construction equals the baseline's, and 0 baseline tensors change.
- **Optimiser placement of new parameters:**
  - Muon (Moonlight rule, nn.Linear with min dim > 1) unless stated otherwise;
  - scalars and gains go to AdamW **no-decay**;
  - HC / DWA / Derf / λ parameters are plain nn.Parameter, not Linear, so the Muon filter skips them. Their names must reach the no-decay group: the train.py decay filter `p.ndim >= 2 and "emb" not in n` would otherwise decay HC's 2-D W tensors, so add a name filter `("hc.", "dwa.", "derf")` → no_decay.
- **Screen command** = r6 BASE2 (= sx_base):
  ```
  --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout
  --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.1
  --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 1000 --seed {23,24}
  ```
  Arm flags are appended last.
- **Stage-2 command** = the F7 recipe (run_final7b.sh = r6 BASE10, lr 2e-3, eval 2500, `--save_steps 50000,60000,70000,77500`) + arm flags, seeds 23/24, against r7_base10_s23/s24.

### 5.1 muon_in: input projections on AdamW (training; priority 1)
- **Flag:** `--muon_inputs adamw` (default `muon`).
- In train.py, remove `{c_mlp.0.weight, c_lin.weight, h_lin.weight, u_lin.weight, retr.wk.weight}` from `muon_names`. They go to the AdamW **decay** group: wd 0.02, the same lr and schedule, betas (0.9, 0.98).
- Log both partitions: Muon drops from 46 to 41 matrices.
- **Mechanism:**
  - These five matrices read fixed, non-learned feature vectors: 104 standardised tabular features, and 128-d content embeddings for candidate / history / user / retrieval keys.
  - Muon orthogonalises the momentum, so every input direction gets the same step regardless of its gradient scale.
  - For heterogeneous tabular inputs (point masses, skew, binary flags), AdamW's per-coordinate normalisation is the right geometry; it is what RealMLP's high-lr diagonal scaling layer also emulates.
- **Evidence:** Muon's author: "scalar and vector parameters ... as well as the input and output layers, should be optimized by a standard method such as AdamW", "important ... even though these are typically 2D" [V]. Moonlight / Kimi / Qwen3.8 keep embeddings and gates on AdamW [R].
- **Params:** 0. **Exact at init:** yes (same model). **Cost:** 1.0×.
- **Expected:** −0.0003..+0.0010.
- **Diagnostics:** spectral norms of the five matrices vs the paired baseline, and the c_ln gain.

### 5.2 batch2s: batch ×2 under the SDE square-root rule (hyperparameter; priority 2; cost lever)
- **Flags:**
  ```
  --cand_budget 65536 --max_feeds 1024 --lr 1.414e-3 --warmup 250 --eval_every 500 --muon_momentum 0.90 --adam_betas 0.8,0.96
  ```
  New flags `--muon_momentum` (default 0.95) and `--adam_betas` (default 0.9,0.98).
- **Unchanged:** `loss_norm feed` with denom = max_feeds keeps the per-feed loss scale (feeds per batch / max_feeds ≈ 0.62 in both). The K9 normalisation is per batch.
- **Stage 2:** lr 2.83e-3, eval_every 1250, save_steps 25000,30000,35000,38750.
- **Pairing:** same init; batch composition differs, so pairing is weaker than for the other arms.
- **Pre-declared extra (cost) rule, alongside the standard strength rule:**
  - **COST-PASS** if pooled val Δ ≥ −0.0005 AND holdout Δ ≥ −0.0003 (about −1 SE each);
  - AND training minutes ≤ 0.75× the paired baseline run on the **same GPU in the same wave**;
  - COST-PASS goes to stage 2. There, adoption also needs the 2-member TO ensemble ≥ the baseline pair's on the holdout, and within-feed seed correlation ≤ the baseline pair's + 0.005, because batch changes plateau noise (r6 §5).
- **Expected:** strength −0.0005..+0.0002; wall-clock 0.55-0.75×.
- **Memory:** activations double, roughly +10-20 GB per process. Check headroom.

### 5.3 resdrop0: residual / FFN / retrieval dropout 0 (hyperparameter; priority 3)
- **Flag:** `--res_drop 0` (default −1 = use `--drop`). It sets `Block.dr`, `Block.ff[2]` and `HistRetrieval.dp` to p = 0. `Attn.drop` (attention-probability dropout) stays at `--drop` = 0.1, so this does not overlap r6 adrop0.
- **Mechanism:** a 1.01M-param model on 3.15M feeds × 2 epochs under-fits (29/29 two-epoch runs select one of the last two evals, 26/29 the last; r6_recap §3.3). Dropout slows the fit and adds noise.
- **10-epoch risk:** more pool memorisation in the late cooldown. The stage-2 pair decides; log the end − selected decline.
- **Params:** 0. **Cost:** ≈ 0.9×.
- **Expected at 2 ep:** +0.0002..+0.0010. The 10-ep sign is unknown.
- If both resdrop0 and adrop0 promote, stage 2 should also run the combination drop 0 (all sites), pre-declared.

### 5.4 pbld: RealMLP PBLD periodic embeddings into c_mlp (input-encoding; priority 4)
- **Flag:** `--pbld 16` (number of frequencies; 0 = off). Built under extra_init after all baseline modules.
  - `w1 ∈ R^{104×16} ~ N(0, 0.1²)`, `b1 ∈ R^{104×16} ~ U[−π, π]`;
  - `W2 ∈ R^{104×16×3}` (per-feature, default Linear-style init), `b2 ∈ R^{104×3} = 0`;
  - `P = nn.Linear(312, 2d, bias=False)`, **zero-init**.
- **Forward**, with X the standardised, clamped dense input [B, M, 104]:
  ```
  per = einsum('bmfk,fkj->bmfj', cos(2π·w1·X[...,None] + b1), W2) + b2       # [B,M,104,3]
  h   = c_mlp[0](X) + P(per.flatten(-2));  c_mlp_out = c_mlp[2](GELU(h))         # replaces c_mlp(X)
  ```
  This is the RealMLP form with the raw x kept through c_mlp[0], and exact baseline at init.
- **Optimiser:**
  - P → Muon (Moonlight rule);
  - W2 (3-D) → AdamW decay;
  - w1, b1, b2 → AdamW no-decay.
  - Under muon_in, P joins the AdamW inputs.
- **Params:** +88K (+9%). **Cost:** [32768, 104, 16] cos ≈ 55M elements, ≈ 1.05×.
- **Why not the closed PLE:**
  - PLE's quantile-bin steps fit production step functions that do not transfer (B1: val −0.0004, holdout +0.0008).
  - Periodic features with σ = 0.1 have periods of about 10 z-units, i.e. smooth per-feature curves.
  - [D]: 21 features have point masses > 50% and 22 have |skew| > 3.
- **Expected:** 0..+0.0008.

### 5.5 hc4: dynamic Hyper-Connections, n = 4, candidate stream (architecture; priority 5)
- **Flag:** `--hc 4` (0 = off). It refactors Block into two sublayer callables:
  - `attn_sub(x, ctx, kmask, bias) = dr(attn(ln1(x), lnc(ctx) or ln1(x)))`;
  - `ff_sub(x) = dr(ff(ln2(x)))`.
  - The flag-off path keeps the original `Block.forward`.
- One HC unit per candidate-stream sublayer (k = 0..7 in order cross0.attn, cross0.ff, set0.attn, set0.ff, cross1.attn, cross1.ff, set1.attn, set1.ff). Each holds:
  - static `A_m = e_{k mod 4}` (4), `A_r = I_4` (16), `B = 1_4` (4);
  - dynamic `W_m ∈ R^{d×1}`, `W_r ∈ R^{d×4}`, `W_b ∈ R^{d×1}`, all **zero**;
  - an RMSNorm(d) with gain 1 for H̄;
  - scalars `s_a = s_b = 0.01`.
- **Forward:**
  ```
  H = c[..., None, :].expand(B, M, 4, d)                       # c = the current c0 (after c_ln, dom, retrieval)
  for each sublayer k:  Hn = rms(H)
      Am = A_m + s_a·tanh(Hn @ W_m)          # [B,M,4,1]
      Ar = A_r + s_a·tanh(Hn @ W_r)          # [B,M,4,4]
      Bk = B   + s_b·tanh(Hn @ W_b)          # [B,M,4,1]
      h0 = (Am * H).sum(-2);  y = sub_k(h0, ...)
      H  = Bk * y[..., None, :] + einsum('bmjk,bmjd->bmkd', Ar, H)
  c = H.mean(-2)                                                # mean (not sum): exact baseline at init, dvec added after
  ```
- The history stream (2 sublayers) is unchanged.
- **All HC params** go to AdamW no-decay (plain Parameters).
- **Params:** about 8.4K. **Memory:** about 4× candidate-stream activations (+ a few hundred MB). **Cost:** ≈ 1.1×.
- **Diagnostics:** static and dynamic magnitudes at the end (A_r off-diagonals, B), s_a, s_b.
- **Expected:** 0..+0.0006, and a likely function change (diversity readout).

### 5.6 dwa: DenseFormer depth-weighted average (architecture; priority 6)
- **Flag:** `--dwa` (store_true). Create `dwa = nn.ParameterList` with 4 vectors of sizes 2, 3, 4, 5, initialised to one-hot on the last entry.
- **Forward**, with ys holding block outputs:
  ```
  ys = [c0];  x = c0
  for i in range(blocks):
      x = cross[i](x, kmask, ctx, sbias); ys.append(x); x = Σ_j dwa[2i][j]·ys[j]
      x = sets[i](x, cmask);              ys.append(x); x = Σ_j dwa[2i+1][j]·ys[j]
  ```
  Exact at init.
- **Params:** 14 scalars, AdamW no-decay. **Cost:** 1.0×.
- **Diagnostic:** the learned α matrix. DenseFormer reports large weight on X_0; ours would let readers lower or raise the input token per block.
- **Expected:** 0..+0.0005.

### 5.7 diff2: DIFF-V2 target attention (architecture; priority 7)
- **Flag:** `--diff_attn cross` (default off).
- **New modules**, under extra_init, in each cross block's Attn:
  - `q2 = Linear(d, d)`;
  - `lam = Linear(d, heads)` (default init, so σ ≈ 0.5 at init, as in V2, which has no λ_init);
  - `bias_scale2 = Parameter(linspace(0, 8, heads))` (deterministic).
- **Forward:**
  ```
  o1 = SDPA(q,  k, v, mask + bias_scale ·S);  o2 = SDPA(q2, k, v, mask + bias_scale2·S)   # same k, v (shared, as V2)
  o  = o1 − sigmoid(lam(x)).view(B,N,h,1).transpose(1,2) * o2
  ```
  Then the existing A1 gate, then W_O.
- No per-head norm (V2 removed it). Attention dropout 0.1 in both calls.
- Set blocks and hist_enc are unchanged.
- **Params:** +34K (+3.4%), Muon on q2 and lam (lam is 128×4, so Muon by the Moonlight rule; acceptable). **Cost:** two SDPA calls in 2 of 5 attention modules, ≈ 1.15×.
- **Diagnostics:** mean λ per head; the learned bias_scale2.
- **Expected:** −0.0005..+0.0008.

### 5.8 swiglu: SwiGLU FFN at matched parameters (architecture; priority 8)
- **Flag:** `--ffn swiglu` (default gelu). In every Block (5):
  - construct the original `self.ff` (consumes the baseline RNG draws);
  - then, under extra_init, build `ff_g = Linear(d, 170)`, `ff_u = Linear(d, 170)`, `ff_o = Linear(170, d)`;
  - delete `self.ff`.
  - Forward: `ff(x) = ff_o(dropout(silu(ff_g(x)) * ff_u(x)))`.
- Gate and up are separate Linears so Muon orthogonalises each (Qwen3.8 per-sub-matrix rule [R]).
- **Params:** 65,748 vs 65,920 per block. **Cost:** 1.0×.
- **Expected:** 0..+0.0005.

### 5.9 derf: Derf instead of LayerNorm in the blocks and heads (architecture; priority 9)
- **Flag:** `--norm derf` (default ln).
- **Class:** `Derf(d)`: `alpha = 0.5`, `s = 0.0` (scalars); `weight = ones(d)`, `bias = zeros(d)`; `forward = weight * erf(alpha * x.float() + s) + bias`.
- **Replaced:** ln1, lnc, ln2 in all 5 blocks, head[0] and aux_head[0].
- **Kept as LayerNorm:** c_ln, h_ln, retr.v_ln. These input-embedding norms combine sources of very different scale, and replacing them would confound the norm test with an input-encoder change.
- No RNG is consumed.
- **Params:** +2 per site, AdamW no-decay. **Cost:** 1.0×.
- **Expected:** −0.0010..+0.0008. Highest function change of the block swaps (diversity readout).
- **Caveat:** the residual stream will settle at a different scale (§2.2), so log α per site.

### 5.10 hl2: second history layer, zero-init outputs (hyperparameter; priority 10)
- **Flag:** `--hist_extra 1`.
- After all baseline modules, under extra_init, build one more `Block(d, heads, drop, ff, gate=attn_gate, split_kv=muon)` and append it to `hist_enc`.
- Zero its `attn.o.weight / bias` and `ff[3].weight / bias`, so it is an exact no-op at init (ReZero-style for the new block).
- **Params:** +148K (+15%). **Cost:** ≈ 1.1× (history tokens ≤ 257 per feed).
- **Expected:** 0..+0.0005.

### 5.11 periln: Peri-LN output norms (architecture; priority 11)
- **Flag:** `--periln`. Each Block gets `ln_ao`, `ln_fo = LayerNorm(d)` (γ = 1, β = 0; deterministic).
- **Forward:**
  ```
  x = x + dr(ln_ao(attn(ln1(x), c)));  x = x + dr(ln_fo(ff(ln2(x))))
  ```
- The embedding norm (c_ln / h_ln) and final norm (head LN) already exist.
- **Params:** +2.6K. **Cost:** 1.0×.
- **Expected:** −0.0005..+0.0005.

### 5.12 ftok: feature-group tokens (input-encoding; priority 12)
- **Flag:** `--ftok 8`. Valid for the cvnd_pre2s layout (n_dense 104; assert it). Group index lists are stored in cfg.
- **Groups** over the standardised X (indices follow dense_features order; derivation in §8):

  | group | name | indices |
  |---|---|---|
  | 1 | f1..f5 raw / z / pct | 0-14 |
  | 2 | content similarity | 15-34, 51-60 |
  | 3 | category affinity | 35-47 |
  | 4 | feed context + user0 | 48-50, 61-62 |
  | 5 | co-visitation | 63-65, 94-95 |
  | 6 | LUO popularity | 66-70, 96-97 |
  | 7 | neighbour context (ctx_nd / hn / hs) | 71-90 |
  | 8 | pre2 encoder | 91-93, 98-103 |

- **New modules**, under extra_init:
  - per group `Linear(n_g, d)` + group embedding;
  - a learned CLS (d);
  - 1 × `Block(d, heads, drop, ff, gate=attn_gate, split_kv=muon)` over the 9 tokens (no mask);
  - `Zf = Linear(d, d)` **zero-init**.
- **Forward:** `c_pre = c_lin(...) + cat + c_mlp(X) + miss + Zf(Block([CLS; tokens])[:, 0])`, then `c_ln`. Exact at init.
- **Params:** about +165K (+16%). **Cost:** 9 tokens × 32K candidates through one small layer, ≈ 1.2×.
- **Expected:** 0..+0.0006.

### 5.13 postln (reserve): Post-LN with DeepNorm
- **Flag:** `--postln`. Block forward:
  ```
  x = ln1(α x + dr(attn(x, lnc(ctx) or x)));  x = ln2(α x + dr(ff(x)))
  ```
  - α = (2N)^¼ with N = 4 (candidate blocks) → 1.682; history block N = 1 → 1.189.
  - Init: multiply the existing (baseline-drawn) v, o, ff[0], ff[3] weights by β = (8N)^−¼ (0.420; history 0.595). q / k are unchanged. No RNG change.
- **Expected:** −0.002..+0.0005. Run only on idle lanes.

---------------------------------------------------------------------------------------------------------------------

## 6. Notes for the screen agent

### 6.1 Order and budget
- **Jobs:**
  - fresh `r7_base_s23`, `r7_base_s24`, plus an optional A/A rerun `r7_base_s23r` (rule: |Δval| ≤ 0.0010, |Δhold| ≤ 0.0005; otherwise audit the copy);
  - then the arms in priority order: muon_in, batch2s, resdrop0, pbld, hc4, dwa, diff2, swiglu, derf, hl2, periln, ftok (postln reserve).
  - That is 24 arm runs + 2-3 baseline runs ≈ 27 runs.
- **Pairing:** put each arm's two seeds on different GPUs. Where possible put an arm and its baseline seed on the same GPU in the same wave, for the batch2s wall-clock rule.
- **Cost checks:** log min / 1000 steps (base ≈ 2.5-2.6 at 3 processes per GPU, `nn_x/solvers/logs/sx_base_s23.log`) and peak memory.

### 6.2 Readouts (beyond the pre-declared rule)
- **Slices:** M<50, M≥50, Tue, Thu, late holdout. The r6_eval slice code can be reused.
- **Scale diagnostics (§2.4):** top sim-bias scale, c_ln gain, retr.a, cross0 W_O RMS, plus the arm's own scalars. Mark SCALE-SUSPECT per §2.4.
- **Secondary, information only:** cross-seed function change ρ_x vs ρ_bb (r6 FAMILY-CANDIDATE definition). Structural arms may be same-strength, different-function members for stage 3 even if they do not promote.

### 6.3 Statistics
- **Stage 1** (2 ep): pooled per-feed paired val SE 0.00045-0.00061 (r6_design §4.4). The same-seed rerun gap of 0.0002-0.0005 is inside that.
  - Promotion at > 1.5 SE ≈ +0.0008.
  - Expected false promotions across 12 arms ≈ 0.4.
- **Stage 2** (10 ep): same-seed non-determinism is |Δval| ≈ 0.0007 late in the plateau (r6_recap §4.5), comparable to the per-feed SE. The per-feed SE therefore **understates** the run-level noise.
  - Recommended calibrated SE: `sqrt(SE_feed² + 0.0007²/n_seeds)` on val, and `sqrt(SE_feed² + 0.0003²/n_seeds)` on the holdout.
  - With 2 seeds, 2 SE ≈ 0.0014-0.0016. Effects of +0.0005 are not confirmable without about 8 seeds.

### 6.4 Stage-2 combination (pre-declared)
- If ≥ 2 arms promote and they act on different parts (e.g. muon_in + pbld, or hc4 + swiglu), stage 2 also runs the combination of all promoted arms (2 seeds), judged by the same unified rule.
- Combination order of flags is irrelevant; all are additive and flag-gated.

### 6.5 r6-owned arms (separate list)
- Flags, from `nn_x/r6/queue_morning.sh`:
  - `a4u --retr_uniq`;
  - `wd01 --wd 0.1` (10 ep: `--wd 0.05`);
  - `cset --cretr_m 8 --set_sim_bias`;
  - `typed --typed_cross`;
  - `adrop0 --attn_drop 0`;
  - `a4w --retr_m 32 --retr_prod`.
- r6's design, tests and acceptance are in `reports/r6_design.md`.
- If r6 has **not** launched them by the time r7 launches (check `nn_x/r6/logs/RUNNING_r6.txt` and `nn_x/r6/logs/queue/screen/`), port the r6 diff (model.py +144, train.py +23, data.py `hidx`; all flag-gated) into nn_x/r7 and screen them against r7's fresh baseline.
- **Do not edit nn_x/r6.** If r6 runs them itself, read its `r6_eval_screen.txt` and do not duplicate.

---------------------------------------------------------------------------------------------------------------------

## 7. Not selected

| item | reason |
|---|---|
| RMSNorm swap | same function class at d = 128; no speed gain here |
| DyT | Derf ≥ DyT in every row of its authors' comparison [V]; one normalisation-free arm is enough |
| nGPT | conflicts with Muon + decoupled wd and with the scale growth the model relies on (§2.4); QK-norm part closed |
| mHC | the instability it fixes (composite gain ~3000× at 27B) cannot occur over 8 sublayers; doubly-stochastic mixing forbids amplification, which the model visibly uses (c_ln 3.2). Fallback only if hc4 spikes |
| LayerScale / ReZero / SkipInit | ⊂ DWA (per-update scalars); ReZero-style init used in hl2 |
| value residual | both cross blocks read one context; ctx already carries h through the residual |
| GeGLU (separate arm) | ≈ SwiGLU in the source; one GLU arm |
| ff ratio 4 / 1 | FFNs least-used [D]; capacity flat (d192, deep, d256) |
| heads 8 / 2 | 8 closed; 2 has no mechanism and halves the sim-bias heads; diff2 covers "more query maps" |
| attention sinks / registers / SSMax / QK-norm | no sink to fix [D]; SSMax and QK-norm closed |
| robust scaling + smooth clip | no headroom [D §2.1] |
| TabM, PLE, DCNv2, MoE | closed / weak on LTR [R] |
| stochastic depth, dropout 0.2, history dropout, aux-anneal target, cooldown_frac | horizon-dependent: continuation harness (§4.2) |
| warmup, AdamW betas, clip, Muon momentum alone | no instability; momentum is a de-noising knob (r6 §5); scaled inside batch2s only |
| scalar-lr multiplier for sim-bias / gains | would exploit the 2-ep scale-growth gap (§2.4); a pure horizon artefact |

---------------------------------------------------------------------------------------------------------------------

## 8. Caveats
- **Single checkpoint.** The [D] diagnostics use one F7 member and 12 val batches (335K candidates), in eval mode on CPU fp32. Treat the branch ratios as ±10%.
- **Effect priors.** The effect sizes of the external sources come from 85M-27B-parameter models or LLM loss. Every expected range in §5 is my judgement for a 1M-parameter ranker, not a measurement.
- **DIFF V2** details come from the authors' blog mirror (preliminary, no paper) [S]. diff2 follows that form; V1 with a per-head norm is the alternative if V2 is unstable.
- **The PBLD frequency std** of 0.1 assumes inputs at robust-scaled ±3 magnitudes. Ours are z-scores clamped at ±6, about 1-2× larger, so the effective frequency is a little higher than RealMLP's. Acceptable; noted.
- **ftok group indices** follow `dense_features` order with the cvnd_pre2s xt layout:
  - the 63 base features;
  - then 31 kept xt columns (relonly pre2_fe_s / pre2_new_s / pre2_hmax32 dropped);
  - then 5 × [pct, z] rel columns, in XT_REL order.
  - Assert n_dense = 104 and print the group names at build time.
- **batch2s** changes batch composition, so seed pairing is weaker. Its SDE rule is a small-lr approximation; Muon is not Adam. The rule is a declared choice, not a theorem for Muon.
- **Promotion is not a ranking.** The final value of any promoted arm is decided at the 10-epoch stage and at ensemble level (stage 3, the lead).

---------------------------------------------------------------------------------------------------------------------

## 9. Sources
- DyT: https://arxiv.org/abs/2503.10622, https://arxiv.org/html/2503.10622v2 [V]
- Derf (Stronger Normalization-Free Transformers, CVPR 2026): https://arxiv.org/abs/2512.10938, https://arxiv.org/html/2512.10938v1 [V]
- Peri-LN: https://arxiv.org/abs/2502.02732, https://arxiv.org/html/2502.02732v3 [V]
- DeepNet / DeepNorm: https://arxiv.org/abs/2203.00555 [K]
- Hyper-Connections: https://arxiv.org/abs/2409.19606, https://arxiv.org/html/2409.19606v3 [V]
- mHC: https://arxiv.org/abs/2512.24880, https://arxiv.org/html/2512.24880v2 [V]
- DenseFormer: https://arxiv.org/abs/2402.02622, https://ar5iv.labs.arxiv.org/html/2402.02622 [V]
- Value residual (ResFormer): https://arxiv.org/html/2410.17897 [R: nnx_research_llm_arch]
- nGPT: https://arxiv.org/html/2410.01131 [R]
- GLU variants: https://arxiv.org/abs/2002.05202 [R]
- Differential Transformer: https://arxiv.org/html/2410.05258 [R]; DIFF V2 blog mirror: https://chanhi2000.github.io/bookshelf/huggingface.co/diff-attn-v2.html [S]
- RealMLP: https://arxiv.org/abs/2407.04491, https://ar5iv.labs.arxiv.org/html/2407.04491 [V]
- Numerical embeddings (PLR / PLE): https://arxiv.org/abs/2203.05556 [R]
- FT-Transformer: https://arxiv.org/abs/2106.11959 [R]
- TabM, TabR, ModernNCA: [R: nnx_research_tabular_dl]
- Muon (which parameters): https://kellerjordan.github.io/posts/muon/ [V]
- Muon at large batch: https://arxiv.org/abs/2505.02222 [S]; Muon critical batch size: https://arxiv.org/abs/2507.01598 [S]
- SDE square-root scaling for adaptive optimisers: Malladi et al., NeurIPS 2022 [K]
- Project: reports/r6_recap.md, r6_design.md, r6_external.md, nnx_VERDICT.md, nnx_design.md, COVERAGE_MAP.md [R]; nn/model.py, nn/train.py, nn/data.py [V]
