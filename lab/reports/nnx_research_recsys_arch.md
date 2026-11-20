# Industrial ranking architectures 2024-2026: what transfers to our target-attention listwise ranker

Date: 2026-11-11. Author: research agent (no GPU experiments). Scope: Meta HSTU follow-ups (ULTRA-HSTU, FuXi-α, LiGR), Wukong, InterFormer, DHEN;
ByteDance RankMixer, TokenMixer-Large, LONGER, OneTrans; Meituan MTGR; Kuaishou OneRec, TWIN-V2, QARM, HoME; Pinterest TransAct V2; LinkedIn LiRank, LiGR;
Google HiFormer, DCNv2 successors (GDCN, FCN/"DCNv3"); MMoE/PLE and sparse MoE; target-attention variants (DIN, TWIN, SIM, ETA, SDIM); feature tokenization;
scaling behaviour. One LLM-side reference (Qwen gated attention) is included because three recsys papers use the same mechanism.

Tags: **[V]** read in the primary source this session (full PDF text through `pdftotext`; URL given); **[S]** search snippet only; **[K]** background knowledge, not re-checked.
"Our model" means `<repo>/nn/model.py` as of today (v2_feed_k9 recipe: d=128, 4 heads, 1 history layer, 2 cross blocks with sim bias,
2 set blocks, fp32 head, listwise CE + listwise production-order distillation). Prior reports already cover DIN, TWIN v1, TransAct v1, HSTU basics,
DCNv2, PLR/PLE numeric embeddings, TabM, RealMLP (`reports/deep_neural_rankers.md`). This report does not repeat them; it adds the 2024-2026 industrial work.
All "expected effect" numbers are my own estimates, not literature numbers.

---------------------------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

**Regime first.** Every 2024-2026 industrial paper reports its gains at 10^9-10^12 training samples with dense labels.
Our train set has 3.15M lists, 247M candidates and **about 5.45M positives (2.207% x 247M), i.e. about 6 positives per parameter** of the 0.86M-parameter model.
The papers' own scaling evidence says capacity only pays when data grows with it:
TokenMixer-Large needs 14 days of data to converge a 30M->90M model and 60 days for 500M->2.3B [V];
Wukong gains about 0.1% LogLoss per 4x compute on 146B samples [V];
LiGR: "most architecture changes we experimented with had little impact on the prediction AUC. The majority of performance appeared to be driven by features and model scale" [V].
Our own ablations agree: d=256, 3 blocks and 8 heads are all neutral.
So: **do not import capacity tricks** (sparse MoE, per-token FFN scaling, deeper stacks, token merge).
**Do import** three other kinds of trick:
(a) tricks that add **supervision** (dense auxiliary targets);
(b) tricks that add **information** the network cannot build from its inputs (learnable semantic-ID memory over the frozen embedding, code-match cross features);
(c) cheap **inductive-bias and stability** fixes (attention gating, separate parameters per token type).

Ranked transfer list (detail in §3, experiment plan in §4):

| # | Trick (source) | Change in our code | My expected val gain | Cost |
|---|---|---|---|---|
| P1 | **Gated target attention**: elementwise sigmoid gate on the SDPA output (Qwen gated attention [V]; HSTU U-gating [V]; LiGR gated residual [V]; MTGR Ṽ⊙U [V]), plus an optional **count-preserving SiLU channel** (HSTU [V]) | `Attn`: `o = o * 2σ(W_g·hx)` with zero init, so it is the identity at init and can warm-start from current checkpoints | +0.0005..+0.002 | 30 lines, +16K params per block |
| P2 | **Semantic IDs (RQ-k-means codes) of the content embedding as learnable embeddings**, shared by candidate and history tokens, plus **code-prefix match features** (QARM [V], LiGR SID prefix n-grams [V], OneRec RQ-Kmeans [V]) | offline 3-level residual k-means on the 128-d item vectors; hashed uni/bi/tri-gram code embeddings (zero-init projection, ID-dropout); 6-9 new dense features | +0.001..+0.003 | ~1 h of code + 30 min offline k-means |
| P3 | **History as supervision**: a TransAct-V2-style *next action loss* on (event_type 1, source 1) history events, which are almost the same action as our target; also a lighter *item-propensity head* (K4 variant) | aux contrastive loss on the history encoder (w 0.01-0.05), or a propensity MLP trained on history event types and fed to candidates | +0.0005..+0.002 (lite), more for the full variant | 1-3 h |
| P4 | **Mixed parameterization + group LayerNorm** for heterogeneous context tokens: the user token gets its own LN/KV/FFN (OneTrans [V], HiFormer [V], MTGR GLN [V]) | `Block`: split `lnc`, `kv`, `ff` for token 0 vs history tokens | +0.0..+0.001 | 40 lines |
| P5 | **LLM-hygiene bundle** as used by TokenMixer-Large/OneTrans: SwiGLU FFN, down-projection small init (ReZero-like), RMSNorm pre-norm, no biases before norms | `Block.ff`, init | +0.0..+0.0007; better stability for later lr increases | 20 lines |
| P6 | **Duplicate-aware attention** `α' = α + ln n` (TWIN-V2 [V]); 48% of history events fall on repeated items (EDA) | dedup history per (feed,item) and add `ln(count)` per key to the attention logits, or simply add a per-head learned `β·ln(n_item)` bias | +0.0..+0.001 | 30 lines (needs item ids on GPU) |
| P7 | **Semantic group tokenizer + RankMixer-style interaction** for the candidate stem (RankMixer [V], OneTrans group-wise tokenizer [V]) | 6-8 group MLP tokens from the 61 dense features + content token + category token -> 1 token-mixing/per-token-FFN block -> pool | ±0.001 | 60 lines |
| P8 | **PLE-lite for main vs production-distillation task** (MMoE/PLE [K], HoME expert-norm fixes [V]) | last block FFN -> shared + task-specific experts with per-task softmax gates | ±0.001 | 40 lines |
| P9 | **Gated cross layer in the head** (GDCN [V], LiRank Residual DCN [V]) | 2 GDCN layers on [token, normalized dense] before the head MLP | +0.0..+0.001 | 20 lines |

Not recommended here (reasons in §3.10): sparse MoE / per-token MoE scaling, a HSTU/LiGR-style generative (interleaved, causal) rewrite, LONGER token merge or learnable query compression,
SIM/ETA/SDIM/TWIN-V2 retrieval (GSU) stages, dropping hand-crafted features (MTGR shows −0.0136 AUC without cross features [V]), more depth or width.

---------------------------------------------------------------------------------------------------------------------------------------

## 1. Regime analysis: why most industrial gains will not show up at our scale

| Paper | Training data | Model | Scaling statement (verified) |
|---|---|---|---|
| RankMixer (ByteDance, 2025) [V] | Douyin, trillions of records per day, 2 weeks | 100M-1.1B dense | "model quality correlates primarily with the total number of parameters, and different scaling directions (depth L, width D, tokens T) yield almost identical performance"; final 100M config D=768, T=16, L=2 |
| TokenMixer-Large (ByteDance, Feb 2026) [V] | Douyin ads / e-commerce / live | up to 15B offline, 7B online | Table 4: 30M->90M converges in 14 days of data (+0.94% UAUC); 500M->2.3B gives +0.41% after 30 days but +0.70% after 60 days: "Larger models require more data for convergence" |
| Wukong (Meta, 2024) [V] | 146B samples, 720 features | up to 637B params | quality vs compute fit `y = −100 + 99.56·x^0.00071`; "approximately translating to a 0.1% improvement for every quadrupling of complexity" |
| InterFormer (Meta, 2024) [V] | 70B samples | 1-4 layers | 1->2 layers +0.13% NE, then +0.05% and +0.04% |
| ULTRA-HSTU (Meta, Feb 2026) [V] | 6B samples, histories 3,072-16,384 | 6-18 layers, d=512 | stacked cross-attention saturates at 9 layers; self-attention keeps improving with depth |
| MTGR (Meituan, 2025) [V] | 23.7B exposures | 3/5/15 layers, d 512-768 | monotone small->medium->large (CTR AUC 0.7631 -> 0.7645 -> 0.7661), with the lr lowered as the model grows (3e-4, 3e-4, 1e-4; the DLRM UserTower uses 8e-4) |
| LiGR (LinkedIn, 2025; arXiv v1, later withdrawn) [V] | Feed logs | 12+ layers | scale and features dominate; architecture changes mostly neutral apart from training stability |
| TransAct V2 (Pinterest, 2025) [V] | 6.9B instances | 2 layers, d=64, FFN 32, 1 head, seq 192 | chosen config; the extra size went into sequence length, not depth |
| **Ours** | 3.15M lists, 247M candidates, ~5.45M positives, 584M history events | 0.86M params | d=256 (3M params), 3 blocks, 8 heads: all neutral; 4 epochs worse than 2 |

Reading: labels, not FLOPs, are our binding constraint. Douyin "finish" labels are dense; ours are 2.2% positive. Two facts from our own results support this.
(i) The biggest train-stage gains so far are dense supervision (the listwise production-order distillation, which supervises all 247M candidates: +0.0024) and strong priors (the raw-cosine sim bias: +0.0024).
(ii) Capacity knobs are flat.
The transferable industrial ideas are therefore the ones that change the inductive bias, the supervision or the information content, not the parameter count.

Implication for MoE (user asked): sparse MoE is a **capacity-per-FLOP** device. RankMixer's motivation is "increase model capacity with a minimal increase in computational cost" [V].
TokenMixer-Large's per-token sparse MoE is "first enlarge, then sparse" [V]. Neither helps a model whose dense version does not gain from more capacity.
The only MoE flavour that might help us is **hard routing by a known discrete prior** (token type, category vertical, task), because that is a structural prior, not capacity.
TokenMixer-Large explains its per-token MoE's advantage over a standard MoE the same way: "equivalent to giving standard MoE a routing prior" [V].

---------------------------------------------------------------------------------------------------------------------------------------

## 2. Paper digest (formulas and verified numbers)

### 2.1 Feature-interaction backbones

**RankMixer** [V] https://arxiv.org/abs/2507.15551 (ByteDance, Douyin; deployed 1B dense)
- Tokenization: features are grouped by semantics, concatenated `e_input = [e_1;...;e_N]`, split into T chunks and projected: `x_i = Proj(e_input[d(i−1):di])`.
- Block: `S_{n−1} = LN(TokenMixing(X_{n−1}) + X_{n−1})`, `X_n = LN(PFFN(S_{n−1}) + S_{n−1})` (post-norm). Output = mean pool over tokens.
- Multi-head token mixing (parameter-free): split each token into H heads, `s_h = Concat(x_1^h, …, x_T^h)`, with H = T so the residual shapes match.
- Per-token FFN: `v_t = W2^t GELU(W1^t s_t + b1^t) + b2^t` (separate weights per token). `#Param ≈ 2kLTD²`, `FLOPs ≈ 4kLTD²`.
- Sparse MoE: ReLU routing `G_ij = ReLU(h(s_i))`, `v_i = Σ_j G_ij e_ij(s_i)`, `L = L_task + λ Σ_ij G_ij`.
  Dense-Training / Sparse-Inference (DTSI): two routers; L_reg is applied only to h_infer.
- Ablation (100M): w/o token mixing −0.50% AUC; per-token FFN -> shared FFN −0.31%; w/o skip −0.07%; w/o LN −0.05%.
  Replacing token mixing with self-attention among feature tokens gives −0.03% at +71.8% FLOPs. The authors argue inner products across heterogeneous feature spaces are hard to learn.
- At ~100M params vs DLRM-MLP: RankMixer +0.64% finish AUC; DCNv2 +0.13%, DHEN +0.18%, Wukong +0.29%, HiFormer +0.48%.

**TokenMixer-Large** [V] https://arxiv.org/abs/2602.06563 (ByteDance, Feb 2026)
- Fixes RankMixer's residual: "Mixing & Reverting" (mix tokens, per-token SwiGLU, then revert to the original token layout so residuals align): w/o it −0.27% AUC.
- Per-token SwiGLU `pSwiGLU(x) = FC_down(Swish(FC_gate x) ⊙ FC_up x)`. Per-token SwiGLU -> shared SwiGLU −0.21%; per-token SwiGLU -> per-token GELU FFN −0.10%.
- Pre-norm RMSNorm. Post-norm gives "+0.01% -> NaN"; sandwich norm −0.03%. Biases are removed (Llama-style).
- Interval residual (every 2-3 layers) plus an auxiliary loss on lower-layer logits: −0.04% without.
- Down-matrix small init: xavier std 0.01 for FC_down, ReZero-style: −0.03% without.
- Sparse per-token MoE: `y = α Σ_{i<k} g_i(x) Expert_i(x) + SharedExpert(x)`, with α ≈ 1/sparsity (best α=2 at 1:2 and α=4 at 1:4).
  The shared expert is −0.02% without; per-token MoE vs standard MoE at equal params is −0.10%.
- "Pure model" claim: at scale, the stacked blocks subsume DCN/LHUC-type fragment operators.

**OneTrans** [V] https://arxiv.org/abs/2510.26104 (ByteDance, WWW'26)
- One causal transformer over [S-tokens (behaviour sequences); NS-tokens (non-sequential feature tokens)]. NS-tokens come after S-tokens, so each NS token attends to the whole sequence (target attention).
- **Mixed parameterization**: `W_i^Ψ = W_S^Ψ` for all sequence tokens, and `W_{NS,i}^Ψ` token-specific for each NS token, Ψ ∈ {Q,K,V}; FFN likewise. Shared parameters instead: −0.15% CTR AUC, −0.29% UAUC.
- RMSNorm pre-norm "to align scales across token types". Full instead of causal attention: +0.00%. Without the pyramid: −0.05% AUC.
- Group-wise tokenizer vs auto-split: −0.10% (auto-split better). Timestamp-agnostic vs timestamp-aware sequence merge: −0.09%; without [SEP] tokens −0.13%.
- Table 2 (CTR AUC vs DCNv2+DIN): RankMixer+DIN +0.27%, RankMixer+Transformer +0.49%, OneTrans-S (6 layers, d=256) +1.13%, OneTrans-L +1.53%. Data: 29.1B impressions.

**HiFormer** [V] https://arxiv.org/abs/2311.05884 (Google Play)
- Heterogeneous attention: feature-specific Q/K/V projections per feature token (the first "token-specific parameters" result), a heterogeneous FFN per feature, and low-rank composite projections.
  Online engagement: HeteroAtt +2.33%, Hiformer +2.66%, DCN +2.20% over a 1-layer transformer baseline.

**Wukong** [V] https://arxiv.org/abs/2403.02545 (Meta)
- Stacked layers `X_{i+1} = LN(concat(FMB_i(X_i), LCB_i(X_i)) + X_i)` with
  `FMB(X) = reshape(MLP(LN(flatten(X Xᵀ Y))))` (optimized FM, Y ∈ R^{n×k}, computed as X(XᵀY) in O(nkd)) and `LCB(X) = W_L X`. Scaling statement as in §1.

**DHEN** [V] https://arxiv.org/abs/2203.11014 (Meta): each layer is an ensemble of interaction modules (AdvancedDLRM, self-attention, linear, DCN, conv); +0.27% NE over AdvancedDLRM.

**InterFormer** [V] https://arxiv.org/abs/2411.09852 (Meta)
- Interleaves an Interaction Arch (non-sequence features + sequence summary), a Sequence Arch (MHA with RoPE over [CLS; sequence]) and a Cross Arch.
- **Personalized FFN**: `W_PFFN = X_sum W ∈ R^{d×d}`, `S' = W_PFFN S`. This is a hypernetwork, so the user/context summary transforms every history token.
- Self-gating `Gating(X) = σ(X ⊙ MLP(X))`. Summaries: CLS + PMA (learnable-query pooling) + K most recent tokens.
- Ablations on public data: PMA tokens up to 0.004 AUC; gating up to 0.003. Industrial: 3-layer InterFormer +0.15% NE vs the internal SOTA at similar FLOPs.

**DCN family**
- GDCN [V] https://arxiv.org/abs/2311.04635: `c_{l+1} = c_0 ⊙ (W^c_l c_l + b_l) ⊙ σ(W^g_l c_l) + c_l` (cross term times an information gate).
- FCN ("DCNv3") [V] https://arxiv.org/abs/2407.13349: the Linear Cross Network keeps anchor x_1 (`x_{l+1} = x_1 ⊙ LCA(x_l) + x_l`).
  The Exponential Cross Network uses the current input as anchor (`x_{2^l} = x_{2^{l−1}} ⊙ LCA(x_{2^{l−1}}) + x_{2^{l−1}}`), so the interaction order doubles per layer.
  Tri-BCE: auxiliary losses on each sub-network with adaptive weights `w = max(0, L_sub − L)`.
- LiRank Residual DCN [V] https://arxiv.org/abs/2402.06859: an attention low-rank cross net in parallel with a low-rank cross net, plus skip connections.
  The Feed "contributions" replay table shows +1.26% for low-rank DCNv2, +2.15% for Residual DCN, +1.00% for "Dense Gating" (Gate-Net-style gates on hidden layers) and +1.66% for TransAct.
  The cumulative stack reaches +3.62%, and +4.14% with sparsely gated MMoE.

### 2.2 Sequence / target-attention side

**HSTU** [V] https://arxiv.org/abs/2402.17152 (Meta, ICML'24). Pointwise aggregated attention `A(X)V(X) = φ2(Q Kᵀ + rab^{p,t}) V`, φ2 = SiLU, no softmax, then `Y = f2(Norm(A V) ⊙ U)`.
Ranking NE: softmax without rab .5067; SiLU without rab .4980; SiLU with position+time rab .4937. Motivation: softmax hides "the number of prior data points related to target".

**ULTRA-HSTU** [V] https://arxiv.org/abs/2602.16986 (Meta, Feb 2026)
- Item and action embeddings are *added* into one token instead of interleaved, halving the sequence. We already do this: `h = Linear(e_h) + emb(type) + emb(source) + …`.
- Semi-local attention (local window K1 + global window K2) gives linear cost.
- **Mixture of Transducers**: separate HSTU modules for engagement events and consumption events "mitigates the signal competition that occurs when diverse input signals are constrained within a single module".
- Stacked self-attention beats stacked cross-attention as depth grows (at 6B samples and 3k-16k histories).

**FuXi-α** [V] https://arxiv.org/abs/2502.03036 (Huawei): adaptive multi-channel self-attention.
Semantic channel = QK attention; positional channel and temporal channel = attention weights that depend only on relative position / time difference (no Q/K); the three channels share V and are fused by a multi-stage FFN.
w/o AMS: ML-1M NDCG@10 0.1563 vs 0.1934 (a generative retrieval setting, so do not over-read it).

**LiGR** [V] https://arxiv.org/pdf/2502.03417v1 (LinkedIn, KDD'25; **the arXiv entry was later withdrawn**, so treat its numbers as indicative)
- Gated residual on both attention and FFN: `h_{j+1} = h_j + F(h_j) ⊙ σ(h_j W)`.
- Replacing LiGR layers with HSTU layers dropped long-dwell AUC 76.03 -> 75.87 at equal compute.
- Set-wise in-session attention: +0.50% click AUC, +0.20% long-dwell AUC.
- Semantic IDs (see 2.3).

**MTGR** [V] https://arxiv.org/abs/2505.18654 (Meituan, CIKM'25)
- All candidates of a user in one sample (= our feed formulation). HSTU-style blocks with **Group LayerNorm** (a separate LN per token domain: user features, sequence, real-time sequence, candidates).
- `Ṽ = SiLU(Kᵀ Q) M V / N`, `X = MLP(GroupLN(Ṽ ⊙ U)) + X`. Mask: static history visible to all, real-time events causal, **candidates see only themselves**. They aggregate candidates over a time window, so cross-candidate attention would leak there; our feeds are simultaneous, so set attention is legitimate.
- Ablation (CTR AUC / GAUC): full 0.7631 / 0.6826; w/o cross features 0.7495 / 0.6689; w/o GLN 0.7606 / 0.6809; w/o dynamic mask 0.7620 / 0.6810.

**LONGER** [V] https://arxiv.org/abs/2505.04421 (ByteDance, RecSys'25)
- Global tokens: target item, CLS, UID tokens prepended to the sequence; they "stabilize attention", citing StreamingLLM attention sinks.
- Token merge (concat K neighbours, optional InnerTrans): merge-8 (concat) improves AUC 0.85111 -> 0.85291 at fewer FLOPs, because it gives more parameters per merged token.
- Hybrid attention: the first layer is cross-causal with queries = global tokens + k sampled sequence tokens. Recent-100 0.85290, uniform-100 0.85183, **learnable-100 0.84946 (worst)**.
  Query count 50 -> 250: 0.85235 -> 0.85332, so 40% of the queries keep about 95% of the gain.

**TWIN-V2** [V] https://arxiv.org/abs/2407.16357 (Kuaishou, CIKM'24)
- Hierarchical clustering of life-long histories. Cluster-aware target attention keeps TWIN's logit `α = (K_h W_h)(q W_q)ᵀ/√d_k + (K_c W_c)β` (behaviour features + scalar cross-feature bias) and adds the cluster size:
  **`α' = α + ln n`** ("the cluster with more items should be considered more important").
- `softmax(α + ln n)_j = n_j e^{α_j} / Σ_i n_i e^{α_i}` is exactly attention over the un-deduplicated list with identical items merged.

**SIM / ETA / SDIM** (retrieval stages for 10^3-10^5 histories)
- SIM [V] https://arxiv.org/abs/2006.05639: GSU hard-search (same category as the candidate) or soft-search (MIPS on embeddings), then ESU target attention over the top-K.
- ETA [V] https://arxiv.org/abs/2108.04468: SimHash fingerprints and Hamming distance select the top-k end-to-end.
- SDIM [V] https://arxiv.org/abs/2205.10249: sums behaviours whose LSH signature collides with the target. The expected weight is `E[p̃_j] = (1 − arccos(qᵀ s_j)/π)^τ`, a fixed angular kernel on the raw embeddings.
- Our sim bias is the learned, per-head version of the same idea, `a_h · cos(e_c, e_h)` added to the logits, which explains why it helps (+0.0024). With 256 events no GSU stage is needed.

**TransAct V2** [V] https://arxiv.org/abs/2506.02267 (Pinterest, 2025)
- Candidate-anchored NN search over the lifelong (10^4), real-time and impression sequences, plus the r most recent actions.
- Early fusion: `F = CONCAT(E_PinSage(S_all), e_c) + E_act + E_surf + E_pos`; 2-layer causal transformer.
- **Next Action Loss (NAL)**: `L(u_t, p_{t+1}) = −log( e^{⟨u_t,p_{t+1}⟩} / (e^{⟨u_t,p_{t+1}⟩} + Σ_n e^{⟨u_t,n⟩}) )`, `L = L_CE + w_NAL · L_NAL`, w_NAL = 0.01 (tuned), sampled softmax beats CE.
  **Impression-based negatives beat in-batch negatives** (HIT@3/repin +1.10% vs +0.63%).
- Table 2 (HIT@3/repin vs W&D): TransAct +7.74%; +NAL in-batch +8.41%; +NAL impressions +8.92%; full V2 +13.31%. Online: NAL_imp alone −6.26% hides; full V2 +6.35% repins.

### 2.3 Semantic IDs (learnable discrete codes on top of frozen content embeddings)

- **QARM** [V] https://arxiv.org/abs/2411.11739 (Kuaishou)
  - Item alignment of the MLLM embedding, then **residual k-means codes**: `R1 = Kmeans(M, N)`, `M1 = M − R1[r1]`, `R2 = Kmeans(M1, N)`, …, plus top-k VQ codes.
  - The codes are used as **learnable ID features** on the item side, in user sequences, and as **target-aware cross features** ("the latest first one-code matching item sequence, the latest two-code matching item sequence").
  - Ads CTR: aligned frozen representation alone +0.02% AUC; +VQ code +0.11%; +RQ code +0.10%; +VQ & RQ +0.18% AUC / +0.29% UAUC. Shopping CTCVR: +0.35% AUC / +1.00% UAUC.
  - Takeaway: **making the frozen embedding learnable through codes matters more than improving the frozen vector.**
- **LiGR SIDs** [V]: RQ-VAE, 3 codebooks of 1000.
  - Prefix encoding: uni-gram (c1), bi-gram (c1,c2) and tri-gram (c1,c2,c3), "hashed into an embedding bag of size 10^6", 32-d each, concatenated.
  - Add SID: +0.26% long-dwell AUC, +0.22% contributions AUC. Replacing post ID with SID: +0.4%. SID only: −6.5%.
- **OneRec technical report** [V] https://arxiv.org/abs/2506.13695: RQ-Kmeans over 3 layers (`s_i^l = argmin_k ‖R_i^l − c_k^l‖`, `R^{l+1} = R^l − c_{s^l}`). It reports better reconstruction, codebook utilisation and balance than RQ-VAE.
  It notes that codes built from content-only features are suboptimal, so collaborative signals must be fused first. Our item vectors already come from Avito's interaction-trained two-tower, so they are already collaborative.

### 2.4 Multi-task / MoE

- **MMoE** [K]: `y_k = h_k(Σ_i g_k(x)_i f_i(x))`, `g_k = softmax(W_k x)`. **PLE** [K]: per task, experts = shared experts ∪ task-specific experts, with a task gate over both; stacked progressively.
- **HoME** [V] https://arxiv.org/abs/2408.05430 (Kuaishou) lists three MMoE pathologies and their fixes:
  - **expert collapse**: one expert outputs 100x smaller and sparser activations (ReLU >90% zeros), yet gets gate weight >0.98. Fix: **normalise each expert's output and use Swish**.
  - **expert degradation**: shared experts get monopolised by one task. Fix: hierarchy masks.
  - **underfitting of sparse tasks**: sparse tasks ignore their own experts. Fix: feature-gate and self-gate.
  - Average result: +0.52% GAUC offline.
- RankMixer SMoE and TokenMixer-Large S-P MoE: see 2.1.

### 2.5 LLM-side mechanism used by three recsys papers

- **Gated attention** [V] https://arxiv.org/abs/2505.06708 (Qwen team).
  - `Y' = SDPA(Q,K,V) ⊙ σ(X W_θ)`, where X is the pre-norm hidden state of the query token; head-specific, elementwise ("G1").
  - 15B-A2B MoE, 400B tokens: PPL 6.026 -> 5.761, MMLU 58.79 -> 60.82, versus +4 experts 5.964.
  - Head-wise gating (1.6M params) gets most of it (5.792). The attention sink (share of attention on the first token) falls from 46.7% to 4.8%.
  - Two mechanisms: non-linearity between W_V and W_O, and query-dependent sparsity.
- Recsys analogues: HSTU `Norm(AV) ⊙ U(X)`, MTGR `Ṽ ⊙ U`, LiGR `F(h) ⊙ σ(hW)`. Qwen3-Next ships gated attention [K].

### 2.6 Overfitting and embeddings

- **GPSD** [V] https://arxiv.org/abs/2506.03699 (Alibaba): discriminative recsys models show "one-epoch overfitting" (a sudden drop at the epoch boundary) and "within-one-epoch overfitting", both driven by sparse (embedding) parameters.
  Fix: generative pretraining, then freeze the sparse parameters during discriminative training. **Relevant if we add learnable code or ID tables (P2).**
- **Embedding collapse** [V] https://arxiv.org/abs/2310.04400: embedding matrices occupy a low-dimensional subspace and enlarging them does not help. Fix: multi-embedding sets with set-specific interaction modules. Low relevance for us (few ID tables).

---------------------------------------------------------------------------------------------------------------------------------------

## 3. Concrete transfers to `nn/model.py`

Notation: `hx` = LN(candidate token) (query input), `ctx` = encoded [user; history] (K' = 1+K keys), `S` = raw cosine bias [B,M,K'], h = 4 heads, d_h = 32.

### 3.1 P1: gated target attention (+ optional count channel)

**Why it fits us.** Softmax target attention must put total weight 1 somewhere.
For a candidate with no related history (user0, short histories, a new vertical), the mass goes to the user token or spreads over weakly similar events, and the residual stream receives a non-informative average.
A query-dependent sigmoid gate lets each head switch off: the "sparse gating" effect, which the Qwen paper measures as the removal of the attention sink.
A count-preserving channel lets "5 contact events on near-duplicates" differ from "1 view". HSTU's stated reason for SiLU is exactly this intensity signal.
Our hand features (`ccnt`, smax, lse) cover intensity only partly, and only through the dense MLP, not per head inside attention.

**Math.**
- Gate (G1, elementwise, head-specific): `o_h = softmax(q_h K_hᵀ/√d_h + a_h S + m) V_h`, `g = 2·σ(W_g hx + b_g)` with `W_g = 0, b_g = 0`, `o' = o ⊙ g`, then `W_o`.
  The 2σ form (PEPNet/LHUC-style γ=2 [K]) is exactly 1 at init, so the gate can be added to an existing checkpoint or to stage 2 without a jump.
- Count channel (one extra head or a parallel branch per cross block):
  `c_cnt = W_o' Σ_k SiLU(q'·k'_k/√d_h + a' S_ck) v'_k / 256`, with the fixed normaliser 256 = max history (not H, so the magnitude carries the count). `W_o'` is zero-initialised.

**Code sketch** (in `Attn.__init__` / `forward`):
```python
self.gate = nn.Linear(d, d); nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
...
o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=...)          # [B,h,N,dh]
g = 2 * torch.sigmoid(self.gate(x)).view(B, N, h, D // h).transpose(1, 2)      # x = hx (query input)
o = o * g
```
Apply it in the cross blocks first; the history encoder and set blocks are a second ablation. Cost: +16.5K params per Attn, negligible FLOPs.
Head-wise variant: `nn.Linear(d, h)`, broadcast over d_h.

**Risks.** The gate saturating at 0 for some heads (watch the mean gate per head). The count channel in bf16: the sum over 256 keys of SiLU outputs is fine in bf16, but compute it with the explicit `[B,h,M,K']` score tensor (~67 MB at 32K candidates).
**Expected:** +0.0005..+0.002 val; mostly visible on short-history and user0 feeds (log NDCG by H bucket).

### 3.2 P2: semantic-ID codes over the content embedding

**Why it fits us.**
1. Today the model sees the item only through linear maps of a frozen vector (`c_lin([e_c, e_c⊙u])`, `h_lin(e_h)`) plus dot products. A code table adds a cheap, high-capacity, piecewise-constant memory over embedding space ("this region of item space is popular / converts").
2. QARM shows learnable codes beat a frozen aligned embedding by 0.1-0.3% AUC. LiGR shows +0.2% AUC on top of everything else.
3. Codes exist for every item, including the 39.3% of test rows whose item never appeared in train, unlike the planned hashed item-ID residual (N4).
4. The same table on candidate and history tokens lets attention learn code-to-code co-occurrence, a learned, cluster-level co-visitation signal. The report's open item says "a train-stage version of the ALS/co-visitation signals is the most promising next step" (fine-tune gain +0.0037). This is that, end to end.

**Offline (≈30 min on GPU N).**
- Dequantise and L2-normalise a 4-8M sample of train item vectors.
- Residual k-means, 3 levels × N=256 (or 1024/256/256), with OneRec/QARM RQ-Kmeans: level 1 on the normalised vectors, then on the residuals.
- Assign codes to every row of the train/val/test item tables in chunks (argmin over 256 centroids per level: a [chunk,128]×[128,256] matmul).
- Store int16 `[n_items, 3]` next to each split's table, indexed by `emb_idx` like the table itself.
- Check the balance per code (OneRec reports RQ-Kmeans is better balanced than RQ-VAE) and the share of val/test items whose (c1,c2) prefix occurs in train (should be ~100%).

**Model.**
- Prefix n-gram embeddings (LiGR): `u1 = E1[c1]` (256×32), `u2 = E2[hash(c1,c2) mod 2^16]`, `u3 = E3[hash(c1,c2,c3) mod 2^18]`, `z = W_z [u1;u2;u3]` (96->d, **zero-init**).
- Candidate: `c += z_c` before `c_ln`. History: `h += z_h` before `h_ln`. Use the same E1..E3 for both, so the code embedding means the same thing on both sides.
- **SID-dropout**: zero `z` for 30-50% of tokens in training (as `_id_vec` already does). This keeps the model from leaning on memorised cells, which matters because the cells partly memorise production-position bias.
- Optimiser: separate param group for E1..E3 with wd=0. If the holdout shows a step down at the epoch boundary (GPSD "one-epoch overfitting"), freeze E1..E3 for epoch 2 or halve their lr.

**Code-match dense features** (QARM target-aware cross features = SIM hard search with finer keys than our 54 categories).
For level ℓ ∈ {1,2,3} let `P_ℓ[m,k] = 1[prefix_ℓ(c_m) = prefix_ℓ(h_k)] ∧ hmask`. Add:
- `log1p(Σ_k P_ℓ)` for ℓ = 1, 2, 3;
- `log1p(Σ_k P_2 ∧ (type=1 ∧ src=1))`, the same feed-open action as the target (EDA: the target ≈ type-1/source-1 open);
- `1/(1 + first k with P_2)`;
- within-feed z-score of the level-2 count.

That is 6-9 columns in `dense_features` (`N_DENSE` += 9; `f_mean/f_std` need recomputing). Optional attention bias: `β_{h,ℓ}·P_ℓ` per head. Note that the same-category term inside `sim_bias=2` was neutral, so try the features first.

**Expected:** +0.001..+0.003 train-only. Gains may shrink after fine-tuning, which already injects collaborative features.
**Risks:** overfitting to production-biased cells (mitigated by SID-dropout, holdout selection and the domain token); memory of about 8.4M extra embedding params with Adam states (fine).

### 3.3 P3: history as supervision (dense auxiliary targets)

**Why it fits us.** EDA (`eda_history_leak.md`) shows:
- the target is "≥1 interaction started from this feed", mostly a type-1/source-1 card open;
- 92% of a train feed's positives reappear as such events in the user's next history;
- type-1/source-1 is 28% of all val history events (707K of 2.5M), i.e. on the order of 50 per history.

The 584M history events therefore contain tens of millions of past positives *of the same action*, from impressions we never see. That includes val/test users' own histories: transductive and label-free, the same kind of input the fine-tune branch already uses (history popularity).
This is the most information-rich unused signal. In TransAct V2 the next-action loss with impression negatives added +1.18 pp HIT@3 on top of TransAct's +7.74 pp.

Three variants, from cheap to expensive:

**P3a: item-propensity head (K4 variant, cheapest).**
- An MLP `π(e, cat) -> logits over 10 event types` (or binary "type1∧src1 vs other"), trained with CE on all valid history tokens in the batch, weight λ = 0.05.
- Its output on candidates, `log π(type1 | e_c, cat_c)` and the full log-probability vector, enters the candidate token through a zero-init `Linear(10, d)` and the wide term.
- This is a content-based generalisation of the strongest popularity feature found so far ("train history counts, esp. event_type 1: +0.0032..0.0035", EDA). It covers unseen items because it is a function of the embedding.

**P3b: NAL-lite on the history encoder (TransAct V2).**
- Run a second, **causal**, history-only pass. In newest-first order, token k may attend only to keys with index > k. Drop the user token from this pass, since the request-time user embedding already contains the newer events.
- For each of the ≤8 newest type-1/source-1 events t in the first 64 positions: `u_t = W_n · enc(older than t)` at position t+1, and `L = −log softmax(⟨u_t, e_t⟩ / τ)` against negatives:
  - (i) in-batch items of other users;
  - (ii) **this feed's non-positive candidates** (impression negatives, which TransAct V2 found clearly better).
- `L_total += w·L_NAL` with w ∈ {0.01, 0.05}. Cost: about 1 more history-layer pass (cheap next to the cross/set blocks).

**P3c: history-derived pseudo-lists through the full ranker (most expensive, highest ceiling).**
- For a random 25% of feeds, shift the history to just before a type-1/source-1 event t (gather indices k>t, re-index recency from 0).
- Score a pseudo-list = {item t} ∪ the feed's candidates with the same ranker, with the f1..f5-derived dense columns **constant for all items in the pseudo-list**. History items have no f-features, and leaving them would leak which item is the positive.
- Add a listwise CE with weight ~0.2. It trains the content+history pathway on extra positives drawn from the test-like population.

**Expected:** P3a +0.0005..+0.0015; P3b +0.0..+0.001; P3c unknown, potentially the largest. Start with P3a.
**Risks:** P3b/P3c: leakage through the user embedding (exclude it) and through recency features (re-index). Distribution mismatch: past feed-opens come from production feeds, so they carry popularity bias; weight them below the main loss.

### 3.4 P4: mixed parameterization and group LN for [user; history]

Our context sequence mixes one user token (a projection of Avito's user tower output) with 256 event tokens. All of them share `lnc`, the K/V projections and the history-encoder FFN.
OneTrans (token-specific QKV/FFN for non-sequential tokens; −0.15% AUC if shared), HiFormer (feature-specific projections) and MTGR (Group LN; −0.0025 AUC without) all find that heterogeneous token types want their own normalisation and projections.

Change:
- in `Block.forward`, when `ctx` is given, `c = cat([lnc_u(ctx[:, :1]), lnc_h(ctx[:, 1:])], 1)`;
- in `Attn`, `kv_u` for token 0;
- in the history encoder, separate `ln1/ln2/ff` for token 0.

+~100K params. Optionally also give the **candidate token its own LN per traffic domain** (production vs exploration) as a group-LN by domain. Cheap, and it reuses the existing `dom` vector.
**Expected:** 0..+0.001.

### 3.5 P5: LLM/industrial "hygiene" bundle (one run)

TokenMixer-Large and OneTrans converge on the Llama recipe, each with small measured gains at their scale:
- SwiGLU FFN: `FFN(x) = W_2(Swish(W_g x) ⊙ W_1 x)` with hidden 2/3·2d ≈ 170 to keep params equal; per-token SwiGLU vs GELU-FFN was +0.10% AUC.
- Down-projection small init: std ×0.01 on `W_2` and on `attn.o`, ReZero-like; +0.03%.
- RMSNorm pre-norm; no biases in linears before norms.

Expected 0..+0.0007 at our scale. The more useful by-product is the stability to raise lr from 1e-3 to 2e-3, or to shorten warmup. Gated attention (P1) also "tolerates larger learning rates" [V].

### 3.6 P6: duplicate-aware attention (TWIN-V2 `ln n`)

48% of history events fall on items with more than one event in the same history (EDA; types 5/6/8 average 6-7 events per item).
Softmax over duplicates already up-weights repeated items: n copies with equal logits get n× weight. But they arrive as copies with different type/recency embeddings, so heads waste capacity.

Two options:
- (a) Keep the duplicates but add a per-head learned `β_h · ln(n_item(k))` to the logits (cheap: needs `hitem` on the GPU; `load_item_ids` exists).
- (b) Deduplicate to one token per item with aggregated type counts and add `ln n` (TWIN-V2 exactly). This also frees sequence budget, but changes the inputs a lot.

Expected 0..+0.001. Low priority.

### 3.7 P7: semantic group tokenizer + RankMixer interaction in the candidate stem

Today: `c = LN(c_lin([e_c,e_c⊙u]) + cat + c_mlp(61 dense) + miss + dom)`. One MLP mixes all 61 standardised features, which is RankMixer's "single token degrades into a DNN" case.

Group tokenizer (OneTrans group-wise / RankMixer semantic) with T=8 tokens:
- G1 f1..f5 raw/z/pct (15);
- G2 content-user (miss, norm, cos_u, dot_u) (4);
- G3 history similarity (6 + 10 per type);
- G4 category affinity (14);
- G5 feed context (log M, log H, user0, fcs);
- G6 within-feed relatives (10);
- G7 = `c_lin([e_c, e_c⊙u])`;
- G8 = category embedding.

Each group gets its own `MLP_i -> d`. Then one block of multi-head token mixing (H=T=8, d_h=16) plus per-token FFN (k=2), mean-pooled, which replaces `c_mlp`.
Variant: keep G7 (content-only) as a **separate query token** for the cross blocks, so one query is not mixed with the dense features. Our sim bias already gives raw content retrieval, so this is second order.
Expected ±0.001; it also overlaps with the PLR/PLE numeric-embedding proposal in `deep_neural_rankers.md` §5. Test one of the two, not both.

### 3.8 P8: PLE-lite for the main and production-distillation tasks

The aux task (production order, weight 0.5, `aux_head` on the same token `c`) is useful but conflicts in part with the main task (exploration clicks).
In the last cross/set block, replace `ff` with 3 experts {E_s, E_main, E_aux} (each d->2d->d, with LayerNorm on the expert output + Swish, per HoME):
- `c_main = c + g_m·[E_s(c), E_main(c)]`, `c_aux = c + g_a·[E_s(c), E_aux(c)]`, `g = softmax(W c)`;
- head on `c_main`, aux head on `c_aux`.

Expected ±0.001. A cheaper control: an aux head reading the block-1 output (deep supervision) instead of the final token.

### 3.9 P9: gated cross layers in the head (GDCN / LiRank)

`x0 = [c; X_std]` (128+61). Two layers `x_{l+1} = x0 ⊙ (W_l x_l + b_l) ⊙ σ(W^g_l x_l) + x_l` with rank-32 `W_l = U V`, then the existing LN->Linear->GELU->Linear head.
LiRank and the prior report estimate ≤+0.002. TokenMixer-Large says such operators get subsumed at scale, but we are not at scale. Expected 0..+0.001.

### 3.10 Rejected or deprioritised, with reasons

- **Sparse MoE / per-token MoE (RankMixer, TokenMixer-Large):** a capacity-per-FLOP device; our dense capacity curve is flat (§1). Revisit only if P2 or P3 make the model capacity-bound (if d=256 starts to help).
- **HSTU/LiGR/OneTrans unified causal stack with candidates as tokens (early fusion):** costs M×K or (K+M)² tokens.
  - Our sim-bias cross-attention is the efficient version of target attention (OneTrans NS tokens attend to S tokens).
  - ULTRA-HSTU's self-attention-over-cross-attention advantage appears with depth at 6B samples, and our extra depth is neutral.
  - The richer per-type bias variant (`sim_bias=2`) was neutral, so pairwise feature interactions inside attention are not the bottleneck.
- **LONGER token merge / learnable queries:** for 2k-10k histories. LONGER itself finds learnable queries worst. Our 256 events are cheap.
- **SIM/ETA/SDIM/TWIN-V2 GSU retrieval:** histories ≤256 need no retrieval stage. SDIM's fixed angular kernel is a weaker form of our sim bias.
- **Removing hand-crafted features (LiGR/HSTU philosophy):** MTGR's ablation (−0.0136 AUC without cross features) and our own feature results say keep them.
- **FuXi-α positional/temporal channels:** we have no timestamps, and the recency embedding is already in the keys. A recency-only attention channel equals a learned recency-weighted user summary, which the user token and history encoder already provide.
- **Set-attention changes:** set attention was neutral for us. LiGR's in-session attention was +0.2-0.5% AUC there, but they use it for diversity. Keep ours; do not invest.
- **Interval residual / auxiliary loss on lower layers (TokenMixer-Large):** made for deep stacks; −0.04% without, even at their depth. Our 2-block model does not need it.

---------------------------------------------------------------------------------------------------------------------------------------

## 4. Experiment plan (GPU N only, ≤2 concurrent runs, 2 paired seeds each)

Baseline = the v2_feed_k9 command from the task brief (`--no_pos --aux_pos 0.5 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --epochs 2 --loss_norm feed --aux_mode listwise --aux_tau 10`), seeds 21 and 23.
Judge on paired val differences **and** holdout differences; per-run noise SE ≈ 0.0005-0.0013 paired; accept a trick only if both seeds are positive and the mean is ≥ +0.0007.

| Round | Runs (each ×2 seeds) | Wall time (2 parallel) |
|---|---|---|
| R1 | P1 gate (cross blocks only); P1 gate + count channel | ~45 min |
| R2 | P2 (codes, 3 levels × 256, SID-dropout 0.3) with code-match features; P2 features only (no code embeddings) | ~1 h incl. offline k-means |
| R3 | P3a propensity head (λ 0.05); P4 mixed param + group LN | ~45 min |
| R4 | P5 hygiene bundle; P6 duplicate bias (if R1-R3 leave time) | ~45 min |
| R5 | Combine the winners, then stage 2 (`--only_random 0.5 --lr 2e-4 --epochs 2 --keep last`), then the n2full fine-tune of 1-2 members, then add them to the ensemble | ~1.5 h |

Logging to add per run: NDCG by H bucket (0, 1-32, 33-128, 129-256) and for user0. P1 and P2 should move short-history feeds more. Also log the mean gate per head (P1) and the holdout curve around the epoch boundary (P2, GPSD warning).

Diversity note: these variants change the architecture or the inputs, so even neutral ones are better ensemble members than new seeds (seed ensembles saturated at +0.0005).

---------------------------------------------------------------------------------------------------------------------------------------

## 5. Sources (read this session unless tagged)

- RankMixer, arXiv 2507.15551 [V]
- TokenMixer-Large, arXiv 2602.06563 [V]
- OneTrans, arXiv 2510.26104 [V]
- LONGER, arXiv 2505.04421 [V]
- MTGR, arXiv 2505.18654 [V]
- HSTU, arXiv 2402.17152 [V]
- ULTRA-HSTU ("Bending the Scaling Law Curve…"), arXiv 2602.16986 [V]
- FuXi-α, arXiv 2502.03036 [V]
- LiGR, arXiv 2502.03417v1 [V] (entry withdrawn on arXiv; v1 PDF read)
- LiRank, arXiv 2402.06859 [V]
- Wukong, arXiv 2403.02545 [V]
- InterFormer, arXiv 2411.09852 [V]
- DHEN, arXiv 2203.11014 [V]
- HiFormer, arXiv 2311.05884 [V]
- GDCN, arXiv 2311.04635 [V]
- FCN/DCNv3, arXiv 2407.13349 [V]
- TWIN-V2, arXiv 2407.16357 [V]
- QARM, arXiv 2411.11739 [V]
- OneRec Technical Report, arXiv 2506.13695 [V]
- TransAct V2, arXiv 2506.02267 [V]
- HoME, arXiv 2408.05430 [V]
- SIM, arXiv 2006.05639 [V]
- ETA, arXiv 2108.04468 [V]
- SDIM, arXiv 2205.10249 [V]
- GPSD, arXiv 2506.03699 [V]
- Embedding collapse, arXiv 2310.04400 [V]
- Gated Attention for LLMs (Qwen), arXiv 2505.06708 [V]
- MMoE (KDD'18), PLE (RecSys'20), PEPNet (KDD'23), Qwen3-Next gated attention [K]
- VISTA (Meta, ICLR'26) and TRM (ByteDance, arXiv 2601.22694) [S], not used for recommendations
- Project files: `nn/model.py`, `nn/train.py`, `nn/data.py`, `reports/nn_report.md`, `reports/eda_shift_overlap.md`, `reports/eda_history_leak.md`, `reports/deep_neural_rankers.md`
