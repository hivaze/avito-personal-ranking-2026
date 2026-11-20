# Which recent LLM architecture tricks transfer to the Avito feed ranker (2024-2026)

Status: research only (no GPU runs), 2026-11-11. Written for the `nn/` ranker (`<repo>/nn/model.py`).
Evidence tags: **[V]** = primary source read this session (URL in the Sources list); **[S]** = search snippet or secondary write-up only; **[K]** = background knowledge, not re-verified; **[inf]** = my own inference or arithmetic.

## TL;DR

**Our regime is very different from the regime these tricks were designed for.** The ranker has 0.86M parameters and about 5 attention sublayers in total (1 history self-attention layer, 2 target-attention blocks, 2 set-attention blocks). Key sets hold at most 257 tokens (user + history) and query sets at most 300 candidates. Total training compute is about 3e15 FLOPs [inf: about 1.8 MFLOP forward per candidate, x3 for backward, x247M candidates, x2 epochs]. Width 256, 3 blocks and 8 heads were all neutral, so capacity and depth are not the bottleneck.

Most 2025-2026 LLM innovations target problems we do not have:
* long-context cost: linear, sparse and latent attention (Gated DeltaNet, Lightning, KDA, DSA, MSA, QSA, MLA, sliding-window interleaving);
* compute scaling: MoE and its load balancing;
* very deep residual streams: hyper-connections (mHC, Gated Residual) and Attention Residuals;
* low-precision stability: QK-clip, softcapping, FP8, QAT.

**What does transfer is the small set of changes that make softmax attention more selective, or that allow "attend to nothing".** Target attention is our biggest component (-0.006 to -0.010 NDCG without it), and the raw-cosine bias already showed that sharper retrieval pays (+0.0024).

Ranked list. Every item is cheap; each needs 2 seeds paired against `v2_feed_k9`:

| # | trick | where in `model.py` | added params | prior on gain (val NDCG) | priority |
|---|---|---|---|---|---|
| 1 | **Gated attention, G1**: per-head, elementwise sigmoid gate on the SDPA output, computed from the query token: `o <- o * sigmoid(W_g x_q)` (Qwen3-Next / Qwen3.5 / Qwen3.8; HSTU uses the same `Norm(AV) * U(X)` pattern in recsys) | `Attn.forward`, all 5 attention modules | 16.5K per module (headwise variant: 516) | +0.0005 to +0.002 [inf] | **high** |
| 2 | **QK-norm**: per-head RMSNorm on q and k, optionally zero-centred gain + weight decay as in Qwen3-Next, then try LR 2e-3 | `Attn.forward` | 64 per module | 0 to +0.001, mainly stability and a higher usable LR [inf] | medium-high |
| 3 | **MTP-style aux-weight annealing**: production-order distillation weight 0.5 -> 0.1 in the last ~1/3 of training (DeepSeek-V3 and MiniMax-M2 anneal the MTP weight 0.3 -> 0.1) | `train.py` loss | 0 | 0 to +0.001 [inf] | medium-high |
| 4 | **Weight EMA or WSM checkpoint merging** in the train stage (Ling 2.0 replaces LR decay with merging; our fine-tune snapshot mean already gave +0.0017) | `train.py` | 0 | +0.0002 to +0.001 [inf] | medium-high |
| 5 | **Forgetting-attention bias (FoX)**: data-dependent cumulative recency decay `D_k = sum_{l<k} log sigmoid(w_h . h_l + b_h)` added to the target-attention logits | `Ranker.forward` + `Attn` | 516 | 0 to +0.0015 [inf] | medium |
| 6 | **Muon** (`torch.optim.Muon`, `adjust_lr_fn="match_rms_adamw"`) on hidden 2-D matrices; AdamW for embeddings, input projections, gates, norms, 1-row heads | `train.py` optimizer | 0 | -0.001 to +0.002, high variance [inf] | medium |
| 7 | **Length-aware attention temperature (SSMax)**: logits x `s_h * log n_valid`. Mainly for set attention, where M ranges 5-300 (log M ranges 1.6-5.7) | `Attn.forward` | 4 per module | 0 to +0.001 [inf] | medium-low |
| 8 | SwiGLU (or ReLU^2) FFN at matched parameters (hidden 176 instead of GELU 256) | `Block.ff` | ~0 | 0 to +0.0007 [inf] | low-medium |
| 9 | Differential attention in target attention (2 diff heads of 2x32) | `Attn` | small | -0.001 to +0.001 [inf] | low |
| 10 | Learned attention sink / null key (gpt-oss) or sigmoid-attention "intensity" channel (HSTU, Apple sigmoid attention) | `Attn` | tiny | overlaps with #1 | low (only if #1 is negative) |

**Not recommended (reasons in section 4):** linear-attention hybrids (Gated DeltaNet, Lightning, KDA); sparse attention (DSA, MSA, QSA, NSA); MLA; local/global interleaving and partial RoPE; MoE in any form (shared experts, fine-grained experts, aux-loss-free bias, global-batch LBL); hyper-connections, mHC, Gated Residual and AttnRes; value residual; nGPT; logit softcapping; z-loss; n-gram/Engram-style ID tables, because 39% of test candidate rows are cold items; FP8/QAT.

**Suggested order on GPU N** (2 processes at a time, ~15-20 min each): (1) gate x2 seeds; (2) QK-norm x2; (3) aux annealing + EMA together with an EMA-off control; (4) FoX x2; (5) Muon x2; (6) combine the winners and re-run 2 seeds, then do stage 2 and the fine-tune.

---

## 1. Our model in LLM terms (what the tricks act on)

From `nn/model.py` (read this session):

* `Attn`: `q = Linear(d,d)`, fused `kv = Linear(d,2d)`, `o = Linear(d,d)`, SDPA with an additive float mask. In cross-attention the mask carries the **similarity bias** `a_h * cos_raw(e_c, e_k)` (`bias_scale` initialised to linspace(0, 8)). There is no gate, no QK-norm and no sink.
* `Block`: pre-LN, `x + Attn(LN(x), LN(ctx))`, then `x + FFN(LN(x))`, where FFN = `Linear(d,2d) -> GELU -> Dropout -> Linear(2d,d)` (ff=2).
* Context for target attention = [user token, 256 history tokens]. **The user token is always valid at key index 0, so it is a built-in attention sink** (relevant to section 2.1).
* History tokens carry a learned absolute recency embedding `rec(0..255)`, newest first. `sim_bias=2` (a static per-head recency/type table) was neutral.
* Set attention over candidates: no positional encoding; the key count M varies from 5 to 300.
* Optimizer: AdamW (0.9, 0.98), lr 1e-3, wd 0.02 on 2-D non-"emb" parameters, warmup 500 + cosine to 3%, clip 1.0, bf16 autocast, fp32 head.
  * Side note: the decay filter is `p.ndim >= 2 and "emb" not in n`. The tables `cat.weight`, `et.weight`, `es.weight`, `rec.weight` and `dom.weight` do not contain "emb", so **they are weight-decayed**. That is probably harmless at wd 0.02, but it is not what the filter intends.
* Aux head: listwise CE toward softmax(-pos/10), weight 0.5 x (1 - p_random), constant over training.

Scale: about 3e15 training FLOPs [inf]. For comparison, the smallest experiments in the cited LLM papers are 82M-params ResFormer, 125M FoX, 162M SSMax and 340M softpick, trained on billions of tokens; the main results are at 0.4B-1000B+ params. **Every effect size below is a prior, not a prediction.**

---

## 2. Tricks that plausibly transfer (exact math, evidence, concrete change)

### 2.1 Gated attention (SDPA output gate, "G1"): HIGH

**Math** (Qiu et al., 2025, arXiv 2505.06708) [V]:

`Y' = Y ⊙ sigma(X W_theta)`

* Y is the SDPA output, i.e. the per-head attention result before concatenation and `W_O`.
* X is the **hidden state after pre-normalization**, the same tensor that produced Q.
* The gate is head-specific and multiplicative. Elementwise means W_theta in R^{d x d}; headwise means R^{d x h}.

Other positions tested: G2 after V, G3 after K, G4 after Q, G5 after W_O. G1 is the best.

**Measured** [V]:

| model | baseline | G1 elementwise | G1 headwise |
|---|---|---|---|
| 15A2B MoE, 400B tokens: avg PPL | 6.026 | 5.761 | 5.792 |
| 15A2B MoE: MMLU | 58.79 | 60.82 | 60.05 |
| 15A2B MoE: GSM8k | 52.92 | 55.27 | 54.44 |
| 15A2B MoE: added params | 0 | 201M | 1.6M |
| 1.7B dense, 28 layers: avg PPL | 7.499 | 7.404 | - |

* Mean gate value: 0.116 (elementwise) vs 0.335 for an input-independent gate. Query-dependent sparsity is the claimed mechanism.
* First-token attention share: 46.7% -> 4.8%. Massive activations: 1053 -> 94.
* Learning rate: at LR 8e-3 the 48-layer dense baseline degrades to PPL 9.195 (and diverges at 1T tokens), while the gated model reaches 7.325. Sandwich norm alone reaches 7.407.

**Corroboration:**
* FoX "Pro" ablation at 360M params / 7.5B tokens [V]: removing the output gate raises PPL from 6.62 to 6.86, the largest single Pro-component loss after the forget gate itself.
* Qwen3-Next adopted it "to reduce low-rank issues in attention" and to "eliminate ... Attention Sink and Massive Activation" [V, Alibaba Cloud blog].
* Qwen3.5 (layout 15x(3xGDN + 1x**Gated Attention**)) [V, HF card] and Qwen3.8-Flash-Next (which also switches the GDN output gate from SiLU to sigmoid, "consistent improvements") [V] keep it.
* In recommendation, HSTU computes `Y = f2(Norm(A(X)V(X)) ⊙ U(X))` with U from a SiLU projection of the same token [V, arXiv 2402.17152]. That is the same idea already validated for user sequences.

**Why it should help here [inf]:**
* Each candidate's target attention must output a convex combination over [user, history]. A candidate unrelated to the history cannot say "nothing relevant": its mass goes to the user token (our built-in sink) or to near-noise matches.
* A query-dependent gate gives each candidate a per-dimension "how much to trust this retrieval" switch. This is the missing **intensity / absence** signal that HSTU argues softmax removes.
* The same applies to set attention, which has no sink at all.
* `nohist` decays after 2k steps, and the sim bias helped. Both suggest that attention selectivity matters.

**Change** (`Attn`; `x` passed in is already `ln1(x)`, exactly the paper's X):
```python
# __init__:  self.gate = gate  ("none" | "elem" | "head")
if gate == "elem": self.g = nn.Linear(d, d)        # head-specific elementwise (paper's best)
elif gate == "head": self.g = nn.Linear(d, heads)  # 1 scalar per head (cheap variant)
# forward, after SDPA:
o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=...).transpose(1, 2)  # [B,N,h,dh]
if self.gate != "none":
    o = o * torch.sigmoid(self.g(x)).view(B, N, h, -1)   # dh-wide (elem) or broadcast (head)
return self.o(o.reshape(B, N, D))
```
* Initialization:
  * From scratch: default init, so the gate starts near 0.5.
  * Grafting onto an existing v2 checkpoint (stage 2 or fine-tune, a cheap first look): use `2*sigmoid(.)` with `W_g = 0, b_g = 0`, which is the exact identity at init. Qwen3.8's Gated-Residual write gate uses the same `s = 2 sigma(.)` trick [V]. Because `finetune.py` loads with `strict=not n_extra`, the new keys must be present or initialised; add a cfg key with a default so old checkpoints still load.
* Apply to all 5 attention modules first, then ablate cross-only.
* Cost: 5 x 16.5K = 82K params (+10%), negligible FLOPs.
* Diagnostics to log:
  * mean gate value (expect it to fall well below 0.5);
  * per-head attention mass on the user token before and after. If the baseline puts more than ~30% on the user token, sink behaviour is present.
* Follow-up if positive: the paper's second claim is that the gate tolerates a larger LR. Try lr 2e-3 with gate + QK-norm.

### 2.2 QK-norm (with learnable gain; zero-centred variant): MEDIUM-HIGH

**Math:** per head, `q_h <- RMSNorm(q_h) ⊙ gamma_q`, `k_h <- RMSNorm(k_h) ⊙ gamma_k`, then the usual `softmax(q k^T / sqrt(d_h) + bias)`.
* Logits are then bounded: |q.k|/sqrt(d_h) <= sqrt(d_h) * |gamma_q| |gamma_k| (= 5.7 x gains for d_h = 32).
* This removes the "attention logit growth" failure mode, in which logits grow through the q/k norms and not through alignment [V, Wortsman et al. 2309.14322].

**Evidence:**
* Qwen3 removed the QKV bias and added QK-Norm "to ensure stable training" [V].
* Gemma 3: "we replace the soft-capping of Gemma 2 with QK-norm" [V].
* Ling 2.0: QKNorm "for training robustness under low precision" [V].
* Wortsman et al.: with qk-layernorm, LR sensitivity is much lower and models train at high LR [V].
* FoX Pro ablation (360M): removing QK-norm raises PPL from 6.62 to 6.79 [V], a measurable gain at small scale.
* Qwen3-Next found that some QK-norm gains grew abnormally large. It moved to **zero-centred RMSNorm** (gain = 1 + w, w initialised to 0) **with weight decay on the norm weights** [V, Alibaba Cloud blog].
* Kimi K2 could not use QK-norm because MLA never materialises K at inference, so it built QK-Clip instead [V]. That constraint does not apply to us.

**Transfer [inf]:**
* Our content logits share the softmax with the raw-cosine bias `a_h cos`, which spans ±8.
* QK-norm makes the learned-space logit a bounded cosine-like term with a learned temperature, the same "cosine plus scale" form as the sim bias (nGPT also uses normalized q, k with a learned `s_qk` [V]).
* It cannot hurt much and it enables the higher-LR follow-up.

**Change:** in `Attn.__init__`, add `self.qn = nn.RMSNorm(d // heads, eps=1e-6); self.kn = nn.RMSNorm(d // heads, eps=1e-6)`; in `forward`, apply `q, k = self.qn(q), self.kn(k)` before SDPA.
* Zero-centred variant: store `w` and use `weight = 1 + w`; put `w` in the weight-decay group.

### 2.3 Aux-loss weight annealing, the MTP schedule: MEDIUM-HIGH (nearly free)

**What the LLMs do:**
* DeepSeek-V3 MTP: `L_MTP = (lambda/D) sum_k L_MTP^k` with D = 1, **lambda = 0.3 for the first 10T tokens, then 0.1 for the last 4.8T** [V].
* MiniMax-M2: MTP weight 0.3, annealed to 0.1 in the decay phase [V].
* Ling 2.0: weight 0.1 [V].

In all three the auxiliary prediction target is a representation-learning aid that is down-weighted at the end, so the final optimisation is dominated by the main objective.

**Transfer [inf]:**
* Our K9 head (listwise CE toward the production order) is exactly such an aid: +0.0024 val. But its target is a *biased teacher*: the production ranker plus position bias.
* Keeping weight 0.5 until the last step may pull the shared trunk toward the production order during the low-LR phase, which is where the final ranking is set.

**Change:** in `train.py`, `w_aux = a.aux_pos if step < 0.67 * total else a.aux_end`, with `aux_end` in {0.1, 0}. A smooth alternative is `w_aux = 0.1 + (0.5 - 0.1) * lr_at(step) / a.lr`, which couples it to the cosine.

### 2.4 Weight averaging: WSM / checkpoint merging / EMA in the train stage: MEDIUM-HIGH (free)

**Math:**
* WSM (Ling team, arXiv 2507.17634) [V]: with a constant LR after warmup, merging checkpoints emulates a decay schedule: `theta_hat_{n+k} = theta_n - sum_{i=1}^{k} w_i g_{n+i-1}`, with merge coefficients `c_k = w_k`, `c_j = w_j - w_{j+1}`, `c_0 = 1 - w_1` [V, Ling 2.0 report].
* Example [inf]: a uniform average of k+1 equally spaced checkpoints gives `w_i = (k+1-i)/(k+1)`, i.e. a linear decay. The abstract says **merge duration** (the window length) matters most, more than the checkpoint interval or count.
* Reported gains over WSD: MATH +3.5%, HumanEval +2.9%, MMLU-Pro +5.5% [V, abstract].
* Ling 2.0 uses WSM and averages the top N = 32 checkpoints of the last mid-training stage [V].
* modded-nanogpt uses "weight averaging (EMA) with a raised LR floor" [V].

**Transfer [inf]:**
* The 5-fold fine-tune already showed that a snapshot mean beats the best single epoch by +0.0015-0.002.
* The train stage selects one checkpoint on the holdout every 1000 steps, so the selection itself is noisy.
* Two cheap variants:
  * **(a)** keep the cosine and add an EMA of the weights (decay 0.9995, about a 2K-step horizon, roughly 10% of 20K steps), updated from 50% of training on; evaluate both raw and EMA weights on the holdout.
  * **(b)** WSM proper: constant LR (no cosine), then a uniform mean of checkpoints over the last 30%.
* EMA weights also give stage 2 a smoother starting point.
```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9995), use_buffers=True)  # copies f_mean/f_std
...  # after opt.step():
if step >= total // 2: ema.update_parameters(model)
```

### 2.5 Forgetting attention: data-dependent recency decay for history (FoX): MEDIUM

**Math** (Lin et al., ICLR 2025, arXiv 2503.02130) [V]:
* Scalar forget gate per head: `f_t = sigma(w_f^T x_t + b_f)`.
* Attention: `O = softmax(Q K^T + D) V`, with `D_ij = sum_{l=j+1}^{i} log f_l` (D_ii = 0).
* No positional embedding is needed (adding RoPE to FoX-Pro: 6.62 -> 6.63).
* Ablation (360M, 7.5B tokens): removing the forget gate gives PPL 6.82 (Transformer-Pro without RoPE) and 7.40 with RoPE, vs 6.62 with the forget gate. Sizes tested: 125M, 360M, 760M.
* It generalises ALiBi (`D_ij = -m_h (i-j)`) to a content-dependent slope: the model "forgets faster" after informative events.

**Transfer [inf]:**
* Our history has no timestamps, only a recency index. The current recency signal is static: the absolute `rec` embedding, plus the per-head `sim_r` table in `sim_bias=2`, which was neutral.
* FoX adds what neither has: a decay that depends on *what happened in between*. A burst of contacts in a new category, or a session switch, should down-weight everything older for some heads.
* For a candidate (query time = now) and history item k (newest first), the in-between events are l < k, so `D_k = sum_{l<k} log f_l`.
* Cost: a d->h linear and a cumsum.

**Change** (after `hist_enc`, passed into the cross blocks as an extra additive term on the mask):
```python
# __init__: self.fg = nn.Linear(d, heads); nn.init.zeros_(self.fg.weight)
#           self.fg.bias.data = torch.logit(torch.tensor([0.999, 0.995, 0.98, 0.95]))  # ALiBi-like spread
logf = F.logsigmoid(self.fg(ctx[:, 1:]).float()) * b["hmask"][..., None]   # [B,K,h], padding: no decay
D = torch.cumsum(logf, 1) - logf                                            # exclusive cumsum
dbias = torch.cat([torch.zeros_like(D[:, :1]), D], 1).permute(0, 2, 1)[:, :, None, :]  # [B,h,1,K+1], user token 0
# in Attn: am = am + dbias  (and convert the boolean-mask path to float: dbias.masked_fill(~kmask[:,None,None,:], -inf))
```
* With that bias init, `D` at k = 255 is about -0.26 / -1.3 / -5.2 / -13 per head. Heads span "no decay" to "last ~20 events".

### 2.6 Muon optimizer: MEDIUM (high variance)

**Math:**
* Momentum step: `M_t = mu M_{t-1} + G_t`. Update: `O_t = NS_k(M_t) * 0.2 sqrt(max(A,B))`. Weights: `W <- W - eta (O_t + lambda W)`.
* NS = Newton-Schulz iteration toward the orthogonal polar factor. The 0.2 sqrt(max(A,B)) factor matches AdamW's update RMS, so AdamW's LR and WD can be reused [V, Kimi K2 report; V, local `torch.optim.Muon` docstring, torch 2.11 in `.venv`, `adjust_lr_fn="match_rms_adamw"`].

**Evidence:**
* Moonlight: about 2x compute efficiency vs AdamW at compute-optimal scale [V, abstract 2502.16982].
* modded-nanogpt (124M GPT-2): Muon "~1.5x better sample-efficiency, <2% wallclock overhead"; the record that introduced it went from 31.4 to 24.9 min [V].
* Qwen3.8-Flash-Next [V]:
  * Muon with Nesterov (mu = 0.95), gamma = 0.2 sqrt(max(A,B)), 8 NS steps (fewer gradient-norm spikes).
  * Fused QKV and fc1 are **orthogonalised per sub-matrix**.
  * **AdamW** is kept for embeddings, the output head, the router, low-rank gate projections, decay/beta projections and output gates.
  * Muon on the router destabilised it.
  * At 4x the optimal LR, AdamW spiked 183 times per 10K steps; Muon (+ gated residual) did not spike at all.

**Transfer [inf]:**
* Most of our parameters are 128x128 / 128x256 hidden matrices, which is Muon's sweet spot. 2 epochs over 3.15M feeds is a fixed budget, and better per-step efficiency may turn into a better final model.
* Risks:
  * The schedule and WD interplay needs a mini-sweep (lr 1e-3 and 2e-3).
  * Ranking quality is not LM loss.
  * I found no primary source of Muon on recsys rankers.

**Change** (two optimizers, same `lr_at` schedule):
```python
def is_muon(n, p):  # hidden 2-D maps only
    if p.ndim != 2 or min(p.shape) == 1: return False   # wide (1 x F), head.3 / aux_head.3 (1 x d) -> AdamW
    skip = ("cat.", "et.", "es.", "rec.", "dom.", "id_emb", "pos_emb", "rev_emb",   # embedding tables
            "c_lin", "h_lin", "u_lin", "c_mlp.0", "extra.0",                        # input projections of fixed features
            ".g.", "fg.")                                                           # gates (AdamW per Qwen3.8)
    return not any(s in n for s in skip)
opt_m = torch.optim.Muon([p for n, p in model.named_parameters() if is_muon(n, p)], lr=a.lr,
                         weight_decay=a.wd, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")
```
* Split the fused `kv` into separate `k` and `v` Linears for new runs, so each is orthogonalised separately as in Qwen3.8.
* A cheap second arm also puts the input projections on Muon.

### 2.7 Length-aware attention temperature (Scalable-Softmax, SSMax): MEDIUM-LOW

**Math** (arXiv 2501.19399) [V]: `softmax((s log n) q K^T / sqrt(d))`, with n = number of keys and s a learnable scalar per layer and head. Optional bias variant: `(s log n + b)`.
* Result on a single 162M model: lower training loss (about -0.008) and much better length extrapolation (needle retrieval at ~10x the training length).
* Rationale: as n grows, the max softmax weight shrinks (attention fading) unless the logits scale with log n.
* Related: sigmoid attention with bias `b = -log n` [V, Apple 2409.04431] and Qwen's log-n attention scaling [K].

**Transfer [inf]:**
* Set attention sees M from 5 to 300 (log M from 1.6 to 5.7, a 3.5x temperature range); target attention sees H+1 from 1 to 257.
* With one temperature for all feeds, attention over a 300-candidate feed is far flatter than over a 20-candidate feed. That may be one reason set attention is "neutral".

**Change:** in `Attn`: `n = kmask.sum(-1).clamp_min(2).float()`; `q = q * (self.ss[None,:,None,None] * n.log()[:,None,None,None])`, with `ss` initialised to 1/log(96) ≈ 0.22 per head, so a mean-length feed is unchanged at init.

### 2.8 GLU feed-forward (SwiGLU / GEGLU) or ReLU^2: LOW-MEDIUM

**Math** (Shazeer 2020) [V, PDF text]:
* `FFN_SwiGLU(x) = (Swish(x W) ⊙ x V) W2`; GEGLU uses GELU in place of Swish. No biases.
* d_ff is reduced by 2/3 to keep the parameter count (3072 -> 2048).
* T5-base, C4 held-out log-perplexity:

| FFN | 65K steps | 524K steps |
|---|---|---|
| ReLU (baseline) | 1.997 | 1.677 |
| GELU | 1.983 | 1.679 |
| GEGLU | 1.942 | 1.633 |
| SwiGLU | 1.944 | 1.636 |

* ReLU^2: Primer at 1.9B used 1/3 of the compute for the same one-shot quality [V, abstract], though without an isolated ablation. modded-nanogpt bundles "ReLU², zero-init projections, QK-norm" in one record (22.3 -> 15.2 min) [V].
* All modern LLMs in scope use SwiGLU: Qwen3, Ling 2.0, gpt-oss (a "clamped" variant with a residual) [V].

**Transfer [inf]:** the gain is in the low single-digit % of log-loss for LMs. Our FFNs (ff = 2) sit in only 5 blocks, and the candidate token already passes through a GELU MLP. The expected effect is small, but the change is free.

**Change:** `Block.ff = SwiGLU(d, hid=176)`, i.e. `w12 = Linear(d, 2*176)`, `out = Linear(176, d)`, `out(silu(a) * b)`. This holds 67.6K params vs 65.5K for GELU ff = 2.

### 2.9 Differential attention: LOW

**Math** (Ye et al., arXiv 2410.05258) [V]:
* `DiffAttn(X) = (softmax(Q1 K1^T/sqrt d) - lambda softmax(Q2 K2^T/sqrt d)) V`.
* `lambda = exp(l_q1 . l_k1) - exp(l_q2 . l_k2) + lambda_init`, with `lambda_init = 0.8 - 0.6 exp(-0.3(l-1))`.
* Per-head RMSNorm (GroupNorm), then multiply by (1 - lambda_init). The head count is halved.

**Evidence:**
* 6.8B Diff matches 11B Transformer; 160B tokens match 251B.
* 1.4B ablation: loss 3.062 vs 3.087. **Without GroupNorm it is worse than the baseline (3.122).**
* The smallest scaling-study model is 830M [V].

**Transfer [inf]:**
* The aim is noise cancellation in retrieval, i.e. our target attention. But the sim bias already does exact retrieval, and with 4 heads x 32 dims, Diff leaves only 2 heads.
* More code and sensitivity (GroupNorm is mandatory) for a small expected gain. Try it only after #1-#5.

### 2.10 Attention sinks done explicitly; non-softmax attention: LOW (fallback for #1)

* **gpt-oss**: "Each attention head has a learned bias in the denominator of the softmax" [V]: `a_ij = exp(z_ij) / (exp(s_h) + sum_j exp(z_ij))`.
  * Implementation: one extra all-zero key/value slot whose mask entry is `s_h`.
  * Our cross-attention already has a quasi-sink (the user token, whose value is not zero), and set attention has none.
* **Softpick** [V]: `ReLU(e^x - 1) / sum |e^x - 1|`. 0% sink rate and much lower kurtosis at 340M (benchmarks ~equal), but **worse at 1.8B** (e.g. ARC-e -5.2, Lambada -5.9) with dead heads. Not recommended.
* **Sigmoid attention** [V]: `sigma(QK^T/sqrt d + b)` with `b = -log n`. It needs QK-norm, plus a hybrid norm (`x + norm(attn)`) at 1B/4K, and then matches softmax (1B/2K: 49.5 vs 49.4).
* **HSTU** [V]: SiLU pointwise attention `phi2(QK^T + rab)V` with LayerNorm and gating. Pointwise beats softmax by up to 44.7% HR@10 (synthetic streaming data). The rationale is that **the number of related history items encodes preference intensity**, which softmax normalises away.
* **Gu et al. 2410.10781** [V]: sinks act like key biases and come from softmax normalisation; unnormalised (sigmoid) attention prevents them up to 1B.
* [inf] Our dense features already include counts and log-sum-exp pooled similarity (`lse`, `ccnt`). An "intensity" channel would therefore mainly duplicate them. The G1 gate (#1) gives the same "attend to nothing" capacity with less risk.
* If #1 is negative, try one sigmoid-attention head (with `b = -log(H+1)` and QK-norm) in target attention, concatenated with the softmax heads. This is item N5 from the coordinator's list.

---

## 3. Model landscape 2025-2026 (verified architecture facts)

| model | token mixing | MoE | norms / stability | other | tag |
|---|---|---|---|---|---|
| Qwen3 (May 2025) | full GQA, RoPE | 128 experts, 8 active, **no shared expert**, **global-batch LBL** | **QK-Norm**, QKV-bias removed, RMSNorm pre-norm, SwiGLU | - | [V] 2505.09388 |
| Qwen3-Next-80B-A3B (Sep 2025) | 48 layers = 12x(3x **Gated DeltaNet** + 1x **Gated Attention**); attention head dim 256, partial RoPE on 25% of dims | 512 experts, 10 routed + 1 shared | **zero-centred RMSNorm + WD on norm weights** (replacing QK-norm gain growth), router params normalised at init | MTP; claims 10% of Qwen3-32B training cost | [V] HF card, Alibaba Cloud blog |
| Qwen3.5-397B-A17B (2026) | 60 layers = 15x(3x GDN + 1x Gated Attention); GDN 64 V / 16 QK heads | 512 experts, 10 + 1 shared, expert dim 1024 | - | multi-step MTP | [V] HF card |
| Qwen3.8-Flash-Next (Aug 2026, described as the Qwen4 architecture preview; Qwen4 itself is not released as of this writing [S]) | 3 GDN : 1 full attention, full layers swapped for **Qwen Sparse Attention** (micro-block indexer) in continued pretraining | 125B total / 6B active + **51B n-gram embedding** at layer 2 | **Gated Residual** (4 residual branches, sigmoid read gate, `2 sigma` write gate) beats mHC (loss 1.590 vs 1.594, pre-norm 1.617); GDN output gate SiLU -> sigmoid; **Muon** (8 NS steps; AdamW for router and gates) | n-gram table: loss falls monotonically with size, downstream saturates | [V] 2608.30320 |
| MiniMax-01 (Jan 2025) | Lightning (linear) attention hybrid | 32 experts, 456B / 45.9B | - | - | [V] abstract; the 7:1 ratio is [K] |
| MiniMax-M2 (Oct 2025; report May 2026) | **full attention, all 62 layers** (48 Q / 8 KV heads); SWA hybrid rejected (RULER-128K CWE 90 -> 72, SWE-verified 54.7 -> 50.2) | 256 experts, 8 active; sigmoid gate + per-expert bias (aux loss reduced, not removed) | - | MTP weight 0.3 -> 0.1 at decay | [V] 2605.26494, MiniMax blog |
| MiniMax-M3.1-Flash (Oct 2026) | **MiniMax Sparse Attention** (block top-k per GQA group via an index branch; 28.4x less attention compute at 1M context) everywhere | ~428B / 23B active (HF metadata says 220B, which conflicts) | QAT to NVFP4 | DSpark draft head replaces the EAGLE-like MTP | [V] HF card, 2606.13392 abstract |
| Ling 2.0 (Oct 2025; inclusionAI) | GQA, **QKNorm**, partial RoPE (64 dims) | 256 experts, 8 + 1 shared (3.5% activation), aux-loss-free (gamma = 1e-3), router gate scaling 2.5 | RMSNorm, SwiGLU, FP8 (<=0.25% gap to BF16) | **WSM** schedule (merge the top 32 checkpoints); MTP weight 0.1; Ling scaling laws | [V] 2510.22115 |
| Ling-3.0-flash (2026) | 35 **KDA** + 7 gated MLA (5:1) | 512 experts, 8 active + 1 shared | - | MTP | [V] HF card |
| DeepSeek-V3 (Dec 2024) | MLA | DeepSeekMoE: sigmoid affinities, top-K normalised, shared experts, **aux-loss-free bias** (gamma = 1e-3) + sequence balance loss alpha = 1e-4 | - | MTP D = 1, lambda 0.3 -> 0.1 | [V] 2412.19437 |
| DeepSeek-V3.2 (Dec 2025) | **DSA**: lightning indexer `I_ts = sum_j w_tj ReLU(q_tj . k_s)`, top-2048 tokens, indexer warmed up by KL to dense attention | as V3 | - | - | [V] 2512.02556 |
| DeepSeek-V4 (2026) | CSA (4x KV compression + indexer) / HCA (128x) | DeepSeekMoE | **mHC** residuals | Engram cited for the future (V5) | [V] HF blog (partial); Engram [V] abstract; mHC [V] abstract |
| Kimi K2 (Jul 2025) | MLA, 64 heads | 384 experts, 8 + 1 shared, sparsity 48 | **MuonClip**: per head, if `S_max^h > tau = 100`, scale `W_q, W_k` by `sqrt(tau/S_max^h)` | - | [V] 2507.20534 |
| Gemma 2 / 3 | local:global 1:1 (window 4096) -> **5:1** (window 1024; ratio "minimal impact on perplexity") | dense | Gemma 2: **pre+post RMSNorm**, softcap 50 (attention) / 30 (final); Gemma 3: **QK-norm replaces softcap** | Gemma 2 distillation: 2B 60.3 -> 67.7 | [V] 2408.00118, 2503.19786 |
| gpt-oss (Aug 2025) | alternating banded (128) / dense, GQA 64 Q / 8 KV | 128 or 32 experts, top-4, softmax over the selected experts | pre-LN RMSNorm, **learned per-head attention sink**, clamped SwiGLU + residual | - | [V] 2508.10925 |

Recurring 2025-2026 consensus: (i) attention **output gating** (Qwen family, FoX-Pro, gpt-oss sinks, HSTU in recsys); (ii) **QK normalisation** (Qwen3, Gemma 3, Ling 2.0; Qwen3-Next refines the gain); (iii) **Muon** at scale (Kimi K2, Qwen3.8, DeepSeek-V4 [S]); (iv) **fine-grained MoE with a shared expert and bias-based balancing**; (v) **linear or sparse attention for long context**, with M2 a notable counterexample that kept full attention because of retrieval and multi-hop deficits.

Only (i)-(iii) are about the quality of a small, short-sequence model.

---

## 4. Tricks that do NOT transfer (and why)

* **Linear-attention hybrids** (Gated DeltaNet `S_t = S_{t-1} alpha_t (I - beta_t k_t k_t^T) + beta_t v_t k_t^T`, `o_t = S_t q_t` [V]; Lightning; KDA):
  * They replace O(n^2) attention with a fixed-size state, i.e. *lossy compression of the context*. They pay off only when n is in the tens of thousands.
  * Our worst case is 300 x 257 logits per head, which is trivial.
  * MiniMax found that hybrids look fine at small scale but lose retrieval and multi-hop ability ("clear deficits in complex, multi-hop reasoning tasks"; RULER CWE 90 -> 72 for SWA) [V].
  * Target attention *is* retrieval, so keep full softmax.
  * The one idea worth keeping, a data-dependent decay, is imported more cheaply through FoX (2.5).
* **Sparse attention** (DSA top-2048, MSA block top-k, QSA, NSA, CSA/HCA): these select k of n keys when n >= 1e5. With n <= 257 the "sparse" set would be all keys. The sim bias plus a gate already gives soft selection.
* **MLA / GQA / KV compression**: inference-memory tricks for KV caches. There is no decode loop here.
* **Interleaved local/global attention, partial RoPE, YaRN**: our cross-attention has no relative positions (candidates have no time), and the history is short. FoX shows that a forget gate makes positional encodings unnecessary [V].
* **MoE** (fine-grained experts, a shared expert, `g = topK(sigmoid(u e_i))` normalised, aux-loss-free bias `b_i <- b_i ± gamma`, global-batch `LBL = N_E sum f_i p_i`) [V]:
  * Ling's law is `EL(A,G,C) = A_hat^{alpha + gamma (log G)^2 + beta log G}` with `alpha = a + d log C`, `d > 0`, i.e. **MoE leverage grows with compute** [V]. It was fitted on 300+ models up to 28B params.
  * We are about 3 orders of magnitude or more below that compute range, and width 256 / 3 blocks were already neutral (capacity is not binding). Expected leverage is about 1 [inf].
  * MoE adds routing noise, balance losses and seed variance.
  * The only recsys-flavoured variant, hard routing by a known domain (production / exploration), is what the domain token already tested (neutral).
* **Hyper-connections / mHC / Gated Residual / Attention Residuals**: these fix residual-stream dilution in 28-80-layer pre-norm stacks (GR: loss 1.590 vs pre-norm 1.617 at 25B-A3B [V]; AttnRes on a 48B model [S]). We have about 5 sublayers. Not worth it.
* **Value residual learning** (`V_n = lambda1 V_1 + lambda2 H_{n-1} W_V^n`; -16% params at equal loss for 468M; later layers benefit most [V]): it needs a deep self-attention stack. Our two cross blocks already read the same encoded context.
* **nGPT** (all states on the hypersphere, `h <- Norm(h + alpha (h_A - h))`, no WD, no warmup [V]): the 4-20x token-efficiency claim is for 0.5-1B LMs at 1-8K context, with 60-80% step overhead and hyperparameter sensitivity (1B/8K needed alpha_init = 0.1) [V]; independent replications have been mixed [K]. Too invasive for our pipeline. Only its QK-normalisation piece is kept (2.2).
* **Logit softcapping** (`c tanh(z/c)`, 50 / 30 in Gemma 2 [V]): Gemma 3 dropped it for QK-norm [V]. Our listwise CE is shift-invariant, and fp32 logits already avoid ties.
* **z-loss** (`1e-4 log^2 Z` [V]): it stabilises LM output softmax drift. For a per-feed listwise softmax, log Z carries no ranking information, so it is a no-op for NDCG.
* **QK-Clip / MuonClip**: needed only when QK-norm is impossible (MLA) and logits exceed 1000 at 9B-active scale [V]. With QK-norm (2.2) it is unnecessary.
* **Peri-LN / sandwich norm** (`y = x + Norm_out(F(Norm_in(x)))`; 400M avg benchmark 49.69 -> 51.57, seed std halved [V]): its purpose is to control hidden-state growth across many layers. At depth 5, Pre-LN growth is negligible [inf]. Optional one-off run only.
* **n-gram / Engram / hashed-ID tables** (Qwen3.8: 51B-param table, loss improves monotonically but downstream saturates [V]; Engram: +3-5 points on knowledge/reasoning at 27B [V]):
  * The analogue for us is item-ID memorisation, but 28.9% of val and **39.3% of test** candidate rows are cold items, and private is the coldest split (`reports/eda_shift_overlap.md`).
  * A content-free analogue is possible: hashed (event_type_t, event_type_{t+1}, same_category) transition embeddings, or a FoX-style KV-shift `k_t = alpha k_{t+1} + (1-alpha) k_t` on history keys (FoX: removing KV-shift raises PPL 6.62 -> 6.80 [V]). Low priority, since the history self-attention layer with recency embeddings can already see neighbours.
* **MTP proper**: there is no next-token structure. Its useful content, a down-weighted auxiliary target with a schedule, is item 2.3.
* **FP8 / NVFP4 / QAT**: speed only. Note that bf16 logits already cost 0.001-0.002 NDCG through ties.

Related, from Gemma 2: **distillation from a stronger teacher gives large gains at small size** (2B: 60.3 -> 67.7 avg; validation PPL 23 -> 21 at 200M [V]). This supports the already-proven production-order distillation, and K3 (distilling the val-fine-tuned ensemble with its ALS / co-visitation / near-duplicate features into the train-stage model). That idea is out of scope here but probably worth more than any single architecture trick above.

---

## 5. Experiment protocol (GPU N only)

* Work in a copy of `nn/` (e.g. `nnx/`). In the copy's `common.py`, change the guard to accept `"5"`, and `export CUDA_VISIBLE_DEVICES=N`. Run at most 2 training processes at a time, each with <=24 CPU threads (`OMP_NUM_THREADS=24`), in the foreground or waited on.
* Baseline = the `v2_feed_k9` recipe (`--no_pos --aux_pos 0.5 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --epochs 2 --eval_every 1000 --loss_norm feed --aux_mode listwise --aux_tau 10`) on seeds 21 and 23. Re-run the baseline on GPU N rather than reusing the old logs, because timing and kernel choices differ.
* New flags (all defaulting to off so old checkpoints load unchanged):
  * `--attn_gate {none,head,elem}`;
  * `--qk_norm {0,1,2}` (2 = zero-centred gain with WD);
  * `--forget`;
  * `--ssmax`;
  * `--aux_end 0.1 --aux_anneal_at 0.67`;
  * `--ema 0.9995`;
  * `--opt {adamw,muon}`;
  * `--ffn {gelu,swiglu}`.
* **Acceptance rule:**
  * Mean paired delta >= +0.0008 on val **and** a positive delta on the 30K train holdout for both seeds.
  * If the result is borderline (+0.0004 to +0.0008), add a third seed.
  * Report val, holdout, and step-matched curves.
* Order and grouping:
  1. `--attn_gate elem` x2;
  2. `--qk_norm 1` x2;
  3. `--aux_end 0.1 --ema 0.9995` x2 (the EMA readout is free: log raw and EMA metrics in the same run, which isolates aux annealing from EMA);
  4. `--forget` x2;
  5. `--opt muon` x2 (lr 1e-3; add lr 2e-3 if it is close);
  6. winners combined (+ lr 2e-3 if gate and QK-norm both pass) x2;
  7. stage 2 + n2full fine-tune of the best combined model.
* Cheap pre-check for #1: graft a `2*sigmoid` zero-init gate onto `s2_v2_feed_k9` and run stage 2 only (~3 min). A clearly negative result there lowers the prior. A neutral result says nothing, since the from-scratch dynamics matter.

---

## Sources (read this session unless tagged otherwise)

* Gated attention: https://arxiv.org/abs/2505.06708 and https://arxiv.org/html/2505.06708 [V]
* Qwen3 technical report: https://arxiv.org/html/2505.09388 [V]
* Qwen3-Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct [V]; https://www.alibabacloud.com/blog/qwen3-next-towards-ultimate-training-%26-inference-efficiency_602580 [V]
* Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-397B-A17B [V]
* Qwen3.8-Next design paper: https://arxiv.org/abs/2608.30320, https://arxiv.org/html/2608.30320v1 [V]. Qwen4 status: yottalabs / medium posts [S]
* Gated DeltaNet: https://arxiv.org/abs/2412.06464, https://arxiv.org/html/2412.06464 [V]
* Global-batch load balancing: https://arxiv.org/abs/2501.11873 [V]
* MiniMax-01: https://arxiv.org/abs/2501.08313 [V]; MiniMax-M2 report: https://arxiv.org/abs/2605.26494, https://arxiv.org/html/2605.26494v2 [V]; "Why did M2 end up as a full attention model": https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model [V]
* MiniMax-M3.1-Flash: https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash [V]; MiniMax Sparse Attention: https://arxiv.org/abs/2606.13392 [V]
* Ling 2.0: https://arxiv.org/html/2510.22115 [V]; WSM: https://arxiv.org/abs/2507.17634 [V]; Ling-3.0-flash: https://huggingface.co/inclusionAI/Ling-3.0-flash [V]
* DeepSeek-V3: https://arxiv.org/html/2412.19437 [V]; DeepSeek-V3.2: https://arxiv.org/html/2512.02556v1 [V]; DeepSeek-V4: https://huggingface.co/blog/deepseekv4 [V, partial]; mHC: https://arxiv.org/abs/2512.24880 [V]; Engram: https://arxiv.org/abs/2601.07372 [V]
* Kimi K2 / MuonClip: https://arxiv.org/html/2507.20534 [V]; Muon is scalable: https://arxiv.org/abs/2502.16982 [V]; Attention Residuals: https://arxiv.org/abs/2603.15031 [S]
* Gemma 2: https://arxiv.org/html/2408.00118 [V]; Gemma 3: https://arxiv.org/html/2503.19786 [V]
* gpt-oss model card: https://arxiv.org/html/2508.10925 [V]
* Differential Transformer: https://arxiv.org/html/2410.05258 [V]; Value residual: https://arxiv.org/html/2410.17897 [V]; nGPT: https://arxiv.org/html/2410.01131 [V]
* Softpick: https://arxiv.org/html/2504.20966 [V]; SSMax: https://arxiv.org/html/2501.19399 [V]; Peri-LN: https://arxiv.org/html/2502.02732 [V]
* Attention sink emergence: https://arxiv.org/abs/2410.10781 [V]; Sigmoid attention: https://arxiv.org/html/2409.04431 [V]
* Forgetting Transformer: https://arxiv.org/html/2503.02130 [V]
* GLU variants: https://arxiv.org/abs/2002.05202 (PDF text extracted locally) [V]; Primer: https://arxiv.org/abs/2109.08668 [V]
* Small-scale proxies (qk-layernorm, z-loss, AdamW epsilon, independent WD): https://arxiv.org/html/2309.14322 [V]
* HSTU: https://arxiv.org/html/2402.17152 [V]
* modded-nanogpt records: https://github.com/KellerJordan/modded-nanogpt [V]
* Local: `torch.optim.Muon` docstring (torch 2.11.0 in `<repo>/.venv`) [V]; `<repo>/nn/model.py`, `train.py`, `reports/nn_report.md`, `reports/eda_shift_overlap.md` [V]
