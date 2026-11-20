# Tabular deep learning, 2024-2026: tricks that transfer to our feed ranker

Research only: no GPU was used. Written 2026-11-11 for the Avito ML Cup 2026 NN (`<repo>/nn/`).

Tags: **[V]** = I read the primary source in this session (URL given). **[S]** = search snippet or secondary summary only. **[K]** = background knowledge, not re-checked. **[D]** = a measurement on our data made in this session with a CPU-only script (no GPU).

Local copies: the arXiv PDF texts of TabM, RealMLP, "On Embeddings for Numerical Features", FT-Transformer and DIN were already in the session scratchpad (`.../scratchpad/dl/*.txt`). I read them there. The `tabm.py` and `rtdl_num_embeddings.py` sources come from the official GitHub repos.

---

## 0. TL;DR (ranked by expected value per GPU-hour)

| # | Trick (source) | Where in our code | Expected gain | Cost | Priority |
|---|---|---|---|---|---|
| T1 | **Fix the extra-feature preprocessing**: a ±inf bug kills `als_et1_exp5`; add NaN indicators; periodic `hour`; signed-log `*_gap` (tabular preprocessing practice; TabPFN missing indicators) | `data.py::attach_extra` (fine-tune branch) | +0.0002 to +0.0008 OOF | CPU + 1 fine-tune | **high** (one part is a bug fix) |
| T2 | **Fine-tune optimiser**: β2 0.999 → 0.95, plus a cyclic per-epoch cosine LR so each snapshot sits at a cycle minimum (RealMLP β2/multi-cycle schedule; snapshot ensembles) | `finetune.py` | ±0.0005 to +0.001 OOF | 2 fine-tunes | **high** (cheap) |
| T3 | **Numerical embeddings for the dense path**: PLE version "B" (linear + zero-init piecewise-linear) or PLR-lite, plus explicit floor/ceiling flags for f1..f5 (Gorishniy 2022; TabM†; RealMLP PBLD) | `model.py::dense_features`, new `NumEmb` before `c_mlp` | +0.0003 to +0.0015 train-only | 4 runs + 2 baselines (2 seeds, paired) | **high** |
| T4 | **TabR-style values in target attention**: top-m retrieved history items, shared-key L2 similarity, value `W_Y(type) + T(k_c − k_j)`, plus a "no match" sink (TabR ablation; DIN's un-normalised intensity) | new `HistRetrieval` module before `self.cross` | +0.0005 to +0.002 (speculative) | ~0.5 day dev + 4 runs | **high/medium** |
| T5 | **Refit on 100% of val** for the test prediction (fixed snapshot epochs), averaged with the 5 fold models (RealMLP: refit beats bagging) | `finetune.py` (new mode) | +0.0003 to +0.0008 on test (cannot be measured on val) | 1 fine-tune | medium |
| T6 | **TabM-mini (BatchEnsemble), k=4** on the candidate pathway (TabM, ICLR 2025) | `Ranker.forward` (replicate the candidate batch k times) | +0.0005 to +0.0015 | about 3-4x train time | medium/low |
| T7 | **TabM head** (k=16 BatchEnsemble heads, mean of k listwise losses) | `self.head` | 0 to +0.0005 | cheap | low |
| T8 | **Scalable softmax** `q·s_h·log n` in cross/set attention (TabICLv2) | `Attn.forward` | ±0.0005 | cheap | medium/low |
| T9 | **Register / sink tokens** in context and set attention (TabPFN-2.5 "thinking rows") | `Ranker.forward` | ±0.0005 | cheap | low |
| T10 | Smooth recency embedding: PLE/periodic of log(1+hpos) instead of a 256-row lookup | `self.rec` | ~0 to +0.0003 | cheap | low |
| T11 | Muon for 2-D matrices (TabICLv2) | `train.py` optimizer | ±0.001 | 2 runs | low |
| T12 | RealMLP diagonal scaling layer (high LR) on the 63 + 67 inputs | `c_mlp`/`extra` input | ~0 | cheap | low |
| — | **Do not do**: label smoothing, quantile/Gaussianising of f1..f5, robust-scale/smooth-clip of the 61 dense features (our data shows no need), FT-Transformer feature tokens, TabPFN/TabICL as model or stacker, Mambular, tabular MoE, DCNv2, TabR/ModernNCA retrieval over train candidates | — | — | — | skip |

Main conclusions:
1. **The 2024-2026 tabular-DL consensus is that MLPs plus numerical embeddings plus (implicit) ensembling are the strongest family on large and shifted data.** TabM and RealMLP lead tabular DL, and both need ensembling to beat GBDTs. Attention-over-features and retrieval models do not pay off at scale or under temporal shift (TabM, TabArena, TabReD, BeyondArena).
2. **On the one learning-to-rank dataset in these benchmarks (MSLR-WEB10K, 723K rows), the tricks have small effects.** Measured as relative RMSE vs a plain MLP: TabM −0.58%, periodic embeddings −0.41%, PLE −0.13%, a 5-seed MLP ensemble −0.20%; GBDTs are still best. Expect gains of +0.0005 to +0.0015 NDCG per trick, not +0.005.
3. **Our dense path is already well-conditioned [D].** f1..f5 are already uniform "interpolated bins". The standardised features exceed |z|>6 only at the −1.5 "no history" sentinel (1.6% of val candidates). So RealMLP-style scaling has no headroom. Numerical embeddings can still help with the point masses and non-monotone regions.
4. **The best transfer targets are**:
   - the fine-tune stage (small data, where regularisation and ensembling tricks pay: T1, T2, T5);
   - the history target attention, our largest gain source, via TabR's value design (T4).

---

## 1. What our model does today (relevant parts, from `nn/model.py`, `data.py`, `finetune.py`)

* Dense path: `X = dense_features(b)` gives 63 features (61 + 2 user0). `X = ((X − f_mean)/f_std).clamp(−6, 6)`, with mean/std from 40 train batches. Then `c_mlp = Linear(63→256) → GELU → Linear(256→128)` is added to `c_lin([e_c, e_c⊙u]) + cat_emb + ...`, followed by LayerNorm. A linear "wide" term on X is added to the logit.
* History: `Linear(e_h) + emb(type) + emb(source) + emb(cat) + emb(recency 0..255)`, then 1 pre-LN layer over [user, history]. Next come 2 × (cross-attention cand → [user, history] with a per-head scaled raw-cosine bias, then set self-attention).
* Fine-tune: a zero-init `extra` branch `Linear(67→256) → GELU → Linear(256→128)` (last layer zero) is added to the candidate token. Preprocessing in `attach_extra`:
  - `log1p` if min ≥ 0 and max > 20;
  - standardise with val stats;
  - `clip(±6)`;
  - `nan_to_num(0)`.
* Fine-tune optimiser: `torch.optim.AdamW(groups, lr, weight_decay)`, i.e. **default betas (0.9, 0.999)**. LR is constant after 30 warmup steps; the snapshot mean is taken over epochs 1..6. (The train stage uses betas (0.9, 0.98) and cosine decay.)

### 1.1 Measurements on our data [D] (CPU-only scripts, no GPU)

**f1..f5 (3M-row samples per split).** These are "normalised interpolated bins". Each is close to uniform on [floor, ceil] with point masses at both ends. The floor/ceiling values sit on a 1/(2N) grid:

| feature | floor / ceil (exact) | implied bin grid | floor mass train / val | ceil mass train / val |
|---|---|---|---|---|
| f1 | 1/56 / 55/56 | N = 28 | 15.4% / 17.6% | 2.2% / 1.9% |
| f2 | 0 / 63/64 | 64 | 11.8% / 15.4% | 1.7% / 1.9% |
| f3 | 1/62 / 61/62 | 31 | 5.3% / 5.6% | 1.9% / 1.6% |
| f4 | 1/64 / 63/64 | 32 | 1.2% / 1.1% | 2.1% / 1.9% |
| f5 | 0 / 61/62 | 31 | 1.4% / 1.3% | 1.8% / 1.9% |

Target rate by region (floor / middle / ceiling):
- **train**: f1 0.0097 / 0.0239 / 0.0413; f2 **0.0232 / 0.0218** / 0.0318 (non-monotone: the f2 floor behaves like "missing"); f3 0.0040 / 0.0229 / 0.0344.
- **val**: f1 0.0084 / 0.0171 / 0.0312; f2 0.0170 / 0.0154 / 0.0222; f4 **0.0038** / 0.0159 / 0.0161.
- **The f4 floor shifts meaning** between train (0.0166, neutral) and val (0.0038, very bad).

So the point masses carry sharp, sometimes non-monotone effects. A single linear unit on standardised f cannot represent a step at a point mass. The NN currently relies on downstream layers for this.

**The 63 dense features after train standardisation (3000 val feeds, 288K candidates).** All within-feed z/rank features have std ≈ 1 and |z|>6 = 0. The similarity family (smax, smax8/32, lse, top3) has |z|>6 for 1.6%. That is exactly the share of the −1.5 sentinel used when a feed has no history (mode mass 0.016). `cmiss` and `lognorm` have skew ±84 (0.01% missing embeddings). `user0` is 5.5% of candidates. Conclusion: **the clamp only bites on sentinels, so robust scaling or smooth clipping would change almost nothing.**

**The 67 n2full extra features (val):**
- **Bug**: `als_et1_exp5` contains ±inf (val: 12 +inf, 91 −inf; test: 135 / 706). `np.nanmean`/`np.nanstd` return NaN, so `(X−NaN)/NaN` is NaN, and `nan_to_num` sets it to 0. **The whole column is constant 0 in every n2full fine-tune** (verified: after the transform, 100% zeros).
- NaN shares:
  - `als_exp2/5/10, als_last10` 37.9-38.2%, `als_et1_exp5` 38.8%, `als_samecat_exp10` 47.3%. Target rate on NaN rows is 0.0115-0.0126 vs 0.0158 overall. `als_has` flags only 37.2%, so about 10% of `als_samecat` missingness is unflagged.
  - `seq_*` and `ctx_nd_*h*` 1.4% (no-history feeds, target rate 0.0119).
  - Today all NaNs become the val mean, with no indicator.
- `hour` gets `log1p` (max 23 > 20). It is cyclical, so log1p(hour) is a poor encoding.
- `it_hist_all_luo_gap` ranges over [−4960, 0] (q01 = −2847). It is not log-transformed because min < 0, so it enters as a heavy-tailed linear value.
- Zero-inflated counts: `seq_cv_*` 89-96% zeros, `it_tr_pos` 80%, `it_hist_et*_luo` 41-96%.

---

## 2. Source-by-source: math, evidence, transfer

### 2.1 TabM: parameter-efficient ensembling (Gorishniy, Kotelnikov, Babenko; ICLR 2025)
Sources [V]: https://arxiv.org/abs/2410.24210 (v3, 18 Feb 2025; PDF text), https://github.com/yandex-research/tabm (README, `tabm.py` v0.0.3).

**Math** (BatchEnsemble, §3.2):
- Member i of a linear layer: `l_i(x_i) = s_i ⊙ (W (r_i ⊙ x_i)) + b_i`, equivalently `W_i = W ⊙ (s_i r_iᵀ)`.
- Batched over k members: `l_BE(X) = ((X ⊙ R) W) ⊙ S + B`, with X ∈ R^{k×d}.
- Overhead: 3d parameters per layer per member.

**Variants:**
- TabM_packed: k independent MLPs.
- TabM_naive: adapters in all N layers, random ±1 init.
- **TabM_mini**: only the *first* adapter R.
- **TabM**: all adapters, but every R and S except the first initialised to 1. At init it behaves like TabM_mini.
- With feature embeddings (TabM†), the first R is initialised from N(0,1).

**Training and inference:**
- **"The mean loss must be optimized, not the loss of the mean prediction"** [V README].
- At inference, average the k predictions ("probabilities should usually be averaged, not logits" for classification) [V README].
- Default k = 32, not tuned [V].
- "Too high values of k can be detrimental"; narrow (d=64) or shallow (n=1) backbones hold fewer useful members [V §5.3].
- The optimal LR is higher than for an MLP (effectively a larger batch) [V A.3].

**Shared batches** (all members see the same mini-batch): "only minor performance loss on average" [V §3.4]. This is the cheap way.

**Why it works** [V §5.1-5.4]:
- Members are individually overfit; the collective generalises better.
- "Even the best submodel of TabM is no better than a simple MLP."
- Dead neurons drop from 0.29 to 0.14 (k=1 vs 32).
- The first adapter is critical: TabM_bad (all adapters except the first) is worse.

**Constraint that matters for us** [V A.4]:
- "The k prediction branches should not interact with each other (e.g. through attention, pooling, etc.) and should always be trained separately."
- For normalisation layers, add per-member affine adapters after the norm (FiLM-Ensemble).

**Evidence:**
- 46 datasets: TabM is the top DL model; TabM†mini has mean rank 1.7 ± 1.2 [V Fig. 3].
- MSLR-WEB10K ("microsoft", LTR, 723K train rows, 136 features), RMSE ↓, single / 5-seed ensemble [V Table 19]:
  - MLP 0.7475 / 0.7460
  - MLP‡ (PLR) 0.7444 / 0.7429
  - MLP† (PLE) 0.7465 / 0.7448
  - **TabM 0.7432 / 0.7426**
  - TabM†mini 0.7423 / 0.7416
  - FT-T 0.7460 / 0.7422
  - DCN2 0.7499
  - TabR 0.7503
  - CatBoost 0.7412 / 0.7406
- **A single TabM beats a 5-seed MLP ensemble here** (−0.0043 vs −0.0015 RMSE).
- TabReD (industrial, time splits), e.g. maps-routing: MLP 0.1625, TabM 0.1612 (♠ shared batches), XGBoost 0.1616 [V Table 20]. Small but consistent.

**Transfer to our ranker:**
- (a) **TabM-mini over the candidate pathway (T6).**
  - Replicate the candidate tokens k=4 times; members share every weight.
  - Member i gets a first adapter `r_i ⊙ [e_c, e_c⊙u]` (random ±1, or N(0,1) when T3 embeddings are used) and `r'_i ⊙ X` on the dense input, before `c_lin`/`c_mlp` mix features.
  - The history encoder (candidate-independent) runs once.
  - The cross-attention and set-attention blocks run on the k×-sized candidate batch. Each member's candidates attend only to the same feed's (shared) history, and set attention runs per member.
  - Loss = (1/k) Σ_i listwise CE(logit_i) (+ aux). Inference = mean of per-feed log-softmax or z-scored member logits.
  - Cost ≈ k× the candidate-side compute: 15-20 min becomes about 50-70 min for k=4.
  - Our 16-seed ensemble gave only +0.0005 over the best single model. That bounds the pure variance-reduction part. TabM's MSLR gain beyond seeds came from regularisation, so I expect +0.0005 to +0.0015 at best. Medium/low priority because of cost.
- (b) **TabM head (T7)**: k=16 BatchEnsemble copies of `self.head` with random-sign `r_i` on `LN(c)`. Nearly free, but the first-adapter evidence says diversity comes from mapping *inputs* into k spaces before mixing; the head sees an already-mixed token. Expect ≤ +0.0005. Use it mainly as a free extra ensemble member.

```python
class BEHead(nn.Module):  # TabM-style head: k members, shared W, per-member r,s,b (fp32 like the current head)
    def __init__(self, d, k=16):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.r1 = nn.Parameter(torch.randint(0, 2, (k, d)).float() * 2 - 1)   # random signs = the critical first adapter
        self.W1 = nn.Linear(d, d, bias=False)
        self.s1 = nn.Parameter(torch.ones(k, d)); self.b1 = nn.Parameter(torch.zeros(k, d))
        self.W2 = nn.Linear(d, 1, bias=False); self.b2 = nn.Parameter(torch.zeros(k))
    def forward(self, c):                               # c [B,M,d] -> [k,B,M]
        x = self.ln(c)[None] * self.r1[:, None, None]
        h = F.gelu(self.W1(x) * self.s1[:, None, None] + self.b1[:, None, None])
        return self.W2(h).squeeze(-1) + self.b2[:, None, None]
# loss = mean_i listwise_loss(logit[i] + wide, y, m, ...); predict = logit.mean(0) + wide
```

### 2.2 Numerical feature embeddings (Gorishniy, Rubachev, Babenko; NeurIPS 2022) and the `rtdl_num_embeddings` package
Sources [V]: https://arxiv.org/abs/2203.05556 (PDF text); https://github.com/yandex-research/rtdl-num-embeddings (`package/README.md`, `rtdl_num_embeddings.py` v0.0.12).

**PLE (Eq. 1)**, with bins `[b_{t−1}, b_t)`, t = 1..T:
`e_t = 0` if `x < b_{t−1}` and t > 1; `e_t = 1` if `x ≥ b_t` and t < T; otherwise `e_t = (x − b_{t−1})/(b_t − b_{t−1})`.
- Feature embedding: `f(x) = v_0 + Σ_t e_t v_t = Linear(PLE(x))`.
- Bins come from quantiles ("Q") or from a single-feature decision tree ("T", target-aware).

**Periodic (Eq. 2):** `Periodic(x) = concat[sin(v), cos(v)]` with `v = [2πc_1x, …, 2πc_kx]`, trainable `c ~ N(0, σ)`.
- PLR = `ReLU(Linear(Periodic(x)))`. "PLR-lite" shares the outer Linear across features.
- **σ is the important hyperparameter**. Package defaults: `n_frequencies=48`, `frequency_init_scale=0.01` (truncated at 3σ), `d_embedding=24`.
- README: the optimal σ is < 0.05 for 50% of tasks and < 0.5 for 75%. "Prioritize testing smaller values."

**Package recommendation:** `PiecewiseLinearEmbeddings(bins, d_embedding=12, activation=False, version="B")`.
- Version B = `LinearEmbeddings(x)` + `NLinear(PLE(x))`, with **the PLE weights zero-initialised**: "at initialization, the whole embedding behaves like a linear embedding. The piecewise-linear component is incrementally learnt during training" [V source].

**Evidence:**
- MLP-PLR has the best average rank (3.0 vs MLP 8.5, CatBoost 3.6) over 11 GBDT-friendly datasets [V Table 6].
- On MI (= MSLR-WEB10K), single model: MLP 0.747, MLP-PLR 0.746, MLP-Q-LR 0.747 [V Table 4/5]. Ensembles: MLP 0.746 vs MLP-PLR 0.743 [V Table 6]. **Tiny gains on the LTR dataset**; big gains only on CA/AD/CO-type data.

**Transfer (T3):**
- Replace `c_mlp(X)` input by `flatten(NumEmb(X))`, 63 × d_e = 12 → 756 inputs. `Linear(756→256) → GELU → Linear(256→128)` keeps the rest unchanged.
- Bins: per-feature quantiles (T=16) from the same 40 `feature_stats` batches, on standardised X. For f1..f5, use instead the **natural grid** {k/N} (N = 28/64/31/32/31, §1.1), so each original bin gets its own vector.
- Store the bins as a buffer and put `n_bins, d_e` into `cfg` so `predict`/`finetune` rebuild the shapes.
- Add **10 explicit flags** `1[f_k = floor_k]`, `1[f_k = ceil_k]` to `dense_features`. PLE alone cannot represent a point mass, because zero-width quantile bins are removed. Values (train min/max, exact):
  - floor = (1/56, 0, 1/62, 1/64, 0)
  - ceil = (55/56, 63/64, 61/62, 63/64, 61/62)
  - Use `x <= floor + 1e-6`.
  - Because of the f4 floor shift, consider leaving the f4 flags out of the train stage and letting fine-tune learn them through the extra branch.
- Cost: a [32768, 63, 16] PLE tensor (≈66 MB in bf16) plus an einsum. Negligible next to attention.
- Warm-start variant for stage 2 or fine-tune: keep `c_mlp(X)` and add a residual `c += Z(GELU(Linear(flatten(PLE(X)))))` with Z zero-initialised. It is exact at init.
- A/B plan: (i) PLE-B T=16 d_e=12; (ii) PLR-lite with k=24 frequencies, σ=0.02, d_e=12. 2 seeds each, paired with 2 baseline seeds; judge on both val and the 30K holdout.
- Expected +0.0003 to +0.0015 train-only. f1..f5 already enter three times (raw, within-feed z, within-feed mid-rank percentile), and the network has plenty of downstream nonlinearity. The likely gain is the sharp point-mass and non-monotone structure from §1.1.

```python
class NumEmb(nn.Module):
    """rtdl 'version B' piecewise-linear embedding: linear term + zero-init PLE term. x [...,F] -> [...,F,d_e]."""
    def __init__(self, edges, d_e=12):                  # edges [F, T+1], standardized units, strictly increasing
        super().__init__()
        F_, T1 = edges.shape
        self.register_buffer("lo", edges[:, :-1].clone())
        self.register_buffer("inv", 1.0 / (edges[:, 1:] - edges[:, :-1]).clamp_min(1e-4))
        self.w0 = nn.Parameter(torch.empty(F_, d_e).uniform_(-1, 1)); self.b0 = nn.Parameter(torch.zeros(F_, d_e))
        self.wp = nn.Parameter(torch.zeros(F_, T1 - 1, d_e))           # zero init: starts as a linear embedding
    def forward(self, x):
        e = (x[..., None] - self.lo) * self.inv                         # [...,F,T]
        e = torch.cat([e[..., :1].clamp_max(1), e[..., 1:-1].clamp(0, 1), e[..., -1:].clamp_min(0)], -1)
        return x[..., None] * self.w0 + self.b0 + torch.einsum("...ft,fte->...fe", e, self.wp)
```

### 2.3 RealMLP: "Better by Default" (Holzmüller, Grinsztajn, Steinwart; NeurIPS 2024)
Source [V]: https://arxiv.org/abs/2407.04491 (PDF text).

**Math:**
- **Preprocessing**: `x ↦ f(s·(x − q_{1/2}))` with `f(x) = x/√(1+(x/3)²)` (a smooth clip to (−3, 3)).
  - `s = 1/(q_{3/4} − q_{1/4})`, or `2/(q_1 − q_0)` if the IQR is 0, otherwise 0.
- **PBLD embedding**: `[x_i, W^{(2,i)} cos(2π w^{(1,i)} x_i + b^{(1,i)}) + b^{(2,i)}] ∈ R⁴`, with 16 frequencies.
- **Scaling layer** `x_i ← s_i x_i`: `s_i` initialised to 1, with a higher LR factor (6 in RealMLP-TD; ×96 effective in RealTabR-D).
- **NTP** `z = d_l^{−1/2} W x + b`.
- **Parametric activation** `σ_α(x) = (1−α)x + α σ(x)`, α initialised to 1 (SELU for classification, Mish for regression).
- AdamW **β2 = 0.95**, 256 epochs, batch size 256.
- **coslog_k(t) = ½(1 − cos(2π log2(1 + (2^k − 1)t)))** with k = 4 cycles. flat_cos schedules for dropout and weight decay. Best-epoch selection is the *last* of tied best epochs.
- Label smoothing ε = 0.1.

**Ablation** (Table B.1, error increase when the component is removed, classification / regression):
- PBLD → PL: 0.7 / 0.5; → PLR: 4.2 / 19.0; → none: 2.3 / 20.6.
- β2 = 0.999: 2.0 / **22.8**.
- Cosine instead of multi-cycle LR: 1.1 / 0.4. Constant LR: 1.8 / 13.5.
- No label smoothing: 1.8 (classification).
- No scaling layer: 1.4 / 1.0.
- Non-parametric activation: 0.5 / 4.8.
- Standard parametrisation instead of NTP: "insignificantly worse".
- No data-dependent init: 0.9 / 1.2.

**Preprocessing** (Table B.2, error increase vs robust+smooth clip, classification / regression):
- robust only: 0.5 / 9.5
- standardise + smooth clip: 1.6 / 1.2
- standardise: 2.1 / 8.8
- quantile→normal: 2.3 / 6.3
- RTDL quantile: 2.6 / 2.6

**Caveat** [V §5]: "label smoothing is influential but can be detrimental for metrics like AUROC". **NDCG is a ranking metric like AUROC, so skip label smoothing.**

**Bagging vs refit** (RealMLP-TD, Table B.4, error reduction vs one fold, meta-train class):
- bagging, 5 models: 6.7%
- refitting, 5 models: 7.6% (individual stopping) / 8.2% (joint stopping)
- refitting, 1 model with joint stopping: 5.3%

**Transfer:**
- **T2 (fine-tune optimiser).** `finetune.py` uses AdamW defaults (β2 = 0.999) and a constant LR. Change to `betas=(0.9, 0.95)` (LLM pretraining also uses 0.95 [K]). Use a per-epoch cosine cycle so every snapshot is taken at a cycle minimum (snapshot ensembles, Huang et al. 2017 [K]; RealMLP's multi-cycle schedule is the same idea):
  `lr(t) = lr_max · mult · min(1, (step+1)/warmup) · (ε + (1−ε)·½(1 + cos(π·τ)))`, with `τ = (step within epoch)/(steps per epoch)` and ε = 0.1.
  Our snapshot mean already gives +0.0017 over the best single epoch. Better-converged and more diverse snapshots should add a little. Run it paired (same folds and seeds) against the current recipe.
- **T5 (refit).** For the *test* prediction, add a refit on 100% of val (14.4K vs 11.5K feeds per model) with the fixed snapshot epochs 1..6 (joint stopping = our fixed recipe). Use 3-5 seeds and average 50/50 with the 5-fold test predictions. It cannot be validated on val; use it only with the fixed recipe.
- **T12 scaling layer** on the extra branch (67 features, small data): `xe ← s ⊙ xe` with s = 1 at init and an LR multiplier ×10 on s. It acts as soft feature selection. Cheap; expected ≈ 0.
- **Not worth it for the 61 dense features**: robust scaling / smooth clip (our profile shows nothing to fix), NTP, parametric activations, data-dependent init (our net trains on 3.15M feeds with LayerNorm everywhere).

### 2.4 TabR (Gorishniy et al.; ICLR 2024): retrieval with label-aware values
Source [V]: https://arxiv.org/abs/2307.14338 (HTML).

**Math (Eq. 5):**
- Keys `k = W_K(x̃)`, `k_i = W_K(x̃_i)`, with **no separate query projection**.
- Similarity `S = −‖k − k_i‖²`, not scaled by d^{−1/2}. Use the top-m (m = 96) candidates.
- **Value** `V = W_Y(y_i) + T(k − k_i)`, with `T = LinearNoBias(Dropout(ReLU(Linear(·))))`. Dropout is applied to the attention weights.

**Ablation (Table 2):**
- Vanilla attention ≈ MLP.
- Adding labels to values: no gain.
- **Key-only L2 similarity is "a turning point".**
- `T(k − k_i)` in the value: further gains.
- Removing labels, key-only keys or L2 drops performance back to MLP level.

**Limitations:**
- "May not scale to truly large datasets as-is."
- Training-time retrieval must mirror deployment ("for time series, retrieve only from the past").
- On TabReD, retrieval models lose their edge under temporal shift [V TabReD abstract; V TabM Table 20, where TabR is worse than MLP on homesite and ecom].

**Our history attention is already TabR-like**: the candidates are queries, the user's history events are the "labelled context", and event type/source play the role of the "label" y. The +0.0024 from the raw-cosine sim bias matches TabR's "similarity structure is the turning point". **What we lack is TabR's value design.** Our values `V(h_j)` do not depend on the candidate. Softmax also removes the absolute similarity level, because it is shift-invariant: Σ_j p_j V_j cannot tell "everything is moderately similar" from "one exact match" unless V carries it. DIN explicitly drops the softmax constraint "to reserve the intensity of user interests" [V DIN §4.3, https://arxiv.org/abs/1706.06978].

**Transfer (T4)**, a retrieval branch added before the cross blocks:
- For each candidate, take the top-m (m = 16-32) history items by the already-computed raw cosine `S`.
- `k_c = W_K e_c`, `k_j = W_K e_j` (shared, dk = 64).
- logits `−‖k_c − k_j‖² + a·S_cj + β_{type_j}`, plus **one learned sink logit** with value 0 (the "no match" option, which restores intensity).
- `v_cj = emb_type(type_j) + emb_src(src_j) + T(k_c − k_j)`.
- `r_c = Σ_j α_cj v_cj`, then `c ← c + Z(LN(r_c))` with Z zero-initialised (exact warm start).
- Cost: [B, M, m, dk] = 32768 × 16 × 64 ≈ 34M elements per batch, the same order as the existing [B, h, M, K] attention scores.
- There is no label leakage: the context is the user's own past, identical at train and test.

```python
def topm_gather(t, idx):            # t [B,K,D], idx [B,M,m] -> [B,M,m,D]
    B, K, D = t.shape
    flat = (idx + torch.arange(B, device=t.device)[:, None, None] * K).reshape(-1)
    return t.reshape(B * K, D)[flat].view(*idx.shape, D)
# S: [B,M,K] raw cosine (masked to -9 where invalid); top, idx = S.topk(m, -1); valid = top > -8
# kc = Wk(ce)[:, :, None]; kj = topm_gather(Wk(he), idx); logits = -((kc - kj)**2).sum(-1) + a*top + beta[topm_gather(het)]
# append sink logit s0 (learned scalar) with zero value; alpha = softmax(masked logits); v = Ety + Esrc + T(kc - kj)
```

### 2.5 ModernNCA (Ye, Yin, Zhan, Chao; ICLR 2025)
Source [V, via the HTML]: https://arxiv.org/abs/2407.03257.

**Math:**
- `p_ij = exp(−d(φ(x_i), φ(x_j))) / Σ_{l≠i} exp(−d(φ(x_i), φ(x_l)))` and `ŷ_i = Σ_j p_ij y_j`.
- Classification loss = −log Σ_{j: y_j = y_i} p_ij (the log version beats the original NCA sum).
- φ = Linear, then MLP blocks with BatchNorm, then PLR-lite. d = Euclidean by default.
- **Stochastic Neighbourhood Sampling (SNS)**: each batch uses a random 5-60% subset of the training set as candidate neighbours. 30-50% beats the full set (qualitative figure); inference uses all.

**Evidence:**
- 300 datasets: average rank ModernNCA 4.56, CatBoost 4.61, TabR 5.36, FT-T 6.29, MLP 7.53.
- Distance ablation (rank, classification / regression): cosine 2.11 / 2.53, Euclidean 2.59 / 2.50, dot product 3.85 / 3.22.

**Transfer:**
- Full NCA over 247M train candidates is out of scope. The neighbour-label signal already exists as features (ctx_nd_*, it_tr_*) in the fine-tune branch.
- Transferable details:
  - (i) cosine/L2 beat dot product for similarity, which supports raw-cosine biases (done) and the L2 keys in T4;
  - (ii) SNS → **history-key dropout**: randomly mask 10-30% of history tokens per feed in training (`hmask &= rand > p`). It is a cheap regulariser and robust to the H-distribution shift. Expected ±0.0005; low priority.
- TabArena notes that ModernNCA and RealMLP are over-represented in ensembles "likely due to validation overfitting" [V TabArena].

### 2.6 FT-Transformer (Gorishniy et al.; NeurIPS 2021)
Source [V for the limitation, K for the formula]: https://arxiv.org/abs/2106.11959.

- Feature tokenizer `T_j = b_j + x_j W_j` [K], plus [CLS]; a Transformer over feature tokens.
- "Requires more resources … quadratic complexity of the vanilla MHSA with respect to the number of features" [V].
- With numerical embeddings, "MLP-like architectures … perform on par with the Transformer-based models" [V, embeddings paper §4.7]. On MSLR, FT-T 0.7460 vs MLP-PLR 0.7444 [V].
- **For us**: 63 tokens per candidate × 32K candidates per batch is ~2M tokens, for no expected gain. **Skip.**

### 2.7 TabPFN v2 / 2.5, TabICL / TabICLv2: in-context tabular foundation models
Sources [V]:
- TabPFN v2: Hollmann et al., Nature 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/
- TabPFN-2.5: https://arxiv.org/abs/2511.08667 (v2, Feb 2026)
- TabICL: https://arxiv.org/abs/2502.05564 (ICML 2025)
- TabICLv2: https://arxiv.org/abs/2602.11139 (Feb 2026)
- BeyondArena: https://arxiv.org/abs/2606.30410 (NeurIPS 2026 D&B)

**TabPFN v2:**
- One representation per *cell*. Attention alternates between the row (features) and the column (samples). **"Test samples … only attend to the training data."**
- About 130M synthetic SCM datasets.
- Default ensemble of 4 (regression 8) preprocessing variants (quantile, power, SVD, feature shuffles).
- Limits 10K samples / 500 features.

**TabPFN-2.5:**
- 18-24 layers; features in groups of 3; 50K rows / 2K features recommended (about 160K × 500 is possible on an H100).
- **64 learned "thinking rows"** that "provide extra computation and can act as attention sinks".
- Preprocessing ensemble includes **robust scaling with soft clipping**, quantile and standard scaling.
- Distillation to MLP/trees ("TabPFN-as-MLP") for large or low-latency use.

**TabICL:**
- Column-then-row embedding, then an ICL transformer.
- Pretrained up to 60K rows; handles 500K rows; up to 10× faster than TabPFNv2.

**TabICLv2:**
- Beats RealTabPFN-2.5 on TabArena/TALENT without tuning; 1M rows in under 50 GB.
- Three gains of ≈100 Elo / 64% win rate each:
  - **QASSMax scalable softmax**: `q̃ = q · MLP_base(log n) · (1 + tanh(MLP_gate(q)))`, extending SSMax `q̃ = q · s_h · log n`;
  - **Muon** instead of AdamW (LR 8e-4, wd 0.01, clip 10, per-parameter LR × 0.2·√max(n, m));
  - early target inclusion.

**BeyondArena** (11 models, 142 datasets):
- TFMs "are competitive or excel on datasets up to 100k rows". TabICLv2 is rank 1 on 19% of datasets.
- **On grouped, temporal and large (>100K) data, tuned GBDT/RealMLP/TabM dominate**. In a multivariate analysis, only ">100k training rows and grouped splits" were significant.
- Using IID validation for grouped data "can drastically distort model rankings" (Kendall τ 0.49 / 0.60).

**Transfer:**
- *As models or stackers*: no.
  - Our train set is 247M rows with a production→exploration shift.
  - Val stacking would be 1.39M rows and grouped by user (exactly where BeyondArena says TFMs lose).
  - They are pointwise, with no within-feed context.
  - The GBDT stackers already add ≈ 0 over the NN.
- *Architectural ideas*:
  - (i) **"test attends only to train"** is what our target attention already is: candidates attend to labelled history events. TabICL's "early target inclusion" corresponds to our event-type embedding added to history tokens early (done).
  - (ii) **thinking rows → T9 register tokens**: R = 4 learned tokens appended to the cross-attention context (always unmasked, sim bias 0) and to set-attention keys. This gives "attend to nothing" for H = 0 feeds (1.6% of candidates [D]) and weak matches.
  - (iii) **SSMax → T8**: `q ← q · s_h · log(n_valid)`. Initialise s_h = 1/log(n̄) so warm starts are unchanged (n̄ ≈ mean H+1 for cross, ≈ 96 for set). Our n ≤ 300, so attention fading is mild: ±0.0005.
  - (iv) **Muon → T11**: low priority; it belongs to the LLM-tricks track.
- Our validation already follows the BeyondArena advice: the GroupKFold by user and the random-like train holdout are non-IID-aware.

### 2.8 Benchmarks: TabArena, TALENT, TabReD (what matters)
- **TabArena** (NeurIPS 2025 D&B; v4, Nov 2025; 51 datasets of 500-250K rows) [V https://arxiv.org/abs/2506.16791]:
  - CatBoost is first under conventional tuning.
  - "After post-hoc ensembling, neural networks are the strongest single models": top-3 TabM, LightGBM, RealMLP, and "without ensembling all three would be worse than … CatBoost".
  - "Peak performance requires post-hoc ensembling".
  - Holdout validation "greatly underestimates all models".
  - 8-fold CV ensembles.
- **TALENT** (300+ datasets) [V https://arxiv.org/abs/2407.00956]: ensembling helps both families; top performance is concentrated in a few models; "the interplay of categorical and numerical features" decides DL vs trees.
- **TabReD** (8 industrial datasets with time splits) [V https://arxiv.org/abs/2406.19380]: "simple MLP-like architectures and GBDT perform best", and time-based splits re-rank methods.
- **Implication for us**: keep the MLP + target-attention core. Spend on (implicit) ensembling and on the fine-tune stage. Architecturally different members (PLE, TabM, T4 variants) are better ensemble additions than seeds, which saturate (+0.0005 for 16 runs).

### 2.9 Mambular, MoE, DCNv2 (checked, not recommended)
- **Mambular** [V abstract, https://arxiv.org/abs/2408.06291]: features as a sequence through Mamba blocks. Our features are an unordered set, and our history (≤ 256 events) is cheap for attention. No evidence that an SSM beats target attention for candidate-conditioned history. Skip.
- **Tabular MoE**: "GG MoE" (Gumbel-gated, arXiv 2502.03608) reports the best mean results on 38 datasets, but differences vs an MLP with embeddings were "not always statistically significant" [S]. Our "domain token" and "category-intent" variants were neutral. Skip.
- **DCNv2** `x_{l+1} = x_0 ⊙ (W x_l + b) + x_l` [K]: MSLR 0.7499 vs MLP 0.7475 [V TabM Table 19]. Skip.

---

## 3. T1 in detail: extra-branch preprocessing (bug fix + best practice)

Proposed `attach_extra` (fine-tune only; val stats reused for test):
1. `X[~np.isfinite(X)] = np.nan` **before** any statistic. This fixes `als_et1_exp5`, which is constant 0 today.
2. For each column with NaN share > 0.5%, append `m_j = 1[isnan(x_j)]`. Affected: the 6 `als_*` columns plus the `seq_*`/`ctx_nd_*h*` group; one shared indicator per group is enough, so about 3 new columns. Then impute NaN with the median (val) and standardise.
3. `hour` → `[sin(2πh/24), cos(2πh/24)]` (a periodic encoding; Time2Vec idea [K]) instead of `log1p(h)`.
4. Signed log for heavy-tailed signed columns: `sign(x)·log1p(|x|)` for `*_gap` (e.g. `it_hist_all_luo_gap`).
5. Optional: `1[x = 0]` flags for zero-inflated counts (`seq_cv_*`, `it_tr_pos`, `it_hist_et*_luo`), or PLE (T3) on the extra branch with T = 8-16.

Run with the same checkpoint, folds, seed and lr as `s2_v2_feed_k9` n2full (0.38572 OOF), and compare paired.

---

## 4. Suggested GPU-5 experiment queue (for the experiment agents; ≥ 2 seeds paired, judge val and holdout)

1. T1 + T2 as fine-tune-only A/B runs (cheapest; fine-tune runs on val/test only).
2. T3 PLE-B + flags vs PLR-lite vs baseline (2 seeds each, train stage 2 epochs, then stage 2 + n2full fine-tune for the winner).
3. T4 retrieval branch (m = 16, sink) vs baseline (2 seeds). If positive, try m = 32 and putting it after block 1.
4. T8 + T9 (SSMax + registers) together in one run (both cheap), with an ablation only if positive.
5. T6 TabM-mini k = 4 only if time remains (about 3-4× train cost).
6. Final: add every architecturally distinct variant that is not worse than the baseline to the ensemble (per-feed z-mean), because diversity beats seeds here.

Noise reminders (from the task brief): one val score has SE ≈ 0.0018; a paired difference has SE ≈ 0.0005-0.0013; seed spread is 0.001-0.002. A trick needs at least 2 seed pairs that agree on both val and the 30K holdout.

## 5. Source list
- TabM: https://arxiv.org/abs/2410.24210 ; https://github.com/yandex-research/tabm [V]
- Numerical embeddings: https://arxiv.org/abs/2203.05556 ; https://github.com/yandex-research/rtdl-num-embeddings [V]
- RealMLP: https://arxiv.org/abs/2407.04491 [V]
- TabR: https://arxiv.org/abs/2307.14338 [V]
- ModernNCA: https://arxiv.org/abs/2407.03257 [V]
- FT-Transformer: https://arxiv.org/abs/2106.11959 [V/K]
- TabReD: https://arxiv.org/abs/2406.19380 [V]
- TabArena: https://arxiv.org/abs/2506.16791 [V]
- TALENT: https://arxiv.org/abs/2407.00956 [V]
- TabPFN v2 (Nature 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/ [V]
- TabPFN-2.5: https://arxiv.org/abs/2511.08667 [V]
- TabICL: https://arxiv.org/abs/2502.05564 [V]
- TabICLv2: https://arxiv.org/abs/2602.11139 [V]
- BeyondArena: https://arxiv.org/abs/2606.30410 [V]
- Mambular: https://arxiv.org/abs/2408.06291 [V]
- GG MoE: https://arxiv.org/abs/2502.03608 [S]
- DIN: https://arxiv.org/abs/1706.06978 [V]
- SSMax (Nakanishi 2025, arXiv 2501.19399) [S, via TabICLv2]
- Snapshot ensembles (Huang et al. 2017, arXiv 1704.00109) [K]
- Time2Vec (arXiv 1907.05321) [K]
- Measurement scripts (CPU only): `/tmp/scratchpad/scripts/dense_profile.py`, plus the inline numpy checks described in §1.1.
