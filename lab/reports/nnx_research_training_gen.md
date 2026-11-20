# Training and generalization tricks for the feed ranker (2024-2026 literature to `nn/`)

Research note, no GPU experiments. Written 2026-11-11 against `nn/model.py`, `nn/train.py` and `nn/finetune.py` as they stand on disk, plus `reports/nn_report.md` and the training logs.

**Evidence tags**
- **[V]**: I read the primary source in this session (URL in the Sources section).
- **[S]**: search-result snippet only.
- **[K]**: background knowledge, not re-read here.
- **[L]**: measured from this project's own logs or code.

**Gains** are in val full NDCG. The noise floor is a paired SE of 0.0005-0.0013 per comparison and a seed spread of 0.001-0.002. So almost every trick below is "+0 to +0.001", and each needs at least 2 paired seeds.

---

## 0. TL;DR: what to try, in order

| # | Trick | Stage | Cost | Expected Δ val | Priority |
|---|---|---|---|---|---|
| 1 | **Weight EMA / LAWA inside fine-tune**, combined with a short LR decay. This replaces the epoch-snapshot mean of predictions. | fine-tune | 0 (4 min per test) | +0.0003 to +0.0012 OOF | high |
| 2 | **LP-FT**: train the zero-init extra branch (plus head and wide) with the backbone frozen for 1 epoch, then fine-tune everything. | fine-tune | 0 | +0.0003 to +0.0012 OOF | high |
| 3 | **EMA of weights in stage 1 and stage 2** (β ≈ 0.9995 per step), evaluated next to the raw weights on the holdout. | train | ~0 | +0.0000 to +0.0005, mostly lower checkpoint noise | high (free) |
| 4 | **Under-training probe**: 3 epochs with the same cosine, or WSD with a 20% 1-sqrt cooldown that mixes the exploration feeds into the cooldown (the MiniCPM "decay-phase data" recipe). Val is still rising at the last step. | train | 1.5x | -0.0005 to +0.0015 | high |
| 5 | **Muon** (`torch.optim.Muon`, `adjust_lr_fn="match_rms_adamw"`, already in the local torch 2.11) for the hidden 2-D matrices; AdamW for everything else. | train (+ fine-tune with the same optimizer) | ~1.05x | +0.0000 to +0.0015 | high/medium |
| 6 | **Attention-probability dropout 0** (keep residual/FFN dropout), and a `--drop 0.05/0` arm for stage 1. | train | 0 | -0.0005 to +0.0010 | medium |
| 7 | **Born-again self-distillation**: listwise KL toward a frozen train-only teacher (s2_v2_feed_k9) on top of the label loss and K9. | train | 1.3x | +0.0000 to +0.0012 | medium |
| 8 | **10-fold fine-tune**, then a full-val refit for test. | fine-tune | 2x | +0.0003 to +0.0008 (test) | medium |
| 9 | **R-Drop** or **SAM/ASAM** in fine-tune only (2 forward passes on small data). | fine-tune | 2x | +0.0000 to +0.0008 | medium |
| 10 | **TabM-style BatchEnsemble** (k = 4-8 implicit members; head-only, or candidate side). | train | 1.2-3x | +0.0000 to +0.0010, plus ensemble diversity | medium |
| 11 | **Periodic (PLR/PBLD) numerical embeddings** for the 61 dense features. | train | ~1.05x | +0.0000 to +0.0010 | medium/low |
| 12 | **WiSE-FT** (interpolate train-only and fine-tuned weights) and **decoupled L2-SP** (decay toward θ0). | fine-tune | 0 | +0.0000 to +0.0005 | low/medium |
| 13 | Head-specific **gated attention output** (Qwen) or **FoX-style data-dependent recency decay** in target attention. | train (architecture) | ~1.02x | +0.0000 to +0.0007 | low/medium |
| 14 | Schedule-Free AdamW, AdEMAMix, SOAP/Shampoo, Lion, LAMB. | train | varies | ≈0 versus a tuned AdamW+cosine at this scale | low |
| 15 | Label smoothing, mixup/cutmix, stochastic depth, MoE, z-loss, candidate-permutation TTA. | - | - | ≤ 0, or no mechanism here | do not prioritise |

**Two structural facts from the logs [L] drive most of the ranking:**

1. **Stage 1 is still improving when the LR reaches its floor.** In `v2_feed_k9.log`, val goes 0.3795 (10k) → 0.3815 (14k) → 0.3819 (16k) → 0.3818 (19k) → 0.3819 (19.67k). The holdout goes 0.2979 → 0.2993 → 0.2998 → 0.3009. Most of the gain arrives during the anneal. This regime rewards a better optimizer (Muon), more steps and averaging; it does not reward capacity (d=256, 3 blocks and 8 heads were all neutral).
2. **The fine-tune overfits after 1-2 epochs at a constant LR of 5e-5.** OOF by epoch was 0.38271 → 0.38593 → **0.38611** → 0.38553 → 0.38449 → 0.38418 → 0.38349, and the snapshot mean of epochs 1-6 was 0.38572 (`s2_v2_feed_k9_n2f_ft.log`). A constant LR with no averaging of weights is exactly where averaging gives "substantial improvements" [V Ajroldi 2025]. The measured +0.0017 from the snapshot mean already confirms this. Better averaging plus a decay should at least recover the best-epoch score without selecting an epoch.

---

## 1. Diagnosis of the current recipe (what the code actually does) [L]

| Item | Current | Comment |
|---|---|---|
| Optimizer, stage 1/2 | AdamW, lr 1e-3, betas (0.9, 0.98), wd 0.02, clip 1.0 | wd goes to the `ndim>=2` params whose name lacks "emb". **So `cat`, `et`, `es`, `rec` and `dom` are decayed** (their names don't contain "emb"); only `pos_emb`, `rev_emb` and `id_emb` are excluded. This is harmless at this wd (see the next row), but it is probably not what was intended. |
| wd strength | EMA-timescale view [V Wang & Aitchison 2024 abstract; formula K]: AdamW θ_t = (1-η_tλ)θ_{t-1} - η_t u_t, so the weights are an EMA of updates with timescale τ = 1/(ηλ) = 1/(1e-3·0.02) = **50k steps**, against 19.7k total steps. Integrated shrinkage is Σ_t η_t λ ≈ 0.02·Σ_t η_t ≈ 0.2, i.e. e^-0.2 ≈ 0.82 for directions that receive no gradient. | In the fine-tune, the shrinkage is 0.02·5e-5·~800 steps ≈ 8e-4, so **weight decay does nothing there**. Any regularization in the fine-tune comes from early stopping and averaging. |
| Optimizer, fine-tune | AdamW with default betas (0.9, 0.999), wd 0.02 on all params (LN and biases included), 30 warmup steps, **constant LR**, fresh optimizer state | With ~800 steps, β2 = 0.999 averages almost the whole run. RealMLP found β2 = 0.95 better than 0.999 on small tabular data [V]. Use (0.9, 0.98) for consistency. |
| Schedule, stage 1 | warmup 500, cosine to 3% over 19,670 steps | Fully annealed. Averaging on top adds little in expectation [V Ajroldi; V Hägele]. |
| Dropout | 0.1 on the residual branches, after the FFN activation, **and on the attention probabilities** (`dropout_p` in SDPA), in the history encoder, the cross blocks and the set blocks | The sim-bias heads do near-exact nearest-history-item retrieval, which gives peaked attention. Attention dropout deletes the matched event 10% of the time. Modern LLMs set attention dropout to 0 [K]. |
| Capacity | 0.86M params | d=256 (3.0M) was worse; 3 blocks and 8 heads were neutral [L]. Capacity is **not** the bottleneck, so MoE-style capacity tricks are deprioritised (§8). |
| Ensembling | 16 seeds give only +0.0005 over the best single model [L] | Diversity has to come from the recipe (optimizer, loss, regularizer, architecture), not from seeds [K: Gontijo-Lopes et al. 2022, "No one representation to rule them all"]. |
| Checkpoint selection | stage 1: best of the holdout evals every 1000 steps; stage 2: last step; fine-tune: snapshot mean of epochs 1..E | Good leak hygiene. Keep it: every trick below is designed to be selected on the train holdout (stages 1/2) or with one global choice on OOF (fine-tune). |

---

## 2. Optimizers

### 2.1 Muon (and the "scaled" Moonlight variant): recommended to test

**Math** [V Jordan blog; V Moonlight; V local `torch/optim/_muon.py`]. For each 2-D weight W ∈ R^{A×B}:

```
B_t = μ B_{t-1} + (1-μ) G_t                     (torch uses the lerp/EMA form; μ = 0.95)
Ũ_t = (1-μ) G_t + μ B_t                         (Nesterov)
O_t = NS_5(Ũ_t / ||Ũ_t||_F),  NS step: X <- aX + b(XXᵀ)X + c(XXᵀ)²X,  (a,b,c) = (3.4445, -4.7750, 2.0315)
W_t = (1 - ηλ) W_{t-1} - η · 0.2·sqrt(max(A,B)) · O_t      ("match_rms_adamw", Moonlight)
```

- O_t ≈ UVᵀ, the polar factor of the momentum. This is steepest descent under the spectral norm: every singular direction of the update moves at the same rate.
- The 0.2·sqrt(max(A,B)) factor sets the update RMS to about 0.2, which is the typical AdamW update RMS ("usually around 0.2 to 0.4") [V Moonlight]. As a result, **AdamW's lr and wd can be reused unchanged**.
- Embeddings, the output head, and gains/biases stay on AdamW. Q, K and V should be orthogonalized separately [V Jordan].

**Evidence**
- NanoGPT speedrun: 1.35x [V Jordan].
- Moonlight: about 2x compute efficiency in its scaling-law fits [V].
- The careful re-benchmark of Wen et al. 2025 (ICLR 2026): matrix optimizers (Muon, SOAP) give **about 1.4x at 0.1B params, shrinking to 1.1x at 1.2B**. All tuned scalar optimizers come within 1.2x of AdamW [V]. Rankings can flip before the LR decay finishes, so compare at the end of training [V].
- RecSys: MuonRec (2026) reports 32.4% fewer steps to converge and +12.6% relative NDCG@10 on sequential and generative recommenders [V abstract]. Its Adam baselines may be under-tuned, in the sense Wen et al. warn about.
- Qwen3.8-Next (2026) says Muon together with its architecture "shift[s] the optimal learning rate and batch size upwards" and improves stability under stress tests [V abstract].
- **Fine-tuning mismatch:** AdamW-pretrained models fine-tune poorly with Muon and vice versa [V Moonlight]. Community and 2026 reports say Muon can overfit or memorise more on small SFT sets [S].

**Transfer to `train.py`.** These are drop-in changes; the param groups are explicit.

```python
import torch.nn as nn
muon_w, adam_p = [], []
for mod_name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and mod.out_features > 1:      # hidden matrices incl. c_lin, c_mlp, h_lin, q, kv, o, ff, head[1]
        muon_w.append(mod.weight)
ids = {id(p) for p in muon_w}
adam_p = [p for p in model.parameters() if id(p) not in ids]     # embeddings, LN, biases, head[3] (d->1), wide, bias_scale, c_miss...
opt_m = torch.optim.Muon(muon_w, lr=a.lr, weight_decay=a.wd, momentum=0.95, nesterov=True,
                         adjust_lr_fn="match_rms_adamw")
opt_a = torch.optim.AdamW([{"params": [p for p in adam_p if p.ndim >= 2], "weight_decay": 0.0},   # embeddings: no wd
                           {"params": [p for p in adam_p if p.ndim < 2], "weight_decay": 0.0}],
                          lr=a.lr, betas=(0.9, 0.98))
# each step: set g["lr"] = lr_at(step) in both optimizers; clip_grad_norm_ as now; opt_m.step(); opt_a.step()
```

- **Fused `kv` (256×128):** torch's Muon orthogonalizes it as one matrix. To follow Keller's "separately for Q, K, V", either split `Attn.kv` into `k` and `v` Linears for new runs, or write a tiny Muon wrapper that applies NS to `W.view(2, d, d)` per slice.
- The **fine-tune must then also use Muon** for the same matrices, at lr 5e-5 (×10 for the extra branch, whose matrices also get Muon). Otherwise the mismatch effect can eat the gain. Run both fine-tune optimizers once to check.
- Expected: +0.000 to +0.0015 train-only, largest if stage 1 really is optimization-limited (fact 1 above). Muon members also add **recipe diversity** to the ensemble.
- Risks:
  - The 61→256 dense-input projection is an "input layer". Keller keeps input layers on AdamW in LMs, but those are one-hot embeddings. For a dense projection, Muon is commonly fine. Optional ablation arm: keep `c_mlp.0` on AdamW.
  - Re-tune only lr ∈ {1e-3, 2e-3}, with 2 seeds each.

### 2.2 SOAP / (Distributed) Shampoo: not first

- SOAP runs Adam in the eigenbasis of the Shampoo preconditioners L = Σ G Gᵀ and R = Σ Gᵀ G. The eigenbases are refreshed every f steps, so the only new hyperparameter is the preconditioning frequency [S].
- Distributed Shampoo won the AlgoPerf external-tuning track and Schedule-Free AdamW the self-tuning track [V AlgoPerf 2025].
- Wen et al.: SOAP and Kron overtake Muon only at data-to-model ratios ≥ 8x Chinchilla [V]. Ours is enormous (247M candidate rows for 0.86M params), so in principle SOAP's regime. But for 128×128 matrices, Muon and SOAP should land within noise of each other, and SOAP needs an external implementation (`github.com/nikhilvyas/SOAP`).
- **Priority low.** Test only if Muon helps and there is spare time.

### 2.3 Schedule-Free AdamW: low

- **Math** [V GitHub README]:
  ```
  y_t = (1-β) z_t + β x_t
  z_{t+1} = z_t - γ ∇f(y_t)      (Adam-preconditioned)
  x_{t+1} = (1 - c_{t+1}) x_t + c_{t+1} z_{t+1},   c_{t+1} = 1/(t+1)
  ```
  The gradient is taken at y, evaluation uses x, and no schedule is needed. A warmup-weighted c, c_{t+1} = γ_t²/Σ_{i≤t} γ_i², is in the paper [K].
- You must call `optimizer.train()/eval()`. β is sensitive (0.9 by default; 0.95-0.98 for long runs). The LR should be 1-10x the AdamW LR [V].
- Evidence: it won the AlgoPerf self-tuning track [V]. Hägele et al. found that even with tuned momentum it is "matched or outperformed by the cooldown schedule" [V].
- Use here: only attractive if anytime checkpoints are wanted (e.g. stage 1 to stage 2 without re-annealing). The WSD/EMA options below give the same flexibility with plain AdamW. **Low.**

### 2.4 AdEMAMix: low

- **Math** [K]:
  ```
  m1 = β1 m1 + (1-β1) g
  m2 = β3 m2 + (1-β3) g        (β3 = 0.9999)
  v  = β2 v + (1-β2) g²
  θ -= η[(m̂1 + α m2)/(√v̂ + ε) + λθ]
  ```
  with α ≈ 5-8, and warmup schedulers on α and β3.
- Claims: a 1.3B model on 101B tokens matches AdamW on 197B, and the method "significantly slows down model forgetting" [V abstract].
- At our 20k steps the slow EMA's horizon (1/(1-β3) = 10k steps) is half the run, and the paper needs schedulers for short runs. **Low.** The "less forgetting of early data" argument is weak with only 2 epochs.

### 2.5 Lion, LAMB: no

- Lion: update = sign(β1 m + (1-β1) g) + λθ; m ← β2 m + (1-β2) g. It needs a smaller LR [V abstract] (3-10x [K]) and a larger wd. Its gains grow with batch size [V abstract]. It mainly saves memory, which we don't need.
- LAMB targets very large batches (layer-wise trust ratio ||θ||/||update||) [K]. Our batch of 512 feeds is ordinary.
- **Skip both.**

---

## 3. LR schedules and weight averaging

### 3.1 Definitions (math)

- **Cosine** (current): η_t = η_max[0.03 + 0.97·½(1 + cos(π t'/T'))] after warmup.
- **WSD** (warmup-stable-decay): constant η_max until T_s, then a cooldown. Hägele's 1-sqrt cooldown is η_t = η_max·(1 - sqrt((t - T_s)/(T - T_s))). It beats a linear cooldown, the gain from longer cooldowns plateaus at about **20%** of steps, and with 1-sqrt it "can match cosine" [V Hägele 2024].
- **MiniCPM WSD:** exponential decay, ~10% decay fraction, and from any stable checkpoint you can branch a decay [V].
- **EMA:** θ̄_t = βθ̄_{t-1} + (1-β)θ_t, with time constant 1/(1-β) steps.
- **LAWA:** θ̄ = (1/L)·Σ_{l<L} θ_{t-lν} (a rolling window of checkpoints) [V Ajroldi].
- **SWA:** a uniform average over a high-constant-LR tail [K].
- **WSM** (Ling team, Ant Group; ICLR 2026 oral; used in Ling 2.0 / Ling-1T [V WSM; S Ling-1T card]): constant LR with no decay; the evaluated model is a merge θ̂ = Σ_j c_j θ_{n+j}. Exact identity [V]: θ̂_{n+k} = θ_n - Σ_{i=1}^{k} w_i g_{n+i-1} with w_i = Σ_{j≥i} c_j. **Merging equals applying a synthetic LR-decay schedule w_i to the last k updates**, and conversely c_k = w_k, c_j = w_j - w_{j+1}, c_0 = 1 - w_1 (Theorem 3.1). A mean merge corresponds to linear decay and EMA to an exponential decay.

### 3.2 Evidence

- WSM: merge *duration* (window length) matters more than checkpoint interval or count. Mean and 1-sqrt merges beat EMA. WSM beats WSD by about 1.3 points on average. **Decay-then-merge and merge-then-decay hybrids gave nothing**: "alternative pathways to a similar optimization objective" [V].
- Hägele: SWA on a constant LR "does not reach the loss values of explicit cooldowns". It is "strong, but not optimal" [V].
- Ajroldi et al. (AlgoPerf, ICML 2025):
  - LAWA (L=20, ν=64-128) and EMA (β=0.9 at ν=64-128) reach the targets in ~82% of the steps.
  - **When the LR is annealed fully to zero, the average converges to the annealed model.** Averaging combined with annealing is best, but by small margins.
  - Generalization gains are "minimal, mostly non-significant".
  - Recommended window ≈ 1% of the budget [V].
- EMA of weights improves generalization, label-noise robustness, calibration and transfer, and "needs less LR decay" [V Morales-Brotons et al. 2024].
- **For fine-tunability:** pre-training with warmup-stable-only (no decay) "consistently outperforms decay-based schedulers in terms of performance after SFT" at 1B/8B; decay "lead[s] models into sharper minima" [V abstract arXiv 2603.16127]. A related 2026 paper blames LR-decay-induced sharpness for catastrophic forgetting during SFT [S].

### 3.3 Transfer

**(a) Stage 1 and stage 2: per-step EMA with negligible cost. Do it for every run.**

```python
ema = [p.detach().clone().float() for p in model.parameters()]           # after init (or after warmup)
def ema_update(beta):
    with torch.no_grad():
        torch._foreach_lerp_(ema, [p.detach().float() for p in model.parameters()], 1 - beta)
# after opt.step():  ema_update(0.9995 if step > a.warmup else 0.0)   # time constant 2000 steps ≈ 10% of the run
# at each eval: swap in EMA weights (copy into a second Ranker), eval holdout + val, select on holdout among {raw, ema}
```

- Buffers (`f_mean`, `f_std`) are constant, so there is no BatchNorm issue.
- Expected: about +0.0000 to +0.0005, because the cosine already anneals to 3%.
- **But** the per-eval noise in the last 4k steps of `v2_feed_k9` is about ±0.0003 on val and ±0.0003 on the holdout [L]. EMA removes most of that SGD noise, which makes A/B tests of the other tricks cleaner. That alone justifies it.
- In stage 2 (1,288 steps), use β ≈ 0.995 (time constant 200 steps).

**(b) WSD instead of cosine, combined with "decay-phase data" (the MiniCPM recipe) that merges stage 2 into stage 1.**

- MiniCPM adds high-quality/SFT data during the decay phase. In their ablation this gave large downstream gains at 2.4B, e.g. C-Eval 40.0 → 52.6 and GSM8K 27.7 → 42.3 [V].
- Our analog of high-quality data is the 164K random-order exploration feeds: the same distribution as val/test, no production-order bias.
- Today they are used in a separate stage 2 (+0.0012 on average [L]), while up-weighting them during the whole run hurt (-0.001 [L]).
- Proposal:
  - Stable phase: lr 1e-3 on all feeds for ~80% of the steps.
  - 20% 1-sqrt cooldown: batches drawn with ~25-35% exploration feeds (the rest production). Keep K9's aux weight of 1-p_random, so random-order feeds get no production-order target.
  - Optionally a short stage 2 afterwards.
- One stable run can then branch several cooldowns, e.g. with mixture fractions 0.15 / 0.3 / 0.5. Each costs 20% of a run. This is also a cheap source of **ensemble members that differ in their late-phase data mixture**, which is more diverse than seeds.
- Expected: +0.0000 to +0.0010 over cosine followed by stage 2. Risk: low, since everything is selected on the holdout.

**(c) Training length.**
- Val and holdout rise through the end, but nobody ever finished a >2-epoch run with a matched schedule: `nopos_e4` was stopped at 24k/40k steps while its LR was still 3.7e-4, so it is not evidence either way [L].
- The model has no sparse ID tables, so the CTR "one-epoch phenomenon" does not apply. That phenomenon is caused by overfitting of sparse ID embeddings [S Kuaishou 2305.19531 / 2407.01607]. It **would** apply to the planned N4 hashed item-ID residual, so keep ≤ 1 epoch over the ID tables there.
- Run a 3-epoch cosine (or WSD with 3 epochs of stable phase plus cooldown) with 2 seeds, about 27-30 min each.
- Expected: -0.0005 to +0.0015. This is the single most informative run for deciding where the train stage's remaining headroom lies.

**(d) WSM-style checkpoint merging instead of decay.** This is theoretically equivalent to (b) and (a) combined [V WSM]. It is useful only if anytime models are wanted. Our runs are short, so a real cooldown is simpler. **Not recommended over (b).**

**(e) Fine-tune from a less-annealed checkpoint** [V 2603.16127].
- The fine-tune currently starts from the fully annealed stage-2 model.
- WSO evidence says pre-decay checkpoints adapt better. But here the fine-tune data is only 11.5K feeds per fold, and the train-stage anneal itself is worth about +0.002 [L].
- If tried, fine-tune from a WSD checkpoint taken mid-cooldown, with `--add_random 12000` (the anneal then happens on val folds plus exploration feeds) and a fine-tune LR that decays from ~2e-4 to 0.
- Medium-low priority, and it adds a 2-seed × 2-arm experiment.

**(f) Fine-tune (highest value).** See §9.1: EMA/LAWA of weights plus a decay, instead of a constant LR plus a prediction-snapshot mean.

---

## 4. Regularization

### 4.1 Weight decay (decoupled versus "independent")

- PyTorch AdamW is decoupled from the gradient but coupled to the LR: θ ← θ - ηλθ.
- In the original AdamW, the decay is scaled by the schedule multiplier only, not by the base α [K]. "Independent" wd (Wortsman et al. 2023 small-scale proxies [K]; the abstract [V] only says wd and μP affect LR sensitivity) means not multiplying by the base LR, so changing lr does not change the regularization.
- The EMA view [V abstract] says the right object is τ = 1/(ηλ) measured in epochs, which should stay constant as data and model sizes change.
- Here τ ≈ 2.5 runs (§1). LLM practice is λ = 0.1 with lr ~4e-4 [V WSM; V Moonlight], i.e. τ ≈ 25k steps.
- **Transfer:**
  - (i) Exclude the `cat/et/es/rec/dom` embeddings from wd (a name-filter fix).
  - (ii) Optional λ ∈ {0.02, 0.1} sweep, only after the optimizer is settled (with Muon, λ multiplies the same lr).
  - Expected |Δ| < 0.0005. **Low.**
  - In the fine-tune, replace decay-to-zero with **decay toward θ0** (§9.3).

### 4.2 Dropout variants

| Variant | Math | Here |
|---|---|---|
| Residual/FFN dropout (current 0.1) | x + Drop_p(f(x)) | Stage 1 is data-rich and under-fitted (fact 1). LLM pre-training uses p = 0 [K]. Test `--drop 0.05` and `0`, but keep 0.1 (or the `--drop` override) in the fine-tune. |
| **Attention-probability dropout** (current 0.1 via SDPA `dropout_p`) | A' = Drop_p(softmax(QKᵀ/√d + bias)) | Hurts the retrieval-like sim-bias heads: with peaked attention it zeroes the single matched history item 10% of the time. **Arm: `dropout_p = 0` in SDPA, residual dropout kept.** One-line change (add an `attn_drop` arg to `Attn`). |
| History-event (token) dropout / stochastic length | drop each history event with p, or crop the history to the most recent L' < 256 at random (HSTU's Stochastic Length: for n > N^{α/2}, subsample with probability 1 - N^α/n² [V HSTU]) | An augmentation for robustness to short histories. But history-length TTA showed shorter histories hurt [L], so expect ≈0. Low. If tried, recompute the dense features on the cropped history (they come from `b["he"]`, so this happens automatically). |
| Feature-group dropout | zero whole dense-feature groups (e.g. all f1..f5 transforms) per feed with p ≈ 0.1 | Encourages redundancy against train→test feature shift. No specific evidence for it here. Low. |
| Stochastic depth | drop a whole residual block with probability p_l, scaling at inference [K] | With 2 cross + 2 set + 1 history block, too shallow to matter. Skip. |

### 4.3 R-Drop (consistency between two dropout passes)

- Listwise version (R-Drop's bidirectional-KL form [V abstract]; formula [K]): with p^{(1)} and p^{(2)} the per-feed softmaxes from two dropout-sampled passes,
  ```
  L = ½[CE(t, p1) + CE(t, p2)] + α·½[KL(p1||p2) + KL(p2||p1)],   t = y/Σy
  ```
- Evidence: gains in 18 datasets across 5 task types, with "substantial improvements when fine-tuning large pre-trained models" [V].
- **Here: fine-tune only** (val is small and it overfits fast). There is 2x compute but the fine-tune takes 4 min. α ∈ {0.5, 1}.
- Expected +0.000 to +0.0008. Risk: with attention dropout off, there is less dropout noise to regularize; keep residual dropout at 0.1.

### 4.4 SAM / ASAM

- **Math** [K]:
  - SAM: ε̂ = ρ∇L(w)/||∇L(w)||₂, then w ← w - η∇L(w + ε̂). Typically ρ ≈ 0.05.
  - ASAM: ε̂ = ρ T_w²∇L / ||T_w∇L||, with T_w = diag(|w| + η₀). It is scale-invariant [V abstract]; typically ρ ≈ 0.5-2.
- Evidence in recsys:
  - SAMRec: SASRec-style transformers "converge to extremely sharp local minima if not adequately regularized". SAM matches the contrastive and self-supervised methods (CL4SRec, S³Rec) without pre-training [V abstract].
  - Helen (CTR): a frequency-wise Hessian-eigenvalue regularizer [S].
- Train stage: 2x cost for 20 min runs, in a data-rich regime where sharpness is less of a problem. Low.
- **Fine-tune**: plausible. Use ρ ∈ {0.02, 0.05} with AdamW as the base optimizer: take the first backward, perturb, take the second backward, restore, step. Expected +0.000 to +0.0008.
- A SAM-trained fine-tune member also adds diversity.

### 4.5 Label smoothing for listwise losses: do not use

- t_ε = (1-ε)·y/Σy + ε/M·**1**, and ∂L/∂z = p - t_ε. It is an entropy regularizer that pulls every feed toward uniform scores [K].
- RealMLP found ε = 0.1 useful for tabular *classification* [V], but there accuracy and calibration are what matter.
- For a pessimistic-ties ranking metric, the logit scale is irrelevant, and smoothing erases information about relative order [K: Müller et al. 2019, which also shows LS teachers distil worse].
- The useful non-uniform version, smoothing toward the production order softmax(-pos/τ), **already exists** as K9.
- Skip.

### 4.6 Mixup / cutmix for tabular and ranking: do not use

- I found no primary work applying mixup to listwise LTR or recsys ranking [S search].
- Cross-feed mixing breaks feed semantics: a mixed feed has no consistent user or history.
- Within-feed candidate mixing creates synthetic candidates that the set-attention block then sees as context.
- Tabular-DL baselines (TabM, RealMLP) use **no** data augmentation [V].
- Skip.

---

## 5. Distillation and auxiliary losses

### 5.1 Born-again / self-distillation: medium

- Math (listwise, per feed; standard KD [K]):
  ```
  L = CE(y/Σy, softmax(z_S)) + β·τ²·KL(softmax(z_T/τ) || softmax(z_S/τ)) + 0.5·L_K9
  ```
- Evidence:
  - Born-again networks: students "parameterized identically to their teachers" "outperform their teachers significantly" (CIFAR, LM) [V abstract].
  - CTR ensemble distillation: a plain DNN student gains over SOTA teachers, with >1% AUC on Avazu and online A/B wins [S, Zhu et al. CIKM 2020].
  - "A good teacher is patient and consistent": the distillation recipe, with consistent inputs and long schedules, matters a lot [V abstract].
- **Why it should matter more here than in classification:** each train feed has ~1.73 positives among ~78 candidates [L, computed: every train feed has ≥ 1 positive]. The label loss only says "positives above the rest". A teacher's softmax over the whole feed adds dense information about the order among the ~76 negatives. K9 shows that dense listwise targets are worth +0.0024 here [L].
- Transfer:
  - Teacher = frozen `s2_v2_feed_k9`, run in the same batch (forward only, about +30% step time). Alternatively, precompute 247M fp16 teacher logits (~0.5 GB).
  - Settings: τ = 1, β ∈ {0.5, 1}. Student initialised from scratch; same seed pairing.
  - The teacher's train-feed outputs are in-sample, but with 0.86M params, 2 epochs and 3.1M feeds the train/holdout gap should be small. Check the teacher's NDCG on train feeds against the holdout before relying on it.
- Expected +0.000 to +0.0012. Risk: confirmation bias (the student copies the teacher's errors, so the ensemble gains less from it).
- **N6** (soft pseudo-labels on test from a fold-safe teacher) is the semi-supervised cousin and is already planned.

### 5.2 Auxiliary multi-task and contrastive losses: low

- Already measured: production-order distillation is strongly positive, the category-intent head is negative or neutral [L].
- Contrastive auxiliaries (CL4SRec-type sequence augmentations [K]; InfoNCE between the encoded user token and positive candidates using in-batch cross-feed negatives) mainly improve *content* representations.
- The pretrain report shows that a content-only pretrained sequence score adds ≈0 on top of the NN; the incremental signal is in the ID part [L, `pretrain_report.md`]. So expect ≈0 from a content-contrastive aux.
- Low.

---

## 6. Tabular-NN tricks (the 61 dense features and the head)

### 6.1 TabM / BatchEnsemble (ICLR 2025): medium

- **Math** [V]:
  - Member i's linear layer is l_i(x) = s_i ⊙ (W(r_i ⊙ x)) + b_i, i.e. W_i = W ⊙ (s_i r_iᵀ), with W shared.
  - **TabM-mini** keeps only the first adapter r_i (random init, ±1 or N(0,1)).
  - k = 32 by default. Train on the **mean of the k losses**, predict the **mean of the k predictions**.
  - "Individual submodels look overfitted; the collective generalises substantially better". There are also fewer dead neurons (0.14 vs 0.29) [V].
- Transfer, cheapest form:
  - Apply TabM-mini at the candidate token before the cross/set blocks: c_i = c ⊙ r_i for i = 1..k, with k = 4.
  - The history encoder is shared, so only the candidate side costs k times more. Compute grows about 2-3x, i.e. ~40 min runs.
  - A head-only version (LN → (c ⊙ r_i) → MLP, k = 8) is nearly free but gives less diversity.
  - Loss = (1/k)Σ_i listwise(z_i) (+ K9 on the mean). Predict (1/k)Σ z_i.
- Expected +0.000 to +0.0010, and the members are internally diverse.
- Caveat: TabM's wins are on small/medium tabular sets, where overfitting dominates. We are data-rich in stage 1, but the regularization argument applies directly to the **fine-tune**.

### 6.2 Numerical-feature embeddings (PLR / PBLD): medium/low

- **PLR** [V abstract; formula K]: v_j = ReLU(W_j[sin(2πc_j x_j); cos(2πc_j x_j)] + b_j), with c_j ∈ R^K drawn from N(0, σ²) and trainable; K ≈ 16-48, output 8-16 per feature.
- **PBLD** (RealMLP) [V]: (x_j, W2·cos(2π·w1·x_j + b1) + b2) ∈ R⁴, using a 16-dimensional hidden layer.
- In RealMLP's ablation, removing numerical embeddings costs 2.3% (classification) to 20.6% (regression) of benchmark score [V].
- Transfer: replace `c_mlp`'s input `X` (61 standardised features clamped to ±6) with concat_j v_j (61 × 8-16 dims), then the same MLP.
- Many of our features are already within-feed ranks or z-scores, which are smooth, so the gain should be modest. +0.000 to +0.0010.

### 6.3 RealMLP preprocessing details: low

- Robust scaling plus smooth clipping: x' = f(s·(x - median)), with f(x) = x/sqrt(1 + (x/3)²) and s = 1/IQR [V].
- Our pipeline standardises with mean/std and hard-clamps at ±6. Smooth clipping avoids zero gradients beyond the clamp, but our inputs feed a linear layer, so the gradient with respect to the input is irrelevant.
- RealMLP's "β2 = 0.95 instead of 0.999" lesson [V] applies to the fine-tune (§1).

---

## 7. Sequence and time-series ideas without timestamps

The history is ordered newest-first, with a recency index 0..255 and no timestamps.

| Idea | Math | Status here | Verdict |
|---|---|---|---|
| Learned absolute recency embedding | h_j += E_rec[j] | **Already in** (`self.rec`). With 3M feeds the 256 rows are well estimated. | keep |
| Per-head learned recency bias in the target-attention logits | logit_{ij,h} += r_h[j] | **Already tested** as `sim_r` inside `sim_bias=2` (`aux_sim2`): neutral at 10k steps [L] | do not repeat alone |
| ALiBi-style fixed slopes | logit_{ij,h} -= m_h·j (or m_h·log(1+j)), geometric m_h [K, Press et al. 2022] | A special case of the above. Its benefit is length extrapolation, which is irrelevant here (max 256 seen in training). | skip |
| Time2Vec on the recency index | t2v(τ) = [ωτ + φ, sin(Ωτ + Φ)] [K] | Redundant with a fully learned 256-row table | skip |
| **Data-dependent forgetting (FoX)** | f_l = σ(w_fᵀx_l + b_f), D_ij = Σ_{l=j+1}^{i} log f_l, O = softmax(QKᵀ + D)V. No positional embeddings needed; the Pro version adds QK-norm and an output gate [V FoX]. | **Novel adaptation:** for candidate i → history event j, D_{j,h} = Σ_{l<j} log σ(w_hᵀh_l + b_h), a per-head cumulative "forgetting" from now back to event j. It is computed from the history tokens, so the decay rate can depend on content, e.g. a sharp topic switch or a session-like change in event_source can cut the influence of older events. Init b_h = +5 (almost no decay). Costs one Linear(d, heads) and a cumsum. | low/medium, +0.000 to +0.0005 |
| HSTU relative attention bias and pointwise attention | A(X)V = SiLU(QKᵀ + rab^{p,t})V, Y = f2(Norm(AV) ⊙ U). Softmax was worse than pointwise attention in their ablations (ranking NE .4980 → .5067 without rab), and rab helps further (.4980 → .4937) [V HSTU] | Pointwise (non-normalised) pooling keeps the *intensity* signal ("how many related events"). That is the planned **N5 SiLU pooling channel**, and `arch="din"` now has a SiLU channel too. Covered by the architecture track. | cross-ref |
| Stochastic Length | see §4.2 [V HSTU] | augmentation | low |

The time-series lesson that transfers best: **recency handled as an attention-logit bias rather than an additive embedding** is already partly in place and was neutral. Remaining headroom is in content-dependent decay (FoX) and in intensity-preserving pooling (HSTU/N5), not in fancier fixed position codes.

---

## 8. LLM-architecture-adjacent stability tricks (brief; the architecture track owns these)

- **Gated attention** (Qwen; NeurIPS 2025 best paper; used in Qwen3-Next/3.5): a head-specific elementwise sigmoid gate after SDPA, o' = o ⊙ σ(xW_g). It improves performance and stability, tolerates larger LRs and removes attention sinks. Shown on 30 variants at 15B-MoE and 1.7B-dense scale on 3.5T tokens [V abstract].
  - **Why it might matter here:** in candidate → history attention, softmax must put its mass somewhere even when no history event is relevant. Today the user token acts as a sink.
  - A query-dependent gate lets a candidate take "no history readout". This is +d² params per Attn, i.e. ~16K per block.
  - Expected +0.000 to +0.0007. It could be combined with Muon, which Qwen3.8-Next pairs with a gated residual [V abstract].
- **QK-norm** (Ling 2.0 "QKNorm for outlier mitigation" [S]; FoX Pro [V]): q, k ← RMSNorm(q), RMSNorm(k) with a learnable scale. This bounds q·k, so the per-head sim-bias scale (initialised up to 8) cannot be swamped or swamp the content term. Low priority at d = 128 with bf16 and an fp32 head; no instability has been observed.
- **z-loss** (log Z)² [K]: listwise softmax is shift-invariant per feed and the metric is scale-invariant, so there is nothing to fix. Skip.
- **MoE.** MoE adds parameters at constant FLOPs. Capacity is not binding here: d = 256 was worse and 3 blocks / 8 heads were neutral [L]. The traffic-domain token (the cheapest "expert by domain") was also neutral [L]. MiniMax M3/M3.1-Flash, Qwen3.5/3.8 and Ling 2.0 are all sparse MoE at 10⁹-10¹² parameters [S; V Qwen3.8 abstract]; their MoE gains come from scale we do not have. **Skip MoE.** The only MoE-flavoured idea worth trying is TabM-style implicit ensembling (§6.1), which is "many experts, all active, shared weights".
- **MTP-style multi-target heads:** the transferable analogue (extra dense targets on the same trunk) is already the K9 production-order head (+0.0024 [L]) and the proposed self-distillation term (§5.1).
- No public training details were found for MiniMax M3.1-Flash; sources only describe sparse attention and MoE [S]. Nothing from it transfers to a 0.86M-param ranker.

---

## 9. Fine-tuning on 14K lists (5-fold, ~11.5K feeds per fold, ~135 steps per epoch)

### 9.1 Averaging plus decay instead of a constant LR plus a prediction snapshot: high

**Facts:**
- OOF is best at epoch 2 (0.38611), the snapshot of epochs 1-6 gives 0.38572, and epochs 4-6 are below epoch 1 [L].
- The snapshot mean was +0.0017 over the single best epoch in v1 [L], and averaging is most useful at a constant LR [V Ajroldi; V EMA paper].

**Recipe:**
- Linear warmup 30 steps.
- Then either cosine/linear decay to 0 over E = 3 epochs, or keep the constant LR.
- Maintain a weight EMA with β = 0.99 (time constant ~100 steps ≈ 0.75 epoch), started after epoch 0.5. Or use LAWA: the mean of weights saved every 20 steps over epochs 1-3.
- Predict OOF and test **once**, from the averaged weights.
- Arms (each about 4 min, 2 seeds):
  - A0 = current recipe.
  - A1 = constant LR + EMA(0.99), E = 3.
  - A2 = cosine-to-0 over E = 3 + EMA(0.99).
  - A3 = A2 with E = 4.
- One global choice of arm across members, on OOF.
- Expected: reach or exceed the best-epoch score, i.e. +0.0003 to +0.0012 over the snapshot mean. Train-only + fine-tune OOF is the right unit of comparison.
- Keep the prediction-snapshot as an extra (z-mean of the EMA prediction and the per-epoch predictions) only if it adds value on OOF.

### 9.2 LP-FT for the zero-init feature branch: high

- **Evidence:** full fine-tuning distorts pretrained features while the new head is still being learned. LP-FT (linear-probe first, then full fine-tune) gives about +1% in-distribution and **+10% OOD** over full fine-tuning across 10 shift datasets [V].
- Our extra branch (67 features, zero-init, ×10 LR) is the analogue of the new head.
  - In the first ~100 steps the backbone receives gradients that try to fit residual signal the branch would explain later, which is feature distortion.
  - Because the branch is zero-init, its gradient at step 0 is non-zero only through the last layer, so it starts slowly.
- Recipe:
  - Phase A, 1 epoch: train only `extra.*`, `head.*` and `wide`, at lr 5e-4. The backbone is frozen and its dropout stays on.
  - Phase B: full fine-tune at lr 5e-5 (branch ×3, not ×10), with decay and EMA (§9.1).
  - It is about 20 lines in `finetune.py`: two param groups, and toggle `requires_grad` at the phase boundary.
- Expected +0.0003 to +0.0012. Risk: phase A overfits the branch. Use 1 epoch, never more.

### 9.3 Regularize toward the pretrained weights: decoupled L2-SP / WiSE-FT / LoRA (low/medium)

- **Decoupled L2-SP** [K, Li et al. 2018; decoupled form is mine]: after each AdamW step, θ ← θ - ηλ_sp(θ - θ0) with λ_sp ∈ {2, 10}.
  - Per step the pull is 1e-4 to 5e-4; over ~400 steps it removes about 4-20% of the drift.
  - It is decoupled because a penalty inside the loss gets rescaled by Adam's √v.
  - It replaces the inert wd-to-zero (§1).
- **WiSE-FT** [V]: θ_α = (1-α)θ0 + αθ_ft. It gave +1.6pp in-distribution and 4-6pp under shift on ImageNet, and +0.8-3.3pp on 7 transfer sets, at no extra cost.
  - Here, the fine-tune moves the backbone only slightly (lr 5e-5), so weight-space interpolation ≈ output-space interpolation. That was already tested as "mix 0.2 train-only": v1 +0.0005, v2 neutral [L].
  - It needs per-fold weights to be saved (`finetune.py` does not save them).
  - Expected ≤ +0.0005, using one global α from {0.6, 0.8, 1.0}.
- **LoRA / adapters** (W = W0 + (α/r)·BA on the attention/FFN matrices, r = 4-8): "learns less and forgets less", and "mitigates forgetting more than … weight decay and dropout" [V Biderman et al. 2024]. Here forgetting the train-stage knowledge is the main failure (the val fine-tune overfits within 2 epochs). But §9.1-9.2 address that more simply, and a 0.86M-param model has no memory problem. Low; try only if §9.1-9.2 fail.
- **Layer-wise LR decay / partial freezing** (LLRD: lr_l = lr_top·ξ^{L-l}, ξ ≈ 0.7-0.9 [K, ELECTRA/BEiT]).
  - Groups:
    - top: head, wide, extra;
    - next: set blocks;
    - next: cross blocks;
    - bottom: the history encoder, embeddings and input projections.
  - Few-sample BERT work found that the gains of such special methods "diminish significantly" once the basic fine-tune protocol is fixed (bias correction, enough iterations) [V abstract].
  - Expected +0.000 to +0.0008. Medium/low; it combines naturally with LP-FT phase B.

### 9.4 More data per fold model: 10-fold, plus a full-val refit for test (medium)

- Data scale is the dominant lever in the fine-tune: from-scratch on val gives 0.352/0.366 versus 0.365/0.372 fine-tuned on folds 0/1 [L].
- 10-fold gives each fold model 90% of val instead of 80%. Test is then the mean of 10 models.
- A full-val refit (same recipe, E fixed from CV) adds one model trained on 100%. It cannot be measured on val, so give it at most an equal weight to the fold mean.
- Expected +0.0003 to +0.0008 on test; partly visible as OOF change. Cost 2x (8 min).

### 9.5 Early stopping without leakage (best practice)

- Never choose the epoch, α or LR per fold on that fold's held-out feeds and then report OOF: that leaks into the OOF estimate.
- Choose **one global setting** (from a small grid) on OOF, apply it to all members, and expect ≈1 SE of optimism when the grid is small.
- Weight averaging and the snapshot mean remove most of the need for epoch selection.
- Stage-1/2 decisions belong on the 30K train holdout (as now), so val stays clean for the fine-tune decisions.
- Optimizer state: the fine-tune starts a fresh AdamW. That is fine, but use betas (0.9, 0.98) like stage 1, not the default 0.999 (§1).

### 9.6 R-Drop / SAM in the fine-tune: medium

See §4.3-4.4. Both cost 2x (8 min) and target exactly the small-data overfitting regime. Run them after §9.1-9.2 so they are tested on top of the improved base.

---

## 10. Test-time augmentation

- **Candidate-permutation TTA:** the model is permutation-equivariant over candidates (set attention, no positional encoding), so permutations change only numerics. Useless.
- **History-length TTA:** already tested and rejected (every combination was worse than 256 alone) [L].
- **MC-dropout averaging** (k = 8 passes with dropout on, averaging per-feed z-scores) [K]: an inference-only test on existing checkpoints (~2 min). It usually does no better than the deterministic pass for a well-trained net, so expect -0.0005 to +0.0005. Low, but nearly free.
- **TabM-style implicit members** (§6.1) are the principled form of "TTA by sub-models".

---

## 11. Concrete experiment plan for GPU N (≤ 2 concurrent jobs; ~20 min per stage-1 run alone)

Every stage-1 arm: 2 paired seeds (21, 23), the baseline command from the task, select on the holdout, report val and holdout. **Accept** if the mean paired Δ is > 0 on both val and holdout and the 2-seed mean Δ is ≥ 2·SE_paired (≈ 0.001), or the sign is consistent on 3 seeds.

1. **Code prep** (no GPU): an EMA tracker plus "eval raw and EMA, select on holdout" in `train.py`. Add `--attn_drop`, `--opt {adamw, muon}`, `--sched {cosine, wsd}` with `--cooldown_frac 0.2` and `--cooldown_random_frac`, and `--distill_from`. In `finetune.py`: `--ema`, `--decay {const, cos}`, `--lpft_epochs`, `--rdrop`, `--sam_rho`, `--folds 10`, `--betas`. Change only *your copy's* `common.py` guard to allow "5".
2. **Fine-tune bundle first** (cheap; reuses `s2_v2_feed_k9` with `n2full`):
   - A1/A2/A3 from §9.1 (3 arms × 2 seeds ≈ 25 min);
   - then LP-FT on the winner (2 × 4 min);
   - then R-Drop α = 1 and SAM ρ = 0.05 (2 × 8 min each);
   - then 10-fold of the final recipe.
3. **Stage 1:**
   - (a) EMA plus attention dropout 0 (2 seeds);
   - (b) Muon (2 seeds, lr 1e-3; add lr 2e-3 if (b) ≥ 0);
   - (c) a 3-epoch cosine (2 seeds);
   - (d) WSD with a 20% 1-sqrt cooldown, 30% exploration feeds in the cooldown (2 seeds), compared with cosine followed by stage 2;
   - (e) self-distillation from `s2_v2_feed_k9` (2 seeds).
   - Then stage 2 and the fine-tune bundle on each winner.
4. **Diversity members** for the final ensemble: Muon member(s), a TabM-mini member, a SAM-fine-tuned member, and WSD members with different cooldown mixtures. Hierarchical per-feed z-mean as now.

Rough budget: fine-tune bundle ~1.5 GPU-h; stage-1 arms 5 × 2 × 20 min ≈ 3.5 GPU-h at 2 jobs in parallel.

## 12. Do-not-do (with reasons)

- **Mixup or cutmix on feeds:** no mechanism for listwise sets, no evidence, and the tabular SOTA does not use augmentation (§4.6).
- **Uniform label smoothing:** destroys order information; the useful non-uniform version is K9 (§4.5).
- **Stochastic depth, MoE, z-loss:** too shallow, capacity not binding, shift-invariant loss (§4.2, §8).
- **Lion or LAMB:** the regime is wrong (small batch, no memory pressure) (§2.5).
- **Muon in stage 1 with AdamW in the fine-tune**, or the reverse, without testing: the optimizer-mismatch effect [V Moonlight].
- **Comparing optimizers or schedules at intermediate checkpoints:** rankings flip during the decay [V Wen et al.]. Compare at the end, or with EMA weights.
- **More than 1 epoch over any new sparse ID table:** the one-epoch phenomenon [S].
- **Selecting epochs or α per fold on its own held-out data** (§9.5).

---

## Sources

Primary sources read this session [V]:
- WSM, Tian et al. (Ling Team, Ant Group), arXiv 2507.17634, ICLR 2026: https://arxiv.org/html/2507.17634v2
- Qwen3.8-Next design paper (Qwen et al.), arXiv 2608.30320: https://arxiv.org/abs/2608.30320
- Muon blog (K. Jordan): https://kellerjordan.github.io/posts/muon/
- Muon is Scalable for LLM Training (Moonlight), arXiv 2502.16982: https://arxiv.org/html/2502.16982
- PyTorch 2.11 `torch.optim.Muon` source (`adjust_lr_fn="match_rms_adamw"`): local `.venv/lib/python3.12/site-packages/torch/optim/_muon.py`
- Fantastic Pretraining Optimizers and Where to Find Them, arXiv 2509.02046: https://arxiv.org/abs/2509.02046
- MuonRec, arXiv 2603.00416: https://arxiv.org/abs/2603.00416
- The Road Less Scheduled (Schedule-Free), arXiv 2405.15682: https://arxiv.org/abs/2405.15682 ; README: https://github.com/facebookresearch/schedule_free
- AlgoPerf competition analysis, arXiv 2502.15015: https://arxiv.org/abs/2502.15015
- Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (Hägele et al.), arXiv 2405.18392: https://arxiv.org/html/2405.18392
- When, Where and Why to Average Weights? (Ajroldi et al.), arXiv 2502.06761: https://arxiv.org/html/2502.06761v3
- EMA of weights: dynamics and benefits, arXiv 2411.18704: https://arxiv.org/abs/2411.18704
- MiniCPM (WSD, decay-phase data), arXiv 2404.06395: https://arxiv.org/html/2404.06395
- Pre-training LLM without LR decay enhances SFT, arXiv 2603.16127: https://arxiv.org/abs/2603.16127
- WiSE-FT, arXiv 2109.01903: https://arxiv.org/abs/2109.01903
- LP-FT, arXiv 2202.10054: https://arxiv.org/abs/2202.10054
- LoRA Learns Less and Forgets Less, arXiv 2405.09673: https://arxiv.org/abs/2405.09673
- Revisiting few-sample BERT fine-tuning, arXiv 2006.05987: https://arxiv.org/abs/2006.05987
- TabM, arXiv 2410.24210: https://arxiv.org/html/2410.24210v3
- RealMLP (Better by Default), arXiv 2407.04491: https://ar5iv.labs.arxiv.org/html/2407.04491
- On Embeddings for Numerical Features, arXiv 2203.05556: https://arxiv.org/abs/2203.05556
- Forgetting Transformer (FoX), arXiv 2503.02130: https://arxiv.org/html/2503.02130
- HSTU (Actions Speak Louder than Words), arXiv 2402.17152: https://arxiv.org/html/2402.17152v3
- Gated Attention for LLMs, arXiv 2505.06708: https://arxiv.org/abs/2505.06708
- AdEMAMix, arXiv 2409.03137: https://arxiv.org/abs/2409.03137
- R-Drop, arXiv 2106.14448: https://arxiv.org/abs/2106.14448
- SAMRec, arXiv 2308.10347: https://arxiv.org/abs/2308.10347
- ASAM, arXiv 2102.11600: https://arxiv.org/abs/2102.11600
- Lion, arXiv 2302.06675: https://arxiv.org/abs/2302.06675
- How to set AdamW's weight decay (EMA view), arXiv 2405.13698: https://arxiv.org/abs/2405.13698
- Small-scale proxies for large-scale Transformer instabilities, arXiv 2309.14322: https://arxiv.org/abs/2309.14322
- Born-Again Neural Networks, arXiv 1805.04770: https://arxiv.org/abs/1805.04770
- Knowledge distillation: a good teacher is patient and consistent, arXiv 2106.05237: https://arxiv.org/abs/2106.05237

Snippet-only [S]:
- Ling 2.0 report, arXiv 2510.22115 (WSM, QKNorm, FP8), and the Ling-1T model card: https://huggingface.co/inclusionAI/Ling-1T
- SOAP, arXiv 2409.11321: https://arxiv.org/abs/2409.11321
- Ensemble KD for CTR, arXiv 2011.04106: https://arxiv.org/abs/2011.04106
- Multi-epoch CTR / one-epoch phenomenon, arXiv 2305.19531 and 2407.01607
- Helen (CTR sharpness), arXiv 2403.00798
- Qwen3.5 architecture (community analysis): https://huggingface.co/blog/mlabonne/qwen35
- MiniMax M3 / M3.1-Flash pages (no training details): https://minimax.io/blog/minimax-m31f
- Muon fine-tuning overfitting reports (2025-2026, community and arXiv 2605.06654)
- Catastrophic overtraining and LR-decay sharpness, arXiv 2604.13627

Background [K]:
- Optimization and averaging: SAM (Foret et al. 2021), SWA (Izmailov et al. 2018), model soups (Wortsman et al. 2022).
- Fine-tuning: L2-SP (Li et al. 2018), LLRD (ELECTRA / BEiT), LoRA (Hu et al. 2021), Mixout.
- Positions and time: ALiBi (Press et al. 2022), Time2Vec (Kazemi et al. 2019), TiSASRec.
- Regularization and augmentation: label smoothing (Müller et al. 2019), stochastic depth (Huang et al. 2016), CL4SRec (Xie et al. 2020), Noisy Student (Xie et al. 2020).
- LLM tricks: z-loss (PaLM / ST-MoE), QK-norm (Dehghani et al. 2023).
- Ensembles: diversity from training methodology (Gontijo-Lopes et al. 2022).
