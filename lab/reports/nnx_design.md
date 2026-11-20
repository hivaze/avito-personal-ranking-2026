# NNX design: 3 parallel experiment groups on GPU N (modern NN tricks for the feed ranker)

Written 2026-11-11 by the NN architect. Inputs: `nn/{model,train,data,finetune,common}.py` as they are on disk, `reports/nn_report.md` (round 1 and round 2), and the four research notes `reports/nnx_research_{llm_arch,recsys_arch,tabular_dl,training_gen}.md`.
All gains are val full NDCG (exact `src/metrics.py`). The research ranges are priors, not measurements.

## 0. Summary

Three experimenters share GPU N. Each runs at most 2 training processes at a time. Every experiment is a paired A/B against the same baseline (seeds 23 and 24), trained for the full 2-epoch recipe.

| group | # | experiment | flag(s) | prior Δ val | GPU cost (x baseline) | dev cost |
|---|---|---|---|---|---|---|
| **attn** | A1 | Gated attention output (Qwen G1, elementwise sigmoid), all 5 attention modules | `--attn_gate elem` | +0.0005..+0.002 | 1.05 | 15 min |
| attn | A2 | FoX data-dependent forgetting bias on target attention (learned recency decay) | `--fox` | 0..+0.0015 | 1.02 | 30 min |
| attn | A3 | QK-norm + length-aware temperature (SSMax) | `--qk_norm --ssmax` | 0..+0.001 | 1.03 | 20 min |
| attn | A4 | TabR-style candidate-conditioned retrieval values (top-16 history items, L2 keys, sink) | `--retr_m 16` | +0.0005..+0.002 (speculative) | 1.15 | 1.5 h |
| attn | A5 (optional) | Fine-grained shared-expert MoE FFN (DeepSeekMoE / Ling 2.0), aux-loss-free balancing | `--moe 8` | -0.001..+0.001 | 1.15 | 1 h |
| **feat** | B1 | Piecewise-linear numeric embeddings (PLE, rtdl version B) for the 63 dense features | `--num_emb ple` | +0.0003..+0.0015 | 1.05 | 45 min |
| feat | B2 | History-supervised item propensity head (event-type distribution of an item, from 584M history events) | `--prop_head 0.05` | +0.0005..+0.0015 | 1.05 | 45 min |
| feat | B3 | Semantic-ID codes (2-level residual k-means of the content embedding): learnable code embeddings + code-match counts | `--sid` | +0.0005..+0.002 (risk: memorisation) | 1.05 | 2 h (offline k-means on CPU) |
| feat | B4 | Multi-scale recency-decayed (EWMA) history features ("time-series" smoothing) | `--ts_feats` | 0..+0.0008 | 1.0 | 30 min |
| feat | B5 (optional) | TabM-mini BatchEnsemble k=4 on the candidate path | `--tabm 4` | +0.0005..+0.0015 | ~3 | 1.5 h |
| **optim** | C0 | **Shared baseline** (2 seeds) + its stage 2 + its fine-tune; everyone pairs against it | - | - | - | 30 min |
| optim | C1 | Weight EMA in stage 1 (3 betas evaluated inside the baseline runs, no extra training) | `--ema 0.999,0.9995,0.9998` | 0..+0.0005 | 0 extra runs | 30 min |
| optim | C2 | Muon for hidden matrices (RMS-matched to AdamW), AdamW for the rest | `--muon` | -0.001..+0.002 | 1.05 | 30 min |
| optim | C3 | Fine-tune recipe: cosine decay + weight EMA (arm a); + LP-FT (arm b) | `finetune.py --decay cos --ema 0.99 [--lpft_epochs 1]` | +0.0003..+0.0012 OOF | 0.3 per arm | 45 min |
| optim | C4 | LLM decay-phase recipe: WSD 1-sqrt cooldown + exploration-feed mixing in cooldown + aux-weight annealing (MiniCPM / MiniMax-M2 / DeepSeek-V3) | `--sched wsd --cooldown_mix 0.3 --aux_anneal 0.1` | -0.0005..+0.0015 (vs baseline + stage 2) | 1.0 | 45 min |
| optim | C5 | Born-again listwise self-distillation from a frozen teacher (s2_v2_feed_k9) | `--distill <logits> --distill_w 0.5` | 0..+0.0012 | 1.05 (+5 min for teacher logits) | 45 min |

Order inside each group = expected gain / cost. Run them in that order. Optional items (A5, B5) run only after the others are judged and GPU memory allows.

---

## 1. Ground rules (all groups)

### 1.1 Baseline config (identical for all three groups)
This is the computed-task baseline, i.e. `v2_feed_k9` (seed 23: val 0.38176 / holdout 0.30091, stage 2 0.38271 / 0.30094). No `--xt`: the round-2 co-visitation inputs are fragile (5 of 16 already pruned from disk), take +8 GB per process and are owned by the NN agent. Architecture and optimizer effects should mostly carry over; the integration step (section 6) re-tests winners with `--xt cv`.

```bash
PY=<repo>/.venv/bin/python
BASE="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
      --select_on holdout --epochs 2 --eval_every 1000 --loss_norm feed --no_test"
export CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 POLARS_MAX_THREADS=16 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Seeds are **23 and 24** for every arm. Seed 25 is the tie-breaker (see 1.6). `--no_test` applies to all screening runs; test predictions for winners come later via `predict.py`.

### 1.2 Shared baseline: owned by group optim (C0), reused by attn and feat
The optim experimenter runs these first, in `<repo>/nnx/optim`, with the EMA instrumentation (C1) switched on. EMA only reads the weights, so the raw-weight results are the baseline:
```bash
cd <repo>/nnx/optim
$PY train.py --variant base_s23 $BASE --seed 23 --ema 0.999,0.9995,0.9998 --ema_start 0.5   # in parallel with:
$PY train.py --variant base_s24 $BASE --seed 24 --ema 0.999,0.9995,0.9998 --ema_start 0.5
# stage 2: exact production recipe of nn/stage2_ft_v3.sh, seed 0 for every stage-2 run
$PY train.py --variant s2_base_s23 --init base_s23 --only_random 0.5 --holdout 30000 --aux_pos 0.5 --loss_norm feed \
    --lr 2e-4 --warmup 100 --epochs 2 --eval_every 400 --keep last --no_test --seed 0      # same for s24
```
Reference files for everyone:
`<repo>/nnx/optim/preds/nn_base_s{23,24}_trainonly_val.npy`, `.../nn_base_s{23,24}_holdout.npz`, `.../logs/base_s{23,24}.log` (per-1000-step curves for the kill rule), and the same for `s2_base_s*`.
attn and feat start their first treatment runs at t=0 and compare once the baseline lands. They do **not** re-run the baseline.
Free check: compare `base_s23` with the existing `<repo>/preds/nn_v2_feed_k9_trainonly_val.npy` (same config and seed, round 1) using `paired.py`. That difference measures GPU non-determinism; write it into results.

If a third seed is needed: the first experimenter who needs `base_s25` creates `<repo>/nnx/base_s25.lock` (`set -o noclobber`), runs it in their own copy, and writes the owner's path into the lock file. The others wait for that file and reuse it.

### 1.3 Harness patch H0 (each experimenter, in their own copy, before anything else)
```bash
G=attn    # feat | optim
mkdir -p <repo>/nnx/$G/{ckpt,logs,preds}
cp <repo>/nn/{common,data,model,train,finetune,blend,predict}.py <repo>/nnx/$G/
```
1. `common.py`: the guard becomes `if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "5": raise RuntimeError(...)`. Add `HERE = Path(__file__).resolve().parent` and `CKPT, LOGS, PREDS = HERE/"ckpt", HERE/"logs", HERE/"preds"`; this matters because the original writes into the team's shared `preds/`. Add `torch.set_num_threads(16)`. Data stays read from `<repo>/data`.
2. `common.eval_subset(..., return_scores=False)`: when True, return `(ndcg, sc)`.
3. `train.py`, right after `model.load_state_dict(best[1])`, so the holdout becomes pairable per feed:
   ```python
   if hold is not None:
       ndh, sch = eval_subset(model, tr, hold, hold_rows, return_scores=True)   # hold is sorted -> rows align
       np.savez(PREDS / f"nn_{a.variant}_holdout.npz", scores=sch.astype(np.float32),
                y=tr.target_np[hold_rows].astype(np.float32), M=tr.M_np[hold])
   ```
4. `data.attach_extra` and the existence check in `finetune.py`: replace `ROOT/"data"/"feats"` with `Path(os.environ.get("NNX_FEATS", "<repo>/data/feats"))`. Fine-tunes use the frozen snapshot (C0) with `NNX_FEATS=<repo>/nnx/optim/feats`.
5. `model.py`: build every new module under a forked RNG, so the baseline parameters get exactly the baseline init and the A/B differs only by the treatment:
   ```python
   import contextlib
   @contextlib.contextmanager
   def extra_init(offset=7919):
       with torch.random.fork_rng(devices=[]):      # model is built on CPU, then .to(dev)
           torch.manual_seed(torch.initial_seed() + offset)
           yield
   ```
6. Every new option is a CLI flag that defaults to off, plus a `Ranker.__init__` kwarg with a default that is stored in `self.cfg`. With all flags off, the code path must be the baseline's: old checkpoints (`--init`, the teacher) must still load. Anything that feeds the logit stays in the fp32 head region; bf16 logits create ties, which the metric breaks pessimistically.
7. Copy `paired.py` (below) into the copy.

```python
# paired.py: paired A/B on val (+ 30K random-like train holdout). {seed} in prefixes is substituted.
#   $PY paired.py --base <repo>/nnx/optim/preds/nn_base_s{seed} --treat <dir>/preds/nn_<variant>_s{seed}
#   fine-tune OOF: --val_suffix _val_oof.npy (no holdout files needed)
import argparse, os, sys
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
FL = "<repo>/data/flat"
p = argparse.ArgumentParser()
p.add_argument("--base"); p.add_argument("--treat"); p.add_argument("--seeds", default="23,24")
p.add_argument("--val_suffix", default="_trainonly_val.npy")
a = p.parse_args()
f = pl.read_parquet(f"{FL}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{FL}/val/cand_target.npy").astype(np.float64)
def pf(s, y, off, M): return ndcg_per_feed(s, y, off, M)
D = {"val": [], "hold": []}
for s in a.seeds.split(","):
    b, t = a.base.format(seed=s), a.treat.format(seed=s)
    vb, vt = pf(np.load(b + a.val_suffix), y, off, M), pf(np.load(t + a.val_suffix), y, off, M)
    D["val"].append(vt - vb)
    msg = f"seed {s}: val {vb.mean():.5f} -> {vt.mean():.5f} d={vt.mean()-vb.mean():+.5f} se={(vt-vb).std(ddof=1)/np.sqrt(len(vb)):.5f}"
    if os.path.exists(b + "_holdout.npz") and os.path.exists(t + "_holdout.npz"):
        hb, ht = np.load(b + "_holdout.npz"), np.load(t + "_holdout.npz")
        assert (hb["M"] == ht["M"]).all()
        Mh = hb["M"].astype(np.int64); oh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
        xb, xt = pf(hb["scores"], hb["y"], oh, Mh), pf(ht["scores"], ht["y"], oh, Mh)
        D["hold"].append(xt - xb)
        msg += f" | hold {xb.mean():.5f} -> {xt.mean():.5f} d={xt.mean()-xb.mean():+.5f} se={(xt-xb).std(ddof=1)/np.sqrt(len(xb)):.5f}"
    print(msg)
for k, v in D.items():
    if v:
        d = np.mean(v, 0); se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"POOLED {k}: d={d.mean():+.5f} se={se:.5f} z={d.mean()/se:+.2f} (n_seeds={len(v)})")
```

### 1.4 GPU memory and threads
* GPU N is shared with the lead's jobs (85 GB were in use at design time) and the two other groups. A baseline-sized process takes about 30 GB. Before every launch, wait until `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i N` reports at least **[n]** MiB (A4/A5/B3: [n], B5 TabM: [n], fine-tune: [n]). C0's baselines have priority at t=0.
* At most 2 GPU processes per experimenter (stage-1, stage-2, fine-tune and teacher-logit jobs all count). Use 16 threads per process (24 max). The B3 k-means runs on CPU.
* Run jobs attached, or wait on them, so they end with you. If a job dies with an OOM error, relaunch it unchanged.

### 1.5 Naming and logging
Variants are named `<group>_<exp>_s<seed>` (e.g. `attn_gate_s23`); stage-2 runs add an `s2_` prefix and fine-tunes add `_ft<arm>`. Each group appends one line per finished comparison to `<repo>/nnx/<group>/results.tsv`:
`exp  arm  seeds  val_base  val_treat  d_val  se_val  hold_base  hold_treat  d_hold  se_hold  best_steps  minutes  params  verdict  note`.
Log the mechanism diagnostics listed for each experiment (gate means, expert loads, ...) at every eval. A gain without a working mechanism is suspect.

### 1.6 Decision rule (anti-overfitting protocol: holdout first, val second)
Let Δv and Δh be the 2-seed pooled paired deltas on val and holdout, with per-feed SEs SEv and SEh, as printed by `paired.py`.
* **WIN**: Δv > 0, Δh > 0, max(Δv/SEv, Δh/SEh) ≥ 2, and no single seed has Δv < −0.0010.
* **LIKELY**: Δv + Δh > 0, both > −0.0003, max z in [1, 2). Run seed 25 if budget allows and re-judge on 3 seeds; otherwise report as LIKELY.
* **NO**: anything else. Do not tune hyper-parameters of a NO on val. At most one follow-up arm per experiment, and only where it is pre-specified below.
* The per-feed SE ignores training noise (seed spread 0.001-0.002). That is why we use 2 seeds and the per-seed sign check, and why the holdout (30K feeds, train-side) gates as strongly as val.
* **Kill rule**: compare each eval against `base_s{seed}.log` at the same step. At step ≥ 8000 (≥ 12000 for C2/C4, whose curves differ by design), if both seeds trail on holdout by > 0.002 **and** on val by > 0.001, stop the arm and record NO (killed).

---

## 2. Group attn: attention and token-mixing blocks (copy `<repo>/nnx/attn`)

The model is history target attention (the biggest component, +0.006..+0.010) plus a raw-cosine similarity bias (+0.0024). Width, depth and heads are neutral, so capacity is not binding. This group therefore changes how attention allocates mass, not how much capacity it has.

### A1. Gated attention output (Qwen3-Next / Qwen3.5 G1)  [priority 1]
* **Math**: `Y = (softmax(Q K^T / sqrt(d_h) + B_sim) V) ⊙ σ(X W_g + b_g)`, per head, elementwise, before `W_O`. X is the pre-LN query token (the tensor that produced Q).
* **Why it should help here**: softmax mass sums to 1, so a candidate with no relevant history event must still average something. Today the always-valid user token acts as an implicit sink, and set attention has no sink at all. A query-dependent gate can switch the read off. It also puts a non-linearity between the low-rank `W_V W_O` pair. Evidence: arXiv 2505.06708 (PPL 6.026 -> 5.761, first-token attention 46.7% -> 4.8%, tolerates higher LR). Qwen3-Next, Qwen3.5 and Qwen3.8 adopted it. HSTU uses the same `Norm(AV) ⊙ U(X)` in recsys.
* **Code**: in `Attn.__init__`, when `attn_gate == "elem"`: `with extra_init(): self.g = nn.Linear(d, d)`, default init (paper: mean gate about 0.5 at init, learned sparsity). In `forward`, after SDPA (`o`: [B,h,N,dh]): `o = o * torch.sigmoid(self.g(x)).view(B, N, h, D // h).transpose(1, 2)`, then reshape and `self.o`. Apply to all 5 modules (hist_enc, 2 cross, 2 set). Thread `attn_gate` through `Block` and `Ranker`. About +82K params.
* **Run**: `$PY train.py --variant attn_gate_s{23,24} $BASE --seed {23,24} --attn_gate elem`
* **Diagnostics**: mean gate per module; attention mass on the user token in cross block 0 (mean over candidates), raw vs gated.

### A2. FoX forgetting bias: data-dependent recency decay in target attention  [priority 2]
* **Math** (Forgetting Transformer, arXiv 2503.02130): for history token l (newest first) and head h, `f_{l,h} = σ(w_h^T x_l + b_h)`. The target-attention logit for key j gets `D_{j,h} = Σ_{l<j} log f_{l,h}` (exclusive cumulative sum over newer events), and `D = 0` for the user token. So `logit_{c,j,h} = q_c·k_j/sqrt(d_h) + a_h S_{cj} + D_{j,h}`.
* **Why**: this is the time-series piece. Decay becomes input-dependent exponential smoothing, the same α_t forget gate the Gated-DeltaNet layers of Qwen3.5/Qwen3-Next use, moved into softmax attention without losing exact retrieval (MiniMax-M2 kept full attention for retrieval for this reason). A burst of unrelated events, such as a topic switch, closes the window faster; a coherent session keeps it open. The static per-position recency bias (`sim_bias=2`) was neutral, so any gain must come from the data dependence. A2 is materially different from it.
* **Code**: `Ranker.__init__`: `with extra_init(): self.fox = nn.ModuleList([nn.Linear(d, heads) for _ in range(blocks)])`, weight zero, bias `logit([0.999, 0.995, 0.98, 0.95])` (an ALiBi-like spread of horizons, about 1000/200/50/20 events). This is exactly static at init and becomes data-dependent through learning. In `forward`, after `hist_enc`, for cross block i, in fp32 (outside autocast, because a cumsum in bf16 loses precision):
  `logf = F.logsigmoid(self.fox[i](ctx[:, 1:].float())) * b["hmask"][..., None]` -> [B,K,h]; `D = logf.cumsum(1) - logf`; `Dfull = torch.cat([zeros(B,1,h), D], 1).permute(0, 2, 1)[:, :, None, :]` -> [B,h,1,K+1].
  Pass it to `Block.forward(..., extra_bias=Dfull)` -> `Attn.forward`. In the `bias_scale` branch, add it before `masked_fill`: `am = bias[:, None] * scale + extra_bias`. 516 params per block.
* **Run**: `... --variant attn_fox_s{23,24} $BASE --seed {23,24} --fox`
* **Diagnostics**: per head, the mean effective horizon `E[#events until D < −1]`, and the across-feed std of D at j=32 (the data dependence). If the std stays near 0, the gate collapsed to static decay.

### A3. QK-norm + length-aware temperature (SSMax)  [priority 3]
* **Math**: QK-norm `q̂ = γ_q ⊙ q / RMS(q)`, `k̂ = γ_k ⊙ k / RMS(k)` per head, so `|q̂·k̂|/sqrt(d_h) ≤ sqrt(d_h)·max|γ_q|·max|γ_k|`. Logits stay on a bounded scale next to the raw-cosine bias (a_h up to 8). SSMax: `q̃ = q̂ · s_h · log n`, with n = number of valid keys. Without it, the maximum attention weight fades roughly as 1/n for comparable logits. With it, sharpness holds across set sizes M = 5..300 and history lengths H+1 = 1..257 (Nakanishi, arXiv 2501.19399; TabICLv2 QASSMax).
* **Why**: Qwen3, Gemma 3 and Ling 2.0 use QK-norm (stability, wider LR range). Set attention is neutral today, possibly because one temperature cannot serve both M=5 and M=300 feeds. Both are logit-geometry fixes with small priors, so they are bundled; dissect only if WIN.
* **Code**: `Attn.__init__`: `with extra_init(): self.qn, self.kn = nn.RMSNorm(dh, eps=1e-6), nn.RMSNorm(dh, eps=1e-6)` (shared across heads, as in Qwen3); `self.ss = nn.Parameter(torch.full((h,), 1 / math.log(n_ref)))`, where n_ref = 64 for set blocks (median train M 48, val 67) and 192 for cross and hist_enc (median min(H,256)+1 about 192-226). In `forward`, after the transposes: `q, k = self.qn(q), self.kn(k)`; `n = kmask.sum(-1).clamp_min(2).float()`; `q = q * (self.ss[None, :, None, None] * torch.log(n)[:, None, None, None]).to(q.dtype)`. All 5 modules. 72 params per module.
* **Run**: `... --variant attn_qkss_s{23,24} $BASE --seed {23,24} --qk_norm --ssmax`
* **Diagnostics**: learned γ norms and s_h; attention entropy vs M in set block 0 (4 M buckets).
* **Pre-registered follow-up** (only if A1 or A3 WIN, in closeout): the combined winners at `--lr 2e-3`. Gating and QK-norm are the LLM "stability enables higher LR" pair.

### A4. TabR-style candidate-conditioned retrieval values  [priority 4]
* **Math** (TabR, arXiv 2307.14338): take the top-m history items by raw cosine S (m=16). Keys come from a shared projection of the raw normalised embeddings, `k_c = W_K ê_c`, `k_j = W_K ê_j` (128 -> 64, no bias). Logits: `ℓ_cj = −‖k_c − k_j‖² + a·S_cj + β[type_j]`, plus one learned sink logit `s0` whose value is zero. Values: `v_cj = E_type[type_j] + E_src[src_j] + E_rec[⌊log2(1+j)⌋] + T(k_c − k_j)`, with `T = Linear(128→d, no bias) ∘ Dropout(0.1) ∘ ReLU ∘ Linear(64→128)`. Then `r_c = Σ_j softmax(ℓ)_j v_cj` and `c ← c + Z(LN(r_c))`, with Z a zero-init `Linear(d, d)`.
* **Why**: today the target-attention values `W_V h_j` do not depend on the candidate. The model never sees how a matched item differs from the candidate (near duplicate vs related item vs same category). In TabR the `T(k − k_i)` value term is the reported turning point. Ablating TabR's L2 key similarity or its value terms brings it back to MLP level. Cost [B,M,16,128] is about 67M elements per batch, the same order as the existing attention scores. Zero-init Z makes it an exact no-op at init.
* **Code**: new `HistRetrieval` module (under `extra_init`), called in `Ranker.forward` after `c_ln` / domain / extra and before the cross loop: `c = c + self.retr(ce, he, hv, het, hes, S)`. Reuse S from the sim-bias computation. Gather with `idx = S.masked_fill(~hv[:, None], -2).topk(16, -1).indices` and flat indices; invalid picks get logit −inf (H=0 feeds then read only the sink, so r = 0). Learnable scalar a (init 4.0); `β = Embedding(11, 1)`, zero init. Compute the softmax in fp32.
* **Run**: `... --variant attn_retr_s{23,24} $BASE --seed {23,24} --retr_m 16`. Memory gate 50 GB.
* **Diagnostics**: `‖Z(LN(r))‖ / ‖c‖`; mean sink mass; sink mass for feeds with smax < 0.5 vs > 0.9.

### A5 (optional). Fine-grained shared-expert MoE FFN  [priority 5; run only if A1-A4 are done]
* **Math** (DeepSeekMoE / DeepSeek-V3 / Ling 2.0 / MiniMax-M2 routing): `y = FFN_shared(x) + Σ_{i ∈ TopK(s + b)} (s_i / Σ_{j ∈ TopK} s_j) · E_i(x)`, with `s = σ(W_r x)` (sigmoid router, fp32). The bias b is used **only for selection**. It is a non-gradient buffer updated after every optimizer step by `b_i ← b_i + γ·sign(mean_load − load_i)`, γ = 1e-3, with load counted over valid candidate tokens (aux-loss-free balancing). Experts are SwiGLU `E_i(x) = W2_i(silu(W1_i x) ⊙ W3_i x)` of width 64. There are 8 routed experts, top-2, plus 1 shared SwiGLU expert of width 128, so the activated width 128 + 2·64 = 256 matches the baseline FFN (2d).
* **Why include it, and why it is last**: the user asked about MoE. MoE raises parameters at constant activated FLOPs. That pays off when capacity is binding (Ling's efficiency-leverage law grows with compute). Here d=256 was neutral and labels are scarce (about 6 positives per parameter), so the capacity argument does not apply. The one plausible mechanism is conditional computation over heterogeneous regimes: 54 verticals, user-0 and no-history feeds, short vs long feeds. This is a falsification test at matched activated compute.
* **Code**: replace `Block.ff` with `MoEFF` only in the 4 candidate blocks (cross and set), not in hist_enc. Use dense dispatch: compute all 8 experts as one batched matmul (8·64 = 512 hidden; cheap at this size), then multiply by the sparse gate (zeros off the top-2). The module stores `self.last_load`; `train.py` updates b after `opt.step()`. Router weights go in the no-decay group. About +0.4M params.
* **Run**: `... --variant attn_moe_s{23,24} $BASE --seed {23,24} --moe 8 --moe_topk 2`
* **Diagnostics**: expert load entropy per block (should stay > 0.9·log 8); mutual information between expert choice and category (the hoped-for mechanism); max/min load ratio.

**attn closeout**: if there are ≥ 2 WIN/LIKELY items, run the combination (2 seeds); also the pre-registered lr 2e-3 arm if A1 or A3 won. Then run stage 2 and the baseline fine-tune (C0 recipe) on the best config, paired with `s2_base_s*` / `ftb_s*`.

---

## 3. Group feat: feature and tabular path (copy `<repo>/nnx/feat`)

The candidate token is `Linear([e_c, e_c⊙u]) + cat + MLP(63 standardised dense features)`. Only the main loss supervises it (about 1.7 positives per feed). This group improves how numbers enter the network and adds dense supervision and content-derived signals that survive item churn (39% of test candidate rows are cold items).

**Shared infrastructure (build once, first)**: a generic dense-extras mechanism. `dense_features(b, n_dense, extras=())` appends named feature blocks **after** the user-0 pair and **before** `xt`. `Ranker.cfg["dense_extras"]` (a tuple of names) is passed from `Ranker.forward`, `train.py` and `common.feature_stats`. The `xt` condition becomes `n_dense > N_DENSE + 2 + n_extra_cols`. This keeps f_mean/f_std, the wide term and `c_mlp` consistent.

### B1. Piecewise-linear numeric embeddings (PLE, rtdl version B)  [priority 1]
* **Math** (Gorishniy et al. arXiv 2203.05556; TabM; RealMLP): for feature j with bin edges `b_0 < … < b_T`, `e_t(x) = clamp((x − b_{t−1}) / (b_t − b_{t−1}), 0, 1)`; the first bin is unclamped below and the last unclamped above. Embedding `φ_j(x) = x·w_j + β_j + Σ_t e_t(x)·v_{j,t} ∈ R^12`, with **v zero-init**, so training starts as a linear embedding (a strict superset of today). `c_mlp` input = `flatten(φ(X)) ∈ R^{63·12}`. The `wide` term stays on raw X.
* **Why**: on MSLR-WEB10K (the only LTR set in those benchmarks) MLP-PLE/PLR beat a plain MLP and match a 5-seed MLP ensemble. In RealMLP, dropping numeric embeddings costs +2.3% error. Our f1..f5 have point masses at floor/ceiling (15-18%) with very different target rates (f1 floor/mid/ceiling 0.0097/0.0239/0.0413, f2 non-monotone). A one-layer GELU MLP on a standardised scalar handles such steps poorly; PLE handles them directly.
* **Code**: `NumEmbPLE` module (under `extra_init`), T=16. Edges are per-feature quantiles of the standardised and clamped X over the same 40 `feature_stats` batches, deduplicated, with the f1..f5 floor/ceiling values forced in as edges (raw-f columns are indices 0, 3, 6, 9, 12) and padded to T+1 with zero-width bins masked out. Store them as a buffer (`ple_edges` [F, T+1]) so checkpoints carry them. `c_mlp[0]` becomes `Linear(63·12, 2d)`. Memory [32768, 63, 16] is about 33M floats.
* **Run**: `$PY train.py --variant feat_ple_s{23,24} $BASE --seed {23,24} --num_emb ple --ple_bins 16 --ple_dim 12`
* **Diagnostics**: `‖v‖` per feature (which features use the non-linear part); per-feature attribution of the top 5.

### B2. History-supervised item propensity head  [priority 2]
* **Math**: `π_θ(t | ê, cat) = softmax(MLP([ê; E_cat'(cat)]))` over 11 classes: 0 = type1∧src1 (≈ the target: card open from the feed), 1 = type1∧src≠1, 2..10 = event types 2..10. Train it with CE on every valid history token, `L = L_main + λ·mean_{valid h} CE(π(e_h, cat_h), class_h)`, λ = 0.05. Candidates receive `ℓ_c = log π(· | ê_c, cat_c)` **detached** through a zero-init `Linear(11, d)` into the token (before `c_ln`) and a zero-init `Linear(11, 1)` into the fp32 logit.
* **Why**: labels are the binding constraint (5.45M positives), while history holds 584M typed events (type1∧src1 = 28% of val history). The EDA found the target is essentially a type-1/source-1 open. The head learns "what kind of interaction does this content attract" from roughly 100x more labels. It is content-based, so it covers cold items, and it uses no information from the same user's later snapshots. TransAct-V2 shows the related next-action loss helps. Detaching keeps it a clean supervised feature rather than a second candidate MLP.
* **Code**: `PropHead`: `Linear(128+32, 128) -> GELU -> Linear(128, 11)` plus `prop_cat = Embedding(54, 32)`, all under `extra_init`. Its loss is stored as `self.prop_loss` in `forward` when training, and `train.py` adds `a.prop_head * model.prop_loss`. Class from `het`, `hes`; mask `hmask & hemb_ok`.
* **Run**: `... --variant feat_prop_s{23,24} $BASE --seed {23,24} --prop_head 0.05`
* **Diagnostics**: prop-head CE on val history (vs the marginal-entropy baseline); correlation of `ℓ_c[0] − log prior` with the label on val.

### B3. Semantic-ID codes (residual k-means of the frozen content embedding)  [priority 3]
* **Math** (RQ-k-means as in OneRec / QARM): `c1 = argmin_k ‖ê − C1_k‖` (K1 = 256); `r = ê − C1_{c1}`; `c2 = argmin_k ‖r − C2_k‖` (K2 = 64), giving 16,384 cells of about 5K train items each. That is not item-level, which matters because the N4 item-ID table memorised items and cost −0.0075 on cold-positive feeds. Learnable `z = W_z [E1(c1); E2(c1·64 + c2)]` (32 + 32 dims; W_z zero-init, separate W_z for candidates and history) is added before `c_ln` and `h_ln`. SID-dropout in training: zero the E2 part with p=0.5 and the whole z with p=0.2, per token. Dense extras (5): `log1p(#hist with same c1)`, `log1p(#hist with same (c1,c2))`, `log1p(#type1∧src1 hist with same (c1,c2))`, `1/(1 + first k with same (c1,c2))` (0 if none), and the within-feed z-score of the second.
* **Why**: content clusters always exist, including for cold items. Learnable code embeddings can absorb cluster-level collaborative signal (popularity, CTR) that the frozen two-tower embedding lacks. The leak-safe collaborative inputs (`--xt cv`) were worth +0.0032, so this signal matters. Code-match counts generalise the useful 54-category affinity features to a much finer neighbourhood. QARM reports +0.18..+0.35% AUC; LiGR SID n-grams +0.2%.
* **Offline (CPU, ≤ 24 threads, no GPU slot)**: `nnx/feat/sid.py`. Fit spherical k-means (k-means++ init on 50K rows, 25 Lloyd iterations) on 2M rows sampled from the train item table (int8 -> float32, L2-normalised), then on the level-1 residuals. Assign every row of the train/val/test tables in 1M-row chunks, and write `<repo>/nnx/feat/sid/{split}_codes.npy` (int16 [n_items, 2]). Missing embeddings get c1 = 256 and pair id = 16384. Run it while B1/B2 train.
* **Code**: `data.py` loads the codes to GPU; `gather` adds `ccode`/`hcode`. Embedding names contain `emb` (`sid_emb1`, `sid_emb2`), which keeps them out of weight decay under the existing filter. dense extras use the name `sid`.
* **Run**: `... --variant feat_sid_s{23,24} $BASE --seed {23,24} --sid`. Memory gate 50 GB.
* **Risk check (mandatory)**: watch for the N4 signature, i.e. val falling in epoch 2 while the holdout rises. Report val at steps 10000 vs 19670 and the val-holdout gap vs the baseline. If the holdout-selected step is ≤ 10000 and val drops after it, record NO (memorisation) regardless of the score.

### B4. Multi-scale recency-decayed (EWMA) history features  [priority 4]
* **Math**: weights `w_k^τ = exp(−k/τ)·hv_k` over recency index k, for τ ∈ {4, 16, 64}:
  `dmean_τ = Σ w S_ck / Σ w`; `dlse_τ = (1/5)·log(Σ w e^{5 S_ck} / Σ w)`; `dcat_τ = Σ w·1[cat_k = cat_c] / Σ w` (hmask instead of hv for the category). Add `dlse_16` and `dcat_16` restricted to type1∧src1 events, and the within-feed z-score of `dlse_16`: 12 features in all. Empty sums give sentinels (−1.5 for similarities, 0 for shares).
* **Why**: this is classic exponential smoothing at several half-lives, the time-series view of a history with only an order index. The current inputs are hard windows (max over 8/32/all). In the fine-tune branch a single half-life-5 decayed cosine (`seq_dcos_hl5`) belonged to the useful SEQ group. It is cheap and complements A2's learned decay (that is the attention route; this is the explicit-feature route).
* **Code**: a dense extra named `ts` in the shared mechanism, computed in `dense_features` from the existing S, E and `het`/`hes` tensors.
* **Run**: `... --variant feat_ts_s{23,24} $BASE --seed {23,24} --ts_feats`

### B5 (optional). TabM-mini BatchEnsemble, k = 4, on the candidate path  [priority 5]
* **Math** (TabM, arXiv 2410.24210): member i uses `W_i = W ⊙ (s_i r_i^T)`. TabM-mini keeps only the first adapter: `x_i = r_i ⊙ x`, with r_i trainable and initialised to random ±1. Apply `r_i^{(1)} ∈ R^256` on `[ê_c, ê_c⊙u]` and `r_i^{(2)} ∈ R^63` on X, then run c_lin/c_mlp and the cross and set blocks on a k·B batch. The history encoder runs once and is shared via `repeat_interleave`. Loss = **mean of the members' listwise losses** (TabM: "the mean loss must be optimised"), and K9 aux is averaged over members. Prediction = mean of member logits (fp32).
* **Why last**: the 16-seed ensemble only gained +0.0005, so any gain must come from TabM's regularisation (on MSLR, TabM-mini 0.7423 vs MLP 0.7475 RMSE and the 5-seed ensemble 0.7460). It costs about 3x compute and +20 GB, on a shared GPU.
* **Run**: `... --variant feat_tabm_s{23,24} $BASE --seed {23,24} --tabm 4`. Memory gate 70 GB; run it only after B1-B4 are judged.

**feat closeout**: combine WIN/LIKELY items (2 seeds), then stage 2 and the baseline fine-tune on the best config, paired vs `s2_base_s*` / `ftb_s*`. Note that the fine-tune branch already holds collaborative features, so B3's gain may shrink after fine-tune. Report both numbers.

---

## 4. Group optim: optimisation and generalisation (copy `<repo>/nnx/optim`)

### C0. Shared baseline (first, see 1.2) + frozen fine-tune features
1. Run `base_s23` and `base_s24` (with `--ema`), then `s2_base_s23` and `s2_base_s24`.
2. Freeze the fine-tune feature set, because feature dirs get pruned by other agents and 7 of the 56 `n2clean` files are already gone. Resolve the `n2clean` preset against what is on disk now (49 features at design time). Write the list to `<repo>/nnx/optim/n2clean_frozen.txt` (comma-separated) and copy those files to `<repo>/nnx/optim/feats/{val,test}/`. At design time none had ±inf; 7 had 0.9-1.4% NaN, which is just the no-history feeds.
3. Baseline fine-tune (production recipe) on each stage-2 checkpoint:
   `NNX_FEATS=<repo>/nnx/optim/feats $PY finetune.py --variant s2_base_s23 --out ftb_s23 --extra $(cat n2clean_frozen.txt) --lr 5e-5 --epochs 6` (same for s24). Outputs: `preds/nn_ftb_s{23,24}_val_oof.npy`.

### C1. Weight EMA in stage 1 (measured inside C0, no extra runs)  [priority 1]
* **Math**: `θ̄ ← β θ̄ + (1 − β) θ` after every optimizer step, starting from a copy of θ at `ema_start · total` (50%). The time constants are 1/(1−β) = 1000, 2000 and 5000 steps. WSM view (Ling 2.0's schedule): merging is a synthetic decay, `θ̂ = θ_n − Σ_i w_i g_{n+i−1}`. With cosine annealing to 3% the expected gain is small (Ajroldi et al. 2025), but EMA also removes checkpoint noise (±0.0003 between evals), which makes every other A/B cleaner.
* **Code**: `--ema` (comma list of betas) and `--ema_start`. Keep one fp32 `copy.deepcopy(model)` per β and update it with `torch._foreach_lerp_(ema_params, params, 1 − β)` (buffers are constant; there is no BatchNorm). At each eval after the start, score every EMA on holdout and val. Each EMA keeps its own holdout-selected best, saved as `nn_<variant>_ema<β>_trainonly_val.npy`, `nn_<variant>_ema<β>_holdout.npz` and `ckpt/<variant>_ema<β>.pt`. Raw-weight selection and outputs are unchanged.
* **Judge**: `paired.py --base .../nn_base_s{seed} --treat .../nn_base_s{seed}_ema0.9995` (and the other two betas; pick one β on the **holdout** only).

### C2. Muon for hidden matrices  [priority 2]
* **Math** (Moonlight / Kimi K2 RMS matching; Qwen3.8-Next uses Muon): Nesterov momentum `M_t = μM_{t−1} + G_t`; `O_t = NS5(M_t)` (quintic Newton-Schulz ≈ `UV^T`, i.e. steepest descent under the spectral norm); `W ← (1 − ηλ)W − η·0.2·sqrt(max(A,B))·O_t`. The 0.2·sqrt(max(A,B)) factor matches the AdamW update RMS, so lr 1e-3 and wd 0.02 carry over.
* **Code**: `--muon` implies `split_kv=True` (the fused `kv` Linear becomes separate `k` and `v` so each is orthogonalised on its own; this is part of the treatment). `opt_m = torch.optim.Muon(P_muon, lr=a.lr, weight_decay=a.wd, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")`, where P_muon = every 2-D `nn.Linear` weight with min(out, in) > 1 (q, k, v, o, ff, c_lin, h_lin, u_lin, c_mlp, head.1, aux_head.1; the Moonlight rule: all matrices except embedding tables and 1-output layers). `opt_a = AdamW(rest)` with **the same decay/no-decay grouping as the baseline** (do not also fix the embedding-wd quirk; that would confound). Apply the same `lr_at(step)` to both optimizers, `clip_grad_norm_` over all params before both steps, then step both.
* **Run**: `$PY train.py --variant optim_muon_s{23,24} $BASE --seed {23,24} --muon`
* **Pre-registered follow-up**: `--lr 2e-3` (Muon arm only) if lr 1e-3 is not NO.
* **Caveat for closeout**: AdamW-pretrained models fine-tune poorly with Muon and vice versa (Moonlight). Stage 2 and fine-tune the Muon winner with AdamW (default). If the gain disappears after fine-tune, report that rather than chasing it.

### C3. Fine-tune recipe: cosine decay + weight EMA (arm a), + LP-FT (arm b)  [priority 3]
* **Why**: the fine-tune over-fits after epoch 2 (n2full OOF by epoch: 0.3827, 0.3859, 0.3861, 0.3855, 0.3845, 0.3842, 0.3835) at a constant LR. The snapshot mean of predictions is a crude average. EMA plus annealing is the principled version (Ajroldi 2025: averaging and annealing combined is best). LP-FT (Kumar et al. 2022): fit the new zero-init 49-feature branch before the backbone moves, so its gradients do not distort pretrained features. That gives about +1% in-distribution and about 10% under shift.
* **Math**: arm a: `lr(s) = lr·mult·min(1, (s+1)/30)·½(1 + cos(π s / S_total))` over E = 4 epochs; EMA β = 0.99 (about 100 steps ≈ 1 epoch per fold), started at epoch 0.5. Predict OOF and test **once, from the EMA weights** at the end (no snapshot). Arm b: phase A = 1 epoch with only `extra.*`, `head.*`, `wide.*` trainable at lr 5e-4 (constant after warmup); phase B = arm a on all params, with `--extra_lr_mult 3`.
* **Code** (`finetune.py`): `--decay {const,cos}`, `--ema BETA`, `--ema_start` (in epochs), `--lpft_epochs N`, `--lpft_lr`. Rebuild param groups at the phase boundary (toggle `requires_grad`). Keep per-epoch OOF logging of the EMA weights for diagnostics only.
* **Runs** (pairs are the two stage-2 checkpoints):
  `... finetune.py --variant s2_base_s{23,24} --out fta_s{23,24} --extra $(cat n2clean_frozen.txt) --lr 5e-5 --epochs 4 --decay cos --ema 0.99 --ema_start 0.5`
  `... finetune.py --variant s2_base_s{23,24} --out ftlp_s{23,24} --extra $(cat n2clean_frozen.txt) --lr 5e-5 --epochs 4 --decay cos --ema 0.99 --ema_start 0.5 --lpft_epochs 1 --lpft_lr 5e-4 --extra_lr_mult 3`
* **Judge**: `paired.py --val_suffix _val_oof.npy --base .../nn_ftb_s{seed} --treat .../nn_fta_s{seed}` (no holdout). Fine-tune decisions are made on val OOF, so stop at these 2 arms (protocol: ≤ 3 coarse choices on val). WIN requires both checkpoints positive and pooled z ≥ 2.

### C4. LLM decay-phase recipe: WSD + exploration-data cooldown + aux annealing  [priority 4]
* **Math**: warmup 500, then constant `η_max` until `T_s = 0.8T`, then the 1-sqrt cooldown `η(t) = η_min + (η_max − η_min)(1 − sqrt((t − T_s)/(T − T_s)))`, with η_min = 0.03 η_max (the baseline floor). During the cooldown, each step draws a batch from the random-order exploration pool with probability 0.3: train feeds with p_random > 0.5 and M ≥ 50, **holdout excluded**, the same pool as stage 2 (about 134K feeds after removing the holdout). Aux (K9) weight annealing: `w_aux(t) = 0.1 + 0.4·(η(t) − η_min)/(η_max − η_min)`, i.e. 0.5 → 0.1 over the cooldown. The per-feed aux weight (1 − p_random) stays, so exploration batches carry almost no aux.
* **Why**: these are three verified LLM practices. Hägele 2024: 1-sqrt cooldown ≥ cosine, plateauing at about 20% cooldown. MiniCPM: high-quality data in the decay phase (here, unbiased exploration feeds). MiniMax-M2 / DeepSeek-V3: anneal the auxiliary MTP weight 0.3 → 0.1 in the decay phase (here, the biased production-order teacher). Up-weighting exploration feeds for the whole run hurt (−0.001), but a separate stage 2 helped (+0.0012). This tests whether the gain is larger when exploration data arrives during the anneal. The run also tells us where the train stage's headroom is.
* **Code** (`train.py`): `--sched {cos,wsd}`, `--cooldown_frac 0.2`, `--cooldown_mix 0.3`, `--aux_anneal 0.1`. Pre-build exploration batches with `tr.batches(pool, ...)` (reshuffle when exhausted); draw with an rng seeded by `a.seed + 1`; keep total steps = 19670 (equal compute). Pool = `feeds[(prand_np[feeds] > 0.5)]` after the holdout `setdiff1d`.
* **Run**: `$PY train.py --variant optim_wsd_s{23,24} $BASE --seed {23,24} --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1`
* **Judge**: primary vs `s2_base_s{seed}` (equal data, less compute than stage 1 + stage 2); secondary vs `base_s{seed}`. Do not apply the kill rule before step 15700 (WSD trails cosine until the cooldown by construction).

### C5. Born-again listwise self-distillation  [priority 5]
* **Math**: per feed, `L = CE(y/Σy, softmax(z_S)) + β·τ²·KL(softmax(z_T/τ) ‖ softmax(z_S/τ)) + 0.5·L_K9`, with β = 0.5 and τ = 1, using the same equal-per-feed normalisation as the main loss (sum / max_feeds).
* **Why**: train feeds have about 1.7 positives among about 78 candidates, so the label says nothing about the order of the remaining ~76. A dense listwise target already gave +0.0024 once (K9, production order). A frozen model's own ranking is a second, less biased dense target (Born-Again Networks: same-size students beat their teachers).
* **Teacher**: `<repo>/nn/ckpt/s2_v2_feed_k9.pt` (read only; val 0.38271). Both of its stages excluded the same 30K holdout, so holdout evaluation stays clean. Precompute its logits for the whole train split once: `nnx/optim/teacher.py` loads the checkpoint into the copy's `Ranker` and runs `predict(model, tr)` with no `tr.dom` (random-like domain, the teacher's unbiased view), saving `<repo>/nnx/optim/teacher/s2_v2_feed_k9_train.npy` as fp16 (247M × 2 B ≈ 0.5 GB). Memory gate 35 GB, about 5 min.
* **Code** (`train.py`): `--distill PATH`, `--distill_w`, `--distill_tau`. Load as a GPU fp16 tensor; per batch `zT = T[b["ci"]].float().masked_fill(~m, -1e4)`; compute the KL in fp32 with the padded rows masked.
* **Run**: `$PY train.py --variant optim_kd_s{23,24} $BASE --seed {23,24} --distill <repo>/nnx/optim/teacher/s2_v2_feed_k9_train.npy --distill_w 0.5 --distill_tau 1`
* **Caveat**: the teacher is seed 23 of the same config, so the seed-23 student starts from the teacher's own init. Seed 24 is the cleaner read; report the seeds separately. Also report the student-teacher prediction correlation vs the base-teacher correlation on val (ensemble-diversity cost).

**optim closeout**: if C1 wins, turn the chosen β on for every closeout run. Combine C2/C4/C5 winners (2 seeds), then stage 2 and the best fine-tune recipe (C3 winner or the baseline recipe).

---

## 5. Schedule per experimenter (2 GPU slots each; one slot is about 25-45 min on the shared GPU)

| slot | attn | feat | optim |
|---|---|---|---|
| 0 (t=0) | H0; implement A1; launch A1 s23/s24 as soon as ≥ 40 GB is free | H0 + dense-extras infra; implement B1; launch B1 | H0; launch **base_s23/24** (+EMA) immediately; freeze features |
| 1 | A2 (implement while A1 runs) | B2; start the B3 k-means on CPU | s2_base ×2 → ftb ×2 → teacher logits |
| 2 | A3 | B3 | C2 Muon |
| 3 | A4 (written during slots 0-2) | B4 | C3 fine-tune arms (4 short runs) |
| 4 | A5 (optional) or closeout | B5 (optional) or closeout | C4 WSD |
| 5 | closeout | closeout | C5 KD; closeout |

Judge each experiment as soon as its 2 runs finish (results.tsv), and kill early per 1.6.

## 6. Integration (after all three groups; for the lead / next workflow step)
1. Union of WIN items across groups → one config, 2 seeds, paired vs base. Check additivity: expect less than the sum of the parts.
2. The same config **with `--xt cv`** (round-2 winner) vs `v4_cv` from the NN agent (when its seed matches), then stage 2 and the best fine-tune recipe on the frozen `n2clean` list. Only then is a trick "in" for the final ensemble.
3. Recipe diversity for the ensemble: a WIN that also lowers the correlation with the baseline (e.g. Muon, TabM, PLE) is worth adding as a separate member even if it is merely LIKELY. Seeds alone are saturated (16 runs = +0.0005).

## 7. Trick mapping: considered, adopted and rejected (with reasons)

| source | trick | decision | reason |
|---|---|---|---|
| Qwen3-Next / 3.5 / 3.8 | sigmoid output-gated attention | **A1** | sink-free attention, verified across scales; cheap |
| Qwen3, Gemma 3, Ling 2.0 | QK-norm | **A3** | bounded logits next to the cosine bias; enables higher LR |
| Qwen3.5 / Qwen3-Next GDN, FoX, Mamba | data-dependent decay / forget gate | **A2** (inside softmax attention) | time-series decay without giving up exact retrieval |
| SSMax, TabICLv2 | log-n temperature | **A3** | M spans 5..300; set attention neutral |
| DeepSeek-V3, Ling 2.0, MiniMax-M2, Qwen3.5 | fine-grained MoE + shared expert, sigmoid router, aux-loss-free bias | **A5**, optional | capacity is not binding (d=256 neutral, about 6 positives/param); tested once at matched activated width |
| Kimi / Moonlight, Qwen3.8 | Muon | **C2** | free in torch 2.11; RMS-matched LR |
| Ling 2.0 WSM, modded-nanogpt | checkpoint merging / EMA | **C1**, C3 | near-free; less checkpoint noise |
| MiniCPM, Hägele 2024 | WSD + decay-phase data | **C4** | exploration feeds are our "high-quality" data |
| DeepSeek-V3, MiniMax-M2, Ling 2.0 | MTP-weight annealing | **C4** (aux 0.5 → 0.1) | the production-order teacher is biased |
| TabR, DIN | candidate-conditioned retrieval values | **A4** | values currently ignore the candidate |
| rtdl PLE/PLR, RealMLP, TabM | numeric embeddings | **B1** (PLE; PLR parked) | point masses in f1..f5; PLE starts linear, so no σ to tune |
| TabM | BatchEnsemble | **B5**, optional | regularisation; 3x cost |
| OneRec, QARM, LiGR | semantic IDs | **B3** (coarse 256×64 cells) | cold-item-safe collaborative signal; avoids the N4 failure |
| TransAct-V2, HSTU | history as supervision | **B2** (propensity head); next-action loss parked | 100x more typed events than labels |
| BAN, KD | self-distillation | **C5** | dense listwise target, analogous to K9 |
| MiniMax-M3.1-Flash / M2, Qwen3.8 QSA, DSA/MSA, Lightning, KDA, MLA, SWA | linear / sparse / latent attention | rejected | n ≤ 300 keys; target attention is retrieval (M2 itself kept full attention for retrieval) |
| Qwen3.8 Gated Residual, mHC, value residual, AttnRes | residual-stream fixes | rejected | about 5 sublayers deep |
| Gemma 2 softcap, z-loss | logit control | rejected | QK-norm supersedes attention softcap; listwise CE is shift-invariant (z-loss / output softcap do nothing) |
| Qwen3.8 n-gram tables, Engram, hashed IDs | memorisation tables | rejected | N4 measured −0.0022 (−0.0075 on cold-positive feeds) |
| SwiGLU / RMSNorm / small-init bundle | hygiene | parked | FFNs are a small share of the model; used inside A5's experts |
| RealMLP robust scaling, smooth clipping | preprocessing | rejected | dense features are already well conditioned (|z| > 6 only at the no-history sentinel) |
| DCNv2 / GDCN, FT-Transformer, RankMixer | feature crossing / tokenisation | rejected or parked | MSLR: DCN2 ≈ MLP; RankMixer gains are at 1e8 params; DCNv2 is already inside the NN agent's `--arch din` |
| label smoothing, mixup, stochastic depth, permutation TTA | regularisers | rejected | ranking metric (smoothing hurts); depth 5; the model is permutation-equivariant |
| width / depth / heads, static recency bias, set-attn removal, BCE, position tower, domain / user-0, intent head, history-length TTA, exploration up-weighting, `--xt`, `--arch din/gru`, pseudo-labels | already tested or owned by the NN agent | do not repeat | nn_report.md rounds 1-2 |

## 8. Parking lot (only if a slot frees up; each is one 2-seed run)
* Attention-probability dropout 0 (stage 1 is under-fitted; sim-bias heads retrieve one event, and 10% dropout removes it).
* PLR periodic embeddings (σ = 0.1 on standardised inputs) if B1 is NO.
* Grouped LN / separate user-token parameters in hist_enc (OneTrans / MTGR).
* Duplicate-aware target attention (TWIN-V2: `+β_h ln n_item`; 48% of history events sit on repeated items).
* A 3-epoch cosine probe (val still rising at the last step).
* R-Drop (α = 0.5) or SAM (ρ = 0.05) in fine-tune; 10-fold fine-tune with a full-val refit for test.
* Next-action loss on type1∧src1 history events (TransAct-V2 NAL with impression negatives).
