# NNX group "feat" (experimenter 2): numeric embeddings, history supervision, semantic IDs, EWMA history features

Date: 2026-11-11. Code: `<repo>/nn_x/feat/` (copy of `nn/`). Predictions: `<repo>/preds_x/feat/`.
Per-comparison log: `<repo>/nn_x/feat/results.tsv`. Design: `reports/nnx_design.md`, section 3.
All runs used GPU N only, at most 2 of my training processes at a time, 16 CPU threads per process.

## 0. Summary

Every arm is a paired A/B against the shared baseline of group optim (`base_s23`, `base_s24` in
`<repo>/preds_x/optim/`; v2_feed_k9 config, 2 epochs, holdout-selected checkpoint). Both seeds (23, 24) were trained for the
full 2-epoch recipe. The deltas are pooled per-feed paired differences (val: 14,423 feeds; holdout: the fixed 30K random-like
train feeds). Verdicts follow design rule 1.6.

| # | experiment | flag | Δ val ± SE | Δ holdout ± SE | per seed Δval (s23 / s24) | verdict | cost |
|---|---|---|---|---|---|---|---|
| B1 | PLE numeric embeddings (T=16, d_e=12) | `--num_emb ple` | −0.00043 ± 0.00060 | **+0.00078 ± 0.00037** (z 2.1) | −0.00118 / +0.00032 | **NO** (val < 0) | 1.0x time, +0.19M params |
| B2 | history-supervised propensity head | `--prop_head 0.05` | +0.00007 ± 0.00040 | +0.00014 ± 0.00024 | +0.00001 / +0.00013 | **NO** (neutral) | 1.0x, +0.02M |
| B3 | semantic-ID codes (RQ-k-means 256×64) + code-match extras | `--sid` | +0.00038 ± 0.00049 | +0.00003 ± 0.00028 | +0.00075 / +0.00000 | **NO** | 1.0x, +0.55M (sparse) |
| B4 | multi-scale EWMA history features (12) | `--ts_feats` | −0.00002 ± 0.00045 | −0.00021 ± 0.00026 | −0.00002 / −0.00002 | **NO** (neutral) | 1.0x, +3K |
| B5 | TabM-mini BatchEnsemble k=4 (3 seeds) | `--tabm 4` | +0.00009 ± 0.00042 | +0.00032 ± 0.00025 (z 1.3) | +0.00024 / −0.00003 (s25 +0.00006) | **LIKELY** by rule, practically neutral | ~1.45x time, +1.3K params |
| B1 closeout | PLE + stage 2 | | −0.00002 ± 0.00065 | **+0.00092 ± 0.00039** (z 2.4) | −0.00067 / +0.00062 | NO | |
| B1 closeout | PLE + stage 2 + baseline fine-tune (OOF) | | −0.00010 ± 0.00064 | n/a | −0.00013 / −0.00007 | NO | |

Sources of the tricks: B1 rtdl numeric embeddings / RealMLP / TabM (tabular DL 2022-2025); B2 history-as-supervision
(TransAct-V2, HSTU); B3 semantic IDs by residual k-means (OneRec, QARM, LiGR); B4 classic multi-half-life exponential smoothing (the
explicit counterpart of the Qwen3.5 / Gated-DeltaNet decay gates that group attn tested inside attention as FoX); B5 BatchEnsemble
(TabM). The LLM-side tricks (gating, QK-norm, MoE, Muon, WSD, EMA) belong to groups attn and optim.

Reference scale: same config and seed re-run on a different day differs by +0.0004 on val (GPU non-determinism, optim C0 check). Seed-to-seed
spread is 0.001.

Main findings:
1. None of B1-B4 changes val NDCG beyond noise. The feature path is not where the remaining headroom is: the information these
   tricks add (finer numeric shape, item-level event-type propensity, content-cluster memory, smoothed history similarity) is already
   available to the network through the 63 dense features, the 128-d content embedding and the history target attention.
2. **B1 (PLE) is the one arm with a significant effect, and it is distribution-specific.** It is +0.0008 on the train-side holdout
   (z = 2.1, both seeds positive) but −0.0004 on val. Val loses most on feeds with many f1/f2/f3 floor candidates (−0.0011..−0.0014
   vs +0.0001..+0.0006 on the other half). Those are the point masses whose target rate differs between production and
   exploration traffic. PLE learns the production-traffic step functions more exactly, and they transfer worse.
3. PLE models are also the most *different* members: correlation with the baseline 0.983 vs 0.988 seed-to-seed, and a cross-seed
   base+PLE pair is +0.00046 ± 0.00043 over a base+base pair (no weights fitted). After stage 2 this grows to +0.0007 ± 0.0005, but
   the baseline val fine-tune removes it (+0.0002 ± 0.0004; single model −0.0001). So PLE is not a useful member of the
   fine-tuned final ensemble either (section 10).
4. B3 (semantic IDs) shows no memorisation: val rises to the end of epoch 2 and cold-positive feeds are +0.0008 ± 0.0013 (the N4
   item-ID table had −0.0075 there). The cell-level collaborative signal is real but small at this cell size.
5. B2's propensity head learns (val-history CE 1.846 vs marginal entropy 1.997) but its output is nearly uncorrelated with the label
   (point-biserial 0.012) and the model barely uses it.
6. B5 (TabM-mini) is LIKELY by the rule on 3 seeds, but only +0.0001 val. Its 4 members are 0.999-correlated: a multiplicative
   first-layer adapter cannot diversify members that share four attention/FFN layers. Not worth 1.45x compute.

`best_combo`: **none (keep the baseline feature path)**. No feat arm is WIN. The only LIKELY arm (B5) is +0.0001 val on 3 seeds at
1.45x cost. Its holdout signal (+0.0003, z 1.3) is what made it LIKELY, so it was not combined. A B5+B3 combination would be
val-driven cherry-picking of NO/neutral arms, which the protocol forbids.

Side outputs useful to other groups: the shared **`base_s25`** (lock `<repo>/nnx/base_s25.lock`;
`preds_x/feat/nn_feat_base_s25_{trainonly_val.npy,holdout.npz}`) and the semantic-ID codes of all three item tables
(`nn_x/feat/sid/{train,val,test}_codes.npy`).

## 1. Implementation (code: <repo>/nn_x/feat/)

Copy of `nn/` with the H0 harness patch: GPU-5-only guard, own `ckpt/` and `logs/`, predictions in
`<repo>/preds_x/feat/`, `eval_subset(return_scores=True)`, pairable holdout scores
(`nn_<variant>_holdout.npz`), `NNX_FEATS`, and `extra_init` (every new module is built under a forked CPU RNG).
Checked: with all flags off the state dict at seed 23 is bit-identical to `nn/model.py` (119/119 tensors) and the forward pass
matches exactly. With a treatment on, every baseline tensor keeps its baseline init. Only the tensors the treatment must change
differ: `c_mlp.0` and `wide` columns for new inputs (baseline columns copied, new columns drawn from the forked RNG), and the new
modules. Every option is a CLI flag that defaults to off, so old checkpoints still load.

| flag | what | files |
|---|---|---|
| (infra) | dense-extras mechanism: `dense_features(b, n_dense, extras)` appends named blocks after the user-0 pair and before `xt`; `cfg["dense_extras"]`; `Ranker.dense(b)` used by train / predict / fine-tune / feature_stats | model.py, common.py, train.py, finetune.py |
| `--num_emb ple --ple_bins 16 --ple_dim 12` | B1 `NumEmbPLE` + `ple_edges_from_sample` | model.py, train.py |
| `--prop_head 0.05` | B2 `PropHead` (+ val diagnostics `prop_diag`) | model.py, train.py |
| `--sid` | B3 code embeddings + 5 dense extras; codes from `sid.py` | sid.py, data.py, model.py |
| `--ts_feats` | B4 12 EWMA dense extras | model.py |
| `--tabm 4` | B5 TabM-mini | model.py, train.py |

Helpers: `paired.py` (design doc), `judge.py` (paired.py + rule 1.6 + `results.tsv` line), `curves.py` (per-eval deltas vs
`base_s{seed}.log`, for the kill rule), `sh/run.sh`, `sh/queue.sh`, `sh/queue2.sh`, `sh/closeout.sh`.
Reproduce any arm: `cd <repo>/nn_x/feat && source sh/env.sh && $PY train.py --variant feat_<x>_s23 $BASE --seed 23 <flags>`,
where `$BASE = --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout
--epochs 2 --eval_every 1000 --loss_norm feed --no_test`. The B3 codes come first: `OMP_NUM_THREADS=24 $PY sid.py` (CPU, 97 s).
Stage 2 / fine-tune: `bash sh/closeout.sh feat_ple 23`. Judge: `$PY judge.py --exp B1 --arm ple --treat feat_ple_s{seed}`.

## 2. B1: piecewise-linear numeric embeddings (PLE, rtdl version B)

**Math.** For feature j with edges `b_0 < … < b_T`: `e_t(x) = clamp((x − b_{t−1}) / (b_t − b_{t−1}), 0, 1)`, the first valid bin
unclamped below, the last unclamped above, zero-width padding bins masked. `φ_j(x) = x·w_j + β_j + Σ_t e_t(x)·v_{j,t} ∈ R^12`, with v
zero-initialised, so step 0 is a linear embedding. `c_mlp[0] = Linear(63·12, 256)`. The wide term stays on the standardised X.

**Edges.** From the 40 `feature_stats` batches (657K sampled candidate rows), standardised and clamped like the model input:
deduplicated quantiles at t/16. For the raw f1..f5 columns (0, 3, 6, 9, 12), every point mass with ≥ 1% sample frequency is forced in
as an edge, plus a step edge at p + 0.001 (p − 0.001 for the ceiling), so each point mass sits alone in a narrow bin and the
step in target rate is representable. Without the step edge, the floor value and the values just above it share one linear bin.
Near-duplicates (< 5e-4, float noise of the interpolated bins) are merged, the result is trimmed to ≤ 17 edges, then padded at the
front. Example f5 edges: 0 (floor, 1.3%), 0+δ, 1/62 (10.8% mass), 1/62+δ, …, 61/62−δ, 61/62. Valid bins per feature: 1-16
(mean 12.5); binary features get one bin.

**Results.**

| seed | val base → PLE | Δ val | holdout base → PLE | Δ holdout |
|---|---|---|---|---|
| 23 | 0.38216 → 0.38098 | −0.00118 ± 0.00073 | 0.30084 → 0.30104 | +0.00020 ± 0.00045 |
| 24 | 0.38124 → 0.38156 | +0.00032 ± 0.00074 | 0.29983 → 0.30119 | +0.00136 ± 0.00044 |
| pooled | | −0.00043 ± 0.00060 | | **+0.00078 ± 0.00037** |

Curve: PLE led on both metrics through epoch 1 (+0.0010..+0.0016 val and holdout at steps 2k-5k). In epoch 2 the holdout gain
stayed (+0.0002..+0.0021) while the val gain disappeared (seed 23 down to −0.0012).

**Mechanism.** The non-linear part ‖v_j‖ is largest for f5 raw (2.0), dot(cand, user) (1.5-1.7), cos(cand, user) (1.4-1.6), feed
category share (1.4), f4 raw (1.4) and f1 raw (1.3-1.4); mean 0.78. So PLE does model the f-point-masses and the user-similarity
shape. Val slices (pooled per-feed Δ):

| slice | Δ val PLE | slice | Δ val PLE |
|---|---|---|---|
| f1 floor share > median | −0.00140 ± 0.00088 | ≤ median | +0.00055 ± 0.00083 |
| f2 floor share > median | −0.00110 ± 0.00086 | ≤ median | +0.00024 ± 0.00084 |
| f3 floor share > median | −0.00093 ± 0.00085 | ≤ median | +0.00008 ± 0.00086 |
| M < 50 | −0.00184 ± 0.00112 | M 50-120 | +0.00155 ± 0.00108 |
| cold-positive feeds | −0.00066 ± 0.00168 | warm-positive | −0.00085 ± 0.00071 |

The losses concentrate where the floor point masses are frequent. The EDA (`nnx_research_tabular_dl.md` 1.1) shows that the floor
target rates shift between train and val (f1 floor/mid ratio 0.41 train vs 0.49 val; the f4 floor is neutral on train and very bad on
val). Fitting the train-side step functions more exactly helps the train-side holdout and does not transfer. Verdict **NO** (Δv < 0).
Parked follow-up (design parking lot): PLR periodic embeddings, not run.

## 3. B2: history-supervised item propensity head

**Math.** `π(t | ê, cat) = softmax(MLP([ê; E'_cat(cat)]))` over 11 classes (0 = type1∧src1, 1 = type1∧src≠1, 2..10 = event types
2..10). `L = L_main + 0.05·mean_{valid h} CE(π(ê_h, cat_h), class_h)`, trained on every valid history token. Candidates receive
`ℓ_c = log π(· | ê_c, cat_c)`, detached (computed under `no_grad`), through zero-init `Linear(11, d)` into the token before `c_ln` and
zero-init `Linear(11, 1)` into the fp32 logit. The main loss never reaches the head, so it is a clean supervised feature.

**Diagnostics (final eval).** Val-history CE 1.846 vs marginal entropy 1.997 (−7.6%); train CE 1.756-1.770. The head learns. The
point-biserial correlation of `ℓ_c[0] − log prior` with the candidate label on val is only 0.012-0.013. The logit weights stay
small (|w| ≤ 0.05; class 0 +0.049, class 1 −0.044), and `‖W_c‖` grows to 2.0.

| seed | Δ val | Δ holdout |
|---|---|---|
| 23 | +0.00001 ± 0.00055 | −0.00024 ± 0.00034 |
| 24 | +0.00013 ± 0.00055 | +0.00051 ± 0.00033 |
| pooled | +0.00007 ± 0.00040 | +0.00014 ± 0.00024 |

Slices: H 1-31 +0.0018 ± 0.0017, cold-positive +0.0004 ± 0.0011 (both n.s.). Verdict **NO**. "What kind of interaction does this
content attract" is an item-level prior. The target is a within-feed ranking that depends on the user (history similarity is the
biggest feature), and the frozen content embedding already carries the item prior. The very small paired SE (0.0004) shows that the
predictions barely changed.

## 4. B3: semantic-ID codes (2-level residual k-means)

**Offline** (`sid.py`, CPU, 24 threads, 97 s total): spherical k-means K1 = 256 on 2M L2-normalised train-table rows (k-means++ on 50K,
25 Lloyd iterations), then Euclidean K2 = 64 on the residuals `ê − C1_{c1}`. Residual energy 0.439 after level 1, 0.327 after level 2.
Every row of the train/val/test item tables is assigned → `nn_x/feat/sid/{split}_codes.npy` (int16 [n, 2]). Train: 11,698 of
16,384 cells used, median 3,162 items per cell. Val: 9,991 cells, median 108 items. Test: 10,358 cells, median 196 items.

**Model.** `z = W_z[E1(c1) (257×32); E2(c1·64+c2) (16385×32)]`, W_z zero-init, separate for candidates (before `c_ln`) and history
(before `h_ln`). In training: SID-dropout (E2 zeroed with p = 0.5, whole z with p = 0.2, per token); no weight decay on the tables.
Five dense extras: log1p #hist same c1, log1p #hist same (c1,c2), log1p #type1∧src1 hist same (c1,c2), 1/(1 + first matching k), and the
within-feed z of the second. ‖W_z,c‖ 1.9, ‖W_z,h‖ 4.2-4.3: the history side uses the codes more.

| seed | Δ val | Δ holdout | val@10k (base) | val@19670 (base) | selected step |
|---|---|---|---|---|---|
| 23 | +0.00075 ± 0.00062 | −0.00005 ± 0.00036 | 0.38070 (0.37904) | 0.38315 (0.38200) | 19000 |
| 24 | +0.00000 ± 0.00066 | +0.00011 ± 0.00039 | 0.37908 (0.37981) | 0.38172 (0.38109) | 16000 |
| pooled | +0.00038 ± 0.00049 | +0.00003 ± 0.00028 | | | |

**Memorisation check (mandatory): passed.** Val rises through epoch 2 on both seeds (above the baseline at the last step). Seed 24's
holdout-selected step is 16000, but val does not drop after it (0.38124 → 0.38172 at the end). Cold-positive feeds (17.4% of val; no
positive item seen in train) +0.00082 ± 0.00129, warm-positive +0.00027 ± 0.00058. That is the opposite of the N4 item-ID signature.
Verdict **NO** (holdout flat, val z 0.8). Coarse cells avoid the memorisation failure but carry little extra signal. The
fine-tune branch already has the collaborative features, so a gain after fine-tune is even less likely.

## 5. B4: multi-scale recency-decayed (EWMA) history features

For τ ∈ {4, 16, 64}, `w_k = e^{−k/τ}·valid_k` over the recency index k (newest first): `dmean_τ = Σ w S / Σ w`,
`dlse_τ = (1/5)·log(Σ w e^{5S} / Σ w)`, `dcat_τ = Σ w·1[same cat] / Σ w` (hmask weights), plus `dlse_16` and `dcat_16` over
type1∧src1 events, plus the within-feed z of `dlse_16`. Empty sums give −1.5 / 0. Computed on the GPU in `dense_features` from
the existing S and category-match matrices.

| seed | Δ val | Δ holdout |
|---|---|---|
| 23 | −0.00002 ± 0.00060 | −0.00014 ± 0.00036 |
| 24 | −0.00002 ± 0.00061 | −0.00027 ± 0.00035 |
| pooled | −0.00002 ± 0.00045 | −0.00021 ± 0.00026 |

Verdict **NO**. The history target attention with the per-head raw-cosine bias and the learned recency embedding already
implements soft, recency-weighted similarity pooling. The existing max/lse/top-3 windows (8/32/all) cover the rest. Group attn's FoX
(learned data-dependent decay, A2) was also NO, so recency smoothing in either form adds nothing here.

## 6. B5: TabM-mini BatchEnsemble (k = 4)

**Math** (TabM, arXiv 2410.24210). Member i uses first-layer adapters `x_i = r_i ⊙ x`: `r_i^(1) ∈ R^256` on `[ê_c, ê_c⊙u]` and
`r_i^(2) ∈ R^63` on the standardised X. Both are trainable, initialised to random ±1 (forked RNG). `c_lin`, `c_mlp`, the 2 cross
blocks and the 2 set blocks run on a k·B batch (rows b·k + i). The history encoder runs once, and `ctx`, `kmask` and the sim bias are
shared via `repeat_interleave`. Category, domain, propensity and SID terms are shared per candidate. Loss = mean over members of the
listwise CE; K9 aux = mean over members. Prediction = mean of member logits (fp32); the wide term is shared.
Cost: 3.8-4.3 min per 1000 steps vs 2.2-2.9 for the baseline on the shared GPU (~1.4-1.5x), about 30 GB per process, no new
parameters beyond 4·(256+63).

LIKELY on 2 seeds, so seed 25 was run (rule 1.6). The shared `base_s25` was run here: lock
`<repo>/nnx/base_s25.lock`, variant `feat_base_s25` = baseline flags, val 0.38156, holdout 0.30090.

| seed | Δ val | Δ holdout | member val NDCG (4) | mean pairwise member corr |
|---|---|---|---|---|
| 23 | +0.00024 ± 0.00066 | +0.00026 ± 0.00039 | 0.38257 0.38287 0.38208 0.38257 | 0.9994 |
| 24 | −0.00003 ± 0.00067 | +0.00053 ± 0.00039 | 0.38096 0.38111 0.38100 0.38093 | 0.9991 |
| 25 | +0.00006 ± 0.00066 | +0.00017 ± 0.00039 | 0.38168 0.38151 0.38107 0.38162 | 0.9992 |
| pooled (3 seeds) | +0.00009 ± 0.00042 | +0.00032 ± 0.00025 (z 1.3) | | |

Verdict by rule: **LIKELY** (both deltas > −0.0003, max z 1.3). In practice it is **neutral**: +0.0001 val on 3 seeds is a quarter of
the GPU non-determinism, at 1.4-1.5x cost. The mechanism explains why. The adapters move (mean |r| deviates from 1 by 0.36-0.38), but
the members stay correlated at 0.999, because four attention/FFN layers are shared after a single multiplicative input adapter.
So there is no implicit ensemble, and the 4-member mean equals a single member (ensemble vs mean member: −0.0001..+0.0002). Diversity
with the baseline is also nil (cross-seed base+TabM pair +0.00000 ± 0.00037; corr 0.987). Not recommended. A real BatchEnsemble would
need per-member adapters in every Linear (full TabM, much more cost) or separate heads. That fits our data regime poorly: seeds are
already saturated.

## 7. Ensemble diversity (val, equal weights, nothing fitted)

Per-feed z-scored score averages of two models. "cross-seed base+treat" = mean of (base_s23 + treat_s24) and (base_s24 + treat_s23),
compared with base_s23 + base_s24 (0.38189; baseline seed-to-seed correlation 0.9883).

| treat | corr(base, treat), same seed | base+treat pair | Δ vs base+base |
|---|---|---|---|
| PLE | **0.9834** | 0.38235 | **+0.00046 ± 0.00043** |
| SID | 0.9884 | 0.38222 | +0.00033 ± 0.00036 |
| TS | 0.9903 | 0.38211 | +0.00022 ± 0.00034 |
| PROP | 0.9916 | 0.38207 | +0.00018 ± 0.00030 |

PLE is the most decorrelated member despite its lower single-model val score. That matches design 6.3 (recipe diversity:
seeds alone are saturated).

## 8. Side checks

* **bf16 dense features at inference.** `predict()` computes the dense features inside autocast, so their einsums run in bf16, while
  training computes them in fp32. Re-scoring `nn/ckpt/v2_feed_k9` and `s2_v2_feed_k9` (read-only) with fp32 features: −0.00004 and
  −0.00026 val. No gain, so nothing to fix.
* GPU non-determinism (optim C0): base_s23 vs round-1 v2_feed_k9 = +0.0004 val. Every B1-B4 delta except B1's holdout is within
  about 1x of this.

## 9. Recommendations

* Do not add B1-B4 to the main recipe.
* Do not use PLE as a separate ensemble member. Its diversity disappears after the val fine-tune (section 10).
* Do not adopt TabM-mini: +0.0001 val for 1.45x compute, and its members are not diverse.
* B3's offline codes (`nn_x/feat/sid/`) are cheap and cold-safe. If cell-level collaborative signal is wanted later, a finer third
  level or code n-gram features in the fine-tune branch would be the next step. That is a separate experiment, not a val-tuned rerun.
* The feature path is saturated at this data scale. The headroom is in what the target attention reads: group attn's A4,
  TabR-style candidate-conditioned retrieval values, is WIN at +0.0019 val (their report), and A1 gated attention is LIKELY.
  It is also in the val-side stages.

## 10. Closeout: PLE stage 2 + baseline fine-tune

No feat arm is WIN, so there is no feat combination to carry forward. PLE is the only arm with a significant effect (holdout) and
a diversity signal, so I tested whether either survives the production pipeline. Stage 2 used the same recipe as optim's
`s2_base` (`$S2`: random-order train feeds, lr 2e-4, 2 epochs, keep last, seed 0). The fine-tune was the baseline `ftb` recipe:
frozen 48-feature `n2clean` list from `nnx/optim/feats`, lr 5e-5, 6 epochs, 5-fold GroupKFold, snapshot mean. Pairs:
`s2_base_s*` and `ftb_s*` of group optim.

| stage | Δ val ± SE | Δ holdout ± SE | per seed Δval | diversity: cross-seed base+PLE vs base+base |
|---|---|---|---|---|
| stage 1 | −0.00043 ± 0.00060 | +0.00078 ± 0.00037 | −0.00118 / +0.00032 | +0.00046 ± 0.00043 |
| + stage 2 | −0.00002 ± 0.00065 | **+0.00092 ± 0.00039** (z 2.4) | −0.00067 / +0.00062 | **+0.00071 ± 0.00046** (corr 0.981 vs 0.987) |
| + fine-tune (OOF) | −0.00010 ± 0.00064 | n/a | −0.00013 / −0.00007 | +0.00016 ± 0.00044 (corr 0.977 vs 0.984) |

The holdout gain grows slightly in stage 2. The val fine-tune (which re-learns the val-side feature semantics through the extra
branch and the backbone) removes both the single-model difference and most of the diversity gain. PLE is therefore **not** worth a
slot in the fine-tuned final ensemble on this evidence. It could still be a stage-2-level (train-only) blend member, but the
final pipeline fine-tunes every member.
Files: `preds_x/feat/nn_s2_feat_ple_s{23,24}_{trainonly_val.npy,holdout.npz}`, `preds_x/feat/nn_ftb_feat_ple_s{23,24}_{val_oof,test,test_folds}.npy`.
