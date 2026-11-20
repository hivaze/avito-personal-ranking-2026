# NNX group "attn": attention / token-mixing experiments (experimenter 1)

Written 2026-11-11 by the attn experimenter. Design: `reports/nnx_design.md` section 2.
Code: `<repo>/nn_x/attn/` (copy of `nn/` + harness H0). Every trick is a CLI flag that defaults to the exact baseline code path. A unit check confirms that with any flag on, every baseline parameter gets bit-identical init (`extra_init` forks the CPU RNG, with a per-call offset so e.g. the 5 gates are not copies of each other). Old checkpoints still load.
Predictions: `<repo>/preds_x/attn/` (`nn_<variant>_trainonly_val.npy`, `nn_<variant>_holdout.npz`; fine-tunes: `nn_ftb_<variant>_{val_oof,test,test_folds}.npy`).
Per-comparison log: `<repo>/nn_x/attn/results.tsv`. Run logs with per-eval mechanism diagnostics: `<repo>/nn_x/attn/logs/`.

## TL;DR

| # | experiment | flag | val Δ (SE) | holdout Δ (SE) | verdict |
|---|---|---|---|---|---|
| A1 | gated attention output (Qwen G1, elementwise sigmoid), all 5 attention modules | `--attn_gate elem` | +0.00042 (0.00046) | +0.00050 (0.00028) | LIKELY |
| A2 | FoX data-dependent forgetting bias on history target attention | `--fox` | -0.00019 (0.00042) | -0.00014 (0.00025) | NO |
| A3 | QK-norm + length-aware temperature (SSMax) | `--qk_norm --ssmax` | -0.00006 (0.00038) | -0.00010 (0.00023) | NO |
| **A4** | **TabR-style candidate-conditioned retrieval values (top-16 history items)** | `--retr_m 16` | **+0.00186 (0.00077)** | **+0.00326 (0.00046)** | **WIN** |
| A4 | ... after stage 2 (vs `s2_base`) | | +0.00161 (0.00079) | +0.00309 (0.00047) | WIN |
| A4 | ... after stage 2 + baseline val fine-tune (OOF, vs `ftb`) | | +0.00149 (0.00072) | - | WIN |
| A1+A4 | combination | `--attn_gate elem --retr_m 16` | **+0.00273 (0.00079)** | **+0.00389 (0.00047)** | **WIN** |
| A1+A4 | ... after stage 2 (vs `s2_base`) | | +0.00292 (0.00081) | +0.00377 (0.00049) | WIN |
| A1+A4 | ... after stage 2 + baseline val fine-tune (OOF, vs `ftb`) | | **+0.00195 (0.00074)** | - | **WIN** |
| A5 | fine-grained shared-expert MoE FFN, aux-loss-free balancing | `--moe 8 --moe_topk 2` | +0.00027 (0.00047) | +0.00053 (0.00028) | LIKELY (weak) |
| A1+A4+A5 | full combination (A5 increment over A1+A4: +0.00015 / +0.00041 train-only, -0.00034 after fine-tune) | `--attn_gate elem --retr_m 16 --moe 8` | +0.00287 (0.00080) | +0.00430 (0.00049) | WIN, but A5 adds nothing to A1+A4 |
| A4-meta | dissection: A4 without the `T(k_c - k_j)` value term | `--retr_m 16 --retr_vals meta` | +0.00016 (0.00051) | +0.00038 (0.00031) | ≈ baseline; vs full A4: -0.00169 val (z -2.3), -0.00288 holdout (z -6.4) |

All deltas are 2-seed (23, 24) paired per-feed differences against the shared baseline C0 (`<repo>/preds_x/optim/nn_base_s{23,24}`, same config and seeds, run by the optim group). Val = 14,423 exploration feeds; holdout = the fixed 30K random-like train feeds that are excluded from training and used for checkpoint selection. Decision rule = design section 1.6.

**Main result: A4 (retrieval values that depend on the candidate) is the one clear win, and it survives the whole pipeline.** With the cheap A1 gate added (the recommended config `--attn_gate elem --retr_m 16`): +0.0027 val / +0.0039 holdout train-only, +0.0029 / +0.0038 after stage 2, and **+0.0020 OOF after the baseline val fine-tune** (0.38681 / 0.38733 vs 0.38480 / 0.38544). A4 alone gives +0.0019 / +0.0033 -> +0.0016 / +0.0031 -> +0.0015. A single fine-tuned A1+A4 model (0.3871 OOF) beats the round-1 fine-tuned 16-member ensemble (0.3863), and 6 fine-tuned A4-type members (A4, A1+A4, A1+A4+A5; seeds 23/24) blend to **0.38813 OOF**. The cost is +139K parameters (A1+A4) and no measurable wall-clock for A4.
A dissection shows that ~90% of A4's gain comes from the TabR difference term `T(k_c - k_j)`: values made only of the matched items' metadata are at baseline level.

The LLM-style attention fixes (gating, QK-norm, SSMax, FoX) and MoE are at most small. Their mechanisms engage as intended (sparse gates, data-dependent decay horizons, M-dependent temperatures, category-specialised experts; see the diagnostics), but they redistribute information the model already has. What helped was giving the target attention new information: how a matched history item differs from the candidate.

Free check (GPU non-determinism): `base_s23` vs round-1 `nn_v2_feed_k9` (identical config and seed): +0.00040 val (SE 0.00032). Any single-seed effect below ~0.0005 is at the noise floor, which is why every arm uses 2 seeds.

---

## 1. Protocol and deviations from the design

* Directories follow the computed task: `<repo>/nn_x/attn` (not `nnx/attn`) and `PREDS = <repo>/preds_x/attn`. The `common.py` guard allows only `CUDA_VISIBLE_DEVICES=N`, with 16 threads per process and at most 2 GPU processes of mine at a time, each gated on free GPU memory (40 GB; 50 GB for A4/A5).
* Harness H0 is implemented as specified: `eval_subset(return_scores=True)`, per-feed holdout scores saved after the best checkpoint is restored, `NNX_FEATS` for fine-tune features, `extra_init`, `paired.py`. Additions: `diag.py` (mechanism diagnostics on a fixed 3,000-feed val subset at every eval, also usable on any checkpoint), `curve.py` (per-eval comparison with the baseline log, for the kill rule), `judge.py` (paired test + decision rule + `results.tsv` line), `slices.py` (paired deltas by H and M).
* Screening: the shared baseline exists only as the full 2-epoch recipe, so every arm runs that recipe and is screened at equal steps against `base_s{seed}.log` (the kill rule) instead of with separate 1-epoch runs. A 1-epoch screen would have needed its own 1-epoch baseline. No arm reached the kill threshold.
* A4: the LayerNorm sits on each value vector (`v_j = LN(E_type + E_src + E_rec + T(k_c - k_j))`), not on the aggregate `r` as in the design draft. `LN(r)` is scale-invariant, so it would cancel the sink: with sink mass p0, `r = (1 - p0) r'` and `LN(r) = LN(r')`. Candidates with a missing embedding read only the sink.
* Seed 25: I claimed `<repo>/nnx/base_s25.lock` when A1 was LIKELY, then released it unused. A4 decided the group on 2 seeds, and the A1+A4 combination measured A1's increment directly.
* A5: the specified architecture (8 routed SwiGLU experts of width 64 + 1 shared SwiGLU of width 128, in 4 blocks) adds +0.72M parameters net (+0.99M new, -0.26M dense FFNs), not the +0.4M stated in the design. The dense FFN is still constructed before being replaced, so the RNG stream of all other parameters is unchanged.

## 2. Experiments

### A1. Gated attention output (Qwen3-Next / Qwen3.5 G1): LIKELY
`Y = (softmax(QK^T/sqrt(d_h) + B_sim) V) ⊙ σ(X W_g + b_g)` per head, elementwise, before `W_O`, in all 5 attention modules (hist_enc, 2 cross, 2 set); X is the pre-LN query token. +82,560 parameters.

| seed | val base -> treat | Δ (SE) | holdout base -> treat | Δ (SE) |
|---|---|---|---|---|
| 23 | 0.38216 -> 0.38268 | +0.00052 (0.00063) | 0.30084 -> 0.30106 | +0.00022 (0.00038) |
| 24 | 0.38124 -> 0.38157 | +0.00033 (0.00061) | 0.29983 -> 0.30060 | +0.00077 (0.00037) |
| pooled | | +0.00042 (0.00046), z 0.9 | | +0.00050 (0.00028), z 1.8 |

* Both seeds are positive on both metrics, and the training loss was lower than the baseline at every eval (by 0.001-0.003). During training the holdout lead was steadier (+0.0009..0.0014 at steps 6k-13k), then it narrowed in the anneal.
* Mechanism (final checkpoint vs baseline checkpoints, `diag.py`): the gates become sparse (mean sigmoid 0.09-0.11 in cross block 0, 0.17-0.18 in cross block 1, 0.20 in hist_enc, 0.23-0.28 in set attention). They are lower for candidates of no-history feeds (0.055-0.066 vs 0.09-0.11 in cross block 0), i.e. "read nothing" is learned. The user-token attention mass falls (cross block 1: 0.055 / 0.087 -> 0.040 / 0.044), and so does the attention entropy (hist_enc 5.06 / 5.00 -> 4.60 / 4.49), i.e. the reads become sharper. However, the user token was never a large sink in the baseline (5-9% of mass), so the "attention sink" motivation is weak here.
* Slices: the gain sits in long-history feeds (H > 128: +0.00075, SE 0.00054). The no-history slices are too small to read (H = 0: n = 194).
* The pre-registered lr 2e-3 follow-up was conditional on A1 or A3 being a WIN, so it was not run.

### A2. FoX forgetting bias (data-dependent recency decay): NO
`logit_{c,j,h} = q_c·k_j/sqrt(d_h) + a_h S_cj + D_{j,h}` with `D_{j,h} = Σ_{l<j} log σ(w_h·x_l + b_h)` (exclusive cumulative sum over newer events, fp32), and D = 0 for the user token. Init w = 0 and `b = logit([0.999, 0.995, 0.98, 0.95])`. 1,032 parameters.

| seed | val Δ (SE) | holdout Δ (SE) |
|---|---|---|
| 23 | -0.00025 (0.00056) | -0.00020 (0.00034) |
| 24 | -0.00013 (0.00058) | -0.00009 (0.00034) |
| pooled | -0.00019 (0.00042) | -0.00014 (0.00025) |

* The mechanism worked and did not collapse to a static decay. By the end, 1-2 heads per block keep a finite horizon (mean number of events until D < -1: 20-65), and their D at j = 32 varies across feeds (std 0.3-0.7, against means of -0.5 to -1.8), i.e. topic switches do shorten the window. The other heads opened up to the full history (horizon = H, D ≈ 0). Which heads decay differs by seed.
* Interpretation: recency is already available through the learned absolute recency embedding plus content, and the sim-bias heads want exact retrieval over the whole history. Together with the neutral static bias (`sim_bias=2`, round 1), this closes the recency-bias direction for target attention.

### A3. QK-norm + SSMax: NO
Per-head RMSNorm on q and k (gain shared across heads), then `q <- q · s_h · log(n_valid_keys)`, with s_h initialised at 1/log(n_ref) (n_ref = 64 for set blocks, 192 for hist_enc and cross). 340 parameters.

| seed | val Δ (SE) | holdout Δ (SE) |
|---|---|---|
| 23 | -0.00006 (0.00052) | -0.00044 (0.00031) |
| 24 | -0.00005 (0.00052) | +0.00024 (0.00031) |
| pooled | -0.00006 (0.00038) | -0.00010 (0.00023) |

* SSMax does what it claims: the normalised set-attention entropy H/log M in set block 0 now falls with M (0.91 / 0.89 for M ≤ 32 down to 0.85 for M > 128), where the baseline's rises (0.86 -> 0.89/0.90). That changes nothing in NDCG.
* The learned temperatures shrink: s_h drops from 0.19 to 0.08-0.13 in hist_enc and cross (from 0.24 to 0.11-0.32 in set blocks), and the q/k gains shrink slightly (norm 5.66 -> 5.1-5.2). The model softens the learned QK logits and leaves sharp retrieval to the raw-cosine bias (a_h up to ~10).
* A post-hoc slice shows user-0 feeds at +0.0042 (SE 0.0018), but it is 1 of 9 slices and is not used for any decision.

### A4. TabR-style candidate-conditioned retrieval values: WIN
For each candidate c: the top-16 history items j by raw cosine S_cj (valid embeddings only). Keys `k = W_K ê` (128 -> 64, no bias, shared by candidate and history). Logits `ℓ_cj = -||k_c - k_j||² + a·S_cj + β[type_j]`, plus a learned sink logit with a zero value; softmax in fp32. Values `v_cj = LN(E_type[t_j] + E_src[s_j] + E_rec[⌊log2(1+j)⌋] + T(k_c - k_j))`, with `T = Linear(128->d) ∘ Dropout(0.1) ∘ ReLU ∘ Linear(64->128)`. Output `c <- c + Z(Σ_j p_cj v_cj)`, Z zero-init (exact no-op at init), added before the cross/set blocks. +56,333 parameters; memory [B, M, 16, 128] in bf16.

| stage | seed 23 val Δ | seed 24 val Δ | pooled val Δ (SE) | pooled holdout Δ (SE) |
|---|---|---|---|---|
| train-only (vs `base`) | +0.00124 (0.38216 -> 0.38340) | +0.00247 (0.38124 -> 0.38371) | **+0.00186 (0.00077), z 2.4** | **+0.00326 (0.00046), z 7.1** |
| + stage 2 (vs `s2_base`) | +0.00098 (0.38260 -> 0.38358) | +0.00225 (0.38193 -> 0.38418) | +0.00161 (0.00079), z 2.1 | +0.00309 (0.00047), z 6.6 |
| + stage 2 + baseline fine-tune (OOF, vs `ftb`) | +0.00182 (0.38480 -> 0.38661) | +0.00117 (0.38544 -> 0.38661) | +0.00149 (0.00072), z 2.1 | - |

* The lead appeared early and held: +0.0013..0.0020 val and +0.001..0.0026 holdout at step 3000, then +0.0015..0.0031 val and +0.0028..0.0041 holdout over steps 10k-18k on both seeds. The training loss was lower throughout (by about 0.005).
* Mechanism: ||Z(r)||/||c|| = 0.60-0.62 (the branch is used heavily). The sink mass is 12-13% for candidates whose best history cosine is < 0.5 and 0.15% when it is > 0.9: the sink acts as "no relevant history item". The learned cosine weight a rose from 4.0 to 5.9, and the sink logit from 0 to 0.83-0.90.
* Slices: the gain is largest in long-history feeds (H > 128: +0.0027, SE 0.0009) and present for every feed size (M ≤ 32 +0.0016, 33-96 +0.0024, > 96 +0.0016).
* Why it helps where the others did not: in the existing target attention, the values `W_V h_j` do not depend on the candidate, and softmax is shift-invariant, so the absolute match level and the kind of match are lost. A4 hands each candidate "what the best-matching history items were (event type, source, recency) and how they differ from me (`T(k_c - k_j)`)". This is the TabR finding (the candidate-dependent value term is what makes retrieval work) carried over to per-user history retrieval.
* **Dissection (informational, does not change the selected config)**: `--retr_vals meta` keeps the retrieval, the logits and the sink, but drops `T(k_c - k_j)` from the values (values = type + source + recency embeddings of the matched items only). It lands at baseline level (+0.00016 val, +0.00038 holdout vs base) and is clearly worse than full A4 (-0.00169 val, z -2.3; -0.00288 holdout, z -6.4; both seeds). **About 90% of A4's gain comes from the candidate-minus-history difference vector**, exactly the ingredient TabR identifies (its ablation without the `T(k - k_i)` value falls back to MLP level). Knowing what kind of event the best-matching history items were adds nothing beyond what the existing attention and dense features already carry; knowing how the candidate differs from them (near duplicate vs related item vs different item of the same topic) is the new information.
* Fine-tuned A4 predictions (val OOF and test, 5-fold mean, snapshot epochs 1..6) are ready for blending: `<repo>/preds_x/attn/nn_ftb_attn_retr_s{23,24}_{val_oof,test}.npy`.

### Closeout: A1 + A4 combination: WIN (best config of the group)
`--attn_gate elem --retr_m 16` (+139K parameters, 0.999M total).

| comparison | seed 23 val Δ | seed 24 val Δ | pooled val Δ (SE) | pooled holdout Δ (SE) | verdict |
|---|---|---|---|---|---|
| A1+A4 vs base (train-only) | +0.00250 (0.38216 -> 0.38466) | +0.00295 (0.38124 -> 0.38419) | **+0.00273 (0.00079), z 3.5** | **+0.00389 (0.00047), z 8.2** | WIN |
| A1+A4 vs A4 alone (= A1 increment) | +0.00126 | +0.00048 | +0.00087 (0.00047), z 1.8 | +0.00063 (0.00029), z 2.2 | WIN |
| A1+A4 + stage 2 (vs `s2_base`) | +0.00299 (0.38260 -> 0.38559) | +0.00284 (0.38193 -> 0.38477) | **+0.00292 (0.00081), z 3.6** | **+0.00377 (0.00049), z 7.7** | WIN |
| A1+A4 + stage 2 + baseline fine-tune (OOF, vs `ftb`) | +0.00201 (0.38480 -> 0.38681) | +0.00189 (0.38544 -> 0.38733) | **+0.00195 (0.00074), z 2.6** | - | WIN |
| A1+A4 vs A4 after stage 2 + fine-tune (OOF) | +0.00019 | +0.00072 | +0.00046 (0.00051), z 0.9 | - | NO (small positive) |

Ensembles of the fine-tuned models (per-feed z-score mean of val OOF, `blend.feed_z`):

| members | OOF |
|---|---|
| baseline `ftb_s23` + `ftb_s24` | 0.38608 |
| A4 s23 + s24 | 0.38746 |
| A1+A4 s23 + s24 | 0.38723 |
| A4 s23 + s24 + A1+A4 s23 + s24 (4 members) | **0.38765** |

Within-feed z-score correlation: A1+A4 vs baseline (same seed) 0.9625, vs A4 0.981; seed pairs of the same config 0.980-0.984. The A4-type models are therefore further from the baseline than two seeds of either, which adds diversity for the final blend.

* In the train stage, the gate's increment on top of A4 (+0.0009 val / +0.0006 holdout) is at least as large as on the baseline (+0.0004 / +0.0005), and across the 4 train-stage paired comparisons (2 seeds x 2 contexts) A1 is positive on both metrics every time. After the val fine-tune the increment shrinks to +0.0005 OOF (z 0.9), and in 2-member ensembles it is gone (A4 pair 0.38746 vs A1+A4 pair 0.38723). Net: A1 is a real but small train-stage effect, and most of it is absorbed by fine-tuning and averaging. It is cheap and harmless, so keep it in the config, but do not count on it at the leaderboard level.
* Mechanisms coexist: in the combination the cross-attention gates stay sparse (0.07-0.13), and the retrieval branch is slightly smaller (||Z(r)||/||c|| 0.52-0.57 vs 0.60-0.62 for A4 alone) with the same sink behaviour (14% sink mass when the best match has cosine < 0.5, 0.2% when > 0.9).
* Cost: 61.5 min vs 53-54 min for the other arms in the same period. The GPU was more heavily shared in that window, so this is not a clean timing; the expected overhead is about 1.05x (gate) x 1.0x (retrieval).


### A5. Fine-grained shared-expert MoE FFN (DeepSeekMoE / DeepSeek-V3 / Ling 2.0 / MiniMax-M2 routing): LIKELY (weak)
In the 4 candidate blocks (2 cross, 2 set; hist_enc keeps its dense FFN): `y = SwiGLU_shared(x; 128) + Σ_{i ∈ Top2(s + b)} (s_i / Σ_top s) SwiGLU_i(x; 64)` over 8 routed experts, with `s = σ(W_r x)` in fp32. The bias b is a no-gradient buffer used only for selection and updated after every optimizer step by `b_i += 1e-3·sign(mean_load - load_i)` over valid candidate tokens (aux-loss-free balancing). Dense dispatch (all 8 experts in one matmul, multiplied by the sparse gate). Router weights have no weight decay. The activated FFN width is 128 + 2·64 = 256 = the baseline's 2d. Total parameters 1.58M (+0.72M net).

| seed | val Δ (SE) | holdout Δ (SE) |
|---|---|---|
| 23 | +0.00016 (0.00063) | +0.00028 (0.00037) |
| 24 | +0.00038 (0.00064) | +0.00078 (0.00037) |
| pooled | +0.00027 (0.00047), z 0.6 | +0.00053 (0.00028), z 1.9 |

* Routing: load entropy / log 8 = 0.93-0.99 in cross 0, cross 1 and set 0 (max/min load 1.8-6), but **set block 1 collapsed** (0.67, max/min load 80-94) and the sign-based bias update (γ = 1e-3) did not recover it within 19,670 steps. Expert choice carries category information (mutual information with the 54 categories 0.33-0.50 nats in cross block 0 = 16-24% of the routing entropy, 0.05-0.07 in set block 1), so the hoped-for mechanism (conditional computation per vertical) is partly there.
* Value: +0.0003 val / +0.0005 holdout for +84% parameters and about 1.15x FFN compute, with the gain in short feeds (M ≤ 32: +0.0014, SE 0.0011) and long histories. That is in line with the design's prior that capacity is not binding (d = 256, 3 blocks and 8 heads were neutral). It is LIKELY by the formal rule but the weakest positive in the group; it goes into the full combination (below) as the pre-registered closeout, not as a stand-alone recommendation.

### Closeout: full combination A1 + A4 + A5
`--attn_gate elem --retr_m 16 --moe 8 --moe_topk 2` (1.72M parameters).

| comparison | seed 23 val Δ | seed 24 val Δ | pooled val Δ (SE) | pooled holdout Δ (SE) | verdict |
|---|---|---|---|---|---|
| A1+A4+A5 vs base (train-only) | +0.00242 (0.38216 -> 0.38458) | +0.00333 (0.38124 -> 0.38457) | +0.00287 (0.00080), z 3.6 | +0.00430 (0.00049), z 8.8 | WIN |
| A1+A4+A5 vs A1+A4 (= A5 increment) | -0.00008 | +0.00037 | +0.00015 (0.00048), z 0.3 | +0.00041 (0.00027), z 1.5 | LIKELY (formal), negligible on val |
| A1+A4+A5 + stage 2 (vs `s2_base`) | +0.00213 | +0.00281 | +0.00247 (0.00082), z 3.0 | +0.00361 (0.00050), z 7.2 | WIN |
| A1+A4+A5 + stage 2 + fine-tune (OOF, vs `ftb`) | +0.00175 (0.38480 -> 0.38655) | +0.00148 (0.38544 -> 0.38692) | +0.00161 (0.00078), z 2.1 | - | WIN |
| A5 increment over A1+A4 after stage 2 | -0.00086 | -0.00003 | -0.00045 (0.00051) | -0.00016 (0.00030) | NO |
| A5 increment over A1+A4 after stage 2 + fine-tune (OOF) | -0.00026 | -0.00041 | -0.00034 (0.00052) | - | NO |

Ensembles of fine-tuned models (per-feed z-score mean of val OOF; equal weights, nothing fitted on val):

| members | OOF |
|---|---|
| A1+A4+A5 s23 + s24 | 0.38775 |
| A4 s23 + s24 + A1+A4 s23 + s24 (4 members) | 0.38765 |
| A4 + A1+A4 + A1+A4+A5, seeds 23 + 24 (6 members) | **0.38813** |

* As a single-model change, MoE adds about +0.0004 holdout and nothing measurable on val on top of A1+A4 in the train stage, and turns slightly negative after stage 2 (-0.0005 val) and after the fine-tune (-0.0003 OOF), at +0.72M parameters and about 1.15x FFN compute. It is not part of the recommended config.
* As an ensemble member it is useful: the MoE pair alone ensembles to 0.38775 (vs 0.38723 for the A1+A4 pair, despite lower single-model OOF), and adding it to the 4 A4-type members lifts the blend from 0.38765 to 0.38813. Architectural diversity helps the blend even where it does not help a single model.

## 3. Recommendations for the integration step (design section 6)

1. **Adopt A4 (`--retr_m 16`) in every NN member, preferably with A1 (`--attn_gate elem --retr_m 16`).** It is the only item of this group with a robust gain at every stage: +0.0027 val / +0.0039 holdout train-only (A1+A4), +0.0029 / +0.0038 after stage 2, and +0.0020 OOF after the baseline fine-tune. A1 is optional: small, mostly absorbed by fine-tuning, and slightly behind A4 alone in a 2-member blend (0.38723 vs 0.38746). It is non-negative in every paired single-model comparison and cheap.
2. **Re-test with the other groups' winners, especially Muon (optim C2: +0.0020 val / +0.0030 holdout, WIN) and `--xt cv`.** These act on different things: the optimizer for Muon, collaborative inputs for `xt`, the history-retrieval values for A4. Additivity is plausible but not guaranteed. Muon orthogonalises 2-D weights, including the new `wk`, `t1`, `t2` and `Z` matrices of the retrieval branch; Z is zero-init, which Muon handles, since its first update is the orthogonalised gradient. Suggested order: A1+A4+Muon (2 seeds) -> + `--xt cv` -> stage 2 -> fine-tune.
3. **Ensemble**: the A4-type fine-tuned models are less correlated with the baseline (within-feed z corr 0.962) than seeds are with each other (0.98), and 4 A4-type members already reach 0.38765 OOF. Ready-to-blend files (val OOF + test, 5-fold mean, snapshot epochs 1..6):
   * `<repo>/preds_x/attn/nn_ftb_attn_retr_s{23,24}_{val_oof,test}.npy` (A4; OOF 0.38661 / 0.38661)
   * `<repo>/preds_x/attn/nn_ftb_attn_gate_retr_s{23,24}_{val_oof,test}.npy` (A1+A4; OOF 0.38681 / 0.38733)
   * `<repo>/preds_x/attn/nn_ftb_attn_gate_retr_moe_s{23,24}_{val_oof,test}.npy` (A1+A4+A5; OOF 0.38655 / 0.38692; best used as an ensemble member: 6-member A4-type blend 0.38813 OOF)
4. **Do not pursue further**: FoX / learned recency decay (A2), and QK-norm + SSMax (A3). Both mechanisms work as intended and change nothing. Also MoE (A5) as a quality lever: +0.0003 / +0.0005 alone, ≈0 on top of A1+A4, for +84% parameters. The pre-registered lr 2e-3 arm was not triggered (A1 and A3 were not WINs).
5. **Follow-ups for A4 (not run; each a 2-seed arm)**: (a) more retrieved items (m = 32 or 64; the H > 128 slice carries the gain); (b) retrieval keys that include the category and event type (`W_K [ê; E_cat]`); (c) feeding the retrieval output into the fp32 head as well (today it enters only the candidate token); (d) the same candidate-conditioned value trick inside the existing cross-attention (values `W_V h_j + T(W_K ê_c - W_K ê_j)`) instead of a separate top-16 branch. The dissection says the difference vector is the active ingredient, so (b) and (d) are the most promising.

## 4. Reproduction
```bash
cd <repo>/nn_x/attn
./run.sh attn_gate 23 --attn_gate elem          # A1 (run.sh = the design's BASE config, CUDA_VISIBLE_DEVICES=N; same for seed 24)
./run.sh attn_fox 23 --fox                      # A2
./run.sh attn_qkss 23 --qk_norm --ssmax         # A3
./run.sh attn_retr 23 --retr_m 16               # A4
./run.sh attn_gate_retr 23 --attn_gate elem --retr_m 16   # A1+A4
./run.sh attn_moe 23 --moe 8 --moe_topk 2       # A5
./run.sh attn_gate_retr_moe 23 --attn_gate elem --retr_m 16 --moe 8 --moe_topk 2   # A1+A4+A5
./run.sh attn_retrmeta 23 --retr_m 16 --retr_vals meta    # A4 dissection (no T(k_c - k_j) value term)
./s2ft.sh attn_retr_s23                         # stage 2 (C0 recipe, seed 0) + baseline fine-tune (frozen n2clean); also run for attn_gate_retr_s*, attn_gate_retr_moe_s*
../../.venv/bin/python judge.py --exp A4 --arm retr_m16 --treat attn_retr --no_write   # paired test vs base_s{23,24}
../../.venv/bin/python diag.py --ckpt ckpt/attn_retr_s23.pt                            # mechanism diagnostics
```
