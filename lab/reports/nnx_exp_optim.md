# NNX group "optim": optimisation and generalisation experiments (C0-C5)

Experimenter 3, 2026-11-11. Code: `<repo>/nnx/optim/` (also reachable as `<repo>/nn_x/optim`).
Predictions: `<repo>/preds_x/optim/` (`nnx/optim/preds` is a symlink to it). Per-comparison table:
`<repo>/nnx/optim/results.tsv`. Design: `reports/nnx_design.md` section 4.
All numbers are val full NDCG with the exact `src/metrics.py` metric (binary gain, pessimistic ties), plus NDCG on the
fixed 30K random-like train holdout ("hold"). Deltas are paired per feed and pooled over seeds 23 and 24.
SE is the per-feed paired SE, so it ignores training noise (seed spread 0.001-0.002).

## Summary

| # | experiment | flags | Δval (SE) | Δhold (SE) | verdict |
|---|---|---|---|---|---|
| C1 | stage-1 weight EMA, β chosen on holdout (0.999) | `--ema 0.999,0.9995,0.9998 --ema_start 0.5` | +0.00003 (0.00016) | -0.00004 (0.00009) | NO (neutral) |
| C2 | Muon (RMS-matched) for hidden matrices, lr 1e-3 | `--muon` | **+0.00201** (0.00060) | **+0.00295** (0.00037) | **WIN** |
| C2 follow-up | Muon lr 2e-3 (vs Muon lr 1e-3) | `--muon --lr 2e-3` | +0.00079 (0.00050) | +0.00073 (0.00029) | WIN (vs base +0.00280 / +0.00368) |
| C3a | fine-tune: cosine + weight EMA 0.99, 4 epochs (OOF) | `finetune.py --epochs 4 --decay cos --ema 0.99 --ema_start 0.5` | +0.00025 (0.00035) | - | NO on AdamW models (WIN on Muon models: +0.00112) |
| C3b | fine-tune: LP-FT 1 epoch + C3a (OOF) | `... --lpft_epochs 1 --lpft_lr 5e-4 --extra_lr_mult 3` | -0.00014 (0.00038) | - | NO |
| C4 | WSD 1-sqrt cooldown + exploration mix + aux anneal, vs baseline **with** stage 2 | `--sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1` | **+0.00114** (0.00049) | **+0.00074** (0.00028) | **WIN** (and replaces stage 2) |
| C5 | born-again listwise KD from s2_v2_feed_k9 | `--distill <teacher> --distill_w 0.5 --distill_tau 1` | +0.00064 (0.00041) | +0.00098 (0.00024) | WIN (holdout-driven) |
| combo | **Muon + WSD, no stage 2** (vs base stage 1 + stage 2) | `--muon --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1` | **+0.00362** (0.00075) | **+0.00338** (0.00044) | **WIN, best** |
| combo pipeline | Muon + WSD -> C3a fine-tune vs base -> s2 -> production fine-tune (OOF) | as above + `finetune.py ... --epochs 4 --decay cos --ema 0.99 --ema_start 0.5` | **+0.00318** (0.00071) | - | **WIN, best** |
| closeout | all winners (Muon lr 2e-3 + WSD + KD) vs Muon + WSD | `--muon --lr 2e-3 --sched wsd ... --distill ...` | -0.00079 (0.00054) | +0.00075 (0.00031) | NO (lr 2e-3 under WSD costs val; KD neutral) |

Deltas: stage-1 rows vs `base_s{23,24}` unless stated; C4 and combo vs `s2_base_s{23,24}` (stage 1 + stage 2); fine-tune
rows are val OOF vs `ftb_s{23,24}`. Every WIN above is positive on both seeds.

**Best result: Muon + WSD.** Train-only single models reach val 0.38605 / 0.38572 (previous best train-only single:
0.3827). After the val fine-tune: OOF 0.38833 / 0.38827, and the 2-seed blend reaches 0.38859. The baseline pipeline
scores 0.38480 / 0.38544 (blend 0.38608), and the production 8-model NN ensemble `nn_final2` scores 0.38634.
The recipe is one stage shorter than before (no stage 2) and uses the same number of steps; Muon's
Newton-Schulz iterations cost about 1.15x wall time per step at this model size.

What worked and why:
* **Muon** was the biggest single gain. Stage 1 is optimisation-limited: val was still rising at the last step, and
  width, depth and heads were all neutral. Muon's orthogonalised updates use the 128-d content-embedding projection
  much more evenly: the effective rank of `c_lin` goes from 30 to 79, and its top singular value from 29 to 7.
* **WSD with exploration data in the cooldown**, the MiniCPM / Hägele recipe, is better than a separate re-warmed
  stage 2 on the same data. Once it is in place, stage 2 hurts.
* The two gains add up almost fully: +0.0020 (Muon) + 0.0011 (WSD vs stage 2) ≈ +0.0036 measured.
* **Fine-tune recipe matters only for Muon models.** The cosine + EMA recipe recovers the fine-tune gain that the
  6-epoch constant-LR snapshot recipe loses on Muon checkpoints, because those over-fit faster.
* **What did not help:** EMA in stage 1 (the cosine already anneals), LP-FT, lr 2e-3 under WSD, and KD on top of
  Muon + WSD. KD and lr 2e-3 both raise the train-distribution holdout more than val.


---

## 0. Setup, shared baseline (C0) and noise floor

* Copy of `nn/` with the H0 harness patch: GPU-5 guard, own `ckpt/` and `logs/`, `PREDS = preds_x/optim`,
  `eval_subset(return_scores=True)`, per-feed holdout scores saved as `nn_<v>_holdout.npz` (so the holdout can be
  paired too), `NNX_FEATS` for the fine-tune features, `extra_init()` (a forked RNG for treatment-only modules), and
  `paired.py`. Every experiment is a CLI flag that defaults to the baseline code path. Old checkpoints load unchanged.
* Shared baseline = `v2_feed_k9` config (BASE flags of the design doc), seeds 23/24, 2 epochs = 19,670 steps:

| run | val | hold | best step | minutes (shared GPU) |
|---|---|---|---|---|
| base_s23 | 0.38216 | 0.30084 | 19000 | 53 |
| base_s24 | 0.38124 | 0.29983 | 19000 | 53 |
| s2_base_s23 (stage 2, last step) | 0.38260 | 0.30111 | 1288 | 3 |
| s2_base_s24 | 0.38193 | 0.30005 | 1288 | 3 |
| ftb_s23 (val fine-tune, production recipe, OOF) | 0.38480 | - | - | 8 |
| ftb_s24 | 0.38544 | - | - | 8 |

* Frozen fine-tune features: the `n2clean` preset resolves to 49 of 56 files on disk (missing: `seq_cv_invd20_z,
  seq_cv_w5, seq_cv_fwd5, seq_cv_t1, seq_cv_seqs, seq_dcos_hl5, seq_nd90_cnt`). The list is in
  `nnx/optim/n2clean_frozen.txt` and the files are copied to `nnx/optim/feats/{val,test}/` (no inf; 0.02-1.4% NaN,
  i.e. the no-history feeds). All fine-tunes use `NNX_FEATS=<repo>/nnx/optim/feats`.
* Screening: every arm ran the full 2-epoch recipe (the shared baseline is 2 epochs, and C2/C4 change the schedule,
  so 1-epoch screens would not have been comparable), with the per-step kill rule of design section 1.6 checked via
  `curves.py`. No arm reached the kill threshold. Jobs ran through a 2-worker queue (`sh/worker.sh`), never more than
  2 GPU processes on GPU N.
* **GPU non-determinism**: `base_s23` against the round-1 `preds/nn_v2_feed_k9` (same config, same seed):
  val 0.38176 -> 0.38216, d = +0.00040 (SE 0.00032). Identical runs therefore differ by about 0.0004 on val, the
  same size as the per-feed SE of a paired comparison.


## 1. C1: weight EMA in stage 1 (measured inside the baseline runs)

**Math.** After every optimizer step from 50% of training, `θ̄ ← β θ̄ + (1-β) θ`, one fp32 copy per β
(time constants 1/(1-β) = 1000, 2000, 5000 steps), updated with `torch._foreach_lerp_`. By the WSM identity a weight
average is a synthetic decay of the last updates, `θ̂ = θ_n - Σ_i w_i g_{n+i-1}`, so it should add little on top of a
cosine schedule that already decays to 3%.

**Result: NO (neutral; β = 0.9998 hurts).** Each EMA kept its own holdout-selected checkpoint.

| β | Δval (SE) | Δhold (SE) |
|---|---|---|
| 0.999 (holdout choice) | +0.00003 (0.00016) | -0.00004 (0.00009) |
| 0.9995 | +0.00001 (0.00018) | -0.00009 (0.00010) |
| 0.9998 | -0.00057 (0.00026) | -0.00051 (0.00015) |

Diagnostics: in the middle of training the EMA is ahead on the holdout (β = 0.999: +0.0004..+0.0010 at 10-13k
steps), and that lead vanishes as the cosine anneals. This matches the WSM / Ajroldi 2025 view (averaging and LR decay
are the same mechanism). The EMA does lower eval-to-eval noise (mean |Δval| between the last 5 evals: 0.00035 -> 0.00027
on seed 23, 0.00027 -> 0.00015 on seed 24), but not enough to justify the extra 3 evals per eval point. Not used in the
closeout.

## 2. C2: Muon for the hidden matrices (RMS-matched), AdamW for the rest

**Math.** For every hidden 2-D Linear weight W ∈ R^{A×B} (37 matrices, 0.80M of 0.86M parameters: q, k, v, o, FFNs,
c_lin, h_lin, u_lin, c_mlp, head.1, aux_head.1): momentum `B_t = μB_{t-1} + (1-μ)G_t`, Nesterov `M_t = (1-μ)G_t + μB_t`,
5 quintic Newton-Schulz steps `O_t = NS5(M_t) ≈ U V^T` (steepest descent under the spectral norm), and
`W ← (1 - ηλ)W - η · 0.2·sqrt(max(A,B)) · O_t`. The 0.2·sqrt(max(A,B)) factor (`adjust_lr_fn="match_rms_adamw"`)
gives the update the RMS of a typical AdamW update, so lr 1e-3 and wd 0.02 carry over unchanged. Embeddings, LayerNorms,
biases, the scalar sim-bias scales and the 1-output layers (head.3, aux_head.3, wide) stay on AdamW with the baseline's
decay/no-decay grouping. The fused `kv` projection is split into `k` and `v` (initialised as exact slices of the
fused weight, so the init is identical to the baseline; verified: identical forward outputs). Built-in
`torch.optim.Muon` (torch 2.11). Cost: 43-45 min on the shared GPU (the baseline's 53 min include 3 extra EMA
evaluations per eval point; the WSD and KD arms without EMA took 37-39 min, so Muon's NS5 overhead is about 1.15x at
this small model size, where 37 tiny 5-step matmul chains per step are launch-bound).

**Result: WIN, the largest single effect in the group.**

| comparison | Δval (SE) | Δhold (SE) | per seed Δval |
|---|---|---|---|
| stage 1: muon vs base | **+0.00201** (0.00060), z 3.4 | **+0.00295** (0.00037), z 7.9 | +0.00136, +0.00266 |
| + stage 2 (AdamW): s2_muon vs s2_base | +0.00204 (0.00068) | +0.00235 (0.00040) | +0.00223, +0.00186 |
| + fine-tune, production recipe (ftb, AdamW) | +0.00097 (0.00069), LIKELY | - | +0.00210, -0.00016 |
| + fine-tune, cos+EMA recipe (fta) vs base ftb | **+0.00210** (0.00066), z 3.2 | - | +0.00289, +0.00131 |

* The lead appears by step 2000 (+0.0018 val / +0.0023 hold) and stays between +0.0012 and +0.0036 for the whole run.
  Stage 1 was optimisation-limited, and Muon moves faster within the same 2 epochs.
* Mechanism (weight spectra of the final seed-23 checkpoints): Muon spreads the update over many directions.
  `c_lin` (the content-embedding projection): top singular value 29.0 -> 7.3, effective rank (Σs)²/Σs² 30 -> 79 of 128.
  `head.1`: s_max 4.4 -> 1.8, effective rank 64 -> 76. Cross-attention q / FFN show the same pattern, smaller.
  AdamW lets a few directions of the 128-d content embedding dominate; Muon uses more of them.
* Ensemble diversity: per-feed correlation with the baseline 0.976-0.978 (two AdamW seeds: 0.988; two Muon seeds: 0.984).
  2-seed blends: Muon 0.38464 vs AdamW 0.38189.
* **Fine-tune caveat (Moonlight optimizer mismatch), confirmed and fixed.** With the 6-epoch constant-LR AdamW fine-tune
  the Muon model over-fits faster than the AdamW model (OOF by epoch, seed 23: 0.3848, 0.3873, **0.3876**, 0.3861,
  0.3855, 0.3841, 0.3821; the baseline peaks lower and decays slower). The epoch-snapshot mean then loses half of the gain
  and seed 24 ends at -0.00016. Two pre-specified fixes were compared on val OOF (3 fine-tune choices in total, the
  protocol's limit):
  * Muon also in the fine-tune (`finetune.py --muon`, same lr; Moonlight's advice): +0.00029 (0.00039) over ftb. NO.
  * The C3 arm-a recipe (cosine decay to 0 over 4 epochs + weight EMA 0.99, predict from the final EMA weights):
    +0.00112 (0.00038) over ftb on the same Muon checkpoints, both seeds positive (+0.00078, +0.00146). This is the
    fine-tune recipe to use for Muon models. On the AdamW baseline the same recipe is neutral (C3 below), so its gain is
    specific to the faster over-fitting of the Muon solution.
* Pre-registered follow-up `--lr 2e-3` (Muon only): **WIN** against Muon at lr 1e-3: val +0.00079 (0.00050), z 1.6;
  hold +0.00073 (0.00029), z 2.5; per seed +0.00089 / +0.00069 (val 0.38441 / 0.38459, hold 0.30432 / 0.30372).
  Against the baseline: val **+0.00280** (0.00067), z 4.2; hold **+0.00368** (0.00041), z 9.1. As reported for
  Qwen3.8-Next, Muon moves the optimal LR up; lr 2e-3 with AdamW was not tested here (the stability argument applies to
  Muon's bounded spectral-norm steps, not to AdamW).

## 3. C3: fine-tune recipe (on the AdamW baseline)

**Math.** Arm a: `lr(s) = lr · mult · min(1, (s+1)/30) · ½(1 + cos(π s / S))` over 4 epochs, EMA β = 0.99 from epoch
0.5, OOF and test predicted once from the final EMA weights. Arm b (LP-FT): first 1 epoch with only `extra.*`,
`head.*`, `wide.*` trainable at lr 5e-4 (the backbone frozen, so the new zero-init 49-feature branch is fitted before
the backbone moves), then arm a on all parameters with `--extra_lr_mult 3`.

**Result: both NO on the baseline.**

| arm | OOF s23 | OOF s24 | Δ vs ftb (SE) |
|---|---|---|---|
| ftb (production: const lr, 6 epochs, snapshot mean) | 0.38480 | 0.38544 | - |
| a: cos + EMA, 4 epochs | 0.38553 | 0.38521 | +0.00025 (0.00035) |
| b: LP-FT + a | 0.38533 | 0.38463 | -0.00014 (0.00038) |

The cosine decay removes the late-epoch decline (baseline OOF by epoch 0.3826, 0.3853, 0.3859, 0.3850, 0.3844, 0.3831,
0.3821 on seed 23; arm a 0.3826, 0.3852, 0.3859, 0.3857, 0.3854), but on the AdamW model the snapshot mean already
averages that decline away. LP-FT: the linear-probe epoch alone gets +0.0018 over the train-only model; the full arm
ends no better. The fine-tune is not the bottleneck for AdamW models. Arm a is, however, clearly better for the
over-fitting-prone Muon models (section 2).

## 4. C4: LLM decay-phase recipe (WSD 1-sqrt cooldown + exploration-feed mixing + aux annealing)

**Math.** Warmup 500, constant η_max = 1e-3 until T_s = 0.8T = 15,736, then
`η(t) = η_min + (η_max - η_min)(1 - sqrt((t - T_s)/(T - T_s)))`, η_min = 0.03 η_max (Hägele et al. 2024). During the
cooldown each step draws, with probability 0.3 (own RNG, seed+1), a batch from the exploration pool (train feeds with
p_random > 0.5, M ≥ 50, holdout excluded: 134,118 feeds = 644 batches; 1,195 cooldown batches = 1.9 passes, about the
size of stage 2). The replaced production batches are skipped, so compute is unchanged (19,670 steps). K9 weight
`w_aux(t) = 0.1 + 0.4 (η(t) - η_min)/(η_max - η_min)`, i.e. 0.5 -> 0.1 over the cooldown (DeepSeek-V3 / MiniMax-M2
MTP-weight annealing; here the auxiliary target is the biased production order).

**Result: WIN against the baseline with its separate stage 2, in one stage and less compute.**

| comparison | Δval (SE) | Δhold (SE) | per seed Δval |
|---|---|---|---|
| primary: wsd (stage 1 only) vs s2_base (stage 1 + stage 2) | **+0.00114** (0.00049), z 2.3 | **+0.00074** (0.00028), z 2.6 | +0.00148, +0.00080 |
| secondary: wsd vs base (stage 1) | +0.00170 (0.00052) | +0.00098 (0.00033) | +0.00192, +0.00149 |
| stage 2 on top of wsd vs wsd | -0.00065 (0.00040) | -0.00019 (0.00022) | -0.00124, -0.00007 |

* Trajectory: WSD trails the baseline by 0.0005-0.002 during the constant phase (as expected) and overtakes it within
  ~1,500 cooldown steps; the best checkpoint is the last step for both seeds.
* Exploration data arriving during the anneal is worth more than the same data in a separate re-warmed stage 2
  (+0.0011 val), and a stage 2 after the WSD cooldown now hurts (it re-warms the LR to 2e-4 and undoes part of the
  anneal). WSD therefore replaces stage 2: one stage, 1,288 fewer steps.
* The three parts were not dissected (one 2-seed arm, as designed).

## 5. C5: born-again listwise self-distillation

**Math.** Per feed, `L = CE(y/Σy, softmax(z_S)) + β τ² KL(softmax(z_T/τ) ‖ softmax(z_S/τ)) + 0.5 L_K9`, β = 0.5, τ = 1,
same equal-per-feed normalisation as the main loss (sum over feeds / 512; all feeds, including the ones without a
positive, get the dense target). Teacher: the frozen round-1 `nn/ckpt/s2_v2_feed_k9.pt` (val 0.38271), whose logits over
all 246.7M train candidates were precomputed once with the random-like domain (`teacher.py`, 7.8 min, fp16, 0.49 GB;
sanity check: its holdout NDCG from the fp16 logits is 0.3000, vs 0.3009 in its own log with the soft production domain).

**Result: WIN by the decision rule, driven by the holdout; small on val.**

| comparison | Δval (SE) | Δhold (SE) | per seed Δval |
|---|---|---|---|
| kd vs base (stage 1) | +0.00064 (0.00041), z 1.6 | **+0.00098** (0.00024), z 4.1 | +0.00040 (s23), +0.00087 (s24) |
| kd vs s2_base (for reference) | +0.00008 (0.00046) | +0.00074 (0.00026) | -0.00004, +0.00019 |

* Like Muon, KD gives a large early lead (+0.003 val at 2-5k steps, the teacher is a finished model) that shrinks to
  +0.0004..+0.0009 by the end of the cosine.
* Seed 24 (the clean read, no shared init with the teacher) gains more than seed 23.
* Diversity cost: per-feed correlation with the teacher on val rises from 0.9822 (base_s24) to 0.9880 (kd_s24); seed 23
  (teacher's own init) stays at 0.992. KD students are less useful as separate ensemble members.

## 6. Closeout: combining the winners

Rule from the design: combine the C2/C4/C5 winners (2 seeds), then stage 2 and the best fine-tune recipe. EMA (C1) and
the C3 recipe were NO on the baseline and are not turned on by default; the C3 arm-a recipe is used only where it was
shown to help (Muon models).

**Muon + WSD (lr 1e-3; `optim_muwsd`)** is clearly additive:

| comparison (stage 1, train-only) | Δval (SE) | Δhold (SE) | per seed Δval |
|---|---|---|---|
| muon+wsd vs base | **+0.00419** (0.00079), z 5.3 | **+0.00362** (0.00048), z 7.6 | +0.00389, +0.00448 |
| muon+wsd vs s2_base (baseline stage 1 + 2) | **+0.00362** (0.00075), z 4.9 | **+0.00338** (0.00044), z 7.7 | +0.00345, +0.00379 |
| muon+wsd vs muon (WSD on top of Muon) | +0.00217 (0.00057) | +0.00067 (0.00033) | +0.00253, +0.00182 |
| muon+wsd vs muon + stage 2 | +0.00158 (0.00054) | +0.00103 (0.00030) | +0.00122, +0.00193 |
| muon+wsd vs wsd (Muon on top of WSD) | +0.00248 (0.00067) | +0.00264 (0.00039) | +0.00197, +0.00299 |
| stage 2 on top of muon+wsd | -0.00121 (0.00048) | **-0.00147** (0.00026) | -0.00140, -0.00101 |

Train-only val 0.38605 / 0.38572 (hold 0.30393 / 0.30398); the previous best train-only single model was 0.3827.
Stage 2 hurts after the WSD cooldown (holdout z -5.7; the decision is taken on the holdout), so the pipeline is
stage 1 -> fine-tune, without stage 2.

**Full pipeline (fine-tune OOF, frozen 49 `n2clean` features):** Muon + WSD -> cos+EMA fine-tune (`fta_optim_muwsd`)
OOF 0.38833 / 0.38827.

| comparison (val OOF) | Δ (SE) | per seed |
|---|---|---|
| vs baseline pipeline (base -> s2 -> ftb) | **+0.00318** (0.00071), z 4.5 | +0.00353, +0.00283 |
| vs base -> s2 -> fta (same fine-tune recipe) | **+0.00293** (0.00069), z 4.3 | |
| vs muon -> s2 -> fta (WSD instead of stage 2) | +0.00109 (0.00051), z 2.1 | |

2-seed blends (per-feed z mean): baseline pipeline 0.38608, Muon pipeline 0.38789, **Muon+WSD pipeline 0.38859**.
For reference, the production 8-model NN ensemble `preds/nn_final2_val_oof.npy` (round 1, n2full features) scores
0.38634; adding the 2 Muon+WSD models to it gives 0.38936 (they are only 0.963 correlated with the baseline pipeline).

**All winners (Muon lr 2e-3 + WSD + KD; `optim_mu2wsdkd`)** and its decomposition (`optim_mu2wsd` = the same
without KD), all at stage 1 (train-only):

| comparison | Δval (SE) | Δhold (SE) | per seed Δval |
|---|---|---|---|
| all winners vs muon+wsd (lr 1e-3) | -0.00079 (0.00054) | +0.00075 (0.00031) | -0.00081, -0.00078 |
| lr 2e-3 under WSD: mu2wsd vs muwsd | **-0.00097** (0.00058) | +0.00056 (0.00032) | -0.00106, -0.00089 |
| KD inside the combo: mu2wsdkd vs mu2wsd | +0.00018 (0.00053) | +0.00019 (0.00030) | +0.00025, +0.00012 |
| all winners vs base | +0.00339 (0.00072) | +0.00437 (0.00044) | |

**NO.** The loss comes from lr 2e-3, not from KD: under WSD the LR stays at its peak for 80% of the run, so doubling
the peak doubles most of the LR integral, whereas under the cosine (C2 follow-up) it only matters early. The extra fit
shows up on the train-distribution holdout (+0.0006) and not on val (-0.0010), i.e. it does not transfer to the
exploration feeds. KD is neutral once Muon + WSD are in (its C5 gain on the baseline overlapped with what WSD's
exploration cooldown and Muon already give). After the fine-tune the all-winners model equals Muon + WSD
(OOF +0.00006, SE 0.00053), and the 4-model blend of both recipes reaches 0.38905 (recipe diversity:
per-feed correlation 0.981 between the two recipes on the same seed).

Fine-tune check of the lr 2e-3 arm: `fta_optim_mu2wsd` OOF 0.38882 / 0.38808 vs `fta_optim_muwsd` 0.38833 / 0.38827: +0.00015 (0.00055), NO.

## 7. Recommendations for the integration step

1. **Adopt Muon + WSD as the new stage-1 recipe and drop stage 2**:
   `train.py $BASE --muon --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1`, then
   `finetune.py --epochs 4 --decay cos --ema 0.99 --ema_start 0.5` (AdamW fine-tune; `--muon` in the fine-tune is not
   needed). Keep lr 1e-3 under WSD; lr 2e-3 helps only with the cosine schedule.
2. Re-test it with `--xt cv` (design section 6). Both changes are optimizer/schedule changes and should be orthogonal to
   the input features, but that is unmeasured. The fine-tune feature set should come from the frozen 49-file list
   (`nnx/optim/n2clean_frozen.txt`, copies in `nnx/optim/feats/`), because feature dirs keep getting pruned.
3. Ensemble: Muon models are less correlated with AdamW models (0.963-0.978) and with each other (0.977-0.984) than
   AdamW seeds are (0.985-0.988). Seeds were saturated for AdamW, but a 2-seed Muon + WSD blend still adds +0.0003 over
   its best member. The 6-model blend of the three Muon + WSD recipes (lr 1e-3, lr 2e-3, lr 2e-3 + KD) reaches 0.38917.
4. Ready-made test predictions (per-feed z-scored, mean of the 5 fold models' EMA weights) are in
   `<repo>/preds_x/optim/`:
   `nn_fta_optim_muwsd_s{23,24}_test.npy` (best), `nn_fta_optim_mu2wsd_s{23,24}_test.npy`,
   `nn_fta_optim_mu2wsdkd_s{23,24}_test.npy`, `nn_fta_optim_muon_s{23,24}_test.npy`, with matching `_val_oof.npy` and
   per-fold `_test_folds.npy`. Train-only checkpoints are in `nnx/optim/ckpt/`.
5. Caveats: all numbers come from 2 seeds on a shared GPU, and identical runs differ by about 0.0004 on val. Three
   fine-tune recipes were compared on val OOF for the Muon models (the protocol's limit), so the +0.0011 fine-tune
   recipe gain carries a small selection bias. The all-winners closeout arm (lr 2e-3 + KD) was chosen over the
   pre-registered lr 1e-3 three-way combo to save one 2-seed slot. Its lr effect was then isolated by the extra
   `optim_mu2wsd` arm, so KD and lr are both separately measured on top of Muon + WSD at lr 2e-3, but KD was not
   measured at lr 1e-3.
6. Not done: dissecting C4 into its three parts (WSD alone / mixing / aux anneal), and the parking-lot items.



## 8. Implementation notes and reproduction

All flags default to the baseline behaviour; checkpoints keep the original cfg keys (`split_kv` is stored only when on).

```bash
cd <repo>/nnx/optim && source sh/env.sh      # GPU N, 16 threads, $PY, $BASE, $S2, waitmem()
# C0 + C1 (shared baseline with the EMA instrumentation), stage 2, baseline fine-tune
$PY train.py --variant base_s23 $BASE --seed 23 --ema 0.999,0.9995,0.9998 --ema_start 0.5
$PY train.py --variant s2_base_s23 --init base_s23 $S2
NNX_FEATS=<repo>/nnx/optim/feats $PY finetune.py --variant s2_base_s23 --out ftb_s23 --extra $(cat n2clean_frozen.txt) --lr 5e-5 --epochs 6
# C2 Muon (+ stage 2 and fine-tunes with AdamW)
$PY train.py --variant optim_muon_s23 $BASE --seed 23 --muon
# C3 fine-tune arms
NNX_FEATS=... $PY finetune.py --variant s2_base_s23 --out fta_s23 --extra $(cat n2clean_frozen.txt) --lr 5e-5 --epochs 4 --decay cos --ema 0.99 --ema_start 0.5
NNX_FEATS=... $PY finetune.py --variant s2_base_s23 --out ftlp_s23 --extra $(cat n2clean_frozen.txt) --lr 5e-5 --epochs 4 --decay cos --ema 0.99 --ema_start 0.5 --lpft_epochs 1 --lpft_lr 5e-4 --extra_lr_mult 3
# C4 WSD
$PY train.py --variant optim_wsd_s23 $BASE --seed 23 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1
# C5 KD (teacher logits once)
$PY teacher.py
$PY train.py --variant optim_kd_s23 $BASE --seed 23 --distill <repo>/nnx/optim/teacher/s2_v2_feed_k9_train.npy --distill_w 0.5 --distill_tau 1
# best pipeline: Muon + WSD stage 1 (no stage 2) -> cos+EMA fine-tune
$PY train.py --variant optim_muwsd_s23 $BASE --seed 23 --muon --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1
NNX_FEATS=<repo>/nnx/optim/feats $PY finetune.py --variant optim_muwsd_s23 --out fta_optim_muwsd_s23 --extra $(cat n2clean_frozen.txt) --lr 5e-5 --epochs 4 --decay cos --ema 0.99 --ema_start 0.5
# paired judgement
$PY paired.py --base <repo>/nnx/optim/preds/nn_base_s{seed} --treat <repo>/nnx/optim/preds/nn_optim_muon_s{seed}
$PY paired.py --val_suffix _val_oof.npy --base .../nn_ftb_s{seed} --treat .../nn_fta_optim_muwsd_s{seed}
```

Files: `train.py` (`--ema/--ema_start`, `--muon`, `--sched/--cooldown_frac/--cooldown_mix/--aux_anneal`,
`--distill/--distill_w/--distill_tau`; holdout npz), `model.py` (`extra_init`, `split_kv`), `finetune.py`
(`--decay`, `--ema/--ema_start`, `--lpft_epochs/--lpft_lr`, `--muon`), `teacher.py`, `paired.py` (+ `--tsv`),
`curves.py` (per-step comparison for the kill rule), `corr.py` (per-feed prediction correlation), `sh/` (job queue,
`queue.txt` / `worker.log`). Logs: `nnx/optim/logs/<variant>.log|.json`, `<out>_ft.log|.json`.
Bugs found and fixed during the run: (1) `copy.deepcopy` of a model holding the non-leaf `aux_out` forward cache failed
at the fine-tune EMA start (4 fine-tunes re-run); (2) a queue race started one fine-tune before its stage-2 checkpoint
existed (re-queued behind a guard). No result is affected.

