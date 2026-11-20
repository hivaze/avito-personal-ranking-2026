# NNX verdict: adversarial verification of the modern-tricks workflow

Written 2026-11-11 by the verifier. Inputs: `reports/nnx_exp_{attn,feat,optim}.md`, code in `nn_x/attn`, `nn_x/feat`,
`nnx/optim`, every saved prediction in `preds_x/{attn,feat,optim}`, the run logs/args, and one fresh seed pair run for
this verdict (code copies `nn_x/verify/{attn,optim,feat}`, outputs `preds_x/verify/`, GPU N only).
Analysis scripts (re-runnable, CPU): `nn_x/verify/recheck.py` (all paired deltas + null calibration; output `nn_x/verify/recheck.out`), `nn_x/verify/s25.py` (seed-25 check).

## TL;DR

**Confirmed. Merge these into every final NN member. All of them are already ported into `nn/`, and I checked the port line by line against the experiment code:**

| # | trick | flags (nn/train.py) | evidence (2 seeds, paired per feed) | seed-25 replication |
|---|---|---|---|---|
| 1 | **Muon + WSD cooldown with exploration mix + aux anneal, no stage 2** | `--muon --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1` | vs base+stage 2: val **+0.00362** (SE 0.00075, z 4.8), holdout +0.00338 (calibrated z 4.5); both seeds and both metrics positive. Full pipeline OOF +0.00318 (z 4.5) | **replicated**: val +0.00194 (z 2.2), holdout +0.00317 (calibrated z 3.2). 3-seed pooled val **+0.00306** (z 4.6), holdout +0.00331 |
| 2 | **A4: TabR-style retrieval values (top-16 history items)** | `--retr_m 16` | vs base: val **+0.00186** (z 2.4), holdout +0.00326 (calibrated z 4.3). Holds after stage 2 (+0.00161) and after fine-tune (OOF +0.00149, z 2.1). Dissection: the `T(k_c-k_j)` term carries it (meta-only arm vs A4: holdout -0.00288, calibrated z -3.8) | (with A1, below) |
| 3 | **A1+A4 together** (the gate adds a little on top of A4) | `--attn_gate --retr_m 16` (nn/); `--attn_gate elem --retr_m 16` in nn_x/attn | vs base: val **+0.00273** (z 3.5), holdout +0.00389 (calibrated z 5.1). After stage 2 +0.00292; after fine-tune OOF **+0.00195** (z 2.6); per-seed deltas within 0.0005 of each other at every stage | **replicated**: val +0.00289 (z 3.3), holdout +0.00369 (calibrated z 3.7). 3-seed pooled val **+0.00278** (z 3.8), holdout +0.00382; per seed +0.00250 / +0.00295 / +0.00289 |
| 4 | **Cos + EMA fine-tune for Muon-trained checkpoints** | `finetune.py --epochs 4 --decay cos --ema 0.99 --ema_start 0.5` | on Muon checkpoints vs the const-LR snapshot fine-tune: OOF +0.00112 (SE 0.00038), both seeds positive; neutral on AdamW checkpoints (+0.00025) | not re-run |

Muon alone (`--muon`, +0.00201 val / +0.00295 holdout) and WSD alone are confirmed parts of row 1. On the AdamW baseline, WSD replacing stage 2 is only moderate (val +0.00114, z 2.3; holdout +0.00074, calibrated z 1.1). On top of Muon it is clear: Muon+WSD vs Muon+stage 2 gives val +0.00158 (SE 0.00054), and stage 2 after Muon+WSD hurts (holdout -0.00147, calibrated z -2.2).

**Rejected or not supported:** A2 FoX, A3 QK-norm+SSMax, A5 MoE as a single-model change, B1 PLE, B2 propensity head, B3 semantic IDs, B4 EWMA features, B5 TabM-mini, C1 stage-1 EMA, C3a on AdamW checkpoints, C3b LP-FT, C5 born-again KD, Muon lr 2e-3 (under WSD), and stage 2 after WSD (harmful). The reasons are in section 5.

**Downgraded from what the reports claim:** A1 on its own (LIKELY), the A1 increment over A4 (reported as WIN), C5 KD (WIN), and the Muon lr 2e-3 follow-up (WIN). Each of these "wins" rested on holdout z-scores, and the holdout per-feed SE turns out to be about 2.3x too small (section 2.2). With calibrated SEs, none of them clears 2 SE.

---

## 1. What was checked, and the outcome

| check | method | outcome |
|---|---|---|
| Reported numbers | Recomputed all ~55 paired comparisons from the saved per-candidate predictions with `src/metrics.ndcg_per_feed` (val 14,423 feeds; holdout npz files) | **All reproduce to 5 decimals** (val/holdout means, per-seed deltas, pooled delta, SE, z) |
| Paired SE math | `paired.py`/`judge.py`: per-feed delta averaged over seeds, then sd/sqrt(n_feeds) | Correct. It keeps the cross-seed correlation of per-feed deltas (slightly larger than the naive sqrt(se1²+se2²)/2). It ignores run-level noise, see 2.2 |
| Seed pairing | Compared `args` in every run's `logs/*.json` against `base_s{23,24}` | Every treatment uses the same seed as its baseline (23<->23, 24<->24). Stage 2 uses `--seed 0` for both arms. Fine-tunes use identical args, identical frozen 49-feature list, GroupKFold by user (deterministic) |
| Identical configs | Same JSON diff, ignoring defaults that are off | **The only differences are the intended treatment flags.** `base_s23/24` additionally carry `--ema 0.999,0.9995,0.9998`. That instrumentation keeps shadow fp32 copies updated by `_foreach_lerp_` after `opt.step()` and never feeds back into the raw model, so the raw baseline trajectory is unchanged (nnx/optim/train.py:241-249, 310-317). Attn arms ran in a different code copy from the baseline. I diffed `nnx/optim/train.py` against `nn_x/attn/train.py`: with flags off, the training paths are identical (same param groups, schedule, loss); the extra `run_diag` call is eval-only and restores `model.train()` (diag.py:25,156) |
| Checkpoint selection | Code + `best_step` in logs | Stage 1: `--select_on holdout` in every arm (train.py:343-348). Stage 2: `--keep last`, so the last step always overwrites the "best", and the val-valued `select_on` default has no effect. Fine-tune: snapshot mean over all epochs or final EMA weights, no OOF-based epoch choice |
| Val-label leakage | Code read | The holdout is a fixed seed-12345 draw (`p_random>0.9`, M>=50), removed from `feeds` before batching. The WSD cooldown pool is drawn from `feeds`, so it also excludes the holdout (train.py:135-139, 225). Val is used only for logging and diagnostics. A4 retrieves only from the feed's own history, which is a strict snapshot at feed time (`reports/eda_history_leak.md`), with no cross-feed links. KD teacher is train-only (its own checkpoint was holdout-selected). |
| Tie artefacts (pessimistic ties) | Within-feed tied-score fraction of val predictions | <= 0.006% of candidates, 0-0.005% of positives, in base and treatment alike. No tie artefact |
| Holds on both val and holdout | See table in 2.3 | Yes for Muon, Muon+WSD, A4, A1+A4. WSD alone vs stage 2: val yes, holdout only at calibrated z 1.1 |
| Fresh seed pair | Seed 25, baseline `feat_base_s25` (pre-registered shared baseline, never used to judge attn/optim arms; run with the identical BASE flags) | Section 3 |

### Minor protocol caveats (none flips a verdict)
* The Muon fine-tune recipe (cos+EMA) was picked from 3 fine-tune recipes on val OOF. That is a small selection effect, of order a few 1e-4.
* The final optim recipe uses lr 1e-3, chosen over lr 2e-3 on **val**. The holdout preferred 2e-3 (+0.00056), so this is a val-driven choice between 2 options. After the fine-tune the two are equal (+0.00015, SE 0.00055).
* The stage-2 baseline uses the production stage-2 recipe, which (via the argparse default) trains the aux head with MSE, not the listwise K9. Its weight on random-like feeds is (1-p_random), close to 0. So "WSD replaces stage 2" is measured against the production stage 2, not against a re-tuned one.
* The 6-member A4-type blend (0.38813) and the 0.38958 Muon+A4 blend below are equal-weight, but which members to include was decided after seeing val. Treat them as indicative.

## 2. Statistics

### 2.1 Pooled paired SE (as reported) is computed correctly
`d_f = mean_seeds(treat_f - base_f)`, `SE = sd(d_f)/sqrt(14,423)`. My recomputation gives the same values. Per-feed and between-seed views agree for the big wins: e.g. Muon+WSD vs base+s2 per-seed val deltas are +0.00345 / +0.00379, A1+A4 vs base +0.00250 / +0.00295, A1+A4 OOF +0.00201 / +0.00189.

### 2.2 Null calibration: val SE is about right, holdout SE is ~2.3x too small
Same-config, different-seed runs (`base_s23`, `base_s24`, `feat_base_s25`) and an identical-config rerun (`nn_v2_feed_k9` vs `base_s23`):

| null pair | val d (z) | holdout d (z) |
|---|---|---|
| s24 - s23 | -0.00092 (-1.47) | -0.00101 (**-2.75**) |
| s25 - s23 | -0.00060 (-0.94) | +0.00006 (+0.15) |
| s25 - s24 | +0.00032 (+0.52) | +0.00107 (**+2.85**) |
| rerun, same seed | +0.00040 (+1.27) | n/a |

* Val: rms z ≈ 1.0, so the per-feed SE describes the real run-to-run noise. Ten older same-config seed pairs (`aux_sim_s*`) give rms z 0.57.
* **Holdout: rms z ≈ 2.3.** The run-to-run sd of a holdout delta is ≈ 0.00085, against a per-feed SE of 0.00037. Likely causes: the larger, lower-variance M>=50 holdout feeds make the per-feed SE small, and checkpoint selection takes the max over ~20 evals on this same holdout, which adds a run-level shift.
* Correction used in this verdict: `SE_hold,cal = sqrt(SE_feed² + 0.00085²/n_seeds)`.

### 2.3 Calibrated table of claimed wins (2 seeds, 23/24)

| claim | val d (SE, z) | holdout d (reported z → calibrated z) | both seeds +? | verdict |
|---|---|---|---|---|
| Muon + WSD vs base+s2 | +0.00362 (0.00075, 4.8) | +0.00338 (7.7 → **4.5**) | yes / yes | **CONFIRMED** |
| Muon + WSD vs base | +0.00419 (0.00079, 5.3) | +0.00362 (7.5 → 4.7) | yes / yes | CONFIRMED |
| Muon + WSD → fta vs base→s2→ftb (OOF) | +0.00318 (0.00071, 4.5) | - | yes | CONFIRMED |
| Muon + WSD → fta vs base→s2→fta (OOF, same FT recipe) | +0.00293 (0.00069, 4.3) | - | yes | CONFIRMED |
| Muon (C2) vs base | +0.00201 (0.00060, 3.4) | +0.00295 (7.9 → 4.2) | yes / yes | CONFIRMED |
| Muon + s2 vs s2_base | +0.00204 (0.00068, 3.0) | +0.00235 (5.9 → ~3.5) | yes / yes | CONFIRMED |
| WSD (C4) vs s2_base | +0.00114 (0.00049, 2.3) | +0.00074 (2.6 → 1.1) | yes / yes | LIKELY alone; confirmed inside the Muon combo |
| Stage 2 after Muon+WSD | -0.00121 (0.00048, -2.5) | -0.00147 (-5.7 → -2.2) | both negative | CONFIRMED harmful: drop stage 2 |
| A4 retrieval vs base | +0.00186 (0.00077, 2.4) | +0.00326 (7.1 → **4.3**) | yes / yes | **CONFIRMED** |
| A4 + s2 vs s2_base | +0.00161 (0.00079, 2.1) | +0.00309 (6.6 → 4.1) | yes / yes | CONFIRMED |
| A4 → s2 → ftb (OOF) | +0.00149 (0.00072, 2.1) | - | yes | CONFIRMED |
| A1+A4 vs base | +0.00273 (0.00079, 3.5) | +0.00389 (8.3 → **5.1**) | yes / yes | **CONFIRMED** |
| A1+A4 + s2 vs s2_base | +0.00292 (0.00081, 3.6) | +0.00377 (7.7 → 4.9) | yes / yes | CONFIRMED |
| A1+A4 → s2 → ftb (OOF) | +0.00195 (0.00074, 2.6) | - | yes | CONFIRMED |
| A1 alone vs base | +0.00042 (0.00046, 0.9) | +0.00050 (1.8 → 0.75) | yes / yes | NOT DEMONSTRATED (small) |
| A1 increment over A4 | +0.00087 (0.00047, 1.8) | +0.00063 (2.2 → 0.94) | yes / yes | LIKELY small (reported WIN) |
| A1 increment over A4 after FT (OOF) | +0.00046 (0.00051, 0.9) | - | yes | not significant |
| Muon fine-tune: fta vs ftb on Muon ckpts (OOF) | +0.00112 (0.00038, 2.9) | - | yes | CONFIRMED (small selection bias) |
| Muon lr 2e-3 vs 1e-3 (cosine) | +0.00079 (0.00050, 1.6) | +0.00073 (2.5 → 1.1) | yes / yes | NOT DEMONSTRATED (reported WIN) |
| Muon lr 2e-3 vs 1e-3 (under WSD) | **-0.00097** (0.00058, -1.7) | +0.00056 (1.8 → 0.8) | val both negative | REJECT 2e-3 under WSD |
| C5 KD vs base | +0.00064 (0.00041, 1.6) | +0.00098 (4.1 → 1.5) | yes / yes | NOT DEMONSTRATED (reported WIN); neutral inside the combo |
| A5 MoE vs base | +0.00027 (0.00047, 0.6) | +0.00053 (1.9 → 0.8) | yes / yes | REJECT as a single model |

A1 viewed across all its train-stage contexts: 4 independent run pairs (gate vs base, gate+retr vs retr, seeds 23/24) are all positive on val (+0.00052, +0.00033, +0.00126, +0.00048). Their mean is +0.00065 with SE ≈ 0.0003. That is real but small, and the val fine-tune absorbs most of it (OOF +0.00046, z 0.9; in 2-member blends the A4 pair 0.38746 beats the A1+A4 pair 0.38723). **Keep the gate because it is cheap and never negative, but do not count on it.**

## 3. Fresh seed-pair confirmation (seed 25, GPU N)

Outputs: `preds_x/verify/nn_{attn_gate_retr_s25,optim_muwsd_s25,s2_feat_base_s25}_{trainonly_val.npy,holdout.npz}`, checkpoints `nn_x/verify/{attn,optim,feat}/ckpt/`, logs `nn_x/verify/*/logs/`.
GPU N was shared with the pretraining agent's job (its allocation per `CURRENT_STATUS.md`). No workflow jobs were running on it, so I used the workflow's 2 process slots. Runs: `nn_x/verify/attn` A1+A4 seed 25; `nn_x/verify/feat` stage 2 of the existing `feat_base_s25` (the S2 recipe, seed 0); `nn_x/verify/optim` Muon+WSD seed 25. The baseline is the pre-registered `feat_base_s25` (BASE flags, holdout-selected; val 0.38156, holdout 0.30090), so this is one genuinely new seed for both claims.

| seed-25 run | val | holdout (selected step) |
|---|---|---|
| `feat_base_s25` (baseline, stage 1) | 0.38156 | 0.30090 (19670) |
| `s2_feat_base_s25` (baseline + stage 2) | 0.38241 | 0.30097 (last) |
| `attn_gate_retr_s25` (A1+A4) | 0.38446 | 0.30458 (19000) |
| `optim_muwsd_s25` (Muon + WSD, no stage 2) | 0.38435 | 0.30414 (19000) |

| comparison (paired per feed) | val d (SE, z) | holdout d (SE, calibrated z) |
|---|---|---|
| A1+A4 vs base, seed 25 | **+0.00289** (0.00087, 3.3) | +0.00369 (0.00053, 3.7) |
| A1+A4 vs base, seeds 23/24/25 pooled | **+0.00278** (0.00074, 3.8) | +0.00382 (0.00045, 5.8) |
| Muon+WSD vs base+stage 2, seed 25 | **+0.00194** (0.00089, 2.2) | +0.00317 (0.00052, 3.2) |
| Muon+WSD vs base (stage 1), seed 25 | +0.00279 (0.00091, 3.1) | +0.00324 (0.00054, 3.2) |
| Muon+WSD vs base+stage 2, seeds 23/24/25 pooled | **+0.00306** (0.00067, 4.6) | +0.00331 (0.00040, 5.3) |
| (stage 2 on the baseline, seed 25) | +0.00084 (0.00057, 1.5) | +0.00007 (cal z 0.1) |

**Both claims replicate on a fresh seed, on val and on the holdout.**
* A1+A4: the replication effect (+0.0029 val) matches the original +0.0027, and the three per-seed val deltas lie within 0.0005 of each other.
* Muon+WSD: the seed-25 val gain (+0.0019) is smaller than on seeds 23/24 (+0.0035, +0.0038). The 3-seed estimate is therefore revised to about +0.0031 val / +0.0033 holdout against the production stage-1+stage-2 pipeline. One caveat: on seed 25 the holdout picked step 19000, not the last step, and val was 0.38344 at the last step (eval-to-eval jitter ≈ 0.001 during the cooldown). With last-step selection the seed-25 val gain would be about +0.0010.
* Not re-run here: the fine-tuned (OOF) versions of either claim. The fine-tune shrinks the A1+A4 gain from about +0.0027 to +0.0020. For Muon+WSD it keeps most of it (+0.0032 OOF on seeds 23/24).

## 4. Merge recipe for the final ensemble (exact flags and code locations)

All four confirmed items already exist in `nn/` (ported by the NN agent, defaults off). I diffed the ports against the experiment code. They are faithful: same math, same init under a forked RNG, same parameter grouping, same schedule.

**Stage 1 (single stage, no stage 2):**
```
cd nn && CUDA_VISIBLE_DEVICES=<allowed> ../.venv/bin/python train.py --variant NAME --no_pos --aux_pos 0.5 --aux_mode listwise \
  --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --epochs E --eval_every N \
  --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1 \
  --retr_m 16 --attn_gate [--xt ...] --seed S
```
**Fine-tune (Muon checkpoints):** `finetune.py --variant NAME --extra <preset> --lr 5e-5 --epochs 4 --decay cos --ema 0.99 --ema_start 0.5`
(with `nn/ft_list.sh`: `FTARGS="--epochs 4 --decay cos --ema 0.99 --ema_start 0.5"`).

| trick | experiment code | ported code in nn/ |
|---|---|---|
| A4 retrieval values | `nn_x/attn/model.py:186-249` (`HistRetrieval`), built at :444-447, applied at :563-567 (`c = c + Z(r)` before the cross blocks) | `nn/model.py:151-198` (class), :332-335 (built under `extra_init`), :472-475 (applied); flag `nn/train.py:109` |
| A1 gated attention | `nn_x/attn/model.py:267-269` (gate built under `extra_init`), :311-315 (`o * sigmoid(W_g x)`, per head, before `W_O`); in hist_enc, cross, set blocks | `nn/model.py:202-260` (`Attn(gate=...)`), :348-355; flag `nn/train.py:108` (`--attn_gate`, store_true = "elem") |
| Muon | `nnx/optim/train.py:181-196`; split k/v with identical init `nnx/optim/model.py` (`Attn(split_kv)`) | `nn/train.py:214-228`, `nn/model.py:210-219`. Muon covers every 2-D Linear with min dim > 1, including the A1 gate and the A4 `wk/t1/t2/Z` matrices. Embeddings, LayerNorms, biases, scalars and 1-output layers stay on AdamW |
| WSD + cooldown mix + aux anneal | `nnx/optim/train.py:204-231` (schedule, `aux_w`, pool), :259-267 (batch substitution), :294-304 (annealed K9 weight) | `nn/train.py:236-259`, :270-274, aux weight `aux_w(step)` in the K9 loss |
| Cos + EMA fine-tune | `nnx/optim/finetune.py:59-63` (flags), :256 (cosine), :279-303 (EMA, final-EMA predictions) | `nn/finetune.py:55-59`, :216, :248 |

## 5. Rejected tricks and reasons

| trick | result | reason |
|---|---|---|
| A2 FoX forgetting bias (`--fox`) | val -0.00019, hold -0.00014 | Null. Recency is already covered by the learned recency embedding plus the sim-bias heads |
| A3 QK-norm + SSMax (`--qk_norm --ssmax`) | val -0.00006, hold -0.00010 | Null. The mechanism works (entropy falls with M), but NDCG does not move |
| A5 fine-grained MoE (`--moe 8 --moe_topk 2`) | alone val +0.00027; over A1+A4 +0.00015 train-only, -0.00045 after s2, -0.00034 after FT | No single-model gain, +84% parameters, and one router collapsed (set block 1). Useful only as a diversity member in a blend |
| B1 PLE numeric embeddings | val -0.00043, hold +0.00078 (calibrated z 1.1); OOF -0.00010 | Fits production-traffic step functions that do not transfer to exploration feeds; the holdout "gain" is within run noise |
| B2 propensity head | val +0.00007 | Null; the head learns, but its output is uncorrelated with the label |
| B3 semantic-ID codes | val +0.00038 (z 0.8), hold +0.00003 | Null (no memorisation, but no signal either) |
| B4 EWMA history features | val -0.00002 | Null; duplicates what target attention already computes |
| B5 TabM-mini k=4 | val +0.00009 (3 seeds) | Null at 1.45x cost; members are 0.999-correlated |
| C1 stage-1 weight EMA | +0.00003 (β 0.999); -0.00057 (β 0.9998) | The cosine anneal already averages; a long EMA hurts |
| C3a cos+EMA FT on AdamW checkpoints | +0.00025 (z 0.7) | Neutral on AdamW models. Use it only for Muon models (row 4) |
| C3b LP-FT | -0.00014 | Null |
| C5 born-again KD | val +0.00064 (z 1.6), hold calibrated z 1.5; inside the combo +0.00018 | Not significant once the holdout SE is calibrated. Neutral on top of Muon+WSD, and it raises correlation with the teacher (less ensemble diversity) |
| Muon lr 2e-3 | cosine: +0.00079 val (z 1.6), holdout calibrated z 1.1. Under WSD: **-0.00097 val** (both seeds -0.0011/-0.0009), +0.00056 holdout. After FT +0.00015 | The extra fit shows up only on the train-distribution holdout, not on val. **Use lr 1e-3 with WSD** |
| Stage 2 after WSD | Muon+WSD: val -0.00121, holdout -0.00147 (calibrated z -2.2); WSD: val -0.00065 | Re-warming to 2e-4 undoes the cooldown. Drop stage 2 for WSD members |

## 6. Flags for the lead (affect the running v12 members)

1. **lr 2e-3 under WSD is not supported.** The v12 runs (`v12_all_*`, `v12_allprew_*`) use `--muon --lr 2e-3 --sched wsd`. At 2 epochs this costs train-only val -0.00097 (both seeds, z -1.7) for +0.00056 holdout, and ends equal after the fine-tune. At 10 epochs WSD holds the peak LR for 8 epochs, so the "fits the train distribution, not exploration feeds" effect is likely larger, not smaller. Recommendation: use `--lr 1e-3` for new members, or at least run one paired lr 1e-3 vs 2e-3 check at 10 epochs before seeding more 2e-3 members.
2. **The v12 fine-tunes use the wrong recipe for Muon checkpoints.** `ft_list.sh` without `FTARGS` runs the const-LR 6-epoch snapshot fine-tune (ftb). On Muon checkpoints, the cos+EMA 4-epoch recipe is +0.00112 OOF (both seeds). Set `FTARGS="--epochs 4 --decay cos --ema 0.99 --ema_start 0.5"`. The argparse last value wins, so this overrides `--epochs 6`.
3. **The cooldown mix becomes a different regime at 10 epochs.** `--cooldown_mix 0.3` with a 20% cooldown is about 1.9 passes over the 134K-feed exploration pool at 2 epochs. At 10 epochs it is 0.3 × 19,670 / 644 ≈ **9 passes**. This was never tested. The holdout is excluded from the pool, so it would reveal over-fitting to the pool. Watch the holdout curve during the cooldown, and consider `--cooldown_mix 0.1`, which keeps about 3 passes.
4. **The Muon × A4 × gate interaction was never measured by the workflow at 2 epochs.** The only evidence is the NN agent's single-seed 10-epoch run (c), which is +0.0037 val / +0.0048 holdout at 20k steps, plus the blend check here: fine-tuned Muon+WSD pair + A1+A4 pair = **0.38958**, against 0.38859 for the Muon+WSD pair alone (+0.00098, SE 0.00059). The two act on different things and look complementary. Combining them in one model is reasonable, but one more paired seed at the final epoch count would confirm additivity.
5. Holdout z-scores must be read about 2.3x smaller than printed by `paired.py`/`judge.py`. Gate acceptance on val (paired, >= 2 seeds) plus "holdout not worse".
