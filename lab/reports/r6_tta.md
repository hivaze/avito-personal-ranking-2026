# R6: inference-time tricks (TTA) on existing checkpoints, CPU replay

**Status:** DONE 2026-11-12 03:40 UTC. CPU only (`CUDA_VISIBLE_DEVICES=""`, 12 threads, 2 processes). No training, and nothing written outside `solvers/r6/`, the scratchpad and this report.
- Section 0 was written before the first val pass. The report file mtime is 01:05:41, and the first pass ended at 01:06:57 after 66.8 s.
- Section 0b was written after the unit tests only.

## TL;DR

| arm (6 members, train-only val, paired per feed vs deterministic CPU replay) | 6-member ensemble Δ ± SE (z) | members > 0 | mean member Δ | verdict |
|---|---|---|---|---|
| **T1** MC-dropout, K = 8 | −0.00009 ± 0.00024 (−0.4) | 4 / 6 | +0.00006 | **FAIL** |
| **T2** history-event dropout 10%, K = 8 | −0.00008 ± 0.00034 (−0.2) | 5 / 6 | +0.00036 | **FAIL** (ensemble criterion) |
| **T3** history crop {256, 192, 128, 64}, averaged | **−0.00100 ± 0.00045 (−2.2)** | 0 / 6 | −0.00087 | **FAIL** (harmful) |

- **No arm passes.** No TTA on the current NN members, and no member predictions written (`solvers/r6/out/` holds only the evaluation JSONs).
  - T2 is the classic within-member TTA gain: +0.00036 per member, 5 of 6 positive. It disappears completely in the 6-member z-mean. The cross-member ensemble already removes the variance that TTA removes, consistent with sol_ensemble.md.
  - MC-dropout converges to the deterministic model from below as K grows: ensemble K = 1 / 2 / 4 / 8 gives −0.00063 / −0.00079 / −0.00033 / −0.00009.
- **Replay check: the literal bar was NOT met, but the replay is faithful.** Mean per-feed Spearman vs the stored GPU preds is 0.99921-0.99939 for all 6 members, against a bar of 0.9999.
  - The bar is below the model's own bf16 noise floor: a CPU bf16-autocast emulation vs CPU fp32 gives 0.99934, the same as GPU vs CPU (0.99929 on the same 1,500 feeds).
  - Pearson is 0.9999984-0.9999992, and NDCG CPU − GPU is within ±0.00016 (< 1 SE) for every member. The replay also matches the earlier stale replay to 1.5e-5.
  - The TTA deltas are paired against this same CPU replay, so they do not depend on the GPU match (section 1).
- **Findings that matter beyond TTA** (section 5):
  1. **History length is binding.** Cropping the history costs, monotonically, −0.0007 (192), −0.0015 (128) and −0.0041 ± 0.0008 (64, z −5.4) on the ensemble. 29% of val feeds sit at the 256-event cap, and events 193-256 are worth −0.0015 on the 49% of feeds that have them. Keep `max_hist` 256.
  2. **The retrieval top-16 is ambiguous.** 30% of history events repeat an item already in the history.
     - Ties at the top-16 boundary: 29% of candidate rows with fp32 cosine, and **50% with the bf16 cosine** the GPU actually uses. 99.5% of rows have ties inside the top 16.
     - Which duplicate gets retrieved is therefore decided by `torch.topk` tie-breaking (it changes with padding width and device).
     - Candidate new mechanism (needs GPU, not tested here): unique-item retrieval with fp32 scores.
  3. fp32 vs bf16 inference: −0.00003 ± 0.00016. No gain from fp32 inference.

## 0. Pre-declaration (frozen before running)

- Code: `solvers/r6/tta.py`; replay copy `solvers/r6/{data.py,model.py}`.
  - `data.py` is copied from `solvers/stale/data.py`, the read-only variant of `nn/data.py` with no write path.
  - `model.py` is a byte-identical copy of `nn/model.py`.
- CPU only: `CUDA_VISIBLE_DEVICES=""`, 12 threads per process, at most 2 processes. Per-pass outputs go to scratch (`.../scratchpad/r6/tta/passes/`).
- Members (stage-1 train-only checkpoints, read-only from `nn/ckpt/`):
  - F7: `v17_pre2s_s101`, `s102`, `s103`, `s104` (inputs `cvnd_pre2s`);
  - F6: `v14_s93`, `v14_s95` (inputs `cvnd`).
- Replay check (gate for everything else): for each member, compare the deterministic CPU fp32 eval replay with `preds/nn_<v>_trainonly_val.npy` (GPU, bf16 autocast).
  - Required: mean per-feed Spearman >= 0.9999.
  - Also reported: min / 1st percentile, Pearson, and NDCG of both.
- Baseline: the deterministic CPU replay of the same member, from the same code path. TTA and baseline therefore differ only in the trick.
- Arms (K stochastic passes per member, averaged as the mean of raw logits per candidate):
  - **T1 MC-dropout, K = 8.**
    - `model.eval()`, then `.train()` only on every `nn.Dropout` and every `Attn` module. This covers attention-probability dropout inside SDPA, residual / FF dropout and the retrieval-value dropout, all with p = 0.1.
    - The Ranker module itself stays in eval mode, so there is no xt-dropout / aux-head branch. No other change.
  - **T2 history-event dropout, K = 8.**
    - Each valid history event (of the first min(H, 256)) is dropped independently with p = 0.1.
    - Survivors are compacted into a contiguous newest-first prefix, so recency indices shift and H decreases; H = 0 is allowed (train has 35K H = 0 feeds).
    - Every in-model history-derived input is recomputed from the new history: dense similarity / category / H features, target attention, history encoder and retrieval.
    - Precomputed train-stage inputs (xt: `ctx_*`, `pre2_hmax32`, `seq_cv_*`, ...) stay as they are.
  - **T3 history crop, 4 passes.** Keep the most recent H' events for H' in {all (256), 192, 128, 64}, with the same compaction and recompute as T2, and average the 4 passes. This arm is deterministic.
- Seeds: `torch.manual_seed(20261112 + 1000*arm + 10*member_idx + k)` before pass k. arm = 1 for T1, 2 for T2; member_idx 0..5 in the order listed above. The T2 drop masks come from a `torch.Generator` seeded the same way.
- Ensemble: member-equal per-feed z-mean (`nn/zmean.py: fz`) of the 6 members. The deterministic ensemble uses the 6 deterministic replays; the TTA ensemble uses the 6 pass-averaged members.
- Metric: `src/metrics.py ndcg_per_feed` on all 14,423 val feeds, with no subsampling.
  - Deltas are paired per feed (TTA − deterministic). SE = sd(per-feed delta) / sqrt(14,423).
- **Acceptance (per arm):** PASS only if
  - the 6-member ENSEMBLE delta > 2 SE, AND
  - at least 5 of 6 members have a positive point delta.
  
  Slices M<50 / M>=50 / Tue / Thu are reported for information only. Tue/Thu use the `nn/pcmp.py` day split.
- Non-decisional diagnostics: single-pass NDCG for each crop H' (sensitivity to history length, relevant to `max_hist`); the K-curve (K = 1, 2, 4, 8) for T1/T2; T1+T2 combined.
- If an arm passes:
  - report its cost (passes × measured CPU inference time per pass);
  - write the pass-averaged member predictions to `solvers/r6/out/` (never `preds/`).

### 0b. Amendment (~01:05 UTC, after unit tests only, before any val TTA result)

- **Unit tests** (scratch `unit.py`, 640 feeds, `v17_pre2s_s101`):
  - (a) compaction with nothing dropped is the identity (max |Δlogit| 0.0), and crop 256 equals det (0.0).
  - (b) T2 compaction equals physically deleting the same events from the flat history arrays and rebuilding `hist_off`/H (max |Δ| 0.0).
  - (d) MC mode switches 16 modules (Dropout + Attn) and leaves `Ranker.training` False. Same seed gives the same output; eval mode restores det exactly.
- **Width effect found in test (c).** Crop 64 differs from a physical crop by up to 0.118 logit. Bisection (`unit2.py`):
  - Inputs and dense features are identical, and `hist_enc` agrees to 1e-6. Only `HistRetrieval` differs (max 0.24).
  - Cause: its `topk(16)` over the candidate-to-history cosine has many exact ties from duplicate history items. In the test batch, 89% of candidate rows have a tie inside the top 16 and 49% have one at the 16/17 boundary.
  - `torch.topk` breaks those ties differently depending on the padded history width (64 vs 128/256). The top values are identical, but the chosen index sets differ for 23% of rows, which changes the recency bucket, type and source of the retrieved values.
  - So the model output depends on batch padding, an implementation-defined noise that also exists on GPU.
- **Consequence for the design.** T2/T3 keep the det batch width (compaction in place), so arm-vs-det differences come from the history change and not from padding.
- **Added non-decisional diagnostic W256.** det with every batch padded to width 256 (6 passes) measures how much NDCG this tie-breaking noise moves. It is not an arm, and no acceptance applies to it.

---------------------------------------------------------------------------------------------------------------------

## 1. Replay check: deterministic CPU fp32 vs `preds/nn_<v>_trainonly_val.npy` (GPU, bf16 autocast)

Source: `solvers/r6/out/tta_replay_check.json` (`tta.py check`).

| member | per-feed Spearman mean | p01 | share ≥ 0.9999 | Pearson | NDCG CPU | NDCG GPU | CPU − GPU ± SE |
|---|---|---|---|---|---|---|---|
| v17_pre2s_s101 | 0.99939 | 0.9924 | 0.39 | 0.9999992 | 0.39365 | 0.39361 | +0.00005 ± 0.00016 |
| v17_pre2s_s102 | 0.99937 | 0.9926 | 0.37 | 0.9999991 | 0.39526 | 0.39532 | −0.00005 ± 0.00017 |
| v17_pre2s_s103 | 0.99928 | 0.9909 | 0.32 | 0.9999988 | 0.39446 | 0.39439 | +0.00007 ± 0.00018 |
| v17_pre2s_s104 | 0.99934 | 0.9909 | 0.36 | 0.9999992 | 0.39578 | 0.39568 | +0.00010 ± 0.00017 |
| v14_s93 | 0.99921 | 0.9893 | 0.29 | 0.9999984 | 0.39387 | 0.39400 | −0.00013 ± 0.00017 |
| v14_s95 | 0.99923 | 0.9909 | 0.30 | 0.9999985 | 0.39264 | 0.39249 | +0.00016 ± 0.00019 |
| 6-member z-mean | | | | | 0.39877 | 0.39880 | −0.00003 ± 0.00016 |

- **Verdict on the pre-declared bar (mean per-feed Spearman ≥ 0.9999): NOT MET for all 6 members.**
- **Cause: bf16 numerics, not a wrong model.** `solvers/r6/diag/bf16diag.py` runs the same checkpoint on 1,500 random val feeds under CPU `torch.autocast(bfloat16)`, with the head and wide part kept in fp32 as `nn/model.py` does on CUDA:

  | pair | per-feed Spearman mean | p01 | share ≥ 0.9999 | Pearson |
  |---|---|---|---|---|
  | GPU (stored) vs CPU fp32 | 0.999289 | 0.9909 | 0.401 | 0.9999993 |
  | **CPU bf16 vs CPU fp32** (same machine, same code) | **0.999344** | 0.9909 | 0.380 | 0.9999993 |
  | GPU vs CPU bf16 | 0.999478 | 0.9910 | 0.481 | 0.9999994 |

  Switching only bf16 → fp32 on one device moves per-feed rank order exactly as much as GPU → CPU does. So 0.9999 is not reachable by any fp32 replay of these models; the noise floor is about 0.9993.
  - Much of it comes from the bf16 cosine matrix (`nn/model.py:76`, `:473` under `nn/common.py:51` autocast). It quantises cosines to about 0.004 steps and changes which tied history items `topk` retrieves (section 5.2). Max |Δlogit| reaches 0.64-0.77.
- **Other evidence of faithfulness:**
  - Pearson ≥ 0.9999984;
  - NDCG within 1 SE for every member, with mixed signs;
  - ensemble 0.39877 vs 0.39880;
  - `v17_pre2s_s101` det equals the CPU2 stale replay `solvers/out/stale/replay/cf_v17_pre2s_s101_base.npy` to max |Δ| 1.5e-5.
- **Decision.** I continued, and flag it here.
  - All TTA deltas are paired against the deterministic CPU fp32 replay of the same checkpoint, from the same code path. The only difference between arm and baseline is therefore the trick.
  - What this check cannot rule out: a trick whose effect exists only under bf16. That is irrelevant here, because nothing passed.
  - **Recommendation for future replay gates:** use Pearson ≥ 0.99999 plus NDCG within 1 SE, or per-feed Spearman against a CPU-bf16 emulation. Per-feed Spearman ≥ 0.9999 against bf16 GPU preds is unreachable.

## 2. Arm results (all 14,423 val feeds, no subsampling; `solvers/r6/out/tta_eval.json`)

Deterministic CPU NDCG: s101 0.39365, s102 0.39526, s103 0.39446, s104 0.39578, s93 0.39387, s95 0.39264; 6-member z-mean **0.39877**.

### 2.1 Ensemble (decisional) and slices (informational)

| arm | ensemble NDCG | Δ ± SE | z | M<50 | M>=50 | Tue | Thu | corr (ens vs det ens) |
|---|---|---|---|---|---|---|---|---|
| T1 MC-dropout K = 8 | 0.39869 | −0.00009 ± 0.00024 | −0.36 | −0.00020 ± 0.00048 | −0.00001 ± 0.00024 | −0.00071 ± 0.00035 | +0.00060 ± 0.00033 | 0.99958 |
| T2 hist-drop K = 8 | 0.39869 | −0.00008 ± 0.00034 | −0.23 | +0.00011 ± 0.00064 | −0.00021 ± 0.00037 | −0.00026 ± 0.00050 | +0.00011 ± 0.00045 | 0.99885 |
| T3 crop avg | 0.39777 | −0.00100 ± 0.00045 | **−2.22** | −0.00143 ± 0.00087 | −0.00071 ± 0.00047 | −0.00182 ± 0.00062 | −0.00011 ± 0.00066 | 0.99525 |

- The T1 Tue/Thu split has opposite signs (−0.0007 / +0.0006), both at about 2 SE. This is the usual day-level noise; the pooled value is null.

### 2.2 Members (Δ ± SE vs that member's deterministic replay)

| member | T1 | T2 | T3 |
|---|---|---|---|
| v17_pre2s_s101 | −0.00028 ± 0.00034 | +0.00007 ± 0.00038 | −0.00082 ± 0.00047 |
| v17_pre2s_s102 | +0.00016 ± 0.00034 | −0.00007 ± 0.00040 | −0.00048 ± 0.00046 |
| v17_pre2s_s103 | +0.00028 ± 0.00034 | +0.00090 ± 0.00045 | −0.00063 ± 0.00050 |
| v17_pre2s_s104 | −0.00038 ± 0.00033 | +0.00005 ± 0.00040 | −0.00088 ± 0.00046 |
| v14_s93 | +0.00006 ± 0.00039 | +0.00046 ± 0.00050 | −0.00083 ± 0.00048 |
| v14_s95 | +0.00053 ± 0.00038 | +0.00076 ± 0.00053 | −0.00158 ± 0.00050 |
| positive | 4 / 6 | 5 / 6 | 0 / 6 |
| mean | +0.00006 | +0.00036 | −0.00087 |

### 2.3 Acceptance (pre-declared: ensemble Δ > 2 SE AND ≥ 5/6 members positive)

| arm | ensemble Δ > 2 SE? | ≥ 5/6 members positive? | result |
|---|---|---|---|
| T1 | no | no (4/6) | **FAIL** |
| T2 | no | yes (5/6) | **FAIL** |
| T3 | no (significantly negative) | no (0/6) | **FAIL** |

Nothing was written to `solvers/r6/out/` as member predictions. `tta.py write <arm>` refuses unless the arm passed.

## 3. Non-decisional diagnostics

### 3.1 K-curves (prefix of the same seeds)

| K | T1 ensemble Δ | T1 mean member Δ (members > 0) | T2 ensemble Δ | T2 mean member Δ (members > 0) |
|---|---|---|---|---|
| 1 | −0.00063 ± 0.00040 | −0.00125 (0) | +0.00006 ± 0.00040 | −0.00015 (2) |
| 2 | −0.00079 ± 0.00031 | −0.00070 (0) | +0.00013 ± 0.00036 | +0.00007 (3) |
| 4 | −0.00033 ± 0.00028 | −0.00017 (2) | −0.00013 ± 0.00034 | +0.00044 (4) |
| 8 | −0.00009 ± 0.00024 | +0.00006 (4) | −0.00008 ± 0.00034 | +0.00036 (5) |

- **T1.** A single dropout pass costs −0.0013 per member, and averaging recovers it, approaching the deterministic model. The deterministic forward pass is already a good approximation of the MC mean (corr of the K = 8 mean vs det is 0.99998 for s101), so there is nothing to gain past K → ∞.
- **T2.** The member-level gain saturates at about +0.0004 by K = 4, and the ensemble stays at 0. Dropping history events is a perturbation the members already tolerate (single-pass mean member −0.00015). Averaging gives each member a little smoothing that the 6-member z-mean already provides.
- **T1 + T2** (16 passes per member): ensemble +0.00003 ± 0.00026, members positive 3 / 6.
- **T2 by history length** (ensemble): H < 64: +0.00094 ± 0.00081; 64-192: −0.00051 ± 0.00057; ≥ 192: −0.00004 ± 0.00049. No reliable segment.

### 3.2 Single crops (one pass at H', no averaging): sensitivity to history length

| H' | ensemble Δ ± SE (z) | Δ on feeds that the crop changes | members > 0 |
|---|---|---|---|
| 192 | −0.00074 ± 0.00032 (−2.3) | −0.00150 ± 0.00064 (H > 192, n = 7,123) | 1 / 6 |
| 128 | −0.00146 ± 0.00050 (−2.9) | −0.00207 ± 0.00071 (H > 128, n = 10,148) | 0 / 6 |
| 64 | **−0.00408 ± 0.00075 (−5.4)** | −0.00474 ± 0.00087 (H > 64, n = 12,430) | 0 / 6 |

- Val H distribution: median 191; 29.0% of feeds are at the 256 cap; 49.4% have > 192 events, 70.4% > 128 and 86.2% > 64.
- Feeds the crop does not touch are bit-identical (max |Δ| 0). This confirms the crop implementation and the fact that `xt` inputs were left unchanged.
- The averaged T3 loses because 3 of its 4 passes are worse members, and averaging cannot recover information the crops removed.

### 3.3 W256 (padding-width tie-break noise) and fp32 vs bf16

- **W256** (every batch padded to width 256): Δ = 0.00000 for all 6 members. Only 0.015% of candidate rows change (1,246 rows in all).
  - With `cand_budget 32768 / max_feeds 512`, almost every val batch already contains a feed with H > 192, so it is padded to 256 anyway.
  - The padding-dependent tie-break found in the unit test (0b) matters only for small or odd batches, for example single-feed inference.
- **fp32 CPU vs bf16 GPU, same checkpoints:** ensemble −0.00003 ± 0.00016. Members range from −0.00013 to +0.00016, all within 1 SE. Running inference in fp32 gives nothing.

## 4. Cost

Wall time per full-val pass, 12 threads. The host was oversubscribed by other jobs during the run (a high load average on a shared machine), so absolute times are pessimistic; ratios are the useful part. Source: `solvers/r6/out/tta_passes_timing.jsonl`.

| arm | s per pass (mean, n) | passes per member | cost per member vs det |
|---|---|---|---|
| det | 80.6 (6) | 1 | 1x |
| T1 | 217.8 (48); attention dropout forces SDPA's math path and draws Bernoulli masks for [B, 4, 257, 257] history attention | 8 | about 21.6x |
| T2 | 86.1 (48) | 8 | about 8.5x |
| T3 | 82.1 (18) | 4 (det + 3) | about 4x |

Total compute for this study: 126 passes, about 4.7 process-hours at 12 threads (17,040 s).

## 5. Implications for hyperparameters and model structure (user request: "also think about overall hyperparams, models structures")

1. **`max_hist` = 256 is binding, and old events carry signal.**
   - Section 3.2: losing events 193-256 costs −0.0015 on the feeds that have them, and keeping only 64 costs −0.0041 overall (z −5.4).
   - The provider caps history at 256, and 29% of val feeds sit at the cap, so the model would probably use more if it existed.
   - Practical rules:
     - never shorten history for speed;
     - an encoder that represents old events better is a plausible target. The current history encoder is one layer, and old events reach the candidates only through target attention and retrieval.
     - Candidates, in order of cost: `hist_layers 2` at d = 128; and a separate cheap summary token per recency band (for example events 0-31 / 32-127 / 128-255, mean-pooled) added to `ctx`.
   - Neither is tested. Both would be new different-recipe families, the documented ensemble lever, so they should be judged by the family gates (a) + (b).
2. **The TabR retrieval (A4, `--retr_m 16`) chooses among duplicate history items by `topk` tie-breaking** (`nn/model.py:179`).
   - `solvers/r6/diag/ties.py` (1,500 val feeds): on average 30.4% of a feed's history events (median 30.7%) repeat an item already in that history (same embedding row).
   - Over candidate rows with an embedding and more than 16 valid history items:

     | | tie at the 16/17 boundary | tie inside the top 16 |
     |---|---|---|
     | fp32 cosine | 29.3% | 86.8% |
     | **bf16 cosine** (GPU training and inference: the einsum at `nn/model.py:473` runs under autocast) | **49.6%** | **99.5%** |

   - So the 16 retrieved slots are often filled with repeats of one item, and which of its events (type, source, recency bucket) are kept is arbitrary. It even changes with padding width (unit test 0b: 23% of rows got a different index set in a width-64 vs width-256 batch).
   - The verdict says the `T(k_c − k_j)` term carries A4. Duplicates give the same `k_j`, so slots spent on them carry little new information.
   - **Candidate mechanism (GPU, new family, not tested here):**
     - retrieve over UNIQUE history items (first occurrence = most recent);
     - carry per-item aggregates into the value: count per event type, most recent recency bucket, number of distinct sources;
     - compute the top-k score in fp32;
     - break remaining ties by recency.
   - This is cheap (same m = 16) and deterministic. It changes what the retrieval sees, so it is a genuine different-recipe family rather than a seed.
3. **Dropout and stochastic inference.** MC-dropout gives nothing (3.1), and the deterministic pass already matches the dropout mean.
   - So dropout 0.1 acts purely as a training regulariser, and there is no inference-side "free ensemble" in these models.
   - Ensemble value has to come from seeds and recipes, as sol_ensemble.md found.
   - History-event dropout as a training-time augmentation is not in the coverage list (the closed items were xt-input dropouts). Its prior is low: the members are already robust to it at inference (single-pass −0.00015), and its TTA value vanished in the ensemble.
4. **Precision.** bf16 vs fp32 inference makes no difference (3.3), so keep bf16. If the retrieval is reworked (point 2), compute only its top-k score in fp32. That removes the extra bf16 ties at no measurable cost.
5. **Replay protocol.** Future CPU replays should use the gate in section 1 (Pearson plus NDCG within 1 SE), not per-feed Spearman ≥ 0.9999 against bf16 preds.

## 6. Reproduce / files

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/r6/tta.py run --members v17_pre2s_s101,v17_pre2s_s102,v17_pre2s_s103 --arms det,W256,T3,T1,T2
CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/r6/tta.py run --members v17_pre2s_s104,v14_s93,v14_s95 --arms det,W256,T3,T1,T2
CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/r6/tta.py check     # -> solvers/r6/out/tta_replay_check.json
CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/r6/tta.py eval      # -> solvers/r6/out/tta_eval.json
CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/r6/diag/{unit,unit2,bf16diag,ties}.py   # unit tests / diagnostics (4 threads)
```

- **Code:**
  - `solvers/r6/tta.py`: `set_mc` (l. 67), `compact_hist` (l. 79), `hist_drop_fn` (l. 95), `pad_hist_fn` (l. 102), `hist_crop_fn` (l. 114), acceptance (l. 316);
  - `solvers/r6/data.py`: copy of `solvers/stale/data.py`, no write path;
  - `solvers/r6/model.py`: byte-identical to `nn/model.py`.
- **Per-pass predictions** (126 files, float32, val cand-aligned): `/tmp/scratchpad/r6/tta/passes/`.
- **Logs:** `.../scratchpad/r6/tta/logs/`.
