# RECHECK 11-14, lens "winner_model": structural decisions of the ranker seen from #1's side

Scope: CPU only (12 threads or fewer), no training process touched, nothing written to `preds/` or `subs/`. Scripts, logs and outputs: `nn_x/w12/recheck/winner_model/` (`c01`..`c07`, `out_c0*.txt`, `out/`). Read first: readme, docs 01 / 02, `nn_x/w5/{model,data,train,cd_arms,common}.py`, COVERAGE_MAP (all 298 lines), `E_kd_predeclared.txt`, the three TOP1 reports, `r6_tta.md`.

Every delta below is paired per feed on the 14,423 val feeds (SE = sd / sqrt(n)) unless stated. "Measured" = run here today; "inferred" = argument.

## 0. Verdict

1. **No structural decision of the pipeline is both untested in the current regime and likely to be worth +0.002.** I went through 15 of them (§2). Twelve are closed by a measurement in the Muon / 10-epoch / KD regime or by a data fact; three small ones were never varied under KD (§3 item 4).
2. **Seven CPU checks, five of them nulls** (§1): re-shuffle TTA is an exact no-op; sub-feed TTA hurts; fp32 inference is null at pool level; a label-free "production selection" score is already inside the pool; the fine-tune has no feed-size slice where it helps.
3. **One finding concerns arms that are running now.** The `kwsub5 / kwsub2` screen cannot decide function-matching KD as configured: its stated second motive is false on measurement, and it moves the training feed-size mix away from test (§3 item 1).
4. **One new data fact:** the exploration candidate pool is wider than the production top-300. Production candidates at rank 200+ are still separable from exploration candidates at AUC 0.70, and no sub-sampling of a production feed reproduces a val/test candidate set (§1.4, §1.5).
5. What I could not find: an explanation of #1's +0.0038. Nothing in the model structure accounts for it.

## 1. What was measured

### 1.1 Data facts (`c01`)

| Fact | train | val | test |
|---|---|---|---|
| Label values | {0, 1} only | {0, 1} only | - |
| Feeds without a positive | 0 of 3,151,208 | 0 of 14,423 | - |
| Feeds with 1 / 2 / 3+ positives | 66.6% / 18.6% / 14.9% | 74.5% / 15.4% / 10.2% | - |
| M range | 5-300 | 5-299 | 5-300 |
| Exploration feeds (v2 posterior > 0.5) | 291,763 (9.3% of feeds, 12.4% of rows) | all | all |

There is no graded or second target, no zero-positive feed and no feed below M = 5. The questions "more than one positive type?" and "are feeds without a positive used?" are closed by the data.

### 1.2 Permutation and sub-feed behaviour of a saved student (`c02`, `c03`)

CPU fp32 replay of `w5_e10_kd05w192_s951-954` on val. Sub-feeds are label-free: the candidates are split at random into G groups and the feed is scored without one group (G = 2: two halves; G = 5: 80% kept). All in-model set inputs are recomputed; the 20 precomputed `ctx_*` columns still describe the full feed.

| Read | Result |
|---|---|
| Permuting the candidate axis, padding included (114,740 rows) | max abs logit change 1.2e-4; order changed in 1 feed (an fp tie). The model is exactly permutation-equivariant. |
| Per-feed Spearman, full-feed vs sub-feed logits, 50% kept (1 member) | 0.986 (M < 50: 0.974; M >= 150: 0.9945) |
| Same, 80% kept (4 members) | 0.9944-0.9949 (M < 50: 0.989-0.990) |

| TTA arm | Member delta vs full feed | 4-member pool delta |
|---|---|---|
| 50% kept, 4 repetitions, mean of sub-feed z (1 member) | -0.00353 ± 0.00071 (M < 50: -0.0071; M >= 150: -0.0006) | - |
| 50% kept, mixed 0.5 / 0.5 with the full-feed score | -0.00088 ± 0.00047 | - |
| 80% kept, 4 scores per candidate | -0.00042, -0.00073, -0.00121, -0.00016 (SE 0.00032 each) | **-0.00072 ± 0.00033** |
| 80% kept, mixed 0.5 / 0.5 with the full-feed score | -0.00035, -0.00034, -0.00052, +0.00026 (SE 0.00022 each) | -0.00015 ± 0.00023 |

- The 50% curve improves with repetitions (-0.0072, -0.0056, -0.0049, -0.0035 for 1-4) but a 1/K fit ends near -0.002. Removing context costs accuracy; it is not noise that averaging removes.
- Verdict: **re-shuffle TTA is a no-op, sub-feed TTA is negative at member and pool level.** This closes the "SCTTA" row with a direct measurement (it had been closed by analogy with history-dropout TTA).
- Caveat: the stale `ctx_*` columns handicap sub-feed scoring; a version with recomputed columns was not run.

### 1.3 fp32 versus bf16 inference (`c03`)

CPU fp32 replay minus the stored GPU predictions: members +0.00038, +0.00001, +0.00015, -0.00023 (SE 0.00017-0.00019); 4-member pool **-0.00012 ± 0.00015** (0.40122 vs 0.40134). Null, as in `r6_tta.md` for d128 members.

### 1.4 Does sub-sampling make a production feed look like a val/test feed? (`c04`)

LightGBM on label-free, position-free set aggregates (mean / sd / min / max of f1-f5 and four cosines, categories per candidate), 5-fold AUC of [val + test feeds of a size band] against constructed train populations of the same size. 0.5 = same composition.

| Population (train) | size 9-15 | size 24-36 | size 60-90 |
|---|---|---|---|
| Exploration, natural size (floor: time drift) | 0.694 | 0.606 | 0.596 |
| Exploration M >= 200, random subset (user-mix control) | 0.607 | 0.624 | 0.612 |
| **Production, natural size** | 0.977 | 0.978 | 0.977 |
| Production M = 2x target, random half (what `--aug_cand 0.5` reaches) | 0.950 | 0.955 | 0.966 |
| Production M >= 200, random subset | 0.845 | 0.915 | 0.955 |
| Production M >= 200, subset with every positive kept | 0.873 | 0.928 | 0.959 |

SE 0.002-0.007. Halving a production feed moves the AUC from 0.977 to 0.95-0.97. Even a 3- to 20-fold thinning of the longest production feeds stays at 0.85-0.96, against a floor of 0.61.

### 1.5 "SELECT": the production ranker's candidate selection as a teacher (`c06`)

New mechanism, not in COVERAGE_MAP. A per-candidate classifier "comes from a production feed (1) or an exploration feed (0)" is a label-free proxy of the production score. Fit on 160K M-matched train feeds, raw per-candidate columns only, no click label, no position, holdout feeds excluded, user-disjoint test fold.

| Read | A: content + affinity (23 columns) | B: A + popularity / co-visitation (28) |
|---|---|---|
| Held-out candidate AUC | 0.749 | 0.785 |
| Production rank 0-4 / 20-49 / 200+ vs exploration candidates | 0.836 / 0.754 / 0.696 | 0.862 / 0.793 / 0.724 |
| Score alone, val NDCG (random 0.289; f1 alone 0.328) | 0.3427 | 0.3480 |
| Score alone, holdout NDCG | 0.2646 | 0.2727 |
| Within-feed Spearman to the W8 pool, val | 0.51 | 0.52 |
| Nested residual on W8, val | +0.00035 ± 0.00032 | +0.00028 ± 0.00023 (weight -0.02) |
| Nested residual on the W8 TO pool, val | +0.00003 ± 0.00037 | 0 (weight 0 chosen) |
| Nested residual on the W8 TO pool, holdout | -0.00024 ± 0.00016 | -0.00038 ± 0.00020 |

- The score tracks production rank monotonically, so it is a real proxy of the production score. Without one click label it reaches about 60% of a click-trained GBDT's lift over random.
- **It adds nothing to the pool.** Residual form closed.
- Measured fact with a consequence: rank-200+ production candidates are still separable from exploration candidates (0.70). Inference: exploration feeds sample a pool that is wider than the shown top-300. Alternative reading not excluded: a different user mix among deep scrollers (the feed-level control in §1.4 bounds that at about 0.62).

### 1.6 Smaller reads

| Check | Result |
|---|---|
| FT minus TO by feed size, W8 and C3 (`c05`) | Overall W8 -0.00069 ± 0.00061, C3 -0.00060 ± 0.00057. No size bin has FT above TO beyond 1 SE (largest +0.00115 ± 0.00110, C3, M 100-199). The fine-tune is not a short-feed repair. |
| Gradient clipping at 1.0 (logs of `w5_e10_kd05w192_s951`, `base_s951`) | Clipped share 0.000 on the plateau and in the cooldown (max norm 0.82 / 1.02). The clip is inactive after warm-up; the "never varied" row needs no test. |
| Feed-size mix under `--aug_cand` (`c07`) | Train mean M 78.3 → 59.2 (0.5) → 47.7 (0.2); share M >= 100: 0.286 → 0.208 → 0.144. Val 96.4 / 0.371, test 101.6 / 0.393. |

## 2. Structural decisions: what we do, the strong alternative, status

| # | Decision (file:line, `nn_x/w5/`) | Strong alternative | Status and regime | Verdict |
|---|---|---|---|---|
| 1 | Binary click target, target = y / sum(y) (`data.py:86`, `model.py:653`, `train.py:488-491`) | Graded target; zero-positive feeds; hindsight depth | No other label exists (§1.1). EVENT-DEPTH dropped; hindsight closed 11-14 (label copy) | Closed by data |
| 2 | Feed-normalised listwise CE (`train.py:491`, `model.py:663`) | Batch norm; `idcg` weights (`model.py:658-662`); metric-aware feed weights | Batch -0.0015 (K1); reweighted / truncated CE negative (CPU zoo, rho70 -0.0035). `idcg` has no recorded run | Closed except `idcg` (P 0.02) |
| 3 | Every train feed once per epoch, M-bucketed batches (`data.py:107-124`, `train.py:389`); 30K holdout never trained (`train.py:290-292`) | Recency, exploration or learnability sampling; bigger batches | rec33, DW3, rho70, bs4, cbs all negative under hard labels | Closed |
| 4 | 2 set-attention blocks without positions + within-feed z / percentile inputs (`model.py:457-459`, `614-618`, `35-48`) | Candidate retrieval, similarity bias, early fusion; set TTA | cset, JOINT closed; removal -0.0020. TTA measured here: negative | Closed |
| 5 | Frozen int8 item vectors, unit-normalised, through Linear on [e, e·u] (`data.py:97`, `139`; `model.py:60`, `411`, `530`); no ID table | Hashed ID table; processed embeddings; encoder states | `kw_id` -0.0057 under KD; 0 of 5 treatments; K/V branch -0.0001 | Closed |
| 6 | 256 newest events, summed field embeddings + recency index, 1 block (`data.py:141`, `model.py:449`, `549-553`) | 2 layers, GRU, surface-restricted view, longer history | `kw_hl2` null under KD; HS1 -0.003; crop -0.0041; earlier snapshots null | Closed |
| 7 | User vector as one token and as a product with the candidate (`model.py:443`, `544`, `530`) | Non-history residual, clusters | P1, P2 closed | Closed |
| 8 | Feed knowledge: log M, log(1 + H), user-0 flags, domain token; no time (`model.py:105`, `109-111`, `439`, `536-541`; `train.py:364`) | Time inputs; domain at inference; M < 50 domain labels | TIME closed; production vs random-like token +0.00011 ± 0.00017; XPOOL null | Closed |
| 9 | 1.0M (d128) / 2.1M (d192) parameters (`train.py:106-110`) | 10-50M with distilled targets | Under KD: d192 +0.0005 pool, d256 / 3 blocks / FFN x4 fail, d224 running; LLM backbones -0.004 | Closed; see §3 item 6 |
| 10 | Exploration feeds: weight 1 in epochs, domain soft mix, 2% cooldown replay; p_random zeroed for M < 50 (`train.py:286`, `364`, `423`, `466`) | Higher replay, re-weighting, all-M posterior | m01 / m04 / m10, DW3, XPOOL all null or negative, **hard labels only** | Open under KD (§3 item 4) |
| 11 | K9 production-order aux, weight 0.5 → 0.1 (`train.py:503-513`) | No aux or other dose when the teacher already carries it | ap1, aa0 closed under hard labels; **never varied under KD** | Open under KD (§3 item 4) |
| 12 | Inference: one full-feed pass, bf16 trunk, fp32 head (`common.py:48-60`) | fp32; TTA | Both measured here: null / negative | Closed |
| 13 | No task from history events inside the ranker (pre2 scalars only) | Next-event aux, pseudo-feeds, future-window aux | SSL trigger FAIL, PSEUDO-DIRECT closed, FUTWIN closed | Closed |
| 14 | Dropout 0.1, weight decay 0.02, clip 1.0 (`train.py:110`, `121`, `381-385`, `523`) | Stronger regularisation against the measured width memorisation | drop 0 -0.0006 under KD; cooldown dropout 0.2 = parity; clip inactive (§1.6). **Weight decay never varied in any regime** | Open: wd only (§3 item 4) |
| 15 | Production-ranker knowledge enters only as within-top-M order (K9) | Its candidate selection (§1.5) | Measured here: already in the pool | Closed (residual form) |

## 3. Ranked items

**1. Risk: the running `kwsub5 / kwsub2` screen cannot decide function-matching KD.** (measured; P 0.85 that the handicaps are real)
- Its second motive in `E_kd_predeclared.txt` ("a random sub-feed of a production feed is closer to the val/test format") does not hold: AUC 0.977 → 0.95-0.97 (§1.4).
- It shifts the training size mix away from test: the share of M >= 100 falls from 0.286 to 0.208 / 0.144 against 0.39 on test (§1.6). On val, feeds with M >= 100 carry 44% of W8's loss (`out_c05.txt`).
- Sub-feeds are harder inputs for this model family (-0.0007 pool at 80% kept), and 20 of the 34 extra inputs stay stale.
- The teacher targets themselves are fine: full-feed logits restricted to a sub-feed agree with sub-feed logits at Spearman 0.986-0.995 (§1.2), which answers the open probe of TOP1_GAP_EXTERNAL lever 1.
- Inferred: the 10-epoch member bar (+0.0008 val, +0.0005 holdout) tests the augmentation at a short schedule. The function-matching paper reports its gain at long schedules (from memory, not re-read), and `kw_e20` shows that the plain long schedule still memorises under KD.
- Reading guide when the arms finish: report the member delta for M < 50, 50-99, 100+. A loss on M >= 100 with a gain below 50 is the size-shift signature, not a verdict on function matching.

**2. Lever (conditional on item 1): size-preserving sub-sampling with a long schedule.** P 0.06.
- Sub-sample only feeds with M >= 100, q in [0.7, 1], so the size mix stays near test; pair it with 20 epochs and compare with the existing `kw_e20` (-0.0008 / -0.0004 vs 10 epochs).
- Pass if the member beats `kd05w192` by +0.0010 val and +0.0005 holdout on 2 seeds. Cost about 13 GPU-h plus a small `train.py` flag.
- Zero-GPU pre-check first: run `c02_perm_subfeed.py` on a finished `kwsub` checkpoint. If its sub-feed TTA is still negative, the student did not become sub-feed consistent and the long-schedule argument weakens.

**3. Null: test-time augmentation by re-shuffling or sub-setting.** Closed with the numbers of §1.2. Do not build.

**4. Lever (low): three knobs never varied under KD.** P about 0.04 each, 0.10 for any.
- K9 aux weight (0 and 0.25): the teacher's soft target may already carry what the aux taught.
- Cooldown replay (0.05, 0.10): the hard-label optimum of 0.02 was set by click memorisation on the replayed pool, which soft targets damp.
- Weight decay (0.05, 0.1): never varied in 300+ runs, and it is the direct counter to the measured width memorisation.
- One pre-declared batch on the `kd05w192` base, 2 seeds per arm, Holm over 6 arms, same screen bar as the `kw_` levers. About 40 GPU-h. Every one of the 5 capacity levers re-tested under KD failed, so the prior is low.

**5. Null: production selection as a teacher (SELECT).** Residual form closed (§1.5). An NN-input or aux form is not worth a run: the target is a deterministic function of inputs the model already has, unlike K9, whose target carries the production ranker's hidden inputs. P 0.03.

**6. Long bet, not recommended: a 10-50M parameter regime.** P 0.03.
- Inferred: 5.4M positives at about log2(M) bits each is roughly 3e7 bits of label information; KD re-uses the same labels. That supports 1-2M parameters, which is where the pipeline sits.
- Measured by others: d192 memorises at the plateau; d256 and 3 blocks fail under KD.
- A regime change needs new label information. The only untapped source (about 50M history positives) was tried twice: pre2 scalars (+0.0008 at 10 epochs) and pseudo-feeds (+0.0011 ± 0.0007 member, closed 11-14).

**7. Nulls for the record.** fp32 inference (§1.3); target and zero-positive feeds (§1.1); fine-tune by feed size (§1.6); gradient clip inactive (§1.6).

**8. Documentation fix.** docs/01 §3 and the XPOOL log entry describe exploration feeds as a random M-subset of the list whose top-M production shows. §1.4 and §1.5 say the pool is wider than the top-300. Consequence for future proposals: production feeds contain no non-elite candidates for their user, so no re-sampling of production feeds can emulate a test candidate set; only the 292K exploration feeds and val do.

## 4. Not checked

- Sub-feed TTA with recomputed `ctx_*` columns, and any TTA on fine-tuned models (no fold checkpoints are saved).
- TTA and fp32 on the holdout (val only; 4 members for the 80% arm, 1 member for the 50% arm).
- The SELECT classifier with val/test candidates as the reference class, and SELECT as a ranker input.
- `kwsub` checkpoints (still training at the time of writing).
- Files `out/*_sub2x1_*` are leftovers of a 1,500-feed smoke test and should be ignored.
