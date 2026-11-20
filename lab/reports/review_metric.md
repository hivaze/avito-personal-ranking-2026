# Review: metric, evaluation and statistics

Reviewer scope: metric correctness, in-training evals, submission writer, blending, standard errors, calibration, statistical power.
Constraints: CPU only, read-only on the project. All scripts are in
`/tmp/scratchpad/review/metric/`
(`ref_metric.py`, `ref_real.py`, `ties.py`, `blendties.py`, `se.py`, `aa.py`, `hold_se.py`, `subchk.py`, `calib1.py`, `calib2.py`, `cold.py`, `zchk.py`, `nanscan.py`).

## TL;DR
* **The metric, the in-training evals, the z-blending and make_sub are all correct.** I compared them against two independent brute-force references, on random data with heavy ties and on real val predictions. Max difference: 3e-16.
* **Ties do not matter.** All saved predictions are fp32, and exact within-feed ties cover at most 0.017% of rows. Pessimistic and random tie-breaking give the same val NDCG to 6 decimals, so val does not under- or over-estimate test because of ties.
* **The SE premise of the task does not hold.** Val feeds are *not* clustered by user: 13,715 of 13,740 non-zero users have exactly one val feed.
  * User-clustered SE / iid SE = 1.000 on every delta I checked; it is also 1.000 on the train holdout.
  * An A/A test over 14 disjoint same-config/different-seed pairs gives rms z = 1.04.
  * So the per-feed SEs are calibrated and also cover seed noise. **Reported SEs are not understated.**
* **What does break the protocol is how the rule was applied, not the SE formula.** `prew_*` was accepted at val z = 1.44 (+0.00106 ± 0.00074; the status says ±0.0007). Several other levers were accepted through the OR-rule `max(z_val, z_hold) >= 2`.
* **calib.py is sound but has a large error bar.**
  * Total prediction-error SD is 0.0033 for public and 0.0027 for private: val sampling 0.0020 (user bootstrap) plus LB sampling 0.0026 / 0.0017.
  * The 4 public residuals (+0.0003..+0.0019) share both error sources, so together they act as roughly one draw. "Calib is pessimistic" and "val gains transfer >= 1:1" are not statistically established.
  * Richer covariates (M×H×user0, logistic density ratio) predict the public points *worse*.
  * Cold-share covariates are time-confounded and fail badly out of sample (+0.010 error predicting Thu from Tue). Never reweight val by item coldness.
* **Private LB noise.** Two submissions whose true private difference is below ~0.0002 are a coin flip (≤ 60% chance the better one wins). A +0.0005 val edge holds on private with only ~70-80% probability.

---

## Findings (ranked)

### METRIC-1 (medium, verified): `prew_*` train-stage input accepted below the 2-SE rule on val
* Evidence: `se.py` pair `nn_v8_nd_e3_trainonly_val -> nn_v12_prew3_trainonly_val`.
  * d = **+0.00106**, iid SE 0.00074, **z = 1.44**. User-clustered SE is identical (1.000×).
  * By day: Tue **+0.00009 ± 0.00103**, Thu +0.00212 ± 0.00105.
  * Holdout (`hold_se.py`): +0.00084 ± 0.00042, z = 2.00 (user-clustered also 2.00). A day-clustered SE (G = 12 train days) is 0.00065 (z = 1.29), because the prew effect is structurally day-dependent (the in-window construction).
* Single seed. CURRENT_STATUS 3g / nn_report l.441 write "+0.0011 ± 0.0007 → passes". Under the declared rule ("paired delta > 2 SE"; "new train-stage inputs must improve TRAIN-ONLY val"), val does not pass.
  * An inverse-variance combination of val and holdout gives z ≈ 2.45. The two sets measure different populations, though, and that val/holdout asymmetry is exactly what burned train-stage `pre_*`.
* Impact: at most ~±0.001 on val/private either way. But 3 of the 6 round-4 members (`v12_allprew_s83/84/85`) depend on it.
* Fix / test: when round 4 finishes, compare the z-mean of `v12_all_{e10,s81,s82}` against the z-mean of `v12_allprew_{s83,s84,s85}`, on train-only val (paired per feed) and on the holdout.
  * The per-feed SE is valid for this unpaired-seed comparison (A/A result below).
  * Accept prew only if val z ≥ 2 and the holdout is not negative. Otherwise keep both families with equal weights as a hedge rather than switching.

### METRIC-2 (medium, verified): the acceptance rule is applied as an OR of two tests, and several accepted levers have val z < 2
* `nn_x/*/paired.py`: WIN if `max(zv, zh) >= 2` (plus a loose seed check). This is an OR of two tests, so the false-positive rate is about 2× nominal.
  * nnx ran about 45 comparisons, so about one false WIN is expected under the null.
  * Accepted with val z < 2, from `nn_x/*/results.tsv` and my recomputation:
    | lever | val z | holdout z | follow-up |
    |---|---|---|---|
    | C5 KD | 1.56 | 4.1 | later no effect inside the combo |
    | A1+A4 gate increment over retr | 1.85 | 2.17 | after s2+ftb: +0.00046 ± 0.00051, verdict NO; `--attn_gate` is still in the live round-4 recipe |
    | 3-epoch train stage, stage 1 / stage 2 | 1.89 / 1.41 | 3.5 | accepted on holdout; later vindicated by 10 vs 5 epochs, val z 2.13 / holdout z 4.05 |
    | mix 0.5 FT + 0.5 TO over FT alone | 1.59 | – | over train-only: z 2.51; a pre-declared coarse choice, so acceptable |
  * The mix choice flip-flops across rounds: on nn_final3, OOF + train-only was −0.0007 ± 0.0004 (err_SYNTHESIS), so the mix weight sits at noise level.
* Impact: each such lever is worth ≤ 0.001. The composite round-4 recipe is strongly supported (+0.006 / +0.005 at 10k steps), so this is about hygiene, not a hidden loss.
* Fix: pre-declare one primary statistic per change type.
  * Train-stage changes: train-only val z ≥ 2 AND holdout ≥ 0.
  * Or a Stouffer combination z ≥ 2.5 when both point the same way.
  * Drop the OR-rule.
  * A finding "LIKELY" at stage 1 must be re-tested in the full pipeline before it stays in the recipe (the gate).

### METRIC-3 (low, verified): calib residuals are over-interpreted (status 3h)
* Evidence (`calib2.py`, 2000× bootstrap over users of v10):
  * Predicted public 0.38592 ± 0.00203; predicted private 0.38856 ± 0.00204. These are val sampling errors.
  * The realised LB has its own sampling noise: public 0.00263 (n = 8,562, per-feed SD 0.243); private 0.00172.
  * Total prediction-error SD: **public 0.0033, private 0.0027**.
  * The 4 residuals (+0.0003, +0.0013, +0.0014, +0.0019) share the val sample and the public sample. Their common component dominates, so "mean residual +0.0012" is about 0.4 SD of a single draw.
* Transfer:
  * v5→v10 val gain: +0.00707 ± 0.00108 (paired). The expected public-side SE of the realised gain is 0.00141. The observed public gain is +0.00766.
  * Transfer ratio 1.08, roughly 95% CI [0.6, 1.6]. Consistent with 1:1; ">= 1:1" is not established.
  * "Raw val-public gap shrinking (0.0077 → 0.0061)": the steps are within paired public noise (0.001-0.0014).
* Minor: with today's `src/calib.py`, v5 (exactly reconstructed as 0.5 z(nn_final2_val_oof) + 0.5 z(stk2_res2_r3_val)) predicts **0.37896**, not 0.3788. Its residual is therefore +0.0012, not +0.0014.
* Best private estimate for sub_v10: 0.3886 (calib).
  * Shrinking toward the public residual uses the val-error share of the public residual variance, 0.0020² / (0.0020² + 0.0026²) ≈ 0.37. That gives ≈ **0.3893 ± 0.0022**.
  * Treat any "lower bound" language as unsupported.

### METRIC-4 (low, verified): public/private boundary is off by 18 feeds
* Test timestamps are quantised to 128 s. `calib.py` uses `ts < 1790497920`, which gives 8,562 public feeds (0.29943).
* The organisers' rule (page_desc: closest to 30% by count, equal timestamps kept together) is met by `ts <= 1790497920`: 8,580 feeds (0.30005, |Δ| = 1.5 vs 16.5 feeds). The 18 feeds at exactly 1790497920 are public.
* Impact: predicted public +0.00003, private −0.00001. Fix: use `<=` in `calib.py`.

### METRIC-5 (low, verified): the organisers' text says "IDCG@30" while DCG runs over all candidates
* page_desc.html (scratchpad) defines `DCG = Σ target_i/log2(i+1), i = 1..M`, but says "IDCG@30 — DCG@30 at the ideal order".
* If implemented literally, a feed with > 30 positives scores DCG/IDCG@30 > 1. On val that affects 3 feeds (max 2.07) and adds **+0.00014** to the mean, the same for v5 and v10.
* NDCG@30 (both truncated) would give 0.333 / 0.342 on val (predicted public 0.322 / 0.331), versus actual public 0.380 / 0.388. That variant is ruled out.
* `src/metrics.py` (full IDCG) is correct for decisions: the possible offset is constant across models. It may explain about 0.00015 of the positive calib residual.

### METRIC-6 (low, verified): fine-tune GroupKFold puts the whole user-0 bucket in fold 0
* `nn/finetune.py` uses `GroupKFold(5).split(..., groups=va.user_id)`. All 658 user-0 feeds (a pooled unknown bucket, not one user) land in fold 0; folds 1-4 have none.
* So the fold-0 fine-tune never trains on user-0 feeds, while 4 of the 5 fold models averaged for test did. OOF on user-0 feeds is therefore pessimistic relative to test. Measured FT − TO on user-0 feeds: +0.0028 ± 0.0033.
* Impact: ≤ 0.0002 overall, and in the conservative direction.
* Fix: make user-0 feeds singleton groups, consistent with the rule "exclude user 0 from user-level stats".

### METRIC-7 (low, verified, historical): early stopping on the evaluated fold
* `src/stack_val.py` (unless `--fixed_rounds`) and `src/cv_val.py` pass the evaluated fold as `valid_sets` and keep `best_iteration`. The OOF NDCG is therefore optimistic.
* Not used by the current pipeline: the stk_ line is stopped, and `eval_family.py` and `stack2.py` use fixed rounds, which is correct.
* Fix: nested early stopping or fixed rounds if these scripts are revived.

### METRIC-8 (low, verified): fine-tuned OOF comparisons are noisier than one fine-tune suggests
* `aa.py`: across the 4 `v9_e10*_n2cp` fine-tunes (same recipe, different stage-1 seeds), SD of the val means is 0.00087. The per-feed SE implies 0.00062. One A/A pair reaches z = −2.30.
* Train-only A/A pairs are well calibrated (SD of means 0.0002-0.0006).
* Fix: for fine-tune recipe decisions, use ≥ 2 fine-tune seeds, or treat z < 2.5 as inconclusive.

### METRIC-9 (info, verified): a reported per-day split is not reproducible
* CURRENT_STATUS 3d gives nn_final3 OOF "Tue 0.3931 / Thu 0.3868". The current file gives **0.39047 / 0.38860**, whether days are split in MSK, UTC or at the median timestamp.
* The quoted pair also contradicts the overall 0.38958 (its weighted mean is 0.3901).
* Treat earlier "Thursday-half" confirmations as unverified.

### METRIC-10 (info, verified): day heterogeneity and the colder test period
* Per-day deltas (`se.py`):
  | comparison | Tue | Thu | heterogeneity z |
  |---|---|---|---|
  | final4 vs final3 OOF | +0.0016 | +0.0058 | −2.63 |
  | FT over train-only | −0.0009 | +0.0019 | −1.91 |
  | mix over train-only | +0.0007 | +0.0018 | |
* Thu is colder than Tue: the share of candidates never seen as a train candidate is 0.387 vs 0.313.
* Test is much colder and gets colder every day: Sat 0.434, Sun 0.457, Mon 0.474, Tue 0.490. Public averages 0.439, private 0.477, val 0.348.
* This suggests the fine-tuned (cold re-calibration) component is worth more on private than its val average. Across about 11 comparisons, though, one z of −2.6 is borderline.
* It cannot be quantified by reweighting (next item). Keep the FT component in the final blend as a hedge.

### METRIC-11 (info, verified negative result): coldness covariates must not be used to reweight val
* Post-stratifying on per-feed cold share to predict Thu from Tue gives 0.4036 vs actual **0.3944** (+0.0099 error). Thu from Tue in reverse: +0.0068.
* A logistic density ratio with cold share, LUO-zero share and user activity is worse still: +0.028 / +0.019 error, ESS ≈ 3,000. It predicts private 0.49-0.50.
* Why: within a day, high-cold-share feeds score higher (U-shape: 0.371 in the 0.2-0.4 bucket vs 0.54 at 0.6-0.8). The *cross-day* rise in cold share is a time effect (items become cold as time passes since the train period ends) and does not raise NDCG.
* Any importance-weighting of val toward test's coldness (for calibration or model selection) would be badly biased.

---

## Statistical power and LB noise (task item 6)
SD of the per-feed delta, sd(d), comes from real pairs (`se.py`). SEs use n_val = 14,423, n_public = 8,580 and n_private = 20,015.

| pair type | sd(d) | SE val | MDE val (80% power, z>2 rule) | SE public | SE private | private coin flip (≤ 60% correct) below | P(better on private \| val Δ = +0.0005) | \| val Δ = +0.001 |
|---|---|---|---|---|---|---|---|---|
| blend tweak (mix vs +0.1 pre) | 0.050 | 0.00042 | 0.0012 | 0.00054 | 0.00036 | 0.00009 | 0.82 | 0.97 |
| same-arch variant / seed | 0.089 | 0.00074 | 0.0021 | 0.00096 | 0.00063 | 0.00016 | 0.70 | 0.85 |
| model generation (final4 vs final3) | 0.096 | 0.00080 | 0.0023 | 0.00104 | 0.00068 | 0.00017 | 0.68 | 0.83 |
| v10 vs v5 | 0.130 | 0.00108 | 0.0031 | 0.00140 | 0.00092 | 0.00023 | 0.64 | 0.76 |

* Power of the "z > 2" rule at a true +0.001 effect: 0.69 (SE 0.0004), 0.37 (SE 0.0006), 0.23 (SE 0.0008). With single seeds, most real +0.001 levers will be (correctly) labelled inconclusive.
* For paired comparisons, val (14.4K labelled feeds) is *more* informative than public (8.6K). The public LB cannot confirm changes below ~0.002 between our own submissions, so don't spend submissions on small variants.
* Final selection: variants within 0.0005 on val are 70-80% likely to keep their order on private. Picking the 2 final submissions to differ meaningfully hedges better than picking near-duplicates.

---

## Checked and correct
1. **`src/metrics.py`**
   * Matches two independent brute-force references (`ref_metric.py`: sort-based with negatives first in ties, and explicit tie groups where positives take the worst slots).
   * On 9,000 random feeds (M 1..59, score cardinality 2 / 4 / 1000, ~35% feeds with P = 0): max diff 2.2e-16.
   * On real val preds (nn_final4_val_oof, nn_v9_e10_trainonly_val, pre_final_val), raw and rounded to 1/8 to force ties: max diff 3.3e-16 (`ref_real.py`). Forced ties cost about 0.017 NDCG, which shows why the fp32 head mattered.
   * Binary gain, 1/log2(rank+1), ideal DCG, pessimistic ties, P = 0 → 1 and the mean over feeds all match the organisers.
   * No feed in val, test or the train holdout has P = 0. Val M ranges 5..299, so no M = 1 feeds.
2. **In-training evals all call `src/metrics.ndcg`**: `nn/common.py` eval_ndcg / eval_subset (subset offsets recomputed correctly), `nn/holdout_eval.py`, `nn/finetune.py`, and `nn_x/*/paired.py`.
   * Logged values reproduce exactly from the saved preds: v9_e10 0.39141, v9_e10_s61 0.39131, v12_prew3 0.38839, v8_nd_e3 0.38733.
   * The holdout feed set in train.py equals holdout_eval.py; the reconstruction `M` matches `preds/holdout/holdout_meta.npy`.
   * The holdout is 30,000 feeds from 28,383 users, M ≥ 50, no P = 0.
3. **Prediction dtypes and ties** (`ties.py`, `blendties.py`):
   * All preds are float32.
   * Within-feed exact ties: 0-0.017% of rows, 0-3 positives.
   * Pessimistic minus random tie-break NDCG = 0.00000 for 13 main preds and for the v10 blend (3 tied pairs on val, 3 on test).
   * `*_test_folds.npy` are fp16, but they are used only as pseudo-label teachers.
4. **`src/make_sub.py`** (`subchk.py`):
   * Orientation: rank M−1 goes to the highest blend score, which matches the readme ("большее значение соответствует более высокой позиции").
   * Tie-free: 0 duplicate (feed, score); every feed has min 0 and max M−1.
   * Keys: the join on (feed_id, item_id) gives a key set identical to sample_submission; no nulls, NaN or duplicate keys; schema Int32 / Int32 / Float32.
   * `data/raw/sample_submission.parquet` is byte-identical to `data/competition/...`.
   * **sub_v10 exactly reproduces 0.5 z(nn_final4_test) + 0.5 z(nn_final4_trainonly_test) + 0.1 z(pre_final_test)**: rank match 1.00000, max diff 1 on the 3 float ties. sub_v5 reproduces exactly.
5. **Per-feed z-scoring** (`src/stack_val.feed_rank_z`, `nn/blend.feed_z`, evalq.py, nn/ensemble.py):
   * A constant feed gives z = 0 (no NaN), thanks to the +1e-9 / +1e-6 terms.
   * M = 1 cannot occur.
   * NaN would propagate to the whole feed and rank it last (pessimistic, silent). `nanscan.py` found no NaN or inf in any of 639 val/test pred files, and no constant feeds except the raw `pre_v*_hmax/idmax` component files.
   * make_sub asserts finiteness.
6. **SE clustering** (`se.py`, `hold_se.py`):
   * Val has 14,423 feeds and 13,741 users; only 25 users have 2 feeds; user 0 has 658 feeds.
   * Clustered/iid SE ratio on 11 reported deltas: 1.000 (user-0 feeds as singletons), 0.98-1.12 (user 0 as one cluster).
   * Holdout ratio: 1.000.
7. **Seed noise is captured by the per-feed SE** (`aa.py`): 14 disjoint A/A pairs give rms z 1.04 and mean |z| 0.90 (ideal 0.80). The per-model seed SD on val is about 0.0004-0.0006 for train-only models.
8. **Re-verified deltas (iid = clustered SE):**
   | comparison | Δ | ± SE | z |
   |---|---|---|---|
   | final4 vs final3, OOF | +0.00364 | 0.00080 | 4.6 |
   | final4 vs final3, train-only | +0.00401 | 0.00079 | 5.1 |
   | 10 vs 5 epochs, val | +0.00171 | 0.00080 | 2.1 |
   | 10 vs 5 epochs, holdout | +0.00183 | 0.00045 | 4.0 |
   | v10 mix + 0.1 pre | −0.00002 | 0.00042 | |
   | nn_final3 + 0.1 pre | +0.00019 | 0.00040 | |
   | FT over train-only | +0.00041 | 0.00073 | |
9. **`src/calib.py` math:**
   * The weights are a correct post-stratification, w = p_test(bucket) / p_val(bucket), mean-normalised.
   * All 10 buckets are populated (≥ 930 val feeds each); weights range 0.875-1.204; ESS is 14,297 (public) and 14,374 (private) of 14,423, so the reweighting adds almost no variance.
   * It reproduces v10 at 0.38592 / 0.38856.
   * Tue↔Thu cannot test the M reweighting, because the M mix is the same on both days (mean log M 4.0799 vs 4.0792). Cross-day error over 27 models is ∓0.0007 on the mean, and the RMS of model *differences* is 0.0017, i.e. half-sample noise.
   * Richer covariates (M×H×user0 cells; logistic on log M, log H, user0) shift predictions by −0.0002 / −0.0010. They fit both reproducible public points worse: residuals +0.0015 / +0.0022 and +0.0023 / +0.0030, versus +0.0012 / +0.0019 for M-only. Keep M-only.
10. **Other eval scripts:** `src/eval_family.py` and `src/stack2.py` use fixed rounds (no early stopping on the evaluated fold), and the SE of the seed-averaged per-feed delta is appropriate. `paired.py` pairs feeds correctly and asserts holdout M equality.

## Ideas
* **Settle prew and the gate with the round-4 runs** at no extra GPU cost: compare the 3-member z-means v12_all vs v12_allprew, and fta with vs without gate if both exist, using paired per-feed SEs on train-only val plus the holdout.
* **Decision rule:** pre-register the primary statistic; replace `max(zv, zh)` with "val z ≥ 2 AND holdout ≥ 0" or a Stouffer z ≥ 2.5. Re-test stage-1 "LIKELY" items after fine-tune before keeping them.
* **Secondary yardstick for the colder private period:** report Thursday-only and coldest-tercile val as tie-breakers (never as weights). Prefer blends that keep the fine-tuned (cold re-calibration) component.
* **Final 2 picks:** make them differ meaningfully, e.g. v10 vs the round-4 ensemble. Sub-0.0005 val differences are ~70-80% coin flips on private.
* **Robustness asserts** (for when the live jobs are done):
  * `metrics.py`: `assert len(scores) == M.sum() and (offsets == np.r_[0, np.cumsum(M)[:-1]]).all()`.
  * `assert np.isfinite(scores).all()`, so NaN can no longer silently rank last.
  * `calib.py`: `<=` boundary.
  * `finetune.py`: user-0 feeds as singleton groups.
* **Fine-tune recipe decisions:** use ≥ 2 fine-tune seeds (FT A/A spread is ~1.4× the per-feed SE).
* **Report level predictions with error bars**: private 0.3886 (calib) ± 0.0027, or ≈ 0.3893 after shrinking toward the public residual.
