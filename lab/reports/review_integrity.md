# Review: data integrity, alignment and input parity

Reviewer: integrity audit (CPU only, read-only on project files). Date: 2026-11-11.
Scripts and raw outputs are in `/tmp/scratchpad/review/integrity/`:
`align.py`, `basic.py`, `days.py`, `subcheck.py`, `featparity.py` (→ `featparity.csv`, `predparity.csv`), `pairs.py` (→ `pairs.parquet`), `pairs_agg.py`, `lhcheck.py`, `retention.py`, `typeshare.py`, `t1rule.py`, `t1cap.py`, `t1hist.py`, `typecap.py`, `cap200.py`, `trunc_lgb.py`, `idcontrib.py`, `trainxt.py`, `ownhist.py`, `user0leak.py`, `uezero.py`, `samets.py`.

## Bottom line

- **The submission path is clean.**
  - Every feature and prediction file has the right length.
  - Val/test row order is consistent: no test file loses its correlation with a shared reference model.
  - v10 matches the raw test ids and the sample submission, has no ties, and can be rebuilt bit-for-bit from its three components.
- **One real construction mismatch between train and val/test histories (INTEGRITY-1).**
  - Val/test histories cap view events (type 1) at about 100-110.
  - 28.5% of train feeds carry 113-210 views.
  - Below hpos 128, train histories are about half views; val/test histories are 11-28% views.
  - This is a step change at the split boundary. It is not a drift over time and not a traffic-type effect.
  - A LightGBM proxy shows no measurable cost from it: training on val-like truncated train histories gives −0.0014 ± 0.0008. So it is a parity fact to know about, not an accuracy leak.
- **The later-history signal is structurally present in test** (INTEGRITY-2). Test→test pairs look exactly like val pairs, and the v5_lh boost was applied correctly.
  - The zero public change must therefore come from the label or scoring side.
  - If the boosted items had all been negatives, public would have moved by −0.0005, which would show at 4 decimals. If all positives, +0.0037. Exactly 0.0000 is unexplained.
  - Private exposure is only 46 feeds, so this is not a private lever whatever the cause.
- **Smaller verified issues:**
  - `pre_id_contrib` is built differently for val and test (INTEGRITY-3). Negligible for the NN.
  - `it_hist_*_luo` has a non-causal density asymmetry between train and val/test (INTEGRITY-4).
  - LUO treats same-tick sibling snapshots as allowed (INTEGRITY-5). Negligible.
  - The share of zero user embeddings differs between val and test (INTEGRITY-6).
  - Val/test have hour-long collection gaps (INTEGRITY-7).
- **No feed has zero positives in train or val** (exact count 0). The "no positives scores 1" rule never fires on val, so nothing dilutes the metric.

---

## Findings (ranked by severity)

### INTEGRITY-1 (medium, verified): train histories are built with a larger per-type event cap than val/test histories

**Evidence**

**Type-1 count per history** (`t1hist.py`, `typecap.py`; train = 200K random feeds, val/test = all feeds):

| | train | val | test |
|---|---|---|---|
| feeds with ≥ 120 views | **26.8%** | 0.37% | 0.45% |
| p99 / p99.9 of views per feed | 208 / 213 | 114 / 174 | 114 / 167 |
| plateau in the count histogram | 100-109 **and** 200-209 | 100-109 only | 100-109 only |

Other types show the same pattern (p99.9 / max):

| type | train | val | test |
|---|---|---|---|
| 4 | 179 / 221 | 117 / 120 | 117 / 122 |
| 6 | 58 / 64 | 41 / 44 | 41 / 44 |
| 8 | 50 / 64 | 42 / 44 | 42 / 44 |
| 10 | 57 / 66 | 39 / 43 | 40 / 45 |

**Where it shows in the history** (type-1 share by hpos, `t1rule.py`):

| hpos | train | val | test |
|---|---|---|---|
| 0-15 | 0.578 | 0.596 | 0.596 |
| 16-63 | 0.563 | 0.573 | 0.575 |
| 64-127 | 0.569 | 0.553 | 0.555 |
| **128-191** | **0.477** | **0.283** | 0.285 |
| **192-255** | **0.404** | **0.114** | 0.112 |

- Uncapped feeds have identical view counts: train 67.3 vs val 67.1 per feed.
- The difference sits entirely in heavy users. Among H = 256 train feeds, 55% exceed the val/test cap.

**It is a property of the split, not of time or traffic** (`typeshare.py`, `cap200.py`):
- The share of train feeds with ≥ 113 views is flat across all 12 train days (0.277-0.291).
- Type-1 share is 0.515-0.522 on every train day, 0.438-0.439 on both val days and 0.440-0.443 on all four test days.
- Random-like (exploration) train feeds behave like train, not like val: 24.1% exceed the cap and their type-1 share is 0.512.
- Within a user it is only partly persistent: P(next train feed exceeds | this one exceeds) = 0.50, against 0.31 otherwise.

**Gap-matched retention** (`retention.py`). Share of A's newest 64 view events that are still in the same user's later history B:

| gap (days) | tr→tr | tr→va | tr→te |
|---|---|---|---|
| 2-3 | 0.941 | 0.844 | |
| 3-4 | 0.917 | 0.849 | |
| 4-6 | 0.882 | 0.743 | |
| 6-8 | 0.844 | 0.724 | 0.620 |
| 8-11 | 0.808 | 0.645 | 0.606 |
| 11-16 | 0.784 | 0.604 | 0.505 |

- Types 2 and 4 are retained about equally across the three columns.
- The lower retention also explains why train-feed positives are found in later test histories at recall 0.61, against 0.84 for tr→tr at gaps over 4 days.

**Impact**

- *Proxy measurement* (`trunc_lgb.py`):
  - Setup: 40K random-like train feeds, f1-f5 + category + 8 history-category count features, LightGBM lambdarank, 300 rounds, 3 seeds, evaluated on all of val.
  - Variant (a): features from the raw train history. Val 0.3349 / 0.3359 / 0.3367.
  - Variant (b): features from a val-like history (views beyond the K-th newest dropped, K ~ U{100..109}). Val 0.3356 / 0.3352 / 0.3327.
  - **(b) − (a) = −0.00135 ± 0.00082.** Making train look like val does not help count features. If anything it removes some useful training signal.
- **The NN's sequence encoder was not tested** (that needs a GPU).
- Consequences for decisions:
  - (i) The random-like train holdout has train-style deep histories, so it is not fully val-like. This supports the existing rule that "train-only val must improve".
  - (ii) Any history-position-dependent feature or encoder sees different semantics at hpos ≥ 128 in val/test: mostly old contact and message events, rather than views.

**Fix / test (optional, low priority).** A paired 2-3 epoch probe on the train stage:
- In `nn/data.py` (or a copy in `nn_x/`), drop train-split type-1 events beyond view-rank K ~ U{100..109}.
- Drop type 4 beyond about 117, and types 5/6/8/10 beyond about 41-44.
- Score on train-only val. Apply the same truncation to the holdout, so that the holdout becomes val-like.

Given the LightGBM null result, the expected gain is small. Do not spend a GPU slot on this before the higher-value round-4 work.

### INTEGRITY-2 (medium for understanding, info for private): the later-history structure in test is identical to val, so the zero public gain is on the label/scoring side

**Evidence**

Pair statistics (`pairs.py`, `pairs_agg.py`, `lhcheck.py`):

| pair type | A feeds | A feeds with ≥ 1 match in B's history | matches per A feed | matches that are type 1 / source 1 | matches in B's new part |
|---|---|---|---|---|---|
| te→te | 108 | 103 | 1.69 | 0.87 | 0.96 |
| val→val + val→test | 117 | 110 | 1.56 | 0.72-0.82 | 0.94-0.95 |

- On val, the precision of the match against val labels is 0.97 (va→va) and 0.94 (va→te).
- Test histories are live snapshots (B's new events at the front). Test→test B histories gain new events (median k = 11), and A's history is never identical to B's (hist_eq = 0).
- Nothing points to frozen or cut-off test histories.

The v5_lh boost was applied correctly:
- It touched only the matched rows: 183 rows, 97 feeds changed, 0 other feeds changed.
- Those rows moved from a mean rank of 30.0 (from the top) to 0.79.
- v4_lh is the same (32.1 → 0.79).

Expected public effect, simulated on val with nn_final2 OOF (the v5 base):
- Moving 2 negatives to the top costs **−0.079 per feed**, or **−0.0005** on public (57 / 8,562 feeds).
- Moving 1 positive to the top gains **+0.549 per feed**, or **+0.0037** on public.
- An exact 0.0000 change is consistent with only three explanations:
  - (a) a mix of about 13% positives among the boosted items (still about 8x the base rate);
  - (b) those feeds are not scored;
  - (c) the recorded 0.3802 is a leaderboard best-of rather than the submission's own score.

**Impact.** None on private decisions:
- Only 46 private feeds have matches (49 have a later test feed).
- Even at val precision, the maximum would be +0.0014 private.
- The rule stays as it is.

**Action (cheap).** Read v5_lh's own score from the platform's submissions table, not the leaderboard. This tells "neutral" (labels lack the signal) apart from "harmful" (test labels may be defined differently). It is worth knowing, but it does not change any current choice.

### INTEGRITY-3 (low, verified): `pre_id_contrib` / `pre2_id_contrib` / `prew_id_contrib` are computed differently on val and test

**Mechanism.** In `pretrain/assemble.py:28-46,55`:
- Val uses the v3A/v4A models, and fe and fe_noid are moment-matched with **separate** affine maps.
- For example, for v4A: fe a = 0.9981, b = 0.3910; fe_noid a = 0.9973, b = 0.4071.
- id_contrib = fe − fe_noid is therefore never exactly 0 on val.

**Evidence** (`idcontrib.py`):
- On val rows without an ID (raw fe == fe_noid; 54.8%), id_contrib = −0.0031 ± 0.0017, with correlation 0.980 to fe_noid.
- On test, rows without an ID (59.2%) have id_contrib exactly 0.
- Zero rates: val 0.000 vs test 0.592 (pre_ and pre2_). prew_: 0.0001 vs 0.2625.
- Single-feature val NDCG of pre_id_contrib: 0.3108 as built, against 0.3044 with test-like zeros (−0.0064). A rank-based learner (trees, the feature-eval harness) can use the micro-variation on val but not on test.

**Impact.**
- NN fine-tune (n2c_pre includes pre_id_contrib): after val standardisation (sd about 1.2), the spread is about 0.0014 sd. **Negligible.**
- Small optimism in GBDT and harness numbers for the pre_ family (the "+0.00445" family eval).

**Fix.** In `assemble.py`, compute id_contrib = a_fe·(raw_fe − raw_noid), or use the fe affine for both. Rebuild before any further tree or stacker use.

### INTEGRITY-4 (low, verified): `it_hist_*_luo` (train-stage input) is non-causal, so train rows see much denser "future" popularity than val/test rows

**Evidence** (`trainxt.py`; nonzero share of it_hist_all_luo):

| | share nonzero |
|---|---|
| train, Sep 6 | 0.903 |
| train, Sep 12 | 0.883 |
| train, Sep 17 | 0.834 |
| train, Sep 19 | 0.780 |
| val | 0.621 |
| test | 0.553 |

- Mean value: train 34.1, val 23.5, test 20.9.
- The decline across train days is the contribution of other users' later train snapshots.
- The single-feature lift is similar (train random-like M ≥ 50: +0.0177; val M ≥ 50: +0.0167). This is **not a label leak**: LUO removes the same user's later snapshots, and user 0 is checked below. But "count = 0" means something different, and train rows see about 3x fewer zeros.
- By contrast, seq_cv_* is causal (`feats_v2/seq_/sources.py`, `feat_covis.py`: expanding day cutoff for train rows, causal val/test histories, same user skipped).
- **prew_\*** is NaN where it_hist_all_luo == 0 on every split, but also NaN on all of Sep 6-8 (27.4% of train feeds, the window gap). So in train, has:prew_fe = 0 often co-occurs with it_hist_all_luo > 0, a combination that cannot occur in val/test.

**Impact.** Unknown, probably small. cvnd and cvnd_prew both passed paired train-only-val acceptance. Treat this as an improvement lever (see Ideas), not a bug.

### INTEGRITY-5 (low, verified): LUO keeps same-user snapshots taken in the same 128-s tick

- All timestamps are multiples of 128 s (`samets.py`: ts % 128 == 0 in every split).
- `src/features_luo.py:52` removes only `th > tc`.
- 323 same-user same-tick train groups (646 feeds, 2e-4 of train) keep each other's snapshot. 25% of these sibling pairs have identical histories, and their candidate sets differ (Jaccard median 0.035).
- **Negligible.** For hygiene, use `th >= tc` together with "different feed".

### INTEGRITY-6 (info, verified): zero user embeddings are 5x more common in val than in test

- Val: 205 feeds (1.42%); 64% have H > 0 and only 8% are user 0.
- Test: 80 feeds (0.28%); 71% have H = 0.
- Train: 2.56%.
- Val feeds with a zero user embedding score 0.370, against 0.394 for the rest (nn_final4 OOF).
- Effect on calibration: test should be about **+0.0003** better than the M-reweighted val prediction. Negligible. The current calibration is, if anything, slightly pessimistic.

### INTEGRITY-7 (info, verified): collection gaps

- **Val** has no feeds in 20:00-21:00 MSK on Tuesday (9 feeds), nor in 20:00-22:00 on Thursday (0 and 1).
- **Test** is missing Saturday 18h (4) and 20h (3), and Monday 18h (3).
- Val also has 1 feed on Sep 23 and 2 on Sep 25 00:0x MSK.
- `hour` is a fine-tune input (ITEM_CLEAN). Val never shows hour 20; test has 1,475 feeds at hour 20 (Sun-Tue). It is a continuous, feed-level input, so the risk is low.

### INTEGRITY-8 (info, verified): holdout composition vs val

Holdout pool (p_random > 0.9, M ≥ 50; the 30K holdout is drawn from it) compared with val:

| | holdout pool | val |
|---|---|---|
| mean M | 154.5 | 96.4 |
| H = 0 | 3.0% | 1.35% |
| user 0 | 5.2% | 4.6% |
| view count over the val/test cap | 24% | 0.4% |

- 9,359 of the 28,382 non-zero holdout users have 15,125 other feeds in training. Their later histories contain holdout positives, but only as inputs, never as targets, and viewed items are filtered from later candidates.
- This is not a leak of any consequence. The holdout is simply not a perfect val proxy, which the existing acceptance rule already handles.

### INTEGRITY-9 (info, verified): duplicates and degenerate feeds

- **Duplicate items within a feed:** train has 1 feed (fidx 1395589, M = 139) with 4 duplicated item_ids, all negatives, at different pos (f1 identical in 3 of 4). Val and test have 0. make_sub asserts unique (feed_id, item_id) on test.
- **Feed size:** M ≥ 5 in every split; no feed has M < 2.
- **Feed ids:** unique within each split and disjoint across splits. No duplicated feeds.

**Positives per feed:**

| | 0 | 1 | 2 | ≥ 3 | mean | positive rate |
|---|---|---|---|---|---|---|
| train | 0 | 66.6% | 18.6% | 14.9% | 1.729 | 2.21% |
| val | **0 (exact)** | 74.5% | 15.4% | 10.2% | 1.524 | 1.58% |

The "no positives scores 1" branch of the metric is therefore never exercised on val, and almost certainly not on test (the builder keeps only feeds with positives).

---

## Checked and correct

**Alignment and submission**
- **Feature lengths** (`align.py`): all 741 feature files (train 229, val 256, test 256) are 1-D float32 with exactly 246,745,836 / 1,389,971 / 2,904,806 rows. The val and test name sets are identical.
- **Prediction lengths:** all 637 1-D `preds/*.npy` match the val or test length. None is NaN. No train-length preds sit in preds/.
- **Feature row order** (`featparity.py`): within-feed correlation of each feature with nn_final4_trainonly, val vs test.
  - Max |Δ| = 0.035, median 0.004. No collapse on test, so no misaligned test feature file.
  - Distribution shifts are small (|Δmean|/sd ≤ 0.15) except `it_cand_cnt_split` (0.51). That one is a raw in-split count, already excluded from ITEM_CLEAN.
- **Prediction row order:** the same check over all 326 val/test prediction pairs gives max |Δcorr| = 0.016.
- **Flat files vs raw data** (`subcheck.py`): the flat test arrays equal the raw parquet sorted by (timestamp, feed_id) for feed_id, item_id, f1, history item ids and user_id.
- **Submission v10:**
  - `data/raw/sample_submission.parquet` equals `data/competition/sample_submission.parquet`.
  - v10 rows are in flat order and join to the sample's keys with 0 nulls. Keys are unique, scores are 0..M−1 per feed, 0 within-feed ties.
  - v10 is reproduced exactly (100% rank match) from 0.5 z(nn_final4_test) + 0.5 z(nn_final4_trainonly_test) + 0.1 z(pre_final_test).
  - The sample's row order differs from flat order, but the grader joins on keys: the public LB calibration matches to 0.0003-0.0014.

**History construction**
- **Ordering:** histories are newest-first in every split. A's candidates that appear in B are in B's new front part 81-96% of the time for every pair type.
- **Caps and timestamps:** H ≤ 256 everywhere (H = 256 share: train 0.423, val 0.290, test 0.288). Timestamps are 128-s quantised.
- **No events after the request time in any split:**
  - B's positives found in A's history: 0.13% (tr→tr), 0.24% (tr→va), 0 of 25 (va→va), at base-rate precision (0.027 / 0.034).
  - B candidates in A history: te→te 0.04%, va→te 0.07%, versus 0.11% for tr→tr.
- **Candidates never appear in their own history** (`ownhist.py`): val 0 / 1.39M, test 0 / 2.90M, train 1 / 23.5M (a 300K-feed sample; that one is a positive).
- **Test histories include the train period:** train positives appear in later test histories with recall 0.61 and precision 0.74, and test→test histories gain new events. There is no frozen cut-off.

**Feature construction**
- seq_cv_* is causal and leaves out the same user.
- ctx_* (`feats_v2/ctx_/build_ctx*.py`) is computed inside the feed only: candidates, history, user embedding, f1. No labels.
- The fine-tune extra branch standardises with val statistics and reuses them for test (`nn/finetune.py:166-167`). Folds are GroupKFold by user_id.
- **Holdout:** `nn/train.py:159-163` and `nn/holdout_eval.py` select the same 30K feeds, and `holdout_meta.npy` / `holdout_y.npy` match a recomputation. p_random uses features and pos only, no labels (eda_labels_position 2.5). The holdout is excluded from the cooldown pool (`train.py:256`).
- **No user-0 residual LUO leak** (`user0leak.py`):
  - it_hist_all_luo is more predictive for user-0 feeds on val too (lift 0.0379 vs 0.0218 for non-zero users), where no leak is possible. So the larger user-0 lift in train (0.0237 vs 0.0052) is a population effect.
  - Removing later user-0 snapshot counts moves the train user-0 lift only from 0.0237 to 0.0224.
- No nn/ preset in use contains x_lh_*, non-LUO it_hist_*, or later-history flags. n2c_pre is ITEM_CLEAN + COVIS + SEQ + CTX + PRE.

**Train-stage inputs (cvnd_prew) on train rows** (`trainxt.py`)
- Every feature carries a train-sample NDCG lift of the same sign and similar size as on val M ≥ 50: it_hist_all_luo +0.018 / +0.017, ctx_hs_t10 +0.049 / +0.056, ctx_hn_bestrk −0.029 / −0.028. So the train feature files are row-aligned too.
- prew_* lift is lower on train (+0.024 vs +0.048) because Sep 6-8 rows are NaN.

---

## Ideas (from this audit)

1. **Causal it_hist_\*_luo for the train stage.** Count only other users' snapshots with ts < the feed ts, or use the day cutoff as for seq_cv_*. Keep the val/test definition causal too, then test it paired on train-only val. This removes the 0.90 vs 0.62 density mismatch in INTEGRITY-4. Of the levers here, it has the best chance of a small train-stage gain.
2. **prew_\* window gap.** Exclude Sep 6-8 feeds from the prew run, or use a separate availability flag so that "window gap" is not encoded like "cold item".
3. **Val-like view truncation (INTEGRITY-1)** for train and holdout histories, as one paired probe. Low priority given the LightGBM null result (−0.0014 ± 0.0008). A cheaper hedge is to expose view-rank (rank among type-1 events) as an input, so the encoder can tell the train-only deep views apart.
4. **Fix `pretrain/assemble.py` id_contrib** (one line) before any further GBDT, stacker or harness use of pre_*/pre2_*.
5. **Read v5_lh's own submission score** to settle neutral vs harmful (INTEGRITY-2). This is diagnostic only.
6. **Calibration sanity:** the val/test difference in zero-user-embedding share implies about +0.0003 in test's favour. No action needed beyond knowing the predictions are slightly pessimistic.
