# EDA: train/val/test distribution shift, user/item overlap, public vs private split

Scripts: `<repo>/eda/shift_overlap/` (`s01`..`s10b`, `featlib.py`, `common.py`). Raw logs: `<repo>/eda/shift_overlap/out/*.log`.
All NDCG numbers are full-list NDCG on val computed with `src/metrics.py` (pessimistic ties) unless stated. "Random tie-break" means a tiny shared noise was added to the score before evaluation. Val has 14,423 feeds, so the SE of a mean NDCG is about 0.0025; the SE of a paired per-feed difference is about 0.0005-0.0012 (given per row).

---

## 0. TL;DR

1. **Val and test match closely; train is the odd one out.** For every marginal (f1..f5, category, M, H, event type/source), the val-vs-test Jensen-Shannon divergence is 0.00001-0.0013. Train vs val/test is 0.003-0.016. Adversarial LightGBM on intrinsic features gives AUC **0.524 val-vs-test at feed level and 0.512 at candidate level**. Train-vs-val gives **0.925 at feed level and 0.643 at candidate level**. The train shift is **not temporal**: train per-day means are flat across the 12 effective days of Sep 6-19. It reflects the different traffic type (production-ranked feeds sampled per impression vs exploration feeds sampled roughly once per user).
2. **The time gap since train is the main val→test shift, and it acts through item coverage.** The share of candidate rows whose item was a train candidate falls by about 2 pp per day: val Tue 69.1%, val Thu 61.4%, test Sat 56.9%, Sun 54.7%, Mon 52.9%, Tue 51.4%. Unseen items (not in train candidates or history) make up **28.9% of val rows vs 39.3% of test rows**. Adversarial AUC val-vs-test jumps from 0.52 to **0.91** once you add "share of candidates seen in train". Any train-derived item statistic therefore has lower coverage on test, and on private in particular, than on val.
3. **Public/private boundary: 2026-09-27 11:32 MSK** (ts = 1790497920; 8,562 public feeds = 29.94%; 18 feeds share the boundary 128-s tick). Public is all of Saturday Sep 26 plus Sunday morning, so it is 100% weekend. Private is Sunday afternoon plus Monday and Tuesday (24% weekend). **Val (Tue + Thu) is weekday-only and resembles private more than public** (feed-level adversarial AUC: val vs public 0.533, val vs private 0.521). Within test, public vs private is close to indistinguishable on intrinsic features (AUC 0.512 feed-level, 0.525 candidate-level). It becomes separable (AUC 0.657) only through train-overlap features: train-candidate coverage is 56.4% in public vs 52.7% in private.
4. **Users barely overlap across splits, and almost never within a split.** 31.8% of val users and 30.3% of test users appear in train. Only 92 test users (0.34%) appear in val. Excluding `user_id = 0`, val/test have about **1.00 feeds per user** (25 val users and 105 test users have 2 or more feeds). `user_id = 0` is a pooled "missing user" bucket: 3.9-4.6% of feeds in every split, H mean about 55 vs 180, and 123,904 train feeds. It must never be treated as one user.
5. **Items in val/test are almost all singletons inside the split.** 91% (val) and 86% (test) of candidate items appear in exactly one feed of the split. The most frequent item appears in only 29 / 60 feeds. In-split candidate frequency has **no value** (−0.0018 when blended with f1). In-split *history* frequency (the item appears in other users' val history) gives a small gain: pos rate 2.05% vs 1.54%, **+0.0014..0.0020** blended with f1 (about 2 SE). Train history counts, especially event_type 1, give the most of any item-popularity signal: **+0.0032..0.0035 (about 3 SE)**. This gain is robust to a 15-day gap. Train candidate counts and train CTR give **0 or less**, because they are position-biased.
6. **Cross-feed label leak exists but is tiny on test.** In train, **92.0% of a feed's positives reappear in the same user's next-feed history** (79.9% with event_type 1), vs 0.40% of negatives. On val it is 88% (37/42). Test has only 108 feeds (0.38%; 59 public, 49 private) with a later feed from the same non-zero user. The val oracle raises those feeds' NDCG from 0.300 to 0.984, which implies about **+0.0026 on test-all, +0.0047 on public and +0.0017 on private**. Useful to know (it also tells us the target action is close to event_type 1 in history), but it cannot decide the private LB.
7. **Expected metric level from feed-size composition.** Test feeds are larger (mean M 101.6 vs 96.4) and NDCG falls with M. Reweighting val's per-M-bucket NDCG to test's M distribution predicts the same model scores **about 0.006 lower on test than on val** (f1: val 0.3362 → test 0.3304; public 0.3285, private 0.3312). LB-vs-val gaps of that size are expected and are not a sign of overfitting.
8. **The feature-label relation differs between train and val; f4 flips sign.** In train, the lowest f4 decile has the *highest* positive rate (1.22× mean); in val it has the *lowest* (0.46×). Single-feature f4 NDCG vs random is −0.006 on train but +0.009 on val, and −f4 costs −0.022 on val. f1's lift is larger on val than on train (+0.048 vs +0.036). Train has strong position bias: positive rate is 6.26% at pos 0-9 vs 0.99% at pos 150+. Production order alone scores 0.3634 on train, about the same as f1 (0.3619). **The f4 inversion is a position artefact.** At pos ≥ 50 in train, the lowest f4 decile converts at 0.53×, matching val. A plain LGB fit on train (f1..f5 + category) scores **below f1 alone on val (0.3301 vs 0.3365)**. The same model with `pos` as a feature, inferred at a deep constant position (pos = 250), scores **0.3400 (+0.0099 vs plain, +0.0036 ± 0.0011 vs f1)**.
9. **Ties matter.** f1 puts 17.6% of val candidates on its minimum value (1/56). The pessimistic-tie metric then gives f1 **0.3279 vs 0.3364 with random tie-break** (−0.0085). Every submitted score must be tie-free, for example with a tiny secondary key.

---

## 1. Time structure

Timestamps are Unix seconds. Day boundaries align with **Moscow time (UTC+3)**: the first train feed is 2026-09-05 21:00:48 UTC = Sep 6 00:00:48 MSK. All dates and hours below are MSK.

| split | days with traffic (feeds) | notes |
|---|---|---|
| train | Sep 6-14 (203K-292K/day), **Sep 15: 42 feeds, Sep 16: 0**, Sep 17-19 (233K-269K/day), 34 feeds at Sep 20 00:00 | 12 effective days; Tue-Wed Sep 15-16 missing |
| val | **Tue Sep 22: 7,542; Wed Sep 23: 1; Thu Sep 24: 6,878**; 2 feeds at Sep 25 00:00 | effectively 2 days, both weekdays |
| test | Sat Sep 26: 6,050; Sun 27: 7,367; Mon 28: 7,578; Tue 29: 7,600 | 4 days, Sat/Sun included |

- Within val/test, several **hour-long outages** have 0-9 feeds instead of about 400: val Sep 22 20h, Sep 24 04h and 20-21h; test Sep 26 10h, 18h, 20h, Sep 27 12h, Sep 28 18h, Sep 29 14h. They distort hour-of-day histograms (JS(hour) val-test 0.035 vs train-test 0.005). They are not a behavioural shift.
- The hourly profile is otherwise the same in all splits: a trough at 02-04h (about 1.1% of feeds per hour) and a plateau at 09-22h (5-6% per hour).

## 2. Distribution comparison

### 2.1 Feed size M and history length H

| | train | val | test |
|---|---|---|---|
| M mean / median | 78.3 / 48 | 96.4 / 67 | 101.6 / 72 |
| M q10 / q90 | 9 / 213 | 12 / 247 | 13 / 256 |
| H mean | 185.2 | 173.3 | 173.9 |
| H = 256 (capped) | 42.3% | 29.0% | 28.8% |
| H = 0 | 1.11% | 1.35% | 0.86% |
| positive rate | 2.207% | 1.582% | — |
| positives / feed | 1.72 (train sample) | 1.52 | — |

JS(M, 10-wide bins): train-val 0.0154, train-test 0.0209, **val-test 0.0013**. JS(H): 0.0151 / 0.0164 / **0.0006**.
Train feeds come from heavier users (more H=256). This fits per-impression sampling in train vs roughly one feed per user in val/test.
M rises slightly from val to test (96.4 → 101.6). Even test weekdays are higher than val weekdays (Mon/Tue 99.6/100.0 vs Tue/Thu 96.5/96.2).

### 2.2 f1..f5

All five features are interpolated bins clipped to the train range: f1 ∈ [1/56, 55/56], f3 ∈ [1/62, 61/62], f4 ∈ [1/64, 63/64], and f2/f5 have an extra exact-0 value. **No value can fall outside the train range.** Drift therefore shows up only as moved mass.

| feature | train mean | val mean | test mean | special mass (train / val / test) | PSI train-val | PSI train-test | PSI val-test |
|---|---|---|---|---|---|---|---|
| f1 | 0.443 | 0.400 | 0.397 | at min 1/56: 15.4 / 17.6 / 17.4% | 0.026 | 0.029 | 0.001 |
| f2 | 0.443 | 0.440 | 0.440 | exact 0: 11.9 / 15.4 / 15.0% | 0.025 | 0.020 | 0.005 |
| f3 | 0.488 | 0.451 | 0.447 | at min: 5.3 / 5.6 / 5.4% | 0.035 | 0.036 | 0.007 |
| f4 | 0.504 | 0.505 | 0.508 | at min: 1.2 / 1.1 / 0.7% | 0.017 | 0.021 | 0.005 |
| f5 | 0.480 | 0.442 | 0.444 | exact 0: 1.4 / 1.4 / 1.1% | 0.041 | 0.037 | 0.001 |

Per-day means show the shift is a level difference between traffic types, not a trend:
- train f1 stays in 0.432-0.455 on every day from Sep 6 to Sep 19;
- val is 0.397 / 0.402;
- test is 0.399 / 0.398 / 0.391 / 0.399.
f3 and f5 behave the same way. All PSIs are below 0.05 (conventionally "no significant shift"), but in multivariate form they separate train from val easily (§5).

### 2.3 Categories

- Candidate category JS: train-val 0.0032, train-test 0.0037, **val-test 0.0002**.
- History category JS is at most 0.0004 between any two splits, so history is stable.
- All splits use 52 candidate categories.
- The biggest movers between train and val/test, in share of candidates:
  - cat 48: 4.08% → 2.69 / 2.65%
  - cat 2: 9.60 → 11.18 / 11.26%
  - cat 50: 7.78 → 8.95 / 8.86%
  - cat 11: 2.55 → 3.09 / 3.16%
  - cat 49: 0.11 → 0.28 / 0.31%
  - cat 27: 1.81 → 1.49 / 1.52%
- Category positive rates correlate **r = 0.851** between train and val (49 categories with ≥2,000 val candidates). The ordering is broadly preserved, but levels differ (train rate 2.2% vs val 1.6%).

### 2.4 History event type / source

| event_type share (all history) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 51.9 | 11.6 | 10.8 | 9.3 | 4.4 | 4.1 | 3.7 | 2.6 | 1.2 | 0.44 |
| val | 43.9 | 13.9 | 12.0 | 10.7 | 5.3 | 4.7 | 4.2 | 3.2 | 1.5 | 0.53 |
| test | 44.1 | 13.8 | 12.0 | 10.7 | 5.3 | 4.7 | 4.3 | 3.2 | 1.5 | 0.49 |

- JS(event_type): train-val 0.0048, **val-test 0.00001**.
- Restricted to the 32 most recent events, the mixes match: event_type 1 is 56.9% train vs 58.5% val vs 58.7% test. The all-history gap is therefore a user-mix / H effect: heavy train users have long, view-dominated histories.
- event_source 0 is 39.3% train vs 45.4% / 45.2% val/test; source 1 is 37.6% vs 33.8% / 34.3%. JS(source) val-test is 0.00004.
- Per-feed **shares** (normalised by H) are safer than raw counts.

### 2.5 Missing values

| | train | val | test |
|---|---|---|---|
| candidate emb_idx = -1 | 0.003% | 0.015% | 0.016% |
| history emb_idx = -1 | 15.5% | 14.2% | 14.0% |
| user_embedding all-zero | **2.53%** | **1.42%** | **0.28%** |
| category 0 (cand / hist) | 0 / 0.003% | 0 / 0.003% | 0 / 0.003% |

- The all-zero user embedding is the one clear val→test shift among missing-value flags.
- It is not explained by user 0 or by H = 0. Feeds with an all-zero embedding that are neither user 0 nor H = 0 make up 2.23% / 0.87% / 0.08% of all feeds. In test it varies by day: 0.12, 0.20, 0.67 and 0.09%.
- **Do not let a model rely on "user embedding missing"**: its prevalence changes 5×.

## 3. Users

| | val | test |
|---|---|---|
| users (feeds) | 13,741 (14,423) | 27,309 (28,595) |
| users present in train (excl. user 0) | 31.80% | 30.25% |
| feeds whose user is in train (excl. user 0) | 31.86% | 30.38% |
| users present in val | — | 92 users (0.34%) |
| feeds/user excl. user 0 | 1.0018 (25 users with 2 feeds) | 1.0040 (105 users with ≥2) |
| `user_id = 0` feeds | 658 (4.56%) | 1,179 (4.12%) |

- **Train history depth of test users.**
  - 66.8% of test feeds belong to users with 0 train feeds;
  - 18.8% to users with 1 train feed, 9.5% with 2-4, and 0.8% with 5-9;
  - the 4.2% with "10+" are almost entirely `user_id = 0`.
  - Train candidates per test feed: median 0, q75 36, q90 229.
  - **User-level target statistics from train are therefore unavailable for about 70% of test feeds.** User information has to come from the in-feed history and the user embedding.
- **user 0.**
  - Shares by split: train 3.93%, val 4.56%, test 4.12%.
  - Mean H is 54-60 vs 179-190 for other users; H = 0 for 15-19% of them; M is larger (val 106 vs 96).
  - It is a mix of anonymous users. f1 NDCG on val is 0.3313 (lift over random +0.058, the largest of any segment, because random is low at 0.274).
- **Train users.** 17.2% of non-zero users have more than one feed, and those feeds are 32.8% of train. Gaps between consecutive feeds: median 68.7 h, 25% ≤ 24 h, 3.1% ≤ 1 h.
- **Repeat users in val/test (excl. user 0).** Gaps between consecutive feeds: median 13.1 h in val (n = 25) and 23.2 h in test (n = 108). Only 0-1% of gaps are ≤ 10 min. There are no burst sessions.
- **Same (user, item) re-impressions.**
  - Excluding user 0, 0.19% of val and 0.13% of test candidate rows were shown to the same user in train. That is 0.58% / 0.42% among users present in train.
  - Their val positive rate is 1.65% vs 1.59%, which carries no signal; adding ±1 for re-impression changes NDCG by −0.0005..+0.0005.
  - Only 0.04-0.05% of candidates appear in the same user's *train history*, consistent with the documented "already-viewed items removed".
- **Segments in val.** f1 NDCG barely differs between known users (0.3349) and new users (0.3379). Segment shares are about the same in val, public and private: known 30.4 / 30.3 / 28.6%, new 65.0 / 65.6 / 67.2%.

## 4. Items

### 4.1 Overlap / cold start (row-weighted; unique-item-weighted values are 0-4 pp lower)

| candidate item seen in ... | val rows | test rows |
|---|---|---|
| train candidates | 65.4% | **53.8%** |
| train history | 58.7% | 51.5% |
| train candidates or history | 71.1% | 60.7% |
| train positives (target = 1 at least once) | 19.7% | 15.3% |
| val candidates | — | 12.7% |
| val history | — | 6.7% |
| train or val (cand or hist) | — | 63.6% |
| same-split history of any feed (transductive) | 8.5% | 14.1% |
| **cold**: none of the above (val: not in train or val history) | **28.1%** | **34.1%** |

Unique item counts:
- train: 38.95M candidate items and 94.7M history items;
- val: 1.24M candidate items, 1.58M history items;
- test: 2.38M candidate items, 3.01M history items.

History items are much better covered: 84.1% (val) and 79.4% (test) of history rows hit train history, 87.8% / 83.3% hit any train source. Item-embedding / history-similarity features are fully available for cold items (candidate emb missing is only 0.015%). Only ID-based statistics suffer.

### 4.2 Coverage decay with the gap since train end (Sep 19)

| day | gap (days) | in train cand | in train cand or hist | seen on the last train day |
|---|---|---|---|---|
| val Tue 22 | 3 | 69.1% | 74.5% | 52.5% |
| val Thu 24 | 5 | 61.4% | 67.5% | 45.9% |
| test Sat 26 (public) | 7 | 56.9% | 63.5% | 42.2% |
| test Sun 27 (public/private) | 8 | 54.7% | 61.5% | 40.4% |
| test Mon 28 (private) | 9 | 52.9% | 59.9% | 38.9% |
| test Tue 29 (private) | 10 | 51.4% | 58.5% | 38.0% |

Coverage falls about 2 pp per day (item churn). Train-candidate coverage is 56.4% in public vs 52.7% in private.

### 4.3 In-split item frequency

| | train | val | test |
|---|---|---|---|
| candidate rows per unique item | 6.33 | 1.12 | 1.22 |
| items in exactly 1 feed | 42.5% | 91.0% | 85.8% |
| q99 / max feeds per item | 69 / 9,374 | 3 / 29 | 4 / 60 |
| rows with in-split count = 1 / 2-4 / 5-19 / 20+ | — | 81.1 / 17.4 / 1.5 / 0.0% | 70.4 / 25.3 / 4.2 / 0.1% |
| (count − 1) per 10K feeds, per row | — | 0.22 | 0.23 |
| same-split history count per 10K feeds, per row | — | 0.115 | 0.117 |

Raw transductive counts scale with split size: test has 2× val's feeds. After per-10K-feed normalisation they are consistent between val and test. **Raw counts computed on val must not be fed to a model scored on test.** Normalise by the number of feeds, or use within-feed ranks.

### 4.4 Target rate by item status (val)

| status | share of rows | pos rate |
|---|---|---|
| seen in train candidates | 65.4% | 1.694% |
| not in train candidates | 34.6% | 1.368% |
| seen in train history | 58.7% | 1.769% |
| not in train candidates or history | 28.9% | 1.247% |
| in val history of other users | 8.5% | **2.054%** |
| not in val history | 91.5% | 1.538% |
| in-split candidate count ≥ 5 | 1.5% | 1.245% |

Cold items convert worse (1.25% vs 1.69-1.77%). Test has more cold items, so the test base rate is probably slightly lower than val's, which is irrelevant for ranking. It also means "warm" flags carry less weight on test.

### 4.5 Value of popularity / overlap signals on val (fair baseline)

Baseline: f1 with random tie-break, **0.3366**. Blend: within-feed percentile rank of f1 + w × percentile rank of the signal, with the same tie-break noise. Δ is the paired per-feed difference vs the baseline; its SE is 0.0005-0.0010. A pure random signal in the blend costs −0.0017 / −0.0049 / −0.0098 at w = 0.1 / 0.25 / 0.5, which shows what "no information" looks like.

| signal | alone, pess. ties | alone, random tie | Δ w=0.1 | Δ w=0.25 | Δ w=0.5 |
|---|---|---|---|---|---|
| CONTROL: random | 0.2895 | 0.2895 | −0.0017 | −0.0049 | −0.0098 |
| val candidate count (in-split) | 0.2395 | 0.2869 | −0.0007 | −0.0018 | −0.0046 |
| val history count (in-split, all events) | 0.2399 | 0.3001 | +0.0011 | +0.0014 | **+0.0020** |
| val history count, event_type 1 only | 0.2329 | 0.2986 | +0.0010 | +0.0015 | +0.0019 |
| train candidate count | 0.2975 | 0.3031 | +0.0014 | +0.0005 | −0.0005 |
| **train history count** | 0.3027 | **0.3100** | +0.0019 | **+0.0032** | +0.0031 |
| train history count, event_type 1 | 0.3020 | 0.3109 | +0.0020 | **+0.0035** | +0.0034 |
| train history count, event_type 2 / 3 / 4 | — | 0.303 / 0.302 / 0.302 | +0.0014 / +0.0013 / +0.0008 | +0.0021 / +0.0019 / +0.0021 | +0.0017 / +0.0002 / +0.0011 |
| train positive count | 0.2632 | 0.3049 | +0.0017 | +0.0019 | +0.0015 |
| train CTR (pos+1)/(cnt+45) | 0.2816 | 0.2975 | −0.0002 | −0.0016 | −0.0044 |
| seen-in-train flag | 0.2148 | 0.2961 | +0.0002 | +0.0001 | +0.0005 |

Notes:
- Count signals have huge tie groups, so on their own their pessimistic NDCG falls **below random** (0.21-0.24). Use them only as model features, never as the final key.
- Train CTR and train candidate count are hurt by **position bias**: production impressions concentrate on already-top-ranked items. Train *history* counts carry organic interest (views from all surfaces) and transfer better.
- These are rank blends. A GBDT/NN using the raw counts plus coverage flags will extract more, but the ordering of signals should hold.

### 4.6 Does the train-statistic value survive the larger test gap?

The test gap is 7-10 days vs val's 3-5 days. To simulate it, train counts were recomputed with data truncated at earlier cutoffs, and also on a fixed 3-day window (equal volume) placed at different gaps (`s03c`):

| train history count from ... | gap to val | coverage | Δ (w=0.25) |
|---|---|---|---|
| all train ≤ Sep 19 | 3 d | 58.7% | +0.0032 |
| all train ≤ Sep 13 | 9 d | 47.7% | +0.0031 |
| all train ≤ Sep 9 | 13 d | 39.9% | +0.0028 |
| all train ≤ Sep 7 | 15 d | 32.6% | +0.0030 |
| 3-day window Sep 17-19 | 3 d | 48.8% | +0.0029 |
| 3-day window Sep 12-14 | 8 d | 40.2% | +0.0025 |
| 3-day window Sep 6-8 | 14 d | 36.9% | +0.0029 |

Coverage falls by about a third, but the lift stays within noise. **Train-history item popularity is a slow-moving, robust signal.** Train-candidate counts are about 0 at every gap, and negative (−0.0014..−0.0029) at w = 0.5 for gaps ≥ 9 days.

## 5. Adversarial validation (LightGBM, 5-fold CV, AUC)

Feed-level features: M, H, user0 flag, user-embedding stats, history event-type/source shares, distinct categories, per-feed mean/std of f1..f5, f2-zero share, f1-at-min share and emb-missing share. "Overlap" features add the share of candidates in train candidates/history, the mean log train-history count, the history-in-train share and user-in-train. Candidate level uses feeds as CV groups and 600K rows per side. The train side is 30K random feeds from Sep 17-19.

| comparison | feed: intrinsic | feed: + overlap | feed: + hour | feed: user_emb 128-d | cand: f1..f5 + cat | cand: + M | cand: item-emb 16-d | cand: + train counts |
|---|---|---|---|---|---|---|---|---|
| **val vs test** | **0.524** | **0.909** | 0.577 | 0.516 | **0.512** | 0.512 | 0.501 | 0.630 |
| train vs val | 0.925 | (1.0)* | 0.930 | 0.513 | 0.643 | 0.669 | 0.552 | (0.858)* |
| train vs test | 0.928 | (1.0)* | 0.928 | 0.531 | 0.647 | 0.677 | 0.557 | (0.907)* |
| **test public vs private** | **0.512** | 0.657 | — | 0.510 | **0.525** | — | — | 0.558 |
| val vs test-public | 0.533 | | | | | | | |
| val vs test-private | **0.521** | | | | | | | |

\* Degenerate: the train side's own candidates are included in the train counts.

- **val vs test.**
  - Top single features: number of distinct candidate categories (AUC 0.522), M (0.517) and H-mix (0.51).
  - With overlap features, `c_in_train_cand` alone takes 37% of the gain: test feeds have systematically lower train coverage.
  - "+hour" (0.577) reflects the outage-hour gaps and the weekend days in test, not user behaviour.
- **train vs val/test.** The shift is in the *within-feed shape* of features, not their marginals. The strongest single features are the per-feed std of f5 (AUC 0.645), the per-feed mean of f3 (0.340), the share of f1 at its minimum (0.629), the share of f2 = 0 (0.625), H, and the event_type-1 share. Compared with val, production feeds have higher mean f1/f3, fewer candidates at the f1 floor or at f2 = 0, a *lower* within-feed std of f5 but a *higher* std of f4, and fewer distinct categories per feed (AUC 0.594, partly through smaller M). This fits production retrieval/ranking pre-selecting candidates. Exploration feeds look like broader retrieval.
- **User and item embeddings are stable across all splits** (AUC 0.50-0.56).
- **public vs private.** On intrinsic features there is almost nothing to separate them (0.512 / 0.525). The candidate-level 0.525 comes from f3 and category, plausibly weekend vs weekday mix. Single-feature AUC of train-candidate coverage is 0.600.

### 5.1 Public/private details

| | public (ts < 1790497920) | private |
|---|---|---|
| period (MSK) | Sat Sep 26 00:00 → Sun Sep 27 11:32 | Sun Sep 27 11:32 → Tue Sep 29 24:00 |
| feeds | 8,562 (29.94%) | 20,033 |
| weekend share | 100% | 24.2% |
| M mean / median | 103.6 / 73 | 100.7 / 71 |
| H = 256 | 29.4% | 28.6% |
| user 0 | 4.10% | 4.13% |
| feeds of users in train (excl. 0) | 31.5% | 29.9% |
| candidate rows seen in train candidates | 56.4% | 52.7% |
| hour mix 00-12 / 12-18 / 18-24 | 55.7 / 26.2 / 18.0% | 30.2 / 32.9 / 36.9% |

- Users in both public and private: 42 (0.15%).
- JS(category) public-private 0.0004; val-public 0.00044; val-private 0.00023. JS(M) public-private 0.0012; val-public 0.0024; val-private 0.0011.
- **Private is closer to val than public is on M, category and weekday/weekend.** Its main difference from val is lower train coverage (val 65%, private 53%).
- The exact boundary is ambiguous only for the 18 feeds at ts = 1790497920.

### 5.2 Expected NDCG level by composition

Per-feed NDCG depends strongly on M:
- M in [5, 10): random 0.541, f1 0.624;
- M in [250, 300]: random 0.193, f1 0.228.

Reweighting val's per-M-bucket NDCG to each target's M distribution:

| target | random | f1 |
|---|---|---|
| val | 0.2887 | 0.3362 |
| test public | 0.2820 | 0.3285 |
| test private | 0.2843 | 0.3312 |
| test all | 0.2836 | 0.3304 |

- **Expect LB ≈ val − 0.005..0.008 for the same model.** Public should sit about 0.003 below private from M composition alone.
- Hour also matters through composition: val f1 NDCG is 0.3236 at 18-24h vs 0.3440 at 12-18h, with random scores showing the same pattern. Private is heavier on 18-24h (36.9% vs val 25.1%).
- Val day-to-day: Tue 0.3380 vs Thu 0.3343 (diff 0.0037, about 1 SE). In val, the first 30% vs last 70% by time is 0.3429 vs 0.3334 (diff 0.0095, about 2.4 SE).
- **No weekend effect in the feature-label relation** (train, 20K feeds/day, `s08`). f1 lift over random is +0.0386 / +0.0416 / +0.0414 / +0.0371 on the four weekend days (Sun 6, Sat 12, Sun 13, Sat 19) vs +0.0352..+0.0411 on weekdays. Positives per feed are 1.69-1.76 on all days. A weekend-only public is therefore mainly a *composition* difference (M, hours, item coverage), not a different relation.

## 6. Cross-feed leakage via later-feed history

- **Train.** 150K consecutive feed pairs of the same non-zero user were sampled (median gap 68 h). Item i from earlier feed A appears in the history of later feed B:
  - for **91.98% of positives** vs 0.40% of negatives;
  - by gap: ≤ 1 h 96.1%, 1-24 h 96.8%, > 24 h 90.3%.
- **Event types of those reappearances, per 100 positives vs per 100 negatives.**
  - et 1: 79.9 vs 0.335 (P(target | et 1 in B) = 85%);
  - et 2: 9.4; et 3: 8.6; et 4: 4.0; et 5: 3.6; et 6: 3.4; et 7: 1.0; et 8: 1.1; et 9: 0.2; et 10: 0.01.
  - The **target action is (or is almost always accompanied by) a history event of type 1**. Types 2-8 also reappear far more often for positives (P(target) 75-88% when present) and are probably follow-up actions.
- **Val.** 25 feeds have a later feed from the same non-zero user. 37/42 positives (88%) vs 0.03% of negatives reappear. Ranking flagged items first lifts those 25 feeds from 0.300 to 0.984 NDCG, +0.0012 on val overall.
- **Test.** 108 feeds (0.38%; 59 public / 49 private) can be "leaked", with 183 flagged candidates. At the val per-feed effect that is about **+0.0026 test-all / +0.0047 public / +0.0017 private**.
  - This is a property of the provided data (transductive use of later test histories).
  - It is too small to matter for private, and whether it is allowed is a rules question. Do not let it inflate public-LB-based decisions: its public effect is about 2.8× its private effect.
- **Validation-only leak.** 94 val feeds belong to users who also have test feeds; their test histories contain the val outcomes. Never use test history to build val features.

## 7. Train (production) vs val (exploration): label-feature relation and normalisation

(`s07`; logs `out/s07_desc.log` and `out/s07_norm.log`. Train sample: 60K random feeds from Sep 17-19, 4.68M candidates. NDCG uses random tie-break, mean of 3 seeds.)

### 7.1 Single-feature NDCG: train vs val

| score | train | val | train lift vs random | val lift vs random |
|---|---|---|---|---|
| random | 0.3256 | 0.2881 | — | — |
| +f1 / −f1 | 0.3619 / 0.2873 | 0.3365 / 0.2539 | +0.036 | **+0.048** |
| +f2 / −f2 | 0.3252 / 0.3265 | 0.2911 / 0.2939 | −0.000 (−f2: +0.001) | +0.003 (−f2: +0.006) |
| +f3 / −f3 | 0.3499 / 0.2882 | 0.3177 / 0.2531 | +0.024 | +0.030 |
| **+f4 / −f4** | **0.3194 / 0.3274** | **0.2974 / 0.2666** | **−0.006 (−f4 is better)** | **+0.009 (−f4 costs −0.022)** |
| +f5 / −f5 | 0.3417 / 0.3012 | 0.2981 / 0.2632 | +0.016 | +0.010 |
| production order (−pos) | 0.3634 | n/a | +0.038 | — |

**f4 reverses sign between train and val.** Positive rate by val-defined f4 decile, as a ratio to the split mean:

| f4 decile | 1 (lowest) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | **1.22** | 1.12 | 0.97 | 0.95 | 0.94 | 0.94 | 0.93 | 0.97 | 0.97 | 0.93 |
| val | **0.46** | 0.89 | 0.97 | 1.05 | 1.13 | 1.11 | 1.12 | 1.08 | 1.13 | 1.06 |

- The lowest-f4 decile is the *best* decile in production traffic but by far the *worst* in exploration traffic. This is classic selection bias: in production a low-f4 item reaches the feed or its top only when other signals are strong.
- A model fit on train labels will learn an f4 shape that is wrong for val/test.
- f1 and f3 keep their monotone shape (f1 deciles 2-10: train 0.45→1.57, val 0.54→1.67), but the train curve is steeper at the bottom.
- f5 is relatively stronger in train (+0.016 vs +0.010).

### 7.2 Position bias and production ranking in train

- Positive rate by `pos` bucket (0-9, 10-19, ...): 6.26, 3.54, 2.68, 2.22, 1.96, 1.76, 1.62, 1.50, 1.41, 1.38, 1.26, 1.23, 1.12, 1.12, 0.98, 0.99%. Top-10 positions convert **6.3×** better than position 150.
- The production order (−pos) reaches only 0.3634 NDCG, about the same as f1 alone (0.3619). The within-feed Spearman correlation between pos and f1 is weak: mean −0.077, q10/q50/q90 = −0.25 / −0.07 / +0.08, and 51% of feeds have |ρ| < 0.1. **The production ranker does not use f1**, which explains why f1 works as an independent signal.
- `pos` has gaps from filtering: M / (max pos + 1) averages 0.893, and only 13.3% of train feeds are gap-free.

### 7.3 Within-feed normalisation

Candidate-level adversarial AUC on f1..f5 only (200K rows per side, feed-grouped 5-fold):

| representation | train vs val | val vs test |
|---|---|---|
| raw | 0.626 | 0.511 |
| within-feed percentile rank | **0.595** | 0.507 |
| within-feed z-score | 0.616 | 0.500 |

Normalisation removes part of the train→val covariate shift, but not all of it: the *joint* within-feed structure also differs. It makes val vs test essentially identical.

Effect on ranking quality (val NDCG, random tie-break; f1 alone = 0.3365):

| model | raw | rank only | z only | raw + rank |
|---|---|---|---|---|
| pointwise LGB, 5-fold CV *within val* (f1..f5) | 0.3337 | 0.3281 (−0.0055 ± 0.0016) | 0.3354 (+0.0017 ± 0.0014) | 0.3317 (−0.0020 ± 0.0016) |
| pointwise LGB *trained on train* (60K feeds; f1..f5 + category) → val | **0.3250** | 0.3240 (−0.0010 ± 0.0014) | — | **0.3296 (+0.0046 ± 0.0013)** |

- **A model fit on production-ranked train data scores below f1 alone on val** (0.3250 vs 0.3365). The train-label relation (position bias, f4 flip) does not transfer. Adding within-feed ranks to raw values improves the transfer by +0.0046 (about 3.5 SE).
- Inside val (same regime), rank-only features *lose* information (−0.0055): the absolute level of f1..f5 matters. z-scores are neutral.
- Recommendation: **keep raw values and add within-feed ranks/z-scores.** Do not replace raw values.

### 7.4 The f4 inversion is a position artefact; position-aware training fixes the transfer (`s10`, `s10b`)

Train positive-rate ratio by val-defined f4 decile, *conditioned on production position*:

| pos range | d1 (lowest f4) | d2 | d3 | d5 | d8 | d10 |
|---|---|---|---|---|---|---|
| [0, 10) | **1.31** | 1.07 | 0.91 | 0.87 | 0.88 | 0.89 |
| [10, 50) | 0.83 | 1.06 | 1.07 | 1.01 | 1.04 | 0.94 |
| [50, 400) | **0.53** | 1.00 | 1.01 | 1.09 | 1.10 | 0.98 |
| val (random order) | **0.46** | 0.89 | 0.97 | 1.13 | 1.08 | 1.06 |

Low-f4 items are placed higher by production: mean pos is 65 in the lowest decile vs about 90 in the middle. Away from the top slots, the train relation matches val. The "low f4 is good" pattern in train is entirely the top-slot effect.

Train-fit LGB (60K train feeds, f1..f5 raw + within-feed rank + category, 300 rounds), evaluated on val; f1 alone = 0.3365:

| variant | val NDCG | vs f1 alone |
|---|---|---|
| plain | 0.3301 | −0.0063 ± 0.0013 |
| drop f4 | 0.3334 | −0.0031 ± 0.0011 |
| + pos as a feature, inferred at pos = 0 | 0.3305 | −0.0060 ± 0.0014 |
| **+ pos as a feature, inferred at pos = 20** | **0.3383** | **+0.0019 ± 0.0012** |
| drop f4, + pos, inferred at pos = 0 | 0.3349 | −0.0015 ± 0.0012 |

Sweep of the inference-time pos constant (same model, `s10b`):

| pos | 0 | 5 | 10 | 20 | 40 | 80 | 150 | 250 |
|---|---|---|---|---|---|---|---|---|
| val NDCG | 0.3305 | 0.3333 | 0.3349 | 0.3383 | 0.3389 | 0.3390 | 0.3394 | **0.3400** |

The gain is monotone in pos: +0.0099 over the plain train-fit model and +0.0036 ± 0.0011 over f1 at pos = 250. The deep-position regime, where the production ranker's choice matters least, is the closest proxy for random-order exploration traffic.

**Takeaway.** Position bias in train is not a nuisance to ignore: it *reverses* f4 and drags a naive train-fit model below a single feature. Using `pos` as a training feature with a non-top constant at inference (or equivalently fitting on pos ≥ 10-50 only, or IPS weighting) restores and exceeds f1. Inferring at pos = 0 does not help, because the top-slot regime is exactly where the relation is inverted.

## 8. Implications and recommendations

**Stable, safe features** (val ≈ test, JS ≤ 0.001, adversarial AUC ≈ 0.5):
- f1..f5 and category of candidates. Their marginals do shift between train and val/test, so prefer within-feed normalised versions as well (§7).
- User embedding and item embeddings (val-vs-test AUC 0.50-0.52; train-vs-val/test 0.51-0.56) and similarity features between them.
- History-based user features *as shares* (event-type mix, category mix, source mix).
- Train-*history* item counts: robust to the larger test gap and the best of the popularity signals.

**Features needing care:**
- Anything ID-based from train (item CTR, item candidate counts, user statistics). Coverage drops from 65% (val) to 54% (test) and 53% (private) of rows, and train counts/CTR are position-biased. Always pair them with an explicit "seen" flag or count so the model can tell "unknown" from "low". Ideally, train on val-like data with the same cold rate as private: simulate the gap by computing train statistics with an earlier cutoff when building training features (§4.6).
- In-split transductive counts (same-split candidate or history frequency). They are only comparable after normalising by split size: (count − 1) / n_feeds and history count / n_feeds are consistent val vs test (0.22 vs 0.23, 0.115 vs 0.117 per 10K feeds). In-split candidate frequency is useless; in-split history frequency is weakly positive (+0.0014..0.0020).
- user_id = 0 must be treated as "no user". The "user_embedding all-zero" flag shifts 5× (1.42% → 0.28%), so do not rely on it.
- Hour-of-day: distorted by logging outages and by weekend/weekday mix. Weak and risky.
- Raw M as a feature: fine within a feed (it is constant per feed and does not change the ranking alone), but its distribution differs train vs val (78 vs 96).

**Normalisation:** the train→val shift is partly a within-feed shape difference (std / mean of f-features per feed, share at the f1 floor). Within-feed percentile ranks lower train-vs-val adversarial AUC from 0.626 to 0.595 and improve train→val transfer by +0.0046 when added to the raw values (§7.3). Inside the val/test regime, rank-only features lose information (−0.0055). **Use raw + within-feed rank/z**, not either alone.

**Train labels are biased relative to the exploration-traffic target (§7):**
- strong position bias (6.3× top vs pos 150+);
- f4's relation inverts (lowest f4 decile: 1.22× in train vs 0.46× in val);
- a plain model fit on train (f1..f5 + category) scores *below f1 alone* on val (0.3250-0.3301 vs 0.3365);
- the same model with `pos` as a feature, inferred at a deep constant (pos ≥ 20, best at 250), reaches 0.3383-0.3400, i.e. +0.0099 over the plain model (§7.4).

Treat train as a source of representations and statistics (embeddings, item/user histories, popularity), not as the place to learn the final scoring function on f-features. Fit the final ranker, or at least its feature-weighting/calibration layer, on val exploration labels, or debias train: pos-as-feature with a deep inference constant, training only on pos ≥ 50, or IPS (§7.4). Always check train-fit models against "f1 alone" on val.

**Validation design to mimic private:**
1. Use **val as the primary holdout**. It is weekday traffic of the same exploration type and closest to private on M, category and weekday. Report val NDCG with **paired per-feed bootstrap**: the SE of differences is about 0.001, so effects under 0.002 are noise.
2. Make the **feature gap match private**. Private items are 9-10 days past train end vs val's 3-5. When computing train-derived item features for val, drop the last 4-6 train days (cutoff about Sep 14-17) so val's train coverage (about 53-58%) matches private's (about 53%). Alternatively, report val metrics stratified by train coverage.
3. **Expect the LB to be about 0.005-0.008 lower than val** from feed-size composition (§5.2). Public is about 0.003 harder than private by M, but weekend-only. Do not over-tune on public: it is 8.5K feeds (SE about 0.003), weekend-only, and carries a larger share of the leakable feeds.
4. If val is used for training (it is the only unbiased labelled data), use **GroupKFold by feed**, or split val by day (Tue → Thu) to check temporal stability. The Tue/Thu f1 NDCG difference was 0.0037 (about 1 SE).
5. **Break ties** in every evaluated and submitted score. Pessimistic ties cost f1 0.0085.

---

### Appendix: script index
- `s01_feeds_users.py`: per-day/hour volumes, M/H, user overlap, feeds/user, gaps, user 0.
- `s02_cand_dists.py`: f1..f5 quantiles/PSI/JS, per-day drift, categories, missing rates, event type/source.
- `s03_items.py`: item overlap/cold start, frequency distributions, target rate by item status (its blend Δs use a pessimistic base; superseded by s03c).
- `s03b_coverage_decay.py`: coverage by day since train end (its cutoff-simulation Δs use a pessimistic base; superseded by s03c).
- `s03c_signals_fair.py`: popularity signals vs fair f1 baseline, cutoff/gap simulation.
- `s04_adversarial.py` + `featlib.py`: adversarial validation (val/test/train, public/private).
- `s05_public_private.py`: boundary, composition, M-reweighted expected NDCG.
- `s06_user_item_overlap.py`: re-impressions, later-feed-history leak in train/val/test, event-type mapping.
- `s07_train_vs_val_relation.py`: train vs val single-feature NDCG, position bias, normalisation vs adversarial AUC and transfer (descriptive part logged in `out/s07_desc.log`, normalisation part in `out/s07_norm.log`, run with `SKIP_DESC=1`).
- `s08_dow_and_leak_bound.py`: train per-day stability, leak upper bound.
- `s09_segments.py`: per-segment val NDCG and segment shares val/public/private.
- `s10_train_transfer_debias.py`: train-fit LGB → val transfer: dropping f4, pos-as-feature debiasing, f4 relation conditioned on pos.
- `s10b_pos_constant_sweep.py`: inference-time pos constant sweep for the pos-aware train model.
