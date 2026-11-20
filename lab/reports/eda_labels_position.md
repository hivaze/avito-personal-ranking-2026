# EDA: labels, metric behaviour and position bias

Scripts: `<repo>/eda/labels_position/` (numbered `01_...` to `10_...`). Raw outputs (JSON, logs, per-feed arrays) are in `<repo>/eda/labels_position/out/`.
All NDCG values use the exact competition metric (full list, binary gain, pessimistic ties, feeds without positives = 1). A numba re-implementation (`common.ndcg_pf`) matched `src/metrics.py` exactly (max abs per-feed difference 0.0 on val) and is about 10x faster. When I say "val NDCG SE", I mean the per-feed standard error, about 0.0017 to 0.0019 for model scores. Comparisons use paired per-feed differences where I say so.
Timestamps are converted to Moscow time (UTC+3).

**Note on compute:** during this run the shared CPUs were heavily oversubscribed (a high load average on a shared machine). A LightGBM lambdarank run on 4M rows took about 3 s per boosting round. So the model experiments in section 6 use **GPU XGBoost `rank:ndcg`** instead (pairs taken from the top 300, which covers the whole list because M ≤ 300; no exponential gain; GPUs N and 6). They match the requested LightGBM lambdarank design in every other way.

---

## 0. TL;DR (most important first)

1. **Train contains a sizeable share of random-order (exploration-like) feeds.** Two independent order statistics show it.
   - A proxy of the production ranker (a GBDT that predicts within-feed `pos` from features) gives a per-feed z-score of Spearman(pred, pos). For M ≥ 50 this z-score fits a two-component mixture in which one component is exactly N(0,1), i.e. order unrelated to features. The weight of that component is a stable **12 to 14% for M in [50,250)** and **18.6% for M in [250,300]**.
   - The share of "removed already-viewed items" (gaps in `pos`) is 3 to 4 times lower in those feeds: about 2% versus about 8%. P(z < 0) rises from 2% to 28% as the gap share goes to 0.
   - In the likely-random feeds (P > 0.9, about 90K feeds with M ≥ 50), the target rate is nearly flat in position: 2.5% at rank 0, 1.15% at rank 30, base 1.15%. In likely-production feeds it is 9.1% at rank 0 against a base of 1.6%. The M distribution of the estimated random feeds matches val almost exactly (table in 2.5).
   - Estimated count: about **206K random feeds among the 1.55M train feeds with M ≥ 50**. Per-feed P(random) is saved in `out/train_prandom.npz`. It is unreliable for M < 50.
2. **Position bias in train is strong but mostly the ranker's relevance, plus a large "end-of-feed" spike.** The target rate is 11.8% at pos 0, 4.1% at pos 10, 1.3% at pos 100 and 0.92% at pos 200. Then it climbs back: the **last shown item has an 18% (random-like feeds: 29%) positive rate**, versus a 2.2% base. The feed is truncated right after the last interaction.
   - NDCG on train: pos ascending 0.3615; pos *descending* 0.4056; "last item first, then pos" 0.4250; random 0.3240.
   - **None of this is usable on val/test**, where the order is hidden, but it confounds training.
3. **Position confounding flips feature directions.** f4 (an item-level attribute) predicts positives on val (feed AUC 0.545; NDCG +f4 0.2974 vs −f4 0.2667) but *negatively* on train (feed AUC 0.491; NDCG −f4 0.3265 > +f4 0.3174). The ranker puts high-f4 items lower, so in train they get fewer clicks.
4. **What f1 to f5 are.**
   - f3: a user×category affinity. It is constant within (feed, category) in 100% of cases.
   - f1: a finer user×sub-category affinity (variance explained by (feed, category) η² = 0.97; Spearman with f3 0.78 to 0.82; 15 to 18% of values sit at the floor 1/56).
   - f4: a static item attribute (item η² = 0.98 to 0.995).
   - f5: a mostly-item, slowly varying attribute such as popularity (item η² = 0.92 to 0.97).
   - f2: a mixed item/user feature with 82 to 85% within-feed ties; weak on its own (feed AUC 0.494).
   - None of them is the production ranker score: |Spearman(pos, f)| < 0.08 per feed.
5. **Pessimistic ties are expensive.**
   - A constant score gives 0.1986 on val, against 0.2890 for random order.
   - f3 alone: 0.2518 pessimistic vs 0.3175 with random tie-breaking. f1 alone: 0.3279 vs 0.3365.
   - Simply adding `1e-3*f5` to f1 lifts it from 0.3279 to **0.3376**.
   - Always emit continuous, tie-free scores (check that the float32 cast does not create ties; it did not for the GBDT scores here).
6. **Metric.**
   - Full NDCG is about 0.06 to 0.08 above NDCG@30 (random 0.290 vs 0.207; best GBDT 0.361 vs 0.301), but it ranks models the same way.
   - Random order gives 0.289 (val) and an expected ~0.282 / 0.284 (test public / private), because test feeds are longer.
   - Small feeds (M < 30, 27% of feeds) give about 40% of the gain f1 achieves over random.
   - Under an equal-variance Gaussian model, the LB top of 0.3935 corresponds to a within-feed AUC of about 0.70.
   - An oracle that knows the positive's category reaches 0.479 (0.505 with f1 inside the category).
7. **Train vs val for training** (GPU XGBoost rank:ndcg; details in section 6):
   - A naive model on 200K train feeds with f1..f5+category (0.3327) is *worse* than the hand-made `f1+1e-3·f5` (0.3376) and than val-only 5-fold CV (0.3349).
   - With richer features (embedding dot/cos, history-category counts, history cosine, within-feed ranks), train alone reaches 0.3611. Val-only CV reaches 0.3572 with rank:ndcg and 0.3611 with binary.
   - **Debiasing with `pos` + `rkend` (rank-from-end) as training-only features, fixed at pos ≈ 20 and rkend ≈ 50 at inference, gives 0.3652 (+0.0042 ± 0.0012 paired).** Fixing pos at 0 *hurts* (−0.005 to −0.016).
   - **Random-like train feeds are worth more per feed**: 50K of them beat 50K M-matched production-like feeds by +0.0055 to +0.0066 and uniform feeds by +0.002 to +0.005. 50K random-like feeds + pos gives 0.3647, the same as 200K uniform feeds + pos + rkend.
   - Adding val to train (weight 1: −0.0006; weight 5: −0.0016), fine-tuning on val (−0.0002), P(random) weighting (−0.001) and M re-weighting (+0.0009) all gave no significant gain over the naive train model.
   - Rank-average blends lose 0.0085 to pessimistic ties; z-score blends do not.

---

## 1. Label structure: train vs val (vs test)

| | train | val | test |
|---|---:|---:|---:|
| feeds | 3,151,208 | 14,423 | 28,595 |
| candidates | 246.7M | 1.39M | 2.90M |
| mean M | 78.3 | 96.4 | 101.6 |
| M quantiles 10/25/50/75/90% | 9/19/48/113/213 | 12/27/67/145/247 | 13/30/72/157/256 |
| feeds with M in [250,300] | 5.5% | 9.7% | 11.0% |
| positive rate (per candidate) | **2.21%** | **1.58%** | – |
| mean positives / feed | 1.73 | 1.52 | – |
| feeds with exactly 1 positive | 66.6% | 74.5% | – |
| 2 / 3 / ≥4 positives | 18.6 / 7.0 / 7.9% | 15.4 / 5.0 / 5.1% | – |
| mean H (history length) | 185 | 173 | 174 |
| feeds with H = 256 (capped) | 42.3% | 29.0% | 28.8% |
| feeds with H = 0 | 1.1% | 1.3% | 0.9% |
| distinct users / feeds per user | 2.455M / 1.28 | 13,741 / 1.05 | 27,309 / 1.05 |

- Users seen in train: 31.8% of val users and 30.3% of test users. Only 0.34% of test users also appear in val.
- Train appears to be all traffic weighted per feed, so heavy users (long histories) are over-represented compared with val/test. Val/test look like a per-request sample of exploration traffic: val and test agree closely (H = 256 share 29.0% vs 28.8%; mean M 96 vs 102).
- **Positive rate vs M.** The rate falls roughly as 1/M because every feed has ≥ 1 positive (Spearman(feed rate, M) = −0.88 train, −0.89 val), while the number of positives grows with M (corr(npos, M) = 0.24 / 0.20).
  - Val is lower than train in every M bin: M in [5,10): 15.7% vs 16.8%; [30,50): 3.5% vs 4.0%; [100,150): 1.34% vs 1.67%; [250,299): 0.82% vs 1.08%.
  - Production order puts relevant items where they are seen, which raises clicks per shown item.

| M bin | train share | train rate | train npos | val share | val rate | val npos | val frac 1-pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| [5,10) | 10.6% | 16.8% | 1.15 | 7.2% | 15.7% | 1.08 | 0.94 |
| [10,20) | 14.5% | 9.06% | 1.29 | 10.4% | 8.12% | 1.16 | 0.89 |
| [20,30) | 11.5% | 5.86% | 1.43 | 9.3% | 5.05% | 1.23 | 0.84 |
| [30,50) | 14.3% | 4.05% | 1.56 | 13.9% | 3.49% | 1.33 | 0.78 |
| [50,75) | 12.0% | 2.82% | 1.73 | 12.6% | 2.28% | 1.40 | 0.77 |
| [75,100) | 8.5% | 2.19% | 1.89 | 9.5% | 1.80% | 1.55 | 0.72 |
| [100,150) | 10.5% | 1.67% | 2.04 | 12.9% | 1.34% | 1.64 | 0.69 |
| [150,200) | 6.6% | 1.29% | 2.24 | 8.0% | 1.05% | 1.81 | 0.63 |
| [200,250) | 5.9% | 1.05% | 2.38 | 6.4% | 0.87% | 1.96 | 0.61 |
| [250,300] | 5.5% | 1.08% | 2.90 | 9.7% | 0.82% | 2.26 | 0.55 |

- **Positive rate vs history length H.** It increases with H.
  - Train: H = 0: 1.62%; [1,16): 1.93%; [16,64): 2.01%; [64,128): 2.11%; [128,256): 2.25%; 256: 2.28%.
  - Val: 1.18 / 1.31 / 1.46 / 1.57 / 1.62 / 1.62%.
  - Mean npos also rises: train 1.34 → 1.82; val 1.18 → 1.57.
- **Hour of day (MSK).**
  - Train rate peaks around 10:00 to 11:00 (2.34%) and is lowest at 20:00 to 21:00 (2.00 to 2.03%). Feeds are longer late in the evening and at night (mean M 88 at 21:00 to 00:00 vs 72 at 10:00 to 12:00). So part of the hourly rate pattern is the M effect.
  - Val follows the same shape with more noise (1.41 to 1.85%).
  - The amplitude (±8% relative) is small compared with M effects.
- **Days.**
  - Train covers Sep 6 to 14 and Sep 17 to 19. **Sep 15 to 16 are missing** (a 48 h gap; only 42 feeds on Sep 15). Daily positive rate is stable at 2.16 to 2.25%.
  - **Val is only Sep 22 and Sep 24** (7,542 + 6,878 feeds; Sep 23 is missing, a 24 h gap). The two days give equal NDCG (f1: 0.3290 vs 0.3266).
  - Test covers Sep 26 to 29 completely (6.0K to 7.6K feeds per day). Small 1 to 2 h holes appear around 19:00 to 21:00 on several days in all splits.
  - **Public LB = test feeds up to 2026-09-27 11:32 MSK** (first 30% by timestamp). Private = the rest (Sep 27 midday to Sep 29).

## 2. Position bias in train

### 2.1 Target rate vs `pos` (full train, 246.7M candidates)

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 10 | 20 | 30 | 50 | 100 | 150 | 200 | 250 | 280 | 290 | 299 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rate % | 11.81 | 7.66 | 6.63 | 6.44 | 5.33 | 5.08 | 4.05 | 2.98 | 2.25 | 1.82 | 1.27 | 1.01 | 0.92 | 0.93 | 1.32 | 2.07 | 15.33 |

- The shape (pos 2≈3, 4≈5, ...) suggests a 2-column grid with a special first slot.
- The raw curve mixes in M composition, because high positions exist only in long feeds, which have low rates. Within M bins the shape is the same, for example M in [200,300]: 6.9% at pos 0 → 2.1% at pos 10 → 0.94% at pos 100 → 0.74% at pos 199 → 15% at pos 299.
- **End-of-feed spike.** By within-feed rank from the end (`rkend`):
  - last item 17.95%; second to last 4.47%; third to last 17.5%; then about 2.7%.
  - In feeds with M ≥ 50 only the last item spikes: 22.6% in production-like feeds and **28.8% in random-like feeds**, against bases of 1.6% and 1.15%.
  - Interpretation: the logged feed ends at (or one grid row after) the last item the user interacted with. So M is a function of user behaviour: users who find something early produce short feeds.
  - 10.4% of all train positives are the last item of their feed (6.5% are the first).
- Excluding the last 20 ranks of each feed, the curve is monotone: 10.0% at pos 0, 2.75% at pos 10, 0.98% at pos 100, 0.75% at pos 200.

### 2.2 NDCG of position rules on train (all 3.15M feeds)

| ordering | train NDCG |
|---|---:|
| random | 0.3240 |
| pos ascending | 0.3615 |
| pos descending | 0.4056 |
| U-shape min(rank, rank-from-end) | 0.4104 |
| last item first, then pos ascending | **0.4250** |

By M bin, pos ascending is best only for M < 20 (M < 10: 0.530 vs pos-descending 0.493 and last-first 0.441). For M ≥ 20, "last-first" beats pos ascending by 0.07 to 0.15 (M in [20,50): 0.473 vs 0.374; [50,100): 0.469 vs 0.321; [100,200): 0.392 vs 0.286; ≥200: 0.333 vs 0.262). **`pos`, rank from end and M carry a large label signal in train that does not exist on val/test**, so they must not leak into inference features (section 6 shows how to use them safely).

### 2.3 Gaps in `pos` and filtering

- 85.2% of train feeds contain pos 0. 11.9% start at pos 1 and 2.1% at pos 2.
- Only 14.5% of feeds have contiguous positions. The median feed has 4 missing positions below its max pos; the mean is 7.9; a mean of 9.9% of positions are missing.
- Duplicated pos values are essentially absent (15 feeds).
- The missing-position rate falls with pos: 14.8% at pos 0, 12.2% at pos 9, 9.1% at pos 50, 8.0% at pos 100, 6.8% at pos 200.
- These are the "already viewed by the user" items removed from all splits (readme). The ranker evidently promotes previously viewed items, so they are removed more often near the top.
- Val/test have the same filter applied, but the gaps are invisible there.
- **Val/test feeds are longer** (median M 67/72 vs 48). This has two consistent causes:
  - In random order users scroll further before interacting.
  - Fewer items are removed from random feeds (gap share about 2% vs 8%; section 2.5).
- The M distribution of the random-like train feeds matches val (table in 2.5).

### 2.4 Is the production order related to f1..f5? (per-feed Spearman)

- Per-feed Spearman(pos, f) over all train feeds:

  | feature | f1 | f2 | f3 | f4 | f5 |
  |---|---:|---:|---:|---:|---:|
  | mean Spearman(pos, f) | −0.081 | −0.013 | −0.076 | +0.071 | −0.026 |

  The distributions are unimodal. For M ≥ 50 the spread (SD about 0.1) is about what sampling noise alone gives (1/√(M−1)), so **no feature is (close to) the ranker score**.
- The production ranker puts high f1/f3/f5 slightly higher and high f4 lower.
- Category adjacency in pos order (same-category neighbours) equals the random-permutation expectation (obs/exp = 0.990; per-feed z ≈ N(−0.08, 1)). So there is no visible category clustering or diversification either.

### 2.5 Detecting random-order (exploration) feeds inside train

**Method.**
- A GBDT predicts the within-feed normalised pos rank. It uses f1..f5, category, user·item dot/cos, item norm, history-category counts, the maximum cosine to the last 32 history items, within-feed percentile ranks, M, H, and item-level mean pos rank (target-encoded on a disjoint feed set).
- It is trained on 200K feeds and evaluated on other feeds. Its mean per-feed Spearman with pos is only 0.195: production order is mostly driven by signals we do not have. The top gain features are f4 15%, f3 13%, category 11%, hsim_max 6% and M 6%.
- For each feed, z = Spearman(pred, pos)·√(M−1). Under random order z ~ N(0,1).

**Evidence.**

(a) **Mixture fit.** The model is π·N(0,1) + (1−π)·N(μ_M, σ_M), with an EM fit per M bin.
- In the 04b held-out sample (Sep 13 to 19), π = 0.136 / 0.140 / 0.152 for M in [50,100), [100,200), [200,300], while μ_M grows from 2.06 to 2.89.
- The likelihood gain over a single Gaussian is ΔLL = 463 to 801 per bin.
- The lower tail matches N(0,1) scaled by π. For M ≥ 200: P(z < −1.5) is 1% (π·0.067 = 1.0%); P(z < −0.5) is 5% (π·0.31 = 4.7%).
- A homogeneous production population with μ = 2.9, σ = 1.25 would give P(z < 0) = 1%, but 8.5% is observed.

(b) **Gap share independently separates them.** On the held-out sample with M ≥ 50:

| gap share (missing / (missing+M)) | [0,.02) | [.02,.04) | [.04,.06) | [.06,.08) | [.08,.10) | [.10,.15) | [.15,.20) | ≥.20 |
|---|---|---|---|---|---|---|---|---|
| P(z < 0) | 0.276 | 0.170 | 0.085 | 0.044 | 0.028 | 0.020 | 0.019 | 0.034 |
| median z | 0.99 | 1.78 | 2.15 | 2.31 | 2.34 | 2.31 | 2.26 | 2.10 |

- The gap share depends only on pos values, not on features.
- The 2-D mixture on all 1.64M scored feeds (z plus log gap share; full run `04g` / `04h`) estimates a gap share of **1.4 to 2.4% in random feeds vs 5.6 to 8.7% in production feeds**.
- High-P(random) feeds have 6.5 gaps vs 15.1 for production-like feeds at a similar M (151 vs 140).

(c) **Behaviour of high-posterior feeds** (P > 0.9, M ≥ 50, 89,794 feeds) vs production-like feeds (P < 0.05, 1.01M feeds):

| within-feed rank | 0 | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 100 | base | last item |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random-like rate % | 2.51 | 2.23 | 1.75 | 1.77 | 1.52 | 1.43 | 1.31 | 1.15 | 1.02 | 1.15 | 28.8 |
| production-like rate % | 9.15 | 5.71 | 4.16 | 3.64 | 3.11 | 2.54 | 1.92 | 1.61 | 1.20 | 1.58 | 22.6 |

- NDCG of pos ascending: random-like 0.2288 vs 0.2235 for random scores (+0.005); production-like 0.3067 vs 0.2420 (+0.065).
- The positive rate of random-like feeds matches val in the same M bins (M ≥ 200: 0.85% vs val 0.82 to 0.87%; production-like 1.14%).
- Their mean f1 / f5 (0.41 / 0.45) are close to val (0.40 / 0.44) and below production-like feeds (0.45 / 0.48). This fits a production list that is a *ranker-selected* top of the pool, while exploration shows a random subset.

(d) **Not user-bucketed and not time-sliced.**
- Train feeds of val/test users have P(z < 0) = 0.075, against 0.082 in a random control sample of train feeds (M in [50,100)). The two are equal within noise.
- The mean posterior is flat across days (0.134 to 0.153) and hours (0.128 to 0.157).
- So exploration is randomised per request.

**Estimated amount** (2-D mixture, all scored train feeds):

| M bin | train share | π (random) | est. random feeds | random-feed M share within M≥50 | val share within M≥50 | all-train share within M≥50 |
|---|---:|---:|---:|---:|---:|---:|
| [50,75) | 12.0% | 0.119 | 44.9K | 0.218 | 0.213 | 0.245 |
| [75,100) | 8.5% | 0.126 | 33.7K | 0.164 | 0.160 | 0.173 |
| [100,150) | 10.5% | 0.132 | 43.8K | 0.213 | 0.218 | 0.215 |
| [150,200) | 6.6% | 0.135 | 28.0K | 0.136 | 0.135 | 0.134 |
| [200,250) | 5.9% | 0.123 | 22.9K | 0.111 | 0.109 | 0.121 |
| [250,300] | 5.5% | 0.186 | 32.3K | 0.157 | 0.165 | 0.113 |

- About **206K random-order feeds among the 1.55M train feeds with M ≥ 50 (13.3%)**. Their M profile reproduces val, including the excess of very long feeds.
- For M < 50 the mixture is not identifiable: the z separation is ≤ 1.6 σ and the gap share is discrete. π estimates there (0.09 to 0.57) are not trustworthy.
- If the M ≥ 50 rate also held for short feeds, train would contain about 0.4M exploration feeds (about 30 times val).
- File: `out/train_prandom.npz` with arrays `fidx`, `p_random`, `z` and `gap_share`, indexed by train fidx. `p_random` is NaN for the 400K feeds used to fit the proxy (parts A∪B of `04b`). Values for M < 50 exist but are **unreliable**; use them only for M ≥ 50.
- Precision: 89.8K feeds with M ≥ 50 have P > 0.9. Their mean posterior is 0.97, so the model implies about 3% contamination. The real figure is likely higher, because the production component is not exactly Gaussian.

## 3. What are f1..f5? Marginals, ties, item vs user level

Computed on a 300K-feed train sample plus all of val and test.

| | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|
| η² explained by item_id (items in ≥3 feeds), train / val | 0.45 / 0.52 | 0.59 / 0.66 | 0.50 / 0.54 | **0.98 / 0.995** | **0.92 / 0.97** |
| η² by (feed, category) | **0.97** | 0.54 | **1.00** | 0.35 | 0.38 |
| η² by (user, category) | 0.94 | 0.52 | 0.95 | 0.33 | 0.36 |
| η² by category alone | 0.19 | 0.07 | 0.24 | 0.05 | 0.11 |
| (feed, cat) groups with a constant value | 30% | 37% | **100%** | 1.3% | 2.6% |
| within-feed tie fraction (train / val / test) | 23 / 26 / 26% | 82 / 85 / 86% | 92 / 91 / 92% | 3.1 / 2.9 / 2.7% | 6.5 / 12 / 12% |
| floor value (share train / val) | 0.0179=1/56 (15.4 / 17.6%) | 0 (11.8 / 15.4%) | 0.0161=1/62 (5.3 / 5.6%) | 0.0156=1/64 (1.2 / 1.1%) | 0.0161 (5.6%), 0 (1.4%) |
| mean train / val / test | 0.443 / 0.400 / 0.397 | 0.443 / 0.440 / 0.440 | 0.488 / 0.451 / 0.447 | 0.504 / 0.505 / 0.508 | 0.481 / 0.442 / 0.444 |
| global Spearman with pos (train) | −0.042 | +0.003 | −0.070 | +0.079 | −0.030 |
| mean on positives vs negatives, train | 0.569 / 0.440 | 0.453 / 0.443 | 0.611 / 0.485 | **0.483 / 0.505** | 0.559 / 0.479 |
| same, val | 0.517 / 0.398 | 0.447 / 0.440 | 0.556 / 0.449 | **0.544 / 0.504** | 0.509 / 0.441 |

Feature correlations (Spearman, val): f1–f3 0.82, f4–f5 0.37, f1–f5 0.17, the rest |ρ| < 0.1.

- **f3**: a user × category affinity. It takes exactly one value per (feed, category), varies between users for the same category, and changes slowly over time for the same user.
- **f1**: user × finer category (microcategory) affinity. It is nearly constant within (feed, category), strongly correlated with f3, and has a floor at 1/56 for about 16% of candidates (the "never interacted" bucket).
- **f4**: a static item attribute (price-like or age-like). It is item-determined (η² ≈ 0.99) and not personalised.
- **f5**: a slowly changing item attribute (popularity/CTR-like). Item η² is 0.92 to 0.97, and it has more within-item drift than f4.
- **f2**: a mixed item/context feature (item η² 0.59, (feed, cat) η² 0.54) with heavy ties.
- "Normalised interpolated bins" fit on train give approximately uniform marginals on train (f1 histogram flat apart from the floor bin). Val/test are shifted *down* for the personalised and popularity features (f1, f3, f5) but not for f4/f2. This fits exploration feeds showing a random subset of the pool, while production shows the ranker's top.

## 4. Single-feature NDCG: train vs val (position confounding)

Full NDCG. "pess" = official pessimistic ties; "exp" = expected value under random tie-breaking.

| scorer | val pess | val exp | train pess | train exp | val feed-AUC | train feed-AUC |
|---|---:|---:|---:|---:|---:|---:|
| random | – | 0.2890 | – | 0.3239 | 0.5 | 0.5 |
| constant | 0.1986 | 0.2890 | 0.2201 | 0.3239 | – | – |
| +f1 | 0.3279 | 0.3365 | 0.3530 | 0.3631 | 0.623 | 0.608 |
| −f1 | 0.2402 | 0.2537 | 0.2748 | 0.2854 | | |
| +f2 | 0.2360 | 0.2904 | 0.2614 | 0.3237 | 0.494 | 0.494 |
| −f2 | 0.2557 | 0.2935 | 0.2861 | 0.3257 | | |
| +f3 | 0.2518 | 0.3175 | 0.2710 | 0.3510 | 0.601 | 0.594 |
| −f3 | 0.2360 | 0.2532 | 0.2616 | 0.2860 | | |
| **+f4** | **0.2944** | **0.2974** | 0.3144 | 0.3174 | **0.545** | **0.491** |
| **−f4** | 0.2663 | 0.2667 | **0.3252** | **0.3265** | | |
| +f5 | 0.2952 | 0.2981 | 0.3376 | 0.3411 | 0.561 | 0.564 |
| −f5 | 0.2547 | 0.2630 | 0.2938 | 0.2996 | | |

Paired SE of each feature vs random on val: 0.0008 to 0.0014.

- **f4 changes sign between train and val.**
  - On val, high f4 is good: +0.008 over random, +0.031 over −f4.
  - On train, high f4 is bad: −f4 beats +f4 by 0.009.
  - Spearman(pos, f4) = +0.071 per feed, so the ranker puts low-f4 items into the top slots, and with a 5 to 10x top-slot lift that reverses the marginal association.
  - Inside likely-random train feeds the val direction comes back (M ≥ 50: +f4 0.2647 vs −f4 0.2389). Inside production-like feeds it does not (0.2638 vs 0.2729). This is partly a selection effect, because the proxy uses f4, but the direction agrees.
- f1/f3 are a bit *stronger* on val (feed AUC 0.623 vs 0.608) and f5 is equal. f2 is useless alone in both.
- **Train absolute NDCG values are higher only because train feeds are shorter** (random 0.324 vs 0.289). Compare lifts over random, not levels.

## 5. Metric nuances

### 5.1 Full NDCG vs NDCG@k (val)

| scorer | full | @10 | @30 | @100 |
|---|---:|---:|---:|---:|
| random | 0.2902 | 0.1404 | 0.2069 | 0.2654 |
| f1 | 0.3279 | 0.1875 | 0.2579 | 0.3103 |
| f1 + 1e-3·f5 | 0.3376 | 0.1993 | 0.2706 | 0.3218 |
| GBDT (train, rich feats) | 0.3611 | – | 0.3008 | – |

- Full NDCG = NDCG@30 + about 0.06 to 0.08 for these scorers. The remaining credit comes from positives ranked below 30, where each still earns 1/log2(r+1): 0.20 at rank 30, 0.15 at 100, 0.12 at 300.
- Orderings of models agree between full and @30, but full NDCG rewards pushing the hopeless tail *upward* a little. That is why a lambdarank-type objective should consider all pairs (truncation ≥ max M), not the default 30.
- Random-order NDCG has a closed form: E = k·mean(disc_1..M)/IDCG_k. On val: 0.2890.

### 5.2 Dependence on M (val; each feed has equal weight)

| M bin | feed share | random | f1 | "headroom" share Σ(1−random) | share of f1's gain over random | share of rank-sum(f1,f3,f5) gain |
|---|---:|---:|---:|---:|---:|---:|
| [5,10) | 7.2% | 0.539 | 0.611 | 4.7% | 13.5% | 14.2% |
| [10,20) | 10.4% | 0.416 | 0.467 | 8.6% | 13.8% | 14.7% |
| [20,30) | 9.3% | 0.345 | 0.399 | 8.5% | 12.9% | 11.1% |
| [30,50) | 13.9% | 0.302 | 0.339 | 13.7% | 13.0% | 14.9% |
| [50,75) | 12.6% | 0.261 | 0.297 | 13.1% | 11.4% | 11.9% |
| [75,100) | 9.5% | 0.242 | 0.280 | 10.1% | 9.4% | 8.9% |
| [100,150) | 12.9% | 0.222 | 0.250 | 14.1% | 9.4% | 9.4% |
| [150,200) | 8.0% | 0.208 | 0.232 | 8.9% | 5.1% | 4.4% |
| [200,250) | 6.4% | 0.198 | 0.227 | 7.3% | 4.9% | 4.2% |
| [250,300] | 9.7% | 0.194 | 0.221 | 11.0% | 6.9% | 6.4% |

- Short feeds are worth disproportionately much. M < 30 is 27% of feeds and 22% of the headroom, but about 40% of the realised gain of simple scorers. Long feeds (M ≥ 150, 24% of feeds) contribute only 15 to 17%.
- A single positive in a 300-item feed moving from rank 150 to rank 30 changes that feed's NDCG by only 0.064 (0.138 → 0.202).
- **Test public vs private composition.**
  - Public: 8,578 feeds, mean M 103.6. Private: 20,017 feeds, mean M 100.7. Val: mean M 96.4.
  - Expected random NDCG (using val's npos | M): public 0.2818, private 0.2844, val 0.2890.
  - So the same model should score about 0.005 to 0.007 lower on the LB than on val (slightly less on private), before any drift.

### 5.3 Rough ceilings (val)

- Perfect ranking = 1.0.
- Equal-variance Gaussian simulation on val's exact (M, npos) structure (negatives N(0,1), positives N(d,1)):

  | within-feed AUC | 0.50 | 0.57 | 0.64 | 0.70 | 0.76 | 0.81 | 0.86 | 0.92 |
  |---|---|---|---|---|---|---|---|---|
  | NDCG | 0.289 | 0.316 | 0.352 | 0.392 | 0.441 | 0.491 | 0.552 | 0.672 |

  The public LB top (0.3935, on test where random is about 0.282) needs a within-feed AUC of about 0.70 to 0.71. Around that level, +0.01 NDCG ≈ +0.012 to 0.016 AUC. f1 alone has a feed AUC of 0.623.
- **Oracle: the positive's category is known** (put all candidates of the positives' categories first, random inside): **0.479**. Adding f1 inside the category: **0.505**. On average 32% of candidates share a positive's category, and feeds hold 12.0 distinct categories. Category-level intent is therefore the biggest available lever; history-category features are the natural proxy.

### 5.4 Pessimistic ties

- A constant score = all positives last: 0.1986 on val (0.2201 on train), 0.09 below random.
- Discrete features lose a lot: f3 0.2518 pessimistic vs 0.3175 expected; f2 0.2360 vs 0.2904; f1 0.3279 vs 0.3365.
- Fixes: tie-break with any other informative continuous feature (f1 + 1e-3·f5: **0.3376**; f3 + 1e-6·f1: 0.3266), or any model output.
- Trained GBDT scores here had no material ties: pessimistic = expected to 1e-5, and the float32 cast did not change NDCG. Very shallow models, or models on only categorical/discrete inputs, can still produce within-feed ties. Always check `pess` vs `expected`.

## 6. Using train (biased) vs val (unbiased, small) for training

Setup:
- GPU XGBoost `rank:ndcg` (pairs from the top 300 = all pairs; label gain 1), η = 0.05, depth 8, min_child_weight 50, subsample 0.8, λ = 10. Val-CV models use depth 6 and min_child_weight 100.
- Feature sets:
  - BASIC = f1..f5 + category.
  - RICH = BASIC + user·item dot/cos, item-embedding norm, 4 history-category counters (all events / share / last 20 / non-type-1), max/mean cosine to the last 32 history items (all events and non-type-1), within-feed percentile ranks of f1/f3/f4/f5/dot, M, H.
- Train sample: 200K random feeds (15.6M candidates).
- Val numbers are full val (14,423 feeds). For CV they are out-of-fold over 5 user-grouped folds.
- Checkpoints are reported at 100/200/400/600 rounds. "best" is selected on val itself, so it is slightly optimistic (by about 0.001 to 0.002).

### 6.1 Results (val NDCG, exact pessimistic metric; paired Δ ± SE vs the reference model)

| # | training data | features | objective | val NDCG | paired Δ |
|---|---|---|---|---:|---|
| ref0 | – (hand-made) | f1 + 1e-3·f5 | – | 0.3376 | |
| a1 | train 200K | BASIC | rank:ndcg | 0.3327 (@200) | −0.0048 ± 0.0015 vs ref0 |
| a2 | train 200K | BASIC | binary | 0.3273 | −0.0054 vs a1 |
| **a3** | train 200K | RICH | rank:ndcg | **0.3611** (@200; 0.3578 @600) | reference for the rows below |
| b1 | train 200K | BASIC + pos, inference pos = 20 | rank | 0.3383 | +0.0055 ± 0.0012 vs a1 |
| b2 | train 200K | BASIC + pos + rkend, inference pos = 50, rkend = 50 | rank | 0.3394 | +0.0067 ± 0.0013 vs a1 |
| b3 | train 200K | RICH + pos, inference pos = 20 | rank | 0.3637 (@200) | +0.0026 ± 0.0011 vs a3 |
| **b4** | train 200K | RICH + pos + rkend, inference pos = 20, rkend = 50 | rank | **0.3652** (@600, still rising) | **+0.0042 ± 0.0012** vs a3; +0.0074 ± 0.0012 vs a3@600 |
| c1 | val only, 5-fold CV (OOF) | BASIC | rank | 0.3349 (@100) | a1 − c1 = −0.0022 ± 0.0016 |
| c2 | val only, 5-fold CV | RICH | rank | 0.3572 (@200) | a3 − c2 = +0.0039 ± 0.0014 |
| c3 | val only, 5-fold CV | RICH | binary | 0.3611 (@200) | a3 − c3 = 0.0000 ± 0.0014 |
| d1 | train 200K + 4/5 val (group weight 1), OOF | RICH | rank | 0.3606 (@300) | −0.0006 ± 0.0011 vs a3 |
| d2 | train 200K + 4/5 val (group weight 5), OOF | RICH | rank | 0.3597 (@300) | −0.0016 ± 0.0012 vs a3 (weight 20 not run: time) |
| e1 | pretrain on train 200K (300 rounds), continue on 4/5 val | RICH | rank | 0.3611 (@300) | −0.0002 ± 0.0011 vs a3 |
| f1 | 50K **random-like** train feeds (P > 0.9, M ≥ 50) | BASIC | rank | 0.3345 | +0.0066 ± 0.0015 vs f2; +0.0049 ± 0.0015 vs f3 |
| f2 | 50K production-like (P < 0.05), M-matched | BASIC | rank | 0.3280 | |
| f3 | 50K uniform M ≥ 50, M-matched | BASIC | rank | 0.3297 | +0.0017 ± 0.0013 vs f2 |
| f4 | 50K random-like | RICH | rank | 0.3585 | +0.0055 ± 0.0015 vs f5; +0.0022 ± 0.0015 vs f6 |
| f5 | 50K production-like | RICH | rank | 0.3530 | |
| f6 | 50K uniform M ≥ 50 | RICH | rank | 0.3564 | +0.0033 ± 0.0014 vs f5 |
| g2 | 50K random-like + pos (inference pos = 20) | RICH + pos | rank | 0.3647 (@400) | +0.0062 ± 0.0012 vs f4; +0.0036 ± 0.0013 vs a3 |
| g1 | train 200K, group weight 1 + 4·P(random) | RICH | rank | 0.3601 (@200) | −0.0010 ± 0.0012 vs a3 |
| g3 | train 200K, group weights matching val's M distribution | RICH | rank | 0.3620 (@200) | +0.0009 ± 0.0011 vs a3 |
| h1 | z-score blend 0.5·a3 + 0.5·c3 | – | – | 0.3639 | +0.0028 ± 0.0011 vs a3 |
| h2 | z-score blend 0.7·b4 + 0.3·c3 | – | – | 0.3658 | +0.0006 ± 0.0007 vs b4 |
| h3 | **rank-average** 0.5·b4 + 0.5·c3 (creates ties) | – | – | 0.3568 (0.3643 with expected ties; 0.3647 with a 1e-6 tie-breaker) | −0.0085 ± 0.0008 vs b4 |

How the fixed inference value of `pos` matters (val NDCG at 400 rounds):

| inference pos | 0 | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| BASIC + pos (b1) | 0.3169 | 0.3182 | 0.3230 | 0.3339 | 0.3375 | **0.3383** | 0.3382 | 0.3370 | 0.3349 |
| RICH + pos (b3) | 0.3558 | – | – | **0.3635** | – | 0.3619 (0.3637 @200) | 0.3566 | 0.3521 | – |

- With rkend: BASIC with (pos, rkend) = (50, 50) gives 0.3394 and (5, 50) gives 0.3341. Setting rkend to 0 or 2 (the "end-of-feed" values) hurts: 0.3262 / 0.3275 vs 0.3341. So does rkend = 200: 0.3304.
- In b4 the gain shares are rkend 57% and pos 15%: the model spends most of its capacity on the train-only position structure, which is exactly what we want it to absorb.

### 6.2 Interpretation

- **Position confounding is real and costs accuracy when train is used naively.**
  - A GBDT on 200K train feeds with BASIC features is *worse* than the hand-made `f1 + 1e-3·f5` (−0.0048 ± 0.0015), and worse than a val-only CV model (−0.0022 ± 0.0016).
  - The naive RICH model peaks at 200 rounds and then *degrades* (0.3611 → 0.3578 at 600): it keeps fitting production-order artefacts.
- **Adding `pos` (and rank-from-end `rkend`) as training-only features and fixing them at inference is the cheapest robust fix**: +0.004 to +0.007. With these features the model keeps improving with more rounds (0.3652 at 600 and still rising).
  - Fix them at a **mid-list value** (pos ≈ 5 to 50, rkend ≈ 50), not at 0. Pos = 0 is the special hero slot and costs −0.005 (RICH) to −0.016 (BASIC).
  - Treat the fixed value as one hyperparameter tuned on val.
- **Random-order train feeds are the most valuable train data per feed.** At equal size and M profile they beat production-like feeds by +0.0055 to +0.0066 (≥ 3.7 SE) and uniform feeds by +0.0049 ± 0.0015 (BASIC) and +0.0022 ± 0.0015 (RICH, not significant).
  - 50K random-like feeds plus `pos` (0.3647) match 200K uniform feeds with pos and rkend (0.3652).
  - Simply re-weighting a uniform sample by P(random) did not help (−0.001). Use *selection*: build training sets from the ~0.2M (M ≥ 50) random-like feeds plus pos-debiased production feeds, rather than soft weights.
  - They are also a candidate **large pseudo-validation set** (about 14 times val) for early stopping. I have not verified that their NDCG tracks val.
- **Val as training data.**
  - With BASIC features, 11.5K val feeds beat 200K train feeds.
  - With RICH features, val-only CV equals train-only, but only with a *pointwise* objective (binary 0.3611 vs rank:ndcg 0.3572). On small data the pairwise objective overfits.
  - Adding val folds to the train sample (weight 1: −0.0006; weight 5: −0.0016) or fine-tuning on val (−0.0002) gave no measurable gain at this scale.
  - A *z-score* blend of a train model and a val-CV model adds +0.0028 over the naive train model, but nothing over the debiased model b4.
  - Recommendation: keep val primarily for model selection and for the fixed-pos hyperparameter. For the final submission, adding val with group weight 1 is neutral (weight 5 was −0.0016 ± 0.0012). A z-score blend with a val-trained pointwise model is the safer way to use val's unbiased labels.
- **Objective.** Pairwise over the full list (rank:ndcg, top-300 pairs) beat binary on train (+0.0054 BASIC). On the small val set, binary was better (+0.0039 RICH).
- **M re-weighting to val's M distribution is neutral** (+0.0009 ± 0.0011).
- **Ties.** Model scores had no ties (pessimistic = expected; the float32 cast is harmless). But **rank-averaging two models creates many exact ties and lost 0.0085**. Blend z-scores or raw scores, or add a tie-breaker.

### 6.3 Concrete recommendations

1. Validate only on val, with the exact pessimistic metric and paired per-feed comparisons. A paired SE for two GBDTs is about 0.0011 to 0.0016, so treat |Δ| < 0.003 as noise.
2. Train on train with **`pos` and `rkend` (= M−1−rank of pos) as training-only features**, set at inference to (≈20, ≈50) (tune on val). Never feed the real pos/rkend/M-derived end signals at inference: they do not exist in val/test.
3. Prefer **random-like feeds** (`out/train_prandom.npz`, P > 0.9, M ≥ 50: about 90K feeds identified at high confidence; about 0.2M expected) when sub-sampling train. Over-represent them relative to their 13% base rate.
4. Use full-list pairwise ranking objectives (truncation ≥ 300) on train-sized data and pointwise on val-only models. Blend with z-scores, never with plain ranks.
5. Always output strictly continuous scores. Add a tiny tie-breaker built from an informative continuous feature (e.g. `+1e-6·f1`).
6. Expect LB ≈ val − 0.005 (test feeds are longer; random baseline 0.282 / 0.284 vs 0.289). Public (Sep 26 to 27 11:32) and private (the rest) differ by about 0.003 in baseline alone.

## 7. Other observations

- **Future-history leakage across feeds of the same user.**
  - In train, for feeds that have a later feed of the same user (pairs within Sep 6 to 8: 61.5K feeds, median gap 16.9 h), **93.6% of the earlier feed's positives appear in the later feed's history**, against 0.28% of its negatives. Mostly they appear as event_type 1 (99K hits), then 3 (21K) and 2 (12K).
  - So the target action is logged as a history event, most often type 1. Any cross-feed user×item aggregate built from *other* feeds' histories (for example "has the user ever interacted with this item" over all train histories) leaks the label for training feeds and must be restricted to histories with an earlier timestamp.
  - In val only 526 feeds (3.6%) have a later val feed of the same user, and only 4.5% of their positives show up there (median gap 4 minutes; history is apparently not updated that fast). The exploitable within-test signal is therefore negligible (about 25 feeds in val, ≤ 0.001 NDCG).
- Candidates never appear in their own feed's history, because viewed items are removed (readme). Those removed items are exactly the `pos` gaps.
- **The end-of-feed spike tells us how M is defined.** M is roughly "items up to the last interaction (+ about one grid row)". So M itself is label-informative: P(positive) per candidate ≈ 1/M scale. That is fine to use (M is known at test time), and models do use it (M gain about 4%).

## 8. Reproducibility

| script | content |
|---|---|
| `common.py`, `featlib.py`, `xgbexp.py`, `lgbexp.py` | metric (numba, verified), feature builder (dot/cos, history-category counts, history cosine), XGBoost/LightGBM helpers |
| `01_labels.py` | label structure, M/H/hour/day tables, user overlap |
| `02_position.py`, `02b_end_effect.py` | rate vs pos, per-feed Spearman(pos, f), pos-rule NDCG, end-of-feed effect |
| `03_features.py` | f1..f5 marginals, ties, η² decomposition |
| `04_explore_detect.py`, `04b_...` to `04i_...` | pos-proxy, mixture fits, user-bucket test, characterisation, gaps, joint z × gap share, full-train P(random) (`out/train_prandom.npz`), examination curves |
| `05_adjacency.py` | category adjacency order test |
| `06_single_feature.py` | single-feature NDCG (all tie modes) and feed-AUC, train vs val |
| `07_metric.py`, `07b_oracles.py` | full vs @k, M dependence, AUC→NDCG simulation, ties, public/private composition, oracles, time ranges |
| `08a_build.py`, `08_xgb.py`, `08f_subsets.py`, `08g_weighting.py`, `08b_lgbm_train.py` (unused: CPU contention) | model experiments |
| `09_future_history_leak.py`, `10_paired.py`, `10b_blend.py` | leakage check, paired comparisons, blends |

