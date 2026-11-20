# EDA: item/user statistics and collaborative-filtering signals

Avito ML Cup 2026 "Personal ranking". All numbers are on **val** (14,423 feeds, 1.39M candidates) and use the exact
organizer metric (`src/metrics.py`: full-list NDCG, binary gain, pessimistic ties, mean over feeds).
Scripts: `<repo>/eda/collab_stats/` (logs `out_*.log`, artifacts in `out/`).

## 0. TL;DR

* **Ties are a big trap.** The metric uses pessimistic ties, so a feature with many equal values (counts, a category-level CTR,
  "missing" filled with a constant) gets heavily penalized. f1 scores 0.3279 with pessimistic ties and **0.3362 with a random
  tie-break**. f3 goes from 0.2518 to 0.3172, and the in-sample category CTR from 0.2521 to 0.2920. All single-feature numbers
  below therefore use a random secondary key (mean of 3 tie-break seeds). The random baseline is **0.2890** (mean of 20 seeds).
  **Always add a tiny tie-breaker (e.g. f1) to submitted scores.**
* **The strongest CF signal is GPU implicit ALS on deduplicated train+val+test histories.** The candidate is scored by cosine
  similarity to an exponentially decayed profile of the last history items (tau=5). That gives **0.3310 (+0.042 over random)**
  on its own, close to f1 (0.3367). Item2vec (SGNS) peaks at 0.3250 and the ALS user-factor dot product at 0.3210. The ALS
  profile is the only collaborative signal I found that adds clearly to f1 in a plain within-feed rank blend:
  **f1 + ALS profile = 0.3440 (+0.0077 ± 0.0014 over f1)**. Item2vec adds +0.0035 to f1 in the blend and nothing on top of ALS there; inside the GBDT it adds +0.0015.
* **Item target encoding from train is weak on the randomized val traffic.** Position-debiased COEC (+0.0118) beats smoothed
  CTR (paired +0.0022 ± 0.0007). Raw popularity beats both: train click count +0.0158, count of train-history events of
  type 1 **+0.0222**. In a linear rank blend with f1, CTR/COEC hurts (−0.0014) and popularity adds about +0.003. Inside a GBDT the item_te group does help (+0.0045 alone, +0.0014 on top of everything else), mainly through interactions such as mean train position and coverage.
* **Much of what history/category statistics carry is already in f1–f5.** Within-feed rank correlation:
  own-history category affinity vs f1 = 0.68 and vs f3 = 0.78; train-history popularity vs f4 = 0.47.
* **Repeated exposure (same user, same item in train) is negligible.** 1.7% of val rows; overall lift +0.0008 ± 0.0008.
* **Transductive "future history" signal.** In train, **54%** of positives of an earlier feed show up in the history of a *later*
  feed of the same user, against 0.29% of negatives, and 47% of positives come back as event_type 1. In val, a candidate
  found in a later same-user history (later val feeds plus test feeds) is positive 27.6% of the time (17× the base rate). It
  touches 354 feeds and adds **+0.0048** over f1. On test only 334 feeds can use it (within test), worth about +0.002 expected.
* **Train-based coverage decays over time.** The share of candidate rows whose item was a train candidate: val day0 69%,
  val day2 61%, test day0 57%, test day3 51%. Private (last 70% of test) is 52.7% against Public 56.4%. Item statistics built
  only from train will be weaker on Private than on val. ALS/item2vec trained on test histories too are less affected
  (ALS item coverage: val 62.8%, test 55.8%).
* **LightGBM, 5-fold GroupKFold on val** (rank_xendcg, out-of-fold, exact metric):
  * f1–f5 + category + M only: **0.3368** (equal to f1 alone, 0.3367).
  * Adding all my groups: **0.3537** (+0.0169 ± 0.0016). A re-run with a different fold assignment and seeds gives 0.3548 (+0.0183).
  * Best single group: **cf_als +0.0107**, then cf_i2v +0.0078, future_hist +0.0058, item_te +0.0045, pop_train +0.0044,
    pop_transductive +0.0024, own_hist_cat +0.0021, user_cat_train +0.0013, repeat +0.0010.
  * A compact set (base + ALS + train popularity + future_hist) reaches 0.3522, about 90% of the full gain.

Script index:
* `01_basic.py`: overlaps.
* `02a_train_aggs.py`, `02b_eval_item_te.py`, `02c_user_feats.py`, `02d_paired.py`: target encoding.
* `03_popularity.py`: popularity and transductive counts.
* `04a_build_interactions.py`: (user, item) interaction table.
* `04c/04d`: item2vec prep and GPU SGNS. `04e_i2v_profiles.py`: profile variants.
* `04f_als_gpu.py`: GPU CG-ALS. (`04b_als.py` is the stalled CPU `implicit` attempt.)
* `05_repeat_exposure.py`, `05b_target_vs_history_events.py`, `05c_future_history_leak.py`: exposure and future history.
* `06_build_features.py`: feature table. `07*_lgb_*.py`: ablations.
* `08*_blend/corr.py`: redundancy with f1–f5. `09_test_drift.py`: coverage over time.

## 1. Evaluation protocol

* Single features: per-feed NDCG averaged over 3 random tie-break seeds (the feature is the primary key, the random number
  the secondary key). Missing values sort to the bottom of the feed. Lifts are paired per-feed differences against the
  20-seed random baseline (per-feed mean), reported as ± one standard error over the 14,423 feeds.
* Random NDCG varies by seed (0.2878–0.2905 over 5 seeds), so a single random draw is not a reliable baseline.
* Effect of pessimistic ties on raw columns (val):

| feature | unique values | NDCG, pessimistic ties | NDCG, random tie-break |
|---|---:|---:|---:|
| f1 | 1,012,226 | 0.3279 | 0.3362 |
| f2 | 215,165 | 0.2360 | 0.2897 |
| f3 | 132,365 | 0.2518 | 0.3172 |
| f4 | 333,480 | 0.2944 | 0.2977 |
| f5 | 1,132,477 | 0.2952 | 0.2981 |
| category CTR (in-sample on val) | 53 | 0.2521 | 0.2920 |

## 2. Coverage and overlap (01_basic.py)

| quantity | val | test |
|---|---:|---:|
| candidate rows whose item was a train candidate | 65.4% | 53.8% |
| ... seen in train histories | 58.7% | 51.5% |
| ... train candidate or train history | 71.1% | 60.7% |
| ... seen in same-split histories (other feeds) | 8.5% | 14.1% |
| ... candidate in >1 feed of the same split | 18.9% | 29.6% |
| feeds whose user appears in train | 35.0% | 33.2% |
| feeds whose user has a later feed in the same split | 4.7% | 4.5% |

* The median val candidate has 3 train impressions (quantiles 10/25/50/75/90% = 0/0/3/16/58). For test the median is 1.
* Only 0.7% of val users appear in test. Train has 1.28 feeds per user on average (median 1). One user has
  **123,904** train feeds, which looks like a bot or test account, so deduplicate per user before any user-level aggregation.
* 12.7% of test candidate rows are val candidates (val labels could serve as unbiased TE for those rows).
* `item_id` is **not** a time proxy: Spearman(item_id, first train-seen timestamp) = 0.001, and item_id as a score is exactly random (0.2890).
* Train covers days 0–8 and 11–12 from its first timestamp. Days 9, 10 and 13 have only 42, 0 and 34 feeds. Val has data on
  two days only (7,542 and 6,878 feeds).

## 3. Item / category / user target encoding (02a, 02b, 02c, 02d)

**Position curve in train** (CTR by original `pos`): pos0 11.8%, pos1 7.7%, pos2 6.6%, pos5 5.1%, pos10 4.1%, pos20 3.0%,
pos50 1.8%, pos100 1.3%, pos200 0.9%. pos 299 is an artifact at 15.3% (n=9.8K). The curve also depends strongly on feed size
(M<20: base CTR 11.1%; M≥150: 1.1%), because every train feed has at least one positive.

Expected clicks per impression, three definitions:
`Eg` = global position CTR; `Ef` = feed-normalized (feed positives × w(pos) / Σ_feed w(pos)); `Eu` = feed positives / M (no position).
Then COEC = (clicks + a) / (E + a) and CTR_bayes = (clicks + a·p0) / (impressions + a).

| item feature (train, all days) | val NDCG | lift vs random |
|---|---:|---:|
| ctr_raw | 0.3029 | +0.0139 ± 0.0012 |
| ctr_bayes a=5 / 20 / 100 / 500 | 0.2970 / 0.2979 / 0.2986 / 0.2998 | +0.008 to +0.011 |
| COEC_Eg a=20 | 0.3005 | +0.0115 ± 0.0013 |
| **COEC_Ef a=20** | **0.3008** | **+0.0118 ± 0.0013** |
| COEC_Eu a=20 | 0.3002 | +0.0112 ± 0.0013 |
| mean original position (lower=better) | 0.2944 | +0.0054 ± 0.0012 |
| impressions count | 0.3033 | +0.0143 ± 0.0013 |
| clicks count | 0.3048 | +0.0158 ± 0.0013 |

Paired comparisons (5 tie-break seeds):

* COEC_Ef − CTR_bayes(a=100) = **+0.0022 ± 0.0007**. COEC_Eg − CTR_bayes = +0.0019 ± 0.0006. Position debiasing helps a little.
* COEC_Ef − COEC_Eg = +0.0003 ± 0.0004 and COEC_Ef − COEC_Eu = +0.0006 ± 0.0004. The exact debiasing scheme does not matter.
* clicks − COEC_Ef = **+0.0041 ± 0.0010** and COEC_Ef·log1p(n) − COEC_Ef = +0.0041 ± 0.0014. Volume matters more than rate.
* Smoothing: stronger smoothing is monotonically better (a=500 beats a=5 by about 0.003), which confirms the rate component is noisy.
* Coverage strata: feeds where under 40% of candidates have train history get +0.023 (COEC) and +0.048 (clicks). Being seen
  in train is itself informative: target rate is 1.69% for items seen as a train candidate against 1.37% for unseen items,
  and 1.77% against 1.31% for train-history presence.

Staleness and recency windows (COEC_Ef a=2): all 14 days 0.2996. First 9 days only (gap ≥7 days to val, coverage 50%)
0.2971. Last 2 active days only (coverage 54%) 0.2956. Exponential day decay with half-life 1/2/4 days gives
0.2950/0.2967/0.2988, so **no recency weighting is better**. Lift over random is +0.0125 on val day0 and +0.0086 on val day2.

**Category level**: category CTR/COEC from train is 0.2911/0.2938 with tie-break (+0.002 to +0.005). It is useless with
pessimistic ties (0.248). Even the in-sample val category CTR only reaches 0.2920.

**User level** (feeds whose user is in train: 35%): user CTR is constant within a feed and cannot change the ranking by itself;
it only matters through interactions in a model. User × category from train candidate impressions:

| feature | val NDCG | lift | on the 35% feeds with a train user |
|---|---:|---:|---:|
| uc_ctr (a=5, prior = user CTR) | 0.2931 | +0.0041 ± 0.0008 | 0.3006 vs 0.2878 |
| uc_coec | 0.2927 | +0.0038 ± 0.0008 | 0.2997 |
| uc_clicks | 0.2938 | +0.0048 ± 0.0008 | 0.3027 |
| uc_impressions | 0.2940 | +0.0050 ± 0.0008 | 0.3032 |

Reference, user × category from the feed's **own history** (no labels needed, 100% of feeds): count of history events in the
candidate category gives 0.3095 (+0.0206). The last-20 events only give **0.3159 (+0.0269)**. Event type 1 only gives 0.3135.
This beats the train-label version by a wide margin, but it is highly redundant with f1/f3 (Section 6).

## 4. Popularity and transductive counts (03_popularity.py)

| feature | val NDCG | lift vs random |
|---|---:|---:|
| train-history event count, all types | 0.3100 | +0.0210 ± 0.0013 |
| **train-history count, event_type 1** | **0.3112** | **+0.0222 ± 0.0013** |
| ... event_type 2 / 3 / 4 | 0.3033 / 0.3026 / 0.3020 | +0.013 to +0.014 |
| ... event_type 5–10 | 0.292–0.298 | +0.003 to +0.009 |
| last time seen as train candidate (recency) | 0.2982 | +0.0092 ± 0.0012 |
| −age since first train impression | 0.2935 | +0.0045 ± 0.0012 |
| val: number of val feeds with the item as candidate | 0.2875 | −0.0015 ± 0.0011 |
| val: distinct users with the item as candidate | 0.2875 | −0.0014 ± 0.0011 |
| val: history events in val (other feeds) | 0.3005 | +0.0115 ± 0.0012 |
| val: history events in val, event_type 1 | 0.2994 | +0.0104 ± 0.0011 |
| (future split) test history events, scored on val | 0.3038 | +0.0148 ± 0.0012 |
| val LOO item CTR from val labels (18.9% rows covered) | 0.2921 | +0.0031 ± 0.0009 |
| val day0 labels → item CTR for day2 feeds (9.5% covered) | 0.2899 | +0.0012 (day2 only) |

* Paired: train-hist_et1 − clicks = +0.0060 ± 0.0011. Adding clicks on top of hist_et1 gives +0.0000 ± 0.0005.
* Candidate frequency within the split is slightly *negative*. Items shown in many exploration feeds are not better.
* Label-based item CTR from val adds little (+0.003 in-sample LOO). For test, val labels cover only 12.7–14.3% of rows.

## 5. Collaborative filtering (04a–04f)

**Interaction matrix** (04a): train+val+test histories (596.5M events) plus train positive impressions (5.43M), deduplicated
by (user, item) into **345.5M pairs** with a bitmask of event types. For ALS, items are kept if they have ≥2 users or are a
val/test candidate: 2.48M users × 47.4M items, 296.9M nnz. Confidence c = 1 + alpha · w, with
w = 1 + (#distinct event types) + 3·[train positive] ("evt") or w = 1 ("bin").

**ALS**: the CPU `implicit` library stalled. It ran more than 45 minutes without finishing 12 iterations, stuck on lock
contention: the machine load average was 450–600 on a shared machine. I replaced it with my own CG-ALS (Hu-Koren-Volinsky, 3 CG
steps) in torch on GPU (`04f_als_gpu.py`), at **about 20 s per iteration** for f=64 (12 iterations in 7 minutes).

| model | scoring | val NDCG | lift vs random |
|---|---|---:|---:|
| ALS f64 evt a=2 | user·item dot | 0.3195 | +0.0305 ± 0.0014 |
| | cos(item, mean of last 5 / 20 / 256 history items) | 0.3272 / **0.3298** / 0.3260 | +0.038 / +0.041 / +0.037 |
| ALS f64 bin a=2 | last-20 profile | 0.3273 | +0.0383 |
| ALS f128 evt a=5 | user·item dot | 0.3210 | +0.0320 ± 0.0014 |
| | last-5 / last-10 / last-20 profile | 0.3285 / 0.3302 / 0.3298 | +0.040 to +0.041 |
| | **exp-decay profile tau=5** | **0.3310** | **+0.0420 ± 0.0015** |
| | exp-decay tau=2 / 10 / 30 | 0.3299 / 0.3303 / 0.3285 | |
| | event_type 1 only (tau=5) / other types only (tau=10) | 0.3300 / 0.3227 | |
| | same-category profile (tau=10, coverage 52.7%) | 0.3250 | +0.0360 |
| item2vec d64, 4 epochs, + next-item pairs | cos(item, mean last 5) | 0.3148 | +0.0258 |
| item2vec d64, 4 epochs, no next-item pairs | cos last 5 | 0.3163 | +0.0273 |
| item2vec d128, 10 epochs, + next-item pairs | cos last 5 | 0.3201 | +0.0311 |
| **item2vec d128, 10 epochs, no next-item pairs** | cos last 5 / max-cos last 5 / in·out last 5 | 0.3223 / 0.3192 / 0.3068 | +0.033 / +0.030 / +0.018 |
| | same-category profile | **0.3250** | +0.0360 ± 0.0014 |

* Item2vec is my own SGNS in torch on GPU: window ±5 over each user's latest history per split (388.7M tokens, vocabulary
  40.4M items with count ≥3, or any val/test candidate), 5 negatives drawn from unigram^0.75, SparseAdam. It takes about
  2.5 minutes per 4 epochs at d64 and 12 minutes for 10 epochs at d128.
* "Next-item" pairs (train positive candidate ↔ last-10 history items, 38.8M pairs) do **not** help (−0.0015 at d64, −0.0022 at d128).
* Implicit ALS beats item2vec: best ALS profile 0.3310 against best item2vec 0.3250/0.3223, a gap of +0.006 to +0.009. ALS profile scoring (item-item) is clearly better
  than the user-factor dot product (+0.010).
* Coverage of ALS factors: val 62.8%, test 55.8% (Public 57.8%, Private 54.9%). Item2vec: val 61.3%, test 54.6%.

## 6. Redundancy with f1–f5 and simple blends (08_blend.py, 08b, 08c)

Within-feed rank correlation (Pearson on within-feed percentile ranks):

| feature | f1 | f2 | f3 | f4 | f5 |
|---|---:|---:|---:|---:|---:|
| own-history category count, last 20 | **0.678** | 0.084 | **0.780** | 0.026 | 0.088 |
| train-history count, event 1 | 0.016 | 0.005 | 0.046 | **0.471** | 0.313 |
| ALS exp-decay profile | 0.118 | – | 0.127 | 0.410 | 0.329 |
| item2vec last-5 profile | 0.151 | 0.030 | 0.161 | 0.334 | 0.334 |
| item clicks (train) | 0.032 | 0.177 | 0.138 | 0.098 | 0.062 |
| item COEC (train) | 0.034 | 0.068 | 0.061 | −0.162 | −0.043 |
| user × category impressions (train) | 0.245 | 0.217 | 0.388 | −0.008 | 0.032 |

Lift over f1 alone (0.3363) for f1_rank + w·feature_rank, with w chosen on val from {0.1, 0.25, 0.5, 1}. This is optimistic
by about the grid choice; the paired SE is shown:

| added to f1 | best w | lift over f1 |
|---|---:|---:|
| **ALS exp-decay profile** | 1.0 | **+0.0077 ± 0.0014** |
| ALS last-10 profile | 0.75 | +0.0068 ± 0.0013 |
| ALS same-category profile | 0.75 | +0.0062 ± 0.0013 |
| item2vec last-5 profile | 0.5 | +0.0035 ± 0.0011 |
| item2vec same-category profile | 0.5 | +0.0033 ± 0.0011 |
| train-history count, event 1 | 0.25 | +0.0031 ± 0.0009 |
| item clicks (train) | 0.25 | +0.0008 ± 0.0009 |
| own-history category count, last 20 | 0.1 | −0.0001 ± 0.0003 |
| item COEC (train) | 0.1 | −0.0014 ± 0.0006 |
| user × category impressions | 0.1 | −0.0005 ± 0.0005 |
| f3 | 0.1 | −0.0001 ± 0.0003 |
| item2vec on top of f1 + 0.5·ALS | 0.25 | −0.0002 |
| train-hist event 1 on top of f1 + 0.5·ALS | 0.25 | +0.0006 ± 0.0006 |

## 7. LightGBM ablation (5-fold GroupKFold on val)

Setup (`07_lgb_ablation.py`, `07c_lgb_als_addon.py`):
* Out-of-fold predictions over all 14,423 val feeds, scored with the exact metric. Same folds for every config, so the
  comparisons are paired per feed.
* LightGBM `rank_xendcg`, 200 rounds, lr 0.05, 31 leaves, min_data_in_leaf 100, max_bin 63, feature/bagging fraction 0.8,
  lambda_l2 10.
* Training folds keep all positives and 30% of negatives, for speed on the shared CPU. Round count checked on one fold:
  100/200/300/400/600 rounds gave 0.3326/0.3331/0.3327/0.3318/0.3289 for base.
* All train-derived features use only train labels, so there is no leakage into val. Within-val transductive features use
  no val labels.

Feature groups:

* base = f1..f5, category (categorical), M.
* item_te = train clicks, impressions, CTR_bayes, COEC_Ef, mean pos.
* pop_train = train-history counts (all, event 1–4), age since first/last seen.
* pop_transductive = val candidate-feed count, distinct users, val history counts.
* user_cat_train = user CTR, user feeds, user × category CTR/clicks/impressions/COEC.
* own_hist_cat = own-history category counts (all, event 1, last 20), share, H.
* cf_i2v = 8 item2vec d128 profile scores.
* cf_als = ALS f128 dot, last-20, exp-decay, same-category.
* repeat = n_prev, pos_prev.
* future_hist = later-history match count.

| config | OOF NDCG | Δ vs base | Δ vs reference |
|---|---:|---:|---:|
| f1 only (random tie-break) | 0.3367 | −0.0000 ± 0.0016 | |
| **base** (f1–f5, cat, M) | **0.3368** | 0 | |
| base + item_te | 0.3413 | +0.0045 ± 0.0013 | |
| base + pop_train | 0.3411 | +0.0044 ± 0.0013 | |
| base + pop_transductive | 0.3391 | +0.0024 ± 0.0011 | |
| base + user_cat_train | 0.3380 | +0.0013 ± 0.0010 | |
| base + own_hist_cat | 0.3388 | +0.0021 ± 0.0011 | |
| base + cf_i2v | 0.3445 | +0.0078 ± 0.0014 | |
| **base + cf_als** | **0.3474** | **+0.0107 ± 0.0014** | |
| base + cf_als + cf_i2v | 0.3490 | +0.0122 ± 0.0015 | |
| base + cf_i2v + pop_train | 0.3471 | +0.0104 ± 0.0015 | |
| base + repeat | 0.3377 | +0.0010 ± 0.0010 | |
| base + future_hist | 0.3426 | +0.0058 ± 0.0011 | |
| **base + cf_als + pop_train + future_hist** | **0.3522** | +0.0154 ± 0.0015 | −0.0015 ± 0.0011 vs ALL+als |
| ALL without ALS | 0.3505 | +0.0137 ± 0.0015 | |
| **ALL + ALS** | **0.3537** | **+0.0169 ± 0.0016** | +0.0032 ± 0.0011 vs ALL without ALS |
| ALL+ALS − cf_i2v | 0.3522 | | −0.0015 ± 0.0010 |
| ALL+ALS − item_te | 0.3523 | | −0.0014 ± 0.0010 |
| ALL+ALS − own_hist_cat − user_cat_train | 0.3528 | | −0.0009 ± 0.0010 |
| ALL+ALS − pop_train − pop_transductive | 0.3537 | | +0.0000 ± 0.0010 |
| ALL(no ALS) − cf_i2v | 0.3462 | | −0.0042 ± 0.0012 |
| ALL(no ALS) − future_hist | 0.3497 | | −0.0008 ± 0.0011 |
| ALL(no ALS) − item_te / − pop_train / − own_hist / − user_cat | 0.3514 / 0.3508 / 0.3509 / 0.3506 | | +0.0009 / +0.0004 / +0.0005 / +0.0001 (all within noise) |
| ALL(no ALS) − repeat / − pop_transductive | 0.3522 / 0.3522 | | +0.0017 ± 0.0009 / +0.0018 ± 0.0010 (removing them *helps*) |

Gain shares in ALL+ALS, top 12: M 0.142, cat 0.133, f1 0.059, f5 0.051, f3 0.040, own-hist cat last-20 0.037,
ALS same-category 0.034, f4 0.033, ALS exp-decay 0.029, item mean pos 0.027, i2v same-category 0.026, ALS dot 0.025.
Pessimistic-tie NDCG of the OOF predictions is 0.35369; with an f1 tie-break it is 0.35370, so the model outputs barely tie.

Reading the ablation:
* With only 11.5K training feeds per fold, the GBDT on f1–f5 does not beat f1 alone (0.3368 vs 0.3367).
  The val-only training set is small. A train-trained model will have a different baseline, but the *relative* value of
  the groups should carry over.
* The CF group is the main contributor: cf_als +0.0107 alone, and +0.0032 on top of everything else.
  Item2vec is partly redundant with ALS (+0.0015 on top).
* future_hist (+0.0058 alone) and item_te/pop_train (+0.0045 each alone) help on their own but overlap heavily with each
  other and with CF. Each adds ≤0.0015 to the full set.
* The compact set base + cf_als + pop_train + future_hist (24 features) gets 91% of the full gain.
* repeat and pop_transductive look like overfitting noise in this small-data regime.

**Robustness re-run** (`07d_seed_check.py`: different group-to-fold assignment, negative-subsample seed and LGB seed):

| config | run 1 | run 2 |
|---|---:|---:|
| base | 0.3368 | 0.3365 |
| base + cf_als | 0.3474 (+0.0107) | 0.3473 (+0.0107) |
| base + cf_i2v | 0.3445 (+0.0078) | 0.3435 (+0.0070) |
| base + cf_als + pop_train + future_hist | 0.3522 (+0.0154) | 0.3515 (+0.0150) |
| ALL without ALS | 0.3505 (+0.0137) | 0.3513 (+0.0148) |
| ALL + ALS | 0.3537 (+0.0169) | 0.3548 (+0.0183) |
| ALS on top of ALL | +0.0032 ± 0.0011 | +0.0035 ± 0.0011 |

The group effects reproduce to within ±0.001.

## 8. Repeated exposure and the "future history" signal (05, 05b, 05c)

**Repeated exposure** (val user shown the same item in an earlier train feed): 1.70% of val rows and 12.9% of feeds.

| prior train exposure of (user, item) | rows | target rate |
|---|---:|---:|
| never shown | 1,366,306 | 1.58% |
| shown before, never positive | 22,200 | 1.75% |
| shown before and positive before | 1,465 | **3.82%** |
| shown ≥5 times, never positive | 4,075 | 1.89% |

NDCG lift: n_prev +0.0008 ± 0.0008, pos_prev +0.0008 ± 0.0007 overall. On the 12.9% of feeds with at least one
previously-shown candidate: 0.2575 vs random 0.2481. Repeated negative exposure does **not** lower the rate, and
re-shown positives are 2.4× the base rate, but the volume is tiny. Within val itself, 1,410 rows had an earlier val
impression of the same (user, item), with target rate 0.85%.

**Target action ↔ history events** (train sample: 60K users with 2–20 feeds, 141K feeds, 11.4M candidates).
For each candidate, I looked for the (user, item) pair in the history of the user's other feeds:

| | positives (n=266,636) | negatives (n=11,156,035) |
|---|---:|---:|
| appears in a **later** feed's history (any event) | **54.4%** | 0.29% |
| ... as event_type 1 | 47.2% | 0.24% |
| ... as event_type 2 / 3 | 5.6% / 5.1% | 0.04% / 0.03% |
| ... as event_type 4 / 5 / 6 | 2.5% / 2.3% / 2.1% | ≤0.013% |
| appears in an **earlier** feed's history | 0.09% | 0.07% |

So the target action is almost certainly logged as a history event, mostly event_type 1, in later feeds. Items the user
already interacted with essentially never appear as candidates (0.07–0.09%).

**Transductive use on val/test**: score = number of events of (user, candidate) in histories of the same user's later feeds
in the same split (for val also test feeds, which are later in time).
* val: 681 candidate rows matched; target rate **27.6%** against 1.58% base. They cover 0.86% of all val positives across 354 feeds.
  Alone: +0.0054 ± 0.0009 over random. As a lexicographic key ahead of f1: 0.3367 → 0.3416 (**+0.0048**).
* test (within-test only): 558 candidate rows in 334 feeds (1.2% of feeds) can be flagged. That is roughly **+0.002**
  on the test metric, extrapolating the val gain per affected feed (+0.196 per feed).
* The same mechanism makes the history of later train feeds leak the labels of earlier train feeds. This is harmless
  for features computed from a feed's own history, which precedes the feed. It must be avoided for user-level aggregates
  built from *all* of a user's histories, such as ALS user factors on train rows if they are used to train a model on train.

## 9. Recommendations

1. Use the **ALS (GPU, evt weights, f128, alpha=5) exp-decay profile cosine** as a feature. Also add the last-10/20 profile,
   the same-category profile and the user·item dot product. They are cheap (7 min training). They are the only
   collaborative signals that add clearly to f1 (+0.0077 in a rank blend, +0.0107 as a GBDT group, +0.0032 on top of all other groups). Retrain including val+test histories for the final model; it was trained
   that way here.
2. Add **train popularity counts**: train-history counts by event type (type 1 especially), train clicks and impressions.
   Prefer these over CTR/COEC. If a rate is needed, use COEC with feed-normalized position expectation and strong smoothing
   (a≥20); it is about +0.002 better than smoothed CTR.
3. For test, refresh popularity and TE statistics with **val labels and val+test histories**: coverage of train-only
   statistics drops to 51–53% on Private.
4. Add the **future-history flag** for test feeds whose user has a later test feed (about +0.002).
5. Skip feed-constant user statistics, repeat-exposure counts and category CTRs as standalone signals. Their lift is ≤0.005
   or redundant with f1/f3.
6. Always break ties in submitted scores, for example score + 1e-9·rank(f1).

## 10. Risks and pitfalls

* **Pessimistic ties**: count features and category/user aggregates are full of ties. Model outputs from coarse trees can
  tie too, so add a tie-breaker.
* **Train-label TE leakage if the ranker is trained on train**: item/user TE features for train rows must be out-of-fold,
  for example time-based (past days only). Otherwise the model over-trusts them. The val-CV here avoids this because all TE
  comes from train and is evaluated on val.
* **Position bias and selection bias in train**: CTR from production-ranked train traffic transfers poorly to the
  randomized val/test traffic (CTR/COEC +0.011 only, and negative in blends).
* **Temporal decay**: train coverage falls from 69% (val day0) to 51% (test day3). Private is the later 70%. Expect
  train-only statistics to be worth less on Private than on val.
* **Transductive count features differ by split size**: test has 2× the feeds and history rows of val, so raw counts within
  the split are not on the same scale. Normalize by split totals if a model trained on val is applied to test.
* **Bot users**: one user has 123,904 train feeds. Deduplicate per user, or cap per-user weight, for user-based aggregates and CF.
* **Noise**: val NDCG SE is about 0.0013 for single features (paired differences 0.0004–0.0015). Differences under about
  0.002 between variants are within noise unless paired.
