# EDA: user and item embeddings (Avito ML Cup 2026, Personal ranking)

All NDCG numbers are **full-list NDCG on val** (14,423 feeds), computed with the exact metric (`src/metrics.py`: binary gain,
pessimistic ties, mean over feeds). Unless stated otherwise, a single signal is scored with **random tie-breaking**: undefined
values (no history, zero user vector) go to the bottom of the feed and ties are broken at random. Pessimistic ties would otherwise
put every positive last in a feed where the score is constant. The exact expected NDCG of a random order is **0.2890**. The f1 baseline is **0.3366**.
"vs X (SE)" is a paired per-feed difference with its standard error. With 14.4K feeds, a difference of about 0.003 or less is noise.

Scripts: `<repo>/eda/embeddings/` (logs `*.log` next to each script; cached features in `cache/`).

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **Both towers live in one space, and every vector is unit-norm.** Item and user vectors are L2-normalised and then multiplied by 256
   (norm q01..q99 = 255.4..256.7). So **dot = 65536 x cosine**, and the **item norm carries no information** (NDCG 0.2895 vs random 0.2890).
   No int8 values saturate (no -128 anywhere; 127 appears in 0.01-0.02% of rows).
2. **The user embedding comes from a real-time two-tower user model.** It is compatible with the item space:
   `dot(user, item)` reaches **0.3432** (+0.054 over random, +0.0066 over f1). cos(user, mean of last 20 history items) = 0.70, against 0.32 for a shuffled user.
   The vector is recomputed for every request: 0.1% of consecutive same-user feeds have an identical vector, cos = 0.94 within 10 min and 0.80 when 2+ days apart.
3. **History similarity is the strongest embedding signal, and recency is what matters.** The recency-weighted mean cosine to history
   (exp decay with a half-life of 3-5 events) gives **0.3493** (+0.0127 +/- 0.0017 over f1). This is the best single feature we found, f1 included.
   last-5 mean = 0.3474, max-sim = 0.3436, mean over all history = 0.3294 (much worse). Adding the user dot to the history sim (z-scored)
   gives 0.3532; adding f1 to both gives **0.3553**, with no training.
4. **Near-duplicates of recent history are strong positives.** For candidates with max cos to history >= 0.99 the positive rate is 2.8x the base rate (1.9x for 0.95-0.99).
   Below cos 0.6 it is 0.27x.
5. **Learned scorers.** A tiny DIN-style target-attention model trained on train (about 330 parameters: event_type, event_source and recency biases, attention
   temperature, user term) reaches **0.3584** val (+0.0152 +/- 0.0014 over the user dot). A full bilinear `u^T W c` adds only **+0.006**
   over the identity dot (train-fit 0.3496, val-CV 0.3485). It also **hurts (0.3386) unless the logit scale is learnable**: a listwise softmax over raw cosines
   is nearly uniform, so its gradient is misaligned with NDCG.
6. **Model lift.** In a LightGBM lambdarank trained on a 200K-feed train subsample, f1..f5 + category reach 0.3303 on val (below f1 alone: the train-to-val
   shift hits the f-features). Adding 8 handcrafted embedding features gives **0.3617 (+0.0314 +/- 0.0016)**, and all 39 give **0.3651 (+0.0348)**.
   The 39 embedding features alone, without f1..f5 or category, reach 0.3588.
   Val 5-fold CV (unbiased labels, 11.5K training feeds per fold) gives +0.0225 for the 8 features, plus a further **+0.0044 +/- 0.0010** from adding the train-fitted
   attention and bilinear scores (0.3601).
7. **Item embeddings mostly encode category, not "item quality".** A linear probe recovers category_id with 98.5% accuracy (kNN-1: 95.5%).
   The embeddings partly encode f1/f2/f3 (kNN R2 0.31/0.30/0.35, category-mean R2 0.21/0.06/0.25) and only weakly f4/f5 (0.10/0.16).
   The item-level target rate cannot be predicted from the embedding (kNN R2 < 0, corr 0.02). **The signal lies entirely in user/history-relative similarity.**
8. **Stability across splits.** Each split has its own table, but for item_ids shared with train, 83% (val) / 72% (test) of the vectors are bit-identical
   and the mean cosine is 0.999. Val and test feature distributions are practically identical. Train candidates sit closer to history
   (median max-sim 0.875 vs 0.829), consistent with production-ranked traffic.

---------------------------------------------------------------------------------------------------------------------

## 1. Embedding statistics (`01_stats.py`)

### 1.1 Value distributions

| table | rows (sampled) | mean / std of int8 values | P(0) | min / max | rows with any value at 127 | L2 norm q0/q1/q50/q99/q100 | random-pair cos | PCA top-10 / top-64 var | dims for 90% var |
|---|---|---|---|---|---|---|---|---|---|
| item train | 81.56M (2M) | 6.62 / 21.64 | 2.10% | -115 / 127 | 0.02% | 250 / 255.4 / 256.0 / 256.7 / 257.4 | 0.306 | 0.283 / 0.773 | 90 |
| item val | 2.52M (all) | 6.75 / 21.60 | 2.13% | -111 / 127 | 0.01% | 249 / 255.4 / 256.0 / 256.7 / 257.4 | 0.309 | 0.279 / 0.761 | 92 |
| item test | 4.71M (2M) | 6.74 / 21.60 | 2.12% | -116 / 127 | 0.01% | 251 / 255.4 / 256.0 / 256.7 / 257.5 | 0.307 | 0.278 / 0.761 | 92 |
| user train | 3.15M (1M) | 8.46 / 20.68 | 4.33% | -125 / 127 | <0.005% | 0 (2.5% all-zero) / ... / 256.0 / 256.7 | 0.391 | 0.281 / 0.735 | 97 |
| user val | 14,423 | 8.60 / 20.76 | 3.23% | -119 / 109 | 0 | 0 (1.42% all-zero) / ... / 256.0 / 256.7 | 0.406 | 0.277 / 0.734 | 97 |
| user test | 28,595 | 8.68 / 20.86 | 2.12% | -116 / 127 | <0.005% | 0 (0.28% all-zero) / ... / 256.0 / 256.7 | 0.413 | 0.275 / 0.733 | 97 |

* Every non-zero vector has norm 256 +/- 0.3. The vectors are unit vectors quantised with scale 256 (per-element |v| q99.9 = 91, so int8 range is not a constraint).
  **Consequences:** cosine = dot/65536; the item norm is pure quantisation noise (NDCG 0.2895); the L2 distance is a monotone function of the dot
  (NDCG 0.3433 vs 0.3432).
* **Strong anisotropy.** The per-dimension means range from -52 to +66, ||mean vector|| / mean norm = 0.55 (items) and 0.64 (users), and a random item pair has cos 0.31.
  Centring is irrelevant for within-feed ranking by user dot, because it subtracts a per-feed constant. It does matter for max/softmax pooling and for NN input normalisation.
* The participation ratio is about 64-67 of 128 dimensions, so the space is fully used and no low-rank shortcut exists.
* **All-zero user vectors** (1.42% val, 0.28% test, 2.5% train) mean "no user vector": cos is undefined, so use NaN rather than 0. The zero vector is only
  partly explained by an empty history: P(H=0 | zero user) = 0.36 val / 0.71 test / 0.09 train, and P(zero user | H=0) = 0.38 / 0.23 / 0.21.
  **62% of val feeds with no history still have a valid user vector.** In those feeds the user dot is the best signal (+0.053 over random on 194 feeds; history signals do nothing there).

### 1.2 Missing embeddings

| | candidate emb_idx = -1 | history emb_idx = -1 |
|---|---|---|
| train | 0.003% | 15.5% |
| val | 0.015% | 14.2% |
| test | 0.016% | 14.0% |

History missing rate by event_type (val): 1: 5.8%, 2: 20%, 3: 16%, 4: 24%, 5: 19%, 6: 20%, 7: 25%, 8: 26%, **9: 33%**, 10: 1.3%.
Missing history items are real item_ids (none is 0), with 222K distinct missing items in val. They are probably items that were deleted or never embedded.
**Mask them; do not pool zero vectors.**

### 1.3 item_id <-> emb_idx mapping and cross-split identity

* Within each split the mapping is an **exact bijection**: train 81,557,944 items = 81,557,944 idx = table rows; val 2,516,711; test 4,710,948.
  Each table holds exactly the items referenced in that split (candidates + history). **emb_idx is split-local. Join across splits by item_id.**
* Shared items:

| pair | shared item_ids | bit-identical vectors | mean cos | cos q01 / q10 |
|---|---|---|---|---|
| val vs train | 2.03M (80.7% of val items) | 82.8% | 0.9991 | 0.985 / 0.9998 |
| test vs train | 3.42M (72.6% of test items) | 72.1% | 0.9986 | 0.972 / 0.9992 |
| val vs test | 0.54M | 79.3% | 0.9991 | 0.984 / 0.9996 |

  This is consistent with periodic snapshots from the same model, where a minority of items drift slightly (probably edited content). It is safe to treat them as one space,
  for example to pre-train on train and apply to val/test.
* Candidate novelty: 71.0% (val) / 60.6% (test) of unique candidate items appear somewhere in train (as a candidate or in a history). Only 6.8% / 10.5% appear in
  any same-split history. Pure item-ID models will be cold for 30-40% of test candidates. **Content embeddings are the main item
  representation for unseen items.**

### 1.4 Is the user embedding constant per user?

No. It is recomputed for every request (feed):

| same-user consecutive feed pairs (train, 135K pairs from 100K users) | identical vector | mean cos | cos q10 |
|---|---|---|---|
| all | 0.10% | 0.824 | |
| dt < 10 min | 1.2% | 0.940 | 0.942 |
| dt < 1 h | 0.6% | 0.920 | 0.890 |
| same day | 0.21% | 0.890 | 0.816 |
| next day | 0.08% | 0.854 | 0.759 |
| 2+ days | 0.08% | 0.800 | 0.683 |
| val feed vs the user's last train feed | 0.04% | 0.749 | 0.549 |
| random user pair | | 0.391 | |

User-days with more than one feed have a distinct vector per feed in 99.8% of cases.
Users rarely repeat: 13.7K users across 14.4K val feeds and 27.3K users across 28.6K test feeds. 35% of val feeds and 33% of test feeds come from users seen in train, and 4.5% of test feeds from users seen in val.

### 1.5 What does the user tower look at? (`03_user_vs_history.py`)

cos(user, normalised mean of the last-k valid history item vectors), val:

| k | 1 | 2 | 5 | 10 | **20** | 50 | 100 | 256 | shuffled user (k=10) |
|---|---|---|---|---|---|---|---|---|---|
| mean cos | 0.565 | 0.609 | 0.656 | 0.682 | **0.698** | 0.697 | 0.690 | 0.685 | 0.323 |

By event type (all events of the type): type 1 = 0.683, type 3 = 0.636, type 10 = 0.635, type 2 = 0.626, type 4 = 0.593, types 5-9 = 0.53-0.57.
A linear map from [last1, mean last5, mean last20, mean all] to the user vector reconstructs it with held-out R2 = 0.56 (cos 0.86).
**Reading:** the user vector is a sequence encoding of roughly the last 20 events, weighted toward the most frequent event type (type 1, probably views), plus information not in the
256-event history. That extra information explains the valid vectors when H = 0. The user vector and history similarity are correlated (within-feed Spearman of
user_dot with rec_hl20 = 0.67) but complementary (Section 2.4).

---------------------------------------------------------------------------------------------------------------------

## 2. Predictive power on val (`02_val_signals.py`, `03_user_vs_history.py`, `09_bins.py`)

Expected random 0.2890; f1 0.3366. SE of the difference vs random is about 0.0015 and vs f1 about 0.0017.
"cov" = share of feeds where the signal is non-constant.

### 2.1 Single signals (best sign shown; every signal is "higher = better")

| signal | NDCG | vs random | vs f1 | cov |
|---|---|---|---|---|
| **rec_hl3**: sum_j w_j cos(c, h_j), w_j = exp(-j/3) over valid events j | **0.3493** | +0.0603 | **+0.0127** | 0.986 |
| rec_hl5 | 0.3492 | +0.0603 | +0.0126 | 0.986 |
| rec_hl2 / hl8 / hl12 / hl1 | 0.3488 / 0.3479 / 0.3459 / 0.3454 | | | |
| last5_cos (cos to normalised mean of last 5) | 0.3474 | +0.0584 | +0.0108 | 0.986 |
| last10_cos / last20_cos / last50_cos | 0.3458 / 0.3444 / 0.3381 | | | |
| rec_hl20 / rec_hl50 | 0.3438 / 0.3378 | | | |
| max_sim (max cos over history) | 0.3436 | +0.0547 | +0.0070 | 0.986 |
| -L2(user, item) | 0.3433 | +0.0544 | +0.0067 | 1.0 |
| **user_dot** | **0.3432** | +0.0543 | +0.0066 | 0.986 |
| user_cos | 0.3430 | +0.0540 | +0.0063 | 0.986 |
| type1_max (max cos over type-1 events) | 0.3426 | +0.0537 | +0.0060 | 0.969 |
| top5_sim (mean of top-5 cos) | 0.3424 | +0.0534 | +0.0058 | 0.986 |
| softmax pooled sim, tau = 0.1 / 0.2 / 0.05 | 0.3414 / 0.3404 / 0.3375 | | | |
| last1_cos | 0.3398 | +0.0508 | +0.0032 | 0.986 |
| n_sim_gt0.9 (count of history events with cos > 0.9) | 0.3335 | +0.0445 | -0.0032 | 0.971 |
| type1_mean | 0.3335 | +0.0445 | -0.0032 | 0.969 |
| hmean_cos (mean of all history) = mean_sim | 0.3294 | +0.0404 | -0.0072 | 0.986 |
| **centrality_loo** (cos to the mean of the other candidates in the feed) | 0.3170 | +0.0280 | -0.0197 | 1.0 |
| type3_max / type2_max / type4_max | 0.3170 / 0.3140 / 0.3089 | | | 0.90 / 0.85 / 0.77 |
| same-category centrality (LOO) | 0.3077 | +0.0187 | | 0.988 |
| type5..type9 max/mean | 0.292-0.305 | +0.003..+0.016 | | 0.39-0.64 |
| type10_mean (only 6.7% of feeds; on those feeds 0.333 vs random 0.287) | 0.2917 | +0.0027 | | 0.067 |
| item_norm | 0.2895 | +0.0005 | -0.047 | 1.0 |

Observations:
* **Recency dominates.** The optimal exponential half-life is 2-5 events. A uniform mean over the whole history loses 0.02 NDCG.
* Per-event-type sims are weaker on their own, mostly because of coverage. On covered feeds, type 10 is the most informative (+0.046 over random on 6.7% of feeds).
* The **within-feed centrality** of a candidate is a real positive signal (+0.028). Candidates on the main theme of the retrieved set are more often positive.
  Its correlation with f1 is low (Spearman 0.29), so it is a useful diverse feature.
* Within-feed Spearman with f1/f3: user_dot 0.47/0.42, rec_hl20 0.49/0.45, max_sim 0.37/0.31, centrality 0.29/0.29; with f4/f5 all below 0.15.
  f1/f3 are partly "embedding-like" scores, but much of the information is not shared.

### 2.2 Positive-rate profiles

| max cos to history | share of cands | positive rate | lift |
|---|---|---|---|
| < 0.6 | 8.6% | 0.42% | 0.27x |
| 0.6-0.7 | 12.1% | 0.57% | 0.36x |
| 0.7-0.8 | 20.6% | 1.03% | 0.65x |
| 0.8-0.9 | 29.1% | 1.72% | 1.09x |
| 0.9-0.95 | 15.5% | 2.32% | 1.46x |
| 0.95-0.99 | 11.4% | 2.94% | 1.86x |
| >= 0.99 | 1.2% | 4.38% | **2.77x** |

Within-feed decile lift, top to bottom: user_dot 1.98, 1.54, 1.43, 1.09, 0.96, 0.87, 0.65, 0.56, 0.47, 0.34; rec_hl5 2.13 ... 0.44; f1 1.89 ... 0.39.

### 2.3 Lift by history length and feed size (NDCG minus exact expected random)

| H | feeds | user_dot | rec_hl5 | max_sim | centrality | f1 |
|---|---|---|---|---|---|---|
| 0 | 194 | **+0.053** | +0.015 | +0.015 | +0.037 | +0.047 |
| 1-9 | 285 | +0.067 | **+0.089** | +0.085 | +0.045 | +0.057 |
| 10-49 | 1051 | +0.071 | +0.070 | +0.063 | +0.039 | +0.059 |
| 50-149 | 3766 | +0.058 | +0.066 | +0.059 | +0.030 | +0.050 |
| 150-255 | 4951 | +0.052 | +0.058 | +0.055 | +0.026 | +0.046 |
| 256 | 4176 | +0.050 | +0.055 | +0.048 | +0.025 | +0.043 |

The absolute lift shrinks with M (from +0.08 at M < 20 to +0.035 at M >= 200), and the ordering of the signals is the same in every M bucket.
Val first 30% vs last 70% of feeds by time (a proxy for public/private): rec_hl5 0.3540 / 0.3472, user_dot 0.3481 / 0.3415, f1 0.3429 / 0.3334.
All signals drop by a similar amount, so the gap is a property of the feeds, not of the embeddings.

### 2.4 Simple training-free fusions (within-feed z-scores summed)

| fusion | NDCG | vs f1 |
|---|---|---|
| user_cos + rec_hl3 | 0.3532 | +0.0166 (0.0017) |
| f1 + user_cos | 0.3493 | +0.0127 (0.0014) |
| f1 + rec_hl3 | 0.3517 | +0.0150 (0.0014) |
| **f1 + user_cos + rec_hl3** | **0.3553** | **+0.0187 (0.0015)** |

Additions on top of f1 + user + rec_hl5 (0.3548; `03b_fusion.py`): + 0.5 max_sim gives **0.3569** (+0.0022 +/- 0.0005), + 0.5 centrality gives 0.3514
(-0.0033 +/- 0.0006). In a *linear* fusion, centrality is redundant or harmful. Its value only shows up in a tree model, where it gets 6% of the gain in compact8 (Section 5).

---------------------------------------------------------------------------------------------------------------------

## 3. Two-tower compatibility and learned maps (GPU; `06_bilinear.py`, `06b`, `06d`, `06c`)

All models use a listwise softmax cross-entropy per feed (target distribution uniform over positives). They are trained either on a
**200K-feed random train subsample** (15.6M candidates) or by **5-fold CV on val** (out-of-fold predictions).
"Held-out train" = 20K train feeds not used for fitting.

### 3.1 Bilinear u^T W c

| model | logit scale | held-out train | val | vs dot |
|---|---|---|---|---|
| dot (W = I, no training) | n/a | 0.3570 | 0.3432 | 0 |
| bilinear, train-fit, lr 1e-4 | **fixed (raw cos)** | 0.3504 | 0.3248 | -0.018 |
| bilinear, train-fit, lr 1e-3 | fixed | 0.3592 | 0.3392 | -0.004 |
| bilinear, val 5-fold CV | fixed | | 0.3367 | -0.0065 (0.0016) |
| **bilinear, train-fit** | **learnable (init 20, ends 13.3)** | 0.3613 | **0.3496** | **+0.0064 (0.0011)** |
| bilinear + item bias b^T c, train-fit | learnable | 0.3617 | 0.3498 | +0.0066 (0.0011) |
| bilinear, val 5-fold CV | learnable | | 0.3485 | +0.0053 (0.0010) |
| bilinear + bias, val 5-fold CV | learnable | | 0.3485 | +0.0053 (0.0010) |
| history query q = W u + sum_k A_k pool_k (+ bias), train-fit | learnable | 0.3658 | 0.3511 | +0.0079 (0.0014) |
| + MLP query/item towers ("deep"), train-fit | learnable | 0.3667 | 0.3490 | +0.0058 (0.0015) |
| history query, val 5-fold CV | learnable | | 0.3423 | -0.0009 (overfits: about 98K params on 11.5K feeds) |

* **The user tower is compatible with the item space**: the identity dot already gives +0.054 over random. A learned full-rank W adds only about +0.006,
  and the item bias b^T c adds nothing (consistent with Section 4: there is no global "attractiveness" direction).
* **Pitfall:** with cosine logits in [-1, 1] the softmax is nearly uniform. The gradient then approximates mean(pos) - mean(all), and the model reshapes W (eigenvalues 0.2-6.2)
  in a way that lowers NDCG even on held-out train. **Always use a learnable temperature (scale of about 13-20)** for listwise losses on cosine scores.
* Fitting on train transfers to val about as well as fitting on val itself (+0.0064 vs +0.0053), so the train labels are usable for embedding-interaction models.
* The pooled-history query (last1/last5/last20/all/recency pools) adds +0.0045 on held-out train over the bilinear model but only +0.0015 on val.
  It is weaker than target attention (below).

### 3.2 Tiny interpretable history attention (`06c_history_attention.py`, about 330 parameters, 1 epoch on train subsample)

score_i = w_pool * c_i . pool + w_attn * sum_j softmax_j(beta * c_i.h_j + bias_j) * c_i.h_j + w_user * c_i.u, with
bias_j = a[event_type_j] + b[event_source_j] + r[hpos_j].

| variant (train-fit) | val NDCG | vs user-only |
|---|---|---|
| user only | 0.3432 | 0 |
| pool: uniform mean | 0.3294 | -0.0138 |
| pool: + learned recency | 0.3496 | +0.0064 |
| pool: + recency + event_type | 0.3509 | +0.0077 |
| pool: + recency + type + source | 0.3513 | +0.0081 |
| **target attention** + recency + type + source | 0.3573 | +0.0141 |
| pool + attention | 0.3575 | +0.0143 |
| **full: pool + attention + user** | **0.3584** | **+0.0152 (0.0014)** |
| full, val 5-fold CV (4 epochs) | 0.3513 | |

Learned parameters (full model, train-fit):
* Attention temperature beta = 10.4. The three heads are weighted about equally (pool 1.67, attention 2.16, user 2.0).
* **Recency bias r[hpos] relative to hpos 0:** 1: -0.71, 2: -1.29, 4: -2.09, 8: -2.85, 16: -3.75, 32: -3.87, 64: -4.22, 128: -4.29.
  The weight decays by about e^-2 within 4 events and flattens after about 30 events.
* **event_type bias** (relative to type 1): 5: +0.34, 6: +0.04, 1: 0, 8: -0.75, 10: -0.89, 4: -1.02, 2: -1.24, 3: -1.73, 9: -2.35, 7: -2.52.
  Types 5, 1 and 6 are the most "target-predictive" history events; types 7 and 9 are nearly ignored.
* **event_source bias** (relative to mean; frequent sources 0/1/2 = 1.14M/0.84M/0.34M val events): 1: +0.79, 8: +0.74, 11: +0.62, 4: +0.48, 2: +0.21,
  0: -0.22, 3: -1.22, 7: -1.30. Source 1 gets about 2.7x the weight of source 0.

**History truncation** (`06e_attn_truncation.py`; same full model, only events with hpos < K kept): K = 256: 0.3584, 128: 0.3579, 64: 0.3575,
**32: 0.3584 (+0.0000 +/- 0.0006)**, 16: 0.3575 (-0.0008), 8: 0.3530 (-0.0054 +/- 0.0010). For embedding similarity, everything beyond the last
~32 events is redundant.

**Takeaway:** target attention (each candidate attends to the history) beats any fixed pooling by about 0.006-0.008. Event type and source add about 0.0015.
The cheapest strong neural input is (candidate emb, history emb, hpos, event_type, event_source, user emb).

---------------------------------------------------------------------------------------------------------------------

## 4. Structure of the item space (`07_structure.py`, `07b_knn_f.py`)

300K unique val candidate items (category is constant per item in 99.999% of cases), 52 categories, majority class 12.9%.

| probe | value |
|---|---|
| linear probe (logreg) category accuracy | **0.985** |
| kNN-1 category accuracy / kNN-10 same-category rate | 0.955 / 0.905 (chance 0.055) |
| mean top-1 neighbour cos | 0.921 |
| between-category share of variance (R2 of category one-hot) | 0.235 |
| eta^2(category) of PC1/PC2/PC3/PC4..10 | 0.77 / 0.80 / 0.62 / 0.15-0.43 |
| spherical KMeans k = 53: NMI / purity | 0.570 / 0.619 |
| spherical KMeans k = 256: NMI / purity | 0.541 / 0.709 |

The top 3 PCs (12% of the variance) are essentially category axes. Category is linearly separable, but most of the variance (77%) is within-category,
which is useful for fine-grained similarity.

**f1..f5 vs item identity and embeddings** (train subsample, 9.7M rows of repeated items; val items for the regressions):

| | share of variance explained by item_id | within item-day residual share | ridge R2 from embedding | kNN-20 R2 from embedding | category-mean R2 |
|---|---|---|---|---|---|
| f1 | 0.53 | 0.38 | 0.240 | 0.310 | 0.210 |
| f2 | 0.64 | 0.31 | 0.298 | 0.298 | 0.060 |
| f3 | 0.57 | 0.36 | 0.262 | 0.353 | 0.252 |
| f4 | **0.98** | 0.001 | 0.077 | 0.098 | 0.030 |
| f5 | 0.93 | 0.03 | 0.134 | 0.157 | 0.073 |
| item target rate | | | | -0.046 (corr 0.02) | 0.000 |

* **f4 and f5 are static item attributes** (98% / 93% item-level, constant within an item-day): price-like or age-like. The embedding barely encodes them
  (kNN R2 0.10 / 0.16), so they complement it.
* **f1, f2, f3 are about half item-level and half user/context-dependent.** The embedding explains roughly a third of their item-level part, f2 especially beyond
  category (0.30 vs 0.06).
* The embedding **does not predict item-level attractiveness** (item mean target R2 < 0). All embedding signal is relative to the user, the history or the feed.

---------------------------------------------------------------------------------------------------------------------

## 5. Lift inside a ranker (`05_lgb_lift.py`, `05c_xgb_valcv.py` + `05d_merge_xgbcv.py`, `08_train_vs_val_signals.py`)

### 5.1 LightGBM lambdarank trained on a 200K-feed train subsample, evaluated on val (500 rounds; 200/1000 rounds within +/- 0.002)

| features | val NDCG | vs base |
|---|---|---|
| base = f1..f5 + category_id | 0.3303 | 0 |
| base + user_dot | 0.3467 | +0.0164 (0.0014) |
| base + rec_hl5 + max_sim | 0.3587 | +0.0284 (0.0016) |
| base + compact8 (user_dot, rec_hl5, last5_cos, max_sim, smax_t0.1, centrality_loo, hmean_cos, type1_max) | **0.3617** | **+0.0314 (0.0016)** |
| **base + all 39 embedding features** | **0.3651** | **+0.0348 (0.0016)** |
| embedding features only (39; no f1..f5, no category) | 0.3588 | +0.0285 (0.0018); +0.022 over f1 alone |

Top gain shares, compact8 model: category 0.157, last5_cos 0.083, f4 0.081, f5 0.080, f3 0.076, max_sim 0.071, user_dot 0.067, type1_max 0.065,
centrality_loo 0.062. All-emb model: category 0.137, f4 0.046, f5 0.044, user_dot 0.035, cand_cat_centrality 0.033, last5_cos 0.032.

**Caveat:** the train-fit base model (0.3303) is *worse than f1 alone* (0.3366). Two effects are at work. (a) Some f-features behave differently on
production-ranked train than on random-order val: f4 is below random on train (0.3177 vs 0.3232) but above random on val, and f5's lift is +0.018 on train vs +0.008 on val.
(b) f2..f5 add little beyond f1 even in-distribution: the val-CV GBDT on base also stays below f1 (0.3332 vs 0.3362, Section 5.2).
The embedding features transfer well, as their single-feature train vs val numbers show:

| signal | train-subsample NDCG (random 0.3232) | val NDCG (random 0.2902) |
|---|---|---|
| f1 | 0.3625 (+0.039) | 0.3362 (+0.046) |
| f5 | 0.3412 (+0.018) | 0.2979 (+0.008) |
| f4 | 0.3177 (-0.006) | 0.2976 (+0.007) |
| user_dot | 0.3583 (+0.035) | 0.3435 (+0.053) |
| rec_hl5 | 0.3705 (+0.047) | 0.3492 (+0.059) |
| max_sim | 0.3622 (+0.039) | 0.3436 (+0.054) |
| centrality_loo | 0.3511 (+0.028) | 0.3170 (+0.027) |
| -pos (production order, train only) | 0.3612 (+0.038) | n/a |

Train positive rate by original position: pos 0-4: 7.5%, 5-9: 4.6%, 10-19: 3.5%, 20-49: 2.3%, 50-99: 1.5%, 100-199: 1.1%, 200+: 1.0%.
The within-feed Spearman of every signal with -pos is small (|rho| <= 0.08), so the production order is mostly driven by something we do not observe.

### 5.2 Val 5-fold CV (unbiased traffic; LightGBM 400 rounds, 31 leaves)

XGBoost `rank:ndcg` on GPU (`05c_xgb_valcv.py`, merged by `05d_merge_xgbcv.py`): eta 0.1, 200 rounds, depth 7. Each training fold has about 11.5K feeds.
Per-feed out-of-fold NDCG is averaged over 2 fold seeds; differences are paired. LightGBM lambdarank with truncation 300 was too slow on the shared CPU,
so this check uses XGBoost. "attn" and "bilinear" are the val predictions of the *train-fitted* models from Section 3 (no val leakage).

| features | OOF NDCG | vs base | vs base+compact8 |
|---|---|---|---|
| f1 alone (no model) | 0.3362 | | |
| base = f1..f5 + category | 0.3332 | 0 | -0.0225 |
| base + user_dot | 0.3464 | +0.0133 (0.0012) | -0.0092 |
| base + compact8 | 0.3557 | +0.0225 (0.0014) | 0 |
| base + compact8 + within-feed rank versions | 0.3547 | +0.0215 | -0.0010 (0.0011) |
| base + all 39 embedding features | 0.3545 | +0.0213 | -0.0012 (0.0011) |
| embedding features only (39, no f's / category) | 0.3506 | +0.0174 | -0.0051 |
| base + attn score | 0.3547 | +0.0215 (0.0013) | -0.0010 |
| base + compact8 + attn | 0.3578 | +0.0246 | **+0.0021 (0.0009)** |
| **base + compact8 + attn + bilinear** | **0.3601** | **+0.0269 (0.0015)** | **+0.0044 (0.0010)** |

Reading:
* The ranking from 5.1 is confirmed on unbiased labels: embedding features add **+0.022 to +0.027** over the f-features.
* With only 11.5K training feeds, 39 features overfit compared with 8. With 200K train feeds (5.1), the full set is better (+0.0034 over compact8).
  Within-feed rank copies did not help the tree model here.
* **The train-fitted learned scores (attention and bilinear) add on top of the handcrafted features (+0.0044 +/- 0.0010).** Fitting the embedding
  interactions on the large train set and feeding the scores to the ranker works.
* Even on val folds, a GBDT on f1..f5 + category does not beat f1 alone (0.3332 vs 0.3362). The extra f-features carry little transferable signal; the gain
  comes from the embeddings.
* For comparison, the training-free z-sum f1 + user + rec_hl5 + 0.5 max_sim reaches 0.3569 (Section 2.4), on par with these small-data GBDTs.

---------------------------------------------------------------------------------------------------------------------

## 6. Covariate drift of embedding features (`10_drift.py`)

| feature | train q10/q50/q90 | val q10/q50/q90 | test q10/q50/q90 |
|---|---|---|---|
| user_dot | 24829 / 34354 / 40249 | 22676 / 32036 / 38787 | 23210 / 32132 / 38786 |
| rec_hl5 | 0.363 / 0.553 / 0.760 | 0.340 / 0.512 / 0.714 | 0.339 / 0.509 / 0.711 |
| max_sim | 0.668 / 0.875 / 0.976 | 0.613 / 0.829 / 0.959 | 0.613 / 0.830 / 0.959 |
| centrality_loo | 0.569 / 0.735 / 0.858 | 0.551 / 0.706 / 0.824 | 0.552 / 0.705 / 0.822 |
| n_sim_gt0.9 (mean) | 3.14 | 1.83 | 1.78 |
| NaN rate (no valid history) | 1.2% | 1.4% | 0.9% |

**Val and test are indistinguishable.** Train candidates are systematically closer to the user's history (median max_sim 0.875 vs 0.829), which fits
production-ranked train traffic. Within-feed normalised versions (rank, z-score, gap to the feed max) remove this level shift. They did not help in the
val-CV GBDT (-0.0010 +/- 0.0011), where there is no shift, but they are worth testing for models trained on train.

---------------------------------------------------------------------------------------------------------------------

## 7. Recommendations

### 7.1 Embedding features for a GBDT ranker (in priority order)
1. **Recency-weighted history similarity** `rec_hl{3,5}` = sum_j exp(-j/hl) cos(c, h_j) / sum_j exp(-j/hl) over valid events (j = index among valid events).
   Also `last5_cos`, `last20_cos`.
2. **max_sim / top5_sim / softmax-pooled sim (tau = 0.1)** over history, plus the near-duplicate count `n_sim_gt0.9`; consider `n_sim_gt0.95`.
3. **user_dot** (= cosine; set NaN for all-zero user vectors). It is the only good embedding signal when H = 0.
4. **Per-event-type max/mean sims** for types 1, 5, 6, 10 (learned weights highest), and per event_source group (0, 1, 2, other).
5. **Within-feed centrality** (LOO cosine to the candidate-set mean) and the same-category centrality.
6. **Within-feed normalisations** of all of the above (rank percentile, z-score, value minus feed max) to absorb the train/val level shift (Section 6).
   They were neutral in val-CV (-0.0010 +/- 0.0011); test them in train-fit models.
7. **A learned history-attention score** (Section 3.2, about 330 params, trains in minutes): 0.3584 as a single feature, ahead of every handcrafted signal.
   It can be trained on train and used as a feature, with no val leakage. On top of compact8 it adds +0.0021 +/- 0.0009, and +0.0044 +/- 0.0010 together with the
   train-fitted bilinear score (val-CV, Section 5.2).
8. Skip: item norm, raw L2 (redundant with dot), uniform full-history mean on its own (dominated), and global item-quality projections b^T c (no signal).

### 7.2 Neural model (candidate-set transformer with history attention)
* **Inputs that matter:** normalised candidate item embedding; user embedding (use its zero-ness as a flag); history item embeddings **with learned
  embeddings for hpos (recency), event_type and event_source**; missing-embedding mask (14% of history). The tabular f1..f5 + category go as candidate tokens.
* **Architecture:** candidate-to-history **target attention** (DIN/transformer cross-attention) is worth about 0.006-0.008 over any fixed pooling. Self-attention
  across candidates captures the centrality signal (+0.028 alone). Recency decays steeply (bias -2 at hpos 4, -3.75 at hpos 16). **Truncating the history to the last 32 events lost nothing** for the
  attention scorer (0.3584 vs 0.3584); 8 events lose 0.005. A 32-64 event window is enough for the embedding pathway, though longer windows may still help count or category features. Keep a user-tower token or a dot term.
* **Losses and scale:** use a listwise softmax (or LambdaLoss) with a **learnable temperature**. The cosine logits must be scaled (about x13-20) or the
  training signal is misaligned (Section 3.1).
* **Initialise near identity.** The pretrained space is already aligned: dot gives 0.343 at step 0. Use W = I or residual adapters with zero init, and freeze or
  slow-learn the item projection, because 30-40% of test candidate items never occur in train.
* **Train on train, validate on val.** Embedding-interaction models transfer from train to val (bilinear gains +0.0064 train-fit vs +0.0053 val-CV). f-feature models
  are more exposed to the train/val shift. Small val-CV models overfit fast: the ~98K-parameter history query model gives -0.0009 in val-CV but +0.0079 when trained on train.

---------------------------------------------------------------------------------------------------------------------

## 8. Pitfalls checklist
* emb_idx is **split-local**. Never use a val/test idx in the train table; join by item_id.
* Pessimistic ties: a constant score in a feed (no history, zero user vector) is the worst case. Always back off to another signal and add a tiny random or
  secondary tie-breaker.
* Mask missing history embeddings (14-15%; 33% for type 9) instead of pooling zeros. Treat all-zero user vectors as missing.
* Dot, cosine and -L2 give the same ranking (all norms are 256). Dot products of these int8 vectors reach about 65,000, which overflows int8/int16 accumulators: convert to float32 (or int32) first.
* User vectors change per request. Do not aggregate them per user across feeds or splits as if they were static (val vs train cos is only 0.75).
* LightGBM trained on train with only f-features underperforms f1 alone on val. Check every train-fit model against f1 and against the val-CV numbers.
* Val has 14.4K feeds. Differences below about 0.003 are noise; use paired per-feed SEs.

---------------------------------------------------------------------------------------------------------------------

## 9. Reproducibility (scripts in `<repo>/eda/embeddings/`)

| script | content |
|---|---|
| `common.py`, `emb_features.py` | data access; numba kernel computing the 43 candidate-level embedding features for any split/feed subset |
| `01_stats.py` | value/norm/PCA stats, missing rates, item_id-to-idx bijection, cross-split identity, user-vector constancy |
| `02_val_signals.py` | single-signal exact NDCG on val (`val_signals.csv`, `val_feats.npy`), Spearman with f1..f5 |
| `03_user_vs_history.py`, `03b_fusion.py` | user vector vs history pooling, half-life sweep, training-free fusions |
| `04_train_feats.py` | the same features for 200K random train feeds (seed 42), saved in `cache/` |
| `05_lgb_lift.py` | LightGBM lambdarank train-fit, evaluated on val (its own val-CV part was stopped: too slow on the shared CPU) |
| `05c_xgb_valcv.py`, `05d_merge_xgbcv.py` | 5-fold val-CV XGBoost (GPU) lift table. `05b_lgb_valcv_learned.py` is the abandoned LightGBM version |
| `06_bilinear.py`, `06b_bilinear_diag.py`, `06d_bilinear_scaled.py` | bilinear/history-query models; fixed vs learnable logit scale |
| `06c_history_attention.py`, `06e_attn_truncation.py` | tiny DIN-style attention scorer, learned type/source/recency biases, history truncation |
| `07_structure.py`, `07b_knn_f.py` | category probes, PCA, KMeans, f1..f5 item-level share, ridge/kNN regressions |
| `08_train_vs_val_signals.py` | train-subsample vs val single-signal NDCG, positive rate by original position |
| `09_bins.py` | positive rate by similarity bins and deciles, lift by H/M, time split |
| `10_drift.py` | train/val/test feature quantiles (also saves `cache/test_feats.npy`) |
