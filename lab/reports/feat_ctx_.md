# Feature family `ctx_`: feed context and competition among candidates

Code: `<repo>/feats_v2/ctx_/` (builders `build_ctx.py`, `build_ctx2.py`, `build_ctx3.py`, `build_ctx4.py`, `build_ix.py`; tools `single_ndcg.py`, `blend_screen.py`, `eval_ens.py`, `compare_oof.py`, `forward.sh`, `importance.py`, `drift.py`, `drift_days.py`).
Features: `<repo>/data/feats/{train,val,test}/ctx_*.npy` (float32, aligned with `cand_*.npy`). **The 24 KEEP features exist for all 3 splits. Every dropped feature file was deleted.**
Harness log: `reports/feature_evals.jsonl` (tags `ctx_*`). Seed-ensemble log: `feats_v2/ctx_/ens_evals.jsonl`. Saved candidate OOFs of KEEP 24 (5 seeds x 2 column orders): `feats_v2/ctx_/oof/{setA_hsx,keep24_alpha}_s*.npy`.

---

## 0. TL;DR

1. **The strongest new signal is not FeatureSalad-style normalization. It is competition among the feed's candidates for each recent history item** (groups `hn` and `hs`, a new idea).
   - For each of the user's last K history items, the candidates of the feed compete: softmax over candidates of cos(c, h_j)/tau, or the candidate's rank among the candidates for that history item.
   - A candidate then collects the recency-weighted "attention mass" or reciprocal rank it wins.
   - `ctx_hs_t10` scores **0.3544** single-feature NDCG and `ctx_hn_soft05` scores 0.3535. That beats `maxsim32` (0.349) and every other non-fused single feature in the project.
   - Being the nearest candidate of at least one recent history item lifts the positive rate from 1.18% to 3.2% (one item), 4.3% (two) and 6.2% (five).
2. **Similarity-graph features (group `nd`) also add value.** They use the candidate embeddings of the same feed:
   - kNN-smoothed cos_u / maxsim32 / f1 over the candidate's neighbours in the feed, and the candidate's deviation from its neighbourhood;
   - user and history direction after removing the feed's centroid;
   - similarity to the feed's top-5 candidates by cos_u and by maxsim32.
3. **Harness deltas (3 seeds, paired; base = 118 frozen features, base NDCG 0.3634):**
   - hn/hs compact 10: **+0.00161 +/- 0.00076**.
   - nd (17 features): +0.00152 +/- 0.00073; nd compact 10: +0.00119 +/- 0.00073.
   - hn10 + nd10: **+0.00251 +/- 0.00078**.
   - **KEEP 24: +0.00321 +/- 0.00080 (tag `ctx_setA_plus_hsx_s3`) and +0.00200 +/- 0.00080 (tag `ctx_KEEP24_final_s3`).** These two runs have the *same* 24 features in a different column order (see item 5).
   - Whole family before pruning (147 features): +0.00340 +/- 0.00085.
4. **Seed-ensemble deltas (5 seeds, mean of within-feed z-scored OOFs; this removes most LightGBM run-to-run noise):**
   - KEEP 24: **+0.00305 +/- 0.00100** (base ensemble 0.36710 -> 0.37015).
   - The same 24 features in alphabetical column order: +0.00194 +/- 0.00099.
   - Both orders pooled (10 models): **+0.00282 +/- 0.00096**.
   - Honest range for KEEP 24: **+0.0015 to +0.0030, about +0.0022**. All 10 seed x order runs are positive.
5. **Most FeatureSalad / BlackPearl ideas gave nothing measurable on top of the base set:**
   - x/max, x/sum, x - median;
   - within-category rank / z / gap;
   - category-level ranks;
   - feed-level mean/std/skew/entropy;
   - z-sum / Borda fusion;
   - near-duplicate counts;
   - item cross-feed feed-size stats.

   Forward selection on top of the KEEP core (5 seeds, paired) gives these increments, all < 2 SE:
   - cr -0.00006, ct -0.00064, fs -0.00011, zs -0.00055;
   - cl+ce +0.00037, sg +0.00070, ix (BlackPearl) +0.00052, all borderline ones together +0.00031.

   The base already has within-feed rank/z/gap of 14 columns, which covers most of that space.
6. **Harness noise is of the same size as the effects. Read single 3-seed deltas with care.**
   - Reordering the same 24 columns moves the 3-seed delta between +0.0020 and +0.0032. LightGBM's `feature_fraction` samples by column index.
   - Per-seed deltas of one configuration range from 0.0004 to 0.0051.
   - Seed-averaging the base model alone gains +0.0027 (single-seed 0.3634 -> 5-seed ensemble 0.3671). The final pipeline should average seeds and column permutations.
7. **The features are robust to the val -> test shift:**
   - They use only the feed's own candidates, its history and the item embeddings: no train labels, no time windows, no train item coverage.
   - PSI val vs test-public / test-private <= 0.007 for 23 of 24 KEEP features. `ctx_nd_ucent` has 0.028 / 0.017, caused only by the share of zero user vectors, which turns into NaN: 1.4% in val vs 0.1-0.3% in test.
   - Per-day medians are flat from Tue Sep 22 to Tue Sep 29, weekend days included.
   - They cover cold items: 98.6% non-NaN, and NaN only when the feed has no history or the item has no embedding.

## 1. What was built (all groups; KEEP marked)

All features are computed per feed from that feed's rows only, with exactly the same code for train, val and test. The only exception is `ix`, which uses train candidate appearances: an expanding window for train rows, all of train for val/test rows. No feature uses labels: neither val labels nor train targets.

| group | prefix | #feat | definition | status |
|---|---|---|---|---|
| fs | `ctx_fs_` | 32 | feed-level (constant within feed):<br>- mean/std/skew of f1..f5, maxsim32, cos_u, cos_h10<br>- category entropy, #distinct categories, max category share<br>- share of candidates in history categories<br>- mean pairwise candidate cos, share of near-duplicate pairs<br>- cos(user, feed centroid), cos(last-10 history, feed centroid) | dropped |
| ct | `ctx_ct_` | 21 | x - median (8 columns); x/max and x/sum for f1..f5; min-max for the similarity columns | dropped |
| cr | `ctx_cr_` | 22 | within (feed, category): pct rank, z, gap to category max of f1, f2, f4, f5, maxsim32, cos_u, cos_h10; category size.<br>**f3 is constant within (feed, category)**, so f3 is a user x category score and its within-category versions are useless. | dropped |
| cl | `ctx_cl_` | 6 | rank of the candidate's category among the feed's categories by its best maxsim32 / cos_u / f1 and by count; history share / feed share; category mean maxsim32 minus feed mean | dropped |
| ce | `ctx_ce_` | 3 | category mean f1 / cos_u minus feed mean; mean cos to the top-5 candidates by f1 | dropped |
| zs | `ctx_zs_` | 9 | unsupervised fusion: z(f1)+z(maxsim32)+z(cos_u) (+cos_h10, attn32), Borda mean rank; within-category rank and gap of these | dropped |
| sg | `ctx_sg_` | 2 | "source signature" (which of f1..f5 sit on their minimum or exact-zero value) and its share in the feed; feed share of f1-at-minimum | dropped |
| ix | `ctx_ix_` | 10 | BlackPearl item cross-feed context over the item's train appearances:<br>- mean/std log M; this feed's log M minus the mean<br>- mean within-feed pct rank of the production position, mean log1p(pos)<br>- mean f1, f1 minus mean f1<br>- within-feed ranks of these | dropped |
| nd | `ctx_nd_` | 17 | similarity graph among the feed's candidates (see §1.2) | **10 KEEP** |
| hn | `ctx_hn_` | 22 | candidates competing for history items: rank-based plus softmax (see §1.1) | **6 KEEP** |
| hs | `ctx_hs_` | 11 | softmax-competition variants (K, decay, temperature, event type) | **8 KEEP** |

Two side checks:
- The in-feed array order carries no signal: index NDCG is 0.2880 / 0.2886, vs random 0.289.
- Neither does item_id order (0.2890 / 0.2885).

### 1.1 History competition (`hn`, `hs`)

Notation:
- c_i are the feed's candidates and h_j (j = 0 newest) are the last K non-missing history items. Both are L2-normalized int8 item embeddings.
- S_ij = cos(c_i, h_j) and w_j = decay^j.
- "mass" features are multiplied by M, so 1.0 means a uniform share of attention.

| feature | K | decay | definition |
|---|---|---|---|
| `ctx_hn_soft05` | 32 | 0.85 | M * sum_j w_j softmax_i(S_ij / 0.05) / sum_j w_j |
| `ctx_hs_t10` | 32 | 0.85 | same, tau = 0.10 |
| `ctx_hs_d7` | 32 | 0.70 | same, tau = 0.05 |
| `ctx_hs_k128` | 128 | 0.95 | same, tau = 0.05 (top LightGBM gain among all features except category_id) |
| `ctx_hs_k128t10` | 128 | 0.95 | tau = 0.10 |
| `ctx_hs_k256` | 256 | 0.98 | tau = 0.05 |
| `ctx_hs_e1` | 32 | 0.85 | history restricted to event_type 1 (the target-like event), tau = 0.05 |
| `ctx_hs_e1k64` | 64 | 0.95 | event_type 1 only |
| `ctx_hn_rrw` | 32 | 0.85 | sum_j w_j / (1 + rank of c_i among candidates for h_j), divided by sum_j w_j |
| `ctx_hn_e1_rrw` | 32 | 0.85 | same over event_type-1 history |
| `ctx_hn_top1w` | 32 | 0.85 | recency-weighted count of history items whose nearest candidate is c_i |
| `ctx_hs_mutual` | 32 | 0.85 | same, but only when c_i and h_j are mutual nearest neighbours |
| `ctx_hn_margin` | 32 | | S(c_i, h*) - max_{k != i} S(c_k, h*), where h* is c_i's best-matching history item |
| `ctx_hn_bestrk` | 32 | | rank of c_i among candidates for h* (0 = c_i is h*'s nearest candidate), / (M - 1) |

Dropped variants (no increment on top of the core: forward selection -0.00042): `top1`, `top3w`, decay 0.7 and 0.95 rank versions, event-type-1 top1w and margin, `soft02`, `hs_k64`, `hs_e2`.

Why it works: `maxsim32` says "this candidate is close to some recent history item". The competition features add "and no other candidate in this feed is closer to that item". The retrieval stage often puts several look-alikes of a viewed item into the feed. The one closest to the item the user actually engaged with wins. Softmax normalization over the feed also makes the value comparable across feeds of different density.

### 1.2 Similarity graph (`nd`)

These use the normalized candidate embeddings of the feed (S = V V^T, self excluded) and a kernel weight exp(20 (s - 1)).

| feature | definition |
|---|---|
| `ctx_nd_knn_u` / `_h` / `_f1` | kernel-weighted mean of the neighbours' cos_u / maxsim32 / f1 |
| `ctx_nd_knn_u_dev` / `_h_dev` | own value minus the neighbourhood mean ("best of its cluster") |
| `ctx_nd_ucent` | cos(c, u) - cos(c, feed centroid) |
| `ctx_nd_hcent` | cos(c, last-10-history mean) - cos(c, feed centroid) |
| `ctx_nd_top5u` / `_top5h` | mean cos to the 5 best other candidates by cos_u / maxsim32 |
| `ctx_nd_max` | max cos to any other candidate (near-duplicate) |

Dropped: `mean` (about the existing `centrality`), counts with cos > 0.8 / 0.9 / 0.95, and "#near-duplicates with better maxsim/f1/cos_u" (forward selection -0.00024).

Near-duplicates do not change the positive rate:
- 22% of candidates have another candidate with cos > 0.95 in the feed;
- their positive rate is 1.51%, vs 1.58% overall.

## 2. Single-feature NDCG (val, random tie-break, NaN at the bottom; random = 0.2902 with this noise)

### 2.1 KEEP features, with drift

PSI uses 10 val-quantile bins plus a NaN bin. Public = test before 2026-09-27 11:32 MSK.

| feature | NDCG up | NDCG down | coverage | PSI val-pub | PSI val-priv | PSI pub-priv | median val / pub / priv |
|---|---|---|---|---|---|---|---|
| `ctx_hn_soft05` | 0.3535 | 0.2490 | 0.986 | 0.0037 | 0.0024 | 0.0002 | 0.17 / 0.166 / 0.165 |
| `ctx_hn_rrw` | 0.3514 | 0.2501 | 0.986 | 0.0057 | 0.0029 | 0.0007 | 0.0223 / 0.0211 / 0.0216 |
| `ctx_hn_e1_rrw` | 0.3493 | 0.2503 | 0.969 | 0.0073 | 0.0029 | 0.0011 | 0.0235 / 0.0222 / 0.0228 |
| `ctx_hn_top1w` | 0.3415 | 0.2702 | 0.986 | 0.0039 | 0.0023 | 0.0002 | 0 / 0 / 0 |
| `ctx_hn_margin` | 0.3382 | 0.2501 | 0.986 | 0.0038 | 0.0023 | 0.0002 | -0.153 / -0.155 / -0.154 |
| `ctx_hn_bestrk` | 0.2509 | **0.3346** | 0.986 | 0.0037 | 0.0024 | 0.0002 | 0.089 / 0.0882 / 0.0879 |
| `ctx_hs_t10` | **0.3544** | 0.2492 | 0.986 | 0.0037 | 0.0024 | 0.0002 | 0.553 / 0.55 / 0.547 |
| `ctx_hs_d7` | 0.3531 | 0.2504 | 0.986 | 0.0037 | 0.0024 | 0.0002 | 0.0995 / 0.0966 / 0.096 |
| `ctx_hs_k128` | 0.3503 | 0.2457 | 0.986 | 0.0037 | 0.0023 | 0.0002 | 0.326 / 0.319 / 0.321 |
| `ctx_hs_e1` | 0.3523 | 0.2492 | 0.969 | 0.0051 | 0.0024 | 0.0006 | 0.214 / 0.207 / 0.206 |
| `ctx_hs_k256` | 0.3447 | 0.2451 | 0.986 | 0.0037 | 0.0023 | 0.0002 | 0.44 / 0.435 / 0.435 |
| `ctx_hs_e1k64` | 0.3490 | 0.2470 | 0.969 | 0.0051 | 0.0023 | 0.0006 | 0.351 / 0.343 / 0.342 |
| `ctx_hs_k128t10` | 0.3510 | 0.2464 | 0.986 | 0.0037 | 0.0024 | 0.0002 | 0.677 / 0.672 / 0.672 |
| `ctx_hs_mutual` | 0.3335 | 0.2743 | 0.986 | 0.0038 | 0.0023 | 0.0002 | 0 / 0 / 0 |
| `ctx_nd_knn_u_dev` | 0.3240 | 0.2545 | 1.000 | 0.0014 | 0.0011 | 0.0000 | -0.00993 / -0.0104 / -0.0105 |
| `ctx_nd_ucent` | 0.3106 | 0.2658 | 0.986 | 0.0281 | 0.0170 | 0.0012 | -0.229 / -0.228 / -0.228 |
| `ctx_nd_knn_h_dev` | 0.3253 | 0.2547 | 0.986 | 0.0037 | 0.0023 | 0.0002 | -0.0185 / -0.0184 / -0.0185 |
| `ctx_nd_knn_h` | 0.3235 | 0.2607 | 0.986 | 0.0038 | 0.0024 | 0.0005 | 0.799 / 0.797 / 0.798 |
| `ctx_nd_max` | 0.3070 | 0.2599 | 1.000 | 0.0004 | 0.0002 | 0.0001 | 0.885 / 0.887 / 0.886 |
| `ctx_nd_hcent` | 0.3407 | 0.2637 | 0.986 | 0.0037 | 0.0025 | 0.0002 | -0.0773 / -0.0773 / -0.0775 |
| `ctx_nd_knn_f1` | 0.3241 | 0.2605 | 1.000 | 0.0002 | 0.0007 | 0.0004 | 0.375 / 0.375 / 0.369 |
| `ctx_nd_knn_u` | 0.3173 | 0.2621 | 1.000 | 0.0018 | 0.0018 | 0.0003 | 0.511 / 0.511 / 0.512 |
| `ctx_nd_top5u` | 0.3244 | 0.2616 | 1.000 | 0.0004 | 0.0001 | 0.0003 | 0.536 / 0.536 / 0.536 |
| `ctx_nd_top5h` | 0.3283 | 0.2620 | 1.000 | 0.0004 | 0.0002 | 0.0001 | 0.514 / 0.513 / 0.512 |

For reference: maxsim32 0.349, dot_u 0.3435, f1 0.336.

### 2.2 Dropped groups

Best single feature per group:
- `ct`: monotone within feed, so the same as the raw column (maxsim32_med 0.3492, cos_h10_med 0.3461, f1_med 0.3363).
- `cr`: maxsim32_z 0.3220, cos_u_z 0.3201, cos_h10_z 0.3184, f1_z 0.3139.
- `cl`: best_f1_rk 0.3188 (down), mean_h_dev 0.3177.
- `ce`: top5f1 0.3320.
- `zs`: z3 / z5 0.3553 (a fusion of base columns).
- `sg`: share 0.2970.
- `ix`: df1 0.3153; the others 0.293-0.299.
- `fs`: constant within the feed, so random (0.2902) by construction.

The full table for all 147 built features is in `feats_v2/ctx_/single_val.json`; drift for all of them is in `drift_all.json`.

## 3. Drift by day (median [q10, q90]; val = Tue/Thu, test = Sat..Tue)

| feature | Tue22 | Thu24 | Sat26 | Sun27 | Mon28 | Tue29 |
|---|---|---|---|---|---|---|
| hn_soft05 | 0.169 [0.0027,2.8] | 0.17 [0.0027,2.8] | 0.165 [0.0027,2.8] | 0.165 [0.0026,2.8] | 0.164 [0.0028,2.8] | 0.168 [0.0027,2.8] |
| hs_k128 | 0.326 [0.0093,2.7] | 0.326 [0.0091,2.7] | 0.317 [0.0089,2.8] | 0.319 [0.009,2.7] | 0.32 [0.0095,2.8] | 0.324 [0.0092,2.7] |
| hs_e1 | 0.213 [0.004,2.8] | 0.214 [0.0039,2.8] | 0.207 [0.0036,2.8] | 0.206 [0.0038,2.8] | 0.206 [0.0039,2.8] | 0.207 [0.0036,2.9] |
| hn_rrw | 0.0222 [0.0068,0.11] | 0.0223 [0.0067,0.12] | 0.0211 [0.0066,0.11] | 0.0211 [0.0066,0.11] | 0.0217 [0.0068,0.11] | 0.0219 [0.0067,0.11] |
| nd_knn_u_dev | -0.0099 [-0.076,0.027] | -0.0100 [-0.076,0.028] | -0.0104 [-0.076,0.028] | -0.0103 [-0.076,0.027] | -0.0103 [-0.076,0.028] | -0.0107 [-0.077,0.027] |
| nd_ucent | -0.228 [-0.31,-0.12] | -0.229 [-0.32,-0.12] | -0.228 [-0.31,-0.12] | -0.228 [-0.31,-0.12] | -0.228 [-0.32,-0.12] | -0.227 [-0.31,-0.12] |
| nd_knn_h | 0.799 [0.56,0.92] | 0.799 [0.55,0.92] | 0.797 [0.55,0.92] | 0.797 [0.55,0.92] | 0.798 [0.56,0.92] | 0.8 [0.56,0.92] |
| nd_top5h | 0.514 [0.33,0.74] | 0.515 [0.33,0.74] | 0.514 [0.33,0.73] | 0.511 [0.33,0.73] | 0.512 [0.33,0.73] | 0.513 [0.32,0.74] |

All 24 KEEP features behave like this; the full output comes from `drift_days.py`. There is no weekend effect and no trend with distance from train. That is expected: the features use no train statistics.

The dropped `ix` group is the only one with real drift: PSI val-priv about 0.07, because train coverage falls from 65% of val rows to 53% of private rows.

Train (production traffic) is shifted the usual way: train candidates sit closer to history, e.g. median `hs_k128` is 0.54 on train vs 0.33 on val. The features carry strong signal on train labels too (single NDCG on the first 200K train feeds: `hs_d7` 0.370, `hs_t10` 0.369, `hn_rrw` 0.368).

## 4. Harness results

### 4.1 Official harness (`src/eval_family.py`, paired, fixed 350 rounds)

| tag | #feat | seeds | delta | SE | per-seed |
|---|---|---|---|---|---|
| ctx_hn_s1 (hn, 13 rank/top1 variants) | 13 | 1 | +0.00190 | 0.00123 | |
| ctx_nd_s1 | 17 | 1 | +0.00252 | 0.00121 | |
| ctx_cr_s1 | 22 | 1 | -0.00071 | 0.00119 | |
| ctx_ct_s1 | 21 | 1 | -0.00058 | 0.00117 | |
| ctx_fs_s1 | 32 | 1 | -0.00075 | 0.00119 | |
| ctx_cl_+ctx_ce_s1 | 9 | 1 | -0.00041 | 0.00117 | |
| ctx_zs_s1 | 9 | 1 | -0.00006 | 0.00115 | |
| ctx_sg_s1 | 2 | 1 | -0.00147 | 0.00116 | |
| ctx_ix_s1 | 10 | 1 | +0.00035 | 0.00118 | |
| ctx_ix_s3 | 10 | 3 | +0.00074 | 0.00070 | 0.00035 / -0.00029 / 0.00214 |
| ctx_hnhs_s3 (all 22 hn+hs available then) | 22 | 3 | +0.00108 | 0.00079 | 0.00003 / -0.00013 / 0.00333 |
| ctx_hnhs_compact10_s3 | 10 | 3 | **+0.00161** | 0.00076 | 0.00048 / 0.00117 / 0.00318 |
| ctx_nd_s3 | 17 | 3 | +0.00152 | 0.00073 | 0.00252 / -0.00019 / 0.00224 |
| ctx_nd_compact10_s3 | 10 | 3 | +0.00119 | 0.00073 | 0.00064 / 0.00072 / 0.00221 |
| ctx_setA_hn10_nd10_s3 | 20 | 3 | **+0.00251** | 0.00078 | 0.00215 / 0.00119 / 0.00418 |
| ctx_setA_plus_ix_s3 | 30 | 3 | +0.00260 | 0.00081 | 0.00236 / 0.00229 / 0.00314 |
| **ctx_setA_plus_hsx_s3 (= KEEP 24)** | 24 | 3 | **+0.00321** | 0.00080 | 0.00134 / 0.00317 / 0.00514 |
| **ctx_KEEP24_final_s3 (= KEEP 24, alphabetical column order)** | 24 | 3 | **+0.00200** | 0.00080 | 0.00135 / 0.00036 / 0.00430 |
| ctx_ALL_s3 (whole family, 147) | 147 | 3 | +0.00340 | 0.00085 | 0.00284 / 0.00203 / 0.00534 |

The last two KEEP rows contain identical features. The gap between them (0.0012) is pure LightGBM column-sampling noise.

### 4.2 Seed-ensemble evaluation (`eval_ens.py`, 5 seeds, same folds/params; base ensemble 0.36710)

| set | #feat | ensemble delta | SE | mean single-seed delta |
|---|---|---|---|---|
| hn10 | 10 | +0.00171 | 0.00092 | +0.00131 |
| setA (hn10 + nd10) | 20 | +0.00244 | 0.00096 | +0.00205 |
| **KEEP 24 (setA + 4 hs)** | 24 | **+0.00305** | 0.00100 | **+0.00277** |

### 4.3 Forward selection on top of KEEP 24

This uses 5 seeds and is paired against the saved KEEP-24 OOFs, so the SE of a difference is about 0.00055 (single) and 0.0008 (ensemble).

| added group | #added | mean single delta | SE | ensemble delta | SE | decision |
|---|---|---|---|---|---|---|
| hn/hs rest | 11 | -0.00042 | 0.00054 | -0.00005 | 0.00082 | drop |
| nd rest | 7 | -0.00024 | 0.00052 | +0.00027 | 0.00079 | drop |
| ix (BlackPearl) | 10 | +0.00052 | 0.00056 | +0.00101 | 0.00084 | drop (<2 SE) |
| cl + ce | 9 | +0.00037 | 0.00053 | +0.00150 | 0.00080 | drop (<2 SE) |
| sg | 2 | +0.00070 | 0.00053 | +0.00131 | 0.00081 | drop (<2 SE) |
| ix + cl + ce + sg | 21 | +0.00031 | 0.00057 | +0.00134 | 0.00085 | drop |
| cr | 22 | -0.00006 | 0.00056 | +0.00019 | 0.00084 | drop |
| ct | 21 | -0.00064 | 0.00053 | -0.00007 | 0.00083 | drop |
| fs | 32 | -0.00011 | 0.00054 | +0.00112 | 0.00082 | drop |
| zs | 9 | -0.00055 | 0.00054 | -0.00004 | 0.00081 | drop |

Adding the 4 extra hs variants (k256, e1k64, k128t10, mutual) to setA gave +0.00072 +/- 0.00054 (single) and +0.00061 +/- 0.0008 (ensemble). That is weak, but they are kept: they share the KEEP mechanism and `hs_k128`-like long-history versions carry the highest gain.

### 4.4 KEEP 24 in alphabetical column order (`keep24_alpha`, 5 seeds)

The same 24 features in alphabetical order (as `--family ctx_` loads them):

| run | ensemble delta | SE | mean single-seed delta | per-seed |
|---|---|---|---|---|
| KEEP 24, alphabetical order | +0.00194 | 0.00099 | +0.00155 | 0.00139 / 0.00036 / 0.00430 / 0.00049 / 0.00119 |
| KEEP 24, order of §4.2 | +0.00305 | 0.00100 | +0.00277 | 0.00134 / 0.00317 / 0.00514 / 0.00137 / 0.00286 |
| difference (alphabetical - §4.2 order), paired | -0.00110 | 0.00077 | -0.00123 | SE 0.00052 |
| **both orders pooled (10 models) vs base (5 models)** | **+0.00282** | 0.00096 | | |

Even after averaging 5 seeds, a pure column permutation shifts the result by about 0.0012. The per-feed paired SE does not capture this model-level variance, because one model's errors are correlated across feeds. So the honest estimate for KEEP 24 is **+0.0015 to +0.0030 NDCG (about +0.0022 on average)**. It is positive in every run made: 2 orders x 5 seeds.

The pooled number is slightly flattering, because the candidate side averages 10 models and the base averages 5.

### 4.5 Where the gain comes from

Per-feed ensemble delta of KEEP 24 vs base, 5 seeds:

| subgroup | feeds | delta | SE |
|---|---|---|---|
| M [0,30) | 3,883 | +0.0024 | 0.0023 |
| M [30,70) | 3,493 | +0.0049 | 0.0022 |
| M [70,150) | 3,558 | +0.0018 | 0.0019 |
| M [150,301) | 3,489 | +0.0031 | 0.0015 |
| H [50,200) | 6,092 | **+0.0058** | 0.0015 |
| H [200,257) | 6,801 | +0.0007 | 0.0015 |
| unseen-item share < 0.3 | 9,428 | +0.0041 | about 0.0012 |
| unseen-item share 0.3-0.45 | 4,209 | +0.0019 | 0.0018 |
| unseen-item share >= 0.45 | 786 | -0.0031 | 0.0050 (noise) |

The gain is largest for mid-length histories.

LightGBM gain ranks in a full-val fit of base + hn10 + nd17:
- `ctx_hs_k128` is #2 of 155 features (4.2% of gain; only category_id is higher);
- `ctx_hs_e1` is #3;
- `ctx_nd_knn_u_dev` is #5;
- `ctx_nd_ucent` is #11;
- `ctx_hn_rrw` is #14 and `ctx_hs_t10` is #15.

## 5. KEEP list (24; files present for train/val/test)

```
ctx_hn_soft05 ctx_hn_rrw ctx_hn_e1_rrw ctx_hn_top1w ctx_hn_margin ctx_hn_bestrk
ctx_hs_t10 ctx_hs_d7 ctx_hs_k128 ctx_hs_e1 ctx_hs_k256 ctx_hs_e1k64 ctx_hs_k128t10 ctx_hs_mutual
ctx_nd_knn_u_dev ctx_nd_ucent ctx_nd_knn_h_dev ctx_nd_knn_h ctx_nd_max ctx_nd_hcent ctx_nd_knn_f1 ctx_nd_knn_u ctx_nd_top5u ctx_nd_top5h
```

To rebuild (about 7 min for train at 24 threads; seconds for val/test; peak RSS 39 GB):

```
cd <repo>/feats_v2/ctx_ && export OPENBLAS_NUM_THREADS=1
CTX_KEEP=ctx_nd_knn_u_dev,ctx_nd_ucent,ctx_nd_knn_h_dev,ctx_nd_knn_h,ctx_nd_max,ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_u,ctx_nd_top5u,ctx_nd_top5h \
  python build_ctx.py train,val,test nd 24
CTX_KEEP=ctx_hn_rrw,ctx_hn_top1w,ctx_hn_margin,ctx_hn_bestrk,ctx_hn_e1_rrw,ctx_hn_soft05 python build_ctx3.py train,val,test base,_e1,soft 24
python build_ctx4.py train,val,test ctx_hs_t10,ctx_hs_d7,ctx_hs_k128,ctx_hs_e1,ctx_hs_k256,ctx_hs_e1k64,ctx_hs_k128t10 24   # also writes ctx_hs_mutual
```

Dropped groups can be regenerated with `build_ctx.py <splits> fs,ct,cr,cl`, `build_ctx2.py <splits> zs,sg,ce` and `build_ix.py <splits>`.

## 6. Caveats and next steps

- **Noise.** Single 3-seed harness deltas of about 0.001-0.002 are not reliable. The same feature set gave +0.0020 and +0.0032 in two column orders.
  - The per-feed paired SE (about 0.0008) understates the true uncertainty, because model-level variance from seed and column order is about 0.001 even with 5 seeds.
  - The forward-selection "drop" decisions for the borderline groups (+0.0003..0.0007 single) are therefore within noise.
  - Use paired, saved-OOF, multi-seed comparisons (`eval_ens.py` + `compare_oof.py`), ideally over several column orders, for decisions.
  - Average seeds and column orders in the final model: seed averaging alone adds +0.0027 on the base.
- **Linear blends of hn/hs onto the base OOF give a similar size of gain**: about +0.003 for z(base) + 0.1 * z(`ctx_hs_d7`) on the 3-seed base ensemble (optimistic, w picked on val). The trees with 14K training feeds may not fully use these smooth signals. A stacker or NN might get more out of them.
- **The borderline groups may be worth re-testing with a stronger final model:** `cl+ce`, `sg` and `ix` gave ensemble +0.0010..0.0015, < 2 SE. `ix` is the only group that depends on train coverage; it drifts (PSI about 0.07 val -> private) and would weaken on private.
- **The history-competition idea can be extended:**
  - learned per-event-type weights instead of the event_type-1 filter;
  - competition restricted to same-category candidates;
  - the same normalization inside the NN ranker: a softmax over candidates per history token, i.e. a "reverse attention" block.
- Calendar / time-of-day ideas belong to the `cal_` family. The one time-related check here (per-day drift, weekend vs weekday) shows no time dependence of the ctx features.

---

## Verification (adversarial verifier, 2026-11-11)

Scripts: `feats_v2/ctx_/verify/` (`vrun.py`, `vsum.py`, `timesplit.py`, `drift.py`, `trainchk.py`, `repro.py`, `ties.py`, `leak.py`, `reweight.py`, `unseen_single.py`). OOFs were written to the session scratchpad and are not kept.

### V.0 Verdict

**KEEP all 24. The claims hold.**

- No leakage was found, and the features are computed identically for val and test.
- Drift is negligible.
- The family gain is real and survives a placebo control and a time split.

One minor bug was fixed in `build_ctx.py` (zero user vectors, see V.2). After the fix, the official `--family ctx_ --seeds 5` run gives **+0.00201 +/- 0.00067 (3.0 SE)**. The mean over 3 column orders x 5 seeds is **+0.00237**. The expected gain on Private, after reweighting val to the private mix (V.6), is about **+0.0017 +/- 0.0009**.

### V.1 Code audit (leakage and val/test symmetry)

- **Inputs.** `build_ctx.py` (nd), `build_ctx3.py` (hn) and `build_ctx4.py` (hs) read only these inputs:
  - the feed's own candidate `emb_idx`;
  - its history `emb_idx` and `event_type`;
  - the split's own item-embedding memmap;
  - `user_emb`;
  - the base columns `cos_u`, `maxsim32` and `f1`, which are embedding or raw features with no labels (`src/features.py` EMB_NAMES).
- **What they do not read.** No `cand_target`, no `pos`, no timestamps, no other feeds, and no train statistics. Every value is a pure function of one feed, so there is no time window, no split-edge effect, and no val/test asymmetry.
- **History order.** The code takes the first K non-missing history rows as "most recent". I checked that `hpos` is 0, 1, ... from the start of each feed's history block, on 20K feeds of each split (100%). So "last K" really means the newest K on train, val and test.
- **Leak audit.** I used the lead's method: single-feature NDCG on the 117 val feeds whose user has a *later* feed, compared with all other feeds.
  - Every KEEP feature scores the same or lower on the later-user feeds: excess between -0.031 and +0.009.
  - Base `maxsim32` (-0.015) and `cos_u` (-0.024) show the same pattern.
  - There is no sign of same-user later-snapshot information.
- **Files match the code.** I re-ran the kernels on all of val and test and on 3 train slices (20K feeds each: start, middle, end). The output is **bit-identical** to the stored files for all 24 features: max |diff| 0.0 and 0 NaN mismatches.
- **Definitions match the report.** An independent plain-numpy reimplementation on 300 random val feeds and 300 random test feeds matches to <= 5e-5 for 15 of the 17 features checked. The exceptions:
  - `hn_rrw` differs by up to 3e-3 because of rank order among exactly tied similarities (duplicate embeddings). This is deterministic and the same in every split.
  - `nd_top5u` mismatched on 168 val rows. That led to the bug in V.2.

### V.2 Bug fixed: zero user vectors in `nd_top5u`, `nd_knn_u` and `nd_knn_u_dev`

**The bug.** When `user_emb` is all zeros, the base `cos_u` is **0, not NaN**, for every candidate. The kernel only treated NaN as missing, so in those feeds:
- "top-5 candidates by cos_u" was an arbitrary tie order (numba quicksort);
- `nd_knn_u` and `nd_knn_u_dev` collapsed to a fake 0, which sits in the middle of the real `knn_u_dev` distribution.

`nd_ucent` already handled this case with `uok` and returned NaN.

**Affected rows.** Train 2.6%, val 1.4% and test 0.26% of rows. Of the affected test feeds, 71% have no history and 39% are user 0.

**The fix.** Two lines in `_nd_kernel` now also require `uok`, so the three features become NaN in those feeds. I rebuilt the three files for train, val and test with the builder's command:
- `CTX_KEEP=ctx_nd_top5u,ctx_nd_knn_u,ctx_nd_knn_u_dev python build_ctx.py <split> nd 24`;
- train took 42 s, peak RSS 39 GB;
- rows changed outside zero-user feeds: 0.

**Effect.** Fixed minus pre-fix, paired over 15 seed x order runs: **+0.00016 +/- 0.00030**. That is neutral, but the features are now well defined.

**Downstream:** `stk2_res4_clean` (lead) uses `ctx_nd_*`. Any model fitted on the old files should be refit. Otherwise those three columns are inconsistent on the 0.26% of test rows that come from zero-user feeds.

### V.3 Harness re-run (5 seeds; official cached base OOFs s0-s4; base mean 0.36356)

`vrun.py` imports `cv_oof` / PARAMS / ROUNDS / folds from `src/eval_family.py`. For alphabetical order it reproduces the builder's per-seed deltas exactly (0.00139 / 0.00036 / 0.00049 on seeds 0 / 1 / 3).

**Placebo control.** The same 24 columns, each permuted *within feed*. This keeps each feed's values and NaN pattern but breaks the link to the candidate.

| run | 5-seed delta | SE | per-seed | 5v5 seed-ensemble delta |
|---|---|---|---|---|
| **official `eval_family.py --family ctx_ --seeds 5` (post-fix, tag `ctx_VERIFY_KEEP24fix_s5`)** | **+0.00201** | 0.00067 | .00116 / .00167 / .00325 / .00109 / .00290 | |
| pre-fix, alphabetical (= `--family ctx_`) | +0.00155 | 0.00067 | .00139 / .00036 / .00430 / .00049 / .00119 | +0.00194 |
| pre-fix, random order 1 | +0.00257 | 0.00068 | .00315 / .00146 / .00359 / .00218 / .00247 | +0.00383 |
| pre-fix, random order 2 | +0.00252 | 0.00067 | .00166 / .00031 / .00571 / .00243 / .00252 | +0.00292 |
| post-fix, alphabetical (wrapper) | +0.00229 | 0.00067 | .00122 / .00167 / .00429 / .00135 / .00290 | +0.00282 |
| post-fix, random order 1 | +0.00264 | 0.00067 | .00241 / .00244 / .00380 / .00172 / .00282 | +0.00383 |
| post-fix, random order 2 | +0.00220 | 0.00068 | .00242 / .00070 / .00271 / .00333 / .00182 | +0.00294 |
| **placebo (within-feed permuted KEEP 24)** | **-0.00049** | 0.00056 | -.00082 / -.00158 / +.00187 / -.00014 / -.00180 | -0.00020 |

What this shows:
- **All 15 post-fix runs (3 orders x 5 seeds) are positive:** mean +0.00237, min +0.00070, max +0.00429, run-level sd 0.00099. Pre-fix: mean +0.00221, min +0.00031.
- The placebo is negative on average: adding 24 uninformative columns costs about 0.0005. So the delta of the real features is not an artifact of adding columns.
- **Seed 2 is biased upward.** `base_oof_s2` is a low draw (0.3619 vs about 0.3640 for the others), and even the placebo gains +0.0019 on seed 2. Any 3-seed delta is inflated by about +0.0006 from this alone. Without seed 2, the official-order delta is +0.0009 (pre-fix) / +0.0018 (post-fix). The random orders stay at +0.0017 to +0.0023.
- **LightGBM is not bit-deterministic under load.** The official harness and my wrapper got identical inputs but differ on seeds 2 and 3 (0.00325 vs 0.00429, 0.00109 vs 0.00135). This adds noise on top of seed and column order. I confirm the builder's point that a single 3-seed delta can move by about 0.001.

**Ablations (post-fix, 5 seeds, 2 column orders each, paired on the same seeds):**

| set | alphabetical | random order 1 |
|---|---|---|
| KEEP 24 | +0.00229 | +0.00264 |
| KEEP 20 (without `hs_k256`, `hs_e1k64`, `hs_k128t10`, `hs_mutual`) | +0.00233 | +0.00202 |
| hn+hs 14 (no nd) | +0.00202 | +0.00223 |
| nd 10 alone | +0.00136 +/- 0.00057 | |

Marginal contributions:
- KEEP 24 minus KEEP 20 (the 4 extra hs variants): +0.00029 +/- 0.00040 (single model), +0.00039 +/- 0.00070 (10 vs 10 ensemble).
- KEEP 24 minus hn+hs 14 (nd on top of hn/hs): +0.00034 +/- 0.00045 (single model), +0.00094 +/- 0.00077 (ensemble).

Both are small and positive but under 2 SE, which agrees with the builder's forward selection. The hn/hs core carries most of the gain.

### V.4 Time-split check (LightGBM, `eval_family` PARAMS, 350 rounds; train on one val day, evaluate on the other; the 10 eval users who also appear on the training day are excluded)

| direction | eval feeds | base | KEEP 24 alphabetical | KEEP 24 random order 1 | placebo |
|---|---|---|---|---|---|
| Tue 22 -> Thu 24 (post-fix) | 6,870 | 0.3618 | **+0.00414 +/- 0.00101** (ens +0.00461) | +0.00345 +/- 0.00102 | -0.00055 +/- 0.00087 |
| Thu 24 -> Tue 22 (post-fix) | 7,533 | 0.3590 | **+0.00280 +/- 0.00096** (ens +0.00360) | +0.00470 +/- 0.00097 | +0.00064 +/- 0.00090 |
| Tue -> Thu (pre-fix) | 6,870 | 0.3618 | +0.00308 +/- 0.00102 | +0.00375 | 0.00000 |
| Thu -> Tue (pre-fix) | 7,533 | 0.3590 | +0.00361 +/- 0.00097 | +0.00481 | -0.00058 |

- The gain carries over to a different day: about 3-4 SE in each direction.
- It is *larger* with half-size training data, which fits the builder's view that compact, smooth features help small-data models most.
- The placebo stays at about 0.

### V.5 Drift (val vs test-public vs test-private, boundary ts 1790497920; weekday vs weekend)

PSI uses 10 val-quantile bins plus a NaN bin. "we-wd" is test Sat+Sun vs Mon+Tue; "privWE-WD" is private Sun vs private Mon+Tue.

| feature | val-pub | val-priv | pub-priv | we-wd | privWE-WD | NaN val / pub / priv |
|---|---|---|---|---|---|---|
| hn_* (6) and hs_* (8), worst case `hn_e1_rrw` | <= 0.0073 | <= 0.0029 | <= 0.0012 | <= 0.0010 | <= 0.0018 | 0.014 / 0.008 / 0.009 (e1: 0.031 / 0.020 / 0.023) |
| nd_knn_h_dev, knn_h, max, hcent, knn_f1, top5h | <= 0.0038 | <= 0.0025 | <= 0.0005 | <= 0.0006 | <= 0.0013 | <= 0.014 / 0.008 / 0.009 |
| nd_ucent, nd_knn_u, nd_knn_u_dev, nd_top5u (post-fix) | 0.027-0.028 | 0.017 | 0.001 | <= 0.0014 | <= 0.0011 | 0.014 / 0.002 / 0.003 |

- The only PSI above 0.01 comes entirely from the **NaN bin**: zero user vectors are 1.4% of val rows vs 0.26% of test rows. That is a data property and the definition is the same in both. The non-NaN medians are identical, for example `nd_top5u` 0.5375 / 0.5361 / 0.5358.
- Per-day medians from Sat 26 to Tue 29 are flat, and there is no weekend effect: we-wd PSI <= 0.0014.
- Adversarial AUC using only the 24 ctx features:
  - val vs test: 0.516 (mostly the NaN-share difference);
  - public vs private: 0.503;
  - test weekday vs weekend: 0.506;
  - val vs test weekdays: 0.515.

### V.6 Generalization to Private's colder item mix

"Unseen" below means the candidate item_id never appears as a train candidate. The unseen row share is 0.346 in val, 0.436 in public and 0.473 in private.

- **Reweighting.** I averaged the per-feed delta over all 15 post-fix runs and reweighted val feeds to the test mix over 5 unseen-share x 4 M cells:
  - public: +0.00198 +/- 0.00074;
  - **private: +0.00170 +/- 0.00088**;
  - unweighted: +0.00237.
- **Where the weight comes from.** Feeds with unseen share >= 0.5 are 12% of val but 42% of private. In that bin the delta is -0.00015 +/- 0.00193, which is noise.
- **The features do not weaken there.** On those feeds `hs_t10` scores 0.422 single-feature NDCG vs 0.418 for `maxsim32`; on the other feeds it is 0.344 vs 0.339. So expect roughly +0.0015 to +0.0024 on Private, not more.

### V.7 Train files

- **Shape and type.** All 24 exist for train, val and test with the right lengths (246,745,836 / 1,389,971 / 2,904,806 rows) and float32.
- **No constant or all-NaN columns.** Sample unique counts: `hs_mutual` has 529 unique values; the others have 25K to 995K.
- **NaN rates.** Train 1.2% (hist-based), 2.8% (e1) and 0.004% (nd non-user). Val: 1.4%, 3.1% and 0.015%.
- **Signal direction.** Single-feature NDCG on 60K-feed samples of Sep 6 and Sep 19 points the same way as on val for every feature (`bestrk` is inverted on both). Examples, Sep 6 / Sep 19 / val:
  - `hs_d7`: 0.369 / 0.370 / 0.352;
  - `hs_t10`: 0.368 / 0.369 / 0.353;
  - `nd_hcent`: 0.354 / 0.355 / 0.339.

  The values do not drift within train.
- **Expected train/val shift.** Production candidates sit closer to history (median `hs_k128` 0.54 train vs 0.33 val), the same shift every similarity feature has.
- **Correlation with position.** `hn_rrw` and `hn_e1_rrw` correlate about -0.46 with the production `pos` on train. That is mostly feed size, because rrw is not normalized by M. It is harmless for within-feed ranking, but worth knowing for train-fit models.

### V.8 Per-feature verdicts

| feature(s) | verdict | evidence |
|---|---|---|
| `ctx_hn_soft05`, `ctx_hn_rrw`, `ctx_hn_e1_rrw`, `ctx_hn_top1w`, `ctx_hn_margin`, `ctx_hn_bestrk` | **KEEP** | Core of the gain: hn+hs 14 gives +0.0020 / +0.0022 (2 orders).<br>Single-feature NDCG 0.335-0.353.<br>Clean, no drift, reimplementation matches. |
| `ctx_hs_t10`, `ctx_hs_d7`, `ctx_hs_k128`, `ctx_hs_e1` | **KEEP** | Strongest single features (0.349-0.354).<br>`hs_k128` has top LightGBM gain.<br>`hn_soft05`, `hs_t10` and `hs_d7` are near-duplicates (Spearman 0.96-0.97), which is fine for trees. |
| `ctx_hs_k256`, `ctx_hs_e1k64`, `ctx_hs_k128t10`, `ctx_hs_mutual` | **KEEP (optional)** | Marginal +0.00029 +/- 0.00040 here, +0.00072 +/- 0.00054 in the builder's run: positive but under 2 SE.<br>`k256` and `k128t10` have Spearman 0.96 with `k128`.<br>Cheap and drift-free; drop these first if columns must be cut. |
| `ctx_nd_knn_h_dev`, `ctx_nd_knn_h`, `ctx_nd_max`, `ctx_nd_hcent`, `ctx_nd_knn_f1`, `ctx_nd_top5h` | **KEEP** | nd 10 alone +0.00136 +/- 0.00057.<br>Marginal over hn/hs +0.0003 (single) / +0.0009 (ensemble).<br>PSI <= 0.004. |
| `ctx_nd_ucent` | **KEEP** | Its PSI of 0.028 / 0.017 comes only from the NaN share (zero user vectors 1.4% val vs 0.26% test); non-NaN values are identical. |
| `ctx_nd_top5u`, `ctx_nd_knn_u`, `ctx_nd_knn_u_dev` | **KEEP (fixed)** | Zero-user tie bug fixed (V.2) and rebuilt for all splits.<br>After the fix they share `nd_ucent`'s NaN pattern. |

**Final KEEP (24, unchanged):** the list in §5.

**Corrections to the builder's numbers:**
- `best_subset_delta` 0.0026 is slightly optimistic: it averages a favourable column order with the alphabetical one. The 3-order x 5-seed mean is +0.0024 post-fix, the official 5-seed run is +0.0020, and the estimate reweighted to Private is +0.0017.
- The builder's "honest +0.0015 to +0.0030" range is confirmed.
