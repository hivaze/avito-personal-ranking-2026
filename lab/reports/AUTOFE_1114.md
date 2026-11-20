# AUTOFE 11-14: automatic feature search on top of W8

Scripts, logs and outputs: `nn_x/w12/autofe/`. CPU only, ≤ 12 threads. No submission files, no existing file modified.

## 0. TL;DR

1. **Nothing passes.** 11,617 automatically generated features (12 operator families, three waves) and 19 nested blocks were tested on top of W8 (val 0.40165). No block and no feature meets the bar (val ≥ +0.0005 at ≥ 2 SE in the user-half nested test AND holdout ≥ 0 AND above the false-positive rate).
2. **The search behaves exactly like its noise controls.**
   - Share of features with nested val z ≥ 2: 1.02% (linear tilt) and 0.38% (binned tilt), against 1.01% and 0.25% for 397 label-free noise features.
   - Formal single-feature passes: 13 of 23,234 tests; the noise controls predict about 29.
   - Features with val z > 2 and holdout z > 2 together: 0.
   - Correlation between val z and holdout z across features: −0.06.
3. **Blocks.** With selection, fit and hyper-parameters inside the fitting half, the best block is +0.00006 ± 0.00016 on val (ridge, monotone transforms). Forced fits cost −0.00004 to −0.0011 on val; the noise block costs −0.0003 / −0.0004.
4. **OpenFE itself** (library v0.0.12 on half A, W8 as init score, about 11,400 candidates): half A declined both of its blocks. Forced: top-10 +0.00009 ± 0.00042 on half B, top-40 +0.00037 ± 0.00044; holdout +0.00035 ± 0.00017 and −0.00006 ± 0.00018, against +0.00025 ± 0.00018 for the forced noise block.
5. **The harness can see a real effect.** A planted pair interaction worth 0.0009 val / 0.0019 holdout was ranked in the top 3 of 3,840 features in both halves and recovered at +0.0006 val (z 1.1) and +0.0012 holdout (z 4.5). So val alone cannot certify a block effect below about +0.0011; the holdout read is twice as sharp, and it is flat here.
6. **Two representation checks that a residual model cannot make** (§6):
   - sentinel −1.5 inflates the train std of the similarity inputs 1.9-2.3×, and of the 10 per-event-type inputs 2.7-6×, so their real variation enters the first layer at 0.17-0.52 SD;
   - test-time standardisation of the dense inputs (val statistics instead of train statistics) is strongly negative: −0.0024 to −0.0057 on W5 members across six variants (SE 0.0006-0.0008). The train→val level shift of the inputs is information, not a nuisance. Closed.
7. **NN-input recipes** for four marginal blocks (33 columns, float16, train / val / test) are built and ready (§7). None is recommended on evidence; they exist because the earlier audit never tested NN-input form. Not launched.

## 1. Protocol

| item | value |
|---|---|
| base on val | 0.5·z(ens_W8kd_ft) + 0.5·z(ens_W8kd_to), NDCG 0.40165, 14,423 feeds |
| base on holdout | mean rank-z of nn_w5_e10_kd05w192_s951..954, NDCG 0.32408, 30,000 feeds |
| split | user hash → halves A / B (7,278 / 7,145 feeds); a second hash bit gives an inner split of each half |
| per-feature test, linear | base + w·z_feed(x); w from a 10-point grid ±0.01..±0.16 or 0, picked on one half by NDCG, scored on the other; both directions concatenated, per-feed SE |
| per-feature test, binned | base + η·d[bin(x)], 24 quantile bins, d = Newton step of the listwise softmax loss on the fitting half, η ∈ {0, 0.25, 0.5, 1} picked on the fitting half |
| selection score for blocks | inner-CV binned Newton gain (1-D, and × 4 base-rank groups) computed inside the fitting half only |
| block models | LightGBM with exact listwise softmax-CE objective, init_score = base (7 leaves, lr 0.03, min leaf 3000, L2 100; rounds ∈ {0, 25, 50, 100, 200} picked on the inner split); ridge on within-feed z (λ ∈ {none, 3, 0.3, 0.03} picked on the inner split) |
| holdout read | selection and fit on full val, applied to the holdout |
| controls | 397 label-free features (N(0,1) and within-feed permutations of real inputs) through the identical pipeline; a block of the 40 best controls; the 56 raw inputs as a block |

All features are label-free and computed per feed or with constants frozen from val inputs (`cache/stats.json`), so test rows get identical values. History-item tables (mean f4, f5, popularity per item) come from train candidate rows and are applied to history items only.

**Harness bug caught and fixed.** LightGBM's `rank_xendcg` cannot be used with an init score here: it cost −0.02 to −0.03 NDCG even for noise features (my reading, not verified: its randomised soft targets pull a sharp base score back towards a flat one). The first block run used it and reported vacuous zeros through the do-nothing option; all block numbers below are from the corrected exact-softmax objective (`blocks_screen_v3*.json`). `blocks_main.json` and `blocks_screen_forced.json` are the superseded broken runs.

## 2. Inventory

Full table of the 56 per-candidate inputs plus feed-level inputs: `nn_x/w12/autofe/inv_table.md` (CSV: `inventory.csv`). Columns: NN scaling, zeros, NaN share val / train, mass point, skew / kurtosis as fed, max |z| on test, PSI train→val and val→test, between-feed variance share, whether the model has a within-feed relative version. The in-model similarity inputs are represented by their precomputed stand-ins in `data/feats` with the model's −1.5 sentinel. Train = 20,000 random non-holdout feeds.

Condensed view:

| group | inputs | NN scaling | distribution as fed | PSI train→val | PSI val→test | flag |
|---|---|---|---|---|---|---|
| f1..f5 | 5 | raw + feed z + feed rank | quantile-coded, kurtosis −1.1..−1.3; mass points 17.6% (f1), 15.5% (f2 = 0), 10.8% (f5) | 0.003-0.031 | ≤ 0.001 | none |
| history similarity (smax, top3, mean, smax32) | 4 stand-ins | raw; z + rank for smax, lse, top3 only | sentinel −1.5 on 1.4% of rows (in the model: all but the mean); skew −5.2..−5.9, kurtosis 34-41; 77-83% of variance is between feeds | 0.04-0.14 | ≤ 0.001 | sentinel mixed with values (§6.1) |
| per-event-type max similarity | 10 | raw only | sentinel on 3% (et1) to 93% (et10) of rows; et1..et4: 90-98% of variance between feeds | 0.01-0.04 (et1..et4) | ≤ 0.001 | sentinel dominates the scale (§6.1) |
| cos_u | 1 | raw + z + rank | bounded | 0.130 | 0.002 | level shift −0.24 SD on val |
| category affinity (counts, share, recency, feed share) | 7 | log1p for counts, raw otherwise | 15-50% zeros, kurtosis ≤ 1.4 as fed | ≤ 0.02 | ≤ 0.002 | none |
| co-visitation seq_cv_invd20 / norm | 2 | log1p | 88.6% zeros; after log1p skew 11.7 / 15.0, kurtosis 189 / 331; max \|z\| 27 / 35 (clamped at 6) | 0.011 | 0.002 | sparse + heavy tail |
| seq_cv_pop | 1 | log1p | 42% zeros, kurtosis 0 | 0.002 | 0.012 | none |
| leave-user-out popularity | 5 | log1p | 38-75% zeros, kurtosis 0-6 | 0.15-0.46 | 0.003-0.019 | known coverage drift; level −0.65 SD val, −0.75 SD test |
| ctx_nd (neighbour density) | 10 | clip ±10 | kurtosis ≤ 8; knn_h / knn_u / max shift −0.23..−0.33 SD on val | 0.01-0.20 | ≤ 0.002 | level shift |
| ctx_hn_top1w, ctx_hs_mutual | 2 | clip ±10 | 87.5% / 89.8% zeros; kurtosis 97 / 204; max \|z\| 18 / 44 | ≤ 0.01 | 0.000 | sparse + heavy tail |
| ctx_hn_rrw, e1_rrw | 2 | clip ±10 | skew 4.2-4.4, kurtosis 26-28; max \|z\| 12 | 0.03 | 0.001 | heavy tail |
| ctx_hs_k128 / e1 / t10, ctx_hn_soft05 | 4 | clip ±10 (cap binds) | skew 2.9-3.1, kurtosis 11-12 | 0.03-0.08 | ≤ 0.001 | known cap (audit §1.3a) |
| pre2 scalars | 5 + flag | z clip ±5; `_s` and hmax32 only as feed z / rank | NaN→0 on 28.4% of train rows, never on val / test; flag constant 1 on val / test | 0.42-0.50 (flag 2.35) | ≤ 0.001 | known blank regime (audit §1.3d) |
| feed level: log M, log H, user0 | 3 | log / raw | H at the 256 cap on 29% of val feeds | 0.06-0.09 | ≤ 0.004 | none |
| feed cold share (derived, not an input) | 1 | — | — | 3.66 | 0.36 | known |
| categoricals (category 52, event type 10, source 27) | — | learned embeddings | — | 0.003-0.027 | ≤ 0.002 | none |

Reading:
- Val→test PSI is ≤ 0.02 for every input; the only large val→test drift is the feed cold share. Agrees with the audit.
- No unlogged count and no unbounded ratio enters the model. What is left is (a) sentinels mixed with real values, (b) four sparse heavy-tailed columns, (c) a train→val/test level shift of −0.2 to −0.75 train-SD on similarity and popularity inputs.
- Candidates never repeat an item of the same feed's history (0 matches in 1.39M val and 4.63M holdout candidates). "Seen before" features are structurally empty.

## 3. Automatic search: per-feature results

Generated families (constant features dropped; names are formulas, e.g. `rmax(cos_u,ctx_nd_knn_h)`):

| family | content | n |
|---|---|---|
| A_feedrel | per input: feed rank, deviation from feed max, from feed median, gap to next value, ratio to feed mean, robust z | 330 |
| B_mono | per input: log1p, sqrt, square, exp, frozen quantile transform (visible only to the linear test) | 275 |
| C_pair | 561 pairs of 34 core inputs × {z·z, z−z, ratio, min rank, max rank} | 2,805 |
| C3_pair | same pairs × {rank difference, rank product, \|z−z\|, max z} | 2,243 |
| C2_pair | the other 924 pairs of the 55 inputs × the first five operators | 4,620 |
| C_ctx | core input z × feed context (log M, log H, user0, cold share, categories per candidate, feed mean similarity) | 204 |
| D_catgrp | within (feed, category): deviation from group mean / max, rank in group, group mean − feed mean | 116 |
| D2_grp | the same within (feed, f2 region level) and (feed, cold / warm) | 200 |
| E_hist, E2_hist | history crosses: cosine to the centroid of the last 1 / 2 / 4 / 16 / 64 events; same-category max / mean / centroid, other-category max; counts above 0.6-0.9; decayed mean; argmax recency; f4, f5 and popularity minus the mean over history items (all / same category / last 8); repeats | 54 |
| F_emb | candidate-candidate: nearest neighbour, counts above 0.8 / 0.9, feed centroid, same-category centroid and neighbour, top-5 mean; 8 PCA projections of the candidate, × user projection, minus history mean | 62 |
| G_efx | the 60 new scalars × 6 anchors (cos_u, f3, f4, f5, smax, popularity), product and max rank | 690 |
| H_het | masked feed-relative versions of per-type similarity et5..et10 | 18 |
| CTRL | label-free noise | 397 |

Results (nested val; "pass" = z ≥ 2 and Δ ≥ +0.0005 and holdout ≥ 0):

| family | n | linear z ≥ 2 | linear mean z (sd) | linear pass | binned z ≥ 2 | binned pass | holdout linear: negative / zero / positive |
|---|---|---|---|---|---|---|---|
| A_feedrel | 330 | 4 (1.2%) | −0.29 (0.99) | 0 | 3 | 0 | 186 / 18 / 126 |
| B_mono | 275 | 5 (1.8%) | −0.28 (1.04) | 0 | 0 | 0 | 159 / 11 / 105 |
| C_pair | 2,805 | 24 (0.9%) | −0.43 (0.96) | 2 | 11 | 2 | 1,774 / 144 / 887 |
| C3_pair | 2,243 | 24 (1.1%) | −0.35 (0.99) | 0 | 8 | 1 | 1,381 / 96 / 766 |
| C2_pair | 4,620 | 49 (1.1%) | −0.24 (0.98) | 4 | 13 | 1 | 2,668 / 293 / 1,659 |
| C_ctx | 204 | 2 (1.0%) | −0.36 (0.98) | 0 | 1 | 0 | 140 / 11 / 53 |
| D_catgrp | 116 | 2 (1.7%) | −0.48 (1.10) | 0 | 1 | 0 | 77 / 5 / 34 |
| D2_grp | 200 | 2 (1.0%) | −0.39 (1.00) | 0 | 1 | 0 | 103 / 7 / 90 |
| E_hist + E2_hist | 54 | 0 | −0.66 | 0 | 0 | 0 | 28 / 4 / 22 |
| F_emb | 62 | 0 | −0.42 (1.09) | 0 | 2 | 0 | 45 / 1 / 16 |
| G_efx | 690 | 6 (0.9%) | −0.45 (1.09) | 1 | 3 | 2 | 394 / 18 / 278 |
| H_het | 18 | 0 | +0.08 (0.67) | 0 | 1 | 0 | 9 / 0 / 9 |
| **all generated** | **11,617** | **118 (1.02%)** | **−0.33 (0.99)** | **7** | **44 (0.38%)** | **6** | 6,964 / 608 / 4,045 |
| **CTRL (noise)** | **397** | **4 (1.01%)** | **−0.27 (1.03)** | **1** | **1 (0.25%)** | **0** | 243 / 25 / 129 |

False-positive accounting:
- z ≥ 2 anywhere: 162 of 23,234 tests (0.70%). The noise controls predict 146; a nominal N(0,1) would give 530. The do-nothing option and the tilt penalty push the null mean to −0.3 z.
- Formal passes: 13 (0.056%). The noise controls give 1 of 794 (0.13%), i.e. about 29 expected.
- Val z > 2 and holdout z > 2 together: 0 features (about 0.7 expected by chance). Val z > 1.5 and holdout z > 1.5: 8 linear + 5 binned.
- Holdout z < −2 on 5.0% of generated features and 5.8% of controls: an uninformative tilt costs about −0.00005 on the holdout, as the audit found.

The 13 formal passes, all at z 2.0-2.9 with a holdout z ≤ 1.2:

| feature | form | val Δ (SE) | z | holdout Δ (z) |
|---|---|---|---|---|
| zmax(feed_cat_share, ctx_hn_soft05) | binned | +0.00058 (0.00020) | 2.88 | +0.00008 (0.6) |
| ratio(cos_het1, item_norm) | linear | +0.00053 (0.00021) | 2.54 | +0.00001 (0.1) |
| rmax(ctx_nd_max, ctx_hs_e1) | binned | +0.00067 (0.00028) | 2.42 | +0.00011 (0.6) |
| rmin(cos_het1, ctx_hs_k128) | linear | +0.00051 (0.00023) | 2.25 | +0.00005 (0.4) |
| rmax(seq_max_all, cat_cnt_et3) | binned | +0.00055 (0.00025) | 2.22 | +0.00014 (0.9) |
| zmul(e_last16, f5) | linear | +0.00057 (0.00026) | 2.16 | +0.00013 (0.9) |
| zmul(cat_share_all, item_norm) | binned | +0.00065 (0.00030) | 2.16 | +0.00011 (0.7) |
| zmul(pc4_c, f3), zmul(pc4_dh, f3) (same column within feed) | binned | +0.00052 (0.00025) | 2.12 | +0.00005 (0.4) |
| zmul(it_hist_et4_luo, ctx_nd_knn_h) | linear | +0.00053 (0.00025) | 2.12 | +0.00006 (0.5) |
| rmax(cos_u, ctx_nd_knn_h) | linear | +0.00058 (0.00028) | 2.06 | +0.00006 (0.4) |
| rmin(it_hist_all_luo, ctx_nd_knn_f1) | linear | +0.00056 (0.00028) | 2.01 | +0.00004 (0.3) |
| rmin(it_hist_et4_luo, ctx_nd_top5h) | linear | +0.00050 (0.00025) | 2.01 | +0.00017 (1.2) |

The two largest val z overall, ratio(f2, ctx_nd_knn_f1) at +0.00068 (z 3.4) and zmul(cos_het3, it_hist_et4_luo) at +0.00075 (z 3.2), are negative on the holdout (z −1.3, −1.1). Under the observed null (z mean −0.33, sd 0.99) about 4 values above z 3 are expected among the 11,617 linear tests alone.

The stage-1 selection score does not replicate either: the top-40 lists of the two halves share 7 features, mostly the same sparse inputs under different monotone transforms; 2.7% / 3.7% of generated features have a positive inner-CV gain against 3.8% / 5.0% of the controls.

## 4. Block tests

Corrected GBM objective. Δ vs W8 alone, ± SE, per-feed paired. "choice" = rounds (GBM) or λ (ridge) picked inside the fitting half; 0 / none = the half itself declined to use the block, Δ is then exactly 0 for the other half.

| block | selection | GBM val | GBM holdout | ridge val | ridge holdout |
|---|---|---|---|---|---|
| RAW (56 current inputs) | none | 0 (declined ×2) | 0 | 0 (declined ×2) | 0 |
| CTRL40 (noise) | inner-CV gain | −0.00037 ± 0.00039 | +0.00006 ± 0.00011 | −0.00023 ± 0.00018 | −0.00003 ± 0.00012 |
| ALL10 | inner-CV gain, all families | −0.00025 ± 0.00040 | −0.00046 ± 0.00027 | 0 (declined ×2) | 0 |
| ALL40 | same | −0.00092 ± 0.00051 | −0.00011 ± 0.00022 | 0 (declined ×2) | 0 |
| ALL100 | same | −0.00141 ± 0.00053 | −0.00017 ± 0.00020 | 0 (declined ×2) | 0 |
| LIN10 | in-half NDCG gain of the linear tilt | −0.00048 ± 0.00043 | +0.00014 ± 0.00014 | −0.00045 ± 0.00038 | +0.00004 ± 0.00014 |
| LIN40 | same | −0.00005 ± 0.00031 | −0.00006 ± 0.00014 | −0.00036 ± 0.00049 | +0.00003 ± 0.00020 |
| A_feedrel top 20 | inner-CV gain | −0.00002 ± 0.00018 | −0.00012 ± 0.00012 | 0 (declined ×2) | 0 |
| B_mono top 20 | same | n/a (trees are invariant) | n/a | +0.00006 ± 0.00016 | −0.00005 ± 0.00012 |
| C_pair top 20 | same | −0.00042 ± 0.00044 | +0.00007 ± 0.00013 | −0.00081 ± 0.00047 | −0.00013 ± 0.00019 |
| C2_pair top 20 | same | −0.00030 ± 0.00029 | +0.00016 ± 0.00013 | 0 (declined ×2) | 0 |
| C3_pair top 20 | same | −0.00073 ± 0.00048 | +0.00018 ± 0.00019 | 0 (declined ×2) | 0 |
| C_ctx top 20 | same | −0.00026 ± 0.00041 | −0.00014 ± 0.00013 | −0.00008 ± 0.00020 | −0.00004 ± 0.00010 |
| D_catgrp top 20 | same | −0.00033 ± 0.00034 | +0.00010 ± 0.00012 | −0.00025 ± 0.00029 | −0.00010 ± 0.00012 |
| D2_grp top 20 | same | 0 (declined ×2) | 0 | 0 (declined ×2) | 0 |
| E_hist top 20 | same | +0.00003 ± 0.00017 | −0.00010 ± 0.00012 | 0 (declined ×2) | 0 |
| F_emb top 20 | same | −0.00006 ± 0.00028 | +0.00003 ± 0.00017 | +0.00002 ± 0.00028 | +0.00015 ± 0.00015 |
| G_efx top 20 | same | −0.00054 ± 0.00039 | −0.00001 ± 0.00012 | −0.00019 ± 0.00028 | +0.00005 ± 0.00017 |
| EF_all (all 134 history / embedding scalars) | none | −0.00066 ± 0.00038 | +0.00006 ± 0.00014 | −0.00033 ± 0.00027 | +0.00003 ± 0.00017 |

- Best val: B_mono ridge +0.00006 (z 0.3). Best holdout: C3_pair GBM +0.00018 (z 0.9) with val −0.00073.
- No block is positive by 1 SE on both reads. The closest is F_emb ridge: +0.00002 val (z 0.1), +0.00015 holdout (z 1.0).

Forced fits (no do-nothing option: GBM 100 rounds, ridge λ 0.3), to show the magnitudes:

| block | GBM val | GBM holdout | ridge val | ridge holdout |
|---|---|---|---|---|
| CTRL40 (noise) | −0.00026 ± 0.00040 | +0.00025 ± 0.00018 | −0.00044 ± 0.00043 | −0.00013 ± 0.00020 |
| RAW | −0.00094 ± 0.00041 | +0.00020 ± 0.00020 | −0.00094 ± 0.00045 | −0.00048 ± 0.00021 |
| ALL40 | −0.00064 ± 0.00046 | −0.00011 ± 0.00022 | −0.00042 ± 0.00043 | −0.00007 ± 0.00020 |
| ALL100 | −0.00099 ± 0.00049 | −0.00017 ± 0.00020 | −0.00112 ± 0.00052 | −0.00025 ± 0.00022 |
| best other block on val | D2_grp −0.00014 ± 0.00046 | −0.00008 | B_mono −0.00004 ± 0.00034 | 0.00000 |
| best other block on holdout | LIN10 −0.00097 val | +0.00016 ± 0.00018 | F_emb −0.00011 val | +0.00018 ± 0.00019 |

Every forced block is ≤ 0 on val (−0.00004 to −0.0011 across 19 blocks, both models). The holdout range is −0.0005 to +0.0003, and the noise block sits at its upper end.

### OpenFE library cross-check

`openfe==0.0.12` was installed into the project venv (no other package changed). Setup:
- Data: 3,000 feeds of half A (286,057 rows), feed-level 75 / 25 split inside the sample; 56 inputs + the W8 score; `feed` and `category` as categorical keys, so OpenFE builds its GroupByThen{Rank, Mean, Median, Min, Max, Std} features per feed and per category.
- Task: binary log-loss with init score = a·log-softmax(W8) + b, calibrated on half A.
- OpenFE's default operator set gives about 11,400 candidates (9,576 pair features with + − × ÷ min max, 1,368 group-by features, 399 unary); 8 data blocks of successive halving, then its stage-2 importance ranking. Fit time 182 s at 12 processes; 1,176 features returned.
- Its top 10 and top 40 were then put through the same nested GBM test: fit on all of half A, score half B (one direction only), holdout by the half-A model.

| block | half A inner choice | forced 50 rounds, half B | forced 50 rounds, holdout |
|---|---|---|---|
| OpenFE top 10 | declined (0 rounds) | +0.00009 ± 0.00042 | +0.00035 ± 0.00017 |
| OpenFE top 40 | declined (0 rounds) | +0.00037 ± 0.00044 | −0.00006 ± 0.00018 |
| reference: forced noise block (full-val fit, 100 rounds) | — | −0.00026 ± 0.00040 | +0.00025 ± 0.00018 |

- Under the strict protocol both blocks are exactly 0: the inner split of half A found no round count with a positive gain.
- The forced top-10 holdout read (+0.00035, z 2.1) is not evidence: the forced noise block reads +0.00025 on the same holdout, because every forced GBM also re-shapes the base score through its `zb` columns. The top-40 block, which contains the top 10, reads −0.00006.
- OpenFE's top formulas are the same kind of object as my families: `GroupByThenRank(f1, cat)`, `GroupByThenRank(ctx_hn_e1_rrw, cat)`, `(seq_cv_pop − pre2_fe_z)`, `min(f5, ctx_nd_top5u)`, `(ctx_hn_rrw / ctx_hs_t10)`. Full list: `openfe_features.json`.

### Power check (planted signal)

The base was damaged with a known construction and the whole pipeline re-run (`plant-*` files).

| planted defect | cost val / holdout | stage-1 rank of the true feature | best block, val | best block, holdout |
|---|---|---|---|---|
| −0.12·z(f4)·z(cos_u) | −0.00087 / −0.00190 | zmul(f4,cos_u): 3rd of 3,840 in half A, 2nd in half B | ALL40 GBM +0.00064 ± 0.00054; C_pair ridge +0.00079 ± 0.00067 | +0.00127 ± 0.00029 (z 4.5); +0.00151 ± 0.00038 |
| −0.10·z(f5) | −0.00072 / −0.00127 | f5 relatives take the top 6 places in both halves | ALL10 ridge +0.00091 ± 0.00058; GBM +0.00070 ± 0.00048 | +0.00131 ± 0.00033 (z 4.0); +0.00089 ± 0.00024 |

- Detection works: the true constructions surface at the top of the selection in both halves, and roughly 65-100% of the damage is recovered. (The second plant was screened on families A, B and D only, 721 features.)
- Val cannot certify it: recovering 0.0007-0.0009 gives z 1.1-1.6, because a block that moves many feeds has a per-feed SE of 0.0005-0.0006. The 2 SE bar on val is reachable only above about +0.0011 for a block. Single-feature tilts with small weights have SE 0.0002-0.0003.
- The holdout sees the same effects at z 3.4-4.6. In the real search the holdout block reads are within ±0.0003 and centred on the noise block.

Bound: a missed residual block worth more than about +0.0006 on the holdout scale would have shown at z > 2 there. Smaller effects are not excluded by this harness.

## 5. Blind spots of the 11-13 audit, checked

| blind spot | what was done here | result |
|---|---|---|
| base was G3 (0.40032) | everything redone on W8 (0.40165) and the w5 holdout pool | null |
| 12 interaction tilts | all 1,485 pairs of the 55 real-valued inputs, 5-9 operators each (9,668 features), plus 204 input × feed-context products | null, identical to noise |
| monotone re-scalings were not tested in linear form | 275 transforms as linear tilts and a ridge block | null (+0.00006 ± 0.00016) |
| no new history / embedding scalars | 134 scalars and 690 crosses with anchors | null; EF_all −0.0003..−0.0007 val, +0.00003..+0.00006 holdout |
| group statistics only by category | also by f2 region level and cold / warm | null |
| selection outside the nest in some probes | selection, fit and hyper-parameters inside the fitting half; label-free controls through the same code | controls match the real features |
| residual form only | 33 NN-input columns built for train / val / test (§7) | not trained (lead's call) |
| scale of sentinel inputs after standardisation | measured (§6.1) | new finding, code-level |
| train→val/test level shift under train standardisation | test-time standardisation on W5 members, CPU forward, 6 variants (§6.2) | −0.0024 to −0.0057; closed |

## 6. Representation ideas that are not column transforms

### 6.1 Sentinel −1.5 sets the scale of the similarity inputs

`dense_features` writes −1.5 for "no history" (smax, smax8, smax32, top3, lse) and for "no event of this type" (10 per-type max-cosine inputs). `feature_stats` then includes those rows in the std. Measured on 20,000 train feeds with the `data/feats` stand-ins:

| input | sentinel share (train) | std all rows | std of real rows | real-row variation after standardisation |
|---|---|---|---|---|
| smax (seq_max_all) | 1.2% | 0.283 | 0.124 | 0.44 SD |
| top3 | 1.2% | 0.282 | 0.127 | 0.45 SD |
| smax32 | 1.2% | 0.292 | 0.153 | 0.52 SD |
| per-type max, et1 | 2.9% | 0.389 | 0.133 | 0.34 SD |
| et2 / et3 / et4 | 16% / 10% / 25% | 0.80 / 0.65 / 0.91 | 0.15-0.16 | 0.19 / 0.24 / 0.18 SD |
| et5..et9 | 38-62% | 1.00-1.03 | 0.17 | 0.17-0.18 SD |
| et10 | 94.5% | 0.493 | 0.181 | 0.37 SD |

- The sentinel is constant within a feed, so it carries no ranking information; it only spends the input's dynamic range. Fifteen raw similarity channels reach the first linear layer 2-6× smaller than every other input.
- The model has feed z / rank only for smax, lse and top3. The per-type inputs, smax8, smax32 and the history mean have no relative version.
- The in-model history mean uses 0, not the sentinel, and is not affected. smax8 and lse have no stand-in file and were not measured; they share the mechanism.
- Residual evidence that this costs anything: none. Masked feed-relative versions of the measured inputs are in A_feedrel and H_het (best: dmax(cos_het1) +0.00044, z 2.2, holdout −0.00005). The typed history attention probably carries the same information.
- Proposal, for the next from-scratch family only: compute mean / std over non-sentinel rows, map sentinel rows to 0 after standardisation, and add one "has type t" flag per event type. Change is confined to `feature_stats` and `Ranker.dense` in `nn_x/w5`. Prior that it moves the pool: low (≤ 0.05); cost: zero extra compute.

### 6.2 Test-time standardisation of the dense inputs

The ranker standardises its 104 dense inputs with train `feature_stats`. On val, 20 of the 104 sit more than 0.15 train-SD away from the train mean (largest −0.64 SD, the popularity inputs; similarity and neighbour-density inputs −0.2 to −0.4 SD). A residual model cannot test whether re-centring helps, so the checkpoint's `f_mean` / `f_std` were swapped for label-free statistics of the val split and the member re-scored on CPU (`tts.py`, `tts2.py`; audit harness, parity corr 1.000000). Paired per-feed NDCG of each member against its own CPU forward (s951: 0.39897):

| variant | columns swapped | members | Δ val NDCG (SE per member) |
|---|---|---|---|
| val mean and std | all 104 | s951, s952, s953 | −0.00489, −0.00572, −0.00524 (0.0008) |
| val mean only | all 104 | s951, s952, s953 | −0.00327, −0.00466, −0.00465 (0.0008) |
| half-way mean and std | all 104 | s951, s952 | −0.00256, −0.00301 (0.0006) |
| val mean and std, without the 4 columns whose scale differs > 1.6× (pre2 flag etc.) | 100 | s951 | −0.00455 (0.00084) |
| val mean only, same 100 columns | 100 | s951 | −0.00362 (0.00078) |
| half-way mean, only the 19 columns shifted by > 0.15 SD | 19 | s951 | −0.00236 (0.00059) |

- Every variant loses 4 to 7 SE on every member tried. The model reads absolute levels (how popular, how similar), and those levels are genuinely lower on val. Re-centring erases that. Closed.
- By-product: the fp32 CPU forward differs from the stored GPU predictions by +0.00038, +0.00001, +0.00015 and −0.00023 NDCG on the four members (corr 1.000000 each) and by −0.00012 ± 0.00015 on their pool. There is no systematic precision effect, but a single member's val NDCG has a numerical floor of about ±0.0003.

### 6.3 Sparse heavy-tailed inputs

seq_cv_invd20, seq_cv_norm, ctx_hn_top1w and ctx_hs_mutual are zero on 87.5-89.8% of rows and keep kurtosis 97-331 after the current scaling; 0.2-0.6% of rows hit the ±6 clamp. A flag plus a frozen quantile transform of the value is the standard cure. In residual form the transforms are null (B_mono). Built as recipe block S.

### 6.4 Things the inventory rules out

- Candidate-in-history repeats: none exist (0 of 6.0M candidates).
- An interaction the model cannot form: 9,668 pair features and 204 context products behave like noise; the planted pair product was found at once. No evidence of a missing second-order term among the current inputs.
- Unlogged counts or unbounded ratios: none enter the model.

## 7. NN-input recipes (not launched)

Arrays: `nn_x/w12/autofe/feats/{train,val,test}/afe_*.npy`, float16, flat candidate order, no NaN, all within ±10. Built by `build_recipes.py <split> [S P C Q]` (train takes about 4 minutes at 12 threads for S + P + C). Names follow `attach_xt`: `*_rk` stays raw in [0, 1]; everything else is clipped at ±10, which never binds.

| block | preset | columns | why |
|---|---|---|---|
| S (14) | `cvnd_pre2s_afeS` | `afe_nz_<x>`, `afe_qn_<x>` for x ∈ {ctx_hn_top1w, ctx_hs_mutual, seq_cv_invd20, seq_cv_norm}; `afe_qn_<x>` for x ∈ {ctx_hn_rrw, ctx_hn_e1_rrw, ctx_hn_soft05, ctx_hs_k128, ctx_hs_e1, ctx_hs_t10} | §6.3: non-zero flag and frozen quantile transform of the sparse / heavy-tailed inputs |
| P (7) | `cvnd_pre2s_afeP` | afe_rmax_cosu_knnh_rk, afe_rmin_luo_knnf1_rk, afe_rmax_ndmax_hse1_rk, afe_rmin_ndmax_hsk128_rk, afe_zmul_catshare_inorm, afe_zdiff_smax_coshall, afe_lratio_top3_coshall | wave-1 pair features with val z ≥ 2 and holdout ≥ 0 |
| Q (5) | `cvnd_pre2s_afeQ` | afe_rmin_het1_hsk128_rk, afe_rmin_luo4_top5h_rk, afe_rmax_smax_catet3_rk, afe_zmul_luo4_knnh, afe_zmax_fcs_soft05 | wave-2/3 formal passes that need no embedding kernel |
| C (7) | `cvnd_pre2s_afeC` | afe_dgm_cos_hall, afe_dgm_f4, afe_grk_f4_rk, afe_grk_f5_rk, afe_grk_seq_max_all_rk, afe_gmf_seq_max_all, afe_gmf_f2 | within-(feed, category) relative versions; grk(f4) is the best holdout read of the family (+0.00034, z 2.3; val +0.00019) |

Not built: the history / embedding scalars (E, F, G). They need the embedding kernel over all 3.15M train feeds (about 95 CPU-minutes at 12 threads; `lib.emb_feats`), and their evidence is the weakest. rmax(cos_u, pre2_new_s) is left out because pre2 is blank on 28% of train feeds.

To run one (the lead's action; nothing below was executed):

```bash
# 1) expose the arrays to feat_path (hard links, no copy)
for s in train val test; do ln <repo>/nn_x/w12/autofe/feats/$s/afe_*.npy <repo>/nn/feat_cache/$s/; done
```

```python
# 2) nn_x/w5/train.py, after the cvnd_pre2s lines
AFE = {
 "S": [f"afe_{k}_{x}" for x in ("ctx_hn_top1w", "ctx_hs_mutual", "seq_cv_invd20", "seq_cv_norm") for k in ("nz", "qn")]
      + [f"afe_qn_{x}" for x in ("ctx_hn_rrw", "ctx_hn_e1_rrw", "ctx_hn_soft05", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10")],
 "P": ["afe_rmax_cosu_knnh_rk", "afe_rmin_luo_knnf1_rk", "afe_rmax_ndmax_hse1_rk", "afe_rmin_ndmax_hsk128_rk",
       "afe_zmul_catshare_inorm", "afe_zdiff_smax_coshall", "afe_lratio_top3_coshall"],
 "Q": ["afe_rmin_het1_hsk128_rk", "afe_rmin_luo4_top5h_rk", "afe_rmax_smax_catet3_rk", "afe_zmul_luo4_knnh", "afe_zmax_fcs_soft05"],
 "C": ["afe_dgm_cos_hall", "afe_dgm_f4", "afe_grk_f4_rk", "afe_grk_f5_rk", "afe_grk_seq_max_all_rk", "afe_gmf_seq_max_all", "afe_gmf_f2"],
}
for k, v in AFE.items():
    XT_PRESETS[f"cvnd_pre2s_afe{k}"] = XT_PRESETS["cvnd_pre2s"] + v
    XT_REL[f"cvnd_pre2s_afe{k}"] = XT_REL["cvnd_pre2s"]
    XT_RELONLY[f"cvnd_pre2s_afe{k}"] = XT_RELONLY["cvnd_pre2s"]
```

```bash
# 3) same command line as w5_e10_kd05w192_s951, with --xt cvnd_pre2s_afeS (or afeP / afeQ / afeC)
```

Notes for whoever runs it:
- `afe_*` names do not start with `seq_cv` / `it_hist`, so they are outside `xt_idcols`; with `--xt_drop 0` (the live recipe) this changes nothing.
- n_dense grows by the block size, so old checkpoints cannot be warm-started.
- Honest prior from this study: each block is a coin flip around zero. If only one lane is free, S is the one with a mechanism (§6.3); P, Q and C are selected-on-noise candidates and most likely cost the usual −0.0002..−0.0005 of an uninformative input block (xrel_more: −0.0019 val).

## 8. Files

| file | content |
|---|---|
| `lib.py` | datasets, numba kernels, all feature generators (formula names → arrays) |
| `screen.py`, `screen_*.jsonl` | per-feature screen and its 12,064 rows |
| `analyze.py`, `analyze_out.txt` | family table, top lists |
| `blocks.py`, `blocks_screen_v3.json`, `blocks_screen_v3forced.json` | nested block tests |
| `plant-*`, `blocks_plant-*` | power check |
| `openfe_run.py`, `openfe_result.json`, `openfe_features.json` | OpenFE cross-check |
| `inventory.py`, `inventory.csv`, `inv_table.md` | inventory |
| `tts.py`, `tts2.py`, `logs/tts.log`, `logs/tts2.log`, `tts_result.json` | test-time standardisation |
| `build_recipes.py`, `feats/` | NN-input arrays |
| `gbm_sanity.py` | objective sanity check |

Leftovers that could not be removed from this session and can be deleted: `screen_test.jsonl`, `blocks_main.json`, `blocks_screen_forced.json`, `openfe_*_smoke.json`.
