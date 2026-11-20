# Error lens: MISSED POSITIVES

Date: 2026-11-11. Scripts and logs: `<repo>/eda/errors/missed_positives/` (`01_frame.py` … `14_fresh_check.log`; each script has a `.log` beside it).

**Conventions**
- Model = the v5 blend 0.5 z(`nn_final2_val_oof`) + 0.5 z(`stk2_res2_r3`): val 0.3869, leak-free 0.3867.
- **The leak mask is applied everywhere.** It drops the 117 val feeds whose non-zero user has a later feed in val or test, leaving 14,306 feeds. Every one of those feeds has at least one positive.
- Ranks are pessimistic and 0-based.
- **Group names:**
  - **MP** (missed positive): a positive ranked at or below M/2.
  - **TP3**: a positive ranked in the top 3.
  - **FA3** (false alarm): a negative ranked in the top 3, in the same feeds.
- "Novel" category = the candidate's category does not occur among the 32 newest history events.
- ± is a paired, feed-level SE unless stated otherwise.

---

## 0. TL;DR

1. **Size of the problem.**
   - 4,533 of 21,759 positives (20.8%) are MPs, spread over 3,827 feeds (26.8%).
   - Those feeds score NDCG 0.277 against 0.427 for the rest, and carry **31.5% of the total (1 − NDCG) mass**.
2. **Who the MPs are.** Mostly *exploration* positives.
   - **42% of MPs are in a novel category** (18% never seen in the history at all, 24% seen only at hpos ≥ 32). That holds for only **5.8% of TP3 and 5.3% of FA3**, and 86% of TP3/FA3 are in a category from the last 8 events.
   - On the model's own signals, MPs look nothing like the items that beat them. Median MP vs FA3 values: maxsim32 0.70 vs 0.93, f1 0.25 vs 0.66, f3 0.31 vs 0.68, co-visitation `seq_cv_invd20` 0 vs 0.32 (mean 0.005 vs 0.315).
   - The FA3 negative wins the within-feed pair on maxsim32 in 89.5% of cases, on `pre_v3_fe` in 88%, on f1 in 83% and on f3 in 75%.
3. **The model is calibrated on MPs; it is not biased against them.**
   - Actual / expected positives per novelty class, where expected comes from the v5 within-feed softmax:

     | category class | act/exp | SE |
     |---|---|---|
     | never in history | 0.95 | 0.023 |
     | only at hpos ≥ 32 | 1.01 | 0.020 |
     | hpos 8-31 | 0.99 | 0.015 |
     | last 8 events | 1.006 | 0.005 |

   - The same holds by maxsim32 quintile (0.95-1.01).
   - Novel categories really are half as likely to be positive (0.66-1.0% vs 2.1%). Ranking them low is the right call given what the model can see. **MPs are mostly irreducible with the current information.**
4. **Exploration intent is worth a lot, but nothing we have can detect it.**
   - An oracle that knows which feeds are "exploration feeds" (15.4% of feeds, where all positives are novel) and lifts novel candidates there would give **+0.032**.
   - **The realistic detector gives nothing.** It uses feed-level features plus the user's own historical exploration rate, reaches AUC 0.71-0.75, and gives **−0.0005 ± 0.00015** (nested). Almost all of that AUC comes from the trivial "share of novel candidates in the feed" (0.706 alone).
   - Which feeds contain an MP is not predictable from user or feed features either: every AUC is ≤ 0.55.
5. **Unexploited signals exist, but they are small in NDCG and they are not about MPs.**
   - In a pairwise residual screen over 530 features, the largest unexploited signal is the pretrain agent's ID-aware sequence encoder (`pre_v3_fe`, t = +10.3). It is not in v5.
   - In the residual stacker it gives **+0.0007 ± 0.0003** (2.1 SE), and **+0.0006 ± 0.0002** inside the v5-style blend. It is about 0 on the Thursday half (+0.0001 ± 0.0005).
   - Its gain comes from familiar-category positives (mean rank −0.12 ± 0.02 positions), **not** from novel ones (−0.00 ± 0.06).
6. **The three new feature groups aimed at MPs do not help (stacker, 3 seeds, paired).**

   | group | Δ | Thursday half |
   |---|---|---|
   | category concentration / streak | **−0.0009 ± 0.0003** | |
   | supply density / atypicality | −0.0004 ± 0.0003 | −0.0012 ± 0.0004 |
   | novelty / affinity surprise | −0.0002 ± 0.0002 | |

   **Recommendation:**
   - Do not chase the MP population with hand-made features.
   - The only actionable item from this lens is to bring `pre_v3_*` (ID-aware CF) into the stack or NN. Expect about +0.0005, and check it on test-like transductive conditions first (see 6.3).
   - The +0.03 exploration headroom needs a genuinely new signal about *which off-history category the user is about to open*. Nothing in the data we have provides it (see 7).

---

## 1. Set-up and sizes (`01_frame.py`)

**Model scores, val NDCG:**

| model | all | leak-free | 117 leak feeds |
|---|---|---|---|
| nn_final2_val_oof | 0.3863 | 0.3861 | 0.4143 |
| stk2_res2_r3 | 0.3857 | 0.3855 | 0.4181 |
| nn_final2_trainonly | 0.3825 | 0.3827 | 0.3539 |
| v5 blend | 0.3869 | 0.3867 | 0.4131 |

**Where the positives sit.** Number of leak-free positives in each decile of relative rank, top to bottom:

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| positives | 6178 | 3637 | 3017 | 2299 | 1893 | 1703 | 1141 | 859 | 663 | 369 |

**Groups:** MP 4,533; TP3 3,892; FA3 39,026.

## 2. Who the missed positives are

### 2.1 Category novelty (`04_segments.log`, plus the composition snippet in this report)

| class | share of candidates | share of positives | positive rate | MP rate of its positives | act/exp, v5 | act/exp, clean NN |
|---|---|---|---|---|---|---|
| category never in history | 0.159 | 0.066 | 0.66% | 0.583 | 0.951 ± 0.023 | 0.949 |
| category only at hpos ≥ 32 | 0.167 | 0.107 | 1.01% | 0.483 | 1.013 ± 0.020 | 1.018 |
| category at hpos 8-31 | 0.165 | 0.148 | 1.42% | 0.279 | 0.988 ± 0.015 | 0.996 |
| category in last 8 events | 0.509 | 0.678 | 2.10% | 0.127 | 1.006 ± 0.005 | 1.003 |

**Composition of each group** (never / hpos ≥ 32 / hpos 8-31 / last 8):

| group | never | hpos ≥ 32 | hpos 8-31 | last 8 |
|---|---|---|---|---|
| MP | 0.182 | 0.239 | 0.192 | 0.388 |
| TP3 | 0.027 | 0.031 | 0.080 | 0.862 |
| FA3 | 0.027 | 0.026 | 0.081 | 0.865 |

TP3 and FA3 are indistinguishable on this axis. 39% of MPs are still in a last-8 category; those are within-category misses (see 2.3).

**By maxsim32 quintile** (q0 = lowest similarity):

| quintile | share of positives | positive rate | MP rate | act/exp, v5 |
|---|---|---|---|---|
| q0 | 0.091 | 0.68% | 0.572 | 0.954 |
| q4 | 0.368 | 2.95% | 0.051 | 1.010 |

### 2.2 Feature families: MP vs TP3 vs FA3 (`03_screen.py`, `screen.parquet`)

**Column definitions:**
- **WR(MP > FA3)** = within-feed win rate of the MP over the FA3 negatives of the same feed, averaged over feeds.
- **t_best / t_clean** = cluster-robust t of the feature in a pairwise conditional logit over all (positive, negative) pairs, with the model's z-difference as offset. Base = v5 (t_best) or the train-only NN (t_clean). This is the "unexploited signal" test.

| feature | median MP | median TP3 | median FA3 | WR(MP > FA3) | WR(TP3 > FA3) | t_best | t_clean |
|---|---|---|---|---|---|---|---|
| maxsim32 (history similarity) | 0.703 | 0.921 | 0.926 | 0.105 ± 0.004 | 0.526 | +2.6 | +5.5 |
| pre_v3_fe (pretrained sequence encoder) | 13.7 | 17.8 | 17.8 | 0.118 ± 0.004 | 0.543 | **+10.3** | +15.1 |
| cos_u (user embedding) | 0.463 | 0.566 | 0.568 | 0.164 ± 0.005 | 0.528 | +3.4 | +6.0 |
| f1 (sub-category affinity) | 0.25 | 0.655 | 0.655 | 0.174 ± 0.005 | 0.529 | +1.4 | +1.1 |
| f3 (category affinity) | 0.306 | 0.695 | 0.683 | 0.248 ± 0.005 | 0.519 | +2.1 | +1.5 |
| cat_cnt_20 | 1 | 9 | 9 | 0.247 ± 0.005 | 0.521 | +1.5 | +1.2 |
| d_cat_last1 (same category as newest event) | 0 | 1 | 1 | 0.322 ± 0.004 | 0.517 | +0.5 | −0.3 |
| d_dup95_h (# history near-duplicates, cos ≥ 0.95) | 0 (mean 0.25) | 0 (3.46) | 0 (2.93) | 0.283 ± 0.003 | 0.513 | +1.6 | +4.2 |
| seq_cv_invd20 (co-visitation) | 0 (mean 0.005) | 0 (0.33) | 0 (0.32) | 0.302 ± 0.003 | 0.516 | +3.1 | +16.5 |
| feed_cat_share | 0.137 | 0.294 | 0.269 | 0.328 ± 0.006 | 0.511 | +2.8 | +1.1 |
| f5 (popularity-like) | 0.385 | 0.598 | 0.592 | 0.349 ± 0.006 | 0.509 | +0.4 | +0.4 |
| it_hist_et1_luo (item popularity in other users' histories) | 1 (mean 12.8) | 4 (29.0) | 4 (31.7) | 0.369 ± 0.006 | 0.501 | +2.9 | +11.1 |
| clu_ccos1024 (typicality: cosine to own cluster centroid) | 0.83 | 0.85 | 0.851 | 0.383 ± 0.006 | 0.491 | **−6.0** | −4.9 |
| trd_skde_s6h (supply density, ±3 h) | 2.6 | 3.35 | 3.43 | 0.407 ± 0.006 | 0.495 | **−5.2** | −4.4 |
| f4 (listing age) | 0.48 | 0.577 | 0.574 | 0.426 ± 0.006 | 0.496 | +0.5 | +1.1 |
| cat_min_hpos | 18 | 0 | 0 | 0.783 ± 0.005 | 0.478 | −0.2 | +0.5 |

- **Every positive-direction family loses heavily for MPs.** Similarity, affinity (f1/f3), co-visitation, popularity and near-duplicate re-contact all have win rates of 0.10-0.37.
- **The only family where MPs "win" is novelty itself** (cat_min_hpos, cluster recency), and that is the definition of the group.
- **Feed- and user-level features do not separate MP feeds** (`09_feedlevel.log`). AUC for "feed has an MP": H 0.532, number of history categories 0.547, M 0.527, user-embedding norm 0.493, type-1 share of the last 32 events 0.524, u_het1..4 0.49-0.51, user_id = 0 0.495, hour 0.495. The only exception is the number of positives (0.630), which is mechanical.

### 2.3 Category-level vs within-category misses (`09_feedlevel.log`)
- **31%** of MPs are category-level misses: no candidate of their category reaches the top half.
- **46%** have a candidate of their category in the top 10 (within-category misses).
- 82% of MPs have at least one same-category candidate ranked above them (median 4). The median MP category has 8 candidates in the feed; 10.5% are singletons.

### 2.4 Thirty concrete worst feeds (`05_worst.log`)
- Sample: feeds where every positive is in the bottom 40% and 30 ≤ M ≤ 300 (986 eligible), drawn at random with seed 0. 32 positives.
- Three patterns:
  - **(a) Exploration into an off-history category: 15/32 (47%).** Examples: feeds 40, 540, 5591, 8516, 8737, 11307, 13778.
    - The positive's category is absent from the last 32 events, or from the whole history in 5/32.
    - Often f1 and f3 are at their floor (0.018 / 0.016). So **the platform's own affinity model also knows nothing** about the user's interest.
    - The user's history is dominated by 1-2 categories (126/256 in category 13, 187/256 in 13, 252/256 in 13, …), and the model fills the top with those.
  - **(b) Low-similarity item inside the dominant category: 7/32 (22%).** Examples: 2372, 3709, 7112, 7682, 8918, 10409, 10892.
    - Same category, but maxsim32 0.46-0.86 against 0.88-0.99 for the top negatives.
    - Several are fresh listings (f4 ≤ 0.2) with f5 near 0.
    - **Checked, and they are not under-ranked as a class**: f5 < 0.03 in a familiar category has act/exp 1.004 (z +0.13); with also f4 < 0.2 it is 0.988 (`14_fresh_check.log`).
  - **(c) Re-visit of a secondary category from recent history: 10/32 (31%).** Examples: 1086, 3575, 4037, 8849, 12248, 13393.
    - The category appeared 2-11 events ago but is a minority. Its f3 is mid-range, and the dominant category's items take the top.
- **Popularity "in the test period" does not explain the misses.** Several exploration positives have high LUO popularity (315, 513, 30), but so do many top negatives (224, 261, 288, 297, 373). On BOTHNOV pairs (defined in 3.1) `it_hist_all_luo` has no residual (t < 1.5).

## 3. Signals the model does not exploit

### 3.1 Pairwise residual by pair type (`07_subset_screen.log`)

**Pair types:**
- **BOTHNOV**: both candidates in novel categories (172K pairs). Model pairwise accuracy 0.687.
  - The only residuals are weak: typicality `clu_ccos1024` t −4.2, supply density `trd_skde_s6h` −4.0, `pre_v3_fe` +3.8.
  - f3 +1.6, f5 +1.2, popularity < 1.5.
  - **Nothing orders novel items much better than the model already does.**
- **SAMECAT**: same category. `pre_v3_fe` +9.5, `pre_v3_new` +9.0, `clu_ccos1024` −5.7, `pre_v3_hmax32` +5.2, `ctx_hs_k256` +4.8, `trd_skde_s6h` −4.2.
- **CROSS**: different categories. Large residuals on category-level features: feed_cat_share +23, f3 +19, cat_share_all +18, d_cat_pur20 +18, d_cat_run +15, `pre_v3_fe` +14.
- POSNOV (pairs whose positive is novel) is selection-biased by construction and is ignored.

### 3.2 Calibration screen (`06_calib.log`)
Deciles of each feature, actual / expected under the v5 softmax, cluster-robust z. With about 5,000 bins the null max |z| is about 4.

**Under-predicted bins:**

| bin | act/exp | z |
|---|---|---|
| all 20 newest events are in the candidate's category (`trd_tp_mx20` NaN bin, 7.9% of candidates) | 1.06 | +7.1 |
| f3 ≥ 0.887 | 1.05 | +6.2 |
| feed_cat_share ≥ 0.547 | 1.04 | +5.2 |
| cat_cnt_20 ≥ 17 | 1.05 | +5.1 |
| top decile of `pre_v3_fe_z` | 1.08 | +5.9 |

**Over-predicted bins:**

| bin | act/exp | z |
|---|---|---|
| lowest decile of `ctx_hs_k256` | 0.75 | −6.9 |
| lowest decile of `pre_v3_fe` | 0.85 | −5.7 |

**This is the opposite of what an MP fix would need.** The residual says the model is slightly *too exploratory*: it should concentrate more on the dominant category and the most similar items.

### 3.3 These residuals barely move NDCG (`10_blendcheck.log`)
- Method: nested 5-fold choice of λ in z(v5) + λ·z_feed(x), leak-free.

| signal | nested Δ |
|---|---|
| pre_v3_fe | +0.0006 ± 0.0004 (λ = 0.1 in 4/5 folds) |
| pre_v3_new | +0.0004 ± 0.0003 |
| d_cat_pur20, d_cat_run, d_top_cat20, feed_cat_share | −0.0001 ± 0.0002 each |
| f3 | +0.0002 ± 0.0003 (and the sign is *negative*) |
| clu_ccos1024 | −0.0004 ± 0.0002 |
| trd_skde_s6h | 0 |
| d_f3_resid | −0.0002 |
| d_dup95_h | −0.0001 |
| it_hist_all_luo | −0.0003 |

- Lesson: pairwise t-statistics over all pairs, and decile calibration, overstate what reaches the head of the list.

## 4. Exploration-intent ceiling (`11_explore_oracle.log`, `12_hist_explore.log`)

| scenario | Δ vs v5 (leak-free 0.3867) |
|---|---|
| oracle feed type, novel candidates +1 z in exploration feeds | +0.0155 |
| same, +3 z (saturates) | **+0.0315** |
| oracle both ways (lift in exploration feeds, push down elsewhere, ±3) | +0.0350 |
| realistic: + c · logit(p_explore) · novel, p_explore out-of-fold from feed-level features, c nested | **−0.0005 ± 0.00015** |

- **The user's own past exploration rate** = share of type-1 events whose category is absent from the 32 older events; mean 0.088.
  - Alone it predicts exploration feeds with AUC 0.576.
  - On top of the novel-candidate share it moves the CV AUC only from 0.706 to 0.713.
- **There is no user-level exploration propensity worth modelling.** Exploration is a per-session event, invisible in the history.

## 5. Quick tests of new features (`08_quicktest.py`)

**Protocol:**
- Residual XGB stacker in the `stack2.py` style. The base margin is z(`nn_final2_trainonly`), so no val labels are in the base.
- Compact baseline features. Leaky ALS is excluded. Files deleted by other agents during this session are also out: `seq_cv_fwd5/seqs/t1/w5/invd20_z`, and in the CPU runs `ctx_nd_c80/c90/c95/dupb_*/mean`. That leaves 30 features.
- 5-fold GroupKFold by user, 3 seeds, the same folds for every config.
- Deltas are per-feed and paired, averaged over seeds, on leak-free feeds. "Blend" = 0.5 z(`nn_final2_val_oof`) + 0.5 z(stacker).
- Run on CPU (16 threads), because GPU N was saturated by 8 other processes, about 10× slower.

| group (added to baseline) | features | stacker NDCG | Δ stacker | Δ on Thursday half | Δ inside blend | per-seed stacker |
|---|---|---|---|---|---|---|
| baseline | 4 model scores (z, rank) + 30 features | 0.38503 | — | — | — (0.38636) | 0.38509 / 0.38458 / 0.38543 |
| **G1** category concentration / streak (new: run length of the candidate's category from the newest event, also type-1-only, purity of the last 20 / last 5, modal-category flag, same as newest event, count in the last 3; plus feed_cat_share, n same-category candidates, cat_share_all, cat_min_hpos, cat_cnt_all) | 12 | 0.38415 | **−0.00088 ± 0.00031** | −0.00054 ± 0.00044 | −0.00039 ± 0.00022 | 0.38428 / 0.38394 / 0.38424 |
| **G2** supply density / atypicality (clu_ccos1024, trd_skde_s6h, trd_kde_sday, new: # history near-duplicates at cos ≥ 0.95 / 0.90, hpos of the nearest near-duplicate) | 6 | 0.38463 | −0.00041 ± 0.00026 | **−0.00120 ± 0.00036** | −0.00023 ± 0.00019 | 0.38497 / 0.38425 / 0.38467 |
| **G3** novelty / affinity surprise (new: d_f3_resid = f3 minus the split mean in the same (cat_cnt_20 bucket, category) cell; plus clu_hmin1024, f3_rk, f1_rk) | 4 | 0.38489 | −0.00015 ± 0.00023 | −0.00030 ± 0.00032 | +0.00000 ± 0.00016 | 0.38500 / 0.38464 / 0.38502 |
| **G4** pretrained ID-aware sequence encoder (pre_v3_fe, _new, _fe_noid, _hmax32, _id_contrib; existing features, not in v5) | 5 | 0.38574 | **+0.00070 ± 0.00034** | +0.00007 ± 0.00048 | **+0.00061 ± 0.00023** | 0.38562 / 0.38617 / 0.38543 |

- G4 passes 2 SE overall (stacker 2.1, blend 2.7) but **is flat on the later day**.
- Where G4's gain comes from (`13_g4_where.log`):
  - Positives in familiar categories move up 0.122 ± 0.023 positions; positives in novel categories move −0.002 ± 0.061.
  - MP-containing feeds get +0.0008 ± 0.0003, in proportion to their share.
  - The MP count only drops from 4,570 to 4,522.

## 6. Interpretation and hypotheses

1. **Missed positives are mostly rational misses.** About 42% are first-time or long-dormant category opens, and many have floor-level f1/f3. Another ~20% are low-similarity items inside a familiar category. The model's probability for each class is calibrated within 1-5%, so better weighting of existing features cannot recover them. This is why the earlier category-intent auxiliary head was neutral: the intent is not in the inputs.
2. **The +0.03 exploration headroom is real but locked.** Unlocking it needs information on *which new category this session is about*. The history (own or population-level transition priors: the `trd_tp_*` family was tested and deleted), the user embedding and f1/f3 do not carry it. Sources that might:
   - query or search context;
   - the feed's own candidate set as a retrieval signal. Exploration candidate sets are uniform, so a skewed category share in the candidate set says little; already tested via feed_cat_share.
   - nothing else visible in this dataset.
3. **The actionable residual points the other way: exploit harder.** The dominant category, the highest f3 and the most history-similar items are under-predicted by 4-8%. But neither linear boosting nor stacker features convert this into NDCG: G1 −0.0009, linear Δ ≈ 0. The NN probably already sits at the head-of-list optimum, and the residual is mid-list mass.
4. **ID-level collaborative signal from the pretrained encoder is the only family that helps (+0.0006-0.0007).** It helps within-category ordering, which is also where 46% of MPs live (category in the top 10).
   - Caveat: `pre_v3` is transductive on other users' later snapshots. Val feeds see up to about 7 days of them, test at most about 3 days (`reports/pretrain_report.md`), and the gain is flat on Thursday.
   - Re-verify with the `v3A` check model (no test snapshots) before adopting it.

## 7. Recommendations
- **Do not spend more time on hand-made novelty, streak or supply features for MPs.** All three targeted groups are null or negative, and one is negative on the later day.
- **Feed `pre_v3_fe`/`pre_v3_new` (ID part) into the residual stacker or the NN fine-tune branch.**
  - Expected gain about +0.0005 val.
  - Accept it only if the `v3A` (no-test-snapshot) variant keeps the gain and the Thursday half is not negative.
- **For the exploration headroom (+0.03 oracle),** the only plausible path is a model that learns P(category opened next | history) *jointly across users from train labels*. That means a category-level next-open head trained on the 3M train feeds, scoring the candidate's category rather than the item.
  - That is the strategist's category-intent idea, and it was neutral inside the NN.
  - The data here suggests why: the open is not predictable from the history (user exploration-rate AUC 0.58; feed-level AUC ≤ 0.55).
  - Treat further attempts as low prior.

## 8. Caveats
- During this session other agents deleted about 320 feature files from `data/feats/{val,test}`, including `clu_ccos1024`, `trd_tp_*`, `ctx_fs_*` and `ctx_nd_c80/c90/c95/dupb_*/mean`.
  - The screens (sections 2-3) ran before the deletion, on 576 features.
  - The quick-test baseline therefore has 30 instead of 37 features; it is paired within the run, so the deltas are valid.
  - `clu_ccos1024` was rebuilt from `data/clu_work/item_ccos_1024_*.npy` into `eda/errors/missed_positives/derived/`.
  - The other inputs used were snapshotted into `eda/errors/missed_positives/derived/{val,test}/`, together with every new feature, built identically for val and test from the feed's own data.
- All selection was coarse: 4 pre-declared groups, 3 seeds. No fine-tuning on val.
