# Error analysis: SYNTHESIS (lead)

Inputs: the four lens reports `err_segments.md`, `err_category_intent.md`, `err_missed_positives.md`, `err_feed_structure.md`, plus new quick tests.
Scripts and logs: `<repo>/eda/errors/synthesis/` (`logs/*.log`, features in `derived/{val,test}/`).

Protocol for all new numbers:
- Val only. Leak mask applied: the 117 feeds whose non-zero user has a later feed in val or test are dropped, leaving 14,306 feeds, 21,759 positives and 2,428,822 within-feed (positive, negative) pairs. These counts match the lenses.
- Every comparison is a paired per-feed delta ± SE.
- Stacker tests use the `stk2_res4_clean` recipe:
  - base margin = affine map of z(`nn_final3_trainonly`);
  - inputs = `nn_final3_trainonly` and `xgb_x300_luo` scores plus the 35 res4 features that still exist;
  - XGB binary, eta 0.02, depth 4, 300 rounds, 5-fold GroupKFold by user, at least 2 seeds with identical folds across configs.
- Single-feature tests use a nested linear blend z(base) + w·z(x): w comes from a 6-value grid chosen on 4/5 of the feeds and is applied to the other 1/5, over 2 seeds.
- "Test-mix" = val reweighted to the test (M bucket × cold-share bucket) mix, using the segments lens code.
- **Base snapshot:** `nn_final3*` was rewritten by the NN agent at 06:59 while this ran. Every number here uses the 06:59 files, copied to `eda/errors/synthesis/snap/` (md5 val trainonly `b3637771…`, oof `2ec3f279…`). The first round of tests ran on the 06:38 version; it was redone and the conclusions did not change (`out_old/`).
- Compute: CPU only, 16 threads per process, up to 8 processes at once. No GPU was needed. faiss ran on CPU. some GPUs were not touched (CURRENT_STATUS reserves them for others), and GPU N was 97% busy with other jobs.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **The base moved under the lenses, and the val-label stage changed character.** All four lenses measured `nn_final2` (train-only 0.3827, OOF 0.3861). On leak-free feeds, `nn_final3_trainonly` = **0.38662** and `nn_final3` OOF = **0.38797** (+0.00135 ± 0.00052). The main train-stage change is the new co-visitation and LUO popularity inputs; nn_final3 also adds DIN/GRU members. Together they shifted where the errors are:
   - nn_final3_tr vs nn_final2_tr: **+0.0039 ± 0.0010** overall. Warm-positive feeds gain **+0.0111 ± 0.0012**; all-cold-positive feeds (17.4% of val, 22.1% at test mix) lose **−0.0206 ± 0.0022**. Cold-positive vs warm-negative pair AUC falls from 0.664 to 0.641 (same category: 0.601 to 0.566).
   - The val fine-tune now gains only on cold-positive feeds: **+0.0050 ± 0.0012** there vs +0.0001 on warm-positive feeds. That is the reverse of the segments-lens finding on nn_final2, where it was −0.016 on cold and +0.009 on warm. So S2's premise ("the val stage over-trusts ID signals") no longer holds.
   - The residual stacker no longer adds anything: −0.0001 ± 0.0006 vs its own base. No equal-weight blend beats nn_final3 OOF alone (0.38797). The v5-style z-mean (OOF + stacker) gives 0.38755 (−0.0004 ± 0.0004), and OOF + train-only gives 0.38731 (−0.0007 ± 0.0004).
2. **The lenses agree on substance.** Category intent is saturated, set/feed structure is absorbed, missed positives are mostly irreducible exploration, and segment routing gives nothing. The only open axis is item discrimination inside a category. Five claims needed correction (§1): co-visitation already exists; the NDCG loss weight was already run; the two "within-category oracle" figures use different definitions; the "70% cross-category error" figure reflects pair counts; and the cold/warm sign has flipped.
3. **No new signal passed the 2-SE rule.** Three untested hypotheses with the highest expected gain were tested with new features, plus re-checks of four "existing but unused" families on the new base and a cold-offset test:
   - T1: item taste profiles learned from train labels ("did users like you click this item in train?");
   - T2: content-kNN train COEC for cold listings (S2a);
   - T3: listing-age proxy for separating near-duplicates (FS-H3).
   - Every stacker and blend delta lies between −0.0004 and +0.0006 with |t| ≤ 1.8, and the later-half (Thursday) stacker delta is ≤ 0 for every group.
   - Only one single-feature nested blend clears 2 SE: tp_cosC_h, +0.0009 ± 0.0004 on nn_final3_tr. It is the best of 11 screened features and drops to −0.00003 on the fine-tuned base.
   - Details in §3.
4. **What to do:**
   - (a) Stop the val-stacker line on nn_final3 and do not add any of the tested families.
   - (b) Submit nn_final3 OOF without a stacker; any extra component must be pre-declared and equal-weight.
   - (c) The one remaining lead is an observation that has not been tested: nn_final3's +0.004 came from trading cold-positive feeds for warm-positive feeds, and the fine-tune wins back +0.005 on the cold-positive side. Find which fine-tune-branch inputs do that and move a train-stage analogue of them into the train stage (§5). Expected 0 to +0.002, larger on private test, which has more cold listings.
   - Getting +0.008 from the information available now looks unlikely. The four lenses and this synthesis tested 20+ signal families, and every one that was not already inside the NN came in below +0.001.

---------------------------------------------------------------------------------------------------------------------

## 1. Cross-check of lens findings

| # | Claim (lens) | Check | Verdict |
|---|---|---|---|
| C1 | Within-category headroom exceeds cross-category headroom: segments within-oracle 0.6169 vs category oracle 0.5576; category_intent 0.679 vs 0.551. | The two within-oracles are defined differently. Segments sorts categories as blocks by their best NN score, then puts positives first. Category_intent reassigns NN score values inside each (feed, category) cell, which keeps the interleaving. Both rank within > cross. | Consistent; the numbers differ only by definition. |
| C2 | "70% of within-feed pairwise error mass is cross-category" (strategist) vs category_intent: "mostly reflects pair counts". | 73.5% of all pos-neg pairs are cross-category (same-category share 0.265). The error rate per pair is higher inside a category (0.334 vs 0.279, category_intent). On nn_final3_tr, same-category AUC is below all-pairs AUC for every pair type (§3.1). | Category_intent is right. |
| C3 | Segments S1: "No item-level co-visitation feature exists in data/feats". | Wrong. `seq_cv_{invd20,norm,pop,invd20_rk}` are OTTO-style item co-visitation features (feats_v2/seq_). They are in stk2_res4_clean, and since v3/v4 they are in the NN **train stage** (+0.0032 train-only, nn_report). Their residual on nn_final3_tr: nested blend +0.0002 ± 0.0005, pairwise residual t +2.2 (it was t +10.7 and nested +0.0016 ± 0.0006 against nn_final2_tr). | S1 reduces to co-visitation variants, which the feature workflow already found "not needed". |
| C4 | Segments: val-label models win on warm, lose on cold (NN oof −0.0159 on all-cold-positive feeds). Missed_positives: fresh listings are calibrated. | Both are true for nn_final2. For nn_final3 the sign flips: the fine-tune goes from −0.0159 (nn_final2) to **+0.0050 ± 0.0012** on cold-positive feeds, warm +0.0001 ± 0.0007. The train stage now carries the warm/cold trade (−0.0206 cold, +0.0111 warm vs nn_final2_tr). nn_final3_tr stays calibrated on cold items: actual/expected positives 1.006 ± 0.014 (cold) vs 0.998 (warm); nn_final2_tr was 0.968. | S2's motivation is obsolete for the val stage; the problem has moved to the train stage. |
| C5 | Feed_structure H1: NDCG-consistent loss weight N/IDCG(N) is untested. | The NN agent already ran it as K1(c) "equal × K/IDCG(K)". v2_idcg 0.38004 vs v2_feed 0.38003 val (holdout +0.0007). Fine-tune A/B within ±0.0002. v4_cv_s33 (idcg) 0.38474 vs v4_cv 0.38469 train-only. | Already tested; neutral. |
| C6 | Missed_positives H1: pre_v3 encoder scores +0.0007 ± 0.0003 (nn_final2 base). | On nn_final3 with the clean v3A model: stacker +0.00038 ± 0.00030, blend +0.00000 ± 0.00020, Thursday half −0.00014. | Absorbed; not reproduced. |
| C7 | Category_intent: catint_S1 ≤ +0.0005. | +0.00023 ± 0.00027 (stacker), +0.00023 ± 0.00019 (blend). | Consistent; not significant. |
| C8 | Segments S3: global cross-user category-transition statistics are "the only untested source". | Category_intent's GBDT already had per-category history statistics, and a 64-event sequence model reached at most the NN (−0.0008 ± 0.0009 category NDCG). Missed_positives reports population transition priors (`trd_tp_*`) and the NN intent head as neutral. | Tested conclusively; drop. |
| C9 | f4 = listing age (segments); f4 at chance on near-duplicates (feed_structure). | My first-appearance age proxy has Spearman 0.63 with f4. f4's anti-informativeness on cold+/warm− pairs (AUC 0.217 here) is selection by coldness, not a ranking signal. | Consistent. |
| C10 | Coldness numbers. | All-cold-positive feeds 0.174 (test-mix 0.221). Cold candidate share 0.289 val / 0.393 test (never in train candidates or histories). Same as segments. | Reproduced. |

---------------------------------------------------------------------------------------------------------------------

## 2. All lens hypotheses ranked by expected gain × confidence

E[gain] = midpoint of the lens's expected gain × prior confidence, before this synthesis's tests.

| Rank | Hypothesis (source) | Lens gain | Prior conf. | E[gain] | Status before | Synthesis result |
|---|---|---|---|---|---|---|
| 1 | ID-level, personalised item signal inside a category: train-label evidence of which users click this item (CI-H1, FS-H3, S1 "warm-positive cell CB +0.0076") | +0.002-0.005 | 0.20 | +0.0007 | untested | **T1 taste profiles: not significant** (§3.2) |
| 2 | Content-kNN quality prior for cold listings (S2a) | +0.001-0.002 | 0.20 | +0.0003 | untested | **T2: fails the gate (cold+/warm− same-category AUC 0.50-0.53, nested ≤ 0)** (§3.3) |
| 3 | pre_v3 ID-aware encoder scores (MP-H1) | +0.0003-0.0007 | 0.50 | +0.00025 | tested on nn_final2 only | re-checked: absorbed (C6) |
| 4 | ctx_hs history-similarity family (FS-H2) | +0.0003-0.001 | 0.40 | +0.00025 | untested in stacker | −0.00028 ± 0.00038 |
| 5 | Listing freshness for near-duplicate disambiguation (FS-H3) | 0-0.003 | 0.15 | +0.0002 | untested | **T3: null** (§3.4) |
| 6 | Cold offset / ID-dropout in the val stage (S2b) | (part of S2) | 0.20 | +0.0002 | offset −0.0002 on nn_final2 | offset null on nn_final3 (§3.5); val-stage ID-dropout obsolete (C4); train-stage analogue → §5 |
| 7 | Co-visitation variants (S1) | +0.001-0.003 | 0.10 | +0.0002 | covis already in NN | residual nested +0.0002 ± 0.0005 → drop |
| 8 | catint_S1 in stacker (CI-H2) | ≤ +0.0005 | 0.50 | +0.0001 | untested | +0.00023 ± 0.00027 → drop |
| 9 | NDCG-consistent loss weight (FS-H1) | 0-0.0005 | 0.30 | +0.0001 | already run (C5) | neutral → drop |
| 10 | Category-switch / transition / next-open-category priors (S3, CI-H3, MP-H2) | 0-0.002 | 0.05 | < +0.0001 | tested (C8) | drop |
| 11 | Set-level re-ranking; concentration, streak, supply, novelty features (FS-H4, MP-H3) | 0 | — | 0 | tested negative by the lenses | drop |

The three hypotheses with the highest E[gain] that no lens had tested (ranks 1, 2 and 5) got new features (T1-T3). Ranks 3, 4, 6 and 8 were cheap, so they were re-run on the new base.

---------------------------------------------------------------------------------------------------------------------

## 3. Quick tests

### 3.1 Reference points (leak-free val, `logs/base_scores.log`, `logs/04_coldmix.log`, `logs/09_blends.log`)

| Model / blend (equal-weight per-feed z) | val | test-mix | vs nn_final3_tr | vs nn_final3 OOF |
|---|---|---|---|---|
| nn_final2_trainonly | 0.38269 | 0.38377 | −0.0039 ± 0.0010 | |
| nn_final2 OOF | 0.38612 | 0.38635 | | |
| **nn_final3_trainonly** | **0.38662** | 0.38710 | — | −0.00135 ± 0.00052 |
| **nn_final3 OOF** | **0.38797** | 0.38865 | +0.00135 ± 0.00052 | — |
| res4-recipe stacker on nn_final3_tr (2 seeds) | 0.38650 | 0.38781 | −0.00012 ± 0.00061 | |
| OOF + stacker (v5-style) | 0.38755 | 0.38820 | +0.00092 ± 0.00053 | −0.00043 ± 0.00044 |
| OOF + train-only | 0.38731 | 0.38787 | +0.00069 ± 0.00035 | −0.00066 ± 0.00035 |
| OOF + train-only + nn_final2 OOF | 0.38777 | 0.38846 | +0.00115 ± 0.00057 | −0.00020 ± 0.00041 |

Segment split of the deltas:

| Delta | val | test-mix | all-cold-positive feeds (17.4%) | warm-positive feeds (71.0%) |
|---|---|---|---|---|
| nn_final3_tr − nn_final2_tr | +0.00393 ± 0.00096 | +0.00333 ± 0.00138 | **−0.02064 ± 0.00218** | **+0.01107 ± 0.00120** |
| nn_final3 OOF − nn_final3_tr | +0.00135 ± 0.00052 | +0.00155 ± 0.00073 | **+0.00503 ± 0.00120** | +0.00007 ± 0.00065 |
| nn_final2 OOF − nn_final2_tr | +0.00342 ± 0.00099 | +0.00258 ± 0.00133 | −0.01594 ± 0.00199 | +0.00877 ± 0.00126 |
| stacker − nn_final3_tr | −0.00012 ± 0.00061 | +0.00071 ± 0.00087 | −0.00052 ± 0.00132 | −0.00000 ± 0.00077 |

Pairwise AUC by pair type (`logs/07_coldpairs.log`; cold = item never in train candidates or histories):

| Pair type (share of pairs) | nn_final2_tr | nn_final3_tr | nn_final3 OOF |
|---|---|---|---|
| cold+ vs warm− (16.1%) | 0.6636 (same-cat 0.6014) | **0.6414 (0.5664)** | 0.6562 (0.5880) |
| warm+ vs cold− (21.8%) | 0.7921 (0.7507) | **0.8143 (0.7849)** | 0.8082 (0.7744) |
| warm+ vs warm− (54.7%) | 0.6903 (0.6437) | 0.6985 (0.6557) | 0.7010 (0.6581) |
| cold+ vs cold− (7.3%) | 0.7547 (0.6973) | 0.7580 (0.7012) | 0.7594 (0.7034) |

The new train stage, most likely through its co-visitation/LUO inputs, acts like a warm-item prior. That pays off whenever the positive is warm and costs whenever it is cold. The model stays calibrated on cold items in aggregate (`logs/05_coldflag.log`: actual/expected positives 1.006 ± 0.014 cold vs 0.998 warm; seq_cv_pop = 0: 0.997), so the cost comes from missing information, not bias.

### 3.2 T1: item taste profiles from train labels (new information; `01_taste.py`)

Every val/test candidate item with train impressions gets four aggregates over those impressions:
- S_j = Σ v_f over all impressions; C_j = Σ v_f over clicks; E_j = Σ ctr(pos_f)·v_f;
- here v is either the unit user embedding (u) or the recency-weighted mean of the last 20 history embeddings (h, half-life 5).

Features for feed g:
- cos(v_g, C_j): similarity to the users who clicked j;
- cos(v_g, S_j): similarity to the users j was shown to;
- the difference between the two;
- a position-debiased click surplus, plus the counts n_j, c_j and the COEC.

Train ends 2 days before val starts, so nothing is from the future. The features are computed identically for val and test. Coverage: impressions for 64.4% of val candidates (53.8% test); a train click for 19.2% (15.3% test).

| Test | Result |
|---|---|
| Single-feature screen vs nn_final3_tr (`logs/02_screen_v2.log`) | Best is tp_cosC_h: same-category AUC 0.567 on covered pairs, residual t +2.1, nested +0.00093 ± 0.00038. tp_cosC_u: t +3.8, nested +0.00015 ± 0.00032. tp_c: t +3.5, nested +0.00012 ± 0.00037. tp_cosC_h is the best of 11 screened features, so a result this good is roughly what chance alone would produce. |
| Same features vs nn_final3 OOF (`logs/02_screen_oofbase.log`) | Every nested blend ≤ 0: tp_cosC_h −0.00003 ± 0.00027, tp_cosC_u −0.00015 ± 0.00017, tp_c −0.00029 ± 0.00025 (residual t +1.3 to +2.9). The +0.0009 against the train-only base does not survive on the fine-tuned base. |
| Calibration of the t-scale | Features already inside the NN show the same residual size against nn_final3_tr: seq_cv t +2.2, it_tr_coec +3.3, f4 −5.2. Against nn_final2_tr, seq_cv was t +10.7. A residual t of 3-4 is the floor for nonlinear miscalibration here and does not signal new information. |
| Stacker, all 11 features (2 seeds) | +0.00017 ± 0.00035; Thursday half −0.00032; blend +0.00006 ± 0.00024 |
| Stacker, u-features only (N_taste_u, 2 seeds) | +0.00057 ± 0.00035 (t 1.6); Thursday half −0.00010; blend +0.00030 ± 0.00024 |
| 4-seed confirmation (pre-declared: N_taste_u and the 5 cosine/count features) | N_taste_u: stacker **+0.00044 ± 0.00029** (t 1.5; per seed +0.00104 / +0.00009 / +0.00045 / +0.00017), Thursday half −0.00001 ± 0.00038, blend +0.00020 ± 0.00020. Cosine/count set: stacker +0.00009 ± 0.00023, Thursday half −0.00036, blend +0.00028 ± 0.00016 (t 1.8; Thursday half −0.00012). |

Verdict: **not confirmed.** The best version is +0.0004 ± 0.0003 in the stacker and +0.0002-0.0003 in the blend, and it is 0 or negative on the Thursday half and on the fine-tuned base. Part of T1's signal was absorbed when co-visitation/LUO entered the train stage: tp_c residual t fell from +8.2 against nn_final2_tr to +3.5 against nn_final3_tr. A train-stage version (day-causal aggregates) would probably add at most +0.0005; low priority.

### 3.3 T2: content-kNN train COEC (S2a; `08_knn.py`)

Database: 16.1M train items with at least 3 impressions. Index: faiss IVF16384 (inner product, nprobe 24), k = 50 neighbours, restricted to the same category, with the item itself excluded. Features: COEC over the 50 neighbours, COEC over neighbours with cos ≥ 0.90 and ≥ 0.95, count of cos ≥ 0.95 neighbours, best cosine, and a similarity-weighted neighbour COEC.

Coverage: 98% of val candidates have a same-category neighbour with cos ≥ 0.90, and 83% one with cos ≥ 0.95 (mean 21 of 50 neighbours at cos ≥ 0.95). The embedding space is dense with look-alike listings.

| Feature | AUC all pairs | same-category | cold+ vs warm− same-category (S2 gate > 0.62) | residual t vs nn_final3_tr | nested blend |
|---|---|---|---|---|---|
| kn_coec50 | 0.5440 | 0.5170 | 0.5044 | +3.1 | −0.00021 ± 0.00028 |
| kn_coec_s90 | 0.5418 | 0.5177 | 0.5051 | +3.1 | −0.00035 ± 0.00024 |
| kn_coec_s95 | 0.5316 | 0.5151 | 0.5125 | +2.1 | 0.00000 (w = 0) |
| kn_ctr_w | 0.5408 | 0.5197 | 0.5052 | +3.0 | −0.00013 ± 0.00009 |
| kn_n95 | 0.5101 | 0.5158 | 0.5313 | +1.2 | −0.00025 ± 0.00017 |
| kn_sim1 | 0.5019 | 0.5117 | 0.4866 | +0.2 | 0.00000 |
| Stacker, all 6 (4 seeds) | | | | | −0.00008 ± 0.00025 (Thursday half −0.00026); blend +0.00004 ± 0.00017 |

Verdict: **fails the S2 gate.** Same-category AUC on cold+/warm− pairs is 0.50-0.53, against the 0.62 required. Neighbour click rates carry about as much information as the 1024-cluster CTR (within-category AUC 0.51, segments lens), which agrees with the embeddings EDA (kNN R² of the item target rate < 0). The listings around a cold item say almost nothing about whether this user clicks it.

### 3.4 T3: listing-age proxy (FS-H3; `03_age.py`)

first_seen(j) = earliest timestamp of any feed, in any split and from any user, where j appears as a candidate or in a history. Age = ts_g − first_seen, capped at 14 days; it is causal because the item is in feed g itself. Also tested: the within-feed age rank, a first-appearance-in-history flag, and the item_id rank.
- Age has Spearman 0.63 with f4, and item_id carries no age information (Spearman 0.001).
- Screen: age AUC 0.534 but residual t −4.6 (fresher favoured after the NN); nested blend +0.00000. ag_rank nested +0.00052 ± 0.00041.
- Stacker: +0.00001 ± 0.00027 (Thursday half −0.00050); blend +0.00020 ± 0.00019.

Verdict: no new information; f4 already carries listing age.

### 3.5 Cold offset on the new base (`06_coldoffset.py`)

score = z(base) + δ·flag, with δ ∈ {−0.25 … 1.0} chosen nested.

| Flag | nn_final3_tr | nn_final3 OOF | OOF + train-only blend |
|---|---|---|---|
| cold_item | −0.00046 ± 0.00017 (test-mix −0.00079) | −0.00034 ± 0.00017 | +0.00008 ± 0.00027 |
| seq_cv_pop = 0 | 0.00000 (δ = 0 always chosen) | −0.00028 ± 0.00025 | −0.00022 ± 0.00025 |

A boost recovers +0.008 to +0.018 on cold-positive feeds but loses more on warm-positive feeds. Verdict: drop. This confirms the calibration result in §3.1.

### 3.6 Existing-but-unused families on the new base (stacker, 2 seeds; `logs/compare_v2.log`)

| Group | Stacker Δ | Thursday half | v5-style blend Δ |
|---|---|---|---|
| pre_v3A (fe, new, fe_noid, hmax32, idmax32) | +0.00038 ± 0.00030 | −0.00014 ± 0.00044 | +0.00000 ± 0.00020 |
| ctx_hs (9 features incl. seq_top3) | −0.00028 ± 0.00038 | −0.00035 ± 0.00054 | +0.00004 ± 0.00025 |
| catint_S1 | +0.00023 ± 0.00027 | −0.00012 ± 0.00038 | +0.00023 ± 0.00019 |
| it_tr_* (item train CTR/COEC/counts) | +0.00013 ± 0.00031 | −0.00019 ± 0.00042 | +0.00013 ± 0.00021 |

---------------------------------------------------------------------------------------------------------------------

## 4. Confirmed new signals

**None.** No feature family raised paired val NDCG by more than 2 SE on the nn_final3 base. Every stacker group has a Thursday-half delta ≤ 0. Nothing from this synthesis should go into the NN train stage, the fine-tune branch or the residual stacker as a confirmed gain.

The nearest miss is T1 (taste profiles), at +0.0004 ± 0.0003 over 4 seeds with a flat Thursday half. If the NN agent has spare GPU time after R2, a train-stage version would be: for train rows of day d, aggregate S/C/E over train days < d; same features and vector types as `01_taste.py`; add the 7 u-features to `--xt`. This is not recommended for the stacker.

---------------------------------------------------------------------------------------------------------------------

## 5. Recommendations and implementation recipes

**R1. Submission composition (immediate, no new training).**
- Use **nn_final3 (fine-tuned OOF / test) as the main component**. Drop the residual stacker: on nn_final3 it is −0.0001 ± 0.0006 alone and lowers every blend.
- If the NN agent wants diversity, add one pre-declared, equal-weight component. The only one that comes close is nn_final2 OOF: 3-way z-mean 0.38777 (−0.0002 ± 0.0004 vs OOF alone). Do not tune weights on val.
- The OOF − train-only gain sits in cold-positive feeds (+0.0050 ± 0.0012). Test has more cold listings (cold share 0.393 vs 0.289), and the test-mix estimate is +0.00155 ± 0.00073, so the gain should carry over.

**R2. The one open lead: what the fine-tune learns about cold listings, moved into the train stage (untested hypothesis; NN agent).**
- Evidence:
  - Relative to nn_final2_tr, nn_final3's train stage (main change: the co-visitation/LUO inputs) costs −0.0206 ± 0.0022 on the 17.4% of val feeds whose positives are all cold. That is 22.1% of feeds under the test mix, where it is worth about −0.0045 of overall NDCG.
  - The fine-tune wins back +0.0050 ± 0.0012 there and nothing on warm feeds. Its branch (`n2clean_als`, see `nn/feat_cache/val/`) contains ALS (`als_*`), near-duplicate/history-neighbour context (`ctx_nd_*`, `ctx_hn_*`, `ctx_hs_*`, `centrality`), train-period item/user statistics (`it_tr_*`, `it_cand_cnt_train`, `u_tr_*`), LUO popularity and co-visitation (`it_hist_*_luo`, `seq_cv_*`), sequence similarity (`seq_*`) and `hour`. It has no in-split counts.
  - On cold+ vs warm− pairs, the history- and user-similarity features beat nn_final3_tr as single scores (`logs/10_coldpair_screen.log`): AUC cos_u 0.656, f1 0.656, ctx_hn_rrw 0.653, maxsim32 0.646, ctx_nd_knn_h 0.642, vs nn_final3_tr 0.641 and OOF 0.656. ID/popularity features are 0.10-0.15 there.
  - Caveat: this pair subset is chosen by the label, so its residuals are not directly actionable (regime routing gained 0.0000 in the segments lens). The actionable part is the ablation below.
- Recipe, step 1: ablate the fine-tune branch on the cold-positive / warm-positive split, with one fine-tune per group removed, 5 folds, same seeds. Groups: (a) ALS; (b) near-duplicate/neighbour context; (c) train-period item/user statistics; (d) LUO + co-visitation. Also run (e) a fine-tune with the branch zeroed, to measure how much of the +0.005 is plain re-calibration of the backbone on val labels. The aim is to find which group produces the +0.005. About 5 × 7 min on some GPUs.
- Recipe, step 2: move that group's train-stage analogue into `--xt` with causal cutoffs (train rows of day d use only days < d; ALS for train rows must be fitted leave-user-out or on days < d). If the zero-rate of the ID inputs differs between train and val/test, randomly blank the co-visitation/LUO inputs per item for a fraction q of train candidates so the zero-rate matches test (seq_cv_pop = 0: val 0.420, test 0.473; it_hist_all_luo = 0: 0.379 / 0.447). Pick q from {0, 0.15, 0.3} on the random-like holdout.
- Acceptance: holdout gain; a val gain on all-cold-positive feeds with warm-positive feeds not negative; test-mix-weighted val delta > 2 SE; at least 2 seeds.
- Expected: 0 to +0.002 val, somewhat more on private.

**R3. Stop or drop (saves compute):**
- the val residual stacker on nn_final3 and every feature family in §3.2-3.6;
- category-intent heads/models/priors (S3, CI-H2, CI-H3, MP-H2);
- set-level re-ranking (FS-H4);
- hand-made concentration/novelty/supply features (MP-H3);
- the NDCG loss weight (already neutral);
- co-visitation variants (S1);
- a cold offset;
- listing-age features;
- content-kNN COEC (S2a; fails the gate);
- train-label taste profiles (stacker; train-stage version optional, ≤ +0.0005).

**R4. Realistic outlook.** Every lens and test agrees that the remaining headroom is item discrimination inside a category, but none of our inputs can recover it. Twenty-plus families, including three new information sources tested here, came in below +0.001 once the NN had co-visitation and LUO popularity. The public top 10 (+0.008 val) probably have information or capacity we lack. The cheap candidates left are not feature ideas:
- (i) R2;
- (ii) more train-stage ensemble diversity, which the NN report measures at +0.0007 from adding a DIN member;
- (iii) longer or larger train-stage runs selected on the holdout.

---------------------------------------------------------------------------------------------------------------------

## 6. Files

- `qt.py` (stacker quick test; snapshot-aware), `compare.py` (paired deltas, Thursday half), `02_screen.py` (AUC, clustered residual t, nested blend), `base_scores.py`, `09_blends.py`.
- New features (aligned with cand arrays): `derived/{val,test}/tp_*` (T1), `ag_*` (T3), `cold_item`, `cold_noimpr`, `cvpop_zero`, `luo_zero`; `derived/val/kn_*` (T2, val only: the test-side search was stopped once T2 failed its gate).
- Analyses: `04_coldmix.py`, `05_coldflag.py`, `06_coldoffset.py`, `07_coldpairs.py`, `10_coldpair_screen.py`. Logs are in `logs/`; per-feed results are in `out/qt_*.npz` (old base: `out_old/`).
