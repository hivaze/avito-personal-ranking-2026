# Solvers workflow: v11 error analysis and solver complementarity

Author: solvers/errors agent. Date 2026-11-11 (after CURRENT_STATUS 3l). CPU only (`CUDA_VISIBLE_DEVICES=""`, 16 threads). Nothing under nn/, src/, pretrain/, data/feats/, preds/ or subs/ was changed.
Code: `<repo>/solvers/errors/` (`s00`..`s16`, `nncpu/`). Logs and tables: `<repo>/solvers/out/errors/` (`sNN*.log`, `*.csv`). Large intermediates: `/tmp/scratchpad/solvers/errors/out/`.

**Base everywhere:** v11 = 0.5 z(`nn_final5_val_oof`) + 0.5 z(`nn_final5_trainonly_val`), per-feed z. Reproduced val **0.39793** (`s00.log`; src/calib.py: public 0.38978 / private 0.39244). All deltas are paired per-feed, "±" = per-feed SE. "Nested" = choice made on 4 of the 5 nn/finetune.py folds (sklearn `GroupKFold(5)` on `user_id`, `nn/finetune.py:194-197`) and applied to the 5th. Random-order baseline = mean of 20 random permutations (0.28904).

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **The NDCG gap sits in the middle of the ranking, not at the bottom** [verified, `s01`, `s03`]. 83% of the gap (1 − 0.398) comes from positives at ranks 3-99. Only 19.9% of positives fall in the bottom half and 1.5% in the bottom 10%. Moving every bottom-half positive up to the median would add only **+0.0069**. Moving *every* positive up by one rank would add +0.046. Low-ranked positives are a small lever; the main lever is ordering among the top 3-30 candidates.
2. **No candidate signal has residual information beyond v11** [verified, four independent tests]:
   - local symmetric-window AUC (positive vs negatives within ±5/±10 ranks of v11): all 70 attributes lie within 0.492-0.504, |z| ≤ 2.9 (`s02b`);
   - nested conditional-logit residual over z(v11) with 32 signals: −0.0006 ± 0.0005 (`s04b`);
   - nested LightGBM residuals: −0.0015 to −0.011, depending on the objective (`s04`);
   - inside near-duplicate "twin" pairs, v11 AUC is 0.629 and the best single signal 0.564 (`s16`).
   
   Low-ranked positives look like average negatives on every attribute (maxsim32 0.70 vs 0.73 for negatives, f1 0.33 vs 0.40, cold share 0.30 vs 0.29).
3. **Every residual learner and every weaker solver moves along a single axis: cold vs warm** [verified].
   - The LightGBM residual gains **+0.011 ± 0.0015 on all-cold-positive feeds** (z ≈ 7) and loses more on warm feeds; net −0.0015.
   - Fine-tuned (FT) solvers win on cold (+0.008-0.010 alone), and train-only (TO) solvers win on warm.
   - v11 sits at the val optimum of this trade:
     - a cold offset gives at most +0.00013 (flat from 0 to +0.1);
     - the FT/TO weight 0.5 beats 0.25 and 0.75 in all three model families;
     - feed-level gates for "this click is cold" or "this click is exploratory" reach AUC 0.66-0.72 and gain nothing, against oracle ceilings of +0.016 to +0.025 (`s06c`, `s07`, `s10`, `s11`).
4. **Complementarity of existing solvers is low, with one exception** (`s05`, `s14`):
   - No single solver adds more than 2 SE to v11 at pre-declared weights of 0.1, 0.2 or 1.0.
   - Every solver more than ~0.005 below v11 standalone *hurts* at w = 0.2. That covers the GBDTs, pre_final, BCE/nohist/noset, older NN generations, the val stacker and the pre2 e3 members.
   - **The exception is pooling the members of different recipes at equal member weight:**
     - 6 v12 plateau + 6 v15 cooldown + the 3 non-collapsed v13 hedges, each half 0.5 z(FT) + 0.5 z(TO), called **S3**: val **+0.00115 ± 0.00044** (z 2.6);
     - S3 is positive on every slice: M<50 +0.0017, M≥50 +0.0008, Tue +0.0012, Thu +0.0011, all-cold +0.0018, all-warm +0.0009;
     - holdout (train-only half) **+0.00128 ± 0.00027** (0.32290 vs 0.32162);
     - stale-val with private-like coverage (train-only half, CPU replay): +0.00136 ± 0.00047, against +0.00083 at base, so the gain does not shrink.
     
     calib.py predicts public 0.3910 / private 0.3936. File: `solvers/out/errors/cand_S3_{val,test}.npy`.
5. **Ensemble scaling:** a 7th same-recipe seed is worth about +0.0001; a member from another recipe is worth +0.0002-0.0005; a whole new recipe family is worth +0.0008 (`s09`, `s14`). Future GPU time is better spent on recipe diversity (cooldown continuations, lr / pool-mix variants, pre2s) than on more seeds of one recipe.
6. **Adversarial validation** (`s08`):
   - val vs private, on the train-stage inputs: AUC 0.58, driven by f4 and the popularity / co-vis counts; 0.517 once those counts are removed.
   - late random-like train vs private: AUC **0.754**, of which popularity / co-vis + f4 is most of the signal; 0.585 without the counts.
   
   This confirms SHIFT-1 from the model-input side: the train-to-test drift is concentrated in the coverage-dependent counts. The cold-item calibration of v11 drifts from Tue to Thu (actual/expected 1.003 → 1.016; n.s.), but the NDCG cost of that drift is ≤ 0.0004 (`s06b`, `s06c`).

---------------------------------------------------------------------------------------------------------------------

## 1. Loss decomposition (v11, val, `s01.log`, `s03.log`, `s15_lossshares.log`)

Per-feed NDCG quantiles (5/25/50/75/95%): 0.156 / 0.221 / 0.310 / 0.500 / 1.000. Other feed-level facts:
- 9.1% of feeds are perfect;
- 36.6% score below their own random expectation;
- 17.3% score below 0.2.

The "gap share" of a segment is its share of Σ(1 − NDCG). "gainR" = (NDCG − random)/(1 − random).

| segment | feeds % | NDCG | random | lift | gainR | gap share |
|---|---|---|---|---|---|---|
| 1 positive | 74.5 | 0.393 | 0.283 | 0.110 | 0.154 | 75.1 |
| 2 positives | 15.4 | 0.395 | 0.291 | 0.105 | 0.147 | 15.4 |
| 3 | 5.0 | 0.407 | 0.305 | 0.102 | 0.147 | 5.0 |
| 4+ | 5.1 | 0.467 | 0.359 | 0.108 | 0.169 | 4.5 |
| M < 10 | 7.2 | 0.687 | 0.540 | 0.147 | **0.319** | 3.8 |
| M 10-24 | 15.1 | 0.529 | 0.397 | 0.132 | 0.219 | 11.8 |
| M 25-49 | 18.5 | 0.433 | 0.310 | 0.123 | 0.179 | 17.4 |
| M 50-99 | 22.1 | 0.362 | 0.253 | 0.109 | 0.146 | 23.4 |
| M 100-199 | 20.9 | 0.307 | 0.216 | 0.091 | 0.116 | 24.1 |
| M 200+ | 16.2 | 0.273 | 0.195 | 0.078 | **0.096** | 19.5 |
| H 0 / 1-10 / 11-50 | 1.4 / 2.2 / 7.3 | 0.376 / 0.433 / 0.406 | | 0.106 / 0.155 / 0.120 | | 1.4 / 2.1 / 7.2 |
| H 51-128 / 129-255 / 256 | 18.8 / 41.4 / 29.0 | 0.404 / 0.401 / 0.387 | | 0.117 / 0.109 / **0.097** | | 18.6 / 41.2 / 29.5 |
| new user | 65.0 | 0.401 | 0.290 | 0.111 | 0.157 | 64.7 |
| user with train feeds | 30.4 | 0.391 | 0.290 | **0.101** | 0.142 | 30.8 |
| user 0 | 4.6 | 0.404 | 0.274 | 0.130 | 0.179 | 4.5 |
| all positives cold | 17.4 | 0.367 | 0.280 | **0.088** | **0.122** | 18.3 |
| all positives warm | 70.9 | 0.402 | 0.287 | 0.116 | 0.162 | 70.4 |
| mixed | 11.7 | 0.418 | 0.318 | 0.099 | 0.145 | 11.3 |
| Tue / Thu | 52.3 / 47.7 | 0.398 / 0.398 | | 0.109 / 0.109 | | 52.3 / 47.7 |

- **Positive count.** Feeds with one positive and feeds with two or more positives contribute to the gap in proportion to their feed share. Per positive, multi-positive feeds are not harder.
- **Feed size.** Feeds with M ≥ 100 are 37% of feeds and carry 44% of the gap; their normalised gain is a third of that of small feeds.
- **Users with train feeds** remain harder after stratifying on H × M: lift −0.0034 vs new users (`s12.log`). Users whose last train feed is ≥ 15 days old have lift 0.092 (n = 645; not significant). The private median gap is 15.5 days (review_shift), so this is a weak staleness hint only.
- **Cold candidates.** Cold positives (never a train candidate nor in a train history) are 22.8% of positives and carry 23.1% of the gap; their mean percentile rank is 0.320 vs 0.266 for warm positives.

**Rank distribution of positives** (21,983 positives):
- mean percentile rank 0.278 (random 0.5);
- top-1 8.2%; top 10% 30.8%; top 25% 55.5%;
- bottom half 19.9%; bottom 25% 5.8%; **bottom 10% 1.5%**;
- bottom-half positives carry 23.5% of the gap, bottom-10% positives 1.8%;
- 15.4% of feeds have their *best* positive in the bottom half (19.9% of the gap).

| positive rank | share of positives | share of gap |
|---|---|---|
| 0 | 8.2% | 0 |
| 1-2 | 11.1% | 9.5% |
| 3-9 | 21.7% | 25.4% |
| 10-29 | 26.4% | **32.0%** |
| 30-99 | 24.7% | 26.0% |
| ≥ 100 | 7.9% | 7.1% |

Rank-space oracles:

| oracle | NDCG | Δ |
|---|---|---|
| every positive moved up 1 rank | 0.4436 | +0.046 |
| every positive moved up 3 ranks | 0.5132 | +0.115 |
| bottom-half positives moved to the median | 0.4048 | **+0.0069** |
| positives below the 25th percentile moved to it | 0.4413 | +0.043 |
| positives already in the top 3 moved to rank 0 | 0.4551 | +0.057 |

The leverage is fine ordering among the top 30, which is mostly *familiar* candidates:
- positives whose category is in the history and whose maxsim32 is ≥ 0.7 are 79.0% of positives and 74.7% of the gap; v11's AUC on such pairs is 0.681 (`s13`);
- history-unrelated positives (maxsim32 < 0.7, 18.1%) carry 21.6% of the gap;
- inside the unfamiliar subsets v11 ranks well (AUC 0.72 in both). Their loss comes from competing with familiar negatives, which are 1.37× more likely to be positive (§2.3).

---------------------------------------------------------------------------------------------------------------------

## 2. Low-ranked positives (`s02.log`, `s02_profile.csv`, `s02b.log`, `s04*.log`, `s10`-`s13`, `s16`)

### 2.1 Profile (means)

Groups: positives in the top 10% (n = 6,762), positives at the 10-50th percentile (10,843), positives in the bottom half (4,378), all negatives (1.37M), and negatives in the top 10% (139K).

| attribute | pos top10% | pos 10-50% | **pos bottom 50%** | neg all | neg top10% |
|---|---|---|---|---|---|
| maxsim32 | 0.886 | 0.816 | **0.702** | 0.734 | 0.872 |
| maxsim32 < 0.7 | 4.2% | 15.4% | **46.2%** | 38.1% | 6.0% |
| category in history | 97.5% | 95.2% | **82.6%** | 84.0% | 96.8% |
| category within the last 3 events | 73.4% | 52.7% | **28.2%** | 39.0% | 70.7% |
| cos_u | 0.551 | 0.519 | 0.453 | 0.473 | 0.547 |
| f1 / f3 / f5 | 0.62 / 0.66 / 0.56 | 0.53 / 0.56 / 0.51 | **0.33 / 0.38 / 0.43** | 0.40 / 0.45 / 0.44 | 0.60 / 0.65 / 0.55 |
| f4 (age proxy) | 0.568 | 0.549 | 0.492 | 0.504 | 0.572 |
| it_hist_all_luo = 0 | 23.8% | 30.2% | **40.4%** | 38.1% | 22.9% |
| seq_cv_pop = 0 | 29.9% | 34.4% | 44.2% | 42.1% | 28.6% |
| cold item | 20.1% | 21.5% | **30.0%** | 29.0% | 18.9% |
| near-duplicate (cos > 0.95) in feed | 25.5% | 20.8% | 13.8% | 21.9% | 28.7% |
| same-category candidates in feed | 35.2 | 34.0 | 20.1 | 38.0 | 51.5 |
| pre2_fe | 17.9 | 16.2 | 13.6 | 14.4 | 17.7 |
| argmax-history event type 1 | 71.9% | 69.1% | 61.0% | 62.7% | 70.8% |
| H / user 0 / new user | 172 / 5.4% / 63.6% | 179 / 4.6% / 61.9% | 182 / 3.7% / 62.4% | 173 / 5.0% / 64.0% | 174 / 5.0% / 64.1% |

- **Reading** [verified]. Low-ranked positives look like average negatives on every candidate attribute, and feed-level attributes (H, user type, history event-type and source shares) barely differ. In other words, they are "negative-looking" clicks.
- **History-event context:**
  - P(pos) lift by the event type of the most similar history item is 1.09 for type 1 and 0.64-0.88 for types 2, 3, 4, 7 and 9;
  - "category has event type k in the history" gives lifts of 1.04-1.15 (1.40 for type 10, a rare type);
  - all of these are already inputs (`cat_cnt_et*`, history tokens with event type and source).

### 2.2 Residual tests (does anything know more than v11?)

| test | result |
|---|---|
| **Symmetric local AUC** (`s02b`): positives with K ≤ rank ≤ M−1−K vs the negatives in ranks r ± K, so that v11 itself is ≈ 0.5 by construction; 149K pairs at K = 5, 237K at K = 10 | All 70 candidate attributes lie in 0.492-0.504; largest \|z\|: f4 −2.9, cat_recent3 −2.2, ag_age −2.2, maxsim_lt07 −2.1, nd95_any +2.1. 70 tests make \|z\| ≈ 2.5-3 the expected maximum under the null. On low-ranked positives only: max \|z\| 2.4. Other models: nn_final4_val_oof 0.5105 (z 4.4), and 0.532 (z 6.6) on low-ranked positives; nn_final5_val_oof 0.5064 vs nn_final5_trainonly 0.4936. These are anti-symmetric by construction, because v11 is their mix. |
| **Nested conditional-logit residual** (`s04b`): per-feed softmax CE, linear in z(v11) + within-feed z of the signals, L2 1e-4, LBFGS | all 32 signals: −0.00058 ± 0.00049 (all-cold +0.0047 ± 0.0011); pre_* only: −0.00036 ± 0.00026; near-dup/neighbour only: +0.00005 ± 0.00027; age/cold only: −0.00043 ± 0.00028 (all-cold +0.0042). The fitted scale on z(v11) is 0.95-0.97, so v11 is already at the softmax-optimal temperature. |
| **Nested LightGBM residual** (`s04`): init = z(v11), 47 candidate features (+16 feed features), exact folds, 2 seeds | binary (stk2-style init): **−0.0110 ± 0.0009** (it learns between-feed base rates; adding M as a feature reduces this to −0.0022). rank_xendcg: −0.025 (trees re-shape z(v11) non-monotonically). lambdarank with a monotone constraint on z(v11): **−0.0015 ± 0.0006 (100 rounds) … −0.0030 (300)**, with all-cold **+0.0121 ± 0.0015**, Tue −0.0025, Thu −0.0004. |
| **Twins** (`s16`): 19.8% of positives have a same-feed negative at item cos ≥ 0.95; the twin outranks the positive in 37% of pairs; perfect twin disambiguation would add +0.0163 | AUC inside (pos, twin-neg) pairs: v11 0.629, pre_fe 0.564, cos_u 0.562, maxsim32 0.561, f1 0.532, popularity 0.52-0.53, f4 0.490, item_id 0.495, ag_age 0.496. f3 is identical within twins, and f2 in 86% of them. No signal can break twin ties beyond what v11 already does. |

**Methodological finding** [verified]: a residual GBDT over a strong listwise NN must use a within-feed objective *and* a monotone constraint on the base score. A binary objective learns feed-level base rates (−0.011), and an unconstrained ranking objective spends its trees re-shaping the base (−0.025). The "stacker adds nothing" verdicts earlier in the project used XGB binary with M as a feature, which is roughly the −0.002 regime here, so the conclusion stands. But any future stacker should use lambdarank + monotone base.

### 2.3 "If the model knew X" upper bounds (`s10`, `s11`)

| oracle (knows) | ceiling Δ | realistic version (nested over user folds) | realistic Δ |
|---|---|---|---|
| all categories of the feed's positives (block-first) | +0.171 | category intent is saturated (err_category_intent) | - |
| whether a feed is "exploratory" (15.2% of feeds; v11 0.281 there vs 0.419 elsewhere), then adds β·z(−maxsim32) | **+0.0159** | LightGBM gate on 22 feed features, AUC 0.72 | ≤ 0 (every β < 0; nested 0.0000) |
| same gate with −cat_cnt_all / log popularity | +0.0146 / +0.0062 | same gate | −0.00007 ± 0.00020 / 0 |
| feed's positives are all cold, then cold boost 1.0 | **+0.0253** | gate AUC 0.66 | −0.00003 ± 0.00021 |
| perfect twin disambiguation | +0.0163 | no signal (§2.2) | - |
| per-feed switch between v11 and its A/A-like twin (FT half = W3 re-FT) | +0.0036 (43.5% of feeds differ) | noise floor of per-feed oracles | - |

- Unfamiliar subsets (`s13`): inside the set of candidates whose category is not in the history, v11 AUC is 0.722 (cos_u 0.635, f3/f5 0.618, popularity 0.600). Inside maxsim32 < 0.7 it is 0.718.
- So v11 orders unfamiliar items well among themselves. The loss on unfamiliar positives comes from the correct prior P(pos | unfamiliar) / P(pos) = 0.42-0.48, which places them below familiar negatives.
- Only per-feed intent knowledge would fix that, and observable intent gates (AUC 0.66-0.72) are far too weak: wrongly flipped warm feeds cost more than the correctly flipped ones gain.

---------------------------------------------------------------------------------------------------------------------

## 3. Complementarity of existing solvers (`s05.log`, `s05_complement.csv`, `s05_segments.csv`, `s07`, `s09`, `s14`)

### 3.1 Single solvers vs v11

"Alone" = the solver alone minus v11. Blends: z(v11) + w·z(S). "Nested" chooses w in {0, 0.1, 0.2, 1} per fold.

| solver | val | alone | win / loss feeds | z-corr | w 0.1 | w 0.2 | w 1.0 | nested |
|---|---|---|---|---|---|---|---|---|
| FT5 (fta OOF) | 0.39784 | −0.0001 | 28.7 / 30.8% | 0.997 | −0.0002 | −0.0002 | −0.0002 | −0.0002 ± 0.0001 |
| TO5 | 0.39712 | −0.0008 | 29.1 / 30.1% | 0.997 | +0.0001 | +0.0001 | −0.0005 | +0.0001 ± 0.0001 |
| FT5v2 (W3 re-FT, u0 singleton) | 0.39764 | −0.0003 | | 0.997 | 0.0000 | +0.0001 | −0.0003 | −0.0001 ± 0.0002 |
| FT5cd / TO5cd | 0.39846 / 0.39702 | +0.0005 / −0.0009 | | 0.989 / 0.983 | +0.0003 / +0.0002 | +0.0003 / **+0.0005** | +0.0002 / +0.0005 | +0.0000 / +0.0001 |
| FT5cd12 / TO5cd12 | 0.39837 / 0.39818 | +0.0004 / +0.0003 | | 0.995 / 0.994 | 0.0000 / +0.0003 | +0.0001 / +0.0001 | +0.0002 / +0.0004 | +0.0000 / +0.0004 ± 0.0004 |
| FT4 / TO4 / mix4 (v10 without pre) | 0.3932 / 0.3928 / 0.3940 | −0.005 / −0.005 / −0.004 | | 0.96-0.97 | ≈ 0 | +0.0002 / +0.0001 / +0.0002 | **−0.0011 / −0.0016 / −0.0010** | +0.0002 / −0.0001 / 0.0000 |
| v13 lr1 mix0 / lr2 mix1 (TO) | 0.39433 / 0.39403 | −0.0036 / −0.0039 | | 0.958 / 0.962 | 0.0000 / +0.0003 | +0.0002 / +0.0001 | −0.0009 / −0.0007 | +0.0002 / +0.0003 ± 0.0002 |
| FT3 / TO3 / FT2 / TO2 | 0.3896 … 0.3825 | −0.008 … −0.015 | | 0.92-0.96 | −0.0002 … −0.0005 | −0.0007 … −0.0010 | −0.003 … −0.004 | 0 (w = 0 chosen) |
| v16 pre2 e3 (2 seeds) | 0.38952 | −0.0084 | | 0.955 | −0.0003 | −0.0006 | −0.0028 | 0 |
| v9 din + gru + deep (e5) | 0.38934 | −0.0086 | | 0.962 | −0.0005 | −0.0009 | −0.0028 | 0 |
| stk2_res4_clean (val stacker) | 0.38491 | −0.0130 | | 0.928 | −0.0003 | −0.0008 | −0.0045 | 0 |
| pre_final (seq encoder) | 0.35900 | −0.0389 | 35.1 / 57.1% | 0.686 | −0.0008 | −0.0011 | −0.0100 | 0 |
| xgb_x300_luo (train GBDT) | 0.36723 | −0.0307 | | 0.813 | −0.0005 | −0.0010 | −0.0049 | 0 |
| cb_tr300k_luo (YetiRank) | 0.36577 | −0.0322 | | 0.768 | −0.0007 | −0.0012 | −0.0077 | 0 |
| bce / nohist / noset (round 1) | 0.373 / 0.368 / 0.374 | −0.025 / −0.030 / −0.024 | | 0.83-0.87 | −0.0001 … −0.0004 | −0.0008 … −0.0010 | −0.005 … −0.007 | −0.0003 / 0 / 0 |

- **Frontier rule** [verified on 25 solvers]:
  - a solver needs to be within ~0.004-0.005 of v11 standalone, with z-corr ≤ ~0.98, to be non-negative at w = 0.2;
  - all 12 solvers with a gap ≥ 0.008 lose at w = 0.2 (−0.0006 to −0.0012), including the most different ones (corr 0.69-0.87);
  - "different paradigm" alone does not buy complementarity here; strength comes first.
- **C4 = eq(v10, v11)** is −0.0010 ± 0.0006 (the mix4 w = 1.0 row). It should not be a hedge candidate.
- **Per-feed oracle switching** is mostly noise: v11 vs its A/A-like twin already gives +0.0036, and v11 vs FT5v2 +0.0080. The oracle grows with weakness (pre_final +0.041) rather than with usefulness. It is reported in `s05.log` only for completeness; do not use it to rank solvers.

### 3.2 Segments: who wins where (solver alone minus v11; `s05.log`)

| solver | all-cold (17.4%) | all-warm (70.9%) | mixed (11.7%) | M<50 | M≥50 |
|---|---|---|---|---|---|
| FT5 | **+0.0077** | −0.0025 | +0.0031 | −0.0014 | +0.0008 |
| TO5 | **−0.0083** | +0.0013 | −0.0026 | +0.0004 | −0.0016 |
| FT5cd | +0.0101 | −0.0023 | +0.0033 | +0.0004 | +0.0006 |
| TO5cd12 | −0.0055 | +0.0018 | −0.0004 | +0.0013 | −0.0005 |
| xgb_x300_luo (0.031 below v11 overall) | −0.0018 | −0.0391 | −0.0226 | −0.0328 | −0.0293 |
| noset / bce / nohist | +0.0043 / −0.0014 / −0.0034 | −0.032 / −0.032 / −0.038 | −0.017 / −0.021 / −0.023 | | |

Pair-type AUC (`s06.log`; share of pairs cold+/warm− 16.1%, warm+/cold− 21.8%, warm/warm 54.7%, cold/cold 7.4%):

| model | cold+ vs warm− | warm+ vs cold− | warm+ vs warm− | cold+ vs cold− |
|---|---|---|---|---|
| v11 | 0.6677 | 0.8169 | 0.7134 | 0.7708 |
| FT5 | 0.6795 | 0.8089 | 0.7131 | 0.7710 |
| TO5 | 0.6546 | 0.8238 | 0.7125 | 0.7698 |
| FT5cd | 0.6817 | 0.8096 | 0.7155 | 0.7722 |
| xgb_x300_luo | 0.6818 | 0.7411 | 0.6726 | 0.7353 |
| pre_final | 0.6276 | 0.7386 | 0.6488 | 0.7127 |

**Under-served segments.** By lift and gain ratio:
- all-cold-positive feeds: lift 0.088 vs 0.116 for all-warm;
- large feeds: gain ratio 0.096 at M ≥ 200;
- H = 256: lift 0.097;
- users with train feeds: lift 0.101.

The only segment where existing solvers disagree *systematically* is cold vs warm, and every solver that is better on cold pairs is worse on warm-vs-cold pairs by about as much:
- FT vs TO: +0.025 cold+/warm− AUC vs −0.015 warm+/cold− AUC;
- xgb_x300_luo matches FT5cd on cold+/warm− (0.682) but loses 0.07 on warm+/cold−.

Same-category pairs are not separately reported here; err_SYNTHESIS covers them. No solver is better on both. This is the signature of a missing-information frontier, not of a mis-weighted blend.

The kind of solver that *would* help cold feeds is one that raises cold+/warm− AUC **without** a warm prior penalty. That requires information about cold items that is not in the current inputs. The content-kNN, taste-profile, age, twin and gate tests here and in err_SYNTHESIS all fail to supply it.

### 3.3 FT/TO weight and routing (`s07.log`)

| family | w_FT 0 | 0.25 | **0.5** | 0.75 | 1.0 |
|---|---|---|---|---|---|
| nn_final5 | −0.0008 | −0.0005 | 0 | −0.0002 | −0.0001 |
| nn_final5cd12 | −0.0006 | −0.0002 | 0 | −0.0005 | −0.0004 |
| nn_final4 | −0.0012 | −0.0004 | 0 | −0.0005 | −0.0008 |

- The mix is a robust optimum: averaging beats either component; it is not a matter of one component being better.
- Nested routing of w_FT ∈ {0.25, 0.5, 0.75} by observable segment:

  | routing variable | nn_final5 | nn_final5cd12 | nn_final4 |
  |---|---|---|---|
  | M | +0.0003 ± 0.0003 | −0.0001 | −0.0002 |
  | cold share | −0.0003 | −0.0003 | −0.0005 |
  | user type | −0.0002 | +0.0003 | −0.0004 |
  | H | −0.0006 | −0.0001 | −0.0003 |

  Nothing replicates across families. Keep 0.5, with no routing.
- Caveat: all user-0 feeds sit in one fold (P7), so u0 routing is degenerate.

### 3.4 Ensemble scaling and cross-recipe pooling (`s09.log`, `s14.log`, `s14b.log`)

Mean val over all subsets of the 6 nn_final5 members:

| k | TO | FT | mix |
|---|---|---|---|
| 1 | 0.39247 | 0.39435 | 0.39392 |
| 2 | 0.39495 | 0.39640 | 0.39629 |
| 3 | 0.39586 | 0.39712 | 0.39695 |
| 4 | 0.39624 | 0.39741 | 0.39739 |
| 5 | 0.39669 | 0.39758 | 0.39782 |
| 6 | 0.39712 | 0.39784 | 0.39793 |

- A fit of a − b/k on the mix gives a(∞) = 0.39866. The predicted gain is +0.0003 at k = 12 and +0.0005 at k = 24 for *same-recipe* seeds.
- The TO curve was still rising faster at k = 6 than this fit predicts, so read the extrapolation as a lower bound.

Adding **one** member from another recipe to the 6-member mix:

| added member | Δ |
|---|---|
| v13_lr1_mix0 | +0.00022 ± 0.00029 |
| v13_lr1_mix1 | +0.00029 |
| v13_lr2_mix1 | +0.00023 |
| v15_cd_all_e10 | +0.00003 |
| v15_cd_allprew_s83 | +0.00048 ± 0.00028 |

Pre-declared pools (equal member weight inside each half, final = 0.5 z(FT half) + 0.5 z(TO half); n2cpf fine-tunes; holdout = TO half, `preds/holdout`):

| pool | members | val | Δ val | M<50 | M≥50 | Tue | Thu | all-cold | all-warm | holdout TO-half | Δ holdout (late 20%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 = v11 | 6 v12 | 0.39793 | - | | | | | | | 0.32162 | - |
| S1 = cd12 | + 6 v15 | 0.39873 | +0.00080 ± 0.00043 | +0.0011 | +0.0006 | +0.0009 | +0.0007 | +0.0016 | +0.0005 | 0.32272 | +0.00109 ± 0.00026 (+0.0007) |
| S2 | + 3 v13 | 0.39874 | +0.00081 ± 0.00039 | +0.0011 | +0.0006 | +0.0003 | +0.0013 | +0.0004 | +0.0008 | 0.32234 | +0.00071 ± 0.00024 (+0.0010) |
| **S3** | **+ 6 v15 + 3 v13 (15)** | **0.39908** | **+0.00115 ± 0.00044** | +0.0017 | +0.0008 | +0.0012 | +0.0011 | +0.0018 | +0.0009 | **0.32290** | **+0.00128 ± 0.00027** (+0.0011 ± 0.0006) |
| F3 (equal family weight) | same 15 | 0.39910 | +0.00117 ± 0.00048 | +0.0020 | +0.0006 | +0.0010 | +0.0014 | +0.0014 | +0.0010 | 0.32274 | +0.00112 ± 0.00029 |

- **Leave-one-out** (`s14b`):

  | S3 minus | Δ val vs v11 |
  |---|---|
  | v13_lr1_mix0 | +0.00107 |
  | v13_lr1_mix1 | +0.00066 |
  | v13_lr2_mix1 | +0.00091 |
  | all v15 | +0.00081 |
  | all v12 (v15 + v13 only) | +0.00085 |
- **Test file:** z-corr with v11 on test 0.9967; finite; 2,904,806 rows (`solvers/out/errors/cand_S3_test.npy`).
- **Selection caveat** [honest]: the five pools were declared in the script docstring before running, but the idea to pool v13 came from the per-member additions in `s09`, which are on val. Expect ~0.0002-0.0003 optimism on val. The holdout confirmation (z 4.7) is independent of that selection, because member checkpoints were holdout-selected and the holdout was not used here to choose the pool.
- **Correlation caveat:** the v13 hedges share seed 23 and are identical to v12_all_e10 up to 77.5k steps (CURRENT_STATUS 3k). v15 members are cooldown continuations of the v12 checkpoints. So "15 members" is fewer than 15 independent draws, and the gain is a lower bound for genuinely independent recipes.
- **Overlap with nn_final6:** v13_lr1_mix1 is also a member of nn_final6 (CURRENT_STATUS 3k, 17:45). Count it once when F6 is pooled.

### 3.5 Stale-val check of the pools (TO halves; `nncpu/cf_pool.py`, `s17.log`)

- **Method.** All 15 train-only members were replayed on CPU with read-only use of nn/ckpt. Every replay matches its stored prediction with corr 0.999999. The replays use val inputs rebuilt to private-like popularity coverage: review_shift's `both` scenario, i.e. train sources cut at Sep 17 and no test snapshots in it_hist_* / seq_cv_*, with prew_* masked to match.
- **Per member,** the stale cost is −0.0004 to −0.0015. Family means: v12 −0.0010, v15 −0.0012, v13 −0.0012.

| pool (TO half) | base val | stale val | Δ vs S0 base | Δ vs S0 stale | diff-in-diff |
|---|---|---|---|---|---|
| S0 (6 v12) | 0.39691 | 0.39542 | - | - | - |
| S1 (+ v15) | 0.39771 | 0.39657 | +0.00080 ± 0.00048 | +0.00115 ± 0.00047 | +0.00035 ± 0.00048 |
| S2 (+ v13) | 0.39724 | 0.39617 | +0.00033 ± 0.00039 | +0.00075 ± 0.00039 | +0.00042 ± 0.00043 |
| **S3 (15)** | 0.39774 | 0.39679 | +0.00083 ± 0.00049 | **+0.00136 ± 0.00047** | +0.00053 ± 0.00049 |
| v15 only | 0.39695 | 0.39625 | +0.00005 | +0.00083 ± 0.00072 | +0.00078 ± 0.00055 |
| v13 only | 0.39637 | 0.39526 | −0.00054 | −0.00016 | +0.00038 ± 0.00058 |

- **Result.** The pooling gain does not shrink under emulated private coverage; if anything it grows (diff-in-diff +0.0005 ± 0.0005, n.s.).
- **Replay offset.** The replayed S0 TO half is 0.39691 vs 0.39712 for the stored file. That is replay float noise near ties, of the same size as review_shift's 0.0001. All comparisons in this table are replay vs replay.

---------------------------------------------------------------------------------------------------------------------

## 4. Time shift, cold calibration, adversarial validation (`s06.log`, `s06b.log`, `s06c.log`, `s08.log`)

### 4.1 Cold-item calibration and FT/TO disagreement over time

Actual / expected positives, where expected = npos · softmax(c·z) per feed and c is fitted by listwise CE (0.94-0.96). Feed bootstrap SE.

| model | cold Tue | cold Thu | LUO = 0 Tue | LUO = 0 Thu | young & LUO = 0 Tue | young & LUO = 0 Thu |
|---|---|---|---|---|---|---|
| v11 | 1.003 ± 0.013 | 1.016 ± 0.011 | 0.991 ± 0.010 | 1.018 ± 0.010 | 0.986 ± 0.016 | 1.001 ± 0.016 |
| FT5 | 0.980 | 0.992 | 0.980 | 1.005 | 0.950 | 0.965 |
| TO5 | 1.022 | 1.038 | 0.999 | 1.028 | 1.017 | 1.034 |

- Every model and definition drifts up by +0.013 to +0.027 from Tue to Thu. That is consistent with cold items becoming more "mainstream" as staleness grows, but each difference is about 1 SE.
- The raw cold lift P(pos | cold)/P(pos) is 0.766 on Tue and 0.807 on Thu.
- **The NDCG value of re-calibrating is tiny**:

  | cold offset δ in z(v11) + δ·cold | Δ all | Tue | Thu |
  |---|---|---|---|
  | +0.05 | +0.00012 | +0.0002 | −0.0000 |
  | +0.10 | +0.00013 | +0.0007 | −0.0005 |
  | +0.20 | −0.0006 | | |

  Thu (colder) prefers a *smaller* offset than Tue, so this view gives no evidence that private wants a cold boost. LUO = 0 offsets behave the same way.

Test side (`s06.log`):
- The FT−TO z gap on cold items is +0.036 on val, +0.043 on public and +0.043 on private.
- corr(zFT, zTO) is 0.990 on every split.
- v11's top-1 cold share divided by the cold share of candidates is 0.69 on val, 0.68 on public and 0.68 on private, flat by day.
- The model treats cold items the same way on every split. No test-side anomaly.

### 4.2 Adversarial validation on the train-stage inputs

Setup: 28 cvnd inputs + f1-f5 + maxsim32 / cos_u / cos_h10 / cat_cnt_all / cat_min_hpos; M and H excluded. Candidate level, 200K rows per side, 5-fold grouped by feed.

| pair | AUC | without the 8 popularity / co-vis counts | top features |
|---|---|---|---|
| val Tue vs Thu (control) | 0.541 | | cat_cnt_all, f3, f4 |
| val vs private | **0.581** | 0.517 | f4 0.24, it_hist_all_luo 0.11, cat_cnt_all 0.10 |
| val M≥50 vs private M≥50 | 0.582 | | same |
| random-like train (last 3 days, M≥50) vs val M≥50 | **0.703** | 0.575 | it_hist_all_luo 0.27, seq_cv_pop 0.26, f4 0.14 |
| random-like train (last 3 days) vs private M≥50 | **0.754** | 0.585 | it_hist_all_luo 0.30, f4 0.24, seq_cv_pop 0.18 |

- **Reading** [verified]: the train/test drift seen by the NN is in the coverage-dependent counts and in their interaction with f4 (age). Even causal `seq_cv_pop` drifts, because test items are staler.
- Content / similarity / context inputs drift only at the val-vs-Tue/Thu control level (0.54-0.585).
- The causal / lagged / horizon rebuilds that would address this are already rejected (FEATURES_REVIEW FR-3/4). The stale-val yardstick (in progress elsewhere) is the right tool to judge any change in this direction.

---------------------------------------------------------------------------------------------------------------------

## 5. Checked and correct

- **v11:** 0.39793 reproduced from its formula; calib.py gives 0.38978 / 0.39244.
- **Loss decomposition:** the per-positive shares sum exactly to 1 − NDCG (0.60207; `s00.log`).
- **Folds:** `GroupKFold(5)` on `user_id` with sklearn 1.9.1 (shuffle=False default) is the split in nn/finetune.py:194-197. All user-0 feeds sit in one fold (known P7). Nested choices for user-0 segments are therefore degenerate; I flagged them and did not use them.
- **pre2 on val/test:** `pre2_fe` / `pre2_final` on val are numerically the same as `pre_fe` / `pre_final` (corr 0.99999; 89% of rows differ only at 1e-6). The pre2 rebuild changed train rows only, so the fine-tune branch's pre_* and a pre2 FT branch would be equivalent on val/test.
- **No residual signal:** the symmetric local-AUC design gives v11 itself 0.5017. It is a valid residual detector. The asymmetric version (`s02`) is biased by top-truncated windows (maxsim32 shows 0.517), so the symmetric version is the one used.
- **Test files of S3:** finite, length = test candidates, z-corr 0.9967 with v11.
- **Leaky inputs:** none used. The GBDTs in §3 are the clean LUO ones (xgb_x300_luo, cb_tr300k_luo); stk2_res4_clean uses clean ALS. The leaky cb_tr500k_pw / lgb_tr300k_* / non-LUO GBDTs were excluded (their json shows non-LUO `it_hist_*`).
- **User-item re-impressions** (the same user saw the item in train): 0.18% of val candidates, lift 1.04. Negligible (`tte_ui_imp`).
- **Earlier conclusions re-confirmed on v11:**
  - set-level re-ranking and category intent are saturated (err_feed_structure, err_category_intent);
  - twins have no disambiguating signal;
  - the cold offset is null;
  - the val stacker on the current base is negative.

---------------------------------------------------------------------------------------------------------------------

## 6. Ranked proposals (expected gain is on val unless stated)

**P1. Cross-recipe equal-member pool as a final candidate (C5). CPU, immediate, no training.**
- **Mechanism.** Members of different recipes have partly independent errors, and their cold/warm biases differ (plateau vs cooled vs lr / pool-mix hedges). Averaging them reduces variance more than more seeds of one recipe do.
- **Evidence (§3.4):**
  - S3 val +0.00115 ± 0.00044 (z 2.6), positive on every slice;
  - holdout TO-half +0.00128 ± 0.00027 (z 4.7), late-20% +0.0011;
  - leave-one-out +0.0007 to +0.0011;
  - calib.py public 0.3910 / private 0.3936 (v11: 0.3898 / 0.3924);
  - stale-val TO half +0.00136 ± 0.00047 (§3.5).
- **Experiment.**
  - Declare C5 = equal member weight over every completed, non-collapsed, holdout-selected member of recipes (c), (ii), (iii) and cd, each with its n2cpf fine-tune: final = 0.5 z(mean z FT) + 0.5 z(mean z TO).
  - Now: v12 ×6 + v15_cd ×6 + v13 {lr1_mix0, lr1_mix1, lr2_mix1}, already in `solvers/out/errors/cand_S3_{val,test}.npy`.
  - Later: nn_final6/7 members, counting v13_lr1_mix1 once.
  - The lead copies the file into preds/ under a new name and runs nn/pcmp.py and the frozen checks.
- **Acceptance (frozen protocol):** paired val > 2 SE; no slice < −2 SE; holdout ensemble ≥ 0.32162; file checks; make_sub tie-free ranks. If v11 stays pick 1, C5 is the natural pick 2: within 1 SE of v11, test Spearman 0.9965.
- **Expected gain:** +0.0008-0.0012 val after ~0.0003 selection optimism; private similar.
- **Cost:** CPU minutes.
- **Risk:** not on the frozen candidate list (a lead decision); correlated members (shared seed / trajectory), so the true gain could be at the low end.

**P2. Spend GPU on recipe diversity: a cooldown-continuation family for every new plateau family. GPU N only.**
- **Mechanism.** One cooled-continuation family added +0.0008 val / +0.0011 holdout to its plateau parent (S1 vs S0), at about 1/10 of the cost of a new family. Six more same-recipe seeds are predicted to add about +0.0003 (k-curve, §3.4).
- **Experiment.**
  - For each nn_final6 member (and later nn_final7), take its holdout-selected plateau checkpoint and run the N2 continuation exactly as for v15_cd: mix 0.1, 1-epoch cooldown, holdout-selected step.
  - Then run the n2c_pre fta fine-tune.
  - Cost: 6 × (~30 + ~7) GPU-min on GPU N, at most 3 concurrent workflow processes.
  - Compare pool(C5 + F6 + F6cd) with pool(C5 + F6) at equal member weight.
- **Acceptance:** ensemble-level z ≥ 2.5 on paired val, or > 2 SE pooled with the second family; holdout TO-half ≥ 0; no slice < −2 SE.
- **Expected gain:** +0.0003-0.0008.
- **Risk:** late-cooldown pool memorisation (seen at mix 0.1 in N2); the holdout-selected step guards against it.

**P3. Judge every cold/warm-trading change on a staleness-robust yardstick, never on val alone. CPU.**
- **Mechanism.** On val, v11 sits at the optimum of the cold/warm trade:
  - residuals gain +0.011 on all-cold feeds and lose more elsewhere;
  - the cold offset is flat;
  - w_FT = 0.5 is optimal in three families;
  - Thu (colder) does not prefer more cold boost.
  
  Private is colder, so the right point may differ, but val cannot tell.
- **Experiment.**
  - For any new train-only member that changes cold handling (e.g. age-conditioned ID dropout, review_ideas #6), replay it on CPU with `solvers/errors/nncpu/cf_pool.py` (~100 s per member per scenario), using the review_shift `both` / `trunc17` counterfactual inputs.
  - Report base-val and stale-val paired deltas and their difference-in-differences.
- **Acceptance:** base val ≥ −1 SE and stale-val > 2 SE (two seeds), or the unified rule as usual. Keep w_FT = 0.5 and no cold offset unless this passes.
- **Expected gain:** it prevents ±0.001 private mistakes; 0 on val.
- **Cost:** CPU only.
- **Risk:** the emulation covers popularity / co-vis coverage only (not history-side co-vis staleness, which is slightly optimistic), and only for TO halves.

**P4. Frontier gate before blending any new-paradigm solver. CPU rule.**
- **Mechanism.** At v11's level a blend component must be strong. All 12 solvers ≥ 0.008 below v11 lose at w = 0.2, even at z-corr 0.69-0.87. The components that do not lose are within ~0.004 (§3.1).
- **Experiment.** Before spending effort on a GBDT, a pointwise / pairwise NN, a zero-shot encoder or a stacker as a blend component, require its standalone val within 0.004 of the matching v11 half. Then run `solvers/errors/s05_complement.py` (pre-declared w ∈ {0.1, 0.2, 1}, nested over the finetune folds).
- **Acceptance:** nested delta > 2 SE and holdout ≥ 0 (for train-only components).
- **Expected gain:** 0 directly; it saves compute.
- **Cost:** none.
- **Risk:** it misses a weak-but-orthogonal solver. None of the 25 tested was one.

**P5. If a residual stacker is ever revisited, use a within-feed objective with a monotone base. CPU.**
- **Mechanism / evidence (§2.2):** a binary residual loses −0.011 (it learns feed base rates); unconstrained xendcg loses −0.025; lambdarank + monotone base loses only −0.0015, and its loss comes entirely from the cold/warm trade.
- **Experiment:** `solvers/errors/s04_resid.py` with `OBJ=lambdarank MONO=1`, over a new candidate signal plus z(v11), 2 seeds, exact folds.
- **Acceptance:** paired val > 2 SE pooled over 2 seeds, Thu ≥ 0, all-warm ≥ −1 SE.
- **Expected gain:** 0 to +0.0003; no current signal is expected to pass.
- **Cost:** about 1 CPU-min per fold.
- **Risk:** low.

**Not recommended (closed by this analysis):**
- feed-intent gates / mixture-of-experts by "exploration" or "cold click" (realistic Δ ≤ 0 vs oracle +0.016 / +0.025);
- FT/TO routing by M, cold share, H or user type;
- the cold offset;
- C4 = eq(v10, v11) (−0.0010);
- twin disambiguation features;
- event-type / source shares of the history as extra features (local z ≤ 2);
- blending pre_final, train GBDTs, BCE / nohist / noset or older NN generations into v11.

---------------------------------------------------------------------------------------------------------------------

## 7. Files

| step | script | log / output |
|---|---|---|
| base arrays | `s00_base.py` | v11 score, ranks, per-positive loss |
| decomposition | `s01_decomp.py` | `s01.log`, perfeed.parquet |
| rank-space oracles | `s03_rankloss.py` | `s03.log` |
| profile | `s02_profile.py` | `s02.log`, `s02_profile.csv`, attrs.npz |
| symmetric local AUC | `s02b_local.py` | `s02b.log`, `s02b_local.csv` |
| LightGBM residual | `s04_resid.py` | `s04_*.log` |
| conditional-logit residual | `s04b_clogit.py` | `s04b_*.log` |
| complementarity | `s05_complement.py` | `s05.log`, `s05_complement.csv`, `s05_segments.csv` |
| cold axis / calibration / offset | `s06_coldaxis.py`, `s06b_coldcal.py`, `s06c_offset.py` | `s06*.log` |
| FT/TO routing | `s07_route.py` | `s07.log` |
| adversarial validation | `s08_adv.py` | `s08.log` |
| ensemble curve | `s09_enscurve.py` | `s09.log` |
| intent gate | `s10_gate.py` | `s10.log` |
| oracles | `s11_oracles.py` | `s11.log` |
| train users | `s12_trainusers.py` | `s12.log` |
| unfamiliar subsets | `s13_unfamiliar.py` | `s13.log` |
| pools | `s14_xrecipe.py`, `s14b_xrecipe_test.py` | `s14*.log`, `cand_S3_{val,test}.npy` |
| twins | `s16_twins.py` | `s16.log` |
| stale-val replay | `nncpu/cf_pool.py`, `s17_stale.py` | `cf_pool.log`, `s17.log` |

Code is in `<repo>/solvers/errors/`, logs and outputs in `<repo>/solvers/out/errors/`.
