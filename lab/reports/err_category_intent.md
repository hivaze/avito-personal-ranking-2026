# Error lens: CATEGORY INTENT

Scripts: `<repo>/eda/errors/category_intent/` (`01`-`09`, shared code in `catlib.py`). Raw outputs (JSON, logs, per-feed arrays, models): `eda/errors/category_intent/out/`.
Protocol: the 117 leak feeds are excluded everywhere (val users who have a later feed in val or test). Every val feed has at least one positive. Category-level metrics use the 14,274 non-leak feeds with 2 or more categories. Item-level metrics use the 14,306 non-leak feeds. "±" is the per-feed SE. Deltas are paired per-feed. Any parameter picked on val was picked by nested 5-fold GroupKFold by user (pick on 4 folds, score the 5th). The stacker reference is now `stk2_res4_clean` (the leaky `stk2_res2_r3` was removed from preds/ during this run). "v5blend" = z(0.5·z(nn_final2_val_oof) + 0.5·z(stk2_res4_clean)), item NDCG 0.3865 on non-leak feeds.
Compute: CPU at 16 threads per process. One short GPU job (torch sequence model, 2 × 5 min) ran on GPU N only.

## 0. TL;DR

1. **Category intent is where the error is, but it is not a missing signal.** The NN's implied category ranking is already the best category ranker any of our inputs can produce. This holds for dedicated train-fitted models (GBDT, listwise, torch sequence model), and also for val-fitted combiners that see the NN plus 80 category features.
2. **The NN ranks categories much better than f3.** We aggregate NN scores per (feed, category) with log-sum-exp (LSE, i.e. the NN's implied P(category)). Category NDCG: **NN 0.7797 vs f3 0.7477 (+0.0319 ± 0.0014)**. f3 is no better than simply counting how many candidates the category has (0.7452; f3 − count = +0.0025 ± 0.0016, n.s.). Max- or mean-aggregation of the NN is much worse (0.752 / 0.693), so LSE is the right reading.
3. **Dedicated category-intent models (train, 464K feeds) do not beat the NN.** Paired deltas in category NDCG vs NN-LSE:
   - GBDT binary: −0.0022 ± 0.0009
   - GBDT lambdarank: −0.0012 ± 0.0009
   - GBDT on random-like feeds only: −0.0025 ± 0.0009
   - torch history-sequence model plus group features: −0.0008 ± 0.0009
   - history-only sequence model (no feed composition): −0.029
4. **Combining them with the NN gives nothing.**
   - Category level: a val-fitted (CV) stacker over NN group features + all 80 handcrafted category features + the train category model gives −0.0001 ± 0.0007 (NN + hand-made features) and +0.0004 ± 0.0007 (NN + catA).
   - Item level: decomposition score = NN with its category marginal swapped or tilted toward the model's log P(category), parameters chosen by nested CV. Deltas: **v5blend +0.0000 ± 0.0002 (S1), −0.0004 ± 0.0002 (A)**; nn_trainonly −0.0003 to −0.0006. The literal version "log P(cat) + within-category z(NN)" gives −0.012.
   - The only positive cell is nn_oof: +0.0004 ± 0.0002 (all 4 category models). The val fine-tune slightly blurs the category marginal, and the stacker already repairs this (0 on v5blend).
5. **Conversion rate.** One unit of category NDCG buys 0.83-0.92 units of item NDCG (partial-oracle sweep). The target +0.009 item would need about **+0.010 category NDCG over the NN (0.780 → 0.790)**. The best thing we built is −0.0008.
6. **Within-category ranking has more headroom than cross-category ranking.** Keep the NN's category marginal fixed and make the order inside each category perfect: 0.3827 → **0.679**. The category oracle gives 0.551. The "70% of pairwise error mass is cross-category" figure mostly reflects that there are more cross-category pairs. Since category ranking is saturated given our inputs, remaining gains must come from **which item inside the category**: item-ID / collaborative / near-duplicate signals.
7. **Recommendation:** stop investing in category-intent heads and features (this confirms the neutral N3 aux head). Do not integrate the category model into the submission. Per-candidate scores are exported in case the lead wants a 1-line stacker check (section 7), but the expected gain is ≤ +0.0005 and not significant on the v5 blend.

## 1. Category structure (val, non-leak)
- Feeds hold 12.0 categories on average. 0.2% have a single category. Positives fall in 1.23 categories per feed. 32% of candidates share a positive's category.
- 92.2% of positive categories appear somewhere in the user's history (vs 58.7% of all feed categories). 74.9% appear in the last 20 events (vs 27.9%).
- **6.5% of feeds have no positive category anywhere in the history.** NN category NDCG there is 0.505 (vs 0.799), and item NDCG is 0.277 vs 0.390. These are "exploration" clicks that history cannot foresee.

## 2. How well do we rank categories? (`01_catlevel_val.py`)
Category NDCG: rank the feed's categories, relevance = the category holds ≥ 1 positive, pessimistic ties. "+tb" = tie-break by category size (a free prior).

| category score | cat NDCG | vs f3+tb | vs NN-LSE |
|---|---|---|---|
| random | 0.4738 | −0.274 | −0.306 |
| n candidates in category (feed category share) | 0.7452 ± 0.0022 | −0.0025 ± 0.0016 | −0.0345 ± 0.0013 |
| f3 (raw, pessimistic) | 0.7458 | −0.0019 ± 0.0002 | −0.0339 |
| **f3 + tb** | **0.7477 ± 0.0022** | 0 | **−0.0319 ± 0.0014** |
| f1 max + tb | 0.7455 | −0.0022 ± 0.0010 | −0.0342 |
| hist exp-decay count (hl 10) + tb | 0.7413 | −0.0065 ± 0.0014 | −0.0384 |
| hist count last 20 / last 5 / all | 0.7386 / 0.7372 / 0.7224 | −0.009 / −0.011 / −0.025 | |
| hist recency (−min hpos) | 0.7230 | −0.0248 | |
| hist count by event type 10 / 1 / 5 / 2 / 3 / 4 | 0.7433 / 0.7324 / 0.7048 / 0.6881 / 0.6876 / 0.6800 | | |
| tte_uc_ctr (train user×category CTR) | 0.6988 | −0.049 | |
| maxsim32 max / cos_u mean | 0.7050 / 0.6505 | | |
| nn_trainonly: max / mean | 0.7522 / 0.6933 | +0.0044 / −0.0545 | −0.0275 / −0.0864 |
| **nn_trainonly: LSE (T=1)** | **0.7797 ± 0.0021** | **+0.0319 ± 0.0014** | 0 |
| nn_trainonly: LSE T=2 | 0.7803 | +0.0326 | +0.0006 ± 0.0008 |
| nn_oof LSE / stk2_res4_clean LSE / v5blend LSE | 0.7803 / 0.7794 / 0.7808 | +0.033 | +0.0007 / −0.0003 / +0.0011 (± 0.0005) |

Top-1 category hit rate:
- category size 0.542
- f3 0.520
- f1 0.516
- history (hl 10) 0.510
- **NN-LSE 0.569**
- v5blend 0.572

The size prior alone matches every single user×category signal. The NN adds +0.032 on top of it.

## 3. Dedicated category-intent models (`02_build_feats.py`, `03_train_catmodel.py`, `07_seq_catmodel.py`)

**Training set.** 464K train feeds:
- all 164K random-like feeds (p_random > 0.5, M ≥ 50)
- a uniform 300K sample of M < 50 feeds, where p_random is unreliable

This gives 4.27M (feed, category) rows. Val has 173K rows, test 357K.

**Features (80).** Category id; n, share, size rank; f3, its rank and gap; f1 max/mean/min, rank and gap; f2/f4/f5 aggregates.

History per category:
- counts: all, last 5, last 20
- min hpos and recency rank among categories
- exp-decay with half-life 3 / 10 / 30
- counts per event type (10)
- exp-decay for event types 1/2/3/5
- top-6 sources
- share, and within-feed ranks

Also tte_uc_ctr/imp and tte_c_ctr (time-aware, from train labels); cos_u and maxsim32 aggregates; M, G, H, hour, user0; 16 PCA components of the user embedding.

**Training protocol.** Early stopping on a 10% user-disjoint train holdout, never on val.

| model | val cat NDCG | vs NN-LSE | vs f3 |
|---|---|---|---|
| A: LightGBM binary, R+S (1297 iters) | 0.7775 ± 0.0021 | −0.0022 ± 0.0009 | +0.0298 ± 0.0014 |
| B: LightGBM lambdarank over the feed's categories (372 iters) | 0.7784 | −0.0012 ± 0.0009 | +0.0307 ± 0.0013 |
| C: LightGBM binary, random-like feeds only (667 iters) | 0.7772 | −0.0025 ± 0.0009 | +0.0294 ± 0.0013 |
| S1: torch, 2-layer transformer over the last 64 history tokens (category, type, source, position) + user token; per-category query = category embedding + MLP(group features); listwise softmax CE; epoch picked on train holdout | 0.7788 | −0.0008 ± 0.0009 | |
| S0: same, without group features (history-only intent, no feed composition) | 0.7506 | −0.0291 ± 0.0013 | |

**GBDT A gain share:**
- share 68%
- f3 7.8%
- h_exp10 2.9%
- n_rk 2.3%
- category 2.0%
- h_et1_exp10 1.8%
- maxsim 1.0%

History-only intent (S0, 0.751) is about as good as the size prior (0.745).

**Complementarity.** Per-feed correlation of category NDCG with the NN is 0.90-0.91. The model is better on 13-14% of feeds and the NN on 15-16%. The "oracle of two" would reach 0.80, but no learned combination recovers any of it (section 4). The disagreement is noise, not complementary information.

## 4. Does it add to the NN? (`04_decomp_val.py`, `06_val_cv_catstack.py`)

**4a. Item level, nested-CV decomposition.**
- s = per-feed z of the base score; lq = model log P(category).
- `swap(T,a)`: s/T + a·(lq − logP_NN(c;T)), where logP_NN(c;T) is the within-category LSE of s/T. Grid: T ∈ {0.5,1,2}, a ∈ {0.1..1}.
- `add(b)`: s + b·lq.
- `literal`: lq + within-category z(s).
- `any`: nested choice over the swap and add grids together.

| cat model | base | literal | swap (nested) | add (nested) | any (nested) |
|---|---|---|---|---|---|
| A | nn_trainonly 0.3827 | −0.0127 ± 0.0011 | −0.0003 ± 0.0002 | −0.0008 ± 0.0002 | −0.0003 ± 0.0002 |
| A | nn_oof 0.3861 | −0.0122 | **+0.0005 ± 0.0002** | +0.0004 ± 0.0002 | +0.0005 ± 0.0002 |
| A | v5blend 0.3865 | −0.0122 | −0.0004 ± 0.0002 | −0.0004 ± 0.0002 | −0.0006 ± 0.0002 |
| B | nn_trainonly / nn_oof / v5blend | −0.013 | −0.0006 / +0.0002 / +0.0002 (± 0.0002-3) | −0.0004 / +0.0004 / −0.0000 | |
| C | nn_trainonly / nn_oof | −0.012 | −0.0004 / −0.0001 | −0.0004 / +0.0004 | |
| S1 | nn_trainonly / nn_oof / v5blend | −0.012 | −0.0006 / +0.0004 / +0.0000 | −0.0008 / +0.0004 / +0.0000 | |

The nested choice always lands on the smallest weight in the grid (a = 0.1, b = 0.05). Even the in-sample best grid point on v5blend is ≤ +0.0002.

**4b. Category-level stacker fitted on val.** LightGBM binary on groups, 5-fold GroupKFold by user, fixed 600 rounds. NN group features = LSE at T = 0.5/1/2, log P_NN, LSE rank, max, mean.

| base | NN only | NN + 80 hand-made features | NN + catA | hand-made only |
|---|---|---|---|---|
| nn_trainonly (vs NN-LSE) | −0.0002 ± 0.0006 | −0.0001 ± 0.0007 | +0.0004 ± 0.0007 | −0.0038 ± 0.0010 |
| v5blend | −0.0001 ± 0.0005 | −0.0007 ± 0.0006 | −0.0009 ± 0.0007 | −0.0048 ± 0.0010 |

Item swap with these OOF category scores: −0.0001 to +0.0002 (± 0.0001-0.0002).

**Segments** (nn_trainonly, NN + hand-made features stacker delta): every segment is within 1 SE of zero.
- M < 30, 30-100, ≥ 100: −0.0003 / +0.0004 / −0.0005, each ± 0.001
- H < 20: +0.0020 ± 0.0027
- user0: +0.0022 ± 0.0029
- feeds whose positive category is not in history: −0.0008 ± 0.0023

Category-trend features (time-of-day/day category shares, `trd_ctod/cres/cctr`) were already tested by the trd agent: no gain (reports/feat_trd_.md).

## 5. Category gains → item gains (`05_sensitivity.py`)
Partial category oracle on the NN: s + k·[category holds a positive], within-category order unchanged.

| k | Δ category NDCG | Δ item NDCG | item per category |
|---|---|---|---|
| 0.05 | +0.0061 | +0.0050 | 0.83 |
| 0.1 | +0.0120 | +0.0100 | 0.83 |
| 0.2 | +0.0234 | +0.0205 | 0.88 |
| 0.5 | +0.0541 | +0.0498 | 0.92 |

A random 10% of feeds given the full oracle: +0.0101 category → +0.0095 item.

**Rule of thumb:** Δitem ≈ 0.85-0.9 · Δcategory NDCG. Our category models move category NDCG by −0.0008 to −0.0025, so the item-level nulls in section 4 are what this predicts.

## 6. Where the headroom actually is (`08_oracles.py`)

| | nn_trainonly | nn_oof |
|---|---|---|
| base | 0.3827 | 0.3861 |
| category oracle (positives' categories first, NN order inside) | 0.5514 | 0.5576 |
| **within-category oracle** (each cell's own NN score values re-assigned so positives get the top ones; category marginal unchanged, verified) | **0.6788** | **0.6845** |
| both | 0.9754 | 0.9776 |

Both oracles are unreachable. Still, the within-category oracle shows more headroom than the category oracle, and category ranking is saturated given our inputs (sections 3-4). So the +0.008-0.010 we need has to come from **item discrimination inside categories**, e.g.:
- leak-free ALS and co-visitation item profiles
- item-ID residuals
- near-duplicate / same-listing signals
- leak-safe item popularity

It will not come from better category intent.

## 7. Integration (if anyone still wants to try)
- Per-candidate exports (float32, aligned with `data/flat/<split>/cand_*.npy`):
  - `eda/errors/category_intent/out/catint_A_{val,test}.npy`: GBDT A, log P(category holds a positive)
  - `catint_S1_{val,test}.npy`: torch S1, listwise logit
- Group-level preds and models (`cat_A.txt` etc.) and group features (`grp_{train,val,test}.npz`) are in the same folder.
- Only plausible use: one extra feature in the residual stacker or the fine-tune branch, to repair the small category-marginal blur of val fine-tuning (nn_oof +0.0004 ± 0.0002). The v5 blend already shows 0, so expect ≤ +0.0005. That is below the 2-SE bar on the final blend, and not worth a submission.

## 8. Files
| script | content |
|---|---|
| `catlib.py` | views, (feed, category) groups, history-per-category numba stats, category NDCG, leak mask |
| `01_catlevel_val.py` | category NDCG of simple signals vs NN aggregations (section 2) |
| `02_build_feats.py` | 80 group features for the train subset / val / test |
| `03_train_catmodel.py` | GBDT category models A/B/C |
| `04_decomp_val.py` | item-level decomposition, nested CV (4a) |
| `05_sensitivity.py` | category → item conversion |
| `06_val_cv_catstack.py` | val-fitted category stacker, coverage, segments (4b, section 1) |
| `07_seq_catmodel.py` | torch sequence category model (GPU N) |
| `08_oracles.py` | category vs within-category oracles, complementarity |
| `09_export.py` | per-candidate exports |
