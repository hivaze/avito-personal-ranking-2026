# Error lens: feed structure and multi-positive feeds

Scripts and logs: `<repo>/eda/errors/feed_structure/` (`01_pairs.py` ... `06_dup_discriminators.py`, a `.log` next to each, JSON outputs `out_*.json`).
All numbers are on val with the **117 leak feeds excluded** (14,306 feeds), using the exact metric (`src/metrics.py`, pessimistic ties).
Base model: `nn_final2_val_oof` (non-leak 0.38612). Where stated, the clean `nn_final2_trainonly_val` (0.38269) was used as a robustness check.
"d = x ± y" is a paired per-feed NDCG delta with its SE. Compute: CPU only (16 threads); every job ran in under 2 minutes, so no GPU was needed.

## 0. TL;DR

1. **Positives in the same feed are clearly similar to each other.** Over the 3,641 feeds with ≥ 2 positives, positive-positive pairs share a category 43.2% of the time, against 26.4% for positive-negative pairs (+0.168 ± 0.006). Their mean cosine is 0.591 vs 0.510 (+0.081 ± 0.003), and P(cos > 0.9) is 6.9% vs 1.6%. They are also closer in f4/f5: the within-feed rank gap is smaller by 0.024 ± 0.003 for f4 and 0.027 ± 0.003 for f5.
2. **Near-duplicates are enriched for shared labels, but most of them do not share one.** Given that i is positive, a same-category candidate j with cos 0.98-0.99 is positive 8.8% of the time, against 1.3% when i is negative (6.6x). Even so, only **6-9% of a positive's near-duplicates (cos > 0.9 / 0.95 / 0.98) are positive**, and only 14-16% of positives with a near-duplicate have a positive one. No candidate shares an item_id with another in the same feed.
3. **None of this can be exploited on top of the NN.** I tested 64 set-level re-ranking rules: similarity propagation from the top-k, softmax-weighted propagation, kNN score smoothing, near-duplicate group max, MMR diversity, category softmax mass and category-of-top-1 boosts. The best gave **+0.00023 ± 0.00015** (t = 1.6). The 2-fold (even/odd feeds) selection estimate is **−0.00076**. On the clean train-only base it is −0.00011.
   - A LightGBM residual on 13 model-relative set features (binary objective, init = calibrated z, GroupKFold by user) gives set − control = **−0.00010 ± 0.00028** (100 rounds) and −0.00001 ± 0.00047 (300 rounds).
4. **Why it fails:** the model's top-1 is positive in only **11.3%** of feeds (4.5% for M ≥ 250), so propagating from it mostly spreads errors. Even with an *oracle* that tells us the top-1 is positive (463 multi-positive feeds), propagation gains only +0.001 to +0.004 in those feeds. That gives an overall ceiling of about +0.0001.
5. **The NDCG-optimal multi-positive correction is ~0.** Ranking by E[t/IDCG(N)] instead of the E[t/N] that the listwise softmax learns gives d = −0.00000 ± 0.00032 (estimated 2-fold).
6. **Ties do not matter.** None of nn_final2_oof, nn_final2_trainonly or stk2_res2_r3 has an exact positive-negative tie (0 of 2.43M pairs). Optimistic minus pessimistic NDCG is 0.000000, and only 0.065% of positive-negative pairs are within |dz| ≤ 1e-3. Tie-breaking with a second model (z + 1e-4·z2) changes nothing, and larger eps is just a blend (≤ +0.0002, n.s.).
7. **Feed size M:** the model's mean percentile rank of the positive is flat across M (0.28-0.31). The lift over random shrinks from +0.138 (M < 10) to +0.069 (M ≥ 250), and P(top-1 positive) falls from 0.36 to 0.045. The number of positives rises with M (1.08 → 2.25; corr 0.195). Positive coherence does not rise with M (same-category pp/pn ≈ 0.4/0.26 in every bin).
8. **Where the error actually is:** inside near-duplicate groups the model ranks the positive above a negative near-duplicate (cos > 0.95) only 61.1% of the time. f4 (49.0%), f5 (50.8%) and f1 (46.3%, 14% ties) do not separate them.
   - An oracle that lifts each positive to the best score in its near-duplicate group would give +0.017 ± 0.001 (cos > 0.95) and +0.046 (cos > 0.9).
   - Across 454 leak-safe features, only the fine-grained history-similarity family (`ctx_hs_k*`, `seq_top3`, `pre_v2x_*`, `attn32`) carries a small residual signal on these pairs: AUC 0.57-0.58, residual t ≈ +2.4 to +3.2, against about 1.2 features expected at |t| > 3 by chance.

**Bottom line for this lens: the set and feed structure is real but already absorbed by the NN (which has set self-attention). This lens yields no new information worth +0.008.** The remaining within-group error needs item-level signals that separate near-identical listings. None of our current features does that.

## 1. Multi-positive structure

| npos | 1 | 2 | 3 | 4 | 5 | ≥6 |
|---|---|---|---|---|---|---|
| val feeds (all) | 10,742 | 2,217 | 726 | 306 | 167 | 265 |
| NDCG nn_final2_oof (non-leak) | 0.3816 | 0.3811 | 0.3943 | 0.4131 | 0.4485 | |
| random NDCG | 0.2827 | 0.2915 | 0.3053 | 0.3210 | 0.3474 | |

25.5% of non-leak feeds have ≥ 2 positives. The lift over random is about the same for single- and multi-positive feeds (+0.099 / +0.090 for npos 1 / 2), so multi-positive feeds are not a special weakness.

### 1.1 Pair similarity (feeds with ≥ 2 positives; per-feed means over pairs, paired over 3,641 feeds)

| statistic | pos-pos | pos-neg | neg-neg | pp − pn | pp − nn |
|---|---|---|---|---|---|
| same category | 0.4315 | 0.2638 | 0.2159 | +0.1678 ± 0.0061 | +0.2156 ± 0.0069 |
| mean cos | 0.5907 | 0.5103 | 0.4913 | +0.0805 ± 0.0025 | +0.0994 ± 0.0028 |
| P(cos > 0.9) | 0.0686 | 0.0164 | 0.0117 | +0.0522 ± 0.0035 | +0.0569 ± 0.0037 |
| P(cos > 0.95) | 0.0299 | 0.0056 | 0.0041 | +0.0243 ± 0.0024 | +0.0258 ± 0.0025 |
| abs diff f4 within-feed rank | 0.3135 | 0.3371 | 0.3464 | −0.0236 ± 0.0029 | −0.0329 ± 0.0032 |
| abs diff f5 within-feed rank | 0.3082 | 0.3353 | 0.3466 | −0.0272 ± 0.0029 | −0.0384 ± 0.0031 |
| abs diff f1 | 0.1900 | 0.2698 | 0.2739 | −0.0798 ± 0.0032 | −0.0839 ± 0.0034 |
| abs diff f3 | 0.1779 | 0.2667 | 0.2754 | −0.0887 ± 0.0034 | −0.0975 ± 0.0036 |
| same item_id | 0 | 0 | 0 | | |

The f4/f5 effect is small: a 2-3 percentile-point gap against a 34-point random gap. The f1/f3 effect mostly reflects the shared category.

### 1.2 Conditional label lift P(t_j = 1 | t_i, cos(i, j), same category), ordered pairs (`01_pairs.log`, section C)

Base rates: P(t_j | t_i = 1) = 0.0217 and P(t_j | t_i = 0) = 0.0105. Same-category pairs:

| cos | anchor + : P(t_j=1) | anchor − : P(t_j=1) | ratio |
|---|---|---|---|
| 0.5-0.6 | 0.0173 | 0.0115 | 1.5 |
| 0.7-0.8 | 0.0400 | 0.0143 | 2.8 |
| 0.9-0.95 | 0.0546 | 0.0151 | 3.6 |
| 0.95-0.98 | 0.0629 | 0.0150 | 4.2 |
| 0.98-0.99 | 0.0878 | 0.0133 | 6.6 |
| 0.99-0.999 | 0.1033 | 0.0127 | 8.1 |

Cross-category pairs show the same gradient: 0.0476 vs 0.0112 at cos 0.7-0.8. Note that a candidate's own positive rate does **not** rise with near-duplicate density under anchor −: about 1.5% at any cos. Co-positivity is therefore an "intent" effect: the user wanted that kind of item. It is not an item-quality effect.

## 2. Near-duplicates (`03_dups_ties_M.log` (a), `05_kappa_oracles.log` (i), `06_dup_discriminators.log`)

| threshold | P(has dup) pos / neg | positives with a dup | share of their dups that are positive | P(≥1 positive dup) | pos ranked best in its group: model / random | oracle "lift positive to its group's best" |
|---|---|---|---|---|---|---|
| cos > 0.9 | 0.423 / 0.440 | 9,213 | 0.060 | 0.159 | 0.386 / 0.318 | +0.0457 ± 0.0011 |
| cos > 0.95 | 0.209 / 0.219 | 4,536 | 0.070 | 0.143 | 0.436 / 0.371 | +0.0172 ± 0.0006 |
| cos > 0.98 | 0.064 / 0.072 | 1,388 | 0.092 | 0.144 | 0.494 / 0.412 | +0.0045 ± 0.0003 |

- Having a near-duplicate is *not* a positive signal: positives have one slightly less often than negatives.
- Pairwise win rate of the positive over its negative near-duplicates (cos > 0.95; 11,509 pairs in 3,387 feeds): base z 0.611, f1 0.463 (13.8% ties), f2 0.072 (86% ties), f3 0.002 (99.8% ties), f4 0.490, f5 0.508.
- Over 454 leak-safe features, the residual coefficient next to Δz (pairwise logistic, feed-clustered SE) has |t| > 3 for only 5 features. Two of these are degenerate (`cat_cnt_et9/et10`, 99.99% ties). The other three come from one correlated family: `ctx_hs_e2`, `ctx_hs_k256` (t = 3.15) and `clu_roc_u1024` (3.0), with about 1.2 expected by chance. The marginal AUC of the best feature is 0.584 (`ctx_hs_k256`), below the base's 0.611.
- Duplicate-group rules on top of the NN are neutral or harmful: group-max pull d ≤ +0.00004, and −0.0012 ± 0.0002 at thr 0.9, b 0.5.

## 3. Feed size M (`03_dups_ties_M.log` (c))

| M | share | mean npos | P(npos ≥ 2) | NDCG | random | lift | P(top-1 positive) | mean pct rank of positives | same-cat pp vs pn (multi-pos feeds) |
|---|---|---|---|---|---|---|---|---|---|
| [0,10) | 7.2% | 1.08 | 0.065 | 0.676 | 0.538 | +0.138 | 0.358 | 0.294 | 0.53 / 0.24 |
| [10,20) | 10.5% | 1.16 | 0.110 | 0.536 | 0.416 | +0.121 | 0.195 | 0.301 | 0.39 / 0.22 |
| [20,30) | 9.3% | 1.23 | 0.155 | 0.456 | 0.345 | +0.112 | 0.137 | 0.305 | 0.41 / 0.27 |
| [30,50) | 14.0% | 1.33 | 0.216 | 0.411 | 0.302 | +0.109 | 0.114 | 0.290 | 0.43 / 0.27 |
| [50,100) | 22.1% | 1.46 | 0.253 | 0.349 | 0.253 | +0.096 | 0.080 | 0.287 | 0.42 / 0.26 |
| [100,150) | 12.8% | 1.64 | 0.309 | 0.303 | 0.222 | +0.081 | 0.062 | 0.289 | 0.42 / 0.26 |
| [150,200) | 7.9% | 1.81 | 0.372 | 0.287 | 0.207 | +0.080 | 0.065 | 0.285 | 0.38 / 0.25 |
| [200,250) | 6.4% | 1.95 | 0.389 | 0.265 | 0.197 | +0.067 | 0.042 | 0.279 | 0.50 / 0.28 |
| [250,301) | 9.7% | 2.25 | 0.448 | 0.262 | 0.194 | +0.069 | 0.045 | 0.277 | 0.47 / 0.26 |

- **The model's relative quality is M-invariant:** the positive sits at about the 28-30th percentile in every bin. The NDCG drop with M is the metric's geometry, not a modelling failure.
- **Order sanity check:** the val array order carries no position information. The target rate at the first / last array index is 0.0377 / 0.0378, against 0.0389 expected from mean(npos/M) under random order, and the rate is flat across array-position quintiles (0.0158-0.0161). Train truncation therefore cannot be exploited on val/test.

## 4. Set-level re-ranking rules on nn_final2_val_oof (`02_rerank_*.log`)

Rules act on per-feed z. "top1+ / top1−" is the delta in feeds where the model's top-1 is / is not truly positive (an oracle split, for diagnosis only).

| rule family (grid) | best variant | d (non-leak) | worst variant |
|---|---|---|---|
| simprop: z + a·max cos to top-k (k 1/3/5/10, a ±0.2..1) | top5, a 0.2 | +0.00014 ± 0.00022 | top1, a 1.0: −0.0021 ± 0.0005 |
| softprop: z + a·Σ softmax(z/τ)·cos (centered) | τ 1, a −0.5 | +0.00023 ± 0.00027 | τ 0.5, a 2: −0.0018 ± 0.0006 |
| kNN smoothing (thr 0.8/0.9/0.95, b 0.1-0.5) | 0.95, b 0.1 | +0.00023 ± 0.00015 | 0.8, b 0.5: −0.0024 ± 0.0007 |
| duplicate group max (thr 0.9-0.98) | 0.95, b 0.1 | +0.00004 ± 0.00004 | 0.9, b 0.5: −0.0012 ± 0.0002 |
| greedy MMR diversity (a 0.05-0.5, −0.1) | a −0.1 | +0.00004 ± 0.00009 | a 0.5: −0.0008 ± 0.0002 |
| category softmax mass (τ 0.5/1, a −0.2..0.5) | τ 1, a 0.1 | +0.00008 ± 0.00026 | τ 0.5, a −0.2: −0.0048 ± 0.0006 |
| boost category of top-1 (a −0.2..0.5) | a 0.1 | −0.00015 ± 0.00051 | a 0.5: −0.0059 ± 0.0012 (top1+ −0.220, top1− +0.021) |

- 2-fold selection (choose on even feeds, evaluate on odd, and vice versa): **−0.00076** (OOF base) and **−0.00011** (train-only base). No rule passes 2 SE, and the signs are not stable across bases.
- The oracle split shows the mechanism. Boosting the top-1's category gains +0.021 when the top-1 is wrong, because the positive is often in the top-1's category. It loses 0.22 when the top-1 is right, because same-category items overtake it. With P(top-1 positive) = 0.11, these effects cancel.
- **Oracle-gated propagation** (top-1 kept first, add a·cos(i, top1)): +0.0011 to +0.0040 in the 463 feeds where the top-1 is positive and npos ≥ 2, against −0.0002 to −0.0036 elsewhere.

### 4.1 Learned set features (`04_setfeat_stack.py`)

The 13 features were: log softmax mass of own category, number of better items in own category, category share, same category as top-1, gap to top-1, feed softmax entropy, cos to top-1, max cos to top-3 and top-10, softmax-weighted cos, number of dups > 0.95, dup group gap, and mean cos to feed. The setup was 5-fold GroupKFold by user on non-leak feeds, with init = z(nn_final2_trainonly).

| objective | rounds | control [z, M] − base | set − base | **set − control** |
|---|---|---|---|---|
| binary (init = logistic-calibrated z) | 100 | −0.00006 ± 0.00009 | −0.00016 ± 0.00027 | **−0.00010 ± 0.00028** |
| binary | 300 | −0.00091 ± 0.00030 | −0.00092 ± 0.00041 | −0.00001 ± 0.00047 |
| lambdarank (init = z) | 100 | −0.00163 ± 0.00057 | −0.00488 ± 0.00096 | −0.00325 ± 0.00098 |

The lambdarank residual overfits val noise; even the control loses. In the binary setup, M takes 73% of the gain, which is feed-level calibration and irrelevant for ranking.

### 4.2 NDCG-optimal multi-positive key (`05_kappa_oracles.py`)

E[NDCG] = Σ_i disc(r_i)·E[t_i / IDCG(N)]. A listwise softmax with target y/N learns about E[t_i / N] instead, which slightly under-weights candidates that tend to be co-positive. I estimated E[N | t_i = 1] from calibrated p and the lift table (fitted 2-fold) and re-ranked.

| key | d |
|---|---|
| A: p·E[1/IDCG] | −0.00061 ± 0.00046 |
| B: p·E[1/IDCG] / E[1/N] | −0.00000 ± 0.00032 |

The training-side equivalent, a per-feed loss weight N/IDCG(N) (1, 1.23, 1.41, 1.56, ...), is free to try in the NN. The post-hoc evidence suggests it is ≤ +0.0005.

## 5. Ties and near-equal scores

| prediction | pessimistic | optimistic | exact pos-neg ties | pos-neg pairs with abs dz ≤ 1e-3 / ≤ 1e-2 |
|---|---|---|---|---|
| nn_final2_val_oof | 0.38612 | 0.38612 | 0 / 2,428,822 | 0.065% / 0.66% |
| nn_final2_trainonly_val | 0.38269 | 0.38269 | 0 | 0.073% / 0.71% |
| stk2_res2_r3_val | 0.38546 | 0.38546 | 0 | 0.065% / 0.66% |

Tie-breaking z + eps·z2 with z2 from nn_final2_trainonly, stk2_res2_r3 or cb: at eps = 1e-4 the delta is exactly 0. At 0.01-0.05 it is +0.0001 to +0.0002 (n.s.), and from there on it is ordinary blending. **Nothing is left to gain from tie handling.** Keep emitting fp32 and z-score blends, never rank averages.

## 6. Recommendations

- Do not spend time on post-hoc set-level rules (propagation, diversity, category coherence, duplicate grouping). Their 2-fold-selected effect is ≤ 0, and the oracle-gated ceiling is about +0.0001.
- The only large reachable-looking ceilings in this lens are perfect within-near-duplicate disambiguation (+0.017 at cos > 0.95) and positive-category knowledge (+0.17, from the strategist). Both require **new item-level or intent information**, such as the item's own interaction history, listing freshness or seller, that separates near-identical listings. The current f4/f5/f1 features are at chance on these pairs.
- Cheap NN-side experiment: weight each feed's listwise loss by N/IDCG(N) (the NDCG-consistent target). Expected effect ≤ +0.0005. Evaluate on the random-like train holdout first.
