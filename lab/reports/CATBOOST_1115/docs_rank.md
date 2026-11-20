# CatBoost round 2, KEY = docs_rank: lever map of CatBoost learning-to-rank TRAINING

Written 2026-11-15. Installed version: CatBoost 1.2.10, CPU only. Marks: **[M]** measured by me on the installed version
(synthetic data, never project data, unless a line says otherwise), **[D]** read in documentation or source (URL or path
given), **[I]** inferred. "Record" = numbers already on file in the project (path given). I trained nothing on project data
and I report no val / holdout number of my own.

## 0. Verdict

1. **No ranking-training lever of CatBoost can alone lift a tree model from 0.385 to the member bar (about 0.3965 at rho 0.90)**: every one re-fits the same 177 inputs [I]. My estimate for their best combination: 0.385 to 0.388 val [I]. The inputs decide (docs_feat track), not the loss.
2. **Never used here and working on 1.2.10 CPU [M]:** teacher targets in native losses (`QueryRMSE` on the teacher logit; `QuerySoftMax` on `(1-w) y/sum(y) + w softmax(teacher/T)`, the NN students' own KD target; `YetiRank` on real-valued labels), trees beyond the GPU cap (depth to 16, `Depthwise`, `Lossguide`), metric-matched YetiRank, `init_model`, `sum_models`.
3. **Known nulls on record:** the hard-label objective (±0.001), more feeds, click-mixed and cross-fitted teacher targets, `1/M` weights, every baseline-on-NN residual form (section 1.2).
4. **Traps on the installed version (section 4):** unknown `YetiRank:` parameters are silently ignored; `LambdaMart` with Gradient leaves ranks backwards; a categorical that is constant inside a feed leaks same-feed labels on CPU; `QueryCrossEntropy` is GPU only.
5. **Verification:** 632 synthetic checks = 467 PASS, 80 rejections (exact error text kept), 63 ignored parameters, 2 defects, 19 errors of my own test code (re-run correctly); 11 mismatches between docs and installed version.
6. **CPU cost per iteration relative to the record's YetiRank [M, synthetic shape]:** `QueryRMSE` 0.3x, `LambdaMart` on clicks 0.4x, `QuerySoftMax` + Newton 0.6x, `mode=NDCG` 1.3x, `Lossguide` 1.3x to 1.7x, pairwise losses on soft labels 2.7x to 2.9x, `Ordered` 4.2x, `StochasticRank` 14x.
7. **Direction check on a synthetic analogue (section 6, 3 seeds, 3,000 eval feeds) [M]:** teacher targets +0.009 to +0.012 over hard-label YetiRank whatever the loss carries them (nine losses within 0.0026); no YetiRank variant, tree shape, sampling or two-stage scheme gained more than +0.0015 ± 0.0007.
8. **Recommendation:** run the student arms A to D of section 1.3 against a CPU hard-label baseline and bundle the survivors as one member; expect the stop rule (best tree model below 0.3900) to bite unless the inputs change.

## 1. Ranked levers

### 1.1 The levers most likely to matter (my ranking; effect sizes are [I] unless a record is cited)

The order is by how much a lever could move the result. Only levers 1 and 3 have a record in their favour; the sign of
levers 2, 4 and 6 is open.

"Never used" = absent from `nn_x/cb26/cb_member.py`, `nn_x/cb26/cb_resid.py`, `nn_x/w25/train_w25.py`,
`reports/deep_gbdt_ranking.md` §2.1 and `reports/sol_solvers.md` §2.5 [D, those files].

| # | Lever | Exact spelling (all verified to train on 1.2.10 CPU [M]) | Why it could matter here | Expected size | Status in the project |
|---|---|---|---|---|---|
| 1 | **Teacher targets in a native loss** (label form) | `loss_function="QueryRMSE"`, label = teacher logit (raw or centred: identical model [M]). Or `loss_function="QuerySoftMax"`, `leaf_estimation_method="Newton"`, label = `(1-w)*y/sum(y) + w*softmax(teacher/T)`. Or `loss_function="YetiRank"`, label = teacher logit | The only label lever with a positive record: LightGBM L2 on the centred teacher logit gave +0.0049 val / +0.0044 holdout over hard labels at 134K feeds (`reports/RECHECK_1114/long_bets.md` §4). `QuerySoftMax` on the mixed target is the students' own loss (`nn_x/w5/train.py:197` [D]) | +0.003 to +0.006 over the hard-label CatBoost record 0.3820; differences BETWEEN soft-label losses probably below 0.002. Cost per iteration: QueryRMSE 0.29x, QuerySoftMax + Newton 0.62x, YetiRank on a real-valued label 2.9x [M] | never used in CatBoost |
| 2 | **Tree shape beyond the GPU cap** | `grow_policy="Lossguide", max_leaves=127..255, min_data_in_leaf=50..200, depth=16` or `grow_policy="Depthwise", depth=8..10, min_data_in_leaf=...` or symmetric `depth=10..12` | The record ran at depth 8, the most the GPU allows for YetiRank [D]. A student that copies a smooth teacher function may need more leaves than 256 oblivious ones | ±0.002. Section 6: depth 8 +0.0015 ± 0.0007 over depth 6; Lossguide −0.002 | never used (GPU cap) |
| 3 | **A bundle of students as ONE member** | member-equal z-mean of 3 to 6 models that differ in loss (lever 1), tree policy (lever 2) and seed; or `sum_models(models, weights=...)` after rescaling each model | The member rule allows a declared bundle. Record: 4 diverse tree singles in a feed-z mean gave +0.0016 ± 0.0007 over the best single (`reports/deep_gbdt_ranking.md` §2.2) | +0.001 to +0.002 on the level; rho to the pool rises too | not tried with students |
| 4 | **Position weighting of hard-label YetiRank** | `"YetiRank:decay=0.95"` / `0.99` (default 0.85), `"YetiRank:mode=NDCG"` (log2 discount, no `top` cut), `"YetiRank:mode=MRR"` | CatBoost analogue of the truncation lever that WAS worth +0.004 in XGBoost (k=32 vs 300, `deep_gbdt_ranking.md` §1). Classic YetiRank weights adjacent pairs by 0.85^position [D]; our metric decays as 1/log2 | ±0.002. Section 6: no variant beats the default; `mode=NDCG` −0.0044 ± 0.0013 | never varied |
| 5 | **Two-stage curriculum** | stage 1 `QueryRMSE` on the teacher; stage 2 `fit(..., init_model=stage1)` with `YetiRank` on clicks at a low learning rate | distil-then-fine-tune; works across objectives and regressors [M] | at most +0.001 (the one-stage click mix is a null on record; section 6: −0.0004) | never used |
| 6 | **Regularisation at an effective size** | `rsm=0.3..0.5`; `l2_leaf_reg` on the loss's own scale; `random_strength`; `sampling_unit="Group", bootstrap_type="Bernoulli", subsample=0.5..0.8`; `model_shrink_rate` | YetiRank's own default `l2_leaf_reg` is 0 (1e-20) [M] and its hessians are small: 30 at about 7K rows per leaf cost −0.040 in section 6. Per row, the record's 5 at about 77K rows per leaf is about 66 times weaker than that [I]. 177 correlated features were never column-sampled | ±0.001; sign open (section 6: all about −0.001) | never varied (only `l2=5`) |
| 7 | **QuerySoftMax mechanics** (needed by lever 1) | `"QuerySoftMax:beta=B;lambda=L"`, `leaf_estimation_method="Newton"` | default leaves are `Gradient` with 1 step and learn slowly [M]; `beta` = inverse temperature on the score; `lambda` damps the Newton step and only acts under Newton [M] | enabling, not a gain | `QuerySoftMax` used at defaults only |
| 8 | **Baseline boosting on a neural score** | `Pool(..., baseline=b)` on train AND eval; final score = `b + trees` | mechanics verified for every loss [M]; but the result is the residual form, null three times on record, and it is not a different-bias member (rho to the pool near 1) | 0 | tried (cb26 E1 / E2, C01): null |
| 9 | **Cheap objectives as CPU budget** | `QueryRMSE` (0.29x YetiRank), `LambdaMart` on clicks (0.37x; 2.7x on real-valued labels), `QuerySoftMax` + Newton (0.62x), `"YetiRank:permutations=3"` (0.80x) | lets a CPU-only track afford seeds and bundles; the record configuration costs 8.3 s per iteration on CPU (`reports/CATBOOST_1115/infra.md` §2.3 [D]) | budget only | never used |
| 10 | **Early-stopping hygiene** | `eval_metric="NDCG:type=Base"`, `od_type="Iter"`, `od_wait=...`, selection set from TRAIN feeds | built-in `NDCG:type=Base` equals `src/metrics.ndcg_per_feed` to 1e-16 including pessimistic ties [M]; the record stopped on val labels, which the member rule now forbids | correctness | metric used; split must change |

### 1.2 Known nulls here (do not re-test without a new reason)

| Lever | Record | Source |
|---|---|---|
| Objective with hard labels: YetiRank / YetiRankPairwise / QuerySoftMax / Logloss / XGBoost binary / rank:ndcg | all within ±0.001, one paired SE about 0.0011, 14,423 feeds | `reports/deep_gbdt_ranking.md` §2.1 |
| More exploration feeds (485,023 vs 134,118) | −0.00125 ± 0.00074 val, −0.00187 ± 0.00044 holdout | `nn_x/w25/out/results.txt` A1 − A0 |
| More teacher-labelled feeds (134K vs 60K) | +0.00014 ± 0.00062 val, −0.00046 ± 0.00036 holdout | `reports/RECHECK_1114/long_bets.md` §4 |
| Teacher logit + 1.0·y as the target | 0.38362 vs 0.38469 val | same |
| Cross-fitted vs in-sample teacher target | +0.00025 ± 0.00061 val | same |
| Weight 1/M per feed | −0.0026 ± 0.0011 | `reports/deep_gbdt_ranking.md` §2.1 |
| Residual / baseline-on-pool forms | E1 −0.00031 / −0.00042; C01 −0.00264 ± 0.00064 val; column residuals at most +0.0006 | `reports/COVERAGE_MAP.md` rows 149, 235, 258 |
| W25 feature groups | +0.0007 ± 0.0005 val | `reports/COVERAGE_MAP.md` row 149 |
| A tree model at weight 0.05 to 0.2 in the pool | no weight gives a holdout gain (largest +0.00003 ± 0.00015, 30,000 feeds); val −0.0003 to −0.0013 on v11 | `reports/sol_solvers.md` §2.5; `long_bets.md` §4 |

### 1.3 Arms I would run first (exact parameters)

Common: `thread_count=12, border_count=254, random_seed=s, allow_writing_files=False, eval_metric="NDCG:type=Base",
od_type="Iter", od_wait=200, use_best_model=True, metric_period=25`, early stopping on a train-side selection set.

```python
# A  student, L2 on the teacher logit (cheapest: about 0.3x the record's cost per iteration)
CatBoostRanker(loss_function="QueryRMSE", depth=8, learning_rate=0.08, l2_leaf_reg=3)            # label = teacher logit
# B  student, the NN students' own listwise KD target
CatBoostRanker(loss_function="QuerySoftMax", leaf_estimation_method="Newton", depth=8, learning_rate=0.08)
#    label = 0.5 * y / sum_feed(y) + 0.5 * softmax_feed(teacher / T), T in {1.0, 2.5}; labels must be >= 0
# C  student, magnitude-aware pairwise
CatBoostRanker(loss_function="YetiRank", depth=8, learning_rate=0.08, l2_leaf_reg=5)             # label = teacher logit; 2.9x cost
# D  the best of A-C with non-symmetric trees
CatBoostRanker(loss_function="QueryRMSE", grow_policy="Lossguide", max_leaves=255, depth=16, min_data_in_leaf=100, learning_rate=0.08)
# E  optional: hard-label control for lever 4 (same data, same seed); section 6 gives it no support
CatBoostRanker(loss_function="YetiRank:mode=NDCG", depth=8, learning_rate=0.08, l2_leaf_reg=5)
```

Read A to D against the hard-label CatBoost baseline of the infra track with the progress rule (both witnesses, 2 SE
pooled, 3 seeds or paired per-feed SE). Then declare the bundle (lever 3) before reading it against the member rule.

## 2. Lever map

Columns: **CPU/GPU** from the docs [D]; **default** = value resolved by `get_all_params()` on 1.2.10 CPU [M] (file
`nn_x/w12/cb/docs_rank/out/resolved_defaults.json`); **used** = in an earlier project run; **cost** = seconds per
iteration relative to Classic YetiRank at depth 8 on 1.49M rows × 177 features, 12 threads [M, section 5].

### 2a. Ranking objectives

Sources [D]: https://catboost.ai/docs/en/concepts/loss-functions-ranking ;
https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/private/libs/algo/yetirank_helpers.cpp ;
`.../catboost/private/libs/algo_helpers/error_functions.cpp` and `.h` ; `.../catboost/private/libs/options/loss_description.cpp` ;
`.../catboost/private/libs/options/catboost_options.cpp` (same repository and tag).

| Objective | What it does [D] | Parameters, defaults ([D] unless marked) | CPU / GPU | Used | Cost | Relevance |
|---|---|---|---|---|---|---|
| `YetiRank` (Classic) | draws `permutations` noisy rankings of the feed; for each ADJACENT pair in a noisy ranking adds weight `0.15 · decay^position · abs(label_i − label_j)` to that pair; then a pairwise logistic loss on the weighted pairs | `mode=Classic`, `permutations=10`, `decay=0.85`, `noise=Gumbel` (also `Gauss`, `No`), `noise_power=1` (Gauss only). `top`, `num_neighbors`, `dcg_type`, `dcg_denominator` are ignored in Classic [M NOEFFECT] | CPU + GPU | yes: the record | 1.00 on clicks; 2.9 on a real-valued label; `permutations=3` 0.80 | baseline. `decay` and `permutations` change the fit [M]; `noise=No` is the one setting the source paper shows to hurt |
| `YetiRank:mode=NDCG` / `DCG` | same sampling, pair weight = \|change of the metric when the two swap\| (YetiLoss) | `top` (unlimited), `num_neighbors=1` (`-1` = all later items), `dcg_type=Base`, `dcg_denominator=LogPosition` [M: `LogPosition` is NOEFFECT, `Position` differs], `permutations`, `noise` | **CPU only** | no | 1.33; `num_neighbors=3` 1.81; `num_neighbors=-1` 15x on small data [M] | metric-matched variant; label scale is irrelevant (bit-identical for y and 10y [M]) |
| `YetiRank:mode=MRR` / `MAP` / `ERR` | MRR and MAP binarise the label at > 0; ERR reads the label as a probability in [0,1] | same | CPU only | no | 1.04 | hard labels only: with all-positive soft labels training aborts ("Too few sampling units") [M] |
| `YetiRankPairwise` | same pairs, pairwise SCORING of splits (slower, "more accurate on big data" per docs) | same; resolved `border_count=32`, Bernoulli 0.5, `random_strength=0` [M] | CPU + GPU (Classic) | yes (11-11 val-CV; `cb_tr500k_pw`) | far above 1: depth 9 took 6x depth 8 on 44K rows [M] | null on record; no non-symmetric trees, no Ordered, no monotone constraints, `rsm` crashes (LAPACK) [M] |
| `LambdaMart` | LambdaMART: for pairs with different labels where at least one is in the top `top`, lambda = `sigma · ΔM · rho` | `metric=NDCG` (`DCG`, `MRR`, `ERR`, `MAP`), `sigma=1`, `norm=true` (= divide ΔM by 0.01 + \|score gap\| AND rescale the feed by log2(1+Σλ)/Σλ), `top=-1`, `type`, `denominator` | **CPU only** | no | 0.37 on clicks; 2.7 on a real-valued label | cheapest ranking loss on clicks. `norm=false` is the analogue of the XGBoost setting that helped on record. `metric=DCG` is bit-identical to `NDCG` [M]. Never set `leaf_estimation_method="Gradient"` (section 4) |
| `StochasticRank` | smooths the metric with Gaussian noise on the scores and estimates its gradient | `metric` REQUIRED (`NDCG`, `DCG`, `PFound`, `FilteredDCG`, `ERR`, `MRR`; `MAP` rejected [M]), `sigma=1`, `num_estimations=1`, `mu=0`, also `nu`, `lambda` (not in the docs, they change the fit [M]), `top` | CPU only | no | 13.8 (`num_estimations=8`: 8x more [M]) | too slow at full list length: cost is O(M·min(M, top)) per feed [D]; `top=10` is 10x cheaper but no longer our metric |
| `QuerySoftMax` | listwise cross-entropy: `−Σ t_i log softmax(beta · s)_i`; gradient `beta · (w_i t_i − S p_i)`, `S = Σ w t` over the feed | `beta=1`, `lambda=0.01` (a term in the second derivative only) | CPU + GPU | yes (11-11; cb26 E1 / E2) | 2.7 with the default Gradient leaves (1.04 at 5.97M rows); **0.62 with Newton leaves** | accepts any labels ≥ 0, so a soft target makes it listwise KD [M]. Default `Gradient` leaves learn slowly; use Newton (section 2d) |
| `QueryCrossEntropy` | pointwise cross-entropy with a per-feed shift | `alpha=0.95` | **GPU only** | no | n/a | **FAIL on CPU**: "QueryCrossEntropy loss function is not supported for CPU learning" [M] |
| `QueryRMSE` | L2 after removing the per-feed mean of the residual: `der1 = t − s − mean_feed(t − s)` | none | CPU + GPU | no | 0.29 | L2 on the feed-centred teacher logit without centring by hand (raw and centred labels give the same model to 2e-7 [M]) |
| `PairLogit` | pairwise logistic over pairs; pairs are given, or generated: every pair of the feed with different labels, winner = larger label, weight = group weight | `max_pairs` (all) | CPU + GPU | no | 2.6 | uses the label ORDER only (identical model for any monotone transform [M]); with real-valued labels it generates M(M−1)/2 pairs per feed |
| `PairLogitPairwise` | same loss, pairwise scoring | `max_pairs` | CPU + GPU | no | not timed | same limits as `YetiRankPairwise` |
| `GroupQuantile` | quantile loss after removing the feed mean | `alpha=0.5` | CPU | no | not timed | no use with clicks (degenerate at the median of 0 / 1 [M]) |
| `StochasticFilter` | filtering objective, needs sorted input | `sigma`, `num_estimations` | CPU | no | not timed | not our task |
| `Logloss`, `CrossEntropy`, `RMSE` | pointwise | Logloss needs 2 label values or `target_border`; CrossEntropy needs labels in [0,1]; RMSE any real | CPU + GPU | Logloss (11-11) | 0.27 to 0.28 | `CatBoostRanker` rejects Logloss / CrossEntropy; use `CatBoost(params)` with `group_id` in the pool so that `NDCG` can be the eval metric [M]. RMSE on the teacher logit works; so do `Huber:delta=1.0`, `MAE`, `LogCosh` [M] |

### 2b. Group machinery

| Item | Behaviour | Mark |
|---|---|---|
| `group_id` | rows of a feed must be contiguous: "queryIds should be grouped". A ranking loss without groups: "Groupwise loss/metrics require nontrivial groups" | [M] |
| group size | no limit met on CPU: 6 feeds of 3,000 rows train with YetiRank, `mode=NDCG`, LambdaMart. YetiRank allocates an M × M float matrix per feed [D] | [M] |
| `group_weight` | must be equal inside a feed ("... is not equal to the weight of group's first element"); multiplies the feed's pair weights / loss; changes every loss tested | [M] |
| `weight` (per row) | accepted; cannot be combined with `group_weight` ("Pool must have either weight or group_weight"). YetiRank and YetiRankPairwise practically ignore it (correlation of predictions 1.000, same NDCG) | [M] |
| `use_weights=false` in a loss string | rejected for an objective: "If weights are present they will necessarily be used in optimization" | [M] |
| `subgroup_id` | accepted, no effect on YetiRank, QuerySoftMax, PairLogit | [M] |
| `pairs`, `pairs_weight` | used by PairLogit and PairLogitPairwise (a pool with pairs and NO label trains); silently ignored by YetiRank and QuerySoftMax; both members must be in one feed (error otherwise) | [M] |
| feeds with all labels equal | YetiRank, PairLogit, LambdaMart: no pairs, zero gradient [D]; QuerySoftMax: derivatives set to 0 [D]; QueryRMSE: pulls the feed's scores together [D]. A train set with 25% all-negative feeds trains under every loss [M]. A constant label over the whole set: "All train targets are equal" [M] | |
| label ties | pairs with equal labels are skipped [D] | |
| prediction ties in the metric | `CompareDocs`: lower label first = pessimistic, the project's rule [D `catboost/libs/metrics/doc_comparator.h`]; equal to `src/metrics.ndcg_per_feed` on tied scores | [M] |
| NDCG of a feed with no positive | CatBoost returns 1 [M]; so does `src/metrics.ndcg_per_feed` (`out = np.ones(n)` where IDCG = 0, [D] `src/metrics.py:23`). No such feed exists in train or val (`deep_gbdt_ranking.md` §6; val re-counted: 0 of 14,423 [M]) | [M] |
| group weights on the EVAL pool | the built-in NDCG becomes a weighted mean (0.2756 vs 0.2810); add `use_weights=false` to the metric or keep the eval pool unweighted | [M] |
| target statistics of categoricals inside a feed | on CPU the ordered target statistic of a row uses earlier rows of the SAME feed; `ctr_history_unit="Group"` is rejected on CPU. Section 4, trap 3 | [M] |

### 2c. Label forms (does the fit run, and what does the label mean)

| Objective | binary | graded | probability in [0,1] | any real (negatives) | Meaning of a real-valued label |
|---|---|---|---|---|---|
| `YetiRank` Classic | PASS | PASS | PASS | PASS | pair weight ∝ label gap; shift-invariant inside a feed (raw and centred logit: same NDCG, rank agreement and prediction sd to every printed digit). Label × 10 moves the fit no more than a seed does [M] |
| `YetiRank:mode=NDCG` | PASS | PASS | PASS | PASS (ran; gain = label, so negative gains are of doubtful meaning: feed non-negative labels) | gain = label; scale-invariant |
| `YetiRank:mode=MRR`, `LambdaMart:metric=MRR` | PASS | PASS (binarised at > 0) | **FAIL** if all labels > 0 | runs, binarised at 0 | none: hard labels only |
| `LambdaMart` (NDCG) | PASS | PASS | PASS | PASS | gain = label |
| `StochasticRank:metric=NDCG` / `DCG` | PASS | PASS | PASS | PASS | gain = label |
| `QuerySoftMax` | PASS | PASS | PASS | **FAIL** "Min target less than 0" | soft target of a listwise cross-entropy; Σ labels = weight of the feed (labels × 3 changes the fit [M]). With `softmax(teacher)` labels the train KL to the teacher fell 0.613 → 0.356 in 150 trees [M] |
| `QueryRMSE` | PASS | PASS | PASS | PASS | regression target up to a per-feed constant |
| `PairLogit` | PASS | PASS | PASS | PASS | order only |
| `CrossEntropy` | PASS | n/a | PASS | **FAIL** "Min target less than 0" | pointwise probability |
| `Logloss` | PASS | n/a | **FAIL** without `target_border` ("Target with classes must contain only 2 unique values") | n/a | `target_border=b` binarises at > b |
| `RMSE`, `Huber`, `MAE`, `LogCosh` | PASS | PASS | PASS | PASS | regression target |

`target_border` also acts on ranking losses: on graded labels it changes YetiRank and QuerySoftMax [M].

### 2d. Tree structure and regularisation

Sources [D]: https://catboost.ai/docs/en/references/training-parameters/common ;
https://catboost.ai/docs/en/concepts/parameter-tuning ; https://catboost.ai/docs/en/concepts/algorithm-score-functions ;
`.venv/lib/python3.12/site-packages/catboost/core.py` lines 4845 to 5301 (docstring).

| Parameter | What it does [D] | Resolved default on 1.2.10 CPU [M] | Support and restrictions | Used | Relevance |
|---|---|---|---|---|---|
| `grow_policy` | `SymmetricTree` (one split per level), `Depthwise`, `Lossguide` (leaf-wise) | `SymmetricTree` | CPU: non-symmetric trees PASS with YetiRank (all modes), QuerySoftMax, QueryRMSE, LambdaMart, StochasticRank, PairLogit; FAIL with YetiRankPairwise / PairLogitPairwise ("Pairwise mode is not supported for nonsymmetric trees on CPU"). Non-symmetric trees exclude Ordered boosting, monotone constraints and, for Lossguide, `PerTreeLevel` sampling [M] | no | lever 2 |
| `depth` (`max_depth`) | tree depth | 6, also for Lossguide (docs: 16) | CPU up to 16 ("Maximum tree depth is 16" at 17); GPU up to 8 for pairwise modes incl. YetiRank [D] | 8 (cb26, W25), 7 (11-11) | lever 2. Cost: depth 10 = 1.6x depth 8; depth 6 = depth 8 |
| `min_data_in_leaf` (`min_child_samples`) | minimum rows per leaf | 1 | Depthwise / Lossguide only; NOEFFECT with symmetric trees [M] | no | needed with lever 2 |
| `max_leaves` (`num_leaves`) | leaves per tree | 31 (Lossguide) | Lossguide only; docs advise against more than 64 for speed; 255 costs 1.3x to 1.7x here [M] | no | lever 2 |
| `border_count` (`max_bin`) | borders per float feature | 254 (32 for the two `...Pairwise` losses) | 1 to 65535 on CPU; docs: little speed effect on CPU | 254 | keep |
| `feature_border_type` | border placement | `GreedyLogSum` | `Median`, `Uniform`, `UniformAndQuantiles`, `MaxLogSum`, `MinEntropy` all train [M] | no | low |
| `per_float_feature_quantization` | more borders for 1 or 2 "golden" features, e.g. `["0:border_count=1024"]` | none | PASS [M] | no | low: candidates are the NN-derived inputs (`pre2_fe_z`) |
| `l2_leaf_reg` (`reg_lambda`) | L2 on leaf values (added to the hessian sum of the leaf) | **1e-20 for YetiRank, YetiRankPairwise, LambdaMart**; 3 for QuerySoftMax, QueryRMSE, PairLogit, StochasticRank; 5 for PairLogitPairwise (docs: 3.0 for all) | setting 0 on YetiRank is NOEFFECT, i.e. the default really is 0 [M] | 5 (cb26, W25), 3 (11-11) | lever 6 |
| `random_strength` | Gaussian noise on split scores, sd ∝ gradient norm, fading as the model grows [D `greedy_tensor_search.cpp`] | 1 (0 for `...Pairwise`) | CPU only; docs call it unsupported for `...Pairwise`, yet it changes their fit [M] | default | lever 6 |
| `random_score_type` | `NormalWithModelSizeDecrease` or `Gumbel` | the former | PASS [M] | no | low |
| `model_shrink_rate`, `model_shrink_mode` | shrinks the whole model each iteration: `Constant` = 1 − rate·lr, `Decreasing` = 1 − rate/iteration | 0, `Constant` | CPU only; PASS on all main losses [M] | no | lever 6 |
| `leaf_estimation_method` | `Newton` / `Gradient` / `Exact` | YetiRank Newton; QuerySoftMax Gradient; QueryRMSE, PairLogit, LambdaMart Newton; StochasticRank Gradient | YetiRank: changing it is prohibited; StochasticRank: Newton unsupported; **QuerySoftMax: Newton allowed and far faster-learning (prediction sd 1.11 vs 0.32 after 40 trees)**; **LambdaMart + Gradient ranks backwards** [M] | default | lever 7; trap 2 |
| `leaf_estimation_iterations` | Newton / gradient steps per tree | 1 for every loss tested (docs: 10 for classification) | **YetiRank with 5 steps at its default l2 diverges** ("degenerate solution ... try to increase l2", predictions 1e15); stable with `l2_leaf_reg` ≥ 1 [M] | default | trap 5 |
| `leaf_estimation_backtracking` | step halving | `No` for YetiRank, LambdaMart, StochasticRank; `AnyImprovement` for the others (docs: AnyImprovement) | YetiRank: backtracking unsupported; `Armijo` is GPU only [M] | default | low |
| `boosting_type` | `Plain`, or `Ordered`: the gradient of a row comes from a model fitted without that row (permutation prefixes), the cure for prediction shift on small data | `Plain` | Ordered PASS with YetiRank, QuerySoftMax, QueryRMSE, LambdaMart, StochasticRank, PairLogit; FAIL with `...Pairwise` and with non-symmetric trees [M]. Cost 4.2x [M] | default | low: about 20M rows, and docs position Ordered for small data |
| `langevin`, `diffusion_temperature` | SGLB: Gaussian noise on the gradients plus shrinkage (the convergence device of StochasticRank) | False, 10000 | CPU only; PASS [M]; no extra cost (1.04x) | no | low |
| `posterior_sampling` | preset: langevin + shrink for uncertainty | False | PASS [M] | no | none |
| `score_function` | split score | `Cosine`, also under Lossguide (docs: NewtonL2 for Lossguide) | **CPU: only `Cosine` and `L2`**; "Only Cosine and L2 score functions are supported for CPU" [M] | default | low: `L2` is the only alternative |
| `model_size_reg` | penalises splits on high-cardinality categorical statistics [D `GetCatFeatureWeight`] | 0.5 | matters only with categorical features | default | docs_feat track |
| `boost_from_average` | start from the best constant | False | rejected for ranking losses [M] | no | none |
| `nan_mode` | NaN as `Min` / `Max` / `Forbidden` | `Min` | PASS [M] | default | keep |

### 2e. Sampling

Sources [D]: https://catboost.ai/docs/en/concepts/algorithm-main-stages_bootstrap-options ;
`.../catboost/private/libs/algo/tensor_search_helpers.cpp` ; `.../catboost/private/libs/options/bootstrap_options.cpp`.

| Parameter | What it does [D] | Resolved default [M] | Restrictions [M] | Used | Relevance |
|---|---|---|---|---|---|
| `bootstrap_type` | `Bayesian` (weight = (−log U)^temperature), `Bernoulli` (keep a row with probability `subsample`), `MVS` (importance sampling by gradient size), `Poisson`, `No` | **CPU: `MVS`, subsample 0.8** (GPU: Bayesian [D]); `Bernoulli` 0.5 for the `...Pairwise` losses | `Poisson`: "poisson bootstrap is not supported on CPU". For ranking losses the gradients come from whole feeds; the sample weights then multiply the per-row derivatives used to score splits [D] | GPU default (Bayesian) in cb26 / W25; Bernoulli 0.8 in cb26 E1 / E2 | a CPU "reproduction" of the record silently changes Bayesian → MVS unless `bootstrap_type="Bayesian"` is set |
| `subsample` | rate for Bernoulli / MVS | 0.8 (MVS) | not with Bayesian | 0.8 (E1 / E2) | lever 6. No speed gain on CPU: Bernoulli 0.5 costs 1.04x, MVS 0.3 0.89x [M] |
| `bagging_temperature` | Bayesian only; 0 = no bagging | 1 | PASS | default | low |
| `mvs_reg` | MVS regulariser | automatic | PASS | no | low |
| `sampling_unit` | `Object`, or `Group` = one draw per feed | `Object` | `Group` makes Bayesian the default; **`Group` + `MVS` fails**: "MVS bootstrap supports per object sampling only" | no | lever 6: `Group` + Bernoulli is the feed-level bagging |
| `sampling_frequency` | `PerTree` or `PerTreeLevel` | **`PerTree`** (docs page: `PerTreeLevel`; the package docstring says PerTree) | `PerTreeLevel` fails with Lossguide | default | low |
| `rsm` (`colsample_bylevel`) | share of features tried at each split | 1 | CPU for all losses; crashes YetiRankPairwise ("LAPACK dposv_ failed with status -5"). No speed gain: 1.06x [M] | no | lever 6 |

### 2f. Training control

| Item | Behaviour on 1.2.10 CPU | Mark |
|---|---|---|
| `eval_metric` forms | `"NDCG"` is stored as `"NDCG:type=Base"`; `"NDCG:top=10;type=Exp;denominator=Position"`, `"MRR"`, `"MAP:top=10"`, `"ERR"`, `"PFound:top=10;decay=0.9"`, `"QueryAUC:type=Ranking"`, `"AverageGain:top=1"`, `"PrecisionAt:top=5"`, `"RecallAt:top=5"`, `"PairAccuracy"`, `"FilteredDCG"`, `"PairLogit"`, `"QuerySoftMax"`, `"QueryRMSE"`, `"Logloss"` all PASS with a YetiRank objective. `YetiRank`, `LambdaMart`, `StochasticRank` are not metrics ("Unsupported metric"); `RMSE` is "incompatible with loss [YetiRank]" | [M] |
| `NDCG:type=Base` vs the project metric | identical on continuous scores, on scores with ties and on all-tied scores (difference 6e-17); identical to `src/metrics.ndcg_per_feed` (0.285803 three ways); identical on staged predictions during training. **On the record's own val predictions** (`nn_x/cb26/data/e3_xm_yeti_val.npy`, 14,423 feeds, evaluation only): project metric 0.38201327 (per-feed SE 0.0020), CatBoost `NDCG:type=Base` 0.38201327, record 0.38201327; after rounding the scores to one decimal (forced ties) 0.36726381 both ways | [M] |
| `hints=skip_train~false` | NDCG, DCG, PFound, AUC, YetiRank metrics are not computed on train by default; the hint turns it on | [D] + [M] |
| the objective's own default metric | YetiRank → `PFound`; LambdaMart, StochasticRank → `NDCG`; early stopping without `eval_metric` uses it | [M] |
| `custom_metric` | a list of extra metrics, reported only | [M] |
| Python custom eval metric | class with `evaluate(approxes, target, weight)`, `is_max_optimal`, `get_final_error`; receives the WHOLE pool in pool order, so a closure over feed offsets gives any per-feed metric (my per-feed NDCG matched the built-in to 9e-16). Slow path, not needed since the built-in is exact | [M] |
| Python custom objective | `calc_ders_range` is called in blocks of at most 1,000 rows without row indices, and `CatBoostRanker` rejects an object: **a listwise custom loss cannot be written through this interface** | [M] |
| early stopping | `od_type="Iter", od_wait=n` and `early_stopping_rounds=n` stop (17 of 400 iterations in the test); `od_type="IncToDec", od_pval=1e-2` ran all 400; `od_pval` with `Iter` is an error | [M] |
| `use_best_model` | default True when an eval set is given; trims to the best iteration; `best_model_min_trees` does not prevent the stop | [M] |
| `get_best_score()` | returns `{}` for YetiRank; read `get_best_iteration()` and `get_evals_result()` | [M] |
| several eval sets | allowed on CPU (`validation_0`, `validation_1`); docs: the last one drives the detector [D]; GPU allows one [D] | [M] |
| snapshots | `save_snapshot=True, snapshot_file=..., snapshot_interval=sec` resumes to the same model as an uninterrupted run; needs `allow_writing_files=True` (error otherwise) | [M] |
| `init_model` | CPU only [D]. PASS: YetiRank → YetiRank, QuerySoftMax → YetiRank, QueryRMSE → `YetiRank:mode=NDCG`, YetiRank → LambdaMart, RMSE regressor → YetiRank, other lr / depth / `border_count`, Plain → Ordered, Lossguide → Lossguide, with a categorical feature. The first trees reproduce the initial model exactly. FAIL: symmetric → non-symmetric; with an embedding feature ("Models summation is not supported for models with embedding features"); together with a baseline ("Specifying baseline for training continuation is not supported") | [M] |
| `baseline` | one starting score per row. PASS with every ranking loss and with Logloss / CrossEntropy / RMSE. The eval pool must carry one too ("Baseline for test is not provided"). The eval metric includes it (iteration 0 = the baseline's NDCG). **`predict` on a pool WITH a baseline returns baseline + trees; on a pool without it, the trees only.** The scale matters (YetiRank: totals 0.740 / 0.750 / 0.771 / 0.770 at scales 0.1 / 0.7 / 3 / 10, trees exploding at 10). With QueryRMSE / RMSE the baseline must be in label units | [M] |
| `sum_models(models, weights)` | exact weighted sum of raw scores (error 4e-15), also with categorical statistics, Lossguide models and different border sets; FAIL with embedding features. Returns a plain `CatBoost` object | [M] |
| staged prediction | `staged_predict(pool, eval_period=k)`, `predict(pool, ntree_end=k)`, `model.shrink(k)` agree; `eval_metrics(pool, ["NDCG:type=Base"], eval_period=k)` gives the curve without re-predicting | [M] |
| threads | `Pool(...)`, `predict`, `staged_predict`, `eval_metrics`, `utils.eval_metric` all default to `thread_count=-1` = all cores [D core.py]: pass `thread_count=12` to each. `Pool.quantize()` has no thread argument; I did not use it | [D] |

### 2g. Constraints and feature control

| Parameter | Behaviour | Mark |
|---|---|---|
| `monotone_constraints` | CPU only [D]. PASS with YetiRank (both modes), QuerySoftMax, QueryRMSE, LambdaMart, StochasticRank, PairLogit on symmetric trees, also with Ordered boosting; enforced (raising the feature never lowered the score). FAIL with `...Pairwise` losses and with non-symmetric trees | [M] |
| `feature_weights` | multiplies the split gain of a feature; PASS, also with Lossguide and YetiRankPairwise | [M] |
| `first_feature_use_penalties`, `per_object_feature_penalties`, `penalties_coefficient` | penalties for the first use of a feature; PASS (the coefficient test saturated: no information) | [M] |
| `ignored_features` | PASS | [M] |
| interaction constraints | **do not exist** in CatBoost: not in the constructor, `TypeError` | [M] + [D core.py:6353] |
| `fixed_binary_splits` | GPU only [D] | not tested |

### 2h. Release notes 1.0 to 1.2.10 that touch ranking

Source [D]: https://raw.githubusercontent.com/catboost/catboost/v1.2.10/RELEASE.md (first 100,000 characters = 1.2.10 back
past 1.0).

| Version | Entry |
|---|---|
| 1.0.1 | group weight is used for generated pairs in pairwise losses |
| 1.0.4 | StochasticRank for FilteredDCG |
| 1.0.5 | fix of group weights in metric calculation |
| 1.0.6 | fix of nan / inf split scores with YetiRankPairwise; fix of nan / inf feature strengths in PairLogit on CPU |
| 1.1 | text and embedding features for regression and ranking; `init_model` and model summation for non-symmetric trees; 2x faster DCG / NDCG for groups of 50 or more; fix of a 2x slowdown of ranking losses on CPU; fix of the QueryRMSE gradient on GPU |
| 1.2 | fix of the `CatBoostRanker.score` computation; fix of the sum of models with categorical statistics; fix of CUDA error 700 in pairwise ranking |
| **1.2.1** | **`mode` parameter of YetiRank / YetiRankPairwise (YetiLoss); LambdaMart metrics MRR, ERR, MAP; StochasticRank metrics ERR, MRR** |
| 1.2.3 | hash collisions of group ids with many groups avoided; `pairs_weight` as numpy array fixed |
| 1.2.5 | training with embedding features on GPU fixed |
| 1.2.6 | GroupQuantile; QueryCrossEntropy speed-up on GPU; custom objective on GPU |
| 1.2.9 | polars input; Lossguide optimised on CPU |
| 1.2.10 | nothing on training |

No release note mentions the traps of section 4.

## 3. Verification record

All checks: `nn_x/w12/cb/docs_rank/out/verify_all.tsv` (632 rows: section, status, check, detail with the exact error text,
seconds). Data: 600 train feeds / 44,411 rows, 200 eval feeds / 15,389 rows, 10 to 300 candidates per feed, 1.73 positives
per feed, 12 numeric features, optional categorical and 16-d embedding, a noisy teacher score. Levels: random scores
0.281, teacher 0.775, trained models 0.66 to 0.74. 40 trees, depth 6, lr 0.2, 4 threads. NOEFFECT means bit-identical
predictions to the reference run; the test is valid because every loss is deterministic for a fixed seed and thread
count (8 of 8 losses: same seed twice identical, 2 threads vs 4 identical [M]). PASS means "ran, and where a reference
exists the model changed"; it is not a statement of quality, and the single-seed NDCG values in the TSV are not
comparisons between arms.

| Section | PASS | FAIL (rejections) | NOEFFECT | other |
|---|---:|---:|---:|---|
| objectives and their parameters | 72 | 9 | 16 | 3 errors of my test code |
| resolved defaults | 15 | 0 | 0 | |
| label forms | 64 | 9 | 0 | 8 errors of my test code |
| groups, weights, pairs, metric | 26 | 7 | 5 | |
| tree structure | 75 | 20 | 29 | |
| sampling | 76 | 16 | 10 | |
| training control | 34 | 8 | 0 | 5 errors of my test code |
| baseline, init_model, sum_models | 30 | 5 | 0 | |
| constraints | 16 | 4 | 1 | |
| embedding features with ranking losses | 12 | 0 | 0 | 3 errors of my test code |
| follow-up | 26 | 0 | 2 | 1 UNSTABLE, 1 wrong direction |
| pointwise losses (re-run of the checks my test code had broken, plus extras) | 21 | 2 | 0 | |
| **total** | **467** | **80** | **63** | 19 + 1 + 1 (+ 1 info row) |

The three questions the task named:

| Question | Answer |
|---|---|
| Does a ranking loss accept `embedding_features`? | PASS for YetiRank, `YetiRank:mode=NDCG`, YetiRankPairwise, QuerySoftMax, QueryRMSE, LambdaMart, StochasticRank, PairLogit with binary labels, and for QuerySoftMax / QueryRMSE / YetiRank with soft labels; also Logloss, CrossEntropy [M]. Resolved processing: `{"default": ["LDA", "KNN"]}`. Cost: 23 s against 1.1 s for the same 44K-row fit at 4 threads. Such models cannot be continued (`init_model`) or summed (`sum_models`) |
| Does it accept real-valued (soft) labels? | section 2c |
| Do `init_model` and `baseline` work with it? | section 2f |

Exact error texts of the rejections that matter (all [M]):

- `QueryCrossEntropy`: "catboost/private/libs/options/catboost_options.cpp:661: QueryCrossEntropy loss function is not supported for CPU learning"
- `StochasticRank` without `metric`: "catboost_options.cpp:907: StochasticRank requires metric param"
- `YetiRank:mode=PFound`: "yetirank_helpers.cpp:42: Unknown weights mode PFound"
- QuerySoftMax with a negative label: "catboost/libs/metrics/metric.cpp:7021: Min target less than 0: -5.19582"
- `mode=MRR` with all-positive soft labels: "tensor_search_helpers.cpp:565: Too few sampling units (subsample=0.8, bootstrap_type=MVS): please increase sampling rate or disable sampling"
- non-symmetric trees with a `...Pairwise` loss: "catboost_options.cpp:764: Pairwise mode is not supported for nonsymmetric trees on CPU."
- Ordered boosting: "catboost_options.cpp:811: Boosting type should be plain for YetiRankPairwise"; "catboost_options.cpp:759: Ordered boosting is not supported for nonsymmetric trees."
- YetiRank leaves: "catboost_options.cpp:308: At the moment, in the YetiRank mode, changing the leaf_estimation_method parameter is prohibited."; "catboost_options.cpp:670: Backtracking is not supported for yetiRank and YetiRankPairwise loss functions"
- score function: "oblivious_tree_options.cpp:146: Only Cosine and L2 score functions are supported for CPU."
- sampling: "bootstrap_options.cpp:38: MVS bootstrap supports per object sampling only."; "bootstrap_options.cpp:29: Error: poisson bootstrap is not supported on CPU"; "catboost_options.cpp:770: PerTreeLevel sampling is not supported for Lossguide grow policy."
- monotone constraints: "catboost_options.cpp:718: Monotone constraints is unsupported for pairwise loss functions."; "monotonic_constraint_utils.h:42: Internal CatBoost Error ... Monotone constraints are unsupported for non-symmetric trees yet"
- baseline: "preprocess.cpp:30: Baseline for test is not provided"; "learn_context.cpp:557: Specifying baseline for training continuation is not supported"
- model sums: "catboost/libs/model/model.cpp:1901: Models summation is not supported for models with embedding features"; "model.cpp:1885: Summation of symmetric and non-symmetric models is not supported [for now]"
- snapshots: "output_file_options.cpp:326: allow_writing_files is set to False, and save_snapshot is set to True."
- target statistics: "json_helper.h:185: Error: change of option ctr_history_unit is unimplemented for task type CPU and was not default in previous run"

## 4. Traps on the installed version (all [M])

1. **Unknown parameters of `YetiRank` are silently ignored.** `"YetiRank:bogus=1"` trains and is bit-identical to
   `"YetiRank"`. A typo (`neighbors`, `Decay`) costs a whole run and shows nothing. QuerySoftMax does validate ("Valid
   parameters are beta, hints, lambda, proba_border, use_weights"). Rule: after a run, compare with the reference
   predictions; identical = the parameter did nothing. Mode-specific parameters behave the same way: `top`,
   `num_neighbors`, `dcg_type` do nothing in Classic; `decay` does nothing in `mode=NDCG`.
2. **`LambdaMart` + `leaf_estimation_method="Gradient"` ranks backwards**: NDCG 0.211 against 0.281 for random scores and
   0.716 for the default Newton; the negated model scores 0.653. Keep Newton.
3. **A categorical that is constant inside a feed leaks same-feed labels on CPU.** Synthetic test with labels independent
   of all features and one categorical = feed id (`Borders` statistic only): NDCG on the train rows DURING training 0.514,
   the finished model on the SAME rows 0.368, importance of the categorical 13.8 of 100. With a "user" id shared by about 3
   feeds: 0.488 / 0.461, importance 6.2. Without the categorical the two numbers agree (0.4985 / 0.4985). The option that
   would prevent it, `ctr_history_unit="Group"`, is rejected on CPU and is the GPU default for groupwise losses [D
   `catboost_options.cpp`]. Consequence for a native-statistics arm: user-level or feed-level ids, and combinations that
   contain them, will spend trees on a pattern that does not exist at inference. `category_id` (54 values, about 365K rows
   each) is not at risk [I]. Files: `verify_ctr_group2.py`, `out/verify_ctr_group2.json`.
4. **`QueryCrossEntropy` is GPU only**; the docs' "alpha" lever is closed for this track.
5. **YetiRank with several leaf steps diverges at its default `l2_leaf_reg`** (which is 0): with
   `leaf_estimation_iterations=5` training stopped at tree 15 of 40, predictions about 1e15 (2 steps survived). Stable with
   `l2_leaf_reg` ≥ 1.
6. **`mode=MRR` / `MAP` need hard labels**; QuerySoftMax and CrossEntropy need labels ≥ 0 (no raw logits).
7. **`predict` adds the baseline only if the pool carries it**; the eval pool must carry one during training.
8. **A CPU re-run of the record is not the record**: bootstrap default MVS 0.8 instead of GPU Bayesian; target statistic of
   `category_id` by row instead of by group; early stopping must leave val. Compare CPU arms only with a CPU baseline.
9. **`get_best_score()` is empty for YetiRank**, `od_type="IncToDec"` did not stop, `YetiRankPairwise` + `rsm` crashes,
   `LambdaMart:metric=DCG` is the same model as `metric=NDCG`.

Documentation against the installed version:

| Item | Docs say | Installed 1.2.10 CPU does [M] |
|---|---|---|
| `sampling_frequency` default | PerTreeLevel (common-parameters page) | PerTree |
| `depth` default with Lossguide | 16 | 6 |
| `score_function` with Lossguide | NewtonL2 | Cosine; Newton scores rejected on CPU |
| `l2_leaf_reg` default | 3.0 | 1e-20 for YetiRank, YetiRankPairwise, LambdaMart |
| `leaf_estimation_backtracking` default | AnyImprovement | No for YetiRank, LambdaMart, StochasticRank |
| `leaf_estimation_iterations` default | objective-dependent, 10 for classification | 1 for every loss tested |
| `dcg_denominator` of YetiRank | "Position" as my fetch of the page read it | LogPosition |
| QuerySoftMax parameters | `beta` | `beta` and `lambda` |
| StochasticRank parameters | `metric`, `num_estimations`, `mu` | plus `sigma`, `nu`, `lambda`, `top` |
| `random_strength` with `...Pairwise` losses | not supported | accepted, changes the fit |
| losses of `CatBoostRanker` | docstring list without LambdaMart | LambdaMart accepted |

## 5. Cost on CPU

Synthetic data at the shape of the exploration training set: feed sizes from the percentiles 54 / 80 / 128 / 214 / 283
(mean 148.9), 1.70 positives per feed, 177 float features, no categorical, depth 8, lr 0.08, `l2_leaf_reg=5`, 254
borders, 12 threads, machine load 83 to 91 during the runs. Seconds per iteration = median over iterations 4 to 16 (4
to 10 at the larger size), read from a callback [M]. Files `out/bench_10000.json`, `out/bench_40000.json`.

| Configuration | 10,000 feeds, 1.49M rows: s / iteration | relative | 40,000 feeds, 5.97M rows: s / iteration |
|---|---:|---:|---:|
| YetiRank Classic (the record's settings) | 0.387 | 1.00 | 1.610 |
| `YetiRank:mode=NDCG` | 0.515 | 1.33 | 2.221 |
| `YetiRank:mode=NDCG;num_neighbors=3` | 0.699 | 1.81 | 3.168 |
| `YetiRank:mode=MRR` | 0.402 | 1.04 | |
| `YetiRank:permutations=30` | 0.650 | 1.68 | |
| `LambdaMart` | 0.145 | 0.37 | |
| `LambdaMart:norm=false` | 0.138 | 0.36 | 0.549 |
| `StochasticRank:metric=NDCG` | 5.351 | 13.8 | |
| `QuerySoftMax` | 1.036 | 2.68 | 1.669 |
| `QueryRMSE` (teacher label) | 0.113 | 0.29 | 0.502 |
| `RMSE` (teacher label) | 0.103 | 0.27 | |
| `PairLogit` (all positive × negative pairs) | 1.001 | 2.59 | |
| `Logloss` | 0.109 | 0.28 | |
| YetiRank, depth 6 | 0.386 | 1.00 | |
| YetiRank, depth 10 | 0.623 | 1.61 | |
| YetiRank, Depthwise, depth 8, `min_data_in_leaf=100` | 0.651 | 1.68 | |
| YetiRank, Lossguide, `max_leaves=255` | 0.646 | 1.67 | 2.173 |
| YetiRank, Lossguide, `max_leaves=63` | 0.478 | 1.24 | |
| QueryRMSE, Lossguide, `max_leaves=255` | 0.646 | 1.67 | |
| YetiRank, `bootstrap_type="No"` | 0.420 | 1.08 | |
| YetiRank, Bernoulli 0.5 by row | 0.403 | 1.04 | |
| YetiRank, `sampling_unit="Group"`, Bernoulli 0.5 | 0.380 | 0.98 | |
| YetiRank, MVS 0.3 | 0.346 | 0.89 | |
| YetiRank, `rsm=0.3` | 0.410 | 1.06 | |
| YetiRank, `boosting_type="Ordered"` | 1.621 | 4.19 | |
| YetiRank, `langevin=True` | 0.404 | 1.04 | |
| YetiRank + eval set of 1.42M rows, `NDCG:type=Base` every iteration | 0.524 | +0.14 s per evaluation | |
| same, `metric_period=50` | 0.407 | | |
| second run, its own reference: YetiRank Classic | 0.308 | 1.00 | |
| second run: `YetiRank:permutations=3` | 0.248 | 0.80 | |
| second run: `QuerySoftMax`, `leaf_estimation_method="Newton"` | 0.190 | 0.62 | |
| second run: YetiRank Classic on a real-valued label | 0.903 | 2.93 | |
| second run: `LambdaMart:norm=false` on a real-valued label | 0.830 | 2.69 | |

Reading:

- Cost is linear in rows: 4.01x the rows cost 4.16x (YetiRank), 3.98x (LambdaMart), 4.44x (QueryRMSE) [M]. QuerySoftMax at
  its defaults is the exception (1.61x); its cost sits in the default leaf step (Gradient with backtracking): with Newton
  leaves the same loss costs 0.62x [M]. I did not look further into the cause.
- The same configuration timed twice gave 0.387 and 0.308 s (machine load): read ratios inside one run, to about ±20%.
- **Pairwise losses get dearer on soft labels**: with a real-valued label every pair of a feed has a label gap, so
  YetiRank costs 2.9x and LambdaMart 2.7x (7.5x its own cost on clicks) [M]. QueryRMSE and QuerySoftMax are O(M) per feed
  and do not change.
- YetiRank's cost is mostly its own gradient step (10 sampled rankings per feed), not the tree: sampling and `rsm` save
  nothing, `permutations` and the loss choice do [I from the table].
- **Absolute anchor on the real data (sibling infra track, `reports/CATBOOST_1115/infra.md` §2.3 [D]):** YetiRank, depth 8,
  19.74M rows × 177 features, `category_id` categorical, 12 threads: 8.31 s per iteration (9.3 s in its reproduction run
  with an evaluation at every iteration). My synthetic table extrapolates to 5.3 s for that row count [I], so the real
  data cost about 1.6x my synthetic data. The RATIOS agree where the two tables overlap: RMSE 0.28x there, 0.27x here;
  `permutations=3` 0.78x / 0.80x; depth 10 1.55x / 1.61x; QuerySoftMax at defaults 1.73x there against 2.7x (1.5M rows)
  and 1.04x (6M rows) here; `rsm=0.3` 0.88x / 1.06x.
- With that anchor, per iteration on the full exploration set [I]: `QueryRMSE` about 2.4 s, `LambdaMart` on clicks about
  3.1 s, `QuerySoftMax` + Newton about 5 s, `YetiRank:mode=NDCG` about 11 s, `Lossguide` 255 leaves 11 to 14 s, YetiRank
  on a real-valued label about 24 s, `Ordered` about 35 s, `StochasticRank` about 115 s.
- The F18 record says 60K teacher-labelled feeds are enough: at 60K feeds a `QueryRMSE` student costs about 1.1 s per
  iteration by the same scaling [I].

## 6. Direction check on a synthetic analogue (not evidence about the real data)

Purpose: the direction of each lever where the mechanics are known, on data shaped like the problem. Script
`synth_levers.py`. Data [M]: 3,000 train feeds / 443,073 rows and 3,000 eval feeds / 449,776 rows; feed sizes as in section 5;
1.71 clicks per feed drawn without replacement from the within-feed softmax of a true logit; the true logit depends on 20
visible features (the student sees them) and on 6 hidden ones (only the teacher sees them), so the student is
representation-limited by construction. Levels: random scores 0.2275, teacher 0.4665, true logit 0.4666. Every arm: seeds
0 / 1 / 2, 200 trees, depth 6, lr 0.1, 12 threads. Delta = per-feed difference of the seed-mean NDCG to the reference
arm, SE over the 3,000 eval feeds.

| Arm (3 seeds each) | eval NDCG | sd over seeds | paired delta to the reference ± SE (3,000 feeds) | z |
|---|---:|---:|---:|---:|
| H0 YetiRank (Classic, decay 0.85) [reference] | 0.3939 | 0.0025 | reference |  |
| H1 YetiRank:decay=0.95 | 0.3933 | 0.0009 | -0.0006 ± 0.0012 | -0.5 |
| H2 YetiRank:decay=0.99 | 0.3923 | 0.0002 | -0.0016 ± 0.0013 | -1.2 |
| H3 YetiRank:mode=NDCG | 0.3894 | 0.0011 | -0.0044 ± 0.0013 | -3.5 |
| H4 YetiRank:mode=NDCG;num_neighbors=3 | 0.3924 | 0.0018 | -0.0014 ± 0.0012 | -1.2 |
| H5 YetiRank:mode=MRR | 0.3908 | 0.0028 | -0.0030 ± 0.0013 | -2.4 |
| H6 YetiRank:permutations=30 | 0.3942 | 0.0011 | +0.0003 ± 0.0011 | +0.3 |
| H7 LambdaMart | 0.3937 | 0.0001 | -0.0002 ± 0.0014 | -0.1 |
| H8 LambdaMart:norm=false | 0.3886 | 0.0012 | -0.0053 ± 0.0018 | -2.9 |
| H9 QuerySoftMax (Gradient leaves = default) | 0.3981 | 0.0016 | +0.0042 ± 0.0014 | +2.9 |
| H10 QuerySoftMax + Newton leaves | 0.3991 | 0.0024 | +0.0052 ± 0.0014 | +3.8 |
| H11 Logloss (pointwise) | 0.3970 | 0.0007 | +0.0032 ± 0.0014 | +2.3 |
| H12 YetiRank Lossguide max_leaves 64 | 0.3882 | 0.0022 | -0.0057 ± 0.0017 | -3.3 |
| H13 YetiRank l2_leaf_reg=30 | 0.3534 | 0.0004 | -0.0404 ± 0.0030 | -13.3 |
| H14 YetiRank rsm=0.5 | 0.3925 | 0.0023 | -0.0014 ± 0.0012 | -1.2 |
| H15 YetiRank sampling_unit=Group Bernoulli 0.5 | 0.3928 | 0.0006 | -0.0011 ± 0.0011 | -1.0 |
| S1 QueryRMSE <- teacher logit | 0.4047 | 0.0002 | +0.0108 ± 0.0017 | +6.5 |
| S2 RMSE <- teacher logit | 0.4055 | 0.0003 | +0.0116 ± 0.0016 | +7.0 |
| S3 QuerySoftMax + Newton <- softmax(teacher) | 0.4047 | 0.0004 | +0.0108 ± 0.0016 | +6.7 |
| S4 QuerySoftMax + Newton <- 0.5 y/npos + 0.5 softmax(teacher) | 0.4034 | 0.0015 | +0.0095 ± 0.0015 | +6.5 |
| S5 YetiRank <- teacher logit | 0.4048 | 0.0004 | +0.0110 ± 0.0016 | +6.7 |
| S6 YetiRank:mode=NDCG <- softmax(teacher) | 0.4050 | 0.0010 | +0.0112 ± 0.0015 | +7.2 |
| S7 LambdaMart:norm=false <- softmax(teacher) | 0.4030 | 0.0008 | +0.0092 ± 0.0016 | +5.7 |
| S8 QueryRMSE <- teacher logit + 1.0 y | 0.4056 | 0.0004 | +0.0117 ± 0.0017 | +7.0 |
| S9 QueryRMSE <- teacher (150 trees) then init_model -> YetiRank on clicks (50 trees, lr 0.03) | 0.4043 | 0.0005 | +0.0104 ± 0.0015 | +6.9 |

| Round 2: tree shape and regularisation of the student (3 seeds each) | eval NDCG | sd over seeds | paired delta to T0 ± SE (3,000 feeds) | z |
|---|---:|---:|---:|---:|
| T0 QueryRMSE <- teacher logit, symmetric depth 6 [reference of round 2] | 0.4047 | 0.0002 | reference |  |
| T1 QueryRMSE <- teacher logit, symmetric depth 8 | 0.4062 | 0.0002 | +0.0015 ± 0.0007 | +2.1 |
| T2 QueryRMSE <- teacher logit, symmetric depth 10 | 0.4053 | 0.0012 | +0.0007 ± 0.0007 | +0.9 |
| T3 QueryRMSE <- teacher logit, Lossguide max_leaves 64 | 0.4025 | 0.0011 | -0.0022 ± 0.0011 | -2.0 |
| T4 QueryRMSE <- teacher logit, Lossguide max_leaves 255 | 0.4024 | 0.0012 | -0.0023 ± 0.0011 | -2.0 |
| T5 QueryRMSE <- teacher logit, Depthwise depth 8 min_data_in_leaf 50 | 0.4037 | 0.0003 | -0.0010 ± 0.0009 | -1.1 |
| T6 QueryRMSE <- teacher logit, rsm 0.5 | 0.4036 | 0.0002 | -0.0011 ± 0.0007 | -1.7 |
| T7 QueryRMSE <- teacher logit, l2_leaf_reg 30 | 0.4046 | 0.0004 | -0.0001 ± 0.0005 | -0.2 |
| T8 QuerySoftMax + Newton <- 0.5 y/npos + 0.5 softmax(teacher), beta=2 | 0.4041 | 0.0006 | -0.0006 ± 0.0013 | -0.5 |

| Round 2b | eval NDCG | sd over seeds | paired delta to H0 ± SE (3,000 feeds) | z |
|---|---:|---:|---:|---:|
| H0 YetiRank (Classic, decay 0.85) [reference] | 0.3939 | 0.0025 | reference |  |
| T9 YetiRank l2_leaf_reg=3 (hard clicks) | 0.3931 | 0.0009 | -0.0008 ± 0.0016 | -0.5 |

Readings, valid for the analogue only:

1. **The label form is the one lever with a large effect.** Every teacher-target arm gains +0.009 to +0.012 over hard-label
   YetiRank. The nine soft-label losses span 0.4030 to 0.4056: the loss that carries the teacher target does not matter.
   This agrees with lever 1 and with the record (soft targets worth +0.005 to +0.008 to LightGBM).
2. **Click mixing and the two-stage curriculum add nothing** over the plain teacher target: S8 0.4056, S4 0.4034, S9
   0.4043 against S1 0.4047 and S2 0.4055. This agrees with the record's null for the click mix.
3. **No hard-label YetiRank variant beats the Classic default.** `mode=NDCG` −0.0044 ± 0.0013, `mode=MRR` −0.0030 ± 0.0013
   and `LambdaMart:norm=false` −0.0053 ± 0.0018 are worse; `decay` 0.95 / 0.99 and `permutations=30` are neutral. Lever 4
   gets no support. The source paper reports the same order for NDCG (YetiRank above YetiLoss).
4. QuerySoftMax and Logloss beat YetiRank on hard labels here (+0.003 to +0.005). The analogue draws clicks from a
   within-feed softmax, which is the model QuerySoftMax fits, so this is expected; on the real data the record is ±0.001.
5. **Tree shape of the student:** depth 8 +0.0015 ± 0.0007 over depth 6, depth 10 +0.0007 ± 0.0007, Lossguide with 64 or
   255 leaves −0.0022 / −0.0023 ± 0.0011, Depthwise −0.0010 ± 0.0009. No support for non-symmetric trees. The analogue's
   target is a low-order function of 20 features, so this says little about copying the real teacher.
6. **`l2_leaf_reg` does not transfer between losses.** 30 on YetiRank: −0.0404 ± 0.0030 (under-fit in 200 trees, about 7K
   rows per leaf); 3 on YetiRank: −0.0008 ± 0.0016; 30 on QueryRMSE: −0.0001 ± 0.0005. YetiRank's hessians are small.
7. `rsm=0.5` and feed-level Bernoulli 0.5: about −0.001, inside noise.
8. Not one lever besides the label form moved the result by more than 0.006 in the good direction; most moved it by less
   than 0.002. That is the size to expect from this track's training levers [I].

## 7. Method

1. Read the state on record: `nn_x/w12/cb/PREDECLARED.txt`, `reports/deep_gbdt_ranking.md`, `reports/sol_solvers.md` §2.5,
   `nn_x/cb26/*.py`, `nn_x/cb26/data/e3_xm_yeti.json`, `nn_x/w25/out/results.txt`, `nn_x/w25/train_w25.py`,
   `reports/RECHECK_1114/long_bets.md` §4, `nn_x/w12/recheck/long_bets/gbdt_kd.py`, `reports/COVERAGE_MAP.md` (grep),
   `nn_x/w5/train.py` lines 196 to 202 (the students' KD target), `reports/CATBOOST_1115/infra.md` and
   `nn_x/w12/cb/infra/README.md` (sibling track, read only).
2. Read the installed package: `.venv/lib/python3.12/site-packages/catboost/core.py` (constructor of `CatBoostRanker`
   line 6353, `fit` 6476, `Pool` 608, parameter docstring 4845 to 5301, `Pool.quantize` 1239).
3. Read primary sources through the web tool (it returns summaries, not verbatim pages): the docs pages listed in
   section 9, the source files of the v1.2.10 tag, RELEASE.md, the ranking tutorial notebook, and the paper Lyzhin et al.,
   "Which Tricks Are Important for Learning to Rank?" (arXiv 2204.01500, pages 3 to 9 and 11 to 13 read as images). Queries
   were generic CatBoost questions; nothing of the project was sent.
4. Verified each claim that matters on the installed version with synthetic data (sections 3 to 6). The one use of project
   data: the record's existing val prediction file was scored with CatBoost's metric and with the project evaluator
   (section 2f); nothing was trained on project data and no label entered a model. Nothing ran on a GPU:
   every command had `CUDA_VISIBLE_DEVICES=""`. At most 12 threads per process, at most two of my processes at a time.
5. Did not touch any file outside `nn_x/w12/cb/docs_rank/` and this report. I stopped four of my own processes by exact
   PID after reading `/proc/<pid>/cmdline` (two restarts of the verification battery to fix my own script).
6. Rule notes, for the record: one early command imported catboost only to print its version without
   `CUDA_VISIBLE_DEVICES=""` (nothing was trained or predicted in it); `get_feature_importance` was called with its
   default thread count in a handful of sub-second synthetic checks; for a few seconds a third light process (a
   4-thread probe, a 1-second smoke test) ran beside the two 4-thread verification batches. Never more than two
   12-thread processes, never a GPU.

## 8. Limits: what I did NOT check

- **No lever was measured on the project's data.** Every effect size in section 1 is an inference from records and
  mechanics; section 6 is synthetic. The ranking in 1.1 is a judgement.
- GPU behaviour of anything (by rule). All "GPU" cells are from the docs.
- The docs and the source were read through a summarising fetch tool. I verified what mattered and found 11 mismatches
  (section 4); other defaults quoted from the docs alone may be wrong in the same way. The `dcg_denominator` row rests on
  the tool's reading of the page.
- Source files not opened: the MVS sampler, the CPU pairwise scoring, the fold permutation code. The statement that
  ordered boosting permutes whole groups is therefore not made here.
- `YetiRankPairwise` and `PairLogitPairwise` at scale (only 44K rows); `StochasticRank` with tuned SGLB parameters (the
  paper tunes `model_shrink_rate`, `diffusion_temperature`, `mu` and sets `l2_leaf_reg=0`).
- Timings are synthetic, single runs, on a machine at load 83 to 91; the real anchor is the infra track's.
- The cause of QuerySoftMax's cost at its default leaf settings (only the remedy, Newton leaves, is measured).
- Trap 3 (target statistics inside a feed) on real ids; trap 2 against the upstream issue tracker; no GitHub issues were
  searched systematically.
- Quantised pool save / load, `Pool.quantize()` thread use, text features, `feature_tags`, `select_features`,
  distributed training, the CLI, RELEASE.md before 1.0.
- Embedding and categorical processing parameters (the docs_feat track's subject): only acceptance by ranking losses.
- `fixed_binary_splits`, `posterior_sampling` beyond acceptance, virtual ensembles.

## 9. Sources

Documentation [D]:
- https://catboost.ai/docs/en/concepts/loss-functions-ranking
- https://catboost.ai/docs/en/references/training-parameters/common
- https://catboost.ai/docs/en/references/training-parameters/quantization
- https://catboost.ai/docs/en/concepts/algorithm-main-stages_bootstrap-options
- https://catboost.ai/docs/en/concepts/algorithm-score-functions
- https://catboost.ai/docs/en/concepts/parameter-tuning
- https://catboost.ai/docs/en/concepts/speed-up-training
- https://catboost.ai/docs/en/concepts/overfitting-detector
- https://catboost.ai/docs/en/concepts/python-reference_pool
- https://catboost.ai/docs/en/concepts/python-reference_catboostranker_fit
- https://catboost.ai/docs/en/concepts/python-reference_sum_models
- https://catboost.ai/docs/en/concepts/python-usages-examples
- https://catboost.ai/docs/en/concepts/input-data_column-descfile
- https://catboost.ai/docs/en/concepts/fstr
- https://catboost.ai/docs/en/features/snapshots
- https://catboost.ai/docs/en/features/embeddings-features
- https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/docs/en/concepts/loss-functions-ranking.md
- https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/docs/en/concepts/loss-functions.md
- https://raw.githubusercontent.com/catboost/catboost/v1.2.10/RELEASE.md
- https://raw.githubusercontent.com/catboost/tutorials/master/ranking/ranking_tutorial.ipynb

Source, tag v1.2.10, prefix https://raw.githubusercontent.com/catboost/catboost/v1.2.10/ [D]:
- catboost/private/libs/options/loss_description.cpp and .h
- catboost/private/libs/options/catboost_options.cpp
- catboost/private/libs/options/enum_helpers.cpp
- catboost/private/libs/options/bootstrap_options.cpp
- catboost/private/libs/algo/yetirank_helpers.cpp
- catboost/private/libs/algo/greedy_tensor_search.cpp
- catboost/private/libs/algo/tensor_search_helpers.cpp
- catboost/private/libs/algo_helpers/error_functions.cpp and .h
- catboost/private/libs/pairs/util.cpp
- catboost/libs/metrics/dcg.cpp
- catboost/libs/metrics/doc_comparator.h

Paper: Lyzhin, Ustimenko, Gulin, Prokhorenkova, "Which Tricks Are Important for Learning to Rank?",
https://arxiv.org/abs/2204.01500. Points used: YetiRank optimises a smoothed exponential-discount DCG with adjacent pairs
only (eq. 11, 13); YetiLoss replaces the weight by the change of the target metric; stochastic smoothing is essential
(Table 3: NDCG@10 on Web10K 50.76 with noise, 48.49 without); the neighbour count has no stable effect (Figure 1, Tables
5 to 10); YetiRank is best for NDCG@10, YetiLoss for MRR and MAP (Table 2); QueryRMSE is competitive; tuned ranges:
learning rate 1e-3 to 1, `l2_leaf_reg` 0.001 to 100, depth 6 to 8 for YetiRank and LambdaMART (Appendix A).

## 10. Exact commands

```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
  NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
D=nn_x/w12/cb/docs_rank
# verification battery (4 threads per process; two processes side by side)
setsid nohup .venv/bin/python $D/verify_synth.py objectives labels groups > $D/logs/verify_A.log 2>&1 < /dev/null &
setsid nohup .venv/bin/python $D/verify_synth.py tree sampling control baseline constraints embedding > $D/logs/verify_B.log 2>&1 < /dev/null &
.venv/bin/python $D/verify_synth.py defaults
setsid nohup .venv/bin/python $D/verify_synth.py followup pointwise > $D/logs/verify_C.log 2>&1 < /dev/null &
# target statistics inside a feed
.venv/bin/python $D/verify_ctr_group.py ; .venv/bin/python $D/verify_ctr_group2.py
# CPU cost (12 threads)
setsid nohup .venv/bin/python $D/bench_cpu.py 10000 16 > $D/logs/bench_10000.log 2>&1 < /dev/null &
setsid nohup .venv/bin/python $D/bench_cpu.py 40000 10 "YetiRank (Classic)" "LambdaMart:norm=false" "QueryRMSE (teacher" \
  "QuerySoftMax" "YetiRank:mode=NDCG" "Lossguide max_leaves 255 depth" > $D/logs/bench_40000.log 2>&1 < /dev/null &
.venv/bin/python -u $D/bench_cpu.py 10000 16 "YetiRank (Classic) =" "X1" "X2" "X3" "X4"      # -> out/bench_10000_extra.json
# direction check (12 threads, 3 seeds per arm)
setsid nohup .venv/bin/python -u $D/synth_levers.py > $D/logs/synth_levers.log 2>&1 < /dev/null &          # arms H0-H15, S1-S9
setsid nohup .venv/bin/python -u $D/synth_levers.py T0 T1 T2 T3 T4 T5 T6 T7 T8 > $D/logs/synth_levers_T.log 2>&1 < /dev/null &
setsid nohup .venv/bin/python -u $D/synth_levers.py "H0 " T9 > $D/logs/synth_levers_H0T9.log 2>&1 < /dev/null &
# (the T arms were added to the script after the first run; an unfiltered run now executes all 35 arms)
```

After the two main timing runs I added four configurations (X1 to X4) to `bench_cpu.py`; since then a filtered run
writes `out/bench_<n>_extra.json`, and the filter `"QuerySoftMax"` also matches X2.

Row count of the exploration selection (read only, for the size of the timing data): a one-off python snippet over
`data/flat/train/feeds.parquet` (column M) and `eda/labels_position/out/train_prandom.npz`: 164,118 feeds with p_random
> 0.5 and M ≥ 50, 24,372,573 rows, M percentiles 54 / 80 / 128 / 214 / 283 [M].

## 11. Files created

All under `<repo>/nn_x/w12/cb/docs_rank/`:

| File | Content |
|---|---|
| `verify_synth.py` | the verification battery (12 sections) |
| `verify_ctr_group.py`, `verify_ctr_group2.py` | target statistics of feed-constant categoricals |
| `bench_cpu.py` | CPU cost per iteration |
| `synth_levers.py` | direction check, 3 seeds per arm |
| `out/verify_<section>.json`, `out/verify_all.tsv` | every check with status and exact error text |
| `out/resolved_defaults.json` | `get_all_params()` per objective |
| `out/verify_ctr_group.json`, `out/verify_ctr_group2.json` | trap 3 |
| `out/bench_10000.json`, `out/bench_10000_extra.json`, `out/bench_40000.json`, `out/bench_300.json` (smoke run) | timings |
| `out/synth_levers.json`, `out/synth_levers_filter_T0.json`, `out/synth_levers_filter_H0.json` | section 6 |
| `out/snap_train_dir/`, `out/snap_train_dir_ref/` | snapshot test folders |
| `logs/*.log` | run logs |

Report: `<repo>/reports/CATBOOST_1115/docs_rank.md` (this file).
