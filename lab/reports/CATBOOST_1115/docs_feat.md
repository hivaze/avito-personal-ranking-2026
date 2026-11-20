# CatBoost feature processing: lever map for the tree-member track (KEY = docs_feat)

Date 2026-11-15 (UTC). Installed CatBoost 1.2.10, CPU only. Tags: **[M]** measured by me in this session, **[D]** read in documentation or source (URL in section 11), **[I]** inferred.
Everything marked [M] on "synthetic" is a capability check on generated data (4,500 to 25,000 rows), not a statement about our data.
Everything marked "pilot" is a small real-data probe (4,000 exploration feeds, 588,983 rows, 177 cb26 features, 300 trees, val 14,423 feeds); it is **not** an arm of the track and no KEEP decision is taken from it.

## 0. Verdict

PILOT_VERDICT_PLACEHOLDER

## 1. Lever list, ranked by expected value for a tree that is representation-limited at about 0.385

Bar to remember (PREDECLARED.txt): stop rule 0.3900, member bar about 0.3965 at rho 0.90. "EV" is my estimate of the val gain of the lever on the 134K-feed model, with the evidence class.

LEVER_TABLE_PLACEHOLDER

## 2. Capability matrix on the installed 1.2.10, CPU (synthetic checks, PASS / FAIL with the exact error)

Source: `nn_x/w12/cb/docs_feat/logs/t1_loss_matrix_run1.json`, `t2_embedding.json`, `t2b_embedding_more.json`, `t3_categorical_*.json`, `t4_*.json` [M].

| loss (CPU) | hard 0/1 label | real-valued label (negative values) | label in (0,1) | `embedding_features` accepted | LDA informative | KNN output | `text_features` | NaiveBayes / BM25 | cat features (CTR) | `ctr_target_border_count` > 1 | `Pool(baseline=)` | `init_model` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logloss | PASS | FAIL (a) | FAIL (a) | PASS | **yes** | per-class neighbour counts (2 features) | PASS | PASS | PASS | FAIL (d) | PASS | PASS |
| CrossEntropy | PASS | FAIL (b) | PASS | PASS | no | constant (useless) with soft labels (e) | PASS | not tested | PASS | FAIL (d) | PASS | PASS |
| RMSE | PASS | PASS | PASS | PASS | no | mean target of the k neighbours (1 feature) | PASS (BoW) | FAIL (c) | PASS | **PASS** | PASS | PASS |
| YetiRank (Classic), YetiRank:mode=NDCG | PASS | PASS | PASS | PASS | no | mean target (1 feature) | PASS (BoW) | FAIL (c) | PASS | FAIL (d) | PASS | PASS |
| YetiRankPairwise | PASS | PASS | PASS | PASS | no | mean target | PASS (BoW) | not tested | PASS | FAIL (d) | PASS | PASS |
| QuerySoftMax | PASS | FAIL (b) | PASS | PASS | no | mean target | PASS (BoW) | not tested | PASS | FAIL (d) | PASS | PASS |
| QueryRMSE | PASS | PASS | PASS | PASS | no | mean target | PASS (BoW) | FAIL (c) | PASS | FAIL (d) | PASS | PASS |
| LambdaMart | PASS | PASS | PASS | PASS | no | mean target | PASS (BoW) | FAIL (c) | PASS | FAIL (d) | PASS | PASS |
| StochasticRank:metric=NDCG | PASS | PASS | PASS | PASS | no | mean target | PASS (BoW) | not tested | PASS | FAIL (d) | PASS | PASS |
| PairLogit, PairLogitPairwise | PASS | PASS (only the order is used) | PASS | PASS | no | mean target | PASS (BoW) | not tested | PASS (default CTR is `Counter` only: no target statistics) | FAIL (d) | PASS | PASS |
| QueryCrossEntropy | FAIL (f) | | | | | | | | | | | |

Exact error texts [M]:
- (a) `catboost/private/libs/target/target_converter.cpp:410: Target with classes must contain only 2 unique values for binary classification`
- (b) `catboost/libs/metrics/metric.cpp:7015: Min target less than 0: -3.83699` (QuerySoftMax: line 7021)
- (c) `catboost/private/libs/options/text_processing_options.cpp:220: Text feature processing feature calcer has type NaiveBayes that is supported only for classification` (same for BM25)
- (d) `catboost/private/libs/options/catboost_options.cpp:490: Setting TargetBorderCount is not supported for loss function YetiRank` (same text for Logloss, CrossEntropy, QueryRMSE, QuerySoftMax, YetiRankPairwise, LambdaMart, StochasticRank, PairLogit, PairLogitPairwise, Huber, Tweedie). Accepted for RMSE, MAE, Quantile, LogCosh, Poisson [M]; the source lists RMSE, LogCosh, Quantile, MultiQuantile, LogLinQuantile, Poisson, MAPE, MAE, MultiClass, MultiRMSE, SurvivalAft, RMSPE [D].
- (e) no error; the model trained on the embedding column alone gives a constant prediction (eval NDCG 0.2167 = all ties) [M].
- (f) `catboost/private/libs/options/catboost_options.cpp:661: QueryCrossEntropy loss function is not supported for CPU learning`

Two crashes found (process dies with exit code 139, no Python exception) [M]:
- `final_ctr_computation_mode="Skip"` then `predict()` on a model with CTR features.
- `has_time=True` together with `Pool(timestamp=...)` whose values decrease over the rows.

## 3. (a) Embedding features

### 3.1 What the library does

- Input [M]: `Pool(df, embedding_features=["col"])` with an object column of arrays, or `Pool(df, embedding_features=["a","b"], embedding_features_data={"a": X_a, "b": X_b})` with 2-D float arrays (no object column, the efficient form; docstring `core.py` lines 671-681). Command line: column type `NumVector` [D].
- "We do not use coordinates of embedding features in our models"; to use raw coordinates "add them as numerical features together with embedding ones" [D, embeddings-features page]. The model only sees the **estimated features** of the calcers.
- Two calcers exist in 1.2.10: `LDA` and `KNN` [D source, M: any other name fails with `Unknown feature estimator type PCA`; a text calcer name such as `BoW` is accepted and silently produces nothing].
- Default per embedding column: `['LDA', 'KNN']` [M, `get_all_params()`].
- **LDA** [D lda.cpp, embedding_feature_estimators.cpp]: generalised eigen-problem between the class-centroid scatter and the pooled within-class scatter (LAPACK `ssygst`/`ssyev`), diagonal regulariser `reg` (default 5e-5), output = projection on `components` directions (default min(classes - 1, dim - 1); 1 for a non-classification target), optional per-class Gaussian likelihoods (`likelihood`, default false). With a non-classification target every row is put in ONE class, so the between-class matrix is zero and the direction carries no target information [D]; measured: at random level under every ranking / regression loss (table 3.3) [M]. The documentation page says "for classification the features will be calculated as Gaussian likelihood values for each class"; the installed default is the projection (1 feature for 2 classes) [M].
- **KNN** [D knn.cpp]: approximate nearest neighbours with an online HNSW index, L2 distance, `k` default 5. Classification: raw counts of each class among the k neighbours (2 features for a binary label). Non-classification (RMSE and all ranking losses): ONE feature = mean target of the k neighbours. The whole training cloud (all vectors, float32) is stored in the model.
- Leak safety during training [D base_embedding_feature_estimator.h]: the estimator is "online": rows are visited in the fold's permutation order, the feature of a row is computed from the rows BEFORE it, then the row is added (`Compute(...)` then `calcerVisitor.Update(target[line], ...)`). Eval / apply rows use the calcer built on the whole learn set. `permutation_count` is 4 by default, 1 with `has_time=True` [M].
- With groups the permutation shuffles the group order and the rows inside each group; groups stay contiguous [D objects_grouping.cpp]. So the prefix of a row contains earlier feeds AND a random part of its own feed.
- Options, exact spelling [M]: pass through the generic `CatBoost(params)` constructor; `CatBoostRanker(embedding_calcers=...)` raises `TypeError: ... unexpected keyword argument 'embedding_calcers'`.
  - `embedding_calcers=["LDA:components=1:reg=0.01:likelihood=true", "KNN:k=50"]` (one list for all embedding columns; several calcers of the same type are allowed).
  - per column: `embedding_processing={"embedding_processing": {"default": ["LDA:components=1"], "0": ["KNN:k=7"], "1": [...]}}`; the key is the index among the EMBEDDING columns (0 = first embedding column). A flat feature index as key is silently ignored [M]. The un-nested form `{"default": [...]}` fails with `json_helper.h:214: Invalid parameter: default`.
  - both parameters together: `embedding_processing_options.cpp:120: You should provide either embedding_processing option or embedding_calcers options.`
  - `LDA:components=4` on a 2-class target is accepted and returns 4 columns [M].
- Several embedding columns: each gets its own calcers (`src_feature_id=0,1,2` in the split descriptions) [M].
- Tasks [D release notes]: 0.24.2 "Supported embedding features as input and linear discriminant analysis"; 1.1 "Support text and embedding features for regression and ranking"; 1.2.5 "Fix training with embedding features on GPU". GPU not tested here (rule).
- The estimated feature VALUES cannot be exported: there is no API for them [M: only `_get_tree_splits` shows them as `src_feature_id=0 calcer_id=<guid> local_id=0`]. Work-around: recompute them by hand (LDA = one closed-form direction, section 3.4; KNN = any ANN library, out-of-fold by user for the train rows).
- A model with embedding features: `calc_leaf_indexes`, `staged_predict`, SHAP, LossFunctionChange, Interaction, save / load `cbm` all PASS; every other export format fails with `model_exporter.cpp:157: Can save model with embedding features only in cbm format`; `sum_models` fails with `model.cpp:1901: Models summation is not supported for models with embedding features` [M].

### 3.2 Which calcer carries information under which target (synthetic, the label depends on the dot product u.e; the embedding column is the element-wise product u*e) [M]

25,000 rows, 300 train / 200 eval groups, model trained on the embedding column alone, eval NDCG mean ± SE over 200 groups. References: random score 0.3227 ± 0.0084, plain dot product 0.4826 ± 0.0137.

| calcers | hard label, Logloss | hard label, YetiRank | soft real label, YetiRank | soft real label, RMSE | label in (0,1), CrossEntropy |
|---|---|---|---|---|---|
| default (LDA + KNN:k=5) | 0.4722 ± 0.0143 | 0.3779 ± 0.0119 | 0.4295 ± 0.0132 | 0.4293 ± 0.0130 | 0.3139 ± 0.0075 |
| LDA:components=1 | **0.4722 ± 0.0139** | 0.3276 ± 0.0089 | 0.3180 ± 0.0077 | 0.3212 ± 0.0084 | 0.3219 ± 0.0084 |
| LDA:components=4 | 0.4661 ± 0.0134 | 0.3879 ± 0.0118 | 0.3886 ± 0.0118 | 0.3879 ± 0.0116 | 0.3824 ± 0.0116 |
| KNN:k=5 | 0.2922 ± 0.0086 | 0.2922 ± 0.0086 | 0.4220 ± 0.0123 | 0.4201 ± 0.0122 | 0.2167 (constant) |
| KNN:k=50 | 0.3802 ± 0.0108 | 0.3773 ± 0.0103 | **0.4734 ± 0.0143** | 0.4678 ± 0.0140 | 0.2167 (constant) |

Reading:
- LDA works only with a classification loss (it then recovers the dot product: 0.472 vs 0.483). Under every ranking loss it is an arbitrary direction.
- KNN with a sparse binary label and k=5 is worse than random (a discrete feature with ties); it needs a large k, or a dense real-valued target: with teacher-like soft labels KNN:k=50 reaches the level of the dot product under a ranking loss.
- So **"CatBoostRanker + embedding_features with defaults" is close to a no-op** (0.378 vs 0.323 random vs 0.483 for the plain dot); the two usable combinations are (LDA, Logloss) and (KNN with large k, soft targets, RMSE or a ranking loss).

### 3.3 Cost on real pair embeddings (u*e, 128-d float32, exploration pool), 5 trees, 12 threads, machine load about 90 [M]

Source: `logs/p1_emb_cost.jsonl`.

| rows | dim | calcer | fit time | predict 200K val rows | model file | max RSS of the process |
|---|---|---|---|---|---|---|
| 101,552 | 128 | LDA:components=1 | 1.0 s | 0.1 s | < 0.1 MB | 5.4 GB |
| 415,806 | 128 | LDA:components=1 | 4.1 s | 0.1 s | < 0.1 MB | 10.6 GB |
| 101,552 | 128 | KNN:k=5 | 63.5 s | 3.8 s | 57.1 MB | 5.4 GB |
| 415,806 | 128 | KNN:k=5 | 320.3 s | 4.8 s | 233.4 MB | 10.6 GB |
| 1,245,766 | 128 | KNN:k=5 | 928.6 s | 7.2 s | 709.7 MB | 12.8 GB |
| 415,806 | 32 (PCA of the pair vector) | KNN:k=5 | 186.5 s | 3.0 s | 73.7 MB | 10.6 GB |

- KNN fitting is single-threaded (the estimator does not use the executor [D]; the process sat at 100% of one core [M]) and about linear in rows: 0.62 to 0.77 ms per row at 128-d. Extrapolation to the full pool (19,741,607 rows): **4 to 5.5 hours of one core and a model file of about 11 GB (570 bytes per row)** for KNN:k=5 [I from M]; about 0.6x of the time and 0.31x of the size after PCA to 32-d.
- KNN:k=50 is much slower than k=5: KNN_K50_COST_PLACEHOLDER
- LDA is negligible: about 10 microseconds per row, about 3.5 minutes for the full pool [I from M].
- Memory: the process grew by about 2.7 GB per million rows between the two larger runs (my float32 array, the Pool copy and the clouds of the permutations); about 60 GB for the full pool [I from M]. The machine has ample RAM [M].

### 3.4 Pilot on real data: does the pair embedding add to the 177 cb26 features?

PILOT_TABLE_PLACEHOLDER

## 4. (b) Text features

- Our data has **no text** [M]: the competition readme lists only ids, five numeric features, embeddings and history arrays; `data/raw` holds two zip archives and the sample submission; `data/flat` holds numeric arrays only.
- A text feature is still the only native way to give CatBoost a variable-length SET of ids (history items, categories, event types): a string of space-separated tokens [I].
- Pipeline [D text-to-numeric page]: tokenizer -> dictionaries -> calcers.
  - `BoW`: one 0/1 feature per dictionary token ("whether the object contains the token_id"), option `top_tokens_count` (default 2000 [D source]); label-free.
  - `NaiveBayes`: multinomial naive Bayes, one feature per class; `BM25`: class-relevance scores. Both use the target and are "computed online on several dataset permutations, similarly to the estimation of CTRs" [D].
- Installed defaults [M, `get_all_params()`]: tokenizer `Space` (delimiter " ", no lowercasing); dictionaries `Word` (gram_order 1) and `BiGram` (gram_order 2), both `occurrence_lower_bound` 5 (the documentation page says 3) and `max_dictionary_size` 50000; calcers: `BoW` on both dictionaries for every loss, plus `NaiveBayes` on `Word` for classification losses only.
- Parameter spelling [M]: `tokenizers=[{"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " "}]`, `dictionaries=[{"dictionary_id": "W", "occurrence_lower_bound": "1", "max_dictionary_size": "50000", "gram_order": "1"}]`, `feature_calcers=["BoW:top_tokens_count=2000", "NaiveBayes", "BM25"]`, or one `text_processing={"tokenizers": [...], "dictionaries": [...], "feature_processing": {"default": [{"dictionaries_names": [...], "feature_calcers": [...], "tokenizers_names": [...]}]}}` (not together with the three separate parameters [D]).
- With ranking and regression losses only `BoW` exists (error (c) in section 2) [M]; release 1.1 "Support text and embedding features for regression and ranking", 1.2 "only Bag-of-Words features are generated for now" (multi-target) [D].
- Synthetic check [M] (15,000 rows, 100 eval groups; text = candidate category crossed with each of 8 history-category tokens, `c3_h7 c3_h1 ...`): numeric only 0.4559 ± 0.0199 (Logloss); numeric + crossed text 0.5478 ± 0.0208 (Logloss, BoW + NaiveBayes), 0.5108 ± 0.0193 (YetiRank, BoW only). A text that is constant inside a group (the history alone) gives 0.2167 = constant: under a ranking loss a group-constant feature can only act through interactions, and BoW features are split one token at a time.
- Leak [M]: with a group-constant text and pure-noise labels the learn Logloss falls from 0.6931 to 0.5516 (NaiveBayes), 0.5193 (BM25), 0.5434 (default) while eval stays at 0.694; BoW stays at 0.6931. The supervised text calcers have the same within-group prefix leak as CTRs (section 5.3).
- Model with text features: `calc_leaf_indexes` PASS, SHAP PASS, `sum_models` FAIL (`Models summation is not supported for models with text features`), export only `cbm` [M].
- Relevance here: an untried carrier for "candidate cluster x history cluster / event type" co-occurrence. Under YetiRank it is label-free BoW (up to 2000 binary columns per dictionary), which duplicates the hand-made `clu_*`, `cat_cnt_*`, `ctx_*` counts; the label-aware calcers need Logloss, which costs about 0.005 against YetiRank in the pilot (section 3.4). Volume: 24.2M history events in the pool (mean H 180), so a crossed text over the last 20 events is about 395M tokens for 19.7M rows [M counts, I product].

## 5. (c) Categorical features

### 5.1 Mechanics [D unless marked]

- Steps: random permutation, label quantisation, statistic "calculated individually for each object using data from previous objects" (cat-to-numeric page).
- CTR types and formulas (cat-to-numeric page): `Borders`: (countInClass + prior) / (totalCount + 1) with countInClass = number of earlier objects of this value whose label exceeds border i (one feature per target border); `Buckets`: the same with "label equals bucket i" (k + 1 features); `BinarizedTargetMeanValue`: countInClass = sum of the quantised labels / k; `Counter`: (curCount + prior) / (maxCount + 1), label-free.
- CPU types: Borders, Buckets, BinarizedTargetMeanValue, Counter. `FeatureFreq` and `FloatTargetMeanValue` are GPU-only: `Ctr type FeatureFreq is not implemented on CPU yet` [M].
- Installed defaults under YetiRank, Logloss, RMSE on CPU [M, `get_all_params()`]: `simple_ctr` = `combinations_ctr` = `Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1` + `Counter:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1`. PairLogit and PairLogitPairwise: `Counter` only, so no target statistics at all [M, D source].
- So a categorical feature becomes, per prior, a number quantised into **15 uniform bins** (16 levels). `CtrBorderCount` up to 255 is accepted on CPU [M]; the CTR border type on CPU must be Uniform [D].
- Target for the statistics: one border (MinEntropy) by default. With a binary label that is exact. With a real-valued (teacher) label one border keeps one bit of the target; more borders are allowed only for the RMSE family (error (d)), i.e. **not** under a ranking loss [M].
- `per_feature_ctr` needs explicit priors: `["5:Borders:Prior=0.5/1:Prior=1/1"]` PASS, without a prior `catboost_options.cpp:497: Provide at least one prior for CTR` [M]. The key is the flat feature index.
- `counter_calc_method`: `SkipTest` (default) or `Full` (also counts the eval pool); `PrefixTest` / `FullTest` of the docstring do not exist: `Key 'PrefixTest' not found in enum ECounterCalc. Valid options are: 'Full', 'SkipTest'` [M].
- `one_hot_max_size` effective default [M]: 10 for YetiRank and PairLogit, 2 for Logloss and RMSE, 1 for YetiRankPairwise (one-hot is not available in pairwise scoring on CPU [D]). Features at or below the threshold are one-hot encoded and get no CTR. `category_id` (52 values in the pool [M]) is therefore CTR-encoded in every model of record.
- Feature combinations (paper 1810.11363): "No combinations are considered for the first split in the tree. For the next splits CatBoost combines all combinations and categorical features present in current tree with all categorical features in dataset ... all the splits selected in the tree are considered as categorical with two values and used in combinations in the same way as categorical ones." Limit `max_ctr_complexity`, effective default **1 when iterations <= 100 and 4 from 300 iterations on** [M: 10 -> 1, 100 -> 1, 300 -> 4, 1000 -> 4]. The organisers' baseline sets `max_ctr_complexity=1`.
- `model_size_reg` (effective default 0.5 [M]): every split score is multiplied by (1 + u/U)^M where u is the number of distinct values of the combination present in the training data (CPU) [D model-size-reg page]. It pushes the model away from high-cardinality ids: on the synthetic set `item_id` importance is 14.7 at `model_size_reg=0`, 9.2 at the default, 0.0 at 5 [M].
- `ctr_leaf_count_limit` (CPU only) keeps the N most frequent values; `store_all_simple_ctr` protects simple CTRs from that limit [D]; both accepted [M].
- `final_ctr_computation_mode="Skip"` trains without the final tables; predicting with such a model kills the process [M].
- `has_time=True`: no random permutation, one permutation in the input order [D, M `permutation_count` 1]. `Pool(timestamp=...)` must be constant inside a group (`preprocess.cpp:121: timestamps[1] = 489 is not equal to the timestamp of group's first element`) and must be non-decreasing (otherwise the process crashes) [M].
- `ctr_history_unit` (`Sample` default, `Group`): on GPU the source sets `Group` for group-wise losses; on CPU it cannot be changed: `json_helper.h:185: Error: change of option ctr_history_unit is unimplemented for task type CPU and was not default in previous run` [M, D]. The `CatBoostRanker` constructor accepts the keyword; the error comes at `fit`.
- Apply time: the final tables are computed on all training rows of the pool; unseen values get the prior [D, I].

### 5.2 Other parameters accepted on CPU under YetiRank and Logloss [M]

`simple_ctr`, `combinations_ctr` (types Borders, Buckets, BinarizedTargetMeanValue, Counter; priors in both spellings `Prior=0.5` and `Prior=0.5/1`), `max_ctr_complexity` 1 / 2 / 6, `one_hot_max_size` 20 / 255 / 2000, `ctr_leaf_count_limit`, `store_all_simple_ctr`, `has_time`, `model_size_reg`, `boosting_type="Ordered"`, `approx_on_full_history`, `fold_permutation_block`, `grow_policy` Lossguide / Depthwise: all PASS (`logs/t3_categorical_a.log`, `t3_categorical_a2.log`).

### 5.3 The within-feed prefix leak (the main finding of this section) [M]

Test: 800 groups of 10 rows, 5 positives each, labels are pure noise; the only feature is a categorical id that is constant inside the group (a user id). An honest feature cannot reduce the learn loss.

| setting (CPU) | Logloss: learn first -> last iteration (0.6931 = nothing learnt) | eval Logloss | YetiRank: learn NDCG first -> last (0.5410 = nothing learnt) | eval NDCG |
|---|---|---|---|---|
| default CTR (history unit Sample) | 0.6459 -> **0.5627** | 0.6932 | 0.8645 -> **0.9575** | 0.5410 |
| `boosting_type="Ordered"` | 0.6459 -> 0.5728 | 0.6932 | 0.8585 -> 0.9581 | 0.5410 |
| `has_time=True`, rows in random order inside the group | 0.6625 -> 0.5738 | 0.6941 | 0.8639 -> 0.9541 | 0.5410 |
| `has_time=True`, rows label-sorted inside the group | 0.2919 -> **0.1388** | 4.4460 | 0.9896 -> 0.9896 | 0.5410 |
| one-hot (`one_hot_max_size=1000`) | 0.6931 -> 0.6931 | 0.6931 | 0.5410 -> 0.5410 | 0.5410 |
| `simple_ctr=["Counter"]`, `combinations_ctr=["Counter"]` | 0.6931 -> 0.6931 | 0.6931 | 0.5410 -> 0.5410 | 0.5410 |
| `ctr_history_unit="Group"` | FAIL on CPU (error above) | | FAIL on CPU | |
| group-constant embedding, KNN:k=5 | 0.6931 -> 0.6924 | 0.6931 | 0.5416 -> 0.5416 | 0.5410 |
| group-constant embedding, LDA | 0.6931 -> 0.6931 | 0.6932 | 0.5410 -> 0.5410 | 0.5410 |
| group-constant text, NaiveBayes / BM25 | 0.6480 -> 0.5516 / 0.6206 -> 0.5193 | 0.6939 / 0.6979 | not run | |

Reading: on CPU the ordered statistic of a key that repeats inside a feed is computed from the labels of the same feed's earlier rows. The model learns "a click was already seen in this feed", which does not exist at apply time. Any feed-level or user-level key (user_id, feed_id, user cluster, hour, and every cross that contains one of them together with a candidate key that repeats inside a feed) is unsafe as a native CTR feature on CPU. The only native fixes are one-hot, `Counter`, or the GPU (`ctr_history_unit=Group`, not allowed now). The project's hand-made leave-user-out `tte_*` columns are the correct form.
`has_time=True` makes it worse when the row order inside a feed is related to the label. In the pool, the first flat row of a feed has a target rate of 0.01528 against 0.01244 for all rows [M, `logs/d1_data_look.json`], so rows must be shuffled inside feeds before any `has_time` run.

## 6. (d) Numeric features

- Quantisation [D quantization page]: `border_count` (alias `max_bin`), CPU default 254, GPU 128 (32 for the pairwise losses), allowed 1 to 65535; `feature_border_type` default GreedyLogSum, values Median, Uniform, UniformAndQuantiles, MaxLogSum, MinEntropy, GreedyLogSum; `nan_mode` Min (default), Max, Forbidden; `input_borders` / `output_borders` files.
- `per_float_feature_quantization=["0:border_count=1024"]`, several settings comma-separated: `"0:border_count=1024,nan_mode=Max,border_type=Median"` PASS; a colon between the settings fails (`util/string/split.h:452: Split: number of fields more than number of Split output arguments`) [M].
- "Golden features" [D parameter-tuning page]: "It is recommended to use an increased number of borders (1024) for this feature", "should not be set for all features", "recommended to set it for one or two golden features".
- All of border_count 32 / 254 / 1024 / 4096 / 65535, the six border types, nan_mode Min / Max PASS under YetiRank and Logloss; `nan_mode="Forbidden"` with NaNs fails as documented [M, `logs/t4_quant.log`]. On the synthetic set every variant is inside ± 1 SE of the default.
- YetiRankPairwise on CPU silently uses `border_count=32`, Bernoulli 0.5 and Gradient leaf estimation [M `get_all_params()`]: an arm that switches to it changes the quantisation too.
- cb26 used 254 borders (GPU, set by hand); the CPU default is already 254. Candidates for 1024 borders: the two top columns of e3_xm_yeti (`it_hist_et1_luo` 6.2, `pre2_fe_z` 4.8).

## 7. (e) Combinations, selection, importance

- Combinations: only categorical x categorical and categorical x (binary split of a numeric feature already in the tree), section 5.1. CatBoost does **not** build numeric x numeric products, ratios or differences; with ONE categorical feature (`category_id`) the only combinations are category x numeric thresholds, up to 4 conditions [D, I].
- `select_features` [D page, M]: recursive elimination; `algorithm` = `RecursiveByShapValues` (default, "most accurate"), `RecursiveByLossFunctionChange`, `RecursiveByPredictionValuesChange` ("not recommended for ranking losses"); `steps` = number of retrainings; `shap_calc_type`; `train_final_model`. Returns `selected_features`, `eliminated_features` (+ `_names`) and `loss_graph` [M]. Runs with YetiRank, with and without categorical features, all three algorithms [M].
- Importance types [D fstr page, M under YetiRank]: `PredictionValuesChange` (default for non-ranking losses; "formula values inside different groups may vary significantly in ranking modes", so group-wise features look more important than they are); `LossFunctionChange` (default for ranking losses, needs a dataset: without one `Feature importance type EFstrType.LossFunctionChange requires training dataset`); `ShapValues`, `ShapInteractionValues`, `Interaction` (pairs of features in the same tree), `SageValues` all PASS [M]; `PredictionDiff` needs whole groups and no categorical features (it failed on a 2-row subset: `Subset's last group size (2) is less than corresponding source group size (50)`) [M].
- What they would tell us about the 177-feature ranker: all cb26 / W25 tables on record use `PredictionValuesChange`, the type the documentation warns about for ranking. `M` is the top feature (9.9) of the pilot's Logloss model and absent from the top of the YetiRank model [M pilot]: an illustration. A `LossFunctionChange` read on val (1.39M rows; the library subsamples to max(2e5, 2e9 / features) rows [D]) plus `select_features` with `RecursiveByShapValues` would say which of the 177 columns carry loss; the W25 ablations already put every block inside the one-seed spread, so the expected gain is small [I].

## 8. (f) Getting things out of a model [M unless marked]

| API | numeric model | with CTR features | with text | with embedding |
|---|---|---|---|---|
| `calc_leaf_indexes(pool, ntree_start, ntree_end)` -> uint32 [rows, trees]; `iterate_leaf_indexes` | PASS | PASS | PASS | PASS |
| `staged_predict(pool, eval_period=...)`, `predict(ntree_end=...)` | PASS | PASS | PASS | PASS |
| `sum_models(models, weights, ctr_merge_policy)` | PASS, exact to 2e-15, also across losses (YetiRank + QuerySoftMax) | PASS with IntersectingCountersAverage, LeaveMostDiversifiedTable, KeepAllTables (prediction equal to the weighted mean to 2e-15); FailIfCtrIntersects refuses | FAIL | FAIL |
| export | cbm, json, cpp, python, onnx, coreml, pmml | cbm, json, cpp, python; onnx / coreml / pmml FAIL | cbm only | cbm only |
| `shrink`, `drop_unused_features`, `set_scale_and_bias`, `get_borders`, `get_leaf_values` | PASS | | | |
| `Pool(baseline=...)` (start from another model's score), `fit(init_model=...)` | PASS for every CPU loss of section 2 | | | |

- "Model-based features" in practice: (1) leaf indexes of model A as categorical inputs of model B (synthetic: a second YetiRank model on 20 leaf-index columns alone reaches 0.5111 ± 0.0212 against 0.5477 ± 0.0207 for the first model, n = 100 groups; so it runs, and is not a gain by itself); (2) the score of model A as `baseline` of model B (residual boosting) or as a column.
- For train rows both forms must be out-of-fold by user, otherwise model B reads the labels through model A [I, the track's leak rule]. COVERAGE_MAP row 258 closed leaf-SVD64 as a residual on the NN pool (-0.00264 ± 0.00064); the NN-input form is still marked untested there.
- `sum_models` only saves bookkeeping (one file for a seed bag); it is identical to averaging raw scores and the track averages per-feed z-scores instead.

## 9. Our data: which raw objects could go into CatBoost natively

Counts from `logs/d1_data_look.json` [M]: pool = 134,118 exploration feeds (p_random > 0.5, M >= 50, holdout excluded, overlap 0), 19,741,607 rows, 125,825 distinct users, 1.83 positives per feed; item-embedding tables are int8 [N, 128] per split (train 81,557,944 rows), `user_emb.npy` is int8 [feeds, 128]; all vectors are unit-norm x 256 (reports/eda_embeddings.md).

| raw object | native form | facts [M] | leak / mismatch risk | my reading |
|---|---|---|---|---|
| candidate item embedding e | embedding column | 11.39M distinct vectors in the pool | none for the rules; KNN / LDA on e alone = an item-level prior | skip: item-level target rate is not predictable from e (eda_embeddings: kNN R2 < 0) |
| user embedding u, mean / last-k history embedding | embedding column | constant inside a feed | KNN neighbours of a train row are its own feed's rows (measured effect on noise labels: none, section 5.3) | useless alone under a ranking loss (constant inside the feed) |
| pair vector u*e (row sum = cosine) | embedding column | pilot, section 3.4 | calcers see pool labels only: rule-clean. Train-time neighbours can be rows of the same feed (same u); apply-time neighbours never are [I] | tested in the pilot |
| e - mean(last-k history), e * mean(history), [u ; e] (256-d) | embedding column | not built | same as above | untested; the same kind of signal as the pair vector and the existing `cos_*`, `maxsim32`, `attn32` columns |
| item_id | categorical | 11,391,438 distinct in the pool, 69.5% seen once, 1.73 rows per item; val rows whose item is in the pool: 41.0% (11.4% with >= 5 pool rows); in ALL train rows: 65.4% (44.3% with >= 5) | random permutations put LATER feeds of the same user into the statistic (rule 6); `has_time=True` on the time-sorted flat order (verified sorted) avoids that. No within-feed repetition (0 duplicates) | dominated by the hand-made `tte_it_*` (all train rows, leave-user-out); needs `model_size_reg` near 0 and an 11M-value table |
| category_id | categorical (CTR) or one-hot (`one_hot_max_size=64`) | 52 values in the pool | negligible: about 380K pool rows per category against at most M same-feed rows [I] | already used; one-hot is a free variant |
| candidate cluster ids (`data/clu_work/cand_cl_{256,1024,4096}_*.npy`) | categorical | about 4,800 pool rows per cluster at 4096 | low (same argument) | cheap untested arm: cluster CTR and cluster x category x numeric-split combinations |
| user_id, feed-level ids, user x category | categorical | val feeds whose user is in the pool: 2.2% (in all train: 30.4%); test 2.2% / 29.1% | **high on CPU**: within-feed prefix leak (section 5.3) | do not use natively; keep `tte_u*`, `tte_uc*` |
| history as a token set (category, cluster, event type, crossed with the candidate) | text column | 24.2M history events in the pool, mean H 180 | BoW none; NaiveBayes / BM25 leak for feed-constant text and need Logloss | untested |
| teacher logits (`teacher_oof.npy`, `teacher_mean11.npy`) | label, `baseline`, or KNN target | `teacher_oof` = sum of 3 fold-excluded models with a per-feed offset (values 73 to 122 in the pool): centre per feed before use | as KNN target the neighbours' teacher values become INPUTS: use the out-of-fold file only | section 3.4 |

Rule 6 check for the natively fitted statistics: the pool excludes the 30K holdout by construction, val / test labels never enter a Pool that is fitted, so CTR tables and KNN clouds are train-only. The two real risks are (1) the within-feed prefix (section 5.3) and (2) same-user later feeds under random permutations; (2) is small inside the pool (134,118 feeds for 125,825 users) and absent with `has_time=True`.

## 10. Method, limits, what I did NOT check

Method: documentation and version-matched source through the web tools (generic CatBoost URLs only), the installed package (`core.py` docstrings, strings of `_catboost.so`), then one synthetic test per claim on CPU; one read-only pass over the flat arrays; one cost probe and one small pilot on real rows.

Not checked / limits:
- **Nothing on GPU** (rule). GPU-only items are [D] only: `ctr_history_unit=Group`, FeatureFreq / FloatTargetMeanValue, embedding and text calcers on GPU, QueryCrossEntropy.
- The web tool returns a model-written digest of each page, not the raw text. Quotes above are as returned; every claim that matters for a decision was re-tested on the installed version and is marked [M]. Source-only statements that I did NOT verify by a run: the LDA solver details, the HNSW search width (300), the within-group shuffle of `objects_grouping.cpp` (consistent with, but not proven by, the leak tests), `ctr_history_unit=Group` being the GPU default for group-wise losses.
- The pilot: 4,000 feeds (3% of the pool), 300 trees, no early stopping, at most 2 seeds, val only. **No holdout read** (the progress rule needs both witnesses), no rho to W16, no pool-level blend. Single-seed differences below 0.003 are noise by the track's own rule; the two-seed paired SE is quoted where two seeds exist.
- No text-feature arm, no cluster-id arm, no item_id arm, no `select_features` run, no golden-feature run on real data.
- KNN with k > 5 at full scale was not timed beyond the pilot; KNN on [u ; e] or on history-difference vectors was not built.
- `embedding_features` together with `boosting_type="Ordered"` and with `has_time` were only smoke-tested on synthetic data.
- I did not test `Pool.quantize()` / quantised pool files with embedding or text columns (the binary says "Quantized pool file format does not support embedding features yet" [D string]).
- Thread note: see "rule notes" below.

Rule notes:
- CPU only, `CUDA_VISIBLE_DEVICES=""` in every command, `task_type="CPU"` everywhere, no GPU context was opened.
- `thread_count` was 4 to 12 in every fit and predict. **Deviation:** until 00:20 UTC my scripts built `catboost.Pool` objects with the default `thread_count=-1`, so CatBoost created a pool of many threads in those processes; they ran for well under a second per Pool build (0.45 CPU-seconds per thread measured on the largest case) and then stayed idle. From 00:20 UTC the pilot script passes `thread_count=12` to `Pool`. Synthetic tests (seconds each) were affected the same way.
- At most two heavy processes of mine ran at a time. I stopped one process of my own (the chain wrapper, PID 3808231, after reading its `/proc/<pid>/cmdline`) with a plain `kill -TERM <pid>`; nothing else was signalled.
- No file outside `nn_x/w12/cb/docs_feat/` and this report was written; cb26 / src code was imported read-only. No submission file, no installs.

## 11. Exact commands

All from `<repo>`, with `export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=<4..12> MKL_NUM_THREADS=... OPENBLAS_NUM_THREADS=... NUMBA_NUM_THREADS=... POLARS_MAX_THREADS=...`, `S=nn_x/w12/cb/docs_feat/scripts`, `PY=.venv/bin/python`.

```
$PY $S/t1_loss_matrix.py                      # loss x feature-type matrix                    -> logs/t1_loss_matrix_run1.{log,json}
$PY $S/t2_embedding.py                        # embedding calcers, options, model APIs         -> logs/t2_embedding.{log,json}
$PY $S/t2b_embedding_more.py                  # calcer x target table (200 eval groups)        -> logs/t2b_embedding_more.{log,json}
$PY $S/t3_categorical.py a|a2|skip|soft|leak|leak2|tbc   # CTR parameters, leak tests          -> logs/t3_categorical_<section>.{log,json}
$PY $S/t4_text_out_quant.py text|out|quant    # text calcers, output APIs, quantisation        -> logs/t4_<section>.{log,json}
$PY $S/t5_timestamp.py                        # has_time x timestamp x groups                  -> logs/t5_timestamp.{log,json}
$PY $S/t6_emb_keys.py                         # key of embedding_processing                    -> logs/t6_emb_keys.{log,json}
$PY $S/d1_data_look.py                        # cardinalities / coverage (read-only)           -> logs/d1_data_look.{log,json}
$PY $S/p1_emb_cost.py <n_feeds> <none|LDA|KNN> [dim]     # cost probe on real pair embeddings  -> logs/p1_emb_cost.jsonl
$PY $S/p2_pilot.py <arm> <seed> [n_feeds=4000] [iters=300]   # pilot arm                       -> logs/p2_pilot.jsonl, cache/p2_<arm>_s<seed>_n4000_val.npy
$PY $S/p2_eval.py                             # paired per-feed read-out of the pilot          -> logs/p2_eval_n4000.json
```
Long jobs were started as `setsid nohup bash -c "<loop>" > nn_x/w12/cb/docs_feat/logs/<name>.log 2>&1 < /dev/null &`.

## 12. Files created

- `reports/CATBOOST_1115/docs_feat.md` (this file)
- `nn_x/w12/cb/docs_feat/scripts/`: `synth.py`, `realdata.py`, `t1_loss_matrix.py`, `t2_embedding.py`, `t2b_embedding_more.py`, `t3_categorical.py`, `t4_text_out_quant.py`, `t5_timestamp.py`, `t6_emb_keys.py`, `d1_data_look.py`, `p1_emb_cost.py`, `p2_pilot.py`, `p2_eval.py`
- `nn_x/w12/cb/docs_feat/logs/`: the `.log` / `.json` / `.jsonl` files named above, `t7_defaults.log` (effective defaults from `get_all_params()`), `p2_chain{A,B,C,D}.log`
- `nn_x/w12/cb/docs_feat/cache/`: `so_strings.txt` (strings of `_catboost.so`), `p2_*_val.npy` (pilot val scores, 1,389,971 float32 each), `tmp*/` (model files of the export tests)

## 13. Sources

Documentation (catboost.ai, current pages, read 2026-11-14/15):
- https://catboost.ai/docs/en/features/embeddings-features
- https://catboost.ai/docs/en/concepts/algorithm-main-stages_embedding-to-numeric
- https://catboost.ai/docs/en/features/text-features
- https://catboost.ai/docs/en/concepts/algorithm-main-stages_text-to-numeric
- https://catboost.ai/docs/en/references/text-processing__feature_calcers
- https://catboost.ai/docs/en/references/text-processing__test-processing__default-value
- https://catboost.ai/docs/en/features/categorical-features
- https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic
- https://catboost.ai/docs/en/references/training-parameters/ctr
- https://catboost.ai/docs/en/references/training-parameters/common
- https://catboost.ai/docs/en/references/training-parameters/quantization
- https://catboost.ai/docs/en/references/model-size-reg
- https://catboost.ai/docs/en/concepts/parameter-tuning
- https://catboost.ai/docs/en/concepts/fstr
- https://catboost.ai/docs/en/concepts/feature-interaction
- https://catboost.ai/docs/en/concepts/python-reference_catboost_select_features
- https://catboost.ai/docs/en/concepts/python-reference_sum_models
- https://catboost.ai/docs/en/concepts/python-reference_catboost_calc_leaf_indexes
- https://catboost.ai/docs/en/concepts/python-reference_pool

Source at tag v1.2.10 (https://raw.githubusercontent.com/catboost/catboost/v1.2.10/...):
- `catboost/private/libs/options/embedding_processing_options.cpp`, `cat_feature_options.cpp`, `catboost_options.cpp`
- `catboost/private/libs/feature_estimator/embedding_feature_estimators.cpp`, `base_embedding_feature_estimator.h`, `text_feature_estimators.cpp`
- `catboost/private/libs/embedding_features/knn.cpp`, `lda.cpp`
- `catboost/private/libs/algo/estimated_features.cpp`, `catboost/libs/data/objects_grouping.cpp`
- `RELEASE.md` (first 100,000 characters, releases 1.2.10 back to 0.9)
- `catboost/docs/en/...` markdown of the same pages (template placeholders, used for cross-checking only)

Paper: Dorogush, Ershov, Gulin, "CatBoost: gradient boosting with categorical features support", arXiv 1810.11363 (feature-combinations paragraph quoted in 5.1).

Installed package: `<repo>/.venv/lib/python3.12/site-packages/catboost/core.py` (docstrings, lines 614-681, 3163-3195, 4635-4720, 4850-5300), `_catboost.so` (strings).
