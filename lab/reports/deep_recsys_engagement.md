# Deep dive: engagement-prediction challenges (RecSys Challenge 2020-2023) and the NVIDIA KGMON playbook, mapped to Avito ML Cup 2026 feed ranking

Date: 2026-11-11. Tags: [V] = verified by reading the primary source this session (URL given); [S] = seen only in a search snippet; [K] = background knowledge, not re-verified.
Primary sources were read as full PDFs/code (cloned `NVIDIA-Merlin/competitions` and `rapidsai/deeplearning` with `gh`/`git`, papers downloaded from arXiv/U-Toronto). Medium posts (rapids-ai, nvidia-merlin) returned HTTP 403 and were NOT read; the same content is in the papers/code.

---------------------------------------------------------------------------------------------------

## 0. TL;DR for our problem (ranked by expected value; details and evidence in sections 2-7)

1. **Train big, calibrate small: stage-1 models on the 247M production rows, stage-2 on unbiased val.** This is exactly how NVIDIA won RecSys 2021 when "train and test have different distributions ... but the validation and test dataset cover the same time period". They trained 3 XGB + 3 NN on the old/large data, then XGBoost stage-2 on the newer validation set with (a) stage-1 predictions, (b) target encodings (TE) computed on the big train set, (c) TE and count encodings (CE) computed inside the validation set with 5-fold/LOO. Greedy forward selection showed that using all 6 diverse stage-1 models was best [V]. We have `stack_val.py`. The cheap win is **more diverse stage-1 models trained on train**, plus the stage-2 recipe (train-TE + val-OOF-TE + CE + stage-1 preds).
2. **TE of crosses that change within a feed.** In RecSys 2020/2021 the top features were TE of *entity × low-cardinality context*, e.g. `TE(user, tweet_type, language)` [V]. In our feeds the user is constant, so user-only TE cancels inside a feed. The transferable recipe is TE of item/category crossed with something that encodes the user: `TE(item, user_cluster)` and `TE(category, user_cluster)` (k-means on the 128-d user embedding, K=64..1024), `TE(category, user_top_history_category)`, `TE(item, event_source-profile)`. Compute them on 247M train rows with COEC position debiasing [K] and an expanding window (as in our `features_tte.py`). Also give the model **raw (sum, count) pairs** next to the smoothed value; NVIDIA's XGBoost 3 in 2021 did this "so that the model can learn a smoothing function" [V].
3. **Multi-target / cross-target encodings from histories.** NVIDIA fed the TE of *every* target as a feature into each target's model; for example, the reply model used TE of like, retweet and retweet-comment [V]. Our 10 history event types act as 10 free auxiliary "targets" over all items and categories in all histories (train+val+test users): per-item rates of each event type, contact-like/view ratios, and per (user_cluster, category) event-type rates.
4. **Transductive count encodings and time-window counts.** NVIDIA 2020 count-encoded over **train+test concatenated** (code `CountEncoder`/`count_encode_cudf_v2`) [V]. Engagement counts of the user in the last 5/60/240/480/1440 minutes were key time features [V]. Synerise 2021 counted interactions with the same tweet id across the val/test period [V]. For us: item appearance counts in val+test feeds (global and within ±1h/±6h/±24h of the feed timestamp), item appearances in test-period histories, and time since the item's first appearance anywhere (item-age proxy). Check `eda/history_leak` findings and the rules before using future-looking variants.
5. **History-vs-candidate features per event type, plus a target-attention NN.** Layer 6 2020 (2nd place) attended from the target tweet over the last 10 history tweets **separately per engagement type** [V]. Synerise counted interactions per reaction type and pair [V]. Our max-sim over the last 32 events is already the best single feature (0.349). Split it by event_type and event_source, add top-k-mean and recency-weighted variants and counts of the candidate's category per event type in the last N events. Then train an NN with per-event-type target attention on train, fine-tune it on val and feed it to stage-2.
6. **Treat f1..f5 as possibly obfuscated counters.** In RecSys 2023 (anonymized features), the 9th-place team found that some continuous features were arithmetic progressions (artificial noise, v_n = v_1 + (n-1)Δ). Dividing by Δ recovered integers and helped (LB 6.2648 → 6.2188) [V]. Check the unique-value spacing of f1..f5 per category/day and try ratios f_i/f_j if they are counters. Run **adversarial validation** (train-production vs val/test) per feature. Two precedents: NVIDIA 2020 saw adversarial AUC ≈ 1.0 [V], and Kim&Lee dropped features with AUC ≥ 0.75 [V].
7. **Neural model with self-supervision for the small unbiased set.** Layer 6 won RecSys 2023 with a Transformer over feature tokens: quantile-binned numeric embeddings, a contrastive SSL loss with feature corruption 5-25%, embedding masks 5-25% and dropout twins, α=0.6 joint training, then supervised fine-tuning. Log loss 0.3297 vs tuned LightGBM 0.3615 / XGBoost 0.3582 / supervised-only Transformer 0.3471 [V]. Pretrain on production train, fine-tune on val (Synerise: 2 epochs on 1B train rows, then 3 epochs on val [V]).
8. **Ensembling discipline.** Use seed bagging (Grandmasters Playbook: 100 XGB seeds gave MAP@3 ~0.376 → ~0.379 [V]), hill-climbing weights on OOF NDCG [V], a 2- or 3-stage stack (NVIDIA 2022: three stage-2 models at 0.2073/0.2074/0.2076, weighted blend 0.2082 [V]) and diverse model families with ~80% prediction correlation (NVIDIA WSDM 2021 [V]). Refit on 100% of val after choosing hyperparameters [V].

What does **not** transfer: user-level TE/CE alone (constant within a feed), BERT/text (no text here; even NVIDIA 2020 saw no gain from BERT features [V]), per-user-group models (that was about a fairness metric), and the 6 ms CPU latency engineering.

---------------------------------------------------------------------------------------------------

## 1. Why these competitions are only partially analogous (read before copying recipes)

| Aspect | RecSys 2020/2021 (Twitter) | RecSys 2022 (Dressipi) | RecSys 2023 (ShareChat) | Ours (Avito 2026) |
|---|---|---|---|---|
| Unit | (user, tweet) pair, pointwise | session → item (MRR@100 over ~28K items) | impression, pointwise | feed (5-300 candidates), list |
| Metric | RCE + PR-AUC, pooled across users | MRR | normalized cross-entropy | per-feed full-list NDCG, binary gain, pessimistic ties |
| Key consequence | user-level features (user CTR) rank users against each other and dominate importance | candidate generation matters | calibration matters | **only features that vary within a feed matter**; user features help only via interactions |
| Train/test shift | train weeks → test week (time); 2021 released val from the test period | time | time (day 22) | production (position-biased) train → uniformly shuffled exploration val/test |
| History | engagements inside the dataset | session only | aggregated counters | up to 256 events/user with item embeddings and 10 event types |

So, for example, the top features `TE_b_user_id_like`, `TE_a_user_id_reply` in NVIDIA's importance tables [V] are pooled-ranking features. The transferable ones are their crosses with item-side context (section 2.1 shows that the importance tables are full of `TE(entity + tweet_type + language)` crosses).

---------------------------------------------------------------------------------------------------

## 2. RecSys Challenge 2020 (Twitter, ~200M user-tweet pairs, 4 binary targets)

### 2.1 NVIDIA, 1st place: "GPU Accelerated Feature Engineering and Training for Recommender Systems" (Schifferer, Titericz, Deotte, Henkel, Onodera, Liu, Tunguz, Oldridge, Moreira, Erdem)
Source [V]: https://github.com/NVIDIA-Merlin/competitions/blob/main/RecSys2020_Challenge/RecSysChallenge2020.pdf and code in `RecSys2020_Challenge/02_ModelsCompetition`, `03_ModelsBenchmark`.

- **Result**: 1st in 7 of 8 metrics; final blend 132.42 public / 132.55 private (sum of RCE ranks).
- **Feature-engineering vs. model**: "building features improved our final score significantly, whereas model design played a secondary role". 20 raw features; they experimented with an extra 200 and used 120 in the final models. "By far Target Encoding was the best feature engineering technique and was responsible for the biggest scores improvements." [V]
- **Validation**: two schemes. (1) The last two days of train as validation. (2) A 50/50 split grouped by tweet_id. Even so, "results for validation set and test set were not perfectly correlated, forcing us to tune some hyperparameters based on public leaderboard". Adversarial validation (train vs test) reached AUC ≈ 1.0 [V]. A cautionary local-vs-LB gap: XGBoost 2 Like RCE was 70.0 local vs 40.3 LB (Table 1) [V].
- **Target encoding (exact)** [V]:
  `TE = (count(cat) * mean_target(cat) + w * mean_target_global) / (count(cat) + w)`, with `w = 20`, 5-fold OOF on train and full train for test.
  In code (`target_encode_cudf_v3`), folds are **contiguous blocks** (`fold = idx // (n/5)`, `shuffle=False`), which gives time-ordered blocks when the data is time-sorted. There is an optional `min_ct` that maps categories with count ≤ min_ct to an "other" class.
  - Encoded columns: single (`media, tweet_type, language, a_user_id, b_user_id, tw_hash0, tw_rt_uhash, user_flag`) × each of 4 targets, plus multi-column combos, e.g. `[domains, language, b_follows_a, tweet_type, media, a_is_verified]` (+`user_flag`) [V].
  - They also TE'd a **continuous non-label variable** (`elapsed_time`) per category, i.e. a group mean [V].
  - In the XGBoost 3 importance tables, the top features include `TE_b_user_id_tweet_type_language`, `TE_tw_last_word_tweet_type_language` (least popular word), `TE_tw_first_word_...`, `TE_hashtags_media_tweet_type_language`, and TE of *binned numeric differences* such as "TE difference mode a_following_count + mode a_follower_count + tweet_type + language" [V]. In other words, they target-encoded **combinations of binned numeric features with categorical context**.
- **Count encoding (exact)** [V]: `CountEncoder.fit_transform(train, test, col)` counts over **train+test concatenated**. `count_encode_cudf_v2` emits both `CE_col_norm` (count/len within each split separately) and `CE_col` (count over train+valid together). Used on media, tweet_type, language, a_user_id, b_user_id.
- **Difference / lag encoding (exact)** [V]: `diff_encode_cudf_v1(df, col='b_user_id', tar=t, sort_col='timestamp', sft=±1)`. Sort by (user, time) and compute `t - t.shift(sft)`, zeroed when the previous/next row belongs to a different user. Used for `t ∈ {b_follower_count, b_following_count, language}` with shifts +1 and -1, which means **future rows are used too**. `add_timeshift(shift=±1)` gives the absolute time gap to the user's previous/next interaction (filled with 15 days). `diff_time` gives the time between the engaging user's own tweets and the current one. `diff_language` flags whether the tweet language differs from the user's most frequent language (computed on train+test).
- **Time-series features** [V]: "number of engagements of the engaging users in the last 5, 60, 240, 480 and 1440 minutes, the time past since an engaging user posted a tweet or since he/she interacted with another tweet."
- **Graph features** [V]: A follows B (1st/2nd degree) from partial follow info.
- **Final model** [V]: blend of 3 XGBoost models trained on different feature sets and data subsets. Per-target blend weights were tuned on public LB (Table 2, e.g. Like: 0.28/0.12/0.60). XGB hyperparameters were tuned by simple grid search.
- **Alternatives and measured outcomes** [V]:
  - LightGBM and CatBoost: "similar performance to XGBoost", abandoned for CPU speed.
  - FTRL-Proximal: 42.5 RCE overall (far below).
  - ALS user/item embeddings, with hashtags/links as "artificial items" because tweets are cold: +1.3 PR-AUC on validation for Like, smaller on test.
  - DistilBERT mean embedding (768 → 32 by averaging every 24 dims) added to XGB: no gain. Fine-tuned DistilBERT out-of-fold predictions: no gain.
  - FFNN on the same features: tried MinMax, standardization, power transform and GaussRank, and **MinMax was best**. They argued that for a shallow NN all categoricals should be target-encoded. Like RCE 60.7 vs XGBoost 3 62.6. It was "good performance, diversity and fast training" but not ensembled (built late).
- **GPU speed table** (Table 3, seconds) [V]:

  | step | pandas | 20-core CPU (dask) | 1×V100 | 4×V100 | 4×V100+UCX |
  |---|---|---|---|---|---|
  | target encoding | 30120 | 1785 | 302 | 108 | 46.4 |
  | count encoding | 540 | 485 | 50 | 18 | 7.5 |
  | lag features | 1380 | 121 | 62 | 7 | 3.3 |
  | train & predict | 6900 (LightGBM CPU) | 844 | 180 | 56 | 56 |
  | total | 38940 | 3570 | 1020 | 270 | 138 |

  "This design enabled us to create and explore hundreds of different features and to run thousands of experiments." [V]

### 2.2 Layer 6 + Cohere, 2nd place: "Predicting Twitter Engagement With Deep Language Models" (Volkovs et al.)
Source [V]: https://www.cs.toronto.edu/~mvolkovs/recsys2020_challenge.pdf
- **Data partitioning**: a 24h non-overlapping sliding window. Engagements in the window are targets; **all other engagements, including those forward in time, are used for feature extraction**. "We observe a considerable improvement in performance from applying this procedure as it significantly increases the amount of data available for feature extraction." 5 windows → 109.4M training rows; validation is the last 4h (3.6M rows), which "generalises well to the public leaderboard" [V].
- **467 features** in 4 groups: engaging user (history by engagement type, follower trends; "time since last engagement ... significant boost"), tweet creator, tweet content, and interactions. The interaction group covers content similarity between the target and the user's created/engaged tweets, user-creator joint history, and **CF similarity between rows/columns of the binary user×creator matrix**. Numeric features are Yeo-Johnson transformed and categoricals one-hot encoded for the NN [V].
- **History attention**: target-tweet embedding as query, last n=10 engaged tweets as keys/values, **separate attention per engagement type**, concatenated [V].
- **Ablation (public LB, avg PR-AUC / avg RCE)** [V]:

  | model | PR-AUC | RCE |
  |---|---|---|
  | 467 feats + XGBoost | 38.57 | 20.33 |
  | 467 feats + MLP | 38.30 | 20.26 |
  | + M-BERT emb | 39.21 | 21.24 |
  | + XLM-R emb | 39.48 | 21.71 |
  | + XLM-R + history attention | 39.74 | 21.74 |
  | blend | 40.28 | 22.52 |

  The MLP on the same features ≈ XGBoost. History attention added +0.26 PR-AUC in this content-dominated setting. The Like target overfit early, so they used a different early snapshot per target. Accuracy rises steeply with the first ~10 historical engagements and plateaus after ~20 (Fig. 3c) [V].
- The winner's private score was far ahead (NVIDIA 45.50 PR-AUC / 33.14 RCE vs Layer 6 40.33 / 23.23) [V]. That gap came mostly from feature engineering, i.e. TE and the leaky-but-allowed engagement structure.

---------------------------------------------------------------------------------------------------

## 3. RecSys Challenge 2021 (Twitter, ~1B pairs, 3 weeks train + val/test from week 4, 6 ms/example CPU limit)

### 3.1 NVIDIA, 1st place: "GPU Accelerated Boosted Trees and Deep Neural Networks for Better Recommender Systems" (Deotte, Liu, Schifferer, Titericz)
Source [V]: https://github.com/NVIDIA-Merlin/competitions/blob/main/RecSys2021_Challenge/GPU-Accelerated-Boosted-Trees-and-Deep-Neural-Networks-for-Better-Recommender-Systems.pdf and code `RecSys2021_Challenge/01_training`.

- **Setup**: >500 features tried, ≤151 used in one model. The validation set (same period as test) was released 2 weeks before the end; "stage 1" means before that and "stage 2" after.
- **TE variants (new vs 2020)** [V]:
  1. 5-fold OOF (as in 2020, smoothing 20).
  2. **Leave-one-out TE**.
  3. **Time-split TE**: the train data is split into a large part (TE statistics) and a small part, **20% of week 3**, on which XGBoost is trained. "As target encoding features can distill information of a large portion of the training dataset, a model such as XGBoost can be trained on a small sample of the training dataset" [V]. In code (giba), TE maps are built on weeks 1-2 plus 80% of week 3, and the model is trained on the held-out 20% of week 3.
  4. **Raw statistics instead of smoothed TE**: "we provided raw statistics (sum of positive interactions and count of observations) as features, so that the model can learn a smoothing function." [V] In code: `sums_te_reply-b_user_id-tweet_type`, `counts_te_reply-b_user_id-tweet_type`, etc.
  5. **Reversed TE**: author and reader ids are jointly label-encoded, so TE statistics of a user *as reader* are applied to the same user *as author* and vice versa ("if a reader has a high like probability, will his/her tweets be liked by other users?") [V]. Code: `TE_switch_*` (columns renamed a_user_id ↔ b_user_id before the merge).
  6. **Cross-target features**: each target's XGBoost gets the sums for all four targets of the same key. For example, the reply model uses `multi_reply, multi_retweet, multi_retweet_comment, multi_like, multi_counts` for key `[media, language, tweet_type, a_is_verified, b_is_verified, b_follows_a, tw_last_quest, decline, group]` [V].
  7. Combos used (giba XGB-3): `(a_user_id, tweet_type)`, `(b_user_id, tweet_type)`, `(tw_word0, tweet_type)`, `(tw_original_user0/1, tweet_type)`, `(tw_rt_user0)`, a big low-cardinality 9-way combo, and binned follower/following counts (`// 819`, `// 712`) × tweet_type × language × b_follows_a [V].
  8. Regularization details from the code [V]:
     - Chris's TE drops categories with count ≤ 1 (`FILTER=1`), uses `SMOOTH=20` and stores TE as **int8** (`te*255-128`).
     - Benny's stage-2 TE is `.round(3)` and sets author-Like TE to NaN when the author has ≤ 1000 observations.
     - **Validation-set TE uses LOO** inside the validation data: `(sum - y*is_train + mean*20) / (count - is_train + 20)` [V].
- **XGBoost model types (Table 3)** [V]:

  | | XGB 1 | XGB 2 | XGB 3 |
  |---|---|---|---|
  | TE strategy | 5-fold | LOO | time split |
  | #features | 70 | 77-79 | 69 |
  | #trees | 250 | 450-1000 | 1000 |
  | per user group | no | no | yes |
  | bags | 3 | 1 | 1 |
  | training data | 150M rows with users present in week 3 and a prior week | weeks 2+3 | 20% of week 3 |

  giba's XGB-3 params [V]: `max_depth 8, lr 0.04, subsample 0.5, sampling_method gradient_based, colsample_bytree 0.8, gpu_hist, 550 rounds`.
- **NN model types (Table 4)** [V]: three diverse NNs (TF/PyTorch), each with text module + categorical module + continuous module + MF dot products, concatenated → FC.
  - Token layer: mean-pool / GRU-64 / GRU-128 over 48 or 42 frozen multilingual BERT word-embedding tokens.
  - User embeddings: dims 96/64/128. Frequency threshold 5/7/3: users rarer than that share one embedding.
  - MF terms: author×reader, reader×tweet. Model sizes 3.9-7.4 GB. 2-3 epochs on 1-4 V100, 20-87 h.
  - "We sacrificed some single model performance in exchange for more diverse models, which was later proved to be key for stage 2 stacking." [V]
  - Bo's NN code trains with a frequency threshold of 25, then continues training with threshold 10, then 3. This is a curriculum that progressively un-merges rare-ID embeddings [V, `01_training/stage1/bo/README.md`].
- **Stage-1 results on validation (Table 1)** [V], with avg AP / avg RCE:
  - XGB1 0.315/18.05, XGB2 0.317/17.83, XGB3 0.306/16.68.
  - NN1 0.318/17.30, NN2 0.319/17.59, NN3 0.297/14.04.
  - **Weighted-average ensemble 0.337/20.20**, i.e. +0.018 AP over the best single.
- **Stage 2 (stacking on the newer data)** [V]:
  - "We experimented with both approaches [retraining / continuous training] and experienced the best performance by stacking." Stage-2 XGBoost inputs: train-derived TE features, stage-1 predictions, and new TE and CE computed on the validation set.
  - "The stage 1 models were added one-by-one using a greedy approach ... the best performing version is to leverage all six stage 1 models."
  - Local CV: an 80/20 random split of validation. Two stage-2 XGB, each bagged 3×. Results: XGB4 0.372/23.56, XGB5 0.373/23.69, average **0.377/24.05** local; final LB 0.379/24.356.
  - Caveat: Table 1 is on the full val set and Table 2 on a 20% holdout, so the 20.2 → 24.05 gain is indicative, not a controlled ablation. The text says "23.054", a typo for the table's 24.054.
  - Stage-2 XGB params (Chris): `max_depth 8, lr 0.1, subsample 0.8, colsample 0.8, gpu_hist`, early stopping 25 [V, code].
- **GPU inference** [V]: preprocess 4155 s → 69 s; feature engineering 10334 s → 119 s; XGB via cuML FIL 7193 s → 43 s (168×); NN 8540 s → 100 s; overall 92× vs 8-core CPU, 257× vs 1 core, cost −88%.

### 3.2 Synerise AI, 2nd place: "Twitter user engagement prediction with a fast neural model"
Source [V]: https://arxiv.org/pdf/2109.12985
- **Data partitioning** [V]: 24h non-overlapping windows over the 21 train days (targets = one day; features = all other days, *including the future*). The released validation set was **split into 10 random parts, each used once as targets with features from the other 9**. The model trains on train, then is fine-tuned on validation (2 epochs train ~24 h on one V100, 3 epochs val ~60 min). AdamW lr 1e-4, wd 0.01, batch 256, linear decay.
- **Features** [V]:
  - Counts per reaction type between the (author, reader) pair, per author and per reader.
  - Reader interactions with tweets of the same language as the target, and with the target's hashtags.
  - Interactions of the reader with **users similar to the author** (Jaccard of follower sets above a threshold).
  - **Leiden community detection** on 4 interaction graphs (same-partition flag plus inverse partition size).
  - **Counts of interactions with the same tweet id in the validation set** (tweets recur between val and test).
  - Account features, content features, and an EMDE (LSH-based density sketch) of fine-tuned DistilBERT token embeddings.
- **Model** [V]: 3-layer residual MLP, 1500 units, leaky ReLU, BN, 4 sigmoid outputs. **Fourier Feature Encoding** maps each numeric x to 16 dims, `sin/cos(x / 2^s)` for s ∈ {-1..6}. The authors note it "does not usually offer significant performance gains but it facilitates the usage of continuous features".
- **Ablation (Table 2)** [V]:
  - Interactions-only → +content: AP ↑11.8% avg, RCE ↑17.4% avg.
  - +account features: +0.5% AP / +1% RCE.
  - +Leiden clusters: avg AP 0.362 → 0.365, RCE 23.0 → 23.4.
  - Removing EMDE (mean token embedding instead): −2.8% AP, −2.3% RCE.
- Final LB [V]: NVIDIA, then Synerise, then Layer 6 (Table 1). For example, retweet AP was 0.4614 / 0.4514 / 0.4317.

---------------------------------------------------------------------------------------------------

## 4. RecSys Challenge 2022 (Dressipi fashion, session → purchased item, MRR@100)

Final top-5 [V, Layer 6 paper Table 3]: 1. zzh 0.2160, 2. Layer 6 0.2148, 3. NVIDIA RAPIDS.AI 0.2086, 4. MooreWins 0.2076, 5. THLUO 0.2062. No paper from "zzh" was found (searched).

### 4.1 Layer 6, 2nd place: "Session-based Recommendation with Transformers"
Source [V]: https://www.cs.toronto.edu/~mvolkovs/Recsys_2022.pdf
- **Validation**: last month of the 17 train months, with the same random truncation (≤50% of items dropped) as the test set; "generalizes well to the public leaderboard". After tuning, they retrain on all 17 months [V].
- **Features (88)** [V]:
  - 73 categorical item attributes (embedding size 8).
  - Item purchase/view counts over **last month, last 2 months, all time** (6 features).
  - 4 normalized **co-occurrence** features (candidate vs session items, and between session items). These are computed **from the previous month only** for each training month, because of strong time dependence.
- **Model**: [mask]-token Transformer (d=128, FF 256, 4 layers, 4 heads, dropout 0.1, max length 100), learned positional and month embeddings. Loss = α·MSM (masked session modeling, 15% mask, weight-tied softmax restricted to items active that month) + PIP (sigmoid classifier on h0 concatenated with extracted features, 19 sampled negatives). AdamW, warmup 2000 to 1e-3, cosine [V].
- **Measured** [V]:
  - LB MRR: XGBoost 0.1993, VAE 0.2026, DAE 0.2035, **Transformer 0.2121**.
  - α ablation: MSM 0.2109, PIP 0.2030, MSM+PIP(α=0.1) 0.2121, which beats α=0.2/0.5/1.0 (0.2115/0.2108/0.2104).
  - The number of negatives in PIP peaks at **19 (20:1)**. MRR rises with max session length up to 100.
  - The self-supervised term "allow[s] to use a much larger model size, around 10x larger than the baselines, without overfitting." Final submission: linear ensemble of all four.

### 4.2 NVIDIA, 3rd place: "A Diverse Models Ensemble for Fashion Session-Based Recommendation"
Source [V]: https://github.com/NVIDIA-Merlin/competitions/blob/main/RecSys2022_Challenge/A-Diverse-Models-Ensemble-for-Fashion-Session-Based-Recommendation.pdf
- **Data augmentation was "our most important preprocessing technique"**, making training data 5-10× larger: truncation 50-100%, swapping the target with one of the last N viewed items (N=1-3), shuffling the last N, and combinations. Applied to train only [V].
- **Feature selection by stability**: keep item features present in a large fraction of months, i.e. with a stable distribution over time [V].
- **Stage 1**: 17 models (conditional popularity, session-kNN ×2, LightGBM lambdarank and XGBoost binary on 200 candidates/session, MLPs, GRUs, XLNet Transformers), bagged 6× per NN [V].
  - GBDT features: purchase/view counts over the last 30 days; min/mean/max/std of the co-occurrence frequency of (session item, candidate) over 28 days; unique sessions per candidate.
- **Stage 2** [V]:
  - Optimized weighted average (weights fit on validation MRR).
  - LightGBM lambdarank on candidates recommended by ≥2 stage-1 models, **25 bags**, each an 85/15 random session split.
  - XGBoost binary with 4 folds of consecutive days, 3 seeds.
  - Extra stage-2 features: per-attribute "candidate shares value with last viewed item" flags, and the % of attributes shared with all session items.
- **Stage 3**: weighted average 0.5 (opt-avg) / 0.16 (LGBM) / 0.32 (XGB), chosen conservatively, "lowest risk of overfitting". LB: 0.2073 / 0.2074 / 0.2076 → **0.2082** public, 0.2086 final [V].
- **Other measured tricks** [V]:
  - Latent-cross event-type context: +0.003 validation MRR.
  - Multi-task (also predict item attributes): +0.001 (MLP) and +0.002 (Transformer).
  - RNN "sample rolling": drop the oldest 20% more each epoch so recent data is seen more often, because simply removing old data hurt.
  - Mask viewed items (the target is never in the session).
  - Best single RNN 0.2028 LB, average of variants 0.2044.

---------------------------------------------------------------------------------------------------

## 5. RecSys Challenge 2023 (ShareChat ad installs; anonymized features f_0..f_79; normalized cross-entropy)

### 5.1 Layer 6, 1st place: "Robust User Engagement Modeling With Transformers and Self Supervision" (Lu, Volkovs)
Source [V]: https://www.cs.toronto.edu/~mvolkovs/Recsys_2023.pdf
- **Final top-5** [V]: Layer 6 5.744062; LearningFE 5.892977; hahaha 5.904369; Ainvest 5.949816; Shield 5.958641. Over 2.5% relative better than 2nd.
- **Input** [V]: every feature becomes a token embedding. Categorical and binary features use standard embeddings. Numeric features use **quantile binning** (equal-frequency bins fit on train; a feature with fewer than t unique values gets one bin per value), then an embedding. A [CLS] token is prepended. 6 Transformer blocks, d=128, FF 128, 8 heads.
- **SSL** [V]: two augmented views per instance, SimCLR-style head, InfoNCE with cosine similarity and in-batch negatives.
  - Feature corruption: replace 5-25% of features with values sampled uniformly from that feature's training values.
  - Embedding perturbation: per-feature Bernoulli masks, rate 0.05-0.25.
  - Model perturbation: two dropout passes.
  - Training: joint loss with α=0.6 (dropout 0.1), then supervised-only fine-tuning (dropout 0.05). AdamW, batch 4096, warmup 2000 to 3e-4, cosine. Validation = last train day (3.39M train / 98K val / 161K test).
  - HP search with TPE across GPUs, **each HP setting evaluated as a 5-seed ensemble**, then retrain on all 21 days. Hardware: 8×A100. Final submission: linear blend of seeds/models.
- **Ablation (validation log loss, lower is better)** [V]:

  | method | log loss |
  |---|---|
  | LightGBM (extensively tuned) | 0.3615 |
  | XGBoost (extensively tuned) | 0.3582 |
  | Transformer supervised | 0.3471 |
  | + augmentation (supervised) | 0.3448 |
  | value estimation SSL (VIME) | 0.3385 |
  | value + mask estimation | 0.3353 |
  | **joint supervised + contrastive** | **0.3297** |
  | contrastive with FC only / EP only / MP only | 0.3417 / 0.3438 / 0.3455 |
  | FC+EP | 0.3337 |

  "Across all experiments we were unable to achieve competitive results with the supervised only setting so self-supervised learning is essential for high capacity models." The supervised-only model overfits after epoch 3 (Fig. 2) [V].

### 5.2 Upstage (Kim & Lee), 9th place: "Lightweight Boosting Models for User Response Prediction Using Adversarial Validation"
Source [V]: https://arxiv.org/pdf/2310.03778
- **Adversarial validation per feature** (train vs test classifier on each feature alone); they dropped features with AUC ≥ 0.75 [V].
- **Denoising anonymized continuous features** [V]. Groups of features had sorted unique values forming arithmetic sequences (Δ≈0.0385 for f42,f52-57,f74-76; Δ≈0.5711 for f44-50,f71-73), "presumed to be artificial noise injection". Dividing by Δ gives integers, "more descriptive and informative when crossing features".
- **Frequency encoding**: compared counts over the previous day, the previous week, and all data up to the previous day; **the previous-week count was best**. CatBoost (ordered, time-aware) encoding fit on data up to the previous day, for the click and install targets [V].
- **Measured (local log loss / LB NCE)** [V]:
  - Vanilla LightGBM 0.375506 / 6.466970.
  - + frequency encoding 0.368485 / 6.264767.
  - + denoising 0.361400 / 6.218796.
  - + CatBoost encoding + FE 0.361309 / 6.071975.
  - + HP tuning **0.360991 / 6.059065** (num_leaves 491, depth -1, 10000 iters, ES 100).
  - CatBoost: 0.359850 local but a worse LB of 6.106765. TabNet 6.221323.

(Polimi "Gabibboost", 1st academic / 9th overall: CatBoost+LightGBM+HistGB ensemble with a distribution-shift post-processing and "pessimistic rescaling" [S, core.ac.uk abstract]; the PDF was not retrievable.)

---------------------------------------------------------------------------------------------------

## 6. NVIDIA playbooks (KGMON / Merlin / RAPIDS)

### 6.1 RecSys 2020 tutorial "Feature Engineering for Recommender Systems" (Deotte, Schifferer, Oldridge)
Source [V]: https://github.com/rapidsai/deeplearning/tree/main/RecSys2020Tutorial (notebook markdown read)
- **Combining categories**: start by crossing low-cardinality categoricals (10-50 values on 100M rows). Watch observation count per crossed value, because more columns means fewer observations per value. Measure information gain on samples quickly with cuDF [V].
- **Target encoding**: smoothing formula above (w=20 in the exercise). OOF: "split the training dataset in k-folds and encode the i-th fold by using all folds except of the i-th one; use the full training dataset to encode the validation/test dataset". "The k-fold can be generated by a random split or by a timestamp depending on the dataset." The smoothed version has a lower error for categories with 1-3 observations [V].
- **Count encoding**: three options: fit on train and apply to val; count each split separately; or **count on concatenated train+val** (the last is fine in competitions; only the first is valid in industry) [V].
- **Binning**: bin price *per category group* by within-category quantiles (cat_0×cat_1×cat_2), because "what is a good deal" depends on the category [V]. Analog for us: within-category quantile bins of f1..f5, in addition to the within-feed ranks we already have.
- **GaussRank**: rank → scale to [-1,1] → erfinv, for NN inputs (80× GPU speedup) [V].
- **Historical window features**: rolling sums/means/probabilities per item over the last 1/3/7 days, **shifted by one row to avoid including the current row (leakage)** [V].
- **Differences**: e.g. price change versus the previous record of the same item; week-over-week purchase-count differences [V].
- Reported GPU speedups: TE 11.6×, CE 15.8×, binning 28.2×, GaussRank 80.3×, rolling window 372× [V].

### 6.2 "The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data" (NVIDIA blog, Sep 2025)
Source [V]: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data
- Foundations: a fast GPU pipeline (cuDF, GPU XGBoost/LightGBM/CatBoost). Match CV to the test structure (GroupKFold for users, TimeSeriesSplit for time).
- 1) EDA for train/test shift and temporal target patterns (Amazon KDD Cup '23 example).
- 2) Diverse baselines (linear, GBDT, small NN). Re-run baselines after data changes to catch leakage.
- 3) Large-scale feature generation: all pairwise categorical crosses (8 cats → 28 features); "thousands of engineered features" won the Backpack and Insurance playgrounds.
- 4) **Hill climbing**: start from the best model, add models/weights only if CV improves, repeat. Vectorized on GPU with CuPy to score thousands of weight combinations.
- 5) **Stacking**: on OOF predictions or on residuals. Example: 3-level stack, L1 = Lasso/SVR/KNN/RF/MLP/TabPFN/XGB/LGBM, L2 = XGB + MLP, L3 = weighted average.
- 6) **Pseudo-labeling**: soft labels, filter low confidence, k pseudo-label sets for k folds (no leakage), "pretrain on pseudo-labels, then fine-tune on the original data".
- 7) **Extra training**: multi-seed (Playground S5E6: 100 XGB seeds gave ~0.379 MAP@3 vs ~0.376 single-seed average), then full-data refit after tuning [V].

### 6.3 "Grandmaster Pro Tip: Winning First Place in Kaggle Competition with Feature Engineering Using NVIDIA cuDF-pandas" (Deotte, Backpack price)
Source [V]: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas
- **Over 10,000 engineered features were generated and tested; the best 500 went into XGBoost** (1st place).
- Recipes:
  - `groupby(COL1)[COL2].agg(STAT)` with STAT ∈ {mean, std, count, min, max, nunique, skew} (target as COL2 means nested-CV TE).
  - Per-group histogram bins (7 bins).
  - Per-group quantiles [5,10,40,45,55,60,90,95].
  - NaN-pattern bitmask.
  - Rounding the key numeric feature to several precisions.
  - Digit extraction.
  - All 28 categorical pairs.
  - Ratios of aggregates (count/nunique, std/count).
  - Reference value from an external/original dataset [V].
- Analog for us: per-category and per-(category, day) quantiles/histograms of f1..f5. Per-item aggregates of f's across all feeds where the item appears (mean/std/min/max/nunique over appearances in train+val+test). Deviations of the candidate's f from its item mean and from its category-in-feed mean.

### 6.4 Other NVIDIA RecSys wins (brief)
- **WSDM WebTour 2021 (Booking.com), 1st** [V, https://developer.nvidia.com/blog/how-to-build-a-winning-deep-learning-powered-recommender-system-part-3/]:
  - Accuracy@4 0.5939, 2.8% better than 2nd.
  - Features: conditional probabilities P(city|month), P(city|month,country), P(city|week), P(city|week,country). Rare cities (<9 reservations) grouped (44K → 11K ids). Reversed trips as augmentation (2× data).
  - Models: 3 architectures (MLP-SMF, GRU-MS-SMF with a 2-stage top-50 rerank head, XLNet-SMF), 5-fold CV **using test sequences too** (test trips lack only the final city). Bagging 8/7/5. Pairwise prediction correlation ~80% gave the ensemble gain.
- **Amazon KDD Cup 2023, 1st on all 3 tasks** [V, abstract only via OpenReview API, https://openreview.net/forum?id=J3wj55kK5t]: "pipeline of candidate generation, reranking, and ensemble ... Candidate sets were merged and ranked using gradient boosting (XGBoost and CatBoost) to maximize MRR". The full PDF was blocked by an OpenReview challenge. [S]: Dask-XGBoost across the GPUs; cuML for nearest-neighbour search over product embeddings.
- **SIGIR eCom 2021 purchase intent (2nd)** [V, `SIGIR_eCommerce_Challenge_2021/task2_purchase_prediction/README.md`]:
  - Only `product_id` was target-encoded in the final model (OOF, NVTabular); other categoricals label-encoded.
  - Similarity features: mean/std cosine between the added-to-cart product embedding and the other session items, before/after the add event.
  - Optuna 100-trial XGB search.

---------------------------------------------------------------------------------------------------

## 7. Concrete recipes for Avito (implementation notes + how to validate)

All validation is GroupKFold-by-user on val (our current CV), reporting OOF NDCG. Also report on a time-split of val (last day) to mimic public → private. Each item below says which source it comes from.

### 7.1 Stage-1 on train, stage-2 on val (NVIDIA 2021 §3.1; Layer 6 2020 partitioning §2.2)
- Stage-1 candidates, all trained on 247M production rows with labels debiased for position:
  - (a) LGBM/XGB lambdarank and binary, using our TTE features and `pos` as a feature set to a constant at inference, or IPS weights [K].
  - (b) The same with a different TE strategy (time-split: compute TE on days 1-10, train on days 11-14 only, à la XGB-3 2021).
  - (c) A history-attention NN (7.5).
  - (d) A two-tower/dot model fine-tuned from the provided embeddings.
- Predict val and test, then run the stage-2 LGBM (lambdarank and binary, both) on val. Inputs: stage-1 scores (raw + within-feed rank/z), train-TE features, **val-internal OOF TE** (fold-out inside GroupKFold; LOO for test using all val), and CE.
- Add stage-1 models greedily and keep a model only if OOF NDCG rises (NVIDIA's greedy add).
- Alternatively use stage-1 as `init_score` and continue boosting on val (already supported in `stack_val.py`). NVIDIA reported stacking > retraining/fine-tuning for the new period [V].
- Partitioning trick for computing train features: for each train day d, features from **all other days (past and future)**, not only the past. Layer 6 reports a "considerable improvement" [V]. This applies when building features for *training rows*. For val/test rows the features come from all of train, so the feature distribution stays similar. Compare against our expanding-window version.

### 7.2 Target-encoding variants worth adding (NVIDIA 2020/2021)
- Keys that vary within a feed:
  - `item`, `category`, `(category, user_cluster_K)` and `(item, user_cluster_K)` for K ∈ {64, 256, 1024} (k-means on user embeddings on GPU).
  - `(category, user_top1_hist_category)`, `(category, user_top1_hist_event_source)`.
  - `(item, feed_size_bucket)`.
  - binned f-values × category (quantile bins per category, NVIDIA-style "TE of binned numeric + context").
- Forms: smoothed (w=20 as the default; tune w ∈ {5, 20, 100}), **raw (pos_sum, imp_count, expected_clicks) triples**, and COEC = clicks/expected-clicks-at-position [K, already in `features_tte.py`]. Optional min-count NaN masking and rounding/int8 quantization as regularizers.
- **Multi-target**: one TE per "target" from the histories. Over all users' histories, compute per item and per category the counts of each event_type (1..10) and event_source (0..26), plus ratios like (contact-like events)/(all events). This is the analog of NVIDIA giving every target's TE to every model. The history is not filtered by split, so this is legal as long as each feed's own future is not used (check `eda/history_leak`).
- **"Reversed" TE analog**: items appear both as candidates (with labels) and as history items (with event types). Join candidate-side TE onto history items and aggregate per user, then cross with the candidate. Example feature: `sim-weighted mean candidate-CTR of the user's history items in the candidate's category`.

### 7.3 Count encodings and time-local counts (NVIDIA 2020 code; Synerise 2021)
- `CE_item_all` = item appearances across train+val+test feeds; `CE_item_split` = appearances within the same split, normalized by split size.
- Time-window counts around the feed timestamp: appearances in feeds within ±1h/±6h/±24h, and appearances in test-period histories.
- Count of the item in histories by event type (popularity); we already have the overall version.
- Within-feed rank of each CE (popularity rank inside the feed).
- If an item's f's vary over time, add `DE_item_fk_±1`: f minus f at the item's previous/next appearance, sorted by timestamp (NVIDIA `diff_encode_cudf_v1`).
- Add `time_since_item_first_seen` and `time_to_item_next_seen`.

### 7.4 Anonymous features f1..f5 (Kim&Lee 2023; NVIDIA tutorial binning)
- For each f: sort unique values per (category, day). Test for constant Δ and integer recoverability. If they recover to counters, build ratios (CTR-like) and log-counts.
- Per-category quantile bins. Per-item aggregates across appearances (mean/std/min/max), and deviation of the current value from the item mean (Backpack recipe).
- Adversarial validation of train vs val/test per feature, and of val vs test. Drop or neutralize features with large shift. Production-only artifacts (blender/category sampling, position) are the expected culprits.

### 7.5 History features and the NN (Layer 6 2020; Synerise 2021; Layer 6 2022/2023)
- Per event_type e (and per coarse event_source group):
  - max / top-3-mean / mean cosine of the candidate with history items of type e;
  - the same restricted to the last 8/32/128 events (position in history as a recency proxy);
  - count of history events of type e in the candidate's category;
  - the cosine rank of the candidate within the feed for each of these.
  Layer 6 found per-type attention useful; Synerise found per-type counts useful [V].
- **Item-item co-occurrence from histories**: a co-visitation matrix from all user histories (consecutive events within a window, weighted by event type), then the candidate's score against the user's last N history items. This is the strongest feature family in Layer 6 2022 and NVIDIA 2022 GBDTs [V], and in OTTO-type recsys [K].
- NN: target-attention over history (query = candidate embedding plus learned features) with separate heads per event type, plus a set-transformer over the feed's candidates (listwise). Use quantile-bin embeddings for numerics (Layer 6 2023) or MinMax (NVIDIA 2020 found it best for a shallow FFNN) [V].
  - Pretrain on production train (pos as input, neutralized at inference), then fine-tune on val folds (Synerise: 2 epochs pretrain, 3 epochs fine-tune [V]).
  - Add a contrastive SSL term (feature corruption 5-25%, embedding masks 5-25%, dropout twins; α≈0.6, then fine-tune) to control overfitting on 1.4M val rows [V].
  - For the rare-ID user embeddings: a frequency threshold plus the NVIDIA curriculum 25 → 10 → 3 [V].
- Optional SSL pretraining on **test-user histories** (unlabeled, transductive), e.g. masked next-event prediction. This is the analog of NVIDIA using test trip sequences in WSDM CV [V], and Layer 6's MSM [V].

### 7.6 Ensembling and refit (Grandmasters Playbook; NVIDIA 2021/2022)
- Bag every stage-2 model over ≥10-20 seeds and fold splits; the playbook shows gains up to 100 seeds [V]. NVIDIA 2022 used 25 LGBM bags in stage 2 [V].
- Hill-climb weights on OOF NDCG over rank-normalized predictions within feed. Keep weights conservative, as in the NVIDIA 2022 stage-3 choice [V].
- Refit on 100% of val with the tuned number of rounds (×1.1-1.25 for the larger data [K]).
- Do not tune blend weights on public LB. NVIDIA 2020 had to, because of poor val/test correlation [V]; our public is ~8.6K feeds (SE ~0.003).
- Break ties: full-list NDCG uses pessimistic ties, so add a tiny secondary score (e.g. the stage-1 prediction) [K].

### 7.7 GPU and compute plan (NVIDIA 2020/2021)
- cuDF (or polars on a shared machine) for all groupby TE/CE.
- XGBoost `device=cuda` with `rank:ndcg` / `rank:pairwise` on full train; Dask-XGBoost across the 4 GPU.
- CatBoost GPU YetiRank for diversity [K].
- Generate hundreds of TE crosses and keep them by greedy/importance selection, following the Backpack "10,000 → 500" pattern [V].

---------------------------------------------------------------------------------------------------

## 8. Unverified / not found
- 1st place RecSys 2022 ("zzh"): no paper found.
- RecSys 2023 2nd ("LearningFE") / 3rd ("hahaha"): no papers found.
- NVIDIA KDD Cup '23 feature list: only the abstract was readable.
- Medium writeups by NVIDIA for RecSys 2020/2021 (rapids-ai, nvidia-merlin) were blocked (403); the papers and code cover them.
- RecSys Challenge 2025 (Synerise universal behavioral modeling; 5 event types, very close to our history structure): only snippets were seen. 2nd place used a GRU sequence autoencoder; 4th (Yandex) used a heterogeneous causal transformer with multi-task training, concatenated with engineered features [S, https://arxiv.org/abs/2508.06970]. Worth a separate read for history encoders.

## 9. Sources read this session
- https://github.com/NVIDIA-Merlin/competitions (RecSys2020_Challenge/RecSysChallenge2020.pdf + `02_ModelsCompetition/XGBoost2/sub040_1_save_public.py` + `03_ModelsBenchmark/GPU/Demo-e2e-MultiGPU_4xV100.ipynb`; RecSys2021_Challenge PDF + `01_training/stage1/{giba,chris,bo}` + `01_training/stage2/{benny,chris}`; RecSys2022_Challenge PDF; SIGIR_eCommerce_Challenge_2021/task2 README)
- https://github.com/rapidsai/deeplearning/tree/main/RecSys2020Tutorial (notebooks 03_1, 03_3, 03_4, 04_1, 04_3, 05_1, 05_2)
- https://www.cs.toronto.edu/~mvolkovs/recsys2020_challenge.pdf (Layer 6, RecSys 2020 2nd)
- https://arxiv.org/pdf/2109.12985 (Synerise, RecSys 2021 2nd)
- https://www.cs.toronto.edu/~mvolkovs/Recsys_2022.pdf (Layer 6, RecSys 2022 2nd)
- https://www.cs.toronto.edu/~mvolkovs/Recsys_2023.pdf (Layer 6, RecSys 2023 1st)
- https://arxiv.org/pdf/2310.03778 (Upstage, RecSys 2023 9th)
- https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data
- https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas
- https://developer.nvidia.com/blog/how-to-build-a-winning-deep-learning-powered-recommender-system-part-3/
- https://developer.nvidia.com/blog/nvidia-earns-1st-place-in-recsys-challenge-2021/ (announcement only, no technical content)
- https://openreview.net/forum?id=J3wj55kK5t (abstract via api2.openreview.net)
- https://github.com/recsyspolimi/recsys-challenge-2023-sharechat (README only)
