# Deep dive: RecSys Challenge 2024 (Ekstra Bladet, EB-NeRD), and what carries over to Avito personal ranking

Date: 2026-11-11.

Tags: [V] = I read the primary source in this session (URL given). [S] = seen only in a search snippet. [K] = background knowledge, not checked this session.
"Code" means I read the released solution code myself (cloned from GitHub). The ACM full-text PDFs (dl.acm.org, DOIs 10.1145/3687151.*) returned HTTP 403 and were **not** read. For those papers I only read the abstracts, via the Semantic Scholar API.

---------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **The 2024 leaderboard was decided by transductive "leak" features, not by modelling.**
   - The 1st-place Transformer scores test AUC 0.8864 with all features.
   - Without article statistics computed over the future (L1) and without future impressions (L2), it drops to 0.8680.
   - Without any test-period impressions (L3 removed too), it drops to 0.7699 [V slides p.26].
   - The organizers' table shows every top team losing 2–11.7 AUC points without these features. Without them, the 2nd-place team ranks best (82.20 vs 76.99 for 1st place) [V arXiv 2409.20483].
2. **Checked on our data this session: that driver does NOT carry over.** Exploration traffic is too sparse, and candidate items almost never repeat across val/test feeds (1.24M unique items among 1.39M val candidates). Time-windowed transductive popularity from other val/test feeds' recent history:
   - covers at most 0.6% of candidates for ±1 h windows, and adds nothing;
   - even with all positions and all time it covers 8.5% and adds +0.003 NDCG over f1;
   - our existing global `it_hist_all` adds +0.009 (§7.1).
   - Restricting train item statistics to the last 1, 3 or 7 days is also not better than using all 14 days (§7.2).
3. **What does carry over is the pipeline design of the 1st place (team :D, DeNA).**
   - (a) Stage-1 models trained on the big "other" period.
   - (b) A stage-2 LightGBM-lambdarank stacker trained on the target-distribution period. Its features are the stage-1 scores with within-impression z-score, min-max, rank and normalized rank, plus pairwise differences and ratios between models.
   - (c) An Optuna-weighted mean, averaged with the stacker.
   - (d) Retraining or fine-tuning with the most recent labelled period included: +0.0097 to +0.0123 test AUC [V].
   - (e) Pseudo-labelled test impressions with soft ensemble labels for the last neural model [V code].
   - (f) Training-time regularizers: at most 3 impressions per user per epoch, and random history dropout [V code].
   - Ensemble vs best single model: +0.0060 test AUC (0.8864 → 0.8924) and +0.0068 val [V].
4. **The listwise Transformer's +0.043 AUC over an MLP (DeNA ablation) should not be expected here.**
   - Over GBDTs that already have within-impression rank features, it was only +0.0047 to +0.007 [V].
   - Most of the gap to the MLP comes from comparing transductive popularity across candidates, and our data does not have that signal.
   - Our own NN logs agree: at step 16000, removing set attention costs 0.0003 NDCG (0.3693 vs 0.3696), while removing history costs about 0.0085 (0.3611) (§7.3).
   - So invest in user-history modelling. That was the strongest non-leak approach in 2024 (BlackPearl HUIM, 82.20 without leaks).
5. **Cheap feature ideas worth a test** (unverified for our data):
   - FeatureSalad-style contextual normalizations: x/max, x/sum, minus median, feed-level std/skew/entropy, number of distinct categories in the feed.
   - BlackPearl's top feature family: statistics of the sizes of the feeds an item appears in, ranked within the feed.
   - An item-age proxy (time since first seen). Time-since-publication was the #1 GBDT feature for :D.
   - CatBoost `QueryCrossEntropy`, which was BlackPearl's best single tree model.

---------------------------------------------------------------------------------------------------

## 1. Task comparison: EB-NeRD vs ours

| | RecSys 2024 / EB-NeRD | Avito personal ranking (ours) |
|---|---|---|
| Unit | impression (in-view article list), NP-ratio about 10.5 [V 2410.03432] | feed of 5–300 candidates (mean about 96), 1.6% positives |
| Display order | "Impressions were shuffled to remove positional bias" [V 2410.03432] | train: real production positions (`pos`); val/test: uniformly shuffled |
| Label | click | target action (contact-like) |
| Metric | per-impression AUC (+MRR, nDCG@5/10) | per-feed full-list NDCG, binary gain, pessimistic ties |
| Splits | train / val / test = consecutive 7-day behaviour logs, plus 21-day click histories [V] | train 14 d (Sep 6–19), val 3 d (Sep 22–24), test 4 d (Sep 26–29), 1-day gap before test [V local readme] |
| Volume | 37.97M impressions, 1.10M users, 125.5K articles [V] | 3.15M train feeds, 14.4K val, 28.6K test |
| Item repetition | very high: few articles, heavily re-shown | very low in val/test: 1.24M unique items in 1.39M val candidates; 2.38M in 2.90M test [V, my check] |
| Item side info | text, topics, NER, publish time, future totals (inviews/pageviews/read time) | f1..f5, category (54), int8 128-d embedding. No item timestamp; `item_id` is not time-ordered (Spearman(item_id, first-seen time) = 0.00002 [V, my check]) |
| History | clicked articles with timestamps, read time, scroll | up to 256 events, newest first, event_type and event_source, no timestamps |
| Users per split | about 34 impressions per user | about 1.05 feeds per user in val/test |

The structural difference that matters most: EB-NeRD test impressions are dense in time, and articles are re-shown thousands of times. That is why counting an article's other exposures in the same minutes was so powerful there. Our val/test is a thin random sample of exploration traffic.

---------------------------------------------------------------------------------------------------

## 2. Organizers' findings

Overview paper: https://arxiv.org/html/2409.20483v1 [V]. Dataset paper: https://arxiv.org/html/2410.03432v1 [V].

- **Participation:** 145 teams (11 academic), 202 participants, 774 submissions [V].
- **Final leaderboard (AUC):**
  - 1 `:D` 89.24 (MRR 73.27, nDCG@5 79.05, nDCG@10 79.85)
  - 2 BlackPearl 88.15
  - 3 Tom3TK 87.07
  - 4 axi2017 86.92
  - 5 hrec 86.67
  - 6 FeatureSalad 85.13 (top academic)
  - Academic #2 DArgk 77.13 [V]
- **Baselines (AUC):**
  - Organizers' simple baselines: Clicks popularity 59.70, Read-time 59.49, In-view 54.50, Random 49.98 [V 2409.20483].
  - In the EB-NeRD paper: DIN (FuxiCTR) 71.53, NRMS 61.03 [V 2410.03432].
  - FuxiCTR DIN reached test AUC 0.7154 with no leak features [V https://github.com/reczoo/RecSys2024_CTR_Challenge].
- **Leakage policy.** Features containing future information, such as "how popular the article was going to be", were allowed. Papers had to report results with and without them [V]. Measured AUC with all features (++) vs without future information (−):

| Team | ++ | − | Δ |
|---|---|---|---|
| :D | 88.64 | 76.99 | 11.65 |
| BlackPearl | 88.15 | 82.20 | 5.95 |
| Tom3TK | 85.25 | 77.75 | 7.50 |
| axi2017 | 86.92 | 80.15 | 6.77 |
| hrec | 82.94 | 80.91 | 2.03 |
| FeatureSalad | 81.13 | 79.01 | 2.12 |

  [V] Note: the ++ values for :D, Tom3TK, hrec and FeatureSalad differ from their leaderboard scores; the organizers do not explain this. :D's 88.64 equals its single Transformer's test AUC.
- **Other stated findings.**
  - Collaborative filtering was weak "likely due to the severe cold-start problem".
  - Most top solutions are ensembles of GBDTs (LightGBM/CatBoost) and deep models, including Transformers.
  - Models with similar AUC produce very different category mixes (beyond-accuracy analysis) [V].

---------------------------------------------------------------------------------------------------

## 3. First place, team ":D" (DeNA: Fujikawa, Murakami, Sugawara): full breakdown

Sources:
- Slides: https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning [V]. Text via fetch; I also viewed slide images for pages 23, 26, 30, 31 and 32.
- JSAI2025 deck: https://speakerdeck.com/unonao/jsai2025-recsyschallenge2024-you-sheng-bao-gao [V].
- Code: https://github.com/k-fujikawa/recsys-challenge-2024-1st-place [V code].
- Paper: RecSys Challenge '24, pp. 42–47, doi 10.1145/3687151.3687160 (abstract only [V]).

### 3.1 Data splitting / recency
- Tune hyperparameters on train→val. Then retrain on train+val (5/18–6/1) with the same hyperparameters and predict test [V].
- Test AUC, training on 5/18–5/24 → 5/18–6/1:

| Model | Train on 5/18–5/24 | Train on 5/18–6/1 | Gain |
|---|---|---|---|
| Transformer | 0.8764 | 0.8864 | +0.0100 |
| LightGBM | 0.8694 | 0.8817 | +0.0123 |
| CatBoost | 0.8691 | 0.8805 | +0.0114 |
| Ensemble | 0.8827 | 0.8924 | +0.0097 |

  [V slide 23; gains computed by me]
- GBDT training code [V code: `kami/experiments/015_train_third/run.py`]:
  - "first" stage: fit on train, early-stop on val, predict val.
  - "second" stage: fit on val only, using `best_iteration` from first.
  - "third" stage: fit on train+val, using `best_iteration` from first, predict test (used in the final submission).
- Local validation:
  - Full validation AUC takes 4–5 hours on 12M impressions.
  - They used `ebnerd_small` (244K impressions, sampled per user), which correlated strongly with full-set AUC [V slide 30].
  - GBDTs were trained on a 20% random sample of impressions [V config `sampling_rate: 0.2`].
  - kami: "final model trained on about 20% of the data performed about as well as the full set" [V https://speakerdeck.com/unonao/n-equals-1-notui-jian-xi-konpenozhan-ifang].

### 3.2 Leakage taxonomy and its measured contribution
- **L1:** host-provided article totals over 7 days after publication (`total_inviews`, `total_pageviews`, `total_read_time`).
- **L2:** test-set impressions after prediction time, e.g. `future_article_log_counts`, `global_article_-5m_5m_normed_counts`.
- **L3:** test-set impressions before prediction time, e.g. `past_article_log_counts`, `global_article_-11m_-1m_normed_counts`. Counted as leakage because it needs real-time processing [V].

| Transformer feature set | L1 | L2 | L3 | Val AUC | Test AUC |
|---|---|---|---|---|---|
| Full | ✓ | ✓ | ✓ | 0.8734 | 0.8864 |
| w/o future impressions | – | – | ✓ | 0.8495 | 0.8680 |
| w/o test impressions | – | – | – | 0.7483 | 0.7699 |

[V slide 26]

- L3 alone is worth about +0.098 test AUC: past impressions in the same period, "from just a minute before prediction".
- L1+L2 add about +0.018.
- Authors' conclusion: process the last few minutes in real time and retrain hourly [V].

### 3.3 The Transformer (best single model)
From `kfujikawa/src/exp/v1xxx_training/v1174_111_L8_128d_smpl3_drophist.py` [V code] and slides.

**Formulation**
- Token classification over the impression: one token per in-view article, plus one prepended impression token.
- No positional encoding in the `torch.nn.TransformerEncoder` variants, so the model is permutation-equivariant over candidates.
- The HuggingFace BERT variants (v1157, v1184) use `position_embedding_type="absolute"`. This was harmless only because EB-NeRD in-view lists were shuffled.

**Article token**
- Every numeric feature gets its own sinusoidal embedding (dim 128: sin/cos with frequencies `exp(-log(10000)*i/64)`, padding → 0).
- These are concatenated with mean-pooled learned embeddings of topics and category.
- Plus one "similarity" scalar per categorical field: the cosine between the candidate's learned topic/category embedding and the mean learned embedding of the user's history.
- Then Dropout(0.1) → Linear → BatchNorm → ReLU to d=128.
- No text or pretrained embeddings were active in v1174 (the `PreTrained…` field class is defined but never instantiated).
- No article-ID embedding: "static features such as article ID overfit by memorizing popularity" [V slides].

**Impression token**
- Sinusoidal embeddings of hour position, read_time, scroll_percentage, `elapsed_ts_from_history` (log hours since the last history click, clipped to [0,5]) and `num_history_articles`.
- Plus `elapsed_ts_from_past/future` and `num_past/future_articles` (the user's own other impressions in the split).
- Plus a device-type embedding.

**Encoder and head**
- 8 layers, d_model 128, 4 heads (d/32), FFN 256, GELU, pre-LN, dropout 0.1.
- Head: Linear–BN–ReLU–Linear → logit per article [V code; "8 layers, 128 dims, 4 heads" V JSAI deck].
- Loss: BCE per article. A `binary_listnet_loss` exists in the code but is unused [V code].

**Optimization**
- AdamW (lr 1e-3, wd 1e-5, eps 1e-6), CosineAnnealing (T_max 5, η_min 1e-6).
- 4 epochs, batch 64 impressions, fp16-mixed, gradient clip 1.0, a single NVIDIA L4 [V code + README].

**Regularizers (v1174)**
- *User-capped sampling:* at most 3 impressions per user per epoch, reshuffled each epoch, about 25% of impressions per epoch (v1170 uses at most 2) [V code; "25%" V slides].
- *History dropout:* for 30% of training samples, drop each history click with p=0.3, always keeping the latest. History features are recomputed on the masked history [V code].
- Their ablation scripts `v2003_no-drophist` and `v2006_no-sample` exist, but no numbers for them are published in the slides.

**Per-article input features** [V code]
- *Freshness:* `article_published_ts_diff` (log hours, clipped to [0,5]) and the within-impression rank of publish time.
- *History match:* how many times the article is in the user's history, its position in the history, hours since that history click, and the within-impression rank of normalized topic/category match counts.
- *User's own other impressions in the split:* log counts and rank of the article among past impressions (L3) and future impressions (L2).
- *Global windowed exposure* (from all impressions of the split, aggregated per minute):
  - windows −5..+5 min, −5..+55, −55..+5 (L2), and −11..−1, −61..−1 (L3);
  - statistics: counts, `normed_counts = counts / max over the impression`, within-impression rank of counts, read-time sum and its rank, read-time mean, scroll mean, scroll-zero ratio.
- *L1 totals:* log values and within-impression ranks.
- Every popularity quantity enters as *both* a value normalized within the impression and a within-impression rank. This makes features computed on a split with different traffic volume comparable.

**Ablation (Table 2 of the paper)** [V slide 31]

| Configuration | Val | Test |
|---|---|---|
| Full model | 0.8734 | 0.8864 |
| w/o impression features (no impression token) | 0.8679 | 0.8835 |
| w/o Transformer (replaced by MLP) | 0.8280 | 0.8435 |

In the code, the MLP variant also drops the impression token (`v2005_no-transformer`).

**Pseudo-labelling (final neural model v1184: HF-BERT, 4 layers, 256-d, 5 epochs)**
- Training data:
  - fold 0: train + pseudo-labelled validation;
  - fold 2: train + validation + pseudo-labelled test.
- Labels are *soft*: `sigmoid(weighted logit ensemble)` of LGBM×2 + CatBoost×1 and Transformer v1174×2 + v1170×1. That ensemble scores AUC 0.8791 on val.
- A confidence value (top-1 minus 3rd score) is computed but not used in the loss [V code `v8004_015_016_v1170_v1174.py`].
- The gain from pseudo-labelling is not reported.

### 3.4 GBDTs
**LightGBM** [V `kami/experiments/015_train_third`]:
- objective lambdarank, lr 0.1, num_leaves 128, max_depth 8, feature_fraction 0.8, bagging 0.8 every iteration, λ2 0.1, min_child_samples 10, max_bin 256, up to 1000 rounds, early stop 40 on NDCG.
- 20% impression sample.
- Hand-made interaction features, e.g. `topics_sim × total_inviews`, and `time_min_diff / impression-mean time_min_diff`.

**CatBoost:** depth 8, lr 0.1, 2000 iterations, eval NDCG [V `016_catboost`].

**Top-10 LightGBM features** [V slide 32]:

| # | Feature | Leak type |
|---|---|---|
| 1 | `c_time_hour_diff` (hours since publication) | – |
| 2 | rank of Σ(read_time / #in-view) over all impressions within ±5 min | L2+L3 |
| 3 | `c_time_min_diff` | – |
| 4 | rank of the user's own past exposures to the article in the split | L3 |
| 5 | same as #2, future 5 min | L2 |
| 6 | same as #2, future 1 h | L2 |
| 7 | `a_inviews_per_pageviews` | L1 |
| 8 | `c_time_sec_diff` | – |
| 9 | cosine(mean CountVectorizer+SVD topic vector of history, candidate) | – |
| 10 | same as #2, past 5 min | L3 |

Implementation detail: counts are computed over the user and over all users ("common"), for windows 5m / 1h / all, past / future / both. Each count is turned into `ratio = x / Σ over the impression` and rank ascending/descending within the impression [V `c_appear_imp_count_v7`].

**Transition feature** `y_transition_prob_from_first`:
- P(next click = candidate | user's last clicked article), estimated from all users' histories [V code].
- Similar to our covisitation features.

### 3.5 Stacking and ensembling
Code: [V `sugawarya/src/stacking.py`, `weighted_mean.py`, `make_submission.py`]

**Stage-2 training data**
- Validation-split predictions of five stage-1 models: LGBM, CatBoost and three Transformers, each trained on train only ("first").
- At test time the same stacker is applied to the stage-1 models retrained on train+val ("third"). The authors accept this mismatch.

**Stage-2 features**
- Per model: raw score; impression mean/max/min/std; z-score within the impression; min-max within the impression; rank and descending rank; rank / impression length.
- For every pair of models: difference, ratio, max and min of raw score, rank, normalized rank, z-score and min-max values.
- Counts: impression length, user count, (article, user) count.

**Stage-2 model**
- LightGBM lambdarank, lr 0.1, feature/bagging fraction 0.8, max_bin 1024, early stop 50.
- 4-fold GroupKFold by impression; each fold trains on every 10th impression.

**Weighted mean**
- Optuna, 200 trials, weights in [0,1], maximizing mean impression AUC on a 1% impression sample of val.

**Final score** = `stacking_pred + weighted_mean`: raw sum, a lambdarank score plus a probability mix.

---------------------------------------------------------------------------------------------------

## 4. Second place: BlackPearl (AUC 0.8815)

Sources: https://github.com/BlackPearl-Lab/RecSys-Challenge-2024-2nd-Solution [V code/README]; abstract of doi 10.1145/3687151.3687163 [V abstract].

**Single models and blend** [V README]

| Model | Offline GAUC | Online |
|---|---|---|
| DIN-style NN | 0.856 | 0.873 |
| CatBoost PairLogitPairwise | 0.860 | 0.873 |
| CatBoost **QueryCrossEntropy** | 0.866 | **0.879** |
| Blend | 0.8678 | 0.8808 |

- Blend: per-impression min-max normalization of each score, then 0.5·QueryCE + 0.3·NN + 0.2·PairLogit [V `code/blend.py`].
- The final "alldata" CatBoost scripts fit on the **validation week only**: `model.fit(valid_set)`, 5000 iterations, depth 8, lr 0.05, GPU, border_count 254 [V `model2_train_catboost_query_loss_alldata.py`].
- Hardware: 8×A100 + 1 TB RAM [V].

**Key insight in the README**
- "the length of the news list displayed in the exposure sequence and the hourly granularity data of news exposure within the exposure interval on the same day are very strong features" [V].

**Top features in their importance file** [V `features/imp_catboost_0620_pair_rank_loss_0.8601_filter_feature.csv`]
1. `user_id_impression_id_article_impression_day_hour_article_inview_num_mean_rank_reverse`:
   - for each article, the mean in-view-list length of all impressions containing it in that day-hour (`user_impression_freq` = #articles in the impression [V `8.0merge_feature_v1.py`]);
   - reverse-ranked within the impression.
2. `impr_article_impr_hour_article_cnt_rank`: within-impression rank of the article's exposure count in that day-hour.
3. `ctr` = total_pageviews / total_inviews (L1).
4. Embedding PCA component, etc.

There are many `*_mean_impr_diff` features (value minus impression mean) and `*_rank / *_rank_reverse` within-impression transforms [V `14.0add_new_feature_0619.py`].

**NN (HUIM)** [V README/config]
- DIN target attention over a history of length 200 (history file augmented with behaviour data).
- Item side info: id, category, subcategory, sentiment, type, read-time bucket, scroll bucket.
- Frozen pretrained text/image vectors reduced to 64-d.
- ParallelMaskNet feature weighting (2 blocks × 64), MLP 1024-512-256, batch 8192, Adam 1e-3, 1 epoch, BCE.
- The training step is meant to add a query-level softmax cross-entropy auxiliary loss (λ = 0.2). As written, `listwise_loss` stays 0, and the per-query losses are computed with `compute_loss` (BCE), not with the softmax `compute_loss_v2`. So the intended listwise term is not actually applied [V `util/model.py` lines 844–903].
- Abstract: long-term interests from multimodal history plus short-term interests from real-time context and "behavioral path patterns", fused with query-level cross-entropy [V abstract].
- Best no-leak AUC among top teams: 82.20 [V organizers].

---------------------------------------------------------------------------------------------------

## 5. Other teams

- **Tom3TK (3rd, 87.07; doi …3687159, closed access).**
  - LightGBM ensemble.
  - "Article timeliness features" plus content-based features from history.
  - Multilingual embedding models; "the importance of suitable embedding models".
  - "Performance plateauing with the expansion of training samples" [V abstract only].
- **hrec (5th, 86.67; Huawei, Jieming Zhu et al.; doi …3687158).**
  - "Real-time feedback learning mechanisms to capture users' immediate interests".
  - "Generative sequence modeling … to learn impression-level user behaviors".
  - Tree + deep ensemble [V abstract only]. No code found.
- **FeatureSalad (6th, best academic; https://github.com/remaplab/recsys-challenge-2024-ekstrabladet [V code]).**
  - *Stacking:* level-1 = CatBoost classifier and ranker, LightGBM classifier and ranker, MLP, GANDALF, Deep&Cross, Wide&Deep; level-2 = CatBoost classifier and LightGBM classifier. Level-1 is trained on train → predicts val; level-2 is trained on val. Level-1 is then retrained on train+val → predicts test [V README].
  - *"Contextual normalizations"* [V `pre_new_with_recsys.py`, `_norm_and_stats.py`]:
    - for popularity, recency and similarity features: x / max over the impression ("l_inf");
    - rank ascending or descending within the impression;
    - x − median over the impression;
    - impression std / skew / kurtosis / entropy;
    - `n_unique(category | sentiment | article_type)` per impression ("list diversity");
    - normalization over the user and over the article.
  - *"Endorsement"* [V `_catboost.py`]:
    - per article, rolling 10 h count of in-view exposures (times rounded to 1 minute);
    - divided by the sum over the same minute;
    - minus its own rolling mean; MACD (rolling-5 minus rolling-10 mean); divided by the minute's 0.8-quantile.
  - *Trendiness:* topic publication counts over 3 or 5 days.
  - *Classic recommenders* (content-based KNN on several embedding sets) as features.
  - Without leaks: 81.13 → 79.01 [V organizers].
- **SBJ Partners / tetsuro731 (8th, AUC 0.8169; https://github.com/tetsuro731/RecSys-Challenge-2024-tetsuro731 [V]).**
  - LightGBM ranker with about 140 features.
  - Trained on 60% randomly sampled rows; random row sampling beat negative downsampling.
  - One model on the train week and one on the val week, averaged 1:1, times 3 seeds.
  - NER-cluster features had high importance but overfit, so they were removed.
  - User vector = mean of the last 5 clicks (PCA-16 text embeddings).
  - Feature-group ablation (nDCG@10 on leaderboard) [V `paper_work/input/recsys_local.csv`]:

| Feature set | nDCG@10 (LB) |
|---|---|
| 42 baseline features | 0.6161 |
| + user×article interaction features (119) | 0.6713 |
| + session features, incl. time to next session (126) | 0.6919 |
| + impression-time distribution and embeddings (130) | 0.6996 |

  - train+valid vs train-only at equal features: +0.0148 and +0.0021 on the leaderboard.
- **DIVAN (Politecnico di Bari).** DIN for personal interest plus a "virality-aware" click predictor from temporal popularity features, mixed with a user-specific weight [V abstract only].
- **DArgk (academic 2nd, 77.13; paper abstract says 0.7709).** Transformer over user reading history; 8 epochs on train, then 1 epoch fine-tuning on validation [V https://github.com/knife982000/RecSys2024Challenge README].

---------------------------------------------------------------------------------------------------

## 6. Synthesis by theme

**Impression-level listwise modelling**
- The only explicitly listwise neural ranker among the winners is :D's set Transformer with an impression token.
- Value over an MLP: +0.043 AUC. Value over GBDTs with rank features: +0.005 to +0.007. Ensemble gain over the Transformer: +0.006 [V].
- No other top team used cross-candidate attention. They achieved the "listwise" effect through within-impression rank/ratio/diff features in GBDTs, plus group losses (LambdaRank, QueryCrossEntropy, PairLogit).

**Impression-relative features (every top team)**
- rank asc/desc, rank / list length, x / max, x / sum, x − mean, x − median, z-score;
- impression-level dispersion statistics;
- in-view list length, and aggregates of the list lengths an article appears in.
- These appear in the importance tops of :D (5 of the top 10 are ranks within the impression) and BlackPearl (top 2) [V].

**Transductive / future features**
- Dominant: worth +0.10 to +0.12 AUC for :D, and 2–6 points for the others [V].
- They work because the test week contains millions of impressions of the same about 20K live articles.

**History features**
- Freshness relative to the user (time since the last click), topic/category affinity with history (counts and learned-embedding cosine), whether the article was already in the history, transition probabilities, embedding cosine (mean of the last 5 clicks).
- DIN-style target attention over long histories was the strongest non-leak approach (BlackPearl).
- ID-heavy static features overfit, according to both :D and SBJ [V].

**Recent data**
- Retraining with the latest labelled week: +0.010 to +0.012 AUC (:D) [V]. BlackPearl's final CatBoost was trained on val only [V]. SBJ averaged train-week and val-week models [V].
- Data size saturates: 20% ≈ 100% (kami) and "plateauing" (Tom3TK) [V].

**Stacking**
- Two levels, with the level-2 model trained on the most recent labelled split using level-1 out-of-period predictions (:D, FeatureSalad) [V].
- Then a simple average with an optimized weighted blend (:D) [V].

**Losses**
- :D: LambdaRank for GBDTs, BCE for the Transformer.
- BlackPearl: QueryCrossEntropy was the best tree loss (+0.006 online over PairLogitPairwise) [V].
- kami recommends trying `rank_xendcg` as well as LambdaRank [V].

---------------------------------------------------------------------------------------------------

## 7. Transfer to our data: checks run this session, and recommendations

### 7.1 Check: time-windowed transductive popularity (the :D L2/L3 analog)
Script: `<repo>/reports/deep_recsys2024_checks/timeloc_pop.py`. Val only.

**Feature.** For each val candidate: the number of times its item appears in the top-K history events of *other users'* val feeds whose timestamp is within ±W of this feed. Split into past / future / both. K ∈ {8, 32, 256}; W ∈ {1 h, 6 h, 1 d, all}.

**Results** (val NDCG of f1 alone = 0.3279; the `it_hist_all` blend reaches 0.3371):

| Setting | Coverage | Target rate (nonzero / zero) | NDCG of rank(f1) + 0.5·rank(feature) |
|---|---|---|---|
| ±1 h, any K | ≤ 0.6% | – | ≤ 0.3279 (+0.0000) |
| K=8, all time, both | 1.3% | 2.2% / 1.6% | 0.3288 |
| K=256, all time, both | 8.5% | 2.03% / 1.54% | 0.3312 (+0.0033) |

**Conclusion.** The density that made :D's ±5 min features dominant does not exist here. Time localization adds nothing. Use global or train-derived item statistics, and drop this direction.

Related, from our own EDA (`eda/history_leak/03_leak.log`): the per-user "future" leak exists but is tiny in coverage. A user's later feed's history contains the candidate in 0.36% of test feeds. On the 113 affected val feeds, NDCG goes from 0.316 to 0.965, which is +0.0051 overall on val. Keep it if the rules allow it.

### 7.2 Check: "recent data matters" applied to item-statistics windows
Script: `<repo>/reports/deep_recsys2024_checks/recency_stats.py`.

Item positive counts from the last N days of train, scored on val as NDCG of rank(f1) + 0.5·rank(pos_count):

| Window | NDCG | Coverage |
|---|---|---|
| last 1 day | 0.3294 | 39.7% |
| last 3 days | 0.3313 | 53.5% |
| last 7 days | 0.3322 | 58.6% |
| all 14 days | 0.3326 | 65.4% |

Smoothed train CTR (prior 0.024, strength 20) gives at most 0.3278 in the same blend: production CTR is position-confounded.

**Conclusion.** Avito ad lifecycles are much longer than news. The recency lesson applies to *which labelled data the model is fitted or fine-tuned on* (val is the target distribution), not to shortening feature windows.

### 7.3 Our NN vs :D's Transformer ablation
Local logs `nn/logs/*.log`, val NDCG at the same step.

| Run | Step 6000 | Step 16000 |
|---|---|---|
| base | 0.3677 | 0.3696 |
| no set attention | 0.3663 | 0.3693 |
| no history | 0.3583 | 0.3611 |
| BCE instead of softmax | 0.3643 | – |
| no position tower | 0.3674 | 0.3712 (best 0.3722) |

Caveats:
- `noset`, `nohist` and `bce` ran on a slightly newer code version (9984 vs 9710 steps per epoch), so step-matched comparisons are approximate.
- Single seeds; some runs were still in progress.

**Conclusion.** Unlike EB-NeRD, cross-candidate attention is close to neutral here, because there is little transductive signal to compare across candidates. History modelling is where the signal is (consistent with BlackPearl being the best no-leak team). Softmax beats BCE on our metric, which differs from :D's BCE choice under an AUC metric.

### 7.4 Recommendations (priority order)
Gains are unverified for our data unless stated otherwise.

1. **HIGH: :D-style two-level stack** (stage-1 trained on train, stage-2 on val).
   - Stage-1 candidates: train-only NN, LightGBM/CatBoost trained on train features (with `pos` handled), the current val-CV LGBM out-of-fold scores.
   - Stage-2: LightGBM lambdarank under 5-fold GroupKFold *by user*. Features: per-model raw score, within-feed z / min-max / rank / rank÷M, feed mean/std/max of each score, pairwise differences/ratios/max/min of all of these, plus M and a few strong raw features.
   - Average with an Optuna-weighted, per-feed-normalized mean.
   - Evidence: +0.006 AUC (test) and +0.0068 (val) over the best single model at :D [V].
2. **HIGH: final fit includes val.**
   - For the NN: choose the number of fine-tune epochs by 5-fold OOF, then fine-tune on all val (or average the fold models, as `finetune.py` already does).
   - For GBDTs: kami's first/third scheme, i.e. fix iterations from CV, then refit on all data.
   - Also try a stage-1 trained on train + upweighted val.
   - Evidence: +0.010 to +0.012 AUC from adding the latest labelled week [V].
3. **MEDIUM-HIGH: soft pseudo-labels on test for the NN** (:D v1184 [V code], gain not reported).
   - Fine-tune on val plus 28.6K test feeds, with soft listwise targets = softmax(ensemble logit / T) per feed.
   - To keep CV honest, build the test pseudo-labels for fold k from teachers that never saw val fold k.
4. **MEDIUM: :D training regularizers in NN pretraining on train.**
   - At most 1–3 feeds per user per epoch: test has about 1.05 feeds per user, and this also removes heavy-user dominance.
   - History dropout: p_sample 0.3, p_event 0.3, keep the newest event. It also simulates colder users; only about 35% of val/test users are seen in train.
   - Both are in :D's best Transformer [V code]; their effect is not published.
5. **MEDIUM: contextual-normalization extensions** (FeatureSalad/BlackPearl [V]).
   - For count/popularity/similarity features: x/max, x/Σ, x − median, x − mean.
   - Feed-level std/skew/entropy of f1..f5 and the key similarities.
   - Number of distinct categories in the feed, and the candidate category's share (already partly present: `feed_cat_share`).
   - This complements the existing `_z/_rk/_gap`. Cheap. Expect small gains.
6. **MEDIUM: ranking losses for the trees.** CatBoost `QueryCrossEntropy` (BlackPearl's best: +0.006 online over PairLogitPairwise [V]) and `YetiRank`, plus LightGBM `rank_xendcg`, alongside lambdarank. Blend after per-feed min-max normalization [V BlackPearl blend].
7. **MEDIUM: item lifecycle proxies.**
   - Time since publication was :D's #1 GBDT feature [V].
   - Since `item_id` is not time-ordered (checked), use instead: first-seen timestamp over train candidates and over feeds whose history contains the item (a history appearance bounds creation time from above); age at feed time; a censoring flag; days active; within-feed rank of age.
   - Check single-feature NDCG first.
8. **LOW-MEDIUM: feed-size statistics of items** (BlackPearl's top feature family [V]): mean/std of log M over the train feeds containing the item, plus its within-feed rank. Cheap to check. Coverage is limited (35% (val) to 47% (test) of val/test candidate items never appear as train candidates).
9. **Do not pursue:**
   - within-val/test time-windowed exposure or popularity (§7.1);
   - shortened train-statistics windows (§7.2);
   - positional embeddings over candidates (val/test are shuffled);
   - static item-ID embeddings in the NN. :D avoided ID features because they memorize popularity and overfit [V]. Many val/test items are unseen in train.

---------------------------------------------------------------------------------------------------

## 8. Sources

**Read this session [V]**
- Organizers' overview: https://arxiv.org/html/2409.20483v1
- EB-NeRD dataset paper: https://arxiv.org/html/2410.03432v1
- :D slides: https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning
  - slide images `files.speakerdeck.com/presentations/94a17cb23fb045fb99224445ec31fd8c/slide_{22,25,29,30}.jpg` (0-indexed; pages 23, 26, 30, 31) and `slide_31.jpg` (page 32)
- :D code: https://github.com/k-fujikawa/recsys-challenge-2024-1st-place
  - `kfujikawa/src/exp/v1xxx_training/*.py`, `v2xxx_ablation/*.py`, `v8xxx_ensemble/v8004*.py`
  - `kami/features/*`, `kami/experiments/015_train_third`, `016_catboost`
  - `sugawarya/src/*.py`
- JSAI2025 winners' report: https://speakerdeck.com/unonao/jsai2025-recsyschallenge2024-you-sheng-bao-gao
- kami, "N=1 recommendation competitions": https://speakerdeck.com/unonao/n-equals-1-notui-jian-xi-konpenozhan-ifang
- RecSys 2024 trip report: https://speakerdeck.com/unonao/recsys2024-can-jia-bao-gao
- BlackPearl code: https://github.com/BlackPearl-Lab/RecSys-Challenge-2024-2nd-Solution
- FeatureSalad code: https://github.com/remaplab/recsys-challenge-2024-ekstrabladet
- SBJ Partners code: https://github.com/tetsuro731/RecSys-Challenge-2024-tetsuro731
- DArgk code: https://github.com/knife982000/RecSys2024Challenge
- FuxiCTR baseline: https://github.com/reczoo/RecSys2024_CTR_Challenge
- Abstracts of all ten RecSys Challenge '24 proceedings papers (DOIs 10.1145/3687151.3687152–3687163), via api.semanticscholar.org and api.crossref.org

**Not readable:** ACM full texts at dl.acm.org (HTTP 403), iris.polito.it, politesi.polimi.it.

**Local checks:** `reports/deep_recsys2024_checks/timeloc_pop.py`, `recency_stats.py`. Our NN logs: `nn/logs/*.log`. EDA: `eda/history_leak/03_leak.log`.
