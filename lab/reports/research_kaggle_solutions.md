# Research: top solutions of similar competitions -> checklist for Avito feed ranking

Confidence legend: [V] verified from fetched page this session; [S] seen in search snippet (secondary); [K] from background knowledge, not re-verified (treat as hypothesis).

## 1. Per-competition notes

### RecSys Challenge 2024 (Ekstra Bladet, EB-NeRD) - closest analogue (in-view impression list + user history, AUC/nDCG)
Sources: https://arxiv.org/pdf/2409.20483 (organizers' paper, PDF unreadable via fetch), https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning (1st place, team ":D" / DeNA)
- [V] 1st place: Transformer where each in-view article is a token and all candidates of an impression are scored jointly (listwise context; sinusoidal embeddings of numeric features; BCE) = best single model (AUC 0.8864); LightGBM and CatBoost scoring each article independently with LambdaRank within impression (0.8817 / 0.8805); stage-2 LightGBM stacker (Optuna) on stage-1 predictions, stage-3 simple average; ensemble 0.8924.
- [V] Features: time since article publication, time since user's last click, topic overlap of article with user's click history; static ID features overfit, so they focused on features less tied to identity.
- [V] Validation/training: tune on a time split, then RETRAIN on the most recent (validation) period with same hyperparameters -> about +0.10 AUC on test from recency. Lesson for us: val (Sep 22-24, random-shuffled, no position bias, closest in time/distribution to test) should be used as training data for final fit (at least fine-tuning / stage-2 model).
- [V] Leakage discussed: article stats covering future periods; test impressions from after and before prediction time (transductive counts of how often an article appears/clicked in other impressions). Winners' AUC dropped 0.886 -> 0.770 when removing such features [S]; i.e. transductive "in-test" counting features (popularity/CTR computed from impressions in test period, including later ones) are hugely powerful. Recent past impressions alone recovered much of the gain.
- [S] FeatureSalad (best academic, 6th): contextual normalization of features within impression and "article endorsement" (how many other users/impressions contain/clicked it).

### RecSys Challenge 2019 (trivago, rank 25 hotels per session by click, MRR)
Sources: https://www.cs.toronto.edu/~cheng/projects/recsys_challenge_2019/ (2nd place), https://github.com/topics/trivago (RosettaAI 4th), https://arxiv.org/pdf/2010.06987 (mentions winner)
- [S] Winner (Univ. of Pisa-ish / LightGBM ensemble) hand-engineered >25,000 features; LightGBM-based [S, secondary].
- [V] 2nd: global + local session contextualization, GBDT + deep reranking, forward-in-time split.
- [S] 4th RosettaAI: LightGBM + XGBoost + DeepFM-like NN with sequential features.
- [K] Known key tricks: per-impression rank/normalized (divide by max/mean, z-score, rank) of price, rating, position, popularity; impression-list order/position leak (hotels were shown in list order); "last interacted item" and interactions-with-item-in-session features (clickout/item-interaction counts), item-item co-occurrence in sessions, per-item CTR with time-aware folds; LambdaRank/binary objective both used; GBDT with many features + NN blends.

### Outbrain Click Prediction (Kaggle 2016-17, MAP@12, ad ranking within display)
Sources: https://aitopics.org/doc/news:B4D73FDC (brain-afk 2nd, intro only), https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle (13th), https://www.kdnuggets.com/2017/06/feature-engineering-help-kaggle-competition-1.html (+ parts 2,3), https://habr.com/ru/post/323938
- [V] 2nd place: heavy feature engineering beat stacking.
- [S] 13th: blend of 5 models (FTRL/SVM on basics, FFM, XGBoost) plus leak-extraction scripts and document similarity.
- [K] Winner (Kaggle "Outbrain" 1st, team including Chinese data scientists; FFM + GBDT stacking): Field-aware FM on categorical crosses (ad_id x doc, user x ad context), XGBoost with target-style CTR features (ad, campaign, advertiser, source, publisher with Bayesian smoothing and confidence), "leak" features = other page_views of same user (uuid) -> user clicked/viewed documents similar to the ad's landing doc, count of times the ad appeared in the display set, user's previous displays (history of which ads/docs they saw in earlier displays), timestamp-order based features; within-display features: number of ads in display, rank of ad's CTR within display, CTR minus display mean. Pairwise/softmax-over-display loss helped FFM.

### OTTO (Kaggle 2022-23, session recsys, Recall@20 weighted)
Sources: https://towardsdatascience.com/building-a-recommender-system-using-machine-learning-2eefba9a692e, https://github.com/Fedorov-Artem/kaggle_OTTO/blob/main/README.md, https://arxiv.org/pdf/2310.07281
- [S] Standard pattern: candidate generation (co-visitation matrices: click->click, click->cart/order, buy->buy; time-window <=5 min or <=20 events; weighting by recency and event type) then GBDT/LambdaRank reranker (XGBoost pairwise / LightGBM); 2024 reproduction: co-visitation + Word2Vec candidates, LightGBM+XGB+CatBoost ensemble.
- [K] 1st place (Chris Deotte / Benedikt Schifferer / Theo Viel / Gilberto Titericz; NVIDIA): ~100-200 candidates/session from many generators (co-visit variants, MF/embeddings), >100 features incl. candidate-generator scores & ranks as features, item/user/session stats, time-since-last-interaction, XGBoost/LightGBM rankers per target, + a NN ranker; trained on subsampled negatives; local validation by time-split using last week's sessions truncated as test. Item-item co-visitation scores from the user's history items to each candidate (several weightings) were the strongest features.

### H&M (Kaggle 2022, MAP@12)
Sources: https://goldandrabbit.github.io/2022/05/15/H&M%20Recommendation%20Competition%20Top1%20%E6%96%B9%E6%A1%88 [S]
- [S] 1st: multiple retrieval strategies (recent popular items etc.), 100 cands/user, 6 weeks train + last week val (stable CV-LB), mostly user-item interaction features (image/text no help), LightGBM best single (CV 0.0441), final 5 LightGBM + 7 CatBoost.
- [K] Strong features: time-decayed purchase counts of item, item "age" (days since first seen), user-item repeat purchase (same item or same product-group), item-pair purchase ("bought together") scores, item popularity in the last 1/2/4 weeks, user-category affinity.

### Expedia Personalized Hotel Sort (ICDM/Kaggle 2013, NDCG@38) - nearest to "groups with positive/negative"
Source: https://arxiv.org/pdf/1311.7679 (5th place team) [S]
- [S] LambdaMART with CTR/booking-rate features at two granularities (prop_id and discretized price), normalized features, within-list rank features (rank of price, star rating, location score): 0.5243 NDCG@38 vs 0.5115 without normalization/rank features. Per-query normalization (z-score by srch_id, by prop_id/destination) consistently big gain.
- [K] Winner (Owen Zhang): GBM/NN blend; random-order "position bias" part of data: Expedia had random-sorted searches (like our val/test) - top solutions trained with position feature and then set position constant/ignored at prediction, or trained mostly on random-order portion; price/rating normalized per search and per property; composite features such as estimated position/historical mean.

### Avito Context Ad Clicks (Kaggle 2015, log-loss)
Source: https://adamdrake.com/a-quick-overview-on-the-kaggle-competition-for-avito.html [S]; arXiv 1502.00527 is Yandex paper not Avito.
- [K] Winners (Kagglers Owen Zhang, Michael Jahrer, Dmitry?): massive user-ad-search-context history counting features (user's previous impressions/clicks of the same ad, of the same category, "has user seen this ad N times"), target-encoded CTR with time-aware smoothing, position/hist-CTR as features, FTRL + GBDT + FM + NNs blended. Prior views of the same ad without click (negative-feedback counts) were highly informative. Counts of impressions from the future (leak-type) were used.

### Yandex Personalized Web Search (Kaggle 2014, NDCG)
Sources: https://blog.dataiku.com/2014/01/14/winning-kaggle (Dataiku, winner), https://www.arxiv.org/pdf/1502.00527 (Context models), https://www.slideshare.net/slideshow/dataiku-pap-is-2014-winning-kaggle-yandex/42476320
- [S] Winner Dataiku: re-ranking framework: keep the baseline ranking info as a feature (unpersonalized rank is very informative), LambdaMART listwise model (~90 features, ~1135 trees) with features from the user's previous queries/clicks of the same URL/domain, features selected by forward selection; point-wise models on same features gave the best 30 features that were fed into LambdaMART. 4th (Volkovs): >100 features.
- [K] Important lesson: label definition via dwell time (satisfied clicks), user's re-finding behavior (user previously clicked same URL for same query -> huge), and leakage via session ordering. Ranking in our case: the "past impression/click of same item by same user" equivalent is only possible via item_id in history (candidates are filtered from own history, but same item may appear in OTHER feeds' histories/candidates across time).

### KDD Cup 2020 Debiasing (Tianchi, NDCG@50_half on rare items)
Sources: https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place (README), https://www.besthub.dev/articles/debiasing-competition-solution-multi-hop-i2i-graph-modeling-for-advertising-recommendation-eaa559dc382c [S], https://github.com/xuetf/KDD_CUP_2020_Debiasing_Rush
- [S] Item-item graph (click co-occurrence + multimodal embedding similarity) with multi-hop random-walk candidates, LightGBM with popularity-weighted loss (weights inverse to item popularity), post-hoc boost of low-popularity items. Rush: CF/GNN recall + GBDT/DIN ranking; item popularity, user activity, interaction time/direction as CF improvements.
- [K] Lesson for us: evaluation is on unbiased (random) traffic while train is biased -> inverse-propensity or inverse-popularity weighting; train/eval on the biased log risks learning "was ranked high by the old ranker". Known trick: use `pos` as a training-time-only feature (or as a bias tower), set it to a constant at inference (as val/test are random-shuffled).

### KuaiRand (unbiased random exposure data) [K]
Source: https://kuairand.com (not fetched). Lessons from papers on it: mix small unbiased data with large biased log (e.g. KDCRec, "AutoDebias", causal embeddings / CausE); fine-tune on random subset; use random subset for validation and meta-learning of propensity weights; combining via multitask/bias tower.

### Avito Demand Prediction (Kaggle 2018) [K/S]
Sources: https://towardsdatascience.com/kaggle-avito-demand-challenge-18th-place-solution-neural-network-ac19efd6e183 and other mirrors. Not a ranking task, but Avito-specific: image/text embeddings + target encoding of categorical (user_id, category), out-of-fold aggregations, GBDT+NN blend; user-level aggregates were strong.

## 2. Cross-cutting patterns
1. Candidate list given -> reranking = pointwise/pairwise/lambdarank GBDT on rich features is the dominant baseline (RecSys24 1st, H&M, OTTO, Yandex, Expedia, trivago). LambdaRank vs binary: both; blend.
2. Within-group normalization (rank, z-score, divide by group max/mean, diff to group mean, percentile) of every numeric feature and of every score is repeatedly reported as a top gain (Expedia, trivago, RecSys24 FeatureSalad).
3. Listwise/set-context neural models (Transformer over candidates of one impression) were the best single model in RecSys24.
4. Transductive counts / "leak" features from other impressions (including later ones in the same split) are the biggest differentiator (Outbrain, Avito 2015, RecSys24, trivago). In our test, same item appears in many test feeds; counts of candidate appearances and of item in test-users' histories (history events include time-after-train period) are allowed by rules in principle (data given in test); check competition rules for legality - note RecSys24 organizers called them leakage but allowed.
5. Recency matters: retrain/fine-tune on latest data (RecSys24: +0.10 AUC from retraining on validation period).
6. Stacking: stage-2 GBDT over NN/GBDT predictions + simple averaging at the end.
7. Time-aware validation, forward in time; keep a final holdout that mimics test distribution (here random-shuffled val).

## 3. Consolidated ranked checklist for the Avito feed task
Rank by expected impact (est.), with how to implement.
1. Train on the right distribution / handle position bias: train has position bias (pos column) while val/test are random. Options: (a) add `pos` (and normalized pos/M) as a feature in train, set to a constant/mean at inference, or use a two-tower (relevance + position bias) / inverse-propensity weights estimated from the data; (b) rely on val for fine-tuning/final training (K-fold across val feeds, or train on train then continue on val with small lr); (c) restrict train to feeds with weakly informative pos. Validate each variant on val.
2. Group-relative features: for each feeds' candidates, rank/percentile/z-score/diff-to-mean/max for f1..f5, category share within the feed, cosine similarities, popularity, etc. Also feed-level context features (M, category distribution entropy, user's history length).
3. User-history-to-candidate matching: cosine of candidate embedding vs user embedding; max/mean/top-k similarity to history items (weighted by event_type, recency decay, event_source); category match counts with history (by event type, last-k windows); co-visitation / item-item scores between history items and candidate (OTTO/H&M style) computed from train histories; time-decayed category affinity.
4. Transductive item statistics across ALL splits (and within test): number of feeds where item appears as candidate, count of appearances in users' histories by event type (history may contain later events than the impression: "future" signal), per-item exposure/click rate from train with temporal folds (only recent weeks, Bayesian smoothed), recent (last 1-3 days of train) popularity. Item-appearance-in-history counts for the test period users are available in test (check rule legality); validate on val the same way.
5. Target encodings with temporal folds: item_id, category, (user, category), (user, item) pair when repeated; compute from earlier time windows only (e.g. features for train week k from weeks <k), apply smoothing; beware that train labels are position-biased -> use position-corrected CTR (clicks / expected clicks given pos).
6. Models: LightGBM lambdarank + binary logloss (or xendcg), CatBoost YetiRank/ binary; Transformer over the candidate set of a feed (per-candidate tokens with features, embeddings, user-history summary) trained with BCE or listwise softmax; blend/stack with a second-stage LGBM and rank-average. Expect NN set-model to be best single (RecSys24).
7. Training set construction: 247M train rows is huge: subsample feeds and keep all positives + hard negatives? For lambdarank keep whole feeds (sample feeds ~10-20%); upweight recent days; final fit on train(late)+val.
8. Candidate-score features: two-stage approach (generator scores like ALS/implicit item-item, two-tower dot, co-visitation as inputs to ranker); ranker sees score and rank of each generator inside the feed.
9. Per-user behavior features: user's CTR-like activity from history (event-type mix, most recent event type/source, time since last event - history lacks timestamps? use hpos), user embeddings x candidate embeddings interactions; user historical propensity vs. feed-level (feeds with many positives).
10. Metric-specific post-processing: pessimistic ties -> avoid ties (add tiny noise/ensemble); full-list NDCG means the whole ordering matters, including bottom: positives with tiny scores penalized heavily -> use listwise/lambdarank with no truncation (lambdarank_truncation_level high), consider a pointwise loss for stability. Feeds without positives are not present, so conditioning "at least one positive" can be used (softmax-over-feed loss, which implicitly assumes >=1 positive).
11. Validation: use val (14.4K feeds, SE ~0.002-0.003) with paired comparisons; build an extra CV from random-shuffled train subset if any exists (check whether some train feeds have pos uncorrelated with order, i.e. exploration traffic); do not over-tune to Public LB (30% feeds); final ranking = Private.
12. Ensembling: several seeds / different objectives, average of per-feed rank-normalized scores; stage-2 stacker over OOF predictions grouped by feed.

## 4. Caveats
Most winner write-ups for Outbrain, OTTO, Avito 2015, Expedia winner, trivago winner were not retrievable in this session (web search yielded secondary summaries); items marked [K] come from the researcher's background knowledge and should be validated by experiment rather than trusted. Only RecSys24 1st place details and the Expedia 5th-place paper were directly read.

## 5. Source URLs
- https://arxiv.org/pdf/2409.20483
- https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning
- https://www.cs.toronto.edu/~cheng/projects/recsys_challenge_2019/
- https://github.com/topics/trivago
- https://arxiv.org/pdf/2010.06987
- https://aitopics.org/doc/news:B4D73FDC
- https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle
- https://www.kdnuggets.com/2017/06/feature-engineering-help-kaggle-competition-1.html
- https://habr.com/ru/post/323938
- https://towardsdatascience.com/building-a-recommender-system-using-machine-learning-2eefba9a692e
- https://github.com/Fedorov-Artem/kaggle_OTTO/blob/main/README.md
- https://arxiv.org/pdf/2310.07281
- https://goldandrabbit.github.io/2022/05/15/H&M%20Recommendation%20Competition%20Top1%20%E6%96%B9%E6%A1%88
- https://arxiv.org/pdf/1311.7679
- https://adamdrake.com/a-quick-overview-on-the-kaggle-competition-for-avito.html
- https://blog.dataiku.com/2014/01/14/winning-kaggle
- https://www.arxiv.org/pdf/1502.00527
- https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place
- https://www.besthub.dev/articles/debiasing-competition-solution-multi-hop-i2i-graph-modeling-for-advertising-recommendation-eaa559dc382c
- https://github.com/xuetf/KDD_CUP_2020_Debiasing_Rush
