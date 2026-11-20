# Deep dive: re-ranking stage of top solutions in OTTO (Kaggle 2023), H&M (Kaggle 2022), Amazon KDD Cup 2023 -> what to transfer to Avito ML Cup 2026

## 0. Sources, tags and how they were read

Tags:
- **[V]**: I read the primary text myself this session (URL given).
- **[V-m]**: I read the full text of the Kaggle discussion post through a mirror: HF dataset `Abirate/all_write_ups` (https://huggingface.co/datasets/Abirate/all_write_ups, file `write_ups_all.jsonl`, which holds the Kaggle write-up HTML incl. author replies). Kaggle pages, OpenReview PDFs and the AIcrowd GitLab could not be fetched directly (JS shell, bot check, connection refused). **Images inside posts (feature-importance plots, pipeline diagrams) were NOT visible.** The mirror keeps the post titles but drops author names.
- **[S]**: seen only in a search snippet.
- **[K]**: my background knowledge, not checked this session.

Primary sources read:
- OTTO: 1st, 2nd (ONODERA's part), 3rd (Chris's post) and 5th to 28th place posts [V-m]. Theo Viel's 3rd-place code: `src/data/fe.py`, `src/model_zoo/xgb.py` and the `notebooks/6-XGB.ipynb` params [V] (https://github.com/TheoViel/kaggle_otto_rs). Chris Deotte's repo README and the `XGB-412-ORDERS.ipynb` params plus feature list [V] (https://github.com/cdeotte/Kaggle-OTTO-Comp). Benny Schifferer's README [V] (https://github.com/bschifferer/Kaggle-Otto-Comp).
- H&M: 1st, 2nd, 3rd, 4th, 5th, 6th, 9th to 13th and 22nd place posts [V-m]. K. Fujikawa's slides [V] (https://speakerdeck.com/kfujikawa/h-and-m-personalized-fashion-recommendation).
- KDD Cup 2023:
  - NVIDIA-Merlin poster (1st place in all 3 tasks) [V]: https://github.com/kddcup23/kddcup23.github.io/blob/main/posters/NVIDIA-Merlin-Poster-v2.pdf
  - NVIDIA tech blog [V]: https://developer.nvidia.com/blog/pro-tips-for-building-multilingual-recommender-systems/
  - Posters from MGTV-REC, YAMALEX, gpt_bot (4th T1 / 3rd T2), AIDA and dx2 [V] (same repo, `posters/`).
  - 3rd place T1 README, CXL's part [V]: https://github.com/Xiuchen519/Amazon-KDDCUP-23
  - 5th place T1 paper [V]: https://arxiv.org/pdf/2307.10650
  - 10th place T2 paper [V]: https://ar5iv.labs.arxiv.org/html/2310.07281
  - The NVIDIA workshop paper itself (https://openreview.net/pdf?id=J3wj55kK5t) could not be fetched (OpenReview bot check).

Our own logs I looked at, to avoid recommending work that is already done: `eda/history_leak/05_covis*.log`, `03_leak.log` and `eda/embeddings/val_feat_names.txt`. What they show:
- Item-level co-visitation from histories already exists (`cv_raw5/rec5/fwd5/invd20`, ...), but **only 6-11% of candidates get a non-zero score**. Item ids are extremely sparse: 91M unique items vs 1.8M in OTTO and about 105k articles in H&M.
- Two-tower embedding features are already aggregated by event type and recency.

---

## 1. Short answer: what the winners' re-rankers had in common (and what transfers)

1. **The strongest re-ranker features were item-to-item scores from each history item to the candidate, aggregated with recency and event-type weights.** In OTTO these were co-visitation counts plus word2vec/MF cosine similarities. In H&M they were item-CF scores, w2v similarities and BPR user-to-item similarity. In KDD'23 they were co-visitation counts, text-embedding similarities and NN scores. In OTTO 1st place's ablation, removing co-visitation cost 0.0052 weighted recall@20 and removing the NN scores cost 0.0053 [V-m]. Removing session features cost only 0.0002.
2. **A learned user-to-item collaborative similarity was the largest single feature when present.**
   - H&M 3rd place: BPR (implicit) user-to-item score had AUC 0.720 on its own, vs 0.680 for the best other single feature and 0.806 for the whole model. It took LB from 0.03363 to 0.03510 (+0.0015, +4.4% relative) [V-m].
   - OTTO 17th place: w2v cosine between the last item and the candidate gave +0.003; tuning the w2v hyper-parameters gave a further +0.001 [V-m].
3. **Scores and ranks from the candidate generators, used as ranker features, were essential.**
   - H&M 3rd: "recalled by strategy X" flags plus rank within each strategy, together with expanding candidates from dozens to about 100, moved LB 0.02855 to 0.03262. CV was "very very poor" without the rank features [V-m].
   - OTTO 6th: the stage-1 rank was the most important feature [V-m].
   - OTTO 1st: popularity "worked well when ranked" [V-m].
   - In our setting this means within-feed ranks and z-scores of every score, which we already do. It also argues for adding many *diverse scorers* to rank.
4. **Labels and negatives.** Label = 1 if the candidate is in the future positive set, 0 for every other candidate. Groups without positives were dropped (Chris, Theo, OTTO 20th) [V]/[V-m]. Negatives were downsampled heavily, and the rate was set mostly by RAM:
   - OTTO 1st: 5% of negatives for clicks, 25% for carts, 40% for orders [V-m].
   - Theo: POS_RATIO 0.2 / 0.5 [V].
   - H&M 1st: 1-2M negatives per week; H&M 3rd: 30 negatives per positive [V-m].
   - Counter-evidence: OTTO 9th said "negative sampling lowers the score" and trained orders without it [V-m]. NVIDIA (KDD'23) trained on 360M rows x ~150 features with 8xV100 Dask-XGBoost and gained **+0.002 MRR** from it [V].
   - **With 4xGPU we can afford full or near-full data, and that is the evidence-backed direction.**
5. **No objective wins consistently**, so ensembling across objectives is the safe choice (table in §6.3). LGBMRanker (lambdarank) won for OTTO 1st, 7th, 16th and 17th. XGB `rank:pairwise` won for Chris (3rd). Binary logloss won for Theo (3rd), OTTO 6th's second model, OTTO 12th, H&M 1st, H&M 6th and KDD AIDA. CatBoost PairLogitPairwise won for OTTO 20th, and YetiRank for H&M 11th.
6. **Validation** was a time-forward split that mimicked the test exactly (Radek's OTTO split; last week for H&M). Reported CV-LB correlation was "almost perfect" (OTTO 1st) and linear with R²=0.99 (H&M 10th) [V-m]. Teams iterated on 5% train / 10% eval subsamples (OTTO 1st) or 1/20 of sessions (OTTO 20th) and kept the correlation [V-m].
7. **Leakage through history-derived features was the main CV/LB trap.**
   - H&M 5th trained w2v on all data: CV 0.0441 but LB 0.0350, a much larger gap than other teams [V-m].
   - H&M 10th: CF models trained on data later than the observation date "boost CV by implicitly using purchase information from the future" [V-m].
   - OTTO 7th built co-visitation matrices separately per period (train / valid / test) to stay leak-free [V-m].
8. **Ensembling gave small but consistent gains when done right.**
   - Averaging the scores of 9 LGBMRankers with different hyper-parameters: 0.604 to 0.605 (OTTO 1st).
   - Rank-sum of members per user (OTTO 3rd: individual 0.601, team 0.6038 private).
   - Calibrated blend of LambdaRank and a GNN (OTTO 7th: +0.003).
   - 5 LGBM + 7 CatBoost (H&M 1st: 0.0367 to 0.0371). [V-m]
9. **Stacking (stage-1 predictions as features) helped some teams** (OTTO 2nd ONODERA, Chris's `click_oof`). **It failed to transfer to LB for others** (OTTO 13th, 15th) [V-m]. Validate it on a clean holdout.
10. **Does not transfer**:
    - Candidate retrieval/recall engineering, since our candidates are given. In H&M, retrieval was described as "the key".
    - Session-level timestamps, which our histories lack (only position).
    - Item-id-level CF density, which is much lower for us (§7.2).

---

## 2. Setting mapping (theirs vs ours)

| Aspect | OTTO | H&M | KDD'23 (Amazon-M2) | Avito (ours) |
|---|---|---|---|---|
| Task unit | session; predict future clicks/carts/orders | customer; purchases next week | session; next item | feed of 5-300 given candidates |
| Candidates | self-generated, ~50-1200 | self-generated, 100-1000 | self-generated, 100-300 | **given** (mean 96) |
| Metric | Recall@20 (weighted 0.1/0.3/0.6) | MAP@12 | MRR@100 | full-list NDCG, binary, pessimistic ties |
| History | session events with ts, type (3) | purchases with dates | prev items, item attributes | up to 256 events with type (10) and source (27), **no timestamps** |
| #items | ~1.8M | ~105k articles | 1.4-1.55M | **~91M ids in histories** (very sparse) |
| Train/test shift | none (time) | none (time) | locale | **train position-biased, val/test random-shuffled** |

Consequences:
- Their entire "covisitation -> candidate rank -> GBDT" machinery is used here only as **features**.
- Their densest signals (item-id co-visitation) are weaker for us because of id sparsity. That pushes towards **cluster/category back-off and learned embeddings** (§7).

---

## 3. OTTO (Kaggle, Nov 2022 - Jan 2023)

### 3.1 1st place (solo; "my" write-up) [V-m]
- **Candidates**, ~1200 per session on average, shared across click/cart/order:
  - visited aids;
  - several co-visitation matrices with different type weights and aggregation periods, "applied at multiple times like beam search" (multi-hop);
  - NN models (MLP or transformer, "no big difference").
- **NN design details** (relevant to our sequence-model plans):
  - the same embedding table for input items and target items;
  - **several future items used as positive targets**, not just the next one;
  - the target type is injected into the session embedding;
  - some models were **trained only on non-visited targets**, so they do not overlap with revisit features. *This matches our setting: our candidates are always filtered out of the user's own history.*
  - Time inputs: hh = ts//3600//1000, hh//24, sin/cos of the hour of day.
- **Reranker**: LGBMRanker. Single model LB 0.604; mean of the scores of 9 LGBMRankers with different hyper-parameters LB 0.605.
- **Features**: ~200 built, ~100 kept per target by LightGBM gain.
  - Session x aid: rank by co-visitation at candidate generation, NN cosine, aid info within the session (when it appeared, what type).
  - Aid: popularity over several windows ("worked well when ranked"), type ratios.
  - Session: length, aid duplication rate, ts gap between the last two events.
- **Negative sampling**: clicks 5%, carts 25%, orders 40%, chosen so that each training set was ~35 GB. Machine: [n] GB RAM.
- **CV**: Radek's split (discussion 364991). Fast iteration: train on 5% of the data, evaluate on another 10%. About 1.5 h to merge features and 1.5 h to train and evaluate.
- **Ablation** (local weighted recall@20; information removed from both candidates and features):

| condition | clicks | carts | orders | weighted | Δ |
|---|---|---|---|---|---|
| full (LB .604) | .5566 | .4364 | .6696 | **.58836** | - |
| w/o visited aid | .5557 | .4356 | .6665 | .58613 | -.0022 |
| w/o covisitation | .5475 | .4302 | .6656 | .58314 | -.0052 |
| w/o NN | .5448 | .4299 | .6660 | .58306 | -.0053 |
| w/o aid features | .5505 | .4334 | .6663 | .58485 | -.0035 |
| w/o session features | .5559 | .4358 | .6697 | .58817 | -.0002 |
| only single NN | .5323 | .4101 | .5648 | .51513 | -.0732 |

Takeaways:
- Co-visitation and NN scores carry equal and *complementary* weight.
- An NN alone is far worse than GBDT on top of it.
- Session-level features add nothing on their own: they matter only through their interactions.

### 3.2 2nd place, ONODERA's part (team with psilogram, senkin13, h4211819) [V-m]
- Used psilogram's candidates. ONODERA built **item-to-item features keyed by (aid_x, type_x, aid_y=candidate, type_y)**:
  - count, time difference, **sequence difference** (positional distance inside the session; psilogram's idea), two weighted variants, and aggregations;
  - 93 base features, about 5k after crossing type combinations (click->order, cart->order, ...); 400-500 used.
- Also used "1st stage prediction features" and "pseudo event features" (details only in images).
- Models: XGBoost and CatBoost, 2-stage. Final blend with teammates by **rank**. Hardware: DGX 4xV100 with cuDF/cuML.
- The teammates' part (discussion 382839) was not retrievable.

### 3.3 3rd place, team G&B&D&T (Giba, Benny, Chris, Theo); private 0.60382 / public 0.60437 [V-m][V]
- **Team blend**: "adding the ranks of each of our predictions per user target type".
- **Chris** (single model LB 0.601) [V-m] + repo [V]:
  - Rules-only with his 3 public covis matrices: LB 0.575. With **20 covisit matrices** (picked from hundreds computed on GPU in under 1 min each): LB 0.590. Adding the XGB reranker over the extracted covis counts: **+0.011 -> 0.601**.
  - Candidates: the top-50 from the rules (the post); the README says 100 per target.
  - Interaction features = co-visit counts extracted **per matrix**, one feature each.
  - Matrix variants worth copying as *patterns*:
    - consecutive-only pairs (|k_y-k_x|==1), |Δk|<=2, |Δk|<=3 / <=6 with time decay (1/2)^(Δt/1h);
    - carts/orders forward pairs within 3 steps;
    - "any -> carts/orders" forward pairs on the last 3 weeks with time decay;
    - recent-window-only matrices (last 1-3 weeks);
    - **cold-start matrices built only from users' first history item**;
    - matrices split by user history length (<6 vs >6 events);
    - matrices split by the hour of the first item (<2pm vs >2pm).
  - Embedding features (author reply): cosine of the candidate vs the last 5 items, the first 5 items and the last 5 buys, plus mean/std aggregates over groups (`last_0..4`, `first_0..4`, `buy_0..4`, `e_mean_k/e_std_k` in the feature list) [V].
  - **XGB params (orders)** [V]: `rank:pairwise`, max_depth 4, lr 0.1, subsample 0.7, colsample_bytree 0.5, eval_metric `map`, GroupKFold(5) by user, gpu_hist, early stopping. The orders model takes the **click model's OOF prediction (`click_oof`) as a feature** (223 features). Orders recall: 0.668165, 0.668704 "with giba".
  - Users without any positive candidate were removed from training ("my pipeline can handle 1000+ features"). Hardware: 4xV100 32GB for matrices; the repo run used 20 CPUs, [n] GB RAM and 1 GPU.
  - Post-competition: XGB+CatBoost 0.6018 vs XGB 0.6012 private [V].
- **Theo** (CV 0.5917 -> LB 0.6028; ~80 candidates from Chris; **744 features**) [V] (README + code):
  - Recipe: for each (session, history item, candidate), compute **item-item scores** (Chris's covis coefficients, w2v, MF cosine). Multiply by **history-item weights**, then aggregate mean/sum/max per (session, candidate).
  - Weights from `compute_weights()` [V]:
    - `w_pos-log` = 2^(0.1+0.9*(L-pos-1)/(L-1)) - 1 (position recency, from 1 for the newest to 0.07 for the oldest);
    - `w_type-163` = {click:1, cart:6, order:3};
    - `w_lastday` = event within 24h of the session's last event;
    - `w_time` = 1/log(2+Δminutes);
    - `w_recsys` = w_time * 1/log(2+pos) * 1/log(1+len) * 1/log(1+cumcount(aid)) ("from RecSys 2022 winners": recency x position x session-length damping x item-popularity damping).
  - Weights are summed per (session, aid) over repeats.
  - Rank features: within-session rank of a feature (ties get the min rank) [V].
  - Benny's w2v features (in Theo's code) [V]:
    - similarity zeroed if <0.5;
    - `sim_max`, `sim_sum`, count of history items with sim>0.5;
    - `sim*wgt` with wgt = (2^0.1-1 .. 1) x type multiplier {click 0.5, **cart 9**, order 0.5};
    - `sim_last` (the last item only);
    - sums are divided by (1 + count).
  - Popularity [V]: per type, week / day / ±1.5h-hour-window popularity normalised by the period total, plus the ratios hour/day and day/week (a "trend" feature).
  - **Model** [V]: XGBoost `binary:logistic`, lr 0.01, max_depth 8, subsample 0.9, colsample 0.7, alpha 0.01, lambda 0.1, min_child_weight 0, eval AUC, early stopping 100, gpu_hist.
  - Training data [V]: **POS_RATIO 0.2 (clicks) / 0.5 (carts, orders)**, i.e. negatives downsampled to that positive share. `use_gt_sessions=True` (only sessions with a positive). 4-fold CV by session. Optuna with 20 trials per fold over depth 5-10, subsample 0.4-1, colsample 0.5-1, alpha/lambda 0.01-100.
- **Benny** (public 0.60102 / private 0.6003) [V]:
  - XGBoost per target over Chris's covis features, his own covis, **w2v, GRU and Transformer (Transformers4Rec) scores as features**.
  - Training data built from 5 chunks of the truncated week-4 split; using all 5 folds (instead of 20% truncated + 80% untruncated) improved LB.

### 3.4 Other gold/silver evidence on the re-ranking stage [V-m]

**6th place, 1st post** (public 0.604 / private 0.603, near single model):
- LightGBM, **5% negatives**, ~400 features, last 2 weeks of train. Adding the second-to-last week: +0.0005.
- **One model for all three targets** (grouped by session, not session x type): +0.001 to +0.002.
- **Graded label gain** orders 6 / carts 3 / clicks 1: +0.0005.
- Stage-1 rank was the most important feature. Candidates 80 -> 120: +0.001; 200 did not help.
- Covis weights normalised by the frequency of the first item; distance-weighted; the reverse direction gets half weight.

**6th place, 2nd post** (single LB 0.60335):
- LightGBM binary, lr 0.02, 5500 rounds. Local CV 0.5915.
- Final fit on train_v1 + train_v2 + valid.
- Similarity features: i2i and i2i2i (two-hop) weighted by position and time, aggregated mean/max/min/std/last.

**7th place** (LightGBM part public 0.60025 / private 0.60008; GNN 0.59894 / 0.59874; blend 0.60311 / 0.60313):
- Candidate score is a **Naive-Bayes-style log-lift**: Score(aid) = log P(aid) + Σ_{i=1..10} ((11-i)/10) · log[ P((aid_i,type_i), aid) / (P(aid_i,type_i) P(aid)) ].
  - Co-visitation within 24h, counted once per session, both directions, over the whole period including test at inference.
  - Missing pairs get ratio 1.
- Top-200 candidates for train, top-300 for test, plus visited items. Recall@200: clicks .697, carts .559, orders .736.
- **LGBM LambdaRank per target, 344 features**:
  - 250 co-visitation features **per history position i=1..10**: covis(aid, aid_i)/count(aid_i) for any->any, any->buy, buy->any, buy->buy, type_i->any/buy, click/cart pairs, one-way and both-direction; ranks; global counts of aid_i;
  - 50 aid features: counts per type, exponential-decay counts, counts in the last n=1..7 days and n=1..4 weeks, revisit counts;
  - 12 session x aid features;
  - 32 session features.
- Covis matrices **computed separately for the training, validation and test periods (leak-free)**.
- Two models trained on different weeks were averaged; gain was "only slight".
- **Blend**: the LambdaRank output is mapped to (0,1) as σ(v + Ĉ), with Ĉ fitted per target type so the mean matches the GNN's mean probability. Then a weighted average (0.6/0.4 and 0.525/0.475).

**9th place** (single LB 0.602; orders CV .67087):
- ~180 candidates, with **more candidates for longer sessions**: n = 10·sqrt(count).
- 59 item-CF patterns, e.g.:
  - weight 1/sqrt(session length);
  - time window 24h;
  - type weights {1,3,6} by the minimum type of the pair;
  - normalised by sqrt(cnt_a·cnt_b);
  - "real sessions" split on 2-4h gaps.
- User-based CF features.
- **Orders trained without negative sampling** ("negative sampling lowers the score"); carts with negatives x0.3, clicks x0.1. Two-stage: the 2nd stage re-ranks the top-50 from the 1st stage.
- CatBoost Logloss, lr 0.02, depth 5, GPU, `scale_pos_weight = neg/pos`; 5 seeds (seeds also varied in the GT split).
- w2v/ALS/BPR did not work for this author.

**Other single-team findings**:

| Team | Objective / ranker | Negatives | Notable findings |
|---|---|---|---|
| 10th | LGBM lambdarank | - | candidate score Σ 1/idx_hist · 1/idx_covis |
| 12th | binary | 1:20 | "ensembling classification and ranking does not provide any boost" |
| 13th | xgb/lgb/cat probability ensemble | - | stacking: good validation, did not reach LB |
| 14th | LGBM binary lr .05 x1000, CatBoost binary lr .05 x6000 | - | i2i weighted by time gap, position gap and type difference: +0.001 CV; orders_prob = 0.7·orders + 0.2·carts + 0.1·clicks: +0.0004; carts+orders trained jointly with an indicator: +0.0003 |
| 15th | 5 LGBM rankers (seeds) | ratio 0.2 | candidates 50 -> 100: +0.0007; cross-target predictions as features: +0.001 CV but not LB |
| 16th | LGBMRanker best | - | beat LGBM binary, CatBoost binary and YetiRank |
| 17th | LGBMRanker best; dart did not help | - | w2v last-aid sim +0.003; w2v tuning +0.001 |
| 20th | CatBoost PairLogitPairwise best; LGB/XGB lambdarank slightly worse | dropped sessions without target + 20% random downsampling | BPR-like MF score among the strongest features; transformer candidate source +0.002 CV but <0.001 LB |
| 28th | XGBoost | 2.5-30% | higher negative fractions did not help |

---

## 4. H&M Personalized Fashion (Kaggle 2022, MAP@12)

**1st place** (senkin13 & 30CrMnSiA; public 0.0372 / private 0.0379) [V-m][V slides]:
- "Candidate generation strategy is the key"; mostly recent popular items. HitNum@100 for the week of 2020-09-16: 39,142 of 213,728 positives, i.e. recall 18.3% [V slides].
- A **logistic-regression pre-ranker** narrows the top-1000 popular items to 50-200.
- Top-N per strategy was tuned by hand to maximise hits within a fixed budget of 100 to 500 candidates.
- Features, 340+, "almost all interaction features":
  - user-item and user-category counts over last week / month / season / same week last year / all, plus time-weighted counts;
  - first/last transaction days;
  - aggregates of age, price, channel;
  - differences and ratios (user age minus the mean age of the item's buyers);
  - **similarity: item-CF scores, w2v item-item cosine, ProNE user-item cosine**;
  - the scores from the retrieval strategies themselves;
  - cosine between the user's latest and second-latest purchased items and the candidate.
- Image and text features did not help (useful only for cold start).
- Labels: a candidate is 1 if bought next week; duplicate purchases per user-week removed.
- **Negatives: random 1-2M per week**.
- Training: 6 weeks of train, last week as validation. Final retrain on all data with **~1.1x rounds**.
- **Model**: LightGBM *classifier* best single (CV 0.0441, LB 0.0367); final 5 LGBM + 7 CatBoost classifiers (LB 0.0371). XGBoost-GPU was worse than LightGBM. FM, MLP and transformer rankers "didn't work".
- Tooling: feature store, TreeLite (2x faster inference).

**2nd place** (wht1996 + paweljankiewicz; public 0.0368 / private 0.0374) [V-m]:
- wht1996: ~600 popular items (+ section-level popular items) -> **a binary LGBM selects 130** + all previously purchased items -> a richer LGBM with **lambdarank**.
  - Features: repurchase probability, "predicted next purchase time" (last purchase + mean inter-purchase interval), itemCF score, popularity by age group.
  - Single 0.0355; with Pawel's candidates 0.0362; ensemble 0.0368.
- Pawel:
  - popularity counters by customer-attribute combinations (**adding postal_code boosted significantly**) over windows of 1, 3, 7, 30 and 90 days;
  - image similarity, in-basket and cross-basket co-occurrence, random walks;
  - "streak" features;
  - a custom **lambdarank-MAP objective for LGBMRanker**, slightly better than NDCG lambdarank.

**3rd place** (solo) [V-m]:
- **Recall-strategy flags plus rank within each strategy** and expanding candidates: 0.02855 -> 0.03262. Without the rank features, CV was very poor.
- **BPR user-to-item** similarity (implicit, retrained each week on data strictly before the target week): AUC 0.720 alone, vs best other feature 0.680 and full model 0.806. LB 0.03363 -> 0.03510.
- Other similarities: w2v item-item (mean/sum/max over history), Jaccard of attributes, text/image embedding cosine.
- Popularity: 7d / 30d / all, 1/days time-weighted, weekly trend.
- **Negatives 30x positives**; only recalled positives used.
- Training: 18 weeks (vs 2 weeks: +0.0005 to +0.0010).
- Final model: average of **16 models** (LGBM, CatBoost, XGB with different lr, depth and seeds).

**4th place** (solo; CV 0.039, LB 0.0349) [V-m]:
- Two-tower MMoE retrieval (gated experts for active vs inactive users); its user-item similarity used as a ranker feature.
- **LightGBM lambdarank**; DCN-v2 ranker was worse.
- 5 folds, 3 used for training; more data did not help.

**5th place** [V-m]:
- 21 recall methods, faiss over w2v / image (Swin) / text (SentenceTransformer) / tfidf embeddings; w2v the most useful.
- Single LGBM ranker.
- Features: article-only and customer-only features computed once per observation date and merged (saves compute).
- **Leak warning**: w2v trained on all data inflated CV (0.0441) much more than LB (0.0350).

**6th place** [V-m], same features, varying the number of candidates:

| model / objective / #cands | CV MAP@12 |
|---|---|
| CatBoost binary / 120 | **0.0403** |
| LightGBM binary / 120 | 0.0395 |
| CatBoost binary / 220 | 0.0402 |
| LightGBM binary / 220 | 0.0396 |
| LightGBM lambdarank / 220 | 0.0400 |
| LightGBM lambdarank / 1000 | 0.0381 |

- Setting the categorical features (department, product type) as LGBM categoricals: +0.0005 to +0.0008.
- 9 weeks of training.
- Giba's part: 200 / 300 / 500 random negatives per customer; blend of LGBM lambdarank, **LGBM xendcg**, XGB lambdarank and CatBoost ranker.

**Smaller notes** [V-m]:
- 10th: CV-LB linear fit R²=0.9896; CF models trained beyond the observation date leak future information into CV.
- 11th: CatBoost YetiRank "much better" than LGBMRanker; default params good; 20h per model on 224 cores; CatBoost on GPU was worse.
- 22nd: classifier CV 0.03559 vs ranker 0.03497.

---

## 5. Amazon KDD Cup 2023 (Amazon-M2; MRR@100)

**NVIDIA-Merlin** (1st place in tasks 1, 2 and 3; Deotte, Onodera, Puget, Schifferer, Titericz). Poster [V], blog [V]:
- **Stage 1 candidates**:
  - co-visitation top-N lists ("counting how often items appear together", top-100 per item, aggregated over the session's items);
  - BERT/LLM text-embedding similarity (5 multilingual encoders compared);
  - **SWIFT/Swing** similarity;
  - fine-tuned session/item embeddings, CNN-based, initialised from a pretrained LM; 4000 random negatives; KNN gives 100 candidates. An embedding-retrieval component alone scored T1 train 0.209 / LB 0.226;
  - Transformers4Rec XLNet (MLM), GRU and CNN scores.
- **Stage 2**: features = session, candidate, item and session-item features, with the stage-1 scores reused as session-item features.
  - **A GRU predicts the probability of the next item's brand / color / size / model per session; P(candidate's attribute) is then used as a feature.**
  - **"Train 360 million rows with ~150 features using 8xV100 DASK XGBoost for +0.002 MRR boost"**: scaling up the training rows paid off.
  - The blog lists binary, pairwise or listwise XGBoost as options and does not say which one they used.
- The OpenReview paper, which may give feature importances, was not reachable.

**3rd place T1, CXL's part** [V]:
- 100 candidates from SASRec i2i + 100 from XLM-R + the co-graph top-100.
- **The XGBoost reranker is trained on the validation set.**
- Features: recommender scores *and normalised scores*, text-model scores, BM25, co-graph counts from 6 strategies, price-vs-session statistics, item and next-item frequency.
- Final model: average of 3 XGB with different features and params.

**4th T1 / 3rd T2, gpt_bot (incl. senkin13)** [V poster]:
- Shared preprocessing: covisit matrices, text transformer, ProNE, BPR, item2vec.
- ~200 candidates per user; ranker = 2 CatBoost models over statistical and embedding-similarity features.

**5th T1** [V] (arXiv 2307.10650):
- CatBoost fusion of ID-based and text retrieval: CV MRR 0.3175 -> 0.3394 with feature engineering (+0.022); LB 0.4033.
- Feature groups: item "hot" features, session features, **graph features on the co-occurrence graph: PageRank, betweenness, Katz and degree centrality, neighbour edge-weight statistics**.

**YAMALEX** [V poster]:
- LightGBM over ~300 candidates per session; **negative down-sampling 7% (T1) / 70% (T2)**.
- Features:
  - session aggregates;
  - **diffs between the last / last-two items and the candidate** (Levenshtein, Jaro-Winkler, title distance, w2v, same-attribute flag, price diff);
  - **recall scores plus their rank within the session**;
  - item features.

**AIDA** [V poster]:
- XGBoost: "we tested the objectives of 'rank' and 'binary' and found that learning classification tasks can achieve higher scores".
- "Lowering the score of goods that did not appear in the test set can significantly improve the final score" (transductive item presence).

**10th T2, dx2** [V]:
- LightGBM binary.
- Co-visitation features c_s, c_axb, c_ab, c_ba, y_axb, y_ab computed **per history position (last 10)**: 60 similarity features.
- Features that look at next_item are computed out-of-fold (5-fold sequential CV) to avoid target leakage.
- Data mixing:
  - adding single other-locale data helped by up to 0.88% relative;
  - naively pooling everything **hurt** (T1 0.36975 vs 0.37584);
  - **the mixing ratio matters.**

---

## 6. Cross-cutting evidence tables

### 6.1 Feature families that dominated (re-rank stage)

| Family | Evidence (gain) | Our analogue |
|---|---|---|
| History-item -> candidate co-visitation, weighted by recency, type and distance, aggregated sum/mean/max, **per history position** | OTTO 1st: -0.0052 if removed; Chris: rules 0.590 -> +0.011 with the reranker; 7th: 250 per-position features | item-level covis from histories (exists, 6-11% coverage) -> add a cluster/category back-off and typed variants |
| Learned CF similarity (BPR/MF u2i, w2v i2i, ProNE) | H&M 3rd BPR +0.0015 LB (biggest); OTTO 17th w2v +0.003; OTTO 20th MF "among strongest"; H&M 1st ProNE u2i | i2v already built (`eda/collab_stats/out/i2v_*`); BPR/ALS u2i not yet |
| NN sequence-model scores as features | OTTO 1st -0.0053 if removed (alone 0.515 vs 0.588); Benny GRU/Transformer; NVIDIA T4Rec/GRU/CNN | sequence model over 256-event histories scoring candidates |
| Generator scores and **within-group ranks / normalised scores** | H&M 3rd +0.004; OTTO 6th top feature; YAMALEX, CXL | within-feed rank/z of every scorer (have; extend to the new scorers) |
| Item popularity over several windows, ranked; trend ratios | OTTO 1st "worked well when ranked"; Theo hour/day/week ratios; 7th last-n-days counts | popularity in histories by event type, train-period exposure counts, test-period transductive counts |
| Attribute-level preference | NVIDIA GRU P(brand/color/size); H&M postal_code/age popularity; YAMALEX same-attribute flags | P(candidate category / embedding cluster \| history) from a sequence model |
| Graph centrality of the candidate | KDD 5th T1 (part of +0.022) | PageRank/degree on the item co-occurrence graph |

### 6.2 Labels and negatives

| Source | Group filter | Negative handling |
|---|---|---|
| OTTO 1st | - | 5% / 25% / 40% (clicks / carts / orders) |
| Theo | sessions with GT only | POS_RATIO 0.2 / 0.5 |
| Chris | users with a positive only | none mentioned |
| OTTO 6th | - | 5% |
| OTTO 9th | - | none for orders ("lowers score"); x0.3 carts, x0.1 clicks |
| OTTO 15th / 12th | - | 0.2 / 1:20 |
| H&M 1st | - | 1-2M per week, random |
| H&M 3rd | - | 30 x positives |
| YAMALEX | - | 7% (T1) |
| NVIDIA KDD | - | trained 360M rows (+0.002 MRR from scale) |

### 6.3 Objectives

| Winner of internal comparison | Where |
|---|---|
| LGBM lambdarank | OTTO 1st (9-model average), 7th, 10th, 16th, 17th; H&M 2nd (final stage), 4th, 5th |
| XGB rank:pairwise | OTTO 3rd (Chris) |
| Binary logloss | OTTO 3rd (Theo, XGB), 6th (2nd post, LGBM), 9th (CatBoost), 12th; H&M 1st (LGBM + CatBoost), 6th (CatBoost binary .0403 > LGBM lambdarank .0400 > LGBM binary .0396), 22nd; KDD AIDA (XGB), dx2 |
| CatBoost PairLogitPairwise / YetiRank | OTTO 20th / H&M 11th |
| Custom lambdarank-MAP | H&M 2nd (Pawel), slightly better than lambdarank |
| Graded label gain (6/3/1) | OTTO 6th: +0.0005 |

### 6.4 Validation and final fit
- **Time-forward, test-mimicking splits**: Radek's split (OTTO); last week (H&M); per-locale 92/8 split with eval sessions used as the ranker's train data (KDD CXL).
- **Quick-iteration subsamples kept the correlation**: OTTO 1st (5% train / 10% eval); OTTO 20th (1/20).
- **Final refit on all data with ~1.1-1.25x rounds**: H&M 1st 1.1x; OTTO 15th 1.25x. OTTO 6th: train_v1 + v2 + valid.
- **Trust local CV over public LB**: H&M 6th ("best CV = best PB"); H&M 10th (R² .99). Exception: H&M 11th saw a negative CV/LB correlation when changing the number of training weeks.

### 6.5 Ensembling
- Score averaging of same-family models: OTTO 1st +0.001; H&M 3rd (16 models); OTTO 15th (5 seeds).
- Rank-sum or rank-vote across members: OTTO 3rd, 2nd, 17th.
- **Calibrate ranker outputs before mixing with probabilities**: OTTO 7th, σ(v + C) per type.
- H&M 1st: LGBM + CatBoost, +0.0004 LB.
- OTTO 12th: no gain from mixing classifier and ranker.

---

## 7. What to do for Avito, concretely (with priority, expected impact, and how to verify)

Context: our OOF is 0.365, LB top 0.3935, top-10 about 0.388. The gap of ~0.025 is the size of "adding a whole new signal family", not of tuning.

### 7.1 [HIGH] Learned collaborative similarities from the 456M history events (H&M 3rd, OTTO 17th / 20th)
- **BPR/ALS user-to-item** (implicit, GPU; or a PyTorch BPR with hard negatives as in OTTO 20th):
  - Training data: (user/feed, item) interactions from histories, weighted by event type.
  - Score `dot(u, i_cand)` plus within-feed rank and z.
  - For unseen candidate items (cold), fall back to an item vector projected from our 128-d two-tower embedding with a ridge map item_emb -> BPR vector, fitted on items present in both. This handles the 91M-id sparsity.
- **i2v similarities** (we already have i2v d64/d128):
  - per-position features cos(c, h_j) for j=0..9 (OTTO 7th / dx2 style), not just aggregates;
  - Benny's thresholded aggregates: count of sim>τ, sum/(1+count), max, last;
  - tune w2v (window, ns_exponent, epochs, subsampling): OTTO 17th +0.001.
- Expected: largest new single signal among non-NN options; H&M saw +4% relative.
- Verify: paired val NDCG via CV on val, then LB.

### 7.2 [HIGH] Fix co-visitation sparsity with back-off levels and typed, normalised variants
Our item-id covis touches only 6-11% of candidates. Keep it, and add:
- **(a) Cluster-level covis**: k-means of the 128-d item embeddings at k = 512 / 8k / 131k (GPU faiss). Covis between the history item's cluster and the candidate's cluster, with the same windows.
- **(b) Category-level transitions**: P(cat_c | cat_h, event_type_h).
- **(c) Destination-type-restricted covis** (the "any -> carts/orders" and "buy2buy" analogues that dominated in OTTO): count pairs (a -> b) only where b's event type is one of the target-like types. Identify those types by their lift over the positive rate.
- **(d) Normalised variants**:
  - P(b|a) = cnt(a,b)/cnt(a) (OTTO 6th);
  - cosine cnt/sqrt(cnt_a·cnt_b) (OTTO 9th);
  - **Naive-Bayes log-lift summed over the last 10 history items with weights (11-i)/10** (OTTO 7th).
- **(e) Variants by window and direction**: consecutive-only, ±2, ±3, ±6, forward-only; hpos-distance decay (1/2)^(Δpos/τ).
- **(f) Matrices conditioned on user state**: users with short vs long histories; users' first item (cold-start) — Chris's `top_20_new` and `top_40_less/more`.
- Aggregate each score with Theo's weight set, adapted to positions only:
  - w_pos-log = 2^(0.1+0.9(L-j-1)/(L-1)) - 1;
  - per-type weights;
  - w_recsys analogue 1/log(2+j)·1/log(1+L)·1/log(1+pop(a_j));
  - aggregate sum/mean/max;
  - then rank and z within the feed.
- Expected: item-level covis already gave +0.015 over f1 alone in rank blends (our log). The back-off should lift coverage from about 10% towards 100% of candidates.

### 7.3 [HIGH] Leak-free feature construction for any model trained on TRAIN feeds (OTTO 7th; H&M 5th / 10th)
- If co-visitation / i2v / BPR are fitted on all histories, a train feed's own user's *later* histories contain the label interaction. Features on train feeds then look far more predictive than on test, where only ~1.05 feeds per user exist and there is no future.
- Fix:
  - build feature statistics from data strictly before the feed date (rolling by day, as OTTO 7th did per period), **or**
  - "leave-user-out": subtract the target user's own contributions.
- For val-only training this is minor: our `03_leak.log` shows later-history overlap in only 0.36% of test feeds. **It becomes critical once we train on the 247M train rows.**

### 7.4 [HIGH] Scale training data with GPU GBDT; use the big biased train set as stage 1 / stacking (NVIDIA KDD'23, H&M 3rd, OTTO 2nd / Chris)
- NVIDIA: 360M rows x 150 features with Dask-XGB gave +0.002 MRR. Our train set is 247M rows.
- Memory estimate [K]: an XGBoost QuantileDMatrix/ELLPACK stores about 1 byte per bin per value, so 247M x 200 is about 50 GB. That fits on one GPU.
- Two routes:
  - **(i)** Fit on train with `pos` handling (see `research_sota_ltr.md`). Use its prediction as a feature (and its within-feed rank) in the val-trained model. This is the analogue of Chris's `click_oof` and ONODERA's "1st stage prediction features".
  - **(ii)** Pool train + val with a domain flag or weights, and choose the mixing ratio empirically. dx2 showed that naive pooling can hurt (0.36975 vs 0.37584).
- Negatives on train: keep whole feeds for lambdarank, or subsample negatives to POS_RATIO about 0.2-0.5 for binary (Theo). Then evaluate whether the full data beats the subsample: OTTO 9th found that negative sampling hurt. Drop feeds without positives (Chris, Theo).

### 7.5 [MEDIUM-HIGH] Sequence-NN scores as GBDT features (OTTO 1st, Benny, NVIDIA)
Train a transformer/GRU on histories (self-supervised next-strong-event prediction over train histories, plus the supervised feed target). Design choices from OTTO 1st:
- tie the input and output item embeddings (we can initialise from the 128-d two-tower embedding, as NVIDIA did with LM init);
- several future positives;
- condition on the target event type;
- **train a variant only on non-history targets**, since our candidates are always unseen in the user's history.

Feed its candidate score and within-feed rank into the GBDT. OTTO 1st: NN-only reached just 0.515 of 0.588, yet removing it cost 0.0053. Expect complementary gains, not standalone wins.

### 7.6 [MEDIUM] Attribute-preference heads (NVIDIA GRU brand/color/size)
- From the history sequence, predict a distribution over (a) the category (54 classes) and (b) the embedding cluster (k = 1-8k) of the next target-like event.
- Features: P(cat_c), P(cluster_c), log-ratio to the feed mean, entropy.
- These are dense, cheap, and complement our hand-built category affinity by event type.

### 7.7 [MEDIUM] User-kNN CF and graph features (OTTO 9th user-CF, H&M 5th U2I, KDD 5th T1)
- **User-kNN**: faiss over user embeddings. Among the k nearest train users, compute the share who interacted with the candidate (by event type) or had it as a positive. Also compute it at cluster level.
- **Graph**: item degree and PageRank on the item co-occurrence graph. Also centrality of the candidate's cluster.

### 7.8 [MEDIUM] Objectives and ensembles
- Train in parallel:
  - LGBM lambdarank (full list; `lambdarank_truncation_level` >= max group size, since the metric has no cutoff);
  - LGBM / XGB binary;
  - XGB rank:pairwise;
  - CatBoost YetiRank / PairLogitPairwise / Logloss.
- Blend by **within-feed rank averaging**, or by **calibrated scores**: map each ranker to a probability (σ(v + C) as in OTTO 7th, or isotonic on OOF), then average with weights fitted on OOF.
- Average same-family models over hyper-parameters and seeds: OTTO 1st gained +0.001 from 9 rankers.
- Final refit on all val data with ~1.1x the CV-optimal rounds (H&M 1st).
- Expected: +0.002 to +0.005.

### 7.9 [LOW-MEDIUM] Smaller, cheap items
- Item popularity at several windows, ranked within the feed: train-period exposure / clicks per day, last-1d / last-3d trend ratios (Theo's day/week ratio), history counts by event type.
- Transductive item presence: count of test-period feeds and test histories containing the item (AIDA found boosting items present in the test set significant).
- "Diff" features between the candidate and the last 1-2 history items: same-category flag, embedding cosine, f-feature difference if history items appear as candidates elsewhere (YAMALEX).
- A 2-stage re-ranker focused on the top-K of stage 1 (OTTO 9th) is optional. Full-list NDCG with ~1.5 positives per feed rewards getting the positive into the top-10.

### 7.10 Validation discipline (OTTO 1st, H&M 10th)
- Keep val-CV as the decision metric: paired differences per feed, SE ~0.0005-0.0013 for blend deltas in our logs. Iterate on subsamples first.
- Treat any CV gain from features fitted on overlapping or future data (i2v, BPR, covis) as suspect until it shows on public LB. Examples:
  - H&M 5th: CV 0.0441 vs LB 0.0350;
  - OTTO 20th: +0.002 CV but under 0.001 LB;
  - OTTO 15th: cross-target stacking +0.001 CV, none on LB.

---

## 8. Pitfalls distilled
1. **Future information in CF / embedding features** inflated CV in H&M 5th and 10th and in OTTO 20th. Build per-period or leave-user-out (§7.3).
2. **Stacking OOF predictions** worked for Chris and ONODERA but not for OTTO 13th and 15th. Hold out a clean fold for the stacker.
3. **More candidates or noise in lambdarank groups hurt**: H&M 6th dropped from .0400 to .0381 going from 220 to 1000 candidates. For us the group size is fixed, but very large feeds (300 candidates) may deserve per-feed weighting or truncation tests.
4. **Negative sampling is a RAM compromise, not a free lunch**: OTTO 9th lost score with it, and NVIDIA gained from more rows.
5. **Ensembling classifier and ranker gave nothing** for OTTO 12th, while it helped others. Always verify with paired val.

## 9. Gaps (not retrieved this session)
- OTTO 2nd place main posts (psilogram / senkin13, discussion 382839) and OTTO 4th / 5th full details.
- Theo's and Benny's own Kaggle posts (382975, 386497); I used their READMEs and code instead.
- NVIDIA KDD'23 workshop paper (OpenReview bot check) and their AIcrowd GitLab code; only the poster and blog were read.
- All images inside Kaggle posts (feature-importance charts, ONODERA's feature tables).
