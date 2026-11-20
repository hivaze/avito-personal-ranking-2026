# Deep research: Russian / ODS ranking and recsys competitions and what they mean for Avito ML Cup 2026 "Personal ranking"

Date: 2026-11-11. Written in English; many sources are in Russian.

Tags: **[V]** = I read the primary source this session (URL given; for Habr/arXiv pages via the fetch tool with verbatim-quote prompts; for GitHub code by cloning and reading the files). **[S]** = seen only in a search snippet. **[K]** = background knowledge, not re-verified. Statements marked **(inference)** are my own reasoning, not taken from a source.

---

## 0. TL;DR: what transfers to our task (ranked)

1. **The data is produced by Avito's main-page stack, and Avito described that stack in detail in Oct 2026** [V, habr 1090584]. CatBoost at L2 picks the **top 300** candidates and a neural L3 reranker reorders them. Our feeds have 5 to 300 candidates, so they are very likely the L2 top-300 minus already-seen items (inference). Avito's best ranker ("model 5") is one transformer over **[filtered user history ; feed candidates]**: candidates attend to the history and (with a mask) to each other, which added about 0.6%. The output goes through a DCN branch and is trained with **ListMLE**. It has about 2.5M parameters. Offline it scored **+7.2% NDCG on target actions over a ~200-feature CatBoost**, and online it gave +3.3% users with target actions. Inputs: history filtered to target actions, capped at 50 events, clicks removed. Per-event representation = item-embedding projection + position + event type + source. Numeric features are quantile-binned (128 bins) with linear interpolation. This is the most directly transferable recipe found anywhere.
2. **Pretrain on histories, then fine-tune on impressions.** Yandex (Khrylchenko) took **2nd place in VK RecSys Challenge 2024 with a pure transformer**: no feature engineering, two submissions [V, slides]. Pretraining reorders each event triple to (action, author, clip) and predicts author and clip given history and action. Fine-tuning predicts the action given history and candidate. ARGUS (Yandex) [V] shows the score is worth far more **as a feature in a GBDT** (+2.32%) than standalone (−5.33%), and that pretraining roughly doubles the gain over fine-tune-only (+1.17% to +2.32%).
3. **Debias production logs with an additive "context tower" that is detached at inference** [V, Yandex arXiv 2310.03481]. During training the model predicts sigmoid(r_ui + r_ctx). At inference only r_ui is used. Here, ctx = position (and feed length). It is the cleanest way to use our 3.15M position-biased train feeds while scoring randomly shuffled val/test. We currently train only on 14K val feeds, which is the biggest structural gap (inference).
4. **Item/user target statistics with out-of-time windows and smoothing.** This is the core of the 2nd place in Avito ML Cup 2024 (CTR) [V, code], the 1st place in Avito ML Cup 2025 [V, code] and the 3rd place in VK 2024 [V, slides]. Variants: target means of all 1- to 3-way key combinations computed on prior days; Bayesian-smoothed "contact ratio" per item and category; CTR keyed by kmeans(item embedding) × user. VK's organizers showed that a single **item like-rate** feature already reaches AUC 0.6175, top 33% of teams [V].
5. **Transductive frequency features from the scored period.** Avito 2024 2nd place computed campaign/banner frequency *within the target day's impressions* and its ratio to the previous day [V, code]. For us that means candidate-item frequency across test feeds and across val/test histories by event type (inference).
6. **Linear autoencoders (EASE-DAN, RDLAE) plus SASRec as collaborative scorers.** These were the retrieval and score features of the **Avito ML Cup 2025 1st place**, whose best submission was a fold-averaged CatBoost YetiRank [V, code]. For us they become item-item scores from candidates to history items, if candidate coverage in histories is high enough (to check).
7. **Blend NN and GBDT by rank, without tuning weights.** Avito's production blends CatBoost and NN ranks 50/50 inside the top 300: +0.7% users with contacts, +0.9% users with target actions [V].
8. **A cheap check with possibly large impact (inference, not from a source):** the official dataset readme says already-seen items are removed from candidates, and history holds the latest events. A target action on candidate X in an earlier feed may therefore show up in the **same user's later feed history**. Check this on val (users with ≥2 feeds) before investing elsewhere, and check the rules.

Public numbers for calibration [V, a participant repository]: the official CatBoost YetiRank baseline (9 features) scores **0.3533 on full val and ~0.3461 on Public**, so val minus Public ≈ +0.007 for a train-fitted model. Public LB top on 11 Nov is 0.3935.

---

## 1. This competition: public information found

### 1.1 Official dataset readme (copied in a participant repo) [V]
Source: a participant repository, file `competition_readme.md` (the official `readme.md` shipped with the data).
- Sizes: train 3,151,208 feeds (512 files, 81.56M embedding rows); val 14,423 feeds (2.52M rows); test 28,595 feeds (4.71M rows).
- Dates: train 2026-09-06 to 09-19, val 09-22 to 09-24, test 09-26 to 09-29. **09-25 is a gap.** Train is meant to be used whole; "Val is built separately from train **by the same rules as test**, to reproduce its distribution."
- Val/test contain **only feeds whose candidates were uniformly shuffled on production before display**. Labels come from feedback after those displays. Files are also shuffled with one common permutation.
- **"Items the user had already viewed are excluded from all parts of the dataset."** In train the remaining candidates keep their original `pos`, so numbering has gaps.
- History: up to 256 events, newest first, "the latest events after filtering". Event types 1..10, sources 1..30, categories 1..53 (0 = unknown). IDs are shared across splits; 0 is reserved for missing.
- "Numeric features are already prepared for the model. **Encoders were fit on train only** and applied identically to val and test."
- Each split has its own embedding table, so embedding indices differ across splits for the same item_id. −1 means no embedding.

### 1.2 Official baselines and participant repos
- **Participant repository A** [V]
  - Baseline B0: CatBoostRanker YetiRank, 9 features (f1..f5, category, embedding dot, cosine, missing flag), full train, GPU.
    - Full val **0.35334**, Public **~0.34607**.
    - Refit on train+val: Public ~0.34601 (no gain).
  - H1 = B0 + count of history events in the same category: val 0.35404 (+0.0007), Public 0.34614.
  - Official neural baseline = candidate-set Transformer (user token + candidate tokens; 2 layers, width 64, 4 heads; BCE; checkpoint chosen by val logloss, not NDCG). **It does not use history.**
  - Metric note: DCG over the full feed, normalized by IDCG@30. With ≤30 positives per feed this equals full NDCG.
- **Participant repository B** [V]: CatBoost YetiRank, 33 features, Public **0.35185** (ensemble of depths 7/6/5).
  - History: in_hist, count/share of candidate category in history, recency Σ1/sqrt(pos+1), match to last and to top-3 last categories, history length.
  - Similarity: cosine to the **last** history item (top-3 importance).
  - Feed-relative: diff to feed mean and normalized rank of f1..f5 / cos / dot, share of the candidate's category in the feed.
- Two more participant repositories [V]: infrastructure only, no scores.
- Announcement [V]: target = 1 if at least one event from the target-action set occurred; all target events weigh the same. Top 10 must submit code. 5 submissions/day.

Takeaway (inference): every public solution so far is a feature-engineered GBDT around 0.346–0.352 Public. The 0.389–0.394 cluster at the top must come from something structurally different: training on the big train set with debiasing, history sequence models, transductive or leak features, or a mix.

---

## 2. The production system behind the data (Avito Habr, primary)

### 2.1 "Main-page ranking: from CatBoost to neural networks" (AvitoTech, 7 Oct 2026) [V]
https://habr.com/ru/companies/avito/articles/1090584/
- Main-page recs ≈ half of ad views and ≈30% of buyer-seller contacts. Pipeline: several candidate generators (thousands of candidates) → ranking → **diversification layer**.
- CatBoost evolution: 2020 had 80K impressions, binary "viewed" target, classification, 19 counters. 2026 has >1M impressions, a **composite target from target actions** ("besides contacts, includes e.g. adding to favorites"), a **ranking loss**, and ~200 features including embedding features and neural predictions.
- Serving: **CatBoost L2 selects top 300 → neural L3 reranks them.** Early candidate-feature binding beats late fusion but is only affordable on the top 300.
- Neural data: 10M impressions.
  - Categorical: per-feature embedding tables.
  - **Numeric: 128 quantile bins per feature, stored as (left bin, right bin, weight).** The interpolated bin position is normalized to [0,1], which keeps monotonicity.
  - Embeddings: memmap; each embedding version gets its own projection.
- Model iterations (offline NDCG vs CatBoost; online A/B):
  1. DCNv2 + MoE written by an AI agent, BCE: −3% NDCG, −6% users with contacts.
  2. **Plain MLP + ListMLE** (sort by target; each step is a softmax over the remaining candidates; a triangular mask computes all steps in parallel): −1.5% / −3%.
  3. Embedding projections + pairwise dot-product layer: **+0.7% NDCG** (first win). Online contacts −1.8%, target actions +0.7%. Favorites dominate the dataset.
  4. Pairwise layer removed; **loss weights penalize contact misranking more.** Rank blend CatBoost:NN = 1:1 in the top 300 (deliberately untuned): +0.7% users with contacts, +0.9% users with target actions.
  5. **History transformer**: history "filtered to useful target actions", baseline capacity **50 events**. Clicks were excluded so they would not crowd out intent signals. Each event = item-embedding projection + positional embedding + event-type and source (recs/search/external) embeddings, summed. Candidates use item embeddings only. **History and candidates go through one transformer**; candidates attend to history; **letting candidates also attend to each other (masked self-attention) added ~0.6%**. The output feeds a parallel DCN. Loss: ListMLE. **About 2.5M parameters.** Offline **+7.2% NDCG (target actions), +4.9% (contacts)**. Online +4.3% users with contacts, +3.3% with target actions, +0.8% with a deal.
- Not disclosed: depth/width, target weights, negative sampling, **position-bias handling or exploration traffic** (none mentioned).

### 2.2 "Transformers in recommendations" (Mikheev, AvitoTech, 27 Feb) [V]
https://habr.com/ru/articles/1004694
- Two-tower model.
  - User tower: transformer over ad actions and search queries (separate subsequences, positions kept). Switched from one CLS vector to **several user embeddings**, each scored against the ad and then aggregated.
  - Item tower: top-10K categorical attributes as embeddings; numeric features via **embeddings at bin boundaries combined as a weighted sum**.
- Training: positives = target-action ads. Random gap of minutes to days between history end and target. **Events from the target's session are removed from history.** In-batch + random-catalog negatives with cross-batch sharing. InfoNCE slightly better than SigLIP. Batch ~30K → ~130K. ~10B samples over 2 months.
- Used as a ranker feature (dot product): **+2.3% target actions; top-1 SHAP feature.**
- Mapping (inference): our int8 128-d user/item embeddings are likely tower outputs of this kind. dot(user, item) and history-vs-candidate similarity are the production-validated features. This also explains why candidates never appear in their own history.

### 2.3 "Diversity on the main page" (9 Dec 2025) [V]
https://habr.com/ru/articles/974682
- Five candidate generators (1 offline, 4 online) → CatBoost (scores each ad independently, uses transformer scores) → **category blender**.
- Blender: group by category keeping ranker order. For each slot, **sample a category from the user-interest distribution** and take the best remaining ad from it. Interest = exponentially time-decayed weighted events, favorites > clicks. Smoothing controls diversity.
- Interest model: Transformer encoder, dim 64, 2 layers, history 512, ~1M params, HardSwish.
  - Event = sum of event-type/category/microcategory/source embeddings, with feature dropout; learnable positions (last to first); **event age mapped to [-1,1] over 2 months through a small MLP**.
  - Target = category shares of favorites+contacts over the next 7 days; CE loss. +1% users with contacts over the heuristic.
- Implication (inference): train `pos` = blender slot. Position reflects category sampling plus within-category ranker order, not a single relevance score. Uniformly shuffled val/test remove both.

### 2.4 "From Markov models to SASRec" (Data Fest × AvitoTech 2024) [V]
https://github.com/avito-tech/avitotech-presentations (PDF "От марковских моделей до SasRec.pdf")
- 28% of users contact from the main page. Level-1 generators recommend **by category** (tree of ~6K microcategory nodes).
- Markov microcategory-transition baseline: +0.9% users with contact from main; deeper tree: +0.5%.
- SASRec over microcategory sequences. Lessons:
  - The ranker must be retrained on the new candidates (A/B 1: −2% precision@5 at first).
  - **Training on contact events only** looks better but gives less diverse output.
  - Off-the-shelf transformer libraries did not work; custom models did.
  - Predicting next-session items was better offline but worse online.
- The talk's author is listed as the task author of ML Cup 2026 [V, ods.ai track page].

---

## 3. Avito ML Cup 2025 (personal recommendations, Recall@40): 1st place code

Sources: <public repository, link removed> ("Avito ML cup 1st Place solution (RecSys track)"). Files read: `hack.ipynb`, `autoencoders/model.py`, `utils.py` [V]. Public LB: #1 at 0.23147 [V, https://ods.ai/competitions/avitotechmlchallenge2025/leaderboard].

- Task: predict nodes (item groups) a user will **contact** among those not interacted with. Recall@40.
- Local validation [V code]: last 14 days = eval. GT = contact events on (user, node) pairs unseen in train, with user and node both present in train.
- Scorers (unseen-recall on local val):
  - **EASE-DAN** ("Why is normalization necessary for linear recommenders?", arXiv 2504.05805). Binary user×node matrix, top 30K nodes, reg 10, drop_p 0.0498, beta = alpha = 0.952. **R@40 0.1735, R@100 0.2820.**
  - **RDLAE** (relaxed-diagonal linear AE, arXiv 2305.12922). Top 50K nodes by contacts, reg 500, drop_p 0.7, xi 0.1. Input X = binary + 2×(contact). Scores multiplied by log1p(**Bayesian-smoothed node contact ratio**). Extra score features from per-pair matrices (counts, contact counts, #days with clicks/contacts, exp position decay 0.01, exp time decay 0.05). **R@40 0.1994, R@100 0.3136.**
  - **SASRec (RePlay)** over nodes: 2 blocks, 2 heads, hidden 128, dropout 0.3, max_seq 100, lr 1e-3, sampled softmax with **4000 negatives and negative sharing**, 10 epochs; best R@40 0.1629. A second SASRec runs **over categories** and its score is joined by (user, category).
- Ranker features [V code]:
  - User: contact ratio, last contact category, last seen category, most frequent location.
  - User × category: total/positive counts and ratios, share of the user's contacts in this category.
  - Item and category: contact counts and ratios, also counted once per unique user.
  - Crosses: node_ratio×user_ratio, category_ratio×user_ratio, category == last contact / last seen.
  - Scorer scores and ranks.
- Ranker training: candidates = RDLAE top 300. **Only users with ≥1 positive; ≤100 positives and ≤30 random negatives per user.** 5-fold KFold over eval users, Optuna. LightGBM `lambdarank` / `rank_xendcg` reaches R@40 0.19–0.21; **CatBoost YetiRank (GPU) 0.21–0.22**. Final note in the notebook: "**the best submission was pure CatBoost**, boosters averaged over folds"; scorers retrained on the full clickstream for inference.

Transfer (inference):
- (a) Linear-AE item-item weights (EASE/RDLAE) on the item_id × user matrix from all histories (3.15M train feeds plus val/test histories). Feature = Σ over history items of W[h, cand], weighted by event type and recency rank. Coverage risk: Avito ads are short-lived, so first measure the share of val candidates present among the top-N history items.
- (b) Bayesian-smoothed per-item and per-category "contact propensity" from history event types (which event_type codes mark strong intent has to be found in EDA).
- (c) SASRec over category sequences → P(next strong-intent category).
- (d) Strong confirmation that CatBoost YetiRank with fold averaging, trained only on users with positives and subsampled negatives, is a solid ranker.

---

## 4. Avito ML Cup 2024 (ad CTR, per-user ROC-AUC): 2nd-place code

Sources: <public repository, link removed> ("The second place solution"; the Habr results post https://habr.com/ru/articles/864532 lists this author 3rd, after two other teams, so the place differs between sources) [V]. Files read: `pipeline.ipynb`, `nn_predictions.py`, `custom_roc_auc.py`.
- Metric: per-user ROC-AUC averaged over users with both classes (`compute_group_auc`), i.e. a within-group ranking metric like ours.
- **Sliding daily windows**: for 8 offsets, the target day is one day and history is all earlier days. Train rows are the 8 target days stacked; test uses all train days as history.
- Features [V code]:
  - **Target mean for every 1-, 2- and 3-way combination** of [user, campaign, creative, banner, location, logcat] over history (−1 if unseen).
  - For non-user combos: **frequency of the combination within the target day itself** (transductive), frequency on the last history day, and their ratio (a "trend").
  - nunique crosses (e.g. #distinct creatives per campaign); grouped nunique between combo pairs on all history and on the last day.
  - **pos_neg_stat**: per campaign/banner, positives weighted by each user's negative count (and vice versa). A pairwise-aware target encoding that matches per-user AUC.
  - **NN prediction**: campaign embedding (8-d) + banner one-hot → MLP trained with **BPR loss** on per-user (pos, neg) samples, used as a feature.
- Training set: only users with both classes. **CatBoost YetiRank, group = user, 20,000 iterations, depth 6, lr 0.05, GPU.**
- Other repo: <public repository, link removed> (11/304) [V]: cumulative per-key views/clicks, previous-day stats, per-banner/platform cumulative CTR.

Transfer (inference): out-of-time target encodings on train feeds (item_id, category, user×category, item×hour, ...). For train-feed rows use only prior days; for val/test use all of train (val starts 3 days after train ends, test 7). Add the pairwise-aware variant (positives weighted by the feed's negatives). Add "frequency of item in the current day's feeds" computed inside each split (transductive). Train only on feeds with ≥1 positive.

---

## 5. Avito ML Cup 2026 (spring; retrieval, Recall@160)
- Task [V, https://ods.ai/competitions/avitotechmlcup2026]: rank up to 160 ads per user that the user will **contact**. Data: 5B events from 21 Jan to 22 Apr 2026, a 12-hour gap, then eval contacts for 94,408 users. Recall@160 over 10 buckets (5 verticals + 5 holdout), public/private 20/80.
- Winners (Data Fest 2026 Avito track) [V, https://ods.ai/tracks/df26_avitotech_mlcup2026]: places 1 to 3 are named there. Talk content and slides were **not found**.
- A participant (30th private) [V, <public repository, link removed>] defines a contact as "call / message / phone reveal". Progression: popular 0.0006 → **item-item co-visitation with time decay and contact boost 0.0795** → LightGBM LambdaRank 0.0885 → CV ensemble 0.0915. Also two-tower over semantic IDs (sid_0..3), ALS, SASRec on sids, personalized PageRank.

---

## 6. VK RecSys Challenge 2024 (VK Clips; predict like / dislike / ignore for given user-clip pairs)

This is the closest Russian analogue to our setting: candidates are given, a within-user pairwise metric, histories and content embeddings.
Organizer page [V]: https://ods.ai/competitions/aivkchallenge. Metric = **per-user ROC-AUC over 3 levels** (like=1 > ignore=0 > dislike=−1), ties 0.5. Oct–Dec 2024, 999 participants, 9,255 submissions.
Organizer slides (Poslavsky, AI VK) [V, https://storage.yandexcloud.net/ds-ods/files/content/2025/01/22/408d9bd47c7c/1_Пославский_Александр_соревнование.pdf]:
- 183K users, 337K clips (author, duration, content embedding), 145M interactions (timespent, like, dislike, share, bookmark). 6 weeks train / 1 week test.
- **The item like-through-rate alone gives AUC 0.6175 = top 33% of teams.** Public top: 0.6855, 0.6849, 0.6802, 0.6782, 0.6776 [V, leaderboard page].

### 6.1 2nd place: Kirill Khrylchenko (Yandex), pure neural, "two submissions were enough" [V]
Slides: https://storage.yandexcloud.net/ds-ods/files/content/2025/01/22/1089653131d0/7.%20Хрыльченко%20Кирилл.pdf
- No feature engineering and no domain or data exploration.
- Model: a **Transformer over events** (author, clip, action) plus static features (gender, age). Candidate encoder = trainable clip vector + author + **content embedding** + duration → DCN + ResNet. A top DCN + ResNet head **predicts pointwise like / dislike / other labels**.
- **Pre-training**: reorder each event triple (author, clip, action) as (action, author, clip) and **predict author and clip given history and action**; 5 epochs.
- **Fine-tuning**: predict the action given history, author and clip; like prediction upweighted; 2 epochs. **score = P(like) − P(dislike).**
- Architecture: 4/6 layers, hidden 512/768, **history length 512/1024**, pre-norm, **SwiGLU, ALiBi, RMSNorm**. Cites DCN-V2 and HSTU.
- The Yandex team "ambitious" (Makeev, Andreev, Baikalov, Tytskiy, Krasilnikov, Khrylchenko) uses the same name as the public #1 here; that it is the same team is inference.

### 6.2 3rd place: Ivan Bragin, "CatBoost is all you need … for 3rd place" [V]
Slides (image-only PDF; I rendered and read every page): https://storage.yandexcloud.net/ds-ods/files/content/2025/01/22/84bae586eaf2/3.%203%20место%20Иван%20Брагин.pdf
- Validation: a "features" window, then a "train" window (labels), then a "val" week, mirroring the test week. Both **relative** (expanding) and **absolute** (fixed 3-week) feature windows.
- Features (~250):
  - **CTR and count per key** for like / dislike / share / bookmark / view_time / view_percent (normalized). Keys: item, user, (user, item), (item, user_age_bin), source, **(item_emb_kmeans, user)** and combinations.
  - **Content similarity**: video embedding vs popular videos, vs **user vectors = mean(liked), mean(disliked), mean(viewed) embeddings**, and vs source vectors built the same way. Raw embeddings, better reduced (t-SNE). Mean age/gender per video/user/source.
  - **ALS on every feedback type** (like, dislike, share, bookmark, watch_percent, watch_seconds) → per-type predictions as features.
  - **Boosting-as-feature**: a first CatBoost trained on an earlier 2-week window predicts each target type; its outputs become features of the final model (stacking over time windows).
  - "User action calibration": normalize per-user rating habits ("gives everyone 1 star").
- Training: CatBoost classifier on 90M rows × 250 features (~20 h, with weight calibration). **Last week switched to a CatBoost ranker with PairLogit** (70M rows, ~3 days). Snapshot continuation 10K → 30K trees (random_strength decays with iteration count, which caused trouble). Hardware: rented RTX 4090 with [n] GB RAM (~$400) plus Databricks (~$50).

### 6.3 4th place: greenwolf-nsk [V]
https://github.com/greenwolf-nsk/vk-recsys-24 — Public 0.6776, Private 0.6798. Pipeline: rank column → split → basic features → **MF scores** → **BERT4Rec checkpoints (768-d, max seq 512) turned into features** → 5-fold CatBoost. Needs [n] GB RAM and an 80 GB GPU.

### 6.4 5th place: Stanislav Chistyakov, simple NN [V]
Slides: https://storage.yandexcloud.net/ds-ods/files/content/2025/01/22/0900453cc4ee/2.%205%20место_Чистяков%20Станислав.pdf
- **Training mimics the test format**: shuffle, split each user's items into **groups of 20**, drop groups with a single target value, batch 4096.
- Model: user_id emb (384) + gender + age; item_id emb (256) + source emb (128) + duration + content embedding → MLP. **Heads for everything known**: timespent (MSE), like/dislike/share/bookmark (BCE), target −1/0/1 via a **within-user margin loss** (margin 1 or 2 by label gap). Losses summed.
- **ALS initialization of the id embeddings** (plain binary interaction ALS was best; ALS weighted by target or timespent was worse) took public 0.62–0.645 to 0.645–0.654. **Freezing embeddings** after overfitting began (5–8 epochs frozen + 1 epoch unfrozen) gave 0.67+.
- Single model 0.6732 → **seed average over validation splits 0.6773** → plus earlier versions 0.6782.

---

## 7. VK RecSys Challenge 2025 "LSVD" (reverse task: top-100 users for cold clips)
- Dataset paper [V, https://arxiv.org/html/2602.04567v2]: 40B+ exposure events, 10M users, ~20M clips, 25+1+1 weeks. Context: place (feed/search), platform, agent. Feedback: timespent (cap 255 s), like, dislike, share, bookmark, author click, comments open. 64-d SVD content embeddings. iALS baseline (watch >10 s positive): GTS ROC-AUC 0.581, NDCG@20 0.026.
- Task: NDCG@100 per clip, each user usable at most 100 times; ~800 participants. Winners' methods were **not found**. Write-ups that were found:
  - A participant [V, https://www.pvsm.ru/kollaborativnaya-fil-tratsiya/442261]: mean embedding of a user's clips plus action-rate features plus a greedy capacity balancer.
  - A ~0.333 NDCG heuristic [V, https://github.com/Voldisoriginal/VK-LSVD-Hackaton]: weighted user-author interest score.
- Relevance to us: low (cold-item, reverse ranking).

---

## 8. Yandex: papers, competition write-ups, Yambda

### 8.1 ARGUS: scaling recommender transformers to 1B (arXiv 2507.15994) [V]
https://arxiv.org/html/2507.15994v1
- Pre-training = **next-item prediction** (logQ-corrected sampled softmax, mixed negatives, cosine/temperature) + **feedback prediction** (sum of K CE heads given history, context and item). Each event is collapsed to one embedding.
- Fine-tuning: user state matched to impressions with a simulated 1-day delay. Score = ⟨h_t, i⟩. **Pairwise logistic loss on adjacent impression pairs with different feedback.** Used as **one feature in a ~1000-feature GBDT ranker**.
- Pairwise-accuracy uplift: **standalone −5.33% vs as a GBDT feature +2.32%** (126M model).
- Ablations:
  - No pretraining: +1.17% (feature). Pretraining + 1 week of fine-tune: +0.63%. Pretraining + 1 year: +2.32%.
  - Fine-tune context **512 → 2048 → 8192 gives +1.01 → +2.32 → +2.77%**.
  - Size: 3.2M params +1.35% … 1B params +2.66%.
- Production: +2.26% listening time, +6.37% like likelihood.

### 8.2 Personalized Transformer-based Ranking for e-Commerce at Yandex (arXiv 2310.03481) [V]
https://arxiv.org/html/2310.03481v2
- User tower: bidirectional transformer over clicks, add-to-cart, favorites, purchases and web-search queries (one fused sequence). Event = content (title/query tokens) + **reversed positional embedding** + event-type embedding. 4 layers, hidden 256, up to 1024 events. Item tower: summed title tokens + residual MLP.
- **Stage 1** = sampled softmax (in-batch + cross-GPU negatives). **Stage 2** = fine-tune on **impressions grouped by request (groups without positives dropped)** with a calibrated pairwise BPR-sigmoid loss per action plus a small click BCE (weight 0.1). One epoch, since longer overfits.
- **Context debiasing**: a separate context tower outputs r_ctx (learned scalars for surface and device). Training predicts sigmoid(r_ui + r_ctx); **at inference the context tower is detached.** Fusing context into the towers or adding dropout did not help.
- Gains (relative NDCG of the GBDT when the cosine is added as a feature, retargeting / discovery):
  - Retrieval-only pretraining: +0.23 / +0.88%. **Two-stage: +0.48 / +1.38%.** Fine-tune only: +0.03 / +0.03%.
  - Context debiasing: +0.02 / +0.014 points. History 256 → 512: +0.02 / +0.18 points.
  - Online: Market homepage +6% recommendation orders.

### 8.3 RecSys Challenge 2025 4th place, team "ambitious" (Yandex) (arXiv 2508.06970) [V]
https://arxiv.org/html/2508.06970v1
- Causal transformer (20 layers × 1024, 252M) over a merged sequence of 5 event types (256 most recent).
  - Content: price bins, quantized 16-d LLM text embeddings, multi-size unified ID embeddings.
  - Context: event type, position, day of week, hour, **log time since last event / since first event**.
  - Multi-task losses: next event type (CE), time delta (MSE), next URL and **next item (sampled softmax, 8192 in-batch + 8192 uniform negatives, logQ)**.
- Next-item prediction mattered most (removing it hurt every task). 126M ≈ 252M on some tasks.
- DCN-v2 head on **piecewise-linear encoded numeric features** (64 quantile intervals × 32) plus a contrastive loss (+0.009 to +0.014 on two tasks). Handcrafted features (>300; windows 7/14/28/60 days/all; kmeans clusters of item-text embeddings; exponentially weighted aggregates) "crucial".

### 8.4 Yambda-5B (Yandex Music dataset, arXiv 2505.22238) [V]
https://arxiv.org/html/2505.22238
- 4.79B events, `is_organic` flag (recs-driven vs organic; ~42–49% of listens/likes/dislikes are recs-driven), audio embeddings.
- Protocol: **Global Temporal Split with a 30-minute gap** and user states frozen at test start. Leave-one-out is rejected.
- Baselines: SASRec is best on Listen+ at 500M/5B scale. **DecayPop (time-decayed popularity) is best on explicit Likes NDCG@10 at every scale.** The event-source analogue for us is `event_source` (organic search vs recs).

### 8.5 Yandex Cup 2022 ML RecSys "Next Like", 2nd place (greenwolf-nsk) [V]
https://github.com/greenwolf-nsk/yandex-cup-2022-recsys
- Candidates: next-item co-occurrence, **"smart co-occurrence" aggregated over the last N history items with 1/(rank+1) weights** (recall 0.49 → 0.62–0.64, the main gain), implicit BM25 item2item, ALS, popular among last items, last-artist tracks.
- LightGBM lambdarank: 3-fold, **negative downsampling 0.3**, custom MRR early stopping, lr 0.04, L1≈1, L2≈8, colsample = subsample = 0.6.
- Features: per-generator score and rank, co-occurrence aggregates (mean/max/std/min), item/artist stats over history offsets (last 10/50 actions). **Final: inverse-rank blend of 3 submissions** (public 0.0849 / 0.0854 / 0.0861).
- Yandex Cup 2023 "RecSys" was music-tag multilabel classification (3rd place traptrip, AP 0.311 private) [V, https://github.com/traptrip/yandex_cup_ml_2023]. Later Yandex Cup ML tracks had no recsys ranking task found [V, https://yandex.ru/cup/ml/analysis/ only lists a genre-classification recsys task]. Not relevant.

### 8.6 Khrylchenko, "ML trends in recommender technologies" (Yandex, Nov 2024) [V]
https://habr.com/ru/articles/857068 — qualitative. Neural rankers beat GBDT through explicit crosses (DCN-v2), learnable high-cardinality embeddings, reuse of upstream embeddings, end-to-end sequence encoders, multitask, scaling; histories growing toward ~8K events. Selection bias from candidate generation and feedback loops are named as problems. No numbers.

---

## 9. Other Russian contests
- **Sber AI Journey 2023 "Personal AI"** (winner from Alfa-Bank) [V, https://habr.com/ru/articles/849420]. Code ran blind on the organizers' servers (100 min, A100).
  - Music: **co-occurrence candidates at several granularities** (last track, last two, ×day, ×user, including test sessions) → CatBoostRanker. Strongest features: **candidate rank under each generator**, and **skip-rate of track, artist and user×artist**.
  - Marketplace: co-occurrence over the last N items weighted by event type and recency + ALS. Features: per-method ranks, median rank, number of methods with the item in the top 10, **view→purchase conversion of item and category** (favor items people convert on, not just view).
- **MTS ML Cup 2023** (age/gender from domain visits) [V, https://habr.com/ru/post/709602]: baseline = implicit ALS user×domain (50 factors) → CatBoost (score 1.485). No winner write-up found. Not a ranking task.
- **Avito Context Ad Clicks (Kaggle 2015), Owen Zhang 1st** [V, https://github.com/owenzhang/kaggle-avito + WinningModelDocumentation PDF].
  - XGBoost only, 65 features. Position useful.
  - **Sequence features ("# of ads seen up to this impression"), average prior response, entropy of user history, credibility-adjusted CTR encodings.**
  - Negatives downsampled 50:1, seed/sample bagging. Keeping only the last 200 impressions per user barely hurt.
- Not found despite searching (RU and EN): any T-Bank/Tinkoff, Alfa or Lamoda ranking contest with public top solutions. Ozon E-CUP 2025 recsys track: only the announcement [S]. A Wildberries RecSys meetup talk "Transformers in WB & Russ" exists [S], content not found.

---

## 10. What was searched but NOT found (gaps)
- Avito ML Cup 2025 2nd/3rd place write-ups; Avito ML Cup 2024 1st- and 2nd-place solutions; Avito ML Cup 2026 (spring) winners' slides/code (talk titles only).
- VK RecSys Challenge 2024 1st place; VK LSVD 2025 winners (only an award video on VK Video, not transcribed).
- Any Telegram summaries of the current competition (chat not indexed). No public repo above ~0.352 Public.
- Avito has never published how exploration (uniformly shuffled) traffic is used, or what exactly the target-action set is (contacts + favorites per 1090584; contact = call/message/phone reveal per a participant).

---

## 11. Mapping to our data: concrete plan (priorities, how, expected impact)

Expected impacts are my estimates unless a source number is quoted. Validate everything with feed-grouped K-fold on val (14.4K feeds; SE of NDCG ≈ 0.003). Do not trust Public (~8.6K feeds).

### HIGH
**H0. Same-user later-history check (1 hour).** The readme says seen items are removed and history = latest events. For val users with ≥2 feeds, test whether earlier-feed positives appear in the later feed's history and with which event_type/source. Do the same on train, where users have many feeds. If so, add a feature "candidate appears in a later history of the same user (by event type)" built inside each split. Also check the rules. Potential gain: large on affected feeds; the overall effect scales with the share of feeds that have a later same-user feed (~5% in val/test if ~1.05 feeds/user). (inference)

**H1. Use the 3.15M train feeds with position debiasing** (Yandex context tower [V]; Avito trained its NN on 10M impressions [V]).
- Neural: logit = f(x) + g(pos_bucket, M_bucket [, pos×category-slot]). Train with softmax CE or ListMLE over feeds with ≥1 positive. Drop g at inference.
- GBDT variants, compared on val:
  - (a) `pos` and `pos/M` as features, set to a constant at inference;
  - (b) drop pos;
  - (c) weight rows by 1/propensity(pos) estimated from the NN's g.
- Caveat (inference): in train, pos is a function of the L2 score, the blender and possibly f1..f5, so g and f compete for credit. Fine-tune or stack on val folds to correct. Expected: the largest structural gain (we fit on 14K feeds today; production-scale training is what made Avito's NN beat CatBoost).

**H2. Avito "model 5" clone: joint transformer over [history ; candidates]** [V, 1090584].
- Tokens:
  - History event = proj(item_emb_int8/127) + emb(event_type) + emb(event_source) + emb(category) + emb(position).
  - Candidate = proj(item_emb) + emb(category) + quantile-interpolated f1..f5 (128 bins, Avito's encoding) + user_emb token.
- Attention: candidates attend to history, and to each other with a mask (+0.6% at Avito).
- Head: DCN/MLP; ListMLE/softmax listwise loss; ~2–10M parameters is enough (Avito: 2.5M).
- Ablations: history = all 256 events vs strong-intent event types only (Avito: target actions, ≤50, no clicks).
- Train on train (H1 bias term), fine-tune on val folds. Expected: best single model.

**H3. Pretrain the history encoder** (Khrylchenko VK'24 2nd [V]; ARGUS [V]; ambitious RecSys'25 [V]).
- Data: all histories in train/val/test, deduplicated per user.
- Objectives, newest-first sequences reversed to causal order:
  - (i) next item: predict the frozen 128-d embedding of the next event's item with InfoNCE (in-batch + random negatives, logQ correction);
  - (ii) next event_type, source and category (CE);
  - (iii) Khrylchenko's trick: condition on the next event's action type and predict its item.
- Then fine-tune as in H2. ARGUS: pretraining roughly doubled the gain; longer context helped (we are capped at 256).

**H4. NN scores as GBDT features plus a rank blend** (ARGUS: feature +2.32% vs standalone −5.33% [V]; Avito 1:1 rank blend [V]). Generate out-of-fold NN scores on val for the LightGBM/CatBoost stacker. For test, average the fold models. Also blend final rankers by ranks.

**H5. Out-of-time item/user target encodings from train feeds** (Avito'24 2nd, Avito'25 1st, VK'24 3rd [V]).
- Keys: item_id, category, user×category, item×hour-of-day, kmeans(item_emb, 256–4096)×user, kmeans×category.
- Stats: positives/impressions; **position-debiased CTR** = Σ target / Σ E[target|pos] with E[·|pos] from train; counts. Bayesian smoothing.
- For train rows: prior days only (sliding daily windows as in Avito'24). For val/test: all train. Add pairwise-aware TE (positives weighted by the feed's #negatives).
- First measure coverage: share of val/test candidates with ≥k train impressions. Expected gain: high if coverage > ~30% (inference).

### MEDIUM
- **M1. Collaborative item-item scores** between candidates and history items: EASE-DAN/RDLAE (Avito'25 1st [V]), or co-occurrence from histories with **1/(rank+1) recency weights** (YC'22 2nd, recall +0.15 [V]), weighted by history event type. Aggregate as max/mean/sum over history.
- **M2. Event-type- and source-specific history centroids**: cosine(candidate, mean of history embeddings with event_type = k) for each k; same per source (VK'24 3rd [V]). Plus a finer user×cluster affinity via kmeans(item_emb) instead of 54 categories.
- **M3. Category interest model** (Avito production [V]): a small transformer predicting the category distribution of positives (target = categories of positives in train feeds) → feature log P(cat(cand) | history), plus its rank within the feed.
- **M4. Transductive popularity in the scored period**: count of an item's appearances as a candidate across val/test feeds (per split, per day), and in val/test histories by event type. Analogues: Avito'24 2nd target-day frequency [V], Yambda DecayPop [V].
- **M5. NN training format and regularization** (VK'24 5th [V]): whole feeds as groups; multi-head auxiliary losses (e.g. predict event types of the user's next history events); seed/fold averaging (+0.004 AUC there); freeze pretrained embeddings early, unfreeze for one epoch.
- **M6. GBDT recipe** (Avito'25 1st, VK'24 3rd, Avito'24 2nd [V]): CatBoost YetiRank and PairLogit vs LightGBM lambdarank/xendcg; only feeds with positives; subsample negatives (≤30 per feed in Avito'25 1st) if training on train scale; fold-average.

### LOW
- User calibration features (VK'24 3rd [V]): user-level share of strong-intent events, normalized affinities.
- History entropy/diversity, "number of items seen so far" (Avito 2015 [V]).
- Event-age features: unavailable (no history timestamps); history index is the only time proxy, used as Avito does with reversed learnable positions.

### Validation protocol recommendations
- K-fold over val feeds grouped by user (users can repeat), fixed folds for every model so OOF stacking is valid. Never select on Public.
- When training on train, also check calibration on the last train days vs val. The 09-25 gap and the val→test shift make recency matter: weight recent train days more, or fine-tune on val (RecSys'24 1st place did this, see research_kaggle_solutions.md).
