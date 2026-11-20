# Web research: Avito ML Cup 2026 "Personal ranking" and related Avito material
Date: 2026-11-11. Web search only; no public winning writeups found for this or earlier cups.

## 1. This competition (ods.ai/competitions/avitotechrecsys2026)
Source: https://ods.ai/competitions/avitotechrecsys2026 (+ /dataset, /leaderboard)
- Task: rank candidates per feed_id; binary target = at least one target action on the ad (all actions equal weight).
- Metric: NDCG, per-feed mean; tied scores place negatives first; feeds without positives = 1 (builder only keeps feeds with positives). Page also references IDCG@30 (ambiguous; our local check says full-list).
- Val/test: candidates uniformly randomly shuffled before display; original positions not published.
- Public = earliest ~30% of test feeds, Private = ~70% (same-timestamp feeds stay together; users may appear in both parts).
- Limits: teams of 4, 5 submissions/day, ends Dec 9 2026 (final selection 00:00-12:00); top-10 teams must submit solutions.
- Prizes: 750k / 450k / 300k RUB. Discussion in a Telegram chat (not accessible via search).
- Official baselines (Yandex Disk, not fetched): CatBoostRanker (YetiRank) on numeric + categorical + embedding-similarity features; and a Transformer jointly processing user embedding and the feed's candidates. Baseline scores are NOT published on the page.
- Leaderboard snapshot (11 Nov): 1) 0.39354 (50 entries), 2) 0.38995, 3) 0.38953, 4) 0.38910, 5) 0.38871, 6) 0.38863, 7) 0.38855, 8) 0.38847, 9) 0.38837, 10) 0.38836. Note the cluster 0.3884-0.3900 in ranks 2-10 and a gap of ~0.0036 to #1; many users have 30-50 submissions, so public-LB overfitting is plausible (public set ~8.6K feeds, SE ~0.003).
- Participants ~1100, 4591 submissions at fetch time.
- No public GitHub repos / Habr posts / Telegram summaries for this contest were found. A Kaggle dataset "Avito ML Cup 2026" exists, content not retrievable. (Other contest at slug avitotechmlcup2026 = relevance retrieval, Recall@160, Apr-May 2026, different task.)

## 2. Previous Avito ML Cups
- 2024: CTR/ROC-AUC on anonymised ad campaign data, 891 participants, winners named in the Habr post. https://habr.com/ru/articles/864532 . No solution writeups found.
- 2025: two tasks (personal recommendations, Recall@40, 30/70 public/private; duplicate search). Ended May 31 2025, 1704 participants. https://ods.ai/competitions/avitotechmlchallenge2025 . Kaggle dataset: <Kaggle dataset, link removed> . A YouTube stream "Как подступиться к задаче про рекомендательную систему на Avito ML Cup" exists (https://www.youtube.com/watch?v=cCXjxPGUFG4), not watched. No winner writeups found via search.

## 3. Avito tech publications (Habr)
### Transformers in recommendations (https://habr.com/ru/articles/1004694)
- Old stack: candidate generators (item2vec, user2vec = time-decayed sum of ad embeddings) -> ranker using generator scores and click features -> final assembly with category sampling.
- New: two-tower; item tower cached (Redis, ANN index); user tower = Transformer over action history (ad actions and search queries as separate subsequences, positional embeddings). Single CLS user vector was too low-capacity -> multiple user embeddings, each scored vs ad, scores aggregated. Feature encoders: DCN crosses per feature group; numeric features encoded via bin-boundary embeddings with proximity-weighted sum (cf. "normalized interpolated bins" here); text/infomodel params as summed token embeddings.
- Training: pair (history, target ad with target action); random time gap (minutes-days) between history end and target; same-session events leading to the target removed from history (cf. candidates never appearing in own history here). In-batch + random catalog negatives, negative sharing; InfoNCE slightly better than SigLIP; batch ~30K -> ~130K; ~10B samples over 2 months.
- Use: as candidate generator and as a ranker feature; ranker feature gave +2.3% target actions on main page and was the top-1 SHAP feature; single-category CG +1.1%.
### Main page diversity (https://habr.com/ru/articles/974682)
- Main feed: 5 candidate generators -> CatBoost ranker scoring items independently -> category blender. Blender samples categories from a user-interest distribution (time-decayed weighted events, favorites > clicks). A 2-layer transformer (dim 64, 512 events, event type/category/microcategory/source embeddings, event-age MLP in [-1,1] over 2 months) predicts category distribution of favorites+contacts over next 7 days. A/B: +2.5% contacts (heuristic), +1% more (model).
- Implication: production (train) feeds are post-blender and category-diversified, so train position carries category-sampling structure, unlike uniformly shuffled val/test.
### Other (not fetched in detail): https://habr.com/post/417571/ (similar ads, bandits), kod.ru Avito Ranker 3 (search ranking).
- Exploration traffic and the exact definition of "target actions" are not described in any article found. Avito speaks of "contacts" (chat/call), favorites as target-like actions.

## 4. Takeaways for modelling (inference from sources)
1. Avito's own best signal is a user-history transformer dot-product with the ad embedding (the given 128-d user/item int8 embeddings are likely similar tower outputs): dot(user_emb, item_emb) and history-vs-candidate similarity are first-class features.
2. Strong production practice: GBDT ranker (CatBoost YetiRank) + neural user-item score as feature; official baselines mirror this.
3. Train is post-blender, position-biased: use pos as a nuisance (debias, drop, or use only at train time); val/test are the target distribution, so weight/validate on val and possibly fine-tune on val (14K feeds).
4. Event-age/recency, category-interest distributions from history, and event_source (search vs recs) are known useful features at Avito.
5. Public LB is saturated at ~0.384-0.390 for ranks 2-10; trust val/CV, not public LB, given 30-50 submissions per top user.
