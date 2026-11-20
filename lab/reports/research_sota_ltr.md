# SOTA research: candidate re-ranking with history, embeddings, biased train + unbiased val/test (Avito ML Cup 2026)

Note: web search was only partially informative (2 searches verified: ICLR21 GBDT-vs-neural paper, AutoDebias/KDCRec). The rest is
from prior knowledge of the literature; numbers marked (recall) are from memory and not re-verified. Treat as guidance, validate on val.

## 0. Key insight for this competition
Val/test are uniformly shuffled => labels are position-unbiased; train (247M cands) is production-ranked => label = P(click | item, pos).
Train signal is large but has (i) position bias and (ii) selection bias (production ranker already put likely-clickers on top, and candidates
set itself is retrieval-filtered). Val is small (1.39M cands, 14K feeds) but exactly in-distribution. Strategy: learn on train with
position-debiasing, then use val (and its ordering) for calibration/stacking/fine-tune, with grouped CV over val feeds.

## 1. What to implement first (ranked)
1. LightGBM lambdarank (or XGBoost rank:ndcg) on hand-crafted features: f1..f5, category, user-history stats per candidate category
   (counts/recency/event-type-weighted per category), user-emb . item-emb dot, cosine of item emb to history item embs (max/mean/top-k, by event type),
   candidate-in-feed relative features (rank/z-score of f1..f5 within feed, M, category share in feed). Train on a feed subsample of train;
   EXCLUDE `pos` as a feature (not available at test). Evaluate with the exact full-NDCG metric on val.
2. Position-bias handling at train time: (a) restrict/weight to train feeds or ranks with less bias (e.g. only pos>some deep range, or
   weight by inverse propensity estimated from pos); (b) PAL / shallow-tower: model logit = f(x) + g(pos) trained jointly, drop g at inference
   (in GBDT: use pos as feature then set pos to a constant/random-average at inference, or use init_score from a position-only model = additive-offset);
   (c) check whether train pos is predictive: if train cands order in feed correlates with f1 etc. then f1 itself is partly a production score.
3. Target-attention neural model (DIN/TransAct-style) over history, with candidate (item emb, category, f1..f5) as query; listwise softmax loss
   within feed. Train on train, fine-tune on val folds. Blend with GBDT.
4. Set-aware re-ranker: small transformer over the candidate list (PRM-style) on top of per-candidate score+features; M up to 299 => cheap.
5. Rank-average (per-feed z-score or rank) ensemble of GBDT + NN; stack on val with grouped CV.

## 2. (a) Listwise / set-aware re-rankers
- PRM (Pi et al., RecSys'19, Alibaba): transformer encoder over initial-ranking list + personalized vector, softmax cross-entropy per list. Gains in
  paper: few % AUC/MAP over DNN base, online CTR +~5% (recall). Important: PRM needs input *order* as position embedding; here order is permuted in
  val/test, so drop position embeddings (use permutation-invariant set transformer / self-attention without positional encoding). Train order
  carries production-rank info => would leak a bias not present at test. Shuffle candidates in train and do NOT feed pos.
- DLCM (Ai et al., SIGIR'18): GRU over list; order-dependent -> less suitable. SetRank (Pang et al., SIGIR'20): permutation-invariant self-attention
  (induced set attention blocks) for LTR; best fit to shuffled lists; gains over MLP baselines on MSLR/Istella were modest (~1-3% NDCG, recall).
- PEAR (Li et al., 2022): re-ranking with both feature-level and item-level contexts, pre-/post-trained with list context; PRS/generator-evaluator
  (Huzhang et al.) are about permutation generation - irrelevant (we only score).
- Context features in GBDT (feed-relative normalization, group stats, category share) capture most of the "set" effect cheaply; Pasumarthi et al.
  (Context-aware LTR / "Self-Attentive Document Interaction Networks", ICTIR'19; Qin et al. ICLR'21 DASALC) show self-attention over list helps
  neural models, especially with log1p/normalized features.
- Pitfall: with mean 1.5 positives per feed and 78-96 candidates, list context helps mostly via normalization (feed-relative features) and
  category competition; expect +0.002-0.01 NDCG, not large.

## 3. (b) User-sequence models
- DIN (Zhou 2018): target attention, softmax-free weighting by (hist item, cand item, outer product). Core: attend over history using candidate as query.
  DIEN adds GRU interest evolution (heavy, small gain over DIN). BST (Alibaba 2019): transformer over history+target, positions. SIM/ETA/SDIM:
  retrieve relevant subset from long histories (here only 256, so full attention is fine). TWIN (Kuaishou 2023): two-stage target attention
  consistent between search and attention. TransAct (Pinterest 2023): transformer over recent actions + early fusion with candidate, combined with
  long-term user embedding (PinnerFormer) and DCNv2; reported online gains in engagement (+~5-10% depending on surface; recall). HSTU (Meta, ICML'24, "Actions Speak Louder than Words"): generative
  recommender, pointwise-normalized attention (SiLU not softmax), relative bias, scales to 1.5T params; big gains in recall/NDCG on
  MovieLens/Amazon vs SASRec; (recall) +12% over baselines on benchmarks. ARGUS (Yandex 2025): autoregressive pretraining of transformer on
  user actions (next item + feedback prediction) then fine-tuning for ranker; scaling up to ~1B params gave +~2-3% in online metrics (recall).
- Practical: here we have item embeddings (128-d int8) for candidates and for history items, event_type (10) and source (27) codes, category.
  Build: history token = Linear(item_emb) + emb(event_type) + emb(source) + emb(category) + time/recency embedding (hpos); candidate = Linear(item_emb)
  + category emb + f1..f5 MLP. Cross attention cand -> history (multi-head, 2 layers), concat with user_emb, MLP -> logit. Add DIN-style
  explicit interaction features (cand*hist, cand-hist) in the attention score. Event-type-aware attention is key (event_type 1 dominates).
- Pre-training on train history (584M events) with next-item contrastive loss (ARGUS/HSTU style) can yield a strong user encoder, but costly; first-pass
  cheap alternative: hand-made similarity features (max/mean cosine to history by event type) into GBDT.
- Candidates are never in own history (filtered), so category-/embedding-similarity, not exact repeat, is the signal; also check cross-split
  item overlap (same item_id appearing as history in other users' feeds -> item popularity/CTR target-encoding features from train; out-of-time).

## 4. (c) Losses for NDCG, GBDT vs NN
- LambdaRank/LambdaMART (Burges 2010): pairwise logistic weighted by |dNDCG|. LambdaLoss (Wang et al. CIKM'18): probabilistic framework; shows
  LambdaRank loss is a special case, proposes bound tighter than RankNet; NDCG-Loss2++ best on benchmarks (small gains ~0.5-1%).
- Softmax cross-entropy / ListNet / ListMLE: Bruch et al. "An Analysis of the Softmax Cross Entropy Loss for LTR" (ICTIR'19) proves softmax CE is a
  bound on NDCG-like metrics, and works as well as LambdaLoss in practice; stable, easy. With binary labels and ~1 positive/feed, softmax CE
  over the full feed (sampled softmax if M large) ~= directly maximizing log prob of positive; recommended default NN loss.
  Multiple positives: softmax with target distribution y/sum(y) (ListNet) or sum over positives.
- ApproxNDCG (Qin 2010), NeuralNDCG (Pobrotyn & Bialobrzeski 2021, NeuralSort relaxation, temperature tau ~1), PiRank (Swezey NeurIPS'21,
  differentiable sorting, scales via divide-and-conquer): reported to match/slightly beat LambdaLoss on MSLR/Istella; in practice sensitive to
  temperature/tuning. Use as a fine-tune stage not primary.
- Metric note: full-list NDCG (no cutoff) with binary gain weights the *whole* rank distribution of the positive: discount 1/log2(r+1) is steep at top
  but a positive moving from rank 60 to 30 still gains. lambdarank truncation_level must be large (LightGBM default 30 -> set to >=300 or
  lambdarank_truncation_level=M max) and label_gain default fine for binary; consider `lambdarank_norm=true` (default in recent LGBM).
  XGBoost rank:ndcg with `lambdarank_pair_method=topk` and lambdarank_num_pair_per_sample set high / `mean` to approximate full-list.
- Qin et al. ICLR'21 "Are Neural Rankers still Outperformed by GBDT?" (verified): on Web30K/Yahoo/Istella, LightGBM LambdaMART beats most NN LTR
  models with raw features; their DASALC (data augmentation, self-attention, log1p input transform, listwise loss with ensembles) reaches parity
  with GBDT. Lessons: transform features (log1p / normalized) before NN, use list-context self-attention, ensemble several NN seeds, data
  augmentation (feature-noise). Later (ShareChat 2023, arXiv 2312.01760): NN slightly > GBDT in large-scale recsys with big data, GBDT cheaper.
  Takeaway: GBDT strong baseline for dense features; NN wins when exploiting embeddings/sequences (our case) -> blend.

## 5. (d) Position bias / unbiased LTR / uniform data
- PAL (Guo et al., RecSys'19, Huawei): P(click) = P(seen|pos) * P(click|seen,x); two towers, product of probabilities, remove position tower at inference.
  Shallow-tower (Zhao et al., RecSys'19 "Recommending What Video to Watch Next", YouTube): additive position logit from small tower trained jointly
  with main tower; use dropout on pos feature (10%) to avoid over-reliance; at inference set pos to missing/default. Easy to implement in NN, and
  in GBDT via boost_from_average/init_score = logit offset from a position-only model (fit g(pos), pass as init_score in LightGBM, predict without).
- IPS / Unbiased lambdaMART (Joachims 2017; Hu et al. WWW'19, "Unbiased LambdaMART", Ai et al. "ULTR via dual learning algorithm" SIGIR'18 with
  propensity estimated jointly (regression-EM, Wang et al. 2018)). Propensity estimation requires intervention (randomization) - we HAVE it:
  val feeds are fully randomized => estimate theta(pos) not from val directly (val has no pos column!), but train vs. val comparisons can calibrate:
  e.g. fit P(click|pos) on train, compare to val's rank-independent base rate. Issue: val has no `pos` -> cannot directly estimate propensity from
  val; instead use two-tower EM on train (identifiable if same item appears at different positions across feeds) or position-gap structure.
  Clip IPS weights (max ~10-20) to control variance (positives at deep rank get huge weight).
- Selection (not only position) bias: production candidates ranked by a model correlated with f1; train labels on top positions inflated.
  Use uniform data to evaluate/choose among debiasing strategies: the metric on val is an unbiased estimate -> val is the arbiter.
- Uniform-data methods: KDCRec (Chen et al.; verified) distills from a teacher trained on uniform data into biased-data student (label, feature,
  sample, model-parameter distillation); AutoDebias (Chen et al. SIGIR'21; verified): meta-learn debiasing weights/imputed labels with uniform data as
  meta-set, works with ~1% uniform data. CausE (Bonner & Vasile RecSys'18): "causal embeddings" - separate control/treatment item embeddings
  with regularization tying them. Practical translation here: (1) pretrain on train, fine-tune on val (5-fold by feed; 14K feeds is small ->
  freeze most weights / low LR / few epochs); (2) stack: train-model score + features into a small GBDT trained on val with grouped CV;
  (3) domain-adaptation: add `is_uniform` indicator, train on train+val with val up-weighted (e.g. x20-50), set indicator=1 at inference;
  (4) distillation: teacher on val gives soft labels for train candidates (pseudo-labels) -> student on 247M. Expect (1)-(3) to be most reliable.
- Also: test is a later period than val (Sep 26-29 vs 22-24; train ends 19) -> drift; training on val for the final model is OK but keep
  holdout of last val days to pick epochs.

## 6. (e) GBDT best practices
- LightGBM lambdarank: objective=lambdarank, metric=ndcg, eval_at=[100,300] (no full-list metric natively -> use custom eval w/ exact metric), 
  lambdarank_truncation_level 300+ (important), learning_rate 0.03-0.05, num_leaves 63-255, min_data_in_leaf 50-200 (relative to huge train),
  feature_fraction 0.7, bagging 0.8, lambda_l2 1-10, max_bin 255, label_gain [0,1]. Sort data by group; groups array. Subsample
  feeds (not rows) of train; 247M rows too big for fully LGBM - 20-50M rows (250-600K feeds) likely enough, check learning curve.
  Early stopping on val full NDCG. Also try objective=rank_xendcg (softmax-like, often as good and faster) and binary logloss on debiased target.
- XGBoost: rank:ndcg / rank:pairwise, tree_method=hist (or gpu_hist on GPU), lambdarank_pair_method=mean, lambdarank_num_pair_per_sample ~8-32,
  max_depth 6-8, eta 0.05, min_child_weight, ndcg_exp_gain=false for binary. Pairwise with `lambdarank_unbiased=true` implements
  unbiased LambdaMART (needs position column - available in train as `pos`!) with lambdarank_bias_norm; worth testing: train with
  pos supplied for debiasing, no pos at test.
- CatBoost: YetiRank (default loss for ranking, permutation-based, strong out-of-the-box NDCG), YetiRankPairwise (slower, often better),
  QuerySoftMax (listwise softmax, good with sparse positives), PairLogitPairwise; need group_id sorted; GPU supported (task_type=GPU, device 4).
  Params: depth 6-8, lr 0.05, l2_leaf_reg, iterations 2-5K, `eval_metric=NDCG:top=300;type=Base`. Known: CatBoost YetiRank is often the best single GBDT
  on small/mid ranking sets (Yandex). Categorical (category_id, event types) handled natively with target stats - leakage-free ordered boosting.
- Ensembling: per-feed rank average or per-feed z-score of each model's scores, weights tuned on val (grouped CV); blends of diverse models
  (LGBM lambdarank + CatBoost YetiRank + NN softmax) typically +0.5-1.5% NDCG over best single (recall, Kaggle/RecSys challenge experience).
  Stacking: second-level GBDT with first-level scores + feed-relative ranks of them. Caution about pessimistic ties: break ties with tiny noise
  or avoid coarse-grained scores (constant scores in trees get exactly equal -> pessimistic tie penalty!). Add small epsilon of a second model's score.

## 7. Pitfalls
- Leak via `pos` or order in train: never use; order in train is production order. Shuffle train candidates for set-models.
- Position-biased train => models learn "what production ranked high" (f1-like features already correlate); verify by comparing val NDCG of
  models trained on top-pos vs deep-pos subsets.
- Embedding idx is per-split: map via split's memmap; do not reuse idx across splits; item_id is shared (use for target encoding).
- Time drift: target-encoding/popularity must be computed out-of-time (e.g. from train only, with val using train stats and train using earlier-days stats).
- Val is tiny: SE ~0.002-0.003; use paired per-feed bootstrap; avoid overfitting hyperparameters on val; Public LB covers only 30% of test.
- Equal weight per feed: feeds with small M and 1 positive have high NDCG variance; loss weighting per feed (1/mean) may align better with metric
  than row-level logloss. Include feed size M as feature (random-baseline NDCG depends on M).
- Ties: pessimistic -> avoid constant outputs.

## 8. Expected gains (rough, from literature and priors; validate)
- GBDT lambdarank with history-similarity features vs best single feature f1 (0.337): large (likely +0.02-0.04 toward LB ~0.38+).
- Position debiasing (PAL/shallow tower/IPS-clipped): +0.001-0.01.
- Target-attention NN blended with GBDT: +0.003-0.01.
- Listwise set-transformer on top: +0.001-0.005.
- Fine-tune/stack on val uniform data: +0.002-0.01 (largest when train bias is strong).
- Ensembling/rank averaging: +0.001-0.003.

## Sources
- https://research.google/pubs/are-neural-rankers-still-outperformed-by-gradient-boosted-decision-trees/ (verified)
- https://arxiv.org/html/2312.01760v1 (GBDT vs neural rankers, ShareChat; verified via search)
- https://arxiv.org/abs/2105.04170 (AutoDebias; verified)
- https://aminer.org/pub/64a9dd1ad68f896efac73859/kdcrec-knowledge-distillation-for-counterfactual-recommendation-via-uniform-data (KDCRec; verified)
- From memory (not re-fetched): PRM arXiv 1904.06813; SetRank arXiv 1912.05891; DLCM arXiv 1804.05936; PEAR arXiv 2206.12025; DIN 1706.06978; DIEN 1809.03672;
  BST 1905.06874; SIM 2006.05639; TWIN 2302.02352; TransAct arXiv 2306.00248; HSTU arXiv 2402.17152; ARGUS (Yandex) arXiv 2507.15994;
  LambdaLoss (Wang et al. CIKM18); Bruch et al. softmax CE ICTIR19 arXiv 1911.09798; NeuralNDCG arXiv 2102.07831; PiRank arXiv 2012.06731;
  PAL RecSys19 (Guo et al.); YouTube shallow tower RecSys19 (Zhao et al.); Unbiased LambdaMART arXiv 1809.05818; Dual learning ULTR arXiv 1804.05938;
  CausE arXiv 1706.07639; CatBoost ranking docs https://catboost.ai/docs/en/concepts/loss-functions-ranking; LightGBM params https://lightgbm.readthedocs.io/en/latest/Parameters.html;
  XGBoost LTR tutorial https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
