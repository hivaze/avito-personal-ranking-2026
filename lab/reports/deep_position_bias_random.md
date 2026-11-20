# Deep dive: training on position-biased logs, evaluating on randomized order

Scope: Expedia ICDM 2013 (the closest Kaggle analogue), KDD Cup 2020 Debiasing, industry position-bias papers (Airbnb, YouTube, Google, LinkedIn, Uber Eats, Meituan, Thumbtack, Tripadvisor, Booking.com co-authored Baidu-ULTR study), propensity estimation (RandTopN/RandPair, Regression-EM, intervention harvesting, Unbiased LambdaMART), and "small random + large biased" methods (KuaiRand, AutoDebias, KDCRec, Open Bandit). Each section ends with what it means for Avito ML Cup 2026.

Tags: [V] = read in a primary source this session (URL given). [V-local] = read from this project's files this session. [S] = seen only in a search snippet. [K] = background knowledge, not re-verified.

---

## 0. TL;DR: what to do, ranked by expected value

1. **Find the random-order feeds inside train and treat them as gold data** (the Expedia `random_bool` analogue). Our EDA mixture model estimates that ~12-13% of train feeds with M>=50 look random (`pi_2d`), and puts the expected count at ~346K feeds and ~25M candidates with posterior > 0.5 [V-local: `eda/labels_position/out/04g_full_posterior.log`]. If even half of that is real, it is roughly 10-20x the unbiased data in val. Three uses: (a) unbiased training rows (weight = posterior); (b) **RandTopN propensities** θ(k) ∝ CTR at slot k within those feeds (Wang et al. 2018). This works because train keeps the `pos` column, which val lacks; (c) a validation set for propensity estimates and debiasing choices. Check the detection first (section 6.1). Expedia's host noted that "some impressions were incorrectly labelled as random", so assume label noise even in official flags [V].
2. **Fix the expected-clicks denominator in the COEC / target-encoding features.** Today `tr_exp` and `tte_*_coec` use a global `pc[pos]` [V-local: `src/features.py`, `src/features_tte.py`]. Two problems. First, it ignores feed length M: the base rate runs from 0.168 for M<10 down to 0.011 for M≥250 [V-local `01_labels.log`]. Second, it ignores the end effect: P(pos) = 0.18 at the last slot and 0.175 at the third-from-last [V-local `02b_end_effect.json`]. Third, a CTR curve from non-random data overstates position bias because it mixes in ranker quality. Thumbtack found exactly this, and switching to COEC features built from randomized propensities gave **+1.6% contacts** in an A/B test [V]. Recompute expected clicks as E[c | slot k, M, rank-from-end] from a position-only model fitted on random feeds (or from item-agnostic logits of the bias tower).
3. **Native position debiasing in GBDT instead of a fixed offset.** LightGBM ≥4.1 accepts `position=` in `Dataset` and learns additive per-position biases b(pos) jointly with the trees. It adds b(pos) to the score before computing lambdas and takes a Newton step per iteration, with `lambdarank_position_bias_regularization` as an L2 penalty [V, source code]. Encode `position_id` as a categorical cross of (slot bucket × rank-from-end bucket × M bucket), because ids are arbitrary categorical codes [V]. XGBoost ≥2.0 has `lambdarank_unbiased=True` (Unbiased LambdaMART, learns t+ and t− per position). It infers position from **row order inside the group**, so rows must be sorted by `pos` [V].
4. **Mixed training with an `is_random` indicator**, the same trick as the Expedia 5th-place team's `random_bool` GBM feature [V]. Train on train production feeds (pos known) plus val (and detected-random train feeds) with `is_random=1` and `pos=NaN`/"unknown" for the random rows. Predict test with `is_random=1`. In an NN, give the bias tower a "random-order" token per M bucket for those rows. The random rows anchor the relevance tower where position cannot act as a shortcut. This is the direct remedy for the confounding problem described in item 5.
5. **Expect position confounding, and regularize the bias tower.** Google (KDD'23) shows that when logging positions come from a good ranker, a two-tower PAL model can be **much worse than a naive model**. On Web30k with oracle logging: naive 0.4191, PAL 0.3333, gradient-reversal 0.4159, observation-dropout 0.4118 (NDCG@5) [V]. Our production pos is clearly confounded: on train, ranking by pos-ascending gives NDCG 0.3615 vs 0.3240 for random order (pos-descending gives 0.4056 because of the end spike) [V-local `02_position.log`]. In the NN, apply dropout to the bias-tower **output** (τ 0.2-0.5, higher the more confounded) and/or gradient reversal (η 0.6-0.8) [V]. Our current NN uses a plain additive g(pos)+g'(M−1−pos) PAL tower [V-local `nn/model.py`].
6. **Keep the loss choice at the centre.** On Baidu-ULTR (expert labels), the **choice of loss** (LambdaRank > listwise softmax > pointwise BCE) mattered far more than any ULTR correction. Pointwise IPS was significantly *worse* than naive. Only Regression-EM and listwise IPS/DLA gave small gains [V]. Debiasing gains in industry are typically +0.2% to +1.8% online and ~1% offline [V: Airbnb, YouTube, Google, LinkedIn]. Do not expect a debiasing trick to close the 0.365 → 0.39 gap alone. The value of train is mainly in **features learned from it** (COEC/TE, distillation of the production rank, a stacked train-model score) and in the **random subset**.

---

## 1. Framing for our data (why ranking by relevance is exactly right, and what changes the math)

- Position-based model (PBM): P(c=1 | x, slot k) = θ_k · γ(x) [V: Wang 2018; Joachims 2017; Airbnb eq. 2].
- In val/test the display slot is uniform and independent of x, so P(c | x, feed with n items) = γ(x) · θ̄(n), where θ̄(n) = mean over k≤n of θ_k. Within a feed θ̄(n) is constant, so **ranking by γ(x) is optimal for our metric**. Feed-level factors (n, user activity) can be ignored for ranking but matter for any pooled pointwise loss.
- With a 1.6% positive rate, logit(p) ≈ log(p), so an **additive logit** model f(x) + g(pos) (YouTube shallow tower, LightGBM GAM, PAL in log-space) is a good approximation of the multiplicative PBM [K, math].
- The labels are conditioned on ≥1 positive per feed (the builder keeps only feeds with positives [V-local `research_competition_avito.md`]). Given exactly one click, P(item i is the click) ∝ θ_k γ_i /(1−θ_k γ_i) ≈ θ_k γ_i, which is a softmax over the feed. A listwise softmax / LambdaRank loss fits this better than pooled BCE [K, math; consistent with Baidu-ULTR loss ranking [V]].
- **Our train is not a pure PBM.** The positive rate by rank-from-end is 0.1795 for the last item, 0.0447 for −1, 0.1752 for −2, and ≈0.027 after that. From the start it is 0.112 for the first item, 0.071 for the second, 0.027 at 20 and 0.009 at 200 [V-local `02b_end_effect.json`]. pos also has gaps of ~12-15% at the top slots [V-local `04e_gaps.log`]. So the bias side must model **slot, rank-from-end and M jointly** (categorical, fine-grained near both ends). The end spike looks like a session-stop or logging-truncation effect, not an examination effect. A cascade-like process implies that items logged after the last click are weak negatives (see the Tripadvisor recipe in section 3.8).
- **Covariate shift caused by order** [K, reasoning; to verify]. If feeds are cut where the user stops scrolling, a production feed of length M holds the production top-M, while a random feed of length M holds a random subset. Within-feed rank/z-score features then have different distributions in train vs val. Compare per-M feature spreads in train-production, train-detected-random and val. This is also a test of the random-feed detector.
- **Pessimistic ties.** In Expedia, a simple prop_id-mean model dropped from 0.51 (train) to 0.43 (test). A forum participant explained that tied scores were broken by file order, which in train was the production order [V: Kaggle Expedia thread 6228]. Our metric places negatives first among ties, so every submission needs a deterministic tie-breaker. Our EDA shows that f1 alone loses 0.009 between expected-tie and pessimistic-tie NDCG [V-local `07_metric.log`: f1 pess 0.3279 vs expected 0.3365].

---

## 2. Expedia Personalized Hotel Search (Kaggle / ICDM 2013): the closest Kaggle analogue

Facts [V, host posts in Kaggle threads 5772, 5808, 5767, 19664, 6203, 6228; https://www.kaggle.com/competitions/expedia-personalized-sort/discussion/5772 etc.]:
- `random_bool=1` meant hotels were sorted randomly ("We show randomly sorted hotels for a tiny fraction of traffic just to get less biased data so that it can be used to train our ranking algorithm"). Non-random impressions were "around 70% of all impressions in the training and testing data" (host). About 30% of sessions were randomized [V: Greminger, arXiv 2210.16408].
- **`position` was removed from test, and test row order was randomized.** The "Position benchmark" (the real Expedia display order, which participants could not see) scored **0.49990** NDCG@38. Winners were ≈0.53-0.54 (5th place 0.53102 private [V: arXiv 1311.7679]).
- Host: "some impressions were incorrectly labelled as random, so you probably need to remove these outliers" [V, thread 19664].
- Ranking: 1st Owen Zhang, 2nd Jun Wang, 3rd Jeremy W. The commendo team (Michael Jahrer) had the highest score but did not meet the contest requirements (host) [V, thread 6203; S for 2nd/3rd names].

How top teams treated position / random_bool:
- **Jahrer (commendo, top score)** [V, thread 6228]: best single model was RankLib LambdaMART at 0.5338. Features were all numeric features plus their **mean/std/median per prop_id computed on train+test**, explicitly *excluding* `click_bool, booking_bool, position, gross_bookings_usd`. The prop_id means took the score from 0.51 to 0.53, "the biggest improvement". In other words, the winning recipe kept position out of the features and used transductive item-level aggregates of non-target features.
- **Dmitry Efimov** (R gbm, pairwise) [V, thread 6228]: target "likelihoods" per prop_id shrunk to the global mean "improved a lot". "In one of our model we used **modified position as a feature with predicted values for this feature on the test set**". This is distillation of the production rank: learn position from features on train, then predict it for test.
- **5th place (Liu et al., arXiv 1311.7679)** [V]: GBM top-20 relevance list includes `random_bool` (1.00) and its count/frequency encodings ("random bool 1 cnt" 2.26, "random bool f" 1.57). Also listwise rank features (rank of price_usd, starrating, location_score2 within the search), one-way count features over train+test, and 10% sampling by srch_id ("very small difference" vs full data). Ensemble of LR/FTRL, RF per country, GBM, ERT, FM, LambdaMART, DNN.
- **Batarat (low score but instructive)** [V, thread 6228]: trained separate models for random-order and model-ordered searches.
- **Owen Zhang (1st)**: his ICDM'13 slides are in a Kaggle-hosted zip that is no longer publicly downloadable (GCS returned 403 this session). The forum confirms "26 R GBM models" [V, thread 6203 question] and Owen's statement that "a single GBM built with 10% data can get into top 10 (0.525xx)" [V, thread 5960]. From memory, he also normalized features by srch_id/prop_id and used "estimated position" features: the average position of the hotel in other (non-random) searches [K, unverified, low confidence].

Lessons for us:
- The Expedia test set mixed random and non-random impressions, so `random_bool` at prediction was 0 or 1. Our test is 100% random, so the transferred trick is: train with an indicator, predict with `is_random=1` (TL;DR #4).
- Item-level aggregates of features and of the production position from the big biased log beat position modelling. Our analogues: per-item mean normalized production rank (pos/M) across train feeds, per-item COEC with correct expected clicks, and a model-predicted production rank for val/test candidates (Efimov). Item IDs are shared across splits. Our EDA found item-TE coverage of 31% in one setup [V-local `04b_detect.log`]. Compute these out-of-time/out-of-fold for train rows.

---

## 3. Industry recipes (verified numbers)

### 3.1 Airbnb, "Improving Deep Learning for Airbnb Search" (KDD'20) [V: https://arxiv.org/abs/2002.05515, Sec. 4]
- Model: dnn(q,u,l,k) = rel(q,u,l,k) · pbias(q,u,k). **Position k is an input feature regularized by dropout (rate 0.15)**, and k is set to 0 at scoring time, so pbias becomes a per-search constant β.
- Without dropout, the position feature cost **−1.3% NDCG** because the network relied on position to predict relevance.
- Dropout tuning: sweep rates and plot NDCG_rel (pos=0 at test) against NDCG_rel+pbias − NDCG_rel. Pick a point where the position contribution is large without a big loss in NDCG_rel. 0.15 was chosen.
- Online: **+0.7% bookings, +1.8% revenue**, +1.1% boutique-hotel bookings. Non-intrusive, with no randomization (positions vary naturally through availability).

### 3.2 YouTube, "Recommending What Video to Watch Next" (RecSys'19) [V: https://daiwk.github.io/assets/youtube-multitask.pdf, Sec. 4.4, 5.3]
- A shallow tower on bias features (position crossed with device) is **added to the final logit**. All impressions are used in training, with **10% feature dropout** on position. At serving, "position feature is treated as missing".
- Live engagement vs baseline: position as input feature −0.07%, adversarial loss +0.01%, **shallow tower +0.24%**.

### 3.3 Google, "Towards Disentangling Relevance and Bias in ULTR" (KDD'23) [V: https://arxiv.org/abs/2212.13937]
- Problem: logged positions come from a ranker that "possess[es] relevance information", so the bias tower absorbs relevance (shortcut learning).
- Fixes: (1) **observation dropout** on the bias-tower *output*: ψ = sigmoid(f + Dropout(g, τ)). (2) **Gradient reversal**: an extra head on the observation tower predicts relevance through a GradRev layer with scale η; adversarial labels can be clicks or relevance-tower predictions, with "comparable" results.
- Numbers (NDCG@5). Web30k, oracle logging: Biased 0.4191 / PAL 0.3333 / DLA 0.3144 / GradRev 0.4159 / Drop 0.4118. Web30k, random logging: PAL 0.4103 vs Biased 0.3860. Yahoo: gains up to 4.7%. Chrome Web Store logs: ~1.3% over PAL. Online over PAL: Biased +1.21% clicks but −5.78% users with clicks; **GradRev +2.90% clicks, +0.96% users with clicks**.
- Tuning: Drop is best at τ 0.2-0.5 and needs a larger τ the more confounded the data. GradRev is best at η 0.6-0.8. Drop+GradRev adds about +0.5%.
- Takeaway: with a strong logging policy, a two-tower model without these fixes can be **worse than ignoring position**. This must be validated on val.

### 3.4 LinkedIn feed, "Isotonic Layer" (KDD'26) [V: https://arxiv.org/abs/2603.06589, Sec. 5]
- Position dropout (sentinel −1, tuned over {0.05,…,0.5}, best 0.15) gave the best offline AUC but **no statistically significant online lift in any configuration**. The authors attribute this to the train/serve mismatch (random masking in training, fixed sentinel at serving).
- Replacement: a dual head on a shared representation. One inference head without position; one calibration head conditioned on position and device through isotonic, monotone bucket weights. Loss α·BCE(inf) + β·BCE(iso) with β ≈ 0.7-0.8. Online +0.17% / +0.20% / +0.36% on engagement categories, with another configuration reaching +0.63% subscription WAU. Platt scaling and per-task isotonic regression were flat or negative online.
- Earlier LinkedIn work used top-K randomization and fair pairs for data collection, plus replay evaluation on random buckets [V: https://arxiv.org/abs/1905.06134], and historical A/B tests as instrumental variables for position effects [V abstract: https://arxiv.org/abs/2205.06363]. IV needs known ranker-assignment buckets, which we do not have.

### 3.5 Uber Eats home feed [V: https://www.uber.com/blog/improving-uber-eats-home-feed-recommendations]
- Store order was randomly permuted for a small share of traffic, and conversion rate vs position on that traffic measures the bias. The earlier attempts were IPW, training only on randomized data, and position as a feature.
- Current model: a true-CVR tower plus a position-bias side tower, logits summed before the sigmoid, with **L1 + dropout on the bias tower** so it does not learn true CVR.
- Offline acceptance test: **on randomized traffic, the debiased model's predictions are uncorrelated with vertical position** (the production model's were correlated). Online orders were significantly up (no effect sizes given).

### 3.6 Meituan DPIN (SIGIR'21) [V: https://arxiv.org/abs/2106.05482, Table 1]
- Test sets come from regular traffic and from **top-k randomized traffic** (5% of traffic). On the randomized set: DIN AUC 0.7836 / PAUC 0.7223; DIN+PosInWide 0.7725 / 0.7239; DIN+PAL 0.7763 / 0.7254; DPIN 0.8015 / 0.7350.
- Position-in-wide and PAL lowered AUC and raised position-wise AUC only slightly (+0.2-0.3 pt). The bigger gain came from position-wise interaction modelling. Note: DPIN uses position at serving (ad slot allocation), which we cannot do.

### 3.7 Thumbtack (marketplace search) [V: https://arxiv.org/abs/2206.11720, Sec. 5-6]
- The RandPair randomization program cost little. The bias measured from it was "much less substantial than the estimates we used from earlier non-randomized data". Non-random CTR curves **overstate** position bias.
- **COEC historical-rate features**, α_u = Σ C_i / Σ θ_{k_i} with θ from randomization, replaced the biased rate features and gave **+1.6% visitors with ≥1 contact, +1.7% with ≥1 match** (p=0.01).
- Then an IPS-weighted ranker (contacts divided by θ_k inside a DCG-style loss) gave **+0.7% contacts (p=0.08), +0.9% matches (p=0.05)** over the COEC model. It survived later challenges from LambdaMART and a trust-bias model.

### 3.8 Tripadvisor hotels [V: https://arxiv.org/abs/2002.12528]
- Propensity without randomization: **P(E=1|k) = E[P(C=1|k)] / E[P(R=1|hotel,k)]**, with historical bookings as the relevance proxy. This is the "relevance-model ratio" estimator. We can use a **val-trained** relevance model as the proxy (section 4.5).
- Training-set construction: keep everything at or above the last clicked position (assumed examined). Sample items below the last click with probability θ_k / θ_lastclick. Online: **+1.5% clicks**. A constant 80% sampling gave −0.2%.

### 3.9 Baidu-ULTR reality check (SIGIR'24, Booking.com co-author) [V: https://arxiv.org/abs/2404.02543, Table 3]
- Large real click log with expert-labelled test. "Standard ULTR techniques robustly improve click predictions but struggle to consistently improve ranking performance".
- Loss: LambdaRank naive > listwise naive > pointwise naive, by large margins (e.g. DCG@10 8.233 vs 7.844 vs 7.546 on their BERT features). Pointwise IPS was significantly worse than naive on BERT features. Pointwise Regression-EM gave small significant gains. Listwise IPS/DLA gave marginal gains. PairD was not better than naive LambdaRank.
- Lesson: better click NLL (two-tower) does not mean better ranking. Validate every debiasing step on the unbiased val NDCG, never on train click loss.

### 3.10 KDD Cup 2020 Debiasing (Tianchi): low relevance
- [V: https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place] The README frames it as "fairness of exposure", with a train/test gap "mostly with respect to the trends and the items' popularity". This is **popularity/exposure debiasing, not position**. Recipes from secondary sources (popularity-weighted loss, boosting rare items) [S] do not transfer to a per-feed NDCG on random order.

---

## 4. Propensity estimation methods (and which fit our data)

### 4.1 RandTopN (full shuffle of top N) [V: Wang et al. WSDM'18, https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46485.pdf §4.2.1]
E(C|k) = θ_k · E[γ], so **θ_k ∝ CTR at slot k in randomized data**. This needs positions in the random data. Val/test do not have them, but **detected-random train feeds do**. Estimate θ per (slot, rank-from-end, M bucket) from high-posterior random feeds. Check that the curve does not move as the posterior threshold rises from 0.5 to 0.9 (contamination by production feeds would make the top steeper).

### 4.2 RandPair / Swap(1,k) [V: Joachims et al. WSDM'17 §5, https://arxiv.org/abs/1608.04468; Wang 2018 §4.2.2]
Swap adjacent pairs (k−1,k) and chain the ratios. Lower cost to the user than RandTopN, and it agrees with RandTopN [V, Wang 2018 Fig. 1]. This needs interventions we do not control.

### 4.3 Regression-EM (no randomization) [V: Wang 2018 §4.3, Alg. 1]
- E-step: P(E,R | c,k) under the PBM. M-step for θ_k is closed form. For γ, **sample a binary relevance label r ~ P(R=1|c,x,k) and fit a GBDT classifier f(x)** on (x, r), instead of using per-(q,d) ids. This works "as long as similar feature vectors appear in different positions".
- Baidu-ULTR: the only pointwise ULTR method with consistent small gains [V].
- Caveat (Google KDD'23): the same confounding affects EM's propensities [V, Sec. 6].
- For us: fit on production train feeds with f = our feature GBDT. Compare the θ it produces with RandTopN on random feeds; the two should agree. If they do not, trust RandTopN.

### 4.4 Intervention harvesting (AllPairs extremum estimator) [V: Agarwal et al. WSDM'19, https://arxiv.org/abs/1812.05161]
- Uses logs from **multiple historical rankers**. An interventional set is formed when the same (query, doc) appears at position k under ranker f and at k' under f'. This requires Pr(f | q) = Pr(f) (randomized A/B assignment) [V, eq. 1]. Swap interventions are "statistically rather inefficient".
- For us: weak fit. A feed's "query" (user state) almost never repeats across rankers, and we do not know ranker buckets. Possible only via (user, item) pairs repeated across the 14 days, with confounding from changes in user state. Low priority.

### 4.5 Ratio estimator with a model from random data (our main option for production feeds) [V method: Tripadvisor eq. 4]
- θ(k, M, e) ∝ Σ_{train rows at (k,M,e)} c / Σ γ̂(x), where γ̂ comes from a model trained on **random data** (val, or detected-random train feeds). γ̂ must not use position-correlated inputs.
- Bias direction [K, reasoning]: the production ranker uses signals that γ̂ lacks, so E[γ|top slots] > E[γ̂|top slots]. The estimated θ is therefore **too steep at the top**. Joachims shows that overestimating small propensities is tolerable while underestimating them hurts. Clipping is one way of overestimating small propensities [V, §7]. So clip and flatten, do not sharpen.

### 4.6 Identification from val alone [K, derivation; low priority]
With random display, E[#clicks in feed | n, items] = (Σ_i γ_i / n) · S(n), where S(n) = Σ_{k≤n} θ_k. Fit S(n) = E[clicks|n] / E[mean γ̂|n], then θ_n = S(n) − S(n−1). This is biased by the ≥1-positive conditioning of feeds, so use it only as a sanity check on curve shape.

### 4.7 Jointly learned biases inside the ranker
- Unbiased LambdaMART [V: https://arxiv.org/abs/1809.05818]: jointly estimates click-position biases t+ and unclick-position biases t− with Lp regularization (they "are relative values"). It reports large offline gains over Regression-EM+LambdaMART (e.g. +3.2% NDCG@1) and an online A/B test. It is in XGBoost.
- LightGBM's additive version is described in section 5, item G1.

---

## 5. Concrete GBDT recipes for our data

**G1. LightGBM native position debiasing (try first; cheap).**
- `lgb.Dataset(X, y, group=..., position=pos_id)` on production train feeds. Source facts [V: https://github.com/microsoft/LightGBM/blob/master/src/objective/rank_objective.hpp]:
  - score_adjusted = score + pos_biases_[pos] is used for the lambdas.
  - Per-position biases get a Newton step from the summed lambdas and hessians each iteration.
  - L2 penalty `lambdarank_position_bias_regularization × instance_count`.
  - The biases live in the objective, not in the trees, so `predict` returns the unbiased f(x).
- Docs [V: https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html]: a GAM s(x,pos) = f(x) + g(pos). Position values are categorical codes whose specific numbers do not matter.
- Build `pos_id` = cross of slot bucket (0,1,2,…,9,10-14,15-19,20-29,…) × rank-from-end bucket (0,1,2,3,4-9,≥10) × M bucket (5 bins), giving ~500-1000 ids. Keep `lambdarank_truncation_level=1000` (we already do [V-local]; the default is 30 [V]). Sweep the regularization over {0, 1, 10, 100}.
- Compared with our `--pos_mode offset` (fixed init_score = α·log CTR(pos)): G1 re-fits the biases against the current f(x), which is the regression-EM idea inside boosting. The fixed offset uses the confounded raw CTR curve.

**G2. Plain position as a feature, then set it to a constant (avoid as the main method).**
Trees learn pos × feature interactions, so the inference constant is arbitrary. Airbnb measured −1.3% for the undropped equivalent [V], and YouTube got −0.07% online for input-feature position [V]. If used at all, randomly replace pos with "missing" in ~15% of training rows (Airbnb/YouTube dropout analogue) and predict with missing.

**G3. `is_random` mixture (Expedia random_bool transfer).**
- Train on [production train feeds: pos, rank-from-end, M, `is_random=0`] ∪ [val folds + detected-random train feeds: pos=NaN, `is_random=1`, weight = posterior × w]. Predict with `is_random=1`.
- Tune w (random rows are 1-10% of rows; try 5-50×). Combine with G1 by giving the random rows a single dedicated `pos_id` ("random", per M bucket).
- Val CV: hold out val folds as usual, and never let the same user's val feeds sit in both train and the OOF fold.

**G4. Per-query IPS weights for LambdaRank (Joachims/Thumbtack-style IPS in pairwise form).**
- LightGBM multiplies each row's lambda and hessian by that row's weight [V, source]. For the 74% single-positive feeds, every pair involves the positive, so a **constant group weight w = 1/θ̂(k⁺, M, e⁺)** is exactly pairwise IPS. For multi-positive feeds, use the mean of 1/θ̂ over the positives (approximation).
- Clip w at the 95-99th percentile, since underestimated propensities hurt [V, Joachims]. Use θ̂ from section 4.1 or 4.5, not the raw CTR curve.

**G5. Truncation-aware negatives (Tripadvisor recipe adapted).**
- Keep all candidates at or above the last positive (by slot). Down-weight items below the last positive by θ̂(k)/θ̂(k_last), or drop them [V method; +1.5% clicks online at Tripadvisor].
- Given our end spike, also try simply dropping rows after the last positive. Validate on val.

**G6. Features from the biased log (largest proven lever in Expedia/Thumbtack).**
- (a) Item / user / user×category / category COEC: Σc / Σθ̂(k,M,e), using a *random-feed* θ̂ with Bayesian shrinkage, computed time-forward for train rows. The current version needs the fix in TL;DR #2.
- (b) Item mean of normalized production rank (pos/M, and rank-from-end/M) across train feeds: the production ranker's opinion, which uses signals we do not have (Jahrer-style prop_id aggregates [V]).
- (c) A model trained on production train feeds to **predict pos/M** from our features, applied to val/test (Efimov's "predicted position" [V]). Our NN has an `aux_pos` head [V-local]. For GBDT, add the out-of-fold prediction as a feature.
- (d) Out-of-fold score of the best debiased train model as a feature in the val-CV LambdaRank. This is the KDCRec "feature-/label-based distillation" family [V: https://dgliu.github.io/pubs/TKDE_2023_KDCRec.pdf].

**G7. XGBoost Unbiased LambdaMART (if G1 helps and we want a second debiaser).**
`objective=rank:ndcg, lambdarank_unbiased=True, lambdarank_bias_norm∈{1,2}`, with rows **sorted by pos within each qid** [V: XGBoost demo `learning_to_rank.py` uses `sort_ltr_samples` by qid then pos]. For full-list NDCG, use `lambdarank_pair_method="mean"` with several pairs per sample, or `topk` with k ≥ max M [V: param docs]. GPU is supported; the docs say the distributed interface does not support debiasing [V].

---

## 6. Concrete NN recipes

**N1. Bias tower design.**
- Inputs: slot embedding, rank-from-end embedding (fine-grained 0..9), M bucket, and optionally slot×category, since layout effects differ by item type (Uber Eats [V]; YouTube crosses position with device [V]).
- Add the bias logit to the relevance logit (YouTube/LightGBM GAM).
- **Observation dropout on the bias logit (τ 0.2-0.5)** and/or **GradRev (η 0.6-0.8)**, with clicks or the relevance-tower prediction as the adversarial label [V, Google KDD'23]. L1 on the bias tower (Uber Eats [V]).
- Drop the bias tower at inference. Our current tower is g(pos)+g'(M−1−pos) without these regularizers [V-local].

**N2. Random-order anchor.**
- Mix val (in CV) or detected-random train feeds into batches. For those rows the bias tower gets a learned per-M "random" embedding, i.e. a constant within the feed. Because position is independent of x there, the relevance tower cannot use position as a shortcut.
- This is the "uniform data supervises debiasing" idea behind AutoDebias. There, MF trained on uniform data alone was terrible (Yahoo!R3 NDCG@5 0.449 vs 0.550 biased), while AutoDebias reached 0.645 [V: https://arxiv.org/abs/2105.04170 Table 3]. Our uniform set is far larger and richer than Yahoo!R3's 2.7K-rating uniform split, so expect smaller relative gains.

**N3. Loss.**
Feed-level listwise softmax or LambdaRank-style loss over all candidates, not pooled BCE (Baidu-ULTR ordering [V]). For production feeds, compute the softmax on f(x)+g(pos). For random feeds, on f(x) plus the per-feed constant.

**N4. Fine-tune on val after pretraining on train** with the bias tower. Already in `nn/finetune.py` [V-local, by name]. Use a low learning rate and early stopping on val OOF. The KDCRec "bridge/refine" strategies are formal versions of this [V].

**N5. Position dropout on the input feature (Airbnb 0.15).**
Acceptable as a baseline only. LinkedIn found no online gain despite the best offline AUC [V].

---

## 6.1 Validation protocol for anything position-related

1. **Detector validity** (before using random train feeds):
   - Detected-random feeds should match val on M distribution, positives per feed and per-M feature spreads.
   - Their CTR-by-slot curve should be flatter than in production feeds, and stable across posterior thresholds.
   - A model trained only on N detected-random feeds should beat one trained on N production feeds when evaluated on val. This is the decisive check.
2. **Uber Eats check** [V]: on detected-random train feeds, the debiased relevance score must be uncorrelated with slot. Report Spearman(score, pos) per feed, averaged.
3. **Airbnb trade-off curve** [V]: for each dropout τ or regularization λ, plot val NDCG (relevance only) against train-holdout NDCG(with bias) − NDCG(without).
4. **θ̂ triangulation**: RandTopN (random feeds) vs Regression-EM vs the ratio estimator. Large disagreement means the PBM is misspecified (end effect) or the detection is contaminated.
5. All decisions on val with a paired bootstrap over feeds (SE ≈ 0.0014 for f1 vs random [V-local `06_single_feature.log`]). Never decide on train click NLL (Baidu-ULTR [V]).

---

## 7. Small-random + large-biased literature: what transfers

- **KuaiRand** (CIKM'22) [V: https://arxiv.org/abs/2208.08696]: random items were inserted into normal feeds for two weeks (item randomization, not order randomization), giving 1.19M random-exposure interactions. It enables unbiased offline evaluation. Our val/test are order-randomized instead, which is easier: only position bias, no exposure bias, because the candidate pool is still production-retrieved.
- **AutoDebias** (SIGIR'21) [V]: meta-learns per-sample weights and imputation on biased data by minimizing loss on a small uniform set (5% train, 5% val, 90% test of the uniform split). It is robust even with a uniform set of 1% of the data, where KD-Label fails. Heavy (bi-level optimization). Our scale (247M rows) makes it costly. **Low priority.** The cheap analogue is G3/N2 with tuned weights.
- **KDCRec** (SIGIR'20/TKDE'23) [V]: distillation from a uniform-data model through four channels: labels (bridge/refine: impute labels for biased data with the uniform model), features (select stable features on uniform data), samples (mutual learning, batch ratio s:1), and model structure (CausE-like embedding alignment). G6(d) and N4 are the practical forms for us.
- **Open Bandit Dataset** (ZOZOTOWN) [V: https://arxiv.org/abs/2008.07146]: logs from uniform-random and Bernoulli-TS policies with 3 positions and logged action probabilities. Useful as a reference for off-policy evaluation. It assumes reward depends only on item and position, not on other items. Not directly applicable.
- **Google 2016** [V: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45286.pdf]: propensities from randomized data in global, segmented (per query class) and generalized (feature-based) forms, used for IPW with MART. For us, this justifies θ per M bucket and per category or slot type rather than one global curve.

---

## 8. Pitfalls checklist

- Raw CTR-by-position from production logs as propensity → overstated bias (Thumbtack [V]; confounding [V Google]). This applies to our current COEC denominator.
- A two-tower or PAL model without disentangling can be **worse than naive** under a strong logging policy (Web30k oracle: 0.3333 vs 0.4191 [V]).
- Pointwise IPS can hurt (Baidu-ULTR [V]). Clip weights; underestimated propensities are the dangerous direction [V Joachims].
- Position feature set to a constant at inference in trees: arbitrary interactions (Airbnb −1.3% without regularization [V]).
- Official "random" flags can be noisy (Expedia host [V]). Our flags are inferred, so they are noisier still. Use soft posterior weights.
- Ties in the score under the pessimistic-tie metric (Expedia tie-order leak [V]).
- Do not assume a pure PBM: our end-of-feed spikes (−0, −2) require rank-from-end in the bias model [V-local].
- Train-trained within-feed features may shift between production and random feeds, depending on which candidates are included [K, to verify].

---

## Sources read this session
- Airbnb KDD'20: https://arxiv.org/abs/2002.05515 (PDF text, Sec. 4)
- YouTube RecSys'19: https://daiwk.github.io/assets/youtube-multitask.pdf
- Google KDD'23 disentangling: https://arxiv.org/abs/2212.13937
- Baidu-ULTR SIGIR'24: https://arxiv.org/abs/2404.02543
- Wang et al. WSDM'18 Regression-EM: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46485.pdf
- Wang et al. SIGIR'16 selection bias: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45286.pdf
- Agarwal et al. WSDM'19: https://arxiv.org/abs/1812.05161
- Joachims et al. WSDM'17: https://arxiv.org/abs/1608.04468
- Unbiased LambdaMART WWW'19: https://arxiv.org/abs/1809.05818
- LightGBM docs (Advanced Topics, Parameters) and `src/objective/rank_objective.hpp` (master)
- XGBoost LTR tutorial, `doc/parameter.rst`, `demo/guide-python/learning_to_rank.py`, `xgboost/testing/data.py::sort_ltr_samples`
- LinkedIn KDD'26 Isotonic Layer: https://arxiv.org/abs/2603.06589; LinkedIn dream jobs: https://arxiv.org/abs/1905.06134; LinkedIn IV: https://arxiv.org/abs/2205.06363 (abstract)
- Thumbtack: https://arxiv.org/abs/2206.11720
- Tripadvisor: https://arxiv.org/abs/2002.12528
- Meituan DPIN: https://arxiv.org/abs/2106.05482
- Uber Eats blog: https://www.uber.com/blog/improving-uber-eats-home-feed-recommendations
- Alibaba general debiasing framework (abstract only): https://arxiv.org/abs/2112.02767
- KuaiRand: https://arxiv.org/abs/2208.08696; AutoDebias: https://arxiv.org/abs/2105.04170; KDCRec TKDE: https://dgliu.github.io/pubs/TKDE_2023_KDCRec.pdf; Open Bandit: https://arxiv.org/abs/2008.07146
- Expedia: 5th-place paper https://arxiv.org/abs/1311.7679; Kaggle threads 6228 (My Approach), 6203 (Winning algorithm), 5772 (random_bool), 5808/5767 (position benchmark), 19664 (randomizing data), 5960 (Owen's 10% GBM comment), read via Kaggle's discussion API; Greminger trade-offs (random-ranking subset stats): https://arxiv.org/abs/2210.16408
- KDD Cup 2020 Debiasing 1st place README: https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place
- Not retrievable: Owen Zhang's ICDM'13 slides (Kaggle GCS zip returned 403; Internet Archive offline); PAL full text (only via LightGBM docs and Google KDD'23 descriptions).
- Local: `eda/labels_position/out/*.log|json`, `src/features.py`, `src/features_tte.py`, `src/train_lgb.py`, `src/cv_val.py`, `nn/model.py`.
