# Deep dive: neural rankers for feed re-ranking with user history and pretrained embeddings

Date: 2026-11-11. Scope: what actually beat strong GBDT, and which parts of that transfer to Avito ML Cup 2026 "personal ranking"
(feeds of 5-300 candidates, ~1.5 positives per feed, full-list binary NDCG; train = 247M position-biased production candidates;
val/test = uniformly shuffled; 128-d int8 user/item embeddings; history of up to 256 typed events; 5 dense features).

Tags: **[V]** = read in the primary source this session (URL given; for arXiv/PDF I read the full text via pdftotext);
**[S]** = seen only in a search snippet or a tool summary of a page; **[K]** = my background knowledge, not re-checked.

Project status at the time of writing (from `nn/logs`): the in-house feed transformer (`nn/model.py`: history target attention, set attention,
PAL position tower, in-GPU dense features), trained on *train*, already reaches val NDCG **0.368-0.370** after about 10 minutes and half an epoch.
That beats the val-CV LightGBM OOF (0.365). Ablations in progress: no history 0.358, no set attention 0.366 (still training).
Everything below is aimed at that model and at combining it with GBDT.

---------------------------------------------------------------------------------------------------------------------------------------

## 0. Bottom line (ranked actions)

1. **Keep the NN trained on *train* as the main engine. Feed its out-of-sample scores to the val-CV GBDT as features, and also rank-blend them.**
   Industry evidence agrees on this. Yandex ARGUS: the transformer *standalone* is **worse** than the production GBDT (-4.8% to -7.5% pair accuracy),
   but as a **feature** inside that GBDT it gives **+1.35% to +2.66%** [V]. The Yandex Ads long-history transformer is also evaluated and deployed as a CatBoost feature [V].
   Avito's own user-history transformer is the top-1 SHAP feature of their ranker (secondary, from our earlier report).
   The NN never sees val, so its val/test scores are clean features for a val-CV LightGBM, with no nested OOF needed.
2. **Target attention over history is where the NN gains come from; set attention is second-order.** Your ablation shows this (+0.010 from history).
   So does the literature. PRM's set-attention gain over DLCM almost disappears when its position embedding is removed (MAP 65.60 -> 64.73; DLCM 64.50) [V].
   DASALC self-attention adds +1.8 NDCG@10 on Web30K over a plain DNN, but that DNN has no list-normalised features [V].
   Our features already include within-feed z-scores and ranks.
   Invest in the history side: an attention bias from event type, source and recency (TWIN/HSTU), a count-preserving non-softmax channel (DIN/HSTU), and 1-2 layers.
   Stacking more set blocks is lower value.
3. **Use per-feature numerical embeddings instead of an MLP on z-scored inputs.**
   Options: PLR, i.e. periodic -> linear -> ReLU (Gorishniy 2022), or piecewise-linear encoding (as in TabM†).
   Add RealMLP-style robust scaling plus smooth clipping before them.
   MLP average rank 8.5 -> 3.0 with PLR. TabM† (PLE embeddings) is rank 1.7, against CatBoost 3.2 [V]. The RecSys'24 winner used sinusoidal numeric embeddings [V/S].
4. **Loss: softmax CE per feed as the main loss, plus a small pointwise BCE term.**
   Softmax CE is the robust neural choice (DASALC Table 6). LambdaRank works poorly for NNs there [V].
   On binarised LTR, Sigmoid+Softmax CE beats Softmax CE alone in NDCG@10 on all three datasets (0.4665 vs 0.4578 Web30K) [V].
   Try LambdaLoss/NeuralNDCG@max only as a fine-tuning stage; the differences are within about 0.5 NDCG point [V].
5. **Ensemble seeds and folds; average weights (EMA).**
   DASALC-ens (3-5 runs) added +1.26 NDCG@5 on Istella [V]. Riiid winner: 5 models gave +0.002 AUC [V]. TabM: k=32 implicit members, plus x5 deep ensemble [V].
   EMA is [K] (standard practice; no measured ablation found in a ranking competition).
6. **Medium priority: self-supervised pretraining of the history encoder on all histories (ARGUS recipe: next-item plus feedback heads).**
   In ARGUS it doubled the feature-mode uplift (+1.17% -> +2.32%) [V], but at 300B-interaction scale.
   We have about 0.6B history events, and the given embeddings probably already come from such a model. Do this only after items 1-5.
7. **Low priority or skip:** HSTU-scale generative models; listwide-label heads (RankFormer, PEAR), since every feed has a positive so there is no list-level signal;
   position embeddings of the train order; TabPFN on 1.4M rows; label smoothing; DCNv2 as a big lever (it is +0.0017 AUC over a tuned DNN on Criteo [V]).

---------------------------------------------------------------------------------------------------------------------------------------

## 1. When did neural rankers actually beat strong GBDT? (evidence table)

| Setting | Data type | Result | Source |
|---|---|---|---|
| Web30K / Yahoo / Istella LTR (ICLR'21) | dense hand features only | DASALC-ens vs LightGBM LambdaMART NDCG@5: Web30K 51.72 vs 49.66 (+4.2%), Yahoo 74.07 vs 74.21 (-0.2%), Istella 71.32 vs 71.24 (+0.1%). Single DASALC is below GBDT on Yahoo/Istella. SetRank 45.15 / 70.29 / 63.45, far below | [V] https://storage.googleapis.com/gweb-research2023-media/pubtools/5954.pdf |
| RankFormer (Amazon, KDD'23), public LTR | dense only | GBDT with 10^4 trees best on 2/3 datasets; RankFormer about equal to MLP. Their conclusion: "strong GBDT models can be too powerful to be outclassed by neural methods" | [V] https://arxiv.org/pdf/2306.05808 |
| RankFormer, Amazon Search, 500k queries | dense + upstream NN embeddings, text, image | RankFormer 37.33 vs GBDT-10^4 37.07 vs MLP 36.54 (NDCG@10). Distilling RankFormer into a 2000-tree GBDT gave 13.7% interleaving revenue credit lift over a GBDT distilled from the MLP | [V] same |
| ShareChat short video, 2B rows | ~500 features, embeddings, history | MMoE beats CatBoost on all 4 signals (Like AUC 0.9420 vs 0.9388). Gap comes from history and embedding handling plus data scale: MMoE 20M->9B rows gives 0.933->0.9417; CatBoost plateaus at 0.939 by 75-110M rows | [V] https://arxiv.org/pdf/2312.01760 |
| RecSys Challenge 2024 1st (DeNA ":D") | impression of in-view articles + history features | Transformer over the impression's candidates (token classification, BCE): AUC 0.8864 vs LightGBM 0.8817 vs CatBoost 0.8805; stage-2/3 ensemble 0.8924 | [V via slide summary] https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning |
| Riiid (Kaggle 2020-21) 1st | long event sequence + target item | 1-layer "last-query" transformer (target attends to history, O(L)) + LSTM, d=128, L=1728; single model 0.816/0.818 public/private AUC, 5-model ensemble 0.818/0.820 | [V] https://arxiv.org/pdf/2102.05038 |
| Yandex Music ARGUS (KDD'26) | 300B interactions, sequences | Standalone transformer below production GBDT (pair accuracy -4.78% to -7.49%); as GBDT feature +1.35% (3.2M params) to +2.66% (1B). A/B +2.26% listening time, +6.37% likes | [V] https://arxiv.org/pdf/2507.15994 |
| Yandex Ads (Jul 2026) | 8k-event histories | Transformer prediction injected as a feature into the production CatBoost: dNLL +11.7% to +12.3% (deployable), +16.2% (full 8k ceiling) | [V] https://arxiv.org/pdf/2607.14331 |
| Meta HSTU (ICML'24) | trillions of events, raw sequences only | GR NE .4845 vs DLRM .4982 (E-task); online +12.4%. DLRM restricted to the same raw features: .5053 | [V] https://arxiv.org/pdf/2402.17152 |
| Tabular benchmarks (TabM, ICLR'25) | small/medium tabular | TabM† (PLE) mean rank 1.7, CatBoost 3.2, LightGBM 3.4, MLP 5.5, FT-T 4.6 | [V] https://arxiv.org/pdf/2410.24210 |

**Pattern.** On dense, already-engineered features, a tuned GBDT ties or wins. NNs win, or add value as a stacked feature, when they consume raw sequences or embeddings that
GBDT can only see through hand-made aggregates, and when trained on large data (ShareChat scaling table).
Our setting is exactly that: 128-d item embeddings for the candidate and for every history event, typed events, and 247M training rows.
The current gap (NN 0.369 > LGBM 0.365 trained on val only) fits this. The GBDT can still close it by training on *train* too, which is a separate question.

---------------------------------------------------------------------------------------------------------------------------------------

## 2. Listwise / set-aware rankers (candidates attend to each other)

### PRM (Alibaba, RecSys'19) [V] https://arxiv.org/pdf/1904.06813
- Transformer encoder over the initial list. Input per item = [features, pretrained personalized vector PV] + **learnable position embedding of the initial rank**.
  Softmax output over the list, NLL loss. Default b=4 blocks, h=3 heads, dropout 0.1, Adam.
- Ablation, Yahoo LETOR MAP: default 65.60; **remove PE 64.73** (DLCM 64.50); remove dropout 65.17; blocks 1/2/6/8 give 65.49/65.49/65.61/65.14; heads 1/2/4 give 65.55/65.57/65.58.
  The "re-ranking" gain mostly comes from encoding the upstream ranker's order. Heads and depth hardly matter.
- E-commerce: PRM-Personalized-Pretrain (PV from a pretrained CTR model over user history) adds +4.5% MAP over PRM-BASE (30.15 vs 28.85).
  Online: PV +3.01%, CTR +2.6%, GMV +6.65% vs no re-ranking.
- **Transfer:** val/test order is random, so position embeddings are useless. Worse, the train order is the production order, so they would leak bias. What does transfer is the personalised user vector.

### SetRank (SIGIR'20) [V] https://arxiv.org/pdf/1912.05891
- Permutation-equivariant stack of MSAB/IMSAB blocks (6 blocks, 256 hidden, 8 heads, induced dimension 20, Adam 1e-3).
- MSLR30K NDCG@10: SetRank without initial ranking 0.4428 vs RankLib LambdaMART 0.4640. RankLib is a weak GBDT; the DASALC paper reports SetRank at 46.96 NDCG@10 vs LightGBM 51.48.
  **Set models alone do not beat a strong GBDT on dense features.**

### DASALC (Google, ICLR'21) [V] (PDF above)
- Recipe: log1p(|x|)·sign(x) input transform; Gaussian noise N(0, σ²) added after log1p (**σ=1.5 best** on Web30K);
  3-6 multi-head self-attention layers over the list (2-5 heads) followed by LayerNorm;
  **latent cross** h_i = (1 + a_i) ⊙ h_out(x_i), where a_i is the item's list-context embedding and h_out is the last hidden layer of the per-item DNN;
  softmax CE loss; hidden sizes up to 4096 ("large models work better when data augmentation is enabled"); ensemble of 3-5 runs.
- Cumulative ablation, Istella NDCG@5: DNN 64.72 -> +log1p 67.09 -> +self-attention 68.32 -> +latent cross 68.80 -> +noise augmentation 70.06 -> +ensemble 71.32.
- Noise **hurts** a plain DNN (σ=1.0 gives -2.17 NDCG@10) and hurts LightGBM (Yahoo σ=1.0 gives -5.4 NDCG@10). It helps only the attention model.
- **Transfer:** our f1..f5 are already normalised bins, so log1p is irrelevant for them (it is relevant for count features such as category counts or popularity).
  Latent cross is a cheap gating to add on top of the set block. Feature-noise augmentation is worth one run with the set model, at σ tuned on val.

### RankFormer (Amazon, KDD'23) [V] https://arxiv.org/pdf/2306.05808
- Listwise transformer + [CLS] token. Listwise softmax loss plus an ordinal "listwide" loss (α=0.25) that learns from lists with no positives.
  3 layers, 1 head, FFN 512, dropout 0.25, 200 epochs Adam. Scoring heads: one 128-d hidden layer.
- **Transfer:** the listwide signal needs lists without positives (or with graded outcomes). Our builder keeps only feeds with positives, so it adds nothing unless train has zero-positive feeds.
  The useful idea is distilling the listwise NN into the GBDT (fitting GBDT on NN scores), which kept the gain online.

### PEAR (Huawei, WWW'22) [V] https://arxiv.org/pdf/2203.12267
- Re-ranker with feature-level + item-level interactions, cross-attention to the user's clicked history, and a list-level CLS auxiliary task.
- Auxiliary task ablation is tiny: nDCG@30 0.6747 -> 0.6755. History length helps most (their Figure 2).

### RecSys Challenge 2024 1st place (Ekstra Bladet news; closest public analogue) [V via WebFetch summary of slides]
- Transformer where each in-view article of an impression is a token. Impression-level features (time since history, counts, device) and per-candidate features
  (publish-time diff, ranks within impression, history-similarity, popularity counts) go in together. Categorical and numeric features are encoded separately;
  **numerics with sinusoidal embeddings**. BCE token classification. Sampling 3 impressions per user per epoch. Re-train on train+val with the same hyperparameters for the final model.
- Stage 2: LightGBM (Optuna) over stage-1 predictions; stage 3: simple average. Numbers in the table in §1.
  Removing the transductive "future impression" features cut AUC 0.8864 -> 0.8680.

**Set-attention takeaways for us**
- Your set block is permutation-equivariant with no positional encoding. That is correct, and it must stay so.
- Expected gain on top of feed-normalised hand features: small (+0.001-0.005; the noset ablation is still running). Cheaper variants with similar effect:
  (a) latent cross (DASALC); (b) a single set block after the target-attention blocks instead of interleaving; (c) feed-mean-pooled context concatenated to each token.
- The listwise context that matters most here is probably category competition within the feed and the feed's score distribution.
  Within-feed z-scores and ranks of key NN-internal similarities (already in `dense_features`) cover much of it.

---------------------------------------------------------------------------------------------------------------------------------------

## 3. Target attention over user history

### DIN (Alibaba, KDD'18) [V] https://arxiv.org/pdf/1706.06978
- Activation unit: an MLP over [e_hist, v_cand, e_hist ⊙/outer v_cand] gives a weight w_i. The weights are **not softmax-normalised**:
  "the constraint of Σ w_i = 1 is relaxed ... to reserve the intensity of user interests". User vector = Σ w_i e_i.
- Amazon Electronics AUC: BaseModel 0.8624 -> DIN 0.8818 -> DIN+Dice 0.8871. MovieLens 0.7300 -> 0.7337 -> 0.7348.
  Mini-batch-aware L2 on ID embeddings was the key regulariser on the 0.6B-ID Alibaba data (AUC 0.5940 without ids -> 0.6031 with ids + MBA reg).

### TWIN (Kuaishou, KDD'23) [V] https://arxiv.org/pdf/2302.02352
- Attention logit = (K_h W_h)(q W_q)^T / sqrt(d_k) + (K_c W_c) β, where K_h are the behaviour's *inherent* item features
  and K_c are user-item *cross* features (e.g. how the user interacted, time gap), each compressed to a scalar bias.
  Softmax over the top-100 selected behaviours.
- Removing the cross-feature bias ("TWIN w/o Bias") "leads to a significant drop in GAUC" while saving little compute.
  TWIN vs best baseline: +0.29% AUC, +0.51% GAUC (0.05% is "enough to bring online benefits"). TWIN-V2 adds hierarchical clustering for life-long histories, which is irrelevant for 256 events.
- **Transfer:** our cross features per history event are event_type (10), event_source (27), recency index and same-category flag.
  Add them as a learned scalar bias per head on the attention logits, in addition to adding them to the key embedding as now. Cost is negligible.

### TransAct (Pinterest, KDD'23) [V] https://arxiv.org/pdf/2306.00248
- 100 realtime actions; token = [action-type embedding, 32-d PinSage embedding (frozen pretrained)].
  **Early fusion "concat"**: each action token is concatenated with the candidate's PinSage embedding before a 2-layer, 1-head transformer (FFN 32).
  **No positional encoding** ("position information is ineffective", since they have timestamps).
  Output = first K=10 tokens + max-pool. A random time-window mask at training time fights the "rabbit hole" effect. A DCNv2 feature cross sits on top.
- Ablations (HIT@3 relative): encoder average pooling +0.21%, CNN +0.08%, RNN -1.05%, LSTM -0.75%, transformer +1.56% repin (hide -8.45%).
  Concat beats append (BST-style). Output first-K+maxpool +9.41% vs max-pool alone +6.38%. 4 layers / FFN 384 was best but +30% latency.
  Removing TransAct: -8.59% repin. Removing the batch user embedding (PinnerFormer): -2.46%. The two are complementary.
- **Transfer:** (i) action types are essential, since a raw embedding sequence cannot tell a "contact" from a "view". You do this with emb(event_type)+emb(source).
  (ii) Our cross-attention (candidate queries over history keys) is the efficient equivalent of early fusion; the true concat costs B×M×K tokens (~24.6k per feed).
  A middle ground: an MLP on [h_k, c, h_k⊙c] to compute attention logits only (DIN-style), keeping values from h_k.
  (iii) Keep the given user embedding as a separate token. It plays the PinnerFormer role (complementary long-term signal).

### HSTU / "Actions speak louder than words" (Meta, ICML'24) [V] https://arxiv.org/pdf/2402.17152
- Pointwise aggregated attention: A = φ2(QK^T + rab_{p,t}) with φ2 = SiLU (no softmax), then LayerNorm(A V) ⊙ U(X) gating.
  Motivation: "the number of prior data points related to target serves as a strong feature indicating the intensity of user preferences, which is hard to capture after softmax".
- Ranking NE ablation (industrial, lower is better), E-task: HSTU without rab and with softmax .5067; without rab (SiLU) .4980; HSTU (rab with position+time) .4937.
  So softmax -> SiLU gives -1.7% NE, and relative position/time bias a further -0.9%. Standard transformers diverged (NaN) on ranking.
- ARGUS later found **no** clear advantage of HSTU over a standard transformer at equal size (pair accuracy uplift +2.34% vs +2.32%) [V].
- **Transfer:** do not adopt the GR paradigm. Do add an intensity channel. Either one non-softmax head: Σ_k SiLU(q·k/√d + bias_k) v_k / K_max.
  Or explicit DIN-style sums Σ_k σ(MLP([h_k, c, h_k⊙c])) for k in history, which generalises the hand-made "count of same-category events" and "max-sim" features.

### Riiid 1st place [V] https://arxiv.org/pdf/2102.05038
- "Last query" transformer: only the target position is the query, keys are the whole history (O(L)). 1 layer was enough: "longer sequences gave better valid AUC than more layers".
  d=128, heads 2/4/8/16/32 for 5 ensemble members, LSTM on top. Continuous features via "continuous embedding" (linear), categorical via embeddings.
- **Transfer:** our cross-attention blocks are exactly this. 1-2 cross blocks are likely enough; extra history self-attention layers may add little (worth an ablation: hist_layers 0/1/2).

### Practical spec for our history encoder (synthesis; [K] unless cited)
- Token = W_e·dequant(item_emb) + emb(event_type) + emb(event_source) + emb(category) + emb(recency index 0..255) (+ missing-embedding vector). That is what you have.
- Attention logit bias per head: β_type[event_type] + β_src[source] + β_rec[bucket(recency)] + β_cat·1[cat_hist == cat_cand] (TWIN cross-feature bias; HSTU rab).
- Two kinds of pooling per candidate: softmax target attention (as now) **plus** one count-preserving pooling (DIN/HSTU).
  Also expose top-k similarity statistics inside the network (you already feed max/mean/LSE similarities as dense features, which is good).
- Separate attention heads or masks per event-type group (e.g. strong actions vs views). TransAct shows that negative and positive actions must be distinguishable.
- History of 256 needs no GSU/ESU retrieval stage (SIM, TWIN, ETA are for 10^3-10^5).

---------------------------------------------------------------------------------------------------------------------------------------

## 4. Pretraining the history encoder (ARGUS recipe)

[V] https://arxiv.org/pdf/2507.15994 (Yandex Music, KDD'26)
- Pretraining: causal transformer over (context, item, feedback) triplets, merged into one embedding per interaction in the "simplified" variant.
  Two heads: **next-item prediction** (sampled softmax, 8192 uniform + 8192 in-batch negatives) and **feedback prediction**
  (multi-factor feedback: likes, skips, completion), L = L_NIP + L_FP.
  Then **impression-level fine-tuning**: user hidden state matched to impressions by timestamp, dot-product scoring, pairwise logistic loss.
- Hyperparameters: Adam, grad clip 1, 3000 warmup steps, dropout 0.1, batch 4096 pretrain / 2048 fine-tune, backbone LR 1e-5 -> 1e-4 -> 1e-4, head LR 1e-3 -> 1e-3 -> 1e-4;
  sequence 512 pretrain / 2048 fine-tune; 1 epoch.
- Ablations (feature-mode pair accuracy uplift): no pretraining + 1 year fine-tune +1.17%; pretraining + 1 week fine-tune +0.63%; pretraining + 1 year +2.32%.
  Context 512 / 2048 / 8192: +1.01 / +2.32 / +2.77%. Model 3.2M -> 1B: +1.35 -> +2.66%.
- Yandex Ads follows the same recipe (pretrain 1 epoch, fine-tune 3 epochs end-to-end, DCNv2 in the head) and evaluates as a CatBoost feature [V] https://arxiv.org/pdf/2607.14331.
- **Transfer and caveats:**
  - Our histories are newest-first without timestamps. Reverse them to oldest-first for a causal model.
  - Deduplicate per user (consecutive feeds share most of the history).
  - The pretraining targets would be the next event's item embedding (contrastive against in-batch items) plus the next event's type/source (feedback head).
  - The given user embedding and item embeddings are probably products of an Avito model trained on far more data (Avito Habr: two-tower, transformer user tower, InfoNCE).
    The marginal gain of our own pretraining on about 0.6B events is therefore uncertain. ARGUS's own data shows "fine-tuning must remain large" and gains scale with data.
  - Cheaper alternative with most of the benefit: train the ranker end-to-end on the 3.15M train feeds (already done).
    Optionally add an auxiliary next-event-type head on history tokens as a regulariser.
  - Priority: medium. Run it once the main ranker plateaus. Compute is fine on 4×GPU.

---------------------------------------------------------------------------------------------------------------------------------------

## 5. Numerical features in neural nets

### Gorishniy et al., "On Embeddings for Numerical Features" (NeurIPS'22) [V] https://arxiv.org/pdf/2203.05556
- Periodic(x) = concat[sin(2π c_i x), cos(2π c_i x)], c_i ~ N(0, σ) trainable. **PLR** = ReLU(Linear(Periodic(x))), separate per feature. "σ is an important hyperparameter".
- PLE (piecewise-linear encoding): quantile (Q) or target-tree (T) bin edges; the component is 1 below x, the fractional position inside x's bin, and 0 above.
  Followed by Linear (+ReLU).
- Average rank across 11 datasets (lower is better): MLP 8.5, MLP-LR 5.5, MLP-Q-LR 5.1, **MLP-PLR 3.0**, ResNet-PLR 3.2, Transformer-PLR 3.9, CatBoost 3.6, XGBoost 4.6.
  "after the MLP-like architectures are coupled with embeddings for numerical features, they perform on par with the Transformer-based models".
- Caveat: parameter count grows a lot (MLP-PLR up to ×250 on some datasets).

### TabM (ICLR'25) [V] https://arxiv.org/pdf/2410.24210
- Uses an "updated piecewise-linear embedding" (quantile bins, Q-L/T-L style, faster parametrisation). PLE bins in UniformInt[8, 32].
- The † variants (with PLE) are the best: TabM†mini mean rank 1.7.

### RealMLP (NeurIPS'24) [V] https://arxiv.org/pdf/2407.04491
- Preprocessing: x -> f(s·(x - median)), s = 1/(q75 - q25) (fallback 2/(q100 - q0)), **f(x) = x / sqrt(1 + (x/3)²)** (smooth clip to (-3, 3)).
  "our numerical preprocessing is easy to adopt and often beneficial for other NNs as well".
- PBLD embedding per numeric feature: [x, W2·cos(2π w1 x + b1) + b2] ∈ R^4 (w1, b1 ∈ R^16).
- Diagonal "scaling layer" (learnable per-feature scale, high LR) for soft feature selection.
- Other parts: NTK parametrisation, SELU/Mish with learnable PReLU-like mix, coslog4 LR schedule, flat_cos schedule for dropout and weight decay (wd = 0.02), Adam β2 = 0.95.
- **Label smoothing (ε = 0.1) "can be detrimental for metrics like AUROC"**, so do not use it for ranking.

### Transfer to our ~70 in-network dense features
- Replace `X -> zscore -> clamp(-6, 6) -> MLP` with `X -> robust scale + smooth clip -> per-feature PLE (16 quantile bins from train) or PLR (k=16-48 frequencies, σ ∈ {0.01, 0.1, 1}) -> d_f = 8-16 each -> concat -> Linear`.
  Keep the raw scalar too (PBLD-style). RecSys'24 winner: sinusoidal numeric embeddings in the candidate transformer.
- f1..f5 are already "normalized interpolated bins" (official README). Avito's production ranker encodes numerics via bin-boundary embeddings with proximity weighting
  (Habr, secondary). PLE is the same idea, and it fits these features naturally.
- Expected gain: modest but reliable (+0.001-0.004 val NDCG is my guess). Cheap to try. Most useful for the features the NN uses most: similarities, f1-f5, ranks.

---------------------------------------------------------------------------------------------------------------------------------------

## 6. Tabular-DL SOTA 2024-2026: relevance for us

- **TabM** [V]: k=32 implicit MLP members via BatchEnsemble adapters r_i, s_i, b_i. Only the first adapter is randomly initialised to ±1; the others start at 1.
  Members are trained in parallel. The loss is the **mean of the k individual losses**; the prediction is the mean of the k predictions.
  Tuning space: layers 1-5, width 64-1024, dropout 0-0.5, LR 1e-4..5e-3 (3e-3 with PLE), wd 0 or 1e-4..1e-1, AdamW, patience 16.
  "individual submodels are weak and overfitted, while their average is strong".
  - Use 1: as the second-level model on val (features + NN scores) alongside LightGBM, for blend diversity. Score with the per-feed softmax loss instead of BCE.
  - Use 2: put a BatchEnsemble head (k=8-32) on the ranker's final MLP. It is a cheap implicit ensemble that also gives seed-like diversity.
- **RealMLP** [V]: strong defaults, which suit a val-only second-level model. Take the preprocessing (§5) even if not the model.
- **FT-Transformer** [V] https://arxiv.org/pdf/2106.11959: default 3 layers, d=192, 8 heads, ReGLU FFN ×4/3, attention dropout 0.2, FFN dropout 0.1, AdamW LR 1e-4, PreNorm.
  The paper says GBDT still dominated on some datasets after tuning, including Yahoo LTR.
  Feature-token attention adds little over per-feature embeddings + MLP (Gorishniy 2022 [V]).
- **TabArena (2025)** [V abstract] https://arxiv.org/abs/2506.16791: "deep learning methods have caught up under larger time budgets with ensembling";
  "foundation models excel on smaller datasets"; cross-model ensembles are SOTA; warning: "some deep learning models are overrepresented in cross-model ensembles due to validation set overfitting".
- **TabPFN v2 / 2.5** [S] https://arxiv.org/html/2511.08667v2: validated up to about 50k rows and 2k features (≤160k rows in memory on an H100). It is a classifier, not a ranker.
  Val has 1.39M rows, so it would need sub-sampling or feeds as separate contexts. Low priority. A distilled or kNN-like variant is not worth the effort here.

**Conclusion:** tabular DL is for *second-level, val-only* modelling and blend diversity. The first-level gains come from the sequence/embedding NN trained on train.

---------------------------------------------------------------------------------------------------------------------------------------

## 7. Explicit feature crossing (DCNv2)

[V] https://arxiv.org/pdf/2008.13535
- Cross layer: x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l. W is numerically low-rank, hence DCN-Mix (mixture of low-rank experts with gating).
- Criteo: DNN logloss 0.4421 / AUC 0.8098 vs DCN-V2 0.4406 / 0.8115 vs DCN-Mix 0.4408 / 0.8112. MovieLens: 0.3201 vs 0.3170 logloss.
  The authors note a tuned DNN "performed neck to neck" with most crossing models. Production: 0.6% relative AUC-loss improvement.
- Used in TransAct (full-rank DCNv2 on top of TransAct output + features) [V] and in the Yandex Ads head [V]. Avito Habr uses DCN crosses per feature group (secondary).
- **Transfer:** our head already multiplies via attention and the ce⊙ue input. A 2-layer DCNv2 on [token, user, pooled history, dense] before the final MLP is cheap. Expect ≤ +0.002.

---------------------------------------------------------------------------------------------------------------------------------------

## 8. Losses for sparse binary targets and full-list NDCG

Metric facts: binary gain, full list, pessimistic ties, mean over feeds. With one positive, NDCG = 1/log2(1 + rank_pos).
The metric is a smooth, steep-at-top function of the positive's rank. Moving a positive from 60 to 30 still matters.

| Evidence | Numbers | Source |
|---|---|---|
| Neural DNN, Web30K NDCG@5 by loss | SigmoidCE 46.85, RankNet 46.67, **LambdaRank 46.55**, **Softmax 48.22**, ApproxNDCG 47.87, GumbelApproxNDCG 48.07, NeuralSortNDCG 47.19. "LambdaRank does not work well for neural models" | [V] DASALC App. B.2 |
| Neural transformer (allRank), Web30K NDCG@10 | NeuralNDCG@max 53.46, **LambdaRank@max 53.47**, ListNet 52.80, RMSE 52.46, ApproxNDCG 50.90, ListMLE 51.82 | [V] https://arxiv.org/pdf/2102.07831 |
| Binarised LTR, 3-layer DNN, NDCG@10 | Web30K: SoftmaxCE 0.4578, SigmoidCE 0.4626, **Sigmoid+Softmax 0.4665**, Sigmoid+ListCE(σ) 0.4680. Yahoo: 0.6993 / 0.6852 / 0.7008 / 0.7050. Istella: 0.6839 / 0.6560 / 0.6861 / 0.6900 | [V] https://arxiv.org/pdf/2211.01494 |
| GBDT: softmax (ListNet) vs LambdaMART | Web30K-b NDCG@5 75.83 vs 78.16; Yahoo-b 89.95 vs 90.56. Trees prefer lambda gradients; the gap narrows on binary labels | [V] https://storage.googleapis.com/gweb-research2023-media/pubtools/5124.pdf |
| GBDT: XE-NDCG (= LightGBM rank_xendcg) | Web30K NDCG@5 48.23 vs λMART 48.08; more robust to label noise | [V] https://arxiv.org/pdf/1911.09798 |
| GBDT: LambdaLoss NDCG-Loss2++ (µ=5) in LightGBM | Test NDCG@5: Yahoo 75.70 vs 75.40, Web30K 51.21 vs 50.74 | [V] https://storage.googleapis.com/gweb-research2023-media/pubtools/4591.pdf |
| Theory | Softmax CE bounds log-MRR and log-NDCG for binary relevance | [V] same Bruch ICTIR'19 PDF |

Recommendations:
- **NN:** loss per feed = SoftmaxCE(s, y/Σy) + λ·mean_i BCE(s_i, y_i) with λ ≈ 0.1-0.5 (tune).
  - Average over **feeds**, not candidates, because the metric weights feeds equally. Per-candidate BCE overweights long feeds.
  - For feeds with several positives, test the "one softmax per positive, others excluded" variant (each positive vs all negatives) against ListNet's y/Σy.
  - Fine-tune stage to try: LambdaLoss NDCG-Loss2++ or NeuralNDCG with the full-list discount (no truncation). Expect ≤0.002; keep it only if val improves.
- **GBDT:** lambdarank with `lambdarank_truncation_level` ≥ max feed length (otherwise only the top-30 pairs get gradients). Compare against rank_xendcg (and YetiRank in CatBoost).
- **Calibration does not matter** for per-feed NDCG. The BCE term is there as a regulariser, and it gives stable, comparable scores for stacking features.
- Position bias (train): keep the additive PAL/shallow-tower logit inside the loss and drop it at inference. Your nopos vs base runs show it is not yet clearly helping (0.370 vs 0.369 mid-training).
  Decide on the final checkpoint with paired per-feed bootstrap, not on one step.

---------------------------------------------------------------------------------------------------------------------------------------

## 9. Training recipe (with what is verified vs assumed)

**Data and batching**
- One feed = one list. Bucket feeds by M (candidates) and H (history length) to limit padding. A candidate budget per batch (you have `cand_budget`) beats a fixed number of feeds.
  Shuffle candidates inside train feeds, so nothing order-dependent can be learned (set model, no PE).
- Gather int8 embeddings on GPU from item tables (dequantise in-kernel). Compute dense features on GPU (you already do).
- Sampling: RecSys'24 winner sampled 3 impressions per user per epoch (~25% of data) to decorrelate users [V]. Here about 1.05 feeds per user in val/test.
  In train, check feeds per user; if heavy users dominate, cap feeds per user per epoch.

**Optimisation**
- AdamW, β = (0.9, 0.98) [K]. LR 1e-3 with warmup and cosine is fine for d=128 (you have this).
  ARGUS used a separate, higher head LR than backbone LR in fine-tuning [V]. Grad clip 1.0 [V ARGUS].
- bf16 autocast plus SDPA (flash/memory-efficient kernels). Keep the softmax/loss in fp32. You already cast the loss to float. [K]
- Epochs: with frozen pretrained embeddings and no ID embeddings, multi-epoch training is fine.
  If you add **learnable item_id/user_id embeddings**, expect the "one-epoch phenomenon": deep CTR models with ID embeddings "exhibit a dramatic degradation at the beginning of the second epoch",
  and "training the models for more epochs does not improve over models trained in one epoch with appropriate hyperparameters" [V] https://arxiv.org/pdf/2209.06053.
  Either train ID embeddings for 1 epoch only, or use strong regularisation (DIN's mini-batch-aware L2 [V]).
- Model selection on the **exact** metric (full NDCG, pessimistic ties) on val (or a held-out val slice if val is also used for fine-tuning).
  Logloss-based selection, as in the official baseline, is misaligned.

**Averaging and ensembling**
- Seeds: DASALC-ens, average of 3-5 runs: +1.26 NDCG@5 Istella, +0.80 Web30K [V]. Riiid: 5 models (different head counts) +0.002 AUC [V].
  TabM: implicit k=32 plus x5 deep ensemble both help [V]. 4 GPUs make 5-10 seeds per config cheap.
- EMA of weights (decay 0.999-0.9999) or SWA over the last 20-30% of the cosine schedule: standard and nearly free [K]. A KDD Cup winner lists EMA among its tricks [S].
  Verify on val: expect a smoother curve and +0.001-0.002.
- Diversity sources for the blend: different losses (softmax vs BCE vs both), with and without set attention, different history encoders, PLE vs PLR. Average per-feed ranks or z-scores.

**Fine-tuning on val (uniform data)**
- Initialise from the train model; k-fold over val feeds; low LR (1e-4 to 3e-5); 1-3 epochs; EMA; freeze the embeddings/encoder for the first epoch [K].
  Compare against the alternative of using the NN score as a GBDT feature. Stacking is lower variance on 14.4K feeds.
- If fine-tuning: keep a time-ordered holdout (last val day) to choose epochs and LR, because test (and especially private, the last 70%) is later than val.

---------------------------------------------------------------------------------------------------------------------------------------

## 10. Combining NN and GBDT (what the strongest teams and companies do)

1. **NN score -> GBDT feature (stacking).** ARGUS (+1.35% to +2.66% pair accuracy in feature mode versus negative standalone), Yandex Ads (+11.7-12.3% dNLL as a CatBoost feature),
   RecSys'24 winner (stage-2 LightGBM over stage-1 NN+GBDT predictions) [all V].
   For us: train the NN on train (no val), score val+test, then add [score, within-feed rank, z-score, gap to feed max] as LGBM features in the existing val-CV.
   Also add internal NN quantities, such as the attention-pooled similarity and the count-preserving pooled value.
   No nested OOF is needed because the NN never saw val. If the NN *is* fine-tuned on val, produce out-of-fold val predictions with the same fold split as the GBDT.
2. **Distillation (RankFormer [V]):** fit GBDT on NN scores for the train candidates. This is useful for transferring listwise knowledge into a fast model.
   Here it mainly gives diversity; the stacking route is simpler.
3. **Rank/z-score blend** of NN seeds/variants and GBDTs, with weights tuned by grouped CV on val. Final stage of the RecSys'24 winner = simple average [V].
4. Watch the TabArena caveat: blends tuned on a small validation set overfit [V abstract]. Use few blend weights and paired per-feed bootstrap.

---------------------------------------------------------------------------------------------------------------------------------------

## 11. Concrete experiment plan (ordered; expected deltas are my estimates, not literature numbers)

| # | Experiment | Expected val Δ | Cost |
|---|---|---|---|
| 1 | Finish full-epoch runs (base / nopos / noset / nohist), 3 seeds each, then paired bootstrap | establishes noise floor (~±0.002) | 4 GPU × hours |
| 2 | NN score features -> val-CV LightGBM (stacking) + rank blend | +0.005 to +0.015 over the best single model | small |
| 3 | Attention logit bias from event_type/source/recency/same-category (TWIN/HSTU) + one count-preserving SiLU/sigmoid pooling channel (DIN/HSTU) | +0.002 to +0.006 | small |
| 4 | Per-feature PLE/PLR numeric embeddings + robust-scale/smooth-clip | +0.001 to +0.004 | small |
| 5 | Loss: softmax + λ·BCE (feed-averaged), multi-positive variant | +0.000 to +0.003 | trivial |
| 6 | EMA + 5-10 seeds x 2-3 architectural variants, averaged per feed | +0.003 to +0.008 | GPU time only |
| 7 | Scale: d 128 -> 256, cross blocks 2 -> 3, full 2-3 epochs on all 3.15M feeds (check overfit per epoch on val) | +0.002 to +0.008 | moderate |
| 8 | Set-model upgrades (latent cross, feature noise σ tuned) | +0.000 to +0.003 | small |
| 9 | k-fold fine-tune on val (low LR) vs stacking only | uncertain (±) | moderate |
| 10 | ARGUS-style pretraining of the history encoder on deduplicated user histories (next-item contrastive + next event-type) | +0.000 to +0.006 (uncertain) | large |
| 11 | Learnable item_id embeddings (1 epoch, regularised): item quality/popularity beyond the given embeddings | uncertain; risk of overfit | moderate |

---------------------------------------------------------------------------------------------------------------------------------------

## 12. Things that look attractive but are wrong or low-value here

- **Position embeddings / order-aware re-rankers** (PRM PE, DLCM GRU, Seq2Slate): val/test order is random, and the train order is production order (bias leak). PRM's gain is mostly PE [V].
- **Listwide-label heads** (RankFormer α, PEAR CLS task): need lists without positives; the PEAR aux gain was +0.0008 nDCG@30 even when applicable [V].
- **HSTU/GR at scale**: their gains need trillion-event streams; ARGUS found no HSTU advantage at equal size [V]. Borrow only the SiLU-attention and rab ideas.
- **LambdaRank loss as the primary NN loss**: worst listwise choice in DASALC's neural benchmark [V]. Only LambdaRank@max on par in allRank [V]. Use softmax CE first.
- **Label smoothing** (RealMLP default for classification): hurts AUROC-like metrics [V].
- **Gaussian feature noise on a plain MLP or GBDT**: hurts [V]. Only helps attention models.
- **TabPFN** on 1.4M val rows or 247M train rows: out of its validated range (≤50k rows for 2.5) [S], and it is not a ranking model.
- **Logloss-based checkpoint selection** (official baseline): select on exact full-list NDCG.
- **Over-trusting public LB**: 30% of test, ~8.6K feeds, SE ~0.003. Use paired per-feed bootstrap on val.

---------------------------------------------------------------------------------------------------------------------------------------

## 13. Sources

Read this session [V] (full text unless noted):
- DASALC, "Are Neural Rankers still Outperformed by GBDT?", ICLR'21: https://storage.googleapis.com/gweb-research2023-media/pubtools/5954.pdf
- PRM, Personalized Re-ranking for Recommendation, RecSys'19: https://arxiv.org/pdf/1904.06813
- SetRank, SIGIR'20: https://arxiv.org/pdf/1912.05891
- RankFormer (Amazon), KDD'23: https://arxiv.org/pdf/2306.05808
- PEAR (Huawei), WWW'22 companion: https://arxiv.org/pdf/2203.12267
- RecSys Challenge 2024 overview: https://arxiv.org/pdf/2409.20483 ; 1st place slides (via WebFetch summary): https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning
- Riiid 1st place, Last Query Transformer RNN: https://arxiv.org/pdf/2102.05038
- DIN, KDD'18: https://arxiv.org/pdf/1706.06978
- TWIN, KDD'23: https://arxiv.org/pdf/2302.02352 ; TWIN-V2 (skimmed): https://arxiv.org/pdf/2407.16357
- TransAct, KDD'23: https://arxiv.org/pdf/2306.00248
- HSTU, ICML'24: https://arxiv.org/pdf/2402.17152
- ARGUS (Yandex), KDD'26: https://arxiv.org/pdf/2507.15994
- Long-History User Transformers for Real-Time Ad Ranking (Yandex, 2026): https://arxiv.org/pdf/2607.14331
- ShareChat GBDT vs neural rankers, FIRE'23: https://arxiv.org/pdf/2312.01760
- On Embeddings for Numerical Features, NeurIPS'22: https://arxiv.org/pdf/2203.05556
- TabM, ICLR'25: https://arxiv.org/pdf/2410.24210
- RealMLP, NeurIPS'24: https://arxiv.org/pdf/2407.04491
- FT-Transformer, NeurIPS'21: https://arxiv.org/pdf/2106.11959
- TabArena (abstract only): https://arxiv.org/abs/2506.16791
- DCN-V2, WWW'21: https://arxiv.org/pdf/2008.13535
- Bruch et al., Softmax CE for LTR with binary relevance, ICTIR'19: https://storage.googleapis.com/gweb-research2023-media/pubtools/5124.pdf
- Bruch, XE-NDCG, WWW'21: https://arxiv.org/pdf/1911.09798
- LambdaLoss, CIKM'18: https://storage.googleapis.com/gweb-research2023-media/pubtools/4591.pdf
- NeuralNDCG: https://arxiv.org/pdf/2102.07831
- Regression Compatible Listwise Objectives (Google), CIKM'23: https://arxiv.org/pdf/2211.01494
- One-epoch overfitting in deep CTR models, CIKM'22: https://arxiv.org/pdf/2209.06053
- gSASRec (skimmed; negative-sampling overconfidence, mostly irrelevant since our lists are complete): https://arxiv.org/pdf/2308.07192
- Project files: <repo>/baseline_org/neural_baseline.py (official transformer baseline: Linear(142)->width, user token, TransformerEncoder norm_first, BCE, logloss selection), <repo>/nn/model.py, <repo>/nn/logs/*.log

Search-snippet only [S]: TabPFN-2.5 limits (https://arxiv.org/html/2511.08667v2); "Rankformer" WWW'25 is a graph transformer for CF (https://arxiv.org/html/2503.16927v1), not a listwise re-ranker.
**No Google "Rankformer 2025" listwise paper exists** as far as two searches could find; the listwise RankFormer is Amazon's KDD'23 paper above.
KDD Cup winner write-up listing EMA/bf16/torch.compile (https://www.besthub.dev/articles/how-i-won-the-kdd-cup-using-deepseek-s-web-interface-117b74f34ef8).
