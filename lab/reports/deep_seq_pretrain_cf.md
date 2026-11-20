# Deep dive: collaborative and sequence signals from user histories, as reranker features or pretraining

Date: 2026-11-11. Scope: which history-derived signals (co-visitation, item2vec, SASRec/BERT4Rec/gSASRec, ALS/EASE/SLIM, LightGCN/graph, contrastive adapters on frozen embeddings, transductive pretraining) give the largest gains as reranking features when strong pretrained user and item embeddings are already provided. The report ends with recipes sized for our data: about 465M history tokens, about 40M items seen at least twice (91M unique), and 4x GPU GPUs.

Tags: **[V]** = verified this session by reading the source (URL given). **[S]** = seen only in a search snippet. **[K]** = my background knowledge, not re-verified. **[L]** = our own local EDA logs (path given). Gain numbers I estimate myself are marked "estimate".

---

## 0. TL;DR (ranked)

| # | Signal | Why it should add on top of the provided embeddings | Evidence strength | Cost | Priority |
|---|---|---|---|---|---|
| 1 | **Hybrid sequence model pretrained on all histories** (train + val + test, one leak-safe sequence per user). Item = frozen provided emb -> MLP **plus** a learnable ID residual. Objective: "dense all-action" next-*new*-item prediction plus event-type (feedback) prediction. Then (a) its score as an LGB feature and (b) its weights to initialise the NN ranker's history encoder, fine-tuned listwise on feeds | Provided item embeddings are probably content-only (no item ID), so there is no item-level memorisation. A model trained on this exact surface's histories adds ID memorisation, event-type semantics and "explore-only" targets. Strongest production evidence: ARGUS, PinnerFormer, TransAct, Airbnb, Avito's own transformer | Strong (several [V] production ablations) | 1-2 days of work; hours of GPU | **high** |
| 2 | **Event-type-aware, directional co-visitation** (exact, query-driven), built transductively from all histories, aggregated over the feed history with recency and event-type weights and several normalisations | ID-level "people who did A then did B" signal. In OTTO and KDD Cup 23 it dominated feature importance. Already +0.0153 over f1 in a simple rank blend [L] | Strong ([V] OTTO / KDD Cup writeups + [L]) | Hours on CPU | **high** (extend what exists) |
| 3 | **Leak-safe construction protocol** for all history-derived features (earliest-snapshot corpus or leave-user-out) | Without it, any feature computed on train feeds memorises train positives that leak into later snapshots of the same user (54% of positives reappear later [L]) | Necessary condition | Low | **high** |
| 4 | Tuned item2vec (SGNS) on ID sequences: window 3-7, negative-sampling exponent alpha < 0, sub-sampling 1e-5, more epochs | Cheap ID-level similarity; tuning roughly doubles HR@10 vs defaults [V]. Largely subsumed by #1 and #2 | Medium | 1-2 GPU hours | medium |
| 5 | Content-to-CF adapter (contrastive MLP on frozen emb trained on co-visitation / next-item pairs) to score **cold** candidates (~45% of candidates have no ID history) | Transfers CF to items with no history. This is the content path of #1, so skip it if #1 is built with ID-dropout | Medium ([V] NVIDIA KDD Cup 23, UniSRec) | Low | medium (as part of #1) |
| 6 | iALS / EASE / SLIM / LightGCN / TwHIN user-item models | Redundant with sequence models when those exist. iALS and LightFM were the weakest components in RecSys Challenge 2025 [V]. A graph model added ~0 next to a sequence encoder [V]. EASE is infeasible at 40M items | Weak | Medium | **low** |
| 7 | BERT4Rec | SASRec trained with CE or sampled softmax beats it at equal loss [V] | Strong negative | n/a | skip |

Expected incremental OOF NDCG in our LGB (estimate, wide uncertainty): #2 extended +0.003..0.008; #1 as a feature +0.005..0.015; #1 as initialisation of the NN ranker: further gain in the NN, then via stacking; #4 +0.000..0.003 after #1 and #2; #6 approximately 0.

---

## 1. What the provided embeddings probably already contain (and therefore what is left)

Avito's production two-tower transformer, described by Avito's personalisation team on Habr [V] https://habr.com/ru/articles/1004694:
- **Item tower = content only**: binary, categorical (category ID, location ID, top-10k info-model attributes), numeric (price, seller rating; bin-boundary embeddings), and text as a sum of token embeddings. Features are crossed with DCN per group, then DCN again. **No ad-ID embedding is mentioned**. The article only lists "embeddings for category ID, location ID and other categorical features".
- **User tower** = transformer over two sub-sequences (ad interactions and search queries) with positional embeddings. In production it reads the history *as item-tower embeddings from a Redis cache*. Later versions use several user embeddings, each scored against the ad, and aggregate the scores.
- **Training**: pairs of (history, ad with a target action). Targets are described as "clicks, adding to favourites and other activities". A random gap of minutes to days is inserted between history and target, and same-session events are removed. Loss: InfoNCE (beat SigLIP slightly). Negatives: in-batch, random from the catalogue, and exchanged across GPUs. Effective batch about 30K on 4x80GB, about 130K on the GPUs, about 16K negatives per GPU. About 10B samples over 2 months.
- **Downstream**: the user-ad dot product is a ranker feature for every candidate. It was the **top-1 SHAP feature**: +2.3% buyer target actions on the main page. As a candidate generator in "Personal items": +8.1% there.
- Avito's head of recommendations also said that cross-category recommendations moved to SASRec, "about two times better on metrics". Listing attractiveness (views and view-to-contact conversion) is used as a separate signal, and "clickbait" listings (many views, few contacts) are suppressed [V] https://kod.ru/monetizatsiya-rekomendatsii-i-llm-v-avito.

We do not know that the competition embeddings come from this exact model. They are int8 and 128-d, and the article gives no dimension or quantisation. Our data agrees with this picture, though: user_dot adds +0.0164 to the f-features in LGB, and history max-sim adds +0.028 [L] `eda/embeddings/05_lgb_lift.log`.

**What is left for us, given that picture:**
1. **Item-ID memorisation.** A content tower gives two ads with similar content similar vectors whatever their actual appeal. Item-level popularity, conversion and "who-also-acted" signals are only available from IDs: co-visitation, ID embeddings, per-item event-type ratios.
2. **This surface and target.** Avito's model mixes targets and surfaces. Our target is one specific feed action, and later history occurrences of positives are 47% et=1 [L] `eda/collab_stats/out_05b.log`.
3. **Explore-only targets.** Feed candidates are never in the feed's own history, while history sequences are full of repeats (consecutive same-item gaps have median 1 [L] `eda/history_leak/01_semantics.log`). A model trained to predict the next *new* item matches our task. A generic next-event model mostly learns to repeat.
4. **Event-type and source semantics** at fine grain (10 types, 27 sources).

Quick check to run (5 minutes): regress per-item log history count (from all histories) on the 128-d item embedding. If R² is low, the item embedding carries little popularity or ID information and the ID-based features (#1, #2) have more room. Also check whether exact duplicate embeddings occur across different item_ids (reposts): that would suggest a pure content tower.

---

## 2. Evidence: what history signals gained, where

### 2.1 Sequence models as features or pretraining on top of a strong production ranker (most relevant)

- **ARGUS (Yandex Music, KDD 2026)** [V] https://arxiv.org/html/2507.15994v1. Pretraining loss = next-item prediction (logQ-corrected sampled softmax, mixed negatives, cosine/temperature, Q from count-min sketch) **+ feedback prediction** (given history, context and item, predict the feedback factors with a CE per factor). Events are (context, item, feedback) triplets merged into one token. As a *feature* added to the production ranker (pair-accuracy uplift): 3.2M params +1.35%, 18.9M +1.82%, 126M +2.32%, 1B +2.66%. As a *standalone* ranker it is always worse than production (-4.8 to -7.5%). **Pretraining matters**: no pretraining with 1 year of fine-tuning gives +1.17% as a feature, pretraining with 1 year of fine-tuning gives +2.32%. Longer context helps: 512 -> +1.01%, 2048 -> +2.32%, 8192 -> +2.77%. Online: +2.26% listening time, +6.37% like likelihood. Takeaway for us: **expect the sequence model to be weaker than our ranker alone but additive as a feature. Pretrain, then fine-tune on the ranking data.**
- **Yandex ads, "Long-History User Transformers for Real-Time Ad Ranking"** [V abstract] https://arxiv.org/abs/2607.14331: offline encoder pretrained autoregressively (feedback prediction + next item), then fine-tuned for CTR. A/B: +2.77% on the primary ranking metric in search ads, +2.1% on the ad network.
- **PinnerFormer (Pinterest)** [V] https://arxiv.org/pdf/2205.04507 (Tables 3, 4, 7). Objectives, Recall@10: next action 0.186; SASRec softmax 0.198; all-action (28-day window) 0.224; **dense all-action (28d) 0.229**, which means predicting a random positive from a future window at random positions with causal masking. Negatives with sample-probability (logQ) correction: random only 0.139; in-batch only 0.167; **mixed 0.229**. Without correction, in-batch only gives 0.071. Feature ablation: removing the PinSage (pretrained content) embedding input drops Recall@10 from 0.229 to 0.142, so the pretrained embedding is the backbone and the sequence model adds on top. Used **as a feature in the Homefeed ranker: +7.5% repins** online.
- **TransAct (Pinterest, KDD 2023)** [V] https://arxiv.org/pdf/2306.00248. Realtime action sequence (PinSage embedding + action-type embedding), **early fusion by concatenating the candidate embedding to every sequence token** (better than appending the candidate as an extra token), 2-layer transformer, output = first K=10 columns + max-pool. HIT@3/repin vs a model with no sequence: average pooling +0.21%, LSTM -0.75%, plain transformer +1.56%, full TransAct +9.40%. In the hybrid ablation, removing TransAct gives -8.59% repin and removing the batch user embedding (PinnerFormer) gives -2.46%. Realtime sequence and a batch user embedding complement each other, which is our situation: provided user embedding plus our own history encoder.
- **Airbnb (2025 blog)** [V] https://airbnb.tech/ai-ml/personalizing-airbnb-search-by-learning-from-the-guest-journey/. Causal transformer over long-term events (80) and short-term views (200, 21 days). Offline NDCG: long-term alone +0.44%, adding the short-term view sequence +1.48%, setwise ranker a further +2.3%. The *recent view sequence* gave most of the gain.
- **Layer 6, RecSys Challenge 2022 (2nd place)** [V] https://www.cs.toronto.edu/~mvolkovs/Recsys_2022.pdf. Transformer MRR 0.2121 vs XGBoost 0.1993, VAE 0.2026, DAE 0.2035. A self-supervised masked-session objective on top of the supervised candidate objective, "PIP" (the candidate replaces the mask token, i.e. early fusion), raises MRR from 0.2030 (PIP only) to 0.2121 (MSM+PIP, alpha=0.1). This is direct evidence that an **auxiliary self-supervised loss on histories improves a supervised ranker** and allows about 10x larger models without overfitting.
- **RecSys Challenge 2025 (universal behavioural profiles)**:
  - 4th place (Yandex "ambitious") [V] https://arxiv.org/html/2508.06970v1: causal transformer over merged event types (20 layers, d=1024, 252M params, max 256 events). Multi-task objectives: next event type, time delta, next URL, and next item with sampled softmax + logQ and mixed 8,192 in-batch + 8,192 uniform negatives. Removing the sequence encoder hurts most (category 0.8147 -> 0.8027). The **TwHIN graph embedding adds nothing** once the sequence encoder is present (0.8146 vs 0.8147). Scaling the encoder from 3M to 252M improved category 0.7957 -> 0.8048.
  - 2nd place [V] https://arxiv.org/html/2508.07748v1: single-component validation scores (sum of three tasks): GRU autoencoder 2.3953 > transformer next-product 2.2641 > handcrafted 2.2384 > **iALS 2.1951 / 2.1777 > LightFM 2.0216**.
- **Avito's own production** (section 1): sequence-transformer user embedding as a ranker feature gave +2.3% target actions and was top-1 SHAP [V].

### 2.2 Co-visitation and item2vec as reranker features (competition evidence)

- **OTTO 3rd place (Chris Deotte / Giba / Theo / Benny)** [V] https://github.com/TheoViel/kaggle_otto_rs and https://github.com/bschifferer/Kaggle-Otto-Comp. Features = **item-item scores between each candidate and each session item** (w2v similarity, MF similarity, Chris's co-visitation coefficients). Each session item gets a **weight encoding position, timestamp and event type**, and scores are aggregated over the session. About 744 features, XGBoost; CV 0.5917 -> LB 0.6028. Benny's part also trained a **GRU and a Transformer to score session x candidate pairs as features** (`01d_FE_GRU`, `01e_FE_Transformer`). The READMEs give no per-component ablation.
- **Chris Deotte's public co-visitation definitions** (as reproduced in) [V] https://gladdduck.github.io/2024/03/14/%E5%AE%9E%E4%B9%A0-Kaggle-OTTO%E6%AF%94%E8%B5%9B%E5%9B%9E%E9%A1%BE/index.html:
  - "carts-orders" matrix: last 30 events per session, pairs within 24 h, **weight by event type of the second item {click 1, cart 6, order 3}**.
  - "buy2buy": carts and orders only, 14 days, weight 1.
  - "clicks": 24 h window, time weight 1 + 3·(ts - t_min)/(t_max - t_min), so recent pairs count up to 4x.
  - Keep the top 15-40 neighbours per item.
  - Computing the matrices on train + test sessions is my recollection [K], not verified.
- **OTTO silver (Fedorov)** [V] https://github.com/Fedorov-Artem/kaggle_OTTO/blob/main/README.md: **directional** matrices with *different windows per relation*. click2click < 5 min (later events weighted more); click2buy = click earlier, buy later, < 10 h (smaller gap weighted more); click2buy-short < 2 h; buy2buy < 5 days; exact-next-click. Two w2v models (short window for clicks, long for buys). Reranking added only +1-2 recall points over the candidate stage.
- **OTTO (nicolaivicol)** [V] https://github.com/nicolaivicol/otto-recommender: LightGBM gain importance is dominated by co-visitation counts (click_to_click_perc_pop 0.365; click_to_cart_or_buy_count 0.288). w2v distance ranks low (0.026), and cosine with the session embedding is 0.024.
- **KDD Cup 2023, NVIDIA (1st on all tasks)** [V] https://developer.nvidia.com/blog/?p=69059. Co-visitation with the top 100 per item. **Item embeddings initialised from a multilingual text model (stsb-xlm-r) and trained contrastively**: the last 3 consecutive history items predict the next item, with 4,096 random negatives and a cosine loss. Ranker features include cosine(last item, candidate) from LLM text embeddings. This is the "contrastive fine-tuning on top of pretrained content embeddings" pattern, used for retrieval and features. No ablation numbers.
- **KDD Cup 2023, 10th place, Task 2** [V] https://ar5iv.labs.arxiv.org/html/2310.07281. Directional co-visitation variants: B anywhere after A, B immediately after A, B immediately before A; counts per session over the last 10 items give 60 features. Plus item2vec and USE text similarity. "item2vec_similarity also has high importance". No ablation.
- **Session kNN vs neural (Ludewig et al.)** [V abstract] https://arxiv.org/abs/1910.12781: across many datasets, simple nearest-neighbour session methods were usually preferable to neural session-based recommenders.
- **item2vec hyperparameters (Caselles-Dupré et al., RecSys 2018)** [V] https://arxiv.org/pdf/1804.04212, Table 1. Out-of-the-box SGNS (L=5, n=5 epochs, t=1e-3, alpha=0.75) vs fully optimised:
  - 30Music HR@10: 11.16 -> 23.75 (L=3, n=110, t=1e-5, **alpha=-0.5**)
  - Click-stream HR@10: 3.07 -> 26.26 (L=7, n=150, t=1e-5, alpha=-1)
  - E-commerce HR@10: 22.21 -> 26.34 (alpha=1)

  A negative alpha samples *unpopular* items as negatives more often.

### 2.3 Combining frozen content embeddings with IDs

- **UniSRec (KDD 2022)** [V] https://arxiv.org/pdf/2206.05941, Table 3. Frozen BERT text embeddings, parametric whitening and an MoE adaptor, contrastive pretraining. Adding ID embeddings ("transductive", item vector = text + ID) beats text-only on most datasets: Office R@10 0.1013 -> 0.1280; Online Retail 0.1449 -> 0.1537; Arts 0.1066 -> 0.1239.
- **Semantic IDs (YouTube)** [V abstract] https://arxiv.org/abs/2306.08121: replacing item IDs with content embeddings *loses memorisation*. Semantic IDs (RQ-VAE codes of frozen content embeddings, sub-pieces hashed) keep overall quality and improve new and long-tail items.
- **ID vs modality (Yuan et al., SIGIR 2023)** [V abstract] https://arxiv.org/abs/2303.13835: modality-based recommenders match ID ones *only with expensive end-to-end training*. With frozen features, IDs win.
- **LLM-embedding initialisation (Harte et al., RecSys 2023)** [V abstract] https://arxiv.org/abs/2309.09261: initialising BERT4Rec with LLM embeddings gives +15-20% NDCG.
- **PinnerFormer**: removing the pretrained content embedding input collapses Recall@10 from 0.229 to 0.142 [V].

Conclusion: the best configuration is **frozen provided embedding (generalisation, cold items) + learnable ID residual (memorisation)**. Neither alone is optimal.

### 2.4 Losses for large catalogues

- **SASRec vs BERT4Rec**: with the same (full-softmax CE) loss, SASRec "significantly outperforms" BERT4Rec in quality and speed [V abstract] https://arxiv.org/abs/2309.07602. gSASRec (more negatives + gBCE with calibration parameter t ∈ [0,1]; beta = alpha·(t(1-1/alpha)+1/alpha)) beats BERT4Rec (+9.47% NDCG on ML-1M, -73% training time) and scales past 1M items [V] https://arxiv.org/pdf/2308.07192.
- **eSASRec (RecSys 2025)** [V] https://arxiv.org/html/2508.06450v1: SASRec shifted-sequence objective + LiGR layers (pre-norm, gated residual h + F(h)·σ(hW), SwiGLU FFN×4) + sampled softmax with 256 uniform negatives; d=256, 2-4 blocks, dropout 0.1-0.3, lr 1e-3, batch 128.
  - Sampled softmax vs BCE: ML-20M NDCG@10 0.1192 -> 0.1527.
  - LiGR vs plain layers: Kion 0.1497 -> 0.1657.
  - BERT4Rec+LiGR+SS is worse than eSASRec (ML-20M 0.1364 vs 0.1563).
  - In-batch negatives lower NDCG but raise coverage.
- **Scalable Cross-Entropy** [V abstract] https://arxiv.org/abs/2409.18721: approximates full CE over MIPS-selected hard negatives with up to 100x lower peak memory. Useful if we want closer-to-full-softmax training over 40M items.
- **Mixed negatives + logQ** (PinnerFormer Table 4, RecSys 2025 4th, ARGUS, Avito) [V]: the consistent industrial recipe.

### 2.5 Target attention over history (relevant to how the NN ranker uses history)

- **SIM (Alibaba)** [V] https://arxiv.org/pdf/2006.05639, Tables 2-4. Filtering long histories to candidate-relevant behaviours before attention is a big win over average pooling (Taobao AUC 0.9281 -> 0.9330 with category "hard search", or 0.9357 with embedding "soft search", first stage only). Adding attention on the filtered set: soft 0.9416, hard 0.9332. Industrial: soft 0.6625 vs hard 0.6604, and hard was deployed for cost. Our history is short (≤256), so full target attention is affordable. The lesson is that the **max/top-k similarity filter is the strongest part**, which our max-sim feature already captures [L].

---

## 3. Our data facts that shape the recipes [L]

- 2.455M train users (one sequence each), 448.7M tokens; val 13.7K sequences / 2.46M tokens; test 27.3K / 4.89M. Vocabulary of items with count ≥ 2: 40.4M, covering 85.2% of tokens. 91.4M unique items in histories. `eda/collab_stats/out_04c.log`, `eda/history_leak/05_covis.log`
- Candidate-item coverage by train histories: et=1 55.9% of val candidate rows; any et ~58%. Coverage **decays with time gap**: 53.3% when the history cutoff is 5 days before val, 25.7% at 16 days. Test public 56.4% vs private 52.7%. In-split histories cover 8.5% (val) and 14.0% (test) of candidate rows. `eda/shift_overlap/out/s03*.log`, `s05.log`. So ID-based signals exist for only about half the candidates, and fewer for late private days. **Transductive use of val/test histories and a content fallback both matter.**
- Existing co-visitation (val, single feature, random tie-break): cv_invd20 0.3266 (11.3% of rows nonzero; positive rate 3.0% when nonzero vs 1.4% when zero). Blended with f1: rank(f1)+rank(cv_invd20) 0.3432 vs f1 0.3279 (+0.0153, paired SE 0.0013). Larger windows with inverse-distance weighting beat window-5 variants, and forward-only was worse than symmetric. `eda/history_leak/05_covis.log`
- Positives reappear in **later** snapshots of the same user 54.4% of the time (et=1: 47.2%), vs 0.29% for negatives. This is the source of transductive leakage for train-feed features (section 5). `eda/collab_stats/out_05b.log`
- Val/test have about 1.00 feeds per user. 92 users are in both val and test. 17% of train users have more than one feed. `eda/shift_overlap/out/s01.log`
- The current item2vec run uses W=5, 5 negatives, alpha=0.75, no sub-sampling, 4 epochs, plus 15% supervised (history item -> clicked train candidate) pairs. `eda/collab_stats/04d_item2vec_train.py`. That is the "out-of-the-box" regime that Caselles-Dupré et al. show is far from optimal.

---

## 4. Recipes

### R1 (high). Hybrid sequence model "HSM": pretrain on all histories, export features, fine-tune into the ranker

**Corpus** (leak-safe, see section 5):
- one sequence per train user, truncated at that user's first train feed: use the earliest snapshot, or equivalently drop events after the first feed's timestamp when merging snapshots;
- plus every val and test snapshot.

Reverse the order to oldest -> newest (histories are newest-first). Optionally upsample val+test sequences 3-5x to adapt to the test period; validate on val.

**Token for event t** (sum, d = 384 or 512):
- `W_c · norm(dequant(item_emb128))`: frozen provided embedding, learnable projection;
- `+ E_id[item]` for items with count ≥ 2 (40.4M rows; a dim-128 table projected to d, or semantic-ID sub-tokens if memory matters). Apply **ID-dropout p≈0.3-0.5** in training so the content path works alone for cold items (DropoutNet idea, [K]);
- `+ E_et[event_type] + E_src[event_source] + E_cat[category] + E_pos[recency index]`;
- items without a provided embedding get a learned "missing" vector.

A "context token" with the feed-level provided user embedding can be prepended at fine-tune time.

**Encoder**: causal transformer, 4-8 layers, LiGR or pre-LN blocks, 6-8 heads, max length 256, dropout 0.1. ARGUS and RecSys-2025 scaling says go as large as the data supports. With about 465M tokens, 20-100M non-embedding params is sensible; tune on val.

**Item (output) tower**: `v(i) = MLP(norm(emb128(i))) + P·E_id[i]`, with ID-dropout. Score `s = cos(h, v)/τ`, learnable τ clipped to [0.01, 100] as in ARGUS [V].

**Objectives** (sum):
1. **Dense all-action next-new-item.** From each position t, predict one target drawn from F_t = {x_j : t < j ≤ t+K, item_j not in x_1..x_t, et_j in T}. K is in events (no timestamps), with K ∈ {8, 32, 128}. Here T is the event types that match feed positives: start with et=1 only, then try all types with weights proportional to the "later appearance" rates in `out_05b`. "New item only" mirrors the fact that candidates are never in their own history. PinnerFormer's dense all-action beat next-action by +23% Recall@10 [V].
2. **Feedback prediction** (ARGUS-FP [V]): from `MLP([h_{t-1}, v(item_t)])` predict et_t (10-way CE) and optionally src_t. This learns "given this item, which action" (view vs contact-like), the closest proxy we have for the feed target.
3. Loss for (1): **sampled softmax with mixed negatives and logQ correction**. Use 8-16K in-batch targets (Q from exact item frequencies in the corpus) plus 8-16K uniform negatives drawn from the union of history items and feed candidates. Mask accidental hits. Mixed+logQ beat in-batch-only by 37% and random-only by 65% in PinnerFormer [V]. Optional upgrade: SCE-style MIPS hard negatives [V abstract].

**Compute** (estimate): about 1e17 FLOPs per epoch for d=512, 6 layers, dense supervision and 16K shared negatives. Even at 20-30% MFU on 4x GPU this is minutes of math per epoch, so the data pipeline (int8 embedding gathers) is the bottleneck. Budget 10-30 min per epoch and 10-30 epochs. Memory: ID table 40.4M×128 fp32 = 20.7 GB, plus Adam states; fits on one GPU, or use row-wise Adagrad / SparseAdam. Keep the frozen 128-d int8 embeddings on GPU: 91M×128 B ≈ 11.7 GB.

**Exports as LGB features** (val/test feeds, and train feeds if the corpus is leak-safe):
- `hsm_score(c) = cos(h_last, v(c))`, plus its within-feed rank, z-score and gap to the feed max;
- the content-only score (ID part zeroed) and the ID-only score;
- the feedback head's P(et ∈ target types | h_last, c);
- max/mean cosine between v(c) and the last-k history v(·): learned-space similarity, which differs from the provided-space max-sim already in the model.

Scoring 4.2M val+test candidate rows takes seconds.

**Fine-tune into the ranker** (the ARGUS lesson: pretrain, then fine-tune on impressions). Initialise the NN ranker's history encoder (`nn/model.py`, the L_h layers) and item projection from HSM. Train listwise softmax over each feed's candidates on train feeds, keeping the existing train-only position tower (PAL), then on val folds. Use early fusion (TransAct concat, or Layer 6's "candidate as token" PIP) [V]. Keep a small MSM/next-item auxiliary loss (alpha≈0.1, as in Layer 6 [V]) during fine-tuning to regularise.

**What to watch**: as a standalone ranker HSM will probably lose to the LGB, which is expected per ARGUS. Judge it only by *incremental* gain inside the LGB/stack.

### R2 (high). Co-visitation family, extended (CPU, exact, query-driven)

Our 05_covis already builds postings (item -> (user, position)) and computes exact scores only for the pairs that occur (candidate × feed-history item), so no top-K truncation is needed. That is the right design: val+test is about 4.2M candidate rows × ≤256 history items. Extend it as follows.

1. **Event-type-pair-specific matrices**, not a single matrix. Count pairs (a in history at position p_a, b at position p_b, p_b later than p_a in the same corpus sequence) separately by (et_a group, et_b group). Groups: {1}, {2,3,4} and {5..10}; check which types behave like contact/favourite with the et-later-appearance table. OTTO's best features were exactly the type-directional ones (click2buy, buy2buy, carts-orders with weights {1, 6, 3}) [V]. Let LGB learn the weights rather than hand-setting them.
2. **Direction and distance**: forward-only (b after a) at windows {5, 20, 64, all} with 1/d weighting. Restrict b to the *first* occurrence of b in that sequence (a "new item" event), mirroring the candidate definition. The current symmetric window is good, so keep it as well.
3. **Normalisations** for each raw count C(a,b): P(b|a) = C/n_a; cosine C/sqrt(n_a·n_b); lift C·N/(n_a·n_b); and a BM25-style saturated count. Raw counts are popularity-confounded, and the candidate-popularity component is already a separate feature. Swing-style down-weighting of heavy co-users is an option [K].
4. **Aggregation over the feed history**: sum, max, mean of top-3, and weighted sum with history-position half-life h ∈ {3, 10, 50} and per-history-event-type weights (OTTO 3rd: weight per session item encoding position, time and type [V]). Also the count of distinct history items with C>0, and within-feed rank/z of each aggregate (the feed is the ranking unit).
5. **Transductive corpus**: one global matrix from {leak-safe train sequences} ∪ {val} ∪ {test} snapshots, used for both val and test features, so they are consistent. Ablate it against a train-only matrix on val to measure the transductive gain. Expect it to matter more for private test days because of coverage decay [L].
6. **Category-level fallback**: the same statistics on (category_a -> category_b) and (item_a -> category_b) for candidates with no ID co-occurrence (89% of rows have cv_invd20 = 0 [L]).

### R3 (medium). Tuned item2vec

On the existing GPU trainer:
- set the negative-sampling exponent to alpha ∈ {-0.5, 0, 0.75};
- add frequent-token sub-sampling with t=1e-5;
- try window L ∈ {3, 7, 15};
- increase epochs (8, 16, 32 at our corpus size; the paper's 90-150 were for much smaller corpora) [V Caselles-Dupré].

Train on the same leak-safe transductive corpus. Keep the supervised (history item -> clicked candidate) pairs only if they come from *train* feeds and the features are used for val/test. Features: max/mean cosine to the last-k items, per event type; in·out ("next-item") score with a recency-weighted profile. Expect little beyond R1+R2 (OTTO w2v importance was low [V nicolaivicol]). Its main value is diversity for the stack.

### R4 (medium, folded into R1). Content-to-CF adapter for cold candidates

If R1 is delayed: train `g(emb128)` (2-layer MLP, d=128-256) contrastively so that g(e_a)·g(e_b) is high for (history item -> later new item) pairs. Use in-batch + 4-16K random negatives, as in the NVIDIA KDD Cup 23 item model (4,096 random negatives, text-initialised [V]). Then use max/mean similarity of g(candidate) to g(history) as features. This gives CF-shaped similarity for the ~45% of candidates with no ID history. It is the same as R1's content path, so do not build both.

### R5 (low). iALS / EASE / SLIM / LightGCN / TwHIN

- **iALS** (implicit-gpu, user × item with event-type confidence weights): about 15-30 min on one GPU for 2.5M users × 40M items. Score = u·v(c) for users with history, plus item-item cosine to history. Expected small: iALS was the weakest family in RecSys 2025 [V], and MF similarity was only one of hundreds of OTTO features [V]. Only worth it as a cheap stack diversifier.
- **EASE**: closed-form inversion of the item Gram matrix is O(n³) [K]. That is infeasible at 40M items, and even top-200K items covers few long-tail ads. SANSA (sparse approximate EASE) scales further [K], but co-visitation normalisations deliver the same item-item signal more cheaply. Skip.
- **LightGCN / TwHIN / random walks**: TwHIN added nothing next to a sequence encoder [V RecSys 2025 4th]. RP3beta (popularity-discounted 2-hop walk) is a cheap item-item alternative if a graph signal is wanted [K; used as a candidate generator by a RecSys 2022 team [S]]. Skip LightGCN.

---

## 5. Leakage and transductive protocol (necessary for R1-R3)

1. **Train-feed features leak through later snapshots of the same user.** For train feed f of user u at time t with positive X, u's later snapshots contain X after u's earlier items in 54% of cases [L]. Any co-visitation, item2vec or sequence model that includes those later snapshots scores X highly *because* of the label, and val/test do not have this (about 1 feed per user). Two clean options:
   - **(a) Earliest-snapshot corpus**: per train user, keep only events before their first train feed. This loses a little data (17% of users have multiple feeds) and needs no per-feed bookkeeping. Recommended for R1.
   - **(b) Leave-user-out subtraction** for count features: C_global(a,b) - C_u(a,b), with u's own pairs subtracted when scoring u's feeds. This is exact and cheap for co-visitation.

   For neural models where subtraction is impossible, use (a) or 2-4 user-fold models.
2. **Val/test features** may use everything unlabelled: train histories, val and test snapshots, and train-feed labels (e.g., the supervised item2vec pairs). Exception: test snapshots of the 92 users who are also in val may contain their val positives. Drop those users' test snapshots when computing val features (negligible, but free).
3. **Scale consistency.** One global matrix serves both val and test, so co-visitation features are on the same scale. In-split *count* features (e.g., test-history popularity) differ in scale between splits: val is 3 days / 14.4K feeds, test 4 days / 28.6K feeds. Normalise per 10K feeds or per day, as already done in `s03.log`, or use within-feed ranks.
4. **Hyperparameter selection** for pretraining should use val NDCG of the *downstream* LGB with the new feature (paired SE over feeds; about 0.001 resolution with 14.4K feeds), not next-item recall alone.

---

## 6. Suggested execution order and checks

1. (2-4 h, CPU) R2 extensions on the existing 05_covis code: type-pair, direction, normalisation and aggregation variants. Add them to the LGB in groups, keeping a group only if the paired delta is greater than 2 SE.
2. (½ day) Leak-safe earliest-snapshot corpus (+ val/test snapshots); rebuild the R2 matrices on it if LGB training on train feeds is planned.
3. (1-2 days, 4x GPU) R1 pretraining. Sweep {K window, T event set, ID-dropout, d/L}. Export features, measure LGB delta, then initialise and fine-tune the NN ranker.
4. (2 h, 1 GPU) R3 tuned item2vec, for stack diversity.
5. Only if time remains: iALS as a stack diversifier.

Sanity checks:
- report the delta separately on candidates *with* vs *without* ID coverage; R1's content path should help the latter;
- compare val vs test feature distributions (adversarial AUC) for every new feature; transductive counts are the classic drift source;
- confirm that the gain survives in the full LGB, not only in a rank blend with f1.

---

## 7. Pitfalls specific to this topic

- Training sequence models on raw next-event targets learns *repeat* behaviour (consecutive duplicates are common). Our candidates are never repeats, so filter targets to new items.
- BCE with 1 negative (vanilla SASRec) is badly overconfident and underperforms. Use sampled softmax with logQ, or gBCE [V].
- In-batch negatives alone without logQ are a popularity-biased disaster (PinnerFormer R@10 0.071) [V].
- Without ID-dropout, an ID-heavy model gives cold candidates (about half) near-random scores, and LGB then splits on "has ID".
- Full-list NDCG with pessimistic ties: co-visitation features are zero for about 89% of rows. Never use a sparse feature alone as a ranking score. Always pair it with a dense score (f1 / HSM) or use it inside the LGB.

---

## 8. Sources read this session

- Avito Habr, two-tower transformer: https://habr.com/ru/articles/1004694 [V]
- Avito interview (SASRec, attractiveness signals): https://kod.ru/monetizatsiya-rekomendatsii-i-llm-v-avito [V]
- ARGUS: https://arxiv.org/html/2507.15994v1 [V]
- Yandex long-history ads transformer: https://arxiv.org/abs/2607.14331 [V abstract]
- PinnerFormer: https://arxiv.org/pdf/2205.04507 [V full text]
- TransAct: https://arxiv.org/pdf/2306.00248 [V full text]
- SIM: https://arxiv.org/pdf/2006.05639 [V full text]
- Airbnb guest-journey sequence model: https://airbnb.tech/ai-ml/personalizing-airbnb-search-by-learning-from-the-guest-journey/ [V]
- Layer 6 RecSys Challenge 2022: https://www.cs.toronto.edu/~mvolkovs/Recsys_2022.pdf [V full text]
- RecSys Challenge 2025 4th: https://arxiv.org/html/2508.06970v1 [V]; 2nd: https://arxiv.org/html/2508.07748v1 [V]
- OTTO 3rd: https://github.com/TheoViel/kaggle_otto_rs, https://github.com/bschifferer/Kaggle-Otto-Comp, https://github.com/cdeotte/Kaggle-OTTO-Comp [V READMEs]
- OTTO silver: https://github.com/Fedorov-Artem/kaggle_OTTO/blob/main/README.md [V]; https://github.com/nicolaivicol/otto-recommender [V]
- Chris's co-visitation definitions (reproduction): https://gladdduck.github.io/2024/03/14/%E5%AE%9E%E4%B9%A0-Kaggle-OTTO%E6%AF%94%E8%B5%9B%E5%9B%9E%E9%A1%BE/index.html [V]
- NVIDIA KDD Cup 23 tutorial: https://developer.nvidia.com/blog/?p=69059 [V]; KDD Cup 23 workshop: https://kddcup23.github.io [V]; 10th place: https://ar5iv.labs.arxiv.org/html/2310.07281 [V]
- eSASRec: https://arxiv.org/html/2508.06450v1 [V]; gSASRec: https://arxiv.org/pdf/2308.07192 [V]; Klenitskiy & Vasilev: https://arxiv.org/abs/2309.07602 [V abstract]; SCE: https://arxiv.org/abs/2409.18721 [V abstract]
- UniSRec: https://arxiv.org/pdf/2206.05941 [V full text]; Semantic IDs: https://arxiv.org/abs/2306.08121 [V abstract]; ID vs modality: https://arxiv.org/abs/2303.13835 [V abstract]; Harte et al.: https://arxiv.org/abs/2309.09261 [V abstract]
- item2vec hyperparameters: https://arxiv.org/pdf/1804.04212 [V full text]; session kNN vs neural: https://arxiv.org/abs/1910.12781 [V abstract]
- Not accessible (blocked): Kaggle discussion writeups (OTTO 1st/2nd, Benny's 3rd-place post, H&M 1st), OpenReview PDFs of the KDD Cup 23 NVIDIA / YAMALEX papers, Airbnb KDD'18 paper (Medium 403). Claims that depend on them are tagged [K]/[S] or omitted.
