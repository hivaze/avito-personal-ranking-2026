# Industry and academic neural ranking, 2024-2026: what could add information or diversity to our ranker

Research only: web reading plus a read-only look at `nn/model.py` and `nn/train.py`. No GPU work, no training, and no existing file was edited. Date 2026-11-12.

Tags:
- **[E]** = a number measured in the cited paper or post.
- **[E-abs]** = the claim appears only in an abstract or a secondary summary; the full tables were not read.
- **[O]** = my opinion or inference.
- **[L]** = a fact from this repo (file cited).

**Status vs COVERAGE_MAP:**
- **NEW** = never run here.
- **PARTIAL** = a related variant was run.
- **CLOSED-variant** = the same mechanism already failed here.

Before searching, I read the existing reports so as not to repeat them: deep_neural_rankers, deep_seq_pretrain_cf, deep_position_bias_random, nnx_research_recsys_arch, r6_external, r7_research, COVERAGE_MAP, pretrain_report, ROUND5/r5_agedrop, feat_clu_, feat_life_.

Already covered there, so not repeated below: HSTU / ULTRA-HSTU, TWIN / TWIN-V2, TransAct (v1), LiGR, LiRank, RankMixer, OneTrans, LONGER, MTGR, OneRec, Wukong, SIM/ETA/SDIM, ARGUS, PinnerFormer, gSASRec / eSASRec, AutoDebias / KDCRec / CausE / KuaiRand, the ULTR literature, DropoutNet, semantic IDs, embedding collapse.

---

## 0. Bottom line

Industry gains in 2024-2026 come mostly from **new information channels**:
- pretrained ID representations that are *transferred into* the ranker;
- longer sequences;
- popularity signals;
- exploiting recent data.

They do not come from block changes. Feed SR (LinkedIn, 2026) reports that depth and width "added capacity without new information" [E]. This matches our null results on block and capacity changes.

Most of our cold-start interventions are already closed here (age-conditioned dropout, gates, routers, cluster target encoding). The literature mostly confirms that they trade warm accuracy for cold.

Five things look NEW and worth GPU time, in order of value per cost:

| # | Idea | Why | Cost |
|---|---|---|---|
| 1 | **Causal-mask history encoder** as a new family (optionally with a next-action auxiliary loss) | Two industrial ablations say a causal mask beats a fully visible one on history (GenRank: AUC −0.0015 when fully visible; PinFM: −1.4% when bidirectional) [E]. Our history layer is fully visible [L]. It is a different inductive bias, so mainly a diversity lever. | 1-line mask change; 2-seed screen, then a family |
| 2 | **Recency-weighted training / recency-focused cooldown** | Test is 1-10 days after train end with strong staleness. Weighting losses by day index gave +0.0006 to +0.0055 AUC [E] (TIF). LinkedIn warm-starts daily on new data only [E]. `nn/train.py` has no date weighting [L]. | sampler weights; 2-seed screen read on Tue vs Thu and stale-val |
| 3 | **"Example age" feed input** (YouTube): train with feed age = days before train end; set it to 0 (or negative) at inference | Removes the model's bias toward the *average* of the training window [E, qualitative]. It gives a principled extrapolation knob for the later private days. | one scalar input; 2-seed screen |
| 4 | **Transfer *frozen* pretrained item-ID embeddings into the ranker** (GPSD "sparse transfer + sparse freeze") | Leak-safe corpus; PinFM's candidate-ID randomization for cold robustness; Netflix-style kNN-imputed IDs for cold candidates. On our data the pretrain's incremental signal is its ID part [L], but today the ranker sees it only as 2 scalar scores. GPSD: freezing transferred sparse params lifts AUC 0.6375 → 0.6834 where plain transfer gives ~0 [E]. PinFM: a frozen module ≈ no gain, a fine-tuned one +3.76% [E]. | one pretraining run + an input path; medium-high engineering |
| 5 | **Temporal-decay diagnostic of the m02 under-transfer** (CPU) | YouTube's argument (a window-trained model predicts the window average) gives a concrete, testable hypothesis for why m02 gains shrink from val to public. If the m02 delta decays with days since train end (Tue > Thu > weekend), private (latest) gets even less. | CPU only |

Expected sizes [O]:
- ideas 1-3: each +0.0000 to +0.0010 single-model, with most of the value as family diversity;
- idea 4: +0.001 to +0.003 if the ID channel survives the cold share; it is also the only idea here that brings genuinely new information.

Do NOT port, from this angle:
- HSTU blocks or SiLU attention: Feed SR measured both as worse than standard softmax/pre-LN [E].
- Contrastive SSL objectives: augmentation alone matches contrastive learning [E-abs], and our augmentation arm is already running.
- Text-LLM recommenders (HLLM, 360Brew): no text exists, and LLM backbones are closed here.

---

## 1. Ideas (ranked by value per cost)

### I1. Causal-mask history encoder (+ optional next-action auxiliary loss): NEW
- **Mechanism.**
  - In the history self-attention layer (`hist_enc`, today fully visible over [user token, history] [L: nn/model.py, hist_enc / kmask]), each history event attends only to itself and *older* events. History is stored newest-first, so this is a lower-triangular mask in reverse order. The user token still attends to everything.
  - Candidates' cross-attention is unchanged.
  - Optional (TransAct V2 NAL): add an auxiliary loss on history positions. From the state at event t (causal), predict the next newer feed-originated open event (event_type 1 & source 1, the footprint of 85% of feed positives [L: pretrain_report]). Use a sampled softmax over in-batch history items, weight w = 0.01. The causal mask is what makes that loss leak-free inside the ranker.
- **Evidence.**
  - [E] GenRank (Xiaohongshu, arXiv 2505.04180): a fully visible mask at historical positions dropped AUC by more than 0.0015, and the drop grew with model size. Computing loss on historical positions cost more than 0.0100 AUC in *their* setup (pointwise CTR labels).
  - [E] PinFM (Pinterest, arXiv 2507.12704): making the module bidirectional cost 1.4% in evaluation metrics.
  - [E] TransAct V2 (Pinterest, arXiv 2506.02267): NAL with impression negatives at w 0.01 gave +1.10% repin HIT@3 offline. Weights 1e-4 to 0.1 gave +0.54% to +1.10%. Online, NAL alone was repin +0.27% (n.s.).
- **Relevance.**
  - A pure inductive-bias change. Feature-wise it is the same model, so expect a small single-model delta [O].
  - Its main value is a *differently-biased trunk* for the member-equal pool. The pool's lever is "same-strength different-recipe families" (frontier law R² 0.95 [L: COVERAGE_MAP]).
  - The NAL head adds 400M+ history events of extra supervision. GenRank's warning: in the ranker, keep the aux weight small; a full-weight history loss hurt them.
- **Cost.** Low. Mask: about 5 lines. NAL: about 40 lines (needs history content embeddings as targets). One 2-epoch paired screen (2 seeds), then a 6-seed family if it passes gate (a). Judge at ensemble level (r6 protocol).

### I2. Recency-weighted training / recency-focused cooldown: NEW
- **Mechanism.** Two variants:
  - (a) TIF: weight each train feed's loss by w(t) = t/N (t = day index of the feed, N = days) or by the exponential form, normalised to mean 1.
  - (b) Cheaper and closer to our recipe: keep the plateau uniform, but during the WSD cooldown draw production batches from the last K days of train (Sep 17-19) with probability q (e.g. 0.5), alongside the m02 exploration replay. The cooldown sets the final function [L: r6_recap "cooldown improves the mean function"], so this targets exactly the part that matters.
- **Evidence.**
  - [E] TIF (arXiv 2311.16878): linear day weighting over 10 Avazu days: DNN AUC 0.7462 → 0.7479, DeepFM +0.0041, FiBiNET +0.0055. On a 7-day industrial CVR set: +0.0001 to +0.0007. Anti-TIF (old-weighted) was worse.
  - [E] LinkedIn Feed SR (arXiv 2602.12354): daily warm start computes the loss only on newly observed interactions.
  - [E] MEDA (Kuaishou, arXiv 2305.19531): results comparable to 1 month of data with 2 weeks.
- **Relevance.**
  - Strong staleness: cold share 0.29 val / 0.37 public / 0.41 private, and the co-vis censoring mismatch [L: FEATURES_REVIEW FR-4].
  - Variant (b) is a time-local analogue of m02. The m02 public under-transfer is a warning that cooldown data choices can over-fit to val [O]. So read this arm on **Thu vs Tue**: the later day should gain *more* if the mechanism is recency. Also read it on stale-val and the late-holdout slice (Sep 17-19), and treat a gain that only appears on the late-holdout slice as selection bias, because (b) trains on those days. Exclude holdout feeds from the recency pool.
- **Cost.** Low. A sampler weight vector (feed day is in feeds.parquet) and one flag. 2-seed cooldown-branch screen from existing plateau snapshots (~30 GPU-min per member [L: EXPERIMENT_REGISTRY]).

### I3. "Example age" input (YouTube DNN): NEW
- **Mechanism.**
  - A feed-level scalar a = (train_end − feed_time) in days: 13 → 0 over train.
  - It is fed to the head (and/or added as a learned embedding to candidate tokens).
  - At inference set a = 0, or slightly negative to extrapolate toward the later private days.
- **Evidence.** [E, qualitative] Covington et al., RecSys 2016:
  - "we feed the age of the training example as a feature during training. At serving time, this feature is set to zero (or slightly negative) to reflect that the model is making predictions at the very end of the training window."
  - Their Fig. 4: without it the model predicts "approximately the average likelihood over the training window."
- **Relevance.**
  - Our train-row features have time-dependent semantics. Coverage and censoring of co-vis/popularity change across the 14 days [L: FR-4, review_shift]. Without a time input, the network learns their average meaning.
  - Example age lets it condition, and a = 0 matches the end-of-window regime nearest to val/test.
  - It also gives a hedge knob for private (a < 0), which could be chosen *a priori* (e.g. a = −2 for private-like dates) rather than tuned on val [O].
- **Risk.**
  - It is distinct from the closed "posting time / diurnal phase" features (feat_life_ R4, −0.0017), which were per-item and calendar features, not a training-time index.
  - It could still behave like the closed "later-history" signal: check the has-later probe and the presub gate.
- **Cost.** Low. One dense input; 2-seed screen.

### I4. Frozen pretrained item-ID embeddings transferred into the ranker: NEW (PARTIAL: pre2s scores exist; ID residual from scratch CLOSED)
- **Mechanism.** Combines GPSD, PinFM and Netflix.
  1. Pretrain the existing ARGUS-style encoder (`pretrain/`) once on a corpus that is leak-free for train rows: each non-zero user's **earliest** train snapshot, plus user-0 feeds, plus (for the val/test version) the earliest val/test snapshot.
     - A user's train feed then never sees a same-user snapshot taken after it (the rule in CURRENT_STATUS §3).
     - The current windowed encoders give 3 different ID spaces and cannot be shared across splits [L: pretrain_report §8].
     - For symmetric horizons, build one table without test snapshots (val side) and one with them (test side), as the pretrain report already does.
  2. Import its ID table (items with count ≥ 3) **frozen** (GPSD "ST&SF") as an extra 64-128-d input:
     - to candidate tokens (projected, added to `c_lin(content)`);
     - to history tokens;
     - as a second key space in `HistRetrieval`: top-m by ID-space cosine, in addition to content cosine.
  3. Cold handling:
     - Items without an ID get a learned "no-ID" vector (PinFM GSLT-style token).
     - Netflix's trick: a similarity-weighted mean of the ID vectors of the item's top-k content neighbours among ID-bearing items, flagged as imputed.
     - Plus PinFM's candidate-ID randomization: replace the candidate's ID input with the no-ID token for ~10% of training candidates.
- **Evidence.**
  - [E] GPSD (Alibaba/AliExpress, arXiv 2506.03699) Table 3, L4H128A4:
    - no transfer 0.6375;
    - full transfer 0.6376;
    - sparse transfer 0.6395;
    - **full transfer + sparse freeze 0.6838; sparse transfer + sparse freeze 0.6834.**
    - Discriminative models show one-epoch and within-epoch overfitting driven by sparse params; freezing removes it.
    - Online 30 days: GMV +7.03%, CTR +3.78%.
  - [E] PinFM Table 5: frozen module Save +0.10%, fine-tuned Save +3.76% (Hide −2.77%).
  - [E] PinFM Table 2 (fresh items):
    - no cold-start handling: 28-day items −4.40%, 7-day −23.98%;
    - + candidate-ID randomization (10%): +1.25% / −4.38%;
    - + age-dependent dropout: +10.71% / +8.16%;
    - + content embedding + learnable token: +17.72% / +12.01%, with overall +3.76%.
  - [E-abs] Netflix FM (Mar 2025, via secondary summary): ID and metadata embeddings are mixed by an attention layer weighted by entity age, and new titles are initialised from metadata-similarity-weighted combinations.
  - [L] In our data the ID part is what carries the pretrain's incremental signal: removing it costs 0.004-0.005 zero-shot; content-only scores add ~0 over the NN [pretrain_report TL;DR].
- **Relevance.**
  - The only idea here that adds genuinely new information: a 40M-item collaborative space learned from ~460M events, versus ~250M labelled candidate rows.
  - The ranker currently receives it only as 2 clipped scalar scores (`pre2_fe_s`, `pre2_new_s`) [L: FEATURES_REVIEW FR-1].
  - The cold share caps the upside: ID coverage is 46% val / 41% test [L: pretrain_report §8].
  - Frozen IDs + 10 epochs avoids the ID overfitting that likely killed the closed hashed residual, which was trained from scratch and limited to ≤1 epoch.
- **Risks.**
  - (i) Other users' later snapshots are still in the corpus. This is the same LUO-type transductive information as `it_hist_*_luo` (SHIFT-1 accepted), but run the has-later probe and the stale-val replay.
  - (ii) Train and val/test ID coverage differ (62% vs 41-46%). Candidate-ID randomization at a rate matched to the test no-ID share is the PinFM answer; stale-val is the arbiter.
- **Cost.** Medium-high. One pretraining run (~25-45 min on one GPU [L]), the ID-table plumbing (an int32 index per candidate and history event already exists: `cand_item_id`, `hist_item_id` [L]), and a 2-seed 2-epoch screen, then a family.
- **Heavier sibling (PinFM "early fusion, fine-tuned"):** run the whole pretrained 4-layer encoder as a branch whose output feeds the ranker, fine-tuned at lr/10 with an MSE alignment to the ranker's logit. Only after I4 shows the ID channel carries signal.

### I5. Temporal-decay diagnostic for the m02 under-transfer (CPU): NEW analysis
- **Hypothesis** [O], from YouTube's "bias toward the past" argument and TIF:
  - The m02 cooldown replays train-period exploration feeds and pulls the final function toward window-specific item/popularity patterns.
  - These decay with time, which predicts that the m02-minus-non-m02 per-feed delta **shrinks** from val Tue (Sep 22) → Thu (Sep 24) → public (Sep 26-27) → private (Sep 28-29), and is concentrated on warm/popular candidates.
  - The status log already shows Tue +0.0023 vs Thu +0.0008 for PF7x vs v12 [L: CURRENT_STATUS §1]. The m02-specific split is the needed number.
- **Test** (paired per feed, same members except the m02 cooldown):
  - Regress the delta on day (Tue/Thu), the feed's cold share, and candidate age (f4 knots).
  - Then do the stale-val replay with m02 vs m0 members.
- **Decision.**
  - If the delta decays with day at z ≥ 2, extrapolate to private (2 more days) and weight m02 members down in the private-oriented pick (or use I2(b) instead of m02).
  - If it does not decay, the public gap is more likely noise (public SE ~0.0012).
- The running shift diagnostic (workflow wpa63h0vs) may already include this; coordinate before duplicating.
- **Cost.** CPU minutes, from existing val preds.

### I6. Content-anchored bounded ID delta, reset per epoch (Let It Go + MEDA): PARTIAL
- **Mechanism.** If the item-ID channel is reopened without pretraining:
  - Learn a per-item delta d_i in the 128-d content space, so that e_i = c_i + d_i, with ‖d_i‖ ≤ δ_max (0.3-0.5 of the unit-norm content vector).
  - Use exact IDs with ≥ k train occurrences, zero-init.
  - Cold items get d = 0, i.e. exactly the content model.
  - MEDA: re-initialise the delta table at each epoch start (or freeze it in the cooldown), so 10 epochs do not memorise IDs.
- **Evidence.**
  - [E] "Let It Go? Not Quite" (arXiv 2507.19473, SASRec), Amazon-M2 HR@10:
    - cold: content init 0.435 → bounded delta 0.509 (plain ID SASRec 0.000);
    - total 0.607 → 0.609.
    - Small or neutral warm effects on Beauty / Zvuk.
  - [E] MEDA Table 1, second-epoch AUC with embedding reset vs one-epoch baseline:
    - DIN Taobao 0.8804 → 0.9265;
    - DIN Books 0.8477 → 0.8617;
    - online at Kuaishou AUC +0.14%, revenue +4.6%.
- **Relevance.** The closed N4 residual (hashed, zero-init, ID-dropout 0.5, ≤1 epoch [L: STRATEGY F4, nn/model.py id_buckets]) lacked both the bounded-norm anchoring and multi-epoch protection. Expected gain small [O]. Value: a different information channel and so a diversity family. Do I4 first; I6 is the no-pretraining fallback.
- **Cost.** Medium.

### I7. PinFM-style cold bundle as a hedge family, not a replacement: CLOSED-variant (agedrop)
- **What differs from our closed agedrop.**
  - Our `prv_rl13` blanked all 8 ID/popularity inputs with p up to 0.96 for items 0-6 h old [L: r5_agedrop]. PinFM uses p = 0.7 (< 7 days) / 0.5 (7-28 days), only on the pretrained-module path, plus a content path and a learnable token.
  - Feed SR (LinkedIn) found candidate popularity features worth +2.5% AUC [E], which suggests **not** dropping popularity counts (`it_hist_*`, `seq_cv_pop`).
- **Proposal** [O]: one family with p halved, applied only to `pre2_*_s` and co-vis similarity inputs, judged *as a family inside the pool*.
  - Agedrop's stale-val was +0.0009 and young-positive feeds +0.0247 [L], so a minority of such members can hedge the colder private set.
  - It must pass gate (a) on holdout; agedrop was −0.0019 at 2 epochs.
- **Cost.** Low (flag exists). Low prior.

### I8. PDA-style inference-time popularity adjustment: CLOSED-variant (lagged/causal popularity closed)
- **Mechanism.** Train with popularity as is. At test, replace the item's popularity feature by an extrapolated m̃ = m_T + α(m_T − m_{T−1}) from the last two train stages.
- **Evidence.** [E] PDA (SIGIR 2021, arXiv 2105.06067): on Kwai, Recall@20 0.0143 (PD) → 0.0288 (PDA with predicted popularity). MF backbone; large relative gains on sparse public sets.
- **Relevance.** Low [O]: our popularity inputs are already LUO counts, and the fine-tune re-weights them on val; lagged/causal variants failed adversarial review. Listed only so the idea is not re-proposed without a new mechanism.

### I9. Temporal DRO over item groups (TDRO): NEW, low priority
- **Mechanism.**
  - Group items (k-means on content; we have clusters [L: feat_clu_]) and split train into day-periods.
  - Up-weight groups by (1−λ)·loss + λ·⟨g_group, Σ_e β_e g^e⟩ with β_e = exp(p·e), i.e. gradient alignment with the most recent periods.
- **Evidence.** [E] TDRO (AAAI 2024, arXiv 2312.09901): cold Recall@20 CLCRec on Amazon 0.0135 → 0.0166 (+23%), Micro-video 0.0623 → 0.0761. Two-tower cold-start backbones, not rankers.
- **Relevance.** Low-medium [O]. It is the principled form of I2, but "segment weights" closed here and per-group gradients are costly. Try only if I2 shows a recency signal.

### I10. Invariance penalty across time environments (V-REx / InvCF-style): NEW, low priority
- **Mechanism.** Add a penalty on the variance of per-day (or per item-age bucket) listwise losses within a batch, to remove features whose relation to the label drifts.
- **Evidence.** [E-abs] InvCF (WWW 2023, arXiv 2302.05328) beats debiasing baselines under temporal and OOD popularity shift on Douban/Meituan. No ranking-model evidence.
- **Relevance.** Low [O]. Our past cold/warm interventions traded rather than generalised. Cheap to try as a loss flag (~20 lines).

### I11. Relative position encoding (RoPE) in the history encoder: NEW, low priority
- **Mechanism.** Replace the learned recency-index embedding (0..255) added to history tokens [L: nn/model.py `self.rec`] with RoPE in `hist_enc` attention. Keep the retrieval module's log2 recency buckets.
- **Evidence.** [E] Feed SR: learned absolute vs RoPE: −0.19% long-dwell AUC, −0.16% contributions AUC. No positions: −0.91%. RoPE also stabilised score levels.
- **Relevance.** Small [O]. GRU ≈ transformer here, so order information is already captured. Mainly a cheap diversity arm, and it combines naturally with I1.

---

## 2. Confirmatory and negative evidence (what not to spend GPU on)

| Topic | Evidence | Implication for us |
|---|---|---|
| HSTU pointwise SiLU attention / HSTU blocks | [E] Feed SR: SiLU −0.09% / −0.05%, ReLU −0.25%, sigmoid −0.15% vs softmax; HSTU blocks −0.23% / −0.28%; Pre-LN essential (AUC 0.5 without it) | Do not port HSTU blocks. If intensity matters, keep the count channel (nnx P1) as an *extra* head, not a softmax replacement. |
| Capacity scaling | [E] Feed SR: depth and width "added capacity without new information"; sequence length gave the most consistent gains. [E] Zhang et al. (RecSys'24): epochs 2-5 clear gains, 6-12 diminishing, 13-30 ~0. | Matches our closed d256 / deep / heads results and the 10-epoch plateau. Our sequence length is capped at 256 by the data [L], so the strongest industrial lever is unavailable. |
| Long-sequence retrieval (TWIN, LONGER, TransAct V2 lifelong) | [E] TransAct V2 lifelong adds ~4.4 HIT@3 points, but only with O(10⁴)-token histories | n/a: H ≤ 256; extending from earlier snapshots was tried and failed [L: err_segments §6]. |
| Contrastive SSL for sequences | [E-abs] Zhou et al. (WWW'24): sequence augmentation alone matches or beats several CL methods | Skip CL objectives; the augmentation arm (`--aug_cand / --aug_hist`) is already running. |
| Frozen pretrained module as features | [E] PinFM: frozen +0.10% Save vs fine-tuned +3.76% | Explains why `pre2s` scalar scores gave only +0.0004 to +0.0009. The value is in transfer (I4), not in more scalar features. |
| Text-LLM recommenders (HLLM, 360Brew) and LLM ranker backbones | No text; our LLM-backbone test was closed at every size [L: llm_large] | Skip. |
| Generative-ranking "paradigm" | [E] GenRank: the gain comes from the architecture, not the generative training paradigm | No reason to switch to autoregressive training. |

---

## 3. General intuitions

1. **Information beats architecture.** Every large 2024-2026 industrial gain I read adds an information channel: pretrained IDs, longer sequences, popularity, profile embeddings for sparse users (Feed SR +2% AUC for members with < 10 actions). Block-level ablations move AUC by about 0.1-0.3%. Our null results on blocks are the expected outcome, not bad luck.
2. **Transfer, then freeze sparse and train dense.** IDs learned discriminatively over-fit across epochs (GPSD, MEDA). IDs learned generatively and then frozen transfer well.
3. **A cold-start fix must give cold items a substitute path, not only remove the warm path.** PinFM's dropout alone raised fresh items but needed the content embedding plus a learnable token to get the full gain without hurting the overall metric. Our agedrop removed the warm path without adding a cold-specific substitute, and so traded warm for cold.
4. **A model trained on a window predicts the window average.** This holds unless it is told *when* each example happened (example age) or recent data is weighted up (TIF, warm-start). Test is later than val and private later than public. Any recipe whose val gain is time-local should decay in the same order, so test for that directly.
5. **Causal masks over history** are a cheap inductive bias that two independent industrial teams found better than fully visible attention.
6. **Use exploration data as an evaluation and anchoring set, not as extra training mass.** Industry uses randomized traffic mainly for unbiased evaluation and candidate generation (Tubi 2025; Google item-centric exploration 2025). This is consistent with our observation that m02 replay helps val but under-transfers.

---

## 4. Sources

Read this session. **[full]** = full text or HTML read; **[abs]** = abstract or secondary summary only.

1. PinFM: Foundation Model for User Activity Sequences, Pinterest, 2025. https://arxiv.org/html/2507.12704v3 [full]
2. GPSD: Generative Pretraining for Scalable Discriminative Recommendation, Alibaba/AliExpress, 2025. https://arxiv.org/html/2506.03699v2 [full]
3. An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking (Feed SR), LinkedIn, Feb 2026. https://arxiv.org/html/2602.12354 [full]
4. Towards Large-scale Generative Ranking (GenRank), Xiaohongshu, 2025. https://www.alphaxiv.org/abs/2505.04180.md [full summary page]
5. TransAct V2: Lifelong User Action Sequence Modeling, Pinterest, 2025. https://arxiv.org/html/2506.02267 [full]
6. Foundation Model for Personalized Recommendation, Netflix Tech Blog, Mar 2025. The original (netflixtechblog.com) returned 403, so it was read via https://www.zenml.io/llmops-database/foundation-model-for-personalized-recommendation-at-scale [abs, secondary]
7. Deep Neural Networks for YouTube Recommendations (example age), Google, RecSys 2016. https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf [full, pdftotext]
8. Temporal Importance Factor for Loss Functions for CTR Prediction, 2023. https://ar5iv.labs.arxiv.org/html/2311.16878 [full]
9. Multi-Epoch Learning for Deep CTR Models (MEDA), Kuaishou, 2023. https://ar5iv.labs.arxiv.org/html/2305.19531 [full]; extension https://arxiv.org/pdf/2407.01607 [abs]
10. Let It Go? Not Quite: Item Cold Start in Sequential Recommendations with Content-Based Initialization, 2025. https://arxiv.org/html/2507.19473v1 [full]
11. Temporally and Distributionally Robust Optimization for Cold-Start Recommendation (TDRO), AAAI 2024. https://arxiv.org/html/2312.09901v3 [full]; code https://github.com/Linxyhaha/TDRO
12. Causal Intervention for Leveraging Popularity Bias in Recommendation (PD/PDA), SIGIR 2021. https://ar5iv.labs.arxiv.org/html/2105.06067 [full]
13. Scaling Law of Large Sequential Recommendation Models, RecSys 2024. https://ar5iv.labs.arxiv.org/html/2311.11351 [full]
14. Is Contrastive Learning Necessary? Data Augmentation vs CL in Sequential Recommendation, WWW 2024. https://arxiv.org/abs/2403.11136 [abs]
15. Invariant Collaborative Filtering to Popularity Distribution Shift (InvCF), WWW 2023. https://arxiv.org/abs/2302.05328 [abs]
16. Aligning Distillation for Cold-start Item Recommendation (ALDI), SIGIR 2023. https://web.comp.polyu.edu.hk/xiaohuang/docs/2023SIGIR_Feiran.pdf [abs]. Teacher(warm) → student(content) alignment; low relevance because our cold routers and gates are closed.
17. Item-centric Exploration for Cold Start Problem, Google, RecSys 2025. https://arxiv.org/html/2507.09423v1 [abs]
18. Where to Explore: Reach and Cost-Aware Unbiased Data Collection, Tubi, 2025. https://arxiv.org/html/2512.14733v1 [abs]
19. Generalized user representations for large-scale recommendations, Spotify Research, Sep 2025, via a mirror: https://scour.ing/p/https://research.atspotify.com/2025/9/generalized-user-representations-for-large-scale-recommendations [abs]. User cold start only; not applicable.
20. Avito and Yandex Habr posts re-checked for new Avito ranking material (https://habr.com/ru/articles/932204, /857068, /898416). Nothing new beyond research_competition_avito.md.

Local files used: nn/model.py, nn/train.py, data/competition/readme.md, reports/{COVERAGE_MAP, CURRENT_STATUS, pretrain_report, ROUND5, r5_agedrop, FEATURES_REVIEW, feat_clu_, feat_life_, r6_tta, nnx_research_recsys_arch, deep_*}.md.
