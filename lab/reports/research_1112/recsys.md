# research_1112 / recsys: recommender and ranking competitions 2023-2026, checked against what we already did

Date: 2026-11-12. Research only: no GPU, no training, no project files edited. The only CPU work was parsing a public dump of Kaggle write-ups.

Tags:
- **[V]** = I read the primary page this session. **[S]** = search snippet only.
- **[E]** = a number measured in the source. **[Q]** = a qualitative claim by the source. **[O]** = my opinion or inference.
- Kaggle discussion pages do not render for the fetch tool. I read the Kaggle write-ups through the public HF dump `t2ance/mle-skills-kaggle-writeups-clean` (815 write-ups, March 2026 snapshot), then cite their original kaggle.com URLs.

Prior coverage that I skipped on purpose (already read in earlier reports):
- RecSys Challenge 2020-2024
- OTTO, H&M, Amazon KDD Cup 2023
- VK RecSys 2024, Avito ML Cups 2024-2026
- The RecSys Challenge 2025 4th / 2nd place papers, ARGUS, Baidu-ULTR, KDD Cup 2020 debiasing

So this note focuses on sources that are **new to the repo**: KDD Cup 2026 (Tencent UniRec), the Tencent ad challenge 2025, the WWW'25 multimodal CTR challenge, and 2024-25 Kaggle write-ups whose training lessons transfer.

---------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **The 2026 competition ablations repeat our own record: the blocks are cheap and small, the inputs are large.** In KDD Cup 2026 UniRec (35M samples, sequences plus 142 feature columns, AUC):
   - Every architecture mechanism is worth 0.0002-0.0011 AUC [E].
   - The dense-feature representation is worth 0.0095, the Muon-family optimizer 0.0028, absolute target-time context 0.0020, and missing/padding disambiguation 0.0013 [E].
   - Nothing there argues for another block-form screen.
2. **The one genuinely new and cheap lever is mutual learning between peers (online co-distillation). We have never tried it.**
   - Evidence: Jigsaw ACRC 2025 6th place, where both the peers and their ensemble improved [E]. KDD Cup 2026 runner-up UniDot, where two-path mutual learning gave +0.135% test AUC [E, confounded by averaging the two paths].
   - It fits our cooldown harness: co-cool two saved plateau snapshots with a KL term, then judge at **ensemble level**. A sign-flipped "anti-distillation" arm, which Google Ads used to push ensemble components apart [E], tests the opposite direction in the same harness.
3. **Two-stage "big data, then target data, with soft labels" is how winners handled small target-distribution subsets** (AES 2.0 2nd/3rd place). We already do the two stages (train, then val fine-tune). We have never used **soft targets inside the fine-tune**: a mix of the val labels and the train-only member's per-feed softmax. FT-only and cheap.
4. **On the m02 public under-transfer:** two 2024-25 Kaggle cases show a target-focused change that was flat on public and clearly positive on private. In AES 2.0 the target-focused second stage gave public +0.000, private +0.010 [E]. CIBMTR 3rd place found the public split sat at the ~75th percentile of their emulated split distribution [E].
   - Our public is weekend-only, while private (Mon/Tue) is weekday like val (Tue/Thu). A weekend-specific shortfall need not reach private.
   - Proposed CPU check: the m02 delta on weekend vs weekday slices of the random-like train holdout (§3 R5).
5. **Missing/padding indicators measured −0.0013 AUC when removed** (STAR, KDD Cup 2026) [E]. That is the second-largest effect in that paper. Our `has:` mechanism exists but covers only some families (FEATURES_REVIEW FR-8 is parked as "~0"). External evidence says to promote it to a real arm.
6. **Do not transfer:**
   - Time-of-day / weekend context. It was STAR's largest effect, but our feat_cal_ already measured that it hurts cross-day transfer.
   - Multi-teacher → single-student distillation (KD is closed for an ensemble-final pipeline).
   - Semantic IDs (closed). Larger ID tables (UniDot / Z Lab saw val up, test down: the same lesson as our item-ID residual).
   - Anything text- or LLM-based (Eedi, Learning Equality, CIKM 2025, RecSys Challenge 2026).

---------------------------------------------------------------------------------------------------

## 1. Sources (≈25 distinct; primary unless marked)

| # | competition (year) | place / author | what I used | URL | tag |
|---|---|---|---|---|---|
| 1 | KDD Cup 2026 Tencent UniRec, industrial track (pCVR, AUC) | 1st, QueryFormer (Zhou, Zeng) | abstract + full ablation summary | https://arxiv.org/abs/2609.16548 ; https://www.besthub.dev/articles/queryformer-winning-kdd-cup-2026-with-cross-attention-query-generation-e6f0c3619418 | [V] |
| 2 | KDD Cup 2026 UniRec, industrial | runner-up, UniDot (Lin et al.) | ablations, mutual learning, FM highway | https://arxiv.org/html/2608.16797 | [V] |
| 3 | KDD Cup 2026 UniRec, academic | STAR (Xu et al.; repo slug suggests academic rank 2 in round 1) | ablation table | https://arxiv.org/html/2608.12986 | [V] |
| 4 | KDD Cup 2026 UniRec | 10th, Z Lab "Dense Feature Representation over Sequence Modeling" | ablation table, optimizer | https://arxiv.org/html/2609.19787 | [V] |
| 5 | KDD Cup 2026 UniRec | 9th, FA-RankMixer | dual-stream bilinear fusion, positive-weighted BCE | https://arxiv.org/pdf/2607.15590 | [S] |
| 6 | KDD Cup 2026 workshop recap | organizers (Tencent) | winners, CRAFT, CoSFormer | https://www.besthub.dev/articles/real-world-industry-problems-take-the-kdd-stage-full-highlights-of-the-taac-2026-jeju-island-workshop-8fc3ecfe1db7 | [V] |
| 7 | HyFormer (the UniRec baseline backbone; ByteDance 2026) | - | design summary | https://arxiv.org/abs/2601.12681 | [S] |
| 8 | Tencent Ad Algorithm Challenge 2025 (all-modality generative rec) | organizers' paper: 1st-3rd and innovation award | Section 7 "Challenge Summary" | https://arxiv.org/html/2604.04976v1 | [V] |
| 9 | WWW'25 EReL@MIR Multimodal CTR Challenge (MicroLens) | 1st (Xu, Zhao, Hu, Song) | ablation table | https://arxiv.org/html/2505.03543v1 | [V] |
| 10 | Kaggle FlightRank 2025 (Aeroclub; per-search ranking) | winner, via MLContests | GBDT findings only (write-up not reachable) | https://mlcontests.com/state-of-machine-learning-competitions-2025 | [V] |
| 11 | Kaggle Learning Agency Lab AES 2.0 (2024) | 2nd place | two-stage training, threshold lesson | https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/writeups/yao-2nd-place-solution | [V] |
| 12 | same | 3rd place | two-stage with cross-domain selection plus soft labels; public/private table | https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/writeups/ho-3rd-place-solution | [V] |
| 13 | Kaggle Jigsaw Agile Community Rules (2025) | 6th, "Online Distillation via Deep Mutual Learning" | DML tables | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/6th-place-solution | [V] |
| 14 | Kaggle LEAP ClimSim (2024) | 1st place | confidence head, multiple representations, dropout at scale | https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/writeups/greysnow-no-leaky-1st-place-solution-for-the-leap- | [V] |
| 15 | Kaggle CIBMTR equity post-HCT (2025) | 3rd place | public/private emulation, NN vs GBDT table | https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/survivalhorror-3rd-place-solution | [V] |
| 16 | Kaggle ISIC 2024 | 7th, "A good CV is all you need" | CV that simulates unseen hospitals | https://www.kaggle.com/competitions/isic-2024-challenge/writeups/former-zlp-dss-7th-place-solution-a-good-cv-is-all | [V] |
| 17 | Kaggle Home Credit Stability (2024) | public 8th / private 253 | diversity from independent data processors; what failed | https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/evgeniia-grigoreva-public-8th-private-253-solution | [V] |
| 18 | Kaggle Eedi misconceptions (2024) | 2nd place (1st via Eedi blog [S]) | retrieve, then listwise LLM rerank | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution | [V] |
| 19 | Kaggle Learning Equality curriculum rec (2023) | 1st place | retrieval plus LGBM with group-normalised scores | https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/writeups/epicurus-1st-place-solution | [V] |
| 20 | Google Ads "On the Factory Floor" (industrial; co-distillation, anti-distillation) | Anil et al. 2022 | §4.1.2, §5 numbers | https://arxiv.org/abs/2209.05310 | [V] |
| 21 | WSDM Cup 2023 ULTR (Baidu) | 2nd THUIR, 3rd Yu et al. | abstracts | https://arxiv.org/abs/2304.12650 ; https://arxiv.org/abs/2302.07530 | [V] |
| 22 | RecSys Challenge 2025 (Synerise) | 1st "rec2" has no write-up; PoliMi 5th overall | leaderboard facts | https://recsys.synerise.com/summary ; https://github.com/recsyspolimi/recsys-challenge-2025-synerise | [S] |
| 23 | RecSys Challenge 2026 (Music-CRS) | conversational LLM task | not transferable | https://huggingface.co/simransu/recsys2026-minimaestro-lambdarank | [S] |
| 24 | Muon+ (used as "MuonPlus" by the KDD Cup 2026 winner) | Zhang et al. 2026 | one extra normalisation after orthogonalisation | https://arxiv.org/abs/2602.21545 | [S] |
| 25 | Searched, nothing found | Ozon E-CUP 2025 recsys track, Yandex Cup 2025 (no recsys track), VK LSVD 2025 winners, Tianchi 2025 recsys, FlightRank write-ups (Kaggle JS-only) | - | - | [S] |

---------------------------------------------------------------------------------------------------

## 2. What the new competitions say (facts first)

### 2.1 KDD Cup 2026, Tencent UniRec (pCVR; four behaviour-domain sequences of length 256 plus 142 columns; AUC; 5,746 teams)

This is the closest 2026 analogue to our model: a single network that fuses sequence modelling with feature interaction. Every top paper starts from HyFormer: global query tokens that decode the sequences, plus token mixing.

**QueryFormer, 1st place** (test AUC 0.83254) [V]
- Ablations, validation AUC [E]:
  - Full model: 0.84606.
  - Cross-attention query generation replaced by an MLP: −0.00038 (the largest single drop).
  - No user-item interaction: −0.00029.
  - No candidate-relevant retrieval: −0.00023.
  - No self-attention: −0.00020.
  - All four removed: −0.00110.
- Number of independent tokenisation "views" H [E]:
  - H = 1 / 2 / 4 / 8: 0.84540 / 0.84575 / 0.84606 / 0.84615.
  - Latency goes from 31 to 59 µs.
- Scaling [E]:
  - Width 272 → 544: +0.00025.
  - Blocks 1 → 3: +0.00018.
  - Epochs 4 → 11: +0.00010 AUC, but worse logloss.
- Optimizers: MuonPlus for dense parameters, Adagrad for sparse embeddings, plus EMA.
- Auxiliary MAE on log conversion time: −0.00016 when removed [E].
- Final submission: full-data training (+0.0001) and multi-teacher → single-student distillation (+0.0007) [E].

**UniDot, runner-up** [V]
- Feature interaction and sequence retrieval share one dot-product token space.
- Ablations, tiny mode, AUC [E]:
  - Full model: 0.83657.
  - Without the **FM Highway**: −0.127%. This is a set of per-layer dot products (item state · retrieved history vector per domain, user-item cross dots, a Gram matrix of item vs attention-aggregated tokens) that bypass the residual stack and are concatenated and LayerNorm'd straight into the classifier.
  - Without only the cross-bus dots: −0.087%.
  - Without the token-mixing bus: −0.067%.
  - Without sequence cross-attention: −0.053%.
- **Multi-path mutual learning:**
  - Setup: two paths share the sparse table. Each path also gets λ/(N−1) · MSE(p, stop-grad(p_other)), with λ = 20, switched on after epoch 1. Logits are averaged at inference.
  - Test AUC: single path 0.82993 → two paths 0.83128 (+0.135%). d = 128 with two paths: 0.83196 [E].
  - Caveat [O]: the two-path number includes the ensembling effect of averaging the paths. It is not a pure "mutual learning" effect.

**STAR, academic track** [V]. Single-run test AUC; full model 0.836546 [E]:
- Without absolute target-time features (time of day, weekend): **−0.00204**.
- Without padding / missing / real-zero disambiguation: **−0.00133**.
- Without symmetric user-item InfoNCE (τ 0.07, weight decayed 0.1 → 0.01): −0.00034.
- Without the DIN query decoder: −0.00019.
- Without sequence hash / frequency remapping of rare IDs: −0.00012.
- Without user-item pair-product tokens: −0.00003. The paper itself calls this one marginal.

**Z Lab, 10th place** [V]. Single seed; full model 0.827932 [E]:
- Removing the **dense-feature representation stack**: **−0.00950**. The stack is subgroup tokens (Linear + LN + SiLU), log1p on heavy-tailed raw statistics, and two heads per group averaged (mid-scale plus raw-statistic).
- Muon-family optimizer (AMUSE) → AdamW: **−0.00282**. Plain Muon instead of AMUSE: +0.00027, which is within seed noise.
- AdagradAR (an anti-memorisation regulariser on rare sparse IDs): −0.00089.
- Dense-weight EMA: −0.00035.
- Every sequence-modelling component: ≤ 0.0005.
- **"Enlarging high-cardinality ID embeddings raised validation AUC while lowering test AUC"** [Q].

**Organizers' recap** [V]: no single winning design. The prizes went to the following [Q]:
- query generation (QueryFormer);
- per-layer information preservation (CRAFT, academic champion);
- adding new information rather than parameters when scaling (CoSFormer, scaling award).

### 2.2 Tencent Ad Algorithm Challenge 2025 (all-modality generative recommendation; 1M / 10M users; exposure / click / conversion) [V organizers' summary, no per-component numbers]

- **1st place:**
  - Per-position **action conditioning** (gated fusion, FiLM, attention biasing).
  - Hierarchical time features (absolute, gaps, sessions) with multi-frequency Fourier encodings.
  - RQ-k-means semantic IDs over multimodal embeddings, with a "random-k" regulariser for long-tail items.
  - InfoNCE with large negative banks; Muon + AdamW hybrid [Q].
- **2nd place:**
  - **Pre-train on exposure events, fine-tune on click / conversion** with InfoNCE.
  - Next-behaviour-type conditioning [Q].
- **3rd place:**
  - Scaled in-batch negatives up to 380K with "substantial gains".
  - "Performance was often driven more by scale than by intricate model design" [Q].
- Model ensembling was prohibited by the rules.

### 2.3 WWW'25 Multimodal CTR Challenge (MicroLens-1M; frozen multimodal item embeddings) [V], 1st place

- Model:
  - The target item is concatenated to every history item before a 2-layer Transformer.
  - The last k = 16 outputs give short-term interest; max-pool gives long-term interest.
  - A DCNv2 cross network runs in parallel with an MLP.
- AUC [E]:
  - Frozen embeddings hurt DIN (0.9326 → 0.8577) but helped the transformer model (0.9729 → 0.9776).
  - Removing DCNv2: 0.9776 → 0.9632.
- The authors' explanation: the frozen embeddings are not aligned with the CTR task and need a learned interaction path [Q].
- For us: retr16 is already candidate-conditioned early fusion over retrieved history items. Our DCNv2 lives in the DIN family (F11), which is weak. Little new here.

### 2.4 Other recsys / ranking contests: little or nothing to transfer

- **RecSys Challenge 2025.**
  - The 1st-place team "rec2" has no write-up.
  - 2nd (GRU autoencoder) and 4th (Yandex) are already in deep_seq_pretrain_cf / deep_ru.
  - The task (universal user embeddings) is close to what our pre2 encoders already do.
- **RecSys Challenge 2026** is LLM conversational music recommendation; the reported rerankers are LambdaRank. Not transferable.
- **FlightRank 2025** (Kaggle, per-search option ranking): "XGBoost consistently performed best, with LightGBM adding useful diversity; CatBoost underperformed", with Polars count features [V via MLContests]. GBDT is closed for us.
- **WSDM Cup 2023 ULTR** (train on biased clicks, evaluate on expert labels):
  - 3rd place reports that neural models trained directly on top-10 clicks "performed poorly", and that ULTR-learned propensities did not decrease with position, so they recalibrated them from click ratios.
  - They also adjusted labels, added true negatives, reweighted samples and ensembled two training schemes [V abstract].
  - This matches the Baidu-ULTR "loss and features beat debiasing" finding already in deep_position_bias_random.
- **Eedi 2024 / Learning Equality 2023** (retrieve → rerank on text): listwise LLM rerankers with sliding windows; LGBM over match scores normalised by group max / min. Text-only; we already have within-feed z and rank.

### 2.5 2024-25 Kaggle contests whose *training* lessons transfer

- **AES 2.0 (target-like subset inside train; hidden test resembles the subset).**
  - 2nd place: "train on the big set first, load the weights, train on the target-like set with the same loss". Threshold search on the target-like subset only "gives much better CV, but LB drops a lot" [E/Q].
  - 3rd place, recipe:
    - Stage 1 on A + B, checkpoints selected on B.
    - Stage 2 on B only, selected on A "to avoid overfitting to B".
    - Soft labels from the stage-1 OOF in stage 2 "improved CV and public".
  - 3rd place, results [E]:

    | | A | B | CV | Public | Private |
    |---|---|---|---|---|---|
    | GroupKFold | 0.826 | 0.792 | 0.818 | 0.826 | 0.828 |
    | Focus on B | 0.832 | 0.815 | 0.828 | **0.826** | **0.838** |

- **Jigsaw ACRC 2025, 6th place: deep mutual learning between two LLM peers** (CE + KL to the peer's soft predictions) [E]:

  | | Public | Private |
  |---|---|---|
  | Qwen3-14B, independent | 0.925 | 0.921 |
  | Qwen3-14B, DML | 0.929 | 0.925 |
  | Qwen2.5-14B, independent | 0.923 | 0.919 |
  | Qwen2.5-14B, DML | 0.929 | 0.924 |
  | Ensemble of the DML peers | **0.932** | **0.9271** |
  | Ensemble of the independent models | 0.929 | 0.925 |

- **LEAP 2024, 1st place.**
  - An auxiliary **confidence head** that predicts each target's own loss lifted the best single model's private score from 0.78631 to 0.78869 [E].
  - Dropout became unnecessary once he ensembled at 40-80M samples [Q].
  - He fed "multiple data representations" of the same inputs [Q].
- **CIBMTR 2025, 3rd place.**
  - 4-fold CV × 20-100 split seeds to emulate public and private. "The public fold was approximately at the 75-quantile" of the emulated distribution.
  - They submitted a second, private-oriented pick with a lower LB that won on private [E/Q].
  - Their NN was the best single model on private (0.69694); a GBDT + NN blend with a logreg stack reached 0.69937 [E].
- **ISIC 2024, 7th place.** A CV that **simulates the test novelty** (train on 5 hospitals, predict all 7) correlated very well with LB, where the patient-group CVs did not [Q].
- **Home Credit Stability 2024 (public 8th / private 253).**
  - Ensemble diversity came from "repeating the feature engineering from scratch … six different data processors" [Q].
  - What failed: recency weighting of samples, excluding features by adversarial validation, and separate models by history availability [Q].
- **Google Ads "On the Factory Floor"** (industrial, not a contest).
  - Online two-pass distillation gave +0.41% accuracy at no student cost.
  - 3-component ensembles cut prediction difference (PD) from 17% to 12%.
  - **Anti-distillation** cut PD to 10% "with no accuracy loss".
  - Distillation and rank losses were ramped in by curriculum for stability [E].

---------------------------------------------------------------------------------------------------

## 3. Ideas, ranked by expected value per GPU-hour for *our* state

Status words follow COVERAGE_MAP. "Cost" uses the R6_PLAN figures: a re-cooled member takes ~10 min alone on a GPU (45-60 min under shared load), and a 10-epoch pair takes ~4.5 GPU-h.

### R1. Mutual learning in the cooldown ("co-cooling"), with an anti-distillation arm in the same harness. NEW
- **Mechanism.**
  - Load two plateau snapshots (for example two F7 roots @77500, or one F7 and one F9 root at the same step).
  - Cool them **jointly** on identical batches, with m02 as now.
  - Loss for each peer: its own listwise CE + K9 aux + λ(t) · KL( softmax(s_peer / T).detach() ‖ softmax(s_self / T) ), computed per feed. Ramp λ in over the first ~10% of the cooldown (UniDot only switched it on after epoch 1; Google ramps distillation by curriculum).
  - Arms:
    - **DML+**: λ ∈ {0.5, 1}, T = 1.
    - **DML−** ("anti-distillation"): a small negative weight on a per-feed correlation penalty between the peers' within-feed z-scores. The aim is lower member correlation at equal strength.
  - Each peer stays a separate pool member.
- **Source.** Jigsaw ACRC 2025 6th (DML); UniDot, KDD Cup 2026 runner-up (multi-path mutual learning); Google Ads (anti-distillation).
- **Evidence.**
  - [E] Jigsaw: both peers +0.004 private, and the DML-peer ensemble beat the independent ensemble by +0.002 private.
  - [E] UniDot: +0.135% AUC, confounded by path averaging.
  - [E] Google: anti-distillation lowered PD with no accuracy loss.
  - [O] At our scale, the DML+ effect on single members is likely +0.0005-0.001. The open question is the ensemble.
- **Relevance / match.**
  - Never tried: born-again KD is CLOSED (it raises correlation with a *fixed* teacher), and weight averaging hurt the pool (−0.0025, avg3). DML differs: both peers move, and the Jigsaw evidence is at ensemble level.
  - Our pool gain is driven by diversity at equal strength (frontier law, R² 0.95). So DML− is the arm with a mechanism aimed directly at that law.
  - Risk: DML+ raises member correlation and repeats the avg3 failure. **Judge only by the k-member ensemble plus member correlation (r6 protocol).**
- **Cost.**
  - Code: ~60-80 lines in the cd harness (two models, two optimizers, one batch stream).
  - GPU: N pairs (8 members) per arm ≈ 2× one re-cool wave ≈ 1.5-3 GPU-h.
- **Test.**
  - Compare an 8-member pool from co-cooled peers against the same 8 roots cooled independently with m02 (F7x members).
  - Report val, holdout, priv-mix, the coldpos slice and the mean pairwise corr. Use the existing PASS-CD gates.

### R2. Soft-target val fine-tune (AES 2.0 two-stage recipe). NEW at FT level
- **Mechanism.**
  - In `finetune.py`, replace the target y/K with
    `(1−α) · y/K + α · softmax(s_TO / T)`
    over the feed's candidates. s_TO is the member's own train-only logits on val. Those are clean: that model never saw val.
  - Grid: α ∈ {0.25, 0.5}, T = 1.
  - Variant B: use the train-only *pool* z-mean as the teacher (a stronger teacher, but it raises FT-pool correlation).
- **Source.** AES 2.0 3rd place: stage-2 on the target subset with stage-1 OOF soft labels improved CV and public. 2nd place: two-stage training was key.
- **Evidence.**
  - [E] Focusing stage 2 on the target subset: B 0.792 → 0.815, private +0.010 (public flat).
  - The soft-label part is [Q] (no separate number).
- **Relevance / match.**
  - Our FT overfits early (epoch-1 peak in 37 of 52 runs; COVERAGE "shorter fine-tune" row).
  - We blend 0.5 z(FT) + 0.5 z(TO) in *output* space. Soft targets are the *training-space* version and can regularise each fold model.
  - Not in COVERAGE: label smoothing, the LS proxy and train-stage KD were tested; FT soft targets were not.
  - Watch-out: the final 0.5/0.5 mix then double-counts TO. Evaluate the final mix, not only FT OOF, and keep w_ft = 0.5 (w_ft is closed).
- **Cost.** FT-only (one FT pass per member per α); no train-stage GPU. Rule: ≥ 2 FT seeds or z ≥ 2.5.

### R3. Locate the m02 under-transfer: weekend vs weekday slices on random-like train feeds (CPU). Diagnostic, possibly NEW
- **Mechanism.**
  - Public is Sat/Sun; private is Mon/Tue; val is Tue/Thu.
  - If the 30K random-like train holdout spans Sep 6/12/13/19 (weekends) as well as weekdays, compute the paired m02 vs c0 (F7x vs F7) member and pool deltas separately on the weekend and weekday holdout feeds.
  - A weekend-only shrinkage supports "public shift that does not reach private". Equal deltas support "public noise".
  - Then emulate public and private as in CIBMTR 3rd: many bootstrap draws of val reweighted to each target's M × cold mix (pmix exists). Locate where the observed public delta of each pool falls in the emulated public-delta distribution.
- **Source.** CIBMTR 3rd (public at the ~75th percentile; a private-oriented pick won). AES 2.0 (public flat, private +0.010). ISIC 7th (CV that mimics the test novelty).
- **Evidence.** [E] as cited. [O] Our PLAYBOOK already notes that val is closer to private on M, category and weekday.
- **Relevance.** This is today's open problem. If workflow wpa63h0vs (the weekend/shift diagnostic) already did this, skip it.
- **Cost.** CPU minutes; the predictions exist.

### R4. Missing / padding / zero indicators for every NaN-filled family. PARTIAL → promote
- **Mechanism.**
  - Extend the existing `has:<name>` mechanism (`nn/data.py:attach_xt`) to every family where NaN → 0 collides with real zeros. FR-8 lists `ctx_hn_*`, which is 0 on 87.7% of val rows.
  - Add an all-zero user-embedding flag (`u_missing`), an H == 0 flag, and has-flags for the remaining ctx / seq families.
- **Source.** STAR (KDD Cup 2026).
- **Evidence.** [E] −0.00133 test AUC without disambiguation: the 2nd-largest component, ahead of InfoNCE and the target-aware decoder.
- **Relevance / match.**
  - FEATURES_REVIEW FR-8 / item 5 parks this as "~0, free when a preset changes".
  - Caution from our own pretrain_report: an availability bit can encode a train/val difference (the LUO-cold bit misfired). Build every flag identically for train and val/test, and run the presub has-later probe.
- **Cost.** Only in the next from-scratch preset: 2-epoch screen pair, then a 10-epoch pair if it passes.

### R5. FM-highway: learned dot-product shortcuts into the scoring head. NEW
- **Mechanism.**
  - In every block, compute dot products between the candidate state and each of: the target-attention output, the retr16 output, the user token, and the feed-mean candidate state. Use per-head or per-group splits so there are several scalars per pair.
  - Concatenate across blocks, apply LayerNorm, and feed the result to the final head next to its current input, bypassing the residual stream.
  - Zero-init the new head weights.
- **Source.** UniDot (KDD Cup 2026 runner-up).
- **Evidence.** [E] −0.127% AUC without the FM highway; −0.087% for the cross-bus dots alone.
- **Relevance / match.**
  - Our dense input has raw-embedding cosines (cos_u, smax, lse, top3 with within-feed z / rank). It has no *learned-space, per-layer* dot products.
  - Not in COVERAGE (gated attention, DCN in F11, hyper-connections, DenseFormer are different).
  - [O] The prior is modest (+0.0002-0.0008), because the block-form record is mostly null. As a cheap structural variant it can also be a diversity family.
- **Cost.** ~40 lines in `model.py`; 2-epoch screen (2 seeds), then the e10 pair rule.

### R6. Input-pipeline diversity families ("independent data processors"). NEW as a family design
- **Mechanism.** Build members whose input presets differ at the family level. For example, one family drops the cvnd co-vis block and keeps pre2s plus history; another drops pre2s and keeps cvnd. Every other recipe setting stays fixed.
- **Source.** Home Credit Stability 2024: six independent feature pipelines "significantly enhanced diversity" [Q]. OTTO 1st place averaged LGBMRankers with different hyperparameters (already in deep_otto_hm).
- **Relevance / match.**
  - Our ensemble lever is same-strength different-recipe families.
  - Feature-subset families were never run (grep: none).
  - Gate (a) (within 0.002 of the best family) may reject a weaker subset. The frontier law says lower correlation can pay even at a small strength loss, so measure the correlation first on 2 seeds.
- **Cost.** One family (6 seeds × 10 epochs ≈ 13 GPU-h) after a 2-seed probe (≈ 4.5 GPU-h).

### R7. Cross-attention query generation and multi-view target attention (QueryFormer). NEW, low-medium
- **Mechanism.** Two changes:
  - (a) The target-attention query of each candidate is produced by a small cross-attention from the candidate token over [user token; feed-context token; the candidate's own feature-group tokens], instead of a linear projection.
  - (b) H = 2-4 independent "views": separate input projections of the candidate, each reading history with its own attention, concatenated.
- **Evidence.**
  - [E] Query generation vs MLP: +0.00038 AUC.
  - [E] H 1 → 4: +0.00066 at 1.4× latency.
- **Relevance / match.**
  - Multi-head attention already gives part of (b), and block-form changes have been null for us.
  - Rank as a diversity-family candidate rather than a strength lever [O].
- **Cost.** ~80 lines; 2-epoch screen.

### R8. Cross-feed contrastive auxiliary (STAR / Tencent 2025). NEW in-ranker; partly covered by pre2s
- **Mechanism.**
  - Symmetric InfoNCE between the pooled history representation and the representations of the feed's positive candidates.
  - Negatives: candidates (positives and negatives) from **other feeds** in the batch.
  - Weight decayed 0.1 → 0.01; τ = 0.07; positive rows only.
  - The aim is to shape the candidate tower against a wider negative set than the ~100 candidates of the feed. That could help cold candidates.
- **Evidence.**
  - [E] STAR: −0.00034 AUC without it.
  - [Q] Tencent 2025 3rd: scaling negatives gave "substantial gains" (in retrieval).
- **Relevance / match.** Our pre2 encoders already carry a next-item contrastive signal (pretrain_report). In-ranker contrastive training was never run.
- **Cost.** ~40 lines; 2-epoch screen.

### R9. Loss-prediction ("confidence") auxiliary head (LEAP 1st). NEW, low
- **Mechanism.** A small head predicts the per-feed main loss (stop-grad target), with weight about 0.1.
- **Evidence.** [E] Same spec: private 0.78631 → 0.78869.
- **Relevance / match.** [O] The mechanism is unclear; probably a representation regulariser. Low priority. It is cheap if bundled with R5 or R8.

### R10. Action-conditioned history tokens (Tencent 2025 1st). Low
- **Mechanism.** Replace the additive event_type / event_source embedding of history tokens with FiLM (scale and shift on the item embedding), plus a learned per-head attention bias by event type.
- **Evidence.** [Q] only; the organizer summary gives no numbers.
- **Relevance / match.**
  - The r6 "typed" view FAILED at 2 epochs.
  - a4u (per-type counts in retrieval) PROCEEDs at e10.
  - So typed information enters better through counts and retrieval than through key views [O]. Only worth it as part of a diversity family.

### R11. Muon+ (one extra normalisation after orthogonalisation). Low, nearly free
- **Evidence.**
  - [S] Muon+ paper: consistent perplexity gains on 60M-1B LMs.
  - [E, Z Lab] Optimizer choice is worth ~0.003 AUC; AMUSE vs Muon is within noise.
- **Relevance.** Muon is adopted. Muon+ is a one-line variant: run it only as a cheap diversity or strength probe inside an already-scheduled pair.

### R12. Rare-ID hygiene in the pre2 encoder (frequency capping, AdagradAR). Low
- **Mechanism.** pre2 has a 40M-row item-ID table. The proposal: map rare IDs (fewer than k occurrences) to a shared bucket, and use Adagrad or AdagradAR for the sparse table.
- **Evidence.** [E] STAR: −0.00012 for frequency remapping. Z Lab: −0.00089 for AdagradAR; larger ID tables raised val but lowered test [Q].
- **Relevance.** Retraining pre2 and rebuilding the features is expensive, and pre2's marginal value is +0.0004-0.0009. Only worth it if pre2 is rebuilt for another reason.

---------------------------------------------------------------------------------------------------

## 4. New evidence that confirms closed items or current practice (no action)

| finding (source) | our status | note |
|---|---|---|
| Time-of-day / weekend context: −0.0020 AUC without it (STAR); Fourier time (Tencent 1st); per-domain time embedding −0.0002 (Z Lab) | CLOSED (feat_cal_: the weekend flag hurts cross-day-type transfer by −0.0008 to −0.0014; hour is weak) | Their task is pooled pCVR AUC, where time shifts the base rate. Ours is within-feed ranking, where it largely cancels. History has no timestamps |
| Multi-teacher → single-student distillation +0.0007 (QueryFormer) | KD CLOSED | Distillation pays when you must ship one model. We ship an ensemble |
| Larger ID tables: val up, test down (Z Lab); AdagradAR anti-memorisation | item-ID residual CLOSED (−0.0022; cold −0.0075) | same lesson |
| Semantic IDs plus random-k for the long tail (Tencent 1st) | CLOSED | - |
| Dense-feature representation is the biggest lever (Z Lab −0.0095) | covered: log1p, within-feed z and rank in `model.py`; PBLD / feature tokens null | our dense path is already rich. Nothing to add beyond R4 / R6 |
| Muon-family optimizer −0.0028 vs AdamW (Z Lab); MuonPlus (QueryFormer 1st) | ADOPTED (Muon +0.0031) | an independent replication of our choice |
| Each architecture mechanism is worth 0.0002-0.0011 (QueryFormer, UniDot, STAR) | many block forms CLOSED | expect at most +0.0005 per block idea; use blocks for diversity |
| Epochs 4 → 11: AUC +0.0001, logloss worse (QueryFormer); EMA −0.00035 (Z Lab) | 10 ep adopted; stage-1 EMA null | consistent |
| Dropout unnecessary when ensembling at large data (LEAP 1st) | dropout NEVER varied; d0 / d20 wave running | supports d0 as the default if wave 2b is neutral |
| Pre-train on exposure / production signal, fine-tune on the target signal (Tencent 2nd; AES 2.0) | ADOPTED (train → val FT; m02) | - |
| Recency weighting and adversarial-validation feature dropping failed (Home Credit) | coverage-weighted val CLOSED; adversarial verdicts used only for leak screening | consistent |
| GBDT: XGB > LGBM > CatBoost on per-search flight ranking (FlightRank) | GBDT rankers and stackers CLOSED (val ~0.37) | - |
| Neural models trained directly on biased clicks underperform; propensities needed recalibration (WSDM'23 ULTR 3rd) | covered (deep_position_bias_random) | - |

---------------------------------------------------------------------------------------------------

## 5. General intuitions (cross-competition)

1. **The ranking of levers is the same everywhere in 2025-26.**
   - Information and inputs (features, availability flags, absolute context): 1e-3 to 1e-2.
   - Optimizer and schedule: ~3e-3.
   - Each architecture trick: 1e-4 to 1e-3.
   - Ensembling and diversity: 1e-3 to 5e-3 where it is allowed.
   - Our record has the same shape. The remaining room is in diversity at equal strength, and in any input that adds information.
2. **Online peer learning appeared twice in 2025-26 as a cheap booster** (Jigsaw DML, UniDot). It is the one "training trick" family not yet in our COVERAGE_MAP. Our diversity-driven pool needs the anti-correlation variant to be tested next to the standard one.
3. **Small target-distribution data is handled in two stages, with the big data first.** Winners regularise the second stage with soft labels from the first, and select checkpoints on the *other* domain (AES 2.0). Tuning thresholds or weights on the target-like slice alone looked great on CV and failed on LB (AES 2nd). This matches our closed "val-fitted ensemble weights".
4. **The public LB is a sample of a sub-population, not a noisy copy of private.** CIBMTR, AES 2.0 and Home Credit all had public and private diverge for compositional reasons. Our public is weekend-only and private is weekday. Use emulation (R3) before reading a 0.0015 val-vs-public gap as overfitting.
5. **Frozen content embeddings need a learned interaction path** (MMCTR: they hurt DIN but helped a transformer + DCN). Our candidate-set transformer plus retrieval over history is that path. Pairing them with ID memorisation hurts out-of-time test (Z Lab, STAR).

---------------------------------------------------------------------------------------------------

## 6. Gaps

- Not found: the 1st-place write-ups of RecSys Challenge 2025 ("rec2"), FlightRank 2025, Ozon E-CUP 2025 (recsys track), the VK LSVD 2025 winners and Tianchi 2025 recsys contests. Kaggle discussion pages do not render for the fetch tool.
- The Tencent 2025 top-solution details are the organizers' qualitative summary only.
- The CRAFT / CoSFormer (KDD Cup 2026 award) papers were not found.
- UniDot's multi-path gain is confounded by averaging the two paths. The Jigsaw DML evidence comes from LLM classifiers on small data. Both are transfer risks for R1.
