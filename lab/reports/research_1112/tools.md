# Tabular-DL and GBDT tools, 2024-2026: can a GBDT or TabPFN-type component add value to the NN ensemble?

Research angle "tools" of workflow wp7s449jt (2026-11-12). Research only: no GPU and no training. Three light CPU checks were run on saved predictions (scripts in the session scratchpad, §1).

Tags:
- **[E]** = measured in the cited source.
- **[E-local]** = measured this session on our saved predictions (CPU, numpy).
- **[S]** = secondary summary or search snippet; the primary page could not be fetched (Kaggle discussions are JS-rendered).
- **[R]** = taken from an earlier report in `reports/`.
- **[O]** = my opinion or inference.

Skimmed first, so this note does not repeat them:
- `deep_gbdt_ranking.md`: LightGBM/XGB/CatBoost LTR semantics and our GBDT runs.
- `nnx_research_tabular_dl.md`: TabM, RealMLP, PLE, TabR, TabPFN v2/2.5, TabICL v1/v2, BeyondArena, TabArena, TabReD.
- `sol_ensemble.md`: frontier law, stackers.
- `SOLUTIONS_REVIEW.md`, `COVERAGE_MAP.md`.

Everything below is either a new source or a new use of an old one.

---

## 0. Verdict (short answer to the question asked)

**No credible GBDT or TabPFN component adds value as an ensemble member or as a stacker over the ~0.400 NN pool.** Four independent lines of evidence agree:

1. **Our own frontier law** [R sol_ensemble §2.5].
   - A member helps only if its val gap to the pool is below 0.049·(1−ρ).
   - At GBDT ρ ≈ 0.8, a non-NN solver would need val ≳ 0.388. The best GBDT is 0.367-0.371.
   - Stackers, LightGBM residuals and the linear-wide stacker were all null in nested CV. The linear-wide stacker gave GBDTs ~0.03 weight.
2. **Google KDD'19 (Gmail/Drive LTR, hundreds of millions of queries)** [E] measured the exact case "trees weaker than the NN":
   - Decision trees alone were −3.73% / −1.50% MRR vs the NN.
   - A linear blend (our "GBDT member") gained only **+0.10% / +0.14%**.
   - Trees boosting the NN (our "GBDT residual / stacker") gained +0.13% / +0.16%.
   - **Feeding the tree score INTO the NN gained 2-3× more: +0.27% / +0.37% (stacking) and up to +0.31% / +0.46% (boosted stacking with an adapter).**
   - Our gap is larger: −7.5% relative (0.37 vs 0.40). That pushes every variant down.
3. **Scale and shift both favour the NN.**
   - ShareChat (FIRE'23) [E]: CatBoost AUC flattens with data (75M→110M rows: 0.9388→0.939), while the neural ranker keeps improving up to 9B rows.
   - TabReD (ICLR'25) [E]: "XGBoost decreas[es] its performance margin to MLPs when evaluated on temporally shifted test sets". Our private split is the most shifted one (cold share 0.41).
4. **When NN+GBDT blends did win in 2024-25 competitions, one of two conditions held.**
   - (a) The GBDT saw a different input view from the NN:
     - ISIC 2024: image NN + metadata GBDT;
     - Open Polymer 2025: SMILES-text NN + descriptor GBDT;
     - PREPARE: TabPFN features into a GBDT on small data.
   - (b) The two families were within about 1% of each other (TabArena, CIBMTR with TabM + GBDT).
   - Neither condition holds here: the GBDT sees a subset of the NN's inputs and is 3 points behind.
   - The only Jane Street 2024 write-up that blended NN + GBDT + Ridge placed 26th on public. The 6th (public) solution was NN-only [E/S].

**The one GBDT use with a measured mechanism and no closed precedent: put a cross-fitted GBDT score inside the NN as an input or as a frozen base logit** (idea T3; the Google "T-S-N / T-B-N" pattern). The prior is low to moderate: expected +0.000 to +0.001 single-model. Its main value may be a new same-strength family. Everything else in this angle (TabPFN-3/TabICLv2/Mitra, CatBoost embedding features, linear_tree, leaf embeddings, fusion rules) is "do not" or "only if T3 is positive".

**Side result for today's open problem** [E-local, §1]:
- The m02 − c0 gain is **flat across days**:
  - weekend vs weekday on the 30K train holdout;
  - val Tue vs Thu.
- So nothing we can label shows a day-type or short-staleness mechanism behind the weak public transfer of m02-heavy pools. That favours "noise" over "weekend shift", but it cannot test the weekend × 4-5-day staleness interaction that only exists on public.

---

## 1. Light CPU checks run this session [E-local]

Scripts are in the session scratchpad (`/tmp/scratchpad/scripts/`):
- `hold_dow.py`
- `val_day.py`
- `fuse_check.py`

Each read the 7-root c0 vs m02 cooldown-arm predictions (`nn_x/r6/preds{,/holdout}/nn_r6_cd_{c0,m02}_r10{1..7}_*`) and scored them with `src/metrics.ndcg_per_feed`. The holdout feed set was reconstructed exactly as in `nn/train.py:163`; the M arrays matched `holdout_meta`.

### 1.1 m02 − c0 by day type, 30K random-like train holdout

Train contains Sat/Sun Sep 6, 12, 13, 19.

| | weekend (n=9,632) | weekday (n=20,368) | weekend − weekday |
|---|---|---|---|
| single-member mean (7 roots) | +0.00126 ± 0.00039 | +0.00138 ± 0.00027 | −0.00012 ± 0.00048 |
| 7-member z-ensemble | +0.00176 ± 0.00057 | +0.00119 ± 0.00040 | +0.00058 ± 0.00070 |

Per-day single-member gains range from +0.00016 to +0.00219 (SE ~0.0008) with no weekend pattern.

### 1.2 m02 − c0 on val by day (the staleness proxy: Thu is 2 days staler)

| | Tue (n=7,542) | Thu (n=6,878) |
|---|---|---|
| single-member mean | +0.00122 ± 0.00055 | +0.00112 ± 0.00057 |
| 7-member z-ensemble | +0.00089 ± 0.00080 | +0.00130 ± 0.00086 |

### 1.3 Fusion rule on the same 14-member pool

The pool is 7 c0 + 7 m02 roots; val z-mean = 0.40037, holdout 0.32313. Deltas vs z-mean (paired per-feed SE):

| rule | val | holdout |
|---|---|---|
| raw-logit mean | +0.00007 ± 0.00012 | +0.00007 ± 0.00006 |
| per-feed softmax-probability mean, T=1 (probability-ranking-principle fusion) | +0.00010 ± 0.00019 | −0.00016 ± 0.00010 |
| probability mean, T=2 | +0.00016 ± 0.00017 | −0.00004 ± 0.00009 |
| z-mean − 0.25·z-std (risk-averse "consensus/confidence") | +0.00007 ± 0.00018 | +0.00002 ± 0.00011 |
| z-mean + 0.25·z-std (optimistic) | −0.00001 ± 0.00020 | −0.00007 ± 0.00010 |

The fusion rule and the use of member disagreement (Deotte-style "mean/std consensus" features, idea T12) are **null**. The z-mean stays.

---

## 2. Where NN + GBDT combinations won in 2024-2026, and why

| source (year, place) | what was combined | condition that made it work | transfer to us |
|---|---|---|---|
| Google KDD'19 "Combining DTs and NNs for LTR in Personal Search" [E, PDF Table 3] | LambdaMART on dense features + NN on sparse and dense features. 12 combination variants. | **DT score as NN input (T-S-N) or NN boosting a frozen DT score (T-B-N) ≫ linear blend ≫ DTs boosting the NN** | the only measured "weaker-DT" recipe → **T3** |
| ISIC 2024 skin cancer (1st-10th) [S, Zenn summary of the Kaggle write-ups] | image-NN OOF predictions as GBDT features + metadata | different input view (image vs metadata); NN prediction standardised **+ N(0, 0.1) noise**; pred/patient-mean features; rank ensemble. The 3rd-place write-up lists GBDT stacking among things that did not help. 10th place capped 50 negatives per patient. | our GBDT has no extra view; the noise trick → **T11** |
| Open Polymer 2025 (1st) [S] | ModernBERT on SMILES + AutoGluon on RDKit descriptors; an ensemble of 41 XGBoost models produced side features for AutoGluon | different input views | no |
| DrivenData PREPARE 2025 (1st, SDOH track) [E, DrivenData blog] | TabPFN-generated quantile features → LightGBM → XGBoost → CatBoost → Lasso chain | small data ("relatively few samples") | no (we have 247M rows) |
| CIBMTR post-HCT 2025 (1st, 3rd-6th used TabM) [S, speakerdeck; ML Contests] | TabM + GBDTs + kNN/GNN | families of similar strength on small tabular data | TabM closed for us (members 0.999-correlated) [R] |
| Kaggle Playground Apr-2025 (1st, Deotte) [E, NVIDIA blog] | 3-level stack; 75 L1 models (GBDT, NN, SVR, KNN, TabPFN…) from 500; L2 = 1 GBDT + 1 NN on forward-selected OOFs plus mean/std consensus features; **4 target framings** | many cheap, similar-strength models on one small table | consensus null (§1.3); framings → **T4** |
| Jane Street 2024 (6th on public; GRU only) [E, author's solution.md on GitHub] | NN only. "MLP, time-series transformers, cross-symbol attention and embeddings didn't work for me". | online learning +0.0078 CV (GRU 0.0112 → 0.0190); aux targets +0.001; 6-model mean ensemble LB 0.0112 vs best single 0.0105 | recency is the lever there, not GBDT → **T13** |
| Jane Street 2024, other write-ups [S, 2kkkkk notes] | 26th public: online TabM + AE-MLP + offline XGBoost + Ridge. A CatBoost online-training post: staged boosting from previous predictions, retrain every 7 days, ~0.008 LB. | only a lower-placed solution blended NN + GBDT | no |
| ML Contests "State of ML Competitions 2025" [E] | GBDTs in winners: LightGBM 14, XGBoost 14, CatBoost 8. AutoGluon in 2 winners; TabPFN in 1 (as a feature generator); TabM in the CIBMTR winner. | GBDTs still dominate small and medium single-table competitions | our data is large and sequence-heavy |
| OTTO e-commerce LTR (RecSys'25 industry) [E, arXiv 2507.20753] | DNNs vs production LambdaMART on 43M samples | simple two-tower with softmax CE: NDCG_click **+4.32%**; 8-week A/B clicks **+1.86%**. The list transformer was *worse* on order NDCG (−1.48%). | confirms NN > GBDT at scale |
| ShareChat (FIRE'23) [E, arXiv 2312.01760] | MMoE vs CatBoost | NN wins slightly and scales: AUC 0.9420 vs 0.9388 (likes); CatBoost flat 75M→110M rows | same |
| TabArena (NeurIPS'25 D&B) [E] | portfolio of all model families | "both model families contribute to ensembles that strongly outperform individual model families". Holdout validation "greatly underestimates" models and favours already-ensembled ones. | members are within a few % there; ours are not |
| "Ensembling Tabular Foundation Models: A Diversity Ceiling and A Calibration Trap" (arXiv 2605.18696, May 2026) [E] | 6 ensemble strategies over 6 TFMs, 153 OpenML tasks | mean pairwise Q-statistic **0.961**; best (cascade stacking) **+0.18%** acc at **253× compute**, not significant; 3 of 6 strategies significantly worse than the best single model; LR meta-learner "destroys calibration" | same regime as our pool (seed ρ 0.93-0.95): the gain is diversity-bound |

---

## 3. Ideas (each with mechanism, source, evidence, relevance and cost)

Priority key: **P1** = do now (cheap or decision-relevant); **P2** = worth one pre-declared screen; **P3** = only if the trigger is met; **X** = do not do (closed with a reason).

### T1 [X]. GBDT ranker as a z-mean ensemble member (any library, any objective)
- **Mechanism:** variance reduction via a decorrelated member.
- **Source:** Google KDD'19 Table 3; our frontier law; TabReD; ShareChat.
- **Evidence:**
  - [E] LIN blend +0.10-0.14% MRR when DTs are 1.5-3.7% weaker.
  - [R] break-even needs val ≳ 0.388 at ρ ≈ 0.8; best GBDT 0.367-0.371.
  - [E] the GBDT margin shrinks under temporal shift (TabReD).
- **Relevance:** this restates and strengthens the existing closure. Today's tuned-CatBoost run (wpa63h0vs) should judge its output with the frontier rule, not with "val went up in-sample".
- **Cost:** none.

### T2 [X]. GBDT stacker or GBDT residual boosting on top of the NN ("N-B-T")
- **Mechanism:** trees fit what the NN missed.
- **Source:** Google KDD'19 (N-B-T +0.13/+0.16% vs T-B-N +0.26-0.46%); ISIC 2024 3rd place ("stacking of GBDT models did not help") [S]; AutoGluon DyStack, which exists because stacked overfitting is common [S].
- **Evidence:** our side [R]:
  - nested LightGBM lambdarank stackers null;
  - LightGBM residuals −0.0015..−0.011;
  - 70 attributes with local AUC 0.492-0.504.
- **Relevance:** closed. The new source explains why: when trees are the weaker model, boosting *from* the NN is the inferior direction.
- **Cost:** none.

### T3 [P2]. Cross-fitted GBDT score inside the NN: as an input feature (T-S-N) or as a frozen base logit that the NN boosts (T-B-N / T-B-NT)
- **Mechanism:**
  - Trees learn sharp, axis-aligned thresholds and interactions on the dense features. An NN with smooth inductive bias learns these slowly.
  - Handing the NN the tree score as one more candidate feature lets attention and history context decide when to trust it.
  - The alternative is to start the listwise softmax from `logit = w·g_gbdt(x) + NN(x)` with `w` learnable (a scalar adapter, Google's "T-B-NT"), so the NN spends its capacity on what trees cannot do: history-conditioned relevance.
- **Source:** Li, Qin, Wang, Metzler, KDD 2019 (Google), Table 3. The trivago RecSys'19 analogue, where leaf-SVD inputs added +0.003 MRR, is already in `featrev_research.md` F6 [R].
- **Evidence:** [E] T-S-N +0.27/+0.37% MRR and T-B-NT +0.29-0.31/+0.40-0.46% over a well-trained NN, with DTs alone −3.73/−1.50%. Simpler adapters (linear) were as good as complex monotone ones. Scores beat ranks as the boosted signal (R-B-N +0.09-0.13%).
- **Relevance / match to what we tried:**
  - It is not a member, a stacker or KD.
  - Closest closed items:
    - PLE/feature tokens: null, which suggests the NN already digests the 34 features well.
    - "No residual signal", which says the tree function of those features is mostly subsumed.
  - So my prior is +0.000 to +0.001 per member [O]. The second payoff is a **new family** whose inductive bias differs (score-anchored) at near-equal strength. Diversity is the binding constraint of the pool.
- **Design** (to make it leak-safe and shift-safe):
  - (i) LightGBM lambdarank (truncation 300, `bagging_by_query`), or LightGBM with `position=` GAM debiasing (+0.0022 measured [R deep_gbdt_ranking]), on the same `cvnd_pre2s` features the NN uses.
  - (ii) **2-fold cross-fit by user on train.** Every train feed gets an out-of-fold score; val and test get the fold mean.
  - (iii) Standardise per feed (z) before input.
  - (iv) Add N(0, 0.1) noise to the input score during training (ISIC 1st, T11), so the NN does not over-trust a score whose train/test reliability differs.
  - (v) Two arms: input-feature (T-S-N) and base-logit (T-B-NT with a learnable scalar w).
- **Kill criterion:** pre-declared 2-epoch or CD-continuation proxy, 2 seeds. Stop if Δval < +0.0005 **and** Δhold < +0.0003. Then also check ρ to the pool: keep it as a family only if it passes gates (a) + (b).
- **Cost:**
  - CPU: GBDT on ~2 × 1M feeds, about 3-6 h at ≤24 threads. Use `CUDA_VISIBLE_DEVICES` hygiene if XGBoost/CatBoost GPU is used.
  - Inference on 3.1M train feeds: about 30-60 min CPU.
  - NN: 2-4 proxy runs.
  - Dev: about half a day (a new input column in `xt` presets and a base-logit hook in the listwise loss).

### T4 [P3]. "Residual framing" family without a GBDT: NN boosts a frozen simple base model (linear / conditional-logit on the 34 features or f1)
- **Mechanism:** the same as T3's T-B-N with a cheaper, smoother base. The NN learns `y ~ base + residual`, which changes what the trunk represents (a diversity source).
- **Source:** Deotte, NVIDIA blog (Apr 2025): 1 of 4 target framings was "predict the residual of a linear regression". The Feb-2025 Playground was won with that framing alone [E as reported]. Google T-B-N with a "lin" mapping [E].
- **Evidence:** only diversity evidence in other domains; nothing on LTR beyond Google.
- **Relevance:** a family-diversity knob at roughly equal strength, never tried. The fine-tune and the K9 aux head are different mechanisms.
- **Cost:** a nested conditional-logit base already exists (`SOLUTIONS_REVIEW` row "nested conditional-logit residual"), so this needs dev of about 2 h plus proxy runs.
- **Trigger:** run it only if T3's base-logit arm shows the score-anchored family decorrelates (ρ < 0.93) without losing strength.

### T5 [X]. TabPFN-3 / TabPFN-2.5 / TabICLv2 / Mitra / TabDPT as member, stacker or per-feed model
- **Mechanism (claimed):** in-context Bayesian inference over a labelled context.
- **Source:**
  - TabPFN-3 tech report (arXiv 2605.13986, May 2026): 1M rows × 200 features on one H100, or 100K × 2,000.
  - TabPFN-3.5 accepts 1M rows [S PyPI].
  - Mitra (NeurIPS'25): mixed SCM + tree priors.
  - AutoGluon 1.4 "extreme" uses TFMs only for ≤30K samples [E release notes].
  - The TFM-ensembling paper (2605.18696): diversity ceiling.
  - BeyondArena: TFMs lose on grouped and >100K data [R].
- **Evidence:**
  - [E] no published TFM has a ranking or listwise objective. My searches found none; TabDPT's "ranking" is Elo evaluation, not an objective.
  - Our val stacking table is 1.39M rows (above the 1M ceiling), grouped by user, and pointwise.
  - The target structure (one positive among ~100 shuffled candidates) is the listwise case these models are not trained for.
- **Relevance:** our history retrieval (retr16 + target attention) is already a learned, list-aware "in-context" model over the user's own labelled events. The only TFM idea that ever transferred was architectural: thinking rows and SSMax, both covered in `nnx_research_tabular_dl` (T8/T9).
- **Cost of a check:** an H100-day for a subsampled stacker that the frontier law already rejects. Not worth it.

### T6 [X]. TFM or GBDT as a feature generator (PREPARE-style)
- **Mechanism:** model outputs become features for another model.
- **Source:** DrivenData PREPARE 1st place (TabPFN quantile features into LightGBM → XGBoost → CatBoost) [E blog]; H2O Driverless AI + TabPFN embeddings into LightGBM [S].
- **Evidence:** wins are on small data only.
- **Relevance:** for us this is exactly T3 with a more expensive and pointwise generator, so T3 dominates it.
- **Cost:** none.

### T7 [P3]. NN on GBDT leaf indices (DeepGBM / iLTM / trivago leaf-SVD)
- **Mechanism:** each tree becomes a tokeniser; leaf one-hots (or their SVD) are NN inputs.
- **Source:** iLTM (arXiv 2511.15941, 2025) uses deliberately under-tuned GBDT leaf embeddings inside a TFM; tree-regularised embeddings (Amazon, arXiv 2403.00963); trivago +0.003 MRR [R]. The trivago GBDT was their best single model, ours is the weakest.
- **Evidence:** positive only where the GBDT was strong.
- **Relevance:** a richer version of T3. Feature tokens and PLE were null for us [R].
- **Trigger:** only if T3's input arm passes, and then as an ablation (score vs 64-d leaf-SVD).
- **Cost:** the same GBDT as T3 + SVD + 1 proxy pair.

### T8 [X]. CatBoost `embedding_features`, LightGBM `linear_tree`, CatBoost/XGB new objectives as "new tools"
- **Mechanism:**
  - CatBoost embedding features compute LDA projections and kNN class counts. Raw coordinates are not used (official docs).
  - `linear_tree` fits linear models in leaves for extrapolation and is memory-heavy (LightGBM docs, PILOT paper).
- **Evidence:**
  - Embedding calcers are classification-oriented (class-count kNN) and do not target per-feed ranking.
  - Our shift is missing information (coverage), not a linear trend that leaves could extrapolate [R eda_shift_overlap: train per-day means flat; coverage falls about 2 pp/day].
  - Objective choice already ties within ±0.001 [R deep_gbdt_ranking].
- **Relevance:** none beyond what T3 needs: a plain LightGBM lambdarank with full-list truncation.
- **Cost:** none.

### T9 [P1, done here, recommended as a standing gate column]. Stability-aware selection across time slices (Home Credit 2024 "gini stability" analogue)
- **Mechanism:**
  - Home Credit CRMS 2024 scored mean weekly Gini + 88·min(0, slope) − 0.5·std(residuals) [S, metric description] and rewarded models whose gain did not decay over weeks.
  - The analogue for pick decisions: report every candidate's gain per day (12 holdout days, val Tue/Thu) and its slope/variance, and prefer pools whose gain is flat. The public slice is one weekend, the private slice is mostly weekdays, and both are staler than val.
- **Source:** Home Credit CRMS metric [S]; TabArena and BeyondArena on IID-vs-grouped validation distorting rankings [E/R].
- **Evidence:** [E-local §1.1-1.2]: for m02 vs c0 the gain is flat:
  - weekend − weekday −0.00012 ± 0.00048 (single), +0.00058 ± 0.00070 (ensemble);
  - Tue ≈ Thu.
- **Relevance:**
  - It directly informs the m02 public under-transfer diagnosis. Day type and 2-day staleness do not explain it in labelled data. The remaining explanations are public noise, or a weekend × 4-5-day staleness interaction that no labelled slice contains.
  - It also gives a cheap standing check: add "per-day gain slope and SD over the 12 holdout days + 2 val days" to `presub_gate` as an INFO column. It is not a selector, to avoid another fitted rule.
- **Cost:** CPU minutes; scripts are ready (§1).

### T10 [P2]. GBDT adversarial-validation score as a per-feed shift coordinate, then regress member gains on it
- **Mechanism:**
  - Train a LightGBM classifier to separate public-like from val-like feeds using intrinsic + coverage features. The feed-level AUC is 0.533 val-vs-public, 0.657 public-vs-private through coverage [R].
  - Score the holdout and val feeds, then check whether the m02 (or any candidate) per-feed gain trends with the "public-likeness" score.
  - A flat trend means the val gain should transfer (supports noise). A negative slope extrapolated to the public's mean score predicts the under-transfer.
- **Source:** TabFSBench (ICML 2025): "the importance of shifted features has a linear relationship with performance degradation" [S]; our own adversarial work [R sol_advval]. Importance weighting is *closed* (ESS 2,880), so this is used only as a regression covariate, never as weights.
- **Evidence:** none yet.
- **Relevance:** it is the only labelled-data way to say something about the public slice. It may overlap with the running weekend/shift diagnostic; coordinate before running.
- **Cost:** CPU < 1 h.

### T11 [P2 as part of T3]. Noise-injected upstream scores
- **Mechanism:** when one model's prediction is fed to another, add Gaussian noise (σ ≈ 0.1 of the standardised score) during training only. The downstream model cannot over-trust a score whose reliability differs between in-fold, out-of-fold and test.
- **Source:** ISIC 2024 1st place: standardised image predictions + N(0, 0.1) noise into the GBDT; pred/patient-mean features [S, Zenn summary of the Kaggle write-up].
- **Evidence:** a qualitative component of the winning solution; no ablation number published in the summary.
- **Relevance:** it applies only if a score-as-input design (T3/T7) is built. It is not useful for the closed stackers.
- **Cost:** 1 line.

### T12 [X, measured]. Consensus/confidence features and probability-space fusion
- **Mechanism:** mean/std across members as L2 features (Deotte), or probability-ranking-principle fusion (average per-feed softmax probabilities instead of z-scores).
- **Source:** NVIDIA Deotte stacking post (Apr 2025) [E as reported].
- **Evidence:** [E-local §1.3] every variant is within ±0.00016 of the z-mean on val and on the holdout.
- **Relevance:** closed. The z-mean stays.
- **Cost:** none.

### T13 [P3, low prior]. Recency as a lever (Jane Street online learning)
- **Mechanism:** the biggest single gain in the Jane Street 2024 write-ups came from updating on the most recent labelled data:
  - GRU +0.0078 CV, vs +0.0007 for a 6-model ensemble;
  - re-training on the last 7 days with a fresh optimizer "+0.001" (17th public) [E/S].
- **Analogues for us:**
  - (a) the val fine-tune, already adopted and the dominant recency lever;
  - (b) a recency-weighted cooldown that oversamples the last 2-3 train days (Sep 17-19) in place of, or in addition to, the m02 exploration replay.
- **Evidence against:** our train per-day means are flat (no drift in the feature → label map) [R eda_shift_overlap]. The shift acts through item coverage, which features carry rather than weights, and item IDs are banned. Refit-on-full-val was −0.00025 [R].
- **Relevance:** low. Listed because the user asked for intuitions from other competitions; if a cooldown-arm slot is free, (b) is a 1-flag CD-harness arm.
- **Cost:** 1 CD-harness arm × 4 roots.

### T14 [X]. More GBDT seeds or bigger GBDT data as a path to 0.388
- **Mechanism:** the "100 XGBoost seeds" playbook trick (Fertilizer: MAP@3 0.379 vs 0.376 single) [E NVIDIA playbook]; more training feeds.
- **Evidence:** [R] 3 GBDT seeds +0.0008; [E ShareChat] GBDT gains flatten with rows. Reaching +0.017 is not credible [O].
- **Cost:** none.

### T15 [X]. Self-supervised pretraining to beat GBDT under label scarcity
- **Mechanism:** pretrain the ranker on unlabeled lists, then fine-tune on scarce labels.
- **Source:** Hou et al., "Pretrained deep models outperform GBDTs in LTR under label scarcity" (TMLR; v5 Sep 2025): up to 38% over GBDT under label scarcity [E abstract].
- **Relevance:** we are not label-scarce in train (3.1M feeds). The exploration domain is the scarce one, and it is handled by the val fine-tune. Our sequence pretraining was already studied (`pretrain_report.md`, `deep_seq_pretrain_cf.md`). Nothing new to add.

### T16 [X]. Drift-aware TFMs (Drift-Resilient TabPFN) or DeepCORAL/DFR-style shift adaptation
- **Mechanism:** condition on a time/domain index and learn drift from a prior (Helli et al., NeurIPS 2024). Alternatively, align representations (CORAL) or re-weight the last layer (DFR).
- **Evidence:**
  - [E] Drift-Resilient TabPFN: acc 0.688 → 0.744, AUC 0.786 → 0.832 vs the strongest baselines on 18 small datasets with drift.
  - [E TabReD] "Both DFR and DeepCORAL do not improve upon the MLP baseline" on industrial time-split data.
- **Relevance:** small-data method. Our own analogues (age-conditioned input dropout, cold gates/routers) were closed with a measured cold-vs-warm trade-off [R].
- **Cost:** none.

### T17 [X]. Relational foundation models (KumoRFM, RelBench RDL) as an alternative backbone
- **Source:** RelBench (NeurIPS'24 D&B). On rel-avito user-clicks, RDL 65.90 vs LightGBM 53.60 AUROC (test). TabPFN-3 also claims SOTA on RelBench.
- **Relevance:** the LightGBM baseline there has no joins or history features. Our NN already consumes the joined history, so the RDL-vs-GBDT gap does not transfer. LLM-backbone rankers were closed at every size [R llm_large].
- **Cost:** none.

---

## 4. Tool status table (2024-2026 tools named in the brief)

| tool | what it is (2026 state) | status for us | why |
|---|---|---|---|
| TabM (ICLR'25) | BatchEnsemble MLP; used by the CIBMTR 1st and 3rd-6th places [S] | CLOSED [R] | members 0.999-correlated, null at 1.45× cost (nnx_VERDICT B5) |
| RealMLP (NeurIPS'24) | tuned-default MLP + PBLD embeddings + robust scaling | CLOSED / no headroom [R] | inputs already well-conditioned; PBLD null (r7) |
| TabPFN v2 / 2.5 / 3 / 3.5 | ICL TFM; v3: 1M × 200 on an H100, 20× faster than 2.5; v3.5 up to 1M rows [S] | X (T5) | pointwise, no ranking objective, grouped data, > row ceiling; stacker already null |
| TabICL / TabICLv2 | ICL TFM; v2 beats RealTabPFN-2.5 untuned; QASSMax, Muon [R] | X as a model; architecture ideas already mined [R] | same |
| Mitra (Amazon, NeurIPS'25) | TFM on mixed SCM + tree priors; in AutoGluon 1.4 | X | same; AutoGluon uses it only at ≤30K samples |
| AutoGluon 1.4/1.5 "extreme" | portfolio of TFMs + TabM + GBDTs; +290 Elo on TabArena | X | ≤30K-sample portfolio; weighted ensembling that we do better with family gates |
| CatBoost YetiRank / embedding_features | ranking losses; LDA + kNN calcers | YetiRank ≈ other objectives [R]; embeddings X (T8) | no gain over lambdarank or binary; calcers are classification-oriented |
| LightGBM `bagging_by_query`, `position=` GAM | query-level bagging; GAM position debias (+0.0022) | use only inside T3 [R] | the right config for a T3 base score |
| LightGBM `linear_tree` | linear models in leaves | X (T8) | the shift is coverage, not trend |
| GBDT score / leaves into the NN | Google T-S-N / T-B-N; DeepGBM, iLTM | **T3 P2**, T7 P3 | the only measured weaker-tree recipe |
| GBDT adversarial validation | shift localisation | **T10 P2** (diagnostic only) | importance weighting closed; regression covariate OK |
| Stacked-overfitting guards (AutoGluon DyStack, nested CV) | detect L2 leakage | already practised (nested 5×4) [R] | — |

---

## 5. General intuitions

1. **In 2024-26, GBDT + NN wins come from different input views or from parity.** The ISIC, Polymer and PREPARE winners gave the GBDT a view the NN did not have (metadata, descriptors, small-data TFM features). TabArena and CIBMTR blends are between families within about 1% of each other. When the NN sees everything the GBDT sees and is clearly stronger, the literature has no example of the GBDT helping as a member. The measured recipe is to put the tree *inside* the NN (Google KDD'19).
2. **The direction of boosting matters.** With a weaker tree model, "NN boosts trees" beat "trees boost NN" 2-3× (Google). Our closed stacker and residual experiments were the weak direction; the strong direction has never been tried (T3).
3. **Scale and shift move the frontier toward NNs, not GBDTs.**
   - ShareChat: GBDT flat after ~75M rows.
   - TabReD: the GBDT margin shrinks on time splits.
   - OTTO RecSys'25: a simple two-tower beats production LambdaMART online.
   - For a staler private split, any GBDT weight is more likely to cost than to help.
4. **Ensembles of similar models hit a diversity ceiling.** TFM ensembles have Q 0.961 and +0.18% at 253× compute; three of six strategies were worse than the single best. This matches our frontier law (R² 0.95). The productive investments are new *families* at equal strength, not new fusion rules: §1.3 shows every fusion variant within ±0.0002.
5. **Recency beat ensembling in the competition with real drift (Jane Street).** Our data has no feature → label drift across train days; it has coverage drift. So recency levers are weak here and the val fine-tune already captures most of them.
6. **Stability checks across slices are cheap insurance.** Home Credit 2024 made stability the metric. For us it is a reporting column: §1 shows the m02 gain is flat across 12 holdout days and both val days. That makes "public noise" the leading explanation for the m02 public under-transfer, with the caveat that weekend × 4-5-day staleness exists only on public.
7. **Kaggle tabular "tricks" do not transfer to our scale and structure.** Mass feature generation, hill climbing, 3-level stacks and pseudo-labels are small-data, single-table, IID-split tools. Our equivalents were tested and closed: feature families, Caruana, stackers, PL −0.0004.

---

## 6. Sources (URL; what; year; tag)

1. Li, Qin, Wang, Metzler. "Combining Decision Trees and Neural Networks for Learning-to-Rank in Personal Search". KDD 2019 (Google). PDF read, Table 3 and Table 4. https://storage.googleapis.com/gweb-research2023-media/pubtools/5046.pdf ; https://research.google/pubs/pub48133 [E]
2. Lutz, Wilm, Duwe. "Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank". RecSys 2025 industry, OTTO. https://arxiv.org/html/2507.20753v1 [E]
3. Jeunen et al. "On GBDTs and Neural Rankers: A Case-Study on Short-Video Recommendations at ShareChat". FIRE'23. https://arxiv.org/html/2312.01760 [E]
4. E. Volkova. Jane Street Real-Time Market Data Forecasting 2024 solution.md (6th on public per a secondary summary). https://raw.githubusercontent.com/evgeniavolkova/kagglejanestreet/master/solution.md [E]
5. Jane Street 2024 top write-ups (notes compiled from the Kaggle discussions 556542/556544/556610/556541/556548/556686). https://2kkkkk.github.io/OneNotes/Kaggle/Jane%20street/%E6%80%BB%E7%BB%93.htm [S]
6. ISIC 2024 skin cancer: top-10 solution summary (Zenn), with links to the Kaggle 2nd-place write-up https://www.kaggle.com/competitions/isic-2024-challenge/discussion/530183. https://zenn.dev/ottantachinque/articles/2024-09-12_isic2024-top-solution [S]
7. ML Contests. "The State of Machine Learning Competitions 2025". https://mlcontests.com/state-of-machine-learning-competitions-2025/ [E for counts]
8. C. Deotte (NVIDIA). "Grandmaster Pro Tip: Winning First Place in a Kaggle Competition with Stacking Using cuML" (Playground Apr 2025, 1st). https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ [E as reported]
9. NVIDIA. "The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data" (2025). https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data [E as reported]
10. TabM usage in CIBMTR 2025 (1st and 3rd-6th places), speaker deck. https://speakerdeck.com/ryushi496/zatukuritabmwozhi-ru [S]
11. DrivenData. "Meet the winners of Phase 2 of the PREPARE Challenge" (TabPFN features into a GBDT chain). https://blog.drivendata.org/blog/prepare-phase2-winners [E]
12. Open Polymer Prediction 2025: winner summary (ModernBERT + AutoGluon; XGBoost side features). https://xebia.com/blog/winning-recipe-for-predicting-polymer-properties/ ; post-competition report https://arxiv.org/html/2512.08896 [S]
13. Tanna et al. "Ensembling Tabular Foundation Models: A Diversity Ceiling and A Calibration Trap". arXiv 2605.18696, May 2026. https://arxiv.org/abs/2605.18696 [E abstract]
14. AutoGluon 1.4.0 release notes ("extreme" preset: TabPFNv2, TabICL, Mitra, TabM; ≤30K samples; +290 Elo). https://auto.gluon.ai/dev/whats_new/v1.4.0.html [E]
15. TabPFN-3 Technical Report. arXiv 2605.13986 (2026). https://arxiv.org/abs/2605.13986 ; Prior Labs changelog https://docs.priorlabs.ai/changelog/tabpfn-3 [E abstract / S]
16. Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models. NeurIPS 2025. https://arxiv.org/pdf/2510.21204 [S]
17. TabArena. NeurIPS 2025 D&B. https://arxiv.org/html/2506.16791 [E]
18. TabReD. ICLR 2025 (Yandex). https://arxiv.org/html/2406.19380v4 [E]
19. Helli et al. "Drift-Resilient TabPFN". NeurIPS 2024. https://arxiv.org/abs/2411.10634 [E abstract]
20. Hou et al. "Pretrained deep models outperform GBDTs in Learning-To-Rank under label scarcity". TMLR, v5 2025. https://arxiv.org/abs/2308.00177 [E abstract]
21. Thomas et al. "Retrieval & Fine-Tuning for In-Context Tabular Models" (LoCalPFN), 2024. https://arxiv.org/abs/2406.05207 [E abstract]
22. Ye, Liu, Chao. "A Closer Look at TabPFN v2". 2025. https://arxiv.org/abs/2502.17361 [E abstract]
23. iLTM: Integrated Large Tabular Model (GBDT leaf embeddings in a TFM), 2025. https://arxiv.org/pdf/2511.15941 ; tree-regularised tabular embeddings https://arxiv.org/html/2403.00963v1 [S]
24. RelBench (rel-avito results), NeurIPS 2024 D&B. https://arxiv.org/html/2407.20060v1 ; RelGNN https://arxiv.org/pdf/2502.06784 [S]
25. CatBoost embedding features (official docs). https://catboost.ai/docs/en/features/embeddings-features [E]
26. LightGBM linear trees: extrapolation notes https://takeshimg92.github.io/posts/linear_trees_in_LightGBM ; PILOT https://arxiv.org/pdf/2302.03931 [S]
27. Home Credit CRMS 2024 stability metric description (student deck; the Kaggle page is JS-only). https://ai3011.plaksha.edu.in/Spring%202024/PDFs/Krishna%20Sharma.pdf [S]
28. TabFSBench (ICML 2025): feature-shift benchmark. https://mlanthology.org/icml/2025/cheng2025icml-tabfsbench [S]
29. AutoGluon-Tabular (stacked-overfitting rationale for OOF stacking). https://arxiv.org/pdf/2003.06505 [S]

Local evidence:
- `/tmp/scratchpad/scripts/hold_dow.py`
- `/tmp/scratchpad/scripts/val_day.py`
- `/tmp/scratchpad/scripts/fuse_check.py`
- Google paper text: `/tmp/scratchpad/dl_tools/g5046.txt`
