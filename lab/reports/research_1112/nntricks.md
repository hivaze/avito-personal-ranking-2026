# nntricks: generic NN-training and ensemble tricks from top writeups (2023-2026), matched to our pipeline

Angle: generic NN-training and ensembling tricks from recent top Kaggle writeups in any domain (LLM, audio, vision, tabular), plus the papers behind them. Each item is matched against COVERAGE_MAP / CURRENT_STATUS (as of 2026-11-12 14:45 UTC).
Research only: web reading plus light CPU (pdftotext). No GPU jobs, no process touched, no existing file edited.

Evidence tags:
- [E] = a number measured in the cited primary source, read in this session.
- [S] = a search-engine excerpt or secondary summary of a writeup whose full body could not be fetched (Kaggle writeup pages are JS-rendered; only the LMSYS 1st-place body was read in full).
- [K] = prior knowledge, not re-fetched in this session.
- [O] = our opinion or inference.

## 0. Bottom line

1. **Highest-value new lever: treat the cooldown's data composition as a search space.** "m02" (2% exploration-pool replay in the cooldown) is one instance of a well-documented LLM-pretraining trick: upsample target-like data only in the last 10-20% of training. That literature also uses short "micro-anneals" as cheap data ablations, which is exactly our cd_lane harness. Untested axes: recency-weighted regular batches, and feed-cold-share-weighted pool batches. Weekend is NOT a useful axis (eda_shift_overlap: no weekend effect in the feature-label relation). → T1.
2. **Fixed-step "full-data" members.** m02 members already select their last step, so the 30K holdout no longer does real selection for them. Its feeds (the purest p_random > 0.9 exploration feeds) could join the cooldown pool (+22%) for final-only members. This is Kaggle's standard "refit on 100% after the schedule is fixed". → T2.
3. **A label-free check of the val→public/private transfer.** Agreement-on-the-line uses unlabeled test predictions, which we have for ~100 members, to predict whether a val gain should shrink on a shifted split. It directly targets today's open problem: the m02-heavy pools transferred 0.2x (M) to 0.6x (G) of their val gain to public; PF7x 0.9x. → T3.
4. **Ensemble→student distillation is not worth it for us.** Every 2024-25 Kaggle win with it (LMSYS, WSDM Cup, Eedi efficiency prize) was driven by an inference budget or by abundant unlabeled target-domain data. We have neither. Students that "beat the teachers' members" exist ([E] LMSYS 1st), but a student strongly correlated with the pool adds ~0 to a 40-member pool (our strength-diversity law). The one variant with a new mechanism is "relabel the large source set with a model trained on target-like data" (AES 2.0 1st, +0.005 [S]). That is deep_PLAYBOOK K3, never run; low priority. → T7.
5. **External evidence agrees with our closed list on SWA/EMA/soups, hill climbing, test pseudo-labels, TTA and mixup** (section 3). The exceptions:
   - late-phase regularisers (late dropout → R-Drop → AWP), only if the running d20 arm beats d0 (T5);
   - data-bagged trunks as a decorrelator, a coin flip with literature on both sides (T6).

## 1. Ranked list

| id | idea | status vs our closed list | expected (val, [O]) | cost |
|---|---|---|---|---|
| T1 | Cooldown data-mixture micro-anneals: recency and cold-share sampling on top of m02 | NEW (m02 / m0 / pp02 vary only the pool dose and threshold) | +0.0003..+0.0015 member; unknown at ensemble level | ~50-line flag + 4 roots x ~1 GPU-h per arm |
| T2 | Holdout feeds into the m02 cooldown pool; fixed-step members for the final pool | NEW (refit on full val is the FT stage; this is the train stage) | +0..+0.0006 member (pool +22%) | 7 cooldowns (~7 GPU-h); costs the holdout as evaluator for those members |
| T3 | Agreement-on-the-line: predict test-public/private deltas from unlabeled test predictions | NEW (pmix reweights val; it never uses test predictions) | diagnostic | CPU 1-3 h |
| T4 | Final selection rules when public is a non-representative slice | partly in place | protects private | 0 |
| T5 | Late-phase regularisers in the cooldown: late dropout (d20, running) → R-Drop → AWP | PARTIAL (d0/d20 running; SAM in FT closed; AWP / R-Drop never) | +0..+0.0005 | R-Drop about +70% cooldown time; AWP about +100% |
| T6 | Data-bagged trunks (k-fold user subsamples) to lower the pairwise rho | NEW | ±0.0005 ensemble; a coin flip | 0 extra if it replaces planned seeds |
| T7 | Ensemble→student KD (incl. K3 "relabel with the target-like teacher") | KD born-again CLOSED; test PL CLOSED; K3 NEVER | ~0 at ensemble level; K3 0..+0.001 | K3: inference + 1 cooldown arm |
| T8 | Hyper-deep ensembles (random hyperparameters per member) | NEW in form; the evidence predicts little (cross-family rho ≈ seed rho) | ~0 | needs a gate-protocol change |
| T9 | FT-stage small-data tricks: layer-wise LR decay, head re-init, holdout feeds as extra FT data | NEVER varied (FT lr/wd/batch) | +0..+0.0005 | cheap FT runs |
| T10 | Shrink-and-perturb before branching a cooldown (cheap but decorrelated members) | NEW, speculative | unknown | 1 flag + 4 branches; CPU pre-check first |
| T11 | Within-cooldown checkpoint ensembling (output space) | NEW in form; near-duplicate of the closed avg3 | ~0 | inference only |
| C1-C7 | SWA/EMA/soups/WiSE-FT, hill climbing, test PL, TTA/mixup, TTT, fold merging, multi-seed validation | CLOSED / adopted; external evidence agrees | - | - |

## 2. Idea cards

### T1. Cooldown data-mixture micro-anneals (generalising m02)
- **Mechanism.**
  - r6 showed that the cooldown sets the final mean function: the 7-seed cooled ensemble minus the plateau end is +0.0031 ± 0.0006, with seed correlation unchanged. Moving 2% of cooldown batches to the exploration pool (m02) gave +0.0012 per member (z 3.0) and +0.0045 on coldpos.
  - So the cooldown is the phase where *what data you see* has the largest effect per step.
  - LLM pretraining uses exactly this: in the final 10-20% of training (the LR decay), upsample the target-like domain. It also uses short anneals from a mid-training checkpoint to rank candidate data mixtures cheaply.
- **Sources and evidence.**
  - Blakeney et al. 2024, *Does your data spark joy? Performance gains from domain upsampling at the end of training* (Databricks, ICML 2024) [E, abstract]:
    - up to +6.90 MMLU, +8.26 GSM8K and +6.17 HumanEval vs the base mix (7B, 1T tokens);
    - the best upsampling duration is 10-20% of training (tested 5-30%);
    - as a data-ablation tool it costs about 10x less than full runs.
  - Llama 3 report (Meta 2024) [S, paper excerpt]:
    - annealing on a small high-quality set lifted Llama 3 8B by +24.0% on GSM8k and +6.4% on MATH;
    - new data sources are judged by annealing a 50%-trained 8B to LR 0 over 40B tokens with 30% weight on the candidate data. This is a "micro-anneal" exactly like our cd_lane.
  - OLMo 2 (AI2, 2025) [S]: "microannealing" ablations rank mid-training data sources. The final base model is a soup of several anneals of the same mixture.
  - Ibrahim et al. 2024, *Simple and scalable strategies to continually pre-train LLMs* [E, abstract]: LR re-warming + re-decaying + replay of old data matches full retraining on new data. This is the recipe for "branch from a plateau snapshot and cool down on newer data".
  - Kaggle analogues:
    - AES 2.0 1st (2024): pre-train on all data, then fine-tune on the test-like subset, +0.015 QWK [S]. The 2nd, 3rd and 5th places also used two-stage training.
    - HMS 2nd (2024): stage 2 trains 5 epochs on the high-agreement subset only [S].
- **Untested axes.** Our m02 / m0 / pp02 arms vary only the pool dose and the p_random threshold.
  - (a) **Recency:** draw the regular (non-pool) cooldown batches with weight ∝ exp(-age_days/τ), τ ≈ 7 days, so the last train week is about 2-3x upsampled. Rationale: test is 7-10 days after train and its item population is newer; the most recent train feeds have the closest item and category mix.
  - (b) **Cold-share composition:** draw pool batches with weight ∝ (feed candidate cold share)^γ, so the cooldown sees private-like (cold share 0.41) composition. This is sampling only, not input perturbation, so it differs from the closed "age-conditioned input dropout" and "staleness tricks". Byrd & Lipton 2019 [K] showed that importance weighting fades over long training of over-parameterised nets. Applying it only in the cooldown is the LLM-annealing way around that.
  - (c) Weekend upsampling is NOT recommended: eda_shift_overlap §s08 found no weekend effect in the feature-label relation (f1 lift and positives per feed are flat across days).
- **Relevance.** High.
  - The harness, pre-registered rules (r6_eval_cdw2/3 stage 1 → stage 2 PASS-CD) and the pmix veto all exist.
  - Read coldpos and priv-mix as well as val and holdout. Arm (b) is designed to move them.
  - Caveat: the holdout is in-period and not stale, so it will not reward (a) or (b). Pre-declare val + priv-mix as primary for these arms, with holdout ≥ -1 SE as a guard.
- **Cost.** One sampler flag in nn_x/r6/train.py, with flag-off identity tests as in wave 2 (~1-2 h of build), then 4 roots x ~1 GPU-h per arm for stage 1.

### T2. "Full-data" fixed-step members: holdout feeds in the m02 pool
- **Mechanism.** The Kaggle default is: fix the schedule on CV, then refit on 100% of the data with a fixed number of steps.
  - For m02 members, holdout-based selection is nearly void: the mean selected step is 19,001 of 19,670, and the end-minus-selected decline is +0.00009.
  - The 30K holdout feeds (p_random > 0.9, M >= 50) are the purest exploration feeds and are never trained on. Adding them to the cooldown pool is +22% pool (134K → 164K) at the 2% dose.
- **Sources and evidence.**
  - NVIDIA *Kaggle Grandmasters Playbook* (Deotte et al., 2025) [E, blog]:
    - "refit on 100% of the training data" after tuning, which gave "an additional leaderboard gain" in Playground S5E6;
    - 100 XGBoost seeds 0.379 vs 0.376 average single seed (MAP@3).
  - The pp02 arm (pool threshold 0.2, +47% pool from lower-purity feeds) is running. It gives our own pool-size elasticity first.
- **Relevance.** Medium.
  - These members can no longer be scored on the holdout, so gates and pool choices for them rest on val only.
  - Safer protocol: split the holdout into A/B halves. Run cooldowns with A in the pool and evaluate on B, and vice versa, to measure the effect on 4 roots before any final-only use.
  - Conflicts with T9(c), which also wants the holdout feeds.
- **Cost.** Same as one m02 cooldown wave (7 roots x ~1 GPU-h). Final members: one more wave.

### T3. Agreement-on-the-line as a label-free transfer check (val → public / private)
- **Mechanism.** Across many models, out-of-distribution (OOD) accuracy vs in-distribution (ID) accuracy often lies on a line ("accuracy-on-the-line").
  - Baek et al. show that pairwise agreement OOD vs ID lies on a line with the same slope and bias.
  - So the shrinkage of a model difference under shift can be estimated from unlabeled OOD predictions alone.
- **Sources and evidence.**
  - Baek, Jiang, Raghunathan, Kolter, *Agreement-on-the-Line* (NeurIPS 2022) [E, abstract]: the prediction method beats prior OOD-accuracy estimators. It holds for neural classifiers, including across architectures.
  - Mehra et al. 2024 (arXiv 2404.01542) [S]: the effect needs diversity in the model pool; lightly fine-tuned foundation models lack it. Our ~100 random-init members have it.
- **How for us [O].**
  - Agreement between members i and j on a feed = NDCG of i's ranking using j's top-k (k = number of positives expected, e.g. 2) as relevance; or per-feed Spearman.
  - Compute it on val, test-public (all weekend) and test-private, using the public/private boundary ts 1790497920 from eda_shift_overlap.
  - Fit the OOD-vs-ID agreement line per split. If the public slope is clearly below the private slope, a weekend/coverage shift explains the m02 under-transfer and private should behave more like val. If the slopes match, the public shortfall is noise or selection optimism.
  - Calibrate with the 5 public-scored subs (v3/v4/v5/v10/v13 + today's 3).
  - Caveat: the ranking-agreement adaptation is untested in the literature.
- **Relevance.** Directly addresses today's open diagnosis, using test inputs that pmix does not use. **Cost:** CPU 1-3 h on existing preds/.

### T4. Final-selection rules when public is a non-representative slice
- **Evidence.**
  - AES 2.0 1st (2024), "Trust CV (and LB a bit)": rose from 619th public to 1st private by validating on the test-like Kaggle-only subset [S].
  - BirdCLEF 2025 2nd (CEUR paper) [E]: the OOF-Optuna-selected ensemble scored private 0.929 vs 0.928 for the public-selected one. The authors call the difference not significant, but the direction favours CV.
  - Playground S5E5 (2025): an ensemble pruned by public LB fell from public #3 to private #40. The author's lesson: positive-only weights and a few strong models [S].
  - ISIC 2024 4th: the author's best-private submission was not his best-CV one, which is selection noise [S].
  - Roelofs et al., NeurIPS 2019, meta-analysis of 120 Kaggle competitions [E, text]: "little evidence of substantial overfitting" from public-LB reuse. Large public-private gaps come from non-i.i.d. splits or tiny test sets. Both apply to our public: weekend-only, 8.5K feeds.
- **For us.**
  - Public is 100% weekend; private is 24% weekend and closer to val (adversarial AUC val-vs-private 0.521 vs val-vs-public 0.533).
  - Today's public spread among M / G / PF7x is 0.00039, about 0.3 SE of the ~0.0012 paired SE.
  - Use public only as a veto for large failures (< -2 SE vs the reference). Never order candidates by public.
  - Keep the current two-pick structure (holdout-best + val-best, with a no-m02 hedge until m02 is verified).
  - Apply a winner's-curse discount to val-best pools: the measured selection optimism among 6 designs was +0.0005, and more decisions have been taken on val since.

### T5. Late-phase regularisers (only if d20 ≥ d0): late dropout → R-Drop → AWP
- **Mechanism.** Late-cooldown memorisation of the pool (c0 selects at step ~7300 and declines -0.0026) is a late-phase overfitting problem.
  - Late dropout (dropout switched on only late in training) targets overfitting regimes.
  - R-Drop adds KL(p1 ‖ p2) between two dropout passes of the same input. For us, that is the listwise softmax over a feed's candidates.
  - AWP takes an adversarial weight-perturbation step before each update (flat-minimum regularisation).
- **Sources and evidence.**
  - Liu et al., *Dropout Reduces Underfitting* (ICML 2023) [E, abstract]: late dropout "consistently improves generalization accuracy" for overfitting models; early dropout helps underfitting ones.
  - MAP 2025 3rd (public 1st), monsaraida & Masaya: R-Drop against very noisy labels; AWP on top gave a further small CV gain [S].
  - USPPPM 2022 8th: AWP +0.005 (text classification) and +0.01 (token classification); adv_lr matters more than adv_eps [S, older].
  - ASL Signs 2023 1st: AWP switched on only late in training [K, not verified this session].
- **Relevance.** Low-medium.
  - m02 already removed most of the late decline (m02 selects its last step), so the headroom is smaller than when c0 was the default.
  - The d0/d20 arms (`--drop_cd`, running since 08:43) are the late-dropout test.
  - R-Drop needs dropout on: run it only if d20 ≥ d0 at stage 1, as one more cd_lane arm (α ∈ {0.5, 1}).
  - AWP is lowest: SAM in the FT stage was below MDE, and AWP is the same family.
- **Cost.** R-Drop adds a second forward/backward pass on cooldown steps (~+70% cooldown time); AWP ~+100%.

### T6. Data-bagged trunks (k-fold user subsamples) as a decorrelator
- **Mechanism [O].**
  - With N ≈ 40 members, the pool is limited by pairwise correlation (gain ≈ α(1-ρ) - β·gap; α 0.019, β 0.38 on val).
  - Changing the history encoder (GRU), width (d192) or lr kept cross-family ρ at the seed level (0.92-0.94). So recipe variation is not a decorrelator here.
  - Training each member on a different 80% of users is the classic decorrelator. It also yields OOF predictions on train feeds for free.
- **Evidence for.**
  - Zhu et al. CIKM 2020 (Huawei, ensembled CTR via KD) [E]: teachers trained on different data partitions beat teachers that differ by architecture or seed.
  - BirdCLEF 2025 2nd [E, qualitative]: models trained on different pseudo-label subsets added ensemble diversity.
- **Evidence against.**
  - Lee et al. 2015 (*Why M heads are better than one*) and Nixon/Lakshminarayanan (*Why aren't bootstrapped neural networks better?*) [E/S]: for deep nets, bagging lowers ensemble-mean accuracy vs full data + random init, because a bootstrap keeps only ~63% of unique data.
  - Our own data-scaling point: B23/B24 at 150K feeds score 0.379-0.381 vs ~0.395 at full data, about 0.0035 per data doubling. So an 80% subsample costs about -0.001 per member (β·gap ≈ -0.0004). The diversity gain is unknown.
- **Relevance.** Medium-low; worth only as a free pre-registered pair when new trunks are launched anyway (e.g. the next m02 seeds): 4 bagged vs 4 full-data seeds; compare single strength, ρ and the 4-member ensemble on val + holdout.
- **Cost.** About 0 extra GPU (replaces planned seeds) + one `--user_fold k/K` flag.

### T7. Ensemble → student distillation: not for us, except K3
- **Evidence that students can beat teachers' members.**
  - LMSYS Chatbot Arena 2024 1st (sayoulala, writeup read in full) [E]:
    - teachers (5-fold) reached CV log loss 0.869-0.881 (Qwen2-72B, mean 0.876) and 0.873-0.877 (Llama3-70B, mean 0.875);
    - the distilled Gemma2-9B student reached CV 0.858-0.876, mean 0.867 (lower is better), so the student beat each teacher member on log loss;
    - loss = CE + KL + cosine per teacher, all equally weighted (repo code);
    - the 5 fold students were merged by averaging their LoRA weights.
    The student was needed to fit the T4 inference budget.
  - WSDM Cup 2025 (multilingual arena) [S]:
    - 1st: CE + KL + cosine on soft labels;
    - 3rd: best temperature 5, soft-loss weight 0.9; KL+cosine averaging "not obvious";
    - 6th: KD helped Gemma2-9B but not Qwen2.5-14B.
  - Zhu et al. 2020 [E]: KD students "consistently outperform their teachers" and "can even surpass the ensembled teacher". Diverse teacher sets need gating: 6 teachers without gating were worse than 3.
  - Stanton et al. 2021, *Does KD really work?* [E, abstract]: students often do not match the teacher even with enough capacity (optimisation difficulty), and higher fidelity does not always mean better generalisation.
  - BirdCLEF 2025 5th [S]: "self-distillation is all you need", with a 5-model averaged teacher repeated 4-5 rounds.
  - BirdCLEF 2025 2nd [E]: soft pseudo-labels from an 8-model ensemble on target-domain soundscapes gave private 0.889 → 0.906 → 0.909 → 0.907 over iterations 1-3, with CV flat at 0.984-0.985. The gain is domain adaptation.
- **Why it does not transfer to us.**
  - (i) No inference cap: the 40-member pool is allowed, so a student only matters if it beats the pool. Nobody reports that at our scale.
  - (ii) A student near the pool mean adds ~0 to it (the law: high ρ).
  - (iii) Teacher scores on train feeds are in-sample for our full-data trunks; LMSYS used 5-fold teachers.
  - (iv) The target-domain unlabeled set is small (28.6K test feeds), and the model has no item-ID memory to adapt. That is why test PL gave -0.0004 and born-again KD +0.0006 n.s.
- **The one variant with a new mechanism (K3).** AES 2.0 1st [S]: train on the test-like subset, relabel the large old-criteria set with that model, retrain: +0.005 QWK. Our analogue is deep_PLAYBOOK K3 (KDCRec label channel, never run): use the val-fine-tuned fold models to soft-label the exploration pool and holdout feeds, then add a KL term on pool batches during the m02 cooldown. Low priority.
- **Cost of K3.** FT-model inference over ~164K feeds + one cooldown arm.

### T8. Hyper-deep ensembles (random hyperparameters per member)
- **Source.** Wenzel et al., NeurIPS 2020 [E, abstract]: random search over hyperparameters, stratified over seeds, beats fixed-hyperparameter deep ensembles. They credit weight diversity plus hyperparameter diversity.
- **For us.** Our cross-family correlations (GRU vs transformer, d192, lr 1e-3) sit at the seed level, so the extra diversity from hyperparameters is small [O]. Implementing it would also break the family gates (a)+(b). Low; skip unless gates are redesigned.

### T9. FT-stage small-data tricks (FT lr / wd / batch never varied)
- **Mechanism.** The FT stage trains on ~11.5K feeds per fold, 4 epochs, one lr (5e-5) for all parameters.
  - (a) Layer-wise LR decay: lower lr for the history encoder and trunk, higher for the head and the extra-feature branch.
  - (b) Re-initialise the final scoring layer.
  - (c) Add the 30K holdout feeds (never trained on by any train-only member) to each fold's FT training set with weight w ∈ {0.3, 1}. Val stays the OOF evaluator, so the FT evaluation is not contaminated.
- **Sources.**
  - Zhang et al. ICLR 2021, *Revisiting few-sample BERT fine-tuning* [K]: re-init of top layers + longer training helps small-data fine-tunes.
  - LLRD is standard in Kaggle NLP fine-tunes [K].
  - LMSYS 1st [E]: "post-pretrain" on extra data before the target-data fine-tune.
  - AES 2.0 [S]: extra data only helped when weighted or staged toward the test-like part. A naive add of Persuade did not help in one public repo.
- **Relevance.** Low-medium: the FT branch is 50% of the final blend and cheap to rerun. (c) competes with T2 for the holdout.
- **Cost.** A few FT runs (5 folds x 4 epochs on ≤ 42K feeds) per arm; minutes to 1 GPU-h.

### T10. Shrink-and-perturb before a cooldown branch (cheaper decorrelated members)
- **Mechanism [O].**
  - Same-root branches correlate 0.966-0.974 (m02 vs c0), vs ~0.94 across seeds, because they share the plateau iterate.
  - θ ← λθ + σε with λ ≈ 0.6-0.8 before the cooldown forces partial re-learning under a new noise path. That moves branch members toward seed-like diversity at branch cost.
- **Source.** Ash & Adams, *On warm-starting neural network training* (NeurIPS 2020) [E/S]: shrink-perturb closes most of the warm-start generalisation gap; a ReScience reproduction confirmed most results. Using it for ensemble diversity is our extrapolation.
- **Pre-check (CPU).** When the wave-3 cf4 branches (@60000) finish, compare ρ(cf4 branch, @77.5k m02 branch) on the same root with ρ across seeds. If an earlier branch point already gives seed-level ρ, prefer that over any perturbation flag.
- **Cost.** One flag + 4 branches. Speculative.

### T11. Within-cooldown checkpoint ensembling (output space)
- **Idea.** Average the predictions of the last 2-3 eval checkpoints of each m02 run (checkpoint ensembles, Chen et al. 2017 [K]).
- **For us.** These checkpoints correlate ~0.99. The closed avg3 result (weight space, plateau) says the pool loses diversity when members are smoothed. Expected ~0 at ensemble level. Run it only if the checkpoints exist, as a 10-minute CPU check.

## 3. Closed items: what the external evidence adds

| item | our status | external evidence 2023-26 | different mechanism worth reopening? |
|---|---|---|---|
| C1 SWA / EMA / soups / WiSE-FT | plateau averaging CLOSED (pool -0.0024); stage-1 EMA null; FT EMA 0.99 adopted; WiSE-FT closed by argument | CSIRO Image2Biomass 2026 1st: SWA of the final epochs "did well in CV but scored poorly on the LB" [S]. RSNA aneurysm 2025 1st: EMA for inference [S]. NemoSkills AIMO-2 1st: weight averaging across training stages for accuracy and speed [S]. OLMo 2: souped anneals [S]. DiWA (Rame et al. NeurIPS 2022): averaging runs helps OOD only from a shared init [S] | No. Every single-model competition benefits from averaging; a member-equal pool loses diversity. DiWA needs a shared init (our seeds have none); TO↔FT weight interpolation duplicates the 0.5/0.5 output blend |
| C2 Hill climbing / Caruana / fitted weights | CLOSED (+0.0014 optimism) | S3E24 4th: "robust hill climbing" with CV, 7 of 25 OOFs [S]. S5E5: public-pruned blend #3 → #40; positive-only weights pick 5-10 OOFs [S]. S5E5 7th: Ridge beat hill climbing; blender notebooks overfit the LB [S]. S5E11 6th: high-CV blends scored lower on LB [S]. BirdCLEF 2025 2nd: OOF-Optuna 0.929 vs public-selected 0.928 private [E] | No. Member-equal over gated families is the robust limit of these recipes |
| C3 Pseudo-labels on test | CLOSED (-0.0004) | BirdCLEF 2025 2nd [E]: PL pays when the unlabeled target set is large (9,726 soundscapes) and the input shift is learnable. Their recipe: ensemble soft labels; keep chunks with max p ≥ 0.5; zero p < 0.1; 40% sampling; 2 iterations, plateau at the 3rd. Grandmasters Playbook: k separate pseudo-label sets to avoid leakage [E] | No. Our unlabeled target set is small and the shift lives in item coverage, which the model cannot memorise |
| C4 TTA / mixup | TTA CLOSED (r6_tta); mixup never (not applicable) | ASL Signs 2023 1st: CutMix/MixUp failed because the mixed label of two variable-length inputs is ill-defined [S]. BirdCLEF mixup works because labels are additive multi-label | No. Mixing two feeds' candidate lists has no meaningful target. The running "aug" arm (candidate subset + history-event drop) is the right analogue |
| C5 Test-time training | never | Jigsaw Agile Community Rules 2025: a 15th-place review deck notes Unsloth's TTT support and lists the 1st, 2nd, 7th, 8th, 9th and 18th places as users; the excerpt is ambiguous between "used Unsloth" and "used TTT" [S] | Reject. Jigsaw test rows came with labelled rule examples [K]. A self-supervised TTT on test-period histories would derive from later snapshots of the same users, which breaks the §3 leak rule |
| C6 Fold-model merging (LoRA averaging, SLERP) | test fold-bagging adds ~0 (folds correlate 0.9974) | LMSYS 1st averaged fold LoRAs [E]; the Eedi 2024 efficiency winner SLERP-merged 6 models [E, Eedi blog] | No. These are inference-budget tools; we have no budget |
| C7 Multi-seed validation | adopted (≥ 2 seeds, z rules) | MAP 2025 1st: "multi-seed validation was the most critical aspect of this competition" for noisy labels [E, AIME-Con 2026 overview] | Confirms the protocol |
| two-stage training / source head | adopted (FT on val; `--domain` token; m02) | AES 2.0: two-stage +0.015 (1st); the 4th used a data-source classification head instead [S] | Already in place. T1 extends it |

## 4. General intuitions (cross-domain)
1. **Distillation and merging in Kaggle are budget tools.**
   - They win when inference is capped (LMSYS / WSDM 9-14B students; Eedi efficiency merges) or when a large unlabeled target-domain set exists (BirdCLEF PL).
   - With no cap and a small test set, the ensemble itself is the best "student".
2. **When a late phase sets the function, change the late-phase data before the architecture.** m02 is one data point. LLM annealing, two-stage Kaggle fine-tunes and HMS-style "clean subset last" are the same idea. r7's 0/13 block forms vs m02's +0.0012 per member fits this.
3. **At N ≈ 40 members the pool is correlation-limited.**
   - More seeds or recipes at the same ρ give little.
   - New value comes from raising every member (a better cooldown) or from lowering ρ through input or data variation. Architecture variation did not lower ρ here.
4. **Averaging tricks trade diversity for single-model quality** (SWA, EMA, soups, plateau averaging, fold merges). External winners use them because they submit one model. For a member-equal pool, evaluate them only at ensemble level (r6 rule).
5. **A non-representative public split is for vetoes, not selection.**
   - Public here is weekend-only and 8.5K feeds (SE ~0.0012 paired).
   - Kaggle meta-evidence (Roelofs 2019) says LB "overfitting" is mostly noise or split mismatch.
   - The AES 2.0 winner climbed 618 places by trusting the test-like CV.
6. **Winner's curse.** A candidate chosen as val-best after many val-based decisions carries optimism. Today, the holdout-best pool (G) transferred 0.6x and the val-best pool (M) 0.2x. That is consistent with optimism plus public noise, before any shift story [O].
7. **Pseudo-labelling and self-training pay under input-domain shift with lots of unlabeled target data, and saturate after 2-3 rounds** (BirdCLEF 2025 2nd, 5th; the 25th-place retrospective). Without both conditions, expect ~0 or slightly negative, as we measured.

## 5. Sources (fetched or excerpted this session unless marked [K])

Competition writeups and repos:
- LMSYS Chatbot Arena 2024, 1st place, sayoulala, "Distill is all you need" (full body read): https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629 ; code: https://github.com/shyoulala/LMSYS_BlackPearl ; review: https://nebius.com/blog/posts/chatbot-arena-competition-review
- WSDM Cup 2025 multilingual chatbot arena, 1st place (whitefebruary): https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/whitefebruary-1st-place-solution ; 3rd place: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/plamo-1000000b-3rd-place-solution [S]
- BirdCLEF+ 2025, 2nd place, Sydorskyi & Goncalves, CEUR Vol-4038 paper 256 (full text read): https://ceur-ws.org/Vol-4038/paper_256.pdf ; code: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- BirdCLEF+ 2025, 5th place, "Self-Distillation is All You Need": https://www.kaggle.com/competitions/birdclef-2025/writeups/noir-5th-place-solution-self-distillation-is-all-y [S]
- BirdCLEF 2025 1st / BirdCLEF+ 2026 1st (Babych, multi-iterative noisy student): https://www.kaggle.com/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati ; https://github.com/GrayBloom/Reproduction-For-BirdCLEF2026 ; top-3 review deck: https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/Birdclef_2025_top3_solution_review.pdf [S]
- Learning Agency Lab AES 2.0, 2024, 1st place "Trust CV (and LB a bit)": https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/writeups/flg-1st-place-solution-trust-cv-and-lb-a-bit ; top-5 summary: https://hippocampus-garden.com/kaggle_aes2/ [S]
- HMS Harmful Brain Activity 2024, 2nd place summary: https://dev.to/moyuto/kaggle-hms-2nd-solution-explaind-34jj [S]
- MAP Charting Student Math Misunderstandings 2025, organiser overview (Crossley et al., AIME-Con 2026): https://aclanthology.org/2026.aimecon-main.33.pdf ; 3rd place (monsaraida & Masaya): https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution [S]
- Eedi Mining Misconceptions 2024 winners (organiser blog): https://www.eedi.com/news/from-wrong-answers-to-real-insights-how-we-used-a-kaggle-challenge-to-map-student-misconceptions
- Jigsaw Agile Community Rules 2025, 15th-place review deck (TTT usage among top teams): https://speakerdeck.com/hasibirok0/map-jigsaw-code-golf-zhen-rifan-rihui-by-guan-dong-kagglerhui-jigsaw-15th-solution [S]
- ASL Signs 2023, 1st place (hoyso48): https://www.kaggle.com/competitions/asl-signs/discussion/406684 [S]
- USPPPM 2022, 8th place (AWP): https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/writeups/team-n-8th-place-solution-predicting-targets-at-on [S]
- CSIRO Image2Biomass 2026, 1st place (SWA): https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution ; RSNA Intracranial Aneurysm 2025, 1st place (EMA): https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution ; AIMO-2 2025, NemoSkills 1st place (weight averaging): https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills [S]
- Playground hill-climbing writeups: S5E5 https://www.kaggle.com/c/playground-series-s5e5/discussion/582591 and https://www.kaggle.com/c/playground-series-s5e5/discussion/582556 ; S3E24 https://www.kaggle.com/c/playground-series-s3e24/discussion/455296 ; S5E11 https://www.kaggle.com/c/playground-series-s5e11/writeups/6-solution-ensembling-was-the-key ; S5E6 https://www.kaggle.com/c/playground-series-s5e6/discussion/587398 [S]
- ISIC 2024, 4th-place review (zenn, yuto_mo; one of these two articles): https://zenn.dev/yuto_mo/articles/9585f0bf4d458b , https://zenn.dev/yuto_mo/articles/8627736450dab7 ; 4th-place code: https://github.com/dungnb1333/ISIC-2024 [S]
- NVIDIA, *The Kaggle Grandmasters Playbook* (2025): https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data

Papers:
- Blakeney et al. 2024, domain upsampling at the end of training: https://arxiv.org/abs/2406.03476
- Llama 3 Herd of Models 2024 (annealing): https://arxiv.org/abs/2407.21783 [S]
- OLMo 2 2025 (microannealing, souped anneals): https://arxiv.org/abs/2501.00656 [S]
- Ibrahim et al. 2024, continual pretraining (re-warm, re-decay, replay): https://arxiv.org/abs/2403.08763
- Baek et al. 2022, Agreement-on-the-Line: https://arxiv.org/abs/2206.13089 ; Mehra et al. 2024: https://arxiv.org/abs/2404.01542
- Zhu et al. 2020, Ensembled CTR Prediction via Knowledge Distillation: https://arxiv.org/abs/2011.04106
- Stanton et al. 2021, Does Knowledge Distillation Really Work?: https://arxiv.org/abs/2106.05945
- Wenzel et al. 2020, Hyperparameter Ensembles: https://arxiv.org/abs/2006.13570
- Liu et al. 2023, Dropout Reduces Underfitting (early and late dropout): https://arxiv.org/abs/2303.01500
- Rame et al. 2022, DiWA: https://arxiv.org/abs/2205.09739
- Lee et al. 2015, Why M Heads are Better than One: https://arxiv.org/abs/1511.06314 ; Nixon et al., Why aren't bootstrapped NNs better?: https://www.gatsby.ucl.ac.uk/~balaji/why_arent_bootstrapped_neural_networks_better.pdf
- Ash & Adams 2020, On Warm-Starting NN Training: https://arxiv.org/abs/1910.08475
- Roelofs et al. 2019, A Meta-Analysis of Overfitting in ML (120 Kaggle competitions): https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2019/a_metaanalysis_of_overfitting_in_machine_learning__682ee801.txt
- [K] Byrd & Lipton 2019 (importance weighting in deep learning); Zhang et al. 2021 (few-sample BERT fine-tuning); Chen et al. 2017 (checkpoint ensembles); Liang et al. 2021 (R-Drop); Wu et al. 2020 (AWP); Wortsman et al. 2022 (model soups, WiSE-FT).

Internal cross-references: COVERAGE_MAP.md, CURRENT_STATUS.md (Log 08:43-14:44), R6_PLAN.md §2, sol_ensemble.md §2.5, eda_shift_overlap.md (§ public/private boundary, s08 weekend), llm_large.md (B23/B24 150K-feed baselines), deep_position_bias_random.md and deep_PLAYBOOK.md K3 (KDCRec).
