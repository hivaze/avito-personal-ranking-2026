# web_competitions: what produced the last 0.5-1% in ranking competitions, and what of it is left for this pipeline

Written 2026-11-15 21:35 UTC. Scope: web research (41 sources, 32 opened and read, section 7) plus two CPU read-outs on stored predictions (section 4). No training, no GPU, no edit outside `reports/TRICKS_1115/`. Read first: `reports/CURRENT_STATUS.md` sections 1-6 and the log from 11-14, `docs/02_SOLUTION.md`, all 351 lines of `reports/COVERAGE_MAP.md`, the ledger `nn_x/w5/logs/E_kd_predeclared.txt` (blocks 11-15), and the source lists of `EXTERNAL_SCAN_1113.md`, `TOP1_GAP_EXTERNAL_1114.md`, `TRICKS_SWEEP_1113.md`, `RECHECK_1114/SYNTHESIS.md`.

Markers: [M] measured for this report, [R] record on disk (path given), [W] web source (number refers to section 7), [I] judgement. A web number is what the source prints; none was reproduced.

---

## 0. Bottom line

1. Across the write-ups read, the last 0.5-1% came from five things: (a) averaging models of different families and information, (b) using the labelled or unlabelled test-adjacent period (dev data in training, features and vocabularies from test, soft pseudo-labels), (c) list or session augmentation, (d) a regulariser that let a larger network train, (e) online updates when test labels arrive. Numbers are in section 2.
2. Of these, (a) is exhausted inside one model family here and needs another team's model; (b) is in the recipe (val fine-tune, popularity counts and co-visitation over all splits, trajectory encoder); (c) was run and closed in the form the data allow; (e) does not exist in this competition. Only (d) has an untested form, and its prior is low.
3. No public source read today describes a lever with an honest expected gain above +0.0001 at file level that this project has not run, other than a merge with a different model family [I].
4. Two external results explain why "more members" and "more regularisation" both stopped paying here. An ensemble of students distilled from an ensemble does not exceed the ensemble of the original models [W 20]. In large ensembles the members should be less regularised than the best single model [W 31, 32]. The second one gives a new reason for one never-run arm (weight decay 0, proposal P1).
5. One zero-training idea from the write-ups was tested today under a rule written before the run: softmax-probability pooling instead of the per-feed z-mean (the form the second-place MIND report found best [W 8]). Result: NOT SUPPORTED, holdout +0.00007 +- 0.00008, val +0.00005 +- 0.00012; rank averaging is worse on the holdout, -0.00033 +- 0.00018 [M, section 4]. The pooling form of the file stays.
6. Ranked list (section 5): the merge (owner's decision, existing), then P1 under-regularised students read at pool level, P2 the choice of pick 2 by decorrelation, P3 a contrastive feature-corruption regulariser as a capacity enabler, P4 softmax pooling (tested, closed). Sum of expected file gains of P1-P3: about +0.0001 to +0.00015, most of it conditional on the better of two finals counting [I].
7. The sources also carry clear warnings that fit this competition: a tree model stacked on a neural news ranker gained +0.088 AUC on next-day validation and lost 0.034 AUC on a test set one week later; hand-made features (+0.018 / -0.028) and a popularity term (+0.002 / -0.001) had the same sign flip [W 8]. This is the external twin of the project's rule that val reads of second-level models and of popularity-dependent changes are not evidence.

---

## 1. Method

- Order of work: project records first, then searches with generic technical terms only (no project data, scores, file names or identities were sent anywhere), then primary sources: winners' papers and reports, code repositories, and the theory papers that price ensembles and distillation.
- PDF sources were downloaded by the fetch tool and converted with `pdftotext`; quotes come from the text.
- A source counts as "read" only when its text was opened. Abstract-only and search-snippet sources are marked in section 7.
- Sources already in the project's lists were not re-read unless a number was needed (RecSys Challenge 2024 first place, OTTO, H&M, Booking first place, KKBox, Riiid and others are in `EXTERNAL_SCAN_1113.md` section 7).
- Limits: Kaggle discussion pages and Medium posts returned errors (JavaScript or HTTP 403); the OTTO second place, the RecSys Challenge 2021 Medium post and the RecSys Challenge 2023 academic winner were reached only through search snippets. No write-up of the VK RecSys Challenge 2024 winner or of a Yandex Cup 2023-2024 recommender track was found.

---

## 2. What the sources say, by question

Each table: the finding with the source's own numbers, then the status of the same thing in this project.

### 2.1 Second-level models over neural scores

| Source | What worked or failed | Number [W] | Status here [R] |
|---|---|---|---|
| NVIDIA, RecSys Challenge 2022, 3rd [18] | 17 stage-1 models of different families (session kNN, MLP, GRU, transformers, GBDT); stage 2 = three stackers on the validation month: optimised weighted average of rankings, LightGBM lambdarank (25 bags), XGBoost (4 folds x 3 bags) with extra session and candidate features; stage 3 = weighted average 0.5 / 0.16 / 0.32 | stage 2: 0.2073 / 0.2076 / 0.2074 public MRR; stage 3: 0.2082 public, 0.2086 final | Stackers over member scores: 0 of 78 arms pass val and holdout, val-fitted weights anti-transfer (z -2.85) (`COVERAGE_MAP.md` l.219). All members share one input set, so a stacker has no new information |
| Layer 6, MIND, 2nd [8] | XGBoost second stage over a neural news ranker | validation AUC 0.6856 -> 0.7733, test AUC 0.6826 -> 0.6486 against the default model of the report's table (test is one week later) | Tree members and tree stackers closed (`COVERAGE_MAP.md` l.341: trees 0.3836-0.3866 against 0.4016) |
| LogicAI, RecSys Challenge 2019, 1st [30] | 37 LightGBM rankers on different feature sets; blend coefficients found by direct optimisation of MRR on validation | best single 0.69068 MRR; blend score not printed | Same reason as row 1: needs members with different inputs |
| NVIDIA, RecSys Challenge 2021, 1st [34, snippet] | 3 XGBoost + 3 networks, then 2 XGBoost models trained on their predictions | not printed in the snippet | closed class |
| Burges et al., Yahoo challenge, 1st [4] | learned combiners (linear LambdaRank, pairwise optimal combiner, 2-layer net) against a plain sum of normalised scores | ERR 0.457-0.458 for every learned combiner, 0.459 for the plain sum | member-equal z-mean is the holdout optimum (`COVERAGE_MAP.md` l.219) |

Reading: second levels paid when stage-1 models differed in information and the stage-2 rows came from the test regime. Where the test set was later than validation the stacker was the most damaging thing tried. Nothing to add here.

### 2.2 Per-segment and per-day blending

| Source | Finding | Number [W] | Status here [R] |
|---|---|---|---|
| Chapelle and Chang, Yahoo challenge overview [3] | an oracle that picks, per query, the best of the top-10 submissions | ERR 0.4972 against 0.4686 for the winner | routers and gates: the oracle here is a label echo; learned gates null in 8 studies; 0 of 95 gated rule families beat the plain mix nested (`COVERAGE_MAP.md` l.280-281, l.343) |
| Layer 6, MIND, 2nd [8] | expert models per scenario (by history length) | validation 0.6845, test 0.6825 against 0.6856 / 0.6826 for one default model | per-vertical experts closed at the CPU pre-gate (l.283) |
| MIND 1st [7] | training "day by day" to weight recent data | "final result is not as good as random shuffle" | recency-weighted cooldown rec33: -0.0021 val (l.138) |

Reading: per-query headroom is large on paper in every competition and nobody collects it. No source reports a per-day blend that held on the private period.

### 2.3 Pseudo-labels, test-period adaptation, use of the unlabelled test period

| Source | What was done | Number [W] | Status here [R] |
|---|---|---|---|
| Netease, KDD Cup 2022 task 1 (NDCG), 1st [1] | self-distillation plus soft pseudo-labels on the public test set, kept only above probability 0.7; "soft labels worked better than hard labels" | +0.0022 public NDCG for the two together (0.8960 -> 0.8982) | distillation adopted (W8 - C3 public +0.001032, `docs/02_SOLUTION.md` section 11); hard test pseudo-labels closed (l.102); teacher-only labels on an unlabelled user fold (U20) undecided at 2 seeds and explained by "more teacher-like", seeds 953-954 running (ledger 03:13Z, 04:16Z, 07:49Z) |
| MIND 1st [7] | dev day added to training; item vocabulary from train + dev + test | test AUC 0.707 -> 0.7104 | val enters through the fine-tune: public +0.00048 +- 0.00055 (`docs/02_SOLUTION.md` section 8) |
| Booking challenge, 6th [19] | evaluation-set trips (last city masked) used as extra training sequences | best single 0.5043 against 0.4927 without (that model also changed graph features and loss) | histories of all splits already feed the popularity counts, co-visitation and the encoder corpus (`reports/TRICKS_1115/int_weak.md` section 4.3); pseudo-feeds from histories: member +0.00108 +- 0.00072 val, rule failed (l.295) |
| Yandex Cup 2022, 2nd [9] | co-occurrence dictionaries from train and test pairs | recall 0.49 -> 0.64 over three submissions (features changed too) | same as above |
| Volkova, Jane Street 2024 [10] | one gradient pass per test day as labels arrive | about +0.008 on CV as printed; the 6-model file scores 0.0112 on the board | no test labels arrive here; the val fine-tune is the only analogue |
| RecSys Challenge 2024 organisers [2] | features from future impressions of the test period | AUC loss without them: 11.65 points (1st), 5.95 (2nd), 7.5 (3rd) | fresh exposure counts from val + test feeds: no signal over the pool, click lift 1.008 +- 0.021 (l.345). The news effect (article life of hours) does not exist for listings |
| NVIDIA playbook [11] | soft pseudo-labels, k-fold sets of pseudo-labels, final fine-tune on true labels only | one example (BirdCLEF 2024), no number | the order "train with soft targets, then fine-tune on true labels" is the current recipe |
| Stanton et al. [22] | enlarging the distillation set with fresh unlabelled data | fidelity rises, accuracy falls in self-distillation; with a 3-model teacher accuracy "slightly" rises; ImageNet student 75.3 / 75.5 / 75.6% for 1 / 3 / 5 teachers | matches the U20 witness: "nothing is specific to the unlabelled feeds" (ledger 04:16Z) and kd08: member +0.00126, pool -0.00049 (ledger 07:49Z) |
| Tang and Wang, Ranking Distillation [24] | teacher top-K on unlabelled items as extra positives for a small student | MAP +5% to +15% relative for a student much smaller than the teacher | students here have the teacher's size; not the same regime |

Reading: every winner used the test-adjacent period, almost always as inputs or as extra labelled data. Pseudo-label gains are reported for single models with clean labels; the two sources that look at ensembles (KKBox in the project's list, Stanton et al. [22]) say the pool does not gain. Finishing U20 at 4 seeds by its rule is enough; a test-feed transfer set is not proposed (section 6).

### 2.4 Snapshot and multi-checkpoint ensembles

| Source | Finding | Number [W] | Status here [R] |
|---|---|---|---|
| Fort et al. [14] | checkpoints of one trajectory are functionally close; independent seeds are not | Jensen-Shannon divergence of predictions (on out-of-distribution inputs) 0.153 within a trajectory against 0.384 between seeds | plateau weight averaging: 7-member pool -0.0025 val / -0.0016 holdout; seed correlation 0.975 against 0.941 (l.30) |
| Huang et al., snapshot ensembles [26] | M snapshots from one cyclic run | "comparable with ensembling of 2 independent models"; CIFAR-100 DenseNet-100 19.25 -> 17.41% error | plateau snapshots as members fail the gate (l.34) |

Reading: snapshots are a cost saver when independent runs are not affordable. Here independent runs are affordable and the pool is at its asymptote. Closed correctly.

### 2.5 Many-seed bagging: how many models, what curve

| Source | Models averaged | Gain over the best single [W] |
|---|---|---|
| Burges et al., Yahoo, 1st [4] | 12 rankers (8 LambdaMART, each a 10-fold bag; 2 nets; 2 regressions) | ERR 0.464 -> 0.4657 (+0.4% relative) |
| Netease, KDD Cup 2022, 1st [1] | several cross-encoder families, weights by public score and CV | NDCG 0.9022 -> 0.9057 public, 0.9015 -> 0.9043 private (+0.3-0.4%) |
| MIND 1st [7] | 4 copies of one model on different negative samples | AUC 0.7104 -> 0.7133 (+0.4%) |
| Volkova, Jane Street 2024 [10] | 2 architectures x 3 seeds | 0.0105 -> 0.0112 |
| NVIDIA playbook [11] | 100 XGBoost seeds | MAP@3 about 0.376 -> 0.379 (+0.8%) |
| NVIDIA, KDD Cup 2023 [13, 35 snippet] | "something like 100 different models" | not printed |
| Riiid 1st [36 snippet] | 5 models (history length 1728) | public AUC 0.816 -> 0.818 |

Curve: for independently trained networks the ensemble metric follows `metric(n) = metric(inf) + b / n` with an exponent "slightly larger than -1"; three or four ensemble sizes predict the rest [W 5]. The project measured the same law: all 255 subsets of W8 follow `a - b/n` to 1e-4, and W8 sits 0.00024 +- 0.00012 (val) below the asymptote (`reports/RECHECK_1114/pool_composition.md` l.9) [R]. The ledger prices 16 -> 32 members at +0.00008 and 16 -> 64 at +0.00013 per branch (block 2026-11-15T21:07Z) [R]. The external law gives no reason to expect more: growth is safe and small.

Why the asymptote itself does not move with more students: after distilling an ensemble into single models, "its performance does not exceed the ensemble over the original single models"; ensembles gain where members learn different views of the data, and the student already holds all views [W 20]. The disagreement-to-error bound says the same: ensemble improvement is at most the ratio of member disagreement to member error, and members that fit their training targets closely have a low ratio [W 15]. A larger single model reproduces what an ensemble of smaller ones gives, with per-example gains correlated 0.76-0.81 [W 16]. Together: more seeds, wider students and a second distillation generation draw from the same function. Only new information lifts it. This agrees with `pool_composition.md` l.244 [R].

One external lever on the asymptote that was never run here: in large ensembles "one should use under-regularized students" [W 31]; a 2025 study that tuned weight decay for the ensemble score found a lower optimum than for the single model (CIFAR-10 WRN-16-4: 5.62e-4 against 7.5e-4), with effect sizes that vary by task [W 32, abstract and snippet]. See P1.

### 2.6 Rank against score averaging

| Source | Finding | Number [W] |
|---|---|---|
| Layer 6, MIND, 2nd [8] | three forms tried: mean rank, mean raw logit, mean softmax probability; "scaled probabilities ensemble works the best" | no number printed |
| Burges et al. [4] | sum of scores normalised to unit variance was as good as any learned combiner | ERR 0.459 against 0.457-0.458 |
| Bruch et al. [12] | convex combination of normalised scores against reciprocal rank fusion | +0.015 to +0.032 NDCG on nine datasets; rank fusion is parameter-sensitive |
| Yandex Cup 2022, 2nd [9] | inverse-rank blend of three near-identical rankers | 0.0849 / 0.0854 / 0.08608 single -> blend not separated |

Status here: the file form is the member-equal mean of per-feed z-scores of raw logits (`src/stack_val.py` `feed_rank_z`, column 1) [R]. Rank and reciprocal-rank fusion were closed earlier (`COVERAGE_MAP.md` l.111). The softmax-probability form was not on record and was tested today: section 4.

### 2.7 Post-processing of ranked lists

No write-up read today reports a gain from re-ordering a finished list under a per-list rank metric (diversity, de-duplication, item-pair rules). The one post-processing paper found, the academic winner of RecSys Challenge 2023, rescales predicted probabilities for a log-loss metric [W 33, snippet]; under per-feed NDCG a monotone rescaling changes nothing. Status here: 64 set-level rules (similarity propagation, duplicate-group maximum, MMR diversity, category mass) gave at best +0.00023 +- 0.00015 in-sample and -0.00076 under 2-fold selection (`reports/err_feed_structure.md` l.12) [R]. Closed correctly.

### 2.8 Per-list NDCG with several positives

- Yahoo challenge: 4 of the winner's 8 tree rankers were trained for ERR and 4 for NDCG; the mix of objectives added less than the bagging [W 4]. The organisers' lesson: once the function class is rich, "the choice of the loss function is of secondary importance" [W 3].
- Status here: the loss zoo (LambdaLoss, NeuralNDCG, ApproxNDCG, ListMLE, pairwise, mixtures) is closed (`COVERAGE_MAP.md` l.87-88); the NDCG-consistent ordering key for several positives reads -0.00006 +- 0.00032 (l.269) [R]. No source offers a further trick.

### 2.9 A regulariser that unlocked capacity (the one item with an untested form)

Layer 6 won RecSys Challenge 2023 by 2.5% relative with a transformer over tabular impression features trained jointly on the supervised loss and a contrastive loss between two corrupted views of the same row [W 23]. Their ablation (validation log loss):

| Objective | Log loss [W 23] |
|---|---|
| supervised only | 0.3471 |
| supervised + the same augmentations, no contrastive term | 0.3448 |
| contrastive, feature corruption only (5-25% of features replaced by values of other rows) | 0.3417 |
| contrastive, embedding masking only | 0.3438 |
| contrastive, two dropout passes only | 0.3455 |
| all three views | 0.3297 |

Their supervised model "exhibits significant overfitting after the 3rd epoch despite having dropout in every Transformer block and weight regularization" (2M parameters, 3.4M rows). Schedule: joint loss first (weight 0.6 on the supervised term), then supervised only.

Counter-evidence, same class: the original feature-corruption paper found the joint form worse than pre-train-then-fine-tune and an average gain of about 2.4% relative with full labels [W 25]; a tabular pre-training study found contrastive objectives no better than reconstruction and "small or absent" gains on its largest datasets [W 27].

Status here: the two-dropout-pass form (R-Drop) ran in the cooldown only, on width-128 hard-label members: val -0.0001, holdout +0.0006, correlation 0.990 (`COVERAGE_MAP.md` l.205) [R]. Feature corruption as a contrastive view never ran. Width 256 fails under distillation, with and without dropout 0.2 (l.300; ledger 20:50Z: -0.00163 / -0.00075) [R]. See P3 and its honest prior.

### 2.10 Validation next to training, test later: the MIND reports

MIND had a dev set of one day right after training and a test set of one week later (new items 32.3% in dev, 87.5% in test) [W 7]. What the two top reports found [W 7, 8]:

| Change (base: default model 0.6856 validation / 0.6826 test AUC in [W 8]) | Dev / validation | Test |
|---|---|---|
| hand-made features (history length, category counts) | 0.7031 | 0.6545 |
| tree second stage | 0.7733 | 0.6486 |
| popularity term | 0.6877 | 0.6814; the method "relies heavily on" how recent the supervised data is |
| context features of the same impression | +0.4% AUC on dev | lower on test |
| masking 92% of unseen item ids in dev | dev and test then moved together | - |

This is the same mechanism the project found on 11-15 (popularity inputs 5.69 days stale on val, 9.34 on public, 11.23 on private; `CURRENT_STATUS.md` section 1) [R], and the same cure (a test-like validation view: stale-val, pmix). It supports the evidence policy of 02:1xZ and the 0.2 popularity-light admixture. It adds no new action.

Two 2025-2026 industry papers treat drifting engagement features by removal of unstable features [W 29, abstract] or by resampling the drifting factors in training [W 28, abstract]. Both forms were run here (popularity-light views; hz and hz2 stale-input cooldowns) and the file-level read of hz is negative on val and on pmix-private (ledger 12:51Z: -0.00047 +- 0.00035, pmix-private -0.00072 +- 0.00060) [R].

---

## 3. Where the last 0.5-1% came from, mapped to this pipeline

| Lever in the sources | Typical size [W] | Here [R] | Left? |
|---|---|---|---|
| Averaging different model families | +0.3% to +0.4% relative (2.5) | one family; pool 0.0002 below its asymptote; GRU, DIN, early-fusion, tree and LLM members are near copies or weak | only through another team's model |
| Seeds of one recipe | 1/n law | measured, same law | +0.00008 for 16 more students |
| Dev or test-adjacent labelled data in training | +0.5% AUC (MIND) | val fine-tune, +0.0005 public | weight of the fine-tuned branch: a public probe is planned (ledger 21:07Z item C) |
| Unlabelled test period as inputs | large in news, moderate elsewhere | in the three stale-input families already | no |
| Soft pseudo-labels on test | +0.24% NDCG with self-distillation (KDD Cup 2022) | distillation adopted; unlabelled fold undecided | finish U20 by its rule |
| List or session augmentation | best single models in the Yahoo winner | sub-feed arms closed as implemented; history dropout harmful (-0.00098 val, ledger 03:14Z) | no |
| Regulariser that enables capacity | -5% log loss (RecSys Challenge 2023) | distillation was that step (+0.0010 public); capacity closed above width 192 | P3, low prior |
| Under-regularised members for a large ensemble | theory + one study | pool falls with weight decay 0.02 / 0.05 / 0.10; 0 never run (`int_weak.md` section 0 item 6) | P1 |
| Online updates on test labels | dominant in Jane Street | not available | no |
| Second-level model | +0.03% to +0.4% when stage-1 models differ | closed | no |
| Post-processing | none reported for rank metrics | closed | no |

---

## 4. CPU check run for this report: pooling form

Scripts and log: `reports/TRICKS_1115/web_competitions_checks/pool_forms.py`, `pool_forms_holdout.py`, `pool_forms.log`. Read-only on `preds/`, `nn_x/w5/preds/holdout/`, `nn_x/abl/preds/holdout/`. 12 CPU threads, about 10 seconds each. Side effect, as for every read-out tool that imports `src/stack_val.py`: numba refreshed its compiled-function cache for `feed_rank_z` in `src/__pycache__/` (two cache files, no source or data file); nothing else outside the report directory was written [M, `find -newermt`].

Forms compared inside each pool (the 0.8 / 0.2 mix of the two pools and the 0.5 / 0.5 mix of the two branches stay in the file form): per-feed z-mean (file form), mean within-feed rank, mean softmax probability at temperature 1 (no scan), mean raw logit.

Check 1 (first look, no rule): the 16 + 4 full-data members of FD16P20, final val. The z-mean reproduces 0.40271 [M].

| Form | Final val | Minus z-mean |
|---|---|---|
| z-mean | 0.40271 | - |
| rank mean | 0.40271 | -0.00000 +- 0.00026 |
| softmax mean | 0.40290 | +0.00019 +- 0.00013 |
| raw mean | 0.40278 | +0.00007 +- 0.00007 |

Check 2 (rule written in the script header before the run, first run 2026-11-15T21:21:58Z): the held-out twins (16 wide students `w5_e10_{kd05w192,kd05t25w192}_s951-958` + 4 wide popularity-light students), fresh witness = train-only pools on the 30,000 holdout feeds, repeat witness = final val. Rule: softmax pooling SUPPORTED iff holdout delta >= +2 SE and val delta >= 0.

| Form | Holdout, 16 + 4 pool | Minus z-mean | Holdout, wide 16 alone | Val final | Minus z-mean |
|---|---|---|---|---|---|
| z-mean | 0.32414 | - | 0.32460 | 0.40214 | - |
| softmax mean | 0.32421 | +0.00007 +- 0.00008 | -0.00005 +- 0.00007 | 0.40219 | +0.00005 +- 0.00012 |
| raw mean | 0.32412 | -0.00001 +- 0.00002 | +0.00006 +- 0.00004 | 0.40210 | -0.00004 +- 0.00005 |
| rank mean | 0.32381 | -0.00033 +- 0.00018 | -0.00029 +- 0.00014 | 0.40221 | +0.00007 +- 0.00026 |

Verdict by the rule: NOT SUPPORTED (holdout z +0.88) [M]. The +0.00019 of check 1 was one of three forms tried in-sample and did not repeat on the twins. Score-based forms are level with each other to 0.0001; rank averaging is the only form that reads lower on the holdout (z -1.8), in line with [W 8, 12]. Numbers tried in this section: 3 alternative forms, 2 member sets, 2 witnesses.

Row for `COVERAGE_MAP.md` (for the lead to paste): "softmax-probability pooling (T = 1), raw-logit pooling, rank pooling in place of the per-feed z-mean | CLOSED 11-15 | held-out twins: holdout +0.00007 +- 0.00008 / -0.00001 +- 0.00002 / -0.00033 +- 0.00018; val +0.00005 +- 0.00012 | reports/TRICKS_1115/web_competitions.md section 4".

---

## 5. Proposals, ranked by expected file gain times probability

All probabilities and gains are [I]. "File gain" means the final file on private. The paired SE between near-copy files is 0.0004-0.0007 on public and about 0.65 times that on private (20,033 feeds against 8,562) [R for the public SE, `CURRENT_STATUS.md` section 1; I for the scaling].

| Rank | Proposal | New or new reason | Cost | P | Gain if real | Expected |
|---|---|---|---|---|---|---|
| 0 | Merge with a team whose model is not a history-attention ranker | existing (owner's decision); external reason added | 0 GPU | 0.25-0.35 given a partner [R `RECHECK_1114/SYNTHESIS.md` bet 1] | +0.0010 to +0.0026 [R `CURRENT_STATUS.md` section 5] | +0.0003 to +0.0008 |
| 1 | P1 UNDER-REG: students with weight decay 0 (then 0 + dropout 0), read at pool level | never run; new external reason | 13.6 GPU-h for 4 seeds, 27 for 8 | 0.12 | +0.0003 | +0.00004 |
| 2 | P2 PICK2-DECORR: pick 2 = the equal-level file least correlated with pick 1 | new arithmetic for an open decision | 0 | 1.0 if the better of two counts, else 0 | +0.0001 to +0.0002 | +0.0001 conditional |
| 3 | P3 CONTRA-CAP: contrastive feature-corruption loss in the plateau, then width 256 | new mechanism for a closed row | about 60 GPU-h for 8 seeds | 0.03 | +0.0005 | +0.000015 |
| 4 | P4 SOFTMAX-POOL | tested today | 0 | - | - | closed, 0 |

### P0. Merge (not new; listed because it is the only item above +0.0003)

- External reason added today: in the Yahoo challenge the top submissions scored within 0.0025 ERR of each other and "most of the differences are not statistically significant", yet a per-query oracle over the top 10 reached 0.4972 against 0.4686, so the submissions were different functions [W 3]. The winner reports that "including the less accurate models ... also helped" [W 4]. Every family ensemble in section 2.5 gained 0.3-0.4% relative, about the size of the gap to the leader (0.0033 of 0.395 = 0.8%).
- Nothing else in this report is of that size. The tool and rule exist (`src/merge_check2.py`, `reports/MERGE_PROTOCOL.md`) [R].

### P1. UNDER-REG: under-regularised students for the pool

- What: a student arm with `--wd 0` (recipe otherwise `kd05w192`), judged only at pool level. If it passes, a second arm `--wd 0` plus dropout 0, and new full-data students of the growth plan take the passing recipe.
- Mechanism: the pool averages away the noise that weight decay is there to control, so the member optimum is over-regularised for the pool. The analytic result for ensembles of linear students: "In large ensembles, one should use under-regularized students in order to maximize the benefits of the variance-reducing effects of ensemble learning" [W 31]. A 2025 study that tuned weight decay against the ensemble score found a lower optimum than for the single model, with effect sizes that "vary significantly" by task [W 32].
- Evidence here: the pool falls with weight decay on record: 0.05 gives member +0.00056 val and 8-pool -0.00043 +- 0.00049 val / -0.00063 +- 0.00027 holdout (ledger 03:48Z); member correlation rises 0.9722 / 0.9810 / 0.9848 at 0.02 / 0.05 / 0.10 (`reports/TRICKS_1115/int_weak.md` section 0 item 6) [R]. The same "member up, pool down" pattern holds for teacher weight 0.8 (ledger 07:49Z) [R]. The dose below 0.02 was never run (`COVERAGE_MAP.md` l.47).
- Closest closed rows and the difference: l.47 "weight decay (0.02): kwwd05 x8, kwwd10 x4" tested only the upward direction; "dropout 0" under distillation has 2 seeds only (member and 2-seed asymptote read, l.46, l.300; a 2-seed screen of that kind passes 36% of true +0.0005 effects, l.315). The downward direction with a pool-level rule is not on record. The sibling audit lists the same arm (T01); this report adds the external reason and the staging.
- Attach: lane line `--wd 0` appended to the `kd05w192` arm in a new lane file (write-new; no edit of `nn_x/w5/lane.sh`), held-out members first (`--holdout 30000`), seeds 951-954 then 955-958.
- Cost: 13.6 GPU-h for 4 seeds, 27 GPU-h for 8 (figures of `int_weak.md` section 5) [R].
- Rule, written now: one-stage 8-seed rule of the ledger (11-14 21:2xZ): 8-pool val delta >= 0, holdout delta >= 0, pooled z >= 2 against `kd05w192` seeds 951-958; REPLICATE band 1.6-2; matched-N reference from `arm_read8.py`. Stop after 4 seeds if the 4-pool pooled z < 0. Secondary, no decision: mean member correlation must be below the reference (0.9722 on val); if it is not, the mechanism is absent and a pass is treated as a draw. Number of arms in this family: 2 at most.
- Why the probability is 0.12 and not higher: the measured shift of the optimum in [W 32] is small (factor 0.75); the decay here is already small (0.02, applied to the AdamW decay group and to the Muon matrices, `nn_x/w5/train.py` l.121, l.396-400 [M, code read]); and width 192 already memorises in the cooldown (`COVERAGE_MAP.md` l.287, l.292) [R], which less decay can make worse.

### P2. PICK2-DECORR: design of the final pair

- Applies only if the better of two finals counts (not settled, ledger 20:30Z item 1) [R].
- Arithmetic: for two files of equal expected level whose private difference has standard deviation s (true difference plus sampling noise of the private set; both count, because the board ranks the realised score), E[max] - E[one] = s / sqrt(2 pi) = 0.40 s. Near copies: public paired SE 0.0003-0.0007 on record -> private s about 0.0002-0.00045 -> +0.00008 to +0.00018. A file of equal expected level with s 0.0008 would give +0.0003; a file 0.0005 lower in expectation with s 0.00065 gives +0.00008 (s phi(mu/s) + mu Phi(mu/s)) [I, arithmetic].
- Consequence: pick 2 should be the file with the same expected level and the lowest test correlation to pick 1, not the closest copy. On disk the candidates are FD16P20vf (differs in 4 of 24 members) and W8P20 (held-out members, test correlation 0.9975, the lowest on record; `docs/02_SOLUTION.md` section 9) [R]. The external support is thin: winners' write-ups pick by cross-validation and do not discuss the second pick; ensembles do not add robustness beyond their accuracy [W 16], so the hedge has to come from different inputs or members, not from size.
- Closest record and the difference: `TRICKS_SWEEP_1113.md` PICK-EMAX ("tie-break only, +0.00003 to +0.0001"). The number above is of the same order; the point added is the direction (prefer the less correlated equal-level file) and that the unioned or grown files reduce it. No test is possible; it is a decision.

### P3. CONTRA-CAP: contrastive feature-corruption regulariser, then width

- What: in the plateau phase add a contrastive loss between the clean candidate representation and the representation of a corrupted copy of the same candidate (10-25% of the 34 dense inputs replaced by values of the same column from other candidates of the batch; random mask on the candidate token; second dropout mask). Negatives = the other candidates of the feed. Weight annealed to 0 at the cooldown, as the K9 loss anneals. The supervised, distillation and K9 losses keep seeing clean inputs.
- Mechanism: instance discrimination under input corruption penalises functions that depend on exact input values, which is what memorising a click label needs. In the winner's ablation the two-dropout-pass view alone gave the least and feature corruption the most [W 23].
- Closest closed rows and the difference: l.205 R-Drop and SAM in the cooldown (the weakest of the three views, on width-128 hard-label members that do not memorise late: @last holdout drop -0.00002 against -0.00105 at width 192, l.287); l.290 SSL-WIDTH trigger (it measured the information of a masked-event prediction task, not a consistency regulariser); l.52 uniform input dropout (it corrupted the supervised input itself, old recipe); l.300 and ledger 20:50Z capacity under distillation (regularisers tried: dropout, weight decay). A contrastive corrupted view in the plateau is on no row.
- Attach: copy of `nn_x/w12/stale/fulldata/train_full.py` into a new directory with three flags (`--con_w`, `--con_fc`, `--con_mask`); the history encoder output is shared between the two passes, so only the candidate path runs twice. Expected step time about 1.6-1.8 times (the early-fusion arm measured 1.84 times, `COVERAGE_MAP.md` l.216) [R, I].
- Cost: 8 held-out seeds at width 192, about 7 hours each on shared GPUs: about 60 GPU-h. Width 256 only after a pass.
- Rule, written now: stage 1 = the 8-seed pooled-z rule against `kd05w192` (as P1), plus the mechanism check: mean member correlation not above the reference by more than 0.003 and the @last holdout drop smaller than the reference's. Stage 2 only on a stage-1 pass: 4 seeds at width 256 with the same loss against the stage-1 arm, same rule at 4 seeds with the 1.6 band. One arm, no dose scan before stage 1.
- Why 0.03: three regularisers in a row raised the member and lowered or left the pool (weight decay 0.05, teacher weight 0.8, R-Drop at correlation 0.990) [R]; the source data set was small enough to overfit in 3 epochs and this one has 246.7M training candidates; two papers in the same class report weak or absent gains with full labels on large tables [W 25, 27]. P1 and P3 pull in opposite directions; if P1 passes, P3 should not be started.

### P4. SOFTMAX-POOL: tested, closed (section 4).

---

## 6. Considered and not proposed

| Idea from the sources | Why not |
|---|---|
| Test and val feeds as a distillation transfer set in the cooldown (soft targets from the pool, later-snapshot feeds excluded) | Same family as U20, which reads "more teacher-like, nothing specific to the unlabelled feeds" [R ledger 04:16Z]; the control for more teacher weight is negative at pool level [R ledger 07:49Z]; fidelity gains from extra unlabelled data do not turn into accuracy [W 22]; pseudo-labels help single models, not ensembles (KKBox, `EXTERNAL_SCAN_1113.md` T12) [R]. No offline witness for test rows. P 0.03 [I]. Finish U20 by its rule |
| Multi-sample dropout, adversarial weight perturbation, gradient-sign noise on inputs (KDD Cup 2022 winner [W 1]) | Same class as dropout 0.2, SAM and input dropout, all read under this recipe and null or negative [R l.205, l.339, ledger 20:46Z]. P 0.02-0.03 each [I] |
| Members trained on lists with candidates deleted (Yahoo winner's Aug50 / Aug70 [W 4]) | Sub-feed arms closed as implemented, and the clean form was dropped with reasons (sub-feed logits equal full-feed logits, KL 0.002-0.005 nats; `RECHECK_1114/SYNTHESIS.md` section 5) [R]. No new reason found |
| Hyperparameter-diverse members (hyper-deep ensembles [W 21]) | External gain shrinks for strong models (WRN-28-10: accuracy 0.826 -> 0.828 at 4 members); here eight constructions with correlation 0.955-0.972 share one asymptote (`pool_composition.md` l.244) [R]. Only the directed form P1 is kept |
| Subsampled training sets for under-regularised members [W 31] | User-bagged members read -0.0011 val / -0.0018 holdout (l.308) [R]; the model gains +0.0037 val per data doubling, so every subsample costs more than averaging returns |
| Forecasting fresh popularity for test rows | The stale-input cooldown under distillation already has the form "stale inputs, fresh-input teacher" and fails at file level [R ledger 12:51Z]; extrapolated trends read -0.00035 (`EXTERNAL_TOP1_1113.md` l.68) [R] |
| Test-time training on a test user's own history [W 37, snippet] | The masked-history task carries almost no signal here (recall@10 nonzero for 4.3% of feeds, l.290) [R]; retrieval over the user's own history is already a per-user adaptation |
| Probability rescaling for shift (RecSys Challenge 2023 academic winner [W 33]) | No-op under per-feed rank metrics |
| Longer histories (Riiid winner [W 36]) | Closed here for lack of a decidable test and of train / test parity (l.227) [R]; the external evidence is known to the project |

---

## 7. Sources

Depth: R = text opened and read; A = abstract only; S = search snippet only.

| # | Source | Depth | Contribution |
|---|---|---|---|
| 1 | https://ar5iv.labs.arxiv.org/html/2208.02958 (KDD Cup 2022 task 1, 1st) | R | step table: self-distillation + soft pseudo-labels +0.0022; ensemble +0.0035 public / +0.0028 private; no post-processing |
| 2 | https://arxiv.org/html/2409.20483v1 (RecSys Challenge 2024, organisers) | R | top-3 summaries; AUC loss without future-impression features |
| 3 | http://proceedings.mlr.press/v14/chapelle11a/chapelle11a.pdf (Yahoo challenge overview) | R | top-5 gaps, paired tests, per-query oracle 0.4972, "loss of secondary importance", transfer track null |
| 4 | http://proceedings.mlr.press/v14/burges11a/burges11a.pdf (Yahoo, 1st) | R | 12 models, 10-fold bags, plain normalised sum as good as learned combiners, augmented-list members |
| 5 | https://ar5iv.labs.arxiv.org/html/2007.08483 (power laws in deep ensembles) | R | 1/n law of the ensemble metric; prediction from sizes 1-4 |
| 6 | https://msnews.github.io/competition.html (MIND competition) | R | winners and report links |
| 7 | https://msnews.github.io/assets/doc/1.pdf (MIND, 1st) | R | dev next day, test one week later; id masking in dev; +dev data +0.0034 AUC; 4-model mean +0.0029; what failed on test |
| 8 | https://msnews.github.io/assets/doc/3.pdf (MIND, 2nd, Layer 6) | R | probability > logit > rank averaging; tree second stage, features, experts and popularity lose on test |
| 9 | https://github.com/greenwolf-nsk/yandex-cup-2022-recsys (Yandex Cup 2022, 2nd) | R | co-occurrence from train + test; inverse-rank blend of 3 rankers |
| 10 | https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md (Jane Street 2024) | R | 6 models, online update per test day +0.008 on CV |
| 11 | https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ | R | soft and k-fold pseudo-labels, 100-seed curve, full-data refit |
| 12 | https://ar5iv.labs.arxiv.org/html/2210.11934 (fusion functions) | R | normalised score mix beats reciprocal rank fusion by 0.015-0.032 NDCG |
| 13 | https://developer.nvidia.com/blog/pro-tips-for-building-multilingual-recommender-systems/ (KDD Cup 2023) | R | pipeline only; no ensemble numbers |
| 14 | https://ar5iv.labs.arxiv.org/html/1912.02757 (loss landscape of deep ensembles) | R | trajectory snapshots are functionally close; seeds are not |
| 15 | https://ar5iv.labs.arxiv.org/html/2305.12313 (when are ensembles effective) | R | improvement bounded by disagreement over error; low for members that fit their targets |
| 16 | https://ar5iv.labs.arxiv.org/html/2202.06985 (are deep ensembles necessary) | R | no robustness beyond accuracy; a larger single model matches an ensemble |
| 17 | https://ceur-ws.org/Vol-2855/ (Booking challenge proceedings) | R | paper list |
| 18 | https://raw.githubusercontent.com/NVIDIA-Merlin/competitions/main/RecSys2022_Challenge/A-Diverse-Models-Ensemble-for-Fashion-Session-Based-Recommendation.pdf | R | 17 models, three stackers, stage-3 average, scores per stage |
| 19 | https://ceur-ws.org/Vol-2855/challenge_short_7.pdf (Booking challenge, 6th) | R | evaluation-set trips as training data; weighted averaging |
| 20 | https://ar5iv.labs.arxiv.org/html/2012.09816 (ensemble, distillation, self-distillation) | R | ensemble of distilled models does not exceed the ensemble of the originals; multi-view condition |
| 21 | https://ar5iv.labs.arxiv.org/html/2006.13570 (hyper-deep ensembles) | R | hyperparameter diversity: +1.6 points for a weak net, +0.2 for a strong one at 4 members |
| 22 | https://ar5iv.labs.arxiv.org/html/2106.05945 (does distillation really work) | R | more distillation data raises fidelity, not accuracy; student accuracy saturates with teacher size |
| 23 | https://www.cs.toronto.edu/~mvolkovs/Recsys_2023.pdf (RecSys Challenge 2023, 1st, Layer 6) | R | joint supervised + contrastive objective; ablation of three corruption views |
| 24 | https://ar5iv.labs.arxiv.org/html/1809.07428 (ranking distillation) | R | teacher top-K on unlabelled items; gains for small students |
| 25 | https://ar5iv.labs.arxiv.org/html/2106.15147 (feature-corruption contrastive learning) | R | corruption recipe; joint form worse than pre-training; about 2.4% relative with full labels |
| 26 | https://ar5iv.labs.arxiv.org/html/1704.00109 (snapshot ensembles) | R | snapshots comparable to 2 independent models |
| 27 | https://ar5iv.labs.arxiv.org/html/2207.03208 (pre-training objectives for tabular models) | R | contrastive not better than reconstruction; small gains on the largest tables; gains survive ensembling |
| 28 | https://arxiv.org/abs/2511.21032 (temporal shift, resampled drifting factors) | A | training-time augmentation of drifting factors; +2.33% online |
| 29 | https://arxiv.org/abs/2605.15299 (pruning unstable features) | A | engagement features are the unstable ones |
| 30 | https://github.com/logicai-io/recsys2019 (RecSys Challenge 2019, 1st) | R | 37 rankers, weights by direct metric optimisation |
| 31 | https://proceedings.neurips.cc/paper/1995/file/1019c8091693ef5c5f55970346633f92-Paper.pdf (learning with ensembles) | R | under-regularised students are optimal for large ensembles |
| 32 | https://arxiv.org/abs/2511.04160 (joint regularisation in deep ensembles) | A + S | ensemble-tuned weight decay lower than member-tuned; effects vary |
| 33 | https://re.public.polimi.it/retrieve/bc69b04d-7b89-4436-9a46-6cd412b7b344/pessimistic-rescaling-and-distribution-shift-of-boosting-models-for-impression-aware-online-advertising-recommendation.pdf (RecSys Challenge 2023, academic winner) | S | probability rescaling for a log-loss metric |
| 34 | https://medium.com/nvidia-merlin/winning-the-recsys2021-challenge-by-a-diverse-set-of-xgboost-and-neural-network-models-4c5422a642d8 | S | 3 + 3 models, 2 stackers on their predictions |
| 35 | https://blogs.nvidia.com/blog/recommendation-systems-win (KDD Cup 2023) | S | "something like 100 different models" |
| 36 | https://arxiv.org/pdf/2102.05038 (Riiid, 1st) | S | 5 models, history length 1728 |
| 37 | https://arxiv.org/abs/2504.01489 (test-time training for sequential recommendation) | S | per-batch self-supervised adaptation on test histories |
| 38 | https://recsys.acm.org/challenges/ | R | winners and paper titles 2019-2024 |
| 39 | https://github.com/k-fujikawa/recsys-challenge-2024-1st-place | R | file layout only (weighted mean + stacking outputs); details are in the project's `deep_recsys2024.md` |
| 40 | https://arxiv.org/abs/2604.06420 (data scaling for recommenders) | A | accuracy still rises with data at 100M interactions |
| 41 | https://habr.com/ru/articles/925034/ (RecSys Challenge 2025, 2nd overall) | R | embedding concatenation; no ensemble detail |

Not reachable or not found: Kaggle discussion pages (OTTO 2nd place), VK RecSys Challenge 2024 winner, Yandex Cup 2023-2024 recommender winners, Avito ML Cup 2025 write-ups beyond the repositories already in `EXTERNAL_SCAN_1113.md`.

---

## 8. Files

- This report: `<repo>/reports/TRICKS_1115/web_competitions.md`
- Check scripts and log: `<repo>/reports/TRICKS_1115/web_competitions_checks/pool_forms.py`, `pool_forms_holdout.py`, `pool_forms.log`
- Converted source texts (scratch, not part of the project): `/tmp/scratchpad/webcomp/txt/`
