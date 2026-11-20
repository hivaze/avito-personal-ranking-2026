# Web scan: growing or strengthening a deep ensemble and its distilled students cheaply (2019-2026)

Written 2026-11-15 21:28 UTC (`date -u`) by the research scout of TRICKS_1115. CPU only; nothing trained, nothing submitted, no project file outside `reports/TRICKS_1115/` written.
Marks: [M] measured for this report (script and output in `reports/TRICKS_1115/web_ensembles_meas/`), [R] copied from a project record (path given), [W] web source (number in section 7), [I] judgement. ± = 1 SE paired per feed.
Read before searching: `reports/CURRENT_STATUS.md` sections 1-6 and the log of 11-15, `docs/02_SOLUTION.md`, `reports/COVERAGE_MAP.md` (all 350 rows), `reports/RECHECK_1114/kd_literature.md`, `pool_composition.md`, `SYNTHESIS.md` section 5, `reports/STALE_1115/veto.md` section 5.
Sources: 41 pages opened and read through the fetch tool (section 7); 6 more seen as search snippets only and marked so.

## 0. Summary

1. **No technique of this family is expected to move the final file by more than about +0.0002.** The 14 families asked for fall into three groups: variance reduction (snapshot / cyclic / FGE members, BatchEnsemble, MIMO, more seeds), which is exhausted at 16 roots; diversity by construction (hyper-parameter ensembles, negative correlation, multi-teacher, mutual learning), which the literature itself reports as neutral or harmful for high-capacity members and which the project measured as null; and combination rules (greedy selection, stacking), which cannot gain on exchangeable members. The one item with a literature-sized effect is a member from a different training method, which here means the partner merge already on the owner's list.
2. **"Dramatically increase the ensemble" has a measured ceiling.** On the 16 held-out wide roots the size law gives an infinite-root value of +0.00015 val / +0.00013 holdout above the 16-pool [M, section 2.1]. Tripling the pool with two re-cools per root (the cheap, snapshot-style way) is worth +0.00003..+0.00004 val by the same law [M + I], and reads +0.00002 ± 0.00015 val, +0.00013 ± 0.00008 holdout directly [M]. Public cannot see either (paired SE 0.0004-0.0007 [R, STATUS section 1]).
3. **A same-root re-cool is worth 0.31-0.43 of a new root** [M, section 2.1]. Member correlation is 0.9914 for two re-cools of one plateau snapshot and 0.9721 for different roots: 57% to 69% of the member-to-member difference is made before step 77,500. This matches the published picture: models collected along one trajectory disagree less than independent runs (FGE: 14.57% against 19.97% [W35]; Fort et al.: divergence 0.153 against 0.384 [W16]) and "dozens of samples ... are essentially equivalent to a handful of independently trained models" [W1].
4. **Plateau weight averaging: what differs from the literature.** Published gains of averaged members come from runs that have no learning-rate decay (EoA, SWAD) or from fine-tunes of one pre-trained start (soups, DiWA, WiSE-FT). The project's cooldown already does what averaging does there and adds a mean-function gain of +0.0031 ± 0.0006 [R]; averaging three high-rate plateau snapshots replaces neither. The claimed out-of-distribution edge of weight averaging over output averaging is not consistent in the primary sources (section 4.1). No reason to re-open.
5. **Distillation side: nothing new that is not bounded.** Ranking-distillation benchmarks pick the loss the project uses (listwise softmax, labels and teacher both active) [W19]; diversity-preserving and multi-teacher methods never lift a student above its teacher ensemble [W4, W29], while the project's students are already above theirs; self-training gains need 17x-357x unlabelled lists plus noise [W22] against 1.4% here [R]; a fine-tuned teacher cannot be built on train rows and is barred by the leak rule.
6. **Ranked proposals (section 5):** P1 partner merge (owner's open decision; new supporting evidence only), P2 view-matched teacher for the popularity-light students (the only new own-model arm; prior 0.10; the CPU pre-read made for this report does not support it), P3 soup of same-root re-cools as a zero-training closure, P4 the 48-member re-cool file (measured, not recommended), P5 weight-space TO/FT interpolation and P6 disagreement-on-target members (both not recommended). Expected total of the own-model items: below +0.0001.
7. **Honest reading for the owner's request.** The distance to #1 is 0.0033 public; the whole seed-variance headroom of the wide pool is 0.00015. Ensemble and distillation technique cannot close that gap; only different information can (a partner model, or an input the pool lacks).

## 1. Project facts that bind this topic

| fact | value | source |
|---|---|---|
| Size law of the wide-student pool | NDCG(n) = a - b/n to 1e-4 over all 255 subsets of W8; W8 is 0.00024 ± 0.00012 val / 0.00017 ± 0.00007 holdout below a | [R] `reports/RECHECK_1114/pool_composition.md` section 0 item 2 |
| 16 against 8 wide students | val -0.00007 ± 0.00034, public -0.000003 | [R] `docs/02_SOLUTION.md` section 11 |
| One mean function | every d192 construction (teacher 11 / 25 / students as teacher, teacher weight 0.3 or 0.5, no dropout, 20 epochs, FFN x4, 3 blocks) has the same infinite-seed value 0.4014-0.4021 val / 0.3246-0.3249 holdout | [R] `pool_composition.md` section 0 item 3 |
| A pool of mixed hyper-parameter variants (kw4 = 4 variants x 2 seeds) | -0.00005 ± 0.00058 val / +0.00010 ± 0.00032 holdout against W8 | [R] `pool_composition.md` section 2 |
| Teacher identity is not a diversity source | student-student correlation 0.9718 (different teacher) against 0.9722 (same teacher) | [R] `reports/RECHECK_1114/kd_literature.md` M6 |
| Plateau weight averaging | member +0.0004 ± 0.0004 val; correlation 0.975 against 0.941; 7-member ensemble -0.0025 val / -0.0016 holdout | [R] `reports/COVERAGE_MAP.md` row "plateau weight averaging" |
| Cooldown changes the mean function | 7-seed ensemble cooled minus plateau end +0.0031 ± 0.0006 | [R] `COVERAGE_MAP.md` row "cooldown improves the mean function" |
| Higher teacher weight | member up, pool down: kd08w192 8-pool -0.00049 ± 0.00048 val / -0.00029 holdout; member correlation 0.990 | [R] STATUS log 2026-11-15 07:53Z; `kd_literature.md` section 2 |
| Matching is not input-limited | students match the teacher equally on seen and unseen feeds (0.9933 against 0.9933) | [R] `reports/RECHECK_1114/long_bets.md` line 11 |
| Fitted weights and selection | 0 of 78 second-level combiners pass; val-fitted forward selection reads -0.00095 ± 0.00028 on the holdout | [R] `COVERAGE_MAP.md` row "second-level combiners"; `pool_composition.md` section 0 item 6 |
| The one view ensemble that works | popularity-light admixture at 0.2: public +0.00052 (W8P20 - W8), +0.00041 (FD16P20 - W16) | [R] `docs/02_SOLUTION.md` section 11 |
| BatchEnsemble in the ranker (TabM-mini, k = 4) | +0.00009 ± 0.00042 val at 1.45x cost; members 0.999-correlated | [R] `reports/nnx_exp_feat.md` line 20 |

## 2. Measurements made for this report

All three scripts are CPU only and read stored prediction files; they write nothing outside their own directory. Environment: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=<scratchpad>`, run from `<repo>` with `.venv/bin/python`. Loader check: the 16-pool of the original students at the last step reads 0.40157 on val, the value on record (`docs/02_SOLUTION.md` section 10.1, A3) [M]; the wide popularity-light pool at 0.2 on W16 reads val +0.00047 ± 0.00031, holdout -0.00046 ± 0.00022, the values of `COVERAGE_MAP.md` row "Wide pop1 students" [M]. These are descriptive reads: nothing was selected from them, and the number of statistics printed is about 30 (M1), 36 (M1b) and 60 (M2).

### 2.1 M1: what a same-root re-cool adds to a pool (`m1_recool_growth.py` -> `m1_recool_growth.txt`; `m1b_pairings.py` -> `m1b_pairings.txt`)

Members: the 16 held-out wide students (`w5_e10_kd05w192_s951-958`, `w5_e10_kd05t25w192_s951-958`) and two m02 re-cools of each one's step-77,500 snapshot with different seeds (`w5_{k,l}cd_ref_s*`, `w5_{k,l}cd_ref2_s*`), all at the last step, train-only. Witnesses: val, in-period holdout, stale-val (scenario lt3, `nn_x/w12/stale/hz2n_review/preds_stale_lt3/`).

| quantity | val | holdout | stale-val |
|---|---|---|---|
| member correlation, two re-cools of one root | 0.9914 | 0.9919 | 0.9905 |
| member correlation, different roots | 0.9721 | 0.9735 | 0.9691 |
| share of (1 - correlation) regenerated by a re-cool | 0.31 | 0.31 | 0.31 |
| 2-pool gain over a single member, same root | +0.00046 ± 0.00007 | +0.00033 ± 0.00004 | +0.00039 ± 0.00007 |
| 2-pool gain over a single member, different roots | +0.00106 ± 0.00013 | +0.00106 ± 0.00007 | +0.00106 ± 0.00013 |
| ratio (worth of a re-cool in new roots) | 0.43 | 0.31 | 0.37 |
| 32-pool (16 roots x 2 re-cools) minus the mean of its two 16-pools | +0.00002 ± 0.00015 | +0.00013 ± 0.00008 | +0.00034 ± 0.00015 |
| the same for the two other pairings (re-cool + original) | +0.00002, 0.00000 | +0.00019, +0.00013 | +0.00008, +0.00013 |
| 16 members on 16 roots minus 16 members on 8 roots (three pairings) | +0.00033, +0.00024, +0.00054 (± 0.00019) | +0.00025, +0.00022, +0.00025 (± 0.00011) | +0.00019, +0.00033, 0.00000 (± 0.00019) |
| 16 roots minus 8 roots, one member each | +0.00016 ± 0.00014 | +0.00011 ± 0.00008 | +0.00011 ± 0.00014 |

Reading [M unless marked]:
- Size law on these members: member 0.39923, 16-pool 0.40150 (mean of the two re-cool pools) gives a = 0.40165, b = 0.00242 on val; holdout a = 0.32439, b = 0.00209. Infinite roots add b/16 = +0.00015 val / +0.00013 holdout to a 16-pool.
- With the cooldown share c = 0.31-0.43 of member noise, k re-cools per root give an effective size 16 / (1 - c + c/k). For k = 3 (48 members) this is 20-22 roots: +0.00003..+0.00004 val [I from the law]. For k infinite: 23-28 roots, +0.00005..+0.00007.
- The single stale-val read of +0.00034 (z 2.3) is not repeated by the two other pairings (+0.00008, +0.00013) or inside the two teacher groups (+0.00009, -0.00006); the per-feed SE understates the scatter between member sets (unions of two 8-pools read from -0.00026 to +0.00019 against their mean on val). It is treated as noise.
- Roots matter, re-cools hardly: at equal member count 16, 16 roots beat 8 roots x 2 by +0.0002..+0.0005 val.
- Cost on record of one d192 re-cool: 47-58 minutes logged (`nn_x/w5/logs/w5_kcd_ref_s95*.json`, `w5_fdxhz_kd05w192_s95*.json`, field `train_minutes`) [M], against 221-259 minutes for a full student [R, `docs/02_SOLUTION.md` section 12]: 0.23 of the cost for 0.31-0.43 of the variance reduction. Cheap members are cost-efficient and still worthless at 16 roots, because the variance they remove is already gone.

### 2.2 M2: which popularity-light pool is the better 0.2 partner (`m2_poplight_partner.py` -> `m2_poplight_partner.txt`)

Base: the W16 train-only pool (val 0.40144, holdout 0.32460). Partner pools, all held-out, train-only, selected checkpoints.

| partner pool | member val | pool val | member correlation to the base pool | pool correlation to the base pool | admixture at 0.2: val | pmix-private | holdout (biased against the view) |
|---|---|---|---|---|---|---|---|
| HL4 hard-label pop1 s951-954 (d128, no teacher) | 0.39418 | 0.39747 | 0.9581 | 0.9769 | +0.00015 ± 0.00036 | +0.00014 ± 0.00056 | -0.00049 ± 0.00023 |
| HL4b hard-label pop1 s955-958 | 0.39516 | 0.39843 | 0.9585 | 0.9773 | +0.00051 ± 0.00035 | +0.00070 ± 0.00054 | -0.00065 ± 0.00022 |
| HL8 hard-label pop1 s951-958 | 0.39467 | 0.39841 | 0.9583 | 0.9805 | +0.00031 ± 0.00035 | +0.00031 ± 0.00053 | -0.00066 ± 0.00023 |
| KDn4 pop1 students d128, teacher_mean25 | 0.39821 | 0.39943 | 0.9746 | 0.9835 | +0.00008 ± 0.00032 | +0.00026 ± 0.00049 | -0.00056 ± 0.00022 |
| KDw4 pop1 students d192, teacher_mean11 | 0.39859 | 0.40000 | 0.9737 | 0.9844 | +0.00047 ± 0.00031 | +0.00050 ± 0.00048 | -0.00046 ± 0.00022 |

Reading [M]: distillation from a full-view teacher raises a popularity-light member by 0.004 and its correlation with the wide pool from 0.958 to 0.974. As 0.2 partners the weaker, less correlated hard-label pool and the stronger, more correlated student pool are equal inside noise (HL8 - KDw4: val -0.00015 ± 0.00027, pmix-private -0.00019 ± 0.00040, holdout -0.00020 ± 0.00015). Two draws of the same hard-label recipe differ by 0.00036 (HL4 against HL4b), which is the size of every contrast in the table. The read neither supports nor excludes proposal P2.

### 2.3 Disk facts used by the proposals [M]

- Fine-tune checkpoints are not stored: `ls nn_x/w5/ckpt nn_x/abl/ckpt | grep n2cpv3` returns 0 files; `finetune.py` saves predictions only. Any weight-space operation on fine-tuned models needs the fine-tunes to be run again.
- Re-cool checkpoints of two seeds per root exist for all 16 held-out roots (`nn_x/w5/ckpt/w5_{k,l}cd_ref_s95*.pt`, `..._ref2_...`).
- Sums of the hard-label popularity-light teacher members exist: `nn_x/w12/kd/out/teacher_p1.npy`, `teacher_p2.npy` (4 members each, files `.n` = 4).

## 3. Technique by technique

"Gain" is the gain reported by the source over a plain seed ensemble of the same size, unless the row says otherwise. "16+" answers whether the evidence covers a base ensemble of 16 or more members.

| # | family | reported gain [W] | does it survive at 16+ members | status in this project |
|---|---|---|---|---|
| 1 | Ensemble size itself | Loss follows c + b·n^a with a close to -1 [W2]; "most of the gains ... with only 5 models", "diminishing returns beyond size 5" [W27]; 100 seeds of one tree model: MAP@3 about 0.376 -> 0.379 against a single seed [W23] | By the law the remaining gain at n = 16 is b/16 | Law holds to 1e-4 [R]; +0.00015 val left at 16 roots [M] |
| 2 | Snapshot ensembles, cyclic schedules | Against the single model: CIFAR-100 error 19.25 -> 17.41 (DenseNet-100); "comparable with ensembling of 2 independent models"; 4-8 cycles useful [W34] | No. Below a seed ensemble of equal member count; methods bound to one trajectory saturate [W1] | The re-cool harness is this technique. One re-cool = 0.31-0.43 of a new root [M]; 48 members = 20-22 roots [I] |
| 3 | Fast geometric ensembles | At equal training budget FGE beats independent ensembles (CIFAR-100 WRN: 16.95 / 16.88 against 17.48 / 17.01 at 2x / 3x budget) because it has more members per budget; two models of one FGE run disagree on 14.57% against 19.97% for independent runs [W35] | No evidence beyond a handful of budgets; the advantage is cost, not level | Cost is not binding and N is saturated: closed by M1 |
| 4 | Deep-ensemble equivalents of cheap ensembles | "many popular ensembling techniques require dozens of samples ... yet are essentially equivalent to a handful of independently trained models"; deep ensembles "dominate other methods given a fixed test-time budget" [W1]; within-trajectory samples are "very similar in predictions" [W16] | No | Same reading as M1 |
| 5 | Model soups (weights of fine-tunes of one start) | Greedy soup 81.03 against logit ensemble 81.19 in distribution; 50.75 against 50.77 under shift; uniform soup 79.97 (below the best single model 80.38) in distribution, 51.45 under shift [W6] | Not a gain over the output ensemble; it is the ensemble at one model's inference cost | Inference cost is free here; section 4.1; P3 is the closure test |
| 6 | DiWA, model averaging of diverse fine-tunes | DomainBed: 66.8 against output ensemble 67.1 (random head), 67.6 against 67.2 (shared linear-probe head), M = 20; needs "shared initialization and mild hyperparameter ranges" [W7] | Mixed sign at M = 20 | No shared pre-trained start exists except the two cases of section 4.1 |
| 7 | Ensemble of averages, SWAD | Ensemble of 6 averaged models 68.0 against 66.8 for 6 unaveraged ones; single averaged 66.5, single plain 64.0; constant learning rate, averaging from iteration 100 [W3]. SWAD +3.6 points over ERM on DomainBed, +0.5 on ImageNet [W25] | Tested at 6 members; the comparison members are un-annealed | The cooldown is the annealer here (+0.0031 at ensemble level [R]); plateau averaging in place of it -0.0025 [R] |
| 8 | MultiSWA / MultiSWAG (ensembles of independently trained SWA solutions) | "outperform Deep Ensembles for a small number of independent models"; under high corruption "even for many independent models"; no numbers in the text [W15] | Stated for few models | Stage-1 weight EMA +0.00003 and tail EMA at most +0.0001: closed [R, `COVERAGE_MAP.md` row "stage-1 weight EMA"; `reports/TRICKS_SWEEP_1113.md` section 4.1 row "Snapshot ensembles / SWAG / Lookahead"] |
| 9 | PAPA (population pushed toward its average) | "Baseline ensembles have higher accuracy than PAPA variants" (CIFAR-100 ResNet-18: 84.3 against 82.2); feature similarity 88-99.8% against 31-63% for independent models [W18] | No | Not applicable |
| 10 | Weight averaging under temporal shift | SWA against ERM on six time-shift tasks: better on 1, tied on 2, worse on 3 [W28] | - | No support for re-opening averaging as a staleness lever |
| 11 | BatchEnsemble, MIMO | BatchEnsemble (4) 80.32 against naive ensemble (4) 81.02 on CIFAR-100 [W30]; MIMO (3) 81.98 against deep ensemble (4) 82.7; "the ensemble peaks between M=2 and M=4" [W17] | No: below a seed ensemble at size 4 | TabM-mini k = 4: +0.00009 ± 0.00042, members 0.999-correlated [R] |
| 12 | TabM (k implicit members, shared weights) | "weak individually, but powerful collectively"; claimed above k independent MLPs, numbers only in figures; "too high values of k can be detrimental"; TabM still gains from a deep ensemble of 5 [W11] | k = 32 by default inside one model; no result on top of 16 independent models | As row 11; a full-adapter form was never run (prior 0.03 [I]) |
| 13 | Hyper-parameter ensembles | Size 4: +0.2 to +0.3 points accuracy for wide networks, +1.6 for ResNet-20 on CIFAR-100; hyper-batch against batch ensemble 0 to +0.2 [W10] | Tables stop at size 4 | Measured: kw4 pool = W8 (-0.00005 ± 0.00058 / +0.00010 ± 0.00032) [R] |
| 14 | Members from a different training method or data | Error inconsistency with a ResNet-50: re-initialisation 0.085, hyper-parameters 0.086-0.106, architecture 0.114, framework 0.128, dataset 0.169-0.230; same-method pair +1.1 points, a similarly accurate model from other data "nearly 7%" [W14] | The only diversity source whose gain does not shrink with more seeds | Own families all sit on one frontier [R]; the partner merge is this lever (P1) |
| 15 | Diversity for shift, trained on unlabelled target inputs (D-BAT) | Ensembles of 2: +0.7 (Office-Home), +1.7 (Waterbirds), +11.0 (Camelyon17) when the unlabelled set is the test set; +0.6 / +5.0 when it is not [W20] | Only sizes 2-5 shown | The popularity-light admixture is the hand-built form (+0.0005 public, four readings [R]); P6 |
| 16 | Negative correlation, joint training | Joint training of the ensemble loss is worse than independent training in every case (ImageNet ResNet-18 x5: 62.74 against 68.10) [W9]; for large networks encouraging diversity "consistently leads to worse ensemble performance" (61% of 134 ensembles worse), discouraging it is neutral or better in 74% of 149 [W8]; M = 16 on CIFAR-100: small networks want joint training, large ones want weight 0-0.7 on the ensemble term [W37]; "we should not be 'maximising diversity'" [W26] | W37 is at 16 members and gives no gain figure for large networks | Consistent with the record: lower correlation from noise buys nothing at the asymptote [R]; R-Drop / SAM / dropout arms closed [R] |
| 17 | Deep mutual learning, online co-distillation | Jointly trained CTR sub-networks get worse with size (Criteo AUC 81.39 -> 81.25 at 10 sub-networks); ensemble-mean teacher plus mutual terms 81.60 at 10, flat after 5 (81.58) [W5]; student of a 3-teacher ensemble beats a single teacher by 12.5 per-mille AUC, more teachers "the increase diminishes" [W36] | W5 stops at 10 and fixes a collapse that independent training does not have | Offline ensemble distillation at weight 0.5 is adopted; more teacher share raises correlation and lowers the pool [R] |
| 18 | Multi-teacher and diversity-preserving distillation | Best student stays below its teacher ensemble: 94.18 against 94.78 (CIFAR-10, 8 teachers), 81.88 against 82.52 (CIFAR-100) [W4]; averaged-subnetwork student 80.02 against 81.37 [W29]; label-based teacher weights +0.2..+2.0 over plain averaging with 3 teachers and a smaller student [W39]; partitioned transfer between trained models +0.3..+1.0 [W33] | Goal is compression; no source shows a student pool above a teacher pool | Students already exceed the teacher (+0.0010 val for W8) [R]; teacher identity is not a diversity source [R] |
| 19 | Ensemble distribution distillation, multi-head students | Accuracy equal to plain ensemble distillation (27.9 against 28.0 error) | - | [R] `kd_literature.md` section 3: precondition fails (M6) |
| 20 | Listwise / ranking distillation losses, temperature | Listwise softmax has the best mean rank (1.8) of six losses; "best performance is achieved when both objectives are active"; the softmax transform of teacher scores helps one loss and hurts another (42.37 against 40.03; 54.87 against 58.49); all students are smaller than the teacher and none beats it on the main metric [W19] | - | The adopted objective is the benchmark's winner; kwT07 / kwT15 closed; teacher flat and calibrated (T* 1.10) [R] |
| 21 | Repeated self-distillation | Up to a factor d in linear regression when singular values are distinct and the signal sits on the leading direction; no gain when they are equal; real data: -47% MSE on one set, 0 on another; needs a tuned, possibly negative or above-one teacher weight per step [W24] | - | Generation 2 closed (+0.00001 ± 0.00051 / -0.00021 ± 0.00029) [R]; teacher weight 1.0 lowers the pool [R] |
| 22 | Distilling from a fine-tuned teacher | No head-to-head study found (search only). Distilling into an already trained student helps in fewer than 40% of teacher-student pairs without partitioning [W33] | - | Not applicable: section 4.3 |
| 23 | Self-training with noisy students on unlabelled lists | Same-architecture retriever: MRR@10 37.8 -> 39.4 with 17.6x synthetic queries; without noise 38.8; with hard labels 38.8; a second round adds nothing; +3.0 nDCG@10 out of domain [W22]. Playbook: soft labels, k sets of pseudo-labels so that validation rows never see labels from models trained on them [W23]. A 2024 competition winner distilled 70B-class fold models into a 9B student [W38] | - | Unlabelled in-period lists = 1.4% of train; matching is not input-limited; U20 at 2 seeds not GO [R]. Section 4.4 |
| 24 | Greedy ensemble selection (Caruana) and its controls | Selection with replacement gives non-negative weights that sum to one and a sparse ensemble; dense numerical weights overfit a ranking metric (rank 1.0 on validation -> 2.0 on test for ROC AUC) until they are normalised and trimmed [W21]; used with stratified seeds in hyper-ensembles [W10]; final layer of multi-layer stacks [W40] | Works on libraries of different models | Members are exchangeable seeds: any selection is a draw pick (two failed replications on 11-15 [R]); val-fitted selection -0.00095 on the holdout [R] |
| 25 | Stacking of rank scores with strong regularisation | A convex combination of normalised scores beats reciprocal-rank fusion in and out of domain (0.454 against 0.425 on MS MARCO); one weight, stable in 0.6-0.8, tuned with under 5% of the queries; z-score and min-max forms are rank-equivalent up to the weight [W31]. Stackers need out-of-fold inputs and repeated bagging; over-fitting compounds across layers [W40] | - | The file is this form with two fixed weights (0.8 / 0.2, 0.5 / 0.5); 0 of 78 fitted combiners and 0 of 95 rule families pass [R] |
| 26 | Ensembles against one larger model; when ensembles help | A larger single model reproduces an ensemble's gains; ensembles show no extra robustness beyond their in-distribution accuracy [W12]; ensemble improvement tracks disagreement relative to error, low for interpolating networks, high for trees [W13] | - | Width 192 was the one capacity step that moved the asymptote; 224 and 256 closed [R] |

## 4. Four questions the task asked to be answered explicitly

### 4.1 Plain plateau weight averaging hurt at ensemble level here; what differs from soups, SWA variants and the ensemble of averages

1. **What is averaged.** In [W3] and [W25] the learning rate is constant, so the average is the only annealing the run gets; the ensemble it is compared with (66.8) consists of un-annealed members. In [W6], [W7] and [W32] the averaged models are low-rate fine-tunes of one pre-trained start, which "lie in the same basin"; models trained from scratch with different data orders interpolate to chance [W6, W32]. The project averaged three snapshots of the high-rate plateau (lr 2e-3) of a from-scratch run and compared the result with the cooled checkpoint of the same run.
2. **What the average replaces.** The cooldown is worth +0.0031 ± 0.0006 at ensemble level with unchanged seed correlation [R]: it is a change of the mean function. The plateau average is a de-noised plateau point; it gains +0.0004 as a member and loses the cooldown gain in the pool (-0.0025) [R].
3. **Where the diversity sits.** 57% to 69% of the member-to-member difference is fixed before the cooldown starts [M, 2.1]. Averaging plateau snapshots removes exactly that part (correlation 0.941 -> 0.975 [R]).
4. **The soup approximation.** For two models the soup loss differs from the output-ensemble loss by a curvature term that favours the soup and a variance term that "always favors the ensemble" [W6]; to first order a weight average of close models equals their output average [W7]. A soup therefore buys an ensemble at one model's inference cost. Inference cost is not a constraint here, so a soup can at best tie the ensemble of its ingredients.
5. **The ingredients that meet the precondition here.** (a) The 15 fine-tune models of a member (one start, lr 5e-5): their output mean is already the test prediction. (b) Re-cools of one plateau snapshot: their output ensemble is worth +0.00003..+0.00004 on top of 16 roots [M + I]. Neither can move the file.
6. **Under shift.** The record judged averaging on in-period witnesses only, so the shift claim was checked in the sources: greedy soup 50.75 against ensemble 50.77, uniform soup 51.45 [W6]; DiWA 66.8 against 67.1 and 67.6 against 67.2 [W7]; SWA under temporal shift better on 1 of 6 tasks and worse on 3 [W28]; MultiSWAG's edge is stated for few models [W15]. No consistent out-of-distribution advantage of weight averaging over output averaging exists to re-open the row. P3 is offered only as a zero-training closure.

### 4.2 Snapshot ensembles and cyclic schedules against the re-cool harness

A re-cool of a stored plateau snapshot with a new seed is one cycle of a snapshot ensemble with a restart from the plateau rate. The literature's own caveat (snapshots of one run are worth about 2 independent models [W34]; a handful [W1]) is reproduced: 3 members per root are worth 1.26-1.40 roots [M + I from 2.1]. Larger restart rates give more diverse snapshots [W34]; the limit of that direction is a new root. Under hard labels the project already used such members (F7x70 next to F7x in G3 [R, `docs/02_SOLUTION.md` section 14]). At 16 roots the question is closed by the size law, not by cost.

### 4.3 Distilling from a fine-tuned teacher

The fine-tuned models exist only on val and test rows: their 42 extra inputs are val/test-only and the `pre_*` family is barred from train rows as a leak [R, `docs/02_SOLUTION.md` sections 4.4-4.5]. A fine-tuned teacher on train rows cannot be built, and "no fine-tuned model enters a teacher" is a rule on record [R, section 4.5]. On val rows the form was run twice as the soft-target fine-tune (-0.00036 ± 0.00031, +0.00002 ± 0.00044 at file level [R, `COVERAGE_MAP.md` row "Soft-target val fine-tune"]). On test rows it is transduction on the pool's own predictions, with the same-user later-snapshot problem. The literature adds no new reason: no source compares fine-tuned and non-fine-tuned teachers under shift (search only), and distilling into a trained student degrades it in most pairings unless the transfer is partitioned by which model is better per sample [W33], which on train rows is a label-conditional target (closed class [R, SYNTHESIS section 5, "co-click target student"]).

### 4.4 Self-training with noisy students on unlabelled lists

The published same-architecture gain (+1.6 MRR@10) needs three things at once: 17.6x to 357x more unlabelled lists than labelled ones, input noise (worth 0.6 of the 1.6), and soft labels (worth 0.6) [W22]. Here every train feed is labelled, the in-period unlabelled lists are 1.4% of train [R], history dropout as noise is closed (kwh30 -0.00098 ± 0.00058 / -0.00066 ± 0.00033 [R]), dropout 0.2 is closed (pooled z -1.88 [R, STATUS log 20:49Z]), and students match the teacher equally on unseen feeds [R]. The U20 arm (2 seeds: val +0.00158 ± 0.00071, holdout -0.00017 ± 0.00039; 2-pool holdout -0.00007 ± 0.00044 [R, `reports/NEXT_1115/u20wit.md`]) is the project's own test and is still queued for seeds 953-954; this scan gives no reason to raise its priority or to add arms to it. The large competition gains from pseudo-labels (bird audio 2024-2025 [W23, W41]) come from an unlabelled set from the test domain that is many times the labelled one; that set does not exist here.

## 5. Ranked proposals

Ranked by expected gain on the final file times probability. Order of use [I]: P1 is a decision, not a run; P3 costs under 1 GPU-h and can go first; P2 is the only arm worth 37 GPU-h and only when a lane is idle; P4-P6 are recorded so that they are not proposed again. Timings: 221-259 minutes per full d192 student, 15-22 minutes per fine-tune [R, `docs/02_SOLUTION.md` section 12]; 47-58 minutes per re-cool [M]. Every rule below is to be copied into `nn_x/w5/logs/E_kd_predeclared.txt` with `date -u` by the lead before the first run; this report does not edit that file.

### P1. MERGE-DIV: a partner file from a different training method (owner's open decision; new evidence only)

- **What.** The equal z-mix with a partner team's model scores under `reports/MERGE_PROTOCOL.md`. Nothing new is built.
- **Why it is listed here.** It is the only member of the "grow the ensemble" family whose published effect does not vanish at 16 members: error inconsistency rises from 0.085 (re-initialisation) to 0.114 (architecture), 0.128 (training framework) and 0.169-0.230 (other data), and the ensemble gain from +1.1 points to about +7 [W14]; diversity added to a fixed recipe by regularisers hurts high-capacity members [W8, W9]. The project's frontier law says the same from its own families [R].
- **Closest closed rows and the difference.** "GBDT members, stackers" and "CatBoost / tree member, round 2" (own trees, 0.017 below W8, input-limited [R]). Difference: a model of equal strength built by other people on other features, not a weak own model on the same inputs.
- **Cost.** 0 GPU-h; the owner's negotiation and a rules check; `src/merge_check2.py`.
- **Expected file gain.** +0.0010..+0.0026 for a partner at correlation 0.86-0.90 and equal strength; P(>= +0.001 given a suitable partner) 0.25-0.35 [R, MERGE_PROTOCOL section 0 item 7]. P(a suitable partner agrees) is not on record; 0.3 [I]. Expected value about +0.0002..+0.0004.
- **Test rule.** The three gates of MERGE_PROTOCOL as written: public gap within 0.002; law on record minus 0.0005 not negative; the equal mix gains at least 2 paired SE on val; no weight fitted.

### P2. VIEW-TEACHER: popularity-light students distilled from a teacher of their own view

- **What.** Today the 4 popularity-light students of FD16P20 are distilled from `teacher_mean11`, a full-view teacher [R, `docs/02_SOLUTION.md` section 7]. Train 8 popularity-light students (seeds 951-958) with a teacher made only of popularity-light members.
- **Attachment.** (1) CPU, 10 minutes: `teacher_pop1_mean8.npy` = per-feed-centred (teacher_p1 + teacher_p2) / 8 from `nn_x/w12/kd/out/`, written to a new file; check its holdout NDCG against the hard-label pop1 8-pool (0.31717 [M, 2.2]). (2) A copy of `nn_x/w12/stale/fulldata/lane_full_p2.sh` (write-new) with a stage-P arm `pop1kd05vtw192`: the `P:pop1kd05w192` command with `--kd_file .../teacher_pop1_mean8.npy`. (3) Fine-tune n2cpv3s3 through `fd_ft_wait.sh`. (4) New keys in a copy of `build_r4.py`: FD16 + 4 and FD16 + 8 of the new students at weight 0.2.
- **Mechanism.** The admixture pays through errors the wide pool does not share. Distillation toward a full-view teacher raises a popularity-light member's correlation with the wide pool from 0.958 to 0.974 [M, 2.2]. Students trained toward another model's outputs lose their own complementary behaviour [W4, W33]. A teacher of the same view keeps the member gain of distillation (variance removal) without pulling the student toward the full-view function.
- **Evidence for.** The mechanism above; the teacher-mean25 students (8 of 25 teacher members popularity-light) read val +0.00056 ± 0.00044, cold-positive feeds +0.00282 ± 0.00084 against their mean11 twins as wide students [R, `reports/STALE_1115/veto.md` re-open list item 7].
- **Evidence against.** M2: the less correlated hard-label pool is not a better partner than the student pool (HL8 - KDw4 at 0.2: val -0.00015 ± 0.00027, pmix-private -0.00019 ± 0.00040). Teacher identity never changed wide students [R, M6]. The arm th read stale-val +0.00008 [R]. The popularity-light teacher pool is weaker (holdout 0.3172 against 0.3237), below the teacher range in which students were shown insensitive (0.3210-0.3245 [R]).
- **Closest closed rows and the difference.** Rows "KD dose and form ... teacher 25 members" and "xd30 ... th popularity-light teacher dose" (a partly popularity-light teacher for WIDE students, judged as a staleness lever for the wide pool); rows "More wide full-data popularity-light students" and "View diversity inside the popularity-light pool" (more students or other views with the SAME full-view teacher). Difference: the student is the popularity-light one, the teacher is of its own view, and the read is the independence of the 0.2 partner, not the level of a wide student.
- **Cost.** 8 students x about 4.2 GPU-h = 34 GPU-h, 8 fine-tunes = 2.7 GPU-h, teacher file on CPU.
- **Expected file gain.** +0.0002 if the mechanism holds [I]; probability 0.10 [I]; expected value +0.00002.
- **Test rule (one look, written before the launch).** Two independent 4-student draws A (s951-954) and B (s955-958). Controls = the 12 existing full-data `abl_fd10_pop1kd05w192` students. PASS iff all four hold: (a) the 8-student file minus the mean of the control 8-student files (all 495 control 8-subsets, or 70 sampled as on 11-15) is >= +1.5 paired SE on final val; (b) EACH draw's 4-student file is above the median of the 70 control 4-pools; (c) the pmix-private delta of (a) is >= 0; (d) mechanism check: mean member correlation with the FD16 pool is at least 0.005 below that of the control members on val. Any miss closes the arm; a pass makes a final-pick candidate, not a claimed gain. Quote the number of things tried as 1 arm, 2 draws.

### P3. SOUP-RC: uniform weight average of the two stored re-cools of each root (zero-training closure)

- **What.** For each of the 16 held-out roots average the weights of `w5_{k,l}cd_ref_s*@last.pt` and `..._ref2_s*@last.pt`, score val, holdout and stale-val, and compare the 16-pool of soups with the 32-pool of their ingredients.
- **Attachment.** A new script that loads two checkpoints, averages every floating-point tensor, and calls the harness's predict path (GPU inference of 16 models, or the CPU replay used on 11-12, which matched GPU predictions at correlation 0.99994 [R, STATUS log 02:35]); outputs under a new directory, never under `preds/`.
- **Mechanism.** Shared start and a low-rate trajectory are the soup precondition [W6, W7]; the two re-cools of a root correlate 0.9914 [M].
- **Closest closed row and the difference.** "plateau weight averaging" (snapshots of the high-rate plateau instead of the cooled model, in-period witnesses only). Difference: cooled endpoints of one start, judged also on stale-val.
- **Cost.** No training; under 1 GPU-h of inference, or a few CPU-hours.
- **Expected file gain.** 0 [I]; probability of >= +0.0002 at pool level 0.03 [I]. Value: it closes "weight averaging under shift" with a number.
- **Test rule.** OPEN iff the 16-pool of soups minus the 32-pool of ingredients is >= +2 paired SE on val AND >= 0 on stale-val AND >= 0 on the holdout; otherwise closed for good, including soups of fine-tune models.

### P4. RECOOL-48: three members per root in the wide pool (measured; not recommended)

- **What.** Two plain m02 re-cools of the step-77,500 snapshot of each of the 16 full-data wide students (snapshots exist, `nn_x/w5/ckpt/w5_fd10_*@77500.pt` [R, `docs/02_SOLUTION.md` section 13.2]), fine-tuned, pooled member-equal with the originals: 48 + 4 members.
- **Mechanism.** Removal of the cooldown share of member noise.
- **Evidence.** M1: +0.00002 ± 0.00015 val, +0.00013 ± 0.00008 holdout, stale-val +0.0001..+0.0003 and inconsistent; by the size law +0.00003..+0.00004.
- **Closest closed row and the difference.** "More same-recipe student seeds (W16, W24, W32)" (new roots). Difference: re-cools at 0.23 of the cost; the measurement shows they are worth 0.31-0.43 of a root, so the row's conclusion carries over.
- **Cost.** 32 re-cools x 0.9 GPU-h + 32 fine-tunes x 0.33 GPU-h = about 39 GPU-h.
- **Expected file gain.** +0.00003 val; public cannot detect it.
- **Test rule (only if GPUs are idle and the owner wants the larger file as a pick).** The file replaces FD16P20 as a pick iff its final val is >= FD16P20 AND its stale-val delta is >= 0 AND the subset curve over the number of re-cools per root (1, 2, 3) does not turn down. No public slot is spent on it.

### P5. WISE-TF: weight-space interpolation between a member and its fine-tune models (not recommended)

- **What.** For each fold model, weights (1 - alpha) x train-only + alpha x fine-tuned with alpha 0.5, in place of the output mix 0.5 z(FT) + 0.5 z(TO).
- **Mechanism and evidence.** The precondition holds (one start, lr 5e-5); the published gain is against the fine-tuned model alone (+8.3 points under shift at alpha 0.5), and output-space mixing is reported to mitigate the same trade-off [W32]. The file already mixes outputs. To first order the two forms are equal [W7].
- **Closest closed row and the difference.** "WiSE-FT" in the tricks sweep (dropped without a run: FT branch saturated). Difference: none in mechanism; the only new fact is that the fine-tune half is worth +0.00048 on public while val reads +0.00009 [R], which leaves the form unjudgeable offline.
- **Cost.** Fine-tune checkpoints are not stored [M]: 20 fine-tunes again with checkpoint saving (about 7 GPU-h) plus inference.
- **Expected file gain.** 0 [I]; probability of >= +0.0002: 0.04 [I].
- **Test rule.** Not proposed for a run. If run: one alpha (0.5), out-of-fold val of the interpolated fold models against the stored 0.5 / 0.5 file, PASS at >= +2 paired SE with pmix-private >= 0.

### P6. DBAT-STALE: a member trained to disagree with the wide pool on unlabelled test-like inputs (not recommended)

- **What.** The D-BAT loss [W20] with test feeds, or train feeds under the hz2 stale view, as the unlabelled set.
- **Why not.** With test feeds it trains on later snapshots of users who also have earlier test feeds: the banned channel [R, STATUS section 3]. With stale-view train feeds it is a harder-to-control form of the popularity-light view and of the hz arms, which passed at pool level and failed at file level three times [R]. Published gains are for strong spurious features at ensemble size 2.
- **Closest closed rows.** "hz popularity-parity cooldown under distillation"; "TTT / Mean-Teacher on val+test".
- **Cost.** A new loss in a copy of the trainer, 8 students: about 35 GPU-h.
- **Expected file gain.** Probability of >= +0.0003: 0.03 [I].
- **Test rule.** Not proposed for a run.

## 6. Looked tempting, not proposed

| idea | why not |
|---|---|
| One student per teacher member (one-to-one distillation as in [W4]) | The student's mean function is close to linear in its target, so the infinite-seed value is the one of distilling the mean; teacher identity adds no diversity [R, M6] |
| Label-based per-sample teacher weights [W39, W5 "examination"] | A label-conditional target; on in-sample train rows it up-weights the teachers that memorised the feed; teacher softmax is flat (entropy 4.41 of 4.90 nats [R]) |
| Extrapolated teacher weight (above 1 or negative per step) [W24] | Teacher weight 1.0 already lowers the pool's infinite-seed value (-0.00230 ± 0.00062 val / -0.00087 ± 0.00036 holdout against 0.5 [R, `pool_composition.md` section 2]) |
| Full TabM adapters in every linear layer [W11] | k = 4 first-layer form gave 0.999-correlated members [R]; no source shows a gain on top of 16 independent models |
| Greedy selection with replacement, bagged selection [W21, W40] | Exchangeable members: the optimum is equal weights; two selected pools failed replication on 11-15 [R] |
| A stacker on rank scores with repeated bagging [W40] | 0 of 78; the between-pool weights are two scalars already fixed without a scan [R] |
| Five-epoch students as a cost lever | Dropped on record: cost is not binding and N is saturated [R, SYNTHESIS section 5]; M1 confirms that cheaper members do not change the conclusion |
| A second round of self-training | "One iteration is best" in [W22]; generation 2 closed [R] |

## 7. Sources

Pages were read through a summarising fetch tool. Where the tool reported that a number sits only in a figure, the number is not quoted. arXiv pages were read as `ar5iv.labs.arxiv.org/html/<id>` or `arxiv.org/html/<id>`.

| # | source | URL | contribution |
|---|---|---|---|
| W1 | Ashukha et al. 2020, Pitfalls of in-domain uncertainty estimation and ensembling | https://arxiv.org/abs/2002.06470 | Deep-ensemble-equivalent score; cheap ensembles equal a handful of independent models |
| W2 | Lobacheva et al. 2020, On power laws in deep ensembles | https://arxiv.org/abs/2007.08483 | c + b·n^a with a near -1; asymptote predictable from 4-6 networks |
| W3 | Arpit et al. 2022, Ensemble of averages | https://arxiv.org/abs/2110.10832 | 68.0 against 66.8 at 6 members under a constant learning rate |
| W4 | Nam et al. 2021, Diversity matters when learning from ensembles | https://arxiv.org/abs/2110.14149 | Students of an ensemble stay below it; perturbation closes part of the gap |
| W5 | Li et al. 2024, Ensemble learning via knowledge transfer for CTR prediction (v2: From collapse to stability) | https://arxiv.org/abs/2411.16122 | Joint CTR ensembles degrade with size; mean-teacher plus mutual learning, flat after 5 sub-networks |
| W6 | Wortsman et al. 2022, Model soups | https://arxiv.org/abs/2203.05482 | Soup against output ensemble in and out of distribution; shared-start precondition; soup-ensemble approximation |
| W7 | Rame et al. 2022, Diverse weight averaging | https://arxiv.org/abs/2205.09739 | Weight average against output ensemble at M = 20; conditions for averaging |
| W8 | Abe et al. 2023, Pathologies of predictive diversity in deep ensembles | https://arxiv.org/abs/2302.00704 | Encouraging diversity hurts large networks |
| W9 | Jeffares et al. 2023, Joint training of deep ensembles fails due to learner collusion | https://arxiv.org/abs/2301.11323 | Joint training worse than independent in every case |
| W10 | Wenzel et al. 2020, Hyperparameter ensembles | https://arxiv.org/abs/2006.13570 | +0.2..+1.6 points at size 4; greedy selection with stratified seeds |
| W11 | Gorishniy et al. 2024, TabM | https://arxiv.org/abs/2410.24210 | Implicit members weak alone, strong together; still gains from deep ensembling |
| W12 | Abe et al. 2022, Deep ensembles work, but are they necessary? | https://arxiv.org/abs/2202.06985 | A larger single model reproduces ensemble gains; no extra robustness |
| W13 | Theisen et al. 2023, When are ensembles really effective? | https://arxiv.org/abs/2305.12313 | Improvement tracks disagreement relative to error |
| W14 | Gontijo-Lopes et al. 2022, No one representation to rule them all | https://arxiv.org/abs/2110.12899 | Ranking of diversity sources: seeds < hyper-parameters < architecture < framework < data |
| W15 | Wilson and Izmailov 2020, Bayesian deep learning and a probabilistic perspective of generalization | https://arxiv.org/abs/2002.08791 | MultiSWA / MultiSWAG: edge for few models and under corruption |
| W16 | Fort et al. 2019, Deep ensembles: a loss landscape perspective | https://arxiv.org/abs/1912.02757 | One trajectory gives similar functions; independent runs do not |
| W17 | Havasi et al. 2021, MIMO | https://arxiv.org/abs/2010.06610 | Below a deep ensemble of 4; peaks at 2-4 subnetworks |
| W18 | Jolicoeur-Martineau et al. 2023, PAPA | https://arxiv.org/abs/2304.03094 | Independent ensembles stay above population averaging |
| W19 | Qin et al. 2023, RD-Suite | https://arxiv.org/abs/2306.04455 | Listwise softmax is the most robust ranking-distillation loss; label transform is delicate |
| W20 | Pagliardini et al. 2023, Agree to disagree (D-BAT) | https://arxiv.org/abs/2202.04414 | Disagreement on unlabelled target data: +0.7..+11 at size 2 |
| W21 | Purucker and Beel 2023, CMA-ES for post hoc ensembling in AutoML | https://arxiv.org/abs/2307.00286 | Dense weights overfit a ranking metric; greedy selection's normalisation and sparsity |
| W22 | Jiang et al. 2023, Noisy self-training with synthetic queries for dense retrieval | https://arxiv.org/abs/2311.15563 | Same-architecture student gains with 17.6x-357x unlabelled lists, noise and soft labels |
| W23 | NVIDIA 2025, The Kaggle Grandmasters playbook | https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data | Hill climbing, stacking, k-fold pseudo-labels, 100-seed averaging (+0.003 MAP@3 against one seed) |
| W24 | Pareek et al. 2024, Understanding the gains from repeated self-distillation | https://arxiv.org/abs/2407.04600 | When repeated self-distillation gains and when it cannot |
| W25 | Cha et al. 2021, SWAD | https://arxiv.org/abs/2102.08604 | Dense averaging under a constant rate; +3.6 over ERM on DomainBed |
| W26 | Wood et al. 2023, A unified theory of diversity in ensemble learning | https://arxiv.org/abs/2301.03962 | Bias / variance / diversity trade-off; diversity is not to be maximised |
| W27 | Ovadia et al. 2019, Can you trust your model's uncertainty? | https://arxiv.org/abs/1906.02530 | Most ensemble gain by 5 members; validation-fitted calibration fails under shift |
| W28 | Yao et al. 2022, Wild-Time | https://arxiv.org/abs/2211.14238 | SWA and other methods do not beat ERM consistently under temporal shift |
| W29 | Nam et al. 2022, Improving ensemble distillation with weight averaging and diversifying perturbation | https://arxiv.org/abs/2206.15047 | Averaged multi-subnetwork student still below the teacher ensemble |
| W30 | Wen et al. 2020, BatchEnsemble | https://arxiv.org/abs/2002.06715 | Below a naive ensemble of 4 by 0.4-0.7 points |
| W31 | Bruch et al. 2022, An analysis of fusion functions for hybrid retrieval | https://arxiv.org/abs/2210.11934 | Convex combination of normalised scores beats rank fusion; one robust weight |
| W32 | Wortsman et al. 2021, Robust fine-tuning of zero-shot models (WiSE-FT) | https://arxiv.org/abs/2109.01903 | Weight interpolation of start and fine-tune; alpha 0.5; needs a shared start |
| W33 | Roth et al. 2023, Fantastic gains and where to find them | https://arxiv.org/abs/2310.17653 | Distilling into a trained student fails in most pairs unless partitioned; gains +0.3..+1.0 |
| W34 | Huang et al. 2017, Snapshot ensembles | https://arxiv.org/abs/1704.00109 | Cyclic snapshots worth about 2 independent models |
| W35 | Garipov et al. 2018, Loss surfaces, mode connectivity and fast ensembling | https://arxiv.org/abs/1802.10026 | FGE numbers at equal budget; 14.57% against 19.97% disagreement |
| W36 | Zhu et al. 2020, Ensembled CTR prediction via knowledge distillation | https://arxiv.org/abs/2011.04106 | Student above a single teacher; returns diminish from 3 to 6 teachers |
| W37 | Buschjager et al. 2020, Generalized negative correlation learning for deep ensembling | https://arxiv.org/abs/2011.02952 | At M = 16 large networks want little or no joint term |
| W38 | Nebius 2024, Chatbot Arena human preference predictions: tech review | https://nebius.com/blog/posts/chatbot-arena-competition-review | Winner distilled large fold teachers into a smaller student and averaged adapter weights (no gain numbers on the page) |
| W39 | Zhang et al. 2022, Confidence-aware multi-teacher knowledge distillation | https://arxiv.org/abs/2201.00007 | Label-based teacher weights against plain averaging, 3 teachers, small students |
| W40 | Erickson et al. 2020, AutoGluon-Tabular | https://arxiv.org/abs/2003.06505 | Out-of-fold stacking with repeated bagging; where stacking over-fits |
| W41 | Sydorskyi and Goncalves 2025, second-place code of a bird-audio challenge | https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place | Pseudo-labelled soundscapes in the final models (README only; no per-round numbers) |

Search snippets only (not opened; no number quoted from them): Guo et al. 2020 online distillation via collaborative learning (CVPR); Caruana et al. 2004 / 2006 ensemble selection; Turc et al. 2019 well-read students (arXiv 1908.08962); arXiv 2205.12428 on label regularisation in fine-tuning; the RecSys Challenge 2024 overview (arXiv 2409.20483, PDF not readable through the tool) and winner slides; CEUR working notes of the 2025 bird-audio challenge.

## 8. Limits

- M1 and M2 use held-out members and the train-only branch. The fine-tuned branch and the full-data twins were not re-read; the size law of the final file (0.5 FT + 0.5 TO) is on record only for W8 subsets [R, `pool_composition.md` section 3].
- M1's stale-val column rests on one witness scenario (lt3) and one val set; its pairings disagree by up to 0.0003.
- M2 compares pools that differ in width, teacher and seed draw at once; it cannot isolate the teacher's view. It is a prior, not a test of P2.
- Probabilities and expected gains in section 5 are judgement. No arm was run.
- No source tests any of these techniques on top of 16 or more independent members of a ranking model; the "16+" column is an inference from the size law and from the project's own measurements, except W37 (M = 16, classification).
- Three sources could be read only in part (W1 figures, W15 figures, W38 without numbers); W41 gave the README only.
