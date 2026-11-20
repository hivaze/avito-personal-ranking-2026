# RECHECK 11-14, lens "kd_literature": what the distillation literature still offers once the student matches the teacher

Scope: web research (32 sources opened, section 7) plus eight CPU measurements on stored predictions, teacher logits and checkpoints. No GPU, no training file touched, no submission file. Scripts and logs: `nn_x/w12/recheck/kd_literature/`.

## 0. Summary

1. **The literature's answer for our regime is mostly "no".** Same-capacity self-distillation gives one step (born-again rankers: +0.2 to +1.5% relative NDCG, alpha 0.5, listwise only) and then saturates; an ensemble of students is no better than the ensemble it was distilled from unless the student gains capacity, noise or data (Allen-Zhu & Li; Noisy Student: a same-size student of a B7 teacher gains 0.0). Our generation-2, bigger-teacher and cross-fit closures match the published pattern.
2. **Seven techniques fail their own precondition here, measured.** The teacher target is nearly flat (entropy 4.41 of 4.90 nats, mass on clicks 0.032) and calibrated (T* 1.10 unseen, 0.95 in-sample), so temperature, decoupled KD, weighted soft labels and confidence weighting have nothing to act on. In-sample memorisation moves 0.2% of target mass, so cross-fitting cannot matter. Different teachers give students no more diverse than different seeds (0.9718 vs 0.9722). Item-level privileged features add nothing to the teacher (nested -0.00005 +- 0.00019). Data-split members trade -0.0028 strength for 0.004 of correlation.
3. **Sub-feed augmentation (running kwsub arms) is a valid test.** Stored full-feed logits are a consistent teacher for a sub-feed: KL to a re-scored teacher is 0.0014-0.0044 nats against a seed-to-seed floor of 0.026-0.049. The stored teacher is even slightly better than a re-scored one (+0.0006 to +0.0013 NDCG on the sub-feed), so kwsub is also a privileged-context distillation toward the test format. A null there closes function matching; no online teacher is needed.
4. **Function matching has a hard ceiling at pool level.** A pure-imitation pool (kd10) sits below its teacher (-0.0015 +- 0.0005 val, -0.0008 +- 0.0003 holdout). A perfect student would be the teacher pool as one model (+0.0012 val / +0.0011 holdout per member), and a pool of such students has no diversity. The hard-label half of the target supplies diversity, not member strength: member level is flat for kd_w 0.5-1.0, member correlation runs 0.978 -> 0.995.
5. **New fact from the logs: the KD gain is made on the plateau and saturates by ~30K steps.** kd10 after 1 epoch equals the hard-label model after 10. The cooldown adds +0.0037 val for hard labels and +0.0013 for kd10. Plateau target and cooldown target have never been varied separately; the 20%-cost re-cool harness can do it (arm A1).
6. Ranked arms: A1 phase-split re-cool grid (9 GPU-h), A2 kwsub follow-ups with a pre-stated plateau-slope rule, A3 wider student with more noise (the one Noisy-Student ingredient not yet run), A4 soft-target fine-tune with a leak-clean teacher, A5 five-epoch students as a cost lever. My probability of >= +0.001 public at pool level is 0.03-0.08 each; the sum over the list is about +0.0004 expected.

## 1. CPU measurements (all paired per feed; val 14,423 feeds, holdout 30,000 feeds)

| # | Measurement | Result | Bounds |
|---|---|---|---|
| M1 | Teacher softmax at T = 1 on holdout (`teacher_calib.py`) | entropy 4.413 nats (uniform 4.896); median max p 0.045; top-10 mass 0.30; mass on clicks 0.032 | DKD, WSL, CWTM, top-k focus |
| M2 | Teacher calibration | T* 1.10 on holdout (NLL gain 0.0019 +- 0.0004 nats), 0.95 in-sample; mean25 1.05 / 0.90; w8g1 1.10 / 0.90 | temperature (kwT07 / kwT15) |
| M3 | In-sample memorisation, matched fold models (12 that trained on the feed vs 3 that did not) | NDCG +0.0195 +- 0.0005; log p(click) +0.110 +- 0.001 nats; mass on clicks 0.0317 -> 0.0340; implied extra hard-label share 0.0023 +- 0.0002 (production feeds 0.0012, M<50 0.0059) | cross-fitted teacher |
| M4 | Sub-feed consistency on val, CPU replay of checkpoints (`subfeed_probe.py`) | aug 0.5, 4 hard-label members: Spearman(full, re-scored) 0.9952, KL 0.0014 nats; seed floor Spearman 0.924, KL 0.049. aug 0.2, 2 wide students: 0.9895, KL 0.0038; floor 0.953, KL 0.026. Re-scored minus stored, NDCG on the sub-feed: -0.00063 +- 0.00038 (1.7 SE) and -0.00126 +- 0.00051 (2.5 SE) | kwsub validity |
| M5 | Fidelity to the teacher pool tf11 (val 0.40054 / holdout 0.32371) (`pools.py`, `w8_vs_teacher.py`) | kd10 4-pool -0.00149 +- 0.00051 / -0.00080 +- 0.00028; kd05 +0.00031 +- 0.00052 / -0.00022 +- 0.00031; W8 +0.00102 +- 0.00054 / +0.00085 +- 0.00032; teacher minus one wide student +0.00119 +- 0.00049 / +0.00108 +- 0.00028 | function matching ceiling |
| M6 | Teacher identity vs seed as diversity source | student-student val corr: same teacher, different seed 0.9702-0.9736; different teacher, different seed 0.9707-0.9720; different teacher, same seed 0.9751-0.9762. Two-teacher minus one-teacher 4-pools: -0.00024 +- 0.00022 val, -0.00001 +- 0.00013 holdout. Teacher-teacher corr 0.994-0.998 | disjoint / stochastic / multi-head teachers, kdmixw192 |
| M7 | Item-level privileged feature for a PFD teacher (`pfd_gate.py`): leave-one-out item CTR / COEC over the train window | within-feed AUC 0.528 on holdout (63% of rows with >= 5 other impressions); nested residual on teacher_mean11 -0.00005 +- 0.00019, on w8g1 0 (w = 0 picked in both halves); every w >= 0.1 hurts (-0.0012 at 0.1, -0.013 at 0.3) | PFD-ITEM (gate was >= +0.003) |
| M8 | Data-split members: 15 fold models on 80% of users (`bagging.py`) | member -0.00275 +- 0.00049 val / -0.00217 +- 0.00028 holdout; corr different fold 0.936, same fold 0.940, full data 0.949; different-fold minus same-fold 3-pool +0.00028 +- 0.00021 / +0.00009 +- 0.00011 | bagging, co-distillation on shards |
| M9 | Popularity slices of the KD gain on val (cold = no leave-user-out popularity) | kd05 member: cold-positive feeds +0.0019 +- 0.0009, warm +0.0029 +- 0.0006. kd10: cold +0.0004 +- 0.0010, warm +0.0028 +- 0.0007. kd05w192: cold +0.0046 +- 0.0010, warm +0.0021 +- 0.0006. W8 minus teacher: cold +0.0036 +- 0.0010, warm -0.0001 +- 0.0007. Mean rank-z of cold candidates -0.328 (base pool), -0.330 (kd05), -0.326 (W8) | popularity-stratified KD |
| M10 | Training curves, 4-seed means from the logs | see section 2 | schedule arms |

Reading of M9: distillation does not push cold items down. Pure imitation gains nothing on cold-positive feeds, and the whole edge of the wide students over their teacher is on cold-positive feeds (24% of val). Width under KD works through the content pathway.

## 2. Where the KD gain is made (M10; `nn_x/w5/logs/w5_e10_<arm>_s951-954.stdout`)

4-seed means of the periodic evals (seeds 951-954; `curves.log`). Plateau = evals at 57.5-77.5K steps, early = 20-40K, cooldown gain = last step minus plateau. SE over seeds is 0.0001-0.0003 for every difference.

| arm (teacher share) | val @10K | plateau val / holdout | late minus early plateau | cooldown gain | last step val / holdout | member corr |
|---|---|---|---|---|---|---|
| base (0) | 0.3876 | 0.3926 / 0.3168 | +0.0013 / +0.0009 | +0.0037 / +0.0031 | 0.3962 / 0.3199 | 0.9485 |
| kd03 (0.3) | 0.3904 | 0.3945 / 0.3185 | +0.0011 / +0.0008 | +0.0037 / +0.0029 | 0.3982 / 0.3214 | 0.9657 |
| kd05 (0.5) | 0.3918 | 0.3956 / 0.3194 | +0.0007 / +0.0006 | +0.0035 / +0.0028 | 0.3991 / 0.3222 | 0.9776 |
| kd08 (0.8) | 0.3957 | 0.3974 / 0.3210 | +0.0004 / +0.0004 | +0.0018 / +0.0020 | 0.3992 / 0.3229 | 0.9904 |
| kd10 (1.0) | 0.3963 | 0.3976 / 0.3213 | +0.0004 / +0.0003 | +0.0013 / +0.0012 | 0.3988 / 0.3225 | 0.9950 |
| kd05w192 (0.5) | 0.3925 | 0.3960 / 0.3199 | +0.0007 / +0.0007 | +0.0031 / +0.0027 | 0.3991 / 0.3226 | 0.9722 |

- The plateau level rises with the teacher share (+0.0030 val at 0.5, +0.0050 at 1.0) and saturates at 0.8. The cooldown gain is constant up to 0.5 (+0.0035 to +0.0037) and collapses above it (+0.0018, +0.0013).
- kd10 after one epoch (0.3963 / 0.3199) equals the hard-label model at its last step (0.3962 / 0.3199).
- Under KD the plateau is flat after 30K steps. This is the published "fixed transfer set" behaviour: long schedules pay only with augmentation (Beyer et al., Table 3: the fixed teacher peaks and then declines, the consistent teacher keeps rising to 9,600 epochs). It explains the kw_e20 closure and says a longer run is justified only on top of kwsub.
- Last-step member level is the same for teacher share 0.5 / 0.8 / 1.0; only the correlation differs, and the pool follows the correlation (4-pool val 0.40085 / 0.40004 / 0.39906).
- Inference, not measured: the hard-label cooldown gain is known to be a mean-function gain that survives pooling (r6: 7-seed ensemble +0.0031 +- 0.0006), while a higher plateau level may be mostly iterate-noise removal that pooling also provides. That would explain why kd10 members are as strong as kd05 members and their pool is not.

## 3. Literature table

Gains are as reported by the source. "Bounded" = today's closed or running arms already test it. "Precondition fails" cites a measurement from section 1.

| technique | source | reported gain | precondition | status here |
|---|---|---|---|---|
| Listwise self-distillation of rankers (SDR) | arXiv 2109.15285 | student vs teacher NDCG@1 +1.49% Web30K, +0.27% Istella; student ensemble beats teacher ensemble by +0.15 to +0.77%; teacher-only target below the teacher; pointwise MSE -1.9 points | alpha 0.5, listwise loss, noisy labels | done (kd05); matches: kd10 below teacher, W8 above C3 |
| Born-again networks, generations | arXiv 1805.04770 | CIFAR-100 error -0.8 to -1.3; generations "saturate after a few", gen 3 worse than gen 2 in one of three | same capacity | bounded: generation 2 closed |
| Ensemble then distil then ensemble | arXiv 2012.09816 | ensemble of distilled models "gives no further performance boost" over the original ensemble | multi-view data | matches: kd05 4-pool = teacher pool (M5) |
| Noisy Student | arXiv 1911.04252 | +1.1 points at 130M unlabeled; without student noise +0.3; same-size student of B7 +0.0, larger student +0.3; plateau at >= 6x labelled data | bigger student AND noise AND much unlabeled data | bigger student: w192 yes, w256 no at dropout 0.1; more noise: never run (A3); unlabeled data: 43K in-period feeds = 1.4% of train, fails |
| Function matching (patient, consistent teacher) | arXiv 2106.05237 | ImageNet R50 82.3 at 9,600 epochs; at 30 epochs function matching is 1.06 below plain consistent teaching; fixed teacher 5.4 below at 9,600; student matches, does not exceed, the teacher | same view for teacher and student; aggressive augmentation; long schedule | kwsub running; consistency holds (M4); ceiling = teacher (M5); follow-up A2 |
| Does KD really work (fidelity) | arXiv 2106.05945 | agreement stays < 90% even with 5x data; with ensemble teachers fidelity and accuracy are positively related; out-of-distribution transfer data hurts | - | matches M5; synthetic or cross-feed transfer sets low prior |
| Cross-fitting and loss correction | arXiv 2104.09732 | up to +4 AUC points (tabular), +1.5% CIFAR-10, when the teacher over-fits; corrected loss = our hard / soft mixture | in-sample targets unrepresentative | precondition fails (M3: 0.2% of mass); closed, now with a mechanism |
| Self-distillation under label noise | arXiv 2301.13304 | optimal teacher weight can exceed 1 at 50% label corruption | high noise, linear probe | bounded: member flat for kd_w 0.5-1.0, pool falls |
| Temperature / logit matching | arXiv 2105.08919 | CIFAR-100 MSE on logits +2.6 over KL at tau 1; under label noise small tau is best, large tau and MSE worst | peaked teacher | precondition fails (M1, M2); kwT07 / kwT15 predicted null, T 1.5 at risk |
| Decoupled KD | arXiv 2203.08679 | +1 to +3 points over KD | confident teacher suppresses the non-target term by (1 - p_target) | precondition fails: 1 - p = 0.97 (M1) |
| Weighted soft labels | arXiv 2102.00650 | +1.0 to +1.5 points over KD | per-sample ratio of student to teacher loss varies | precondition fails: both losses ~4.4 nats on every long feed |
| Early-stopped teacher / early-stopped KD | arXiv 1910.01348 | ImageNet R18 error 30.6 -> 29.0 | student smaller than teacher | fails (student >= teacher); the phase question itself is open (A1) |
| Progressive self-distillation (teacher share rises over training) | arXiv 2006.12000 | CIFAR-100 error 24.2 -> 20.8; rising weight beats fixed | - | never varied by phase here (A1) |
| Relaxed ranking distillation (RRD), RankDistil, top-K ranking distillation | arXiv 2012.04357, PMLR v130 reddi21a, arXiv 1809.07428 | H@5 +6 to +14% over earlier KD at a 10x smaller student; same-size student 1.01-1.03x teacher; pairwise form "disappointing" | small student, top-K metric | precondition fails: metric is full-list NDCG, 66% of the loss is in feeds whose positive sits at rank 8+ |
| Margin-MSE | arXiv 2010.02666 | nDCG@10 +0.001 to +0.004 over pointwise MSE | cross-architecture score scales | near copy; SDR found listwise best |
| Auxiliary-head distillation | arXiv 2408.14678 | RMSE 12.15 vs 12.20 direct; direct = no-distillation control there | direct distillation fails on a biased teacher | fails: direct works here (+0.0026) |
| Co-distillation, deep mutual learning | arXiv 1804.03235, 1706.00384 | matches two-phase distillation in fewer steps; "all students become somewhat more similar"; works much better on disjoint data shards | - | bounded by kd arms; shards closed by M8 |
| Ensemble distribution distillation, Hydra | arXiv 1905.00076, 2001.04694 | accuracy equal to plain ensemble distillation (CIFAR-100 27.9 vs 28.0 error); Hydra +0.6 points with 9-14x the parameters | members differ | precondition fails (M6) |
| Multi-teacher / data-level teachers (CSRec) | arXiv 2311.02446 | Recall@10 +7% over one soft-label teacher on one dataset; "more teachers are not necessarily" better | - | bounded: 25-member teacher closed; M6, M8 |
| Privileged features distillation | arXiv 1907.05171, 2209.08754 | Taobao +5.0% clicks; NDCG@8 +3.7 to +9.5% vs 0 to +1% for self-distillation; non-monotone in the feature's power | a train-only feature that improves the teacher | gate fails (M7); hindsight closed earlier |
| Unbiased KD (popularity) | arXiv 2211.14729 | unpopular-group recall -22% under standard KD | soft labels biased to popular items | not observed (M9) |
| Weight averaging of students (soups) | arXiv 2203.05482 | soup about equal to the logit ensemble (81.03 vs 81.19) | same pre-trained init | closed for the pool in r6; nothing new |
| Learning without forgetting (distil the pre-fine-tune outputs during fine-tuning) | arXiv 1606.09282 | new-task accuracy +0.7 over plain fine-tuning; "an effective regularizer" | fine-tune on a small set | never run on the val fine-tune (A4) |
| Self-training under shift | arXiv 2002.11361 | soft pseudo-labels 80.1 vs hard 82.6; "regularization and label sharpening are essential" | separable classes | fails: top-1 probability 0.045, sharpening would add noise |
| Meta pseudo labels | arXiv 2003.10580 | +1.8 over Noisy Student | teacher trained jointly with student feedback | long bet only |

## 4. Ranked arms

P = my probability of >= +0.001 public at pool level. Costs assume ~3.3 h wall per 10-epoch d192 run and ~40 min per 2-epoch re-cool.

**A1. Phase-split KD by re-cooling stored plateau snapshots (P 0.06, ~9 GPU-h, no train.py change).**
Claim: plateau target and cooldown target do different jobs (section 2), and published schedules disagree on the direction (rising teacher share: PS-KD; teacher first then labels: early-stopped KD). The grid has never been run.
Pool-level mechanism: r6 found that member diversity is made by plateau iterate noise (seed corr 0.94 at 10 epochs, 0.978 at 2 epochs, 0.975 after plateau weight averaging). KD lowers that noise, and student correlation rises with the teacher share. If the KD member gain can be had in the cooldown alone, a hard-label plateau plus a KD cooldown gives students with base-like diversity. If instead the two gains add (kd10 plateau 0.3976 plus a hard-label-type cooldown gain of +0.0035), the member itself moves. Either outcome is new; the null outcome (the result depends only on the cooldown target) still yields the cheap student route below.
Arm: for P in {base, kd05, kd10} and C in {0, 0.5, 1.0}, seeds 951-954 (all snapshots exist):
`train.py --variant w5_rc_<P>c<C>_s<seed> --init nn_x/w5/ckpt/w5_e10_<P>_s<seed>@77500.pt $B_BASE $HYG --seed $((900+seed)) [--kd_file .../teacher_mean11.npy --kd_w <C>]` with `B_BASE` and `HYG` exactly as in `lane.sh` (the stage-B re-cool harness). Launch from a new launcher file.
Read-out per cell: member val / holdout, member-member correlation, 4-pool. Diagonal cells are the controls. A cell promotes if its 4-pool beats the kd05 diagonal by >= +0.0005 val with holdout >= -0.0003 (round-2 rule), Holm over the 6 off-diagonal cells; then replicate from scratch at d192 with a two-line `--kd_w_cd` flag.
Side product: if a hard-label plateau plus a KD cooldown reaches student level, every stored plateau (7 r6 roots, GRU family) becomes a student for 20% of the cost.

**A2. kwsub follow-ups, decided by a rule fixed now (P 0.08 conditional on kwsub not failing).**
- kwsub is a valid test (M4). If both doses fail the member screen and their holdout plateau slope (mean 57.5-77.5K minus mean 20-40K) is not above kd05w192's +0.0007 by >= +0.0010, close function matching, including cross-feed swaps and online teachers.
- If the slope criterion holds, the only justified long run is `E:kwsub5` flags plus `--epochs 15` (kw_e20 failed without augmentation, as the literature predicts).
- Noisy-student input noise, zero patch: `--aug_all --aug_hist 0.15` alone and with `--aug_cand 0.5`. History dropout was only tested as a replay on a trained model, never in training.
- Interim only, one seed at 20K steps: kwsub5 0.3939 / 0.3188 vs kd05w192 0.3948 / 0.3192. Not a read-out.

**A3. Wider student with more noise (P 0.06, 4 runs).**
Claim: Noisy Student needs noise for the larger student; w256 was run only at dropout 0.1, and kw_drop0 shows less dropout is worse (-0.0006). The wide students' edge over the teacher is all on cold-positive feeds (M9).
Arms, 2 seeds each, screen vs kd05w192 (val >= +0.0008 and holdout >= +0.0005): `--kd_file teacher_mean11.npy --kd_w 0.5 --d 256 --drop 0.2`; same with `--d 192`. Add `--aug_all --aug_cand 0.5` if kwsub passes. Generation 2 is worth re-opening only behind a pass here (B7 -> B7 gained 0.0 in the source).

**A4. Soft-target fine-tune, leak-clean form (P 0.05; minutes per member).**
Measured: for W8 the FT branch is 0.40087 vs TO 0.40157 (-0.0007 +- 0.0006), and 0.5 / 0.5 adds +0.00009 +- 0.00043 over TO alone. Fine-tuning on one-hot val labels is worth nothing at pool level.
Arm: copy `finetune.py` to a new file and add the same mixture target as train.py, teacher = mean train-only logit of the W8 members on the val rows, kd_w 0.3 and 0.5. That teacher never saw a val label, so the out-of-fold read stays clean; a teacher built from fine-tuned models of other folds would leak the held-out fold's labels through the soft target. No holdout witness exists for this stage; the judge is the final blend on val (>= +0.0005 and 2 SE) and then a public probe. The ft_redesign reviewer owns the details.

**A5. Five-epoch students as a cost lever (P 0.03 for level; value is 2x students per GPU-hour).**
`E:kd05w192` flags plus `--epochs 5 --cooldown_frac 0.4` (plateau 29.5K steps, cooldown 19.7K steps as now). Read-out: non-inferiority of the 4-pool vs kd05w192 (val >= -0.0005, holdout >= -0.0003) and member correlation. Risk: shorter plateaus were less diverse under hard labels (0.978 vs 0.94).

**Long bet (P 0.03).** Test and val feeds as an unlabeled transfer set in the cooldown (TOP1_GAP_EXTERNAL lever 2). The literature lowers its prior: soft labels alone do not adapt a model, sharpening needs separable classes, and the unlabeled set is 1.4% of train where Noisy Student needed >= 6x.

## 5. What today's results already bound

| idea | bounded by |
|---|---|
| more generations, same student | gen-2 closure; BAN saturation; Noisy Student same-size result |
| bigger, mixed, family-balanced, disjoint or stochastic teacher | t25 closure; M6. kdmixw192 predicted null (its teacher correlates 0.997 with mean11) |
| cross-fitted or early-stopped teacher | kx05 / kx08 closure; M3. The cross-fit teacher was also 0.0027 weaker (3 models, 80% of users), and that deficit is what the students inherited |
| temperature, DKD, weighted soft labels, confidence weighting | M1, M2; kwT arms running |
| rank-distillation losses (top-K, Plackett-Luce, margin) | SDR's own ablation; full-list metric |
| co-distillation, mutual learning | same information as offline ensemble KD; lowers diversity |
| privileged item aggregates | M7 |
| long schedule without augmentation | kw_e20; section 2 |
| data-split members | M8 |

## 6. Not checked

- The sub-feed probe ran on val feeds (already random subsets), not on train production feeds; set dependence could be larger for full production lists.
- M7 is a linear residual on exploration-type holdout feeds with own-row exclusion only. A non-linear teacher or production feeds were not tested; on production feeds the feature is confounded with position.
- Plateau and cooldown numbers come from the periodic evals in the logs (4 seeds), not from re-scored snapshots.
- A1-A5 priors are judgement. No arm was run.
- Fetched pages were read through a summarising tool; where it said a table was missing from the page text, I did not quote that number. RankDistil and Menon et al. were read as abstracts only.

## 7. Sources opened

ar5iv.labs.arxiv.org/html/ + 2106.05237, 1911.04252, 2109.15285, 2408.14678, 2012.09816, 1805.04770, 2104.09732, 2301.13304, 2011.04106, 2012.04357, 2209.08754, 1907.05171, 2203.08679, 1804.03235, 1706.00384, 2106.05945, 2102.00650, 2211.14729, 2311.02446, 1910.01348, 2006.12000, 2001.04694, 2010.02666, 2203.05482, 2105.08919, 1905.00076, 2002.11361, 1809.07428, 1606.09282, 2003.10580; arxiv.org/abs/2106.05237, arxiv.org/abs/1911.04252; proceedings.mlr.press/v130/reddi21a.html; proceedings.mlr.press/v139/menon21a.html. Search snippets only: stochastic single-teacher sampling (SKDBERT, AAAI 2023), reinforced multi-teacher selection (arXiv 2012.06048).
