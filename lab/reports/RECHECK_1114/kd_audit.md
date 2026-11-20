# RECHECK 11-14: correctness audit of the ensemble-distillation (KD) pipeline

**FINAL, 2026-11-14 20:45 UTC.** The 20:16 UTC draft of this file had three placeholders (the pooled FT line, the fine-tune proxy, the proposals); sections 6-8 are new, section 0 was corrected in one point (the fine-tune item is now a measured null), a note on `feed_rank_z` and an audit of the 20:20 UTC `--xfold_unlab` patch were added. Sections 1-5 are unchanged.

Reviewer key `kd_audit`. CPU only. Scripts and logs: `nn_x/w12/recheck/kd_audit/` (01-16). Nothing under `src/`, `nn_x/w5/` or `preds/` was written.
"Measured" = computed here from prediction files, logs or checkpoints. "Inferred" is marked as such. SE = paired per-feed standard error (val 14,423 feeds, holdout 30,000 feeds).

## 0. Verdict

- **No bug and no label leak found** in the KD path (train.py KD / `--aug_all` / `--xfold` / `--prand2` patches, the abl port, dump_teacher.py, teacher assembly, build_w.py / build_k.py / pool_read.py / pool_multi.py / member_vs.py, ft_w5.sh + finetune.py, the lane.sh arms). Every headline number was reproduced from the files.
- The pre-declared read-outs are **slightly biased against KD on the holdout** (checkpoint selection on the same holdout helps the hard-label base more). KD is confirmed, a little stronger than reported.
- **All seven "free win" checks requested are measured nulls.** The one place where the students looked short-changed, the val fine-tune (it adds -0.0004 to the wide students and +0.0007 to the C3 members), was tested with a CPU replica of the fine-tune: a soft-target fine-tune and an early stop both improve the member and both vanish in the 0.5 / 0.5 file (section 6). No GPU is asked for there.
- New measurements (section 4) say what KD is here: a **gradient-noise reduction** (main-loss gradient norm is proportional to 1 - kd_w), insensitive to the teacher (students are equal for teachers between 0.3210 and 0.3245 holdout NDCG), with student diversity coming only from the student's own seed. The same-recipe student pool follows a 1/k law with a val asymptote of 0.4015 to 0.4021 depending on the arm fitted (W8 is 0.40157): **W16 is worth +0.0001 to +0.0003 val over W8 and nothing useful remains beyond it** (same conclusion as `kd_reprice` and `pool_composition`).

## 1. What was verified (all measured)

| check | result | script |
|---|---|---|
| Headline pools | base x4 0.39887 / 0.32328; kd05 x4 0.40085 / 0.32349; kd05w192 x4 0.40134 / 0.32408; W8 TO 0.40157 / 0.32456; W8 file FT 0.40087, TO 0.40157, final 0.40165; C3 final 0.40101; W8 - C3 +0.00064 +- 0.00055 (val). All equal to the log. | 03 |
| Row alignment of the 246.7M-row teacher files | Rows of the 30,000 holdout feeds (4,630,966 rows) of teacher_a/d/e/f/p1/p2/h1/h2/xf0-4 equal the SUM of the members' own saved holdout predictions: centred corr 1.000000, max abs diff < 0.0005 in every file (13 files, 39 members). holdout_y equals the train target on those rows. | 02 |
| Teacher assembly | mean11 = per-feed-centred (a+..+f)/11; mean25 = centred (a..f+g1+g2+p1+p2)/25; w8g1 = (h1+h2)/8 (max diff 8e-6); mix25w8 = 0.5 mean25 + 0.5 w8g1 (4e-6). Centring is harmless (softmax is shift-invariant). Holdout NDCG: mean11 0.32370, mean25 0.32286, w8g1 0.32453, mix 0.32391, oof 0.32100 (= the log). | 02, 15 |
| Teacher members and the holdout | All 25 members of mean11/mean25, the 8 wide students, the xf models and the v17 snapshots behind r6_cd_m02 were trained with `--holdout 30000 --select_on holdout` (checkpoint args). No member touched val labels. No FT model is in any teacher. | 01 |
| dump_teacher.py | `predict()` sets eval mode; the domain token is the training mixture [1-p, p, 0]; xt columns are re-attached per checkpoint cfg; GRU and pop1 (abl) checkpoints reproduce their own holdout files exactly, so the w5 model/data code is valid for them. | 02 |
| KD target code (w5 and abl port) | Formula identical in both files. Teacher softmax is computed in fp32 outside autocast; padded positions point at the feed's first candidate and are masked; target sums to 1, so `--loss_norm feed` weighting is unchanged. | read |
| Feeds without positives | **0 of 3,151,208 train feeds** (0 in the pool, 0 in the holdout, 0 of 14,423 val feeds). `--aug_cand` keeps every positive. The `(y.sum > 0)` gate is dead code: nothing is lost. Multi-positive feeds: 33.4% of train (mean 1.73 positives), 25.5% of val. | 02 |
| bf16 and masked_fill(-1e4) | The main head runs in fp32 (model.py l.623), so the logits (mean abs 131, within-feed sd 1.2) are not quantised and -1e4 masks are safe. The KD softmax and the listwise loss are fp32. Only the K9 aux head output is bf16; its logits on val are small (mean abs 0.18), quantisation step 0.001: immaterial. | 08, read |
| kd_T and centring | No interaction: softmax((t - c)/T) = softmax(t/T). But see R3: kd_T softens the teacher only. | read |
| `--xfold` | Same hash in train.py and assemble_oof.py; holdout excluded before the fold drop; 624,752 feeds dropped for fold 0. Fold teachers have 78,710 steps (80% data and 80% steps). | read, log |
| `--aug_all` | Mask, labels, log M and every within-feed z/rank are recomputed for the sub-feed; teacher logits are renormalised over the kept set. The teacher logits come from the FULL feed, which is fine: see section 5.6. | read, 08 |
| finetune.py for students | Standard n2cpv3s3 recipe, no KD term; domain row 2 initialised from row 1; 42 extras. preds/ copies are byte-equal to nn_x/w5/preds. | read, 09 |
| Test-side sanity (label-free) | W8 / S16 / C3 branch files: 2,904,806 rows, no NaN; corr(FT, TO) test 0.9916 vs val 0.9907; corr(W8, C3) test 0.9938 vs val 0.9939. | 09 |
| Seed coupling teacher - student | Same-seed student vs base member corr 0.9643, different seed 0.9625 (val): +0.002, immaterial. | 07 |

## 2. Risks and unfair comparisons

**R1. Checkpoint selection on the holdout favours the hard-label base (measured).** Selected minus @last, mean over 4 seeds:

| arm | member val | member holdout | 4-pool holdout (sel / last) |
|---|---|---|---|
| base | +0.00030 +- 0.00023 | **+0.00036 +- 0.00012** | 0.32328 / 0.32287 |
| kd05 | +0.00000 +- 0.00008 | +0.00006 +- 0.00004 | 0.32349 / 0.32342 |
| kd05w192 | +0.00005 +- 0.00012 | +0.00011 +- 0.00007 | 0.32408 / 0.32403 |
| kd05t25w192 | +0.00018 +- 0.00015 | +0.00022 +- 0.00008 | 0.32412 / 0.32420 |

KD arms select the last or second-to-last evaluation anyway. Read @last (no selection on the witness):

| arm vs base | member val | member holdout | 4-pool val | 4-pool holdout | 4-pool holdout as logged (selected) |
|---|---|---|---|---|---|
| kd05 | +0.00287 +- 0.00047 | +0.00229 +- 0.00028 | +0.00143 +- 0.00060 | **+0.00055 +- 0.00036** | +0.00021 |
| kd05w192 | +0.00289 +- 0.00048 | +0.00262 +- 0.00028 | +0.00205 +- 0.00065 | **+0.00115 +- 0.00037** | +0.00080 |
| kd05t25w192 | +0.00310 +- 0.00051 | +0.00244 +- 0.00030 | +0.00161 +- 0.00067 | **+0.00133 +- 0.00038** | +0.00084 |
| kd08 | +0.00299 +- 0.00050 | +0.00299 +- 0.00028 | +0.00055 +- 0.00061 | +0.00081 +- 0.00034 | +0.00049 |

No decision changes. Recommendation: read every KD-vs-hard-label comparison @last (files exist for every arm).

**R2. The val bar of +-0.0005 is about 1 SE for pool-vs-pool reads** (SE 0.0005-0.0007); the holdout SE is 0.0003. Example: kd05w192 "failed by 0.00001". At equal pool size, W8 vs the 8 narrow students: val +0.00038 +- 0.00049, **holdout +0.00083 +- 0.00028**. The holdout also agreed with public on W8 > S16 while val did not. For student-pool composition the holdout at 2 SE with a val sign check is the rule the data support.

**R3. `--kd_T` is one-sided.** The teacher is softened by T, the student is scored at T = 1. kwT07 / kwT15 therefore test sharper / smoother targets, not temperature distillation (both softmaxes at T, loss x T^2), which moves gradient from the head of the list to the middle and tail. A null on kwT07/15 does not close that variant. Measured calibration of the teacher: the CE-optimal T of mean11 is 1.10 on the holdout (out-of-sample, CE 4.4688 at T 1 vs 4.4669) and 0.90 in-sample: T = 1 is already near the optimum for the one-sided form.

**R4. The kd_w dose-response is confounded with the K9 auxiliary head (measured from CLIPLOG).** Median gradient norms, 4 seeds:

| arm | plateau main | plateau aux | aux / main plateau | aux / main cooldown |
|---|---|---|---|---|
| base | 0.280 | 0.048 | 0.17 | 0.07 |
| kd03 | 0.207 | 0.049 | 0.24 | 0.10 |
| kd05 | 0.156 | 0.050 | 0.32 | 0.15 |
| kd08 | 0.077 | 0.052 | 0.67 | 0.33 |
| kd10 | 0.037 | 0.051 | **1.37** | **1.23** |
| ap1 (hard labels, aux 1.0) | 0.283 | 0.099 | 0.35 | 0.11 |

The main gradient norm is proportional to (1 - kd_w): the per-batch gradient is label noise and the teacher term adds almost none. The optimisers normalise the update, so the aux share of the update doubles at kd05 and exceeds the main loss at kd10. At kd05 the ratio (0.32) equals that of ap1, which was a measured null under hard labels, so the kd05 / W8 recipe is inside the tested-flat range and the aux weight needs no change there. kd08 and kd10 are outside it: their weaker end level is not a clean statement about "less hard label".

**R5. Cross-fitted teacher: the arm changed three things** (out-of-sample, 3 instead of 11 members, 80% data and steps). The closure still stands, by mechanism (section 5.5).

**R6. The kw_ levers were closed on a member screen that width itself fails** (kd05w192 vs kd05 member: val +0.00007, holdout +0.00038). Re-read at pool level here (2-member pools vs the 2-member pool of the reference arm on the same seeds, selected checkpoints, val / holdout): drop0 -0.0003 / 0.0000, e20 0.0000 / -0.0002, hl2 -0.0003 / -0.0010, ff4 -0.0009 / -0.0006, id -0.0051 / -0.0046, w256 -0.0011 / -0.0005, b3 -0.0010 / -0.0001 (SE 0.0007 / 0.0004). @last gives the same picture (best case drop0 +0.0004 +- 0.0008 val with -0.0002 holdout). **All closures hold at pool level** (same result as `kd_reprice`).

**R7. XPOOL (closed) holdout read was token-confounded**: 3,345 of the 30,000 holdout feeds have p2 <= 0.5, so under `--prand2` they were scored with a production-leaning domain token. The closure rests on val, which is unaffected.

## 3. The free-win checks that were asked for

| question | answer | evidence |
|---|---|---|
| KD target zeroed for feeds without positives | Never happens: 0 such feeds in train, none created by `--aug_cand`. | 02 |
| Teacher is in-sample | Worth nothing: the in-sample teacher puts only +0.0020 to +0.0069 more probability mass on the clicked items than the cross-fitted teacher (0.2% to 0.7% of the label share). | 5.5 |
| K9 aux weight under KD | Its gradient share doubles (R4) but stays in the range where the aux dose was flat. No action at kd_w 0.5. | 04 |
| Cooldown exploration mix with KD targets | Consistent (pool batches get the same mixture and the same domain token as in the teacher dump). Pool feeds are the most memorised feeds (wide students: in-sample NDCG 0.3488 on exploration train feeds vs 0.3245 on the holdout), but in target space that is +0.003 of mass (5.5), so nothing to fix. | 05 |
| Distil on the 30K holdout feeds | No: +1% data and it would destroy the only independent witness. | - |
| Distil on val / test feeds (features only) | In training: 1.4% of the volume; the U20 arms now running measure exactly this exchange rate (7.1). In the fine-tune: tested here with a CPU replica, null in the file (section 6). | 6, 7.1 |
| masked_fill(-1e4) under bf16 | Safe: head and losses are fp32. | 08 |
| kd_T and per-feed centring | No interaction; but kd_T is one-sided (R3). | read |

## 4. What KD is in this pipeline (measured)

**4.1 Noise reduction, not knowledge transfer.** (a) Main gradient norm proportional to 1 - kd_w (R4). (b) Students are equal across four teachers of very different quality:

| teacher | teacher holdout NDCG | student arm (d192, kd_w 0.5) | member val / holdout |
|---|---|---|---|
| cross-fitted (3 members, 80% data) | 0.32100 | kx05 | 0.39916 / 0.32237 |
| mean25 | 0.32286 | kd05t25w192 | 0.39952 / 0.32259 |
| mean11 | 0.32370 | kd05w192 | 0.39918 / 0.32267 |
| 8 wide students | 0.32453 | kd05g2w192 | 0.39914 / 0.32252 |

**4.2 Plateau level rises with kd_w, the cooldown gain does not carry it.** Mean over 4 seeds, val / holdout:

| arm | level at step 77.5K | gain 77.5K -> end | end |
|---|---|---|---|
| base | 0.3930 / 0.3171 | +0.0032 / +0.0029 | 0.3962 / 0.3199 |
| kd03 | 0.3949 / 0.3188 | +0.0033 / +0.0026 | 0.3982 / 0.3214 |
| kd05 | 0.3959 / 0.3196 | +0.0032 / +0.0026 | 0.3991 / 0.3222 |
| kd08 | 0.3975 / 0.3209 | +0.0017 / +0.0020 | 0.3992 / 0.3229 |
| kd10 | 0.3970 / 0.3213 | +0.0018 / +0.0012 | 0.3988 / 0.3225 |

Every KD arm ends at val 0.3988-0.3993: a single-model ceiling that neither the teacher, kd_w >= 0.5, width, depth nor epochs moves.

**4.3 Student diversity comes only from the student's own seed.**

| arm | within-arm corr (val) | member val | 4-pool val | lift |
|---|---|---|---|---|
| base | 0.9485 | 0.39654 | 0.39887 | +0.0023 |
| kd03 | 0.9657 | 0.39806 | 0.39977 | +0.0017 |
| kd05 | 0.9776 | 0.39911 | 0.40085 | +0.0017 |
| kd05w192 | 0.9722 | 0.39918 | 0.40134 | +0.0022 |
| kd08 | 0.9904 | 0.39928 | 0.40004 | +0.0008 |
| kd10 | 0.9950 | 0.39883 | 0.39906 | +0.0002 |

Cross-teacher correlation equals within-arm correlation: kd05w192 x kd05t25w192 0.9718, x kx05 0.9719, x kd05g2w192 0.9709 (within 0.9722). Mixed-teacher 4-pools (2 + 2, all 36 combinations) equal same-teacher 4-pools: val -0.00004 +- 0.00017, holdout +0.00006 +- 0.00010. **Giving each student a different teacher buys nothing; W8 over a 4-pool is a pure size effect.**

**4.4 Pool-size law.** Pools of k = 1, 2, 3, 4 (mean over subsets), val: kd05w192 0.39918, 0.40056, 0.40099, 0.40134; kd05t25w192 0.39952, 0.40062, 0.40105, 0.40100. Fit a - b/k per arm: a = 0.4021 (b 0.0029) and a = 0.4015 (b 0.0020); the two arms pooled give a = 0.4018 and predict W8 0.4015 (measured 0.40157). Inferred: W16 = W8 + 0.0001 to 0.0003, infinite pool at most +0.0005. On the holdout W8 (0.32456) already sits at the fitted asymptote (0.3246). `kd_reprice` (Q 0.40154 / 0.32463) and `pool_composition` (0.40189 / 0.32469) fit more members and land in the same band.

## 5. Details behind the nulls

**5.5 In-sample teacher (measured, labels used for diagnostics only).** Teacher mass on the clicked items, softmax at T = 1:

| feeds | mean11 (in-sample) | cross-fitted | difference | KL(mean11 // oof) per feed |
|---|---|---|---|---|
| holdout (both out-of-sample) | 0.0322 | 0.0316 | +0.0006 | 0.0160 |
| exploration train, M >= 50 | 0.0347 | 0.0317 | +0.0030 | 0.0165 |
| production train, M >= 50 | 0.0376 | 0.0355 | +0.0020 | 0.0114 |
| production train, M < 50 | 0.1365 | 0.1296 | +0.0069 | 0.0102 |

The in-sample NDCG excess (0.344 vs 0.324) looks large but is 0.003 of probability mass. The two teachers differ by the same KL on feeds where both are out-of-sample as on feeds where one is in-sample: the difference is ensemble size, not memorisation. Cross-fitting cannot help; the closure is confirmed.

**5.6 Sub-feed consistency (measured, CPU forward on 2,500 val feeds, exact `--aug_cand` rule).** KL between the full-feed logits restricted to the kept candidates and the logits recomputed on the sub-feed, same model: 0.0021 nats (aug 0.5) and 0.0053 (aug 0.2); within-feed logit corr 0.9978. Reference: KL between two seeds 0.050 (base) / 0.027 (students); teacher to student 0.027. The full-feed teacher is a valid target for a sub-feed, so **kwsub5 / kwsub2 are fair tests of function matching** and a consistent-teacher variant is not needed. Same run: switching the domain token from random-like to production moves the softmax by KL 0.0044, so a "counterfactual random-token teacher" is also a null.

## 6. The fine-tune and KD students

Member-level FT minus TO on val (OOF, n2cpv3s3), measured:

| family | members | FT - TO |
|---|---|---|
| r6_cd_m02 | 7 | +0.00046 +- 0.00040 |
| r6_cd_m02_p70k | 7 | +0.00046 +- 0.00038 |
| v23_gru_pre2s | 6 | +0.00075 +- 0.00039 |
| abl_e10_pop1 | 8 | +0.00118 +- 0.00047 |
| kd05 / kd05t25 (d128 students) | 4 / 4 | +0.00001 +- 0.00044 / -0.00003 +- 0.00042 |
| kd05w192 / kd05t25w192 (W8) | 4 / 4 | -0.00012 +- 0.00044 / -0.00073 +- 0.00045 |

Pooled, paired per feed: 16 students -0.00022 +- 0.00038; the 8 wide students -0.00043 +- 0.00041; the 28 C3 members +0.00073 +- 0.00032; **students minus C3 members -0.00095 +- 0.00024**. (`ft_redesign` shows the same deficit for the hard-label w5_e10_base twins, so it belongs to the recipe family, not to distillation.)

Per-epoch OOF from the fine-tune logs (member mean, epoch 0 = train-only): W8 members peak at epoch 1 (kd05w192 +0.00063, kd05t25w192 +0.00031) and the saved EMA-final is 0.0008-0.0010 below that peak; C3 families peak at epoch 1 too (+0.0011 to +0.0015) and lose only 0.0004-0.0006 afterwards. Two repairs suggest themselves: soft targets inside the fine-tune, and stopping after epoch 1. Both were tested.

**CPU replica of the fine-tune** (`11_ftkd_cpu.py`: the ft_w5.sh recipe in fp32, member w5_e10_kd05w192_s951, folds 0 and 1 = 5,770 held-out feeds, FT seed 0; the replica reproduces the GPU log of fold 0: 0.4060 -> 0.4074 -> 0.4073 -> 0.4069 -> 0.4067 against 0.4056 -> 0.4073 -> 0.4064 -> 0.4061 -> 0.4061). Soft-target arm: target = 0.5 y / sum(y) + 0.5 softmax(mean centred train-only logits of the 8 wide students on val), label-free teacher, same seed and batch order.

| read (paired, 5,770 feeds) | member | in the file: 0.5 z(FT member) + 0.5 z(W8 TO pool) |
|---|---|---|
| soft-target FT minus standard FT (EMA-final) | +0.00040 +- 0.00059 | **+0.00002 +- 0.00044** |
| standard FT, epoch 1 minus EMA-final | +0.00094 +- 0.00071 | **-0.00035 +- 0.00050** |
| standard FT, epoch 2 minus EMA-final | +0.00069 +- 0.00043 | -0.00054 +- 0.00034 |
| standard FT (EMA-final) minus train-only | -0.00019 +- 0.00097 | +0.00017 +- 0.00088 (file minus TO pool) |

The soft-target fine-tune stays closer to the train-only member (corr 0.9934 against 0.9879) and loses nothing, but adds nothing once the TO pool is in the file. The epoch-1 advantage of the member turns into a small loss in the file. One member, one seed, two folds: a proxy, but both signs agree with the independent pool-level analysis in `ft_redesign` (the fine-tune shift is common to all members and is not improved by shrinking it). **Soft-target fine-tune and one-epoch fine-tune: CLOSED as file levers, 0 GPU-h.**

Zero-GPU file recipes with a different member set in the FT branch than in the TO branch (`13_recipes.py`, val, vs W8 final 0.40165): FT = C3 -0.00029 +- 0.00039; FT = 0.5 z(C3) + 0.5 z(W8) -0.00017 +- 0.00027; FT = 36 members member-equal -0.00001 +- 0.00034; FT = S16 +0.00009 +- 0.00024. All null.

**Side observation (bf16 inference).** The replica's epoch-0 prediction (fp32) and the stored GPU train-only file (bf16 trunk) of the same checkpoint differ within feeds by sd 0.013 logits (p99 0.047, max 0.61; centred corr 0.99994). On folds 0 + 1 the fp32 pass reads +0.00081 +- 0.00034. `winner_model` ran the full val for four members (+0.00038, +0.00001, +0.00015, -0.00023; 4-pool -0.00012 +- 0.00015), so this is one member's fluctuation, not a free win. Consequence for reading numbers: a CPU fp32 replay of a member is not interchangeable with its stored GPU file at the 0.0005 level.

## 7. Proposals, ranked

### 7.1 Audit of the 20:20 UTC patch (`--xfold_unlab`, arms u20a / u20b / u20c, kwaux0, kwwd05, kwh30), read before the runs started

- `--xfold_unlab`: the fold's feeds stay in `feeds`; their target is the teacher softmax only (kd weight 1 per feed), the label enters only through the always-true `y.sum > 0` gate; the K9 aux weight is zeroed for them; they are removed from the cooldown pool; holdout feeds were excluded earlier. **No label path, no bug found.**
- `--kd_T 3` on teacher_xf0 (a SUM of 3 members) is the correct sum-to-mean conversion, and it is correct only because kd_T divides the teacher alone (R3).
- kwaux0: `--aux_pos 0` builds the model without the aux head and skips the aux loss; the CLIPLOG split then differentiates a zero tensor (harmless).
- **Read-out caveat 1 (step budget).** u20a trains 78,710 steps, u20b and u20c 98,350. The declared ratio (B - A) / (C - A) is the right statistic because B and C share the budget; B - A alone contains "+25% steps".
- **Read-out caveat 2 (who is the witness).** 5,974 of the 30,000 holdout feeds belong to fold-0 users (A never saw the user, B saw the user's feeds without labels, C with labels; about 300 of them are user-0 feeds assigned by feed index); the other 24,026 measure the cross-user effect. On val only 4,386 of 14,423 feeds have a user with any train feed (852 of them in fold 0), so val is almost purely the unseen-user case, which is the case that matters for "val / test feeds as a transfer set". The U20 rule should be read on three columns: holdout fold-0 users, holdout other users, val. A gain that lives only in the first column does not support the transfer-set programme. Fold ids are saved: `nn_x/w12/recheck/kd_audit/out/xfold_holdout.npy` (aligned with holdout_meta order), `out/xfold_val.npy` (-1 = user 0).

### 7.2 Ranked

| # | item | kind | evidence (measured unless noted) | cost | P |
|---|---|---|---|---|---|
| 1 | The KD pipeline is clean: no bug, no leak, numbers reproduce | null | section 1 | - | real 0.95 |
| 2 | Read U20 by user fold (7.1) | risk | 20% of holdout feeds carry the same-user effect; 70% of val feeds are unseen users | 0 | real 0.5 |
| 3 | Read KD vs hard-label arms @last | risk | base loses 0.0004 of 4-pool holdout without selection, KD pools lose 0; kd05 pool +0.00021 -> +0.00055 +- 0.00036 | 0 | real 0.9 |
| 4 | kd08 / kd10 are confounded with the aux head; kwaux0 decides whether the dose needs a re-read | risk | aux / main gradient ratio 0.17 (base), 0.32 (kd05), 0.67 (kd08), 1.37 (kd10) | 0 now | real 0.3 |
| 5 | Phase-split re-cool from the stored @77500 snapshots (kd08 plateau -> kd_w 0.5 cooldown; plus one arm kd_w 0.8 with `--aux_pos 0.2`) | lever | plateau level kd08 - kd05 +0.0016 / +0.0013; cooldown gain kd05 +0.0032 / +0.0026 vs kd08 +0.0017 / +0.0020; but every KD arm ends at val 0.3988-0.3993 (ceiling) | 16 d128 re-cools x 23 min = 6 GPU-h, no code change (`--init` + `--kd_file` already work together) | 0.04 |
| 6 | Two-sided temperature (both softmaxes at T, loss x T^2) is not closed by kwT07 / kwT15 | lever | kd_T is one-sided in the code; teacher calibrated at T 1.0-1.1; interim plateau read kwT07 ahead, kwT15 behind (not a decision) | 6 lines in a train.py copy + 2 d192 seeds = 6.5 GPU-h | 0.03 |
| 7 | Soft-target fine-tune, one-epoch fine-tune, asymmetric FT / TO member sets | null | section 6 | - | closed |
| 8 | Different teacher per student | null | cross-teacher corr 0.9718 = within-arm 0.9722; mixed 4-pools -0.00004 / +0.00006 | - | closed |
| 9 | More same-recipe wide students after W16 | null | 1/k law, asymptote val 0.4015-0.4021 (W8 0.40157); holdout already at the asymptote | - | closed |
| 10 | Cross-fitted teacher | null | +0.2% to 0.7% label mass; KL identical where both teachers are out-of-sample | - | closed (confirmed) |
| 11 | Consistent (re-scored) teacher for sub-feeds; random-token teacher | null | KL 0.002-0.005 vs 0.027-0.050 seed floor; token KL 0.004 | - | closed |
| 12 | Decision rule for student pools: holdout at 2 SE, val as sign check | risk | pool-vs-pool val SE 0.0005-0.0007 = the bar; holdout SE 0.0003 and agreed with public on W8 > S16 | 0 | real 0.7 |

Item 5 in commands (control and treatment share the harness; variant names are suggestions):

```
B="--epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.02 --loss_norm feed --holdout 30000 --select_on holdout --eval_every 1000 --save_last --save_holdout --select_min_step 12000 --cliplog 50"
T=<repo>/nn_x/w12/kd/out/teacher_mean11.npy; C=<repo>/nn_x/w5/ckpt
python train.py --variant w5_cdk_k05k05_s951 --init $C/w5_e10_kd05_s951@77500.pt $B --seed 1851 --kd_file $T --kd_w 0.5      # control, reproduces kd05
python train.py --variant w5_cdk_k08k08_s951 --init $C/w5_e10_kd08_s951@77500.pt $B --seed 1851 --kd_file $T --kd_w 0.8      # control, reproduces kd08
python train.py --variant w5_cdk_k08k05_s951 --init $C/w5_e10_kd08_s951@77500.pt $B --seed 1851 --kd_file $T --kd_w 0.5      # phase split
python train.py --variant w5_cdk_k08a02_s951 --init $C/w5_e10_kd08_s951@77500.pt $B --seed 1851 --kd_file $T --kd_w 0.8 --aux_pos 0.2   # aux share back to the kd05 level
```
Rule to fix before running: k08k05 (or k08a02) vs k05k05, 4 seeds, read @last: 4-pool holdout >= +2 SE (about +0.0007) and val >= 0, otherwise the kd_w schedule and the aux-share question are closed together. My expectation is a null (the common end level of all KD arms points to a ceiling that the route does not change).

### 7.3 Small notes

- `feed_rank_z(...)[:, 1]` is the per-feed z-score of the raw score (column 0 is the rank). Pools are therefore means of z-scored logits, not rank-normalised scores; all numbers here use the same function as build_w.py. Naming only.
- Interim matched-step plateau reads of the running arms vs kd05w192 (same seed, evaluations from step 10K; kd08 showed that a plateau lead does not predict the end): kwsub5 -0.0002 / -0.0005 and -0.0005 / -0.0003; kwsub2 -0.0004 / -0.0005 and -0.0004 / -0.0007; kwT07 +0.0007 / +0.0002 and +0.0012 / +0.0010; kwT15 -0.0005 / -0.0007 and -0.0006 / -0.0006; kd05w224, GRU and kdmix within +-0.0005 (`12_interim.py`).
- W8 + 4 pop1-view students at TO level: val +0.00020 +- 0.00046, holdout -0.00113 +- 0.00030 (control W8 + 4 kd05t25: -0.00007 / -0.00025). The W8P20 probe is declared with the holdout not a veto; this is the size of what is being overridden.

## 8. What I could not check

- Anything that needs a GPU forward on train rows: the K9 aux head's logit range under the production token (measured only on val, random-like token), teacher logits under a different domain token on train feeds, a holdout read of the fine-tune proxy (the fine-tune uses val labels, so only val can witness it).
- The fine-tune proxy is one member, one FT seed, two folds; the pool-level statement rests on its agreement with `ft_redesign`.
- `w5_eval_e_ap1.py` (the E-rule read-out script) and `src/presub_gate.py` were not re-audited.
- The xp2 (XPOOL) holdout confound (R7) was not re-computed with matched tokens; the closure does not depend on it.

## 9. Files

- `nn_x/w12/recheck/kd_audit/01_ckpt_args.py` teacher-member provenance; `02_counts_align.py` zero-positive counts, alignment, assembly; `03_pools.py` pool numbers, selected vs @last, w_ft scan; `04_cliplog.py` gradient norms; `05_teacher_calib.py` teacher temperature and label mass; `06_curves.py` curves by step; `07_diversity.py` correlation structure, mixed-teacher pools, size law; `08_subfeed_cpu.py` (+ .log) sub-feed / domain-token consistency; `09_ft_test.py` FT vs TO by family, test sanity; `10_kw_pool.py` kw_ levers at pool level; `11_ftkd_cpu.py` (+ .log, out/) fine-tune proxy; `12_interim.py` matched-step read of running arms; `13_recipes.py` file recipes, pooled FT - TO, W8 + pop1 students; `14_seedsd.py`; `15_centre_check.py`; `16_ftkd_read.py` proxy read-out; `out/xfold_holdout.npy`, `out/xfold_val.npy` user-fold ids for the U20 read-out; `out/ftkd_*.npy` proxy predictions. Not run (superseded): `11b_ftkd_cpu_epochs.py`, `17_e1_read.py`, `18_fp32_infer.py`.
