# RECHECK 11-14, lens "long_bets": multi-day bets for the remaining 3.5 weeks

Written 2026-11-14 20:44 UTC (final; a first draft was on disk from about 20:00 UTC). CPU only (≤ 12 threads), nothing trained on a GPU, no running job touched, no file under `src/` or `nn_x/w5/` edited, no submission file written.
Scripts and logs: `nn_x/w12/recheck/long_bets/` (logs in `out/`). Deltas are paired per feed, "a ± b" = mean ± 1 SE; "val" = 14,423 feeds, "hold" = the 30K in-period holdout.
P1 / P2 = my probability of ≥ +0.001 / ≥ +0.002 public at pool level.
Status note: the lead acted on the first draft of this report (U20 patch applied 20:20Z; arms u20a/b/c, kwh30, kwaux0, kwwd05 queued as wave 1; F1 copied to CURRENT_STATUS). The probes F9 and F12 were run after that draft and change two priors, stated in §3: U20 is now predicted KILL and kwh30 predicted null. Both predictions are written here before any number of those runs exists.

## 0. Bottom line

1. **I could not design an own-model bet with P2 above 0.03.** Fourteen CPU probes agree: every teacher is the same function, students add seed noise around it, and the distilled pool has a measured limit of val ≈ 0.4020 / hold ≈ 0.3250 (W8 is at 0.40157 / 0.32456). The only item that can reach +0.002 is a merge with a team whose model is not a neural history ranker.
2. **"Distil on more data" is closed by measurement.** Students match their teacher exactly as well on feeds they never saw as on feeds they trained on (logit corr 0.9933 vs 0.9933, 30K feeds each). Function matching is not input-limited, so teacher-labelled sub-feeds, test / val feeds, pseudo-feeds and candidate swaps have nothing to add. The running kwsub arms are predicted null (interim −0.0002..−0.0006).
3. **What limits the ranker is labels.** Removing 20% of users costs a hard-label member −0.00275 ± 0.00049 val / −0.00217 ± 0.00028 hold and a 3-member pool −0.00167 ± 0.00061 / −0.00186 ± 0.00035: +0.005 to +0.006 per doubling at POOL level (the record says +0.0037, member level). No labelled feeds are unused except the 30K holdout. Rule that follows: never train members on a subset of users.
4. **Own bets, ranked** (§3): the (teacher weight × width) surface (P1 0.05); schedule levers re-screened under KD (P1 0.04); full-data refit of the final students (P1 0.03, free); soft-target fine-tune (P1 0.03 after the `ft_redesign` pilot failed its rule); U20 (queued, predicted KILL). Expected sum ≈ +0.0004 public.
5. **Closed today** (§4): disjoint-teacher students, user-bagged members, cross-fitted chains, GBDT student, privileged item-CTR teacher, co-click label-conditional targets, noisy-student views, transductive KD.

## 1. Measurements (all new today)

| # | question | result | reads as | log (`out/`) |
|---|---|---|---|---|
| F1 | local data slope: hard-label base without one user fold (xf0-4 × 3 seeds = 15 runs, 2.50M feeds, 78.7K steps) vs all users | member −0.00275 ± 0.00049 val / −0.00217 ± 0.00028 hold; 3-pool, mean of 5 folds vs the same-seed base 3-pool: −0.00167 ± 0.00061 / −0.00186 ± 0.00035 | +0.0074 / +0.0065 per doubling (member), +0.0052 / +0.0058 (pool). Present in every M bucket (−0.0037 at M < 20, −0.0021 at M ≥ 120). Caveat: 20% fewer optimiser steps ride along; the epoch dose-response (15 / 20 epochs negative) bounds that at about 0.0003 | p1_variance, p1b_poolslope |
| F2 | seed noise vs data noise | member corr: same fold, different seed 0.9404 val; different folds 0.9362 (two folds share 75% of their users) | seed noise dominates member disagreement | p1_variance |
| F3 | user-bagged members | 4 members from 4 folds − 4 all-user members: −0.00114 ± 0.00062 val / −0.00184 ± 0.00036 hold; 15 bagged vs 4 base: −0.00041 ± 0.00061 / −0.00078 ± 0.00036 | bagging by users loses more strength than it gains diversity | p1_variance, p1b |
| F4 | pool-size limit of wide students | W8 subsets k = 1 / 2 / 4 / 8: val 0.39935 / 0.40053 / 0.40118 / 0.40157, hold 0.32263 / 0.32358 / 0.32411 / 0.32456; 1/k extrapolation 0.40195 / 0.32500 (kd05 d128: 0.40142 / 0.32387) | W16 ≈ +0.0002 / +0.0002 over W8; an infinite pool +0.0004 | p1_variance |
| F5 | function matching on the holdout (teacher_mean11 is honest there: 0.32370) | pure-teacher students (kd10): member-member corr 0.9951, 4-pool = teacher −0.00079 ± 0.00028. kd05w192 4-pool = teacher +0.00038 ± 0.00033 | a student has a seed-shared gap to the 11-model mean; the hard-label half is what lifts the pool above the teacher | p5_fmatch |
| F6 | does a stronger teacher give stronger students? | teacher hold 0.3210 / 0.3229 / 0.3237 / 0.3245 (oof, mean25, mean11, w8g1) → d192 4-pool hold 0.32347 / 0.32412 / 0.32408 / 0.32410 | saturated above 0.323. The cross-fitted arm lost because its teacher is 3 members on 80% of users (F1), not because it is out of sample | p5_fmatch |
| F7 | what the in-sample teacher memorises (30K exploration train feeds, in-sample − cross-fitted logit) | clicked items +0.141 logit (teacher sd 1.18); non-clicked items nearest to a click +0.042, farthest −0.027 | the label copy is small; an "honest" teacher is not a lever | p2_coclick |
| F8 | students of different teachers (mean11, mean25, wide students, cross-fitted) | cross-teacher member corr 0.9699-0.9720 val = within-teacher 0.9702-0.9736; pool-pool 0.992-0.994; mixed 2 + 2 pools −0.0002..+0.0005 ± 0.0004 | no lineage diversity: all teachers are one function | p9_lineage |
| F9 | **seen vs unseen feeds**: student pool (4 × kd05w192, dumped on all train rows) vs its teacher, per-feed logit corr | seen exploration train feeds 0.9933, unseen holdout feeds 0.9933 (non-clicked candidates only: 0.9933 / 0.9933). Two hard-label groups: 0.9740 / 0.9740 | function matching has no generalisation gap. More teacher-labelled inputs cannot improve it | p11_seen_unseen |
| F10 | are cheap augmented views new inputs? CPU inference of kd05w192_s951 on 8,000 / 4,000 val feeds | random 50% sub-feed: logit corr to the full-feed logits 0.9949 (20% sub-feed 0.981); stored full-feed logits rank the sub-feed as well as recomputed ones (recomputed −0.0009 ± 0.0009). History-event drop 15% / 30% / 50%: corr 0.9957 / 0.990 / 0.980, NDCG −0.0008 ± 0.0009 / −0.0006 ± 0.0011 / −0.0042 ± 0.0014 | the score is nearly separable per candidate and robust to history thinning. Stored logits are a valid sub-feed target, so kwsub is a fair test | p7_subfeed, p7b_histdrop |
| F11 | student agreement by split (W8, train-only) | 1 − member corr: hold 0.0261, val 0.0274, public 0.0279, private 0.0277 | no extra function gap on test feeds | p4_partner |
| F12 | where extra capacity goes (member corr to its own teacher, holdout) | kd_w 0.5, teacher mean11: d128 0.9867 → d192 0.9834; teacher mean25: d128 0.9867 → d192 0.9831 → d256 0.9795; at d192 also 20 epochs 0.9790, no dropout 0.9771, FFN x4 0.9822. kd_w 0.8: d128 0.9923 → d192 0.9924 with member hold +0.0004, val +0.0008 (1 seed) | at teacher weight 0.5 capacity is spent on the one-hot half (the student moves away from the teacher); at 0.8 it is not | p12_width_match |
| F13 | frontier law at the W8 level, 10 partner pools on disk | the law on record predicts each equal mix within 0.0002 (C3 −0.00013 vs −0.00027 ± 0.00039; G3 −0.00038 vs −0.00044; CatBoost −0.0045 vs −0.0046). Hard-label NN pools sit at rho 0.984-0.994 to W8; no blend is positive | nothing on disk adds to W8; the law can price a merge partner | p4_partner |
| F14 | fine-tune effect per member (n2cpv3s3 val OOF − train-only) | pop1 +0.00118, GRU +0.00075, F7x +0.00046, kd05 +0.00001, W8 **−0.00043 ± 0.00041** (M < 50: −0.00114). W8 FT pool − TO pool −0.00069 ± 0.00061 | the member gain of the fine-tune is gone for the w5_e10 recipe family and negative on short feeds. `ft_redesign` shows the hard-label twins (w5_e10_base) share it (−0.00029 ± 0.00041), so it is the recipe family, not distillation | p4_partner |
| F15 | checkpoint selection for students | W8 pool @last − selected: +0.00014 ± 0.00020 val / −0.00009 ± 0.00011 hold (hard-label base: +0.0005 / −0.0004) | distilled students no longer need the holdout to pick a checkpoint | p10_last |
| F16 | privileged item-level click aggregate (TOP1_GAP_EXTERNAL lever 3): leave-one-out item CTR / COEC over the whole train window, added to the teacher on the holdout (user-half fit) | +0.00005 ± 0.00018 (CTR), +0.00019 ± 0.00021 (COEC); the exposure-count control alone +0.00035 ± 0.00021 | the teacher already knows item click propensity | p6_pfd_item |
| F17 | label-conditional information: hide one positive, reveal the feed's other positives (10,720 holdout feeds with ≥ 2 positives) | hidden-positive NDCG 0.26919 → 0.27911 (+0.00992 ± 0.00067) | real, but a student trained on the shifted target is worse (§4) | p2_coclick |
| F18 | GBDT as a student (LightGBM, cb26 feature set, 177 features, exploration feeds) | table in §4 | soft targets are worth +0.005..+0.008 to a GBDT; level stays 0.017 below W8 | p8_gbdt_eval, log_gbdt_* |

Interim reads of the running arms, mean over matched steps ≥ 5K vs kd05w192 same seed (logs only, not final; `interim.py`): kwsub5 −0.0002 val / −0.0002 hold; kwsub2 −0.0005 / −0.0006; kwT07 +0.0007 / +0.0004; kwT15 −0.0007 / −0.0006; kd05w224 +0.0001 / 0.0000; kdmixw192 +0.0003 / 0.0000; kd05w192gru +0.0001 / +0.0001.

## 2. What the measurements imply

- **One function, seed noise around it** (F5, F6, F8). Teacher identity does not matter above holdout 0.323 and does not decorrelate students. Averaging the seed noise ends at 0.4020 / 0.3250 (F4). Generation 2, bigger, mixed and cross-fitted teachers and other-view students change none of these terms, which is why they all landed on the same line.
- **The student under-fits its teacher; it does not over-fit it** (F9). So nothing that adds inputs or input noise can help matching (F10 adds that the cheap views are not new inputs anyway). Only capacity or optimisation can, and at teacher weight 0.5 capacity is diverted into the one-hot half (F12).
- **Labels are the scarce resource** (F1). Teacher targets carry no new labels. Unused labels: the 30K holdout (F15 frees it) and nothing else.
- **The fine-tune is the one stage still trained on raw one-hot labels** (F14), but the `ft_redesign` lens bounds any redesign of it at about +0.0005 and its soft-target pilot failed at file level.

## 3. Ranked bets

### Bet 1. Cross-team merge (owner's decision, 0 GPU): P1 0.35 given a merge, P2 0.12

Mechanism: the only source of a different error pattern. F13 confirms the law at the W8 level, so a partner's value can be read from two label-free numbers after the merge.

| partner pool | rho to W8 on test (assumed range) | public gap | equal-mix gain over the better pool: law on record / refit on today's 10 partners |
|---|---|---|---|
| another neural history ranker | 0.96-0.98 | 0 | +0.0004..+0.0007 / +0.0002..+0.0004 |
| the same, 0.001 apart | 0.96-0.98 | 0.001 | 0.0000..+0.0004 / −0.0001..+0.0001 |
| GBDT or feature pipeline at our level | 0.86-0.90 | 0 | +0.0019..+0.0026 / +0.0010..+0.0014 |
| the same, 0.002 weaker | 0.86-0.90 | 0.002 | +0.0011..+0.0018 / +0.0004..+0.0008 |
| anything | ≥ 0.99 | any | ≤ 0 |

- Partner to look for: public within 0.002 of ours AND a model that is not a history-attention neural ranker. A second neural team is worth about a quarter of that.
- Decisive check, after the formal merge: within-feed correlation of the two test files + the public gap → law. Blend only if law − 0.0005 ≥ 0. One public slot; kill if blend < better solo − 0.0018 (GLOBAL_RECAP §6.4).
- Exchange deadline Dec 1 (rules §5.5).
- We cannot build this partner ourselves: a distilled GBDT is at rho 0.90 and 0.017 below W8, and the law needs ≤ 0.005 (§4).

### Bet 2. The (teacher weight × width) surface: P1 0.05, P2 0.01

Mechanism: F12. Width moved the pool limit by +0.0005 val / +0.0011 hold from d128 to d192 (F4) and then stopped (d256 failed, d224 flat so far), all at teacher weight 0.5, where extra capacity is spent on the one-hot half. At weight 0.8 the one seed on record keeps its teacher correlation and gains. The surface has only been explored along one line.
- Stage 1 (zero patch; 4 seeds each because member gains do not predict pools here, 12 runs, ~3 h): `kd065w256`, `kd08w256`, `kd08w320` (§6).
- Rule (the generation-2 rule): 4-pool vs kd05w192 4-pool (0.40134 / 0.32408) has val ≥ 0 AND hold ≥ 0 AND mean ≥ +0.0004; and [W8 + arm × 4] vs W8 has hold ≥ +0.0003, val ≥ 0.
- Information arm, 2 runs: `kd10w192`. If its pool reaches the teacher (F5 gap −0.0008 closed), capacity has done all it can for matching and the bet closes whatever stage 1 says.
- Stage 2 (1 day, only on a pass): 8 seeds of the passing cell, FT, file.
- Risk seen in the data: higher teacher weight makes students collapse (kd08 corr 0.990, pool 0.40004 < kd05 0.40085).

### Bet 3. Schedule levers re-screened under KD: P1 0.04, P2 0.005

Mechanism: these levers closed under hard labels because the cooldown memorises the exploration pool. Students do not show that (F15: @last = selected). The lead re-screened capacity levers under KD; the schedule levers were not. Under-fitting (F9) argues for more optimisation in the low-lr phase.
- Zero patch, base kd05w192. Already in lane.sh since 20:25Z: `kwm05`, `kwcf35`, `kwaux0` / `kwaux1`, `kwwd05` / `kwwd10` (kwaux0 and kwwd05 are queued in wave 1). Still missing: cooldown mix 0.10, lr 1.4e-3, kd_w 0.65 at d192 (§6).
- Rule: the lead's 20:21Z "ceiling shift" rule (asymptote delta vs kd05w192 on the same 4 seeds), which is the right level to read at.
- My order inside this group: cooldown mix first (the cooldown is where the test-format adaptation happens, +0.0012 per member at mix 0.02), then lr, then aux and weight decay.

### Bet 4. Full-data refit of the final students: P1 0.03, free

Mechanism: F15 removes the reason to keep 30,000 exploration feeds out of the final models: 18% of the 164K-feed exploration pool, the only unused labels (F1). Worth +0.0001 at uniform feed value, up to +0.0004 if an exploration feed is worth 5 production feeds (docs/01 §3: 50K exploration feeds beat 50K production feeds by +0.0055 in a GBDT).
- Recipe: the chosen student command without the holdout: `--holdout 0 --select_on val --keep last`, and without `--save_holdout` / `--select_on holdout` (`save_holdout` asserts `--holdout 30000`); `--keep last` means no label picks the checkpoint. Needs a 20K-feed smoke run first: the path has never been executed.
- Cannot be judged on the holdout, and the val MDE is far above the effect: adopt only as a pre-declared refit of an already chosen recipe, with the unchanged file as the other pick.
- 16 runs, in the freeze week.

### Bet 5. U20, the GPU test of whether teacher-labelled unlabelled feeds add anything: P1 0.02 (queued by the lead 20:21Z; predicted KILL)

The cheapest experiment that decides the whole "more teacher-labelled inputs" family; it also gives the data slope of a distilled student (arm C − A), which is unmeasured. The lead applied the patch at 20:20Z and queued the six runs (wave 1, rule in E_kd_predeclared.txt 20:21Z block, copied from the draft of this report).
- `teacher_xf0.npy` = SUM of 3 hard-label models that never saw user fold 0 (so `--kd_T 3` gives the mean logit); on fold-0 feeds it is an honest teacher of unlabelled data.
- Arms (d192, kd_w 0.5, seeds 951 / 952): A = `--xfold 0` (80% of users, labelled); B = `--xfold 0 --xfold_unlab` (the same + fold 0 with teacher-only targets, no K9 aux, not in the cooldown pool); C = all users labelled.
- Patch: `nn_x/w12/recheck/long_bets/u20_train.patch` (60 diff lines, one flag, default off); audited by the `kd_audit` lens (no label path).
- Rule: GO iff B − A ≥ +0.0008 hold AND ≥ 0 val AND (B − A) / (C − A) ≥ 0.5. KILL iff B − A < +0.0004 hold.
- **Prediction, written before any U20 number exists:** B − A = 0.0000 ± 0.0004 on the holdout and on val (KILL), C − A = +0.001..+0.002 on the holdout. Basis: F9 (measured after the draft) and the GBDT analogue: a GBDT student on 134K instead of 60K teacher-labelled feeds gains +0.00014 ± 0.00062 val / −0.00046 ± 0.00036 hold, while the hard-label GBDT gains +0.0036 / +0.0041 from the same extra feeds (§4).
- Two read-out caveats from `kd_audit` §7.1 that I agree with: A has 78,710 steps and B, C 98,350, so the ratio is the statistic, not B − A alone; and the 5,974 holdout feeds of fold-0 users must be read separately from the other 24,026 (a gain only on fold-0 users does not support a transfer-set programme).
- If the read is GO against this prediction, F9 is wrong somewhere and the more-inputs programme opens: stage 1 = views that are new inputs by F10's measure (logit corr to the stored view ≤ 0.98: history drop ≥ 0.5 or prefix truncation) with an online teacher; stage 2 = candidate swaps between same-day feeds with rebuilt pair features, 150-300 GPU-h, upside +0.002..+0.004 by F1.

### Bet 6. Soft-target fine-tune: P1 0.03, P2 0.005 (tested on CPU by another lens; no GPU)

Mechanism as I had it: the fine-tune re-injects one-hot noise from 11.5K feeds per fold into a de-noised trunk (F14), while its period adaptation was worth +0.0005 on public for hard-label pools.
- The `ft_redesign` lens ran the cheapest decisive experiment today (CPU, 2 W8 members, 3 of 5 folds, rule written first): soft targets inside the fine-tune give the member +0.00053 ± 0.00038 over the plain fine-tune and the file −0.00036 ± 0.00031; its bound for any redesign that shrinks or re-weights the same shift is +0.0005. Its rule failed, so no 8-member GPU run.
- I have no measurement that contradicts this. What is left is the nested form (teacher = pool fine-tuned on other folds), which changes the target, not only its noise; I price it at P1 0.03 and would not spend GPU on it before the bets above.
- Leak rule for anyone who tries: the teacher of a val row must never have seen that row's label.

### Not bets: running or queued arms this lens re-prices
- kwsub5 / kwsub2: predicted null by F9 + F10; do not extend them to longer schedules. The synthesis note that a kwsub FAIL "does not close sub-feed distillation" (stale context inputs, size-mix shift) is fair about the arm, but F9 closes the idea from the other side: a cleaner sub-feed arm has no matching gap to close.
- kwh30 / kwh10 (queued): predicted null. A 30% history drop changes the logits at corr 0.990 and costs −0.0006 ± 0.0011 at inference (F10); it regularises an over-fit to the teacher that F9 shows does not exist.
- kwT07: positive at the plateau (+0.0007 / +0.0004). Plateau reads over-stated kd08 before (better members, worse pool), and a sharper teacher acts like a higher teacher weight (F12), so read it at pool level with the correlation column. If it passes, T 0.7 belongs inside Bet 2's surface.
- W16: worth about +0.0002 by F4.

## 4. Considered and closed by measurement

| idea | test | result | status |
|---|---|---|---|
| Distilling on more data: sub-feeds, test / val feeds as a transfer set, pseudo-feeds with teacher targets, candidate swaps | F9, F10, F11 | no generalisation gap in matching (0.9933 seen vs 0.9933 unseen); cheap views are not new inputs; no extra gap on test | CLOSED on CPU evidence; GPU confirmation = Bet 5 (queued) |
| Noisy-Student ladder (wider + input noise each generation) | F9, F10, F6 | noise regularises over-fitting to the teacher, which does not exist; students do not follow a stronger teacher | CLOSED as a ladder; the width part lives in Bet 2 |
| Disjoint-teacher students (TOP1_GAP_EXTERNAL lever 4) | F8 | cross-teacher corr = within-teacher corr to 0.001 | CLOSED |
| User-half / user-fold members (EXTERNAL_SCAN N6) | F3 | −0.0011 ± 0.0006 val / −0.0018 ± 0.0004 hold at equal count | CLOSED |
| Cross-validated distillation chains | F1, F6, F7 | each cross-fit costs the teacher 20% of users (−0.0027 per member); the label copy it removes is 0.14 logit | CLOSED. The kx closure was a teacher-strength effect, not an in-sample effect |
| Teacher with privileged item-level click aggregates (TOP1_GAP_EXTERNAL lever 3) | F16 | +0.0001 ± 0.0002 on top of the teacher | CLOSED at the CPU pre-gate |
| Teacher with the feed's other clicks (new today) | F17 + GBDT proxy below | information +0.0099 on hidden positives, but the proxy student loses −0.00109 ± 0.00072 val / −0.00176 ± 0.00043 hold | CLOSED by proxy; the shift is label-dependent noise the student cannot see. Target file kept (`out/teacher_mean11_coclick.npy`; flags `--kd_file <it> --kd_w 0.5 --d 192`) |
| GBDT student as a different-bias family | table below | rho 0.90, gap 0.017; blend ≤ 0 on the holdout at every weight | CLOSED |
| Position-debiasing teacher; pre-trained encoder scale-up under KD; hindsight teacher | record only (position tower neutral; kw_hl2 +0.0002 / −0.0010 under KD; hindsight = label copy, and F7 shows label copy is not what helps) | no new evidence | not re-opened |

GBDT student (LightGBM, same rows and features per column, fixed 600 rounds unless noted; hard-label rows are quoted at their BEST logged round, which favours them):

| target | feeds | val | hold | rho to W8 TO pool (val) | 0.95 W8 + 0.05 GBDT, val / hold |
|---|---|---|---|---|---|
| hard click, lambdarank (best round) | 60K | 0.37643 | 0.29912 | 0.847 | +0.00011 ± 0.00028 / 0.00000 ± 0.00014 |
| hard click, lambdarank (best round) | 134K | 0.37998 | 0.30322 | 0.863 | +0.00017 ± 0.00026 / −0.00011 ± 0.00014 |
| teacher_mean11 logit, L2 | 60K | 0.38469 | 0.30807 | 0.901 | +0.00037 ± 0.00025 / −0.00019 ± 0.00014 |
| teacher_mean11 logit, L2 | 134K | 0.38484 | 0.30761 | 0.902 | +0.00029 ± 0.00026 / −0.00005 ± 0.00015 |
| cross-fitted teacher logit, L2 | 60K | 0.38444 | 0.30768 | 0.900 | +0.00017 ± 0.00025 / −0.00015 ± 0.00014 |
| teacher logit + 1.0 y | 60K | 0.38362 | 0.30818 | 0.901 | +0.00041 ± 0.00026 / −0.00011 ± 0.00014 |
| teacher logit + 1.0 y + co-click shift | 60K | 0.38252 | 0.30642 | 0.888 | +0.00029 ± 0.00024 / +0.00003 ± 0.00015 |

Blend rows for the hard-label models use their round-600 predictions (the saved ones). Readings: soft targets are worth +0.0083 val / +0.0090 hold to a GBDT at 60K feeds and +0.0049 / +0.0044 at 134K; with teacher targets 60K feeds are already enough (134K − 60K: +0.00014 ± 0.00062 / −0.00046 ± 0.00036); in-sample and cross-fitted teachers are the same target for it (+0.00025 ± 0.00061 / +0.00039 ± 0.00036). Reference: cb26 CatBoost YetiRank on the same 134K feeds, 0.38201 val.

## 5. Calendar and GPU budget

| when | what | runs |
|---|---|---|
| queued (wave 1, lead) | U20 × 6, kwaux0 / kwwd05 / kwh30 × 4 seeds | 18 runs |
| Nov 15, next free wave | Bet 2 stage 1 (12 runs) + kd10w192 (2); Bet 3: kwm05, kwm10, kwlr14, kd065w192 × 4 seeds (16) | 30 runs ≈ 35 GPU-h |
| Nov 16-18 | 8-seed confirmation, fine-tune and file for any cell that passes its asymptote rule | ≤ 40 GPU-h |
| by Dec 1 | merge negotiation, exchange, blend, one public read | 0 |
| Dec 2-6 | Bet 4 refit of the chosen students, gate, freeze | 16-32 GPU-h |

This is under 5% of the available GPU time. More seeds of the current recipe buy +0.0002 (F4); the rest of the budget has no use I can defend.

## 6. Lane arms still missing in `nn_x/w5/lane.sh` (flags appended after `E_BASE HYG --seed S`; T11 = `<repo>/nn_x/w12/kd/out/teacher_mean11.npy`)

Already there since 20:25Z: u20a / u20b / u20c, kwh30, kwh10, kwaux0, kwaux1, kwwd05, kwwd10, kwm05, kwcf35, kd08w256, kw256d2.

```
E:kd065w256) echo "--kd_file T11 --kd_w 0.65 --d 256" ;;
E:kd08w320)  echo "--kd_file T11 --kd_w 0.8 --d 320" ;;
E:kd10w192)  echo "--kd_file T11 --kd_w 1.0 --d 192" ;;
E:kd065w192) echo "--kd_file T11 --kd_w 0.65 --d 192" ;;
E:kwm10)     echo "--kd_file T11 --kd_w 0.5 --d 192 --cooldown_mix 0.10" ;;
E:kwlr14)    echo "--kd_file T11 --kd_w 0.5 --d 192 --lr 1.4e-3" ;;
```
Launch form: `bash nn_x/w5/lane.sh <gpu> "E:kd08w256:951 E:kd08w256:952 ..."` (`DRY=1` first). Read-out: `nn_x/w12/recheck/pool_composition/pc_arm_readout.py` (asymptote level), as the lead's 20:21Z rule says.
Not verified by a run: d320 memory at 2-3 runs per GPU; `kd10w192` trains without any one-hot term, so its K9 aux share of the gradient is the largest of all arms (`kd_audit` §7.2 item 4): read it together with kwaux0.

## 7. Limits of this report

- F1 is measured on hard-label members; the data slope of a distilled student is unmeasured (Bet 5 arm C − A).
- F9 compares a 4-student POOL with its teacher; single-student dumps on train rows do not exist. The seed-noise term is the same on both sides, so the comparison stands.
- F10 and F12 use one checkpoint or one seed in places (kd08w192 has 1 seed); the pre-computed xt columns stay full-view in F10 exactly as they do under `--aug_*`.
- The co-click closure rests on a GBDT proxy (one run per arm) plus the variance argument; no NN student was trained on that target.
- GBDT arms are single runs with fixed rounds; by the W25 lesson differences below 0.003 between GBDT arms are noise, and none of the conclusions above needs a smaller one except the 134K − 60K slope, which is quoted with its paired SE.
- Pool-level effects of kwT07, GRU students, kdmix and d224 were not final when this was written.
- Compute note: one 1-minute probe (`p4_partner.py`) ran with uncapped numba threads next to the 10-thread GBDT job, so the 12-thread cap was exceeded for about a minute; later probes set `NUMBA_NUM_THREADS=2`. No GPU was used and no process of mine is left running.
