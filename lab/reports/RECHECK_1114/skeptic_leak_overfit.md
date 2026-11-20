# RECHECK 11-14, skeptic review "leak_overfit": leaks, overfit and private-leaderboard risk

Written 2026-11-14 ~21:40 UTC. CPU only (12 threads), read-only on predictions, logs and code; no GPU, no running job touched, no file under `src/`, `nn_x/w5/`, `preds/` or `subs/` written.
Scripts and logs: `nn_x/w12/recheck/leak_overfit/` (`r1_pools.py` .. `r9_w8p20_test.py`, `verdicts.py`, `out/*.log`, `out/verdicts.json`).
All deltas are paired per feed; "a / b" = val (14,423 feeds) / in-period holdout (30,000 feeds, M >= 50, train-only level); ± = 1 SE.
"Measured" = computed here. "Inferred" is marked.

## 0. Verdict in six lines

1. **No leak found.** No finding in the list uses val/test labels, a user's later snapshots or own-feed labels as a model input. The KD path, the U20 patch and the fine-tune folds are clean by code read; every headline pool number reproduces from the files.
2. **The realised W8 reads above its own recipe on both witnesses, and rules that compare with it carry a hidden bar.** All 16 wide students finished at 21:20Z: W16 = 0.40144 / 0.32460 (-0.00013 ± 0.00031 / +0.00004 ± 0.00019 vs W8); the independent second 8-pool of the same recipe reads -0.00034 ± 0.00047 / -0.00042 ± 0.00028 vs W8; across 8-subsets the val excess and the holdout excess correlate at -0.19. An identical-recipe 8-pool would fail the declared confirmation rule against W8 by 0.0008 on the holdout. Four reports forecast W16 at +0.0001..+0.0003 over W8 on val; it reads below.
3. **The pop1-view probe (W8P20) is the one place where public is about to be used as the judge.** Its val gain is view-specific but weak (+0.00049 ± 0.00031 at the final level, one val day carries it), the holdout is against it on every day group (-0.00063 ± 0.00022), "confirmed three times on public" is one public draw, and the paired public SE is 0.00041, not 0.0002-0.0003: the declared ±0.0003 thresholds are ±0.7 SE.
4. **Sub-feed arms and the proposed short-feed holdout sample the wrong format.** Keeping every positive gives 1.88 positives per short feed and 39% multi-positive feeds; real val M<50 has 1.22 and 15%; a uniform subset conditioned on at least one positive gives 1.19 and 14%.
5. **"Public Δ ≈ 0.4 val Δ + 0.4 holdout Δ" is not established.** The val slope is 0.42 ± 0.18 with W8 and S16 and 0.21 ± 0.21 without them; the holdout coefficient moves from 0.00 to 1.66 with the pop1 files. The two-witness acceptance rule does not need these numbers and stands.
6. Of 106 findings: 66 keep, 31 modify, 9 drop. Nothing with an own-model P(+0.001 public) above 0.06 survives; the levers that remain are decision hygiene (section 5) and the owner's merge decision.

## 1. What I re-ran

| # | claim (finding ids) | my result | agrees |
|---|---|---|---|
| R1 | headline pools; selection vs @last (kd_audit#1, #3; kd_reprice#1) | base x4 0.39887 / 0.32328; kd05 x4 0.40085 / 0.32349; kd05w192 x4 0.40134 / 0.32408; W8 0.40157 / 0.32456; files W8 0.40165, S16 0.40173, C3 0.40101, G3 0.40032. Selected minus @last, 4-pool holdout: base +0.00041 ± 0.00019, kd05 +0.00007, kd05w192 +0.00005, W8 +0.00009 ± 0.00011. KD vs base @last: +0.00055 ± 0.00036, +0.00115 ± 0.00037, +0.00133 ± 0.00038 | yes, to the last digit |
| R1b | inherited selection (new control) | Teacher tf11 pool on the holdout: 0.32371 with selected members, 0.32373 with the 4 base members @last (-0.00002 ± 0.00011). Selection of teacher members on the holdout does not reach the students. W8@last - G3 = +0.00068 ± 0.00034 (selected: +0.00076 ± 0.00033) | new |
| R2 | file minus TO pool; FT transplant (ft_redesign#1, #3; winner_model#9) | file - TO: W8 +0.00009 ± 0.00043, S16 +0.00001 ± 0.00041, C3 -0.00002 ± 0.00039, G3 -0.00003 ± 0.00039. FT(W8) + TO(S16) = 0.40172 vs 0.40173 (-0.00001 ± 0.00023), test corr 0.99972 | yes |
| R3 | pop1 admixture (kd_reprice#5, pool_composition#10, W8P20) | section 2.2 | numbers yes, reading weaker |
| R4 | transfer factor (lb_stats#2) | section 2.4 | partly |
| R5 | W8 vs S16 public SE (lb_stats#3) | per-feed sd 0.0404 -> SE public 0.00044, private 0.00029 | yes |
| R6 | pool-size law and W16 forecast (kd_audit#9, kd_reprice#3, lb_stats#4/#6, pool_composition#1, long_bets#4) | section 2.1 (the finished W16) | val: only kd_reprice#3; holdout: all but long_bets#4 |
| R7 | positives per feed by size (lb_stats#5, pool_composition#12, kd_audit#10, kwsub) | section 2.3 | new |
| R8 | U20 witness composition (kd_audit#2) | holdout feeds by user fold 5974 / 6003 / 6087 / 5931 / 6005; val: 4,386 feeds with a train user, 852 in fold 0 | yes, exact |
| R9 | code read: KD target, `--xfold`, `--xfold_unlab`, holdout exclusion, `augment_pool_batch`, fine-tune folds | section 3 | yes |

Not re-run: the 246.7M-row teacher alignment, the CPU fine-tune replicas, GBDT students, the residual-gate studies, the Monte Carlo of picks.

## 2. Measurements that change a reading

### 2.1 The realised W8 reads above its own recipe; "beat W8" rules carry a hidden bar (measured; `out/r6_w16.log`, `r6b_luck.log`, `r6c_w16.log`)

All 8 extension seeds (kd05w192 and kd05t25w192, s955-958) had train-only and holdout predictions at 21:20Z, so the forecasts can be scored and the recipe has a second, independent 8-pool.

| pool (train-only) | val | holdout |
|---|---|---|
| W8 (seeds 951-954 of both wide arms) | 0.40157 | 0.32456 |
| NEW8 (seeds 955-958, same two arms) | 0.40122 (-0.00034 ± 0.00047 vs W8) | 0.32413 (**-0.00042 ± 0.00028**) |
| **W16** (all 16) | **0.40144** (-0.00013 ± 0.00031) | **0.32460** (+0.00004 ± 0.00019) |
| W16 @last | 0.40157 | 0.32449 |
| random 8-subsets of the 16 (198 distinct), mean (sd) | 0.40139 (0.00025) | 0.32437 (0.00014) |
| size curve over subsets of the 16, N = 1 / 2 / 4 / 8 / 12 | 0.39938 / 0.40042 / 0.40100 / 0.40138 / 0.40152 | 0.32251 / 0.32354 / 0.32407 / 0.32432 / 0.32450 |
| member mean: 8 new seeds vs original 8 | 0.39929 vs 0.39935 | 0.32247 vs 0.32263 |

| W16 forecast | val | holdout | scored (val / holdout) |
|---|---|---|---|
| kd_reprice#3 | 0.4014 ± 0.0002 | 0.3245 ± 0.0001 | hit / upper edge |
| pool_composition#1 | 0.40165 | 0.32459 | 0.0002 high / hit |
| lb_stats#4, #6 | 0.40161-0.40175 | 0.32456-0.32470 | band missed by 0.0002 / hit |
| kd_audit#9 | 0.4016-0.4019 | 0.3245-0.3247 | band missed by 0.0002 / hit |
| long_bets#4 | W8 + 0.0002 | W8 + 0.0002 | 0.0003 high / 0.0002 high |

- The size law holds for the recipe (Q about 0.4016-0.4017 / 0.3246 from the 16). What failed is anchoring the val forecast on the realised W8, which lies above 76% of the 8-subsets on val and 90% on the holdout (+0.00018 and +0.00019 over the subset mean) and 0.0003-0.0004 above the independent NEW8 on both witnesses.
- Is that excess a property of the members? Correlation across the 198 subsets between val excess and holdout excess: **-0.19**; between two disjoint user halves of val: -0.01; between two halves of the holdout: +0.05. Member level, 58 members of 12 arms, within-arm deviations: corr(val, holdout) **-0.09** (SE 0.15). No sign of a transferable seed quality. Against that, NEW8 is lower than W8 on both witnesses at once (0.7 and 1.5 SE; joint chance a few percent). I cannot separate "favourable first batch" from "slightly weaker second batch"; both readings lead to the same two actions.
- Action 1, the reference: an identical-recipe fresh 8-pool FAILS the 20:21Z confirmation rule "8-seed pool vs W8: holdout >= +0.0004 and val >= 0" by 0.0008 on the holdout. Any rule read against the realised W8 has a hidden bar of 0.0002 (vs the subset mean) to 0.0004 (vs a fresh 8-pool) on each witness. Read new recipes against matched-N subsets of the 16 (and "[W8 + 4 X]" tests against W8 + 4 of the new wide seeds); this applies to the GRU rule of 17:32Z and to the second clause of long_bets#5.
- Action 2, the finals: W16 and W8 together. If the excess is noise, W16 is the better file by the size law (+0.0001); if the first batch is truly better, W8 is. "Accept W16 iff val Δ >= 0 and holdout Δ >= 0 vs W8" (lb_stats#4) rejects it on val; the lead's 20:16Z block (C), "built regardless of its reading", is the correct form.
- W8's quoted val over-states its recipe by 0.0001-0.0003; pick Monte Carlos that take it at face value over-state P(W8 best) against W16.

### 2.2 The pop1-view admixture (measured; `out/r3_pop1.log`, `r3_pop1_e_4of4.log`, `r9_w8p20_test.log`)

Train-only level, (1 - w) z(W8) + w z(family pool) minus W8, val / holdout (z in brackets):

| family mixed in | w 0.10 | w 0.15 | w 0.20 | w 0.30 |
|---|---|---|---|---|
| pop1kd x4 (pop1 view, students) | +0.00043 (1.8) / -0.00027 (-1.8) | +0.00053 (1.8) / -0.00043 (-2.2) | +0.00055 (1.6) / -0.00063 (-2.8) | +0.00021 / -0.00104 (-3.7) |
| pop1 x8 (pop1 view, hard labels) | +0.00028 / -0.00032 | +0.00028 / -0.00051 | +0.00027 / -0.00064 | -0.00016 / -0.00117 |
| controls, same view: base x4, kx05 x4, kd05 x4, gen-2 x4, GRU F10 x6, F7x x7, kd10 x4 | -0.00022..+0.00014 / -0.00017..0.00000 | -0.00011..+0.00018 / -0.00023..+0.00003 | -0.00032..+0.00013 / -0.00037..-0.00005 | -0.00056..+0.00022 / -0.00041..-0.00006 |

- The val gain is specific to the view (controls give 0), so it is not the "anything helps at 0.15" artefact. It is still weak: best cell 1.8 SE after a weight scan; nested weight choice on user halves (40 splits) picks 0.21 ± 0.10 and returns +0.00021 (sd over splits 0.00050).
- Final level at the declared W8P20 weight 0.2, both branches: +0.00029 ± 0.00031 with 3 fine-tuned members (20:50Z), **+0.00049 ± 0.00031** with all 4 (21:00Z); the best weight moved from 0.10-0.15 to 0.20 between the two reads. Slices at 0.2: M<50 +0.00003 ± 0.00058, M>=50 +0.00080 ± 0.00035; val Tuesday +0.00100 ± 0.00045, Thursday -0.00008 ± 0.00043.
- Holdout at 0.2 by day group: weekend (public-like) -0.00094 ± 0.00039, Monday + Tuesday -0.00063 ± 0.00044, all weekdays -0.00048 ± 0.00028, last 20% of days -0.00041 ± 0.00039. No day group is positive. (For the hard-label analogue C3 - G3 the weekday pattern of lb_stats#11 reproduces: weekend -0.00023 ± 0.00051, Monday + Tuesday -0.00189 ± 0.00058; val Tuesday / Thursday +0.00075 / +0.00062.)
- Public evidence for the view: Q2a, C1, C3 are nested builds on one public draw; the family contrast is +0.0006 at a paired SE of 0.0005 (1.2 SE), not three confirmations.
- Paired public SE of W8P20 - W8 from the val per-feed sd (0.0376): **0.00041** (test corr to W8 0.99942, same on public and private days). The 20:16Z rule assumes 0.0002-0.0003. At ±0.0003 the rule adopts a zero-effect file with probability 0.23 and closes a +0.0003 file with probability 0.07.
- Safe form: one slot for the family, fixed weight, thresholds in SE (>= +1.5 SE, +0.0006, to invest in wide pop1 students; >= +2 SE, +0.0008, to replace pick 1; anything else: pick 2 at most). The wide pop1 arm (B) should not be fine-tuned before that reading.

### 2.3 Short feeds are uniform subsets with positives thinned, not positives-kept subsets (measured; `out/r7_npos.log`)

| feeds | mean positives | P(>= 2 positives) | P(>= 3) |
|---|---|---|---|
| real val, M < 50 (5,893 feeds) | 1.220 | 0.148 | 0.039 |
| holdout feeds M >= 100 sub-sampled to the val M<50 sizes, **positives kept** (the `--aug_cand` rule, cd_arms.py l.76; lb_stats#5) | 1.877 | 0.391 | 0.189 |
| same, **uniform subset**, feeds without a positive dropped (21% survive) | 1.189 | 0.140 | 0.031 |

Positives per feed by size agree between val and the holdout wherever both exist (M 50-75: 1.40 vs 1.49; 100-150: 1.64 vs 1.69; 200+: 2.14 vs 2.10).
- kwsub5 / kwsub2 therefore train on sub-feeds whose composition depends on the labels (not an input leak; a format mismatch), in addition to the stale ctx inputs and the size-mix shift. kd_audit#10, kd_literature#2 and long_bets#1 are right that the teacher TARGET is consistent; a kwsub null still closes only this implementation.
- Any short-feed witness (lb_stats#5, pool_composition#12) and any repaired sub-feed arm (kd_reprice#6, winner_model#2) must use the uniform rule. With KD a sub-feed without a positive can keep a teacher-only target (the U20 mechanism), so nothing is wasted.

### 2.4 Transfer of offline deltas to public (re-fit of the lb_stats table; `out/r4_misc.log`)

Weighted slope through the origin of public Δ on val Δ (deltas vs G3, weights 1 / SE²; files share one public draw, so the SEs are optimistic):

| files | k val |
|---|---|
| modern 13 | 0.42 ± 0.18 (leave-one-out 0.33..0.52) |
| 11-12 files (6) | 0.09 ± 0.25 |
| 11-13/14 files (7) | 0.83 ± 0.27 |
| the same without W8, S16 | 0.56 ± 0.41 |
| all without W8, S16 (11) | 0.21 ± 0.21 |

Holdout slope: 0.23 ± 0.32 (12 files); 0.00 ± 0.36 on the 11-13/14 files; +1.66 ± 0.53 without the four popularity-view files. Joint fit: val 0.51 ± 0.19, holdout 0.09 ± 0.33 (lb_stats with the full covariance: 0.39 ± 0.23 and 0.41 ± 0.31).
- The overall "0.6" rests on W8 and S16, two 0.9989-copies on the same public feeds. The lineage split was made after seeing the data (stated in lb_stats §7). The holdout coefficient is not identified; it depends on which family is in the sample.
- What survives: val deltas transfer partially; a val-only or holdout-only gain is unproven; both >= 0 with combined z >= 2 is a sound rule. Quote no coefficient.

## 3. Leak audit of the list

| item | status |
|---|---|
| KD target (train.py l.498-502) | label enters only as (1 - w) y / sum(y); teacher logits are indexed by train row; target sums to 1. Clean. |
| holdout | the 30,000 feeds are removed from `feeds` before the fold drop, the pool and the KD lookup; teacher files hold out-of-sample values on those rows and they are never read in training. Holdout feeds share users with train (in-period witness, all users seen); 30% of val and test feeds have a train user. |
| selection on the holdout | direct for hard-label members (+0.0004 at 4-pool level), about 0 for students, <= 0.0001 through the teacher (R1b). Read KD vs hard-label @last. |
| `--xfold_unlab` (U20) | unlabelled feeds: teacher weight 1, K9 aux weight 0, removed from the cooldown pool. No label path. Read-out by user fold (kd_audit#2). |
| `--aug_cand` | no label value enters an input, but which candidates stay depends on the labels (section 2.3). |
| fine-tune | GroupKFold by user, user-0 feeds as singleton groups; test = fold mean. 0.33% of test feeds share a user with val. |
| soft-target fine-tune | clean with a train-only teacher. The nested form (teacher = other-fold fine-tuned pool, long_bets#11) is clean only if the teacher excludes the row's fold AND the student's evaluation fold. |
| co-click targets (long_bets#9) | the feed's other clicks enter the target of a train row: allowed (train-only), measured harmful. Do not run the NN screen. |
| earlier-labelled-feed bank (forensics_skeptic#3, #9) | null. For test rows such a bank may use train labels only. |
| transductive items (U20-GO programme, kd_literature#12, ft_redesign#10) | test features without labels are computable for private rows; the effect on test rows cannot be validated offline. |
| public as a selector | used once so far (W8, a post-hoc file, best of 3 scored that day). W8P20 and the TO-only probe would be the second and third uses. |

## 4. Contradictions between the nine reports

| question | positions | what the data say |
|---|---|---|
| W16 vs W8 | above W8 by +0.0001..+0.0003: kd_audit#9, lb_stats#4/#6, pool_composition#1, long_bets#4. Not above: kd_reprice#3 | finished W16: -0.00013 ± 0.00031 val, +0.00004 ± 0.00019 holdout; kd_reprice#3 is right on val, all but long_bets#4 on the holdout; the size law holds for the recipe mean |
| Is a kwsub null a closure? | yes: kd_audit#10, kd_literature#2, long_bets#1. No: kd_reprice#2, winner_model#1 | target valid, inputs and sampling out of format (2.3): closes the implementation only; the seen = unseen matching result keeps the prior of a clean form low |
| Which witness decides? | holdout: kd_audit#11, pool_composition#7. Holdout worst: lb_stats#2. W8 > S16 on public is noise: lb_stats#3 | no single witness; two-witness rule |
| Soft-target fine-tune | proposed at P 0.05-0.07: kd_reprice#7, kd_literature#4, forensics_skeptic#1. Measured null: ft_redesign#2, kd_audit#5. Demoted: long_bets#11 | closed in the file |
| TO-only public probe | submit: ft_redesign#8, forensics_skeptic#10. No decision value: lb_stats#9 | lb_stats#9; information only |
| Pick 2 | S16: lb_stats#1. C3: pool_composition#11 | not resolvable; W16 when built, W8 as the scored second |
| Residual-gate bar | +0.0005 at 2 SE: ft_redesign#12. +0.0013: forensics_skeptic#6 | +0.0013 (measured detection floor, planted control) |
| Public SE of the pop1 probe | 0.0002-0.0003 (20:16Z block), 0.0005 (pool_composition#10) | 0.00041 at w 0.2 |
| "Anything helps at 0.15 on val" (kd_reprice §3) | read as val noise of the reference | against W8 the same-view controls give 0 (-0.0001..+0.0002): the effect is specific to the 11-wide reference used there, not a property of val |

## 5. Future ideas from this lens (none of the nine reports proposes them)

They protect or sharpen decisions; none is a score lever by itself. Cost is CPU unless stated.

| # | idea | why now (evidence above) | cost | what would falsify or stop it |
|---|---|---|---|---|
| A | **One frozen reference pool for every pool claim**: the 16 wide students, deltas reported at matched N (mean over random N-subsets) instead of against the realised W8 | 2.1: the realised W8 handicaps every new arm by 0.0002-0.0004 on each witness; an identical recipe fails the 8-seed confirmation rule | 1 script, minutes | a positive cross-witness correlation of subset excess (>= +0.3) in a later, larger member set |
| B | **Final pair by construction**: W16 and W8 (the FT branch of W16 may be W8's, ft_redesign#3). Every final must have one public reading (validity) | 2.1: the pair is right under both readings of the W8 excess; no selection involved; W8, already scored, doubles as an operational hedge against a build error | 0 GPU beyond W16; 1 slot | W16 public more than 2 paired SE (about 0.0007) below W8 |
| C | **SE-denominated public protocol**: before a probe, compute the paired public SE from the val per-feed deltas; thresholds in SE (2 SE to replace pick 1, 1.5 SE to spend GPU on the family, else no action); one public read per family | 2.2: a ±0.0003 rule turned out to be ±0.7 SE | 1 minute per probe | - |
| D | **Uniform sub-feed sampler as a shared utility** (label-independent subset, sizes from the val M<50 distribution, ctx inputs recomputed) for a short-feed witness and for any sub-feed arm | 2.3; 39% of test is M<50, the holdout has none, and the KD gain over C3 on val M<50 is +0.0014 ± 0.0010 | 0.3 GPU-h or a few CPU-h of inference for 16 checkpoints | premise check on two more label-free descriptors (category mix, cold share) fails; or the witness merely repeats the full-holdout delta (corr of per-feed deltas > 0.9) |
| E | **Lagged-popularity holdout as a mechanism test for the pop1 dispute** (pre-declared, symmetric): rebuild the five LUO popularity columns of the 30K holdout feeds with the 7-10 day lag that test rows have (`feat_new/it_hist_hz.py` has the lag logic; the existing `_holdout_hz` files use lag 0 and do not do this), re-score the W8 and pop1kd members, read W8P20 - W8 there | the holdout veto on pop1 is overridden on the grounds of an in-period popularity bias that has never been measured for students; public can give one reading at SE 0.0004, this gives SE 0.0002 | CPU rebuild of 5 columns for 4.6M rows + inference of 12 checkpoints | rule fixed before the run: penalty shrinks to >= -0.0002 -> the bias explanation holds; stays <= -0.0005 -> close the view for student pools without a slot. Stop before inference if the label-free parity check (age x coverage distribution of the lagged columns vs test) fails. My prior that it is decisive: 0.3. It must stay a mechanism test: stale judges were refuted when used to re-judge arms trained on the stale view (COVERAGE_MAP rows 160, 220, 252) |
| F | **Private-day drift column in the pre-submission gate**: per test day, candidate vs pick 1: mean per-feed corr, cold-minus-warm offset, top-1 agreement | ft_redesign#6: the FT cold offset doubles from val Tuesday to the last private day; private is 8-10 days after train, val 3-5 | minutes | - |
| G | **Merge protocol** (if the owner merges): exchange val predictions as well, verify the blend on val (paired SE) and by test-day correlation before the public slot; blend weight fixed by the law, not scanned | long_bets#3 relies on one public reading at SE 0.001-0.0015 | 0 | - |
| H | **Rule template**: every new rule names its reference pool (same seeds or matched N), its statistic and an SE-denominated bar; post-hoc files are flagged and get one slot | today: 6 dated addenda, about 25 arms read on 2 witnesses with 3-4 statistics, one post-hoc promotion | 0 | - |

Where own-model score can still come from, with my priors for >= +0.001 public: never-varied knobs under KD (kwaux0 / kwwd05, queued) 0.05; pop1 view at low weight 0.03; (teacher weight x width) cells 0.03; schedule batch 0.03; phase-split re-cool 0.03; everything else <= 0.02. Their sum is about +0.0003..+0.0005 expected, consistent with lb_stats and long_bets. The merge is the only item above 0.3.

## 6. Verdict per finding

P = adjusted probability in the finding's own sense (levers and recipes: >= +0.001 public at pool level; risks: that the risk is real; nulls: that the idea still pays).

| id | verdict | P | reason |
|---|---|---|---|
| kd_audit#1 | keep | 0.93 | Re-ran every headline pool from files (base 0.39887/0.32328, kd05 0.40085/0.32349, kd05w192 0.40134/0.32408, W8 TO 0.40157/0.32456, file 0.40165, C3 0.40101): exact. Code read of the KD target, --xfold(_unlab), holdout exclusion, aug and fine-tune folds: no label path. New control: holdout selection inherited through the teacher is <= 0.0001 (11-pool with base members @last: -0.00002 +- 0.00011). Not re-run by me: the 246.7M-row alignment, presub_gate. |
| kd_audit#2 | keep | 0.6 | Counts re-derived exactly (holdout by user fold 5974/6003/6087/5931/6005; val 4,386 seen users, 852 in fold 0). The fold split is the control that keeps a same-user effect from being read as a transfer-set effect; apply GO/KILL to the cross-user columns only. |
| kd_audit#3 | keep | 0.9 | Re-derived: selected minus @last 4-pool holdout base +0.00041 +- 0.00019, kd05 +0.00007, kd05w192 +0.00005, W8 +0.00009; KD vs base @last +0.00055 / +0.00115 / +0.00133. Selection-free read also for the freeze rule: W8@last - G3 = +0.00068 +- 0.00034 (z 2.0; G3 itself is selected, so this is a lower bound). |
| kd_audit#4 | keep | 0.3 | Measured gradient shares, inferred consequence; no decision rests on it and the queued kwaux0 arm decides. Do not re-open kd08/kd10 on this argument alone. |
| kd_audit#5 | keep | 0.02 | Null; file-level part re-derived (W8 file - TO pool +0.00009 +- 0.00043; FT - TO -0.00069 +- 0.00061). The CPU replica was not re-run; it agrees with the independent ft_redesign pilot. There is no holdout witness for any fine-tune variant, so a positive would have been val-only. |
| kd_audit#6 | keep | 0.03 | Code confirmed (train.py l.500: softmax(teacher/kd_T), student at T=1). Correct scope note for the kwT read-out; the two-sided arm itself has a low prior (teacher calibrated, flat). |
| kd_audit#7 | modify | 0.03 | Same experiment as kd_literature#1; run once, only the 4-re-cool falsification. Safe as declared (controls from the same snapshots, @last, pool holdout >= 2 SE, val >= 0). Every KD arm ends at the same member ceiling, so expect a null. |
| kd_audit#8 | keep | 0.02 | Null supported by three lenses with independent scripts; consistent with my own control (admixing students of another teacher into W8 moves val/holdout by <= 0.0002). |
| kd_audit#9 | modify | 0.02 | The stop rule stands, the forecast does not: fits anchored on the realised W8 predict W16 at +0.0001..+0.0003 over W8 on val; the finished W16 (21:20Z) reads 0.40144 / 0.32460 = -0.00013 +- 0.00031 val / +0.00004 +- 0.00019 holdout vs W8, and subset excess does not transfer between witnesses (corr -0.19 over 8-subsets, -0.09 over 58 members). W16 is still the better file in expectation (size law +0.0001). |
| kd_audit#10 | modify | 0.02 | The TARGET consistency is real (three lenses, KL 0.002-0.005), but "kwsub is a fair test" is too strong: the INPUT side is out of format (6 stale ctx columns, size-mix shift) and the sampling rule is label-dependent: positives always kept gives 1.88 positives per short feed and 39% multi-positive feeds against 1.22 and 15% in real val M<50 (uniform subsets conditioned on >= 1 positive give 1.19 and 14%). A kwsub null closes this implementation only. |
| kd_audit#11 | modify | 0.4 | Val SE = the bar: agreed. "Holdout decides" is not supported: its agreement with public on W8 > S16 is a 0.95-SE public gap (lb_stats#3), and the holdout mis-read the pop1 files. It is blind to M<50 (39% of test) and to the FT branch. Use the two-witness rule (both >= 0, combined z >= 2) with val M<50 >= -1 SE; and compare against a same-seed or matched-N reference, not the realised W8 (an identical-recipe second 8-pool reads -0.00042 +- 0.00028 on the holdout against it). |
| kd_audit#12 | keep | 0.9 | Reading rule (CPU replay vs CPU replay); pool-level null by winner_model#7. No decision impact. |
| kd_reprice#1 | keep | 0.85 | Width fails the member screen: verified (kd05w192 vs kd05 pool numbers reproduce). The extra Qhat branch is a third look on the same 2 seeds; it is safe only because it triggers MORE DATA (4 seeds), not a promotion. Its threshold (+0.0004 holdout) is about 0.6 SE, so expect one false extension per 3-4 arms. |
| kd_reprice#2 | keep | 0.7 | Stale-input confound is measured. Add a seventh confound measured here: --aug_cand keeps every positive (cd_arms.py l.76), so sub-feeds carry 2.6x the multi-positive share of real short val feeds. |
| kd_reprice#3 | keep | 0.02 | The only pool-size forecast that the finished seeds support on val: W16 = 0.40144 / 0.32460 (forecast 0.4014 +- 0.0002 / 0.3245 +- 0.0001; W8 0.40157 / 0.32456); random 8-subsets of the 16: 0.40139 (sd 0.00025) / 0.32437 (sd 0.00014), W8 at the 76th / 90th percentile. The independent second 8-pool reads -0.00034 +- 0.00047 / -0.00042 +- 0.00028 vs W8; cross-witness correlation of subset excess -0.19 (member level -0.09 over 58 members), so no sign that the excess of the realised W8 is a transferable property. W16 needs one public reading as a final (validity), not as a level test. |
| kd_reprice#4 | keep | 0.03 | Fixed-N mixing gain is the right statistic (no selection, seed noise cancels). Control to add for the running GRU rule: read [W8 + 4 GRU] against [W8 + 4 new wide seeds] (matched N), not against W8, which sits 0.0002 above the mean 8-subset of its recipe and 0.0004 above the independent second 8-pool on both witnesses. |
| kd_reprice#5 | modify | 0.03 | Re-derived with controls. Val gain is specific to the pop1 view (same-view admixtures give 0.0000 +- 0.0002) but weak: TO +0.00055 +- 0.00035 at w 0.2 after a weight scan, nested weight choice +0.00021 (sd 0.0005), all of it on one val day (+0.0010 +- 0.0005 vs -0.0001 +- 0.0004); final level moved from +0.00029 to +0.00049 +- 0.00031 when the fourth fine-tuned member arrived. Holdout -0.00063 +- 0.00022, negative on weekends too (-0.00094 +- 0.00039). "Confirmed on public three times" is ONE public draw of nested files (+0.0006, 1.2 SE). The paired public SE of the probe is 0.00041, so the declared +-0.0003 thresholds are +-0.7 SE (23% false adoption at zero effect). Safe form: one slot, fixed weight, adoption for further work only at >= +1.5 SE (+0.0006), never pick 1 below +2 SE (+0.0008); at most pick 2. |
| kd_reprice#6 | modify | 0.03 | If built: sample sub-feeds UNIFORMLY (label-independent), keep the hard-label half only where a positive survives (teacher-only target otherwise, U20 mechanism), recompute the six ctx inputs, keep the size mix near test. Gate on the kwsub by-M read and on U20 B - A; the seen = unseen matching result and the XPOOL null keep the prior low. |
| kd_reprice#7 | drop | 0.01 | Measured null in the file by two independent CPU pilots today (ft_redesign#2 member +0.00053, file -0.00036 +- 0.00031; kd_audit#5 file +0.00002 +- 0.00044) and bounded at +0.0005 (ft_redesign#1). No holdout witness exists for the stage, so a pass would be val-only on a 1-SE bar. |
| kd_reprice#8 | drop | 0.01 | Re-judging a closed arm with substitute judges (stale-val, holdout_hz, public) because the standard judge said no: the pattern COVERAGE_MAP rows 160 and 252 refuted (outcome-chosen judge, circular stale lens). Its pool-level val +0.00079 +- 0.00051 is the best of 55 re-read arms (1.5 SE). |
| kd_reprice#9 | drop | 0.01 | Superseded: kwh30 (history drop 0.3, 4 seeds, asymptote rule) is queued and answers the question; a 0.1 arm adds a second look at the same lever. Read kwh30 first. |
| kd_reprice#10 | modify | 0.02 | kwaux0 is queued with the asymptote rule. Run the aux 1.0 arm only if kwaux0 shows the head is still load-bearing; otherwise two more looks at a flat dose. |
| kd_reprice#11 | modify | 0.04 | Six arms on a 0.6-SE extension trigger give about one false extension. Acceptable if the confirmation is fixed now: 4-seed pool vs the same-seed kd05w192 pool, holdout >= 2 SE @last and val >= 0, Holm over the batch; never a comparison with the realised W8. |
| kd_reprice#12 | modify | 0.02 | Leak-clean (OOF offset in training, honest fold models on val/holdout/test), but train and inference offsets differ in kind: mean of 3 models that never saw the user vs mean of 15. Score with 3-model offsets (one fold group at a time, averaged over the 5 groups) for parity, @last only, 2-epoch falsification first. |
| kd_literature#1 | modify | 0.03 | Run once (with kd_audit#7), the 4-run falsification first; Holm over the off-diagonal cells is right. A cell that passes must be confirmed on 4 fresh seeds before any d192 replication (9 cells x 2 witnesses is a wide garden). |
| kd_literature#2 | modify | 0.03 | Target validity confirmed, but the conclusion "a kwsub null closes function matching" conflicts with measured input and sampling confounds (kd_reprice#2, winner_model#1, positive density here). Drop the plateau-slope clause and the 15-epoch follow-up: a window statistic chosen after seeing the curves is one more tuned degree of freedom. |
| kd_literature#3 | modify | 0.03 | Read at 2-pool / Qhat level (the member bar fails width itself). The cold-positive concentration of the wide-student gain is one slice among several looked at (z about 2-3 before multiplicity); it is a motive, not evidence. |
| kd_literature#4 | drop | 0.01 | Same lever as kd_reprice#7 and forensics_skeptic#1; tested null in the file on CPU today. The leak note in it is correct and worth keeping: a teacher made of other-fold fine-tuned models leaks the held-out labels. |
| kd_literature#5 | keep | 0.02 | Null at a honest gate (holdout, teacher out of sample); confirmed independently by long_bets#9. |
| kd_literature#6 | keep | 0.02 | Null, three lenses agree. |
| kd_literature#7 | keep | 0.02 | Null by measured calibration; diagnostics use labels only for calibration read-outs, not for targets. |
| kd_literature#8 | keep | 0.85 | Correct reading rule (pool + member correlation). Measured on closed arms. |
| kd_literature#9 | drop | 0.01 | A cost lever for pool growth that is closed (pool at its asymptote). Its non-inferiority margins (-0.0005 val, -0.0003 holdout) are larger than everything W16 is expected to add, so a pass would prove nothing. |
| kd_literature#10 | keep | 0.01 | Null; same data as long_bets#2 with the same sign and size. |
| kd_literature#11 | keep | 0.02 | Null. The cold/warm slice is fine as a diagnostic column, not as a selection criterion. |
| kd_literature#12 | modify | 0.01 | Only behind a U20 GO on the cross-user columns. Test features without labels are computable for private rows (no leak), but the effect on the test rows themselves cannot be validated offline (they are in-sample for the student), so the only judge would be public. |
| lb_stats#1 | keep | 0.9 | Pick 1 = W8 today: three witnesses agree vs G3 (re-derived val +0.00133 +- 0.00058, holdout +0.00076 / @last +0.00068, public +0.00167). Caveat for the Monte Carlo: W8 reads 0.0002-0.0004 above its own recipe on each witness (second 8-pool, 8-subsets of the 16) with no sign that this transfers, so P(W8 best) is over-stated against W16; the robust final pair is W16 + W8. The protocol rewrite is needed before 12-07. |
| lb_stats#2 | modify | 0.5 | Re-fitted from the per-file table: val slope 0.42 +- 0.18 on all 13, 0.21 +- 0.21 without W8 and S16 (two 0.9989-copies on one public draw carry the "0.6"); the lineage split is post hoc; the holdout coefficient is not identified (0.09 +- 0.33 in my fit vs their 0.41 +- 0.31; +1.66 without the pop1 files, 0.00 with them). Keep "val transfers partially"; do not quote "0.4 val + 0.4 holdout" as a predictor. The two-witness acceptance rule does not depend on these numbers. |
| lb_stats#3 | keep | 0.02 | Re-derived: per-feed sd of the paired val delta 0.0404, SE of the public delta 0.00044; the observed +0.00041 is under 1 SE. |
| lb_stats#4 | modify | 0.8 | Right that public cannot confirm W16; wrong acceptance rule. "val >= 0 and holdout >= 0 vs W8" rejects the finished W16 (0.40144 / 0.32460: -0.00013 +- 0.00031 / +0.00004 +- 0.00019), and the val forecast band 0.40161-0.40175, fitted on the realised W8, is missed by 0.0002 (the holdout band is hit). Accept W16 on member-mean parity (new seeds 0.39929 / 0.32247 vs 0.39935 / 0.32263) plus the size law; public = validity only. |
| lb_stats#5 | modify | 0.02 | Evaluation instrument, not a model input, so labels may shape it; but the proposed rule (positives kept) is out of format: 1.88 positives and 39% multi-positive feeds vs 1.22 and 15% in real val M<50. Uniform subsets conditioned on >= 1 positive reproduce val (1.19, 14%). Also recompute the ctx columns, and declare it an information column before the first read (a third witness adds forking paths). |
| lb_stats#6 | modify | 0.02 | Stop rule stands (16 -> 32 is worth 0.00006). The val forecast (+0.00015 over W8) was anchored on the realised W8: W16 reads -0.00013 +- 0.00031 val / +0.00004 +- 0.00019 holdout; kd_reprice#3 had the unbiased one. |
| lb_stats#7 | keep | 0.8 | Inferred with an unknown rival correlation, stated as such. Correct consequence: only true gains of +0.0005 or more move P(top 3). |
| lb_stats#8 | keep | 0.9 | Arithmetic on the per-feed sd (0.247); common-mode, does not move ranks. |
| lb_stats#9 | keep | 0.02 | Correct, and it should bind the two TO-only probe proposals as well (ft_redesign#8, forensics_skeptic#10). Compute the paired SE from val before every probe. |
| lb_stats#10 | keep | 0.0 | Null, measured. |
| lb_stats#11 | keep | 0.05 | Re-derived the side observation: C3 - G3 holdout weekend -0.00023 +- 0.00051, Mon+Tue -0.00189 +- 0.00058; val (both weekdays) +0.00075 / +0.00062, so it is the in-period bias. New: the student pop1 admixture shows no weekday pattern (weekend -0.00094, Mon+Tue -0.00063). Composition: seen-user share public 0.302 / private 0.286; M<50 0.387 / 0.392. |
| lb_stats#12 | keep | 0.6 | The readme has no rule text on finals (checked). Highest value-per-minute item in the list: an automatic best-two-by-public selection would turn every public probe into a selection. |
| ft_redesign#1 | keep | 0.02 | Re-derived file - TO for four pools: W8 +0.00009 +- 0.00043, S16 +0.00001, C3 -0.00002, G3 -0.00003; FT - TO -0.0005..-0.0009. The bound for untested variants is an extrapolation of a quadratic fit and is stated as inferred. |
| ft_redesign#2 | keep | 0.01 | Rule written before the read, failed at file level; the half-lr control shows the member gain is shrinkage. Not re-run by me. |
| ft_redesign#3 | keep | 0.01 | Re-derived: FT(W8) + TO(S16) = 0.40172 vs 0.40173 (-0.00001 +- 0.00023), test corr 0.99972. A cost saver with no expected score effect; no selection involved. |
| ft_redesign#4 | keep | 0.85 | Twin comparison is the right control; removes "distillation breaks the fine-tune" as a motive. |
| ft_redesign#5 | keep | 0.02 | Null; decomposition uses val labels only to score, not to fit, except the nested global offset (worth 0). |
| ft_redesign#6 | keep | 0.9 | Label-free measurement; the one place where private days (8-10 days after train) differ from val (3-5 days). Effect bounded below 0.0002; worth a column in the gate. |
| ft_redesign#7 | keep | 0.02 | Nested by user halves; null. |
| ft_redesign#8 | modify | 0.01 | Information only. A single reading with paired SE 0.0005 (0.00035 pooled) cannot carry the declared decision ("drop the FT half if pooled < 0" is a sign test at about 1 SE on weekend feeds, while the FT shift grows with the day). ft_redesign#3 already removes the cost of the FT stage, so nothing hinges on it: lowest slot priority. |
| ft_redesign#9 | drop | 0.01 | Ceiling is zero by the reviewer's own decomposition; val-only judge. |
| ft_redesign#10 | drop | 0.01 | Cannot be validated offline (val OOF is blind to it); would be selected on public alone. No expected gain by the shrink bound. |
| ft_redesign#11 | keep | 0.01 | Clean checks; 94 test feeds share a user with val (my count: 0.33% of test feeds). |
| ft_redesign#12 | modify | 0.02 | Right protocol, wrong bar: +0.0005 at 2 SE is below the measured detection floor of the residual gate (+0.0013, forensics_skeptic#6) and below the multiplicity of the 11,617 features AUTOFE already screened. Gate at >= +0.0013 val with a planted-signal control and holdout >= 0 where a holdout form exists; the input must be computable for private rows without later snapshots or val labels. |
| pool_composition#1 | modify | 0.02 | Size law holds for the recipe mean; the fit on subsets of the realised W8 over-predicts val (forecast 0.40165, finished W16 0.40144) and hits the holdout (0.32459 vs 0.32460). Its falsification clause is not triggered. Public ceiling 0.3947 inherits the val anchoring. |
| pool_composition#2 | keep | 0.06 | Asymptote read is a sound statistic (reference on the same seeds). Its SE at 2 seeds (0.0011 / 0.0006) means a screen can only extend, never promote. |
| pool_composition#3 | keep | 0.02 | Two-witness cross-fit is the right design; nothing transfers. This is the template for every future composition claim. |
| pool_composition#4 | keep | 0.03 | Null, agrees with kd_audit#8 and kd_literature#6. |
| pool_composition#5 | keep | 0.02 | Rule timestamped before the numbers; conclusion is to submit none. Recipe (5) is the best of five on val (+0.00035 +- 0.00019): selection, not evidence. |
| pool_composition#6 | keep | 0.02 | Null on both witnesses; my final-level control agrees (0.8 W8 + 0.2 C3: +0.00008 +- 0.00024 val). |
| pool_composition#7 | modify | 0.7 | First half stands (W8 > S16 is under one public SE). "The holdout called both signs" is n = 2 with one sign at noise level; it does not make the holdout the better judge (see lb_stats#2, kd_audit#11). |
| pool_composition#8 | keep | 0.8 | Measured; member-level only. No action. |
| pool_composition#9 | keep | 0.02 | Pool-level re-read confirms the closures. |
| pool_composition#10 | modify | 0.03 | Same lever as kd_reprice#5 at a third weight (0.3 after five weights were read). Holdout -0.00117 +- 0.00029; val gain on one day only. One probe for the family in total (the declared W8P20), with SE-denominated thresholds; the 13 GPU-h wide pop1 arm only after a public reading of >= +1.5 SE. |
| pool_composition#11 | keep | 0.7 | Owner decision that the data cannot settle (S16: +0.0001 expected excess; C3: structural hedge at -0.00004). Add one operational criterion: pick 2 should be a file that already has a public score. |
| pool_composition#12 | modify | 0.02 | Premise check done here and it passes for the uniform rule: sub-sampled holdout feeds conditioned on >= 1 positive give 1.19 positives and 14.0% multi-positive feeds vs 1.22 and 14.8% in real val M<50. Needs recomputed ctx inputs and a pre-declared role (information column). |
| winner_model#1 | keep | 0.9 | Measured handicaps; plus the positive-density mismatch measured here. The by-M read-out is the right control. |
| winner_model#2 | modify | 0.02 | Two changes at once (new augmentation flag + 20 epochs) for 13 GPU-h, against a measured absence of a matching gap (long_bets#1). Only after the kwsub by-M read shows the size-shift signature; sample uniformly, not positives-kept. |
| winner_model#3 | keep | 0.01 | Null, measured; equivariance check is a useful invariant. |
| winner_model#4 | modify | 0.05 | kwaux0 and kwwd05 are queued with 4 seeds and the asymptote rule; do not run a parallel member-bar batch on the same knobs. Remaining arms (mix 0.05 / 0.10, wd 0.1) behind those reads, Holm at confirmation. |
| winner_model#5 | keep | 0.02 | Null in residual form on both witnesses. Do not run the variant with val/test candidates as reference class: that is a period detector, not a relevance signal. |
| winner_model#6 | keep | 0.75 | Two measurements, one confound not excluded; the sharper check was not run. Consequence is a caution, not an action. |
| winner_model#7 | keep | 0.01 | Null at pool level. |
| winner_model#8 | keep | 0.02 | Audit; data facts verified by two lenses (0 zero-positive feeds, binary labels). |
| winner_model#9 | keep | 0.02 | Overall numbers re-derived (W8 -0.00069 +- 0.00061, C3 -0.00060 +- 0.00057). |
| winner_model#10 | keep | 0.03 | Not recommended; closes on the running d224 arm. |
| forensics_skeptic#1 | drop | 0.01 | Same lever as kd_reprice#7 / kd_literature#4; CPU pilots null in the file. Its own pass rule (blend >= +0.0005) sits at the 95% upper bound of what the stage can give. |
| forensics_skeptic#2 | keep | 0.01 | Model example of in-half selection removing a +2 SE post-hoc find; holdout confirms the null. |
| forensics_skeptic#3 | keep | 0.01 | Null on both witnesses with a non-linear residual. Note for any future positive: for test rows the bank must be built from train labels only (val labels of the same user are not an allowed input). |
| forensics_skeptic#4 | keep | 0.01 | Null; the pocket is label-defined. |
| forensics_skeptic#5 | keep | 0.01 | Null under nested group-conditional weights. |
| forensics_skeptic#6 | keep | 0.9 | Planted-signal control and a no-signal control at exactly 0: this is the standard every residual gate should meet. Detection floor +0.0013 should replace the +0.0005 bars used elsewhere. |
| forensics_skeptic#7 | keep | 0.01 | Null; 6 of 512 train parts compared by value. |
| forensics_skeptic#8 | keep | 0.8 | Observation at about 3 SE over two witnesses; no lever follows. |
| forensics_skeptic#9 | drop | 0.005 | Found after about ten contrasts, weight at the grid edge, 2 SE on val only, no holdout form, and null in the stronger residual protocol. For test users the conditioning set would include val labels. |
| forensics_skeptic#10 | modify | 0.01 | Same probe as ft_redesign#8. Its rule (skip the fine-tune if public >= W8 - 0.0003) is a non-inferiority call inside 1 SE. Information only. |
| forensics_skeptic#11 | modify | 0.01 | Contingent on a kwsub pass that the interim reads and three lenses do not expect; sub-feed TTA is measured negative on current students. |
| forensics_skeptic#12 | keep | 0.8 | Consistent with every measured bound here. |
| long_bets#1 | keep | 0.02 | Prediction registered before the runs; seen = unseen matching is a fair argument that fidelity is not input-limited. Keep "closed" conditional on U20 read by user fold. |
| long_bets#2 | keep | 0.85 | Measured, with the step-count confound stated and bounded. Rule (never train members on a user subset) follows. |
| long_bets#3 | modify | 0.35 | Owner decision. Control to add: exchange val predictions too (train-only or user-grouped OOF) and verify the blend gain on val with a paired SE before the public slot; one public reading of a corr-0.9 blend has SE 0.001-0.0015 and the declared kill line (-0.0018) is not a test. |
| long_bets#4 | keep | 0.02 | Null. The quoted pool limit (0.40195 / 0.32500) is extrapolated from the realised W8; the finished W16 (0.40144 / 0.32460, forecast +0.0002 over W8 on both) says it is about 0.0002-0.0003 high. |
| long_bets#5 | modify | 0.03 | Motive rests on one seed (kd08w192). Three cells plus an information arm with no multiplicity control: about 30% false pass on the first clause. Second clause compares with the realised W8, against which an identical recipe loses 0.0002-0.0004; use a matched-N reference (W8 + 4 new wide seeds). Require holdout >= 2 SE @last on the 4-pool and Holm over cells. |
| long_bets#6 | modify | 0.03 | Acceptable with the lead's asymptote rule; seven schedule arms need Holm at the confirmation step and the same-seed reference. |
| long_bets#7 | modify | 0.02 | No leak, but no witness: the holdout becomes in-sample and val cannot resolve +0.0001. Conditions: strictly the last checkpoint (verify the saved step; --select_on val must not pick), smoke run before the freeze week, equivalence gate against the holdout-trained twin (test corr >= 0.998, val TO within 2 SE), and pick 2 only. |
| long_bets#8 | keep | 0.02 | Pre-registered prediction and rule; with the kd_audit#2 fold split the read-out is clean. |
| long_bets#9 | keep | 0.02 | Null at honest gates. Skip the optional NN screen: a co-click target is a label-conditional target (the feed's own other clicks), which adds label noise the student cannot observe. |
| long_bets#10 | keep | 0.01 | Null; GBDT single runs, differences below 0.003 not used. |
| long_bets#11 | keep | 0.02 | Demotion is right. Leak control for the untested nested form: the teacher of a row must exclude the row's fold AND the student's evaluation fold (20 teacher fits per member), otherwise the OOF read is contaminated through the soft targets. |
| long_bets#12 | keep | 0.5 | Reading risk for a running arm; pool-level read with the correlation column is the control. |

## 7. Limits of this review

- 2.1 rests on 16 wide students (two disjoint 8-pools); the transfer correlation has an SE of about 0.15 at member level. It rules out a strong transferable seed quality, not a small one, and it cannot say whether the first batch was favourable or the second slightly weak.
- 2.2 uses val-based per-feed sds for public SEs (lb_stats validated this noise model on 120 file pairs; rms z 1.09 for near copies).
- 2.4 ignores the covariance between files that share members; it is a fragility check, not a replacement of the lb_stats fit.
- Idea E needs a lagged rebuild of the popularity columns for the holdout rows (the stored hz view has lag 0 there) and stales only those five inputs; I did not run it.
- I did not audit `src/presub_gate.py`, the 42 fine-tune extras or the pre-trained encoder inputs (earlier audits cover them).
