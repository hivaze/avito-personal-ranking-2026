# RECHECK 11-14: SYNTHESIS (decision document for the lead)

Written 2026-11-14 22:15 UTC by the synthesiser. Inputs: 106 findings of 9 lens reports, each reviewed by 3 skeptics (`reports/RECHECK_1114/*.md`), `nn_x/w5/logs/E_kd_predeclared.txt` read to its last block (21:47Z), `reports/COVERAGE_MAP.md` (all 311 rows), `reports/CURRENT_STATUS.md` sections 1-6 and the 21:30Z log, `nn_x/w5/lane.sh`.
Conventions: "a / b" = val (14,423 feeds) / in-period holdout (30,000 feeds, M >= 50, train-only level); ± = 1 SE, paired per feed. [M] = measured (in a lens / skeptic report or by me), [I] = inferred, [E] = my estimate. P = P(true pool or file gain >= +0.0005) unless stated. GPU-h = share of one GPU (a d192 10-epoch run is about 1 GPU-h, a re-cool about 0.25).
One measurement is mine: the A/A null of the pooled-z rule (section 1.4; `nn_x/w12/recheck/synthesis/aa_null.py`, `out/aa_null.txt`). I built no file, used no GPU and edited nothing outside this report and that folder.

## 0. The decisions in eight lines

1. **Nothing has to be retracted.** No bug and no leak in the KD path, the U20 patch or the fine-tune; every headline number reproduces in three independent recomputations (kd_audit, two skeptics). No candidate file needs a rebuild.
2. **W8 is a favourable draw of its own recipe** (W8 minus the second 8-pool of the same recipe: pooled z +1.60, the 97.6th percentile of the A/A null). Reference for every pool claim = matched N from the 16 wide students. Default final pair after tomorrow's validity reading: **W16 + W8**.
3. **Submissions for the day that opened at 21:00 UTC:** W16 (validity), W8P20 (one SE-denominated read of the popularity-light view), W8TO (information; only if finals are chosen by hand). Two slots stay unused. The owner first confirms how the platform takes the two finals.
4. **GPU:** the lead's wave 1, the kd08w192 extension and the re-cool harness are the right queue. Add: d192 with dropout 0.2 (8 seeds), sim_bias 2 (4 seeds), a second-seed harness control (the harness's own null), the true phase-split re-cool from the kd08w192 plateaus, then the bundle. Start nothing from section 5.
5. **New calibration:** under the same-recipe null the pooled z of two 8-pools has sd 0.80, not 1. "Pooled z >= 2" passes 0.3% of nulls (not 4%) and a pass is real with P about 0.8 [I], not 1/3. Add a REPLICATE band at 1.6 <= pooled z < 2.
6. **Own-model levers left are worth about +0.0003..+0.0005 public in total** [I, three skeptics agree]. The team merge is worth 5-8x that at 0 GPU-h; the owner's decision is needed by about 11-24 to meet the 12-01 exchange deadline.
7. **Most likely outcome of all running arms is "all null"** (P about 0.7 [E]). Plan for it: freeze W16 + W8, keep pick 2 as an option slot, stop new recipes on 11-28.
8. Future ideas: five staged long bets (section 4) and an idea bank of eight items (section 4.6).

## 1. Verdict on today's results

### 1.1 Bugs and leaks: none

- KD target, `dump_teacher.py`, teacher assembly, `--xfold` / `--xfold_unlab`, holdout exclusion, fine-tune folds: clean by code read (kd_audit#1, leak skeptic section 3). 246.7M-row teacher alignment: centred corr 1.000000 on the holdout rows [M, kd_audit; not re-run by a skeptic].
- Headline pools recomputed from files, exact: base x4 0.39887 / 0.32328; kd05 x4 0.40085 / 0.32349; kd05w192 x4 0.40134 / 0.32408; W8 TO 0.40157 / 0.32456; files W8 0.40165, S16 0.40173, C3 0.40101; W16 TO 0.40144 / 0.32460 (my script agrees with `src/metrics.py` to 5 digits).
- Holdout selection inherited through the teacher: <= 0.0001 [M, leak skeptic].
- Three hazards to carry forward, none active in a file: (a) `--aug_cand` keeps every positive, so sub-feed composition depends on labels; (b) a nested soft-target fine-tune is clean only if the teacher excludes the row's fold and the student's evaluation fold; (c) public has been used as a selector once (W8, post hoc, best of 3 that day); W8P20 would be the second use.

### 1.2 Confirmed risks and their fix

| # | risk (finding) | evidence | fix | status |
|---|---|---|---|---|
| R1 | Rules read against the realised W8 carry a hidden bar (leak skeptic 2.1; kd_reprice#3) | W16 TO 0.40144 / 0.32460 = -0.00013 ± 0.00031 / +0.00004 ± 0.00019 vs W8; second 8-pool -0.00034 ± 0.00047 / -0.00042 ± 0.00028; an identical recipe fails "8-pool vs W8, holdout >= +0.0004" by 0.0008 [M] | matched-N reference from the 16 wide students | adopted 21:24Z (C') and 21:26Z. **Still open:** GRU rule of 17:32Z (read [W8 + 4 GRU] against [W8 + 4 new wide seeds]); kd05w224 / kdmixw192 rule of 18:24Z (its reference, kd05w192 s951-954, is 0.40134 on val against 0.40100 for the mean 4-subset of the 16; equal on the holdout, 0.32408 vs 0.32407) |
| R2 | Member-level screen cannot see the kind of gain width gave (kd_reprice#1) | kd05w192 vs kd05 member +0.00007 ± 0.00041 / +0.00038 ± 0.00024 (fails both bars); asymptote +0.0002 / +0.0007 [M] | pool / asymptote read | adopted 20:21Z (3) |
| R3 | 2- and 4-seed screens keep 11-22% of true +0.0005 effects (feasibility skeptic 3) | power table [I, normal model] | one stage, 8 seeds, pooled-z rule | adopted 21:26Z; refined in 1.4 |
| R4 | Holdout-selected checkpoints favour the hard-label base (kd_audit#3) | selected minus @last, 4-pool holdout: base +0.00041 ± 0.00019, students +0.00005..+0.00009 [M] | read KD vs hard labels @last | no decision changes; KD is confirmed more strongly (kd05w192 4-pool vs base @last +0.00115 ± 0.00037) |
| R5 | kwsub5 / kwsub2 are confounded three ways (kd_reprice#2, winner_model#1, leak skeptic 2.3) | 6 of 20 ctx inputs stale by 0.8-3.6 SD; mean M 78.3 -> 59.2 / 47.7 (test 101.6); positives kept gives 1.88 positives and 39% multi-positive short feeds against 1.22 and 15% in val M<50 [M] | a kwsub null closes this implementation only; report deltas by M bin | adopted 20:21Z (3) |
| R6 | `--kd_T` divides the teacher only (kd_audit#6) | train.py l.500 | scope note on the kwT read-out; no arm | note |
| R7 | U20 pooled holdout read mixes same-user and cross-user feeds; u20a has 78,710 steps vs 98,350 (kd_audit#2) | 5,974 of 30,000 holdout feeds are fold-0 users; val has 852 [M] | GO / KILL on the cross-user columns; the ratio (B-A)/(C-A) is the budget-fair statistic | adopted 21:26Z |
| R8 | W8P20 public thresholds were ±0.7 SE (two skeptics) | paired public SE 0.00041 [M] | SE-denominated rule | adopted 21:24Z (A') |
| R9 | "Public Δ = 0.4 val Δ + 0.4 holdout Δ" is not identified (leak skeptic 2.4) | val slope 0.42 ± 0.18 with W8 and S16, 0.21 ± 0.21 without; holdout coefficient 0.00..1.66 depending on the pop1 files [M] | quote no coefficient; the two-witness rule does not need one | **open: CURRENT_STATUS section 1 still quotes it** |
| R10 | Residual gates recover little of a real input (duplicates skeptic 1.3) | a block worth +0.00852 ± 0.00066 as a trained input reads +0.00086 (linear) and +0.00222 (monotone GBDT) on the holdout, nothing on val [M; one block] | about 300 feature rows are closed "in residual form only"; a further input test must be a student arm read at pool level; neither +0.0005 nor +0.0013 is a closing bar | **open: wording in COVERAGE_MAP** |
| R11 | It is not recorded how the platform takes the two finals (lb_stats#12) | readme has no rule text [M] | owner checks before any further probe | **open (section 6)** |
| R12 | "Fine-tune does not help KD students" is a recipe-family effect, not distillation (ft_redesign#4) | hard-label twins FT - TO -0.00029 ± 0.00041; W8 members -0.00043 ± 0.00041 [M] | drop the motive; FT half is 0 ± 0.0004 at pool level in every pool | wording |
| R13 | CPU fp32 replicas differ from stored bf16 files by up to 0.0008 for one member (kd_audit#12) | pool level -0.00012 ± 0.00015 [M] | compare CPU replays with CPU replays | note |
| R14 | Cost units differ by 4 between reports | 3.2 h wall at 4 jobs per GPU = 0.8-1.15 GPU-h [M] | budget in GPU-h as defined above | note |

### 1.3 Numbers and texts to correct (I edited none of these files)

| where | now | should read |
|---|---|---|
| any quote of W8 as the recipe level | TO 0.40157 / 0.32456 | W8 is the realised first batch. Recipe at N = 8: 0.40139 (sd 0.00025) / 0.32437 (sd 0.00014); at N = 16: 0.40144 / 0.32460; asymptote about 0.4016-0.4017 / 0.3246 [M, leak skeptic] |
| W16 forecasts in kd_audit, lb_stats, pool_composition, long_bets (0.4016-0.4019 val; pool limit 0.40195 / 0.32500; public ceiling 0.3947) | anchored on the realised W8 | 0.0002-0.0003 too high on val; only kd_reprice#3's forecast (0.4014 ± 0.0002 / 0.3245 ± 0.0001) held |
| E_kd 20:16Z (ii), COVERAGE_MAP l.302, CURRENT_STATUS section 6 | "public confirmed pop1 pools 3 of 3" / "confirmed +0.00053 public level effect" | one public draw of nested files: +0.0006 at a paired SE of 0.0005 (1.2 SE) |
| CURRENT_STATUS section 1, level-line bullet | "transfer at about 0.4-0.6; best predictor 0.4 val + 0.4 holdout" | delete; keep the level line and the two-witness rule |
| CURRENT_STATUS section 1, evidence rule | pop1 pools "judged on public (documented in-period holdout bias)" | the bias is documented for hard-label pools; for student pools the holdout penalty has no weekday pattern (weekend -0.00094 ± 0.00039, Mon+Tue -0.00063 ± 0.00044); mechanism test pending (bet 4) |
| CURRENT_STATUS section 1, pick 2 | S16 | W8 once W16 has its validity reading (section 2.3); S16 is worth +0.0001 at most and has a lower asymptote than W16 |
| CURRENT_STATUS section 5, "levers left" | lists soft-target fine-tune and full-data refit | both dropped (section 5) |
| COVERAGE_MAP l.304 | soft-target val fine-tune "OPEN, low" | CLOSED at file level (two independent CPU pilots: -0.00036 ± 0.00031 and +0.00002 ± 0.00044; half-lr control equal) |
| COVERAGE_MAP l.46-47 | dropout / weight decay "NEVER" | dropout varied (kw_drop0, LATE-DROPOUT, WIDTH-GATE drop02); weight decay running (kwwd05, kwwd10) |
| COVERAGE_MAP feature rows closed by a nested residual | CLOSED | CLOSED in residual form only (R10) |
| docs/04_SUBMISSIONS.md section 1 | pick 1 = G3, pick 2 = Pall | W8 / S16 today; add W8, S16 rows with paired SEs |
| FREEZE_PROTOCOL_1207.md | centred on G3 | rewrite before 12-07: default W16 + W8; replacement by the two-witness rule at matched N; public = validity (not more than 2 paired SE below) |
| winner_model#8 | "--loss_norm idcg never reported" | measured neutral before (v2_idcg 0.38004 vs 0.38003) |

### 1.4 New measurement: A/A null of the pooled-z rule [M, mine]

Random disjoint splits of the 16 wide students (one recipe), pool A minus pool B, 400 splits = 800 ordered comparisons.

| split | per-feed paired SE, val / holdout | sd of the pool delta over splits | sd of z_val / z_hold / pooled z | corr(z_val, z_hold) | P(both >= 0) | pass rate at pooled z >= 1.3 / 1.6 / 2.0 / 2.5 (with both >= 0) |
|---|---|---|---|---|---|---|
| 8 vs 8 | 0.00047 / 0.00026 | 0.00040 / 0.00023 (0.86 of the SE on both) | 0.86 / 0.86 / **0.80** | -0.15 | 0.215 | 0.054 / 0.021 / **0.0025** / 0 |
| 4 vs 4 | 0.00057 / 0.00032 | 0.00053 / 0.00027 | 0.94 / 0.83 / 0.87 | -0.04 | 0.234 | 0.068 / 0.031 / 0.006 / 0 |

Reference points: kd05t25w192 x8 minus kd05w192 x8 (same seeds): +0.00044 ± 0.00048 / -0.00021 ± 0.00027, pooled z +0.09. W8 minus the second 8-pool: +0.00034 ± 0.00047 / +0.00042 ± 0.00028, pooled z **+1.60** (2.4% of null comparisons reach it). kd05w192 x8 (the wave-1 reference): 0.40110 / 0.32430.

What follows:
- Seed noise is not an extra variance term on top of the per-feed SE: split-to-split sd is 0.86 of it. There is no transferable pool quality (ratio below 1, cross-witness correlation negative). For pool deltas this answers skeptic contradiction C9 (seed noise against feed noise): it is one noise seen two ways, not two terms to add.
- The adopted rule (both >= 0, pooled z >= 2) is conservative: it passes 0.25% of same-recipe nulls (0.6% by a normal tail at sd 0.80), not the 4% assumed. For a true +0.0005 on both witnesses the expected pooled z is 2.11, so the pass probability is about 0.56; at a threshold of 1.6 it is about 0.74 [I, normal model]. At a prior of 0.04 a PASS is real with P about 0.8 [I]; a reading in [1.6, 2) with P about 0.3 [I].
- The wave-1 reference kd05w192 x8 is 0.0003 below the matched-N mean on val (0.40110 vs 0.40139) and at it on the holdout (0.32430 vs 0.32437): a mild tailwind on val for every arm. Guard in 3.0.
- Limits: one recipe, 16 members, splits share members; an arm with another member correlation can have another null; the snapshot-paired harness has its own null (queue item 1).

## 2. Submissions for the day that opened 2026-11-14 21:00 UTC (5 slots)

### 2.0 Before anything is submitted
The owner confirms on the platform whether the two finals are selected by hand and whether the better of the two counts (lb_stats#12; all three skeptics rank it first by value per minute). If the two best public scores are taken automatically, submit items 1 and 2 only: W8TO is not a file we want among the finals (val -0.00009 ± 0.00043, private-mix reweighting -0.00104 ± 0.00067).

### 2.1 Ordered list (all three files are built and gate PASS; none is submitted yet)

| # | file | members and weights | val (final) | holdout (TO) | test corr to W8; paired public SE | what it tests | reading rule (pre-declared) |
|---|---|---|---|---|---|---|---|
| 1 | `subs/cand_W16kd_r3.parquet` (sha 34579efe...) | 16 wide students: w5_e10_kd05w192_s951-958 + w5_e10_kd05t25w192_s951-958; member-equal per-feed z-mean; 0.5 z(FT n2cpv3s3) + 0.5 z(TO) | 0.40158; -0.00007 ± 0.00034 vs W8 (FT 0.40070, TO 0.40144) | 0.32460; +0.00004 ± 0.00019 (@last +0.00002) | 0.9990; about 0.00035 [I] | validity of the intended pick 1; not a level test (expected +0.0001 over an average 8-pool by the size law) | C' (21:24Z): keeps its place unless public < W8 - 0.0007 |
| 2 | `subs/cand_W8P20kd_r3.parquet` (sha 406e5705...) | per branch 0.8 z(W8 pool) + 0.2 z(mean z of abl_e10_pop1kd05t25_s951-954); final 0.5 / 0.5 | 0.40214; +0.00049 ± 0.00031 (M>=50 +0.00080, M<50 +0.00003; Tue +0.00100, Thu -0.00008) | -0.00063 ± 0.00022, negative in every day group | 0.9994; 0.00041 [M] | whether the popularity-light view pays out of period for a student pool; the only measured structural diversity (same-view admixtures give 0.0000 ± 0.0002) | A' (21:24Z): >= +0.0008 replaces pick 1; +0.0006..+0.0008 invest (fine-tune the 4 wide pop1 students); inside ±0.0006 no action, pick 2 at most; <= -0.0006 view closed. One read for the family. See 2.3 for my reservation on the first clause |
| 3 | `subs/cand_W8TOkd_r3.parquet` (sha 96294e79...) | z of the W8 train-only pool, no fine-tune half | 0.40157; -0.00009 ± 0.00043 | 0.32456 | 0.9979; about 0.0005 [I] | public value of the fine-tune half for students (G3 analogue: -0.00047) | D': information only. No pick or recipe decision. Optional; only under manual final selection |
| 4-5 | unused | | | | | | a slot is spent only on a file that passed a pre-declared rule (none can before the next day) |

Ordering rule: (1) pick candidates get their validity reading first; (2) then one read per new family that has an SE-denominated rule written before the read; (3) information-only files last; (4) never an ordering of near copies, a blend-weight or w_ft variant.

### 2.2 Not to submit
S16 again, W8C5, K5, KD4, K2, a W16-based P20 twin (one read per family), any other pop1 weight, the five tie recipes of pool_composition#5 (all >= 0.9996 to W8; the best, +0.00035 ± 0.00019 on val, is the best of five looks), w_ft variants.

### 2.3 Picks after the readings
- Today: pick 1 = W8 (public 0.394486), pick 2 = S16.
- If W16 reads >= W8 - 0.0007: **pick 1 = W16, pick 2 = W8.** Reason: the pair is right under both readings of the W8 excess (noise: W16 is the better file by a/N, about +0.00013 [I]; a truly better first batch: W8), and no selection is involved.
- W8P20, my reservation on A' clause 1: even at public >= +0.0008 the evidence is val +0.00049 ± 0.00031, public one reading at SE 0.00041, holdout -0.00063 ± 0.00022 against, and "private is mostly weekdays" cuts both ways. Two skeptics say pick 2 at most. Recommendation: treat a >= +0.0008 reading as "pick 2 now, pick 1 only if the lagged-popularity holdout test (bet 4) also clears it". This tightens a declared rule before the read; the owner decides (section 6).
- Pick 2 is an option slot if the better of two finals counts: after W16 is validated, W8 as pick 2 is worth about +0.0001 [I, lb_stats]; a file that passed a pre-declared rule costs almost nothing when it is a false pass and pays when it is real. So: pick 2 = the best rule-passing upside file if one exists on 12-07 (a passing recipe, the pop-light admixture, a merge blend), else W8.

## 3. GPU queue for the next 12-24 hours

### 3.0 Reads due tonight (no new GPU) and how to read them

Template for every read: train-only pools, selected checkpoints, paired per feed against (a) the declared same-seed kd05w192 pool and (b) the matched-N reference vector (per-feed NDCG averaged over about 200 random N-subsets of the 16 wide students). A PASS has to hold against both. Columns without decision weight: holdout @last, asymptote, member-member corr, val M<50 / M>=50, cold / warm, free HOLD-SHORT (`feasibility_value/s3_holdshort_free.py`).

| arm (finishing) | declared rule | forecast from the reviews | what to add |
|---|---|---|---|
| kd05w224 x4 | 18:24Z: 4-pool vs kd05w192 s951-954, val >= 0, holdout >= 0, mean >= +0.0004 | flat (interim) | read (b) too: the declared reference is +0.0003 above matched N on val. FAIL on both -> width axis closed at 192 at dropout 0.1. FAIL on (a) only with pooled z >= 1.6 on (b) -> seeds 955-958, 8-seed rule |
| kdmixw192 x4 | same | tie (teacher identity gives no diversity: 4 lenses) | nothing; no extension |
| kd05w192gru x4 | 17:32Z: [W8 + 4 GRU] vs W8, holdout >= +0.0003, val >= 0 | fixed-N mixing gain 0 (kd_reprice#4) | read against [W8 + 4 of the new wide seeds], mean over the 70 choices |
| kwsub5 / kwsub2 x2 | member rule + Qhat extension (20:21Z) | null or negative | by-M table; a null closes this implementation only; `winner_model/c02_perm_subfeed.py` + `c03_eval_tta.py` on one kwsub checkpoint (8 CPU-min) |
| kwT07 / kwT15 x2 | same | kwT07 members up, pool not (the kd08 pattern) | member corr and Qhat; one-sided T only |
| wide pop1 students x4 | (B): pool-level G_mix and dQ, trained-only | cost about 2 x G_mix | not fine-tuned unless A' reads >= +0.0006 |
| u20a / b / c x2 | 20:21Z + 21:26Z: cross-user columns; GO iff B - A >= +0.0008 holdout, >= 0 val, ratio >= 0.5; KILL iff < +0.0004 | KILL (B - A = 0.0000 ± 0.0004, long_bets) | also dump the three arms on the fold-0 exploration-like train feeds (bet 3) |
| kwaux0 x8, kwwd05 x8, kwwd10 x4, kwh30 x4, kd08w192 x8 | 21:26Z / 21:47Z: both >= 0, pooled z >= 2 | each P 0.03-0.05 | the REPLICATE band below |

**Proposed amendment (write it before the first wave-1 arm finishes, about 00:30Z):** REPLICATE iff both deltas >= 0 and 1.6 <= pooled z < 2 at 8 seeds: 4 fresh seeds (959-962) of the arm and of kd05w192, decision on the 12-seed read with the unchanged PASS rule. Basis: section 1.4 (2.1% of nulls land in or above the band; a true +0.0005 lands there or above with P about 0.74). Bundle entry: both >= 0 and pooled z >= 1.0 (about 10% of null knobs [I]) instead of "point estimate >= 0 on both" (21.5% of null knobs [M]).

### 3.1 Queue (in order; items 2 and 3 are already declared by the lead)

| # | arm | exact flags / patch | seeds | pass rule | P | GPU-h |
|---|---|---|---|---|---|---|
| 1 | Harness control and its null | declared: `K:ref`, `L:ref` (lane.sh stages K / L, 21:46Z). **New: `ref2`** = the same command with cooldown seed 2000 + seed (one lane line; write-new + mv) | 951-958 x K, L (16 + 16 re-cools) | ref vs originals: faithful iff both deltas within ±2 SE (declared). ref2 vs ref gives the harness's own A/A: pooled z on 16 vs 16 and on random 8-subsets of snapshots; if its sd differs from 0.80 by more than 0.2, rescale the harness thresholds before any arm is read | - | 8 |
| 2 | Harness arms (declared 21:46Z) | `K:/L:` kw08, dcd2, dcd0, wd05, wd0, aux0, m05, m10, m0, e4, kw03, in this order of launch | 16 snapshots each | declared: both >= 0 and pooled z >= 2.5 vs ref (PASS); 2-2.5 = bundle candidate. Do not add kw0, e1, lr1, lr4 (in the lane, not declared) without raising the bar | 0.02-0.03 each; about 0.15 that one is real [E] | 44 |
| 3 | kd08w192 (declared 21:47Z) | `E:kd08w192` = `--kd_file .../teacher_mean11.npy --kd_w 0.8 --d 192` | 952-958 (951 exists) | declared: 8 vs kd05w192 x8, both >= 0, pooled z >= 2, the 7 new seeds with the same sign. Report member corr (d128 kd08 collapsed to 0.990) | 0.03 | 7 |
| 4 | **kwd2**: d192 with dropout 0.2 (new) | new lane line `E:kwd2) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --drop 0.2" ;;` | 951-958 | one-stage 8-seed rule + REPLICATE band. Reason: dropout above 0.1 was never run at 10 epochs in any regime; d192 memorises at the plateau; dropout 0 was slightly worse | 0.03-0.04 | 8 |
| 5 | **kwsb2**: sim_bias 2 (in the lane) | `E:kwsb2` = `... --kd_w 0.5 --d 192 --sim_bias 2` | 951-954 | 4-seed statistic: both >= 0 and pooled z >= 1.6 -> seeds 955-958 and the 8-seed rule; pooled z < 1 -> closed. Never completed in any regime (COVERAGE_MAP l.243) | 0.02 | 4 |
| 6 | Width with regularisation, gated | `E:kw256d2` (`--d 256 --drop 0.2`) only if kd05w224 is not below kd05w192 on both references; `E:kd08w256` only if item 3 reads pooled z >= 1.6 | 951-954 first | as item 5 | 0.03 | 4-8 |
| 7 | True phase split at d192 (new stage, after item 3) | re-cool `ckpt/w5_e10_kd08w192_s<seed>@77500.pt` with the stage-K command and `--kd_w 0.5` (E_BASE already saves @77500; one new lane stage) | 951-958 | vs `K:ref` of the same seeds, harness rule (pooled z >= 2.5). Tests kd_audit#7's claim: the kd08 plateau is higher (0.3975 / 0.3209 vs 0.3959 / 0.3196 at d128) and the cooldown gain is intact only at kd_w <= 0.5. `K:kw08` is the opposite direction and serves as its control | 0.03 | 2 |
| 8 | Bundle (declared 21:26Z and 21:46Z) | one from-scratch arm with every wave knob that enters by the rule above, and one re-cool bundle | 951-958 (16 snapshots for the re-cool form) | same rules vs kd05w192 x8 / vs ref. On PASS: fine-tune only the new members, FT branch may stay FT(W16) (ft_redesign#3), file, gate, one slot | 0.05 [E] | 8 + 4 |

Sum beyond wave 1: about 90 GPU-h, inside one day of the cluster (about 190 GPU-h). Idle lanes after that: the rotation references of bet 3, not more arms.

### 3.2 Do not start, and do not stop
- Do not stop anything that is running; every run is about 1 GPU-h and has a rule. kwh30 keeps its 4 seeds; no extension below pooled z 1.6.
- Do not fine-tune new members before an arm passes: the file with FT(W8) and a larger TO pool is the same file (-0.00001 ± 0.00023).
- Do not fine-tune the wide pop1 students before A' reads >= +0.0006.
- Do not launch: more same-recipe wide seeds for level (16 -> 32 is worth 0.00006), any teacher variant, GRU / DIN / CATGRAD / JOINT / lr / user-fold students for diversity, kwh10, kwaux1 (unless kwaux0 is clearly worse), retrieval 32, heads 6, cooldown fraction 0.35 from scratch (the harness has e4), d320, kd_w 0.65, lr 1.4e-3, 15- or 20-epoch function matching, the sub-feed repair, two-sided temperature (unless a kwT arm reads Qhat >= +0.0004 holdout and >= 0 val), residual boosting, the co-click student, a full-data refit, any fine-tune redesign.
- Stop rule for the class "closed under hard labels, retried under distillation" (1 flip in 8 so far): after items 3-6 and wave 1, if nothing reads pooled z >= 1.6, close the class.

## 4. Long bets (multi-day)

### Bet 1. Team merge (owner's decision). P(>= +0.001) 0.25-0.35 given a suitable partner [reviewers]; 0 GPU-h
- Why: the only item above 0.1. Every own lineage is one mean function; the frontier law (refit at the W8 level on 10 partners: gain = 0.0103 (1 - rho) - 0.304 |gap|, within 0.0002 on each) prices a partner at rho 0.86-0.90 and equal strength at +0.0010..+0.0026, a second neural history ranker at +0.0002..+0.0007. Caveat: no partner at corr below 0.95 and gap below 0.005 was ever observed; the range is an extrapolation.
- Gates: (0) owner yes / no and the rules check (exchange deadline 12-01, rules 5.5). (1) Partner: public within 0.002 of ours and not a history-attention neural ranker. (2) Label-free compatibility: within-feed correlation of the two test files, per test day, into the law; proceed iff law - 0.0005 >= 0. (3) Exchange val predictions too (train-only or user-grouped OOF); the blend at the law's weight must gain >= 2 paired SE on val. No weight scan. (4) One public slot, read in SE of the paired delta (0.001-0.0015 at corr 0.9); the old kill line "-0.0018" is not a test.
- First command: write `nn_x/w12/merge/compat.py` (two submission parquets with feed_id, item_id, score -> per-feed z correlation by test day -> law). Validate it on pairs with known answers: W8 vs C3 (test corr 0.9938; law -0.00013, measured -0.00027 ± 0.00039) and W8 vs the CatBoost file (law -0.0045, measured -0.0046). CPU, about an hour.
- Later stage, speculative [E]: if the blend gains >= +0.001 and the partner can score our train rows out of fold, a partner-in-the-teacher student is the one teacher change that alters the target function (all teacher variants so far were the same function).

### Bet 2. The capacity surface: width x teacher weight x regularisation. P about 0.08-0.10 for the whole surface [E]
- Why: width 128 -> 192 is the only change that moved the asymptote under distillation (+0.0002 / +0.0007), d256 failed at dropout 0.1, d192 memorises at the plateau, and at kd_w 0.5 a wider student moves away from its teacher (holdout corr to the 25-member teacher 0.9867 -> 0.9831 -> 0.9795 for d128 / 192 / 256) while at 0.8 it does not (0.9923 -> 0.9924 for d128 / 192, one seed) [M, long_bets#5].
- Stage 1 (running or queued): kd05w224, kwwd05, kwwd10, kd08w192, kwd2, the harness doses (wd0, wd05, dcd0, dcd2, kw03, kw08).
- Reading [E, proposal]: besides the per-arm rule, one pre-declared dose-response statistic per knob with three or more doses (weight decay 0 / 0.02 / 0.05 / 0.10; dropout 0 / 0.1 / 0.2; kd_w 0.3 / 0.5 / 0.8): the slope of the pooled z on the dose. A monotone trend over three doses is stronger evidence than any single arm and uses runs that exist anyway.
- Stage 2 (only on a stage-1 reading of pooled z >= 1.6): cross the winning knobs at d224 / d256 (`E:kd08w256`, `E:kw256d2`, plus weight decay), 8 seeds each.
- Stage 3: bundle x16, confirmation on rotated holdouts (bet 3), fine-tune, file, one slot, pick-2 option.
- Kill: stage 1 without any pooled z >= 1.6 and kd05w224 <= kd05w192 -> the width axis is closed at 192 and the class is closed.
- First command: `W5_GPUS="<ids>" bash nn_x/w5/lane.sh <gpu> "E:kwd2:951 ... E:kwd2:958"` after the lane line of item 4 is added.

### Bet 3. Resolution programme: more evaluation feeds instead of more arms. No score by itself; it decides whether a real +0.0005 is ever seen
- Why: every remaining lever is worth +0.0002..+0.0005; the holdout SE of an 8-vs-8 read is 0.00026; 39% of test feeds have M < 50 and the holdout has none.
- Stage a (CPU, now): the free short-feed column (stored holdout scores restricted to uniform sub-feeds; SE 0.0003; per-feed corr 0.42-0.52 to the full holdout) in every read-out, declared as information.
- Stage b (CPU, when U20 finishes): dump u20a / b / c on the fold-0 exploration-like train feeds (about 11,000 natural short and 26,000 long feeds [I, duplicates skeptic N2]); read B - A and C - A there. Natural short feeds match val in positives per feed (1.06 / 1.14 / 1.23 / 1.33 vs 1.08 / 1.16 / 1.25 / 1.36); sub-sampled long feeds do not.
- Stage c (patch): one flag for a holdout id file (3 lines in a train.py copy, write-new + mv) gives three disjoint 30K rotations from the 89,794 eligible feeds; rotation 0 is today's holdout. Smoke with `W5_SMOKE=1`. Control: a rotation-0 run reproduces an existing member within seed noise.
- Stage d (idle lanes): kd05w192 x8 under rotations 1 and 2 (16 GPU-h). Used only for confirmations: per-feed SE x 0.58, power for a true +0.0005 from about 0.5 to about 0.7 [I, feasibility skeptic].
- Kill: the rotation-1 / 2 reference pools differ from rotation 0 by more than 2 SE in member mean (the cooldown pool changes with the exclusion) -> no rotations, buy seeds instead.
- First command: `CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/recheck/feasibility_value/s5_holdout_pool.py` (lists the eligible feeds), then write the three id files.

### Bet 4. Popularity staleness: a mechanism test before any more popularity-light work. Prior that the test is decisive 0.3 [leak skeptic]
- Why: the pop1 view is the only measured structural diversity (mixing gain +0.00077 ± 0.00029 / +0.00101 ± 0.00019 at a strength cost of -0.0016 / -0.0070). Val and the one public draw favour it, the in-period holdout rejects it at -2.9 SE, and the claim that the holdout is biased against it has never been measured for students. Private is the coldest part of test (cold-share >= 0.45 feeds: 8.4% of val, 21.5% of public, 32.7% of private).
- Design (leak skeptic E): rebuild only the five leave-user-out popularity columns of the 30K holdout feeds with the 7-10 day lag that test rows have (`feat_new/it_hist_hz.py` has the lag logic; the stored `_holdout_hz` files use lag 0), re-score the 8 W8 and 4 pop1kd checkpoints, read W8P20 - W8 there.
- Rule, fixed before the run: penalty shrinks to >= -0.0002 -> the bias explanation holds and the pop-light admixture is eligible as pick 2 (and as pick 1 together with A' >= +0.0008); penalty stays <= -0.0005 -> the view is closed for student pools without another slot; between -> pick 2 at most.
- Gate before inference: label-free parity of the lagged columns against test rows (age x coverage); stop if it fails. It stays a mechanism test: it re-judges no arm that was trained on a stale view (COVERAGE_MAP rows 160, 220, 252).
- Follow-ups only on "bias holds": fine-tune the 4 wide pop1 students, then hzcd students (kd_reprice#8).
- First command: a copy of `feat_new/it_hist_hz.py` restricted to the holdout feed list with lag 7-10 days; then the parity table.

### Bet 5. New information into the student: unlabelled inputs (U20) and trained inputs. P <= 0.03 each [E]
- Why: with capacity closed, only new information can move the common mean function. Two forms remain.
- (a) Unlabelled inputs. U20 (6 runs, queued) decides in one step; the forecast is KILL, because students match the teacher equally on seen and unseen feeds (logit corr 0.9933 / 0.9933). KILL closes test / val feeds as a transfer set, pseudo-feeds with teacher targets, candidate swaps and long function-matching schedules. GO (cross-user columns only) opens: a uniform, label-independent sub-feed sampler with recomputed ctx inputs and teacher-only targets where no positive survives (leak skeptic D), then test feeds as a transfer set, which only public can judge.
- (b) Trained inputs. The residual gates that closed about 300 feature rows cannot see a real input of +0.001 (R10), and the five direct NN tests of added inputs were all under hard labels. One pre-declared student arm, not a hunt: the xrel_more block (4 within-feed rank / z channels; hard-label 10-epoch read -0.0019 val, the same sign width had before distillation) through `nn_x/abl/train.py`, which has the KD block: the `nn_x/abl/pop1/launch_e10.sh` command without its two pop1 flags, plus `--kd_file .../teacher_mean11.npy --kd_w 0.5 --d 192 --xt_rel_extra seq_cv_pop,it_hist_et1_luo,ctx_nd_knn_h,ctx_nd_ucent`, 4 seeds, the 4-seed rule of item 5.
- Gate for (b): run it only if bet 2 shows that regularised capacity is usable (any pooled z >= 1.6); the mechanism is the same. Otherwise it stays unrun and the class is closed with capacity.
- Before any further closing gate is trusted: calibrate it on a real positive control (stored twins without ctx_nd, ctx_hn + ctx_hs, retrieval, pre2; duplicates skeptic N1). A gate closes an idea only if it recovers at least half of a known block of similar size on the same data.

### 4.6 Idea bank (my additions; not reviewed by the skeptics unless marked)

| # | idea | basis | cost | value |
|---|---|---|---|---|
| I1 | REPLICATE band and A/A-calibrated thresholds | section 1.4 [M] | 0; 4-8 GPU-h per triggered arm | pass probability for a true +0.0005 from about 0.56 to about 0.74 at the first stage [I] |
| I2 | The harness's own null (`ref2`) | the harness pairs by snapshot; its noise is unknown | 4 GPU-h | makes 11 arm reads interpretable |
| I3 | Pick 2 as an option slot | section 2.3; needs "better of two counts" | 0 | turns a pass that is real with P 0.3-0.8 into expected score at no downside [I] |
| I4 | Dose-response reading per knob | bet 2 | 0 | sees monotone sub-threshold trends that single-arm rules miss |
| I5 | Positive-control calibration of every gate (skeptic N1) | R10 | CPU | stops both false closures and feature hunts |
| I6 | Private-day drift column in `presub_gate` output (leak skeptic F): per test day, corr to pick 1, cold-minus-warm offset, top-1 agreement | the FT cold offset grows from +0.021 (val Tuesday) to +0.044 (last test day) [M, ft_redesign#6] | minutes; a new wrapper script, not an edit of `src/` | catches a final that drifts on the private days |
| I7 | Daily leaderboard snapshot into `lb_stats/s5_picks.py` | #1 unchanged since 11-12; ranks 2-5 within one SE | 1 CPU-min a day | a rival jump of more than 0.001 is the only outside sign that a trick or a merge exists |
| I8 | Calendar with a stop-loss [E]: 11-15..18 wave 1, harness, items 3-7; 11-18..22 bundle, stage 2, rotations; owner's merge decision by 11-24; last new recipe launched 11-28; last confirmation 12-03; every final gets its public validity reading by 12-06; freeze 12-07 | P(all arms null) about 0.7 [E]; late passes are mostly false | 0 | keeps the last week free of selection on public |

Rule template for every new rule (leak skeptic H): name the reference pool (same seeds and matched N), the statistic and an SE-denominated bar; flag post-hoc files; one public read per family.

## 5. Dropped, and why (do not propose again without a new reason)

| item (finding) | reason |
|---|---|
| Soft-target fine-tune on val (kd_reprice#7, kd_literature#4, forensics_skeptic#1, long_bets#11) | two independent CPU pilots null at file level; half-lr control equal; class bounded at +0.0005; no holdout witness for the stage |
| Low-dimensional fine-tune; test-anchored pseudo fine-tune (ft_redesign#9, #10) | ceiling zero by the author's decomposition; no offline witness |
| w_ft != 0.5, group- or M-conditional FT weight (ft_redesign#7, forensics_skeptic#5, winner_model#9) | nested pick worth 0 in every pool |
| Sub-feed distillation with recomputed ctx inputs (kd_reprice#6) | sub-feed logits equal full-feed logits (KL 0.002-0.005 nats); natural short exploration feeds in the pool were null (XPOOL); a day of code on the training path |
| Size-preserving sub-sampling with 20 epochs (winner_model#2); 15-epoch function matching (kd_literature#2) | no matching gap; kw_e20 negative; in the U20 KILL list |
| History-drop 0.1 arm (kd_reprice#9) | kwh30 x4 answers it |
| Five-epoch students as a cost lever (kd_literature#9) | cost is not binding and the pool is saturated in N |
| Test / val feeds as a transfer set (kd_literature#12) | decided by U20; judgeable on public only |
| Full-data refit of the finals (long_bets#7) | +0.0001 expected, unmeasurable, never-executed code path, removes the second witness |
| Residual boosting on the cross-fitted teacher (kd_reprice#12) | the limit is estimation noise, not bias; needs a train.py copy and 15-model offsets at inference; at most a 0.3 GPU-h falsification on an idle day |
| Two-sided temperature arm (kd_audit#6) | teacher flat and calibrated (T* 1.10, entropy 4.41 of 4.90 nats); only if a kwT arm moves Qhat |
| 36-run d128 phase grid (kd_literature#1) | no power; replaced by queue items 2 and 7 at d192 |
| d320, kd_w 0.65 cells, lr 1.4e-3 (long_bets#5, #6) | motive rests on one seed; kd08w192 x8 decides first |
| Retrieval 32 pair, heads 6, HS1 port, aux 0.25, aux 1.0 (kd_reprice#10, #11, winner_model#4) | closed under hard labels two or three times; 1 flip in 8 under distillation; no KD port for HS1 |
| hzcd students (kd_reprice#8) | substitute judges for a closed arm; only behind bet 4 |
| `--loss_norm idcg` pair (winner_model#8) | measured neutral before |
| Co-click target student (long_bets#9) | label-conditional target; proxy student -0.00109 ± 0.00072 / -0.00176 ± 0.00043 |
| Teacher variants: 25 members, mixed, generation 2, cross-fitted, disjoint, stochastic, multi-head; DKD, WSL, confidence weighting (kd_audit#8, kd_literature#6, #7, pool_composition#4, long_bets#4) | all teachers are one function; in-sample label copy is 0.2-0.7% of target mass |
| More same-recipe seeds; students for diversity (GRU, DIN, CATGRAD, JOINT, lr, folds); C3 or narrow students mixed into the wide pool; member selection or weighting (lb_stats#6, kd_reprice#4, pool_composition#3, #6) | size law saturated; fixed-N mixing gain 0 for every same-view family; nothing fitted on one witness transfers to the other |
| User-bagged members; GBDT student; item-CTR privileged teacher (kd_literature#5, #10, long_bets#2, #10) | -0.0011 / -0.0018 at equal count; 0.017 below W8; +0.00005 ± 0.00018 |
| Re-shuffle TTA, sub-feed TTA, fp32 inference (winner_model#3, #7, forensics_skeptic#11) | equivariant to 1e-4; -0.00072 ± 0.00033; -0.00012 ± 0.00015 |
| Production-membership score; earlier-labelled-feed bank; cold-share gates; seen-user blends; TO branch on previously clicked categories (winner_model#5, forensics_skeptic#3, #4, #5, #9) | null on both witnesses in residual form; the last was found after about ten contrasts with no second witness |
| HOLD-SHORT by re-scoring sub-sampled holdout feeds (lb_stats#5, pool_composition#12) | positives-kept sub-feeds have the wrong label format; the free form (bet 3a) and natural short feeds (bet 3b) replace it |
| Label-free control variate for paired deltas (lb_stats#10) | variance reduction 0.000-0.004 |
| "Holdout decides" (kd_audit#11, pool_composition#7); "accept W16 iff val >= 0 vs W8" (lb_stats#4); public transfer coefficients (lb_stats#2) | two-witness rule at matched N; the holdout misreads popularity-light pools and has no short feeds; coefficients not identified |
| Public slots for W8C5, K5, KD4, K2, weight variants, near-copy orderings (lb_stats#9, pool_composition#5) | expected public z below 0.8; no decision can change |
| A 10-50M parameter model (winner_model#10) | about 3e7 bits of labels [I]; closes with kd05w224 unless bet 2 passes |

## 6. Open questions only the user can answer

1. **Finals mechanics.** Are the two final submissions chosen by hand, and does the better of the two count on private? This gates W8TO today, the option-slot use of pick 2, and how freely probes may be submitted until 12-07.
2. **Team merge: yes or no, and by when.** If yes: is a partner within 0.002 public with a non-neural (or at least not history-attention) model reachable; do the rules allow exchanging val and test predictions before the formal merge; who negotiates. The exchange deadline is 12-01, so the decision is needed by about 11-24.
3. **Pick 2 philosophy.** (a) W8 as the scored twin of W16 (robust, worth about +0.0001); (b) C3 as a structural hedge without distillation (-0.00004 in expectation, test corr 0.9938); (c) an option slot for the best rule-passing upside file. I recommend (c) with (a) as the fallback, if question 1 is answered "by hand, better of two".
4. **W8P20 at >= +0.0008 public.** Keep the declared clause "replaces pick 1", or tighten it now, before the read, to "pick 2 unless the lagged-popularity holdout also clears it" (section 2.3)?
5. **Attribution.** Was the C3 score 0.393454 listed again today a fresh submission of the C3 file? If so it closes item 1 of docs/04 section 9.
6. **Stop-loss.** Is 11-28 acceptable as the last day for launching a new recipe, with the final week reserved for confirmations, validity readings and the merge blend?
7. **Unused slots.** Is it acceptable to leave 2-3 of 5 daily slots unused? Every extra near-copy read adds selection on one public sample of 8,562 feeds and no information.

## Files
- This report: `<repo>/reports/RECHECK_1114/SYNTHESIS.md`
- A/A null: `<repo>/nn_x/w12/recheck/synthesis/aa_null.py`, output `<repo>/nn_x/w12/recheck/synthesis/out/aa_null.txt`
- Rules in force: `<repo>/nn_x/w5/logs/E_kd_predeclared.txt` (blocks 20:16Z to 21:47Z)
- Skeptic reports: `<repo>/reports/RECHECK_1114/skeptic_{duplicates,leak_overfit,feasibility_value}.md`
