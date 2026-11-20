# INSIGHTS 2026-11-12: what the five deep analyses say, and what to run next

Written 2026-11-12 ~19:00 UTC by the synthesis step of workflow wpatyoqh2.

**Sources.** Five CPU analyses, each in nn_x/insights/<name>/findings.txt with its scripts and logs:
- errors: error anatomy of G3 and G3+NEW;
- labels_loss: labels, metric and loss alignment;
- data_usage: training-data usage and position bias;
- ensemble: ensemble behaviour and the val fine-tune (FT) stage;
- dynamics: training dynamics and schedules.

Context also comes from CURRENT_STATUS (to 17:25), the wave-4 readout nn_x/r6/logs/r6_eval_cd_w4.txt (17:46), the running Optuna FT study nn_x/ftopt, COVERAGE_MAP and RESEARCH_SWEEP_1112.

**New in this synthesis.** Three CPU checks, in nn_x/insights/synthesis/:
1. The pre-declared dynamics E1 readout, re-run now that 4 v30 seeds are complete. Files: dyn_rerun/a7_v30_vs_f7x_rerun.txt.
2. TUNE-half M-slices of every finished ftopt trial. CONFIRM labels were not read. Files: ftopt_tune_mslice.log.
3. The M-split blend under private-mix weights. Files: msplit_pmix.log.

**Conventions.**
- ± is a paired per-feed SE unless stated otherwise.
- TO = train-only, FT = val fine-tune.
- holdout = the 30K train-period random-like M>=50 feeds.
- "pmix" = val re-weighted to the private-test mix (src/pmix.py).
- Label-defined slices (coldpos/warmpos, novelty, rank of the positive) are descriptive, not causal evidence.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

**Ensemble: members limit the pool, not the combiner, and adding more members of the same recipes does not help.**
1. G3 is saturated in member count.
   - Fitting a - b/k gives G3 an infinite-member headroom of about +0.0002 val and +0.0002 holdout (errors) or +0.00013 / -0.00017 (ensemble).
   - Adding the 12 from-scratch m02 members lowers the asymptote: N1 - G3 is -0.00025 ± 0.00028 val and -0.00017 ± 0.00016 holdout.
   - Per-feed member disagreement behaves like noise. The best-member-per-feed oracle is 0.85 of what pure-noise copies give, and a 7-seed single-family pick equals a 7-member cross-family pick.
   - This is why stacking, weights, routing and aggregation all stopped at about +0.0003.
2. m02 makes members more alike.
   - Pairwise correlation rises by +0.009..+0.015 in every family that selects late in the cooldown: F7 0.941 -> F7x 0.950, GRU 0.941 -> 0.951, d192 0.932 -> 0.948.
   - As a result, m02's holdout gain at the 3-member pool level is only +0.0004..+0.0006.

**Where the loss is.** G3 val loss is split exactly per positive (it sums to 1 - NDCG).
3. 53% of the loss is the category choice: the positive's category is ranked #2 or lower.
   - Category NDCG moved only +0.004 in two model generations, and the category-oracle headroom is unchanged (0.1715 -> 0.1708).
   - Novel-category positives are 32% of positives and 37% of the loss. They are anti-ranked against recent-category negatives (AUC 0.37-0.56), yet calibrated (act/exp 0.975-1.016).
   - 42% of missed positives are missed by all 32 members. Feeds that contain one cannot be predicted from observable features (AUC 0.566).
   - So this block is information-limited.
4. 37% of the loss (43% under the private mix) is "right category, buried": the positive is in G3's top category with 3 or more same-category items above it.
   - The beaters are more popular (+0.37 log LUO), older, more history-similar and warmer than the positive. They are not near-duplicates (mean cos 0.70; 6.5% above 0.9).
   - Content and popularity inputs point the wrong way here. Only an item-identity or collaborative signal could separate these pairs.
5. Every recent lever moves along one cold<->warm axis: m02, the FT half, a4u, cset and the NEW members.
   - On warm|recent / cold|recent positives (NDCG points): m02 -0.00064 / +0.00102; a4u -0.00118 / +0.00173; cset -0.00131 / +0.00159.
   - The warm|recent cell (46% of positives, 42% of the loss) responds to no current lever. Only the generational train-stage changes moved it (+0.0061).
   - NDCG is decided at the head of the list: 49% of the NDCG-weighted misordering involves a negative in the top 3, while missed positives (rank >= M/2) carry only 13.6% of the loss.

**Loss and labels: the loss is not the limit; two untested mechanical mismatches remain.**
6. The listwise CE already spends its gradient like the metric.
   - Negative-gradient share in the top 5: CE 32% vs LambdaRank 35%. On same-category negatives: 39.0% vs 40.2%.
   - Across 152 members, Spearman(NDCG, -CE) is 0.95.
   - Every feed re-weighting was null or negative: by M, by number of positives, by difficulty/NDCG sensitivity, by exploration share. A nested Optuna search over these weights gave -0.0010 ± 0.0011.
   - The two untested mismatches:
     - the K9 aux is normalised per batch, so its per-feed weight rises 4.3x with M, and pool batches carry it at 5-7x per unit of production mass (never run: P8);
     - truncation (item 7).
7. **The largest new lever is end-of-feed truncation, which the NN has never seen correctly indexed.**
   - Every NN and MLP position tower used rank-from-end = M-1-raw pos. 85.5% of train feeds have gaps, so the NN's "last" bucket held 13.5 candidates at a 2.6% click rate, while the true last item is clicked 23.5% of the time.
   - A train-only offset on the TRUE rank-from-end, on production feeds only, gives +0.0022 ± 0.0007 val and +0.0021 ± 0.0006 holdout in a CPU GBDT proxy. That proxy reproduces the NN's rho70 result: -0.0041 vs -0.0035.
   - The same offset on exploration feeds hurts: -0.0010 val and -0.0011 holdout, z -3.7.
8. About 1 in 6 exploration positives is a terminal click (the last shown item), and these cannot be identified at test. 92% of train positives reappear in the same user's next history, which is a banned signal.

**Data usage: agreeing with production is signal, and the cooldown fails through memorisation.**
9. Every way of selecting or re-weighting against production-agreeing feeds hurts.
   - NN rho70: -0.0035 val (z -5.7) and -0.0020 holdout.
   - In the proxy: calibrated IPS -0.0058; dropping feeds with a top-slot positive -0.0082.
   - Wave 4 is all STOP: rho70 -0.00350 / -0.00195; rec33 -0.00212 (z -4.3) / -0.00120; hcap +0.00007 / +0.00002.
   - The common failure is late-cooldown memorisation once a subset gets more than ~1 extra pass (rec33 ~2.7 passes, mix 0.1 ~2 passes, m02 +0.65 passes).
   - The main phase buys warm positives and the cooldown buys cold positives.

**FT stage: the small-feed harm is the only actionable part.**
10. The FT half's pool-level value disappeared with pre2s trunks plus the cooldown, not with m02. It was already about 0 on F7 and F7b; cvnd families such as F5cd still get +0.001.
    - On modern trunks FT only reduces variance. The FT branch's infinite-member limit is 0.0003-0.0006 below the TO branch, and FT raises member correlation from 0.945 to 0.957.
    - The pool-level FT effect is negative on M<50 and positive on M>=50. For the G3 pool: M<20 -0.0024 ± 0.0011, M>=200 +0.0014.
    - The M-split blend (TO-only for M<50, 0.5/0.5 for M>=50) gives G3 +0.00054 ± 0.00028 and PF7x +0.00063 ± 0.00029.
    - **New:** under the private mix it is +0.00040 ± 0.00040 (G3), so there is no veto.
    - **New:** on the ftopt TUNE half, every trial's gain comes from M<50, and none beats the M-split (best trial +0.00085 ± 0.00066 vs M-split +0.00094 ± 0.00041).
    - **New:** loss_norm=feed alone (trial 8) removes only ~13% of the M<50 harm (M<50 +0.00029 ± 0.00057 against an FT-half harm of -0.00229 ± 0.00101 on TUNE). So the batch-normalisation skew is not the main cause.
    - Fine-tuning 3 of G3's 20 members reproduces the FT branch of all 20 (k=3: +0.00001 ± 0.00022).
11. The TO halves are miscalibrated by listing age: act/exp is 1.136 ± 0.028 at 1-3 days and 0.957 ± 0.011 above 90 days.
    - The excess sits in young listings with LUO count 0-3. This fits it_hist_*_luo counting other users' future snapshots, which train rows have more of than val/test rows.
    - Most of the FT half's per-member gain is this re-calibration. Its ceiling is small: a nested age offset gives +0.0003 at pool level.

**Dynamics and measurement.**
12. **The re-cool val excess is real at the same recipe (new readout, 4 v30 seeds).**
    - Late-cooldown val-minus-holdout gap, F7x minus v30: +0.00153 ± 0.00020, against the pre-declared bar of +0.0006.
    - Holdout gain over each run's own plateau is the same for both: last checkpoint -0.00019 ± 0.00028.
    - So re-cooled members carry about +0.0015 of val that the holdout does not confirm. The SE is optimistic because the 7 F7x runs share CDSEED 811.
    - The mechanism is open. Optimizer-state reset is the leading hypothesis; lr and data order are ruled out.
13. Screen-reading rules:
    - m02's whole gain over mix 0.1 appears after k~10k of the cooldown, so judge cooldown arms at k >= 12k.
    - Eval curves are step-locked to the batch sequence, so a shared CDSEED makes selection luck common across roots.
    - There is an epoch-phase sawtooth: late-epoch evals beat early-epoch evals by +0.00025 val / +0.00029 holdout (z 4.3 / 7.0).
    - For end-rising m02 families, holdout selection buys only luck (+0.0002..+0.0008 per member).
    - Seeds cannot be triaged early: corr(val at 40k steps, selected val) is +0.09.
    - Schedule knobs have no headroom left: tail EMA <= +0.0001, and single-run sd (0.0008-0.0011) exceeds the effects (<= 0.0005).

---------------------------------------------------------------------------------------------------------------------

## 1. Where the remaining NDCG is lost, and which lever can reach it

G3 val: 1 - NDCG = 0.600. Shares are of that loss, from errors 05_mass and 08_levers, with private-mix shares where measured. The blocks in A are exclusive; B gives cross-cutting views of the same loss.

**A. Exclusive split by the category rank of the positive**

| block | share of loss (val / private mix) | what limits it | levers that moved it | lever that could reach it | realistic |
|---|---|---|---|---|---|
| A1. Category choice: the positive's category is G3's #2 or lower | 53% (cat#2 18.7, cat#3-4 18.1, cat#5+ 16.2) | Information. Category NDCG +0.004 in two generations; category-oracle headroom 0.171 unchanged; novel-category positives anti-ranked but calibrated; exploration intent undetectable (MP-feed AUC 0.566) | Generational changes only (+0.0047) | None in the data. Intent heads, transition priors and exploration detectors are closed | ~0 |
| A2. Right category, buried: top category, >= 3 same-category items above the positive | 37% / 43% | Information. Beaters are more popular, older, more similar and not duplicates; CE already spends 39% of its negative gradient on same-category negatives (~LambdaRank's 40%) | Generational (+0.0061 on warm, recent) | (a) Item-identity / collaborative channel (P8; gate it on exactly these pairs, plan item 5). (b) Removing the production confound: the truncation offset (item 1) could help where the beater is production-favoured; unmeasured | +0.0003..+0.001 if P8 passes (P ~0.1) |
| A3. Top category, 0-2 same-category items above | ~10% | Mostly irreducible near the top (1/6 of exploration positives are unidentifiable terminal clicks) | - | - | ~0 |

**B. Cross-cutting views (overlap with A)**

| view | share | lever status |
|---|---|---|
| warm, recent-category positives | 46% of positives, 42% of loss | No current lever moves it; every arm is 0 or negative there. Candidates: truncation (item 1), P8 (item 5) |
| cold, recent-category positives | 22% of positives, 21% of loss (27% private mix) | m02, a4u, cset and FT all gain here and pay on warm. Input parity (hzcd, item 6) may remove the cause instead of trading |
| small feeds, M<50 | 34% of loss | The FT half hurts at pool level (G3 -0.0013 on M<50 feeds). The M-split fixes it for free (item 2) |
| positives at rank 3-29 | 58% of loss | Head ordering; judge arms by top-3 pair AUC, not missed-positive counts |
| missed positives (rank >= M/2) | 13.6% of loss | Systematic: 42% missed by all 32 members. Information-limited; stop |
| ensemble variance | <= +0.0002 left | Saturated; stop seed scaling (item 8) |
| TO calibration by listing age | <= +0.0003 pool | hzcd (item 6) targets the cause; post-hoc offsets are not worth adopting |
| production-specific half of the trunk fit | snapshot advantage 0.0265 on production train vs 0.0131 on val (up to 0.51 production-only) | Truncation removes one known confound (item 1); the domain-token diagnostic (item 7) locates the rest |

**Reading.**
- About 90% of the loss sits in A1 and A2, and both are information-limited.
- A truncation offset can add +0.0003..+0.001 per member. The free M-split can add ~+0.0004. Calibration and parity fixes add ~+0.0002. Ensemble work adds ~0.
- The only lever with a large ceiling is a new information channel (P8), and its prior is low.

---------------------------------------------------------------------------------------------------------------------

## 2. Where the analysts disagree, and how this plan resolves it

| conflict | resolution |
|---|---|
| errors X1 (within-category auxiliary listwise loss) vs labels_loss ("do not run a within-category hard-negative term") | labels_loss measured that CE already gives same-category negatives LambdaRank's share of the gradient (39.0 vs 40.2%). A within-(feed, category) softmax is algebraically a re-weighting of the within-category part of CE, which is 2.31 of 3.74 nats. errors' own 06_within shows the beaters differ only in identity and popularity. **X1 goes to "later, idle lane only" (P ~0.07).** |
| data_usage E1 (truncation offset for 3 groups, including random-like) vs labels_loss E2 (production only) | labels_loss: the exploration-row offset hurts (A3 -0.0010 / -0.0011, z -3.7). data_usage: the random-like-only arm is +0.0006 / -0.0006, and the production-only arm keeps all the gain (+0.0022 / +0.0021). **Merged as TRUNC-PROD: production feeds only, weight (1 - p_soft).** |
| data_usage E4 (front-load the replay to the high-lr half) vs dynamics E6 (back-load it to the low-lr end) | Opposite directions; both expect val ~0, because m02 = m0 on val at every k. **Later, as one symmetric pair at equal dose, only on idle lanes.** |
| ensemble E2 (FT with loss_norm feed) vs the ftopt trial-8 control | The trial-8 TUNE M-slice: loss_norm=feed removes ~13% of the M<50 harm. **No dedicated GPU run**; loss_norm stays only as a study dimension. |
| errors ("do not adopt M-routing as a blend rule", nested +0.00056 ± 0.00038) vs ensemble E1 (pre-declared M-split confirmation) | Compatible: neither adopts yet. Run ensemble's pre-declared fresh-pool test plus a pmix veto. Caveat: fresh members reuse the same val feeds, so this is member-level, not feed-level, replication. The holdout cannot test M<50 because it holds only M>=50 feeds. |
| critical review D2 (from-scratch m02 trunks as the main GPU spend) vs errors X4, ensemble a8 and decide5 | decide5 showed that adding NEW to G3 lowers S, and the pool asymptote is lower. **Finish the running D2 jobs and launch no new seed waves** unless they pass the asymptote gate (item 8). |
| dynamics: from-scratch m02 members are "honest" vs decide5: re-cools are stronger on the holdout too | Both hold. The new readout shows equal holdout gain over each run's own plateau. The decide5 holdout edge comes from trunk quality (lr1 and d192 trunks are weaker), not from the re-cool. |

---------------------------------------------------------------------------------------------------------------------

## 3. Ranked plan

### 3.0 Summary

GPU-h counts wall-clock lanes on shared GPUs. "Member" means a single train-only member, paired over roots or seeds. The tomorrow GPU items fit in one wave of ~30 runs; GPUs N were idle at 18:45.

| rank | item | when | GPU-h | expected gain (P) | kill |
|---|---|---|---|---|---|
| 1 | TRUNC-PROD: true rank-from-end truncation offset, production feeds, training only | tomorrow | ~1 impl + 1.5 (screen A) + 2 (CD arm B) | member +0.0003..+0.0010, pool ~half (P ~0.3) | A: val < 0 or hold < 0; B: val < 0 or hold < 0 vs m02, M<50 < -1 SE, pmix/stale veto |
| 2 | FT small-feed fix: confirm the M-split; read the ftopt freeze against the M-split; fine-tune only ~4 members per pool | tomorrow | 0 (saves ~75% of FT GPU) | +0.0003..+0.0005 final pool (P ~0.6) | fresh-pool delta <= 0; candidate delta < +1 SE; pmix < -1 SE |
| 3 | AUXF: feed-normalised K9 aux, as a CD arm and a 2-epoch pair | tomorrow | ~0.5 impl + 2 + 1 | member -0.0005..+0.0008, sign unknown (P ~0.2) | val < 0 or hold < 0 vs m02 / base |
| 4 | Re-cool excess mechanism: v30 trunks re-cooled vs their own continuous cooldown, plus an opt-warm arm | tomorrow | ~2-3 (arm A) + 2-3 (arm B) | decision value: sets D for 70-100% re-cool pools (~0.001 of the S rule) | A - continuous gap <= +0.0003 -> stop; B - A within ±0.0003 -> reset hypothesis rejected |
| 5 | P8 item-ID CPU gate, re-targeted to warm in-category beater pairs | tomorrow (CPU) | 0 (GPU only if the gate passes) | +0.0003..+0.001 (P ~0.08-0.1) | cell AUC < 0.53 or in-cell nested gain < 2 SE or overall < +0.0003 |
| 6 | hzcd popularity-parity cooldown (train rows read it_hist_hz_* in the cooldown), with a free age read of the wave-4 arms first | tomorrow (pre-flight) / tomorrow-late (arm) | ~1 impl + 2 | TO member +0.0003..+0.0006, pool +0.0001..+0.0003 (P ~0.15-0.2) | arm's act/exp at 1-3 d not < 1.08; val < 0 or hold < 0; pmix < -1 SE |
| 7 | DOMAIN-DIAG: re-score the snapshots under the inference domain token | tomorrow | ~0.2 | decision value; gates the exam-offset arms (L3) | n/a |
| 8 | Wave-5 hygiene and seed freeze: root-specific CDSEED, keep the last checkpoint, epoch-end snapshots, phase-matched reads, asymptote gate before any seed wave | tomorrow (rules) | 0 (+2 for new m02 references, which double as members) | up to +0.0002 holdout at pool level; cleaner screens; saves 10-20 GPU-h per declined seed wave | n/a |
| L1 | TRUNC stage C: from-scratch 10-epoch members/family (and a combination with AUXF if both pass) | after 1 (and 3) pass | ~4.5 per member | member gain carried into new trunks | family rule + ensemble-level check |
| L2 | FT study v2 on M>=50 only (baseline = M-split): ft_minM, member-specific FT seeds, frozen trunk, epochs 1-2 | if item 2 passes | ~2-3 | pool 0..+0.0003 | CONFIRM < +1 SE vs M-split |
| L3 | Examination offsets: EXAM-NN (RandTopN theta, production rows) / XOFF (exploration rows, half strength) | if DOMAIN-DIAG shows the production-specific half is in the shared trunk | ~2 | 0..+0.0005 (P ~0.1-0.15) | val < 0 or hold < 0; drop if E1+exam <= E1 |
| L4 | X1 within-category aux (wcat) | idle lane only | ~2 | member 0..+0.0005 (P ~0.07) | val < 0 or hold < 0; same-category AUC gain < +0.002 |
| L5 | Equal-dose replay timing pair (front-loaded vs back-loaded) | idle lane only | ~4 | val ~0; holdout/coldpos ±0.002 | val < 0 or hold < 0; pmix veto |
| L6 | Short-exploration relabel (soft p for the domain token / K9 weight in M<50) | idle lane only | ~2 | 0..+0.0004 (P ~0.15) | stage-1 rule; M<50 < -1 SE |
| L7 | Cooldown dropout 0.2 as a memorisation brake for the d192/lr1 m02 trunks | only if those families are wanted in a final pool (currently not) | 6-10 | +0.0004..+0.0008 per member of those families | d20 - m02 val < 0 or hold < 0 |

### 3.1 Tomorrow (2026-11-13): cheap, high information

**1. TRUNC-PROD: true rank-from-end truncation offset in the NN loss.** Merges data_usage E1 and labels_loss E2; the labels_loss E3 CLIPLOG logging rides on it.
- **Mechanism.**
  - Production feeds end at about the last click, so the last shown item, which the ranker put deep, is a positive 23.5% of the time (M>=50).
  - A position-blind model learns from this that "items the production ranker puts last are good": an anti-production confound in 95.7% of its gradient.
  - A fixed training-only logit offset on rank-from-end 0-2 explains these clicks away. Inference is unchanged.
  - The NN never tested this, because every tower used M-1-raw pos (wrong in 85.5% of feeds).
- **Protocol.**
  - Build one NEW harness copy for wave 5 (do not edit nn_x/r6) with `--trunc_off a`, default off. Prove flag-off bit-identity as in tests/test_cdw4.
  - Offset `a * L[group, rkend] * (1 - p_soft_f)` for rkend 0-2:
    - added to the main listwise logits on training batches only; never to K9, eval or predict;
    - group is production M<50 or production M>=50, with lifts from nn_x/insights/data_usage/out/nn_artifacts/trunc_lift.json (prod M<50: 1.00 / 0.54 / 2.01; prod M>=50: 2.81 / -0.20 / -0.22);
    - rkend comes from train_rkend_true.npy;
    - p_soft comes from eda/labels_position/out/train_prandom.npz for all M, so the ~214K exploration feeds hidden among the M<50 feeds are excluded from the offset;
    - a = 1 (the proxy is flat for a in 0.5-1.2).
  - Stage A, in parallel with stage B: a 2-epoch from-scratch paired screen with the r7 harness, seeds 23/24, base vs trunc. The base runs are shared with item 3.
    - Log the pre-clip gradient norm (the return value of clip_grad_norm_) and the per-Mpad-bucket main and aux gradient norms every 50 steps. This is CLIPLOG.
    - Close clipping as an issue if fewer than 20% of plateau steps are clipped.
  - Stage B: CD arm `truncp` from @77500 on roots 101-104 vs m02, under the wave-5 rules (item 8).
  - Read:
    - val, holdout, M<50/M>=50, Tue/Thu, stale-val, pmix, at k >= 12k;
    - the per-class lever table (errors 08_levers.py: does warm|recent improve?);
    - the cat#1|w3+ loss block (05_mass.py).
  - If B passes, run stage 2 on roots 105-107. If A or B passes, go to L1.
- **Expected gain.** Member +0.0003..+0.0010, P ~0.3; pool about half.
  - Shrunk from the proxy's +0.0022: the proxy base over-fits production artefacts more than the NN does, and the NN already has K9 and a domain token.
  - The proxy shows the offset adds to a domain input (dom+trunc1 +0.0021 val / +0.0039 holdout).
- **Cost.** ~1 h implementation and tests; A ~1.5 GPU-h (4 runs); B ~2 GPU-h (4 runs).
- **Kill.**
  - A: pooled val < 0 or holdout < 0 vs base.
  - B: member val < 0 or holdout < 0 vs m02, or M<50 < -1 SE, or the pmix/stale-val veto.
- **Prior evidence.**
  - Only mis-indexed towers were run in the NN:
    - round-1 rev_aux -0.0010;
    - base vs nopos -0.0002 (nn_report);
    - MLP zoo 'pos' -0.0006 (sol_solvers §2.1, loss_zoo.py:202);
    - PAL tower neutral.
  - COVERAGE_MAP "position tower CLOSED (old recipe)" covers only those.
  - Correctly indexed rank-from-end exists only in GBDTs, and all of them are positive: EDA b4 +0.0042; LightGBM GAM +0.0022 / +0.0024; this proxy, 3 seeds, +0.0022 / +0.0021.
  - The exploration-row version is negative: labels_loss A3, -3.7 SE. Not in the registry.

**2. FT small-feed fix: M-split confirmation, the ftopt freeze readout, and an FT budget cut.** Merges ensemble E1/E3/E4 and errors X3; ensemble E2 is dropped.
- **Mechanism.**
  - At pool level, FT collapses member diversity in small feeds without reducing error there. FT-half pool effect: M<20 -0.0024 ± 0.0011, M>=200 +0.0014 (G3); on TUNE, M<50 -0.00229 ± 0.00101 and M>=50 +0.00133 ± 0.00061 (set A).
  - Per member, FT-only minus TO is -0.00042 on M<50 (66% of 32 members negative) and +0.00097 on M>=50 (88% positive).
  - The pattern holds in all 11 pre2s/m02 families; the cvnd families show the opposite sign, which explains why S1 failed on 11-11.
- **Protocol.** 0 GPU, all pre-declared.
  - (a) When >= 6 members that a6 did not use are complete, run nn_x/insights/ensemble/a9_fresh_msplit.py on their pool.
    - Fresh so far: v30 s902/903/904 and v27 s516. Due tonight: v30 s905/906, v27 s517/518, v29 s714/715.
    - Then run it on the candidate pools (G3, PF7x, and the Dec-9 candidates).
    - Pass iff the fresh-pool delta > 0 AND the candidate delta > +1 SE AND pmix >= -1 SE. The pmix rule is added here because the FT half is a cold/warm lever.
    - Today: G3 pmix is +0.00040 ± 0.00040, so no veto.
    - Report the M<20 and M20-49 slices.
  - (b) At the ftopt freeze (~20:45 UTC), apply the study's own pre-declared rule, which compares against n2cpv3.
    - Also report, descriptively, best trial vs M-split and best trial + M-split on CONFIRM.
    - If no trial beats the M-split on CONFIRM M>=50, close FT-recipe search on full feeds.
  - (c) From now on, fine-tune only ~4 members per pool, spread over families. Build the FT branch from that subset and the TO branch from all members.
    - Check once on the next pool: FT(k=5) vs FT(all), kill at < -1 SE.
- **Expected gain.** +0.0003..+0.0005 for the final pool. This is shrunk from +0.00054 because the M-split was found on val.
  - Test has a similar M<50 share: 0.390 (public 0.387, private 0.392) vs 0.409 on val.
  - (c) frees ~75% of FT GPU time.
- **Cost.** CPU minutes.
- **Kill.** Fresh-pool delta <= 0; or candidate delta < +1 SE; or pmix < -1 SE.
- **Prior evidence.**
  - sol_ensemble §2.2 S1 (M-specific w_ft on F5-era pools: nested +0.00034, not replicated on F5cd/V13), hence "segment / routing weights CLOSED".
  - New since then: the sign reverses between cvnd and pre2s trunks; it replicates across 11/11 modern families and on the TUNE half of a different 4-member pool (+0.00094 ± 0.00041).
  - nnstack g3_gate_nofam gave +0.00044 (M<50 +0.00106).
  - The trial-8 control shows the batch-normalisation skew explains about 13% of the M<50 harm, so ensemble E2 gets no dedicated run.

**3. AUXF: feed-normalised K9 auxiliary loss.** labels_loss E1, i.e. the never-run REVIEW_SYNTHESIS P8.
- **Mechanism.**
  - The K9 aux (listwise production-order distillation) is a per-batch mean over M-bucketed batches, while the main CE is per feed.
  - Relative to the main loss, the aux per-feed weight therefore rises 4.3x from M<75 to M>=250: 52% of the aux weight sits on M>=100 feeds vs 28% of the main weight.
  - m02 pool batches carry the full aux weight on ~17% production mass, i.e. x5-7.
  - K1 (+0.0020) changed the main-loss normalisation and with it raised aux:main from 0.50 to 0.81 (2.39 on M~300), so the aux M-profile has never been isolated.
- **Protocol.**
  - Same new harness copy: `--aux_norm feed`, aux = aux_w(step) * 1.695 * sum_f (1 - p_f) * CE_aux_f / max_feeds. The constant 1.695 keeps the epoch-mean aux weight. Flag-off bit-identity required.
  - (i) CD arm `auxf` on roots 101-104 vs m02. This tests the pool-batch part that m02 introduced.
  - (ii) 2-epoch from-scratch pair, seeds 23/24, with --cooldown_mix 0.02, sharing the base runs with item 1A. This tests the M-profile.
  - Read the same slices as item 1, plus M<50/M>=50 (the main expected movement).
  - If both pass: 2 from-scratch 10-epoch members vs the v30 seeds (L1).
- **Expected gain.** Member -0.0005..+0.0008; the sign is unknown a priori. P ~0.2. Pool about half.
- **Cost.** ~0.5 h implementation; (i) ~2 GPU-h; (ii) ~1 GPU-h.
- **Kill.** (i) val < 0 or holdout < 0 vs m02; (ii) pooled val < 0 or holdout < 0 vs base.
- **Prior evidence.**
  - The aux strength and target were varied: aux_pos 1.0 null; tau 30 -0.0009; PL20 null (sol_gpu); aa5 STOP.
  - The normalisation never was: COVERAGE_MAP "P8 pool-batch aux normalisation NEVER"; no flag in nn_x/r6/train.py.

**4. Re-cool excess: mechanism A/B on the v30 trunks.** dynamics E2, with its E1 now read here.
- **Readout already done (pre-declared rule, 4 v30 seeds).**
  - Late val-minus-holdout gap, F7x - v30: +0.00153 ± 0.00020, above the +0.0006 bar. The excess is re-cool-specific, about +0.0015 val per re-cooled member.
  - Holdout gain over the own plateau is equal: selected -0.00013 ± 0.00028, last -0.00019 ± 0.00028.
  - So, by the E1 rule, use ~0.0015 as the same-recipe D for re-cooled members in S = min(V, V - D), replacing the CDSEED-based ~0.0010.
  - The SE is optimistic: the 7 F7x runs share CDSEED 811, and their eval residuals are step-locked.
- **Mechanism under test.**
  - A re-cool differs from a continuous cooldown in four ways: fresh AdamW moments and Muon momentum with no warmup, a new data permutation, 1180 fewer stable steps, and a finer eval grid.
  - lr and data order are already ruled out (CDSEED 811/812/813 all show the excess).
- **Protocol.**
  - Arm A: re-cool v30_f7_m02_s901..904@77500 with m02 (all 4 snapshots exist in nn/ckpt). Compare root-paired with each trunk's own continuous m02 cooldown.
  - This needs a NEW lane script with an INIT override; cd_lane.sh resolves only v17 and r6 snapshots.
  - Arm B: `--opt_warm 200` (200 steps at lr 0 that only fill the optimizer state), in the new harness copy with a flag-off parity test.
  - Read:
    - the val-minus-holdout gap at matched k;
    - last-checkpoint val and holdout;
    - **per-feed slices of the A - continuous val delta: Tue vs Thu, M, feed cold share, coldpos, pmix, stale-val.**
  - The excess is test-relevant only if it grows from Tue to Thu or with coldness. A flat or random pattern means a val-specific artefact.
- **Expected gain.** Decision value. D enters S for G3 (70% re-cooled) and M2 (100%) at ~0.001-0.0015, larger than any expected member gain this week.
  - If B removes the excess, opt-warm re-cools become "honest" members.
- **Cost.** A 2-3 lane-h, B 2-3 lane-h.
- **Kill.** A - continuous gap <= +0.0003 on 4 roots: the excess is specific to the F7 trunks; stop and skip B. B - A within ±0.0003: the reset hypothesis is rejected; keep D.
- **Prior evidence.**
  - CDSEED-813 (gap813 +0.00104, data order not the driver); pubdiag p5_recool.
  - COVERAGE_MAP: "warmup, Muon momentum, AdamW betas, clip NEVER"; nothing on optimizer state for --init continuations.
  - No re-cool of a from-scratch trunk exists.

**5. P8 item-ID CPU gate, re-targeted to the warm|recent within-category cell.** errors X5, an edit to RESEARCH_SWEEP P8.
- **Mechanism.**
  - In A2 (37-43% of the loss) the beaters are more popular, older and more history-similar than the positive, and not duplicates.
  - Content and popularity point the wrong way there. Only "who engages with this specific listing" can separate them.
  - An overall-average gate dilutes such a signal over cells where it cannot help.
- **Protocol.** CPU only.
  - Build the P8 gate feature as specified: top-16 mean/max cosine of the candidate ID vector to history ID vectors, for warm candidates.
  - (a) AUC on (warm positive, warm same-category beater) pairs in G3-top categories, stratified by G3 margin.
  - (b) Nested user-grouped blend z(G3) + w·z(x), overall and within the cell (05_mass / 08_levers split).
  - Go to P8's 2-epoch screen only if the in-cell gain is > 2 SE AND the overall gain is >= +0.0003.
- **Expected gain.** +0.0003..+0.001 at P ~0.08-0.1. This is the only lever with a large ceiling.
- **Cost.** ~2.5 h CPU. GPU (~1.5 h screen, then ~1 day of plumbing) only if the gate passes.
- **Kill.** Cell beater-pair AUC < 0.53, or in-cell nested gain < 2 SE, or overall gain < +0.0003.
- **Prior evidence.**
  - idmax32 null (err_SYNTHESIS:183); hashed ID residual -0.0022 (nn_report N4); agedrop FAIL (ROUND5 §2a); semantic IDs closed.
  - T1 taste profiles +0.0004 ± 0.0003 (not confirmed).
  - Targeting these specific pairs is new.

**6. hzcd: popularity-parity cooldown.** errors X2; the wave-4 gate is now resolved.
- **Mechanism.**
  - The TO half of every family under-ranks young listings and over-ranks old ones: act/exp 1.136 ± 0.028 at 1-3 days, 1.08-1.14 for young listings with LUO count 0-3, 0.957 ± 0.011 above 90 days. nf2_tr was calibrated.
  - it_hist_*_luo counts other users' FUTURE snapshots, and train rows have more future than val/test rows.
  - The FT half's per-member gain is mostly this re-calibration, and it cancels at pool level.
  - Fixing the input semantics removes the cause of the cold/warm trade.
- **Protocol.**
  - Step 0 (CPU, free): run errors 12_age_calib.py and 14_age_pop_calib.py on the TO preds of the wave-4 arms vs m02, same roots.
    - rec33 (STOP, -0.0021) still tells whether recency parity moves the age calibration; hcap is a null control.
  - Pre-flight (CPU): compare the age x count distribution and zero-rate of the train-row it_hist_hz_* against val LUO.
    - Proceed only if hz is closer to val than the train-row LUO is (zero-rate context: train 17.5-29.7%, private 45.6%).
  - Arm (new harness copy):
    - in the cooldown, train rows read data/feats_new/train/it_hist_hz_{all,et1..et4}.npy;
    - val/test inputs stay byte-identical;
    - score the holdout both as built and with hz inputs;
    - roots 101-104 vs m02 under the wave-5 rules;
    - the act/exp-by-age table is a required diagnostic.
- **Expected gain.** TO member +0.0003..+0.0006, pool +0.0001..+0.0003, P ~0.15-0.2.
  - The upper bound from a pure calibration fix is a nested age offset: +0.0006 ± 0.0004 (G3_to) and +0.0003 (G3).
  - There is private upside because private is colder: cold-share >= 45% feeds are 51.5% of private vs 25.9% of val.
- **Cost.** Step 0 and the pre-flight are CPU minutes. The arm is ~1 h of plumbing plus ~2 GPU-h.
- **Kill.** Pre-flight fails; or the arm's act/exp at 1-3 days is not below 1.08; or val < 0 or holdout < 0 vs m02; or pmix < -1 SE.
- **Prior evidence.**
  - COVERAGE_MAP: "hzcv, ixh, it_hist_hz ... causal / lagged popularity CLOSED".
  - featnew_it_hist_hz.md rejected it on GBDT gates only: +0.00111, z 1.49, a cold/warm trade, pmix +0.0008 ± 0.0009. It was never run in the NN or in a cooldown.
  - P4 rec33 and P5 hcap (wave 4) targeted parity by other routes and are both STOP.
  - The measured age x popularity miscalibration, and its link to what FT does, are new.

**7. DOMAIN-DIAG: which part of the production-specific fit the domain token absorbs.** data_usage E2.
- **Mechanism.**
  - On production train feeds the snapshot beats a pool-only reference by 0.0265; on val by 0.0131 (up to 0.51 production-specific).
  - L_snap was scored with the training (production) token.
  - If re-scoring under the inference token collapses the advantage toward 0.0131, the token already absorbs the production-specific part, and the exam-offset arms (L3) have no room.
- **Protocol.** nn_x/insights/data_usage/gpu_domain_diag.py (written, not run), from nn_x/r6, one GPU, roots 101/102, on the fixed 300K production-feed sample. Compare per M bin and per rank of the positive.
- **Expected gain.** Decision value only: it gates L3.
- **Cost.** ~5 GPU-min per root.
- **Prior evidence.** The domain token (K2) was judged only as "neutral" on val (nn_report v2).

**8. Wave-5 hygiene and seed freeze.** Rules, not experiments. Merges dynamics E3/E4 and errors X4; ensemble E5 feeds the Dec-9 rule.
- **Seeds and references.**
  - CDSEED = 900 + root, the same for every arm of a root.
  - New m02 references on these seeds: 4 runs, which double as re-cool members with a new data order.
  - Why: eval residuals are step-locked to the batch sequence, so a single seed makes root-level selection luck common; seed-mixed 4-member pools gained +0.00015 / +0.00025 on the holdout.
- **Snapshots and checkpoints.**
  - Take branch snapshots only at epoch ends (68845, 78680).
  - Add the final step to --save_steps (or use --keep last) in every new launch.
  - Use the last checkpoint for end-rising m02 families, so the holdout becomes an unselected judge. Selected minus last is -0.00009 for F7x and +0.00004 for GRU m02, while argmax luck is +0.0002-0.0008 per member.
- **Reading screens.**
  - Judge cooldown arms at k >= 12k. Reads at k <= 8k are early warnings only.
  - Compare checkpoints only at matched epoch phase.
  - Add the errors 08_levers per-class table (warm|recent / cold|recent) to every arm readout.
- **Seed freeze.**
  - Let the running D2 jobs finish; they are needed for item 2 and as honest members for Dec 9. Launch no new seed waves.
  - Any further seed wave first passes the asymptote gate (errors 07_scaling.py, after >= 4 complete members): the G3-plus-family asymptote must beat G3 by > +0.0003 on the holdout and be >= 0 on val.
- **Dec-9 candidates (ensemble E5).** From-scratch members are replacement candidates, not additions. Pre-declare (a) G3, (b) F10 + from-scratch only, (c) G3 with F7x70 replaced by from-scratch members, scored with S using D ≈ 0.0015 per re-cooled share.

### 3.2 Later (conditional)

- **L1. TRUNC (and AUXF) in from-scratch members.** Only if item 1 (or 3) passes.
  - 2 from-scratch 10-epoch members of the v30 recipe + the flag, judged against the v30 seeds by the unified rule plus the ensemble-level check.
  - Then a family if they pass. If both pass, a combined arm.
  - ~4.5 GPU-h per member.
- **L2. FT study v2.** Only if item 2's M-split passes.
  - Objective on M>=50 TUNE feeds only, with the M-split as the baseline.
  - Dimensions: ft_minM {26, 50} (FT trained without small feeds), member-specific FT seed (all members now share 1000k+s), frozen trunk / extras-only, epochs {1, 2}.
  - Confirm on a fresh member set; the CONFIRM half of set A will already have been used.
  - Kill: CONFIRM < +1 SE vs the M-split.
  - Prior: shorter FT was closed at mix level on older families; LP-FT closed at member level; pool-level objectives were never tried.
- **L3. Examination offsets.** Only if DOMAIN-DIAG shows the production-specific share stays in the shared trunk.
  - EXAM-NN: 0.5·log theta_rand(rank) on production rows, as a second arm next to TRUNC.
  - XOFF: 0.5·g_noend on exploration rows, no end term.
  - Proxy: exam alone +0.0008 val / +0.0015 holdout, but +0 on top of trunc. XOFF full strength +0.00003 ± 0.00024 (6 seeds); half strength +0.0004 (z 1.3).
  - Kill: drop if E1+exam <= E1.
- **L4. X1 within-category aux (wcat).** Only on a lane that would otherwise idle; see §2 for why it is demoted. If run:
  - L = 0.2 in the cooldown;
  - kill unless the member same-category AUC (03_pairs.py) gains >= +0.002 AND warm|recent loss falls.
- **L5. Replay timing at equal dose.** One pair, front-loaded mix(t) = 0.04(1 - t/T) vs back-loaded 0.04 t/T, against m02.
  - Expected val ~0; judge on holdout, pmix and coldpos.
  - Design rule from data_usage/dynamics: no subset gets more than ~1 extra pass.
- **L6. Short-exploration relabel.** Soft p_random for the domain token and the K9 weight in M<50 feeds (nn/train.py:160 sets p = 0); these feeds stay out of the pool. Report M<50.
- **L7. Cooldown dropout 0.2 on the d192/lr1 m02 trunks.** Only if those families are wanted in a final pool; today they are not in any candidate.

### 3.3 Where Optuna is, and is not, worth using
- Three studies ran today.
  - labels_loss loss weights: 30 TPE trials over M, K, exploration and difficulty weights. Nested -0.0010 ± 0.0011: closed.
  - data_usage data knobs: 36 trials. CONFIRM +0.0019 ± 0.0012 (z 1.6), not accepted, and no better than the plain truncation offset. fANOVA: drop_hi 0.85, i.e. dropping feeds is always harmful.
  - nn_x/ftopt FT stage: 14 trials so far, all within ~1.3 SE on TUNE. Every gain comes from M<50, and the free M-split dominates all of them on TUNE.
- **Use Optuna only where a trial costs minutes and the objective is pool-level: the FT stage (L2), with the M-split as the baseline.**
- Do not use it for schedule knobs. Single-run val sd (0.0008-0.0011) exceeds the effects (<= 0.0005), and each trial would need >= 4 roots with root-specific seeds (~2 lane-h).
- Do not use it for loss or data weights (nested negative), or for the truncation strength (flat for a in 0.5-1.2).

### 3.4 Wave-5 launch notes
- One new harness copy (for example nn_x/r7cd/) carries all wave-5 flags: --trunc_off, --aux_norm feed, hz-in-cooldown, --opt_warm, INIT override.
  - Each defaults to off, with an AST check and an end-to-end bit-identity test (as test_cdw4), plus one reviewer pass before launch.
- Approximate run count:
  - 4 m02 references (CDSEED 900+root);
  - 4 truncp + 4 auxf + 4 hzcd CD runs;
  - 6 two-epoch screens (base, trunc, auxf x seeds 23/24);
  - 8 v30 re-cools (A + B);
  - 2 domain-diag.
  - About 32 runs, 15-20 lane-h.
- GPU rules for every launch:
  - CUDA_VISIBLE_DEVICES set to exactly one id;
  - count compute processes with nvidia-smi before each launch: at most 4 per GPU, at least 30 GB free;
  - launch detached and record PIDs;
  - stop only your own PIDs, by exact PID.
  - CPU analysis runs with OMP/MKL/POLARS threads = 12.

---------------------------------------------------------------------------------------------------------------------

## 4. Do-not list additions (2026-11-12)

| do not | why (number) | source |
|---|---|---|
| Re-weight feeds in the main loss by M, number of positives, difficulty / NDCG sensitivity, exploration share or test composition; LambdaLoss-type or difficulty-weighted CE, in the train stage or the cooldown | CE pull vs sensitivity mismatch is real (16x) but every correction is null or negative; Optuna nested -0.0010 ± 0.0011; M-weighting ceiling +0.0003 | labels_loss 02/03/08 |
| Top-k focal, within-category hard-negative re-weighting, PL / per-positive / positive-count variants; temperature or calibration losses | CE negative-gradient allocation ≈ LambdaRank (top-5 32 vs 35%, same-category 39.0 vs 40.2%); multi-positive anti-gradient 0.007% of mass; Spearman(NDCG, -CE) 0.95 | labels_loss 02/09/12 |
| Event-type multi-task heads or graded-relevance targets | deeper event types on 18% of train feeds, rank-percentile difference about 1.4 SE, and the signal comes from later snapshots | labels_loss 01/05 |
| Any end-of-feed / truncation offset or tower on exploration feeds (p_random > 0.5); terminal-click discounting or up-weighting | A3 -0.0010 val / -0.0011 holdout (z -3.7); A6 -0.0002 | labels_loss 07/07c |
| Selection or re-weighting against production-agreeing feeds: learnability (RHO / JEST) curricula, IPS of any calibration, feed dropping (top-slot, highest-excess, last-item), P7 clean-teacher toward the pool-only reference | NN rho70 -0.0035 (z -5.7) / -0.0020; proxy IPS -0.0058, droptop -0.0082, rhoinv70 -0.0060 | data_usage; r6_eval_cd_w4 |
| Recency cooldown (P4 rec33), history-format parity cooldown (P5 hcap) | rec33 -0.00212 (z -4.3) / -0.00120, pmix veto, lost only cold positives (-0.0084); hcap +0.00007 / +0.00002, stale -0.00009 | r6_eval_cd_w4.txt |
| Any cooldown stream that gives a subset more than ~1 extra pass | late-cooldown memorisation: mix 0.1 (~2 passes) and rec33 (~2.7) track m02 to k~8-10k, then lose | dynamics a2; data_usage |
| Optuna or tuning of data-weight knobs (exam, trunc strength, pool weight, IPS clip, drop rates) | 36 trials, CONFIRM z 1.6, no better than the single truncation offset | data_usage 09 |
| Optuna or sweeps over schedule knobs (tail EMA/SWA, lr floor, Muon momentum, warmup, stable-phase exploration share, cooldown length or start) | tail EMA <= +0.0001; single-run sd 0.0008-0.0011 > effects; cooldown length closed (wave 3) | dynamics a3 |
| New lr 1e-3 families | lr1 leads through the plateau but loses in the cooldown (-0.0028 val at 90-98k); the m02 increment on lr1 is only +0.0010 val / +0.0004 holdout, and lr1 m02 members are the weakest m02 members (holdout 0.31886) | dynamics a1 |
| Early seed triage on plateau val | corr(v40k, selected val) +0.09 | dynamics a1 |
| Re-cool branches of existing trunks as a diversity source | same trunk, different data order: corr 0.98 | dynamics a4 |
| Checkpoint comparisons across epoch phases; final verdicts on cooldown arms at k <= 8k | sawtooth +0.00025 / +0.00029 (z 4.3 / 7.0); m02's gain appears after k~10k | dynamics a5, a2 |
| More same-recipe seed waves without the asymptote gate; adding existing families to G3 by default | G3 headroom +0.0002; N1 asymptote below G3; G3 + any existing family <= 0 on val and holdout | errors 07; ensemble a8; decide5 |
| Per-feed member/family selection, routing or gating stackers; further combiner work | best-member oracle = 0.85 x pure noise; best family per feed +0.020 vs same-family seeds equal | errors 02; ensemble a2; nn_x/stack |
| Global w_ft, per-family Optuna weights, linear FT-on-pool residuals, FT-shift transplants | nested: +0.00018 / +0.00008 ± 0.00034 / -0.0001..-0.0005 / no consistent sign | ensemble a4/a5/a3 |
| Fine-tuning every member | FT of 3 of 20 members = FT of all 20 (+0.00001 ± 0.00022) | ensemble a4_ft_fewer |
| A dedicated GPU run of FT with loss_norm feed | trial-8 control: removes ~13% of the M<50 harm, overall +0.00010 ± 0.00031 on TUNE | this synthesis, ftopt_tune_mslice.log |
| Judging FT changes by member-level OOF | member FT gains are variance reduction that the pool already has; the FT branch's k->inf is below TO | ensemble a4 |
| New FT extras without information the trunk lacks; the val-label item CTR prior (critical review D3) | the pre_* channel moved into the trunk (R2 0.117 -> 0.027); items clicked in other val folds are 0.4% of val / 0.3% of test candidates, and removing the FT shift on them changes the pool by -0.00002 | ensemble a3 |
| Missed-positive-targeted features, exploration / novel-category detectors, transition priors, category-intent heads | MP feeds unpredictable (AUC 0.566); 42% of MPs missed by all 32 members; novel-category positives calibrated; category-oracle headroom unchanged | errors 02/03/04/05 |
| Near-duplicate or set-level re-rankers | near-duplicate oracle +0.016 (cos > 0.95), unchanged since err_feed_structure | errors 06 |
| Segment routing (history length, diversity, cold share, hour) | after M x npos adjustment, effects are small and not routable | errors 01 |
| Post-hoc age or cold offsets as a blend rule | nested +0.0003 ± 0.0003 at pool level | errors 13 |

---------------------------------------------------------------------------------------------------------------------

## 5. Notes for the Dec-9 pick rule (inputs, not decisions)
- **Re-cool discount.** Same-recipe D ≈ +0.0015 val per re-cooled member at equal holdout (item 4 readout). G3 is 70% re-cooled, M2 100%, PF7x 18%. Keep S = min(V, V - D) with this D until item 4 resolves the mechanism.
- **M-split.** If item 2 passes, apply the M-split to every candidate pool identically. It changes only M<50 feeds.
- **Honest members.** From-scratch members are replacement candidates for re-cooled ones (item 8). Score them on the unselected last checkpoint, so that the holdout is not inflated by selection luck.
- **Public LB.** Context only. The public paired SE for close pools is 0.0005-0.0007, which is larger than every expected gain in this plan except possibly TRUNC.

## 6. Files
- This synthesis:
  - nn_x/insights/synthesis/findings.txt
  - nn_x/insights/synthesis/ftopt_tune_mslice.py (+ .log)
  - nn_x/insights/synthesis/msplit_pmix.py (+ .log)
  - nn_x/insights/synthesis/dyn_rerun/ (copies of the dynamics parse_logs/fams/a7 scripts pointed at this directory; a7_v30_vs_f7x_rerun.txt)
- Analyses: nn_x/insights/{errors,labels_loss,data_usage,ensemble,dynamics}/findings.txt and their scripts and logs.
- Ready artifacts:
  - nn_x/insights/data_usage/out/nn_artifacts/ (train_rkend_true.npy, trunc_lift.json, theta_rand_rank.npy)
  - nn_x/insights/labels_loss/out/pool_pos_offset.npz
  - nn_x/insights/data_usage/gpu_domain_diag.py
  - nn_x/insights/errors/{07_scaling,08_levers,12_age_calib,14_age_pop_calib}.py
  - nn_x/insights/ensemble/a9_fresh_msplit.py
