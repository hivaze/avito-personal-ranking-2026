# research_1112 / match: dedup of the 5 research angles against what we already did

Written 2026-11-12 ~15:20 UTC by the match step of workflow wp7s449jt. Research only: CPU reads, no GPU, nothing edited outside this file.
Inputs: the IDEAS JSON from the shift / recsys / industry / nntricks / tools angles (reports/research_1112/{shift,recsys,industry,nntricks,tools}.md),
matched against reports/COVERAGE_MAP.md, EXPERIMENT_REGISTRY.md, CURRENT_STATUS.md (Log to 14:44), SOLUTIONS_REVIEW.md, deep_PLAYBOOK.md, R7_ARCH_HPARAM.md,
R6_PLAN.md, ROUND5.md, nnx_VERDICT.md, sol_*.md, plus review_ideas.md, REVIEW_SYNTHESIS.md, nn_report.md, pretrain_report.md, FEATURES_REVIEW.md, featrev_*.md,
nnx_research_*.md, r6_external.md, nn_x/r6/logs/r6_eval_screen_*.txt and the running public-gap workflow outputs in nn_x/pubdiag/ (14:07-15:02).

Scales used below:
- **Expected gain** = realistic effect on PRIVATE NDCG of the final pool. Member MDE is ~0.001 val (COVERAGE_MAP Statistics); pool gains are ~0.4-0.6x member gains (ensemble dilution; pubdiag overfit/p1: val->holdout transfer ratio 0.38). Most numbers below are therefore BELOW one MDE.
- **Priority 0-10** = value x feasibility before 2026-12-09 on the GPUs, discounted when the mechanism duplicates a closed experiment.
- Status: done-adopted / tried-closed / partially-tried / untried.

## 0. Bottom line

1. **The ideas collapse to ~44 distinct items.** The most duplicated one, a recency-weighted (fresh-day) cooldown, was proposed by 4 of the 5 angles (shift T1, industry #2, nntricks T1a, tools T13). It is also **already on our books**: review_ideas.md #10 ("over-sample Sep 17-19 in the WSD cooldown", never run), and REVIEW_SYNTHESIS "Later / low prior". Our only evidence is a weak LightGBM proxy (review_ideas §1.6: LATE-EARLY val +0.00061 +- 0.00091, mid control -0.00121 +- 0.00088). It has never been run in the NN. Now that m02 proved cooldown data composition works (+0.00117 val / +0.00134 holdout), it is the best untried GPU item. Priority 7.
2. **Most of the zero-cost "selection policy" ideas are already being implemented right now** by the running public-gap workflow (nn_x/pubdiag/build/rule.json, declared 14:42 UTC):
   - shift S1 / nntricks T4 (joint val+holdout criterion S = min(E0, E1) with E1 = Vp - (Vto - H));
   - D2 / tools T9 / recsys R3 (weekday/weekend, public-like/private-like and hour holdout slices);
   - S2 (broad pools PX70 / PX70r / HG);
   - the pmix veto (e5).
   They are not new; they need adoption into CURRENT_STATUS, not research.
3. **One proposed GPU idea duplicates a closed mechanism:** shift T2 (low-LR "online" continuation AFTER the cooldown). Post-cooldown continuation is exactly "stage 2 after Muon+WSD": val -0.00121, holdout -0.00147, "re-warming undoes the cooldown" (nnx_VERDICT §2.3, §5). R6_PLAN's do-not list says "re-cool already-cooled checkpoints: no". Fold the fresh-data part into the in-cooldown recency arm instead.
4. **Agreement-on-the-line (shift D1, nntricks T3) is effectively answered.** Pool and same-root agreement are identical on val, public and private: PF7x mean pairwise rho 0.9312 val, 0.9308 public, 0.9309 private; every same-root pair agrees to 4 decimals (nn_x/pubdiag/overfit/p3_divers.txt). An AoL slope of ~1 predicts no split-specific shrinkage. That supports "noise + selection" for the m02 public gap, which matches the other checks (M2 public miss t -2.7 vs the earlier-subs fit, noise/noise_out.txt; weekend-weekday holdout M2-PF7x +0.00104 +- 0.00058, i.e. M2 is if anything better on weekends in-period).
5. **Genuinely new and worth GPU, in order:**
   - (a) the recency cooldown and (b) cold-share/late-day-weighted pool batches, run as one wave in the existing cd_lane harness;
   - (c) a frozen pretrained item-ID channel (GPSD/PinFM), the only idea that adds information;
   - (d) a decorrelating co-cool (anti-distillation, "DML-");
   - (e) soft-target val fine-tune;
   - (f) a causal-mask history encoder with a small next-action aux loss.
   Everything else is below MDE, duplicates a closed mechanism, or is a diagnostic already done.
6. **Honest expectation:** P(at least one of (a)-(f) is adopted) ~0.4. If adopted, private gain is about +0.0002..+0.0008 at pool level. That is the same order as m02, and below the public resolution (~0.0025).

## 1. Duplicate clusters (merged before matching)

| merged item | angle ids |
|---|---|
| Recency-weighted / fresh-day cooldown | shift T1, industry "recency-weighted training", nntricks T1(a), tools T13 |
| Post-cooldown online continuation | shift T2 (variant of the above, but a different and closed mechanism) |
| Joint val+holdout pick, winner's curse, public as veto | shift S1, nntricks T4 |
| Breadth over concentration / hedge knobs | shift S2, S3 |
| DOW, late-day, stability, under-transfer location | shift D2, tools T9, recsys R3, industry "temporal-decay diagnostic" |
| Agreement-on-the-line | shift D1, nntricks T3 |
| Val-FT teacher relabels train (K3) | shift T5, nntricks T7 (K3 exception) |
| Weight averaging (WiSE-FT, soups, fold merge, SWA/EMA) | shift T4, shift L3, nntricks C1, nntricks T11 |
| Public-likeness / adversarial coordinate | tools T10, shift D2 / D3 context |
| GBDT as member / stacker / tools | tools T1, T2, T5, T6, T8, T14, T15, T16, T17 |
| GBDT score inside the NN | tools T3, T7, T11 |
| Do-not confirmations | shift "Do-not", nntricks C1-C4, industry HSTU, industry PDA, tools T12 |

## 2. Match table

Codes: CM = COVERAGE_MAP.md, ER = EXPERIMENT_REGISTRY.md, CS = CURRENT_STATUS.md, SR = SOLUTIONS_REVIEW.md, PD = nn_x/pubdiag/.

### 2.1 Selection, diagnostics, policy (no training)

| # | idea (angles) | status | our evidence | exp. private gain | GPU | prio |
|---|---|---|---|---|---|---|
| 1 | Joint val+holdout pick with winner's-curse shrinkage; public as veto, never selector (shift S1, nntricks T4) | partially-tried | PD/build/rule.json 14:42: S = min(E0, E1), E1 = Vp - (Vto - H); PD/build/score_out.txt computes it (M2 S +0.00027, G3 +0.00025). Public-as-veto is already a rule (CS §1: never choose < 0.003 apart). decide4's val-best contender rule is disclosed as non-blind (CS 12:59). Holdout argmax inflation +0.00055 +- 0.00006 (PD/overfit/p2_ckpt.txt). | protects ~0.0005-0.001 of selection error | 0 | 8 (adopt) |
| 2 | Breadth over concentration; hedge shift-dependent knobs (keep c0 / m02 / lr families mixed; no m02-only pool) (shift S2, S3) | partially-tried | Member-equal multi-family policy ADOPTED (CM Ensembling; SR §4.1). Public / val transfer: PF7x 0.9x, G 0.6x, M 0.2x (CS 14:04); val->holdout ratio 0.38 for Pall/PF7x/G3 vs 0.20 for M2 (PD/overfit/p1_decomp.txt). PD candidates C3 PX70, C4 PX70r (root-dedup), C5 HG test breadth. | hedge, ~0.0005 | 0 | 7 (adopt) |
| 3 | DOW / late-day / per-slice stability columns; locate the m02 under-transfer (shift D2, tools T9, recsys R3, industry temporal-decay) | done-adopted | Late-holdout slice mandatory (CS §3 Acceptance). PD/noise/weekend_holdout.txt: M2-PF7x weekend +0.00072 vs weekday -0.00032 (diff +0.00104 +- 0.00058); PD/shift/p1_holdout_weekday.log: private-like M2-PF7x -0.00041 +- 0.00044; PD/noise/val_day_split.txt: Thu-Tue +0.00034 +- 0.00094. Weekday/weekend holdout slices are eligibility e3 in rule.json. Not done: a regression on candidate age. | 0 (diagnostic) | 0 | 3 |
| 4 | Agreement-on-the-line from unlabeled test (shift D1, nntricks T3) | partially-tried | PD/overfit/p3_divers.txt: pool rho val 0.9312 / public 0.9308 / private 0.9309; same-root pairs identical across splits. sol_advval ADV-7: FT~TO Spearman 0.9847 on public and on private. The AoL line has slope ~1, so no shift-specific shrinkage is predicted. | 0 | 0 (CPU) | 2 |
| 5 | Stale-val replay with VETO power for pool picks (shift D3) | partially-tried | Stale-val ADOPTED report-only (CM Shift); mandatory for cold/warm trades (CS §3). pmix veto adopted (CS 12:42; rule.json e5). Not run on the m02 pools (m02 stage 2 had "no stale-val trigger", CS 10:55). Replay is NN inference, so by the user rule (CS §3 Compute, 03:35) it must run on GPU, not CPU. | protects 0-0.0005 | ~0.5 GPU-h | 5 |
| 6 | Public-likeness adversarial coordinate (tools T10) | done-adopted | PD/shift/p2_adv_weights.npz ('private_intrinsic+hour'), used as Vp in rule.json; sol_advval (feed AUC val-public 0.546, coverage 0.897). | 0 | 0 | 1 |
| 7 | Do-not: importance / adversarial reweighting, DRO / IRM / CORAL, post-hoc cold boosts, LB-tuned constants, val-fitted ensemble weights (shift do-not, nntricks C2) | tried-closed | ESS 2,880 (sol_advval; CM Shift); cold offsets learned 0, oracle +0.025 (SR §2); Caruana optimism +0.0014 (sol_ensemble); segment weights closed (CM Ensembling). | 0 | 0 | 0 |

### 2.2 Cooldown and training data composition (cd_lane harness)

| # | idea (angles) | status | our evidence | exp. private gain | GPU | prio |
|---|---|---|---|---|---|---|
| 8 | **Recency-weighted cooldown**: draw regular cooldown batches with P(Sep 17-19) = q, or with day weights exp(-age/tau) (shift T1, industry, nntricks T1a, tools T13) | partially-tried (proxy only; never in the NN) | review_ideas #10 + §1.6 LightGBM proxy (LATE-EARLY val +0.00061 +- 0.00091; mid control -0.00121 +- 0.00088); REVIEW_SYNTHESIS "Later / low prior"; eda_collab_stats: recency windows for statistics are not better. nn_x/r6/train.py has no day weighting (checked). Mechanism precedent: m02 cooldown data selection +0.00117 val / +0.00134 holdout, coldpos +0.0045 (CS 10:55). Train it_hist zero rate 17.5% -> 29.7% from Sep 6 to Sep 19 (review_shift.md:22), so late days are the most test-like. | +0.0002..+0.0005 (member +0.0003..+0.0008, P(PASS-CD) ~0.3) | ~50-line sampler flag; stage 1 = 4 roots x ~1 GPU-h per arm, stage 2 = 3 roots | **7** |
| 9 | Cold-share-weighted or late-day pool batches in the m02 cooldown (nntricks T1b; shift T1 option) | untried | Pool arms so far varied only dose and purity: m02 / m0 / pp02 (R6_PLAN §4.1; CS 08:43). m02's gain sits in coldpos (+0.0045 z 5.7); every re-cool promotes cold candidates equally on val / public / private (PD/overfit/p5_recool.txt: m02 +0.0203 / +0.0216 / +0.0212). | +0.0001..+0.0004 | same wave as #8 | **6** |
| 10 | Low-LR "online" continuation on the last days AFTER the cooldown (shift T2) | tried-closed (mechanism) | Post-cooldown continuation = stage 2 after Muon+WSD: val -0.00121, holdout -0.00147 (nnx_VERDICT §2.3, §5). R6_PLAN Do-NOT: "re-cool already-cooled checkpoints". Only the data axis is new, and #8 covers it inside the cooldown. | <= 0 | 10-20 GPU-min/member | 1 |
| 11 | Fixed-step full-data members: add the 30K holdout feeds to the m02 pool (nntricks T2) | untried | m02 selects at the end (sel_frac 0.96, end - sel +0.00009; PD/overfit/p2_ckpt.txt), so little selection is lost. pp02 (+47% pool) is RUNNING (CS 08:43) and gives the pool-size elasticity. The cost: those members have no holdout, so the H / D terms of rule #1 cannot be computed for any pool containing them. | +0.0001..+0.0003 | 7 roots x ~1 GPU-h | 3 (only if pp02 passes) |
| 12 | Late regularisers in the cooldown: late dropout (d20), then R-Drop, then AWP (nntricks T5) | partially-tried | d0 / d20 wave-2b arms RUNNING (CS 08:43; nn_x/r6/logs/r6_cd_d20_*). w10 (wd 0.1) STOP (CS 08:25). adrop0 FAIL (r6 screen). SAM in fta below MDE (CM Fine-tune). R-Drop / AWP never run. | 0..+0.0003 | R-Drop +70%, AWP +100% cooldown time; 4 roots x 1.5-2 GPU-h | 3 (R-Drop only if d20 >= d0) |
| 13 | Shrink-and-perturb a plateau snapshot before branching a cooldown (nntricks T10) | untried | Branches are not seed-like: F7x~F7x70 same-root 0.9705 vs cross-root 0.944; F7x~F7b 0.9675 (PD/overfit/p3_divers.txt). Within-family cross-seed 0.938-0.946. | +0.0001..+0.0003 (cheaper decorrelated members) | flag + 4 branches ~4 GPU-h | 3 |
| 14 | Data-bagged trunks: 80% user subsamples per member (nntricks T6) | untried | Cross-family corr 0.92-0.94 ~ seed level (R6_PLAN §0.4); GRU / d192 did not lower rho. Counter-evidence: bagging hurts deep ensembles (Lee 2015); our ~0.0035 per data doubling implies ~-0.001 per member at 80%. | -0.0005..+0.0005 | ~0 extra if it replaces planned seeds | 3 |
| 15 | Time-forward holdout for checkpoint selection (shift T3) | untried | Holdout interleaved over all 12 days (shift.md check); holdout argmax inflation +0.00055, of which +0.00034 replicates on val (PD/overfit/p2_ckpt.txt). m02 members select at the end anyway. The late-holdout slice (8,279 feeds) already exists as a report column. | 0..+0.0002 | flag only; new runs | 2 |

### 2.3 Fine-tune stage

| # | idea (angles) | status | our evidence | exp. private gain | GPU | prio |
|---|---|---|---|---|---|---|
| 16 | Soft-target val FT: target (1-a) y/K + a softmax(s_TO) (recsys R2) | untried | FT targets never changed. Label smoothing (trunk) and born-again KD (trunk) are closed (CM Losses; nnx_VERDICT §5). FT peaks at epoch 1 in 37/52 runs (CM Fine-tune). The FT half adds less as members improve: FT-TO F7x +0.00045 vs F5cd +0.00147 (PD/overfit/p1_decomp.txt). The output-space w_ft curve is flat +-0.00025 (ROUND5 §2c), so expect a small effect. | 0..+0.0003 | FT only, ~10 min/member/alpha; needs >= 2 FT seeds | 4 |
| 17 | FT small-data tricks: layer-wise LR decay, head re-init, holdout feeds as extra FT data (nntricks T9) | partially-tried | --add_random 12000 random-like train feeds in FT: neutral, +0.0003/+0.0004 (nn_report:101; sol_solvers:505). Branch lr x10 vs x3 null (review_overfit:156). "FT lr, batch, wd, extra_lr_mult: NEVER, low prior" (CM). LLRD / re-init never run. | 0..+0.0003 | minutes to ~1 GPU-h per arm | 3 |
| 18 | WiSE-FT weight-space TO<->FT interpolation (shift T4) | partially-tried | Output-space w_ft: nested +0.00024 (z 0.96), sign flips with folds (ROUND5 §2c). Closed by argument: R6_PLAN §2 ("our 0.5 FT + 0.5 TO is already the output-space version"); nnx_research_training_gen §9.3 (FT lr 5e-5 moves weights little, so weight space ~ output space; "mix 0.2 TO" v1 +0.0005, v2 neutral). | 0 +- 0.0002 | inference only | 2 |
| 19 | Fold soup for FT test inference (shift L3) | untried | Test fold models correlate 0.9974 (sol_ensemble.md:50), so the soup equals the mean to ~4 decimals. | ~0 | minutes | 1 |
| 20 | Relabel train exploration-like feeds with the val-FT teacher; KD in the cooldown (K3) (shift T5, nntricks T7) | untried | deep_PLAYBOOK K3 (eval trap: the teacher saw val), never run (nn_report:169). Different-mechanism relatives closed: born-again KD with a TO teacher +0.0006 n.s. and it raises correlation (nnx_VERDICT §5); test PL -0.0004 (CM). Needs a 2-fold val split for a clean read. | 0..+0.0004 | high: teacher inference on ~164K feeds + 2 re-cools per root + FT | 3 |

### 2.4 New member families: architecture, inputs, auxiliary losses

| # | idea (angles) | status | our evidence | exp. private gain | GPU | prio |
|---|---|---|---|---|---|---|
| 21 | **Frozen pretrained item-ID channel** (GPSD sparse transfer + freeze; PinFM candidate-ID randomisation; kNN-imputed IDs for cold) (industry #4) | untried | Hashed ID residual from scratch CLOSED -0.0022, multi-epoch memorisation (nn_report N4). pre2s scalar scores ADOPTED (+0.0019 at 3 ep, +0.0009 at 10 ep). The pretrain's incremental signal is its ID part (-0.004/-0.005 zero-shot without it; pretrain_report TL;DR). Risks: the LUO-cold availability bit (pretrain_report §8b); ID coverage train 62% vs val 46% / test 41%. The stale replay found the pre2 ID part "nearly inert" (SR §3 CPU2 A). | +0.0003..+0.001 if it works (P ~0.25) | pretrain 25-45 GPU-min + ~1 day of plumbing + 2-seed screen + family ~13 GPU-h | **5** |
| 22 | Mutual learning in the cooldown: DML+ (KL to the peer) and DML- (decorrelation / anti-distillation) (recsys R1) | untried | Not in CM / ER (only a KDCRec mention, deep_position_bias_random:214). Related closed: KD raises teacher correlation (nnx_VERDICT §5); weight averaging kills diversity (avg3 pool -0.0024, CS 02:35). The pool is correlation-limited: neff ~1.07 at rho 0.93 (PD/overfit/p3_divers.txt); frontier law gain ~0.019(1-rho). DML+ alone is likely <= 0 at pool level. | DML- +0.0001..+0.0004 if rho drops >= 0.01 at equal strength | ~60-80 lines + 1.5-3 GPU-h per arm | 4 |
| 23 | Causal-mask history encoder + small next-action aux loss (NAL, w ~0.01) (industry #1) | untried | hist_enc is fully visible with a learned recency embedding (nn/model.py:346, :442). NAL "parked" (r6_external #18; nnx_design §8; nnx_research_recsys_arch P3). The pretrain encoder is causal + RoPE (pretrain_report §2). r7 0/13 block forms; GRU family adds nothing to the pool (R6_PLAN §0.4). | 0..+0.0004 (NAL dense supervision; diversity) | low: mask ~5 lines, NAL ~40; 2-seed 2-ep screen | 4 |
| 24 | "Example age" input: feed day index on train, 0 or negative at inference (industry #3) | untried | The closest closed items are different mechanisms: calendar / hour features hurt cross-day transfer (feat_cal_.md:8, :178); per-item posting time (feat_life_.md:212). Train-row co-vis is day-causal (nn_report Round 2 `--xt cv`), so its meaning drifts by day. Must pass the has-later probe. | 0..+0.0003 | one input; 2-seed 2-ep screen ~1.5 GPU-h | 3 |
| 25 | Cross-fitted GBDT score as an NN input or frozen base logit (+ N(0, 0.1) noise; leaf-SVD as the richer variant) (tools T3, T11, T7) | untried | Only the weak direction was run: LightGBM residuals -0.0015..-0.011, nested clogit residual -0.0006 +- 0.0005, "no residual signal" (SR §2). Leaf embeddings are on the FEATURES_REVIEW do-not list (unmeasured, judged low; featrev_research F6). The GBDT sees the same 34 features, so the information is mostly subsumed. | 0..+0.0004 (mostly diversity) | CPU 3-6 h GBDT (OOF by user AND time) + 0.5 day dev + 2-4 NN proxy runs | 3 |
| 26 | Missing / zero indicators for every NaN->0 family + u_missing + H==0 flags (recsys R4) | partially-tried | has: flags exist for pre2_fe / prew_fe (FEATURES_REVIEW:144); FR-8 parked as "~0, hygiene, free when a preset changes" (FEATURES_REVIEW item 5; featrev_prep PREP-4/5). Availability flags can encode train/val differences (pretrain_report §8b). | 0..+0.0002 | bundle with the next from-scratch preset | 3 |
| 27 | FM-highway per-layer learned dot products to the head (recsys R5) | untried | r7 block forms 0/13 promote (R7_ARCH_HPARAM); dwa nearest miss +0.00064 val / -0.00027 holdout. | 0..+0.0003 | ~40 lines; 2-ep screen | 3 |
| 28 | Content-anchored, norm-bounded item-ID delta with per-epoch reset (Let It Go + MEDA) (industry #6) | partially-tried | N4 hashed ID residual CLOSED -0.0022 (no anchoring, no reset; nn_report). The new mechanism is the anchoring and the reset. Fallback for #21. | 0..+0.0003 | medium; 2-seed screen | 2 |
| 29 | Input-pipeline / feature-subset diversity families (recsys R6) | partially-tried | Families already differ in inputs (F6 cvnd, F7 pre2s, F5 prew) at near-seed cross corr 0.92-0.94 (R6_PLAN §0.4). Dropping a block is costly: cvnd alone was +0.0032 val (nn_report v3_cv), so a subset family fails gate (a). | ~0 | 2-seed probe ~4.5 GPU-h | 2 |
| 30 | QueryFormer query generation + multi-view (H = 2-4) target attention (recsys R7) | untried | Overlaps r6 typed / a4u / cset: typed FAIL (val +0.00043 z 1.0, holdout +0.00020; nn_x/r6/logs/r6_eval_screen_*.txt); a4u e10 +0.00036 (CS 12:37). | 0..+0.0002 | ~80 lines; 2-ep screen | 2 |
| 31 | Action-conditioned history tokens (FiLM by event type / source + per-head bias) (recsys R10) | partially-tried | r6 typed view FAIL (above); N5 typed bias `aux_sim2` neutral (nn_report:169); a4u per-type counts PROCEED (CS 12:37). | ~0 | 2-ep screen | 2 |
| 32 | Cross-feed InfoNCE auxiliary on positives (recsys R8) | untried | pre2 encoders already carry a next-item contrastive signal (pretrain_report); the aug arm (history drop + candidate subset) is RUNNING (CS 08:43). | 0..+0.0002 | ~40 lines; 2-ep screen | 2 |
| 33 | RoPE instead of the learned recency embedding (industry #8) | untried | Learned self.rec(256) (nn/model.py:346); FoX (a learned recency decay) null (nnx_VERDICT §5). | ~0 (diversity only) | low | 2 |
| 34 | Muon+ (an extra normalisation after orthogonalisation) (recsys R11) | untried | Muon ADOPTED (CM Optimisation); muon_in -0.00101 (R7_ARCH_HPARAM). | ~0 | negligible; piggyback on a pair | 2 |
| 35 | Hyper-deep ensembles: random hyperparameters per member (nntricks T8) | done-adopted (as families) / closed per member | Same-strength different-recipe families ADOPTED as the lever (CM; sol_ensemble §2.5). Per-member randomisation breaks gates (a)+(b); cross-family rho is already seed-level. | ~0 | - | 1 |
| 36 | Loss-prediction "confidence" aux head (recsys R9) | untried | Analogous aux heads are null: B2 propensity head +0.00007 (nnx_VERDICT §5); N3 intent head -0.0014 (nn_report v2). | ~0 | ride along a screen | 1 |
| 37 | Rare-ID hygiene in pre2's 40M-row ID table (recsys R12) | untried | pre2 ID part "nearly inert" under stale replay (SR §3 CPU2 A); a rebuild is expensive. | ~0 | pre2 retrain + feature rebuild + family | 1 |
| 38 | PinFM-style milder age dropout on similarity inputs only (industry #7) | tried-closed (variant) | agedrop FAIL: stale +0.0009 (z 1.4), holdout -0.0019 (z -5.6) (ROUND5 §2a); uniform xt_drop closed (CM Regularisation). a4u / cset coldpos gains did not survive pmix (CS 12:43). | ~0 | low | 1 |
| 39 | Residual-framing family: the NN boosts a frozen linear / clogit base (tools T4) | untried | The nested clogit base exists (sol_errors s04b). No LTR evidence beyond Google T-B-N. | ~0 | ~2 h dev + proxy runs | 2 |
| 40 | Temporal DRO over item groups (TDRO) and V-REx / InvCF variance penalties over time environments (industry #9, #10) | untried | Segment weights closed (CM Ensembling); every cold/warm intervention traded rather than generalised (ER Closed); TabReD / Wild-Time evidence against. | ~0 | medium / low | 1 |

### 2.5 Test-time, averaging and the GBDT / tools closures

| # | idea (angles) | status | our evidence | exp. private gain | GPU | prio |
|---|---|---|---|---|---|---|
| 41 | AdaBN-style test-time re-standardisation (shift L1) / TTT on test histories (shift L2) | untried (L1) / partially-tried (L2) | Train-stage inputs are mostly within-feed z / rank; extras are already standardised with val stats (featrev_inventory:206). TTA all <= 0, crop harmful (R6_PLAN §3). The transductive pre2 already uses test snapshots (+0.0004..+0.0009). | ~0 | minutes / medium | 1 |
| 42 | SWA / EMA / soups / fold merge / within-cooldown checkpoint ensembles for the pool (nntricks C1, T11; shift T4 / L3 context) | tried-closed | avg3 pool -0.0024 val / -0.0014 holdout (CS 02:35); stage-1 EMA null (nnx_VERDICT §5); F7p plateau snapshots fail the gate (CM); 2-ep seeds near-duplicates (SR §3). | <= 0 | - | 0 |
| 43 | GBDT / TabPFN-family member, stacker, more seeds, CatBoost embedding features, linear_tree, TFM feature generators, drift-aware TFMs, RDL backbones, prob / logit / std fusion (tools T1, T2, T5, T6, T8, T12, T14-T17) | tried-closed | Frontier law (SR §2; sol_ensemble). PD CatBoost +0.02 z on G3: -0.00002..+0.00013 (PD/build/info_out.txt (c)). Probability and raw-logit fusion tie the z-mean (sol_ensemble rows D / E). LLM backbones closed (llm_large.md). | 0 | - | 0 |
| 44 | Test pseudo-labels / mixup / TTT / HSTU-SiLU / contrastive SSL / PDA popularity extrapolation (nntricks C3, C4; industry HSTU, PDA) | tried-closed | PL -0.0004 (CM Fine-tune); TTA closed (R6_PLAN §3); DIN + SiLU channel F11 fails (ER); causal / lagged popularity closed (FEATURES_REVIEW §4 do-not); "direction in time adds nothing" (deep_PLAYBOOK C2 #3). | 0 | - | 0 |

## 3. Top untried (or untried-mechanism) ideas, ranked

1. **#8 Recency-weighted cooldown (prio 7).** One cd_lane wave from the @77500 snapshots, reference m02, the R6_PLAN §4.1 PASS-CD rules plus the pmix veto.
   - Dose arithmetic: the cooldown draws ~6.24M feeds (19,670 x 317). Sep 17-19 holds ~0.75M feeds (233-269K/day; eda_shift_overlap.md:28). Baseline gives them 2 passes; q = 0.5 gives ~4.2.
   - Pre-declare q in {0.33, 0.5} or tau = 5 d, not higher. Pool overdose (4.65 passes) is what caused the late-cooldown decline.
   - Judge on val Tue/Thu, stale-val and the PD rule #1 S-score. The late-holdout slice is same-period, so it is optimistic for this arm. Do not use it as evidence.
2. **#9 Cold-share / late-day weighted pool batches (prio 6),** in the same wave.
   - Restricting the pool to Sep 17-19 cuts it to ~27% (~37K feeds); at mix 0.02 that is ~3.4 passes. Keep the dose fixed: either re-scale the mix to ~0.005, or weight by cold share instead of restricting.
3. **#21 Frozen pretrained item-ID channel (prio 5).** The only new-information idea.
   - Gate it on the has-later probe and on the §8b LUO-cold availability check before any NN run.
   - Randomise ~(1 - 41/62) of train candidate IDs to "no-ID" so that coverage matches test.
4. **#5 Stale-val veto on pool picks (prio 5,** partially-tried policy). Run the replay on GPU (user rule) for the m02 pools before the Dec 9 picks.
5. **#22 DML- decorrelating co-cool (prio 4).** Judge only by k-member ensemble value and pairwise rho (the r6 protocol). Run DML+ only as its control.
6. **#16 Soft-target val fine-tune (prio 4).** FT-only and cheap. It needs 2 FT seeds or z >= 2.5. Read the 0.5/0.5 mix, not the OOF alone.
7. **#23 Causal-mask history encoder + NAL 0.01 (prio 4).** 2-epoch paired screen. If it is null on strength but rho_x <= rho_bb - 0.005, it is a family candidate.
8. **#24 Example-age input (prio 3).** Cheap 2-epoch screen. Needs the has-later probe and the presub gate.
9. **#25 GBDT score as NN input / frozen base logit (prio 3).** Only in the strong direction. Kill rule as in tools.md (dval < +0.0005 and dhold < +0.0003 -> stop).
10. **#13 Shrink-and-perturb branches (prio 3)** and **#14 data-bagged trunks (prio 3).** Both are cheap decorrelation probes. Run them only as replacements for seeds that are being launched anyway.
11. (Conditional) **#11 full-data members**, if pp02 shows pool-size elasticity. **#12 R-Drop**, if d20 >= d0. **#20 K3 relabel**, only after #8 / #9 have read out.

## 4. Facts found while matching (useful outside this sweep)

- The re-cool itself, F7b c0 vs F7, gives val +0.00090 / holdout -0.00046 / coldpos +0.0048. The m02 increment (F7x - F7b) is holdout-confirmed (+0.00134) (PD/overfit/p5_recool.txt). So the val-only excess of m02-heavy pools comes from the re-cool harness (finer eval grid, new data order, cold promotion), not from m02. Any new cooldown arm should therefore be read against its own c0 / m02 re-cool control and through the S-score, never against the original F7 members.
- The m02 pools promote cold candidates equally on val, public and private (label-free; PD/overfit/p5_recool.txt). A split-specific behaviour change is ruled out, which leaves the noise + selection explanation.
- shift T2 / tools T13 / industry recency each propose "fresh data"; only the in-cooldown form is compatible with WSD (see #10).
- Some ideas came with an "already covered" or "closed" claim. The project records confirm every one of those claims except two:
  - WiSE-FT is closed only by argument (the weight-space version was never measured);
  - FR-8 indicators are parked, not measured.
