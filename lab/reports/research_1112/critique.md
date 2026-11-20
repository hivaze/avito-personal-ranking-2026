# research_1112 / critique: an adversarial review of the matched idea table

Written 2026-11-12 ~15:45 UTC by the critique step of workflow wp7s449jt. This was research only: web search, reading, and read-only greps of reports/ and nn_x/pubdiag/. No GPU was used, nothing running was touched and no existing file was edited.
Input: the matched table in research_1112/match.md (44 items, 11 in "top untried").

Evidence tags:
- **[E]** = measured in the cited paper or writeup.
- **[L]** = measured in this project (the file is cited).
- **[K]** = from my prior knowledge of a paper, not re-read this session.
- **[S]** = search-snippet only.
- **[O]** = opinion.

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **The match step's priorities are too generous.** Seven of the eleven "top untried" items fail an internal fact the project already holds. Most are duplicates of closed mechanisms under a new name.
   - **#16 soft-target fine-tune.** It is the existing 0.5/0.5 output blend plus FT EMA; w_ft is flat [L ROUND5 §2c].
   - **#23 causal history encoder.** The GRU family is already a causal encoder, and it added nothing to the pool [L CURRENT_STATUS 04:05].
   - **#24 example age.** Feature means have no trend across train days [L eda_shift_overlap §2.2], and calendar inputs hurt cross-day transfer [L feat_cal_ §0.3].
   - **#25 GBDT score as input.** The "no residual signal" law covers it [L SOLUTIONS_REVIEW §2], and the pre_final encoder term was -0.0008 [L COVERAGE_MAP].
   - **#13 / #14 extra decorrelated members.** Member count is saturated. Doubling the m02 members (F7x 7 → F7x+F7x70 14) moved val by -0.00003 and holdout by +0.00029 [L CURRENT_STATUS 11:31, 13:55].
   - **#22 DML- (anti-distillation).** External evidence says induced diversity hurts high-capacity deep ensembles [E Abe et al. 2024].
   - **#9 cold-share pool weighting.** Feed cold share is confounded with user-0 and short-history feeds [L review_shift SHIFT-4]. "coldpos ≠ feed cold share" already sank a4u/cset under pmix [L CURRENT_STATUS 12:43].
2. **#8 (recency cooldown, the match step's #1) is weaker than presented.**
   - The industry evidence (TIF on Avazu) comes from ID-embedding CTR models. Our ranker is ID-free.
   - The train→val shift is a traffic-type level difference with no within-train trend [L eda_shift_overlap: "train f1 stays in 0.432-0.455 on every day"].
   - The one real recency channel is LUO-popularity parity (it_hist zero rate 17.5% Sep 6 → 29.7% Sep 19 vs 45.6% private). Co-vis features have the opposite parity direction (expanding window: early days are emptier).
   - Keep it as **one** cheap arm, not a sweep. P(PASS-CD) is about 0.15, not 0.3.
3. **#21 (frozen ID channel) also has an internal refutation the match missed.** The ID-space history similarity `idmax32` was already tested inside the pre_v3A family: +0.00038 ± 0.00030 / -0.00014 / +0.00000 [L err_SYNTHESIS:183].
   - PinFM-style ID randomisation is the agedrop mechanism, which FAILED: stale +0.0009, holdout -0.0019, z -5.6 [L ROUND5 §2a].
   - Keep it only behind a cheap CPU gate. P is about 0.08-0.1.
4. **The m02 "public under-transfer" is not, by itself, evidence of overfit.**
   - The joint test vs val expectations gives chi2(3) 4.77, p 0.19. M2-PF7x residual z = -1.94 [L nn_x/pubdiag/noise/noise_out.txt].
   - External meta-evidence finds little adaptive overfitting on Kaggle holdouts [E Roelofs et al. 2019].
   - A precision-weighted combination of val (winner's-curse shrunk), holdout and public puts **G3 ≈ M2 ≈ +0.0004 over PF7x**. G3 is the only candidate that is non-negative on all three sources (§3.2). This agrees with the pre-declared S-rule's tie-break toward G3.
5. **Three gaps, added with sources:**
   - (a) **Learnability-based cooldown data selection** (RHO-LOSS / JEST, the scalable sample channel of AutoDebias). This is the only new mechanism here that generalises the one lever that worked (m02 cooldown data composition).
   - (b) **Use public as a third estimator, not only a veto**, with an explicit winner's-curse shrink.
   - (c) **History-format parity in the cooldown** (INTEGRITY-1: train histories are built with deeper per-type caps than val/test, *including the m02 pool*). It was never run in the NN.
   - Searched with nothing transferable found: FlightRank 2025 (Kaggle grouped flight ranking; writeups not retrievable, public notebooks are GBDT rankers) and BELKA 2024 (unseen-building-block shift; chemistry-specific).
6. **Statistical realism.** Private paired SE between strong pools is 0.0003-0.0005 [L nn_x/pubdiag/noise/private_se.txt]. Every surviving training idea has an expected *pool* gain of +0.0002..+0.0006, i.e. ≤ 1.5 private SE.
   - Selection protocol and cheap, decisive cooldown arms therefore beat new architecture families.
   - About 27 days remain to Dec 9, so the binding constraint is idea quality, not GPU.

---------------------------------------------------------------------------------------------------------------------

## 1. Attempted refutations of the top-ranked untried ideas

| # | idea (match prio) | strongest counter-evidence | survives? | revised P(adopt) / prio |
|---|---|---|---|---|
| 8 | Recency-weighted cooldown (7) | (i) [E] TIF gains: Avazu DNN AUC 0.7462 → 0.7479, industrial CVR +0.0001..+0.0007 (industry.md I2). Both are ID-feature CTR models where item/user/site ID embeddings go stale; our ranker has no ID input. (ii) [L] No within-train drift: per-day f1/f3/f5 means are flat Sep 6-19; the shift is train-vs-exploration level (eda_shift_overlap §2.2). (iii) [L] The only mechanism is feature parity, and it points both ways: it_hist zero rate rises with day (late days are more test-like), while expanding-window seq_cv is emptier on early days (feat_seq_: "drop Sep 6", FR-4 censoring 8.6% train vs 20.1% val). (iv) [L] LightGBM late-vs-early proxy +0.0006 ± 0.0009, mid control -0.0012 ± 0.0009: noise. (v) [L] Recency windows for statistics are no better (eda_shift_overlap: all-train ≤ Sep 19 +0.0032 vs Sep 17-19 window +0.0029). | partially: there is a real but small parity mechanism, and the harness makes it cheap | 0.15 / 5 (one arm, q = 0.33) |
| 9 | Cold-share / late-day pool batches (6) | [L] In val, feed cold share is confounded with user-0 and short-history feeds (17-88% user 0 at cold share > 0.6; review_shift SHIFT-4 and lines 144-147). On test, cold share rises for *every* feed with time, so it is not a covariate a weight can match. [L] The pmix readout showed that coldpos gains (a4u +0.0095 z 7.1) did not carry to the private-mix (a4u -0.00108 / -0.00032). The "late-day pool" variant is #8's mechanism on a 37K-feed pool (overdose). | no | 0.05 / 2 |
| 21 | Frozen pretrained ID channel (5) | [L] ID-space history similarity (`idmax32`, `pre_v3A` family) was null on top of the ensemble: +0.00038 ± 0.00030 val, -0.00014 holdout (err_SYNTHESIS:183). [L] The required "randomise IDs to test coverage" step is the agedrop mechanism: FAIL, holdout -0.0019 z -5.6 (ROUND5 §2a). [L] Test ID coverage is 41%, and the IDs that staleness removes carry small contributions (mean \|id_contrib\| 0.71 vs 1.26; sol_cpu_CPU2). [E] GPSD's +0.046 AUC (sparse freeze) is for discriminative models that *overfit sparse ID params in epoch 1*. Our ranker has no sparse params to overfit, so GPSD's mechanism does not apply. In favour: the remaining gap is in the mid-list order of *familiar* candidates, where only new information helps (SOLUTIONS_REVIEW §2). | only behind a CPU gate | 0.08-0.1 / 4 (gated) |
| 5 | Stale-val veto on pool picks (5) | No refutation: cheap, protective and pre-declarable. Weakness: stale-val emulates popularity/co-vis coverage only (not pre2_hmax32, ctx_* or history co-vis), and it overshoots private's LUO-zero share (SOLUTIONS_REVIEW:231). It is a pessimistic stress test, so use it as a veto at -1 SE, never as a selector. | yes | policy / 2 |
| 1 | Joint val+holdout pick, public as veto (8) | Already implemented in rule.json (S = min(E0, E1)). Refinement in §3.2: public is ~8.6K labelled test feeds, and the rule's "public enters nothing" discards information. It should enter as a precision-weighted estimate with a weekend caveat. | yes, extended | policy / 1 |
| 2 | Breadth over concentration (7) | Holds. With pool rho 0.93 (n_eff 1.07), breadth helps through *different mean functions*, not member count. Breadth = families that pass gate (a), not more seeds. | yes | policy (merged into item 1) |
| 22 | DML- decorrelating co-cool (4) | [E] Abe, Buchanan, Pleiss and Cunningham (TMLR 2024; ~600 classification ensembles): diversity-encouraging regularisers hurt high-capacity deep ensembles; the best ensembles use more accurate, *less* diverse members. [L] Our pool's diversity comes from plateau-iterate noise; removing it hurt (avg3 -0.0024). That does not imply that *adding* diversity by penalty helps. [L] Gate (a) requires single-member strength within 0.002, which a decorrelation penalty spends first. Caveat: Abe et al. reason about the probability simplex; we z-average per feed, so the transfer is partial. | weakly | 0.07 / 1 |
| 16 | Soft-target val FT toward the TO softmax (4) | It is a weight-space version of what the output blend already does: final = 0.5 z(FT) + 0.5 z(TO), with FT lr 5e-5, EMA 0.99 and an epoch-1 peak. w_ft is flat ±0.00025 and its sign flips with folds (ROUND5 §2c). Label smoothing and KD are closed. Expected +0.0000..+0.0003 sits below the FT-level resolution (needs ≥ 2 FT seeds or z ≥ 2.5). | no (duplicate) | 0.03 / 1 |
| 23 | Causal-mask history encoder + NAL (4) | The "new diversity family" argument is already tested: F10 GRU is a causal recurrent encoder, passes gates a+b, and adding it to the pool did not help (holdout 0.32329 vs 0.32334; CURRENT_STATUS 04:05). Next-action supervision is already in the ranker through pre2s (a pretrained causal next-feed-open encoder). r7 block forms went 0/13. | no | 0.03 / 1 |
| 24 | Example-age input (3) | The YouTube rationale (Covington 2016) is popularity drift in ID-based candidate generation. [L] Per-day input means are flat. [L] Calendar context hurt cross-day transfer: weekend → weekday -0.0014 (~-2 SE; feat_cal_ §0.3). The inference value lies outside the training range (test is 7-10 days after train end), so its only effect would be an extrapolation. | no | 0.02 / 0 |
| 25 | GBDT score as NN input / base logit (3) | [L] GBDT ~0.37 vs NN ~0.40. LightGBM residual -0.0015..-0.011, nested clogit -0.0006, "no input carries information the NN lacks" (SOLUTIONS_REVIEW §2). The NN already sees the same 34 features. The encoder-score-as-input analogue (pre_final term) was -0.0008. | no | 0.03 / 0 |
| 13/14 | Shrink-and-perturb branches; data-bagged trunks (3) | [L] Same-recipe member count is saturated: F7x (7) val 0.40097 vs F7x+F7x70 (14) 0.40094, holdout 0.32332 → 0.32361. A cheaper decorrelated *member* buys ≤ +0.0003 holdout and nothing on val. Bagging costs ~-0.001 per member at 80% data. | no | 0.05 / 1 |
| 11 | Holdout feeds into the m02 pool (cond.) | Mechanism-consistent with m02. It depends on the pp02 elasticity readout (pending), and those members lose the holdout as selector (they select at the end anyway, sel_frac 0.96). | conditional | 0.15 / 6 (if pp02 PASS) |
| 12 | R-Drop / AWP late regularisers (cond.) | Conditional on d20 ≥ d0 (pending). SAM in fta was below MDE, and AWP doubles the cooldown cost. | conditional | 0.05 / 2 |
| 20 | K3: relabel with the val-FT teacher (cond.) | The val-FT teacher contaminates val evaluation (needs a 2-fold split). Born-again KD was n.s. and raised correlation. **Fix:** use a *pool-trained* reference (exploration feeds only, holdout excluded) as teacher. It is then clean and shares the model built for §3.1. | only in the fixed form | 0.08 / 7 (piggyback on §3.1) |

### 1.1 Items that are already done under another name (grep evidence)
- **#16** = FT EMA + output-space interpolation (w_ft) [L ROUND5 §2c, r5_wft.md: "Do not reopen w_ft"].
- **#23** = F10 GRU family (causal recurrent history encoder) + pre2s (next-feed-open pretraining) [L CURRENT_STATUS 04:05; pretrain_report TL;DR].
- **#21 information content** = `pre_v3A` `idmax32` / `pre2_*` ID part [L err_SYNTHESIS:183; pretrain_report TL;DR "the incremental signal is the ID part"].
- **#21 cold handling** = agedrop [L ROUND5 §2a].
- **#24** ⊂ calendar context `cal_dow` / `cal_mod` / `cal_payday` (closed) [L feat_cal_.md].
- **#25** ⊂ stacker / residual / encoder-term family [L SOLUTIONS_REVIEW §2; COVERAGE_MAP "pre_final / encoder term -0.0008"].
- **#9** ⊂ coverage-weighted val and segment weights (closed) [L sol_advval ADV-9; review_shift SHIFT-4].
- **#13 / #14** ⊂ "same-recipe extra members" (F8 rejected, F7x70 tie on val) [L SOLUTIONS_REVIEW §3; CURRENT_STATUS 13:55].
- **Item mean production position** (often proposed as "the production ranker's opinion") = `tte_it_pos` / `it_tr_pos`: null [L featrev_inventory:621, :680].

---------------------------------------------------------------------------------------------------------------------

## 2. What the sweep missed: the three largest gaps

### 2.1 Learnability-based data selection in the cooldown (RHO-LOSS / JEST), the scalable sample channel of AutoDebias: NEW

**Why it is the natural next step.** m02 proved that the *composition of the cooldown data* moves the mean function: +0.00117 val, +0.00134 holdout, PASS-CD on 7 roots [L]. Every follow-up so far is a hand-picked heuristic: dose (m0/m02/pp02), recency (#8), cold share (#9). The literature has a principled, cheap selector for exactly this situation: a large noisy/biased source plus a small clean target-distribution set.
- [E] **RHO-LOSS** (Mindermann et al., ICML 2022, arXiv 2206.07137):
  - Method: select points with high *reducible holdout loss* = current-model loss − loss of a model trained on clean holdout data.
  - Result: on web-scraped, label-noisy Clothing-1M it "trains in 18x fewer steps and reaches 2% higher final accuracy than uniform data shuffling".
  - [K] The irreducible-loss model can be small and is computed once.
- [E] **JEST** (Evans et al., NeurIPS 2024, arXiv 2406.17711):
  - Method: "learnability" = learner loss − reference loss, where the reference is trained on a small curated set.
  - Result: up to 13x fewer iterations and 10x less compute.
  - The authors stress that steering selection toward the curated distribution through the reference model is essential.
- [E via deep_position_bias_random] **AutoDebias** (SIGIR'21) meta-learns per-sample weights on biased data from a small *uniform* set: Yahoo!R3 NDCG@5 0.645 vs 0.550 biased. Our report parked it as "bi-level, too heavy at 247M rows". RHO/JEST is its first-order, one-pass approximation.

**Mapping.**
- The "clean holdout" is our exploration pool: ~134K random-like train feeds, with the 30K holdout already excluded.
- The "noisy source" is the 87% production feeds. Their labels are position-biased: target rate 11.8% at pos 0 vs 1.3% at pos 100 [L eda_labels_position].
- A pool-trained reference cannot explain position-driven positives, so they get high reference loss and are *deprioritised*. Feeds the current model already fits are also deprioritised.
- This attacks the bias in the 98% non-pool part of the cooldown, which m02 leaves untouched.

**Cheapest form (static, no online overhead).**
1. Train a reference ranker on the pool only. Use the same nn/ ranker for 3-4 epochs, pool feeds only, holdout and val untouched. Report its val; expect ~0.37-0.38 [O].
2. Run one GPU inference pass over the production feeds with (a) the reference and (b) the @77500 plateau snapshot the cooldown starts from. Compute per-feed excess = L_snap − L_ref, using the listwise CE per feed, normalised by log M.
3. In the cooldown, draw production batches with P ∝ rank-weight(excess) at keep-fraction 0.7, or a soft weight clipped to [0.5, 2]. Keep the m02 pool share fixed at 0.02.
   - The dose must stay inside the m02 regime: 0.7 keep → ~2.9 passes over kept feeds vs 2.0 now; do not go below 0.5.
4. Run it as one arm in the existing cd_lane harness: roots 101-104 vs m02, then 105-107 if stage 1 passes. Readout is PASS-CD plus the pmix and stale-val vetoes.

**Pre-flight CPU checks (kill before any cooldown).**
- (a) Spearman(excess, L_snap) < 0.9. If it is higher, this is just hard-example mining, and focal loss is already closed.
- (b) Kept feeds must show a lower share of positives at production positions 0-2 than dropped feeds. That confirms the debiasing mechanism.
- (c) Reference val NDCG ≥ 0.36.

**Expected effect** [O]: member +0.0005..+0.0015 if it works, P ~0.2-0.25; pool about half of that.
**Cost**: reference ~1-1.5 GPU-h, scoring ~0.5 GPU-h, stage 1 4 × ~1 GPU-h.
**Piggyback:** the same reference is the *clean* teacher for the fixed K3 (#20): λ = 0.2 soft-label mixing on production feeds in the cooldown. That arm does not need the 2-fold val split.

### 2.2 Selection: public as a third estimator, with winner's-curse shrinkage: REFINES #1/#2

- [E] Roelofs et al., "A Meta-Analysis of Overfitting in Machine Learning" (NeurIPS 2019): across ~120 Kaggle competitions, public→private rank changes show little evidence of substantial adaptive overfitting. The holdout method was robust across domains, model classes and analysts.
  - Transfer to us: our public is a *weekend* slice of ~8.6K feeds. It is noisy (paired SE 0.0005-0.0007), but it is genuinely held out. The val set (14K) is the reused one: it is used both for FT training and for many decide-rules, two of them disclosed as non-blind.
  - So the larger adaptive-overfitting risk is on **val**, not on public. That is the reverse of how the current rule treats them: public enters nothing, val enters fully.
- [L] Numbers (nn_x/pubdiag/build/score_out.txt, noise/noise_out.txt), deltas vs PF7x:

  | cand | val (SE) | holdout | public (paired SE) | val shrunk for selection* | precision-weighted val'+public |
  |---|---|---|---|---|---|
  | G3 | +0.00058 (0.00039) | +0.00020 (best) | +0.00010 (0.00051) | +0.00058 (not val-selected) | **+0.00040** |
  | M2 | +0.00120 (0.00047) | +0.00002 | -0.00029 (0.00061) | ~+0.00080 | **+0.00039** |

  *M2 was the val-best of ~6 pre-declared candidates (decide4), so its val is shrunk by ~0.8 SE (expected max of ~6 correlated normals at rho ~0.7) [O]. Holdout is left out of the weighting because it selected checkpoints.
- **Sensitivity.** With public down-weighted ×0.5 (weekend mismatch), G3 is +0.00047 and M2 +0.00055. That is still a tie, well inside one private SE.
- **Reading.** G3 and M2 tie at about +0.0004 over PF7x, about 1 private SE. G3 is the only candidate non-negative on val, holdout *and* public, and it has more families. This agrees with the pre-declared S-rule (S: G3 +0.00025, M2 +0.00027; < 0.0001 apart → "prefer lower re-cooled share" → G3).
- **Recommendation** [O]:
  - Keep rule.json as declared for today. For the Dec 9 freeze, pre-declare a combined estimator of the form (val shrunk by the number of looks) ⊕ public, precision-weighted, with public down-weighted ×0.5 for the weekend/weekday mismatch.
  - Use the S-rule's min-over-λ as the robustness bound.
  - The second pick is chosen to maximise E[max(A, B)], i.e. the eligible candidate *least correlated* with pick 1 on test (e.g. Pall or HG vs a G3-type pick 1), not the second-best by S.
  - With private paired SD of the difference ~0.0004, the hedge is worth ~+0.0001..+0.0002 in expectation at zero cost.

### 2.3 History-format parity in the cooldown (INTEGRITY-1): NEW as an NN arm

- [L] review_integrity INTEGRITY-1: train histories are built with a deeper per-type cap than val/test.
  - Feeds with ≥ 120 views: train 26.8% vs val 0.37%.
  - Type-1 share at hpos 192-255: train 0.404 vs val 0.114.
  - Type-1 share by day: 0.515-0.522 on every train day vs 0.438-0.443 val/test.
  - **The random-like pool behaves like train (24.1% over the cap)**, so the m02 cooldown fits a history format the test never shows.
- Status: it was only tested as full-train truncation in LightGBM (-0.0014 ± 0.0008) and as inference-time truncation of a trained member (~0 val, -0.0006 holdout; REVIEW_SYNTHESIS:85). It was **never tested as a cooldown-only parity in the NN.** The cooldown-only form is what m02 taught us matters: the final low-LR phase sets the function the test sees. The deep positions also feed retr16 retrieval and the history attention, which the LightGBM aggregates do not model.
- Arm: in cooldown batches only, both production and pool, cap per-type event counts at the val/test levels (type 1 ≈ 110, type 4 ≈ 117, types 6/8/10 ≈ 41-44), keeping the newest events, then re-pack. Apply the same truncation to the holdout used for selection (review_integrity:104).
- Prior: low-moderate [O], P ~0.1. Cost: one more arm in the same cd_lane wave (~4 GPU-h stage 1).
- **Judge on val and stale-val.** The untruncated holdout is train-format, so it is the wrong judge for this arm.

### 2.4 Searched, nothing to add
- **FlightRank 2025 (Kaggle "Aeroclub RecSys Cup")**: the structurally closest 2025 Kaggle task (one search session = a group of candidate flights, one chosen).
  - Writeups are not retrievable (Kaggle client-side rendering).
  - The public notebooks are XGBoost/LightGBM/CatBoost rankers [S kaggle.com/code/ka1242/xgboost-ranker-with-polars, stpeteishii notebooks].
  - A top-15/578 repo reports DCNv2 + rank ensembling [S].
  - Nothing beyond our closed GBDT/rank-fusion items.
- **Kaggle solutions corpus** (github.com/TaeyanG4/kaggle-top-solutions-and-strategies, 7,237 writeup summaries, 2010-2026; CPU grep): searched for recency or time-decay weighting.
  - Only non-recsys cases matched (sports, game-agent behaviour cloning, energy).
  - OTTO 7th uses recency-weighted *co-visitation*, i.e. statistics, not sample weighting.
  - So there is no recsys-competition precedent for recency *sample* weighting, which supports the #8 downgrade.
- **BELKA 2024** (unseen building blocks in test): its cold shift is chemistry-specific (molecular graph generalisation). No ranking analogue.

---------------------------------------------------------------------------------------------------------------------

## 3. Corrected priority list (top 8)

Rules for all GPU arms:
- The existing cd_lane harness, stage 1 on roots 101-104 vs **m02** (not c0), then stage 2 on 105-107.
- PASS-CD as in R6_PLAN §4.1, plus the pmix veto (-1 SE) and the stale-val veto (-1 SE).
- Arms 3-5 can run as one wave: 3 arms × 4 roots ≈ 12-16 GPU-h.

| rank | item | experiment | kill criterion (one line) | cost |
|---|---|---|---|---|
| 1 | Final-pick protocol (#1 + #2 + §2.2) | Freeze today's rule.json outcome (G3-type pick 1 by the S tie-break). For Dec 9, pre-declare the combined val'(shrunk) ⊕ 0.5·public estimator and an E[max]-hedge pick 2 (least test-correlated eligible pool) | If the combined estimator and S disagree by < 1 private SE (0.0004), pick by S and stop refining the rule | 0 GPU, ~1 h CPU |
| 2 | Stale-val + pmix veto replay (#5) on every final candidate's TO half | GPU replay of the stale-val features through the candidate pools; report the delta vs the reference pool | A candidate whose stale-val OR pmix delta vs reference is < -1 SE is not eligible for pick 1 | ~0.5 GPU-h |
| 3 | **Learnability-selected cooldown (§2.1, NEW)** | Pool-only reference ranker → static excess-loss sampler over production feeds (keep 0.7) + m02 pool share; cd_lane arm vs m02 | Pre-flight: Spearman(excess, L_snap) ≥ 0.9, OR no drop in top-position positive share among kept feeds, OR ref val < 0.36 → do not launch. Stage 1: member val < 0 OR holdout < 0 vs m02 → stop | ~1.5 GPU-h prep + ~4 GPU-h stage 1 (+3 stage 2) |
| 4 | Recency cooldown, ONE arm (#8 downgraded) | q = 0.33 of non-pool cooldown batches from Sep 17-19 (holdout excluded), m02 pool unchanged; no tau sweep | Stage 1 val < 0 OR holdout < 0 vs m02, OR Thu gain ≤ Tue gain (the recency mechanism predicts the staler day gains more) → close #8 for good | ~4 GPU-h |
| 5 | History-format parity cooldown (§2.3, NEW) | Per-type val/test caps on cooldown batch histories (newest kept); holdout evaluated with the same truncation | Stage 1 val < 0 OR stale-val < 0 vs m02 → stop (the untruncated holdout is not a judge here) | ~4 GPU-h |
| 6 | Holdout feeds into the m02 pool (#11), conditional | Only if pp02 shows positive pool-size elasticity at stage 1 | pp02 stage 1 val < 0 or holdout < 0 → never run; if run, stage 1 val < 0 vs m02 → stop | ~4 GPU-h |
| 7 | Clean-teacher K3 (#20 fixed): pool-trained reference as soft-label teacher on production feeds in the cooldown (λ 0.2) | Reuses item 3's reference; one arm in the same harness | Run only if item 3's reference passes pre-flight (c); stage 1 val < 0 OR rho to the m02 member > 0.98 with \|Δval\| < 0.0003 → stop | ~4 GPU-h |
| 8 | Frozen ID channel (#21), **gated** | CPU gate first: in the nested clogit / FT-extra harness, add ID-space retrieval statistics (top-16 mean / max cos to history ID vectors, warm candidates only) beyond pre2s and idmax32 | CPU gate Δval < +0.0003 (or z < 1.5) → do not build the NN channel; if built, 2-ep paired screen val < +1 SE OR holdout < 0 → close | CPU ~2 h; then ~1 day plumbing + ~1.5 GPU-h |

Dropped from the list (the reasons are in §1): #9, #13, #14, #16, #22, #23, #24, #25, #12 (wait for d20), #15, #18, #26-#41.

---------------------------------------------------------------------------------------------------------------------

## 4. Notes for the lead

- **Judge every new cooldown arm against m02 on the same roots, never against the original F7 members.** The re-cool harness itself adds val (+0.0009) that the holdout does not confirm [L match §4; pubdiag p5_recool]. This applies to items 3-7.
- **Expect ≤ 1 of items 3-7 to pass PASS-CD** [O]. If one does, its pool gain is ~+0.0003-0.0006, i.e. 1-1.5 private SE. That still matters at the top of a 0.3928 vs 0.3935 leaderboard.
- **The highest-certainty value left is selection discipline** (items 1-2), which protects ~0.0005-0.001. Do not open new architecture families. r7 went 0/13, GRU/DIN/d192 add no pool value, and member count is saturated.
- **One extra CPU diagnostic.** match.md listed "regression of per-feed member gain on candidate age" as the only missing piece of #3. Run it on the M2-vs-PF7x per-feed deltas; it costs minutes. If the m02 gain falls with candidate age, the S-rule's λ should lean toward 1 for m02-heavy pools.

---------------------------------------------------------------------------------------------------------------------

## 5. Sources (fetched or searched this session unless marked)

External:
- Mindermann et al., "Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt" (RHO-LOSS), ICML 2022: https://arxiv.org/abs/2206.07137 ; code https://github.com/OATML/RHO-Loss [E: Clothing-1M 18x fewer steps, +2% acc]
- Evans et al., "Data curation via joint example selection further accelerates multimodal learning" (JEST), NeurIPS 2024: https://arxiv.org/abs/2406.17711 [E: up to 13x fewer iterations, 10x less compute; reference-model steering essential]
- Abe, Buchanan, Pleiss, Cunningham, "Pathologies of Predictive Diversity in Deep Ensembles", TMLR 2024: https://arxiv.org/abs/2302.00704 [E: diversity-encouraging regularisers hurt high-capacity ensembles, ~600 ensembles]
- Roelofs et al., "A Meta-Analysis of Overfitting in Machine Learning", NeurIPS 2019: https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning [E: ~120 Kaggle competitions, little adaptive overfitting]
- AutoDebias (Chen et al., SIGIR 2021): https://arxiv.org/abs/2105.04170 ; KDCRec (Liu et al., TKDE 2023): https://dgliu.github.io/pubs/TKDE_2023_KDCRec.pdf [E via reports/deep_position_bias_random.md]
- TIF time-interval weighting, arXiv 2311.16878; GPSD, arXiv 2506.03699; PinFM; Covington et al., RecSys 2016 [E via research_1112/industry.md]
- Kaggle top-solutions corpus (7,237 writeup summaries): https://github.com/TaeyanG4/kaggle-top-solutions-and-strategies (downloaded to the scratchpad, grep only) [S]; OTTO 7th place, recency-weighted co-visitation: https://www.kaggle.com/competitions/otto-recommender-system/discussion/383769 [S]
- FlightRank 2025 notebooks: https://www.kaggle.com/code/ka1242/xgboost-ranker-with-polars , https://www.kaggle.com/code/stpeteishii/flightrank-catboost-ranker-explanation [S]
- BELKA 1st place pointer: https://nips.cc/virtual/2024/109287 (Kaggle discussion 519020; not read) [S]

Internal (all under <repo>):
- reports/CURRENT_STATUS.md (04:05, 11:31, 12:43, 13:55, 14:04)
- reports/COVERAGE_MAP.md, reports/SOLUTIONS_REVIEW.md §2-3, reports/ROUND5.md §2a/§2c, reports/r5_wft.md
- reports/err_SYNTHESIS.md:183, reports/eda_shift_overlap.md §2.2/§4.2, reports/review_shift.md (SHIFT-1, SHIFT-4)
- reports/review_integrity.md (INTEGRITY-1), reports/REVIEW_SYNTHESIS.md:85, reports/feat_cal_.md §0, reports/deep_position_bias_random.md §5-6
- reports/sol_cpu_CPU2_stale_replay_pre2s_and_arms.md, reports/featrev_inventory.md:621
- nn_x/pubdiag/noise/noise_out.txt, nn_x/pubdiag/noise/private_se.txt, nn_x/pubdiag/build/score_out.txt, nn_x/pubdiag/build/rule.json, nn_x/pubdiag/overfit/p3_divers.txt
