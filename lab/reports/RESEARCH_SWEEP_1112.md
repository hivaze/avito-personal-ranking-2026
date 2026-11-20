# RESEARCH SWEEP 2026-11-12: recent competition and industry tricks, matched against our pipeline

Written 2026-11-12 by the synthesis step of workflow wp7s449jt. Research only: no GPU was used, nothing running was touched, and no existing file was edited.

Inputs:
- five research angles in `reports/research_1112/`:
  - `shift.md`: winners under distribution shift and public/private shake-ups;
  - `recsys.md`: 2025-26 recsys and CTR competitions;
  - `industry.md`: industrial ranking papers;
  - `nntricks.md`: Kaggle neural-net training tricks;
  - `tools.md`: GBDT and tabular foundation models;
- `match.md`: the 44 deduplicated ideas matched against COVERAGE_MAP, EXPERIMENT_REGISTRY, CURRENT_STATUS and the sol_* / review_* reports;
- `critique.md`: an adversarial review that downgraded most of the match ranking and added three gaps.

This file is the decision summary. Full arguments and per-source detail are in the angle reports.

Evidence tags:
- **[E]**: measured in the cited paper or writeup.
- **[E-x]**: a number from a search extract or secondary report of a Kaggle writeup. Kaggle pages are client-rendered and block automated access, so verify these before quoting them as decisive.
- **[L]**: measured in this project (file cited).
- **[O]**: opinion or estimate.

Scales:
- "Pool gain" means private NDCG of the final member-equal pool. Pool gains are ~0.4-0.6x member gains (val→holdout transfer ratio 0.38; `nn_x/pubdiag/overfit/p1_decomp.txt`).
- Private paired SE between strong pools is 0.0003-0.0005 (`nn_x/pubdiag/noise/private_se.txt`).
- Member MDE is ~0.001 on val.

------------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

**New intuitions**
1. **The late training phase is our proven lever, and the data choice inside it is the knob.**
   - m02 (2% exploration-pool replay in the WSD cooldown) gave member +0.00117 val and +0.00134 holdout over 7 roots [L].
   - It is the same mechanism as end-of-training domain upsampling in LLMs (Blakeney 2024: +6.9 MMLU at 7B [E]; Llama 3 annealing) and as the pretrain→target-subset fine-tune of the AES 2.0 winners (+0.015 QWK [E-x]).
   - Architecture screens went 0/13 (R7).
   - So every new GPU item below is a cooldown-data arm in the existing cd_lane harness, judged against m02 on the same roots.
2. **The m02 public under-transfer is best explained by noise plus val selection, not by shift.**
   - Public vs val expectations: chi2(3) 4.77, p 0.19 [L `nn_x/pubdiag/noise/noise_out.txt`].
   - The m02 gain does not differ between weekend and weekday holdout days, or between Tue and Thu val [L].
   - Pool agreement is identical on val, public and private (mean rho 0.9312 / 0.9308 / 0.9309). In agreement-on-the-line terms the slope is ~1, so no split-specific shrinkage is predicted [L `overfit/p3_divers.txt`].
   - The public order G3 > PF7x > M2 follows the holdout, not val.
   - The val-only excess comes from the re-cool procedure itself (+0.0009 val / −0.0005 holdout [L `overfit/p5_recool.txt`]). The CDSEED-813 test of this is running.
   - Do not drop m02. Discount val for re-cooled pools.
3. **Selection is the highest-certainty value left, and val is the over-reused set, not public.**
   - Winners of the biggest shake-ups trusted a test-like CV and averaged broadly (AES 2.0: 619th public → 1st private; CMI-PIU 1st).
   - Roelofs 2019 (~120 Kaggle competitions): adaptive overfitting of held-out leaderboards is small.
   - Our val trains the FT and has fed many decide rules. Public is held out but small and weekend-only.
   - Combining val (shrunk for the winner's curse), the holdout and half-weight public puts G3 ≈ M2 ≈ +0.0004 over PF7x. G3 is the only candidate that is non-negative on all three (§3).
4. **Information beats architecture, and the pool is correlation-limited.**
   - In the 2025-26 ablations (KDD Cup 2026 QueryFormer / UniDot / STAR, LinkedIn Feed SR [E]), each architecture mechanism is worth 1e-4..1e-3 AUC, while input/information channels are worth 1e-3..1e-2.
   - Our pool rho is 0.93 (n_eff ~1.07), and doubling same-recipe members moved val by −0.00003 [L].
   - These have no transferable precedent here: GBDT/TFM components, distillation, model merging, pseudo-labels, DRO/IRM, and recency *sample* weighting (no recsys case among 7,237 Kaggle writeup summaries).
   - Each surviving training arm is worth ≤ +0.0006 pool (≤ 1.5 private SE). Expect at most one of plan items P3-P7 to pass.

**Ranked plan**

Common rules for GPU arms: cd_lane harness; stage 1 on roots 101-104 vs **m02**, stage 2 on roots 105-107; PASS-CD (R6_PLAN §4.1) plus the pmix veto and the stale-val veto at −1 SE. P3-P5 fit in one wave (3 arms × 4 roots ≈ 12-16 GPU-h).

5. **P1 Final-pick protocol.** Cost: 0 GPU, ~1 h CPU. Protects ~0.0005-0.001 of selection error; the hedge adds +0.0001..+0.0002.
   - For Dec 9, pre-declare a precision-weighted estimator: val shrunk by the number of looks ⊕ 0.5·public.
   - Use the S-rule (S = min(V, V−D)) as the robustness bound.
   - Pick 2 = the eligible pool least test-correlated with pick 1 (maximises E[max]).
   - Kill: if the estimator and S differ by < 0.0004, pick by S and stop refining.
6. **P2 Stale-val + pmix veto replay** on each final candidate's train-only half. Cost: ~0.5 GPU-h. Protects 0..0.0005 private.
   - Rule: a stale-val or pmix delta < −1 SE vs the reference makes a candidate ineligible for pick 1.
7. **P3 Learnability-selected cooldown (RHO-LOSS / JEST; NEW).** Cost: ~1.5 GPU-h prep + ~4 stage 1 (+~3 stage 2). Expected: member +0.0005..+0.0015 at P ~0.2-0.25; pool about half.
   - Build: a pool-only reference ranker; sample production cooldown feeds by excess loss L_snap − L_ref (keep 0.7); m02 pool share fixed.
   - Kill pre-flight if Spearman(excess, L_snap) ≥ 0.9, OR kept feeds show no drop in top-position positive share, OR reference val < 0.36.
   - Stop if stage-1 val < 0 OR holdout < 0 vs m02.
8. **P4 Recency cooldown, one arm.** Cost: ~4 GPU-h. Expected: pool +0.0002..+0.0005 at P ~0.15.
   - Build: q = 0.33 of non-pool cooldown batches from Sep 17-19.
   - Kill: stage-1 val < 0 OR holdout < 0 vs m02, OR Thu gain ≤ Tue gain → close for good.
9. **P5 History-format parity cooldown (INTEGRITY-1; NEW).** Cost: ~4 GPU-h. Expected: pool 0..+0.0004 at P ~0.1 [O].
   - Build: cap per-type history counts of cooldown batches at the val/test levels, keeping the newest events.
   - Kill: stage-1 val < 0 OR stale-val < 0 vs m02. The untruncated holdout cannot judge this arm.
10. **P6 Holdout feeds into the m02 pool.** Cost: ~4 GPU-h stage 1. Expected: +0.0001..+0.0003.
    - Run only if pp02 shows positive pool-size elasticity; never run if pp02 stage-1 val < 0 or holdout < 0.
    - Kill: stage-1 val < 0 vs m02.
11. **P7 Clean-teacher K3.** Cost: ~4 GPU-h. Expected: 0..+0.0004.
    - Build: λ = 0.2 soft labels from the P3 reference on production cooldown feeds.
    - Run only if the reference passes pre-flight. Kill: stage-1 val < 0, OR rho to the m02 member > 0.98 with |Δval| < 0.0003.
12. **P8 Frozen pretrained item-ID channel (GPSD / PinFM), gated.** Cost: CPU gate ~2 h, then ~1 day plumbing + ~1.5 GPU-h screen (+~13 GPU-h family). Expected: +0.0003..+0.001 at P ~0.08-0.1.
    - Gate: add ID-space retrieval stats in the nested clogit harness. If Δval < +0.0003 or z < 1.5, do not build.
    - Kill: 2-epoch screen val < +1 SE OR holdout < 0 → close.

------------------------------------------------------------------------------------------------------------------------

## 1. Plan details

| rank | item | mechanism (source) | experiment | exp. gain (P) | cost | kill criterion |
|---|---|---|---|---|---|---|
| P1 | Final-pick protocol | Winner's curse in selection (Cawley & Talbot 2010); little adaptive overfit on held-out LBs (Roelofs 2019); breadth under shift (Ovadia 2019); "Trust CV ②" (rsakata 2025) | Keep today's decide5 (CURRENT_STATUS 15:22). For Dec 9, pre-declare: E = val shrunk by ~0.8 SE for a val-best-of-~6 ⊕ 0.5·public, precision-weighted, with the holdout as a guard only (it selected checkpoints); S = min(V, V−D) as the robustness bound; pick 2 = the eligible pool least correlated with pick 1 on test | protects 0.0005-0.001; hedge +0.0001-0.0002 | 0 GPU | E and S disagree by < 0.0004 → pick by S, stop refining |
| P2 | Stale-val + pmix vetoes | Gains measured on warm data vanish on colder data (BELKA generalisability study; UM MCTS / Eedi entity-held-out CV) | GPU replay of the stale-val features through every final candidate's TO half; report the delta vs the reference pool next to the pmix delta | protects 0..0.0005 | ~0.5 GPU-h | delta < −1 SE on either → not eligible for pick 1 |
| P3 | Learnability-selected cooldown | RHO-LOSS (Mindermann 2022), JEST (Evans 2024); first-order AutoDebias (SIGIR 2021). A pool-trained reference cannot explain position-driven production positives (target rate 11.8% at pos 0 vs 1.3% at pos 100 [L eda_labels_position]), so it down-weights them in the 98% non-pool part of the cooldown that m02 leaves untouched | (1) Reference ranker on pool feeds only, 3-4 epochs, holdout and val untouched. (2) One pass of the reference and the @77500 snapshot over production feeds: excess = L_snap − L_ref, using per-feed listwise CE / log M. (3) Cooldown draws production feeds by excess rank (keep 0.7, i.e. ~2.9 passes; never below 0.5), with m02 pool share 0.02 | member +0.0005..+0.0015 (0.2-0.25) | ~1.5 + 4 (+3) GPU-h | pre-flight: Spearman(excess, L_snap) ≥ 0.9 (just hard-example mining; focal loss is closed), no drop in pos-0..2 positive share among kept feeds, or ref val < 0.36 → do not launch; stage 1 val < 0 or holdout < 0 → stop |
| P4 | Recency cooldown, one arm | Domain upsampling at the end of training (Blakeney 2024); online learning on fresh data (Optiver / Enefit / Jane Street winners). The only real channel here is LUO-popularity parity: it_hist zero rate 17.5% (Sep 6) → 29.7% (Sep 19) vs 45.6% private | q = 0.33 of non-pool cooldown batches from Sep 17-19 (holdout excluded), m02 pool unchanged; no tau sweep | pool +0.0002..+0.0005 (0.15) | ~4 GPU-h | val < 0 or holdout < 0, or Thu gain ≤ Tue gain → close for good |
| P5 | History-format parity cooldown | review_integrity INTEGRITY-1: train histories use deeper per-type caps than val/test (type-1 share at hpos 192-255: train 0.404 vs val 0.114), and the m02 pool has train format too (24.1% over the cap) | Cap per-type history events in cooldown batches (type 1 ≈ 110, type 4 ≈ 117, types 6/8/10 ≈ 41-44; newest kept); evaluate the selection holdout with the same truncation | pool 0..+0.0004 (0.1) [O] | ~4 GPU-h | val < 0 or stale-val < 0 → stop |
| P6 | Holdout feeds into the m02 pool | Full-data refit after CV fixes the schedule (NVIDIA GM Playbook 2025). m02 members select at the end anyway (sel_frac 0.96; `overfit/p2_ckpt.txt`) | Add the 30K holdout feeds to the pool at a fixed 0.02 mix (+22% pool) | +0.0001..+0.0003 (0.15) | ~4 GPU-h | only if pp02 passes stage 1; then val < 0 → stop. These pools lose the H/D terms of S |
| P7 | Clean-teacher K3 | AES 2.0 1st: relabelling the old data with a new-data model gave +0.005 [E-x]; KDCRec label channel. The val-FT teacher form is rejected because it contaminates val | Soft-label mixing λ = 0.2 toward the P3 reference on production cooldown feeds | 0..+0.0004 (~0.08) | ~4 GPU-h | reference fails pre-flight → skip; val < 0, or rho to the m02 member > 0.98 with \|Δval\| < 0.0003 → stop |
| P8 | Frozen pretrained ID channel, gated | GPSD sparse transfer + freeze (AUC 0.6375 → 0.6838 [E]); PinFM fresh-item handling. Internal counter-evidence: idmax32 null (err_SYNTHESIS:183); ID randomisation = the failed agedrop | CPU gate: top-16 mean/max cosine to history ID vectors (warm candidates) on top of pre2s and idmax32 in the nested clogit / FT-extra harness; only then a frozen table with a learned no-ID token, 2-epoch paired screen | +0.0003..+0.001 (0.08-0.1) | CPU 2 h; then 1 day + 1.5 (+13) GPU-h | gate Δval < +0.0003 or z < 1.5 → stop; screen val < +1 SE or holdout < 0 → close |

Free CPU diagnostic (minutes, run with P1): regress the per-feed M2 − PF7x delta on candidate age and feed cold share. If the m02 gain falls with candidate age, lean the S-rule's λ toward 1 for m02-heavy pools, because private is the coldest split.

Scheduling notes:
- Judge every cooldown arm against **its own m02 re-cool on the same roots**, never against the original F7 members. The re-cool harness alone adds +0.0009 val that the holdout does not confirm (match.md §4).
- **Do not schedule** new architecture families (r7 0/13; GRU/DIN/d192 add no pool value) or extra same-recipe members (member count is saturated).

------------------------------------------------------------------------------------------------------------------------

## 2. Idea table

Priority:
- **P1-P8**: the ranked plan.
- **0-10**: value × feasibility before Dec 9, discounted for duplicated mechanisms. It is the critique's revision where one exists, otherwise the match score.
- **0**: closed.
- **done**: already in place.

### 2.1 Selection, diagnostics, policy

| idea | source | status for us (evidence file) | priority |
|---|---|---|---|
| Joint val+holdout pick with winner's-curse shrink; public as a precision-weighted third estimator (half weight) | Cawley & Talbot, JMLR 2010; Roelofs et al., NeurIPS 2019; rsakata deck 2025; AES 2.0 1st (flg, 2024) | partially done: S = min(V, V−D) in `nn_x/pubdiag/build/rule.json`; decide5 pre-declared (CURRENT_STATUS 15:22); public enters nothing yet | **P1** |
| Breadth over concentration; pick 2 = least test-correlated eligible pool | Ovadia et al., NeurIPS 2019; The Kaggle Book; Home Credit CRMS 9th (Dxfory, 2024); CMI-PIU 1st (Haupt, 2024) | partially: member-equal multi-family policy (COVERAGE_MAP; SOLUTIONS_REVIEW §4.1). Public/val transfer PF7x 0.9x, G 0.6x, M 0.2x (CURRENT_STATUS 14:04) | **P1** (merged) |
| Hedge shift-dependent knobs at the midpoint (keep c0/m02 and lr families mixed) | Home Credit CRMS 1st (yuuniee, 2024) [E-x] | adopted in spirit: Pall / PF7x / G3 mix recipes (CURRENT_STATUS §1) | **P1** (merged) |
| Private-coverage emulation (stale-val) + pmix with veto power | Leash BELKA 1st + generalisability study (2024/26); UM MCTS 1st (2024); CIBMTR 3rd (2025) | partially: stale-val is a report column (COVERAGE_MAP "Shift"); pmix veto adopted (CURRENT_STATUS 12:42; rule.json e5); not yet run on the m02 pools | **P2** |
| Regress the m02 per-feed gain on candidate age / cold share | match.md #3; critique.md §4 | missing piece of the slice diagnostics; caches in `nn_x/pubdiag/noise/` | free CPU (with P1) |
| DOW / late-day / hour / per-slice stability columns | Home Credit CRMS stability metric (2024); Optiver / Enefit / Jane Street calendar validation; CIBMTR 3rd emulated splits | done: `nn_x/pubdiag/noise/weekend_holdout.txt`, `val_day_split.txt`, `hour_reweight.txt`, `shift/p1_holdout_weekday.log`; late-holdout slice mandatory (CURRENT_STATUS §3) | done (INFO only) |
| Agreement-on-the-line from unlabelled test feeds | Baek et al., ICML 2022; Mehra et al. 2024 | answered: rho equal across splits (`nn_x/pubdiag/overfit/p3_divers.txt`); FT~TO Spearman 0.9847 on public and private (sol_advval ADV-7) | done |
| Public-likeness adversarial coordinate as a covariate | TabFSBench, ICML 2025; sol_advval | done: `nn_x/pubdiag/shift/p2_adv_weights.npz` used as Vp in rule.json | done |
| Time-forward holdout for checkpoint selection | Ji et al., TOIS (timeline leakage); Wild-Time Eval-Fix (NeurIPS 2022) | untried; argmax inflation +0.00055, of which +0.00034 replicates on val (`overfit/p2_ckpt.txt`); m02 selects at the end anyway | 2 |

### 2.2 Cooldown and training-data composition

| idea | source | status for us (evidence file) | priority |
|---|---|---|---|
| Learnability-selected cooldown (excess loss vs a pool-trained reference) | Mindermann et al., ICML 2022 (RHO-LOSS); Evans et al., NeurIPS 2024 (JEST); AutoDebias, SIGIR 2021 | untried, NEW. AutoDebias was parked as too heavy (deep_position_bias_random.md §5-6) | **P3** |
| Recency / fresh-day cooldown (q of non-pool batches from Sep 17-19) | Blakeney et al. 2024; Llama 3 (2024); TIF (arXiv 2311.16878); LinkedIn Feed SR (2026); Optiver 1st, Enefit 1st (hyd), Jane Street 2nd (2025) | partially: only a LightGBM proxy (+0.0006 ± 0.0009; review_ideas §1.6); never run in the NN; no within-train drift (eda_shift_overlap §2.2) | **P4** (5) |
| History-format parity in the cooldown | internal: review_integrity.md INTEGRITY-1 | partially: LightGBM full-train truncation −0.0014 ± 0.0008; inference-time truncation ~0 val / −0.0006 holdout (REVIEW_SYNTHESIS:85); never a cooldown-only NN arm | **P5** |
| Holdout feeds added to the m02 pool (full-data members) | NVIDIA Kaggle GM Playbook (2025) | untried; conditional on pp02 (running, CURRENT_STATUS 08:43) | **P6** (cond.) |
| Clean-teacher K3: pool-trained reference as soft teacher on production feeds | AES 2.0 1st relabelling (2024) [E-x]; KDCRec (TKDE 2023) | untried; K3 in deep_PLAYBOOK never run (nn_report:169); val-FT-teacher form rejected (val contamination) | **P7** |
| Cold-share-weighted or late-day pool batches | nntricks T1b; Blakeney 2024 | refuted: feed cold share is confounded with user-0 / short history (review_shift SHIFT-4); coldpos gains failed pmix (CURRENT_STATUS 12:43); a late-day pool would be an overdose (37K feeds) | 2 |
| Low-LR online continuation after the cooldown | Jane Street RTMDF 2nd (Yam), 8th (Grigoreva), 162nd (2025) | closed: stage 2 after WSD −0.0012 val / −0.0015 holdout, "re-warming undoes the cooldown" (nnx_VERDICT §2.3, §5) | 1 |
| Late regularisers: late dropout → R-Drop → AWP | Liu et al., ICML 2023; MAP 2025 3rd; USPPPM 2022 8th | partially: d0/d20 running (CURRENT_STATUS 08:43); R-Drop / AWP never; SAM in FT below MDE | 2 (only if d20 ≥ d0) |
| Mutual learning co-cool (DML+ / anti-distillation DML−) | Jigsaw ACR 2025 6th; KDD Cup 2026 UniDot; Google "Factory Floor" (2022) | untried; Abe et al. (TMLR 2024): induced diversity hurts high-capacity ensembles; gate (a) is spent first | 1 |
| Shrink-and-perturb plateau snapshots before branching | Ash & Adams, NeurIPS 2020 | untried; same-recipe member count saturated: F7x 7 vs 14 members val −0.00003 (CURRENT_STATUS 13:55) | 1 |
| Data-bagged trunks (80% user subsamples) | Zhu et al., CIKM 2020; BirdCLEF+ 2025 2nd; against: Lee et al. 2015, Nixon et al. | untried; data slope implies ~−0.001 per member (R6_PLAN §0.4) | 1 |

### 2.3 Fine-tune stage

| idea | source | status for us (evidence file) | priority |
|---|---|---|---|
| FT small-data tricks: layer-wise LR decay, head re-init, holdout feeds as extra FT data | Zhang et al., ICLR 2021; LMSYS 1st (2024); AES 2.0 (2024) | partially: `--add_random 12000` neutral (nn_report:101; sol_solvers:505); LLRD / re-init never (COVERAGE_MAP "FT knobs NEVER") | 3 |
| WiSE-FT weight-space interpolation TO↔FT | Wortsman et al., CVPR 2022 | closed by argument only (R6_PLAN §2; nnx_research_training_gen §9.3); output-space w_ft nested +0.00024, z 0.96 (ROUND5 §2c) | 2 |
| Soft-target val fine-tune toward the member's own TO softmax | AES 2.0 3rd (ho, 2024); 2nd (yao, 2024) | duplicate of FT EMA + the 0.5/0.5 output blend; w_ft flat ± 0.00025, sign flips with folds (ROUND5 §2c; r5_wft.md) | 1 |
| Fold soup for FT test inference | LMSYS 1st (sayoulala, 2024) | untried; fold models correlate 0.9974 (sol_ensemble.md:50) | 1 |

### 2.4 New member families: inputs, architecture, auxiliary losses

| idea | source | status for us (evidence file) | priority |
|---|---|---|---|
| Frozen pretrained item-ID channel (sparse transfer + freeze, candidate-ID randomisation, kNN-imputed IDs) | GPSD (Alibaba, arXiv 2506.03699, 2025); PinFM (Pinterest, 2507.12704, 2025); Netflix FM blog (2025) | untried; idmax32 null (err_SYNTHESIS:183); hashed ID residual −0.0022 (nn_report N4); agedrop FAIL (ROUND5 §2a); pre2 ID part −0.004/−0.005 zero-shot (pretrain_report TL;DR) | **P8** (gated) |
| Missing / zero indicators for every NaN→0 family + u_missing + H==0 | KDD Cup 2026 UniRec STAR (arXiv 2608.12986) | partially: has: flags for pre2_fe / prew_fe; FR-8 parked as hygiene (FEATURES_REVIEW item 5) | 3 (bundle with the next from-scratch preset) |
| Content-anchored norm-bounded item-ID delta with per-epoch reset | Let It Go? (arXiv 2507.19473, 2025); MEDA (Kuaishou, 2305.19531) | partially: N4 closed without anchoring or reset (nn_report) | 2 (fallback for P8) |
| FM-highway: per-layer learned dot products to the head | KDD Cup 2026 UniDot (runner-up) | untried; r7 block forms 0/13 (R7_ARCH_HPARAM.md) | 2 (diversity only) |
| Cross-attention query generation + multi-view target attention | KDD Cup 2026 1st QueryFormer | untried; overlaps the typed FAIL / a4u arms (`nn_x/r6/logs/r6_eval_screen_*.txt`) | 2 |
| Cross-feed InfoNCE auxiliary on positives | KDD Cup 2026 STAR; Tencent TAAC 2025 2nd/3rd | untried; pre2 encoders already carry a contrastive signal (pretrain_report) | 2 |
| Causal-mask history encoder + next-action aux loss | GenRank (Xiaohongshu, 2025); PinFM; TransAct V2 (Pinterest, 2025) | duplicate: the F10 GRU causal family added nothing (CURRENT_STATUS 04:05); next-action signal enters via pre2s (pretrain_report) | 1 |
| Action-conditioned history tokens (FiLM by event type) | Tencent TAAC 2025 1st | partially: r6 typed view FAIL (`r6_eval_screen_a4u_wd01_cset_typed_adrop0_a4w.txt`); a4u counts PROCEED | 1 |
| RoPE in history self-attention | LinkedIn Feed SR (arXiv 2602.12354, 2026) | untried; FoX null (nnx_VERDICT §5) | 1 |
| Muon+ | Muon+ (arXiv 2602.21545); KDD Cup 2026 1st | untried; Muon adopted (nnx_VERDICT §2.3) | 1 (free arm only) |
| Loss-prediction confidence head | LEAP ClimSim 2024 1st | untried; analogous aux heads null (nnx_VERDICT §5 propensity; nn_report N3) | 1 |
| Input-subset diversity families | Home Credit CRMS public 8th (2024) | partially: families already differ by inputs; dropping cvnd −0.0032 fails gate (a) (nn_report v3_cv) | 1 |
| Rare-ID hygiene in pre2's 40M-row ID table | KDD Cup 2026 STAR, Z Lab 10th | untried; pre2 ID part inert under stale replay (sol_cpu_CPU2) | 1 |
| PinFM-style milder age dropout on similarity inputs only | PinFM (2025) | variant of closed agedrop (ROUND5 §2a; r5_agedrop.md) | 1 |
| Residual-framing family (NN boosts a frozen clogit base) | Deotte, Playground Apr-2025 1st (NVIDIA) | untried; its trigger (GBDT base-logit arm) is dropped | 1 |
| Temporal DRO over item groups; V-REx / InvCF time-environment penalty | TDRO (AAAI 2024); InvCF (WWW 2023) | untried; Wild-Time / TabReD: no gain over ERM; segment weights closed (EXPERIMENT_REGISTRY) | 1 |
| AdaBN-style re-standardisation; TTT on test histories | Schneider et al., NeurIPS 2020; Akyürek et al. 2024; T2ARec 2025 | partially: TTA ≤ 0 (r6_tta.md); transductive pre2 +0.0004..+0.0009 (pretrain_report) | 1 |
| Example-age input (train day index, 0 at inference) | Covington et al., RecSys 2016 (YouTube) | ⊂ closed calendar context (feat_cal_.md §0.3); per-day input means flat (eda_shift_overlap §2.2) | 0 |
| Cross-fitted GBDT score as NN input / base logit (+ noise, leaf-SVD) | Li, Qin, Wang, Metzler, KDD 2019 (Google); ISIC 2024 1st; trivago RecSys 2019 | ⊂ "no residual signal" (SOLUTIONS_REVIEW §2); pre_final encoder term −0.0008 (COVERAGE_MAP) | 0 |

### 2.5 Closures confirmed by this sweep

| idea | source | status for us (evidence file) | priority |
|---|---|---|---|
| Importance / adversarial reweighting, DRO / IRM / CORAL, post-hoc cold boosts, LB-tuned constants | Byrd & Lipton, ICML 2019; Wild-Time 2022; TabReD (ICLR 2025); KDD Cup 2020 Debiasing 1st | closed: ESS 2,880 (sol_advval); learned cold offsets 0 (SOLUTIONS_REVIEW §2) | 0 |
| Fitted ensemble weights / hill climbing | Playground S5E5 / S3E24 (2023-25); BirdCLEF+ 2025 2nd | closed: Caruana optimism +0.0014 (sol_ensemble.md) | 0 |
| SWA / EMA / soups / plateau averaging / checkpoint ensembles in the pool | CSIRO Image2Biomass 2026 1st; RSNA Aneurysm 2025 1st; OLMo 2; DiWA | closed: plateau-averaged pool −0.0025 val / −0.0016 holdout (COVERAGE_MAP; r6_recap §5) | 0 |
| GBDT / TFM members, stackers, CatBoost, linear_tree, TFM feature generators, RDL backbones | Google KDD 2019; ShareChat FIRE'23; TabReD; TabPFN-3 (2026); RelBench rel-avito | closed: frontier law needs GBDT val ≥ 0.388 vs best 0.382 (SOLUTIONS_REVIEW §2); CatBoost closed in every role (`nn_x/cb26`; CURRENT_STATUS 14:11) | 0 |
| Probability / raw-logit / std-adjusted fusion | Deotte stacking post (NVIDIA, 2025) | closed: all within ± 0.0002 of the z-mean (sol_ensemble D/E; research_1112/tools.md) | 0 |
| Test pseudo-labels / noisy student | BirdCLEF+ 2025 1st/2nd/5th | closed: −0.0004 (COVERAGE_MAP) | 0 |
| Ensemble → single-student KD | LMSYS 1st (2024); WSDM Cup 2025 1st/3rd/6th | closed: born-again KD +0.0006 n.s., raises correlation (nnx_VERDICT §5) | 0 |
| Mixup / TTA | ASL Signs 2023 1st; Jigsaw ACR 2025 | closed (r6_tta.md; R6_PLAN §3) | 0 |
| HSTU / SiLU pointwise attention, contrastive SSL, LLM recommenders | LinkedIn Feed SR 2026; GenRank 2025; Zhou et al., WWW 2024 | closed: F11 DIN fail (EXPERIMENT_REGISTRY:43); LLM backbones (llm_large.md) | 0 |
| PDA popularity extrapolation | PD/PDA, SIGIR 2021 | closed: causal / lagged popularity (FEATURES_REVIEW.md) | 0 |
| Hyper-deep ensembles | Wenzel et al., NeurIPS 2020 | done in family form (sol_ensemble §2.5) | done |

------------------------------------------------------------------------------------------------------------------------

## 3. Selection under shift

### 3.1 Where we sit

rsakata's shake-up taxonomy (2025) calls our case **"Trust CV ②"**: a known public ≠ private shift, so make CV mimic private and treat public as a bug check.

| axis | val | public | private | implication |
|---|---|---|---|---|
| day type | Tue + Thu | 100% weekend (Sat, Sun morning) | 24% weekend (Sun pm, Mon, Tue) | val resembles private more than public does |
| days after train end (Sep 19) | 3 / 5 | 7 / 8 | 8 / 9 / 10 | no Tue→Thu gradient in pool gains [L val_day_split.txt] |
| item coverage | 65% | 56% | 53% | private is the coldest split → stale-val veto (P2) |
| cold-item share | 0.29 | 0.37 | 0.41 | same |
| hour mix | 25% evening | 56% morning | 37% evening | pool gains do not depend on hour [L hour_reweight.txt] |
| size | 14.4K feeds | 8.6K (paired SE vs earlier subs ~0.0012; between strong pools 0.0005-0.0006) | 20.0K (paired SE 0.0003-0.0005) | Roelofs: < 10K examples is not decision-grade; never select on public deltas < 0.003 alone |

### 3.2 What top competitors did when CV and LB disagreed

| competition (place, author, year) | situation | decision | outcome | lesson for us |
|---|---|---|---|---|
| AES 2.0 (1st, flg, 2024) | train = old + new essays; test = new style; public misleading | Built CV on the test-like subset; two-stage training; plain average of 4 models; "trust CV (and LB a bit)" | 619th public → 1st private; 4th place rose from 1277th [E-x] | Validate on the test-like slice; do not chase public |
| AES 2.0 (3rd, ho, 2024) | same | Stage 2 focused on the target subset, with soft labels | public 0.826 → 0.826, **private 0.828 → 0.838** [E] | A real gain can be invisible on a non-representative public |
| Enefit Prosumers (1st, hyd, 2024) | private = 8 later months | Kept online retraining although it "didn't help much in public LB … crucial in private" | 1st [E-x] | Trust the mechanism, not a small public delta |
| Home Credit CRMS (1st, yuuniee, 2024) | a shift-dependent knob; private optimum uncertain (0.02-0.04) | Chose the neutral middle value, not the public-optimal 0.03 extreme | 1st [E-x] | Hedge knobs whose best value depends on unseen conditions: keep m02 and c0 families mixed |
| Home Credit CRMS (public 8th → private 253rd, Grigoreva, 2024) | strong public rank | Six independent feature pipelines for diversity | private 253rd (writeup title) | A top public rank gives no protection in a shift competition |
| CIBMTR (3rd, 2025) | public/private composition differ | Emulated public and private splits; public sat at ~75th percentile of the emulated distribution; submitted a private-oriented second pick | private-oriented pick scored better on private [E, qualitative] | Emulate both splits (our pmix / stale-val) before reading a public gap as overfit |
| Leash BELKA (1st + generalisability study, 2024/26) | private dominated by unseen building blocks | Larger models "improved public but failed to generalize … with the private and more OOD set" [E-x] | n/a | Warm-data gains can vanish on the colder private → stale-val veto |
| CMI-PIU (1st, Haupt, 2024) | tiny noisy test, huge shake-up | 10-20x repeated K-fold, voted ensemble, "focus on robustness" | 1st (self-described luck) [E-x] | Reduce selection variance; average broadly |
| BirdCLEF+ 2025 (2nd, CEUR paper) | public vs OOF disagreement on blend weights | Compared OOF-optimised vs public-selected blends | OOF blend private 0.929 vs public-selected 0.928 [E] | Do not select blends on public |
| Playground S5E5 (2025) | many OOFs | Pruned the ensemble by public LB | public #3 → private #40 [E-x] | Same |
| Roelofs et al. (NeurIPS 2019, ~120 Kaggle comps) | meta-analysis | Public − private gaps are centred at zero; outliers come from non-i.i.d. or tiny test sets | [E] | Little adaptive overfit of held-out LBs; the reused set is the danger, and for us that is val |
| Cawley & Talbot (JMLR 2010) | selection bias | Overfitting the selection criterion costs as much as algorithm differences; the criterion's variance matters as much as its bias | [E] | Shrink the val-best; prefer lower-variance joint criteria |
| The Kaggle Book (validation chapter) | final 2 picks | One LB-best + one CV-best; if LB is untrusted, two strong-CV but very different models | [O, practitioner] | Pick 2 = maximally different eligible pool |

### 3.3 Today's disagreement (m02-heavy pools: val +0.0019, public +0.0004)

| hypothesis | test | result | verdict |
|---|---|---|---|
| Weekend shift | m02 vs c0 on weekend vs weekday holdout days | diff −0.00012 ± 0.00048 (member) [L shift.md §1.1]; M2 − PF7x weekend − weekday +0.00104 ± 0.00058 [L weekend_holdout.txt] | rejected (if anything M2 is better on weekends) |
| Staleness | val Thu − Tue pool gain | +0.00034 ± 0.00094 [L val_day_split.txt] | no gradient |
| Split-specific behaviour change | label-free cold promotion and agreement on val / public / private | equal on all three [L p5_recool.txt, p3_divers.txt] | rejected |
| Private-like coverage | stale-features replay | M2 − Cd shrinks by 0.0006 ± 0.0005, all of it from the F7 lineage; the m02 part is unchanged [L CURRENT_STATUS log] | within noise |
| Noise + val selection | joint chi2 of public vs val expectations | chi2(3) 4.77, p 0.19; M2 − PF7x residual z −1.94 [L noise_out.txt] | **consistent** |
| Re-cool procedure inflates val | F7b c0 − F7 | +0.0009 val, −0.0005 holdout [L p5_recool.txt]; CDSEED-813 test running (CURRENT_STATUS 15:22) | likely; pending |

### 3.4 Protocol for the Dec 9 picks

1. **Pre-declare** candidates and the rule before any new number exists. decide5 does this; decide4 was disclosed as non-blind.
2. **Eligibility** (all must hold):
   - holdout ≥ −1 SE vs the reference pool;
   - no val slice (M<50, M≥50, Tue, Thu, coldpos, warmpos) below −2 SE;
   - stale-val and pmix deltas ≥ −1 SE (P2);
   - presub_gate PASS.
3. **Estimator:** E = val shrunk for the winner's curse (~0.8 SE for the best of ~6 correlated candidates at rho ~0.7, scaled with the number of looks) ⊕ 0.5·public, precision-weighted.
   - Weight public by half for the weekend/weekday mismatch.
   - The holdout is a guard, not a term, because it selected checkpoints (inflation ~0.0002 per member, equal across families).
   - Apply the re-cool D-discount to re-cooled pools unless gap813 ≤ +0.0002.
4. **Robustness bound:** S = min(V, V − D). If E and S disagree by < 0.0004 (one private SE), pick by S.
5. **Pick 1** = the best eligible pool by the rule above. Prefer broad multi-family pools; on ties, prefer the lower re-cooled share.
6. **Pick 2** = the eligible pool with the lowest test-set correlation to pick 1, provided its E is within one private SE. This maximises E[max(A, B)] and is worth ~+0.0001-0.0002 at no cost. It is **not** the second-best by S.
7. **Never:**
   - choose on public deltas < 0.003 alone;
   - fit ensemble weights on val or public;
   - tune constants on public.

Worked example with today's numbers, deltas vs PF7x (critique.md §2.2; `nn_x/pubdiag/build/score_out.txt`):

| cand | val (SE) | holdout | public (paired SE) | val shrunk | E (val' ⊕ public) | E with public ×0.5 | S |
|---|---|---|---|---|---|---|---|
| G3 | +0.00058 (0.00039) | +0.00020 | +0.00010 (0.00051) | +0.00058 (not val-selected) | +0.00040 | +0.00047 | +0.00025 |
| M2 | +0.00120 (0.00047) | +0.00002 | −0.00029 (0.00061) | ~+0.00080 | +0.00039 | +0.00055 | +0.00027 |

Reading:
- The two candidates tie well inside one private SE.
- G3 is the only one that is non-negative on val, holdout and public, and it has more families.
- The S-rule's tie-break also gives G3.

------------------------------------------------------------------------------------------------------------------------

## 4. Already covered

Each line is an idea from the sweep that we already did or closed (file in parentheses).

- Two-stage training, big source then target-like data (AES 2.0 design): train-only → 5-fold val FT, 0.5/0.5 blend (CURRENT_STATUS §2).
- End-of-training target-domain upsampling: m02 exploration replay in the WSD cooldown, adopted (COVERAGE_MAP; `nn_x/r6/logs/r6_eval_cd_stage2_m02.txt`).
- WSD cooldown sets the mean function: +0.0031 on a 7-seed ensemble (r6_recap §5).
- Muon-family optimizer vs AdamW: adopted (nnx_VERDICT §2.3); external Z Lab 10th agrees.
- Multi-seed validation and ≥ 2-seed acceptance rules (CURRENT_STATUS §3).
- Member-equal multi-family ensemble, i.e. hyper-deep ensembles in family form (sol_ensemble §2.5; COVERAGE_MAP).
- Joint val + holdout selection rule S (`nn_x/pubdiag/build/rule.json`).
- pmix veto (CURRENT_STATUS 12:42).
- Weekend / weekday / hour / late-day slices (`nn_x/pubdiag/noise/*.txt`; CURRENT_STATUS §3).
- Agreement-on-the-line answered: rho equal across splits (`nn_x/pubdiag/overfit/p3_divers.txt`).
- Public-likeness adversarial coordinate (sol_advval; `nn_x/pubdiag/shift/p2_adv_weights.npz`).
- Post-cooldown online continuation (Jane Street style): closed (nnx_VERDICT §2.3, §5).
- Soft-target FT ≈ FT EMA + output blend; w_ft flat (ROUND5 §2c; r5_wft.md).
- Refit on full val: worse (r5_refit.md).
- WiSE-FT: closed by argument (R6_PLAN §2; nnx_research_training_gen §9.3).
- Causal history encoder: F10 GRU family, no pool value (CURRENT_STATUS 04:05; EXPERIMENT_REGISTRY).
- Next-action / next-feed-open pretraining: pre2s (pretrain_report.md).
- Example age ⊂ calendar context: closed (feat_cal_.md).
- Age-dependent input dropout (PinFM-like): FAIL (ROUND5 §2a; r5_agedrop.md).
- ID-space history similarity idmax32: null (err_SYNTHESIS.md:183).
- Hashed item-ID residual: −0.0022 (nn_report.md N4).
- Typed / action-conditioned history keys: FAIL (`nn_x/r6/logs/r6_eval_screen_a4u_wd01_cset_typed_adrop0_a4w.txt`).
- Block forms (norms, SwiGLU, hyper-connections, diff-attention, etc.): 0/13 (R7_ARCH_HPARAM.md; r7_screen.md).
- MoE, PLE, TabM, FoX, QK-norm, semantic IDs, propensity head: closed (nnx_VERDICT §5).
- Label smoothing: closed (sol_solvers §2.4).
- Born-again KD: closed (nnx_VERDICT §5).
- Test pseudo-labels: −0.0004 (COVERAGE_MAP).
- TTA variants: ≤ 0 (r6_tta.md).
- Plateau weight averaging / SWA / EMA in the pool: closed (r6_recap §5; COVERAGE_MAP).
- Caruana / hill climbing / val-fitted weights: closed (sol_ensemble.md).
- Probability / raw-logit / std fusion: tie with the z-mean (sol_ensemble D/E; research_1112/tools.md).
- GBDT rankers and stackers: closed (deep_gbdt_ranking.md; SOLUTIONS_REVIEW §2).
- CatBoost in every role: closed (`nn_x/cb26`; CURRENT_STATUS 14:11).
- GBDT residual / encoder-score-as-input: closed (SOLUTIONS_REVIEW §2; COVERAGE_MAP).
- Importance / adversarial reweighting: ESS 2,880 (sol_advval.md).
- Segment weights: closed (EXPERIMENT_REGISTRY).
- Coverage-weighted val: closed (sol_advval ADV-9).
- Post-hoc cold offsets: learned 0 (SOLUTIONS_REVIEW §2).
- Causal / lagged popularity (PDA-like): closed (FEATURES_REVIEW.md).
- Item mean production position: null (featrev_inventory.md:621).
- Same-recipe extra members / 2-epoch seed families: saturated (SOLUTIONS_REVIEW §3; CURRENT_STATUS 13:55).
- Recency windows for co-vis statistics: no better (eda_shift_overlap.md; eda_collab_stats.md).
- LLM backbones 0.6B-9B: much worse (llm_large.md).
- OTTO / H&M candidate-generation and co-visitation recipes (deep_otto_hm.md).
- RecSys 2024 / Russian competitions playbooks (deep_recsys2024.md; deep_ru_competitions.md; deep_PLAYBOOK.md).

------------------------------------------------------------------------------------------------------------------------

## 5. Sources

Access caveat: Kaggle writeup pages could not be fetched directly. Read in full: the LMSYS 1st-place body, the BirdCLEF+ 2025 2nd CEUR paper, and (via a Hugging Face dump of Kaggle writeups) AES 2.0 2nd/3rd, Jigsaw ACR 6th, LEAP 1st, CIBMTR 3rd, ISIC 2024 7th, Home Credit public 8th, Eedi 2nd and Learning Equality 1st. All other Kaggle numbers come from search extracts or secondary reports ([E-x]).

**Competitions: shift, selection, two-stage training**
- AES 2.0 1st (flg, 2024): https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/writeups/flg-1st-place-solution-trust-cv-and-lb-a-bit ; summary of the top 5: https://hippocampus-garden.com/kaggle_aes2/
- AES 2.0 2nd (yao, 2024): https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/writeups/yao-2nd-place-solution ; 3rd (ho, 2024): https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/writeups/ho-3rd-place-solution
- Home Credit CRMS 1st (yuuniee, 2024): https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/yuuniee-1st-place-solution-my-betting-strategy ; 9th (Dxfory): https://github.com/Dxfory/home-credit ; public 8th / private 253 (Grigoreva): https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/evgeniia-grigoreva-public-8th-private-253-solution
- Optiver Trading at the Close 1st (hyd, 2023-24): https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- Enefit Prosumers 1st (hyd, 2024): https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/writeups/hyd-1st-place-solution ; 5th (loh-maa): https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/writeups/loh-maa-5th-place-solution
- Jane Street RTMDF 2nd (Patrick Yam, reproduction by cweill, 2025): https://github.com/cweill/jane-street-causal-forecasting ; 8th: https://www.kaggle.com/c/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution ; 162nd: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/max-abhi-private-lb-162nd-solution ; Volkova solution: https://raw.githubusercontent.com/evgeniavolkova/kagglejanestreet/master/solution.md
- CIBMTR equity post-HCT 3rd (2025): https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/writeups/survivalhorror-3rd-place-solution
- Leash BELKA 1st (Shlepov, 2024): https://nips.cc/virtual/2024/109287 ; generalisability study: https://pmc.ncbi.nlm.nih.gov/articles/PMC13663858
- CMI Problematic Internet Use 1st (Haupt, 2024): https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/writeups/lennart-haupts-first-place-write-up-or-how-i-won-t
- UM MCTS 1st (jday96314, 2024): https://github.com/jday96314/MCTS
- Stanford Ribonanza top 3 (2023-24): https://kaggle-solutions-quartz.pages.dev/competitions/Stanford-Ribonanza-RNA-Folding
- Eedi 1st (2024): https://github.com/rbiswasfc/eedi-mining-misconceptions ; 2nd: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution ; organiser blog: https://www.eedi.com/news/from-wrong-answers-to-real-insights-how-we-used-a-kaggle-challenge-to-map-student-misconceptions
- KDD Cup 2020 Debiasing 1st (aister): https://awesome.ecosyste.ms/projects/github.com%2Faister2020%2FKDDCUP_2020_Debiasing_1st_Place
- ISIC 2024 7th: https://www.kaggle.com/competitions/isic-2024-challenge/writeups/former-zlp-dss-7th-place-solution-a-good-cv-is-all ; 4th review: https://zenn.dev/yuto_mo/articles/9585f0bf4d458b ; top-10 summary: https://zenn.dev/ottantachinque/articles/2024-09-12_isic2024-top-solution
- rsakata, "shake-upを科学する" (2025): https://speakerdeck.com/rsakata/shake-upwoke-xue-suru
- The Kaggle Book, validation chapter (summary): https://alibakh62.github.io/pages/content/book_summary/the_kaggle_book/designing_good_validation.html

**Competitions: recsys / CTR 2025-26**
- KDD Cup 2026 UniRec:
  - 1st QueryFormer: https://arxiv.org/abs/2609.16548
  - runner-up UniDot: https://arxiv.org/html/2608.16797
  - STAR: https://arxiv.org/html/2608.12986
  - 10th Z Lab: https://arxiv.org/html/2609.19787
  - 9th FA-RankMixer: https://arxiv.org/pdf/2607.15590
  - HyFormer baseline: https://arxiv.org/abs/2601.12681
  - workshop recap: https://www.besthub.dev/articles/real-world-industry-problems-take-the-kdd-stage-full-highlights-of-the-taac-2026-jeju-island-workshop-8fc3ecfe1db7
- Tencent Advertising Algorithm Challenge 2025 (organisers, top 3): https://arxiv.org/html/2604.04976v1
- WWW 2025 EReL@MIR Multimodal CTR 1st: https://arxiv.org/html/2505.03543v1
- WSDM Cup 2023 ULTR, 2nd: https://arxiv.org/abs/2304.12650 ; 3rd: https://arxiv.org/abs/2302.07530
- RecSys Challenge 2025: https://recsys.synerise.com/summary ; https://github.com/recsyspolimi/recsys-challenge-2025-synerise
- FlightRank 2025 notebooks: https://www.kaggle.com/code/ka1242/xgboost-ranker-with-polars ; https://www.kaggle.com/code/stpeteishii/flightrank-catboost-ranker-explanation
- Learning Equality curriculum recommendations 1st (2023): https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/writeups/epicurus-1st-place-solution
- OTTO 7th (recency-weighted co-visitation): https://www.kaggle.com/competitions/otto-recommender-system/discussion/383769
- Kaggle top-solutions corpus (7,237 summaries, grep only): https://github.com/TaeyanG4/kaggle-top-solutions-and-strategies
- ML Contests, State of ML Competitions 2025: https://mlcontests.com/state-of-machine-learning-competitions-2025

**Competitions: NN training tricks**
- LMSYS Chatbot Arena 1st (sayoulala, 2024): https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629 ; code: https://github.com/shyoulala/LMSYS_BlackPearl
- WSDM Cup 2025 Multilingual Arena 1st: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/whitefebruary-1st-place-solution ; 3rd: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/plamo-1000000b-3rd-place-solution
- Jigsaw Agile Community Rules 2025, 6th (DML): https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/6th-place-solution ; 15th deck: https://speakerdeck.com/hasibirok0/map-jigsaw-code-golf-zhen-rifan-rihui-by-guan-dong-kagglerhui-jigsaw-15th-solution
- BirdCLEF+ 2025 2nd (CEUR Vol-4038 paper 256): https://ceur-ws.org/Vol-4038/paper_256.pdf ; 5th: https://www.kaggle.com/competitions/birdclef-2025/writeups/noir-5th-place-solution-self-distillation-is-all-y ; BirdCLEF 2026 1st: https://www.kaggle.com/competitions/birdclef-2026/writeups/1st-place-solution-noisy-student-meets-distillati
- LEAP ClimSim 1st (2024): https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/writeups/greysnow-no-leaky-1st-place-solution-for-the-leap-
- MAP 2025 organiser overview: https://aclanthology.org/2026.aimecon-main.33.pdf ; 3rd: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution
- HMS 2nd (2024): https://dev.to/moyuto/kaggle-hms-2nd-solution-explaind-34jj ; ASL Signs 1st (2023): https://www.kaggle.com/competitions/asl-signs/discussion/406684 ; USPPPM 8th (2022): https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/writeups/team-n-8th-place-solution-predicting-targets-at-on
- CSIRO Image2Biomass 1st: https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution ; RSNA Aneurysm 1st: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution ; AIMO-2 1st: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills
- Playground: https://www.kaggle.com/c/playground-series-s5e5/discussion/582591 ; https://www.kaggle.com/c/playground-series-s3e24/discussion/455296 ; https://www.kaggle.com/c/playground-series-s5e11/writeups/6-solution-ensembling-was-the-key ; https://www.kaggle.com/c/playground-series-s5e6/discussion/587398
- NVIDIA Kaggle Grandmasters Playbook (2025): https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data ; Deotte stacking (Apr 2025): https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/

**Industry papers**
- PinFM (Pinterest, 2025): https://arxiv.org/html/2507.12704v3
- GPSD (Alibaba, 2025): https://arxiv.org/html/2506.03699v2
- LinkedIn Feed SR (2026): https://arxiv.org/html/2602.12354
- GenRank (Xiaohongshu, 2025): https://arxiv.org/abs/2505.04180
- TransAct V2 (Pinterest, 2025): https://arxiv.org/html/2506.02267
- Netflix foundation model (secondary): https://www.zenml.io/llmops-database/foundation-model-for-personalized-recommendation-at-scale
- Covington et al., YouTube DNN (RecSys 2016): https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf
- TIF: https://arxiv.org/abs/2311.16878
- MEDA (Kuaishou): https://arxiv.org/abs/2305.19531
- Let It Go?: https://arxiv.org/html/2507.19473v1
- TDRO: https://arxiv.org/html/2312.09901v3
- PD/PDA: https://arxiv.org/abs/2105.06067
- InvCF: https://arxiv.org/abs/2302.05328
- Zhou et al., WWW 2024: https://arxiv.org/abs/2403.11136
- Google Ads "On the Factory Floor": https://arxiv.org/abs/2209.05310
- Li, Qin, Wang, Metzler, KDD 2019: https://storage.googleapis.com/gweb-research2023-media/pubtools/5046.pdf
- OTTO DL vs GBDT (RecSys 2025): https://arxiv.org/html/2507.20753v1
- ShareChat FIRE'23: https://arxiv.org/html/2312.01760

**Methods papers**
- Selection and shift:
  - Roelofs et al., NeurIPS 2019: https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning
  - Cawley & Talbot, JMLR 2010: https://www.jmlr.org/papers/v11/cawley10a.html
  - Baek et al., ICML 2022: https://arxiv.org/abs/2206.13089
  - Mehra et al. 2024: https://arxiv.org/abs/2404.01542
  - Wild-Time: https://arxiv.org/abs/2211.14238
  - Ovadia et al. 2019: https://research.google/blog/can-you-trust-your-models-uncertainty/
  - Byrd & Lipton 2019: https://proceedings.mlr.press/v97/byrd19a.html
  - Ji et al.: https://arxiv.org/abs/2010.11060
  - Schneider et al. 2020: https://arxiv.org/abs/2006.16971
  - TabReD: https://arxiv.org/html/2406.19380v4
- Data selection and annealing:
  - RHO-LOSS: https://arxiv.org/abs/2206.07137
  - JEST: https://arxiv.org/abs/2406.17711
  - AutoDebias: https://arxiv.org/abs/2105.04170
  - KDCRec: https://dgliu.github.io/pubs/TKDE_2023_KDCRec.pdf
  - Blakeney et al.: https://arxiv.org/abs/2406.03476
  - Llama 3: https://arxiv.org/abs/2407.21783
  - OLMo 2: https://arxiv.org/abs/2501.00656
  - Ibrahim et al.: https://arxiv.org/abs/2403.08763
- Ensembles and training:
  - Abe et al., TMLR 2024: https://arxiv.org/abs/2302.00704
  - WiSE-FT: https://arxiv.org/abs/2109.01903
  - Zhu et al., CIKM 2020: https://arxiv.org/abs/2011.04106
  - Stanton et al.: https://arxiv.org/abs/2106.05945
  - Wenzel et al.: https://arxiv.org/abs/2006.13570
  - Liu et al., late dropout: https://arxiv.org/abs/2303.01500
  - DiWA: https://arxiv.org/abs/2205.09739
  - Lee et al. 2015: https://arxiv.org/abs/1511.06314
  - Ash & Adams: https://arxiv.org/abs/1910.08475
  - Muon+: https://arxiv.org/abs/2602.21545
- Test-time training:
  - Akyürek et al.: https://arxiv.org/abs/2411.07279
  - T2ARec: https://arxiv.org/abs/2504.01489
- Tabular foundation models:
  - TabPFN-3: https://arxiv.org/abs/2605.13986
  - Ensembling TFMs: https://arxiv.org/abs/2605.18696
  - TabArena: https://arxiv.org/html/2506.16791
  - Drift-Resilient TabPFN: https://arxiv.org/abs/2411.10634
  - RelBench rel-avito: https://arxiv.org/html/2407.20060v1

**Internal** (all under <repo>):
- `reports/research_1112/{shift,recsys,industry,nntricks,tools,match,critique}.md`
- `reports/CURRENT_STATUS.md`, COVERAGE_MAP.md, EXPERIMENT_REGISTRY.md, SOLUTIONS_REVIEW.md, ROUND5.md, R6_PLAN.md, nnx_VERDICT.md, review_integrity.md, review_shift.md, sol_advval.md, sol_ensemble.md, err_SYNTHESIS.md, eda_shift_overlap.md
- `nn_x/pubdiag/{build,noise,overfit,shift}/`
