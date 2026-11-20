# int_moe: has mixture-of-experts been checked? Internal audit of the project record

Written 2026-11-15 14:35-15:05Z by the internal auditor for mixture-of-experts (MoE), gating and routing ideas. CPU only, stored predictions only, nothing trained, no submission file. Work directory `nn_x/w12/lit/int_moe/`; my own reads were declared in `DECLARATION.txt` there before any labelled number.

Marks: **[M]** measured by me in this audit, **[D]** read in a project file (path given), **[I]** inferred. "+-" is one standard error of a per-feed paired difference on the 14,423 val feeds unless stated. No web search was made for this report; external papers are quoted only as the record quotes them.

## 0. Answer

1. **Yes, MoE was checked, but unevenly.** Gates and routers between existing models were tested in eight separate studies, three of them today, and all are null. Experts inside one network were trained exactly once: a sparse MoE feed-forward at 2 epochs under the old optimiser with 2 seeds (val +0.00027 +- 0.00047). Every other in-network form was closed by a CPU proxy or by argument.
2. **The naming in COVERAGE_MAP l.61 overstates coverage.** "PLE" there is piecewise-linear numeric embeddings and "TabM" is a 4-member weight-sharing ensemble. Neither is a mixture of experts. Multi-task expert networks (MMoE, the other PLE) were never trained.
3. **No MoE-type closure rests on the in-period holdout alone**, with one exception that is already repaired: the popularity-light pool as an ensemble partner, first vetoed by the holdout, now the team-best file (public +0.00052). The routing study of 11-13 quotes a holdout headline (-0.0025, z -6.0) that is the known holdout bias times the share of routed feeds, not evidence about routing; its val-nested null stands.
4. **Oracles are large and half noise.** A perfect per-feed switch between today's base pool and the popularity-light pool is worth +0.0190 [M], but the same switch between a 12-member pool and 4 students of the same recipe is worth +0.0092 [M]. Learned routers reach AUC 0.494-0.511 against a null of 0.496-0.503 [D].
5. **New here:** a candidate-level gate (a different popularity-light weight on cold rows, or on young / low-count warm rows) has no signal at pool level: nested gain over a constant weight -0.00011 +- 0.00003 and -0.00017 +- 0.00018 on two member sets [M]. The admixture earns its gain in the order among warm rows, not by moving cold rows.
6. **Remaining headroom.** Gates across models: 0 to +0.0002, and the only candidate (weight 0.4 on feeds with M >= 50) already has its public slot. Experts inside the network: the one run cannot exclude an effect below 0.001, but width, diversity and distillation evidence put the expected file-level gain at 0 to +0.0001; my probability of +0.0003 or more is about 0.05 [I].
7. **Do we need to try them?** No further gate search. One fair in-network test (sparse MoE students under the current recipe, 8 seeds, about 14-18 GPU-h) is the only run that would replace an argument by a measurement; it is optional filler, not a priority.

## 1. Terms

| term | meaning |
|---|---|
| member | one trained ranker. TO = its train-only predictions; FT = the same model after the 5-fold fine-tune on val |
| pool, file | pool = member-equal mean of per-feed z-scores; file = 0.5 z(FT pool) + 0.5 z(TO pool) |
| G3 | the 20-member hard-label ensemble of 11-13 (val 0.40032) |
| W16 / FD16 | 16 wide distilled students, trained without / with the 30,000 holdout feeds |
| popularity-light | members trained without the 8 popularity inputs, with one exposure count added ("pop1" view) |
| PLAIN | popularity-light pool mixed into the base pool at weight 0.2 (the scored W8P20 form) |
| M-rule | weight 0.2 on feeds with M < 50 and 0.4 on feeds with M >= 50 |
| witnesses | val (two later days); in-period holdout (30,000 train-period feeds, wrong in sign for popularity contrasts); pmix (val reweighted to the public or private feed mix); stale-val; lagged holdout; public |
| nested | the rule or gate is fitted on training users and scored on held-out users (user-grouped folds) |

## 2. What was built: one row per MoE-type experiment

### 2.1 Experts or gates inside one network (trained)

| # | experiment; COVERAGE_MAP line; source | where the experts sit; what the gate saw | scale | val | other witnesses | how judged; what decided | fair test? |
|---|---|---|---|---|---|---|---|
| T1 | **A5 sparse MoE feed-forward**; l.61; `reports/nnx_exp_attn.md` A5, `reports/nnx_VERDICT.md` 2.3 and 5 | Trunk. The feed-forward of the 4 candidate blocks (2 cross-attention, 2 set-attention) became 1 shared expert (width 128) + 8 routed experts (width 64), top-2. Gate: a sigmoid router on each candidate token's own hidden state, load balanced by a bias without an auxiliary loss. History encoder unchanged | 2 epochs, AdamW, d128, seeds 23 / 24; +0.72M parameters (+84%), activated width equal to the dense layer | alone +0.00027 +- 0.00047 [D]; re-read from the archived files +0.00027 +- 0.00045 (seeds +0.00016 / +0.00038) [M]. On top of retrieval + attention gate +0.00015 +- 0.00048; after the fine-tune -0.00034 +- 0.00052 [D, M] | holdout +0.00053 +- 0.00028 (z 1.9 printed, 0.8 after the run-noise calibration) [D]. Private mix +0.00003 +- 0.00072; M >= 50 +0.00002 +- 0.00050 [M]. 2-member pools: MoE pair minus dense pair +0.00059 +- 0.00054 (alone), -0.00070 +- 0.00055 (train-only, on top of retrieval), +0.00052 +- 0.00060 (fine-tuned) [M]. Diversity margin +0.0032, equal to the attention gate's +0.0033 [D `scratchpad/power/skeptic_m2upcycle/a5_margin.log`] | Design rule 1.6: paired val on 2 seeds plus "holdout not worse". Rejected "as a single model" on the val null. **Not a holdout closure**: the holdout was the favourable witness | **No.** 2 of the recipe's 10 epochs; before Muon, the cooldown and distillation; 2 seeds resolve about 0.001, not 0.0003; one router collapsed (set block 1, load ratio 80-94); no dense control at equal parameter count. Never repeated: `--moe` exists only in `nn_x/attn/train.py` [M, grep] |
| T2 | **B5 "TabM-mini"**; l.61; `reports/nnx_exp_feat.md` 6 | Not MoE. 4 implicit members share the whole trunk and differ by one multiplicative input adapter; output = mean of the 4 logits; no gate | 2 epochs, AdamW, 3 seeds, 1.45x time | +0.00009 +- 0.00042 [D] | holdout +0.00032 +- 0.00025; members correlated 0.999 [D] | Same rule: "likely" by the letter, rejected as neutral at 1.45x cost | Only the mini form. Adapters in every layer or separate heads were never run |
| T3 | **CATGRAD category-factorised head**; l.214; `nn_x/w9/catgrad/logs/w9_eval_S_catgrad.txt` | Head. Score = log P(category) + log P(item given category), with a separate detached category head: a hierarchical mixture over the 54 categories | 2-epoch screen, Muon recipe, seeds 23 / 24 | -0.00097 +- 0.00061 [D] | holdout -0.00120 +- 0.00034 (z -3.5); diversity margin +0.0113; grafting its category part on the base -0.00095 +- 0.00043 val [D] | Pre-declared stage-S rule; failed the holdout clause (>= -0.0005) and a val clause (category graft >= -0.0006) | Screen only (2 epochs). The holdout co-decided, but val agrees in sign and this is not a popularity contrast |
| T4 | **Gates that were adopted** (for completeness): attention-output gate A1, the gate on the retrieval branch, the traffic-domain token; l.59; `reports/nnx_VERDICT.md` 2.3, `docs/02_SOLUTION.md` l.188, l.387 | Gates, not experts: an elementwise sigmoid on each attention output; a 3-way domain embedding (production / random-like / exploration) added before the head | A1: 4 seed pairs at 2 epochs | A1 mean of 4 pairs +0.00065 (SE about 0.0003) [D]; domain token and similarity bias "+0.0004 each (inconclusive)" [D] | A1 holdout +0.00050 +- 0.00028 [D] | Adopted as cheap and never negative | - |

### 2.2 Gates and routers across ensemble members (CPU, stored predictions)

| # | experiment; line; source | experts; what the gate saw | scale | val (nested unless stated) | other witnesses | how judged; what decided | fair test? |
|---|---|---|---|---|---|---|---|
| P1 | **Segment-routed blend weights, day 1**; l.111; `reports/err_segments.md` l.21, l.181-186 | NN against a stacker / CatBoost / the train-only NN; blend weight routed by M, history length, user type, cold share, category counts, model disagreement, hour, day | nested CV, day-1 models | -0.0010 / -0.0008 / -0.0004 / -0.0006 / -0.0005 / -0.0009 / -0.0010 / -0.0005 / -0.0008 / 0.0000 against one global weight (SE 0.0003-0.0004); regime-routed weights (regime AUC 0.772) 0.0000 [D] | none | val only | Yes for the form; models are obsolete |
| P2 | **Cold-click and exploration gates**; l.120, l.281; `reports/sol_errors.md` 2.3; rerun `scratchpad/power/moe/cold_mech.log` | One model. "Cold expert" = a +1.0 z boost of cold candidates, switched on per feed by a LightGBM gate on 16-31 label-free feed covariates | nested 5-fold by user; v11 (11-11) and G3 (11-13) | gate AUC 0.66 -> -0.00003 +- 0.00021 (v11); AUC 0.654-0.671 -> -0.00005 +- 0.00004 (G3); exploration gate AUC 0.72 -> 0.0000 [D] | none | val only | Yes for a feed-level gate. A cold expert trained inside the network was never run |
| P3 | **Fine-tuned against train-only weight, routed**; l.100, l.303; `reports/sol_errors.md` 3.3, `reports/DATA_PIPELINE_1113.md` 4, `reports/RECHECK_1114/ft_redesign.md`, `reports/MIX_1115/gateB.md` | Two branches of the same members; weight of the fine-tuned branch by M, cold share, user type, history length | nested; 3 families (11-11), 10 fresh member sets (11-13), 4 compositions (11-15) | by M +0.0003 +- 0.0003 / -0.0001 / -0.0002 across three families; "0.3 for M < 50, 0.6 for M >= 50" nested +0.00064 (z 2.96), then +0.00004 +- 0.00009 on 10 fresh member sets; today no cell moves the optimum off 0.5 [D] | the holdout cannot read it (no fine-tuned holdout predictions) | val nested + replication on other member sets | Yes. The 11-13 result is the textbook case: a nested z of 3 that is member-set noise |
| P4 | **Second-level combiners, 78 arms**; l.219; `reports/DATA_PIPELINE_1113.md` 4 | Family weights (GRU family / two transformer families), feed-conditional weights, stackers, pooling functions | nested, 20 members of G3 | family weights +0.00049 (z 1.47); GBDT stacker -0.0029..-0.0048; MLP residual +0.00005 / +0.00022 [D] | holdout: family weights -0.00048 (z -2.85); 0 of 78 with holdout z >= 2 [D] | val > 2 SE AND holdout >= 0 AND pmix >= -1 SE; 0 of 78 pass | The holdout co-decided. These are same-view families, not a popularity contrast; `reports/STALE_1115/veto.md` row 219 re-read it: "stands" |
| P5 | **Trajectory-type conditioning (P1 of the pre-training plan)**; l.192; `reports/PRETRAIN_PLAN_1113.md` 7.1, `nn_x/pretrain2/P1/` | Three G3 families; per-cluster family weights and per-cluster fine-tune weight. Gate: k-means cluster (K 8 / 32) of the user vector or of a history profile | nested, 8 arms + random-label controls | best arm +0.00022 +- 0.00025, below the random-label control +0.00028 +- 0.00024; per-cluster fine-tune weight best +0.00047 +- 0.00022 (val only) [D] | holdout +0.00056 +- 0.00015 for the best arm, transfers in neither direction (-0.00018 / -0.00032) [D] | Gate: val >= +0.001 at z >= 3.58 AND holdout > 0. Failed on val. Not a holdout closure | For re-weighting existing families, yes. It was also used to close cluster FiLM, prefix tokens and "per-cluster MoE" inside the network: **those were never trained** (closure by proxy; the plan's own per-cluster signal arm was not run either) |
| P6 | **Feed-conditional routing between G3 and popularity-light pools**; l.280; `scratchpad/power/moe/route_check.log`, `reports/SOLVER_POWER_1113.md` 4.2 | G3 against 8 pop1 members, 4 popularity-free members, the mixed pool C3, and an A/A split of G3. Gate: 31 label-free feed covariates including model agreement and G3's confidence; LightGBM, ridge, logistic, soft conditional weight | nested 5 folds x 2 seeds | router AUC 0.494-0.511; permuted null 0.496-0.503; A/A 0.481-0.500. Routed minus global -0.0003..-0.0010; soft conditional weight -0.00087 +- 0.00049 / +0.00024 +- 0.00048 / -0.00030 +- 0.00024 [D] | holdout: val-fitted router -0.00247 +- 0.00042 (z -6.0); routers fitted inside the holdout AUC 0.506-0.516, -0.0002..-0.0010 [D] | Kill rule: AUC <= null + 0.02 OR nested val gain <= 0 OR holdout transfer < 0. All three fired | Yes on val. **Flag:** the z -6.0 is not routing evidence. The router sends 44% of feeds to a pool that reads 0.0066 lower on the holdout by the known bias: 0.442 x -0.00662 = -0.0029 [I]. Closed on val; veto.md row 280 says the same |
| P7 | **Popularity-light pool at a constant weight** (no gate); l.217, 302, 312, 332; `reports/CURRENT_STATUS.md` 1 | Two "experts" that differ by INPUT VIEW: full-popularity students and popularity-light students, weight 0.2 | 8 + 4 students | +0.00049 +- 0.00031 [D] | **public +0.00052 +- 0.00041**; pmix-private +0.00079 +- 0.00048; in-period holdout -0.00063 +- 0.00022; lagged holdout +0.00024 (best of nine compositions; their mean +0.00003 +- 0.00023) [D `reports/STALE_1115/premortem.md`] | First closed on the holdout clause (11-13, 11-14), re-opened by one SE-denominated public read | **The one holdout-only closure in this family; already repaired.** It is also the only item in this table that paid |
| P8 | **M-rule** (feed-size step in the popularity-light weight); l.none yet; `nn_x/w5/logs/E_kd_predeclared.txt` 09:27Z-14:03Z, `reports/MIX_1115/review.md` | Same two pools; hard gate on M >= 50 | 8 member compositions | in-sample +0.00026 (FD16 + 4), +0.00051 (FD8 + 4), +0.00071 (W16 + 4), +0.00043 (W8 + 4); honest (nested, cross-member-set) +0.00002 +- 0.00027 / +0.00015 / +0.00030 / +0.00015 [D]; reproduced +0.00026 +- 0.00021 and +0.00071 +- 0.00021 [M] | lagged holdout: against 0.4 on long feeds in 9 of 9 compositions, mean -0.00045 +- 0.00023 [D premortem]; on full-data members the gate equals a random gate of the same size (p 0.55) [D review]; public: pair in the slots, not read at 14:33Z | Reading rule amended 14:03Z before any public score | "Supported, not shown". Chosen from about 75 rules on the same val feeds |
| P9 | **Declared rule families and a learned gate over the four pools**; `reports/MIX_1115/rulesA.md`, `gateB.md`, `review.md` | Base and popularity-light pools, fine-tuned and train-only. rulesA: 95 families, 131,547 rules on gates M, cold share (two definitions), history length, user 0. gateB: stumps and ridge-logistic gates on 8 label-free features including pool disagreement | nested 5 x 5 folds, placebos, 4-8 compositions | rulesA: 0 of 95 families pass. gateB primary gate minus M-rule -0.00022 +- 0.00021 (FD16 + 4) and -0.00038 +- 0.00016 (W16 + 4); in-sample optimism 0.0004-0.0010 [D] | pmix reported; reviewer's 250,469 own rules agree [D] | val nested, placebo-calibrated, replicated across member sets. No holdout (members trained on it) | Yes; the most complete gate study on record |
| P10 | **Candidate-level gates** (this audit); `nn_x/w12/lit/int_moe/cand_route.log`, `cand_route2.log` | Same pools; weight of the popularity-light pool by ROW class: cold / warm; cold / warm-young / warm-old; cold / warm low-count / high-count | nested 5 x 5 folds, 2 compositions, 14 placebos | nested family minus nested constant: cold / warm -0.00011 +- 0.00003 (FD16 + 4), -0.00017 +- 0.00018 (W16 + 4); by listing age -0.00024 +- 0.00006 / -0.00032 +- 0.00020; by count level -0.00027 +- 0.00009 / -0.00022 +- 0.00022 [M] | private mix -0.00016 / -0.00031 / -0.00030 (FD16 + 4) [M] | Declared criterion (>= +2 SE primary, >= +1 SE replication, above placebos, pmix >= 0): NO SIGNAL three times | Yes at pool level. Says nothing about a trained expert |

### 2.3 Closed without a run

| # | idea; line; source | what closed it | flag |
|---|---|---|---|
| N1 | **Sparse MoE grown from the plateau snapshots ("upcycled")**; l.282; `reports/SOLVER_POWER_1113.md` 4.3, 6.4 | Two reviewers. T1's diversity margin +0.0032; 32 stored cooldown-only arms have margin median +0.0016, max +0.0104; swapping 4 of 20 members keeps pool correlation >= 0.9993 [D] | The bar used was +0.002 (margin 0.10). For +0.0004 the same law needs margin about 0.022 [I]; the closure still holds by a factor of 2-7 |
| N2 | **Per-vertical experts / adapters**; l.283; `scratchpad/power/skeptic_m3vert/vert_resid.log` | 99.2% of feeds mix verticals (mean 6.0). A per-vertical correction function fitted on G3's output: minus the global one -0.00095 +- 0.00039 (z -2.4) [D, log read] | A residual-form test. The record's own scope note (l.323) shows residual tests under-read trained inputs by 4-10x. The stronger evidence is T1: its router did learn category routing (0.33-0.50 nats) and gained nothing |
| N3 | **Feed-length cooldown specialists**; `reports/SOLVER_POWER_1113.md` 4.4 | Not reviewed. By analogy: every cooldown data-selection arm hurt (-0.0021, -0.0035) | No run, no proxy |
| N4 | **Event-type expert heads (MMoE / multi-task PLE)**; l.272; `nn_x/insights/ccb_1113/RESULT.md` C15 | No per-type label at train time; 83-87% of positives are plain card views. Pre-gate: gradients of the main loss and the production-order auxiliary loss are near-orthogonal (median cosine +0.007..+0.014), so there is no task conflict to separate [D] | No model trained. The CPU "stage 0" of the proposal was not run either |
| N5 | **Domain-specialist output heads**; l.168; `reports/TRICKS_SWEEP_1113.md` 4.1 | Val and test are 100% random-order feeds, so the domain is constant at inference; the arm without the domain token reads +0.00044 val / +0.00001 holdout against the base, so the token carries no measurable signal [D] | By argument |
| N6 | **Trajectory-cluster FiLM / MoE in the network**; l.192 | Closed through P5 | By proxy (see P5) |
| N7 | **Staleness-aware gating in the network** (per-feed blanking of the popularity inputs with an indicator; a lag input); l.167; `reports/STALE_1115/veto.md` rows 167, item 8 | Refuted on 11-13 only through another arm's holdout veto | **Holdout-based refusal of a popularity idea.** veto.md already lists it for re-opening behind the stale-inputs recipe, which failed at file level today (-0.00047 +- 0.00035) |

## 3. Closures that leaned on the in-period holdout

The holdout is wrong in sign for popularity contrasts [D `reports/CURRENT_STATUS.md` 1]. Against that standard:

| row | role of the holdout | verdict |
|---|---|---|
| T1, T2, P5 | the favourable witness; closed on val | no issue |
| P1, P2, P3, P9, P10 | not used | no issue |
| P4 | co-decided; same-view families | re-read by veto.md row 219: stands |
| T3 | co-decided; val -0.00097 +- 0.00061 agrees in sign; not a popularity contrast | stands as a screen result |
| P6 | headline number is the bias (0.442 x -0.0066) | closed on val; drop the "z -6.0" from summaries |
| P7 | sole reason for the first closure | repaired; now the team-best file |
| N7 | sole reason; never run | open by veto.md, low priority after today's file-level failure of the stale-inputs recipe |
| P8 | the lagged holdout contradicts val and pmix | one public pair decides |

## 4. MoE forms never run

| form | nearest evidence on record | why it was not run |
|---|---|---|
| Sparse MoE feed-forward under the current recipe (Muon, 10 epochs, distillation, d192 students) | T1 only (2 epochs, old optimiser). Width under distillation: d128 -> d192 val +0.00038 +- 0.00049, holdout +0.00083 +- 0.00028 [D veto.md]; d192 -> d224 pooled z < 1.4 [D l.331] | "capacity is not binding"; the flag was never ported |
| Multi-gate experts in the candidate-set encoder, gate fed feed-level context (M, cold share, history length) | T1's gate read only the token itself. Post-hoc gates on exactly these variables: P3, P6, P9 null | the dense net is argued to condition on them already |
| Experts by feed size M inside the network | P8 (pool level, not shown); N3 never run | no random-order training feeds with M < 50 exist in usable form: the all-M exploration pool was a null lever (l.293) and sub-feed augmentation failed as implemented (l.319, l.331) [D] |
| Experts by cold / warm candidate inside the network | P2 (feed gate), cold offset +0.00013, P10 (row gate) all null. Data-side equivalents were run: the popularity-light view (adopted, P7); popularity-input dropout in the cooldown (closed 11-15: val -0.00066 +- 0.00048, stale-val +0.00001); the stale-inputs recipe (failed at file level 11-15) [D `reports/CURRENT_STATUS.md` log 08:58Z, 13:10Z] | closed as a "cold/warm trade" |
| Scenario towers by traffic domain | N5 | constant at inference |
| Per-vertical experts or adapters, trained | N2; T1's category routing | CPU pre-gate |
| Trajectory-cluster experts, trained | P5 | proxy |
| Multi-task experts with event-type heads | N4 | no label, no conflict |
| MoE over the history encoder (experts per event-type group) | none; filed with T1's class in the solver-power study | by class |
| Full weight-sharing ensemble (adapters in every layer, separate heads) | T2 (mini) | seeds already saturated |
| Gate fed the staleness of the counters | N7 | holdout veto of a sibling arm |

## 5. Oracle evidence

"Oracle" = the per-feed choice made with the label. All oracle numbers are in-sample upper bounds.

| choice | perfect router | noise reference | learned router | source |
|---|---|---|---|---|
| boost cold candidates only in feeds whose positives are all cold (17.4% of feeds) | +0.0253 (v11), +0.0257 (G3) | - | gate AUC 0.65-0.67 -> -0.00003 +- 0.00021 / -0.00005 +- 0.00004. Value-weighted AUC 0.52-0.53. AUC 0.85 would give +0.006, 0.94 +0.014 | [D] sol_errors 2.3; cold_mech.log |
| tilt "exploratory" feeds away from history similarity | +0.0159 | - | gate AUC 0.72 -> 0.0000 | [D] sol_errors 2.3 |
| G3 or 8 popularity-light members, per feed | +0.0217 | A/A split of G3: +0.0083 | AUC 0.498-0.506 (null 0.496); -0.0003..-0.0007 | [D] route_check.log |
| G3 or 4 popularity-free members | +0.0218 | same | AUC 0.506-0.511 (null 0.503); +0.00014..-0.00104 | [D] |
| G3 or the mixed pool C3 | +0.0071 | same | AUC 0.494-0.498; -0.0001..-0.0005 | [D] |
| best member per feed among 32 | 0.85 of what pure-noise copies give | - | - | [D] `reports/INSIGHTS_1112.md` l.34 |
| base pool or popularity-light pool, per feed (today's members) | **+0.0190** (FD16 + 4), **+0.0188** (W16 + 4) | 12 base + 4 same-recipe students: **+0.0092**; with the real popularity-light 4 on the same 12: +0.0190 | no router was fitted on this exact pair; the weight gates of P9 are its soft form: at best the M-rule | [M] cand_route.log |
| weight 0.2 or 0.4, per feed (the M-rule's two settings) | **+0.0046** / **+0.0042** | same-recipe: +0.0021 | M-rule in-sample +0.00026 / +0.00071 (6% / 17% of the oracle); honest about +0.0002; learned gates -0.00022 / -0.00038 below the M-rule | [M] oracle; [D] MIX_1115 |
| best of 64 candidate-level cold / warm weight pairs, per feed | **+0.0181** / **+0.0185** | same-recipe: +0.0094 | nested -0.00011 +- 0.00003 / -0.00017 +- 0.00018 against a constant | [M] |

Reading:
- **About half of every pool-pair oracle is noise selection.** The real pair reads 2.0-2.1 times the same-recipe reference in all three rows I measured. The earlier A/A reference (+0.0083 against +0.0217) gives the same ratio.
- **The other half needs the label.** The one strongly predictive quantity is an attribute of the clicked item (is it cold), which no feed covariate carries: in every cold-share quintile the gate's posterior is below the local break-even [D cold_mech.log].
- **No covariate tells which model is right.** The top router features are the same for real pairs and for the A/A control (top-1 agreement, top-1 margin, log M): they say how much two pools disagree [D].

## 6. Own measurements

Declared in `nn_x/w12/lit/int_moe/DECLARATION.txt` (block 1 at 14:34:42Z, block 2 after block 1 was read and before its own numbers). Reproduction passed: PLAIN and M-rule files of both compositions equal the record to 6 decimals; own NDCG equals `src/metrics.ndcg_per_feed` to 2e-16 [M].

**What was tested.** The popularity-light pool gets a weight per ROW instead of per feed. Compositions: FD16 + 4 full-data popularity-light students (primary) and W16 + 4 held-out students (replication; other trained models, same val feeds). Cold row = no interaction count (37.9% of val rows; 30.3% of positives) [M].

| family | FD16 + 4: nested family minus nested constant | W16 + 4 | placebos on FD16 + 4 (class label permuted within the feed, 5 draws) | verdict |
|---|---|---|---|---|
| cold / warm (64 rules) | -0.00011 +- 0.00003 | -0.00017 +- 0.00018 | -0.00048 .. -0.00011 | NO SIGNAL |
| cold / warm with listing age below / above the warm median | -0.00024 +- 0.00006 | -0.00032 +- 0.00020 | -0.00054 .. -0.00018 | NO SIGNAL |
| cold / warm with count below / above the warm median (9) | -0.00027 +- 0.00009 | -0.00022 +- 0.00022 | -0.00050 .. -0.00016 | NO SIGNAL |
| cold / warm x (M < 50 / M >= 50), information only | minus the M-rule -0.00024 +- 0.00022 | -0.00006 +- 0.00021 | - | nothing beyond the M-step |

Further facts [M]:
- On FD16 + 4 the best in-sample rule of every family is the constant 0.4. On W16 + 4 it puts MORE popularity-light weight on warm rows (cold 0.2, warm 0.6), the opposite of a "cold expert", and loses out of fold.
- **Where the admixture earns its gain.** Weight 0.2 on warm rows only equals PLAIN (-0.00000 +- 0.00014 and -0.00002 +- 0.00016). Weight 0.2 on cold rows only loses -0.00062 +- 0.00029 and -0.00066 +- 0.00029 (z -2.1, -2.3). The popularity-light pool does raise cold rows on average (+0.039 z against -0.024 on warm rows), but that shift is not what pays; the changed order among warm rows is. This agrees with the closed cold offset.
- The paired SE of a nested difference understates selection noise: with 4 same-recipe students in the popularity-light role, the same statistic read +0.00028 +- 0.00012 (z 2.45) in one of four splits. A nested z of 2 to 3 on one member set is not evidence here (the same lesson as P3).
- The archived T1 files reproduce the record (section 2.1).

Limits: val only (pmix reported); no stale-val, lagged holdout or public witness; two compositions that share the val feeds.

## 7. Remaining headroom

**Gates and routers across existing models: 0 to +0.0002.**
- Eight studies, from 10 segmentations on day 1 to 131,547 declared rules and a learned gate today, plus three candidate-level families here. Each ends at or below a constant weight once fitted out of fold.
- The measured optimism of a fitted gate (0.0004-0.0010) is larger than the gains now sought (0.0003-0.0005) and equal to the paired SE.
- The only survivor is the M-rule at about +0.0002 +- 0.0003, contradicted by the lagged holdout. Its public pair is already in the slots. Nothing else in this class deserves a slot or more CPU.

**Experts inside the network: expected 0 to +0.0001 at file level; P(>= +0.0003) about 0.05 [I].**
- What the record can exclude: an effect of +0.001 or more at 2 epochs (T1's SE is 0.00047). What it cannot: +0.0003 to +0.0005 under the current recipe.
- Strength. MoE buys parameters at constant compute. Under distillation width paid once (d128 -> d192) and then stopped (d192 -> d224 null). T1 itself was +84% parameters for +0.0003 +- 0.0005.
- Specialisation. T1's router found the category structure and it was worth nothing. The variables an expert could specialise on (category, M, coldness, user 0) are inputs the dense net already uses, and no post-hoc test finds a segment where a different function wins.
- Diversity. By the project's frontier law an equal-strength family needs a correlation margin of about 0.022 for +0.0004 [I]; T1 measured 0.0032, worth about +0.00006. Distillation narrows this further: students of one teacher share "one common mean function" (correlation 0.999+, l.299), and GRU students added for diversity were closed (l.331).
- The gain that did come from this family came from a second expert with different INPUTS at a constant weight (P7). That lever is being pushed now (more wide popularity-light students; a second and a third input view against a matched control, reads due about 16:30-17:15Z [D]). It is where the "experts" budget belongs.

**If the owner wants the MoE question closed by a run instead of by argument**, this is the one fair test. It is optional filler for idle GPUs, below the view-diversity and dose reads in priority.

| item | content |
|---|---|
| arm | the wide-student recipe (d192, distillation weight 0.5) with the T1 layer in the 4 candidate blocks: 8 routed experts + 1 shared, top-2 |
| code | port `MoEFF` and `--moe`, `--moe_topk` from `nn_x/attn/` to the current trainer by write-new + mv; flag-off bit-identity test; log expert load entropy per block (T1's set block 1 collapsed) |
| scale | 8 seeds (951-958) against the existing 8 students of the same seeds; about 1.7 GPU-h per member [D solver-power MoE lens], so about 14-18 GPU-h with fine-tunes [I] |
| read | 8-pool level, not a popularity contrast, so the standing rule applies: val AND in-period holdout, pooled z >= 2 at matched N, replicate band 1.6-2; pmix reported; plus an in-pool replay (8 of 16 members swapped) |
| stop | member val < 0 or pool z < 1: close in-network MoE for good, with a measurement behind it |

Not worth a run at any price in GPU-hours [I]: per-vertical adapters, domain towers, event-type heads, cluster experts, feed-length specialists. Each lacks either a routing variable at inference, a label, or a segment where the function differs.

## 8. Not checked

- I did not re-run `route_check.py`, `cold_mech.py`, the 78-arm study, the cluster study or the day-1 studies; their numbers are [D] from reports and, for route_check, cold_mech, a5_margin and vert_resid, from the logs themselves.
- `reports/MIX_1115/rulesA.md` was read to section 2 (of 833 lines) and `review.md` to section 2; their verdict blocks are quoted.
- My reads have no stale-val, lagged-holdout or public witness.
- The frontier-law figures in section 7 reuse the project's fitted coefficient (0.0186 per unit of correlation margin); I did not refit it on student pools.
- No web search. External papers appear only as cited in `reports/SOLVER_POWER_1113.md` and `reports/nnx_research_recsys_arch.md`, unverified by me.
- Whether a round-4 public score exists after 14:33Z.

## 9. Files

- `nn_x/w12/lit/int_moe/DECLARATION.txt` (criteria with time stamps)
- `cand_route.py`, `cand_route.log`, `cand_route.json` (cold / warm row gate, oracles with the same-recipe reference, placebos)
- `cand_route2.py`, `cand_route2.log`, `cand_route2.json` (three-class row gates)
- `a5_check.py`, `a5_check.log` (re-read of the archived sparse-MoE files)
- `G_C1.npy`, `G_C3.npy` (per-feed nested differences on val; not submissions)

Rows the lead may add to COVERAGE_MAP (this audit edits nothing else):

| row | text |
|---|---|
| new: candidate-level gates between the base and popularity-light pools (weight by cold / warm row; by listing age or count level among warm rows) | CLOSED 11-15 (CPU, nested, placebo-calibrated): -0.00011 +- 0.00003 / -0.00017 +- 0.00018 against a constant weight; three-class forms -0.0002..-0.0003. The admixture pays through the order among warm rows (warm-only weight = PLAIN; cold-only weight -0.0006). `reports/LIT_1115/int_moe.md` |
| l.61 note | "PLE" = piecewise-linear numeric embeddings, "TabM" = weight-sharing ensemble; the only MoE in that row is A5 (2 epochs, AdamW, 2 seeds). In-network MoE has no run under the current recipe |
| l.280 note | the holdout figure (-0.0025, z -6.0) equals route share x the popularity-light holdout bias; the closure rests on the val-nested null |
