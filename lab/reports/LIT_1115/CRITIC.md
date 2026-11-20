# LIT_1115 / CRITIC: adversarial review of SYNTHESIS.md (mixture-of-experts and graph networks)

Written 2026-11-15 15:51-17:05Z by the critic of workflow a workflow run. CPU only, no GPU process, nothing trained, no submission file.
Read in full: `reports/LIT_1115/{SYNTHESIS,int_moe,int_gnn,lit_moe,lit_gnn}.md`, `reports/COVERAGE_MAP.md` (337 lines), `reports/CURRENT_STATUS.md` sections 1-5 and the log from 07:13Z to 16:46Z. Read in part: `reports/feat_trd_.md` (sections 0-3, V.1-V.3, V.8), `reports/FRESH_1115/pregate.md` (sections 0-3), the logs and scripts of `nn_x/w12/lit/syn/`.
Own work directory: `nn_x/w12/lit/critic/` (rules in `DECLARATION.txt`, 8 time-stamped blocks, each rule written before its read). Nothing else was written or edited.

Marks: **[M]** measured by me today (log named); **[D]** read in a project file (path given); **[W]** read on the web by me today; **[I]** inferred. "+-" is one standard error of a per-feed paired difference on val unless stated. LR = likelihood-ratio statistic (twice the gain in log-likelihood).

Two failure modes were hunted: (A) the document recommends work whose expected value does not cover its cost; (B) the document dismisses a cheap idea that has a real mechanism for the measured bottleneck (stale popularity, 43-46% cold candidates) on an argument instead of a run.

## 0. Verdict

1. **The answer to the owner stands and can be acted on: nothing in the MoE or graph-network families needs to be tried.** A trained GNN, routers and gates, multi-task and per-token MoE are rejected for measured reasons that I re-checked (section 1).
2. **Two of the document's "new [M]" findings must not be quoted as written.** The NDCG lenses behind them return zero on inputs that are known to be worth something:
   - Lens of 2.2 (density columns): it reads **+0.00016 +- 0.00026** for the fresh popularity counters on a stale pool that is **0.00164 +- 0.00051** worse for lack of them [M].
   - Lens of 2.1 (cold rows re-ordered, warm rows fixed): it reads **-0.00007 +- 0.00005** for the encoder score on members trained without it, and -0.00082 +- 0.00028 when the cold rows are put in the order of the model that has that input [M].
   So "every CPU pre-gate is negative" (verdict item 3) is not evidence for the two graph feature forms. Only the likelihood-ratio test has power.
3. **I ran the confirmatory stage the document declared and skipped** (all val feeds, 49 CPU-minutes), with the lens that passed its positive control. The neighbour-completed scores DO carry detectable information on cold rows beyond the file: LR 4.40 (p 0.036) in z form, 11.37 (p 0.0007) in rank form; on the 11,369 feeds nobody had read: 3.48 / 8.56. The size is small: coefficient +0.024 against +0.116 for the positive control. Declared word: **WEAK**. "No detectable information" is wrong; "no build" is right.
4. **Density columns (G2): no CPU lens can read them** (item 2). The "no" rests on leverage: on the popularity-light host a pool gain of +0.0005 is +0.0001..+0.00015 in the file, below every witness.
5. **Optional runs.** C-MoE: do not run, not even as filler (failure mode A). C-Graph: allowed as filler on idle GPUs (all were idle at 16:47Z and 16:58Z [M]) only with three changes: the decision at file level, one column dropped, both hosts in the batch (section 4).
6. **Staleness-aware gate (M4).** The train-only half of its re-open trigger was readable on CPU; I read it. One unread arm meets it, and so does the parent recipe that then failed in the fine-tuned branch, so the half does not discriminate. No re-open (2.4).
7. Failure mode (A) is present only in the optional C-MoE. Failure mode (B) was present in form: two ideas with a mechanism were priced with lenses that cannot show a gain. With lenses that can, the action is the same.

## 1. Factual claims checked

| # | claim of SYNTHESIS | result | source |
|---|---|---|---|
| 1 | The gate class has no survivor: the feed-size rule read -0.00015 (-0.5 SE) on public at 14:34Z | TRUE [D] | `reports/CURRENT_STATUS.md` l.8, l.890 |
| 2 | In-network MoE was trained once: val +0.00027 +- 0.00047, -0.00034 +- 0.00052 after the fine-tune; 2 epochs, old optimiser, 2 seeds; the code exists only in `nn_x/attn` | TRUE [D, M] | `reports/nnx_VERDICT.md` l.91, l.149; `nn_x/w12/lit/int_moe/a5_check.log`; grep: 12 + 7 lines in `nn_x/attn/{model,train}.py`, none in the `nn/`, `nn_x/w5/`, `nn_x/abl/` trainers [M] |
| 3 | Diversity margin 0.0032 against 0.027 needed; 0 of 7 capacity arms pass under distillation; dense feed-forward x4 -0.00068 val | TRUE with two caveats [D] | `reports/SOLVER_POWER_1113.md` l.22, l.274; `reports/RECHECK_1114/kd_reprice.md` l.56; `COVERAGE_MAP.md` l.300. The log behind 0.0032 is no longer at the cited path. The -0.00068 is a 2-seed statistic with a run-to-run sd of about 0.0009 (same file, l.49): a direction, not a measurement |
| 4 | A GNN on the interaction graph has no node for 42.8% / 45.6% of public / private rows; two hops reach 97% of the rest | TRUE [D] | `nn_x/w12/lit/int_gnn/s1_basic.log` (stored count == 0: 0.3794 / 0.4276 / 0.4558), `s2_train.log` (0.5701 / 0.5874 = 0.971) |
| 5 | New read 2.1 (neighbour-completed scores on cold rows, 3,000 feeds): LR p 0.34 / 0.47, re-order -0.00030 +- 0.00020 | Numbers TRUE: LR 0.92 / 0.66 / 14.71 reproduced twice, from the stored file and from my own rebuild [M, `lr_B_OLD.log`]. Interpretation NOT supported: 2.2, 2.3 | `nn_x/w12/lit/syn/{gate_A,lr_A,placebo_A,ceiling_A}.log` |
| 6 | New read 2.2 (fresh-density columns on the file): -0.00030 +- 0.00031, LR p 0.20 | Numbers TRUE: reproduced to the last digit, per fold seed -0.00024 / -0.00045 / -0.00022 [M, `stale_trd_read.log`]. Interpretation NOT supported: 2.1 | `nn_x/w12/lit/syn/trd_read.log` |
| 7 | "No stale-val view exists for the columns of sections 2.1 and 2.2" | FALSE for 2.2 [M]: the columns are lag-0 by construction (`reports/feat_trd_.md` section 1) and stale-val predictions of the 16 wide students are on disk (`nn_x/w12/stale/hz2n_review/preds_stale_lt3/`). I ran that read (2.1) | - |
| 8 | The hz-family runs hznFS / hzt / hz2nF are "on disk unread" | TRUE, and readable in part on CPU [M]: 16 train-only val prediction files per arm in `nn_x/w5/preds/`. Read in 2.4 | - |
| 9 | C-Graph needs "no code, no build": `--xt_add` exists and is refused with `--init`; 20 `trd_` files for train, val, test | TRUE [M] | `nn_x/abl/train.py` l.282, l.299-300; `data/feats/{train,val,test}/trd_*.npy` (20 each) |
| 10 | Popularity-light admixture: public +0.00052 / +0.00041; the stale-inputs recipe failed at file level on three member sets | TRUE [D] | `CURRENT_STATUS.md` l.879, l.891 |
| 11 | Graph embedding next to a sequence encoder: -0.0001 / -0.0001 / +0.0002 | TRUE [W]: arXiv 2508.06970 table 6, all components 0.7531 / 0.8147 / 0.7899, without TwHIN 0.7530 / 0.8146 / 0.7901. The task there is user embeddings for churn and propensity targets, not the ranking of cold candidates, so it is an analogy | arxiv.org/html/2508.06970 |
| 12 | TokenMixer-Large abstract: sparse per-token MoE for "efficient parameter expansion", 7 and 15 billion parameters | TRUE [W] | arxiv.org/abs/2602.06563 |
| 13 | Independent 6-dataset multi-scenario benchmark | EXISTS [W]: Scenario-Wise Rec, CIKM 2025, 6 datasets, 12 models. Its table was not re-read by me | arxiv.org/abs/2412.17374 |
| 14 | "Tuned EASE / RP3beta / kNN match published GNNs" | FAIR as "match" [W, D]: the abstract of arXiv 2308.00404 states no verdict; in lit_gnn's transcription of its tables a graph model is ahead on 2 of 5 datasets | arxiv.org/abs/2308.00404; `lit_gnn.md` l.104 |
| 15 | "GPU time is not the reason" | TRUE [M]: 7 of the GPUs at 0% and 0 MiB at 16:00Z and 16:10Z, 8 of 8 at 16:47Z and 16:58Z (`nvidia-smi`). The 4,600 GPU-h figure ignores the owner's stop-loss for new recipes on 11-28 [D, `CURRENT_STATUS.md` l.44] | - |

## 2. What I measured

All reads are on val, CPU, stored arrays.

### 2.1 The lens of SYNTHESIS 2.2 cannot see a popularity-type input

Lens = the conditional logit of `nn_x/w12/lit/syn/trd_read.py`, copied unchanged (nested by user, 3 fold seeds, ridge 10). Script `stale_trd_read.py`, log `stale_trd_read.log`.

Pools (train-only predictions): WL = the 16 wide students with fresh val inputs (val 0.40157); S = the same 16 under private-like stale inputs (0.39993); P = 4 wide popularity-light students (0.40000); WP / SP = 0.8 z(base) + 0.2 z(P) with the fresh / stale base (0.40189 / 0.40082).
By-products [M]: the stale view costs the base pool **-0.00164 +- 0.00051** on val. The admixture is worth **+0.00089 +- 0.00033** on the stale base against +0.00032 +- 0.00031 on the fresh base.

Arms: T10 = the 10 density columns of SYNTHESIS (20 features); T4 = `trd_{kde,skde,n95,sn95}_sday` (8 features); **PC = positive control**: the stored FRESH counters `it_hist_all_luo` and `seq_cv_pop` (4 features). On S the positive control is the information the stale view removed. On P it is the information those students were built without.

| pool | arm | nested gain | in-sample LR (df), p | worth of the same information as a trained input |
|---|---|---|---|---|
| S (stale base) | **PC fresh counters** | **+0.00016 +- 0.00026** | 9.9 (4), 0.042 | 0.00164 +- 0.00051 (S against WL) [M] |
| P (popularity-light) | **PC fresh counters** | **+0.00006 +- 0.00034** | 26.6 (4), 0.00002 | about 0.0016 (P against WL at pool level; six more inputs included) [M] |
| WL (fresh base) | PC fresh counters | +0.00025 +- 0.00024 | 6.9 (4), 0.14 | 0 (already inputs) |
| S | T4 / T10 density | +0.00015 +- 0.00028 / -0.00012 +- 0.00033 | 16.4 (8), 0.037 / 30.4 (20), 0.064 | unknown |
| SP | T4 / T10 | -0.00020 +- 0.00027 / -0.00038 +- 0.00032 | 15.7 (8), 0.047 / 29.7 (20), 0.074 | unknown |
| P | T4 / T10 | -0.00014 +- 0.00027 / -0.00035 +- 0.00032 | 14.7 (8), 0.066 / 29.5 (20), 0.078 | unknown |
| WL | T4 / T10 | -0.00021 +- 0.00025 / -0.00013 +- 0.00032 | 13.4 (8), 0.098 / 29.1 (20), 0.087 | unknown |
| WP | T4 / T10 | -0.00024 +- 0.00025 / -0.00034 +- 0.00032 | 13.2 (8), 0.105 / 28.6 (20), 0.096 | unknown |
| F (the file) | T10 (= SYNTHESIS A8) | -0.00030 +- 0.00031 | 25.1 (20), 0.199 | unknown |

Placebo, T4 on S (10 draws, fold seed 0): mean +0.00007, sd 0.00017; the real read of that seed is +0.00007 (above 5 of 10). T4 on P: placebo mean -0.00005, real -0.00015 (above 3 of 10).

By the declared words: reproduction OK; lens has power = **NO**; mechanism visible = NO; overall **INCONCLUSIVE**.

Reading:
- The nested NDCG lens returns +0.0002 (0.6 SE) for an input whose removal costs 0.0016. It recovers about a tenth of a known effect and cannot separate it from zero. COVERAGE_MAP l.323 states this in general; this is the measurement for the popularity family.
- So "-0.00030 +- 0.00031, no signal" (SYNTHESIS 2.2) says nothing about a lag-0 input to a trained ranker. The negative sign of the 20-feature arms is the lens's own over-fit cost: the placebo of SYNTHESIS has mean -0.00050.
- The in-sample likelihood ratio does have power (p 0.00002 for the positive control on P). On that scale the density columns read p 0.04-0.20 on six correlated pools, lowest on the two stale pools. It is a hint in the direction the mechanism predicts, about a third of the positive control's excess on P. It is not evidence.

### 2.2 The NDCG lens of SYNTHESIS 2.1 is biased against any change; its likelihood-ratio test has power

Positive control for the cold-row lens: the pre-trained encoder score `pre2_fe_noid` (on cold rows it has no ID part [D, `reports/STALE_1115/hz2n.md` l.62]). X = the pool of the two 10-epoch members trained WITHOUT the six encoder inputs (`nn_abl_e10_xdrop_pre2_s951/952`); B = their matched base members. Worth of the channel on record: val +0.00077 +- 0.00073 (members), +0.00061 +- 0.00083 (2-pool) [D, `nn_x/abl/logs/readout_pre2_10ep.txt`]. Scripts `cold_lens_pc.py`, `cold_lens_pc2.py`, `cold_lens_pc3.py`.

| read (in the forms of SYNTHESIS 2.1) | sample A (3,000 feeds, 1,076 with a cold pair) | all 14,423 val feeds (5,266 with a cold pair) |
|---|---|---|
| LR, pool vs pool + column, 1 df, z form [A5] | **5.1, p 0.024**, coefficient +0.095 | **37.5, p 9e-10**, coefficient +0.116 |
| the same, rank form | 5.1, p 0.025, coefficient +0.086 | 40.1, p 2e-10, coefficient +0.109 |
| slot-preserving cold re-order, nested [A2] | 0 (weight 0 in every fold; in-sample curve negative at every weight) | **-0.00007 +- 0.00005**; in-sample maximum +0.00002 |
| cold rows of X put in the order of the base pool B, slots kept | -0.00037 +- 0.00057 | **-0.00082 +- 0.00028** |
| additive on cold rows only, nested | -0.00037 +- 0.00047 | +0.00025 +- 0.00024 |
| additive on all rows, nested (the project's standard residual) | -0.00081 +- 0.00094 | +0.00046 +- 0.00044 |

Reading:
- The slot-preserving re-order (A2, and A3 / A4, built the same way) reads zero or negative for a column that the likelihood-ratio test detects at p 9e-10. It loses 0.0008 even when the cold rows are put in the order of the model that HAS the input. The construction keeps the old model's confidence (the slots) and discards the new score's, so every change of order is penalised. "-0.00030 +- 0.00020" and "best linear use in sample +0.00017 +- 0.00037" are properties of the lens.
- No NDCG lens resolves an input of this size on 3,000 feeds (SE 0.0005-0.0009).
- The likelihood-ratio test is the one valid read of 2.1. At 3,000 feeds its power against an input as informative as the encoder channel was 0.5-0.75, and about 0.2 against half of that [I, non-centrality 4-7]. The document's own declared word for 2.1 was INCONCLUSIVE; its verdict prints that word and, in the same item, "no detectable information beyond the current file".
- The placebo of the rank-residual AUC was declared for block sizes 4 and 8 and reported for 4 only. With blocks of 8 the placebo reads 0.493-0.514 and real minus placebo is +0.027..+0.035 (2.7-3.5 SE) [D, `nn_x/w12/lit/syn/placebo_A.log`]. The statistic is biased for the 5-step column (+0.013..+0.025) and not for the 3-step column (-0.007..+0.004).

### 2.3 The confirmatory stage that SYNTHESIS declared and did not run

Build: copies of the five int_gnn scripts in `nn_x/w12/lit/critic/stageB/`, `s5_coldsignal.py 14423` (all val feeds: 527,379 cold rows, 6,659 cold positives; 48 min 50 s at 6.3 busy cores, 13.9 GB peak). Read: `lr_B.py` = the synthesis's `lr_read.py` with a feed-subset argument. Rule written before the build (block 4): PASS iff LR >= 10.83 for the 3-step column in z form on all feeds, with support from the 5-step column and from the unread feeds; FAIL iff LR < 3.84; otherwise WEAK.

| conditional logit over cold rows, pool F (the scored file) | sample A (read before) | the 11,369 feeds outside sample A | all val feeds | positive control, all feeds |
|---|---|---|---|---|
| pair feeds / cold positives | 1,076 / 1,347 | 4,189 / 5,270 | 5,265 / 6,617 | 5,266 / 6,618 |
| 3-step column, z form: LR, p, coefficient | 0.92, 0.34, +0.024 | 3.48, 0.062, +0.023 | **4.40, 0.036, +0.024** | 37.5, 9e-10, +0.116 |
| 5-step column, z form | 0.66, 0.42, +0.022 | 4.79, 0.029, +0.029 | 5.39, 0.020, +0.028 | - |
| 3-step column, rank form | 2.83, 0.093, +0.053 | 8.56, 0.0034, +0.047 | **11.37, 0.0007, +0.049** | 40.1, 2e-10, +0.109 |
| 5-step column, rank form | 0.50, 0.48, +0.022 | 8.11, 0.0044, +0.044 | 8.19, 0.0042, +0.040 | - |
| all 15 extras (15 df) | 14.7, 0.47 | 27.4, 0.026 | 25.3, 0.047 | - |

Pool W (the 16 wide students) gives the same picture: 4.31 / 5.23 in z form on all feeds. Build check: the sample-A column reproduces the synthesis to the last digit [M, `lr_B_{OLD,NEW,ALL}.log`].
Additive cold-row residual on all feeds, information only: -0.00010 +- 0.00015 (3-step) and -0.00013 +- 0.00013 (5-step) on pool F; the positive control reads +0.00025 +- 0.00024 in this form [M, `stageB_r1.log`].

Declared word: **WEAK** (3.84 <= 4.40 < 10.83).

Reading:
- The information is there. The coefficient is the same on the old and the new feeds (+0.024 / +0.023); sample A simply could not resolve it.
- It is small: 0.21 of the positive control's coefficient in z form, 0.45 in rank form (12% and 28% of its likelihood ratio). The positive control is worth +0.0006..+0.0008 as a trained input. Between quadratic and linear scaling that puts the neighbour-completed block at +0.00004..+0.00035 [I].
- An effect of that size cannot be confirmed by the reading rule in force, which detects a true +0.0005 in 56% of cases [D, COVERAGE_MAP l.324].
- Limits: int_gnn's construction (whole-sequence walk, catalogue-wide bank of items with >= 10 users), not lit_gnn's windowed count through same-cluster neighbours; val only.

### 2.4 The unread stale-recipe arms (re-open trigger of M4)

SYNTHESIS names as the trigger "a finished hz2n / hzt run that reads >= 0 on val and >= +1 SE on pmix-private at FILE level". Train-only val predictions exist for 16 re-cools of each of hzt, hznFS, hz2nF, with their references. No fine-tuned and no stale-val prediction exists, so only the train-only half can be read. Script `hz_unread.py`; the lead's own rules for these arms need the stale-val witness and are neither applied nor replaced.

| 16-pool, last checkpoint | arm - ref: val / pmix-private | on top of the admixture, [0.8 arm + 0.2 P] - [0.8 ref + 0.2 P]: val / pmix-private | arm - hz: val / pmix-private |
|---|---|---|---|
| hz (on record, check) | +0.00024 +- 0.00038 / +0.00128 +- 0.00065 | +0.00004 +- 0.00033 / +0.00054 +- 0.00052 | - |
| hzt | -0.00011 +- 0.00043 / +0.00090 +- 0.00070 | -0.00010 +- 0.00039 / +0.00049 +- 0.00060 | -0.00035 +- 0.00029 / -0.00038 +- 0.00044 |
| hznFS | +0.00001 +- 0.00039 / +0.00055 +- 0.00065 | **+0.00014 +- 0.00035 / +0.00094 +- 0.00056** | -0.00023 +- 0.00021 / -0.00073 +- 0.00036 |
| hz2nF | +0.00057 +- 0.00057 / +0.00185 +- 0.00091 | +0.00010 +- 0.00050 / +0.00048 +- 0.00077 | +0.00034 +- 0.00048 / +0.00058 +- 0.00075 |

By the declared word the train-only half of the trigger is met by hznFS (1 of 3 looks). The parent hz meets the same half (+1.03 SE) and failed in the fine-tuned branch on three member sets [D]. So this half does not discriminate, and as a pool hznFS is below hz. Deciding the trigger properly costs 16 fine-tunes per arm (about 5 GPU-h) plus stale-val inference. I see no reason to expect a different fine-tuned branch and do not recommend it.

## 3. Agreement and disagreement, item by item

### 3.1 Recommendations

| item of SYNTHESIS | my position | reason |
|---|---|---|
| "MoE: checked partly; graphs: checked partly" | AGREE | int_moe 2.1-2.3 and int_gnn 1 list every row; the two naming corrections to COVERAGE_MAP l.61 are right [D] |
| "TRY NONE" as the answer to "do we need to try them?" | **AGREE** | Sections 1 and 2. The action survives lenses that can show a gain |
| Bar "prior 0.15 for +0.0005" | AGREE as a working bar | The rule in force confirms a true +0.0005 in 56% of cases, so an arm at prior 0.15 yields a confirmed gain in about 8% of runs. With all GPUs idle [M] the bar is set by the lead's attention and the 11-28 stop-loss |
| Priors: G1 0.03, G2 0.03, G3 0.03, M1 0.02, M2 0.02, M4 0.02 | AGREE within a factor of 2, for other reasons | Mine [I]: G1 0.05 (information measured, a fifth to under a half of the encoder channel's). G2 0.05-0.10 at the popularity-light pool, at most +0.00015 at file level. M1 0.02: the project's own rate for "closed under hard labels, re-tested under distillation" was 0.054 [D, `reports/RECHECK_1114/skeptic_feasibility_value.md` l.53] before further arms of that class closed (kd08w192, kd05w224 [D, `CURRENT_STATUS.md` l.858, COVERAGE_MAP l.331]). M4 0.02 (2.4) |
| "Every CPU pre-gate that could exist is on record or was run today, all negative or empty" | **DISAGREE** | For M2, M5-M9, G3 and G4 the gates are real and negative. For G1 and G2 the NDCG gates fail their positive controls (2.1, 2.2). For G1 the powered gate is positive but small (2.3). For G2 the document says no stale view exists; one does |
| "No detectable information beyond the current file" (verdict item 4) | **DISAGREE** | LR 4.40 / 11.37 on all feeds, replicated on the unread feeds (2.3) |
| Optional C-MoE (sparse-MoE wide students) | **DISAGREE: do not run** (failure mode A) | No mechanism for either bottleneck; prior 0.02; a 0.5-1 day port into the trainer that produces the final members. Its "stage 1" is not a gate: it stops only if two correlated witnesses are both negative, so a null arm continues in about 60% of cases [I]; the project measured that 2-seed screens pass 15% of nulls and 36% of true +0.0005 [D, COVERAGE_MAP l.315]. Expected cost 8-22 GPU-h, not 3-8 |
| Optional C-Graph (density columns on the popularity-light students) | PARTLY AGREE: allowed as idle-GPU filler, with three changes | (i) Decide at FILE level with SE thresholds. As written the file clause is two sign conditions on correlated witnesses and passes a file-level null in about 40% of cases [I], while the pool clause can pass merely because a popularity proxy is given back to popularity-light students. (ii) Drop `trd_sn95_sday`: weekday / weekend PSI 0.52 [D, `reports/feat_trd_.md` V.8]; val is weekday-only, public is weekend; the project's parity rule is PSI < 0.5. (iii) Put the base-student host (leverage 0.8) in the same batch: "the fresh counters dominate there" is an argument [I], and on the popularity-light host the file cannot move by a readable amount |
| "A null on C-Graph closes the graph class" | AGREE for the lag-0 density form only | G1 is a separate row (2.3) |
| "A trained GNN is not on the list at any price" | AGREE | Reach and two-hop saturation are measured [D]; the ID table costs -0.0022 / -0.0051 [D]; private rows are colder and staler |
| "What would change the answer": a covariate with router AUC above the null; a file-level pass of a stale-view recipe | AGREE | Second trigger: its train-only half is read in 2.4 and does not discriminate |
| The independent verification of the density columns | OMISSION | `reports/feat_trd_.md` V.1 recommends against using the 20 columns ("no better than random noise features" in its time-split check) and V.8 lists per-column defects. SYNTHESIS cites the family's builder numbers only |
| Rows proposed for COVERAGE_MAP (section 6 of SYNTHESIS) | **DISAGREE with rows 1 and 2 as written** | They would record "NO SIGNAL" from lenses that read zero on known positives. Replacement text: section 6 |

### 3.2 Rejections ("do not do")

| rejected item | my position | reason |
|---|---|---|
| LightGCN / NGCF / SGL / SimGCL / graph transformers on the interaction graph | AGREE | No node for 42.8% / 45.6% of rows; edges 9-11 days old; ALS, item2vec, group EASE, RP3beta 0 to -0.0005 [D] |
| PinSage / TwHIN / OmniSage pre-training | AGREE | Item vectors are already a two-tower output; table 6 of arXiv 2508.06970 verified [W] |
| Session-graph encoders, temporal graph networks | AGREE | No timestamps in histories; GRU equals the transformer [D] |
| A GNN as a diversity member | AGREE | Stand-alone graph scorers 0.33-0.34 against members at 0.40 [D] |
| MMoE / PLE / HoME | AGREE | One task; gradient cosine +0.007..+0.014 [D] |
| STAR / PEPNet / AdaSparse / M3oE | AGREE | Scenario constant at inference [D] |
| RankMixer / TokenMixer-Large / SORT as capacity | AGREE | Parameter-per-cost device [W]; capacity arms non-positive under distillation (2-seed evidence) [D] |
| POSO-type cold gates, cold / warm experts, per-vertical, per-cluster, feed-size specialists | AGREE | Gates between pools are measured nulls [D]. The in-network form was never trained; the nearest recent paper (section 5) maps to the popularity-light view that is in both picks |
| Any further router or learned gate between pools | AGREE | 8 studies, 131,547 rules, public -0.00015 [D] |
| "Perfect switch" oracles | AGREE | Same-recipe reference +0.0092 of +0.019 [D] |
| Graph centrality, more co-visitation variants, cross-feed co-exposure | AGREE | Degree is an input; coverage 0.1-4.5% [D] |
| "Do not quote int_gnn's partial AUC 0.545" | PARTLY DISAGREE | 0.545 overstates (the 4.5 SE is not a test), but the claim behind it is right. Fair sentence: "on all val feeds the neighbour-completed scores carry conditional information on cold rows beyond the file (LR p 0.036 in z form, 0.0007 in rank form), at a fifth to under a half of the encoder score's coefficient" |

## 4. Is an idea wrongly dropped or wrongly promoted?

| idea | verdict | smallest experiment that settles it | cost |
|---|---|---|---|
| G1 neighbour-completed co-visitation for cold rows | Not wrongly dropped; wrongly described | Was: the all-feeds likelihood-ratio read. DONE (2.3): WEAK. The next step would be the trained arm on BASE students (cold rows only, full file leverage) | Done: 49 CPU-min. Trained arm: build 5 CPU-h + 1-1.5 days of parity checks + 12-31 GPU-h; P 0.05 [I]; not recommended. Re-open trigger: a variant (lit_gnn's windowed count through same-cluster neighbours) with LR >= 10.83 in z form on all val feeds; one CPU-hour with the scripts in `stageB/` |
| G2 fresh-density columns | Not wrongly dropped as a priority; wrongly described as pre-gated | The student arm itself; no CPU lens has power (2.1) | 8 seeds x 2 hosts = 24-62 GPU-h on idle GPUs, zero code, zero build; P(file gain >= +0.0003 on the private-like witnesses) 0.03 [I] |
| M4 staleness-aware gate | Not wrongly dropped | Train-only half read (2.4); the full trigger needs fine-tunes | about 5 GPU-h per arm; not recommended |
| M1 sparse-MoE students (C-MoE) | Wrongly offered, even as optional | none needed | - |
| trained GNN, gates, multi-task and per-token MoE | Correctly dropped | none needed | - |

Nothing is wrongly promoted in the main recommendation.

Rule to use if the G2 filler is run (replaces K-STALE for this arm; to be copied into `nn_x/w5/logs/E_kd_predeclared.txt` with `date -u` before the first run):
- Arms: A-light = wide popularity-light students + `--xt_add trd_kde_sday,trd_skde_sday,trd_n85_sday,trd_n95_sday`; A-base = wide base students (kd05w192) + the same columns. 8 seeds each, references = the existing students of the same seeds.
- Decision at FILE level only: the file with the arm's members in place of their references (admixture 0.2 kept) minus the reference file. PASS iff val >= 0 AND pmix-private >= +1 SE AND stale-val of the train-only branch >= +1 SE AND pooled z over val and stale-val >= 2.
- A PASS buys ONE SE-denominated public read, not an adoption. Anything else: CLOSED, and the lag-0 density form with it. Pool-level numbers are printed and have no clause.

## 5. Recent work not in the four input reports

One relevant paper is absent from all of them [W, generic search today; `grep` of `reports/` and `docs/` finds no mention]: "Warmer for Less: A Cost-Efficient Strategy for Cold-Start Recommendations at Pinterest", arXiv 2512.17277 (submitted to WWW 2026). Three parts: a residual connection for non-historical (content) features so that the ranker does not under-weight them, a score regulariser between cold and non-cold items, manifold mixup for sparse cold labels; +10% fresh-content engagement online at about 5% more parameters [W, abstract only].
- Mapping: the first part is the in-network form of the popularity-light view, which is in both picks at ensemble level (public +0.00052 / +0.00041 [D]). The second is the cold offset (closed, +0.00013 [D]). The third is a regulariser of a class that is null under distillation [D].
- It adds no candidate. It supports the document's point 7: the member of the "experts" family that pays is a second input view. That extension was read by the lead at 16:46Z: a two-view popularity-light pool, val 0.40286, expected public difference about +0.0001 [D, `CURRENT_STATUS.md` log 16:46Z].

A second hit, TAG-HGT (arXiv 2601.02381: cold nodes linked to their semantic nearest neighbours), is the mechanism of G1 again [W, search summary only].

## 6. Corrections for the record (the lead may copy them; this report edits nothing)

| where | text |
|---|---|
| COVERAGE_MAP, new row: neighbour-completed collaborative scores on cold rows | INFORMATION SMALL, NO BUILD 11-15 (CPU, all val feeds). Conditional logit over cold rows on the scored file: LR 4.40 (p 0.036) z form, 11.37 (p 0.0007) rank form; unread 11,369 feeds 3.48 / 8.56. Coefficient +0.024 against +0.116 for the encoder score on members trained without it (LR 37.5; trained worth +0.0006..+0.0008). Expected worth as a trained input +0.00004..+0.00035; trained form not run, P 0.05. NDCG residual reads of this construction are not usable. `reports/LIT_1115/CRITIC.md` 2.2-2.3 |
| COVERAGE_MAP, new row: fresh-density columns (`trd_`) as ranker inputs | NOT CLOSED, NOT READABLE ON CPU 11-15. The nested residual lens reads -0.00030 +- 0.00031 on the file, and +0.00016 +- 0.00026 for the fresh counters on a stale pool that is 0.00164 worse without them (positive control fails). In-sample LR p 0.04-0.20 on six pools. Only a student arm can read it; expected file effect at most +0.00015 on the popularity-light host. `CRITIC.md` 2.1 |
| COVERAGE_MAP, note to the fresh-exposure closure (`reports/FRESH_1115/pregate.md`; lead's log 16:46Z) | The closure stands on its AUC read (0.494-0.506 against 0.5335 for train exposure). Its nested gain (-0.00044 +- 0.00015) comes from the lens that fails the positive control above and should not be quoted as the reason |
| COVERAGE_MAP l.107 / l.111 note (cset) | pmix-private -0.00016 +- 0.00105 (SYNTHESIS 2.3) has an SE three times the effects sought: "no evidence either way", not "closure also holds on the private mix" |
| Method note for future CPU pre-gates of inputs | Run a positive control first (an input the pool is known to lack). Use the in-sample likelihood ratio, with a placebo, as the detection statistic. Do not use a slot-preserving re-order. Size the sample so that the positive control reaches p < 0.001 |

## 7. Issues, by severity

1. MAJOR (evidence, not action). Verdict items 3-5 and the table of 4.1 present two NDCG reads as negative pre-gates. Both lenses fail positive controls (2.1, 2.2), and the powered read of the first idea is positive (2.3). The two proposed COVERAGE_MAP rows would enter the record as "NO SIGNAL".
2. MEDIUM. Rule K-STALE as applied to C-Graph: no size control at file level, a pool clause that does not answer the question.
3. MEDIUM. C-Graph column list contains `trd_sn95_sday` (PSI 0.52 between weekdays and weekend); the independent verification of the family is not cited.
4. MEDIUM. C-MoE is offered with a "stage 1" that is not a gate and a cost that is not the expected cost; it has no mechanism for either bottleneck.
5. MEDIUM. "No stale-val view exists" is false for the density columns; "on disk unread" hz-family arms were readable on CPU in part.
6. LOW. The rank-residual placebo is reported for one of its two declared block sizes; the bias claim holds for one of the two columns.
7. LOW. "The pool loses 0.0033-0.0042 under private-like staleness" is the holdout figure; on val the same view costs 0.00164 +- 0.00051 [M].
8. LOW. cset "closure stands" rests on a read with SE 0.00105.
9. LOW. One relevant 2025 paper missing from the literature inputs (section 5); it changes nothing.
10. LOW. The 4,600 GPU-h budget ignores the 11-28 stop-loss; the source log of the 0.0032 margin is gone; the -0.00068 of the dense feed-forward arm is a 2-seed number.

## 8. My ranked list

1. Needed: **nothing**.
2. Optional filler, only on GPUs that stay idle and below everything in the lead's queue: the G2 density arm on both hosts under the file-level rule of section 4. Expected gain zero; P 0.03.
3. Not to be run: C-MoE; any gate or router; a trained GNN; the G1 build (re-open trigger in section 4); fine-tunes of the unread hz-family arms.

## 9. Not checked, and limits

- No model was trained and no GPU was used. Every prior is my estimate [I].
- All my reads are on val. The stale view is a simulation of private-like staleness (scenario SVL), not the test set. The popularity-light pool was not re-predicted under stale inputs.
- Positive controls: the "known worth" of the fresh counters is the pool difference S - WL, which also contains the six other stale inputs. The worth of the encoder channel rests on 2 seeds (+0.00077 +- 0.00073).
- The conversion from a likelihood-ratio coefficient to a trained-input gain (2.3) is an assumption with a wide range, not a measurement.
- The 12 likelihood-ratio tests of 2.1 are correlated and uncorrected; the 3 looks of 2.4 are disclosed.
- "40% file-level pass under a null" and "60% continuation of a null arm" are computed from the sign conditions with an assumed witness correlation, not simulated on stored pools.
- Literature: 5 sources re-read (an abstract or one table each) and 3 generic searches. No project detail was sent.
- I did not re-run the routing, cold-gate, 78-combiner or graph-reach studies; their numbers are quoted as reported. I did not read `reports/MIX_1115/`, `reports/SEG_1115/` or the full text of `reports/FRESH_1115/pregate.md`.
- GPU idleness is four `nvidia-smi` snapshots between 16:00Z and 16:58Z.
- Resources: one heavy process at a time; largest 48 min 50 s at 6.3 busy cores and 13.9 GB. The first 5-second call of `lr_B.py` ran before its thread line was patched (12-thread setting of the original; single-threaded in practice).

## 10. Files

All in `nn_x/w12/lit/critic/`:
- `DECLARATION.txt`: rules and results, 8 blocks. `RUNNING.txt`: the one long process and its PID (finished).
- `stale_trd_read.py`, `.log`, `.json`, `.time`, `delta_*.npy`: section 2.1.
- `cold_lens_pc.py`, `cold_lens_pc2.py`, `cold_lens_pc3.py` and their logs: section 2.2.
- `stageB/` (copies of the five int_gnn scripts, `s5.log`, `s5.time`, `s5_cold_val.npz`), `lr_B.py`, `lr_B_{OLD,NEW,ALL}.log`, `stageB_r1.py`, `stageB_r1.log`: section 2.3.
- `hz_unread.py`, `hz_unread.log`: section 2.4.
