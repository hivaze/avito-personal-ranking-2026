# TRICKS SWEEP 11-13: synthesis

Written 2026-11-13 04:40 UTC by the synthesiser in a finder -> merger -> skeptic -> Tier-A tester workflow. It builds on and goes past reports/RESEARCH_SWEEP_1112.md and reports/INSIGHTS_1112.md.

**Inputs**
- 63 finder candidates from six lenses: postproc 9, labels 10, preproc 13, kaggle 8, nntricks 13, metric 10.
- The merger's ranking: 12 ranked items and 45 dropped or merged entries.
- Two independent skeptic passes, giving 24 verdicts.
- Tier-A test results. **The results JSON was empty, and `nn_x/tricks/` did not exist at 04:37 UTC (`date -u`), so no Tier-A test ran in this sweep.** §2 therefore reports the pre-existing evidence that answers each Tier-A item and the reduced closure test the skeptics allow, if any.

**Bottom line**
- No item in this sweep is an adoption candidate.
- Both skeptics refuted all 12 ranked items (24/24) as unable to move private. Every one is a near-copy or cold/warm-trade lever with a prior below the MDE (0.0012 val / 0.0006 holdout), and near-copies transfer to public/private at about 0.1-0.3x.
- The one generational-type idea, RECYCLE, has a specific training flaw: its pass-1 scores are in-sample on train.
- G3 stays pick 1. Expected private change from this sweep is about 0.
- What remains is a short GPU-4 queue for idle lanes only (§3), with the skeptics' fixes built in.

**Update 11-13 ~05:00 UTC (§2b).** Six pre-declared closure probes followed the critique.

Closed:
- RECYCLE v2 / head-only cascade: the top-8 headroom share is 0.29 val / 0.16 holdout, against >= 0.40.
- HDROP-ALL: TTA Δ is -0.00013 at corr 0.9957.
- FUTWIN and the whole category-intent family, including PFD: the oracle gives +0.0028, 0.09x the label echo.

Not closed:
- RDROP-CD and SAM-CD: the dropout KL is 0.011 nats, against a 1e-3 bar.

Done:
- HOLD-CLEAN Step 2: E is out-of-format, 6/10, so future arms get a dE/dL info column.

Pending:
- LATE-DROPOUT: the 10-ep pair waits for the pop-free runs to finish.

None of this changes the picks.

---

## 1. Recap: what was closed before, and what this sweep adds

### 1.1 Closed before 11-13 (condensed from reports/COVERAGE_MAP.md)

| category | closed (status, key number) |
|---|---|
| Optimisation / schedule | Plateau weight averaging (7-member ens -0.0025 val / -0.0016 hold); stage-1 EMA; longer plateau (77.5k vs 60k -0.0011); cooldown mix 0.3 (collapse), m0 vs m02 (coldpos -0.0023, z -4.5), aa5, cooldown length cf4/e3/s2c2; KD born-again (+0.0006 n.s.); 2-ep seed families (corr 0.978). Adopted: Muon + WSD, m02, 10 ep. |
| Regularisation | Label smoothing / anchor (MLP proxy); uniform xt dropout; age-conditioned input dropout (hold -0.0019); r7 G6-G9: d0 -0.0012, d20 -0.0002, resdrop0 +0.0003 / +0.0004 null at 2 ep, adrop0 FAIL. |
| Architecture / capacity | FoX, QK-norm/SSMax, MoE, PLE, TabM, propensity head, semantic IDs (all ±0.0005); d256, heads 8, blocks 3 (2 ep +0.0001 / +0.0004; 10 ep +0.0003 / -0.0004); retr_m 32 (10 ep -0.0002 / -0.0003); retr32u (-0.0003 / -0.0003); xrel_more (10 ep -0.0019 / -0.0007, pmix -0.0032); every component removal (worth +0.0014..+0.0034 each; sim_bias and domain INCONCLUSIVE). |
| Inputs | Train-row pre_* (leak); hzcv / ixh / ud / causal popularity; *_id_contrib banned; external data (±0.0003); item exploration prior (union +0.0004, z 1.4). |
| Losses / aux | Focal, Lambda, ListMLE, ApproxNDCG, NeuralNDCG, BCE, RankNet, label smoothing, PL-over-positives; K9 tau 30 / PL top-20 / aux_pos 1.0; AUXF; aux head as a member (W6: rho 0.03, gap 0.094); intent head; TRUNC-PROD / TRUNC2 (GBDT proxy +0.0022..+0.0031 -> NN -0.0005); AGE-PROD. |
| Fine-tune | w_ft != 0.5 (nested +0.0002, z 0.96); refit on full val; FT bagging; Optuna FT retune (CONFIRM +0.00055, z 1.3); pseudo-labels; LP-FT, SAM in fta (-0.00014); valcd (= FT's value). |
| Ensembling | GBDT members, stackers, rank/RRF fusion, segment/routing weights; pre_final term (-0.0008); W25 (rho 0.858, nested +0.0002 / hold -0.0001); STUDY C aggregation geometry (±0.0003). |
| Shift / staleness | Cold offsets / gates / routers / MoE by coldness (oracle +0.025, learned 0); coverage-weighted val; later-history signal (banned, test-null); hzcd (val +0.0003, hold -0.0005 z -2.8, stale +0.0011: cold/warm trade); cooldown-data arms hcap (+0.00007), rec33, rho70. |
| Backbones / external recipes | LLM soft-token rankers 0.6B-9B (-0.0038..-0.0053); pretrained ID encoder (R3/P8 gate FAIL); 2025-winners recipe (W25). |

### 1.2 What this sweep adds

1. **Closures by existing evidence, at zero cost.**
   - DOMAIN-DIAG (INSIGHTS_1112 item 7, "never run") is already answered by the W6 side observation: main head under the production token +0.00011 ± 0.00017, so XOFF/EXAM stay closed.
   - The metric-fact screen confirms that monotone, per-feed, tie and diversity tricks are no-ops or already measured.
   - Six post-hoc combiners (risk-averse pooling, SML weights, head-weighted z, segment fusion, FT/TO mixing variants, HEAD-ARB) are bounded by the measured infinite-member headroom (+0.0002 val / ~0 holdout). None can reach the +0.0010 near-copy adoption bar.
2. **Three new mechanism-level closures** that are not in COVERAGE_MAP:
   - FN-PU: later-interacted negatives are true negatives under the eval labels.
   - XUSER-RIM: the additive concat key reduces to item-content kNN, which is the measured null.
   - RECYCLE as specified: pass-1 scores are in-sample on train.
3. **A process lesson** (skeptic 1 and 2, HOLD-CLEAN). Never pick the judge after seeing which arms it flips. A holdout split may be adopted only on label-free evidence, before any arm delta is computed, and then only as an information column for future arms.
4. **A short idle-lane queue (§3)** of regulariser and second-pass arms. Each is restated as a single decisive 10-ep pair with holdout > +2 SE and a diversity clause, and each has a CPU pre-probe that can close it before any GPU spend.
5. **Dependencies on running jobs** (re-entry triggers, §4):
   - ANCHOR-REPLAY waits on the wave-6 m04 readout.
   - MetaBalance waits on aa0 / ap1.
   - POP-REL / MODDROP-POP / HZCOND / FPCD wait on the pop-free family and on a label-free holdout-format check.

---

## 2. Tier-A results

No Tier-A test was executed. At 04:40 UTC there was no `nn_x/tricks/*/RESULT.txt`, and the results input was `[]`. The closure probes run after this critique are in §2b. The table records the pre-existing evidence that settles each item and the skeptic-approved residual test.

| trick | val Δ ± SE | holdout Δ ± SE | pmix | corr to G3 | verdict |
|---|---|---|---|---|---|
| HOLD-CLEAN (Sep 6-8 vs Sep 9-19 holdout judge) | not run; existing hzcd: member val +0.00026 (z 0.8) | not run; existing hzcd late20% (Sep 17-19) -0.00031 ± 0.00038 @last, -0.00008 ± 0.00042 selected; full -0.00052 ± 0.00019 | hzcd +0.00048 (z 0.9) | n/a (judge change) | **REFUTED as specified** (outcome-chosen judge; P(false flip) ~0.1 over 6 correlated arms; L cuts power ~17%). Allowed residual: label-free Step 2 only (pre2-NaN share, LUO zero rate and log1p LUO for E/L/val/test), then dE/dL as an info column for future arms only. |
| PRODDIR / DOMAIN-DIAG (inference-time production-direction removal; K9 logit adjustment) | **answered by existing data**: main_prod vs main_rl +0.00011 ± 0.00017 (12 members); all-cold-positive feeds +0.0026 ± 0.0004, other feeds ~-0.0007 | n/a | n/a | main_prod vs main_rl 0.9982 | **CLOSED (pure cold/warm trade)**. Arm A only rescales one measured axis. Arm B (aux_rl, rho 0.006 to main, -0.31 with LUO) is a hand-made popularity/cold offset (closed: +0.0003 ± 0.0003). DOMAIN-DIAG recorded as \|Δ\| < 1 SE, so XOFF / EXAM stay closed. |
| XUSER-RIM (cross-user labelled kNN) | not run | not run | - | - | **CLOSED by mechanism**. The inner product on concat(αû, ĉ) is additive (αu·u' + c·c'), so neighbours are dominated by item similarity, which makes f1 roughly a content-kNN label mean (k_ audit null) and an item exploration prior (closed, union +0.0004, z 1.4). u_ wAUC 0.511. The holdout bank also overlaps holdout days. Reopen only with a two-stage key (similar users, then items within their impressions), an as-of bank, one fixed w = 0.05, and a pre-check that neighbour sets differ from item-kNN. |
| FUTWIN Step A (future-window intent oracle) | not run | not run | - | - | **Unfunded beyond a closure readout.** The oracle is dominated by the category echo of this feed's click. The gate must add a **label-echo control** (an aux from the current positive's category) next to u_past. Step B only if the gap >= 24 h margin over both controls is >= +0.01, a frozen-trunk linear probe beats u_past by the same margin, and a u_past-target control arm is included. |
| PICK-EMAX (pick 2 by E[max]) | 0 by construction | - | - | - | **Not a work item.** Claimed gain +0.00003..+0.0001 is far below the public paired SE (0.0005-0.0007), and mu_hat is winner's-cursed. At final-pick time: confirm on ods.ai that the best of 2 selected files counts (REVIEW_SYNTHESIS:321 assumes it). Then compute Clark's gain as a tie-break, with mu_hat shrunk x0.1-0.3, and switch from R6 only if the gain difference > 1 bootstrap SE. |
| FN-PU Step A (false-negative gate) | not run | not run | - | - | **CLOSED by mechanism.** Items the user interacts with later are negatives in val/test labels too, so CE push-down is metric-aligned. Prevalence is 0.402% of negatives, ~0.3 per affected feed on 18% of feeds, i.e. ~0.05 per average feed. Step B never. Step A is allowed only as an error-analysis fact. |

Interpretation: two items (PRODDIR, FN-PU) are settled by data or arithmetic already in hand. One (XUSER-RIM) is settled by the form of its key. Three (HOLD-CLEAN, FUTWIN, PICK-EMAX) shrink to label-free or final-pick-time checks of no direct value. Nothing reaches a val or holdout readout worth a gate run.

---

## 2b. Closure probes run after the critique (11-13)

Added 2026-11-13 ~05:00 UTC by the editor.

**Why these probes ran.** The skeptics had defaulted to "refuted". That is the right default for GPU spend, but it is wrong for cheap CPU readouts (lead decision, CURRENT_STATUS 04:45). So the lead launched the closure probes the skeptics themselves allowed.

**How they ran.**
- Each rule was written with `date -u` into `nn_x/tricks/<item>/PREDECLARED.txt` before any number was computed.
- Every probe is a closure or information readout. Nothing was adopted, nothing was submitted, and no file in subs/, preds/, nn/, nn_x/r6, nn_x/w5 or nn_x/abl was modified. The only exception is the LATE-DROPOUT log lines appended to nn_x/abl/RUNNING.txt.
- Five probes have finished: three on CPU, and two on GPU N for 1-2 minutes each with no training. The LATE-DROPOUT 10-ep pair is set up and waiting for GPUs.
- The full text of each probe is in `nn_x/tricks/<item>/RESULT.txt`.

| item | pre-declared (UTC) | decisive numbers | verdict |
|---|---|---|---|
| RECYCLE Step 0 (§3 old #2) | 04:44:58 | Top-8 share of the headroom (headshare_8): 0.2937 val / 0.1640 holdout; the bar is >= 0.40 on both. Pair AUC within the top-8 (AUC_head8): 0.616 / 0.578; the bar is < 0.75. | **CLOSED**: RECYCLE v2 and the head-only cascade |
| RDROP-CD KL probe (§3 old #4, with SAM-CD #7) | 04:45:42 | Two-pass dropout KL: 0.01111 nats at the m02 cooldown end, 0.00526 at the cooldown start, 0.01121 on the e10 base. The bar is 1e-3 nats. | **NOT CLOSED**: RDROP-CD and SAM-CD stay queued |
| HDROP-ALL T2 replay (§3 old #3) | 04:46:10 | TTA(4) gain at rate 0.1: D(0.1) = -0.00013 ± 0.00036. TTA vs full-history correlation: C(0.1) = 0.9957. D(0.2) = -0.00098 ± 0.00041. | **CLOSED** (branch C) |
| FUTWIN Step A (§2, §3 old #8) | 04:45:44 | Future-window oracle u_fut: +0.00275 ± 0.00068 (the bar is +0.005). Its ratio to the label echo is 0.093 (the bar is 0.30). | **CLOSED for good**: the category-intent family, including PFD |
| HOLD-CLEAN Step 2 (label-free) | 04:45:51 | 6 of 10 statistics count; 5 of 10 without the by-construction S8 | "E is out-of-format", so FUTURE arms get a dE/dL info column only |
| LATE-DROPOUT 10-ep pair (§3 #1) | 04:50:54 (+ addendum 04:51:13) | No arm number exists yet. CPU flag probe PASS 9/9; SE_gate 0.00076575 | **PENDING** (§3 #1 has the status) |

### 2b.1 RECYCLE Step 0: where the remaining G3 error lives (CPU)
- **Pre-declared:** 2026-11-13 04:44:58 UTC (`nn_x/tricks/recycle_step0/PREDECLARED.txt`).
- **Rule:** the head-recycle family (RECYCLE v2, head-only cascade) stays QUEUED only if both conditions hold on val G3 (ens_G3_final_val) AND on holdout G3 (the 20-member z-mean):
  - headshare_8 >= 0.40, where headshare_8 = (top-8 perfect-reorder ceiling - base NDCG) / (1 - base NDCG);
  - pooled within-feed pos-vs-neg AUC over pairs with both items in the top-8 (ties 0.5) < 0.75.

  Otherwise the family is CLOSED. Everything below is information only: the train-only pool (ens_G3_to_val), M buckets, Tue/Thu, K != 8, feed-macro AUC and best-positive ranks. The §3 "+0.01 top-8 ceiling" drop clause is reported only and can only close.
- **Sanity:** val G3 0.40032 ± 0.00205 and holdout G3 0.32379 ± 0.00116 both reproduce the records. G3 TO val is 0.40035.

**Top-K perfect-reorder ceiling: headshare (gain ± SE)**

| K | val G3 (headroom 0.59968) | holdout G3 (headroom 0.67621) |
|---|---|---|
| 3 | 0.0954 (+0.05721 ± 0.00118) | 0.0436 (+0.02945 ± 0.00057) |
| 5 | 0.1882 (+0.11288 ± 0.00168) | 0.0926 (+0.06260 ± 0.00088) |
| **8** | **0.2937 ± 0.0037** (+0.17613 ± 0.00206) | **0.1640 ± 0.0018** (+0.11087 ± 0.00118) |
| 16 | 0.4880 (+0.29262 ± 0.00244) | 0.3210 (+0.21706 ± 0.00159) |
| 32 | 0.6884 (+0.41285 ± 0.00251) | 0.5360 (+0.36248 ± 0.00181) |

**Feeds with no positive in the top-8:** 51.2% of val feeds, holding 66.0% of the headroom; 66.7% of holdout feeds, holding 77.1%.

**Val headshare_8 by slice**

| slice | headshare_8 | NDCG |
|---|---|---|
| M<50 | 0.5706 ± 0.0073 | 0.51646 |
| M>=50 | 0.1577 ± 0.0034 | 0.32009 |
| Tue | 0.2987 | - |
| Thu | 0.2882 | - |

**Pooled pair AUC (no ties in any population)**

| population | head8 | head16 | all pairs |
|---|---|---|---|
| val | **0.6162 ± 0.0038** (macro 0.6209; 52,037 pairs in 7,041 feeds) | 0.6265 | 0.7362 ± 0.0025 |
| val M<50 | 0.6358 | - | 0.7267 |
| val M>=50 | 0.5866 | - | 0.7369 |
| holdout | **0.5780 ± 0.0032** (77,459 pairs in 10,005 feeds) | 0.5894 | 0.7430 ± 0.0013 |

**Rank shares of positives** (buckets r1 / r2-3 / r4-8 / r9-16 / r>16)

| population | shares | median rank |
|---|---|---|
| val, all 21,983 positives | .084 / .111 / .170 / .158 / .476 | 15 |
| val, M<50 | .169 / .218 / .295 / .200 / .119 | 5 |
| val, M>=50 | .043 / .060 / .109 / .138 / .650 | 28 |
| holdout, all positives | .044 / .059 / .116 / .139 / .642 | 28 |

**Train-only pool (info).** ens_G3_to_val is practically identical to G3: headshare_8 0.2935, AUC_head8 0.6163, and the same rank shares.

**§3 secondary clause.** The top-8 ceiling gain is +0.1761 val and +0.1109 holdout. Both are >= +0.01, so that clause alone would not have dropped the arm.

**Verdict: CLOSED.** headshare_8 fails on both populations (0.294 val, 0.164 holdout, against a bar of 0.40). The AUC_head8 leg holds everywhere, but in-head AUC is depressed by range restriction, so it was a weak condition. The closure therefore means the error sits in the tail, not that the head is well ordered. Inside the top-8, the G3 order is only weakly better than random. Even so, a perfect top-8 re-order reaches only 29% of the val headroom and 16% of the holdout headroom, and most of the headroom is in feeds with no positive in the top-8 at all.

**Side note.** The sweep's "head misordering is 49% of NDCG-weighted error" matches K = 16 (0.488 val), not K = 8.

**What it closes.** RECYCLE v2 / head-only cascade (old §3 #2), together with its GPU-4 arm and the depth-matched blocks3 control. Neither is to be launched.

**Re-entry** (§4.2) is limited to two designs:
- A non-head-only design, such as a full-list score-conditioned pass.
- An M<50-specific head arm. Val M<50 alone meets both conditions (0.571 / 0.636). Such an arm needs its own pre-declared rule and a non-holdout judge, because the holdout is M>=50 only.

### 2b.2 RDROP-CD KL probe: is dropout inert at the cooldown end? (GPU N, no training)
- **Pre-declared:** 2026-11-13 04:45:42 UTC (`nn_x/tricks/rdrop_cd_kl/PREDECLARED.txt`). Post-hoc instrument checks were declared at 04:48:29 UTC in POSTHOC_CHECKS.txt; they are info only.
- **Rule:**
  - The statistic S is the mean over feeds with M >= 2 of the per-feed symmetric KL, 0.5[KL(p1‖p2) + KL(p2‖p1)].
  - p1 and p2 come from two train-mode, no-grad, bf16 forwards with independent dropout masks. Every nn.Dropout and SDPA attention site is at p = 0.1, with drop_cd off. p is the softmax over the feed's valid candidates.
  - Data: the first 64 regular training batches (non-holdout train feeds, rng 1113), using the nn_x/abl path with the training domain token.
  - Decision set: (a) w5_cd_m02ref_r101.pt, (b) its @last, (c) the cooldown start v17_pre2s_s101@77500, (d) w5_e10_base_s951.
  - If S < 1e-3 nats on ALL four, RDROP-CD, SAM-CD and any two-pass consistency regulariser are CLOSED as inert. If S >= 1e-3 on ANY of them, they stay QUEUED with the measured KL as the prior.
- **Sample:** 64 batches, 22,248 feeds, M 5-288. Every feed has M >= 2. SEs are batch-cluster SEs.

**Train vs train (two dropout passes)**

| checkpoint | S, nats (SE) | median | p90 | Spearman mean | top-1 changed |
|---|---|---|---|---|---|
| (a) = (b) m02ref_r101 | **0.011109** (0.000065) | 0.009622 | 0.018730 | 0.9606 | 24.85% |
| (c) cooldown start @77500 | 0.005256 (0.000031) | 0.004562 | 0.008540 | 0.9733 | 20.67% |
| (d) e10_base_s951 | 0.011209 (0.000063) | 0.009691 | 0.018866 | 0.9601 | 24.94% |
| r102 (info) | 0.010981 (0.000053) | 0.009464 | 0.018610 | 0.9610 | 24.87% |

**Eval vs one train pass**

| checkpoint | symKL mean | top-1 changed |
|---|---|---|
| (a) | 0.006125 | 18.29% |
| (c) | 0.002884 | 15.31% |
| (d) | 0.006167 | 18.14% |

**Scale references**
- Seed r101 vs r102 in eval mode: 0.032600 nats, Spearman 0.916, top-1 changed 36.77%.
- Listwise CE on the same batches: 3.2019 eval vs 3.2102 train for (a).

**Post-hoc checks on (a), info only**
- Two eval passes give KL exactly 0.
- fp32 train vs train: 0.010545.
- A bf16 rerun is bit-identical.
- A replicate with batch seed 2026 and torch seed 7 gives 0.011068.

**Verdict: NOT CLOSED.**
- S is above 1e-3 on all four decision checkpoints: 11x at the m02 end, and more than 160 SE above the threshold everywhere.
- The sweep expected KL ≈ 0 from the MC-mean vs deterministic corr of 0.99998. That does not hold for single-pass pairs: averaging many masks removes exactly the per-pass noise that R-Drop penalises.
- Dropout KL doubles across the cooldown, from 0.0053 to 0.0111, as the model sharpens.
- At alpha 1.0 the R-Drop term is about 0.011 nats per feed, about 0.35% of CE. "Not inert" means only that the loss term is non-zero. **The benefit prior is unchanged (P ~0.05).**

**What stays queued.**
- RDROP-CD (§3 #2) goes on to its training design.
- SAM-CD (§3 #5) stays only through the bundling rule. Its own preconditions are untouched: a published Muon-compatible SAM form, and rho calibrated on the CPU.

### 2b.3 HDROP-ALL: history-dropout TTA replay on the 10-epoch members (GPU N, no training)
- **Pre-declared:** 2026-11-13 04:46:10 UTC (`nn_x/tricks/HDROP-ALL/PREDECLARED.txt`).
- **Rule:** the probe is evaluated on val at rate 0.1, with 4-mask TTA (the mean of raw logits), pooled over w5_e10_base_s951 and s952.
  - D is the per-feed paired NDCG delta, TTA minus full history.
  - C is the mean member within-feed z correlation, TTA vs full history.
  - The branches are checked in order:
    - **A:** |D(0.1)| <= 0.0003 AND C(0.1) >= 0.999. The model is already robust: CLOSED.
    - **B:** D(0.1) > +0.0005 or D(0.2) > +0.0005. QUEUED with that number as its prior.
    - **C:** otherwise, record and close.
  - Holdout, single-pass deltas, the 2-member ensemble and all slices are information only.
- **Mask:** each valid event among the first min(H, 256) is dropped with probability p. The newest event is always kept. The survivors are compacted newest-first, and all in-model history inputs are recomputed. The xt inputs are left unchanged, as in T2.
- **Replay check:** the full-history replay equals the stored preds bit for bit (max|d| 0.0). Full-history NDCG: val 0.39632 / 0.39691, holdout 0.32084 / 0.32000.

**Deltas, pooled over 2 members**

| mode | val Δ ± SE | val corr | holdout Δ ± SE |
|---|---|---|---|
| **rate 0.1, TTA(4)** | **-0.00013 ± 0.00036** (z -0.35) | **C = 0.995741** | +0.00012 ± 0.00018 |
| rate 0.1, single pass | -0.00077 ± 0.00032 (z -2.40) | 0.992571 | -0.00056 ± 0.00016 (z -3.47) |
| rate 0.2, TTA(4) | -0.00098 ± 0.00041 (z -2.37) | 0.993669 | -0.00014 ± 0.00022 |
| rate 0.2, single pass | -0.00170 ± 0.00037 (z -4.58) | 0.988844 | -0.00099 ± 0.00020 (z -4.96) |

**Other information**
- Per member, val rate-0.1 TTA: s951 +0.00002 ± 0.00050, s952 -0.00027 ± 0.00049.
- 2-member ensemble at rate 0.1: val -0.00066 ± 0.00042, holdout -0.00010 ± 0.00021.
- Rate-0.1 TTA slices, all within 2 SE:

| slice | Δ ± SE |
|---|---|
| M<50 | -0.00040 ± 0.00070 |
| M>=50 | +0.00006 ± 0.00038 |
| Tue | +0.00050 ± 0.00051 |
| Thu | -0.00082 ± 0.00051 |

**Verdict: branch C, recorded and CLOSED.**
- |D(0.1)| = 0.00013 meets the 0.0003 bound, but C(0.1) = 0.9957 is below 0.999, so branch A does not apply.
- Neither rate exceeds +0.0005, so branch B does not apply.
- Thinning the history only removes information: a single pass loses 0.0008 val / 0.0006 holdout at 10%. Averaging 4 masks only buys that back to about 0 at 10%, and does not even do that at 20%.
- No train-vs-eval depth shift shows up that dropout would close. The r6 T2 per-member gain of +0.00036 at 2 epochs has disappeared at 10 epochs.
- The sweep's earlier wording also gives close: the per-member TTA gap is +0.00002 / -0.00027, below +0.0002.

**What it closes.** HDROP-ALL (old §3 #3). The 2 × 4.5 GPU-h pair is no longer queued. This is consistent with SCTTA and T2-TTA, which are already closed in §4.1.

### 2b.4 FUTWIN Step A: future-window category-intent oracle vs controls (CPU)
- **Pre-declared:** 2026-11-13 04:45:44 UTC (`nn_x/tricks/FUTWIN/PREDECLARED.txt`).
- **Rule:**
  - Score: s = zf(G3 TO holdout pool) + 0.1·zf(u), where zf is the per-feed z. It is scored by paired per-feed NDCG on S.
  - S = the holdout feeds whose non-zero user has a later train feed at least 24 h away; b is the earliest such feed.
  - The future window W is b's history minus a's history, as a multiset difference on (item_id, event_type).
  - FUTWIN stays DEFERRED only if all three hold:
    1. Δ(u_fut) >= +0.005 with z >= 3.
    2. Δ(u_fut) - Δ(u_past) >= +0.003.
    3. Δ(u_fut) >= 0.30·Δ(u_echo).

    Otherwise the category-intent aux family is CLOSED for good. Other weights and slices are information only.
- **Set:**
  - The holdout rebuilt by the rng-12345 rule reproduces holdout_meta and holdout_y. The G3 pool scores 0.32379 on all 30K.
  - S = 4,476 feeds (14.9%), with a median gap of 97.6 h. G3 scores 0.32204 on S.
  - |W| has median 68. In 93.1% of S feeds, W contains an event on one of the feed's own positive items.

**Deltas on S at w = 0.1**

| signal | Δ ± SE (z) |
|---|---|
| u_fut | **+0.00275 ± 0.00068** (z +4.02) |
| u_past | -0.00123 ± 0.00055 (z -2.24) |
| u_echo | +0.02963 ± 0.00113 (z +26.1) |
| fut - past | +0.00398 ± 0.00064 (z 6.2) |
| fut - echo | -0.02688 ± 0.00117 |

- The ratio fut/echo is **0.093**.
- Info: u_fut without events on the feed's own positive items gives **+0.00072 ± 0.00067** (z 1.07).
- Info: at w = 0.3, u_fut gives +0.00522 ± 0.00130, against +0.09036 for the echo.
- Within-feed corr(u_fut, u_past) is 0.728 pooled.

**Verdict: CLOSED for good.** Conditions (1) and (3) fail; only (2) passes.
- (1) +0.00275 is below +0.005, although z = 4.0.
- (3) The echo ratio is 0.093, against 0.30 required.
- Even an oracle that sees the user's next ≥ 24 h window adds only +0.0028 on the 15% of holdout feeds that have one. Most of that is the echo of this feed's own click: once it is removed, +0.0007 (z 1.1) is left.
- The past-history category share already hurts G3 (-0.0012).
- The sweep's stricter Step-B bar (margin >= +0.01 over both controls) also fails by a wide margin: u_fut - u_echo = -0.027.
- **Validity caveat:** 22.7% of window events sit deeper than b's newest |W| events, from reordering or scroll-out. This cannot overturn a 3x miss on condition (3).

**What it closes.**
- FUTWIN Step A and Step B (old §3 #8).
- PFD (the old §4.2 row, whose trigger was this gate).
- Future-window or dense category-intent aux in general.

### 2b.5 HOLD-CLEAN Step 2: label-free format readout of the holdout, E = Sep 6-8 vs L = Sep 9-19 MSK (CPU)
- **Pre-declared:** 2026-11-13 04:45:51 UTC (`nn_x/tricks/hold_clean/PREDECLARED.txt`).
- **Rule:**
  - There are ten primary label-free statistics:
    - S1 mean log M
    - S2 mean log1p H
    - S3 share of user_id 0
    - S4 per-feed category entropy
    - S5 share of rows with any f1..f5 NaN or 0
    - S6 share of candidates seen in the feed's history
    - S7 share of listings younger than 24 h
    - S8 share of pre2_fe NaN
    - S9 share of LUO == 0
    - S10 mean log1p LUO
  - A statistic counts as out-of-format when all three hold: |E - val50| > |L - val50|, |E - test50| > |L - test50|, and |E - L| > 2 SE.
  - If at least 3 of the 10 count, record "E is out-of-format" and recommend a pre-declared dE/dL information column for FUTURE arm readouts, and nothing else.
  - The comparators are format-matched M>=50 subsets of val and test.
- **Setup:**
  - The holdout index reproduces exactly: 30,000 feeds, 4,630,966 rows. G3 scores 0.32379.
  - E = 7,823 feeds; L = 22,177 feeds. There is no Sep 15-16 data.

**Statistics that count** (E / L / E-L ± SE / val50 / test50)

| statistic | E | L | E-L ± SE | val50 | test50 |
|---|---|---|---|---|---|
| S4 category entropy | 1.9404 | 1.9560 | -0.0156 ± 0.0062 | 1.9920 | 2.0228 |
| S5 f2 or f5 == 0 (no NaN anywhere) | .1519 | .1568 | -.0048 ± .0016 | .1665 | .1597 |
| S7 age < 24 h | .1220 | .1185 | +.0035 ± .0012 | .1139 | .1074 |
| S8 pre2 NaN (by construction: the pre2 windows start Sep 9) | 1.0 | 0 | +1.0 | 0 | 0 |
| S9 LUO zero rate | .1786 | .2352 | -.0566 ± .0011 | .3791 | .4465 |
| S10 log1p LUO | 2.293 | 2.027 | +.265 ± .007 | 1.554 | 1.380 |

**Statistics that do not count**
- S1, S2 and S3 fail the guard; on S2 and S3, E is closer to val and test than L is.
- S6 is exactly 0 in every split, because no candidate ever appears in the same feed's history.

**Information only**
- p_random is identical on E and L (.970 vs .970).
- G3 scores 0.32513 ± 0.00229 on E and 0.32332 ± 0.00134 on L: E-L = +0.00181 ± 0.00266 (0.7 SE).
- The SD of member E-L is 0.00084; the correlation of member deviations between E and L is +0.35.
- A member-vs-member delta has about 2.0x the full-holdout SE on E and about 1.15x on L.

**Verdict: "E is out-of-format"**, with 6 of 10 counting (5 without S8, 3 without S8-S10).
- The main drivers are the pre2 flag and LUO popularity inflation on Sep 6-8. The smaller shifts in S4, S5 and S7 are ones where L is already off val.
- G3 itself does not differ between E and L.

**Recommendation, and nothing else.** FUTURE arm readouts carry a pre-declared dE/dL information column, each with its own SE. The full holdout stays the judge wherever it was pre-declared, and no closed arm is recomputed or re-judged.

**What it closes and what it feeds.**
- It closes the HOLD-CLEAN Step 2 residual. HOLD-CLEAN as a judge stays REFUTED (process).
- It supplies the label-free half of the §4.2 POP-REL / MODDROP-POP / HZCOND / FPCD / PRE2T trigger. The other half, the pop-free family Q1 readout, is still open.

### 2b.6 LATE-DROPOUT 10-epoch pair: set up, PENDING
- **Pre-declared:** 2026-11-13 04:50:54 UTC, with the kill-gate addendum at 04:51:13 UTC (`nn_x/tricks/late_dropout/PREDECLARED.txt`). Both were written before any arm run existed.
- **Arm:** abl_e10_latedrop_s951 / s952, which is E_BASE + HYG + `--drop 0 --attn_drop 0.1 --drop_cd 0.1` on the nn_x/abl harness. It is paired per seed against w5_e10_base_s951 / s952.
- **CPU flag probe (PASS 9/9):**
  - The parser differs from the base only in drop, attn_drop and drop_cd.
  - Before t_cool = 78,680 of 98,350 steps, all 11 nn.Dropout sites are at p = 0 (identity) and the 5 attention-probability sites are at 0.1.
  - From t_cool on, every site is at 0.1 (measured zeroed share 0.097-0.102).
  - Eval mode applies no dropout.
- **Final rule:** read once, on the holdout-selected checkpoints (step >= 12000). Deltas are per feed, pooled over the 2 seeds. PASS needs all five:
  1. val Δ > 0.
  2. holdout Δ > +2 SE.
  3. No val slice z < -2 (M<50, M>=50, Tue, Thu).
  4. pmix (private weights) >= -1 SE.
  5. rho_x <= rho_bb - 0.005, measured pool-level with disjoint seeds:
     - rho_x = corr(LD {951,952}, base {953,954});
     - rho_bb = corr(base {951,952}, base {953,954}) = 0.97353, so the clause needs rho_x <= 0.96853.

  Anything else is FAIL. A run that dies without its json gives INCOMPLETE.
- **Kill gate (the only interim look):** evaluated at step 60,000 on both seeds.
  - SE_gate = 0.00076575. This is the RMS of base seed-pair holdout differences over 15 pairs × 16 plateau evals.
  - KILL iff LD_s951 < 0.31506035 AND LD_s952 < 0.31586547.
  - Step 60,000 is mid-plateau, not the plateau end. This was disclosed before any run.
- **Readout check:** the readout script reproduces the registered blocks3 10-ep numbers exactly.
- The status (launch condition, waiter PIDs) is kept current in §3 #1.

---

## 3. Tier-B/C queue (GPU N only, idle lanes only; ranked)

**Updated 11-13 ~05:00 UTC after §2b.** The CPU and GPU-minute pre-probes have run:
- RECYCLE v2 (old #2), HDROP-ALL (old #3) and FUTWIN Step B / PFD (old #8) are CLOSED and moved to §4.1 with their evidence.
- RDROP-CD passed its KL gate.
- LATE-DROPOUT is set up and pending.

The surviving five items are re-ranked below in their original relative order.

Common rules for every item below:
- GPU: `CUDA_VISIBLE_DEVICES=N` only. Before each launch, `nvidia-smi` must show < 4 compute processes and >= 60 GB free on GPU N. The exception is LATE-DROPOUT: by the lead's decision (CURRENT_STATUS 04:45) it runs on GPUs N once the pop-free runs end, under the same per-GPU < 4 processes / >= 60 GB check.
- CPU: 12 threads (`OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 POLARS_MAX_THREADS=12`).
- Code and logs go in `nn_x/tricks/<trick>/`, as a copy of the nn_x/abl or w5 CD harness. Every new flag defaults off, and a test_abl_cpu-style test proves same-seed logits/loss/grad bit-identity with the flag off.
- Pre-declare the rule with `date -u` before launch. No interim reads count.
- **Skeptic-consolidated success bar** (stricter than the original E rule; it replaces +1 SE because a 4-member +1 SE holdout bar has a ~16% one-sided false-pass rate):
  - pooled member val > 0;
  - pooled holdout > **+2 SE**;
  - no slice (M<50, M>=50, Tue, Thu) < -2 SE;
  - pmix >= -1 SE;
  - **diversity:** member seed corr to base rho_x <= rho_bb - 0.005.
- A pass yields pool material only. It reaches a pick only through the decide6 holdout-led pool rule vs G3 plus presub_gate. Replacing pick 1 needs holdout >= +2 SE vs G3 and val >= 0.
- Honest overall P(any item below produces a generational change, >= +0.002 val at corr < 0.99): **< 0.05**.

| # | item | status after §2b | mechanism | exact design | pre-declarable rule | cost | honest prior |
|---|---|---|---|---|---|---|---|
| 1 | **LATE-DROPOUT** (single pair) | **SET UP, PENDING LAUNCH** (§2b.6; status below the table). Rule pre-declared 04:50:54 UTC; CPU flag probe PASS 9/9; no arm number yet. | Plateau is gradient-noise dominated (B_simple 4,900-9,400 feeds vs 317-512 per step), so stable-phase dropout may only add noise. The cooldown keeps 0.1 as the memorisation brake. Counter-argument (skeptic 2): the 10-ep plateau already saturates by ~6 ep, and dropout 0 may accelerate the plateau overfit. resdrop0 per-seed val was -0.00030 / +0.00091. | No new code: nn_x/abl harness, E_BASE + HYG + `--drop 0 --attn_drop 0.1 --drop_cd 0.1` (verified: plateau nn.Dropout 0 and attention-probability 0.1; every site 0.1 from t_cool = 78,680 of 98,350). Variants abl_e10_latedrop_s951 / s952 vs the existing w5_e10_base_s951 / s952. | **Kill gate at step 60,000** (mid-plateau, disclosed): KILL iff LD_s951 < 0.31506035 AND LD_s952 < 0.31586547 (base - 2·SE_gate, SE_gate 0.00076575). **Final:** the consolidated bar on selected checkpoints, with the diversity clause at pool level: rho_x = corr(LD {951,952}, base {953,954}) <= rho_bb - 0.005 = 0.96853. s953/954 only after a PASS, under a new declaration. | ~2 x 4.5 GPU-h (cooldown ~1.5 h of that is skipped on kill) | Member +0.0000..+0.0005; P(clears bar) ~0.08; private ~0.1-0.3x unless rho_x drops |
| 2 | **RDROP-CD** (R-Drop in the m02 cooldown) | **KL gate PASSED** (§2b.2). S = 0.0111 nats at the m02 end (0.0053 at the cooldown start, 0.0112 e10 base), against a bar of 1e-3. Benefit prior unchanged. | Regularises mask sensitivity without raising p. The sweep's "KL ~0" premise is refuted for single-pass pairs: one dropout pass changes the top-1 in ~25% of feeds (Spearman 0.961), about 1/3 of the seed-to-seed KL (0.033). At alpha 1.0 the term is ~0.35% of CE. | Next step: implement `--rdrop_cd` (default off, plus a bit-identity test), then `--rdrop_cd 1.0`, roots 101-104 from @77500, CDSEED 900+root, vs w5_cd_m02ref_r101-104, <= 3 lanes. | Wave-6 B rule plus a **pool-level** holdout >= +2 SE and stale-val >= 0; Holm over the 2 cooldown arms if SAM-CD also runs. Seed-corr guard: <= m02ref + 0.01. | code ~1 h; 4 runs ~6-7 lane-h | P ~0.05 |
| 3 | **CTXDROP** (TabR context dropout on retrieval probs) | Unchanged; no probe applies. | No probability dropout in HistRetrieval. Every measured dropout direction points against more dropout, and duplicate concentration is already handled by a4u (retr_uniq + tcnt, e10 +0.00036 / +0.00018). | Fold in as **one info arm of the a4u family** when a lane idles: `--retr_ctx_drop 0.2` after p = softmax(lg) (nn_x/abl/model.py ~l.211 and the uniq path ~l.283), sink slot untouched, training only. 10-ep s951/952. No 2-ep screen. | Consolidated bar (info unless it clears it). | 2 x 4.5 GPU-h | P ~0.05 |
| 4 | **MODSUP** (deep supervision on retrieval-only / pre-set logits) | Unchanged; no probe applies. | Listwise targets on intermediate logits; the merger's "next in line". Retrieval looks saturated (retr32 null) and R7 block forms went 0/13. | `--modsup 0.2`: two zero-init linear heads, CE on each, annealed like aux_anneal. 10-ep pair. | Consolidated bar. | ~1 h code; 2 x 4.5 GPU-h | P ~0.05 |
| 5 | **SAM-CD** (info only) | Kept only through the bundling rule with #2. The KL probe does not touch its own preconditions. | A flatness regulariser; the cooldown failure mode is a data dose, which SAM does not address. rho is uncalibrated under Muon. | Only with a published Muon-compatible SAM form and rho calibrated to \|\|eps\|\|/\|\|w\|\| ~0.005-0.01 on CPU first. Bundled with #2 under Holm. | As #2; info only. | 4 runs ~6-8 lane-h | P ~0.04 |

**LATE-DROPOUT pair status (as of 05:01 UTC)**
- **Launch condition.** The pair starts only when all four `nn_x/abl/logs/abl_e10_xdrop_pop_s95{<ids>}.json` exist; the pop-free ETA is ~06:10-06:20 UTC. Each GPU must then show < 4 compute processes and >= 60 GB free, checked under the per-GPU flock `nn_x/abl/logs/locks/gpu<g>.lock`.
- **Launch waiter.** PID **1814946** (`launch_waiter.sh`, log `launch_waiter.log`) is set to start s951 on GPU N and s952 on GPU N. It writes the train.py PIDs to `nn_x/tricks/late_dropout/pids.txt` and to nn_x/abl/RUNNING.txt.
- **Readout waiter.** PID **1815082** (`wait_readout.sh`, CPU only) runs the step-60000 kill gate (`kill_gate.py` -> `gate.json`; ETA ~07:35-08:10 UTC). On KILL it stops both train.py PIDs by exact number. Otherwise it runs `readout_late_dropout.py` -> `RESULT.txt` / `result.json` once both runs have their json plus selected holdout and trainonly_val preds (ETA ~08:30-09:30 UTC).
- **Readout rule.** As in the table and §2b.6. A run that dies without its json gives INCOMPLETE.
- **Process state.** Both waiters were alive at 05:01 UTC (checked in `/proc/<pid>/cmdline`) and are recorded in nn_x/abl/RUNNING.txt. No training run has started.
- **Possible handover (not confirmed).** The probe agent reported a planned handover: the orchestrator would stop both waiters by exact PID, and the lead would launch both runs on GPU N. Neither the stop nor a new launcher appeared in `/proc` or RUNNING.txt as of 05:01 UTC. If the handover happens, the pre-declared rule, the kill-gate constants and `readout_late_dropout.py` stay as written. Only the GPU assignment and the PIDs change, and RUNNING.txt carries the new ones.

**Order of operations (updated)**
- The CPU and GPU-minute pre-probes are done (§2b).
- GPU spend now goes, in order, to:
  1. The LATE-DROPOUT pair (pending, see above).
  2. RDROP-CD, on GPU-4 idle lanes once `--rdrop_cd` exists with its bit-identity test. SAM-CD is bundled under Holm only if its own preconditions are met.
  3. CTXDROP and MODSUP, as a4u-family info arms when a lane idles.
- Every 10-ep arm also records:
  - the M50-HOLDOUT columns (a random-like M<50 train holdout), if that set is built;
  - a dE/dL holdout information column (Sep 6-8 vs Sep 9-19, each ± SE; §2b.5) in any readout declared from now on. For LATE-DROPOUT it may be added post hoc as information only, never as a gate.

---

## 4. Dropped ideas (do not re-propose without a new mechanism)

### 4.1 Closed: already measured, or a no-op by the metric
| idea | reason (evidence) |
|---|---|
| Metric-fact screen: per-feed monotone transforms, geometric mean, P(no positive), MMR / diversity, Kemeny, tie tricks | Exact no-ops under per-feed binary NDCG with pessimistic ties, or already measured (review_metric, err_feed_structure, STUDY C, r6_tta). |
| Risk-averse member pooling (mean - λ·sd, lower CVaR) | STUDY C geometry flat ±0.0003; infinite-member headroom +0.0002; near-copy. |
| Spectral / one-factor label-free weights (SML) | fam-eq -0.00002, trunk-dedup -0.00028 already span the axis; headroom +0.0002. |
| Head-weighted per-feed normalisation (DCG-z, top-10 z, MAD z) | STUDY C V8 raw/gscale/rgauss within ±0.0002, holdout 0. |
| Rank-segment fusion (ProbFuse / SegFuse) | COVERAGE "segment / routing weights CLOSED"; A2c nested +0.0002 / holdout -0.0001. |
| Label-free FT/TO mixing, GAIN-WFT, HEAD-ARB, STALE-WFT | w_ft closed 3 times (nested +0.0002, z 0.96); M-split fresh -0.00017; S2 gate stacker val +0.00056 -> public -0.00035; the holdout cannot see FT. |
| NDCG-consistent multi-positive correction (ndcgcons + NCD) | Training form neutral (v2_idcg 0.38004 vs 0.38003); acts on 26% of feeds; ceiling ±0.0001. |
| PRM / set re-ranker trained on the 30K holdout | nn_x/stack neural listwise stacker +0.0000..+0.0004. |
| Train-scale rank-of-score second stage (D2 learning-curve gate) | cb26 holdout residual -0.00031 (z -2.3). A 4-point curve slope SE ~0.0004 exceeds the +0.0003-per-doubling bar, so the gate cannot discriminate. |
| Coldness-routed pop-free inclusion | Routers by coldness CLOSED; the flat Q1 = G3 + pop-free decision is already pre-declared (CURRENT_STATUS 03:50). |
| RCR-ListCE | Loss zoo CLOSED; Spearman(NDCG, -CE) 0.95. |
| EVENT-DEPTH graded positives | INSIGHTS do-not; metric binary; rank-percentile diff ~1.4 SE. |
| SCTTA sub-context TTA | T2 +0.00036 per member -> -0.00008 ensemble (r6_tta). |
| Snapshot ensembles / SWAG / Lookahead | Plateau averaging -0.0025 ens; EMA closed; MC-dropout TTA FAIL; tail EMA <= +0.0001. |
| Never-varied HP micro-sweep | R7 0/13, G6-G9; INSIGHTS do-not "sweeps over schedule knobs". The only live dropout cell is §3 #1. |
| TTT / Mean-Teacher on val+test | Opens the banned same-user later-feed channel; pseudo-labels CLOSED. |
| EVAL-EMU | METRIC-4/5: IDCG@30 touches 3 val feeds; the overlay matches G3 to 6 decimals. |
| CROSSFIT-READOUT | Half-split slope 0.79 ± 0.42 already measured; the frozen holdout-led rule handles the winner's curse. |
| PPI-LENS | 40 composition reweightings explained nothing; holdout fits public (chi2 0.27). |
| WiSE-FT | FT branch saturated (FT of 3/20 = all 20); pool ceiling ~+0.0001; the holdout cannot see FT. |
| RESID-TARGET boosting member | Its pre-check is a holdout stacker (closed class); ~10 GPU-h at P ~0.03. |
| :D candidate sub-list sampling; multi-sample sub-list consistency; SLS | Keeping all positives makes them label-dependent (the wave-2b aug failure mode, STOP); +70% step time; no M<50 holdout. |
| PRODDIR / DOM-X / domain CFG / K9 logit adjustment | §2: W6 side observation is a pure cold/warm trade (+0.00011 ± 0.00017). |
| FN-PU | §2: true negatives under the eval labels; ~0.05 masked negatives per average feed. |
| XUSER-RIM as specified | §2: the additive key reduces to item-kNN plus an item prior (both closed). |
| XOFF exploration-row position debias | Proxy only (E4 +0.00003 ± 0.00024); gated on DOMAIN-DIAG, which is now \|Δ\| < 1 SE. |
| Domain-specialist output heads | Propensity head and MoE closed; the domain token carries ~no signal (no_domain +0.00044 / +0.00001). |
| DFR last-layer re-fit on the exploration pool | LP-FT -0.00014; pp02 / valcd reproduce only FT's value; pool is M>=50 only. |
| COLDMIMIC | Cold/warm-trade class; content-only scores add ~0. |
| INOISE | xt dropout closed; d20 null. |
| PICK-EMAX as a work item | §2: a final-pick-time tie-break only. |
| **RECYCLE v2 / head-only score-conditioned cascade** (old §3 #2), including its GPU-4 arm and the depth-matched blocks3 control | **Closed 11-13 by Step 0 (§2b.1, pre-declared 04:44:58).** The top-8 perfect-reorder share of the headroom is 0.2937 ± 0.0037 val and 0.1640 ± 0.0018 holdout, against a bar of >= 0.40. AUC_head8 is 0.616 / 0.578, so the head is weakly ordered, but most error is in the tail: 51% of val feeds (66% of the headroom) and 67% of holdout feeds (77%) have no positive in the top-8. The median positive rank is 15 val / 28 holdout. The sweep's "49% head error" is the K = 16 share (0.488), not K = 8. Re-entry only as in §4.2. |
| **HDROP-ALL** (full-run history-event dropout, old §3 #3) | **Closed 11-13 by T2 replay on w5_e10_base_s951/952 (§2b.3, pre-declared 04:46:10, branch C).** Val TTA(4) at rate 0.1 is -0.00013 ± 0.00036 with C = 0.9957; at rate 0.2 it is -0.00098 ± 0.00041. A single pass loses 0.00077 val / 0.00056 holdout at 10%. The per-member TTA gap is +0.00002 / -0.00027, below +0.0002. The 2-ep T2 member gain (+0.00036) is gone at 10 ep. History thinning only removes information. |
| **FUTWIN Step A/B, PFD, future-window or dense category-intent aux** (old §2, §3 #8, §4.2 PFD) | **Closed for good 11-13 (§2b.4, pre-declared 04:45:44).** Oracle u_fut is +0.00275 ± 0.00068 on the 14.9% of holdout feeds that have a ≥ 24 h future window, against a bar of +0.005. Its ratio to the label echo is 0.093 (+0.0296), against 0.30 required. Without the echo of the feed's own positive it is +0.00072 (z 1.1). u_past is -0.00123. The Step-B margin over the echo is -0.027, against +0.01 required. |
| HOLD-CLEAN Step 2 (label-free E/L format readout) | **Done 11-13 (§2b.5, pre-declared 04:45:51).** "E is out-of-format": 6/10 (5 without S8), driven by the pre2 flag and LUO inflation (LUO zero rate E .179 / L .235 / val50 .379 / test50 .447). G3 E-L is +0.0018 ± 0.0027. The only output is a dE/dL info column for FUTURE arms. HOLD-CLEAN as a judge stays REFUTED, and no closed arm is re-judged. |

### 4.2 Deferred, with re-entry triggers
| idea | re-enter only if |
|---|---|
| ANCHOR-REPLAY (self-distil anchor on pool replay) | wave-6 m04 < m02 with a visible late decline (it is moot if m04 >= m02). |
| MetaBalance / PCGrad on the K9 aux | wave-6 aa0 or ap1 moves the member by > +1 SE (every K9 rescale so far is null). |
| POP-REL, MODDROP-POP, HZCOND | The pop-free family (abl_e10_xdrop_pop_s951-954) Q1 readout is in, AND a label-free holdout-format check (HOLD-CLEAN Step 2), declared before any arm delta, shows Sep 6-8 is out-of-format. Even then, the stale-val AND pmix >= +2 SE clause and the generational bar apply before any pick. **11-13 update:** the label-free half is now MET (§2b.5: 6/10, 5/10 without S8; mainly the pre2 flag and LUO inflation). The pop-free Q1 readout is still outstanding (chain PID 1698066). |
| FPCD (no Sep 6-8 feeds in the cooldown), PRE2T | Same label-free trigger as above; hcap (+0.00007) predicts null. **11-13 update:** the label-free half is MET (§2b.5); still waiting on the pop-free Q1 readout. |
| Cross-feed negatives (INBATCH-NEG, mixed negatives, same-user union) | A new argument that the category block is not information-limited (INSIGHTS A1). P ~0.05. |
| Score-conditioned second pass, non-head-only (full-list) or M<50-only head arm (successor of the closed RECYCLE v2) | A new design that is not head-only. Alternatively, an M<50-specific head arm: val M<50 alone meets both Step-0 conditions (headshare_8 0.571 ± 0.007, AUC_head8 0.636). It needs its own pre-declared rule and a non-holdout M<50 judge (e.g. M50-HOLDOUT), because the 30K holdout is M>=50 only. The in-sample pass-1 fix from §1.2 still applies. |
| M50-HOLDOUT | Attach to the next from-scratch 10-ep family; not standalone. It is also the judge an M<50 head arm would need (row above). |

---

## 5. Submission implications (frozen 11-13 rule)

- **Pick 1 stays G3** (val 0.40032, holdout 0.32379, public 0.392820, #5).
  - Nothing in this sweep produced a candidate, let alone one with holdout >= +2 SE vs G3 and val >= 0.
  - The queue in §3 cannot change that before it produces a 10-ep member pair, which is >= 9 GPU-h per arm on GPU N. Under the near-copy lesson, even a passing arm adds ~0.1-0.3x of a sub-MDE gain to private.
- **Pick 2:** keep the VAL_PUBLIC_1113 R6 rule (eligible, non-dominated pool with the lowest test correlation; currently PF7x-type).
  - At final-pick time only: confirm on ods.ai that the best of the 2 selected files counts.
  - If it does, run the Clark E[max] check as a tie-break with mu shrunk x0.1-0.3, and switch only on > 1 bootstrap SE.
- **The gap to #1 (public 0.398256, -0.0054) and #2 (0.395115, -0.0023) is not closable by any of the 63 items.** By the central lesson, only a generational member family (>= +0.002 val, corr < 0.99) could move private, and the best candidate here (RECYCLE v2) has P ~0.03 of being one. RECYCLE v2 was then closed by its own Step 0 readout (§2b.1), so no generational candidate remains in this sweep's queue.
- Any future candidate still needs src/presub_gate.py PASS. That includes leak checks: no val label may reach test except through nested user-grouped 5-fold, and no same-user later-feed information.

---

## Sources
- Internal (2026): reports/COVERAGE_MAP.md; reports/EXPERIMENT_REGISTRY.md; reports/INSIGHTS_1112.md; reports/RESEARCH_SWEEP_1112.md; reports/VAL_PUBLIC_1113.md; reports/R7_ARCH_HPARAM.md (resdrop0 row l.88, diversity l.114); reports/sol_cpu_CPU3_aux_head_member_W6.md (side observation, lines 26-29); reports/r6_tta.md; reports/REVIEW_SYNTHESIS.md; nn_x/w5/logs/w5_eval_B_hzcd.txt (lines 22, 32); nn_x/abl/logs/abl_eval.txt; nn_x/r6/logs/r6_eval_cd_w2b.txt; nn_x/insights/labels_loss/out/01_labels.txt.
- External: Gorishniy et al., TabR, ICLR 2024; Qin et al., RIM, KDD 2021; Pei et al., PRM, RecSys 2019; Jumper et al. 2021 (recycling); Nogueira et al. 2019 (mono/duo); Liu et al., "Dropout Reduces Underfitting", ICML 2023; Liang et al., R-Drop, NeurIPS 2021; Foret et al., SAM, ICLR 2021; Zhou et al., ICLR 2025 (arXiv 2410.10373); Ho & Salimans 2022 (CFG); Wei et al., MACR, KDD 2021; Menon et al., logit adjustment, ICLR 2021; Xu et al., PFD, KDD 2020; Pennington et al., PinnerFormer, KDD 2022; Ktena et al., RecSys 2019; Elkan & Noto, KDD 2008; Ji et al., TOIS 2023 (arXiv 2010.11060); Clark, Oper. Res. 1961; Xie et al., CL4SRec; Zhou et al., DA4Rec, WWW 2024.

---

## Registry one-liners (for the lead to paste)

### reports/COVERAGE_MAP.md: new block

```
## Tricks sweep 11-13 (reports/TRICKS_SWEEP_1113.md; 63 candidates, 12 ranked, 24/24 skeptic refutations, 0 Tier-A tests run)
| idea | status | best evidence | file |
|---|---|---|---|
| DOMAIN-DIAG / PRODDIR / DOM-X / domain CFG / K9 logit adjustment (inference-time production-direction removal) | CLOSED 11-13 (existing data) | main_prod vs main_rl +0.00011 ± 0.00017, cold +0.0026 / rest -0.0007 (pure cold/warm trade); XOFF/EXAM stay closed | sol_cpu_CPU3_aux_head_member_W6.md; TRICKS_SWEEP_1113 §2 |
| FN-PU (mask/soften later-interacted negatives) | CLOSED 11-13 (mechanism) | true negatives under eval labels; 0.402% of negatives, ~0.05 per avg feed | TRICKS_SWEEP_1113 §2 |
| XUSER-RIM cross-user labelled kNN, additive concat key | CLOSED 11-13 (mechanism) | key reduces to item-kNN + item prior (k_ null; item prior union +0.0004 z 1.4); reopen only with a two-stage key + as-of bank | TRICKS_SWEEP_1113 §2 |
| HOLD-CLEAN re-judging closed cold/warm arms on Sep 9-19 holdout | REFUTED 11-13 (process) | outcome-chosen judge; hzcd late20% -0.00031 ± 0.00038; allowed only as a label-free, pre-declared info column for future arms | TRICKS_SWEEP_1113 §2 |
| FUTWIN future-window intent aux / PFD | DEFERRED (strict gate) | needs label-echo + u_past controls, margin >= +0.01, frozen-trunk probe | TRICKS_SWEEP_1113 §2-3 |
| PICK-EMAX | final-pick tie-break only | claimed +0.00003..+0.0001 << public SE 0.0005 | TRICKS_SWEEP_1113 §5 |
| post-hoc combiners: risk-averse pooling, SML weights, head-weighted z, SegFuse, FT/TO confidence mixing, GAIN-WFT, HEAD-ARB, STALE-WFT, NCD | CLOSED 11-13 | bounded by infinite-member headroom +0.0002 val / ~0 hold; STUDY C ±0.0003 | TRICKS_SWEEP_1113 §4.1 |
| LATE-DROPOUT (res/FFN/retr drop 0 in plateau, 0.1 in cooldown) | QUEUED (GPU-4 idle lane, single pair, plateau-end kill gate) | resdrop0 2 ep +0.00030 / +0.00040, rho_x 0.9745 | TRICKS_SWEEP_1113 §3 #1 |
| RECYCLE v2 (score-conditioned 2nd pass, noised pass-1 + depth-matched control) | QUEUED (CPU Step 0 first; generational criterion only) | v1 flaw: in-sample pass-1 scores on train | TRICKS_SWEEP_1113 §3 #2 |
| HDROP-ALL / RDROP-CD / CTXDROP / MODSUP / SAM-CD | QUEUED behind CPU probes (T2 replay on w5_e10; KL probe >= 1e-3 nats) | priors P 0.04-0.07 | TRICKS_SWEEP_1113 §3 |
| ANCHOR-REPLAY, MetaBalance-K9, POP-REL, MODDROP-POP, HZCOND, FPCD, PRE2T, cross-feed negatives | DEFERRED (re-entry triggers: m04, aa0/ap1, pop-free Q1 + label-free format check) | - | TRICKS_SWEEP_1113 §4.2 |
| SCTTA, snapshot/SWAG, TTT/Mean-Teacher, :D sub-list, SLS, multi-sample sub-list, WiSE-FT, DFR, RCR-ListCE, EVENT-DEPTH, COLDMIMIC, INOISE, RESID-TARGET, D2 learning curve, EVAL-EMU, CROSSFIT, PPI-LENS, domain heads, XOFF | CLOSED 11-13 (dropped with reasons) | see §4.1 | TRICKS_SWEEP_1113 §4.1 |
```

### reports/EXPERIMENT_REGISTRY.md: lines

```
- 11-13 04:40 TRICKS SWEEP (reports/TRICKS_SWEEP_1113.md): 63 finder candidates (6 lenses) -> 12 ranked -> 24/24 skeptic refutations; 0 Tier-A tests executed (nn_x/tricks absent). No adoption; G3 stays pick 1.
- 11-13 DOMAIN-DIAG recorded from existing W6 side observation: production-token main head +0.00011 ± 0.00017 (cold +0.0026 / rest -0.0007) -> PRODDIR, DOM-X, domain CFG, K9 logit adjustment, XOFF, EXAM CLOSED.
- 11-13 FN-PU CLOSED by mechanism (later-interacted negatives are eval negatives; ~0.05 per average feed). XUSER-RIM CLOSED by mechanism (additive key = item-kNN).
- 11-13 HOLD-CLEAN refuted as an outcome-chosen judge; allowed residual = label-free E/L format readout + dE/dL info column for future arms only.
- 11-13 Queue (GPU N, idle lanes): LATE-DROPOUT s951/952 (plateau-end kill), RECYCLE v2 (CPU Step 0, noised pass-1, depth control), HDROP-ALL (after T2 replay probe), RDROP-CD (after KL probe), CTXDROP/MODSUP (a4u info arms). Bar: val > 0, holdout > +2 SE, slices, pmix, rho_x <= rho_bb - 0.005.
```

### Addendum after §2b (11-13 ~05:00 UTC): these lines supersede the RECYCLE, HDROP, FUTWIN/PFD and QUEUED rows above

COVERAGE_MAP rows (replace the FUTWIN, RECYCLE v2 and HDROP/RDROP/... rows; add the rest):

```
| RECYCLE v2 / head-only score-conditioned cascade (+ blocks3 depth control) | CLOSED 11-13 (Step 0, pre-declared) | top-8 perfect-reorder headroom share 0.294 val / 0.164 hold (bar >= 0.40); AUC_head8 0.616 / 0.578; 66% val / 77% hold headroom in feeds with no positive in top-8; re-entry only full-list or M<50 head arm with a non-holdout judge | nn_x/tricks/recycle_step0/RESULT.txt; TRICKS_SWEEP_1113 §2b.1 |
| HDROP-ALL full-run history-event dropout | CLOSED 11-13 (T2 replay on w5_e10_base_s951/952, branch C) | val TTA(4) d0.1 -0.00013 ± 0.00036, corr 0.9957; d0.2 -0.00098; single pass -0.00077 val / -0.00056 hold; 2-ep T2 gain gone at 10 ep | nn_x/tricks/HDROP-ALL/RESULT.txt; TRICKS_SWEEP_1113 §2b.3 |
| FUTWIN future-window intent aux / PFD / dense category-intent aux | CLOSED for good 11-13 (Step A oracle, pre-declared) | u_fut +0.00275 ± 0.00068 (bar +0.005), 0.093x label echo (bar 0.30), echo-free +0.00072 (z 1.1), u_past -0.00123 | nn_x/tricks/FUTWIN/RESULT.txt; TRICKS_SWEEP_1113 §2b.4 |
| RDROP-CD / SAM-CD / two-pass consistency regularisers | QUEUED 11-13 (KL gate PASSED, benefit prior unchanged P ~0.05) | two-mask symKL 0.0111 nats at m02 end, 0.0053 at CD start, 0.0112 e10 base (bar 1e-3); top-1 flips 25% of feeds per pass; next: implement --rdrop_cd, 4 roots vs m02ref | nn_x/tricks/rdrop_cd_kl/RESULT.txt; TRICKS_SWEEP_1113 §2b.2 |
| HOLD-CLEAN Step 2 label-free holdout format (E Sep 6-8 vs L Sep 9-19) | DONE 11-13: "E is out-of-format" (6/10; 5 w/o S8) | pre2 flag + LUO inflation (LUO zero E .179 / L .235 / val50 .379); G3 E-L +0.0018 ± 0.0027; dE/dL info column for FUTURE arms only, no re-judging; supplies the label-free half of the POP-REL/MODDROP-POP/HZCOND/FPCD/PRE2T trigger | nn_x/tricks/hold_clean/RESULT.txt; TRICKS_SWEEP_1113 §2b.5 |
| LATE-DROPOUT 10-ep pair (abl_e10_latedrop_s951/952) | PENDING 11-13 (pre-declared 04:50:54; flag probe PASS 9/9) | waits for abl_e10_xdrop_pop_s951-954 json; kill iff both seeds' holdout@60000 < base - 2·0.00076575; PASS = val > 0, hold > +2 SE, slices, pmix >= -1 SE, pool rho_x <= 0.96853 | nn_x/tricks/late_dropout/PREDECLARED.txt; TRICKS_SWEEP_1113 §3 #1 |
```

EXPERIMENT_REGISTRY lines:

```
- 11-13 04:45-04:51 TRICKS closure probes (pre-declared, nn_x/tricks/*; nothing adopted/submitted; TRICKS_SWEEP_1113 §2b): RECYCLE Step 0 CLOSED (top-8 headroom share 0.294 val / 0.164 hold < 0.40; error is in the tail); HDROP-ALL CLOSED (w5_e10 T2 replay: TTA d0.1 -0.00013 ± 0.00036, corr 0.9957, branch C); FUTWIN/PFD CLOSED for good (oracle +0.00275 ± 0.00068 = 0.093x label echo; echo-free +0.0007); RDROP-CD KL gate PASSED (0.0111 nats at m02 end vs 1e-3; stays queued, P ~0.05); HOLD-CLEAN Step 2 "E out-of-format" 6/10 -> dE/dL info column for future arms only.
- 11-13 LATE-DROPOUT abl_e10_latedrop_s951/952 (E_BASE+HYG --drop 0 --attn_drop 0.1 --drop_cd 0.1) pre-declared 04:50:54, flag probe PASS 9/9; launch after the pop-free json (waiters PID 1814946 launch / 1815082 kill gate + readout); kill gate step 60000 (SE_gate 0.00076575); readout = consolidated bar, pool-level rho_x <= 0.96853.
```
