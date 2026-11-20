# Ensemble / stacking design study (sol_ensemble)

Date: 2026-11-11, 17:45-18:40 UTC. CPU only (`CUDA_VISIBLE_DEVICES=""`, <= 16 threads per process). Read-only use of `preds/`, `data/`, `nn/`.
Code: `<repo>/solvers/ensemble/exp*.py` (shared helpers in `ens_common.py`). Logs and JSON: `<repo>/solvers/out/ensemble/exp*.{log,json}`.
Candidate score files (not submissions): `<repo>/solvers/out/ensemble/cand_D*_{val,test}.npy`.

## Notation
* **Family mix** = 0.5 z(mean of the members' per-feed z, fine-tuned) + 0.5 z(same, train-only). v11 = mix(F5).
* **Families:**
  * **F5** = nn_final5: v12_all_{e10,s81,s82} and v12_allprew_{s83,s84,s85}, fta fine-tune `n2cpf`.
  * **F5cd** = the N2 one-epoch cooldown continuations v15_cd_* (`n2cpf`).
  * **V13** = the three working nn_final6-recipe hedges v13_lr1_mix1 / v13_lr2_mix1 / v13_lr1_mix0 (all seed 23, `n2cpf`).
  * **F4** = nn_final4: v9_e10{,_s61,_s62,_s63} (`n2cp`).
  * **OLD5ep** = 8 earlier 5- or 8-epoch members, incl. GRU / DIN / deep (`n2cp`).
  * Single-model families: F3 (nn_final3), pre_final / pre2_final (encoder zero-shot), xgb_x300_luo, cb_tr300k_luo, stk2_res4_clean.
* **eq(A,B,...)** = equal-weight sum of the family mixes (per-feed z).
* **Deltas** are paired per-feed differences vs v11 on val (14,423 feeds), as mean +- SE.
  * Slices: M<50 / M>=50 / Tue / Thu.
  * `priv_w` = src/calib.py private M-mix weights.
  * `adv` = adversarial private-like weights (2.7).
* **Holdout** = the 30K random-like train feeds (`preds/holdout/`, train-only members only). Every member's checkpoint was picked on this holdout, so holdout levels are slightly optimistic. Family-set comparisons on it are fair, because no family set was chosen on it here.
* **Folds** = nn/finetune.py's `GroupKFold(5)` over val feeds grouped by user_id. Verified identical: `nn/finetune.py:194-202` and `nn/data.py:67-73` read feeds.parquet in the same order as `ens_common.py:39-45`. Fold sizes 2885/2885/2885/2884/2884; all user-0 feeds are in fold 0.

## 0. TL;DR

1. **The combiner is already right.**
   * The per-feed z-mean ties the raw-logit mean and the softmax-probability mean:
     * val: +0.00004 +- 0.00015 and +0.00005 +- 0.00021;
     * holdout: -0.00011 +- 0.00006 and -0.00020 +- 0.00012.
   * Rank fusion loses clearly:
     * rank-mean: val -0.0020 +- 0.00035, holdout -0.0014 +- 0.0002;
     * RRF(60): -0.0007 +- 0.0003;
     * Gaussianised ranks: -0.0008 +- 0.0003.
   * Per-member and per-family equal weights are identical inside one family.
2. **Keep w_ft = 0.5. Do not use segment weights.**
   * Nested CV picks 0.5 in 5/5 folds from {0.33, 0.5, 0.67}. w_ft 0.67 scores -0.0003 +- 0.0002.
   * M-specific weights look good on F5 alone: nested +0.00034 +- 0.00025 (1.4 SE). The pattern does not replicate on 4 other member pools: F4 has the opposite sign, and F5cd / V13 / OLD5ep show no M effect.
3. **No stacker beats equal-weight family mixes.** All nested 5-fold, vs v11:
   * LightGBM lambdarank: narrow -0.0020 +- 0.0006; wide +0.0004 +- 0.0007; residual -0.0008 +- 0.0004.
   * Listwise linear: wide +0.00095 +- 0.00064.
   * Caruana greedy selection: +0.0010 +- 0.0005, with +0.0014 optimism.
   * The pre-declared equal mix eq(F5,F5cd,V13) scores +0.00117 +- 0.00048 with zero fitted parameters.
4. **Strength-vs-diversity law (all family pairs).** gain over the better member ≈ α(1-ρ) - β·gap.
   * Val: α 0.019, β 0.38, R² 0.95. Holdout: α 0.061, β 0.62, R² 0.96.
   * A family helps only if gap < ~0.05(1-ρ) on val. For example, within 0.001 of v11 at ρ 0.98, or within 0.01 at ρ 0.8.
   * Every GBDT / encoder / stacker / older-generation addition fails this test, even at its best in-sample small weight.
5. **The lever is same-strength, different-recipe families, not more seeds and not more fine-tunes.**
   * Six cooled members (F5cd) on top of F5 add +0.0008 val and +0.0011 holdout. On the holdout this beats the infinite-seed limit of the F5 recipe (0.32272 vs a_inf 0.32232).
   * More fine-tunes of the same train-only checkpoints add +0.0002 +- 0.0002. Same-member fine-tune variants correlate 0.998, vs 0.950 across seeds.
   * On test, the 5 fold models of a fine-tune correlate 0.9974, so test fold-bagging adds about nothing. Val and test ensemble statistics match to 3 decimals.
6. **A better candidate exists now, from existing files.**
   * **D4 = member-equal mix over F5+F5cd+V13** (15 members per branch): val **0.39908 = +0.00115 +- 0.00044 (z 2.6)**.
     * Every slice is positive.
     * Holdout +0.0013 +- 0.0003 (late slice +0.0011).
     * Test Spearman vs v11 0.992.
   * D3 = eq(F5,F5cd,V13) is equivalent (+0.00117 +- 0.00048).
   * Both pass the frozen replacement rule. Selection optimism among the 6 pre-declared designs is +0.0005 (bootstrap), so the honest val gain is about +0.0007.
7. **FINAL design (pre-declarable)** = member-equal z-mean of every family that passes a strength gate, with 0.5 fine-tuned + 0.5 train-only branches. Projected +0.0015..+0.0025 over v11 (section 4).
   * Best use of GPU, in order: cooldown continuations of the F6/F7 plateau checkpoints, then new-recipe families.
   * Not worth GPU: more seeds of one recipe, more fine-tunes, GBDTs, stackers.

## 1. What was run

| exp | script | content | runtime |
|---|---|---|---|
| 1 | exp1_combiners.py | 8 combiners on the v11 pool; w_ft grid; 5 segment-weight schemes in nested CV | 22 s |
| 2 | exp2_stacker.py | 4 plain family mixes; listwise linear stackers (narrow, wide); LightGBM lambdarank (narrow, wide, residual), nested 5x4 | 84 s |
| 3 | exp3_diversity.py | 17-family per-feed correlation matrix (val and test); v11-pool structure; test fold-model correlation; diminishing-returns curves (val, holdout) | 20 min |
| 4 | exp4_families.py | holdout combiners; family- vs member-equal on val and holdout; M-segment w_ft replication on 7 pools | 2 min |
| 5 | exp5_adv.py | val-vs-test adversarial validation on per-feed ensemble statistics; importance-weighted deltas | 1 min |
| 6 | exp6_marginal.py | marginal gain of adding j members of each kind to F5 vs same-recipe extrapolation (holdout, val) | ~25 min |
| 7 | exp7_pairs.py | pairwise strength-vs-diversity law (val 36 pairs, holdout 10 pairs) | 1 min |
| 8 | exp8_candidates.py | pre-declared final designs D0-D5 on val / holdout / test similarity; writes cand files | 1 min |
| 9 | exp9_caruana.py | Caruana greedy selection (27 mix units, 25 steps) in nested CV | 6 min |
| 10 | exp10_optimism.py | bootstrap selection optimism of picking among D0-D5 | 10 s |
| 11 | exp11_ftbag.py | averaging several fine-tune variants of the same 6 train-only members | 20 s |

Reproduce:
```
cd <repo>
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 .venv/bin/python -u solvers/ensemble/expN_*.py
```
Run exp5 before exp8, because exp8 reads `adv_w_private_val.npy`.

## 2. Findings

### 2.1 Combiners (exp1, exp4a), verified
v11 recomputed from the 12 member files = 0.39793, the same as from `nn_final5_{val_oof,trainonly_val}` (exp1.log:11).

| combiner (12 members of v11) | val | delta vs v11 | M<50 | M>=50 | holdout (6 train-only) |
|---|---|---|---|---|---|
| A v11: z of family z-means, 0.5/0.5 | 0.39793 | 0 | 0 | 0 | 0.32162 (z-mean) |
| B per-member z-mean (no family re-z) | 0.39792 | -0.00001 +- 0.00001 | | | |
| D raw-logit mean | 0.39797 | +0.00004 +- 0.00015 | +0.00017 | -0.00006 | -0.00011 +- 0.00006 |
| E softmax-probability mean | 0.39798 | +0.00005 +- 0.00021 | +0.00019 | -0.00005 | -0.00020 +- 0.00012 |
| C within-feed rank mean | 0.39593 | **-0.00200 +- 0.00035** | -0.00312 | -0.00122 | **-0.00140 +- 0.00021** |
| G reciprocal-rank fusion (k=60) | 0.39724 | -0.00069 +- 0.00034 | | | |
| H Gaussianised ranks, z | 0.39712 | -0.00081 +- 0.00032 | | | |

* Score margins carry information that rank fusion throws away: rank-mean loses at 5.7 SE on val and 6.7 SE on the holdout.
* The scales of z, raw logits and softmax are interchangeable here, because all members share one architecture and loss.
* Rule: z-mean stays. It is also the only option that is invariant to member temperature when a different loss or family enters later.

### 2.2 Fine-tuned share and segment-specific weights (exp1, exp4c, exp5)
**Global w_ft.**
* The nested CV choice from {0.33, 0.5, 0.67} is 0.5 in all 5 folds (`S0b`, exp1.log).
* A 5-value grid {0.3..0.7} chooses 0.4/0.5 and loses 0.00023 +- 0.00013 nested. That is pure selection noise.
* w_ft 0.67 = -0.00032 +- 0.00024. It is +0.00006 +- 0.00031 under adversarial private-like weights, so there is no shift argument for it.

**Segment weights**, chosen per segment from {0.33, 0.5, 0.67} in nested CV, delta vs w = 0.5:

| scheme | nested | in-sample | chosen |
|---|---|---|---|
| S1 M<50 / M>=50 | +0.00034 +- 0.00025 | +0.00034 | 0.33 / 0.67 in 5/5 folds |
| S2 cold-share terciles (label-free) | -0.00023 +- 0.00020 | +0.00011 | unstable |
| S3 user0 vs known | -0.00005 +- 0.00005 | 0 | |
| S4 M x cold (2x2) | +0.00015 +- 0.00023 | +0.00034 | |

**Replication of the M effect (ft minus to, per pool; exp4.log:25-32):**

| pool | ft-to M<50 | ft-to M>=50 | nested S1 |
|---|---|---|---|
| F5 (n2cpf) | -0.0018 +- 0.0011 | +0.0025 +- 0.0008 | +0.00034 +- 0.00025 |
| F5 (n2cp) | -0.0028 +- 0.0012 | +0.0006 +- 0.0008 | +0.00077 +- 0.00025 |
| F5 (n2cpv2) | -0.0015 +- 0.0010 | +0.0019 +- 0.0007 | -0.00022 +- 0.00015 |
| F5cd | +0.0011 +- 0.0012 | +0.0017 +- 0.0008 | +0.00021 +- 0.00016 |
| V13 | +0.0019 +- 0.0012 | +0.0011 +- 0.0008 | +0.00014 +- 0.00026 |
| F4 | +0.0014 +- 0.0013 | -0.0003 +- 0.0009 | -0.00035 +- 0.00013 |
| OLD5ep | +0.0010 +- 0.0014 | +0.0016 +- 0.0009 | -0.00002 +- 0.00028 |

* The "train-only is better on small feeds" pattern belongs to the six v12 plateau train-only checkpoints. It shows up under all three fine-tunes of those checkpoints and nowhere else, including their own cooled continuations.
* Once F6/F7/F5cd enter the ensemble the effect will dilute. **Reject segment weights.** Earlier sign flips of per-segment weights (review_ideas.md:403) were the same phenomenon.
* The listwise-linear narrow stacker learns the same direction (ft x [M>=50] +0.24, to x [M>=50] -0.13; exp2.json) and gains nothing nested (-0.00001 +- 0.00023).

### 2.3 Stackers (exp2), verified
Plain pre-declared mixes vs v11:

| design | val | delta | M<50 | M>=50 | Tue | Thu |
|---|---|---|---|---|---|---|
| eq(F5, F4) | 0.39692 | -0.00101 +- 0.00059 | -0.0021 | -0.0003 | -0.0018 | -0.0002 |
| eq(F5, F5cd) | 0.39869 | +0.00076 +- 0.00043 | +0.0011 | +0.0006 | +0.0008 | +0.0007 |
| eq(F5, F5cd, V13) | 0.39910 | **+0.00117 +- 0.00048** | +0.0020 | +0.0006 | +0.0010 | +0.0014 |
| eq(F5, F4, F5cd, V13) | 0.39875 | +0.00082 +- 0.00051 | +0.0006 | +0.0010 | -0.0001 | +0.0018 |

Learned stackers, nested 5-fold. LightGBM uses 4 inner models per outer fold, each trained on 3 folds and early-stopped on the 4th; the outer prediction is the mean of the inner z.

| stacker | val | delta vs v11 | inner single-model ES delta | notes |
|---|---|---|---|---|
| LIN-narrow: F5_ft, F5_to x [1, M>=50, cold_share, user0] | 0.39793 | -0.00001 +- 0.00023 | in-sample -0.00001 | M<50 -0.0006, M>=50 +0.0004 |
| LIN-wide: 14 family z's (L2 1e-3) | 0.39889 | +0.00095 +- 0.00064 | in-sample +0.00089 | weights: F5cd_to 0.51, V13_ft 0.33, F3_to -0.21, GBDTs ~0.03 |
| LGB-narrow: 2 scores + ranks + M, H, user0, cold, cold_share | 0.39590 | -0.00203 +- 0.00055 | -0.0123 | best_iter 4-78 |
| LGB-wide: + 12 family z's + rank(v11) | 0.39836 | +0.00043 +- 0.00071 | -0.0038 | the 4-model bag gains +0.004 over single inner models |
| LGB-res: wide, init_score = 2·z(v11) | 0.39714 | -0.00079 +- 0.00040 | +0.00022 | best_iter 1-10; optimism ~+0.0010 |

* Tree stackers cannot reproduce a near-linear blend of two 0.99-correlated scores with a few shallow trees (LGB-narrow).
* The residual version finds no signal. Its early stopping chooses 1-10 trees on noise, and those trees hurt out of fold.
* The linear-wide stacker works only because it re-weights towards the same-strength families (F5cd, V13). It is no better than their equal mix: LIN-wide 0.39889 vs eq(F5,F5cd,V13) 0.39910.
* Optimism of fitting on val ranges from about +0.001 (LGB-res, Caruana +0.0014) down to ~0 (14-parameter linear model).

### 2.3b Caruana greedy ensemble selection (exp9)
Setup: library of 27 mix units (F5, F5cd, V13, F4, OLD5ep), forward selection with replacement, selected on 4 folds and evaluated on the 5th.

| steps | 1 | 3 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|
| nested delta vs v11 | -0.0020 | -0.0008 | +0.0002 | +0.0005 | +0.0002 | +0.0008 | **+0.0010 +- 0.0005** |

* In-sample (4-fold) delta at 25 steps is +0.0024. **Optimism +0.0014.**
* Caruana(25) minus eq(F5,F5cd,V13) = -0.00015 +- 0.00042.
* Picks concentrate on F5cd / V13 / F5 units; F4 and OLD units appear 0-2 times per fold. Data-driven selection lands on the same family set as the strength gate below, but with noisier weights.
* Verdict: no reason to fit selection or weights on val.

### 2.4 Diversity (exp3), verified
**Per-feed Pearson correlation of family z-scores** (exp3.log:19-37; val above the diagonal, test below). Selected pairs:

| pair | val | test |
|---|---|---|
| F5_ft-F5_to | 0.991 | 0.991 |
| F5_to-F5cd_to | 0.984 | 0.984 |
| F5_to-V13_to | 0.976 | 0.976 |
| F5_to-F4_to | 0.965 | 0.965 |
| F5_to-OLD5ep_to | 0.967 | 0.967 |
| F5_ft-stk2_res4_clean | 0.925 | 0.926 |
| F5_ft-xgb_x300_luo | 0.809 | 0.805 |
| F5_ft-cb_tr300k_luo | 0.766 | 0.762 |
| F5_ft-pre_final | 0.697 | 0.693 |
| pre_final-pre2_final | 1.000 | 1.000 |

pre2_final re-scores only 228-306 val rows (pretrain_report.md:257), so it is a duplicate signal.

**Member level, v11 pool** (6 ft + 6 to; val | test):
* seed-to-seed train-only 0.9356 | 0.9354; seed-to-seed fine-tuned 0.9503 | 0.9510;
* fine-tuned vs its own train-only start 0.9861 | 0.9869; cross-member ft-to 0.9351 | 0.9358.
* "Effective number of models" is 1.11 (eigenvalue participation) or 1.05 (N/(1+(N-1)ρ̄)). This number is uninformative here: one shared component dominates, and ensembling only polishes the 5-7% unshared variance. The useful quantities are the curves in 2.6.

**Recipe diversity beyond seed noise.** Assume a same-recipe family is "shared signal + independent seed noise", with ρ̄ = 0.9356. Then two family means are predicted to correlate as follows; observed values in parentheses:
* F5 (6) vs F5cd (6): 0.9887 (0.984);
* F5 (6) vs V13 (3): 0.9831 (0.976);
* F5 (6) vs F4 (4): 0.9859 (0.965).

So F5cd, V13 and F4 carry structured differences beyond seed noise: excess decorrelation 0.005, 0.007 and 0.021. F4 is also 0.004 weaker, which is why it hurts (2.5).

**Fine-tuned branch on test.**
* Test fine-tuned predictions are the mean of 5 fold models. Those fold models correlate **0.9974** with each other (exp3.log:43), against 0.950 between seeds.
* By the curve model (gain ∝ (1-ρ)(1-1/k)), fold-bagging is worth ≈ 0.0041 × 0.0026/0.0497 × 0.8 ≈ +0.0002 for a single member, and less inside the 30-model ensemble.
* So the val-calibrated 0.5/0.5 mix transfers to test without a hidden fine-tune advantage. A full-val refit can only add through the extra 25% of data, not through bagging.

### 2.5 Strength-vs-diversity law (exp7), verified
Equal-mix gain over the better family = α(1-ρ) - β·|gap|, fitted by OLS without an intercept:

| set | pairs | α | β | R² |
|---|---|---|---|---|
| val, family mixes + weak models, gap < 0.01 | 17 | 0.0186 | 0.378 | 0.95 |
| val, all pairs | 36 | 0.0130 | 0.291 | 0.80 |
| holdout, train-only families | 10 | 0.0615 | 0.620 | 0.96 |

* **Break-even gap ≈ 0.049(1-ρ) on val** (0.099(1-ρ) on the holdout). Examples:
  * ρ 0.98 → 0.0010;
  * ρ 0.96 → 0.0020;
  * ρ 0.90 → 0.005;
  * ρ 0.80 → 0.010;
  * ρ 0.70 → 0.015.
* Observed pairs with v11 (exp7.log), equal-mix gain over the better member:
  * F5cd +0.00041, V13 +0.00054 (helpful: gap < 0.0004, ρ 0.98-0.99);
  * F4 -0.0010 (gap 0.0039, ρ 0.967); OLD5ep -0.0024;
  * stk2 -0.0045; xgb_x300_luo -0.0049; pre2_final -0.0100.
* Even the best in-sample small weight (grid 0.05-0.5) of pre2_final, xgb, stk2 or OLD5ep stays below v11. F4 at w 0.2 is +0.0002 in-sample, i.e. noise.
* **Implication for "other solver types":** a non-NN solver (GBDT, encoder, CF) would need val ≳ 0.388 at its current ρ ≈ 0.8, or ρ ≲ 0.7 at ~0.383, to add anything. The best one is at 0.367. A different NN family must be within ~0.001 of the current family level at ρ ~0.98, or within ~0.002 at ρ ~0.96.

### 2.6 Diminishing returns and marginal gains (exp3, exp6)
Mean ensemble NDCG over random member subsets (≤ 40 per size), fitted by a - b/k. Power-law fits a - b/k^c give c 0.66-0.99 with a_inf 0.0003-0.0006 higher.

| pool | k=1 | k=6 | a_inf (a - b/k) | 6 → 12 same recipe | 6 → ∞ |
|---|---|---|---|---|---|
| val F5 train-only | 0.39247 | 0.39712 | 0.39774 | +0.00045 | +0.0006 |
| val F5 fine-tuned | 0.39435 | 0.39784 | 0.39847 | +0.00034 | +0.0006 |
| val F5 mix units (ft+to) | 0.39392 | 0.39792 | 0.39866 | +0.00040 | +0.0007 |
| holdout F5 train-only | 0.31790 | 0.32162 | 0.32232 | +0.00037 | +0.0007 |
| holdout F5cd train-only | 0.31940 | 0.32279 | 0.32331 | +0.00033 | +0.0005 |
| val F5+F5cd mix units (12) | 0.39478 | 0.39827 | 0.39899 | (k=12: 0.39870) | |
| holdout F5+F5cd (12) | 0.31865 | 0.32234 | 0.32303 | (k=12: 0.32272) | |

**Marginal gain of adding j members of another kind to the 6 F5 members**, vs the same-recipe extrapolation a - b/(6+j). Holdout (train-only), exp6.log:

| added | +1 | +2 | +3 | +4 | +6 |
|---|---|---|---|---|---|
| F5cd | +0.00030 | +0.00041 | +0.00055 | +0.00066 | +0.00077 |
| V13 | +0.00017 | +0.00039 | +0.00051 | | |
| F4 | +0.00019 | +0.00014 | -0.00004 | -0.00025 | |
| OLD5ep | -0.00021 | -0.00049 | -0.00084 | -0.00113 | -0.00158 (+8: -0.00202) |

Absolute holdout values with all members added: +F5cd 0.32272, +V13 0.32234, +F4 0.32162, +OLD5ep 0.31998; F5 alone 0.32162.

Val, mix units (fine-tuned + train-only enter together); within-F5 fit a 0.39866, b 0.00476; values are the diff vs the same-recipe extrapolation:

| added | +1 | +2 | +3 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|
| F5cd | +0.00031 | +0.00043 | +0.00041 | +0.00038 | +0.00044 | |
| V13 | +0.00019 | +0.00036 | +0.00063 | | | |
| F4 | +0.00003 | -0.00030 | -0.00050 | -0.00075 | | |
| OLD5ep | -0.00059 | -0.00122 | -0.00155 | -0.00190 | -0.00274 | -0.00334 |

Absolute val values: F5 + 6 F5cd = 0.39870 (+0.00077 over F5); F5 + 3 V13 = 0.39876 (+0.00083); F5 + 4 F4 = 0.39743; F5 + 8 OLD5ep = 0.39498.

Reading:
* **Per member, a cooled or new-recipe member is worth 2-3x a same-recipe seed** at this ensemble size:
  * F5cd: +0.0008 val / +0.0011 holdout for 6 members;
  * V13: +0.0008 val / +0.0007 holdout for only 3 members;
  * 6 more F5 seeds would give +0.0003..0.0004.
* Weaker members help only at count 1-2 (F4 on the holdout) and hurt beyond; OLD5ep hurts from the first member. A strength gate is needed, not a "more is better" rule.
* **Gate calibration (single-member train-only means, val / holdout):**
  * F5cd 0.39426 / 0.31940 (best);
  * V13 0.39408 / 0.31857 (gap 0.0002 / 0.0008, helps);
  * F5 0.39247 / 0.31790 (gap 0.0018 / 0.0015, helps on val, neutral on holdout);
  * F4 0.39127 / 0.31652 (gap 0.0030 / 0.0029, hurts);
  * OLD5ep 0.38876 / 0.31462 (gap 0.0055 / 0.0048, hurts).
  * A gate of "mean single-member train-only score within 0.002 of the best family on both val and the holdout" separates helpful from harmful families exactly. It does not depend on member count.

### 2.7 Ensemble-level adversarial validation (exp5)
Setup: LightGBM val-vs-test classifier on per-feed statistics, 5-fold AUC.

| task | meta (M, H, user0, cold_share) | + ensemble stats (no hour) | + hour |
|---|---|---|---|
| val vs private | 0.697 | 0.706 | 0.737 |
| val vs public | 0.620 | 0.633 | 0.683 |
| public vs private | 0.564 | 0.570 | 0.687 |

* cold_share dominates (gain share 0.53-0.71). It is the known staleness shift: val 0.381, public 0.429, private 0.459.
* Ensemble statistics add only +0.01 AUC. Their means are essentially equal across splits (val | public | private):
  * corr(F5_ft, F5_to): 0.9906 | 0.9911 | 0.9912;
  * member disagreement: 0.209 | 0.210 | 0.210;
  * top-1 margin: 0.275 | 0.264 | 0.267.
* The members agree and disagree on test the way they do on val, so blend conclusions transfer.

Importance-weighted deltas use w = p/(1-p) from the val-vs-private classifier (clipped [0.2, 5]; ESS 8,725 of 14,423):

| design | unweighted | adversarially weighted |
|---|---|---|
| eq(F5, F5cd) | +0.00076 +- 0.00043 | +0.00098 +- 0.00056 |
| eq(F5, F5cd, V13) | +0.00117 +- 0.00048 | +0.00104 +- 0.00061 |
| eq(F5, V13) | +0.00054 +- 0.00049 | +0.00087 +- 0.00063 |
| w_ft 0.67 | -0.00032 +- 0.00024 | +0.00006 +- 0.00031 |
| w_ft by M (0.33/0.67) | +0.00034 +- 0.00025 | +0.00061 +- 0.00032 |
| train-only F5 alone | -0.00081 +- 0.00044 | -0.00072 +- 0.00053 |

All conclusions keep their sign. M-specific weights look better under private-like weights, but they fail replication (2.2).

### 2.8 Pre-declared final-design candidates from existing files (exp8, exp10)

| design | val | delta vs v11 | M<50 | M>=50 | Tue | Thu | adv | holdout (to) | holdout delta (late) | test Spearman vs v11 | calib pub / priv |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D0 v11 | 0.39793 | 0 | | | | | | 0.32162 | | 1 | 0.38978 / 0.39244 |
| D1 eq(F5,F5cd) | 0.39869 | +0.00076 +- 0.00043 | +0.0011 | +0.0006 | +0.0008 | +0.0007 | +0.0010 | 0.32271 | +0.0011 +- 0.0003 (+0.0007) | 0.9934 | 0.39063 / 0.39324 |
| D2 eq(F5,V13) | 0.39847 | +0.00054 +- 0.00049 | +0.0006 | +0.0005 | -0.0007 | +0.0019 | +0.0009 | 0.32236 | +0.0007 +- 0.0003 (+0.0008) | 0.9898 | 0.39036 / 0.39301 |
| **D3 eq(F5,F5cd,V13)** | **0.39910** | **+0.00117 +- 0.00048** | +0.0020 | +0.0006 | +0.0010 | +0.0014 | +0.0010 | 0.32274 | +0.0011 +- 0.0003 (+0.0012) | 0.9904 | 0.39098 / 0.39361 |
| **D4 member-equal F5+F5cd+V13** | **0.39908** | **+0.00115 +- 0.00044** | +0.0017 | +0.0008 | +0.0012 | +0.0011 | +0.0010 | 0.32290 | +0.0013 +- 0.0003 (+0.0011) | 0.9923 | 0.39099 / 0.39361 |
| D5 eq(F5,F5cd,V13,F4) | 0.39875 | +0.00082 +- 0.00051 | +0.0006 | +0.0010 | -0.0001 | +0.0018 | +0.0007 | 0.32257 | +0.0010 +- 0.0003 (+0.0014) | 0.9887 | 0.39068 / 0.39331 |

* D3 and D4 satisfy the frozen replacement rule (CURRENT_STATUS 3k):
  * > 2 SE on paired val (z 2.4 / 2.6);
  * no slice below -2 SE (all positive);
  * holdout ensemble >= 0.32162 (0.32274 / 0.32290);
  * file checks pass: finite, length 2,904,806 (exp8 assert).
* D4 has the smaller SE and the higher holdout. It is also the natural form when families have unequal sizes, because each member gets equal weight.
* Bootstrap selection optimism of picking the best of D0-D5 is +0.00048 (exp10.log). Picks: D3 43%, D4 36%. Out-of-bag delta of the pick +0.00083; P(OOB < 0) 0.10. Expected real gain is therefore about +0.0007 val.
* Independent confirmation on the holdout: +0.0011..0.0013 at z ≈ 4, with fresh labels for the train-only branch.

### 2.9 More fine-tunes per member (exp11)
Fine-tuned branch = mean over several fine-tune variants of the same 6 train-only checkpoints; mix 0.5/0.5 with train-only:

| fine-tuned branch | ft alone | mix | delta vs v11 |
|---|---|---|---|
| n2cpf (v11) | 0.39784 | 0.39793 | 0 |
| n2cpf + n2cpv2 | 0.39755 | 0.39816 | +0.00022 +- 0.00020 |
| n2cpf + n2cp + n2cpv2 | 0.39749 | 0.39812 | +0.00018 +- 0.00019 |

Fine-tune variants of the same member correlate 0.9976-0.9977; fine-tuned members across seeds correlate 0.950. **Fine-tune diversity is not a lever.**

## 3. Checked and correct

* v11 = 0.5 z(nn_final5_val_oof) + 0.5 z(nn_final5_trainonly_val) reproduces 0.39793 from the 12 member files (exp1.log:11). nn_final5_val_oof = z-mean of the 6 `n2cpf` members (0.39784, matching `nn/logs/ensemble_final5.out`).
* My val folds equal nn/finetune.py's folds: same `GroupKFold(5).split(np.arange(n), groups=user_id)` on the same parquet order (`nn/finetune.py:194-202`, `nn/data.py:67-73`). No `--u0_singleton` appears in nn/*.sh. Fold sizes 2885/2885/2885/2884/2884. All 658 user-0 feeds are in fold 0, as review P7 already noted.
* The current combiner is optimal among simple fusions on two independent label sets (2.1). The current w_ft = 0.5 is the nested-CV choice in 5/5 folds (2.2).
* Val-to-test transfer of ensemble behaviour: member correlations, ft-to agreement and disagreement statistics match to <= 0.001 (2.4, 2.7).
* Test fine-tuned predictions are 5-fold means with fold correlation 0.9974, so there is no hidden train/test asymmetry in the fine-tuned branch (2.4).
* Earlier verdicts confirmed with new evidence:
  * "GBDTs / stackers add nothing to the NN blend" (CURRENT_STATUS 3, 3b): explained by the strength-diversity law (2.5).
  * "+0.1 z(pre_final) now hurts" (3j): pre2_final at any weight 0.05-0.5 is below v11.
  * "w_ft = 2/3 refuted" (REVIEW_SYNTHESIS:168): -0.0003 nested, ~0 under adversarial weights.
  * "per-segment blend weights unstable" (review_ideas:403): not reproducible across pools.
* nn_final5cd12 (CURRENT_STATUS 3k, +0.0008 +- 0.0004) is the same mechanism as D1 (+0.00076). D3 and D4 extend it with V13 and pass 2 SE.
* pre2_final ≈ pre_final (per-feed corr 1.000). This matches pretrain_report.md:257; it is not an error, but there is no point using both.
* No output of this study is written under preds/, subs/, nn/, src/ or data/. All outputs are in solvers/out/ensemble/.

## 4. Recommended FINAL design (pre-declarable) and what raises it most

**Design rule (freeze before nn_final6/7 land):**
1. **Families:** every completed NN family built on the live leak-free inputs, i.e. F5, F5cd, F6 (+ the extra V13 hedges lr2_mix1 / lr1_mix0), F7, plus any cooled continuations F6cd / F7cd.
2. **Strength gate**, judged on label sets not used for fitting:
   * the family's **mean single-member train-only** NDCG is within **0.002** of the best family's, on both val and the holdout;
   * the check is count-independent; calibration is in 2.6;
   * otherwise the family is excluded.
   On current pools this admits F5cd, V13 and F5 and rejects F4 (gap 0.003) and OLD5ep (0.005). GBDTs, encoder, stackers and pre_* are far outside it, consistent with the 2.5 break-even.
3. **Combiner:** per-feed z of every member. Fine-tuned branch = z(mean over all fine-tuned members); train-only branch = z(mean over all train-only members); final = 0.5 + 0.5. Member-equal (D4 form), no fitted weights, no segment weights, no rank fusion.
4. **Acceptance vs v11** uses the frozen rule unchanged: > 2 SE paired val, no slice < -2 SE, holdout ensemble >= 0.32162.
5. **Picks:**
   * Pick 1 = the full design.
   * Pick 2 = the same design without the newest family (e.g. without F7). That is the most different near-equal candidate, and it is robust to a late-family problem.
   * Until F6/F7 exist, **D4 (`solvers/out/ensemble/cand_D4_member_eq15_test.npy`) is the best available pick 1.**
   * Its pick 2 is the frozen hedge nn_final5cd12_mix:
     * D4 - nn_final5cd12_mix = +0.00038 +- 0.00032, i.e. within ~1 SE;
     * per-feed delta SD 0.038, so it is not a near-duplicate;
     * it differs from D4 by excluding the F6-like recipe.
   * v11 is 2.6 SE below D4.

**Quantified projection.** These are suspected values, extrapolated from the measured curves under stated assumptions.
* Family-level curve from the measured designs: 1 family 0.39793 → 2 families 0.39872 on average (D1, D2, eq(F5cd,V13)) → 3 families 0.39910.
  * a - b/k fit: a ≈ 0.3995, b ≈ 0.0016.
  * At current family strength: 4 families ≈ 0.3991, 6 families ≈ 0.3992, ∞ ≈ 0.3995.
* Strength uplift: V13-recipe single members are on average +0.0016 better on train-only val than v12 members (0.39408 vs 0.39247; same-seed paired +0.0027, nn_report.md:485). If F6 and F7 mixes are each ~+0.001 above F5's level, a 5-6-family design gains another +0.0004..+0.0008.
* **Projected final val ~0.3995-0.4005 (+0.0015..+0.0025 over v11)**, i.e. public ~0.392-0.393 and private ~0.394-0.395 with calib + typical residual (± 0.003).

**Gain per GPU-hour, highest first** (current ensemble ~15 members per branch):

| addition | GPU cost | expected ensemble gain | evidence |
|---|---|---|---|
| cooldown continuation (N2 recipe, 1 epoch, mix 0.1) of the 6 F6 plateau ckpts; same for F7 | ~3 GPU-h per family (30 min/member) + fta | +0.0004..0.0008 per family | F5cd: +0.0008 val / +0.0011 holdout on top of F5 (2.3, 2.6) |
| a new same-strength recipe family (F6, F7) of 6 members | ~15-20 GPU-h per family | +0.0004..0.0007 diversity + its strength uplift | V13 (3 members, same seed): +0.0005..0.0008 val, +0.0007 holdout |
| plateau snapshots (W1, e.g. 60k/70k/77.5k) of running members | 0 extra training, inference + fta only | unknown, likely < cooled ckpts (same trajectory) | not measured. Test first on the holdout: snapshots vs cooled members at equal count |
| 6 more seeds of an existing recipe | ~15-20 GPU-h | +0.0003..0.0004 | a - b/k fits: 6 → 12 = +0.00033..0.00045 |
| more fine-tunes of existing members | ~1 GPU-h | +0.0002 (n.s.) | exp11 |
| GBDT / encoder / stacker members | CPU | negative | 2.3, 2.5 |

## 5. Ranked proposals

1. **Adopt D4 as the current pick-1 candidate** (replaces v11; pick 2 = nn_final5cd12_mix). The submission file is built by the lead from `solvers/out/ensemble/cand_D4_member_eq15_test.npy`; this study does not write subs/.
2. **Freeze the final design rule of section 4 now:** member-equal, 0.5/0.5, strength gate, no fitted weights.
3. **Queue cooldown continuations for F6 and F7** as soon as their plateau checkpoints exist, and add them as families.
4. **Test W1 snapshots against cooled continuations on the holdout** before spending GPU on snapshots.
5. **Stop-list (saves time):** stackers of any kind, Caruana or val-fitted weights, rank / RRF fusion, segment weights, w_ft ≠ 0.5, adding F4 / OLD5ep / GBDT / pre_* / stk2, more fine-tune variants.

## 6. Literature anchors (extends nnx_research_*; [K] = from prior knowledge, not re-verified in this session)
* Score normalisation for fusion: z-score CombSUM is a strong default when member scores are comparable (Montague & Aslam 2001) [K]. RRF (Cormack et al. 2009) helps when scores are not comparable [K]. Here the members are comparable, so z wins and RRF loses 0.0007.
* Ensemble selection (Caruana et al. 2004) [K] overfits small validation sets without bagging. We measure +0.0014 optimism.
* Ensemble-size scaling (Lobacheva et al. 2020, power laws in deep ensembles) [K]: our NDCG(k) fits a - b/k^c with c 0.66-0.99.
* Ambiguity / bias-variance-covariance decomposition (Krogh & Vedelsby 1995; Brown et al. 2005) [K]: the measured law gain ≈ α(1-ρ) - β·gap is its empirical form for NDCG.
* Snapshot / checkpoint ensembles (Huang et al. 2017; Garipov et al. 2018) [K] and cooldown checkpoints (Hägele et al. 2024; nnx_research_training_gen.md) [K]: F5cd shows that a different point on the same run's schedule is a cheap, strong ensemble member.
* LTR stacking practice (Yahoo LTR 2010, Burges 2011) [K]: winners used linear blends of many LambdaMART / NN models. Non-linear stackers paid off only when base models used different inputs. Here all strong members share inputs, so a linear equal mix is the ceiling.
