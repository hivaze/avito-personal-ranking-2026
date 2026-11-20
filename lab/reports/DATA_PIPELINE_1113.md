# DATA PIPELINE + VALIDATION AUDIT (2026-11-13)

Provenance: synthesis of a four-lens audit (input pipeline, nonlinear dependencies and aggregates, validation representativeness and transfer, second-level combiners) run on 2026-11-13 with two skeptics per proposal. The synthesiser's text is reproduced below; the lens scripts, logs and extracts are in nn_x/audit/dpa_1113/. Section numbers are the synthesiser's.

## 0. TL;DR

1. **Input pipeline.** It is correct and clean.
   - The CPU rebuild reproduces the stored preds (corr 0.999999).
   - The ±6 clamp binds on ≤1.6% of rows, and only on sentinels.
   - Val→private PSI is ≤0.042 on 102 of 104 inputs.
   - No scaling, clipping or one-hot re-encoding has measurable value:
     - attention-mass cap excess: +0.00000;
     - warm-only popularity rank: val −0.00020, holdout −0.00049;
     - category as categorical: −0.00040 / −0.00071.
2. **Residual structure over G3.** There is none.
   - GBDT[G3 + 92 inputs] − GBDT[G3]: val +0.00005 ± 0.00021, holdout −0.00026.
   - Adding 38 feed/user/category aggregates: +0.00012 / −0.00045.
   - Of 94 label-free constructions, 6 reach z > 2 (2.1 expected) and 82 are negative on the holdout.
   - G3 under-weights user affinity (contrast ratio 1.12-1.22), but only outside each feed's top-10. Correcting it costs −0.0010 to −0.0034.
3. **Val vs test.** Val matches test on every intrinsic covariate (PSI vs public ≤ 0.01). Its only gap is train-derived item staleness: 5.7 d val, 9.3 d public, 11.2 d private, with no overlap between val and public. No reweighting fixes this: exact public-matched weights give transfer k 0.10, against 0.23 for unweighted val.
4. **The holdout is not test-like.** It has only M ≥ 50 feeds and fresh popularity (luo0 PSI 3.24 vs public, staleness 1.75 d). It is biased against popularity-light pools: val_TO(M≥50) − H is z +3.5 for Q1a and +2.8 for Q2a, and public sided with val on both.
5. **Public cannot order near-copies.**
   - The 6 public contrasts vs pure noise: χ² 5.38 on 6 df.
   - Transfer k: val 0.23 ± 0.26, holdout 0.71 ± 0.51, stale-val 0.45 ± 0.26. Stale-val's Kendall 0.71 is the best of 15 lenses picked after the fact (null P 0.21-0.34).
   - Level: calibM offset +0.0005 ± 0.0007.
6. **Second level.** 0 of 78 arms pass both reads.
   - Member-equal is the holdout optimum (in-sample best +0.00007).
   - Val-fitted weights anti-transfer (holdout z −2.85).
   - The M-conditional w_ft fails fresh-member replication (+0.00004 ± 0.00009).
7. **Tests.** One GPU arm survives two skeptics: FPCD-lite, prior 0.015.
   - Its pre-gate (my x2 run) lands in the grey zone: blank-vs-normal within-feed Pearson 0.989-0.990, Spearman 0.983-0.984.
   - POP-TILT closes: stale-val −0.00019 ± 0.00025 (my x1 run).
   - Everything else is closed or becomes a zero-cost info column.
   - No change to the decision rules.

## 1. How the data enters the NN

### 1.1 The chain (verified)

**Dense block, built in nn_x/w5/model.py dense_features (about lines 50-115).**
- Per-candidate f1..f5, each as raw value, within-feed z and percentile.
- Embedding norm, cos and dot with the user vector, candidate-history cosine statistics, counts.
- In total: 61 in-model columns plus 2 user0 columns.

**34 pre-computed cvnd_pre2s columns, attached by attach_xt (nn_x/w5/data.py:180-199).**
- seq_cv* and it_hist* get log1p; *_z columns are clipped to ±5; the rest are clipped to ±10; NaN becomes 0; storage is fp16.
- 31 columns are kept raw. The other 3 are kept only as relative versions.
- 5 xt_rel columns get an in-model within-feed mid-rank and z: seq_cv_invd20, it_hist_all_luo, pre2_fe_s, pre2_new_s, pre2_hmax32. That adds 10 columns.
- Result: n_dense = 104.

**Standardisation.**
- feature_stats (nn_x/w5/common.py:96) runs over 40 train batches; then (X − mean)/std with a ±6 clamp (model.py:527).
- The checkpoint statistics match a 20K-feed train sample: max |Δmean| 0.072 SD, std ratio 0.918-1.067.
- F7x and F10 use identical statistics.

**Tokens.** Linear(unit embeddings) plus learned categorical embeddings for category, event type and source, i.e. one-hot times a learned matrix.

**Parity check.** A CPU forward pass reproduces the stored val preds:
- r6_cd_m02_r101: 0.39733 vs 0.39730 stored.
- v23_gru_pre2s_s501: 0.39500 vs 0.39487.
- Both at corr 0.999999 (D/pipeline/out/s01b_parity.log).

### 1.2 What is right (measured)

- **Clamps.**
  - The ±6 clamp binds on ≤1.6% of val rows for any input, and only on sentinels: the smax family's no-history value −1.5 (1.4% of val, 2.6% of holdout) and user0_log1pH (1.6%).
  - The ±5 clip binds on 0%.
  - Builder caps never bind: seq_cv_pop and it_hist_all_luo each sit at their max on 0.00001 of train rows (D/pipeline/out/s02b.log §D).
- **fp16 ties.** At most 0.33% of within-feed pairs (ctx_nd_knn_f1); pre2_*_s 0.23%.
- **Support.** The median input has 0.03% of private rows outside the train [q0.1%, q99.9%] range (nominal 0.2%). The maximum is logM at 0.65% (D/pipeline/out/s06.log).
- **val→private drift.** PSI ≤ 0.042 on 102 of 104 inputs. The only exceptions are the trainer's own rank/z of it_hist_all_luo: 0.191 and 0.130 (D/pipeline/out/stats_X.csv, column psi_val_priv).
- **Categoricals are stable.**

  | categorical | PSI val→priv | PSI trn→val |
  |---|---|---|
  | candidate category | 0.0013 | 0.0175 |
  | history category | 0.0018 | 0.0030 |
  | event type | 0.0001 | 0.0269 |
  | source | 0.0002 | 0.0181 |
  | (type, source) | 0.0004 | 0.0279 |

  Source: D/pipeline/out/s02b.log §E.

### 1.3 Suspicious items and what they are worth

**(a) Cap at 10 on the four attention-mass inputs** (ctx_hn_soft05, ctx_hs_k128, ctx_hs_e1, ctx_hs_t10; val wAUC 0.646-0.653).
- Size:
  - The cap binds on 1.15 / 0.55 / 0.90 / 0.18% of val rows.
  - 26.0% of val feeds have ≥ 2 soft05 rows tied at the cap.
  - Capped-row CTR is 0.0384 vs 0.0158 overall.
- Measured worth:
  - Within capped mixed-label groups, G3 orders pairs correctly 0.681 of the time; the raw uncapped value manages 0.582 (holdout 0.650 vs 0.557).
  - Re-ordering capped rows by the raw value costs val −0.00099 ± 0.00028 and holdout −0.00220.
  - Nested cap-excess residual on G3: val +0.00000.
  - Uncapped − capped GBM, paired: val +0.00030 ± 0.00018, holdout −0.00003.
  - The whole ctx_hn + ctx_hs block is worth only +0.00051 ± 0.00057 val (abl_eval_wave2).
  - Sources: skeptic_hygA/k01.log, k02.log, out/k04_paired.log.
- Verdict: harmless. The FT stage already log-transforms these columns (nn/data.py:204), so the two stages differ only cosmetically.

**(b) Exact duplicates.**
- cos_u ≡ dot_u (r 1.0000); cmiss ≡ −log1p_norm (r −0.9999).
- pre2_*_z ≈ the in-model z of pre2_*_s (r 1.0000, but the stored _z is clipped at ±5 while the in-model one reaches −10.03).
- This touches 4 of 104 columns and only re-parametrises the first linear layer. There is no evidence of value.
- Verdict: cosmetic. Fold it into the next from-scratch family only.

**(c) The trainer's mid-rank and z of it_hist_all_luo / seq_cv_invd20 tie cold rows, so they encode the feed's cold share.**
- Size:
  - PSI val→priv: 0.191 (rank) and 0.130 (z) vs 0.024 for the raw count.
  - On warm rows only, PSI trn→priv is 1.252 for the trainer's rank vs 0.003 for a warm-only rank.
  - Cold-row mean rank goes from 0.088 on train to 0.236 on private.
- Measured worth of the exact spec as a nested G3 residual (skeptic_hygB/k02.log, skeptic_p2/k01.log):
  - val −0.00020, pmix −0.00008, holdout −0.00049 (z −2.97), late holdout −0.00049.
  - The cited +0.00051 came from the 2 columns outside the spec (et1, seq_cv_pop), i.e. the closed xrel_more family.
- Verdict: a real invariance defect with no measurable value. The information is already in the raw count and in the set attention.

**(d) pre2-blank regime.**
- Size:
  - 27.5% of train feeds and 26.1% of holdout feeds (Sep 6-8 Moscow time) have every pre2 input NaN→0 and has:pre2_fe = 0. This never happens on val or test.
  - 9 of the 104 inputs differ on these rows (skeptic recount; the lens listed 13 pre2-derived columns).
  - PSI trn→val of the pre2 columns is 0.85-2.36.
- Measured worth: blanking pre2 on val costs the four m02 roots −0.00191 / −0.00255 / −0.00223 / −0.00239 (D/synth/x2.log). The blank mode vs the normal mode:
  - within-feed Pearson 0.989-0.990;
  - Spearman 0.983-0.984;
  - top-1 agreement 0.830-0.840.
- Verdict: the model has a working, mildly different no-pre2 mode. This is the only untested mechanism, tested by FPCD-lite in §5.

**(e) Popularity coverage drift.**
- Raw it_hist_*_luo PSI is 0.15-0.47 trn→val and 0.20-0.68 trn→priv.
- luo0 (share of candidates with it_hist_all_luo = 0): train 0.128, holdout 0.220, val 0.379, public 0.428, private 0.456.
- The pop-free and pop1 families measure this (§3.3). The popularity-view families agent is already working on it.

**(f) No history timestamps; recency is the position index.**
- The H = 256 cap covers 42.0% of train feeds vs 29.0% of val feeds.
- H = 0 covers 3.01% of holdout feeds vs 1.35% of val feeds.
- Verdict: a data limitation, not a bug.

### 1.4 Categoricals and one-hots

- **Categorical embeddings.** Codes already enter as learned embeddings. Event types and sources are frequency-rank codes (CODES_DECODING_1113 §0).
- **Category as a categorical** in a residual GBM over G3: val −0.00040, holdout −0.00071 (z −3.07) (D/pipeline/out/s03.log T5).
- **f mass-point one-hot flags** (LIN): val +0.00027 (z 1.4), holdout −0.00026 (z −2.4) (T6).
- **Within-(feed, category) rank of G3:** val −0.00018, holdout −0.00024 (D/nonlinear/resid_B.log).
- **The 34 decoding-enabled cues** were already CLOSED (COVERAGE_MAP:204).

## 2. Nonlinear dependencies and aggregates

### 2.1 Residual structure over G3

Every test here is nested: user-grouped 5-fold on val, with the holdout scored by the full-val fit. Baseline: val 0.40032, holdout (20 TO members) 0.32379.

| probe | val Δ (SE) | holdout Δ (z) | source |
|---|---|---|---|
| LightGBM listwise, init z(G3), conservative config, 3 seeds: L1 (92 inputs) − L0 | +0.00005 (0.00021) | −0.00026 (−2.28) | D/nonlinear/agg_gb.log |
| L2 (+38 aggregates: within-(feed, cat) pct, feed mean/sd, category entropy/n_cat, cold proxy, history entropy/type shares, cat_last8/32, decayed same-category shares) − L0 | +0.00012 (0.00027) | −0.00045 (−3.35) | agg_gb.log |
| default config, L1−L0 / L2−L0 | −0.00016 / −0.00040 | −0.00058 / −0.00092 | agg_gb.log |
| residual MLP 64/32, L1−L0 / L2−L0 | −0.00010 / −0.00007 | −0.00037 / −0.00010 | agg_mlp.log |
| GBDT trained on the 30K holdout, applied to val, L1 (3 seeds) | −0.00037 / −0.00011 / −0.00065 | n/a | h2v_L1_*.log |
| GBM on all 104 inputs | −0.00067 (0.00048) | −0.00046 (−2.18) | D/pipeline/out/s03.log T0a |
| 16 key inputs + 16 feed aggregates | +0.00029 (0.00041) | +0.00008 | s03.log T4b |
| big GBM (300 rounds, 31 leaves; 104 + category + aggregates) vs a matched 121-noise-column control | −0.00506 vs −0.00475 | −0.00424 vs −0.00378 | s03.log T7a/b |
| feed-z of the 23 inputs that lack a within-feed version (LIN) | −0.00034 | −0.00100 (−3.8) | s03.log T2a |
| 94 label-free constructions: A affinity shapes, B aggregates, C within-feed relative versions of 28 xt inputs, D interactions | OOF z mean −0.01, sd 1.22; 6 at z > 2 (2.1 expected) | 82/94 negative, 38 at z < −2 | D/nonlinear/sum_b.log |
| the 2 formal passes: rel_ctx_nd_knn_h / rel_ctx_nd_knn_u | +0.00066 / +0.00050 | +0.00003 / +0.00004 | D/nonlinear/robust.log |

The two formal passes do not survive on G3's halves. Fitted nested on each half alone:
- knn_h: TO −0.00033, FT −0.00016.
- knn_u: TO −0.00044, FT +0.00005.

**Hygiene finding.** An uninformative tilt costs about −0.0002 to −0.0004 on the holdout. Evidence:
- the GBM noise control gives −0.00041 (z −2.15);
- a 4-column noise LIN reads holdout z −2.70 (skeptic_hygB);
- the 94 constructions average −0.00026.

So the holdout per-feed SE ignores weight-fit noise, and "holdout z ≥ 0" is a conservative criterion in this harness.

### 2.2 What G3 captures and what it misses (partial-dependence lens)

**G3 is under-confident on user-affinity inputs.**
- Contrast ratio CR = label lift / model lift, top decile vs bottom decile. On val:

  | input | CR (val) |
  |---|---|
  | ctx_hs_k128 | 1.219 |
  | f3 | 1.188 |
  | ctx_hn_margin | 1.170 |
  | ctx_hs_t10 | 1.158 |
  | ctx_nd_knn_h | 1.140 |
  | lse | 1.139 |
  | smax | 1.123 |

- On the holdout, CR is 1.05-1.12.
- Across 60 inputs, the val-vs-holdout Spearman of LR/df is 0.76.
- Sources: D/nonlinear/pd_val.log, pd_hold.log, pd_repl.log.

**The miscalibration sits in the tail.**
- Actual/expected in the lowest affinity bins is 0.84-0.90 (ctx_hs_k128 bin 0: 0.842, z −4.4).
- Inside each feed's G3 top-10 it is 0.99-1.00; in the rest it is 0.80-0.87 (D/nonlinear/pd_ctrl.log).
- NDCG is decided in the top-10, so this does not cost ranking quality.

**Correcting it hurts.** Nested bin-GAM over g-bins (D/nonlinear/gam_results.log):

| block | λ | val | holdout |
|---|---|---|---|
| affinity (10 inputs) | 10 | −0.00102 | −0.00119 |
| affinity (10 inputs) | 100 | −0.00088 | −0.00101 |
| affinity (10 inputs) | 1000 | −0.00031 | −0.00041 |
| all 59 inputs | 10 | −0.00343 (z −4.45) | −0.00311 |

**No saturation and no missed non-monotonicity.** There are 0 adjacent-bin reversals > 2 SE. f4, f2 and mx_e2-4 are non-monotone in the label, and G3 follows them.

**The ensemble already sharpens.** Single members are flatter still: train-only CR f3 1.28-1.32, ctx_hs_k128 1.36-1.40. Pooling and FT reduce this (D/nonlinear/cr_members.log).

**Interactions exist in likelihood but do not monetise.**
- Large likelihood excesses: ctx_hn_e1_rrw × feed-cold 5.29, pre2_hmax32 × M 5.15.
- Across 240 input × stratum cells, the val-vs-holdout correlation is +0.02.
- Only user0 × affinity replicates (3 cells), and guests are 4.6% of val. Guest-only tilts add +0.00002 / +0.00004 overall (u0_tilt.log).
- 0 of 12 interaction tilts pass (resid_D.log).

### 2.3 Popularity: the one regime-dependent residual

- **Over-reliance shows on val only.** CR on val: seq_cv_pop 0.957, it_hist_all_luo 0.964. On the holdout: 0.998 / 0.971.
- **Demotion tilts are + on val/pmix and − on the holdout and on each G3 half.** Example, rel_it_hist_et1_luo: val +0.00058, pmix +0.00093, holdout −0.00009; halves TO −0.00061, FT −0.00016 (robust.log).
- **POP-TILT** = G3 − 0.02 · z_feed(log seq_cv_pop):
  - val +0.00034 ± 0.00026, pmix +0.00075 ± 0.00043, holdout −0.00024 (z −1.80);
  - nested over 10 fold seeds: mean +0.00023;
  - its cold/warm signature matches the CLOSED cold offset: all-cold-positive feeds +0.0060 and all-warm −0.0011, vs +0.0058 / −0.0012 for the offset (skeptic_poptilt/check.log, coldwarm.log).
- **New (synthesiser): POP-TILT on the stale-val lens.**
  - Setup: G3 TO pool from the 20 *_both preds, with the stale counterfactual seq_cv_pop. Script D/synth/x1_poptilt_stale.py, output D/synth/x1.log.
  - Result: the pool's stale drop is −0.00089, reproducing skeptic_t1_stale.

    | w | live TO half | stale TO half | stale pmix |
    |---|---|---|---|
    | −0.01 | +0.00001 | −0.00015 (SE 0.00018) | −0.00008 |
    | −0.02 | +0.00014 | −0.00019 (SE 0.00025) | +0.00008 |
    | −0.03 | +0.00003 | −0.00010 (SE 0.00031) | +0.00023 |

  - The tilt fails the project's lens for cold/warm trades (CURRENT_STATUS §3: "stale-val for any cold/warm trade"). CLOSED.

### 2.4 Power caveat (applies to all of §2)

**Positive controls.** The same GBM probe on single-member bases recovers val +0.00076 (r6_cd_m02_r101), +0.00150 (v9_e10_s61) and +0.00165 (s2_v4_cv_s33). Each comes with a negative holdout of −0.00040 to −0.00068 (D/pipeline/out/s03_ctrl_*.txt). s2_v4_cv_s33 lacks the ctx and pre2 families, a known gap of 0.006-0.012, and the probe recovers about 15% of it.

**What the nulls do and do not bound.** They bound a post-hoc residual over G3 at about +0.0005-0.0010. They say little about representation changes at NN training time, which only NN arms can price. Those arms are already closed:

| arm | result | source |
|---|---|---|
| PBLD | val −0.00057, holdout +0.00042 | R7_ARCH_HPARAM.md:91 |
| PLE | −0.00043 | nnx_VERDICT.md:150 |
| xrel_more, 10 epochs | val −0.0019, holdout −0.0007 | COVERAGE_MAP:150 |
| FTOK | −0.00114 | — |

## 3. Validation

### 3.1 Distributions

Source: D/validation/out/s1_dist.log. PSI columns are measured against public.

| descriptor | val | holdout | public | private | PSI val | PSI hold | PSI priv |
|---|---|---|---|---|---|---|---|
| M mean | 96.4 | 154.4 | 103.6 | 100.7 | 0.0084 | 2.38 | 0.0019 |
| M<50 share | 0.409 | 0.000 | 0.387 | 0.392 | | | |
| H mean | 173 | 176 | 175 | 174 | 0.0030 | 0.077 | 0.0014 |
| uid0 | 0.046 | 0.051 | 0.041 | 0.041 | 0.0005 | 0.0024 | 0.0000 |
| user seen in train | 0.304 | 0.162 | 0.303 | 0.286 | 0.0000 | 0.113 | 0.0012 |
| category mix (53) | | | | | 0.0024 | 0.0089 | 0.0024 |
| n_cat | 12.0 | 14.7 | 12.6 | 12.4 | 0.0098 | 0.321 | 0.0018 |
| cold_tte | 0.348 | 0.356 | 0.439 | 0.477 | 0.64 | 1.27 | 0.12 |
| luo0 | 0.381 | 0.220 | 0.429 | 0.459 | 0.19 | 3.24 | 0.072 |
| warm-item staleness (d) | 5.69 | 1.75 | 9.34 | 11.23 | 5.08 | 8.29 | 2.56 |
| weekend share | 0 | 0.32 | 1.0 | 0.24 | | | |

**Staleness supports are disjoint.**
- val p90 is 7.42 d; public p10 is 8.17 d; private p10 is 9.71 d.
- By day: val Tue 4.64 d, Thu 6.85 d; public 9.03-10.09 d; private 10.09-12.08 d.
- The holdout's cold_tte is bimodal: feeds on day 20702 are 100% cold by construction.

### 3.2 Which proxy tracks public

Sources: D/validation/out/s2_transfer.log and D/skeptic_t1_stale/k1_check_v2.log.

Public deltas vs G3:

| contrast | public Δ |
|---|---|
| Cd | −0.00074 |
| Pall | −0.00038 |
| PF7x | −0.00010 |
| M2 | −0.00039 |
| S2 | −0.00035 |
| TOonly | −0.00047 |
| Q1a | +0.00007 |
| Q2a | +0.00039 |

Paired public SE is 0.00036-0.00060.

Fits of public = k · proxy (GLS). "8" = all eight contrasts above; "H6" = the six holdout-defined ones (Cd, Pall, PF7x, M2, Q1a, Q2a).

| proxy | k ± SE | gR² | LOO rmse | contrasts |
|---|---|---|---|---|
| val | 0.23 ± 0.26 | 0.15 | 0.00038 | 8 |
| calibM_pub | 0.25 ± 0.26 | 0.18 | 0.00037 | 8 |
| pmix_pub / pmix_priv | 0.19 / 0.17 | 0.12 | 0.00040 | 8 |
| EB_pub (new exact public-matched weights) | 0.10 ± 0.28 | 0.03 | 0.00043 | 8 |
| val_TO | 0.35 ± 0.27 | 0.39 | 0.00030 | H6 |
| holdout | 0.71 ± 0.51 | 0.47 | 0.00042 | H6 |
| stale_TO (S) | 0.45 ± 0.26 | 0.67 | 0.00026 | H6 |
| 0.5H + 0.5S | 0.72 ± 0.39 | 0.73 | 0.00029 | H6 |
| stale_final (FT + stale TO, the submitted form) | 0.19 ± 0.28 | 0.10 | 0.00053 | H6 |
| predict 0 | — | — | 0.00041 | — |

**The six H6 contrasts carry no detectable signal.** Against pure public noise their χ² is 5.38 on 6 df.

**S's lead is a multiplicity artefact** (Monte Carlo, D/skeptic_t1_stale/k1_check_v2.log):
- Best of 15 proxies reaching S's Δχ² of 3.59 under noise: P = 0.336.
- Max Kendall ≥ 0.714 under noise: P = 0.214.
- If the holdout is the true lens, S still beats it by the observed LOO margin with P = 0.172.

**The S − val_TO gap is one shared offset, not a staleness effect specific to each contrast.**
- For 5 of 6 contrasts the difference is −0.00034 to −0.00040.
- That offset is G3's own smaller stale drop: −0.00089, vs −0.00123 to −0.00128 for every other pool.

**Levels** (public − weighted val, mean over 9 pools; D/validation/out/s6_levels.log):

| weighting | offset |
|---|---|
| uniform | −0.00757 |
| calibM | +0.00052 ± 0.00066 (best) |
| pmix_pub | −0.00446 |
| EB_pub | −0.00244 |

Across generations (13 files): public = 0.0078 + 0.962 (± 0.029) · val, residual sd 0.00072.

### 3.3 The holdout's bias

**Same population, different verdict.** On M ≥ 50 feeds, train-only half, val_TO − holdout (D/validation/out/s7_m50.log):

| contrast | z |
|---|---|
| Cd | −1.1 |
| Pall | −1.2 |
| PF7x | −0.6 |
| M2 | +1.4 |
| Q1a | +3.5 (+0.00149 ± 0.00043) |
| Q2a | +2.8 (+0.00117 ± 0.00042) |

Public sided with val on both regime contrasts: public − holdout is +0.00059 for Q1a and +0.00041 for Q2a.

**Mechanism: popularity's value on the holdout decays with the row's train day, not with its cold share.** Base − pop-free, seed-paired (D/pipeline/out/s04b.log; skeptic_p4_fpcd/c01.log):

| slice | Δ |
|---|---|
| holdout Sep 6-8 | +0.01061 ± 0.00115 |
| holdout Sep 9-14 | +0.00899 ± 0.00083 |
| holdout Sep 17-19 | +0.00415 ± 0.00095 |
| val | +0.00251 ± 0.00081 |
| pmix | +0.00052 ± 0.00120 |
| holdout, cold share ≥ 0.3 | +0.00868 |
| val, cold share ≥ 0.3 | +0.00250 |

- The decline is a smooth gradient: Sep 6-8 vs Sep 9-14 z 1.1; Sep 9-14 vs Sep 17-19 z 3.9.
- Matching cold share does not remove it (last two rows).

**Other limits of the holdout.**
- 26.1% of holdout feeds are in the pre2-blank format, which never occurs at inference.
- The holdout cannot see FT/w_ft or M < 50 feeds.
- Member-order Spearman, val vs full holdout: 0.755 over 131 modern members (D/pipeline/out/s04.log).

### 3.4 Power

Source: D/validation/out/s4_power.log. MDE at 80% power (2.8 SE):

| effect type | val | holdout | public | private |
|---|---|---|---|---|
| near-copy pool | 0.00086 | 0.00060 | 0.00112 | 0.00073 |
| family-different pool | 0.00128 | — | — | — |
| single member | 0.0029 | — | — | — |

Reliability:
- Tue-vs-Thu replication of the 8 contrasts: corr +0.936.
- User-grouped half-split: +0.81.
- Only 25 val users have 2 feeds, so the user-clustered SE equals the iid SE.

### 3.5 Answer: are val and holdout representative, and should we change them?

**Val.** Representative of test on everything except train-derived staleness, which no reweighting can reach. Keep val unweighted as the population gate. Keep src/calib.py (M-only) for levels: it runs +0.0005 pessimistic, inside its ±0.003 band.

**Holdout.** Not representative. It is a valid falsifier for changes that do not touch popularity: 82 of 94 val-selected tilts failed it, as did every val-fitted residual. By construction it is biased against popularity-light changes and changes to how popularity is represented.

**Changes measured and rejected:**
- public-matched reweighting (EB_pub k 0.10);
- pmix as a promoter (k 0.19, i.e. about 5x overshoot of deltas; level −0.0045);
- stale-val as a decision lens (T1);
- as-of stale holdout (T4);
- true temporal holdout (T5);
- folding rl60 or holdout feeds into val: they are training feeds of every train-only member, so this would leak and mix regimes.

**Recommendation.** No change to the decision rules. Additions are pre-declared with `date -u` before the next arm.

- **R-A.** The frozen selection rule (CURRENT_STATUS §1) stays verbatim. Do not re-fit it on the 6 public points (VAL_PUBLIC_1113 l.23).
- **R-B.** Every arm or pool readout adds INFO columns with no gate role:
  - holdout E = train days 20702-20704 (Sep 6-8 Moscow time, 7,823 feeds);
  - holdout L = days 20705-20715 (22,177 feeds);
  - holdout late3d = days 20713-20715 (Sep 17-19, 8,279 feeds), defined from timestamps, as in r6_eval.setup;
  - stale-val S on the TO half: D/validation/s3b_stale_gpu.py, about 5 s per member of inference on GPU N. Copy it to a durable path first.
  - the regime diagnostic Δ_R = val_TO(M≥50) − H, with its SE. If |Δ_R| > 2.5 SE, the readout must say "holdout not decisive (popularity regime)"; the decision still follows R-A.
- **R-C.** Before every public read, log the predictions 0.28·V, 0.71·H, 0.45·S and the calibM level. Revisit the role of S only after ≥ 10 new public contrasts, or ≥ 3 contrasts with |0.45·S − 0.71·H| ≥ 0.0008 (about 2 public SE). Score by cumulative χ² against the holdout lens and disclose the 15-lens multiplicity. A LOO-rmse comparison on 3 points picks S 51% of the time even when the holdout is the true lens.
- **R-D.** Nested-residual protocol on G3:
  - RESID-HALVES: a residual counts only if its nested gain is ≥ 0 on the TO pool and on the FT pool separately.
  - Always report a matched-noise control column.

## 4. Second level: does anything beat member-equal on both reads?

No. Sources: D/stacking/*.txt, with the pre-declared plan in D/stacking/PLAN.txt (written 14:58:56Z); D/skeptic_freeze_stats; D/skeptic_stackfreeze.

**Counts over 78 tests.**
- Holdout z ≥ +2: 0 of 78.
- PASS-WEAK (val > 2 SE, holdout z ≥ 0, pmix ≥ −1 SE, no slice < −2 SE): 0.
- Val z ≥ 2: 8 arms. Seven are one mechanism (more TO weight on short feeds); the eighth is rawc.

**Pooling functions (no fitted parameters).**
- rawc: val +0.00023 ± 0.00010 (z 2.32), holdout −0.000003.
- Rank pooling costs the holdout 0.00015-0.00023.
- clip2: val −0.00127, holdout −0.00374 (z −15). The z tails carry signal.
- 15 of 18 variants have holdout ≤ 0.

**Family weights (F10 / F7x / F7x70).**
- Val in-sample maximum: +0.00107 over member-equal. Holdout in-sample maximum: +0.00007 at (0.3, 0.4, 0.3), i.e. essentially member-equal (0.30, 0.35, 0.35).
- Nested val fit: val +0.00049 (z 1.47), holdout −0.00048 (z −2.85).
- Holdout-fitted, OOF: −0.00031 (z −2.27). Over 30 random user splits: mean −0.00013, 0 of 30 splits above zero (v1_verify.txt).
- Weights fitted on the val TO form also anti-transfer: −0.00015 to −0.00029 (v3_estimand.txt). So this is not an FT/TO estimand artefact.

**M-conditional w_ft** (0.3 for M < 50, 0.6 for M ≥ 50).
- Nested: +0.00064 (z 2.96), stable across 30 splits (median z +3.14).
- But random same-size buckets already give +0.00027 (median z 1.23, max 2.29) (v2_wft_mcond.txt).
- Frozen rule on 10 fresh member sets: +0.00004 ± 0.00009 (z 0.41).
- On a wider w_ft grid it falls to +0.00023 (z 0.67).
- The holdout cannot read it.
- Verdict: a member-specific winner's curse, the same finding as the 11-12 M-split closure.

**Stackers.**
- GBDT lambdarank residual: val −0.0029 to −0.0048, holdout −0.0032 to −0.0041. Overfit: at 100 rounds, in-sample +0.0129 vs OOF −0.0051.
- MLP residual: val +0.00005 / +0.00022 (z 0.87); TO-twin holdout −0.00025 to −0.00034.
- MLP trained in-domain on the 30K holdout: own OOF −0.00024 to −0.00030.

**Near-copy ceiling.** The largest robust val gains here are +0.0002 to +0.0007. At near-copy transfer k 0.05-0.2 they map to +0.00001 to +0.00014 public, below the public SE of 0.0004-0.0006.

**Verdict.** Freeze the combiner at member-equal 0.5 z(FT) + 0.5 z(TO). With these 20 members, "build on top of the powerful models" has no second-level lever. Value can only come from members whose test within-feed rho to G3 is below 0.99.

## 5. Proposed tests, ranked

### 5.1 Surviving GPU arm

**T-1. FPCD-lite: m02 re-cool without the Sep 6-8 pre2-blank feeds.** Pipeline P4, amended by both skeptics.

**Mechanism (restated).** Remove the out-of-format regime (9 of 104 inputs blank, has:pre2_fe = 0; 27.4% of train feeds) from the m02 cooldown, so that the late, memorising phase fits only the input format seen at inference.

The popularity half of the original rationale is dropped (skeptic_p4_fpcd/c01.log):
- Sep 6-8 vs Sep 9-14 popularity advantage is +0.00162 ± 0.00142 (z 1.1).
- Dropping Sep 6-8 moves the train luo0 share only from 0.1280 to 0.1411, against val 0.379. That closes 5.2% of the gap.

**Evidence for.**
- The pre2 pathway is worth 0.0019-0.0026 per m02 root.
- The model keeps a distinct no-pre2 mode (D/synth/x2.log).

**Evidence against.**
- hcap, the format-parity cooldown arm: +0.00007, null.
- 0 of more than 20 cooldown-data arms beyond m02 reached +0.0005 member val with holdout ≥ 0.
- Cooldown-data arms are near-copies (same-root corr 0.993-0.997).
- rec33 (late-day oversampling): val −0.00212.
- Without Sep 6-8, the production-feed mix tilts to week 2 (share 0.462 → 0.636). Week-2 production transfers negatively to val: −0.00034, z −2.0 (nn_x/audit/external/findings.txt:62-63).

**Pre-gate.** The skeptic declared it before my run: corr ≥ 0.99 → CLOSE; corr < 0.98 → run. Measured blank vs normal, r6_cd_m02_r101-104 (D/synth/x2.log):

| statistic | value |
|---|---|
| within-feed Pearson | 0.9904 / 0.9898 / 0.9893 / 0.9904 |
| Spearman | 0.9840 / 0.9832 / 0.9826 / 0.9834 |
| top-1 agreement | 0.830-0.840 |

Neither branch fires. The blank mode differs from the normal mode about as much as the closed R-Drop/SAM cooldown arms differ from m02 (corr 0.990, COVERAGE_MAP:205), and much less than m02 differs from c0 (0.966-0.974).

**Spec.**
- New flag `--cd_exclude_days 0,1,2` (Moscow days from Sep 6, i.e. train days 20702-20704) in a write-new copy of the CD-harness train.py.
- The flag masks those feeds out of each pre-computed regular cooldown batch and each pool batch:
  - same 19,670 steps, lr and K9 schedule;
  - effective cooldown batch about 0.725x, inside wave 8's null 0.5x-2x band.
- Do NOT redraw or refill batches. Refilling raises passes per feed from 2 to 2.76, toward the cf4/e20 harm region.
- Log realised feeds per step, passes per feed and pool share.
- Add a holdout leak guard and a flag-off bit-identity test (AST plus same-seed logits).
- Roots 101-104 from @77500, m02, CDSEED 811.
- Control arm rnd27: the same masking applied to a seeded random 27.45% of non-holdout feeds (dates ignored), with the same roots and seed.

**Cost.**
- 8 cooldown runs (4 FPCD + 4 rnd27): about 3-3.5 GPU-h, plus about 1 h of build and tests.
- Stage 2 adds 3 runs, about 1.2 GPU-h.

**Pre-declared rule** (write it with `date -u` before any run). Primary read is @last; m02ref @last holdout preds already exist (preds/holdout/nn_w5_cd_m02ref_r10X@last_holdout.npy).
- **Stage 1 (roots 101-104).** Continue only if all hold:
  - FPCD − m02ref member val ≥ 0;
  - full holdout ≥ −1 SE (the E-regime handicap is disclosed in advance);
  - L-holdout (days 20705-20715) ≥ 0;
  - FPCD − rnd27 ≥ 0 on val AND on the L-holdout;
  - stale-val ≥ −1 SE and pmix ≥ −1 SE.
  - dE (the Sep 6-8 part) is INFO only and is expected to be negative.
- **Stage 2.** Decide on roots 105-107 alone, or require a 7-root z ≥ 2.5. Apply PASS-CD (R6_PLAN §4.1), with:
  - holdout = L ≥ 0 AND full ≥ −1 SE;
  - pmix ≥ 0.
- **Pick.** Any pick additionally needs the TRICKS §4.2 clause (stale-val AND pmix ≥ +2 SE), the generational bar and the frozen G3 rule.
- Otherwise CLOSED.

**Prior** P(≥ +0.0005 member val AND holdout/pmix not below): 0.015. The skeptics gave 0.02 and 0.0175; I lowered it for the grey-zone pre-gate. At pool level, near-copy transfer is 0.1-0.3x, so P(pool ≥ +0.0005) < 0.01.

**Priority.** Run it only on GPUs that would otherwise sit idle, below the popularity-view families (pop1 x8, pop2).

### 5.2 Zero-cost protocol items

These carry no gain prior by construction; they are decision hygiene.

- **T-2. Info columns and prediction ledger** (§3.5 R-B, R-C). Cost: CPU minutes per readout, plus about 5 s of GPU-3 inference per member for S.
- **T-3. RESID-HALVES and a matched-noise column** (§3.5 R-D). Evidence: both formal passes among the 94 constructions, and every popularity tilt, are negative on each half alone (TO −0.00033 to −0.00061; FT −0.00016 to +0.00005; robust.log).
- **T-4. FREEZE-ME.** The combiner stays at member-equal. A future combiner claim needs all three:
  - (i) holdout z ≥ 0 in TO form;
  - (ii) a fresh-member replication like D/stacking/e3c_fixed_rule.py, with mean > 0 at z ≥ 2;
  - (iii) a result above a placebo of random same-size buckets.

  Evidence: §4.
- **T-5. Cosmetic hygiene, with no screen.** Whenever a family is trained from scratch anyway: log1p on the 4 attention-mass columns, and drop dot_u and log1p_norm. Declare in advance that this gets no screen: the pass bar exceeds the whole ctx_hn + ctx_hs block's worth (+0.00051 ± 0.00057).

### 5.3 Refuted or closed

- **P1 HYG-A** (log1p on the mass columns plus dropping duplicates, 2-epoch pair). REFUTED by both skeptics.
  - The cap excess orders capped pairs worse than G3 (0.582 vs 0.681).
  - The cited single-member +0.00053 had no capped control; paired, it is +0.00005 ± 0.00019.
  - Learnable numeric encodings (PBLD, PLE) are already closed.
  - The pass bar exceeds the whole block's worth.
- **P2 HYG-B** (warm-only rank/z for luo and invd20). REFUTED by both skeptics.
  - The cited +0.00051 came from 2 columns outside the spec (the closed xrel_more family).
  - The exact spec: val −0.00020, holdout −0.00049 (z −2.97), late holdout −0.00049.
  - Setting cold z = 0 raises the all-row z PSI trn→priv from 0.188 to 0.553.
- **T1 STALE-TO as a decision lens or regime-conflict flag.** REFUTED by both skeptics.
  - S was picked post hoc from 15 proxies, and the public contrasts give χ² 5.38 on 6 df.
  - The S − val_TO gap is G3's own smaller stale drop, not contrast-specific staleness.
  - Choosing the judge from public outcomes repeats the HOLD-CLEAN failure (COVERAGE_MAP:160).
  - The pre-registered R5(a) holdout-break check already covers the flag.
  - S is kept as an info column only (R-B, R-C).
- **T4 stale holdout** (as-of d−7 popularity on the 30K holdout). My reading of the skeptic logs: it fails its own gates (skeptic_T4_validity/k1_asof_coverage.log, k2_rule_power.log; skeptic_T4_validation/se_gate.log).
  - 63.6% of holdout feeds get zero as-of history. luo0 becomes 0.845 vs public 0.429; PSI vs public 3.19, vs 3.24 for today's holdout.
  - On the 10,925 feeds that keep some history: PSI 1.56, and the SE ratio to S is 0.87-1.03 against a ≤ 0.7 gate (FAIL).
  - The Kendall ≥ 0.5 gate has a null pass rate of 0.068 at n = 6, and the current fresh holdout already scores 0.467.
- **T5 true temporal holdout** (train on days ≤ 20710, evaluate the 24,992 exploration feeds of days 20713-20715). My reading of the skeptic logs: not worth building (skeptic_T5/k3_t5_staleness.log, k1_counts.log, k2_byday.log).
  - Cutoff 20710 gives staleness 5.09 d and cold share 0.382: val-like, not public-like (9.34 d / 0.438).
  - Public-like staleness needs cutoff 20706 (8.64 d). That overshoots the cold share (0.535) and keeps only 45.5% of train feeds.
  - The evaluation set has no M < 50 feeds, and the MDE is about 0.0006.
  - Cost: about 13 GPU-h for a G3-like pool, plus 20-40 GPU-h per comparison.
  - The late days of today's holdout show no Q2a regime flip: day 20713 −0.00011 ± 0.00065; days 20714-15 +0.00005 ± 0.00048.
- **POP-TILT** (nonlinear lens). CLOSED: holdout −0.00024; same signature as the cold offset; stale-val −0.00019 ± 0.00025 (D/synth/x1.log).
- **EB_pub reweighting; pmix as a promoter or tie-break replacement.** CLOSED (k 0.10 / 0.19).
- **Also CLOSED (§2, §4):**
  - GBDT, MLP, bin-GAM, aggregate and interaction residuals over G3;
  - family weights, feed-conditional weights and holdout-fitted weights;
  - rank and robust pooling;
  - rawc pooling.

## 6. Registry one-liners and COVERAGE_MAP rows

### Registry (append)

- 11-13 16:10 DATA-PIPELINE + VALIDATION AUDIT (a workflow run; reports/DATA_PIPELINE_1113.md).
  - Input pipeline clean: CPU parity corr 0.999999; ±6 clamp binds on ≤ 1.6% of rows (sentinels only); PSI val→priv ≤ 0.042 on 102/104 inputs.
  - No nonlinear, aggregate or categorical residual over G3: GBDT L1−L0 val +0.00005 / holdout −0.00026; L2 +0.00012 / −0.00045; 82/94 constructions holdout-negative.
  - Val matches test except staleness (5.7 / 9.3 / 11.2 d, disjoint); EB reweighting k 0.10.
  - Holdout regime-biased against pop-light pools (Q1a z 3.5, Q2a z 2.8).
  - Second level 0/78.
  - One GPU arm proposed: FPCD-lite, prior 0.015.
- 11-13 16:05 POP-TILT (G3 − w·z_feed(seq_cv_pop)): stale-val TO −0.00019 ± 0.00025 at w −0.02 (live TO +0.00014); holdout −0.00024; cold-offset signature. CLOSED. (D/synth/x1.log)
- 11-13 16:04 FPCD pre-gate (pre2-blank mode of r6_cd_m02_r101-104 on val): NDCG −0.00191 / −0.00255 / −0.00223 / −0.00239; within-feed Pearson 0.989-0.990; Spearman 0.983-0.984; top-1 0.83-0.84. Grey zone of the skeptic gate (≥ 0.99 close, < 0.98 run). (D/synth/x2.log)
- 11-13 15:4x HYG-A, HYG-B and STALE-TO-as-rule refuted by 2 skeptics each. T4 (as-of stale holdout) fails its own SE and Kendall gates. T5 (temporal holdout ≤ 20710) gives a val-like staleness of 5.09 d: not built.

### COVERAGE_MAP rows

| idea | status | best evidence | file |
|---|---|---|---|
| log1p for the 4 attention-mass inputs (cap 10) + drop 4 duplicate inputs (HYG-A) | CLOSED 11-13 (2 skeptics; cosmetic-only default for future from-scratch families) | capped-group pair order G3 0.681 vs raw 0.582; nested cap-excess +0.00000; whole ctx_hn+hs block +0.0005 ± 0.0006 | DATA_PIPELINE_1113 §1.3 |
| warm-only within-feed rank/z of it_hist_all_luo / seq_cv_invd20 (HYG-B) | CLOSED 11-13 | warm-row PSI trn→priv 1.252 → 0.003, but nested spec val −0.00020, holdout −0.00049 (z −3.0), late −0.00049 | §1.3 |
| GBDT / MLP residual on [G3, 92 inputs] (+38 feed/user/category aggregates); holdout-trained residual | CLOSED 11-13 | L1−L0 val +0.00005 / holdout −0.00026 (z −2.3); L2 +0.00012 / −0.00045; MLP null; holdout→val no transfer | §2.1 |
| 94 label-free constructions (affinity shapes, aggregates, within-feed relative versions, interactions) as nested G3 residuals | CLOSED 11-13 | 6 at z > 2 (2.1 expected); 82/94 holdout-negative; 2 formal passes negative on both G3 halves | §2.1 |
| affinity calibration (bin-GAM; G3 contrast ratio 1.12-1.22) | CLOSED 11-13 | val −0.0010 to −0.0034, holdout −0.0012 to −0.0031; deviation sits outside the top-10 | §2.2 |
| category as categorical / f mass-point one-hots in the residual | CLOSED 11-13 | −0.00040 / holdout −0.00071; +0.00027 / holdout −0.00026 | §1.4 |
| POP-TILT (G3 − 0.02·z_feed(seq_cv_pop)) | CLOSED 11-13 | val +0.00034, pmix +0.00075, holdout −0.00024; stale-val TO −0.00019 ± 0.00025; signature = cold offset | §2.3 |
| exact public-matched val reweighting (entropy balancing, 11 moments) | CLOSED 11-13 | transfer k 0.10 vs 0.23 uniform; staleness supports disjoint | §3.2 |
| stale-val S as near-copy decision lens / regime-conflict flag (T1) | INFO ONLY 11-13 | Kendall 0.71, but 1 of 15 post hoc (P 0.21-0.34); public contrasts χ² 5.38/6 | §3.2 |
| as-of (d−7) stale holdout (T4) | CLOSED 11-13 (design) | 63.6% of feeds zero-history; luo0 0.845 vs public 0.429; SE ratio 0.87-1.03 on usable feeds | §5.3 |
| true temporal holdout (train ≤ 20710; T5) | NOT BUILT 11-13 | staleness 5.09 d (val-like); public-like needs ≤ 20706 (45.5% of train); no M<50; MDE ~0.0006 | §5.3 |
| second-level combiners (pooling functions, family / feed-conditional / holdout-fitted weights, GBDT and MLP stackers) | CLOSED 11-13 (FREEZE-ME) | 0/78 pass; holdout-fit OOF −0.00013 (0/30 splits > 0); M-conditional w_ft fresh +0.00004 ± 0.00009 | §4 |
| FPCD-lite (m02 re-cool without Sep 6-8 feeds, in-batch masking, rnd27 control) | PROPOSED 11-13 (prior 0.015; optional) | pre2 pathway worth 0.0019-0.0026 per root; blank-mode Pearson 0.989-0.990 (grey-zone pre-gate) | §5.1 |
| holdout E/L/late3d, stale-val S, Δ_R regime diagnostic, public prediction ledger | ADOPT 11-13 (INFO columns, no gate) | holdout popularity value decays by day: +0.0106 / +0.0090 / +0.0042; val_TO(M≥50) − H: Q1a z 3.5, Q2a z 2.8 | §3.5 |
====

## Appendix: skeptic-judged proposals (structured)

[
 {
  "name": "P4 FPCD-lite: m02 cooldown without Sep 6-8 feeds (CD harness)",
  "lens": "pipeline",
  "survives": true,
  "p_gain": 0.0175,
  "cost": "4 cooldown runs, about 1.5-2 GPU-h, plus about 1 h for the flag build and tests.",
  "rule": "Stage-1 futility: member delta vs m02 >= 0 on val AND >= 0 on late-holdout. Stage 2: PASS-CD (R6_PLAN 4.1) on roots 105-107. Otherwise CLOSED.",
  "verdicts": [
   "ok: NOT REFUTED. None of the allowed grounds applies to the arm as a whole. No closed row has the same mechanism. FPCD is recorded as DEFERRED, not CLOSED (TRICKS_SWEEP_1113.md:485; COVERAGE_MAP 'ANCHOR-REPLAY ... FPCD, PRE2T' row). Both re-entry halves are met as written: the label-free format check (TRICKS \u00a72b.5, 'E is out-of-format' 6/10) and the pop-free Q1 readout (CURRENT_STATUS log 06:39). There is no leak, because the selection is by timestamp only, without labels. Still, one half of the stated evidence is a statistical misreading, and recorded facts cut against the 'closer to test format' premise on two of its three axes. I measured all of this myself.\n\nScript: /tmp/scratchpad/dpa/skeptic_p4_fpcd/c01_check.py, output in c01.log in the same directory. It ran on CPU and reuses the pipeline lens's extracts and the stored preds.\n\n(1) The '1.5x popularity on Sep 6-8' claim is not specific to Sep 6-8. Holdout base minus xdrop_pop (4 seed-paired members):\n- Sep 6-8: +0.01061 \u00b1 0.00115\n- Sep 9-14: +0.00899 \u00b1 0.00083\n- Sep 17-19: +0.00415 \u00b1 0.00095\n\nContrasts:\n- Sep 6-8 vs Sep 9-14: +0.00162 \u00b1 0.00142 (z 1.1, not significant)\n- Sep 6-8 vs Sep 17-19: z 4.3\n- Sep 9-14 vs Sep 17-19: z 3.9\n\npop1 shows the same shape: +0.0014 \u00b1 0.0014 (z 1.0). The proposer's +0.0106 vs +0.0072 contrast pools Sep 9-19, and its low value is carried by Sep 17-19. That is a smooth time gradient, and its test-like end, Sep 17-19, is exactly what rec33 oversampled in the cooldown. rec33 did harm: val -0.00212 (z -4.3), holdout -0.00120, pmix -0.0023, stale-val -0.0023, coldpos -0.0084 (nn_x/r6/logs/r6_eval_cd_w4.txt).\n\n(2) The popularity dose is tiny. The train-sample LUO-zero share rises smoothly by day, from 0.091 on Sep 6 to 0.215 on Sep 19. Dropping Sep 6-8 moves it from 0.1280 to 0.1411. For reference: holdout 0.2204, val 0.3794, public 0.4276, private 0.4558. That closes 5.2% of the train-to-val gap and 4.0% of the train-to-private gap. The full-dose version of the same idea, hzcd (popularity-parity cooldown), was CLOSED as a cold/warm trade: val +0.0003, holdout -0.0005 (z -2.75), same-root corr 0.993 (nn_x/w5/logs/w5_eval_B_hzcd.txt).\n\n(3) The production-regime axis moves away from val. Without Sep 6-8, the share of production feeds from week 1 (Sep 6-11) vs week 2 (Sep 12-19) goes from 0.538/0.462 to 0.364/0.636. The external audit records that the post-Sep-12 production regime transfers negatively to val: -0.00034 \u00b1 0.00016 (z -2.0), M>=50 z -2.9; \"'use the most recent demand' is wrong here\" (nn_x/audit/external/findings.txt:62-63, 258-263).\n\n(4) That leaves the pre2-flag regime as the only intact and untested part of the mechanism. The model sees an explicit has:pre2_fe flag (train mean 0.725, always 1 on val/test). G3 scores the same on E and L: +0.0018 \u00b1 0.0027 (TRICKS \u00a72b.5). The nearest format-parity cooldown arm, hcap, was null (+0.00007 val, +0.00002 holdout), and TRICKS \u00a74.2 itself writes \"hcap (+0.00007) predicts null\".\n\n(5) Two problems with the rule:\n- Stage 1 gates on the late (L) holdout. The record allows L only as a pre-declared info column, and the full holdout stays the judge (TRICKS \u00a72b.5; COVERAGE_MAP HOLD-CLEAN row).\n- The stage-2 PASS-CD gates are on the full holdout, and 26% of its feeds are the Sep 6-8 regime that this arm stops cooling. That is the same handicap the reviewer flagged for hcap (CURRENT_STATUS:338).\n\nThe \u00a74.2 clause \"stale-val AND pmix >= +2 SE ... before any pick\" is also missing from the rule.\n\n(6) Base rate: 0 of more than 20 cooldown arms beyond m02 reached +0.0005 member val with holdout >= 0 (COVERAGE_MAP rows for waves 2-8, hzcd, AUXF, rdrop/sam). Cooldown-data arms are near copies (corr 0.993-0.997), so a member gain would transfer only about 0.1-0.3x to the pool.\n\nNet: this is a cheap test of one untested sub-mechanism, the pre2-flag regime. Its popularity rationale is wrong, and its regime rationale is partly contradicted by the record. | fix: 1. Rewrite the mechanism. Drop the popularity half: Sep 6-8 vs Sep 9-14 pop advantage is +0.0016 \u00b1 0.0014 (z 1.1), and the dose closes only 4-5% of the LUO gap. State the arm as a test of the pre2-flag regime only. Pre-declare the expected sign of the week-2 production tilt (0.462 -> 0.636) as negative on val M>=50 / Tue (external findings \u00a75.4).\n2. Keep the cooldown dose fixed. Implement --cd_exclude_days by masking Sep 6-8 feeds out of each precomputed cooldown batch (same 19,670 steps, effective batch ~0.725x, inside the flat cbs05..1x range of wave 8). Do NOT refill the batches, which would raise passes per feed from 2 to 2.76, toward the cf4/e20 memorisation regime. Apply the same masking to pool batches so that pool passes per feed do not rise 1.38x toward the closed m04 dose. Log the realised feeds per step and the pool share.\n3. Judge per the record:\n   - Stage-1 futility: member val >= 0 AND full holdout >= -1 SE (the E-regime handicap is disclosed in advance).\n   - Late-holdout (Sep 9-19) and dE/dL go in pre-declared INFO columns only (TRICKS \u00a72b.5).\n   - Add stale-val >= -1 SE and pmix >= -1 SE as vetoes. Per \u00a74.2, any pick additionally needs stale-val and pmix >= +2 SE, plus the generational bar.\n4. Optional cheap pre-gate before the GPU runs (inference only, GPU N): on r6_cd_m02_r101-104, re-predict val with the pre2 block zeroed and has:pre2_fe=0. If the within-feed corr to normal preds is >= 0.99, the flag=0 path barely differs and the arm is a near-copy (CLOSE). Run the arm only if corr < 0.98.\n5. Lower the prior to ~0.02. Report member corr vs m02ref and the 4-member pool delta, given the near-copy transfer of cooldown-data arms (0.993-0.997).",
   "ok: NOT REFUTED. No closed row tests this mechanism, there is no leak, and the readout flaws can be fixed. But half of the evidence does not hold up, and the prior should be lower than 0.04.\n\n(1) Why it stands.\n- No closed row isolates pre2-format parity in the cooldown. TRICKS_SWEEP_1113.md:485 DEFERS FPCD and does not close it. Both triggers are met: HOLD-CLEAN 6/10, and the pop-free Q1 readout was in at 06:39 (CURRENT_STATUS log 06:39).\n- The date filter uses train timestamps only, and the holdout is excluded by the existing guard pattern (cd_arms4.py:241). No leak.\n- Feed shares check out (scratchpad/dpa/skeptic_p4/k1_daygroups.py -> k1.log): Sep 6-8 is 27.43% of all train feeds, 27.45% of non-holdout feeds and 25.83% of pool-eligible feeds.\n\n(2) The popularity half of the evidence is a misread. Same script, base minus xdrop_pop, 4 seed-paired members, holdout:\n- Sep 6-8: +0.01061 \u00b1 0.00115\n- Sep 9-14: +0.00899 \u00b1 0.00083\n- Sep 17-19: +0.00415 \u00b1 0.00095\n- Sep 6-8 minus Sep 9-14 is +0.00162 \u00b1 0.00142 (z 1.1). pop1 gives the same picture (+0.00140 \u00b1 0.00137).\n\nSo the \"1.5x\" comes from the last three days. Sep 6-8 is not a separate popularity regime, and after FPCD 67% of the cooldown is Sep 9-14 with the same popularity reliance. By my arithmetic from the day shares, FPCD moves the cooldown's mean popularity reliance from 0.0083 to 0.0074. rec33 moved it further, to 0.0069, and lost val -0.00212 (z -4.3) and holdout -0.00120 (r6_eval_cd_w4.txt). rec33 is confounded by about 2x repetition, so this undercuts the claim but does not refute it.\n\n(3) Changing the cooldown's day mix leaves no day-specific trace on the holdout (k1.log, E minus L per-feed delta):\n- rec33: +0.00035 \u00b1 0.00067. It over-sampled Sep 17-19, yet lost the most there (-0.00153).\n- rho70: -0.00070 \u00b1 0.00078.\n- hcap: +0.00001 \u00b1 0.00031.\n- m02 minus c0: +0.00054 \u00b1 0.00062.\n\"Fitting the cooldown on rows closer to the format\" has no measured effect.\n\n(4) The pre2 half is real but small.\n- Only 9 of the 104 inputs differ on Sep 6-8 rows, not 13 (checked on the proposer's hold_X). One of the 9 is the explicit has:pre2_fe missingness flag.\n- Inference-only CPU probe (skeptic_p4/k2_pre2blank.py -> k2.log): putting val into the Sep 6-8 pre2 format costs r6_cd_m02_r101 -0.00191 \u00b1 0.00056 and v23_gru_pre2s_s501 -0.00205 \u00b1 0.00051.\n- So the whole pre2 pathway is worth about 0.002. Reaching +0.0005 would need FPCD to raise its effective value by 25% or more, just by removing flag-marked rows from the last 20% of training. The probe also shows the trained model has a working no-pre2 mode.\n\n(5) Statistical flaws in the spec.\n- **(a) Dose confound.** If cooldown batches are redrawn at a fixed step count (the CoolStream/rho70 template), passes per remaining feed rise x1.378 (2 -> 2.76 epoch-equivalents) and pool passes x1.35. Both point in measured harm directions: e3 coldpos z -4.2, m04 val z -2.3, and rho70 used this template (-0.0035).\n- **(b) Selection bias.** The CD harness picks the checkpoint on the full holdout, and 26% of it (7,823 feeds) is the format the arm stops training. That biases selection and the stage-2 \"holdout >= 0\" gate against the arm.\n- **(c) Winner's curse.** PASS-CD pools 7 roots, including the 4 that passed stage 1. My Gaussian arithmetic puts the bias at about +0.6 SE_7 under the null, so the null pass rate of the 2-SE bar goes from 2.3% to about 4%.\n- **(d) Missing pick clause.** The rule leaves out the \u00a74.2 pick clause recorded on the POP-REL row (line 484, on the same trigger): stale-val AND pmix >= +2 SE, plus the generational bar. | fix: 1. **Implementation.** Filter Sep 6-8 feeds out of each pre-computed regular and pool cooldown batch. Keep the same 19,670 steps and lr/K9 schedule, so passes per feed stay at 2 (cooldown) and 0.93 (pool). The effective cooldown batch becomes about x0.725, which is inside wave 8's null range of 0.5x-2x. Do NOT redraw through CoolStream. Log passes per feed. The flag-off bit-identity test stays.\n\n2. **Control arm rnd27** on the same roots 101-104 and CDSEED 811: the same in-batch filter applied to a seeded random 27.45% of non-holdout feeds, ignoring dates. The format mechanism counts only if FPCD minus rnd27 is >= 0 on val and on the L-holdout. This separates \"format\" from \"less cooldown data\". Cost: +4 runs, about 1.5 GPU-h.\n\n3. **Readout, pre-declared with date -u before any run.**\n- Primary read is @last for both arms; m02ref @last preds already exist (preds/holdout/nn_w5_cd_m02ref_r10X@last_holdout.npy). Holdout-selected checkpoints are information only.\n- The holdout judge at BOTH stages is the L part (Sep 9-19 MSK, 22,177 feeds, defined from train timestamps as in r6_eval.setup).\n- The E part (Sep 6-8) is a dE info column and is expected to be negative by construction.\n\n4. **Stage 2.**\n- Decide on roots 105-107 alone, or require a 7-root z >= 2.5, to remove the stage-1 filter bias.\n- Keep PASS-CD's slice, coldpos and stale-val conditions, and add pmix >= 0.\n- Any pick also needs the TRICKS \u00a74.2 clause (stale-val AND pmix >= +2 SE, plus the generational bar) and the frozen G3 rule.\n\n5. **Correct the evidence before launch.** It is 9 pre2-derived inputs, not 13. The popularity contrast is a gradient ending in the last three days (Sep 6-8 minus Sep 9-14 = +0.0016 \u00b1 0.0014), not a Sep 6-8 regime."
  ]
 },
 {
  "name": "P1 HYG-A: log1p for the attention-mass inputs and removal of 4 exact duplicates (representation-only 2-epoch arm)",
  "lens": "pipeline",
  "survives": false,
  "p_gain": 0.02,
  "cost": "2 runs x 37 min (the abl_base 2-epoch time) = about 1.2 GPU-h, plus about 1 h build and tests. CPU readout.",
  "rule": "Promote to a 10-epoch pair only if all hold: pooled member val > +1.5 SE, holdout >= 0, pmix >= -1 SE, and no val slice (M<50, M>=50, Tue, Thu) < -2 SE. Otherwise CLOSED. At 10 epochs the frozen E rule and the pool-level rule apply.",
  "verdicts": [
   "REFUTED: REFUTED, for three concrete reasons. I measured everything on CPU. Scripts are in /tmp/scratchpad/dpa/skeptic_hygA/ (k01_resid_capctrl.py, k02_capped_auc.py, k03_aucse.py, k04_paired.py). Logs are in out/.\n\n(1) STATISTICAL ERROR in the only positive number in the evidence. The single-member 'T1c +0.00053 (z 1.1)' has no capped control. I re-ran the proposer's own harness (s03_resid.py, user-grouped 5-fold, full-val weights applied to the holdout) on base r6_cd_m02_r101, adding K1 = the same 8 columns with the cap kept (min(x,10)):\n- K1 alone gives val +0.00048 (z 0.98) and holdout -0.00019.\n- Paired uncapped minus capped: val +0.00005 \u00b1 0.00019 (z 0.25), pmix -0.00022, holdout -0.00004 (k04_paired.log).\n- So the +0.00053 is GBM re-ordering of a single member, and the cap contributes nothing to it.\n- On G3 the paired uncapped minus capped is val +0.00030 \u00b1 0.00018 (GBM, below the +0.003 noise bar), pmix +0.00007, holdout -0.00003. In the LIN version (log-z): val -0.00007 \u00b1 0.00010, holdout +0.00002.\n\n(2) CONTRADICTION WITH RECORDED FACTS for mechanism (1), 'restore ordering above the cap'.\n- reports/featrev_research.md:51 and :159-170 (RESEARCH-4, 'verified') tested these exact four columns. The \u00b110 clip is harmless: the positive-rate lift above 10 (2.3-3.2) equals the lift in the 8-10 band. The 1.15% share of capped soft05 rows matches the proposer's s02b.\n- My k03 check: inside groups of capped rows that contain a positive (val 379 feeds), the uncapped value has AUC 0.58 against G3's 0.68 (diff -0.099 \u00b1 0.025).\n- On the pairs G3 gets wrong, the uncapped value orders correctly only 0.44 \u00b1 0.03 on val and 0.41 \u00b1 0.02 on the holdout (below chance). The cap excess does not fix G3's errors.\n- Contested capped groups hold 0.3-2.7% of positives (k02).\n\n(3) CLOSED ROWS for mechanism (2), 'bulk resolution for the NN'.\n- r7 pbld learned per-feature RealMLP periodic encodings on all 104 standardised dense inputs. These include the four mass columns (nn_x/r7/model.py:306 and :687 PBLD(n_dense); r7_pbld_s23.json: xt cvnd_pre2s, pbld 16, 2 epochs).\n- It ran on the same 2-epoch screen with seeds 23/24: val -0.00057 \u00b1 0.00061, holdout +0.00042, FAIL (R7_ARCH_HPARAM.md:91; CURRENT_STATUS.md:92 'pbld null').\n- B1 PLE (quantile piecewise-linear numeric embeddings) is CLOSED: val -0.00043, holdout +0.00078, 'NO' (nnx_exp_feat.md:16; COVERAGE_MAP 'PLE \u2026 CLOSED').\n- A fixed log1p widens the bulk by only 1.79 to 2.23 std units, which is the proposer's own s02b number.\n\nBound: removing the entire 10-column ctx_hn + ctx_hs block costs only val +0.00051 \u00b1 0.00057 and holdout +0.00002 (nn_x/abl/logs/abl_eval_wave2.txt:66). The rule needs about +0.0008-0.0009 val at 2 epochs (+1.5 SE_cal at about 0.0005-0.0006). That is more than the whole block is measured to be worth.\n\nThe exact proposal (F5, log1p of these four columns) was already pre-estimated at '\u2264 +0.0003, only worth adding in a run that changes the preset anyway' (featrev_research.md F5, around line 295).\n\nMechanism (3), dropping duplicates (cos_u = dot_u, cmiss \u2248 log1p_norm, pre2_*_z \u2248 in-model z of pre2_*_s; s05 confirms r = 1.0000 / -0.9999), is not closed. It is only a reparametrisation of the first linear layer, and the proposal offers no evidence that it has any value.\n\nSide note: the proposer's s03_resid.py imports s02_stats, which opens out/s02.log with 'w' at import. This truncated their s02.log (now 0 bytes). My copy avoids that. | fix: Do not spend a dedicated 2-epoch pair on this. If kept at all, fold --xt_log (the four columns) plus --drop_dup into the next arm that already changes the preset, as F5 recommended, and treat it as hygiene with no claimed gain. The capped control (K1) belongs in any future residual evidence. Before any GPU time, a zero-cost check could run: an inference-only probe that feeds the uncapped values to an existing checkpoint is meaningless (the model was trained on capped inputs), so the honest options are no test or a free rider on an unrelated run.",
   "REFUTED: REFUTED on four concrete grounds. Each is measured or cited. My scripts and logs are in /tmp/scratchpad/dpa/skeptic_hygA/ (k01_capgroup.py with k01.log, k02_capexcess_nested.py with k02.log, k03_member.py with k03.log). All ran on CPU only, with 12 threads.\n\n(1) Mechanism (1), \"restore the ordering the cap destroys\", is contradicted by measurement. The capped group is the rows with raw value >= 10, which the trainer sees as exactly 10.\n- k01 on val: in 379 feeds with mixed labels inside the capped group, G3 already orders the soft05 pairs correctly 0.681 of the time; the raw uncapped value gets 0.582. On the holdout it is 0.650 vs 0.557. Single members also beat raw: r101 0.660, s501 0.666.\n- Keeping G3's positions but re-ordering the capped rows by raw value costs val -0.00099 (SE 0.00028) and holdout -0.00220 (SE 0.00027) for soft05. For ctx_hs_e1 it costs -0.00037 val and -0.00161 holdout.\n- k02, nested user-grouped 5-fold on G3, holdout scored with the full-val weight. The destroyed information is cap excess = log(max(x,10)/10). For the sum of the 4 columns, the in-sample best val gain is +0.00000, the nested val delta is -0.00002 and the holdout delta is +0.00000. Per column, nested val runs from -0.00004 to -0.00022, with hs_k128 at z -2.3 and hs_e1 at z -2.1. The noise control's in-sample optimism is +0.00027.\n- The log-vs-capped representation direction gives val -0.00027 and holdout -0.00025 (z -1.84).\n- k03, member-level: cap excess gives val +0.00004 / +0.00003, holdout +0.00004 / -0.00001. The representation direction gives holdout -0.00058 (z -3.0) for r101 and -0.00075 (z -3.0) for s501. Its val sign flips between the two members.\n- These agree with the proposer's own G3 results (T1b -0.00023, T1c -0.00045 with holdout -0.00034).\n\n(2) The claim \"information content unchanged, a GBDT would be invariant\" is false. Uncapping does restore information. The GBDT test that is not invariant to it (T1c: uncapped log plus feed rank) is the one that came out negative.\n\n(3) Mechanism (2), \"more resolution in the bulk\", already has closed rows. RealMLP PBLD periodic embeddings and B1 PLE are learnable per-feature non-linear encodings of the same standardised dense inputs. They can learn any fixed log-like rescale. Results:\n- pbld: val -0.00057, holdout +0.00042, FAIL (reports/R7_ARCH_HPARAM.md:91).\n- PLE: val -0.00043 (reports/nnx_VERDICT.md:150).\nMechanism (3), the duplicates, adds no information and the columns are not exact duplicates. pre2_fe_z is pre2_fe_s_zf clipped at \u00b15: train min -5.0 vs -10.03 (pipeline/out/stats_X.csv rows 91 and 99). Dropping columns also changes parameter shapes, so seeds 23/24 are no longer paired on initialisation.\n\n(4) Ceiling and statistical errors.\n- Removing all 10 ctx_hn and ctx_hs columns is worth only val +0.00051 \u00b1 0.00057 and holdout +0.00002 \u00b1 0.00031 (nn_x/abl/logs/abl_eval_wave2.txt; COVERAGE_MAP.md:191). The proposal's bar of > +1.5 SE_cal needs about +0.00074 to +0.00092 val (SE_cal 0.00049 to 0.00061 in the same file). A true pass would need re-encoding 4 of the 10 columns to beat the worth of the whole block.\n- Hypothesis, not measured: P(pass | null) is about 0.03, so a pass would most likely be noise.\n- The cited single-member evidence is selectively reported. \"T1c +0.00053 (z 1.1)\" is the best of the 3 members tested; the other two gave -0.00009 and -0.00014. The holdout was negative for all 3 (-0.00023, -0.00001, -0.00008). That member's own noise control gave -0.00057 (pipeline/out/s03_base_*.log).\n- Effects seen at 2 epochs have shrunk or vanished at 10 epochs before: pre2s fell from +0.0019 to +0.0009, and the resdrop0 2-epoch gain did not hold up when tested at 10 epochs (COVERAGE_MAP).\n\nNo leakage found: these are feed-internal inputs, and log1p of a non-negative value cannot leak. | fix: Do not spend GPU time on this arm. If the team wants the cleanup for code tidiness, fold it in as a default only when a new family is trained from scratch anyway: log1p on the 4 mass columns, and drop dot_u / log1p_norm, which correlate at |r| >= 0.9999 with cos_u / cmiss. Declare in advance that this is cosmetic and gets no screen; at 2-epoch power it cannot be measured, since the bar exceeds the whole block's worth. Any re-open needs a CPU pre-gate that has not passed yet: the member-level nested residual of the log-vs-capped representation must be >= +0.0003 val with holdout >= 0 on at least 2 of 3 members. Today it is holdout -0.00058 and -0.00075 (k03.log)."
  ]
 },
 {
  "name": "P2 HYG-B: warm-only within-feed rank and z for the popularity counts in xt_rel",
  "lens": "pipeline",
  "survives": false,
  "p_gain": 0.02,
  "cost": "About 1.2 GPU-h, plus about 10 GPU-min of stale-val replay and about 1 h build (shared with P1).",
  "rule": "Promote only if all hold: val > +1.5 SE, pmix >= 0, stale-val delta >= 0, late-holdout >= -1 SE, full holdout >= -2 SE (veto only, because of the documented in-period popularity bias), and no val slice < -2 SE. Otherwise CLOSED.",
  "verdicts": [
   "REFUTED: Statistical error: the cited evidence does not measure the proposed change. The G3 LIN result (val +0.00051 z 1.5, pmix +0.00101, holdout -0.00081 z -3.8) is the proposer's T3a in dpa/pipeline/s03_resid.py. Its f_warm uses POP = [it_hist_all_luo, seq_cv_invd20, it_hist_et1_luo, seq_cv_pop], so it is 4 columns x (warm rank, cold flag). P2 changes only 2 of them. The other two are exactly the columns whose rel channels were closed by xrel_more (COVERAGE_MAP.md:150: 10-ep val -0.0019, holdout -0.0007, pmix -0.0032).\n\nI re-ran the same nested residual on CPU: dpa/skeptic_p2/k01_spec_resid.py, log k01.log. It uses the G3 base, user-grouped 5-fold, and holdout weights fit on full val.\n- A, 4-column reproduction: val +0.00041 (z 1.23), pmix +0.00092, hold -0.00079 (z -3.62), late (Sep 17-19, 8,279 feeds) -0.00061.\n- B, P2's 2 columns, warm rank + flag: val -0.00013 (z -0.44), pmix -0.00013, hold -0.00045 (z -2.67), late -0.00052 (z -1.81).\n- C, exact spec (warm rank + warm z + flag): val -0.00036 (z -1.05), pmix -0.00029, hold -0.00054 (z -2.96), late -0.00055 (z -1.64).\n- D, the 2 columns outside the spec (et1_luo, seq_cv_pop): val +0.00027, pmix +0.00052, hold z -3.33. This is where the cited gain comes from.\n- E/F/G, control with the trainer's TIED rank in place of the warm-only rank: 2-col E val -0.00011, pmix +0.00003, hold z -2.46; 4-col G val +0.00032, pmix +0.00074, hold z -3.92. These are about the same as A and B, so the residual does not tell warm-only from tied encoding at all.\n\nSo on the proposed spec, every judge is negative: val, pmix, full holdout and the proposer's own late holdout. The 'biased holdout' escape therefore does not apply. The project's G3 residual hygiene rule (holdout z >= 0) fails on every variant. The proposer's own T3b GBM version was also val -0.00020 with hold z -3.51.\n\nWhat survives is only the label-free mechanism, from k02_psi_trn.py / k02.log. On warm rows, PSI trn->priv of the it_hist_all_luo rank is 1.272 for the tied rank vs 0.003 for the warm-only rank. That is the same information re-parameterised: the net already sees the raw log count, and the cold share via set attention.\n\nThe spec also has a design flaw. Cold z = 0 puts the 45.6% cold private rows at the warm MEAN. This raises the all-row z PSI trn->priv from 0.188 (tied) to 0.553 (warm z).\n\nThe closest analogous lever is the FT-extras warm-only ranks (AUDIT_1112 L4-C / nn_x/audit/features/findings.txt P5). It was proposed with expected <= +0.0003 and never run. The val+/holdout- and coldpos+/warmpos- signature (coldpos +0.0023..+0.0032, warmpos -0.0006..-0.0015) matches the closed cold/warm trades (hzcd, COVERAGE_MAP.md:144; DOMAIN-DIAG, line 157). | fix: 1. Drop the T3a numbers as evidence. If the arm is kept as label-free hygiene, it should be justified only by the input-invariance PSI: warm-row rank trn->priv 0.003 vs 1.272 (dpa/skeptic_p2/k02.log).\n2. Encode cold rows so they do not collide with the warm mean. Two options: (a) give cold z a constant below the warm-row minimum, plus an explicit cold-flag channel; or (b) replace only the rank and keep the trainer z. The spec's cold z = 0 raises the all-row z PSI trn->priv to 0.553 from 0.188.\n3. Before GPU, run a train-fitted proxy instead of a G3 residual: the GBDT proxy nn_x/insights/data_usage/07_gbdt_arms.py, with tied vs warm-only rank/z of the 2 spec columns as features, 3 seeds. Read val, pmix and the stale-val replay. Go to GPU only if stale-val and pmix are >= +2 SE and val >= -1 SE (the AUDIT_1112 L4 rule). A G3 residual cannot price an input re-encoding of information the net already has.\n4. Use the full holdout as a gate at the standard >= -1 SE, not as a -2 SE veto. The late-holdout escape fails anyway: the spec's late holdout is -0.00052 (z -1.81) on the CPU proxy.\n5. Cheaper same-mechanism alternative: AUDIT_1112 L4-C (FT extras warm-only ranks, about 1.5 GPU-h, 8 members x 2 FT seeds) tests the same idea without a from-scratch pair.",
   "REFUTED: REFUTED: the evidence is misattributed, which is a statistical error. I re-measured it on CPU with the proposer's harness (G3 base, LIN listwise softmax, user-grouped 5-fold on val, full-val weights applied to the holdout). Late holdout = 8,279 feeds. Scripts are scratchpad/dpa/skeptic_hygB/k01_check.py and k02_exactscale.py; logs are k01.log and k02.log.\n\n(1) I reproduced the cited T3a exactly: val +0.00051 (z 1.52), pmix +0.00101, holdout -0.00081 (z -3.77). But T3a has 8 features: warm rank plus a cold flag for 4 columns. Two of those columns, it_hist_et1_luo and seq_cv_pop, are not in the spec.\n- T3a restricted to the spec's 2 columns (luo, invd20): val -0.00018 (z -0.61), pmix -0.00002, holdout -0.00047 (z -2.82).\n- T3a on the 2 other columns only: val +0.00029, pmix +0.00062.\n- So the cited val, pmix and cold-feed gain comes from et1 and seq_cv_pop. Adding those as in-model rel inputs is already CLOSED: xrel_more, COVERAGE_MAP.md:150, 10-epoch FAIL with val -0.0019 and pmix -0.0032.\n\n(2) The exact spec encoding, built on the trainer's log1p+fp16 scale (parity with the trainer's rk/z: max diff 3e-8 and 3e-3), adds nothing over the controls:\n| arm | val | pmix | holdout | late holdout | cold-share>=0.45 slice |\n|---|---|---|---|---|---|\n| spec (warm-only rank/z, cold rows 0) | -0.00020 (z -0.66) | -0.00008 | -0.00049 (z -2.97) | -0.00049 | -0.00010 |\n| spec + cold flags | -0.00040 (z -1.18) | -0.00012 | -0.00034 | -0.00035 | n/a |\n| control: trainer midrank+z, same 2 columns | +0.00003 (z +0.13) | -0.00018 | -0.00037 | n/a | n/a |\n| control: 4 N(0,1) noise columns | +0.00022 (z +0.89) | +0.00044 | -0.00037 (z -2.70) | n/a | n/a |\nThe noise control shows the holdout per-feed SE ignores weight-fit noise, so holdout z is not calibrated in this harness, in either direction.\n\n(3) Winner's curse: T3a was the best val z of 19 tests in pipeline/out/s03.log.\n\n(4) The pre-declared rule moves the goalposts after the holdout failed. It downgrades the full holdout to a -2 SE veto, while the project's hygiene rule requires holdout z >= 0.\n\n(5) The PSI mechanism is partly real, but its framing is off. For a train-fitted NN the relevant drift is trn->priv, not val->priv:\n- luo rank: trainer 0.803 vs spec 0.556.\n- luo z: trainer 0.863 vs spec 0.581.\n- invd20: trainer 0.55/0.59 vs spec 0.025/0.026.\n- Warm rows only, luo rank: 1.252 vs 0.003.\n- Most of the remaining luo drift is the cold-share point mass (0.128 -> 0.456), which the raw count also carries.\n- A closed experiment already tried this kind of fix. nn_report.md:253-261 (R2 step 2) used ID-input dropout to simulate cold rows before the in-model rank/z. It gave val -0.0004/-0.0007 with a lower holdout (cold +0.013, warm -0.0043) and was rejected (q=0).\n\nNet: the only remaining support is PSI. P(+0.0005 val with the holdout and pmix not below) is about 0.02. | fix: Do not run as specified. If it is run at all:\n1. Restate the evidence with the spec-only measurements: spec encoding val -0.00020, pmix -0.00008, holdout -0.00049 (scratchpad/dpa/skeptic_hygB/k02.log).\n2. Keep the project's standard gate: holdout z >= 0, not a post-hoc -2 SE veto.\n3. Pre-declare a noise-calibrated holdout SE. In this harness a 4-column noise residual reads holdout z -2.70.\n4. Fold it into an already-scheduled 2-epoch pair at near-zero marginal cost, rather than spending 1.2 GPU-h.\n\nA separate proposal could target the part of T3a that is not just noise: warm-only rank of et1 and seq_cv_pop, val +0.00029, pmix +0.00062, holdout -0.00060. It must first explain why that differs from the closed xrel_more family (its 10-epoch pair cost -0.0019 val)."
  ]
 },
 {
  "name": "T1 STALE-TO second near-copy lens + regime-conflict flag (rule change)",
  "lens": "validation",
  "survives": false,
  "p_gain": 0.015,
  "cost": "About 5 s of GPU inference per member (80 members took about 7 min on GPU N); no training; CPU readout under 2 min.",
  "rule": "Pre-declared validation of the lens:\n- Before the next public read of any pool with train-only members, log the predicted public deltas 0.45*S, 0.71*H and 0.72*(H+S)/2.\n- After >= 3 new public points, keep S iff its cumulative LOO rmse is <= the holdout's; otherwise revert to H-only.",
  "verdicts": [
   "REFUTED: REFUTED for a statistical error that conflicts with recorded rules. The proposal's own retroactive check shows it changes no current decision. All numbers were re-run on CPU by /tmp/scratchpad/dpa/skeptic_t1_stale/k1_check.py and k1_check_v2.py (logs k1_check.log and k1_check_v2.log); the proposer's numbers reproduce exactly.\n\n(1) The 6 public contrasts carry no detectable signal. Their chi2 against pure public noise is 5.38 on 6 df (p about 0.5; public SE 0.00039-0.0006). VAL_PUBLIC_1113 \u00a74 already records that 'H0 all equal, shrinkage and holdout are statistically indistinguishable'. S was picked post hoc from at least 15 proxies. Monte Carlo with N=2000, public redrawn from N(0, SP):\n- The best of 15 proxies reaches S's dchi2 of 3.59 in 33.6% of draws, and Kendall >= 0.714 in 21.4%. The quoted Kendall p of 0.03 is the uncorrected single-proxy figure (6.1% in the simulation).\n- If the holdout is the true lens (truth = 0.71*H), S still beats H by the observed LOO margin of -0.00016 in 17.2% of draws.\n- The proposer's keep rule ('keep S iff its LOO rmse <= the holdout's') passes S 50.9% of the time when the holdout is true, which is a coin flip.\n\n(2) The gap between S and val_TO is not a staleness effect specific to each contrast. For 5 of 6 contrasts the difference-in-differences (S minus val_TO, vs G3) is one shared offset: -0.00037, -0.00040, -0.00036, -0.00034, -0.00034, each z of about -0.9. That offset is G3's own smaller stale drop: -0.00089 \u00b1 0.00053, against -0.00123 to -0.00128 for every other pool.\n- Adding the stale-robust families (POP1 family drop -0.00060, POPFREE -0.00004) makes the pool drop larger: Q2a and Q1a both drop -0.00123. That is the opposite of the stated mechanism.\n- Only M2 has a specific difference-in-differences (-0.00092, z -2.5), and that is a single point.\n\n(3) Two other readings contradict the premise.\n- The submitted form (0.5 FT + 0.5 stale TO) fits worse than predicting zero: LOO 0.00053 vs 0.00041 (dchi2 0.56).\n- The staler val half transfers worse than the fresher half: Thu dchi2 0.42 / LOO 0.00043, against Tue 2.63 / 0.00030. This matches VAL_PUBLIC_1113 line 20 ('staler Thu half is not a better public proxy', chi2 10.3 vs 7.3).\n\n(4) The rule-process reasons are recorded facts.\n- Choosing a judge by its fit to public outcomes is the HOLD-CLEAN failure: COVERAGE_MAP.md:160 'REFUTED (process) outcome-chosen judge'; TRICKS_SWEEP_1113.md:64 'Never pick the judge after seeing which arms it flips'.\n- CURRENT_STATUS.md:11 (frozen rule): 'public verifies LEVEL only, never the order of near-copies'. The 6 contrasts are near copies; Q1a and Q2a have corr 0.9995 to G3.\n- VAL_PUBLIC_1113 \u00a73 says the rule 'must not be re-fitted' on the same public points.\n- The pre-registered R5(a) holdout-break test already passes for the two new reads: |P-H| is 0.00059 (z 1.33) for Q1a and 0.00041 (z 0.89) for Q2a, both against a 2.5 SE threshold of about 0.0011. The proposed 'regime-conflict flag' duplicates R5(a) with val in place of public.\n- Stale-val already has its recorded role: COVERAGE_MAP.md:122 'ADOPTED (report only)'. CURRENT_STATUS.md:546 (hzcd) records that 'a private-only bet on stale-val alone would be a new rule; not taken'.\n\n(5) The proposal itself states 'G3 still stays; Q2a E2 = +0.000005 (tie)'. On this metric a selection-rule change cannot produce a val gain. | fix: Keep S only as a logged information column with no decision role, as the allowance in TRICKS_SWEEP_1113.md:64 permits. Before each public read, write 0.45*S, 0.71*H and the 0.5/0.5 combination to the prediction ledger.\n\nDo not change the pick rule, the decide9 tie-break or the regime-conflict flag. R5(a) already covers the flag, and it currently says the holdout lens is not broken.\n\nRevisit only under a pre-declared, adequately powered test. Example: at least 10 new public contrasts, or contrasts where |0.45*S - 0.71*H| >= 2 public SE (about 0.0008). Score them by cumulative chi2 against the holdout, with the multiplicity of the 15 proxies disclosed. A LOO-rmse comparison on 3 points cannot work: it picks S 51% of the time when the holdout is the true lens.\n\nIf the stale lens is ever to be used, it must cover the submitted form. That means FT fold models re-inferred under the same counterfactual inputs, since the TO-only S fits worse than predicting zero once the FT half is added (LOO 0.00053 vs 0.00041).",
   "REFUTED: T1 is refuted for three reasons: the statistics are wrong, the stated mechanism contradicts the proposer's own data, and the lens is validated by ordering near-copies by public, which the project's recorded findings rule out.\n\nRe-run on CPU: /tmp/scratchpad/dpa/skeptic_T1/k1_check.py, log at the same dir out/k1_check.log. It reproduces the proposer's base-TO pools exactly (assert passed), and S, LOO S 0.00026, LOO H 0.00042 and Kendall 0.714 all match.\n\n(1) STATISTICAL ERROR: the headline evidence is not significant once the correlated public noise is simulated. 4000 draws, public noise covariance SP built from the public-matched val weights as in the proposer's s5:\n- Public = pure noise (all pools equal):\n  - P(S beats H on LOO by >= the observed 0.00016) = 0.178\n  - P(LOO_S <= 0.00026) = 0.181\n  - P(Kendall_S >= 0.714) = 0.069, not the claimed p 0.03, which is a permutation p that ignores the shared-G3 noise\n  - P(max Kendall over the 5 lenses >= 0.714) = 0.174\n- Holdout lens true (k 0.71): P(S beats H by the observed margin) = 0.166.\n- val_TO lens true: P(S beats val_TO by the observed margin) = 0.32.\n\nThe proposer's own logs add three more problems:\n- Null: s5_stale_transfer.log has chi2(k=0) = 5.38 over the 6 contrasts, and the null rmse 0.00041 is below the mean public SE 0.00048.\n- Multiplicity: about 20 proxies were fitted (12 in s5, 8 in s8) and the best was reported.\n- Unfavourable variant left out: stale_both_final (FT half + stale TO, the form actually submitted) has LOO rmse 0.00053, worse than the null 0.00041, and gR2 0.10. It is not in the evidence.\n- At k=1, i.e. how E2 enters the rule, rmse(public \u2212 0.5H\u22120.5S) is 0.00043 vs 0.00031 for the holdout alone (s8_combo.log, last column).\n\n(2) MECHANISM CONTRADICTED. S \u2212 val_TO (the part staleness adds), paired over val feeds:\n\n| contrast | S \u2212 val_TO | z |\n|---|---|---|\n| Cd | \u22120.00037 \u00b1 0.00049 | \u22120.8 |\n| Pall | \u22120.00040 \u00b1 0.00046 | \u22120.9 |\n| PF7x | \u22120.00036 \u00b1 0.00042 | \u22120.8 |\n| Q1a (G3 + pop-free) | \u22120.00034 \u00b1 0.00038 | \u22120.9 |\n| Q2a (G3 + pop1) | \u22120.00034 \u00b1 0.00039 | \u22120.9 |\n| M2 | \u22120.00092 \u00b1 0.00037 | \u22122.5 |\n\nThe pop-light additions get the same stale penalty as the popularity-heavy pools. Stale drop of each pool: G3 \u22120.00089, G3+POPFREE \u22120.00123, G3+POP1 \u22120.00123, and every other pool \u22120.0011 to \u22120.0018. G3 is the least stale-hurt pool of the 12 tested, so S is val_TO plus one shared, roughly 1 SE, G3-specific offset plus M2. It is not \"pop-light families lose less when stale\", although popfree members alone drop +0.00004 (s5b_members.log). Since corr(val, S) = 0.93 (s8), S re-imports val under another name.\n\n(3) CONTRADICTS RECORDED FACTS:\n- CURRENT_STATUS.md line 12, frozen rule: \"public verifies LEVEL only, never the order of near-copies\".\n- CURRENT_STATUS.md line 11: the 7 pools lie within public sampling noise.\n- 14:12 log: \"ordering near copies by public is the winner's-curse trap\".\n- VAL_PUBLIC_1113.md line 72: \"differences < ~2 in LOO sum z^2 are not evidence\". S vs holdout is 1.42 vs 2.93, a gap of 1.5.\n- VAL_PUBLIC_1113.md line 88: the null \"all equal\" fits public as well.\n- VAL_PUBLIC_1113.md lines 20/129: the staler Thu half was already tested and is not a better public proxy.\n- CURRENT_STATUS 22:54: \"a private-only bet on stale-val alone would be a new rule; not taken\".\n\n(4) POST-HOC FLAG, WEAK VALIDATION RULE, NO GAIN:\n- The regime-conflict flag (2.5 SE) was derived from the two contrasts (Q1a/Q2a) that motivated it.\n- The \"keep S after >= 3 public points\" rule cannot tell the lenses apart. With 6 points, S already wins about 17% of the time when the holdout is the true lens, so 3 more points (public SE about 0.0004) give little information. This is a hypothesis; I did not simulate 3-point power.\n- As a rule change it adds no val by construction. Retroactively it changes nothing: G3 stays, and the Q2a E2 tie of +0.000005 is the proposer's own number. | fix: Keep stale-val in its recorded role: a reported column and a veto for cold/warm trades (S >= -1 SE), never a promoter or a tie-break. Do not calibrate any lens on the public order of near-copies. If a staleness lens is wanted, validate it on labelled data:\n- Option A: apply the same stale_pred 'both' counterfactual to the 30K holdout (cut the popularity inputs about 7 days before each holdout feed's day).\n- Option B: apply it inside val by user-grouped folds, and require that its contrast deltas track per-pool popularity exposure (pop-free/pop1 additions must lose less), which is the falsifiable prediction T1's own data currently fails.\nAny future public-based lens comparison needs >= 10 new contrasts and a pre-declared likelihood-ratio threshold, with simulated power reported, not 'rmse <=' after 3 points."
  ]
 },
 {
  "name": "FREEZE-ME (closure + re-entry trigger; no new run)",
  "lens": "stacking",
  "survives": false,
  "p_gain": 0.015,
  "cost": "0 now; about 1 CPU minute per re-entry.",
  "rule": "Adopt non-equal family weights only if ALL hold:\n- the holdout user-grouped OOF of holdout-fitted weights is > +2 SE vs ME;\n- the holdout-fitted weights applied to val are >= 0;\n- pmix >= -1 SE;\n- presub_gate PASS.\nOtherwise stay member-equal.",
  "verdicts": [
   "REFUTED: REFUTED AS A DUPLICATE, NOT AS A CONCLUSION. The freeze is right, but it is already in force and already recorded under the same mechanism on the same pool. The proposal brings no new action.\n\nMy independent re-check (CPU) is /tmp/scratchpad/dpa/skeptic_stackfreeze/verify_holdfit.py, with output in verify_holdfit.log in the same folder. It is built straight from preds/holdout/nn_<member>_holdout.npy and does not use the proposer's cache.\n- Member-equal (ME) holdout TO pool: 0.323793.\n- Best point of the 231-point simplex: a=(0.3, 0.4, 0.3), +0.00008 in-sample.\n- Val-fitted a=(0.05, 0.35, 0.6) applied to the holdout: -0.00048 +- 0.00017 (z -2.85).\n- GroupKFold out-of-fold (OOF), holdout-fitted, step .05: -0.00031 +- 0.00014 (z -2.27). Step .1: -0.00020 (z -1.86).\n- Robustness: 10 random user-grouped 5-fold splits give OOF mean -0.00020 (range -0.00033 to -0.00008), 0/10 positive. The proposer's z -2.27 sits at the pessimistic end of that range, but the sign holds.\n- The numbers reproduce, so the evidence is sound. It is a replication.\n\nThe same mechanism is already closed:\n1. Study A, 11-12 (nn_x/stack/hfit/plan.txt, pre-declared 15:36:55; logs/P3_W1-W2-W3-W4-W5.txt). Pool P3 = F10/F7x/F7x70 = G3. Holdout-fitted family weights, variants W1-W5 including direct NDCG coordinate ascent: 0/5 accepted, holdout 2-fold OOS -0.00003 to -0.00011, ME val 0.40032 / holdout 0.32379 (identical).\n2. CURRENT_STATUS 11-12 16:58 log entry:\n   - Study B: neural listwise stacker on G3, best +0.00056 with holdout -0.00006, a chance maximum of about 30 configs.\n   - Study C: aggregation functions (median, trimmed, softmax, rank, family-equal) CLOSED.\n   - Finding L1 of that entry reads, verbatim: \"The members, not the combiner, are the limit.\" That is the proposal's mechanism sentence.\n3. EXPERIMENT_REGISTRY lines 73-75 (cb26): holdout-trained GBDT residual -0.00042 (z -2.3); val-nested residual stacker -0.00009.\n4. Conditional weights:\n   - ROUND5.md line 118: \"Do not reopen w_ft or a w_ft per M segment\".\n   - sol_ensemble.md section 2.2, lines 110-133: M<50 / M>=50 w_ft 0.33 / 0.67 was replicated on 7 pools and was pool-specific.\n   - The proposer's M a1_wft (0.3 / 0.6, nested +0.00064) is the same scheme, and its own e3c 10-set replication gives +0.00004 +- 0.00009.\n5. TRICKS_SWEEP_1113 line 446: STUDY C V8 rawc / gscale / rankgauss within +-0.0002, holdout 0. The proposer's 18 pooling functions repeat these.\n6. COVERAGE_MAP rows already closing all of this:\n   - line 111: GBDT members, stackers, rank / RRF fusion, segment / routing weights;\n   - line 100: w_ft != 0.5;\n   - line 163: post-hoc combiners, including FT/TO mixing, GAIN-WFT and STALE-WFT;\n   - line 192: per-cluster weights and FT/TO mix;\n   - line 108: ME ADOPTED.\n7. CURRENT_STATUS section 6 already lists \"stackers, GBDT members, rank fusion, weights, M-split, w_ft\" as closed. Section 1 carries the frozen selection rule.\n8. Recorded public forward reads (CURRENT_STATUS lines 406-410 and 557; EXPERIMENT_REGISTRY line 31). Every val-favoured combiner on G3 already landed below G3 (0.392820):\n   - M2 (family weights 0 / 0.5 / 0.5): 0.392434;\n   - S2 stacker: 0.392473;\n   - TOonly (w_ft = 0): 0.392346.\n\nThe only new element is the re-entry trigger. It is procedural and is already covered: decide6, decide7 and decide9 test every new family at member-equal against G3. | fix: Do not add a new COVERAGE_MAP row, and do not create re-entry scripts. Instead, append one line of 11-13 replication evidence to the existing rows (COVERAGE_MAP line 111 and line 163):\n- 'G3 replication 11-13 (dpa/stacking):\n  - holdout-fitted family-weight OOF -0.00020 averaged over 11 user splits, 0/11 positive;\n  - val-fitted weights on the holdout -0.00048 (z -2.85);\n  - GBDT residual OOF -0.0027 to -0.0054;\n  - MLP null;\n  - 18 poolings null on the holdout;\n  - public reads vs G3: M2 -0.00039, S2 -0.00035, TOonly -0.00047.'\n\nKeep member-equal 0.5 FT + 0.5 TO under the existing frozen selection rule.\n\nRe-entry is already handled: decide6, decide7 and decide9 evaluate any new gated family at member-equal against G3. Only if such a family has test within-feed correlation < 0.99 should the holdout-fitted weight OOF be re-run as information.",
   "ok: The freeze stands, and I found no leak. I reran the checks on CPU from the raw preds, using src/metrics.ndcg_per_feed (scripts and logs in /tmp/scratchpad/dpa/skeptic_freeze_stats/).\n\n**Reproduced numbers (v1_verify.txt)**\n- Member-equal (ME) baseline: holdout 0.32379, val 0.40032.\n- Holdout-best family weights (0.3, 0.4, 0.3) gain only +0.00007 in-sample, which confirms the proposer's headroom figure.\n\n**Val-fitted family weights give nothing out of sample**\n- The fit never reaches significance on val. Nested over 30 random user-grouped 5-fold splits: mean +0.00021, median z +0.65, 0 of 30 splits at z > 2, mean pmix +0.00010. The TO-only form gives the same: +0.00021, median z +0.68 (v3_estimand.txt).\n- Every val-fitted point loses on the holdout:\n  - blend fit: -0.00034, z -2.25\n  - TO-only fit: -0.00015, z -0.81\n  - M>=50-only fit: -0.00029 to -0.00039\n- So the proposer's val-fitted holdout number does not depend on the FT-vs-TO estimand mismatch.\n- On val evidence alone, non-equal family weights would not be adopted.\n\n**What the proposal overstates (not grounds to refute)**\n1. **Holdout-fit OOF size.** z -2.27 comes from one GroupKFold split on the step-0.05 grid; the proposer's own step-0.1 run gave z -1.86. On my step-0.1 grid over 30 random splits: mean -0.00013, range -0.00031 to -0.00001, median z -1.13, 0 of 30 positive. The direction is robust; the size is about 2x overstated.\n2. **\"Every fitted combiner is val noise\" is false for one arm.** The M-conditional w_ft (0.3 for M<50, 0.6 for M>=50) is robust within val (v2_wft_mcond.txt): nested +0.00068, median z +3.14, 30 of 30 splits at z > 2. It still fails on other grounds:\n   - Against the unconditional w_ft fit it gives median z +1.53, short of 2. Placebo random buckets of the same size give +0.00027 (z 1.23).\n   - It does not hold inside any G3 family (frozen rule):\n     - F10: +0.00012\n     - F7x: +0.00000\n     - F7x70: -0.00012\n   - It does not hold on fresh families:\n     - popfree: +0.00022\n     - pop1: -0.00012\n     - the proposer's 10 sets: +0.00004 \u00b1 0.00009\n   - It matches the recorded closure (CURRENT_STATUS.md:424-425 fresh-confirmation fail; :435 the val-best M2/S2 below G3 on public).\n3. **The re-entry rule leans on an unrepresentative holdout.** The holdout is TO-only, 0% M<50 (test public 38.7%, private 39.2%), mean M 154 (test 100-104), inside the train period (CURRENT_STATUS.md:668), and is the checkpoint-selection set for every member (:16). Its family-weight surface correlates +0.734 with val M>=50 but only +0.209 with val M<50.\n   - 18 pooling functions: rawc reaches val +0.00023 (z 2.32), pmix +0.00032 \u00b1 0.00013, holdout -0.00000 (z -0.03). It misses PASS-WEAK's holdout z >= 0 only by 0.03 and is the chance maximum of 18 tests. It is a near copy (rho 0.99967), so expected public is about +0.00005.\n\n**Redundancy**\nThe closure already exists: COVERAGE_MAP.md:100 (w_ft != 0.5), :111 (GBDT members, stackers, rank fusion, segment/routing weights) and CURRENT_STATUS.md:42 (\u00a76). The proposal re-confirms these for G3; it adds no new mechanism. | fix: 1. Record this as an amendment to COVERAGE_MAP rows :100 and :111 ('re-confirmed on G3 11-13'), not as a new row.\n2. Report the holdout-fit OOF as 'never positive in 30 of 30 random user-grouped splits, typical -0.00013 (median z -1.13)', not z -2.27 from one split.\n3. Restate the M-conditional w_ft closure reason:\n   - It is robust on val: +0.00068, 30 of 30 splits at z > 2.\n   - It is closed for lack of out-of-sample support:\n     - vs the unconditional control, median z 1.53\n     - not present in any G3 family alone or in fresh families\n     - fresh confirmation failed (CURRENT_STATUS.md:424)\n     - public read below G3 (:435)\n   - It is not val noise.\n4. Re-entry rule: do not make the holdout the primary gate, because it is TO-only, has 0% M<50 feeds against about 39% on test, and is the checkpoint-selection set. Use this instead:\n   - val nested OOF with median z > 2 over at least 10 random user-grouped splits;\n   - the frozen weights applied to an independent seed set of the same families give a delta > 0;\n   - pmix >= 0;\n   - val M<50 slice not below -1 SE;\n   - holdout >= -1 SE as a veto;\n   - presub_gate PASS.\n5. Disclose rawc (val z 2.32, holdout z -0.03, near copy) as the chance maximum of the 18 pooling functions."
  ]
 },
 {
  "name": "P4 POP-TILT (post-hoc popularity demotion on G3, g - 0.02*z_feed(seq_cv_pop)): not proposed",
  "lens": "nonlinear",
  "survives": false,
  "p_gain": 0.015,
  "cost": "0",
  "rule": "Fails 'holdout z >= 0' by the fresh-vs-stale regime construction; not submittable under the frozen rule.",
  "verdicts": [
   "REFUTED: This is the closed \"cold offset\" lever in a different form. A score-level popularity demotion and a score-level cold boost are the same thing here: corr(z_feed(seq_cv_pop), cold) = -0.553 on val. COVERAGE_MAP.md:120 records \"cold offsets / gates / routers / MoE by coldness\" as CLOSED. SOLUTIONS_REVIEW.md:112 says \"Cold offset flat (+0.00013)\" and calls it \"a missing-information frontier, not a mis-weighted blend\". EXPERIMENT_REGISTRY.md:65 says staleness levers, cold offsets included, are \"All closed; each one trades cold feeds against warm ones\". TRICKS_SWEEP_1113.md:80 closes an inference-time \"hand-made popularity/cold offset\" (aux_rl) at +0.0003 \u00b1 0.0003. COVERAGE_MAP.md:144 closes hzcd, a popularity-parity training-time version, with the same pattern: val +0.0003, holdout -0.0005 (z -2.8), stale +0.0011.\n\nMy CPU re-run reproduces the proposer's numbers and shows the tilt carries the same cold/warm trade (script dpa/skeptic_poptilt/coldwarm.py, coldwarm.log):\n- POP-TILT at w = -0.02: all +0.00034 \u00b1 0.00026; all-cold-positive feeds (17.4%) +0.00601 \u00b1 0.00065; all-warm-positive feeds (70.9%) -0.00110 \u00b1 0.00032.\n- A plain binary cold offset (+0.01\u00b7z(cold)): +0.00036; all-cold +0.00580; all-warm -0.00116. The two signatures are nearly identical, so the tilt adds nothing beyond the closed lever.\n\nStatistical problems with the headline number (dpa/skeptic_poptilt/check.py, check.log):\n- +0.00034 is the in-sample argmax of a 13-point weight grid.\n- User-grouped nested 5-fold over 10 fold seeds averages +0.00023 (individual seeds range from -0.00012 to +0.00034), z about 0.9.\n- At fixed w = -0.02 the train-only half of G3 gains only +0.00002 \u00b1 0.00024, and that is the half comparable to the train-only holdout pool. The FT half gains +0.00020 \u00b1 0.00025.\n- The holdout falls as the tilt grows: w -0.01 gives +0.00002 (z 0.18), -0.02 gives -0.00024 (z -1.80), -0.03 gives -0.00050 (z -3.03), -0.05 gives -0.00089 (z -4.12).\n- pmix private is +0.00075 \u00b1 0.00043 (z 1.7). That is the expected cold-share reweighting of a cold/warm trade, not new evidence.\n\nThe proposer also marks it \"not proposed\", and the project's statistical hygiene rule requires holdout z >= 0, which this fails.\n\nScript paths are relative to /tmp/scratchpad/. | fix: None. Any score-level popularity or cold weight is a one-axis cold/warm trade that has already been closed. The re-entry route is new information (the pop1 and pop-free families), and COVERAGE_MAP.md:208 already records the pop1 family as CLOSED.",
   "REFUTED: I re-ran the numbers on CPU with /tmp/scratchpad/dpa/skeptic_p4/rerun.py (log: rerun.log in the same folder). The script reads the proposer's val.npz and hold.npz and scores g - w*z_feed(xt_seq_cv_pop) with src/metrics.py ndcg_per_feed. It reproduces the proposer's figures exactly. I am refuting because of four concrete problems.\n\n(1) The +0.00034 val figure is the best possible draw, not a typical one. It equals the in-sample peak of the w grid (w = -0.02: +0.00034 +- 0.00026, z +1.29, the best of 13 grid points). The fold seed the proposer reported (seed 0) chose -0.02 in all 5 folds. Across 20 user-grouped fold seeds the nested mean is +0.00019, ranging from -0.00013 to +0.00034, so seed 0 is the maximum. That gives z about 0.7. The curve is also jagged (-0.03: +0.00021, -0.02: +0.00034, -0.01: +0.00019). The pmix figures (private +0.00075 +- 0.00043, public +0.00057 +- 0.00034) were computed in-sample at the selected w and were never nested.\n\n(2) The proposer explains the holdout loss as a 'fresh-vs-stale regime' effect. The val data contradicts this. The holdout scores the train-only pool, and the val counterpart of that pool is gto, which is in the same stale regime as val and test. On val, gto at w = -0.02 gives +0.00002 in-sample, and its nested result is negative in all 20 of 20 seeds (mean -0.00033, best -0.00007). The FT half gives a nested mean of -0.00004. So the tilt does nothing for the train-only pool in the stale regime either. The holdout result agrees with val rather than being a regime artefact, and it gets worse as the dose increases:\n\n| w | holdout delta | z |\n|---|---|---|\n| -0.01 | +0.00002 | +0.18 |\n| -0.02 | -0.00024 | -1.80 |\n| -0.03 | -0.00050 | -3.03 |\n| -0.05 | -0.00089 | -4.12 |\n\nThe rule 'holdout z >= 0' fails.\n\n(3) Closed rows with the same mechanism, all in reports/COVERAGE_MAP.md:\n- Line 144: 'hzcd popularity-parity cooldown, CLOSED 11-12: val +0.0003, holdout -0.0005 (z -2.8), stale-val +0.0011 (cold/warm trade)'. This is the same signature: popularity demotion gains on val and loses on the in-period holdout.\n- Line 163: post-hoc combiners are CLOSED, 'bounded by infinite-member headroom +0.0002 val / ~0 hold'.\n- Line 150: seq_cv_pop as an extra relative input is CLOSED (val -0.0019, holdout -0.0007).\n- Line 208: the principled popularity-light route (the pop1 member family) is CLOSED as a pick; Q2a is kept only as an alternate.\n\n(4) Expected public impact. The tilt is a near copy of G3 (proposer's figure: corr ~0.999). For near copies, val deltas have transferred to public at k ~0.2 (CURRENT_STATUS line 664). Applied to the honest nested +0.00019, that predicts about +0.00004 on public. The paired public SE is about 0.0004 (line 668), so public could not confirm a gain of this size.\n\nNo leakage was needed to reach this verdict: the result fails on statistics alone. | fix: None. Keep it closed. Any future post-hoc tilt on G3 should be reported as the mean over at least 20 user-grouped fold seeds, with pmix nested and not in-sample at the chosen w. It must also show a gain on the val train-only half (gto) before anyone uses a 'fresh vs stale regime' argument to explain a holdout failure. The popularity-light hedge already exists as Q2a (pop1 family)."
  ]
 },
 {
  "name": "T5 True temporal holdout (retrain on MSK days <= 20710, evaluate on days 20713-20715 exploration feeds)",
  "lens": "validation",
  "survives": false,
  "p_gain": 0.0125,
  "cost": "1-2 agent-days of features plus about 13 GPU-h for G3 and 20-40 GPU-h per comparison set.",
  "rule": "Build only if a GENERATIONAL candidate (val >= +0.002, corr < 0.99) appears whose holdout and stale reads disagree in sign. Not recommended before the Dec-7 freeze.",
  "verdicts": [
   "ok: I found no grounds to refute T5. The registry has no closed row with the same mechanism (retrain on truncated train days, then score later labelled days with inputs built only from earlier data). The nearest rows work differently:\n- COVERAGE_MAP:121 reweighted holdout, CLOSED (invalid, ESS 2,880): reweighting, no retraining.\n- COVERAGE_MAP:122 stale-val replay, ADOPTED as report-only: makes the inputs stale, but the model weights are still trained on the full period.\n- COVERAGE_MAP:160 HOLD-CLEAN, refuted on process: date slices of the existing in-period holdout.\n- COVERAGE_MAP:186 holdout format check, INFO only.\n- FEATURES_REVIEW FR-3/FR-4: causal or lagged inputs as a model change, not as an evaluation protocol.\n\nThe proposer's numbers reproduce. Script: /tmp/scratchpad/dpa/skeptic_T5/k1_counts.py, output in k1_counts.log.\n- Eval sample (random-like feeds, M>=50, days 20713-20715): 24,992 = 8,279 holdout + 16,713 others.\n- Training on days <=20710 keeps 75.9% of feeds and 76.0% of candidates. The public-like cut at <=20708 keeps 59.8%.\n- Per-feed sd of the disputed contrasts (Q2a, Q1a, M2, PF7x vs G3) on the holdout's last 3 days is 0.029-0.040. That gives an MDE (2.8 SE) at n=24,991 of 0.00051-0.00070. For family-different pools (Cd vs G3) it is 0.00085. So \"MDE about 0.0007\" is correct for near-copies.\n- The cost is plausible. CURRENT_STATUS 10:51 records 150k steps in 86 min on one GPU, so 13 roots at 76% data come to about 9 GPU-h. Adding 14 re-cools at 10-19 min each (wave 8, 12:20) gives about 13 GPU-h.\n\nThe value is low, though:\n1. The 3-5 day gap is the same gap val already has, and val is already out-of-period for the train-only (TO) weights. T5 adds sample size (about 25K vs about 8.5K val M>=50 feeds) and one weekend day. It does not reach the public gap (7-8 days) or the private gap (8-10 days).\n2. Short feeds stay invisible. p_random cannot be used for M<50 (eda_labels_position.md:209-211). Yet 69,835 M<50 train feeds on days 20713-20715 have p>0.9, and M<50 is 39% of private.\n3. The population gap stays. sol_advval \u00a70.5 finds random-like train feeds differ from exploration feeds by population, not by time (intrinsic AUC 0.756).\n4. The existing holdout's later days already agree with its earlier days on the disputed contrasts. Script k2_byday.py: Q2a\u2212G3 is \u22120.00002\u00b10.00025 on days <=20710, \u22120.00011\u00b10.00065 on 20713, +0.00005\u00b10.00048 on 20714-15. Q1a\u2212G3 is \u22120.00055, \u22120.00054, \u22120.00040.\n5. It is a measuring tool, not a model change. It can only help through a pick decision, and it is gated on a trigger that has not fired in this round: a +0.002 generational candidate whose holdout and stale reads disagree in sign. Every recent arm is at or below +0.0006.\n\nSpec gap: production selects checkpoints on the 30K holdout (CURRENT_STATUS:16), which contains all 8,279 of these eval feeds. The truncated models must select on the 21,721 holdout feeds dated <=20710.\n\np_gain 0.015: about 0.1 that the trigger fires before the freeze, times about 0.15 that T5 then changes a pick in a way worth >= +0.0005 without losing on holdout or pmix. | fix: Keep T5 dormant, gated on its trigger. If it is ever built:\n(a) Select checkpoints and cooldowns only on the 21,721 holdout feeds dated <=20710. Remove every day >=20713 feed from training, the m02 exploration mix and checkpoint selection.\n(b) Retrain the pre2 encoder on histories up to day 20710 instead of accepting the leak. The pre2 channel is worth +0.0013 val / +0.0005 holdout (COVERAGE_MAP:190).\n(c) Pre-declare a calibration step before T5 can act as a judge. Train truncated copies of at least 3 of the publicly scored contrasts (Q2a, Q1a, M2 vs G3) and require the T5 deltas to match the public sign at least as well as stale_TO (Kendall 0.714 in the validation lens's s9_rank.log). Otherwise T5 stays an unvalidated lens, the same status the holdout has (RECAP_1113:46).\n(d) Report T5 next to val-TO M>=50, never alone, because the two have the same 3-5 day gap.",
   "REFUTED: The proposer's counts check out. As written, though, T5 cannot change the decision it would be built for, it measures the wrong regime for its own trigger, and no measured holdout-vs-public gap exists for it to fix. All scripts are CPU only and live in /tmp/scratchpad/dpa/skeptic_T5/.\n\n(0) Claims that hold:\n- k1_days.log: the train days are 20702-20710 and 20713-20715. Days 20711 and 20716 hold only 42 and 34 feeds.\n- The T5 set has 24,992 feeds: 8,279 holdout plus 16,713 pool134, with no overlap, from 23,543 distinct users. Only 177 of these feeds belong to users who also have a val/test feed.\n- MDE (k4_mde.log) at 2.8 SE is 0.00066 for near-copies (sd 0.037). For the generational use the rule targets it is 0.00084 to 0.00097 (sd 0.0473 to 0.055), before any seed noise. 'About 0.0007' is therefore optimistic.\n\n(1) Process conflict with a recorded rule. The build rule is triggered by outcomes: build T5 only when a specific candidate's holdout and stale reads disagree in sign. reports/TRICKS_SWEEP_1113.md:64 says: \"A holdout split may be adopted only on label-free evidence, before any arm delta is computed, and then only as an information column for future arms.\" COVERAGE_MAP.md:160 records HOLD-CLEAN as REFUTED (process) for the same reason (an outcome-chosen judge). The frozen pick-1 rule (CURRENT_STATUS.md:10) still requires the holdout to beat G3 by >= 2 SE. So for the candidate that triggers T5, its result could only be an information column. Using it to break ties between disagreeing judges also loosens the frozen bar, which raises the false-adopt rate.\n\n(2) T5 at the specified cutoff (<= 20710) is the val regime, not the stale/test regime its trigger asks about. From k3_t5_staleness.log, as cold share / mean staleness:\n- T5 at 20710: 0.382 / 5.09 days\n- val: 0.348 / 5.69\n- public: 0.438 / 9.34\n- private: 0.477 / 11.23\n\nThe trigger candidate has already passed val at >= +0.002, and the val M>=50 slice (n 8,530) already gives an out-of-period, val-staleness, M>=50 read. T5 only triples that sample.\n\n'Public-like at <= 20708' matches coldness only: at 20708 cold is 0.466 but staleness is 6.99. Matching staleness needs cutoff 20705 (9.50 days), where cold rises to 0.578, well above test. At cutoff 20706, 45.5% of train feeds remain, against 75.9% at 20710.\n\n(3) No measured holdout-vs-public disagreement exists to repair. Across the 6 recorded contrasts (k2_proxy_vs_public.log, built on the s7/s8 numbers):\n- The holdout already matches public at scale k=1: rmse 0.00031, against a paired public SE of about 0.00045. s8 gives chi2 2.83 on 6 contrasts.\n- Public is also consistent with zero (chi2k0 5.38).\n- Given the holdout's small spread, even a perfect holdout proxy would show a median correlation with public of only 0.515. The observed 0.626 is therefore not evidence of bias.\n- Val M>=50 correlates 0.79 with public but overshoots it about 3x (k 0.36).\n\n(4) Spec errors, which reduce cost but do not change the verdict:\n- The 'pre2 retrain or accepted leak' choice is a false dilemma. Windowed leak-free pre2 corpora already exist: W1/W2 end at Sep 17 00:00 MSK, and W3 covers all 758,764 feeds from Sep 17 on, built like val/test (pretrain/build_corpus_window.py header; data/pretrain/win*/info.json). An 'accepted leak' would in any case break CURRENT_STATUS \u00a73.\n- The 8,279 holdout-part feeds are currently used for checkpoint selection. They would have to be removed from selection in the truncated runs.\n- The count features must be rebuilt with val parity (transductive within the evaluation period, LUO), not strictly as-of, or T5 is stricter than val/test.\n\np_gain: T5 is a measurement, not a lever. It would need a generational candidate to appear before Dec 7 (this is uncommon; CURRENT_STATUS \u00a76 lists every remaining bet as low probability), plus a sign disagreement, plus a rule change made in advance. In half of the trigger cases the holdout read is negative, which fails 'holdout not below' by definition. My estimate is about 0.01. | fix: If T5 is wanted at all, pre-declare it NOW, before any candidate's deltas exist, as a label-free information column for all future arms, not as a tie-breaker that disagreement triggers. Changes:\n- Choose the cutoff on label-free regime matching. Cutoff 20708 gives cold 0.466, close to test's 0.438-0.477, and keeps 59.8% of train feeds; report that staleness stays at 6.99 days vs 9.3-11.2.\n- Reuse the existing windowed pre2 features: W1/W2 for train rows <= 20710, W3 for evaluation rows. No encoder retrain and no 'accepted leak'.\n- Do checkpoint selection only on holdout feeds dated <= the cutoff, never on the 8,279 evaluation-day holdout feeds.\n- Rebuild the count inputs with val/test parity (LUO, transductive within the evaluation period, nothing from days after the evaluation day for training rows).\n- Use the honest MDE: about 0.0009 for family-level contrasts, plus seed noise.\n\nUntil then the cheaper equivalent read already exists: the candidate's val-TO M>=50 slice (n 8,530, near-copy SE about 0.0004) plus stale-val."
  ]
 },
 {
  "name": "T4 Stale-holdout counterfactual (as-of popularity inputs for the 30K holdout feeds; inference only)",
  "lens": "validation",
  "survives": false,
  "p_gain": 0.01,
  "cost": "CPU feature rebuild of about 4-6 agent-hours (hypothesis); GPU about 5 s per member.",
  "rule": "Adopt as S's sample only if, on the 6 H-defined contrasts, Kendall(stale-holdout, stale-val) >= 0.5 and its paired SE is <= 0.7x S's. Otherwise close.",
  "verdicts": [
   "REFUTED: T4 is refuted. Its own spec contradicts the recorded data, and the SE gain it claims exists only in the part of the sample that is not public-like.\n\n1. The d-7 cutoff wipes out popularity for most of the holdout. Train feeds run from day 20702 to 20715 (measured; day 20712 is empty). Under 'history snapshots of feeds with day < d-7', every holdout feed on days 20702-20709 gets no history at all: 19,075 of 30,000 feeds (63.6%) have as-of luo0 = 1.000 (asof_cover.py / asof_cover.log in /tmp/scratchpad/dpa/skeptic_T4_validation/).\n   - Feed-mean as-of luo0 is 0.845. Under the same train-snapshot-only definition, public is 0.470, private 0.496 and the stale-val cut (<20713) 0.507.\n   - The live holdout luo0 is 0.220, which reproduces s1_dist's 0.2202.\n   - Public luo_zero has mean 0.429 and sd 0.119 (dpa/validation/out/s1_dist.log), so an all-cold feed is about 5 sd off public. Those 19K feeds sit in the popularity-zeroed (xdrop_pop / G1) regime, not public staleness.\n   - Only 10,925 feeds keep any history, and they get 1-6 days of it, against 14 for public. That is fewer feeds than val's 14,423, so the 'second, 2x larger sample as stale as public's' does not exist.\n   - Other lags do not rescue it. L=1 gives luo0 0.544 with 17% zero-history feeds, and L=5 gives 0.766. A 14-day train period cannot match both public's gap (7-8 days) and its window (14 days).\n\n2. The SE clause passes only because of the zero-history feeds (se_gate.py; hypothesis: per-feed delta sd equals the live holdout's, from s4_power.log).\n   - All 30K feeds: SE ratio to S is 0.52-0.62, which passes.\n   - The 10,925 feeds with history: 0.86-1.03, which fails the 0.7x gate.\n   - Days 20713-15 only: 0.99-1.19, which also fails.\n\n3. The Kendall clause cannot tell lenses apart (kendall_gate.py, using the per-contrast numbers in s5_stale_transfer.log).\n   - Against S, the un-staled val lens already scores tau +0.600 and passes, as does hold_weekend (+0.600). The live holdout reaches +0.467, one swap away.\n   - With n=6, the null pass rate is 0.068.\n   - So a pass would not show that the stale holdout measures what S measures.\n\n4. Process. S was itself picked post hoc as the proxy (about 12 tried) that best ordered the 6 public contrasts (s9_rank.log: tau 0.714, p 0.030 uncorrected). Adopting a new judge sample because it agrees with S on those same contrasts breaks the recorded rule 'never pick the judge after seeing which arms it flips; a holdout split may be adopted only on label-free evidence ... and then only as an information column' (TRICKS_SWEEP_1113.md l.64). It also conflicts with VAL_PUBLIC_1113 \u00a73, a pre-registered rule that is not to be re-fitted.\n\n5. Staling fixes neither of the holdout's other measured gaps (s1_dist PSI vs public):\n   - M>=50-only composition: PSI 2.38.\n   - n_train_feeds_all: PSI 2.43.\n\nThe proposer's quoted numbers (PSI 3.2 / 8.3; S SE 0.00034-0.00052) are cited correctly; what is wrong is the construction. | fix: If anything survives, it is a label-free information column, never S's sample. Rebuild one-sided as-of counts at L=1 (snapshots of feeds with day < d-1) on the 24,836 holdout feeds that keep at least one history day. Their luo0 is 0.449, against 0.470 for public under the same definition. This removes the two-sided LUO inflation (AUDIT_1112 LEAKAGE-1) but does not emulate staleness. Pre-declare it before any contrast delta is computed and report it as a dE/dL-style column only. Otherwise keep S at its current SE and narrow it by pooling more seeds or members on val, rather than by borrowing a non-comparable sample.",
   "REFUTED: REFUTED, on four measured or cited reasons. Scripts: /tmp/scratchpad/dpa/skeptic_T4_validity/k1_asof_coverage.py (.log) and k2_rule_power.py (.log), both CPU-only.\n\n(1) The spec cannot produce \"public-like\" staleness. The train data begins on day 20702 (data/flat/train/feeds.parquet; days 20711-20712 are nearly empty). The history arrays have no event-timestamp column: hist_* = category, emb_idx, event_source, event_type, fidx, hpos, item_id. So \"snapshots of feeds with day < d-7\" is the only as-of definition available. Under it, 19,075 of the 30,000 holdout feeds (0.6358, days 20702-20709) have zero eligible train feeds, and every popularity input there becomes zero. Measured as-of values:\n- luo0 is 1.0000 on each of days 20702-20709, then 0.695 on 20710, 0.553 on 20713, 0.533 on 20714 and 0.522 on 20715.\n- Holdout-wide feed-mean as-of luo0 is 0.845, and as-of cold_tte is 0.846. Public is 0.429 / 0.439 and val is 0.381 / 0.348.\n- luo0 PSI vs public: current holdout 3.239, as-of holdout 3.190 (no better, now overshooting in the cold direction), and 1.561 even on the 10,925 feeds that have any as-of history.\nSo the mechanism \"inputs as stale as public's\" is contradicted. The result is a pop-free regime on 64% of feeds plus 1-6 days of history on the rest, against public's 14+ days. Note that LUO removes nothing at an as-of cutoff, so the k1 coverage figures are exact for it_hist_all and for train-candidate tte/xp.\n\n(2) The \"2x larger sample shrinks S's SE\" claim fails on the proposal's own clause. With the current fresh inputs, full-30K paired holdout SEs are 0.52-0.62x S's (Cd 0.00027, Pall 0.00025, PF7x 0.00022, M2 0.00019, Q1a 0.00021, Q2a 0.00021). On the 10,925 feeds that actually receive as-of history, the ratios are 0.87, 0.89, 0.88, 0.87, 1.02 and 0.98, so all six fail the <= 0.7x clause. Any shrinkage would therefore come from the 19K degenerate all-cold feeds. These are fresh-input SEs; the as-of SEs are not measured, but the subset size alone caps the gain.\n\n(3) The adoption rule does not discriminate. Kendall >= 0.5 against stale-val on the 6 contrasts is already met by plain val (+0.600) and by val_TO (+0.733). The fresh holdout gets +0.467, and its days>=20710 subset gets -0.200. The exact null P(tau >= 0.5 | n=6) is 0.068. The shared main effect (Cd and Pall below G3) drives the concordance, so the rule would \"adopt\" almost any proxy.\n\n(4) The premise is fitting noise, and it contradicts recorded rules.\n- On the six public contrasts, chi2 at k=0 is 5.38 on 6 df (p = 0.496). Null rmse is 0.00041 against a mean public SE of 0.00048 (proposer's s5_stale_transfer.log).\n- S was selected as the best of at least 7 proxies (s9_rank.log, tau 0.714, p 0.03 before multiplicity), which is a winner's curse.\n- reports/CURRENT_STATUS.md line 10 says modern pools are within public noise. Line 11 (frozen selection rule) says \"public verifies LEVEL only, never the order of near-copies\".\n\nAlso, stale inputs cannot make the holdout out-of-period:\n- Members train on every other train feed from all 14 days; the holdout is excluded by feed only (nn_x/w5/train.py lines 282-287).\n- Every family's checkpoint is selected on these same 30K feeds (select_on 'holdout' in nn/logs/v13-v26 and abl_e10_pop1 logs).\n- So the weights still carry in-period knowledge, and the sample is selection-contaminated.\n\nThe task is a measurement tool with no direct metric effect. At best it would re-order near-copy pools whose public gaps are about 1 SE. | fix: No version of this can do what it claims. The only as-of variant matched to public coverage would use the holdout feeds on days 20713-20715 (8,279 feeds) with a global cutoff near day 20709-20710. That sample is smaller than val's 14,423, so it cannot shrink S's SE, and the in-period weights and holdout-based checkpoint selection would still apply. Keep the existing stale-val as the stale-regime proxy, and keep using public for level checks only, per the frozen rule."
  ]
 }
]

## Appendix: validation recommendations per lens

The input pipeline is clean: exact CPU parity, fixed train-fitted scaling, val->test PSI <=0.042 on 102 of 104 inputs, and no residual structure over G3 in the NN's own inputs. So there is no input-side reason to distrust val as the population. Keep val, with pmix as the private proxy.

The in-period holdout is the lens with an input-regime problem:
- 26% of its feeds (Sep 6-8) carry an input state that never occurs at inference (pre2 blank, has:pre2_fe=0).
- Its LUO popularity is two-sided, and the effect shrinks with the row's day: the pop-free gap is +0.0106 on Sep 6-8, +0.0072 on Sep 9-19, +0.0034 on Sep 19, against val +0.0025 and pmix +0.0005. Matching cold share does not remove this.
- It has 2.2x val's H=0 share and only M>=50 feeds.

Recommendation, without changing the frozen rule:
- Add a fixed label-free late-holdout column (Sep 17-19, 8,279 feeds) and an early column (Sep 6-8) to every readout.
- Treat the full holdout as biased against popularity-light or popularity-representation changes (Q2a/pop1-type contrasts, P2): for those, use it as a veto only and lean on val + pmix + late-holdout.

A caveat on the nested-residual probes: positive controls show they recover only about 15% of a known 0.006-0.012 input gap, and val-fitted residuals always transfer negatively to the holdout. So "null on G3" is weak evidence about NN training-time representation changes; only cheap 2-epoch arms (P1, P2) can price those.

The REPORT.md write was blocked by the harness (subagents must return findings as text). The full content is in these fields, and the numbers are in the scratch out/ logs: s02.log, s02b.log, s03*.log/txt, s04.log, s04b.log, s05-s07.log, stats_X.csv, table_inputs.txt.

Validation: the holdout acted as an effective falsifier in this lens. Val-selected tilts lost on it in 82/94 cases, and every val-trained GBDT/MLP residual lost there by 0.0001-0.0015. The exception is the popularity regime: the in-period holdout is the fresh-popularity population, so it votes against stale-regime corrections by construction, and only public/private can judge those. Add the RESID-HALVES rule (P2) to the nested-residual protocol. The in-period holdout contains only M>=50 feeds and has no stale items, so a 'cold' interaction can be read there only through the LUO==0 proxy. Process note: the harness refused to write REPORT.md (subagent report-file block). report_path therefore points to a plain concatenation of the raw logs that every number cites (ALL_LOGS.txt); the narrative is in these structured fields. Scripts and per-run logs are in /tmp/scratchpad/dpa/nonlinear/. CPU only, no GPU used, no project file modified.

There is no REPORT.md: the harness blocks subagents from writing report files. The full content is in the findings and proposals fields, and every number is in the logs under /tmp/scratchpad/dpa/validation/out/ (s1_dist, s2_transfer, s4_power, s5_stale_transfer, s5b_members, s6_levels, s7_m50, s8_combo, s9_rank, s10_predict).

VAL: keep it, unweighted. It matches public on every intrinsic covariate: PSI <= 0.01 for M, H, users, categories, f1/f4 and entropy, the same as private-vs-public. Its only gap is train-derived staleness, which no reweighting can reach:
- the exact public-matched weights I built (EB_pub, out/w_val_pubmatch.npz) give transfer k 0.10, worse than uniform val;
- pmix overshoots the deltas about 5x (k 0.19) and the level by +0.0045, so it should not be used as a promoter or a tie-break;
- calibM stays the level predictor (offset +0.0005).

HOLDOUT: keep it, but downgrade it from the only near-copy ranking lens. It is not representative (M>=50 only, popularity coverage PSI 3.2, staleness 1.75 d vs 9.3 d). It is significantly regime-biased against popularity-light candidates: on M>=50, the train-only val delta minus the holdout delta is z +3.5 for Q1a and z +2.8 for Q2a. Public sided against the holdout on both (+0.00059 / +0.00041).

ADD a stale-val train-only lens S: a GPU inference probe of about 5 s per member, already run for all 80 members of the scored and unscored pools. It orders the 7 scored pools most like public (Kendall +0.71 vs +0.33 for both holdout and val; LOO rmse 0.00026 vs 0.00042 / 0.00034), though it needs scaling (k 0.45).

New near-copy rule:
- E2 = 0.5H + 0.5S. Switch only if E2 > 2 SE_E2 AND H >= -1 SE AND S >= 0, plus the existing val, slice and presub vetoes.
- Regime-conflict flag: if |val_TO(M>=50) - H| > 2.5 SE, the holdout is not decisive and S is primary.
- Replace the pmix tie-break with S.
- Applied retroactively, nothing changes: G3 stays pick 1 and Q2a stays the tie-level alternate (E2 +0.000005).

Do not fold in-period feeds into val (leak and regime mix). Do not build a true temporal holdout before Dec 7 (20-40 GPU-h per comparison set plus 1-2 days of features, near-copy value only).

Statistical caveat: about 12 lenses were fitted to the same 6-8 public points with public SE 0.00048, so no lens is statistically separable. The stale lens is a better-supported tie-breaker, not a validated predictor.

- Keep G3's member-equal z-mean (0.5 FT + 0.5 TO).
- Nothing on the second level survives both reads at 2 SE: 0 of 78 tests have holdout z >= +2. The 8 arms with val z >= 2 are either one M<50 / w_ft mechanism or a near-copy pooling function (rawc), and they fail the holdout, have no holdout read, or fail fresh-member replication (+0.00004 +- 0.00009).
- Validation lesson for this lens: the train-period holdout is blind to w_ft and to M<50 feeds, and that is exactly where every val-only combiner gain sits. Such a claim needs a fresh-member replication like D/e3c_fixed_rule.py, not another val read. An M<50 holdout-2 is needed only if trunks are retrained anyway.
- Val-fitted family weights anti-transfer to the holdout (z -2.85). Holdout-fitted weights do not beat member-equal out of fold even inside the holdout.
- So no combiner or stacker submission. Spend effort only on members with test within-feed rho < 0.99 to G3.