# CCB-1113 RESULT: 16-item CPU closure batch (EXTERNAL_SCAN_1113 §2.3)

Written 2026-11-13 19:09 UTC by the synthesiser of workflow a workflow run. Rule: `nn_x/insights/ccb_1113/PREDECLARED_ccb.txt` (stamped Fri Nov 13 18:31:28 UTC 2026, before any number). CPU only, read-only on the project; this file is the batch's only project write.

**Verdict: 0 of 16 PASS. All 16 CLOSED.**
- No item reaches criterion (1), nested val >= +0.0005, even before multiplicity correction. The best is C07 at +0.00031 ± 0.00038.
- Holm-adjusted p is 1.0000 for every item. The smallest raw p is 0.2109 (C07), against a first Holm threshold of 0.02275/16 = 0.00142.
- Nothing from this batch is pooled, fine-tuned or submitted. G3 stays pick 1, and the frozen pick-1 rule is untouched.

Where the numbers come from:
- Synthesis script: `/tmp/scratchpad/ccb/synth/synth_holm.py`. Its output is `.../ccb/synth/synth_holm.log` and `.../ccb/synth/synth_holm.json`.
- The script re-reads every item's numbers from the testers' own result files, not from the workflow JSON, and the two agree:
  - `tester1/out/results.jsonl`
  - `tester2/logs/results.jsonl`
  - `tester3/out/C{03,06,09,12}.json`
  - `tester3/c15/c15_gradcos.json`
- It then computes z = Δ/SE, one-sided p = 1 − Φ(z), the STOP overrides, the Holm step-down at α = 0.02275, and criteria (1)-(5).
- Per-item lines are in each tester's log, cited in the table notes.
- All scratch paths are under `/tmp/scratchpad/ccb/`. That folder is session-local: copy it to `nn_x/insights/ccb_1113/scratch/` to keep it.

Every tester harness passed the base sanity check: val 0.40032 (14,423 feeds, 1,389,971 rows) and holdout 0.32379 (30,000 feeds, 4,630,966 rows), for example `tester1/logs/c01_eval.log` line 1, `tester2/logs/c02.log` line 2 and `tester3/logs/c03_test.log` line 1.

---

## 1. Table

Column notes:
- **Δ**: nested val delta, PRIMARY LIN minus control, paired per feed over 14,423 feeds (B5).
- **holdout**: theta_full applied to the 30K train-only G3 holdout, minus b_h (B6).
- **late / early**: the same delta on Sep 17-19 (8,279 feeds) and on Sep 6-8 (7,823 feeds).
- **pmix**: the private-weighted mean of the nested per-feed deltas. "veto" means pmix < −1 SE.
- **fails**: the criteria of PREDECLARED §D that the item misses: (1) val >= +0.0005; (2) Holm; (3) holdout >= 0; (4) late >= 0; (5) pmix >= −1 SE.
- **cols**: number of LIN parameters, β included.

| id | name | built | cols | val Δ ± SE | z | raw p1 | Holm p | holdout (z) | late | early | pmix ± SE | verdict (fails) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | GBDT-OOF leaf-SVD64 | yes | 129 | −0.00264 ± 0.00064 | −4.11 | 1.0000 | 1.0000 | −0.00136 (−3.82) | −0.00040 | −0.00254 | −0.00302 ± 0.00102 veto | CLOSED (1,2,3,4,5) |
| C02 | Streak / EWMA + per-type same-cat recency | yes | 45 | −0.00084 ± 0.00041 | −2.06 | 0.9801 | 1.0000 | +0.00002 (+0.11) | +0.00036 | −0.00008 | −0.00061 ± 0.00060 veto | CLOSED (1,2,5) |
| C03 | Latent geometry (loc16, modes, W-geometry) | yes | 18 | −0.00008 ± 0.00030 | −0.28 | 0.6102 | 1.0000 | −0.00029 (−2.08) | −0.00054 | −0.00007 | +0.00044 ± 0.00046 | CLOSED (1,2,3,4) |
| C04 | Co-vis variants + NB log-lift | yes | 11 | −0.00020 ± 0.00034 | −0.59 | 0.7221 | 1.0000 | −0.00030 (−1.70) | −0.00011 | −0.00038 | +0.00033 ± 0.00053 | CLOSED (1,2,3,4) |
| C05 | IDF / IUF / Swing / BM25 co-vis | yes | 11 | −0.00034 ± 0.00031 | −1.09 | 0.8631 | 1.0000 | −0.00013 (−0.72) | −0.00030 | −0.00034 | −0.00028 ± 0.00044 | CLOSED (1,2,3,4) |
| C06 | Bradley-Terry item strength | yes | 5 | +0.00028 ± 0.00037 | +0.78 | 0.2183 | 1.0000 | −0.00015 (−0.78) | −0.00031 | +0.00001 | +0.00059 ± 0.00055 | CLOSED (1,2,3,4) |
| C07 | Item trend / lifecycle (gap-matched) | yes | 14 | +0.00031 ± 0.00038 | +0.80 | 0.2109 | 1.0000 | −0.00024 (−1.52) | −0.00047 | −0.00005 | +0.00031 ± 0.00059 | CLOSED (1,2,3,4) |
| C08 | EMDE multi-partition sketch | yes | 19 | −0.00063 ± 0.00036 | −1.78 | 0.9625 | 1.0000 | −0.00055 (−2.96) | −0.00070 | −0.00056 | −0.00084 ± 0.00055 veto | CLOSED (1,2,3,4,5) |
| C09 | RP3beta whole-sequence walk | yes | 5 | +0.00012 ± 0.00036 | +0.32 | 0.3736 | 1.0000 | −0.00013 (−0.61) | −0.00008 | +0.00056 | −0.00056 ± 0.00055 veto | CLOSED (1,2,3,4,5) |
| C10 | User aggregates × cues (F26 + F27) | yes | 171 | −0.00155 ± 0.00058 | −2.66 | 0.9961 | 1.0000 | −0.00141 (−4.45) | −0.00137 | −0.00120 | −0.00342 ± 0.00096 veto | CLOSED (1,2,3,4,5) |
| C11 | Typed implicit rating cvx | yes | 11 | +0.00006 ± 0.00033 | +0.18 | 0.4289 | 1.0000 | −0.00018 (−1.07) | −0.00009 | −0.00032 | −0.00071 ± 0.00048 veto | CLOSED (1,2,3,4,5) |
| C12 | Key-B kappaB, residual form | yes | 3 | −0.00006 ± 0.00032 | −0.19 | 0.5772 | 1.0000 | +0.00003 (+0.16) | +0.00017 | −0.00016 | −0.00037 ± 0.00050 | CLOSED (1,2) |
| C13 | Pending favourite | yes | 9 | −0.00010 ± 0.00017 | −0.57 | 0.7141 | 1.0000 | −0.00006 (−0.78) | −0.00010 | −0.00019 | −0.00022 ± 0.00027 | CLOSED (1,2,3,4) |
| C14 | Cross-feed prediction consensus | STOP | 5 | [info] +0.00022 ± 0.00018 | [info] +1.19 | 1 (STOP) | 1.0000 | [info] −0.00050 (−2.68) | [info] −0.00058 | [info] +0.00004 | [info] +0.00067 ± 0.00035 | CLOSED (coverage gate: 4.54% < 10%) |
| C15 | PLE / PCGrad gradient-cosine pre-gate | yes, not a residual | n/a | n/a | n/a | 1 (no test) | 1.0000 | n/a | n/a | n/a | n/a | CLOSED (no val delta; PROCEED = False) |
| C16 | REGIME4 Step 0 (regcorr) | yes | 3 | +0.00007 ± 0.00013 | +0.51 | 0.3038 | 1.0000 | −0.00005 (−1.07) | −0.00010 | −0.00002 | +0.00006 ± 0.00021 | CLOSED (1,2,3,4) |

Holm order (from `synth_holm.log`, line 1): C07 0.2109, C06 0.2183, C16 0.3038, C09 0.3736, C11 0.4289, C12 0.5772, C03 0.6102, C13 0.7141, C04 0.7221, C05 0.8631, C08 0.9625, C02 0.9801, C10 0.9961, C01 1.0000, C14 1.0000, C15 1.0000. The first comparison already fails (0.2109 > 0.00142), so no item is rejected.

Source lines per item:

**tester1**
- C01: `tester1/logs/c01_eval.log` line 3, "RESULT C01_leafsvd64 [PRIMARY] cols 129".
- C04: `tester1/logs/c04_eval.log` line 12.
- C07: `tester1/logs/c07_eval.log` line 17.
- C10: `tester1/logs/c10_eval.log` line 23.
- C13: `tester1/logs/c13_eval.log` line 5.
- C16: `tester1/logs/c16_eval.log` line 12.

**tester2**
- C02: `tester2/logs/c02.log` lines 39-43.
- C05: `tester2/logs/eval_covis.log` lines 16-20.
- C11: `tester2/logs/eval_covis.log` lines 32-36.
- C08: `tester2/logs/c08.log` lines 20-24.
- C14: `tester2/logs/c14.log`, line 4 (the coverage gate, printed first) and lines 12-16.

**tester3**
- C03: `tester3/logs/c03_test.log` line 9.
- C06: `tester3/logs/c06_test.log` line 9.
- C09: `tester3/logs/c09_test.log` line 5.
- C12: `tester3/logs/c12_test.log` line 12.
- C15: `tester3/logs/c15.log`, last line, "PROCEED flag ... False w5_e10_base_s951@70000 1/53 | s952@70000 9/53 | s951@last 7/53 | s952@last 10/53".

---

## 2. Registry one-liner (whole batch)

- 11-13 19:09 **CCB-1113 CPU CLOSURE BATCH (a workflow run; the 16 items of EXTERNAL_SCAN_1113 §2.3; rule nn_x/insights/ccb_1113/PREDECLARED_ccb.txt 18:31:28Z; result nn_x/insights/ccb_1113/RESULT.md): 0/16 PASS, 16 CLOSED.**
  - Method: a nested LIN residual on G3 (user-grouped 5-fold); the full-val weight is applied to the 30K holdout; Holm over 16 at α 0.02275.
  - Every Holm p is 1.0. No item reaches +0.0005 val even unadjusted: best C07 item trend +0.00031 ± 0.00038 (z +0.80), C06 Bradley-Terry +0.00028 ± 0.00037.
  - Holdout is below 0 on 13 of 15 items, and so is late. The pmix veto fires on 6 (C01 C02 C08 C09 C10 C11).
  - Worst: C01 leaf-SVD64 −0.00264 (z −4.1; the 3-column GBDT-score discriminator gives −0.00008 ± 0.00021) and C10 user-aggregate × cue interactions −0.00155 (z −2.7). The negative tail tracks design size: corr(LIN cols, Δ) = −0.87.
  - C14 consensus: STOP (val coverage 4.5% < 10%).
  - C15 main-vs-K9 gradient cosine: PROCEED False (median cos ≈ +0.01, near-orthogonal, so there is no conflict to resolve).
  - Scope: this closes the post-hoc residual form only, because the lens recovers ~15% of a known input gap. The NN-input forms (leaf-SVD, history sketches) remain unpriced, but no item gained any evidence. No GPU arm follows.

---

## 3. COVERAGE_MAP rows (section "CPU closure batch 11-13 (CCB-1113)")

These rows replace the single §6.1 row "CPU closure batch (streak, latent geometry, ...) | UNTESTED-EXTERNAL / RE-OPEN". The lead should also update two existing rows:
- "Cross-feed prediction consensus | UNTESTED-EXTERNAL" becomes CLOSED (C14).
- "GBDT leaf-SVD / GBDT-OOF score as NN input | UNTESTED-EXTERNAL" keeps the NN-input form as UNTESTED, with the residual form CLOSED (C01).

| idea | status | best evidence | file |
|---|---|---|---|
| C01 GBDT-OOF score + leaf-SVD64 (LightGBM lambdarank on 134K XM feeds, pred_leaf → SVD64) as a residual on G3 | CLOSED 11-13 (CCB, residual form only; NN-input form untested) | val −0.00264 ± 0.00064 (z −4.11), hold −0.00136 (z −3.82), pmix veto; score-only discriminator val −0.00008 ± 0.00021, hold −0.00004 | ccb/tester1/logs/c01_eval.log; RESULT.md |
| C02 Streak / EWMA (τ 4/16/64) + strike Σ2^-k + per-type same-category recency | CLOSED 11-13 (CCB) | val −0.00084 ± 0.00041 (z −2.06), hold +0.00002, late +0.00036, pmix veto | ccb/tester2/logs/c02.log |
| C03 Latent geometry (loc16 subspace, mode6/96 sides, f2-supervised W: g_shift, g_f2res, g_cmatch) | CLOSED 11-13 (CCB) | val −0.00008 ± 0.00030, hold −0.00029 (z −2.08), late −0.00054; best single column g_cmatch +0.00022 ± 0.00014 (info) | ccb/tester3/logs/c03_test.log |
| C04 Co-vis variants w5/fwd5/t1/seqs + OTTO-7th NB log-lift (B9 day source) | CLOSED 11-13 (CCB) | val −0.00020 ± 0.00034, hold −0.00030 (z −1.70); coldpos +0.00595 / warmpos −0.00245 (cold/warm trade) | ccb/tester1/logs/c04_eval.log |
| C05 IDF / IUF / BM25 / idf-damped invd20 / Swing co-vis counts | CLOSED 11-13 (CCB) | val −0.00034 ± 0.00031, hold −0.00013, late −0.00030; invd20 copy validated (corr 0.9997 val / 1.00000 hold) | ccb/tester2/logs/eval_covis.log |
| C06 Bradley-Terry / first-choice item strength (λ 3; train p_random > 0.5) | CLOSED 11-13 (CCB) | val +0.00028 ± 0.00037, hold −0.00015, late −0.00031; mostly a cold/warm trade (bt_theta AUC 0.5006); it_tr_coec alone +0.00060 ± 0.00033 (info) | ccb/tester3/logs/c06_test.log |
| C07 Item trend / lifecycle at a gap-matched as-of (it_n7, it_trend, it_slope7, it_q95gap, exhaust) | CLOSED 11-13 (CCB) | val +0.00031 ± 0.00038 (best of the batch), hold −0.00024, late −0.00047; coldpos +0.00432 / warmpos −0.00123; exhaust test z-shift −0.307 | ccb/tester1/logs/c07_eval.log |
| C08 EMDE multi-partition sketch (32 × 6 projections, k64, per type group) | CLOSED 11-13 (CCB) | val −0.00063 ± 0.00036, hold −0.00055 (z −2.96), late −0.00070, pmix veto | ccb/tester2/logs/c08.log |
| C09 RP3beta whole-sequence walk (β 0.5 / 0) | CLOSED 11-13 (CCB) | val +0.00012 ± 0.00036, hold −0.00013, pmix veto; coldpos −0.00400 / warmpos +0.00211 | ccb/tester3/logs/c09_test.log |
| C10 User aggregates F26 + F27 × 10 cues (160 interactions), holdout as-of tte_u_* | CLOSED 11-13 (CCB) | val −0.00155 ± 0.00058 (z −2.66), hold −0.00141 (z −4.45), pmix veto; inter_resid GBM F26 +0.00022 ± 0.00026 / hold +0.00016 (info) | ccb/tester1/logs/c10_eval.log, c10_info_gbm.log |
| C11 Typed implicit rating cvx_g1/g2/g3 / intent / tw | CLOSED 11-13 (CCB) | val +0.00006 ± 0.00033, hold −0.00018, pmix veto | ccb/tester2/logs/eval_covis.log |
| C12 Metric-optimal post-processing key B (kappaB) on G3, residual form | CLOSED 11-13 (CCB) | val −0.00006 ± 0.00032, hold +0.00003, late +0.00017; deterministic key log p + kappaB +0.00025 ± 0.00031 / hold −0.00003 (info) | ccb/tester3/logs/c12_test.log |
| C13 Pending favourite (saved, not contacted) p1/p2/p3 | CLOSED 11-13 (CCB) | val −0.00010 ± 0.00017, hold −0.00006, late −0.00010 | ccb/tester1/logs/c13_eval.log |
| C14 Cross-feed prediction consensus (TO-G3 z of the item in other users' earlier feeds) | CLOSED 11-13 (CCB STOP, coverage gate) | val rows with n >= 1: 4.54% < 10% (day 20718 0% by B9); info val +0.00022 ± 0.00018, hold −0.00050 (z −2.68) | ccb/tester2/logs/c14.log |
| C15 PLE / PCGrad pre-gate: main vs K9 gradient cosine on w5_e10_base | CLOSED 11-13 (CCB; PROCEED = False) | median cos < 0 on 1/53 (s951@70000), 9/53 (s952@70000), 7/53 and 10/53 @last; median of medians +0.007..+0.014 (near-orthogonal) | ccb/tester3/logs/c15.log |
| C16 REGIME4 Step 0: Sep-12 production-regime indicator, GBDT proxy (regcorr) | CLOSED 11-13 (CCB; the spec's own kill read also closes) | val +0.00007 ± 0.00013, hold −0.00005, late −0.00010; corr(regime, base) 0.995; kill read: no arm > +1 SE (droppre −0.00268, droppost −0.00126) | ccb/tester1/logs/c16_eval.log |

(`ccb/` = `/tmp/scratchpad/ccb/`, session-local.)

---

## 4. Honest reading

1. **A clean null, slightly below null in aggregate.** Over the 14 tested, non-STOP items, `synth_holm.log` gives:
   - mean z −0.767, Stouffer sum z/√n −2.869;
   - 5 of 14 with z > 0, max z +0.80 (C07);
   - 0 items with Δ >= +0.0005.

   Under a pure null, the largest of 14 z values would usually sit near +1.5 to +2. No item comes close to the bar in either direction that matters.

2. **The negative tail measures fit penalty more than harm.** corr(LIN parameter count, Δ) = −0.871 (`synth_holm.log`).
   - The two most negative items are also the two largest designs: C01 has 129 parameters and C10 has 171.
   - LIN is fitted on 4/5 of 14,423 feeds without a penalty, so many weak columns cost NDCG out of fold.
   - For these two, the cleaner reads are the small information contrasts, and they are also null:
     - C01: GBDT score vs [b], val −0.00008 ± 0.00021, holdout −0.00004 (`tester1/logs/c01_eval.log` line 4);
     - C10: GBM inter_resid F26, val +0.00022 ± 0.00026, holdout +0.00016 (`tester1/logs/c10_info_gbm.log`).
   - So "C01/C10 hurt" should be read as "a high-dimensional post-hoc residual of these inputs does not help G3", not as evidence that the information is harmful.

3. **Holdout and late agree with val.** The holdout point estimate is below 0 on 13 of 15 items, and the late column also on 13 of 15 (`synth_holm.log`). The 16:10 audit found that an uninformative val-fitted tilt costs about −0.0002..−0.0004 here (DATA_PIPELINE_1113 §2.1), so this is the expected pattern, not new harm. The only non-negative holdouts are C02 (+0.00002) and C12 (+0.00003), both with negative val. Criteria (3)-(5) were never decisive, because (1) failed everywhere.

4. **Cold/warm trades again.** These slice values come from the tester logs, not from `synth_holm.log`:
   - C04: coldpos +0.00595, warmpos −0.00245;
   - C07: +0.00432 / −0.00123;
   - C06: +0.00315 / −0.00094;
   - C09: reverse sign, −0.00400 / +0.00211.

   These are the popularity-regime trades that net to about 0 and lose on the holdout (DATA_PIPELINE_1113 §2.3; CURRENT_STATUS §3 "stale-val for any cold/warm trade"). No item found item-level signal beyond what G3's popularity and co-visitation inputs already carry.

5. **What the batch bounds, and what it does not.**
   - The one-sided val upper bounds (Δ + 2 SE) reach at most +0.00108 (C07), with a median near +0.0005 (`synth_holm.log`). That bounds a post-hoc residual over G3.
   - The lens recovers only about 15% of a known input gap (DATA_PIPELINE_1113 §2.4). pre2 fails this lens yet is worth +0.0012 as an NN input (PRETRAIN_PLAN:312).
   - So the batch is weak evidence about the NN-input forms of C01 (leaf-SVD), C02, C03, C08 (history sketches) and C04, C05, C09, C11 (co-visitation variants). Those forms remain unpriced.
   - It also produced no positive signal that would raise any item's prior (0.01-0.02). Under the project rule, a proxy gain never justifies NN GPU, and here there is not even a proxy gain. No GPU arm follows from this batch.

6. **Item-specific information.**
   - **C15:** the main and K9 gradients are near-orthogonal (median cosine about +0.01), not conflicting. PCGrad/PLE has no conflict to remove, so the premise of a K9-interference fix is absent.
   - **C14:** the coverage failure is structural. Under the B9 rule, val day 20718 has no earlier val day (0% coverage), and test is never a source. Its information read, holdout −0.00050 (z −2.68), points the same way.
   - **C16:** the regime indicator barely moves the GBDT proxy (corr 0.995 to base), and the spec's own kill read closes it too.
   - **C12:** post-processing key B adds nothing, either as a residual (−0.00006) or as a deterministic key (+0.00025 ± 0.00031, holdout −0.00003).

7. **Deviations and process notes (disclosed).**
   - **Noise calibration:** B8 asked for 4 N(0,1) noise columns run once in the LIN harness. Only a 1-column version ran (`tester1/logs/t00_sanity.log`): val +0.00009 ± 0.00014, holdout −0.00000 ± 0.00006. The s04 band (−0.00047..+0.00022) serves as the null reference instead.
   - **C01:** the E3 JSON lists 177 features, not the 174 in the predeclaration. All 177 were used.
   - **C10:** the spec's "150 interactions" is an arithmetic slip; its 16 named aggregates × 10 cues give 160, and all 160 were used.
   - **C12:** pair counts were recomputed over all 14,423 val feeds, because 05's cache had excluded 117 feeds (tester3's note; the log prints only "pair H built (all val feeds)", `tester3/logs/c12_test.log` line 2).
   - **C15:** @60000 is not on disk, as pre-disclosed; @70000 and @last were used.
   - **C07:** the holdout as-of (feed day − 5 d) gives much lower coverage than val (it_n7 nonzero 9.9% vs 23.1%).
   - **C04 / C05 / C09:** the 2,674 Sep-6 holdout feeds have an empty source and get 0.
   - **Marginal vetoes:** the pmix veto fires only marginally on C02 (−0.00061 vs SE 0.00060) and C09 (−0.00056 vs 0.00055). Both items fail (1) anyway.
   - No item was re-specified after its first number. Every extra variant is labelled INFORMATION.

8. **Where this leaves the program.** This batch, wave 10 (0/4) and the two 11-13 scans agree: no visible post-hoc or proxy lever over G3 is left among the external techniques. The remaining open items are the NN-input forms named in point 5, at priors <= 0.02, plus the §2.2 queue items the 18:23 log lists as remaining (R1 EXPLO-STACK gate, R5 example-age gate, R6 KD/PL in the cooldown, R8 ixh input, R10 ADT truncated CE). None of them is justified by evidence from this batch.
