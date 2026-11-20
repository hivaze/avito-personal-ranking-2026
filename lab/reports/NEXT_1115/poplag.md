# poplag: LAGGED-POPULARITY HOLDOUT TEST (rule block 2026-11-14T22:17:17Z in `nn_x/w5/logs/E_kd_predeclared.txt`)

Written 2026-11-15 by the poplag agent. CPU only. Work directory `nn_x/w12/next/poplag/`. Conventions: ± = 1 SE paired per feed; [M] = measured by me, [I] = inferred. "Zero share" = share of candidate rows whose counter is 0. Holdout = the 30,000 in-period train feeds (M >= 50), 4,630,966 candidate rows.

## 0. Verdict

1. **No reading was made: the label-free gate FAILS for the private-like (decision) view.** One of five numeric criteria, written before any column was built, is missed: `xp_imp_c50`, listings aged 10-24 d, zero share 0.246 on the lagged holdout rows against 0.130 on private test rows (gap 0.116, bar 0.10) [M]. By the declaration, no labelled read of any lagged view followed. The fixed rule (>= -0.0002 / <= -0.0005 / between) therefore returns no branch.
2. **Operational meaning:** the lagged test does not clear the popularity-light view, so amendment (9) gives W8P20 no route to pick 1 through this test. It does not close the view either. Pick-2 eligibility stays with rule A' (the public read).
3. The failing column is the pop1kd members' only recency-dependent input. **The eight popularity inputs of the W8 members reach parity with private rows** after the rebuild: age-weighted zero-share gap of `it_hist_all_luo` 0.2474 -> 0.0131, of `seq_cv_pop` 0.1473 -> 0.0090, largest age bucket 0.035 [M].
4. **Control VALID** [M]: the CPU fp32 path with stored inputs reproduces all 12 stored member holdout NDCGs within 0.00018 (bar 0.0010) and the statistic at -0.00047 ± 0.00023 against -0.00063 ± 0.00022 from the stored files (difference +0.00016 ± 0.00011, bar ±0.0003), 30,000 feeds.
5. The premise of the bias argument holds at input level [M, label-free]: 22.0% of `it_hist_all_luo` values are zero on stored holdout rows against 45.5% on private rows (M >= 50); 1.7% of holdout feeds have a cold share >= 0.45 against 50.9% of private feeds. Test-like inputs change the W8 pool's top-1 in 40.5% ± 0.3% of feeds, the pop1kd pool's in 18.2% ± 0.2%. What that does to NDCG is not measured.
6. The residual sits in holdout days Sep 6-12. On holdout days >= Sep 13 (13,424 feeds) all five criteria hold for both views [M, label-free, information only]. The 12 members' predictions on the private-like view are staged and unread; a read on that subset is one command and needs the lead's pre-declaration (SE about 0.00033 [I], too coarse to separate -0.0002 from -0.0005).

## 1. Measurements

### 1.1 Lags and what depends on them (step 1) [M]

| split | day (MSK) | feeds | lag L = day - Sep 20 |
|---|---|---|---|
| train (dense) | Sep 6-14, Sep 17-19 (12 full days) | 3,151,208 (203,548-291,760 per day) | dense data end at Sep 20 00:00 MSK |
| val | Sep 22 / Sep 24 | 7,542 / 6,878 | 2 / 4 |
| test, public (ts < Sep 27 11:32 MSK) | Sep 26 / Sep 27 | 6,050 / 2,512 (8,562) | 6 / 7; mean t - cutoff 6.79 d |
| test, private | Sep 27 / 28 / 29 | 4,855 / 7,578 / 7,600 (20,033) | 7 / 8 / 9; mean t - cutoff 8.76 d |

Lags imposed on the holdout feeds, drawn by a hash of `feed_id` from the test counts above: public-like L = 6 on 21,283 feeds, L = 7 on 8,717; private-like L = 7 on 7,356 feeds, L = 8 on 11,362, L = 9 on 11,282.

How each column is computed on a test row (code read), and so what depends on the lag:

| columns | source | on a test row | on a stored holdout row |
|---|---|---|---|
| `it_hist_{all,et1..4}_luo` (5; W8) | `src/features_luo.py`: occurrences of the item in the histories of all snapshots of train + val + test, minus later snapshots of the same user | dense snapshots end L days before the row; afterwards only val / test snapshots at 1/37 of the dense density | counts also the dense snapshots AFTER the row (not causal) |
| `seq_cv_{invd20,pop,norm}` (3; W8) | `feats_v2/seq_/feat_covis.py`: co-visitation in de-duplicated user sequences | all train sequences + val / test histories before the row | train days before the own day only (0 dense days on Sep 6, 11 on Sep 19) |
| `xp_imp_c50` (1; pop1kd) | `log1p(min(tte_it_imp, 50))`, train impressions of the item | all 12 train days, ending L days before the row | train days before the own day |
| the other 26 xt inputs (`ctx_nd_*`, `ctx_hn_*`, `ctx_hs_*`, `pre2_*`) | own feed and own history only | no dependence | left as stored |

So the pop1kd members are not free of recency: `xp_imp_c50` was rebuilt as well.

### 1.2 Gate (step 3): label-free parity against test rows, by item age [M]

Criterion (fixed at 22:34:33Z in `DECLARATION.txt`, before any column existed). Populations: lagged holdout rows against test rows of the matching part from feeds with M >= 50 (public 806,026 rows of 5,248 feeds; private 1,829,811 rows of 12,186 feeds). dZ = holdout-row-weighted mean over seven age buckets of |zero-share difference|; dM = the same for |mean difference| on the model's input scale, in test SDs.
PASS iff G1 dZ <= 0.05 for the three primaries; G2 no bucket with >= 5% of rows above 0.10 in zero share for the primaries; G3 dM <= 0.20 for the primaries; G4 `it_hist_all_luo` dZ at most one third of its lag-0 value; G5 the six secondary columns dZ <= 0.10 and dM <= 0.30. FAIL of the private-like view stops every lagged read.

Final construction (v3), summary:

| view | column | dZ lag 0 | dZ lagged | largest bucket | dM lagged | pooled zero share: lag 0 / lagged / test |
|---|---|---|---|---|---|---|
| private-like | `it_hist_all_luo` | 0.2474 | 0.0131 | 0.035 | 0.031 | 0.220 / 0.470 / 0.455 |
| private-like | `seq_cv_pop` | 0.1473 | 0.0090 | 0.030 | 0.025 | 0.475 / 0.493 / 0.476 |
| private-like | `xp_imp_c50` | 0.2414 | 0.0362 | **0.116 (G2 FAIL)** | 0.079 | 0.356 / 0.504 / 0.472 |
| private-like | six secondary columns | 0.032-0.243 | <= 0.0144 | <= 0.032 | <= 0.045 | - |
| public-like | `it_hist_all_luo` | 0.2190 | 0.0183 | 0.043 | 0.041 | 0.220 / 0.449 / 0.427 |
| public-like | `seq_cv_pop` | 0.1449 | 0.0097 | 0.017 | 0.029 | 0.475 / 0.482 / 0.464 |
| public-like | `xp_imp_c50` | 0.2227 | 0.0374 | 0.0999 | 0.091 | 0.356 / 0.477 / 0.435 |
| public-like | six secondary columns | 0.031-0.216 | <= 0.0175 | <= 0.039 | <= 0.048 | - |

**Private-like view: FAIL (G2 `xp_imp_c50`). Public-like view: PASS, by 0.0001 on the same criterion; not read, because the declaration stops all lagged reads when the private-like view fails.**

Private-like view by age bucket (zero share: stored holdout / lagged holdout / private test rows):

| age of listing | rows | `it_hist_all_luo` | `seq_cv_pop` | `xp_imp_c50` |
|---|---|---|---|---|
| < 1 d | 11.9% | 0.532 / 0.982 / 0.975 | 0.981 / 0.997 / 0.996 | 0.793 / 1.000 / 1.000 |
| 1-3 d | 8.0% | 0.320 / 0.936 / 0.916 | 0.748 / 0.967 / 0.955 | 0.291 / 1.000 / 1.000 |
| 3-7 d | 10.2% | 0.229 / 0.895 / 0.859 | 0.513 / 0.930 / 0.899 | 0.204 / 1.000 / 1.000 |
| 7-10 d | 4.8% | 0.192 / 0.822 / 0.786 | 0.415 / 0.853 / 0.816 | 0.210 / 0.804 / 0.709 |
| 10-24 d | 12.7% | 0.176 / 0.389 / 0.395 | 0.379 / 0.410 / 0.410 | 0.267 / **0.246 / 0.130** |
| 24-76 d | 16.8% | 0.149 / 0.190 / 0.192 | 0.342 / 0.212 / 0.208 | 0.315 / 0.238 / 0.192 |
| > 76 d | 35.7% | 0.144 / 0.187 / 0.200 | 0.338 / 0.208 / 0.214 | 0.337 / 0.262 / 0.288 |

History of the gate (all label-free; the criterion never changed, the construction did twice; each step was written into `DECLARATION.txt` before it was built):

| construction | private-like | public-like |
|---|---|---|
| v1: exposure-matched lag (section 2) | FAIL: `it_hist_all_luo` bucket 10-24 d 0.101; `xp_imp_c50` dZ 0.0617, bucket > 76 d 0.120 | FAIL: `xp_imp_c50` dZ 0.0582, bucket 0.105 |
| v2: v1 + exclusion band g around the row; g chosen by J(g) = sum of the three primary dZ on the private-like view: 0.1452, 0.1294, 0.0941, **0.0693**, 0.0989, 0.1771, 0.2659 for g = 0..6 -> g = 3 | FAIL: `xp_imp_c50` bucket 10-24 d 0.193 | FAIL: `xp_imp_c50` bucket 0.156 |
| v3: v2 at g = 3, `xp_imp_c50` keeps its exposure days under the band (declared as the last change) | **FAIL: `xp_imp_c50` bucket 10-24 d 0.116** | PASS (0.0999) |

Where the residual sits (`xp_imp_c50`, private-like, zero-share gap of the 10-24 d bucket by holdout day): Sep 6-8 +0.234 (7,823 feeds), Sep 9-12 +0.115 (8,753), Sep 13-14 +0.072 (5,145), Sep 17-19 +0.035 (8,279) [M]. Cause [I]: a listing created before Sep 6 has no observable first days of life; impressions are front-loaded, so a later window of the same length undercounts them.

Information only, not the declared gate: the same five criteria on holdout days >= Sep 13 (13,424 feeds, 2,089,648 rows) are all met: private-like `it_hist_all_luo` dZ 0.0156 / bucket 0.050 / dM 0.045, `seq_cv_pop` 0.0145 / 0.045 / 0.038, `xp_imp_c50` 0.0225 / 0.049 / 0.053, secondary dZ <= 0.0163; public-like 0.0237 / 0.064, 0.0126 / 0.033, 0.0197 / 0.034 [M].

Feed-level view of the same fact (per-feed share of rows with `it_hist_all_luo` = 0; q10 / q50 / q90 and share of feeds >= 0.45) [M]:

| population | q10 | q50 | q90 | feeds >= 0.45 |
|---|---|---|---|---|
| holdout, stored (lag 0) | 0.113 | 0.210 | 0.340 | 1.7% |
| holdout, private-like v3 | 0.346 | 0.468 | 0.600 | 57.3% |
| private test feeds, M >= 50 | 0.337 | 0.452 | 0.581 | 50.9% |
| holdout, public-like v3 | 0.325 | 0.447 | 0.579 | 48.9% |
| public test feeds, M >= 50 | 0.309 | 0.425 | 0.551 | 40.0% |

### 1.3 Control (step 4): CPU fp32 replay of the stored inputs against the stored bf16 files, 30,000 feeds [M]

| member | stored | CPU replay | paired CPU - stored |
|---|---|---|---|
| w5_e10_kd05w192_s951 | 0.32258 | 0.32247 | -0.00012 ± 0.00010 |
| w5_e10_kd05w192_s952 | 0.32298 | 0.32299 | +0.00002 ± 0.00009 |
| w5_e10_kd05w192_s953 | 0.32263 | 0.32250 | -0.00013 ± 0.00009 |
| w5_e10_kd05w192_s954 | 0.32247 | 0.32250 | +0.00003 ± 0.00010 |
| w5_e10_kd05t25w192_s951 | 0.32261 | 0.32255 | -0.00006 ± 0.00010 |
| w5_e10_kd05t25w192_s952 | 0.32303 | 0.32298 | -0.00005 ± 0.00010 |
| w5_e10_kd05t25w192_s953 | 0.32253 | 0.32237 | -0.00016 ± 0.00010 |
| w5_e10_kd05t25w192_s954 | 0.32220 | 0.32210 | -0.00010 ± 0.00009 |
| abl_e10_pop1kd05t25_s951 | 0.31596 | 0.31613 | +0.00016 ± 0.00009 |
| abl_e10_pop1kd05t25_s952 | 0.31653 | 0.31645 | -0.00008 ± 0.00010 |
| abl_e10_pop1kd05t25_s953 | 0.31623 | 0.31604 | -0.00018 ± 0.00010 |
| abl_e10_pop1kd05t25_s954 | 0.31627 | 0.31621 | -0.00006 ± 0.00010 |
| W8 pool | 0.32456 | 0.32449 | -0.00007 ± 0.00009 |
| pop1kd pool | 0.31715 | 0.31708 | -0.00006 ± 0.00009 |

| statistic at lag 0 | stored files | CPU replay | CPU - stored |
|---|---|---|---|
| w = 0.1 | -0.00027 ± 0.00015 | -0.00015 ± 0.00015 | +0.00013 ± 0.00010 |
| w = 0.2 | **-0.00063 ± 0.00022** | -0.00047 ± 0.00023 | +0.00016 ± 0.00011 |
| w = 0.3 | -0.00104 ± 0.00028 | -0.00093 ± 0.00028 | +0.00011 ± 0.00012 |

Control: VALID (largest member difference 0.00018 < 0.0010; statistic within ±0.0003). The stored statistic -0.00063 ± 0.00022 reproduces exactly from the stored files. Note for any later read: the numerics alone move the statistic by +0.00016, half the distance between the two rule thresholds. Judge the paired change (lagged minus lag 0, CPU against CPU) added to -0.00063, not the CPU level.

### 1.4 Read (steps 5-6): not made

No lagged NDCG of any kind was computed: neither the statistic at any weight, nor a pool or member level change, nor a cold-share slice. `read.py read` and `read_late.py` were never run. Rule applied: gate FAIL -> no reading -> none of the three branches.

### 1.5 Label-free prediction shift of the staged private-like predictions (information; no label opened)

Private-like v3 predictions against the lag-0 CPU replay, 30,000 feeds [M]. These are W8 inputs that pass every criterion; the pop1kd rows carry the failed `xp_imp_c50` column.

| pool | per-feed corr(lag 0, lagged) | feeds whose top-1 changes | top-10 overlap | same, holdout days >= Sep 13 |
|---|---|---|---|---|
| W8 (8 members) | 0.9810 ± 0.0001 | 40.5% ± 0.3% | 0.784 | corr 0.9875, top-1 changed 32.8% |
| pop1kd (4 members) | 0.9942 ± 0.0000 | 18.2% ± 0.2% | 0.897 | corr 0.9974, top-1 changed 13.3% |

- Per-feed corr(W8 pool, pop1kd pool): 0.9765 at lag 0, 0.9795 lagged (paired +0.00306 ± 0.00005). The two pools move closer when the popularity inputs go stale.
- Mean change of the within-feed z by row class: rows that lose their history count (25.0% of rows) W8 -0.120, pop1kd -0.019; rows already cold at lag 0 (22.0%) W8 +0.108, pop1kd 0.000; rows that keep a count (53.0%) W8 +0.011, pop1kd +0.009.
- Reading [I]: the W8 pool ranks by the fresh counters to a degree the pop1kd pool does not: test-like inputs reorder its top-1 in two feeds of five. That supports the premise of the bias argument. It says nothing about the sign of the NDCG effect, which is the labelled read that was not made.

## 2. Method

**Why not a plain "counts up to T - lag".** The dense data start on Sep 6: a holdout row of Sep 6 + k with lag L has k - L dense days where a test row has 12. With L = 7-9 most feeds get an empty source. That is the known T4 artefact (`COVERAGE_MAP` row 220: 63.6% of feeds without history). Depth and lag have to be separated.

**Construction v3** (per holdout feed f, candidate row with item j; `build_lag2.py`, `build_xp3.py`):
- L_f from the hash of `feed_id`; cutoff c_f = 00:00 MSK of day(t_f) - L_f.
- Listing creation time t_post = t_f - age(f4) (`data/feats/life_f4_age_knots.npy`); E = c_f - t_post is the listing's age at the cutoff, the quantity a test row has as Sep 20 - t_post.
- Dense cutoff day D = day boundary nearest to min(Sep 20, max(c_f, Sep 6 + clip(E, 0, 14 d))). The row keeps min(E, 14 d) days of dense exposure, as a test row does. For a listing created inside the data this is exactly "counts up to T - lag". For E >= 14 d it is the full dense window. For a listing created before Sep 6 with partial exposure it is the first E days of the data (11.1% of rows in the private-like view).
- Exclusion band g = 3 days: a dense snapshot or impression of day d counts only if d < D and |d - day(t_f)| >= 3. On a test row every dense snapshot is at least L days before the row; g = L is the T4 depth collapse, so g is a calibrated compromise (one parameter, chosen on the summed zero-share gap of the private-like view, label-free).
- Sparse days: the six days c_f + {2, 4, 6, 7, 8, 9} (the offsets of the val / test days from Sep 20). Sparse snapshots are train feeds of the shared 1/37 hash sample (`feat_new/hz_common.sparse_flag`) or val / test feeds on those days. Days before Sep 6 or without data are lost, not replaced (4.26 of 6 days available on average, private-like).
- `it_hist_*`: dense occurrences + sparse occurrences + the own snapshot; later snapshots of the same non-zero user removed (stored rule).
- `seq_cv_*`: the stored kernel (copied) with membership per candidate: dense = latest feed of each user before D outside the band, sparse = sparse feeds before the own timestamp; same-user sequences skipped; scan cap 3000.
- `xp_imp_c50`: impressions on the first (D - Sep 6) days outside the band, minus the own feed's row and rows of later feeds of the same non-zero user.
- No label is opened in the construction, the gate, or the calibration of g. Test rows enter only through their input columns.

**Code checks** [M]: with lag-0 semantics the same kernels reproduce the stored columns on all 4,630,966 holdout rows: the five `it_hist_*` columns, `seq_cv_invd20` and `seq_cv_pop` bit for bit; `seq_cv_norm` and `xp_imp_c50` to 5e-7. `build_xp3.py` at v2 semantics equals the `build_lag2.py` file exactly.

**Inference path** (`infer.py`): a mini split of the 30,000 holdout feeds in the `SplitData` layout, the unmodified `gather` / `batches` / `scatter` of `nn_x/w5/data.py`, `Ranker` from the checkpoint cfg (`nn_x/w5/model.py`, identical to the abl copy), batching of `common.predict` (65536 / 1024), train-feed domain mix [1 - p_random, p_random, 0], fp32 on CPU. 4.0-6.4 min per d192 member and 2.8-3.7 min per d128 member at 12 threads (machine load 40-85).

**Pools and statistic** (`read.py`): member mean of per-feed z as `src/stack_val.py feed_rank_z[:, 1]`; S(w) = NDCG((1 - w) z(W8 pool) + w z(pop1kd pool)) - NDCG(W8 pool); NDCG = `src/metrics.py`.

## 3. Limits, and what I think of the rules

- **The gate bar that failed is mine, and I think it is stricter than the question needs.** It treats the pop1kd members' single weak counter like the W8 members' eight inputs, and it fails by 0.016 in one bucket holding 12.7% of rows. The bias claim is about the W8 inputs, and those pass with a wide margin. I applied the bar as written.
- **I rebuilt twice after the first FAIL.** A strict reading of "if FAIL: stop" ends at v1. Both rebuilds were label-free, were written down before they were built, and left the criterion untouched; the outcome is the same (no read). The lead may prefer to count v1 as the gate result.
- **The old-listing part of the view is calibrated, not derived.** g = 3 was chosen to minimise the coverage gap against private rows. One parameter then matches seven age buckets of two families and their means, which supports the mechanism, but the gate is not an independent check of that parameter.
- **Marginal parity is not joint parity.** The gate compares distributions by item age. It cannot show that the right items are cold.
- **Known residuals of v3** [M]: listings younger than 10 d are 0.02-0.04 colder than private rows in zero share (lost sparse days on early holdout days); the history-neighbour side of `seq_cv_invd20` is fresher than on a test row when D > c_f (mean +0.043 SD).
- **Resolution** [I]: the full-holdout statistic has SE 0.00023 and the two rule thresholds are 0.0003 apart. Even with a passing gate the reading would land between the thresholds with sizeable probability; the leak skeptic's prior that the test is decisive was 0.3.
- The public-like view passes by 0.0001. I would not lean on that pass.

**What would let the test be read** (each needs a line in `E_kd_predeclared.txt` before the read):
1. Read on holdout days >= Sep 13 only (13,424 feeds; all criteria met for both views). One command, `read_late.py priv_v3`; SE about 0.00033 [I].
2. Repair `xp_imp_c50` for listings created before Sep 6 with an impression life-cycle correction estimated on listings created inside the data, then re-run the gate on the 30,000 feeds. Only the four pop1kd members need new inference (about 13 CPU-min); the W8 predictions staged here stay valid.

## 4. Not checked

- No lagged NDCG of any kind (section 1.4). So: whether the penalty shrinks, the W8 and pop1kd pool level drops under lag, weights 0.1 / 0.3, cold-share bins.
- Public-like predictions were not produced (about one hour at 12 threads: `infer.py pub_v3`).
- `@last` checkpoints, the fine-tuned branch, the 16-member pool, the wide pop1 students, hard-label pop1 members.
- A GPU bf16 replay of the lagged inputs (CPU fp32 only; rule 1).
- Whether the item-age estimate from f4 is equally accurate on holdout and test rows.
- Sensitivity of the gate to the lag hash, to the day-rounding of D and to the choice of M >= 50 test feeds as the reference.

## 5. Commands to reproduce

```
cd <repo>/nn_x/w12/next/poplag
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1
PY=<repo>/.venv/bin/python
$PY infer.py lag0                      # mini split + CPU replay of the stored inputs, 12 members (50-60 min)
$PY build_lag.py                       # v1 columns (views pub, priv) + code checks (about 65 s)
$PY gate.py                            # gate v1 -> logs/gate.txt (kept as logs/gate_v1.txt)
$PY build_lag2.py 0,1,2,3,4,5,6        # v2 columns for the band values (about 165 s)
$PY gate.py _v2cal priv_g0 priv_g1 priv_g2 priv_g3 priv_g4 priv_g5 priv_g6 pub_g0 pub_g1 pub_g2 pub_g3 pub_g4 pub_g5 pub_g6
$PY build_xp3.py                       # v3 xp_imp_c50; views priv_v3, pub_v3
$PY gate.py _v3 priv_v3 pub_v3         # FINAL gate -> logs/gate_v3.txt
$PY gate.py _v3_late priv_v3 pub_v3    # information: holdout days >= Sep 13
$PY gate_sub.py                        # residual by holdout day, feed-level cold share
$PY read.py control                    # control -> logs/control.txt
$PY infer.py priv_v3                   # label-free staging of the private-like predictions (about 60 min)
$PY diag.py priv_v3                    # label-free prediction shift -> logs/diag_priv_v3.txt
# NOT RUN (needs a pre-declaration):  $PY read_late.py priv_v3    |    $PY read.py read priv_v3
```

## 6. Files created

All under `nn_x/w12/next/poplag/` unless stated.
- `DECLARATION.txt`: design, gate criterion and the two construction amendments, each time-stamped before its build.
- Scripts: `pl_common.py`, `infer.py`, `build_lag.py` (v1), `build_lag2.py` (v2), `build_xp3.py` (v3), `gate.py`, `gate_sub.py`, `read.py`, `read_late.py` (not run), `diag.py`, `chain.sh`, `chain2.sh`.
- Logs: `logs/gate_v1.txt`, `logs/gate_v2cal.txt`, `logs/gate_v3.txt` (final), `logs/gate_v3_late.txt`, `logs/gate_sub.txt`, `logs/control.txt`, `logs/build_lag.log`, `logs/build_lag2.log`, `logs/infer_*.log`, `logs/diag_priv_v3.txt`.
- Cache (4.3 GB): `cache/mini/` (mini split), `cache/lag_<view>_<column>.npy` (views pub, priv, pub_g0..6, priv_g0..6, pub_v3, priv_v3; raw feat_cache scale, holdout row order), `cache/view_<pub|priv>_{L,D,A}.npy`, `cache/pred_lag0_<member>.npy` and `cache/pred_priv_v3_<member>.npy` (12 each), `cache/gate*.json`.
- `reports/NEXT_1115/poplag.md` (this file).

## 7. Compliance

- CPU only, `CUDA_VISIBLE_DEVICES=""` in every command, at most 12 threads per process, one heavy process at a time. Four short jobs (`gate.py`, `gate_sub.py`, `build_xp3.py`, `read.py control`; about a minute or less each, 1-4 threads) ran next to the inference.
- Processes stopped: only my own, by exact PID after reading `/proc/<pid>/cmdline` (two `infer.py lag0` runs at member boundaries, one `chain.sh`). Nothing is left running.
- **One breach of rule 3 / 4, by accident:** an exploratory `import hz_common` at 22:22:33Z made Python create `feat_new/__pycache__/` and `feat_new/__pycache__/hz_common.cpython-312.pyc` (12,222 bytes). Neither existed before. No existing file was edited, moved or overwritten, and nothing a lane executes changed. I left both in place rather than delete outside my directory; `rm -r feat_new/__pycache__` restores the previous state. Later commands imported no module from a directory without an existing bytecode cache.
- No install, no network, no submission file. No val / test label was opened; test rows were used through their input columns only. Same-user later snapshots are excluded in every rebuilt column. No quarantined file was used.
