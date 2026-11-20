## 2. What the arms show

### 2.1 Level against the number of trees (staged predictions, val / holdout) [M]

All trees up to N are used here; the arm table uses the best train-side selection iteration (for S1f: 858 trees, 0.38404, against 0.38425 with all 886).

| trees | S1f RMSE lr 0.3 | S1 RMSE lr 0.1 | S3 = S1f + pca block | S2 depth 10 | S4b QuerySoftMax, soft label | S4 QuerySoftMax, KD-mix |
|---|---|---|---|---|---|---|
| 100 | 0.37910 / 0.30204 | | | 0.37955 / 0.30326 | 0.38076 / 0.30478 | 0.37955 / 0.30373 |
| 200 | 0.38082 / 0.30436 | | | 0.38106 / 0.30523 | 0.38265 / 0.30687 | 0.37981 / 0.30506 |
| 300 | 0.38183 / 0.30510 | 0.38038 / 0.30264 | 0.38038 / 0.30513 | 0.38189 / 0.30609 | 0.38368 / 0.30774 | 0.37994 / 0.30518 |
| 400 | 0.38268 / 0.30614 | | | (cap 307) | 0.38361 / 0.30821 | (cap 316) |
| 500 | 0.38323 / 0.30651 | | 0.38237 / 0.30681 | | 0.38456 / 0.30861 (497, cap) | |
| 600 | 0.38337 / 0.30675 | 0.38157 / 0.30481 | 0.38282 / 0.30706 | | | |
| 721 | 0.38377 / 0.30696 | | 0.38359 / 0.30732 (cap) | | | |
| 886 | 0.38425 / 0.30737 (cap) | | | | | |
| 1000 | | 0.38328 / 0.30637 | | | | |
| 1500 | | 0.38379 / 0.30716 | | | | |
| 2018 | | 0.38449 / 0.30754 (cap) | | | | |

Every arm except S4 was stopped by its wall-time cap while its selection curve was still rising slowly (the best selection iteration is within 30 trees of the last one). The reported levels are therefore levels at a CPU budget, not plateaus; the plateau read for the best arm is the "long" run of section 3.

### 2.2 Paired comparisons (per feed; val | holdout | pooled z) [M]

| comparison | val | holdout | pooled z | reading |
|---|---|---|---|---|
| S1 - S1f (135 min vs 55 min) | +0.00049 ± 0.00065 | +0.00026 ± 0.00037 | +1.03 | lr 0.1 x 2,014 trees is not measurably better than lr 0.3 x 858 trees |
| S1 - hard-label reference `ts_lr020_s0` | +0.00303 ± 0.00093 | +0.00300 ± 0.00055 | +6.18 | the teacher target is worth +0.003 of level to a CatBoost model |
| S1f - `ts_lr020_s0` | +0.00254 ± 0.00101 | +0.00275 ± 0.00058 | +5.12 | same |
| S3 - S1f, equal rounds 300 | -0.00144 ± 0.00079 | +0.00003 ± 0.00048 | -1.24 | |
| S3 - S1f, equal rounds 500 | -0.00086 ± 0.00084 | +0.00031 ± 0.00048 | -0.27 | |
| S3 - S1f, equal rounds 721 | -0.00018 ± 0.00087 | +0.00036 ± 0.00049 | +0.38 | the embedding block adds nothing (1.7x the cost per tree) |
| S3 - S1f, final vs final (75 vs 55 min) | -0.00051 ± 0.00087 | -0.00011 ± 0.00049 | -0.57 | |
| S2 - S1f, equal rounds 100 | +0.00045 ± 0.00067 | +0.00122 ± 0.00043 | +2.45 | |
| S2 - S1f, equal rounds 200 | +0.00023 ± 0.00071 | +0.00086 ± 0.00044 | +1.62 | |
| S2 - S1f, equal rounds 300 | +0.00006 ± 0.00072 | +0.00099 ± 0.00044 | +1.68 | depth 10 gains per tree on the holdout only |
| S2 - S1f, equal wall time (45 min: 305 vs 721 trees) | -0.00152 ± 0.00075 | -0.00065 ± 0.00044 | -2.47 | depth 10 loses per CPU minute (2.4x the cost per tree) |
| S4b - S1f, equal rounds 100 | +0.00165 ± 0.00082 | +0.00274 ± 0.00052 | +5.20 | |
| S4b - S1f, equal rounds 300 | +0.00185 ± 0.00084 | +0.00264 ± 0.00049 | +5.39 | |
| S4b - S1f, equal rounds 497 / 500 | +0.00133 ± 0.00085 | +0.00210 ± 0.00047 | +4.24 | the listwise soft-label loss learns faster per tree |
| S4b - S1f, equal wall time (55 min: 497 vs 858 trees) | +0.00052 ± 0.00084 | +0.00124 ± 0.00047 | +2.32 | and is still ahead at equal CPU, on the holdout mainly |
| S4b - S1 (55 min vs 135 min) | +0.00003 ± 0.00078 | +0.00099 ± 0.00044 | +1.61 | |
| S4b - S4 | +0.00435 ± 0.00103 | +0.00359 ± 0.00055 | +7.64 | mixing the click into the listwise label costs 0.004 |
| S4 - `ts_lr020_s0` | -0.00129 ± 0.00102 | +0.00040 ± 0.00054 | -0.37 | the KD-mix label gives a hard-label-like tree (section 4) |
| S4b - `ts_lr020_s0` | +0.00306 ± 0.00098 | +0.00399 ± 0.00057 | +7.15 | |

Readings.
- **Target form** [M]: the students' own listwise target with the click mixed in (S4, the spec arm) is the worst S arm: 0.38021 / 0.30501, at the level of the hard-label tree. With 147 candidates per feed and 1.8 clicks on average (one click in 63% of the feeds), `0.5 y / sum(y)` puts half of the label mass on one or two rows and the tree fits those rows (the 15,000-feed pilot over-fitted after 100 trees: val 0.37513 -> 0.37207; at full scale the level is flat from 100 trees on: 0.37955 -> 0.37994). The same loss on the pure soft label `softmax(teacher)` (S4b) is the best S arm per tree and per CPU minute.
- **Learning rate / rounds** [M]: RMSE at lr 0.3 reaches in 55 minutes what lr 0.1 reaches in 135.
- **Raw embeddings (S3)** [M]: the 128-column block (32 PCA components of the candidate vector and their products with the user, mean-history and last-5-history vectors) takes 12.9% of the model's importance, raises rho to W16 from 0.897 to 0.910, does not raise NDCG and lowers the NATS from 0.00062 / 0.00036 to 0.00030 / 0.00023. The audit's small joint fit (+0.0012 ± 0.0008 val / +0.0010 ± 0.0005 holdout) and the 15,000-feed pilot here (+0.0003 ± 0.0008 / +0.0011 ± 0.0005) do not survive at 134K feeds.
- **Tree shape (S2)** [M]: symmetric depth 10 is slightly better per tree on the holdout, not on val, and costs 2.4x per tree; Depthwise (pilot only) was worse.
- **Against the LightGBM students** [M, their files read with the same tools]: S1 - LightGBM K (134K feeds, 600 rounds, 0.38484 / 0.30761) = -0.00030 ± 0.00070 val / +0.00001 ± 0.00040 holdout: the same model. S1 - LightGBM KL0 (60K feeds, 2,400 rounds = its plateau, 0.38613 / 0.30963) = -0.00160 ± 0.00078 / -0.00201 ± 0.00046 (pooled z -4.6): CatBoost at this CPU budget is below the LightGBM plateau; CatBoost is not a better student than LightGBM on these 177 features. LightGBM students' NATS: +0.00056 ± 0.00026 / +0.00038 ± 0.00014 and +0.00054 ± 0.00026 / +0.00045 ± 0.00016: the same as the CatBoost students'.

### 2.3 Pilots that shaped the arms (15,000 feeds = 13,463 fit feeds, lr 0.3, 300 trees, 4 threads; NOT arms) [M]

Paired at 300 trees against RMSE depth 8 on the 177 features of the same pilot:

| pilot | val | holdout | pooled z | cost per tree | consequence |
|---|---|---|---|---|---|
| pca block (128 columns) | +0.00032 ± 0.00084 | +0.00114 ± 0.00051 | +1.84 | 1.55x | taken to full scale as S3 |
| cand block (128 raw dims) | +0.00051 ± 0.00086 | +0.00089 ± 0.00052 | +1.64 | 1.72x | same as pca within noise; not run at full scale |
| raw block (512 columns) | +0.00064 ± 0.00085 | +0.00114 ± 0.00052 | +2.08 | 3.1x | same as pca within noise (raw - cand +0.00014 ± 0.00082 / +0.00024 ± 0.00050); not run at full scale |
| Depthwise depth 8, min_data_in_leaf 100 | -0.00035 ± 0.00098 | -0.00079 ± 0.00058 | -1.22 | 1.6x | dropped |
| symmetric depth 10 | +0.00061 ± 0.00079 | +0.00162 ± 0.00048 | +2.95 | 3.3x | S2 = symmetric depth 10 |
| QuerySoftMax, KD-mix label | -0.00851 ± 0.00126 | -0.00542 ± 0.00069 | -10.3 | 1.7x | S4 kept as the spec arm |
| QuerySoftMax, pure soft label | +0.00096 ± 0.00091 | +0.00175 ± 0.00052 | +3.15 | 2.0x | added as S4b |
| RMSE with rsm 0.5 | +0.00037 ± 0.00077 | +0.00018 ± 0.00046 | | 1.2x | dropped |

The pilots were read on val and the holdout (both, paired) and decided which variants went to full scale; that is a selection among variants on the two witnesses, at a 9x smaller scale and with fresh fits afterwards. The full-scale reads of S3 and S2 came out smaller than their pilots; S4b's came out larger.

## 4. Student against hard-label tree: the same or different pool-independent information? [M]

Conditional logit of the label on per-feed z scores (the nats tool's loss); S = the student, H = the hard-label reference `ts_lr020_s0` (YetiRank on clicks, 509 trees, 0.38150 / 0.30462). "S | W16,H" = gain of adding S to [W16, H].

| student S | witness | W16 + S | W16 + H | W16 + S + H | W16 + bundle(S, H) | S given [W16, H] | H given [W16, S] | corr(S, H) | partial corr given W16 |
|---|---|---|---|---|---|---|---|---|---|
| S1f (best arm, seed 0) | val | +0.00062 ± 0.00028 | +0.00117 ± 0.00038 | +0.00119 ± 0.00039 | +0.00094 ± 0.00034 | +0.00002 ± 0.00005 | +0.00058 ± 0.00027 | 0.945 | 0.791 |
| | holdout | +0.00036 ± 0.00014 | +0.00051 ± 0.00017 | +0.00051 ± 0.00017 | +0.00046 ± 0.00016 | +0.00000 ± 0.00001 | +0.00016 ± 0.00009 | 0.947 | 0.796 |
| S1 | val | +0.00057 ± 0.00027 | +0.00117 ± 0.00038 | +0.00124 ± 0.00039 | +0.00091 ± 0.00034 | +0.00007 ± 0.00009 | +0.00067 ± 0.00029 | 0.952 | 0.817 |
| | holdout | +0.00036 ± 0.00014 | +0.00051 ± 0.00017 | +0.00051 ± 0.00017 | +0.00046 ± 0.00016 | +0.00000 ± 0.00000 | +0.00015 ± 0.00009 | 0.953 | 0.821 |
| S3 | val | +0.00030 ± 0.00019 | +0.00117 ± 0.00038 | +0.00135 ± 0.00041 | +0.00075 ± 0.00031 | +0.00018 ± 0.00015 | +0.00105 ± 0.00036 | 0.934 | 0.743 |
| | holdout | +0.00023 ± 0.00011 | +0.00051 ± 0.00017 | +0.00053 ± 0.00017 | +0.00039 ± 0.00015 | +0.00001 ± 0.00003 | +0.00030 ± 0.00013 | 0.936 | 0.751 |
| S2 | val | +0.00039 ± 0.00022 | +0.00117 ± 0.00038 | +0.00141 ± 0.00042 | +0.00078 ± 0.00031 | +0.00024 ± 0.00017 | +0.00102 ± 0.00035 | 0.951 | 0.812 |
| | holdout | +0.00035 ± 0.00014 | +0.00051 ± 0.00017 | +0.00051 ± 0.00017 | +0.00045 ± 0.00016 | +0.00000 ± 0.00000 | +0.00016 ± 0.00009 | 0.952 | 0.816 |
| S4 | val | +0.00074 ± 0.00030 | +0.00117 ± 0.00038 | +0.00118 ± 0.00038 | +0.00097 ± 0.00035 | +0.00001 ± 0.00004 | +0.00044 ± 0.00024 | 0.973 | 0.901 |
| | holdout | +0.00048 ± 0.00016 | +0.00051 ± 0.00017 | +0.00053 ± 0.00017 | +0.00052 ± 0.00017 | +0.00002 ± 0.00003 | +0.00005 ± 0.00005 | 0.974 | 0.905 |
| S4b | val | +0.00071 ± 0.00030 | +0.00117 ± 0.00038 | +0.00119 ± 0.00038 | +0.00098 ± 0.00035 | +0.00002 ± 0.00005 | +0.00048 ± 0.00024 | 0.963 | 0.862 |
| | holdout | +0.00043 ± 0.00015 | +0.00051 ± 0.00017 | +0.00052 ± 0.00017 | +0.00049 ± 0.00016 | +0.00001 ± 0.00002 | +0.00009 ± 0.00007 | 0.964 | 0.867 |

The declared read with the nats tool itself (bundle form, member-equal z-mean of S1f seed 0 and `ts_lr020_s0`): level 0.38457 / 0.30767, NATS +0.00098 ± 0.00035 / +0.00048 ± 0.00016, not promising; member rule -0.00048 ± 0.00039 / +0.00005 ± 0.00021, pooled z -0.69, no pass.

Readings.
- **Adding BOTH is not better than the hard-label tree alone**: W16 + S + H = W16 + H within 0.0000 to 0.0002 nats on both witnesses for every student; the equal-weight bundle of the two is not better than the hard-label tree alone, and on val it is worse (val 0.00075-0.00098 against 0.00117; holdout 0.00039-0.00052 against 0.00051), because it dilutes H with a score that carries nothing beyond W16 and H.
- **The student's pool-independent information is a subset of the hard-label tree's**: S given [W16, H] is zero (largest value +0.00024 ± 0.00017); H given [W16, S] stays positive (+0.0004 to +0.0011 val; +0.0001 to +0.0003 holdout, S4: +0.00005 ± 0.00005). In the 3-covariate fit the student's coefficient is negative or not distinguishable from zero (z between -2.8 and +1.1) while the hard-label tree keeps z +1.9 to +5.8.
- What the two trees share beyond W16 is large (partial correlation 0.74 to 0.91): it is the common "177-feature tree view" of the data, and that view is what carries the 0.0005-0.0012 nats. A student keeps the part of it that survives fitting to the teacher (a quarter to two thirds of the hard-label tree's NATS on val, 45% to 95% on the holdout); the teacher target adds level (+0.003) and no new information.
- Consequence for the lead [I]: for pool-independent information take a hard-label tree; a distilled tree is a cheaper, weaker copy of the pool. A bundle of a student and a hard-label tree is not a route to the screen.

## 5. M bins and the lead's notice of 02:19Z (freshness) [M]

No S arm builds a label-aware or count statistic: the inputs are the 177 tabular columns of record (with their known freshness: fresh on in-period rows, stale on val / test) and, for S3, label-free embedding columns. The notice therefore applies through the READ only: val M >= 50 is reported separately, and holdout-only gains are flagged.

| arm | val M < 50 (5,893 feeds): gap to W16 | NATS | member rule 0.9 / 0.1 | val M >= 50 (8,530 feeds): level | gap to W16 | NATS | member rule 0.9 / 0.1 | holdout (30,000, all M >= 50): gap |
|---|---|---|---|---|---|---|---|---|
| hard-label reference `ts_lr020_s0` | 0.02263 ± 0.00262 | +0.00156 ± 0.00073 | -0.00043 ± 0.00072 | 0.30253 | 0.01807 ± 0.00163 | +0.00090 ± 0.00041 | +0.00020 ± 0.00044 | 0.01998 ± 0.00087 |
| S1 | 0.01914 ± 0.00250 | +0.00101 ± 0.00058 | -0.00076 ± 0.00070 | 0.30524 | 0.01536 ± 0.00152 | +0.00032 ± 0.00025 | -0.00030 ± 0.00040 | 0.01698 ± 0.00082 |
| S1f (best arm, seed 0) | 0.02130 ± 0.00250 | +0.00090 ± 0.00055 | -0.00082 ± 0.00071 | 0.30590 | 0.01470 ± 0.00151 | +0.00042 ± 0.00028 | +0.00004 ± 0.00042 | 0.01723 ± 0.00082 |
| S3 | 0.02141 ± 0.00247 | +0.00081 ± 0.00052 | -0.00127 ± 0.00069 | 0.30511 | 0.01549 ± 0.00151 | +0.00009 ± 0.00013 | +0.00010 ± 0.00042 | 0.01734 ± 0.00082 |
| S2 | 0.02195 ± 0.00255 | +0.00093 ± 0.00056 | -0.00085 ± 0.00071 | 0.30332 | 0.01728 ± 0.00155 | +0.00014 ± 0.00016 | +0.00006 ± 0.00043 | 0.01829 ± 0.00085 |
| S4 | 0.02630 ± 0.00276 | +0.00100 ± 0.00058 | -0.00078 ± 0.00079 | 0.30288 | 0.01772 ± 0.00158 | +0.00051 ± 0.00031 | +0.00021 ± 0.00043 | 0.01958 ± 0.00086 |
| S4b | 0.01993 ± 0.00247 | +0.00143 ± 0.00070 | -0.00084 ± 0.00071 | 0.30584 | 0.01476 ± 0.00151 | +0.00032 ± 0.00025 | +0.00006 ± 0.00042 | 0.01599 ± 0.00081 |

(The NATS of a bin is refitted on the feeds of that bin, same convention as the tool. W16 on val: M < 50 0.51844, M >= 50 0.32060.)

- On val M >= 50, the population closest to the holdout, the students' NATS is +0.0001 to +0.0005 (hard-label tree +0.0009) and the member-rule statistic is zero (-0.0003 to +0.0002, SE 0.0004). Nothing hides in a bin.
- The gap to W16 is larger on short feeds (0.019-0.026) than on long ones (0.015-0.018) in absolute terms and smaller in relative terms (3.7-5.1% of the W16 level against 4.6-5.6%).
- Holdout-only gains, to be read with the notice in mind [I]: S4b over S1f (+0.0005 ± 0.0008 val, +0.0012 ± 0.0005 holdout; on val M >= 50 the two are equal, 0.30584 against 0.30590), depth 10 over depth 8 per tree (+0.0001 val, +0.0010 holdout), the pca block in the pilot (+0.0003 val, +0.0011 holdout). In each case the val part is within one SE of zero. A model that copies the in-sample, in-period teacher more closely is rewarded on in-period rows; none of these is claimed as a gain for test.
- S3 block columns, label-free distribution check on all rows (`reads/dist_pca.json`): standardised mean difference to the xm training rows at most 0.017 on the holdout, 0.072 on val, 0.077 on test (median 0.006 / 0.014 / 0.018); sd ratio 0.93-1.08; two-sample KS at most 0.012 / 0.038 / 0.039. The largest shifts are the first principal components of the candidate vector (`pe0`, `pe2`: the candidate mix drifts with time). No train / apply mismatch of a size that matters.

## 6. Method [M unless marked]

- Data: cbx `xm` = 134,118 exploration feeds, 19,741,607 rows; `cbx.trainside_split`: 120,591 fit feeds (17,758,444 rows) and 13,527 selection feeds (1,983,163 rows), user-grouped. The 30,000 holdout feeds are in no fit.
- Every arm goes through `cbx.run_arm` (`run_s.py`): fit on the fit feeds, eval set = the selection feeds with the CLICK label and `NDCG:type=Base`, no early stop (`od_wait` = iterations), prediction at the best selection iteration, then the standard evaluation. No val or holdout label is read before that evaluation. Thread cap 12 (`cbx.THREADS`) in the constructor, pools and predict.
- Common parameters: 177 features with `category_id` as a native categorical (as the record), `border_count` 254, `l2_leaf_reg` 5, symmetric trees, depth 8 unless stated, seed 0, bootstrap MVS with subsample 0.8 (CatBoost default, resolved in the logs).
- Targets: S1 / S1f / S2 / S3: `RMSE` on `tr.teacher("mean11", "centred")` (per-feed mean 0, sd 1.207). S4: `QuerySoftMax` with `leaf_estimation_method="Newton"` on `0.5 y / sum(y) + 0.5 softmax(teacher_mean11)` (non-negative, sum 1 per feed; every xm feed has a click). S4b: the same loss on `softmax(teacher_mean11)` alone.
- S3 block (`fs_common.py`): PCA basis = 32 leading principal directions of the candidate vectors, fitted without labels on a 3M-row sample of the fit feeds (53.5% of the variance); columns = the 32 projections of the candidate vector and, for each of the user, mean-history and last-5-history vectors (unit-normalised, per feed), the 32 element-wise products in that basis: 128 float16 columns.
- Wall-time caps (the fit stops cleanly and the best selection iteration so far is used): S1 135 min, S1f 55, S3 75, S4 35, S4b 55, S2 45. Cost per tree at 12 threads on the loaded machine: RMSE depth 8 3.7-4.1 s, with the pca block 6.2 s, depth 10 8.8 s, QuerySoftMax Newton 6.65 s, YetiRank on the real-valued teacher logit about 24 s [I from the dry run: 6.5x RMSE].
- Reads: `fs_read.py` (wrapper of `nats.py`: `nats`, `bundle`; `cmp` = paired per-feed NDCG difference; `joint` = the conditional logits of section 4; `mbin`), `fs_m50.py` (nats and member rule per M bin of val), `fs_staged.py` (predictions at fixed tree counts), `fs_curve.py` (selection curve). `fs_read.py nats ts_lr020_s0` reproduces the reference line (+0.00117 ± 0.00038 / +0.00051 ± 0.00017).
- rho: the nats tool prints the global Pearson of the per-feed z scores (S1f 0.897 / 0.897); `cbx.Evaluator` prints the mean within-feed Pearson (S1f 0.889 / 0.896). Both are given where it matters.
- Best arm for the seeds, written before any arm (`PREDECLARED_famS.txt`): highest train-side selection NDCG; arms within 0.0005 of the top are tied and the cheapest is taken; arms whose two extra seeds do not fit the budget are not eligible. Selection NDCG: S3 0.31200 | S1f 0.31165 | S1 0.31133 (not eligible, 135 min per seed) | S4b 0.31110 | S2 0.30989 | S4 0.30731. Top S3, tie band {S3, S1f}, cheaper = **S1f**. The val / holdout level of S4b (the highest) was known when the rule was applied and was not used.

## 7. Leak statement, one line per arm

- All arms: no val / test / holdout label is an input, a fitting target or a selection signal; tree count by the train-side selection split (click NDCG of 13,527 train feeds); the holdout feeds are in no fit. `teacher_mean11` is used as a TARGET only (it is in-sample on the fit feeds, which is allowed for a target and is the students' own setting); no NN-derived column is an input. The 177 inputs are the record's, unchanged, with the record's own target statistics (`tte_*`) and their known freshness; `category_id` is a candidate-level native categorical exactly as in the record and in the reference B0 (not a feed- or user-constant key, so the CPU within-feed prefix leak of docs_feat 5.3 does not apply).
- S1, S1f, S2 (RMSE on the teacher logit; depth 8 / 8 / 10): clean by the line above; nothing else enters.
- S3 (+ pca block): the block is label-free: PCA fitted without labels on fit-feed candidate vectors only; products with the feed's own user / history vectors as supplied by the competition data (history = events before the feed). Distributions on train, holdout, val and test rows agree (section 5).
- S4 (QuerySoftMax, KD-mix label): the click enters as the training LABEL of the row's own feed (fit feeds only), never as an input; otherwise as above.
- S4b (QuerySoftMax, soft label): teacher target only; as above.
- Seeds of S1f and their bundle: same as S1f; the bundle is the member-equal mean of per-feed z, declared before any arm.
- Arm selection: pilots (section 2.3) were read on val and holdout and decided which variants were run at full scale (disclosed); the best arm for the seeds was chosen on the train-side selection split.
