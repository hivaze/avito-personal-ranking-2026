# NEXT_1115 / merge (r2): merge compatibility check v2 and private-day drift table, corrected and extended

Written 2026-11-15 01:45 UTC. Second pass of this study. The first pass (2026-11-14 22:46 UTC, kept unchanged in `nn_x/w12/next/merge/r1/`, report `r1/merge_r1.md`) was reviewed in `reports/NEXT_1115/merge_verify.md`; this pass repairs what that review found, re-derives every table with the repaired code and adds the CatBoost reading the owner asked for.
CPU only, 12 threads, one process at a time, no GPU, nothing written outside `nn_x/w12/next/merge/` and this file. [M] = measured by me in this pass, [I] = inferred. ± = 1 SE paired per feed unless stated; feeds: val 14,423, test 28,595 (public 8,562, private 20,033).

## 0. Verdict

1. **0.9908 vs 0.9938 is the correlation definition only** (same two files, both on test) [M]: 0.9908 = pooled Spearman of two rank files (v1); 0.9938 = pooled Pearson of per-feed z of the model scores (lens, refit 0.0103 / 0.304); the law on record (0.0186 / 0.378) was fitted with the feed-mean Pearson on val (0.9935). The tool computes that quantity and **reproduces both known answers**: W8 + C3 law -0.00012, measured -0.00027 ± 0.00039; W8 + CatBoost law -0.00440, measured -0.00458 ± 0.00096 (14,423 val feeds).
2. **The three defects found by the verifier are repaired** [M]: the equal mix with a rank file has an explicit tie-break (9.4% tied rows for C3; the read no longer depends on the scale of our scores), a partly corrupted partner file is refused (10% of feeds inverted: was "PROCEED", now FAIL), `--blend` writes only after gates 1-3 or with `--force`. 41 of 41 synthetic checks pass.
3. **Error of each law on the 10 on-disk partners** [M]: on record rms 0.00035 (max 0.00098), refit rms 0.00023 (max 0.00043, in-sample), all inside the paired SE of the measurements. **Where a merge would use it the law is not identified**: equal-mix gain at rho 0.90 is [-0.0013, +0.0034] at gap 0 and [-0.0016, +0.0025] at gap 0.002 (5-95%); no on-disk pair has rho < 0.95 and gap < 0.005.
4. **A rank file costs less than the first pass said** [M]: with ties given to the better file the equal mix reads -0.0001 against the z-mix of the model scores (10 partners; the -0.0011 of the first pass was the tie-break). A cost appears when the rank file carries the larger weight (w 0.6: -0.0013 on average, to -0.0044). Asking for model scores is still right.
5. **Drift: no candidate drifts on the late private days beyond what val shows, except W8TO** (known cold promotion of the fine-tune half: -0.0094 on val Tuesday to -0.0210 on Tue 09-29) [M]. New same-recipe A/A null: the feed-level SE of the cold-offset contrasts is 1.25-1.65 times too small, so the 3 SE flag is about 2 null sd; no reading changes.
6. **CatBoost (owner's question)** [M]: a tree arm is priced in one command (IN-HOUSE mode). CatBoost of record: law -0.0044, measured -0.00458 ± 0.00096 against W8 and -0.00470 ± 0.00097 against W16. At its rho 0.84 an equal mix breaks even only at val 0.3937 (on record) to 0.3962 (refit), +0.012 to +0.014 above the model. The law prices equal mixes only; it says nothing about the 0.1-weight member rule of the CatBoost track.

## 1. Measurements

### 1.1 Why 0.9908 and 0.9938 differ (W8 file vs C3 file, test, 28,595 feeds) [M]

Both files are the rank transform of 0.5 z(FT pool) + 0.5 z(train-only pool) of `preds/ens_<TAG>_*` (the tool checks that the stored scores reproduce at least 99.99% of the ranks of the file). So the lens and the lead's run compared the same two objects on the same split.

| definition | all | M >= 50 | M < 50 | who uses it |
|---|---|---|---|---|
| pooled (M-weighted) Spearman = `np.corrcoef` of z of the file ranks | 0.9908 | 0.9914 | 0.9850 | `src/merge_check.py` (v1) |
| feed-mean Spearman | 0.9873 | 0.9910 | 0.9814 | - |
| pooled Pearson of per-feed z of the blended model scores | 0.9938 | 0.9938 | 0.9934 | lens `p4_partner.py` (`wcorr`); the refit 0.0103 / 0.304 (val: 0.9938) |
| feed-mean Pearson of per-feed z of the model scores | 0.9935 | 0.9938 | 0.9929 | **the law on record**: `solvers/ensemble/exp7_pairs.py` `pf_corr` = `(bincount(fid, a*b) / M).mean()` on val; `solvers/out/ensemble/exp7.json` "val_fit_gap<0.01" = 0.018577, 0.37766 (val: 0.9935) |
| v2 from two rank files (ranks restored on the shape of our scores), feed-mean / pooled | 0.9948 / 0.9943 | 0.9943 | 0.9955 | `merge_check_v2.py` when the partner sends ranks |

- [I] Spearman is lower because ranks weight every list position equally, while z of the scores (within-feed skewness -0.40) weights the candidates far from the bulk, where two models agree best.
- Rank-only estimates of the test value the law needs, 10 partners: shape restoration +0.0014 on average (max 0.0020) for the 7 neural pools and +0.0009 (max 0.0015) for the 3 non-neural models, at most 0.00004 in the law; feed-mean Spearman -0.0069 (max 0.0100) and -0.0272 (max 0.0329). The pooled Spearman that v1 feeds into the law on record is 0.0027 (C3) to 0.0238 (XGBoost) below the law's rho (CatBoost 0.0126): up to +0.0004 in the law, towards blending.
- The same file `exp7.json` holds two other fits of the same form: all 36 pairs 0.0130 / 0.291, holdout 0.0615 / 0.620. The coefficient of diversity on record already varies by a factor of 4.7 with the subset.

### 1.2 Calibration: the 10 on-disk partners of W8 (val, 14,423 feeds; gain of the equal mix over the better file) [M]

W8 final val 0.40165. rho = val feed-mean Pearson (definition on record) / val pooled Pearson (refit) / the value the tool gets from two TEST rank files. Errors are law minus measured. Last column: what the tool prints when the partner sends a RANK file (equal mix, ties to the better file), against the better file.

| partner | val | gap ± SE | rho rec / refit / tool | law on record | refit | on record, tool rho | measured z-mix of model scores | error on record | error refit | rank-file mix (tool r2) |
|---|---|---|---|---|---|---|---|---|---|---|
| C3 = pool Q2a8s3 (final) | 0.40101 | +0.00064 ± 0.00055 | 0.9935 / 0.9938 / 0.9948 | -0.00012 | -0.00013 | -0.00014 | -0.00027 ± 0.00039 | +0.00015 | +0.00014 | +0.00013 ± 0.00024 |
| pool G3 (final) | 0.40032 | +0.00133 ± 0.00058 | 0.9931 / 0.9935 / 0.9944 | -0.00037 | -0.00034 | -0.00040 | -0.00044 ± 0.00040 | +0.00006 | +0.00010 | -0.00046 ± 0.00025 |
| pool Pall (final) | 0.39896 | +0.00269 ± 0.00063 | 0.9911 / 0.9917 / 0.9925 | -0.00085 | -0.00073 | -0.00088 | -0.00095 ± 0.00044 | +0.00010 | +0.00022 | -0.00055 ± 0.00030 |
| pool Cd (final) | 0.39901 | +0.00264 ± 0.00065 | 0.9903 / 0.9909 / 0.9918 | -0.00082 | -0.00071 | -0.00084 | -0.00061 ± 0.00044 | -0.00021 | -0.00010 | -0.00012 ± 0.00032 |
| pool S16kd (final; better than W8) | 0.40173 | -0.00008 ± 0.00034 | 0.9989 / 0.9989 / 0.9993 | -0.00001 | -0.00001 | -0.00002 | +0.00005 ± 0.00022 | -0.00006 | -0.00007 | -0.00014 ± 0.00011 |
| F10 GRU x6 (TO, hard labels) | 0.39848 | +0.00317 ± 0.00073 | 0.9828 / 0.9836 / 0.9848 | -0.00088 | -0.00080 | -0.00092 | -0.00068 ± 0.00051 | -0.00020 | -0.00011 | -0.00051 ± 0.00039 |
| base x4 (TO, hard labels) | 0.39887 | +0.00278 ± 0.00067 | 0.9879 / 0.9883 / 0.9897 | -0.00083 | -0.00072 | -0.00086 | -0.00115 ± 0.00045 | +0.00032 | +0.00043 | -0.00055 ± 0.00032 |
| **GBDT CatBoost YetiRank xm** | 0.38201 | +0.01964 ± 0.00144 | 0.8397 / 0.8422 / 0.8384 | -0.00444 | -0.00434 | -0.00442 | -0.00458 ± 0.00096 | +0.00014 | +0.00024 | -0.00494 ± 0.00090 |
| GBDT xgb_tr1m_ndcg | 0.36770 | +0.03395 ± 0.00167 | 0.7730 / 0.7749 / 0.7681 | -0.00861 | -0.00800 | -0.00852 | -0.00763 ± 0.00111 | -0.00098 | -0.00037 | -0.00820 ± 0.00109 |
| LLM Qwen3-0.6B x2 | 0.37676 | +0.02489 ± 0.00145 | 0.8404 / 0.8464 / 0.8398 | -0.00644 | -0.00598 | -0.00643 | -0.00624 ± 0.00093 | -0.00020 | +0.00026 | -0.00772 ± 0.00091 |

| error of each law (law - measured) | all 10: mean / rms / max | 7 neural pools: rms / max | 3 non-neural: rms / max |
|---|---|---|---|
| law on record, its own rho (val feed-mean Pearson) | -0.00009 / 0.00035 / 0.00098 | 0.00018 / 0.00032 | 0.00059 / 0.00098 |
| law on record as the lens evaluated it (val pooled Pearson) | -0.00011 / 0.00037 / 0.00102 | 0.00018 / 0.00032 | 0.00062 / 0.00102 |
| refit 0.0103 / 0.304 (in-sample: fitted on these 10) | +0.00007 / 0.00023 / 0.00043 | 0.00020 / 0.00043 | 0.00030 / 0.00037 |
| law on record with the tool's rho from two test rank files | -0.00009 / 0.00033 / 0.00089 | 0.00018 / 0.00029 | 0.00054 / 0.00089 |

- Tool re-derivations of the two known answers (`out/known_*.txt`, IN-HOUSE mode: the gap is the val gap, as in the fit): W8 + C3 score files: rho 0.9935 / 0.9938, law on record -0.00012 (lens -0.00013), refit -0.00013, equal mix -0.00027 ± 0.00039 (z -0.70). W8 + CatBoost (`nn_x/cb26/data/e3_xm_yeti_{val,test}.npy` read directly): test rho 0.8375 / 0.8398, law on record -0.00440 (lens -0.0045; with the val rho 0.8397: -0.00444), refit -0.00432, equal mix -0.00458 ± 0.00096 (z -4.76). With the real public scores the C3 file reads -0.00029 (0.394486 / 0.393454, gap 0.001032, rank files).
- The largest error is XGBoost, 0.88 paired SE. Neither law is rejected by these 10 partners; the refit is fitted on them, so its smaller error is in-sample (leave-one-partner-out rms 0.00049-0.00052 for either definition: 0.0002 for the neural pools, 0.0008-0.0009 for the non-neural models).
- Refit under each definition: val pooled Pearson 0.0103 / 0.304 (reproduces the lens); val feed-mean 0.0089 / 0.296; test feed-mean 0.0096 / 0.302; test shape-restored 0.0105 / 0.307. Leave-one-out range of a [0.0040, 0.0378], of b [0.256, 0.483].

Per-test-day output of the tool (item 2), two partners, law-rho feed-mean ± SE over feeds [M]:

| partner | Sat (public) | Sun am (public) | Sun pm (private) | Mon (private) | Tue (private) | PUBLIC | PRIVATE | private - public | M < 50 | M >= 50 | same top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C3 (scores) | 0.9935 ± 0.0001 | 0.9934 ± 0.0001 | 0.9935 ± 0.0001 | 0.9933 ± 0.0001 | 0.9936 ± 0.0001 | 0.9935 | 0.9935 | -0.0000 ± 0.0001 | 0.9929 | 0.9938 | 82.5% |
| CatBoost (scores) | 0.8363 ± 0.0012 | 0.8372 ± 0.0019 | 0.8398 ± 0.0013 | 0.8362 ± 0.0011 | 0.8382 ± 0.0011 | 0.8366 | 0.8378 | +0.0012 ± 0.0012 | 0.8337 | 0.8398 | 36.9% |

Feeds per group: 6,050 / 2,512 / 4,855 / 7,578 / 7,600; M < 50 11,161, M >= 50 17,434. Neither partner agrees less with W8 on the private days.

### 1.3 Uncertainty of the law (item 4) [M]

Design of the 10 partners: corr(1 - rho, |gap|) = 0.992; 7 partners at rho >= 0.98 with gap <= 0.0032, 3 at rho 0.77-0.84 with gap 0.020-0.034; none with rho < 0.95 and gap < 0.005. Feed bootstrap, 2,000 resamples of the val feeds, all pairs resampled together. "Line" = 5-95% of the fitted line; "PI" = line plus the leave-one-out residual of that fit (0.0005).

| target (rho, gap) | law on record | 10 partners of W8: point, line, PI | 55 pairs of the 11 models: point, line | 52 pairs (without the 3 pairs of two non-neural models): point, line, PI |
|---|---|---|---|---|
| 0.97, 0 | +0.00056 | +0.00027, [-0.0004, +0.0010], [-0.0008, +0.0015] | +0.00057, [+0.0002, +0.0011] | +0.00042, [-0.0000, +0.0010], [-0.0005, +0.0015] |
| 0.95, 0 | +0.00093 | +0.00045, [-0.0006, +0.0017], [-0.0009, +0.0020] | +0.00096, [+0.0003, +0.0018] | +0.00070, [-0.0001, +0.0017], [-0.0004, +0.0021] |
| 0.95, 0.002 | +0.00017 | -0.00014, [-0.0009, +0.0008], [-0.0013, +0.0011] | +0.00023, [-0.0002, +0.0008] | +0.00004, [-0.0005, +0.0008], [-0.0010, +0.0012] |
| **0.90, 0** | +0.00186 | +0.00089, [-0.0013, +0.0034], [-0.0014, +0.0036] | +0.00192, [+0.0007, +0.0035] | +0.00140, [-0.0001, +0.0034], [-0.0003, +0.0036] |
| **0.90, 0.002** | +0.00110 | +0.00030, [-0.0016, +0.0025], [-0.0017, +0.0027] | +0.00119, [+0.0001, +0.0026] | +0.00074, [-0.0006, +0.0024], [-0.0008, +0.0027] |
| 0.90, 0.005 | -0.00003 | -0.00059, [-0.0020, +0.0011], [-0.0022, +0.0014] | +0.00009, [-0.0008, +0.0012] | -0.00025, [-0.0013, +0.0010], [-0.0016, +0.0014] |
| 0.86, 0 | +0.00260 | +0.00125, [-0.0018, +0.0048], [-0.0018, +0.0050] | +0.00268, [+0.0009, +0.0049] | +0.00196, [-0.0002, +0.0047], [-0.0003, +0.0048] |
| 0.86, 0.002 | +0.00185 | +0.00066, [-0.0021, +0.0039], [-0.0022, +0.0040] | +0.00195, [+0.0003, +0.0040] | +0.00130, [-0.0006, +0.0038], [-0.0008, +0.0039] |

- Coefficients: 10 partners a 0.0089 [-0.013, +0.034], b 0.296 [0.15, 0.47]; 55 pairs a 0.0192 [+0.007, +0.035], b 0.364 (the law on record recovered on other pairs); 52 pairs a 0.0140 [-0.001, +0.034], b 0.330; the 28 pairs among the 8 neural pools alone a 0.0498, b 0.541 (all at rho >= 0.98, an extrapolation). **The diversity coefficient is known to a factor of about 6**; the positive lower bound of the 55-pair fit rests on 3 pairs of two weak models (level 0.37-0.38), as the verifier found.
- **Interval I can defend** for a strong, different partner at rho 0.90: equal-mix gain between **-0.0014 and +0.0036 at gap 0** and between **-0.0017 and +0.0027 at gap 0.002** (W8-level evidence, 5-95%, with residual). Share of that interval above the 0.0005 margin of the rule: 0.68 and 0.53 (0.83 and 0.66 on the 52-pair fit); a rough share of a bootstrap, not a posterior.
- Not in the interval: (a) the public gap itself is noisy: paired SE about 0.0017 at rho 0.90 on 8,562 feeds [I, interpolated from the val SEs of 1.2], ± 0.0006 in the gap term, and |gap| is biased upwards; (b) every fit is on val gaps, the transfer to public and private was never measured; (c) linearity in 1 - rho between the two clusters is an assumption.
- The only on-disk pairs with rho < 0.95 and gap < 0.01 are two pairs of weak models: CatBoost + LLM (rho 0.930, gap 0.0053) measured -0.00067 ± 0.00066, law -0.00068; XGBoost + LLM (rho 0.897, gap 0.0091) measured +0.00001 ± 0.00088, law -0.00151 (1.7 SE too pessimistic).
- Gate 3 ("the blend at the law's weight gains >= 2 paired SE on val") has little power [I]: the paired SE of a mix against the better file is 0.0004 at rho 0.99 and 0.0010 at rho 0.84, so at rho 0.90 the bar is about +0.0017, above the law's own prediction at gap 0.002. Applied as written; I think it should be a veto ("not below -1 SE"), and say so without changing it.

### 1.4 Mixing a partner RANK file: the tie-break, and what a rank file really costs [M]

When one file is a rank file restored on the shape of the other, two items with swapped ranks get exactly the same equal mix: 7-15% of the val rows for a neural partner, 2-3% for a non-neural one. In the first pass these ties were decided by a float residue in the partner's favour. Now the mix is rounded to 1e-8 and exact ties go to the file with the better public score (in IN-HOUSE mode: better val), then to a fixed random key.
Equal mix of W8 scores and the partner's rank file, minus the z-mix of the model scores at the same weight (paired, 14,423 val feeds; w = weight of the partner; `out/c2_extend.txt` T9):

| partner | val gap | tied rows at w 0.5 | ties to the BETTER file (tool r2) | ties at random | ties to the weaker file (= first pass) | w 0.4 (no ties) | w 0.6 (no ties) |
|---|---|---|---|---|---|---|---|
| C3 | +0.00064 | 9.4% | +0.00040 ± 0.00033 | +0.00038 ± 0.00033 | +0.00006 ± 0.00032 | +0.00027 ± 0.00029 | +0.00034 ± 0.00030 |
| G3 | +0.00133 | 9.2% | -0.00002 ± 0.00035 | +0.00010 ± 0.00033 | -0.00032 ± 0.00032 | +0.00008 ± 0.00031 | -0.00024 ± 0.00030 |
| Pall | +0.00269 | 8.5% | +0.00041 ± 0.00036 | +0.00019 ± 0.00035 | -0.00105 ± 0.00033 | -0.00020 ± 0.00032 | -0.00137 ± 0.00032 |
| Cd | +0.00264 | 8.2% | +0.00049 ± 0.00037 | +0.00010 ± 0.00035 | -0.00106 ± 0.00036 | +0.00045 ± 0.00033 | -0.00089 ± 0.00035 |
| S16kd | -0.00008 | 15.2% | -0.00020 ± 0.00020 | +0.00002 ± 0.00022 | -0.00016 ± 0.00023 | -0.00025 ± 0.00020 | -0.00009 ± 0.00019 |
| GRU x6 | +0.00317 | 6.7% | +0.00017 ± 0.00039 | -0.00013 ± 0.00038 | -0.00098 ± 0.00039 | -0.00022 ± 0.00036 | -0.00105 ± 0.00038 |
| base x4 | +0.00278 | 7.6% | +0.00060 ± 0.00037 | -0.00001 ± 0.00038 | -0.00067 ± 0.00039 | +0.00044 ± 0.00035 | -0.00037 ± 0.00036 |
| CatBoost | +0.01964 | 2.6% | -0.00036 ± 0.00056 | -0.00156 ± 0.00056 | -0.00181 ± 0.00056 | +0.00016 ± 0.00052 | -0.00122 ± 0.00056 |
| XGBoost | +0.03395 | 2.1% | -0.00057 ± 0.00062 | -0.00161 ± 0.00062 | -0.00193 ± 0.00062 | -0.00012 ± 0.00055 | -0.00379 ± 0.00065 |
| LLM x2 | +0.02489 | 2.6% | -0.00148 ± 0.00058 | -0.00198 ± 0.00058 | -0.00305 ± 0.00060 | -0.00063 ± 0.00051 | -0.00442 ± 0.00059 |
| mean of 10 (7 neural / 3 non-neural) | | | -0.00006 (+0.00026 / -0.00080) | -0.00045 (+0.00009 / -0.00172) | -0.00110 (-0.00060 / -0.00226) | -0.00000 (+0.00008 / -0.00020) | -0.00131 (-0.00053 / -0.00314) |

- The -0.0011 "rank-file penalty" of the first pass is the column "ties to the weaker file". With the ties given to the better file the mean is -0.0001 and no partner is beyond 2.6 SE; the tie-free read at w 0.4 says the same (0.0000).
- A real cost exists when the rank file carries the larger weight: w 0.6 reads -0.0013 on average and -0.0038 / -0.0044 for XGBoost / LLM. [I] The restored shape drops the partner's own confidence, which matters in proportion to its weight. In a merge the rank file would be the partner's, at w* <= 0.5 if it is the weaker file and up to 1 - w* if it is the better one: with a partner that is better on public, ask for scores or send ours as the rank side.
- The tie-break to the better file tilts the "equal" mix towards the better file; that is the declared behaviour, printed with the share of tied rows.
- Independence of the scale of our scores: the equal mix with the C3 rank file reads 0.401783 with our raw scores multiplied by 1, 3 and 0.3 (the verifier measured a swing of 0.0015 with the Pall rank file in the first pass); W8TO (unit-variance scores) + C3 rank file at gap 0 reads -0.00019 ± 0.00027 (first pass: -0.00557; z-mix of the model scores +0.00004).

### 1.5 Reader robustness (item 5): 41 of 41 checks pass [M] (`out/t1_reader_tests.txt`)

| synthetic partner file | what the tool does |
|---|---|
| keys as strings / as float64; score as string / as integer; column names in another case plus an extra column | WARN, converted, same result as the clean file |
| 500 feeds missing; 300 rows missing; 200 null keys | FAIL "rows are missing", diagnostics on complete feeds, DECISION: NONE, blend refused |
| 1,000 exact duplicate keys / 1,000 duplicate keys with other scores | WARN, one copy kept / FAIL "CONFLICTING scores" |
| 300 NaN + 50 +inf + 50 -inf; three values of 1e200 in a float64 file (new) | FAIL with the counts; the feeds are left out; DECISION: NONE |
| 400 constant feeds; scores // 10 (99.9% of rows tied) | FAIL (a constant feed scores as the worst order); DECISION: NONE |
| constant everywhere; 0 rows; text file named .parquet; no score column; val keys instead of test keys; .npy of the wrong length (new) | NOT USABLE, no correlation, no law |
| score negated; score = list position (1 = best) | FAIL "INVERTED" (mean per-feed Spearman -0.9873), REFUSED also with `--force`; with `--flip_partner` the baseline is reproduced exactly |
| random scores | FAIL "no usable agreement", REFUSED |
| **10% / 30% of the feeds negated, 12% of the feeds shuffled (verifier's files), 1% negated (new)** | **FAIL "per-feed agreement distribution"** (10.1% / 29.8% / 11.8% / 0.97% of the feeds with M >= 20 below 0.3), DECISION: NONE; the first pass printed PROCEED at law +0.0033 / +0.0104 / +0.0019 |
| 0.3% of the feeds negated (new) | WARN (0.28%), decision unchanged (law moved by +0.0001) |
| model scores multiplied by 1e6 and shifted (float64) | recognised as a score file, rho unchanged (0.9935) |
| `--blend`: file with FAILs / inverted file with `--force` / clean file with gate 2 not met / the same with `--force` / `--w law` at w* = 0 / rank-file partner with `--force` | refused / refused / refused, the missing gates are named / tie-free order of 0.5 z + 0.5 z handed to `make_sub` (stubbed: nothing written) / refused / order with 9.1% tied rows resolved in our order |

Calibration of the new distribution check (test, share of the feeds with M >= 20 whose per-feed Spearman against the W8 order is below 0.3; `c2_extend.txt` T10): the 7 neural pools 0.000%, LLM 0.033%, CatBoost 0.084%, XGBoost (val 0.368, the weakest model on disk) 0.447%. A copy of C3 with 0.2 / 0.5 / 1 / 2 / 5 / 10% of its feeds negated reads 0.20 / 0.61 / 0.95 / 2.0 / 4.8 / 10.0% and lifts the law on record by +0.00007 / +0.00022 / +0.00036 / +0.00073 / +0.0018 / +0.0037 (shuffled feeds: half of that). Settings: FAIL above 0.75%, WARN above 0.2% (mine, fixed from these numbers; the verifier proposed 1%, which a 1% inversion passes at 0.95%). Damage below the WARN line moves the law by less than 0.0001, below the FAIL line by less than 0.0003. `--allow_low_agreement` turns the FAIL into a WARN after the feeds were inspected by hand.

Gates exercised at a non-trivial weight (the verifier noted that w* was 0 for every on-disk partner): CatBoost scores with a pretended public score equal to ours: gates 1 and 2 met (law +0.00302), w* 0.5, gate 3 reads z -4.76, blend refused. With a pretended gap of 0.004: gate 1 not met, law +0.00151, w* 0.375, mix -0.00251 ± 0.00080 (z -3.12). Gate 3 is what protects against a wrong or lucky public gap.

### 1.6 CatBoost: what the merge law says about a tree branch (owner's question) [M unless marked]

| | against W8 | against W16 |
|---|---|---|
| CatBoost of record (`e3_xm_yeti`, val 0.38201): val gap ± SE | 0.01964 ± 0.00144 | 0.01957 ± 0.00144 |
| law-rho, test feed-mean / pooled (val feed-mean) | 0.8375 / 0.8398 (0.8397) | 0.8380 / 0.8403 (0.8404) |
| law on record / refit | -0.00440 / -0.00432 | -0.00438 / -0.00430 |
| measured equal mix on val | -0.00458 ± 0.00096 (M >= 50 -0.00380, M < 50 -0.00572) | -0.00470 ± 0.00097 |
| break-even gap at this rho: law on record / refit | 0.0080 / 0.0054 | 0.0080 / 0.0054 |
| with the 0.0005 margin of the rule | 0.0067 / 0.0038 | 0.0067 / 0.0038 |

| rho of a tree model to W8 | break-even val, law on record | break-even val, refit | 5-95% of the break-even gap, 52-pair fit | 10 partners of W8 |
|---|---|---|---|---|
| 0.84 (CatBoost of record) | 0.3938 (gap 0.0079) | 0.3962 (0.0054) | [-0.0009, +0.0117] | [-0.0130, +0.0119] |
| 0.90 (distilled tree student) | 0.3967 (0.0049) | 0.3983 (0.0034) | [-0.0005, +0.0073] | [-0.0081, +0.0074] |
| 0.93 | 0.3982 (0.0034) | 0.3993 (0.0024) | [-0.0004, +0.0051] | [-0.0057, +0.0052] |
| 0.95 | 0.3992 (0.0025) | 0.4000 (0.0017) | [-0.0003, +0.0036] | [-0.0040, +0.0037] |

A negative lower bound means "no level breaks even" is inside the interval.

- The law prices this model well (error +0.00014 at a paired SE of 0.00096), as it does every tree row of `reports/CATBOOST_1115/audit.md` 1.1. That audit evaluated the law on record with the pooled correlation; the definition it was fitted with (feed-mean) reads 0.002 lower for CatBoost, 0.00005 in the law: nothing changes there.
- As an **equal-weight** partner the CatBoost of record is 0.0116 (law on record) to 0.0142 (refit) of val short of break-even at its own rho, and break-even moves up by about 0.0005 of val for every 0.01 of rho the model gains (table). [I, with the audit's finding that trees fitted on what the pool knows plateau at val 0.3865 and rho 0.91] the equal-mix route is closed for a tree branch; it would need new label information at unchanged rho.
- **The law does not price the member rule of the CatBoost track** (0.9 z(W16 TO) + 0.1 z(X)): it is a law of equal mixes. [I, from `reports/CATBOOST_1115/audit.md` 1.1 and `nn_x/w12/cb/PREDECLARED.txt`] the audit's own attempt to fit a w 0.1 law found it unidentified, and its NATS screen replaces it. My gap-adjusted weight w* is 0 for every tree model on disk, and its model is 0.0003-0.0004 too pessimistic at small weights (1.7), so w* = 0 is not evidence against a 0.1 member.
- What this study gives the CatBoost track: one command per new arm (usage, block C) that prints level, rho in the right definition by test day and feed size, both laws with the break-even gap, and the measured equal mix with paired SE; and the per-feed check, in which CatBoost has 0.08% low-agreement feeds (M < 50: 0.23%), so a tree arm with a few percent of such feeds has a data problem, not diversity.
- Drift table of the CatBoost model against the W16 file (`out/drift_CatBoost_e3_vs_W16.txt`, flat `.npy` input, 28,595 test feeds): per-feed Spearman 0.8153 ± 0.0006 (public 0.8151, private 0.8153), same top-1 item in 37.3% of the feeds, W16's top-3 items move by 6.9 positions on average (M >= 50: 9.8). The tree model ranks cold candidates lower than W16 does: cold-minus-warm offset (rank-z) -0.094 ± 0.002, flat over the test days (public -0.097 ± 0.003, private -0.094 ± 0.002, last day -0.094; chi2 1.9 on 4 df). No drift: the disagreement between the tree and the pool is the same on public and private days.

### 1.7 Gap-adjusted weight (item 3) [M]

Weight of the weaker file w* = clip(0.5 - b |gap| / (4 a (1 - rho)), 0, 0.5) with a, b of the law on record: the maximum of G(w) = 4 w (1 - w) a (1 - rho) - 2 b |gap| w, which equals the law at w = 0.5 and 0 at w = 0 (diversity term = variance reduction of a two-member mean, strength term linear with the law's slope). Examples: rho 0.90, gap 0 gives 0.50; gap 0.002 gives 0.40; every on-disk partner 0.00. Against the lens grid (9 partners weaker than W8): model minus measured -0.00030 (rms 0.00049) at w 0.1, -0.00043 (0.00073) at 0.2, -0.00030 (0.00073) at 0.3, -0.00009 (0.00037) at 0.5: too pessimistic at small weights, so w* is conservative. Not tuned. The val check prints the equal mix and the w* mix, nothing else.

### 1.8 Drift table (part B): candidate vs `subs/cand_W8kd_r2.parquet` [M]

Feeds: val Tue 7,542, val Thu 6,878, Sat 6,050, Sun am 2,512 (public), Sun pm 4,855, Mon 7,578, Tue 7,600 (private). Cold = `it_hist_all_luo == 0`; cold share per feed 0.361 / 0.403 on the val days and 0.427 / 0.435 / 0.447 / 0.456 / 0.471 on the test days. Val rows come from `preds/ens_<TAG>_{ft,to}_val.npy` (no labels). Every number of the first pass reproduces; new are the chi2 column and the A/A row.

Cold-minus-warm offset, candidate minus reference, z of the model scores (`cw_score`; the unit of ft_redesign):

| candidate | val Tue | val Thu | Sat (pub) | Sun am (pub) | Sun pm (priv) | Mon (priv) | Tue (priv) | PUBLIC | PRIVATE | PRIVATE - PUBLIC | last day - val Tue | chi2 over the 5 test-day groups (df 4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W16 | -0.0002 ± 0.0003 | -0.0001 ± 0.0003 | +0.0004 ± 0.0003 | +0.0009 ± 0.0004 | -0.0001 ± 0.0003 | +0.0002 ± 0.0002 | +0.0006 ± 0.0002 | +0.0005 ± 0.0002 | +0.0003 ± 0.0001 | -0.0002 ± 0.0003 (z -0.9) | +0.0007 ± 0.0004 (z +2.0) | 4.8 (p 0.31) |
| W8P20 | +0.0055 ± 0.0002 | +0.0057 ± 0.0002 | +0.0063 ± 0.0002 | +0.0065 ± 0.0003 | +0.0054 ± 0.0002 | +0.0056 ± 0.0002 | +0.0054 ± 0.0002 | +0.0064 ± 0.0002 | +0.0055 ± 0.0001 | -0.0009 ± 0.0002 (z -4.2) | -0.0001 ± 0.0003 (z -0.2) | 18.5 (p 0.001) |
| W8TO | -0.0094 ± 0.0004 | -0.0136 ± 0.0004 | -0.0162 ± 0.0004 | -0.0184 ± 0.0006 | -0.0174 ± 0.0004 | -0.0193 ± 0.0003 | -0.0210 ± 0.0003 | -0.0169 ± 0.0003 | -0.0195 ± 0.0002 | -0.0026 ± 0.0004 (z -7.1) | -0.0116 ± 0.0005 (z -22.0) | 105.6 (p < 0.001) |
| S16 | -0.0007 ± 0.0003 | -0.0014 ± 0.0003 | -0.0016 ± 0.0003 | -0.0011 ± 0.0004 | -0.0023 ± 0.0003 | -0.0016 ± 0.0003 | -0.0011 ± 0.0003 | -0.0015 ± 0.0002 | -0.0016 ± 0.0002 | -0.0001 ± 0.0003 (z -0.4) | -0.0005 ± 0.0004 (z -1.3) | 9.7 (p 0.046) |
| C3 (scale) | -0.0012 ± 0.0007 | -0.0020 ± 0.0007 | -0.0013 ± 0.0007 | +0.0018 ± 0.0010 | -0.0032 ± 0.0008 | -0.0020 ± 0.0006 | -0.0015 ± 0.0006 | -0.0004 ± 0.0006 | -0.0021 ± 0.0004 | -0.0017 ± 0.0007 (z -2.5) | -0.0003 ± 0.0009 (z -0.3) | 15.6 (p 0.004) |
| A/A null: 20 random 8-vs-8 splits of the 16 wide students (train-only pools) | | | | | | | | | | mean +0.0001, sd 0.0007; **sd of z 1.25**, max abs z 2.8 | last - first test day: **sd of z 1.65**, max abs z 3.5 | mean 5.8 (expected 4); 10% of splits p < 0.05; max 14.9 |

Agreement of the two orders (mean per feed):

| candidate | Spearman: val Tue / Thu | Sat / Sun am / Sun pm / Mon / Tue | PUBLIC / PRIVATE (contrast) | same top-1: val Tue / Thu | PUBLIC / PRIVATE (contrast) | top-3 move (positions): val Tue / Thu | PUBLIC / PRIVATE (contrast) |
|---|---|---|---|---|---|---|---|
| W16 | 0.9964 / 0.9963 | 0.9967 / 0.9968 / 0.9965 / 0.9965 / 0.9966 | 0.9967 / 0.9966 (-0.0002 ± 0.0001, z -1.8) | 93.4% / 93.2% | 93.3% / 93.2% (z -0.1) | 0.179 / 0.180 | 0.184 / 0.182 (z -0.4) |
| W8P20 | 0.9975 / 0.9975 | 0.9975 / 0.9976 / 0.9975 / 0.9974 / 0.9975 | 0.9976 / 0.9975 (-0.0001 ± 0.0001, z -0.9) | 93.9% / 93.6% | 93.4% / 93.4% (z +0.2) | 0.176 / 0.178 | 0.179 / 0.177 (z -0.5) |
| W8TO | 0.9942 / 0.9944 | 0.9953 / 0.9953 / 0.9951 / 0.9949 / 0.9949 | 0.9953 / 0.9950 (-0.0003 ± 0.0001, z -2.7) | 89.8% / 89.8% | 90.7% / 90.6% (z -0.4) | 0.286 / 0.300 | 0.287 / 0.276 (z -1.8) |
| S16 | 0.9962 / 0.9962 | 0.9963 / 0.9964 / 0.9962 / 0.9963 / 0.9964 | 0.9964 / 0.9963 (-0.0000 ± 0.0001, z -0.4) | 93.2% / 93.3% | 92.9% / 92.8% (z -0.0) | 0.188 / 0.191 | 0.195 / 0.193 (z -0.5) |
| C3 (scale) | 0.9871 / 0.9870 | 0.9875 / 0.9870 / 0.9873 / 0.9869 / 0.9876 | 0.9874 / 0.9873 (-0.0001 ± 0.0002, z -0.5) | 83.5% / 83.4% | 82.4% / 82.6% (z +0.4) | 0.513 / 0.520 | 0.538 / 0.533 (z -0.6) |
| A/A level: two independent 8-pools of the recipe (train-only) | | 0.9915 on test | PRIVATE - PUBLIC -0.0001, sd of z 0.71 | | 87.4%; sd of z 0.57 | | 0.362; sd of z 0.46 |

Per-feed SE: Spearman 0.0001-0.0005, top-1 0.2-0.8 points, top-3 move 0.002-0.014 (every SE in `out/drift_*.txt`). The rank-z version of the cold offset (`cw_rank`, the only one available for a foreign file) tells the same story: W8TO -0.0106 (val Tue) to -0.0232 (Tue 09-29); W8P20 +0.0081 public / +0.0067 private; W16 +0.0006 / +0.0004; S16 -0.0012 / -0.0013; C3 +0.0027 / -0.0003.

A/A null, reading [M]: for the cold offsets the z of a day contrast has sd 1.25 (PRIVATE - PUBLIC) and 1.65 (last - first test day) instead of 1, and the chi2 has mean 5.8 instead of 4; [I] the same cold items appear in many feeds of a day, so feeds are not independent for this statistic. For Spearman, top-1 and top-3 move the feed-level SE is conservative (sd of z 0.5-0.7). The 20 splits share members, so these sd carry about ± 20%. Consequence: a cold-offset contrast needs about 4-5 nominal SE to be a 3 sd event.

One-line reading per candidate:

- **W16**: no drift. Agreement with W8 flat over the six days (0.9963-0.9968), cold offset 0.000 ± 0.001 everywhere, chi2 4.8 on 4 df, no contrast above 2.0 SE.
- **W8P20**: no drift on the late private days (last day - val Tuesday -0.0001 ± 0.0003). It carries a constant cold promotion of +0.0055; the public part is the outlier (+0.0064, 0.0009 ± 0.0002 above private; nominal z 4.2, about 3.4 after the A/A inflation; chi2 18.5 against an A/A maximum of 14.9): the one public read of this file is taken where its cold promotion is 16% larger than on private. Size: a quarter of the fine-tune offset below; [I] far below anything NDCG can show.
- **W8TO**: flagged, and it is the known effect. Against W8 its cold offset is -0.0094 on val Tuesday, -0.0136 on val Thursday, -0.0169 on public, -0.0195 on private, -0.0210 on the last day: a monotone 2.2 times growth, equal to half of ft_redesign's +0.021 to +0.044 (known-answer check of the tool), 22 nominal SE. The val read-out of "file vs train-only pool" (-0.00009 ± 0.00043) is taken at about half the private offset. [I, ft_redesign] effect below 0.0002 at w_ft 0.5. Every file with a fine-tune half shares it, so it cancels between W16, W8, W8P20 and S16.
- **S16**: no drift. Cold offset -0.0015 / -0.0016 against val -0.0007 / -0.0014; chi2 9.7 (p 0.046 nominal, inside the A/A range).
- **C3 (scale reference)**: the size of a different pool: Spearman 0.987, same top-1 82.5%, top-3 move 0.53 positions, flat by day. The cold offset is not homogeneous over the day groups (chi2 15.6, at the A/A maximum of 14.9), driven by Sunday morning (+0.0018 against -0.0013 to -0.0032 elsewhere; 2,512 feeds), not by a trend: last day - val Tuesday -0.0003 ± 0.0009.
- For the pair W16 + W8 nothing separates the two files on the private days. The only day-dependent quantity in any of our files is the cold promotion of the fine-tune half, common to all of them.

## 2. What changed against the first pass

| point of `merge_verify.md` | status in r2 |
|---|---|
| 2.1 numbers of the rank-file penalty were a tie-break | re-measured with an explicit tie-break (1.4); verdict line 4 rewritten; the tool's NOTE rewritten |
| 2.2 val check of a rank-file partner depended on the scale of our scores | fixed: the shape is the sorted values of our own z (`mc_core.sorted_vals`), the mix is rounded to 1e-8, ties go to the better file (`mc_core.mix_rank`); tests 33-35 |
| 2.3 a partly corrupted file was passed as the ideal partner | fixed: per-feed distribution check, FAIL above 0.75%, WARN above 0.2% (1.5); tests 26-30 |
| 2.4 55-pair interval carried by 3 pairs; "prediction interval" was a band of the line | 52-pair fit and line + residual added (1.3); the tool's UNCERTAINTY text rewritten |
| 2.4 gate 1 not printed | printed; DECISION now needs gate 1 and gate 2 (the first pass and v1 decide on gate 2 alone) |
| 2.4 `--blend` wrote although the rule said no | refused unless gates 1-3 passed or `--force`; reader FAILs and an inverted file are never overridable |
| 2.4 gate 3 read on my w* | unchanged, and still the lead's call: the tool prints the z of the w* mix (decides) and of the equal mix |
| 2.5 a copy of a valid submission in my cache; 1e200 values passed | the copy is deleted and the test reads `subs/` in place; scores beyond 1e30 are a FAIL |
| drift flag had no A/A null | `a1_drift_null.py` (1.8) |
| new | IN-HOUSE mode (flat `.npy`, val gap) for our own models; break-even line; chi2 column in the drift table; section 1.6 |

## 3. Method

- **Correlation**: per-feed Pearson r_f of two flat vectors (`mc_core.pf_corr`); Spearman = the same on within-feed mid-ranks; aggregated as feed-mean (SE over feeds) and as M-weighted mean (identical to `np.corrcoef` of concatenated per-feed z).
- **Law-rho from files**: our model scores from `preds/ens_<TAG>_{ft,to}_test.npy` (0.5 z + 0.5 z, as `build_r3.py`) after checking that they reproduce >= 99.99% of the ranks of our parquet. A partner file is a rank file when its score is an exact affine function of its own within-feed rank in >= 99% of the feeds; a rank file is mapped onto the sorted per-feed z of our scores, a score file is z-scored as it is.
- **Calibration** (`c1_calib.py`, `c2_extend.py`): partners, loaders and blends of `p4_partner.py`; correlation under every definition on val and test; refits, leave-one-out, feed bootstrap (2,000 resamples, all pairs together), the 55 / 52 / 28-pair fits, the tie-break table, the per-feed distributions, break-even gaps.
- **Per day**: Moscow-time day (ts + 3 h) // 86400 as `presub_gate.py` / ft_redesign; public = timestamp < 1790497920 (`src/pmix.py`, `src/calib.py`) = Saturday plus Sunday before 11:32 MSK = the first 8,562 feeds in time order.
- **Val check**: partner val file through the same reader on the val key set; NDCG by `src/metrics.ndcg_per_feed` on the tie-free order; paired SE against the better file; warns when the partner's val minus public exceeds ours by more than 0.003.
- **Drift**: per-feed statistics from the two files, mean ± SE per group; cold offset = mean z of cold minus mean z of warm candidates within a feed (feeds with both kinds), candidate minus reference. Flag = (PRIVATE - PUBLIC or last - first test day at >= 3 SE) and private outside the range of the val days and public by more than 3 SE; defined in the first pass before the candidates were run, unchanged. A/A: 20 random 8-vs-8 splits of the 16 wide students, train-only test predictions.
- **Rules applied unchanged**: gate 1 public within 0.002; gate 2 law on record - 0.0005 >= 0; gate 3 >= 2 paired SE on val. Tool settings of mine, not rules on record: the tie-break, the 0.75% / 0.2% lines, the 1e30 bound, w*.

## 4. Limits and what I did NOT check

- No real partner file exists; every number is from our own models. Nothing on disk has rho < 0.95 with gap < 0.005.
- The law is fitted and checked on val gaps; a merge has to use the public gap. Val-to-public transfer and the private side are not measured (no labels).
- The rank-file cost for a STRONG partner at rho 0.90 is unknown; the reads at low rho are weak models. The tie-break to the better file was compared on val only, for partners that are weaker than W8 (and S16).
- Shape restoration is validated on 10 partners (7 neural pools of ours, 2 GBDT, 1 LLM). A partner score file in probabilities instead of logits is not tested; the tool prints the within-feed skewness of both files.
- The distribution check is calibrated on 10 genuine models against W8 and on synthetic damage of one file (C3). A genuine partner far from every model on disk could exceed 0.75%; then `--allow_low_agreement` after inspection. Damage of less than 0.2% of the feeds, or damage that keeps the per-feed correlation above 0.3 (for example a few swapped items per feed), is not detected.
- The bootstrap covers feed noise, not misspecification; the 55 pairs are not independent; the "share above the margin" is not a probability of success.
- A partner's val predictions cannot be proven label-free; the val-minus-public warning is a heuristic.
- `--blend` was never executed to disk (no submission file may be written); every blend path ran with `make_sub` stubbed or ended in a refusal. The lead's first real use is the first real write.
- The A/A null of the drift table uses 20 splits that share members, train-only pools only (no fine-tune half), one recipe. The NDCG size of a drift is not measured here (label-free tool); the bound quoted is ft_redesign's.
- `cw_score` exists only when both files are ours and `preds/ens_<TAG>_*` reproduces them; a partner or blended file gets `cw_rank` and no val rows.
- Section 1.6 uses the one CatBoost model of the lens list. The other tree rows (w25, LightGBM students, the round-2 arms) were not re-run here; their law values are in `reports/CATBOOST_1115/audit.md`. I did not test whether any tree model passes the member rule: that is the CatBoost track's read, at w 0.1 with the holdout.
- Whether the platform accepts duplicated keys, float64 or non-rank scores was not checked. Days are not split finer than the public bound on Sunday.
- `c1_calib.py` tables T4-T8 were re-run with the first pass's seeds; `c2_extend.py` uses other seeds, so its bounds differ from T4 / T8 by up to 0.0001.
- Rule notes, for the record. (a) One one-line Python call that only read a JSON file (`solvers/out/ensemble/exp7.json`, standard library only) was started without `CUDA_VISIBLE_DEVICES=""` on its command line; every numerical run had it and `mc_core.py` sets it itself. (b) The folder `nn_x/w12/next/merge/` already held the first pass of this same KEY; I treated it as my own directory, copied all of it to `r1/` first and then edited in place; I deleted one cache file (`T_00_baseline.parquet`, a byte copy of a valid submission) and five superseded `out/known_*` texts, all preserved in `r1/out/` except the cache file. (c) One waiting loop of my own shell was stopped by its task id; no other process was signalled. (d) The known-answer and test inputs in `cache/` (`NOTSUB_*`, `T_*`) have the (feed_id, item_id, score) layout because the tool reads that layout; none is a candidate or a blend.

## 5. Usage (paste)

```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
# A. partner check. Put the partner files in their own directory; ask for MODEL SCORES (logits) on test and on val, not for a submission file.
.venv/bin/python -I nn_x/w12/next/merge/merge_check_v2.py --partner <dir>/partner_test.parquet --partner_public <their public> \
    --partner_val <dir>/partner_val.parquet            # optional: [--ours subs/cand_W16kd_r3.parquet --ours_public <ours>] [--flip_partner] [--allow_low_agreement]
#    exit code 0 = read, 2 = file not usable, 3 = inverted or unrelated order. Read the lines GATE 1, GATE 2, DECISION, GATE 3.
#    Blend only after the three gates:  ... --blend subs/cand_MERGE_<tag>.parquet --w law     (eq = 0.5, law = gap-adjusted weight, or a number = weight of the partner; --force = owner's override)
# B. drift table of any candidate against pick 1 (about 6 s; a flat .npy of our own model is accepted too)
.venv/bin/python -I nn_x/w12/next/merge/drift_by_day.py subs/cand_W16kd_r3.parquet [subs/cand_W8kd_r2.parquet] [--json out.json] [--no_val]
# C. IN-HOUSE: price one of our own models (for example a CatBoost arm) against W16 with the VAL gap: level, rho by day, both laws, break-even gap, measured equal mix
.venv/bin/python -I nn_x/w12/next/merge/merge_check_v2.py --ours subs/cand_W16kd_r3.parquet --partner <arm>_test.npy --partner_val <arm>_val.npy
```

## 6. Commands to reproduce

```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
D=nn_x/w12/next/merge; C=$D/cache; O=$D/out; PY=".venv/bin/python -I"
.venv/bin/python $D/t0_make_known.py                                # known-answer inputs in cache/ (NOTSUB_*; tool inputs, not submissions)
.venv/bin/python $D/c1_calib.py > $O/c1_calib.txt                   # tables 1.1-1.3, 1.7 (65 s)
.venv/bin/python $D/c2_extend.py > $O/c2_extend.txt                 # tables 1.3 (52 pairs, PI), 1.4, 1.5 (calibration), 1.6 (2 min)
$PY $D/merge_check_v2.py --partner subs/cand_Q2a8_ftbag3_memeq.parquet --partner_public 0.393454 > $O/known_W8_vs_C3_file.txt
$PY $D/merge_check_v2.py --partner $C/NOTSUB_c3_test_scores.parquet --partner_val $C/NOTSUB_c3_val_scores.parquet > $O/known_C3_scores_inhouse.txt
$PY $D/merge_check_v2.py --partner $C/NOTSUB_c3_test_ranks.parquet --partner_val $C/NOTSUB_c3_val_ranks.parquet > $O/known_C3_ranks_inhouse.txt
$PY $D/merge_check_v2.py --partner nn_x/cb26/data/e3_xm_yeti_test.npy --partner_val nn_x/cb26/data/e3_xm_yeti_val.npy > $O/known_CatBoost_npy_inhouse.txt
$PY $D/merge_check_v2.py --partner $C/NOTSUB_catboost_test_ranks.parquet --partner_val $C/NOTSUB_catboost_val_ranks.parquet > $O/known_CatBoost_ranks_inhouse.txt
$PY $D/merge_check_v2.py --ours subs/cand_W16kd_r3.parquet --partner nn_x/cb26/data/e3_xm_yeti_test.npy --partner_val nn_x/cb26/data/e3_xm_yeti_val.npy > $O/known_W16_vs_CatBoost_npy_inhouse.txt
$PY $D/merge_check_v2.py --partner $C/NOTSUB_catboost_test_scores.parquet --partner_public 0.394486 --partner_val $C/NOTSUB_catboost_val_scores.parquet --blend $C/T_blend_must_not_exist.parquet --w law > $O/exercise_gate3_pretend_equal_public.txt   # refused at gate 3, nothing written
$PY $D/merge_check_v2.py --partner $C/NOTSUB_catboost_test_scores.parquet --partner_public 0.390486 --partner_val $C/NOTSUB_catboost_val_scores.parquet > $O/exercise_wstar_gap004.txt
$PY $D/t1_reader_tests.py > $O/t1_reader_tests.txt                  # 41 checks (3 min; --rebuild rewrites the synthetic files)
for c in W16kd_r3 W8P20kd_r3 W8TOkd_r3 S16kd_r2 Q2a8_ftbag3_memeq; do $PY $D/drift_by_day.py subs/cand_$c.parquet --json $O/drift_$c.json > $O/drift_$c.txt; done
$PY $D/drift_by_day.py nn_x/cb26/data/e3_xm_yeti_test.npy subs/cand_W16kd_r3.parquet --json $O/drift_CatBoost_e3_vs_W16.json > $O/drift_CatBoost_e3_vs_W16.txt
.venv/bin/python $D/a1_drift_null.py > $O/a1_drift_null.txt         # A/A null of the drift contrasts (2 min)
```

## 7. Files created or changed

- Tools: `<repo>/nn_x/w12/next/merge/merge_check_v2.py` (tool A, r2), `<repo>/nn_x/w12/next/merge/drift_by_day.py` (tool B, r2), `<repo>/nn_x/w12/next/merge/mc_core.py` (shared reader and correlation code; both tools import it).
- Studies in the same folder: `c1_calib.py` (one column changed), `c2_extend.py` (new), `a1_drift_null.py` (new), `t1_reader_tests.py` (r2), `t0_make_known.py` and `x0_explore.py` (unchanged).
- Outputs: `<repo>/nn_x/w12/next/merge/out/`: `c1_calib.txt` / `.json`, `c2_extend.txt` / `.json`, `a1_drift_null.txt`, `t1_reader_tests.txt`, `known_*.txt` (6), `exercise_*.txt` (2), `drift_<candidate>.txt` / `.json` (5 candidates and the CatBoost model against W16).
- First pass, unchanged: `<repo>/nn_x/w12/next/merge/r1/` (scripts, `out/`, `merge_r1.md`).
- Cache (715 MB, can be deleted): `<repo>/nn_x/w12/next/merge/cache/NOTSUB_*.parquet` (10 known-answer inputs), `T_*.parquet` / `T_*.npy` (37 synthetic test inputs).
- This report: `<repo>/reports/NEXT_1115/merge.md` (replaces the first pass's report, kept as `r1/merge_r1.md`).
- `src/merge_check.py` and every other existing file outside the folder are unchanged; no file in `subs/`.
