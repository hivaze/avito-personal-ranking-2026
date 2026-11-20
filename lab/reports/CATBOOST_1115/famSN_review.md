# CatBoost round 2: adversarial review of family S (teacher-target students) and family N (native levers). KEY = famSN_review

Written 2026-11-15 06:37-06:58Z. CPU only, at most 6 threads per process, no fit except two diagnostic fits of 40 and 80 trees on 2,500 feeds. Nothing written outside `nn_x/w12/cb/famSN_review/` and this file.
Marks: **[M]** measured here with my own code, **[D]** read in a project file or log, **[I]** inferred. "a ± b" = mean ± 1 SE, paired per feed; "x / y" = val (14,423 feeds) / holdout (30,000 feeds).
Reviewed: `reports/CATBOOST_1115/famS.md` (as rewritten on disk at 06:39:43Z) and `reports/CATBOOST_1115/famN.md` (06:18Z), their scripts, logs, models and prediction files.

## 0. Verdict

1. **Both verdicts are CONFIRMED.** Nothing is PROMISING, nothing passes the member rule, the stage-2 gate is not met, and no native CatBoost lever shows an effect. Every number the two verdicts rest on was re-derived from the saved prediction files with code that shares nothing with the studies' readers; the largest deviation is 0.00002 nats; pooled z values agree within 0.02 for the member rule and within 0.11 for one paired NATS difference [M].
2. **I could not break the result** [M]. Twelve full-scale tree scores next to W16 (nine family S models, the hard-label reference, the two LightGBM students), with 13 coefficients fitted on the witness itself, read +0.00174 ± 0.00046 val / +0.00072 ± 0.00020 holdout against the bar 0.0016 on both; the val figure carries about 0.0004 of in-sample optimism [I]. With coefficients taken from the other witness the hard-label tree reads +0.00100 ± 0.00026 / +0.00033 ± 0.00025 and three students +0.00054 to +0.00074 / +0.00027 to +0.00048. No blend weight between 0.02 and 0.3 gives pooled z >= 2 for the bundle, the hard-label tree or S4b (largest about +1.1, the bundle at 0.02, with a negative holdout delta). On val M >= 50 alone the largest NATS of the 45 runs is 0.00123 and the largest member-rule z is +0.78.
3. **No leak that could create or hide a result** [M]. The tree count of all 44 runs is the maximum of the train-side selection curve; saved predictions reproduce exactly from the saved models; evaluation labels appear only in the readers; the hand-made LDA / KNN columns are leave-user-fold-out as declared and have the same distribution and label association on fit, selection, holdout and val rows.
4. **Four statements should be weakened; none changes a decision** (section 3): family S "QuerySoftMax on the soft label beats RMSE per CPU minute" holds on the holdout only; family S "0.0015 below the LightGBM plateau" is not a like-for-like comparison; family N "has_time is worth +0.0016" is seed noise; family N's native KNN arms are confounded on CPU, so the clean KNN answer is the hand-made arm (also null).
5. **Rule issues, all minor**: family S processes created numba cache files inside the read-only `nn_x/w12/cb/infra/__pycache__/`; family S ran three fitting processes for about 80 minutes (disclosed); family N did not act on the lead's notice of 02:19Z (the missing reads are supplied here and find nothing). No GPU process and no change of a read-only data or source file.

## 1. Claim by claim

My code: `nn_x/w12/cb/famSN_review/rv_common.py` (feed meta read from the parquet and the holdout meta file, per-feed z, NDCG with pessimistic ties checked against a per-feed loop on tied scores, W16 rebuilt from the 16 member prediction files, conditional logit by scipy BFGS with an analytic gradient). It was checked on the reference arm before any other read; `nats.py` was not imported.

### Family S

| claim | study's number | my number | agree |
|---|---|---|---|
| W16 train-only pool level | 0.40144 / 0.32460 | 0.40144 / 0.32460, rebuilt from the 16 member files; max abs difference to the infra cache 6e-7 | yes |
| reference `ts_lr020_s0`: NATS; member rule z | +0.00117 ± 0.00038 / +0.00051 ± 0.00017; -0.36 | +0.00117 ± 0.00038 / +0.00051 ± 0.00017; -0.36 | yes |
| NATS range of the ten S models | +0.00030 to +0.00074 / +0.00023 to +0.00048 | +0.00030 (S3) to +0.00074 (S4) / +0.00023 (S3) to +0.00048 (S4) | yes |
| member rule, pooled z of the ten models | -1.05 to +0.06 | -1.05 (S1) to +0.06 (S4) | yes |
| levels, rho, NATS and member rule of each of the ten models | arm table of the report | identical to every printed digit (`logs/famS_own.log`) | yes |
| declared bundle (3 seeds of S1f): level, rho | 0.38461 / 0.30797, 0.900 | 0.38461 / 0.30797, 0.900 | yes |
| declared bundle: NATS | +0.00056 ± 0.00026 / +0.00040 ± 0.00015 | +0.00056 ± 0.00026 / +0.00040 ± 0.00015 | yes |
| declared bundle: member rule | -0.00027 ± 0.00037, -0.00000 ± 0.00020, z -0.52 | -0.00026 ± 0.00037, -0.00000 ± 0.00020, z -0.50 | yes (float64 against float32 scores) |
| saved bundle file = member-equal z-mean of the three seeds | stated | max abs difference 5e-7 val / 7e-7 holdout | yes |
| seed-1 member = the long run cut at its selected tree | stated (`fs_trunc.py`) | `s1long.cbm` at 884 trees reproduces the saved file exactly (difference 0); a 40-tree run equals the first 40 trees of an 80-tree run exactly (2,500 feeds) | yes |
| best level reached four ways | 0.38461 / 0.30797; 0.38451 / 0.30792; 0.38453 / 0.30762; 0.38456 / 0.30861 | the same four pairs | yes |
| bundle minus hard-label tree | +0.00311 ± 0.00097 / +0.00335 ± 0.00056, z +6.51 | +0.00311 ± 0.00097 / +0.00335 ± 0.00056, z +6.51 | yes |
| bundle minus LightGBM student at 600 rounds | -0.00022 ± 0.00071 / +0.00036 ± 0.00040 | -0.00022 ± 0.00071 / +0.00036 ± 0.00040 | yes |
| bundle minus LightGBM student at its plateau | -0.00152 ± 0.00077 / -0.00166 ± 0.00044, z -4.08 | -0.00152 ± 0.00077 / -0.00166 ± 0.00044, z -4.08 | number yes; reading weakened (3.2) |
| stage-2 gate, val >= 0.3900 | not met, short by 0.0054 | best val 0.38461; short by 0.0054 | yes |
| gap of the bundle to W16; by M bin | 0.0168 ± 0.0013 / 0.0166 ± 0.0008; 0.0194, 0.0150, 0.0166 | 0.01682 ± 0.00134 / 0.01663 ± 0.00081; 0.01942, 0.01503, 0.01663 | yes |
| student given [W16, hard-label tree] | +0.00007 ± 0.00010 / +0.00000 ± 0.00001 | +0.00006 ± 0.00009 / +0.00001 ± 0.00002 | yes |
| hard-label tree given [W16, student] | +0.00070 ± 0.00030 / +0.00012 ± 0.00008 | +0.00068 ± 0.00029 / +0.00012 ± 0.00008 | yes |
| W16 + student + hard-label tree | +0.00124 / +0.00052 | +0.00123 / +0.00052 | yes |
| bundle(S1f seed 0, hard-label tree) | 0.38457 / 0.30767; +0.00098 ± 0.00035 / +0.00048 ± 0.00016; z -0.69 | the same | yes |
| bundle(3-seed bundle, hard-label tree) | 0.38493 / 0.30812; +0.00093 ± 0.00034 / +0.00050 ± 0.00016; z +0.08 | the same | yes |
| val M >= 50, bundle: NATS; member rule | +0.00034 ± 0.00025; -0.00009 ± 0.00041 | +0.00034 ± 0.00026; -0.00008 ± 0.00041 | yes |
| NATS(bundle) minus NATS(hard-label tree) | -0.00062 ± 0.00022 / -0.00011 ± 0.00009 | -0.00063 ± 0.00022 / -0.00012 ± 0.00009 | yes |
| ten staged comparisons (plateau, S3 / S2 / S4b / S4y at equal rounds and equal wall time, S1 minus S1f) | sections 2.2, 2.4, 3.2 | identical to every printed digit (`logs/staged_own.log`) | yes |
| S4b ahead of RMSE per CPU minute | +0.00052 ± 0.00084 / +0.00124 ± 0.00047 | the same; against the RMSE plateau run +0.00005 ± 0.00086 / +0.00068 ± 0.00048, z +1.06 | number yes; reading weakened (3.1) |
| tree count chosen on the train-side split | stated | 10 of 10 runs sit at the maximum of the selection NDCG curve | yes |

### Family N

| claim | study's number | my number | agree |
|---|---|---|---|
| levels, rho, NATS and member rule of all 34 runs | tables of section 3 | identical to every printed digit (`logs/famN_own.log`) | yes |
| largest NATS of a single run | +0.00111 ± 0.00037 val (`Y_oh64` s1); +0.00050 ± 0.00016 holdout (`Y_q1024` s1) | +0.00111 ± 0.00037; +0.00050 ± 0.00016, the same runs | yes |
| member rule, pooled z over all runs | -1.42 to +0.67 | -1.42 (`Yref_nt` n6k s0) to +0.67 (`Lref` n30k s0) | yes |
| any arm with 2 SE of NATS over its reference on both witnesses, or on either | none | none of the 14 arm-against-reference reads, none of their single-seed reads | yes |
| largest paired NATS gains | +0.00010 ± 0.00014 / +0.00004 ± 0.00005; +0.00009 ± 0.00009 / +0.00003 ± 0.00005 | the same (`Y_def_all32`, `R_knn50_pair`) | yes |
| text token sets, YetiRank: d NATS | -0.00002 ± 0.00010 / -0.00002 ± 0.00004 | the same | yes |
| LDA under Logloss: d NATS; d level against Logloss; against YetiRank | -0.00009 ± 0.00013 / +0.00002 ± 0.00005; -0.0012 ± 0.0008 / -0.0008 ± 0.0004; -0.0047 / -0.0016 | the same; -0.00117 ± 0.00077 / -0.00078 ± 0.00038; -0.00472 ± 0.00086 / -0.00155 ± 0.00048 | yes |
| native KNN k=50, 6,000 feeds: d level; d NATS | +0.0006 ± 0.0008 / +0.0008 ± 0.0004; +0.00009 ± 0.00009 / +0.00003 ± 0.00005 | +0.00064 ± 0.00075 / +0.00080 ± 0.00044; the same | number yes; arm confounded (3.4) |
| hand-made KNN columns at 30,000 feeds: zero | `R_hknn3` +0.00002 ± 0.00051 / +0.00031 ± 0.00031; d NATS -0.00014 ± 0.00008 / -0.00001 ± 0.00004 | the same | yes |
| best level of the family (`Rref`), and its NATS against hard-label YetiRank | 0.38315 / 0.30634; paired -0.00038 ± 0.00020 / -0.00010 ± 0.00009 | 0.38315 / 0.30634; -0.00038 ± 0.00020 / -0.00010 ± 0.00009 | yes |
| 90 selected features | -0.0017 ± 0.0007 / -0.0011 ± 0.0004 | -0.00174 ± 0.00065 / -0.00106 ± 0.00039; both seeds negative on both witnesses | yes |
| has_time against default permutations | +0.0016 ± 0.0006 / +0.0007 ± 0.0004 | the same mean; single seeds -0.00087 ± 0.00084 / +0.00075 ± 0.00051 and -0.00234 ± 0.00090 / -0.00207 ± 0.00054 | number yes; it is seed noise (3.3) |
| bundle of the two `Yref` seeds | +0.00100 ± 0.00035 / +0.00045 ± 0.00016 | the same | yes |
| LDA direction on u*e against `cos_u`; KNN teacher means against W16 | 0.67; 0.48 to 0.58 | 0.669; 0.475 to 0.573 | yes |
| raw hand-made columns as scores | `hknn50_pair` +0.00065 ± 0.00028 / +0.00013 ± 0.00008; `hlda_pair` +0.00035 ± 0.00020 / +0.00024 ± 0.00011 | the same | yes |
| importance of the new columns | `t_cat` 1.39, `t_clu` 0.02; `e_pair` 1.25 | read from the saved models: the same | yes |
| 1024 borders address the two intended columns | stated, no log | saved border files: 925 and 1,024 borders on features 78 and 172, 254 elsewhere [D]; one-hot size 64 and the text / embedding processing are in the saved models [M] | yes |
| tree count chosen on the train-side split | stated | 34 of 34 runs sit at the maximum of the selection NDCG curve | yes |
| CPU cost of the native calcers, LossFunctionChange rank correlation, select_features details, linear R² values | sections 4 and 6 | not re-derived (logs read only) | not checked |

## 2. Leak audit

| input | what feeds a TRAIN row | what feeds a val / holdout / test row | finding |
|---|---|---|---|
| S arms: `teacher_mean11` | used as the training target only (`run_s.py`); no NN-derived column is an input | not used | clean [M by code search] |
| S3 block | label-free: PCA fitted on fit-feed candidate vectors, products with the feed's own vectors | the same projection | clean; the study's own distribution read exists (`reads/dist_pca.json`) [D] |
| N reference and text arms | 177 record features; token strings from the candidate's category / cluster and the user's history | the same construction | clean, label-free [M by code reading] |
| N hand-made LDA columns | direction fitted on the fit rows of the other four user folds | direction of all fit rows | fold exclusion re-derived: my out-of-fold direction reproduces the stored column to 2e-7, the all-fold direction differs by 0.09 [M] |
| N hand-made KNN columns | mean out-of-fold teacher value of the nearest fit rows of the other user folds | nearest among all fit rows | measured on 400 rows: stored column correlates 0.987 with the exact other-fold 50-NN mean and 0.891 with the version that keeps the own fold [M] |
| N hand-made columns, train against apply | within-feed correlation with the click on fit rows 0.048 (`hknn50_pair`), selection rows 0.048, holdout 0.049, val 0.057; means and standard deviations within 3% | | no train-side inflation and no mismatch; all nine columns in `logs/hx_audit.log` [M] |
| N hand-made columns, user and time | no user of the fit subset is split over two folds | 184 of 30,000 holdout feeds (0.61%) have their user in the fit subset, 100 (0.33%) with a LATER fit feed; val 77 of 14,423 (0.53%), all earlier | a breach of "never a user's later snapshots" in letter for 0.33% of the holdout feeds; too small to matter, and the arms are null [M] |
| N native LDA / KNN (CatBoost online calcers, CPU) | earlier feeds in time order plus the shuffled earlier rows of the row's OWN feed | all fit rows | train / apply mismatch for KNN, measured: 25% of a row's 50 nearest earlier rows and 57% of its 5 nearest come from its own feed (own-feed prefix = 93 of about 539,000 rows); correlation of the 50-NN teacher mean with the row's own target 0.576 online against 0.470 with the own feed excluded [M] |
| selection of trees, columns, arms | train-side split only in all 44 runs; `select_features` uses fit and selection feeds | | clean [M] |
| holdout feeds in a fit | `xm` excludes the 30,000 holdout feeds by construction (asserted in cbx); feed sizes of cbx `hold` equal the holdout meta file | | clean [M] |

Evaluation labels (`cand_target`, `holdout_y`, the readers' `m.y`) appear only in the evaluation scripts of both families; `data/feats_quarantine` and `cbx.nn_input` are not referenced [M by code search].

## 3. Issues by severity

No issue changes a verdict.

### Medium (wording or interpretation)

3.1 **Family S, verdict line 6: "only S4b beats RMSE per tree and per CPU minute"** is a holdout-only read. At equal wall time S4b minus S1f is +0.00052 ± 0.00084 on val and +0.00124 ± 0.00047 on the holdout; against the RMSE plateau run it is +0.00005 ± 0.00086 / +0.00068 ± 0.00048 (pooled z +1.06) [M]. S4b was also added after a pilot that was read on both witnesses. The report's section 5 says this; the verdict line does not.

3.2 **Family S, verdict lines 3 and 7: "0.0015 / 0.0017 below the LightGBM plateau"** compares a CatBoost plateau at learning rate 0.3, and a learning-rate 0.1 run stopped at 2,018 of 3,000 trees with its selection curve still rising (0.31102 at 1,800 trees, 0.31133 at 2,014), with a LightGBM plateau at learning rate 0.08 and 2,400 rounds [D]. The number is right; "CatBoost stops below LightGBM" is an inference. The best CatBoost level is a level at a CPU budget: S4b, the highest single model, was at its last tree when the cap stopped it and has no plateau run and no seeds. This cannot reach the gate: S4b's selection curve gained 0.0008 over its last 200 trees, so 0.386 to 0.387 is the most a longer run could give [I].

3.3 **Family N, by-product "has_time reads +0.0016 ± 0.0006 / +0.0007 ± 0.0004"** is seed noise. The two seeds disagree in sign on the holdout (table above), one control run selected 189 trees, and two seeds of the SAME configuration differ by as much: `Yref` n6k +0.00083 ± 0.00086 / +0.00095 ± 0.00051, `Rref` n6k +0.00065 ± 0.00073 / +0.00114 ± 0.00043 [M]. It should not be carried forward.

3.4 **Family N, native KNN arms are confounded on CPU.** `R_knn50_pair`, `Y_def_pair32` and `Y_def_all32` train on a column that is largely a within-feed statistic and apply a cross-feed one (section 2). The report treats the own-feed prefix as a harmless caveat. These three arms therefore say only that the CPU-native KNN as available gives nothing at 6,000 feeds. The clean answer is the hand-made leave-user-fold-out KNN, which is null at 6,000 and at 30,000 feeds, so the class (i) reading stands.

3.5 **Family N, "no-op" is a bound.** Two seeds at 22% of the data (4.5% for the native calcers) detect about 0.0015 of level and 0.0003 of NATS. That is enough for the SCREEN, which needs +0.0004 val and +0.0011 holdout over the full-scale hard-label reference. One half of N2 was not really tested: the cluster-token dictionary covers 16% (500 tokens) and 32% (2,000 tokens) of the occurrences, and `t_clu` has 0.02% importance [D]. The category tokens were tested (80% and 99.5% coverage).

3.6 **Family N did not act on the lead's notice of 02:19Z**: no val M >= 50 read, no distribution of the new columns on train against val and test rows, no lag-built statistics. Supplied here: val M >= 50 for all 34 runs (largest NATS 0.00123 ± 0.00048, `Y_def_all32`, one seed at 6,000 feeds; largest member-rule z +0.78) and the fit / selection / holdout / val distributions of the nine hand-made columns (no mismatch). Still open: no hand-made column exists for test rows, so the test distribution cannot be read. It is moot while the arms are null.

### Low (rules and record)

3.7 Family S processes created files in read-only directories: numba cache files in `nn_x/w12/cb/infra/__pycache__/` at 01:16:41 (`cbx._feed_center`; the first S fits logged their start at 01:17:04) and at 02:31:24 (`cbx._feed_softmax`; `pilot15k_qsm` logged its start at 02:31:25), and probably `nn_x/w12/cb/__pycache__/nats.cpython-312.pyc` at 01:24:11. No existing file changed. The report says nothing was written outside its directory. Family N redirected its caches, as did this review.

3.8 Family S ran three fitting processes (28 threads) at 01:25-02:01Z, 02:23-02:54Z and 02:56-03:12Z; the log time ranges confirm the report's own disclosure. Family N never had more than two [M from log time ranges].

3.9 `famS.md` was rewritten at 06:39:43Z, after its FINAL stamp of 06:36Z: a provenance sentence only, no number changed (checked against the returned verdict). The file now says the second run was not cut and was alive beside the third from 06:01Z; the returned verdict's own-account item (e) says it was cut.

3.10 Family N changed from early stopping to fixed budgets after one run's level had been seen (disclosed). Family S pilots chose S4b, depth 10 and the PCA block on both witnesses (disclosed). Neither can matter for a null result.

## 4. Selection audit

- SCREEN applied as written: >= 0.0016 on val AND holdout; no run or bundle of either family reaches it on either witness. Only my 13-coefficient in-sample stack exceeds it, on val alone [M].
- MEMBER RULE applied at weight 0.1 against the W16 train-only pool, which I rebuilt from its 16 member files [M].
- Family S: the bundle was declared before the reads; the arm for the seeds was chosen by the train-side selection NDCG (S3 0.31200, S1f 0.31165, S1 0.31133, S4b 0.31110, read in the curve files), not by the witnesses. The highest-level model, S4b, was not taken [M for the curve values, D for the time order of the declarations].
- Family N: no arm, column or weight was chosen on a witness; removals followed nulls [D].
- No reported winner exists in either family, so there is no best-of-N on one witness.

## 5. Read-only files and GPU

- `src/`: no file newer than 01:00Z. `data/`, `subs/`, `nn_x/cb26/`, `nn_x/w25/`: none. `nn_x/w5/` (1,727 files), `preds/` (20), `nn/` (8), `nn_x/abl/` (44): checkpoints, logs and predictions of the lead's neural lanes; none is a CatBoost file [M].
- `nn_x/w12/cb/infra/`: only the cache files of 3.7 and the infra agent's own `repro_valA` outputs of 01:06-01:07Z; `nn_x/w12/cb/docs_feat/`: the first-round agent's last pilot files of 01:07-01:08Z. `nats.py` (01:09Z) and `PREDECLARED.txt` (02:19Z) are the lead's own edits [M].
- `nvidia-smi --query-compute-apps`: 32 processes, all neural training (`train.py`, `train_full.py`, `train_full_abl.py`); none from this track. Neither family's code contains a GPU task type; every launch script sets `CUDA_VISIBLE_DEVICES=""` [M].

## 6. What the lead should conclude

- Families S and N give the STOP rule nothing to count. The track's fate rests on family B alone.
- The only tree object with pool-independent information is the hard-label tree, and it holds too little: +0.00117 / +0.00051 fitted on the witness, +0.00100 / +0.00033 with the other witness's coefficients. A distilled student holds a subset of it.
- The owner's question has the same answer on both paths examined here: CatBoost is not under-used on the teacher-target path or through its native feature processing in any way the screen can see.
- Not known, because not run: the plateau and seeds of S4b (level only, about +0.001 to +0.002), native KNN beyond 6,000 feeds (not affordable on CPU), cluster tokens with a dictionary that covers them.

**(i) Is any native CatBoost lever more than a no-op on this data?**
No. Text token sets, embedding LDA and KNN, the default calcers, 1024 borders, one-hot category and select_features move neither the level by 2 SE on both witnesses nor NATS by 2 SE on either; the bound is about 0.0015 of level and 0.0003 of NATS at 30,000 feeds.
Two were not tested in a form that could work: native KNN (own-feed neighbours on CPU, 6,000 feeds) and cluster tokens (16 to 32% dictionary coverage). The clean hand-made KNN and LDA columns are null.

**(ii) Is the best student a candidate for stage 2, and which files would be used?**
No. The gate val >= 0.3900 is missed by 0.0054 (0.3846), and the student adds nothing beyond [W16, hard-label tree] (+0.00006 ± 0.00009 / +0.00001 ± 0.00002); if a tree score is tried as an input, the hard-label tree is the one to try.
Files on disk cover val / holdout / test only: `nn_x/w12/cb/famS/preds/bundle_s1f_3seeds_z_{val,hold,test}.npy` (declared bundle), `nn_x/w12/cb/famS/preds/s4b_qsm_soft_lr030_s0_{val,hold,test}.npy` (highest single model), `nn_x/w12/cb/infra/out/ts_lr020_s0_{val,hold,test}.npy` (hard-label tree). The out-of-fold scores for train rows that stage 2 needs do not exist for any of them: five user-fold fits of the chosen recipe come first (S1f: about 55 minutes each at 12 threads).

## 7. Method, commands, files, limits

Environment of every command: `cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 ...`; scripts that import cbx or the family N data code also set `PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=nn_x/w12/cb/famSN_review/cache/numba CBX_CACHE_EXTRA=nn_x/w12/cb/famSN_review/cache CBX_THREADS=6`.

| script (under `nn_x/w12/cb/famSN_review/`) | what | log |
|---|---|---|
| `rv_common.py`, `rv_check0.py` | own evaluation code; W16 rebuild; NDCG check; reference arm | `logs/check0.log` |
| `rv_famS.py`, `rv_natsdiff.py`, `rv_staged.py`, `rv_lgb.py` | every family S read | `logs/famS_own.log`, `natsdiff_own.log`, `staged_own.log`, `lgb_own.log` |
| `rv_famN.py`, `rv_hxraw.py` | every family N run and paired read; raw columns; redundancy | `logs/famN_own.log`, `hxraw_own.log` |
| `rv_m50.py` | val M >= 50 for all 45 runs | `logs/m50_own.log` |
| `rv_oracle.py` | in-sample stack of all tree scores; cross-witness NATS; other weights | `logs/oracle_own.log` |
| `rv_hx.py`, `rv_knnfold.py` | hand-made column audit; user overlap; fold exclusion | `logs/hx_audit.log`, `knnfold.log` |
| `rv_models.py repro levers trunc knn` | predictions from saved models; levers inside the models; truncation check; online-KNN emulation | `logs/models.log` |

Leak statement for my own work: I fitted two diagnostic models (RMSE on the teacher logit, 40 and 80 trees, 2,500 `xm` feeds) to test the truncation claim; they were read on train-side selection rows only. Val and holdout labels were used for evaluation reads and for the within-feed association of the hand-made columns (an evaluation read; nothing was fitted or chosen from it).

Own conduct: up to four light processes of mine overlapped for a few minutes (evaluation reads at 4 to 6 threads; the 6-thread model checks among them). No process was signalled. State log: `nn_x/w12/cb/famSN_review/PROGRESS.md`.

NOT done:
- No arm was re-fitted. Fit times, model sizes, the CPU cost table, the LossFunctionChange and select_features reads and the linear R² values of family N were read in logs, not re-measured.
- The teacher files (`teacher_mean11`, `teacher_oof`) and the 177 record features were taken as documented by the infra README; their construction was not audited here. One sanity read: the hard-label tree's gap to W16 is the same on val and on the holdout (0.01994 / 0.01998), so the features show no differential leak on the holdout.
- The time order of family S's declarations (`PREDECLARED_famS.txt`, last modified 06:36Z) cannot be verified from file times; I checked only that the decision log matches the selection curves.
- Native calcer columns cannot be extracted from a CatBoost model, so the own-feed share is an emulation of the calcer's row order, not a read of the fitted column.
- No test-row check of any kind (no label, and family N built no test columns).
