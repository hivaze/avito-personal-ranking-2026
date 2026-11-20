# CatBoost round 2, KEY = famN: CatBoost-native feature processing on real data (family N of the round-2 plan)

Written 2026-11-15 about 06:20 UTC. CPU only, at most 12 threads per process and two heavy processes; nothing written outside `nn_x/w12/cb/famN/` and this file; no submission file; no network.
Question (owner): CatBoost has many options, embedding features among them, and may be under-used. Family N = the NATIVE feature-processing levers: `embedding_features` (LDA, KNN), `text_features` (token sets, BoW), quantisation, one-hot, feature selection.
Tags: **[M]** measured here, **[D]** read in a project file, **[I]** inferred. "a ± b" = mean ± 1 SE, paired per feed; "x / y" = val (14,423 feeds) / holdout (30,000 feeds). NATS = the shared screen `nn_x/w12/cb/nats.py` (gain over the W16 train-only pool; PROMISING iff >= 0.0016 on both).
Scale: **n30k** = a fixed subset of 30,000 fit feeds (4.40M rows) + 5,000 train-side selection feeds, the same for every arm; **n6k** = its first 6,000 + 1,500 feeds (native KNN only). 2 seeds per arm unless stated. Every arm is read against a reference of the same loss at the same scale.

## 0. Verdict

1. **Nothing is PROMISING and nothing is a winner.** 34 runs (14 arms and 5 reference configurations): the largest NATS of any single run is +0.00111 ± 0.00037 val (`Y_oh64` s1) and +0.00050 ± 0.00016 holdout (`Y_q1024` s1), against the bar 0.0016 on both; no arm gains 2 SE of NATS over its reference on both witnesses, or on either one (largest paired gains: +0.00010 ± 0.00014 / +0.00004 ± 0.00005, default calcers on three vectors, and +0.00009 ± 0.00009 / +0.00003 ± 0.00005, native KNN:k=50; one seed each) [M].
2. **Nothing passes the member rule**: the 0.9 / 0.1 blend with W16 has pooled z between -1.42 and +0.67 over all runs (the rule needs >= 2) [M]. No bundle was declared, because no arm was PROMISING.
3. **Every native feature lever is a no-op on this data (class (i))**: text token sets, LDA under Logloss, the default calcers under YetiRank, KNN on teacher targets (native at 6,000 feeds, hand-made at 30,000), 1024 borders on the two top features, one-hot `category_id`. No lever shows a level gain above 2 SE on both witnesses either (class (ii) is empty); none adds pool-independent information (class (iii) is empty) [M].
4. **Best level in this family**: 0.38315 / 0.30634 (2-seed mean; best single run 0.38346 / 0.30686), the RMSE reference on the out-of-fold teacher at 30,000 feeds. It is a target lever of family S, not a native feature lever, and it LOWERS the information: NATS +0.00058 / +0.00033 against +0.00096 / +0.00043 for hard-label YetiRank (paired -0.00038 ± 0.00020 / -0.00010 ± 0.00009) [M].
5. **The reason is redundancy** [M]: the LDA direction on u*e is the cosine the tree already has (correlation 0.67 with `cos_u`, linear R² 0.68 on the 177 features); the KNN teacher means correlate 0.48 to 0.58 with W16 and are 31 to 54% linear in the 177 features (a noisy copy of what the pool knows: path A); the token features duplicate the `cat_cnt_*` / `clu_*` counts (importance 1.4% and 0.02%).
6. **Cost on CPU** [M]: native KNN is single-threaded: 0.3 to 0.6 ms per row at k=5, 1.3 to 2.0 ms at k=50, more than 4.4 ms at k=200 (32-d PCA); a 6,000-feed fit with KNN:k=50 takes 38 minutes against 2.5 and stores a 359 MB model. The full 134K-feed set would need more than 10 hours of one core per column at k=50.
7. **No full-scale run was made** (the protocol reserves it for winners) and no third seed (nothing is near the screen). Family N gives no arm for the STOP rule of the plan to count.
8. By-products (information): `has_time=True` on the time-ordered pool reads +0.0016 ± 0.0006 val / +0.0007 ± 0.0004 holdout of LEVEL over CatBoost's default permutations at 6,000 feeds (2 seeds; not 2 SE on both; no NATS change: +0.00013 ± 0.00009 / +0.00006 ± 0.00004); `LossFunctionChange` ranks the features differently from the importance type on record (rank correlation 0.39); the 90 features kept by `select_features` cost -0.0017 ± 0.0007 / -0.0011 ± 0.0004 of level and change no information: the 177 columns are redundant, but not free to cut.

## 1. Lever by lever

Class: (i) no-op on this data; (ii) level gain on path A (the tree learns what the pool knows); (iii) new pool-independent information (NATS). d = arm minus its same-loss reference, paired per feed, seed-paired; n30k unless stated.

| lever (CatBoost-native unless marked) | arm(s) | d level val / holdout | d NATS val / holdout | class | cost against the reference |
|---|---|---|---|---|---|
| N2 `text_features`: token sets (candidate category x history categories; candidate 256-cluster x history clusters), BoW, 500 tokens per column, YetiRank | `Y_txt_s`, 2 seeds | +0.00041 ± 0.00062 / +0.00002 ± 0.00035 | -0.00002 ± 0.00010 / -0.00002 ± 0.00004 | **(i)** | fit 19.5 min against 17.3 |
| same, 2000 tokens per column, YetiRank | `Y_txt_l`, 1 seed | -0.00008 ± 0.00086 / -0.00020 ± 0.00050 | +0.00004 ± 0.00013 / +0.00000 ± 0.00006 | **(i)** | fit 29.4 min against 17.3 |
| same, 500 tokens, RMSE on the teacher | `R_txt_s`, 1 seed | -0.00065 ± 0.00066 / +0.00074 ± 0.00039 | -0.00006 ± 0.00008 / +0.00001 ± 0.00004 | **(i)** | fit 15.1 min against 8.9 |
| N1 `embedding_features` (u*e, e - hmean, [u ; e]) + LDA under Logloss | `L_lda_all`, 2 seeds | -0.00117 ± 0.00077 / -0.00078 ± 0.00038 (against the YetiRank reference: -0.00472 ± 0.00086 / -0.00155 ± 0.00048) | -0.00009 ± 0.00013 / +0.00002 ± 0.00005 | **(i)**, level slightly negative | fit 10.1 min against 6.4; RSS 44 GB |
| N1 `embedding_features` (u*e, 32-d) + default calcers (LDA + KNN:k=5) under YetiRank, n6k | `Y_def_pair32`, 2 seeds | -0.00053 ± 0.00060 / +0.00024 ± 0.00035 | -0.00005 ± 0.00009 / +0.00001 ± 0.00004 | **(i)**: the expected no-op is confirmed | fit 14.4 min against 4.3; model 150 MB against 2.4 |
| same with the three vectors as three columns, n6k | `Y_def_all32`, 1 seed | +0.00056 ± 0.00083 / +0.00110 ± 0.00050 | +0.00010 ± 0.00014 / +0.00004 ± 0.00005 | **(i)** within noise (level z +0.7 / +2.2 on one seed, whose reference seed selected only 293 trees; the two-seed pair arm above reads zero) | fit 29.1 min against 4.3; apply 5.1 min; model 445 MB |
| N1 `embedding_features` (u*e, 32-d) + KNN:k=50 on the teacher target, RMSE, n6k | `R_knn50_pair`, 1 seed | +0.00064 ± 0.00075 / +0.00080 ± 0.00044 | +0.00009 ± 0.00009 / +0.00003 ± 0.00005 | **(i)** within noise; if real it is (ii), not (iii) | fit 38.0 min against 2.5; apply 6.5 min; model 359 MB |
| N1 KNN:k=200 native | not affordable (more than 4.4 ms per row, single thread) | | | not run | see section 6 |
| N1x hand-made: KNN teacher mean, k = 50 and 200, u*e, as two numeric columns, RMSE, n6k (the like-for-like partner of the native arm) | `R_hknn`, 2 seeds | +0.00007 ± 0.00052 / -0.00021 ± 0.00030 | -0.00004 ± 0.00006 / -0.00001 ± 0.00003 | **(i)** | 3 minutes to build |
| N1x hand-made: KNN teacher mean, k = 50 and 200, three vectors (6 columns), RMSE | `R_hknn3`, 2 seeds | +0.00002 ± 0.00051 / +0.00031 ± 0.00031 | -0.00014 ± 0.00008 / -0.00001 ± 0.00004 | **(i)** | 25 minutes to build all 9 columns |
| N1x hand-made: 3 LDA columns + 6 KNN columns as numeric inputs of YetiRank | `Y_hall`, 2 seeds | -0.00013 ± 0.00068 / -0.00007 ± 0.00039 | -0.00002 ± 0.00013 / -0.00005 ± 0.00006 | **(i)**; rho to W16 0.820 -> 0.835 (toward the pool, no level) | fit 21.3 min against 17.3 |
| N3 `per_float_feature_quantization`, 1024 borders on `it_hist_et1_luo` and `pre2_fe_z` | `Y_q1024`, 2 seeds | -0.00052 ± 0.00059 / -0.00051 ± 0.00035 | -0.00007 ± 0.00009 / +0.00003 ± 0.00004 | **(i)** | none |
| N3 `one_hot_max_size=64` (`category_id` one-hot instead of CTR) | `Y_oh64`, 2 seeds | -0.00017 ± 0.00062 / +0.00051 ± 0.00035 | +0.00001 ± 0.00010 / -0.00000 ± 0.00004 | **(i)** | none |
| N3 `select_features`: 90 of the 177 features | `Y_sel`, 2 seeds | -0.00174 ± 0.00065 / -0.00106 ± 0.00039 | -0.00006 ± 0.00011 / -0.00007 ± 0.00005 | a level LOSS (z -2.7 / -2.8), no information change | selection run 26.4 min |
| protocol: CatBoost's default random permutations instead of `has_time=True` on time-ordered feeds, n6k | `Yref_nt` minus `Yref`, 2 seeds | -0.00161 ± 0.00058 / -0.00066 ± 0.00035 | -0.00013 ± 0.00009 / -0.00006 ± 0.00004 | `has_time` is not worse; a level edge of z 2.8 / 1.9 (one `Yref_nt` seed selected only 189 trees), no information | same speed (0.43 s per iteration both) |

Context rows (not levers of this family; same subset, 2 seeds):

| reference | level val / holdout | rho to W16 | NATS val / holdout | against hard-label YetiRank: d level | d NATS |
|---|---|---|---|---|---|
| `Yref` YetiRank on clicks | 0.37880 / 0.30080 | 0.820 / 0.825 | +0.00096 / +0.00043 | | |
| `Rref` RMSE on the out-of-fold teacher | 0.38315 / 0.30634 | 0.883 / 0.889 | +0.00058 / +0.00033 | +0.00436 ± 0.00089 / +0.00553 ± 0.00054 | -0.00038 ± 0.00020 / -0.00010 ± 0.00009 |
| `Lref` Logloss on clicks | 0.37525 / 0.30003 | 0.804 / 0.816 | +0.00095 / +0.00046 | -0.00355 ± 0.00084 / -0.00077 ± 0.00048 | -0.00002 ± 0.00016 / +0.00002 ± 0.00007 |
| bundle of the two `Yref` seeds (z-mean), reference only | 0.37995 / 0.30194 | 0.827 / 0.829 | +0.00100 ± 0.00035 / +0.00045 ± 0.00016 | | |
| full scale, 134K feeds (`infra/out/ts_lr020_s0`, re-read with the tool) | 0.38150 / 0.30462 | 0.847 / 0.847 | +0.00117 ± 0.00038 / +0.00051 ± 0.00017 | | |

Readings:
- The screen needs +0.0005 val and +0.0011 holdout of NATS over the full-scale hard-label model. No lever of this family moved NATS by more than 0.0001 in either direction; the measured seed-to-seed spread of one fit is about 0.0002 val / 0.00005 holdout. Scaling a no-op to 134K feeds would not change that.
- The teacher target is the clearest picture of path A in these data: +0.0044 / +0.0055 of level and LESS information for the pool. A lever that raised the level the same way would still fail the screen.
- The raw hand-made columns, read as scores by themselves (`logs/hx_raw_columns_nats.txt`), carry NATS +0.00065 ± 0.00028 / +0.00013 ± 0.00008 (KNN:k=50 teacher mean on u*e) and +0.00035 ± 0.00020 / +0.00024 ± 0.00011 (LDA on u*e); the two other vectors carry nothing (at most +0.00012 ± 0.00012). Inside a tree with the 177 features none of it survives as an addition.

## 2. One row per arm

References first. "trees" = selected tree count per seed. d = arm minus its same-loss reference, paired per feed and seed-paired (section 5); NATS = gain of the shared screen `nats.py` over the W16 train-only pool (bar 0.0016 on val AND holdout).

### Scale n30k: one row per arm (mean of its seeds); d = paired per-feed difference to the same-loss reference (seed-paired)

| arm | what | seeds | trees | fit min / seed | model MB | level val / holdout | rho | NATS val / holdout (seed mean; single-seed SE) | d level val / holdout | d NATS val / holdout | winner (>= 2 SE on both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Yref | reference: YetiRank, 177 features, has_time=True (time-ordered feeds, rows shuffled in the feed) | [0, 1] | 534/548 | 17.3 | 2.5 | 0.37880 / 0.30080 | 0.820 / 0.825 | +0.00096 ± 0.00034 / +0.00043 ± 0.00015 | reference | reference | reference |
| Rref | reference: RMSE on the centred out-of-fold teacher logit | [0, 1] | 716/753 | 8.9 | 3.5 | 0.38315 / 0.30634 | 0.883 / 0.889 | +0.00058 ± 0.00027 / +0.00033 ± 0.00013 | reference | reference | reference |
| Lref | reference: Logloss on clicks | [0, 1] | 362/310 | 6.4 | 2.6 | 0.37525 / 0.30003 | 0.804 / 0.816 | +0.00095 ± 0.00034 / +0.00046 ± 0.00016 | reference | reference | reference |
| L_lda_all | N1 native: the three embedding columns together, LDA, Logloss | [0, 1] | 285/398 | 10.1 | 2.5 | 0.37408 / 0.29925 | 0.808 / 0.819 | +0.00086 ± 0.00032 / +0.00047 ± 0.00016 | -0.00117 ± 0.00077 / -0.00078 ± 0.00038 | -0.00009 ± 0.00013 / +0.00002 ± 0.00005 | no (vs Lref, seeds [0, 1]) |
| Y_txt_s | N2 native: text columns t_cat + t_clu (token sets), BoW, dictionary 500 tokens / column (lower bound 50), YetiRank | [0, 1] | 462/486 | 19.5 | 2.5 | 0.37921 / 0.30082 | 0.821 / 0.825 | +0.00094 ± 0.00034 / +0.00041 ± 0.00015 | +0.00041 ± 0.00062 / +0.00002 ± 0.00035 | -0.00002 ± 0.00010 / -0.00002 ± 0.00004 | no (vs Yref, seeds [0, 1]) |
| Y_txt_l | N2 native: same, dictionary 2000 tokens / column (lower bound 20), YetiRank | [0] | 528 | 29.4 | 2.6 | 0.37836 / 0.30067 | 0.817 / 0.822 | +0.00089 ± 0.00033 / +0.00045 ± 0.00016 | -0.00008 ± 0.00086 / -0.00020 ± 0.00050 | +0.00004 ± 0.00013 / +0.00000 ± 0.00006 | no (vs Yref, seeds [0]) |
| R_txt_s | N2 native: text columns, 500 tokens / column, RMSE on the teacher | [0] | 759 | 15.1 | 3.4 | 0.38219 / 0.30655 | 0.884 / 0.889 | +0.00052 ± 0.00025 / +0.00035 ± 0.00014 | -0.00065 ± 0.00066 / +0.00074 ± 0.00039 | -0.00006 ± 0.00008 / +0.00001 ± 0.00004 | no (vs Rref, seeds [0]) |
| Y_q1024 | N3: per_float_feature_quantization border_count=1024 on it_hist_et1_luo and pre2_fe_z, YetiRank | [0, 1] | 267/568 | 17.9 | 2.4 | 0.37828 / 0.30030 | 0.818 / 0.821 | +0.00089 ± 0.00033 / +0.00046 ± 0.00016 | -0.00052 ± 0.00059 / -0.00051 ± 0.00035 | -0.00007 ± 0.00009 / +0.00003 ± 0.00004 | no (vs Yref, seeds [0, 1]) |
| Y_oh64 | N3: one_hot_max_size=64 (category_id one-hot instead of CTR), YetiRank | [0, 1] | 385/535 | 15.3 | 2.4 | 0.37862 / 0.30131 | 0.821 / 0.825 | +0.00097 ± 0.00034 / +0.00043 ± 0.00015 | -0.00017 ± 0.00062 / +0.00051 ± 0.00035 | +0.00001 ± 0.00010 / -0.00000 ± 0.00004 | no (vs Yref, seeds [0, 1]) |
| Y_sel | N3: the 90 features kept by select_features (RecursiveByShapValues), YetiRank | [0, 1] | 293/378 | 14.4 | 2.4 | 0.37705 / 0.29974 | 0.813 / 0.817 | +0.00090 ± 0.00033 / +0.00036 ± 0.00014 | -0.00174 ± 0.00065 / -0.00106 ± 0.00039 | -0.00006 ± 0.00011 / -0.00007 ± 0.00005 | no (vs Yref, seeds [0, 1]) |
| Y_hall | N1x hand-made: 3 LDA columns + 6 KNN teacher-mean columns, YetiRank | [0, 1] | 485/493 | 21.3 | 2.4 | 0.37867 / 0.30073 | 0.835 / 0.839 | +0.00094 ± 0.00034 / +0.00038 ± 0.00014 | -0.00013 ± 0.00068 / -0.00007 ± 0.00039 | -0.00002 ± 0.00013 / -0.00005 ± 0.00006 | no (vs Yref, seeds [0, 1]) |
| R_hknn3 | N1x hand-made: 6 KNN teacher-mean columns, RMSE on the teacher | [0, 1] | 779/689 | 10.2 | 3.4 | 0.38317 / 0.30664 | 0.891 / 0.896 | +0.00044 ± 0.00023 / +0.00033 ± 0.00013 | +0.00002 ± 0.00051 / +0.00031 ± 0.00031 | -0.00014 ± 0.00008 / -0.00001 ± 0.00004 | no (vs Rref, seeds [0, 1]) |

### Scale n6k: one row per arm (mean of its seeds); d = paired per-feed difference to the same-loss reference (seed-paired)

| arm | what | seeds | trees | fit min / seed | model MB | level val / holdout | rho | NATS val / holdout (seed mean; single-seed SE) | d level val / holdout | d NATS val / holdout | winner (>= 2 SE on both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Yref | reference: YetiRank, 177 features, has_time=True (time-ordered feeds, rows shuffled in the feed) | [0, 1] | 293/581 | 4.3 | 2.4 | 0.37368 / 0.29534 | 0.793 / 0.797 | +0.00096 ± 0.00034 / +0.00037 ± 0.00014 | reference | reference | reference |
| Yref_nt | control: YetiRank with CatBoost's default random permutations (has_time=False, file row order) | [0, 1] | 559/189 | 4.2 | 2.4 | 0.37208 / 0.29468 | 0.790 / 0.794 | +0.00083 ± 0.00032 / +0.00032 ± 0.00013 | -0.00161 ± 0.00058 / -0.00066 ± 0.00035 | -0.00013 ± 0.00009 / -0.00006 ± 0.00004 | no (vs Yref, seeds [0, 1]) |
| Rref | reference: RMSE on the centred out-of-fold teacher logit | [0, 1] | 547/693 | 2.5 | 3.4 | 0.38140 / 0.30523 | 0.878 / 0.882 | +0.00048 ± 0.00024 / +0.00031 ± 0.00013 | reference | reference | reference |
| Y_def_pair32 | N1 native: embedding column u*e (32-d PCA), DEFAULT calcers (LDA + KNN:k=5), YetiRank | [0, 1] | 549/482 | 14.4 | 149.9 | 0.37315 / 0.29558 | 0.792 / 0.797 | +0.00091 ± 0.00033 / +0.00039 ± 0.00014 | -0.00053 ± 0.00060 / +0.00024 ± 0.00035 | -0.00005 ± 0.00009 / +0.00001 ± 0.00004 | no (vs Yref, seeds [0, 1]) |
| Y_def_all32 | N1 native: the three vectors (32-d PCA each) as three embedding columns, default calcers (LDA + KNN:k=5), YetiRank | [0] | 542 | 29.1 | 444.9 | 0.37383 / 0.29597 | 0.794 / 0.799 | +0.00108 ± 0.00036 / +0.00041 ± 0.00015 | +0.00056 ± 0.00083 / +0.00110 ± 0.00050 | +0.00010 ± 0.00014 / +0.00004 ± 0.00005 | no (vs Yref, seeds [0]) |
| R_knn50_pair | N1 native: u*e (32-d PCA), KNN:k=50 on the teacher target, RMSE | [0] | 772 | 38.0 | 359.4 | 0.38172 / 0.30547 | 0.880 / 0.885 | +0.00051 ± 0.00025 / +0.00032 ± 0.00013 | +0.00064 ± 0.00075 / +0.00080 ± 0.00044 | +0.00009 ± 0.00009 / +0.00003 ± 0.00005 | no (vs Rref, seeds [0]) |
| R_hknn | N1x hand-made: 2 KNN teacher-mean columns (u*e), RMSE on the teacher | [0, 1] | 540/612 | 2.5 | 3.4 | 0.38147 / 0.30502 | 0.878 / 0.882 | +0.00044 ± 0.00023 / +0.00030 ± 0.00013 | +0.00007 ± 0.00052 / -0.00021 ± 0.00030 | -0.00004 ± 0.00006 / -0.00001 ± 0.00003 | no (vs Rref, seeds [0, 1]) |

## 3. The nats.py numbers of every (arm, seed)

The literal one-line outputs of the tool are in the process logs (`grep NATS nn_x/w12/cb/famN/logs/p*.log`) and in `nn_x/w12/cb/famN/logs/nats_lines.txt`; this table carries the same numbers.

### Scale n30k: the nats.py line of every (arm, seed)

| tag | trees | fit min | s / iter | model MB | level val / holdout | rho | NATS val | NATS holdout | PROMISING (>= 0.0016 both) | member rule 0.9 / 0.1: val, holdout, pooled z | importance of the new columns (PredictionValuesChange, %) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Yref_n30k_s0 | 534 | 15.8 | 1.49 | 2.6 | 0.37845 / 0.30087 | 0.819 / 0.824 | +0.00085 ± 0.00032 | +0.00045 ± 0.00016 | no | -0.00057 ± 0.00040, +0.00003 ± 0.00022, z -0.89 no pass | - |
| Yref_n30k_s1 | 548 | 18.9 | 1.89 | 2.4 | 0.37915 / 0.30073 | 0.820 / 0.825 | +0.00108 ± 0.00036 | +0.00042 ± 0.00015 | no | -0.00002 ± 0.00042, -0.00008 ± 0.00023, z -0.28 no pass | - |
| Rref_n30k_s0 | 716 | 8.9 | 0.65 | 3.5 | 0.38285 / 0.30581 | 0.883 / 0.888 | +0.00057 ± 0.00027 | +0.00034 ± 0.00014 | no | -0.00031 ± 0.00039, +0.00008 ± 0.00021, z -0.29 no pass | - |
| Rref_n30k_s1 | 753 | 9.0 | 0.67 | 3.4 | 0.38346 / 0.30686 | 0.884 / 0.889 | +0.00059 ± 0.00027 | +0.00033 ± 0.00013 | no | +0.00007 ± 0.00038, +0.00002 ± 0.00021, z +0.19 no pass | - |
| Lref_n30k_s0 | 362 | 6.4 | 0.64 | 2.6 | 0.37553 / 0.30000 | 0.806 / 0.818 | +0.00087 ± 0.00033 | +0.00047 ± 0.00016 | no | +0.00018 ± 0.00044, +0.00013 ± 0.00024, z +0.67 no pass | - |
| Lref_n30k_s1 | 310 | 6.5 | 0.65 | 2.6 | 0.37497 / 0.30006 | 0.803 / 0.813 | +0.00103 ± 0.00035 | +0.00044 ± 0.00015 | no | +0.00012 ± 0.00041, +0.00011 ± 0.00023, z +0.52 no pass | - |
| L_lda_all_n30k_s0 | 285 | 10.2 | 1.02 | 2.5 | 0.37462 / 0.29912 | 0.809 / 0.818 | +0.00080 ± 0.00031 | +0.00048 ± 0.00016 | no | -0.00018 ± 0.00041, -0.00014 ± 0.00023, z -0.74 no pass | e_pair 1.25, e_ue 0.78, e_ehm 0.44 |
| L_lda_all_n30k_s1 | 398 | 9.9 | 0.99 | 2.5 | 0.37354 / 0.29938 | 0.806 / 0.820 | +0.00092 ± 0.00033 | +0.00047 ± 0.00016 | no | +0.00027 ± 0.00043, -0.00028 ± 0.00024, z -0.38 no pass | e_pair 1.23, e_ue 0.76, e_ehm 0.37 |
| Y_txt_s_n30k_s0 | 462 | 18.4 | 1.84 | 2.5 | 0.37899 / 0.30073 | 0.819 / 0.823 | +0.00090 ± 0.00033 | +0.00039 ± 0.00015 | no | -0.00039 ± 0.00041, +0.00010 ± 0.00023, z -0.37 no pass | t_cat 1.39, t_clu 0.02 |
| Y_txt_s_n30k_s1 | 486 | 20.6 | 2.06 | 2.5 | 0.37942 / 0.30092 | 0.822 / 0.826 | +0.00097 ± 0.00034 | +0.00043 ± 0.00015 | no | -0.00038 ± 0.00042, +0.00001 ± 0.00022, z -0.63 no pass | t_cat 1.51, t_clu 0.01 |
| Y_txt_l_n30k_s0 | 528 | 29.4 | 2.94 | 2.6 | 0.37836 / 0.30067 | 0.817 / 0.822 | +0.00089 ± 0.00033 | +0.00045 ± 0.00016 | no | -0.00037 ± 0.00042, -0.00007 ± 0.00022, z -0.84 no pass | t_cat 1.47, t_clu 0.02 |
| R_txt_s_n30k_s0 | 759 | 15.1 | 1.13 | 3.4 | 0.38219 / 0.30655 | 0.884 / 0.889 | +0.00052 ± 0.00025 | +0.00035 ± 0.00014 | no | -0.00030 ± 0.00039, +0.00012 ± 0.00021, z -0.15 no pass | t_cat 3.74, t_clu 0.42 |
| Y_q1024_n30k_s0 | 267 | 17.9 | 1.79 | 2.4 | 0.37806 / 0.29985 | 0.816 / 0.819 | +0.00082 ± 0.00031 | +0.00042 ± 0.00015 | no | -0.00013 ± 0.00041, +0.00002 ± 0.00023, z -0.16 no pass | - |
| Y_q1024_n30k_s1 | 568 | 18.0 | 1.80 | 2.4 | 0.37850 / 0.30074 | 0.820 / 0.824 | +0.00097 ± 0.00035 | +0.00050 ± 0.00016 | no | -0.00040 ± 0.00041, +0.00008 ± 0.00022, z -0.45 no pass | - |
| Y_oh64_n30k_s0 | 385 | 15.7 | 1.57 | 2.4 | 0.37885 / 0.30116 | 0.819 / 0.823 | +0.00084 ± 0.00032 | +0.00040 ± 0.00015 | no | -0.00052 ± 0.00040, +0.00005 ± 0.00022, z -0.77 no pass | - |
| Y_oh64_n30k_s1 | 535 | 14.8 | 1.48 | 2.4 | 0.37840 / 0.30146 | 0.822 / 0.827 | +0.00111 ± 0.00037 | +0.00046 ± 0.00016 | no | -0.00011 ± 0.00039, -0.00001 ± 0.00022, z -0.22 no pass | - |
| Y_sel_n30k_s0 | 293 | 15.1 | 1.51 | 2.4 | 0.37705 / 0.29985 | 0.812 / 0.816 | +0.00090 ± 0.00033 | +0.00038 ± 0.00014 | no | -0.00042 ± 0.00040, +0.00000 ± 0.00022, z -0.73 no pass | - |
| Y_sel_n30k_s1 | 378 | 13.6 | 1.36 | 2.4 | 0.37706 / 0.29963 | 0.813 / 0.819 | +0.00090 ± 0.00033 | +0.00034 ± 0.00014 | no | -0.00015 ± 0.00041, -0.00024 ± 0.00022, z -1.02 no pass | - |
| Y_hall_n30k_s0 | 485 | 22.0 | 2.20 | 2.4 | 0.37903 / 0.30079 | 0.835 / 0.839 | +0.00101 ± 0.00035 | +0.00035 ± 0.00014 | no | -0.00020 ± 0.00041, +0.00009 ± 0.00023, z -0.06 no pass | hknn50_ehm 3.49, hlda_pair 1.79, hknn50_pair 1.22, hknn200_ehm 1.17, hknn50_ue 1.16, hknn200_pair 1.08, hknn200_ue 0.67, hlda_ehm 0.63, hlda_ue 0.47 |
| Y_hall_n30k_s1 | 493 | 20.6 | 2.06 | 2.4 | 0.37831 / 0.30068 | 0.835 / 0.839 | +0.00088 ± 0.00033 | +0.00042 ± 0.00015 | no | -0.00057 ± 0.00040, -0.00001 ± 0.00022, z -1.04 no pass | hknn50_ehm 2.95, hlda_pair 1.58, hknn200_ehm 1.38, hknn200_pair 1.29, hknn200_ue 0.95, hknn50_ue 0.91, hknn50_pair 0.85, hlda_ehm 0.65, hlda_ue 0.63 |
| R_hknn3_n30k_s0 | 779 | 10.3 | 0.77 | 3.4 | 0.38308 / 0.30679 | 0.891 / 0.896 | +0.00043 ± 0.00023 | +0.00033 ± 0.00013 | no | -0.00047 ± 0.00037, -0.00006 ± 0.00021, z -1.10 no pass | hknn50_ehm 5.57, hknn200_ehm 1.8, hknn200_pair 1.13, hknn50_ue 1.08, hknn50_pair 0.76, hknn200_ue 0.14 |
| R_hknn3_n30k_s1 | 689 | 10.2 | 0.76 | 3.4 | 0.38327 / 0.30649 | 0.890 / 0.895 | +0.00045 ± 0.00024 | +0.00032 ± 0.00013 | no | -0.00063 ± 0.00039, -0.00000 ± 0.00021, z -1.18 no pass | hknn50_ehm 6.46, hknn50_ue 1.15, hknn200_pair 0.93, hknn50_pair 0.87, hknn200_ehm 0.61, hknn200_ue 0.19 |

### Scale n6k: the nats.py line of every (arm, seed)

| tag | trees | fit min | s / iter | model MB | level val / holdout | rho | NATS val | NATS holdout | PROMISING (>= 0.0016 both) | member rule 0.9 / 0.1: val, holdout, pooled z | importance of the new columns (PredictionValuesChange, %) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Yref_n6k_s0 | 293 | 4.2 | 0.42 | 2.4 | 0.37327 / 0.29487 | 0.791 / 0.794 | +0.00099 ± 0.00035 | +0.00037 ± 0.00014 | no | -0.00017 ± 0.00044, +0.00000 ± 0.00024, z -0.27 no pass | - |
| Yref_n6k_s1 | 581 | 4.4 | 0.44 | 2.4 | 0.37410 / 0.29582 | 0.796 / 0.800 | +0.00094 ± 0.00034 | +0.00038 ± 0.00014 | no | -0.00032 ± 0.00044, +0.00009 ± 0.00023, z -0.22 no pass | - |
| Yref_nt_n6k_s0 | 559 | 4.4 | 0.44 | 2.4 | 0.37240 / 0.29561 | 0.795 / 0.799 | +0.00086 ± 0.00032 | +0.00037 ± 0.00014 | no | -0.00060 ± 0.00044, -0.00015 ± 0.00023, z -1.42 no pass | - |
| Yref_nt_n6k_s1 | 189 | 4.1 | 0.41 | 2.4 | 0.37175 / 0.29375 | 0.786 / 0.788 | +0.00079 ± 0.00031 | +0.00027 ± 0.00012 | no | -0.00017 ± 0.00044, +0.00022 ± 0.00024, z +0.38 no pass | - |
| Rref_n6k_s0 | 547 | 2.5 | 0.19 | 3.4 | 0.38107 / 0.30466 | 0.877 / 0.881 | +0.00043 ± 0.00023 | +0.00029 ± 0.00012 | no | -0.00064 ± 0.00038, +0.00012 ± 0.00021, z -0.78 no pass | - |
| Rref_n6k_s1 | 693 | 2.6 | 0.19 | 3.3 | 0.38173 / 0.30580 | 0.878 / 0.883 | +0.00053 ± 0.00026 | +0.00033 ± 0.00013 | no | -0.00038 ± 0.00039, -0.00021 ± 0.00021, z -1.39 no pass | - |
| Y_def_pair32_n6k_s0 | 549 | 14.9 | 1.49 | 149.9 | 0.37354 / 0.29570 | 0.791 / 0.797 | +0.00099 ± 0.00035 | +0.00040 ± 0.00015 | no | -0.00042 ± 0.00044, +0.00015 ± 0.00023, z -0.23 no pass | e_pair 0.98 |
| Y_def_pair32_n6k_s1 | 482 | 13.8 | 1.38 | 149.9 | 0.37277 / 0.29546 | 0.793 / 0.797 | +0.00083 ± 0.00032 | +0.00037 ± 0.00014 | no | -0.00010 ± 0.00043, -0.00001 ± 0.00023, z -0.20 no pass | e_pair 1.24 |
| Y_def_all32_n6k_s0 | 542 | 29.1 | 2.91 | 444.9 | 0.37383 / 0.29597 | 0.794 / 0.799 | +0.00108 ± 0.00036 | +0.00041 ± 0.00015 | no | -0.00049 ± 0.00044, -0.00002 ± 0.00023, z -0.85 no pass | e_ehm 1.34, e_ue 1.26, e_pair 0.82 |
| R_knn50_pair_n6k_s0 | 772 | 38.0 | 2.85 | 359.4 | 0.38172 / 0.30547 | 0.880 / 0.885 | +0.00051 ± 0.00025 | +0.00032 ± 0.00013 | no | -0.00028 ± 0.00038, +0.00001 ± 0.00021, z -0.48 no pass | e_pair 6.41 |
| R_hknn_n6k_s0 | 540 | 2.6 | 0.19 | 3.4 | 0.38111 / 0.30491 | 0.877 / 0.882 | +0.00045 ± 0.00024 | +0.00030 ± 0.00013 | no | -0.00054 ± 0.00039, -0.00000 ± 0.00021, z -0.99 no pass | hknn200_pair 1.36, hknn50_pair 1.26 |
| R_hknn_n6k_s1 | 612 | 2.4 | 0.18 | 3.4 | 0.38183 / 0.30513 | 0.878 / 0.883 | +0.00043 ± 0.00023 | +0.00029 ± 0.00013 | no | -0.00016 ± 0.00039, -0.00003 ± 0.00021, z -0.41 no pass | hknn200_pair 1.75, hknn50_pair 0.75 |

rho in these tables is the `cbx.Evaluator` value (mean within-feed correlation); the `nats.py` line prints one Pearson over all rows, 0.003 to 0.008 higher on val. A tag that appears twice in `logs/results.jsonl` (`Yref_n30k_s1`, `Y_txt_s_n30k_s0`) has its discarded od100 run first; the last line counts.

## 4. N3 information reads: importance and selection (train-side feeds only, no val / holdout label)

**4.1 LossFunctionChange importance** of the two `Yref` n30k models on the 5,000 selection feeds (728,986 rows; 24 to 27 s per model) [M, `logs/n3_lfc.json`, `logs/n3_lfc.log`]. Unit: change of the library's loss when the feature is removed (for YetiRank the library substitutes a pairwise loss; which one was not verified [I]).

- The two seeds agree: correlation of the 177 values 0.970.
- The ranking differs from `PredictionValuesChange` (the type behind every importance table on record): rank correlation 0.39. Example: `seq_max_all` is rank 7 by loss change and rank 62 by PredictionValuesChange; `ctx_hs_e1k64` is rank 22 against rank 10; `f1_rk` rank 18 against rank 7.
- Concentration: the 20 features with the largest loss change carry 76% of the total positive loss change, 50 features 95%, 90 features 99.6%. 36 of the 177 features have a loss change <= 0 in BOTH seeds; 64 have a mean <= 0.
- Top 12 (mean of the two seeds): `it_hist_et1_luo` 0.00435, `pre2_fe_z` 0.00201, `seq_max_s1` 0.00159, `ctx_nd_knn_u_dev` 0.00153, `seq_max_all_rk` 0.00150, `seq_cv_invd20_rk` 0.00135, `seq_max_all` 0.00119, `seq_cv_pop` 0.00097, `ctx_hs_d7` 0.00092, `ctx_hs_t10` 0.00085, `seq_dcos_hl2` 0.00066, `tte_it_imp` 0.00062. The two "golden" features named in the plan are the top two on this read as well.

**4.2 select_features** (`RecursiveByShapValues`, `shap_calc_type="Approximate"`, 3 steps, 177 to 90 features, 200 trees of the `Yref` configuration per step, fit on the 30,000 fit feeds, eval set = the 5,000 selection feeds; 26.4 minutes) [M, `logs/n3_select.json`, `logs/n3_select.log`].

- Removed 36, then 28, then 23 features. Selection-feed NDCG of the three retrained models (200 trees each): 0.30247 with 177 features, 0.30417 with 141, 0.30355 with 113 (one fit each; the unpaired SE of a 5,000-feed NDCG is about 0.002, so these are equal). The library's own loss graph reads 0.22610 / 0.22865 / 0.22848 at the three retraining points (lower is better) and 0.23280 for the final 90-feature set, which is an estimate, not a retrained model.
- What went: 15 of the 18 `cat_*` features, 9 of the 12 `tte_*` target statistics (all seven of `tte_u_*` / `tte_uc_*` / `tte_ui_*`, `tte_c_ctr` and `tte_it_pos`; the item statistics `tte_it_ctr`, `tte_it_coec`, `tte_it_imp` stay), 8 of the 10 `u_het*_share`, 8 of the 14 `it_hist_*`, 10 of the 23 `cos_*`, `M`, `H`, `f3`, `f5`, `dot_u`, `dot_hall`. What stays: `category_id`, both golden features, 19 of 26 `seq_*`, 17 of 24 `ctx_*`, 4 of 5 `pre2_*`.
- Agreement with 4.1 is weak: 56 of the 90 kept features are among the top 90 by loss change (46 expected by chance); of the 36 features with a non-positive loss change in both seeds only 20 were removed (36 of the 64 with a non-positive mean); three of the top 20 by loss change were removed (`ctx_hs_d7` rank 9, `seq_dcos_s1` rank 14, `tte_c_ctr` rank 16). With 177 correlated columns the SHAP ranking of one 200-tree model is not stable: the list describes redundancy (many columns can stand in for each other); it is not a recommended feature set. One draw.
- The model retrained on the 90 kept features (`Y_sel`, 2 seeds, section 2) answers whether a smaller set changes the information: it does not (d NATS -0.00006 ± 0.00011 / -0.00007 ± 0.00005), and it loses level (-0.00174 ± 0.00065 / -0.00106 ± 0.00039; z -2.7 / -2.8). The removed half is redundant for the pool, not for the tree's own ranking.

## 5. Method

- **Data** [M]: `cbx.load("xm")` (134,118 exploration feeds; the 30K holdout is excluded by construction), the 177 features of the CatBoost of record, `category_id` native categorical.
- **Reduced scale n30k** (`scripts/fn.py: subset`): 30,000 fit feeds drawn with seed 20261115 from the 120,591 train-side FIT feeds (4,397,391 rows, 54,870 positives) and 5,000 selection feeds drawn from the 13,527 train-side SELECTION feeds (728,986 rows) of `cbx.trainside_split` (user-grouped: no user on both sides, asserted). The same two feed sets for every arm and every seed. **n6k** (native KNN only) = the first 6,000 / 1,500 feeds of the same two permutations (879,369 / 219,426 rows), nested in n30k.
- **Model** (the CPU reference of the track, `ts_lr020_s0`): depth 8, `l2_leaf_reg` 5, `border_count` 254, 12 threads, `eval_metric="NDCG:type=Base"` on the selection feeds with click labels. YetiRank lr 0.2; RMSE on the centred **out-of-fold** teacher logit (`teacher("oof", "centred")`) lr 0.2; Logloss on clicks lr 0.1. Generic `catboost.CatBoost(params)` constructor (needed for `embedding_calcers` and the text parameters). `cbx.run_arm` cannot carry derived embedding or text columns, so `fn.run_arm2` repeats its steps (fit on the fit feeds, selection feeds as eval set, `cbx.Budget` callback, prediction at the best iteration, `cbx.Evaluator.evaluate`) with ready pools; test predictions are not written at reduced scale.
- **Tree count, protocol v2** (changed at 02:00Z; no arm-versus-reference difference had been computed, but the level and NATS line of the one od100 text run had been seen): FIXED budgets, YetiRank 600 iterations, RMSE 800, Logloss 600, no overfitting detector; the model is read at the best iteration of the selection feeds inside the budget (train-side selection, no val label anywhere). Reason [M]: with `od_wait=100` on 5,000 selection feeds the first text run stopped at 175 trees while the reference ran to 534 (the selection curve is flat within about ± 0.0005 from tree 200 on), which would have put a tree-count difference into every comparison. `Yref` s0 (best 533 of 634 built) and `Rref` s0 (best 715 of 816 built) are the same models under v2 and were kept; the two discarded od100 runs (`Yref` s1 at 370 trees, `Y_txt_s` s0 at 175 trees) are in `preds/od100/`.
- **Row order of the training pool**: feeds in file order (= time order: the feed timestamps of `xm` are non-decreasing [M]), rows shuffled inside each feed with a fixed seed, `has_time=True` (one permutation in that order). Every ORDERED statistic of CatBoost (category CTR, LDA, KNN) then sees only earlier feeds and a label-independent part of the row's own feed. Speed [M]: no reliable difference to the default four permutations (0.15 against 0.23 s per iteration in a 30-tree dry run at 3,000 feeds, but 0.43 against 0.43 over 600 trees at 6,000 feeds). The control arm `Yref_nt` (default permutations, file row order) reads the effect on the model (section 1).
- **Vectors** (label-free, `fn.Block.vec`): u = user vector / 256, e = candidate item vector / 256, hmean = mean history item vector / 256 (cbx caches). (a) `pair` = u * e element-wise, 128-d (row sum = cosine); (b) `ehm` = e - hmean, 128-d; (c) `ue` = [u ; e], 256-d. Missing vectors are zero rows. For KNN: PCA to 32 dimensions fitted on 3,000 fit feeds (explained variance 0.615 for pair, 0.414 for ue [M]).
- **Token sets** (label-free, `fn.Block.text`): for a candidate row, the candidate's native category id crossed with each DISTINCT category among the user's 20 most recent history events (`t_cat`, e.g. `c15_7 c15_13 c15_15`), and the candidate's 256-cluster (`data/clu_work/cand_cl_256_*`, k-means of the item vectors [D `data/clu_work/kmeans.log`]) crossed with each distinct 256-cluster of those events (`t_clu`); `c15_none` when the feed has no usable event. Tokenizer `Space`, one `Word` dictionary (gram order 1; the default `BiGram` dictionary is meaningless for a set and is off), calcer `BoW`. Setting **s**: `occurrence_lower_bound=50`, `max_dictionary_size=top_tokens_count=500` per column; setting **l**: 20 / 2000 / 2000 (2000 = the library default `top_tokens_count`). Token statistics on 3,000 fit feeds [M]: `t_cat` 3.65 tokens per row, 2,506 distinct tokens, the 500 most frequent cover 80% of the occurrences and the 2000 most frequent 99.5%; `t_clu` 5.0 tokens per row, 44,100 distinct tokens, coverage 16% (500) and 32% (2000).
- **Reads**: (1) the shared SCREEN `nats.py` for every (arm, seed), on val AND on the holdout (section 3); (2) `scripts/read.py`: arm minus the same-loss reference, paired per feed and seed-paired (arm seed s with reference seed s), per-feed differences averaged over the seeds, then mean and SE over feeds: d level = NDCG difference, d NATS = difference of the two-score conditional-logit losses of `nats.py` (the W16-only term cancels). WINNER = d NATS >= 2 SE on val AND on the holdout. The SE is over feeds and does not contain the seed-to-seed variance of a fit, which was measured once: the two seeds of the YetiRank reference differ by +0.00021 ± 0.00014 val / -0.00004 ± 0.00006 holdout in NATS and by +0.00045 ± 0.00085 / -0.00030 ± 0.00051 in level (od100 runs) [M].
- No arm was chosen by a witness. Every arm was fixed before its read; the decisions taken on reads only REMOVED runs after a null (second seed of `Y_txt_l`; `R_txt_l`; the single-vector LDA arms). Three things were ADDED during the work, each declared in `PROGRESS.md` before its run: the n6k scale for native KNN (after the cost probe), the control `Yref_nt`, and `Y_def_all32` (05:21Z, in place of the queued pair-only LDA arm, to cover the two vectors that had no default-calcer arm).

## 6. Cost of the native calcers on CPU [M]

| what | rows | time | per row | note |
|---|---|---|---|---|
| LDA, three embedding columns (128 + 128 + 256 d), online, `has_time` | 4.40M fit + 0.73M selection | `L_lda_all` 1.0 s per iteration against 0.64 for `Lref` (about +4 minutes per fit, pool building 60 s, peak RSS 44 GB against 22) | | cheap |
| KNN:k=5, 32-d PCA, `has_time` (one permutation), dry run | 439,426 fit + 146,324 selection | 179 s | 0.31 ms | single thread; n30k would be 27 to 50 minutes per fit and column |
| LDA + KNN:k=5 (the default calcers), 32-d PCA, n6k, one column | 879,369 fit + 219,426 selection | fit 13.8 to 14.9 minutes against 4.3 for the reference (calcer stage about 10.5 minutes) | 0.6 ms | apply on val + holdout 1.8 min; model 150 MB (the cloud is stored) against 2.4 MB |
| the same with three columns | 879,369 fit + 219,426 selection | fit 29.1 minutes (calcer stage about 26 minutes) | 0.5 ms per column | apply 5.1 min; model 445 MB |
| KNN:k=50, 32-d PCA, `has_time` | 148,516 fit + 44,559 selection | 253 s | 1.31 ms | single thread; apply on val + holdout (6.02M rows) 192 s; model 50 MB |
| KNN:k=50, 32-d PCA, `has_time`, n6k (`R_knn50_pair`) | 879,369 fit + 219,426 selection | fit 38.0 minutes against 2.5 (calcer stage about 36 minutes) | 2.0 ms | apply 6.5 min; model 359 MB |
| KNN:k=200, 32-d PCA, `has_time` | 148,516 fit + 44,559 selection | not finished after 858 s (stopped by exact PID) | more than 4.4 ms | n30k would be more than 6 hours per fit |
| text BoW | per 1000 token features | +0.5 to +0.7 s per iteration at 4.4M rows | | `Y_txt_l` 2.94 s per iteration against 1.5 to 1.9 for the reference |
| hand-made columns (faiss IVF 4096 lists / nprobe 16, 12 threads), k = 200, 32-d | cloud 4.40M rows, 11.15M queries | 7 to 9 minutes per vector | 0.04 ms | `scripts/hx.py`; out-of-fold by user through an id-range selector |

The native KNN cost is about linear in k (the library's online HNSW uses k as the graph degree [D docs_feat 3.1]); the docs_feat figure of 0.7 ms per row was k=5 at 128-d with four permutations. The calcer is single-threaded on CPU, so the 12-thread allowance does not help it.

## 7. Leak statement per arm (rule 6)

All arms: fitting and selection use train feeds of `xm` only (the 30K holdout is not in `xm`, asserted by cbx); val / test / holdout labels are read only by `cbx.Evaluator` and `nats.py` after the predictions exist; `data/feats_quarantine` is not touched; `teacher_mean11` is not used at all (every teacher value is `teacher_oof`).

| arm | why it is clean |
|---|---|
| Yref, Lref, Y_q1024, Y_oh64, Y_sel | the 177 record features only (a subset for `Y_sel`); CatBoost's category CTR is computed on the fit rows, with `has_time` on earlier rows only. The feature subset of `Y_sel` comes from `select_features` on the fit and selection feeds (train side). |
| Yref_nt | as Yref with the record's default random permutations (the setting of the CatBoost of record: category CTR over a random prefix of the fit rows). |
| Rref | as Yref; the regression TARGET is the cross-fitted teacher (`teacher_oof`); nothing NN-derived is an input. |
| Y_txt_s, Y_txt_l, R_txt_s | BoW is label-free: tokens are built from the candidate's category / cluster and the user's history, both available at inference; the dictionary is built on the fit pool only. |
| L_lda_all | the LDA direction is fitted by CatBoost on click labels of the fit pool only; online in time order (`has_time=True`), so a row never sees a later feed (no later snapshot of its user). Not removable on CPU: the prefix contains the earlier rows of the row's OWN feed, in shuffled order (`ctr_history_unit=Group` is GPU-only); the statistic is one 128 / 256-parameter direction shared by all rows, not a per-key statistic, and docs_feat 5.3 measured no learn-loss gain from it on noise labels. Not leave-user-out: 226 of the 30,000 fit feeds share a user with another fit feed; only the EARLIER one enters the later one's prefix. |
| Y_def_pair32, Y_def_all32 (n6k) | as L_lda_all, plus KNN:k=5 on click labels over earlier rows (rows of the own feed's prefix can be among the neighbours: same caveat; docs_feat 5.3 measured no gain on noise labels). |
| R_knn50_pair (n6k) | the KNN target is the pool label = `teacher_oof` (the legal NN-derived input for train rows); neighbours are earlier rows in time order; apply-time neighbours are fit rows only. Same own-feed-prefix caveat. |
| Y_hall, R_hknn3, R_hknn (hand-made columns) | `scripts/hx.py`: click labels and `teacher_oof` values of FIT feeds only; a fit row gets its LDA direction and its neighbours from the fit rows of the OTHER user folds (the teacher's five user folds, `cbx.xfold_of`), so never from its own feed nor from any feed of its own user (no later snapshot either); selection / val / holdout rows use all fit rows (no selection user is in the fit feeds; val / holdout are other feeds by construction). PCA is label-free. Residual caveat [I]: a neighbour's `teacher_oof` value comes from models that excluded the NEIGHBOUR's fold, not the row's, so those models did see the row's own label; this can only make the column look better in training than at inference and does not touch the val / holdout reads. |

## 8. Limits

- Reduced scale: 30,000 of 120,591 fit feeds (22% of the rows). A lever whose value grows with data (a KNN cloud, a token dictionary) is read below its full-scale value. The full-scale re-run was reserved for winners and there was none.
- Two seeds per arm (one where stated). The paired SE is over feeds; the seed-to-seed spread of one fit is of the same size (section 5), so a difference below about 2 SE is not evidence in either direction, and "no-op" means "no effect larger than about 0.001 in level and 0.0002 in NATS", not "exactly zero".
- The tree count selected on 5,000 (n30k) or 1,500 (n6k) selection feeds is noisy (267 against 568 trees for the two seeds of `Y_q1024`, 293 against 581 for `Yref` at n6k); it is part of the seed-to-seed spread of every comparison and is the likely source of the isolated z of about 2 on one witness (`Y_def_all32`, `R_txt_s`, `Yref_nt`).
- Native KNN could not be run at n30k (section 6). Its real-data answer at n30k is the hand-made column (the same statistic; approximate neighbours by faiss IVF instead of the library's HNSW; neighbours out-of-fold by user instead of the online prefix) plus one native arm at n6k.
- The RMSE target here is `teacher_oof` (the S family uses `teacher_mean11`); the RMSE levels are therefore not the S-family levels.
- The W16 pool is fixed; NATS coefficients are fitted on each witness (as `nats.py` does).
- Token sets use the 256-cluster only (1024 / 4096 exist on disk) and the last 20 events only. A larger `top_tokens_count` costs 0.5 to 0.7 s per iteration per 1000 tokens.
- `select_features` was run once (200 trees per training, 177 to 90 features in 3 steps, approximate SHAP); it is one draw.
- GPU-only items (`ctr_history_unit=Group`, embedding calcers on GPU) were not touched (rule).

## 9. Exact commands

```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
  POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 CBX_CACHE_EXTRA=<repo>/nn_x/w12/cb/famN/cache
S=nn_x/w12/cb/famN/scripts; L=nn_x/w12/cb/famN/logs       # every fit: setsid nohup ... > $L/<name>.log 2>&1 < /dev/null &
# dry runs (3,000 / 1,000 feeds, 30 trees) and cost probes
.venv/bin/python -u $S/run.py --arms Yref Yref_nt Lref Rref --seeds 0 --nfit 3000 --nsel 1000 --iters 30 --dry
.venv/bin/python -u $S/run.py --arms Y_txt_s R_txt_l L_lda_pair Y_def_pair32 --seeds 0 --nfit 3000 --nsel 1000 --iters 30 --dry
.venv/bin/python -u $S/run.py --arms R_knn50_pair R_knn200_pair --seeds 0 --nfit 1000 --nsel 300 --iters 30 --dry
CBX_THREADS=4 .venv/bin/python -u $S/hx.py --nfit 1000 --nsel 300 --kinds pair --nlist 256 --nprobe 8
# references and arms at n30k (protocol v2 = the defaults of run.py)
.venv/bin/python -u $S/run.py --arms Rref Lref Yref --seeds 1 0            # Yref s0 / Rref s0 come from the first launch (od_wait 100; the same models)
.venv/bin/python -u $S/run.py --arms Y_txt_s Y_txt_l --seeds 0 1           # stopped by PID before Y_txt_l s1
.venv/bin/python -u $S/run.py --arms R_txt_s --seeds 0
.venv/bin/python -u $S/hx.py --nfit 30000 --nsel 5000 --nlist 4096 --nprobe 16
.venv/bin/python -u $S/run.py --arms Y_hall R_hknn3 L_lda_all --seeds 0 1
.venv/bin/python -u $S/run.py --arms Y_q1024 Y_oh64 --seeds 0 1
.venv/bin/python -u $S/run.py --arms Y_sel --seeds 0 ; .venv/bin/python -u $S/run.py --arms Y_sel --seeds 1     # after n3.py select
# native KNN scale n6k
.venv/bin/python -u $S/hx.py --nfit 6000 --nsel 1500 --kinds pair --what knn --nlist 1024 --nprobe 16
.venv/bin/python -u $S/run.py --nfit 6000 --nsel 1500 --arms Yref Rref R_hknn Y_def_pair32 --seeds 0 1
.venv/bin/python -u $S/run.py --nfit 6000 --nsel 1500 --arms R_knn50_pair --seeds 0
.venv/bin/python -u $S/run.py --nfit 6000 --nsel 1500 --arms Yref_nt --seeds 0 1
CBX_THREADS=4 .venv/bin/python -u $S/run.py --arms Y_def_all32 --seeds 0 --nfit 300 --nsel 100 --iters 5 --dry    # dry run
.venv/bin/python -u $S/run.py --nfit 6000 --nsel 1500 --arms Y_def_all32 --seeds 0
# N3 information reads
.venv/bin/python -u $S/n3.py lfc ; .venv/bin/python -u $S/n3.py select 200 90
# reads and tables (4 threads)
CBX_THREADS=4 .venv/bin/python $S/read.py --n 30 ; CBX_THREADS=4 .venv/bin/python $S/read.py --n 6
CBX_THREADS=4 .venv/bin/python $S/mktable.py 30 ; CBX_THREADS=4 .venv/bin/python $S/mktable.py 6
bash $S/mklines.sh > $L/nats_lines.txt ; CBX_THREADS=4 .venv/bin/python $S/mkreport.py     # this file = report_head.txt + tables + report_n3.txt + report_static.txt
# the shared screen on one arm from the shell
.venv/bin/python nn_x/w12/cb/nats.py Yref_n30k_s0 nn_x/w12/cb/famN/preds/Yref_n30k_s0_val.npy nn_x/w12/cb/famN/preds/Yref_n30k_s0_hold.npy
```

## 10. Files (all under `<repo>/nn_x/w12/cb/famN/`)

- `PROGRESS.md`: log of the work with times.
- `scripts/fn.py` (subset, row blocks, vectors, token sets, pools, `run_arm2`), `scripts/run.py` (arm registry and runner), `scripts/hx.py` (hand-made LDA / KNN columns), `scripts/read.py` (paired reads), `scripts/mktable.py`, `scripts/mklines.sh`, `scripts/mkreport.py` and the three text parts `scripts/report_{head,n3,static}.txt` (this report is assembled from them), `scripts/n3.py` (LossFunctionChange, select_features, parameter checks), `scripts/chain*.sh` (the queues that were run; `chainD.sh` and `chainE.sh` were stopped while waiting and replaced by `chainD2.sh` and `chainE2.sh`).
- `preds/<tag>_{val,hold}.npy`, `<tag>.eval.json` (cbx.Evaluator), `<tag>.curve.json`, `<tag>.cbm` (dropped when above 300 MB); tag = `<arm>_n<K>k_s<seed>`. `preds/od100/` = the two discarded runs of protocol v1. `preds/dry/` = dry runs. Test predictions exist only for the four reference runs `Yref_n30k_s{0,1}` and `Rref_n30k_s{0,1}` (`<tag>_test.npy`, written afterwards from the saved models, which reproduce the saved val predictions exactly); they are reduced-scale models and no member candidates.
- `logs/results.jsonl` (one line per arm and seed: the nats.py numbers, trees, fit time, model size, importances), `logs/results_dry.jsonl`, `logs/compare_n30k.json`, `logs/compare_n6k.json`, `logs/read_n30k.txt`, `logs/read_n6k.txt`, `logs/nats_lines.txt`, `logs/ref_bundle_nats.txt`, `logs/hx_raw_columns_nats.txt`, `logs/hx_redundancy.txt`, `logs/n3_lfc.json`, `logs/n3_select.json`, the process logs `logs/p*.log`, `logs/hx_*.log`, `logs/n3_*.log`.
- `cache/hx/n30k/`, `cache/hx/n6k/`: the hand-made columns; `cache/pca32_*.npz`; `cache/nats/`: per-feed reads.

## 11. What I did NOT run (stated plainly)

- **No full-scale (134K-feed) run and no test prediction of any lever arm**: the protocol reserves them for winners and there is none. Every number above is at 30,000 or 6,000 fit feeds.
- **No third seed**: nothing is near the screen. `Y_txt_l`, `R_txt_s`, `R_knn50_pair`, `Y_def_all32` have ONE seed; every other arm has two.
- **Native KNN**: k=200 was not run (the cost probe did not finish 193K rows in 14 minutes); k=50 was run for the pair vector only, at 6,000 feeds only, one seed. For e - hmean and [u ; e] the KNN-on-teacher statistic exists only as the hand-made columns (k = 50 and 200, 30,000 feeds). Native KNN at 128 dimensions was not run (32-d PCA everywhere).
- **Native LDA**: only the three embedding columns together under Logloss (`L_lda_all`); the three single-column arms were not run after the joint arm read null (the pair-only arm was dry-run on 3,000 feeds). LDA with `components` > 1 or `likelihood=true`, and LDA alone under YetiRank, were not run.
- **Default calcers under YetiRank**: at 6,000 feeds and 32-d PCA only (pair vector 2 seeds; the three vectors together 1 seed); not at 30,000 feeds, not at 128 dimensions, not per single vector for e - hmean and [u ; e].
- **Hand-made columns**: one YetiRank arm with all nine columns and one RMSE arm with the six KNN columns; the separate LDA-only and KNN-only YetiRank arms of the registry (`Y_hlda`, `Y_hknn`, `Y_hknn3`) were not run.
- **Text**: `R_txt_l` (2000 tokens under RMSE) and the second seed of `Y_txt_l` were not run after the nulls; single-column arms (`Y_txtcat_l`, `Y_txtclu_l`) not run; tokens from the 1024 / 4096 clusters, from event types, or from more than the last 20 events were not built; no BiGram dictionary; the supervised text calcers (NaiveBayes, BM25) were not run (they need a classification loss and leak inside the feed on CPU, docs_feat 4).
- **Quantisation**: only 1024 borders on the two named features; no other border type or count. **Selection**: one `select_features` run (one algorithm, 3 steps, 200 trees per step).
- **`has_time` control**: at 6,000 feeds only.
- Nothing on GPU; no bundle and no member-rule read beyond the line that `nats.py` prints for each run (no arm was PROMISING, so no bundle was declared).

## 12. Appendix: the literal `nats.py` lines

```
# protocol v2 runs (the runs of the report)
Rref_n30k_s1: level 0.38346 / 0.30686 (W16 0.40144 / 0.32460) | rho 0.891 / 0.890 | NATS +0.00059+-0.00027 / +0.00033+-0.00013 (beta_X z +4.4 / +4.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val +0.00007+-0.00038, holdout +0.00002+-0.00021, pooled z +0.19 -> no pass
Lref_n30k_s1: level 0.37497 / 0.30006 (W16 0.40144 / 0.32460) | rho 0.812 / 0.814 | NATS +0.00103+-0.00035 / +0.00044+-0.00015 (beta_X z +5.8 / +5.7) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val +0.00012+-0.00041, holdout +0.00011+-0.00023, pooled z +0.52 -> no pass
Lref_n30k_s0: level 0.37553 / 0.30000 (W16 0.40144 / 0.32460) | rho 0.817 / 0.819 | NATS +0.00087+-0.00033 / +0.00047+-0.00016 (beta_X z +5.3 / +5.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val +0.00018+-0.00044, holdout +0.00013+-0.00024, pooled z +0.67 -> no pass
Yref_n30k_s1: level 0.37915 / 0.30073 (W16 0.40144 / 0.32460) | rho 0.824 / 0.826 | NATS +0.00108+-0.00036 / +0.00042+-0.00015 (beta_X z +5.9 / +5.6) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00002+-0.00042, holdout -0.00008+-0.00023, pooled z -0.28 -> no pass
Y_txt_s_n30k_s0: level 0.37899 / 0.30073 (W16 0.40144 / 0.32460) | rho 0.823 / 0.824 | NATS +0.00090+-0.00033 / +0.00039+-0.00015 (beta_X z +5.4 / +5.4) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00039+-0.00041, holdout +0.00010+-0.00023, pooled z -0.37 -> no pass
Y_txt_s_n30k_s1: level 0.37942 / 0.30092 (W16 0.40144 / 0.32460) | rho 0.825 / 0.826 | NATS +0.00097+-0.00034 / +0.00043+-0.00015 (beta_X z +5.6 / +5.6) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00038+-0.00042, holdout +0.00001+-0.00022, pooled z -0.63 -> no pass
Y_txt_l_n30k_s0: level 0.37836 / 0.30067 (W16 0.40144 / 0.32460) | rho 0.820 / 0.822 | NATS +0.00089+-0.00033 / +0.00045+-0.00016 (beta_X z +5.4 / +5.8) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00037+-0.00042, holdout -0.00007+-0.00022, pooled z -0.84 -> no pass
R_txt_s_n30k_s0: level 0.38219 / 0.30655 (W16 0.40144 / 0.32460) | rho 0.891 / 0.890 | NATS +0.00052+-0.00025 / +0.00035+-0.00014 (beta_X z +4.1 / +5.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00030+-0.00039, holdout +0.00012+-0.00021, pooled z -0.15 -> no pass
Y_hall_n30k_s0: level 0.37903 / 0.30079 (W16 0.40144 / 0.32460) | rho 0.838 / 0.839 | NATS +0.00101+-0.00035 / +0.00035+-0.00014 (beta_X z +5.7 / +5.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00020+-0.00041, holdout +0.00009+-0.00023, pooled z -0.06 -> no pass
Y_hall_n30k_s1: level 0.37831 / 0.30068 (W16 0.40144 / 0.32460) | rho 0.840 / 0.840 | NATS +0.00088+-0.00033 / +0.00042+-0.00015 (beta_X z +5.4 / +5.5) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00057+-0.00040, holdout -0.00001+-0.00022, pooled z -1.04 -> no pass
R_hknn3_n30k_s0: level 0.38308 / 0.30679 (W16 0.40144 / 0.32460) | rho 0.898 / 0.897 | NATS +0.00043+-0.00023 / +0.00033+-0.00013 (beta_X z +3.7 / +4.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00047+-0.00037, holdout -0.00006+-0.00021, pooled z -1.10 -> no pass
R_hknn3_n30k_s1: level 0.38327 / 0.30649 (W16 0.40144 / 0.32460) | rho 0.897 / 0.896 | NATS +0.00045+-0.00024 / +0.00032+-0.00013 (beta_X z +3.8 / +4.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00063+-0.00039, holdout -0.00000+-0.00021, pooled z -1.18 -> no pass
L_lda_all_n30k_s0: level 0.37462 / 0.29912 (W16 0.40144 / 0.32460) | rho 0.817 / 0.818 | NATS +0.00080+-0.00031 / +0.00048+-0.00016 (beta_X z +5.1 / +5.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00018+-0.00041, holdout -0.00014+-0.00023, pooled z -0.74 -> no pass
L_lda_all_n30k_s1: level 0.37354 / 0.29938 (W16 0.40144 / 0.32460) | rho 0.818 / 0.821 | NATS +0.00092+-0.00033 / +0.00047+-0.00016 (beta_X z +5.5 / +5.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val +0.00027+-0.00043, holdout -0.00028+-0.00024, pooled z -0.38 -> no pass
Y_q1024_n30k_s0: level 0.37806 / 0.29985 (W16 0.40144 / 0.32460) | rho 0.819 / 0.819 | NATS +0.00082+-0.00031 / +0.00042+-0.00015 (beta_X z +5.2 / +5.6) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00013+-0.00041, holdout +0.00002+-0.00023, pooled z -0.16 -> no pass
Y_q1024_n30k_s1: level 0.37850 / 0.30074 (W16 0.40144 / 0.32460) | rho 0.824 / 0.825 | NATS +0.00097+-0.00035 / +0.00050+-0.00016 (beta_X z +5.6 / +6.0) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00040+-0.00041, holdout +0.00008+-0.00022, pooled z -0.45 -> no pass
Y_oh64_n30k_s0: level 0.37885 / 0.30116 (W16 0.40144 / 0.32460) | rho 0.823 / 0.824 | NATS +0.00084+-0.00032 / +0.00040+-0.00015 (beta_X z +5.2 / +5.4) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00052+-0.00040, holdout +0.00005+-0.00022, pooled z -0.77 -> no pass
Y_oh64_n30k_s1: level 0.37840 / 0.30146 (W16 0.40144 / 0.32460) | rho 0.826 / 0.827 | NATS +0.00111+-0.00037 / +0.00046+-0.00016 (beta_X z +6.0 / +5.8) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00011+-0.00039, holdout -0.00001+-0.00022, pooled z -0.22 -> no pass
Y_sel_n30k_s0: level 0.37705 / 0.29985 (W16 0.40144 / 0.32460) | rho 0.814 / 0.816 | NATS +0.00090+-0.00033 / +0.00038+-0.00014 (beta_X z +5.4 / +5.3) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00042+-0.00040, holdout +0.00000+-0.00022, pooled z -0.73 -> no pass
Y_sel_n30k_s1: level 0.37706 / 0.29963 (W16 0.40144 / 0.32460) | rho 0.817 / 0.819 | NATS +0.00090+-0.00033 / +0.00034+-0.00014 (beta_X z +5.4 / +5.0) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00015+-0.00041, holdout -0.00024+-0.00022, pooled z -1.02 -> no pass
Yref_n6k_s0: level 0.37327 / 0.29487 (W16 0.40144 / 0.32460) | rho 0.794 / 0.794 | NATS +0.00099+-0.00035 / +0.00037+-0.00014 (beta_X z +5.7 / +5.2) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00017+-0.00044, holdout +0.00000+-0.00024, pooled z -0.27 -> no pass
Yref_n6k_s1: level 0.37410 / 0.29582 (W16 0.40144 / 0.32460) | rho 0.800 / 0.800 | NATS +0.00094+-0.00034 / +0.00038+-0.00014 (beta_X z +5.5 / +5.3) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00032+-0.00044, holdout +0.00009+-0.00023, pooled z -0.22 -> no pass
Rref_n6k_s0: level 0.38107 / 0.30466 (W16 0.40144 / 0.32460) | rho 0.883 / 0.882 | NATS +0.00043+-0.00023 / +0.00029+-0.00012 (beta_X z +3.7 / +4.6) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00064+-0.00038, holdout +0.00012+-0.00021, pooled z -0.78 -> no pass
Rref_n6k_s1: level 0.38173 / 0.30580 (W16 0.40144 / 0.32460) | rho 0.885 / 0.884 | NATS +0.00053+-0.00026 / +0.00033+-0.00013 (beta_X z +4.1 / +4.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00038+-0.00039, holdout -0.00021+-0.00021, pooled z -1.39 -> no pass
R_hknn_n6k_s0: level 0.38111 / 0.30491 (W16 0.40144 / 0.32460) | rho 0.884 / 0.883 | NATS +0.00045+-0.00024 / +0.00030+-0.00013 (beta_X z +3.8 / +4.7) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00054+-0.00039, holdout -0.00000+-0.00021, pooled z -0.99 -> no pass
R_hknn_n6k_s1: level 0.38183 / 0.30513 (W16 0.40144 / 0.32460) | rho 0.885 / 0.884 | NATS +0.00043+-0.00023 / +0.00029+-0.00013 (beta_X z +3.7 / +4.7) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00016+-0.00039, holdout -0.00003+-0.00021, pooled z -0.41 -> no pass
Y_def_pair32_n6k_s0: level 0.37354 / 0.29570 (W16 0.40144 / 0.32460) | rho 0.797 / 0.798 | NATS +0.00099+-0.00035 / +0.00040+-0.00015 (beta_X z +5.7 / +5.4) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00042+-0.00044, holdout +0.00015+-0.00023, pooled z -0.23 -> no pass
Y_def_pair32_n6k_s1: level 0.37277 / 0.29546 (W16 0.40144 / 0.32460) | rho 0.797 / 0.798 | NATS +0.00083+-0.00032 / +0.00037+-0.00014 (beta_X z +5.2 / +5.2) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00010+-0.00043, holdout -0.00001+-0.00023, pooled z -0.20 -> no pass
R_knn50_pair_n6k_s0: level 0.38172 / 0.30547 (W16 0.40144 / 0.32460) | rho 0.887 / 0.886 | NATS +0.00051+-0.00025 / +0.00032+-0.00013 (beta_X z +4.1 / +4.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00028+-0.00038, holdout +0.00001+-0.00021, pooled z -0.48 -> no pass
Yref_nt_n6k_s0: level 0.37240 / 0.29561 (W16 0.40144 / 0.32460) | rho 0.798 / 0.799 | NATS +0.00086+-0.00032 / +0.00037+-0.00014 (beta_X z +5.3 / +5.2) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00060+-0.00044, holdout -0.00015+-0.00023, pooled z -1.42 -> no pass
Yref_nt_n6k_s1: level 0.37175 / 0.29375 (W16 0.40144 / 0.32460) | rho 0.789 / 0.789 | NATS +0.00079+-0.00031 / +0.00027+-0.00012 (beta_X z +5.1 / +4.5) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00017+-0.00044, holdout +0.00022+-0.00024, pooled z +0.38 -> no pass
Y_def_all32_n6k_s0: level 0.37383 / 0.29597 (W16 0.40144 / 0.32460) | rho 0.798 / 0.799 | NATS +0.00108+-0.00036 / +0.00041+-0.00015 (beta_X z +5.9 / +5.5) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00049+-0.00044, holdout -0.00002+-0.00023, pooled z -0.85 -> no pass
Yref_n30k_s0: level 0.37845 / 0.30087 (W16 0.40144 / 0.32460) | rho 0.823 / 0.824 | NATS +0.00085+-0.00032 / +0.00045+-0.00016 (beta_X z +5.2 / +5.7) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00057+-0.00040, holdout +0.00003+-0.00022, pooled z -0.89 -> no pass
Rref_n30k_s0: level 0.38285 / 0.30581 (W16 0.40144 / 0.32460) | rho 0.890 / 0.889 | NATS +0.00057+-0.00027 / +0.00034+-0.00014 (beta_X z +4.3 / +5.0) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00031+-0.00039, holdout +0.00008+-0.00021, pooled z -0.29 -> no pass
# discarded od100 runs (protocol v1; predictions in preds/od100/)
Yref_n30k_s1: level 0.37890 / 0.30057 (W16 0.40144 / 0.32460) | rho 0.820 / 0.821 | NATS +0.00106+-0.00036 / +0.00041+-0.00015 (beta_X z +5.9 / +5.5) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00006+-0.00042, holdout -0.00014+-0.00022, pooled z -0.53 -> no pass
Y_txt_s_n30k_s0: level 0.37728 / 0.29911 (W16 0.40144 / 0.32460) | rho 0.813 / 0.813 | NATS +0.00085+-0.00032 / +0.00030+-0.00013 (beta_X z +5.3 / +4.7) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00021+-0.00041, holdout -0.00001+-0.00022, pooled z -0.38 -> no pass
# references read with the tool
bundle(Yref_n30k_s0,s1): level 0.37995 / 0.30194 (W16 0.40144 / 0.32460) | rho 0.827 / 0.829 | NATS +0.00100+-0.00035 / +0.00045+-0.00016 (beta_X z +5.7 / +5.8) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00016+-0.00041, holdout -0.00002+-0.00022, pooled z -0.32 -> no pass
ts_lr020_s0(full scale, infra): level 0.38150 / 0.30462 (W16 0.40144 / 0.32460) | rho 0.847 / 0.847 | NATS +0.00117+-0.00038 / +0.00051+-0.00017 (beta_X z +6.1 / +6.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00006+-0.00039, holdout -0.00008+-0.00022, pooled z -0.36 -> no pass
```
