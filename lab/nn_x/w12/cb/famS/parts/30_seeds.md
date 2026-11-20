## 3. The best arm at three seeds, its bundle, and its plateau [M]

Best arm = **S1f** (RMSE on the centred teacher logit, depth 8, lr 0.3, tree cap T0 = 886), chosen at 04:31Z without val or holdout labels by the rule written before any arm (highest train-side selection NDCG; arms within 0.0005 of the top are tied and the cheapest is taken): S3 0.31200 | S1f 0.31165 | S1 0.31133 (not eligible: 135 min per seed) | S4b 0.31110 | S2 0.30989 | S4 0.30731. S4b, whose val / holdout level is the highest of the single models, is outside the tie band and was not taken.

### 3.1 Seeds and the declared bundle

| model | trees used | sel NDCG | val | holdout | rho | NATS val | NATS holdout | member rule 0.9 / 0.1: val, holdout, pooled z |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 858 | 0.31165 | 0.38404 | 0.30737 | 0.897 / 0.897 | +0.00062 ± 0.00028 | +0.00036 ± 0.00014 | -0.00031 ± 0.00038, +0.00017 ± 0.00020, z +0.02 |
| seed 1 (first 886 trees of the long run) | 884 | 0.31195 | 0.38326 | 0.30696 | 0.897 / 0.897 | +0.00045 ± 0.00024 | +0.00040 ± 0.00015 | -0.00021 ± 0.00038, +0.00012 ± 0.00020, z +0.03 |
| seed 2 | 882 | 0.31194 | 0.38385 | 0.30765 | 0.898 / 0.898 | +0.00054 ± 0.00026 | +0.00041 ± 0.00015 | -0.00009 ± 0.00037, -0.00003 ± 0.00020, z -0.28 |
| mean of the seeds (sd over seeds) | | | 0.38372 (0.00040) | 0.30733 (0.00035) | | +0.00053 | +0.00039 | |
| **bundle of the 3 seeds** (member-equal z-mean, declared 01:25Z) | | | **0.38461** | **0.30797** | 0.900 / 0.900 | **+0.00056 ± 0.00026** | **+0.00040 ± 0.00015** | **-0.00027 ± 0.00037, -0.00000 ± 0.00020, z -0.52: no pass** |

Paired reads (val | holdout | pooled z):
- seed 1 - seed 0: -0.00078 ± 0.00072 | -0.00040 ± 0.00042 | -1.44; seed 2 - seed 0: -0.00019 ± 0.00073 | +0.00029 ± 0.00043 | +0.30; seed 2 - seed 1: +0.00059 ± 0.00074 | +0.00069 ± 0.00042 | +1.73. One seed moves the level by 0.0004 (sd), as the rules warn.
- bundle - seed 0: +0.00058 ± 0.00053 | +0.00060 ± 0.00030 | +2.20 (bundle minus the mean of the seeds: +0.0009 / +0.0006). Three seeds are worth about +0.0006 to +0.0009 of level and **nothing in NATS** (0.00056 / 0.00040 against 0.00053 / 0.00039 for one seed): averaging removes seed noise, and seed noise is not information about the label.
- bundle - hard-label reference: +0.00311 ± 0.00097 | +0.00335 ± 0.00056 | +6.51. NATS(bundle) - NATS(hard-label reference), paired: -0.00062 ± 0.00022 | -0.00011 ± 0.00009 | -2.77.
- bundle - S4b (one model, 55 min): +0.00005 ± 0.00079 | -0.00064 ± 0.00043 | -1.00; bundle - S1 (one model, 135 min): +0.00008 ± 0.00055 | +0.00035 ± 0.00031 | +0.90.
- bundle - LightGBM student K (134K feeds, 600 rounds): -0.00022 ± 0.00071 | +0.00036 ± 0.00040 | +0.41; bundle - LightGBM student KL0 (60K feeds, 2,400 rounds, its plateau): **-0.00152 ± 0.00077 | -0.00166 ± 0.00044 | -4.08**.
- On val M >= 50 alone (8,530 feeds): bundle level 0.30557, NATS +0.00034 ± 0.00025, member rule -0.00009 ± 0.00041.

### 3.2 Plateau of the best arm (seed 1 run to 2 x T0; declared in addendum 2 (c) before the run)

1,761 trees were built in the 112-minute guard (cap 1,772); best train-side selection iteration 1,608. Selection NDCG: 0.30869 / 0.31023 / 0.31116 / 0.31175 / 0.31211 / 0.31220 / 0.31230 / 0.31267 at 300 / 500 / 700 / 886 / 1,000 / 1,200 / 1,400 / 1,600 trees, best 0.31281, last 0.31271.

| trees (all trees up to N) | 300 | 500 | 700 | 886 | 1,000 | 1,200 | 1,400 | 1,608 (selected) | 1,761 |
|---|---|---|---|---|---|---|---|---|---|
| val | 0.38103 | 0.38196 | 0.38287 | 0.38322 | 0.38349 | 0.38407 | 0.38473 | 0.38451 | 0.38460 |
| holdout | 0.30507 | 0.30597 | 0.30658 | 0.30703 | 0.30716 | 0.30763 | 0.30791 | 0.30792 | 0.30801 |

- 1,200 - 886 trees: +0.00085 ± 0.00034 | +0.00060 ± 0.00019 | z +3.93; 1,400 - 886: +0.00151 ± 0.00042 | +0.00088 ± 0.00022 | z +5.37; **1,761 - 1,400: -0.00013 ± 0.00033 | +0.00010 ± 0.00017 | z +0.14**. The arm was round-limited at 886 trees by about +0.0013 / +0.0010 (selected model against the 886-tree one: +0.00125 ± 0.00047 | +0.00096 ± 0.00025) and is flat from about 1,400 trees.
- **Plateau of a CatBoost RMSE student on the 177 features: val 0.3845 to 0.3847, holdout 0.3079 to 0.3080**, about 90 minutes of 12 threads. NATS at the plateau +0.00049 ± 0.00025 / +0.00043 ± 0.00015 (886 trees of the same seed: +0.00045 / +0.00040): more trees do not add pool-independent information. Member rule -0.00028 ± 0.00037 / +0.00007 ± 0.00020, pooled z -0.29.
- The plateau model against the others: - seed 0 at 858 trees +0.00047 ± 0.00076 | +0.00056 ± 0.00043; - S1 (lr 0.1, 2,014 trees) -0.00002 ± 0.00072 | +0.00030 ± 0.00041; - S4b -0.00005 ± 0.00086 | -0.00068 ± 0.00048; - the 3-seed bundle -0.00011 ± 0.00059 | -0.00004 ± 0.00032; - LightGBM K (600 rounds) -0.00033 ± 0.00079 | +0.00032 ± 0.00044; - LightGBM KL0 (plateau) -0.00162 ± 0.00082 | -0.00171 ± 0.00045 (z -4.08).
- Reading [M for the numbers, I for the interpretation]: four different ways of spending CPU on a CatBoost student (lr 0.1 x 2,014 trees, lr 0.3 x 1,608 trees, three seeds x 886 trees, the listwise soft-label loss x 497 trees) end at the same val level, 0.3845 to 0.3846. The LightGBM student at its plateau is 0.0015 / 0.0017 higher, so nothing here points to a CatBoost-specific reserve on these inputs: its symmetric trees stop a little below LightGBM's leaf-wise trees; S4b, the only arm still rising at its cap, is the one place where another +0.001 on the holdout may be left.

### 3.3 Stage-2 gate

val >= 0.3900 is **not met**: the best CatBoost student reads 0.38461 (bundle), 0.38456 (S4b), 0.38453 (S1), 0.38451 (plateau run): 0.0054 short. Out-of-fold tree scores for the training rows were therefore not produced.
