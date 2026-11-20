# witness: staleness witnesses as tools, their calibration on the public readings, and the READ of the re-opened arm hz

Written incrementally on 2026-11-15 by the witness agent (KEY = witness). CPU only. Work directory `nn_x/w12/stale/witness/`. Conventions: ± = 1 SE paired per feed, feed count given; [M] measured by me, [D] read in a project file, [I] inferred. "Subset" = in-period holdout feeds with day >= Sep 13 (13,424 feeds, 2,089,648 rows), the only part of the holdout where the label-free parity gate of the lagged view holds. Pools = member mean of per-feed z, train-only level.

STATUS OF THIS FILE: complete (closed 2026-11-15 07:20Z). The first run (02:18-03:22Z) was cut by an account limit; resumed 03:31Z from `PROGRESS.md`, the caches re-checked first (`logs/recheck_cache_0335Z.txt`: 36 lagged and 35 val files intact, control A numbers reproduced from the cache). Sections: 1 tool A, 2 tool B, 3 calibration, 4 commands, 5 files, 6 the hz read, 7 limits.

## 0. Verdict

1. **hz READ (16 of 16 snapshot pairs, read once at 06:31Z, selected checkpoints): no PASS, CLOSED AGAIN under KD by the rule as written.** val -0.00009 ± 0.00040 (clause val >= 0 fails by 0.2 SE), stale-val +0.00113 ± 0.00043 (+2.60 SE), val M<50 -0.00025 ± 0.00072; in-period holdout -0.00076 ± 0.00022; lagged holdout lag 0 -0.00009 ± 0.00030, private-like +0.00042 ± 0.00034 [M].
2. The verdict is fragile in the letter and robust in substance: at @last the same rule would PASS (val +0.00024, stale-val +2.57 SE), but the ref pool is a low draw under stale inputs (ref - originals -0.00056 ± 0.00030, ref2 - ref +0.00046 ± 0.00031); against ref2, the originals or all 48 same-recipe members hz reads stale-val +0.0006 to +0.0007 (1.2-1.5 SE), val -0.0002 to 0.0000, private-like lagged holdout -0.00011 ± 0.00037 [M, post-hoc].
3. The two staleness levers mostly SUBSTITUTE: on top of the popularity-light admixture hz keeps 43% of its stale-val effect (+0.00049 ± 0.00039) and nothing on val; on a 16-member base the admixture itself is flat on val (+0.00008 ± 0.00032) and on the private-like lagged holdout (-0.00014 ± 0.00030), against +0.00055 / +0.00024 on the 8-member W8 that was scored [M].
4. Calibration on the 4 public pairs (my train-only pools are exactly the train-only halves of the scored files, r = 1.00000): val X2 0.29, SV2 0.33, stale-val 0.38, pmix-public 0.67 (1.7x the public size), pmix-private 0.99, in-period holdout 2.44 with the wrong sign on both staleness pairs; on the sum of the two staleness pairs public is +0.00116, val +0.00123, stale-val +0.00117, holdout -0.00136 (2.7 SE from public). The private-like lagged holdout scores 0.20 on its three pairs [M].
5. **Recommended decision statistic, by my pre-declared rule: plain val of the train-only pools.** SE on a 16-vs-16 contrast: 0.00029 for same-recipe pools paired by snapshot, 0.00040 as measured on hz - ref, 0.00038 for independent seeds [I]. My opinion, not the rule: SV2 = per-feed mean of the val and stale-val deltas (SE 0.00024 / 0.00034 / 0.00032), because public sits 46% and private 73% of the way from val to stale-val on the label-free cold-share axis; four pairs cannot tell the two apart.
6. Tools work [M]: lagged holdout (`lagged_score.py`; reviewer reproduced: W8 pool 0.32334 -> 0.32002, S_lag +0.00024 ± 0.00031; 6.5 CPU min per member for lag 0 + private-like under the early load, 4.1 on average, 2.6 at the end) and stale-val on CPU (`stale_val_cpu.py`; 2.4 / 1.8 / 1.0 min per member); `wt_read.py pair A B` prints every witness; it reproduces `arm_read8.py` to the digit. Control b2 of the stale-val tool failed by the letter (1.04 SE against my 1 SE bar), the re-sized control B' is valid: the lead decides.
7. Side findings [M]: every student pool loses 0.0026-0.0033 NDCG under the private-like lag, the hard-label base as much as the distilled pools (hz reduces it by 0.0005 ± 0.0004); judge re-cool arms against more than one reference draw (ref alone is off by 0.0005 on stale-val and on the lagged holdout).

## 1. Tool A: lagged holdout (`lagged_score.py`)

Control A (declared in `DECLARATION.txt` block 1 before any prediction existed; `logs/control_A.txt`) [M]:

| check | result |
|---|---|
| my `feed_z` / `ndcg_fast` against `src/stack_val.feed_rank_z[:, 1]` / `src/metrics.ndcg_per_feed`, 13,424 feeds | max difference 0 / 0 |
| `w5_e10_kd05w192_s952`, my scoring with the UNCUT reference batches, views lag0 / priv / pub against the staged files | max logit difference 1.5e-5 / 1.5e-5 / 0; subset NDCG 0.32177 / 0.31810 / 0.31871, identical to staged |
| `w5_e10_kd05t25w192_s953`, lag0 uncut, priv / pub with anchored sub-batches | 1.5e-5 each; 0.32097 / 0.31740 / 0.31790, identical |
| `abl_e10_pop1kd05t25_s952`, anchored sub-batches | 1.5e-5 each; 0.31715 / 0.31625 / 0.31627, identical |
| W8 pool, lag0 -> priv (reviewer: 0.32334 -> 0.32002) | 0.32334 -> 0.32002: -0.00332 ± 0.00075 |
| pop1kd pool, lag0 -> priv | 0.31773 -> 0.31724: -0.00049 ± 0.00038 |
| S_lag, w 0.2, private-like (reviewer: +0.00024 ± 0.00031) | +0.00024 ± 0.00031 |
| S_0 / D (reviewer: -0.00050 ± 0.00032 / +0.00074 ± 0.00040) | -0.00050 ± 0.00032 / +0.00074 ± 0.00040 |
| public-like: W8 pool, S_lag, D (reviewer: 0.32055, +0.00018 ± 0.00031, +0.00068 ± 0.00039) | 0.32055, +0.00018 ± 0.00031, +0.00068 ± 0.00039 |

**Control A: VALID.** The other nine staged members (lag0 / priv from the poplag study, pub from the reviewer) were cut to my layout by `lagged_score.py import-staged` (list in `cache/pred/IMPORTED.txt`).

Cost and use [M] (`logs/timing.tsv`; two workers of 12 threads side by side, machine load 130-160):

| step | CPU time per member |
|---|---|
| one lagged view of the subset (2,089,648 rows), anchored sub-batches | 3.2 min (mean 194 s, range 175-210 s; d128 and d192 alike) |
| the usual read (lag0 + private-like) | **6.5 min per member** at that load (with the public-like view 9.7 min); 4.1 min on average over all 118 views of the run; 2.6 min after 06:30Z, when the machine load had dropped |
| the same with the uncut reference batches (`WT_REFBATCH=1`) | 5.7-6.6 min per view (341-393 s) |
| benchmark on 1,500 feeds (`logs/bench2.log`; the second worker was scoring uncut batches) | 28.7 K rows/s against 8.3 K uncut; in production two anchored workers reach 10.8 K rows/s each |

The model output depends on the history padding of a batch (max |logit diff| 5.6e-2 when a sub-batch pads to a shorter history), so the fast path cuts every reference batch into sub-batches of about 2,048 candidates that each keep one feed with the reference padding (`wt_common.anchored_batches`; max |logit diff| to the uncut batches 1.5e-5, the level between two CPU runs). Read-out: `lagged_score.Lag` (`level(spec, view)`, `compare(A, B)` -> paired deltas per view, `D_<view>` = delta(view) - delta(lag0), pool drops) or the CLI `wt_read.py pair '<A>' '<B>'`, which prints every witness at once. Pool specs: `'W8'`, `'hz@last'`, `'name1,name2'`, `'0.8*W8+0.2*pop1kd'` (sum of w * z(group mean)), `'0.3~F10h+0.7~F7xh'` (raw-weighted group means).

Seed sensitivity of the admixture statistic on the subset [M] (`subpool_D.py`, 60 random half-size sub-pools, 4 of 8 W8 + 2 of 4 pop1kd): sd over sub-pools S_0 0.00025, S_lag 0.00024, D 0.00030, against a mean per-feed SE of the same half-size statistics of 0.00033 / 0.00031 / 0.00040 (ratio 0.77 / 0.76 / 0.75). Two consequences: (1) D is not less seed-sensitive than the deltas, so a reduced pool cannot stand in for a full pool "because only D matters" (this is why pair (iv) has no lagged column, section 3); (2) the per-feed SE already covers the seed noise of this statistic (ratio below 1: no seed component that is common to all feeds), so +0.00024 ± 0.00031 needs no extra seed term.

## 2. Tool B: stale-val on the CPU (`stale_val_cpu.py`)

CPU fp32 copy of `nn_x/w5/stale_pred.py` (= `nn_x/abl/stale_pred.py`, byte-identical): same scenario `both_pre2both` (private-like `it_hist_* x5`, `seq_cv_* x3` from `nn_x/r6/stale_cf`, `pre2_fe / _z / _s` from `solvers/out/stale/cf`), same `SplitData('val')` and `Ranker`, no autocast. Output `cache/staleval/nn_<member>_staleval.npy` (and `..._valbase.npy` without overrides).

Control B as declared in block 1 (`logs/control_B.txt`) [M], val 14,423 feeds:

| member | CPU without overrides vs stored GPU val | CPU stale-val vs stored GPU stale-val |
|---|---|---|
| `w5_cd_hzcd_r101` | 0.39687 vs 0.39693: -0.00006 ± 0.00016, corr 0.999999 | 0.39615 vs 0.39615: +0.00000 ± 0.00016 |
| `w5_cd_m02ref_r101` | (run later for B') | 0.39578 vs 0.39554: +0.00024 ± 0.00018 |
| `abl_pop1_s23` | 0.39248 vs 0.39278: -0.00030 ± 0.00020, corr 0.999993 | 0.39245 vs 0.39279: -0.00034 ± 0.00020 |
| `abl_base_s23` | (run later for B') | 0.39178 vs 0.39191: -0.00014 ± 0.00015 |
| `w5_e10_kd05w192_s951` | 0.39897 vs 0.39859: +0.00038 ± 0.00019, corr 1.000000 | no GPU file |
| pair `hzcd_r101 - m02ref_r101`, stale-val | | CPU +0.00037 ± 0.00075, GPU +0.00061 ± 0.00075, CPU - GPU **-0.00024 ± 0.00023 (1.04 SE)** |
| pair `abl_pop1_s23 - abl_base_s23`, stale-val | | CPU +0.00068 ± 0.00106, GPU +0.00088 ± 0.00107, CPU - GPU -0.00020 ± 0.00023 |

- b1 (CPU without overrides reproduces the stored val predictions within fp32 / bf16 noise): **VALID** (bar 0.0010 per member, corr >= 0.9995).
- b2: the four members are within 0.00034 (bar 0.0010), but my bar for the pair statistic was "within 1 SE" and the first pair sits at 1.04 SE. **b2 FAILS BY THE LETTER.** Applied as written: control B does not clear the tool. My opinion, stated and not applied: the bar was mis-sized by me (a pure numerics difference is outside 1 SE in 32% of pairs; for at least one of two pairs in 54% of cases).
- Because of that I declared a second control B' (`DECLARATION.txt` block 3, 03:03Z, before any of its numbers) with 2 SE bars and with the member's fp32 / bf16 offset cancelled. Result (`logs/control_Bprime.txt`) [M]:

| B' check | CPU fp32 | stored GPU bf16 | CPU - GPU |
|---|---|---|---|
| effect of the override (stale - plain), `w5_cd_hzcd_r101` | -0.00072 ± 0.00045 | -0.00078 ± 0.00046 | +0.00006 ± 0.00020 |
| same, `w5_cd_m02ref_r101` | -0.00152 ± 0.00057 | -0.00167 ± 0.00057 | +0.00015 ± 0.00023 |
| same, `abl_pop1_s23` | -0.00003 ± 0.00012 | +0.00001 ± 0.00012 | -0.00004 ± 0.00009 |
| same, `abl_base_s23` | -0.00156 ± 0.00056 | -0.00157 ± 0.00055 | +0.00001 ± 0.00020 |
| mean of the four differences (bar ±0.00015) | | | +0.00005 |
| 4-pool `w5_cd_hzcd_r101-104` vs 4-pool `w5_cd_m02ref_r101-104`, stale-val delta (bars 2 SE and 0.0003) | +0.00078 ± 0.00054 | +0.00084 ± 0.00054 | -0.00006 ± 0.00022 |

**Control B': VALID.** Reading: the CPU runner applies the same override as the GPU script (the effect of the override agrees per member to 0.00015 and at pool level to 0.00006); what differs between a CPU and a GPU file of one member is a numerics offset of up to 0.0004 NDCG (seven members: -0.00034 .. +0.00038, sd 0.00027) that is the same with and without overrides. So: **never compare a CPU stale-val file with a GPU file; within one platform the tool is faithful.** Every stale-val number below is CPU against CPU. The lead decides whether the failed letter of b2 or the valid B' governs; I use the tool and flag it.

Cost [M]: 2.4 min per member and mode under the early load (mean 143 s over the first 45 runs, range 101-186 s; val 14,423 feeds; two workers); 1.8 min on average over all 159 runs of the run, 1.0 min after 06:30Z.

Where stale-val sits between val and test, label-free [M] (`sv_parity.py`, `logs/sv_parity.txt`; no label is read; zero share = share of candidate rows whose input is 0, i.e. cold):

| input | val as stored | STALE-VAL override | test public (8,562 feeds) | test private (20,033) | lagged subset, lag 0 | lagged subset, private-like |
|---|---|---|---|---|---|---|
| `it_hist_all_luo` zero share | 0.379 | **0.485** | 0.428 | 0.456 | 0.255 | 0.469 |
| `it_hist_all_luo` median / p90 of log1p on warm rows | 2.30 / 4.41 | 2.20 / 4.25 | 2.30 / 4.41 | 2.30 / 4.41 | 2.48 / 4.47 | 2.30 / 4.33 |
| `it_hist_et1_luo` zero share | 0.411 | 0.515 | 0.458 | 0.485 | 0.289 | 0.497 |
| `seq_cv_pop` zero share | 0.420 | 0.474 | 0.464 | 0.476 | 0.389 | 0.490 |
| `seq_cv_invd20` zero share | 0.887 | 0.890 | 0.898 | 0.902 | 0.889 | 0.902 |
| feed cold share (mean over feeds) | 0.381 | 0.486 | 0.429 | 0.459 | 0.256 | 0.471 |

Stale-val is a little MORE stale than private (cold share 0.486 against 0.459; its warm counts are also thinned, median 2.20 against 2.30) and clearly more than public (0.429). On the cold-share axis public sits 46% of the way from val to stale-val and private 73% [I: linear position, (0.429 - 0.381) / (0.486 - 0.381) and (0.459 - 0.381) / 0.105]. That is the label-free reason why the per-feed mean SV2 = 0.5 val + 0.5 stale-val is the public-like read, and why a private-like read would weight stale-val about 0.7. Stale-val alone over-states the staleness of either test part. The same table shows the in-period holdout's problem in one number: its cold share is 0.256, below every other sample.

Side finding for the hz rule [M]: the numbers quoted for the 11-12 closure in the re-opening block ("val +0.00026 ± 0.00033, stale-val +0.00107 ± 0.00036 (z +2.95)") are the MEAN MEMBER delta at the @last checkpoints (I reproduce both exactly from the GPU files). At the level RULE hz uses (pool, selected checkpoints) the old arm reads: 4-pool stale-val +0.00084 ± 0.00054 (z 1.5), 4-pool val +0.00079 ± 0.00051; @last 4-pool stale-val +0.00115 ± 0.00051, val +0.00050 ± 0.00047.

## 3. Calibration against the public readings (task C)

Rule: `DECLARATION.txt` block 1 (3), fixed at 02:25Z before any number of mine; blocks 4-5 (03:36Z / 03:41Z) for pair (iv). Script `c_calib2.py`, logs `logs/calib2*.txt`.

### 3.1 My pools are the scored files' train-only halves [M]

Feed-mean Pearson r of per-feed z between my train-only pool and the lead's stored half `preds/ens_<tag>_to_val.npy` (label-free): W8 1.00000, W16 1.00000, S16 1.00000, G3 (20 members, member-equal) 1.00000, C3 = `Q2a8s3` (28 members) 1.00000, 0.8 z(W8) + 0.2 z(pop1kd x4) = `W8P20kd` 1.00000. So every "val" column below is the train-only half of the file that was scored; the public file adds the fine-tune half (0.5 z(FT) + 0.5 z(TO)), which my CPU witnesses cannot see. The lead's FINAL-level val deltas are printed next to mine to show the size of that mismatch.

Cross-check of the read-out against the lead's own script [M] (`xcheck_arm_read8.py`): the harness control logged at 03:14Z from `arm_read8.py` (ref re-cools vs originals, 12 snapshot pairs: val +0.00036 ± 0.00033, holdout -0.00008 ± 0.00019, mean member -0.00018 / -0.00015) is reproduced digit for digit by `wt_read.py` (+0.00036 ± 0.00033, -0.00008 ± 0.00019, -0.00018 / -0.00015). At all 16 pairs the same control reads val -0.00016 ± 0.00029, holdout -0.00023 ± 0.00017, HOLD-SHORT -0.00027 ± 0.00016, mean member -0.00029 / -0.00016: still inside ±2 SE on both witnesses (faithful), with the four later pairs (lcd s952, 954, 956, 958) at val -0.00032 ± 0.00043.

### 3.2 First read (03:38Z, `logs/calib2_run1.out`): columns that need no further CPU run [M]

| pair (A - B) | PUBLIC (FINAL files) ± paired SE | val FINAL [lead's files] | val train-only | pmix-public | pmix-private | in-period holdout (30,000) |
|---|---|---|---|---|---|---|
| (i) 0.8 z(W8) + 0.2 z(pop1kd x4) - W8 | +0.00052 ± 0.00045 | +0.00049 ± 0.00031 | +0.00055 ± 0.00035 | +0.00095 ± 0.00042 | +0.00124 ± 0.00052 | **-0.00063 ± 0.00022** |
| (ii) W16 - W8 | -0.00000 ± 0.00040 | -0.00007 ± 0.00034 | -0.00013 ± 0.00031 | +0.00026 ± 0.00039 | +0.00048 ± 0.00049 | +0.00004 ± 0.00019 |
| (iii) S16 - W8 | -0.00041 ± 0.00042 | +0.00008 ± 0.00034 | +0.00015 ± 0.00033 | +0.00037 ± 0.00041 | +0.00051 ± 0.00051 | -0.00028 ± 0.00019 |
| (iv) C3 - G3 (28 vs 20 members) | +0.00063 ± 0.00058 | +0.00069 ± 0.00040 | +0.00068 ± 0.00045 | +0.00101 ± 0.00057 | +0.00117 ± 0.00070 | **-0.00073 ± 0.00029** |
| (v) W8 - base x4 (no public reading) | n/a | n/a | +0.00270 ± 0.00064 | +0.00234 ± 0.00078 | +0.00207 ± 0.00095 | +0.00128 ± 0.00037 |
| (vi) W8 - C3 (information) | +0.00103 ± 0.00072 | +0.00064 ± 0.00055 | +0.00054 ± 0.00056 | +0.00018 ± 0.00069 | +0.00004 ± 0.00084 | +0.00149 ± 0.00034 |

Public SE = sd of the val per-feed delta of the train-only pools / sqrt(8,562). Only pairs (i) and (iv) have |public| >= 1 public SE (pair (iii) is at 0.97 SE), so only two pairs count for the SIGN score; both are staleness-family pairs and on both the in-period holdout has the wrong sign at -2.8 and -2.5 of its own SE.

### 3.3 A/A: how much of a delta is seed noise, and what the per-feed SE already covers [M]

Random disjoint N-vs-N splits of the 16 wide students (same recipe family, 40 splits each; `c_calib2.py aa`, final run `logs/calib2_final.out`, 06:55Z):

| statistic | 4 vs 4: sd over splits / mean per-feed SE | 8 vs 8: sd over splits / mean per-feed SE |
|---|---|---|
| val | 0.00055 / 0.00057 = 0.98 | 0.00042 / 0.00046 = 0.90 |
| val M<50 | 0.00074 / 0.00107 = 0.69 | 0.00076 / 0.00086 = 0.89 |
| stale-val | 0.00057 / 0.00057 = 1.00 | 0.00035 / 0.00047 = 0.75 |
| SV2 = per-feed mean of val and stale-val deltas | 0.00047 / 0.00051 = 0.94 | 0.00032 / 0.00040 = 0.78 |
| pmix-public | 0.00068 / 0.00069 = 0.98 | 0.00057 / 0.00057 = 1.00 |
| pmix-private | 0.00083 / 0.00084 = 0.99 | 0.00070 / 0.00069 = 1.00 |
| in-period holdout (30,000) | 0.00027 / 0.00032 = 0.84 | 0.00021 / 0.00026 = 0.82 |
| lagged subset, private-like | 0.00048 / 0.00048 = 1.01 | 0.00037 / 0.00038 = 0.96 |
| lagged subset, lag 0 (CPU) | 0.00042 / 0.00048 = 0.88 | 0.00034 / 0.00039 = 0.87 |
| D_priv = delta(private-like) - delta(lag 0) | 0.00058 / 0.00050 = **1.15** | 0.00052 / 0.00044 = **1.19** |
| the 35 complementary 4-vs-4 splits of the 8 W8 members alone (run 3, before the 8 newer students were scored): lag 0 / private-like / public-like / D_priv / D_pub | 1.25 / 0.72 / 0.80 / 1.44 / 1.61 | |

Reading: with the feeds held fixed, re-drawing the seeds moves a same-recipe delta by 0.7-1.0 of its per-feed SE. So the per-feed paired SE is a fair SE of a pool contrast INCLUDING seed noise (seed noise acts feed by feed; adding a separate seed term would count it twice, which is what my formula of `DECLARATION.txt` block 4 (7) does: stated in block 7, both numbers are given in 3.5). The exception is D between two different sets of seeds: the seeds differ a little in how much they lean on fresh popularity, a component common to all feeds, and the per-feed SE of D under-states its noise by a factor of about 1.2 (1.15 / 1.19 on the 16 students; 1.44 on the complementary splits of only 8). For an admixture on a FIXED base (pair (i)) this does not apply (ratio 0.75, section 1).

### 3.4 By-product: dose of the popularity-light admixture under every witness [M]

`dose.py` (`logs/dose.txt`): [(1 - w) z(base) + w z(pop1kd x4)] vs base, train-only.

| base, w | val | val M>=50 | val M<50 | pmix-public | stale-val | SV2 | in-period holdout | lag 0 (subset) | lag private-like | lag public-like |
|---|---|---|---|---|---|---|---|---|---|---|
| W8, 0.1 | +0.00043 ± 0.00024 | +0.00064 | +0.00013 | +0.00075 ± 0.00030 | +0.00037 ± 0.00023 | +0.00040 ± 0.00018 | -0.00027 ± 0.00015 | -0.00020 ± 0.00021 | +0.00013 ± 0.00021 | +0.00019 ± 0.00021 |
| W8, 0.2 (= the scored file) | +0.00055 ± 0.00035 | +0.00081 | +0.00018 | +0.00095 ± 0.00042 | +0.00088 ± 0.00036 | +0.00071 ± 0.00029 | -0.00063 ± 0.00022 | -0.00050 ± 0.00032 | +0.00024 ± 0.00031 | +0.00018 ± 0.00031 |
| W8, 0.3 | +0.00021 ± 0.00042 | +0.00094 | -0.00085 | +0.00067 ± 0.00052 | +0.00076 ± 0.00044 | +0.00049 ± 0.00037 | -0.00104 ± 0.00028 | -0.00095 ± 0.00041 | -0.00015 ± 0.00038 | -0.00033 ± 0.00038 |
| W8, 0.4 | +0.00013 ± 0.00051 | +0.00094 | -0.00105 | +0.00094 ± 0.00062 | +0.00121 ± 0.00051 | +0.00067 ± 0.00045 | -0.00170 ± 0.00034 | -0.00144 ± 0.00048 | -0.00053 ± 0.00045 | -0.00059 ± 0.00045 |
| W16, 0.1 | +0.00023 ± 0.00021 | +0.00046 | -0.00011 | +0.00042 ± 0.00026 | +0.00053 ± 0.00023 | +0.00038 ± 0.00017 | -0.00023 ± 0.00015 | -0.00032 ± 0.00023 | -0.00012 ± 0.00020 | not scored |
| W16, 0.2 | +0.00008 ± 0.00032 | +0.00061 | -0.00069 | +0.00012 ± 0.00040 | +0.00081 ± 0.00035 | +0.00044 ± 0.00027 | -0.00056 ± 0.00022 | -0.00046 ± 0.00034 | -0.00014 ± 0.00030 | not scored |
| W16, 0.3 | +0.00013 ± 0.00040 | +0.00095 | -0.00107 | +0.00031 ± 0.00049 | +0.00084 ± 0.00042 | +0.00048 ± 0.00035 | -0.00128 ± 0.00028 | -0.00149 ± 0.00042 | -0.00036 ± 0.00038 | not scored |
| W16, 0.4 | +0.00000 ± 0.00049 | +0.00091 | -0.00131 | +0.00032 ± 0.00060 | +0.00122 ± 0.00050 | +0.00061 ± 0.00044 | -0.00176 ± 0.00033 | -0.00165 ± 0.00049 | -0.00068 ± 0.00046 | not scored |

No witness other than stale-val wants more than w = 0.2; val M<50 and the lagged holdout turn against the admixture from 0.3 on. On the 16-member base the admixture is weaker on every witness but stale-val: plain val is flat at every dose (+0.00008 ± 0.00032 at 0.2), the private-like lagged holdout is negative at every dose (-0.00014 ± 0.00030 at 0.2, against +0.00024 ± 0.00031 on the 8-member base), stale-val keeps +0.0008 (2.3 SE). The public reading (+0.00052) exists for the 8-member base only.

### 3.5 All witnesses side by side, the scores, and the recommended statistic [M]

Train-only pools, A - B, ± 1 SE paired per feed (val-sample columns 14,423 feeds; holdout 30,000; lagged subset 13,424). `logs/calib2_run4.out` (04:31Z). Stale-val of pair (iv) is on the FULL pools (28 and 20 members; the four foreign-family controls pass: CPU val without overrides within 0.00013 of the stored GPU file, corr 0.999999).

| pair | PUBLIC | val | pmix-public | pmix-private | stale-val | SV2 | in-period holdout | HOLD-SHORT | lagged lag 0 | lagged private-like | lagged public-like |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (i) W8P20 - W8 | **+0.00052** ± 0.00045 | +0.00055 ± 0.00035 | +0.00095 ± 0.00042 | +0.00124 ± 0.00052 | +0.00088 ± 0.00036 | +0.00071 ± 0.00029 | -0.00063 ± 0.00022 | -0.00008 ± 0.00019 | -0.00050 ± 0.00032 | +0.00024 ± 0.00031 | +0.00018 ± 0.00031 |
| (ii) W16 - W8 | -0.00000 ± 0.00040 | -0.00013 ± 0.00031 | +0.00026 ± 0.00039 | +0.00048 ± 0.00049 | +0.00012 ± 0.00034 | -0.00000 ± 0.00026 | +0.00004 ± 0.00019 | +0.00021 ± 0.00018 | see 3.6 | see 3.6 | not scored |
| (iii) S16 - W8 | -0.00041 ± 0.00042 | +0.00015 ± 0.00033 | +0.00037 ± 0.00041 | +0.00051 ± 0.00051 | +0.00010 ± 0.00033 | +0.00012 ± 0.00027 | -0.00028 ± 0.00019 | +0.00002 ± 0.00018 | see 3.6 | see 3.6 | not scored |
| (iv) C3 - G3 | **+0.00063** ± 0.00058 | +0.00068 ± 0.00045 | +0.00101 ± 0.00057 | +0.00117 ± 0.00070 | +0.00029 ± 0.00046 | +0.00048 ± 0.00039 | -0.00073 ± 0.00029 | -0.00023 ± 0.00025 | not scored | not scored | not scored |
| (v) W8 - base x4 | n/a | +0.00270 ± 0.00064 | +0.00234 ± 0.00078 | +0.00207 ± 0.00095 | +0.00223 ± 0.00063 | +0.00246 ± 0.00057 | +0.00128 ± 0.00037 | +0.00164 ± 0.00035 | see 3.6 | see 3.6 | not scored |
| (vi) W8 - C3 (information) | +0.00103 ± 0.00072 | +0.00054 ± 0.00056 | +0.00018 ± 0.00069 | +0.00004 ± 0.00084 | +0.00012 ± 0.00055 | +0.00033 ± 0.00050 | +0.00149 ± 0.00034 | +0.00134 ± 0.00032 | not scored | not scored | not scored |
| x1: 0.8 z(W16) + 0.2 z(pop1kd) - W16 (information) | n/a | +0.00008 ± 0.00032 | +0.00012 ± 0.00040 | +0.00026 ± 0.00049 | +0.00081 ± 0.00035 | +0.00044 ± 0.00027 | -0.00056 ± 0.00022 | -0.00025 ± 0.00019 | see 3.6 | see 3.6 | not scored |
| hc: ref re-cools - original wide students, 16 snapshot pairs (expected 0) | n/a | -0.00016 ± 0.00029 | +0.00002 ± 0.00037 | +0.00014 ± 0.00045 | -0.00056 ± 0.00030 | -0.00036 ± 0.00024 | -0.00023 ± 0.00017 | -0.00027 ± 0.00016 | not scored | not scored | not scored |

Slices of the two staleness pairs (same sign pattern on both): val M>=50 +0.00081 / +0.00156, val M<50 +0.00018 / -0.00061; stale-val M>=50 +0.00172 ± 0.00039 / +0.00175 ± 0.00050, stale-val M<50 -0.00034 ± 0.00066 / -0.00182 ± 0.00087; cold positives (val) +0.00246 / +0.00743, warm positives -0.00041 / -0.00213. The popularity-light admixture is a long-feed, cold-positive gain that costs on short feeds; stale inputs make both halves larger.

**Scores** (rule of block 1 (3); 4 scored pairs; SIGN counts only the pairs with |public| >= 1 public SE = (i) and (iv)):

| candidate | pairs | SIGN right | X2 (mean squared standardised distance to public) | slope of public on candidate | mean SE over the pairs |
|---|---|---|---|---|---|
| val | 4 | 2 / 2 | **0.29** | +0.82 | 0.00036 |
| SV2 = per-feed mean of the val and stale-val deltas | 4 | 2 / 2 | 0.33 | +0.83 | 0.00030 |
| stale-val | 4 | 2 / 2 | 0.38 | +0.69 | 0.00037 |
| pmix-public | 4 | 2 / 2 | 0.67 | +0.46 | 0.00045 |
| pmix-private | 4 | 2 / 2 | 0.99 | +0.35 | 0.00055 |
| in-period holdout | 4 | **0 / 2** | 2.44 | -0.67 | 0.00022 |
| lagged private-like (pairs i, ii, iii; pair iv has no lagged read) | 3 | 1 / 1 | 0.20 | +1.23 | 0.00028 |
| lagged lag 0 on the same subset (pairs i, ii, iii) | 3 | 0 / 1 | 1.14 | -0.25 | 0.00029 |
| lagged public-like (pair i only) | 1 | 1 / 1 | 0.39 | | 0.00031 |
| inverse-variance means with the private-like lagged holdout (pairs i, ii, iii): (val, lag-priv) / (stale-val, lag-priv) / (SV2, lag-priv) | 3 | 1 / 1 each | 0.18 / 0.18 / 0.20 | +1.49 / +1.00 / +1.09 | 0.00021 / 0.00022 / 0.00020 |
| information, FINAL level (lead's files): val / pmix-public / pmix-private | 4 | 2 / 2 each | 0.21 / 0.13 / 0.21 | +0.91 / +0.68 / +0.56 | 0.00035 / 0.00043 / 0.00053 |

Joint test on the two staleness pairs (sum of the two deltas; per-feed correlation of the two pairs on val +0.27, used for the public SE):

| | sum of (i) + (iv) | ratio to public | z of (public - candidate) |
|---|---|---|---|
| PUBLIC | +0.00116 ± 0.00082 | | |
| val | +0.00123 ± 0.00063 | 1.06 | -0.07 |
| stale-val | +0.00117 ± 0.00066 | 1.01 | -0.01 |
| SV2 | +0.00120 ± 0.00056 | 1.03 | -0.04 |
| pmix-public | +0.00196 ± 0.00080 | 1.69 | -0.70 |
| pmix-private | +0.00241 ± 0.00098 | 2.08 | -0.98 |
| in-period holdout | -0.00136 ± 0.00042 | -1.17 | **+2.71** |

**Which witness has the public sign and size.** Plain val, stale-val and their per-feed mean SV2 all have the public sign on both staleness pairs and the public size (ratio 1.0-1.1 on the sum). pmix-public has the sign but reads 1.7 times the public size at train-only level (2.1 times for pmix-private); at FINAL level the lead's pmix-public is closer (X2 0.13, slope 0.68). The in-period holdout has the wrong sign on both and is 2.7 SE away from public. The private-like lagged holdout has three scored pairs: +0.00024, +0.00024, -0.00028 against public +0.00052, 0.0000, -0.00041 (X2 0.20, slope 1.23); it turns the in-period holdout's -0.00050 on pair (i) into +0.00024 and it is the only witness outside the holdout family that has the public sign on pair (iii). That pair is at 0.97 public SE; part of its public loss sits in the fine-tune half (FINAL-level pmix-public -0.00022 against +0.00037 train-only), which none of my train-only witnesses can see. Pair (iv), the second staleness pair, has no lagged read, so the lagged holdout is calibrated on ONE staleness pair only.

**Recommended decision statistic, by the rule as written: plain val of the train-only pools** (smallest X2, 0.29; SV2 0.33 and stale-val 0.38 are inside the 0.25 tie band, the tie-break by SE would give SV2, but the rule lets a CPU statistic replace a free one only when its X2 is smaller by more than 0.25, which it is not).

SE of a pool contrast [M] (`se_pairing.py`, `logs/se_pairing.txt`; `c_calib2.py aa`):

| statistic | 16 vs 16 PAIRED by snapshot (ref re-cools vs originals) | 8 vs 8 paired by snapshot | 8 vs 8 independent seeds (three contrasts) | 16 vs 16 independent seeds [I: paired value x the 8-vs-8 ratio] | my block 4 (7) formula (double-counts, see 3.3) |
|---|---|---|---|---|---|
| val | **0.00029** | 0.00037 / 0.00036 | 0.00048 / 0.00050 / 0.00047 | 0.00038 | 0.00042 |
| stale-val | 0.00030 | 0.00037 / 0.00037 | 0.00047 / 0.00048 / 0.00047 | 0.00038 | 0.00039 |
| SV2 | **0.00024** | 0.00031 / 0.00030 | 0.00041 / 0.00042 / 0.00040 | 0.00032 | 0.00033 |
| pmix-public | 0.00037 | 0.00046 / 0.00046 | 0.00060 / 0.00062 / 0.00058 | 0.00048 | 0.00055 |
| in-period holdout | 0.00017 | 0.00021 / 0.00021 | 0.00028 / 0.00029 / 0.00028 | 0.00023 | 0.00023 |
| lagged subset, private-like | 0.00026 | not computed | A/A 8 vs 8: 0.00038 | 0.00027 [I: 0.00038 x sqrt(1/2)] | 0.00037 |
| D (private-like - lag 0) | 0.00033 per feed; 0.00039 with the seed excess of 3.3 | not computed | A/A 8 vs 8: 0.00044 per feed, sd over splits 0.00052 | 0.00037 [I] | 0.00050 |

So the recommended statistic has SE 0.00029 on a 16-vs-16 contrast that is paired by snapshot (hz against ref) and 0.00038 for 16 independent seeds against 16; a 2 SE bar is +0.0006 / +0.0008.

My opinion, stated and not applied: SV2 is the better decision statistic for this family. It has the same sign record, the same slope (0.83) and an X2 that is indistinguishable from val's (0.33 against 0.29 with four pairs), it is 17% tighter (0.00024 / 0.00032), and half of it is measured under private-like inputs, which is the regime the family is about (val alone read the 16-member admixture x1 at +0.00008 where stale-val reads +0.00081 ± 0.00035). It costs 2.4 CPU minutes per member. What four public pairs cannot do is separate val, stale-val and SV2 from each other: all three sit within 0.1 X2.

Note on the harness control under stale inputs [M]: the ref re-cools read below the original students on stale-val (16 pairs -0.00056 ± 0.00030, -1.8 SE; M<50 -0.00129 ± 0.00058; the two paired halves -0.00085 ± 0.00037 and -0.00054 ± 0.00037), while val (-0.00016 ± 0.00029) and the holdout (-0.00023 ± 0.00017) are faithful. The second-seed re-cools resolve it (`ref2_null.py`, `logs/ref2_null.txt`): ref2 - originals is -0.00010 ± 0.00033 on stale-val and ref2 - ref +0.00046 ± 0.00031, so re-cooling as such costs nothing under stale inputs and the ref pool is a low draw of about 0.0005 on this one witness. That matters for every arm judged against ref on stale-val (section 6.3).

A/A null of a snapshot-paired re-cool contrast, ref2 - ref, 16 pairs [M]: val +0.00011 ± 0.00033, val M<50 +0.00057 ± 0.00059, stale-val +0.00046 ± 0.00031 (z +1.5), SV2 +0.00028 ± 0.00026, pmix-public -0.00008 ± 0.00042, in-period holdout +0.00013 ± 0.00018, HOLD-SHORT -0.00005 ± 0.00016; single members: sd of the member delta over the 16 snapshots 0.00061 (val), 0.00061 (stale-val), 0.00029 (holdout), 8 / 8 / 9 of 16 positive.

### 3.6 Lagged-holdout columns of the other pairs [M]

Subset, 13,424 feeds, CPU fp32 on both sides; D = delta(private-like) - delta(lag 0).

| pair | PUBLIC | lag 0 | private-like | D | pool drop A / B under the private-like lag | inverse-variance mean of SV2 and private-like |
|---|---|---|---|---|---|---|
| (i) W8P20 - W8 | +0.00052 ± 0.00045 | -0.00050 ± 0.00032 | +0.00024 ± 0.00031 | +0.00074 ± 0.00040 | -0.00259 ± 0.00065 / -0.00332 ± 0.00075 | +0.00050 ± 0.00021 |
| (ii) W16 - W8 | -0.00000 ± 0.00040 | +0.00006 ± 0.00027 | +0.00024 ± 0.00027 | +0.00018 ± 0.00032 | -0.00314 ± 0.00075 / -0.00332 ± 0.00075 | +0.00012 ± 0.00019 |
| (iii) S16 - W8 | -0.00041 ± 0.00042 | -0.00039 ± 0.00027 | -0.00028 ± 0.00027 | +0.00011 ± 0.00034 | -0.00321 ± 0.00076 / -0.00332 ± 0.00075 | -0.00007 ± 0.00019 |
| (iv) C3 - G3 | +0.00063 ± 0.00058 | not scored | not scored | not scored | not scored | not scored |
| (v) W8 - base x4 | n/a | +0.00258 ± 0.00054 | +0.00217 ± 0.00051 | -0.00041 ± 0.00053 | -0.00332 ± 0.00075 / -0.00291 ± 0.00076 | +0.00230 ± 0.00038 |
| x1: 0.8 z(W16) + 0.2 z(pop1kd) - W16 | n/a | -0.00046 ± 0.00034 | -0.00014 ± 0.00030 | +0.00032 ± 0.00041 | -0.00282 ± 0.00066 / -0.00314 ± 0.00075 | +0.00018 ± 0.00020 |
| hc: ref re-cools - original wide students (16 pairs; expected 0) | n/a | -0.00046 ± 0.00027 | **-0.00053 ± 0.00026** | -0.00007 ± 0.00033 | -0.00321 ± 0.00075 / -0.00314 ± 0.00075 | |

Readings [M]: (1) every pool of students loses 0.0026-0.0033 when the holdout inputs are lagged (the hard-label base x4 loses 0.0029: distillation does not change the dependence on fresh popularity; the KD gain itself survives the lag, +0.0022 ± 0.0005). (2) D is +0.0007 for the popularity-light admixture on W8 and +0.0001 to +0.0002 for the two non-staleness pairs: the lag moves only the contrast it should move. (3) The harness control fails on this witness: the ref re-cools sit 0.0005 below the originals under both views (-2.0 SE private-like), the same low draw of the ref pool that stale-val shows (3.5). Pair (iv) was not scored: its members come from three other code families and the reduced 13-member pool I had planned failed my own representativeness check on the holdout (+2.98 SE, block 5), and a full 28-member lagged read did not fit the budget.

## 4. Exact commands

All from `<repo>/nn_x/w12/stale/witness/` (every script sets `CUDA_VISIBLE_DEVICES=""`, at most 12 threads, no bytecode; nothing outside this directory is written). `PY=../../../../.venv/bin/python`.

```
# prediction caches (heavy: 12 threads each, at most two at a time)
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY lagged_score.py prep                                   # once: subset mini split, 6 s (already done)
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY lagged_score.py score lag0,priv <member,member|group>  # cache/pred/<view>/<member>.npy
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY stale_val_cpu.py stale <member,member|group>           # cache/staleval/nn_<member>_staleval.npy
#   member = any name in nn_x/w5/ckpt, nn_x/abl/ckpt, nn_x/r6/ckpt, nn/ckpt (prefix w5_ / abl_ / r6_ / other), '@last' allowed; group = a key of wt_common.G

# a queue of both kinds for two workers; lines "lag <member> <views csv>" / "sv <member> <stale|base|both>", highest priority first, re-read before every pick;
# harness runs (w5_kcd_*, w5_lcd_*) are picked up when their logs/<variant>.json exists
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 setsid nohup $PY wt_worker.py A 12:00 hz_tasks.txt > logs/worker_A.out 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 setsid nohup $PY wt_worker.py B 12:00 hz_tasks.txt > logs/worker_B.out 2>&1 < /dev/null &
#   a worker stops at the UTC time given or on `touch logs/STOP_A`; progress in logs/worker_A.log and logs/timing.tsv

# read-outs (light: 4 threads)
WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY wt_read.py pair '<pool A>' '<pool B>' [label]   # every witness for one pair of pools
#   pool: 'W16' | 'hz@last' | 'name1,name2' | '0.8*W16+0.2*pop1kd' | '0.3~F10h+0.35~F7xh+0.35~F7x70h'
WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY d_hz.py [arm = hz] [ref = ref]                  # rule ARM hz: logs/read_<arm>_vs_<ref>.txt
WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY c_calib2.py aa                                  # calibration + A/A + 16-vs-16 SE: logs/calib2.txt
WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY c_control.py A                                  # control A;  c_controlB2.py = control B'
WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $PY dose.py        # also: subpool_D.py, se_pairing.py, sv_parity.py, recheck_cache.py
```

A new arm of the staleness family (members `m1..mk` against reference members `r1..rk`): put `sv <member> stale` (2.4 min each) and, if wanted, `lag <member> lag0,priv` (6.5 min each) for BOTH sides into a tasks file, run one or two workers, then `wt_read.py pair 'm1,..,mk' 'r1,..,rk'`. Never mix a CPU file with a stored GPU file inside one column. A re-cool arm `<arm>` of the K / L harness (for example the full-dose `hz2` that started training at 05:05Z): replace `hz` by `<arm>` in `hz_tasks.txt` and run `d_hz.py <arm> ref`.

## 5. Files created

`nn_x/w12/stale/witness/`:

| file | what |
|---|---|
| `DECLARATION.txt` | my judging rules and what had been seen at each change, blocks 1-16; each rule written before the read it judges, the post-hoc block 15 marked as such |
| `PROGRESS.md` | state of the work |
| `wt_common.py` | paths, pool groups (`G`), metric, per-feed z, SE, anchored CPU batches |
| `lagged_score.py` | TOOL A: subset mini split, CPU fp32 scoring under the lag0 / private-like / public-like views, `Lag` read-out |
| `stale_val_cpu.py` | TOOL B: CPU fp32 copy of `nn_x/w5/stale_pred.py` (stale-val and no-override val) |
| `wt_worker.py`, `tasks.txt`, `hz_tasks.txt` | queue runner (two workers), the queue of this run, the hz-only queue for a later read |
| `wt_read.py` | every witness for a pair of pools (val, slices, pmix, stale-val, SV2, SVpriv, holdout, HOLD-SHORT, lagged, D) |
| `c_control.py`, `c_controlB2.py`, `recheck_cache.py` | controls A, B, B' and the cache re-check of the resume |
| `c_calib.py`, `c_calib2.py` | calibration against the public readings (v2 is the one reported), joint test, A/A, 16-vs-16 SE |
| `d_hz.py`, `mk_hz_tables.py` | the hz read (rule ARM hz) and its markdown tables; `d_hz.py hzcd m02ref` = the 11-12 arm in the same format (`logs/test_d_hz_oldarm.out`) |
| `hz_refs.py`, `ref2_null.py`, `xcheck_arm_read8.py` | an arm against every same-recipe reference pool (post-hoc), the ref2 - ref null with every witness, the cross-check against `arm_read8.py` |
| `dose.py`, `subpool_D.py`, `se_pairing.py`, `sv_parity.py`, `bench.py`, `bench2.py` | admixture dose table, seed sensitivity of sub-pools, SE paired / independent, label-free parity of the stale-val inputs, batching benchmarks |
| `cache/sub/` | subset mini split and lagged columns (897 MB) |
| `cache/pred/<view>/<member>.npy` | lagged-holdout predictions on the subset, CPU fp32 (`IMPORTED.txt` lists the files cut from the poplag study) |
| `cache/staleval/nn_<member>_{staleval,valbase}.npy` | CPU stale-val / no-override val predictions |
| `cache/readout/*.json` | machine-readable read-outs (calibration, scores, A/A, SE, dose, hz) |
| `logs/` | controls, calibration runs, worker logs, `timing.tsv`, queue versions `tasks_v*.txt`, the hz read |

Nothing outside `nn_x/w12/stale/witness/` and this report was written. No GPU was used, no process of another owner was touched (two workers of my first run were stopped at 02:35Z by exact PID and at 02:44Z through their STOP file).

## 6. READ of the re-opened arm hz (task D)

Rule: "ARM hz" of `nn_x/w5/logs/E_kd_predeclared.txt` (2026-11-15T02:15:30Z), applied as written; my implementation choices in `DECLARATION.txt` block 1 (4), read time in blocks 9 and 14. Script `d_hz.py hz ref`, output `logs/read_hz_vs_ref_0631Z.txt`, `cache/readout/hz_vs_ref.json`.

**What was read.** All 16 of 16 (hz, ref) snapshot pairs: `w5_kcd_{hz,ref}_s951-958` (teacher mean11) and `w5_lcd_{hz,ref}_s951-958` (teacher mean25); the last hz run finished at about 05:55Z. The read was made ONCE, at 06:31Z, when every scheduled file of all 16 pairs existed (CPU fp32 stale-val of the selected and the @last checkpoints of all 32 members; lagged lag 0 + private-like of the selected checkpoints of all 32 members). No hz number was computed before that line of the declaration. Cross-check [M]: the lead's `arm_read8.py rc:hz` gives the same val, val M>=50 / M<50, holdout, holdout @last and HOLD-SHORT to the last digit (`logs/xcheck_arm_read8_hz.txt`).

### 6.1 Verdict by the rule

| clause (16-pools, SELECTED checkpoints, paired by snapshot, 14,423 val feeds) | value | met |
|---|---|---|
| val delta >= 0 | **-0.00009 ± 0.00040** (z -0.22) | **no** |
| stale-val delta >= +2 SE | +0.00113 ± 0.00043 (+2.60 SE) | yes |
| val M<50 delta >= -1 SE | -0.00025 ± 0.00072 (-0.34 SE) | yes |
| PASS (all three) | | **no** |
| closing clause "val < 0 OR stale-val < +1 SE" | val < 0 | **met** |

**RULE hz: no PASS; CLOSED AGAIN under KD** (the half-dose view; the ON PASS actions, fine-tuning the 16 hz members and the files H16 / H16P20, do not apply). The full-dose arm hz2 is a separate arm and was already training at 05:05Z.

### 6.2 All witnesses [M]

| witness (hz - ref, paired per feed) | SELECTED: hz 16-pool - ref 16-pool (decision) | @last: hz - ref | SELECTED: [0.8 z(hz) + 0.2 z(pop1kd x4)] - [0.8 z(ref) + 0.2 z(pop1kd x4)] | @last, same admixture |
|---|---|---|---|---|
| pool level arm / reference, val | 0.40119 / 0.40127 | 0.40154 / 0.40130 | 0.40126 / 0.40137 | 0.40148 / 0.40153 |
| pool level arm / reference, stale-val | 0.40052 / 0.39939 | 0.40083 / 0.39966 | 0.40093 / 0.40044 | 0.40086 / 0.40055 |
| pool level arm / reference, in-period holdout | 0.32361 / 0.32437 | 0.32388 / 0.32432 | 0.32336 / 0.32396 | 0.32331 / 0.32383 |
| val, all 14,423 feeds | -0.00009 ± 0.00040 (z -0.2) | +0.00024 ± 0.00038 (z +0.6) | -0.00010 ± 0.00035 (z -0.3) | -0.00005 ± 0.00035 (z -0.1) |
| val M>=50 (8,530) | +0.00002 ± 0.00045 (z +0.0) | +0.00011 ± 0.00042 (z +0.3) | -0.00030 ± 0.00037 (z -0.8) | -0.00044 ± 0.00038 (z -1.1) |
| val M<50 (5,893) | -0.00025 ± 0.00072 (z -0.3) | +0.00042 ± 0.00069 (z +0.6) | +0.00017 ± 0.00066 (z +0.3) | +0.00052 ± 0.00064 (z +0.8) |
| pmix-public | +0.00034 ± 0.00052 (z +0.6) | +0.00086 ± 0.00051 (z +1.7) | -0.00006 ± 0.00044 (z -0.1) | -0.00004 ± 0.00046 (z -0.1) |
| pmix-private | +0.00066 ± 0.00065 (z +1.0) | +0.00128 ± 0.00065 (z +2.0) | +0.00011 ± 0.00055 (z +0.2) | +0.00010 ± 0.00058 (z +0.2) |
| **stale-val** (CPU fp32) | +0.00113 ± 0.00043 (z +2.6) | +0.00117 ± 0.00045 (z +2.6) | +0.00049 ± 0.00039 (z +1.3) | +0.00031 ± 0.00039 (z +0.8) |
| stale-val M>=50 | +0.00117 ± 0.00048 (z +2.4) | +0.00087 ± 0.00051 (z +1.7) | +0.00068 ± 0.00042 (z +1.6) | +0.00030 ± 0.00043 (z +0.7) |
| stale-val M<50 | +0.00108 ± 0.00080 (z +1.3) | +0.00159 ± 0.00083 (z +1.9) | +0.00022 ± 0.00074 (z +0.3) | +0.00032 ± 0.00073 (z +0.4) |
| SV2 (public-like position) | +0.00052 ± 0.00034 (z +1.5) | +0.00070 ± 0.00034 (z +2.1) | +0.00019 ± 0.00030 (z +0.6) | +0.00013 ± 0.00030 (z +0.4) |
| SVpriv (private-like position) | +0.00080 ± 0.00037 (z +2.2) | +0.00092 ± 0.00038 (z +2.4) | +0.00033 ± 0.00033 (z +1.0) | +0.00021 ± 0.00033 (z +0.6) |
| cold positives, val (3,422) | +0.00377 ± 0.00092 (z +4.1) | +0.00458 ± 0.00089 (z +5.2) | +0.00247 ± 0.00084 (z +2.9) | +0.00302 ± 0.00083 (z +3.6) |
| warm positives, val (9,118) | -0.00169 ± 0.00049 (z -3.4) | -0.00156 ± 0.00047 (z -3.3) | -0.00129 ± 0.00042 (z -3.1) | -0.00130 ± 0.00043 (z -3.1) |
| cold positives, stale-val | -0.00095 ± 0.00088 (z -1.1) | -0.00211 ± 0.00092 (z -2.3) | -0.00165 ± 0.00079 (z -2.1) | -0.00185 ± 0.00083 (z -2.2) |
| warm positives, stale-val | +0.00224 ± 0.00058 (z +3.9) | +0.00251 ± 0.00060 (z +4.2) | +0.00133 ± 0.00052 (z +2.6) | +0.00121 ± 0.00051 (z +2.4) |
| in-period holdout, 30,000 (reported only) | -0.00076 ± 0.00022 (z -3.5) | -0.00044 ± 0.00022 (z -2.0) | -0.00059 ± 0.00020 (z -3.0) | -0.00052 ± 0.00019 (z -2.7) |
| in-period holdout, subset 13,424 | -0.00023 ± 0.00030 (z -0.8) | -0.00005 ± 0.00030 (z -0.2) | -0.00006 ± 0.00029 (z -0.2) | -0.00016 ± 0.00028 (z -0.6) |
| HOLD-SHORT | -0.00040 ± 0.00019 (z -2.1) | -0.00028 ± 0.00019 (z -1.5) | -0.00048 ± 0.00017 (z -2.7) | -0.00039 ± 0.00017 (z -2.3) |
| pool level arm / reference, lagged subset lag 0 | 0.32286 / 0.32295 | not scored | 0.32247 / 0.32270 | not scored |
| pool level arm / reference, lagged subset private-like | 0.32016 / 0.31973 | not scored | 0.32016 / 0.31976 | not scored |
| lagged subset, lag 0 (CPU) | -0.00009 ± 0.00030 (z -0.3) | not scored | -0.00023 ± 0.00029 (z -0.8) | not scored |
| lagged subset, private-like | +0.00042 ± 0.00034 (z +1.3) | not scored | +0.00040 ± 0.00031 (z +1.3) | not scored |
| D = private-like - lag 0 | +0.00051 ± 0.00041 (z +1.2) | not scored | +0.00063 ± 0.00039 (z +1.6) | not scored |
| arm pool drop under the private-like lag | -0.00270 ± 0.00069 (z -3.9) | not scored | -0.00231 ± 0.00061 (z -3.8) | not scored |
| reference pool drop under the private-like lag | -0.00321 ± 0.00075 (z -4.3) | not scored | -0.00294 ± 0.00065 (z -4.5) | not scored |

Lagged columns of the @last checkpoints [M] (second run of `d_hz.py` at 07:17Z, `DECLARATION.txt` block 16; the selected part of that run is identical to the 06:31Z read, `logs/read_hz_vs_ref_FIRST_0631Z.txt`): scored for the kcd half only, 8 snapshot pairs: lag 0 +0.00017 ± 0.00032, private-like +0.00050 ± 0.00036, D +0.00034 ± 0.00045 (8-pool levels 0.32307 / 0.32290 at lag 0, 0.31976 / 0.31925 private-like; the same 8 @last pairs read val -0.00010 ± 0.00040, stale-val +0.00122 ± 0.00046, in-period holdout -0.00049 ± 0.00023). Not scored: @last lagged of the lcd half, the public-like view (section 7).

Per snapshot (single re-cool member, hz - ref): SELECTED positive in 14 / 14 / 0 of 16 snapshots on val / stale-val / in-period holdout; @last 13 / 15 / 0. Halves, selected: kcd (8 snapshots) val +0.00023 ± 0.00041, stale-val +0.00105 ± 0.00045, holdout -0.00040 ± 0.00023; lcd val +0.00033 ± 0.00039, stale-val +0.00071 ± 0.00045, holdout -0.00060 ± 0.00024.

How the delta changes with the pool size (selected):

| level | val | stale-val | in-period holdout |
|---|---|---|---|
| mean single member (16 snapshots) | +0.00046 ± 0.00022 [`arm_read8.py`] | +0.00097 (mean of the 16 member deltas) | -0.00043 ± 0.00014 |
| 8-pools (the two halves) | +0.00023 / +0.00033 | +0.00105 / +0.00071 | -0.00040 / -0.00060 |
| 16-pool | -0.00009 ± 0.00040 | +0.00113 ± 0.00043 | -0.00076 ± 0.00022 |

The val gain of a single hz member is real at member level (+2.1 SE) and is gone in the 16-pool: on val, hz does for one member what pooling already does. The stale-val gain does not shrink with pooling: that part is a change in how the model uses stale inputs, not variance reduction. The in-period holdout loss grows with pooling and is negative in every one of the 16 snapshots.

### 6.3 How much of the verdict depends on the checkpoint and on the reference [M]

1. Checkpoint. The harness selects the "selected" checkpoint on the in-period holdout (`--select_on holdout`), the sample the new policy calls biased against this family. At the selection-free @last checkpoints the three clauses read val +0.00024 ± 0.00038, stale-val +0.00117 ± 0.00045 (+2.57 SE), val M<50 +0.00042 ± 0.00069: the same rule would say PASS. The rule decides on the selected checkpoints, so the verdict stands; but it is decided by the sign of a val delta of 0.2 SE.
2. Reference (POST-HOC, information only, `DECLARATION.txt` block 15, `hz_refs.py`, `ref2_null.py`). Three same-recipe 16-pools exist for the same snapshots: the ref re-cools, the ref2 re-cools (second cooldown seed) and the original wide students. Among them the ref pool is a low draw under stale-val: ref - originals -0.00056 ± 0.00030, ref2 - ref +0.00046 ± 0.00031, ref2 - originals -0.00010 ± 0.00033 (on val the three agree: -0.00016 ± 0.00029, +0.00011 ± 0.00033, -0.00005 ± 0.00032; so the harness is faithful and the ref deficit under stale inputs is a draw, not a property of re-cooling).

| hz 16-pool (selected) minus | val | stale-val | SV2 | SVpriv | pmix-private | in-period holdout | lagged lag 0 | lagged private-like | D |
|---|---|---|---|---|---|---|---|---|---|
| ref (the rule's reference) | -0.00009 ± 0.00040 | **+0.00113 ± 0.00043 (+2.6)** | +0.00052 ± 0.00034 | +0.00080 ± 0.00037 | +0.00066 ± 0.00065 | -0.00076 ± 0.00022 | -0.00009 ± 0.00030 | +0.00042 ± 0.00034 | +0.00051 ± 0.00041 |
| ref2 | -0.00020 ± 0.00043 | +0.00067 ± 0.00046 (+1.5) | +0.00024 ± 0.00037 | +0.00044 ± 0.00040 | +0.00078 ± 0.00072 | -0.00089 ± 0.00025 | not scored | not scored | not scored |
| original wide students (= W16) | -0.00025 ± 0.00042 | +0.00058 ± 0.00047 (+1.2) | +0.00016 ± 0.00037 | +0.00035 ± 0.00040 | +0.00080 ± 0.00070 | -0.00098 ± 0.00024 | -0.00055 ± 0.00034 | **-0.00011 ± 0.00037** | +0.00044 ± 0.00045 |
| ref + ref2 + originals (48 members) | -0.00002 ± 0.00040 | +0.00066 ± 0.00044 (+1.5) | +0.00032 ± 0.00034 | +0.00048 ± 0.00037 | +0.00100 ± 0.00066 | -0.00086 ± 0.00022 | not scored | not scored | not scored |

Against any reference other than ref the stale-val clause (+2 SE) is not met either, val is zero or negative against all of them, and the private-like lagged holdout, +0.00042 against ref, is -0.00011 ± 0.00037 against the original students (the ref pool is 0.0005 low there too, section 3.6). Only D is stable across references (+0.0005 / +0.0004, about 1 SE): hz loses less under lag, from a lower starting level. So the closing is not an accident of the selected checkpoint alone: with the 48-member reference the read is "val 0.0000, stale-val +0.0007 at 1.5 SE", which is neither PASS nor a signal worth 16 fine-tunes.

### 6.4 Do the two staleness levers add or substitute? [M]

Value of each lever alone and together, selected checkpoints, against the plain ref 16-pool (sums of the paired deltas of the table and of `logs/read_hz_vs_ref_0631Z.txt`):

| | val | stale-val | lagged subset, private-like | in-period holdout |
|---|---|---|---|---|
| hz alone (hz - ref) | -0.00009 ± 0.00040 | +0.00113 ± 0.00043 | +0.00042 ± 0.00034 | -0.00076 ± 0.00022 |
| pop1kd admixture alone (0.8 z(ref) + 0.2 z(pop1kd) - ref) | +0.00009 ± 0.00036 | +0.00105 ± 0.00035 | +0.00003 ± 0.00030 | -0.00041 ± 0.00022 |
| hz on top of the admixture (second table) | -0.00010 ± 0.00035 | +0.00049 ± 0.00039 | +0.00040 ± 0.00031 | -0.00059 ± 0.00020 |
| admixture on top of hz (0.8 z(hz) + 0.2 z(pop1kd) - hz) | +0.00008 ± 0.00034 | +0.00041 ± 0.00032 | +0.00001 ± 0.00028 | -0.00025 ± 0.00022 |
| both (sum of rows 2 and 3) [I] | -0.00001 | +0.00154 | +0.00043 | -0.00100 |
| if they added (sum of rows 1 and 2) [I] | +0.00000 | +0.00218 | +0.00045 | -0.00117 |

By my declared definition (share of the plain hz - ref delta that survives on top of the admixture, on val AND stale-val): at the selected checkpoints the share is not defined (the plain val delta is not positive); stale-val alone keeps 0.43. At @last: val -0.19, stale-val 0.26 -> **SUBSTITUTE**. In words: under stale-val each lever keeps about 40% of its effect when the other is present (together +0.0015 instead of +0.0022), so they mostly substitute there; on val neither is worth anything on a 16-member base; on the lagged holdout hz keeps its +0.0004 with or without the admixture and the admixture is worth nothing on either 16-pool (it was +0.00024 ± 0.00031 on the 8-member W8).

### 6.5 What I read in it (opinion; no rule applied)

- hz does what it was built for. Cold positives +0.00377 ± 0.00092 and warm positives -0.00169 ± 0.00049 on val; with stale inputs warm positives +0.00224 ± 0.00058; the pool loses less when the holdout inputs are lagged (-0.00270 against -0.00321). It moves the model away from fresh popularity.
- The price with fresh inputs is paid in full (in-period holdout -0.0008 to -0.0010 against every reference, HOLD-SHORT -0.0004 to -0.0007), and on val (staleness 5.7 d) price and gain cancel: 0.0000 ± 0.0004.
- Where the test sets sit [I, from the label-free position of section 2: public 46%, private 73% of the way from val to stale-val, and the 48-member reference]: public about +0.0003 ± 0.0003 (SV2), private about +0.0005 ± 0.0004 (SVpriv). An H16 file would be compared with W16 = the original students: SV2 +0.00016 ± 0.00037, and the public paired SE of that contrast is 0.00055; public cannot see it.
- The lagged holdout does not add support: +0.00042 ± 0.00034 against ref, -0.00011 ± 0.00037 against the original students; the stable part is D = +0.0005 (hz loses 0.0027 under the private-like lag, the references 0.0031-0.0032).
- For hz2 (full dose): the signature to look for is the one above (cold up, warm down on val; warm up under stale inputs; holdout down). The decisive question is whether the full dose moves the val-neutral point, i.e. whether val turns positive in the 16-POOL (not in the mean member, which was already +0.00046 here and did not survive pooling).

## 7. Limits and what was NOT checked

Calibration.
- Four public pairs, of which two count for the sign (|public| >= 1 public SE) and both are popularity-light admixtures. The calibration separates the in-period holdout (wrong) from everything else; it cannot separate val, stale-val and SV2 from each other (X2 0.29 / 0.38 / 0.33), and it says nothing about a lever of the hz type, where val and stale-val disagree by 0.0012 and no file was ever scored.
- The public readings are FINAL files (0.5 fine-tuned + 0.5 train-only); every witness here is train-only. The fine-tune half is not witnessed by any of my tools (pair (iii) loses on public mostly there).
- Pair (iv) C3 - G3 has no lagged-holdout read (reduced pools failed my representativeness check, full pools did not fit). The lagged holdout is therefore calibrated on one staleness pair and two neutral pairs.
- The public-like lagged view exists only for the 12 members staged by the poplag study (pair (i)); every other lagged read is lag 0 + private-like.
- pmix was scored at train-only level, where it over-reads the public size by 1.7x; the lead's FINAL-level pmix-public (X2 0.13) was read from the lead's files, not rebuilt.
- SVpriv (0.27 val + 0.73 stale-val) and the "46% / 73%" positions are a linear reading of ONE label-free axis (cold share of `it_hist_all_luo`) [I]; nothing calibrates a private-like statistic, since there is no private reading. Stale-val also thins warm counts more than the test rows show.
- The 16-vs-16 SE for independent seeds (0.00038 val, 0.00032 SV2) is inferred from the 8-vs-8 ratio, not measured.

Tools.
- Control b2 of tool B failed by the letter of my own bar (1.04 SE against 1 SE); the properly sized control B' is valid. The lead decides which governs; every stale-val number in this report depends on it.
- There is no stored GPU stale-val file for any hz / ref / ref2 / student member, so the CPU stale-val files of this run have no cross-platform check beyond controls B / B' (7 members of other arms).
- The lagged columns were taken from the poplag study (`nn_x/w12/next/poplag/cache/lag_*_v3_*`); I did not rebuild them or re-run the label-free parity gate (my `sv_parity.py` only shows that the private-like subset has a zero share of 0.469 against 0.456 on test private).
- The cold / warm positive slices are defined on the val inputs as stored, also in the stale-val rows (a positive that turns cold under the override stays in the "warm" slice).
- HOLD-SHORT for weighted pools is my generalisation of `arm_read8.py`; it was checked against that script for unweighted pools only (two reads, equal to the digit).

hz read.
- Lagged columns exist for the SELECTED checkpoints of all 16 pairs and for the @last checkpoints of the kcd half (8 pairs) only. No public-like view of hz. No lagged read of hz against ref2.
- The robustness read against ref2 / the originals / all 48 members (6.3) is POST-HOC: I thought of it after seeing that ref is a low draw. It is information; the rule's verdict uses ref.
- I did not look at the hz test predictions, at per-day or per-category slices, or at the fine-tuned level (no hz member is fine-tuned).
- The arm hz2 (full dose, training since about 05:00Z) was not read; `hz_tasks.txt` with `hz` replaced by `hz2`, then `d_hz.py hz2 ref`, `hz_refs.py hz2` do it (at the machine load of 07:00Z about 35 worker-minutes for the stale-val of its 16 selected + 16 @last checkpoints and 45 for the lagged subset of the 16 selected ones).

Process.
- CPU only; two workers of 12 threads each; the read-out scripts (4 threads, 1-5 minutes each) ran next to them. No file outside `nn_x/w12/stale/witness/` and this report was written; imports of project modules were made with bytecode writing off and the numba cache redirected into my directory. The lead's `arm_read8.py rc:hz` was run once, read-only, as a cross-check.
- The queue order was changed six times (tasks v4-v10); each change is in `DECLARATION.txt` with what had been seen at that moment, and none was made after a number of the statistic it re-ordered had been seen.
