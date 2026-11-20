# CatBoost round 2, family B (KEY = famB): label information the neural pool lacks

Written 2026-11-15 (UTC). CPU only, at most 12 threads per process, at most two heavy processes, no GPU touched, nothing written outside `nn_x/w12/cb/famB/` and this file, no submission file, no network.
Marks: **[M]** measured here, **[D]** read in a project file, **[I]** inferred. "a ± b" = mean ± 1 SE, paired per feed; val = 14,423 feeds, holdout = the 30,000 in-period feeds.
Rules applied unchanged: `nn_x/w12/cb/PREDECLARED.txt`, block "ROUND-2 PLAN" (SCREEN: NATS ≥ 0.0016 on val AND holdout; MEMBER RULE at weight 0.1). Every arm is read with `nn_x/w12/cb/nats.py`.
My own decisions were written to `nn_x/w12/cb/famB/PROGRESS.md` before the reads they concern (time-stamped there).

## 0. Verdict

1. **Nothing in family B is PROMISING.** Best arm B3 (B0 + 70 hand-made statistics from all 3.12M non-holdout train feeds): NATS +0.00147 ± 0.00043 val / +0.00080 ± 0.00021 holdout at seed 0 and +0.00144 ± 0.00042 / +0.00081 ± 0.00021 at seed 1, against the bar 0.0016 on both (B0: 0.00117 / 0.00051). On val M ≥ 50: +0.00110 ± 0.00046 [M].
2. **Nothing passes the member rule.** Best pooled z +0.65 (two seeds of B3 as one bundle: +0.00023 ± 0.00039 val, +0.00007 ± 0.00021 holdout); the rule needs +2 [M].
3. **Best level**: B3 0.38362 / 0.30480 as a single model, 0.38476 / 0.30575 as a two-seed bundle (B0 0.38150 / 0.30462; W16 0.40144 / 0.32460). B3 over B0 is real and small: +0.00041 ± 0.00016 / +0.00030 ± 0.00008 nats and +0.0025 ± 0.0008 / +0.0009 ± 0.0004 of level, two seeds each [M].
4. **The label statistics themselves are empty.** In a linear read all 51 click statistics of the three blocks add +0.00004 ± 0.00007 (val) / +0.00006 ± 0.00006 (holdout) nats to [W16, B0], and +0.00013 ± 0.00013 on val when fitted with the test lag; B3's gain comes from two label-FREE columns (cosine to the nearest much-shown train items) and exposure counts. The user-level statistics exist for 18 to 30% of the feeds, are 10 to 15 days old on val / test, and under the test lag only 5.4% of the training feeds have one [M].
5. **B1 (native cluster CTRs) and B2 (native item_id CTR) are nulls**: NATS 0.00096 / 0.00054 (complexity 4), 0.00088 / 0.00052 (complexity 2), 0.00112 / 0.00051 (item_id; level -0.0020 / -0.0030 against B0) [M].
6. **Two CatBoost facts to keep** [M]: the default native CTR (15 borders) cannot resolve click rates near 1% (set `CtrBorderCount=255` in `simple_ctr` AND `combinations_ctr`); a user-level native key leaks massively on CPU (learn NDCG 0.56 against 0.27 on noise labels), candidate-level keys do not at the real scale (+0.0002 over chance on 120,591 feeds).
7. **STOP rule, family B part**: the one family that could pass has no arm and no bundle at 0.0016 on both witnesses (best 0.00151 / 0.00083). Not run: B4 as a candidate read (nothing was PROMISING), a B3 tree refitted on lag-matched statistics (the lead's freshness notice was read 50 minutes before the end; its linear form is in section 5a), B13.
8. [I] Label statistics for a tree are exhausted at about +0.0003 nats. If one thing is carried forward it is the density pair `kn_cos1` / `kn_cos20` as a direct input of the neural students: +0.0002 to +0.0003 nats next to the pool, identical distribution on train, val and test rows.

## 1. Arm table

All arms: CatBoostRanker YetiRank, depth 8, learning rate 0.2, `l2_leaf_reg` 5, `border_count` 254, 520 iterations, `od_wait` 100, hard click labels, train-side selection of the tree count through `cbx.run_arm` (120,591 fit feeds, 13,527 selection feeds of `xm`; no val label anywhere), seed 0 unless the tag ends in `_s1`. Arms with a wall-clock budget below the full 520 trees say so; B0 cut to the same tree count is in section 6. "B0" = the infra reference `ts_lr020_s0` (same recipe, budget-stopped at 516 trees).
Columns "level", "rho", "NATS", "screen", "member rule" are the fields of the one-line output of `nats.py` (val / holdout). The full lines are in section 9.

| tag | what changed against B0 | trees used (built) | fit min | level val / holdout | rho | NATS val / holdout (bar 0.0016 on both) | PROMISING | member rule 0.9 / 0.1: val / holdout, pooled z |
|---|---|---|---|---|---|---|---|---|
| `b0` | B0 reference (infra `ts_lr020_s0`): the 177 features, `category_id` native with default CTR | 510 (516) | 68 | 0.38150 / 0.30462 | 0.847 / 0.847 | +0.00117 ± 0.00038 / +0.00051 ± 0.00017 | no | -0.00006 ± 0.00039 / -0.00008 ± 0.00022, -0.36 (no pass) |
| `b0_s1` | B0 recipe, seed 1 (62-minute budget) | 352 (362) | 62 | 0.38132 / 0.30402 | 0.843 / 0.844 | +0.00096 ± 0.00035 / +0.00051 ± 0.00017 | no | -0.00060 ± 0.00039 / +0.00010 ± 0.00021, -0.75 (no pass) |
| `b3_abc_s0` | B3: B0 + 70 hand-made columns (blocks a, b, c: user x cluster / category statistics, transition and user-cluster x item-cluster lifts, nearest-train-item statistics) | 495 (520) | 101 | 0.38362 / 0.30480 | 0.845 / 0.846 | +0.00147 ± 0.00043 / +0.00080 ± 0.00021 | no | -0.00013 ± 0.00040 / +0.00017 ± 0.00022, +0.30 (no pass) |
| `b3_abc_s1` | B3, seed 1 (80-minute budget) | 325 (425) | 80 | 0.38350 / 0.30510 | 0.841 / 0.842 | +0.00144 ± 0.00042 / +0.00081 ± 0.00021 | no | +0.00023 ± 0.00040 / +0.00002 ± 0.00022, +0.47 (no pass) |
| `b1_mcc4_s0` | B1: B0 + cl256 / cl1024 / cl4096 native, `has_time`, rows shuffled inside feeds, 255 CTR borders, max_ctr_complexity 4 | 516 (520) | 70 | 0.38148 / 0.30480 | 0.851 / 0.850 | +0.00096 ± 0.00034 / +0.00054 ± 0.00017 | no | -0.00039 ± 0.00040 / +0.00004 ± 0.00022, -0.57 (no pass) |
| `b1_mcc2_s0` | B1 with max_ctr_complexity 2 (45-minute budget) | 290 (294) | 45 | 0.38051 / 0.30415 | 0.845 / 0.845 | +0.00088 ± 0.00033 / +0.00052 ± 0.00017 | no | -0.00025 ± 0.00040 / +0.00009 ± 0.00021, -0.13 (no pass) |
| `b2_item_s0` | B2: B0 + `item_id` native (`has_time`, 255 CTR borders, complexity 1, model_size_reg 0, ctr_leaf_count_limit 2M), pool xm + xs (436,320 fit feeds) | 294 (394) | 57 | 0.37947 / 0.30159 | 0.834 / 0.831 | +0.00112 ± 0.00037 / +0.00051 ± 0.00017 | no | -0.00087 ± 0.00039 / +0.00024 ± 0.00021, -0.75 (no pass) |
| `bundle(b0 + b3_abc_s0)` | member-equal mean of per-feed z (information only: no arm is PROMISING) |  |  | 0.38361 / 0.30589 | 0.851 / 0.851 | +0.00139 ± 0.00042 / +0.00068 ± 0.00019 | no | -0.00008 ± 0.00039 / +0.00015 ± 0.00021, +0.35 (no pass) |
| `bundle(b0 + b0_s1 + b3_abc_s0 + b3_abc_s1)` | member-equal mean of per-feed z (information only: no arm is PROMISING) |  |  | 0.38432 / 0.30593 | 0.850 / 0.851 | +0.00134 ± 0.00041 / +0.00069 ± 0.00019 | no | +0.00023 ± 0.00039 / +0.00004 ± 0.00021, +0.56 (no pass) |
| `bundle(b3_abc_s0 + b3_abc_s1)` | member-equal mean of per-feed z (information only: no arm is PROMISING) |  |  | 0.38476 / 0.30575 | 0.846 / 0.847 | +0.00151 ± 0.00043 / +0.00083 ± 0.00021 | no | +0.00023 ± 0.00039 / +0.00007 ± 0.00021, +0.65 (no pass) |
| `bundle(b0 + b0_s1)` | member-equal mean of per-feed z (information only: no arm is PROMISING) |  |  | 0.38227 / 0.30483 | 0.848 / 0.849 | +0.00110 ± 0.00037 / +0.00053 ± 0.00017 | no | -0.00030 ± 0.00039 / +0.00003 ± 0.00021, -0.45 (no pass) |
| `bundle(b0 + b1_mcc4_s0)` | member-equal mean of per-feed z (information only: no arm is PROMISING) |  |  | 0.38219 / 0.30566 | 0.852 / 0.852 | +0.00110 ± 0.00037 / +0.00054 ± 0.00017 | no | -0.00020 ± 0.00040 / +0.00007 ± 0.00021, -0.11 (no pass) |
| `bundle(b0 + b1_mcc4_s0 + b3_abc_s0)` | member-equal mean of per-feed z (information only: no arm is PROMISING) |  |  | 0.38426 / 0.30628 | 0.854 / 0.854 | +0.00127 ± 0.00040 / +0.00065 ± 0.00019 | no | -0.00015 ± 0.00039 / +0.00007 ± 0.00021, -0.04 (no pass) |

## 2. B3: hand-made label statistics from all 3.12M non-holdout train feeds

### 2.1 What the 177 features already have [D, `src/features_tte.py`]

`tte_u_{imp,ctr,coec}` (user), `tte_uc_{imp,ctr}` (user x category), `tte_ui_{imp,pos}` (same user and item shown / clicked before), `tte_it_*` (item), `tte_c_ctr` (category): expanding window by DAY (a row on day d reads days < d; val / test read all of train), position prior over all train rows, the 30K holdout feeds are among the sources. Not present: anything at cluster level for the user, position-normalised user x category, any recency weighting, any transition table, any neighbour statistic. `feats_v2/clu_/build.py` once built user-cluster x item-cluster and history-cluster -> candidate-cluster lifts (not in the 177; a 36-column family read -0.00014 ± 0.00072 in a LightGBM, `data/clu_work/ev_clu_labelbased_s3.log` [D]); those files include holdout labels, so nothing of them is reused.

### 2.2 The four blocks built here (all [M]; builders `b3_build_{a,b,c,d}.py`, columns in `feat/{a,b,c,d}_<set>.npy` for xm / hold / val / test)

| block | columns | key | sources | build time |
|---|---|---|---|---|
| a | 39 | the USER's own earlier impressions: clicks / impressions / clicks-over-expected of the candidate's category and 256 / 1024 / 4096 cluster; recency-weighted (half-life 2 days) variants; cosine of the candidate to the user's earlier clicked and shown-not-clicked candidates (max, mean, kernel sums with tau 0.05 / 0.2, kernel COEC, top-5); the same item shown / clicked before and hours since | the user's non-holdout train feeds with a strictly earlier timestamp (128 s resolution; val / test: all of them) | 23 s |
| b | 25 | TRANSITION click rates P(click on candidate key c given history key h) for category, 256- and 1024-cluster, separately for the newest 1 / 5 / 20 history events (mean and pooled log lift over the candidate-key base rate, max over 20); user-cluster x item-cluster lifts (256 x 1024, 1024 x 1024, 64 x 4096); candidate-key base rates | every non-holdout train feed of the 4 OTHER user folds on EARLIER days | 28 s |
| c | 6 | NEAREST-TRAIN-ITEM statistics: the 20 most similar index items of the candidate's own 4096-cluster (own item skipped): clicks over expected of the top 5 / top 20 (plain and cosine-weighted), their expected clicks, and two label-free columns (mean cosine of the 20, highest cosine) | index = a SAMPLE of train items: per cluster the at most 1,500 most-shown items with ≥ 5 non-holdout impressions (5,862,592 of 38,352,979 train items, 67.6% of impressions); labels from other folds, earlier days | 191 s |
| d | 11 | NEAR-DUPLICATE statistics on the same index: number of index items with cosine > 0.99 / 0.97 / 0.95 / 0.90, cosine of the 2nd / 5th / 10th neighbour (label-free), clicks over expected and expected clicks of the items above 0.97 / 0.95 | as c | 330 s (4 threads) |

Support of block a (share of feeds whose user has a source feed) [M]: xm 19.0%, holdout 18.0%, val 30.3%, test 28.9%; 1.4 source feeds (about 130 rows) per covered train feed, 1.6 on val.
**Block a is fresh in training and stale at apply time** [M, `out/block_a_recency.txt`]: hours between the target feed and the user's most recent source feed: median 72 h on xm and 70 h on the holdout (25% under 24 h, 3.4% under 1 h); **240 h on val and 357 h on test, none under 24 h** (train ends 2 days before val and 6 days before test).

### 2.3 Pre-screens (train labels only)

Pool proxy on train feeds = `cbx.nn_input` (cross-fitted teacher, z per feed: "z_oof"); W16 itself is in-sample on train feeds.

(i) Per column, conditional logit of [z_oof, column z] over [z_oof] on 40,000 random xm feeds (`b3_uni.py`, `logs/b3_uni_*.log`) [M]:

| block | best columns: nats over z_oof | for scale: existing features |
|---|---|---|
| a | `ua_item_imp` +0.00022 ± 0.00008 (z -4.5; +0.0020 on the 10.9% of feeds where it varies), `ue_k20_coec` +0.00019 ± 0.00008 (+0.0010 on the 18.8% covered feeds), `ua_cat_coec` +0.00012 ± 0.00007; 27 of the 36 within-feed-varying columns below 0.0001 | `tte_it_coec` +0.00040 ± 0.00012 |
| b | `tr_cat_l20m` +0.00025 ± 0.00010, `tr_cat_l5m` +0.00022, `tr_c1024_l20m` +0.00020, `ux_u1024c1024` +0.00012 | `tte_ui_imp` +0.00018 ± 0.00007 |
| c | label-FREE `kn_cos1` +0.00053 ± 0.00014 (z -7.5), `kn_cos20` +0.00051; label-aware `kn_coec20` +0.00006 ± 0.00005, `kn_coec5` 0.00000 | `f3` +0.00017 ± 0.00008 |
| d | label-FREE `kn_cos2` +0.00058 ± 0.00015, `dup_n97` +0.00034; label-aware `dup_coec97` +0.00001 ± 0.00002, `dup_coec95` +0.00005 ± 0.00005 | `tte_uc_ctr` +0.00009 ± 0.00006 |

(ii) Per block, the rule declared in PROGRESS.md before the read: LightGBM lambdarank (63 leaves, 300 rounds, 30,000 fit feeds) on [z_oof, 177, block] against [z_oof, 177], read on the 13,527 selection feeds by nats over z_oof; a block is dropped when its increment is below 1 paired SE (`b3_prescreen.py --mode joint`, `logs/b3_prescreen_joint.log`) [M]:

| fit | columns | nats over z_oof on the selection feeds | increment over the reference | declared rule |
|---|---|---|---|---|
| reference [z_oof, 177] | 178 | +0.00019 ± 0.00015 | | |
| + a | 217 | +0.00019 ± 0.00015 | +0.00000 ± 0.00016 (z 0.0) | DROP |
| + b | 203 | +0.00032 ± 0.00019 | +0.00013 ± 0.00019 (z 0.7) | DROP |
| + c | 184 | +0.00021 ± 0.00016 | +0.00002 ± 0.00017 (z 0.1) | DROP |
| + a + b + c | 248 | +0.00034 ± 0.00020 | +0.00015 ± 0.00020 (z 0.7) | DROP |

**By the declared rule no block survives.** An increment of the size the screen needs (+0.0005 on val, +0.0011 on the holdout over B0) would have read 2.5 to 5 SE.
Block d was screened by (i) only: its label-aware columns are dead and its label-free columns repeat `kn_cos1`; it is in no arm.

Decision written before the run (PROGRESS.md 01:46): the plan's "B3 = B0 + surviving columns" has no surviving column; because the full-scale fit on the real witnesses is the decisive read and the budget allowed it, ONE confirmation arm with all 70 columns of a + b + c was run (`b3_abc_s0`), with the stated expectation "not promising, at most +0.0003 nats over B0". No block subset was tried after its read.

(iii) Evaluation-only read on the witnesses (`b3_witness_info.py`; used for no choice): a LINEAR composite of each block, coefficients fitted on 40,000 train feeds over z_oof, then read like a model score [M]:

| composite | nats over z_W16, val | holdout | over [z_W16, z_B0], val | holdout |
|---|---|---|---|---|
| block a (36 columns) | +0.00001 ± 0.00004 | +0.00011 ± 0.00007 | +0.00001 ± 0.00004 | +0.00011 ± 0.00007 |
| block b (25) | +0.00022 ± 0.00016 | +0.00009 ± 0.00007 | +0.00004 ± 0.00007 | +0.00002 ± 0.00004 |
| block c (6) | +0.00018 ± 0.00015 | +0.00029 ± 0.00012 | +0.00026 ± 0.00018 | +0.00037 ± 0.00014 |
| a + b + c (67) | +0.00034 ± 0.00020 | +0.00041 ± 0.00015 | +0.00023 ± 0.00017 | +0.00036 ± 0.00014 |
| B0 itself | +0.00117 ± 0.00038 | +0.00051 ± 0.00017 | | |

Best single column over z_W16: val `tr_cat_base` +0.00037 ± 0.00021; holdout `kn_cos1` +0.00028 ± 0.00012 (negative coefficient), `ue_k20_coec` +0.00018 ± 0.00010.

The same read with the 67 columns split by KIND instead of by block (`b3_witness_split.py`, written after the B3 read; evaluation only) [M]:

| composite (coefficients from train) | columns | over [z_W16, z_B0], val | holdout |
|---|---|---|---|
| label-free density: `kn_cos1`, `kn_cos20` (cosine to the nearest much-shown train items) | 2 | +0.00018 ± 0.00015 | +0.00032 ± 0.00013 (z 4.9) |
| every click-FREE column: density + exposure counts (impressions, shares, kernel mass, expected clicks of the neighbours) | 16 | +0.00026 ± 0.00017 (z 3.0) | +0.00042 ± 0.00015 (z 5.6) |
| **every CLICK statistic** (CTR / COEC / lift columns of blocks a, b, c) | 51 | **+0.00004 ± 0.00007** | **+0.00006 ± 0.00006** |
| click statistics of the user block a alone | 23 | 0.00000 ± 0.00001 | +0.00011 ± 0.00007 |
| all other columns than the two density columns | 65 | +0.00009 ± 0.00010 | +0.00015 ± 0.00009 |
The density signal next to the finished B3 model (`b3_witness_density.py`, evaluation only) [M]: the composite of `kn_cos1`, `kn_cos20` still adds +0.00007 ± 0.00009 (val) / **+0.00022 ± 0.00011 (holdout, z 4.0)** over [z_W16, z_B3], although both columns are inputs of B3 (importance rank 10 and 15). The nine label-free density columns of blocks c + d together read the same (+0.00007 / +0.00023): block d adds nothing to the two columns of block c.

Readings: (1) block a, the only block made of information the pool cannot have (what the user was shown and did not click), is worth nothing on val (+0.00001) and +0.0001 on the in-period holdout: its sources are 10 days old on val (section 2.2). (2) Block b is already inside B0 (increment over [z_W16, z_B0] +0.00004 / +0.00002). (3) What block c adds is its two LABEL-FREE columns: a candidate that has close neighbours among the much-shown train items is clicked less than the pool predicts. (4) **In the linear read the click statistics of all three blocks together add +0.00004 / +0.00006 nats to [z_W16, z_B0]; the click-free columns add +0.00026 / +0.00042.** The label information this family was built to find is not there; what moves is catalogue density and exposure, which is not label information. (5) The density signal is a correction to the POOL (it stays visible next to a tree that has the columns), so a tree member is a lossy carrier for it: if it is worth anything it is as a direct input of the neural students or of a second-level model, at a size of 0.0002 to 0.0003 nats.

### 2.4 The full-scale arm `b3_abc_s0`

B0 recipe + the 70 columns of blocks a, b, c (247 features); 520 trees built, 495 used, 101 minutes (11.7 s per tree) [M].

| read | val | holdout |
|---|---|---|
| level | 0.38362 ± 0.00199 | 0.30480 ± 0.00108 |
| level minus B0 (paired) | +0.00212 ± 0.00088 | +0.00019 ± 0.00050 |
| **NATS (screen, bar 0.0016)** | **+0.00147 ± 0.00043** | **+0.00080 ± 0.00021** |
| NATS minus NATS(B0) (paired per feed) | +0.00030 ± 0.00019 | +0.00028 ± 0.00009 |
| NATS on feeds whose user has an earlier train feed (30.3% / 18.0% of feeds) | +0.00207 ± 0.00077 (+0.00078 ± 0.00035 over B0) | +0.00091 ± 0.00046 (+0.00047 ± 0.00021 over B0) |
| NATS on the other feeds | +0.00121 ± 0.00052 (+0.00009 ± 0.00023 over B0) | +0.00077 ± 0.00023 (+0.00024 ± 0.00010 over B0) |
| member rule 0.9 z(W16) + 0.1 z(X) | -0.00013 ± 0.00040 | +0.00017 ± 0.00022 (pooled z +0.30: no pass) |

- **Not PROMISING**: the holdout NATS is half the bar (3.8 SE below it). The arm adds about +0.0003 nats to B0 on both witnesses (holdout z 3.1), which is what the linear composites of 2.3 (iii) predicted (+0.00023 / +0.00036) and what was written down before the run ("at most about +0.0003").
- Importance of the 70 new columns: 8.9 of 100 = block c 4.65 (`kn_cos1` 2.03 and `kn_cos20` 1.76, the two LABEL-FREE columns, rank 10 and 15 of 247), block b 3.64 (spread over 25 columns, largest 0.42), block a 0.58.
- Where the gain sits: about +0.0005 to +0.0008 nats on the feeds whose user has an earlier feed (z 2.2 on each witness) and +0.0001 to +0.0002 on the rest. A tree does use the user's earlier impressions where they exist, but they exist for 18 to 30% of the feeds and they are 10 to 15 days old at apply time.
- The val level 0.38362 is the highest hard-label tree level on file (record 0.38201 / 0.38268 [D]). At seed 0 the holdout level did not move (+0.0002 ± 0.0005); the seed-1 replicate (section 5) reads 0.38350 / 0.30510, and two seeds of B3 against two seeds of B0 read +0.0025 ± 0.0008 val / +0.0009 ± 0.0004 holdout: a level gain of about 0.002 on val and 0.001 on the holdout.
- Train-side selection NDCG (13,527 feeds) at 500 trees: B0 0.30708, `b1_mcc4_s0` 0.30765, `b3_abc_s0` 0.30896.

## 3. B1: native CatBoost CTRs on candidate-level keys

### 3.1 Finding: the default native CTR is blind at our click rate [M]

On CPU the `Borders` statistic (clicks + prior) / (count + 1) is cut by `CtrBorderCount` uniform borders over [0, 1]. The default is 15, i.e. cuts at multiples of 0.067. With a click rate of 1.3% almost every key with support lies below the first cut.
Synthetic check (`out/ctr_border_check.txt`: Logloss, 600,000 rows, one key with 200 values whose click rates run from 0.4% to 3.0%, 200,000 eval rows):

| setting | eval Logloss (constant model 0.08673, oracle 0.08488) | distinct predicted levels over the 200 keys | corr(prediction, true rate) |
|---|---|---|---|
| default CTR (15 borders) | 0.08673 | 2 | 0.057 |
| `CtrBorderCount=255` | 0.08504 | 23 | 0.947 |
| `CtrBorderCount=255`, `has_time=True` | 0.08503 | 25 | 0.948 |

Consequences: (a) `category_id` as a native categorical in every CPU CatBoost of this track (B0 included) carries close to no target statistic; (b) B1 and B2 were run with `simple_ctr` AND `combinations_ctr` = `Borders:CtrBorderCount=255:...:Prior=0/1:Prior=0.5/1:Prior=1/1` + `Counter` (`fb_common.CTR255`; CatBoost warns that `simple_ctr` alone does not change the combinations); (c) even 255 borders give a resolution of 0.39 percentage points, which hand-made float columns do not have. A default-CTR arm was not run at full scale.

### 3.2 Leak test on noise labels (before the real run)

Configuration under test = the arm's: `has_time=True`, feeds in time order, rows shuffled inside each feed, `CTR255`, max_ctr_complexity 2 / 4, YetiRank d8 lr 0.2 l2 5.
Design notes [M]: with labels shuffled inside feeds and candidate-level keys a tree DOES lower its train loss by ordinary in-sample fit, so "learn loss must not move" (the docs_feat test, where the key was feed-constant) is not a usable criterion; and comparing with labels shuffled across the pool is confounded under a ranking metric (different chance levels: learn NDCG 0.20 against 0.40). The reads below therefore compare with a leak-free twin (the same keys as hand-made ordered statistics computed from earlier FEEDS only) and with the random-score level.

| test | result |
|---|---|
| 2,000 feeds, keys only, hand-made twins, 5 label seeds (`b1_noise2.py`) | YetiRank learn NDCG after 100 trees: group twin (cannot leak) 0.2651, prefix twin (same-feed earlier rows included, 254 borders) 0.2804: **+0.0153 ± 0.0018** (z 8.6); mean within-feed correlation of the statistic with the noise label -0.005 to -0.007 (prefix) against -0.001 to 0.000 (group). The same-feed prefix of a candidate-level key IS a channel at this size when the statistic is finely binned |
| 2,000 feeds, keys only, native CTR255, 3 label seeds (`b1_noise3.py`) | YetiRank learn NDCG 0.2766 ± 0.0061 (mcc 2), 0.2776 ± 0.0050 (mcc 4): +0.010 ± 0.008 and +0.011 ± 0.007 over the group twin of the same 3 seeds (0.2662; not significant, of the size of the prefix twin). Logloss learn loss, labels inside feeds minus across the pool: -0.00024 ± 0.00028, -0.00020 ± 0.00028 |
| positive control: + `user_id` native (feed-constant) | YetiRank learn NDCG **0.558** (twin 0.266); Logloss excess 0.0043 ± 0.0005: the test has power, a user key leaks massively on CPU (confirms docs_feat 5.3) |
| 2,000 feeds, exact arm inputs (177 + keys), 3 label seeds | native mcc 2 minus group twin +0.002 ± 0.008; native mcc 4 -0.014 ± 0.007; prefix twin minus group twin +0.0002 ± 0.0067: no measurable excess once the 177 features are there |
| **120,591 feeds (the real fit pool), keys only, native CTR255, mcc 4** (`b1_noise_full.py`) | YetiRank learn NDCG after 30 trees **0.22763 against the random-score level 0.22741 (+0.0002)**: at the real scale the configuration cannot rank noise labels. (Logloss learn loss inside-feed 0.066835 against across-pool 0.067101: confounded, the row base rate depends on the feed under inside-feed labels, which keys may learn honestly; a ranking loss is immune) |

Verdict of the leak test [M]: at 2,000 feeds a small prefix effect of candidate-level keys exists in principle (twin +0.014, native not distinguishable from it); at the scale of the real run it is not measurable (+0.0002 over chance). The real arm was run. Whatever is left can only cost the arm (statistics seen in training are slightly more informative than at apply time); it cannot inflate a val / holdout read.

### 3.3 Result `b1_mcc4_s0` [M]

520 trees built, 516 used, 70 minutes (8.1 s per tree). Importance (PredictionValuesChange): `category_id` 1.94 (rank 15; it was outside the top 16 of B0), `cl256` 0.36, `cl1024` 0.21, `cl4096` 0.20.
Level against B0: val -0.00002 ± 0.00078, holdout +0.00018 ± 0.00046. NATS against B0 (paired): val -0.00021 ± 0.00014, holdout +0.00002 ± 0.00006.
The train-side selection curve had shown +0.0005 to +0.0010 over B0 at equal trees; it did not transfer. **Native cluster CTRs are a null.**

### 3.4 Result `b1_mcc2_s0` (max_ctr_complexity 2, 45-minute budget) [M]

Budget stop at 294 trees, 290 used. Level 0.38051 / 0.30415, NATS +0.00088 ± 0.00033 / +0.00052 ± 0.00017. B0 cut to the same 290 trees: level 0.38189 / 0.30360, NATS +0.00109 / +0.00045. Importance: `category_id` 1.98, `cl256` 0.41, `cl1024` 0.04, `cl4096` 0.04. Null, like complexity 4.

Both B1 arms are at or below B0 on val and equal on the holdout: by the family rule of the task (first two arms below half the bar on the holdout, at B0 on val) no further B1 variant was run.

## 4. B2: native item_id CTR

`b2_item_s0`: `item_id` as a native categorical (`has_time`, CTR255, `max_ctr_complexity` 1, `model_size_reg` 0, `ctr_leaf_count_limit` 2,000,000), pool = xm + xs fit feeds in time order (436,320 feeds, 22,581,927 rows), rows shuffled inside feeds. An item never repeats inside a feed, so the within-feed prefix does not exist for this key. Stopped by the overfitting detector at 394 trees (294 used), 57 minutes [M].

| read | val | holdout |
|---|---|---|
| level | 0.37947 | 0.30159 |
| level minus B0 | -0.00203 ± 0.00097 | -0.00302 ± 0.00058 |
| level of B0 cut to the same 294 trees | 0.38186 | 0.30370 |
| NATS | +0.00112 ± 0.00037 | +0.00051 ± 0.00017 |
| NATS minus NATS(B0) (paired) | -0.00005 ± 0.00020 | 0.00000 ± 0.00009 |
| member rule | -0.00087 ± 0.00039 | +0.00024 ± 0.00021 (pooled z -0.75) |

Importance of `item_id` 0.57 of 100 (`category_id` 2.61). As stated before the run, the native item statistic is dominated by the hand-made `tte_it_*` (all 3.15M train feeds against 436K here): it adds no information (NATS difference 0.0000). The level loss is the known cost of the xm + xs pool (record A1 - A0: -0.00125 ± 0.00074 val, -0.00187 ± 0.00044 holdout [D, `nn_x/w25/out/results.txt`]); item_id on xm alone was not run, so the two effects are not separated here.

## 5. B4: bundles and the member rule

**No arm is PROMISING, so B4 as planned (3 seeds of the PROMISING arm, the declared 3 + 3 bundle, its member-rule read as a candidate) did not trigger.** What was run instead, declared in PROGRESS.md at 03:35 as information only: one more seed of the reference and of the best arm, and the bundle reads.

Seed replicates [M]:

| pair | NATS val | NATS holdout | level val | level holdout |
|---|---|---|---|---|
| `b0_s1` (352 trees) minus B0 seed 0 cut to 352 trees | -0.00015 ± 0.00012 | +0.00005 ± 0.00005 | -0.00089 ± 0.00074 | +0.00024 ± 0.00042 |
| `b3_abc_s1` (325 trees) minus `b3_abc_s0` cut to 325 trees | +0.00008 ± 0.00014 | +0.00007 ± 0.00007 | +0.00002 ± 0.00074 | +0.00051 ± 0.00044 |
| `b3_abc_s0` minus B0 (seed 0 both) | +0.00030 ± 0.00019 | +0.00028 ± 0.00009 | +0.00212 ± 0.00088 | +0.00019 ± 0.00050 |
| `b3_abc_s1` minus `b0_s1` (seed 1 both) | +0.00048 ± 0.00018 | +0.00030 ± 0.00009 | +0.00218 ± 0.00084 | +0.00108 ± 0.00048 |
| bundle(B3 s0, B3 s1) minus bundle(B0 s0, B0 s1) | **+0.00041 ± 0.00016** | **+0.00030 ± 0.00008** | **+0.00250 ± 0.00080** | **+0.00093 ± 0.00042** |

- The B3 read replicates: seed 1 gives NATS +0.00144 ± 0.00042 / +0.00081 ± 0.00021 (seed 0: +0.00147 / +0.00080). The seed-to-seed spread of NATS is about 0.0001 to 0.00015 on val and 0.00005 to 0.00007 on the holdout.
- B3 over B0 is a real, small effect: +0.0003 to +0.0005 nats and +0.002 of val level (+0.001 of holdout level) at two seeds each. It covers 0.0003 of the 0.0011 that B0 lacks on the holdout: the holdout NATS stays at half the bar at both seeds.

Bundle reads (member-equal mean of per-feed z; rows of section 1) [M]:

| bundle | level val / holdout | NATS val / holdout | member rule: val / holdout, pooled z |
|---|---|---|---|
| declared form at 2 seeds: B0 s0 + B0 s1 + B3 s0 + B3 s1 | 0.38432 / 0.30593 | +0.00134 ± 0.00041 / +0.00069 ± 0.00019 | +0.00023 ± 0.00039 / +0.00004 ± 0.00021, +0.56 (no pass) |
| B3 s0 + B3 s1 | 0.38476 / 0.30575 | +0.00151 ± 0.00043 / +0.00083 ± 0.00021 | +0.00023 ± 0.00039 / +0.00007 ± 0.00021, +0.65 (no pass) |
| B0 + B3 (seed 0) | 0.38361 / 0.30589 | +0.00139 ± 0.00042 / +0.00068 ± 0.00019 | -0.00008 ± 0.00039 / +0.00015 ± 0.00021, +0.35 (no pass) |
| B0 s0 + B0 s1 | 0.38227 / 0.30483 | +0.00110 ± 0.00037 / +0.00053 ± 0.00017 | -0.00030 ± 0.00039 / +0.00003 ± 0.00021, -0.45 (no pass) |
| B0 + B1 + B3 (seed 0) | 0.38426 / 0.30628 | +0.00127 ± 0.00040 / +0.00065 ± 0.00019 | -0.00015 ± 0.00039 / +0.00007 ± 0.00021, -0.04 (no pass) |

- **Bundling raises the level, not the information**: two seeds of B0 read 0.00110 / 0.00053 (one seed 0.00117 / 0.00051); mixing B0 into B3 lowers the NATS of B3. The best bundle on the screen is B3 with itself (0.00151 / 0.00083), still below the bar on both witnesses.
- **Member rule: no pass anywhere.** The best pooled z is +0.65 (needs +2 with both deltas ≥ 0); the largest 0.9 / 0.1 blend gain is +0.00023 ± 0.00039 on val and +0.00017 ± 0.00022 on the holdout.
- **STOP rule, family B part**: no arm and no bundle of family B reaches 0.0016 nats on both witnesses. The best is 0.00151 ± 0.00043 (val) / 0.00083 ± 0.00021 (holdout). Family B does not keep the track open; the decision for the track also needs families S and N.

## 5a. Freshness: the lead's notice of 02:19:49Z (read by me at 05:10Z, after every arm had been built)

The notice appended to `PREDECLARED.txt` says that count and label statistics are FRESH on in-period rows (training feeds, holdout) and STALE on val (2 to 5 days after the last train day) and test (6 to 9 days), that train-row statistics should be built with the test lag, that every new column's distribution should be compared between train, val and test rows, and that val M ≥ 50 should be reported separately. My arms were built without the lag (a train row reads everything up to the previous day, block a up to the previous 128 seconds). What I could still do, and what not:

**(1) val M ≥ 50 separately** (`read_m50.py`; 8,530 of the 14,423 val feeds; the holdout is M ≥ 50 by construction) [M]:

| model | val M >= 50: level | NATS | member rule 0.9 / 0.1 | val M < 50: NATS | member rule |
|---|---|---|---|---|---|
| `b0` | 0.30253 (W16 0.32060) | +0.00090 ± 0.00041 | +0.00020 ± 0.00044 | +0.00156 ± 0.00073 | -0.00043 ± 0.00072 |
| `b0_s1` | 0.30242 (W16 0.32060) | +0.00077 ± 0.00038 | -0.00027 ± 0.00043 | +0.00122 ± 0.00064 | -0.00108 ± 0.00073 |
| `b3_abc_s0` | 0.30396 (W16 0.32060) | +0.00110 ± 0.00046 | +0.00022 ± 0.00045 | +0.00201 ± 0.00083 | -0.00065 ± 0.00074 |
| `b3_abc_s1` | 0.30397 (W16 0.32060) | +0.00119 ± 0.00048 | +0.00015 ± 0.00045 | +0.00181 ± 0.00078 | +0.00034 ± 0.00074 |
| `b1_mcc4_s0` | 0.30256 (W16 0.32060) | +0.00073 ± 0.00037 | -0.00017 ± 0.00042 | +0.00126 ± 0.00065 | -0.00070 ± 0.00076 |
| `b1_mcc2_s0` | 0.30129 (W16 0.32060) | +0.00067 ± 0.00036 | +0.00014 ± 0.00045 | +0.00118 ± 0.00063 | -0.00081 ± 0.00073 |
| `b2_item_s0` | 0.29930 (W16 0.32060) | +0.00081 ± 0.00039 | -0.00040 ± 0.00042 | +0.00163 ± 0.00074 | -0.00155 ± 0.00074 |
| `b3_abc_s0+b3_abc_s1` | 0.30470 (W16 0.32060) | +0.00118 ± 0.00048 | +0.00034 ± 0.00043 | +0.00198 ± 0.00082 | +0.00004 ± 0.00073 |
| `b0+b0_s1+b3_abc_s0+b3_abc_s1` | 0.30484 (W16 0.32060) | +0.00105 ± 0.00045 | +0.00026 ± 0.00043 | +0.00177 ± 0.00078 | +0.00016 ± 0.00073 |

On the M ≥ 50 part of val every arm is further from the bar than on all of val (B3 +0.00110 against +0.00147); the val M < 50 reads are twice as noisy and carry the higher values.

**(2) Label-free distribution of the new columns, train against val and test rows** (`col_dist.py`, all 70 columns in `out/col_dist.md`, lagged train rows in `out/col_dist_lag.md`; cells = share of NaN rows, then p10 / p50 / p90 of the finite values) [M]:

| column | xm, fresh | xm, built with the test lag | holdout, fresh | val | test |
|---|---|---|---|---|---|
| `kn_cos1` (label-free) | 0.000, 0.913 / 0.972 / 0.994 | same | 0.000, 0.912 / 0.972 / 0.994 | 0.000, 0.911 / 0.972 / 0.994 | 0.000, 0.911 / 0.972 / 0.994 |
| `kn_cos20` (label-free) | 0.000, 0.863 / 0.937 / 0.980 | same | 0.000, 0.863 / 0.937 / 0.979 | 0.000, 0.861 / 0.937 / 0.980 | 0.000, 0.862 / 0.937 / 0.980 |
| `kn_ex20` (support of the neighbour statistic) | 0.086, 1.21 / 4.98 / 13.8 | 0.590, 0.90 / 3.56 / 9.94 | 0.090, 1.15 / 4.86 / 13.5 | 0.000, 5.00 / 9.49 / 21.6 | 0.000, 5.01 / 9.50 / 21.6 |
| `kn_coec20` | 0.086, 0.567 / 0.966 / 1.49 | 0.590, 0.572 / 0.959 / 1.50 | 0.090, 0.569 / 0.965 / 1.49 | 0.000, 0.564 / 0.982 / 1.45 | 0.000, 0.565 / 0.982 / 1.45 |
| `tr_cat_l20m` | 0.106, -0.241 / -0.046 / 0.156 | 0.599, -0.243 / -0.046 / 0.156 | 0.114, -0.242 / -0.047 / 0.149 | 0.014, -0.238 / -0.049 / 0.162 | 0.009, -0.237 / -0.051 / 0.154 |
| `ux_u1024c1024` | 0.120, -0.278 / -0.009 / 0.268 | 0.601, -0.263 / -0.009 / 0.269 | 0.123, -0.278 / -0.010 / 0.264 | 0.014, -0.307 / -0.007 / 0.262 | 0.003, -0.309 / -0.010 / 0.260 |
| `ua_cat_coec` | 0.805, 0.498 / 0.924 / 2.03 | 0.945, 0.506 / 0.933 / 1.98 | 0.817, 0.493 / 0.920 / 2.02 | 0.692, 0.506 / 0.941 / 2.04 | 0.709, 0.513 / 0.938 / 1.99 |
| `ue_k20_coec` | 0.805, 0.462 / 0.931 / 2.05 | 0.945, 0.472 / 0.927 / 1.98 | 0.817, 0.457 / 0.919 / 1.99 | 0.692, 0.483 / 0.958 / 2.04 | 0.709, 0.486 / 0.946 / 1.97 |
| `ua_dt_last` (hours since the user's last source feed) | 0.805, 6.5 / 72 / 217 | 0.945, 157 / 217 / 284 | 0.817, 6.4 / 70 / 211 | 0.692, 95 / 240 / 377 | 0.709, 205 / 358 / 492 |

- The ratio columns (COEC, lifts) have the same distribution on train, val and test rows; what differs is SUPPORT and AGE: val / test rows read all 15 train days (twice the expected clicks behind `kn_ex20`, no missing rows), their user sources are 10 to 15 days old, and a row of the first train day has nothing to read (8.6% NaN).
- The two label-free density columns have identical distributions on all four sets: the one signal B3 found is not a freshness artefact of this kind. (Its index of much-shown items is built from all 15 train days; see section 7.)
- **Under the test lag the user block has almost no training support**: a source feed that is 6 to 9 days old exists for **5.4% of the xm feeds** (19.0% fresh; 30% of val / test feeds have one, because they look back over the whole train window); blocks b and c have lagged statistics for 41% of the xm feeds (91% fresh).

**(3) Train rows built with the test lag** (`FB_LAG=1` builds of blocks a, b, c: `feat/lag_{a,b,c}_{xm,hold}.npy`; a train row on day d reads days ≤ d - L only, L drawn from 6 to 9 per feed; block a: the user's feeds of those days) and read in LINEAR form (`lag_read.py`: composite coefficients fitted on 40,000 xm feeds over z_oof from fresh or from lagged train rows, read over [z_W16, z_B0]) [M]:

| composite | witness | coefficients from FRESH train rows | coefficients from LAGGED train rows |
|---|---|---|---|
| all 67 columns | val (its natural lag, 2 to 5 days) | +0.00023 ± 0.00017 | +0.00028 ± 0.00018 |
| click statistics (51) | val | +0.00004 ± 0.00007 | +0.00013 ± 0.00013 |
| user block a, click statistics (23) | val | 0.00000 ± 0.00001 | +0.00012 ± 0.00012 |
| click-free columns (16) | val | +0.00026 ± 0.00017 | +0.00024 ± 0.00017 |
| all 67 columns | holdout, FRESH columns | +0.00036 ± 0.00014 | +0.00022 ± 0.00011 |
| click statistics (51) | holdout, fresh | +0.00006 ± 0.00006 | +0.00003 ± 0.00004 |
| user block a, click statistics (23) | holdout, fresh | +0.00011 ± 0.00007 | +0.00013 ± 0.00008 |
| click-free columns (16) | holdout, fresh | +0.00042 ± 0.00015 | +0.00032 ± 0.00013 |
| all 67 columns | holdout, columns rebuilt with the 6 to 9 day lag | +0.00037 ± 0.00014 | +0.00033 ± 0.00013 |
| click statistics (51) | holdout, lagged | +0.00004 ± 0.00005 | +0.00003 ± 0.00004 |
| user block a, click statistics (23) | holdout, lagged | +0.00009 ± 0.00007 | +0.00011 ± 0.00007 |
| click-free columns (16) | holdout, lagged | +0.00040 ± 0.00015 | +0.00037 ± 0.00014 |

- Lag-matched training moves the val read of the click statistics from +0.00004 to +0.00013 ± 0.00013 (user block: 0.00000 to +0.00012 ± 0.00012, z 2.1 on its coefficient): a mismatch existed, and removing it is worth about 0.0001 nats.
- On a holdout whose columns are as stale as test, the click statistics read +0.00003 to +0.00004; the click-free columns keep their +0.0004. The density signal survives the lag; the label statistics have nothing to lose.

**NOT run: a B3 tree fitted on the lagged columns** (one fit is 100 minutes; the notice was read 50 minutes before my budget ended). What bounds its value: (i) the in-period holdout is the FAVOURABLE case for these statistics (fresh inputs, matched to training), and there B3 reads +0.00080, half the bar; (ii) in linear form the lag-matched click statistics are worth +0.0001 nats on val and +0.00003 on a stale-matched holdout, against the +0.0005 (val) and +0.0011 (holdout) a tree would have to add to B0; (iii) nothing in family B is PROMISING on either witness, so there is no "holdout yes, val no" artefact to remove.

## 6. Method

- Data, pools, selection split, evaluation: `nn_x/w12/cb/infra/cbx.py` (`cbx.load`, `cbx.make_pool`, `cbx.trainside_split`, `cbx.run_arm`); every arm was dry-run with `max_feeds=3000` first (`preds/dry/`).
- B1 / B2 need rows shuffled inside feeds and (B2) a time-ordered union of two sets: `fb_common.ShufDS` and `fb_common.TimeCat` are thin views that serve every per-row array through a permutation, so `cbx.make_pool` / `cbx.run_arm` run unchanged. Feeds stay in file order = time order (asserted).
- Screen: `nats.py` (conditional logit of the label inside the feed on [z_W16, z_X]; 2 parameters fitted on the witness). Paired NATS differences between two arms = per-feed difference of their losses (`read_arms.py`).
- Statistics of blocks b, c, d use the 5 user folds of the cross-fitted teacher (`cbx.xfold_of`) and 15 day cuts: tables are accumulated per (fold, day), a row reads "other folds, earlier days". Expected clicks = position prior (click rate by production position, 512 numbers, from non-holdout train rows).
- B0 at fixed tree counts for like-for-like reads of budget-stopped arms: `b0_trunc.py` re-predicts the infra model file (510 trees reproduce the infra predictions exactly) [M]: NATS 200 trees 0.00101 / 0.00042, 300: 0.00110 / 0.00044, 400: 0.00113 / 0.00049, 510: 0.00117 / 0.00051.

## 7. Leak statement per arm

| arm | why it is clean |
|---|---|
| `b1_mcc4_s0`, B1 arms | Native statistics are computed by CatBoost from the fit pool only (120,591 xm feeds: no holdout feed, no val / test row); keys are candidate-level (category, clusters), never a feed- or user-constant key; `has_time` with feeds in time order, so a row never reads a later feed; rows shuffled inside feeds; the within-feed prefix was tested on noise labels at the real scale (+0.0002 over chance). Selection of the tree count on train-side feeds |
| `b2_item_s0` | As B1 with `item_id` as the key and the pool = xm + xs fit feeds in time order (holdout excluded by construction, asserted); an item never repeats inside a feed, so there is no within-feed prefix; a row reads earlier feeds only |
| `b3_abc_s0`, `b3_abc_s1` | Block a: the user's OWN non-holdout train feeds with a strictly earlier timestamp (asserted on 20,000 feeds per set), never the row's feed, never a later snapshot; val / test rows read train labels only. This block is same-user by design (task item B3 (a), like the existing `tte_u*`): it is time-aware, not leave-user-out. Blocks b, c: labels of other user folds on earlier days only (never the row's feed, never the same user, never a same-day or later feed), holdout feeds are never a source; block c skips the candidate's own item. No NN-derived input. No val / test / holdout label is read by any builder |
| pre-screens | `b3_uni.py`, `b3_prescreen.py`: train labels of xm feeds only. `b3_witness_info.py` and `read_arms.py` read val / holdout labels for EVALUATION (the same statistic as `nats.py`); no column, block or arm was chosen from them |

Not time-aware by design [M]: the position prior (512 numbers from 244M non-holdout train rows of all days), the cluster assignments / embeddings (label-free), and the MEMBERSHIP of the block c / d index (items with at least 5 non-holdout impressions over all 15 train days: an exposure count, no label; for a train row it includes items that were shown later). The click statistics read from that index are time-aware. The label-free columns `kn_cos1` / `kn_cos20` therefore see slightly more of the catalogue on a train row than a val / test row can; their distributions are identical on the four sets (section 5a) and their sign and size hold on val, which reads past items only.
The 177 features of record are used as they are: their `tte_*` builder does not exclude the 30K holdout feeds from its sources [D, `src/features_tte.py` has no holdout handling]; that is a property of B0 and of every model of record, not of the columns added here.

## 8. Limits and what was NOT run

NOT run, stated plainly:
- **B4 as planned (3 seeds of a PROMISING arm + the 3 + 3 bundle + the member-rule read of it)**: no arm is PROMISING, so there is no candidate. What exists instead is information: 2-arm bundles at one seed and one 2 + 2 seed bundle (section 5).
- **B13** (B1 keys + B3 columns in one model): declared at 02:16, cancelled at 02:53 before the B3 read, because B1 had just measured the keys as a null.
- **A default-CTR (15 borders) B1 arm at full scale**: shown degenerate on synthetic data (3.1) and not run.
- **Full-scale ablations of B3** (block a only, b only, c only): not run; a subset chosen after the read was ruled out in advance. The split of the gain between blocks rests on importance, on the coverage split and on the linear composites.
- **Block d in an arm**; **item_id x user-cluster statistics** (audit section 5, row 1); 4096 x 4096 transition tables; item-level co-click tables; neighbours outside the candidate's own 4096-cluster; k 50 for the neighbour statistics (k 5 and 20 only).
- **item_id as a native key on xm alone** (B2 was run on xm + xs as planned; its level loss mixes the key and the larger pool).
- B1 at max_ctr_complexity 2 and B2 ran with a reduced wall-clock budget (45 and 70 minutes); the seed replicates with 62 (B0) and 80 (B3) minutes. Three seeds of anything were not run (two of B0 and of B3).
- **A B3 tree fitted on lag-matched statistics** (lead's notice, section 5a): not run; only the linear form was read. Lag-matched variants of B1 / B2 (native CTRs cannot be lagged on CPU) were not attempted.

Limits of what was run:
- One seed per arm for B1 and B2; two seeds for B0 and B3 (section 5). Level differences below 0.002 between single tree arms are seed noise; the conclusions rest on the NATS screen and its paired per-feed differences.
- The pre-screen pool proxy is the cross-fitted teacher (3 models), not W16, and its joint-tree form had a resolution of 0.0002 nats: it could exclude an effect of the size the screen needs, not the +0.0003 that the full arm then showed.
- Block c / d index = a sample (the at most 1,500 most-shown items per 4096-cluster with at least 5 impressions: 15% of train items, 68% of impressions); the nearest neighbour is searched inside the candidate's own cluster only.
- Block a caps the sources at the 4,000 most recent rows per feed and has no value for unknown users (4% of feeds) or for users without an earlier feed (70 to 82% of feeds).
- **Block a is fresher in training than at apply time** (median 72 h against 240 h on val and 357 h on test): the in-period holdout flatters every user-level statistic, and test is staler than val.
- The B1 leak test at the real scale was run for keys only at max_ctr_complexity 4 and 30 trees (not with the 177 features, not for complexity 2).
- Test predictions exist for every arm (`preds/<tag>_test.npy`); nothing was assembled into a candidate or a submission file.
- The lead's freshness notice (file time 02:19:49Z) was read at 05:10Z: I had read the plan once at 01:10Z and did not re-read it while the arms ran. Every arm was built before I knew of it.
- Compute: two 12-thread fits of mine at a time at most. For about one minute at 01:33Z three of my processes with a 12-thread cap overlapped (a noise test, a block build and the per-column screen, which is mostly single-threaded). Light jobs (dry runs at 4 threads, `b3_build_d.py` at 4 threads, single reads) ran next to them. One launch of two background jobs in one command was refused by the permission system and was redone as two commands. Three of my own waiting loops (shell `until` loops, one with a self-matching `pgrep`) were stopped through the task tool; no process of another user was signalled, no STOP file of mine was needed.

## 9. Exact commands and full screen lines

Every job was started as (one command per job):

```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 \
  CBX_CACHE_EXTRA=<repo>/nn_x/w12/cb/famB/cache && \
  setsid nohup .venv/bin/python -u nn_x/w12/cb/famB/<script> <arguments> > nn_x/w12/cb/famB/logs/<name>.log 2>&1 < /dev/null &
```

| step | script and arguments | log |
|---|---|---|
| B1 leak tests | `b1_noise.py` (default CTR), `b1_noise2.py` (hand-made twins), `b1_noise3.py` (CTR255, exact configuration), `b1_noise_full.py 30` (120,591 feeds) | `logs/b1_noise*.log` |
| B3 columns | `b3_build_a.py`, `b3_build_b.py`, `b3_build_c.py`, `b3_build_d.py` (the last with every thread variable at 4) | `logs/b3_build_*.log` |
| B3 pre-screens | `b3_uni.py a b`, `b3_uni.py c a`, `b3_uni.py d`; `b3_prescreen.py --mode joint --blocks a b c --all`; `b3_witness_info.py` (evaluation only) | `logs/b3_uni_*.log`, `logs/b3_prescreen_joint.log`, `logs/b3_witness_info.log` |
| dry runs (3,000 feeds, `CBX_THREADS=4`) | `b1_arm.py --tag dry_b1_mcc4 --keys clu --mcc 4 --max_feeds 3000 --iters 40 --out nn_x/w12/cb/famB/preds/dry`; `b3_arm.py --tag dry_b3_abc --blocks a b c --max_feeds 3000 --iters 40 --out ...`; `b1_arm.py --tag dry_b2_item --keys item --mcc 1 --set xm+xs --msr 0 --leaf_limit 2000000 --max_feeds 3000 --iters 40 --out ...`; `b3_arm.py --tag dry_b13 --blocks a b c --native_clu --max_feeds 3000 --iters 40 --out ...` | terminal |
| B1 | `b1_arm.py --tag b1_mcc4_s0 --keys clu --mcc 4 --seed 0 --budget_min 105`; `b1_arm.py --tag b1_mcc2_s0 --keys clu --mcc 2 --seed 0 --budget_min 45` | `logs/b1_mcc4_s0.log`, `logs/b1_mcc2_s0.log` |
| B2 | `b1_arm.py --tag b2_item_s0 --keys item --mcc 1 --set xm+xs --msr 0 --leaf_limit 2000000 --seed 0 --budget_min 70` | `logs/b2_item_s0.log` |
| B3 | `b3_arm.py --tag b3_abc_s0 --blocks a b c --seed 0 --budget_min 110` | `logs/b3_abc_s0.log` |
| seed replicates (information) | `b0_arm.py --seed 1 --budget_min 62`; `b3_arm.py --tag b3_abc_s1 --blocks a b c --seed 1 --budget_min 80` | `logs/b0_s1.log`, `logs/b3_abc_s1.log` |
| lead's notice (freshness) | `read_m50.py <tags and bundles>`; `col_dist.py`, `col_dist.py lag`; with `FB_LAG=1` in the environment: `b3_build_a.py`, `b3_build_b.py`, `b3_build_c.py`; `lag_read.py`; evaluation-only reads `b3_witness_split.py`, `b3_witness_density.py`; `pair_nats.py A B ...`; `b3_trunc.py 200 300 400 495 325`; `fill_report.py` | `logs/read_m50.log`, `logs/col_dist*.log`, `logs/b3_build_lag.log`, `logs/lag_read.log`, `logs/b3_witness_*.log`, `logs/pair_*.log`, `logs/b3_trunc.log` |
| reads | `b0_trunc.py 200 300 400 510`; `read_arms.py b0 b1_mcc4_s0 b3_abc_s0 --bundle b0+b3_abc_s0 b0+b1_mcc4_s0+b3_abc_s0 b1_mcc4_s0+b3_abc_s0`; `mk_table.py ...`; the shared tool: `python nn_x/w12/cb/nats.py <tag> <X_val.npy> <X_hold.npy> [more pairs = bundle]` | `logs/b0_trunc.log`, `logs/read_*.log` |

Full `nats.py` lines of every arm and bundle:

```
b0: level 0.38150 / 0.30462 (W16 0.40144 / 0.32460) | rho 0.847 / 0.847 | NATS +0.00117+-0.00038 / +0.00051+-0.00017 (beta_X z +6.1 / +6.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00006+-0.00039, holdout -0.00008+-0.00022, pooled z -0.36 -> no pass
b0_s1: level 0.38132 / 0.30402 (W16 0.40144 / 0.32460) | rho 0.843 / 0.844 | NATS +0.00096+-0.00035 / +0.00051+-0.00017 (beta_X z +5.5 / +6.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00060+-0.00039, holdout +0.00010+-0.00021, pooled z -0.75 -> no pass
b3_abc_s0: level 0.38362 / 0.30480 (W16 0.40144 / 0.32460) | rho 0.845 / 0.846 | NATS +0.00147+-0.00043 / +0.00080+-0.00021 (beta_X z +6.8 / +7.6) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00013+-0.00040, holdout +0.00017+-0.00022, pooled z +0.30 -> no pass
b3_abc_s1: level 0.38350 / 0.30510 (W16 0.40144 / 0.32460) | rho 0.841 / 0.842 | NATS +0.00144+-0.00042 / +0.00081+-0.00021 (beta_X z +6.8 / +7.7) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val +0.00023+-0.00040, holdout +0.00002+-0.00022, pooled z +0.47 -> no pass
b1_mcc4_s0: level 0.38148 / 0.30480 (W16 0.40144 / 0.32460) | rho 0.851 / 0.850 | NATS +0.00096+-0.00034 / +0.00054+-0.00017 (beta_X z +5.6 / +6.2) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00039+-0.00040, holdout +0.00004+-0.00022, pooled z -0.57 -> no pass
b1_mcc2_s0: level 0.38051 / 0.30415 (W16 0.40144 / 0.32460) | rho 0.845 / 0.845 | NATS +0.00088+-0.00033 / +0.00052+-0.00017 (beta_X z +5.3 / +6.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00025+-0.00040, holdout +0.00009+-0.00021, pooled z -0.13 -> no pass
b2_item_s0: level 0.37947 / 0.30159 (W16 0.40144 / 0.32460) | rho 0.834 / 0.831 | NATS +0.00112+-0.00037 / +0.00051+-0.00017 (beta_X z +6.0 / +6.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00087+-0.00039, holdout +0.00024+-0.00021, pooled z -0.75 -> no pass
BUNDLE(b0+b3_abc_s0): level 0.38361 / 0.30589 (W16 0.40144 / 0.32460) | rho 0.851 / 0.851 | NATS +0.00139+-0.00042 / +0.00068+-0.00019 (beta_X z +6.6 / +7.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00008+-0.00039, holdout +0.00015+-0.00021, pooled z +0.35 -> no pass
BUNDLE(b0+b0_s1+b3_abc_s0+b3_abc_s1): level 0.38432 / 0.30593 (W16 0.40144 / 0.32460) | rho 0.850 / 0.851 | NATS +0.00134+-0.00041 / +0.00069+-0.00019 (beta_X z +6.5 / +7.1) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val +0.00023+-0.00039, holdout +0.00004+-0.00021, pooled z +0.56 -> no pass
BUNDLE(b3_abc_s0+b3_abc_s1): level 0.38476 / 0.30575 (W16 0.40144 / 0.32460) | rho 0.846 / 0.847 | NATS +0.00151+-0.00043 / +0.00083+-0.00021 (beta_X z +6.9 / +7.8) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val +0.00023+-0.00039, holdout +0.00007+-0.00021, pooled z +0.65 -> no pass
BUNDLE(b0+b0_s1): level 0.38227 / 0.30483 (W16 0.40144 / 0.32460) | rho 0.848 / 0.849 | NATS +0.00110+-0.00037 / +0.00053+-0.00017 (beta_X z +5.9 / +6.2) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00030+-0.00039, holdout +0.00003+-0.00021, pooled z -0.45 -> no pass
BUNDLE(b0+b1_mcc4_s0): level 0.38219 / 0.30566 (W16 0.40144 / 0.32460) | rho 0.852 / 0.852 | NATS +0.00110+-0.00037 / +0.00054+-0.00017 (beta_X z +5.9 / +6.3) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00020+-0.00040, holdout +0.00007+-0.00021, pooled z -0.11 -> no pass
BUNDLE(b0+b1_mcc4_s0+b3_abc_s0): level 0.38426 / 0.30628 (W16 0.40144 / 0.32460) | rho 0.854 / 0.854 | NATS +0.00127+-0.00040 / +0.00065+-0.00019 (beta_X z +6.4 / +6.9) -> not promising (bar 0.0016 on both) | member rule 0.9/0.1: val -0.00015+-0.00039, holdout +0.00007+-0.00021, pooled z -0.04 -> no pass
```

## 10. Files

All under `<repo>/nn_x/w12/cb/famB/`:

- `PROGRESS.md`: time-stamped log, including every decision written before its read.
- Code: `fb_common.py` (ShufDS, TimeCat, B0 recipe, CTR255), `b0_arm.py`, `b1_arm.py` (B1 / B2), `b3_arm.py` (B3), `b3_build_{a,b,c,d}.py`, `b3_uni.py`, `b3_prescreen.py`, `b3_witness_info.py`, `b3_witness_split.py`, `b3_witness_density.py`, `lag_read.py`, `read_m50.py`, `col_dist.py`, `pair_nats.py`, `b3_trunc.py`, `fill_report.py`, `b1_noise.py`, `b1_noise2.py`, `b1_noise3.py`, `b1_noise_full.py`, `b0_trunc.py`, `read_arms.py`, `mk_table.py`.
- `feat/{a,b,c,d}_{xm,hold,val,test}.npy` + `feat/{a,b,c,d}_names.json` (float32 column blocks, row order of the cbx sets), `feat/lag_{a,b,c}_{xm,hold}.npy` (test-lag variant of the train rows), `feat/pos_prior.npy`. About 16 GB.
- `preds/<tag>_{val,hold,test}.npy`, `preds/<tag>.eval.json`, `preds/<tag>.curve.json`, `preds/<tag>.cbm` for `b3_abc_s0`, `b3_abc_s1`, `b1_mcc4_s0`, `b1_mcc2_s0`, `b2_item_s0`, `b0_s1`; `preds/b0trunc_<N>_{val,hold}.npy`; `preds/dry/` (dry runs).
- `out/`: `final_table.md`, `final_table.json`, `final_nats_lines.txt`, `ctr_border_check.txt`, `b1_noise*.json`, `b3_uni_*.json`, `b3_prescreen_joint_a_b_c.json`, `b3_witness_{info,split,density}.json`, `lag_read.json`, `read_m50.json`, `col_dist*.md`, `block_a_recency.txt`, `read_*.json`.
- `logs/`: one log per job.
- Reference (read-only, not mine): `nn_x/w12/cb/infra/out/ts_lr020_s0_{val,hold,test}.npy` (B0).
