# Adversarial review of family B (KEY = famB_review): `reports/CATBOOST_1115/famB.md`

Written 2026-11-15 06:40Z. CPU only, at most 12 threads per process and two heavy processes, no GPU, no process signalled, nothing written outside `nn_x/w12/cb/famB_review/` and this file.
Marks: **[M]** measured here with my own code, **[D]** read in a file of the study or the project, **[I]** inferred. "a ± b" = mean ± 1 SE, paired per feed; val = 14,423 feeds, holdout = 30,000 feeds (every feed has a click).

## 0. Verdict

1. **The study's verdict is CONFIRMED: nothing in family B is PROMISING, nothing passes the member rule, family B does not keep the tree-member track open.** I could not break it.
2. **Every number the verdict rests on was re-derived** from the raw labels, my own holdout definition, the 16 W16 member files and the saved predictions, with code that imports nothing of the study: all 13 arm / bundle rows agree to the printed digit (largest difference: level 0, NATS 7e-11, SE 1.5e-8, member delta 0, pooled z 4e-5) [M].
3. **No leak found.** Key columns of all three B3 blocks were rebuilt from raw data for 5,000 train rows and 4,996 val rows by my own code and match (19 columns; counts exact, ratios to 1e-6, transition lifts corr 0.99999). No val / test / holdout label enters any input, statistic, tree count or arm choice [M / D].
4. **The screen and the member rule were applied as written** (≥ 0.0016 on BOTH witnesses; weight 0.1 against the W16 train-only pool). Best arm B3: +0.00147 ± 0.00043 val / +0.00080 ± 0.00021 holdout; the holdout is 3.8 SE below the bar. Best pooled z +0.65, the rule needs +2 [M].
5. **Two sub-claims are weaker than written** (they do not touch the verdict): (a) "B3's gain comes from two label-free columns" is not established: in a tree the density pair alone reads +0.00010 ± 0.00010 on the holdout against +0.00030 ± 0.00012 for all 70 columns; (b) the suggestion to carry the density pair into the neural students rests on the holdout only (val +0.00018 ± 0.00015 [D]), which the lead's notice names as the pattern of a freshness artefact.
6. **The lag-matched B3 refit that was not run cannot change the verdict** [M, reduced scale]: training on test-lag statistics reads +0.00024 ± 0.00028 on val and -0.00023 ± 0.00012 on the holdout against fresh training. The holdout stays at half the bar.

## 1. Claim table

| claim | study's number | my number | agree |
|---|---|---|---|
| W16 train-only pool, level val / holdout | 0.40144 / 0.32460 | 0.40144 / 0.32460 (rebuilt from the 16 member files; max difference to the infra z cache 6e-7) | yes |
| `b0`: level; NATS; member rule 0.9 / 0.1, pooled z (val / holdout) | 0.38150 / 0.30462; +0.00117 ± 0.00038 / +0.00051 ± 0.00017; -0.00006 ± 0.00039 / -0.00008 ± 0.00022, z -0.36 | identical | yes |
| `b0_s1` | 0.38132 / 0.30402; +0.00096 ± 0.00035 / +0.00051 ± 0.00017; -0.00060 ± 0.00039 / +0.00010 ± 0.00021, z -0.75 | identical | yes |
| `b3_abc_s0` (best arm) | 0.38362 / 0.30480; +0.00147 ± 0.00043 / +0.00080 ± 0.00021; -0.00013 ± 0.00040 / +0.00017 ± 0.00022, z +0.30 | identical | yes, not PROMISING, no pass |
| `b3_abc_s1` | 0.38350 / 0.30510; +0.00144 ± 0.00042 / +0.00081 ± 0.00021; +0.00023 ± 0.00040 / +0.00002 ± 0.00022, z +0.47 | identical | yes |
| `b1_mcc4_s0` | 0.38148 / 0.30480; +0.00096 ± 0.00034 / +0.00054 ± 0.00017; -0.00039 ± 0.00040 / +0.00004 ± 0.00022, z -0.57 | identical | yes |
| `b1_mcc2_s0` | 0.38051 / 0.30415; +0.00088 ± 0.00033 / +0.00052 ± 0.00017; -0.00025 ± 0.00040 / +0.00009 ± 0.00021, z -0.13 | identical | yes |
| `b2_item_s0` | 0.37947 / 0.30159; +0.00112 ± 0.00037 / +0.00051 ± 0.00017; -0.00087 ± 0.00039 / +0.00024 ± 0.00021, z -0.75 | identical | yes |
| `bundle(b3_abc_s0 + b3_abc_s1)` (best member-rule read) | 0.38476 / 0.30575; +0.00151 ± 0.00043 / +0.00083 ± 0.00021; +0.00023 ± 0.00039 / +0.00007 ± 0.00021, z +0.65 | identical | yes, no pass |
| `bundle(b0 + b0_s1 + b3_abc_s0 + b3_abc_s1)` (declared form at 2 + 2 seeds) | 0.38432 / 0.30593; +0.00134 ± 0.00041 / +0.00069 ± 0.00019; +0.00023 ± 0.00039 / +0.00004 ± 0.00021, z +0.56 | identical | yes |
| the other four bundles (`b0+b3_abc_s0`, `b0+b0_s1`, `b0+b1_mcc4_s0`, `b0+b1_mcc4_s0+b3_abc_s0`) | rows of the study's table | identical (`out/rv_compare_short.md`) | yes |
| val M ≥ 50 NATS (8,530 feeds), 9 reads | b0 +0.00090 ± 0.00041, b3_abc_s0 +0.00110 ± 0.00046, b3_abc_s1 +0.00119 ± 0.00048, B3 x2 +0.00118 ± 0.00048, ... | identical for all 9 | yes |
| B3 minus B0, seed 0: NATS; level (val / holdout) | +0.00030 ± 0.00019 / +0.00028 ± 0.00009; +0.00212 ± 0.00088 / +0.00019 ± 0.00050 | identical | yes |
| B3 x2 minus B0 x2: NATS; level | +0.00041 ± 0.00016 / +0.00030 ± 0.00008; +0.0025 ± 0.0008 / +0.0009 ± 0.0004 | +0.00041 ± 0.00016 / +0.00030 ± 0.00008; +0.00250 ± 0.00080 / +0.00093 ± 0.00042 | yes |
| B1 (complexity 4) minus B0, NATS | -0.00021 ± 0.00014 / +0.00002 ± 0.00006 | identical | yes: null |
| B2 minus B0, level | -0.00203 ± 0.00097 / -0.00302 ± 0.00058 | identical | yes |
| B3 minus B0 on feeds whose user has an earlier train feed (30.3% / 18.0% of feeds), seed 0 | +0.00078 ± 0.00035 / +0.00047 ± 0.00021 | identical; at two seeds each +0.00084 ± 0.00030 / +0.00048 ± 0.00019 (other feeds +0.00022 ± 0.00019 / +0.00026 ± 0.00009) | yes |
| fit feeds, selection feeds, trees used (eval json and logs) | 120,591 / 13,527; 495 of 520 (B3 s0), 325 of 425 (B3 s1), 516 (B1), 290 (B1 complexity 2), 294 of 394 (B2, 436,320 fit feeds), 352 (B0 s1) | same in `preds/<tag>.eval.json` [D] | yes |
| block a support: feeds with a user source, fresh; under the test lag | xm 19.0%, holdout 18.0%, val 30.3%; lag 5.4% | holdout 18.0%, val 30.3%; lag 5.36% | yes |
| block a recency (hours to the newest source), median | 72 xm / 240 val / 357 test | 72 / 241 / 358 | yes |
| density columns have the same distribution on train, holdout, val, test | p10 / p50 / p90 equal | equal; share of rows with `kn_cos1` ≥ 0.99: 0.188 / 0.182 / 0.187 / 0.186 | yes |
| default native CTR (15 borders) is blind near a 1% click rate | 2 predicted levels, corr 0.057; with 255 borders corr 0.947 | 2 levels, corr 0.017; with 255 borders corr 0.951 (own synthetic run) | yes |
| "the click statistics are empty" (linear read: +0.00004 ± 0.00007 / +0.00006 ± 0.00006) | linear composite | in a tree at reduced scale: click columns alone +0.00025 ± 0.00026 val / +0.00004 ± 0.00009 holdout | yes on the holdout; not resolvable on val |
| "B3's gain comes from two label-free columns and exposure counts" | linear composite, importance | density pair alone +0.00023 ± 0.00025 / +0.00010 ± 0.00010; all click-free columns +0.00032 ± 0.00026 / +0.00017 ± 0.00011; all 70 +0.00018 ± 0.00026 / +0.00030 ± 0.00012 | partly: the click-free family yes, the two columns alone not shown |
| a user-level native key leaks on CPU (learn NDCG 0.56 against 0.27 on noise labels); candidate-level keys +0.0002 over chance at full scale | `logs/b1_noise3.log`, `logs/b1_noise_full.log` | read in the logs: 0.558 against 0.266; 0.22763 against 0.22741 [D]; not re-run | yes [D] |

## 2. Issues by severity

No issue changes the verdict. None is a leak.

### Medium (they change how two sub-claims should be read)

1. **Verdict item 4 and report item 8 over-state the attribution** [M].
   - The split "click statistics empty, gain from two label-free columns" comes from a linear composite of per-feed z-scores. I tested it in tree form at reduced scale (section 4): the click statistics alone add nothing on the holdout (+0.00004 ± 0.00009), which supports the first half. The density pair alone adds +0.00010 ± 0.00010 on the holdout and +0.00023 ± 0.00025 on val, a third of what all 70 columns add on the holdout (+0.00030 ± 0.00012). The gain belongs to the click-free family as a whole (density, impression counts, neighbour support), not to two columns.
   - On val the gain of B3 over B0 sits on the feeds whose user has earlier train feeds: +0.00084 ± 0.00030 there against +0.00022 ± 0.00019 elsewhere (two seeds each). The linear read gives the user block +0.00001 on val. The linear read does not describe what the tree uses on val.
   - Report item 8 proposes the density pair as an input of the neural students at "+0.0002 to +0.0003 nats". That size is a holdout reading (+0.00032 ± 0.00013 [D]); on val the study reads +0.00018 ± 0.00015 and +0.00007 ± 0.00009 next to B3 [D]. The index behind the pair is the set of much-shown items of all 15 train days: contemporaneous for train and holdout rows, 2 to 11 days old for val and test. Equal marginal distributions do not show that its relation to the click is equal. By the lead's notice this is a freshness suspect, not a find.
2. **The lead's freshness notice was not carried into a tree, and the mismatch is real** [M].
   - Train rows carry fresh statistics, val / test rows stale ones. Measured on 400,000 rows per set: `ua_item_dt` median shifted by +1.05 (val) and +1.91 (test) train ranges, `ua_dt_last` by +0.80 and +1.36, `kn_ex20` by +0.36.
   - In the study's `out/col_dist.md` but not in its report text: the four recency-weighted columns (`ua_cat_imprw`, `ua_cat_posrw`, `ua_c1024_imprw`, `ua_c1024_posrw`) keep only 0.06 to 0.18 of their train spread on val and 0.01 to 0.03 on test. They are close to constant at apply time.
   - The direction is against B3 on val and test and neutral on the holdout, which is the binding witness. My reduced-scale read of lag-matched training (section 4): +0.00024 ± 0.00028 on val, -0.00023 ± 0.00012 on the holdout. It cannot double the holdout NATS.
3. **"Label statistics for a tree are exhausted at about +0.0003 nats" is wider than the evidence** [I]. One design was measured: one arm, two seeds, fresh training, a neighbour index over 15% of the train items. The planned variants that were not run are listed in the study's section 8. The measured statement is "this design adds +0.0003 to +0.0004 nats to B0".

### Minor

4. Two seeds instead of three, and both seed-1 runs are shorter (352 and 325 trees against 510 and 495). The study compares them at equal tree counts. B4 did not trigger, so no rule depends on it.
5. B3 was run although the study's own pre-screen rule dropped all three blocks. The deviation works against a negative verdict, so it is harmless here. It also shows the pre-screen could not see an effect of 0.0003.
6. The B1 noise-label criterion declared before the run was replaced after the first result (declared confounded). The replacement shows a within-feed prefix channel for candidate-level keys at 2,000 feeds (+0.015 learn NDCG) and none at full scale, tested for keys only, 30 trees, complexity 4. A leak of that kind can only lower a val / holdout read, so the B1 null stands.
7. The on-disk code is not byte-identical to the code that produced `b3_abc_s0`: `b3_arm.py` was edited at 02:16Z (B13 option) and the builders at 05:11Z (lag option), after the runs. The stored columns (file times 01:27Z to 01:37Z) equal what the present non-lag code path defines (my rebuild), and the eval json records the arguments and the 70 columns.
8. The time stamps inside the study's `PROGRESS.md` cannot be checked against file metadata (one modification time). The log times are consistent with the declared order: pre-screen 01:44Z, B3 launch 01:46Z, first witness read 01:58Z to 02:13Z, B3 read 03:28Z.
9. `nats.py` prints rho as one Pearson over all rows (0.847 on val for B0); the mean within-feed Pearson is 0.843. Cosmetic.
10. Block b tables are accumulated in float32: `tr_cat_l20m` differs from my float64 rebuild by up to 0.006 on 91 of 4,687 rows (corr 0.999993). Harmless.
11. The lag emulation confounds staleness with support: under it 59% of the train rows have no block b / c statistic and 5.4% of the feeds a user source, while val / test rows have all of them. A lagged holdout is therefore not an exact twin of test.
12. Pre-existing, not of this study [D]: the `tte_*` builder of the 177 features (`src/features_tte.py`) has no holdout handling, so holdout feeds are sources of base-feature statistics for later train rows. The window is by day, so no row reads its own label. It is common to B0, every arm and every model of record.
13. Four numba cache files appeared in `nn_x/w12/cb/infra/__pycache__/` at 01:16Z and 02:31Z. They cannot be attributed (famB has its own cache directory; three agents were importing `cbx`). No source or data file under `infra/`, `src/`, `data/` or the reference predictions changed.

## 3. Audits

### 3.1 Leak audit: which rows and labels feed each label-aware input

| input | a TRAIN row reads | a val / test row reads | own feed excluded | same user excluded | time-aware | holdout feeds excluded | val / test labels |
|---|---|---|---|---|---|---|---|
| block a, 39 columns | the same user's non-holdout train feeds with a strictly earlier timestamp, at most the 4,000 newest rows | all non-holdout train feeds of the user | yes (rebuilt) | no, by task item B3 (a) | yes | yes | none |
| block b, 25 columns | non-holdout train feeds of the 4 other user folds on earlier days | the 4 other folds, all 15 days | yes | yes (fold) | yes (same day excluded) | yes | none |
| block c, 4 click columns | clicks of the 20 nearest index items in the 4 other folds on earlier days | the 4 other folds, all days | yes; own item skipped | yes (fold) | yes | yes | none |
| block c, `kn_cos1` / `kn_cos20` and the index membership | impression counts of all 15 days, including later days and the row's own feed | the same index | no label involved | no label involved | no (label-free) | yes | none |
| position prior, 512 numbers | all non-holdout train rows of all days | same | no (1 row in 244M) | no | no | yes | none |
| B1 native CTRs (category, three cluster keys) | CatBoost ordered statistics on the 120,591 fit feeds: earlier feeds and a random prefix of the own feed | the whole fit pool | no: within-feed prefix, +0.0002 over chance at full scale [D] | no | yes | yes | none |
| B2 native CTR (item_id) | fit pool xm + xs (436,320 feeds), earlier rows | the whole fit pool | yes in effect: no item repeats inside a feed (0 of 30M train rows, 0 of all val rows) [M] | no | yes | yes | none |
| tree count of every arm | NDCG on the 13,527 train-side selection feeds (other users than the fit feeds) | | | | | | none |
| the 177 base features `tte_*` (pre-existing) | train days before the row's day, holdout feeds included | all of train | yes (by day) | no | yes | NO | none |

Rebuild from raw data (`rv_rebuild.py`; 2,500 random rows plus 2,500 rows of user-covered feeds per set; no import of the study's code) [M]:

| column(s) | xm rows: result | val rows: result |
|---|---|---|
| `kn_cos1`, `kn_cos20` (own index: 5,862,592 items, 67.6% of impressions, as the study) | 5,000 rows, max difference 9e-8 | 4,996 rows, max difference 9e-8 |
| `kn_ex20`, `kn_coec20` | 4,752 rows with a value, 1 row differs (tie among neighbours); NaN pattern identical | 4,996 rows, 2 rows differ |
| `ua_ns`, `ua_npos`, `ua_cat_{imp,pos,coec}`, `ua_c1024_{imp,pos,coec}`, `ua_item_{imp,pos}`, `ue_k20_coec`, `ua_dt_last` | 2,998 covered rows exact (ratios to 5e-7), 2,002 NaN rows identical | 3,278 covered rows exact, 1,718 NaN rows identical |
| `tr_cat_base`, `tr_cat_l1`, `tr_cat_l20m` | max difference 1.5e-4, 1.6e-4, 6e-3 (corr 0.999993); NaN pattern identical | 1.3e-4, 1.5e-4, 1.4e-3 |
| position prior | max difference 7e-9 | |
| lag build of block a (`lag_a_xm.npy`), 3,000 feeds | source selection exact; youngest source 5.1 days old (day granularity, not 6.0) | |

Not rebuilt: block d (in no arm), the cluster-level transition columns, the three `ux_*` columns, the other 27 columns of block a, any holdout or test row.
Val and train share the item id space (same id: median cosine 1.000 between the two embedding tables), and `kn_cos1` shows no own-item artefact on val (share ≥ 0.9999: 0.15% val, 0.19% xm) [M].

### 3.2 Train / apply mismatch of the new columns (`rv_dist.py`, 400,000 rows per set) [M]

| column | xm: NaN share, p10 / p50 / p90 | val: NaN change, median shift, spread ratio | test: NaN change, median shift, spread ratio |
|---|---|---|---|
| `kn_cos1`, `kn_cos20` | 0.000 | no shift | no shift |
| `ua_item_dt` | 0.997, 1.8 / 37 / 170 | +0.001, +1.05, 1.79 | +0.002, +1.91, 1.72 |
| `ua_dt_last` | 0.805, 6.5 / 72 / 218 | -0.114, +0.80, 1.33 | -0.096, +1.36, 1.36 |
| `ua_cat_imprw` | 0.805, 0 / 2.4 / 31 | -0.114, -0.07, 0.16 | -0.096, -0.08, 0.03 |
| `ua_c1024_imprw` | 0.805, 0 / 0.32 / 9.8 | -0.114, -0.03, 0.13 | -0.096, -0.03, 0.03 |
| `ua_cat_posrw` | 0.805, 0 / 0 / 0.94 | -0.114, 0.00, 0.18 | -0.096, 0.00, 0.03 |
| `kn_ex20` | 0.086, 1.2 / 5.0 / 13.9 | -0.086, +0.36, 1.32 | -0.086, +0.36, 1.31 |
| `ua_c1024_posrw` | 0.805, 0 / 0 / 0.27 | -0.114, 0.00, 0.06 | -0.096, 0.00, 0.01 |
| ratio columns (COEC, lifts, base rates; 34 columns) | | shift below 0.06, spread 0.72 to 1.16 | same |

Shift = change of the median in units of the xm p10-to-p90 range; spread ratio = p10-to-p90 range against xm. The holdout equals xm on every column (shift at most 0.02, NaN share within 0.01).

### 3.3 Selection audit [D / M]

- Tree counts: `cbx.run_arm`, eval set = the 13,527 train-side selection feeds; no val label is read before the final evaluation.
- Columns: B3's 70 columns were fixed at the launch (01:46Z), before the first read of a witness. No block subset was run afterwards. Block d was built after the first witness read and entered no arm.
- Arms: five configurations (B0, B1 x2, B2, B3) and two seed replicates, each read once on both witnesses. The reported "best" figures are maxima over these rows; for a negative verdict that is conservative.
- Screen: applied as ≥ 0.0016 on both witnesses. No arm reaches it on either witness; the best val read is the B3 two-seed bundle, +0.00151 ± 0.00043.
- Member rule: weight 0.1, both deltas ≥ 0 and pooled z ≥ 2, against W16 rebuilt here from `nn_w5_e10_kd05w192_s951-958` and `nn_w5_e10_kd05t25w192_s951-958` train-only files (unchanged since 11-14).

### 3.4 Read-only files and GPU [M]

- `nvidia-smi --query-compute-apps` (look only): no command line of a listed GPU process belongs to the CatBoost track; the readable ones are the lead's `train.py` / `train_full.py` lanes, and 30 listed PIDs have no readable command line from this shell. No famB process is alive. The stored parameters of every famB model contain no `task_type` (CPU default) [D].
- No file under `src/`, `data/flat`, `data/feats`, `data/clu_work`, `nn_x/w12/cb/audit`, `nn_x/w12/cb/infra/out`, `nn_x/w12/cb/infra/cache` or `nats.py` is newer than the study's start, apart from issue 13. New files under `nn_x/w5/` and `preds/` are the lead's training lanes; none is a tree prediction, and `subs/` has no new file.

## 4. Reduced-scale tree diagnostic (information; no arm, no candidate)

Method [M]: LightGBM lambdarank (63 leaves, learning rate 0.1, 300 rounds, no subsampling, hard click labels) on 30,000 random xm fit feeds, three replicates on different feed samples, the three models of an arm bundled by the member-equal mean of per-feed z. Same rows for every arm inside a replicate. No selection: fixed 300 rounds. Read with my NATS code over the W16 pool. R = the 177 features.

| arm (3 replicates bundled) | level val / holdout | NATS val / holdout | NATS minus R, val | NATS minus R, holdout |
|---|---|---|---|---|
| R: 177 features | 0.36659 / 0.29373 | +0.00097 ± 0.00034 / +0.00038 ± 0.00014 | | |
| A: R + all 70 columns (the B3 inputs), fresh | 0.37035 / 0.29620 | +0.00115 ± 0.00038 / +0.00068 ± 0.00019 | +0.00018 ± 0.00026 | +0.00030 ± 0.00012 |
| F: R + the 18 click-free columns | 0.36931 / 0.29535 | +0.00129 ± 0.00040 / +0.00055 ± 0.00017 | +0.00032 ± 0.00026 | +0.00017 ± 0.00011 |
| K: R + the 52 click-statistic columns | 0.36803 / 0.29464 | +0.00122 ± 0.00039 / +0.00042 ± 0.00015 | +0.00025 ± 0.00026 | +0.00004 ± 0.00009 |
| D: R + `kn_cos1` + `kn_cos20` | 0.37112 / 0.29563 | +0.00120 ± 0.00038 / +0.00048 ± 0.00016 | +0.00023 ± 0.00025 | +0.00010 ± 0.00010 |
| L: R + all 70, train rows built with the test lag | 0.36856 / 0.29327 | +0.00139 ± 0.00041 / +0.00045 ± 0.00016 | +0.00042 ± 0.00028 | +0.00007 ± 0.00010 |

Direct pairs: A minus K +0.00026 ± 0.00012 on the holdout (val -0.00007 ± 0.00027); A minus F +0.00013 ± 0.00012 (val -0.00015 ± 0.00026); **L minus A +0.00024 ± 0.00028 on val, -0.00023 ± 0.00012 on the holdout**. With the holdout columns also rebuilt under the lag (two replicates): L minus A -0.00022 ± 0.00013, L minus R +0.00001 ± 0.00011.

Readings:
- The diagnostic reproduces the size of the full-scale effect on the holdout (A minus R +0.00030 ± 0.00012; the study's B3 minus B0 +0.00030 ± 0.00008), so its splits are meaningful there. On val nothing is resolved (every contrast within 1.5 SE of zero).
- Click statistics alone: nothing on the holdout. Click-free columns: about half to all of the gain. The density pair alone: a third, not significant.
- Lag-matched training shows no sign of a larger holdout effect; when training and holdout are both lagged the gain over the 177 features is zero (+0.00001 ± 0.00011). The +0.0003 of B3 on the in-period holdout therefore looks like a gain that needs fresh statistics [I].

Limits: a weaker model than the study's arms (level 0.367 against 0.3815), LightGBM instead of YetiRank, 30,000 fit feeds; single-replicate contrasts range from -0.0003 to +0.0007 nats on val, so only the bundled reads are quoted; arm L on the lagged holdout has two replicates.

## 5. What the lead should conclude

1. Close family B as the study says. Its table can be quoted as it stands; I re-derived every row.
2. Do not spend the 100 minutes on a lag-matched B3 refit. The holdout is the binding witness at half the bar, and lag-matching moves it down, not up.
3. Do not take the density pair forward as a measured find. If it is tried as an input of the students, it needs the pooled-z rule on val, with the index built at the test lag.
4. Read the study's item 4 as "the click statistics add nothing on the in-period holdout; the small gain of B3 comes from click-free exposure and density columns and looks freshness-dependent". Do not read it as "label statistics are exhausted".
5. Keep the CatBoost fact about CTR borders (replicated here): with the default 15 borders a native categorical carries no usable target statistic at a 1% click rate, including `category_id` in B0 and the CatBoost of record.
6. The track decision still needs families S and N; this review covers family B only.

## 6. What I did NOT run

- No full-scale CatBoost fit: no lag-matched B3, no ablation of B3 by block at full scale, no third seed.
- No re-run of the B1 noise-label tests or of the B3 pre-screens (read in the logs).
- No rebuild of holdout or test rows, of block d, of the cluster-level transition columns, of the `ux_*` columns or of the remaining 27 block a columns.
- No check of how the W16 members or the cross-fitted teacher were trained.
- The study's self-reported process breaches (three processes for a minute, snippets without the CUDA variable) cannot be verified from files.

## 7. Commands and files

Every job: `cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 CBX_CACHE_EXTRA=<repo>/nn_x/w12/cb/famB_review/cache && setsid nohup .venv/bin/python -u nn_x/w12/cb/famB_review/<script> <arguments> > nn_x/w12/cb/famB_review/logs/<name>.log 2>&1 < /dev/null &`

All under `<repo>/nn_x/w12/cb/famB_review/`:

| script | what | arguments | output |
|---|---|---|---|
| `rv_eval.py` | own NDCG, conditional logit, member rule; all 13 rows, val M ≥ 50, 5 pairs | none | `logs/rv_eval.log`, `out/rv_eval.json` |
| `rv_compare.py` | side-by-side with the study's stored numbers | none | `out/rv_compare.md`, `out/rv_compare_short.md` |
| `rv_rebuild.py` | rebuild of 19 columns from raw data | `2500` | `logs/rv_rebuild.log`, `out/rv_rebuild.json` |
| `rv_dist.py` | column distributions on xm, lag-xm, holdout, val, test | none | `logs/rv_dist.log`, `out/rv_dist.json` |
| `rv_ctr.py` | synthetic CTR border check (4 threads) | none | `logs/rv_ctr.log` |
| `rv_diag.py`, `rv_diag2.py`, `rv_diag3.py` | reduced-scale diagnostic: replicate 11 (all arms), replicates 12 and 13 (R, A, F, K, L), arm D for 12 and 13 | `30000`; `30000 12`, `30000 13`; `30000 12`, `30000 13` | `logs/rv_diag*.log`, `out/diag_<arm>[_s<seed>]_{val,hold,holdlag}.npy` |
| `rv_diag_read.py` | bundled reads of the three replicates | none | `logs/rv_diag_read.log`, `out/rv_diag_read.json` |
| inline snippets | coverage split, lag-build check, item-id checks | | `logs/rv_cov_split.log`, `logs/rv_lagcheck.log` |

Cross-check of the shared tool after my code agreed: `python nn_x/w12/cb/nats.py b3_abc_s0_xcheck nn_x/w12/cb/famB/preds/b3_abc_s0_val.npy nn_x/w12/cb/famB/preds/b3_abc_s0_hold.npy` prints the study's line.
Progress log: `PROGRESS.md`.
