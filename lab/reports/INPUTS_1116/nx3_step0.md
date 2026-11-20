# nx3 STEP 0: build of record, gates, install, smoke

Builder's report for the trained-input test nx3 (ledger `nn_x/w5/logs/E_kd_predeclared.txt`, block of 2026-11-16T03:45:41Z, STEP 0 as amended by A1-A6). Written 2026-11-16 04:20 UTC. Work directory `nn_x/w12/nx3/`. Every number below names its file; paths are relative to `<repo>`.

**STEP 0 reads PASS by the rule as written.** The files are installed, both smoke runs passed, and stage A is ready but not launched.

Three things need the lead's eye before stage A:

1. **One comparison value sits outside a bar under a second reading of C4.** The ledger says "train (has = 1)"; BUILD_SPEC C4 says "train has = 1 rows (pool feeds)". I fixed the BUILD_SPEC reading as the deciding one in `nx3_gates.py` before any gate value existed, and it passes with margin (PSI at most 0.091). On all 115,150,397 has = 1 train rows, `nx_rp3_z` reads PSI 0.127 against val, 0.260 against public and 0.346 against private. If the lead rules that the all-rows reading binds, the verdict becomes "FAIL G-parity only -> NOT TESTABLE AT PARITY". Detail in section 3.3.
2. **The smoke ran before the STEP 0 read was written to the ledger.** The rule says no GPU process of the arm before STEP 0 "is read and written here"; my task ordered the smoke after a PASS and forbade me to write to the ledger. The gate file is dated 04:10:06Z and the smoke runs started at 04:11:30Z. The ledger text is in section 10.
3. **No stored smoke of the stage-A command exists**, so the smoke control reproduces the precedent on inputs and data, not on trained outputs. What I compared is in section 6.

---

## 1. What was built

### 1.1 Files

| File | Shape, dtype | Bytes | md5 |
|---|---|---|---|
| `data/feats/train/nx_rp3_z.npy` | (246,745,836,) float32 | 986,983,472 | `76a972ad5d24c5e775f6ea883866f10d` |
| `data/feats/train/nx_rp3_rk.npy` | (246,745,836,) float32 | 986,983,472 | `d341219dd0b058c6a95a331fb9b9ef88` |
| `data/feats/val/nx_rp3_z.npy` | (1,389,971,) float32 | 5,560,012 | `cac796b8154a497a1807a8c26997e3f6` |
| `data/feats/val/nx_rp3_rk.npy` | (1,389,971,) float32 | 5,560,012 | `5bc4ab00876d314d878a3105853fcf3f` |
| `data/feats/test/nx_rp3_z.npy` | (2,904,806,) float32 | 11,619,352 | `ea21458551afe139137db76e0895345b` |
| `data/feats/test/nx_rp3_rk.npy` | (2,904,806,) float32 | 11,619,352 | `8f141b543c77cda30b31d47ff66119f9` |
| `nn_x/w12/nx3/staleval/nx_rp3_z.npy` (stale-val twin) | (1,389,971,) float32 | 5,560,012 | `b1340e87ec87d74140674d472d0b1fac` |
| `nn_x/w12/nx3/staleval/nx_rp3_rk.npy` (stale-val twin) | (1,389,971,) float32 | 5,560,012 | `d865bf469bd5e4561f2f5eda8112d896` |

- Each of the six `data/feats` files has a hard link `nn/feat_cache/<split>/<name>.npy` (link count 2), created with exclusive create after checking that the names existed nowhere [`nn_x/w12/nx3/out/install.txt`].
- The twins are not linked into `nn/feat_cache`.
- Staged copies with the same md5 are in `nn_x/w12/nx3/out/files/<split>/`. Raw `rp3b05` and `p3a` (float64) are in `nn_x/w12/nx3/out/raw/`.
- K = 1 / median non-zero `rp3b05` of val rows = 1.932062e+05 (critic's advance build: 1.9545e+05). K for `p3a` = 4.190647e+04; `p3a` is not an input [`out/build.log`].

### 1.2 The has flag

`has:nx_rp3_z` needs no file. `nn_x/abl/data.py` l.186-187 builds it as the non-NaN indicator of `nx_rp3_z`, and the trainer's missing-file check (`nn_x/abl/train.py` l.344) strips the prefix, so only `nx_rp3_z.npy` must exist for the three splits. Both train files hold NaN on every row of a feed without the view and nowhere else. As the loader reads them, the flag is 1 on 0.4667 of train rows and on every val and test row [`out/install.txt`].

### 1.3 Construction as executed

Kernel of record: `rp3`, `Index`, `source_fidx`, `day_of` of `nn_x/insights/ccb_1113/scratch/tester3/c09_build.py` (md5 `0975dfd62bed53be8b75a268ebf0b800`), imported unchanged. My code is the driver (`nx3_common.py`) and the checks.

| Rows | Dense members (always admitted) | Sparse members (admitted iff timestamp < own timestamp) | Counts |
|---|---|---|---|
| Train, lag view | `source_fidx` of train feeds on slots < cut' | the 1/37 sample (`hz2_common.sparse_flag`, 85,196 feeds) on slots >= cut' | has = 1 on 1,470,065 feeds, 115,150,397 rows, 15,225 of the 30,000 holdout feeds; cut' 5 / 6 / 7 / 8 on 1,306,359 / 114,224 / 49,475 / 7 feeds |
| Val, test (natural) | all train (2,579,285 sequences, slots 0-14) | all val (14,423) and test (28,595) feeds | one index for both splits |
| Val, stale-val twin | `source_fidx` of train slots < slot - L | the 1/37 sample on slots >= cut, plus all val feeds | cuts 7-12 on 1,965 / 1,987 / 3,751 / 3,460 / 1,760 / 1,500 feeds |

- cut' = max(slot - L, 5) with L from `nn_x/w12/stale/hz2/feats_L/train/hz2_lag.npy` (md5 `d9f5989fa8e300b0eb0c27e11104854f`); twin lags from `hz2_common.assign_lags`.
- Timestamp admission uses the kernel unchanged: a sparse member carries its timestamp where c09 carried a day, and the query carries its own timestamp, so the kernel's own test skips anything not strictly earlier.
- Degree = posting count of the item in the dense part of the index, floor 1, one rule for every split.
- Every sequence of the row's own non-zero user is skipped by the kernel. Each user-0 feed is its own sequence; nothing links them.
- No label array is opened by the build.
- Forms: `nx_rp3_z` = within-feed z of log1p(K x rp3b05) over all candidates, 0 on a feed without a non-zero value, clipped to ±5; `nx_rp3_rk` = within-feed mid-rank percentile minus 0.5. Both use `feed_z` / `feed_rk` of `reports/INPUTS_1116/residual_rows/lens.py`.

All from `nn_x/w12/nx3/out/build.log`. Build time 591 s at 12 threads.

---

## 2. NaN and zero shares per split and per day

Source: `nn_x/w12/nx3/out/build.log`, last table. "Non-zero" = raw `rp3b05` > 0 among non-NaN rows. "Flat feeds" = share of feeds with the view that have no non-zero value (their rows read 0 in both files).

| Rows | Feeds | Rows | NaN share | Non-zero | `nx_rp3_z` = 0 exactly | Flat feeds | `nx_rp3_z` mean / sd |
|---|---|---|---|---|---|---|---|
| Train Sep 6-11 (slots 0-5), each day | 247,429-291,760 | 19.2M-23.8M | 1.0000 | - | - | - | - |
| Train Sep 12 (slot 6) | 203,548 | 16,000,942 | 0 | 0.2611 | 0.0221 | 0.0575 | -0.0056 / 0.9517 |
| Train Sep 13 (slot 7) | 263,115 | 21,017,427 | 0 | 0.2487 | 0.0240 | 0.0632 | -0.0063 / 0.9464 |
| Train Sep 14 (slot 8) | 244,596 | 18,996,470 | 0 | 0.2399 | 0.0253 | 0.0691 | -0.0067 / 0.9426 |
| Train Sep 15 (slot 9) | 42 | 3,091 | 0 | 0.2388 | 0.0676 | 0.1667 | -0.0067 / 0.9192 |
| Train Sep 17 (slot 11) | 269,029 | 20,653,225 | 0 | 0.2157 | 0.0286 | 0.0807 | -0.0085 / 0.9273 |
| Train Sep 18 (slot 12) | 256,678 | 19,952,992 | 0 | 0.2167 | 0.0286 | 0.0781 | -0.0087 / 0.9258 |
| Train Sep 19 (slot 13) | 233,023 | 18,522,726 | 0 | 0.2226 | 0.0277 | 0.0743 | -0.0087 / 0.9260 |
| Train Sep 20 (slot 14) | 34 | 3,524 | 0 | 0.2701 | 0.0230 | 0.1176 | -0.0050 / 0.9581 |
| **Train, all** | 3,151,208 | 246,745,836 | **0.5333** | 0.2333 | 0.0262 | 0.0710 | -0.0075 / 0.9363 |
| Holdout feeds, Sep 12 / 13 / 14 | 1,801 / 2,499 / 2,646 | 276,857 / 386,615 / 413,923 | 0 | 0.1669 / 0.1639 / 0.1546 | 0.0386 / 0.0387 / 0.0426 | 0.0489 / 0.0460 / 0.0499 | sd 0.887 / 0.890 / 0.878 |
| Holdout feeds, Sep 17 / 18 / 19 | 2,903 / 2,718 / 2,658 | 447,286 / 423,882 / 417,942 | 0 | 0.1388 / 0.1376 / 0.1382 | 0.0341 / 0.0415 / 0.0411 | 0.0410 / 0.0482 / 0.0497 | sd 0.859 / 0.849 / 0.845 |
| Holdout feeds, all 30,000 | 30,000 | 4,630,966 | 0.4890 | 0.1486 | 0.0394 | 0.0471 | -0.0155 / 0.8664 |
| Val Sep 22 (slot 16) | 7,542 | 727,571 | 0 | 0.2090 | 0.0275 | 0.0589 | -0.0107 / 0.9128 |
| Val Sep 24 (slot 18) | 6,878 | 661,957 | 0 | 0.1873 | 0.0282 | 0.0628 | -0.0130 / 0.8941 |
| **Val, all** (incl. 3 feeds on slots 17, 19) | 14,423 | 1,389,971 | 0 | 0.1987 | 0.0278 | 0.0607 | -0.0118 / 0.9039 |
| Val twin, slot 16 / 18 / all | as val | as val | 0 | 0.1594 / 0.1555 / 0.1576 | 0.0338 / 0.0335 / 0.0337 | 0.0822 / 0.0804 / 0.0813 | all: -0.0155 / 0.8698 |
| Test Sep 26 / 27 / 28 / 29 | 6,050 / 7,367 / 7,578 / 7,600 | 624,428 / 766,256 / 754,413 / 759,709 | 0 | 0.1808 / 0.1758 / 0.1725 / 0.1713 | 0.0224 / 0.0245 / 0.0251 / 0.0244 | 0.0633 / 0.0622 / 0.0686 / 0.0643 | sd 0.887 / 0.881 / 0.879 / 0.875 |
| **Test, all** | 28,595 | 2,904,806 | 0 | 0.1748 | 0.0242 | 0.0647 | -0.0150 / 0.8801 |
| Test public / private | 8,562 / 20,033 | 886,924 / 2,017,882 | 0 | 0.1800 / 0.1726 | 0.0226 / 0.0249 | 0.0621 / 0.0658 | sd 0.885 / 0.878 |

Two readings.

- **All train rows carry more non-zero values than val and test rows (0.233 against 0.199 and 0.175); pool and holdout feeds carry fewer (0.138-0.177).** Train is mostly production feeds, val and test are random-like. This is the source of the flag in 3.3.
- **After the loader's NaN -> 0, `nx_rp3_z` has sd 0.640 on train rows against 0.904 on val and 0.880 on test** [`out/install.txt`]. The model standardises with train statistics, so val and test values come out 1.4 times wider than train values. This follows from the missing flag on 53% of train rows, a stated defect of the view (DESIGN 1.2).

---

## 3. The gates

Source of my values: `nn_x/w12/nx3/out/nx3_gates.txt` (read 04:10:06Z), `out/c1_repro.txt`, `out/build.log`. Critic's advance values: `reports/INPUTS_1116/critic/out/c03_gates.txt`, `c05_coef.txt`, `c06_joint.txt`.

`nx3_gates.py` (md5 `2990bad514f9e6dfca12fb53d6b8cd81`) was written while the build was running. Its header states what decides and what is printed for comparison only. After the build log was visible I changed one thing in it, the label of a comparison row. It was run once.

### 3.1 Table

| Gate | Bar | Build of record | Critic's advance | |
|---|---|---|---|---|
| **G-code C1** stored `c09_val` reproduced (c09 conventions) | 1e-5 relative | max abs diff 0 on 1,389,971 rows, 0 zero-pattern mismatches | 0 | ok |
| **G-code C1** stored `c09_hold` reproduced, cut = own day, no sparse members | 1e-5 relative | max abs diff 0 on all 4,630,966 rows (14 day indexes) | 0 on 2 days (694,799 rows) | ok |
| **G-code C2** causality counts, every admitted (row, sequence) pair | all 0 | 0 of 1,388,410,566 (train), 17,218,069 (val), 32,741,802 (test), 12,808,523 (twin) | not run | ok |
| **G-parity C3** test-like pool rows vs private: dZ / largest bucket | 0.05 / 0.10 | 0.0295 / 0.0499 | 0.0365 / 0.0665 (holdout feeds) | ok |
| C3 test-like pool rows vs public | 0.05 / 0.10 | 0.0270 / 0.0527 | 0.0383 / 0.0695 | ok |
| C3 val-like pool rows vs val | 0.05 / 0.10 | 0.0316 / 0.0511 | 0.0417 / 0.0675 | ok |
| C3 on the holdout feeds (comparison): private; public; val | no bar | 0.0366 / 0.0664; 0.0382 / 0.0693; 0.0416 / 0.0674 | as above | |
| **G-parity C4** PSI `nx_rp3_z`, pool has = 1 rows vs val / public / private | 0.25 | 0.0363 / 0.0100 / 0.0129 | 0.0884 / 0.0230 / 0.0109 (critic bins, holdout rows) | ok |
| C4 PSI `nx_rp3_rk`, same | 0.25 | 0.0914 / 0.0293 / 0.0165 | 0.1710 / 0.0681 / 0.0451 | ok |
| C4 with the critic's bins on the holdout rows (comparison): z; rank | no bar | 0.0858 / 0.0221 / 0.0099; 0.1681 / 0.0670 / 0.0455 | as above | |
| **C4 on ALL has = 1 train rows (comparison): z; rank** | no bar | **0.1267 / 0.2603 / 0.3462**; 0.0410 / 0.1264 / 0.1732 | not built | flag |
| **G-parity C5** lowest non-zero share of a test day | >= 0.1045 (half of val's 0.2090) | 0.1713 (days: 0.1808 / 0.1758 / 0.1725 / 0.1713) | 0.1705 | ok |
| **G-info C7** holdout rows, lag view (15,225 feeds): cz of `nx_rp3_z` / `nx_rp3_rk` | + and >= 2.5, z or rank | **+4.98 / +4.32** | +5.02 / +4.43 | ok |
| C7 val rows, stale-val twin, train-only pool | same | **+6.46 / +4.78** | +6.64 / +4.90 | ok |
| C7 val rows, stale-val twin, file FD16P20 | same | +6.51 / +4.79 | +6.70 / +4.92 | ok |
| C7 natural val files (information): train-only pool; file | no bar | +6.78 / +5.99; +6.84 / +6.01 | +6.84 / +5.97; +6.91 / +6.00 | |
| **G-info C6** 20 permutations, largest \|cz\|: holdout lag view / val twin | < 3.0 | 1.78 / 1.63 (sd 0.91 / 0.82) | 1.93 / 1.55 (sd 0.89 / 0.91) | ok |
| **J1** P(walk = 0 given `seq_cv_invd20` > 0): holdout lag view Sep 12-14 / Sep 17-19 | no bar | 0.122 / 0.228 | 0.123 / 0.229 | |
| J1 on all lag-view train rows, slots 6-9 / 11-14 | no bar | 0.118 / 0.216 | not built | |
| J1 on val / public / private rows; val twin | no bar | 0.0004 / 0.0005 / 0.0004; 0.154 | 0.002 / 0.002 / 0.003; 0.157 | |
| **J2** one-step coefficient of `nx_rp3_z`: holdout lag view all / Sep 12-14 / Sep 17-19 | no bar | 0.0297 / 0.0290 / 0.0221 | 0.0299 / 0.0285 / 0.0230 | |
| J2 on val: natural / twin | no bar | 0.0554 / 0.0501 | 0.0556 / 0.0513 | |
| J2 R2 of `nx_rp3_z` with pool + own inputs: holdout lag view / val natural / twin | no bar | 0.25 / 0.52 / 0.42 | 0.25 / 0.52 / 0.42 | |

**G-code PASS, G-parity PASS, G-info PASS -> STEP 0 PASS.** All six cz of C7 (z and rank form; holdout, twin on both bases) are above 2.5, so the "z or rank" wording decides nothing.

### 3.2 Checks behind G-code

- **The lens run is the critic's.** The stored fresh `rp3b05` on all 30,000 holdout feeds reads +8.20 / +9.75, the value on record [`out/nx3_gates.txt`].
- **Independent re-derivation.** A numpy implementation of BUILD_SPEC 2 with the true timestamps, slots and users reproduces the kernel output on 247 train feeds (25,039 rows), 60 val, 60 test and 150 twin feeds: largest relative difference 1.0e-14, 0 zero-pattern mismatches [`out/build.log`].
- **The checks can fail.** With the driver deliberately wrong (sparse members always admitted) the audit counts 1,572 violations on 400 feeds and the re-derivation disagrees [`out/minitest.txt`].
- **Natural val against the stored `c09_val`** (BUILD_SPEC C1 asks for the correlation only): corr 0.884 raw, 0.996 after log1p; 666 rows non-zero in mine only, 1 in the stored array only [`out/build.log`]. The single row was not traced; I take it for an effect of the 500-posting cap.

### 3.3 The flag on C4

- **What decides.** BUILD_SPEC C4 names "train has = 1 rows (pool feeds)": p_random > 0.5, M >= 50, not holdout, 69,173 feeds and 10,254,287 rows with the view. That is the decision population of the project's parity gates (`nn_x/w12/stale/hz2/gate.py`, trPool). It passes at 0.036 / 0.010 / 0.013 (z) and 0.091 / 0.029 / 0.017 (rank).
- **What is flagged.** On all 115,150,397 has = 1 train rows `nx_rp3_z` reads 0.127 / 0.260 / 0.346 against val / public / private (0.127 / 0.264 / 0.347 with the critic's bins). The rank file stays below the bar (at most 0.187).
- **What the same output says about the cause.** The adopted input `seq_cv_invd20` is non-zero on 0.190 of all lag-view train rows against 0.114 (val), 0.102 (public), 0.098 (private), and on 0.108-0.112 of holdout rows [`out/nx3_gates.txt`, J1 lines]. The gap between all train rows and test rows is the feed mix of train, and an adopted input shows it too.
- **What I did not do.** I added no look after the read: the PSI of adopted inputs on all train rows is not computed.
- **Decision for the lead.** If "train (has = 1)" in the ledger is ruled to mean all rows, G-parity fails on this one cell and the outcome is "NOT TESTABLE AT PARITY IN 14 DAYS OF DATA".

### 3.4 Thin places that are not gates

- On the Sep 17-19 half of the holdout alone the files read +2.89 (z) and +2.59 (rank); the critic had +3.00 / +2.76. The gate is on all 15,225 feeds.
- The lag view keeps 0.72 (z) and 0.77 (rank) of the fresh column's cz on the same feeds, and half of its likelihood (0.00044 against 0.00088 nats).
- J1 and J2 repeat the critic's two limits: on lag-view train rows 12-23% of windowed hits have no walk value, against 0.04-0.05% on val and test rows; the coefficient that fits train rows is 0.54 of the one that fits val rows.

---

## 4. Difference to the critic's advance build (timestamp against day)

Source: `nn_x/w12/nx3/out/diff_critic.txt`. The critic admitted sparse members by day and counted them in the degree; the build of record admits by timestamp and counts dense postings only, floor 1.

| Rows | Non-zero share, record / critic | Rows non-zero in the record only / critic only | corr `nx_rp3_z` | corr `nx_rp3_rk` |
|---|---|---|---|---|
| Natural val | 0.1987 / 0.1982 | 666 / 1 | 0.9958 | 0.9982 |
| Natural test | 0.1748 / 0.1742 | 1,957 / 2 | 0.9910 | 0.9970 |
| Stale-val twin | 0.1576 / 0.1569 | 976 / 2 | 0.9910 | 0.9967 |
| Holdout lag view (2,366,505 rows) | 0.1486 / 0.1481 | 1,279 / 0 | 0.9942 | 0.9975 |

- **My driver in the critic's convention reproduces the critic's holdout array exactly** (max abs diff 0, 0 zero-pattern mismatches), so every difference comes from the two conventions.
- **Degree rule alone** (day admission kept): no row changes its zero pattern, 9.6% of rows change value, corr of `nx_rp3_z` 0.9994.
- **Admission rule alone:** 1,279 more non-zero rows, 0.18% of rows change value, corr 0.9949.
- **Effect on the gates:** C3 on the holdout feeds moves by at most 0.0002, C7 by at most 0.19 in cz (the critic's K differs too), J1 on val and test rows falls from 0.002-0.003 to 0.0004-0.0005.

---

## 5. Launcher copies

`nn_x/w12/nx3/launch_nx3_hl.sh` (stage A) and `nn_x/w12/nx3/launch_nx3_kd.sh` (stage B; not run, not to be run before stage A is read). Diffs against the designer's scripts: `nn_x/w12/nx3/out/launch_nx3_hl.diff`, `out/launch_nx3_kd.diff`. Both pass `bash -n`.

Diff of the stage A launcher, comment lines left out:

```
< g=${1:?gpu}; s=${2:?seed}; arm=${3:?nx|nx4|base}
< NX7=nx_rp3_z,nx_rp3_rk,nx_p3a_z,has:nx_rp3_z,nx_bt_z,nx_bt_rk,nx_bt_seen
< NX4=nx_rp3_z,nx_rp3_rk,nx_p3a_z,has:nx_rp3_z
< case $arm in nx) XA="--xt_add $NX7" ;; nx4) XA="--xt_add $NX4" ;; base) XA="" ;; *) echo "arm must be nx | nx4 | base"; exit 1 ;; esac
> g=${1:?gpu}; s=${2:?seed}; arm=${3:?nx3|base}
> NX3=nx_rp3_z,nx_rp3_rk,has:nx_rp3_z
> case $arm in nx3) XA="--xt_add $NX3" ;; base) XA="" ;; *) echo "arm must be nx3 | base"; exit 1 ;; esac

< t0=$(date +%s); until bash ftbag/gpu_ok.sh $g; do ... exit 3; }; sleep 60; done
> exec 9><repo>/nn_x/w5/logs/locks/gpu${g}.lock; flock 9
> t0=$(date +%s); until bash ftbag/gpu_ok.sh $g; do ... flock -u 9; exit 3; }; sleep 60; done

<  > logs/$v.stdout 2>&1 < /dev/null &
>  > logs/$v.stdout 2>&1 < /dev/null 9>&- &

< [[ -n $p ]] || { echo "could not resolve the train.py PID of $v"; exit 5; }
> [[ -n $p ]] || { echo "could not resolve the train.py PID of $v"; flock -u 9; exit 5; }
> tl=$(date +%s); until { nvidia-smi ... | grep -qx "$p" && grep -q "steps/epoch" logs/$v.stdout; } || ! kill -0 $p || (( $(date +%s) - tl > 1800 )); do sleep 5; done
> flock -u 9; exec 9>&-
```

plus the text of the registry line. Every training flag is unchanged. The stage B copy has the same three kinds of change. The added lock is the lanes' protocol: taken before the GPU check, held until the job is visible in `nvidia-smi` and has loaded its data, 30 minutes at most.

---

## 6. Smoke

Commands, from `<repo>`, 04:11:30Z:

```
SMOKE=1 bash nn_x/w12/nx3/launch_nx3_hl.sh 2 23 base    # abl_smoke_base_s23, PID 1374456
SMOKE=1 bash nn_x/w12/nx3/launch_nx3_hl.sh 4 23 nx3     # abl_smoke_nx3_s23,  PID 1374453
```

- **GPU rule.** GPU N and GPU N each had 3 compute processes and 90 GB free at the check under the lock [`nn_x/w12/nx3/logs/smoke_launch_*.out`]. Both runs had ended by 04:14:05Z and both GPUs were back at 3 processes.
- **Names.** Outputs are `abl_smoke_{base,nx3}_s23*` in `nn_x/abl/{logs,ckpt,preds,preds/holdout}`; no real variant has these names. Two registry lines were appended to `nn_x/abl/RUNNING.txt`, as the launcher does.
- **Seed 23, not 1.** The stored smokes of the harness have seed 23, which gives a precedent.

Read: `nn_x/w12/nx3/out/smoke_check_r2.txt`, **PASS**.

| Check | Result |
|---|---|
| Base log lists 34 inputs | yes; names and order equal to the logged list of the stored control `w5_e10_base_s951` |
| nx3 log lists 34 + 3 inputs | yes, 37: the base list + `nx_rp3_z`, `nx_rp3_rk`, `has:nx_rp3_z` at indices 34, 35, 36 |
| Base command against the stored control's logged arguments | differs only in variant, seed and the five smoke keys; every input and distillation flag of the harness off |
| nx3 command against base | differs only in variant and `--xt_add nx_rp3_z,nx_rp3_rk,has:nx_rp3_z` |
| Checkpoint cfg | base = `w5_e10_base_s951` on every key; nx3 differs in `xt_names` (37) and `n_dense` (107 against 104) only |
| Parameters | nx3 has 777 more values = 3 x (256 + 1) weights + 6 statistics |
| **Smoke control against the precedent** | the base smoke's input statistics `f_mean` / `f_std` (104 inputs, 40 batches of the 19,827 training feeds) are **bitwise equal** to the stored `nn_x/abl/ckpt/abl_smoke_kv64_s23.pt` and `nn_x/r7/ckpt/r7_smoke-base_s23.pt` |
| Data lines of the log | equal to the stored `abl_smoke_kv64_s23` and `abl_smoke_pre2noid_s23`: 30,000 holdout feeds, 19,827 training feeds, 73 steps, cooldown from step 58, pool 843 feeds |
| nx3 statistics on the 104 old inputs | bitwise equal to the base smoke |
| nx3 statistics of the 3 new inputs | `nx_rp3_z` sd 0.637, `nx_rp3_rk` sd 0.141, flag mean 0.457; the files give 0.640, 0.142, 0.467 |
| Predictions | val, test and holdout written for both arms, all finite |

- **Limit of the control.** No stored smoke uses the stage-A command (lr 2e-3, hard labels, d128), so there is no trained output to reproduce. The precedent is reproduced on what does not depend on the learning rate: inputs, standardisation statistics, feed sample, step counts.
- **The first read printed FAIL on one line** [`out/smoke_check.txt`]. My check compared the Muon line with `abl_smoke_kv64_s23`, an arm that carries an adapter (48 matrices against 46). `nx3_smoke_check_r2.py` compares the two model-size lines with `abl_smoke_pre2noid_s23` instead; nothing else changed.
- **Smoke NDCG is not evidence** (73 steps, one seed, different first-layer shape): base val 0.34384 / holdout 0.26693, nx3 0.34170 / 0.26183.

---

## 7. Stage A commands (not launched)

Run from `<repo>`, each on a GPU that qualifies at that moment, so that the launcher holds the lock for about a minute and not while it waits. `<g>` = GPU id. None of the ten variant names exists in `nn_x/abl/{logs,ckpt,preds}`.

```
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 957 base   # abl_e10_base_s957
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 958 base   # abl_e10_base_s958
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 951 nx3    # abl_e10_nx3_s951
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 952 nx3    # abl_e10_nx3_s952
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 953 nx3    # abl_e10_nx3_s953
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 954 nx3    # abl_e10_nx3_s954
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 955 nx3    # abl_e10_nx3_s955
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 956 nx3    # abl_e10_nx3_s956
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 957 nx3    # abl_e10_nx3_s957
bash nn_x/abl/ftbag/gpu_ok.sh <g> && bash nn_x/w12/nx3/launch_nx3_hl.sh <g> 958 nx3    # abl_e10_nx3_s958
```

The launcher issues, in `nn_x/abl`, with 12 CPU threads and `CUDA_VISIBLE_DEVICES=<g>`:

```
<repo>/.venv/bin/python train.py --variant abl_e10_<arm>_s<seed> --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain
 --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2
 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500 --save_last --save_holdout --select_min_step 12000 --cliplog 50
 --seed <seed> [--xt_add nx_rp3_z,nx_rp3_rk,has:nx_rp3_z   for arm nx3 only]
```

- **GPU state at 04:14Z:** every GPU had 3 compute processes and 90 GB or more free, so eight runs fit at once and two wait.
- **Read:** `.venv/bin/python reports/INPUTS_1116/design/read_nx.py abl_e10_nx3:951-958 w5_e10_base:951-956+abl_e10_base:957-958 <out>`. The tool predates the amendments: it prints "CLOSED by the rule" (l.105) and has neither the ORDER step, the VALUE clause nor the has = 0 report of A3. A write-new copy is needed before the read.
- **Stale-val clause:** the twins are in `nn_x/w12/nx3/staleval/`, not in `nn_x/nx/staleval/` as DESIGN 2.2 says for the override table. The twin has no NaN, so the flag stays 1.

---

## 8. Deviations from BUILD_SPEC.txt

| Item | BUILD_SPEC | Done | Why |
|---|---|---|---|
| Work directory | `nn_x/nx/` | `nn_x/w12/nx3/` | the lead's task |
| Files | 6 names | 2 (`nx_rp3_z`, `nx_rp3_rk`) | amendment A1; family T and `nx_p3a_z` are not built |
| Degree | "number of dense source sequences that hold the item" | posting count in the dense part, floor 1 | the kernel of record counts postings, and C1 can only be met that way |
| C1 natural val | correlation with `c09_val` printed | done; in addition the stored val array is reproduced exactly in the c09 conventions | the task asks for both stored arrays reproduced |
| C3, C4 train side | pool feeds | pool feeds decide; the holdout feeds and all has = 1 rows are printed | comparison with the critic; the ledger's shorter wording |
| C4 bins | 10 val deciles of the non-zero values + a zero bin | applied to the file values as they are | literal reading; the critic's 9 edges over all values are printed next to it |
| C7 | "for each column" | the two files are scored as entered (NaN -> 0), not re-transformed | the files are the z form and the rank form; the critic's formula on my raw array gives the same cz to two decimals |
| Sparse train members | "train feeds with sparse_flag" | every sampled feed is its own sequence, not de-duplicated by user | as val / test feeds in the adopted builder and as the critic's build |
| Stage B launcher, smoke seed, lock | - | see sections 5 and 6 | |

Nothing was transformed, re-binned or re-lagged after a gate was read.

---

## 9. Everything run

| Time (UTC) | What | Output |
|---|---|---|
| 03:57 | mechanics test: one index, 400 feeds, correct and deliberately wrong driver (label-free) | `out/minitest.txt` |
| 03:58-04:01 | C1 reproduction, 169 s | `out/c1_repro.txt` |
| 03:58-04:08 | build of record, 591 s | `out/build.log`, `out/files/`, `out/raw/`, `staleval/` |
| 04:08-04:10 | gates, one run, 101 s | `out/nx3_gates.txt` |
| 04:08-04:09 | difference to the critic's build (label-free), 62 s | `out/diff_critic.txt` |
| 04:10 | install | `out/install.txt` |
| 04:11-04:14 | two smoke runs on GPU N and GPU N | `nn_x/abl/logs/abl_smoke_{base,nx3}_s23.*` |
| 04:14 | smoke read, then the corrected read | `out/smoke_check.txt`, `out/smoke_check_r2.txt` |

- At most two 12-thread CPU processes of mine ran at once. No process of this task is left.
- I signalled no process. I wrote nothing to the ledger, to `reports/CURRENT_STATUS.md`, to the memory directory or to `subs/`. Nothing was submitted; no external service was used.
- Existing files touched: `nn_x/abl/RUNNING.txt` gained two registry lines from the launcher. At 04:16Z I probed the lock files of GPU N and GPU N with a non-blocking `flock`; both were free and were released at once.
- Not done: the fine-tune input check on the smoke checkpoint (DESIGN section 5); the copies of `stale_pred.py` / `holdout_eval.py` for the stale-val and VALUE clauses; the amended read tool.

---

## 10. Text for the lead to record

```
=== <UTC at copy> nx3 STEP 0 READ (builder; report reports/INPUTS_1116/nx3_step0.md; gate file nn_x/w12/nx3/out/nx3_gates.txt dated 2026-11-16T04:10:06Z) ===
 BUILD OF RECORD: kernel c09_build.py (md5 0975dfd6...) imported unchanged; sparse members by TIMESTAMP, deg = dense posting count floor 1 (BUILD_SPEC 2).
   Files nx_rp3_z, nx_rp3_rk in data/feats/{train,val,test} + hard links in nn/feat_cache (md5 train 76a972ad... / d341219d...); has:nx_rp3_z from the NaN pattern
   (1 on 1,470,065 train feeds = 0.4667 of rows; always 1 on val / test). Stale-val twins nn_x/w12/nx3/staleval/. K = 1.932062e+05.
 G-code   PASS: c09_val and c09_hold reproduced with max abs diff 0 on 1,389,971 and on all 4,630,966 rows; causality counts 0 of 1.39e9 (train), 1.7e7 (val),
          3.3e7 (test), 1.3e7 (twin) admitted pairs; numpy re-derivation on 517 feeds agrees to 1e-14.
 G-parity PASS: C3 pool rows dZ 0.030 (private) / 0.027 (public) / 0.032 (val), largest bucket 0.050-0.053; C4 PSI pool has = 1 rows z 0.036 / 0.010 / 0.013,
          rank 0.091 / 0.029 / 0.017 (val / public / private); C5 lowest test day 0.171 against a bar of 0.105.
          FLAG (comparison, not the BUILD_SPEC population): on ALL 115,150,397 has = 1 train rows nx_rp3_z reads PSI 0.127 / 0.260 / 0.346. The lead rules whether
          "train (has = 1)" means the pool feeds of BUILD_SPEC C4 (PASS) or all rows (FAIL G-parity only -> NOT TESTABLE AT PARITY).
 G-info   PASS: holdout lag view cz +4.98 (z) / +4.32 (rank); stale-val twin +6.46 / +4.78 (train-only pool), +6.51 / +4.79 (file); 20 permutations max 1.78 / 1.63.
 J1 (no bar) P(walk = 0 | seq_cv_invd20 > 0): 0.122 (Sep 12-14) / 0.228 (Sep 17-19) on lag-view holdout rows against 0.0004 / 0.0005 / 0.0004 on val / public / private.
 J2 (no bar) coefficient of nx_rp3_z: 0.0297 on lag-view holdout rows against 0.0554 (val natural) and 0.0501 (twin).
 Against the critic's advance build (days, deg with sparse): corr of nx_rp3_z 0.991-0.996, non-zero share +0.0005..+0.0007; no gate changes side.
 VERDICT by the amended rule, BUILD_SPEC reading of C4: STEP 0 PASS.
 SMOKE (seed 23, GPUs N and 4, 04:11:30Z-04:14:05Z, AFTER the read and BEFORE this entry): base lists 34 inputs = w5_e10_base_s951, nx3 lists 34 + 3; base smoke
   input statistics bitwise equal to the stored seed-23 smokes abl_smoke_kv64_s23 and r7_smoke-base_s23; no stored smoke of the stage-A command exists.
 Stage A NOT launched. Launcher nn_x/w12/nx3/launch_nx3_hl.sh <gpu> <seed> nx3|base (designer's flags + the lanes' GPU lock).
```

---

## 11. Files (all in `nn_x/w12/nx3/` unless a path is given)

| File | Content |
|---|---|
| `nx3_common.py` | driver of the kernel of record, causality audit, numpy re-derivation, forms |
| `nx3_minitest.py` -> `out/minitest.txt` | mechanics test |
| `nx3_c1_repro.py` -> `out/c1_repro.txt` | C1 |
| `nx3_build.py` -> `out/build.log` | build of record, C2, per-day table |
| `nx3_gates.py` -> `out/nx3_gates.txt` | C3-C7, J1, J2, verdict |
| `nx3_diff_critic.py` -> `out/diff_critic.txt` | timestamp against day, degree rule |
| `nx3_install.py` -> `out/install.txt` | install into `data/feats` and `nn/feat_cache` |
| `launch_nx3_hl.sh`, `launch_nx3_kd.sh`; `out/launch_nx3_*.diff` | launcher copies and their diffs |
| `nx3_smoke_check.py`, `nx3_smoke_check_r2.py` -> `out/smoke_check.txt`, `out/smoke_check_r2.txt` | smoke read, first and corrected |
| `out/files/<split>/`, `out/raw/`, `staleval/` | staged files, raw arrays, stale-val twins |
| `logs/` | stdout of every run |
