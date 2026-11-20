# nx3 STEP 0: independent verification

Verifier's report on the builder's STEP 0 (`reports/INPUTS_1116/nx3_step0.md`, files in `nn_x/w12/nx3/`). Written 2026-11-16 05:00 UTC. CPU only, no GPU process, nothing submitted, no process signalled. I wrote only into `nn_x/w12/nx3/verify/` and this file. Paths are relative to `<repo>`; every number names its file (`out/` = `nn_x/w12/nx3/verify/out/`).

## 0. Verdict

**GO.** The installed inputs are leak-free, built in the lag view exactly as `BUILD_SPEC.txt` prescribes, aligned row for row with `data/flat`, and correctly wired. The gate table reproduces to the digit.

| Question | Result | Source |
|---|---|---|
| (a) Leak | **0** rows of any split need an inadmissible source. Every installed value is reproduced from label-free inputs by my own code: all 246,745,836 train rows (the 30,000 holdout feeds included), all 1,389,971 val rows, all 2,904,806 test rows, all rows of the stale-val twin | `out/v05_parity_align.txt` A, `out/v02_compare.txt` |
| (b) Lag view | Built as specified: cut' = max(slot - L, 5), newest dense source never younger than 1.001 / 2.001 / 3.001 d on Sep 12 / 13 / 14 and 6.001 d on Sep 17-19. has = 0 exactly on Sep 6-11. **But the view is fresher than val and private rows in distribution** (section 3) | `out/v05_parity_align.txt` B, C |
| (c) Alignment | Proven for every row: the full files equal a recomputation indexed by `data/flat` row. Three invariants that do not use my walk read 0 violations and fail massively on a shifted array | `out/v05_parity_align.txt` A, F |
| (d) Wiring | Launcher flags equal the designer's and the stored controls' logged arguments; smoke logs list 34 and 34 + 3; the fine-tune input check passes on the nx3 checkpoint; no pre-existing file was modified except the registry append | `out/v04_wiring.txt`, `out/v04b_wiring_kd.txt`, `out/v07_ftcheck_r2.txt`, `out/v08_untouched.txt` |
| (e) Gates | G-parity and G-info reproduce the builder's values exactly (pool-feed reading of C3 / C4) | `out/v05_parity_align.txt` E, `out/v06_ginfo.txt` |

**Three things the lead must decide or record. None is a defect of the build.**

1. **The GO rests on the BUILD_SPEC reading of C3 and C4 (pool feeds).** On all 115,150,397 has = 1 train rows the picture is different, and not only for C4 as the builder reported:
   - C4, `nx_rp3_z`: PSI 0.127 / 0.260 / 0.346 against val / public / private (bar 0.25). I confirm the builder's numbers.
   - C3 (not computed by the builder): val-like dZ 0.059 (bar 0.05); largest bucket 0.115 (val-like) and 0.126 (test-like against private) (bar 0.10).
   - Under an all-rows reading the outcome would be "FAIL G-parity only -> NOT TESTABLE AT PARITY".
   - My reading: the rule as written means the pool. `BUILD_SPEC.txt` C3 and C4 name the population explicitly, the ledger's C-labels refer to BUILD_SPEC, and the advance read that the ledger records as inside the bars (A2) was taken on random-like feeds.
2. **"Never fresher than val / test rows" does not hold in distribution.** Median age of the newest contributing source: 3.9 d on val-like pool rows against 5.6 d on val; 8.5 d on test-like pool rows against 8.8 d (public) and 10.9 d (private). The sparse trickle is the newest source on 5.1% / 9.9% of non-zero pool rows against 1.7% (val), 4.5% (public), 7.7% (private). This follows from BUILD_SPEC 1 and 2.2 as written; the ledger's clause "on holdout rows the new columns are lag-built (never fresher than on val)" is not what the files show. I propose it is printed in the STEP 0 read next to J1 and J2 (section 9).
3. **The smoke ran before the STEP 0 read was in the ledger**, against the letter of "NO GPU process of this arm before it is read and written here". The builder disclosed it. As of 04:52Z the read is still not in the ledger, while amendment A7 (04:48Z) makes stage B and the 16 all-data students conditional on this GO.

---

## 1. Method

My code imports nothing from `nn_x/w12/nx3/*.py` and nothing from `c09_build.py`.

- **Walk** (`verify/v_common.py`): a sequence-centric scan. The source sequences of one index are visited in reverse index order; for each event the query rows that hold the item as a candidate are looked up; each admitted (row, sequence) pair adds its contribution. No posting list is built, so the method differs from the kernel of record, which walks an inverted index candidate by candidate.
- **Member selection** is my own: dense = train feeds with slot < cut, the latest feed of each non-zero user plus every user-0 feed; sparse = the listed feeds, admitted iff timestamp < own timestamp; every sequence of the own non-zero user skipped; degree = events of the item in the dense part, floor 1.
- **Inputs opened**: `data/flat/<split>/{feeds.parquet, hist_item_id, hist_hpos, cand_item_id}` and `hz2_lag.npy`. No target array. `train_prandom.npz` only names the holdout and pool populations.
- **Forms** (`my_forms`): my own within-feed z of log1p(K x) and mid-rank, with my own K = 1.932062e+05 from my val walk (the builder's `K.npy` is identical) [`out/v02_compare.txt`].
- **Tie to the record**: on all 30,000 holdout feeds in c09's convention my code reproduces the stored `c09_hold.npz` of 2026-11-13 with max abs diff 0 on 4,630,966 rows [`out/v10_c1.txt`].

Coverage: all val, test and twin rows (`v01_walk.py`), a 13,276-feed train sample with leaky variants (`v01_walk.py train`), then **all 1,470,065 has = 1 train feeds** (`v03_trainfull.py`, 3 processes, 12 minutes).

---

## 2. (a) Leak

### 2.1 Counts

"Not admissible" = rows whose installed value (as the trainer reads it, `nn/feat_cache/<split>/`) differs from my admissible-only recomputation by more than 2e-5 (z) or 1e-6 (rank).

| Split | Feeds | Rows compared | Not admissible | Largest difference z / rank | Source |
|---|---|---|---|---|---|
| Train, has = 1 (all) | 1,470,065 | 115,150,397 | **0** | 2.4e-07 / 1.5e-08 | `out/v05_parity_align.txt` A |
| Train, has = 0 (all) | 1,681,143 | 131,595,439 | 0 (NaN in both files on every row) | - | same |
| Holdout lag-view feeds (inside the first row) | 15,225 | 2,366,505 | **0** | - | same; sample of 3,600 feeds also in `out/v02_compare.txt` |
| Val (all) | 14,423 | 1,389,971 | **0** | 2.4e-07 / 1.5e-08 | `out/v02_compare.txt` |
| Test (all; public 8,562 / private 20,033 feeds) | 28,595 | 2,904,806 | **0** | 2.4e-07 / 1.5e-08 | same |
| Stale-val twin (all) | 14,423 | 1,389,971 | **0** | 2.4e-07 / 1.5e-08 | same |

On val, test, twin and the train sample my raw values are bit-equal to the builder's raw arrays on every row [`out/v02_compare.txt`].

Audit of my own admitted pairs (true timestamps, slots, users from the feeds tables):

| Split | Admitted pairs | Dense member with slot >= cut | Sparse member with timestamp >= own | Own non-zero user | Own feed | Any member with timestamp >= own |
|---|---|---|---|---|---|---|
| Train, all has = 1 rows | 1,398,646,525 | 0 | 0 | 0 | 0 | 0 |
| Val | 17,378,572 | 0 | 0 | 0 | 0 | 0 |
| Test | 32,876,719 | 0 | 0 | 0 | 0 | 0 |
| Twin | 12,927,366 | 0 | 0 | 0 | 0 | 0 |

[`out/v05_parity_align.txt` A; `out/v01_*.log`]. After removing the pairs of query feeds with an empty history, which the builder's audit skips, my pair counts equal the builder's C2 counts exactly: 17,218,069 (val), 32,741,802 (test), 12,808,523 (twin) [`out/v09_extra.txt` 3]. Two independent implementations enumerate the same set of pairs.

### 2.2 The three leak classes: the check can fail

The same scan carries three deliberately wrong variants. Rows whose raw walk would change:

| Variant | Train, all has = 1 rows | Val | Test | Installed value sides with the variant |
|---|---|---|---|---|
| Own user's sequences admitted | 25,090 | 501 | 958 | 0 rows |
| Sparse source at or after the row's time admitted | 2,843,096 | 23,818 | 22,611 | 0 rows |
| User 0 linked (a user-0 query skips every user-0 sequence) | 85,972 | 1,181 | 1,578 | 0 rows |

[`out/v05_parity_align.txt` A; `out/v02_compare.txt`; "sides with" from `out/v09_extra.txt` 1 on val, test, twin and the train sample]. A dense source at or after the cut cannot be in the installed values either: the dense member set of each cut is my own selection, and the files equal the recomputation.

### 2.3 Labels

- The files are reproduced exactly from inputs that hold no label, so no val label and no feed label can have entered.
- `hz2_lag.npy` equals `hz2_common.assign_lags(feed_id)`, a hash of the feed id, on all 3,151,208 feeds [`out/v01_train.log`].
- In the builder's build path (`nx3_common.py`, `nx3_build.py`, `nx3_install.py`) no label array is opened; only `nx3_gates.py` opens labels, for G-info [grep].

---

## 3. (b) Lag view

### 3.1 The has flag

- NaN pattern of the two train files identical; constant inside every feed; has = 1 on 1,470,065 feeds, equal to (slot >= 6) feed by feed.
- has = 0 on every feed of Sep 6-11 (slots 0-5: 291,760 / 281,767 / 290,992 / 286,479 / 282,716 / 247,429 feeds), has = 1 on every feed of Sep 12-14, 17-19 and on the stragglers of Sep 15 (42 feeds) and Sep 20 (34 feeds).
- Holdout: 15,225 feeds with the view, 14,775 without.
- Val and test: no NaN in either file, so the loader (`nn_x/abl/data.py` `attach_xt`: non-NaN indicator, then NaN -> 0) gives has = 1 on every row.

[`out/v05_parity_align.txt` B]

### 3.2 Age of the newest contributing source (non-zero rows; days)

"Sparse-newest" = the newest contributing source is a sparse member. "Dense min / p50" = the newest contributing dense source.

| Rows | Non-zero rows | p25 | p50 | p75 | Younger than 2 d | Younger than 6 d | Sparse-newest | Value share through sparse members | Dense min / p50 |
|---|---|---|---|---|---|---|---|---|---|
| Train, all rows, Sep 12-14 | 13,962,634 | 2.63 | 3.68 | 4.87 | 0.139 | 0.883 | 0.065 | 0.026 | 1.001 / 3.78 |
| Train, pool rows, Sep 12-14 | 809,703 | 2.84 | 3.89 | 5.12 | 0.112 | 0.857 | 0.051 | 0.021 | 1.003 / 3.96 |
| Train, holdout rows, Sep 12-14 | 173,551 | 2.87 | 3.93 | 5.17 | 0.105 | 0.852 | 0.048 | 0.021 | 1.010 / 3.99 |
| **Val** | 276,162 | 4.04 | 5.58 | 10.26 | 0.014 | 0.540 | 0.017 | 0.008 | 2.004 / 5.64 |
| Val Sep 22 / Sep 24 | 152,086 / 124,009 | 3.18 / 5.14 | 4.46 / 6.51 | 9.12 / 11.20 | 0.010 / 0.018 | 0.635 / 0.422 | 0.010 / 0.027 | 0.004 / 0.013 | 2.004 / 4.48; 4.004 / 6.57 |
| Train, all rows, Sep 17-19 | 12,900,368 | 7.10 | 8.26 | 9.65 | 0.045 | 0.103 | 0.122 | 0.055 | 6.001 / 8.50 |
| Train, pool rows, Sep 17-19 | 860,540 | 7.29 | 8.48 | 9.91 | 0.034 | 0.082 | 0.099 | 0.048 | 6.003 / 8.68 |
| Train, holdout rows, Sep 17-19 | 178,150 | 7.32 | 8.50 | 9.94 | 0.032 | 0.077 | 0.093 | 0.045 | 6.010 / 8.69 |
| **Public** | 159,612 | 7.43 | 8.79 | 13.65 | 0.019 | 0.045 | 0.045 | 0.023 | 6.003 / 8.89 |
| **Private** | 348,228 | 9.25 | 10.86 | 15.75 | 0.039 | 0.070 | 0.077 | 0.043 | 7.481 / 11.07 |

By train day, all rows, p50: 2.42 / 3.43 / 4.41 d on Sep 12 / 13 / 14 and 7.44 / 8.29 / 9.03 d on Sep 17 / 18 / 19. On Sep 12, 33.8% of non-zero rows have a source younger than 2 days. [`out/v05_parity_align.txt` C]

Readings.

- **The build follows the spec.** The dense minimum is exactly the stated lag: 1, 2, 3 days on Sep 12-14 and 6 days on Sep 17-19.
- **Val-like rows are fresher than val rows.** The median newest source is 1.7 days younger (3.9 against 5.6 d), and 10-14% of non-zero rows have a source younger than 2 days against 1.4% on val. The critic called this "wrong for Sep 12 but immaterial"; it holds on all three days against Sep 24 and on Sep 12-13 against Sep 22.
- **Test-like rows match public rows and are 2.4 days fresher than private rows** at the median (8.5 against 8.8 and 10.9 d). The design states that lag 9 is missing and lag 8 under-represented.
- **The sparse trickle is denser on train rows.** BUILD_SPEC 2.2 admits the 1/37 sample on every slot from the cut on, while val and test rows only see val / test days. Result: sparse-newest 5.1% (pool, val-like) against 1.7% (val); 9.9% (pool, test-like) against 4.5% (public) and 7.7% (private). This was not measured by the designer or the critic.
- **Depth.** Value-weighted mean source age is 4.6-4.7 d (val-like) against 9.1 d (val), and 9.2-9.3 d (test-like) against 12.5 d (public) and 14.3 d (private). This is the stated 5-7 dense days against 12.
- **Direction.** All four gaps make the trained rows fresher than the rows the model is read on. They are properties of the pre-declared view, not of the build, and no bar covers them.

### 3.3 Non-zero share and PSI by train day

Non-zero share of the walk [`out/v05_parity_align.txt` D]:

| Train rows | Sep 12 | Sep 13 | Sep 14 | Sep 17 | Sep 18 | Sep 19 |
|---|---|---|---|---|---|---|
| All | 0.2611 | 0.2487 | 0.2399 | 0.2157 | 0.2167 | 0.2226 |
| Pool feeds | 0.1853 | 0.1778 | 0.1714 | 0.1508 | 0.1502 | 0.1530 |
| Holdout feeds | 0.1669 | 0.1639 | 0.1546 | 0.1388 | 0.1376 | 0.1382 |

References: val 0.1987 (Sep 22 0.2090, Sep 24 0.1873); public 0.1800; private 0.1726; test days 0.1808 / 0.1758 / 0.1725 / 0.1713. Stragglers: Sep 15 0.2388, Sep 20 0.2701.

PSI against val / public / private, BUILD_SPEC bins:

| Train rows | File | Sep 12 | Sep 13 | Sep 14 | Sep 17 | Sep 18 | Sep 19 |
|---|---|---|---|---|---|---|---|
| Pool | z | 0.005 / 0.036 / 0.065 | 0.005 / 0.020 / 0.042 | 0.013 / 0.009 / 0.023 | 0.082 / 0.024 / 0.013 | 0.093 / 0.032 / 0.018 | 0.090 / 0.031 / 0.019 |
| Pool | rank | 0.015 / 0.009 / 0.016 | 0.033 / 0.008 / 0.010 | 0.053 / 0.010 / 0.005 | 0.168 / 0.073 / 0.047 | 0.174 / 0.076 / 0.050 | 0.152 / 0.065 / 0.043 |
| Holdout | z | 0.019 / 0.013 / 0.024 | 0.021 / 0.014 / 0.023 | 0.049 / 0.018 / 0.014 | 0.153 / 0.060 / 0.032 | 0.184 / 0.081 / 0.048 | 0.184 / 0.082 / 0.053 |
| Holdout | rank | 0.072 / 0.021 / 0.013 | 0.081 / 0.030 / 0.018 | 0.132 / 0.055 / 0.034 | **0.285** / 0.150 / 0.114 | **0.295** / 0.158 / 0.119 | **0.279** / 0.150 / 0.113 |
| All | z | **0.252 / 0.431 / 0.541** | 0.194 / **0.355 / 0.455** | 0.163 / **0.313 / 0.407** | 0.077 / 0.185 / **0.258** | 0.069 / 0.173 / 0.243 | 0.077 / 0.183 / **0.255** |
| All | rank | 0.117 / 0.246 / **0.311** | 0.079 / 0.189 / 0.246 | 0.057 / 0.155 / 0.206 | 0.013 / 0.070 / 0.104 | 0.014 / 0.072 / 0.108 | 0.024 / 0.092 / 0.132 |

Bold = above 0.25. No day is a gate; the gate pools the days. On the pool no day exceeds 0.174. On the holdout feeds of Sep 17-19 the rank file reads 0.28-0.30 against val, which is the cross comparison (test-like days against val); against their own references they read 0.11-0.16.

---

## 4. (c) Alignment

- **Counts.** Every installed file is 1-D float32 with the length of `data/flat/<split>/cand_*.npy`: 246,745,836 / 1,389,971 / 2,904,806. `cand_fidx` equals repeat(arange(feeds), M) on all rows of the three splits, so the feeds table's `cand_off` is the row index the trainer uses (`SplitData`: "candidate order == flat file order", `self.xt[ci]`).
- **Proof by recomputation.** My walk takes each row's item from `cand_item_id` at the flat row index. All 246,745,836 train rows, all val rows and all test rows equal it (section 2.1). A misaligned file cannot pass this.
- **Invariants on the files alone** [`out/v05_parity_align.txt` F]:

| Rows | Feeds with a non-zero rank sum | Rows whose item is in no source sequence | of them not at the feed minimum | P(lowest rank group given `seq_cv_invd20` > 0) |
|---|---|---|---|---|
| Train has = 1, as installed | 0 of 1,470,065 | 46,140,175 (0.401) | **0** | 0.170 |
| The same array shifted by 1 row | 1,452,521 | same | 28,168,151 | 0.513 |
| Shifted by 137 rows | 1,450,757 | same | 31,802,878 | 0.339 |
| Val, as installed | 0 of 14,423 | 534,517 (0.385) | **0** | 0.0004 |
| Val shifted by 1 row | 14,245 | same | 330,071 | 0.547 |
| Test, as installed | 0 of 28,595 | 1,313,122 (0.452) | **0** | 0.0005 |
| Test shifted by 1 row | 28,226 | same | 816,888 | 0.558 |

- **Relation to the stored windowed input as the critic measured it**: P(walk = 0 given `seq_cv_invd20` > 0) is 0.1223 / 0.2280 on lag-view holdout rows of Sep 12-14 / Sep 17-19 and 0.0004 / 0.0005 / 0.0004 on val / public / private rows [`out/v05_parity_align.txt` E, J1], the builder's values.

---

## 5. (d) Wiring

### 5.1 Launchers

- **Stage A** (`nn_x/w12/nx3/launch_nx3_hl.sh`): the flag text of the `train.py` call is identical to the designer's `launch_nx_hl.sh`. The diff holds only the arm names, the `NX3` list, the per-GPU `flock`, and the registry text. All 27 flag keys equal the logged arguments of `w5_e10_base_s951`; the six stored controls differ among themselves only in seed and variant (s955, s956 also log a key `trunc_lifts` = '', off). [`out/v04_wiring.txt` 1-2]
- **Stage B** (`launch_nx3_kd.sh`): flag text identical to the designer's `launch_nx_kd.sh`; all 30 flag keys equal the logged arguments of `w5_e10_kd05w192_s951` (`--d 192`, `--kd_w 0.5`, `teacher_mean11`). [`out/v04b_wiring_kd.txt`]
- **Lock.** The lock is taken before the GPU check and held until the job is in `nvidia-smi` and has printed "steps/epoch", as `nn_x/w5/lane.sh` l.252-264 does. Like the lane, the launcher keeps the lock while it waits for a GPU to qualify (up to 4 h).
- Both pass `bash -n`. None of the stage-A variant names exists in `nn_x/abl/{logs,ckpt,preds}`.

### 5.2 Smoke

| Check | Result |
|---|---|
| Base log | 34 inputs, names and order equal to `w5_e10_base_s951` (and to s952-956) |
| nx3 log | 37 inputs = the 34 + `nx_rp3_z`, `nx_rp3_rk`, `has:nx_rp3_z` at indices 34-36 |
| nx3 against base, logged arguments | differ in `variant` and `xt_add` only |
| Base against the stored control | differs in variant, seed and the five smoke keys; the 13 keys that exist only in the `nn_x/abl` harness are all at their off value |
| Checkpoint cfg | base equals `w5_e10_base_s951` on every key; nx3 differs in `xt_names` (37) and `n_dense` (107 against 104) only; `xt_rel`, `xt_idcols`, `xt_relonly` unchanged |
| Tensors | only `f_mean`, `f_std`, `c_mlp.0.weight`, `wide.weight` change shape; 777 more values |
| Statistics | nx3 equals base bitwise on the first 94 dense inputs; the new inputs sit at dense indices 94-96 with mean / sd -0.0031 / 0.637, 0.000 / 0.141, 0.457 / 0.498; the base smoke equals the stored `abl_smoke_kv64_s23` bitwise on all 104 |
| Predictions | val, test and holdout written for both arms, all finite |

[`out/v04_wiring.txt` 1, 3-5]. The files as the loader reads them give z sd 0.640, rank sd 0.142, has mean 0.467 over all train rows [`out/v05_parity_align.txt` B].

### 5.3 Who reads the columns afterwards, and how I know

| Stage | Reads the columns from | Evidence |
|---|---|---|
| Trainer: val, test and holdout predictions | the run's `xt_names`, attached to train, val and test | `nn_x/abl/train.py` l.362-365, 616-617; the nx3 smoke wrote all three |
| Fine-tune | `ck["cfg"]["xt_names"]` -> `attach_xt` on val and test, after `honour_xt_exclude` recomputes the list from the saved `--xt` / `--xt_add` and refuses a mismatch | `nn_x/abl/popfree/finetune.py` l.119-159, 194-201. I emulated that check on CPU on the nx3 smoke checkpoint: names equal, (xt_rel, xt_idcols, xt_relonly, n_dense) = ([0, 3, 30, 31, 32], [0..7], [30, 31, 32], 107) equal; every cfg input resolves in `nn_x/abl/feat_cache/<split>` [`out/v07_ftcheck_r2.txt`] |
| Holdout evaluation | `cfg["xt_names"]` -> `tr.attach_xt` | `nn_x/abl/holdout_eval.py` l.37-42 |
| Stale-val replay | `cfg["xt_names"]`; files resolved through the patched `feat_path` | `nn_x/abl/stale_pred.py` l.37-47, 71-85. Its table `OVR` has no NX name: without the write-new copy the NX columns stay at their natural val values |

### 5.4 Nothing existing was modified

Files with mtime or ctime after 03:45 UTC [`out/v08_untouched.txt`]:

- `data/feats` and `nn/feat_cache`: only the six new `nx_rp3_*` files, link count 2, same inode in both directories. Their md5 equal the builder's table. The 34 input files of the controls (three splits) were last written 2026-11-11 17:33 (newest ctime 17:37), as the critic recorded at 03:38Z; the newest other pre-existing file in `nn/feat_cache` is of 2026-11-14 00:16.
- `nn_x/abl`: only the `abl_smoke_{base,nx3}_s23` outputs, plus two lines appended to `RUNNING.txt` by the launcher.
- Kernel of record, `hz2`, `feats_v2`, `src`, `subs`, `eda`, `data/flat`: nothing. `c09_build.py` md5 `0975dfd6...`, `hz2_lag.npy` md5 `d9f5989f...`, as the builder states.
- No `nx_p3a_*` or `nx_bt_*` file exists anywhere. `nn/feat_cache` now holds 58 / 104 / 104 files (56 / 102 / 102 before).
- No process of the builder is left. The ledger was last written at 04:49:53Z by others (A7, round 2), not by the builder.

---

## 6. (e) The STEP 0 gate table, re-derived on the installed files

Non-zero indicator = my own walk > 0. G-info uses the lens of record (`lens.py`, `condlib.py`, imported unchanged; md5 as in the builder's log); instrument control: the stored fresh column on all 30,000 holdout feeds reads +8.20 / +9.75, the value on record.

| Gate | Bar | Mine | Builder | |
|---|---|---|---|---|
| C3 pool rows, test-like against private: dZ / largest bucket | 0.05 / 0.10 | 0.0295 / 0.0499 | 0.0295 / 0.0499 | ok |
| C3 pool rows, test-like against public | 0.05 / 0.10 | 0.0270 / 0.0527 | 0.0270 / 0.0527 | ok |
| C3 pool rows, val-like against val | 0.05 / 0.10 | 0.0316 / 0.0511 | 0.0316 / 0.0511 | ok |
| C3 holdout feeds (comparison): private; public; val | no bar | 0.0366 / 0.0664; 0.0382 / 0.0693; 0.0416 / 0.0674 | same | |
| **C3 all has = 1 feeds (comparison, mine)**: private; public; val | no bar | 0.0489 / **0.1257**; 0.0421 / 0.0974; **0.0593 / 0.1152** | not computed | flag |
| C4 `nx_rp3_z`, pool has = 1 rows against val / public / private | 0.25 | 0.0363 / 0.0100 / 0.0129 | same | ok |
| C4 `nx_rp3_rk`, pool has = 1 rows | 0.25 | 0.0914 / 0.0293 / 0.0165 | same | ok |
| C4 holdout has = 1 rows (comparison): z; rank | no bar | 0.0851 / 0.0261 / 0.0130; 0.1802 / 0.0820 / 0.0559 | same | |
| **C4 all has = 1 rows (comparison)**: z; rank | no bar | 0.1267 / **0.2603 / 0.3462**; 0.0410 / 0.1264 / 0.1732 | same | flag |
| C5 lowest non-zero share of a test day | >= 0.1045 | 0.1713 | 0.1713 | ok |
| **G-parity (pool reading)** | | **PASS** | PASS | |
| C7 holdout lag view (15,225 feeds): cz z file / rank file | + and >= 2.5 | +4.98 / +4.32 | +4.98 / +4.32 | ok |
| C7 val, stale-val twin, train-only pool | same | +6.46 / +4.78 | +6.46 / +4.78 | ok |
| C7 val, stale-val twin, file FD16P20 | same | +6.51 / +4.79 | +6.51 / +4.79 | ok |
| C7 read as "forms of each column" (BUILD_SPEC usage example): smallest of the twelve values (3 witnesses x 2 files x 2 forms) | same | +4.24 | not computed | ok |
| C6 20 permutations, largest: holdout / twin | < 3.0 | 1.78 / 1.63 | 1.78 / 1.63 | ok |
| **G-info** | | **PASS** | PASS | |
| J1 P(walk = 0 given `seq_cv_invd20` > 0): holdout Sep 12-14 / 17-19; val / public / private | no bar | 0.1223 / 0.2280; 0.0004 / 0.0005 / 0.0004 | same | |
| J2 coefficient of `nx_rp3_z`: holdout lag view; val natural; twin | no bar | 0.0297; 0.0554; 0.0501 | same | |

[`out/v05_parity_align.txt` E; `out/v06_ginfo.txt`]. G-code is covered more strongly by section 2 than by C1 / C2: the builder's C1 and C2 claims are confirmed (`out/v10_c1.txt`, `out/v09_extra.txt` 3). The stale-val twin files used by C7 were recomputed in full and are equal (section 2.1).

---

## 7. Errors and deviations found

Builder's work:

| # | Item | Weight |
|---|---|---|
| 1 | C3 / C4 population: pool feeds (BUILD_SPEC) pass; all has = 1 rows fail C4 on the z file **and C3 on the val-like dZ and two buckets**. The builder reported C4 only and did not compute C3 on all rows | lead's ruling |
| 2 | Lag view fresher than val and private rows in distribution; sparse trickle 2-3 times denser than on val and public rows (section 3.2). Not in the builder's read, and the ledger's "never fresher than on val" is contradicted | record as a limit |
| 3 | Smoke before the STEP 0 read was in the ledger (disclosed by the builder). The report also prints the 73-step NDCG of both arms | procedure |
| 4 | The report says "val and test values come out 1.4 times wider than train values". On train rows that have the view the sd is 0.936, against 0.904 (val) and 0.880 (test): the widening is the same in every split. What it causes is not stated: the model clamps standardised inputs at 6, so `nx_rp3_z` above 3.82 saturates on 1.36% of has = 1 train rows, 1.70% of val rows, 1.77% of test rows, which is 6-10% of the non-zero rows. The rank file is not affected [`out/v09_extra.txt` 5] | small, design limit |
| 5 | The builder's "independent numpy re-derivation" shares the index object (member list and posting lists) with the kernel, so it does not check the member selection independently. Mine does, and agrees | small |
| 6 | "The single row was not traced": the val row non-zero only in the stored `c09_val` is flat row 574,427; the candidate has 1,216 admitted occurrences, so the 500-occurrence cap moved the window, as the builder guessed [`out/v09_extra.txt` 4] | closed |
| 7 | The gates were read on the staged files, not the installed ones. The six installed files have the md5 of the staged files; my reads are on the installed files | closed |
| 8 | The stage-B launcher was never smoke-run and its header still says "NOT TO BE RUN before stage A is read as PASS", which A7.1 (04:48Z) replaces. The lane copy for the 16 all-data students does not exist, so nothing of it is verified | before A7 runs |
| 9 | Not done, as the builder says: the amended read tool (A3 ORDER, VALUE, has = 0 report), the write-new copies of `stale_pred.py` / `holdout_eval.py`. The twins are in `nn_x/w12/nx3/staleval/`, not in `nn_x/nx/staleval/` | before the read |
| 10 | The stored controls were trained 11-12 in `nn_x/w5`; `nn_x/abl/train.py` was last changed 11-14 07:25. Only the rule's harness check (2 fresh against 6 stored) covers that | by the rule |

I found no error in a number of the builder's report.

My own slips, all corrected and kept on disk:

- `v07_ftcheck.py` first took only the first term of `N_DENSE` and printed a false mismatch [`out/v07_ftcheck.txt`]; the corrected read is `out/v07_ftcheck_r2.txt`.
- `v02_compare.py` prints "installed equals this variant where it differs from V0" on 5 / 2 / 3 rows (test / twin / train sample). They sit at the edge of the 2e-5 tolerance; with a clear margin the count is 0 [`out/v09_extra.txt` 1].
- `v09_extra.py` labels two "zero-pattern mismatch" rows with "(the rank file separates it)". That is wrong: both are the minimum row of a feed that has no zero row at all. The values themselves match.
- The first run of the stage-B flag comparison crashed on a parsing error and left no output; `out/v04b_wiring_kd.txt` is the second run.
- C3 on all has = 1 feeds is a comparison I added. It decides nothing under the pool reading.

---

## 8. Everything run

| Time (UTC) | Script | Output |
|---|---|---|
| 04:28-04:30 | `v01_walk.py train / natural / twin` (3 processes) | `out/v01_*.log`, `out/v01_*.npz` |
| 04:30 | `v02_compare.py` | `out/v02_compare.txt` |
| 04:31-04:44 | `v03_trainfull.py 0..2 3` (3 processes) | `out/v03_trainfull_*` |
| 04:33 | `v04_wiring.py`; later the stage-B comparison | `out/v04_wiring.txt`, `out/v04b_wiring_kd.txt` |
| 04:44-04:46 | `v05_parity_align.py`, `v06_ginfo.py` | `out/v05_parity_align.txt`, `out/v06_ginfo.txt` |
| 04:47-04:48 | `v07_ftcheck.py` (two runs) | `out/v07_ftcheck.txt`, `out/v07_ftcheck_r2.txt` |
| 04:50-04:52 | `v09_extra.py`, `v10_c1.py`, file scan | `out/v09_extra.txt`, `out/v10_c1.txt`, `out/v08_untouched.txt` |

At most three 12-thread CPU processes at once. `verify/` holds 7.3 GB (my recomputed arrays). Nothing was written to the ledger, to `reports/CURRENT_STATUS.md`, to the memory directory or to `subs/`.

---

## 9. Text for the lead to record

```
=== <UTC at copy> nx3 STEP 0 VERIFIED (independent verifier; report reports/INPUTS_1116/nx3_verify.md; outputs nn_x/w12/nx3/verify/out/) ===
 VERDICT: GO. Own code (sequence-centric scan, no posting list, nothing of the builder or of c09_build.py imported), label-free inputs only.
 LEAK   0 rows of any split need an inadmissible source: installed files reproduced on ALL 246,745,836 train rows (115,150,397 has = 1 rows, 15,225 holdout feeds
        included; NaN on all 131,595,439 rows of Sep 6-11), all 1,389,971 val rows, all 2,904,806 test rows, all stale-val twin rows (max |dz| 2.4e-07).
        Admitted pairs audited: 1.40e9 train / 1.7e7 val / 3.3e7 test, every violation count 0. The check can fail: own user admitted changes 25,090 train rows,
        a sparse source at or after the row's time 2,843,096, user 0 linked 85,972. My code reproduces the stored c09_hold.npz with max abs diff 0.
 ALIGN  files have the row counts of data/flat; equality above is by flat row index; 0 of 46,140,175 item-absent train rows off the feed minimum
        (28,168,151 on a 1-row shift).
 WIRING stage-A and stage-B launcher flags = designer's text = logged arguments of w5_e10_base_s951 / w5_e10_kd05w192_s951; smoke logs 34 and 34 + 3;
        fine-tune input check emulated on the nx3 smoke checkpoint passes; only new files and two RUNNING.txt lines since 03:45Z.
 GATES  reproduced to the digit: C3 pool 0.0295 / 0.0270 / 0.0316 (buckets 0.050-0.053); C4 pool z 0.036 / 0.010 / 0.013, rank 0.091 / 0.029 / 0.017; C5 0.1713;
        C7 +4.98 / +4.32 (holdout), +6.46 / +4.78 and +6.51 / +4.79 (twin); C6 1.78 / 1.63.
 FOR THE LEAD'S RULING: the GO is on the BUILD_SPEC (pool) reading. On ALL has = 1 train rows C4 z reads 0.127 / 0.260 / 0.346 and C3 reads val-like dZ 0.059,
        largest bucket 0.115 (val-like) / 0.126 (test-like vs private): an all-rows reading is "FAIL G-parity only -> NOT TESTABLE AT PARITY".
 LIMIT J3 (no bar; add next to J1, J2): the lag view is fresher than its reference rows. Median age of the newest contributing source, pool rows: 3.9 d (Sep 12-14)
        against 5.6 d on val; 8.5 d (Sep 17-19) against 8.8 d public / 10.9 d private. Sparse member is the newest source on 5.1% / 9.9% of non-zero pool rows
        against 1.7% (val) / 4.5% (public) / 7.7% (private). Source younger than 2 d: 11.2% of val-like pool rows against 1.4% of val rows. "Never fresher than
        on val" (rule text, holdout clause) does not hold in distribution.
 LIMIT J4 (no bar): the model clamps standardised inputs at 6; nx_rp3_z above 3.82 saturates on 1.4% of has = 1 train rows, 1.7% of val, 1.8% of test rows.
 OPEN BEFORE THE READ: amended read tool (A3), stale_pred / holdout_eval copies with the twins of nn_x/w12/nx3/staleval/; stage-B launcher never smoke-run;
        all-data lane copy not written. The smoke of 04:11Z preceded this entry.
```
