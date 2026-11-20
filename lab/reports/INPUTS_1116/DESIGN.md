# DESIGN: one test of new information as TRAINED inputs of the ranker

Written 2026-11-16 03:11 UTC by the designer of workflow a workflow run (ledger block 2026-11-16T01:34:55Z). CPU only, read-only on the project; every file of mine is in `reports/INPUTS_1116/design/` (section 9). Nothing was built, trained or submitted. The four inventories were read in full: `reports/INPUTS_1116/{model_inputs,residual_rows,catboost_blocks,raw_fields}.md` (below: A, B, C, D).

Tags: **[M]** measured by me (script and output named; outputs are in `reports/INPUTS_1116/design/out/`), **[R]** record on disk (path), **[I]** judgement. "±" = 1 SE paired per feed. "cz" = the calibrated score statistic of inventory B (score test given pool + 84 of the ranker's own input columns; N(0,1) under "nothing new"; placebo sd 0.86-1.06 in my runs).

Contents: 0 answer; 1 the bundle; 2 the arm; 3 the rule and its power; 4 the prior; 5 schedule; 6 everything rejected; 7 what I measured; 8 limits; 9 files.

---

## 0. The answer

**0.1 One bundle, seven train-stage inputs, two families. No second bundle.**

| Family | Inputs (appended to the 34 by `--xt_add`) | Information | Support on record (cz val train-only / val file / holdout) | Train-row build |
|---|---|---|---|---|
| W: whole-sequence co-visitation walk | `nx_rp3_z`, `nx_rp3_rk`, `nx_p3a_z`, `has:nx_rp3_z` | collaborative: a degree-normalised two-step walk candidate -> other users' whole histories -> the feed's history items; label-free | +6.0 / +6.0 / +10.0, the strongest of 305 columns [R B 4.1]; reproduced by my run of the same lens: +5.84..+6.01 / +5.85..+6.01 / +7.89..+10.02 [M `p05_cond_DE.txt`] | lag view (1.2) |
| T: click strength on exploration feeds | `nx_bt_z`, `nx_bt_rk`, `nx_bt_seen` | train labels of other feeds (first-choice Bradley-Terry strength of the item on random-like feeds) | +5.4 / +5.3 / +6.9; independent of W (+5.4 after W is in the null model) [R B 4.1, 4.4] | the same lag view; enters only if its own gate passes |

Everything else of the four short lists is rejected (section 6). Two rejections are new measurements of mine, because inventory D's candidates had been read only with the plain likelihood ratio that inventory B showed to be unspecific: read with B's calibrated lens, the **f1 / f2 equality flags give cz -1.2..-1.9 on val** (holdout -1.5..-2.1; only `f2_home` reaches -3.3..-3.5, on the holdout alone), and the **same-user earlier-feed response gives +2.0..+2.2 on val and +1.2..+1.9 on the holdout** [M `p05_cond_DE.txt`]. Neither passes the bar that 25 other columns pass (|cz| >= 2.5 on both witnesses). So the "second, different kind of information" (the user link) does not deserve an arm, and no fine-tune-extras bundle is designed.

**0.2 Parity is shown by construction for the lag, by measurement for the rest.** A train row reads only sources older than its cut; the cut is 1-3 days back on Sep 12-14 (val rows: 2 and 4 days) and 6-8 days back on Sep 17-19 (test rows: 6-9 days), from 5-7 dense source days (val / test: 12). Sep 6-11 rows have no admissible source and carry a missing flag (46.7% of train feeds have the view) [M `p06_lagview_counts.txt`]. Exact parity of lag AND depth cannot be built from 14 days of data for any cross-feed input (lag + depth needs 18-23 days of past [R `reports/STALE_1115/hz2.md` section 2]); the entered forms are within-feed rank / z of a degree-normalised walk, which thinning of the source leaves unchanged in expectation, and coverage is checked label-free before any GPU hour.

**0.3 A CPU gate decides whether there is an arm at all (STEP 0, about 6 h, no GPU).** Build, code checks against the stored arrays, label-free coverage and PSI gates, and the calibrated lens on the LAG-BUILT holdout rows and on a stale-val twin. If the walk does not keep |cz| >= 2.5 at the test lag on both, nothing is trained and the family is closed. This is the one unknown the inventories left open (B: "no stale-val read; needs lagged builds").

**0.4 The arm: hard labels first, 8 with + 8 without, then students.** The records support the lead's prior, with two amendments [M]: (i) seed pairing buys nothing (single-pair SE same seed 0.00106, other seed 0.00105 on val), so 6 of the 8 "without" members are the stored `w5_e10_base_s951-956` and only 2 fresh controls are run, which double as the harness check; (ii) the decision statistic is the member-level difference, whose SE is lower than the pool-level one (0.00030 against 0.00047 at 8 + 8 students). 10 hard-label runs (9.5-30 GPU process-hours), on a pass 8 students with the old teacher (10-35 h), on a second pass 16 all-data students and a file.

**0.5 What the test can show.**

| True member-level effect on val | P(PASS at stage A), central case | Source |
|---|---|---|
| 0 | <= 0.017 | [M `p04_rule_power.txt`] |
| +0.0003 (what the inventories expect) | 0.12-0.16 | |
| +0.0005 | 0.28-0.37 | |
| +0.0010 | 0.77-0.86 | |
| +0.0020 | 0.99-1.00 | |

Smallest effect shown with power 0.8: **+0.0009 to +0.0011 at member level.** The test answers "is there a missing input worth +0.001 or more"; it cannot read the +0.0003 that the information on record predicts, and no feasible number of seeds can (SE 0.0001 would need about 100 members per arm).

**0.6 Prior [I].** P(the walk passes STEP 0) 0.6; P(PASS at stage A) 0.12; **P(bundle worth >= +0.001 at member level) 0.05; P(>= +0.0005 at file level on private) 0.03.** Central member-level effect +0.0003. Nothing in the four inventories is of the size of the 0.0032 gap, and this design does not change that.

**0.7 Schedule.** T0 = rule in the ledger. STEP 0 read at T0 + 6 h. Stage A read at T0 + 10..12 h. Stage B read at T0 + 15..17 h. File at T0 + 42 h. Deadline 2026-12-08 20:59 UTC.

---

## 1. The bundle

### 1.1 Why these two families and no other

Selection rule I applied [I]: (a) two-witness support under a lens with measured specificity (B's cz, not a plain likelihood ratio); (b) not derivable from what the ranker reads (A 1.9; B 4.6); (c) stable over the two val days and at the private-mix weights; (d) a train-row build whose lag can be made test-like. Applied to every candidate of the four lists:

| Candidate | (a) cz val / holdout | (b) R2 with pool + inputs | (c) val day 2 d -> 4 d; private weights | (d) lag-buildable | Verdict |
|---|---|---|---|---|---|
| RP3beta / P3 walk | +6.0 / +10.0 | 0.41-0.46 | +3.8 -> +4.5; +3.4 | yes: sources are whole history snapshots; forms invariant to source thinning | **IN** |
| Exploration-feed click strength `bt_theta` | +5.4 / +6.9 | 0.00 | +3.0 -> +2.5; +3.3 | yes, same cuts; coverage to be gated | **IN, gated** |
| Density pair `kn_cos1/20` | -4.0 / -6.7 | 0.25 | -1.8 -> -3.9; **-1.3** | no: the index of much-shown items has a size proportional to depth | out |
| Surface split `X3`, `n_s2` | +4.5 / +3.3 | 0.48-0.65 | stable; but at a 7.5-9.5 d lag the twin statistic falls +5.5 -> +3.3 (z form), +4.6 -> +2.4 (rank form) [M `p05`] | yes | out (reserve) |
| First-seen gap, `never_seen` | -5.6 / -3.3 | 0.20 | stable | no: train rows look back 0-13 d, test rows 7-23 d (raw PSI 6-9) | out |
| All-feed item click rate | +3.5 / +6.1 | 0.02 | **+3.6 -> +1.3** | yes, but it is the decaying form of T | out |
| Same-user earlier feeds (D no. 1) | **+2.1 / +1.2..+1.9** [M `p05`] | 0.02 | - | no: 4.6% of train feeds keep a source at a >= 7 d lag against 24-26% of test rows [R D section 5] | out |
| f1 / f2 equality flags (D no. 2) | **-1.2..-1.9 / -1.5..-2.1** (`f2_home` -3.3..-3.5 on the holdout only) [M `p05`] | 0.19-0.51 | - | exact | out |
| Day-causal ALS | +5.1, val only | - | - | 4-8 h new build; as built transductive | out (reserve) |

Sources of columns (a)-(c): [R B 4.1, 4.2, 4.4] unless marked. The two new rows are explained in section 7.3.

Why T although "item click rate" was closed as a trained input (pop2 view) [I from R]: the closed column was the all-feed rate, fresh on train rows; it loses two thirds of its conditional score between the two val days. The exploration-feed strength keeps +3.0 -> +2.5, reads R2 0.00 with pool + inputs (the ranker has no label statistic at all, A 1.9), and here it is built at the lag. It is the second-strongest independent piece and costs 10 CPU-minutes more. Its risk is coverage parity; that is why it has its own gate and why the bundle has a 4-input fallback.

### 1.2 The lag view (train rows, holdout rows included)

Per train feed f: L_f = the stored test-lag draw `nn_x/w12/stale/hz2/feats_L/train/hz2_lag.npy` (6 / 7 / 8 / 9 days for 21.2 / 25.8 / 26.5 / 26.6% of feeds, the test distribution; the holdout feeds keep the lag every lagged-holdout tool gave them). slot = MSK day - Sep 6. **cut' = max(slot - L, 5); has = 1 iff slot >= 6.** A has = 1 feed reads only sources dated before the cut.

| Train day | Feeds | Holdout feeds | has | Lag to the cut | Dense source days |
|---|---|---|---|---|---|
| Sep 6-11 | 1,681,143 | 14,775 | 0 | - | - |
| Sep 12 / 13 / 14 | 203,548 / 263,115 / 244,596 | 1,801 / 2,499 / 2,646 | 1 | 1 / 2 / 3 d | 5 |
| Sep 17 | 269,029 | 2,903 | 1 | 6 d | 5 |
| Sep 18 | 256,678 | 2,718 | 1 | 6 d (21%), 7 d (79%) | 6 / 5 |
| Sep 19 | 233,023 | 2,658 | 1 | 6 / 7 / 8 d (21 / 26 / 53%) | 7 / 6 / 5 |

[M `p06_lagview_counts.py` -> `out/p06_lagview_counts.txt`]. has = 1 on 1,470,065 feeds (46.7%), 15,225 of the 30,000 holdout feeds; val-like regime 22.6% of train feeds, test-like 24.1%.

Why this rule and not the two alternatives [M `p03_walk_coverage.txt`, `p06`; I]:
- The adopted convention of `seq_cv_*` (sources of all days before the row's day): lag 1 d on train rows against 6-9 d on test rows. That is the defect the standing rule names; rejected.
- Pure lag (cut = slot - L, no floor): only 31% of feeds have any source, most from 1-4 dense days. The stored walk is non-zero on 8.6 / 12.9 / 14.5 / 16.4% of rows at 1 / 2 / 3 / 4 dense days against 18.1% at 5 and 20.8% at 11 [M `p03`]: coverage would be half of a test row's on a third of the mass. Rejected.
- The floor of 5 days is the project's own lag-first value (hz2L, m = 5 [R hz2.md 3.2]); with it coverage is 0.87 of the 11-day value [M `p03`: 18.1 / 20.8], the cuts take three values only (5, 6, 7), and half of the has = 1 rows are built like val rows, half like test rows.

Stated defects of the view: lag 9 is not represented and lag 8 is under-represented (test-like rows: 49 / 35 / 16% at 6 / 7 / 8 d, mean 6.7 d; test feeds: mean 7.6 d) [M `p06`]; the missing flag is 0 on 53% of train feeds and never on val / test (precedent in the shipped model: `has:pre2_fe`, 0 on 27.5% of train feeds [R A 1.3]).

### 1.3 The seven inputs

Full recipe: `reports/INPUTS_1116/design/BUILD_SPEC.txt`. Summary:

| Input | Definition | Train / holdout rows | Val / test rows | Stale-val twin (witness only) |
|---|---|---|---|---|
| `nx_rp3_z` | within-feed z of log1p(K x rp3b05), zeros are values, clip ±5. rp3b05(c) = [sum over the latest 500 source sequences holding c, over history items x in them, of 0.95^hpos(x) / (len(s) x deg(x))] / deg(c)^0.5 (kernel `rp3` of `nn_x/insights/ccb_1113/scratch/tester3/c09_build.py`) | lag view: dense sequences of slots < cut' + the 1/37 sparse sample of later train feeds with an earlier timestamp (`hz2_common.sparse_flag`); NaN where has = 0 | all train sequences + val / test feeds with an earlier timestamp (the convention of the adopted `seq_cv_*`) | cut = slot - L, L by `hz2_common.assign_lags`: 7-10 dense days [M `p06`] |
| `nx_rp3_rk` | within-feed mid-rank percentile of rp3b05 minus 0.5 | same | same | same |
| `nx_p3a_z` | as `nx_rp3_z` without the deg(c) penalty | same | same | same |
| `has:nx_rp3_z` | 1 where the lag view exists (derived by the loader from the NaN pattern) | 1 on Sep 12-19 | always 1 | 1 |
| `nx_bt_z` | within-feed z of theta; theta = Bradley-Terry strength, lambda 3, fitted on train feeds with p_random > 0.5 and >= 1 positive, the 30,000 holdout feeds never a source (`fit_bt` of `c06_build.py`); unseen item theta = 0 | the fit on source feeds of slots < cut' (3 fits) | the all-train fit | the fit on slots < slot - L (6 fits) |
| `nx_bt_rk` | within-feed mid-rank percentile of theta minus 0.5 | same | same | same |
| `nx_bt_seen` | 1 if the item occurs in that fit's source | same | same | same |

Files: float32, flat candidate order, `data/feats/<split>/` + hard link in `nn/feat_cache/<split>/`, exclusive create (precedent `nn_x/abl/pop1/build_xp_imp_c50.py`). Names end in `_z` / `_rk` so that `attach_xt` applies clip ±5 / no transform; `nx_bt_seen` is 0 / 1 [R A 2.3; M code read `nn_x/abl/data.py` l.182-198]. Seven columns cost 3.2 GiB of GPU memory per run [R A 2.3].

**Parity argument.**
1. Lag: by the rule of 1.2. A has = 1 row is never fresher than a val row, and half of them are as stale as a test row.
2. Depth (5-7 dense days against 12): thinning the source sequences by a factor q leaves p3a unchanged in expectation (numerator and deg(x) both scale with q) and multiplies rp3b05 by q^-0.5, the same factor for every candidate of the feed; the entered forms are within-feed z and rank. What depth changes is coverage: 0.87 at 5 days [M `p03`]. theta is not scale-free (shrinkage grows with fewer comparisons), so T enters only as within-feed z / rank plus the seen flag, and its coverage gate is its own.
3. Sparse refresh: test rows also see the few val / test histories dated before them; train rows get the same density from the shared 1/37 sample (the hz2 construction, which reproduces the stored val and test columns of `seq_cv_*` exactly [R hz2.md 3.1]).
4. What is not shown and is therefore gated: coverage by listing age against private, public and val rows; the distribution of each file (PSI); the share of non-default rows by test day.

**Leak argument.** W reads no label. Its sources are history snapshots of OTHER users dated before the cut (train rows) or before the row (val / test); every sequence of the row's own non-zero user is skipped, so no later snapshot of the same user can enter; user 0 is never linked (each user-0 feed is one anonymous sequence, and the row's own feed is dated after its cut). T reads train labels only, of feeds dated before the cut, never the row's own feed, never a holdout feed, never a val label; for val / test rows it is the all-train fit. Earlier exploration feeds of the same user are sources like anyone's (the same (user, item) pair recurs on 0.18% of rows [R D 3.1 V2a]).

**Checks (bars are in the rule, STEP 0).** C1 the kernels with cut = own day reproduce the stored `c09_hold.npz` / `c06_hold.npz`; C2 causality counts on every has = 1 row; C3 coverage by seven listing-age buckets, test-like train rows against private and public rows, val-like against val rows (hz2's dZ and bucket statistics); C4 PSI of the six files, train against val / public / private; C5 non-default share by test day; C6 placebo permutations; C7 the calibrated lens on lag-built holdout rows, on the stale-val twin and on natural val rows.

**CPU cost** [R build logs quoted in B 5.1; I for scaling]: W about 1 h of compute (560 s for val + 30,000 holdout feeds with 14 indexes on record; here 1.47M feeds with 3 indexes); T about 10 min (190 s for 14 fits on record); checks 1 h; writing the builder 2-3 h. Two 12-thread processes at most.

---

## 2. The arm

### 2.1 Do the records support "hard labels first, 8 + 8, then students"? Yes [M unless marked]

| Question | Finding | Source |
|---|---|---|
| Does pairing by seed reduce noise? | No. Single-pair SE, same seed / other seed: 0.00106 / 0.00105 val, 0.00062 / 0.00062 holdout (hard labels); 0.00081 / 0.00085 val (students). Stored controls of any seed serve | `p01_power.txt` (c) |
| Noise of a member-level contrast at 8 + 8 | hard labels: val 0.00036, private mix 0.00052, holdout 0.00021. Students: 0.00030 / 0.00043 / 0.00017; 8 new against the 16 stored: x 0.866 = 0.00026 / 0.00037 / 0.00015 [I from M] | `p01` (a) |
| Is the per-feed SE honest about seed-draw noise? | Yes for these families: realised A/A sd of 8-vs-8 splits 0.00026 val / 0.00011 holdout against nominal 0.00030 / 0.00017 (students); hard-label 3-vs-3 0.00048 / 0.00028 against 0.00058 / 0.00034. One exception: the 8 stored pop1 members hold one seed 0.0029 below the mean of the others on val (sd across seeds 0.00130 against 0.00066-0.00076 in the other hard-label families), so the read tool takes the larger of the per-feed and the seed-level SE | `p01` (b), `p02_member_levels.txt` |
| Member level or pool level? | Member level: SE 0.00030 against 0.00047 (val), 0.00017 against 0.00026 (holdout) at 8 + 8 students. Pool deltas are kept as sign clauses | `p01` (b) |
| Real contrasts between different input views | SE runs about 1.17 times the null value (0.00059 against 0.00051 val, 0.00036 against 0.00030 holdout, one added column, 4 + 4) | `p01` (c2) against (a) |
| Hard labels or students first? | Students have less noise and need 8 runs instead of 10, but half of their target comes from a teacher that never saw the input; the attenuation is not measured anywhere [R A 4.2]. Break-even: students have the higher power only if they keep more than 0.72 of the hard-label effect (0.00026 / 0.00036) [I]. A null under distillation closes nothing; a null under hard labels closes the information question at the test's power. Hard-label members are also the regime of all four positive controls (one added column: +0.0020 ± 0.0005 and +0.0015 ± 0.0004 on the holdout at 4 + 4 [R A 3.2]) | |

### 2.2 Commands (write-new copies; no existing file is edited)

All in `reports/INPUTS_1116/design/`. The launchers create new files in `nn_x/abl/{logs,ckpt,preds}` and append one registry line to `nn_x/abl/RUNNING.txt`, as every launcher of that harness does.

| Step | Command | What it is |
|---|---|---|
| Smoke (3-4 min each) | `SMOKE=1 bash launch_nx_hl.sh <gpu> 1 nx` and `... 1 base` | 20,000 feeds, 1 epoch; the log must list 34 + 7 inputs, `n_dense` 111 |
| Stage A, with | `bash launch_nx_hl.sh <gpu> <seed> nx` for seeds 951-958 -> `abl_e10_nx_s<seed>` | `nn_x/abl/train.py`, the flags of `nn_x/abl/pop1/launch_e10.sh` minus its two pop1 flags, plus `--xt_add nx_rp3_z,nx_rp3_rk,nx_p3a_z,has:nx_rp3_z,nx_bt_z,nx_bt_rk,nx_bt_seen`. Verified: training flags identical to `launch_e10.sh` apart from the input flags, and equal to the logged arguments of `w5_e10_base_s951` [M `diff`, args line of `nn_x/w5/logs/w5_e10_base_s951.log`] |
| Stage A, without | stored `w5_e10_base_s951-956`; `bash launch_nx_hl.sh <gpu> <seed> base` for seeds 957, 958 -> `abl_e10_base_s<seed>` | fresh controls in the same harness = harness check |
| Stage A read | `.venv/bin/python reports/INPUTS_1116/design/read_nx.py abl_e10_nx:951-958 w5_e10_base:951-956+abl_e10_base:957-958 <out>` | member and pool deltas on val, private / public mix, M slices, holdout (all, lag-view, val-like, test-like), selected and last step, decision line. Self-test on a stored contrast: section 7.4 |
| Stale-val | write-new copy of `nn_x/abl/stale_pred.py` with six more entries in its override table `OVR` (NX name -> `nn_x/nx/staleval/<name>.npy`); both arms scored | GPU minutes; the tool swaps val inputs through a patched `feat_path` [M code read l.33-47] |
| Stage B | `bash launch_nx_kd.sh <gpu> <seed>` for seeds 951-958 -> `abl_e10_nxkd05w192_s<seed>` | the flags of `nn_x/w12/kd/launch_pop1w.sh` minus `--xt_exclude $POP --xt_add xp_imp_c50`, plus the bundle; `--kd_file teacher_mean11.npy --kd_w 0.5 --d 192` [M `diff`: identical apart from the input flags] |
| Stage B read | `read_nx.py abl_e10_nxkd05w192:951-958 w5_e10_kd05w192:951-958+w5_e10_kd05t25w192:951-958` | reference = the 16 stored wide students |
| All-data students | write-new copy `lane_full_nx.sh` of `nn_x/w12/stale/fulldata/lane_full_v.sh`: two more arms in `kd_flags` (`nxkd05w192`, `nxkd05t25w192`), their guard in stage P, and `XA="--xt_add <bundle>"` without `--xt_exclude` | stage P runs `train_full_abl.py` [M code read l.52-66, 85-100] |
| Fine-tune | `bash nn_x/w12/kd/ft_lane.sh <gpu> abl <variant>` unchanged | see 2.3 |

### 2.3 How the block enters the fine-tune and the prediction scripts [R A 2.4; M where marked]

- The checkpoint cfg carries the extended `xt_names`. The trainer writes train-only val / test predictions and the holdout predictions from it.
- Fine-tune (`nn_x/abl/popfree/finetune.py` through `ft_one_s3_r6.sh`, `--extra n2c_pre_v3` unchanged): `attach_xt` loads the seven columns for val and test from `nn/feat_cache` by the cfg; its input check recomputes the list from `--xt` and `--xt_add`. `--xt_add_relonly` must not be used (it fails that check). No new fine-tune extra is added: the columns are already model inputs.
- Holdout replay, teacher dump: by cfg, no change.
- `read_nx.py` resolves `abl_*` and `w5_*` members through `pc_common.paths` [M self-test].
- The gate's input audit is a name check and passes `nx_*` silently [R A 2.7]: the six files need the construction review of STEP 0 on record instead.

### 2.4 GPU process-hours [R A 2.6, 4.3 for the per-run minutes]

| Stage | Runs | Per run | Total |
|---|---|---|---|
| Smoke | 2 | 3-4 min | 0.1 h |
| A | 10 hard-label d128, 10 epochs | 57-182 min | 9.5-30 h |
| stale-val reads | 16 members | minutes | < 0.5 h |
| B | 8 students d192 | 78-262 min | 10-35 h |
| all-data | 16 students + 16 fine-tunes | 194-256 min (the medians of the two harnesses) + 5-27 min | 53-75 h |
| replicate step (only in the band) | 8 hard-label | as A | 8-24 h |

GPU state at 02:48 UTC [M `nvidia-smi`, read-only query]: 24 compute processes on the the GPUs, utilisation 94-96%, 93 GB free on seven GPUs and [n] GB on one. Whether a GPU qualifies ("< 4 processes, >= 60 GB free") is checked by the launcher at start; with the machine this full the per-run times are the upper ends of the ranges.

---

## 3. The rule

The text to copy into `nn_x/w5/logs/E_kd_predeclared.txt` before the first build is `reports/INPUTS_1116/design/RULE_LEDGER.txt`. Its clauses:

**STEP 0 (CPU).** G-code: stored arrays reproduced, causality counts 0. G-parity: dZ <= 0.05 and largest bucket <= 0.10 for the non-zero share (W) and the seen share (T) against private, public and val rows; PSI <= 0.25 for every file; non-default share on every test day >= half of val's. G-info: |cz| >= 2.5 with the sign on record on the lag-built holdout rows AND on the stale-val twin. W fails -> no arm, family closed. T fails -> the bundle is the four W inputs.

**Stage A (hard labels, 8 + 8; one look).** Member level = difference of member means, SE = max(paired per-feed SE, seed-level SE), selected checkpoints.

| Clause | Witness | Bar |
|---|---|---|
| val | 14,423 feeds | z >= +1.0 |
| holdout | the 15,225 holdout feeds of Sep 12-19 (where the lag view exists) | z >= +1.0 |
| pooled | (z_val + z_holdout) / sqrt(2) | >= 2.0 |
| private mix | `src/pmix.py` weights on val | z >= -1.0 |
| pool level | 8-pool minus 8-pool, val and lag-view holdout | both >= 0 |
| last step | lag-view holdout, member delta | >= 0 |
| stale-val | project scenario with the NX columns swapped to their twins | member delta >= -1 SE |
| harness | 2 fresh controls minus 6 stored | \|z\| <= 2 on val and holdout, else 6 fresh controls replace the stored ones |

PASS = all clauses. REPLICATE = pooled z in [1.6, 2.0) with both z >= 0: 4 more seeds per arm, one re-read of 12 + 12. Anything else: NOT PASSED. The pooled-z form and the 1.6-2 band are the project's standing rule [R `reports/CURRENT_STATUS.md` l.13; `nn_x/w12/kd/arm_read8.py`].

Why the in-period holdout may carry weight in this arm [I]: on holdout rows the new columns are lag-built, so they are never fresher than on val; the contrast is an information contrast, not a popularity-type one. It is never used alone (z_val >= 1 is required), and only on the feeds that have the view.

**Stage B (students, old teacher; only after a pass of A).** Same clauses, 8 new against the 16 stored. A pass is conservative. Not passed is not a closure; it is reported with the ratio B / A.

**After a pass of B.** 16 all-data students (8 per teacher), production fine-tune, file FDNX16P20 = the FD16P20 recipe with its 16 wide members replaced. File rule: gate PASS and the STEP 0 construction review, final val delta against FD16P20 >= +1 SE, private-mix delta >= 0, val M<50 >= -1 SE -> one experimental slot; candidate for pick 2 under the standing selection rule; pick 1 unchanged.

**Teacher regeneration** is not pre-authorised. It needs its own block (a teacher with the bundle against a teacher built the same way without it [R A 4.3 C]) and is opened only if A passes with pooled z >= 3 and B reads below half of A.

**Closure.** A failure of W at STEP 0, or NOT PASSED at A, closes "new train-stage inputs from the residual-gated rows": the two strongest independent families of 305 columns, in the only form whose parity can be shown. The closure excludes a member-level effect of +0.001 or more; it does not exclude +0.0003.

### 3.1 Power [M `p04_rule_power.py` -> `out/p04_rule_power.txt`; SEs from `p01_power.txt`]

Normal approximation, 400,000 draws. Member-level SE at 8 + 8 hard labels: val 0.00036, private mix 0.00052, lag-view holdout 0.00029 (= 0.00021 x sqrt(30,000 / 15,225)); second column of each cell = the same with SEs x 1.17. Null sd of z taken as 1 (the record's A/A sd of the pooled-z rule is 0.80, so the false-pass rates are upper bounds). P(PASS) includes the replicate step.

| True member-level effect on val | Same effect on holdout and private mix | Holdout 0.7x, private 0.5x (central [I]) | Holdout 0.4x, private 0.5x |
|---|---|---|---|
| 0 | 0.017 | 0.017 | 0.017 |
| +0.0003 | 0.21 / 0.16 | 0.16 / 0.12 | 0.11 / 0.09 |
| +0.0005 | 0.50 / 0.39 | 0.37 / 0.28 | 0.24 / 0.18 |
| +0.0010 | 0.94 / 0.88 | 0.86 / 0.77 | 0.61 / 0.52 |
| +0.0020 | 1.00 / 1.00 | 1.00 / 0.99 | 0.95 / 0.89 |

Smallest true effect with power 0.8 at the first read: +0.00078 / +0.00092 (same effect everywhere), **+0.00094 / +0.00110 (central case)**, +0.00142 / +0.00166 (holdout 0.4x). With val alone at 2 SE it would be +0.00102 / +0.00120; with the private mix alone +0.00148 / +0.00173.

Stage B (member-level SE 0.00026 / 0.00037 / 0.00021): an effect kept by the student of +0.0005 / +0.0010 passes with probability 0.61 / 0.97 (0.49 / 0.93 with SEs x 1.17). If the student keeps half of a hard-label effect of +0.0010, it passes with 0.61 / 0.49.

File level [R ledger 2026-11-16T01:20Z, 02:38Z]: contrasts between files with different wide members have SE 0.0002-0.0005 on val; a file-level effect of +0.0005 is at 1-2.5 SE. The file rule can confirm no harm; it cannot prove the gain. The public board cannot either (paired SE 0.00026-0.00052 between files of the family).

---

## 4. The honest prior [I, with the measured inputs named]

| Step | Reasoning | Number |
|---|---|---|
| Information on record, fresh builds | W 0.00175 / 0.00221 nats per feed (val / holdout), T about 0.0005 / 0.0003 (single-column dnats) [R B 4.1, 4.3]: together about 0.0023 nats on val = 1.0 of the encoder block (0.0022, worth +0.0006 ± 0.0008 val as trained inputs) and 0.4 of the 8-input popularity block (0.0058, worth +0.0013..+0.0016) | |
| Converted at the two clean controls' rate (0.23-0.28 NDCG per nat on val [R B 0.7]) | +0.0005..+0.0006 at pool level for fresh inputs | |
| Lens specificity | two of five controls with a large likelihood read convert at about zero (`ctx_hn` + `ctx_hs`; `ctx_nd` on the holdout) [R B 1.2; C section 1: 4 of 9 unresolved cells read >= 2 SE] | x 0.7 |
| Lag view | trained on 47% of feeds from 5-7 source days; lag 6-8 d on half of them; what W keeps at that lag is unknown (it shows no decay from 2 to 4 days) | x 0.6-0.8 |
| Selection | W is the top of 305 columns on the same two witnesses; cz 6 and 10 against a noise maximum near 3, so little shrinkage | x 0.9 |
| Central member-level effect on val, hard labels | | **+0.0003** (range 0 to +0.0010) |
| P(STEP 0 passes for W) | coverage ratio 0.87 expected; lag unknown; hz2's bars are strict (hz2L itself missed one by 0.003) | 0.6 |
| P(PASS at A) | prior over the effect x the power table, x P(STEP 0) | 0.12 |
| **P(>= +0.001 at member level)** | needs the upper tail of the conversion and no loss at the lag | **0.05** (0.03-0.10) |
| **P(>= +0.0005 at file level, private)** | further: students (teacher blind to the input, attenuation 0.5-1 unmeasured); file = 0.8 wide + 0.2 popularity-light, half fine-tuned; W weakens with the cold share of the feed (+4.65 -> +2.07 by tercile [R B 4.2]) and private is colder | **0.03** (0.02-0.05) |

What the model already reads, against the bundle [R A 1.3-1.5]: windowed co-visitation counts (±20 positions, no degree normalisation, non-zero on 11.3% of val rows against 19.8% for the walk [R B 4.5]), five popularity counts, six encoder scores with an item-ID table, no label statistic. The conditional reads do not move when 20 non-linear forms of the collaborative inputs join the null model [R B 4.5]; the network's own non-linearity over all inputs is not in any CPU lens.

Gate sensitivity, for scale: the residual NDCG gates recovered 10-26% of a block's value on the holdout and are unidentified on val [R A 3.3]; "300 rows closed" therefore carried little evidence against inputs. The calibrated lens then finds 25 of 284 columns with two-witness support, six kinds of information, and prices the best bundle at about +0.0005 [R B 0.5-0.7]; C maps its best three columns to about +0.0013 on fresh rows (range +0.0006 to +0.004) and reads them at +0.00006 ± 0.00047 nats at the private mix [R C 0.1]; D puts its two families at +0.0003 [R D 0.4]. Three independent inventories agree on the order of magnitude: **a tenth to a third of the gap on fresh rows, less at the test lag; not the gap.**

---

## 5. Schedule (hours from T0 = rule copied into the ledger)

| Hours | Step | Resource |
|---|---|---|
| 0-3 | builders for W and T written; C1 code checks on the stored holdout arrays | CPU, 1 process |
| 3-4.5 | builds: train (lag view), val, test, stale-val twins | CPU, 2 processes x 12 threads |
| 4.5-6 | C2-C7; STEP 0 read written to the ledger; an independent reviewer re-derives C1 / C2 on a sample | CPU |
| 6-6.5 | two smoke runs, cfg and log check, fine-tune input check on the smoke checkpoint | GPU minutes |
| 6.5-12 | stage A: up to 8 runs at once (one per GPU where it qualifies), the rest queued; stale-val pass; read | 9.5-30 GPU process-hours |
| 12-17 | on PASS: stage B, 8 students; read | 10-35 h |
| 17-42 | on PASS: 16 all-data students, fine-tunes, file, gate, final val | 53-75 h |

A replicate step adds 3-6 h after stage A. The last upload is 2026-12-08 20:59 UTC; the whole chain ends about two days after T0.

---

## 6. Everything rejected, one line each

| Candidate | Inventory | Reason |
|---|---|---|
| f1 at floor, f2 = feed's modal f2, f1 tie-group size | D no. 2 | calibrated lens: cz -1.2..-1.9 val, -1.5..-2.1 holdout; `f2_home` -3.3..-3.5 on the holdout only; R2 with pool + inputs 0.19-0.51 [M `p05`]. The plain LR of 13-19 was the pool's weighting of inputs it already reads |
| Same-user earlier feeds: user x category response (`uc_ctr`, `uc_coec`), click / skip similarity | D no. 1, B G6, C rank 3 | cz +2.0..+2.2 val, +1.2..+1.9 holdout; click similarity +1.5 / +0.1; skip similarity flips sign [M `p05`]; and a train-row build at the test lag covers 4.6% of feeds against 24-26% of test rows [R D section 5] |
| Re-shown item (`tte_ui_imp`, `ua_item_imp`) | B, C, D | passes the lens (-3.0 / -3.1 / -3.1) on 0.18% of rows; source 72 h old on train rows, 357 h on test [R C 0.3]; no pool-level value |
| Density pair `kn_cos1`, `kn_cos20` | B rank 2, C rank 2 | the index of much-shown items cannot be built at the test lag with equal depth; cz -1.3 at the private weights [R B 4.2]; nats -0.00013 ± 0.00028 at the private mix, zero on one val day [R C section 3] |
| Surface split `X3`, `n_s2`; item source-1 share | B rank 3, D no. 3 | one piece of information; D's form of it, rebuilt at a 7.5-9.5 d lag, falls from +5.5 to +3.3 (z form) and from +4.6 to +2.4 (rank form) under the calibrated lens [M `p05`]; reserve, first to add if stage A passes |
| First-seen gap, `never_seen`, `n_days_active`, `w25_it_dsf` | B rank 5, C | train rows look back 0-13 d, test rows 7-23 d; raw PSI 6-9; the cold / warm axis [R B 5.1, C 0.3] |
| All-feed item click rate (`tte_it_ctr / coec / pos`, `xp_ctr_c`, W25 twins), category base rate | C rank 1, B G3 / G7, D no. 4 | cz +3.6 -> +1.3 between the two val days [R B 4.2]; +3.75 -> +2.4 at the test lag [M `p05`]; its exploration-feed form is family T |
| Day-causal ALS | B rank 6 | val-only support; 4-8 h new build; as built transductive; reserve |
| Graph walk of 11-15 | B rank 7 | the same information as W (W reads +0.4..+1.0 after it) [R B 4.4] |
| `cv_bm25`, windowed / typed co-visitation variants, `seq_cos_last1`, `afe_grk_f5_rk`, `ixh_*` | B | R2 0.91 or one witness or below the bar |
| `tte_u_*`, `tte_uc_*`, family B block a | C | fresh on train rows, stale on test; unexplained later-feed lift of `tte_uc_*` [R C 0.3] |
| Family B block b, W25 scorers and crosses, embedding blocks, audit blocks, KNN-on-teacher columns, native CatBoost statistics | C | the ranker reads the information, or the column is derived from the ranker, or it lives inside a tree model [R C 0.3] |
| Hard-label tree score as an input | C rank 4 | no out-of-fold train score; inherits the freshness defect; lens +0.4 / +1.9 [R B 2.2] |
| Non-LUO `it_hist_*`, in-split candidate counts, `it_tr_*` / `u_tr_*`, train-row `pre_*`, "V-exact" counts, f5 echo, hindsight targets, later-using exposure / co-visitation, val-label priors, embedding-change flags, cross-feed consensus | B 5.2 | leak class or no train-row analogue |
| Missing-embedding recovery | D no. 5 | 0.14% recoverable on train rows against 21.6 / 26.3% on val / test; lens null |
| Feed hour, item placement prior, fresh co-exposure, identifier order | D | null |
| A fine-tune-extras bundle (bundle 2) | this design | its only distinct content, the user link, fails the lens; the fine-tuned branch is not above the train-only branch at member level (-0.00039 ± 0.00039, 16 students [M `p01` (d)]), so that stage has no record of turning an input into NDCG |
| W in the adopted `seq_cv_*` convention; W at a pure lag | this design | fresh on train rows; or 31% of feeds at half coverage (1.2) |

---

## 7. What I measured for this design

### 7.1 Noise of with / without contrasts [M `p01_power.py`, `p02_member_levels.py`]

Stored members: hard-label `w5_e10_base` (6), `abl_e10_pop1` (8), `abl_e10_xdrop_pop` (4), `abl_e10_pop2` (4); students `w5_e10_kd05w192` (8), `w5_e10_kd05t25w192` (8).

| Statistic | Val | Val, private mix | Holdout |
|---|---|---|---|
| Null SE of a member-mean contrast, hard labels, k = 4 / 8 / 12 per arm | 0.00051 / 0.00036 / 0.00029 | 0.00072 / 0.00051 / 0.00042 | 0.00029 / 0.00021 / 0.00017 |
| The same, students | 0.00042 / 0.00030 / 0.00024 | 0.00061 / 0.00043 / 0.00035 | 0.00024 / 0.00017 / 0.00014 |
| Students, 120 random 8-vs-8 splits: A/A sd (nominal SE), member level | 0.00026 (0.00030) | 0.00049 (0.00045) | 0.00011 (0.00017) |
| The same, pool level | 0.00046 (0.00047) | 0.00058 (0.00069) | 0.00025 (0.00026) |
| sd of a member's level across seeds: hard-label base / students | 0.00066 / 0.00039-0.00044 | - | 0.00036 / 0.00021-0.00025 |
| Stored one-column contrasts, 4 + 4 hard labels, member level: `xp_imp_c50`; `xp_ctr_c` | +0.00015 ± 0.00059; +0.00069 ± 0.00057 | -0.00004 ± 0.00090; +0.00095 ± 0.00084 | +0.00184 ± 0.00036; +0.00120 ± 0.00034 |

### 7.2 Coverage of the walk against source depth [M `p03_walk_coverage.py`]

Stored holdout array (sources = train days before the row's day): non-zero row share 0 / 8.6 / 12.9 / 14.5 / 16.4 / 18.1 / 18.5 / 19.5 / 20.1 / 18.7 / 20.0 / 20.8% at 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 / 10 / 11 dense days; 93-97% of feeds have at least one non-zero candidate from 1 day on. Val (12 dense days): 20.9% on Sep 22, 18.7% on Sep 24.

### 7.3 Inventory D's candidates under inventory B's lens [M `p05_cond_DE.py`]

D screened with a plain likelihood ratio conditional on the pool score only; B showed that inputs the ranker already reads print such a ratio of up to 48.9 (median 19.5). Neither inventory read D's columns with the calibrated lens. Controls of my run: the walk reproduces B's values to the digit; holdout row order checked against D's arrays (max difference 2e-7); 60 within-feed permutations: sd 1.06 (val), 0.86 (holdout), largest 2.86.

| Column | Val, train-only pool (z / rank form) | Val, file | Holdout | R2 with pool + inputs |
|---|---|---|---|---|
| f1 at floor | -1.42 / -1.90 | -1.18 / -1.62 | -1.48 / -2.14 | 0.33 |
| f2 = feed's modal f2 | -1.26 / -1.49 | -1.18 / -1.39 | -3.26 / -3.51 | 0.51 |
| f1 tie-group size | -1.35 / -1.49 | -1.20 / -1.32 | -1.65 / -1.59 | 0.19 |
| user x category CTR | +2.14 / +2.10 | +2.17 / +2.10 | +1.23 / +1.84 | 0.02 |
| the same, position-debiased | +2.04 / +2.06 | +2.08 / +2.09 | +1.21 / +1.90 | 0.02 |
| the same, built at a 7.5-9.5 d lag | +2.03 / +2.12 | +2.03 / +2.11 | - | 0.02 |
| similarity to earlier clicks / earlier skips | +1.54 / +1.38; +1.08 / +0.99 | +1.47; +0.90 | +0.11 / -0.21; -2.94 / -2.94 | 0.10; 0.12 |
| item re-shown to the user | -3.04 / -2.89 | -3.08 / -2.93 | -3.11 / -2.99 | 0.00 |
| item clicks over expected: fresh; at the test lag | +3.75 / +3.46; +2.40 / +3.29 | +3.78; +2.44 | - | 0.02 |
| item source-1 share: fresh; at the test lag | +5.53 / +4.64; +3.29 / +2.39 | +5.14; +3.09 | - | 0.30 |

Readings: 70-90% of D's likelihood for the equality flags and for the user x category response is explained by the ranker's own inputs (the flags are functions of f1 / f2, which are inputs; the user's category response follows the per-type category counts of the history). The source share is not dead at the test lag under this lens, as D's plain ratio said (13.0 -> 3.0); it loses 40-50%.

### 7.4 Self-test of the read tool [M `read_nx.py` -> `out/read_nx_selftest_pop1_vs_xdrop.txt`]

Stored contrast "one added column `xp_imp_c50`", 8 members against 4: val +0.00064 ± 0.00060 (z +1.07), lag-view holdout feeds +0.00221 ± 0.00046 (z +4.83), pooled z +4.17, private mix +0.00033 ± 0.00082. The form of the rule prints PASS. Two things follow. The tool runs end to end on stored files. And the rule passes a FRESH popularity-type column on the strength of the holdout with val at z 1.07: the holdout clause is only legitimate here because the NX columns are lag-built on holdout rows (section 3). An arm whose new columns were fresh on train rows must not be read with this rule.

---

## 8. Limits

- Nothing is built. Whether W keeps its information at a 6-8 day lag, and whether its coverage matches test rows, is not known; STEP 0 measures both, and I put 0.4 on a failure.
- The lag view is an approximation: 5-7 source days against 12, lag 9 missing, the missing flag on 53% of train feeds. Its parity is argued in expectation and gated on coverage and PSI; it is not exact, and no exact form exists in 14 days of data.
- T is a label statistic of the family whose fresh all-feed form was closed as a trained input; its inclusion rests on a likelihood lens whose specificity for NDCG is poor (section 4).
- Power uses a normal approximation and null SEs from stored members; contrasts between different input views ran 17% above them (both columns are given). The correlations assumed between the val, private-mix and pool-level reads (0.7, 0.8) are mine.
- Stage A is not the shipping recipe; 7 of 8 levers closed under hard labels did not flip under distillation and one did [R A 4.3]. Stage B exists for that reason and is biased toward zero by an unmeasured factor.
- G-info of STEP 0 uses val labels to decide which of two families enters; the stage A val read is not independent of that choice. With two families and a bar of 2.5 on two witnesses the selection effect is small, and it is in the direction of a pass.
- The holdout clause reads selected checkpoints, and both arms select on the same holdout; the last-step clause guards the sign only.
- I did not run a GPU process, so the smoke checks of section 2.2 are untested; the launch flags were compared with their precedents by `diff` only.
- `p05` uses inventory D's stored arrays as they are; I did not rebuild them.

## 9. Files (all in `reports/INPUTS_1116/design/`)

| File | Content |
|---|---|
| `RULE_LEDGER.txt` | the rule block to copy into the ledger before the first build |
| `BUILD_SPEC.txt` | build recipe for the six files, the lag view, the stale-val twins, checks C1-C7 |
| `launch_nx_hl.sh`, `launch_nx_kd.sh` | write-new launch copies for stage A (arms nx, nx4, base; SMOKE=1) and stage B; `bash -n` clean; flags compared with `launch_e10.sh` / `launch_pop1w.sh` |
| `read_nx.py` | the read tool of both stages; self-test output `out/read_nx_selftest_pop1_vs_xdrop.txt` |
| `p01_power.py` -> `out/p01_power.txt` | SEs, A/A spreads, seed pairing, stored contrasts, fine-tune level |
| `p02_member_levels.py` -> `out/p02_member_levels.txt` | member levels by seed |
| `p03_walk_coverage.py` -> `out/p03_walk_coverage.txt` | walk coverage by source depth |
| `p04_rule_power.py` -> `out/p04_rule_power.txt` | power of the rules |
| `p05_cond_DE.py` -> `out/p05_cond_DE.txt` | inventory D's columns under the calibrated lens |
| `p06_lagview_counts.py` -> `out/p06_lagview_counts.txt` | counts of the lag view and of the stale-val twin |

Run form (CPU): `cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=reports/INPUTS_1116/design/out/numba_cache .venv/bin/python reports/INPUTS_1116/design/<script>`. Longest job about 6 minutes; never more than one 12-thread job of mine at a time.
