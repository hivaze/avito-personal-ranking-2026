# CRITIC of the trained-input design (`reports/INPUTS_1116/DESIGN.md`)

Written 2026-11-16 03:15-03:50 UTC. CPU only (12 threads per process, at most two processes of mine at a time), no GPU process, nothing submitted, no web, no process signalled. Read-only on the project; every file of mine is in `reports/INPUTS_1116/critic/` (section 9). Read in full: `DESIGN.md`, `design/RULE_LEDGER.txt`, `design/BUILD_SPEC.txt`, the four inventories (`model_inputs.md` = A, `residual_rows.md` = B, `catboost_blocks.md` = C, `raw_fields.md` = D), the design's outputs `out/p01`..`p06`, its scripts `p04`, `p05`, `p06`, `read_nx.py` and both launchers, and the two build scripts of record (`nn_x/insights/ccb_1113/scratch/tester3/c09_build.py`, `c06_build.py`).

Tags: **[M]** measured by me (script and output named; outputs in `reports/INPUTS_1116/critic/out/`), **[R]** record on disk (path), **[I]** judgement. "±" = 1 SE paired per feed. "cz" = the calibrated score statistic of inventory B (score test given pool + 84 of the ranker's own inputs; N(0,1) when the column is nothing new). "Lag view" = the designer's train-row build: cut' = max(slot − L, 5), sources before the cut, on the feeds of Sep 12-19.

What I did beyond reading: I imported the two kernels of record unchanged and built the designer's columns in the lag view on the 15,225 holdout feeds that get it, the stale-val twin on all val rows, and the natural columns on val and test rows. Then I read the designer's own STEP 0 gates on them. The gate bars were fixed in `design/RULE_LEDGER.txt` (file time 03:05 UTC) before my reads (03:21-03:37 UTC).

---

## 0. The answer

**0.1 Verdict per input.**

| Input | Verdict | Reason (detail in the section named) |
|---|---|---|
| `nx_rp3_z` | **KEEP** | No leak path in the code; the stored arrays are reproduced exactly. Lag and coverage parity are inside the designer's bars on my advance build, and the information survives the lag (cz +5.0 on lag-built holdout rows, +6.6 on the stale-val twin). Two parity defects remain and are stated in 1.4; both act toward under-use |
| `nx_rp3_rk` | **KEEP** (weak) | Same information; given `nx_rp3_z` it reads only +1.1 / +2.2 / +0.5. It has the largest distribution shift of the family (PSI 0.17 train against val, bar 0.25) |
| `nx_p3a_z` | **DROP** | A copy: R2 0.95-0.96 with the two inputs above; cz given them −0.60 / +0.33 / +0.28 |
| `has:nx_rp3_z` | **FIX** | The flag is 1 exactly on Sep 12-19, i.e. it is a period indicator that is always on at val / test. Keep it only with the value-attribution read of 4.5 (permute the values, keep the flag) |
| `nx_bt_z`, `nx_bt_rk`, `nx_bt_seen` | **DROP** | Parity is not demonstrated: the family fails the designer's own coverage gate by a factor of two (dZ 0.095-0.107 against 0.05, largest bucket 0.167-0.185 against 0.10, PSI 0.48 against 0.25). Its lag-built z form reads cz +1.2 on the holdout |

The bundle is therefore **three inputs holding one piece of information**: `nx_rp3_z`, `nx_rp3_rk`, `has:nx_rp3_z`.

**0.2 The designer's main unknown is resolved: the walk passes STEP 0 on my advance build.** [M `c03_gates.txt`]

| Gate of `RULE_LEDGER.txt` | Bar | Walk (W) | Click strength (T) |
|---|---|---|---|
| G-code: stored arrays reproduced | 1e-5 relative / 1e-4 absolute | max abs diff 0 on 1,389,971 val rows and 694,799 holdout rows | theta exact on val; 1.35e-4 on holdout (warm start); seen flag exact |
| G-parity, test-like train rows against private / public rows | dZ ≤ 0.05; bucket ≤ 0.10 | 0.037 / 0.038; 0.067 / 0.070 | **0.095 / 0.098; 0.167 / 0.170** |
| G-parity, val-like train rows against val rows | same | 0.042; 0.068 | **0.107; 0.185** |
| PSI of the entered forms, train against val / public / private | ≤ 0.25 | z form 0.088 / 0.023 / 0.011; rank form 0.171 / 0.068 / 0.045 | z form 0.046 / 0.018 / 0.014; **rank form 0.479** / 0.131 / 0.054 |
| G-info, lag-built holdout rows (z / rank form) | \|cz\| ≥ 2.5 | +5.02 / +4.43 | +1.17 / +3.82 |
| G-info, stale-val twin | \|cz\| ≥ 2.5 | +6.64 / +4.90 | +3.70 / +5.35 |

My build admits sparse members by day instead of by timestamp and counts them in the degree, as `c09_build.py` does; the official builder must repeat the gates on its own files. The margin on the val-like coverage gate is thin (0.042 against 0.05).

**0.3 Verdict on the arm: sound as a test of "is there +0.001", with five fixes.** Hard labels first and 8 + 8 members are supported by the records; the launch flags match the stored controls [M args line of `nn_x/w5/logs/w5_e10_base_s951.log` against `design/launch_nx_hl.sh`].

1. Bundle of three inputs, fixed now; delete the clause that chooses the composition at STEP 0.
2. Add a value-attribution clause to PASS (4.5). Without it the rule can pass on the period flag alone.
3. Read the harness check before the arm's predictions are opened.
4. Replace "CLOSED" by "NOT SHOWN at this power", and never close "new train-stage inputs" as a class on this arm.
5. Give stage B a pool-level clause and terminal branches; as written, a stage B failure leaves the result open to any reading.

Corrected text: `reports/INPUTS_1116/critic/RULE_AMENDMENTS.txt` (also section 7).

**0.4 Verdict on the rule's numbers.**

| Statement of the design | My re-derivation | Source |
|---|---|---|
| False PASS ≤ 0.017 | 0.008-0.020 at the first read by simulation; on stored members 1 of 300 (students, 8 v 8), 4 of 70 (hard-label pop1, 4 v 4), 0 of 20 (stored controls, 3 v 3) | [M `c04_rule_null.txt`] |
| Power 0.77-0.86 at +0.0010 | 0.71-0.80 at the first read | same |
| Power 0.28-0.37 at +0.0005 | 0.24-0.29 | same |
| Power 0.12-0.16 at +0.0003 | 0.09-0.13 | same |

The power statement is right to within 0.05-0.07. My figures are lower because the tool estimates the SE from the members, the seed noise of the two witnesses is correlated (+0.36 over 22 hard-label members), and I included the last-step and stale-val clauses that `p04` left out.

**0.5 My prior [I].** P(the walk passes STEP 0) 0.85. P(PASS at stage A) 0.15. **P(member-level gain ≥ +0.001 on val) 0.05 (0.03-0.08).** P(file-level private gain ≥ +0.0005) 0.02. Central member-level effect +0.0003. The designer's 0.05 stands, for different reasons: the STEP 0 risk is mostly gone, and against that the bundle is one piece of information that carries about half as much likelihood on lag-view train rows as on val rows (0.00087 against 0.0016-0.0017 nats per feed, 1.3) and fits there with half the coefficient (1.4).

**0.6 What is missing** (section 6): no overlooked raw column. Missing is an instrument, not an input: re-scoring the arm's own members with the new values permuted reads how much they take from the values with an SE several times smaller than any between-arm contrast, and it is what turns a null into a statement.

---

## 1. Leak and parity, input by input

### 1.1 Code read and reproduction of the three highest-ranked inputs

| Input | What the code of record does [M code read] | Reproduction on disk [M] |
|---|---|---|
| `nx_rp3_z`, `nx_rp3_rk` (kernel `rp3` of `c09_build.py`) | Sources are history sequences only (no label array is opened). `source_fidx` takes feeds with `day < cutoff` strictly: the latest feed of each non-zero user and every user-0 feed. The kernel skips every sequence of the row's own non-zero user (`if u != 0 and suser[s] == u: continue`) and every dated sequence that is not earlier (`if sday[s] >= fd: continue`) | Natural val build: `rp3b05` and `p3a` equal `cache/c09_val.npz` on all 1,389,971 rows, max abs diff 0, 0 zero-pattern mismatches. Holdout rows of Sep 12 (276,857) and Sep 19 (417,942) with cut = own day: max abs diff 0 against `c09_hold.npz` [`c01_lagwalk.log`] |
| `nx_p3a_z` | Same kernel, second output | Same |
| `nx_bt_*` (`fit_bt` of `c06_build.py`) | Source feeds = `(p_random > 0.5) & ~is_hold` with ≥ 1 positive; holdout rows read `sday < d`. Train labels only | All-train fit on val rows: theta max abs diff 0, 0 seen mismatches against `c06_val.npz`. Fit on slots < 6 for the holdout rows of Sep 12: 1.35e-4 (the record warm-starts), 0 seen mismatches [`c02_lagbt.log`] |

Leak verdict for the lag view [I from the code read]:
- **Same user's later snapshots:** cannot enter. Every source is dated before the cut, the cut is before the row, and the same non-zero user is skipped as well.
- **Own-feed labels:** W reads no label. T reads labels of feeds dated before the cut, never a holdout feed.
- **User 0:** never linked; each user-0 feed is its own sequence.
- **Causality of label statistics on train rows:** T is causal with a lag by construction.
- I found no leak path in either family. The lag draw `hz2_lag.npy` is a hash of `feed_id` (`hz2_common.assign_lags`), unrelated to labels.

### 1.2 The lag view against test rows: coverage [M `c01_lagwalk.log`, `c02_lagbt.log`, `c03_gates.txt`]

Share of rows with a non-default value.

| Rows | Walk non-zero | Click strength seen |
|---|---|---|
| Train, lag view, Sep 12 / 13 / 14 (holdout feeds) | 0.167 / 0.163 / 0.154 | 0.393 / 0.366 / 0.334 |
| Train, lag view, Sep 17 / 18 / 19 | 0.138 / 0.137 / 0.138 | 0.273 / 0.268 / 0.270 |
| The same rows, fresh build of record (cut = own day) | 0.185-0.201; 0.187-0.208 | 0.459-0.488; 0.421-0.525 |
| Val, natural (Sep 22 / 24) | 0.209 / 0.187 | 0.468 / 0.408 |
| Val, stale-val twin | 0.157 | 0.305 |
| Test, natural, by day Sep 26 / 27 / 28 / 29 | 0.180 / 0.175 / 0.172 / 0.171 | 0.377 / 0.361 / 0.346 / 0.338 |
| Test public / private, feeds with M ≥ 50 | 0.179 / 0.172 | 0.373 / 0.347 |

Readings.
- **The design's "coverage 0.87 of the 11-day value" was measured at a 1-day lag.** At the lags the view really has, test-like train rows carry 0.77-0.80 of a test row's coverage and val-like rows 0.81 of a val row's. That is inside the designer's bars for W.
- **T is outside them by a factor of two.** The seen share falls about 3 points per day of lag and the view has 5 source days instead of 12: 0.27 on test-like train rows against 0.35-0.37 on test rows.
- The cap of 500 postings in the kernel is not a depth problem: 1.5% of non-zero rows have more than 500 postings in the natural index, 0.5-0.9% in the lag view [`c01_lagwalk.log`].

### 1.3 Does the information survive the lag? [M `c03_gates.txt`]

| Column | Witness | Fresh / natural build, cz z / rank | Lag view / twin | Kept |
|---|---|---|---|---|
| `rp3b05` raw (B's form) | holdout, 15,225 lag-view feeds | +6.05 / +5.65 | +4.03 / +4.43 | 0.67 / 0.79 |
| entered form log1p(K x) | the same feeds | +6.93 / +5.65 | +5.02 / +4.43 | 0.72 / 0.79 |
| entered form | holdout, Sep 12-14 (6,946 feeds) | +4.89 / +3.10 | +3.51 / +2.81 | 0.72 / 0.91 |
| entered form | holdout, Sep 17-19 (8,279 feeds) | +3.92 / +3.78 | +3.00 / +2.76 | 0.77 / 0.73 |
| entered form | val, train-only pool | +6.84 / +5.97 | +6.64 / +4.90 | 0.97 / 0.82 |
| entered form | val, file FD16P20 | +6.91 / +6.00 | +6.70 / +4.92 | 0.97 / 0.82 |
| theta | holdout, lag-view feeds | +2.84 / +4.67 | +1.17 / +3.82 | 0.41 / 0.82 |
| theta | holdout, Sep 17-19 | +1.61 / +3.03 | +0.45 / +1.87 | 0.28 / 0.62 |
| theta | val | +3.87 / +5.38 | +3.70 / +5.35 | 0.96 / 0.99 |

Check: the fresh `rp3b05` on all 30,000 holdout feeds reads +8.20 / +9.75, the record's value. Placebo, 20 within-feed permutations of the lag-built column: sd 0.89, largest 1.93 (holdout); sd 0.91, largest 1.55 (val twin).

In likelihood the lag view keeps half: 0.00044 nats against 0.00088 for the fresh column on the same holdout feeds (both per feed of all 30,000; per lag-view feed 0.00087 against 0.00173). On val the twin keeps all of it (0.00163 against 0.00168).

### 1.4 Parity defects that the designer's gates do not see [M `c05_coef.txt`, `c06_joint.txt`]

The gates of STEP 0 are marginal (coverage by listing age, PSI). Two joint properties differ between train rows and val / test rows.

| Property | Train rows, lag view (Sep 12-14 / Sep 17-19) | Val natural / test public / test private | Val twin |
|---|---|---|---|
| P(walk = 0 given the windowed co-visitation input `seq_cv_invd20` > 0) | 0.123 / 0.229 | 0.002 / 0.002 / 0.003 | 0.157 |
| P(walk > 0 given `seq_cv_invd20` = 0 and `seq_cv_pop` > 0) | 0.130 / 0.103 | 0.182 / 0.179 / 0.175 | 0.131 |
| One-step coefficient of `nx_rp3_z` given pool + own inputs | 0.0285 / 0.0230 (all lag-view feeds 0.0299) | 0.0556 (val) | 0.0513 |
| The same, rank form | 0.128 / 0.125 | 0.306 | 0.236 |
| R2 of `nx_rp3_z` with pool + own inputs | 0.13 / 0.13 | 0.52 | 0.42 |

- **Joint structure.** On val and test rows the walk and the adopted `seq_cv_*` inputs come from the same source, so a windowed hit implies a walk hit (99.7-99.8%). On lag-view train rows `seq_cv_*` is the adopted fresh build and the walk is 1-8 days older: 12-23% of windowed hits have no walk value. The network is trained on a pattern that does not exist at test.
- **Relation.** The coefficient that fits train rows is 0.54-0.58 of the one that fits val rows. A weight learned at half its best value keeps about three quarters of the attainable likelihood gain [I: quadratic approximation].
- **Direction.** Both defects make the trained model use the walk less than it could on test rows. They lower the expected effect; they do not create a false gain.
- **No fix exists inside the adopted view.** Joint parity would need `seq_cv_*` lag-built on train rows too, which is the hz2 view (closed at file level, [R `reports/CURRENT_STATUS.md` log 2026-11-15T13:10Z as quoted in C section 5]) and would lose the stored controls. I list both as stated limits (amendment A2), not as bars.
- **Lag itself is not the problem for W.** The coefficient is the same at a 1-3 day lag (0.0285) and at 6-8 days (0.0230); the twin keeps the natural value on val. So the val-like half of the view, which is fresher than any test row, does no measurable harm, and I do not ask to drop it. The designer's sentence "a has = 1 row is never fresher than a val row" is wrong for Sep 12 (lag 1 d against 2 d on val, [R `design/out/p06_lagview_counts.txt`]) but immaterial.

### 1.5 The missing flag `has:nx_rp3_z`

- It is 1 on Sep 12-19 and 0 on Sep 6-11 (46.7% of train feeds, [R `p06_lagview_counts.txt`]), and always 1 on val / test. It is therefore the Sep-12 period indicator of map l.273 handed to the network as an input.
- Record on that indicator: a GBDT proxy read +0.00007 ± 0.00013 on val and −0.00005 on the holdout [R `reports/COVERAGE_MAP.md` l.273]; it was never a trained input.
- Why it matters for the rule: the holdout clause is read on exactly the feeds where the flag is 1. A period-specific fit would show there and not on val, and the rule then needs only z_val ≥ 1, which noise delivers 16% of the time.
- Fix: the value-attribution clause of 4.5. The has = 0 holdout feeds (14,775) are a free negative control for side effects of the arm and should be printed.

### 1.6 Staleness parity of the other clauses

The stale-val clause re-scores val with the project scenario for the popularity inputs and the twin for the walk. The two counterfactuals are built by different mechanisms [M `c08_scenario.txt`]: the scenario leaves `seq_cv_invd20` non-zero on 0.110 of val rows (stored 0.114), the twin cuts the walk from 0.198 to 0.157, and in the combination 13.2% of windowed hits have no walk value. The clause is lenient (≥ −1 SE) and I leave it; it is a hybrid, not a model of a test row.

---

## 2. Duplication: does the ranker already read it?

- What the ranker reads of this kind [R A 1.3-1.5]: three windowed co-visitation columns (±20 positions, no degree normalisation), five history-occurrence counts, six scores of a sequence encoder with an item-ID table.
- On val rows pool + own inputs reproduce half of `nx_rp3_z` (R2 0.52; twin 0.42) [M `c05_coef.txt`]. The conditional statistic stays at +6.6..+6.8, so the other half is new in likelihood.
- Its size is one encoder block: 0.0017 nats per feed against 0.0022 for the encoder scalars, which were worth +0.00061 ± 0.00083 on val and +0.00059 ± 0.00049 on the holdout as trained inputs [R B 0.6].
- Inside the bundle the three walk columns are one column: corr(`nx_rp3_z`, `nx_p3a_z`) 0.97; `nx_p3a_z` given the other two reads −0.60 / +0.33 / +0.28; `nx_rp3_rk` given `nx_rp3_z` reads +1.09 / +2.18 / +0.53 (holdout lag view / val / twin) [M `c07_dup.txt`].
- T is not a duplicate (R2 0.00) but is dropped on parity.

---

## 3. Selection

- **Looked at:** 305 columns x 2 forms x 3 bases in B; 57 reads in D; 27 composites, 34 columns and 30 control cells in C; 15 columns in the designer's `p05`. I added 5 columns in forms fixed by the design, nothing chosen.
- **Best of that many noise draws:** the largest \|cz\| of 300 placebo columns is 2.58 on val and 3.69 on the holdout [R B 1.3].
- **W:** +5.0 (lag-built holdout rows) and +6.6 (twin). Above any noise draw; selection does not explain it.
- **T:** the lag-built holdout read is +1.17 in the z form and +3.82 in the rank form, i.e. at the level of the best of 300 noise draws on that witness. Only the val reads stand clear.
- **What selection does explain:** every NDCG number on record for these columns. The one read above 2 SE (fresh walk, nested, holdout +0.00055 ± 0.00025) is the best of 13 groups x 3 reads, and the same column reads −0.00020 ± 0.00046 on val and −0.00126 at the private mix in that form [R B 4.3]. No NDCG support exists; the arm is the first.

---

## 4. The arm and the rule

### 4.1 Is the reference matched?

| Check | Finding | Source |
|---|---|---|
| Flags | Identical to the logged arguments of `w5_e10_base_s951` (10 epochs, d 128 by default in both trainers, lr 2e-3, selection on the holdout) | [M args line of the log; `nn_x/abl/train.py` l.125] |
| Harness | `nn_x/abl/train.py` with its flags off is bit-identical to `nn_x/w5/train.py` at smoke scale (checkpoints, val and holdout predictions) | [R `nn_x/abl/tests/test_abl.py` l.30-35] |
| Input data of the stored controls | The 34 input files were last written on 2026-11-11 17:33; the controls were trained on 11-12 / 11-13 | [M `c09_env.txt`] |
| Seeds | 951-956 shared; pairing buys nothing | [R `design/out/p01_power.txt` (c)] |
| Harness check, 2 fresh against 6 stored | Its SE is 1.63 times the 8 + 8 SE: 0.00059 on val, 0.00034 on the holdout. It can see an offset of 0.0012 / 0.0007, not one of the size of the expected effect | [I from R `p01_power.txt` (a)] |

The reference is matched as far as the records can show. The harness check is weak, and the rule lets its outcome change the control set; it must therefore be read before the arm (amendment A3).

### 4.2 Noise of the decision statistic [M `c04_rule_null.txt`]

| Family | n | Seed sd, val | Seed sd, lag-view holdout | corr(val, holdout) of seed deviations |
|---|---|---|---|---|
| Hard-label base (the stored controls) | 6 | 0.00066 | 0.00034 | −0.58 |
| Hard-label pop1 | 8 | 0.00130 | 0.00050 | +0.60 |
| Hard-label xdrop_pop / pop2 | 4 / 4 | 0.00076 / 0.00074 | 0.00055 / 0.00039 | +0.14 / +0.72 |
| Hard-label, pooled within family | 22 | 0.00098 (0.00071 without pop1) | 0.00045 | +0.36 |
| Wide students, pooled | 16 | 0.00041 | 0.00043 | +0.24 |

The design treats the two witnesses as independent. They are not: a seed that is good on val tends to be good on the holdout. The effect on the false-pass rate is small (0.008 → 0.020).

### 4.3 Can the rule pass by noise?

| Read | PASS | REPLICATE | Note |
|---|---|---|---|
| Simulation, effect 0, first read | 0.008-0.020 | - | four noise models, `c04_rule_null.txt` part 3 |
| 16 wide students, 8 v 8, 300 signed splits | 1 (0.3%) | 9 (3.0%) | sd of z_val 0.69, of z_holdout 0.84 |
| Hard-label pop1, 4 v 4, 70 signed splits | 4 (5.7%) | 0 | one member 0.0029 below its family on val; sd of z_val 1.13 |
| Stored controls, 3 v 3, 20 signed splits | 0 | 0 | |

By seed noise alone the rule passes in about 2% of nulls, as stated. Two paths are not in that number:
- **The period flag** (1.5): if it had an in-period effect of 3 SE on the holdout, the rule would pass in about 13% of cases with no information used [I: P(z_val ≥ 1) x side clauses]. The record makes such an effect unlikely, but the rule should not depend on that.
- **Selection through the rule:** under my prior a PASS overstates the val effect by a factor of 1.25 on average, and when the true effect is +0.0003 a PASS reads +0.0007 [M `c10_prior.txt`].

### 4.4 Power

| True member-level effect on val | Design (incl. replicate) | Mine, first read, full rule | Mine, z clauses only |
|---|---|---|---|
| 0 | ≤ 0.017 | 0.008-0.020 | 0.011-0.027 |
| +0.0003 | 0.12-0.16 | 0.09-0.13 | 0.11-0.16 |
| +0.0005 | 0.28-0.37 | 0.24-0.29 | 0.29-0.34 |
| +0.0010 | 0.77-0.86 | 0.71-0.80 | 0.77-0.86 |
| +0.0020 | 0.99-1.00 | 0.99-1.00 | 1.00 |

Holdout effect 0.7 x val in every row. The smallest effect shown with power 0.8 is +0.0010 to +0.0011 at member level, as the design says.

Two things the power table does not say:
- **Member level is not pool level.** For two of the three stored input contrasts the pool-level value on val is about half of the member-level one (popularity block +0.00160 against +0.00251; click-rate column +0.00033 against +0.00069); the third, the exposure count, is unresolved at +0.0002 in both [R `p01_power.txt` (c2)]. "+0.001 at member level" is about +0.0005 for a pool, before students and the file mix.
- **Where the information on record puts the effect** [I from M]: 0.0016-0.0017 nats on val, three quarters of it usable at the trained weight, at the clean controls' 0.23-0.28 NDCG per nat, times 0.7 for the lens' specificity: about +0.0002 to +0.0003. At that size the test passes about one time in ten.

### 4.5 The instrument the rule lacks: value attribution inside the arm

Re-score the arm's eight members with the two value columns permuted within each feed (one permutation per feed, the flag kept at 1).
- **What it reads:** how much of the members' NDCG depends on the values. The contrast is between two predictions of the same checkpoint, so seed noise cancels; I expect a paired SE near 0.0001-0.0002 [I], against 0.00036 for arm minus control.
- **As a clause of PASS:** natural minus permuted ≥ half of arm minus control on the lag-view holdout feeds, and > 0 on val. A gain that survives the permutation is the flag, not the walk.
- **As the strength of a null:** if PASS is not reached and natural minus permuted on val is below +0.0005 by 2 SE, the bundle is closed at +0.0005 for this recipe. Otherwise the result is "not shown".
- **Cost:** GPU minutes. Both replay tools resolve input files through `data.feat_path`, which `stale_pred.py` already patches for val; `holdout_eval.py` attaches inputs the same way [M code read `nn_x/abl/stale_pred.py` l.32-47, `nn_x/abl/holdout_eval.py` l.23, l.41].
- **Limit:** permutation importance is an upper bound of the gain from adding the input, since a model trained without it finds substitutes.

### 4.6 Does a PASS trigger a sensible next step?

- **Stage B** (8 students, old teacher) is the right next step, but its outcome space is open: "not passed is not a closure, reported with the ratio B / A", and teacher regeneration only if A had pooled z ≥ 3. A stage A pass at z 2-3 followed by a stage B miss leads nowhere and can be read either way afterwards.
- Under my prior P(B passes given A passed) is about 0.45 [I from the design's stage B power at a kept effect of +0.0005: 0.49-0.61].
- **The file step** costs 53-75 GPU-hours and ends in a file rule that cannot prove a gain, as the design says itself (file-level SE 0.0002-0.0005 against an effect below +0.0003). It is acceptable only because two passes stand before it.
- **Stage B needs a pool-level clause**, since what ships is a pool (4.4).

### 4.7 What still allows re-interpretation after the fact

| Hole | Fix |
|---|---|
| Bundle composition chosen at STEP 0 by a gate that uses val labels (nx or nx4) | Fix the bundle now (A1) |
| Harness check may replace the controls; no order of reads is stated | Read it first, blind to the arm (A3) |
| A PASS that comes from the period flag | Value clause (A3) |
| "CLOSED" for a result whose power is 0.1-0.3 at the expected effect | "NOT SHOWN at this power"; closure only through the permutation read (A3) |
| The closure names a class ("new train-stage inputs ... for this competition") after testing one scalar | Delete (A5) |
| Stage B miss after a stage A pass | Terminal branches (A4) |
| A STEP 0 parity miss recorded as "family closed" | "Not testable at parity", not a closure (A2) |
| Effect size quoted from the read that selected it | The estimate of record is stage B's (A3) |

---

## 5. Cost and schedule

| Item | Design | My reading | Source |
|---|---|---|---|
| W build compute | about 1 h | 15-20 min: 13,362 holdout feeds per 7 s at 12 threads, 1.47M feeds to do; my whole run (val, test, 3 lag cuts, 6 twin cuts, 2 reproductions) took 199 s | [M `c01_lagwalk.log`]; [I] scaling |
| Builder writing | 2-3 h | less: `critic/c01_lagwalk.py` calls the kernel of record in all four forms and reproduces the stored arrays | [M] |
| STEP 0 read | T0 + 6 h | T0 + 3 h is realistic | [I] |
| GPU state | 24 processes, 93 GB free per GPU | 24 processes, all ours: 16 all-data students (`w5_fd10_*`) and 8 `w5_e10_kwwd01` students; 94-96% utilisation, 91-96 GB free per GPU. Under the launcher's rule (< 4 processes, ≥ 60 GB free) each GPU takes one more process | [M `c09_env.txt`] |
| Stage A | 10 runs, read at T0 + 10..12 h | 8 at once as fourth processes, then 2 more. The stored controls took 131-182 min each [R A 4.3]; expect the upper end or beyond. Read at T0 + 12..14 h if STEP 0 takes 6 h, T0 + 9..11 h if it takes 3 h | [I] |
| GPU memory | 3.2 GiB for 7 columns | 1.4 GiB for the 3 columns of the reduced bundle | [R A 2.3] |
| Whole chain | 73-140 GPU-hours | unchanged; P(it reaches a file) about 0.07, P(file gain ≥ +0.0005 on private) about 0.02 | [I] |

The schedule is feasible well before 2026-12-08. The cost question is not the 10-30 GPU-hours of stage A but that the likely outcome is a non-result; 4.5 is how to make that outcome say something.

---

## 6. What is missing

| Item | Status | Judgement |
|---|---|---|
| An overlooked raw column or relation | None found. D's scan of the 19 columns, identifier order and file order is complete as far as I can check; I re-read the kernels' inputs and found no unused field beyond those D lists | - |
| **Value attribution by permutation inside trained members** | In no inventory and in no rule. It is the measurement with the power the between-arm contrast lacks (4.5) | Add it |
| Feed-constant inputs | Invisible to every lens used by B, C and D (a column constant inside a feed drops out of a conditional logit). The design introduces one, the period flag, without a control | Covered by the value clause |
| Token-level collaborative inputs (item factor vectors, or pairwise co-visitation as an attention bias) | Never trained. Closed by argument as "CF-equivalent" [R `reports/COVERAGE_MAP.md` l.178, l.348]. The scalar form of ALS reads +4.0..+5.1 on val and stays at +4.6 after the walk is in the null model [R B 4.4], but as built it is transductive and val-only | The one input kind no CPU lens can price. Not for this arm: a day-causal build is 4-8 h [R B 5.1] with the same 14-day parity limits. If stage A passes, it is the next kind to consider, before more scalars |
| The walk as a fine-tune extra (val / test rows only, no train-row parity needed) | Not considered by the design. The val-fitted forms on record are null: nested −0.00020 ± 0.00046, holdout coefficients on val +0.00001 ± 0.00046 [R B 4.3] | No |
| Joint parity with the inputs the ranker already reads | Not in STEP 0 (1.4) | Report it (A2) |

---

## 7. Corrected rule text

Full text: `reports/INPUTS_1116/critic/RULE_AMENDMENTS.txt`. In short:

- **A1 Bundle:** `nx_rp3_z`, `nx_rp3_rk`, `has:nx_rp3_z` (arm `nx3`). T and `nx_p3a_z` dropped. No composition clause.
- **A2 STEP 0:** gates and bars for W unchanged; my advance values on record; the two joint statistics of 1.4 printed without a bar. "W fails G-parity only" is recorded as not testable at parity, not as a closure.
- **A3 Stage A:** harness check read first. PASS also needs the value clause. Printed next to the verdict: arm minus control on the 14,775 has = 0 holdout feeds; natural minus permuted on val with its SE. "NOT SHOWN at this power" replaces "CLOSED"; "CLOSED at +0.0005 for this recipe" only through the permutation read. The estimate of record after a PASS is stage B's.
- **A4 Stage B:** pool-level pooled z ≥ 1.0 added. pooled z_B ≥ 2 → all-data students; 1.6-2 → 4 more students, one re-read; below 1.6 → stop, no file.
- **A5 Closure:** delete "for this competition" and "no further input arm is opened on a likelihood read alone".
- **A6 Forecast:** P(STEP 0) 0.85; P(PASS at A) 0.15; P(member-level ≥ +0.001) 0.05; P(file-level private ≥ +0.0005) 0.02.

---

## 8. Limits of this critique

- My builds are an advance read, not the build of record: sparse members are admitted by day, not by timestamp, and they enter the degree as in `c09_build.py`; BUILD_SPEC says timestamp and no degree. "Test-like train rows" are the holdout feeds of Sep 17-19 (p_random > 0.9, M ≥ 50), not the p_random > 0.5 pool of the spec.
- I read the information gate with val and holdout labels on 5 columns. The bars were written before; the result did not choose anything for W, and T is dropped on the label-free gate.
- The lens is linear in two forms of a column, in-sample, and says nothing about NDCG. The coefficient comparison of 1.4 is a one-step score coefficient given the pool, not what a network would learn.
- Nothing was trained. The value-attribution read of 4.5 is designed, not tested; its SE is my estimate.
- The false-pass and power figures rest on a normal model with seed noise from 22 hard-label members of four input views; the empirical A/A splits reuse the same 6, 8 or 16 members.
- The 13% false-pass figure for a flag effect is conditional on an effect whose existence the record does not support.
- I did not review the stage B launcher beyond its flags, the all-data lane copy, or the fine-tune path.

## 9. Files (all in `reports/INPUTS_1116/critic/`)

| File | Content |
|---|---|
| `RULE_AMENDMENTS.txt` | corrected rule text A1-A6 |
| `c01_lagwalk.py` → `out/c01_lagwalk.log`, `out/c01_walk.npz` | the walk by the kernel of record: reproduction, lag view on the 15,225 holdout feeds, stale-val twin, natural val and test |
| `c02_lagbt.py` → `out/c02_lagbt.log`, `out/c02_bt.npz` | the click-strength fit of record in the same four forms |
| `c03_gates.py` → `out/c03_gates.txt` | the STEP 0 gates on those builds: coverage by listing age, PSI, calibrated lens with placebo |
| `c04_rule_null.py` → `out/c04_rule_null.txt` | seed noise, A/A of the complete rule on stored members, simulation of false pass and power |
| `c05_coef.py` → `out/c05_coef.txt` | coefficient and R2 of the entered forms, train rows against val rows |
| `c06_joint.py` → `out/c06_joint.txt` | joint distribution of the walk with `seq_cv_*` on train, val and test rows |
| `c07_dup.py` → `out/c07_dup.txt` | the three walk columns as one piece of information |
| `out/c08_scenario.txt` | the stale-val scenario next to the twin |
| `out/c09_env.txt` | file dates of the 34 inputs, stored controls, GPU state |
| `c10_prior.py` → `out/c10_prior.txt` | what a PASS means under a stated prior |

Run form: `cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=reports/INPUTS_1116/critic/out/numba_cache .venv/bin/python reports/INPUTS_1116/critic/<script>`. Longest jobs: `c01` 199 s, `c02` 166 s; the lens reads (`c03`, `c05`, `c07`) about 2 minutes each [M file times and logs].
