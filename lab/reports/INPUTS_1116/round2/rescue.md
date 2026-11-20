# RESCUE (round 2): the pieces dropped in round 1 for parity / staleness

Explorer "rescue", 2026-11-16 04:50-05:40 UTC. CPU only (at most 12 threads in use, split over at most three processes), no GPU process, nothing trained, nothing installed, nothing submitted, no process signalled. Every file of mine is in `reports/INPUTS_1116/round2/rescue/` (section 7), plus this report. Paths are relative to `<repo>`; outputs are in `reports/INPUTS_1116/round2/rescue/out/` unless a path is given.

"cz" = score statistic of `reports/INPUTS_1116/residual_rows/condlib.py` (N(0,1) when the column adds nothing). Here the null model is always **pool + the control block of the ranker's own inputs + `nx_rp3_z` + `nx_rp3_rk`**, so a non-zero read is information beyond nx3. Witness **H** = holdout rows of the 15,225 lag-view feeds; witness **T** = val rows with the stale-val twin of the column (train-only pool of the 16 wide students). Forms, bars and the list of 21 column-forms were fixed in `PREDECLARED.txt` before any array existed.

---

## 0. The answer

**0.1 One construction makes a parity rebuild possible for all seven pieces: the nx3 lag view with EQUAL DEPTH ("D5").** Every row of every split reads the last 5 dense source days below its cut (train rows: the nx3 cut' = max(slot - L, 5); val / test natural: cut 14; stale-val twin: cut = slot - L). Round 1 failed on depth (5-7 source days on train rows against 12 on val / test rows); D5 removes that by construction and pays for it with 7 of the 12 source days on val / test rows.

**0.2 Table.**

| Piece | Rebuild possible | Parity (C3 / C4 / C5, bars of nx3) | cz H (holdout, lag view) | cz T (stale-val twin) | Verdict |
|---|---|---|---|---|---|
| Density `kn1_5`, `kn20_5` | yes | **PASS** (dZ 0.005-0.008, bucket <= 0.015, PSI <= 0.001) | `kn1_5:z` **-4.55** | **-3.22** | **SURVIVES the gates.** Not a lever for private on its own: -0.81 at the private-mix weights, -0.62 on the first val day |
| Surface split `X3_5`, `ns2_5` | yes | FAIL on one clause by 0.009 (val-like rows against val, bucket 3-7 d: 0.109 against 0.10); the two test-like clauses, C4 and C5 pass | `ns2_5:z` **-4.50**; `X3_5:z` +2.15 | **-3.15**; +2.34 | INFORMATION WITHOUT PARITY for `ns2_5`; `X3` not shown. The most robust of the reads (private-mix weights -2.38; test-like half of H -2.85) |
| Click strength `bt5`, `bt5_seen` | yes | FAIL on one clause by 0.002 (same clause: 0.102 against 0.10); test-like dZ 0.017 against the critic's 0.095-0.098; PSI rank form 0.126 against 0.479 | `bt5:rk` **+4.17**; `bt5:z` +0.98; seen -4.22 | **+4.12**; +1.83; -2.98 | INFORMATION WITHOUT PARITY for the rank form. At the test lag on the holdout it reads only +2.1 |
| All-feed click rate `coec5` | yes | FAIL (val-like against val: dZ 0.0503, bucket 0.157) | `coec5:rk` +3.72; `:z` +1.51 | +3.30; +1.25 | INFORMATION WITHOUT PARITY, and not a separate piece: +2.5 / +2.1 given `bt5:rk`; +3.15 -> +1.39 between the val days; +1.04 at the private-mix weights. Stays dropped |
| First-seen gap `gapf5`, `never5` | yes, only as a 5-day window | FAIL (C3 as the click rate; C4 `gapf5:z` 0.29) | `gapf5` -1.32 / -1.19; `never5` +4.66 | -2.48 / -2.41; +3.54 | Gap NOT SHOWN. `never5` is a staleness contrast, not an input: it reads -0.14 on natural val rows and +1.7 given the other pieces. Stays dropped |
| Same-user earlier feeds `uc5`, `reshown5` | yes: coverage is equal in D5 (0.156 against 0.148 / 0.164) | **PASS** | `uc5` +0.83 / +1.24; `reshown5` -1.72 | +1.27 / +2.14; -1.91 | NOT SHOWN |
| ALS `als5` (day-causal fits) | yes | **PASS** (all windows) | +3.44 / +3.63 | +2.19 / +1.97 | NOT SHOWN on val. The stored transductive column reads +3.4 / +3.8 on natural val under the same null model; the causal 5-day build reads +1.7 / +1.4 there |

Sources: `r10_gates_main.txt`, `r10_gates_alsp1.txt`, `r10_gates_alsfull.txt`, `r11_robust_main.txt`.

**0.3 One survivor by the rule, and it is weak where it counts.** `kn1_5` in the z form passes every gate and is independent of nx3 and of every other piece (-4.02 / -3.15 given all of them). Its information sits where the private mix has little weight. I do not recommend an arm for it alone.

**0.4 Two near-misses the lead may want to rule on.** `ns2_5:z` and `bt5:rk` pass the information gate on both witnesses and on natural val rows, with placebo-calibrated z of -4.7 / -3.5 and +4.6 / +3.8 and no placebo beyond them in 100 permutations each. Each misses one parity clause, the same one, by 0.009 and 0.002. The cause is the lag rule inherited from nx3: its val-like train rows are 2-4 days behind their newest source day, val rows 3 and 5 days (`r14_lagtable.txt`). In a view with the test-like rows only (has = slot >= 11) every parity clause passes for both; there `ns2_5:z` reads -2.85 on the holdout (above the 2.5 bar, below the largest of 100 placebos of that half, 3.49) and `bt5:rk` reads +2.06. That view was not pre-declared here; it would need its own block.

**0.5 Size.** The three independent forms together (`bt5:rk`, `kn1_5:z`, `ns2_5:z`) add 0.00085 nats per feed on H and 0.00123 on T, in-sample, 3 df (`r11_robust_bundle3.txt`). nx3 adds 0.00044 and 0.00179 on the same footing. So the bundle is about 1.9 times nx3 on the holdout and 0.7 times nx3 on stale val. With the record's conversion for nx3 (central +0.0003 per model) that is about +0.0002 per model on val-like inputs, less at the private mix. **Nothing here is of the size of the 0.0032 gap.**

**0.6 Looks.** 42 deciding reads (21 column-forms x 2 witnesses), as pre-declared; 4 more of the same kind on the complete ALS build. About 480 further score reads were printed as information (regions, natural val, other null models, overlap, stored columns) and about 3,100 placebo scores.

---

## 1. The view D5

Definition: `PREDECLARED.txt` section 1; code `rs_common.py` (`window`, `Views`). The cuts are nx3's files (`nn_x/w12/nx3/out/raw/train_cut_feed.npy`, `val_twin_cut_feed.npy`), re-derived and asserted equal in `rs_common.py`.

| Rows | Cut | Source slots | Feeds |
|---|---|---|---|
| Train, lag view (holdout feeds; pool feeds) | 5 / 6 / 7 | 0-4 / 1-5 / 2-6 | holdout 13,362 / 1,291 / 572 (14,775 without a view); pool 60,776 / 5,892 / 2,505 |
| Val, test natural | 14 | 7, 8, 11, 12, 13 | 14,423; 28,595 |
| Val, stale-val twin | 7 / 8 / 9-11 / 12 | 2-6 / 3-7 / 4-8 / 5-8 + 11 | 1,965 / 1,987 / 8,971 / 1,500 |

Source: `r01_item.log` header. Pool feeds = p_random > 0.5, M >= 50, not holdout, slot >= 6 (69,173 feeds, 10,254,287 rows), the population that decides C3 / C4 in `nn_x/w12/nx3/nx3_gates.py`. No full-train build was made.

Lag of a row to its newest source day (`r14_lagtable.txt`):

| Rows | Lag, days (share) | Mean | Calendar span of the 5 source days |
|---|---|---|---|
| Train pool, slots 6-8 (val-like) | 2 (0.26), 3 (0.36), 4 (0.38) | 3.11 | 5 |
| Val natural | 3 (0.52), 5 (0.48) | 3.95 | 7 |
| Train pool, slots 11-13 (test-like) | 7 (0.48), 8 (0.35), 9 (0.17) | 7.68 | 5 |
| Test public | 7 (0.71), 8 (0.29) | 7.29 | 7 |
| Test private | 8 (0.24), 9 (0.38), 10 (0.38) | 9.14 | 7 |
| Val stale-val twin | 7 (0.10), 8 (0.25), 9 (0.14), 10 (0.51) | 9.05 | 5 (0.90), 7 (0.10) |

- Equal by construction: the number of source days and the rule that picks them.
- Not equal, and inherited from the nx3 lag rule: val-like train rows are about one day fresher than val rows; lag 10 is absent and lag 9 under-represented on train rows (private rows: 76% at 9-10 days). The val / test window spans 7 calendar days because Sep 15-16 are empty.
- Label statistics are lag-matched by construction. Proof in three parts: (i) `r01_item.py` asserts for every feed with a view that its newest source slot is below its cut and the cut at or below slot - 1 (train) or slot (val / test); printed minimum distance 2 days (train), 3 (val), 7 (twin, test). (ii) Sources exclude the 30,000 holdout feeds; no val or test label array is opened by any builder (test has none). (iii) `r12_checks.txt` part A re-derives the click counts from the raw arrays on 1,700 sampled rows with 0 mismatches.

---

## 2. Piece by piece

### 2.1 Click strength bt

- **Round-1 reason:** `reports/INPUTS_1116/CRITIC.md` l.21: "Parity is not demonstrated: the family fails the designer's own coverage gate by a factor of two (dZ 0.095-0.107 against 0.05, largest bucket 0.167-0.185 against 0.10, PSI 0.48 against 0.25)"; cause at l.98: 5 source days instead of 12.
- **Rebuild:** the fit of record (`c06_build.fit_bt`, lambda 3) on the exploration feeds of the row's 5-day window, 7 fits of 186,255-212,192 feeds (`r01_item.log`). Seen share: pool 0.316, val natural 0.355, test 0.274, twin 0.246.
- **Parity** (`r10_gates_main.txt`): test-like pool rows against private / public dZ 0.0167 / 0.0172, largest bucket 0.0133 / 0.0224. Val-like against val dZ 0.0257, **largest bucket 0.1020 at 3-7 d (bar 0.10)**; on the holdout feeds 0.0971. PSI pool rows against val / public / private: z form 0.018 / 0.013 / 0.022, rank form 0.126 / 0.020 / 0.082. C5: lowest test day 0.258 against a bar of 0.191.
- **Information:** `bt5:rk` +4.17 (H) / +4.12 (T); `bt5:z` +0.98 / +1.83; `bt5_seen:z` -4.22 / -2.98. 100 placebos of `bt5:rk`: sd 0.91 / 1.10, largest 2.31 / 2.64 (`r11_robust_main.txt`).
- **Where it sits:** H slots 6-8 +3.70, **slots 11-13 +2.06**. T slot 16 / 18 +2.85 / +2.75; M >= 50 +4.23, M < 50 +1.20; private-mix weights +2.74. Natural val +3.57.
- **For scale:** the critic's lag view of the same column (5-7 source days on train rows, 7-10 on the twin) reads +4.26 / +5.38 under my null model. Equal depth costs a quarter of the twin read.
- **Verdict:** information without parity, at the bar. In a test-like-only view parity would pass and the holdout read would be +2.1.

### 2.2 All-feed click rate

- **Round-1 reason:** `reports/INPUTS_1116/DESIGN.md` l.59 ("yes, but it is the decaying form of T") and l.269 (cz +3.6 -> +1.3 between the two val days).
- **Rebuild:** `coec5` = (clicks + 1) / (expected + 1) over the window, all non-holdout feeds, position prior from slots 0-4 only (`r01_item.py`).
- **Parity:** val-like against val dZ 0.0503, bucket 0.1572 (FAIL); test-like 0.0328 / 0.0113 and 0.0344 / 0.0083 (ok); PSI <= 0.072.
- **Information:** `coec5:rk` +3.72 / +3.30, `coec5:z` +1.51 / +1.25. Given `bt5:rk`: +2.47 / +2.05. T slot 16 / 18: +3.15 / +1.39. Private-mix weights +1.04.
- **Verdict:** the round-1 judgement stands. It decays between the val days at parity too, and it is the same piece as bt.

### 2.3 Density

- **Round-1 reason:** `DESIGN.md` l.56 ("no: the index of much-shown items has a size proportional to depth") and l.266 (cz -1.3 at the private weights).
- **Rebuild:** `r02_density.py`: block c of `nn_x/w12/cb/famB/b3_build_c.py`, with the index built from the candidate log of the row's window (4.0-4.5 M items per window). Label-free.
- **G-code:** the same kernel with an all-days index reproduces the stored `kn_cos1` / `kn_cos20` of record on all 1,389,757 present val rows, max abs diff 1.2e-07 (`r13_dens_repro.txt`).
- **Parity:** substitute for C3 (share above the natural-val median): dZ 0.0050 / 0.0059 / 0.0077, bucket <= 0.0150. PSI <= 0.0010. PASS.
- **Information:** `kn1_5:z` -4.55 / -3.22; `kn1_5:rk` -4.33 / -2.65; `kn20_5` -4.24 / -2.64 and -4.03 / -2.05. 100 placebos of `kn1_5:z`: sd 0.95 / 0.98, largest 2.27 / 2.59.
- **Where it sits:** H slots 6-8 -3.59, slots 11-13 -2.55. T slot 16 / 18 **-0.62** / -3.64; public-mix weights -1.56; **private-mix weights -0.81**. Natural val -3.36. Coefficient: -0.035 (H), -0.028 (T), -0.030 (natural val).
- **Verdict:** survives the gates. The record's reservation (weak at the private mix, one val day) is confirmed at parity.

### 2.4 Surface split

- **Round-1 reason:** `DESIGN.md` l.57 and l.267: lag-buildable, kept in reserve because D's form fell from +5.5 to +3.3 at the test lag.
- **Rebuild:** `r03_surface.py`: construction V-lf of `nn_x/codes/exploit/02_s1s2contrast/run.py` on the history snapshots of the train feeds of the window, own user removed. Label-free.
- **Parity:** test-like against private / public dZ 0.0361 / 0.0199, bucket 0.0808 / 0.0337 (ok). Val-like against val dZ 0.0451, **bucket 0.1094 at 3-7 d**; holdout feeds 0.0915. PSI: `X3_5` <= 0.092; `ns2_5` 0.006 / 0.107 / 0.164 (z) and 0.008 / 0.116 / 0.185 (rank).
- **Information:** `X3_5:z` +2.15 / +2.34, `X3_5:rk` +2.10 / +0.71 (not shown). `ns2_5:z` **-4.50 / -3.15**, `ns2_5:rk` -3.93 / -2.99. 100 placebos of `ns2_5:z`: sd 0.95 / 0.91, largest 2.53 / 1.90.
- **Where it sits:** H slots 6-8 -2.81, slots 11-13 -2.85. T slot 16 / 18 -1.01 / -2.67; private-mix weights -2.38. Natural val -2.69 (z), -2.86 (rank). Coefficient -0.031 / -0.029 / -0.025.
- **Verdict:** information without parity, by 0.009 on one clause. The piece is the search-surface viewer count, not the ratio that round 1 carried as its primary column.

### 2.5 First-seen gap

- **Round-1 reason:** `DESIGN.md` l.58 and l.268: train rows look back 0-13 days, test rows 7-23 days.
- **Rebuild:** level parity needs equal look-back, so the gap can only exist censored at the 5-day window (`gapf5` in 1..5, `never5`); `r01_item.py`.
- **Result:** `gapf5` -1.32 / -1.19 on H, -2.48 / -2.41 on T: not shown. Parity fails too (C4 `gapf5:z` 0.29 against val).
- **`never5:z` reads +4.66 / +3.54, and I do not accept it:** (i) on natural val rows, where the flag is as fresh as the pool's own inputs, it reads -0.14; on H and T the pool's inputs are fresher than the flag, on test rows they are not. (ii) Given the other forms it reads +1.73 / +1.71. (iii) T slot 18 +1.65, private-mix weights +1.12.
- **Verdict:** stays dropped.

### 2.6 Same-user earlier feeds

- **Round-1 reason:** `DESIGN.md` l.60 and l.264: cz +2.0..+2.2 / +1.2..+1.9, and 4.6% of train feeds keep a source at a >= 7 d lag against 24-26% of test rows.
- **Rebuild:** the coverage gap is a depth gap. With 5 source days for every row the share of feeds whose user has a source feed is 0.154 (pool, slots 11-13) against 0.166 (public) and 0.149 (private), and 0.168 against 0.168 (val-like / val); with every day below the cut it is 0.161 against 0.304 / 0.285 (`r04_user.log`).
- **Parity:** PASS (dZ <= 0.009, PSI <= 0.003).
- **Information:** `uc5:z` +0.83 / +1.27, `uc5:rk` +1.24 / +2.14, `reshown5:z` -1.72 / -1.91 (natural val -3.21; the flag is 1 on 0.06-0.11% of rows).
- **Verdict:** not shown. Parity was never the obstacle; the piece carries too little.

### 2.7 ALS

- **Round-1 reason:** `DESIGN.md` l.62 and l.270 ("val-only support; 4-8 h new build; as built transductive"); `CRITIC.md` l.279.
- **Rebuild:** `r05_als.py`: implicit ALS (f 64, 12 iterations, CG 3, alpha 5, reg 0.05) on the dense sequences of each window, label-free, per-row CG in numba. 7 fits of 1.12-1.29 M sequences and 107-125 M pairs, about 130 s each at 6 threads. Factor norms |X| 0.007-0.008, |Y| 0.18, the regime of the stored GPU fits (`eda/collab_stats/out_04*.log`: 0.004-0.008, 0.05-0.29).
- **Parity** (all windows, `r10_gates_alsfull.txt`): dZ 0.0316 / 0.0170 / 0.0428, bucket <= 0.0884, PSI <= 0.016. PASS. The first pass of three windows fails C3 only because the rows of the other cuts were still empty.
- **Information:** first pass `als5p1` +3.29 / +3.34 (H), +2.10 / +1.99 (T). All windows: +3.44 / +3.63, +2.19 / +1.97. Natural val +1.67 / +1.39 (+2.44 / +2.12 without nx3).
- **Calibration:** the stored transductive `als_exp5` reads +4.04 / +4.42 on natural val without nx3 (the record's value) and +3.38 / +3.77 with nx3 (`r12_checks.txt` part C).
- **Verdict:** not shown on val. By the pre-declared order the first pass decides; the complete build agrees. What the stored column knew on val is not in a causal 5-day build. I did not separate depth from transduction (a causal 12-day fit would do it, and could not be an input at parity either way).

---

## 3. Robustness of the passes (`r11_robust_main.txt`, `r11_robust_bundle3.txt`; written after the gates were read, information only)

**3.0 Two slice estimators.** Every slice value quoted in sections 0 and 2 (holdout slots, val days) is the one of `r10_gates_main.txt`: the column is set to 0 outside the slice and scored again, which is what a view restricted to that slice would hold. `r11_robust_main.txt` prints the other form, the per-feed scores of the full column summed over the slice; it differs by up to 1.0 (`ns2_5:z`: -3.50 on holdout slots 11-13 and -2.01 on val slot 16 there, against -2.85 and -1.01). The private-mix and M reads exist only in the second form.

**3.1 Placebo, 100 within-feed permutations per form and witness.** sd 0.87-1.00 on H and 0.91-1.10 on T; no placebo reaches the read for any of the six passing forms. The pooled 20-per-piece placebos of the gate run give sd 1.04 / 1.06 and maxima 2.70 / 2.99, so the bar was 2.61 on H and 2.66 on T.

**3.2 Overlap.** cz of the row form given the null model plus all the other forms, H / T:

| Form | Alone | Given the other five | Given the other two of the bundle of three |
|---|---|---|---|
| `bt5:rk` | +4.17 / +4.12 | +1.27 / +2.35 | +3.55 / +3.97 |
| `bt5_seen:z` | -4.22 / -2.98 | -0.45 / +0.65 | - |
| `coec5:rk` | +3.72 / +3.30 | +1.38 / +0.86 | - |
| `never5:z` | +4.66 / +3.54 | +1.73 / +1.71 | - |
| `kn1_5:z` | -4.55 / -3.22 | -4.02 / -3.15 | -4.13 / -3.14 |
| `ns2_5:z` | -4.50 / -3.15 | -3.40 / -2.97 | -3.78 / -2.99 |

Three pieces, not six: the click family (bt, click rate, never-seen flag) is one; density and the search-surface count are each their own.

**3.3 Size, log-likelihood per feed.**

| | H (30,000 feeds) | T (14,423 feeds) |
|---|---|---|
| own inputs over the pool | 0.00487 | 0.00899 |
| nx3 over that | 0.00044 | 0.00179 |
| the six forms over nx3 | 0.00101 (LR 60.8, 6 df) | 0.00146 (LR 42.0, 6 df) |
| `bt5:rk` + `kn1_5:z` + `ns2_5:z` over nx3 | 0.00085 (LR 50.9, 3 df) | 0.00123 (LR 35.3, 3 df) |

Joint coefficients of the three agree between the witnesses: +0.092 / +0.127, -0.031 / -0.027, -0.026 / -0.028. For nx3 the train-row coefficient was 0.54 of the val one (`reports/INPUTS_1116/nx3_step0.md` J2).

**3.4 The asymmetry both witnesses share.** On H and on T the column is 2-10 days stale while the pool's own popularity inputs are 1-4 days fresh. On test rows both are stale. The natural-val read has column and inputs at the same staleness, at a shorter lag. A form I would trust passes all three: `bt5:rk` +3.57, `kn1_5:z` -3.36, `ns2_5:z` -2.69, `coec5:rk` +3.19; `never5:z` -0.14.

---

## 4. Checks (`r12_checks.txt`, `r13_dens_repro.txt`)

| Check | Result |
|---|---|
| Re-derivation from the raw arrays, 1,700 sampled rows of the five query sets | 0 mismatches for impressions, clicks over expected, gap, never-seen, bt source membership, n_s1, n_s2; 0 of 600 for `uc5`, `reshown5` |
| Density kernel against the stored columns of record | exact (1.2e-07) |
| The conditioning | `nx_rp3_z` reads +0.000 under a null model that holds it and +4.98 without it, the record's value (`r10_gates_main.txt`) |
| My use of the lens on the stored columns, natural val, without nx3 | bt +3.87 / +5.38, click rate +3.48 / +3.46, density -4.01 / -3.13, X3 +4.54 / +3.34, n_s2 -3.79 / -3.83, gap -5.60 / -5.34, ALS +4.04 / +4.42: the values of `reports/INPUTS_1116/residual_rows.md` 4.1 |
| D5 natural val against the stored full-depth column, Pearson | bt 0.76, click rate 0.77, kn1 0.92, n_s2 0.97, never-seen 0.86, uc 0.91, ALS 0.74 |
| Test-like-only view, PSI of pool rows of slots 11-13 against public / private | `bt5:rk` 0.065 / 0.008, `kn1_5:z` 0.0003 / 0.0004, `ns2_5:z` 0.027 / 0.057 |

---

## 5. What I would do with this

1. Nothing before nx3's stage A is read. None of these pieces changes the size of what is being tested.
2. If stage A passes, the next bundle to consider is nx3 + `ns2_5` (z form) + `bt5` (rank form) + `kn1_5` (z form), in that order of trust. It needs a ruling first: either accept 0.109 / 0.102 against a bar of 0.10 on the val-like clause, or declare the view with test-like rows only, in which `bt5:rk` has no holdout support (+2.06) and `ns2_5:z` reads -2.85.
3. If stage A does not pass, I would not open an arm for these: together they carry 0.7 of nx3's information on stale val.

---

## 6. Limits

- The lens is linear in two forms of a column, in-sample, and says nothing about NDCG. Nothing was trained.
- Train-side arrays exist for the 30,000 holdout feeds and the 69,173 pool feeds only. Gates on all has = 1 train rows (the flag nx3 carries at `nx3_step0.md` 3.3) were not read.
- Both witnesses read a stale column next to fresh pool inputs (3.4). No stale-val re-scoring of the pool was possible on CPU.
- The three near-passes were found among 21 column-forms. Two witnesses, a third agreeing read and 100 placebos each argue against noise, but the sizes in 3.3 are those of the selected forms.
- `bt5` is a label statistic. Its causality is shown at feed level and by re-derivation; a full-train build would need the C2-style audit of nx3 on every row.
- The val / test window spans 7 calendar days, the train window 5; lag 10 is not represented on train rows.
- ALS: one recipe, one seed, depth and transduction not separated.
- `r10_gates.py` and `r11_robust.py` gained one line (a non-finite value becomes a missing value) after their `main` runs and before the ALS runs. It changes nothing for columns without non-finite values; 5 pool rows of `als5p1` had one.
- A first launch of `r01_item.py` died on a thread-count setting before any table was built (`out/r01_item.failed1.log`).

---

## 7. Files

All in `reports/INPUTS_1116/round2/rescue/`.

| File | Content |
|---|---|
| `PREDECLARED.txt` | view, pieces, forms, gates, pass rule; written first |
| `rs_common.py` | the view D5, query sets, array naming |
| `r01_item.py` -> `out/r01_item.log` | bt5, bt5_seen, coec5, ctr5, imp5, gapf5, never5 |
| `r02_density.py` -> `out/r02_density.log` | kn1_5, kn20_5 |
| `r03_surface.py` -> `out/r03_surface.log` | ns1_5, ns2_5, X3_5 |
| `r04_user.py` -> `out/r04_user.log` | coverage step, uc5, uhas5, reshown5 |
| `r05_als.py` -> `out/r05_als.log` | als5p1 (three windows), als5 (all seven) |
| `r10_gates.py` -> `out/r10_gates_{main,alsp1,alsfull}.{txt,json}` | parity gates and the 42 + 4 deciding reads |
| `r11_robust.py` -> `out/r11_robust_{main,bundle3}.txt` | placebo x 100, slices, overlap, size |
| `r12_checks.py` -> `out/r12_checks.txt` | re-derivation, stored columns, calibration, test-like-only PSI |
| `r13_dens_repro.py` -> `out/r13_dens_repro.txt` | density kernel against the record |
| `r14_lagtable.py` -> `out/r14_lagtable.txt` | lag and span of the view |
| `out/arr/<set>__<column>.npy` | 85 float32 arrays; sets `hold` (4,630,966 rows, order of `nn_x/w5/preds/holdout/holdout_meta.npy`, NaN without a view), `pool` (10,254,287), `val_nat`, `val_twin` (1,389,971), `test_nat` (2,904,806) |

The survivor: `out/arr/{hold,pool,val_nat,val_twin,test_nat}__kn1_5.npy`, md5 `0a05806b...`, `0a400e9c...`, `1a0295ab...`, `2affcd24...`, `267ddbd8...`; entered form = `lens.feed_z(x, off, M, 5.0)`. Build: `r02_density.py` (md5 `b9dafb74...`), 123 s at 6 threads.

Run form: `cd reports/INPUTS_1116/round2/rescue && RS_NT=<threads> CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=<threads> MKL_NUM_THREADS=<threads> OPENBLAS_NUM_THREADS=<threads> POLARS_MAX_THREADS=<threads> <repo>/.venv/bin/python -u <script> [args]`. Gates: `r10_gates.py <tag> bt,coec,dens,surf,gap,user`. Arrays and logs are created exclusively; a re-run needs a new tag or an empty `out/arr`.
