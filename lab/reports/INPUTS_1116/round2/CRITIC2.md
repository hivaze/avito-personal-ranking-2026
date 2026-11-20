# CRITIC of round 2 (explorers "rescue" and "new constructions")

Written 2026-11-16 05:40-06:35 UTC. CPU only, at most 12 threads, one process at a time; no GPU process, nothing trained, installed or submitted; no external service. Every file of mine is in `reports/INPUTS_1116/round2/critic/` (section 10) plus this report; nothing else was written (checked with `find -newer`). Paths are relative to `<repo>`; my outputs are in `reports/INPUTS_1116/round2/critic/out/`.

"cz" = score statistic of `reports/INPUTS_1116/residual_rows/condlib.py` (N(0,1) when a column adds nothing). Null model in every read below unless stated: pool of the 16 held-out wide students + the lens's control block of the ranker's own inputs + nx3 (`nx_rp3_z`, `nx_rp3_rk`). **H** = holdout rows in the lag view (15,225 feeds with a view, 0 on the other 14,775). **T** = val rows with the stale-val twin of the column (and of nx3). **N** = val rows with the natural column. The fit of the null model is the lens of record; the score of a candidate column is my own numpy path (explicit residualisation, `critic/cc_common.py`), which reproduces the record's nx3 values (+4.98 / +4.32 on H, +6.46 / +4.78 on T) and every explorer value I re-read, to two decimals.

---

## 0. Verdict

**NOTHING is worth a train-stage test of its own.** No GPU run from round 2.

| Piece | Explorer's status | My verdict | Decisive facts |
|---|---|---|---|
| Density `kn1_5:z` | SURVIVES | **Refuted as a survivor.** Real on the holdout; on val it does not clear a structured placebo; not shown at the private mix | T: 1 of 200 within-feed permutations beyond the read (max 3.67 against a read of 3.23), so the explorer's own clause "above every placebo" holds only for his draw. A content column with no catalogue information ("cosine to a random train item") reads mean -1.07, rms 1.57 on T; 3% of such columns exceed density's read. Private-mix weights -0.84 (T), -1.05 (N) |
| Search-surface count `ns2_5:z` | information without parity | **Clean, and too small.** The only column of round 2 that survived every attack | Parity fails only for the piece-level indicator (n_s1 + n_s2 > 0); for the column's own indicator (n_s2 > 0) every clause passes, with no row-share exemption. Placebo inside popularity strata: -4.93 (H), -3.77 (T). Private-mix weights -2.57 (T). Likelihood 0.00042 nats per feed on T = about +0.0001 per model |
| Click strength `bt5:rk` | information without parity | **Not ready.** More than half of the read is carried by the exposure pattern of the window, not by clicks; a parity gap against private rows is hidden by the gate's 5% exemption | Placebo that keeps the exposure of the window and shuffles the click part reads +2.56 (H) / +2.34 (T); the click part alone is +2.0 / +2.2 sd. Seen share of 7-10 d old items: 0.370 on test-like train rows against 0.143 on private rows |
| Six new constructions | none survives | **Confirmed: nothing.** The cold-row effect is not a new input | The ranker's own `cos_u` and `seq_max_all` show the same cold / warm split, larger: +4.2 / -3.0 and +4.2 / -3.5 on T, +2.0 / -3.8 and +1.1 / -2.9 on H |

**Why no arm.** By the conversion of the two positive controls (0.23-0.28 NDCG per nat on val, 0.40-0.46 on the holdout; `reports/INPUTS_1116/residual_rows.md` l.47-48) the three forms together are worth about +0.0003 per model on val-like inputs, `ns2_5:z` + `bt5:rk` about +0.0002. Read directly in NDCG with the control block re-weighted in both arms (`c09_ndcg.txt`), the trio is +0.0004 ± 0.0004 on val and -0.0002 ± 0.0003 on the lag-view holdout (-0.0004 ± 0.0004 on its test-like half). A stage-A test of the nx3 form needs z >= +1 on both witnesses and has power 0.09-0.13 at +0.0003 (`reports/INPUTS_1116/CRITIC.md` 0.4). My forecast: **P(pass stage A) 0.07 for the trio or the pair as its own arm, 0.05 for `ns2_5:z` alone.**

**What I would keep [judgement].** If nx3's stage A shows that its values are used (the VALUE clause), the next arm that carries nx3 may carry `ns2_5:z` as one more column, under its own rule block and after a full-train build. That is a rider on someone else's test, not a bundle. `bt5:rk` would need the lead's ruling on the 7-10 d bucket first. Density should not ride for the private board.

**Not a holdout-only rejection.** Density is rejected on val (placebo, structured placebo, private-mix weights). `ns2_5:z` is not rejected as information at all; it is too small for the test, by val's own numbers.

---

## 1. The numbers, re-run from the arrays

Source: `c01_reread.txt`. Entries are cz; "nats" = in-sample likelihood gain per feed.

| Form | H | H slots 6-8 / 11-13 | T (twin nx3 in the null) | T with the natural nx3 (explorer's read) | N (natural val) | nats H / T | R2 with the null model H / T |
|---|---|---|---|---|---|---|---|
| `kn1_5:z` | -4.55 | -3.74 / -2.61 | -3.23 | -3.22 | -3.36 | 0.00037 / 0.00037 | 0.16 / 0.26 |
| `ns2_5:z` | -4.50 | -3.14 / -3.50 | -3.51 | -3.15 | -2.69 | 0.00032 / 0.00042 | 0.34 / 0.52 |
| `bt5:rk` | +4.17 | +3.79 / +2.11 | +4.18 | +4.12 | +3.57 | 0.00028 / 0.00059 | 0.08 / 0.15 |
| yardstick `nx_rp3_z`, null model without nx3 | +4.98 | +4.32 / +3.37 | +6.46 | - | +6.78 | 0.00044 / 0.00154 | 0.25 / 0.42 |

- **The explorer's table is correct.** Every value of `rescue/out/r10_gates_main.txt` that I re-read agrees to two decimals, including the forms that fail (`bt5:z` +0.98 / +1.83, `kn1_5:rk` -4.33 / -2.65, `X3_5:z` +2.15 / +2.34; T with the natural nx3 in the null model, as he read it).
- **Conditional on nx3 on both witnesses:** yes for all three forms. Without nx3 in the null model they read -4.31 / -3.33, -3.45 / -3.44, +3.74 / +4.16.
- **The null fits are converged.** My own recomputation of the log-likelihood equals the kernel's to three decimals; the largest gradient over the null model's columns is 2e-4 to 2e-3.
- **New constructions** (`c05_newcons.txt`): all 24 whole-feed values and the 12 cold-row values of `newcons/out/gates.txt` are reproduced (largest whole-feed |cz| 2.00 on H, 2.80 on T, never the same form).

---

## 2. Density `kn1_5:z`: refuted as a survivor

**2.1 The T pass depends on the placebo draw.** [`c01_reread.txt`]

| Witness | Read | 200 within-feed permutations: sd / max / 99th percentile | Placebos beyond the read |
|---|---|---|---|
| H | -4.55 | 0.94 / 2.60 / 2.39 | 0 of 200 |
| T | -3.23 | 1.09 / 3.67 / 2.95 | 1 of 200 |
| N | -3.36 | 1.00 / 2.50 / 2.34 | 0 of 200 |

The pre-declared rule asked for |cz| above every placebo of the witness. The explorer's 120 placebos peaked at 2.99 against a read of 3.22; in his gate run the 20 placebos of this very column had sd 1.43 (`rescue/out/r10_gates_main.txt`). On my draw the clause fails.

**2.2 A structured placebo reads like density on val.** A within-feed permutation destroys every relation between the column and the candidate. Density is a function of the candidate's content vector, which the ranker reads raw and the lens's control block does not hold. I scored content columns that carry no catalogue or exposure information by construction, entered exactly as `kn1_5:z` is. [`c02_structured.txt`, 100 draws each]

| Placebo family | H: mean / sd / max | T: mean / sd / rms / max | T: share at or above 2.5 / above density's 3.23 |
|---|---|---|---|
| Cosine to a random direction | +0.06 / 1.08 / 2.82 | -0.40 / 1.09 / 1.16 / 3.52 | 0.03 / 0.01 |
| Cosine to a random train item | +0.55 / 0.90 / 2.45 | **-1.07 / 1.14 / 1.57 / 3.62** | **0.12 / 0.03** |
| Random effect per cluster, 4096 / 1024 / 256 | sd 0.96 / 1.06 / 1.13, max 2.91 | sd 0.89 / 1.07 / 1.06, max 3.35 | at most 0.01 / 0.01 |

- On H nothing comes near -4.55, so the holdout read stands.
- On T "how similar is this item to an arbitrary item" is itself negative on average, and 12% of such columns pass 2.5. Density on T is 1.9 sd from that family's mean. The val read cannot be told apart from a generic property of the content space.
- The sign of that generic effect is opposite on H (+0.55), so density on H is not it.

**2.3 Not shown at the private mix.** [`c06_private.txt` part B; the pmix weights keep 0.67 of the cz of a homogeneous effect, 6,389 effective feeds of 14,423]

| Form | T all | T feed cold share low / mid / high | T pmix-public | T pmix-private | T pmix-private x private lag mix | N pmix-private |
|---|---|---|---|---|---|---|
| `kn1_5:z` | -3.23 | -2.11 / -2.54 / -0.95 | -1.58 | **-0.84** | -1.17 | -1.05 |
| `ns2_5:z` | -3.51 | -1.75 / -1.33 / -2.98 | -2.81 | **-2.57** | -2.55 | -1.73 |
| `bt5:rk` | +4.18 | +3.28 / +0.81 / +3.23 | +3.05 | **+2.77** | +2.85 | +1.77 |
| `nx_rp3_z` (null without nx3) | +6.46 | +5.14 / +4.28 / +1.54 | +3.73 | +2.14 | +2.80 | +3.66 |

A homogeneous effect would read -2.16 for density at the private weights; it reads -0.84. One val day carries it: slot 16 -1.49, slot 18 -3.11.

**2.4 What stays true.** No leak (label-free, re-derived exactly, 2.6); parity is clean (PSI at most 0.001, no joint shift, section 5); the holdout read is real. As an input it would do no harm. It has no support where the board is decided.

---

## 3. `ns2_5:z`: clean, and too small for its own test

**3.1 Parity passes for the column itself.** `PREDECLARED.txt` fixed the non-default indicator of the piece as n_s1 + n_s2 > 0, which belongs to the ratio `X3_5`. With the entered column's own indicator, n_s2 > 0, and with no row-share exemption: [`c06_private.txt` part A]

| Train rows (pool feeds) against | dZ (bar 0.05) | Largest bucket, row share >= 0.05 (bar 0.10) | Largest of all 7 buckets |
|---|---|---|---|
| slots 11-13 vs private | 0.0302 | 0.0734 | 0.0734 (10-24 d) |
| slots 11-13 vs public | 0.0208 | 0.0345 | 0.0345 (24-76 d) |
| slots 6-8 vs val | 0.0351 | 0.0501 | 0.0952 (7-10 d) |

PSI of the entered form, pool rows against val / public / private: 0.006 / 0.107 / 0.164 (`rescue/out/r10_gates_main.txt`; bar 0.25). This is my reading after the fact and counts as a look; by the letter of the pre-declaration the piece failed by 0.009.

**3.2 The read is specific to the column.** `ns2_5:z` has R2 0.46-0.53 with the ranker's popularity inputs, which the lens controls only linearly. I permuted the raw column inside the feed among rows of the same popularity stratum (8 within-feed bins of the OLS prediction of the form from `it_hist_all_luo`, `it_hist_et1_luo`, `seq_cv_invd20`, `seq_cv_pop`). The placebo keeps the whole relation to popularity (correlation 0.47-0.49 with the true form). [`c07_condperm.txt`, 100 draws]

| Witness | Read | Placebo mean / sd / extremes | (read - mean) / sd | Beyond the read |
|---|---|---|---|---|
| H | -4.50 | -0.11 / 0.89 / -2.41 .. +2.56 | -4.93 | 0 of 100 |
| T | -3.51 | -0.36 / 0.84 / -2.96 .. +1.37 | -3.77 | 0 of 100 |

**3.3 Other attacks it survived.**
- Within-feed placebo: 0 of 200 beyond on H, T and N (`c01_reread.txt`).
- Test-like half of the holdout alone (column zeroed elsewhere): -2.85, 0 of 200 placebos beyond, cz / sd -2.73.
- By lag on H (2 / 3 / 4 / 7 / 8 / 9 d): -1.84 / -3.61 / -0.11 / -1.88 / -2.73 / -1.44. On T (7 / 8 / 9 / 10 d): +0.10 / -1.66 / -1.41 / -3.10.
- Given the other two forms: -3.78 (H), -3.35 (T), -2.58 (N).

**3.4 Limits.**
- It is the most redundant of the three with what the model reads, and more so on test rows: correlation with `it_hist_et1_luo` 0.62-0.64 on train rows against 0.71-0.72 on val, public and private rows (`c03_parity.txt` S1).
- A domain classifier separates train lag-view rows from test rows a little better with it: AUC +0.006 against private rows and +0.008 against public rows on top of the inputs and nx3, where nx3 itself added +0.012 and +0.021. On the val-like comparison it adds +0.009 where nx3 added +0.003 (`c03_parity.txt` S3).
- Size: 0.00042 nats per feed on T, 0.00032 on H, one degree of freedom, in sample.

---

## 4. `bt5:rk`: not ready

**4.1 Most of the read is exposure, not clicks.** [`c07_condperm.txt`] `bt5` is a Bradley-Terry fit on the exploration feeds of the window. I permuted theta inside the feed among rows with the same exposure in that window (8 within-feed bins of `imp5`, a label-free count the ranker does not read). The placebo keeps which items were shown and how often, and shuffles the click part.

| Witness | Read | Placebo within exposure strata: mean / sd / extremes | (read - mean) / sd | Beyond the read |
|---|---|---|---|---|
| H | +4.17 | **+2.56** / 0.79 / +0.42 .. +4.30 | +2.04 | 2 of 100 |
| T | +4.18 | **+2.34** / 0.84 / -0.26 .. +4.71 | +2.20 | 1 of 100 |

Inside popularity strata the placebo is near zero (+0.50 / +0.57), so this is not the ranker's popularity inputs. The click part is positive on both witnesses and below 2.5 on both. The marginal within-feed AUC of `bt5` for the click is 0.49-0.50 (`c04_rederive.txt` C(ii)). The piece is an exposure / novelty contrast with a click component, the same family as `bt5_seen` (-4.22 / -3.04) and `never5` (+4.66 / +3.61).

**4.2 A parity gap against private rows sits in a bucket the gate exempts.** [`c06_private.txt` part A; the gate counts a bucket only if it holds 5% of the view's rows]

| Seen share, train rows (pool) against | dZ | Largest bucket with row share >= 0.05 | Largest of all buckets |
|---|---|---|---|
| slots 11-13 vs private | 0.0167 | 0.0133 | **0.2273** at 7-10 d (0.370 against 0.143; 4.8% of rows) |
| slots 11-13 vs public | 0.0172 | 0.0224 | 0.1027 at 7-10 d (0.370 against 0.473) |
| slots 6-8 vs val | 0.0257 | 0.1020 | 0.1020 at 3-7 d |

Cause: test-like train rows are 7 / 8 / 9 days behind their newest source day (0.48 / 0.35 / 0.17), private rows 8 / 9 / 10 days (0.24 / 0.38 / 0.38); lag 10 does not exist on train rows (`rescue/out/r14_lagtable.txt`). The walk of nx3 has 0.032 in the same bucket. The defect touches 1.1% of rows; it is the reason the piece cannot be called "at parity against private".

**4.3 The holdout does not show it at the private lags.** H by lag 7 / 8 / 9 d: +2.02 / +0.95 / +0.34. Test-like half of H alone: +2.06, 10 of 200 placebos beyond. T by lag 7 / 8 / 9 / 10 d: -0.26 / +1.94 / +1.52 / +3.91, so val does not show a decay. Private-mix weights on T: +2.77.

**4.4 No leak.** Section 6.

---

## 5. Parity: joint behaviour with the inputs the model already reads

Label-free, `c03_parity.txt`. TRv / TRt = pool rows of slots 6-8 / 11-13.

| Check | `kn1_5:z` | `ns2_5:z` | `bt5:rk` | `nx_rp3_z` (accepted) |
|---|---|---|---|---|
| OLS R2 on the four popularity / co-visitation inputs + age: TRv / TRt / val / public / private | 0.07 / 0.07 / 0.06 / 0.05 / 0.05 | 0.48 / 0.48 / 0.51 / 0.53 / 0.53 | 0.16 / 0.17 / 0.18 / 0.20 / 0.20 | 0.29 / 0.26 / 0.34 / 0.34 / 0.34 |
| P(non-default given `it_hist_all_luo` > 0): TRt / public / private | - | 0.357 / 0.432 / 0.436 | 0.349 / 0.452 / 0.437 | 0.203 / 0.313 / 0.317 |
| P(non-default given `seq_cv_invd20` > 0): TRt / private | - | 0.609 / 0.634 | 0.572 / 0.588 | 0.772 / 0.9996 |
| Domain classifier, added AUC over inputs + nx3: vs private / public / val-like vs val | +0.0005 / +0.0013 / +0.0007 | +0.0058 / +0.0084 / +0.0086 | +0.0024 / +0.0082 / +0.0036 | (+0.0115 / +0.0207 / +0.0027 over the inputs) |

- None of the three has a joint shift larger than the one nx3 was accepted with, except `ns2_5:z` on the val-like comparison.
- The classifier starts at AUC 0.89-0.93 from the existing inputs alone: `it_hist_all_luo` is non-zero on 0.74 of test-like pool rows against 0.57 / 0.55 of public / private rows. That gap is the feed mix of train and exists today.
- Not checked by anyone: all has = 1 train rows. Arrays exist for holdout and pool feeds only. nx3 read PSI 0.26-0.35 there.

---

## 6. Leak

No path found. [`c04_rederive.txt`; my own polars / numpy path from the raw arrays, none of the explorer's kernels]

| Column | Re-derivation | Mismatches | Smallest (row day - newest contributing source day): train / val natural / twin, test |
|---|---|---|---|
| `kn1_5` | 750 rows over the five query sets, index rebuilt per window from the raw candidate log of 6 random clusters | 0; largest difference 1.7e-07 | 2 d / 3 d / 7 d |
| `ns2_5` | 1,000 rows from the raw history arrays, own user removed | 0 | 2 d / 3 d / 7 d |
| `bt5` seen flag | 2,000 rows: item is a candidate of a source feed of the window | 0; no row with theta != 0 while unseen | by construction |

- **Row's own labels.** If a pool row's own feed entered its value, `bt5` would predict the click far better on pool rows (their feeds are sources of other rows' windows) than on holdout rows (never a source). Mean within-feed AUC: pool 0.5006 ± 0.0013 (slots 6-8) and 0.4975 ± 0.0011 (slots 11-13); holdout 0.4979 ± 0.0028 and 0.4927 ± 0.0023. The same for `coec5`: 0.5123 / 0.5028 against 0.5054 / 0.5016. No sign of a leak.
- **Same user's later snapshots.** Every source day is before the cut and the cut is before the row's day; `ns2_5` also removes the row's own user. A same user's EARLIER labelled feed can enter `bt5`; that is causal and exists on test rows too.
- **Val / test labels.** No builder opens them (code read of `r01_item.py`, `r02_density.py`, `r03_surface.py`). Val and test rows read train feeds of slots 7, 8, 11, 12, 13 only.
- **Not audited:** a full-train build does not exist. A label statistic on all 115 M has = 1 rows would need the per-feed causality assertion of `r01_item.py` extended to every train feed.

---

## 7. Multiplicity

| | Rescue | New constructions | Critic (information only; nothing here can promote) |
|---|---|---|---|
| Constructions | one view (D5), 7 pieces, 21 column-forms, fixed in `PREDECLARED.txt` | 6, fixed in `PRELIST.txt` | none new |
| Deciding reads | 42 + 4 | 24 | 0 |
| Other cz / NDCG values | about 480 | 468 | about 620 cz, about 130 NDCG deltas |
| Placebo scores | about 3,100 | 270 | 2,600 within-feed, 1,000 structured, 800 conditional, 48 nested |

- **Noise is not the explanation on the holdout.** Three independent forms at |cz| 4.2-4.6 with 0 of 200 placebos beyond; the family-wise chance of one both-witness pass among 21 forms is 0.0016.
- **The threshold is calibrated only against unstructured placebos.** Within-feed permutations give sd 0.94-1.09 on every witness, so 2.5 is the right bar for a column unrelated to the candidate. For a content-derived column on val it is not (2.2). For the two popularity-type columns the conditional placebo is centred near zero inside popularity strata, and at +2.3..+2.6 inside exposure strata for `bt5:rk` (3.2, 4.1).
- **Forms were chosen after both witnesses were seen:** `ns2_5` instead of the pre-declared primary `X3_5` (+2.15 / +2.34, not shown), the rank form of `bt5` instead of the z form (+0.98 / +1.83, not shown). Both were inside the 21 pre-declared forms.
- **Post hoc looks of mine that read above 2.5 on both witnesses and are NOT promoted:** the lag-view cold flag (+2.79 / +2.83, `c11_segment.txt`) and the segment forms of `cos_u` (section 9).

---

## 8. Redundancy and size

**8.1 Three independent pieces.** [`c01_reread.txt`]

| Form | Alone H / T | Given `kn1_5:z` | Given `ns2_5:z` | Given `bt5:rk` | Given the other two | Coefficient in the joint fit H / T / N |
|---|---|---|---|---|---|---|
| `kn1_5:z` | -4.55 / -3.23 | - | -4.26 / -3.20 | -4.37 / -3.17 | -4.13 / -3.14 | -0.031 / -0.027 / -0.028 |
| `ns2_5:z` | -4.50 / -3.51 | -4.17 / -3.48 | - | -4.07 / -3.38 | -3.78 / -3.35 | -0.026 / -0.031 / -0.024 |
| `bt5:rk` | +4.17 / +4.18 | +3.96 / +4.13 | +3.72 / +4.07 | - | +3.55 / +4.02 | +0.092 / +0.129 / +0.099 |

Joint fit of the three: likelihood ratio 50.9 (H), 38.1 (T), 29.8 (N) on 3 df = 0.00085 / 0.00132 / 0.00103 nats per feed in sample, against 0.00044 / 0.00154 / 0.00166 for nx3. The cold-row columns of the other explorer are independent of the trio (N3: +2.87 alone, +3.14 given the trio on T; `c05_newcons.txt`).

**8.2 Size in NDCG, read directly.** Both arms re-weight the control block; 5 user-grouped folds inside a witness, or fitted on one witness and read on the other. Paired per-feed mean ± 1 SE. [`c09_ndcg.txt`]

| Set | Nested on T: all / pmix-private | Nested on N: all / pmix-private | Nested on H: lag-view feeds / slots 11-13 | H -> T: all / pmix-private | T -> H: lag-view feeds / slots 11-13 |
|---|---|---|---|---|---|
| nx3 over the null without it | +0.00009 ± 0.00044 / -0.00059 ± 0.00066 | +0.00009 ± 0.00045 / -0.00060 ± 0.00066 | +0.00053 ± 0.00026 / +0.00079 ± 0.00033 | -0.00025 ± 0.00032 / -0.00082 ± 0.00047 | +0.00049 ± 0.00038 / +0.00059 ± 0.00052 |
| trio over null + nx3 | +0.00044 ± 0.00039 / +0.00043 ± 0.00060 | +0.00048 ± 0.00034 / +0.00051 ± 0.00052 | -0.00020 ± 0.00030 / -0.00042 ± 0.00038 | +0.00044 ± 0.00036 / +0.00057 ± 0.00056 | -0.00024 ± 0.00032 / -0.00058 ± 0.00044 |
| `ns2_5:z` + `bt5:rk` | +0.00046 ± 0.00036 / +0.00018 ± 0.00055 | +0.00041 ± 0.00032 / +0.00060 ± 0.00049 | -0.00033 ± 0.00027 / -0.00054 ± 0.00034 | +0.00043 ± 0.00033 / +0.00089 ± 0.00049 | -0.00023 ± 0.00030 / -0.00051 ± 0.00041 |
| `ns2_5:z` | +0.00087 ± 0.00029 / +0.00083 ± 0.00043 | +0.00051 ± 0.00025 / +0.00071 ± 0.00040 | -0.00011 ± 0.00024 / -0.00043 ± 0.00029 | +0.00030 ± 0.00030 / +0.00074 ± 0.00043 | +0.00012 ± 0.00023 / -0.00008 ± 0.00031 |

- **Val reads the round-2 columns positive, the holdout negative or zero; nx3 is the reverse.** Most cells are within 2 SE of zero; the largest are `ns2_5:z` nested on T (3.0 SE, next point) and nx3 on the test-like half of H (2.4 SE).
- **The `ns2_5:z` read on val is not as strong as it looks.** With a second fold assignment it is +0.00033 ± 0.00027 on T and +0.00008 ± 0.00023 on N. Twelve conditional placebos read mean +0.00013, sd 0.00014, max +0.00040 on T (`c10_ns2_ndcg_placebo.txt`). So: +0.0003 to +0.0006 on val, about the size of the best placebo.
- **This instrument is weak** (the record: it recovers 10-40% of a trained input and is "not identified" on val). I use it for sign and order of magnitude only. The owner's instruction of 11-15 applies to the sign pattern: these are popularity-light-type columns (two negative coefficients on popularity-shaped columns), where val and the private mix led and the holdout was wrong.
- A first version that added the columns to the pool alone (`c06_private.txt` part C) read -0.0010 ± 0.0004 on the holdout. It mixes the columns with the control block's own re-weighting and is replaced by the table above.

**8.3 Forecast.**

| Arm of the nx3 form (8 hard-label members against controls) | Central effect, val-like inputs | Holdout | P(pass stage A) |
|---|---|---|---|
| `ns2_5:z` alone | +0.0001 (nats) to +0.0003 (nested NDCG) | about 0 | 0.05 |
| `ns2_5:z` + `bt5:rk` | +0.0002 | about 0, sign uncertain | 0.07 |
| trio | +0.0003 | about 0, sign uncertain | 0.07 |
| For comparison, nx3 of record | +0.0003 | +0.0002 | 0.15 |

The rule's false pass is 0.008-0.020 and its power at +0.0003 is 0.09-0.13. The holdout clause (z >= +1) is what holds my numbers below nx3's.

---

## 9. New constructions: confirmed nothing, and what the cold-row effect is

**9.1 The explorer's "VAL ONLY" stands**, and the effect is not a new input. [`c08_coldsplit.txt`; z re-computed among the cold rows / the warm rows of the feed, 0 elsewhere; cold = item in no admitted source sequence of the row's view]

| Column | H: cold / warm | T: cold / warm | T at pmix-private: cold / warm |
|---|---|---|---|
| N3 `aud_cell` | +0.71 / -3.18 | +2.87 / -2.89 | +2.95 / -2.10 |
| N2 `cw_cell` | -0.11 / -2.56 | +2.45 / -2.41 | +2.33 / -2.25 |
| own input `cos_u` | +1.96 / -3.78 | +4.19 / -2.99 | +3.67 / -2.65 |
| own input `seq_max_all` | +1.12 / -2.92 | +4.21 / -3.46 | +4.90 / -3.78 |
| own input `maxsim32` | +1.11 / -3.34 | +3.11 / -2.40 | +4.21 / -2.90 |

The cell columns are content affinity, and content affinity the model already reads shows the split more strongly. (N2 is scaled here by the view's own median; with the explorer's constant it reads +0.02 / +2.49 on cold rows, `c05_newcons.txt`.)

**9.2 The split belongs to the lag-view marking, not to anything the model can see.** [`c11_segment.txt`] With cold defined by the model's own input (`it_hist_all_luo` = 0) the split is gone: `cos_u` -0.39 / -1.47 on H, +1.67 / -1.91 on T; 2-df likelihood ratio 3.4 / 3.6 against 15.2 / 19.3 for the lag-view definition. The lag-view cold flag itself reads +2.79 (H), +2.83 (T), +1.94 at pmix-private. Rows that are cold in the lag view and warm for the model are items that became active after the cut. This is the exposure / novelty contrast of section 4.1 seen from the history side.

**9.3 Consequence for the nx3 read [judgement].** nx3 puts every lag-cold row at the feed minimum, so an nx3 member can learn part of this marking from the walk's zero pattern. A positive VALUE read of nx3 (values permuted within the feed) may therefore owe something to the marking and not only to the walk. Nothing needs building for it now.

---

## 10. Everything I ran

All in `reports/INPUTS_1116/round2/critic/`; outputs `out/<name>.txt` (and the same text as `.stdout`).

| Script | Content | Run time |
|---|---|---|
| `cc_common.py` | shared: my score path, placebo order, forms | - |
| `c01_reread.py` | re-read on H, T, N; slices; 200 placebos; redundancy | 686 s |
| `c02_structured.py` | structured placebos for a content-derived column | 301 s |
| `c03_parity.py` | label-free joint behaviour, domain classifier | 82 s |
| `c04_rederive.py` | re-derivation from raw arrays; own-feed leak test | 14 s |
| `c05_newcons.py` | re-read of the six new constructions | 69 s |
| `c06_private.py` | all-bucket parity; private-mix and cold-share slices; column-only transfer (superseded by c09) | 82 s |
| `c07_condperm.py` | placebo inside popularity strata and inside exposure strata | 256 s |
| `c08_coldsplit.py` | cold / warm split of own inputs | 59 s |
| `c09_ndcg.py` | nested and cross-witness NDCG with the control block in both arms | 361 s |
| `c10_ns2_ndcg_placebo.py` | fold noise and placebo of the nested NDCG read | 205 s |
| `c11_segment.py` | lag-view cold against own-input cold | 98 s |

- `c09_ndcg.py` was edited once before its only run (two of six sets removed to save time).
- One waiting shell of mine could never finish (its `pgrep -f` pattern matched its own command line). I stopped it by its exact PID (1467298) after reading `/proc/1467298/cmdline`. No other process was signalled; none of mine is left.
- Disk: 1.2 MB.

---

## 11. Limits

- The lens is linear in the entered form and in sample. The NDCG reads of 8.2 have an SE of 0.0003-0.0006 and recover a minority of a trained input's value on record. Nothing was trained.
- No full-train array exists for any round-2 column. Parity on all has = 1 train rows is unread.
- Both val witnesses use the same 14,423 feeds and labels; T and N are one sample in two views.
- The pmix weights re-weight feed size and cold share only; the private lag mix was added by re-weighting the twin's lag groups (lag-7 feeds dropped).
- The structured placebo of 2.2 has 100 draws per family; "3% beyond" is 3 columns.
- My judgement on the size uses two positive controls of record, both fresh on train rows; no lag-view input has a measured trained value yet. nx3's stage A will be the first.
