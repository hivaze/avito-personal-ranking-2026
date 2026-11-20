# Lane A: repaired click-strength input: NO_GO

(Written by the lead on 2026-11-16 from the structured result of workflow a workflow run: the lane could not write report files itself. Text below is the lane's own, unedited.)

NO-GO: the repaired click-strength input does not deserve a trained test. Nothing was built for training, nothing installed, no GPU used.

REPORT.md WAS NOT WRITTEN. The harness refused the write ("Subagents should return findings as text, not write report files"), and I did not work around it. The full report text is below; the rule, scripts and every output file exist under <repo>/reports/NEXT_1116/clickstrength/.

# Lane A: a repaired click-strength input

Run 2026-11-16 14:26-14:50 UTC, CPU only, at most two 12-thread processes. No process signalled. Nothing written outside reports/NEXT_1116/clickstrength/ (checked with find -newer on data/feats, nn/feat_cache, reports/INPUTS_1116, nn_x/w12/nx3).

Terms: cz = score statistic of condlib.py, N(0,1) when a column adds nothing; null model = pool of 16 held-out wide students + control block + nx3, as in rescue.md / CRITIC2.md. H = the 30,000 holdout feeds, column 0 outside the 15,225 lag-view feeds. T = val rows with the column built at the private lag. N = val rows with the natural column.

## 0. The answer

The repair itself worked: exposure is out of the read, and parity against private rows holds with no exemption. What is left, the click part, is too small: cz +1.2 to +2.2 on the two deciding witnesses against a bar of 2.74, and zero or negative in NDCG on val.

| Gate | F1 cs3a_c (all feeds) | F2 cs3x_c (exploration feeds) | Result |
|---|---|---|---|
| g1 cz on H / T (Holm bar 2.74 for the better form, then 2.50) | +1.22 / +1.39 | +1.85 / +2.17 | MISS |
| g1 within-feed permutations at or beyond the read, H / T (must be 0 of 100) | 20 / 9 | 5 / 3 | MISS |
| g2 exposure-only placebo mean, H / T, strata of window exposure (bar 1.5) | -0.10 / -0.04 | +0.14 / -0.16 | ok |
| g2 same, strata of the form's own expected clicks | +0.01 / +0.15 | +0.36 / +0.27 | ok |
| g2 click part = (read - placebo mean) / sd, smaller of the two placebos, H / T (bar 2.5) | +1.24 / +1.34 | +1.61 / +1.91 | MISS |
| g3 largest of ALL 7 age buckets, test-like train rows vs private / val-like vs val (bar 0.10, no exemption) | 0.042 / 0.028 | 0.030 / 0.033 | PASS |
| g3 PSI pool rows vs val / public / private (bar 0.25) | 0.018 / 0.010 / 0.026 | 0.016 / 0.004 / 0.019 | PASS |
| g4 nested NDCG: H lag view / H test-like half / T / N (each >= 0) | +0.00010 / +0.00020 / -0.00006 / -0.00015 | -0.000001 / -0.00010 / -0.00007 / -0.00006 | MISS |
| g5 leak scans | 0 violations | 0 violations | PASS |

- Not a holdout-only rejection: val misses g1, g2 and g4 by its own numbers. The holdout-only clause of the rule is not reached.
- At a plain 2.5 without Holm nothing passes either; the largest deciding read is F2 on T, +2.17.
- Forecast written before any number: P(GO) 0.04.
- Why [post hoc]: the same statistic built fresh (three days up to and including the row's own day; not admissible as an input) reads cz +6.94 on the holdout for F1 and +2.91 for F2. At the lags parity requires it reads +1.22 / +1.85, and about zero on the holdout's test-like half (+0.37 / -0.44). The click part decays within days; private rows are 8-10 days behind their newest source day.

## 1. What was built

Rule: RULE.txt, written 2026-11-16T14:29:06Z (md5 5b526db8...), before any array of the new forms existed; every script prints that md5.

View "P3": three consecutive source days for every row of every split.

| Rows | Lag to the newest source day | Realised (feeds) | Target |
|---|---|---|---|
| Train Sep 12-14 | 3 d on Sep 12; 5 d with p 0.668 else 3 d on Sep 13-14 | pool 0.509 / 0.491 | natural val 0.523 / 0.477 |
| Train Sep 17-19 | Sep 17: 8 d (0.390) or 9 d; Sep 18-19: 8 / 9 / 10 d = 0.161 / 0.251 / 0.588 | pool 0.241 / 0.377 / 0.382 | private 0.242 / 0.378 / 0.379 |
| Val, test natural | window Sep 17-19 | val 3 / 5 d; public 7 / 8 d; private 8 / 9 / 10 d | - |
| Val twin (T) | Sep 22: 8 d (0.390) or 9 d; Sep 24: 10 d | 0.205 / 0.318 / 0.477 | private (more 10 d by construction) |

- Three days is the largest depth at which a 10-day lag exists on train rows (data start Sep 6); that was the defect of the five-day view.
- Sources: train feeds of the window that are not holdout feeds; the window ends at least 3 days before the row.
- Statistic per item: clicks c against expected clicks e (the feed's click count shared out by position weight); x = log((c + a) / (e + a)) minus its no-effect expectation, 0 for an unseen item. Entered value = x minus its mean over the rows of the same feed and exposure bin (the critic's 8 bins). Fixed scale, no rank.
- F1: all feeds, a = 9.41. F2: exploration feeds, a = 50 (the rule's cap).
- Rows built: holdout 4,630,966 (2,366,505 with a view), pool 10,253,859 (69,171 feeds), val 1,389,971 (natural and twin), test 2,904,806. No full-train build.
- Tie to the record [post hoc]: F2's item value correlates 0.83 / 0.82 with the stored bt5 theta on val / test rows; F1's 0.75 / 0.74 with log coec5.

## 2. g3 parity: PASS (label-free, read before any witness label; out/c02_parity.txt)

| Clause | Bar | F1 | F2 |
|---|---|---|---|
| Test-like pool rows vs PRIVATE: dZ / largest of all 7 buckets | 0.05 / 0.10, no exemption | 0.0079 / 0.0417 (7-10 d: 0.330 vs 0.289) | 0.0121 / 0.0300 |
| Val-like pool rows vs VAL: dZ / largest of all 7 buckets | 0.05 / 0.10, no exemption | 0.0154 / 0.0283 | 0.0126 / 0.0332 |
| Test-like pool rows vs PUBLIC: dZ / largest bucket with row share >= 0.05 | bar of record | 0.0327 / 0.0427 | 0.0335 / 0.0275 |
| Same vs public over all buckets | no bar (declared) | 0.478 at 7-10 d | 0.317 at 7-10 d |
| PSI pool rows vs val / public / private | 0.25 | 0.0182 / 0.0096 / 0.0262 | 0.0155 / 0.0042 / 0.0191 |
| Lowest non-default share of a test day | half of val's largest | 0.389 (bar 0.284) | 0.212 (bar 0.164) |

- Seen share of 7-10 day old items, exploration form: 0.155 on test-like train rows against 0.143 on private rows (five-day view on record: 0.370 against 0.143).
- Public and private rows differ from each other by 0.52 (F1) / 0.33 (F2) in that bucket, so public cannot be matched at the same time; the view targets private. Declared before any value.
- Not read: all train rows with a view (limit J0 of nx3).

## 3. g1 and g2: MISS (out/c03_info.txt)

Instrument checks in the same run: nx3 reads 0.000 under the null that holds it and +4.98 / +6.46 / +6.78 without it, the record's values.

| | F1 H / T / N | F2 H / T / N |
|---|---|---|
| cz (N is information) | +1.22 / +1.39 / +2.11 | +1.85 / +2.17 / +1.68 |
| 100 within-feed permutations, sd | 1.01 / 0.81 / 0.98 | 0.95 / 0.99 / 0.99 |
| Placebo inside exposure bins: mean, sd | -0.10, 0.96 / -0.04, 1.02 / -0.09, 1.02 | +0.14, 1.06 / -0.16, 1.00 / -0.01, 0.94 |
| Placebo inside bins of expected clicks: mean, sd | +0.01, 0.98 / +0.15, 0.93 / -0.56, 0.94 | +0.36, 0.93 / +0.27, 1.00 / -0.41, 1.06 |
| Click part against the two placebos | 1.37, 1.24 / 1.40, 1.34 / 2.16, 2.84 | 1.61, 1.61 / 2.34, 1.91 / 1.79, 1.97 |
| Nats per feed | 0.000024 / 0.000064 / 0.000159 | 0.000054 / 0.000166 / 0.000094 |

- Exposure is out: every exposure-only placebo is centred within 0.4 of zero on H and T (old rank form: +2.56 / +2.34).
- The click part is positive on all six reads and reaches 2.5 on none, in line with the critic's +2.0 / +2.2 sd for the old column.
- Size: F2 carries about one eighth to one ninth of nx3's information (nx3: 0.00044 / 0.00154 nats), and nx3 showed nothing when trained.
- Where it sits: H val-like / test-like half F1 +1.30 / +0.37, F2 +2.87 / -0.44. H by lag 3 / 5 / 8 / 9 / 10 d, F2: +2.71 / +1.28 / +1.24 / -1.28 / -0.41. T at private-mix weights +0.12 / +1.57; N at private-mix weights +0.45 / +1.83.

## 4. g4 direct NDCG: MISS (out/c04_ndcg.txt; 5 user-grouped folds, both arms re-weight control block and nx3)

| Cell | F1 | F2 | Yardstick nx3 |
|---|---|---|---|
| H lag-view feeds | +0.00010 ± 0.00011 | -0.000001 ± 0.00012 | +0.00053 ± 0.00026 |
| H test-like half | +0.00020 ± 0.00014 | -0.00010 ± 0.00011 | +0.00079 ± 0.00033 |
| T | -0.00006 ± 0.00014 | -0.00007 ± 0.00015 | +0.00009 ± 0.00044 |
| N | -0.00015 ± 0.00019 | -0.00006 ± 0.00013 | +0.00009 ± 0.00045 |

The yardstick column reproduces critic/out/c09_ndcg.txt to the digit. Every cell of the two forms is within 1.5 SE of zero.

## 5. g5 leak scans: PASS (out/c05_leak.txt)

- Causality by real timestamps, every feed with a view: 0 feeds with a source at or after (row time - 2 d); smallest distance 2.001 d (holdout, pool, val), 7.003 d (twin), 6.001 d (test).
- Re-derivation by a second code path (numpy / polars, none of the builder's kernels) on 14,485 rows of 25 feeds per query set: 0 mismatches of impressions, clicks, expected clicks and entered value; position weights and shrinkage constants equal.
- Holdout feeds are never a source; admitting them changes 1,154 of the 14,485 rows, so the check can fail. The builder opens train labels only.
- Own-feed test (within-feed AUC, pool rows minus holdout rows): at most +1.9 SE (bar 3).
- A deliberately leaky build (window includes the row's own day) is caught: AUC 0.93 / 0.99 on pool rows against 0.52 on holdout rows (+125 to +161 SE), and the timestamp check flags every feed.
- Post hoc (out/c06_posthoc.txt): the stored bt5 array reads +4.17 (rank) / +0.98 (z) in my harness, the record's values.

## 6. Deviations

- First launch of c03_info.py stopped on an exact-equality assertion (float64 vs float32 statistic, difference 1.2e-07) before any form was scored (out/c03_info.failed1.txt). Fix: tolerance 1e-6; the read is on the stored array.
- F2's shrinkage sits at the cap: the declared spread estimate is negative for exploration feeds (tau2 -0.29) because it uses a Poisson variance. By the rule a = 50; not changed after seeing it. F2 is therefore close to the plain excess c - e.
- Order: the rule put the label-free part of g5 before g1; c05_leak.py ran while c03_info.py was running. The builder's own causality and holdout-source assertions did precede g1.
- The entered value is multiplied by a constant (25 / 300) before the float32 cast; cz is identical with and without it.
- Looks: 20 deciding reads as declared (4 cz, 8 placebo means, 8 NDCG cells); about 60 further cz, 1,800 placebo scores and 55 NDCG cells as information; post hoc 8 cz and 40 placebo scores.

## 7. Limits

- The lens is linear and in-sample; nothing was trained.
- Two forms only, fixed in advance; other shrinkage, depth or source choices were not tried.
- T and N are the same 14,423 val feeds in two views; the twin holds more 10-day rows than private (0.48 against 0.38).
- An earlier labelled feed of the same user can be a source, on train and test rows alike; its share was not measured.
- The fresh control uses same-day feeds, later ones included: one comparison, not a decay curve.
- Arrays named *_LEAK in out/arr/ are the deliberately leaky build of the leak scan and must never be used as inputs.

## 8. Files (all in <repo>/reports/NEXT_1116/clickstrength/)

RULE.txt; cs_common.py; c01_build.py -> out/c01_build.log (53 s); c02_parity.py -> out/c02_parity.{txt,json} (6 s); c03_info.py -> out/c03_info.{txt,json} (319 s); c04_ndcg.py -> out/c04_ndcg.{txt,json} (290 s); c05_leak.py -> out/c05_leak.{txt,json} (18 s); c06_posthoc.py -> out/c06_posthoc.txt (54 s); out/p00_labelfree_counts.txt; out/arr/<set>__<name>.npy (75 witness-row arrays, 1.3 GB, not installed anywhere).
Run form: cd into the directory, CUDA_VISIBLE_DEVICES="" and the five thread variables at 12, <repo>/.venv/bin/python -u <script>.

## 9. Text for the lead to record

=== <UTC at copy> LANE A "repaired click strength" READ (reports/NEXT_1116/clickstrength/; rule RULE.txt of 2026-11-16T14:29:06Z, md5 5b526db8..., written before any array or witness number) ===
 VERDICT: NO-GO. Nothing built for training, no trained test proposed, no GPU run.
 CONSTRUCTION: view P3 = 3 consecutive source days for every row; lag 3 / 5 d on Sep 12-14 (val profile), 8 / 9 / 10 d on Sep 17-19 (0.241 / 0.377 / 0.382 against private 0.242 / 0.378 / 0.379). Statistic = clicks against expected clicks with shrinkage, centred inside the exposure bins of the critic's placebo, fixed scale. Two pre-declared forms: F1 cs3a_c (all feeds), F2 cs3x_c (exploration feeds); Holm over the two.
 g3 PARITY PASS, no exemption: largest of all 7 age buckets against private 0.042 / 0.030, against val 0.028 / 0.033 (bar 0.10); 7-10 d bucket of the exploration form 0.155 against 0.143 private (five-day view: 0.370 against 0.143). PSI <= 0.026. Public cannot be matched together with private in that bucket (declared; bar of record passes).
 g1 MISS: cz holdout lag view / private-like stale val: F1 +1.22 / +1.39, F2 +1.85 / +2.17 (bar 2.74 for the better form, 2.50 for the other); natural val +2.11 / +1.68.
 g2 exposure-only placebos centred within 0.4 of zero on both witnesses (old rank form +2.56 / +2.34): the read IS the click part; click part 1.2-1.4 sd (F1), 1.6-1.9 sd (F2): MISS.
 g4 MISS: nested NDCG, F1 +0.00010 +- 0.00011 (holdout) / +0.00020 +- 0.00014 (test-like half) / -0.00006 +- 0.00014 (stale val) / -0.00015 +- 0.00019 (natural val); F2 -0.00000 / -0.00010 / -0.00007 / -0.00006 (SE 0.00011-0.00015).
 g5 PASS: 0 causality violations; 14,485 rows re-derived by a second code path with 0 mismatches; own-feed test within 1.9 SE; a deliberately leaky build is caught at > 120 SE.
 NOT a holdout-only rejection: val misses g1, g2 and g4 by its own numbers.
 SIZE: F2 carries 0.000054 / 0.000166 nats per feed, about one eighth to one ninth of nx3 (0.00044 / 0.00154).
 WHY (post hoc): the same statistic built fresh (three days up to the row's own day; not admissible) reads cz +6.94 on the holdout; at the parity lags +1.22, and about zero on the holdout's test-like half (+0.37 / -0.44). The click part decays within days; private rows are 8-10 d behind their newest source day.

## Gates (structured)

- g1 conditional score test, cz on lag-view holdout / private-like stale val (Holm bar 2.50 as second form; first form already missed) | F1 cs3a_c (all feeds) | H +1.22, T +1.39 (natural val +2.11, information); within-feed permutations at or beyond the read 20 / 9 of 100 | pass=False | <repo>/reports/NEXT_1116/clickstrength/out/c03_info.txt
- g1 conditional score test, cz on lag-view holdout / private-like stale val (Holm bar 2.74 on both) | F2 cs3x_c (exploration feeds) | H +1.85, T +2.17 (natural val +1.68, information); within-feed permutations at or beyond the read 5 / 3 of 100 | pass=False | <repo>/reports/NEXT_1116/clickstrength/out/c03_info.txt
- g2 exposure-preserving placebo: |mean| < 1.5 on both witnesses for both strata, and click part (read - mean) / sd >= 2.5 | F1 cs3a_c (all feeds) | placebo means H / T: -0.10 / -0.04 (exposure bins), +0.01 / +0.15 (expected-click bins): mean clause ok; click part 1.37, 1.24 (H) and 1.40, 1.34 (T): below 2.5 | pass=False | <repo>/reports/NEXT_1116/clickstrength/out/c03_info.txt
- g2 exposure-preserving placebo: |mean| < 1.5 on both witnesses for both strata, and click part (read - mean) / sd >= 2.5 | F2 cs3x_c (exploration feeds) | placebo means H / T: +0.14 / -0.16 (exposure bins), +0.36 / +0.27 (expected-click bins): mean clause ok; click part 1.61, 1.61 (H) and 2.34, 1.91 (T): below 2.5 | pass=False | <repo>/reports/NEXT_1116/clickstrength/out/c03_info.txt
- g3 parity on pool feeds: age-bucket check with no exemption vs private and vs val, PSI <= 0.25, test-day share | F1 cs3a_c (all feeds) | vs private dZ 0.0079, largest of all 7 buckets 0.0417; vs val dZ 0.0154, largest 0.0283; vs public (bar of record) dZ 0.0327, bucket 0.0427 (all-bucket 0.478, declared no bar); PSI 0.0182 / 0.0096 / 0.0262; lowest test day 0.389 vs bar 0.284 | pass=True | <repo>/reports/NEXT_1116/clickstrength/out/c02_parity.txt
- g3 parity on pool feeds: age-bucket check with no exemption vs private and vs val, PSI <= 0.25, test-day share | F2 cs3x_c (exploration feeds) | vs private dZ 0.0121, largest of all 7 buckets 0.0300 (7-10 d bucket 0.155 vs 0.143); vs val dZ 0.0126, largest 0.0332; vs public (bar of record) dZ 0.0335, bucket 0.0275 (all-bucket 0.317, declared no bar); PSI 0.0155 / 0.0042 / 0.0191; lowest test day 0.212 vs bar 0.164 | pass=True | <repo>/reports/NEXT_1116/clickstrength/out/c02_parity.txt
- g4 direct NDCG read (nested, control block and nx3 re-weighted in both arms): >= 0 on H lag view, H test-like half, T, N | F1 cs3a_c (all feeds) | H +0.00010 +- 0.00011; H test-like half +0.00020 +- 0.00014; T -0.00006 +- 0.00014; N -0.00015 +- 0.00019 | pass=False | <repo>/reports/NEXT_1116/clickstrength/out/c04_ndcg.txt
- g4 direct NDCG read (nested, control block and nx3 re-weighted in both arms): >= 0 on H lag view, H test-like half, T, N | F2 cs3x_c (exploration feeds) | H -0.000001 +- 0.00012; H test-like half -0.00010 +- 0.00011; T -0.00007 +- 0.00015; N -0.00006 +- 0.00013 | pass=False | <repo>/reports/NEXT_1116/clickstrength/out/c04_ndcg.txt
- g5 leak scans on the witness arrays (causality by timestamps, second-path re-derivation, holdout never a source, own-feed AUC test, leaky build caught) | both forms | 0 causality violations (min 2.001 d to the newest source feed); 14,485 rows re-derived, 0 mismatches; own-feed test at most +1.9 SE (bar 3); deliberately leaky build flagged at +125 to +161 SE | pass=True | <repo>/reports/NEXT_1116/clickstrength/out/c05_leak.txt

## Not run

- REPORT.md file: the harness refused the write; the report text is returned in `summary` only
- Full-train build (246.7M rows) and install into data/feats / nn/feat_cache: kill rule, NO-GO
- Stale-val twin files and proposed trained-test text: NO-GO
- Parity on all train rows with a view (limit J0 of nx3): needs the full-train build
- Leak audit on all 246.7M train rows: only after a GO
- Share of rows whose own user contributed an earlier impression of the same item (stated limit of the rule)
- Any training, any GPU process, any submission
