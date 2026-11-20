# NEXT_1115 / merge_verify (r2): adversarial review of the second pass of the study "merge"

Written 2026-11-15 02:03 UTC. Reviews `reports/NEXT_1115/merge.md` (r2, 01:45 UTC) and `nn_x/w12/next/merge/`. The review of the first pass, which `merge.md` cites by section number, is kept unchanged as `nn_x/w12/next/merge_verify/r1/merge_verify_r1.md`.
CPU only (12 threads, one process at a time), no GPU, nothing written outside `nn_x/w12/next/merge_verify/` and this file.
[M] = measured by me with my own code (`r2/w_common.py`; no import of the study's scripts and none of the first review), [I] = inferred. ± = 1 SE paired per feed unless stated; val 14,423 feeds, test 28,595 (public 8,562, private 20,033).

## 0. Verdict

1. **Every number the study's verdict rests on reproduces** [M]: 27 of 28 numeric rows of section 1 agree exactly or within 0.0001 (bootstrap bounds, other seed); the exception is a noise figure of the A/A null (1.65, see 4). Lines 1, 3, 4, 6 stand as written; line 5 stands in its conclusion.
2. **Line 2 holds only for the three defects it names. The reader is still not safe for an untrusted file: two new holes, both ending in "DECISION: PROCEED"** [M, the study's tool run by me]. (a) The new distribution check reads feeds with M >= 20 only; 16.3% of the test feeds are smaller. A C3 file with 30% of its M < 20 feeds inverted (4.9% of all feeds) gets no FAIL, no WARN, law-rho 0.904, law +0.0014; with all of them inverted, law +0.0054. (b) Gate 2 is not scale-free: W8's OWN order sent as exp(2 z) (Spearman 1.0000, same top-1 in 100% of feeds) reads law-rho 0.682, law +0.0059, "predicted public 0.4004"; C3 sent as softmax probabilities flips from "do NOT blend" (-0.00027) to PROCEED (+0.0021..+0.0039).
3. **The law is not identified where a merge would use it: confirmed** [M]: equal-mix gain at rho 0.90 is [-0.0014, +0.0036] at gap 0 and [-0.0017, +0.0027] at gap 0.002 (5-95%, with residual). Gate 2 is decided by the more optimistic of the two laws; at (0.90, 0.002) the law on record says +0.0006 after the margin, the refit -0.0001.
4. **Drift: no candidate drifts on the late private days except W8TO: confirmed** [M]. Corrections: the A/A inflation is 1.3 for both contrasts on 60 splits (1.65 was 20 splits); that null (independent pools) does not apply to nested files: in the matched design the feed-level SE is not too small (sd of z 0.71), so the W8P20 step is real at nominal size (z -4.2), of which about 40% is feed composition. LEVEL columns are different: two same-recipe pools differ by sd 0.0026, about 10 times the printed SE.
5. **CatBoost (owner's question): confirmed and extended** [M]. Equal mix -0.00458 ± 0.00096 (W8), -0.00470 ± 0.00097 (W16); on the half of val not used for its early stopping -0.00476 ± 0.00133 / -0.00442 ± 0.00133. At small weights nothing: w 0.05 / 0.10 / 0.20 = +0.00014 ± 0.00027 / +0.00009 ± 0.00038 / -0.00040 ± 0.00055 with W8, -0.00021 / +0.00006 / -0.00052 with W16. The model on disk is not under-used by blending at any fixed weight; a tree branch has to bring label information the pool lacks, more weight on today's model brings nothing.
6. Rules: no read-only file changed, no GPU process of the study, no blend written, no leak, thresholds unchanged [M / code read]. **What the lead should conclude:** take the numbers; do not run `merge_check_v2.py` on a real partner file before the two holes of line 2 are closed (three small changes, section 2).

## 1. Claim by claim

| # | claim of the study | study's number | my number [M] | agree |
|---|---|---|---|---|
| 1 | W8 file vs C3 file, test, pooled Spearman of the rank files | 0.9908 (M>=50 0.9914, M<50 0.9850) | 0.9908 (0.9914, 0.9850) | yes |
| 2 | feed-mean Spearman | 0.9873 | 0.9873 (0.9910, 0.9814) | yes |
| 3 | pooled Pearson of per-feed z of the model scores | 0.9938 | 0.9938 (0.9938, 0.9934) | yes |
| 4 | feed-mean Pearson (law on record), test / val | 0.9935 / 0.9935 | 0.9935 (0.9938, 0.9929) / 0.9935 | yes |
| 5 | from two rank files, restored on our shape, feed-mean / pooled | 0.9948 / 0.9943 | 0.9948 / 0.9943 | yes |
| 6 | both files are the rank transform of `preds/ens_*` | >= 99.99% | W8 99.99993%, C3 100.00000% | yes |
| 7 | the law on record was fitted with the feed-mean Pearson on val | `exp7_pairs.py` | code read: `(bincount(fid, a*b) / M).mean()`; `exp7.json` "val_fit_gap<0.01" = 0.018577, 0.37766 | yes |
| 8 | W8 + C3: gap, law on record, refit, measured equal z-mix | +0.00064 ± 0.00055, -0.00012, -0.00013, -0.00027 ± 0.00039 | identical | yes |
| 9 | W8 + CatBoost | +0.01964 ± 0.00144; rho 0.8397 val / 0.8375 test; -0.00444 / -0.00440; refit -0.00434 / -0.00432; -0.00458 ± 0.00096 | identical; the tool run by me prints -0.00440, -0.00432, -0.00458 ± 0.00096 (z -4.76) | yes |
| 10 | the other 8 calibration rows | table 1.2 | identical to 5 decimals in val, gap ± SE, both rho, both laws, measured ± SE | yes |
| 11 | law error (law - measured), 10 partners | on record -0.00009 / 0.00035 / 0.00098 (neural rms 0.00018, non-neural 0.00059); refit +0.00007 / 0.00023 / 0.00043 | identical; largest error 0.89 paired SE (XGBoost) | yes |
| 12 | refit; design | 0.0103 / 0.304 (pooled), 0.0089 / 0.296 (feed-mean); corr(1 - rho, gap) 0.992; LOO rms 0.0005 | identical; LOO rms 0.00052 | yes |
| 13 | bootstrap of a, b, 10 partners, 5-95% | a [-0.013, +0.034], b [0.15, 0.47] | a [-0.0123, +0.0350], b [0.149, 0.478] (seed 777, 2,000 resamples) | yes |
| 14 | equal-mix gain at (0.90, 0), 10 partners: line; with residual | [-0.0013, +0.0034]; [-0.0014, +0.0036] | [-0.0012, +0.0035]; [-0.0014, +0.0036]; 2.5-97.5% of the line [-0.0017, +0.0039] | yes |
| 15 | at (0.90, 0.002) | [-0.0016, +0.0025]; [-0.0017, +0.0027] | [-0.0016, +0.0026]; [-0.0017, +0.0027] | yes |
| 16 | 55 pairs; 52 pairs | a 0.0192 [+0.007, +0.035]; a 0.0140 [-0.001, +0.034]; 52-pair line at (0.90, 0) [-0.0001, +0.0034], at (0.90, 0.002) [-0.0006, +0.0024] | 0.0192 [+0.0065, +0.0352]; 0.0140 [-0.0008, +0.0331]; [-0.0001, +0.0033]; [-0.0005, +0.0024] | yes |
| 17 | share of the interval above the 0.0005 margin, 10 partners / 52 pairs | 0.68 and 0.53 / 0.83 and 0.66 | 0.68 and 0.53 / 0.84 and 0.67 | yes |
| 18 | no pair with rho < 0.95 and gap < 0.005 | 0 | 0 of 55; with gap < 0.01: CatBoost + LLM (0.930, 0.0053, -0.00067) and XGBoost + LLM (0.897, 0.0091, +0.00001) | yes |
| 19 | rank-file mix minus z-mix, mean of 10: ties to better / random / to weaker / w 0.4 / w 0.6 | -0.00006 / -0.00045 / -0.00110 / -0.00000 / -0.00131; range -0.00148 ± 0.00058 .. +0.00060 ± 0.00037; tied rows 2-15% | -0.00006 / -0.00045 (other key) / -0.00110 / -0.00000 / -0.00131; same range; tied 2.1-15.2% | yes |
| 20 | "rank-file mix (tool r2)" column, 10 rows | C3 +0.00013 ± 0.00024 ... LLM -0.00772 ± 0.00091 | identical in all 10 rows | yes |
| 21 | low-agreement share (M >= 20, per-feed Spearman < 0.3) | neural 0.000%, LLM 0.033%, CatBoost 0.084%, XGBoost 0.447% | identical | yes |
| 22 | CatBoost vs W16 | gap 0.01957 ± 0.00144, rho 0.8380 test (0.8404 val), law -0.00438, measured -0.00470 ± 0.00097 | identical | yes |
| 23 | break-even val at rho 0.84 / 0.90 / 0.93 | on record 0.3938 / 0.3967 / 0.3982; refit 0.3962 / 0.3983 / 0.3993 | identical (the "0.3937" of verdict line 6 is 0.39378) | yes |
| 24 | 5-95% of the break-even gap at rho 0.84 | 52 pairs [-0.0009, +0.0117]; 10 partners [-0.0130, +0.0119] | [-0.0005, +0.0116]; [-0.0122, +0.0119] (tail of a ratio, seed noise) | yes |
| 25 | drift, cold-minus-warm offset vs W8 (score-z): 5 candidates x 9 cells, contrasts, chi2 | table 1.8; chi2 4.8 / 18.5 / 105.6 / 9.7 / 15.6 | identical in all 45 cells, all contrasts and all five chi2 | yes |
| 26 | Spearman and same top-1, PUBLIC / PRIVATE | W16 0.9967 / 0.9966, 93.2%; W8P20 0.9976 / 0.9975, 93.4%; W8TO 0.9953 / 0.9950, 90.6%; S16 0.9964 / 0.9963, 92.8%; C3 0.9874 / 0.9873, 82.5% | identical | yes |
| 27 | CatBoost vs W16 drift; CatBoost vs W8 law-rho by day | 0.8153, 37.3%, -0.094 ± 0.002, public -0.097, private -0.094; 0.8363 / 0.8372 / 0.8398 / 0.8362 / 0.8382 | 0.8153 ± 0.0006, 37.3%, -0.0944 ± 0.0016, -0.0967, -0.0935, chi2 1.9; identical by day | yes |
| 28 | A/A null of the cold-offset contrasts | sd of z 1.25 (PRIVATE - PUBLIC), 1.65 (last - first); chi2 mean 5.8, max 14.9 (20 splits) | 1.31, 1.33; chi2 mean 5.2, max 12.7, 8% above the 5% point (60 splits, seed 4242) | partly: 1.65 not reproduced |
| 29 | reader: 41 of 41 checks; "refuses partly corrupted files" | `out/t1_reader_tests.txt` | not re-run as a whole. 9 runs of the tool by me: the two files of the first review now FAIL (9.78%, 11.75%), both known answers print as claimed; 5 new files all get PROCEED (section 2.1, 2.2) | no, as a general statement |

## 2. Issues by severity

### 2.1 HIGH: the distribution check does not look at feeds with M < 20 [M]

`low_share()` in `merge_check_v2.py` uses `S.M >= LOW_M` (20). Feeds with M < 20: test 4,656 (16.3%), public 1,351 (15.8%), val 2,548 (17.7%). The study's limits name damage below 0.2% of the feeds and damage that keeps the per-feed correlation above 0.3; they do not name this.

| synthetic partner file (C3 rank file, partner public 0.393454; `out/tool_X1.txt`, `tool_X2.txt`) | feeds damaged | reader | mean Spearman | "r_f < 0.3" printed | law-rho | law on record | decision |
|---|---|---|---|---|---|---|---|
| every feed with M < 20 negated | 16.3% | no FAIL, no WARN | 0.6701 | 0.000% | 0.6903 | +0.00537 | PROCEED |
| 30% of the feeds with M < 20 negated | 4.86% | no FAIL, no WARN | 0.8928 | 0.000% | 0.9040 | +0.00140 | PROCEED |
| 10% of the feeds with M < 20 negated (my code only) | 1.69% | - | 0.9544 | 0.000% | 0.9600 | law rises by +0.00062 | - |

The second file has exactly the profile a merge is looking for (rho 0.90, gap 0.001). The table of the tool does show "M < 50: law-rho 0.763", but nothing turns it into a FAIL. Gate 3 catches it only if the val file carries the same damage; damage in the test file alone passes all three gates [I].
Fix (not applied, the study's files are read-only for me): add a second line to the check for 2 <= M < 20, on the share of feeds with a NEGATIVE per-feed Spearman. Genuine partners against W8: neural 0.000%, CatBoost 0.41%, LLM 0.49%, XGBoost 1.10% [M, `out/w5_lowshare.txt`]; the three damaged files read 100%, 29.9%, 10.4%. A FAIL line at 3% separates them [threshold is my proposal, not a rule on record]. And WARN when test and val law-rho differ by more than 0.015 (10 genuine partners: at most 0.0051, XGBoost).

### 2.2 HIGH: gate 2 can be passed by re-scaling a file; an identical order is priced at +0.0059 [M]

For a partner SCORE file the tool z-scores the scores as sent. The law was calibrated on raw outputs of models on disk (our W8 scores have a within-feed skewness of -0.40; the files below +1.6, +2.6 and +3.5). The study lists "probabilities instead of logits" as not tested; tested here (`out/tool_X3.txt`, `tool_X4.txt`, `tool_X5.txt`, `out/w4_extra.txt` C):

| partner file | Spearman to W8 | law-rho the tool uses | restored on our shape (printed as INFO) | law on record | gates 1 / 2 | decision | measured equal z-mix on val vs W8 |
|---|---|---|---|---|---|---|---|
| C3 logits (as on disk) | 0.9873 | 0.9935 | 0.9948 | -0.00027 | met / not met | do NOT blend | -0.00027 ± 0.00039 |
| C3 as per-feed softmax(1.0 z) | 0.9873 | 0.8663 | 0.9948 | +0.00210 | met / met | PROCEED | -0.00066 ± 0.00048 (M < 50 -0.00229 ± 0.00086) |
| C3 as per-feed softmax(1.5 z) | 0.9873 | 0.7682 | 0.9948 | +0.00392 | met / met | PROCEED | -0.00054 ± 0.00050 (M < 50 -0.00197 ± 0.00089) |
| W8 ITSELF as exp(2 z), public equal to ours | 1.0000 | 0.6820 | 1.0000 | +0.00592, "predicted public of the equal mix 0.40040" | met / met | PROCEED | 0 exactly (same order) |

The only reaction of the tool is an INFO line ("ask for logits"). Gate 3 stops the three re-scaled files on val (tool runs for the two softmax files; the exp file has our order, so its mix gains exactly 0), so no blend would be written with a val file supplied; without one the decision line stands, and for a genuinely different partner the deflated rho raises w* and the printed forecast by an unknown amount. Mixing raw probabilities also costs on short feeds (table).
Fix: decide gate 2 on the rank-restored rho for EVERY partner (it is scale-free and is already computed); print the score-rho as information; FAIL when the two differ by more than 0.01. Cost in accuracy: none on the 10 on-disk partners by the study's own row (law on record with the tool's rank-file rho: rms 0.00033, max 0.00089, against 0.00035 / 0.00098 with the score rho; the C3 value 0.9948 is re-derived by me, the other nine are the study's); bias at most +0.0020 in rho, towards not blending.

### 2.3 MEDIUM

- **The A/A null of the drift table is the wrong null for the files it is applied to, and its 1.65 is noise** [M, `out/w2_drift.txt` C, `out/w4_extra.txt` A-B]. (a) 60 splits, other seed: sd of z 1.31 (PRIVATE - PUBLIC) and 1.33 (last - first day), chi2 mean 5.2, max 12.7; so "1.25-1.65" should read "about 1.3". (b) That null compares two INDEPENDENT 8-pools. W16, W8P20 and W8TO share their members with W8. Matched null for the W8P20 design at the train-only level (0.8 z(W8 pool) + 0.2 z(4 other wide students) against the W8 pool, all 70 choices): sd of z 0.71, max 2.04, chi2 max 10.4. The real pop1kd admixture at the same level reads PUBLIC +0.00997, PRIVATE +0.00928, contrast -0.00068 ± 0.00022 (nominal z -3.1 = 4.3 null sd of z; 7 null sd of the contrast), chi2 11.2. So the study's "z 4.2, about 3.4 after the A/A inflation" should not be deflated: the step is real at its nominal size. (c) Resampling 1-hour blocks instead of feeds changes the SE of PRIVATE - PUBLIC by a factor 0.94-1.07 (W8TO 1.35: z -7.1 becomes -5.2); the inflation is not a within-hour effect.
- **About 40% of the W8P20 step is feed composition** [M, `out/w2_drift.txt` E]. Inside 20 strata of feed size x cold share the contrast is -0.00054 ± 0.00021 (unstratified -0.00087 ± 0.00021; negative in 12 of the 20 strata); private feeds are colder (cold share 0.459 against 0.429). Sunday pm - am -0.00111 ± 0.00041; the same clock split on the other days reads -0.00020 ± 0.00042 (Sat), -0.00091 ± 0.00040 (Mon), -0.00020 ± 0.00038 (Tue). The reading "the public read is taken where the cold promotion is 16% larger" is right in sign; the part that is not composition is about 8-10% of the level. The NDCG size stays unmeasured.
- **The ± of a LEVEL column of the drift table is not an SE between pools** [M]. Level of the cold offset between two same-recipe 8-pools: sd 0.00257 over 60 splits against a printed feed-level SE of 0.00024 (ratio 10.6; single members differ by sd 0.0056). S16 (-0.0015) and C3 (-0.0021) against W8 are inside that noise. The study draws no conclusion from these two levels, but the tool prints "± 0.0002" next to them. W8P20's level is safe (train-only level +0.0095 against a matched null sd of 0.00028), W8TO's too (same members).
- **Verdict line 4 averages two different populations** [M]. "Ties to the better file -0.0001" is +0.00026 for the 7 neural pools (near-copies of W8) and -0.00080 for the 3 non-neural models (CatBoost -0.00036 ± 0.00056, XGBoost -0.00057 ± 0.00062, LLM -0.00148 ± 0.00058). A merge partner belongs to the second group. The advice (exchange model scores) stands and is, if anything, understated by the headline.
- **Gate 2 is decided by the more optimistic law** [M, arithmetic]. At (rho 0.90, gap 0.002): law on record - 0.0005 = +0.00060 (met), refit - 0.0005 = -0.00008 (not met); the tool prints "OPPOSITE reading" and still "PROCEED". This is the rule as written (the lead's `src/merge_check.py` decides on 0.0186 / 0.378), so it is applied correctly; the lead should know that the W8-level evidence puts the share of the interval above the margin at 0.53 there.
- **Gate 3 is read on the study's own w\*** (unchanged from the first review; the study says so). Two z values are printed (w\* and equal mix); the lead has to fix which one decides before a partner file exists.
- **CatBoost break-even levels are point values of a law whose slope is not identified** [M]: 5-95% of the break-even gap at rho 0.84 is [-0.012, +0.012] from the 10 partners. "Breaks even only at val 0.3937-0.3962" is the law, not a measurement. What is measured is the equal mix (-0.0046, 4.8 SE) and the small weights (0.0000 ± 0.0004); the conclusion of line 6 rests on those and holds.

### 2.4 LOW

- In the tie-break table the "better file" is chosen with the val labels that then score the mix; it can matter only for C3 and S16 (gaps within 1.2 SE): at most 0.0003 on two rows, 0.00005 on the mean.
- The 10 calibration rows are not 10 independent tests of the law on record (all share W8 and the val feeds, and the law was fitted on val pairs of related pools); "all within the paired SE" is a weak test, as the study's leave-one-out row (rms 0.0005) already shows.
- The DRIFT flag for a foreign file (no val rows) fires at 3 nominal SE of an independent-pool contrast, which is 2.3 null sd [M: 3 / 1.31].
- Break-even CI lower bound for the 52-pair fit: -0.0009 (study) against -0.0005 (mine); the 5% point of a ratio, seed noise.

## 3. Additional measurements for the owner's CatBoost question [M]

`nn_x/cb26/data/e3_xm_yeti` (val 0.38201); (1 - w) z(pool) + w z(CatBoost) minus the pool, paired, 14,423 val feeds; fixed grid of the lens, no weight is selected (`out/w1_calib.txt` E, `out/w6_cb_halfB.txt`).

| pool | w 0.05 | w 0.10 | w 0.20 | w 0.30 | w 0.50 |
|---|---|---|---|---|---|
| W8 final (0.40165) | +0.00014 ± 0.00027 | +0.00009 ± 0.00038 | -0.00040 ± 0.00055 | -0.00148 ± 0.00070 | -0.00458 ± 0.00096 |
| W16 final (0.40158) | -0.00021 ± 0.00027 | +0.00006 ± 0.00039 | -0.00052 ± 0.00056 | -0.00141 ± 0.00071 | -0.00470 ± 0.00097 |
| W16 train-only pool (0.40144) | +0.00002 ± 0.00028 | -0.00019 ± 0.00040 | -0.00065 ± 0.00057 | -0.00125 ± 0.00072 | -0.00448 ± 0.00097 |

- The W16 train-only row at w 0.1 equals the member-rule statistic of `reports/CATBOOST_1115/audit.md` 1.2 (-0.00019 ± 0.00041 val; its holdout +0.00007 ± 0.00022): two independent computations agree.
- By size at w 0.1 the sign flips between the two pools (W8: M >= 50 -0.00030 ± 0.00041, M < 50 +0.00066 ± 0.00071; W16: +0.00035 ± 0.00042, -0.00036 ± 0.00074): no stable subgroup.
- The model was early-stopped on val user-half A. On half B (7,211 feeds): CatBoost 0.37870, gap to W8 +0.02034 ± 0.00199, equal mix -0.00476 ± 0.00133, w 0.1 -0.00003 ± 0.00052; on half A (7,212 feeds) gap +0.01894 ± 0.00207, equal mix -0.00440 ± 0.00139. The early stopping does not flatter the mix.
- Reading: on val the CatBoost of record adds nothing at any fixed weight and costs from 0.2 upwards. The merge law prices it correctly at 0.5. Whether a NEW tree arm is worth a branch is the question of the CatBoost track (new label information at rho near 0.84), not of this tool.

## 4. Audit

- **Rules applied as written** [code read]: gate 1 `gap <= 0.002`; gate 2 `0.0186 (1 - rho) - 0.378 |gap| - 0.0005 >= 0` with the coefficients of the lead's v1; gate 3 `z >= 2` of the paired val delta against the better file on val. No threshold moved. Settings of the study's own (tie-break to the better file, 0.75% / 0.2%, 1e30, w\*) are labelled as such. DECISION now needs gates 1 and 2, as SYNTHESIS bet 1 lists them.
- **Leak**: val labels enter only the scoring of blends and the two-coefficient fits, as in the lens. Cold = `it_hist_all_luo == 0` (leave-user-out count). No later snapshots of a user, no quarantined file (`archive/preds/xgb_tr1m_ndcg_*` is the lens's own input) [code read].
- **SEs**: blend deltas and gaps paired per feed; correlations SE over feeds; drift contrasts two independent feed groups. Correct for what they compare; see 2.3 for what the drift SEs do not cover.
- **Read-only files** [M]: `nn_x/w5/lane.sh` 22:14:15, `nn_x/w5/train.py` 20:20:00, `src/merge_check.py` 21:48:36, nothing under `src/` after 21:50 on 11-14; `data.py`, `finetune.py`, `cd_arms*.py`, `make_sub.py`, `presub_gate.py` unchanged since 11-11..11-13; newest file in `subs/` is `cand_W16kd_r3.parquet` (21:30); no `cand_MERGE*`, no `T_blend_must_not_exist*` anywhere. Files changed after 23:08 in `reports/`, `nn_x/abl/preds/`, `nn_x/w12/cb/`, `nn_x/w12/next/poplag*`, `u20wit` belong to the lead's lanes and to other studies.
- **GPU** [M]: 32 compute processes, all `train.py --variant ...` lanes started 21:26-01:44 UTC. The study's scripts contain no torch, no subprocess, no network call; `make_sub` is reached only under `--blend` and is stubbed in its tests (`t1_reader_tests.py` l.112). Its past thread counts and environment cannot be checked.
- **Cached inputs of the study** [M]: `NOTSUB_c3_test_scores.parquet` equals my 0.5 z(FT) + 0.5 z(TO) of `preds/ens_Q2a8s3_*` to 4.8e-07 (float32) with 100.00000% identical order; `NOTSUB_catboost_val_ranks.parquet` equals my within-feed rank of `e3_xm_yeti_val.npy` in 99.99928% of the rows (mid-ranks of ties). None of my numbers uses the study's cache.
- **The study's self-reported deviations** (one JSON read without the CUDA variable, in-place edit of its own first-pass folder with a copy in `r1/`, two read-only `pgrep`) are harmless; the first-pass outputs are in `r1/out/` as stated.

## 5. Method

- `r2/w_common.py`: own per-feed z (float64, no epsilon), own within-feed rank, own parquet join, own NDCG (binary gain, pessimistic ties; maximum per-feed difference to `src/metrics.ndcg_per_feed` 3e-16 on W8 val), the partner list of `p4_partner.py`.
- `w1_calib.py` (116 s): correlation definitions from the two parquet files and from the model scores; 10 calibration rows; refits, leave-one-out, feed bootstrap (2,000 multinomial resamples, seed 777, all 55 pairs resampled together, better file re-decided in each resample); 55 / 52-pair fits; break-even gaps; rank-file mixes with an explicit `np.lexsort` tie-break; CatBoost weight grid.
- `w2_drift.py` (29 s): cold offsets from `preds/ens_*` on val and test, file agreement from the six parquet files, A/A null (60 splits, seed 4242), hour-block resampling (1,000 resamples), stratified contrast, clock placebo, CatBoost against W16.
- `w3_synth.py` + `w3_run_tool.sh`: five synthetic partner inputs (seed 2026) and nine read-only runs of the study's tool (`-B -I`, no `--blend`).
- `w4_extra.py` (41 s): level null, matched null of the W8P20 design, the re-scaling table with my own code, the M < 20 damage with my own code, rebuild of two cached inputs. `w5_lowshare.py`, `w6_cb_halfB.py`: one table each.

## 6. Not checked

- `t1_reader_tests.py`, `c1_calib.py`, `c2_extend.py`, `a1_drift_null.py` and `drift_by_day.py` were not executed by me (they write into the study's folder); their outputs were compared with my own numbers. 32 of the 41 reader cases were not repeated.
- `--blend` to disk (no submission file may be written): the path was read as code only (`mix_rank` gives a tie-free rank with 0 = worst, which `make_sub` re-ranks).
- The weight model behind w\* (T6 grid), the interpolated public gap SE, the per-day rows of the tool for partners other than C3 and CatBoost.
- Anything on the private side or about the val-to-public transfer of the law (no labels). The NDCG size of any drift.
- My two proposed lines (3% negative feeds below M 20; restored rho for gate 2) are not implemented or tested inside the tool; the numbers behind them are from my own code on 10 genuine partners and three damaged copies of one file.
- The matched null is at the train-only level (no fine-tune half) and its 70 choices share one reference pool; the 60 A/A splits share 16 members. Each sd carries roughly ± 15-20%.
- A real partner file does not exist; a strong partner at rho 0.90 is on no disk.

## 7. Commands

```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
D=nn_x/w12/next/merge_verify/r2; O=$D/out
.venv/bin/python -B $D/w1_calib.py > $O/w1_calib.txt      # 116 s
.venv/bin/python -B $D/w2_drift.py > $O/w2_drift.txt      # 29 s
bash $D/w3_run_tool.sh                                     # writes 5 synthetic inputs to $D/cache (154 MB) and runs the study's tool 9 times, 3 min
.venv/bin/python -B $D/w4_extra.py > $O/w4_extra.txt      # 41 s
.venv/bin/python -B $D/w5_lowshare.py > $O/w5_lowshare.txt
.venv/bin/python -B $D/w6_cb_halfB.py > $O/w6_cb_halfB.txt
rm -f $D/cache/XNOTSUB_*.parquet                          # the synthetic inputs have the (feed_id, item_id, score) layout; removed after use
```

## 8. Files created, and rule notes

- Scripts: `<repo>/nn_x/w12/next/merge_verify/r2/w_common.py`, `w1_calib.py`, `w2_drift.py`, `w3_synth.py`, `w3_run_tool.sh`, `w4_extra.py`, `w5_lowshare.py`, `w6_cb_halfB.py`.
- Outputs: `<repo>/nn_x/w12/next/merge_verify/r2/out/`: `w1_calib.txt` / `.json`, `w2_drift.txt`, `w4_extra.txt`, `w5_lowshare.txt`, `w6_cb_halfB.txt`, `tool_X1.txt` .. `tool_X5.txt`, `tool_known_CatBoost_inhouse.txt`, `tool_known_W8_vs_C3_file.txt`, `tool_r1_V_partial_inversion_10pct.txt`, `tool_r1_V_shuffled_12pct_feeds.txt`.
- Cache: empty. The five `XNOTSUB_*` inputs were deleted after the runs; `w3_synth.py` rebuilds them.
- First review, unchanged: `<repo>/nn_x/w12/next/merge_verify/r1/merge_verify_r1.md` (copy of the previous content of this file), with its scripts, outputs and four synthetic files one level up.
- This report: `<repo>/reports/NEXT_1115/merge_verify.md` (replaces the first review).
- Rule notes. No GPU, no install, no network, no signal to any process, no file outside my folder and this report. Every Python process had `CUDA_VISIBLE_DEVICES=""` and 12 threads; shell-only look commands (`ls`, `cat`, `grep`, `stat`, `find`, `ps`, `nvidia-smi --query-compute-apps`) were run without the variable on their line. The study's tool was run with `-B`, so nothing was written into its folder (its `__pycache__` still dates from 01:43).
