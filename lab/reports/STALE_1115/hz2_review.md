# hz2_review: adversarial review of the hz2 training view (reports/STALE_1115/hz2.md)

Reviewer run 2026-11-15 05:49-06:13 UTC (one run; the review directory was empty at the start, nothing to resume). CPU only. Work directory `nn_x/w12/stale/hz2_review/`. Conventions: ± = 1 SE paired per feed; [M] measured by me with my own code, [D] read in a project file, [I] inferred. Judging rules of this review: `REVIEW_RULES.txt` (written before the reads they judge).

## 0. Verdict

1. **CONFIRMED, with a corrected description.** Every number and identity of the study's verdict that I re-derived agrees: causality, the view values, the gate table, the labelled reads, the HZ11 cells, the drop-in identity. No leak found: no label enters any input, nothing dated after the row's day is counted.
2. **"Full-dose test-like" is true for COVERAGE only.** Zero share by listing age and feed-level cold share match private rows on days >= Sep 13. The LAG is not there for warm rows: the latest dense day counted is on average 1.52 d before the row (78.5% of warm rows at 1 d) against 7-10 d on a test row [M]. For listings old enough to have the full test depth (55% of rows) the primary view is the fresh count up to yesterday times a depth factor (1.2 to 2.7 on days >= Sep 9).
3. **NEW ISSUE: the view creates an input combination that does not exist at test time.** Rows that the view makes cold keep a warm encoder score: P(pre2 item-ID contribution != 0 | `it_hist_all` = 0) is exactly 0 on val, test and as-built train rows and 0.22-0.30 under the view; 9.6% of all train rows in the arm as trained [M]. The stale-val witness removes this ("pre2both"); the training view does not. Label-relevant by my pre-declared rule (within-feed AUC 0.528 ± 0.002).
4. **The arm as trained (hz2k, all days) FAILS the study's own gate**; the gate holds on 40.2% of train feeds. Disclosed by the study, but the lead's rule file describes the arm as "per-feed test-like lag (6-9 d)" for all 8 inputs, which 66% of rows do not get.
5. The running hz2 lanes are sound to keep: loader exact over all 246.7M rows, flag-off path untouched, arm hz ran on the old module. Nothing must be stopped.
6. Before a second wave: add the no-ID encoder swap on newly cold rows; decide whether warm-row lag is wanted (only `feats_L` on Sep 17-19 feeds has it, 24% of feeds).

## 1. Claim table

| claim of the study | its number | my number (own code) | agree |
|---|---|---|---|
| rows / lag counts 6, 7, 8, 9 / public-like share | 246,745,836 / 666,704, 811,819, 835,005, 837,680 / 0.2994 | same / same / 0.2994; lag shares by part equal my own test counts (public 0.7066, 0.2934; private 0.2423, 0.3782, 0.3794) | yes |
| lag independent of the feed | (hash of `feed_id`) | corr with M -0.0009, with p_random -0.0016; mean lag by day 7.582-7.588; public-like share by day 0.298-0.301 | yes |
| causality: nothing counted after the row's day, no dense snapshot of the own day | 0 rows (every row, from its own mask files) | 0 of 756,179 counted pairs (primary) and 0 of 577,061 (lag-first), recount from raw data on 37,918 rows of 480 feeds | yes (sample) |
| view files = the declared rule | recount 1.1e-7 on 240 + 160 feeds | `it_hist_hz_all`, `it_hist_hz_et1`, `xp_imp_hz_c50`: 100% of 37,918 rows equal (max rel 1.2e-7), both variants; `seq_cv_hz_pop`: 100% of the 37,376 rows of feeds with a history; the other 257 mismatching rows all sit in feeds with H = 0, where the kernel returns 0 | yes |
| the same rule reproduces stored val / test columns (V1) | exact on every row | my rule reading: `it_hist_all_luo`, `it_hist_et1_luo` exact, `xp_imp_c50` within 1.3e-7 on 23,841 val and 42,106 test rows; `seq_cv_pop` exact on test, 99.92% on val (19 rows not examined; same H = 0 convention expected) | yes |
| exact / young / truncated rows | 34.2% / 32.6% / 65.8% | 34.0% / 32.3% / 66.0% (day-weighted, my sample) | yes |
| own-user pairs | 6 of 405,550 | 10 of 756,179 (earlier snapshots only) | yes |
| keep mask rows | 46,918,534 (19.0%) | 46,918,534 | yes |
| training mass: days >= Sep 13 / Sep 6-8 | 40.2% train, 45.5% pool / 27.4%, 24.9% | 40.19%, 45.46% / 27.43%, 24.88% | yes |
| gate, private-like >= Sep 13, as built (dZ / bucket / dM) | 0.225 / 0.591 / 0.429; 0.111 / 0.440 / 0.172; 0.232 / 0.907 / 0.431 | identical to 3 decimals | yes |
| gate, hz half dose, `it_hist_all` | 0.153 / 0.376 / 0.325 | identical | yes |
| gate, hz2 >= Sep 13 | 0.022 / 0.058 / 0.107; 0.014 / 0.055 / 0.107; 0.036 / 0.070 / 0.144 PASS | identical; PASS by the declared bars | yes |
| gate, hz2L >= Sep 13 | 0.043 / 0.061 / 0.079; 0.039 / 0.064 / 0.098; 0.053 / 0.099 / 0.063 FAIL (G1) | identical | yes |
| gate, all days: hz2 and hz2k | 0.064 / 0.104 / 0.046 ...; 0.032 / 0.113 / 0.160 ... FAIL | identical (all nine triples) | yes |
| public-like >= Sep 13, hz2 dZ | 0.030 / 0.008 / 0.030 PASS | 0.030 / 0.008 / 0.030 | yes |
| Sep 9-12 public-like fails G1 | 0.057 | 0.057 | yes |
| share of the zero-share gap closed | 90% / 88% / 84%; hz 32% | 90.2% / 87.4% / 84.5%; hz 32.0% | yes |
| zero share `it_hist_all` by age, hz2 vs private | 0.989 / 0.949 / 0.906 / 0.788 / 0.337 / 0.178 / 0.191 vs 0.975 / 0.916 / 0.859 / 0.786 / 0.395 / 0.192 / 0.200 | identical | yes |
| feed cold share q50: private / as built / hz / hz2 | 0.452 / 0.230 / 0.303 / 0.456 | 0.452 / 0.230 / 0.303 / 0.456 | yes |
| W8 pool, view - stored, 13,424 feeds | -0.00446 ± 0.00074 | -0.00446 ± 0.00074 | yes |
| same, Sep 17-19 (8,279) / Sep 13-14 (5,145) | -0.00245 ± 0.00085 / -0.00768 ± 0.00135 | -0.00245 ± 0.00085 / -0.00768 ± 0.00135 | yes |
| pop1kd pool, view - stored | -0.00118 ± 0.00035 | -0.00118 ± 0.00035 | yes |
| S(0.2) stored -> view; D | -0.00050 ± 0.00032 -> +0.00032 ± 0.00031; +0.00082 ± 0.00039 | same three numbers | yes |
| lag-first: W8, pop1kd, S(0.2), D | -0.00523 ± 0.00079, -0.00109 ± 0.00042, +0.00058 ± 0.00032, +0.00108 ± 0.00042 | same | yes |
| E4, lag-first minus primary, Sep 17-19 | -0.00035 ± 0.00057 | -0.00035 ± 0.00057 | yes |
| poplag v3, same feeds | -0.00332 ± 0.00075; Sep 17-19 -0.00195 ± 0.00086 | same | yes |
| HZ11 selected, primary: Q_view / I | +0.00067 ± 0.00041 / +0.00023 ± 0.00048 | same; by root +0.00094, +0.00003, +0.00154, +0.00006 | yes |
| HZ11 @last, primary: Q_view / Q_stored / I | +0.00078 ± 0.00038 / -0.00039 ± 0.00036 / +0.00117 ± 0.00047 | same | yes |
| HZ11 lag-first: selected / @last | +0.00124 ± 0.00043 / +0.00116 ± 0.00040 | same; all 16 single-model pairs positive | yes |
| 11-12 read-out, member level @last, 30,000 feeds | -0.00052 ± 0.00019 | -0.00052 ± 0.00019 | yes |
| CPU path control (stored replay vs GPU files) | corr 0.99993-0.99994, NDCG within 0.00023 | corr 0.999936-0.999942; NDCG +0.00001, -0.00006, +0.00011 (SE 0.00014) on 3 models | yes |
| drop-in identical to `cd_arms6` when the variables are unset | bit-identical on 200 + 200 feeds | bit-identical hz tensor over all 246,745,836 x 5 cells and 8 batches of 200 random feeds (13,701 rows), tested on the copy the lanes import | yes |
| drop-in with the view: 8 columns, keep rows stored | T3-T5 on samples | every one of 246,745,836 x 8 cells = keep ? stored : fp16(log1p(view file)); zero rates 0.130 -> 0.408 and 0.423 -> 0.515 as in the live log | yes |
| arm hz never ran on the patched module | all 16 started before 04:57:57Z | latest hz start 04:54:43Z; all 16 hz2 logs show 8 columns and 46,918,534 keep rows | yes |

Added by me, not in the study: the study's CPU reader against the trainer's GPU view path. For the finished run `w5_kcd_hz2_s958`, reader output vs `preds/holdout/nn_w5_kcd_hz2_s958_holdout_hz.npy` on the 13,424 late feeds: correlation of per-feed z 0.999943 (0.994191 against the as-built file of the same checkpoint), NDCG CPU - GPU +0.00015 ± 0.00014 [M]. The reader is fit for the arm read. This is a path control, not a read of the arm.

## 2. Issues by severity

### MAJOR 1. The lag is missing for warm rows; the arm is a coverage-parity arm

- Distance from the row's day to the latest dense day it counts, rows with a dense count > 0 [M, `logs/recount_40_2024.log`]:

| feed days | primary (`feats`): mean / share within 2 d / mean depth factor | lag-first (`feats_L`): mean / share within 2 d / mean depth factor | test row |
|---|---|---|---|
| Sep 6-8 | 1.01 d / 1.000 / 3.80 | 1.01 d / 1.000 / 3.80 | 7-10 d / 0 / 1 |
| Sep 9-12 | 1.15 d / 0.953 / 2.52 | 1.38 d / 0.952 / 2.61 | |
| Sep 13-14 | 1.41 d / 0.904 / 1.52 | 3.69 d / 0.000 / 2.21 | |
| Sep 17-19 | 2.54 d / 0.640 / 1.21 | 7.76 d / 0.000 / 2.12 | |
| all | 1.52 d (78.5% at 1 d) | 3.29 d | |

- Listings with full test depth (X_test = 12; 54-57% of rows in every day group): window = every dense day before the row, 1.0 d back on Sep 9-14 and 1.9 d on Sep 17-19, multiplied by 2.73 / 1.60 / 1.24 [M].
- Consequences I measured on 60,000 random feeds of days >= Sep 9 [M, `logs/infer_check_read.log`]: `seq_cv_pop` of the view is LARGER than the stored value on 47.0% of rows (83% / 79% for listings of 24-76 d / > 76 d); `it_hist_all` on 21.5% (39% / 38%). 44.5% of warm view values are non-integer (65.7% on Sep 17-19); a test row is always an integer count.
- The documented mechanism has two parts: cold share 0.36 -> 0.44-0.48 and warm-item staleness 1.75 d -> 9.3-11.2 d [D, `docs/01_DATA_AND_FINDINGS.md` 8.1]. The primary view reproduces the first and not the second. The study says so in sections 4.4 and 8; its verdict line 1 and the lead's rule (`nn_x/w5/logs/E_kd_predeclared.txt` l.365: "per-feed test-like lag (6-9 d)" for all 8 inputs) do not.
- By my rule R3b (mean < 4 d): flagged for both variants overall; `feats_L` is lagged on Sep 17-19 only (24.1% of train feeds) and fails the study's gate there by G1.

### MAJOR 2. Cold counters with a warm encoder score: a train-only input combination

- Identity on real rows [M, `logs/pre2_tell.log`]: rows with `it_hist_all_luo` = 0 and `pre2_id_contrib` != 0: 0 of 527,423 cold val rows, 0 of 1,299,059 cold test rows, 0.0000 of as-built train rows.
- Under the views:

| population | half-dose hz | hz2 | hz2k (arm as trained) |
|---|---|---|---|
| all train feeds (150,000 random): share of ALL rows cold with ID contribution / P(ID != 0 given cold) | 0.025 / 0.110 | 0.096 / 0.195 | 0.096 / 0.236 |
| all train feeds, days >= Sep 13 | 0.034 / 0.134 | 0.128 / 0.301 | 0.128 / 0.301 |
| all train feeds, Sep 9-12 | 0.036 / 0.163 | 0.140 / 0.317 | 0.140 / 0.317 |
| trPool, private-like, days >= Sep 13 | 0.027 / 0.088 | 0.102 / 0.223 | 0.102 / 0.223 |

- By listing age (trPool late, P(ID != 0 given cold under hz2)): 0.103 / 0.284 / 0.424 / 0.487 / 0.217 / 0.029 / 0.039; private rows 0.000 in every bucket.
- The inputs concerned are `pre2_fe_z` and `pre2_fe_s` (2 of the 34; `pre2_id_contrib` itself is not an input). On those rows `pre2_fe_s` is +0.21 private-row SD above its level on private cold rows; the stored count the view removed has median 10, q90 37.
- Pre-declared materiality rule R3e (train labels of non-holdout pool feeds, hz2-cold rows, 21,732 feeds): within-feed AUC of `pre2_id_contrib` 0.5276 ± 0.0015 (z +18) = "MATERIAL". Positive rate 1.47% on cold rows with ID contribution against 0.95% on cold rows without. For scale: the removed counter has AUC 0.5699 ± 0.0018 on the same rows.
- Against over-reading it: on as-built models, removing that ID part at inference costs 0.00000-0.00005 [D, `reports/sol_cpu_CPU2_stale_replay_pre2s_and_arms.md`]. The size under TRAINING is not known.
- The project already treats this coupling in the stale-val witness: scenario `both_pre2both` replaces `pre2_fe` / `pre2_fe_z` / `pre2_fe_s` by the no-ID score on newly cold rows [D, `nn_x/w5/stale_pred.py`]. So an hz2 model is trained on a pattern that neither test nor stale-val contains, and the lagged-holdout read through this view contains it too.
- The fix is available: `data/feats/train/pre2_fe_noid.npy` exists for all train rows (no NaN on 3M sampled rows; equals `pre2_fe` within 0.028 where the ID contribution is 0) [M].

### MEDIUM 3. The gate holds on 40% of the training mass; the arm as trained fails it

- hz2k, all days, private-like: `it_hist_all` 0.032 / 0.113 / 0.160, `seq_cv_pop` 0.063 / 0.111 / 0.063, `xp_imp_c50` 0.049 / 0.159 / 0.084: FAIL by G1 and G2 (my numbers = the study's).
- Sep 6-8 feeds under the keep mask (27.4% of feeds; 67.6% of their pool rows keep stored values): `it_hist_all` dZ 0.094 / bucket 0.250, `seq_cv_pop` 0.240 / 0.402, `xp_imp_c50` 0.232 / 0.438 [M]. Young listings there are test-like, old ones are as built (fresh, and counting days after the row).
- Sep 9-12 (32.4%): passes for private-like, fails G1 for public-like (0.057).

### MEDIUM 4. Shape of the value distribution for old listings is farther from test than as built

- My rule R3a, total-variation distance of the binned model-scale value, row-weighted over age buckets, private-like late: `it_hist_all` 0.2516 as built -> 0.0999 hz2; `seq_cv_pop` 0.1215 -> 0.1036; `xp_imp_c50` 0.2412 -> 0.1242. Flagged (> 0.10) for `seq_cv_pop` and `xp_imp_c50`; `it_hist_all` sits on the bar.
- By bucket the gain is all in listings younger than 10 d (0.35-0.60 -> 0.01-0.05). For listings of 24-76 d / > 76 d it gets worse: `it_hist_all` 0.09 / 0.07 -> 0.16 / 0.14; `seq_cv_pop` 0.03 / 0.02 -> 0.18 / 0.15; `xp_imp_c50` 0.03 / 0.08 -> 0.18 / 0.23.
- Caveat: bins of 0.25 are sensitive to the non-integer lattice of the depth factor; the level gap (0.2-0.3 on the log1p scale) is the study's own number.

### LOW 5. Wording of the labelled conclusions

- HZ11: the declared read is "not separated" (Q_view +0.00067 ± 0.00041, z 1.6). "Supports" is post hoc. The four cells share models, feeds and two correlated views. Verdict line 4 ("training on a stale-like view helps") is stronger than the declared read allows; two of the four roots are at +0.00003 and +0.00006.
- "Staleness cost -0.0045": the stored baseline counts days after the row. The clean number is Sep 17-19, -0.00245 ± 0.00085 (z 2.9), and given issue 1 it is a cost of coverage, not of lag. Given issue 2 the encoder still knew the items.

### LOW 6. Own-user rule and user-0 feeds

- Task text (b) "nothing from the row's own user" is not met literally: 10 of 756,179 counted pairs come from earlier snapshots of the row's own user (1.3e-5). Same rule as on test rows.
- 123,904 train feeds (3.9%) have user 0, where "later snapshot of the same user" cannot be removed. Under the view only same-day sparse snapshots can be later: 66 of 756,179 pairs (0.009%) are later same-day user-0 snapshots. Negligible, and weaker than on a test row.

### INFO 7. Checks with no finding

- Leak audit: no target file is opened by `prep.py`, `kernels.py`, `views.py`; my recount reproduces the files without any label. The inputs are history snapshots and candidate impressions only.
- Read-only files: `nn_x/w5/cd_arms6.py`, `nn_x/abl/cd_arms6.py` (11-12 22:05), `nn_x/abl/train.py` (11-14 07:25), `nn_x/w5/data.py`, `finetune.py`, every file under `src/`, `feat_new/`, `data/feats_new/`, `nn/feat_cache/`, `data/flat/` (recursive): none modified since 11-15 02:00. Under `nn_x/w5` and `nn_x/abl` the only code files modified since then are `train.py`, `lane.sh` and the new `cd_arms6b.py`.
- `nn_x/w5/train.py` and `lane.sh` changed at 04:57:57Z. The change carries edits that are not in the study's proposal (`--xt_drop_init`, arms xd15 / xd30 / xd50 / hzxd / s3 / h2t / h2x) and the lead's rule file records the adoption at 04:57:59Z: consistent with the lead having applied it.
- `train.py` against the study's pre-patch copy: the import line plus the lead's `--xt_drop_init` block; the flag-off path (`tr_hz = None`, one conditional gather) is untouched; the module does nothing at import beyond reading two environment variables.
- `lane.sh` unsets both variables before every job and exports them only for hz2 / h2t / h2x / hz2L.
- GPU: 32 compute processes, none started from `nn_x/w12/stale/hz2`. No process of the study is running.
- View files keep their 02:46-03:01Z times; every hz2 lane started after 04:58Z.

## 3. What the lead should do

Nothing blocks the running lanes. Before the read and before any second wave:

1. **Rename the arm in the rule file**: hz2 / hz2k = test-like coverage by listing age plus depth scaling, warm rows 1-2 d stale. It does not test the 9-11 d warm-item staleness.
2. **Read `@last` next to the selected checkpoint** on every witness (selection uses fresh-input holdout feeds; on the 11-12 models my CPU-replay pools read +0.00044 ± 0.00040 selected against -0.00039 ± 0.00036 at `@last` under fresh inputs).
3. **Do not treat stale-val and the lagged holdout through this view as two readings of the same inputs**: stale-val has the no-ID encoder on newly cold rows, this view does not.
4. **Second wave**: extend the drop-in so that rows with view `it_hist_all` = 0 and stored `it_hist_all_luo` > 0 read `pre2_fe_s` = (`pre2_fe_noid` + 20) / 6 and a `pre2_fe_z` rebuilt as in `solvers/CPU2_stale_replay_pre2s_and_arms.py::build_pre2both`. Gate: the count of cold rows with ID contribution must be 0, as on test.
5. **If warm-row lag is the target**: `feats_L` on Sep 17-19 feeds only; no arm of that kind has started (no hz2L, h2t, h2x log exists).
6. Sep 6-8 feeds: the keep hybrid fails every criterion; dropping those feeds from the cooldown of hz2-type arms is the cleaner option. I did not test it.
7. For the arm read, each hz2 run's `nn_<v>_holdout_hz.npy` is a valid lagged-holdout score of its selected checkpoint (keep rows only on Sep 6-8 feeds); reference and hz models need `hold_infer2.py feats ... late`, which I validated against that file.

## 4. Method

- `recount.py`: plain numpy from `data/flat` (history events of the sampled items in every train / val / test snapshot, impressions per day, my own later-feed mask). The hz2 rule is implemented from the text of `DECLARATION.txt`, with no import from the study and none of its cache or mask files; it reads only the per-feed lag file and the columns under test. 40 random feeds per dense day (480 feeds, 37,918 rows, seed 2024) plus 250 val and 400 test feeds for the stored rule.
- `gate_own.py`: zero share, mean and binned distribution by listing-age bucket, trPool (134,118 feeds, own holdout rule, equal to poplag's `hold.npy`) against private (12,186 feeds, 1,829,811 rows) and public (5,248 feeds, 806,026 rows) test rows with M >= 50. All days, >= Sep 13 and three day groups; views as built, hz, hz2, hz2k, hz2L.
- `pre2_tell.py`: joint of the swapped counter with the unswapped encoder inputs; within-feed AUC on train labels of non-holdout pool feeds.
- `read_own.py`: my own NDCG (binary gain, pessimistic ties, feeds without a positive = 1) and pooling on the study's cached predictions; labels from `data/flat/train/cand_target.npy` at the holdout rows.
- `dropin_own.py`: my own split object over the full train row space with the 8 stored columns; `cd_arms6.py`, and `nn_x/w5/cd_arms6b.py` loaded twice (variables unset, and the hz2 lane environment).
- `infer_check.py` + `infer_check_read.py`: the study's reader on one finished hz2 checkpoint against the trainer's GPU file.

## 5. Limits and what I did NOT check

- Causality from raw data is a SAMPLE (480 feeds). The every-row check C1 of the study rests on its own mask files; I did not rebuild all rows.
- Not recounted: `it_hist_hz_et2..et4`, `seq_cv_hz_invd20`, `seq_cv_hz_norm` (same code path and the same member set as the columns I recounted); `seq_cv_hz_pop` rows above the 3000 cap.
- The 20 cached models' predictions through the view were taken from the study's cache; I re-inferred one model only (path control against the GPU file). Members of W16 beyond the 8 scored, and the d192 popularity-light students, are not covered by any labelled number.
- No read of the hz2 arm: five runs had finished at 06:05Z; I used one prediction file as a path control and applied no arm rule.
- No val or test label was opened. No GPU, no model trained. The size of issue 2 under training is unmeasured.
- The time order of the study's declarations cannot be verified from the file system (the files were appended later). What is certain: no hz2 run had finished before 05:49Z, so the pre-registration predates every hz2 result.
- Not re-run: `nn_x/w5/tests/test_hzcd.py` against the patched import; the checksums in `MD5SUMS_feats.txt` (modification times only); the lead's `h2t` / `h2x` arms (read, not executed).
- Gate reference is test feeds with M >= 50 and the trPool population, as in the study; no sensitivity to bucket edges or to another lag draw.
- My own slips: two headers of `REVIEW_RULES.txt` and one of `PROGRESS.md` carried estimated clock times about 10-25 minutes late; corrected by a note in each file. The order (rules before reads) is as stated. `infer_check.py` imports the study's reader with bytecode writing off; nothing was written outside my directory and this report.

## 6. Commands

```
cd <repo>
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
PY=.venv/bin/python; D=nn_x/w12/stale/hz2_review
$PY $D/recount.py 40 2024                  # 30 s  -> logs/recount_40_2024.log, recount_{primary,lagfirst}_seed2024.parquet
$PY $D/gate_own.py                         # 85 s  -> logs/gate_own.log, gate_own.json
$PY $D/read_own.py                         # 30 s  -> logs/read_own.log
$PY $D/pre2_tell.py                        # 10 s  -> logs/pre2_tell.log
$PY $D/dropin_own.py                       # 80 s  -> logs/dropin_own.log
$PY $D/infer_check.py w5_kcd_hz2_s958      # 220 s -> chk_hpred_feats_late_w5_kcd_hz2_s958.npy
$PY $D/infer_check_read.py w5_kcd_hz2_s958 # 20 s  -> logs/infer_check_read.log
nvidia-smi --query-compute-apps=pid,name --format=csv   # look only
```

## 7. Files created

All under `nn_x/w12/stale/hz2_review/`: `REVIEW_RULES.txt`, `PROGRESS.md`, `recount.py`, `gate_own.py`, `read_own.py`, `pre2_tell.py`, `dropin_own.py`, `infer_check.py`, `infer_check_read.py`, `logs/` (one log per command), `recount_primary_seed2024.parquet`, `recount_lagfirst_seed2024.parquet`, `gate_own.json`, `chk_hpred_feats_late_w5_kcd_hz2_s958.npy`. Report: `reports/STALE_1115/hz2_review.md`.
