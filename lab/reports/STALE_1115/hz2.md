# hz2: a causal, test-like popularity view for TRAINING rows (all 8 W8 inputs + `xp_imp_c50`)

Written 2026-11-15 by the hz2 agent in two runs (02:18-03:22 UTC, cut by an account limit; resumed 03:31 UTC from `PROGRESS.md`, nothing redone). CPU only. Work directory `nn_x/w12/stale/hz2/`. Conventions: ± = 1 SE paired per feed; [M] measured by me, [D] read in a project file, [I] inferred. "Zero share" = share of candidate rows whose counter is 0. Day slots: Sep 6 = slot 0, Sep 20 = slot 14 (dense cutoff of val / test rows).

## 0. Verdict

1. **The view is built for all 246,745,836 train rows (8 W8 inputs + `xp_imp_c50`, per-feed test-like lag of 6-9 d), it is strictly causal, and the drop-in is live.** No row counts a dense day on or after its own day and no feed a sparse day after its own day (checked on every row, not a sample). The same code reproduces the stored val and test columns (six count columns bit-exact on every row, the three float columns to 3e-6). `cd_arms6b.py` is bit-identical to `cd_arms6` when its two variables are unset; the lead applied it at 04:57 UTC and the first hz2 run logged the expected load (8 columns, 46,918,534 keep rows).
2. **Declared label-free gate: FAIL on all days, PASS on days >= Sep 13 (both parts).** It closes 90% / 88% / 84% of the zero-share gap to private rows (`it_hist_all` / `seq_cv_pop` / `xp_imp_c50`) on those days, against 32% / 0% / 0% for the half-dose hz view. Sep 6-8 (27% of train feeds) cannot be made test-like causally: the dense data start on Sep 6. Two thirds of the rows (listings alive before Sep 6) are a calibrated approximation, not a replay.
3. **The staleness cost is real under a causal view** (13,424 holdout feeds of days >= Sep 13, train labels): W8 pool -0.0045 ± 0.0007 (Sep 17-19, the cleanest days: -0.0025 ± 0.0009); popularity-light pool -0.0012 ± 0.0004; the popularity-light admixture goes from -0.00050 ± 0.00032 (fresh inputs) to +0.00032 ± 0.00031 (test-like), D +0.00082 ± 0.00039.
4. **Training on a stale-like view helps on test-like inputs**: the four half-dose hz models of 11-12 beat their references through the causal view by +0.0007 ± 0.0004 (declared read, selected checkpoints: "not separated" by my rule) and by +0.0008 ± 0.0004 at the @last checkpoints (post hoc: "supports"; fresh-input reading of the same models -0.0004 ± 0.0004); positive in all 16 single-model pairs across two checkpoints and two views.
5. Pre-registered for the hz2 arm (before any run): stale-val +0.0018 [+0.0008, +0.0032] against the reference, +0.0007 over hz; lagged holdout +0.0015; val about 0; in-period holdout -0.0010 (not a veto). Falsifiers F1-F5 in section 7.
6. Not covered: the encoder-score inputs (`pre2_fe_*`) have the same coverage gap and are outside the view; old listings are not lagged in the primary view (the lag-first variant lags them and costs the W8 pool no measurable extra: -0.00035 ± 0.00057).
7. For the lead: read the hz2 arm at `@last` as well as at the selected checkpoint (selection on the fresh-input holdout moved the 11-12 reading by 0.0010); the causal lagged-holdout reader for any two arms is `lagged_pair_read.py` (section 10); do not move or delete `feats/` and `feats_L/` while hz2 lanes are queued; the popularity-light students lose 0.0012 under the view; the drop-in, copied next to `nn_x/abl/train.py`, swaps their one counter (test T3b; no such run exists).

## 1. Why train rows and test rows differ, input by input

Sources read: `src/features.py` l.239-251 and `src/features_luo.py` (`it_hist_*_luo`), `feats_v2/seq_/feat_covis.py` + `sources.py` (`seq_cv_*`), `src/features_tte.py` + `nn_x/abl/pop1/build_xp_imp_c50.py` (`xp_imp_c50`). The reading is confirmed numerically: my kernels with the stored rules reproduce the stored val columns on all 1,389,971 val rows and the stored train columns on 20,000 random train feeds (section 3.1) [M].

Calendar [M, `logs/sizes.log`]: dense train days Sep 6-14 and Sep 17-19 (12 days, 203,548-291,760 feeds each; Sep 15-16 missing). val Sep 22 / 24; public Sep 26 (6,050 feeds) and Sep 27 before 11:32 (2,512); private Sep 27 from 11:32 (4,855), Sep 28 (7,578), Sep 29 (7,600). Relative to the dense cutoff c = Sep 20 00:00 MSK a val / test row therefore sees dense days c + {-14..-6, -3, -2, -1} (the hole sits at c-5, c-4) and sparse days c + {2, 4, 6, 7, 8, 9} at about 1/37 of the dense density (7.0-7.6K feeds per day against 204-292K). Lag of the row L = day - c: public 6 / 7, private 7 / 8 / 9.

| input (count) | what is counted | TRAIN row, as built | TEST row | difference |
|---|---|---|---|---|
| `it_hist_{all,et1..et4}_luo` (5) | occurrences of the item in the history snapshots of ALL feeds of train + val + test, minus snapshots of the same non-zero user with a later timestamp | all 12 dense days: before, on AND after the row's day (not causal), plus every val / test snapshot. Staleness 0: the row sits inside the window | dense days end 6-9 d before the row; afterwards 6 sparse days (the later ones lie after the row: transductive); a listing created after Sep 20 is seen only by sparse snapshots | coverage (dense vs 1/37) for listings younger than the lag; exposure length for listings aged lag..lag+14 d; displacement of the window for old listings; leave-user-later-out is the same rule on both sides |
| `seq_cv_{invd20,pop,norm}` (3) | co-visitation of the candidate with the feed's history items in de-duplicated user sequences (latest train feed per non-zero user; every user-0 feed); most recent 3000 occurrences; sequences of the row's own user skipped | train days strictly BEFORE the own day (expanding window: 0 dense days on Sep 6, 11 on Sep 19); no sparse part; staleness 1 d | all 12 dense days + val / test histories with ts < own ts (sparse, causal); staleness 6-9 d for the dense part | depth (0-11 days vs 12) and lag (1 d vs 6-9 d), plus the same coverage effect for young listings |
| `xp_imp_c50` = log1p(min(n, 50)) (1; popularity-light members) | n = train impressions of the item (candidate rows) | train days before the own day (0 on Sep 6) | all 12 train days; val / test impressions are never counted, so a listing created after Sep 20 is always 0 | pure lag of 6-9 d with no sparse refresh; depth as for `seq_cv` |

Which snapshots count for a listing of age a on a test row with lag L: none of the dense days if a < L (created after the cutoff: sparse only); the first (a - L) days of its life if L <= a < L + 14 d (minus the Sep 15-16 hole); all 12 dense days if older. On an as-built train row the same listing has all of its life inside dense data. That is the whole gap: at trPool rows the zero share of `it_hist_all_luo` is 0.38-0.69 / 0.20-0.43 / 0.16-0.31 for listings aged < 1 d / 1-3 d / 3-7 d against 0.975 / 0.916 / 0.859 on private rows (`logs/diag_m.log`) [M].

## 2. The train-row counterpart (definition; `DECLARATION.txt`, first block, written before any column existed)

Per train feed f (all 3,151,208, the 30,000 holdout feeds included): part = public-like with probability 8,562 / 28,595 (hash of `feed_id`), lag L_f drawn inside the part from the test feed counts (public {6, 7}, private {7, 8, 9}; poplag's hash and cdf, so a holdout feed keeps the lag poplag gave it in the matching view). Result [M]: 666,704 / 811,819 / 835,005 / 837,680 feeds at lag 6 / 7 / 8 / 9; public-like share 0.2994 (test 0.2994). Cutoff c_f = day(f) - L_f.

The row replays the test observation process at its own calendar position:
- dense days = c_f + {-14..-6, -3, -2, -1} (the Sep 15-16 hole is replayed), sparse days = c_f + {2, 4, 6, 7, 8, 9} that are not after the row's day; sparse snapshots = the shared 1/37 hash sample of train feeds (`feat_new/hz_common.sparse_flag`);
- `it_hist_*` = dense occurrences + sparse occurrences + the row's own snapshot, minus later snapshots of the same non-zero user; `seq_cv_*` = the stored kernel over dense members (latest feed per user among the dense days) and sparse members with ts < own ts, own user skipped; `xp_imp_c50` = impressions on the dense days only;
- **nothing dated after the row's own day is counted in any column, and no dense snapshot of the own day**.

EXACT rows (34.2% of the 246.7M rows; 32.6% are listings created on or after the cutoff, 1.6% listings created inside the data before it): every test-pattern day on which the listing was alive is observed, so the value is the plain rule "counts up to T - lag". No listing age and no parameter enters.

TRUNCATED rows (65.8%): the listing was alive before Sep 6, where no data exist. The dense data span 14 days; a causal replay of an old listing needs lag + depth = 18-23 days of past. **A causal view with the test lag and the test depth cannot be built for these rows from this data**; this is the T4 depth collapse (`COVERAGE_MAP` row 220) and the reviewer's closing remark in `poplag_verify.md`. For them the view is an approximation with one parameter m (minimum dense depth in days): dense window = the earliest min(X_test, n) dense days in [Sep 6, min(day, max(c_f, Sep 6 + m))), X_test = number of dense days a test row of this age and lag would see, value = r x dense part + sparse part, r = min(X_test / window days, 4). Small m keeps the lag and loses depth; large m keeps depth and loses the lag. A feed of Sep 6 has no past at all under any m.

Own user. The task text says "nothing from the row's own user". I kept the stored rule of each column instead and say so: `it_hist` counts the row's own snapshot and EARLIER snapshots of the user (a test row has them; the val check requires the same code to reproduce the stored val columns), never a later one; `seq_cv` skips every sequence of the user; `xp_imp` counts earlier impressions only. Nothing that can carry the row's label is counted.

Strict causality has a price that the task accepted: a test row also counts the sparse snapshots of other users dated after it (2 of 6 sparse days at lag 7, 3 at lag 6). Without them the zero share of listings younger than 7 d is 0.989 / 0.947 / 0.905 instead of 0.978 / 0.928 / 0.885, against 0.975 / 0.916 / 0.859 on private rows (`logs/diag_m.log`) [M].

## 3. Code checks and calibration

### 3.1 Code checks V1, V2 (before any view was used) [M]

| check | rows | `it_hist_*` (5), `seq_cv_pop` | `seq_cv_invd20`, `seq_cv_norm`, `xp_imp_c50` |
|---|---|---|---|
| V1: val rows, same kernels, cutoff Sep 20, stored (transductive) sparse rule, against `nn/feat_cache/val` | 1,389,971 | exact on every row | max relative difference 2.8e-6 / 1.3e-6 / 4.9e-8 |
| V2: 20,000 random train feeds, as-built rules, against `nn/feat_cache/train` | 1,580,265 | exact on every row | 1.8e-6 / 1.5e-6 / 4.9e-8 |
| V1 information: the same val rows with the strictly causal sparse rule | 1,389,971 | `it_hist_all_luo` differs on 16.0% of rows; zero share 0.405 against 0.379 stored | `seq_cv_*` unchanged except 30 rows at the 3000 cap |
| V1 on TEST rows, stored rule, against `nn/feat_cache/test` (`logs/v1_test.log`) | 2,904,806 | exact on every row | 2.0e-6 / 1.2e-6 / 4.9e-8 |
| the same test rows with the strictly causal sparse rule | 2,904,806 | `it_hist_all_luo` differs on 6.7% of rows; zero share 0.460 against 0.447 stored | `seq_cv_pop` differs on 38 rows |

So the stored val / test columns are the hz2 construction at lag 2-9 with the transductive sparse rule ("the val-row counterpart built with the same code reproduces the stored val columns": yes, on every row, and the same holds for the test rows), and the stored train columns are the same kernels with other day masks. `logs/v1_final.log`, `logs/v1_test.log`, `logs/v2_final.log`.

### 3.2 Calibration of m (declared rule: smallest m meeting G1 and G2 for `it_hist_all_luo` on trPool private-like feeds against private rows; else argmin dZ) [M]

| m | 0 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | as built |
|---|---|---|---|---|---|---|---|---|---|
| all days (93,819 feeds): dZ | 0.309 | 0.126 | 0.105 | 0.092 | 0.083 | 0.076 | 0.068 | **0.064** | 0.259 |
| all days: largest bucket | 0.481 | 0.201 | 0.170 | 0.150 | 0.135 | 0.124 | 0.111 | 0.104 | 0.647 |
| days >= Sep 13 (42,796 feeds): dZ | 0.178 | 0.085 | 0.067 | 0.054 | **0.043** | 0.034 | 0.019 | 0.022 | 0.225 |
| days >= Sep 13: largest bucket | 0.279 | 0.138 | 0.111 | 0.086 | 0.061 | 0.047 | 0.053 | 0.059 | 0.592 |

- No m meets G1 (dZ <= 0.05) and G2 (bucket <= 0.10) on all days. **Rule as written: m* = 12 (argmin dZ) = the PRIMARY view, `feats/`.**
- I think the rule is wrong for the purpose and applied it anyway. The all-day statistic is dominated by Sep 6-8, where old listings have 0-2 dense days in the past whatever m is (zero share 0.57 against 0.20). On later days m = 12 buys coverage by giving up the lag: the nearest dense day of an old listing is then 1-2 d before the row on every day, so old listings (53% of rows) are not stale at all. The criterion cannot see lag (`logs/diag_m.log`).
- **Second variant, declared at 02:45Z before any build: hz2L (lag-first), `feats_L/`, m_L = 5** = the smallest m meeting G1 and G2 on days >= Sep 13 (the "late" subset named in the first declaration). Nearest dense day of an old listing under hz2L: 6-8 d before the row on Sep 17-19, 2-3 d on Sep 13-14, 0-1 d before Sep 13.

Old listings (> 24 d) by feed-day group, trPool private-like, `it_hist_all_luo` (zero share 24-76 d / > 76 d; private rows 0.192 / 0.200; mean log1p 2.13 / 2.20) [M, `logs/diag_m.log`]:

| feed days | as built | m = 0 (pure lag) | m = 4 | m = 6 | m = 12 (primary) | nearest dense day before the row, m = 0 / 4 / 6 / 12 |
|---|---|---|---|---|---|---|
| Sep 6-8 | 0.123 / 0.111 | 0.887 / 0.888 | 0.575 / 0.565 | 0.575 / 0.565 | 0.575 / 0.565 | none / 1.0 / 1.0 / 1.0 d |
| Sep 9-12 | 0.133 / 0.133 | 0.819 / 0.823 | 0.261 / 0.267 | 0.242 / 0.248 | 0.242 / 0.248 | none / 1.7 / 1.0 / 1.0 d |
| Sep 13-14 | 0.135 / 0.138 | 0.751 / 0.758 | 0.261 / 0.278 | 0.209 / 0.223 | 0.186 / 0.196 | 8.0 (87% of rows have none) / 4.5 / 2.5 / 1.0 d |
| Sep 17-19 | 0.144 / 0.152 | 0.293 / 0.314 | 0.269 / 0.290 | 0.228 / 0.250 | 0.173 / 0.188 | 9.1 / 8.6 / 7.0 / 2.0 d |

Reading: with the true lag (m = 0) the mean of old listings on Sep 17-19 is at parity after the depth factor (2.21 / 2.18 against 2.13 / 2.20) but 3.8 dense days instead of 12 leave 10 points too many zeros. That is the price of depth, not of the rule.

## 4. Label-free gates (criteria in `DECLARATION.txt` before any comparison; `logs/gate_final.txt`, `logs/gate_summary_final.md`) [M]

Populations: trPool = 134,118 train feeds with p_random > 0.5, M >= 50, not holdout (the cooldown pool; decision population); trHold = the 30,000 holdout feeds; trAll = all feeds (information). Reference = test rows of feeds with M >= 50 of the matching part (private 1,829,811 rows of 12,186 feeds; public 806,026 rows of 5,248 feeds). dZ = row-weighted mean over seven listing-age buckets of |zero-share gap|; "bucket" = largest gap among buckets with >= 5% of rows; dM = the same for the mean in reference SD. Bars: G1 dZ <= 0.05, G2 bucket <= 0.10, G3 dM <= 0.20 (three primaries); G4 `it_hist_all_luo` dZ <= one third of the as-built value; G5 six secondary columns dZ <= 0.10, dM <= 0.30. Views: as built; hz (half dose: five `it_hist` files of `data/feats_new`, the other inputs as built); hz2 (primary, m = 12); hz2L (lag-first, m = 5); "k" = the same with the keep mask (truncated rows of Sep 6-8 feeds left as stored).

### 4.1 Decision table: trPool, private-like feeds against private rows (dZ / bucket / dM)

| view | days | `it_hist_all_luo` | `seq_cv_pop` | `xp_imp_c50` | secondary, max dZ | G1 G2 G3 G4 G5 | verdict |
|---|---|---|---|---|---|---|---|
| as built | all (13.81M rows) | 0.259 / 0.647 / 0.531 | 0.144 / 0.407 / 0.266 | 0.228 / 0.809 / 0.397 | 0.256 | - | - |
| hz (half dose) | all | 0.174 / 0.420 / 0.384 | 0.144 / 0.407 / 0.266 | 0.228 / 0.809 / 0.397 | 0.174 | all five fail | FAIL |
| **hz2 (primary)** | all | 0.064 / 0.104 / 0.046 | 0.061 / 0.107 / 0.041 | 0.050 / 0.159 / 0.091 | 0.064 | fail fail ok ok ok | **FAIL** |
| hz2k | all | 0.032 / 0.113 / 0.160 | 0.063 / 0.111 / 0.063 | 0.049 / 0.159 / 0.084 | 0.032 | fail fail ok ok ok | FAIL |
| hz2L | all | 0.083 / 0.134 / 0.056 | 0.080 / 0.137 / 0.043 | 0.084 / 0.177 / 0.075 | 0.084 | fail fail ok ok ok | FAIL |
| as built | >= Sep 13 (6.35M rows) | 0.225 / 0.591 / 0.429 | 0.111 / 0.440 / 0.172 | 0.232 / 0.907 / 0.431 | 0.221 | - | - |
| hz (half dose) | >= Sep 13 | 0.153 / 0.376 / 0.325 | 0.111 / 0.440 / 0.172 | 0.232 / 0.907 / 0.431 | 0.153 | all five fail | FAIL |
| **hz2 (primary)** | >= Sep 13 | 0.022 / 0.058 / 0.107 | 0.014 / 0.055 / 0.107 | 0.036 / 0.070 / 0.144 | 0.021 | ok ok ok ok ok | **PASS** |
| hz2L | >= Sep 13 | 0.043 / 0.061 / 0.079 | 0.039 / 0.064 / 0.098 | 0.053 / 0.099 / 0.063 | 0.046 | fail ok ok ok ok | FAIL (G1 `xp_imp_c50` 0.053) |

- Public-like feeds against public rows: the same verdicts. hz2 all days FAIL (0.083 / 0.065 / 0.064), days >= Sep 13 PASS (0.030 / 0.008 / 0.030); hz2L days >= Sep 13 FAIL by G1 (0.051 / 0.032 / 0.053); hz FAIL everywhere.
- Holdout feeds (information): hz2 days >= Sep 13 PASS for both parts (private-like 0.018 / 0.014 / 0.029, 1.46M rows); hz2L FAIL by G1 and G2 (`xp_imp_c50` 0.059 / 0.110).
- By feed-day group (hz2, private-like, trPool): Sep 6-8 FAIL (0.235 / 0.231 / 0.232; as built 0.301 / 0.280 / 0.351); Sep 9-12 PASS (0.047 / 0.043 / 0.030); Sep 13-14 PASS (0.023 / 0.017 / 0.039); Sep 17-19 PASS (0.024 / 0.014 / 0.034). hz2L: Sep 9-12 and Sep 13-14 PASS, Sep 17-19 FAIL (`xp_imp_c50` 0.067 / bucket 0.119).
- All train feeds (information; production feeds carry more popular items): `it_hist_all_luo` dZ 0.340 as built, 0.243 hz, 0.051 hz2, 0.063 hz2L; on days >= Sep 13 hz2L is the closer one (0.026 against 0.045).
- **Declared verdict: hz2 FAIL on all days, PASS on days >= Sep 13.** The all-day failure is Sep 6-8 (27.4% of train feeds, 24.9% of pool feeds; corrected in the resumed run, the first draft said 17%): no causal past exists there, and the stored `seq_cv_*` / `xp_imp_c50` columns have the same defect on those days (dZ 0.28 / 0.35 as built).
- Where the gate holds, in training mass [M]: days >= Sep 13 (the declared secondary subset; both parts pass, and so do Sep 13-14 and Sep 17-19 separately) = 40.2% of train feeds and 45.5% of pool feeds. Sep 9-12 (32.4% / 29.7%) passes for the private-like part and fails G1 for the public-like part (`it_hist_all_luo` dZ 0.057 against the bar 0.05). Sep 6-8 fails every criterion in both parts. With `HZ2_KEEP` the other 27.4% of feeds keep the stored values on their truncated rows (19.0% of all rows) and take the view on their young listings.
- Share of the as-built gap that a view closes (trPool private-like, all days / days >= Sep 13): hz 33% / 32% for `it_hist_all_luo`, 0% for the `seq_cv` and `xp_imp` inputs; hz2 75% / 90%, 58% / 88%, 78% / 84%; hz2L 68% / 81%, 45% / 65%, 63% / 77%.

Per input, all nine (dZ / dM; trPool feeds of days >= Sep 13; the all-day tables are in `logs/gate_per_input.md`, every bucket in `logs/gate_final.txt`) [M]:

| input | private-like against PRIVATE rows (6,353,027 rows): as built | hz | hz2 | hz2L | public-like against PUBLIC rows (2,709,205 rows): as built | hz | hz2 | hz2L |
|---|---|---|---|---|---|---|---|---|
| `it_hist_all_luo` | 0.225 / 0.429 | 0.153 / 0.325 | 0.022 / 0.107 | 0.043 / 0.079 | 0.197 / 0.386 | 0.125 / 0.282 | 0.030 / 0.102 | 0.051 / 0.074 |
| `it_hist_et1_luo` | 0.221 / 0.418 | 0.153 / 0.320 | 0.021 / 0.110 | 0.046 / 0.084 | 0.194 / 0.375 | 0.125 / 0.278 | 0.027 / 0.104 | 0.053 / 0.078 |
| `it_hist_et2_luo` | 0.138 / 0.275 | 0.105 / 0.224 | 0.014 / 0.084 | 0.046 / 0.051 | 0.125 / 0.246 | 0.091 / 0.196 | 0.013 / 0.073 | 0.044 / 0.042 |
| `it_hist_et3_luo` | 0.135 / 0.290 | 0.101 / 0.232 | 0.011 / 0.078 | 0.046 / 0.031 | 0.123 / 0.261 | 0.090 / 0.204 | 0.010 / 0.069 | 0.043 / 0.025 |
| `it_hist_et4_luo` | 0.095 / 0.205 | 0.074 / 0.169 | 0.017 / 0.055 | 0.045 / 0.032 | 0.086 / 0.185 | 0.065 / 0.150 | 0.014 / 0.050 | 0.041 / 0.025 |
| `seq_cv_invd20` | 0.028 / 0.091 | as built | 0.012 / 0.103 | 0.017 / 0.091 | 0.024 / 0.078 | as built | 0.010 / 0.096 | 0.015 / 0.083 |
| `seq_cv_pop` | 0.111 / 0.172 | as built | 0.014 / 0.107 | 0.039 / 0.098 | 0.099 / 0.147 | as built | 0.008 / 0.097 | 0.032 / 0.086 |
| `seq_cv_norm` | 0.028 / 0.087 | as built | 0.012 / 0.080 | 0.017 / 0.067 | 0.024 / 0.076 | as built | 0.010 / 0.074 | 0.015 / 0.061 |
| `xp_imp_c50` | 0.232 / 0.431 | as built | 0.036 / 0.144 | 0.053 / 0.063 | 0.195 / 0.361 | as built | 0.030 / 0.127 | 0.053 / 0.054 |

The view closes the zero-share gap of every input by a factor of 2.3 to 12; the mean gap dM falls for eight inputs and rises a little for `seq_cv_invd20` (0.091 to 0.103 private-like, 0.078 to 0.096 public-like), whose level was already near the test level on train rows. The lag-first view has the better means (levels of old listings) and the worse zero shares, as designed.

### 4.2 By listing age: trPool private-like feeds of days >= Sep 13 (zero share; last column = private rows)

| listing age | rows | `it_hist_all_luo`: as built / hz / hz2 / hz2L / private | `seq_cv_pop`: as built / hz2 / hz2L / private | `xp_imp_c50`: as built / hz2 / hz2L / private |
|---|---|---|---|---|
| < 1 d | 11.4% | 0.621 / 0.841 / 0.989 / 0.989 / **0.975** | 0.980 / 0.997 / 0.997 / **0.996** | 0.804 / 1.000 / 1.000 / **1.000** |
| 1-3 d | 7.8% | 0.379 / 0.643 / 0.949 / 0.949 / **0.916** | 0.745 / 0.965 / 0.965 / **0.955** | 0.296 / 1.000 / 1.000 / **1.000** |
| 3-7 d | 10.0% | 0.268 / 0.484 / 0.906 / 0.906 / **0.859** | 0.459 / 0.924 / 0.924 / **0.899** | 0.093 / 1.000 / 1.000 / **1.000** |
| 7-10 d | 4.7% | 0.209 / 0.291 / 0.788 / 0.788 / **0.786** | 0.312 / 0.803 / 0.803 / **0.816** | 0.062 / 0.676 / 0.676 / **0.709** |
| 10-24 d | 12.8% | 0.173 / 0.173 / 0.337 / 0.368 / **0.395** | 0.245 / 0.354 / 0.384 / **0.410** | 0.109 / 0.182 / 0.218 / **0.130** |
| 24-76 d | 17.0% | 0.141 / 0.141 / 0.178 / 0.241 / **0.192** | 0.196 / 0.199 / 0.259 / **0.208** | 0.172 / 0.178 / 0.291 / **0.192** |
| > 76 d | 36.3% | 0.147 / 0.147 / 0.191 / 0.261 / **0.200** | 0.208 / 0.211 / 0.278 / **0.214** | 0.212 / 0.219 / 0.351 / **0.288** |

Residuals of the primary view on these days [M]: listings younger than 7 d are 0.014-0.047 colder than private rows (the price of strict causality plus sparse days lost in the Sep 15-16 hole); listings older than 24 d have the right zero share but a level 0.2-0.3 above private rows on the log1p scale (`it_hist_all_luo` 2.42 / 2.39 against 2.13 / 2.20; as built 2.51 / 2.49): **the displacement of the window is not reproduced for old listings in the primary view**. hz2L reproduces a third of it (2.35 / 2.31) and pays with 5-6 points too many zeros.

### 4.3 Joint checks beyond the declared gate (`logs/joint.txt`, label-free) [M]

Per-feed cold share (share of rows with `it_hist_all` = 0) and joint zero pattern `it_hist_all` x `seq_cv_pop` by age bucket (row-weighted total-variation distance to private rows), private-like feeds:

| population | view | cold share q10 / q50 / q90 | feeds >= 0.45 | joint pattern TV |
|---|---|---|---|---|
| private test feeds, M >= 50 (12,186) | stored | 0.337 / 0.452 / 0.581 | 50.9% | 0 |
| trPool, days >= Sep 13 (42,796 feeds) | as built | 0.115 / 0.230 / 0.360 | 2.4% | 0.225 |
| | hz | 0.172 / 0.303 / 0.448 | 9.8% | 0.179 |
| | hz2 | 0.332 / 0.456 / 0.590 | 52.3% | 0.022 |
| | hz2L | 0.366 / 0.497 / 0.633 | 67.7% | 0.045 |
| trPool, all days (93,819 feeds) | as built | 0.086 / 0.193 / 0.325 | 1.3% | 0.317 |
| | hz | 0.150 / 0.282 / 0.431 | 7.6% | 0.255 |
| | hz2 | 0.351 / 0.495 / 0.754 | 64.5% | 0.067 |
| | hz2k | 0.312 / 0.446 / 0.589 | 48.5% | 0.091 |
| trHold, days >= Sep 13 (9,379 feeds) | hz2 | 0.350 / 0.471 / 0.600 | 58.9% | 0.021 |

The keep mask brings the all-day feed-level cold share to the private one (q90 0.589 against 0.581) and worsens the joint pattern on Sep 6-8 (stored `it_hist` warm, stored `seq_cv` shallow). Neither option is test-like on Sep 6-8.

### 4.4 Causality check V3 (`logs/audit_feats.txt`, `logs/audit_feats_L.txt`) [M]

Independent recount in plain numpy from the rule text (no cache file of the builder, no kernel), 240 random train feeds (60 per day group), 19,163 rows:

| quantity | feats (primary) | feats_L |
|---|---|---|
| counted (row, snapshot) pairs of `it_hist` | 405,550 (dense 401,318, sparse 4,232) | 294,230 (dense 287,979, sparse 6,251) |
| pairs dated after the row's day | **0** | **0** |
| dense pairs on the row's own day | 0 | 0 |
| pairs of the same non-zero user with a later timestamp | 0 | 0 |
| pairs from any snapshot of the row's own user | 6 (0.0015%) | 4 (0.0014%) |
| sparse pairs of other users later on the same day (allowed by the rule) | 1,617 | 1,617 |
| recount against the files, nine columns | max relative difference 1.1e-7 | 1.1e-7 |
| `seq_cv_*` recount (80 feeds, 6,320 rows): member sequences dated after the row | 0 (asserted) | 0 |

A candidate item is never in the feed's own history: 0 of 1,389,971 val rows and 0 of 1,566,446 sampled train rows [M]. So the own snapshot contributes nothing and "nothing from the row's own user" holds to 1.5e-5 of the counted pairs under the stored rule.

Checks added by the resumed run (criteria in `DECLARATION.txt` ADDENDUM 5, written before they ran) [M]:

| check | scope | result |
|---|---|---|
| C1 structural causality, every row (`causal_all.py`, `logs/causal_all.txt`): the kernels read counts only on the day slots of the row's dense mask and of the feed's sparse mask | all 246,745,836 rows x both views; all 3,151,208 feeds | rows with a dense day on or after the own day: **0** and **0**; rows with a slot that is not one of the 12 dense train days: 0; feeds with a sparse day after the own day: **0**. PASS |
| C2 cache re-check on a NEW sample (`audit2.py`, seed 11; 160 feeds, 12,192 rows; `logs/audit_feats_seed11.txt`, `logs/audit_feats_L_seed11.txt`) | both views | 224,593 counted pairs (primary) and 160,482 (lag-first): 0 pairs after the row day, 0 later same-user pairs, 0 dense pairs on the own day; nine columns equal to the files (max relative difference 1.2e-7). PASS |
| C3 file integrity | 18 column files + lag files | 246,745,836 float32 values each, no NaN / inf / negative; lags in {6, 7, 8, 9}; lag shares by part equal the test shares to 1e-4 (public 0.7066 / 0.2934, private 0.2423 / 0.3782 / 0.3794). PASS |

C1 also shows how far the lag is real (share of the rows that have a dense day, by distance own day - latest dense day): primary view 65.2% at 1 d, 7.2% at 2 d, 7.7% at 3 d, 19.9% at 4-10 d; lag-first view 39.9% at 1 d, 6.3% at 2 d, 8.0% at 3 d, 45.8% at 4-10 d. 29.8% of all rows have no dense day at all (listings created after the lagged cutoff, plus Sep 6 feeds). **In the primary view the listings that have dense history are mostly fresh; what is test-like is the coverage by listing age, not the displacement of the window.**

### 4.5 Model-free conditional check (ADDENDUM 4; `logs/cond.txt`) [M]

Mean within-feed AUC of one input (positives against negatives of the same feed). Train labels for the views; val labels only for the stored val columns (lag 2-4 d, depth 12 d).

| input | rows | as built | hz | hz2 | hz2L | val, stored (8,530 feeds, M >= 50) |
|---|---|---|---|---|---|---|
| `it_hist_all` | all, trPool >= Sep 13 (42,796 feeds) | 0.5613 ± 0.0012 | 0.5567 | 0.5416 | 0.5394 | 0.5522 ± 0.0027 |
| `it_hist_all` | listings > 24 d (29,478 feeds) | 0.5271 ± 0.0016 | 0.5271 | 0.5231 | 0.5215 | 0.5308 ± 0.0036 |
| `seq_cv_invd20` | all | 0.5511 ± 0.0009 | - | 0.5433 | 0.5349 | 0.5525 ± 0.0020 |
| `seq_cv_invd20` | listings > 24 d | 0.5577 ± 0.0012 | - | 0.5572 | 0.5455 | 0.5632 ± 0.0028 |
| `xp_imp_c50` | all | 0.5230 ± 0.0012 | - | 0.5243 | 0.5215 | 0.5308 ± 0.0028 |

- The views lower the predictive power of `it_hist_all` below val's (0.542 against 0.552), as a row with a longer lag than val should; hz moves a quarter of the way.
- Old listings: the lag costs little information in `it_hist_all` (0.5231 primary, 0.5215 lag-first, 0.5308 val). In `seq_cv_invd20` the lag-first view loses much more (0.5455 against 0.5572) because it has 5 dense days instead of 10: that loss is depth, which a test row does not suffer. This speaks for the declared rule's choice (m = 12) and against my objection in section 3.2, for training purposes.

### 4.6 The stale-val inputs in the same statistic (`staleval_parity.py`, `logs/staleval_parity.txt`; label-free, information) [M]

The stale-val witness re-predicts val with the counterfactual columns `nn_x/r6/stale_cf/both_<col>.npy`. The same G-statistic, val rows of feeds with M >= 50 (8,530 feeds, 1,252,147 rows) against private rows:

| column | stored val: dZ / bucket | stale-val inputs: dZ / bucket | hz2 on trPool, days >= Sep 13: dZ / bucket |
|---|---|---|---|
| `it_hist_all_luo` | 0.079 / 0.243 | 0.028 / 0.062 | 0.022 / 0.058 |
| `seq_cv_pop` | 0.062 / 0.214 | 0.033 / 0.105 | 0.014 / 0.055 |
| `it_hist_et1_luo` | 0.076 / 0.225 | 0.029 / 0.056 | 0.021 (largest secondary dZ) |
| `seq_cv_invd20` | 0.016 / 0.044 | 0.013 / 0.043 | within the 0.021 above |

- Zero share of `it_hist_all_luo` by listing age (< 1 d, 1-3 d, 3-7 d, 7-10 d, 10-24 d, 24-76 d, > 76 d): stale-val 0.981 / 0.948 / 0.921 / 0.835 / 0.389 / 0.217 / 0.231; hz2 (section 4.2) 0.989 / 0.949 / 0.906 / 0.788 / 0.337 / 0.178 / 0.191; private rows 0.975 / 0.916 / 0.859 / 0.786 / 0.395 / 0.192 / 0.200.
- So the stale-val inputs and the hz2 train rows aim at the same target and land within 0.05 of each other in every age bucket (0.052 for 10-24 d); both are a little colder than private rows for listings of 1-7 d. A model cooled on hz2 is therefore evaluated by stale-val on inputs of the kind it was trained on, which is what that witness is for. Against PUBLIC rows the stale-val inputs are too cold (`it_hist_all_luo` dZ 0.055 against 0.051 for the stored val columns): stale-val is a private-like witness only.
- `xp_imp_c50` has no stale-val column (the popularity-light members are not covered by this comparison).

### 4.7 hz2 against poplag's v3 view on the same holdout rows (`vs_poplag.py`, `logs/vs_poplag.txt`; label-free, information) [M]

The private-like feeds of hz2 carry exactly the lag poplag's private-like view gave them (checked: identical on 20,940 holdout feeds). Private-like holdout rows, `it_hist_all_luo` (zero share stored / v3 / hz2; share of rows with the same zero pattern as v3; correlation with v3 on the model scale):

| holdout days | rows | zero share stored / v3 / hz2 | same zero pattern | correlation hz2 ~ v3 | hz2L ~ v3 |
|---|---|---|---|---|---|
| Sep 6-8 | 848,277 | 0.178 / 0.475 / 0.710 | 0.709 | 0.610 | 0.610 |
| Sep 9-12 | 913,579 | 0.204 / 0.467 / 0.499 | 0.917 | 0.943 | 0.943 |
| Sep 13-14 | 554,832 | 0.224 / 0.458 / 0.471 | 0.959 | 0.976 | 0.971 |
| Sep 17-19 | 905,447 | 0.275 / 0.475 / 0.474 | 0.987 | 0.993 | 0.976 |

- On Sep 17-19, where the reviewer found v3 causal (0% of the counted pairs after the row), the causal hz2 view is the same view to a correlation of 0.993 (`seq_cv_pop` 0.994, `xp_imp_c50` 0.977). On earlier days the two diverge exactly where v3 borrowed days after the row: v3 holds the zero share at 0.47 on Sep 6-8 by counting the future; a causal view cannot (0.71).
- The primary hz2 view is closer to v3 than the lag-first one on Sep 17-19 (0.993 against 0.976). v3 gave old listings every dense day up to its exclusion band of 3 days before the row [D, `reports/NEXT_1115/poplag.md` section 2], i.e. a nearest dense day 3-5 d before the row on Sep 17-19; the primary hz2 view has it at 1-2 d, the lag-first view at 6-9 d, a test row at 6-9 d. The staleness cost poplag measured (-0.0033 ± 0.0008 on days >= Sep 13) is therefore mostly a cost of test-like COVERAGE of young listings [I]; the displacement of the window of old listings is represented in part by v3, not by the primary hz2 view, and fully (at reduced depth) only by hz2L on Sep 17-19.
- "within 0.05" in section 4.6 and the correlations here compare different populations (val feeds, trPool feeds, holdout feeds); they are orientation, not a gate.

## 5. Labelled reads through the causal view (train-holdout labels only; rules in `DECLARATION.txt` ADDENDA 2, 3, 5, 6, 8, 9)

Each read was made once, after all of its predictions existed; the read script was tested beforehand on random labels only. No view, gate or rule was changed after a read. Section 5.1 and read 1 of section 5.2 were declared before any labelled number; read 2 of section 5.2 was added after read 1 had been seen and is marked post hoc wherever it appears. No val or test label is involved.

### 5.1 The staleness cost of the W8 pool and of the popularity-light pool, days >= Sep 13 (E1-E4) [M, `logs/hold_read_late.txt`]

Method: `hold_infer.py` = poplag's CPU fp32 path (control VALID in `poplag.md` 1.3) with the eight (W8) or one (`xp_imp_c50`, popularity-light) view columns at the holdout rows; "stored" = poplag's CPU lag-0 replay of the same path. 13,424 holdout feeds of days >= Sep 13 (the subset where the gate holds; 2,089,648 rows); every feed carries its hz2 lag (4,045 public-like, 9,379 private-like). W8 = 8 wide students, pop1kd = 4 popularity-light students; pools = member mean of per-feed z; S(0.2) = NDCG(0.8 z(W8) + 0.2 z(pop1kd)) - NDCG(W8) = the W8P20 form. `hold_read_late.py`.

| quantity, 13,424 feeds | hz2 primary (`feats`) | hz2L lag-first (`feats_L`) | poplag v3 on the same feeds (not causal on Sep 13-14) |
|---|---|---|---|
| dW = W8 pool, view - stored | **-0.00446 ± 0.00074** (z -6.0) | -0.00523 ± 0.00079 (z -6.6) | -0.00332 ± 0.00075 |
| dP = pop1kd pool, view - stored | -0.00118 ± 0.00035 (z -3.4) | -0.00109 ± 0.00042 (z -2.6) | -0.00049 ± 0.00038 |
| dP - dW | +0.00328 ± 0.00080 (z +4.1) | +0.00415 ± 0.00088 (z +4.7) | |
| pop1kd pool - W8 pool: stored -> view | -0.00561 ± 0.00084 -> -0.00234 ± 0.00082 | -> -0.00147 ± 0.00083 | |
| S(0.2): stored -> view | -0.00050 ± 0.00032 -> **+0.00032 ± 0.00031** | -> +0.00058 ± 0.00032 | -> +0.00024 ± 0.00031 |
| D = S_view - S_stored | **+0.00082 ± 0.00039** (z +2.1) | +0.00108 ± 0.00042 (z +2.6) | +0.00074 ± 0.00040 |
| S(0.1) / S(0.3) under the view | +0.00020 ± 0.00020 / +0.00004 ± 0.00039 | +0.00026 ± 0.00020 / +0.00084 ± 0.00039 | |

| subset | n | dW primary | dW lag-first | dP primary | S(0.2) under the primary view |
|---|---|---|---|---|---|
| Sep 13-14 | 5,145 | -0.00768 ± 0.00135 | -0.00914 ± 0.00140 | -0.00199 ± 0.00062 | +0.00003 ± 0.00053 |
| Sep 17-19 | 8,279 | **-0.00245 ± 0.00085** | -0.00280 ± 0.00094 | -0.00068 ± 0.00041 | +0.00050 ± 0.00038 |
| private-like | 9,379 | -0.00480 ± 0.00087 | -0.00532 ± 0.00094 | -0.00145 ± 0.00043 | +0.00028 ± 0.00036 |
| public-like | 4,045 | -0.00367 ± 0.00141 | -0.00503 ± 0.00147 | -0.00056 ± 0.00059 | +0.00043 ± 0.00058 |
| lag 6 / 7 / 8 / 9 | 2,846 / 3,488 / 3,558 / 3,532 | -0.00160 ± 0.00168 / -0.00598 ± 0.00145 / -0.00606 ± 0.00137 / -0.00364 ± 0.00145 | -0.00300 / -0.00689 / -0.00697 / -0.00364 | -0.00112 / -0.00086 / -0.00130 / -0.00142 | -0.00001 / +0.00106 / +0.00038 / -0.00019 |

Every W8 member loses under the primary view (-0.0037 to -0.0050, z -4.9 to -6.8); the four popularity-light members lose -0.0005 to -0.0013.

Against the expectations written before the read (ADDENDUM 2 / 5):
- **E1 (dW <= -0.0015 under both views): holds**, at three times the bar. The staleness cost of the W8 pool is reproduced by a strictly causal view and is larger than poplag's: the causal view minus v3 on the private-like feeds (same lags) is -0.00133 ± 0.00048 (z -2.7).
- **E2 (|dP| <= 0.0010): NOT met**, by 0.0001-0.0002. The popularity-light pool also loses (-0.0012 ± 0.0004): its one counter `xp_imp_c50` is 100% zero for listings younger than 7 d under the view (section 4.2). Its loss is a quarter of the W8 pool's.
- **E3 (D > 0): holds** under both views (+0.00082 ± 0.00039, +0.00108 ± 0.00042). Under test-like inputs the popularity-light admixture reads +0.0003 to +0.0006, the sign and size of the public reading of W8P20 - W8 (+0.000523), where the in-period holdout reads -0.00050 on the same feeds.
- **E4 (hz2L <= primary on Sep 17-19): sign as expected, not resolved**: -0.00035 ± 0.00057 (z -0.6). On the days where the lag-first view really lags old listings by 6-9 d, the W8 pool loses no measurable extra. The cost is carried by the coverage of young listings, which both views share. (On Sep 13-14 the lag-first view costs -0.00146 ± 0.00068 more: there it differs by depth, 5 dense days against 7.5.)
- Counter-criteria ("dW >= -0.0005 under both views" or "dP <= dW"): neither is met. **Verdict of the read: the staleness cost is reproduced by a causal view.**

Two cautions on the size [I]:
- "Stored" is not a causal baseline. The as-built `it_hist_*` columns of a train row count dense days AFTER the row (section 1); a Sep 13 row loses four future dense days (Sep 14, 17, 18, 19) plus its own day when the view is imposed, a Sep 19 row only its own day. The Sep 13-14 loss (-0.0077) therefore mixes staleness with the removal of future information that no test row has. **Sep 17-19 is the cleaner estimate of what the W8 pool pays on test-like inputs: -0.0025 ± 0.0009** (v3 on the same feeds -0.00195 ± 0.00086, the reviewer's number).
- The day-group difference (-0.0077 against -0.0025) is 3.3 SE of its own difference; I have not tested another explanation (for example the 7.5 against 9.7 dense days of old listings).

### 5.2 HZ11: do models TRAINED on a stale-like view do better under the causal view? (ADDENDUM 6)

Why: the hz2 arm only pays if part of the staleness cost is a train / test MISMATCH that training on stale-like inputs removes (falsifier F3 of section 7), not pure information loss. Four models trained with a stale-like view already exist. No GPU and no training are needed to ask which side does better when the inputs are test-like.

Method: H = the four half-dose hz models of 11-12 (`w5_cd_hzcd_r101..104`, d128, hard labels, 2-epoch re-cool with `--hzcd`), R = their references (`w5_cd_m02ref_r101..104`, the same re-cool without the flag; paired by root). Same 13,424 holdout feeds, primary view, CPU both sides; a CPU "stored" replay of the same path is the in-period arm and the path control. Pools = mean of per-feed z of 4 models. Q_view = NDCG(H | view) - NDCG(R | view); Q_stored = the same under stored inputs; I = Q_view - Q_stored. `hold_infer2.py`, `hz11_read.py`, `logs/hz11_read_feats.txt`.

Path control [M]: CPU replay against the stored GPU bf16 holdout files, 8 models, same feeds: NDCG differences -0.00011 to +0.00023 (each within 1.7 SE of zero; SE 0.00014), correlation of z 0.99993-0.99994. VALID by the declared bar. My pooling code reproduces the project's 11-12 read-out of these models from the stored files to the last digit (member level @last -0.00052 ± 0.00019, 4-pool @last -0.00067 ± 0.00027 on 30,000 feeds; `logs/hz11_consistency.txt` against `nn_x/w5/logs/w5_eval_B_hzcd.txt`).

**Read 1, as declared (SELECTED checkpoints), 13,424 feeds [M]:**

| quantity | 4-pool | member level (mean of the 4 paired differences) |
|---|---|---|
| Q_view: hz models - references, both through the causal view | **+0.00067 ± 0.00041** (z +1.6) | +0.00064 ± 0.00031 (z +2.1) |
| Q_stored: the same under stored (fresh) inputs | +0.00044 ± 0.00040 | -0.00021 ± 0.00029 |
| I = Q_view - Q_stored | +0.00023 ± 0.00048 (z +0.5) | +0.00085 ± 0.00034 (z +2.5) |
| staleness cost of the reference pool / of the hz pool (view - stored) | -0.00376 ± 0.00073 / -0.00352 ± 0.00069 | |

- By subset (4-pool Q_view / Q_stored): Sep 13-14 +0.00065 ± 0.00068 / -0.00048 ± 0.00065; Sep 17-19 +0.00069 ± 0.00052 / +0.00101 ± 0.00050; private-like +0.00048 ± 0.00049 / +0.00035 ± 0.00047; public-like +0.00112 ± 0.00078 / +0.00066 ± 0.00074. By root (single models, Q_view): +0.00094, +0.00003, +0.00154, +0.00006: all four non-negative.
- **Reading by the rule as written (pools): "not separated at 4 models / 13,424 feeds".** Q_view >= +1 SE holds; I >= +2 SE does not. It is not "against". My written expectation "Q_stored about -0.0005" was wrong at pool level on these feeds (+0.00044).
- Why the pool-level interaction is uninformative here [M, `logs/hz11_consistency.txt`]: the selected checkpoint of every run is chosen on these same 30,000 holdout feeds with fresh inputs. On days >= Sep 13 the stored GPU files read the hz arm at +0.00061 ± 0.00039 with the selected checkpoints and at -0.00042 ± 0.00038 with the @last checkpoints (the ones the 11-12 rule judged). Q_stored of read 1 carries that selection, and I inherits it.

**Read 2, POST HOC (the @last checkpoints; declared in ADDENDUM 8 after read 1 was seen and before any @last prediction through a view existed; same quantities, same rule), 13,424 feeds [M, `logs/hz11_read_feats_last.txt`]:** path control VALID (differences -0.00007 to +0.00016, correlation 0.99993-0.99994).

| quantity | 4-pool | member level |
|---|---|---|
| Q_view: hz models - references through the causal view | **+0.00078 ± 0.00038** (z +2.1) | +0.00072 ± 0.00028 (z +2.6) |
| Q_stored: the same under stored (fresh) inputs | **-0.00039 ± 0.00036** | -0.00016 ± 0.00026 |
| I = Q_view - Q_stored | **+0.00117 ± 0.00047** (z +2.5) | +0.00088 ± 0.00032 (z +2.8) |
| staleness cost of the reference pool / of the hz pool | -0.00413 ± 0.00075 / -0.00296 ± 0.00070 | |

- By subset (4-pool Q_view / Q_stored): Sep 13-14 +0.00048 ± 0.00060 / -0.00126 ± 0.00059; Sep 17-19 +0.00097 ± 0.00048 / +0.00015 ± 0.00046; private-like +0.00058 ± 0.00044 / -0.00066 ± 0.00043; public-like +0.00124 ± 0.00071 / +0.00024 ± 0.00067. By root: +0.00092, +0.00027, +0.00124, +0.00046.
- **Reading by the same rule: "SUPPORTS M", post hoc.** The same four models that the in-period holdout reads at -0.0004 (and at -0.00067 ± 0.00027 on all 30,000 feeds, the number that closed the arm on 11-12) read +0.0008 once the inputs are test-like. The half-dose cooldown removes 28% of the reference pool's staleness cost (0.00117 of 0.00413).

All four cells, so that none is picked (same 8 models, same feeds, two correlated views: NOT four independent results). The lag-first cells carry no reading rule of their own (ADDENDA 6, 9: information); the line "READING" that the script prints in their logs is mechanical.

| checkpoints | view | Q_view, 4-pool | I, 4-pool | Q_view, member level | Q_view by root (101-104) | status |
|---|---|---|---|---|---|---|
| selected | primary | +0.00067 ± 0.00041 | +0.00023 ± 0.00048 | +0.00064 ± 0.00031 | +0.00094 / +0.00003 / +0.00154 / +0.00006 | declared read: "not separated" |
| selected | lag-first | +0.00124 ± 0.00043 | +0.00080 ± 0.00051 | +0.00071 ± 0.00030 | +0.00079 / +0.00023 / +0.00080 / +0.00103 | information |
| @last | primary | +0.00078 ± 0.00038 | +0.00117 ± 0.00047 | +0.00072 ± 0.00028 | +0.00092 / +0.00027 / +0.00124 / +0.00046 | post hoc: "supports" |
| @last | lag-first | +0.00116 ± 0.00040 | +0.00155 ± 0.00050 | +0.00075 ± 0.00028 | +0.00115 / +0.00040 / +0.00119 / +0.00027 | information |

- Q_view is positive in all four cells (z 1.6 to 2.9) and in all 16 single-model pairs. It is the quantity that needs no fresh-input baseline and no assumption about selection.
- What this is evidence for: a cooldown on a HALF-dose stale-like view makes a model better on full-dose test-like inputs than its twin that never saw one, by +0.0007 to +0.0012 on 13,424 feeds. What it is not: a test of the hz2 arm (different models: d128, hard labels, no KD), and not a public-scale number.
- By-day structure of the in-period reading that closed hz on 11-12 (@last, 4-pool, stored GPU files): Sep 6-8 -0.00070 ± 0.00060; Sep 9-12 -0.00102 ± 0.00046; Sep 13-14 -0.00114 ± 0.00061; **Sep 17-19 +0.00003 ± 0.00048**. The deficit sits on the days whose as-built `it_hist_*` columns count dense days after the row; on the three days where they cannot, it is zero. Sep 17-19 minus Sep 6-14 is +0.00096 ± 0.00058 at the 4-pool (unpaired, z 1.7) and +0.00052 ± 0.00040 at member level (z 1.3): suggestive, not established.
- The same question for the popularity-light admixture under stored inputs (`logs/holdout_by_day_stored.txt`, 30,000 feeds, CPU replay) [M]: S(0.2) = -0.00059 ± 0.00045 (Sep 6-8), -0.00032 ± 0.00046 (Sep 9-12), -0.00103 ± 0.00057 (Sep 13-14), -0.00017 ± 0.00039 (Sep 17-19); Sep 17-19 minus the rest +0.00042 ± 0.00048: **not resolved**. What is resolved is the gap between the two pools, pop1kd - W8: -0.0108 ± 0.0012 on Sep 6-8, -0.0071 ± 0.0011 on Sep 9-12, -0.0076 ± 0.0015 on Sep 13-14, -0.0044 ± 0.0010 on Sep 17-19. The W8 pool's in-period advantage over a popularity-light pool more than halves from the first days to the last, and falls to -0.0023 ± 0.0008 under the causal view (section 5.1).

## 6. Drop-in for the trainer

- `nn_x/w12/stale/hz2/cd_arms6b.py` = `nn_x/w5/cd_arms6.py` (md5 2801de7b…, byte-identical to `nn_x/abl/cd_arms6.py`) with three changes, each marked `# 6b` (`cd_arms6b_vs_cd_arms6.diff`, 84 lines):
  1. the view directory comes from the environment variable `HZCD_DIR` (default `data/feats_new` = the existing half-dose files);
  2. the column map also covers `seq_cv_invd20`, `seq_cv_pop`, `seq_cv_norm`, `xp_imp_c50`; a column is swapped only if its file exists in `<HZCD_DIR>/train/` (`seq_cv_hz_invd20.npy`, `seq_cv_hz_pop.npy`, `seq_cv_hz_norm.npy`, `xp_imp_hz_c50.npy`; none exists in `data/feats_new`);
  3. optional `HZ2_KEEP=<npy, bool per train candidate row>`: flagged rows keep their stored xt values.
- Same scaling as the stored columns: the files are raw-scale float32 like `it_hist_hz_*.npy`; `load_hz` applies the `attach_xt` branch of the name (log1p for `it_hist*` / `seq_cv*`, clip ±10 for `xp_imp*`) and stores fp16. The in-model rank / z of `seq_cv_invd20` and `it_hist_all_luo` follow the swap; `f_mean` / `f_std` stay the as-built statistics, as on test rows.
- Proof (`test_cd_arms6b.py`, `logs/test_cd_arms6b.log`, CPU MiniSplit of 200 random train feeds, 15,471 rows, preset `cvnd_pre2s`) [M]:

| test | result |
|---|---|
| T1 text | `HzTrainView`, `xt_transform`, `load_hz`, `hz_path`, `hz_stats`, `save_holdout_hz` AST-identical; changed: `luo_cols`, `setup_hzcd`; new: `apply_keep` |
| T2 `HZCD_DIR` unset | same 5 columns, bit-identical hz tensor, **8 batches x 22 keys bit-identical** between `cd_arms6` and `cd_arms6b`; `hz_stats` identical |
| T3 `HZCD_DIR` = `feats` | 8 columns swapped (xt 0-7); every swapped cell = fp16(attach_xt transform of the hz2 file); every other key / column = the plain gather |
| T3b popularity-light xt list (27 columns, `xp_imp_c50` last) | `cd_arms6` refuses the list (no LUO column); `cd_arms6b` swaps exactly `xp_imp_c50` |
| T4 `HZ2_KEEP` | 3,339 of 15,471 sample rows flagged: stored values in all 8 columns; the other rows hz2 |
| T5 (resumed run, `test_cd_arms6b_t5.py`, a NEW sample of 200 feeds, 13,701 rows): the body of `setup_hzcd` end to end with `HZCD_DIR` = hz2 view and `HZ2_KEEP` set | 8 columns swapped; 3,303 keep rows read the stored xt, the others fp16(attach_xt transform of the hz2 file); every other key / column = plain gather; the log line carries `[6b: 8 swapped columns ...; HZ2_KEEP rows on stored values: 3303]` |
| T5b: `setup_hzcd` of both modules with both variables unset (the call `train.py` makes for arm hz) | same 5 columns, bit-identical hz tensors, 8 bit-identical batches on the new sample; the log lines differ only by the inserted `[6b: ...]` tag |

- Change to `train.py` (NOT applied; `train_py_hz2.diff`; the same line exists once in `nn_x/abl/train.py`). Copy `cd_arms6b.py` next to `train.py` first (a new file, nothing overwritten):

```diff
--- nn_x/w5/train.py
+++ nn_x/w5/train.py (hz2)
@@ -54,5 +54,5 @@
 from cd_arms4 import setup_wave4
 from cd_arms5 import save_holdout, save_last, save_last_test, setup_wave5
-from cd_arms6 import save_holdout_hz, setup_hzcd  # wave-5 hzcd (INSIGHTS_1112 item 6; OFF by default)
+from cd_arms6b import save_holdout_hz, setup_hzcd  # wave-5 hzcd; 6b = cd_arms6 + env HZCD_DIR / HZ2_KEEP (hz2 view); identical when both are unset
 from cd_arms7 import setup_valcd  # wave-5 valcd (val-in-cooldown, D1; OFF by default)
 from cd_arms8 import setup_rdrop, setup_sam  # wave-7 R-Drop / SAM cooldown regularisers (OFF by default)
```

- One line. With both variables unset every queued `--hzcd` lane (arm hz) runs exactly as before (T2). An hz2 lane is the hz lane plus `HZCD_DIR=<repo>/nn_x/w12/stale/hz2/feats HZ2_KEEP=<repo>/nn_x/w12/stale/hz2/feats/train/hz2_keep_rows.npy` in its environment. `tests/strip_hzcd.py` expects the old import line and needs the same one-word change.
- Lane side (NOT applied; `lane_sh_hz2.diff`, made against the `nn_x/w5/lane.sh` of 03:35 UTC, 6 added lines; the patched copy `proposal/lane.sh.hz2` passes `bash -n` and prints the expected commands under `DRY=1` for `K:hz2:951 K:hz:951 L:hz2L:952`): arms `K:hz2 | L:hz2 | K:hz2L | L:hz2L` = `--hzcd`, and before each launch `unset HZCD_DIR HZ2_KEEP` followed by an `export` for the two hz2 arms only. **The variables must be set per job.** `lane.sh` runs a job list in one shell; a plain `export` at the top of a lane would make a later `K:hz` job of the same lane read the hz2 files without any error. Variants are then named `w5_kcd_hz2_s95x` / `w5_lcd_hz2_s95x`.
- What a run really read is in its log: the `wave-5 hzcd:` line carries the view directory and `[6b: N swapped columns [...]; HZ2_KEEP rows on stored values: K]`. 5 columns = hz, 8 columns = hz2 (1 column for a popularity-light member, whose only popularity input is `xp_imp_c50`), K = 46,918,534 (19.0% of rows; the same mask file in `feats/` and `feats_L/`) with the keep mask.
- The harness selects the checkpoint on the in-period holdout (fresh popularity; `--select_on holdout`). For a staleness arm that selection works against the arm; `@last` is the reading free of it. The module is unchanged in this respect.
- **Applied by the lead at 04:57:57 UTC, while this report was being finished** [D, read-only check]: `nn_x/w5/cd_arms6b.py` is byte-identical to my file (md5 4ff0b0bf...), `nn_x/w5/train.py` l.56 imports it, `nn_x/w5/lane.sh` carries the per-job `unset` / `export` block and the arms `K:hz2 | L:hz2 | K:hz2L | L:hz2L` (plus the lead's own `h2t`, `h2x`). First run `w5_kcd_hz2_s958` started 04:58:25 UTC. Its log line is the full-size check of the drop-in that I could not make on CPU: `[6b: 8 swapped columns ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', ... 'it_hist_et4_luo']; HZ2_KEEP rows on stored values: 46918534]` (my count of the mask: 46,918,534), zero rate `it_hist_all_luo` 0.130 -> 0.408, `seq_cv_pop` 0.423 -> 0.515, rows changed 68.1% / 48.4%, hz tensor 3.68 GiB, load 23 s (`nn_x/w5/logs/w5_kcd_hz2_s958.log`). All 16 lanes of arm hz had started before the patch (03:40-04:54 UTC, `nn_x/w5/logs/RUNNING_w5.txt`), so arm hz ran on the unmodified `cd_arms6` throughout.
- **Consequence: `feats/train/` and `feats_L/train/` are now read by queued lanes. They must not be moved, rebuilt or deleted until the hz2 arms are finished.** Only `cache/` may be cleaned (section 11). Guard added at 05:45 UTC: while the file `LIVE_DO_NOT_REBUILD` exists, `chain.sh` and `views.py masks | build_it | finish_seq` refuse to run (tested: both refuse, the view files keep their 02:54 / 03:01 UTC timestamps); `MD5SUMS_feats.txt` holds the checksums of the 26 view files as the lanes read them.
- Side effects to know: the view tensor is 3.95 GB = 3.68 GiB on the GPU (8 x 246.7M x fp16) against 2.47 GB for hz. `preds/holdout/nn_<v>_holdout_hz.npy` becomes a causal lagged-holdout read of the selected checkpoint (every holdout feed has a lag in the view; with `HZ2_KEEP` its Sep 6-8 truncated rows are as stored).
- Per-feed lag for a later model input: `feats/train/hz2_lag.npy` (int8, 6-9), `hz2_part_pub.npy` (bool), `hz2_cut_slot.npy` (int16, lagged cutoff as day slot; Sep 6 = 0). For val / test rows the same quantity is day - Sep 20: val 2 / 4, test 6-9.

## 7. Pre-registered expectation for the hz2 arm (`PREREG_hz2_arm.txt`, written 03:14Z before any model saw an hz2 view and before the hz arm was read)

Proposed arm: the K: / L: re-cool with `--hzcd` and `HZCD_DIR=feats`, `HZ2_KEEP` on ("hz2k"); reference = the ref re-cools; comparison = hz, paired by snapshot. Central value and 80% interval, 16-pool, selected checkpoints:

| reading | expected | why |
|---|---|---|
| stale-val, hz2k - reference | +0.0018 [+0.0008, +0.0032] | hz (11-12, hard labels) gave +0.00107 ± 0.00036 with a third of the `it_hist` dose and no `seq_cv` dose |
| stale-val, hz2k - hz | +0.0007 [-0.0002, +0.0018] | 2.3-2.8 x the dose, concave response |
| val | 0.0000 [-0.0006, +0.0006] | val rows have lag 2 / 4 d; no feed of the view has a lag below 6 |
| in-period holdout (as built) | -0.0010 [-0.0020, -0.0003] | about twice the hz reading (-0.00052); reported, not a veto |
| causal lagged holdout, days >= Sep 13: hz2k model - reference model, both through the hz2 view | +0.0015 [+0.0005, +0.0030] | a model cooled on these inputs against one that never saw them |

Falsifiers of the mechanism "part of the staleness cost is a train / test mismatch that training removes" (any one): F1 stale-val of hz2k not above +1 SE although its dose is more than twice that of hz; F2 stale-val(hz2k) - stale-val(hz) < -1 SE of the paired difference; F3 the lagged-holdout reading <= 0 (then the cost is information loss, not mismatch); F4 in-period holdout >= 0 and stale-val <= 0 (no-op). Not a falsifier: val in [-0.0006, 0] with stale-val >= +2 SE; if "val >= 0" stays a hard condition, hz2 is closer to failing it than hz was.

Addendum of the resumed run (03:40:09Z: 36 s before the first hz re-cool started, 60 min before the first one finished, 78 min before the first hz2 run started, and before the HZ11 read of section 5.2), the move relative to the half-dose arm on the lagged holdout (days >= Sep 13, both models through the hz2 view, CPU both sides): hz model - reference model +0.0006 [0.0000, +0.0015]; hz2k model - hz model +0.0009 [-0.0002, +0.0020]; falsifier F5 = hz2k model - hz model < -1 SE of the paired difference. Tool: `hold_infer2.py feats <variants> late`, then `lagged_pair_read.py feats <arm A> <arm B> <tag>` (it reproduces the HZ11 read to the last digit and applies no rule). A tool note was appended to the file at 04:58Z; it changes no number.

Two things the lead should have in mind when reading the arm:
- Checkpoint selection. The harness selects on the in-period holdout with fresh inputs. On the 11-12 hz models that selection moved the in-period reading of days >= Sep 13 by 0.0010 (selected +0.00061 ± 0.00039, @last -0.00042 ± 0.00038; section 5.2). Read `@last` next to the selected checkpoints on every witness.
- What already supports the central values, from models that are NOT the arm (11-12, d128, hard labels, half dose; section 5.2): hz - reference through the causal view +0.0007 ± 0.0004 on the selected checkpoints. The expectation line "hz model - reference model +0.0006" was written 71 minutes before that number was read.

## 8. Limits

- **Two thirds of the rows are an approximation.** Exact replay exists for 34.2% of rows (listings created on or after Sep 6 or after the cutoff). For the other 65.8% the test-like value is not observable; the view matches coverage by one calibrated parameter and a depth factor, like poplag's band g, and its parity on those rows is partly by construction.
- **Old listings are not stale in the primary view** (nearest dense day 1-2 d before the row). Their zero share matches private rows on days >= Sep 13; their level is 0.2-0.3 log1p units above. hz2L restores the lag on Sep 17-19 only (6-8 d), with 5 dense days instead of 12.
- **Sep 6-8 (27% of train feeds) cannot be made test-like causally.** The view is far colder than test there (dZ 0.23), the stored columns are far warmer in `it_hist` and equally shallow in `seq_cv` / `xp_imp`. `HZ2_KEEP` leaves the truncated rows of those feeds as stored; it is a choice between two wrong formats, made on the feed-level cold share.
- Strictly causal sparse days make young listings 0.014-0.047 colder than private rows; a test row also sees later sparse days.
- The depth factor makes values non-integer and caps at 4; on Sep 7-8 it under-corrects. A zero stays a zero, so shallow windows have too many zeros (hz2L: +0.05-0.06 for old listings).
- The gate reference is test feeds with M >= 50; the view rows are exploration-like train feeds of the same M range. Production feeds (trAll) have a more popular item mix and are reported for information only.
- Marginal and pairwise-joint parity by listing age is shown; parity of the full joint with the other 26 inputs is not. The conditional check of 4.5 is one input at a time.
- **A third stale input family is outside the view: the trajectory-encoder scores (`pre2_fe_z`, `pre2_fe_s`; 2 of the 34 inputs).** The encoder's item-ID table has the same coverage problem: share of rows whose item-ID contribution is exactly 0, by listing age < 1 d / 1-3 d / 3-7 d / 7-10 d / 10-24 d / 24-76 d / > 76 d: trPool train rows of days >= Sep 13 0.890 / 0.686 / 0.528 / 0.417 / 0.354 / 0.294 / 0.299 (all rows 0.431); val 0.997 / 0.989 / 0.843 / 0.664 / 0.477 / 0.346 / 0.342 (0.547); private rows 0.996 / 0.984 / 0.968 / 0.934 / 0.620 / 0.368 / 0.357 (0.597) [M, `logs/pre2_gap.txt`, label-free, ADDENDUM 7]. The windowed encoders give train rows a staleness of 0-5 d against 6-9 d on test. The stale-val scenario perturbs this family as well ("pre2both"), and there it moved the stale cost of a model by at most 0.00005 while changing 3.1% of val rows [D, `reports/sol_cpu_CPU2_stale_replay_pre2s_and_arms.md` section 4]; on train rows the ID-less share would have to rise by about 17 points (0.431 to 0.597) to be private-like [I], so the size of this channel under training is not known. hz2 leaves these columns as built; on Sep 6-8 train feeds they do not exist at all (`has:pre2_fe` = 0 on every row), which is one more way in which those days are not test-like.
- One draw of the lag hash and of the public / private part.
- The listing age comes from f4 knots (128-second grid); its accuracy on train against test rows is not tested. Exact rows do not depend on it; truncated rows do (through X_test).

## 9. What I did NOT check

- I trained no model on the view and used no GPU. The lead's hz2 runs started at 04:58 UTC; none had finished when this report was closed, so every statement about what training on the view does rests on the half-dose models of 11-12 (section 5.2) and on the pre-registration (section 7).
- The K: / L: hz re-cools (KD, d192; 9 of 16 finished by 04:50 UTC) were NOT scored through the view: that is the lead's arm with its own rule, and the reader is ready for it (section 10).
- No val or test label was used for anything except the stored-val within-feed AUC of section 4.5 (descriptive, declared in ADDENDUM 4).
- The view for val / test rows was not built (the stored columns are the view there: V1).
- A transductive variant (sparse days after the row counted as on a test row) was measured (section 2) but not written.
- A variant with val-like lags (2 / 4 d) mixed in was not built.
- No sensitivity of the gate to the lag hash, to the reference (all-M test feeds) or to the bucket edges.
- `tests/test_hzcd.py` of nn_x/w5 was not re-run against the patched import (I do not edit train.py); the GPU end-to-end part of that test needs a GPU. I never ran the drop-in on a GPU; its full-size load is confirmed only by the log line of the lead's first run (section 6), not by a test of mine, and nothing of mine checks the training batches of that run.
- A train-row view of the encoder scores `pre2_fe_z` / `pre2_fe_s` (section 8) was measured as a gap, not built.
- My `lane.sh` proposal was dry-run on a copy only (`DRY=1`, prints commands; nothing launched by me). The lead's applied version differs from it (extra arms `h2t`, `h2x`, a `DRY` echo of the directory); I read it, I did not test it.
- The stale-val inputs were compared with the view by marginal statistics only (section 4.6), on different feed populations.
- Holdout feeds of Sep 6-12 (16,576 of 30,000) were not scored through the view (ADDENDUM 3: CPU time; the gate fails on Sep 6-8 anyway). Every labelled number of section 5 is on the 13,424 feeds of days >= Sep 13.
- The four HZ11 cells of section 5.2 share their models and feeds; there is no multiplicity correction, the declared cell reads "not separated", and the cell that reads "supports" was added after the first had been seen.
- Section 5 uses 12 pool members (8 W8 + 4 popularity-light d128) and 8 old d128 models. The other 8 members of W16 and the d192 popularity-light students were not scored through the view.

## 10. Commands to reproduce

```
cd <repo>/nn_x/w12/stale/hz2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
PY=<repo>/.venv/bin/python
$PY sizes.py                          # calendar and size facts                          -> logs/sizes.log
$PY prep.py                           # postings + per-day count matrices (3 min)         -> cache/
$PY views.py v1 ; $PY views.py v2     # code checks V1 / V2                               -> logs/v1.log, logs/v2.log
$PY views.py calib ; $PY views.py calib_late ; $PY diag_m.py    # calibration of m       -> logs/calib*.log, logs/diag_m.log
# WARNING: the two chain.sh lines REWRITE feats/train and feats_L/train in place. Do NOT run them while hz2 lanes are queued or running (they read these files at start).
./chain.sh "" 12                      # PRIMARY view, all 246.7M rows x 9 columns (9 min) -> feats/train/
./chain.sh L 5                        # lag-first variant (7 min)                         -> feats_L/train/
$PY gate.py final ; $PY gate_summary.py final                   # gates                  -> logs/gate_final.txt, logs/gate_summary_final.md
$PY audit.py feats 12 ; $PY audit.py feats_L 5                  # causality audit V3     -> logs/audit_feats*.txt
$PY joint.py ; $PY cond.py            # joint parity, model-free conditional check        -> logs/joint.txt, logs/cond.txt
$PY test_cd_arms6b.py                 # drop-in proof T1-T4                               -> logs/test_cd_arms6b.log
$PY hold_infer.py feats all late ; $PY hold_infer.py feats_L all late   # 12 members on holdout feeds of days >= Sep 13 through the views (7-9 min per d192 member at load 130)
$PY hold_read_late.py feats feats_L   # labelled holdout read E1-E4 (train-holdout labels) -> logs/hold_read_late.txt
# resumed run
$PY causal_all.py                     # C1 every-row causality check + C3 file integrity   -> logs/causal_all.txt
HZ2_AUDIT_SEED=11 $PY audit2.py feats 12 40 12 ; HZ2_AUDIT_SEED=11 $PY audit2.py feats_L 5 40 12   # C2 -> logs/audit_feats*_seed11.txt
$PY staleval_parity.py ; $PY vs_poplag.py ; $PY pre2_gap.py     # sections 4.6, 4.7, 8      -> logs/staleval_parity.txt, vs_poplag.txt, pre2_gap.txt
MEM=w5_cd_m02ref_r101,w5_cd_hzcd_r101,w5_cd_m02ref_r102,w5_cd_hzcd_r102,w5_cd_m02ref_r103,w5_cd_hzcd_r103,w5_cd_m02ref_r104,w5_cd_hzcd_r104
$PY hold_infer2.py feats $MEM late ; $PY hold_infer2.py stored $MEM late   # HZ11 (chain_hz11.sh runs them when a slot is free)
$PY hz11_read.py feats                # HZ11 read (ADDENDUM 6)                              -> logs/hz11_read_feats.txt
./chain_hz11_last.sh <pid to wait for> ; HZ11_SUF=@last $PY hz11_read2.py feats   # the same at the @last checkpoints (ADDENDUM 8) -> logs/hz11_read_feats_last.txt
$PY test_cd_arms6b_t5.py              # drop-in T5 / T5b on a new sample                    -> logs/test_cd_arms6b_t5.log
# generic causal lagged-holdout read of any arm A against any arm B (members = checkpoint names, "<variant>@last" allowed); it applies no rule
$PY hold_infer2.py feats <A1,..,B1,..> late ; $PY hold_infer2.py stored <A1,..,B1,..> late ; $PY lagged_pair_read.py feats <A1,..> <B1,..> <tag>   # -> logs/lagged_pair_<tag>.txt
# for the lead, once hz2 runs exist (E-e, E-f, F3, F5 of the pre-registration); 5-8 min per d192 model at 12 threads, two processes at a time:
#   $PY hold_infer2.py feats w5_kcd_ref_s958,w5_kcd_hz_s958,w5_kcd_hz2_s958,... late
#   $PY lagged_pair_read.py feats w5_kcd_hz2_s958,... w5_kcd_ref_s958,... hz2_vs_ref ; $PY lagged_pair_read.py feats w5_kcd_hz2_s958,... w5_kcd_hz_s958,... hz2_vs_hz
# lead's own read-outs for the arm: python nn_x/w12/kd/arm_read8.py rc:hz2   (val / in-period holdout, 16-pool vs the ref re-cools); nn_x/w5/stale_pred.py <variants> (stale-val, GPU)
```

## 11. Files created (all under `nn_x/w12/stale/hz2/` unless stated)

- `DECLARATION.txt` (design, gates, calibration rule; ADDENDA 1-9, each stamped; 1-7 and 9 before the build or read they concern, 8 after read 1 of section 5.2 and before read 2), `PREREG_hz2_arm.txt` (+ addendum E-f / F5 of 03:40Z, tool note of 04:58Z), `PROGRESS.md`.
- **View, primary:** `feats/train/it_hist_hz_{all,et1,et2,et3,et4}.npy`, `seq_cv_hz_{invd20,pop,norm}.npy`, `xp_imp_hz_c50.npy` (float32, raw scale, 246,745,836 rows each), `hz2_lag.npy`, `hz2_part_pub.npy`, `hz2_cut_slot.npy` (per feed), `hz2_keep_rows.npy` (bool per row). 8.6 GB.
- **View, lag-first:** `feats_L/train/` with the same file names. 8.6 GB.
- **Drop-in:** `cd_arms6b.py`, `cd_arms6b_vs_cd_arms6.diff`, `train_py_hz2.diff`, `lane_sh_hz2.diff` (+ `proposal/` = the copies of `lane.sh` and of the two `train.py` that the diffs were made from and applied to), `test_cd_arms6b.py`, `test_cd_arms6b_t5.py`. The lead's applied copies live in `nn_x/w5/` (not mine).
- Code: `hz2_common.py`, `prep.py`, `kernels.py`, `views.py`, `chain.sh`, `gate.py`, `gate_summary.py`, `audit.py`, `joint.py`, `cond.py`, `diag_m.py`, `sizes.py`, `hold_infer.py`, `hold_read.py` (full-holdout form, not used), `staleval_parity.py`, `vs_poplag.py`; resumed run: `hold_read_late.py`, `causal_all.py`, `audit2.py`, `pre2_gap.py`, `hold_infer2.py`, `hz11_read.py`, `hz11_read2.py`, `chain_hz11.sh`, `chain_hz11_last.sh`, `lagged_pair_read.py` (generic reader), `test_cd_arms6b_t5.py`. `report_head_firstrun.md`, `report_tail.md`, `report_tail2.md` = the first run's report parts (superseded by this file). `logs/TEST_hold_read_late.txt` = a plumbing test of the read script on RANDOM labels (meaningless numbers, marked as such).
- Logs: `logs/` (one per command above; `gate_final.txt` holds every population x part x day-group table).
- Cache: `cache/` 42 GB (postings, day matrices, row masks, seq parts, 72 holdout prediction files `hpred_*` = 1.3 GB). `hpred_*`, `gate_final.json`, `calib*.json` are results; `row_dmask*.npy` / `feed_smask.npy` are what `causal_all.py` checks; the rest (36 GB) can be rebuilt by `prep.py` in 3 minutes and may be deleted. `feats/` and `feats_L/` (8.6 GB each) are NOT cache.
- `LIVE_DO_NOT_REBUILD` (guard, section 6), `MD5SUMS_feats.txt`, `logs/gate_per_input.md`.
- `reports/STALE_1115/hz2.md` (this file).
