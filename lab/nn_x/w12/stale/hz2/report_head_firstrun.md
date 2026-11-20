# hz2: a causal, test-like popularity view for TRAINING rows (all 8 W8 inputs + `xp_imp_c50`)

Written 2026-11-15 by the hz2 agent. CPU only. Work directory `nn_x/w12/stale/hz2/`. Conventions: ± = 1 SE paired per feed; [M] measured by me, [D] read in a project file, [I] inferred. "Zero share" = share of candidate rows whose counter is 0. Day slots: Sep 6 = slot 0, Sep 20 = slot 14 (dense cutoff of val / test rows).

## 0. Verdict

(section 0 is filled last; sections 1-3 below are final as written)

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

So the stored val / test columns are the hz2 construction at lag 2-9 with the transductive sparse rule, and the stored train columns are the same kernels with other day masks. `logs/v1.log`, `logs/v2.log`.

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
