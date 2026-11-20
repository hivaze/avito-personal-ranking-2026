
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
- **Declared verdict: hz2 FAIL on all days, PASS on days >= Sep 13.** The all-day failure is Sep 6-8 (27% of train feeds, 17% of pool feeds): no causal past exists there, and the stored `seq_cv_*` / `xp_imp_c50` columns have the same defect on those days (dZ 0.28 / 0.35 as built).
- Share of the as-built gap that a view closes (trPool private-like, all days / days >= Sep 13): hz 33% / 32% for `it_hist_all_luo`, 0% for the `seq_cv` and `xp_imp` inputs; hz2 75% / 90%, 58% / 88%, 78% / 84%; hz2L 68% / 81%, 45% / 65%, 63% / 77%.

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
