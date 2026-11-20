# Feature family `cal_`: calendar and time-of-day context

Code: `<repo>/feats_v2/cal_/` (`calcommon.py`, `e01_calendar_effects.py`, `b01_build.py` builds the features, `e02`-`e07` evaluate them, `run_harness_subsets.sh`). Logs and tables: `<repo>/feats_v2/cal_/out/`.
Harness entries in `reports/feature_evals.jsonl` have tags starting with `cal_`.

## 0. TL;DR

1. **Calendar effects on the ranking target are negligible. No `cal_` feature is kept.** I built and tested 16 features, deleted all of their files, and kept the code (`b01_build.py` rebuilds all 3 splits in about 2 minutes).
2. **Val harness.** Whole family: **-0.00023 ± 0.00067**. The best subset (two embedding "time-direction" features) gave +0.00077 ± 0.00067. A control run that adds **two pure-noise features gives +0.00071 ± 0.00068**, so every `cal_` delta is indistinguishable from noise.
3. **Weekend test on train** (val has no weekend days). LightGBM was trained on train feeds that look randomly ordered (`p_random > 0.5`), with users disjoint between fit and eval:
   - weekday → weekend: base + `cal_` gives -0.0004 / -0.0003 / -0.0007 (± 0.0006) for the label / demand / all groups.
   - weekend → weekday: **-0.0014 / -0.0013** (about -2 SE) / -0.0008. Context features *hurt* when the model is applied to the other day type.
   - Mixed training with an explicit weekend flag: weekend eval -0.0011 ± 0.0006.
   - Size-matched specialization test: a model trained only on weekends is *not* better on weekend feeds than a weekday model: **DiD = -0.0009 ± 0.0009**.
4. **The weekend barely changes the label-feature relationship.**
   - Position-debiased category CTR, weekend/weekday: the positive-weighted mean |log ratio| is 0.0129. The null from splitting weekdays 4 vs 4 is 0.0103 (95th percentile 0.0162).
   - Only 4 sizeable categories move by more than 3%: cat 10 ×0.971, cat 48 ×0.928, cat 25 ×0.909, cat 5 ×1.032.
   - Single-feature NDCG lifts of f1..f5, maxsim32, cos_u, attn32 and cat_cnt change by ≤ 0.0015 between weekday and weekend (similarity features are about 2% weaker on weekends).
5. **Time of day is equally weak.**
   - Category × hour-bucket lift: rms 1.5%, split-half reliability 0.31 on train. The train lift correlates with val's own lift at only 0.09-0.12.
   - Removing the existing base feature `hour` changes val NDCG by +0.00023 ± 0.00068, i.e. it has no measurable value.
   - **Recommendation: drop `hour` from the base set.** It is useless on val, and on test it moves (public is 56% 00-12h, plus logging-outage hours), so it is a pure risk.
6. **Russian calendar.** The official 2026 production calendar (Government decree No. 1466 of 24.09.2025) has **no holidays, no shortened days and no day transfers in September 2026**: 22 working days and 8 weekend days. Test days Sat 26 / Sun 27 are ordinary weekends and Mon 28 / Tue 29 ordinary workdays.
   - Typical paydays (5th/10th/20th/25th) cannot be identified from 12 train days because they are confounded with day of week. Day-level category variation is already at the noise floor.
   - Nothing calendar-specific can be built that val could validate.
7. **Harness calibration.** All runs share one cached baseline per seed, and seed 2's baseline is a "bad draw": adding *any* perturbation, even noise, gives about +0.002 on seed 2. Family deltas below about +0.0010 (noise control +0.0007 plus 1 SE) should therefore be read against a noise control, not against 0. I recommend the orchestrator keep `cal_noise_control_2feats` (+0.00071) as a reference line.
   - Also, rerunning the base in my process reproduced the cached OOF exactly, so LightGBM is deterministic here. The seed-2 offset comes from the shared baseline, not from thread-count nondeterminism.

## 1. Calendar facts

| | value |
|---|---|
| Time zone of day boundaries | MSK (UTC+3). All hours below are MSK. |
| Russian production calendar, Sep 2026 | 22 working days and 8 days off (5,6,12,13,19,20,26,27). No public holidays, no pre-holiday shortened days, no transfers. Sources: [Alfa-Bank calendar Sep 2026](https://kurs.alfabank.ru/work-calendar/year/2026/month/9/), [nalog-nalog.ru Sep 2026](https://nalog-nalog.ru/proizvodstvennyj_kalendar/2026/m9/) |
| Train days | Sun 6, Mon 7 - Fri 11, Sat 12, Sun 13, Mon 14, Thu 17, Fri 18, Sat 19. Weekend days: 6, 12, 13, 19. |
| Val days | Tue 22, Thu 24 (weekdays only) |
| Test days | Sat 26, Sun 27 (public = Sat + Sun until 11:32), Mon 28, Tue 29 |
| Days since last typical payday (5/10/20/25) | train 0-9; val 2 and 4; test 1-4. In val this is just another name for "Tue vs Thu". |

## 2. Do calendar effects exist? (train, 3.15M feeds, `e01`, `e07`)

Debiasing: expected positives per row = npos_feed × θ(slot) / Σθ. θ is a position-only curve (rank bucket × last / second-to-last / third-to-last) fitted on randomly-ordered-looking train feeds (`p_random > 0.9`, M ≥ 50). The flat expectation (npos/M) gives almost the same numbers, so position is not a confounder here.

### 2.1 Category CTR, weekend vs weekday (CI from day-to-day variation: 4 weekend days, 8 weekdays)

| cat | positives | supply share wd / we | CTR ratio we/wd | 95% CI | z |
|---|---|---|---|---|---|
| 13 | 837,452 | 12.95% / 13.33% | 1.007 | [0.990, 1.024] | +0.8 |
| 2 | 605,029 | 9.66% / 9.48% | 1.009 | [0.994, 1.024] | +1.1 |
| 3 | 462,484 | 8.37% / 8.28% | 1.002 | [0.986, 1.018] | +0.3 |
| 50 | 310,960 | 7.78% / 7.78% | 0.988 | [0.973, 1.004] | -1.5 |
| 6 | 309,798 | 5.99% / 5.86% | 0.996 | [0.986, 1.006] | -0.7 |
| 7 | 291,235 | 5.26% / 5.29% | 1.014 | [0.991, 1.038] | +1.2 |
| 10 | 249,672 | 3.95% / 3.98% | **0.971** | [0.962, 0.981] | -5.6 |
| 48 | 239,804 | 4.19% / 3.84% | **0.928** | [0.886, 0.972] | -3.1 |
| 15 | 237,601 | 3.63% / 3.67% | 0.999 | [0.966, 1.034] | -0.0 |
| 11 | 136,542 | 2.52% / 2.61% | 1.010 | [0.996, 1.023] | +1.4 |
| 16 | 134,819 | 2.44% / 2.49% | 1.011 | [1.001, 1.022] | +2.2 |
| 5 | 73,239 | 1.31% / 1.28% | **1.032** | | +3.7 |
| 25 | 22,170 | 0.68% / 0.66% | **0.909** | | -3.7 |

- 7 of 49 categories with more than 5K positives have |z| > 2. About 3-5 would be expected by chance with these few day clusters.
- Positive-weighted mean |log ratio|: 0.0129 for weekend vs weekday, against 0.0103 for a random 4-vs-4 weekday split (95th percentile 0.0162). So the weekend adds only about 0.8% rms of category-level change on top of ordinary day-to-day noise. A ±1-3% multiplicative shift almost never reorders candidates whose scores span orders of magnitude.

### 2.2 Time of day (7 buckets: 0-6, 6-9, 9-12, 12-15, 15-18, 18-21, 21-24)

- **Label lift**, category × hour bucket: rms 1.5%. Split-half (odd vs even days) correlation is 0.31 on weekdays and 0.16 on weekends. The weekday and weekend profiles correlate at 0.34.
- On val (unbiased labels), the train lift correlates with val's own category × hour lift at only 0.115 (ctx) and 0.092 (hour bucket) over 207 cells (`e05`).
- **History demand** (newest 5 history events): rms 5.7%, split-half 0.84 / 0.74. This is much more systematic, but it mostly reflects *who* is active at that hour.
  - Its correlation with the label lift is 0.29 (category × ctx) and 0.54 (weekend ratio per category).

### 2.3 Feature-label relation by day type (single-feature NDCG lift over random, exact expected ties)

| feature | feeds | weekday | weekend | we - wd ± SE | range over 7 hour buckets |
|---|---|---|---|---|---|
| f1 | all | 0.0391 | 0.0393 | +0.0002 ± 0.0002 | 0.0374-0.0405 |
| f1 | random-like | 0.0389 | 0.0398 | +0.0009 ± 0.0006 | 0.0382-0.0407 |
| f3 | all | 0.0270 | 0.0271 | +0.0001 ± 0.0001 | 0.0259-0.0278 |
| f5 | all | 0.0174 | 0.0168 | -0.0006 ± 0.0002 | 0.0163-0.0178 |
| maxsim32 | all | 0.0422 | 0.0418 | -0.0005 ± 0.0002 | 0.0403-0.0439 |
| maxsim32 | random-like | 0.0430 | 0.0419 | -0.0011 ± 0.0007 | 0.0414-0.0439 |
| attn32 | all | 0.0401 | 0.0394 | -0.0007 ± 0.0002 | 0.0382-0.0418 |
| cos_u | all | 0.0347 | 0.0343 | -0.0003 ± 0.0002 | 0.0335-0.0358 |
| cos_h10 | all | 0.0426 | 0.0422 | -0.0004 ± 0.0002 | 0.0406-0.0443 |
| cat_cnt_all | all | 0.0186 | 0.0185 | -0.0002 ± 0.0001 | 0.0181-0.0190 |
| tte_it_coec | all | 0.0104 | 0.0082 | -0.0022 ± 0.0002 | 0.0082-0.0131 (falls monotonically from night to 21-24h) |

All rows, including f2 and f4: `out/e07_feature_relation.csv`.

- Embedding-similarity features are about 1-3% weaker on weekends and peak at 12-18h. The hour pattern also shows up in the random baseline, so it is mostly feed composition.
- **Side finding for the `life_` / `trd_` owners.** `tte_it_coec` loses about 40% of its lift from night to late evening. Items in late-day feeds are newer, so their earlier-day train statistics are missing. This is an item-freshness effect, not a calendar effect. The weekend gap for this feature is an artifact of Sep 6 being the first train day.

## 3. Feature definitions (all built for train/val/test, then deleted)

Feed context: ctx = weekend × 7 + hour_bucket. "Lift" = log((observed + a) / (expected + a)), shrunk toward 0.

| feature | definition | type |
|---|---|---|
| `cal_lab_cat_ctx` | Category lift in the feed's ctx vs the category over all ctx, from position-debiased train labels (K=50 pseudo-positives). Train rows: strictly earlier days only. | cand, label |
| `cal_lab_cat_we` | Same, with ctx = day type only | cand, label |
| `cal_lab_cat_hb` | Same, with ctx = hour bucket only (pooled over day types) | cand, label |
| `cal_lab_emb_ctx` | item_unit · δ_ctx, where δ_ctx = (mean emb of positives − mean emb of expected positives) in ctx minus the same over all ctx, shrunk by n/(n+2000). Dense, so it works for cold items. Time-aware. | cand, label |
| `cal_dem_cat_ctx` | Category lift among the newest-5 history events of train feeds in the same ctx (demand by time) | cand, unlabeled |
| `cal_dem_emb_ctx` | item_unit · (mean newest-5 history emb in ctx − overall mean) | cand, unlabeled |
| `cal_sup_cat_ctx` | Transductive: category lift among the split's own candidates in the same ctx (supply) | cand, transductive |
| `cal_ds_cat_ctx` | dem − sup | cand |
| `cal_drift_cat_day` | Transductive: category share among newest-5 history events of same-split, same-day feeds vs the train-wide share | cand, transductive |
| `cal_mod`, `cal_weekend`, `cal_dow`, `cal_payday` | Minute of day / 1440, weekend flag, day of week, days since the last typical RU payday | feed-level |
| `cal_u_hb_share`, `cal_u_we_share`, `cal_u_ntr` | User activity profile: share of the user's earlier train feeds in the same hour bucket / on weekends, and their count (31% of val users have any) | feed-level |

## 4. Single-feature NDCG on val (expected NDCG under random tie-break; random = 0.2890)

| feature | NDCG + | NDCG − |
|---|---|---|
| cal_lab_cat_ctx | 0.2791 | 0.2811 |
| cal_lab_cat_we | 0.2745 | 0.2855 |
| cal_lab_cat_hb | 0.2812 | 0.2805 |
| cal_lab_emb_ctx | 0.2798 | 0.2863 |
| cal_dem_cat_ctx | 0.2802 | 0.2835 |
| cal_dem_emb_ctx | 0.2789 | 0.2852 |
| cal_sup_cat_ctx | 0.2825 | 0.2818 |
| cal_ds_cat_ctx | 0.2808 | 0.2825 |
| cal_drift_cat_day | 0.2836 | 0.2784 |
| feed-level (mod, weekend, dow, payday, u_*) | 0.2890 | 0.2890 (constant within a feed) |

- Every candidate-level feature scores *below random in both directions*. These features group candidates by category, and rare categories take the extreme values.
- A direct check of Σ(y − e)·x per feed on val shows that the train-derived lifts are *negatively* aligned with val positives (z = -2.6 for cat_ctx, -7.9 for cat_we, -4.2 for lab_emb). This comes from accidental correlation with cross-category main effects (only about 50 categories), not from time effects. Such features can inject spurious category main effects into a model.

## 5. Val harness (paired 3-seed 5-fold GroupKFold, base = 120 features, base NDCG 0.3634)

| run (tag) | n feats | delta | SE | per-seed deltas |
|---|---|---|---|---|
| `cal_all_v1` (whole family) | 16 | **-0.00023** | 0.00067 | -0.0021, -0.0010, +0.0023 |
| `cal_lab_v1` (4 label-based) | 4 | +0.00046 | 0.00067 | -0.0013, +0.0007, +0.0020 |
| `cal_demsup_v1` (demand/supply) | 4 | +0.00037 | 0.00067 | -0.0011, -0.0007, +0.0029 |
| `cal_lab_cat_hb_only` | 1 | +0.00053 | 0.00066 | -0.0010, -0.0003, +0.0029 |
| `cal_emb_dirs` (dem_emb + lab_emb) | 2 | +0.00077 | 0.00067 | +0.0012, -0.0011, +0.0022 |
| **control: 2 pure-noise features** | 2 | **+0.00071** | 0.00068 | +0.0008, -0.0012, +0.0025 |
| ablation: base minus `hour` | -1 | +0.00023 | 0.00068 | -0.0006, +0.0000, +0.0013 |
| null: base re-run (determinism check) | 0 | 0.0 | 0 | 0, 0, 0 |

- No run reaches 2 SE, and none beats the noise control.
- The val harness only sees weekday hours, so it cannot validate weekend values in any case.

## 6. Train weekend test (`e03`, `e06`)

Setup:
- Train feeds with `p_random > 0.5` and M ≥ 20, where labels do not depend on display order. Days ≥ Sep 8, so every label statistic has earlier days behind it.
- Fit and eval users are disjoint (hash halves).
- Same LightGBM parameters and rounds as the harness. Base = the 120 base features.
- Eval sets: 28,945 weekend feeds (Sep 12/13/19) and 35,457 weekday feeds (Sep 14/17/18).

| experiment | group | base | base + cal | delta ± SE |
|---|---|---|---|---|
| A: weekday (60K feeds) → weekend | lab | 0.3416 | 0.3412 | -0.0004 ± 0.0006 |
| A | dem | | 0.3414 | -0.0003 ± 0.0006 |
| A | all (incl. weekend flag, minute of day) | | 0.3409 | -0.0007 ± 0.0006 |
| B: weekend (29K) → weekday | lab | 0.3400 | 0.3386 | **-0.0014 ± 0.0007** |
| B | dem | | 0.3388 | **-0.0013 ± 0.0007** |
| B | all | | 0.3393 | -0.0008 ± 0.0007 |
| C: weekday → weekday (control) | lab | 0.3422 | 0.3423 | +0.0001 ± 0.0006 |
| C | dem | | 0.3426 | +0.0005 ± 0.0006 |
| C | all | | 0.3420 | -0.0002 ± 0.0005 |
| D: mixed (30K + 30K) → weekend | lab | 0.3429 | 0.3433 | +0.0004 ± 0.0006 |
| D | dem | | 0.3427 | -0.0003 ± 0.0006 |
| D | all | | 0.3418 | -0.0011 ± 0.0006 |
| D: mixed → weekday | lab / dem / all | 0.3422 | | -0.0006 / -0.0005 / -0.0006 (± 0.00055) |

**Specialization DiD (`e06`; base features only, 29K feeds per day type, 2 seeds).**

| model | weekend eval | weekday eval |
|---|---|---|
| weekday-trained | 0.3395 | 0.3390 |
| weekend-trained | 0.3389 | 0.3393 |

- Weekend model minus weekday model on weekend feeds: -0.0006 ± 0.0007. On weekday feeds: +0.0003 ± 0.0006. **DiD = -0.0009 ± 0.0009.**
- A model trained on weekday data alone transfers to weekends without measurable loss.
- Implication for test: the weekday-only val set is not a handicap for the weekend part of test (public = 100% weekend, private = 24%). Calendar features add risk, not signal, because a val-trained model sees out-of-range values on weekends.

## 7. Drift check (PSI, 10 val-quantile bins; `out/e04_drift.csv`)

| feature | val→public | val→private | public→private | train→val |
|---|---|---|---|---|
| cal_lab_cat_ctx | 0.245 | 0.013 | 0.187 | 0.118 |
| cal_lab_cat_we | **3.08** | 0.124 | 1.94 | 0.94 |
| cal_lab_cat_hb | 0.015 | 0.012 | 0.046 | 0.101 |
| cal_lab_emb_ctx | 0.843 | 0.057 | 0.468 | 0.134 |
| cal_dem_cat_ctx | 0.246 | 0.020 | 0.147 | 0.033 |
| cal_dem_emb_ctx | 0.142 | 0.041 | 0.044 | 0.035 |
| cal_sup_cat_ctx | 0.067 | 0.103 | 0.087 | 0.258 |
| cal_ds_cat_ctx | 0.143 | 0.087 | 0.069 | 1.79 |
| cal_drift_cat_day | 1.00 | 0.232 | 0.648 | 0.89 |
| cal_mod | 0.207 | 0.161 | 0.384 | 0.046 |
| cal_weekend / cal_dow / cal_payday | 18.4 / 4.9 / 6.7 | 2.0 / 0.23 / 0.00 | | |
| cal_u_hb_share / u_we_share / u_ntr | ≤ 0.005 | ≤ 0.003 | | |

- By construction, every weekend-aware feature shifts strongly from val to public (weekday values in val, weekend values in public).
- `cal_drift_cat_day` mostly measures the val/test-vs-train *population* difference (its train range is ±0.04 against ±0.12 in val/test), not seasonal drift.
- Only `cal_lab_cat_hb` and the user profile features are stable, and they have no signal.

## 8. Ideas considered and not built

- **Holidays / shortened days:** none in Sep 2026, so the feature would be constant.
- **Payday phase:** confounded with day of week across 12 days, and day-level effects are at the noise floor.
- **Sunset/daylight:** equivalent to the hour bucket within the window (sunset moves about 3 min/day).
- **Local time zone inference:** no location signal.
- **Session continuity** (time since the user's previous feed): about 1 feed per user in val/test, so almost always undefined.
- **Item freshness / age and in-split trends:** these belong to the `life_` / `trd_` families. The `tte_it_coec` hour profile in §2.3 suggests freshness is real.
- **Personal category × time-of-day profiles:** under 31% coverage, with mostly 1-2 train feeds per user.
- **Context-specific slopes for f-features:** the relations are flat (§2.3).

## 9. KEEP list

**None.** All `cal_*` feature files were deleted from `data/feats/{train,val,test}`.

Actionable recommendations:
1. Drop `hour` from the base set. It has no value on val, and it shifts on test because of weekend/morning-heavy public and outage hours.
2. Do not add weekday/weekend indicators or time-context lifts to the final model.
3. Treat harness deltas below about +0.001 as noise, using the noise control as the reference.

## Verification (adversarial verifier, 2026-11-11)

**Verdict: DROP the whole family. KEEP list = none. This confirms the builder.** The builder claimed no gain, and every check below agrees: no `cal_` feature or subset beats a pure-noise control, on the val harness or on a Tue→Thu time split. All 16 `cal_` files were rebuilt for these checks with `b01_build.py` (the rebuild is deterministic: same label-direction norms as `out/b01.log`) and then deleted again from `data/feats/{train,val,test}`. Verifier code: `feats_v2/cal_/v01_verify_drift_single.py`, `v02_verify_timesplit.py`. Outputs: `feats_v2/cal_/out/verify/`.

### V1. Leakage and asymmetry audit (code read of `b01_build.py`, `calcommon.py`)
- **Val labels:** never read by any feature. `e05` reads val labels for analysis only.
- **Train labels for train rows:** `cal_lab_cat_*` use `cumsum - current day`, i.e. strictly earlier days. `cal_lab_emb_ctx` uses `acc_cum` (strictly earlier days), and rows without earlier positives get NaN (9.6% of train rows, the first day). Val/test use all train days. Correct.
- **Position curve θ** (`theta_cells.npy`) is fitted on train only.
- **Val vs test symmetry:** both splits go through the same `write_split` path with the same train tables. Transductive parts use only the split's own data, for val and for test alike:
  - `sup_cat_ctx` uses the split's own candidates.
  - `drift_cat_day` uses newest-5 history of same-split, same-day feeds.
  - No future windows, no use of test data for val (or the reverse), and no split-edge effect.
  - The 1 val feed on Wed 00:xx and 2 on Fri 00:xx MSK form their own tiny "day" in `drift_cat_day`, which is harmless.
- **Minor non-leak issues** (moot because nothing is kept):
  - (a) `sup`/`ds`/`drift` mean something different on train than on val/test. They are computed over 3.15M train feeds against 14-29K split feeds, so their range is ±0.03-0.04 on train vs ±0.10-0.12 on val/test. That matters only for a train-fitted model.
  - (b) `demand` tables include each train row's own feed history (unlabeled, aggregated at cat×ctx, negligible).
  - (c) `payday_days` hard-codes a 31-day previous month. That is correct for Sep 2026 only.
- **No bugs needed fixing.**

### V2. Train-file integrity (rebuilt files)
- All 16 features had the correct length in every split (train 246,745,836 / val 1,389,971 / test 2,904,806 rows), dtype float32, and no all-NaN columns.
- `cal_weekend` is **constant (0) on val**, and `cal_dow` / `cal_payday` take only 2 values there. The val harness therefore cannot learn or validate them.
- Single-feature lifts of candidate-level features are below random in both directions on train random-like feeds (`e02`) as well as on val. The semantics are consistent: they act as coarse category groupings, not time signals.

### V3. Single-feature NDCG (val, `src/metrics.py` pessimistic ties + random tie-break jitter, 3 draws; random = 0.2897)
Reproduces §4 within 0.0002. Examples:
- `cal_dem_emb_ctx` 0.2789 / 0.2852
- `cal_lab_emb_ctx` 0.2798 / 0.2863
- `cal_lab_cat_hb` 0.2814 / 0.2808
- `cal_lab_cat_we` 0.2749 / 0.2856

Every candidate-level feature is below random in both directions. Feed-level features score exactly random. Full table: `out/verify/v01_drift_single.csv`.

### V4. Harness re-run, 5 seeds (`--seeds 5 --threads 16`)
| tag | delta | SE | per-seed (s0..s4) | mean without s2 |
|---|---|---|---|---|
| `cal_VERIFY_emb_dirs_s5` (dem_emb + lab_emb, the best subset) | +0.00047 | 0.00052 | +0.0012, -0.0007, **+0.0022**, +0.0003, -0.0006 | +0.00004 |
| `cal_VERIFY_all16_s5` (whole family) | -0.00034 | 0.00051 | -0.0021, -0.0005, **+0.0024**, -0.0017, +0.0002 | -0.00102 |
| `cal_VERIFY_noise2_s5` (2 N(0,1) columns) | -0.00016 | 0.00052 | -0.0005, -0.0011, **+0.0027**, -0.0003, -0.0016 | -0.00088 |
| `cal_VERIFY_drop_hour_s5` (base minus `hour`) | +0.00026 | 0.00051 | -0.0006, +0.0000, **+0.0013**, -0.0004, +0.0010 | +0.00001 |

- The builder's best subset falls from +0.00077 (3 seeds) to **+0.00047 ± 0.00052 (0.9 SE)**, and to +0.00004 when the biased seed 2 is excluded.
- Seed by seed against the noise control: emb_dirs − noise = +0.00063 ± 0.00037 (1.7 SE), and all16 − noise = -0.00018.
- **No subset reaches 2 SE.**

### V5. Time-split check (`v02`)
Setup: LightGBM with the same PARAMS and 350 rounds as the harness, trained on Tue Sep 22 val feeds (7,543) and evaluated on Thu Sep 24 (6,880), then the reverse. 3 seeds per direction; the per-feed delta is averaged over seeds.

| config | Tue→Thu | Thu→Tue | pooled delta ± SE |
|---|---|---|---|
| + emb_dirs (2) | -0.00001 | -0.00003 | **-0.00002 ± 0.00068** |
| + all 16 | -0.00118 | -0.00056 | **-0.00086 ± 0.00068** |
| − `hour` | +0.00006 | -0.00088 | -0.00043 ± 0.00068 |
| + 2 noise columns (control) | +0.00022 | +0.00181 | +0.00105 ± 0.00070 |

- In this check the noise control beats every `cal_` configuration.
- Base NDCG alone varies by up to 0.0037 between seeds (Tue→Thu: 0.3600 / 0.3638 / 0.3626). The variance from refitting the model is larger than any `cal_` effect.

### V6. Drift (PSI with 10 val-quantile bins; public/private boundary ts 1790497920; test weekday = Mon/Tue, test weekend = Sat/Sun)
- Reproduces §7 exactly.
- Weekend-aware features drift strongly from val to public:
  - `cal_weekend` 18.4, `cal_payday` 6.7, `cal_dow` 4.9, `cal_lab_cat_we` 3.1
  - `cal_drift_cat_day` 1.0, `cal_lab_emb_ctx` 0.84
- Weekday-vs-weekend test PSI: `cal_lab_cat_we` 3.06, `cal_lab_emb_ctx` 0.69.
- For the same features, val vs test-weekday PSI is ≤ 0.01. The drift is purely day-type, a region val cannot validate.
- Stable features (PSI < 0.05 everywhere) are `cal_lab_cat_hb` and `cal_u_*`, and they have no signal.
- Base feature `hour`: PSI val→public 0.19, val→private 0.16, public→private 0.38. Public is 55.6% 00-12h; private is 24.2% weekend.

### V7. Per-feature verdicts
Every feature is **DROP**:
- `cal_lab_cat_ctx`, `cal_lab_cat_we`, `cal_lab_cat_hb`, `cal_lab_emb_ctx`
- `cal_dem_cat_ctx`, `cal_dem_emb_ctx`, `cal_sup_cat_ctx`, `cal_ds_cat_ctx`, `cal_drift_cat_day`
- `cal_mod`, `cal_weekend`, `cal_dow`, `cal_payday`
- `cal_u_hb_share`, `cal_u_we_share`, `cal_u_ntr`

Reasons:
- No feature is above noise, either alone or in a group (V4, V5).
- Weekend-aware features also carry large, unvalidatable day-type drift (V6).
- `cal_weekend` is constant on val.

### V8. Harness issues confirmed (for the orchestrator)
1. **Seed-2 bias is real and affects every family.** Over all 42 runs in `feature_evals.jsonl` with ≥3 seeds (as of this check):
   - Mean per-seed delta is +0.00035 / +0.00033 / **+0.00281** for s0/s1/s2.
   - s2 > s0 in 95% of runs.
   - In the 5-seed runs, s3 and s4 behave like s0 and s1.
   - The cached `base_oof_s2.npy` is a low draw. Every 3-seed family delta is inflated by about +0.0008.
   - Recommendation: report the mean over seeds without s2, or compare against a noise-control run, or add seeds 3 and 4.
2. **The feed-level SE understates the uncertainty.** `delta_se` covers only feed sampling (about 0.0005). Model-refit variance per seed is about 0.001-0.0015 (V5 base spread; noise-control per-seed range -0.0016 to +0.0027).
   - A seed-level SE (sd of per-seed deltas / √seeds) should be shown alongside it.
   - A 2-feature noise column gives anything from -0.0009 to +0.0011 depending on protocol.
3. **The harness is probably not fully deterministic.** Identical inputs produced different results in one of three seeds:
   - `cal_emb_dirs` seed 1 gave -0.00109 in the builder's run and -0.00074 in this rerun, while seeds 0 and 2 matched exactly.
   - `drop hour` reproduced exactly.
   - The likely cause is LightGBM's timing-based col-wise/row-wise auto-selection on a loaded machine. Setting `force_col_wise=True, deterministic=True` in PARAMS would remove it. This changes the cached base, so all base OOFs would have to be recomputed.

### V9. On the builder's "drop `hour` from base" recommendation
- The data is neutral. Dropping `hour` gives +0.00026 ± 0.00051 in the 5-seed harness (+0.00001 without s2) and -0.00043 ± 0.00068 on the time split.
- Dropping it is low-risk and removes a moderately drifting feed-level input (V6). Keeping it is not measurably harmful on val either.
- I classify it as optional cleanup, not a required change. If the orchestrator wants a more robust public/private transfer, drop it.
