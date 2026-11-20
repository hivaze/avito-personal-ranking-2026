# Feature family `life_`: item lifecycle, freshness and sold-out proxies

Code: `<repo>/feats_v2/life_/`. Harness results: `reports/feature_evals.jsonl` (tags `life_R1`..`life_R7`).
Harness protocol: `src/eval_family.py`, 3 seeds × 5-fold GroupKFold by user, LightGBM lambdarank on val, on top of the frozen 122-feature base (base NDCG 0.3634).
"Single NDCG" means the full-list NDCG of one feature on val, scored in both directions, with NaN placed last and random tie-breaking. Random order gives 0.2896 on this measure.

---

## 0. TL;DR

1. **`f4` is the listing's exact age, so freshness is already in the base set.** For the same item, f4 never decreases. 97% of 1.1M items seen on three train days increased; none decreased. Its growth over a time gap is a deterministic function of its level.
   A 64-knot piecewise-linear map `age(f4)` (knots at k/64) fitted on 3M pairs of first and last appearances reproduces every observed gap with a **median error of 0.4 s** (q99 49 s).
   Knot ages: 1/64 = 0 h, 9/64 = 24 h, 20/64 = 6.8 days, 30/64 = 24 days, 40/64 = 76 days, 50/64 = 206 days, 63/64 = 4.6 years (cap).
   As a result, **`t_post = t_feed − age(f4)` recovers each listing's creation time.** It lands exactly on the dataset's 128-s timestamp grid and is consistent across appearances of the same item (median std 0.08 s).
   This explains why none of the age or first-seen proxies below adds anything. The model already knows the exact age of every candidate.
   The knots are saved in `data/feats/life_f4_age_knots.npy`; use `post_time.age_hours()` to apply them.
2. **No group improves the base by a significant margin, and one group hurts.** All results are from the harness, 3 seeds:

   | run | features | delta | SE | per-seed |
   |---|---|---|---|---|
   | R1 SS robust (within-split stream) + all-data age | 16 | +0.00034 | 0.00068 | −0.0014 / −0.0012 / +0.0036 |
   | R2 same features with a val+test future (optimistic) | 16 | +0.00085 | 0.00070 | −0.0013 / +0.0012 / +0.0027 |
   | R5 robust core (5 SS features) | 5 | +0.00020 | 0.00068 | +0.0001 / −0.0014 / +0.0019 |
   | R7 same core, val+test future | 5 | −0.00049 | 0.00068 | −0.0013 / −0.0009 / +0.0007 |
   | R3 train-stream lifecycle (TS) | 18 | −0.00013 | 0.00070 | −0.0005 / −0.0013 / +0.0015 |
   | R4 posting time + user freshness preference + diurnal phase | 27 | **−0.00173** | 0.00071 | −0.0024 / −0.0034 / +0.0006 |
   | R6a side test: item-centred f deviations (not lifecycle) | 9 | +0.00072 | 0.00070 | −0.0002 / −0.0008 / +0.0031 |
   | R6b same, 2 best features | 2 | +0.00026 | 0.00067 | −0.0014 / −0.0003 / +0.0025 |

   Note on the harness: in all 44 three-seed runs in the log, **seed 2 gives a systematically positive delta.** The mean per-seed delta is +0.0029 for s2 against +0.0005 for s0 and s1, and 95% of runs are positive on s2. Every small positive life_ result comes from seed 2 alone.
3. **The future-looking "item still alive" signal is real, but almost all of it needs a long future that private-test feeds do not have.**
   The test is: was the item seen again (as a candidate in another feed, or in another user's history) within H hours after the feed? Its conditional-logit LR on top of the base OOF grows with H: H = 24 h gives 3.4, 48 h 9.6, 72 h 16, 96 h 23, 144 h 46, and the full remaining data 53.
   The effect per detected item stays roughly constant (logit +0.06..+0.11). LR grows only because more items get detected.
   With a censoring-robust within-split definition, the strongest feature has LR ≈ 10. Even the optimistic version (LR 49 jointly) does not move NDCG: the cross-fitted linear stack gives −0.0003 and the GBDT R2 gives +0.0009 ± 0.0007.
4. **Censoring handling (as specified).** The sparse-stream (SS) features are kernel or window rates per observed stream feed, and they use a **within-split stream**: val rows see val feeds, test rows see test feeds. That is the identical definition for both, and it is exactly the "as if val were the last split" simulation.
   Under this definition val behaves like private test. The share of rows with a nonzero future signal (`h_kn72`) is 4.9% on val, 6.5% on private (Sun-pm 10.6%, Mon 7.4%, Tue 2.8%) and 13.4% on public.
   The val+test-future variant (R2/R7) is the optimistic one. It gains +0.0005 over R1 on 16 features, the opposite on 5, and neither difference is significant.
5. **KEEP list: empty.** Every feature file `life_*` was deleted from `data/feats/{train,val,test}`. The code stays and can regenerate everything: about 10 min for train, seconds for val/test.
   Two artifacts are kept for other families: the f4→age knots and the finding that f4 is age (§1).

---

## 1. Finding: f4 = exact listing age, which gives the posting time

Evidence (scripts: `post_time.py::fit_knots`, scratch checks reproduced in the code docstrings):
- Train items seen on Sep 6, Sep 10 and Sep 17 (1.10M items). Mean f4 went from 0.595 to 0.622 to 0.653. **97.2% went up, 0.0% went down.** For comparison, f5 went 41% up and 58% down, and f1 50/49.
- The 11-day increment depends only on the starting level. In the f4 bin [1/64, 2/64) the mean increment is 0.357 with std 0.006. In bin 62 it is 0.00048 with std 0.00005, and at 63/64 it is exactly 0 (the cap).
- Least-squares fit of the piecewise-linear map age(f4) on (first, last) appearance pairs, 3M pairs with gap > 1 day: median |error| 0.40 s, q99 49 s.
- `t_post mod 128 = 0` for 99% of items younger than 30 days. Posting times therefore sit on the same 128-s grid as the feed timestamps and are recovered at full resolution. The std of t_post across appearances of the same item has a median of 0.08 s (q90 4 s).

Consequences:
- f4 already carries the candidate's age. Within a feed, any "age", "first seen" or "new since last visit" feature is a monotone function of f4 plus a feed-level constant, so it carries no new ranking information. That matched what we measured: `life_age_any_h` LR 2.2, `life_ua_new_since` LR 0.2.
- f4 drifts between train and val/test because items age; it is not a distribution shift in any deeper sense. A model can use exact age to make train-time statistics age-aware, for example an item's train exposures per day of life.
- Posting-time attributes carry no signal: hour or day of posting, and 128-s-slot overdensity as a bulk-upload or seller proxy. Joint LR < 3 per feature. R4 hurts. In each 128-s slot ~130-200 unrelated listings from all of Russia share the same posting time, so individual sellers' batches cannot be identified at this resolution.

## 2. Feature definitions (everything built, then evaluated)

All features are label-free (train labels are never used, val labels never). Each is defined identically for val and test; train rows use strictly earlier train data.

### 2.1 TS: train-stream lifecycle (`ts_train_stream.py`)
The dense train stream is replayed in time order with a per-item state.
- For train rows the state is read before the feed's own timestamp group is added.
- For val/test rows the state is taken at the end of train, `TR_END` = Sep 20 00:00 MSK, and all gaps are measured relative to TR_END. This makes the definition identical for val and test, with no dependence on the gap to train.
- "Dated interactions" are history appearances among the user's 8 most recent events that were not in the same user's previous feed. Only 18% of train feeds have a previous same-user feed. The history is not a clean sliding window: only 9% of consecutive pairs align perfectly.

Features:
- Candidate exposure: `c_ndays`, `c_span_h`, `c_gap_h`, `c_n`, `c_dec24`, `c_per_day`.
- First appearance and last history appearance: `age_h`, `h_gap_h`.
- Dated interactions: `i_n`, `i_n1` (event_type 1), `i_gap_h`, `i_age_h`, `i_dec24`/`i_dec72` (exp-decayed counts with 24 h and 72 h half-lives), `i_trend` = dec24/(dec72+0.5), `i_per_day`.
- `eng_rate` = i_n/(c_n+10), a label-free engagement rate.
- `hour_aff`, the item's dated-interaction share in the feed's 4-hour MSK bucket relative to the global share.

### 2.2 SS: sparse-stream local lifecycle (`ss_sparse_stream.py`)
The stream is the within-split exploration feeds: val rows use val, test rows use test, and train rows use a seeded 2.8% subsample of train, which matches the density of about 7k feeds/day. The row's own feed and the same non-zero user are excluded.

Appearance kinds:
- `c`: shown as a candidate.
- `h`: in a history (one per feed).
- `f`: in a history at hpos < 8 (a fresh interaction).
- `e`: in a history with event_type 1.

Measures:
- Kernel rates `{k}_kp{T}` / `{k}_kn{T}`, with T = 24 h and 72 h, looking back and ahead. The rate is Σ exp(−|dt|/T) over the item's appearances, divided by (Σ exp(−|dt|/T) over all observed stream feeds on the same side + 200), × 1e4.
- Window rates `{k}_r{p24,p72,n6,n24,n48,n96}`: count / (observed stream feeds in the window + 200) × 1e4.
- `a_kn{T}` = c + h ("served or interacted after t": the still-alive / not-sold proxy).
- `fut_obs` = log1p(observed feeds in (t, t+96 h]), the available-horizon feature.

Censoring: a truncated or gappy window lowers the denominator, not the expected rate, so the estimate is noisier but unbiased. The +200 pseudo-feeds shrink the estimate towards 0 when almost nothing is observed. `fut_obs` was not used: its PSI was 8.3 for val vs public, because the feed-level availability has a different range.
- `life_age_any_h` = t − first appearance in the train stream or in earlier own-split feeds, clipped at 360 h. 360 h is below the censoring point of every val/test feed: the data starts on Sep 6, at least 384 h before any val/test feed.
- `life_never_seen` is its "never seen before" indicator.

### 2.3 POST: posting time from f4 (`post_time.py`)
- `post_hour`, `post_min_of_day_c/_s`, `post_dow`.
- `post_slot_ratio`: distinct items in the same 128-s posting slot divided by the mean of ±30 neighbouring slots. It is a bulk-upload proxy and robust to censoring.
- `post_slot_n`.
- `post_hist_same_slot`: the user's history items with the same posting slot.
- `post_feed_same_slot`.

### 2.4 UA: user freshness preference (`uage.py`)
Built from history items' posting times (known for every history item that appears as a candidate anywhere; covers 98% of feeds):
- `ua_med`: median log-age at t over the user's last 32 events.
- `ua_fresh`: share of those events on items younger than 48 h.
- `ua_n`.
- `ua_diff`, `ua_absdiff`, `ua_z`: candidate log-age against the user's typical value.
- `ua_med_e1`, `ua_diff_e1`.
- `ua_new_since`: candidate posted after the newest-posted item in the user's history.
- `ua_u_gap`.

### 2.5 TZ: diurnal phase / time-zone proxy (`tz_phase.py`)
Russia spans 11 time zones and every timestamp is in MSK. The item's MSK-hour profile is built from its train impressions and fresh history appearances. The user's profile is built from their history items.
- `tz_item_night`: profile share at 00-06 MSK.
- `tz_item_rc/rs`, `tz_user_rc/rs`: phase residual vectors.
- `tz_align`, `tz_dist`: user-item alignment.
- `tz_feed_align`.
- `tz_item_n`.

Sanity check: the user residual rotates with the feed hour as expected.

### 2.6 Side test (not lifecycle): `fdev_side.py`
- `x_d{1,2,3}_mean`: f_k minus the mean of f_k over the item's train appearances.
- `x_d5_last`, `x_slope5`: f5 drift.
- `x_m{1,2,3}`.

## 3. Single-feature NDCG on val (both directions)

Random order scores 0.2896. f1 scores ~0.336.

| feature | nd+ | nd− | coverage val | note |
|---|---|---|---|---|
| ts_i_n1 / ts_i_n | 0.3094 / 0.3090 | 0.293 | 0.71 | popularity-like; base already has it_hist_* |
| ts_i_dec72 / i_per_day | 0.3076 / 0.3077 | 0.293 | 0.71 | |
| ts_age_h | 0.3046 | 0.2920 | 0.71 | |
| ts_eng_rate | 0.3042 | 0.2936 | 0.71 | |
| ts_c_n / c_ndays / c_span_h | 0.3037 / 0.3009 / 0.3007 | ~0.295 | 0.65-0.71 | within popularity strata, more days shown means a *lower* positive rate (staleness) |
| ts_h_gap_h | 0.2965 | 0.3063 | 0.59 | |
| life_age_any_h | 0.3013 | 0.2853 | 0.75 | |
| life_never_seen | 0.2770 | 0.2961 | 1.00 | items never seen in train: positive rate 1.25% vs 1.72% |
| ss_h_rp72 / h_kp72 | 0.2973 / 0.2969 | 0.288 | 1.00 | best past SS feature |
| ss_e_rp72 / e_kn72 / e_rn96 | 0.2961 / 0.2958 / 0.2959 | 0.288 | 1.00 | |
| ss_h_kn72 / h_rn96 | 0.2957 / 0.2959 | 0.288 | 1.00 | |
| ss_f_* | 0.290-0.292 | 0.289 | 1.00 | very sparse (0.7% nonzero) |
| ss_c_* | 0.286-0.288 | 0.290 | 1.00 | |
| x_d1_mean / x_d3_mean | 0.3153 / 0.3124 | 0.275 | 0.65 | side test |
| ua_absdiff | 0.2683 | 0.3014 | 0.98 | proxy of f4 inside the feed |
| ua_diff / ua_z | 0.2970 | 0.2674 | 0.98 | |
| tz_item_n / tz_align / tz_dist | 0.3038 / 0.2977 / 0.2936 | 0.278 / 0.294 / 0.2997 | 0.66-1.0 | |
| post_dow / post_hour / post_slot_ratio | 0.2959 / 0.2883 / 0.2873 | 0.2863 / 0.2909 / 0.2924 | 0.98 | |

**Marginal information over the base** is measured with a conditional logit: a per-feed softmax of `a·base_OOF + b·q(x) + c·isnan(x)`, reported as LR against base only. Script: `clogit_screen.py`. For calibration, a joint LR of 27-49 still gives no NDCG change; see `clogit_combo.py`.

| group | best features (LR) |
|---|---|
| SS within-split (robust) | e_kn72 9.9, e_kn24 9.2, h_rp72 8.6, e_rn96 8.4, f_rn96 7.8, a_kn72 7.7, h_kn72 7.6 (joint 16 features: 26.9; cross-fit ΔNDCG −0.0003) |
| SS val+test future (optimistic) | e_kn24 29.4, e_kn72 28.4, h_kn72 28.2, a_kn72 27.5 (joint 49.3; cross-fit ΔNDCG −0.0003) |
| all-data "seen again later" (optimistic, `next_any_h`) | 54 with a 5-7 day future; 8.9 when val is the last split |
| TS | ≤ 6.9 (df = 2) |
| POST / UA / TZ | ≤ 2.6 |
| side: x_d1_mean / x_d3_mean | 25.5 / 23.6 (joint 47.6; cross-fit ΔNDCG −0.0001) |

## 4. Censoring: simulation and the near-end-of-data analysis

**Horizon dependence** (`horizon_check.py`, val+test stream). The measured event is "item seen again within H hours".

| H | 6 | 12 | 24 | 48 | 72 | 96 | 144 | all |
|---|---|---|---|---|---|---|---|---|
| share seen again (any kind) | 5.8% | 8.2% | 9.1% | 14.6% | 19.9% | 24.3% | 32.1% | 34.5% |
| LR (any kind) | 1.5 | 0.9 | 3.4 | 9.6 | 16.2 | 23.2 | 45.8 | 52.5 |
| LR (in another user's history, event_type 1) | 1.2 | 2.0 | 3.7 | 11.1 | 18.2 | 21.7 | 33.3 | 42.7 |

The per-detection coefficient is stable at about +0.06..+0.12. The signal is cumulative evidence that the item is still alive (sold or withdrawn items stop appearing), so its value depends on how much future is visible.
- Private-test feeds see 0-60 h ahead: Sun pm about 2.5 days, Mon about 1.5 days, Tue < 1 day.
- Public feeds see 2.5-4 days.
- Val feeds with a val+test stream see 5-7 days, which is unrealistically much.

**Two variants, both harness-evaluated:**
- *Robust* (within-split; val behaves as the last split): R1 +0.00034 ± 0.00068 (16 features), R5 +0.00020 ± 0.00068 (core 5).
- *Optimistic* (val rows see test as their future): R2 +0.00085 ± 0.00070 (16 features), R7 −0.00049 ± 0.00068 (core 5).

Neither is significant. The robust variant is the only one that could be used, and it was not kept.

**Feature quality vs time left in the split** (`near_end.py`). Val-Thursday feeds see only the rest of Thursday, the same situation as private-test Tuesday. Columns:
- joint LR of the 10 future features over base OOF (df = 10);
- share of rows with any nonzero future signal;
- per-feed XGB-screen delta (2 seeds).

| val bucket | feeds | LR robust | LR optimistic | nonzero, robust | nonzero, optimistic | ΔNDCG robust | ΔNDCG optimistic |
|---|---|---|---|---|---|---|---|
| Tue (future: rest of Tue + Thu) | 7,543 | 28.5 | 29.9 | 18.6% | 35.5% | +0.0004 ± 0.0010 | −0.0009 ± 0.0010 |
| Thu, 16-24 h left | 1,207 | 8.6 | 9.5 | 14.6% | 35.3% | −0.0009 ± 0.0026 | −0.0013 ± 0.0028 |
| Thu, 8-16 h left | 3,325 | 8.2 | 17.9 | 10.2% | 33.4% | −0.0003 ± 0.0016 | −0.0011 ± 0.0016 |
| Thu, 0-8 h left | 2,348 | 14.6 | 8.8 | 4.2% | 32.6% | −0.0019 ± 0.0018 | +0.0008 ± 0.0018 |

In the robust variant, the share of rows carrying any future signal falls from 19% to 4% as the split end approaches. On Thursday the LR is at noise level (the χ²(10) mean is 10). The optimistic variant keeps 33-35% coverage everywhere, but only because it peeks at test.

## 5. Drift check (val vs test-public vs test-private)

- **TS** (state as of TR_END, identical for val and test): PSI ≤ 0.035 for every feature. Coverage falls from 0.71 (val) to 0.63 (public) to 0.60 (private), following the item-coverage decay with time.
- **life_age_any_h**: PSI 0.24 for val vs public and 0.84 for val vs private. Items seen in train are older at test time (val q25 229 h vs private 311 h). This is inherent drift, and one reason the feature should not be used. `life_never_seen` has PSI ≤ 0.025.
- **SS (sparse)**, as nonzero share for val / pub / prv-Sun / prv-Mon / prv-Tue:
  - future `h_kn72`: 4.9 / 13.4 / 10.6 / 7.4 / 2.8%;
  - past `h_kp72`: 4.5 / 3.1 / 6.8 / 9.2 / 11.8%;
  - `e_kn72`: 4.0 / 11.4 / 8.9 / 6.2 / 2.3%.

  The mean nonzero rate moves the opposite way (fewer observed feeds give a larger per-appearance rate), as intended by the rate normalisation. Val sits inside the private range. Public has 2-3× more future coverage, so it is the out-of-range regime.
- **POST / UA / TZ**: PSI ≤ 0.065, except `ua_n` (0.12 val vs private) and `ua_u_gap` (0.11).

## 6. Other things tried (all negative)

- **Staleness.** Within popularity strata, items shown as candidates on more train days have a lower positive rate (e.g. no train history: 1.61% at 1 day vs 0.91% at 8+ days). The base (tte_it_imp, it_cand_cnt_train, f4) already captures this: TS LR ≤ 7, harness R3 −0.0001.
- **Exploration exposure.** A candidate shown again in another val/test feed within 6-24 h has a lower positive rate (0.9-1.2% vs 1.3-1.6%). There is no marginal information: `c_k*` and `c_r*` LR < 3.5.
- **Time-zone matching** via diurnal activity phase (a user and item region proxy): no marginal information.
- **Posting-time calendar** (weekday/weekend posting, hour, bulk uploads): no marginal information. `post_dow` would also be unsafe, because test's fresh items are posted on weekend days, which val never shows.
- **User freshness preference** (users who hunt fresh listings vs browse old stock): strong single-feature asymmetry (`ua_absdiff` nd− 0.301 vs nd+ 0.268), but it is f4 seen through a per-feed shift. No marginal information.

## 7. KEEP list

**KEEP: none.** No subset reaches delta > 2·SE, and the 27-feature POST/UA/TZ group is significantly harmful (−0.0017 ± 0.0007). All `life_*` feature files were deleted.

Kept artifacts:
- `data/feats/life_f4_age_knots.npy`, the f4→age knots in hours. Use `feats_v2/life_/post_time.py::age_hours(f4, knots)`.
- The code, which regenerates every feature. TS: `ts_train_stream.py`, about 8 min for all splits. SS: `ss_sparse_stream.py {val,test|train} split`, seconds to 2 min. POST: `post_time.py`. UA: `uage.py`. TZ: `tz_phase.py`.

Recommendations for other families:
1. Do not build freshness or age features: f4 is the exact age.
2. Future-looking transductive item statistics (including the base's `it_hist_*`, which count val/test histories of all times) are horizon-dependent. Their value on val, which sees up to 7 days ahead, overstates what private feeds get. Where such a feature matters, define it within-split.
3. Seed 2 of the harness carries a systematic positive bias (+0.0029 mean per-seed delta across 44 runs). Judge small deltas on s0/s1 as well.

---

## Verification (adversarial verifier)

Scripts: `/tmp/scratchpad/verify_life/` (`f4check.py`, `regen_ss.py`, `timesplit.py`, `timesplit_rev.py`, `drift_sf.py`, `wkend_age.py`). Harness tags `life_VERIFY_*` are in `reports/feature_evals.jsonl`.

**Verdict: the builder's conclusion stands. KEEP is empty and every life_ feature is dropped.** The builder claimed no gain, and nothing found here changes that. The f4 = age finding holds independently.

### V1. Leakage and asymmetry audit (code)
- **Val labels.** No feature reads `val/cand_target`. Val labels are used only by the analysis scripts (`clogit_screen`, `near_end`, `horizon_check`, `sf_eval`) and the harness. That is selection on the evaluation set, and it is harmless here because nothing was kept.
- **Train labels.** Never used by any life_ feature. TS replays train strictly before each timestamp group, so there is no same-timestamp look-ahead.
- **SS within-split stream.** Val rows use the val stream and test rows use the test stream; the code is identical for both. The own feed and the same non-zero user are excluded.
  - Small hole: `user_id = 0` rows are not excluded from other user-0 feeds. Per `eda_history_leak`, user-0 feeds share almost no history, so this is negligible.
  - Structural asymmetry, not a code bug: the val stream is Tue + Thu with Wed missing, while the test stream is four continuous days. The definition is identical, but the topology is not; see V2.
- **Train-vs-val semantics.** These would matter only if a train-trained model used the features:
  - TS uses τ = feed time for train rows but τ = TR_END for val/test. Every `*_gap_h`, `age_h` and decay feature therefore means something different on train ("gap to now" vs "gap to the end of train").
  - The SS train version uses a 2.8% train subsample with up to 13 days of future, against at most 2-4 days within val/test.
  - `fdev_side.py` (life_x_*) has no train version at all.

  None of this matters now, because nothing is kept. Any revival must fix it first.
- No trivial bugs were found that needed fixing.

### V2. Drift: val / test-public / test-private (boundary ts 1790497920) and test weekday vs weekend
PSI is about 0 for the SS features because they are more than 90% zeros, so it is uninformative. The share of nonzero values and the median nonzero value are what show the drift.

| feature | nonzero/finite val / pub / prv / test-wkday / test-wkend | median nonzero val / pub / prv | PSI val–prv |
|---|---|---|---|
| ss_e_kn72 (future) | 4.0 / 11.4 / 5.4 / 4.2 / 10.5 % | 1.20 / 0.49 / 0.89 | ~0 |
| ss_h_kn72 (future) | 4.9 / 13.4 / 6.5 / 5.1 / 12.4 % | 1.22 / 0.50 / 0.91 | ~0 |
| ss_h_rp72 (past) | 4.5 / 3.1 / 9.2 / 10.0 / 4.5 % | 1.24 / 1.81 / 0.62 | ~0 |
| ss_e_rp72 (past) | 3.7 / 2.6 / 7.7 / 8.4 / 3.7 % | 1.21 / 1.79 / 0.60 | ~0 |
| x_d1_mean / x_d3_mean | 65.4 / 56.4 / 52.7 / 52.1 / 55.7 % | -0.069 / -0.073 / -0.078 | 0.001 |
| x_d5_last | same coverage | | 0.038 |

- **SS features depend on where the feed sits within its split, not on the item.** Future coverage is 2.5× higher on test weekend rows than on weekday rows, and past coverage is 2.2× higher on weekday rows.
- **For the past-looking SS features, val is outside the private range.** Val coverage is 4.5%, against 6.8-11.8% across the private buckets, and the nonzero magnitudes differ by about 2×. The builder's "val sits inside the private range" holds for the future features only.
- **This is a second, independent reason to drop SS**, on top of the null deltas.
- **life_x_* are stable in value.** Coverage falls with train-item coverage (65% → 53%).

### V3. Re-measurement (features regenerated with the builder's own code, then deleted again)
5-seed harness (`--seeds 5`, 16 threads):

| tag | n | delta | SE | per-seed s0..s4 |
|---|---|---|---|---|
| life_VERIFY_core5_s5 (= R5) | 5 | −0.00010 | 0.00052 | +0.0001 / −0.0014 / +0.0019 / −0.0011 / +0.0001 |
| life_VERIFY_ss14_s5 (= R1 minus age_any/never_seen) | 14 | +0.00010 | 0.00053 | −0.0013 / −0.0009 / +0.0026 / +0.0003 / −0.0001 |
| life_VERIFY_fdev9_s5 (= R6a) | 9 | +0.00012 | 0.00056 | −0.0000 / −0.0014 / +0.0026 / +0.0003 / −0.0008 |

- Seeds 0-2 reproduce the builder's R5 almost exactly.
- With seeds 3 and 4 added, every positive 3-seed number shrinks toward 0.

Time-split check (LightGBM with the same PARAMS/ROUNDS; 5 model seeds; paired per-feed delta):

| set | train Tue → eval Thu | train Thu → eval Tue | pooled |
|---|---|---|---|
| core5 (SS) | +0.00079 ± 0.00078 | −0.00029 ± 0.00076 | +0.00024 ± 0.00054 |
| ss14 (SS) | −0.00039 ± 0.00077 | – | – |
| fdev9 (life_x_*) | +0.00127 ± 0.00085 | +0.00112 ± 0.00087 | +0.0012 ± 0.0006 (~2.0 SE) |
| fdev2 (x_d1_mean, x_d3_mean) | – | −0.00026 ± 0.00079 | – |

- **SS is null by every measure.**
- **fdev9 is inconclusive.**
  - It is borderline in the two time splits (both directions positive, pooled about 2 SE).
  - It is null in the 5-fold harness (0.2 SE), which is the protocol of record.
  - Its 2-feature core is negative.
  - It has no train version, and its coverage falls 13pp from val to private.

  It stays dropped here. If anyone revisits "item-centred f deviation" (f_k now minus the item's train mean), it belongs in a separate family, built for train as well with an expanding window, and retested.

### V4. Single-feature NDCG (val, both directions, NaN last, random tie-break)
Recomputed on the regenerated files, they match the builder's table within about 0.0007:
- ss_e_kn72 0.2969 / 0.2891
- ss_h_rp72 0.2966 / 0.2861
- x_d1_mean 0.3153 / 0.2752
- x_d3_mean 0.3123 / 0.2761
- x_m3 0.3007 / 0.2850

The random baseline is 0.2896.

### V5. f4 = exact listing age: independently confirmed
I applied the builder's knots to splits they were never fitted on and compared posting times recovered from val/test appearances with those from train appearances of the same item:
- **Val:** 681k matched rows. |Δt_post| median 0.3 s, q90 6.7 s, q99 52 s. f4 was below the item's train maximum in 0.0% of rows and above it in 100%.
- **Test:** 1.13M rows. Median 0.5 s, q90 7.6 s, q99 52 s. Again 0% decreased.
- **Within train, Sep 6-7 vs Sep 18-19:** 2.0M items, median 0.5 s, and no f4 decreases.
- **Grid:** for young val items, t_post sits on the 128-s grid within 2 s for 100% of them.

The map extrapolates without change to test's later dates.

**Weekend check of the age signal.** Val has no weekend days, and test does, so this is the risk for private.
- On train, the single-feature NDCG lift of −f4 over random is +0.001 to +0.005 on every day, with no weekday/weekend pattern (Sat +0.0011/+0.0024, Sun +0.0049/+0.0033).
- At fixed position (pos < 10), the positive rate by age bucket is the same on weekdays and weekends. Weekday: 7.92 / 6.10 / 5.54 / 5.37%. Weekend: 7.96 / 6.17 / 5.44 / 5.30%.
- **The freshness effect the base learns from f4 on weekday-only val should carry over to test weekends.** There is no gain to be had from a weekend × age interaction.

### V6. Harness caveat: confirmed and explained
The cached base OOF NDCG per seed is s0 0.36416, s1 0.36413, **s2 0.36190**, s3 0.36391, s4 0.36370.
- Seed 2's base is an unlucky draw, about 0.0022 below the others. Any change to the feature set re-draws LightGBM's feature/bagging randomness and regresses toward the mean, which explains the systematic positive s2 delta.
- `cal_null_base_rerun` gives exactly 0, so the cache is not stale.
- With 5 seeds the bias is diluted. Report 5-seed numbers or judge on s0/s1/s3/s4.

### V7. Files
- `data/feats/{train,val,test}/life_*`: none remain. The verifier regenerated life_ss_* and life_x_* for the checks and deleted them again.
- `data/feats/life_f4_age_knots.npy` is kept. It is a lookup table, not a feature file.

**Per-feature verdicts:**

| features | verdict | reason |
|---|---|---|
| life_ss_* (all) | drop | null; split-position artifact drift |
| life_age_any_h | drop | PSI 0.84 val–private |
| life_never_seen | drop | |
| life_ts_* | drop | null (R3); train semantics differ from val/test |
| life_post_*, life_ua_*, life_tz_* | drop | significantly harmful (R4) |
| life_x_* | drop | null in the harness; borderline only in the time split; no train version |
| life_ssF_* / life_F* | drop | optimistic future, not deployable |
