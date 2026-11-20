# Feature family `trd_`: transductive population trends

Code: `<repo>/feats_v2/trd/`. Features: `<repo>/data/feats/{train,val,test}/trd_*.npy` (float32, aligned with `cand_*.npy`).
All NDCG numbers are full-list NDCG on val (14,423 feeds), computed with `src/metrics.py`.
"Official" = `src/eval_family.py` (LightGBM lambdarank, 5-fold GroupKFold by user, fixed rounds, paired per-feed delta).
"Proxy" = `feats_v2/trd/screen_xgb.py`, the same protocol with XGBoost `rank:ndcg` on GPU N. I used the proxy only for fast screening; no conclusion rests on it alone.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **Population trends add nothing reliable on top of the base set.** I built 107 features in 8 groups and evaluated them all:
   - category "now" lifts, seasonal (hour × weekday/weekend) lifts, candidate supply, and the user's over-indexing relative to the population now;
   - item and embedding-level population demand and supply densities;
   - a long-run density and the trend ratio against it;
   - user-neighbour-weighted densities;
   - a population category-transition prior;
   - item-attribute (f1..f5) profiles of engaged items;
   - category × daypart CTR lifts.

   Whole family, official harness, 5 seeds: **−0.0002 ± 0.0006** (per seed −0.0008, −0.0018, +0.0013, −0.0001, +0.0002). No single group passes `delta > 2·SE`.
2. **The best subset is borderline, not significant.** It is the 20 embedding-space population **demand/supply density** features (`kde/n85/n95/max/ex/skde/sn95/dsr`, windows ±3h and same calendar day):
   - official, 3 seeds: +0.0016 ± 0.0007;
   - official, **5 seeds: +0.0009 ± 0.0006** (seeds 3 and 4 ≈ 0);
   - proxy, 3 seeds: +0.0010 ± 0.0007.

   It beats the random-feature control on all 5 seeds (see 3.), but it fails the keep criterion. Compact sub-subsets are null.
3. **Calibration: the harness null is negative.** Adding 4 pure-noise features gives **−0.0010 ± 0.0005** (official, 5 seeds; per seed −0.0014, −0.0014, +0.0003, −0.0017, −0.0007).
   - Fold seed 2 shows a systematic +0.0015..0.002 offset relative to seeds 0, 1, 3 and 4 for almost every family (base OOF seed 2 is weak), so 3-seed results are optimistic.
   - Use 5 seeds and compare against this control.
4. **Window design matters more than the signal.** Never use 24h windows that cross midnight.
   - Val days are isolated (Tue, Thu) while test days are contiguous. A 24h window is therefore truncated in val but complete in private test: PSI val→private up to **0.58** for category lifts, 0.32 for max-similarity.
   - Fix: calendar-aligned windows (same MSK day) or windows of 6h or less, plus Bayesian shrinkage toward long-run shares. These have the same structure in val and test, with PSI < 0.06 for all density features.
   - Whole-day category lifts still differ between public (weekend) and val (PSI 1.45). This comes from their discreteness: one value per category per day, plus real but modest weekend effects of ±8-22%.
5. **Item-level "seen in other users' recent history" is very sparse.**
   - Exact item in another user's hpos<10 history on the same day: **0.90% of val rows / 0.92% of test rows**, pos rate 2.18% vs 1.58%. Within ±3h: 0.27% / 0.28%.
   - Embedding near-duplicates (cos ≥ 0.95) the same day cover 13.2% of rows, pos rate 1.82% vs 1.55%.
   - Too sparse to move NDCG.
6. **KEEP list: empty** under the `delta > 2·SE` rule. The 20 borderline density features are retained on disk for all 3 splits as an **optional** set for a final-stage ablation (§7). Every other `trd_` file was deleted.

---------------------------------------------------------------------------------------------------------------------

## 1. Design (shared by all groups, `trd_lib.py`)

* **"Recent population events".** History is newest-first, so events with small `hpos` of a feed at time t happened shortly before t. Pool events are history events with `hpos < K` (K = 3/10/20/all; event-type groups: all, et1, et2-10). Each event is timestamped with its feed's timestamp.
* **Pool of feeds.**
  - val/test: all feeds of the same split.
  - train: a deterministic per-day hash subsample of train feeds at val/test density (7,200 feeds/day, about 2.7%). Window estimates in train then have the same noise level as in val/test, so the feature is identically defined in distribution. It is also 37× cheaper.
* **Leakage controls.**
  - The own feed is always excluded.
  - Item- and embedding-level features also exclude events of feeds with the same non-zero `user_id`. This blocks the documented cross-feed label leak: a positive reappears in the same user's later history.
  - No val labels are used anywhere. Train labels are used only in `cctr` (expanding window for train rows).
* **Windows.**
  - Past-only `p3h` = [t−3h, t]; symmetric `s6h` = [t−3h, t+3h].
  - Calendar-aligned `pday` = MSK day start → t; `sday` = whole MSK day.
  - The initial 24h versions (`p24h`, `s24h`) were dropped after the drift check (TL;DR 4).
  - Test's later days have the same window structure as val days, so the features do not degrade along the test period.
* **Shrinkage.** Shares are computed as (n_c + A·p_c)/(N + A) with A = 300 events (3000 for supply), so truncated windows (day starts, the documented hour-long outages) shrink toward the long-run share instead of becoming noisy.
* **Long-run baselines.**
  - val/test: a pooled val+test share, a constant vector for the exploration population.
  - train: the train pool's own share.

## 2. Feature definitions

| group (files) | definition |
|---|---|
| `cl_<g>_<w>` (20) | log(shrunk share of the candidate's category among pool events of group g in window w / long-run share). g ∈ {k3, k10, kall, k10e1, k10act} |
| `ctod_<g>` (5) | log(train share of category at (MSK hour, weekday/weekend) / long-run share): seasonal expectation from dense train histories |
| `cres_<g>_<w>` (8) | log(now-share / seasonal expectation): today's anomaly |
| `sup_<w>` (4) | log(category share among candidates of other feeds in the window / long-run val+test candidate share) |
| `urel_k10_<w>`, `urel_k10_static` (5) | log((O+.5)/(E+.5)): O = user's last-20 events in category c, E = 20 × population now-share. User over-indexing relative to the population |
| `cctr_tod`, `cctr_h` (2) | log CTR lift of (category, 3h daypart, weekday/weekend) or (category, daypart) vs the category CTR, from train labels (k = 2000 smoothing; expanding window for train rows) |
| `ex/n95/n85/kde/kde_e1/n95_e1/max_<w>` (14) | embedding-space population demand around the candidate. Over pool events (hpos<10) of other users in window w: exact item matches; # with cos ≥ 0.95 / 0.85; Σ exp(20(cos−1)) (all / et1); max cos. Counts are per 1K window pool feeds |
| `msc/mg_<w>` (4) | mean cos to window events of the same category / of all categories |
| `skde/sn95_<w>`, `dsr_<w>` (6) | supply: the same kernel over candidates of other pool feeds in the window (1/4 hash subsample, rescaled). `dsr` = log((kde+0.05)/(skde+0.5)), demand/supply ratio |
| `lr_kde/n90/max`, `lr_tr_<w>` (5) | long-run density: the same kernel over a fixed sample of 200K train hpos<10 events. `lr_tr` = log(window density / long-run density), the trend |
| `nb_{k,c,m,kr}_<w>` (8) | user-neighbour-weighted density: events weighted by exp(10(cos(u, u_e)−1)) in user-embedding space; weighted kernel / weighted mean cos / max cos over the top-50 most similar users / personalisation lift. Val only (dropped) |
| `tp_{l1,m5,m20,x5,mx20}` (5) | population category-transition prior: PMI_L(a→b) from all train histories (event at hpos j+L of category a, then an et1 event of category b at hpos j). Score of candidate category c against the user's last 1/5/20 event categories at lag j+1; `x5` / `mx20` = cross-category only |
| `fp_*` (21) | item-attribute profiles. f4/f5 are near-constant per item (within-item sd 0.011/0.022 vs 0.27/0.31), so A_k(item) = mean f_k over all candidate occurrences, attached to history items (coverage of hpos<20 events: 71% val / 65% test). `pz` = z of the candidate's f_k vs the population's engaged items in the category; `ud/uad/ua` = signed / scaled-absolute distance to the user's own same-category (or all-category) profile; `un` = support |

## 3. Harness results

### 3.1 Official (`src/eval_family.py`)

| set | n | seeds | delta | SE | per-seed deltas |
|---|---|---|---|---|---|
| **whole family** (all 107 incl. the val-only `nb`) | 107 | 5 | **−0.0002** | **0.0006** | −0.0008, −0.0018, +0.0013, −0.0001, +0.0002 |
| category windows (`cl`, `cres`, `ctod`, `sup`, `urel`) | 42 | 3 | −0.0001 | 0.0007 | −0.0015, −0.0011, +0.0023 |
| attribute profiles + transitions (`fp`, `tp`) | 26 | 3 | −0.0012 | 0.0007 | −0.0027, −0.0014, +0.0005 |
| embedding densities, all (`kde`…`dsr`, `msc`, `mg`) | 24 | 3 | +0.0007 | 0.0007 | +0.0009, −0.0011, +0.0022 |
| **demand + supply densities** (20) | 20 | 3 | +0.0016 | 0.0007 | +0.0011, +0.0002, +0.0036 |
| **demand + supply densities** (20) | 20 | **5** | **+0.0009** | **0.0006** | +0.0011, +0.0002, +0.0036, −0.0001, −0.0002 |
| CONTROL: 4 uniform-random features | 4 | 3 | −0.0008 | 0.0007 | −0.0014, −0.0014, +0.0003 |
| **CONTROL: 4 uniform-random features** | 4 | **5** | **−0.0010** | **0.0005** | −0.0014, −0.0014, +0.0003, −0.0017, −0.0007 |

Base NDCG is 0.3634 (3 seeds) / 0.3636 (5 seeds). Paired against the control, demand+supply is +0.0025 / +0.0016 / +0.0033 / +0.0016 / +0.0005 per seed. That is a consistent edge over noise features, but the delta against the base itself is only 1.6 SE.

### 3.2 GPU proxy (XGBoost rank:ndcg; base 0.3638-0.3645; noise control −0.0007 ± 0.0009)

| set | n | seeds | delta ± SE |
|---|---|---|---|
| transition prior `tp` | 5 | 1 | −0.0011 ± 0.0012 |
| neighbour-weighted density `nb` | 8 | 1 | −0.0010 ± 0.0012 |
| time-of-day lifts (`cctr_tod`, `cctr_h`, `ctod_k10`, `ctod_k10e1`) | 4 | 1 | −0.0010 ± 0.0012 |
| long-run density + trend `lr` | 5 | 6 | −0.0005 ± 0.0005 |
| embedding densities, all 24 | 24 | seeds 0-2 / 3-5 | +0.0023 ± 0.0007 / **+0.0003 ± 0.0007** |
| demand + supply (20) | 20 | 3 | +0.0010 ± 0.0007 |
| demand only (14) / supply only (6) / mean-sim only (4) | | 3 | +0.0009 / +0.0004 / −0.0007 (± 0.0007) |
| demand+supply, `sday` only (9) / `s6h` only (9) / core 6 (`kde`, `skde`, `dsr`) | | 6 | +0.0003 / −0.0003 / −0.0009 (± 0.0005) |

The density signal is diffuse: no compact subset keeps it, and an independent replication on seeds 3-5 shrank +0.0023 to +0.0003.

## 4. Single-feature NDCG (val; random tie-break, NaN to the bottom; random order = 0.289)

Category-constant features have an **inherent penalty** under this metric. A random score that is constant within a category averages **0.281** (range 0.273-0.284 over 5 draws) vs 0.289 for a fully random score. Positives cluster in a feed's big categories, and block orderings concentrate positions, which the convex discount penalises. Compare category-level features against 0.281, not 0.289. "blend" = Δ when the within-feed percentile rank is added with weight 0.3 (sign shown) to the reference pct(maxsim32)+pct(cos_u)+pct(f1) = 0.3534. SE is about 0.0005.

| group | best single NDCG (direction) | blend Δ | comment |
|---|---|---|---|
| `cl_*` (20) | 0.2843 (`cl_k10act_pday`, asc) | ≤ +0.0001 | at the category-random level |
| `cres_*` (8) | 0.2842 (`cres_k10_pday`, asc) | ≤ +0.0001 | |
| `ctod_*` (5) | 0.2853 (`ctod_k3`, desc) | ≤ +0.0005 | |
| `cctr_*` (2) | 0.2816 (`cctr_h`, asc) | ≤ +0.0002 | |
| `sup_*` (4) | 0.2835 (`sup_p3h`, desc) | ≤ +0.0004 | |
| `urel_*` (5) | 0.3168 (asc) | −0.0003 | same level as the base `cat_cnt_20` |
| `tp_*` (5) | 0.3168 (`tp_m5`, asc) | +0.0002 | |
| `fp_*` (21) | 0.3362 (`fp_ua1`, about raw f1); 0.3136 (`fp_uad5`, desc) | +0.0025 (`pz5`) = raw f5 (+0.0024) | the profile adds nothing beyond raw f |
| `kde/kde_e1/n85_sday` | 0.2996 / 0.3002 / 0.3001 (asc) | ≤ +0.0001 | |
| `max/ex/n95_sday` | 0.2977 / 0.2915 / 0.2936 | < 0 | |
| `skde_sday`, `dsr_sday` | 0.2944, 0.2972 (asc) | +0.0002 | |
| `msc/mg` | 0.2873 / 0.2870 | ≤ +0.0004 | |
| `lr_kde`, `lr_tr_sday` | 0.2997, 0.2869 | +0.0006, −0.0008 | |
| `nb_kr_sday`, `nb_k_sday` | 0.3174, 0.3119 (asc) | +0.0006 | redundant with the user's own similarity features |

The full per-feature table (both directions, NaN share, blend Δ for both signs) is reproducible with `python feats_v2/trd/screen.py --prefix trd_`.

## 5. Drift (PSI with 10 val-quantile bins; max over the features of each group)

| group | val→public | val→private | public→private |
|---|---|---|---|
| `cl` @p3h / @s6h | 0.039 / 0.052 | 0.024 / 0.021 | 0.055 / 0.042 |
| `cl` @pday / **@sday** | 0.129 / **1.446** | 0.094 / **0.766** | 0.136 / 0.556 |
| `cl` @p24h / @s24h (dropped first version) | 0.16 / 0.21 | **0.58** / 0.14 | 0.31 / 0.17 |
| `cres` (all windows) | ≤ 0.133 | ≤ 0.053 | ≤ 0.089 |
| `ctod`, `cctr` (hour-mix composition, expected) | ≤ 0.175 | ≤ 0.091 | ≤ 0.153 |
| `sup` @sday | 0.377 | 0.124 | 0.267 |
| `urel`, `tp`, `fp`, `lr`, `kde`, `kde_e1`, `msc`, `mg`, `ex` | ≤ 0.006 | ≤ 0.005 | ≤ 0.012 |
| `max`, `n85`, `n95`, `n95_e1`, `skde`, `sn95`, `dsr` (day / 6h windows) | ≤ 0.055 | ≤ 0.058 | ≤ 0.048 |
| `max` @p24h (dropped first version) | 0.005 | **0.317** | 0.237 |

* The density features in the borderline set are all PSI < 0.06 (val→public, val→private).
* NaN share is identical across splits (0.02% = missing candidate embedding).
* Whole-day category lifts take one value per category per day, so their PSI is dominated by day identity. The public period is Sat+Sun. Weekend/weekday category ratios are 0.92-1.22 in test and only weakly consistent with train (correlation 0.39 over 33 categories). For example, cat 27: ×1.22 on the test weekend vs ×1.00 in train; cat 48: ×0.92 vs ×0.87.

## 6. Other findings worth reusing

* **f4/f5 are item attributes.** Within-item sd is 0.011 / 0.022 vs 0.27 / 0.31 overall; f1-f3 vary per impression (within-item sd about 0.18). Raw f5 has a positive blend value (+0.0024 on top of maxsim+cos_u+f1). A user-profile distance on f5 (`fp_uad5`, single 0.314) adds nothing in LightGBM, which already has f5.
* **Training throughput on this machine.** Under load, one official 5-seed run on a family of about 20 features takes 10-50 min. The proxy takes 2-5 min for 3 seeds.

## 7. KEEP list and retained files

**KEEP: none.** No `trd_` group passes `delta > 2·SE` in the official 5-seed harness.

**Optional, borderline (retained on disk for train/val/test, NOT in KEEP).** These 20 features are +0.0009 ± 0.0006 at 5 seeds, beat the noise control on 5/5 seeds, and have PSI < 0.06. Try them only in a final-stage ablation of the full model:
`trd_kde_s6h, trd_kde_sday, trd_kde_e1_s6h, trd_kde_e1_sday, trd_n85_s6h, trd_n85_sday, trd_n95_s6h, trd_n95_sday, trd_n95_e1_s6h, trd_n95_e1_sday, trd_max_s6h, trd_max_sday, trd_ex_s6h, trd_ex_sday, trd_skde_s6h, trd_skde_sday, trd_sn95_s6h, trd_sn95_sday, trd_dsr_s6h, trd_dsr_sday`.
To delete them: `rm <repo>/data/feats/{train,val,test}/trd_{kde,kde_e1,n85,n95,n95_e1,max,ex,skde,sn95,dsr}_{s6h,sday}.npy`.
The train versions use the val/test-density train pool (§1). Verified on a 3M-row sample: 246,745,836 rows, float32, NaN 0.004%. PSI val→train is ≤ 0.03 for the demand features (`kde`, `n85`, `n95`, `max`, `ex`) and ≤ 0.06 for `skde`/`sn95`. `dsr` shifts more (0.19 / 0.33): train candidates are production-ranked, so the supply side differs from exploration feeds. Treat `dsr` with care if a model is trained on train rows.

All other `trd_` feature files (category, seasonal, CTR, profile, transition, long-run, neighbour, mean-sim, random controls) were deleted from all splits.

## 8. Code (`<repo>/feats_v2/trd/`)

| file | what |
|---|---|
| `trd_lib.py` | pool definition (train subsample at val/test density), window helpers (incl. calendar-day), per-feed category counters (numba), single-feature NDCG, PSI |
| `build_cat.py` | `cl`, `ctod`, `cres`, `sup`, `urel` (all splits; seconds) |
| `build_emb.py` | GPU embedding demand/supply densities (`CUDA_VISIBLE_DEVICES=N`; val 33 s on a free GPU, train about 60-90 min depending on GPU sharing) |
| `build_lr.py`, `build_nb.py` | long-run density + trend; user-neighbour-weighted density (GPU) |
| `build_tp.py`, `build_fp.py`, `build_ctr_tod.py` | transition PMI prior; item-attribute profiles; category × daypart CTR lift |
| `screen.py` | single-feature NDCG + rank-blend screen |
| `screen_xgb.py` | GPU proxy of the harness (screening only) |
| `drift.py` | PSI / quantiles val vs test-public vs test-private |

---------------------------------------------------------------------------------------------------------------------

## Verification (adversarial verifier, 2026-11-11)

Scope: the 20 retained density features (`trd_{kde,kde_e1,n85,n95,n95_e1,max,ex,skde,sn95,dsr}_{s6h,sday}`). Every other `trd_` file is already deleted: exactly these 20 names exist in train, val and test.
Scripts: `<repo>/feats_v2/trd/verify/` (`sanity.py`, `leak.py`, `tsplit.py`, `sem.py`, `perday.py`; raw numbers in `sanity_results.json`).
Note: the code lives in `feats_v2/trd/`, not `feats_v2/trd_/`.

### V.1 Verdict

**The builder's conclusion holds: KEEP stays empty. I could not reproduce any reliable gain, and I found no leak.** I also recommend not using the 20 "optional" features in the final ablation. The time-split check below shows they do no better than random noise features.

### V.2 Leakage audit (`build_emb.py`, `trd_lib.py`)

* **Val labels:** not read anywhere in the retained code. **Train labels:** none of the 20 retained features use labels.
* **Val vs test symmetry:** the same `build()` code runs on both. The pool is all feeds of the same split, the own feed is excluded, and other feeds of the same non-zero user are excluded. `sday` = the whole MSK calendar day, the same structure in val and test. No test data is used for val, and no val data for test.
* **Cross-feed label leak:** I tested it directly (`leak.py`). Take val candidates whose exact item appears in another feed's hpos<10 history on the same day, using the builder's exclusion rules:
  - matches from **future** feeds only: 6,232 rows, pos rate **2.13%**;
  - matches from **past** feeds only: 6,000 rows, pos rate **2.17%**;
  - base rate 1.58%.

  A label leak would push the future-only rate toward the ~85% seen for the same-user leak. User-0 rows with a match: 440 rows, 3.2% (14 positives), which is noise. **No leak.**
* **Remaining asymmetry (a design issue, not a leak):** `s6h` windows near midnight are truncated in val, because the Tue and Thu days are isolated. In test they are complete except on Sat 00-03 and Tue 21-24.
  - Rows in truncated windows: **19.4% of val vs 6.2% of test**.
  - Count features are normalised per pool feed in the window, so their edge PSI is ≤ 0.01.
  - `max_s6h` is not normalised. At the day edges its median is 0.807 in val vs 0.833 in test (edge-row PSI **0.13**). The overall PSI is still small (0.011).
  - Fix if ever needed: intersect `s6h` with the calendar day. I did not apply it: it needs a 60-90 min GPU rebuild of train for a family that is null anyway.
* Three val feeds stamped 00:00:48 on Sep 23/25 form one-feed `sday` pools (counts 0, `max` NaN). This is negligible.

### V.3 File sanity (train / val / test)

* All 60 files are float32 with the correct length (train 246,745,836; val 1,389,971; test 2,904,806).
* NaN share is 0.00% / 0.015% / 0.016% (missing candidate embedding). `max_sday` val NaN is 0.029% (the three one-feed days).
* No file is constant. The sparse features have these nonzero shares (val / test / train):

  | feature | val | test | train |
  |---|---|---|---|
  | `ex_s6h` | 0.3% | 0.3% | 0.6% |
  | `ex_sday` | 0.9% | 0.9% | **1.9%** |
  | `n95_s6h` | 4.7% | 5.0% | 6.0% |
  | `n95_sday` | 13.2% | 13.5% | 16.0% |
  | `sn95_sday` | 30.7% | 31.8% | 31.5% |

  `ex` fires twice as often on train, because production traffic repeats items.
* **Semantics, train vs val (lift of pos rate by val-quintile bins):** the direction and monotone shape match for all 8 features checked. Examples, val vs train:

  | feature | val lift, bottom → top quintile | train lift, bottom → top quintile |
  |---|---|---|
  | `kde_sday` | 0.87 → 1.23 | 0.89 → 1.12 |
  | `max_sday` | 0.86 → 1.13 | 0.89 → 1.10 |
  | `dsr_sday` | 0.85 → 1.19 | 0.82 → 1.13 |

  The `dsr` distribution is shifted on train: 39% of train rows fall in the top val quintile. PSI val→train is 0.19 / 0.33 for `dsr`, as the builder reported.

### V.4 Drift (PSI, 10 val-quantile bins; public/private boundary ts 1790497920)

| feature | v→pub | v→priv | pub→priv | test weekday→weekend | v→train |
|---|---|---|---|---|---|
| `kde`, `kde_e1` (both windows) | ≤ 0.001 | ≤ 0.001 | ≤ 0.001 | ≤ 0.001 | ≤ 0.007 |
| `max_s6h` / `max_sday` | 0.006 / 0.003 | 0.011 / 0.001 | 0.033 / 0.007 | 0.004 / 0.003 | 0.010 / 0.009 |
| `n85_s6h` / `n85_sday` | 0.011 / 0.009 | 0.001 / 0.014 | 0.004 / 0.023 | 0.009 / 0.012 | 0.007 / 0.030 |
| `n95*`, `n95_e1*`, `ex*` | ≤ 0.001 | ≤ 0.028 | ≤ 0.020 | ≤ 0.005 | ≤ 0.006 |
| `skde_s6h` / `skde_sday` | 0.010 / 0.009 | 0.006 / 0.004 | 0.002 / 0.002 | 0.004 / 0.004 | 0.015 / 0.016 |
| `sn95_s6h` / **`sn95_sday`** | 0.008 / 0.023 | 0.006 / 0.058 | 0.010 / 0.047 | 0.002 / **0.518** | 0.016 / 0.062 |
| `dsr_s6h` / `dsr_sday` | 0.033 / 0.055 | 0.012 / 0.020 | 0.008 / 0.009 | 0.013 / 0.022 | **0.19 / 0.33** |

* `sn95_sday` has a large weekday→weekend PSI inside test. The cause is that it is discretised on a per-day grid (k × 4000 / daily feed count, and Saturday has only 6,050 feeds), so a tree can split on day identity.
* Everything else is stable across val, public and private. The PSI of the `ex*`/`n95*` features is 0 by construction (their quantile bins collapse at 0), so the nonzero shares in V.3 are the better check for those.

### V.5 Harness re-run (official `src/eval_family.py --seeds 5 --threads 16`, tag `trd_VERIFY_dem_sup_s5`)

| run | delta | SE | per seed |
|---|---|---|---|
| builder (reported) | +0.0009 | 0.0006 | +0.0011, +0.0002, +0.0036, −0.0001, −0.0002 |
| **verifier re-run** | **+0.00065** | **0.00056** | +0.0011, +0.0002, +0.0023, −0.0001, −0.0002 |

* The re-run gives 1.2 SE, which **fails** the 2·SE rule.
* Seed 2 moved from +0.0036 to +0.0023 between identical runs. LightGBM is not bit-exact under multi-threading on a loaded machine, so the seed-2 outlier is partly run noise.

### V.6 Time-split check (LightGBM, `eval_family.PARAMS`, 350 rounds; train on one val day, score the other; 5 LightGBM seeds; paired per-feed SE)

| direction | +20 trd features | +4 uniform-random features (control) |
|---|---|---|
| train Tue (7,543 feeds) → eval Thu (6,880) | +0.00078 ± 0.00086 | +0.00038 ± 0.00079 |
| train Thu → eval Tue | +0.00071 ± 0.00081 | +0.00096 ± 0.00080 |

**The family is not distinguishable from pure noise features in either direction.** The "beats the random control on 5/5 seeds" argument from the GroupKFold harness does not carry over to a temporal split.

### V.7 Single-feature NDCG (val, 3 random tie-breaks; "asc" = higher value ranked first; random ≈ 0.290)

| feature | asc Tue | asc Thu | asc all | desc all |
|---|---|---|---|---|
| `kde_e1_sday` (best) | 0.2986 | 0.3020 | 0.3002 | 0.2769 |
| `kde_sday` | 0.2982 | 0.3011 | 0.2996 | 0.2771 |
| `n85_sday` | 0.2994 | 0.3006 | 0.2999 | 0.2794 |
| `max_sday` | 0.2964 | 0.2991 | 0.2977 | 0.2767 |
| `dsr_sday` | 0.2971 | 0.2974 | 0.2973 | 0.2738 |
| `ex_sday` | 0.2938 | 0.2921 | 0.2930 | 0.2888 |
| `skde_sday` | 0.2936 | 0.2952 | 0.2944 | 0.2802 |
| `sn95_*` | ≈ 0.288-0.289 | | | (null) |

* The signal is weak but stable across the two days. It is far below the base features (maxsim32 0.349).
* It is mostly redundant with the user-item similarity already in the base: higher population density means a more mainstream item.

### V.8 Per-feature verdict (all DROP from KEEP)

| feature(s) | verdict | reason |
|---|---|---|
| `kde_s6h`, `kde_sday`, `kde_e1_s6h`, `kde_e1_sday` | DROP | clean, no drift, no leak. Weak (single NDCG 0.297-0.300); no measurable gain as a group |
| `n85_s6h`, `n85_sday` | DROP | clean; same as kde |
| `max_sday` | DROP | clean; null |
| `max_s6h` | DROP | null; day-edge truncation asymmetry (val 19% vs test 6% of rows truncated; edge PSI 0.13) |
| `n95_s6h`, `n95_sday`, `n95_e1_s6h`, `n95_e1_sday` | DROP | sparse (4-13% nonzero); single NDCG at most 0.295 |
| `ex_s6h`, `ex_sday` | DROP | very sparse (0.3% / 0.9%); 2× denser on train; no leak (future ≈ past pos rate) |
| `skde_s6h`, `skde_sday` | DROP | clean; null |
| `sn95_s6h`, `sn95_sday` | DROP | single NDCG at random level; `sn95_sday` weekday/weekend PSI 0.52 (per-day discretisation) |
| `dsr_s6h`, `dsr_sday` | DROP | null; PSI val→train 0.19 / 0.33 |

* **No bugs fixed.** Nothing trivial was found; the `s6h` day-edge asymmetry is a design choice, and fixing it would not change the null result.
* The files are left on disk as the builder retained them. Remove them with: `rm <repo>/data/feats/{train,val,test}/trd_{kde,kde_e1,n85,n95,n95_e1,max,ex,skde,sn95,dsr}_{s6h,sday}.npy`
