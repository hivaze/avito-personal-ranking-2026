# Feature review: external research and what is missing or done badly

Author: featrev research agent. Date 2026-11-11 (~15:00-15:40 UTC). CPU only (`CUDA_VISIBLE_DEVICES=""`, 16 threads).
I modified no live file. New code is in `feat_new/research_expo.py`. New arrays are in `data/feats_new/{train,val,test}/` (all float32, aligned with `cand_*.npy`).
Scratch scripts and logs are in `/tmp/scratchpad/featrev/research/` (`c1`…`c7`, `*.log`).

All NDCG numbers use `src/metrics.py`. "Single NDCG" means the feature alone as the score, random tie-break, 3 draws.
"Nested blend" means: score = z(best mix) + w·z(x), with w picked from a 7-value grid on 4/5 of the users and applied to the other 1/5, averaged over 2 seeds.
"Best mix" is 0.5 z(nn_final4_val_oof) + 0.5 z(nn_final4_trainonly_val) + 0.1 z(pre_final_val), which reproduces val **0.39399**.

## 0. TL;DR

1. **Most of what won comparable competitions is already here, or was measured to be null.** Section 4 maps each technique to our status. Covered techniques:
   - OTTO-style co-visitation (causal for train rows);
   - recency- and event-type-weighted history similarity;
   - leave-user-out popularity;
   - train-label target encodings with position debiasing;
   - within-impression normalisation (FeatureSalad / trivago style);
   - BlackPearl item cross-feed context;
   - item age (exact, through f4);
   - lifecycle and trend, calendar, embedding clusters, pretrained sequence-encoder scores;
   - re-exposure / "seen but not clicked".

   I re-checked three impression-type ideas on our data, and all three are dead:
   - user × item re-exposure;
   - re-candidates from the same user's earlier histories: 0.048% of val rows, 24 positives;
   - per-split embedding changes as a listing-edit signal: nested blend −0.00034 ± 0.00028.
2. **RESEARCH-1 (verified bug, live recipe).** In the `cvnd_prew` NN members (v12_allprew_s83/84/85), the in-model within-feed rank and z of `prew_fe` / `prew_new` are computed on the attach_xt-transformed column.
   - That column is clipped at ±10, but raw `prew_fe` has median 15.3, so 86% / 79% of warm rows saturate at +10.
   - Single NDCG of that rank is 0.3086 / 0.3132, against 0.3494 / 0.3506 for the rank of the raw value.
   - The damage is mostly masked, because the separate `prew_fe_z` / `prew_new_z` inputs carry the same ordering (0.3501 / 0.3508).
   - **Fix:** clip-safe copies `prew_fe_c` / `prew_new_c` are built. A rank of their clipped values gives 0.3494 / 0.3507.
   - **Do not patch `attach_xt` in place.** Existing checkpoints were trained on the clipped inputs.
3. **RESEARCH-2 (mechanism verified): the causal co-visitation inputs have a train/test censoring mismatch.** A listing created after the source cutoff cannot have co-visitation evidence ("censored"). The censored share of rows is:

   | rows | censored share |
   |---|---|
   | train rows (cutoff = start of their own day) | 7.8-8.6% |
   | val | 20.1% |
   | public | 27.9% |
   | private | **31.3%** |

   - Among zero-co-vis rows, the positive-rate lift of censored rows is 0.43 on train vs 0.67 on val. On train, a censored zero means "brand new and unseen"; on val/test it means "posted in the last 0-10 days".
   - The train-only NN therefore under-rates censored zeros on val: actual/expected positives **1.055**, and **1.103** for zeros posted ≤ 3 days before the cutoff, vs 1.000 for long-exposed zeros. The fine-tune repairs this (0.967 / 1.036).
   - Moving the train-row cutoff back by L = 4 / 8 days gives censored shares of 24.2% / 32.5%, which almost exactly matches val (20.1%) and private (31.3%).
   - This is the forecasting rule "lags ≥ horizon" (M5 / Favorita per-horizon features), applied to our 7-10-day staleness. It is feature idea **F1**: horizon-aligned train-row co-vis plus an explicit exposure-time input. It complements the review's stale-val gate and its density-matched popularity idea (review_ideas #1 / #5).
4. **RESEARCH-3 (new, low): history-embedding missingness carries post-feed information about listing closures, and its horizon differs by split.**
   - At a fixed history position, the missing rate falls steadily over the feed's day within each split. Train hpos 50-149: 15.6% (Sep 6) → 11.6% (Sep 19). Val Tue → Thu: 12.3% → 11.5%.
   - So the train table masks old history items that closed up to ~13 days after the feed, while val/test tables mask only 0-3 days of closures.
   - No label leak was found, but the NN's history mask means slightly different things on train and on test. An inference-only sensitivity test is proposed.
5. **The scaling of the other NN inputs is fine.** The `ctx_hs_*` / `ctx_hn_soft05` mass features hit the ±10 clip on 0.2-1.2% of rows. Their positive-rate lift above the clip (2.3-3.2) equals the lift in the 8-10 band, so nothing is lost. Count features get log1p. Fine-tune extras are standardised with val stats that are reused for test (`nn/finetune.py:166-167`).
6. **Expected value of what is left is small.** The new ideas F1-F4 are calibration and parity fixes worth about +0.0003 to +0.0015 on private. None is a new signal family. That agrees with err_SYNTHESIS: the headroom lies in within-category item discrimination, which hand features have not reached.

---------------------------------------------------------------------------------------------------------------------

## 1. Findings (most severe first)

### RESEARCH-1 (medium; verified bug in a live train-stage recipe): `prew_fe` / `prew_new` in-model rank and z are computed on a saturated column

**Evidence**
- **Where the rank and z come from.**
  - `nn/train.py:48-51`: `cvnd_prew` puts `prew_fe`, `prew_new` and `prew_hmax32` into `XT_REL` and `XT_RELONLY`. They enter only through the in-model within-feed pct rank and z.
  - `nn/model.py:123-127` computes `_pctrank(xt[..., j])` / `_zscore(xt[..., j])` on `b["xt"]`.
  - `b["xt"]` is the output of `nn/data.py:182-200` (`attach_xt`): names that are not `_z`, `_rk`, `seq_cv*` or `it_hist*` get `np.clip(x, -10, 10)`, then NaN → 0, then fp16.
- **Live use.** `nn/logs/v12_allprew_s83.log` shows `'xt': 'cvnd_prew'` and `'xt_drop': 0.0`, with the 34 train-stage features listed. These are the nn_final5 candidate members S1/S3 (nn_report).
- **Raw scale** (`c1_scaling.log`):
  - prew_fe: median 15.3, q99 22.9, max 27.6 on val;
  - prew_new: median 13.8.
  - Clipped share of all val rows: 53.4% / 48.8%. That is 86.1% / 78.7% of warm rows (`c2_prew_clip.log`). Train is the same: 86.6% / 79.3% of warm rows.
- **Information loss** (`c2_prew_clip.log`):

  | input | val single NDCG | val warm-only NDCG | random-like train (Sep 17-19) |
  |---|---|---|---|
  | rank of raw prew_fe (NaN → 0) | 0.3494 | 0.5394 | 0.2756 |
  | **rank of clip10(prew_fe), what the NN gets** | **0.3086** | 0.4966 | 0.2356 |
  | prew_fe_z (separate input) | 0.3501 | | 0.2758 |
  | rank of raw prew_new | 0.3506 | 0.5406 | 0.2783 |
  | **rank of clip10(prew_new)** | **0.3132** | 0.5017 | 0.2382 |

**Impact**
- Not measured. Probably small: `prew_fe_z` / `prew_new_z` (offline z over warm candidates) carry the ordering, and xt_drop = 0, so the in-model rank brings no blanking benefit either.
- The intended "rank of prew_fe" input, however, is close to a 3-level code: NaN, below 10, or saturated.
- The paired prew acceptance (+0.0011 ± 0.0007 val) was measured with this bug in place.

**Fix**
- Built: `data/feats_new/{split}/prew_fe_c.npy` = (prew_fe − 15)/2 and `prew_new_c.npy` = (prew_new − 14)/2, with NaN → −10. The −10 keeps cold rows at the bottom, as NaN → 0 does today.
- Zero warm rows hit the clip. Rank NDCG of the clipped values is 0.3494 / 0.3507 (`c7_verify_new.log`).
- **For new members only:** a preset `cvnd_prewc` that uses `prew_fe_c` / `prew_new_c` in `XT_REL` / `XT_RELONLY`, keeping `has:prew_fe`.
- Do not change the `attach_xt` transform in place: it would silently change inference of every existing `cvnd_prew` checkpoint.
- Acceptance: the usual paired train-only val run.

### RESEARCH-2 (medium; mechanism verified, fix impact not measured): train/val/test censoring mismatch of the causal co-visitation inputs

**Construction facts**
- `seq_cv_*` for train rows of day d uses train sources strictly before day d (`feats_v2/seq_/sources.py:1-8`, `feed_day_index`).
- For val/test rows it uses all of train plus earlier val/test histories, so the cutoff is ~2026-09-20 00:00 MSK (`feats_v2/seq_/feat_covis.py:187-196`).
- The listing creation time is exact: t_post = t_feed − age(f4), using the life_ knots (feat_life_ §1).

**Evidence** (`c5_exposure_time.log`; train = 30K random-like feeds from Sep 9 on, 4.6M rows)

| rows | censored share (t_post ≥ cutoff) | censored share among seq_cv_pop = 0 | lift of (cv = 0 & censored) | lift of (cv = 0 & exposed > 3 d) | lift of cv > 0 |
|---|---|---|---|---|---|
| train, cutoff = own day start | **7.8%** (8.6% over all train rows, `research_expo.py`) | 19.6% | **0.43** | 0.99 | 1.12 |
| train, cutoff = d − 4 days | 24.2% | | | | |
| train, cutoff = d − 8 days | 32.5% | | | | |
| val | **20.1%** | 47.2% | **0.67** | 0.99 | 1.12 |
| test public | 27.9% | 58.1% | | | |
| test private | **31.3%** | 61.9% | | | |

**Calibration of the live models on val** (`c7_verify_new.log`): actual / expected positives, expected = n_pos · softmax(s / T), with a global T fitted by likelihood (T = 1).

| group | rows | nn_final4 train-only | nn_final4 fine-tuned OOF |
|---|---|---|---|
| cv = 0 & censored | 19.8% | **1.055** | 0.967 |
| cv = 0 & exposed 0-72 h | 4.9% | **1.103** | 1.036 |
| cv = 0 & exposed > 72 h | 17.3% | 1.000 | 1.009 |
| cv > 0 | 58.0% | 0.983 | 1.002 |

- The SE of each ratio is roughly 1/√(positives in the group): about 0.018 for row 1 and 0.032 for row 2 (feed clustering ignored). Both are about 3 SE away from 1.
- This is the same story as review_ideas §1.4, which split by age × LUO popularity, here shown on the causal co-vis input and with the cutoff made explicit.
- The post-hoc blend cannot use it: the nested blend of `src_censored` gives +0.00028 ± 0.00041. It is a train-stage lever.

**Reading.** On train, a censored zero is rare and means the listing was created that same day. On private, a third of all rows are censored, and it means the listing was created 0-10 days ago. The train stage learns "zero → negative" from a population where zeros are rarer and more negative. Lagging the train-row cutoff by the test horizon reproduces the test censoring distribution almost exactly.

**Fix:** feature ideas F1 + F2 (§5).
- This complements, and does not duplicate, the following review items:
  - review_ideas #1 (stale-val, an evaluation gate);
  - #5 (density-matched popularity counts);
  - #6 (age-conditioned ID dropout);
  - SHIFT-1 (it_hist_luo is non-causal on train rows).
- The co-vis input is the one causal popularity-type input. It is where a lag can be added cleanly without touching the transductive LUO counts, which the leakage review showed must stay transductive (−0.0018 ± 0.0008 when made causal).

### RESEARCH-3 (low; pattern verified, mechanism inferred): history-embedding missingness encodes post-feed listing closures, and the horizon differs by split

**Evidence** (`c6_missing_emb_time.log`): missing rate of `hist_emb_idx` (= −1) by feed day, at fixed hpos.

| hpos bucket | train Sep 6 → Sep 19 (monotone over 12 days) | val Tue → Thu | test Sat → Sun → Mon → Tue |
|---|---|---|---|
| 0-9 | 0.76% → 0.49% | 0.63% → 0.52% | 0.52 / 0.59 / 0.54 / 0.50% |
| 10-49 | 4.98% → 3.53% | 3.61% → 3.61% | 3.66 / 3.55 / 3.48 / 3.49% |
| 50-149 | **15.57% → 11.62%** | 12.31% → 11.49% | 11.97 / 11.98 / 11.81 / 11.37% |
| ≥ 150 | **30.17% → 24.14%** | 29.46% → 27.82% | 28.75 / 29.02 / 28.71 / 27.81% |

- Candidate missingness is ~0.003-0.017% on every day, so candidates are always embedded.
- Train composition per day is flat (eda_shift_overlap §1), so the monotone fall over feed days is best explained by tables snapshotted at the end of each split, with history items that closed before the snapshot losing their embedding.
- The levels line up with that explanation. Train Sep 19 (0 days to the snapshot) at hpos 50-149 is 11.6%, and val Thu (0-1 days) is 11.5%.
- This is consistent with eda_embeddings' note that missing history items are "probably items that were deleted".

**Impact**
- In the train stage, history tokens with a missing embedding are zeroed and masked (`nn/data.py:152-158`, `hemb_ok`). The `seq_*` and `ctx_*` features skip them.
- On train, an old history token is masked up to ~4-6 pp more often than on test, and the mask partly reflects closures that happen after the feed.
- I found no route to the candidate label: candidates are never masked. So I rate this low.

**Test (inference only, CPU possible, as in review_shift `nncpu/`).**
- Re-predict val with the train-only checkpoints after randomly masking non-missing history embeddings at the train-minus-val excess rate per hpos bucket: about +0.3 pp (hpos < 10), +1 pp (10-49), +3 pp (50-149), +2 pp (≥ 150).
- If val NDCG moves by less than 1 SE, ignore the issue.
- If it drops, add train-side mask noise (a history-embedding dropout targeted at old positions) in a future run.

### RESEARCH-4 (info; verified): the generic ±10 clip on the `ctx_hs_*` / `ctx_hn_soft05` attention-mass inputs is harmless

**Evidence** (`c1_scaling.log`):

| feature | share of val rows above 10 | max | lift above 10 | lift in the 8-10 band |
|---|---|---|---|---|
| ctx_hn_soft05 | 1.15% | 125 | 2.43 | 2.35 |
| ctx_hs_k128 | 0.55% | | 2.32 | 2.39 |
| ctx_hs_e1 | 0.90% | | 2.55 | 2.34 |
| ctx_hs_t10 | 0.18% | | 3.15 | 2.85 |

- The signal saturates before the clip.
- These features are right-skewed (median 0.17-0.55, q99 6-11). The model's per-feature standardisation (`nn/train.py`, `feature_stats` / `set_norm`) leaves most of the mass in a narrow band near zero.
- A signed log1p would spread it (optional F5). Expected value is low.
- No other `cvnd` / `cvnd_prew` input is clipped (table in `c1_scaling.log`).

### RESEARCH-5 (info): external impression-style signals are dead on this data (measured)

| idea (source) | our measurement | verdict |
|---|---|---|
| User × item re-exposure, "impressed but not clicked", impression discounting (LinkedIn ImpCount / LastSeen; Ma et al. user-fatigue features; impression-aware RS survey) | 0.18% of val rows; positive rate 1.64% vs base 1.58% (review_ideas §1.7). eda_collab_stats §8: 1.7% of val rows (any earlier train feed), lift +0.0008 ± 0.0008 | dead |
| Re-candidates from the same user's earlier snapshots: expired views come back; allowed, strictly earlier (`c4_ext_hist.log`) | val: 635 rows (0.048%), 484 feeds, 24 positives (lift 2.38; event 2/3: 7 of 42 rows positive); test 0.043% | dead (too sparse: about 0.1% of positives) |
| Listing edits from per-split tables (vector changed between the train table and the val/test table) (`c3_emb_drift.log`) | Changed share among in-train-table rows: 22.9% val, 36.7% test. Positive rate 1.82% (changed) vs 1.69% (identical); the gap holds within f4 quintiles. Nested blend −0.00034 ± 0.00028 (dcos: −0.00031 ± 0.00010) | dead, and it would drift by 14 pp from val to test |
| Time-local transductive impression counts (RecSys 2024 :D L2/L3, trivago) | ≤ 0.6% coverage in ±1 h windows (deep_recsys2024 §7.1); trd_ density features +0.0009 ± 0.0006 (5 seeds) | dead |
| emb_idx / item_id order artifacts (organiser export order) (`c1_scaling.log`) | val single NDCG: rank(emb_idx) 0.2876, rank(−emb_idx) 0.2890, item_id 0.2890 / 0.2885, random 0.2897 ± 0.0016 | none (no artifact) |

---------------------------------------------------------------------------------------------------------------------

## 2. Checked and correct

- **Train-stage transforms** (`nn/data.py:182-200`). All 8 count inputs (`seq_cv_*`, `it_hist_*_luo`) are non-negative, get log1p, and are never clipped. All 10 `ctx_nd_*` and 6 `ctx_hn_*` rank/margin inputs lie within [−1, 1] (`c1_scaling.log`). The only real problem is RESEARCH-1.
- **Fine-tune extras** (`nn/data.py:204-220`): log1p mask and mean/std come from val and are passed to test (`nn/finetune.py:166-167`), so the transform is identical on both. Clip ±6 after standardisation. The `hour` drift (PSI 0.42 / 0.51) is already reported as SHIFT-7 / review_ideas #2 and is not repeated here.
- **Embedding missingness overall:** history −1 rate 17.0% (train), 14.2% (val), 14.0% (test); hpos < 10: 0.72 / 0.58 / 0.54%; candidates 0.003 / 0.015 / 0.016%. The level is comparable. Only the time dependence (RESEARCH-3) is new.
- **New features are leak-free by construction** (`feat_new/research_expo.py`). They read only f4, the feed timestamp, a fixed cutoff rule and prew_*. No labels, no other feeds, no same-user data.
  - The train cutoff reproduces the seq_cv day rule (`sources.py::train_days`/`feed_day_index`).
  - Val/test share one cutoff.
  - Censored share: val 20.07% (matches `c5`), test 30.24%, train 8.58%.
- **Calendar.** Sep 2026 has no Russian holidays, shortened days or transfers (feat_cal_ §1, two official calendars cited there). Paydays (5/10/20/25) are confounded with weekday in 12 train days, and `cal_` features equal a noise control. Nothing further is worth doing.
- **"Items already viewed are excluded".** Verified on all splits (11 of 247M train candidates are in their own history; 0 in val/test, eda_history_leak §2.3). The filter leaks nothing in val/test; `pos` gaps exist only in train and are only an aux target.
- **Per-split tables.** Every builder I read uses the split's own table through split-local `emb_idx` (`nn/data.py:53-56`, feats_v2 builders). Cross-split joins go by item_id only (`c3`).
- **Already covered by our families** (no action):
  - OTTO / H&M / KDD'23 co-visitation and i2i scores → `seq_cv_*` (causal for train rows, +0.0025 in the GBDT harness, in the NN train stage). Type-restricted, cluster-level and Markov variants were tested null (feat_seq_ §2).
  - Recency- and type-weighted history similarity (OTTO 1st, Theo weights) → `seq_*` E group, `ctx_hs_*`, the NN's history attention.
  - Within-impression normalisation (FeatureSalad, trivago, Expedia) → the base `_rk/_z/_gap` features plus the NN's in-model z and rank. Extra ct/cr/fs groups were null (feat_ctx_).
  - BlackPearl item feed-size context → `ctx_ix_*`: +0.0007 ± 0.0007, drifts.
  - Time since publication (RecSys'24 :D, its #1 GBDT feature) → f4 is the exact age (feat_life_ §1).
  - Popularity drift / trend / TIDE-style conformity → `trd_`, `life_` null.
  - Position-debiased item/user TE (Avito'24/'25, VK'24) → `tte_*`, `it_tr_*`, COEC.
  - Pretrained sequence encoders as features (ARGUS, Khrylchenko) → `pre_*`, `prew_*`.

---------------------------------------------------------------------------------------------------------------------

## 3. Research summary: what I read this session and what it adds

The earlier deep dives already cover OTTO, H&M, KDD'23, RecSys'20-'24, VK'24/'25, Avito'24/'25 and Yandex, and I did not redo them:
- `deep_otto_hm.md`;
- `deep_recsys2024.md`;
- `deep_recsys_engagement.md`;
- `deep_ru_competitions.md`;
- `deep_position_bias_random.md`;
- `research_kaggle_solutions.md`;
- `research_competition_avito.md`.

New reading this session ([V] = primary source read):
- **RecSys Challenge 2019 (trivago), Layer 6 2nd place** [V] (cs.toronto.edu/~mvolkovs/recsys2019_challenge.pdf). This is the canonical "impression list given" ranking task. Lessons that transfer:
  - (a) Contrast local session features with global statistics, plus impression-list summaries. Our ctx_ and ix groups already do this.
  - (b) For the NN they used a **per-feature Yeo-Johnson transform chosen to minimise skewness**, sign-preserving embeddings for signed ordinals, and **GBDT leaf indices compressed by SVD as NN inputs**. The leaf inputs gave +0.003 MRR for both LSTM (0.6721 → 0.6750) and Transformer (0.6734 → 0.6765). Their GBDT was their best single model, though. Ours is ~0.025 NDCG below the NN, so the expected transfer is small (F6).
  - (c) Their most important feature was impression position, which has no analogue in shuffled val/test.
- **M5 / Favorita horizon-specific features** [V M5 top-3% README, github.com/btrotta/kaggle-m5]: "for each n I recalculated the features so that the n-day-ahead model is trained on data that has been lagged by n days". Favorita top solutions trained one model per horizon day [S, fast.ai forum, arXiv 2009.07701]. This is the source of F1.
- **Impression-aware recommender systems survey** (Pérez Maurera et al., arXiv 2308.07857 / TORS 2025) [S]. Impression frequency and recency features: LinkedIn ImpCount / LastSeen; user-fatigue counts over 3-1440-minute windows. Our re-exposure measurements make them inapplicable (RESEARCH-5).
- **TIDE** (Zhao et al., arXiv 2109.07946) [S]: popularity = static quality + time-decayed conformity. Our trd_/life_ windows test exactly this and are null.
- **Avito Context Ad Clicks (Kaggle 2015)**: no primary write-up was retrievable. The secondary overview describes position-factor features (adamdrake.com), which have no analogue in shuffled val/test. Not used.

---------------------------------------------------------------------------------------------------------------------

## 4. Technique → status map (comparable competitions)

| technique | where it won | status here | evidence |
|---|---|---|---|
| Co-visitation, time/type-weighted, buy2buy | OTTO 1st-20th | have (causal, item level); type/cluster variants null | feat_seq_ §2-3; nn_report (train stage) |
| i2v / ALS / BPR user-item scores | H&M 3rd, OTTO 17th/20th | have (clean ALS); NN-neutral in the fine-tune | CURRENT_STATUS 2b; err_SYNTHESIS |
| Candidate-generator scores / ranks | OTTO, H&M | not available (only the L2 top-300 minus viewed; f1-f5 are organiser features) | readme; deep_ru_competitions §2.1 |
| Within-impression normalisation | trivago, Expedia, RecSys'24 FeatureSalad | have; extra groups null | feat_ctx_ §1, §4 |
| Item cross-impression context | RecSys'24 BlackPearl | tested, borderline, drifts | feat_ctx_ §4.3 |
| Time-local transductive counts | RecSys'24 :D L2/L3 | tested, no density | deep_recsys2024 §7.1; feat_trd_ |
| Time since publication | RecSys'24 :D | f4 is the exact age | feat_life_ §1 |
| Re-exposure / skip features | Yandex WSCD'14, LinkedIn, trivago | dead (0.05-0.18% rows) | RESEARCH-5 |
| Out-of-time TE with smoothing | Avito'24/'25, VK'24 | have (tte_, it_tr_, COEC) | features_tte.py; eda_collab_stats |
| Position debiasing via a context tower / domain token | Yandex, Airbnb | have (domain token, aux head) | nn_report |
| Horizon-aligned features | M5, Favorita | **missing** → F1/F2 | RESEARCH-2 |
| Skew-minimising NN input transforms | trivago L6 (Yeo-Johnson) | partly (log1p for counts; fixed clip otherwise) → F4/F5 | RESEARCH-1/4 |
| GBDT leaf embeddings as NN input | trivago L6, Facebook GBDT+LR | missing; low expected value → F6 | §3 |
| Calendar (paydays, holidays) | retail forecasting | no holidays; paydays not identifiable | feat_cal_ |

---------------------------------------------------------------------------------------------------------------------

## 5. Feature ideas (only missing or badly done; ranked by expected private gain per cost)

### F1. Horizon-aligned train-row co-visitation: `cvlag_*` (+ the `cv_stale_d` input)
- **Mechanism.** RESEARCH-2: the train stage learns what a zero means from rows that are 0-1 day stale, then is applied to rows that are 3-5 days stale (val) or 7-10 days stale (test). With a lagged cutoff, train rows show test-like censoring: 24.2% at L = 4, 32.5% at L = 8, vs 20.1% val and 31.3% private.
- **Construction (train rows).** Draw L per feed from {0, 4, 8} days with a fixed hash of feed_id (one third each), so the model sees the whole staleness range. The source for a train row of day d is:
  - the latest train feed of each non-zero user with day < d − L, plus all user-0 feeds with day < d − L (= `sources.source_fidx(d − L)`);
  - **plus** a density-matched sparse window: a fixed 1/37 hash sample of train feeds with day in [d − L, d) and timestamp < own timestamp, treated like the transductive val/test parts (`feat_covis.py` Index parts with `True`). This mirrors exactly what val/test rows get: a dense source up to the cutoff, then the sparse exploration-density histories up to the feed.
  - Skip sequences of the same user, as now. Rows with no source keep 0 (like Sep 6 today).
- **Construction (val/test rows).** Unchanged `seq_cv_*` (cutoff Sep 20 + earlier val/test histories). Both sides then have the same structure; only the horizon distribution is matched.
- **Cost.** One index per (day, L) already exists conceptually: `source_fidx(c)` for c = d − L reuses the per-day loop of `feat_covis.py::run_train`. About 3 × 32 min CPU for all train rows at 16 threads, or about 1/3 of that if each train feed is queried only for its own L.
- **`cv_stale_d`** (feed level) = (t_feed − cutoff)/86400. Train: L + day fraction. Val: 2-5. Test: 6-10. Add it only together with the lag mix; without it, train never exceeds 1 day and the model would extrapolate.
- **NN scaling.**
  - Name the columns with the `seq_cv` prefix (e.g. `seq_cvlag_invd20`, `seq_cvlag_pop`, `seq_cvlag_norm`) so `attach_xt` applies log1p and they qualify for `xt_idcols` / `XT_REL`.
  - `cv_stale_d`: divide by 10 and clip to [0, 1.5] when building the file. attach_xt's ±10 clip is then inactive.
- **Test (pre-declared).**
  - (1) Train-fit LightGBM gate (the `pretrain/gate_pre2.py` setup: 25K random-like feeds per window, LUO base). Compare `seq_cv` train rows vs `seq_cvlag` train rows, both scored on val and on the review's stale-val copy.
  - (2) One paired NN run (`cvnd` with `seq_cv*` → `seq_cvlag*` + `cv_stale_d`), accepted on paired train-only val > 2 SE, holdout not worse, and stale-val not worse.
- **Expected.** 0 on val to +0.001 on private. The coverage-mismatch cost measured in the GBDT gap test is 0.0013 ± 0.0003 (feat_seq_ V4).

### F2. Source-exposure time (censoring offset): `src_expo_slog`, `src_censored` (built)
- **Mechanism.** Exposure offsets in count models: a zero count means "unpopular" only if the item could have been counted. E = cutoff − t_post separates "no evidence possible" (lift 0.67 val / 0.43 train among cv-zeros) from "no evidence despite exposure" (lift 0.99 on both).
- **Construction** (`feat_new/research_expo.py`): t_post = t_feed − age(f4) (life_ knots); the cutoff is the seq_cv rule, i.e. the own-day start for train rows and 2026-09-20 00:00 MSK for val/test.
  - `src_expo_h` = E in hours. Capped-f4 (very old) listings get 1e5.
  - `src_censored` = 1[E ≤ 0].
  - `src_expo_slog` = sign(E)·log1p(|E|/24).
  - Under F1, recompute with the lagged cutoff (E − 24L).
- **NN scaling.** Use `src_expo_slog` (range about −2.4 … 8.3, inside the ±10 clip) and `src_censored` (0/1). Do not feed raw hours.
- **Parity.** Same formula everywhere. The distribution differs by design (staleness): censored share 8.6% (train, L = 0), 20.1% (val), 30.2% (test).
- **Expected.** Small on its own: the post-hoc blend is +0.00028 ± 0.00041, and the NN can partly derive it from f4. Its value is as the explicit "this zero is censored" input inside F1. Add it in the same paired run.

### F3. Clip-safe prew inputs: `prew_fe_c`, `prew_new_c` (built)
- See RESEARCH-1. Construction: (prew_fe − 15)/2 and (prew_new − 14)/2, NaN → −10. Fixed constants, identical on all splits.
- **NN scaling:** use them only through `XT_REL` + `XT_RELONLY`, as for the current prew columns. Keep `has:prew_fe` as the availability flag.
- **Expected:** 0 to +0.0005. It removes a degenerate duplicate input; the information is mostly present through `prew_*_z`.
- **Cost:** a preset line, applied to the next members.

### F4. Rule for every future train-row time-sensitive input: lag and density-match it like F1
- This applies when train-label statistics enter the train stage: `it_tr_*`, T1 taste profiles (err_SYNTHESIS §4), clu_ label tables, or any windowed density.
- Build train rows from days < d − L (L mixed in {0, 4, 8}), not from days < d. Add the sparse 1/37 recent window where the val/test analogue has one.
- Without this, every such input repeats the RESEARCH-2 mismatch, and the train holdout will over-reward it (LEAKAGE-1).

### F5. Signed log1p for the right-skewed attention-mass inputs (optional)
- New files `ctx_hs_k128_l`, `ctx_hs_e1_l`, `ctx_hs_t10_l` and `ctx_hn_soft05_l` = log1p(x), all splits.
- It spreads the 0.17-0.55 medians and preserves the saturated tail (RESEARCH-4). This is the Yeo-Johnson idea from Layer 6 in its simplest form.
- Expected ≤ +0.0003. Only worth adding in a run that changes the preset anyway.

### F6. GBDT leaf embeddings as NN inputs (low)
- Construction:
  - fit a LightGBM on random-like train feeds of days < d − L (out-of-time) and K-fold by user inside;
  - take leaf indices for all rows; on train rows, use only out-of-fold, out-of-time leaves;
  - compress them with a truncated SVD fitted on train rows (64 dims);
  - feed them as xt with a has-flag.
- Leak risk: in-fold leaves memorise labels. Folds must be by user and by time.
- Expected small: the trivago transfer was +0.003 MRR, but their GBDT beat their NN, while ours trails by ~0.025. Do this only if GPU time is idle.

**Not proposed (measured dead or covered):** re-exposure / fatigue counts; earlier-snapshot re-candidates; listing-edit flags from table changes; time-local transductive counts; calendar and paydays; new co-vis variants beyond F1; cluster-level label TE.

---------------------------------------------------------------------------------------------------------------------

## 6. Files

- Code: `<repo>/feat_new/research_expo.py` (builds F2 + F3 for all splits in ~5 min at 16 threads).
- Features: `<repo>/data/feats_new/{train,val,test}/{src_expo_h,src_censored,src_expo_slog,prew_fe_c,prew_new_c}.npy`.
- Checks and logs in `/tmp/scratchpad/featrev/research/`:
  - `c1_scaling` (transform clipping, id-order artifact);
  - `c2_prew_clip`;
  - `c3_emb_drift`;
  - `c4_ext_hist`;
  - `c5_exposure_time`;
  - `c6_missing_emb_time`;
  - `c7_verify_new`.
- Caveat: `c2`'s train sample uses random-like feeds from Sep 16 on, because of a mistyped constant (1789506000 = Sep 16 00:00 MSK). That is in-window for prew and does not change the conclusion.

## 7. Sources

- Layer 6, "Robust Contextual Models for In-Session Personalization", RecSys Challenge 2019: https://www.cs.toronto.edu/~mvolkovs/recsys2019_challenge.pdf
- btrotta, Kaggle M5 solution README: https://github.com/btrotta/kaggle-m5
- Kaggle forecasting competitions review (Bojer & Meldgaard): https://arxiv.org/pdf/2009.07701
- Favorita discussion (fast.ai forum): https://forums.fast.ai/t/corporacion-favorita-grocery-sales-forecasting/8359?page=4
- Pérez Maurera et al., "Impression-Aware Recommender Systems": https://arxiv.org/pdf/2308.07857
- Zhao et al., "Popularity Bias Is Not Always Evil" (TIDE): https://arxiv.org/abs/2109.07946
- Adam Drake, overview of the Avito Context Ad Clicks competition: https://adamdrake.com/a-quick-overview-on-the-kaggle-competition-for-avito.html
- Earlier deep dives in `reports/deep_*.md` and `reports/research_*.md`; their sources are listed there.
