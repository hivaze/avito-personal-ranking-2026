# Review: feature leakage and train / val / test construction parity

Author: leakage-review agent (CPU only, read-only on project files). Date 2026-11-11, ~14:00-15:00 UTC.
Scripts and raw outputs: `/tmp/scratchpad/review/leakage/`
(`detect.py` / `detect.csv` / `detect_table.md`, `detect2.py`, `luo_time.py`, `luo_rates.py`, `causal_counts.py`, `lgb_causal.py`,
`item_mem.py`, `stale_calib.py`, `vt_parity.py`, `pre2_check.py`, `harness_base.py`, `idcontrib_lgb.py`, `emb_parity.py`, plus `*.log`).

## TL;DR
* **I found no label leak in the live path**: the nn_final4 / round-4 train stage (`--xt cvnd` / `cvnd_prew`), the n2c_pre fine-tune extras, or the pre_final blend term.
  * Every train-stage input passes the generic detector: its train-vs-val within-feed AUC and missingness behaviour track the reference features.
  * The two outliers have measured, non-label explanations. it_hist_*_luo carries future cross-user popularity. prew_* comes from full-density, zero-staleness window corpora.
  * The item-ID path is off: `id_buckets 0` in all 4 nn_final4 members and all 6 round-4 runs.
* **The most decision-relevant problem is the yardstick, not a feature.**
  * The random-like train holdout over-rewards item-popularity and item-memorisation inputs by about 2.5x relative to val.
  * The reason is that the holdout's candidates have two-sided-in-time item information (train labels and other users' histories before AND after the feed). Val/test rows only have past information, which is also stale.
  * Holdout gains for such inputs must not be counted as evidence. prew_* was accepted on val +0.0011 ± 0.0007 (1.6 SE, 1 seed), below the protocol's 2 SE rule.
* **Verified bugs, all small:**
  1. `pre_id_contrib` is built differently on val and test. On val it is a non-zero linear function of pre_fe on no-ID rows; on test it is exactly 0 on 59% of rows. PSI 3.5. Measured impact on a tree consumer is ±0.0002.
  2. user-0 rows keep later user-0 snapshots in it_hist_*_luo (lift 2.5 on 0.6% of train rows) and in all pretraining corpora.
  3. `nn/feat_cache/train/pre_*` are still hard links to the quarantined leaky files, so the quarantine does not cover the NN loader.
  4. The `src/eval_family.py` frozen base still contains non-LUO it_hist_* and transductive it_cand_cnt_split*. Effect: +0.022 on the 117 later-user val feeds, -0.0004 ± 0.0012 overall.
* **One tempting fix is wrong.** Making it_hist_*_luo causal removes the 56% future share of train counts but lowers val by -0.0018 ± 0.0008 (train-fit LightGBM, 2 seeds). Keep the transductive version.

## Inputs enumerated (task 1)
| path | inputs | builder |
|---|---|---|
| train stage, in-model dense (`nn/model.py` dense_features) | f1..f5 raw / within-feed z / pct rank; cand-emb missing flag; item norm; cos/dot with user emb; cand x history cosine max/max8/max32/mean/lse/top3; per-event-type max; category affinity counts (total, per event type), share, first-recency; feed category share; log M, log H; z/rank of cos_u, smax, lse, top3, cat count; is_user0, is_user0*log(1+H) | computed on GPU from the flat arrays and the split's own item table |
| train stage tokens | item emb (split-specific organiser table), category, event type/source, recency index, user emb (per-feed snapshot); domain token (train [1-p, p, 0], val/test infer_domain); TabR retrieval over the top-16 history items; position tower OFF (`--no_pos`). The production `pos` is used only as the aux listwise target, weighted by 1-p_random | `nn/data.py`, `nn/model.py` |
| train stage xt `cvnd` (`nn/train.py:32`) | seq_cv_invd20 / pop / norm; it_hist_all_luo, it_hist_et1..4_luo; ctx_nd_{hcent, knn_f1, knn_h, knn_h_dev, knn_u, knn_u_dev, max, top5h, top5u, ucent}; ctx_hn_{top1w, rrw, bestrk, margin, soft05, e1_rrw}; ctx_hs_{mutual, k128, e1, t10}. XT_REL: in-model rank + z of seq_cv_invd20, it_hist_all_luo | `feats_v2/seq_/feat_covis.py`, `src/features_luo.py` (on `src/features.py` it_hist_*), `feats_v2/ctx_/build_ctx*.py` |
| train stage xt `cvnd_prew` (`nn/train.py:48-51`) | + prew_fe_z, prew_new_z, has:prew_fe; prew_fe / prew_new / prew_hmax32 through in-model rank + z only (XT_RELONLY) | `pretrain/make_prew.py` (on the window-encoder pre_* for train, v3A/v4A and v3/v4 for val/test) |
| fine-tune extras `n2c_pre` (51 features actually loaded, from `nn/logs/v9_e10_n2cp_ft.log`) | it_cand_cnt_train; it_tr_coec (+rk/z/gap), it_tr_ctr, it_tr_pos, u_tr_ctr, u_tr_feeds; hour; centrality; it_hist_all_luo (+rk/z/gap), it_hist_et1..10_luo; seq_cv_invd20 (+rk), seq_cv_pop, seq_cv_norm; seq_max_all, seq_top3, seq_argmax_hpos; the 10 ctx_nd_*; 9 pre_* | as above; train-label stats from `src/features.py` |
| blend | 0.1 z(pre_final) (val: v3A + v4A moment-matched to v3/v4; test: v3 + v4) | `pretrain/assemble.py` |

## Findings (most severe first)

### LEAKAGE-1 (medium, verified): the train holdout over-rewards item-popularity and memorisation inputs, which biases holdout-based acceptance and selection
Evidence:
* **Time structure of item information.**
  * it_hist_all_luo on the 30K holdout rows (exact nn/train.py holdout): 55.6% of the count mass comes from other users' snapshots taken after the feed. Val: 1.9%. Test: 0.9%.
  * 18.3% of holdout rows are warm only because of future snapshots, vs 2.9% val and 1.7% test (`luo_time.log`).
* **Item-label exposure** (`item_mem.py`):
  * Holdout: 27.3% of rows have an item that is a train positive in another feed (19.2% before the feed, 15.2% after). Val: 19.7%. Test public: 16.3%. Test private: 14.9%.
  * AUC of "item positives elsewhere": holdout 0.5438 two-sided vs 0.5303 past-only; val 0.5256.
  * The NN can memorise item quality through the unique per-item content vector even with the ID table off.
* **Direct measurement** (train-fit LightGBM lambdarank, eval_family params, 50K random-like non-holdout train feeds, 2 seeds; `lgb_causal.py`). Paired deltas:
  * it_hist_*_luo vs no popularity: **holdout +0.0054 ± 0.0005 vs val +0.0022 ± 0.0009**.
  * causal counts vs no popularity: holdout +0.0020 ± 0.0004 vs val +0.0004 ± 0.0007.
  * The holdout inflates the value of popularity inputs by about 2.5-5x.
* **The prew_* acceptance does not meet the protocol.**
  * Recorded result (CURRENT_STATUS 3g, nn_report round 4): val +0.0011 ± 0.0007 (1.6 SE, one seed pair), holdout +0.0008 ± 0.0004.
  * It satisfies the section 3d rule (val improves, holdout not worse) but not the section 4 rule (paired delta > 2 SE, ≥ 2 seeds).
  * prew_* is exactly the kind of input whose train-side construction is richer (LEAKAGE-6), so its holdout gain is expected regardless of val.

Impact:
* No direct NDCG loss.
* Decisions made on the holdout can be tilted toward inputs and checkpoints that exploit concurrent item information: member sets, checkpoint steps, and accepting train-stage inputs.

Fix and tests:
* Keep "train-only val must improve by > 2 SE" binding for every train-stage input, and ignore holdout deltas for popularity-, CF- or encoder-type inputs.
* Re-check prew with the free seeds once they finish. Ensemble v12_allprew_s83/84/85 vs v12_all_e10/s81/s82 on train-only val, unpaired 3 vs 3; report the mean difference and the seed SD.
* For future selection runs, consider a time-separated holdout: the random-like feeds of Sep 18-19, with training on Sep 6-17 for the probe runs only.

### LEAKAGE-2 (low, verified bug): `pre_id_contrib` (and the pre2_/prew_ analogues) are constructed differently on val and test
Evidence:
* `pretrain/assemble.py:44-54` moment-matches each raw key separately (`affine()`, line 27) for the val-side models, then sets `id_contrib = fe - noid` (line 54) after calibration.
* For candidates without an ID, raw fe == fe_noid: 54.8% of val rows in both v3A and v4A, 59.2% of test rows. After separate affine maps:
  * val id_contrib on these rows = -0.0031 ± 0.0017, with correlation 0.980 with pre_fe;
  * test (no calibration) = exactly 0;
  * exact-zero share 0.0002 on val vs 0.59 on test, so **PSI val->test = 3.5** (`vt_parity.py`; every other n2c_pre feature has PSI ≤ 0.23).
* The same pattern holds for prew_id_contrib (zero share: train 0, val 0, test 0.26) and pre2_id_contrib (train 0, val 0, test 0.59). `pretrain/assemble_train.py` also moment-matches per key.
* `pre_id_contrib` is a live fine-tune extra (n2c_pre, `nn/finetune.py:148-151`). prew/pre2_id_contrib are not in any live preset.

Impact:
* For the NN, the standardised val-only variation is ~0.0014 SD, so it is negligible.
* For a tree consumer it is also negligible as measured. Val-CV LightGBM (clean base + 9 pre_*), held-out folds re-predicted with the test-style id_contrib (0 on no-ID rows): delta +0.00018 ± 0.00018 (seed 0) and -0.00020 ± 0.00017 (seed 1); gain share of pre_id_contrib 1.2-1.3% (`idcontrib_lgb.py`).

Fix: in assemble.py compute `id_contrib = a_fe * (fe_raw - fe_noid_raw)` (one slope), or set it to 0 where raw fe == fe_noid. Apply the same in assemble_train.py and before any pre2 hand-off.

### LEAKAGE-3 (low, verified): user-0 rows keep later user-0 snapshots (it_hist_*_luo and all pretraining corpora)
Evidence:
* `src/features_luo.py:23` builds the subtraction set from `user_id > 0` only. user-0 rows therefore get it_hist_all_luo == it_hist_all, which includes later snapshots of the pooled bucket.
* On holdout user-0 rows, "item occurs in a later user-0 snapshot":
  * 0.59% of all rows (11.5% of user-0 rows), P(y) 2.87% vs 0.93% for the other user-0 rows (lift 2.5-3.1);
  * 788 of the ~53.7K holdout positives.
* On val the same signal exists on only 0.036% of rows (lift 2.0), a 16x density asymmetry (`luo_rates.py`).
* AUC of it_hist_all_luo on user-0 feeds: train 0.6473 vs val 0.6055. Relative to maxsim32: -0.037 vs -0.071.
* The same rule appears in `pretrain/build_corpus_window.py:39-40` (every user-0 in-window feed, including later ones) and `pretrain/build_corpus.py:85` (every user-0 val/test feed).

Impact: small. Under 1.5% of train positives carry a weak (lift ~3) same-bucket-later signal, and the model can only reach it through the it_hist counts on user-0 rows.

Fix: treat user 0 as one user in the LUO rule (subtract later user-0 occurrences for user-0 rows), and use only the earliest user-0 snapshot per window/split in corpora, or exclude user-0 snapshots when scoring user-0 rows. Re-run the 117-feed audit including user 0.

### LEAKAGE-4 (low, verified): the quarantine does not cover the NN feature loader
Evidence:
* `nn/feat_cache/train/pre_{fe, fe_z, new, new_z, hmax32, hmax8, fe_noid, id_contrib, idmax32}.npy` are hard links to the quarantined leaky train pre_*: `stat` shows inode 2650183 with link count 2 for both `nn/feat_cache/train/pre_fe.npy` and `data/feats_quarantine/train/pre_fe.npy`.
* `nn/data.py:31-46` (`feat_path` / `feat_exists`) checks FEAT_CACHE first. So `--xt cvnd_pre` or `cvnd_prez` (`nn/train.py:38-46`, still defined) would silently train on the 8b leak.

Impact: none today, since live runs use cvnd / cvnd_prew; it is a guardrail gap.

Fix (lead decision, outside my write permissions): delete `nn/feat_cache/train/pre_*.npy`, or make `feat_path` refuse names present in `data/feats_quarantine/<split>/`. Remove the cvnd_pre / cvnd_prez presets.

### LEAKAGE-5 (low, verified): the val feature harness base contains banned same-user-later features
Evidence:
* `src/eval_family.py:31-38,69` uses `data/eval_cache/base_features.json` as is. That file contains it_hist_all (+_rk/_z/_gap), it_hist_et1..10 (non-LUO, i.e. same-user later snapshots) and it_cand_cnt_split (+_rk/_z/_gap; transductive over the split's feeds, incl. the same user's later feeds). No `swap_luo` is applied.
* Every family verdict (seq_, ctx_, clu_, cal_, trd_, life_, pre_, prew_) was measured on top of this base.
* Measured (`harness_base.py`, eval_family protocol, seed 0), clean base (swap_luo, it_cand_cnt_split* dropped) minus the frozen base:
  * on the 117 later-user val feeds: **-0.0221 ± 0.0142** (0.3633 -> 0.3412), so the frozen base exploits the leak there;
  * on the other feeds: -0.0002 ± 0.0012; overall -0.0004 ± 0.0012.

Impact: verdicts are essentially unaffected (the leak touches 0.8% of feeds), but base NDCG figures quoted from the harness (0.3634) include ~+0.0002 of leak.

Fix: freeze a clean base (swap_luo, drop it_cand_cnt_split*) and invalidate `data/eval_cache/base_oof_s*.npy` before more family screens.

### LEAKAGE-6 (info, verified): prew_* is stronger on train than on val (construction density), not a label leak
Evidence (detector, present rows; `detect.log`, `detect2.py`):
* prew_fe within-feed AUC: train holdout 0.6550 vs val 0.6457. Relative to maxsim32 on the same rows: train +0.0110 (Sep 9-14 +0.0143, Sep 17-19 +0.0101) vs val -0.0041 (M ≥ 50 val -0.0055). Skew ≈ +0.015.
* The content-only prew_fe_noid has a skew of +0.012, prew_id_contrib +0.019 and prew_hmax32 +0.004.
* Consistent with the documented mismatches (pretrain_report 8):
  * in-window corpora at 100% production density vs a ~3% exploration sample;
  * ~0-day background staleness vs 2.5-5 days.
  * Note: the per-item int8 content vector is unique, so the "content" tower can memorise items that other users opened inside the window, after the scored feed.
* **Not the 8b leak:** after the LUO-cold mask every remaining train candidate has content (LUO-warm ⇒ appears in some history ⇒ in the content table), so the availability bit is gone.
* Missingness behaves the same on both splits: P(y|NaN)/P(y) is 0.86 train vs 0.80 val.
* `has:prew_fe` is 0 for all Sep 6-8 train rows (the windows start Sep 9), so in train it means "LUO-cold OR early September", while on val/test it means "LUO-cold" only. The model can separate the two via it_hist_all_luo > 0, so this is harmless but wasteful.

Impact: prew's train-side signal is richer than at test time. This is why its holdout gain is not evidence (LEAKAGE-1). Its val gain, +0.0011 ± 0.0007, is the honest number.

### LEAKAGE-7 (info, verified): it_hist_*_luo time-travels on train rows, but the as-built version beats a causal one on val
Evidence:
* Recomputing the count from the raw history arrays reproduces `it_hist_all` exactly on all three splits (max abs diff 0).
* Decomposition by snapshot time (`luo_time.log`, `luo_rates.py`):
  * train holdout: LUO-warm share 78%, future share of the count 56%; val 62% / 1.9%; test 55% / 0.9%;
  * P(y | LUO-cold)/P(y): 0.66 train vs 0.80 val;
  * within-feed AUC: train 0.5716 (causal part only: 0.5567; future part alone: 0.5683) vs val 0.5561 (causal 0.5523);
  * relative skew vs maxsim32: +0.021.
* **Not a label channel.** Rows that are warm only through other users' future snapshots have lift 0.97 on train, i.e. ordinary popularity. Same-user-later occurrences are correctly removed (rows that were warm only through them: 91% positive).
* Causal counts (snapshots strictly before the feed) give an almost perfectly matched warm share (train sample 61.7%, holdout 59.7%, val 59.2%, test 53.6%; `causal_counts.log`). **But in the train-fit LightGBM test they lose:**
  * causal - as built: val -0.0018 ± 0.0008, holdout -0.0034 ± 0.0005;
  * 4-day-lagged causal - as built: val -0.0027 ± 0.0008.
  * The full-period count is closer in meaning to the val/test count (the whole train period is past for them) than a per-day expanding count.

Recommendation: keep it_hist_*_luo as built. Do not "fix" it to a causal count. A gap-matched variant is listed under ideas.

### LEAKAGE-8 (info): smaller parity notes (no action needed)
* **seq_cv_\* (expanding window, causal):**
  * Single-feature AUC rises with the train day: Sep 6-8 0.520, Sep 9-14 0.545, Sep 17-19 0.552, vs val 0.5515. Raw counts therefore have a day-dependent scale (the NN also gets within-feed rank/z).
  * Val/test sources add every other-user val/test feed history (`feats_v2/seq_/feat_covis.py:191`, not de-duplicated per user, unlike the train source). This only affects the few multi-feed val/test users.
* **Organiser item tables differ slightly per split:** 25% of items shared by the train and val tables have different int8 vectors (median cosine 0.9996, 1st percentile 0.905; `emb_parity.py`). This is a small, unavoidable dampening of content-identity memorisation at val/test.
* **Item staleness val -> test:** the share of a feed's candidates that were train candidates is 0.652 on val, 0.562 on test public and 0.523 on test private.
  * Reweighting val to the joint (M bucket x seen-share) mix would predict +0.0039 public / +0.0050 private above the M-only calib (`stale_calib.py`; NDCG is U-shaped in seen-share).
  * The M-only calibration already matched three public submissions within 0.0003-0.0014, so this axis does not explain any val->test gap. No evidence that item memorisation inflates val relative to test.
* **ctx_hn_top1w / ctx_hs_mutual / ctx_nd_knn_\*** show train-vs-val AUC gaps of ±0.02 on all val feeds. On M ≥ 50 val feeds (matching the holdout's M ≥ 50) they agree within 0.005 (e.g. top1w relative AUC -0.0895 train vs -0.0899 val M50). This is M composition, not construction.

## Generic leak detector (task 3)
Within-feed AUC of each feature alone on the 30K holdout (4.63M rows, P(y) 1.16%) vs all val (1.39M rows, P(y) 1.58%).
* Skew = (|AUC_tr - 0.5| - |AUC_va - 0.5|) - the same quantity for maxsim32 (train 0.6440, val 0.6498), which controls the population difference.
* lift = P(y | condition)/P(y). Full table: `detect_table.md`; all 117 features: `detect.csv`.

| group | feature | AUC holdout | AUC val | skew | NaN tr/va | zero tr/va | lift(y\|0) tr/va | lift(y\|NaN) tr/va | verdict |
|---|---|---|---|---|---|---|---|---|---|
| contrast | it_cand_cnt_split | 0.5416 | 0.4965 | +0.044 | 0/0 | 0/0 | - | - | known leaky (train row counts itself + future); not used live |
| contrast | it_hist_et1 / it_hist_all (non-LUO) | 0.584 / 0.581 | 0.558 / 0.557 | +0.032 / +0.030 | 0/0 | 0.25/0.41 | 0.57/0.80 | - | known leaky (same-user later); not used live |
| ft (val/test only) | it_tr_ctr / coec / pos | 0.533 / 0.527 / 0.544 | 0.510 / 0.507 / 0.526 | +0.029 / +0.026 / +0.024 | 0/0 | - | - | - | train rows are in-sample (leave-own-row-out only); never used in the train stage, legit on val/test |
| xt | it_hist_et1_luo / it_hist_all_luo | 0.5745 / 0.5716 | 0.5571 / 0.5561 | +0.023 / +0.021 | 0/0 | 0.22/0.38 | 0.66/0.80 | - | future cross-user popularity, LEAKAGE-7 (not a label channel) |
| xt | ctx_hn_top1w | 0.5545 | 0.5816 | -0.021 | 0.026/0.014 | 0.88/0.86 | 0.87/0.75 | 0.97/0.75 | M composition (val M ≥ 50: 0.5636) |
| xt | ctx_nd_knn_u / knn_f1 / knn_h | 0.615 / 0.619 / 0.625 | 0.601 / 0.607 / 0.613 | +0.020 / +0.018 / +0.017 | ≤0.036/0.014 | 0/0 | - | ~1.0/0.75-0.93 | M composition (val M ≥ 50: 0.6195 / 0.6223) |
| xt_prew | prew_id_contrib (not live) | 0.5373 | 0.5240 | +0.019 | 0.44/0.38 | 0/0 | - | 0.86/0.80 | density skew + LEAKAGE-2 |
| xt | ctx_hs_mutual | 0.5440 | 0.5669 | -0.017 | 0.026/0.014 | 0.90/0.89 | 0.90/0.80 | 0.97/0.75 | M composition |
| xt | it_hist_et2/3/4_luo | 0.543/0.540/0.539 | 0.533/0.531/0.528 | +0.015..0.017 | 0/0 | 0.57-0.67/0.69-0.75 | 0.88-0.91/0.93-0.94 | - | as it_hist_all_luo |
| xt_prew | prew_fe / prew_new (and _z) | 0.6550 / 0.6547 | 0.6457 / 0.6452 | +0.015 | 0.44/0.38 | 0/0 | - | 0.86/0.80 | density / staleness skew, LEAKAGE-6 |
| xt_prew | prew_fe_noid | 0.6459 | 0.6399 | +0.012 | 0.44/0.38 | 0/0 | - | 0.86/0.80 | same |
| xt | ctx_hn_e1_rrw / rrw / soft05 | 0.651 / 0.649 / 0.653 | 0.645 / 0.645 / 0.651 | +0.012 / +0.010 / +0.008 | 0.03-0.04/0.01-0.03 | 0/0 | - | 0.92-0.97/0.75-0.86 | within noise of base features |
| base | f1 / f3 / f5 | 0.6230 / 0.6018 / 0.5635 | 0.6226 / 0.6007 / 0.5610 | +0.006 / +0.007 / +0.008 | 0/0 | - | - | - | reference: organiser encoders show no in-sample inflation on train |
| xt | seq_cv_invd20 / norm / pop | 0.5407 / 0.5407 / 0.5471 | 0.5515 / 0.5515 / 0.5505 | -0.005 / -0.005 / +0.002 | 0/0 | 0.92/0.89 | 0.90/0.89 | - | OK (early days weaker, LEAKAGE-8) |
| xt | ctx_hs_k128, ctx_hn_bestrk, ctx_nd_ucent | 0.654 / 0.368 / 0.555 | 0.656 / 0.365 / 0.566 | +0.005 / +0.003 / -0.005 | 0.026-0.036/0.014 | - | 2.24/2.80 (bestrk = 0) | ~1/0.75-0.93 | OK |
| xt_prew | prew_hmax32, prew_idmax32 | 0.6333 / 0.5244 | 0.6347 / 0.5230 | +0.004 / +0.007 | 0.44-0.61/0.38-0.55 | 0/0 | - | 0.86-0.90/0.80-0.89 | OK |
| base | emb_missing (cand) | - | - | - | missing share 0.0000 train / 0.0002 val | | | P(y\|miss) 0.006 train / 0.023 val on a handful of rows | no availability bit (unlike 8b) |

Stratified view (relative AUC vs maxsim32 on the same rows; `detect2.py`):

| feature | holdout Sep 6-8 | holdout Sep 9-14 | holdout Sep 17-19 | holdout user-0 | val all | val M ≥ 50 | val user-0 |
|---|---|---|---|---|---|---|---|
| it_hist_all_luo | -0.074 | -0.072 | -0.072 | **-0.037** | -0.094 | -0.101 | -0.071 |
| seq_cv_invd20 | -0.129 | -0.097 | -0.089 | -0.152 | -0.098 | -0.101 | -0.132 |
| prew_fe | NaN | +0.014 | +0.010 | -0.032 | -0.004 | -0.006 | -0.050 |
| pre2_fe (W1/W2 only, in progress) | NaN | +0.020 | NaN | +0.004 | +0.012 | +0.011 | -0.002 |
| f1 | -0.024 | -0.019 | -0.021 | -0.027 | -0.027 | -0.027 | -0.013 |
| cos_u | -0.011 | -0.011 | -0.005 | -0.022 | -0.008 | -0.009 | +0.002 |

The feed-level SE of each AUC is 0.002-0.003 (user-0: 0.007-0.010).

## Item-ID and memorisation path (task 4)
* **The hashed item-ID residual is off.** All nn_final4 members (v9_e10, s61, s62, s63) and the round-4 runs (v12_all_e10, s81, s82, v12_allprew_s83/84/85) log `'id_buckets': 0`. `id_drop` is therefore inactive, and fine-tune `build_model` only freezes IDs when `id_buckets > 0`.
* **Memorisation still happens through the content vector.** A 128-d int8 vector is effectively unique per item. Its legitimate use (item quality learned from train labels) is past-only for val/test, but two-sided for holdout feeds: 27.3% vs 19.7% exposure, with an AUC premium of 0.0135 from future labels. That makes it part of LEAKAGE-1.
* Exposure decays val -> test public -> test private (seen share 0.654 / 0.564 / 0.527; positive-elsewhere share 0.197 / 0.163 / 0.149). The M-only calibration already matches public, so there is no measurable penalty (LEAKAGE-8).

## prew_* and pre2_* (task 5)
* **prew_*:** the mask rule (NaN where it_hist_all_luo == 0, z recomputed) is identical on all splits (`pretrain/make_prew.py`). After it, the content-availability bit cannot exist. The remaining skew is density/staleness (LEAKAGE-6). The train prew_* was derived from the now-quarantined train pre_* (written 12:27, before the 13:00 quarantine). This is fine for the masked rows, because the window models were trained identically and only the scoring of content-less candidates differed.
* **pre2_* (rebuild in progress):** `build_item_tab2.py` gives every train candidate a content row. Gate check on the existing W1/W2 train rows (`pre2_check.py`) passes:
  * mean within-feed z of pre2_fe on LUO-cold candidates: **-0.153 train vs -0.146 val** (the old leaky pre_*: -1.46);
  * cold-only AUC 0.6786 vs 0.6776; warm-only 0.6549 vs 0.6445 (maxsim32 warm-only 0.6370 vs 0.6341, so the warm skew is ~+0.007);
  * the overall relative skew is +0.008, smaller than prew's +0.015.
  * Remaining issues: id_contrib (LEAKAGE-2); user-0 corpus rule (LEAKAGE-3); the val gain must be shown on train-only val, not on the holdout (LEAKAGE-1).

## Checked and correct
* **No label inputs in the train stage.** cvnd/prew builders use no targets.
  * seq_cv for train rows: same-user sequences skipped; source strictly before the feed's day (`feats_v2/seq_/sources.py`, `feat_covis.py`).
  * Val/test: source constrained to snapshots with ts < own ts, so val never sees test and test sees only earlier feeds.
* **LUO arithmetic.** it_hist_all recomputed from raw histories equals the stored file exactly on all splits. Same-user-later occurrences are removed for users > 0 (rows with no earlier occurrence but a same-user later one are 91% positive on train and 93% on val, so the removal targets exactly the leaking rows). The removal is applied to every split.
* **ctx_\*** use only the feed's own candidates, its history and the split's item table (no labels, no time windows). Train-vs-val differences are explained by M composition.
* **Organiser features f1-f5 show no in-sample inflation on train rows.** Train ≈ val AUC (f1 0.6230/0.6226), flat by day (0.6250/0.6233/0.6205).
* **user_emb is a per-feed snapshot.** Only 0.14% of a user's feed pairs have identical embeddings, and cos_u AUC is not higher on early days (0.638 / 0.632 / 0.636 vs val 0.641), so it does not encode later behaviour.
* **No candidate content-availability bit in the organiser tables.** Candidate embedding missing share: 0.0000 train vs 0.0002 val.
* **Feature-file timing.** The ctx_nd user-0 fix rewrote knn_u / knn_u_dev / top5u at 07:19-07:20 on all splits. Every nn_final4 member started later (09:14-11:24), so train / val / test inputs are consistent. All feat_cache files share inodes with data/feats (no stale copies), except the quarantined pre_* (LEAKAGE-4).
* **Holdout mechanics.** The holdout is excluded from training (`nn/train.py:163`) and from the cooldown exploration pool (`nn/train.py:256`, drawn from `feeds`). Holdout feeds use their own [1-p, p, 0] domain mix; val/test use `infer_domain`.
* **Fine-tune.**
  * GroupKFold by user (user 0 = one group).
  * Saved predictions = snapshot mean of epochs 1..6: `saved: snapshot` in all four v9_e10*_n2cp_ft.json, so there is no OOF epoch selection.
  * Extra-branch standardisation is fitted on val without labels and reused for test.
  * The 51 loaded extras contain no ALS, x_lh, non-LUO it_hist or it_cand_cnt_split.
* **pre_* val/test (blend term and fine-tune).** No same-user later snapshot in any corpus (earliest val/test snapshot per non-zero user). Val from encoders without test snapshots; raw scales moment-matched without labels. Apart from id_contrib, val vs test PSI ≤ 0.002 for every pre_* feature.
* **Val vs test parity of the n2c_pre extras** (`vt_parity.py`): PSI ≤ 0.07 for all except pre_id_contrib (LEAKAGE-2), hour (0.16-0.19, calendar mix) and it_tr_coec_rk (0.11 public / 0.23 private, item staleness). The last two are population shifts, not construction differences.

## Ideas
1. Fix pre_id_contrib in assemble.py / assemble_train.py (one slope, or exact 0 for no-ID rows) before pre2 is handed to the NN. Cost: CPU minutes.
2. Treat user 0 as a single user in the LUO subtraction and in corpus selection (earliest snapshot only). Re-run the later-user audit with user 0 included.
3. Make the selection yardstick more test-like:
   * either a time-separated random-like holdout (Sep 18-19 feeds, probe runs trained on Sep 6-17);
   * or at least report holdout/val delta ratios per input family and discount holdout for popularity / CF / encoder inputs (measured ratio 2.5x+).
4. Re-validate prew with the 3 vs 3 seeds of round 4 (ensemble vs ensemble, train-only val) before nn_final5 member selection. If the gain is not > 2 SE, prefer pre2_* (smaller skew, covers cold candidates) once its gate passes on train-only val.
5. A gap-matched popularity variant: for train rows, drop other-user snapshots within ±3 days of the feed, mimicking val's 3-5-day gap to dense data.
   * It is cheap to screen with `lgb_causal.py` (8 fits ≈ 9 min on 12 threads).
   * Accept only on val. The as-built count is the bar to beat (causal and lagged variants lost).
6. Delete `nn/feat_cache/train/pre_*.npy` and add a guard in `nn/data.py:feat_path` that refuses quarantined names. Freeze a clean eval_family base (swap_luo, no it_cand_cnt_split) and clear `data/eval_cache/base_oof_s*`.
7. For the fine-tune branch, consider dropping pre_id_contrib / pre_idmax32 if the fix is not done (the tree test shows they are ±0.0002 either way). The ID signal is already in pre_fe vs pre_fe_noid.
