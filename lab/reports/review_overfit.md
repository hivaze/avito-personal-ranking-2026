# Review: selection, overfitting and nested CV (audit dimension "overfit")

Date: 2026-11-11, about 14:30 UTC. CPU only and read-only: no project file was changed.
Scripts and outputs: `/tmp/scratchpad/review/overfit/a*.py`.
All NDCG values come from `src/metrics.py` (`ndcg_per_feed`). "SE" means the paired per-feed SE unless stated otherwise.

## TL;DR

* **No leak or nested-CV bug in the sub_v10 path.**
  * The fine-tune folds are GroupKFold by user (verified in the code).
  * No val-label-derived feature or val-OOF output feeds any second-level model.
  * The holdout is excluded from every training path, and its train-stage inputs are label-free.
  * sub_v10 reconstructs exactly from its components (rank agreement 1.0000).
* **The val selection optimism behind v10 is small.** Estimate: 0.0005-0.0015 val NDCG. The blend-weight choice measures +0.00012 by cluster bootstrap. Public LB shows no detectable optimism: v5 -> v10 transfer residual +0.0007 +- 0.0018.
* **The real risk is a distribution mismatch, not overfitting.** The test gets colder with distance from training (share of candidates with LUO popularity 0):

  | split | share |
  |---|---|
  | holdout | 22% |
  | val (Tue / Thu) | 38% (36% / 40%) |
  | public | 43% |
  | private | 46% |

  So the protocol rewards warm-item priors, the holdout about 3x more than val. Private is the coldest period.
  * Example: prew_* (accepted at 1.4 SE val / 2.0 SE holdout, single seed) is worth about 0 under private-like weights: +0.0002 / -0.0012 / +0.0008 across three cold covariates.
  * prew_* loses 0.0032 +- 0.0013 on feeds whose positives are all LUO-cold.
* **calib.py is biased low** (exact reconstruction of v5 and v10).
  * Adding a per-feed cold-share covariate to the M reweighting shrinks the public residuals from +0.0012 / +0.0019 to -0.0001 / +0.0007.
  * The private prediction for v10 rises from 0.3886 to about 0.392, but it depends on which cold covariate is used.
* **The 10-epoch members' fine-tune is over-trained.**
  * OOF peaks at fine-tune epoch 1-2, then falls 0.003-0.005 below the train-only model by epoch 6, in 4 of 4 members.
  * An honest nested (leave-one-fold-out) epoch choice beats the snapshot 1..6 recipe by +0.0009 per member on average.

---

## 1. Findings (ranked by severity)

### OVERFIT-1 (medium, verified): holdout and val are warmer than private, so acceptance favours warm-item priors

Share of candidate rows that are "cold" (`a6_cold.py`, `a9_holdcold.py`):

| population | LUO-cold (`it_hist_all_luo==0`) | never a train candidate | co-visitation pop = 0 |
|---|---|---|---|
| 30K holdout (train, Sep 6-19) | **0.220** | 0 by construction | 0.475 |
| val Tue Sep 22 / Thu Sep 24 | 0.359 / 0.403 | 0.309 / 0.386 | 0.397 / 0.444 |
| val all | 0.379 | 0.346 | 0.420 |
| test public (Sat-Sun 11:32) | 0.428 | 0.436 | 0.464 |
| test private (Sun 11:32-Tue) | **0.456** | **0.473** | 0.476 |
| test by day Sat / Sun / Mon / Tue | 0.424 / 0.439 / 0.455 / 0.467 | 0.431 / 0.454 / 0.471 / 0.486 | |

* The trend is monotone in time: every day further from Sep 19 is colder.
* Consequence 1: holdout gains overstate test gains for features that act as warm priors.
  * Co-visitation (`--xt cv`): +0.0032 val vs +0.0092 holdout (nn_report.md:195).
  * Neighbour context (`cvnd`): +0.0005 vs +0.0017 (nn_report.md:274).
  * prew: in-window holdout +0.00125 +- 0.00049 vs val +0.00106 +- 0.00074 (`a5_prew.py`).
* Consequence 2: val itself overstates private gains for the same features.
* Impact: shown by OVERFIT-2. Selection among pre-declared member sets on the holdout (`holdout_eval.py`) inherits this bias whenever the sets differ in warm-prior inputs (e.g. nn_final5 with or without cvnd_prew members).

### OVERFIT-2 (medium, verified): prew_* was accepted below the protocol threshold, and its gain vanishes under the private mix

Protocol (CURRENT_STATUS sec. 4): delta > 2 SE, >= 2 seeds; new train-stage inputs must improve train-only val. Measured on the actual predictions (`a5_prew.py`, `a8_privw.py`, `a11_rob_delta.py`, `a14_day.py`), `nn_v12_prew3` vs `nn_v8_nd_e3`, seed 23 only:

| measure | delta | SE | z |
|---|---|---|---|
| val, all feeds | **+0.00106** | 0.00074 | 1.4 |
| holdout, all | +0.00084 | 0.00042 | 2.0 |
| holdout Sep 6-8 (prew NaN on train rows) / in-window | -0.00032 / +0.00125 | 0.0008 / 0.0005 | |
| val Tue / Thu | +0.00009 / +0.00212 | 0.0010 / 0.0011 | |
| val feeds whose positives are all LUO-warm (n=9,118) | +0.00268 | 0.00102 | 2.6 |
| val feeds whose positives are all LUO-cold (n=3,422) | **-0.00320** | 0.00133 | -2.4 |
| private-like weights (M x LUO-cold share) | +0.00023 | 0.00084 | 0.3 |
| private-like weights (M x never-train share) | -0.00119 | 0.00113 | -1.1 |
| private-like weights (M x covis-pop0 share) | +0.00079 | 0.00080 | 1.0 |

* prew_* is NaN on LUO-cold candidates by design (`pretrain/make_prew.py`). The model learns to boost LUO-warm items, which costs cold positives.
* Private has the highest cold share (OVERFIT-1).
* The status file says "PASSED paired acceptance". By the stated rule it passes only on the holdout, at exactly 2.0 SE with one seed. It is not clean on val (1.4 SE).
* Impact: half of the nn_final5 members (`v12_allprew_s83/s84/s85`) use `--xt cvnd_prew`. They are probably harmless as ensemble diversity, but they are not a demonstrated gain for private.
* Fix: treat prew as unproven.
  * When forming nn_final5, compare the with- and without-prew member sets on val, on private-like weights and on the cold-positive segment, not on the holdout alone.
  * Or keep the prew members at a pre-declared equal share and make no claim about them.

### OVERFIT-3 (medium, verified): calib.py (M-only reweighting) is biased low; a cold-share covariate fixes most of the bias

I reconstructed v5 and v10 exactly from their components. Test ranks match the submitted parquets (rank agreement 1.0000; `a7_calib.py`; v5 uses the quarantined `stk2_res2_r3`, for analysis only).

| | val | calib public (M only) | public (M x LUO-cold, 6 bins) | **actual public** | private (M only) | private (M x LUO-cold) |
|---|---|---|---|---|---|---|
| v5 | 0.38692 | 0.37896 | 0.38031 | **0.3802** | 0.38153 | 0.38455 |
| v10 | 0.39399 | 0.38592 | 0.38721 | **0.38786** | 0.38856 | 0.39172 |

* Per-feed val NDCG depends on a feed's cold-candidate share in a U-shape (quintiles 0.420 / 0.375 / 0.376 / 0.365 / 0.434), and test feeds are colder. M-only weights miss this.
* Caveat: the covariate was chosen after seeing 2 public points, and the result depends on the covariate (`a10_calib_rob.py`):

  | covariate | v5 / v10 public | v10 private |
  |---|---|---|
  | never-train | 0.3832 / 0.3898 (overshoots) | 0.396 |
  | covis-pop0 | 0.3796 / 0.3867 | 0.390 |
  | LUO-cold | (above) | 0.392 |

  Treat the private level as about 0.389-0.396. The relative ordering of candidates is what matters for the final pick.
* The 3h claim "raw val-public gap is shrinking" is not significant. The paired public SE of a submission-to-submission delta is about 0.0010-0.0014 (`a4_lb.py`). Each step changes the residual by less than that.
* Part of v10's better transfer is a cold-mix effect: the fine-tuned half gains on cold feeds. It is not evidence that tuning is free of optimism.

### OVERFIT-4 (medium-low, verified): the fine-tune recipe was tuned on 2-epoch members and over-trains the 10-epoch members

* `nn/ft_list.sh:9` hard-codes `--lr 5e-5 --epochs 6`, and the saved prediction is the snapshot mean over epochs 1..6 (`finetune.py:269-283`). This recipe was chosen on val OOF for 2-epoch stage-2 models (nn_report.md:98).
* Per-epoch OOF of the nn_final4 members, epoch 0 (train-only) -> 6 (`nn/logs/v9_e10*_n2cp_ft.log`):

  | member | epoch 0 | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | epoch 6 |
  |---|---|---|---|---|---|---|---|
  | v9_e10 | 0.3914 | 0.3922 | **0.3923** | 0.3917 | 0.3912 | 0.3903 | 0.3884 |
  | s61 | 0.3913 | **0.3931** | 0.3925 | 0.3917 | 0.3906 | 0.3895 | 0.3881 |
  | s62 | 0.3907 | 0.3917 | **0.3917** | 0.3908 | 0.3905 | 0.3892 | 0.3874 |
  | s63 | 0.3916 | **0.3919** | 0.3908 | 0.3903 | 0.3899 | 0.3881 | 0.3867 |

  Snapshot 1..6 gives 0.39207 / 0.39168 / 0.39108 / 0.39006. For s63 that is below its train-only model.
* Honest nested estimate (`a12_epoch.py`): for each fold, choose the epoch on the other 4 folds' curves and score it on the held-out fold. This gives 0.39209 / 0.39310 / 0.39138 / 0.39192, i.e. **+0.0009 per member over the snapshot recipe**, without in-sample selection.
* The fine-tune alone is not significant for nn_final4: +0.00041 +- 0.00073. The 0.5 train-only mix (+0.0012 +- 0.0005 over train-only) partly compensates, because it acts like adding epoch 0 to the snapshot.
* nn_final5 risk: nnx found that Muon models over-fit faster in fine-tune (nnx_exp_optim.md:131). fta (cos + EMA) was validated only on 2-epoch Muon models, and with `--epochs 4` there (ft_list.sh defaults to 6 unless FTARGS overrides it).
* Fix (pre-declare before seeing nn_final5 OOF): fta with `--epochs 2`, or nested epoch selection (see Ideas). The pattern holds in 4 of 4 members and in most folds, so following it is not noise-chasing.

### OVERFIT-5 (low, verified): fine-tune-related deltas vary between the two val days more than the per-feed SE suggests

From `a14_day.py`:

| delta | Tue | Thu | Thu - Tue (z) |
|---|---|---|---|
| ft - trainonly (nn_final4) | -0.00092 +- 0.0010 | +0.00187 +- 0.0011 | +0.0028 (1.9) |
| nn_final4 ft - nn_final3 ft | +0.00164 | +0.00583 | +0.0042 (**2.6**) |
| nn_final4 trainonly - nn_final3 trainonly | +0.00440 | +0.00359 | -0.0008 (-0.5) |
| mix(0.5) - trainonly | +0.00068 | +0.00178 | +0.0011 (1.2) |
| mix(0.5) - ft | +0.00159 | -0.00009 | -0.0017 (-1.7) |

* Train-only deltas are stable across days. Fine-tuned deltas are not: the fine-tune adapts to the val period, and private is a different 3-day period.
* The 0.5 mix is the most stable choice (positive on both days and under every private-like weighting: +0.0012..+0.0015 over train-only, about 2.2-2.5 SE).
* Do not move toward the fine-tune-only model on val evidence.

### OVERFIT-6 (low, estimated): inventory of val-label decisions and selection optimism

Measured SEs:
* Fine-tune and blend variants: 0.0004-0.0007 (ft - to 0.00073, mix - ft 0.00050, v10 - mix 0.00042).
* Train-only variants: 0.0006-0.0008.
* User-clustered SE equals the naive SE (0.00047 vs 0.00048). 13,715 of 13,740 non-zero val users have a single feed, and user 0 (658 feeds) is one cluster (`a1_blend.py`).

Selection optimism for picking the best of K equally good options with paired SE s is about s/sqrt(2) x E[max of K N(0,1)]. Here E[max] = 0.56 (K=2), 0.85 (K=3), 1.27 (K=6), 2.2 (K=44).

| decision (path to sub_v10) | selected on | K / evidence | est. optimism in v10 |
|---|---|---|---|
| Round-1 architecture/loss (sim_bias, aux, K9, K1) | val + holdout, effects 0.002-0.0046 | large effects | ~0 |
| `cv` train-stage inputs | val + holdout (+0.0032 / +0.0092) | large | ~0 |
| in-model rank/z (`cvraw`) vs precomputed | **val only** (+0.0011 +- 0.0006, holdout equal) | K=2 | ~0.0002 |
| `cvnd`, xt_drop q=0, epochs 3/5/10 | holdout (and val) | holdout-led | ~0 on val |
| member set (four 10-epoch members) | holdout among 4 pre-declared sets | | 0 on val |
| fine-tune lr 5e-5 / 6 epochs / snapshot 1..6 | val OOF (+0.0007, two changes at once) | K=2-3 | ~0.0001-0.0002 (enters v10 at weight 0.5) |
| branch lr x10 vs x3, add_random, 2 seeds, K1, ALS on/off | val OOF, all within +-0.0007 | ~4-5 near-null binary choices | ~0.0003-0.0005 (x0.5) |
| n2c_pre vs n2ca_pre | val, one member (0.38695 vs 0.38612) | K=2 | ~0.0001 |
| pre_* in the branch | val +0.0015 +- 0.0005 (3 SE) | real effect | <0.0001 |
| feature screening that produced the seq_/ctx_ files on disk | val GBDT harness (5 seeds + time split) | family gain +0.007 at 5 SE | small (enters both stages) |
| w_trainonly in {0, 0.2, 0.5} -> 0.5 | val | K=3; **bootstrap optimism +0.00012**, picked 64% of resamples | 0.0001 |
| +0.1 z(pre_final) | val (v7, 2 SE); now -0.00002 +- 0.00042 | | 0 |
| epoch choice (`best_e`) | logged only, not used for saved preds | if used: +0.0003 mean over 108 runs (LOFO) | 0 |
| calibration | label-free reweighting | | 0 |

* Total: about 0.0005-0.0015 val NDCG, mostly from near-null fine-tune choices that enter v10 at about half weight.
* The w_trainonly weight was re-chosen in every round (0.5 -> 0.2 -> 0 -> 0.5; nn_report.md:235, 281, 292; CURRENT_STATUS 3e). That flip-flop shows the optimum is flat and noise-dominated, so the choice costs little in optimism and buys little.
* Bootstrap optimism of other weight grids (`a1_blend.py`): 6 options (w x pre) +0.00025, a fine 11x4 grid +0.00043. A split-half check (choose on Tue, test on Thu and vice versa) shows a fine-grid pick does no better than the fixed v10 weights on the other day (0.39393 vs 0.39435; 0.39366 vs 0.39360).
* Public evidence:
  * v5 -> v10: val +0.0071, actual public +0.0077, calib-predicted +0.0072.
  * Residual vs the cold-aware calibration +0.0007; the SE of (public delta - val delta) is about 0.0018 (`a4_lb.py`).
  * Accumulated optimism v5 -> v10 is therefore < ~0.003 at 95%. There is no sign of it.

### OVERFIT-7 (low, verified): "train-only" means no val-label fitting, not "val-blind"

* The train-stage input set (`seq_cv_*`, `ctx_nd_*`, `ctx_hn/hs_*`) was screened on val with the GBDT harness. Pruning to KEEP lists happened in `feats_v2/*/KEEP.txt`.
* `cvraw` was accepted on val alone, and round-1 architecture choices were compared on val.
* nn_final4_trainonly (0.39281) therefore carries a little selection optimism (<= ~0.0005). Calling it "no val labels" is accurate only for fitting.
* No action needed. Just do not treat its val score as a fully held-out estimate.

### OVERFIT-8 (info, verified): stale docstring in finetune.py

* `finetune.py:4-6` says "the epoch count is chosen by the OOF NDCG ... and the test prediction is the mean of the 5 fold models at that epoch".
* The code saves the snapshot mean over epochs 1..E (default), the final EMA weights (`--ema`), or best_e only with `--select best` (lines 273-286).
* So the saved predictions carry no global-epoch selection optimism. If `--select best` were used, the optimism would be about +0.0003 (OVERFIT-6 table).

### OVERFIT-9 (info, verified): the user-0 LUO hole is negligible for the holdout

* `src/features_luo.py:23` restricts the later-occurrence removal to `user_id > 0`. user-0 rows keep later user-0 occurrences. This follows the rule that user 0 is not a user.
* Holdout exposure (`a3_user0.py`, `a9_holdcold.py`):
  * user-0 feeds are 1,536 of the 30K holdout feeds (5.1%).
  * 3.1% of their candidate rows appear in a later user-0 history within 24 h. Those rows have a positive rate of 3.45% vs 1.08% (3.2x lift).
  * Control: holdout non-zero rows matched against later histories of 124K random other users give a 1.38x lift.
  * The share of positives with such a hit is similar: 9.2% (user 0) vs 8.6% (control).
* So there is some same-person signal in the user-0 pool, but on at most ~0.1% of holdout feeds. Impact < 0.0005; no action.

### OVERFIT-10 (info, verified): all 658 user-0 val feeds fall in fine-tune fold 0

* Fold sizes are [2885, 2885, 2885, 2884, 2884], and fold 0 holds all user-0 feeds (`a2_itemmem.py`). GroupKFold treats user 0 as one group.
* The OOF for user-0 feeds comes from a model that never fine-tuned on user-0 feeds. Test user-0 feeds (1,179) are scored by 5 fold models, 4 of which saw them. The OOF is mildly pessimistic for user-0 feeds (ft - to on user-0 feeds: +0.0028 +- 0.0034).
* This is conservative and acceptable. Splitting user 0 by feed (as stack2.py does) would be less conservative.

### OVERFIT-11 (info, verified): user 0 is pooled in u_tr_ctr / u_tr_feeds for val/test

* `src/features.py:265-278, 354-357` computes `user_stats()` with user_id 0 as an ordinary user. In the fine-tune branch, user-0 val/test rows therefore get the pooled user-0 train CTR and feeds = 123,904.
* This conflicts with the rule "exclude user 0 from user-level stats". In practice it is a constant is_user0 indicator, identical for val and test, so it is harmless.
* `tte_*` handles user 0 correctly (NaN).
* Train rows: `u_tr_*` / `it_tr_*` contain the own feed's label, but train-fit GBDTs exclude both (`LEAKY_ON_TRAIN`, `src/train_lgb.py:13`) and the NN train stage does not load them.

---

## 2. Checked and correct

* **Fine-tune CV (nn/finetune.py):**
  * `GroupKFold(n_splits=5).split(np.arange(va.n), groups=va.user_id)` (lines 180-183).
  * Held-out-fold scores only (`predict(..., ho_f)` returns zeros elsewhere, `common.py:43-55`).
  * Test = mean of the 5 fold models.
  * Extra-feature standardisation is label-free (`data.py:204-220`).
  * Per-fold pseudo-label teachers (rejected N6) were fold-safe.
* **Fold coincidence across val-CV scripts:**
  * `nn/finetune.py`, `src/cv_val.py` and `pretrain/finetune_val.py` use identical folds (GroupKFold(5), unshuffled, feeds.parquet order, raw user_id).
  * `src/stack2.py`, `pretrain/stack2_cpu.py` and `src/eval_family.py` use shuffled per-seed folds (stack2 also splits user 0 by feed).
  * The folds differ, but no second-level model consumes a val-OOF output. Every stk2/prestk json lists only train-only score inputs (`nn_*_trainonly`, `xgb_x300_luo`, `cb/lgb_tr300k`, `xgb_rand50`). So there is no nested leakage.
  * The only full-val fit is the 2-parameter base-margin logistic regression in stack2.py:60, which is negligible.
* **All 51 n2c_pre features actually loaded** (log `v9_e10_n2cp_ft.log`) are label-free with respect to val:

  | features | source |
  |---|---|
  | `it_cand_cnt_train`, `it_tr_coec(+rk/z/gap)`, `it_tr_ctr`, `it_tr_pos`, `u_tr_ctr`, `u_tr_feeds` | train labels only (`src/features.py:229-357`) |
  | `hour`, `centrality` | feed-internal |
  | `it_hist_all_luo(+rk/z/gap)`, `it_hist_et1..10_luo` | history counts minus same-user later occurrences |
  | `seq_cv_invd20(+rk)`, `seq_cv_pop`, `seq_cv_norm` | causal co-visitation over histories; same user skipped (`feats_v2/seq_/feat_covis.py`) |
  | `seq_max_all`, `seq_top3`, `seq_argmax_hpos` | own history plus embeddings |
  | `ctx_nd_*` (10) | within-feed embedding graph |
  | `pre_*` (9) | self-supervised encoders; final checkpoint, not selected on val (`pretrain/train.py:423`) |

  * Label reads in `feats_v2/seq_` exist only in `single_ndcg` (evaluation).
  * No val-label feature exists on disk except the quarantined `x_lh_*`. `tte_*` and `trd_*` use train labels only.
* **User overlap** (`overlap.py`; non-zero users):

  | overlap | users | feeds |
  |---|---|---|
  | val ∩ test | 92 | 94 val feeds (0.65%), 94 test feeds (0.33%; 29 public / 65 private) |
  | val ∩ train | 4,369 | 4,386 val feeds (30.4%) |
  | test ∩ train | 8,262 | 8,328 test feeds (29.1%) |

  * Within splits: 25 val users have 2 feeds; 102 test users have 2 feeds and 3 have 3.
  * User-specific transfer from the val fine-tune to test is bounded by 0.33% of test feeds. It is legitimate anyway, because val precedes test (earlier information).
* **No item-level memorisation optimism across folds** (`a2_itemmem.py`):
  * 16% of val rows show an item that also appears in another fold, but only 0.67% of val positives have an item that is positive in another fold. Positive rate: 2.9% for rows whose item is positive in another fold, 1.5% for rows shown elsewhere but never positive.
  * For comparison, 12.7% of test rows show a val item and 0.33% show a val-positive item.
  * The fine-tune gain is not concentrated on those feeds: -0.003 +- 0.009 on 146 feeds vs +0.0005 elsewhere.
* **Holdout validity:**
  * Definition: rng 12345 over p_random > 0.9 and M >= 50. `train_prandom.npz` dates from 02:53, before every holdout run.
  * Excluded from main training (train.py:159-164), stage 2 (`stage2_ft_v3.sh` passes `--holdout 30000`; log confirms), the cooldown pool (train.py:254-257) and feature_stats.
  * The p_random detector is label-free: rank correlation of position with a label-free position proxy plus gap share (`eda/labels_position/04g/04h`).
  * Train-stage inputs (`cvnd`, `cvnd_prew`) use no labels. `it_tr_*` / `u_tr_*` never enter the train stage.
  * Holdout-selected 10-epoch checkpoints sit at or near the last step (98350, 98350, 85000, 97500), so checkpoint-selection bias is negligible.
* **sub_v10** = 0.5 z(nn_final4_val_oof/test) + 0.5 z(nn_final4_trainonly) + 0.1 z(pre_final), per-feed z. Reconstruction reproduces val 0.39399 and the submitted test ranks exactly.
* **Blend-weight protocol:** 3 coarse values, equal weights, bootstrap optimism +0.0001: compliant. The 0.1 pre term is neutral but harmless.
* **Public LB use:** the only LB-driven decision so far was removing the later-history boost (1 bit, correct).

---

## 3. Recommended protocol for the final 2 private submissions

1. **Pre-declare the candidate set now**, before nn_final5 numbers arrive:
   * A = v10 (public-confirmed 0.38786).
   * B = nn_final5 mix: 0.5 z(fine-tuned) + 0.5 z(train-only), members from pre-declared sets.
   * C = equal z-mean of A and B.

   Do not add new weights or variants afterwards.
2. **Evaluate every candidate the same way:** paired per-feed delta vs v10 on
   * val,
   * val Tue and val Thu separately,
   * private-like importance weights (M x per-feed LUO-cold share; label-free; `a8_privw.py`),
   * the cold-positive and warm-positive segments,
   * and, for train-only parts, the holdout.
3. **Pick #1:** the candidate with the highest val score among those that beat v10 by > 2 SE on val and are >= 0 under private-like weights and on both days. Otherwise keep v10.
4. **Pick #2 (hedge):**
   * If pick #1 is B, submit C. A cross-generation equal blend is rarely worse than its better member and lowers variance.
   * If pick #1 is A, submit the candidate (B or C) that is within 1 SE of v10 on val and differs most from v10 on the cold/warm axis.

   Never submit two near-duplicates (per-feed delta SD < ~0.03).
5. **Do not submit:**
   * `*_lh` or anything built on quarantined features;
   * fine-grid weight searches on val;
   * fine-tune-only ensembles (day-unstable, OVERFIT-5);
   * prew-only member sets chosen on the holdout (OVERFIT-1/2);
   * anything chosen by public-LB differences < ~0.003 (public level SE 0.0026; paired delta SE 0.0010-0.0014).

   Use the public LB only as a gross-failure check: the actual delta should fall within about 2 paired SE of the val-predicted delta.

## 4. Ideas

1. **Nested epoch selection in the fine-tune:** for fold k, pick the epoch (or snapshot window) on folds != k. It is honest and worth about +0.0009 per 10-epoch member (OVERFIT-4). A pre-declared shorter recipe also works (fta `--epochs 2`, or snapshot 1..2).
2. **Cold-aware evaluation everywhere:** report deltas under private-like weights (M x LUO-cold share) and replace calib.py's M-only weights. This is label-free and fixes most of the +0.0012-0.0019 public residual.
3. **Cold-matched holdout:** reweight holdout feeds by their LUO-cold share to the test mix, or carve a "late" random-like holdout from Sep 17-19. The current holdout is 22% LUO-cold vs 46% in private.
4. **Repeated fold assignments:** fold-level scores range from 0.384 to 0.399. Averaging fine-tunes over 2 different GroupKFold shuffles (with the same user grouping) would reduce OOF noise and make fine-tune decisions less noisy. It costs 2x fine-tune time.
5. **Cold-share-dependent mixing:** if the fine-tune vs train-only mix is ever revisited, let w depend on the per-feed cold share (the fine-tune gains on cold feeds and loses on warm ones). Fit it only inside the CV folds (nested), not on full val.
6. **Warm-prior features need a cold check:** for any future warm-prior train-stage input, require no significant loss on cold-positive feeds in addition to the 2-SE val rule. Private is the coldest period.
