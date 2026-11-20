# Pretrained user-history sequence encoder (self-supervised, all histories)

Status: final (sections 1-7: val/test encoders, 04:45-07:25 UTC; section 8: train-stage windowed encoders on GPU N, 08:40-11:15 UTC). Author: pretraining agent. Date 2026-11-11.

## TL;DR
* A SASRec/ARGUS-style causal transformer pretrained on all 2.59M leak-safe history sequences (463.6M events; train + val + test snapshots) to predict the user's **next new feed-originated open** (event_type 1 & event_source 1, the history footprint of 85% of feed positives), with a hybrid item tower (int8 content emb MLP + category + a 40M-row learned item-ID table). d=384, 4 layers, 8M dense params; 2 epochs = 22 min on one GPU.
* Zero-shot val NDCG of a single head score: **0.357-0.3585** (v2-v4s1); final 2-config score `pre_final` **0.3590**; vs maxsim32 0.349, user_dot 0.3435, ALS 0.331. Removing the ID part costs 0.004-0.005.
* Deliverable = 9-feature family `data/feats/{val,test}/pre_*` (mean of configs v3 and v4; val side from the twins pretrained without test snapshots, see section 1) + score `preds/pre_final_{val,test}.npy`. Incremental value on val, all clean (no same-user later snapshot is ever seen; leak audit flat; clean-subset numbers equal overall):
  * harness (`eval_family.py --family pre_`, base 0.3634): **+0.0037 +- 0.0008** (0.3634 -> 0.3671);
  * inside the current residual stacker (stk2_res4_clean config, CPU copy, 2 x 3 seeds): stacker OOF **+0.00075 +- 0.00044**; its 0.5/0.5 blend with nn_final2 (0.3863): **+0.00044 +- 0.00029**;
  * plain z-blend on the best blend 0.5 z(nn_final2) + 0.5 z(stk2_res4_clean) = 0.3865: `+0.1 z(pre_final)` **+0.00082 +- 0.00038** (clean +0.00087; CV-chosen w gives the same); on nn_ens_stage1 (0.3811): +0.00083 +- 0.00041.
* The incremental signal is the **ID (collaborative) part**: content-only scores (`fe_noid`) add ~0 on top of the NN, which already learned the content mapping from 3M labelled feeds. The gain is real but small (+0.0004..+0.0009 on the current best): this encoder is not the +0.008 the team needs; its value is as an extra input to the stacker / NN fine-tune branch, and as initialisation for a train-fed ID-aware ranker (not tried here).
* Fine-tuning the encoder on val labels (5-fold, listwise) made it worse (0.3536 -> 0.350-0.352); not used.

## 1. Leakage protocol
Same-user later snapshots (CURRENT_STATUS.md section 2):
* No snapshot taken after a val or test feed of the same user is in any pretraining corpus. For every non-zero val/test user only the **earliest** val/test snapshot is used; all train snapshots precede all val/test feeds. The 92 val&test users contribute only their val snapshot, the 25 val and ~105 test multi-feed users only their first. So neither the transformer nor the ID embeddings ever see a co-occurrence between a user's earlier history and items from the same user's later snapshots.
* A val/test feed is scored from its own history snapshot only.
* Leak audit (method of section 2b): single-score NDCG on the 117 val feeds whose user has a later val/test feed vs the other 14,306: v3 fe 0.345 vs 0.357, v4 fe 0.342 vs 0.358, final score pre_final 0.348 vs 0.359 (no jump; a leaky feature such as the old als_exp5 showed 0.58 vs 0.33). Every blend number is also reported on the clean 14,306 feeds; they match the overall ones.
* Train-row outputs are leaky by construction (later train snapshots contain train positives) and are never produced.

Other users' future snapshots (transductive CF):
* Val feeds (Sep 22/24) would see up to ~7 days of later other-user snapshots (val + test) while test feeds see at most ~3 days. To keep the information horizon symmetric ("data up to the end of the feed's own split"), the final **val features come from encoders pretrained without any test snapshot** (`--exclude_test 1`: train + val snapshots), and the **test features from the same configs pretrained on train + val + test**. Raw-score features of the val-side model are moment-matched (mean/std on val, label-free) to its test-side twin; within-feed z features are unaffected.
* Effect measured on val (same config/seed, v3 vs v3A): harness +0.00376 vs +0.00371 (equal); blend on nn_ens_stage1 +0.00125 vs +0.00082, paired difference +0.0004 (SE 0.0003): at most a small optimistic bias for the all-snapshot model, removed in the final features.

## 2. Design
Code: `<repo>/pretrain/`
| file | purpose |
|---|---|
| `build_corpus.py` | leak-safe corpus (old -> new), next-target arrays, global int8 content table indexed by item_id, item counts (1 min) |
| `build_multi_targets.py` | next R=4 distinct new feed-open items per position ("dense" targets) |
| `seqenc.py` | model |
| `train.py` | pretraining (sampled softmax, mixed negatives, logQ), periodic zero-shot val eval, final val/test scoring; `--exclude_test`, `--vt_rep`, `--id_bias`, `--multi` |
| `score.py` | re-score val/test from a checkpoint |
| `evaluate.py` | zero-shot NDCG, leak audit, blend tests (vs nn_ens_stage1 or the current best blend), writes per-variant features |
| `assemble.py` | final `pre_*` family: val from `--val` models (moment-matched to `--calib` twins), test from `--test` models, mean over configs |
| `blend_report.py`, `stack_compare.py`, `stack2_cpu.py` | blend report of a score file; paired stacker comparison; CPU copy of src/stack2.py (writes preds/prestk_*) |
| `finetune_val.py` | 5-fold fine-tune of the encoder on val labels (negative result) |

**Corpus**: train = latest train snapshot of each non-zero user (2.45M) + every user-0 train feed (124K); val/test = earliest val/test snapshot per non-zero user + all user-0 feeds. Targets per position t: `fe` = first later event with et 1 & es 1 whose item is not in events <= t (candidates are never in their own history); `new` = first later event with a new item; dense variant samples `fe` uniformly among the next 4 such items. A learned BOS token is prepended (empty histories get a state; BOS predicts the first target).

**Model**: ItemTower(i) = MLP(L2-normalized int8 content emb; 128 -> 2d -> d) (learned vector if missing) + E_cat + P E_id[i] (items with corpus count >= 3: 40.3M rows, dim 64/128, ID-dropout 0.3, SparseAdam) [+ per-item scalar logit bias in v4]. Token = ItemTower + E_event_type + E_event_source; causal pre-LN transformer with RoPE (old -> new). Heads q_k = normalize(W_k h_t), k in {fe, new}; logit = scale <q_k, normalize(W_out ItemTower(i))>. Loss: sampled softmax per head (weights 1 / 0.5) over [positive, 6144 batch fe targets + 2048 batch new targets + 8192 uniform corpus items], logQ correction with the mixture proposal K1 p_fe + K2 p_new + K3/N, accidental hits masked. AdamW 1e-3 (warmup 300, cosine to 10%), bf16, fp32/TF32 logits, 131K padded tokens per batch.

**Exports per val/test candidate** (`preds/pre_<variant>_<key>_{val,test}.npy`): `fe`, `new` (head scores), `fe_noid`/`new_noid` (ID part zeroed), `hmax32`/`hmax8` (max cosine to the last 32/8 history items in the learned output space), `idmax32` (max cosine in the learned ID space; -2 if no ID), `ibias` (v4).

## 3. Variants, training curves, zero-shot val NDCG (14,423 feeds)
| variant | fe | fe_noid | new | new_noid | hmax32 | hmax8 | train time (GPU) |
|---|---|---|---|---|---|---|---|
| v1: d256 L3 (2.8M dense), ID dim 64, 1 ep | 0.3537 | 0.3496 | 0.3547 | 0.3505 | 0.3459 | 0.3481 | 9.3 min |
| v2: d384 L4 (8.0M dense), dense next-4 fe targets, 1.5 ep | 0.3578 | 0.3525 | 0.3583 | 0.3538 | 0.3461 | 0.3483 | 17.3 min |
| v3: v2 + ID lr 0.01 (was 3e-3) + val/test sequences x4, 2 ep | 0.3573 | 0.3536 | 0.3572 | 0.3538 | 0.3448 | 0.3467 | 21.5 min |
| v4: v3 + per-item ID logit bias + ID dim 128 | 0.3575 | 0.3534 | 0.3582 | 0.3549 | 0.3437 | 0.3461 | 43 min (GPU shared) |
| v3A: v3, no test snapshots (val side) | 0.3567 | 0.3540 | 0.3584 | 0.3560 | 0.3454 | 0.3469 | 40 min (GPU shared) |
| v4s1: v4, seed 1 (seed-noise check) | 0.3585 | 0.3543 | 0.3588 | 0.3548 | 0.3454 | 0.3471 | 29 min (GPU partly shared) |
| v4A: v4, no test snapshots (val side) | 0.3566 | 0.3535 | 0.3584 | 0.3552 | 0.3454 | 0.3469 | 21.6 min |
| **pre_final** = mean of z(fe), z(new) over the configs (val: v3A+v4A, test: v3+v4) | 0.3590 | | | | | | |

Curves (val fe zero-shot by step): v1 1000: 0.3474, 2000: 0.3525, 3000: 0.3536, 3752: 0.3537. v2 1500: 0.3500, 3000: 0.3552, 4500: 0.3561, 5628: 0.3578. v3 2000: 0.3511, 4000: 0.3556, 7866: 0.3573. v4 2000: 0.3516, 4000: 0.3572, 6000: 0.3569, 7866: 0.3575. Training sampled-softmax loss (fe head, logQ-corrected, 16K negatives) falls from 10.1 to 6.2 (v1) / 5.2 (v4); fraction of negatives beaten 0.95-0.98. Full curves: `logs/pretrain_<variant>.json`.

## 4. Incremental value per variant
### Harness (`src/eval_family.py`, paired 3-seed LGB val CV, base 0.3634)
| family | features | delta | SE |
|---|---|---|---|
| pre_v1_ | fe, fe_z, new, hmax32 | +0.00237 | 0.00075 |
| pre_v2_ | same 4 | +0.00265 | 0.00080 |
| pre_v2x_ | v2: fe, fe_z, new, new_z, hmax32, hmax8, fe_noid, id_contrib (= fe - fe_noid), idmax32 (NaN if no ID) | +0.00445 | 0.00083 |
| pre_v2x_ without idmax32 | 8 features | +0.00389 | 0.00083 |
| pre_v3_ | v3, the 9 features | +0.00376 | 0.00083 |
| pre_v3A_ | v3A (no test snapshots), the 9 features | +0.00371 | 0.00083 |
| pre_ (v3 config only: v3A val / v3 test) | the 9 features | +0.00428 | 0.00083 |
| **pre_ (final family: mean of v3, v4; val from v3A, v4A)** | the 9 features | **+0.00373** | 0.00084 |

### z-blends: `z(base) + w z(pre)`, w = 0.1 / 0.2 fixed, or chosen in 5 user folds (CV-w); best = 0.5 z(nn_final2_val_oof) + 0.5 z(stk2_res4_clean) = 0.3865; clean = 14,306 feeds without a later same-user feed
| score | zero-shot | later-user feeds (117) | vs nn_ens_stage1 w=0.1 | vs best w=0.1 | vs best w=0.1, clean | vs best w=0.2 | vs best CV-w |
|---|---|---|---|---|---|---|---|
| v1 fe | 0.3537 | 0.3453 | +0.00022 (0.00040) | -0.00003 (0.00038) | -0.00004 (0.00038) | -0.00110 (0.00054) | -0.00008 (0.00023) |
| v1 new | 0.3547 | 0.3456 | +0.00006 (0.00041) | -0.00011 (0.00037) | -0.00011 (0.00037) | -0.00050 (0.00054) | -0.00007 (0.00020) |
| v2 fe | 0.3578 | 0.3599 | +0.00100 (0.00042) | +0.00079 (0.00038) | +0.00081 (0.00038) | +0.00031 (0.00055) | +0.00071 (0.00042) |
| v2 new | 0.3583 | 0.3531 | +0.00072 (0.00041) | +0.00063 (0.00036) | +0.00065 (0.00037) | +0.00073 (0.00055) | +0.00057 (0.00049) |
| v3 fe | 0.3573 | 0.3448 | +0.00125 (0.00040) | +0.00097 (0.00038) | +0.00100 (0.00038) | +0.00065 (0.00055) | +0.00049 (0.00043) |
| v3 new | 0.3572 | 0.3350 | +0.00074 (0.00041) | +0.00089 (0.00038) | +0.00092 (0.00038) | +0.00052 (0.00055) | +0.00089 (0.00038) |
| v4 fe | 0.3575 | 0.3419 | +0.00126 (0.00041) | +0.00094 (0.00039) | +0.00100 (0.00039) | +0.00022 (0.00054) | +0.00094 (0.00039) |
| v4 new | 0.3582 | 0.3368 | +0.00085 (0.00041) | +0.00067 (0.00037) | +0.00071 (0.00038) | +0.00036 (0.00055) | +0.00067 (0.00037) |
| v3A fe | 0.3567 | 0.3515 | +0.00082 (0.00042) | +0.00046 (0.00038) | +0.00051 (0.00039) | +0.00028 (0.00056) | +0.00024 (0.00032) |
| v3A new | 0.3584 | 0.3480 | +0.00054 (0.00041) | +0.00030 (0.00038) | +0.00034 (0.00038) | +0.00029 (0.00056) | -0.00010 (0.00037) |
| v4s1 fe | 0.3585 | 0.3371 | +0.00084 (0.00042) | +0.00124 (0.00038) | +0.00129 (0.00038) | +0.00105 (0.00056) | +0.00080 (0.00042) |
| v4s1 new | 0.3588 | 0.3439 | +0.00045 (0.00041) | +0.00097 (0.00037) | +0.00100 (0.00037) | +0.00127 (0.00055) | +0.00103 (0.00050) |
| v4A fe | 0.3566 | 0.3444 | +0.00077 (0.00042) | +0.00103 (0.00038) | +0.00108 (0.00038) | +0.00021 (0.00057) | +0.00103 (0.00038) |
| v4A new | 0.3584 | 0.3379 | +0.00046 (0.00042) | +0.00074 (0.00037) | +0.00079 (0.00037) | +0.00056 (0.00056) | +0.00026 (0.00040) |
| **pre_final** | 0.3590 | 0.3479 | +0.00083 (0.00041) | +0.00082 (0.00038) | +0.00087 (0.00038) | +0.00059 (0.00056) | +0.00082 (0.00038) |

## 5. Inside the residual stacker (most relevant use)
`pretrain/stack2_cpu.py` = src/stack2.py on CPU (16 threads), stk2_res4_clean configuration (base margin z(nn_final2_trainonly), scores nn_final2_trainonly + xgb_x300_luo, its compact feature list minus 7 ctx_nd_* files that were missing on disk at the time, XGB eta 0.02, depth 4, 300 rounds, 3 seeds, GroupKFold by user), with vs without the 9 `pre_*` features; same seeds, paired; two seed sets (0-2 and 3-5):
| pre_ family | seeds | stacker OOF without -> with | delta (SE) | 0.5 z(nn_final2) + 0.5 z(stacker) | delta (SE) |
|---|---|---|---|---|---|
| v3 only (v3A val / v3 test) | 0-2 | 0.3850 -> 0.3857 | +0.00065 (0.00050) | 0.3863 -> 0.3872 | +0.00091 (0.00033) |
| v3 only | 3-5 | 0.3846 -> 0.3859 | +0.00133 (0.00050) | 0.3864 -> 0.3870 | +0.00060 (0.00034) |
| v3 only | mean of both | | +0.00099 (0.00045) | | +0.00075 (0.00029) |
| **final (v3+v4)** | 0-2 | 0.3850 -> 0.3857 | +0.00073 (0.00049) | 0.3863 -> 0.3867 | +0.00035 (0.00034) |
| **final (v3+v4)** | 3-5 | 0.3846 -> 0.3854 | +0.00078 (0.00050) | 0.3864 -> 0.3869 | +0.00054 (0.00033) |
| **final (v3+v4)** | mean of both | | **+0.00075 (0.00044)** | | **+0.00044 (0.00029)** |
Clean-subset deltas are +0.00005 higher in every row. The v3-only and final families differ by less than the noise; the final (2-config) family was fixed before these results and is the deliverable (choosing the better-looking one on val would be selection bias).

## 6. Final deliverable
* Features (float32, aligned with `data/flat/{val,test}/cand_*`): `data/feats/{val,test}/pre_fe, pre_fe_z, pre_new, pre_new_z, pre_hmax32, pre_hmax8, pre_fe_noid, pre_id_contrib, pre_idmax32` (NaN = candidate or recent history without an ID). Each is the mean over configs v3 and v4; **val** from v3A + v4A (no test snapshots; raw scores moment-matched on val to v3/v4), **test** from v3 + v4. Built by `python pretrain/assemble.py --val v3A,v4A --calib v3,v4 --test v3,v4`. Earlier per-variant feature files were moved to `data/pretrain/feats_archive/{val,test}/` so that `--family pre_` evaluates exactly this family.
* Score: `preds/pre_final_{val,test}.npy` = mean over the two configs of 0.5 (z(fe) + z(new)); val zero-shot 0.3590.
* Per-variant scores: `preds/pre_<variant>_<key>_{val,test}.npy` for variant in {v1, v2, v3, v4, v4s1, v3A, v4A} and key in {fe, new, fe_noid, new_noid, hmax32, hmax8, idmax32, ibias (v4*)}. Note: v3A/v4A test files are not meant for use (those models have no test snapshots); the v3/v4/v4s1 val files carry the small "other users' future" optimism described in section 1.
* Checkpoints: `data/pretrain/ckpt_<variant>.pt` (10-21 GB each, incl. the ID table); corpus arrays in `data/pretrain/` (135 GB total incl. checkpoints).
* Stacker outputs of the measurement runs: `preds/prestk_*` (CPU copy; not meant as submissions).
* Recommended use: add the 9 `pre_*` features to the residual stacker (and/or the NN fine-tune `--extra` branch). Rebuild for a new corpus: `build_corpus.py` -> `build_multi_targets.py` -> `train.py` (twice per config: with and without `--exclude_test 1`) -> `assemble.py`.

## 7. Negative / neutral results
* Val fine-tune of the encoder (5-fold GroupKFold by user, listwise softmax, lr 1e-4 or 2e-5, ID table frozen): OOF 0.3536 (epoch 0) -> 0.350-0.352 after 1-5 epochs. 11.5K labelled feeds are too few to adapt the pretrained scorer; the NN ranker is the better consumer of labels.
* Larger ID lr + val/test oversampling (v3), ID bias + ID dim 128 (v4): same zero-shot NDCG as v2; blend gains differ by less than the noise (paired differences v4-v3 0.0000 +- 0.00025, v3-v2 +0.0002 +- 0.0003).
* Seed noise (v4 vs v4s1, same config): zero-shot fe 0.3575 vs 0.3585; blend delta (w=0.1) vs nn_ens_stage1 +0.00126 vs +0.00084, vs best +0.00094 vs +0.00124 (paired differences 0.0003-0.0004). Variant-to-variant differences above are of the same size, i.e. not resolvable; the v3 vs v3A gap (0.0004-0.0005) is also within seed noise.
* Stacker test repeated with other XGB seeds/fold shuffles (seeds 3-5): stacker OOF 0.3846 -> 0.3859 (+0.00133 +- 0.00050), 0.5/0.5 blend 0.3864 -> 0.3870 (+0.00060 +- 0.00034). Mean over the two seed sets: stacker +0.0010, blend +0.00075.
* `ibias` alone 0.2955 (item appeal as a feed-open target beyond logQ) and `idmax32` alone 0.297 are weak single features; idmax32 helps inside LGB (+0.0006 harness).
* The blend gain is concentrated on long histories (v2 vs nn_ens_stage1: H >= 64 +0.0016 +- 0.0005, H < 64 -0.0015 +- 0.0015, user-0 feeds -0.003 +- 0.003), so a stacker that sees H is the right consumer.
* Correlation of z(id_contrib) with z(als_exp5) is 0.23 and with z(seq_cv_invd20) 0.11: the encoder's ID signal overlaps only partly with the existing CF features.

## 8. Train-stage pre_* from time-windowed encoders (leak-free features for train feeds)
Goal: give train rows the same 9 `pre_*` features so the NN can use them in its train stage. The section 1-6 encoders saw each train user's latest snapshot (= their earlier feeds' positives), so their train-row outputs are leaky.

**Construction** (`pretrain/build_corpus_window.py`, mirrors the val/test rule). Window W_k = [T_k, E_k) in MSK days. corpus_k =
* background: for each non-zero user the **latest** train snapshot with feed time < T_k, plus every user-0 train feed < T_k;
* in-window: for each non-zero user the **earliest** train snapshot inside W_k, plus every user-0 feed inside W_k.

So a window feed never sees a snapshot of its own user taken after it (only its own snapshot or that user's earlier in-window one).

Windows: W1 [Sep 9, Sep 12), W2 [Sep 12, Sep 17) (no data Sep 15-16 apart from 42 boundary feeds), W3 [Sep 17, end) (incl. 34 feeds stamped Sep 20 00:00). Sep 6-8 (864,519 feeds) stay NaN.

Per window, the two final configs v3 and v4 are pretrained on corpus_k, on GPU N only, with the same hyper-parameters (2 epochs, `--vt_rep 1`, see the mismatches below). Each model scores the window's train feeds (`score_window.py`). Its raw-score features are moment-matched (label-free, mean/std on a 60K-feed sample of the window) to the full-corpus twin (v3/v4), the same alignment used for the val side. The 9 features are then built and averaged over the 2 configs as in `assemble.py` (`assemble_train.py`). Window checkpoints were deleted after scoring.

| window | feeds scored | candidate rows | corpus: background seqs + in-window seqs | tokens | ID items (cnt >= 3) | train time v3 / v4 (GPU N) | scoring per model |
|---|---|---|---|---|---|---|---|
| W1 Sep 9-11 | 816,624 | 62.6M | 796,342 + 750,809 | 285.0M | 27.1M | 12.6 / 13.3 min | 1.3-1.5 min |
| W2 Sep 12-14 | 711,301 | 56.0M | 1,465,749 + 659,957 | 389.1M | 34.8M | 25.0 (GPU shared) / 17.7 min | 1.2-1.5 min |
| W3 Sep 17-19 | 758,764 | 59.1M | 2,003,488 + 702,524 | 491.9M | 41.7M | 21.5 / 22.3 min | 1.1-1.2 min |

Coverage: 72.1% of the 246.7M train candidate rows (Sep 6-8 are NaN). Inside the windows, NaN rate is 0 for all features except `pre_idmax32` (NaN when the candidate or all recent history items lack an ID): 38.5% / 36.5% / 35.6% in W1/W2/W3, vs 55.3% on val and 59.5% on test (train candidates are denser in ID coverage, see the mismatches below).

**Leak checks** (`logs/pretrain_train_windows_checks.json`, `logs/pretrain_train_windows_later_control_{1,2,3}.json`).

(1) Single-feature NDCG of the main score (`preds/pre_final_train.npy` = mean over configs of 0.5 (z(fe) + z(new)), the train analogue of pre_final) on random-like train feeds (p_random > 0.9) inside the windows. Train feeds have a different M and positive mix, so raw levels are not comparable with val; leak-free reference features are given for calibration:
| | pre_final | maxsim32 | cos_u | f1 | pre / maxsim32 |
|---|---|---|---|---|---|
| val (14,423 feeds) | 0.3590 | 0.3478 | 0.3417 | 0.3279 | 1.032 |
| W1 random-like (87,994) | 0.4409 | 0.4207 | 0.4096 | 0.4082 | 1.048 |
| W2 random-like (83,011) | 0.4392 | 0.4198 | 0.4097 | 0.4078 | 1.046 |
| W3 random-like (94,830) | 0.4339 | 0.4204 | 0.4129 | 0.4065 | 1.032 |

The train/val ratio is about 1.21-1.23 for every feature (pre 1.21-1.23, maxsim32 1.21, cos_u 1.20-1.21, f1 1.24-1.25), so pre_* is not anomalously strong on train rows. Contrast on each window's 60K-feed sample (random-like feeds): the leaky full-corpus models score W1 0.4573 / W2 0.4457 / W3 0.4336, vs the windowed features' 0.4405 / 0.4366 / 0.4301. The windowed construction removes that leak (+0.017 / +0.009 / +0.004).

(2) Split by whether the feed's user has another feed later in the same window (NDCG; gap = has-later minus no-later, SE in brackets):
| | all feeds: pre_final gap | maxsim32 | cos_u | f1 | random-like: pre_final gap | maxsim32 | cos_u | f1 |
|---|---|---|---|---|---|---|---|---|
| W1 (55.7K has-later) | +0.0132 (0.0009) | +0.0029 | +0.0060 | -0.0013 | +0.0069 (0.0032) | -0.0025 | -0.0038 | -0.0084 |
| W2 (43.4K) | +0.0149 (0.0011) | +0.0042 | +0.0055 | -0.0023 | +0.0177 (0.0037) | +0.0033 | +0.0031 | -0.0025 |
| W3 (48.0K) | +0.0268 (0.0010) | +0.0066 | +0.0056 | -0.0014 | +0.0356 (0.0033) | +0.0079 | +0.0098 | +0.0002 |

pre_* has a larger has-later gap than the simple reference features. **Correction (8b/8c): this gap was largely the content-availability leak of section 8b.** The controls below were scored with the same leaky content table, so they could not rule it out. With the fixed table (8c) the W1 within-window gap drops from +0.0132 to +0.0045 and the any-later-feed gap is -0.0004 (maxsim32: -0.0007). The original reading follows:
* **W1 background-only control** (`w1bg_v3`: same config, pretrained only on the W1 background, i.e. with NO in-window snapshot at all, so it cannot carry any in-window same-user information): has-later gap +0.0134 (all) / +0.0102 (random), vs +0.0136 / +0.0072 for the windowed w1_v3. The content-only score `fe_noid` shows the gap in both models too (+0.0169 vs +0.0174). The gap is the same without in-window data.
* **W3 background-only control** (`w3bg_v3e1`: same config, 1 epoch because GPU N was shared 3 ways, trained only on the W3 background, i.e. data <= Sep 14, no W3 snapshot at all): has-later gap **+0.0273 (all) / +0.0402 (random)**, vs +0.0264 / +0.0343 for the windowed w3_v3. Content-only `fe_noid`: +0.0343 / +0.0460. A model that has never seen any W3 data shows the same, even slightly larger, gap. The larger W3 gap is therefore a property of W3's returning users (the reference features' gaps also grow in W3: maxsim32 +0.008, cos_u +0.010 on random-like feeds), not leakage.
* By construction no same-user snapshot later than the scored feed is in corpus_k (at most the user's own or an earlier in-window snapshot). Same-user same-timestamp feed pairs (128-s timestamp granularity, where "earliest" is ambiguous) are only 136-164 feeds per window.

(3) Per-window coverage and NaN rates: see the table above.

**Diagnostics.**
* On train rows the ID part adds little: random-like feeds `pre_fe` vs `pre_fe_noid` 0.4393/0.4389 (W1), 0.4375/0.4357 (W2), 0.4317/0.4307 (W3), vs 0.3578/0.3537 on val. The leaky full-corpus models gain +0.015-0.018 from their ID part on W1 train rows (that was the leak). The NN may therefore learn little weight for `pre_id_contrib` / `pre_idmax32` in the train stage; the val fine-tune can re-weight them.
* The in-window snapshots themselves are worth +0.005 on W1 random-like feeds (windowed 0.4401 vs background-only control 0.4350, single config).
* Window models scored on val (future for them, 8-13 days stale): fe 0.351-0.356, i.e. the window models are of the same quality as the full ones (0.357-0.3585).

**Remaining mismatches between the val/test and the train construction** (the coordinator asked for this check):
| aspect | val | test | train windows |
|---|---|---|---|
| background staleness | 2.5-4.5 days (train ends Sep 19; val Sep 22/24) | 2-5 days (val snapshots end Sep 24; test Sep 26-29) | about 0 (latest snapshot just before T_k; W3: <= Sep 14 because Sep 15-16 have no data) |
| in-window corpus | earliest val snapshot per user, 14.2K seqs, oversampled x4 | earliest test snapshot per user, 28.2K seqs, x4 (users with a val feed contribute only their val snapshot) | earliest in-window snapshot per user, 660-750K seqs, x1 |
| in-window density | ~3% of traffic (exploration sample) | ~3% | 100% of production traffic |
| candidate ID coverage (cnt >= 3) | 46% | 41% | 62% (W1) |
| window length | 3 days (data on 2) | 4 days | 3 days |
| models | v3A + v4A (no test snapshots), moment-matched to v3/v4 on val | v3 + v4 | w_k_v3 + w_k_v4, moment-matched to v3/v4 on a 60K-feed window sample |
| training steps | 7,544 (2 ep) | 7,866 (2 ep) | 4,598 / 6,278 / 7,944 (2 ep) |

Consequences: train features come from fresher and denser corpora (more in-window CF), but the ID part is weaker on train rows, not stronger. Overall, the single-feature strength relative to the references is similar (ratio 1.03-1.05 vs 1.03 on val). Raw-score distributions (after alignment): `pre_fe` train mean 15.1 / sd 5.9 vs val 14.4 / 4.5 and test 14.5 / 4.4. Train candidates spread more, and the full-corpus models show the same on train rows. `pre_id_contrib` train mean 0.95 vs val 0.34 (higher ID coverage). Within-feed z features have identical distributions. An NN that standardizes features with train statistics will see val/test raw `pre_fe` slightly compressed; the z features are unaffected.

**Files.**
* Features: `data/feats/train/pre_{fe, fe_z, new, new_z, hmax32, hmax8, fe_noid, id_contrib, idmax32}.npy` (float32, aligned with `data/flat/train/cand_*.npy`, NaN outside the windows). Main score: `preds/pre_final_train.npy`. The val/test files are unchanged.
* Code: `pretrain/build_corpus_window.py`, `run_windows.sh`, `score_window.py`, `score_windows.sh`, `score_refs.sh`, `assemble_train.py`, `later_control.py`, `later_control_models.py`; `train.py` now takes `--corpus_dir` and `--no_test_score`.
* Per-window raw scores: `data/pretrain/winscores/` (window models, reference samples, controls); window feed lists and candidate rows: `data/pretrain/win{1,2,3}/{window_feeds,cand_rows,cand_rows_s60000}.npy`, `info.json`.
* Logs: `logs/pretrain_w{1,2,3}_{v3,v4}.{log,json}`, `logs/pretrain_w1bg_v3.log`, `logs/pretrain_w3bg_v3e1.log`, `logs/pretrain_train_windows_checks.json`, `logs/pretrain_train_windows_later_control_{1,2,3}.json`.

## 8b. Train-stage failure diagnosis (CPU only)
Symptom (NN agent): train-row pre_* as train-stage inputs gave holdout +0.006..+0.0085 but train-only val -0.005..-0.009.

**Setup** (`pretrain/diag_trainstage.py`, `diag_bgcontrol.py`, `diag_patch.py`):
* Train-fit LightGBM lambdarank (eval_family parameters, 350 rounds) on 25K random-like (p_random > 0.9) feeds per window, 3.9M rows.
* Features: frozen base (`eval_cache/base_features.json` with `it_hist_*` swapped to `_luo`), +/- pre_* variants.
* Evaluated on val (paired per-feed delta) and on 24K held-out random-like window feeds.

**Reproduction and ablations** (delta vs the base trained on the same rows; SE in brackets):
| exp | features added / change | VAL delta | HOLDOUT delta |
|---|---|---|---|
| E1 | pre9 (all 9 pre_*) | **-0.0090 (0.0013)** | +0.0106 (0.0011) |
| E1 | content-only: pre_fe_noid + its z | -0.0149 (0.0014) | +0.0093 |
| E1 | ID-only: pre_id_contrib + pre_idmax32 | +0.0001 (0.0011) | +0.0083 |
| E1 | z-only: pre_fe_z + pre_new_z | -0.0115 (0.0013) | +0.0100 |
| E2 | pre9, user-0 rows removed from training (H2) | -0.0113 (0.0014) | +0.0109 |
| E3 | pre9, trained on W1 only | -0.0192 (0.0015) | +0.0077 |
| E4 | pre9, trained on W3 only | -0.0011 (0.0013) | +0.0070 |
| E5 | W1, z(fe), z(new) of the windowed model w1_v3 | -0.0180 (0.0014) | +0.0105 |
| E5 | W1, same from the background-only control w1bg_v3 (no in-window snapshot) | -0.0187 (0.0014) | +0.0095 |
| E5 | W3, windowed w3_v3 / background-only w3bg_v3e1 | -0.0053 / -0.0036 (0.0011) | +0.0063 / +0.0056 |
| E6 | pre9, with all `it_hist_*_luo` removed from the base | -0.0009 (0.0011) | +0.0063 |
| E7 | pre9, with all item-popularity features removed (`it_hist_*`, `it_cand_cnt_*`, `tte_it_*`) | +0.0008 (0.0013) | +0.0148 |
| E8 | pre9 set to NaN where the candidate had no content (train only), z recomputed | -0.0110 (0.0012) | +0.0086 |
| **E9** | **pre9 set to NaN where it_hist_all_luo == 0, on train AND val, z recomputed** | **+0.0037 (0.0009)** | +0.0038 (0.0008) |

**Verdicts.**
* **H1 (density of transductive future cross-user popularity): rejected.** A model that never saw any in-window snapshot causes the same val drop (E5). The ID part is harmless (E1 ID-only); the content path is the most harmful (E1).
* **H2 (user-0 feeds): rejected** (E2).
* **H3, the cause: a label leak through the availability of candidate content embeddings, which build_corpus.py inherited.**
  * The global int8 content table was built from all history events (all splits, including the same user's later snapshots) plus the val/test candidate files, but NOT from the train candidate files.
  * A train candidate therefore had a content vector only if it appears in some history. Otherwise it was scored with the learned "missing" vector, which yields extreme low scores.
  * That covers 10.7% of W1 train candidate rows (20% on random-like feeds), 96% of the candidates with `it_hist_all_luo == 0`.
  * Among LUO-cold train candidates, those *with* content are often items that appear only in the same user's later snapshot (that is how LUO excludes them), i.e. positives. Positive rate 10.1% (W1) / 6.4% (W2) / 3.2% (W3) vs 1.0-1.7% for those without content (all train candidates: 2.3%).
  * On val/test every candidate has content (from the candidate files), so the bit does not exist there.
  * A ranker that has the LUO counts can read the bit from pre_*: "LUO-cold but normal pre score" means positive. It learns that on train and misfires on val, where 38% of candidates are LUO-cold with normal scores.
  * This explains every row above:
    * content-only features hurt most, ID-only not at all;
    * the drop disappears without the `it_hist_*_luo` features (E6, E7);
    * it is largest for W1, which has the most later snapshots after it (E3 vs E4);
    * it is the same for the background-only control (E5);
    * NaN-ing only the no-content rows keeps the bit (E8), while masking all LUO-cold rows on both splits removes it and turns the effect positive (E9).
* Supporting diagnostics:
  * Mean within-feed z of pre_fe on LUO-cold candidates is -1.46 (W1) / -1.20 (W3) on train vs -0.15 on val; the background-only control gives -1.44.
  * Within-feed correlation of z(pre_fe) with log it_hist_all_luo is 0.54-0.59 on train vs 0.15 on val, and with maxsim32 0.36-0.39 vs 0.70.
* Why the section 8 leak checks missed it: the bit touches ~2% of positives, so single-feature NDCG barely moves. The has-later split only looked at later feeds in the *same window*, while the leaking later snapshots are mostly in later windows and in val/test.
* **Val/test pre_* are unaffected** (all candidates have content, and IDs come from corpora without same-user later snapshots); the val leak audit (section 1) stands.

**Immediate CPU fix (written, usable now):**
* `data/feats/{train,val,test}/prew_{fe,fe_z,new,new_z,hmax32,hmax8,fe_noid,id_contrib,idmax32}.npy`, built by `pretrain/make_prew.py`.
* Rule: pre_* restricted to history-warm candidates, NaN where it_hist_all_luo == 0, z recomputed over the warm candidates; the same rule on every split.
* NaN shares: train 38% (Sep 6-8 plus cold rows), val 38%, test 45%.
* E9 shows train-fit val **+0.0037 +- 0.0009** with it. Cost: no pre_* signal on cold candidates (38-45% of val/test rows).
* Val-only harness (`eval_family --family prew_`): **+0.0031 +- 0.0008**, vs +0.0037 for the full pre_ family. Masking the cold candidates costs about 0.0006 on val, which the rebuild below would recover.

**Proper rebuild plan (not run). GPU cost about 2 h on a free GPU, ~3 h if GPU N stays shared.**
1. Content table for scoring: add the train candidate embeddings (`cand_emb_idx` -> train `item_embeddings`) for every train candidate. Then every scored candidate on every split has its own content and "has content" carries no information. CPU, 2 min. Training is unaffected (candidates never enter the corpora), so the corpora and configs stay identical.
2. Rebuild the window corpora (`build_corpus_window.py`, 1 min). The window checkpoints were deleted after scoring, so retrain w{1,2,3}_{v3,v4}: about 26 + 35 + 44 = 105 min on a free GPU.
3. Re-score: the window feeds (6 x 1.5 min) and the full-corpus reference samples used for moment matching (6 x 0.7 min), with the fixed table. Then `assemble_train.py` (CPU 10 min).
4. Gate before handing to the NN: rerun E1 and E9 with the rebuilt features (CPU 10 min). Require: VAL delta >= 0 for pre9 with the `it_hist_*_luo` base, and mean z of LUO-cold candidates on train close to val (-0.15). Keep the checkpoints until the NN test passes.
* A density-matched in-window corpus (exploration users only, x4) is **not needed**: the background-only control shows that density was not the problem. It would also leave features on only ~3% of window feeds.

## 8c. Rebuild of the train-stage features (pre2_*)
**Fix.** `pretrain/build_item_tab2.py` writes `item_tab2.npy` / `item_has2.npy` = the section-2 content table plus the train candidate embeddings (14.2M items added; items covered 69.0M -> 83.3M). Every candidate of every split now has its own content vector, so "has content" carries no information. The new table changes no pretraining-corpus token (the added items never occur in any history), so training is unchanged and only the scoring of train candidates differs.

**Val/test.** With item_tab2, 5 val and 11 test candidate items, plus 6 val and 18 test history events, gain content (items whose val/test embedding was missing but which have a train candidate embedding). Re-scoring with the same models (v3A/v4A for val, v3/v4 for test and for the val alignment targets) changes 228-306 val rows and ~1,150 test rows (correlation 0.99999; val NDCG identical to 4 decimals). Per the instruction, `pre2_*` for val/test are written from this same pipeline (`assemble.py --prefix pre2_`). pre2_final val NDCG 0.3590 = pre_final.

**Rebuild** (GPU N only, shared with up to 5 other jobs, 0.14-0.85M tok/s):
* Window corpora rebuilt with `build_corpus_window.py`; identical to section 8.
* The 6 window models were retrained with the same configs and seeds plus `--item_tab item_tab2` (`run_windows_r.sh`, v3 config first):

| model | train time | val fe (future for the model) |
|---|---|---|
| w1r_v3 | 58.5 min (GPU shared 5 ways) | 0.3524 |
| w2r_v3 | 36.7 min | 0.3538 |
| w3r_v3 | 34.6 min | 0.3566 |
| w1r_v4 | 13.4 min | 0.3512 |
| w2r_v4 | 18.0 min | 0.3544 |
| w3r_v4 | 22.6 min | 0.3559 |

* The section-8 models scored 0.351-0.356 on val, so the retrained models are equivalent.
* Re-scored with item_tab2: the window feeds (`score_windows_r.sh`), and the 60K-feed alignment samples of the full-corpus v3/v4 (`score_refs_r.sh`, tags v3t2/v4t2).
* Re-assembled with `assemble_train.py --wtag "w{k}r_{c}" --rtag "{c}t2" --prefix pre2_` (2 configs, the same rule as val/test).
* **Checkpoints `data/pretrain/ckpt_w{1,2,3}r_{v3,v4}.pt` are kept** until the NN test passes.

**Outputs:**
* `data/feats/{train,val,test}/pre2_{fe,fe_z,new,new_z,hmax32,hmax8,fe_noid,id_contrib,idmax32}.npy`. Train is NaN outside the windows (72.1% of candidate rows covered); within the windows only `pre2_idmax32` has NaNs (35.6-38.5%).
* `preds/pre2_final_{train,val,test}.npy` (val 0.3590).
* All three splits share one construction: content table item_tab2, 2 configs (v3, v4), the val side without test snapshots, and moment alignment to the full-corpus twins.

**Gates** (`pretrain/gate_pre2.py`, `logs/pre2_gate.json`). Train-fit LightGBM is the 8b setup (25K random-like feeds per window, frozen LUO base, 2 seeds):

| arm | VAL delta vs base | HOLDOUT delta |
|---|---|---|
| **pre2 (9 features)** | **+0.00535 (0.00086)** | +0.00522 (0.00082) |
| prew (8b interim fix) | +0.00315 (0.00085) | +0.00410 (0.00080) |
| pre2 - prew | +0.00221 (0.00076) | |
| pre2 from v3 config only (intermediate) | +0.00465 (0.00089) | +0.00433 |
| as built, section 8 (8b E1, for contrast) | -0.0090 (0.0013) | +0.0106 |

Train-only W1 (the worst window before, -0.0192) now gives +0.0049 (0.0009).

* **Cold-candidate within-feed z** of pre2_fe (LUO-cold = it_hist_all_luo == 0):
  * val: -0.145;
  * train: W1 -0.272, W2 -0.321, W3 -0.292 (as built: -1.46 / -1.20).
  * The artifact is gone. The remaining ~0.15 z is consistent with the label mix: train LUO-cold items are never-interacted items rather than new ones. Cold/warm positive-rate ratio is 0.58 / 0.61 / 0.76 on train vs 0.71 on val.
  * Inside the cold bin the score discriminates as on val: positive rate top-quintile / bottom-quintile 8.7 / 9.5 / 7.2 on train vs 7.0 on val.
* **Has-later split extended to ANY later feed of the same user** (any train window, val or test). NDCG gap (has-later minus no-later) of pre2_final vs maxsim32:

| window | pre2_final (all / random-like) | maxsim32 (all / random-like) |
|---|---|---|
| W1 | -0.0004 / -0.0075 | -0.0007 / -0.0026 |
| W2 | +0.0005 / +0.0008 | +0.0016 / +0.0038 |
| W3 | +0.0053 / +0.0081 | +0.0064 / +0.0084 |

  * No jump relative to the leak-free references (cos_u and f1 in the json).
  * The within-window split also shrank (W1 all feeds: +0.0132 as built -> +0.0045).
* Leaky-model contrast (random-like feeds of the 60K samples): windowed pre2 0.437 / 0.436 / 0.434 vs full-corpus models 0.456 / 0.445 / 0.438.
* Single-feature NDCG on random-like window feeds: 0.4386 / 0.4380 / 0.4375 (pre/maxsim32 = 1.04 vs 1.03 on val).

**Remaining mismatches** (unchanged from section 8): background staleness (about 0 days on train vs 2-5 days on val/test), and in-window density (all window users vs a ~3% exploration sample, with no x4 oversampling on train). Neither shows up in the gates.


**8c addendum: id_contrib parity fix (review P5).**
* **Problem.** `id_contrib` was computed as `fe - fe_noid` after moment-matching fe and fe_noid separately. On val (and train) the separate slopes left a small linear function of fe on rows without an item ID; on test (no alignment) it was exactly 0.
* **Fix** (`assemble.py`, `assemble_train.py`). `id_contrib = a_fe * (fe_raw - fe_noid_raw)`, with the single fe alignment slope (1 on test). Rows without an item ID are now exactly 0 on every split.
* **Rewritten:** `pre2_id_contrib` for train and val. The test file and all 8 other pre2 features plus `pre2_final_{train,val,test}` are byte-identical (md5 checked); nothing else derives from id_contrib.
* **Parity check:**
  * Rows with `pre2_idmax32` NaN are exactly 0 for 98.1% (train) / 99.1% (val) / 99.5% (test). The rest are candidates with an ID whose recent history has none.
  * Exact-zero shares: 0.36 (train, all window feeds), 0.45 (train, random-like feeds), 0.55 (val), 0.59 (test). These follow ID coverage, not the construction.
  * PSI of the non-zero values (val deciles): val vs test 0.003; train random-like vs val 0.041 and vs test 0.065. Including production feeds, train vs val is 0.096 and vs test 0.130 (production candidates have more and stronger IDs).
* **Not fixed here:** the val/test `pre_id_contrib` and `prew_id_contrib` files still use the old construction. `pre_id_contrib` is a live fine-tune extra, so I left it untouched. The same one-line fix applies by re-running `assemble.py` with `--prefix pre_` when no job reads it.
* **Old families, new names** (`pretrain/make_idcontrib_v2.py`; the live `pre_id_contrib.npy` / `prew_id_contrib.npy` are untouched, md5 checked):
  * Files: `data/feats/{val,test}/pre_id_contrib_v2.npy` and `data/feats/{train,val,test}/prew_id_contrib_v2.npy`.
  * Construction: the same fixed formula with the original section-1/8 models and alignment slopes (v3A/v4A -> v3/v4 on val; w<k>_<c> -> <c> on the window samples; test slope 1).
  * Rows without an item ID are exactly 0 on every split, and NaN patterns are identical to the old files.
  * PSI of non-zero values val vs test is 0.003 for both families. For prew, train vs val is 0.098 and train vs test 0.133 (all production feeds). Zero shares for prew: train 0.255, val 0.272, test 0.262.
