# Feature review: inventory, usage map, strength and coverage gaps

Author: featrev inventory agent. Date: 2026-11-11, 15:05-16:10 UTC. CPU only: `CUDA_VISIBLE_DEVICES=""`, at most 16 threads in total. I read project files but did not modify them.
- Scripts, logs and raw outputs are in `/tmp/scratchpad/featrev/inventory/`:
  - `c01_cache.py`: feature matrices;
  - `c02_stats.py` → `stats.csv`;
  - `c03b_resid.py` / `c03c_resid.py` / `c07_resid_new.py` / `c08_groups.py`: residual value;
  - `c04_redund.py` → `redund.json`;
  - `c05_table.py` → `inventory.csv`, `inventory_table.md`;
  - `c06_gate.py` → `gate_check.json`;
  - `*.log`.
- New code is in `<repo>/feat_new/inventory_*.py`. New arrays are in `<repo>/data/feats_new/{train,val,test}/`.
- Related reviews: `reports/review_leakage.md` (cited for the leak status of every live input), `review_shift.md`, `review_integrity.md`, `featrev_prep.md`, `featrev_research.md`.

## TL;DR

1. **Inventory and usage.** There are 256 feature files on val/test and 238 on train. Missing on train: `als_*` (9) and `pre_*` (9, quarantined).
   - The NN uses 66 of them:
     - 28 as train-stage inputs (`cvnd`, nn_final4);
     - +5 in `cvnd_prew` (round-4 allprew members);
     - 51 as fine-tune extras (`n2c_pre`).
   - The GBDT harness base uses 122. The train-fit LightGBM gate uses `swap_luo(base)` + pre2/prew.
   - **49 files are used by no model.** They are trd_* (20), seq_* (19), clu_* (6) and ctx_hs_{d7,e1k64,k128t10,k256}.
2. **The feature space on disk is saturated relative to the NN (INVENTORY-3).**
   - Test: incremental value over the best val mix inside the exact nn/finetune.py folds. No single feature and no feature family adds more than 2 SE once multiplicity is accounted for:
     - 254 features × 2 residual models: 3 / 2 nominal t > 2 hits against ~6 expected by chance;
     - 11 family groups: best +0.0007 ± 0.0004.
   - Against the train-only NN, the only group at 2 SE is `pre_*`: +0.0010 ± 0.0005. That supports the pre2 train-stage work and nothing else on disk.
   - **Further gains have to come from new information or better use of the existing inputs, not from re-encoding what is already in data/feats.**
3. **INVENTORY-1 (medium, verified, found independently by three agents): the cvnd_prew raw columns `prew_fe` / `prew_new` are clipped at +10 by `attach_xt`.**
   - 86% / 79% of present rows saturate, so their only route into the model (in-model within-feed rank/z) runs on ties. Val AUC as the NN sees it: 0.562 vs 0.608.
   - The same will happen to the `pre2_fe` / `pre2_new` raw columns (p50 14.7, p99 22.4) if they are handed over the same way.
   - Clip-safe copies are built: `data/feats_new/*/prew_{fe,new}_s.npy`.
4. **INVENTORY-2 (medium, verified, measured): the train-fit GBDT gate base contains parity-broken columns.**
   - The columns are `it_cand_cnt_split{,_rk,_z,_gap}` (train PSI 16.6; their within-feed AUC flips sign between train and val) and `it_cand_cnt_train` (train rows count themselves).
   - Removing them gives **val +0.0025 ± 0.0010 / holdout −0.0056 ± 0.0006** (2 seeds, same params). So they carry train-only signal, which the holdout rewards and val penalises.
   - The pre2 / prew gate deltas were measured on this base.
5. **INVENTORY-4 (low): the n2c_pre fine-tune lost 14 of its 65 listed inputs to upstream pruning.**
   - I rebuilt the 7 deleted `ctx_nd_*` features bit-exactly (`feat_new/inventory_ctx_nd_restore.py`).
   - Their residual over the mix is +0.0001 / +0.0005 ± 0.0004, and over the train-only model ~0. The "+0.001" attributed to them in nn_report is not reproduced, so restoring them is optional.
6. **Parity (task 3d).**
   - Apart from the known cases, no feature shows a train-random vs val discrimination gap. The known cases are: non-LUO it_hist (banned); it_hist_*_luo (+0.03 skew, LEAKAGE-7 / SHIFT-1); in-sample it_tr_* and u_tr_* (val/test only); it_cand_cnt_* (INVENTORY-2); prew (LEAKAGE-6); and the in-progress pre2 train files.
   - Smaller parity notes: tte_it_ctr / coec (+0.018, gap mismatch, INVENTORY-7) and trd_skde (+0.024, unused).
7. **Redundancy.** At |Spearman| > 0.95 there are 48 clusters (107 features). Inside the live NN inputs:
   - `seq_cv_norm` ≡ `seq_cv_invd20` (ρ 0.9998; both in XT and FT);
   - `seq_cv_pop` ≈ `it_hist_all_luo` (0.967);
   - `ctx_hs_t10` ≈ `ctx_hn_soft05` (0.966);
   - `pre_fe` ≈ `pre_new` (0.978);
   - on val, `it_tr_ctr` ≡ `tte_it_ctr` and `it_cand_cnt_train` ≡ `tte_it_imp` (ρ 1.000).

---------------------------------------------------------------------------------------------------------------------

## 1. Method (what was computed, exactly)

**Row sets** (`c01_cache.py`):
- val: all 14,423 feeds / 1,389,971 rows;
- test: all rows (no labels; used for PSI and NaN rates);
- **trH**: the exact NN random-like holdout, 30,000 feeds / 4,630,966 rows. It is defined by `nn/train.py:153-162` (p_random > 0.9, M ≥ 50, rng 12345). Its labels and the four nn_final4 member predictions in `preds/holdout/` match my reconstruction exactly (M, offsets and y identical).
- **trU**: 20,000 uniformly drawn train feeds outside trH, 1,563,030 production rows. Used for NaN rates.

**Strength** (`c02_stats.py`):
- Within-feed AUC per feed, averaged over feeds with both classes. NaN is imputed below the feed minimum.
- The sign is set by val (`sign` column). "sAUC" is the sign-aligned AUC on both splits.
- Single-feature NDCG uses `src/metrics.py` (pessimistic ties, NaN ranked last). A per-feed constant scores 0.1986 on val and 0.1631 on trH under pessimistic ties; random order is ~0.29.
- skew(M50) = [sAUC_trH − sAUC_maxsim32,trH] − [sAUC_valM≥50 − sAUC_maxsim32,valM≥50]. This is the review_leakage detector, with val restricted to M ≥ 50 to match trH's M. A feature is flagged only when |skew| > 0.015 **and** the raw sAUC gap exceeds 0.01. The second condition is needed because maxsim32 itself is 0.008 weaker on trH, which inflates skew for weak features such as f2 / f4 / f5.
- PSI uses 10 val-quantile bins plus a NaN bin.

**Residual value** (`c03b_resid.py`, `c03c_resid.py`, `c07_resid_new.py`, `c08_groups.py`):
- **Folds are exactly those of nn/finetune.py**: `GroupKFold(n_splits=5)` without shuffle (`nn/finetune.py:180`), `.split(np.arange(va.n), groups=va.user_id)` (`:183`). `va.user_id` = feeds.parquet `user_id` in feed order (`nn/data.py:73`). Same venv, sklearn 1.9.1.
- Base b = within-feed z of 0.5 z(`nn_final4_val_oof`) + 0.5 z(`nn_final4_trainonly_val`) + 0.1 z(`pre_final_val`). Its NDCG is 0.39399, reproducing review_shift's 0.39399.
- Two residual models per feature, fitted on 4 folds and scored on the 5th:
  - **LIN**: a conditional logit (listwise softmax) on [β·b, z(f), pct-rank(f), isnan(f)], with z and rank over present rows.
  - **GBM**: LightGBM lambdarank (full-list truncation 1000, 60 rounds, 7 leaves, min_data_in_leaf 2000, l2 50) with init_score β·b on [b, f, z(f), isnan(f)]. Its control is the same booster on [b] alone; the control scores 0.39418 (+0.0002 from recalibration).
- Deltas are paired per feed against the control, over all 14,423 feeds.
- 4 pure-noise features calibrate the null: LIN −0.00019..−0.00038 (mean −0.00028), GBM −0.00054..+0.00006 (mean −0.00029). Fitting a useless column costs ~0.0003.
- The same design was run with the **train-only** half as base (`nn_final4_trainonly_val`, 0.39281; 86 relevant features + groups), and on **trH** with the four nn_final4 train-only member predictions as base (0.31810; LIN only, 233 train-available features).
- An earlier, less regularised GBM pass (120 rounds, 15 leaves; `c03_resid.py`, `resid/`) was stopped after 5 features because it overfit (−0.0008..−0.0018 for everything). It is not used.

---------------------------------------------------------------------------------------------------------------------

## 2. Findings (most severe first)

### INVENTORY-1 (medium, verified): `prew_fe` / `prew_new` saturate at the ±10 clip of the NN train-stage transform, so their in-model rank/z channels (their only route into cvnd_prew models) run on ties

This was found independently by featrev_prep (PREP-1) and featrev_research (RESEARCH-1). This section adds the numbers from my side and the pre2 implication.

**How the inputs are built:**
- `nn/data.py:182-200` (`attach_xt`) passes `*_rk` through unchanged, clips `*_z` to ±5, applies log1p to `seq_cv*` / `it_hist*`, and **clips every other column to ±10** (`:199`). NaN then becomes 0 (`:200`).
- In `cvnd_prew`, `prew_fe`, `prew_new` and `prew_hmax32` are `XT_RELONLY` (`nn/train.py:51`). They enter only through the in-model within-feed pct-rank and z (`nn/model.py:123-130`), which are computed on the already clipped column.

**Measured** (val / trH; `c02`, inline check in this report's log):

| column | p50 / p99 raw | present rows at the clip | AUC of the NN-seen column (NaN → 0) | AUC unclipped | rows tied at 10 (all rows) |
|---|---|---|---|---|---|
| prew_fe | 15.3 / 22.9 | 86.1% / 86.1% | **0.5623** / 0.5493 | 0.6080 / 0.6033 | 53.4% |
| prew_new | 13.8 / 21.7 | 78.7% / 78.9% | **0.5697** / 0.5577 | 0.6077 / 0.6031 | 48.8% |
| prew_hmax32 | 0.79 / 0.97 | 0 | 0.6006 | 0.6006 | 0 |
| all 28 cvnd columns | - | ≤ 1.2% (ctx_hn_soft05 1.2%, ctx_hs_e1 1.0%, ctx_hs_k128 0.6%, ctx_hs_t10 0.2%) | unchanged | - | - |

**Impact:**
- This affects every `--xt cvnd_prew` run: `v12_prew3` and `v12_allprew_s83/84/85` (`'xt': 'cvnd_prew'` in `nn/logs/v12_allprew_s83.log` and `v12_prew3.log`).
- The prew acceptance (val +0.0011 ± 0.0007) was measured with the degraded channel.
- The information is partly kept through `prew_fe_z` / `prew_new_z`, which are not clipped (within-feed ρ to the raw column 0.863). So the loss is bounded, but it is not measured.
- **The same trap applies to pre2:** `pre2_fe` / `pre2_new` have the same raw scale (val p50 14.7 / 13.2, p99 22.4 / 21.3). The old `cvnd_pre` / `cvnd_prez` presets (`nn/train.py:38-46`) passed raw `pre_fe` through this clip too.

**Fix (no change to nn/ needed for new runs):**
- Use `data/feats_new/{split}/prew_fe_s.npy` / `prew_new_s.npy` in place of `prew_fe` / `prew_new` in the cvnd_prew preset (`feat_new/inventory_prew_rescale.py`).
- The construction is a fixed affine map (x + 20)/6:
  - raw ranges: train [−15.35, 30.72], val [−9.60, 27.56], test [−10.82, 28.06];
  - all map into [0.78, 8.45], so the clip is never hit;
  - NaN stays NaN, so after NaN → 0 it still sorts below every present row.
- Verified on val: the NN-seen AUC goes 0.5623 → 0.6080 for prew_fe and 0.5697 → 0.6077 for prew_new, with 0% ties at 10.
- Do the same, or exempt XT_RELONLY columns from the clip, before any pre2 train-stage preset.
- Do not change `attach_xt` in place: existing checkpoints were trained on the clipped column.
- Acceptance needs a paired GPU run (≥ 2 seeds, train-only val > 2 SE).

### INVENTORY-2 (medium, verified, measured): the train-fit LightGBM gate base mixes in columns whose train construction differs from val/test

**Where they come from:**
- `pretrain/gate_pre2.py:21` and `pretrain/diag_trainstage.py:54` use `swap_luo(base_features.json)`. `swap_luo` (`src/common.py:28-38`) only swaps it_hist_*. So the gate trains on:
  - `it_cand_cnt_split{,_rk,_z,_gap}` (`src/features.py:337`): the in-split candidate count, which on train includes the row itself and every later production exposure;
  - `it_cand_cnt_train` (`:338`): on train rows it counts the row itself, so it is ≥ 1 for every train row, while 34.6% of val rows have 0;
  - `hour`, which review_shift / feat_cal_ recommend dropping.
- review_leakage LEAKAGE-5 flagged it_cand_cnt_split for the val harness (`eval_family.py`), but not for the train-fit gate.

**Evidence:**

| | val | trH | gap |
|---|---|---|---|
| it_cand_cnt_split sAUC (val sign) | 0.5035 | **0.4584** | sign flip; PSI val→trH 16.6, val→test 10.6 |
| it_cand_cnt_train PSI | - | 5.09 | min value 1 on trH vs 0 on 34.6% of val rows |

- Direct test (`c06_gate.py`): the same LightGBM params and 350 rounds as `diag_trainstage.py`, 25K random-like train feeds outside trH, 2 seeds.
  - Gain share of it_cand_cnt_* in the gate model: **5.85%**.
  - B (minus it_cand_cnt_*) − A (gate base): **val +0.00252 ± 0.00099, trH −0.00565 ± 0.00055**.
  - C (also minus hour) − A: **val +0.00289 ± 0.00100, trH −0.00529 ± 0.00055**.
- This is the signature of a train-only signal: the holdout rewards it, val penalises it.

**Impact and fix:**
- The gate base is 0.0025-0.0029 weaker on val than it should be. pre2 (+0.0034 ± 0.0009) and prew (+0.0015 ± 0.0009) were measured as deltas on top of it (`logs/pre2_gate_w12.json`).
- The direction of the bias on those deltas is unknown. A base that leans on a broken column can make a new signal look either more or less useful.
- Fix: drop `it_cand_cnt_*` and `hour` from the gate base. `tte_it_imp`, already in the base, is the time-aware twin of `it_cand_cnt_train` (ρ 1.000 on val). Re-run the pre2 gate before the NN hand-off.

### INVENTORY-3 (info, decision-relevant, verified): no feature on disk adds measurable value over the current NN mix

**Per feature, over the mix (254 features; full numbers in Appendix A):**
- LIN: the only nominal t > 2 is the f1 family: f1 +0.00056 ± 0.00021 (t 2.66; f1_z / f1_gap identical). GBM for f1 gives −0.00026 ± 0.00034.
- GBM: the only nominal t > 2 are `it_tr_coec` +0.00086 ± 0.00038 and `it_tr_coec_z` +0.00089 ± 0.00035 (t 2.24 / 2.57). LIN gives +0.00030 ± 0.00023.
- Under the null about 5.8 features per model would pass t > 2 one-sided. Bonferroni for 254 tests needs t > 3.5. Median deltas are LIN −0.00002 and GBM −0.00025; t < −2 occurs for 12 (LIN) / 22 (GBM) features.

**Family groups over the mix** (`c08_groups.py`; LIN / GBM, SE ~0.0003-0.0004):

| group | members | LIN | GBM |
|---|---|---|---|
| train-label stats (val/test only) | it_tr_ctr/coec/pos, u_tr_*, it_cand_cnt_train | +0.00044 (0.00029) | +0.00069 (0.00038) |
| tte_* | 12 | +0.00026 (0.00032) | −0.00015 (0.00040) |
| als_* (raw) | 7 | −0.00013 (0.00036) | −0.00085 (0.00042) |
| pre_* (raw) | 6 | −0.00030 (0.00034) | −0.00014 (0.00039) |
| seq_* not used by the NN | 15 | −0.00055 (0.00036) | −0.00068 (0.00039) |
| clu_* KEEP | 6 | −0.00033 (0.00027) | −0.00058 (0.00038) |
| trd_* | 20 | −0.00050 (0.00041) | −0.00011 (0.00041) |
| ctx_hs extras | d7, e1k64, k128t10, k256 | −0.00047 (0.00028) | −0.00053 (0.00035) |
| GBDT-only embedding features | cos_h*, cos_het*, attn32, meansim32, dot_hall, centrality | −0.00004 (0.00041) | −0.00047 (0.00038) |
| GBDT-only category / user features | cat_cnt_20/5, hist_n_uniq_cat, u_het*_share | −0.00017 (0.00017) | −0.00017 (0.00028) |
| restored ctx_nd (7, INVENTORY-4) | | +0.00013 (0.00033) | +0.00047 (0.00036) |

**Over the train-only NN on val** (what the train stage misses; 0.39281):
- Per feature, the best are `als_exp5` LIN +0.00088 ± 0.00040 (2.2; GBM −0.00026), `als_exp5_z` (2.3), `f4` LIN +0.00057 / GBM +0.00092 (1.6 / 1.8), and `centrality` +0.00044 (1.6).
- 86 tests, so 2 hits at 2.2-2.3 SE is chance level.
- Groups: **G_pre LIN +0.00100 ± 0.00050 (2.0 SE)**, GBM +0.00025. Train-label +0.00023 / +0.00036, tte +0.00010 / −0.00024, als +0.00022 / −0.00026, restored ctx_nd −0.00001 / −0.00040.

**On trH over the nn_final4 train-only members** (LIN):
- max +0.00047 ± 0.00026 (`it_tr_coec`, which is in-sample on train);
- non-LUO `it_hist_all_rk` +0.00037 ± 0.00016 (banned);
- nothing else above 2 nominal SE. The trH per-feed SE is ~2.3x too small (nnx_VERDICT 2.2).

**Reading:**
- The GBDT-validated KEEP families (seq_ +0.008, ctx_ +0.003, clu_ +0.001 on the val harness) are fully absorbed by the NN's history encoder and in-model similarity features.
- The val/test-only stats (it_tr_*, als_*, pre_*) are absorbed by the fine-tuned half: pre_* group over the train-only model +0.0010, over the mix −0.0003.
- Moving a val/test-only signal into the train stage can therefore buy at most what it adds to the train-only model, ≤ ~+0.001 (pre_*). The OOF half already gets it through the fine-tune.

### INVENTORY-4 (low, verified): the n2c_pre fine-tune silently lost 14 of its 65 listed inputs; the 7 deleted ctx_nd features are rebuilt but show no residual value

**What was lost:**
- `nn/finetune.py:151` builds `n2c_pre` = n2clean + PRE = 65 names.
- The loader drops what is missing on disk (`:161-164`). `nn/logs/v9_e10_n2cp_ft.log` lists 14 dropped names:
  - `seq_cv_invd20_z, seq_cv_w5, seq_cv_fwd5, seq_cv_t1, seq_cv_seqs, seq_dcos_hl5, seq_nd90_cnt`;
  - `ctx_nd_c80, ctx_nd_c90, ctx_nd_c95, ctx_nd_dupb_f1, ctx_nd_dupb_h, ctx_nd_dupb_u, ctx_nd_mean`.
- So 51 were loaded.
- nn_report (R2 step 1, l. 233/247) attributed "+0.001" to the 7 deleted ctx_nd features (s2_v4_cv fine-tune 0.38636 with them vs 0.38533 without; one seed).

**Rebuild:**
- `feat_new/inventory_ctx_nd_restore.py` copies `_nd_kernel` verbatim from `feats_v2/ctx_/build_ctx.py`, with the numba cache off. It writes `data/feats_new/{val,test}/ctx_nd_{c80,c90,c95,mean,dupb_h,dupb_f1,dupb_u}.npy`.
- Self-check: the 10 ctx_nd columns that still exist are reproduced **bit-exactly** on val and test (max abs diff 0, NaN pattern identical; `c_ctxnd_restore.log`).
- The inputs are feed-internal (own candidates, own history, own user vector, the split's own item table), so a train version is the same call with `train`. I did not run it: ~18 GB RAM, ~30 min at 4 threads.

**Residual of the 7 as a group:**
- over the mix: LIN +0.00013 ± 0.00033, GBM +0.00047 ± 0.00036;
- over train-only: −0.00001 ± 0.00037 / −0.00040 ± 0.00037;
- singles: |Δ| ≤ 0.00044, all < 2 SE.

Recommendation: restoring them is optional (n2full reproducibility). Do not expect +0.001. More generally, the n2c_pre name list should be pruned to what exists, so that logs and reports describe the model that was actually trained.

### INVENTORY-5 (low, verified): fine-tune extra-branch scaling

featrev_prep PREP-3 covers this in more depth; my independent numbers are below. The transform (`nn/data.py:204-219`) is: log1p if min ≥ 0 and max > 20 (decided on val, reused for test), then standardise with val stats, clip ±6, NaN → 0 (the mean), with no missing indicator.

**Measured on val after the transform:**
- `pre_idmax32` is NaN on 55.3% of val rows and 59.5% of test rows, and is imputed to the mean with no flag.
- Nine inputs have IQR = 0 after the transform. They are near-binary with tails to +4..+6 SD: `it_hist_et5..10_luo`, `seq_cv_invd20` (88.7% zeros), `seq_cv_norm`, `it_tr_pos`.
- `hour` gets log1p because its max is 23 > 20. Hour 0 maps to −3.9 SD and 23 sits next to 22, so it is non-cyclic.
- `pre_hmax32` / `pre_hmax8` carry a −2 sentinel (H = 0 feeds, 1.4% of val rows) that lands at the −6 clip.

**Impact:** small. The ablation in nn_report shows that most of the fine-tune's value is recalibration. Fix: see Idea 5.

### INVENTORY-6 (low, verified): duplicated inputs in the live NN presets

**Duplicates:**
- `seq_cv_norm` vs `seq_cv_invd20`: Spearman 0.9998 (within-feed 0.9998). Both are in `cvnd` (`nn/train.py:30-32`) and in n2c_pre.
- Other pairs inside cvnd with |ρ| > 0.9: `seq_cv_pop`–`it_hist_all_luo` 0.967, `it_hist_all_luo`–`it_hist_et1_luo` 0.966, `ctx_hn_soft05`–`ctx_hs_t10` 0.966, `ctx_hn_rrw`–`ctx_hn_e1_rrw` 0.920, `ctx_hn_soft05`–`ctx_hs_k128` 0.911.
- In n2c_pre: `pre_fe`/`pre_new` 0.979, `pre_fe_z`/`pre_new_z` 0.978, `it_tr_coec`/`it_tr_coec_rk` 0.965, and the rank/z/gap triplets.
- On val, `it_tr_{ctr,coec,pos}` ≡ `tte_it_{ctr,coec,pos}` (ρ ≥ 0.999) and `it_cand_cnt_train` ≡ `tte_it_imp` (1.000). They differ only in their train-row construction.

**Impact:** for the NN, harmless but wasteful. For the val-only GBDT harness it matters because duplicates change what `feature_fraction` samples (feat_ctx_ / feat_clu_ report column-order effects). All 48 clusters are listed in Appendix B.

### INVENTORY-7 (low, verified): tte_* is the only leak-free train-label item-quality signal, yet no NN uses it, and its train construction is not gap-matched

**Construction:**
- `src/features_tte.py`: train rows of day d use train days < d (expanding window, gap 0). Val/test use all train days, which is a 3-5 day gap for val and 7-10 days for test.
- Measured parity:
  - `tte_it_ctr` / `tte_it_coec` skew(M50) +0.018 (sAUC trH 0.518 / 0.514 vs val 0.510 / 0.507);
  - `tte_uc_imp` / `tte_u_imp` zero share 78-81% on trH vs 64-69% on val, because early train days have little history;
  - `tte_c_ctr` is exactly the prior on all Sep 6 rows (9% of trH).

**Float detail:**
- The prior for unseen items is 0.01589999907 on train (float32 arithmetic) but 0.01590000093 on val/test (float64 then cast).
- That is harmless for the NN, but an exact-threshold tree can put the two populations on opposite sides of a split. The val→trH PSI of 1.14 for `tte_it_ctr` is this artefact of my quantile bins, not a real shift.

**Value:** the residual over train-only (G_tte +0.0001 / −0.0002; tte_it_ctr +0.00037 / +0.00035) says it would not add much. See Idea 3 for the gap-matched construction if the NN agent wants a train-stage CTR prior anyway.

### INVENTORY-8 (info, verified): 49 feature files are used by no model, and the "KEEP" verdicts of the feature workflow are not used by the NN

- Unused: trd_* (20; family verdict "drop", but the files remain), clu_* (6 KEEP), seq_* (19 of the 26 KEEP: all except seq_cv_*, seq_max_all, seq_top3, seq_argmax_hpos), and ctx_hs_{d7,e1k64,k128t10,k256} (4 of the 24 ctx KEEP).
- They are redundant for the NN (INVENTORY-3 groups ≤ 0), so this is housekeeping: delete or archive them to avoid confusion.
- Note: the val harness base (`data/eval_cache/base_features.json`) is stale. It still contains banned non-LUO it_hist_* and it_cand_cnt_split* (LEAKAGE-5).

### INVENTORY-9 (info, verified): `_rk` features rank NaN rows arbitrarily

- `src/features.py:192-227` (`_rel_feats`) argsorts with NaN last, and NaN never ties. So NaN rows get the top rank (≈ 1.0) in feeds that are only partly NaN, and index order in all-NaN feeds.
- All-NaN feeds are H = 0 feeds: 1.2% of production train rows and 2.6% of trH rows. Index order correlates with production position (corr(index, pos) = 0.41 on trU), but the rank of NaN rows does not (corr −0.008). AUC of `attn32_rk` inside all-NaN production feeds is 0.476 (234 feeds).
- Negligible. The NN computes its own ranks. For GBDTs, prefer NaN for NaN rows.

### INVENTORY-10 (info): pre2 train files were being rewritten during this audit

- `data/feats/train/pre2_fe.npy` mtime is 15:07; my cache read it at ~15:06. In my snapshot, pre2_* on trH is 54% NaN, sAUC is 0.576 against 0.662 on val, and PSI is 7.7.
- These figures describe a partial file and must not be read as a parity verdict. Val/test pre2 (13:56) is complete.
- pre2_id_contrib has the LEAKAGE-2 construction difference: PSI val→test 3.87.

---------------------------------------------------------------------------------------------------------------------

## 3. Coverage gaps (task 3)

**(a) Strong val/test-only features the train stage cannot see:**

| feature | val sNDCG / sAUC | why no train version | leak-free train construction | residual over train-only NN | verdict |
|---|---|---|---|---|---|
| pre_* (9) | 0.359 / 0.663 (best single features on disk) | train pre_* quarantined (8b availability leak) | prew_* (masked; live in cvnd_prew, but see INVENTORY-1); pre2_* (all candidates with content; rebuild running) | group LIN +0.0010 ± 0.0005 | the only gap with ≥ 2 SE. Pursue pre2 with clip-safe scaling and judge on train-only val |
| als_* (9) | 0.321 / 0.579 | ALS factors use all history pairs, so a train row would see same-user later pairs | time-windowed ALS (factors from snapshots before the window), like the pre_* windows | als_exp5 +0.0009 ± 0.0004 LIN, GBM −0.0003; group +0.0002 / −0.0003 | not worth a train build |
| it_tr_* / u_tr_* / it_cand_cnt_train | 0.285 / 0.507 (it_tr_coec) | in-sample on train (leave-own-row-out only); u_tr includes the own label | **already exists: tte_*** (strictly past days), ρ 1.000 to it_tr on val | train-label group +0.0002 / +0.0004 | low value; gap-matched tte (Idea 3) if anything |
| hour | 0.5 (per-feed constant) | none; exists on all splits | identical | GBM −0.0002 | do not add (calendar shift, SHIFT-7 / feat_cal_) |

**(b) Strong features used only by GBDTs:** 119 base columns plus the KEEP families.
- Most have in-model NN analogues: f1-5 raw/z/rank, cos_u, dot_u, smax / smax8 / smax32, smean, lse, top3, per-type max, category counts and shares, first-recency, feed category share, log M / H (`nn/model.py:52-112`).
- The rest are learnable by the history encoder: seq_dcos_*, cos_h10/h50/hall, cat_cnt_20/5, ctx_hs extras, clu_*.
- Group residuals ≤ 0 over the mix and ≤ +0.0004 over train-only: no gap.

**(c) Redundancy clusters:** see INVENTORY-6 and Appendix B. To prune from NN presets: `seq_cv_norm` (≡ invd20). For the GBDT harness: one of each rank/z/gap triplet, `dot_u` (≡ cos_u), the non-LUO it_hist_* (banned anyway), and `tte_it_*` / `it_tr_*` duplicates.

**(d) Train-random vs val discrepancy** (skew(M50) and PSI val→trH; full list in Appendix A):

| feature(s) | skew(M50) | PSI v→trH | status |
|---|---|---|---|
| it_hist_all / et1..4 (non-LUO) | +0.039..+0.025 | 0.13-0.14 | banned (same-user later) |
| it_cand_cnt_split* | −0.033 (sign flip) | 16.6 | in the gate base: INVENTORY-2 |
| it_tr_ctr / coec / pos | +0.033 / +0.031 / +0.030 | 0.17-0.23 | in-sample on train; val/test-only use is fine |
| it_hist_*_luo | +0.031 (all), +0.021..+0.032 (et*) | 0.14 | live XT/FT; LEAKAGE-7 / SHIFT-1 (non-causal popularity); not a label channel |
| it_cand_cnt_train | +0.022 | 5.09 | train self-count: INVENTORY-2 |
| trd_skde_* | +0.024 (raw +0.013) | 0.02 | unused; window supply density is denser on train (same mechanism as LEAKAGE-6) |
| tte_it_ctr / coec | +0.018 | 0.006 (1.14 is a float artefact) | gap mismatch, INVENTORY-7 |
| seq_cv_pop | +0.0125 | 0.035 | ok |
| prew_* | +0.009 | 0.013-0.020 | LEAKAGE-6 (density); the review measured +0.015 against all-M val |
| all other train-available features | ≤ 0.015, or raw gap ≤ 0.01 | ≤ 0.07 (except M: holdout has M ≥ 50) | clean |

Missingness parity (NaN rate val / test / trU / trH):
- prew_* 0.379 / 0.447 / 0.385 / 0.436;
- `ctx_nd_{knn_u, knn_u_dev, top5u, ucent}` 0.014 / 0.003 / 0.026 / 0.036. This follows the share of zero user embeddings: train 2.5%, val 1.4%, test 0.3% (INTEGRITY-6 / PREP-5).
- In the train stage, NaN becomes 0 with no indicator (PREP-4), and 0 is a legitimate value for knn_u_dev / ucent.

---------------------------------------------------------------------------------------------------------------------

## 4. Strength summary (top single features)

Val sNDCG (sAUC):
- pre_new / pre_new_z 0.3592 (0.661);
- pre2_new 0.3584;
- pre_fe 0.3578 (0.663);
- pre2_fe 0.3567;
- pre_fe_noid 0.3537;
- **ctx_hs_t10 0.3530 (0.648)**;
- ctx_hn_soft05 0.3521;
- ctx_hs_d7 0.3518 (unused);
- ctx_hn_rrw 0.3500;
- seq_max_new8 0.3497 (unused);
- ctx_hs_k128t10 0.3496 (unused);
- maxsim32_rk 0.349;
- cos_u 0.342 (0.641).

On trH (train-available), the order is the same: ctx_hs_d7 0.2738, ctx_hs_t10 0.2731, ctx_hn_soft05 0.2729, seq_dcos_hl2_rk 0.2725, ctx_hn_rrw 0.2724. The strength ranking agrees between train-random and val for every live input (Appendix A).

---------------------------------------------------------------------------------------------------------------------

## 5. Checked and correct

- **Row alignment.**
  - All 256 val / 256 test / 238 train feature files have exactly the split's candidate count; a mismatch would have failed the column assignment in `c01_cache.py`.
  - trH labels and the 4 nn_final4 holdout predictions align with the reconstructed holdout (M, offsets and labels identical).
- **The residual folds equal the fine-tune folds:** `GroupKFold(n_splits=5)` (`nn/finetune.py:180`), no shuffle, on `np.arange(n)` with feed-ordered `user_id` (`:183`), same sklearn 1.9.1. The val base mix reproduces 0.39399 (review_shift).
- **Leak status of every live NN input** (cvnd, cvnd_prew, n2c_pre, pre_final blend): no label leak, per review_leakage (TL;DR and "Checked and correct"). Known non-label asymmetries: LEAKAGE-1/3/6/7, SHIFT-1/2.
  - For the non-live features (tte_, trd_, clu_, seq_ / ctx_ extras, cos_*, cat_*, f*), my train-random vs val detector finds no excess skew except trd_skde (unused) and the tte gap (INVENTORY-7).
  - Missingness lifts behave the same on both sides (review_leakage detect table).
- **Val vs test parity.** PSI ≤ 0.06 for every feature except:
  - it_cand_cnt_split* (10.6; banned in FT);
  - pre_id_contrib / pre2_id_contrib (3.5 / 3.9; LEAKAGE-2) and prew_id_contrib (0.49);
  - als_exp5_z (0.19), it_tr_coec_rk (0.19) and it_hist_all_rk (0.15): rank or z of zero-inflated columns under item staleness;
  - hour (0.09; calendar).
- **NN train-stage transforms of the 28 cvnd inputs are sane:**
  - log1p counts reach at most 8.6;
  - no ctx column saturates beyond 1.2% of rows;
  - in-model rank/z are applied to `seq_cv_invd20` and `it_hist_all_luo` (log1p, monotone, so the ranks are unaffected).
- **The ctx_nd builder is reproducible bit-exactly** (INVENTORY-4 self-check).
- **The f1..f5 organiser encoders show no train in-sample inflation:** sAUC trH vs val within 0.004 for all five (f1 0.6230 / 0.6226).
- **The quarantine holds for data/feats:** no als_* or pre_* on train; the leaky files are only in data/feats_quarantine. The nn/feat_cache hard-link gap is LEAKAGE-4, not repeated here.

---------------------------------------------------------------------------------------------------------------------

## 6. Feature ideas (for the top gaps; exact constructions)

1. **Clip-safe prew (built)**
   - Files: `data/feats_new/{train,val,test}/prew_fe_s.npy`, `prew_new_s.npy`.
   - Construction (identical on every split): s = (x + 20)/6, NaN kept. Replace `prew_fe` / `prew_new` in `XT_PRESETS["cvnd_prew"]`, `XT_REL` and `XT_RELONLY` by the `_s` names.
   - In-model rank equals the unclipped rank; z is affine in the raw z; NaN rows still sort lowest.
   - Expected: restores a channel that is now ~70% ties. Probably small, ≤ +0.001; judge on paired train-only val with ≥ 2 seeds.
   - Apply the same map (or exempt from the clip) for `pre2_fe` / `pre2_new` before the pre2 train-stage run.
2. **pre2 train-stage input, judged correctly.** This is the only on-disk signal with a ≥ 2 SE residual over the train-only NN.
   - Construction: pretrain report §8c (window encoders, item_tab2 with every train candidate, val from v3A/v4A without test snapshots).
   - NN transform: precomputed `pre2_fe_z` / `pre2_new_z` (clip ±5) plus clip-safe raw columns through XT_REL.
   - Acceptance: train-only val > 2 SE over ≥ 2 seeds, holdout ignored (LEAKAGE-1). Gate it on the cleaned base (INVENTORY-2).
3. **Gap-matched train-label item quality** (low expected value; only if a train-stage CTR prior is wanted).
   - For a train row of MSK day d: impressions, positives and expected clicks of the item from train rows of days ≤ d − 3, plus the same-user rows of days < d. User 0 is excluded from user-keyed stats.
   - Features: log1p(imp), (pos + 1)/(ex + 1) COEC, has_imp. ex uses the position-CTR curve `pc` of `src/features_tte.py:41-43`.
   - For val/test rows: all train (= the existing `tte_it_*`).
   - Parity target: skew(M50) of the COEC ≤ 0.01 (now +0.018). Check the zero-imp share on random-like train against val (34.6%) and test (46.2%).
   - Residual evidence so far: +0.0002-0.0004 over train-only, so expect ≤ +0.0005.
4. **Clean gate base** (construction fix): `swap_luo(base)` minus `it_cand_cnt_split{,_rk,_z,_gap}`, `it_cand_cnt_train`, `hour`. Keep `tte_it_imp` as the exposure count. Measured +0.0029 ± 0.0010 on val for the base itself.
5. **Fine-tune extra-branch transform v2** (val/test only, no labels). Statistics may be fitted on val + test unlabeled rows, or on val only for consistency with the current code.
   - Add an `is_nan` flag for every extra with a NaN rate > 1% (`pre_idmax32` 55%, ctx_nd_* / seq_* 1.4%).
   - Apply a rank-gauss (quantile → normal) map instead of log1p + z for the heavy-tailed or zero-inflated extras: it_hist_et5..10_luo, seq_cv_*, it_tr_*, u_tr_ctr, it_hist_all_luo_z.
   - Replace `hour` by (sin, cos)(2π·hour/24), or drop it.
   - Dedupe `seq_cv_norm`.
   - Expected: small, because the fine-tune's value is mostly recalibration (nn_report R2). Cheap to test with one fine-tune pair.
6. **Restored near-duplicate context** (built for val/test; train via the same script). Pure n2full reproducibility. Measured residual ~0 (INVENTORY-4), so low priority.

## Appendix A: full inventory table (256 features)

Columns:
- **used by**: XT = `cvnd` train stage (`nn/train.py:30-36`); XTprew = extra in `cvnd_prew` (`:48-51`); FT = n2c_pre fine-tune extras actually loaded (`nn/logs/v9_e10_n2cp_ft.log`); GB = `data/eval_cache/base_features.json` (val harness); gate = train-fit LightGBM gate (`pretrain/gate_pre2.py:21`, `diag_trainstage.py:54`); stk* = the stopped stk2_res4_clean stacker (CURRENT_STATUS §3).
- **NaN**: rates on val / test / trH / trU.
- **resid mix**: residual over the val mix (LIN / GBM, SE in brackets).
- **resid tr-only**: residual over `nn_final4_trainonly_val` (86 features).
- **resid trH**: LIN residual over the nn_final4 train-only members on the holdout.
- **splits**: t/v/t = train/val/test present.

| feature | family | builder | splits | NaN val/test/trH/trU | used by | NN in-model analogue | sAUC val / trH | NDCG val / trH | skew(M50) | PSI v-t / v-trH | resid mix LIN (SE) | resid mix GBM (SE) | resid tr-only LIN / GBM | resid trH LIN (SE) | leak status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H | base-other | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | log1p H | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.07 | - (-) | - (-) | - / - | - (-) | per-feed constant (within-feed AUC n/a) |
| M | base-other | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | log M | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 1.31 | - (-) | - (-) | - / - | - (-) | per-feed constant (within-feed AUC n/a) |
| als_et1_exp5 | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.39/0.45/-/- | stk* |  | 0.577 / - | 0.318 / - | - | 0.02 / - | -0.00019 (0.00029) | -0.00044 (0.00037) | 0.00046 / 0.00028 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| als_exp10 | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.38/0.45/-/- | stk* |  | 0.579 / - | 0.321 / - | - | 0.02 / - | 0.00029 (0.00031) | -0.00074 (0.00037) | 0.00066 / -0.00003 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| als_exp2 | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.38/0.45/-/- | stk* |  | 0.578 / - | 0.320 / - | - | 0.02 / - | 0.00019 (0.00031) | -0.00037 (0.00036) | 0.00057 / 0.00010 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| als_exp5 | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.38/0.45/-/- | stk* |  | 0.579 / - | 0.321 / - | - | 0.02 / - | 0.00015 (0.00033) | -0.00055 (0.00037) | 0.00088 / -0.00026 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| als_exp5_rk | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.00/0.00/-/- | stk* |  | 0.580 / - | 0.326 / - | - | 0.00 / - | 0.00002 (0.00023) | -0.00073 (0.00037) | -0.00002 / -0.00012 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| als_exp5_z | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.00/0.00/-/- | stk* |  | 0.579 / - | 0.321 / - | - | 0.19 / - | 0.00001 (0.00027) | -0.00054 (0.00032) | 0.00076 / -0.00023 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| als_has | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.00/0.00/-/- | stk* |  | 0.544 / - | 0.219 / - | - | 0.02 / - | -0.00017 (0.00010) | -0.00044 (0.00031) | -0.00021 / -0.00075 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| als_last10 | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.38/0.45/-/- | stk* |  | 0.579 / - | 0.320 / - | - | 0.02 / - | 0.00031 (0.00033) | -0.00060 (0.00037) | 0.00058 / -0.00024 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| als_samecat_exp10 | als | src/features_als.py (+src/als_clean_pairs.py) | vt | 0.47/0.53/-/- | stk* |  | 0.586 / - | 0.315 / - | - | 0.01 / - | -0.00005 (0.00028) | -0.00082 (0.00035) | 0.00031 / -0.00109 | - (-) | clean ALS (CURRENT_STATUS 2b); no train version |
| attn32 | emb | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate | lse(all K) | 0.636 / 0.631 | 0.345 / 0.264 | 0.003 | 0.00 / 0.01 | -0.00016 (0.00012) | -0.00068 (0.00034) | -0.00015 / -0.00040 | -0.00017 (0.00018) | detector clean (skew +0.003, raw -0.009) |
| attn32_gap | emb | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.636 / 0.631 | 0.345 / 0.264 | 0.003 | 0.00 / 0.01 | -0.00016 (0.00012) | -0.00068 (0.00031) | -0.00015 / -0.00024 | -0.00017 (0.00018) | detector clean (skew +0.003, raw -0.009) |
| attn32_rk | emb | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.637 / 0.632 | 0.347 / 0.265 | 0.003 | 0.00 / 0.00 | -0.00031 (0.00024) | -0.00070 (0.00032) | 0.00011 / -0.00029 | -0.00009 (0.00018) | detector clean (skew +0.003, raw -0.009) |
| attn32_z | emb | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.636 / 0.631 | 0.345 / 0.264 | 0.003 | 0.00 / 0.01 | -0.00016 (0.00012) | -0.00082 (0.00033) | -0.00015 / -0.00043 | -0.00017 (0.00018) | detector clean (skew +0.003, raw -0.009) |
| cat_cnt_20 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* |  | 0.591 / 0.588 | 0.247 / 0.190 | 0.008 | 0.00 / 0.00 | -0.00003 (0.00015) | 0.00003 (0.00029) | 0.00027 / 0.00016 | -0.00011 (0.00011) | detector clean (skew +0.008, raw -0.004) |
| cat_cnt_5 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.582 / 0.583 | 0.241 / 0.188 | 0.012 | 0.00 / 0.00 | -0.00011 (0.00015) | -0.00043 (0.00029) | -0.00001 / -0.00011 | -0.00006 (0.00010) | detector clean (skew +0.012, raw +0.000) |
| cat_cnt_all | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt (+z/rank) | 0.580 / 0.578 | 0.247 / 0.191 | 0.011 | 0.00 / 0.00 | -0.00000 (0.00018) | -0.00009 (0.00030) | -0.00012 / -0.00056 | -0.00012 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| cat_cnt_all_gap | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.580 / 0.578 | 0.247 / 0.191 | 0.011 | 0.00 / 0.00 | -0.00000 (0.00018) | 0.00019 (0.00029) | - / - | -0.00012 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| cat_cnt_all_rk | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.580 / 0.578 | 0.247 / 0.191 | 0.011 | 0.00 / 0.00 | -0.00018 (0.00018) | -0.00010 (0.00030) | -0.00020 / -0.00003 | -0.00015 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| cat_cnt_all_z | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.580 / 0.578 | 0.247 / 0.191 | 0.011 | 0.00 / 0.00 | -0.00000 (0.00018) | -0.00010 (0.00027) | - / - | -0.00012 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| cat_cnt_et1 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.586 / 0.582 | 0.248 / 0.191 | 0.008 | 0.00 / 0.00 | 0.00005 (0.00019) | -0.00026 (0.00032) | - / - | -0.00008 (0.00011) | detector clean (skew +0.008, raw -0.004) |
| cat_cnt_et10 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.506 / 0.505 | 0.201 / 0.164 | 0.010 | 0.00 / 0.00 | -0.00009 (0.00010) | -0.00006 (0.00014) | - / - | 0.00000 (0.00004) | detector clean (skew +0.010, raw -0.001) |
| cat_cnt_et2 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.536 / 0.536 | 0.232 / 0.182 | 0.012 | 0.00 / 0.00 | -0.00006 (0.00006) | -0.00015 (0.00029) | - / - | 0.00001 (0.00004) | detector clean (skew +0.012, raw +0.000) |
| cat_cnt_et3 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.542 / 0.540 | 0.235 / 0.183 | 0.008 | 0.00 / 0.00 | 0.00005 (0.00008) | -0.00052 (0.00026) | - / - | -0.00004 (0.00005) | detector clean (skew +0.008, raw -0.003) |
| cat_cnt_et4 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.531 / 0.529 | 0.230 / 0.179 | 0.012 | 0.00 / 0.00 | 0.00010 (0.00006) | 0.00022 (0.00026) | - / - | -0.00004 (0.00005) | detector clean (skew +0.012, raw +0.000) |
| cat_cnt_et5 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.524 / 0.526 | 0.220 / 0.175 | 0.016 | 0.00 / 0.00 | 0.00012 (0.00011) | -0.00038 (0.00031) | - / - | -0.00001 (0.00004) | detector clean (skew +0.016, raw +0.004) |
| cat_cnt_et6 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.522 / 0.524 | 0.218 / 0.174 | 0.016 | 0.00 / 0.00 | 0.00010 (0.00011) | -0.00078 (0.00028) | - / - | -0.00002 (0.00003) | detector clean (skew +0.016, raw +0.004) |
| cat_cnt_et7 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.518 / 0.511 | 0.219 / 0.174 | 0.004 | 0.00 / 0.01 | -0.00001 (0.00004) | -0.00038 (0.00027) | - / - | 0.00001 (0.00008) | detector clean (skew +0.004, raw -0.008) |
| cat_cnt_et8 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.519 / 0.520 | 0.215 / 0.173 | 0.014 | 0.00 / 0.00 | -0.00009 (0.00009) | -0.00035 (0.00031) | - / - | 0.00005 (0.00006) | detector clean (skew +0.014, raw +0.003) |
| cat_cnt_et9 | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p ccnt_et | 0.508 / 0.504 | 0.211 / 0.169 | 0.008 | 0.00 / 0.00 | -0.00006 (0.00009) | 0.00028 (0.00023) | - / - | 0.00002 (0.00008) | detector clean (skew +0.008, raw -0.003) |
| cat_min_hpos | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | 1/(1+first) | 0.596 / 0.599 | 0.253 / 0.195 | 0.013 | 0.00 / 0.00 | -0.00014 (0.00013) | -0.00026 (0.00031) | - / - | -0.00009 (0.00008) | detector clean (skew +0.013, raw +0.001) |
| cat_share_all | cat | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate | ccnt/H | 0.580 / 0.578 | 0.247 / 0.191 | 0.011 | 0.00 / 0.01 | -0.00000 (0.00018) | 0.00010 (0.00030) | - / - | -0.00012 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| category_id | base-other | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* |  | 0.524 / 0.524 | 0.246 / 0.192 | 0.010 | 0.00 / 0.01 | 0.00002 (0.00018) | -0.00031 (0.00026) | - / - | -0.00008 (0.00011) | detector clean (skew +0.010, raw -0.002) |
| centrality | emb | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT+GB+gate |  | 0.582 / 0.587 | 0.317 / 0.246 | 0.011 | 0.00 / 0.00 | 0.00002 (0.00012) | -0.00029 (0.00031) | 0.00044 / -0.00020 | -0.00013 (0.00017) | detector clean (skew +0.011, raw -0.000) |
| clu_hcnt1024 | clu | feats_v2/clu_/build.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.592 / 0.585 | 0.267 / 0.203 | 0.002 | 0.00 / 0.00 | -0.00017 (0.00020) | -0.00046 (0.00031) | -0.00003 / 0.00006 | -0.00001 (0.00010) | detector clean (skew +0.002, raw -0.009) |
| clu_het14096 | clu | feats_v2/clu_/build.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.592 / 0.584 | 0.272 / 0.207 | 0.002 | 0.00 / 0.00 | -0.00001 (0.00012) | -0.00029 (0.00033) | -0.00015 / -0.00002 | -0.00008 (0.00008) | detector clean (skew +0.002, raw -0.010) |
| clu_hmin1024 | clu | feats_v2/clu_/build.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.609 / 0.605 | 0.280 / 0.213 | 0.006 | 0.00 / 0.00 | -0.00016 (0.00019) | -0.00033 (0.00034) | -0.00001 / -0.00011 | -0.00020 (0.00013) | detector clean (skew +0.006, raw -0.005) |
| clu_hrw4096 | clu | feats_v2/clu_/build.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.605 / 0.600 | 0.285 / 0.217 | 0.004 | 0.00 / 0.00 | -0.00023 (0.00016) | -0.00042 (0.00032) | -0.00036 / -0.00038 | -0.00016 (0.00011) | detector clean (skew +0.004, raw -0.008) |
| clu_pmi1024_mean | clu | feats_v2/clu_/build.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.605 / 0.603 | 0.280 / 0.213 | 0.007 | 0.00 / 0.01 | -0.00018 (0.00017) | -0.00045 (0.00032) | -0.00004 / -0.00009 | -0.00010 (0.00014) | detector clean (skew +0.007, raw -0.005) |
| clu_pmi11024_mean | clu | feats_v2/clu_/build.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.604 / 0.602 | 0.280 / 0.213 | 0.007 | 0.00 / 0.01 | -0.00018 (0.00020) | -0.00029 (0.00032) | -0.00002 / 0.00030 | -0.00010 (0.00014) | detector clean (skew +0.007, raw -0.005) |
| cos_h10 | cos | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate+stk* |  | 0.627 / 0.626 | 0.345 / 0.266 | 0.007 | 0.00 / 0.01 | 0.00000 (0.00012) | -0.00013 (0.00031) | 0.00032 / -0.00044 | -0.00005 (0.00019) | detector clean (skew +0.007, raw -0.005) |
| cos_h10_gap | cos | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.627 / 0.626 | 0.345 / 0.266 | 0.007 | 0.00 / 0.01 | 0.00000 (0.00012) | -0.00032 (0.00037) | - / - | -0.00005 (0.00019) | detector clean (skew +0.007, raw -0.005) |
| cos_h10_rk | cos | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.627 / 0.626 | 0.346 / 0.268 | 0.006 | 0.00 / 0.00 | -0.00023 (0.00019) | -0.00002 (0.00036) | - / - | 0.00003 (0.00020) | detector clean (skew +0.006, raw -0.005) |
| cos_h10_z | cos | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.627 / 0.626 | 0.345 / 0.266 | 0.007 | 0.00 / 0.01 | 0.00000 (0.00012) | -0.00059 (0.00030) | - / - | -0.00005 (0.00019) | detector clean (skew +0.007, raw -0.005) |
| cos_h50 | cos | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.619 / 0.614 | 0.337 / 0.257 | 0.005 | 0.00 / 0.01 | -0.00023 (0.00017) | -0.00039 (0.00034) | - / - | -0.00011 (0.00018) | detector clean (skew +0.005, raw -0.007) |
| cos_hall | cos | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.605 / 0.602 | 0.328 / 0.252 | 0.006 | 0.00 / 0.01 | -0.00021 (0.00016) | -0.00048 (0.00035) | 0.00011 / -0.00066 | -0.00028 (0.00017) | detector clean (skew +0.006, raw -0.006) |
| cos_hall_gap | cos | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.605 / 0.602 | 0.328 / 0.252 | 0.006 | 0.00 / 0.01 | -0.00021 (0.00016) | -0.00053 (0.00034) | - / - | -0.00028 (0.00017) | detector clean (skew +0.006, raw -0.006) |
| cos_hall_rk | cos | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.606 / 0.602 | 0.329 / 0.254 | 0.005 | 0.00 / 0.00 | 0.00005 (0.00025) | -0.00027 (0.00035) | - / - | -0.00022 (0.00018) | detector clean (skew +0.005, raw -0.006) |
| cos_hall_z | cos | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.605 / 0.602 | 0.328 / 0.252 | 0.006 | 0.00 / 0.01 | -0.00021 (0.00016) | -0.00020 (0.00034) | - / - | -0.00028 (0.00017) | detector clean (skew +0.006, raw -0.006) |
| cos_het1 | cos | src/features.py | tvt | 0.03/0.02/0.04/0.03 | GB+gate | max cos in type (diff. def.) | 0.611 / 0.609 | 0.330 / 0.254 | 0.007 | 0.00 / 0.01 | -0.00018 (0.00023) | -0.00036 (0.00033) | - / - | -0.00028 (0.00018) | detector clean (skew +0.007, raw -0.005) |
| cos_het10 | cos | src/features.py | tvt | 0.93/0.93/0.95/0.95 | GB+gate | max cos in type (diff. def.) | 0.508 / 0.507 | 0.208 / 0.169 | 0.010 | 0.00 / 0.00 | 0.00003 (0.00010) | -0.00013 (0.00019) | - / - | -0.00000 (0.00006) | detector clean (skew +0.010, raw -0.001) |
| cos_het2 | cos | src/features.py | tvt | 0.16/0.16/0.18/0.16 | GB+gate | max cos in type (diff. def.) | 0.552 / 0.549 | 0.296 / 0.226 | 0.009 | 0.00 / 0.00 | -0.00023 (0.00019) | -0.00068 (0.00034) | - / - | -0.00004 (0.00006) | detector clean (skew +0.009, raw -0.002) |
| cos_het3 | cos | src/features.py | tvt | 0.10/0.10/0.12/0.10 | GB+gate | max cos in type (diff. def.) | 0.560 / 0.555 | 0.304 / 0.232 | 0.005 | 0.00 / 0.00 | -0.00021 (0.00019) | -0.00074 (0.00034) | - / - | -0.00005 (0.00009) | detector clean (skew +0.005, raw -0.007) |
| cos_het4 | cos | src/features.py | tvt | 0.24/0.23/0.27/0.25 | GB+gate | max cos in type (diff. def.) | 0.539 / 0.532 | 0.285 / 0.218 | 0.006 | 0.00 / 0.01 | -0.00004 (0.00013) | -0.00038 (0.00035) | - / - | -0.00009 (0.00007) | detector clean (skew +0.006, raw -0.006) |
| cos_het5 | cos | src/features.py | tvt | 0.37/0.38/0.38/0.38 | GB+gate | max cos in type (diff. def.) | 0.542 / 0.544 | 0.270 / 0.211 | 0.014 | 0.00 / 0.00 | -0.00011 (0.00011) | -0.00026 (0.00034) | - / - | -0.00008 (0.00007) | detector clean (skew +0.014, raw +0.002) |
| cos_het6 | cos | src/features.py | tvt | 0.41/0.42/0.42/0.43 | GB+gate | max cos in type (diff. def.) | 0.539 / 0.542 | 0.266 / 0.208 | 0.013 | 0.00 / 0.00 | -0.00011 (0.00013) | -0.00080 (0.00034) | - / - | -0.00008 (0.00008) | detector clean (skew +0.013, raw +0.002) |
| cos_het7 | cos | src/features.py | tvt | 0.35/0.34/0.41/0.38 | GB+gate | max cos in type (diff. def.) | 0.523 / 0.511 | 0.265 / 0.201 | -0.003 | 0.00 / 0.02 | -0.00022 (0.00012) | -0.00026 (0.00027) | - / - | -0.00008 (0.00009) | detector clean (skew -0.003, raw -0.014) |
| cos_het8 | cos | src/features.py | tvt | 0.49/0.50/0.49/0.50 | GB+gate | max cos in type (diff. def.) | 0.533 / 0.534 | 0.256 / 0.202 | 0.014 | 0.00 / 0.00 | -0.00001 (0.00011) | -0.00068 (0.00031) | - / - | 0.00000 (0.00007) | detector clean (skew +0.014, raw +0.002) |
| cos_het9 | cos | src/features.py | tvt | 0.60/0.59/0.65/0.62 | GB+gate | max cos in type (diff. def.) | 0.510 / 0.504 | 0.237 / 0.186 | 0.003 | 0.00 / 0.01 | 0.00004 (0.00013) | 0.00005 (0.00028) | - / - | -0.00003 (0.00010) | detector clean (skew +0.003, raw -0.008) |
| cos_u | cos | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | cos_u (+z/rank) | 0.641 / 0.635 | 0.342 / 0.259 | 0.002 | 0.00 / 0.00 | -0.00036 (0.00016) | 0.00032 (0.00031) | 0.00008 / -0.00014 | -0.00016 (0.00020) | detector clean (skew +0.002, raw -0.010) |
| cos_u_gap | cos | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.641 / 0.635 | 0.342 / 0.259 | 0.002 | 0.00 / 0.00 | -0.00036 (0.00016) | 0.00003 (0.00032) | - / - | -0.00016 (0.00020) | detector clean (skew +0.002, raw -0.010) |
| cos_u_rk | cos | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | rank(cos_u) | 0.641 / 0.635 | 0.342 / 0.259 | 0.002 | 0.00 / 0.00 | 0.00000 (0.00027) | -0.00038 (0.00036) | - / - | -0.00016 (0.00019) | detector clean (skew +0.002, raw -0.010) |
| cos_u_z | cos | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | z(cos_u) | 0.641 / 0.635 | 0.342 / 0.259 | 0.002 | 0.00 / 0.00 | -0.00036 (0.00016) | -0.00005 (0.00030) | - / - | -0.00016 (0.00020) | detector clean (skew +0.002, raw -0.010) |
| ctx_hn_bestrk | ctx_hn | feats_v2/ctx_/build_ctx3.py | tvt | 0.01/0.01/0.03/0.01 | XT |  | 0.633 / 0.628 | 0.271 / 0.223 | -0.000 | 0.00 / 0.01 | -0.00019 (0.00021) | -0.00040 (0.00031) | - / - | -0.00010 (0.00011) | detector clean (skew -0.000, raw -0.012) |
| ctx_hn_e1_rrw | ctx_hn | feats_v2/ctx_/build_ctx3.py | tvt | 0.03/0.02/0.04/0.03 | XT |  | 0.641 / 0.644 | 0.346 / 0.271 | 0.007 | 0.00 / 0.05 | -0.00024 (0.00017) | -0.00063 (0.00034) | - / - | -0.00017 (0.00018) | detector clean (skew +0.007, raw -0.005) |
| ctx_hn_margin | ctx_hn | feats_v2/ctx_/build_ctx3.py | tvt | 0.01/0.01/0.03/0.01 | XT |  | 0.638 / 0.633 | 0.337 / 0.258 | -0.001 | 0.00 / 0.01 | -0.00019 (0.00024) | -0.00005 (0.00035) | - / - | -0.00011 (0.00015) | detector clean (skew -0.001, raw -0.013) |
| ctx_hn_rrw | ctx_hn | feats_v2/ctx_/build_ctx3.py | tvt | 0.01/0.01/0.03/0.01 | XT |  | 0.643 / 0.645 | 0.350 / 0.272 | 0.004 | 0.00 / 0.05 | -0.00003 (0.00015) | -0.00043 (0.00037) | - / - | -0.00021 (0.00017) | detector clean (skew +0.004, raw -0.007) |
| ctx_hn_soft05 | ctx_hn | feats_v2/ctx_/build_ctx3.py | tvt | 0.01/0.01/0.03/0.01 | XT |  | 0.649 / 0.648 | 0.352 / 0.273 | 0.003 | 0.00 / 0.01 | 0.00004 (0.00021) | -0.00034 (0.00033) | - / - | -0.00019 (0.00017) | detector clean (skew +0.003, raw -0.008) |
| ctx_hn_top1w | ctx_hn | feats_v2/ctx_/build_ctx3.py | tvt | 0.01/0.01/0.03/0.01 | XT |  | 0.580 / 0.553 | 0.321 / 0.235 | 0.002 | 0.00 / 0.02 | 0.00004 (0.00019) | -0.00084 (0.00047) | - / - | -0.00012 (0.00011) | detector clean (skew +0.002, raw -0.010) |
| ctx_hs_d7 | ctx_hs | feats_v2/ctx_/build_ctx4.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.643 / 0.645 | 0.352 / 0.274 | 0.006 | 0.00 / 0.01 | 0.00013 (0.00019) | -0.00084 (0.00031) | 0.00006 / -0.00072 | -0.00016 (0.00021) | detector clean (skew +0.006, raw -0.006) |
| ctx_hs_e1 | ctx_hs | feats_v2/ctx_/build_ctx4.py | tvt | 0.03/0.02/0.04/0.03 | XT |  | 0.646 / 0.645 | 0.349 / 0.271 | 0.005 | 0.00 / 0.00 | -0.00012 (0.00023) | -0.00012 (0.00031) | - / - | -0.00017 (0.00018) | detector clean (skew +0.005, raw -0.007) |
| ctx_hs_e1k64 | ctx_hs | feats_v2/ctx_/build_ctx4.py | tvt | 0.03/0.02/0.04/0.03 | none |  | 0.649 / 0.644 | 0.346 / 0.266 | 0.003 | 0.00 / 0.00 | -0.00016 (0.00024) | 0.00002 (0.00033) | 0.00018 / -0.00037 | -0.00023 (0.00016) | detector clean (skew +0.003, raw -0.009) |
| ctx_hs_k128 | ctx_hs | feats_v2/ctx_/build_ctx4.py | tvt | 0.01/0.01/0.03/0.01 | XT |  | 0.653 / 0.649 | 0.349 / 0.268 | 0.002 | 0.00 / 0.01 | 0.00002 (0.00022) | -0.00006 (0.00030) | - / - | -0.00016 (0.00015) | detector clean (skew +0.002, raw -0.010) |
| ctx_hs_k128t10 | ctx_hs | feats_v2/ctx_/build_ctx4.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.652 / 0.648 | 0.350 / 0.268 | 0.001 | 0.00 / 0.01 | -0.00010 (0.00018) | -0.00068 (0.00032) | 0.00013 / 0.00013 | -0.00021 (0.00016) | detector clean (skew +0.001, raw -0.010) |
| ctx_hs_k256 | ctx_hs | feats_v2/ctx_/build_ctx4.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.649 / 0.644 | 0.343 / 0.262 | 0.001 | 0.00 / 0.01 | -0.00008 (0.00024) | -0.00012 (0.00027) | 0.00004 / 0.00003 | -0.00020 (0.00015) | detector clean (skew +0.001, raw -0.011) |
| ctx_hs_mutual | ctx_hs | feats_v2/ctx_/build_ctx4.py | tvt | 0.01/0.01/0.03/0.01 | XT |  | 0.566 / 0.543 | 0.308 / 0.228 | 0.003 | 0.00 / 0.01 | -0.00026 (0.00018) | -0.00064 (0.00038) | - / - | -0.00011 (0.00013) | detector clean (skew +0.003, raw -0.008) |
| ctx_hs_t10 | ctx_hs | feats_v2/ctx_/build_ctx4.py | tvt | 0.01/0.01/0.03/0.01 | XT |  | 0.648 / 0.647 | 0.353 / 0.273 | 0.003 | 0.00 / 0.01 | 0.00003 (0.00017) | -0.00018 (0.00031) | - / - | -0.00019 (0.00020) | detector clean (skew +0.003, raw -0.008) |
| ctx_nd_hcent | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.01/0.01/0.03/0.01 | XT+FT+stk* |  | 0.605 / 0.602 | 0.339 / 0.261 | 0.004 | 0.00 / 0.01 | -0.00013 (0.00021) | 0.00025 (0.00035) | - / - | 0.00002 (0.00018) | detector clean (skew +0.004, raw -0.007) |
| ctx_nd_knn_f1 | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+stk* |  | 0.607 / 0.619 | 0.322 / 0.253 | 0.008 | 0.00 / 0.00 | 0.00021 (0.00022) | -0.00039 (0.00035) | - / - | -0.00014 (0.00017) | detector clean (skew +0.008, raw -0.003) |
| ctx_nd_knn_h | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.01/0.01/0.03/0.01 | XT+FT+stk* |  | 0.612 / 0.621 | 0.322 / 0.251 | 0.001 | 0.00 / 0.01 | 0.00001 (0.00025) | -0.00033 (0.00038) | - / - | -0.00006 (0.00015) | detector clean (skew +0.001, raw -0.011) |
| ctx_nd_knn_h_dev | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.01/0.01/0.03/0.01 | XT+FT+stk* |  | 0.602 / 0.590 | 0.324 / 0.245 | 0.004 | 0.00 / 0.01 | -0.00021 (0.00021) | -0.00035 (0.00035) | - / - | -0.00003 (0.00010) | detector clean (skew +0.004, raw -0.007) |
| ctx_nd_knn_u | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.01/0.00/0.04/0.03 | XT+FT+stk* |  | 0.600 / 0.611 | 0.316 / 0.248 | 0.005 | 0.02 / 0.02 | 0.00009 (0.00024) | -0.00031 (0.00031) | - / - | -0.00021 (0.00018) | detector clean (skew +0.005, raw -0.007) |
| ctx_nd_knn_u_dev | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.01/0.00/0.04/0.03 | XT+FT+stk* |  | 0.602 / 0.592 | 0.323 / 0.245 | 0.003 | 0.02 / 0.02 | 0.00025 (0.00024) | -0.00036 (0.00037) | - / - | -0.00005 (0.00013) | detector clean (skew +0.003, raw -0.008) |
| ctx_nd_max | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+stk* |  | 0.573 / 0.566 | 0.289 / 0.227 | 0.006 | 0.00 / 0.01 | 0.00036 (0.00025) | -0.00011 (0.00037) | - / - | 0.00012 (0.00019) | detector clean (skew +0.006, raw -0.006) |
| ctx_nd_top5h | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+stk* |  | 0.598 / 0.593 | 0.328 / 0.254 | 0.003 | 0.00 / 0.00 | -0.00013 (0.00018) | -0.00029 (0.00037) | - / - | -0.00013 (0.00021) | detector clean (skew +0.003, raw -0.008) |
| ctx_nd_top5u | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.01/0.00/0.04/0.03 | XT+FT+stk* |  | 0.594 / 0.594 | 0.323 / 0.250 | 0.008 | 0.02 / 0.02 | 0.00002 (0.00014) | -0.00028 (0.00032) | - / - | -0.00018 (0.00018) | detector clean (skew +0.008, raw -0.004) |
| ctx_nd_ucent | ctx_nd | feats_v2/ctx_/build_ctx.py | tvt | 0.01/0.00/0.04/0.03 | XT+FT+stk* |  | 0.565 / 0.553 | 0.309 / 0.233 | 0.001 | 0.02 / 0.02 | -0.00048 (0.00020) | -0.00056 (0.00037) | - / - | -0.00007 (0.00014) | detector clean (skew +0.001, raw -0.010) |
| dot_hall | emb | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.605 / 0.602 | 0.328 / 0.252 | 0.006 | 0.00 / 0.01 | -0.00021 (0.00016) | -0.00013 (0.00033) | - / - | -0.00027 (0.00017) | detector clean (skew +0.006, raw -0.006) |
| dot_u | emb | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | dot_u | 0.641 / 0.635 | 0.342 / 0.259 | 0.001 | 0.00 / 0.00 | -0.00030 (0.00016) | 0.00041 (0.00031) | - / - | -0.00014 (0.00020) | detector clean (skew +0.001, raw -0.010) |
| emb_missing | emb | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | cmiss | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.00 | -0.00001 (0.00002) | -0.00002 (0.00006) | - / - | -0.00003 (0.00005) | detector clean (skew +0.012, raw +0.000) |
| f1 | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | raw/z/rank | 0.623 / 0.623 | 0.328 / 0.254 | 0.008 | 0.00 / 0.00 | 0.00056 (0.00021) | -0.00026 (0.00034) | 0.00003 / 0.00018 | -0.00009 (0.00015) | detector clean (skew +0.008, raw -0.003) |
| f1_gap | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.623 / 0.623 | 0.328 / 0.254 | 0.008 | 0.00 / 0.00 | 0.00055 (0.00021) | 0.00021 (0.00035) | - / - | -0.00009 (0.00015) | detector clean (skew +0.008, raw -0.003) |
| f1_rk | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | rank | 0.623 / 0.623 | 0.328 / 0.254 | 0.008 | 0.00 / 0.00 | 0.00005 (0.00021) | 0.00004 (0.00033) | - / - | -0.00001 (0.00014) | detector clean (skew +0.008, raw -0.003) |
| f1_z | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | z | 0.623 / 0.623 | 0.328 / 0.254 | 0.008 | 0.00 / 0.00 | 0.00055 (0.00021) | -0.00004 (0.00035) | - / - | -0.00009 (0.00015) | detector clean (skew +0.008, raw -0.003) |
| f2 | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | raw/z/rank | 0.506 / 0.509 | 0.256 / 0.198 | 0.015 | 0.00 / 0.01 | 0.00010 (0.00025) | 0.00019 (0.00029) | - / - | 0.00003 (0.00011) | detector clean (skew +0.015, raw +0.004) |
| f2_gap | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.506 / 0.509 | 0.256 / 0.198 | 0.015 | 0.00 / 0.00 | 0.00010 (0.00025) | -0.00003 (0.00031) | - / - | 0.00003 (0.00011) | detector clean (skew +0.015, raw +0.004) |
| f2_rk | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | rank | 0.506 / 0.509 | 0.256 / 0.198 | 0.015 | 0.00 / 0.00 | 0.00020 (0.00026) | 0.00015 (0.00031) | - / - | -0.00001 (0.00010) | detector clean (skew +0.015, raw +0.004) |
| f2_z | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | z | 0.506 / 0.509 | 0.256 / 0.198 | 0.015 | 0.00 / 0.00 | 0.00010 (0.00025) | 0.00032 (0.00029) | - / - | 0.00003 (0.00011) | detector clean (skew +0.015, raw +0.004) |
| f3 | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | raw/z/rank | 0.601 / 0.602 | 0.252 / 0.194 | 0.011 | 0.00 / 0.00 | 0.00013 (0.00013) | 0.00011 (0.00028) | - / - | 0.00003 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| f3_gap | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.601 / 0.602 | 0.252 / 0.194 | 0.011 | 0.00 / 0.00 | 0.00012 (0.00013) | 0.00000 (0.00027) | - / - | 0.00003 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| f3_rk | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | rank | 0.601 / 0.602 | 0.252 / 0.194 | 0.011 | 0.00 / 0.00 | -0.00031 (0.00019) | -0.00019 (0.00028) | - / - | -0.00006 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| f3_z | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | z | 0.601 / 0.602 | 0.252 / 0.194 | 0.011 | 0.00 / 0.00 | 0.00012 (0.00013) | 0.00010 (0.00027) | - / - | 0.00003 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| f4 | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | raw/z/rank | 0.545 / 0.546 | 0.294 / 0.226 | 0.019 | 0.00 / 0.00 | 0.00003 (0.00023) | -0.00009 (0.00042) | 0.00057 / 0.00092 | -0.00006 (0.00013) | detector clean (skew +0.019, raw +0.008) |
| f4_gap | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.545 / 0.546 | 0.294 / 0.226 | 0.019 | 0.00 / 0.00 | 0.00003 (0.00023) | 0.00052 (0.00040) | - / - | -0.00006 (0.00013) | detector clean (skew +0.019, raw +0.008) |
| f4_rk | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | rank | 0.545 / 0.546 | 0.294 / 0.226 | 0.019 | 0.00 / 0.00 | -0.00005 (0.00020) | 0.00050 (0.00039) | - / - | -0.00000 (0.00007) | detector clean (skew +0.019, raw +0.008) |
| f4_z | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | z | 0.545 / 0.546 | 0.294 / 0.226 | 0.019 | 0.00 / 0.00 | 0.00003 (0.00023) | 0.00019 (0.00039) | - / - | -0.00006 (0.00013) | detector clean (skew +0.019, raw +0.008) |
| f5 | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate+stk* | raw/z/rank | 0.561 / 0.564 | 0.295 / 0.226 | 0.015 | 0.00 / 0.00 | -0.00014 (0.00024) | 0.00024 (0.00033) | - / - | 0.00001 (0.00009) | detector clean (skew +0.015, raw +0.004) |
| f5_gap | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.561 / 0.564 | 0.295 / 0.226 | 0.015 | 0.00 / 0.01 | -0.00014 (0.00024) | 0.00001 (0.00034) | - / - | 0.00001 (0.00009) | detector clean (skew +0.015, raw +0.004) |
| f5_rk | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | rank | 0.561 / 0.564 | 0.295 / 0.226 | 0.015 | 0.00 / 0.00 | -0.00017 (0.00024) | -0.00020 (0.00034) | - / - | -0.00000 (0.00005) | detector clean (skew +0.015, raw +0.004) |
| f5_z | f1-5 | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | z | 0.561 / 0.564 | 0.295 / 0.226 | 0.015 | 0.00 / 0.00 | -0.00014 (0.00024) | -0.00013 (0.00032) | - / - | 0.00001 (0.00009) | detector clean (skew +0.015, raw +0.004) |
| feed_cat_share | cat | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | fcs | 0.579 / 0.582 | 0.238 / 0.190 | 0.011 | 0.00 / 0.00 | 0.00009 (0.00022) | 0.00000 (0.00031) | - / - | -0.00009 (0.00012) | detector clean (skew +0.011, raw -0.001) |
| hist_n_uniq_cat | base-other | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.01 | 0.00000 (0.00000) | -0.00019 (0.00024) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| hour | base-other | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT+GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.09 / 0.05 | 0.00000 (0.00000) | -0.00019 (0.00019) | 0.00000 / -0.00024 | 0.00000 (0.00000) | calendar shift val->test (SHIFT-7, PSI 0.09) |
| it_cand_cnt_split | it_cand | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.503 / 0.458 | 0.209 / 0.197 | -0.033 | 10.57 / 16.62 | 0.00006 (0.00018) | -0.00043 (0.00029) | - / - | 0.00018 (0.00030) | BANNED in FT: transductive over split incl. same-user later feeds (finetune.py ITEM_CLEAN); train PSI 16.6 |
| it_cand_cnt_split_gap | it_cand | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.503 / 0.458 | 0.209 / 0.197 | -0.033 | 0.25 / 0.25 | 0.00006 (0.00018) | 0.00003 (0.00032) | - / - | 0.00018 (0.00030) | BANNED in FT: transductive over split incl. same-user later feeds (finetune.py ITEM_CLEAN); train PSI 16.6 |
| it_cand_cnt_split_rk | it_cand | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.503 / 0.458 | 0.209 / 0.197 | -0.033 | 0.93 / 1.88 | -0.00002 (0.00018) | 0.00033 (0.00032) | - / - | 0.00039 (0.00028) | BANNED in FT: transductive over split incl. same-user later feeds (finetune.py ITEM_CLEAN); train PSI 16.6 |
| it_cand_cnt_split_z | it_cand | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.503 / 0.458 | 0.209 / 0.197 | -0.033 | 0.37 / 0.45 | 0.00006 (0.00018) | -0.00013 (0.00029) | - / - | 0.00018 (0.00030) | BANNED in FT: transductive over split incl. same-user later feeds (finetune.py ITEM_CLEAN); train PSI 16.6 |
| it_cand_cnt_train | it_cand | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT+GB+gate |  | 0.534 / 0.542 | 0.298 / 0.236 | 0.022 | 0.06 / 5.09 | 0.00019 (0.00022) | -0.00026 (0.00037) | 0.00014 / 0.00016 | 0.00018 (0.00030) | val/test-safe; train rows count themselves (min 1; PSI val->train 5.1) |
| it_hist_all | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.557 / 0.581 | 0.304 / 0.242 | 0.039 | 0.02 / 0.14 | -0.00016 (0.00016) | -0.00035 (0.00033) | - / - | 0.00003 (0.00025) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_all_gap | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.557 / 0.581 | 0.304 / 0.242 | 0.039 | 0.00 / 0.10 | -0.00016 (0.00016) | 0.00003 (0.00035) | - / - | 0.00003 (0.00025) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_all_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+gate+stk* |  | 0.556 / 0.572 | 0.304 / 0.241 | 0.031 | 0.02 / 0.14 | -0.00004 (0.00017) | -0.00050 (0.00033) | - / - | -0.00001 (0.00010) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_all_luo_gap | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.556 / 0.572 | 0.304 / 0.241 | 0.031 | 0.00 / 0.10 | -0.00004 (0.00017) | 0.00007 (0.00035) | - / - | -0.00001 (0.00010) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_all_luo_rk | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.557 / 0.572 | 0.310 / 0.243 | 0.030 | 0.00 / 0.00 | -0.00002 (0.00019) | -0.00065 (0.00033) | -0.00019 / -0.00040 | 0.00001 (0.00009) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_all_luo_z | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.556 / 0.572 | 0.304 / 0.241 | 0.031 | 0.01 / 0.03 | -0.00004 (0.00017) | -0.00005 (0.00033) | - / - | -0.00001 (0.00010) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_all_rk | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.557 / 0.581 | 0.304 / 0.242 | 0.039 | 0.15 / 0.22 | -0.00015 (0.00019) | -0.00035 (0.00031) | - / - | 0.00037 (0.00023) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_all_z | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.557 / 0.581 | 0.304 / 0.242 | 0.039 | 0.01 / 0.03 | -0.00016 (0.00016) | -0.00003 (0.00033) | - / - | 0.00003 (0.00025) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et1 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.558 / 0.584 | 0.303 / 0.242 | 0.041 | 0.02 / 0.13 | -0.00019 (0.00017) | -0.00071 (0.00032) | - / - | 0.00003 (0.00028) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et10 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.513 / 0.522 | 0.237 / 0.199 | 0.021 | 0.00 / 0.00 | -0.00005 (0.00009) | -0.00016 (0.00028) | - / - | -0.00009 (0.00016) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et10_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.513 / 0.522 | 0.237 / 0.199 | 0.021 | 0.00 / 0.00 | -0.00005 (0.00009) | -0.00016 (0.00028) | - / - | -0.00009 (0.00016) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et1_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+gate+stk* |  | 0.557 / 0.575 | 0.303 / 0.241 | 0.032 | 0.02 / 0.13 | 0.00003 (0.00018) | -0.00051 (0.00032) | - / - | -0.00000 (0.00007) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et2 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.533 / 0.546 | 0.277 / 0.225 | 0.026 | 0.01 / 0.06 | 0.00016 (0.00013) | -0.00035 (0.00030) | - / - | 0.00004 (0.00015) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et2_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+gate+stk* |  | 0.533 / 0.543 | 0.277 / 0.224 | 0.023 | 0.01 / 0.06 | 0.00016 (0.00013) | -0.00015 (0.00029) | - / - | -0.00008 (0.00007) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et3 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.531 / 0.544 | 0.276 / 0.226 | 0.025 | 0.00 / 0.04 | -0.00006 (0.00014) | -0.00067 (0.00030) | - / - | 0.00003 (0.00011) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et3_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+gate+stk* |  | 0.531 / 0.540 | 0.276 / 0.225 | 0.021 | 0.00 / 0.04 | -0.00001 (0.00015) | -0.00083 (0.00030) | - / - | -0.00004 (0.00011) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et4 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.528 / 0.541 | 0.271 / 0.223 | 0.025 | 0.00 / 0.03 | 0.00011 (0.00016) | -0.00034 (0.00031) | - / - | 0.00008 (0.00013) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et4_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+gate+stk* |  | 0.528 / 0.539 | 0.271 / 0.222 | 0.023 | 0.00 / 0.03 | 0.00012 (0.00017) | -0.00037 (0.00030) | - / - | -0.00001 (0.00006) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et5 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.513 / 0.520 | 0.241 / 0.204 | 0.020 | 0.00 / 0.00 | -0.00029 (0.00018) | -0.00066 (0.00039) | - / - | 0.00004 (0.00012) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et5_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.512 / 0.518 | 0.241 / 0.203 | 0.018 | 0.00 / 0.00 | -0.00023 (0.00017) | -0.00076 (0.00038) | -0.00009 / -0.00005 | -0.00002 (0.00008) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et6 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.511 / 0.518 | 0.240 / 0.202 | 0.020 | 0.00 / 0.00 | -0.00018 (0.00015) | -0.00060 (0.00036) | - / - | -0.00002 (0.00009) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et6_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.511 / 0.516 | 0.240 / 0.202 | 0.018 | 0.00 / 0.00 | -0.00022 (0.00014) | -0.00060 (0.00036) | - / - | 0.00007 (0.00009) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et7 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.513 / 0.517 | 0.244 / 0.200 | 0.015 | 0.00 / 0.01 | 0.00005 (0.00012) | -0.00042 (0.00029) | - / - | 0.00006 (0.00007) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et7_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.513 / 0.517 | 0.244 / 0.200 | 0.015 | 0.00 / 0.01 | 0.00005 (0.00012) | -0.00042 (0.00029) | - / - | 0.00003 (0.00007) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et8 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.508 / 0.512 | 0.230 / 0.194 | 0.016 | 0.00 / 0.00 | -0.00030 (0.00016) | -0.00092 (0.00030) | - / - | -0.00013 (0.00009) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et8_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.508 / 0.511 | 0.230 / 0.194 | 0.015 | 0.00 / 0.00 | -0.00030 (0.00016) | -0.00101 (0.00030) | - / - | -0.00011 (0.00012) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_hist_et9 | it_hist (non-LUO) | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB |  | 0.505 / 0.503 | 0.221 / 0.182 | 0.008 | 0.00 / 0.00 | 0.00008 (0.00007) | -0.00017 (0.00021) | - / - | -0.00002 (0.00011) | BANNED: same-user later snapshots (CURRENT_STATUS sec 2) |
| it_hist_et9_luo | it_hist_luo | src/features_luo.py | tvt | 0.00/0.00/0.00/0.00 | FT+gate+stk* |  | 0.505 / 0.503 | 0.221 / 0.182 | 0.008 | 0.00 / 0.00 | 0.00008 (0.00007) | -0.00025 (0.00021) | - / - | -0.00002 (0.00012) | no label leak; non-causal cross-user popularity on train (LEAKAGE-7, SHIFT-1/2); user0 later kept (LEAKAGE-3) |
| it_tr_coec | it_tr | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT |  | 0.507 / 0.527 | 0.285 / 0.237 | 0.031 | 0.05 / 0.20 | 0.00030 (0.00023) | 0.00086 (0.00038) | 0.00027 / -0.00002 | 0.00047 (0.00037) | val/test-safe; train rows in-sample (leave-own-row-out only), LEAKAGE detector |
| it_tr_coec_gap | it_tr | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT |  | 0.507 / 0.527 | 0.285 / 0.237 | 0.031 | 0.02 / 0.09 | 0.00030 (0.00023) | 0.00067 (0.00035) | 0.00027 / 0.00028 | 0.00047 (0.00037) | val/test-safe; train rows in-sample (leave-own-row-out only), LEAKAGE detector |
| it_tr_coec_rk | it_tr | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT |  | 0.507 / 0.527 | 0.285 / 0.237 | 0.031 | 0.19 / 0.23 | 0.00022 (0.00025) | 0.00023 (0.00036) | 0.00003 / 0.00013 | -0.00007 (0.00017) | val/test-safe; train rows in-sample (leave-own-row-out only), LEAKAGE detector |
| it_tr_coec_z | it_tr | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT |  | 0.507 / 0.527 | 0.285 / 0.237 | 0.031 | 0.03 / 0.05 | 0.00030 (0.00023) | 0.00089 (0.00035) | 0.00027 / -0.00011 | 0.00047 (0.00037) | val/test-safe; train rows in-sample (leave-own-row-out only), LEAKAGE detector |
| it_tr_ctr | it_tr | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT |  | 0.510 / 0.533 | 0.284 / 0.235 | 0.033 | 0.04 / 0.17 | 0.00015 (0.00021) | 0.00028 (0.00034) | 0.00037 / 0.00035 | 0.00015 (0.00035) | val/test-safe; train rows in-sample (leave-own-row-out only), LEAKAGE detector |
| it_tr_pos | it_tr | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT |  | 0.526 / 0.544 | 0.263 / 0.222 | 0.030 | 0.01 / 0.03 | 0.00023 (0.00014) | -0.00020 (0.00030) | 0.00027 / -0.00019 | 0.00032 (0.00027) | val/test-safe; train rows in-sample (leave-own-row-out only), LEAKAGE detector |
| item_norm | emb | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate | log1p|c| | 0.501 / 0.499 | 0.289 / 0.224 | 0.010 | 0.00 / 0.00 | 0.00028 (0.00017) | 0.00005 (0.00031) | - / - | -0.00004 (0.00013) | detector clean (skew +0.010, raw -0.001) |
| maxsim32 | emb | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate+stk* | smax32 | 0.648 / 0.640 | 0.348 / 0.265 | 0.000 | 0.00 / 0.01 | -0.00013 (0.00021) | -0.00033 (0.00034) | - / - | -0.00008 (0.00016) | detector clean (skew +0.000, raw -0.012) |
| maxsim32_gap | emb | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.648 / 0.640 | 0.348 / 0.265 | 0.000 | 0.00 / 0.01 | -0.00013 (0.00021) | -0.00054 (0.00033) | - / - | -0.00008 (0.00016) | detector clean (skew +0.000, raw -0.012) |
| maxsim32_rk | emb | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.648 / 0.640 | 0.349 / 0.267 | -0.000 | 0.00 / 0.00 | -0.00021 (0.00027) | -0.00040 (0.00036) | - / - | -0.00003 (0.00017) | detector clean (skew -0.000, raw -0.012) |
| maxsim32_z | emb | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate |  | 0.648 / 0.640 | 0.348 / 0.265 | 0.000 | 0.00 / 0.01 | -0.00013 (0.00021) | -0.00040 (0.00032) | - / - | -0.00008 (0.00016) | detector clean (skew +0.000, raw -0.012) |
| meansim32 | emb | src/features.py | tvt | 0.01/0.01/0.03/0.01 | GB+gate | smean(all K) | 0.621 / 0.618 | 0.339 / 0.260 | 0.005 | 0.00 / 0.01 | -0.00015 (0.00011) | -0.00040 (0.00035) | - / - | -0.00014 (0.00018) | detector clean (skew +0.005, raw -0.006) |
| pre2_fe | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.00/0.00/0.54/0.52 | gate |  | 0.662 / 0.576 | 0.357 / 0.216 | -0.076 | 0.00 / 7.75 | -0.00001 (0.00011) | -0.00012 (0.00032) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre2_fe_noid | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.00/0.00/0.54/0.52 | gate |  | 0.655 / 0.571 | 0.354 / 0.214 | -0.076 | 0.00 / 7.73 | -0.00006 (0.00013) | 0.00019 (0.00032) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre2_fe_z | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.00/0.00/0.54/0.52 | gate |  | 0.662 / 0.576 | 0.357 / 0.216 | -0.076 | 0.00 / 7.73 | -0.00001 (0.00011) | -0.00025 (0.00032) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre2_hmax32 | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.00/0.00/0.54/0.52 | gate |  | 0.649 / 0.565 | 0.345 / 0.209 | -0.076 | 0.00 / 7.73 | -0.00004 (0.00024) | -0.00031 (0.00035) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre2_hmax8 | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.00/0.00/0.54/0.52 | gate |  | 0.639 / 0.564 | 0.347 / 0.211 | -0.069 | 0.00 / 7.73 | -0.00023 (0.00017) | -0.00055 (0.00033) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre2_id_contrib | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.00/0.00/0.54/0.52 | gate |  | 0.524 / 0.499 | 0.302 / 0.193 | -0.017 | 3.87 / 11.35 | 0.00023 (0.00019) | -0.00011 (0.00036) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre2_idmax32 | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.55/0.60/0.76/0.70 | gate |  | 0.543 / 0.522 | 0.295 / 0.194 | -0.006 | 0.01 / 0.21 | 0.00027 (0.00016) | -0.00018 (0.00032) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre2_new | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.00/0.00/0.54/0.52 | gate |  | 0.661 / 0.576 | 0.358 / 0.217 | -0.076 | 0.00 / 7.74 | 0.00005 (0.00012) | -0.00013 (0.00032) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre2_new_z | pre2 | pretrain/assemble.py --prefix pre2_ (val/test), pretrain/assemble_train.py (train) | tvt | 0.00/0.00/0.54/0.52 | gate |  | 0.661 / 0.576 | 0.358 / 0.217 | -0.076 | 0.00 / 7.73 | 0.00005 (0.00012) | 0.00016 (0.00028) | - / - | - (-) | rebuild in progress (train file rewritten 15:07); id_contrib LEAKAGE-2 |
| pre_fe | pre | pretrain/assemble.py | vt | 0.00/0.00/-/- | FT |  | 0.663 / - | 0.358 / - | - | 0.00 / - | 0.00004 (0.00011) | 0.00002 (0.00030) | 0.00057 / -0.00014 | - (-) | val/test clean; train version quarantined (8b leak) |
| pre_fe_noid | pre | pretrain/assemble.py | vt | 0.00/0.00/-/- | FT |  | 0.656 / - | 0.354 / - | - | 0.00 / - | 0.00004 (0.00011) | -0.00013 (0.00031) | 0.00016 / -0.00008 | - (-) | val/test clean; train version quarantined (8b leak) |
| pre_fe_z | pre | pretrain/assemble.py | vt | 0.00/0.00/-/- | FT |  | 0.663 / - | 0.358 / - | - | 0.00 / - | 0.00005 (0.00010) | -0.00043 (0.00028) | 0.00056 / -0.00038 | - (-) | val/test clean; train version quarantined (8b leak) |
| pre_hmax32 | pre | pretrain/assemble.py | vt | 0.00/0.00/-/- | FT |  | 0.649 / - | 0.346 / - | - | 0.00 / - | 0.00000 (0.00024) | -0.00034 (0.00035) | 0.00033 / -0.00015 | - (-) | val/test clean; train version quarantined (8b leak) |
| pre_hmax8 | pre | pretrain/assemble.py | vt | 0.00/0.00/-/- | FT |  | 0.639 / - | 0.347 / - | - | 0.00 / - | -0.00007 (0.00015) | -0.00054 (0.00034) | 0.00029 / -0.00065 | - (-) | val/test clean; train version quarantined (8b leak) |
| pre_id_contrib | pre | pretrain/assemble.py | vt | 0.00/0.00/-/- | FT |  | 0.567 / - | 0.311 / - | - | 3.55 / - | -0.00009 (0.00019) | -0.00014 (0.00036) | 0.00026 / 0.00012 | - (-) | val/test clean; train version quarantined (8b leak); val/test construction differs (LEAKAGE-2, PSI 3.5) |
| pre_idmax32 | pre | pretrain/assemble.py | vt | 0.55/0.60/-/- | FT |  | 0.543 / - | 0.294 / - | - | 0.01 / - | 0.00013 (0.00016) | -0.00002 (0.00033) | -0.00032 / -0.00009 | - (-) | val/test clean; train version quarantined (8b leak) |
| pre_new | pre | pretrain/assemble.py | vt | 0.00/0.00/-/- | FT |  | 0.661 / - | 0.359 / - | - | 0.00 / - | 0.00002 (0.00012) | -0.00016 (0.00031) | 0.00038 / -0.00015 | - (-) | val/test clean; train version quarantined (8b leak) |
| pre_new_z | pre | pretrain/assemble.py | vt | 0.00/0.00/-/- | FT |  | 0.661 / - | 0.359 / - | - | 0.00 / - | 0.00007 (0.00012) | -0.00022 (0.00029) | 0.00039 / -0.00006 | - (-) | val/test clean; train version quarantined (8b leak) |
| prew_fe | prew | pretrain/make_prew.py | tvt | 0.38/0.45/0.44/0.38 | XTprew+gate |  | 0.608 / 0.603 | 0.345 / 0.246 | 0.009 | 0.02 / 0.02 | -0.00018 (0.00014) | -0.00022 (0.00034) | - / - | 0.00002 (0.00030) | no label leak; train density skew +0.015 AUC (LEAKAGE-6) |
| prew_fe_noid | prew | pretrain/make_prew.py | tvt | 0.38/0.45/0.44/0.38 | gate |  | 0.606 / 0.599 | 0.343 / 0.245 | 0.006 | 0.02 / 0.01 | -0.00033 (0.00014) | -0.00029 (0.00032) | - / - | -0.00005 (0.00022) | no label leak; train density skew +0.015 AUC (LEAKAGE-6) |
| prew_fe_z | prew | pretrain/make_prew.py | tvt | 0.38/0.45/0.44/0.38 | XTprew+gate |  | 0.608 / 0.603 | 0.345 / 0.246 | 0.009 | 0.02 / 0.01 | -0.00018 (0.00014) | 0.00027 (0.00033) | - / - | 0.00002 (0.00030) | no label leak; train density skew +0.015 AUC (LEAKAGE-6) |
| prew_hmax32 | prew | pretrain/make_prew.py | tvt | 0.38/0.45/0.44/0.38 | XTprew+gate |  | 0.603 / 0.592 | 0.336 / 0.236 | 0.002 | 0.02 / 0.01 | 0.00005 (0.00024) | -0.00039 (0.00037) | - / - | -0.00012 (0.00017) | no label leak; train density skew +0.015 AUC (LEAKAGE-6) |
| prew_hmax8 | prew | pretrain/make_prew.py | tvt | 0.38/0.45/0.44/0.38 | gate |  | 0.599 / 0.592 | 0.337 / 0.241 | 0.006 | 0.02 / 0.02 | -0.00040 (0.00014) | -0.00055 (0.00035) | - / - | -0.00013 (0.00018) | no label leak; train density skew +0.015 AUC (LEAKAGE-6) |
| prew_id_contrib | prew | pretrain/make_prew.py | tvt | 0.38/0.45/0.44/0.38 | gate |  | 0.554 / 0.548 | 0.304 / 0.220 | 0.009 | 0.49 / 0.91 | -0.00011 (0.00025) | 0.00006 (0.00038) | - / - | -0.00006 (0.00025) | no label leak; train density skew +0.015 AUC (LEAKAGE-6); id_contrib val/test zero-share differs (LEAKAGE-2) |
| prew_idmax32 | prew | pretrain/make_prew.py | tvt | 0.55/0.60/0.61/0.55 | gate |  | 0.543 / 0.538 | 0.294 / 0.214 | 0.009 | 0.01 / 0.02 | 0.00013 (0.00016) | -0.00002 (0.00033) | - / - | -0.00006 (0.00010) | no label leak; train density skew +0.015 AUC (LEAKAGE-6) |
| prew_new | prew | pretrain/make_prew.py | tvt | 0.38/0.45/0.44/0.38 | XTprew+gate |  | 0.608 / 0.603 | 0.346 / 0.248 | 0.009 | 0.02 / 0.02 | 0.00011 (0.00016) | -0.00007 (0.00035) | - / - | -0.00004 (0.00028) | no label leak; train density skew +0.015 AUC (LEAKAGE-6) |
| prew_new_z | prew | pretrain/make_prew.py | tvt | 0.38/0.45/0.44/0.38 | XTprew+gate |  | 0.608 / 0.603 | 0.346 / 0.248 | 0.009 | 0.02 / 0.01 | 0.00011 (0.00016) | -0.00052 (0.00034) | - / - | -0.00004 (0.00028) | no label leak; train density skew +0.015 AUC (LEAKAGE-6) |
| seq_argmax_hpos | seq | feats_v2/seq_/feat_hist.py | tvt | 0.01/0.01/0.03/0.01 | FT |  | 0.556 / 0.556 | 0.296 / 0.226 | 0.010 | 0.00 / 0.02 | 0.00014 (0.00021) | -0.00040 (0.00031) | -0.00003 / -0.00010 | -0.00005 (0.00014) | detector clean (skew +0.010, raw -0.001) |
| seq_cos_last1 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.601 / 0.605 | 0.338 / 0.266 | 0.013 | 0.00 / 0.01 | -0.00029 (0.00011) | 0.00005 (0.00036) | 0.00023 / 0.00003 | -0.00012 (0.00019) | detector clean (skew +0.013, raw +0.002) |
| seq_cv_invd20 | seq_cv | feats_v2/seq_/feat_covis.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+stk* |  | 0.552 / 0.541 | 0.281 / 0.217 | -0.000 | 0.00 / 0.01 | -0.00008 (0.00017) | -0.00112 (0.00038) | - / - | 0.00005 (0.00010) | detector clean (skew -0.000, raw -0.012) |
| seq_cv_invd20_rk | seq_cv | feats_v2/seq_/feat_rel.py | tvt | 0.00/0.00/0.00/0.00 | FT+stk* |  | 0.552 / 0.541 | 0.281 / 0.217 | -0.000 | 0.03 / 0.17 | -0.00018 (0.00014) | -0.00014 (0.00033) | -0.00001 / -0.00040 | -0.00003 (0.00008) | detector clean (skew -0.000, raw -0.012) |
| seq_cv_norm | seq_cv | feats_v2/seq_/feat_covis.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+stk* |  | 0.551 / 0.541 | 0.281 / 0.217 | -0.000 | 0.00 / 0.01 | -0.00014 (0.00018) | -0.00076 (0.00037) | - / - | 0.00002 (0.00007) | detector clean (skew -0.000, raw -0.012) |
| seq_cv_pop | seq_cv | feats_v2/seq_/feat_covis.py | tvt | 0.00/0.00/0.00/0.00 | XT+FT+stk* |  | 0.551 / 0.547 | 0.301 / 0.225 | 0.013 | 0.01 / 0.04 | -0.00013 (0.00019) | -0.00043 (0.00033) | - / - | -0.00008 (0.00008) | detector clean (skew +0.013, raw +0.001) |
| seq_dcos_deep | seq | feats_v2/seq_/feat_hist.py | tvt | 0.24/0.24/0.26/0.24 | none |  | 0.548 / 0.547 | 0.284 / 0.220 | 0.009 | 0.00 / 0.00 | -0.00006 (0.00010) | -0.00041 (0.00035) | -0.00001 / -0.00063 | 0.00000 (0.00006) | detector clean (skew +0.009, raw -0.003) |
| seq_dcos_hl2 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.627 / 0.628 | 0.348 / 0.271 | 0.009 | 0.00 / 0.01 | -0.00013 (0.00011) | -0.00023 (0.00034) | 0.00040 / 0.00012 | -0.00009 (0.00021) | detector clean (skew +0.009, raw -0.003) |
| seq_dcos_hl2_rk | seq | feats_v2/seq_/feat_rel.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.628 / 0.629 | 0.349 / 0.273 | 0.009 | 0.00 / 0.00 | -0.00013 (0.00018) | -0.00059 (0.00033) | 0.00034 / -0.00051 | -0.00006 (0.00020) | detector clean (skew +0.009, raw -0.003) |
| seq_dcos_nt_hl10 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.03/0.02/0.04/0.02 | none |  | 0.586 / 0.580 | 0.324 / 0.247 | 0.003 | 0.00 / 0.01 | -0.00044 (0.00018) | -0.00043 (0.00031) | -0.00006 / -0.00039 | -0.00001 (0.00012) | detector clean (skew +0.003, raw -0.009) |
| seq_dcos_rep | seq | feats_v2/seq_/feat_hist.py | tvt | 0.03/0.03/0.05/0.03 | none |  | 0.581 / 0.576 | 0.319 / 0.243 | 0.005 | 0.00 / 0.01 | -0.00021 (0.00016) | -0.00048 (0.00035) | -0.00005 / -0.00074 | 0.00002 (0.00010) | detector clean (skew +0.005, raw -0.007) |
| seq_dcos_s1 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.04/0.03/0.06/0.04 | none |  | 0.620 / 0.620 | 0.340 / 0.264 | 0.008 | 0.00 / 0.01 | -0.00022 (0.00017) | -0.00053 (0.00035) | 0.00029 / -0.00037 | -0.00003 (0.00020) | detector clean (skew +0.008, raw -0.003) |
| seq_dcos_s2 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.08/0.07/0.08/0.06 | none |  | 0.581 / 0.582 | 0.320 / 0.247 | 0.008 | 0.00 / 0.00 | -0.00012 (0.00016) | -0.00075 (0.00037) | 0.00014 / -0.00072 | -0.00025 (0.00015) | detector clean (skew +0.008, raw -0.004) |
| seq_dcos_t1_hl5 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.03/0.02/0.04/0.03 | none |  | 0.627 / 0.631 | 0.344 / 0.268 | 0.011 | 0.00 / 0.00 | -0.00018 (0.00018) | -0.00038 (0.00033) | 0.00013 / -0.00001 | -0.00016 (0.00022) | detector clean (skew +0.011, raw -0.001) |
| seq_dcos_t1_hl5_z | seq | feats_v2/seq_/feat_rel.py | tvt | 0.03/0.02/0.04/0.03 | none |  | 0.627 / 0.631 | 0.344 / 0.268 | 0.011 | 0.00 / 0.00 | -0.00018 (0.00018) | -0.00030 (0.00033) | 0.00013 / -0.00000 | -0.00016 (0.00022) | detector clean (skew +0.011, raw -0.001) |
| seq_dmax_hl10 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.629 / 0.632 | 0.347 / 0.271 | 0.009 | 0.00 / 0.01 | -0.00032 (0.00012) | -0.00059 (0.00034) | -0.00008 / -0.00072 | -0.00014 (0.00018) | detector clean (skew +0.009, raw -0.003) |
| seq_max2_gap | seq | feats_v2/seq_/feat_hist.py | tvt | 0.02/0.01/0.03/0.01 | none |  | 0.503 / 0.503 | 0.255 / 0.198 | 0.011 | 0.00 / 0.01 | 0.00013 (0.00016) | -0.00032 (0.00038) | -0.00004 / -0.00018 | -0.00003 (0.00012) | detector clean (skew +0.011, raw -0.001) |
| seq_max_all | seq | feats_v2/seq_/feat_hist.py | tvt | 0.01/0.01/0.03/0.01 | FT |  | 0.645 / 0.634 | 0.342 / 0.260 | -0.002 | 0.00 / 0.01 | -0.00029 (0.00024) | -0.00026 (0.00034) | -0.00026 / -0.00033 | -0.00011 (0.00014) | detector clean (skew -0.002, raw -0.013) |
| seq_max_all_rk | seq | feats_v2/seq_/feat_rel.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.645 / 0.634 | 0.344 / 0.261 | -0.002 | 0.00 / 0.00 | -0.00001 (0.00030) | 0.00011 (0.00037) | -0.00008 / 0.00009 | -0.00014 (0.00014) | detector clean (skew -0.002, raw -0.014) |
| seq_max_all_z | seq | feats_v2/seq_/feat_rel.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.645 / 0.634 | 0.342 / 0.260 | -0.002 | 0.00 / 0.01 | -0.00029 (0.00024) | -0.00014 (0.00032) | -0.00026 / -0.00014 | -0.00011 (0.00014) | detector clean (skew -0.002, raw -0.013) |
| seq_max_new8 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.01/0.01/0.03/0.01 | none |  | 0.635 / 0.634 | 0.350 / 0.270 | 0.004 | 0.00 / 0.01 | -0.00006 (0.00015) | -0.00034 (0.00035) | -0.00004 / -0.00068 | -0.00008 (0.00018) | detector clean (skew +0.004, raw -0.007) |
| seq_max_nt | seq | feats_v2/seq_/feat_hist.py | tvt | 0.03/0.02/0.04/0.02 | none |  | 0.600 / 0.588 | 0.325 / 0.245 | -0.004 | 0.00 / 0.01 | -0.00015 (0.00024) | -0.00021 (0.00038) | -0.00057 / -0.00048 | -0.00002 (0.00011) | detector clean (skew -0.004, raw -0.015) |
| seq_max_old | seq | feats_v2/seq_/feat_hist.py | tvt | 0.08/0.07/0.10/0.08 | none |  | 0.602 / 0.590 | 0.319 / 0.241 | -0.002 | 0.00 / 0.01 | -0.00018 (0.00013) | 0.00003 (0.00034) | -0.00059 / -0.00076 | -0.00000 (0.00009) | detector clean (skew -0.002, raw -0.014) |
| seq_max_s1 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.04/0.03/0.06/0.04 | none |  | 0.631 / 0.624 | 0.337 / 0.256 | 0.003 | 0.00 / 0.01 | 0.00008 (0.00015) | -0.00048 (0.00034) | -0.00007 / -0.00053 | -0.00014 (0.00016) | detector clean (skew +0.003, raw -0.009) |
| seq_max_t1 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.03/0.02/0.04/0.03 | none |  | 0.639 / 0.631 | 0.339 / 0.258 | 0.001 | 0.00 / 0.00 | -0.00005 (0.00018) | -0.00054 (0.00034) | -0.00030 / -0.00043 | -0.00012 (0.00013) | detector clean (skew +0.001, raw -0.010) |
| seq_top3 | seq | feats_v2/seq_/feat_hist.py | tvt | 0.01/0.01/0.03/0.01 | FT |  | 0.645 / 0.636 | 0.341 / 0.259 | -0.001 | 0.00 / 0.01 | -0.00035 (0.00022) | -0.00030 (0.00034) | -0.00021 / -0.00062 | -0.00013 (0.00016) | detector clean (skew -0.001, raw -0.012) |
| trd_dsr_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.527 / 0.514 | 0.295 / 0.224 | -0.000 | 0.02 / 0.21 | 0.00006 (0.00020) | -0.00012 (0.00035) | - / - | -0.00009 (0.00010) | detector clean (skew -0.000, raw -0.012) |
| trd_dsr_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.537 / 0.521 | 0.297 / 0.224 | -0.006 | 0.03 / 0.38 | -0.00013 (0.00020) | -0.00026 (0.00034) | -0.00012 / -0.00046 | -0.00014 (0.00009) | detector clean (skew -0.006, raw -0.017) |
| trd_ex_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.501 / 0.500 | 0.201 / 0.166 | 0.011 | 0.00 / 0.00 | -0.00003 (0.00008) | -0.00004 (0.00021) | - / - | 0.00001 (0.00003) | detector clean (skew +0.011, raw -0.000) |
| trd_ex_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.502 / 0.502 | 0.207 / 0.171 | 0.011 | 0.00 / 0.00 | 0.00002 (0.00014) | -0.00003 (0.00027) | - / - | -0.00002 (0.00004) | detector clean (skew +0.011, raw -0.001) |
| trd_kde_e1_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.529 / 0.537 | 0.297 / 0.232 | 0.019 | 0.00 / 0.00 | 0.00006 (0.00015) | -0.00006 (0.00033) | - / - | -0.00005 (0.00011) | detector clean (skew +0.019, raw +0.007) |
| trd_kde_e1_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.534 / 0.541 | 0.300 / 0.233 | 0.018 | 0.00 / 0.00 | -0.00015 (0.00013) | -0.00038 (0.00035) | - / - | -0.00007 (0.00012) | detector clean (skew +0.018, raw +0.006) |
| trd_kde_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.528 / 0.535 | 0.297 / 0.232 | 0.019 | 0.00 / 0.00 | 0.00003 (0.00014) | -0.00024 (0.00033) | - / - | -0.00006 (0.00011) | detector clean (skew +0.019, raw +0.007) |
| trd_kde_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.532 / 0.539 | 0.300 / 0.233 | 0.017 | 0.00 / 0.00 | -0.00015 (0.00014) | 0.00010 (0.00033) | -0.00001 / -0.00027 | -0.00014 (0.00012) | detector clean (skew +0.017, raw +0.006) |
| trd_max_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.527 / 0.528 | 0.295 / 0.228 | 0.012 | 0.00 / 0.00 | -0.00012 (0.00017) | -0.00006 (0.00035) | - / - | -0.00010 (0.00012) | detector clean (skew +0.012, raw +0.001) |
| trd_max_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.531 / 0.529 | 0.296 / 0.227 | 0.007 | 0.00 / 0.00 | 0.00004 (0.00016) | 0.00045 (0.00036) | 0.00012 / 0.00021 | -0.00014 (0.00012) | detector clean (skew +0.007, raw -0.004) |
| trd_n85_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.523 / 0.524 | 0.271 / 0.214 | 0.012 | 0.00 / 0.00 | -0.00038 (0.00016) | -0.00010 (0.00030) | - / - | -0.00001 (0.00009) | detector clean (skew +0.012, raw +0.000) |
| trd_n85_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.534 / 0.534 | 0.291 / 0.226 | 0.009 | 0.01 / 0.03 | -0.00019 (0.00014) | -0.00061 (0.00032) | - / - | -0.00005 (0.00011) | detector clean (skew +0.009, raw -0.003) |
| trd_n95_e1_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.503 / 0.503 | 0.218 / 0.178 | 0.011 | 0.00 / 0.00 | -0.00031 (0.00013) | -0.00049 (0.00026) | - / - | 0.00006 (0.00009) | detector clean (skew +0.011, raw -0.001) |
| trd_n95_e1_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.509 / 0.508 | 0.237 / 0.190 | 0.010 | 0.01 / 0.00 | 0.00004 (0.00015) | 0.00000 (0.00025) | - / - | -0.00002 (0.00008) | detector clean (skew +0.010, raw -0.002) |
| trd_n95_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.504 / 0.503 | 0.221 / 0.180 | 0.011 | 0.00 / 0.00 | -0.00040 (0.00015) | -0.00075 (0.00029) | - / - | 0.00001 (0.00010) | detector clean (skew +0.011, raw -0.000) |
| trd_n95_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.510 / 0.509 | 0.242 / 0.193 | 0.010 | 0.01 / 0.00 | -0.00011 (0.00017) | -0.00040 (0.00027) | - / - | -0.00007 (0.00010) | detector clean (skew +0.010, raw -0.002) |
| trd_skde_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.519 / 0.532 | 0.293 / 0.231 | 0.024 | 0.01 / 0.02 | -0.00016 (0.00016) | -0.00008 (0.00037) | - / - | -0.00001 (0.00010) | detector: skew +0.024, raw +0.013 (see findings) |
| trd_skde_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.521 / 0.535 | 0.294 / 0.231 | 0.024 | 0.01 / 0.02 | -0.00018 (0.00015) | 0.00012 (0.00037) | - / - | -0.00010 (0.00009) | detector: skew +0.024, raw +0.013 (see findings) |
| trd_sn95_s6h | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.501 / 0.505 | 0.233 / 0.188 | 0.014 | 0.01 / 0.00 | -0.00010 (0.00014) | -0.00027 (0.00028) | - / - | -0.00002 (0.00010) | detector clean (skew +0.014, raw +0.002) |
| trd_sn95_sday | trd | feats_v2/trd/build_emb.py | tvt | 0.00/0.00/0.00/0.00 | none |  | 0.504 / 0.510 | 0.255 / 0.202 | 0.016 | 0.04 / 0.08 | -0.00009 (0.00016) | 0.00013 (0.00030) | - / - | -0.00013 (0.00010) | detector clean (skew +0.016, raw +0.005) |
| tte_c_ctr | tte | src/features_tte.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.533 / 0.534 | 0.249 / 0.190 | 0.011 | 0.00 / 1.81 | -0.00002 (0.00019) | -0.00039 (0.00030) | -0.00013 / -0.00024 | -0.00001 (0.00010) | train: strictly earlier days (expanding window); val/test: all train |
| tte_it_coec | tte | src/features_tte.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.507 / 0.514 | 0.285 / 0.222 | 0.018 | 0.05 / 0.01 | 0.00035 (0.00023) | 0.00027 (0.00035) | 0.00022 / -0.00009 | 0.00007 (0.00025) | train: strictly earlier days (expanding window); val/test: all train |
| tte_it_ctr | tte | src/features_tte.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.510 / 0.518 | 0.284 / 0.222 | 0.018 | 0.04 / 1.14 | 0.00015 (0.00021) | 0.00028 (0.00034) | 0.00037 / 0.00035 | 0.00015 (0.00025) | train: strictly earlier days (expanding window); val/test: all train |
| tte_it_imp | tte | src/features_tte.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.534 / 0.528 | 0.298 / 0.225 | 0.008 | 0.06 / 0.01 | 0.00019 (0.00022) | -0.00026 (0.00037) | 0.00014 / 0.00016 | 0.00018 (0.00017) | train: strictly earlier days (expanding window); val/test: all train |
| tte_it_pos | tte | src/features_tte.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.526 / 0.526 | 0.263 / 0.207 | 0.013 | 0.01 / 0.01 | 0.00023 (0.00014) | -0.00020 (0.00030) | 0.00027 / -0.00019 | -0.00002 (0.00015) | train: strictly earlier days (expanding window); val/test: all train |
| tte_u_coec | tte | src/features_tte.py | tvt | 0.05/0.05/0.05/0.04 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.09 | 0.00000 (0.00000) | -0.00058 (0.00028) | 0.00000 / -0.00029 | 0.00000 (0.00000) | train: strictly earlier days (expanding window); val/test: all train |
| tte_u_ctr | tte | src/features_tte.py | tvt | 0.05/0.05/0.05/0.04 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.11 | 0.00000 (0.00000) | -0.00016 (0.00025) | 0.00000 / -0.00029 | 0.00000 (0.00000) | train: strictly earlier days (expanding window); val/test: all train |
| tte_u_imp | tte | src/features_tte.py | tvt | 0.05/0.05/0.05/0.04 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.11 | 0.00000 (0.00000) | -0.00020 (0.00025) | 0.00000 / -0.00004 | 0.00000 (0.00000) | train: strictly earlier days (expanding window); val/test: all train |
| tte_uc_ctr | tte | src/features_tte.py | tvt | 0.05/0.05/0.05/0.04 | GB+gate |  | 0.507 / 0.505 | 0.213 / 0.168 | 0.010 | 0.00 / 6.00 | 0.00025 (0.00013) | -0.00075 (0.00032) | 0.00023 / -0.00035 | -0.00008 (0.00007) | train: strictly earlier days (expanding window); val/test: all train |
| tte_uc_imp | tte | src/features_tte.py | tvt | 0.05/0.05/0.05/0.04 | GB+gate |  | 0.522 / 0.512 | 0.213 / 0.168 | 0.002 | 0.00 / 0.06 | 0.00007 (0.00010) | -0.00004 (0.00026) | 0.00014 / -0.00015 | -0.00003 (0.00004) | train: strictly earlier days (expanding window); val/test: all train |
| tte_ui_imp | tte | src/features_tte.py | tvt | 0.05/0.05/0.05/0.04 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.011 | 0.00 / 0.00 | 0.00007 (0.00008) | -0.00002 (0.00013) | -0.00002 / -0.00009 | 0.00001 (0.00006) | train: strictly earlier days (expanding window); val/test: all train |
| tte_ui_pos | tte | src/features_tte.py | tvt | 0.05/0.05/0.05/0.04 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.011 | 0.00 / 0.00 | 0.00002 (0.00001) | -0.00002 (0.00008) | 0.00001 / -0.00016 | -0.00000 (0.00000) | train: strictly earlier days (expanding window); val/test: all train |
| u_het10_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.00 | 0.00000 (0.00000) | -0.00015 (0.00013) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het1_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.14 | 0.00000 (0.00000) | -0.00014 (0.00019) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het2_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.04 | 0.00000 (0.00000) | -0.00012 (0.00023) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het3_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.04 | 0.00000 (0.00000) | -0.00018 (0.00019) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het4_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.02 | 0.00000 (0.00000) | -0.00023 (0.00020) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het5_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.01 | 0.00000 (0.00000) | -0.00028 (0.00021) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het6_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.01 | 0.00000 (0.00000) | -0.00009 (0.00021) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het7_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.03 | 0.00000 (0.00000) | -0.00000 (0.00019) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het8_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.01 | 0.00000 (0.00000) | -0.00017 (0.00021) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_het9_share | u_het | src/features.py | tvt | 0.00/0.00/0.00/0.00 | GB+gate |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 0.02 | 0.00000 (0.00000) | -0.00002 (0.00021) | - / - | 0.00000 (0.00000) | per-feed constant (within-feed AUC n/a) |
| u_tr_ctr | u_tr | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 2.25 | 0.00000 (0.00000) | -0.00007 (0.00024) | 0.00000 / -0.00011 | 0.00000 (0.00000) | val/test-safe; train rows include own label; user0 = a user (SHIFT-6) |
| u_tr_feeds | u_tr | src/features.py | tvt | 0.00/0.00/0.00/0.00 | FT |  | 0.500 / 0.500 | 0.199 / 0.163 | 0.012 | 0.00 / 10.20 | 0.00000 (0.00000) | -0.00019 (0.00019) | 0.00000 / -0.00023 | 0.00000 (0.00000) | val/test-safe; train rows include own label; user0 = a user (SHIFT-6) |

## Appendix B: redundancy clusters (|Spearman| > 0.95 on 2,500 val feeds / 238,256 rows; NaN -> column median rank, complete linkage)

- f4, f4_gap, f4_rk, f4_z
- pre2_fe, pre2_new, pre_fe, pre_new
- pre2_fe_z, pre2_new_z, pre_fe_z, pre_new_z
- als_exp10, als_exp5, als_last10
- ctx_hn_soft05, ctx_hs_d7, ctx_hs_t10
- it_hist_all, it_hist_all_luo, seq_cv_pop
- it_tr_coec, it_tr_coec_rk, tte_it_coec
- pre2_idmax32, pre_idmax32, prew_idmax32
- attn32_rk, attn32_z
- cat_cnt_all_rk, cat_cnt_all_z
- clu_pmi1024_mean, clu_pmi11024_mean
- cos_h10_rk, cos_h10_z
- cos_hall_rk, cos_hall_z
- cos_u, dot_u
- cos_u_rk, cos_u_z
- ctx_hs_k128, ctx_hs_k128t10
- f1_rk, f1_z
- f2_rk, f2_z
- f3_rk, f3_z
- f5, f5_gap
- f5_rk, f5_z
- it_cand_cnt_train, tte_it_imp
- it_hist_all_gap, it_hist_all_luo_gap
- it_hist_all_luo_rk, it_hist_all_rk
- it_hist_all_luo_z, it_hist_all_z
- it_hist_et1, it_hist_et1_luo
- it_hist_et10, it_hist_et10_luo
- it_hist_et2, it_hist_et2_luo
- it_hist_et3, it_hist_et3_luo
- it_hist_et4, it_hist_et4_luo
- it_hist_et5, it_hist_et5_luo
- it_hist_et6, it_hist_et6_luo
- it_hist_et7, it_hist_et7_luo
- it_hist_et8, it_hist_et8_luo
- it_hist_et9, it_hist_et9_luo
- it_tr_ctr, tte_it_ctr
- it_tr_pos, tte_it_pos
- maxsim32, maxsim32_gap
- maxsim32_rk, maxsim32_z
- pre2_fe_noid, pre_fe_noid
- pre2_hmax32, pre_hmax32
- pre2_hmax8, pre_hmax8
- prew_fe, prew_new
- prew_fe_z, prew_new_z
- seq_cv_invd20, seq_cv_norm
- seq_max_all_rk, seq_max_all_z
- trd_kde_e1_s6h, trd_kde_s6h
- trd_kde_e1_sday, trd_kde_sday
