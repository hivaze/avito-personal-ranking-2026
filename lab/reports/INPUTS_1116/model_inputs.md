# INVENTORY A: what the ranker reads today, and how a new trained input is added

Written 2026-11-16 02:09 UTC for workflow a workflow run (ledger block 2026-11-16T01:34:55Z, "TRAINED-INPUT BUNDLE", design only). CPU only; nothing existing was edited; no GPU process, nothing submitted. Scripts and outputs of this report: `reports/INPUTS_1116/model_inputs/` (section 7).

Markers: **[M]** measured by me (script and output named), **[R]** record on disk (path named), **[I]** judgement. ± is one paired per-feed standard error unless stated; it does not contain seed-draw noise.

Contents: 0 answer; 1 inputs by kind of information; 2 how a new train-stage block is added; 3 the positive control of the residual gate; 4 distillation and the test options; 5 corrections to statements on record; 6 limits; 7 files.

---

## 0. The answer

**0.1 What the wide student reads.** Seven raw fields (item content embedding, user embedding, category, f1..f5, the history event list with type and source, feed size M, history length H) and 34 pre-computed columns. Inside the model that becomes three token types, a 104-column dense block (63 computed in the model, 31 pre-computed columns kept raw, 10 within-feed rank / z versions), a retrieval block over the 16 most similar history items, and, in the fine-tuned branch only, 42 extra columns [M `m01_ckpt_inputs.py` -> `out/m01_ckpt_inputs.txt`, `out/dense_columns.tsv`; code `nn_x/w5/model.py`, `data.py`, `train.py`].

| Kind of information | Read by the train stage (all members) | Extra in the fine-tune branch (val / test only) |
|---|---|---|
| Item content | int8 128-d embedding (candidate and history tokens), norm, missing flag | - |
| Category | embedding (candidate, history); same-category events in the history: count, share, count per event type (12 columns); recency of the newest same-category event; feed category share; rank / z of the count (2) | - |
| Organisers' item / pair codes f1..f5 | raw, within-feed z, within-feed percentile (15) | - |
| User-item affinity from content | cos / dot with the user embedding; 6 pooled cosines to the history; max cosine per event type (10); 8 rank / z versions; attention bias; retrieval over the top-16 history items (type, source, recency bucket, key difference); `ctx_hn` 6 + `ctx_hs` 4 (competition of the feed's candidates for the last 32-128 history items) | `seq_max_all`, `seq_top3`, `seq_argmax_hpos` |
| History sequence | up to 256 newest events: content, event type (10), event source (codes 0-30), category, position index; 1 transformer layer. No timestamps exist | - |
| Feed composition | log M, log H, set attention over the candidates (2 blocks), `ctx_nd` 10 (similarity graph among the candidates), traffic-domain token, user-0 flags (2) | `centrality`, `ctx_nd` 10 again |
| Item popularity (cross-user counts of history occurrences) | `it_hist_all_luo`, `it_hist_et1..4_luo` (5) + rank / z of the total (2) | `it_hist_all_luo` rank / z / gap, event types 1-10 (14) |
| Co-visitation (item-item, other users' sequences) | `seq_cv_invd20`, `seq_cv_pop`, `seq_cv_norm` (3) + rank / z of invd20 (2) | `seq_cv_invd20`, its rank, `seq_cv_pop` (3) |
| Self-supervised sequence encoder (learned item-ID table inside it) | `pre2_fe_z`, `pre2_new_z`, availability flag (3 raw) + rank / z of `pre2_fe_s`, `pre2_new_s`, `pre2_hmax32` (6); windowed encoders on train rows | `pre_*` 11 columns from full-corpus encoders, with the ID / no-ID split |
| Exposure (how often the item was shown) | only the 4 popularity-light members: `xp_imp_c50` | - |

**Not read by any member** [M code read of `nn_x/w5/data.py` l.57-104 and `model.py`; I for the list as a whole]: the item ID itself (`id_buckets` 0: identity enters only through the content row, the cross-user counts and the encoder scores); the user ID beyond the user-0 flag; the feed timestamp, hour, weekday (TIME arm closed, COVERAGE l.276); the history field `hist_hpos`; any label statistic of the item, user, category or user x item (click rate, clicks over expected, position prior: arrays `tte_*`, `it_tr_*`, `ixh_*` exist for all three splits and are unused); any count of the item as a CANDIDATE in other feeds (except `xp_imp_c50` in the pop1 view); trend / cluster / lagged co-visitation columns (`trd_*`, `clu_*`, `seq_cvlag_*`, `hzcv_*`, `src_expo_*`: on disk, unused); price, geography and text do not exist as fields (if present at all they are inside f1..f5 or the content embedding). 220 per-candidate arrays in `data/feats` and 29 in `data/feats_new` exist for train AND val AND test and are read by no current member [M `m05_gate_names_and_store.py` -> `out/m05_gate_names_and_store.txt`]; existing is not the same as admissible (`it_hist_all`, `it_cand_cnt_split`, `it_tr_*` are among them).

**0.2 How a new train-stage input is added without editing any file** (detail in section 2).

1. Write one float32 array per split, flat candidate order, to `data/feats/<split>/<name>.npy` (train 246,745,836 rows, val 1,389,971, test 2,904,806; holdout rows are rows of the train array) and hard-link it to `nn/feat_cache/<split>/<name>.npy`. Precedents that did exactly this with exclusive create: `nn_x/abl/pop1/build_xp_imp_c50.py`, `nn_x/abl/pop2/build_xp_ctr.py`, `nn_x/abl/time/build_tm.py`.
2. Train through `nn_x/abl/train.py`, which has `--xt_add n1,n2,...` (append columns after the 34), `--xt_rel_extra` (within-feed rank / z of preset columns) and `--xt_add_relonly` (added columns through rank / z only) AND the distillation block. Command = `nn_x/w12/kd/launch_pop1w.sh` with `--xt_exclude $POP --xt_add xp_imp_c50` replaced by `--xt_add <new columns>`. `nn_x/w5/train.py` has no such flag (a preset name or a bare comma list only; a comma list loses the rank / z channels).
3. Everything downstream reads the column list from the checkpoint cfg: fine-tune, holdout evaluation, stale replays, teacher dump. The full-data lane needs a write-new lane copy with one more arm (as `lane_full_v.sh` did); the pooled-z read tool `arm_read8.py` hard-codes `w5_e10_*` names, so the read needs a write-new copy too.
4. Name rules: `--xt_add` refuses names that start with `seq_cv` or `it_hist`; `data.attach_xt` clips every other name to ±10 (`*_z` to ±5, `*_rk` raw), maps NaN to 0 and stores fp16, so counts must be pre-scaled and a missing value needs its own `has:<name>` flag. Each column costs 0.46 GiB of GPU memory.
5. Costs on record: smoke run 3-4 min wall-clock; a held-out wide student 78-262 min (median 194 in `nn_x/w5`, 256 in `nn_x/abl`); fine-tune 5-27 min.
6. The gate's input audit is name-based (regular expressions on the logged list + a quarantine inode check). It would pass `tte_*`, `xp_ctr_c` and even `it_cand_cnt_split` silently [M `m05`]; it is not a construction review.

**0.3 The positive control** (section 3). It was the block of the 8 popularity / co-visitation inputs: 4 hard-label 10-epoch members trained without it (`abl_e10_xdrop_pop_s951-954`) against their with-input twins (`w5_e10_base_s951-954`), pooled; the removed columns were then offered as a residual on the ablated pool. I reproduced the record to the digit and added a bootstrap and three more controls [M `m02_gate_cal_boot.py` -> `out/m02_hold.txt`, `out/m02_val.txt`]:

| Control (hard labels, d128, 10 epochs) | Witness | Trained-input value (pool) | Linear nested residual | Monotone GBDT residual | Recovered share, linear [95%] | Recovered share, GBDT [95%] |
|---|---|---|---|---|---|---|
| 8 popularity inputs removed (the record's control) | holdout | +0.00852 ± 0.00066 | +0.00086 ± 0.00029 (best of 8) | +0.00222 ± 0.00033 | 0.10 [0.06, 0.17] | 0.26 [0.19, 0.33] |
| ONE added column `xp_imp_c50` (new) | holdout | +0.00201 ± 0.00046 | -0.00021 ± 0.00026 | +0.00007 ± 0.00020 | -0.10 [-0.47, +0.14] | 0.03 [-0.19, +0.22] |
| ONE added label statistic `xp_ctr_c` (new) | holdout | +0.00149 ± 0.00044 | +0.00026 ± 0.00024 | +0.00024 ± 0.00018 | 0.17 [-0.17, +0.57] | 0.16 [-0.09, +0.47] |
| encoder scalars removed, 2 seeds (new on the holdout) | holdout | +0.00059 ± 0.00049 | +0.00028 ± 0.00019 (best of 5) | -0.00010 ± 0.00013 | not identified | not identified |
| all four controls | val | +0.00160 ± 0.00102 / +0.00020 ± 0.00077 / +0.00033 ± 0.00075 / +0.00061 ± 0.00083 | -0.00028 .. +0.00064 | -0.00084 .. +0.00004 | not identified | not identified |

Sensitivity of the residual gate, stated separately: **holdout, linear: 10% for the 8-column block, -10% and +17% for a single added column; holdout, GBDT: 26% for the block, 3% and 16% for a single added column; val, both forms: NOT IDENTIFIED**, because no control has a trained-input value above 1.6 SE on val, so "nothing on val" is the absence of a measurement, not a measured zero. The two single added columns, each worth +0.0015 to +0.0020 as a trained input at 3.4 and 4.4 SE, are REJECTED by both residual lenses on the holdout (z -0.8 / +0.3 and +1.1 / +1.4). All four controls are hard-label members; three of the four are popularity-type, for which the in-period holdout overstates the worth on test rows.

**0.4 Distillation** (section 4). The records hold no clean measurement of how much of a new input's value survives when half of the target comes from a teacher that never saw it. What they do hold [M `m03_kd_view_contrasts.py` -> `out/m03_kd_view_contrasts.txt`]: the gain from distillation is as large for a student whose input view differs from the teacher's (pop1 view: member +0.0040 val / +0.0022 holdout) as for a same-view student (+0.0026..+0.0030 / +0.0019..+0.0020); and what an input block is worth is the same under distillation as under hard labels (full view minus pop1 view, 4-pools on the holdout: +0.0060..+0.0064 distilled, +0.0065 hard labels). Options:

| Option | Runs | Process-hours on record | What it can and cannot show |
|---|---|---|---|
| A. Hard-label members with the input, against the 6 stored `w5_e10_base` members | 4-6 x (10 epochs, d128, `nn_x/abl/train.py` + `--xt_add`) | 57-182 min per run -> 4-18 h | Clean of the teacher; same regime as all four positive controls. Paired 4-pool SE on record: holdout ±0.0005, val ±0.0008. Not the current recipe |
| B. Distilled students with the old teacher, against the 16 stored held-out wide students | 8 x (10 epochs, d192) | 78-262 min per run -> 26-34 h at the median rates (10 h at the fastest), + 8 fine-tunes 1-3.6 h | The current recipe. A pass is conservative; a null is not a closure (teacher half blind to the input). 8-pool SE on record ±0.0005..0.0006 val, ±0.0003 holdout |
| C. Teacher regeneration, then students | A's members double as teacher members; dump <= 4.6 min per 2 checkpoints; then B | A + B; the full `teacher_mean11` recipe is 30.9 h | The only form in which the input "counts fully"; needs a matched control teacher without the input (a differently built teacher moved the students by 0.0005 on record) |

A new input cannot be tested in the 1-hour re-cool harness: `--init` takes the inputs from the snapshot cfg and the input flags are refused with it.

**0.5 Corrections to the premise** (section 5): (i) two trained inputs WERE run as wide distilled students of the current recipe, `xp_imp_c50` (16 students) and the label statistic `xp_ctr_c` (4 students), but only on the popularity-free view and read as 0.2 admixture members; no added input was ever trained on the full 34-input view under distillation. (ii) On val even the 8-column load-bearing block is not resolved at 4 members per arm (+0.0016 ± 0.0010), so a val-led rule for a new input needs more than 4 + 4 members or a larger effect.

---

## 1. Inputs by kind of information

Family: `w5_e10_kd05w192` / `w5_e10_kd05t25w192` (held-out) and `w5_fd10_*` (all data). Command: `nn_x/w5/lane.sh` stage E = `E_BASE` + `HYG` + `--kd_file ... --kd_w 0.5 --d 192` (`HYG` l.61, `E_BASE` l.64-66, arms l.97 and l.99). Input-relevant flags: `--xt cvnd_pre2s --xt_drop 0 --user0 --domain --sim_bias --retr_m 16 --attn_gate --no_pos --aux_pos 0.5` (history cap `--max_hist` 256 by default). Held-out and full-data students have identical input lists; 2,118,848 parameters [M `m01`].

### 1.1 Raw fields and what happens to them

| Raw field (`data/flat/<split>/`) | Used as | Not used |
|---|---|---|
| `cand_emb_idx` -> int8 item-embedding table (128-d) | candidate token `Linear([e, e * u])`; every cosine below; missing flag; log norm | - |
| `user_emb` (int8 128-d per feed) | user token; `e * u` product; cos / dot | - |
| `cand_category_id`, `hist_category_id` (54 codes) | one shared embedding table; category-match counts | - |
| `cand_f1..f5` (quantile codes of unknown item / pair features; `docs/01_DATA_AND_FINDINGS.md` §5.1) | 15 dense columns; also in the linear head | - |
| `hist_emb_idx`, `hist_event_type` (1-10), `hist_event_source` (codes 0-30, embedding table of 32) | history tokens (newest 256), retrieval values, per-type pooled cosines | `hist_hpos` |
| `M`, `H` | log M, log(1+H); padding masks | - |
| `user_id` | only `user_id == 0` (2 columns) | the id itself |
| `cand_item_id`, `hist_item_id` | not loaded (`id_buckets` 0) | the id itself |
| `timestamp` | not a model input (the feature builders use it for causal windows) | hour, weekday, date |
| `cand_pos` (train) | training target of the auxiliary head only; the position tower is off (`--no_pos`) | - |
| `cand_target` | label | - |

### 1.2 The 63 in-model dense columns (`model.dense_features`, `nn_x/w5/model.py` l.52-132)

| Columns (index in `out/dense_columns.tsv`) | n | Kind |
|---|---|---|
| f1..f5: raw, within-feed z, within-feed mid-rank percentile (0-14) | 15 | organisers' codes; feed-relative |
| embedding missing flag, log1p(embedding norm) (15-16) | 2 | item content status |
| cos(candidate, user), dot / 1e4 (17-18) | 2 | user-item affinity |
| cosine to history: max over all, last 8, last 32; mean; soft-max pool (logsumexp, scale 5); mean of top 3 (19-24) | 6 | user-item affinity (history) |
| max cosine to history events of type t, t = 1..10 (25-34) | 10 | affinity by action type |
| log1p(# history events of the candidate's category), share of H (35-36) | 2 | category affinity |
| log1p(# same-category events of type t), t = 1..10 (37-46) | 10 | category affinity by action type |
| 1 / (1 + index of the newest same-category event) (47) | 1 | category recency |
| share of the feed in the candidate's category, log M, log1p(H) (48-50) | 3 | feed composition |
| within-feed z and percentile of cos_u, max cosine, soft-max pool, top-3, log category count (51-60) | 10 | feed-relative affinity |
| is_user0, is_user0 x log1p(H) (61-62) | 2 | user regime |

### 1.3 The 34 pre-computed columns selected by `--xt cvnd_pre2s`

Defined in `XT_PRESETS["cvnd_pre2s"]` = `XT_PRESETS["cvnd"]` + 6 (`nn_x/w5/train.py` l.72-78, l.91-94). Files: `nn/feat_cache/<split>/<name>.npy`, float32, hard links of `data/feats` or `data/feats_new` files, all present for train / val / test with the right row counts [M `out/xt_files.tsv`]. Transform by name in `SplitData.attach_xt` (`nn_x/w5/data.py` l.180-199).

**Co-visitation, 3 columns.** Builder `feats_v2/seq_/feat_covis.py` (docstring l.1-17). Source sequences = de-duplicated histories of OTHER users; a train row of day d reads train sources before day d; val / test rows read all train sources plus val / test histories with an earlier timestamp. Transform log1p.

| Column | Meaning |
|---|---|
| `seq_cv_invd20` | sum over the candidate's occurrences in the source and over neighbours within ±20 positions that are in this feed's history of w / d (w = 0.95^history position, d = distance); also gets in-model rank / z |
| `seq_cv_pop` | number of scanned source occurrences of the candidate (cap 3000; 0 = cold item): an item-popularity count |
| `seq_cv_norm` | `seq_cv_invd20 / sqrt(seq_cv_pop + 1)`; rho 0.9998 with invd20 [R abl ledger row E1] |

**Item popularity, 5 columns.** Builder `src/features.py` (`global_item_stats`, l.231-251: occurrences of the item in ALL history snapshots of train + val + test) and `src/features_luo.py` (subtracts occurrences from later feeds of the same non-zero user, in every split). Transform log1p.

| Column | Meaning |
|---|---|
| `it_hist_all_luo` | history occurrences of the item, all event types; also gets in-model rank / z |
| `it_hist_et1_luo` .. `it_hist_et4_luo` | the same for event types 1-4 (e1 item-card view, e4 favourites; e2 / e3 contact types at low confidence, `docs/01` §5.2) |

These 8 columns are the "popularity block" (`xt_idcols`; the columns the pop1 view removes and the positive control used). The two families are built differently on train rows: the co-visitation source of a train row is causal (earlier train days only), while the `it_hist_*_luo` count of a train row includes other users' LATER snapshots (all splits pooled, only the same user's later feeds removed), a parity gap on record [R `docs/05_METHODOLOGY.md` §5.2, row of 2026-11-15]. They are the stale ones: mean warm-item staleness 1.75 d on the holdout, 5.69 d on val, 9.34 d public, 11.23 d private [R `reports/CURRENT_STATUS.md` §1].

**Similarity graph among the feed's candidates, `ctx_nd`, 10 columns.** Builder `feats_v2/ctx_/build_ctx.py` group `nd` (kernel docstring l.240-246). Own feed, own history, no labels. Clip ±10.

| Column | Meaning |
|---|---|
| `ctx_nd_max` | max cosine to another candidate of the feed |
| `ctx_nd_top5u`, `ctx_nd_top5h` | mean cosine to the 5 candidates with the highest cos_u / the highest history max-similarity |
| `ctx_nd_knn_u`, `ctx_nd_knn_h`, `ctx_nd_knn_f1` | cos_u / history max-similarity / f1 averaged over the candidate's nearest candidates |
| `ctx_nd_knn_u_dev`, `ctx_nd_knn_h_dev` | own value minus that neighbour average |
| `ctx_nd_ucent`, `ctx_nd_hcent` | cos(candidate, user) and cos(candidate, mean of the last 10 history items), each minus cos(candidate, feed mean) |

**Competition of the candidates for the user's recent history items, `ctx_hn` 6 + `ctx_hs` 4.** Builders `feats_v2/ctx_/build_ctx3.py`, `build_ctx4.py`. Own feed and history only. Clip ±10 (binds on 0.2-1.2% of val rows for the four attention-mass columns [R `reports/DATA_PIPELINE_1113.md` §1.3a]).

| Column | Meaning (last K = 32 history items unless stated; recency weight 0.85^j) |
|---|---|
| `ctx_hn_top1w` | weighted number of history items whose most similar candidate is this one |
| `ctx_hn_rrw` | weighted mean reciprocal rank of the candidate among the feed's candidates, per history item |
| `ctx_hn_e1_rrw` | the same over event-type-1 history items only |
| `ctx_hn_bestrk` | rank of the candidate for its own best-matching history item (0 = nearest) |
| `ctx_hn_margin` | own max similarity minus the best other candidate's similarity to the same history item |
| `ctx_hn_soft05` | softmax attention mass (temperature 0.05) of the history items over the candidates, x M |
| `ctx_hs_t10`, `ctx_hs_k128`, `ctx_hs_e1` | the same mass at temperature 0.10; over 128 items with decay 0.95; over event-type-1 items |
| `ctx_hs_mutual` | weighted count of mutual nearest neighbours (candidate <-> history item) |

**Self-supervised trajectory encoder, 6 columns** (§1.4). Builders `pretrain/*`, `pretrain/assemble_train.py`, `feat_new/pre2_rescale.py`.

| Column | Meaning | How it enters |
|---|---|---|
| `pre2_fe_z`, `pre2_new_z` | within-feed z of the two head scores ("next new feed-originated open", "next new item") | raw, clip ±5 |
| `pre2_fe_s`, `pre2_new_s` | the raw head scores mapped by the fixed `(x + 20) / 6` | only through in-model rank and z (`XT_RELONLY`) |
| `pre2_hmax32` | max cosine to the last 32 history items in the encoder's output space | only through rank and z |
| `has:pre2_fe` | 1 where the score exists (0 on all Sep 6-8 train rows: 27.5% of train feeds, never on val / test [R `DATA_PIPELINE_1113.md` §1.3d]) | raw |

`XT_REL` adds rank + z for 5 columns (`seq_cv_invd20`, `it_hist_all_luo`, `pre2_fe_s`, `pre2_new_s`, `pre2_hmax32`); `XT_RELONLY` drops the raw copy of the last three: 63 + 34 + 10 - 3 = 104 [M `m01`: cfg `n_dense` 104, `c_mlp.0.weight` (384, 104)]. The dense block enters the candidate token through `MLP(104 -> 384 -> 192)` and the logit through a linear "wide" term; it is standardised with 40-batch train statistics stored in the checkpoint and clamped to ±6 (`common.feature_stats`, `model.py` l.527).

### 1.4 The encoder behind `pre2_*` [R `docs/02_SOLUTION.md` §4.3; `reports/pretrain_report.md` §2, §8c]

Causal transformer (d 384, 4 layers) over user trajectories with a hybrid item tower: content MLP + category + a learned item-ID table. This is the only place where an item-ID embedding exists in the pipeline. Train rows are scored by six windowed models (windows from Sep 9), val / test rows by full-corpus models; nothing after a val / test feed of the same user. Worth as a ranker input: +0.00128 ± 0.00061 val / +0.00054 holdout at 2 epochs; +0.00077 ± 0.00073 val at 10 epochs [R the same §4.3]; my 2-seed 10-epoch pool read: +0.00061 ± 0.00083 val, +0.00059 ± 0.00049 holdout [M `m02`].

### 1.5 Retrieval block (`--retr_m 16`, `HistRetrieval`, `model.py` l.151-222)

For each candidate, the 16 history items with the highest raw content cosine. Attention logit = -||k_c - k_j||² + a x cosine + a per-event-type prior, with a learned zero-value sink; value = LayerNorm(E_type + E_source + E_recency-bucket(log2 of the history index) + MLP(k_c - k_j)); k = a learned 128 -> 64 projection of the content embedding. Output added to the candidate token. It reads no field beyond those of §1.1: it is a second route for "content similarity to recent actions, by action type, source and recency". Worth +0.0034 val / +0.0029 holdout at 2 epochs [R `docs/02_SOLUTION.md` §11].

### 1.6 The 42 val / test-only inputs of the fine-tune (`--extra n2c_pre_v3`, `nn_x/w5/finetune.py` l.145-165)

Injected through a zero-initialised branch `Linear(42 -> 2d) GELU Linear(2d -> d)` added to the candidate token (`model.add_extra`); standardised with VAL statistics, log1p for count-like columns, clip ±6 (`data.attach_extra`). The list is printed by every FT log [M `m01`, from `nn_x/w5/logs/w5_fd10_kd05w192_s951_n2cpv3s3_ft.log`].

| Kind | Columns | n | Also a train-stage input? |
|---|---|---|---|
| Item popularity | `it_hist_all_luo`, `_rk`, `_zsl` (signed log of the within-feed z), `_gap` (gap to the feed max), `it_hist_et1..et10_luo` | 14 | 5 of 14 (total, et1-4); et5..et10 and the gap only here |
| Co-visitation | `seq_cv_invd20`, `seq_cv_invd20_rk`, `seq_cv_pop` | 3 | 2 raw; the stored rank only here (the train stage computes its own) |
| Content affinity to the history | `seq_max_all` (max cosine), `seq_top3`, `seq_argmax_hpos` (history position of the best match) | 3 | in-model equivalents for the first two; the position only here |
| Feed composition | `centrality` (cosine to the feed mean), `ctx_nd_*` (10) | 11 | the 10 `ctx_nd` yes; centrality only here |
| Encoder, full-corpus models | `pre_fe`, `pre_fe_z`, `pre_new`, `pre_new_z`, `pre_hmax32_c`, `pre_hmax8_c`, `pre_fe_noid` (ID part zeroed), `pre_id_contrib_v2` (ID contribution), `pre_idmax32` (max cosine in the ID space), `pre_idmax32_has`, `pre_hmax32_sent` | 11 | no: the `pre_*` family is banned for train rows (construction leak, `src/presub_gate.py` `BANNED_TRAIN`); the train stage has the windowed `pre2_*` scalars instead |

17 of the 42 are also train-stage inputs; 30 of the 42 have a train array in `data/feats` (all but the 11 `pre_*` and `it_hist_all_luo_zsl`) [M `m01`]. The fine-tune half is worth +0.00048 ± 0.00055 on public and +0.00009 on val out-of-fold [R `docs/02_SOLUTION.md` §8].

### 1.7 The popularity-light view (4 of the 20 members of FD16P20)

`--xt_exclude <the 8 popularity columns> --xt_add xp_imp_c50`: 27 pre-computed inputs, 93 dense columns [M `m01`, checkpoint `abl_fd10_pop1kd05w192_s951`]. `xp_imp_c50` = log1p(min(train impressions of the item, 50)); train rows of day d count train days before d, val / test rows all train days (`nn_x/abl/pop1/build_xp_imp_c50.py`). In the fine-tune 7 of the 8 excluded columns come back through the extras branch [R `docs/02_SOLUTION.md` §4.6].

### 1.8 Training-only signals (targets, not inputs)

Click label; teacher logits (`--kd_file`, weight 0.5, in the target only); production position (auxiliary head, weight 0.5 annealed to 0.1); p_random per train feed (domain-token mix, auxiliary-loss weight, cooldown pool, holdout draw). At inference the domain token is fixed (random-like in the train-only branch, exploration in the fine-tuned branch).

### 1.9 Duplicate check for a proposed feature: what is already covered [I, from §1.1-1.7]

| A proposed feature of this kind ... | ... duplicates |
|---|---|
| any pooled content cosine between the candidate and the history, overall or per event type | 16 in-model columns + retrieval + attention bias + `ctx_hn` / `ctx_hs` + `pre2_hmax32` |
| category match counts, shares, recency | 14 in-model columns + category embeddings |
| within-feed rank / z of f1..f5 or of the similarities | 20 in-model columns; 4 more rank / z channels read -0.0019 val at 10 epochs under hard labels (COVERAGE l.150) |
| near-duplicate / neighbourhood structure of the feed | `ctx_nd` 10 + set attention |
| how often the item occurs in histories, by event type 1-4 | 5 columns (types 5-10 only in the fine-tune) |
| item-item co-occurrence in other users' histories | `seq_cv_*` 3 |
| a sequence-model score of the candidate given the history | `pre2_*` 6 |
| **not covered by any train-stage input** | label statistics (item / user / category / user x item click rates, clicks over expected, position prior); exposure counts of the item as a candidate (full view); time of the feed; item or user identity; cross-feed statistics of the same user's EARLIER feeds; trend / age of the item; cluster-level statistics |

---

## 2. How a new train-stage input block is added without editing an existing file

### 2.1 Where the `--xt` names are defined

`XT_PRESETS`, `XT_REL`, `XT_RELONLY` at module level of each trainer copy: `nn_x/w5/train.py` l.65-100, `nn_x/abl/train.py` l.84-117, and the frozen full-data copies `nn_x/w12/stale/fulldata/train_full.py` / `train_full_abl.py`. `--xt` takes a preset key or a comma list of file names (`train.py` l.330). With a comma list `XT_REL.get(a.xt)` is empty, so the 5 rank / z channels and the rank-only treatment of the 3 pre2 columns disappear: "the 34 names + new ones" as a list is NOT the current model plus inputs [M code read, `nn_x/w5/train.py` l.343-345]. The no-edit route is therefore `nn_x/abl/train.py` (`abl_arms.py`):

| Flag | Effect | Refuses |
|---|---|---|
| `--xt_add n1,n2` | appends the columns LAST; indices of the 34 and of their rank / z channels unchanged; `n_dense` + 1 each; the checkpoint cfg carries the extended `xt_names` | names starting with `seq_cv` or `it_hist`; names already present; names of `XT_REL` / `XT_RELONLY`; `--init` |
| `--xt_rel_extra n1,..` | in-model within-feed rank + z (+2 each) for named columns of the final list; the raw column stays. The helper only requires the name to be in the list after `--xt_add` (`train.py` l.341-357, `abl_arms.xt_rel_extend`), so `--xt_add X --xt_rel_extra X` gives an added column raw + rank + z, and the fine-tune check recomputes the same; this combination was never run (used on record for preset columns only, arm xrel_more) [M code read] | names not in the list; names already in `XT_REL`; `--init` |
| `--xt_add_relonly n1,..` | an added column enters only through rank + z (net +1) | names not in `--xt_add`; `--init`. Not known to the fine-tune's input check (`nn_x/abl/popfree/finetune.py` `honour_xt_exclude` recomputes the cfg from `--xt`, `--xt_exclude`, `--xt_add`, `--xt_rel_extra` only; a checkpoint trained with `--xt_add_relonly` fails its assertion unless `--xt_check off` is passed, which no driver does) [M code read l.119-159] |
| `--xt_exclude n1,..` | drops preset columns with their rank / z channels | unknown names |

`nn_x/abl/train.py` has the distillation block (`--kd_file`, `--kd_w`, `--kd_T`, l.215-217, l.450-454, l.520-521; the target formula is that of `nn_x/w5/train.py`), `model.py` is byte-identical to `nn_x/w5/model.py` (md5 `8942ee18...`), `data.py` / `common.py` differ only in paths [M `md5sum`, `diff`]. With its extra flags off the harness is the `nn_x/w5` code path (test `nn_x/abl/tests/test_abl.py`; STATUS §2). Variant names must start with `abl_`; outputs go to `nn_x/abl/{ckpt,logs,preds}` as new files.

### 2.2 On-disk format and location

- One file per split and name: `nn/feat_cache/{train,val,test}/<name>.npy`, 1-D float32, aligned with `data/flat/<split>/cand_*.npy` (train 246,745,836 rows = 987 MB; val 1,389,971; test 2,904,806) [M `m01`, `out/xt_files.tsv`]. `nn_x/w5/feat_cache` and `nn_x/abl/feat_cache` are symlinks to `nn/feat_cache`; the harnesses only read there and fail loudly on a missing file (`data.py` l.31-41; trainer assertion l.332-336).
- There is no separate holdout array: the 30,000 holdout feeds are train feeds (`p_random > 0.9`, `M >= 50`, fixed seed 12345), so their rows are rows of the train array and are built like every other train row.
- Every current input is a hard link (link count 2) of `data/feats/<split>/` or `data/feats_new/<split>/` [M `out/xt_files.tsv`]. A new input therefore means: NEW files in `data/feats` (or `data/feats_new`) + NEW hard links in `nn/feat_cache`, created with exclusive create. No existing file changes; the directory `nn/feat_cache` gains entries. The three precedents are named in §0.2.
- `nn/feat_cache` holds 56 train / 102 val / 102 test files; 21 names with all three splits are read by no current member (`tm_*`, `xp_ctr_c`, `pre2n_*`, `prew_*`, `it_hist_all_luo_rk` / `_z`, `seq_cv_invd20_rk`, `pre2_new`, `pre2_id_contrib`) [M `m01`, `m05`].

### 2.3 Transform and normalisation

`attach_xt` (fixed, split-independent, chosen by the NAME): `has:<name>` -> non-NaN indicator of `<name>`; `*_rk` unchanged; `*_z` clipped to ±5; `seq_cv*` / `it_hist*` -> log1p(max(x, 0)); every other name clipped to ±10; then NaN -> 0 and fp16 on the GPU. Consequences for a new column [M code read; I for the advice]:

- a raw count or a raw score outside ±10 is silently saturated (this was bug FR-1 for raw `pre2_fe`, `feat_new/pre2_rescale.py` docstring): pre-scale it, as `xp_imp_c50` (log1p of a capped count) and `xp_ctr_c` (clipped log ratio) do;
- "missing" becomes 0: if 0 is also a legal value, pass `has:<name>` as a further `--xt_add` entry (the has-prefix is resolved by `attach_xt` and by the miss check; done before by arm pre2noid with `has:pre2n_fe`, `nn_x/abl/logs/abl_pre2noid_s23.log` l.7 [R]);
- existing lagged columns whose names start with `seq_cv` / `it_hist` (`seq_cvlag_*`, `it_hist_hz_*`, `it_hist_et5_luo`) cannot be passed to `--xt_add`; they need a copy under another name with the log1p applied in the file [M `m05`].

Model side: mean / std of all dense columns over 40 random training batches (seed 0, model in eval mode) are stored in the checkpoint as `f_mean` / `f_std`; inputs are standardised and clamped to ±6 (`train.py` l.374-377, `model.py` l.520-527). The same stored statistics are used on val, test and in the fine-tune; nothing is re-fitted per split ("test-time standardisation" read -0.0024..-0.0057, COVERAGE row of 11-14). An added column enters `c_mlp` (first layer `n_dense -> 2d`) and the linear `wide` head.

Memory: each column is 246,745,836 x 2 bytes = 0.46 GiB on the GPU for the train split; the 34 columns take 15.6 GiB next to 21.2 GB of base tensors (`nn_x/w12/kd/out/dump_a.log`) [M arithmetic; R log].

### 2.4 Who reads the columns afterwards

| Stage | Script | How it finds the inputs | Works for an `--xt_add` checkpoint as is? |
|---|---|---|---|
| Held-out student | `nn_x/abl/train.py` (launch form `nn_x/w12/kd/launch_pop1w.sh`) | flags | yes; a write-new launch script with the changed flags |
| All-data student | `nn_x/w12/stale/fulldata/train_full_abl.py` = `nn_x/abl/train.py` + `--holdout_in_train`, started by `lane_full*.sh` stage P | flags inside the lane (`--xt_exclude $POP` + `--xt_add ...` hard-wired per arm) | needs a write-new lane copy with a new arm; `lane_full_v.sh` is the model (against `lane_full.sh`: one more `kd_flags` case, the arm and seed guards of stage P, and the `--xt_add` choice per arm) [M `diff`] |
| Fine-tune | `nn_x/abl/popfree/finetune.py` via `nn_x/abl/ftbag/ft_one_s3_r6.sh` `MODE=abl`, or `nn_x/w12/kd/ft_lane.sh <gpu> abl <variant>` | `ck["cfg"]["xt_names"]` -> `attach_xt` on val and test; input check against the saved args | yes for `--xt_add` / `--xt_rel_extra`; not for `--xt_add_relonly` (§2.1). The extras preset is unchanged; a new column can also be offered as an extra by passing `--extra` a comma list |
| Train-only predictions | written by the trainer (`nn_<v>_trainonly_{val,test}.npy`, `@last`, `holdout/nn_<v>_holdout.npy`) | cfg | yes |
| Holdout / stale replays | `nn_x/abl/holdout_eval.py`, `nn_x/abl/stale_pred.py` | cfg | yes; the stale scenarios replace the popularity inputs only: a new recency-dependent column "has no counterfactual and stays at its val value", which biases the stale witness toward the arm [R `nn_x/abl/ledger/ledger.md` §7.1] |
| Teacher dump | `nn_x/w12/kd/dump_teacher.py` | cfg `xt_names` -> `attach_xt` on train | yes (it re-attaches when the list changes) |
| Pool read | `nn_x/w12/kd/arm_read8.py` | hard-coded `w5_e10_<arm>_s<seed>` | no: write-new copy; `pc_common.paths` already resolves `abl_*` members |
| Gate | `src/presub_gate.py` check C | the "train-stage extra features" line after the variant's args line in `nn/logs/*.log` (`fd_finish.sh` copies the logs there) | yes; prints "<34 + k> train-stage inputs audited" |

### 2.5 What a write-new copy would have to contain

Minimal route (the one every input arm on record took: pop1, pop2, pf, time, pre2noid) [I]:

1. `<builder>.py`: writes the three arrays with exclusive create, hard-links them, prints parity statistics, asserts the PSI rule (template `build_xp_imp_c50.py`).
2. `launch_<arm>.sh`: copy of `nn_x/w12/kd/launch_pop1w.sh`; the `train.py` line with `--xt_add <columns>` and no `--xt_exclude`; variant `abl_e10_<arm>kd05w192_s<seed>`.
3. `read_<arm>.py`: copy of `arm_read8.py` with the arm's member names; reference = `w5_e10_kd05w192_s<seed>` (and `kd05t25w192`) at matched seeds.
4. Only if the arm goes into a file: `lane_full_<x>.sh` (copy of `lane_full_v.sh` + one arm), then the existing waiters and `ft_lane.sh`.

A fully separate harness (no new entry in `nn/feat_cache` or `nn_x/abl`) would need copies of `train.py`, `common.py` (the only place output paths are set), `data.py` (`FEAT_CACHE` pointing to a new directory of links to all 34 files + the new ones), `model.py`, `cd_arms.py`, `cd_arms4.py`, `cd_arms5.py`, `cd_arms6.py`, `cd_arms7.py`, `abl_arms.py`, `abl_retr.py`, `encdose/kv_probe/{kv_adapter,kv_data}.py` (all imported at the top of `train.py`, l.59-81), `blend.py`, `popfree/finetune.py`, `holdout_eval.py`, `stale_pred.py`. Two things then break: the gate's quarantine check looks for an input only in `nn/feat_cache` and `data/feats` (`presub_gate.py` l.120-124) and would skip the new files without a message, and `dump_teacher.py` reads `nn_x/w5/feat_cache` and would stop on the missing file. I see no gain in the separate harness [I].

### 2.6 Building time and smoke run

| Step | Time on record | Source |
|---|---|---|
| Leave-user-out history counts, all splits | 44 s CPU | `logs/features_luo.log` |
| Causal label statistics `tte_*` (12 columns, expanding window by train day; val / test from all train) | 366 s CPU | `logs/features_tte.log` |
| Co-visitation for train rows (per-day expanding source) | about 31 min CPU | `feats_v2/seq_/log_cv_train.log` |
| Derived column from an existing array (`xp_imp_c50`, `xp_ctr_c`) | not logged; a per-row map over 250M rows [I: minutes] | builders |
| Encoder family (pre2) | 13-59 GPU-min per windowed model, 6 models | `docs/02_SOLUTION.md` §12 |
| Smoke run (20,000 train feeds, 1 epoch, val + test prediction) | 3.0 min (`nn_x/abl`) and 3.9 min (`nn_x/w5`) from the args line to the json; 1.1-1.2 train minutes | [M `m04_timings.py` -> `out/m04_timings.txt`] |
| Held-out wide student, 10 epochs | 77.5-208.7 min in `nn_x/w5` (median 194; 16 runs), 252.8-261.9 min in `nn_x/abl` (4 runs) | [M `m04`] |
| Fine-tune n2cpv3s3 | 5.0-22.4 min (`nn_x/w5`), 13.7-27.3 min (`nn_x/abl`) | [M `m04`] |

Smoke flags: `--train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25 --cliplog 10` with a variant name `abl_smoke_*` (`abl_lane.sh` `ABL_SMOKE=1`, `lane_full.sh` `FD_SMOKE=1`). A smoke run needs a GPU (`common.py` refuses to import without one CUDA id); CPU-side checks are the unit tests `nn_x/abl/tests/test_abl.py` and the checkpoint cfg (`m01` shows how to read it on CPU).

### 2.7 Parity / PSI tools and the gate's input audit

| Tool | What it checks | Limits |
|---|---|---|
| `nn_x/abl/pop1/build_xp_imp_c50.py --stats-only` (and `pop2/build_xp_ctr.py`) | mean, zero share, cap share per split; PSI train vs val and train vs test on natural bins and on 10 train-quantile bins; asserts train-val PSI < 0.5 | written for one named column: a template |
| `feats_v2/<family>/drift.py <prefix>` | PSI on 10 val-quantile bins + NaN bin, val vs public vs private; medians; zero / NaN shares | val -> test only, not train -> val |
| `nn_x/audit/dpa_1113/pipeline/s01_extract.py`, `s02_stats.py` (CPU) | rebuilds the exact dense input of a checkpoint for train sample / holdout / val / public / private; per input: PSI, clip saturation, zero mass, fp16 ties, within-feed label AUC | hard-wired to checkpoint `r6_cd_m02_r101` and its own `out/`: needs a write-new copy pointed at the new checkpoint. This is the check "as the model sees it" [R `reports/DATA_PIPELINE_1113.md` §1.1-1.2] |
| `pretrain/gate_pre2.py` | for a train-row score family: train-fit GBDT deltas on val and holdout, cold-row mean z train vs val, has-later split | specific to `pre2_*` |
| `feat_new/it_hist_hz_gate.py`, `hz_gate.py`; the lagged-holdout gate of `reports/NEXT_1115/poplag.md` | label-free parity gates of lagged popularity views | staleness family |
| `src/pmix.py` | val reweighted to the public / private feed mix (M bin x cold-share bin) | a witness, not a parity check |
| `src/presub_gate.py` check D (has-later leak probe), check F (test / val agreement) | file level | after the fact |

Gate check C ("34 train-stage inputs audited", `nn_x/w12/kd/out/gate_FD16P20kd.txt` l.3-22: 34 for each of the 16 wide members, 27 for the 4 pop1 members): `audit_members` (`src/presub_gate.py` l.104-136) reads the logged input list of each member and applies four pattern lists: `BANNED_ANY` (`^it_hist_(all|et\d+)(_rk|_z|_gap)?$`, `^lh_`, `later`) and `BANNED_TRAIN` (`^pre_(fe|new|hmax|idmax|fe_noid|id_contrib)`, `_id_contrib`) -> FAIL; `WARN_TRAIN` (raw `prew_fe`, `prew_new`, `pre2_fe`, `pre2_new`) and `WARN_ANY` (`^als_`, `^hour$`, `^it_tr_`, `^u_tr_`, `^it_cand_cnt_train$`) -> WARN; plus FAIL if a file is hard-linked to `data/feats_quarantine`. It is a NAME check. [M `m05`]: `tte_it_coec`, `tte_u_ctr`, `tte_ui_pos`, `xp_ctr_c`, `ixh_mf1`, `trd_kde_s6h`, `clu_pmi1024_mean` pass silently; `it_tr_coec`, `u_tr_ctr`, `hour` WARN; and `it_cand_cnt_split` (an in-split candidate count that includes the same user's later feeds, dropped for that reason in `nn/finetune.py` l.87-89) also passes silently. A new block needs its own construction review and parity table; the gate will not supply them.

---

## 3. The positive control of COVERAGE_MAP l.323

### 3.1 What it was [R `reports/RECHECK_1114/skeptic_duplicates.md` §1.3; script `nn_x/w12/recheck/duplicates/d02_gate_cal.py`; outputs `.../out/d02_hold_pop.txt`, `d02_val_pop.txt`, `d02_val_pre2.txt`; SYNTHESIS row R10]

- **Block:** the 8 popularity / co-visitation inputs (`seq_cv_invd20`, `seq_cv_pop`, `seq_cv_norm`, `it_hist_all_luo`, `it_hist_et1..4_luo`).
- **Trained-input value:** pool of the 4 stored hard-label 10-epoch members WITH the block (`w5_e10_base_s951-954`, d128, 34 inputs) minus pool of the 4 members trained WITHOUT it (`abl_e10_xdrop_pop_s951-954`, same recipe and seeds, `--xt_exclude`). Pool = per-feed z-mean; NDCG per feed on the 30,000 in-period holdout feeds, selected checkpoints, paired: 0.32328 - 0.31476 = **+0.00852 ± 0.00066** (z 12.9). On val: 0.39887 - 0.39727 = +0.00160 ± 0.00102 (z 1.6).
- **Residual-form read:** base = the ablated pool. The 8 removed columns (train-array rows of the holdout feeds, z-scored within the feed) were offered (i) linearly, `base + w x z(column)`, one column at a time, w from a 12-point grid picked on one user half and applied to the other, best column picked post hoc; (ii) as a LightGBM lambdarank residual with the base as init score and a monotone constraint on the base columns, features = 6 context columns + the 16 block columns (raw and within-feed z), user-split 2-fold with all tuning (tree config, rounds including 0) inside the fit half (`nn_x/w12/recheck/forensics_skeptic/hx.py`). Holdout: +0.00086 ± 0.00029 (linear, `seq_cv_invd20`) and +0.00222 ± 0.00033 (GBDT). Val: +0.00011 ± 0.00045 and -0.00058 ± 0.00037.
- A second, weaker control in the record: the pre2 scalars on val only, 2 seeds, 3 of the 5 columns found: true +0.00061 ± 0.00083, linear +0.00039 ± 0.00043, GBDT -0.00030 ± 0.00020.

The members are hard-label d128 members, not students; the gate fits on 15,000 holdout feeds (7,200 val feeds) per half while the trained input was learned from 3.1 million feeds; and the contrast is a popularity-type contrast on the in-period holdout, where popularity inputs are 1.75 d stale against 9-11 d on test, so +0.0085 overstates the block's worth on test rows (2-epoch removal: +0.0023 val / +0.0074 holdout, `docs/02_SOLUTION.md` §11).

### 3.2 Re-run with a bootstrap and three more controls [M `m02_gate_cal_boot.py`; `out/m02_hold.txt`, `out/m02_val.txt`, per-feed deltas in `out/m02_*.npz`]

Same helper module and protocol; my run reproduces every number of the record's holdout and val tables for the popularity block (true value, 8 linear reads, GBDT read). Added: (a) a bootstrap over feeds (2000 draws, numerator and denominator drawn together) of recovered / true; for the linear lens the best column is re-picked inside each draw, so the interval carries the optimism of the post hoc pick; (b) the pre2 block on the holdout with all 5 columns; (c) two controls of the form the bundle would have, ONE column ADDED: `xp_imp_c50` (`abl_e10_pop1` against `abl_e10_xdrop_pop`, seeds 951-954) and the label statistic `xp_ctr_c` (`abl_e10_pop2` against the same base). Four controls x two witnesses x two lenses were looked at and all are reported; nothing was selected. For scale: the best of 8 pure-noise columns in the linear lens would read about +0.0004 at SE 0.00029 (expected maximum of 8 independent draws = 1.42 SE; fewer effective draws here, the columns are correlated); the block's best column reads +0.00086.

Holdout (30,000 feeds):

| Control | True value, selected checkpoints (z) | True value, last step | Linear nested (z) | GBDT residual (z) | Share linear [95%] | Share GBDT [95%] |
|---|---|---|---|---|---|---|
| 8 popularity inputs removed, 4 seeds | +0.00852 ± 0.00066 (12.9) | +0.00824 ± 0.00066 | +0.00086 ± 0.00029 (3.0), best of 8; other columns +0.00012 .. +0.00085 | +0.00222 ± 0.00033 (6.8) | 0.100; re-picked median 0.109 [0.057, 0.169] | 0.261 [0.190, 0.329] |
| pre2 scalars removed, 2 seeds | +0.00059 ± 0.00049 (1.2) | +0.00049 ± 0.00050 | +0.00028 ± 0.00019 (1.8), best of 5 | -0.00010 ± 0.00013 (-0.8) | 0.49 [0.04, 7.9] | -0.17 [-2.6, +2.1] |
| `xp_imp_c50` added, 4 seeds | +0.00201 ± 0.00046 (4.4) | +0.00164 ± 0.00046 | -0.00021 ± 0.00026 (-0.8) | +0.00007 ± 0.00020 (0.3) | -0.10 [-0.47, +0.14] | 0.03 [-0.19, +0.22] |
| `xp_ctr_c` added, 4 seeds | +0.00149 ± 0.00044 (3.4) | +0.00143 ± 0.00045 | +0.00026 ± 0.00024 (1.1) | +0.00024 ± 0.00018 (1.4) | 0.17 [-0.17, +0.57] | 0.16 [-0.09, +0.47] |

Val (14,423 feeds):

| Control | True value, selected (z) | True value, last step | Linear nested (z) | GBDT residual (z) |
|---|---|---|---|---|
| 8 popularity inputs removed | +0.00160 ± 0.00102 (1.6) | +0.00145 ± 0.00104 | best of 8 +0.00011 ± 0.00045; range -0.00028 .. +0.00011 | -0.00058 ± 0.00037 (-1.6) |
| pre2 scalars removed | +0.00061 ± 0.00083 (0.7) | +0.00008 ± 0.00085 | best of 5 +0.00039 ± 0.00043 | -0.00084 ± 0.00025 (-3.4) |
| `xp_imp_c50` added | +0.00020 ± 0.00077 (0.3) | -0.00027 ± 0.00079 | +0.00064 ± 0.00027 (2.4) | +0.00004 ± 0.00020 |
| `xp_ctr_c` added | +0.00033 ± 0.00075 (0.4) | +0.00005 ± 0.00077 | -0.00005 ± 0.00020 | -0.00000 ± 0.00025 |

The bootstrap intervals of the share on val are several units wide for every control and lens (for the block: [-0.2, +3.7] linear with the re-pick, [-3.8, +2.7] GBDT; `out/m02_val.txt`).

### 3.3 Sensitivity of the residual gate

| Witness | Lens | Sensitivity (recovered / true) | Basis |
|---|---|---|---|
| holdout | linear nested | **0.10** [0.06, 0.17] for the 8-column block with the post hoc pick; **-0.10** [-0.47, +0.14] and **0.17** [-0.17, +0.57] for one added column. Sum over the three resolved controls: 0.00091 / 0.01202 = 0.08 | [M] |
| holdout | monotone GBDT | **0.26** [0.19, 0.33] for the block; **0.03** [-0.19, +0.22] and **0.16** [-0.09, +0.47] for one added column. Sum over the three: 0.00253 / 0.01202 = 0.21 | [M] |
| val | linear nested | **not identified.** No control has a trained-input value above 1.6 SE on val; the residual reads are -0.00028 .. +0.00064 | [M] |
| val | monotone GBDT | **not identified**, and not harmless: the reads are -0.00084 .. +0.00004; the pre2 block reads -0.00084 ± 0.00025 (z -3.4; among 8 pure-noise reads the largest \|z\| has median 1.7 and 95th percentile 2.7), i.e. the gate reads as harmful a block that is load-bearing by its 2-epoch removal read (+0.00128 ± 0.00061 val, §1.4) | [M] |

What follows [I]:

1. The statement of the task ("10-25% at best") holds for the block. For the object the bundle consists of, a single added column, the holdout sensitivity is 0-17% with upper bounds of 14-57%: both single columns, worth +0.0015 and +0.0020 as trained inputs at 3.4 and 4.4 SE, fail any "2 SE" residual bar on the holdout (z -0.8 / +0.3 and +1.1 / +1.4). Two of two real inputs rejected.
2. On val the residual gate has no measurable relation to the trained-input value in either direction: the one residual read above 2 SE on val (`xp_imp_c50`, linear, +0.00064 ± 0.00027) belongs to a column whose trained-input value on val is +0.0002 ± 0.0008. A val residual read can neither close nor open a column; it also cannot be scaled up by 1 / sensitivity.
3. The val column of the true values is the more important finding for the design: with 4 members per arm, val does not resolve even the 8-column block (+0.0016 ± 0.0010), and reads the two single columns at +0.0002 and +0.0003 ± 0.0008. The in-period holdout resolves all three, but these are popularity-type inputs, the case in which the holdout is not admissible alone.

### 3.4 Caveats

One base pool (`abl_e10_xdrop_pop`) serves three of the four controls, so they are not independent. The bootstrap is over feeds given the fitted residual models and the trained members; it holds no seed-draw or re-fit noise. The GBDT round cap (250) was hit in one holdout half of the block control, so 0.26 may slightly under-read an uncapped gate [R `skeptic_duplicates.md` §6]. All controls are hard-label d128 members. Stored 2-epoch twins without `ctx_nd` and without `ctx_hn` + `ctx_hs` (`abl_xdrop_ctxnd`, `abl_xdrop_ctxhn_hs`, seeds 23 / 24) would give two non-popularity controls; I did not run them (2-epoch, 2-seed pools; true values +0.0014 / +0.0003 and +0.0005 / 0.0000 on record against a 2-seed calibrated SE of about 0.00065 on val, `nn_x/abl/ledger/ledger.md` §7.1).

---

## 4. Distillation: what the records say and the options for a clean test

### 4.1 How an input-level change was handled before under distillation [R unless marked]

| Case | What changed | Teacher | Read |
|---|---|---|---|
| pop1 students, d128 (`abl_e10_pop1kd05t25_s951-954`) | student view: 8 popularity inputs removed, `xp_imp_c50` added | `teacher_mean25` (8 of its 25 members are hard-label pop1 members) | as an equal pool partner of full-view students: FAIL on 11-14 (ledger `E_kd_predeclared.txt` l.36-45); as the 0.2 admixture of W8P20: val +0.00049 ± 0.00031, public +0.00052 (ledger 2026-11-14T21:24Z, 2026-11-15T02:15Z) |
| wide pop1 students (`abl_e10_pop1kd05w192_s951-954`; all-data s951-962) | same view, d192 | `teacher_mean11`, which never saw `xp_imp_c50` and did see the 8 removed inputs | held-out 4-pool 0.40000 / 0.31772; 0.8 z(W16) + 0.2: val +0.00047 ± 0.00031, holdout -0.00046 ± 0.00022 (COVERAGE row "Wide pop1 students"); the all-data ones are the 4 popularity-light members of FD16P20 (public 0.394892) |
| pop2 / pf wide students, all data (`abl_fd10_pop2kd05w192_s951-954`, `abl_fd10_pfkd05w192_s951-958`) | pf = 26 inputs; pop2 = 26 + the label statistic `xp_ctr_c` | `teacher_mean11` | read only as partners in the 0.2 pool: pop2 no; pf first draw adopted, then NOT REPLICATED; COVERAGE l.347 |
| input dropout (K / L:`xd30` re-cools; from-scratch `E:kwxd15` / `E:kwxd30` in the lane) | popularity-input dropout in training | unchanged | re-cools closed: val -0.00066, stale-val 0 (COVERAGE row "xd30 ..."); for the from-scratch arms I found lane logs and partial run files in `nn_x/w5/logs` but no read on record [M listing] |
| hz, hz2, hz2n re-cools | the TRAINING view of the popularity inputs in the cooldown; same columns at inference | unchanged | hz passes at pool level on stale-val, closed at file level; COVERAGE rows of 11-15 |
| th / hzt | the TEACHER: half popularity-light (`teacher_half_p1.npy`) | changed | no pass; ledger 2026-11-15T07:49Z, 08:57Z |
| teacher identity (mean11 vs mean25), generation 2, cross-fitted teacher | teacher only | changed | mean11 = mean25 for the students; generation 2 -0.0005 val; cross-fitted teacher 4-pool 0.40079 / 0.32347 against 0.40134 / 0.32408; ledger l.59, l.84-93 |
| view-teacher (pop1 students from a pop1 teacher) | teacher of the student's view | proposed, 37 GPU-h, never run | `reports/TRICKS_1115/SYNTHESIS.md` l.188 |

In every case the student was read with the OLD teacher; no teacher was ever regenerated for an input change. The input flags of all these students live in `nn_x/abl/train.py`.

### 4.2 What the stored predictions say about a student whose view differs from the teacher's [M `m03_kd_view_contrasts.py` -> `out/m03_kd_view_contrasts.txt`]

Held-out 10-epoch members, seeds 951-954, selected checkpoints (the last-step table gives the same picture):

| Contrast | Val, member mean | Val, 4-pool | Holdout, member mean | Holdout, 4-pool |
|---|---|---|---|---|
| Distillation gain, full view d128, teacher_mean11 (`kd05` - `base`) | +0.00257 ± 0.00047 | +0.00198 ± 0.00061 | +0.00200 ± 0.00028 | +0.00021 ± 0.00036 |
| Distillation gain, full view d128, teacher_mean25 | +0.00303 ± 0.00049 | +0.00208 ± 0.00064 | +0.00192 ± 0.00028 | -0.00011 ± 0.00036 |
| Distillation gain, pop1 view d128, teacher_mean25 (`pop1kd05t25` - `pop1`) | +0.00403 ± 0.00048 | +0.00196 ± 0.00064 | +0.00219 ± 0.00029 | +0.00038 ± 0.00037 |
| View gap full - pop1, hard labels d128 | +0.00236 ± 0.00080 | +0.00140 ± 0.00100 | +0.00623 ± 0.00053 | +0.00651 ± 0.00064 |
| View gap full - pop1, distilled d128 (mean25) | +0.00136 ± 0.00071 | +0.00151 ± 0.00086 | +0.00596 ± 0.00049 | +0.00603 ± 0.00056 |
| View gap full - pop1, distilled d192 (mean11) | +0.00059 ± 0.00074 | +0.00134 ± 0.00090 | +0.00624 ± 0.00053 | +0.00636 ± 0.00061 |
| ONE added column `xp_imp_c50`, hard labels (`pop1` - `xdrop_pop`) | +0.00015 ± 0.00059 | +0.00020 ± 0.00077 | +0.00184 ± 0.00036 | +0.00201 ± 0.00046 |
| ONE added column `xp_ctr_c`, hard labels (`pop2` - `xdrop_pop`) | +0.00069 ± 0.00057 | +0.00033 ± 0.00075 | +0.00120 ± 0.00034 | +0.00149 ± 0.00044 |

All-data wide students, teacher_mean11, val only (their holdout is in-sample):

| Contrast | Member mean | Pool |
|---|---|---|
| `xp_imp_c50` added (pop1 - pf), seeds 951-954 | +0.00035 ± 0.00052 | -0.00027 ± 0.00066 |
| the same, seeds 955-958 | +0.00126 ± 0.00052 | +0.00087 ± 0.00067 |
| the same, 8-pools of seeds 951-958 | +0.00080 ± 0.00042 | +0.00040 ± 0.00060 |
| `xp_ctr_c` added (pop2 - pf), seeds 951-954 | -0.00017 ± 0.00050 | -0.00037 ± 0.00065 |
| A/A, pf seeds 955-958 minus pf seeds 951-954 | -0.00068 ± 0.00043 | -0.00105 ± 0.00056 |
| A/A, pop1 seeds 955-958 minus pop1 seeds 951-954 | +0.00024 ± 0.00043 | +0.00009 ± 0.00055 |

Readings:

1. The benefit of distillation does not depend on the student sharing the teacher's inputs: the pop1-view students gain as much (member +0.0040 val, +0.0022 holdout) as full-view students (+0.0026..+0.0030, +0.0019..+0.0020) [M].
2. A student does not obtain through the teacher what its own inputs do not carry, and what its inputs carry stays worth the same under distillation: the view gap on the holdout is +0.0060..+0.0064 (distilled) against +0.0065 (hard labels) at pool level [M]. This is the mirror case (teacher knows more than the student). It says an input's worth is not absorbed by the teacher term; it does not measure the forward case.
3. The forward case (student has a column the teacher never saw) exists once: `xp_imp_c50` and `xp_ctr_c` in wide students of `teacher_mean11`. It is readable on val only, where the hard-label value of the same columns is itself unresolved (+0.0002 / +0.0003 ± 0.0008) and the distilled reads are +0.00040 ± 0.00060 (8-pools) and -0.00037 ± 0.00065, inside an A/A spread of 0.001 between two 4-pools of one view [M]. No held-out popularity-free student exists, so there is no holdout read. **The attenuation by a blind teacher is not measured anywhere in the records.** Indirect and weak: on the holdout the wide pop1 students sit +0.00095 above the hard-label pop1 pool (0.31772 against 0.31677) and the wide full-view students +0.00080 above the hard-label base pool (0.32408 against 0.32328), so the pop1 students show no missing 0.001-0.002 that a halved `xp_imp_c50` would leave [M levels in `out/m03`; I for the inference; teacher_mean11's own popularity inputs partly stand in for the exposure column, so this is not a clean case].
4. Why attenuation is expected [I]: the target is 0.5 x click distribution + 0.5 x softmax(teacher). The teacher term is a function of the old inputs, so to first order the student's best response moves its probabilities by half of what the new input would move them under hard labels. Because NDCG reads only the order, half a shift usually keeps more than half of the gain; "between half and all of the hard-label value" is my prior, with nothing on record to narrow it. The claim on record that the old teacher "halves its pull" (`reports/TRICKS_1115/int_weak.md` row T16) is this argument, not a measurement.

### 4.3 Options for a clean test, with cost from the logs [M `m04_timings.py` -> `out/m04_timings.txt` for all minutes; process wall-clock on GPUs shared by up to 4 jobs, i.e. upper bounds]

**A. Hard-label members with and without the input.**
- Recipe of the hard-label era: `E_BASE` + `HYG`, d128, 10 epochs, no KD flags (`nn_x/w5/lane.sh` `E:base`; in the input harness `nn_x/abl/pop1/launch_e10.sh` without its two pop1 flags, plus `--xt_add`).
- "Without" is on disk: `w5_e10_base_s951-956`, 6 held-out members with val, holdout and last-step predictions [M file listing]. "With": 4-6 new runs.
- Cost per run on record: 56.8-59.7 min (`abl_e10_pop1`), 118-126 (`abl_e10_xdrop_pop`), 130.8-182.1 (`w5_e10_base`): 1-3 h, so 4-18 process-hours for 4-6 members. A 2-epoch screen is 15-37 min per run (`abl_base`, `abl_pop1`, `abl_time`).
- Paired SE of such 4-pool contrasts on record [M `m03`]: holdout ±0.00044..0.00066, val ±0.00075..0.00102. Hard-label acceptance rule of the era: paired val > 2 SE pooled over >= 2 seeds and holdout >= 0 (STATUS §3).
- For: no teacher in the loop; the regime of all four positive controls; its members are what a regenerated teacher needs. Against: it is not the recipe that ships; 7 of 8 levers closed under hard labels did not flip under distillation and one (width) did (`skeptic_duplicates.md` §5.1), so a hard-label result needs option B or C to be believed in the final recipe.

**B. Distilled students with the old teacher.**
- Command: §0.2 item 2. Reference: the 16 stored held-out wide students `w5_e10_kd05w192_s951-958` and `w5_e10_kd05t25w192_s951-958` at matched seeds (rule "reference for pool claims", COVERAGE row of 11-14).
- Cost per run on record: 77.5-208.7 min in `nn_x/w5` (three runs at 78 min, median 194), 252.8-261.9 min in `nn_x/abl`: 8 students = 26 process-hours at the `nn_x/w5` median, 34 at the `nn_x/abl` one, about 10 at the fastest rate on record ("27 GPU-h" in `int_weak.md`); fine-tunes 5-27 min each if a file is wanted.
- Paired SE of 8-pool contrasts on record: val ±0.00049..0.00058, holdout ±0.00027..0.00033 for same-view arms (COVERAGE rows kwwd05, kwaux0); val ±0.00060 for 8-pools of different views [M `m03`]. A/A null of the pooled-z rule: sd 0.80 (COVERAGE row "A/A null").
- For: the current recipe, the stored reference, the standing 8-seed rule. Against: biased toward zero by an unknown factor (§4.2 item 4): a pass is conservative, a null closes nothing.

**C. Teacher regeneration.**
- What a teacher is: the mean of per-feed-centred logit sums of hard-label train-only members over all 246.7M train candidates (`nn_x/w12/kd/dump_teacher.py`; the dump reads each checkpoint's own input list, so new-input members need no code change; <= 4.6 min per tag of 2 checkpoints; the averaging step has no script on disk) [R `docs/02_SOLUTION.md` §6.1, §12].
- Cost of the members: `teacher_mean11` = 7 ten-epoch roots (117.6-156.7 min) + 7 re-cools (23.1-50.6 min) + 4 base members (180 min) = 30.9 process-hours [R `docs/02_SOLUTION.md` §12]. The 4 base-type members alone (= option A's "with" arm) are 9-12 h. Then 8 students as in B.
- The control problem [I from R]: students of a differently built teacher (3 members per fold, out-of-fold) read 0.0005-0.0006 below students of `teacher_mean11` on both witnesses (ledger l.84), while mean11 and mean25 students are level. A regenerated teacher must therefore be compared with a teacher built the same way without the input (for a 4-member teacher: the 4 stored `w5_e10_base` checkpoints, re-dumped; the stored tags d / e / f mix them with other members [M `nn_x/w12/kd/out/dump_{d,e,f}.log`]), or be the 11-member recipe in full. With a partial replacement (4 of 11 members see the input) the teacher is only 36% aware.
- For: the only arm in which the input counts fully. Against: cost 2-3 times that of B, and one more thing that can differ between arm and reference.

Not available: the re-cool harness (`lane.sh` stages K / L, 52-58 min per run) cannot carry a new input, because with `--init` the inputs and layer shapes come from the snapshot cfg and `nn_x/abl/train.py` refuses the input flags (l.299-300).

Ordering that reuses work [I]: A's "with" members answer the hard-label question on both witnesses, are the positive-control regime, and are the raw material of C; B can start in parallel because its reference is on disk.

### 4.4 Power facts a rule must respect [M unless marked]

| Fact | Number | Source |
|---|---|---|
| 4-pool against 4-pool of different input views, val | SE 0.00075-0.00102 | `out/m03`, `out/m02_val.txt` |
| the same on the holdout | SE 0.00044-0.00066 | `out/m03` |
| 8-pool against 8-pool of different views, distilled, val | SE 0.00060 | `out/m03` part B |
| 8-pool same-view arms, distilled | val SE 0.0005-0.0006, holdout 0.0003 | [R] COVERAGE rows kwaux0, kwwd05 |
| seed-draw noise between two 4-pools of one view on val | -0.00105 ± 0.00056 (pf), +0.00009 ± 0.00055 and +0.00011 ± 0.00056 (pop1) | `out/m03` part B |
| an 8-column block worth +0.0085 on the fresh holdout | +0.0016 ± 0.0010 on val | `out/m02_val.txt` |
| single added columns worth +0.0015..+0.0020 on the fresh holdout | +0.0002..+0.0003 ± 0.0008 on val; +0.0004 ± 0.0006 and -0.0004 ± 0.0007 under distillation | `out/m02_val.txt`, `out/m03` |

For inputs of the popularity / exposure kind the two witnesses disagree by a factor of 5-10 and the in-period holdout is the one that is not admissible alone; for inputs that are not stale on test rows the holdout is the sharper witness by a factor of about 2 [I].

---

## 5. Corrections to statements on record

1. "No student arm with new trained inputs was ever run under the current recipe" (task text; `int_weak.md` T16 "none was ever a trained input of a student"). True for the 39 T16 rows and for the full 34-input view. Not true without that qualifier: `xp_imp_c50` (4 held-out + 12 all-data wide students) and the item click-rate statistic `xp_ctr_c` (4 all-data wide students) were trained inputs of d192 students of `teacher_mean11`, on the popularity-free view, read as admixture members [M `m01` cfg of `abl_fd10_pop1kd05w192_s951`; file listing of `nn_x/abl/preds`]. Their direct with / without contrasts under distillation are in §4.2 (val only, unresolved).
2. "Nothing on val in residual form" (COVERAGE l.323) reads as a measured zero. It is an unidentified ratio: the trained-input value itself is 1.6 SE on val (§3.3).
3. `skeptic_duplicates.md` §1.3 lists five direct NN tests of added inputs as "all null or negative (xrel_more, pop2, TIME, kv64, retr_cat)". Against its own base (`abl_e10_xdrop_pop`) pop2's column reads +0.0015 ± 0.0004 on the holdout at 10 epochs at pool level (z 3.4) and pop1's +0.0020 ± 0.0005 (z 4.4); both are null on val [M `m02`]. The four remaining nulls are rank / z channels of existing columns (xrel_more), 3 feed-level time columns at 2 epochs (TIME) and two architecture changes (kv64, retr_cat): none is a new per-candidate information column on the full view.
4. COVERAGE row "KD re-cool harness" quotes "cooldown-only 0.25 GPU-h runs"; the logged process wall-clock of the 16 reference re-cools is 51.9-57.6 min each on shared GPUs [M `m04`]. Plans should use the logged figure.

---

## 6. Limits

- CPU only: no forward pass was run; the input list comes from checkpoint cfgs, logs and code, the column names of `out/dense_columns.tsv` are mine in the code's order.
- The meaning of each pre-computed column is taken from the builders' docstrings and kernels' comments, not re-derived from the arrays.
- §3 and §4.2 use stored predictions of hard-label d128 members and of students; per-feed SEs carry no seed-draw noise; several contrasts share members.
- The blind-spot list of §0.1 / §1.9 says what is not read, not what is admissible or valuable; admissibility under the leak and parity rules is the business of the other inventories of this workflow.
- I did not open any test label (none exists) and wrote nothing outside `reports/INPUTS_1116/model_inputs/` and this file. Python byte-code and numba caches were kept out of the project (`PYTHONDONTWRITEBYTECODE=1`, `NUMBA_CACHE_DIR` in the scratchpad).

## 7. Files

| File (under `reports/INPUTS_1116/model_inputs/`) | Content |
|---|---|
| `m01_ckpt_inputs.py` -> `out/m01_ckpt_inputs.txt`, `out/dense_columns.tsv`, `out/xt_files.tsv`, `out/ft_extras.tsv` | input lists from three checkpoints, the 104 dense columns in order, file checks, the 42 extras, unused cache files |
| `m02_gate_cal_boot.py val\|hold` -> `out/m02_val.txt`, `out/m02_hold.txt`, `out/m02_*.npz`, `out/m02_*.stdout` | positive controls of the residual gate with bootstrap |
| `m03_kd_view_contrasts.py` -> `out/m03_kd_view_contrasts.txt`, `out/m03.stdout` | hard-label and distilled view contrasts |
| `m04_timings.py` -> `out/m04_timings.txt` | minutes per run family from the run json files |
| `m05_gate_names_and_store.py` -> `out/m05_gate_names_and_store.txt` | the gate's name audit on candidate names; three-split arrays on disk |

Run form (all CPU): `cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=<scratch> .venv/bin/python reports/INPUTS_1116/model_inputs/<script> [args]`. Measured wall-clock at 12 threads: `m02 val` 4 min, `m02 hold` 17 min, `m03` 3 min, the others seconds [M file times of `out/m02_*.stdout`, `out/m03.stdout`].
