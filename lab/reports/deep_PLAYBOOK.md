# deep_PLAYBOOK: verified, integrated implementation plan (Avito ML Cup 2026, personal ranking)

Date: 2026-11-11. Role: verifier and integrator over every report in `reports/`:
- 8 deep reports: `deep_recsys2024`, `deep_recsys_engagement`, `deep_otto_hm`, `deep_position_bias_random`, `deep_ru_competitions`, `deep_neural_rankers`, `deep_seq_pretrain_cf`, `deep_gbdt_ranking`;
- 3 research reports: `research_competition_avito`, `research_kaggle_solutions`, `research_sota_ltr`;
- 4 EDA reports: `eda_labels_position`, `eda_shift_overlap`, `eda_embeddings`, `eda_collab_stats`;
- `nn_report`.

**Tags**
- **[V]**: I read the primary source in this session; URL given.
- **[V-rep:X]**: verified from a primary source by report X; I did not re-read it.
- **[E]**: our own experiment or log; path given. **[E\*]**: I recomputed it this session from `preds/*.npy` with `src/metrics.py`.
- **[S]**: search snippet only.
- **[K]**: background knowledge, not verified.
- "est.": my own estimate of the gain, not a measurement.

All NDCG values are val full-list NDCG (14,423 feeds, pessimistic ties). A single score has SE ≈ 0.0018. A paired difference between two strong models has SE ≈ 0.0005-0.0013.

---

## 0. Where we actually are (the starting point for every item below)

The task status ("LGB OOF ~0.365") is out of date. The neural ranker is now the main engine.

| prediction (val) | NDCG | source |
|---|---|---|
| NN ensemble, train-only (no val labels used) `nn_ens_stage1_trainonly_val` | 0.37900 | [E\*] |
| NN stage-2 on random-like train feeds `nn_s2_nopos_aux_sim_trainonly_val` (checkpoint picked on val, so slightly optimistic) | 0.37988 | [E\*] |
| **NN ensemble, 5-fold val fine-tune, snapshot mean** `nn_ens_stage1_val_oof` | **0.38108** | [E\*] |
| Best single NN fine-tune (`aux_sim_s5` + item_safe extras, snapshot) | 0.38069 | [E\*], `nn_report.md` |
| CatBoost YetiRank, train-fit on 300K feeds `cb_tr300k_none_val` | 0.36997 | [E\*] |
| LightGBM lambdarank, train-fit on 300K feeds `lgb_tr300k_none_val` | 0.36822 | [E\*] |
| XGB `rank:ndcg`, train-fit on 1M feeds `xgb_tr1m_ndcg_val` | 0.36770 | [E\*] |
| LightGBM val-CV OOF (122 features) / with init_score from the train-fit LGB | 0.3665 / 0.3683 | [E] `exp/res/l_base.json`, `logs/stack1.log` |
| Blend NN ens + 0.15·xgb_tr1m (per-feed z-scores) | +0.00054 ± 0.00045 over NN ens | [E\*] |

Translating to the leaderboard:
- **LB ≈ val − 0.005 to 0.008** for the same model. The cause is feed-size composition: test feeds are longer [E `eda_shift_overlap §5.2`, `eda_labels_position §5.2`]. The public baseline repo shows the same gap, 0.3533 val → 0.3461 public [V-rep:deep_ru_competitions].
- So our 0.381 is about **0.374-0.376 public**.
- Public #2-#10 sit at 0.3884-0.3900. That is about **0.394-0.396 in val terms**, or about 0.390 without the later-history leak, which is worth roughly +0.0047 on public (§2, C1).
- **The realistic gap to the top 10 is therefore about +0.008 to +0.012 val NDCG.** That is the size of a new signal family plus stacking and ensembling, not of hyper-parameter tuning.

---

## 1. Verification of the 10 highest-impact [S]/[K] claims

| # | Claim (report) | Primary source read | New status | Consequence for us |
|---|---|---|---|---|
| 1 | "Transductive / leak features: check the rules" (all reports) | Competition page https://ods.ai/competitions/avitotechrecsys2026 [V]; rules PDF text (`scratchpad/rules.txt`, saved by an earlier agent; I searched it for data-use, leakage, test-data and selection clauses and read §5.3, §7.4 and §8.6) | **[V] No clause forbids using test inputs transductively** (test histories, later feeds, test candidate sets). §5.3 forbids third-party IP and proprietary technology, and software must have a commercial-use licence. §8.6.3: the top 10 must send a report plus code. The organiser checks "compliance with the Rules, absence of violations and technical failures" and assigns places **"at its own discretion"**. Page: Public = first ~30% of feeds by time; ties put negatives first; NDCG is defined with IDCG@30, which equals full IDCG while a feed has ≤30 positives. The number of final selections is **not stated** (selection window Dec 9, 00:00-12:00). | Transductive features are allowed by the letter of the rules. The organiser has discretion, so the label-like later-history flag (item 8) is the only part with real risk: keep a final candidate without it. |
| 2 | Expedia 5th place: "0.5243 vs 0.5115 NDCG@38 without normalisation/rank features" (research_kaggle_solutions [S]) | arXiv 1311.7679 PDF, read via pdftotext [V] https://arxiv.org/pdf/1311.7679 | **Refuted as an ablation.** 0.51149 and 0.5243 come from two *different* team members' LambdaMART models with different feature sets (CTR/CVR+raw vs normalised+rank+result features). The paper's real ablation: **one-way count features over train+test (transductive) moved their GBM from 0.50099 to 0.52477** (+0.024). | Within-feed normalisation is supported by *our* data (+0.0046 on train→val transfer [E `eda_shift_overlap §7.3`]), not by this paper. Transductive counts were big at Expedia but are **useless here**: val/test items are singletons inside their split (in-split candidate count −0.0018 [E]). |
| 3 | Expedia 1st place (Owen Zhang) used "estimated position" = the hotel's mean position in non-random searches (deep_position_bias [K, low confidence]) | Two searches; slides not found (Kaggle GCS 403, as before) | **Unverifiable; stays [K].** | Our analogue, item mean original pos from train, scores only +0.0054 over random alone, against +0.0222 for train-history event-1 counts [E `eda_collab_stats §3-4`]. Low priority. |
| 4 | PAL (Guo et al., RecSys'19): P(click) = P(seen\|pos)·P(click\|x), position tower dropped at inference (research_sota_ltr, deep_position_bias [K]) | ACM full text 403; search only [S] https://unpaywall.org/10.1145%2F3298689.3347033 | **Still [S].** Secondary descriptions agree with the claim. | Moot for us: our additive position tower is neutral (nopos 0.37385 vs base 0.37361 [E `nn_report`]). The useful form is the auxiliary pos-distillation head (+0.0018). |
| 5 | DropoutNet-style "ID-dropout p≈0.3-0.5" so a hybrid ID+content model still works for cold items (deep_seq_pretrain_cf [K]) | NeurIPS'17 PDF [V] https://papers.nips.cc/paper_files/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf | **[V]** Preference inputs of a sampled fraction τ of users/items are set to 0 during training. Warm-start recall "remains virtually unchanged decreasing by less than 1% until dropout reaches 0.7". Cold-start rises with dropout, and "even dropout of 0.1 improves it by over 60%". | Any learnable item-ID component (items 17, 19) must use ID-dropout of 0.3-0.5. 35-40% of test candidate items never occur in train. |
| 6 | "Negative downsampling inside feeds is fine for binary objectives; ranking is invariant" (deep_gbdt [K]) | He et al. 2014, Facebook ads, PDF [V] https://quinonero.net/Publications/predicting-clicks-facebook.pdf | **[V]** Best negative downsampling rate was 0.025 on their data. Recalibration is q = p/(p+(1−p)/w), a monotone map, so the order within a feed is unchanged. | We may downsample negatives for **binary/pointwise** train-fit GBDTs (item 16). **Never** for lambdarank/listwise objectives, because that changes ranks and IDCG [V-rep:deep_gbdt, source semantics]. |
| 7 | XGBoost GPU stores "~1-2 bytes per value", so all 247M rows fit (deep_gbdt, deep_otto_hm [K]) | XGBoost v3.4.1 `doc/gpu/index.rst` [V] https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/doc/gpu/index.rst | **[V, approximately]** "The compressed, quantised ELLPACK format will commonly use 1/4 the space of a CSR matrix stored in floating point". CSR fp32 is about 8 B per value, so ELLPACK is about 2 B per value. | 247M × 122 features ≈ 60 GB on GPU N. That only fits when GPU N is mostly free; it swung between 100 and [n] GB free [V-rep:deep_gbdt]. Default to 1-1.5M-feed samples. |
| 8 | LightGBM binned dataset ≈ 1 byte per value (deep_gbdt [K]) | LightGBM v4.7.0 `src/io/bin.cpp` [V] https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/io/bin.cpp | **[V]** `num_bin<=16`: 4-bit; `<=256`: `uint8_t`; `<=65536`: `uint16_t`. | 247M × 122 ≈ 30 GB of host RAM with `max_bin≤255`. CPU-only (positions are CPU-only anyway); 24 threads is the limit. |
| 9 | Chris Deotte (OTTO 3rd) computed co-visitation on train+**test** sessions (deep_seq_pretrain_cf [K]) | Repo notebook `infer/candidates_LB/model-11-1-LB.ipynb` [V] https://github.com/cdeotte/Kaggle-OTTO-Comp | **[V, partially]** The LB notebook loads matrices named `top_20_test` / `top_20_test2` and hard-codes popular items marked "# COMPUTED FROM TEST DATA". Exactly which sessions built each matrix is not stated. | Supports building co-visitation / ALS / i2v transductively from train+val+test histories for **val/test rows**. Train rows still need leave-user-out (C2). |
| 10 | EMA of weights gives +0.001-0.002 (deep_neural_rankers [K]) | Morales-Brotons et al., TMLR, arXiv 2411.18704, abstract [V] https://arxiv.org/abs/2411.18704 | **[V], qualitative only.** "EMA models not only generalize better but also exhibit improved robustness to noisy labels"; EMA "requires less learning rate decay". No ranking numbers. | Our own data is stronger evidence: per-epoch snapshot averaging gave +0.0017 (0.37530 → 0.37704) in val fine-tune [E `nn_report`]. Add EMA to pretraining (item 5). |

Also re-checked:
- The dataset page [V] https://ods.ai/competitions/avitotechrecsys2026/dataset confirms:
  - numeric transforms are "обучены только на train" (fit on train only);
  - each split has its own embedding table;
  - `pos` exists only in train, "после фильтрации позиции могут содержать пропуски" (after filtering, positions can have gaps);
  - the target is "хотя бы одного события из целевого набора" (at least one event from the target set), with every action weighted equally.
- Claims tagged [S] in `research_kaggle_solutions` that other deep reports have since verified:
  - RecSys'24 leak ablation 0.8864 → 0.7699 [V-rep:deep_recsys2024];
  - FeatureSalad contextual normalisation [V-rep:deep_recsys2024];
  - Expedia `random_bool` handling [V-rep:deep_position_bias_random].

---

## 2. Contradictions between reports, resolved

| ID | Contradiction | Resolution (evidence) |
|---|---|---|
| **C1** | Later-history leak size. Task status: "27.6% positive rate, +0.005". `eda_collab_stats`: 27.6%, 354 feeds, +0.0048, "334 test feeds, about +0.002". `history_leak/03_leak.log`: 93% target rate, 113 feeds, +0.0051. `eda_shift_overlap`: within-val 25 feeds, +0.0012; test +0.0026. | **The 27.6% and the 354/334-feed figures are contaminated by `user_id = 0`.** That id is a pooled "no user" bucket holding 4.6% of val feeds and 123,904 train feeds; `eda_collab_stats` calls it "a bot". `out_05c.log` counts "feeds whose user has a later feed" = 4.7%, which is user 0's share. Excluding user 0: val target rate 93.5% on 184 flagged candidates in 110 feeds, using val+test as sources [E `03_leak.log`]. The val +0.005 uses **test** histories for val feeds (94 val users also appear in test), so it is a val-only inflation. **Achievable on test** (test-only sources, user≠0): 103-108 feeds → **about +0.0025 on all of test, +0.0047 public, +0.0017 private** [E `eda_shift_overlap §6`]. `src/later_history.py` already filters `user > 0`. Good. |
| C2 | Positives reappearing in later history: 92-94% (`eda_shift`, `eda_labels`) vs 54.4% (`eda_collab_stats`, `deep_seq_pretrain_cf`). | Different denominators. 92-94% is conditional on the user having a later feed (consecutive pairs). 54.4% is unconditional inside a 60K-user sample, where many positives have no later snapshot. Both mean the same thing: **the target action is logged as a history event (mostly event_type 1) in later snapshots.** Any history aggregate that includes the same user's later snapshots leaks labels into train rows. |
| C3 | Random-order feeds inside train: about 346K (`deep_position_bias_random`) vs about 206K (`eda_labels_position`). | 346K = expected count over all 1.64M scored feeds, including M<50 where the mixture is not identifiable [E `04g_full_posterior.log`]. **Use 206K (M≥50, refit `04h`)**, with about 90K at P>0.9 (10-20% contamination). |
| C4 | Position as a training-only feature. `eda_labels_position`: +0.0042 (XGB, RICH, 200K, pos+rkend at (20,50)). `eda_shift_overlap §7.4`: +0.0019 vs f1 (BASIC, 60K, pos=20). `logs/tr300k_feat.log`: pos only at pos=5 gives 0.3676 vs none 0.3682 (LGB, 110 features, 300K). `deep_gbdt`: LightGBM `position=` +0.0022 (100K). Airbnb: −1.3% for an undropped position feature [V-rep:deep_position_bias_random]. | **The gain shrinks as features get richer:** BASIC +0.0067, RICH +0.0042, 110 features with pos only about 0. Inferring at pos=0 hurts everywhere (−0.005 to −0.016), the same failure mode as Airbnb's. rkend carries 57% of the bias gain share, and the 300K LGB run lacked it. **For the NN the position tower is neutral, while the pos-distillation aux head gives +0.0018.** Conclusion: position debiasing is a ≤+0.002 lever for rich GBDTs and not a priority. |
| C5 | Which objective. Baidu-ULTR: LambdaRank > listwise > pointwise [V-rep]. `deep_gbdt` val-CV: binary ≈ rank:ndcg ≈ YetiRank ≈ QuerySoftMax. `eda_labels`: pairwise beat binary on train (+0.0054), binary better on val-only (+0.0039). RecSys'24/BlackPearl: QueryCrossEntropy best. | It depends on data size: pairwise on train-scale data, pointwise on 14K val feeds. `rank_xendcg` with label 1 is significantly worse (−0.0047, z=−3.9) [E]. NN: softmax ≥ BCE (0.37361 vs 0.37272, n.s.). **Choose objectives for diversity, not for expected gain.** |
| C6 | "Fine-tune / retrain on the latest labelled period gives +0.010 AUC" (RecSys'24 [V-rep]) vs "adding val or fine-tuning on val: ±0.001" (`eda_labels` d1/d2/e1, XGB). | NN val fine-tune without extras: +0.001 (0.37218 → 0.37322). With the 27 val/test-only item features: +0.003. With snapshot averaging: +0.005 (→ 0.37704) [E `nn_report`]. **The gain comes from val/test-only features plus snapshot ensembling, not from val labels as such.** |
| C7 | Set attention: Avito production +0.6% [V-rep:deep_ru] vs our ablation, neutral (noset 0.37386 vs 0.37385). | Trust our data. Within-feed z/rank features already supply list context. Keep the set block, which is cheap, but do not invest in it. |
| C8 | "Best single feature" = max-sim vs last-32 at 0.349 (task) vs rec_hl3 0.3493 / max_sim 0.3436 (`eda_embeddings`). | Different tie handling and definitions. Recency-weighted mean cosine (exp half-life 3-5 events) is the best handcrafted signal, and the learned 330-parameter attention gives 0.3584. All of these are inside the NN already. |
| C9 | Item target encoding: "TE/COEC is the core of Avito'24/'25 winners" (`deep_ru`, `deep_recsys_engagement`) vs "train CTR/COEC weak or negative" (`eda_shift`, `eda_collab`). | Our data: popularity counts > COEC > CTR. train-hist et1 count +0.0222 vs COEC +0.0118 alone; in a blend with f1, COEC −0.0014 and counts +0.003 [E]. Production CTR is position- and selection-confounded. **Use counts with a "seen" flag. Use COEC only with strong smoothing (a≥20).** |
| C10 | Train is "post-blender, category-sampled" (`research_competition_avito`, `deep_ru`) vs "category adjacency equals random" (`eda_labels` obs/exp 0.990). | Consistent. Slots sample categories i.i.d. from the user-interest distribution [V-rep: habr 974682], which gives random-looking adjacency. |
| C11 | "Debiasing the 3.15M train feeds is the biggest structural gain" (`deep_ru` H1, `research_*`). | Partly superseded. The NN trained on train with no debiasing (0.379) already beats every val-only model. Debiasing on top adds ≤0.002 (C4). The remaining value of train is **random-like feeds (C3), pos distillation, and history statistics.** |
| C12 | Transductive popularity "dominant" (RecSys'24 +0.10 AUC; Expedia +0.024) vs "useless" (`deep_recsys2024 §7.1`, `eda_shift §4.5`). | Our val/test are thin exploration samples: 91%/86% of candidate items appear in exactly one feed of the split. In-split *history* counts give only +0.0014-0.0020. Global (train+val+test) history counts do carry signal. |

**New issue found during integration (not in any report): label leak in the existing `it_hist_*` features on train rows.**
- `src/features.py::global_item_stats` counts item events over *all* histories of train+val+test, every snapshot, with no per-user exclusion.
- By C2, a train positive X of user u reappears in u's later train/val/test snapshots, so it gets +k counts from its own label.
- The train-fit GBDTs rank `it_hist_all_rk` and `it_hist_et1` among the top gain features [E `logs/tr300k_*.log`].
- `LEAKY_ON_TRAIN` in `train_lgb.py` excludes only `it_tr_*` and `u_tr_*`.
- Val rows are affected too, through the test snapshots of the 92 users present in both val and test (about 94 val feeds).
- Fix: item 1.

---

## 3. Ranked playbook (top 25)

Ordering: expected val gain × confidence ÷ cost, given the compute rules (GPU N only via `CUDA_VISIBLE_DEVICES=N`, ≤24 CPU threads, machine heavily loaded).

Column names refer to `data/flat/<split>/`:
- per candidate: `cand_item_id`, `cand_category_id`, `cand_emb_idx`, `cand_f1..f5`, `cand_fidx`, `cand_target`, `cand_pos` (train only);
- per history event: `hist_item_id`, `hist_event_type`, `hist_event_source`, `hist_category_id`, `hist_hpos`, `hist_emb_idx`, `hist_fidx`;
- per feed: `feeds.parquet` (`fidx`, `user_id`, `timestamp`, `M`, `H`), plus `user_emb.npy`.

### Correctness first

**1. Make every history-derived item statistic leak-safe.**
- **What:**
  - For **train rows**, replace `it_hist_all` / `it_hist_et{e}` (and their `_rk/_z/_gap`) with a leave-user-out version: `cnt_LUO[row] = cnt_global[item] − n_u_later[item]`. Here `n_u_later` = events of `item` in histories of feeds with the same `user_id ≠ 0` and `timestamp > feed timestamp`, in any split. user 0 feeds are never matched with each other.
  - For **val rows**, compute global counts excluding the test snapshots of the 92 val∩test users.
  - Apply the same rule to anything built from histories and used on train rows: ALS/i2v item vectors, user-level aggregates, `x_lh_*`.
  - Covis `seq_*` already uses day cutoffs (`feats_v2/seq_/log_trans_train.log`). Check `clu_*`/`trd_*` the same way.
- **Why:**
  - C2 (92% reappearance) [E].
  - OTTO 7th built per-period matrices; H&M 5th's leaky w2v gave CV 0.0441 vs LB 0.0350 [V-rep:deep_otto_hm].
- **Gain:** unknown. est. +0.000-0.004 for train-fit GBDTs. It also removes optimism from the val numbers of any model that sees `it_hist_*`.
- **Cost:** CPU, about 1 h (polars join on (user, item) as in `src/later_history.py`).
- **Accept if:** in a 300K-feed sample, mean Δ(global − LUO) for positives is ≫ that for negatives, *and* `tr300k_none` retrained with the LUO features is not worse on val (paired).

### Highest expected gain

**2. Stage-2 val-CV stacker over train-only NN scores plus the full feature bank, then blend with the fine-tuned NN.**
- **What:** LightGBM `lambdarank` (truncation_level 1000, `lambdarank_norm` default, lr 0.02, 31-63 leaves, min_data 200, fixed rounds chosen by CV), **plus** XGB `binary:logistic` (GPU N). Same 5 GroupKFold(user) folds as `nn/finetune.py`; user 0 is split by feed.
- **Inputs:**
  - (a) train-only NN scores: ensemble and 2-3 diverse members, as raw value, per-feed z, rank/M, and gap to feed max. No nested OOF is needed, because they never saw val.
  - (b) train-fit GBDT scores (`cb_tr300k`, `lgb_tr300k`, `xgb_tr1m`) with the same transforms.
  - (c) pairwise differences of the z-scores.
  - (d) 150-300 handcrafted features chosen by family ablation (item 25) from the 655 in `data/feats/val`. Includes `clu_*`, `seq_cv_*`, `life_*`, `trd_*`, `ctx_*`, ALS (item 3).
  - (e) meta features: M, H, user-in-train flag, user-0 flag.
- Final score = feed-z mean of {stacker, fine-tuned NN ensemble}, weight from OOF with ≤3 weights.
- **Why:**
  - RecSys'24 1st: stage-2 LightGBM over stage-1 predictions with z/min-max/rank/diff features; ensemble +0.006 AUC [V-rep:deep_recsys2024].
  - NVIDIA RecSys'21: stage-1 on old data, stage-2 on new period; all 6 stage-1 models kept by greedy add [V-rep:deep_recsys_engagement].
  - ARGUS: a sequence model as a *GBDT feature* +2.32% vs standalone −5.33% [V-rep:deep_neural_rankers].
  - Ours: `init_score` from the train-fit LGB +0.0018 over val-CV base (0.3683 vs 0.3665) [E].
- **Gain:** est. +0.002-0.006 over 0.381. The fine-tuned NN only sees 27 extra features; the stacker sees the whole bank.
- **Cost:** XGB GPU about 3-5 min per 5-fold run. LightGBM at 24 threads about 20-40 min.
- **Pitfalls:**
  - Rank features within feeds need a tie-break.
  - Raw split-count features must be per-10K-feeds normalised (test is 2× val).
  - Do not feed `x_lh_*` as a feature: its coverage differs between val (0.76%) and test (0.36%). Apply it post hoc (item 8).

**3. ALS item-item profile features: the best collaborative signal, not yet in `data/feats`.**
- **What:** reuse `eda/collab_stats/out/alsgpu_f128_i12_evt_a5_{val,test}.npz` (GPU CG-ALS: f=128, 12 iterations, confidence c = 1 + 5·w, w = 1 + #distinct event types of the (user, item) pair + 3·[train positive]). For candidate c in feed f, with valid history events j ordered by `hist_hpos`:
  - `als_prof5 = cos(V[c], Σ_j e^{−j/5} V[h_j])`;
  - `als_last10`, `als_last20` (normalised mean of the last k);
  - `als_samecat` (profile restricted to `hist_category_id == cand_category_id`, τ=10);
  - `als_udot = U[user]·V[c]`;
  - `als_has` = item in the ALS vocabulary;
  - within-feed z/rank/gap of each. NaN when there is no vector.
- Rebuild once with the val∩test users' test snapshots removed for val (item 1). Do not use these features on train rows unless rebuilt from an earliest-snapshot corpus.
- **Why:**
  - Alone 0.3310 (+0.042 over random). f1 + ALS profile is +0.0077 ± 0.0014 over f1. It is the only CF signal that adds clearly to f1 [E `eda_collab_stats §5-6`].
  - It survives coverage decay better than train statistics: ALS item coverage val 62.8%, private 54.9%.
  - H&M 3rd: BPR user-item was the largest single feature [V-rep:deep_otto_hm].
  - The provided item embeddings are probably content-only, with no item-ID memorisation [V-rep:deep_seq_pretrain_cf, habr 1004694]. ALS adds the missing ID-collaborative component.
- **Gain:** est. +0.001-0.004 inside the stacker or NN.
- **Cost:** about 30 min CPU for features. Retraining ALS: about 7 min on GPU N.

**4. NN: use the random-like train feeds as gold data and as a second holdout.**
- **What:**
  - (a) Stage-2 continuation of the train-only NN on feeds with `p_random > 0.9 & M ≥ 50` (about 90K) from `eda/labels_position/out/04g_train_prandom.npz`, mixed 50/50 with ordinary feeds (as `s2rand_aux_sim`: lr 2e-4, about 1000 steps).
  - (b) **Select checkpoints for every train-stage run on a fixed 30K-feed random-like holdout** (never trained on), not on val. The val OOF then stops being optimistic, and val stays a clean training/OOF set.
- **Why:**
  - 50K random-like feeds beat 50K M-matched production feeds by +0.0055-0.0066 [E `eda_labels §6.1` f1/f4].
  - Their positive rate and M profile match val [E].
  - `s2rand_aux_sim` 0.38004 vs its init 0.37848 (val-selected, so optimistic) [E `nn/logs/s2rand_aux_sim.json`].
  - The Expedia host warned that official random flags were noisy [V-rep], so use soft or strict thresholds.
- **Gain:** est. +0.001-0.002, plus honest model selection.
- **Cost:** GPU N, about 3 min per stage-2 run.
- **Accept if:** the random-like holdout NDCG tracks val NDCG across ≥5 checkpoints (Spearman > 0.8). Otherwise keep val selection.

**5. NN ensemble breadth with snapshot/EMA averaging.**
- **What:**
  - 5 seeds × {`nopos_aux_sim`, `aux_sim_b3`, `aux_sim_h8`, s2rand variant}.
  - Each: train-only (EMA decay 0.999 during pretraining) → 5-fold val fine-tune with snapshot mean of epochs 1-4 → test = mean of fold models.
  - Combine by per-feed z-mean, hierarchically: within a variant, then across variants.
- **Why:**
  - Snapshot mean +0.0017 [E].
  - 2-seed ensemble +0.0002 over best single [E `nn_report`], so seeds of one variant add little; variants should add more.
  - DASALC-ens +1.26 NDCG@5 [V-rep:deep_neural_rankers]; EMA [V] (§1 #10).
- **Gain:** est. +0.001-0.003.
- **Cost:** about 15-25 min of GPU N per pretrain plus about 5 min per fine-tune. Roughly 6-8 GPU hours in total, run sequentially.

**6. NN: attention-logit bias from typed cross features plus a count-preserving channel.**
- **What:** in each candidate→history cross-attention head, add a learned scalar bias:
  `β_type[hist_event_type] + β_src[hist_event_source] + β_rec[bucket(hist_hpos)] + β_cat·1[hist_category_id == cand_category_id]`.
  This comes on top of the existing `--sim_bias` cosine term. Add one non-softmax pooling, `Σ_j SiLU(q·k_j/√d + bias_j) v_j / 256`, concatenated to the candidate token.
- **Why:**
  - TWIN: removing the cross-feature bias "leads to a significant drop in GAUC" [V-rep:deep_neural_rankers].
  - HSTU: SiLU over softmax −1.7% NE, needed to keep "intensity" [V-rep].
  - Our sim_bias, a special case, gave +0.002-0.003 [E].
  - The learned recency curve is steep (bias −2 at hpos 4) and the event-type biases are large (type 5 +0.34, type 7 −2.52) [E `eda_embeddings §3.2`].
- **Gain:** est. +0.001-0.003.
- **Cost:** a code change plus 2 seeds at about 20 min on GPU N.

**7. NN fine-tune extra branch: add the new feature families.**
- **What:** extend `finetune.py --extra item_safe` (27 features) with the top 30-60 stacker features that the NN cannot compute from its own inputs:
  - ALS (item 3);
  - `seq_cv_*` covis;
  - `clu_*` (cluster-level TE/PMI);
  - `life_*` (item first/last-seen age from feed timestamps);
  - `it_hist_et1` (leak-safe version);
  - `trd_*` normalised.
- Keep the zero-init branch, branch-LR ×3-10 and snapshot mean.
- **Why:** the item_safe branch alone gave +0.003-0.005 [E]. Only val/test-only quantities need this route.
- **Gain:** est. +0.001-0.003.
- **Cost:** about 5-7 min per fine-tune.
- **Pitfall:** any count computed inside a split must be divided by that split's feed count, as `nn_report` already does.

**8. Later-history boost as a post-hoc reorder, kept as a separate final candidate.**
- **What:** for test, `lh_et1/lh_any` from `src/later_history.py` with `sources=['test']` and `user_id > 0`. Final score = `feed_z(model) + 100·lh_any` (`apply_boost`). For val, report it with `sources=['val']` only, which is the honest analogue.
- **Why:** 93% target rate (C1).
- **Gain:** about +0.0017 private, +0.0047 public, +0.0025 on all of test [E].
- **Cost:** minutes.
- **Risk:** organiser discretion (§1 #1). It also inflates public relative to private by about 2.8×. Never use public LB deltas from runs that include it to make modelling decisions.

### Medium expected gain

**9. Time-consistent construction for train-derived ID statistics (protects Private).**
- **What:** for features built from train *candidates or labels*, the val copy is the training data of every val-fit model while the test copy is what gets scored, so their staleness should match. Features in scope: `it_tr_*`, `tte_*`, `it_cand_cnt_train`, `clu_ic*`, `u_tr_*`.
  - **Option A (safer):** drop raw train-candidate counts and CTRs and keep train-*history* counts. Those are robust to gap: +0.0032 → +0.0030 at a 15-day gap [E `eda_shift §4.6`].
  - **Option B:** build the val copy from train ≤ Sep 14 MSK. That gives an 8-10 day gap and about 48% coverage, similar to Private's 9-10 days and 52.7%.
- **Validation:** train the stacker on val day Sep 22 with standard features. Score Sep 24 with standard features and with features at an earlier cutoff, and adopt option A/B if the standard model degrades more.
- **Why:**
  - Train-candidate coverage: val 65% → Private 52.7%.
  - Train-candidate counts turn negative (−0.0014 to −0.0029) at gaps ≥ 9 days [E `eda_shift §4.2, §4.6`].
- **Gain:** about 0 on val; protects Private by est. 0.000-0.003.
- **Cost:** about 1-2 h CPU.

**10. Per-feature numerical embeddings for the NN dense block.**
- **What:** replace `zscore → clamp → MLP` for the 61 GPU dense features and f1..f5 with:
  - robust scaling `x ↦ s(x−median)`, s = 1/IQR, then smooth clip `x/√(1+(x/3)²)`;
  - then PLE with 16 train-quantile bins (or PLR with 32 frequencies, σ∈{0.01, 0.1, 1}), d=8 per feature;
  - keep the raw scalar alongside.
- **Why:**
  - MLP average rank 8.5 → 3.0 with PLR; TabM† with PLE ranks 1.7 [V-rep:deep_neural_rankers].
  - Avito's production NN uses 128 interpolated quantile bins [V-rep:deep_ru, habr 1090584].
  - f1..f5 are already "normalised interpolated bins".
- **Gain:** est. 0.000-0.003.
- **Cost:** a code change plus 2 runs on GPU N.

**11. Loss refinements for the NN.**
- **What:** per-feed `SoftmaxCE(s, y/Σy) + λ·mean BCE(s, y)` with λ ∈ {0.1, 0.3}, averaged per feed. Variant: one softmax per positive, with the other positives masked out.
- **Why:** Sigmoid+Softmax CE beat Softmax alone on all 3 binarised LTR sets (Web30K 0.4665 vs 0.4578 NDCG@10) [V-rep:deep_neural_rankers, arXiv 2211.01494].
- **Gain:** est. 0.000-0.002.
- **Cost:** trivial.

**12. Category-intent auxiliary signal.**
- **What:**
  - (a) NN aux head: from [user token, pooled history], predict the distribution over 54 categories of the feed's positives. Target = multi-hot of `cand_category_id[cand_target==1]` normalised. CE weight 0.1. Expose `log P(cand_category)` to the head.
  - (b) GBDT feature: the same from a small model trained on train feeds, with features from the feed's own history only (leak-safe), giving `cat_intent_logp` plus within-feed rank.
- **Why:**
  - Oracle knowledge of the positive's category gives 0.479 [E `eda_labels §5.3`].
  - Avito's production blender uses exactly this kind of interest model (+1% contacts) [V-rep:research_competition_avito].
  - NVIDIA KDD'23 used next-item attribute probabilities as features [V-rep:deep_otto_hm].
  - But f1/f3 already encode user×category affinity (within-feed ρ 0.68/0.78) [E].
- **Gain:** est. 0.000-0.002.
- **Cost:** about 1 h dev plus 20 min GPU.

**13. Soft pseudo-labels on test for the final NN fine-tune.**
- **What:** target per test feed = softmax(z_teacher/T), T ∈ {0.5, 1}. Teacher = ensemble of train-only models plus fold models that did not see val fold k (fold-safe). Fine-tune on val folds ∪ all 28.6K test feeds, with weight w_test ∈ {0.3, 1}. Evaluate on the val OOF.
- **Why:**
  - RecSys'24 1st used soft ensemble pseudo-labels in its final neural model [V-rep:deep_recsys2024]; gain not reported.
  - The NVIDIA playbook recommends soft labels and fold-safe pseudo-label sets [V-rep:deep_recsys_engagement].
- **Gain:** est. 0.000-0.003. It can also hurt by reinforcing teacher errors.
- **Cost:** about 10 min on GPU N.

**14. Pretraining regularisers for the train-stage NN.**
- **What:**
  - (a) Cap feeds per user per epoch at ≤3, re-sampled each epoch. Treat `user_id 0` as one user per feed.
  - (b) History dropout: for 30% of feeds drop each history event with p=0.3, always keeping hpos 0. Recompute dense history features after masking.
- **Why:**
  - Both are in RecSys'24 1st's best Transformer (v1174) [V-rep:deep_recsys2024 code].
  - Train is per-impression, with heavy users over-represented (H=256 in 42% vs 29%); val/test have about 1.0 feed per user [E].
- **Gain:** est. 0.000-0.002.
- **Cost:** code plus 1 run.

**15. Co-visitation extensions (stacker features).**
- **What:** on the existing exact, query-driven `seq_cv` code:
  - (a) event-type-pair matrices over et groups {1}, {2,3,4}, {5..10};
  - (b) forward-only pairs where b is the first occurrence of that item in the sequence ("new item", like our candidates), with windows {5, 20, 64};
  - (c) normalisations P(b|a), C/√(n_a n_b), lift;
  - (d) aggregation over the feed history: sum, max, top-3 mean, plus hpos half-lives {3, 10, 50};
  - (e) category→category fallback for the 89% of rows with zero item co-occurrence.
- Train rows need cutoffs or leave-user-out (item 1).
- **Why:**
  - OTTO 1st: removing covis cost 0.0052 [V-rep:deep_otto_hm].
  - Existing `cv_invd20` + f1 gives +0.0153 [E `history_leak/05_covis.log`].
- **Gain:** est. 0.000-0.002 on top of the NN, which already models embedding similarity.
- **Cost:** CPU 2-4 h at ≤24 threads.

**16. Train-fit GBDTs as stage-1 diversity for the stacker (not as blend members).**
- **What:**
  - XGB `rank:ndcg` with `lambdarank_pair_method="topk"`, `lambdarank_num_pair_per_sample=300`, `lambdarank_score_normalization=False` (or `binary:logistic` with negatives downsampled to 10-20%, per §1 #6).
  - Data: 1-1.5M feeds per seed, different feed samples per seed, random-like feeds over-sampled ×3.
  - Leaky features excluded (`it_tr_*`, `u_tr_*`, non-LUO `it_hist_*`).
  - Optional training-only `pos` and `rkend = (M−1) − rank(cand_pos)`, fixed at inference to (20, 50) and tuned on the holdout. Or LightGBM `Dataset(position=id)` with a crossed id slot-bucket × rkend-bucket, on CPU only, not combined with `bagging_by_query`.
- **Why:** XGB topk-300 + no score norm is the best GBDT setting [E `deep_gbdt §2.1`]; C4.
- **Gain:** est. 0.000-0.002 via the stacker. As a direct blend it adds +0.0005 [E\*].
- **Cost:** XGB GPU about 10 min per 1M feeds. CatBoost GPU needs `gpu_ram_part=0.25`.

**17. Learnable item-ID residual in the NN (the cheap form of the "hybrid sequence model").**
- **What:**
  - Item vector = `W·norm(emb128) + E_id[item]` for candidates and history items with ≥2 occurrences in train+val+test histories/candidates (about 40M ids). Use a hashed table of 2^22-2^24 rows × 32-64 dims to fit GPU N.
  - **ID-dropout 0.5** (§1 #5).
  - Train at most 1 epoch over train (one-epoch phenomenon [V-rep:deep_neural_rankers, arXiv 2209.06053]).
  - Freeze it during val fine-tune.
- **Why:**
  - The provided embeddings carry no item-level target signal (kNN R² < 0) [E `eda_embeddings §4`].
  - UniSRec: text+ID beats text-only [V-rep:deep_seq_pretrain_cf].
  - ALS (ID-CF) adds +0.0077 over f1 [E].
- **Gain:** est. 0.000-0.004. Risk of overfitting; check the val delta on candidates with and without ID coverage.
- **Cost:** a code change plus 2 runs.

**18. In-split and cross-split transductive signals for val/test.**
- **What:**
  - (a) In-split history count `n_hist_split[item]/n_feeds_split×1e4` for val and test separately. +0.0014-0.0020 [E `eda_shift §4.5`].
  - (b) Val labels for test candidates that were val candidates (12.7% of test rows): Bayesian item CTR `(pos+0.0158·20)/(n+20)` from val, plus the "(user, item) shown before and positive" flag (3.82% vs 1.58%) [E `eda_collab §8`].
  - Train/evaluate (b) with leave-one-feed-out inside val.
- **Gain:** est. 0.000-0.002.
- **Cost:** about 1 h CPU.

### Low expected gain / protocol items (do them, but they will not move the score much)

**19. HSM pretraining (full sequence model on all histories).**
- **What:** causal transformer over the earliest-snapshot train corpus plus val/test snapshots, reversed to oldest-first.
  - Token = frozen emb → MLP + ID residual (item 17) + emb(et/src/cat/hpos).
  - Objectives: dense next-*new*-item sampled softmax (mixed in-batch + uniform negatives, logQ) + next event_type head.
  - Export the score as a stacker feature and use it to initialise the NN history encoder.
- **Why:**
  - ARGUS: pretraining doubles the feature-mode gain (+1.17% → +2.32%) [V-rep].
  - PinnerFormer: dense all-action beats next-action by 23% Recall@10 [V-rep].
  - But the provided embeddings are probably already outputs of Avito's own transformer two-tower trained on far more data [V-rep:habr 1004694].
- **Gain:** est. +0.000-0.006, wide uncertainty.
- **Cost:** 1-2 days of dev plus several GPU 5 hours. **Only after items 1-8.**

**20. Tuned item2vec** (negative-sampling exponent α=−0.5, sub-sampling t=1e-5, window 3/7, 16+ epochs).
- **Why:** tuning raised HR@10 from 11.16 to 23.75 on 30Music [V-rep:deep_seq_pretrain_cf]. Ours adds +0.0035 over f1 and nothing on top of ALS [E].
- **Gain:** est. 0.000-0.001 (stack diversity only). **Cost:** about 1 h on GPU N.

**21. Contextual-normalisation extensions** (x/max, x−median, feed std/skew/entropy, n distinct categories) for the *new* scores (NN, ALS, covis).
- **Why:** FeatureSalad / BlackPearl [V-rep:deep_recsys2024]. The f-feature versions already exist (`ctx_*`).
- **Gain:** est. 0.000-0.001. **Cost:** minutes.

**22. Blending discipline** (applies to everything above).
- Per-feed z-score every model before averaging; never average raw scores across libraries (raw −0.0013 vs feed-z [E]).
- Never rank-average without a continuous tie-break: −0.0067 / −0.0085 [E].
- Weight hierarchically: within a family, then across families. Use at most a handful of free weights. A weight search on 14.4K feeds overfits at the 0.001 level [V-rep:deep_gbdt; TabArena caveat V-rep].
- Final submitted score = blend + 1e-6·z(f1) as a tie-break, and check that pessimistic NDCG equals expected NDCG.

**23. Final-fit protocol.**
- Fix all folds once: GroupKFold(user_id), with user 0 split by feed.
- NN test prediction = mean of the 5 fold models × seeds (as `finetune.py`).
- Stacker: refit on all val with rounds = mean CV-best × 1.1 (H&M 1st used about 1.1× [V-rep:deep_otto_hm]), *and* average with the fold models. Take the better on paired OOF-proxy.
- Train-only models: retrain from the best config, with checkpoints selected on the random-like holdout (item 4).

**24. Validation protocol.**
- Decide only on val, with paired per-feed SE. With one seed, |Δ| < 0.002 is noise [E].
- Second check: val Tue → Thu (`timestamp` day split).
- Third check: the random-like holdout for train-stage models.
- Expect LB ≈ val − 0.006.
- Public is weekend-only, 8.6K feeds (SE ≈ 0.003), and carries 2.8× the leak effect, so do not tune on it [E `eda_shift §5`].

**25. Feature selection for the stacker.**
- Group the 655 features into about 20 families (f-raw, f-rel, emb-sim, seq/covis, clu, ctx, life, trd, cal, tte, it_hist, ALS, ...).
- Run family-drop ablations in the stacker with 2 seeds. Keep a family if paired Δ > 2 SE.
- Drop single features with adversarial val-vs-test AUC > 0.6 (`feats_v2/*/drift*.json` exist).
- Drop the `user_emb all-zero` flag: prevalence 1.42% → 0.28% [E].
- Drop raw hour: logging outages distort it [E].
- Use within-feed permutation importance, never split gain [K practice, V-rep:deep_gbdt §7].

### Expected cumulative effect (est.)
- Items 1-8: +0.006 to +0.015 val. Plus the leak boost on test: +0.0017 private / +0.0047 public.
- Items 9-18: +0.002 to +0.008, with lower confidence.
- If realised, that reaches the 0.388-0.392 public band.

---

## 4. Techniques that do NOT apply to our data (and why)

| Technique (source report) | Why it does not apply |
|---|---|
| Time-since-last-event, session splitting by time gaps, minute-window co-visitation, time-decay weights `1/log(2+Δmin)` (OTTO, Theo; deep_otto_hm), TWIN time-gap cross feature, Avito "event age MLP over 2 months", RecSys'25 time-delta loss, Layer 6 "time since last engagement" | **History has no timestamps**, only `hist_hpos`. Use hpos as recency; it already decays steeply. Only feeds have `timestamp`. |
| Item publish time / "time since publication" (RecSys'24 #1 feature) | No item timestamp. `item_id` is not time-ordered (Spearman 0.00002) [E]. Use first-seen proxies only (`life_*`, already built). |
| RecSys'24 ±5 min windowed exposure counts (L2/L3); Expedia one-way counts over train+test; Synerise same-tweet counts | Val/test items are singletons inside a split (91%/86% in one feed); ±1 h windows cover ≤0.6% of candidates and add nothing [E `deep_recsys2024 §7.1`]. |
| PRM / DLCM / Seq2Slate / position embeddings over candidates; DPIN (position at serving) | Val/test order is random. Train order is production order (a bias leak). PRM's gain is mostly its PE [V-rep:deep_neural_rankers]. |
| RankFormer listwide loss, PEAR list-CLS, `rank_xendcg` "uniform" gradient | Need lists without positives; every feed in train and val has ≥1 positive [E]. |
| RandPair / swap interventions, intervention harvesting (Agarwal'19), LinkedIn IV from A/B buckets | Need interventions or ranker-bucket logs we do not have. Detected-random train feeds replace them. |
| User-level TE/CE and user CTR as ranking signals (RecSys'20/21 top features) | Constant within a feed. about 67% of test feeds have no train history for the user, and val/test have about 1.0 feed per user [E]. Usable only as interaction context in trees or NN. |
| `is_random=1` mixture trick from Expedia as a *prediction-time switch* | Our test is 100% random. The only form that transfers is "train on random rows" (item 4). |
| KDD Cup 2020 popularity debiasing (inverse-popularity weights, rare-item boost) | It addresses exposure fairness, not position; the metric is per-feed NDCG on random order [V-rep:deep_position_bias_random]. |
| EASE / SLIM full closed form; LightGCN / TwHIN | 40M+ items; TwHIN added 0 next to a sequence encoder [V-rep:deep_seq_pretrain_cf]. |
| Text/BERT features, image features | Not in the data. |
| TabPFN | Validated to about 50K rows and not a ranker [S]. |
| Per-user-group models (RecSys'21 fairness metric) | Different metric. |

---

## 5. Do-not-do list (each item has measured or verified evidence)

1. **Do not feed real `pos`, rank-from-end, or anything derived from train order at inference.** Do not fix pos at 0 (−0.005 to −0.016) [E]. Do not use position embeddings over candidates.
2. **Do not use full IPS / IPW weighting in GBDTs.** XGB `lambdarank_unbiased`, bias_norm=0: −0.0022, z=−2.2 [E]. Pointwise IPS was significantly worse than naive in Baidu-ULTR [V-rep].
3. **Do not use `rank_xendcg` with label 1** (−0.0047, z=−3.9) [E].
4. **Do not keep these defaults or settings:**
   - XGB `topk` k=32: −0.0052 [E];
   - XGB `lambdarank_score_normalization=True` with pairwise: stops at about 10 iterations [E];
   - LightGBM `lambdarank_truncation_level=30` and `eval_at` 1..5;
   - per-feed 1/M weighting (−0.0026) [E].
5. **Do not average raw scores across libraries or rank-average without a tie-break.** Raw mean −0.0013 vs feed-z; rank mean −0.0067 / −0.0085 [E]. Never submit tie-heavy scores: a constant gives 0.1986, below random 0.289 [E].
6. **Do not build history aggregates from all snapshots and then use them on train rows.** That includes `it_hist_*`, ALS/i2v item vectors, user aggregates and covis. 92% of positives reappear in the next snapshot (C2). Do not build val features from test histories of val∩test users.
7. **Do not treat `user_id = 0` as a user**: pooled, 123,904 train feeds [E]. It corrupted the 27.6% leak statistic (C1).
8. **Do not use these as features:**
   - in-split candidate frequency (−0.0018) [E];
   - time-windowed transductive popularity [E];
   - shortened or recency-decayed train-statistic windows (not better than all 14 days) [E];
   - raw train CTR (position-confounded; −0.0014 to −0.0044 in blends) [E].
9. **Do not let raw in-split counts computed on val score test** without per-10K-feed normalisation (test has 2× the feeds) [E]. The same goes for the `user_emb all-zero` flag (5× prevalence shift) and raw hour of day (logging outages).
10. **Do not trust a naive train-fit model on f-features.** It scores below f1 alone (0.3250-0.3303 vs 0.3365) because of the f4 sign flip [E]. Always check train-fit models against f1 and the val-CV numbers.
11. **Do not invest in these NN directions:**
    - bigger d (d=256: 0.37271 vs 0.37385) [E];
    - more set-attention blocks (3 blocks no gain; noset neutral) [E];
    - HSTU/GR-scale models, BERT4Rec (beaten by SASRec at equal loss [V-rep]);
    - label smoothing (hurts AUROC-like metrics [V-rep]);
    - Gaussian feature noise on MLP/GBDT (hurts [V-rep]);
    - LambdaRank as the primary NN loss (worst in DASALC [V-rep]).
12. **Do not select NN checkpoints on logloss** (official baseline). Select on exact full NDCG, ideally on the random-like holdout. Do not run bf16 scoring heads: quantised logits created ties (0.37047 → 0.37361 after the fp32 fix) [E].
13. **Do not tune on public LB or chase single-digit-thousandth public gains.** Public is 8.6K feeds, weekend-only, SE ≈ 0.003, and inflated by the leak. Val is closer to Private on M, category and weekday [E].
14. **Do not break the compute rules:**
    - never touch some GPUs;
    - always `CUDA_VISIBLE_DEVICES=N` (XGBoost `device=cuda:N` alone opened a context on GPU N; CatBoost opened contexts on all GPUs) [V-rep:deep_gbdt];
    - CatBoost `gpu_ram_part ≤ 0.3`;
    - ≤24 CPU threads in total: LightGBM `num_threads` ≤ 24 minus everything else running. Check for orphaned `run_lgb.sh` jobs from the earlier GBDT attempt first [V-rep:deep_gbdt §0].
15. **Do not expect these (refuted on our data):**
    - debiasing as "the biggest lever" (C11);
    - set attention +0.6% (C7);
    - val fine-tuning by itself as a large gain (C6);
    - item TE as a core feature (C9);
    - a stacking or "objective choice" miracle (all objectives within ±0.001 [E]).

---

## 6. Recommended execution order (compute-aware)

1. Day 0 (CPU):
   - item 1 (leak-safe `it_hist`; re-check `clu`/`trd`/`x_lh` provenance);
   - item 3 (ALS features from existing factors);
   - item 25 grouping;
   - item 9 option A.
2. Day 0-1 (GPU N, sequential):
   - item 4 (random-like holdout + stage-2);
   - item 6 (attention bias);
   - item 11 (loss);
   - then item 5 (seeds × variants) with the winners.
3. Day 1:
   - item 2 (stacker on train-only NN + train-fit GBDT scores + selected bank);
   - item 7 (extra branch);
   - blend (item 22).
4. Day 2+: items 12-18. Item 19 only if time remains.
5. Final selection, two candidates if the platform allows two (not stated on the page):
   - (A) best blend + leak boost (item 8);
   - (B) the same blend without the boost.

   Both must be tie-free and checked with `src/metrics.py` pessimistic = expected on val.

---

## 7. Sources read in this session (verification)

- Competition page: https://ods.ai/competitions/avitotechrecsys2026
- Dataset page: https://ods.ai/competitions/avitotechrecsys2026/dataset
- Rules text: local copy `/tmp/scratchpad/rules.txt` (rules of "Avito ML Cup 2026", §5.3, §7.4, §8.6)
- Expedia 5th place: https://arxiv.org/pdf/1311.7679 (text extracted with pdftotext)
- DropoutNet: https://papers.nips.cc/paper_files/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf
- Facebook ads, negative downsampling: https://quinonero.net/Publications/predicting-clicks-facebook.pdf
- XGBoost GPU memory: https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/doc/gpu/index.rst
- LightGBM bin storage: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/io/bin.cpp
- Chris Deotte OTTO repo, notebook `infer/candidates_LB/model-11-1-LB.ipynb`: https://github.com/cdeotte/Kaggle-OTTO-Comp
- EMA of weights: https://arxiv.org/abs/2411.18704
- PAL: only a search result (https://unpaywall.org/10.1145%2F3298689.3347033) [S]. Owen Zhang's Expedia slides were not found.

Local evidence used:
- `preds/*.npy`, rescored with `src/metrics.py`;
- `logs/tr300k_*.log`, `logs/stack1.log`;
- `nn/logs/*.json`;
- `/tmp/scratchpad/exp/res/*.json`;
- `eda/history_leak/03_leak.log`, `eda/collab_stats/out_05b.log`, `eda/collab_stats/out_05c.log`;
- `eda/labels_position/out/04g_full_posterior.log`;
- `src/features.py` (`global_item_stats`), `src/later_history.py`, `src/train_lgb.py` (`LEAKY_ON_TRAIN`).

---

## Critic additions (completeness review, 2026-11-11)

**Scope.**
- Read in full: this playbook, `nn_report.md`, `eda_shift_overlap.md`. Read the TL;DRs of the other three EDA reports. Grepped the eight deep reports and three research reports for every topic in the review brief.
- Ran 8 small val/test checks on CPU (at most 8 threads). Scripts are in `/tmp/scratchpad/`: `roworder.py`, `roworder2.py`, `ownhist.py`, `histlink.py`, `futpop2.py`, `permseg2.py`, `catbias.py`, `catrank.py`, `ties.py`, `ties2.py`.
- New tag **[E-c]**: an experiment I ran in this session. All are val NDCG with the exact metric and paired per-feed SE.

### C0. Corrections to the playbook

1. **§4 wrongly dismisses the Expedia `is_random` switch.**
   - §4 says it transfers only as "train on random rows". `deep_position_bias_random` (TL;DR #4, G3) gave the form that does transfer: train on all traffic with a traffic-type input, then predict test with the exploration value.
   - Expedia 5th place used `random_bool` as a GBM feature [V-rep:deep_position_bias_random, arXiv 1311.7679].
   - Our data already shows the mechanism works. `pos` as a training feature, fixed at a deep constant at inference, gave +0.0099 over a plain train-fit LGB [E `eda_shift §7.4`]. Stage-2 on random-like feeds, the sequential version, gave +0.0012 [E `nn_report`].
   - The NN has no domain input: `grep -i "domain|is_random" nn/*.py` finds nothing. → **K2**.
2. **The NN loss silently up-weights long feeds by up to 4.8×.**
   - `nn/model.py::listwise_loss` returns `l[valid].mean()` per batch. `nn/data.py::batches` puts B = min(max_feeds, cand_budget // Mpad) feeds in a batch.
   - Train stage (32768 candidates / 512 feeds): B = 512 for Mpad ≤ 64, 256 at 128, 128 at 256, and 107 at 304. Fine-tune (8192 / 128): B goes from 128 down to 26.
   - So a feed with M≈300 gets about 4.8× the per-feed gradient weight of a feed with M ≤ 64. About half of val feeds have M > 64 (median 67).
   - The metric weights every feed equally. Nobody chose or tested this weighting. The GBDT result "1/M weighting −0.0026" is a different model and a different direction. → **K1**.
3. **The number of final selections is still unknown, and §6 has no rule for a single selection.**
   - Rules §7.4.2 say "выбора итогового решения" (singular). The FAQ says "выбора финальных решений" (plural) [V local copies `scratchpad/rules.txt`, `scratchpad/page_faq.html`]. Check the "My solutions" UI now, not on Dec 9.
   - If only one selection is allowed, my recommendation is to submit **without** the later-history boost unless the organisers confirm that using test histories this way is acceptable.
   - The boost is worth +0.0017 on private, about 0.6 of the private SE. The downside is a discretionary disqualification, because the flag reads test outcomes out of test histories.
4. **The public #1 lead is about the size of the leak** (this is inference, not evidence).
   - #1 is at 0.3935; #2-#10 are at 0.3884-0.3900, a gap of 0.0036-0.005. The later-history boost is worth +0.0047 on public and +0.0017 on private [E `eda_shift §6`].
   - If #1 uses it, the private leader band is about 0.388-0.391. Do not chase #1's public number.

### C1. Prioritized new items

| # | Item | Evidence type | Est. gain | Cost |
|---|---|---|---|---|
| K1 | Equal per-feed loss weight + NDCG-consistent K/IDCG(K) feed weight in the NN | code read + [V] theory | ±0.001 | ~30 min GPU N |
| K2 | Domain-token joint training (transferable `is_random` switch) | [E] analogues, [V-rep] | +0.000-0.002 | 1 h code + 20 min GPU |
| K3 | Label distillation from val-trained teacher into the train-stage student (KDCRec label channel) | [V-rep] | 0-0.002 | 10 min inference + 20 min GPU |
| K4 | Graded labels / event-type aux head from later snapshots (targets only) | [E] reappearance stats | 0-0.001 | 30 min CPU + 20 min GPU |
| K5 | `is_user0` regime input for the NN | [E-c] segment analysis | ≤ +0.0005 | trivial |
| K6 | Public LB as a *paired* check of val overfitting | arithmetic on SEs | protects selection | submissions only |
| K7 | Private-like reweighting of val OOF for the final choice | [E] adversarial AUCs | protects selection | CPU minutes |
| K8 | User_id-free history-fingerprint linking in the leak module | [E-c] | ≤ +0.0002 test | CPU minutes |
| K9 | Listwise/pairwise production-order distillation instead of MSE | `nn_report` "not tried" | 0-0.001 | one train run |
| K10 | Inference-only TTA over history length | [K] | 0-0.0005 | minutes |

**K1. Equal per-feed loss weight and NDCG-consistent K weighting in the NN (one fine-tune A/B, then the train stage).**
- **What:**
  - Normalise the listwise loss by a constant, `(l*w).sum() / max_feeds`, instead of the per-batch mean. Every feed then weighs the same whatever its M bucket.
  - Variants: (a) current; (b) equal; (c) equal × K/IDCG(K); optionally (d) weight ∝ √M.
  - Run first in `finetune.py`: same folds and seed, snapshot mean, about 6 min each. Carry the winner into one train-stage run.
- **Why K/IDCG(K):**
  - Ravikumar, Tewari & Yang (AISTATS 2011) prove that the NDCG-optimal ranking sorts by E[G(r_i)/IDCG(r)], not by E[G(r_i)]. They show that ListNet cross-entropy over softmax(labels) is NDCG-inconsistent. The consistent version uses the target G(r_j)/IDCG(r) [V https://proceedings.mlr.press/v15/ravikumar11a/ravikumar11a.pdf, Lemma 4, Prop. 7, §3.3].
  - Our target y/K is the L1-normalised version. Softmax CE with target y/IDCG(K) equals the current loss times a per-feed weight K/IDCG(K): 1.000 (K=1), 1.226 (K=2), 1.408 (K=3), 1.562 (K=4). That touches the 26% of feeds with K ≥ 2.
- **Accept if:** paired OOF Δ > 2 SE. The direction of the equal-weight change is genuinely unknown, because long feeds carry more negatives per step.

**K2. Domain-token joint training: the transferable `is_random` switch.**
- **What (NN):**
  - Add a feed-level token d ∈ {production, random-like, exploration}, embedded and added to every candidate token and to the head input.
  - Train stage: every train feed gets d = production. Feeds with M ≥ 50 are soft-mixed toward random-like by `p_random`.
  - Fine-tune: val folds get d = exploration, plus the 12K random-like feeds that `--add_random` already mixes in (+0.0004 [E `nn_report`]).
  - Predict test with d = exploration. As a check, also predict with d = random-like.
- **What (GBDT, item 16):** add `p_random` as a feature (NaN for M < 50), infer at 1.0, alongside pos/rkend fixed at their deep constants.
- **Why:** C0.1. The model keeps all 3.15M production feeds for representation and learns the traffic-dependent shifts explicitly (the f4 sign flip, position-confounded f5) instead of averaging over them.

**K3. Label distillation from the val-trained model into the train-stage model.**
- **What:**
  - Score all 3.15M train feeds with the fine-tuned fold ensemble, with the extra branch zeroed (its `item_safe` features are leaky on train rows).
  - Retrain the train stage with target (1−α)·y/K + α·softmax(z_teacher/T), with α ∈ {0.2, 0.4} and T = 1.
- **Why:** KDCRec distils a uniform-data model into the biased-data model through imputed labels [V-rep:deep_position_bias_random, https://dgliu.github.io/pubs/TKDE_2023_KDCRec.pdf]. It was verified there but never reached the playbook.
- **Evaluation trap:** the teacher saw every val label, so the student's val score, and any val fine-tune of the student, is contaminated.
  - Judge the student only on the random-like train holdout (item 4) and with a paired public submission (K6).
  - Per-fold teachers would make val clean but need 5 students: skip.
- **When:** only after K1/K2.

**K4. Graded labels and an event-type multi-task head from later snapshots (used as targets, never as features).**
- **What:** for train feeds whose user (≠ 0) has a later feed, join candidates to the next snapshot's history (the `src/later_history.py` join).
  - (a) Negatives that reappear there (0.40% of negatives [E `eda_shift §6`]) get gain g ∈ {0.2, 0.3} in the softmax target.
  - (b) An aux multi-label head predicts the event types (1..10) with which a candidate reappears. Per 100 positives: et1 79.9, et2-8 between 1 and 9.4 [E]. Weight 0.1, masked to the 32.8% of train feeds that have a later snapshot.
- **Why:** denser supervision drawn from the same users' follow-up behaviour. The current aux head uses only production order.
- **Leak safety:** used only as targets, so nothing leaks into val/test features.

**K5. `is_user0` regime input for the NN.** The data does not support history-length augmentation; this item replaces it.
- **Evidence [E-c `permseg2.py`]:**
  - Fine-tune gain over train-only (stage-1 ensembles): +0.0021 ± 0.0008 overall.
  - User-0 feeds with H < 32: **+0.0168 ± 0.0053** (358 feeds).
  - Non-user-0 feeds with H < 32: +0.0008 ± 0.0044.
  - User-0 feeds with H ≥ 32: +0.0003 ± 0.0035.
- The train-stage NN has no user-0 input. The fine-tune branch exposes it implicitly, because user train #feeds/CTR are extreme for the pooled id.
- **What:** add `is_user0` and `is_user0·log H` to the 61 dense features. Fold this into the K1/K2 runs.
- **Side note:** on non-user-0 feeds with H < 8, blending CatBoost helps (+0.0055 ± 0.0045, n.s.). Make sure the stacker sees H and is_user0, as item 2(e) already lists.

**K6. Use the public LB as a paired check of val overfitting, not for tuning.**
- Item 24 dismisses public because of SE 0.003, but that is the SE of a *level*. A *difference* between two correlated submissions has SE ≈ val paired SE × √(14423/8562) ≈ 0.0007-0.0017.
- **What:**
  - Make one submission per day, always without the leak boost, comparing NN-only vs stacker blend vs new variant.
  - Keep a (val OOF, public) table per submission.
  - Rule: if a val-trained component (stacker, family selection, weight search) gains Δ_val on val but less than Δ_val − 2·SE_pair on public, treat its val gain as optimistic.
- **Why:** items 2, 22 and 25 all choose on the same 14.4K feeds, and public is the only independent labelled check. Weekend composition shifts the level, but it mostly cancels in a paired delta.

**K7. Private-like reweighting of val OOF for the final choice.**
- **What:**
  - Fit a feed-level val-vs-private classifier on intrinsic plus train-coverage features (AUC 0.52 intrinsic, about 0.9 with coverage [E `eda_shift §5`]).
  - Weight val feeds by clipped p/(1−p).
  - For every final candidate, report the weighted OOF and the OOF on the lowest train-coverage tercile. Prefer candidates whose lead survives both.
- **Why:** item 9 protects private through the features. This protects the *selection*.

**K8. Leak module: user_id-free linking. Tiny and free; "with boost" candidate only.**
- **Method [E-c `histlink.py`]:** link feed pairs (A, then a later B) that share at least 5 rare history items (items in ≤ 30 feeds), covering ≥ 50% of A's rare last-64 history items.
- **Result:**
  - It recovers 205 pairs with the same non-zero user_id and finds **0** pairs between different non-zero user_ids. User_id linking is therefore essentially complete.
  - It adds 8 user-0 pairs: 7 extra test feeds and 17 flagged candidates. Val precision is 3/4 (one feed).
  - Worth ≤ +0.0002 on test.

**K9. Better production-order distillation.**
- `nn_report` lists "listwise distillation of the production order" as not tried.
- Replace the MSE on pos/(M−1) with a listwise CE toward softmax(−rank(pos)/τ), τ ∈ {5, 20}, which is top-heavy like the production objective. Alternatively use a pairwise logistic loss on sampled pos pairs. Keep the 1−p_random weight.
- Run it inside the K2 run as an ablation arm.

**K10. Inference-only TTA over history length.**
- Run `nn/predict.py` with max_hist ∈ {64, 128, 256} and take the per-feed z-mean. No retraining.
- Drop any member that alone is worse by more than 0.002.

### C2. Checked this session: do not pursue

| # | Question | Result [E-c] | Verdict |
|---|---|---|---|
| 1 | Does the val/test row order inside a feed carry signal (display order, pre-shuffle ranker order)? | NDCG by row order 0.2886 asc / 0.2880 desc vs random 0.289. For M ≥ 50, first/last-row positive rate 1.43% / 1.37% vs expected 1.45%. Adjacent-row same-category rate 0.217 val / 0.212 test. Raw first/last rates of 3.8% are only the small-feed effect (E[K/M] = 3.9%). | No order leak. |
| 2 | Do candidates appear in their own feed's history? | 0 of 1.39M val rows and 0 of 2.90M test rows. | The filter is exact on item_id. Only near-duplicate re-posts remain, and they are already used. |
| 3 | Is "future" in-split popularity (other users' later snapshots) better than past popularity? | Mid-rank blend with f1 at w=0.25: all +0.0010 ± 0.0006; past +0.0004 ± 0.0005; future +0.0007 ± 0.0005; future et1 +0.0008 ± 0.0005; future et1 hpos<16 +0.0002 ± 0.0003; test histories as a post-val source +0.0013 ± 0.0007. | Direction in time adds nothing. Keep only the plain normalised count. |
| 4 | Per-segment blend weights (calibration by M / H / user type) | Gain of NN-ft + 0.3·CB by M bucket: <20 +0.0036 ± 0.0018; 20-60 −0.0021 ± 0.0012; 60-150 +0.0007 ± 0.0008; 150+ +0.0008 ± 0.0007. Known users −0.0006 ± 0.0010, new users +0.0010 ± 0.0007. | All within about 2 SE, with sign flips. Keep global weights; let the stacker see M/H. |
| 5 | Per-category additive calibration of the fine-tuned NN OOF (fit out of fold, shrunk) | −0.0009 ± 0.0004 at every shrinkage. | Cross-category calibration is not a lever. |
| 6 | Does within-category production order (blender structure) give a new aux target? | At a fixed global pos bucket, the 4th+ item of its category converts 2-3× more than the 1st (pos 40-49: 1.67% vs 0.54%). That is category intensity (the blender gives the main categories more slots). | Already captured by feed category share. Nothing new. |
| 7 | Float32 ties in current outputs | `nn_final_test.npy` has 18 tied pairs in 12 feeds (`stk_valonly_es_test` 21). The written `subs/sub_v1*`, `sub_v2*` have 0 within-feed ties. | `make_sub` already tie-breaks. Keep the check in the final pipeline. |

### C3. Literature and competition coverage

- Analogue competitions are covered adequately (Expedia, Outbrain, trivago, Yandex PWS, RecSys'24, OTTO, H&M, VK, Avito).
- MIND (MS News 2020) is the only notable gap [K]: impression-level re-ranking with click history and 1-2 clicks per impression. Its lessons (candidate-aware user encoder, popularity-aware ranking) are already in the NN. No action.
- Uniform-data debiasing (KDCRec, AutoDebias, CausE) was verified in `deep_position_bias_random`, but only "fine-tune on val" reached the playbook. K2 and K3 are the cheap forms. AutoDebias (bi-level) stays low priority at 247M rows.
- New primary source this session: Ravikumar et al. 2011 (K1) [V].

### C4. Where these slot into §6

- **Day 0:**
  - K1 with K5 folded in (fine-tune A/B, about 30 min on GPU N);
  - K7 and K8 (CPU, minutes);
  - start K6 (one paired, boost-free submission per day).
- **Day 1:** K2 with K9 as an ablation arm (two train-stage runs of about 20 min on GPU N).
- **Day 2:** K3 and K4, only if K2 is positive on the random-like holdout. K10 at any time.
- **Est. combined:** K1+K2+K5+K9 give +0.000 to +0.003 val. K6 and K7 protect the private selection; they do not add score.
