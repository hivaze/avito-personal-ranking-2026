# W25 study: what the Avito ML Cup 2025 winners did, what transfers, and the build plan

Written 2026-11-12 23:46 UTC (study step of workflow wwt7b9e6r). CPU only, nothing trained, no process touched.
Files in this directory: `study.md` (this), `winner_hack_ipynb_dump.txt` (every cell of the 1st-place notebook with outputs),
`winner_autoencoders_model.py`, `winner_utils.py` (verbatim copies of the winner's code, cloned from GitHub into the scratchpad
and copied here for citation; third-party code, left out of the public release). Line numbers below ("dump:NNN") refer to `winner_hack_ipynb_dump.txt`.

Tags: **[V]** read in the primary source this session (URL or file given); **[E]** our own measured number (report cited);
**[M]** measured in this step (CPU, label-free unless stated); **(inference)** my reasoning.

Pre-declaration timestamp for the acceptance rules in section 4: `date -u` = **Thu Nov 12 23:45:59 UTC 2026**, before any W25
pipeline number was computed. The only numbers computed in this step are the descriptive coverage statistics in section 2.2
(label-free, except the "share of val positives among top-K items", which is descriptive and not an acceptance quantity).

---

## 0. Verdict in five lines

1. The 2025 winner (public 0.23147 Recall@40, #1 of 1,704) is a **retrieval** pipeline: three collaborative scorers over
   *item groups* ("nodes", tens of thousands of them) feed a CatBoost YetiRank re-ranker on ~200-300 candidates per user, with
   contact-ratio target statistics over the whole training window and 5-fold-averaged boosters. The task differs from ours in
   every structural respect (section 1.1): candidates are generated, not given; the target is a coarse group, not a listing; the
   metric is Recall@40 per user, not full-list NDCG per feed; there is no production position bias and no exploration/production split.
2. **We never built their pipeline faithfully, but we did test its information content.** What we tested: ALS/item2vec/co-visitation
   (same item-item information as EASE, different estimator) as scalar residuals on G3 (0 or negative, altapproach 2.1);
   label-based item/category/cluster target statistics (tte_*, itemprior: <= +0.0004 on G3; clu_ label-based TE 0 on a train-fed
   GBDT); CatBoost YetiRank members (cb26 e3: val 0.3820, +0.00015 at w 0.05 on G3). What we did NOT test: a linear autoencoder
   (EASE-DAN / RDLAE) over item groups with their normalisations, the contact-ratio-weighted scorer, the per-pair "history
   matrices pushed through the item-item matrix" features, and a GBDT trained 2025-winner-style on an in-distribution window with
   per-fold Optuna and fold averaging. None of these is a new information channel; all are new estimators of channels measured
   at 0 as residuals. So the honest prior that a faithful W25 reaches the ensemble bar (val >= ~0.393 at rho ~0.85) is low (~0.1).
3. **Item-level EASE is impossible and would not matter**: the top-50K items by history count cover 1.6% of val candidate rows (2.5% of
   positives); the top-5M cover 19% [M]. The faithful analogue of a 2025 "node" is an **item group**: our k-means clusters of the item
   embedding (K = 1024 / 4096, assignments already on disk for all splits) and the 53 categories. EASE/RDLAE over 4096 groups is a
   4096 x 4096 inversion (seconds), with full candidate coverage (99.98% of val/test rows have a cluster) [M].
4. **The plan (section 3)** builds exactly that: group-level EASE-DAN / RDLAE scorers with the winner's hyperparameters and
   normalisations, contact-ratio weighting from exploration-like train labels, their feature families translated key by key
   (time-respecting / OOF), and a CatBoost YetiRank trained on exploration-like train feeds with fold-averaged seeds, plus two
   data arms (production feeds with training-only position columns; val folds OOF). Budget ~12 GPU-hours on one of GPUs N and
   ~4 CPU-hours of feature building.
5. **Acceptance (section 4, pre-declared):** (a) single val NDCG with paired SE vs cb26 e3 (0.3820) and vs the h3 reference
   (0.3706); (b) nested user-grouped z-mix with G3 at w in {0.05, 0.1, 0.2, 0.3}, > 2 SE at the selected w and > 0 at w 0.05,
   holdout >= 0, no slice < -2 SE, pmix >= -1 SE; (c) a "teaching" result = one W25 feature family whose paired ablation inside the
   W25 GBDT is >= +0.005 val with holdout >= 0 and z >= 3, which is then queued as an NN train-stage input.

---

## 1. The 2025 competition and the 1st-place solution, read from the primary sources

### 1.1 Task definition (and how it differs from ours)

Sources: competition page https://ods.ai/competitions/avitotechmlchallenge2025 [V, fetched]; data schema from a participant's
README that copies the organiser's file list, <public repository, link removed> [V, cloned]; the organiser's
`baseline.ipynb` (same repo, authored by the 2026 task author) [V, read]; 1st-place code
<public repository, link removed> [V, cloned and read in full].

| | Avito ML Cup 2025 "Персональные рекомендации" | Ours (2026 "Personal ranking") |
|---|---|---|
| Target | "предсказание актуальных для пользователя **групп товаров**"; "Под "группой товаров" понимается не конкретное объявление, а общая категория"; "Откликом считается событие контакта" [V, ods page]. Positive = (cookie, node) pair with a contact event (`is_contact = 1`) in the next 14 days, **restricted to nodes the user has not interacted with** ("из тех, с которыми они еще не взаимодействовали") and, in the local split, to users and nodes seen in train (dump:21-41). | Binary per candidate listing in a given feed: any event of the (undisclosed) target set after the display. Already-viewed items are removed from the feed. |
| Candidates | **Generated by the solution** (any node); the winner re-ranks its own RDLAE top-300. | **Given**: 5-300 listings per feed (the production L2 top-300 minus viewed items, per deep_ru_competitions §2). |
| Metric | Recall@40 per user ("Метрика соревнования — Recall@40"), public/private 30/70 by users [V]. | Full-list NDCG per feed, pessimistic ties; public = first ~30% of test feeds by time. |
| Data | `clickstream.pq` (cookie, item, event, event_date, surface, platform, node), `events.pq` (event, is_contact), `cat_features.pq` (item, location, category, clean_params, node), `text_features.pq` (title int8[64]), `test_users.pq` [V, README]. One continuous clickstream with timestamps; eval = last 14 days ("14 дней в тесте"). 51 categories (dump:830), tens of thousands of nodes (the winner keeps the top 30K / 50K). | Per-feed snapshots: candidates (ids, category, f1..f5, int8[128] item embedding), the user's last <= 256 history events (item, category, event_type 1..10, event_source, rank), int8[128] user embedding; no timestamps inside histories; 3.15M production-ranked train feeds (position-biased, `pos` kept), 14.4K exploration val feeds, 28.6K exploration test feeds. |
| Bias | None of the production kind: train and test are both "what the user did next". | Train labels are position-biased production impressions; val/test are uniformly shuffled. |
| Organiser baseline | `implicit` ALS (60 factors, 10 iterations) on the user x node binary matrix, top-40 with seen nodes filtered, plus a popularity fallback (baseline.ipynb cells 17-21) [V]. | CatBoost YetiRank on 9 features (val 0.3533, public ~0.3461) and a candidate-set transformer [V, deep_ru §1.2]. |

Consequence (inference): in 2025 the whole difficulty was **candidate generation over a 2-month clickstream**; the GBDT was a
re-ranker of 200-300 collaborative candidates per user, and its features were the retrieval scores plus contact statistics. In our
task retrieval is done by Avito's L2; what remains is their *re-ranking stage*, applied to 100x more feeds, with a position-biased
training set they never had to deal with.

### 1.2 Local validation of the winner [V, dump:5-45]

`EVAL_DAYS_TRESHOLD = 14`: train = clickstream up to max_date - 14 d; eval = contact events after it, anti-joined on (cookie, node)
pairs seen in train, user and node restricted to those in train, deduplicated per (cookie, node). All scorers and all target
statistics are fitted on this train window only, so **every statistic is strictly earlier in time than every label by
construction**. There is no OOF machinery anywhere in the notebook; none was needed. For the final submission "все модели
тренируются заново на полном кликстриме (SasRec, SasRec over category, RDLAE, EASE_DAN) (кроме бустингов, бустинги усредняются по
фолдам, лучший сабмит был чисто на катбусте)" (dump:1674).

### 1.3 Scorers (exact settings)

**EASE-DAN** (`autoencoders/model.py`, class `EASE_DAN`; paper "Why is Normalization Necessary for Linear Recommenders?",
arXiv 2504.05805) [V]:
- Items: the **30,000 nodes with the most distinct users** (dump:82-83). Users: all. X = binary user x node (dump:92).
- Hyperparameters: `reg=10`, `drop_p=0.04977`, `beta=0.95209`, `alpha=0.95209` (the optimised-looking decimals suggest an Optuna
  run that is not in the repo; inference).
- Fit (verbatim logic): user-side normalisation `X_T = (X * user_counts^-beta).T`; `G = X_T X`; diagonal += `reg + drop_p/(1-drop_p) * item_counts`;
  `P = inv(G)`; `B = I - P / diag(P)` (the DLAE closed form); item-side rescaling `W = B * item_counts^{(1-alpha)} (row) * item_counts^{-(1-alpha)} (col)`;
  `diag(W) = 0`.
- Inference: `predictions = X_user @ W`, seen nodes set to -1000, top-300 (dump:94-101). Features for the ranker: `ease_score`, `ease_rank`.
- Local: **UNSEEN-RECALL@40 0.1735, @100 0.2820** (dump:113-114).

**RDLAE** (class `RDLAE`; paper "It's Enough: Relaxing Diagonal Constraints in Linear Autoencoders", arXiv 2305.12922) [V]:
- Items: the **50,000 nodes with the most contact events** (dump:152-153). `reg=500`, `drop_p=0.7`, `xi=0.1`.
- Fit: `G = X^T X`; `gamma = diag(G) * drop_p/(1-drop_p) + reg`; `G += diag(gamma)`; `P = inv(G)`; the relaxed-diagonal Lagrangian
  `lambda = ((1-xi)/diag(P) - gamma) * [1 - gamma*diag(P) > xi]`; `W = -P * (gamma + lambda)` (diagonal left non-zero).
- Fitted on the binary matrix (dump:266); **scored with a weighted input `X = binary + 2 * [contact]`** (dump:280), i.e. contacted
  nodes count 3x in the user vector at inference.
- **Score re-weighting by node contact ratio**: `predictions *= log1p(bayesian_ratio_C)` per node, where
  `bayesian_ratio_C = (mean(c0) * sum(c1)/sum(c0) + c1) / (mean(c0) + c0)` with c1 / c0 = number of distinct users who did /
  did not contact the node (dump:215-238; the "C" prior = mean non-contact count per node). Variants with C=100, a (1,2)-smoothed
  ratio and Gaussian noise injection were coded; the chosen column is `bayesian_ratio_C` (dump:144).
- **Per-pair "history matrices" pushed through W** (dump:159-172, 258-263, 276-288): for each (cookie, node) pair of the train window
  they build `cnt` (events), `is_contact` (contacts), `n_days_clicks`, `n_days_contacts`, `exp_pos = exp(-0.01 * rank of the node
  by last visit within the user)`, `exp_time = exp(0.05 * (last visit - train end) in days)` and `ratio_column` (the node ratio);
  for each such matrix F they compute `F_user @ W` (full W) and `F_user @ truncate(W^T, k=300)` (per-column top-300) at the
  candidate positions, giving 14 extra score features `score_<f>` and `score_20_<f>`. These are **event-type- and
  recency-weighted neighbourhood scores**: exactly the structure of our `seq_cv_*` and ALS-profile features, with a learned W instead
  of co-occurrence counts or factors.
- Local: **UNSEEN-RECALL@40 0.1994, @100 0.3136** (dump:295-296). RDLAE top-300 is the candidate set of the ranker.

**SASRec over nodes** (RePlay implementation, dump:308-700) [V]: top-30K nodes; `block_count=2, head_count=2, hidden_size=128,
dropout 0.3, max_seq_len=100, lr 1e-3, batch 128, loss_sample_count=4000, negatives_sharing=True`, 10 epochs, checkpoint on
recall@40; best **R@40 0.1629 at epoch 5** (dump:607), prediction = top-300 with seen items removed. Features: `sasrec_score`,
`sasrec_rank`.

**SASRec over categories** (dump:705-1070) [V]: the same model over the 51-category sequence (batch 512, 50 negatives, 5 epochs,
checkpoint on ndcg@5); joined to candidates by (cookie, category) as `sasrec_category_score`, `sasrec_category_rank`.

### 1.4 Ranker features (dump:1076-1305) [V]

All computed on the train window (no OOF; see 1.2). Nulls filled with 0 (`fill_null(0)`, dump:1297).

| family | features |
|---|---|
| user | `is_contact_0`, `is_contact_1` (event counts), `contact_ratio = c1/c0` (inf -> 0); `last_contact_category`, `last_seen_category`; most frequent `location` |
| user x category | `user_total_category_count`, `user_pos_category_count`, `user_category_contact_ratio`, `ratio_of_this_category_in_all_user_contact`, `ratio_user_category_click` |
| node | `node_contact_0/1`, `node_contact_ratio` (events); `node_contacts_0/1_last`, `node_contact_ratio_last` (counted once per (cookie, node, is_contact)) |
| category | the same four, by category |
| node x location | `location_in_node_count` (number of listings of the node in the user's top location) |
| crosses | `category_same_as_last_contact`, `category_same_as_last_seen`, `user_node_ratio_prob = node_ratio * user_ratio`, `user_category_ratio_prob` |
| scorers | RDLAE `score`, `rank_rd`, 14 `score_*`/`score_20_*`; EASE score/rank; SASRec node score/rank; SASRec category score/rank |
| categorical (CatBoost) | `location, category, node, last_contact_category, last_seen_category, category_same_as_last_contact, category_same_as_last_seen` (dump:1521) |

No text or image features were used (the `text_features.pq` load is commented out, dump:18).

### 1.5 Ranker training (dump:1306-1674) [V]

- Training rows (`utils.get_dataset`, defaults `k_recs=200, max_pos=100, max_neg=30`): RDLAE **top-200** per user, joined to the
  eval labels; **only users with >= 1 positive among their candidates**; per user **<= 100 positives and <= 30 random negatives**
  (sampled with seed 42); sorted by (cookie, score).
- `node` is set to null for nodes outside the 30K-node inference set "to not train on them" (dump:1317-1324), i.e. the
  high-cardinality `node` categorical is used only where it will exist at inference.
- Folds: `KFold(5)` over the **unique eval users** (dump:1518, 1550) = user-grouped by construction. Validation per fold = the
  full top-300 candidate lists of the held-out users, scored with Recall@40 against the eval ground truth.
- LightGBM (dump:1353-1480): Optuna over `application in {lambdarank, rank_xendcg}`, 20-500 iterations, lr, leaves, L1/L2,
  feature/bagging fraction, min_child_samples; **no early stopping** (iterations are a tuned hyperparameter). Fold bests
  0.2086 / 0.2023 / 0.1927 / 0.2043 / 0.1959 (2 trials each in the committed run).
- **CatBoost (dump:1509-1670)**: `CatBoostRanker(loss_function='YetiRank', task_type='GPU')`, Optuna over iterations 100-1000, lr
  0.01-0.3, depth 4-8, l2_leaf_reg 1-5, bootstrap_type {Bernoulli, No}, random_strength 0.5-2; no early stopping. Fold bests
  **0.2191 / 0.2112 / 0.2050 / 0.2148 / 0.2171** with winning params depth 8, lr 0.21-0.26, 350-760 iterations, l2 2.7-4.7.
- Final: boosters **averaged over the 5 folds**; "лучший сабмит был чисто на катбусте" (dump:1674). Scorers refitted on the full
  clickstream for inference.

### 1.6 Reported gains per component (what the sources allow)

- Retrieval, local unseen Recall@40: EASE-DAN 0.1735 < SASRec-node 0.1629 (worse) < **RDLAE 0.1994**. The ratio weighting and the
  3x contact input are inside the RDLAE number; there is no ablation of either in the repo.
- Ranker on RDLAE top-300: CatBoost folds 0.205-0.219 vs RDLAE 0.1994, i.e. **+0.006..+0.020 Recall@40 (+3..+10% relative)** for the
  whole feature set, of which no per-family ablation exists. LightGBM folds were 0.010 lower than CatBoost.
- Public: #1 0.23147 (99 entries), #2 0.22603, #3 0.22549 [V, leaderboard read in nn_x/audit/top1 3.1];
  the 2nd/3rd-place methods are not public (Data Fest 2025 talks only; searched RU/EN, GitHub, Habr, Telegram again this session:
  nothing).
- Other participants' code found this session (all far below): one at 22/311 (no code), one (CatBoostClassifier on
  four count features, R@40 0.058), one (EDA/preprocessing only).

### 1.7 Related Russian-cup practice that the plan also uses (already read in reports/deep_ru_competitions.md, re-cited here)

- Avito ML Cup 2024 2nd/3rd [V, deep_ru §4]: target means for all 1-3-way key combinations on **sliding daily windows**
  (train rows see only earlier days), transductive same-day frequency, pairwise-aware TE, CatBoost YetiRank 20K iterations depth 6
  lr 0.05 GPU, only users with both classes.
- VK RecSys 2024 3rd (Bragin) [V, deep_ru §6.2]: CTR/count per key for every feedback type, ALS per feedback type as features,
  a first-window boosting as a feature of the second, CatBoost PairLogit at the end.
- Avito production [V, deep_ru §2]: CatBoost L2 with ~200 features and a ranking loss; neural L3 = our architecture.

---

## 2. Mapping every component to our data

### 2.1 What we already measured (our numbers)

| 2025 component | our analogue(s) | measured value here | source |
|---|---|---|---|
| Item-item linear AE scores (EASE-DAN, RDLAE) | implicit ALS f64/f128 profile cosines (exp-decay, last-k, same-category, type-1), item2vec profiles, exact co-visitation `seq_cv_invd20/norm/pop` (train sources), cluster PMI `clu_pmi1024_mean` | single-feature val NDCG 0.3310 (ALS exp5) vs random 0.2890; **+0.0107 in a val-only GBDT**, +0.0032 on top of everything; as scalar residuals on G3 (nested, user-grouped): ALS dot 0.00000, profiles -0.0002..-0.0005, item2vec -0.0002..-0.0004, co-visitation 0.00000; cell AUC 0.39-0.46 (they favour the beaters) | eda_collab_stats §5-7 [E]; altapproach 2.1 [E] |
| Contact-ratio weighting of the scorer | none (ALS profiles are unweighted) | — | gap |
| Per-pair history matrices through W (cnt / contacts / n_days / exp_pos / exp_time) | `seq_cv_*` (1/d-weighted co-visitation over the last 20 history items), ALS profiles with tau 2/5/10, type-1-only and same-category profiles, `cat_cnt_et*`, `u_het*_share` | inside the NN inputs and the 177-feature cb26 set; NN residual of `seq_cv_invd20` exactly 0 | altapproach 2.1 R [E] |
| SASRec over nodes / categories as score + rank | pre2s causal transformer (d 384, 4 layers, 2 epochs) as 9 scalars (`pre2_fe`, `pre2_new`, ...); category PMI transition features | train-stage +0.0019 at 3 ep, +0.0009 at 10 ep (NN); zero-shot 0.359; residual on G3 0 / -0.0002 | altapproach 1, 2.1 D [E] |
| Node contact ratio / counts (events and once-per-user) | `tte_it_imp/pos/ctr/coec` (expanding daily windows on train; val/test from all train), `it_tr_*` (full-train, removed from FT by W3 for parity), `it_hist_*_luo` (label-free popularity by event type) | item TE single: COEC +0.0118, clicks +0.0158, hist_et1 count +0.0222 over random; **item_te group +0.0045 in a val-only GBDT, +0.0014 on top of all**; itemprior (val-label OOF / train-exploration prior) as a residual on G3: -0.0001 / +0.0002 / union +0.0004 (z 1.4) | eda_collab_stats §3-4, §7 [E]; nn_x/audit/itemprior/out.txt [E] |
| Category contact ratio | `tte_c_ctr` | category CTR/COEC single +0.002..+0.005 (random tie-break), useless with ties | eda_collab_stats §3 [E] |
| User contact ratio, last contact / last seen category | `tte_u_ctr/coec/imp`, `u_tr_ctr/feeds`; last-seen category = history hpos 0 (label-free, `cat_min_hpos`) | 30.4% of val feeds have a train user (median 1 train feed; 4.3% have >= 3) [M]; user-level stats are constant within a feed | [M]; eda_collab_stats §3 |
| User x category contact statistics | `tte_uc_imp/ctr` (train labels), `cat_cnt_*`, `cat_share_all`, `cat_cnt_et*` (own history, label-free) | uc_ctr +0.0041 single; own-history last-20 category count **+0.0269** single, but rank-corr 0.68 with f1 / 0.78 with f3 | eda_collab_stats §3, §6 [E] |
| Crosses (ratio products, "same as last") | none explicit (trees can form them) | — | gap (cheap) |
| CatBoost YetiRank, Optuna, fold averaging | cb26 e3: YetiRank GPU depth 8 lr 0.08 l2 5, 177 features, 134K exploration-like train feeds (p_random > 0.5, M >= 50), early-stopped on val user-half A | **val 0.3820** (half A 0.3853 / half B 0.3787), trH 0.3042, rho 0.856 to G3; member +0.00015 +- 0.00024 at w 0.05; law prediction -0.0042 | nn_x/cb26/logs/eval_e3.log [E] |
| Train-fed GBDT references | h3 arm A: LightGBM lambdarank, 53 leak-free features, 150K uniform train feeds, 200 rounds; arm B: val-only 5-fold | 0.3706 / 0.3644 | nn_x/audit/top1/out/h3_results.json [E] |
| Objective choice | XGB binary / rank:ndcg topk-300 / CatBoost YetiRank / YetiRankPairwise / QuerySoftMax / Logloss on val CV | all within +-0.001 (0.3676-0.3686) | deep_gbdt_ranking §2.1 [E] |
| Fold averaging | 5-fold FT models on test correlate 0.9974 | worth ~+0.0002 | sol_ensemble 2.4 [E] |
| The ensemble bar | strength-diversity law: break-even gap 0.049(1-rho) on val | at rho 0.856: gap 0.0071 -> **val >= 0.3932** needed | sol_ensemble 2.5 [E] |

### 2.2 Feasibility measurements made in this step (label-free, CPU, 12 threads)

Script: inline python in this step (hist_item counts cached at scratchpad `w25dl/hist_item_cnt_all.npy`).
- History item counts over train+val+test: 95.6M distinct items with >= 1 event, 63.7M with >= 2, 30.0M with >= 5, 5.57M with >= 20.
- **Share of candidate rows whose item is among the top-K items by history event count** (val / test): K = 30K: 1.2% / 1.0%;
  50K: 1.6% / 1.5%; 200K: 3.8% / 3.4%; 1M: 9.0% / 8.1%; 5M: 19.4% / 17.4%; 20M: 35.2% / 31.1%. Share of val positives in the top-50K /
  200K / 1M / 5M: 2.5% / 5.4% / 12.0% / 24.1%. Rows with >= 1 / >= 5 / >= 20 history events: val 62.1% / 41.2% / 20.4%, test 55.3% /
  36.4% / 18.2%.
  -> An item-level EASE/RDLAE "as in 2025" (30-50K most popular ids) would score 1-2% of our rows. Confirms deep_seq_pretrain_cf R5.
- Item-group coverage: k-means clusters of the int8 item embedding (feats_v2/clu_/kmeans.py; `data/clu_work/{cand,hist,item}_cl_{256,1024,4096}_{split}.npy`)
  are valid for 99.98% of val/test candidate rows and 100% of history rows that have an embedding (85.8% of val history rows).
- Users: 2,483,760 distinct non-zero users over all splits; train has 1.28 feeds per user (median 1). A user x 4096-cluster matrix
  built from **one snapshot per user** has at most ~2.5M x 4096 with ~10^8 non-zeros: trivial.
- Train: feeds per day 204K-292K on 12 full days (Sep 6-19, with 2 near-empty days); exploration-like feeds p_random > 0.5:
  **515,023 feeds / 29.7M rows** (164,118 feeds / 24.4M rows with M >= 50; 89,794 at p > 0.9 & M >= 50). cb26 e3 used 134,118 of
  these (holdout excluded).
- Target-statistic coverage on val rows: `tte_it_imp` >= 1 / 5 / 20 / 100 on 65.4% / 44.3% / 22.3% / 6.2% of rows; 19.7% of rows have
  >= 1 earlier train positive; test: >= 1 on 53.8%, >= 20 on 17.5% (private is colder; eda_collab_stats §2).

### 2.3 Transfer table

| component | transferable? | how | expected on our GBDT (prior) |
|---|---|---|---|
| Candidate generation (RDLAE top-300, EASE, SASRec retrieval) | **no** (candidates given) | — | — |
| Linear-AE item-item scorer | **yes, at item-group level** | EASE-DAN / RDLAE / plain EASE over K = 4096 and 1024 clusters (and 53 categories) with the winner's normalisation and hyper-parameters; user vector = the feed's own history snapshot | small: ALS/co-vis carry the same channel (val-only GBDT +0.0107 vs base, but the NN absorbs it). The cluster-level `clu_pmi` is a crude version (+0.001 label-free family on the GBDT, feat_clu_ §0) |
| Contact-ratio weighting of the scorer | yes | group-level positive ratio from exploration-like train labels (Bayesian, C = mean impressions), score x log1p(ratio); also the label-free "strong-intent share" of history events per group | small; the label-based cluster TE was 0 on a train-fed GBDT (feat_clu_ §0.2) |
| Per-pair history matrices through W | yes | per feed: cluster histogram of history by (all / event_type / recency weight 0.5^(hpos/5) / exp(-0.05 hpos) / newest hpos) pushed through W and through top-300-truncated W^T | small; partly `clu_hrw4096`, `clu_het14096` (KEEP, +0.001) |
| SASRec node / category scores | partly | **no NN training allowed in W25** -> reuse `pre2_*` encoder scalars (exist on all splits) and the category PMI transition features; a category-sequence SASRec (53 tokens) is a follow-up if the GBDT shows a category-sequence signal | 0 (pre2 residual on G3 is 0) |
| Node / category / user / user x category contact statistics | yes | section 3.2 (time-respecting; exploration-only and all-train sources; once-per-user variants) | item TE +0.0014 on top of everything in a val-only GBDT; 0..+0.0004 as a G3 residual |
| last contact / last seen category crosses | yes | last train positive's category per user (30% coverage) and history hpos-0 category (100%) | tiny |
| `location` | **no** (no location field) | — | — |
| Text / images | **no** (none in either task) | — | — |
| CatBoost YetiRank GPU, Optuna, no early stopping, 5-fold averaging | yes | section 3.3 | objectives tie within 0.001; Optuna/fold-averaging worth ~+0.001 at most |
| Training on the in-distribution window only, users with >= 1 positive, <= 30 negatives | partly | our in-distribution labelled data = 14.4K val feeds (too few: val-only 0.3644 < train-only 0.3706, h3) and the 515K exploration-like train feeds; every feed has >= 1 positive by construction; negatives must be kept (full-list NDCG) | the exploration subset is already the cb26 choice (0.3820 > uniform train) |

### 2.4 Honest answer to the user's question

We understood the 2025 recipe and tested its **information**, not its **estimators**. The information (item-item collaborative
structure from histories, item/category/user target statistics, a GBDT of those plus ranks) is measured at zero as a residual on
the NN pool and at +0.0045..+0.0107 inside a weak val-only GBDT. The estimators (linear AE with DAN normalisation at item-group
level, ratio-weighted scores, recency/type-weighted neighbourhoods, per-fold Optuna, fold averaging) were never built. The
plan below builds them so that the question is closed with a faithful pipeline rather than proxies, and so that any feature
family that moves the GBDT by >= +0.005 is identified for the NN. The prior that W25 reaches 0.393 is ~0.1; the prior that it
beats cb26 e3 (0.3820) by >= +0.003 is ~0.35 (the cb26 set lacks the group-level AE scores, the once-per-user and
exploration-only target statistics, and the crosses).

---

## 3. Build plan for `nn_x/w25/` (new files only)

Conventions: python = `<repo>/.venv/bin/python`; 12 threads per process (`OMP_NUM_THREADS=12`, `thread_count=12`);
CatBoost on **one** of GPUs N with `CUDA_VISIBLE_DEVICES=<id>`, `devices="0"`, `gpu_ram_part=0.25`, at most one W25 process per
GPU, `nvidia-smi` check (< 4 compute processes, >= 60 GB free) before each launch; no NN training; stop only own PIDs
(record them in `nn_x/w25/logs/PIDS.txt`). Outputs: `nn_x/w25/data/` (feature arrays, float32 flat aligned with
`data/flat/<split>/cand_*.npy`), `nn_x/w25/out/` (scores), `nn_x/w25/logs/`.

### 3.0 Row sets and folds (fixed before any feature is built)

- **XM** (primary training set): train feeds with `p_random > 0.5` (`eda/labels_position/out/train_prandom.npz`, key `p_random`,
  NaN -> 0), all M, **minus the 30K random-like holdout feeds** (trH, used by cb26; same exclusion list as `nn_x/cb26/cb_member.py`)
  -> ~485K feeds / ~28M rows. Group weight = `p_random` (arm) or 1 (arm).
- **PR** (secondary): 300K uniformly sampled production feeds (`p_random <= 0.5`), with **training-only columns** `pos` and
  `rkend = M-1-rank(pos)` (eda_labels_position §0.7: +0.0042 when fixed at pos 20 / rkend 50 at inference); at inference the columns are
  set to those constants for every split.
- **VF**: val, 5 user-grouped folds (`sklearn.model_selection.GroupKFold(5)`, groups = user_id, user-0 feeds as singleton groups,
  exactly `nn/finetune.py --u0_singleton`); the same folds as n2cpv3 so that ensemble OOF with G3 is valid.
- Evaluation sets: val (14,423 feeds, exact metric `src/metrics.py ndcg_per_feed`), the 30K trH holdout (only for arms that never
  train on it), val user-halves A/B as in cb26 when early stopping is used.

### 3.1 Scorers (group-level linear autoencoders), `nn_x/w25/build_lae.py`

Item group g(i): `cl4096(i)` (primary), `cl1024(i)` (secondary), `category(i)` (53; tertiary). Cluster ids per candidate from
`data/clu_work/cand_cl_{K}_{split}.npy`, per history event from `data/clu_work/hist_cl_{K}_{split}.npy` (-1 = no embedding -> dropped).

1. **User x group matrix for fitting W** (label-free): one row per non-zero user = the user's **latest train snapshot** (train) and
   **earliest snapshot** in val U test (the pre2s corpus rule, altapproach 1), plus every user-0 feed as its own row;
   entry = count of history events in group g (X_cnt) or 1[count > 0] (X_bin). Same-user later snapshots are never merged into an
   earlier feed's row (nothing later-than-feed enters a feed's own vector, see 3). Parity check: refit W from train snapshots
   only and report corr of the resulting scores on val (as `verify_pmi_leak.py` did: 0.99997).
2. **Fits** (numpy float64 on CPU; 4096^2 inversion is < 1 s): (a) EASE-DAN with the winner's hyper-parameters
   (`reg 10, drop_p 0.0498, alpha = beta = 0.952`), on X_bin; (b) RDLAE (`reg 500, drop_p 0.7, xi 0.1`) on X_bin; (c) plain EASE
   (`lambda` in {50, 500}) as a control; (d) the same three on X restricted to strong-intent event types (types chosen by the
   per-type positive association already tabulated as `u_het*_share` / `cat_cnt_et*`; report which types). Hyper-parameter grid on
   the XM-feed single-feature NDCG only (label use: train labels, allowed).
3. **Per-feed scoring** (every split, every feed, from the **feed's own history only**): user vector x_f from the feed's <= 256
   history events; variants = binary, count, strong-intent 3x (the winner's `binary + 2*contact`), recency 0.5^(hpos/5),
   exp(-0.05 hpos), type-1-only, newest-hpos; score s = (x_f @ W)[g(c)] for each candidate c, for W in {DAN, RDLAE, EASE} and for
   W and `truncate(W^T, 300)` (the winner's `score_20_*`). Output ~20 columns `w25_lae_<W>_<variant>` + within-feed rank/z of the
   two main ones (`src/stack_val.feed_rank_z`).
4. **Ratio weighting** (the winner's `log1p(bayesian_ratio_C)`): group positive ratio from **exploration-like train labels only**
   (XM feeds, leave-user-out is unnecessary at group level but the feed's own day is excluded for train rows, see 3.2),
   `r_g = (C * p0 + pos_g) / (C + imp_g)`, C = mean impressions per group; columns `w25_lae_<W>_main_x_logratio`. Also the
   label-free analogue: `r'_g` = share of strong-intent events among history events of group g over all snapshots.

### 3.2 Target statistics (time-respecting), `nn_x/w25/build_te.py`

Keys: item, category, cluster4096, user (user_id > 0 only; user 0 -> NaN), user x category, user x cluster1024, item x hour-of-day
(MSK), category x hour. Sources: (S1) all train feeds; (S2) exploration-like train feeds (p_random > 0.5); (S3) val labels (val and
test rows only).
- For **train rows on day d** (MSK day of the feed timestamp): statistics from train days **< d** only (expanding window; day 0 ->
  NaN), exactly as `src/features_tte.py`. For **val/test rows**: all train days. Val rows additionally get S3 statistics **OOF by the
  VF folds** (the evaluated feed's user never contributes); test rows get S3 from all of val. Never the feed's own label; never a
  same-user later feed.
- Statistics per key: impressions, positives, Bayesian CTR with C = mean impressions per key (the winner's prior), COEC with
  expected clicks from the random-feed position cell table (`data/clu_work/pos_cell_ctr.npy` / `train_ex.npy`, the
  pos x rank-from-end x M cells of feat_clu_ §2), **once-per-user** impressions and positives (the winner's `_last` counts), days
  since the key's last impression and since its first impression, number of distinct days with a positive.
- User-level: positives, impressions, ratio, feeds, **category of the user's most recent train positive** (`last_contact_category`),
  share of the user's positives in the candidate's category, share of the user's impressions in the candidate's category,
  `category == last_contact_category`; label-free: `category == category at hpos 0` (`last_seen`), `cat_min_hpos`.
- Crosses: `item_ratio * user_ratio`, `category_ratio * user_ratio`, `cluster_ratio * user_cluster_ratio`.
- Parity: every column built with the same code path for train/val/test; report train/val PSI and the NaN share per split (the
  FR-5 lesson in FEATURES_REVIEW: parity-broken columns inflate val and hurt the holdout).

### 3.3 Base feature set and the ranker, `nn_x/w25/train_w25.py`

- Base = the cb26 e3 177-feature list (`nn_x/cb26/cb_member.py member_features()`: `base_features.json` with LUO swaps minus the
  parity-broken `it_cand_cnt_*`, `hour`, `it_tr_*`, `u_tr_*`; includes `tte_*`, `clu_` KEEP-6, `ctx_` KEEP-24, `seq_*`, `pre2_*`), so that
  W25 is a strict superset and the ablation is clean. `als_*` columns do not exist on train (features_als.py docstring) and are
  therefore excluded from train-fed arms; they enter only the VF-only reference arm.
- New groups (ablated one at a time): **G1** LAE scores (3.1.3), **G2** ratio-weighted LAE scores and label-free intent-share
  weighting (3.1.4), **G3** target statistics (3.2; sub-split S1 / S2 / S3 and once-per-user), **G4** user last-positive / last-seen
  crosses and ratio products, **G5** recency / type-weighted LAE neighbourhoods (the `score_*` analogues).
- CatBoost: `CatBoostRanker(loss_function="YetiRank", task_type="GPU", devices="0", gpu_ram_part=0.25, thread_count=12,
  border_count=254, depth 8, learning_rate 0.08, l2_leaf_reg 5, bootstrap_type="Bernoulli", subsample 0.8, random_strength 1,
  sampling_unit="Group", iterations <= 3000, od_type="Iter", od_wait 200 on val user-half A, eval_metric NDCG:type=Base)` = the cb26
  settings, so the first comparison is feature-only. Then one 2025-winner-style run: **Optuna (<= 20 trials) over depth 6-8, lr
  0.03-0.25, l2 1-5, random_strength 0.5-2, bootstrap {Bernoulli, No}, iterations 300-1500 with no early stopping**, objective = val
  half-A NDCG, read on half B; and **3 seeds averaged by per-feed z** (`feed_rank_z`) for the test file.
- Arms (each a full val / holdout / slice read-out, paired per feed): A0 = cb26 e3 reproduction (base, XM-M>=50 subset, 1 seed;
  must land within +-0.001 of 0.3820 before anything else); A1 = base, full XM (all M) with p_random group weights; A2 = A1 + G1..G5;
  A3 = A2 + PR with training-only pos/rkend; A4 = A2 + VF folds (row weight 5, OOF over the 5 folds; test = mean of fold models);
  A5 (reference, cheap) = 2025-winner protocol literally: VF-only training (4/5 val folds), Optuna per fold, fold-averaged, with
  `als_*` included (expected ~0.365, h3 arm B 0.3644).
- Cost: cb26 e3 took 51 min for 2,100 iterations on 134K feeds x 177 features (one GPU). XM is 3.6x larger: ~3 h per arm; A2 ablations
  of 5 groups on the M >= 50 subset (~1 h each); Optuna 20 trials x ~20 min on the subset; total **~12-14 GPU-hours on one of GPUs N**
  (two GPUs if both idle, one process per GPU), ~4 CPU-hours for 3.1-3.2.

### 3.4 Ensemble test, `nn_x/w25/eval_w25.py` (reuses `nn_x/cb26/cbrule.py`)

Per-feed z of the W25 score (`feed_rank_z(off, M, s)[:,1]`) mixed into z(G3) = 0.5 z(ft) + 0.5 z(to) (`preds/ens_G3_{ft,to}_val.npy`)
at w in {0.05, 0.1, 0.2, 0.3}; w selected on 4/5 user-grouped folds (VF folds) and applied to the 5th (nested); slices M<50 / M>=50 /
Tue / Thu / user0 / known / cold terciles; holdout = member-equal z-mean of the train-only members on trH (A1-A3 only, never
A4/A5); pmix via `src/pmix.py`; the strength-diversity prediction `0.0186(1-rho) - 0.378*gap` reported next to the measured value.
Replication on PF7x only if G3 passes.

---

## 4. Pre-declared acceptance rules (timestamp: Thu Nov 12 23:45:59 UTC 2026)

All deltas are paired per-feed differences over the 14,423 val feeds with SE = sd/sqrt(n) (and over the 30K trH feeds for the
holdout); nothing is chosen on the public leaderboard; no number below has been computed.

(a) **Single-model value.**
   - Reproduction gate: A0 within +-0.0010 of 0.38201 (cb26 e3). If it fails, fix the harness before any other arm.
   - "GBDT progress": A2 - A1 >= +0.0030 with z >= 2 on val AND holdout delta >= 0 AND no slice < -2 SE. (The cb26 base already holds
     the earlier feature families; +0.003 is 2-3 paired SE for GBDT pairs on val.)
   - "Data arm value": A3 - A2 and A4 - A2 judged by the same rule; A4 on OOF val only (no holdout).
   - Report every arm's val, half A/B, holdout, M<50 / M>=50 / Tue / Thu, user0 / known, cold terciles, train/val PSI of new columns.
(b) **Ensemble value** (the user's "ensemble bar"). The best arm by (a) is a candidate member only if its val >= 0.3932 (law break-even
   at rho 0.856; recomputed with its measured rho as 0.40032 - 0.049(1-rho)). It is ACCEPTED as a member only if, at the nested-selected
   w: val delta > 2 SE, AND delta at w = 0.05 > 0, AND holdout delta >= 0 (point estimate), AND no slice < -2 SE, AND pmix >= -1 SE; and
   the same on PF7x. Otherwise: "GBDT closed again, with a faithful pipeline"; the test file is still written to `nn_x/w25/out/` for
   the record but no `subs/` candidate is produced.
(c) **Teaching result** (features for the NN). A group G in {G1..G5} (and the S2 / once-per-user sub-splits of G3) is TEACHING if its
   paired ablation inside A2 (same seed, same early-stopping half) is >= +0.0050 val with z >= 3 AND holdout >= 0 AND the gain is not
   confined to one slice (both M<50 and M>=50 positive). A teaching group is queued as an NN train-stage input (XT preset) under the
   2-epoch paired screen of nn_x/r7 (val > 1.5 SE and holdout >= 0 to promote). A group at +0.002..+0.005 (z >= 2) is "worth a
   2-epoch NN screen" only if its single-feature cell AUC in the INSIGHTS warm-beater cell is >= 0.53 (the P8 gate), because every
   earlier collaborative scalar pointed toward the beaters (altapproach 2.1).
(d) **Kill rules.** Stop the arm ladder when A2 - A1 < +0.0010 (the 2025 estimators add nothing beyond our features: write the
   ablation table and close); skip Optuna if A2 < 0.385; skip the ensemble test if the best arm < 0.390 (law gap > 0.010 at any rho we
   have seen for GBDTs, 0.80-0.86). Nothing in W25 may use a same-user later feed, test labels, or a val label outside the VF OOF
   scheme; `src/presub_gate.py` must PASS on any `subs/` candidate (none is expected).

---

## 5. Sources

- 1st place code: <public repository, link removed> (README "Avito ML cup 1st Place solution (RecSys track)
  "; files hack.ipynb, autoencoders/model.py, utils.py) [V, cloned 2026-11-12].
- Competition page https://ods.ai/competitions/avitotechmlchallenge2025 [V]; leaderboard (read in nn_x/audit/top1/findings.txt 3.1) [V];
  Data Fest 2025 track https://ods.ai/events/fest2025-avito-msc [S]; recording https://www.youtube.com/watch?v=-lUszYJT5BA (not transcribed).
- Data schema and organiser baseline: <public repository, link removed> (README, data/README.md, notebooks/baseline.ipynb) [V].
- Other 2025 repos found via GitHub search (none with a competitive method): one at 22/311 (README only),
  four more [V listing; three cloned].
- Papers behind the scorers: EASE-DAN arXiv 2504.05805; RDLAE arXiv 2305.12922; SASRec via RePlay [K].
- Our reports: reports/deep_ru_competitions.md, nn_x/audit/top1/findings.txt, nn_x/audit/altapproach/findings.txt,
  nn_x/audit/itemprior/{plan,out}.txt, nn_x/cb26/{cbrule.py,logs/eval_e3.log,logs/e3_xm_yeti.log}, nn_x/audit/top1/out/h3_results.json,
  reports/eda_collab_stats.md, reports/feat_clu_.md, reports/feat_ctx_.md, reports/deep_gbdt_ranking.md, reports/sol_ensemble.md §2.5,
  reports/eda_labels_position.md, reports/FEATURES_REVIEW.md, reports/deep_position_bias_random.md, src/features_tte.py,
  src/features_luo.py, src/features_als.py, feats_v2/clu_/kmeans.py.
- Web searches this session (RU/EN, standard + extended) for 2nd/3rd-place 2025 write-ups, Habr/Telegram summaries and any 2026
  "personal ranking" write-up: nothing beyond the sources above.
