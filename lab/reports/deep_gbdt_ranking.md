# Deep dive: GBDT learning-to-rank for Avito personal ranking

Date: 2026-11-11. Installed versions (checked in `.venv`): LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10.
Claims are tagged as follows:
- **[V]** I read it this session, either in the version-matched source or docs (URL given) or in our own logs.
- **[E]** Our own experiment. The result files are in `/tmp/scratchpad/exp/res/`.
- **[S]** Search snippet only.
- **[K]** Background knowledge, not re-verified.

---

## 0. Compute-rule incident (read first)

The previous attempt of this agent started jobs that **broke the hard compute rules**. `run_all.sh`, `run_xgb2.sh` and `run_xgb3.sh` passed `--gpu N` with no `CUDA_VISIBLE_DEVICES`. Here is what I saw [V, `nvidia-smi`]:
- **CatBoost GPU** jobs (`c_yeti`, `c_yetipw`, `c_qsm`, `c_logloss`) each held **~[n] GB on GPU N or GPU N**. They also held a 616 MiB context on **every one of the GPUs**.
- **XGBoost jobs.** `--gpu N` and `--gpu N` ran on GPUs N and 1. A job given `device=cuda:N` but no `CUDA_VISIBLE_DEVICES` (`tx_unb0`, pid 1504011) **also opened a 616 MiB context on GPU N**.
- **CPU.** About 60-90 threads were running against the 24-thread limit:
  - `tl_none` / `tl_pos`: 24 threads each.
  - Three `ltr_cv` LightGBM jobs: 16 threads each.
  - The XGBoost jobs on top of that.

I tried to stop the jobs on some GPUs, using SIGTERM on processes I had confirmed were this session's (cwd = scratchpad). The auto-mode permission classifier denied this twice, so I left them alone. I did not launch any new jobs.

Status at 04:00:
- **GPUs.** All GPU jobs from the old attempt have finished, and some GPUs show 0 MiB in use. There is no ongoing GPU violation.
- **CPU.** Two LightGBM CV jobs from the old attempt are still running, 16 threads each (`l_trunc30`, `l_xendcg_pos4`), and `l_binary` is queued after them. Together with other load this exceeds the 24-thread budget. The parent script is `run_lgb.sh` in the scratchpad `exp/` folder. A human should decide whether to kill them.

Lessons for every future script:
- Always use `CUDA_VISIBLE_DEVICES=N` with device index 0, for **XGBoost and CatBoost alike**. Passing `device=cuda:N` or `devices='5'` alone is not enough.
- For CatBoost, also set `gpu_ram_part` to 0.2-0.3. As observed, it otherwise grabs about the whole card.

---

## 1. TL;DR: most actionable settings

| # | Setting | Evidence |
|---|---|---|
| 1 | **Full-list truncation.**<br>XGBoost: `lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=300`. The default k=32 is too small.<br>LightGBM: `lambdarank_truncation_level>=300`. The default is 30. Use `eval_at=[300]` (default 1..5). | XGBoost topk-32 = 0.3626 vs topk-300 = 0.3662 (same normalisation), and vs binary 0.3677 (z=-3.9) [E]. LightGBM loop semantics read in source [V]. |
| 2 | **XGBoost: turn off score-difference normalisation** with `lambdarank_score_normalization=False`; keep `lambdarank_normalization=True`.<br>**LightGBM: keep `lambdarank_norm=true`.** It bundles both parts, and switching it off scored 0.3653 vs 0.3665 (n.s.) [E]. | XGBoost `rank:pairwise` mean-8: 0.3606 → 0.3669. With normalisation on, training stopped at ~11 iterations. topk-300: 0.3662 → 0.3672 (3-seed mean) [E]. XGBoost docs warn that it "can sometimes stagnate the training" [V]. |
| 3 | **The objective barely matters on this data.**<br>XGBoost `binary:logistic` ≈ `rank:ndcg` (topk-300, no score norm) ≈ CatBoost YetiRank / YetiRankPairwise / QuerySoftMax / Logloss. All are within ±0.001, while one paired SE is ≈0.0011. Pick objectives for **diversity and speed**, not for an expected gain. | Table in §2 [E]. |
| 4 | **Ensemble with per-feed z-scored scores.** Do not average raw scores across libraries. Do not average ranks without a tie-break. | All 12 OOFs, feed-z mean: 0.3705 (+0.0018 vs best single, SE 0.0008). Raw mean: 0.3692. Rank mean of 2 CatBoost models: 0.3619. The same with a 1e-6·z tie-break: 0.3675. The cause is **pessimistic ties** [E]. |
| 5 | **Position debiasing on train: LightGBM's GAM `position=` helped; XGBoost's IPW did not.** Both trained on biased train feeds and scored on the unbiased val. | LightGBM `position=min(rank,299)`, reg 0, 100K feeds: **0.3666 vs 0.3644 (Δ +0.0022 ± 0.0011, z=+2.1)**, single seed. XGBoost `lambdarank_unbiased`, 200K feeds: Δ -0.0002 (bias_norm=1) and Δ -0.0022 (bias_norm=0, full IPW) [E]. |
| 6 | **label_gain / ndcg_exp_gain / rescaling the positive label are no-ops for lambdarank with binary labels.** They only matter for `rank_xendcg`, where a larger positive label sharpens the target. | LightGBM ΔNDCG = gain_gap·\|Δdisc\|·invMaxDCG, so it is invariant to gain scale [V source]. XGBoost: 2^1-1 = 1 [V]. |
| 7 | **Group-level subsampling for the 247M-row train.**<br>LightGBM ≥4.6: `bagging_by_query=True` + `bagging_fraction` (gradients are computed only for sampled queries).<br>CatBoost: `sampling_unit="Group"`.<br>XGBoost: `sampling_method="gradient_based", subsample≥0.1` (row level, after full-group gradients). | [V] docs and source. |
| 8 | **Position debiasing: which library can model this train data.** LightGBM's `position=` takes arbitrary categorical ids, so it can encode slot × rank-from-end, which this train data needs. XGBoost's unbiased mode treats the row order within each qid as the position and only debiases the first k positions. LightGBM positions are **CPU-only**: the CUDA rank kernels contain no position code. | [V source]. LightGBM GAM gave +0.0022 [E]. |

---

## 2. Our evidence (finished runs, reused from the previous attempt)

### 2.1 Val-only CV: objectives and pair settings

Setup (`ltr_cv.py`):
- Data: 5-fold GroupKFold by user on val, 14,423 feeds, 1.39M rows, 122 features.
- Metric: OOF NDCG computed exactly like the organisers' evaluator (full list, pessimistic ties). Paired per-feed differences are reported against XGBoost `binary:logistic` (seed 0).
- XGBoost: GPU hist, depth 7, η 0.05, subsample 0.8, colsample 0.7, λ 10, 256 bins, early stopping (ES) 200 on the fold's held-out part.
- CatBoost: GPU, depth 7, lr 0.05, l2 3, 254 borders.

| Model | OOF NDCG | Δ vs xgb binary s0 (paired SE) |
|---|---|---|
| xgb `binary:logistic` seeds 0/1/2 | 0.3677 / 0.3686 / 0.3683 | 0 / +0.0009 / +0.0006 (0.0010) |
| CatBoost `Logloss` | 0.3685 | +0.0008 (0.0011) |
| xgb `rank:ndcg` topk-300, score_norm=False, η 0.02 | 0.3684 | +0.0007 (0.0012) |
| xgb `rank:ndcg` topk-300, score_norm=False, seeds 0/1/2 | 0.3671 / 0.3681 / 0.3665 | -0.0007 / +0.0004 / -0.0012 |
| CatBoost `YetiRankPairwise` | 0.3679 | +0.0002 (0.0011) |
| CatBoost `QuerySoftMax` | 0.3678 | +0.0000 (0.0011) |
| CatBoost `YetiRank` | 0.3676 | -0.0001 (0.0011) |
| xgb binary, weight `1/n_pos` on positives | 0.3674 | -0.0003 (0.0010) |
| xgb `rank:pairwise` mean-8, score_norm=False | 0.3669 | -0.0009 (0.0012) |
| xgb `rank:ndcg` topk-300 (defaults: score_norm and norm on) | 0.3662 | -0.0015 (0.0013) |
| xgb `rank:ndcg` topk-300, `lambdarank_normalization=False` | 0.3662 | -0.0015 (0.0012) |
| xgb binary, row weight `mean(M)/M` | 0.3651 | -0.0026 (0.0011), z=-2.3 |
| xgb `rank:ndcg` mean-8 / mean-1 | 0.3648 / 0.3639 | -0.0029 / -0.0038 |
| xgb `rank:ndcg` **defaults** (topk, k=32) | 0.3626 | -0.0052 (0.0013), z=-3.9 |
| xgb `rank:pairwise` mean-8 (score_norm on) | 0.3606 | -0.0071 (0.0013); early-stopped at 8-15 iterations |

LightGBM val-CV rows (CPU, 63 leaves, lr 0.05, min_data_in_leaf 100, ff 0.7, bagging 0.8, λ2 10, truncation 1000, ES 100) [E]:

| Model | OOF NDCG | Δ vs xgb binary s0 (paired SE) | best iterations per fold |
|---|---|---|---|
| LightGBM `lambdarank` (`lambdarank_norm=true`, default) | 0.3665 | -0.0012 (0.0014) | 43 / 168 / 72 / 135 / 34 |
| LightGBM `rank_xendcg` (label 1) | **0.3630** | **-0.0047 (0.0012), z=-3.9** | 47 / 63 / 45 / 98 / 185 |
| LightGBM `lambdarank` `lambdarank_norm=false` | 0.3653 | -0.0024 (0.0013); vs norm on: -0.0012 (0.0014) | 113 / 226 / 169 / 203 / 117 |
| LightGBM `l_trunc30`, `l_binary`, `l_xendcg_pos4` | still running at hand-off; see §2.4 | | |

`rank_xendcg` with binary labels is significantly worse: vs LightGBM lambdarank -0.0035, z=-2.5. This matches the weak-target analysis in §3.2.

In LightGBM, `lambdarank_norm=false` does **not** help (-0.0012, not significant), even though it trains about 2× longer. It removes *both* score-difference normalisation and query normalisation. XGBoost's best combination (score normalisation off, query normalisation on) cannot be expressed in LightGBM. **Keep LightGBM's default `lambdarank_norm=true`.**

Reading the table:
- **Seed-to-seed spread is ~0.001** and one paired SE is ~0.0011. On 14.4K feeds, no difference below ~0.002 can be resolved with a single seed. Use ≥3 seeds or paired tests.
- **The only clearly significant effects are the bad settings:**
  - k=32 truncation;
  - `mean` pair sampling;
  - score normalisation with `rank:pairwise`;
  - per-feed `1/M` weighting. It hurts because the metric weights feeds equally, while `1/M` down-weights the long feeds.

### 2.2 Ensembling (OOF, same folds) [E]

| Blend | NDCG | Δ vs best single (0.3686) |
|---|---|---|
| 3 xgb binary seeds, feed-z mean (raw mean is identical) | 0.3694 | +0.0008 (SE 0.0007) |
| 3 binary + 3 rank:ndcg seeds, **feed-z** | 0.3705 | +0.0018 (0.0008) |
| Same 6 models, **raw-score** mean | 0.3681 | -0.0005 |
| 4 diverse singles (bin, ndcg, YetiRank, YetiRankPW), feed-z | 0.3702 | +0.0016 (0.0007) |
| All 12 OOFs, feed-z / raw / rank-pct | 0.3705 / 0.3692 / 0.3697 | +0.0018 / +0.0006 / +0.0010 |
| Feed-z within each library, then equal weight across libraries | 0.3705 | +0.0019 (0.0008) |
| 2 CatBoost models, rank mean, **no tie-break** | 0.3619 | -0.0067 |
| Same, + 1e-6 · feed-z tie-break | 0.3675 | ≈ feed-z mean (0.3672) |

Notes on the blends:
- Averaging per-feed log-softmax gives the same ordering as averaging raw scores, because it subtracts a per-feed constant from each model's scores. Use feed-z or rank with a tie-break instead.
- Averaging integer ranks across a few models creates exact ties. Under the **pessimistic** tie rule, a positive tied with a negative is ranked below it. Always add a continuous tie-breaker.

### 2.3 Train-fit with position debiasing (biased train → unbiased val) [E]

Setup (`train_pos_exp.py`): 200K random train feeds (15.6M rows, rows sorted by `pos` within each feed). Leaky `it_tr_*` / `u_tr_*` features are dropped. XGBoost `rank:ndcg`, topk-300, score_norm off, depth 8, η 0.05, ES 150 on the full val.

| Run | Val NDCG | best_iter |
|---|---|---|
| `tx_none` | **0.3685** | 290 |
| `tx_unb1`: `lambdarank_unbiased=True`, `lambdarank_bias_norm` default (=1, see pitfall P8) | 0.3683 (Δ -0.0002 ± 0.0009) | 570 |
| `tx_unb0`: bias_norm=0 (exponent 1, i.e. full IPW) | 0.3663 (**Δ -0.0022 ± 0.0010, z=-2.2**) | 553 |
| `tx_featdrop15/50`: pos as a feature with 15%/50% NaN dropout, NaN at inference | **crashed**, no result | |
| LightGBM `tl_none`: lambdarank (norm on), 127 leaves, lr 0.05, 100K feeds (7.8M rows), 24 threads | 0.3644 | 197 |
| LightGBM `tl_pos`: same + `Dataset(position=min(rank_in_feed,299))`, `lambdarank_position_bias_regularization=0` | **0.3666 (Δ +0.0022 ± 0.0011, z=+2.1)** | 120 |

Learned XGBoost position ratios `ti+` [V, model config]. These are relevance-click propensities normalised so that pos0 = 1, raised to 1/(1+bias_norm) = 0.5:
- pos 1: 0.67
- pos 2: 0.54
- pos 5: 0.41
- pos 10: 0.36
- pos 50: 0.20
- pos 100: 0.15
- pos 200: 0.10

So the implied propensity ratio is ≈ ti+²: about 0.13 at pos 10 and 0.02 at pos 100. With bias_norm=0 the learned ti+ is the ratio itself: 0.47 at pos 1, 0.15 at pos 10, 0.028 at pos 100, 0.018 at pos 200. Inverse-propensity weights of 7-50× on deep positions add variance. **The stronger the IPW, the worse the result** (none 0.3685 > sqrt-IPW 0.3683 > full IPW 0.3663). This fits the variance explanation.

Why the featdrop runs crashed: `FileNotFoundError: data/feats/train/ctx_cl_best_f1_rk.npy`. A new feature had appeared in `data/feats/val/` (another worker is adding features) with no train counterpart. `all_feature_names("val")` picks up every val file, so any train-fit script must intersect the val and train feature lists.

Caveat: early stopping and evaluation both use the full val. Both arms are biased the same way, so the comparison is fair, but the absolute level is slightly optimistic.

**How to read the LightGBM GAM result** (+0.0022, z=2.1, one seed, one 100K-feed sample):
- It is the only debiasing variant that has beaten its own baseline.
- It is the GAM (additive logit) form, not IPW, so it does not reweight deep positions. That fits the XGBoost evidence above: IPW variance is what hurts.
- It is **not yet competitive in absolute terms.** The LightGBM 100K-feed models (0.3644 / 0.3666) are below XGBoost with 200K feeds and no debiasing (0.3685). The data size differs and LightGBM `lambdarank_norm` is on, so this is not a library verdict.
- Next steps:
  1. Replicate with 2 more feed samples or seeds at 200K feeds.
  2. Port the learned g(pos) to the other libraries as a fixed offset. LightGBM prints the bias factors via `Log::Debug` in `UpdatePositionBiasFactors` [V source], so run with `verbosity=2` and parse the last iteration. Pass them as `base_margin` (XGBoost) or `baseline` (CatBoost) when training on train, and use no offset at inference. This mirrors the existing `--pos_mode offset` in `src/train_lgb.py`, but with jointly learned factors instead of log CTR(pos).

### 2.4 Still running at hand-off (04:00)

These are queued in `run_lgb.sh` from the old attempt, at 16 threads each, so the CPU is still oversubscribed:
- `l_trunc30` (`lambdarank_truncation_level=30`);
- `l_binary` (LightGBM `binary`);
- `l_xendcg_pos4` (`rank_xendcg` with positive label 4).

When they finish, results land in `/tmp/scratchpad/exp/res/<tag>.json`. For a paired comparison run `python compare.py l_base l_` in the `exp/` folder.

`tx_featdrop15` and `tx_featdrop50` crashed (missing train feature file); see §2.3.

---

## 3. LightGBM 4.7: exact semantics (read in `src/objective/rank_objective.hpp` @ v4.7.0)

Source: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/objective/rank_objective.hpp [V]

### 3.1 `lambdarank`

**Pair loop and truncation.** Items are stable-sorted by the *current* score. The loop runs `for i < min(cnt-1, truncation_level)` and `for j in (i, cnt)`, skipping equal labels.
- So truncation bounds only the *upper* item of each pair, in the model's ranking.
- A positive ranked 50th still receives gradient from the negatives ranked in the top T.
- `inverse_max_dcg` is computed at the truncation level.
- **Setting:** `lambdarank_truncation_level` ≥ max feed size, i.e. 300. Cost per feed per iteration is O(min(T,M)·M) ≈ M²/2: about 3K pair visits for M=78, about 9.5G per round on the full train. That cost is small next to histogram building.

**`label_gain`.** `delta_pair_NDCG = (gain_hi − gain_lo) · |disc_hi − disc_lo| · inverse_max_dcg`. With binary labels, scaling the gain by g multiplies the numerator by g and invMaxDCG by 1/g, so it is **invariant**. `label_gain=[0,10]` is therefore a no-op. To change gradient magnitude, use `sigmoid` or the learning rate.

**`lambdarank_norm=true` (default) does two things:**
1. It divides each pair's ΔNDCG by `0.01 + |s_i − s_j|`. This is score-distance normalisation; it is skipped only when best == worst score.
2. It rescales the whole query by `log2(1+Σλ)/Σλ`.

`lambdarank_norm=false` removes both. XGBoost exposes them separately (§4). Our XGBoost data says (1) hurts and (2) is neutral. LightGBM cannot switch off (1) alone. Switching off both in LightGBM gave 0.3653 vs 0.3665 (n.s.) [E].

**Weights** are **per row** and multiply that row's gradient and hessian.

**Position debiasing** (`Dataset(position=...)`, v4.1+, CPU path):
- Each iteration computes gradients on `score + pos_bias[pos_id]`. The trees fit f(x); the per-position biases g(pos) are updated by one Newton step with step size = `learning_rate`.
- The update is `b ← b + lr·(G − b·r·n) / (|H| + r·n + 0.001)`, where:
  - G = −Σλ over rows at that position;
  - H = Σh;
  - r = `lambdarank_position_bias_regularization`;
  - n = row count at that position.
- So r is on the scale of a per-row hessian. With normalised lambdas that is roughly 0.01-0.1, so a sensible grid is {0, 0.01, 0.1, 1} (my inference from the update rule).
- **Position values are categorical ids.** Advanced-Topics says "LightGBM does not care about the absolute or relative magnitude of the values" [V](https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/docs/Advanced-Topics.rst). So you can pass a crossed id such as slot-bucket × rank-from-end-bucket. That matters here because train has a strong last-slot effect (see `deep_position_bias_random.md`).
- The **validation Dataset gets no positions**. Predictions are f(x) only, which is exactly what the unbiased val/test need.
- **The CUDA rank kernels have no position code.** `src/objective/cuda/cuda_rank_objective.cu` @ v4.7.0 contains 0 matches for "position" or "pos_bias" [V]. Expect the bias to be silently ignored with `device_type="cuda"`. Train position models on CPU.
- **Possible interaction with `bagging_by_query`.** `GetGradientsWithSampledQueries` recomputes gradients only for the sampled queries. `UpdatePositionBiasFactors` then loops over *all* rows, so unsampled rows contribute stale gradients. This is my inference from `gbdt.cpp` and `rank_objective.hpp` [V source, untested]. Do not combine the two until checked.
- **Confounding caveat.** The production positions in train were set by a relevance-aware ranker. Zhang et al. (KDD 2023) show that an additive bias tower then absorbs real relevance and harms the relevance tower [V abstract](https://arxiv.org/abs/2212.13937). That is why the regulariser matters. Do not expect a large gain when the features themselves carry little position signal. Measured on 100K train feeds with reg=0 and 300 position ids: +0.0022 (z=2.1) over the same LightGBM model without positions [E, §2.3].

### 3.2 `rank_xendcg` (Bruch, WWW 2021, arXiv 1911.09798)

- Each iteration draws a per-query target φ_i = 2^{y_i} − γ_i with γ ~ U(0,1), normalised to a distribution. The loss is the cross-entropy against `softmax(scores)` [V source].
- **The binary-label pitfall.** A positive gets 2−γ and each negative gets 1−γ. With M=96 and γ≈0.5, the target mass on the positive is only 1.5/(1.5+95·0.5) ≈ 3%. The optimum then separates positive from negatives by only about log 3 in logit space. This is a weak, heavily smoothed signal.
- Relabelling positives to y=4 (mass ≈ 15.5/(15.5+47.5) ≈ 25%) sharpens it. The still-running `l_xendcg_pos4` tests exactly this.
- Unlike lambdarank, feeds with no positives would still get a "make the scores uniform" gradient. Our data has no zero-positive feeds in train or val [E], so this does not apply here.
- **Positions are a no-op for `rank_xendcg`.** The base class adds `pos_biases_`, but only `LambdarankNDCG` overrides `UpdatePositionBiasFactors`, so the biases stay 0 [V source].
- Docs: "rank_xendcg is faster than and achieves the similar performance as lambdarank" [V](https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/docs/Parameters.rst). It is O(M) per query instead of O(M²).

### 3.3 Other parameters

- **`bagging_by_query`** (new in 4.6.0) [V Parameters.rst]: samples whole queries and computes gradients only for them [V gbdt.cpp]. It requires `data_sample_strategy=bagging`; otherwise it is silently reset to false with a warning [V config.cpp]. Plain row bagging keeps the full-query gradient cost and only subsamples rows for the histograms.
- **`eval_at`** defaults to 1..5. For a full-list metric use `eval_at=[300]` (or 1000, as in our scripts).
- **`objective_seed`** (default 5) affects only `rank_xendcg`.
- **Building the Dataset at 247M × 122.** The raw float32 array alone is ~120 GB. Use `lightgbm.Sequence` (≥3.3.0, "no need to read all data into memory") [V `basic.py`] over the per-feature `.npy` memmaps. The binned Dataset takes ~1 byte per value at `max_bin` ≤ 255, so ~30 GB [K].

---

## 4. XGBoost 3.4.1: exact semantics (`src/objective/lambdarank_obj.{h,cc}`, `src/common/ranking_utils.h` @ v3.4.1)

Sources [V]:
- https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/src/objective/lambdarank_obj.cc
- https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/doc/parameter.rst
- https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/doc/tutorials/learning_to_rank.rst

| Parameter | Default (source) | Meaning / finding |
|---|---|---|
| `lambdarank_pair_method` | `topk` | `topk`: for i < k in the model ranking, pair with every j>i (same loop shape as LightGBM). `mean`: each item gets `num_pair` random partners with a different label. |
| `lambdarank_num_pair_per_sample` | 32 for topk, 1 for mean | For full-list NDCG, set ≥ max M (300). Measured: k=32 → 0.3626, k=300 → 0.3662 [E]. |
| `lambdarank_score_normalization` | true (added in 3.0) | ΔNDCG / (\|Δs\|+0.01). **Set False**: +0.001 to +0.006 [E]. Docs: "can sometimes stagnate the training". |
| `lambdarank_normalization` | true (added in 2.1) | topk: query scale log2(1+Σλ)/Σλ; mean: 1/num_pair. Neutral here [E]. |
| `ndcg_exp_gain` | true | 2^y−1. Identical to linear gain for binary labels, so there is no reason to change it. |
| `lambdarank_unbiased` | false | Unbiased LambdaMART (Hu et al., WWW 2019). |
| `lambdarank_bias_norm` | **1.0 in the source**. The docs say 2.0; that is wrong for 3.4.1, and the model-config dump shows `'1'` [V]. | ti+ = (L_i/L_0)^{1/(1+p)}. |

**Unbiased mode details** [V source]:
- The position is the **row index within the qid**, so rows must be sorted by display position. The tutorial demo does this.
- Only the first `MaxPositionSize()` positions are tracked: k for topk, or min(max group size, 32) for mean. Pairs involving deeper rows are neither reweighted nor used to estimate bias.
- It cannot model rank-from-end effects.
- Position debiasing is "not yet supported for existing distributed interfaces" [V tutorial].

**Weights for `rank:*` must be per group.** The source has `CHECK_EQ(info.weights_.Size(), n_groups)` [V]. Our `ltr_cv.py` only applies row weights to `binary:logistic` for this reason.

**Tutorial guidance** [V]:
- With small data, use NDCG or `rank:pairwise` with `mean` pairs.
- With large data, use the metric-matched objective with `topk`.

On our val-only data, topk-300 beat mean-1 and mean-8 by 0.002-0.003 [E].

**GPU / scale** [V parameter.rst, external_memory.rst]:
- `sampling_method="gradient_based"` lets `subsample` go as low as 0.1 "without loss of model accuracy"; CPU and GPU are both supported since 3.2.0.
- For data that does not fit, use `ExtMemQuantileDMatrix` with a `DataIter`. GPU external memory is recommended mainly on NVLink-C2C systems; from disk it is "likely to make the GPU slower than the CPU".
- **In-core option for us:** `QuantileDMatrix` built from a `DataIter` over feed-chunked memmaps, so there is no [n] GB host copy. The quantised ELLPACK is roughly 1-2 bytes per value, so 30-60 GB for all 247M rows [K]. The GPUs are GPUs with (GPU memory size removed) each [V nvidia-smi]. GPU N's free memory has swung between ~[n] GB and ~[n] GB this session because of other jobs.
- So a full-train in-core fit on GPU N is feasible when it is free. As a fallback, train on ~1-1.5M feeds (80-120M rows) per model, with a different feed sample for each seed (bagging by feed).

---

## 5. CatBoost 1.2.10 ranking losses

Sources [V]:
- https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/docs/en/concepts/loss-functions-ranking.md
- `catboost/private/libs/options/loss_description.cpp`
- `docs/en/presets.yaml`

| Loss | Key params (defaults) | GPU |
|---|---|---|
| `YetiRank` | `permutations=10`, `decay=0.85` (Classic); `mode` ∈ {Classic, DCG, NDCG, MRR, ERR, MAP} (since 1.2.1); `top` (unlimited); `noise=Gumbel`; `num_neighbors=1` | Yes, Classic mode only |
| `YetiRankPairwise` | same params. The docs say YetiRank "gives less accurate results on big datasets compared to YetiRankPairwise but it is significantly faster" | Yes, Classic mode only |
| `PairLogit` / `PairLogitPairwise` | `max_pairs` (default: all pairs) | Yes |
| `QuerySoftMax` | `beta=1` | Yes |
| `QueryCrossEntropy` | `alpha=0.95` | Yes |
| `LambdaMart` | `metric=NDCG`, `sigma=1`, `norm=True` | **CPU only** |
| `StochasticRank` | `metric` (required), `num_estimations=1`, `mu=0` | **CPU only** |

**GPU constraints** [V](https://catboost.ai/docs/en/references/training-parameters/common):
- `depth` ≤ 8 on GPU for YetiRank, PairLogitPairwise and QueryCrossEntropy.
- `random_strength` is not supported for QueryCrossEntropy, YetiRankPairwise or PairLogitPairwise.
- `sampling_unit="Group"` gives group subsampling.
- The default bootstrap for YetiRankPairwise / PairLogitPairwise / QueryCrossEntropy is Bernoulli with subsample 0.5.
- The `NDCG` eval metric "is not implemented on GPU" and runs on CPU. That warning is in our logs [V]. Use `metric_period` 25-50.

**Benchmarks.** Lyzhin, Ustimenko, Gulin, Prokhorenkova, "Which Tricks are Important for Learning to Rank?" (arXiv 2204.01500 v2; published at ICML 2023 [K]) [V PDF tables]. They compare methods in one CatBoost framework plus LightGBM, with tuned hyperparameters and ≤1000 trees.

NDCG@10 with tuned hyperparameters (Table 2):

| Method | Web10K | Web30K | Istella-S |
|---|---|---|---|
| YetiRank | 50.75 | 52.31 | 77.15 |
| LightGBM | 50.39 | 52.21 | 76.48 |
| LambdaMART (CatBoost) | 49.38 | 50.93 | 75.55 |
| StochasticRank | 50.34 | 51.63 | 74.88 |
| QueryRMSE | 50.55 | 52.19 | 74.83 |

- With default hyperparameters (Table 4), LightGBM was the best on Web10K and Web30K (50.34 / 51.06).
- **Stochastic smoothing (the Gumbel noise) is essential.** YetiLoss NDCG@10 on Web10K was 50.76 with logistic noise vs 48.49 with no smoothing (Table 3).
- YetiLoss (`mode=MRR` / `MAP`) wins on MRR and MAP.
- With ~1.5 positives per feed, our full-list NDCG behaves like an MRR-type metric (1/log2(1+rank of the positive)). So `YetiRank:mode=NDCG` or `mode=MRR` on CPU is a reasonable diversity candidate. It is CPU-only and therefore slow; use it on a feed sample.

**On our data [E]:** YetiRank 0.3676, YetiRankPairwise 0.3679, QuerySoftMax 0.3678, Logloss 0.3685. They are indistinguishable from each other and from XGBoost. The 4-model CatBoost blend (feed-z) reaches 0.3683, +0.0007 over YetiRank alone. CatBoost GPU costs 450-980 s per 5-fold run vs 70-150 s for XGBoost GPU.

---

## 6. Sparse positives, group sampling, distribution shift

- **No zero-positive feeds** in either split [E]:
  - Train: 3.15M feeds; positive rate 2.21%; 1.73 positives per feed; mean M = 78.3, max 300.
  - Val: positive rate 1.58%; 1.52 positives per feed; mean M = 96.4.
- So there is no free speedup from dropping all-negative feeds.
- There is **list-length and base-rate shift**: train feeds are shorter and have more positives.
- **Never subsample rows inside a feed for lambdarank.** It changes ranks, discounts and IDCG (my inference from the loop semantics). Subsample **whole feeds** instead:
  - LightGBM `bagging_by_query`;
  - CatBoost `sampling_unit=Group`;
  - an explicit feed sample per seed or model.
- For `binary:logistic`, negative downsampling within feeds is fine, because ranking is invariant to the calibration offset [K].
- **Do not weight feeds by 1/M**: it hurts, -0.0026, z=-2.3 [E] (tested with XGBoost `binary:logistic`). The metric is an unweighted mean over feeds, and lambdarank gradients are already normalised per query.
- **Upweight val in a final train+val fit** (my proposal, untested). Val/test are uniformly shuffled; train is position-biased. Use a group weight such as 5-20× for val feeds:
  - XGBoost: per-group weights;
  - LightGBM: row weights.
- This is the cheapest form of the "use the unbiased data" idea. Choose the weight by CV over val folds, with each fold's val part excluded from training.

---

## 7. Feature selection for rankers

Mostly [K] (practice, not verified this session):

- **Ranking objectives cannot use pure between-feed signal.** Lambdarank, xendcg and YetiRank are invariant to adding a per-feed constant. Features that are constant within a feed (user-level features, M) can only act through interactions. Within-feed transforms carry most of the signal: rank, z-score, gap-to-max. Our status notes already confirm this.
- **Use permutation importance on val NDCG,** permuting **within feeds**. Do not rely on split/gain importance, which favours high-cardinality continuous features. Drop a feature only if its importance is ≤ 0 in most folds and across ≥2 seeds. The noise floor here is ~0.001 NDCG.
- **Use adversarial validation, train vs val/test.** This matters because of position bias. Features built from production impressions (popularity, CTR) can encode the old ranker's ordering. Any feature that separates train from val is a candidate to drop or recompute on unbiased data.
- **Leaky-on-train features (`it_tr_*`, `u_tr_*`) are already excluded** from the train-fit scripts (`LEAKY_ON_TRAIN`) [V code].

---

## 8. Ensembling rankers: recommendations

1. Normalise each model's scores **per feed** before averaging. Z-score is best on our OOF; a per-feed rank with a tie-break is a close second [E].
2. Get diversity from **different libraries and objectives**, not from more seeds:
   - 3 seeds of one model: +0.0008;
   - binary + ndcg families: +0.0018;
   - the CatBoost family on top added nothing further [E].
3. Weight families equally first (hierarchical: feed-z within a family, then the mean across families). Only then try hill-climbing weights. On 14.4K feeds a weight search over more than a handful of models will overfit at the 0.001 level.
4. **Stacking.** A second-level LightGBM `lambdarank` on feed-z OOF scores plus a few meta-features (M, user-seen-in-train flag, history length) needs nested CV on val. Expect ≤ +0.001 over a z-mean unless the meta-features matter (they might: seen vs unseen users) [K].
5. **Always add a tie-break**, because the metric uses pessimistic ties.

---

## 9. Pitfalls checklist

- **P1.** LightGBM defaults: `lambdarank_truncation_level=30`, `eval_at=1..5`. XGBoost default: `topk` with k=32. All are wrong for full-list NDCG.
- **P2.** XGBoost ≥3.0 has `lambdarank_score_normalization=True` by default, and with `rank:pairwise` it stops at ~10 iterations here. LightGBM's `lambdarank_norm` bundles score normalisation with query normalisation. Switching it off in LightGBM did not help (§2.1).
- **P3.** `label_gain`, `ndcg_exp_gain` and the positive label value do nothing for lambdarank with binary labels. For `rank_xendcg`, a positive label of 1 gives a nearly uniform target when M≈100.
- **P4.** `rank_xendcg` ignores `position` (the biases stay 0).
- **P5.** LightGBM positions are CPU-only (no position code in the CUDA kernels).
- **P6.** `position` together with `bagging_by_query` may update the biases from stale gradients (source inference).
- **P7.** XGBoost unbiased mode:
  - the row order *is* the position;
  - only the top k positions are tracked;
  - it is not available in distributed interfaces;
  - `rank:*` weights must be per group.
- **P8.** The XGBoost docs say `lambdarank_bias_norm` defaults to 2.0; the 3.4.1 code uses 1.0.
- **P9.** CatBoost GPU:
  - depth ≤ 8 for YetiRank;
  - non-Classic YetiRank modes, LambdaMart and StochasticRank are CPU-only;
  - the NDCG metric runs on CPU;
  - it grabs most of the card: `gpu_ram_part` defaults to 0.95 [V](https://catboost.ai/docs/en/references/training-parameters/performance);
  - it opened contexts on every visible GPU even with `devices='2'` [V observed]. Always set `CUDA_VISIBLE_DEVICES`.
- **P10.** XGBoost `device=cuda:N` without `CUDA_VISIBLE_DEVICES` still opened a context on GPU N [V observed].
- **P11.** Rank-averaging without a tie-break: -0.006 [E].
- **P12.** Early stopping on the same val you report on inflates the absolute numbers. For a train-fit model, early-stop on half of val (split by user) and report on the other half, or use a round count fixed from CV.
- **P13.** On this val, a single-seed comparison cannot resolve differences below ~0.002.

---

## 10. Suggested next experiments (small, compliant)

Run these only with `CUDA_VISIBLE_DEVICES=N` and ≤24 CPU threads in total.

1. **LightGBM train-fit with `position`: the top priority**, since `tl_pos` gave +0.0022 at z=2.1.
   - Replicate at 200K feeds with 2 seeds, using `position=min(rank,299)`.
   - Then try a crossed id: slot-bucket × rank-from-end-bucket. The last-slot effect needs it.
   - Grid r ∈ {0, 0.01, 0.1, 1}.
   - CPU only, ≤24 threads. Do not combine with `bagging_by_query` (P6).
2. **XGBoost train-fit `binary:logistic` vs `rank:ndcg`** (topk-300, no score norm) on the same 200K-1M feeds. Then blend with feed-z. This checks whether the objective tie seen on val-CV also holds when training on train.
3. **Train+val with val group weight** w ∈ {1, 5, 20}, evaluated on held-out val folds.
4. **CatBoost `YetiRank:mode=NDCG` (CPU)** on a 50K-feed sample, purely as an ensemble-diversity candidate.

---

## Sources

- LightGBM v4.7.0:
  - rank objective source: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/objective/rank_objective.hpp
  - CUDA rank objective: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/objective/cuda/cuda_rank_objective.cu
  - bagging: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/boosting/bagging.hpp
  - gbdt: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/boosting/gbdt.cpp
  - config: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/src/io/config.cpp
  - parameters: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/docs/Parameters.rst
  - advanced topics: https://raw.githubusercontent.com/microsoft/LightGBM/v4.7.0/docs/Advanced-Topics.rst
  - PR #5929 (GAM position bias, merged 2023-09-04, shipped in 4.1.0): https://github.com/microsoft/LightGBM/pull/5929
- XGBoost v3.4.1:
  - https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/src/objective/lambdarank_obj.cc
  - https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/src/objective/lambdarank_obj.h
  - https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/src/common/ranking_utils.h
  - https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/doc/parameter.rst
  - https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/doc/tutorials/learning_to_rank.rst
  - https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/doc/tutorials/external_memory.rst
  - https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/demo/guide-python/learning_to_rank.py
- CatBoost v1.2.10:
  - https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/docs/en/concepts/loss-functions-ranking.md
  - https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/private/libs/options/loss_description.cpp
  - https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/docs/en/presets.yaml
  - https://catboost.ai/docs/en/references/training-parameters/common
- Papers:
  - Lyzhin et al., Which Tricks are Important for LTR?: https://arxiv.org/abs/2204.01500 (PDF tables read)
  - Bruch, XE-NDCG: https://arxiv.org/abs/1911.09798
  - Zhang et al., Towards Disentangling Relevance and Bias in ULTR (KDD 2023): https://arxiv.org/abs/2212.13937
  - Hu et al., Unbiased LambdaMART (WWW 2019): https://arxiv.org/abs/1809.05818 (abstract read; it estimates click and unclick position biases jointly with the ranker)
