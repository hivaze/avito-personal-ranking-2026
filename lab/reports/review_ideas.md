# Review: improvement ideas (critical review workflow, "ideas" lens)

Author: review agent (CPU only, read-only on the project). Date 2026-11-11 (after CURRENT_STATUS 3g, 13:45 UTC).
Scripts and logs: `/tmp/scratchpad/review/ideas/`
(file names below are relative to that folder). All runs: `CUDA_VISIBLE_DEVICES=""`, 12 threads.
All NDCG numbers use `src/metrics.py` (full list, binary gain, pessimistic ties). "SE" = SE of the paired per-feed delta.
It does not include training noise. Seed-to-seed spread of the NN is ~0.0005-0.001.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

### Finding 1: private is the coldest data we will be scored on, and none of our yardsticks measures staleness directly

| | train holdout | val Tue | val Thu | public (Sat-Sun) | private (Sun-Tue) |
|---|---|---|---|---|---|
| candidates never seen in train (cand or hist) | 0% by construction | 25.5% | 32.5% | 36.9% | **40.3%** |
| `it_hist_all_luo == 0` (= `prew_*` NaN) | | 35.9% | 40.3% | 42.8% | **45.6%** |
| listing age 1-7 d with LUO == 0 | 14.5% (train rows) | 72% (val, both days) | | 85% | **88%** |
| `it_cand_cnt_train > 0` (fine-tune branch) | | 65.4% (val, both days) | | 56.4% | 52.7% |

### Finding 2: the two cheap staleness proxies disagree, so neither can be trusted

- **Per-feed cold-share split.** Changes that strengthen warm priors look worse on cold feeds, and vice versa:
  - prew3: +0.0021 ± 0.0010 on feeds with cold share < 0.3, −0.0040 ± 0.0030 on feeds with cold share ≥ 0.45;
  - uniform ID dropout q15: −0.0018 ± 0.0009 on feeds with cold share < 0.3, +0.0046 ± 0.0026 on feeds with cold share ≥ 0.45.
- **Day split.** Same population, but Thursday is 7 points colder than Tuesday. Here the signs reverse:
  - prew3 helps *more* on Thursday (+0.0021 ± 0.0011 vs +0.0001 ± 0.0010);
  - q15 loses on Thursday (−0.0009 ± 0.0009).
- **Reweighting fails too.** Reweighting val to the test cold mix over-predicts v5's public score by 0.0023 (M-only under-predicts by 0.0012).
- **Conclusion:** per-feed cold share mostly reflects the *population* (which users see fresh listings), not staleness.
- **Idea #1** builds the right tool: a val copy whose time-sensitive inputs are rebuilt with a private-like 8-10-day gap. Until it exists, do not accept or reject anything on cold-share arguments, mine included.

### Finding 3: popularity inputs are 37x sparser for post-train items (an asymmetry, not a leak)

- `it_hist_*_luo` counts val/test snapshots, a ~1/37 sample of traffic, at the same weight as train snapshots.
- As a result, young items with val/test-period popularity are under-rated by the train stage: actual/expected positives 1.12-1.18.
- The fine-tune fixes part of this.
- A post-hoc flag adds only +0.0003 ± 0.0003 (nested). Idea #5 fixes it at the feature level.

### Ranked ideas (expected private gain per cost)

Details are in section 2.

| # | idea | exp. private gain | cost | overfit risk |
|---|---|---|---|---|
| 1 | **Stale-val:** a private-gap copy of val (time-sensitive inputs rebuilt with train sources cut at Sep 14), re-predicted by existing checkpoints; a second gate for warm/cold-trading decisions | indirect, prevents ±0.001 mistakes; it is the tool #2/#3/#5/#6 need | CPU 1-2 h + inference only | none |
| 2 | Drop drifting inputs from the fine-tune branch (`hour`: PSI 0.51 val → private; train-candidate stats: coverage 65% → 53%) | 0 on val, +0 to +0.001 private | < 1 GPU-h | none (removal, already measured neutral) |
| 3 | WSD continuation + cooldown-branch members / soup (extends the proven training-length lever at a fraction of the cost) | +0.0005-0.0015 | 0.2-0.5 run per member | low |
| 4 | `pre2_*` (full coverage) vs `prew_*` (NaN on 38-46% of rows, coverage drifts val → private) in the train stage, judged on val + holdout + stale-val | 0 to +0.001 vs prew | 2 runs (pre2 in progress) | low |
| 5 | Density-matched popularity / co-vis counts (val/test snapshot occurrences × ρ ≈ 37) | +0.0003-0.001 | CPU 1 h + 2 runs | low |
| 6 | Age-conditioned ID-input dropout (blank co-vis/LUO inputs mostly for young listings, matching where test zeros occur) | 0 to +0.0015 private | code + 2 runs | medium |
| 7 | Export the production-order head (`aux_out`) at inference as an extra signal | 0-0.0005 | GPU minutes | low with a pre-declared weight |
| 8 | Full-val refit of the fine-tune for the test predictions | +0.0002-0.0005 (test only) | ~1 GPU-h per member set | low, but not measurable on val |
| 9 | Final-pick hedge: best-val blend plus the best stale-val blend (e.g. w_trainonly 0.25) | insurance | 0 | none |
| 10 | Recency-weighted cooldown (over-sample Sep 17-19 in the WSD cooldown) | 0-0.0005 (weak evidence) | flag + 2 runs | low |
| 11 | Per-positive softmax loss (positives do not compete in the denominator) | ~0 (my proxy test is null) | 1 run | low |
| 12 | Capacity re-probe under Muon (d = 192/256 with the (c) recipe) | 0-0.001 | 2 runs | low |

What I measured is in section 1. Done or rejected items that I do not re-propose are in section 3. Checked-and-correct items and closed dead ends are in section 4.

---------------------------------------------------------------------------------------------------------------------

## 1. New measurements behind the ideas

### 1.1 Cold share by split, day and feed (`cold.py`, `prewnan.py`, `blendmix.py`)
Cold = item id never seen as a train candidate or in a train history. This reproduces err_SYNTHESIS (val 28.86%, test 39.26%).

Rows by day:

| day | cold | LUO == 0 |
|---|---|---|
| val Tue Sep 22 | 25.5% | 35.9% |
| val Thu Sep 24 | 32.5% | 40.3% |
| test Sat | 36.5% | 42.4% |
| test Sun | 38.5% | 43.9% |
| test Mon | 40.1% | 45.5% |
| test Tue | 41.5% | 46.7% |

Split by LB part, on rows:

| | train-unseen | LUO == 0 | `seq_cv_pop` == 0 |
|---|---|---|---|
| val | 28.9% | 37.9% | 42.0% |
| public | 36.9% | 42.8% | 46.4% |
| private | 40.3% | 45.6% | 47.6% |

Per-feed cold-share distribution:

| feed cold share | val | test |
|---|---|---|
| < 0.15 | 9.5% | 1.3% |
| 0.15-0.30 | 46.9% | 17.5% |
| 0.30-0.45 | 35.2% | 51.8% |
| 0.45-0.60 | 7.0% | 24.7% |
| ≥ 0.60 | 1.4% | 4.7% |

The effective sample size of val reweighted to the private (M × cold) mix is 43% of 14,423.

### 1.2 Cold-share reweighting is not a valid level predictor; M-only stays the right calibration (`calcheck.py`)
- v5 (0.5 z(nn_final2 OOF) + 0.5 z(stk2_res2_r3), from preds_quarantine, read for evaluation only), val 0.38692:
  - predicted public, M-only: 0.37896;
  - predicted public, M × cold-share: 0.38247;
  - **actual public 0.3802**.
- The cause: on val, high-cold-share feeds score *higher* (at M ≥ 160: 0.365 at cold share ≥ 0.5 vs 0.276 at 0.2-0.35, nn_final4 mix). They are a different population. On test, high cold share comes from staleness.
- This also qualifies playbook K7 ("private-like reweighting of val OOF"): do not use it for levels.

### 1.3 Warm/cold-trading decisions: per-feed cold share vs day split (`qdrop.py`, `daysplit.py`, `wday.py`)

| delta (train-only unless noted) | val | cs < 0.3 | cs ≥ 0.45 | val reweighted to private (M × cold) | Tue (colder −7 pts) | Thu |
|---|---|---|---|---|---|---|
| v12_prew3 − v8_nd_e3 | +0.00106 ± 0.00074 | +0.00213 ± 0.00098 | −0.00402 ± 0.00304 | −0.00062 ± 0.00112 | +0.00009 ± 0.00103 | **+0.00214 ± 0.00105** |
| v7_q15 − v7_q0 | −0.00040 ± 0.00064 | −0.00183 ± 0.00085 | +0.00463 ± 0.00264 | +0.00107 ± 0.00099 | +0.00007 ± 0.00088 | −0.00092 ± 0.00093 |
| v7_q30 − v7_q0 | −0.00071 ± 0.00070 | −0.00236 ± 0.00092 | +0.00488 ± 0.00296 | +0.00131 ± 0.00112 | | |
| v8_nd_q15 − v8_nd_q0 | −0.00088 ± 0.00061 | −0.00173 ± 0.00080 | −0.00070 ± 0.00263 | −0.00055 ± 0.00100 | | |
| nn_final4 fine-tuned − train-only | +0.00041 ± 0.00073 | −0.00012 ± 0.00097 | +0.00388 ± 0.00256 | +0.00137 ± 0.00101 | −0.00092 ± 0.00101 | **+0.00186 ± 0.00106** |
| nn_final4_tr − nn_final3_tr | +0.00401 ± 0.00079 | | | | +0.00441 ± 0.00110 | +0.00359 ± 0.00114 |

Reading:
- The per-feed cold-share split and the day split disagree for prew3 and q15. The day split is the cleaner staleness contrast (same population), and it says prew3 helps more on the staler day.
- So **I do not claim that prew hurts private, or that q15 helps it**. The cold-share pattern is mainly population.
- One signal is consistent across both views: the val fine-tune helps more on colder data. It is better on Thursday (+0.0019 vs −0.0009) and on high-cold-share feeds (+0.0039).
- Mixing weight, w_trainonly vs 0.5:

  | w_trainonly | Thu | Tue | val reweighted to private |
  |---|---|---|---|
  | 0.25 | +0.00054 ± 0.00044 | −0.00050 ± 0.00044 | +0.00007 ± 0.00043 |
  | 1.0 | −0.00114 ± 0.00065 | | −0.00133 ± 0.00061 |
  | 0 | −0.00041 (Thu) | −0.00085 (Tue) | |

  Nothing is beyond 2 SE, so 0.5 stays (protocol). A 0.25 variant is the natural private hedge (#9).

### 1.4 Information-horizon / density asymmetry of the popularity inputs (`luo_asym.py`, `calib_group.py`, `young_boost.py`)
- `it_hist_*` count an item's occurrences in all history snapshots of all splits (`src/features_luo.py:3-8`).
- Train snapshots are dense (3.15M feeds over 12 days ≈ 262K/day). Val/test snapshots are a sparse exploration sample (43K feeds over 6 days ≈ 7K/day).
- An item listed after Sep 19 can only collect counts from the sparse part.

Train rows: 150K random train feeds. Age buckets come from f4 (feat_life_ knots: f4 < 9/64 ≈ 24 h, < 20/64 ≈ 6.8 d).

| rows | age < 24 h: LUO == 0 | 1-7 d: LUO == 0 | > 7 d: LUO == 0 | pos rate LUO == 0 / LUO > 0, age 1-7 d |
|---|---|---|---|---|
| train | 31.7% | 14.5% | 9.0% | 1.42% / 2.40% (ratio 0.59) |
| val | 92.2% | 72.2% | 21.3% | 1.35% / 1.77% (ratio 0.76) |
| public | 94.4% | 85.3% | 24.8% | |
| private | **97.5%** | **88.4%** | 27.5% | |

Calibration on val (actual / expected positives, expected = K·softmax(s/T), one global T):

| group | v9_e10 train-only | v9_e10 fine-tuned | nn_final4 train-only | nn_final4 OOF |
|---|---|---|---|---|
| age < 24 h, LUO == 0 | 0.964 | 0.866 | 0.934 | 0.884 |
| age < 24 h, LUO > 0 | **1.177** | 1.040 | **1.165** | 1.038 |
| age 1-7 d, LUO == 0 | 1.067 | 0.990 | 1.068 | 0.993 |
| age 1-7 d, LUO > 0 | **1.122** | 1.033 | **1.123** | 1.036 |
| age > 7 d, LUO == 0 | 0.994 | 1.018 | 1.014 | 1.022 |
| age > 7 d, LUO > 0 | 0.983 | 1.006 | 0.981 | 1.003 |

- The train stage under-rates young items that do have val/test-period popularity. On train, a young item with a count of 3 is ordinary; on val, a count of 1 from the sparse sample stands for ~37 real occurrences.
- The fine-tune corrects most of this, but over-shoots on young LUO == 0 items (0.87-0.88).
- Post-hoc test on the submission blend (nested 5-fold GroupKFold by user, 7-value grid), adding d·flag:

  | flag | nested delta |
  |---|---|
  | age < 7 d & LUO > 0 | **+0.00034 ± 0.00029** (d = 0.2 in every fold) |
  | age < 24 h & LUO > 0 | −0.00029 ± 0.00014 |
  | age < 7 d & LUO == 0 | −0.00048 ± 0.00026 |

- Verdict: a real mechanism with a small effect at the blend level.

### 1.5 Multiple positives and the listwise loss (`kdist.py`, `exp_loss.py`, log `exp_loss_300k.log`)
Positives per feed:

| split | feeds with K ≥ 2 | share of all positives in K ≥ 2 feeds |
|---|---|---|
| train | 33.4% | 61.5% |
| val | 25.5% | 51.1% |

- The current loss (`nn/model.py:513-534`) is softmax CE with target y/K, with the other positives inside the softmax denominator.
  - The gradient on positive i is p_i − 1/K: it pushes a positive *down* whenever p_i > 1/K.
  - DCG is invariant to the order among positives; it depends only on the number of negatives above each positive.
- The per-positive variant (1/K) Σ_p log(1 + Σ_neg exp(s_n − s_p)) upper-bounds log(1 + #neg above p) directly.
- Proxy test: MLP 68 → 128 → 64 → 1 on 48 precomputed features plus NaN flags, 300K train feeds from Sep 9-19, 3 epochs, 5 seeds; val ≈ 0.372 for both arms.
  - per-positive − ListNet: val **−0.00002 ± 0.00007**, holdout +0.00002 ± 0.00004;
  - K ≥ 2 feeds: +0.00009 ± 0.00009.
- Null in the proxy. The argument holds, but the two losses learn the same ranking here.

### 1.6 Train-window recency (`exp_recency.py`, log `exp_recency.log`)
Setup:
- LightGBM lambdarank, 400 rounds, 41 content/history features. Popularity and co-vis are excluded, because their coverage depends on the day.
- Training: 40K random-like feeds from either the EARLY window (Sep 6-8) or the LATE window (Sep 17-19); 3 seeds.
- Evaluation: val (Sep 22/24), and a middle control window (Sep 11-13, 20K held-out random-like feeds, roughly equidistant from both windows).

Results (LATE − EARLY):

| eval set | delta | per seed |
|---|---|---|
| val | +0.00061 ± 0.00091 | +0.0021 / −0.0005 / +0.0002 |
| mid control | −0.00121 ± 0.00088 | −0.0004 / +0.0001 / −0.0033 |

Both are consistent with "closer training data is better by ~0.001". Neither is significant, and the seed noise is large. Concept drift over two weeks is at most small, so idea #10 has low priority.

### 1.7 Other checks
- **User × item re-exposure** (`tte_ui_imp`: same user, same item, earlier train feed): 0.18% of val rows (0.12% of test rows), positive rate 1.64% vs base 1.58%. 16 val rows had an earlier positive; none is positive now. Dead.
- **Fine-tune-branch drift** (`drift_branch.py`), PSI val → public / val → private:

  | feature | PSI val → public | PSI val → private | note |
  |---|---|---|---|
  | `hour` | **0.42** | **0.51** | |
  | `it_cand_cnt_train` | 0.035 | 0.069 | non-zero 65.4% → 56.4% → 52.7% |
  | `it_tr_coec` | 0.031 | 0.060 | |
  | `it_hist_all_luo` | 0.010 | 0.024 | |
  | `seq_cv_pop` | 0.008 | 0.014 | |
  | `u_tr_*` | ≤ 0.004 | ≤ 0.004 | |

  `hour`, `it_tr_*`, `u_tr_*` and `it_cand_cnt_train` are in `n2c_pre` through `ITEM_CLEAN` (`nn/finetune.py:65-87, 146-151`).
  - feat_cal_ §0.5 recommended dropping `hour` ("useless on val ... on test it moves ... a pure risk").
  - **Protocol gap (not a leak):** `hour` is still an input of every nn_final3/nn_final4 fine-tune.
- **The aux production-order head is discarded at inference**: it is set at `nn/model.py:499`, while `nn/common.py:43-55` (`predict`) scatters only the main logit.
- **WSD continuation is not possible today**: `nn/train.py:320-328` keeps only the holdout-best state_dict. Neither the pre-cooldown weights nor the optimizer state are saved.

---------------------------------------------------------------------------------------------------------------------

## 2. Ideas (ranked)

### 2.1 (#1) Stale-val: a private-gap copy of val as the second yardstick
**Mechanism.**
- Private is 7-10 days after the last train day; val is 3-5 days.
- The NN has no item-ID memory (the ID residual was rejected), so staleness reaches it only through the time-sensitive inputs: co-visitation, train-period item/user stats, LUO popularity coverage and `prew`/`pre2` coverage. The other channel is concept drift, which 1.6 finds small.
- Rebuilding exactly those inputs for val with the train part of each source cut at **Sep 14** gives a val copy with private-like staleness *on the same feeds and users*. That removes the population confound behind the disagreement in 1.3.
- The co-visitation part already exists: `feats_v2/seq_/gap_cv.py` (cut at Sep 14, coverage 8.8% vs private 9.8%; it halved the co-vis gain in the GBDT harness, feat_seq_ §2).
- The NN needs only re-prediction, not retraining.

**How.**
1. CPU, 1-2 h: rebuild for val:
   - `seq_cv_*` with the train source cut at Sep 14;
   - `it_tr_*`, `u_tr_*`, `it_cand_cnt_train` and `tte_*` from train labels ≤ Sep 14;
   - `it_hist_*_luo` with train snapshots ≤ Sep 14 plus the unchanged val/test snapshots;
   - `prew_*` / `pre2_*` re-masked by the new LUO.
2. Write them to a separate feature dir (nn_x supports `NNX_FEATS`; `nn/` reads `data.feat_path`) and re-predict the train-only and fine-tuned checkpoints. Labels are unchanged.
3. Sanity targets: stale-val co-vis non-zero ≈ 9.8%, `it_cand_cnt_train` non-zero ≈ 53%, LUO == 0 ≈ 45% (private values from 1.1 / 1.7).

**Use (pre-declare it).**
- A change that passes the current rule (paired train-only val > 2 SE, holdout not worse) must also not lose more than 2 SE on stale-val.
- For warm/cold-trading options (prew vs pre2, ID dropout, fine-tune weight, branch groups), report the stale-val delta next to the val delta.
- It reuses the same labels, so it is a sensitivity gate. Never tune on it.

- **Cost:** CPU 1-2 h; inference ~2-5 GPU-min per checkpoint.
- **Risk:** none for overfitting; the only risk is mis-built features (check the sanity targets).

### 2.2 (#2) Remove the drifting inputs from the fine-tune branch ("n2c_pre_safe")
**What.** `n2c_pre` minus {`hour`, `it_tr_coec`(+`_rk`/`_z`/`_gap`), `it_tr_ctr`, `it_tr_pos`, `u_tr_ctr`, `u_tr_feeds`, `it_cand_cnt_train`}.

**Mechanism.**
- These are inputs of a model fit on val (Tue/Thu, 3-5-day gap) and applied to test (weekend plus Mon/Tue, 7-10-day gap).
- `hour` has PSI 0.42 / 0.51 (public is 56% 00-12 h, plus logging-outage hours, feat_cal_).
- Train-candidate coverage falls from 65% to 53% (1.7). Playbook item 9 (from eda_shift §4.2/4.6) reports that train-candidate counts turn harmful at gaps ≥ 9 days.

**Evidence that it costs nothing on val.**
- Removing the train-stat group: +0.00002 (nn_report, R2 step 1).
- Removing `hour` from the GBDT base: +0.00023 ± 0.00068 (feat_cal_).

**Validation.**
- Non-inferiority on OOF (delta > −2 SE) for 2 members.
- Plus the stale-val delta (#1), whose `it_tr_*` copy is exactly where any private harm would show.

- **Cost:** one fine-tune per member (~10 min each).
- **Expected:** 0 on val; 0 to +0.001 on private.
- **Risk:** none. It is a removal, pre-declared, and adds no choice on val.

### 2.3 (#3) WSD continuation + cooldown-branch members
**Mechanism.**
- Training length was the biggest lever (2 → 5 → 10 epochs: holdout 0.3117 → 0.3148 → 0.3167; the holdout was still creeping up at 10 epochs).
- With the (c) recipe's WSD schedule, the stable phase can be extended without restarting, and cooldowns can be branched from one stable checkpoint (Hägele et al. 2024; Wen et al. 2024).
- Branches share a basin, so their weights can be averaged into one soup (Wortsman et al. 2022). This does not work across seeds.

**What.**
1. Add `--save_pre_cooldown`: model + optimizer state at `t_cool` (`nn/train.py:236`). The checkpoint save is at `nn/train.py:328`.
2. Continuation: extend the stable phase to 15 epochs, then cool down, at half the cost of a fresh 15-epoch run.
3. Branches: 3 cooldowns (20% of the steps each) with different data-order and cooldown-mix seeds. Use them either as 3 members (prediction mean) or as one uniform weight soup. The model has LayerNorm only, so no statistics need re-estimating.

**Validation.**
- The holdout selects the horizon (as now).
- The member set is pre-declared (all branches, equal weights).
- Soup vs prediction-mean is one coarse choice, made on the holdout.
- Paired train-only val for the record.

- **Expected:** continuation +0.0005-0.0015; branch ensemble +0.0003-0.0008 over one branch. Full seeds saturate at about +0.001 for 4 members, and branches are less diverse.
- **Cost:** about 0.5 run for continuation, 0.2 run per branch.
- **Risk:** low.

### 2.4 (#4) pre2_* vs prew_* in the train stage, judged on three yardsticks
**Mechanism.**
- `prew_*` = `pre_*` with NaN where `it_hist_all_luo == 0`. The indicator `has:prew_fe` equals `LUO > 0`.
- The NaN share drifts from 37.9% (val) to 45.6% (private), so this input has a coverage shift that `pre2_*` (full coverage, pretrain_report 8c) does not.
- The evidence on private impact is contradictory (1.3: cold-share split negative, day split positive). Treat this as a comparison to run, not a known loss.

**What.**
- When `pre2` lands, run the paired test as for v12_prew3.
- Report val, holdout, the day split and stale-val (#1) for both `pre2` and `prew`.
- For nn_final5: if stale-val is not ready, keep the members pre-declared as they are (cvnd_prew members passed the rule) rather than re-deciding on cold-share arguments.

- **Expected:** 0 to +0.001 on private vs prew.
- **Cost:** 2 × 10-epoch runs (3-6 GPU-h on a shared GPU with (c)).
- **Risk:** low. Re-check the 8b-type "has content" availability bit on pre2 train rows: per-row content coverage should be the same for positives and negatives.

### 2.5 (#5) Density-matched popularity and co-visitation counts
**Mechanism (1.4).**
- Val/test snapshot occurrences are a ≈ 1/37 sample of the real interaction volume.
- They nevertheless enter `it_hist_*_luo`, and the val/test part of the co-vis source (`feats_v2/seq_/feat_covis.py:3-16`), with weight 1.
- On train rows young items mostly have dense counts. On val/test they have sparse or zero counts, and the train stage under-rates young items that have any (A/E 1.12-1.18).

**What.**
- Recompute `it_hist_*_luo` as Σ_snapshots w·occurrences, with w = 1 for train snapshots and w = ρ for val/test snapshots. ρ = (train feeds per day) / (val+test feeds per day) ≈ 37, one fixed, label-free constant; same definition for every split.
- Weight the val/test sequences of the co-vis source the same way.
- Retrain one paired `--xt cvnd` member (2 seeds).

**Validation.**
- Paired train-only val > 2 SE, holdout not worse.
- The young-item A/E table (1.4) should move toward 1 for the train-only model.
- Stale-val.

- **Expected:** +0.0003-0.001. The post-hoc flag already gives +0.0003 (n.s.); the count magnitude adds information the flag does not have.
- **Cost:** CPU ~1 h; 2 × 10-epoch runs.
- **Risk:** low (label-free, one constant). It does not touch the same-user rule: LUO subtraction is unchanged.

### 2.6 (#6) Age-conditioned ID-input dropout (a staleness simulation)
**Mechanism.**
- At test time the co-vis/LUO zeros fall on young listings: private LUO == 0 is 97.5% for age < 24 h and 88% for 1-7 d, vs 32% / 15% on train rows.
- The rejected `--xt_drop q` blanked random candidates of any age, which destroyed true signal on old items. That is the warm-feed cost (−0.0043, nn_report R2 step 2).

**What.**
- Blank the ID-based inputs (`seq_cv_*`, `it_hist_*_luo`) of a train candidate with probability p(age), using f4 → age (feat_life_ knots).
- p(age) is chosen so that the train LUO == 0 rate per age bucket matches a target rate, using label-free feature statistics: p = (target − train) / (1 − train).
  - Pre-declare the **val** rates as the target: p ≈ 0.89 (< 24 h), 0.67 (1-7 d), 0.14 (> 7 d). Then train-only val is a fair test.
  - Optionally add a second arm with private rates: 0.96 / 0.86 / 0.20.
- Do #5 first: it fixes a definition, while this simulates one.

**Validation.**
- Train-only val > 2 SE (it should now be measurable, because the target matches val).
- Stale-val.
- The holdout is reported but should not veto: it is warm by construction.

- **Expected:** 0 to +0.0015 on private. The evidence for the uniform version is mixed (1.3), so this is uncertain.
- **Cost:** a code change in the existing `xt_drop` path; 2 runs.
- **Risk:** medium (three pre-declared rates).

### 2.7 (#7) Export the production-order head as a signal
**Mechanism.**
- The aux head (K9) is trained to reproduce the production ranker's order. Distilling it gave +0.0024 (nn_report v2).
- Its output is discarded at inference (1.7). On val/test it estimates where production would have ranked the item, which may complement the main logit.

**What.**
- A predict script that also scatters `model.aux_out`.
- Single-feature NDCG on val and on the holdout.
- Then either (a) a pre-declared blend term +0.1 z(−aux), or (b) −aux as a fine-tune-branch input.

- **Validation:** paired > 2 SE with the weight fixed in advance; holdout check.
- **Expected:** 0-0.0005. The main head shares the representation, so most of it is probably already used.
- **Cost:** GPU minutes.

### 2.8 (#8) Full-val refit of the fine-tune for the test predictions
**Mechanism.**
- Test = mean of 5 fold models, each fine-tuned on 80% of val.
- With the fta recipe (fixed epochs, cosine, EMA, no early stopping), a refit on 100% of val is well defined. It adds 25% to the only exploration-labelled data.
- That data drives the cold-side re-calibration (+0.014 on cold-positive feeds for nn_final4), which matters more on the colder test (1.3: the fine-tune helps more on Thursday).

**What.** Refit each member on all val with 3-5 seeds (keeps the ensembling), same epochs and lr.

**Validation.**
- Not measurable on val directly.
- Optional nested simulation: inside each outer fold, compare the mean of 4 inner-fold models (60%) with refits on 80%, on the held-out 20%.
- Keep the fold-mean as the fallback.

- **Expected:** +0.0002-0.0005 on test.
- **Cost:** ~1 GPU-h per member set (5-8 GPU-h with the nested check).

### 2.9 (#9) Final-pick hedge
- If two final submissions are allowed, pick (a) the best val blend and (b) the blend with the best stale-val score.
- Without stale-val, the natural (b) is w_trainonly = 0.25: Thursday +0.0005 ± 0.0004, Tuesday −0.0005; the fine-tune's cold re-calibration should matter more on private.
- Cost 0.

### 2.10 (#10) Recency-weighted cooldown
- Over-sample Sep 17-19 feeds (e.g. 50% of non-exploration cooldown batches) during the WSD cooldown, so the final low-LR phase fits the most recent behaviour.
- Evidence is weak (1.6: closer training windows are better by ~0.001 in a LightGBM proxy, n.s.).
- One paired run if a slot is free. Accept only by the usual rule.

### 2.11 (#11) Per-positive softmax loss
- The math rationale is in 1.5; the proxy result is null (−0.00002 ± 0.00007 over 5 seeds).
- One flag in `listwise_loss` plus one paired NN run, only in a free slot.

### 2.12 (#12) Capacity re-probe under Muon
- d = 256 was null under AdamW (0.37271 vs 0.37385) and at 27.5k steps of the plain round-4 recipe.
- Muon raises `c_lin`'s effective rank from 30 to 79 (nnx_exp_optim §2), so width may now be usable.
- One paired 10-epoch (c) run at d = 192 or 256, holdout-selected.

---------------------------------------------------------------------------------------------------------------------

## 3. Already done or rejected (not re-proposed)
- **NN tricks:** FoX, qk-norm/SSMax, PLE, propensity head, semantic IDs, EWMA features, stage-1 EMA, LP-FT, KD inside Muon + WSD, MoE top-2, TabM, d = 256 under AdamW, GRU (similar), deeper stacks, set attention / position tower (neutral), BCE.
- **Losses and weighting:** K/IDCG weighting (K1(c), neutral), history-length TTA (K10, worse). Permutation TTA is meaningless for this permutation-equivariant model.
- **Other NN-side negatives:** test pseudo-labels (−0.0004), hashed item-ID residual (−0.0022), uniform ID-input dropout (#6 explains why the age-conditioned version differs).
- **Stackers and features:** wide GBDT stackers (overfit val), later-history boost (banned), calendar / trend / lifecycle families (noise), taste profiles and content-kNN COEC (T1/T2, null), cold offsets (null).
- **Blending:** per-segment blend weights (sign flips), per-category calibration (−0.0009). Val labels as an item prior for test: negligible.
- **Literature:** OTTO, H&M, RecSys 2022-2025, Avito, MIND, KDD Cup and ULTR with uniform data are covered in `reports/deep_*` / `research_*`. My searches found no further technique that fits this data better than the list above. The only literature items I added are WSD continuation/branching and soups (#3).

---------------------------------------------------------------------------------------------------------------------

## 4. Checked and found correct / safe, or closed
- **Calibration:** val → public with M-only weights (`src/calib.py`) is the better of the two predictors I tested. M × cold over-predicts v5 by 0.0023.
- **Mix weight:** the 0.5 / 0.5 fine-tuned / train-only weight is within 1 SE of every alternative, on val, on Thursday and reweighted to private.
- **Co-visitation source** for val/test rows is causal and skips same-user sequences (`feats_v2/seq_/feat_covis.py:3-8`).
- **LUO** removes the same user's later snapshots (`src/features_luo.py:3-6`).
- Both are consistent with the rules. My concern about them is density (#5), not leakage.
- **`tte_ui_*` re-exposure:** dead (1.7).
- **Multi-positive loss:** null in the proxy (1.5).
- **Concept drift within the train window:** at most ~0.001 (1.6).
- **Cold-flag reproduction** matches err_SYNTHESIS (28.86% / 39.26%).

## 5. Sources
- Bruch, Wang, Bendersky, Najork 2019, *An Analysis of the Softmax Cross Entropy Loss for Learning-to-Rank with Binary Relevance* (ICTIR): https://research.google/pubs/pub48321/
- Bruch 2021, *An Alternative Cross Entropy Loss for Learning-to-Rank*: https://arxiv.org/abs/1911.09798
- Multi-positive softmax variants (positives compete for probability mass): https://arxiv.org/html/2602.12727
- RecSys Challenge 2024 overview (winning features relied on leakage; a caution for post-hoc signals): https://arxiv.org/pdf/2409.20483
- Hager, Zoeter, de Rijke 2025, *Understanding Two-Tower Models for Unbiased Learning to Rank* (additive position models need randomised swaps or feature overlap; consistent with our neutral position tower): https://arxiv.org/pdf/2506.20501v1
- Cited from memory, no project data sent:
  - Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (WSD continuation and cooldowns);
  - Wen et al. 2024, *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective*;
  - Wortsman et al. 2022, *Model soups*.
