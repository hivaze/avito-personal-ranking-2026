# int_ft: the fine-tune stage and the final blend. What is still on the table

Written 2026-11-15 21:40 UTC by the fine-tune analyst (CPU only, stored predictions only, no training, nothing submitted). Work directory `nn_x/w12/ft_1115/` (scripts `a_inventory.py` .. `h_seed.py`, outputs `out/*.txt`, staged files `stage/`).

Marks: [M] measured here (script and output file named), [R] record (path named), [W] web source (URL), [I] judgement. Every delta is paired per feed over the 14,423 val feeds; +- is the paired standard error unless stated. FT = fine-tuned branch (`n2cpv3s3`), TO = train-only branch, file(w) = w z(FT pool) + (1 - w) z(TO pool), w = w_ft. pmix-public / pmix-private = val feeds reweighted to the public / private mix of feed size x cold share (`src/pmix.py`). "Paired public SE" = sd of the val per-feed delta / sqrt(8,562).

## 0. Verdict

1. **No gain of +0.0001 or more is left in the fine-tune weight, the fold or seed averaging, the epoch choice or a refit.** The current form (w_ft 0.5, 5 folds x 3 seeds, final EMA weights) sits on the flat top of every curve measured here. [M]
2. **The fine-tune half is worth about +0.0005 on public (+0.00051 +- 0.00030, two witnesses; P(> 0) = 0.96) and +0.0007 +- 0.0006 at the private mix.** For the current picks: FD16P20 +0.00077 +- 0.00051 (public mix) / +0.00068 +- 0.00062 (private mix); W8P20 +0.00040 +- 0.00052 / +0.00066 +- 0.00064. [M `out/f_synth.txt` F3]
3. **"Val out-of-fold under-reads the fine-tune half because 15 models are averaged on test" is not the cause.** Fold averaging is worth +0.00006 .. +0.00008 at file level (+0.00022 .. +0.00033 for the FT branch alone). The gap between val (+0.00003, mean of the two scored pools) and public (+0.00048) is reproduced by the feed mix: the same per-feed deltas reweighted to the public mix read +0.00048, and the residual that val cannot measure is -0.00007 +- 0.00060. [M F3]
4. **w_ft: the evidence supports 0.4 to 0.7 for the test file, centre 0.5 to 0.55, and no weight in that range is better than 0.5 by more than +0.0001 in expectation.** Fitted test-time optimum: FD16P20 0.54-0.56 (16-84% bootstrap range 0.39-0.74), mean of four disjoint 8-member pools 0.49-0.57 (0.35-0.77), W8P20 0.37-0.43 (0.00-0.70); fitted gain of the optimum over 0.5: +0.00000 .. +0.00003. The pools disagree in direction, which is what noise looks like. w_ft 0 costs about 0.0005 and w_ft 1 about 0.0005. [M F2]
5. **The most informative single probe is the fine-tuned-only file of the W8 pool** (public points of the same pool exist at w_ft 0 and 0.5). It is staged. Its decision value is small: the weight would move to 0.6-0.65 only if the probe reads at or above W8 - 0.00005 (probability 0.28), and the expected gain of acting on it is +0.00004. It is a free-slot probe, not a priority. [M F4]
6. **Epoch choice is not a candidate.** Epoch 1 beats the saved model at member level in 20 of 20 members of FD16P20 (+0.00101 +- 0.00011), but at file level the only per-epoch predictions on disk read epoch 1 at -0.00017 +- 0.00034 (val), +0.00001 +- 0.00042 (public mix), +0.00016 +- 0.00050 (private mix) against the saved model, and the epoch-1 shift does not yet contain the promotion of cold items. [M `out/a_inventory.txt`, `out/e_epoch.txt`]
7. **Two files are staged** (section 6): the probe of item 5 and FD16P20 at w_ft 0.60, an equal-expectation twin of the scored FD16P20 (not a gain claim). A third file was not built because nothing else in this lens can be defended.

## 1. What the fine-tune does and what is stored

Code read in full: `nn_x/w5/finetune.py` (314 lines; `nn_x/abl/popfree/finetune.py` is the same loop for the popularity-light members [R `nn_x/abl/ftbag/PREDECLARED_ftbag.txt`]); launchers `nn_x/abl/ftbag/ft_one_s3_r6.sh`, `nn_x/w12/kd/ft_w5.sh`, `nn_x/w12/kd/ft_lane.sh`.

- Recipe, identical in the logged arguments of all 32 members of W8P20 and FD16P20 on 21 keys [M `out/a_inventory.txt`, last line]: epochs 4, lr 5e-5 (extra branch x 10), weight decay 0.02, warm-up 30 steps, cosine to 0, EMA 0.99 from epoch 0.5, listwise softmax with batch normalisation, 5 folds, 3 seeds, extras `n2c_pre_v3` (42 val/test-only inputs through a zero-initialised branch), user-0 feeds as singleton groups.
- Folds: `GroupKFold(5)` without shuffle on user_id, so every member has the same split: 2,885 / 2,885 / 2,885 / 2,884 / 2,884 feeds; 13,741 users; 658 user-0 feeds; seeds 1000 k + {0, 1, 2} for fold k. 94 of 28,595 test feeds belong to a user with a val feed. [M `out/h_seed.txt`]
- Saved predictions: the final EMA weights. Val out-of-fold = mean of the 3 seeds of the one fold model that did not see the feed; test = mean raw logit of 15 models; `test_folds` = per fold, mean over 3 seeds of the per-feed z on test rows (float16).

Files that exist, per member [M `out/a_inventory.txt`]:

| pool | members | `preds/`: val_oof, test, trainonly_val, trainonly_test | `test_folds` (5 x 2,904,806) | fine-tune log + json |
|---|---|---|---|---|
| W8P20 base | 8 (`w5_e10_kd05w192`, `w5_e10_kd05t25w192`, s951-954) | all 8 | `nn_x/w5/preds/` only | `nn_x/w5/logs/` |
| W8P20 popularity-light | 4 (`abl_e10_pop1kd05t25_s951-954`) | all 4 | `preds/` and `nn_x/abl/preds/` | `nn_x/abl/logs/` |
| FD16P20 base | 16 (`w5_fd10_kd05w192`, `w5_fd10_kd05t25w192`, s951-958) | all 16 | `nn_x/w5/preds/` only | `nn_x/w5/logs/` |
| FD16P20 popularity-light | 4 (`abl_fd10_pop1kd05w192_s951-954`) | all 4 | `preds/` and `nn_x/abl/preds/` | `nn_x/abl/logs/` |

Not stored for any of the 32 members: per-seed predictions (the 3 seeds are averaged in memory), per-epoch predictions (the arrays `oof[e]`, `test[e]` are computed and dropped when EMA is on), fine-tuned weights, a one-seed (`n2cpv3`) version. The json keeps the out-of-fold NDCG by epoch (raw weights, mean of 3 seeds) and the per-fold curve of seed 0. One-seed and three-seed out-of-fold files both exist only for the 28 hard-label members of C3 (G3 + 8 popularity-light members). Per-epoch predictions of students exist only in the CPU pilot of 2026-11-14 (`nn_x/w12/recheck/ft_redesign/out/ftcpu_*_oof.npy`: 2 W8 members, folds 0-2, 1 seed).

Stored branch files used below: `preds/ens_W8P20kd_{ft,to}_{val,test}.npy`, `preds/ens_FD16P20kd_*`, `preds/ens_W8kd_*`. The code path of this report at w_ft 0.5 reproduces `subs/cand_W8kd_r2.parquet` and `subs/cand_FD16P20kd_r4.parquet` row by row, and at w_ft 0 reproduces `subs/cand_W8TOkd_r3.parquet` (share of equal rows 1.000000). [M `out/g_build.txt`]

## 2. Honest estimate of the test-time fine-tune half

FT half = file at w_ft 0.5 minus the train-only pool.

### 2.1 What val reads [M `out/b_wft.txt`, `out/d_recipe.txt` D3]

| pool | val out-of-fold | pmix-public | pmix-private | pmix-public minus val (same deltas) |
|---|---|---|---|---|
| W8 (scored pair) | +0.00009 +- 0.00043 | +0.00066 +- 0.00054 | +0.00104 +- 0.00067 | +0.00057 +- 0.00032 |
| G3, one-seed FT (scored pair) | -0.00003 +- 0.00039 | +0.00030 +- 0.00048 | +0.00060 +- 0.00059 | |
| W16 | +0.00014 +- 0.00041 | +0.00053 +- 0.00052 | +0.00071 +- 0.00064 | +0.00039 +- 0.00031 |
| W8P20 | +0.00002 +- 0.00042 | +0.00033 +- 0.00052 | +0.00059 +- 0.00064 | +0.00031 +- 0.00031 |
| FD8P20 | +0.00039 +- 0.00039 | +0.00034 +- 0.00048 | +0.00022 +- 0.00059 | -0.00005 +- 0.00028 |
| FD16P20 | +0.00058 +- 0.00041 | +0.00070 +- 0.00051 | +0.00061 +- 0.00062 | +0.00013 +- 0.00030 |
| C3 | -0.00002 +- 0.00039 | +0.00036 +- 0.00048 | +0.00072 +- 0.00059 | +0.00038 +- 0.00028 |

For the full-data pools the val read is already at the public size; for the held-out pools the feed mix adds +0.0003 .. +0.0006 (each at 1.0 to 1.8 SE).

### 2.2 The seed-averaging part (measured)

- Member level, 3 seeds against 1 on the same folds: 0.00000 +- 0.00005 over the 20 G3 members [M `out/h_seed.txt`]; -0.00000 +- 0.00004 over 28 members [R `nn_x/abl/ftbag/RESULT_ftbag.txt`].
- Pool level: FT pool alone +0.00007 +- 0.00018 (G3), -0.00009 +- 0.00016 (28 members); file +0.00018 +- 0.00012 and +0.00012 +- 0.00011. [M `out/h_seed.txt`] Public, the same two pairs: -0.00007 and +0.00006. [R `docs/04_SUBMISSIONS.md` rows 7, 16-18]
- Size of the noise: seed-noise variance of one fine-tune 0.00076 in per-feed z units; the difference of the one-seed and three-seed FT pools has variance 0.00021 on val and 0.00004 on test. The fold noise of a pool is 0.0025-0.0027 (2.3), so the seed noise left in a three-seed pool (about 0.0001 [I: 0.00021 x 3/2 / 3]) is about 4% of it. Going from 3 to 15 seed-equivalents is worth 0.0000 +- 0.0001. [M]

### 2.3 The fold-averaging part (cannot be measured on val; three estimates)

Fold noise on test rows, label-free [M `out/c_foldavg.txt` C1]:

| FT branch | r between two fold-branches | r(fold, 5-fold mean) | fold-noise variance sigma_e^2 (z units) | share of the branch variance | share of the FT-minus-TO shift variance | after 5-fold averaging |
|---|---|---|---|---|---|---|
| W8 | 0.9973 | 0.9989 | 0.00268 | 0.27% | 16.0% | 3.2% |
| W8P20 | 0.9975 | 0.9990 | 0.00247 | 0.25% | 16.3% | 3.3% |
| FD16 | 0.9974 | 0.9989 | 0.00264 | 0.26% | 16.3% | 3.3% |
| FD16P20 | 0.9974 | 0.9990 | 0.00257 | 0.26% | 16.8% | 3.4% |

The noise is the same on public and private rows (0.00261 / 0.00256 for FD16P20) and is not reduced by the pool (all members share the split; `STALE_1115/premortem.md` row 1 found the same). It explains the size ratio of the shift, test / val = 0.944-0.947 against sqrt(1 - 0.8 x 0.16) = 0.934. [M `out/b_wft.txt` headers]

Estimate A, real-noise extrapolation [M `nn_x/w12/ft_1115/c3_simex.py`, `out/c3_simex.txt`, `out/f_synth.txt` F1]. Six real fine-tune instability fields on val rows (one fine-tune on 80% minus the bag of 5 inner fold models, 3 nested members x 2 seeds, made orthogonal to the bag shift and to the pool's own shift) are scaled to lambda x sigma_e^2, added with both signs to the FT pool (12 draws per lambda, lambda 1, 2, 4), and the loss is extrapolated to lambda = -0.8.

| pool | FT branch alone (linear / quadratic fit) | file at w_ft 0.5 | file at 0.65 | file at 0.75 |
|---|---|---|---|---|
| W8P20 | +0.00026 / +0.00046 | +0.00007 / +0.00005 | +0.00012 / +0.00018 | +0.00014 / +0.00019 |
| FD16P20 | +0.00025 / +0.00038 | +0.00008 / +0.00011 | +0.00010 / +0.00007 | +0.00012 / +0.00011 |
| W8 | +0.00016 / +0.00015 | +0.00005 / +0.00007 | +0.00007 / +0.00003 | -0.00001 / -0.00028 |

SE over draws 0.00002-0.00005 (linear fit). The gain follows G1 w^2 (W8P20: measured +0.00002 / +0.00007 / +0.00012 / +0.00014 / +0.00026 at w 0.25 / 0.5 / 0.65 / 0.75 / 1.0 against +0.00002 / +0.00006 / +0.00011 / +0.00014 / +0.00026). Used below: G1 = +0.00022 (linear) .. +0.00033 (quadratic), which is +0.00006 .. +0.00008 at w_ft 0.5.

Estimate B, on record: +0.00008 .. +0.00015 at file level with synthetic noise. [R `reports/STALE_1115/premortem.md` T4c]

Estimate C, the only nested fine-tune on disk (`nn_x/solvers/preds/refit/nested_<m>.npz`, 3 hard-label members, one seed; the outer folds equal the production folds [M]). For each val feed it holds the mean of 5 inner fold models (the analogue of a test prediction) and one model fitted on the same data (the analogue of the out-of-fold prediction). [M `out/c_foldavg.txt` C2, `out/c2_extra.txt`]

| level | bag of 5 minus one model (mean of two single-model seeds) | pmix-public | pmix-private |
|---|---|---|---|
| member (mean of 3) | +0.00038 +- 0.00015 | +0.00055 +- 0.00019 | +0.00065 +- 0.00023 |
| 3-member pool, FT branch alone | +0.00035 +- 0.00021 | +0.00050 +- 0.00026 | +0.00058 +- 0.00032 |
| file at w_ft 0.5 | -0.00008 +- 0.00016 | -0.00002 +- 0.00019 | +0.00003 +- 0.00022 |
| file at w_ft 0.75 | +0.00024 +- 0.00018 | +0.00020 +- 0.00021 | +0.00027 +- 0.00025 |

The three estimates agree: averaging 5 fold models is worth +0.0002 .. +0.0005 for the FT branch alone and 0.0000 .. +0.0001 for the file at w_ft 0.5. Estimate C compares at equal total data (5 x 64% against 1 x 80%), so it does not contain the second unmeasurable part: the 15 test models together have seen 100% of val, the out-of-fold model 80%.

A first attempt (a control variate that subtracts the nested members' single-minus-bag field from the student pool, `c_foldavg.py` C3) is not used: that field contains a shrink of the common shift (correlation 0.45 with the student shift, "variance removed" 141-149% of sigma_e^2), so it lowers w_ft instead of removing noise. Its numbers are in `out/c_foldavg.txt` and carry no weight.

### 2.4 Bound from the public readings [M `out/f_synth.txt` F3, `out/c_foldavg.txt` C4]

| | W8 pair | G3 pair | pooled |
|---|---|---|---|
| val out-of-fold | +0.00009 +- 0.00043 | -0.00003 +- 0.00039 | |
| feed mix (pmix-public) | +0.00066 +- 0.00054 | +0.00030 +- 0.00048 | |
| + fold averaging (+0.00006 .. +0.00008) = estimate | +0.00072 +- 0.00054 | +0.00036 +- 0.00048 | +0.00054 +- 0.00041 |
| public reading [R `docs/04_SUBMISSIONS.md` rows 12, 19, 23] | +0.000477 +- 0.00055 | +0.000474 +- 0.00050 | +0.000476 +- 0.00043 |
| public minus estimate | -0.00025 +- 0.00077 | +0.00011 +- 0.00070 | -0.00007 +- 0.00060 |

- The two public pairs are not independent readings: the per-feed correlation of the two val deltas is 0.314, so the pooled public SE is 0.00043, not 0.00037. The public reading of the FT half is +0.00048 at 1.1 SE.
- The residual -0.00007 +- 0.00060 is everything val cannot measure (the extra 20% of data seen by the test models, the error of the reweighting, noise). One-sided 95% bounds: -0.00105 .. +0.00091. The public readings cannot bound it more tightly, and they show no missing positive part.
- Val-based estimate and public reading taken as two independent witnesses [I]: FT half on public +0.00051 +- 0.00030 (z 1.7), P(> 0) = 0.96.
- Test-time estimates for the current files: W8P20 +0.00040 +- 0.00052 (public mix), +0.00066 +- 0.00064 (private mix); FD16P20 +0.00077 +- 0.00051, +0.00068 +- 0.00062.

## 3. The fine-tune weight w_ft

### 3.1 Curves of the stored branch files, delta against w_ft 0.5 [M `nn_x/w12/ft_1115/b_wft.py`, `out/b_wft.txt`]

W8P20 (final val 0.40214; FT pool alone 0.40167, TO pool alone 0.40212):

| w_ft | val | pmix-public | pmix-private | paired public SE | test corr to the 0.5 file |
|---|---|---|---|---|---|
| 0.00 | -0.00002 +- 0.00042 | -0.00033 +- 0.00052 | -0.00059 +- 0.00064 | 0.00055 | 0.9981 |
| 0.25 | +0.00009 +- 0.00030 | +0.00013 +- 0.00037 | +0.00019 +- 0.00045 | 0.00039 | 0.9995 |
| 0.35 | +0.00020 +- 0.00024 | +0.00015 +- 0.00029 | +0.00015 +- 0.00035 | 0.00031 | 0.9998 |
| 0.60 | -0.00024 +- 0.00016 | -0.00036 +- 0.00021 | -0.00050 +- 0.00026 | 0.00020 | 0.9999 |
| 0.65 | -0.00011 +- 0.00020 | -0.00021 +- 0.00026 | -0.00030 +- 0.00031 | 0.00026 | 0.9998 |
| 0.75 | -0.00023 +- 0.00026 | -0.00037 +- 0.00033 | -0.00046 +- 0.00042 | 0.00034 | 0.9995 |
| 0.85 | -0.00033 +- 0.00032 | -0.00048 +- 0.00040 | -0.00056 +- 0.00049 | 0.00041 | 0.9991 |
| 1.00 | -0.00047 +- 0.00039 | -0.00060 +- 0.00049 | -0.00061 +- 0.00061 | 0.00050 | 0.9981 |

FD16P20 (final val 0.40271; FT pool alone 0.40190, TO pool alone 0.40213):

| w_ft | val | pmix-public | pmix-private | paired public SE | test corr to the 0.5 file |
|---|---|---|---|---|---|
| 0.00 | -0.00058 +- 0.00041 | -0.00070 +- 0.00051 | -0.00061 +- 0.00062 | 0.00053 | 0.9981 |
| 0.25 | -0.00027 +- 0.00029 | -0.00038 +- 0.00038 | -0.00034 +- 0.00046 | 0.00037 | 0.9995 |
| 0.35 | -0.00015 +- 0.00023 | -0.00028 +- 0.00031 | -0.00028 +- 0.00039 | 0.00030 | 0.9998 |
| 0.60 | -0.00011 +- 0.00016 | -0.00009 +- 0.00020 | -0.00007 +- 0.00024 | 0.00021 | 0.9999 |
| 0.65 | -0.00027 +- 0.00020 | -0.00010 +- 0.00025 | -0.00007 +- 0.00030 | 0.00026 | 0.9998 |
| 0.75 | -0.00043 +- 0.00027 | -0.00030 +- 0.00033 | -0.00029 +- 0.00040 | 0.00035 | 0.9995 |
| 0.85 | -0.00049 +- 0.00032 | -0.00040 +- 0.00039 | -0.00035 +- 0.00047 | 0.00042 | 0.9991 |
| 1.00 | -0.00081 +- 0.00040 | -0.00082 +- 0.00049 | -0.00075 +- 0.00059 | 0.00051 | 0.9981 |

The curves are not smooth (W8P20 at 0.60 against 0.65: -0.00024 against -0.00011) and the pools disagree: W8 and W8P20 lean below 0.5, FD16P20 and FD16P20vf peak at 0.5, FD8P20 leans to 0.75 (pmix-public +0.00035 +- 0.00034), which its 16-member superset does not repeat. `out/b_wft.txt` has the same tables for W8, W16, FD8P20, FD16P20vf, G3 and C3.

### 3.2 Nested estimates and the least selected read

Nested pick of w_ft (10 weights; 5 user-grouped folds x 40 fold assignments; delta against 0.5; sd over repeats) [M `out/b_wft.txt`]:

| pool | picked by | mean picked w_ft | out-of-sample val | out-of-sample pmix-public | out-of-sample pmix-private |
|---|---|---|---|---|---|
| W8P20 | plain val | 0.23 | +0.00000 (0.00012) | -0.00004 (0.00016) | -0.00007 (0.00020) |
| W8P20 | pmix-private | 0.27 | -0.00012 (0.00017) | -0.00032 (0.00021) | -0.00044 (0.00026) |
| FD16P20 | plain val | 0.48 | -0.00026 (0.00019) | -0.00028 (0.00020) | -0.00028 (0.00022) |
| FD16P20 | pmix-private | 0.51 | -0.00034 (0.00018) | -0.00050 (0.00024) | -0.00060 (0.00028) |
| FD8P20 | pmix-private | 0.74 | -0.00012 (0.00018) | +0.00014 (0.00029) | +0.00026 (0.00034) |

No nested pick is worth anything in W8P20 or FD16P20. Looks: 10 weights x 8 pools x 3 views, 3 pick criteria.

Four disjoint 8-member wide pools (held-out s951-954 and s955-958, full-data s951-954 and s955-958), mean of the four per-feed deltas [M `nn_x/w12/ft_1115/d_recipe.py`, `out/d_recipe.txt` D1]:

| w_ft | val | pmix-public | pmix-private | pools above 0 (val / public mix / private mix) |
|---|---|---|---|---|
| 0.00 | -0.00029 +- 0.00029 | -0.00051 +- 0.00037 | -0.00067 +- 0.00045 | 0 / 0 / 0 |
| 0.25 | -0.00003 +- 0.00018 | -0.00009 +- 0.00022 | -0.00008 +- 0.00027 | 1 / 1 / 1 |
| 0.35 | +0.00003 +- 0.00013 | +0.00004 +- 0.00016 | +0.00006 +- 0.00020 | 2 / 3 / 4 |
| 0.60 | +0.00002 +- 0.00010 | +0.00011 +- 0.00012 | +0.00016 +- 0.00014 | 2 / 3 / 3 |
| 0.65 | -0.00004 +- 0.00013 | +0.00006 +- 0.00015 | +0.00010 +- 0.00019 | 2 / 2 / 2 |
| 0.75 | -0.00031 +- 0.00018 | -0.00018 +- 0.00022 | -0.00011 +- 0.00026 | 0 / 2 / 2 |
| 0.85 | -0.00053 +- 0.00023 | -0.00048 +- 0.00028 | -0.00044 +- 0.00034 | 0 / 0 / 0 |
| 1.00 | -0.00077 +- 0.00029 | -0.00073 +- 0.00036 | -0.00063 +- 0.00044 | 0 / 0 / 1 |

Quadratic fit of this curve: optimum 0.38 (val), 0.45 (public mix), 0.49 (private mix); fitted gain of the optimum over 0.5: +0.00003, +0.00001, +0.00000.

### 3.3 The test-time curve and the supported range [M `out/f_synth.txt` F2]

Test-time curve = pmix curve + fold term G1 (w^2 - 0.25), G1 = +0.00022 .. +0.00033 (2.3). Fitted optimum with a user-group bootstrap (2,000 draws):

| pool | view | no fold term | with the fold term | 16-84% range | fitted gain of the optimum over 0.5 |
|---|---|---|---|---|---|
| FD16P20 | public mix | 0.50 | 0.54 .. 0.56 | 0.39 - 0.74 | +0.00000 .. +0.00001 |
| FD16P20 | private mix | 0.49 | 0.54 .. 0.56 | 0.31 - 0.79 | +0.00000 .. +0.00001 |
| mean of 4 disjoint pools | public mix | 0.45 | 0.49 .. 0.52 | 0.35 - 0.69 | +0.00000 |
| mean of 4 disjoint pools | private mix | 0.49 | 0.54 .. 0.57 | 0.36 - 0.77 | +0.00000 .. +0.00001 |
| W8P20 | public mix | 0.32 | 0.37 .. 0.40 | 0.00 - 0.70 | +0.00002 .. +0.00003 |
| W8P20 | private mix | 0.37 | 0.41 .. 0.43 | 0.00 - 0.69 | +0.00001 .. +0.00002 |

With the larger fold term of the nested members (estimate C: +0.00031 +- 0.00021 at 0.75 relative to 0.5, +0.00014 +- 0.00019 at 0.65 [M `out/c2_extra.txt`]) the test-time deltas at 0.65 / 0.75 become +0.00004 / +0.00001 for FD16P20 and +0.00020 / +0.00013 for the four-pool mean at the public mix, each with a total SE of about 0.0003.

Reading. (a) On test the fine-tuned branch is stronger than val shows, by the feed mix and by fold averaging; both move the optimum to the right by 0.05 to 0.10, from 0.45-0.50 to 0.50-0.57. (b) The curve is flat on top: the supported range for the test file is 0.4 to 0.7 and no weight in it is ahead of 0.5 by more than +0.0001 in expectation (central estimate for 0.60-0.65: +0.00005, range -0.00015 .. +0.0002 [I]). (c) Both ends are worse: w_ft 0 by 0.0005 (2.4), w_ft 1 by -0.00048 .. -0.00056 at the public mix for the four-pool mean (SE 0.00036) and -0.00058 .. -0.00066 for FD16P20 (SE 0.00049). (d) The closure of 2026-11-14 was made on val alone; with the public reading and the test-time corrections it stands. `STALE_1115/premortem.md` rule R4 reached the same result for W8 and W16 on 2026-11-15 with the public reading known; this report adds FD16P20, the four-pool read, the fold term per weight and the public bound.

### 3.4 Which public probe is most informative, and what would move the weight [M `out/f_synth.txt` F4]

The W8 pool has public points at w_ft 0 (W8TO 0.394009) and 0.5 (W8 0.394486). A third point at w_ft 1.0 gives a three-point curve on one pool with the longest lever arm.

- Forecast for the fine-tuned-only file: W8 - 0.00051, about 0.39398; predictive sd 0.00078 (prior sd 0.00058 including 0.0003 for the error of the reweighting [I]; paired public SE 0.00052).
- Posterior optimum at or above 0.6 if the probe reads at or above W8 - 0.00005, that is 0.39444 or more: probability 0.28. Then FD16P20 at 0.60-0.65 is the file to prefer, with an expected gain of +0.00005 (reading 1 sd above the forecast) to +0.00035 (2 sd above, probability 0.02).
- Posterior optimum at or below 0.4 only if it reads at or below W8 - 0.0021 (probability 0.02).
- Any reading between 0.3924 and 0.3944 leaves w_ft at 0.5.
- Expected gain of acting on the probe, averaged over readings: +0.00004.
- A probe at w_ft 0.7-0.75 has a smaller paired SE (0.00035) but a quarter of the lever arm and no third point; it identifies the slope 1.7 times less precisely [I]. A w_ft 0.6 file differs from the 0.5 file by a paired public SE of 0.00021 and an expected +0.00005: public cannot read it.

## 4. Epoch choice

Out-of-fold NDCG by epoch from the fine-tune logs, minus the train-only model, mean over members +- SE across members [M `out/a_inventory.txt`]. Epoch 1 is the first fine-tune epoch (the second number of the "ndcg by epoch" lines; the first is the train-only model).

| members | n | epoch 1 | epoch 2 | epoch 3 | epoch 4 | saved (final EMA) | mean of epochs 1-4 | epoch 1 minus saved | epoch mean minus saved |
|---|---|---|---|---|---|---|---|---|---|
| FD16P20 wide students | 16 | +0.00074 +- 0.00010 | +0.00015 +- 0.00006 | -0.00035 +- 0.00008 | -0.00045 +- 0.00010 | -0.00038 +- 0.00009 | +0.00007 +- 0.00007 | +0.00112 +- 0.00012 (16/16) | +0.00045 +- 0.00006 (15/16) |
| FD16P20 popularity-light | 4 | +0.00093 +- 0.00031 | +0.00065 +- 0.00036 | +0.00042 +- 0.00041 | +0.00033 +- 0.00046 | +0.00039 +- 0.00041 | +0.00074 +- 0.00038 | +0.00054 +- 0.00011 (4/4) | +0.00035 +- 0.00009 (4/4) |
| FD16P20 all | 20 | +0.00078 +- 0.00010 | +0.00025 +- 0.00009 | -0.00020 +- 0.00012 | -0.00030 +- 0.00013 | -0.00023 +- 0.00012 | +0.00020 +- 0.00011 | +0.00101 +- 0.00011 (20/20) | +0.00043 +- 0.00005 (19/20) |
| W8P20 all | 12 | +0.00051 +- 0.00012 | +0.00008 +- 0.00019 | -0.00007 +- 0.00021 | -0.00020 +- 0.00021 | -0.00011 +- 0.00021 | +0.00016 +- 0.00018 | +0.00062 +- 0.00015 (10/12) | +0.00026 +- 0.00009 (11/12) |

- FD16P20: the member-level argmax is epoch 1 in 20 of 20 members; over the 100 member-folds (seed 0) the argmax is the train-only model 24 times, epoch 1 54, epoch 2 12, epoch 3 6, epoch 4 4.
- The pattern is fold-specific: folds 1 and 2 lose at every epoch (fold 1: -0.0002, -0.0005, -0.0014, -0.0017), folds 3 and 4 gain +0.0017 / +0.0014 at epoch 1 and then decline.

File level, from the only per-epoch predictions on disk (2 W8 members, 8,655 feeds, file proxy = 0.5 z(z(TO pool) + member shift) + 0.5 z(TO pool)) [M `nn_x/w12/ft_1115/e_epoch.py`, `out/e_epoch.txt`]:

| | RMS shift | regression coefficient on the saved shift | correlation with the saved shift | cold-minus-warm offset | member minus saved | file proxy minus saved: val | public mix | private mix |
|---|---|---|---|---|---|---|---|---|
| epoch 1 | 0.117 | 0.64 | 0.84 | -0.0065 | +0.00044 +- 0.00045 | -0.00017 +- 0.00034 | +0.00001 +- 0.00042 | +0.00016 +- 0.00050 |
| epoch 2 | 0.146 | 0.91 | 0.97 | +0.0196 | +0.00075 +- 0.00027 | -0.00024 +- 0.00021 | -0.00032 +- 0.00025 | -0.00034 +- 0.00029 |
| mean of epochs 1-4 | 0.139 | 0.89 | 0.99 | +0.0129 | +0.00047 +- 0.00023 | -0.00006 +- 0.00018 | +0.00008 +- 0.00021 | +0.00012 +- 0.00025 |
| saved (EMA) | 0.154 | 1 | 1 | +0.0202 | 0 | 0 | 0 | 0 |

- Epoch 1 against a shrink of the saved shift to the same size: -0.00001 +- 0.00029 (val), +0.00000 +- 0.00036, +0.00007 +- 0.00044. At file level an earlier epoch is a lower w_ft (about 0.38), which section 3 prices at zero.
- The epoch-1 shift has no promotion of cold items yet (offset -0.0065 against +0.020 from epoch 2 on). That promotion is the one systematic component of the shift [R `reports/RECHECK_1114/ft_redesign.md` finding 4].

What is different from the closed row "shorter fine-tune (epochs 1-2) / nested epoch selection: CLOSED at mix level" (`reports/COVERAGE_MAP.md` line 97, hard-label era): the member-level gap is larger and uniform for students (20 of 20, +0.0010, against +0.0006 for the hard-label re-cools [R ft_redesign section 3]), and the file-level read now exists at the public and private mix. It is still zero. Nothing reopens the row. An earlier-epoch or epoch-averaged fine-tune is not a candidate; forecast at file level 0.0000 +- 0.0003. The pilot (2 members, 3 folds) cannot exclude +0.0004; a full-pool read needs new fine-tunes (section 5, row E).

## 5. Other fine-tune levers: record, forecast, cost

Cost basis [M `out/a_inventory.txt`]: one member's fine-tune (5 folds x 3 seeds) took 19.9 minutes on a GPU shared by 3-4 processes (FD16P20 members), 1.3 minutes per single fold-seed run.

| | lever | on record | what is different now | forecast at file level | GPU cost | verdict |
|---|---|---|---|---|---|---|
| A | more fine-tune seeds | member 0.00000 +- 0.00004 (28 members); pools val +0.00018 +- 0.00012, +0.00012 +- 0.00011; public -0.00007, +0.00006 [R `nn_x/abl/ftbag/RESULT_ftbag.txt`, premortem row 4] | seed noise measured at a tenth of the fold noise (2.2) | 0.0000 +- 0.0001 | 5 seeds: 4.4 lane-hours for 20 members | no |
| B | more folds, a second fold split, or a different split per member | never run | fold noise measured: a test row keeps sigma_e^2 / 5; removing all of it is a quarter of the 4/5 already removed (2.3) | +0.00002 (FT branch alone +0.00005 .. +0.00008) | 6.7 lane-hours for 20 members and a shuffle flag in `finetune.py` | no; side effect only: with a different split per member the val out-of-fold pool would read near test-time strength |
| C | fine-tune on 100% of val, epoch fixed in advance | nested: one refit against the 5-fold mean -0.00025 +- 0.00018, 0.5 / 0.5 mix -0.00002 +- 0.00012 [R `reports/r5_refit.md`]; the same arrays at pool level: bag minus one model +0.00035 +- 0.00021 for the FT branch (2.3 C) [M] | nothing; the public bound shows no missing "more data" part (-0.00007 +- 0.00060) | replace: -0.0001; add as a 16th model: 0.0000 +- 0.0001 | 1.7 lane-hours for 20 members | no; unreadable on val (no out-of-fold) and below public resolution. [W] AutoGluon documents the same: a refit single model "is usually weaker if the original was a bagged ensemble" |
| D | two-stage fine-tune (val, then val + pseudo-labelled test) | `COVERAGE_MAP.md` line 102 "pseudo-labels (test) CLOSED negative" (-0.0004, old recipe, fine-tuned single teacher [R `EXPERIMENT_REGISTRY.md` line 114]); line 285 "FT-PSEUDO CLOSED by mechanism"; line 240 re-open at P 0.02; ft_redesign V4 (anchor on test feeds, teacher = train-only pool) listed, not run, P 0.02 | the cold promotion of the FT half grows with the test day (val Tue +0.021, last test day +0.044 [R ft_redesign section 6]) and a test anchor is the only variant that acts on it; but the sign of its value is unknown, val is blind to it, and its bounded size is below 0.0002 [R same] | 0.0000 +- 0.0002; P(+0.0003 or more) 0.03 [I] | about 13 lane-hours for 20 members (`--pseudo`, 12,000 test feeds per epoch) | no. [W] self-training helps when the shift between steps is small and pseudo-labels are mostly right, and errors accumulate otherwise |
| E | earlier-epoch or epoch-mean fine-tune | section 4 | file-level read at the test mixes: zero | 0.0000 +- 0.0003; P(+0.0003 or more) 0.05 [I] | 2.7 lane-hours for 8 members (`--ema 0` saves the epoch mean) | no |
| F | a different fine-tune weight for the popularity-light students | premortem T10 (3 x 3 grid, nothing above 1 SE, no nested read); `MIX_1115/rulesA.md` "(c) Per-branch weights (B, I) add nothing" | 5 x 5 grid on W8P20 and FD16P20 with a nested read (below) | 0.0000 +- 0.0002 from the FT-branch weight | none (CPU) | no |
| G | weight-space interpolation between the train-only and the fine-tuned weights, weight averaging of fine-tunes, fine-tuning a subset of layers | `COVERAGE_MAP.md` line 303 "shrink / anchor / low-dim FT: CLOSED by bound", line 103 "LP-FT CLOSED -0.00014" | none: the fine-tune moves predictions by RMS 0.13 z (correlation 0.991), where weight-space and output-space blends coincide to first order [I]; fine-tuned weights are not stored | inside the bound of line 303 (+0.00005, 95% upper +0.0005) | 6.7 lane-hours to re-create weights | no |

Row F, measurement [M `out/d_recipe.txt` D2]. File = 0.5 z[(1 - a) z(base FT) + a z(pop FT)] + 0.5 z[(1 - b) z(base TO) + b z(pop TO)]; scored form a = b = 0.2.

- Along a (FT branch) at b = 0.2: FD16P20 +0.00005, +0.00002, 0, -0.00004, +0.00002 for a = 0, 0.1, 0.2, 0.3, 0.4 (SE 0.00013-0.00021); W8P20 -0.00024, +0.00002, 0, -0.00002, -0.00023. The popularity-light weight inside the FT branch does nothing: after the fine-tune every member has received the fresh val/test-period popularity extras, so the popularity-light students are no longer a different view there [I; the re-entry of the popularity columns through `n2c_pre_v3` is on record, `CURRENT_STATUS.md` log 2026-11-13 04:33].
- Along b (TO branch) at a = 0.2, b = 0.4: FD16P20 +0.00026 +- 0.00024 (public mix +0.00063 +- 0.00030, private mix +0.00092 +- 0.00037); W8P20 +0.00004 +- 0.00025 (+0.00009, +0.00014). Nested pick over the 25 cells: FD16P20 +0.00023 (sd 0.00014), most picked a = b = 0.4; W8P20 +0.00004 (sd 0.00012).
- This is the heavier popularity-light weight again, not a fine-tune lever. It is the closed row `COVERAGE_MAP.md` line 343 (the 0.4 twin of FD16P20 read -0.00015 on public). No new reason is brought here.

## 6. Staged files (`nn_x/w12/ft_1115/stage/`; built with `src/make_sub.py`; nothing in `subs/`) [M `out/g_build.txt`]

| file | content | sha256 | final val | against its w_ft 0.5 file: val | pmix-public | pmix-private | paired public SE | test corr | same top-1 item, public / private |
|---|---|---|---|---|---|---|---|---|---|
| `probe_W8_FTonly_wft100.parquet` | z(fine-tuned pool of W8), `preds/ens_W8kd_ft_test.npy` | `1357e4069e230b798846c77d71a7717e4ecc6401aba8aecf93751b2d71c40f10` | 0.40087 | -0.00078 +- 0.00040 | -0.00072 +- 0.00050 | -0.00053 +- 0.00062 | 0.00052 | 0.9979 | 0.893 / 0.905 |
| `cand_FD16P20_wft060.parquet` | 0.6 z(FT) + 0.4 z(TO) of `preds/ens_FD16P20kd_*` | `4a5fc1b84b0c17a21439f6577de16e27e0dce08069a102adb5783ffec48f1493` | 0.40261 | -0.00011 +- 0.00016 | -0.00009 +- 0.00020 | -0.00007 +- 0.00024 | 0.00021 | 0.9999 | 0.981 / 0.981 |

- `probe_W8_FTonly_wft100`: information only, never a pick. Forecast public 0.39398 (sd 0.00078). Reading rule in 3.4.
- `cand_FD16P20_wft060`: the upper centre of the supported range for the all-data pick. Its own curve reads -0.0001 on every view, the four-pool curve +0.0001, the fold term +0.00003; expected delta against FD16P20 +0.00003 on public and +0.00005 on private, sd 0.00015 [I]; P(above FD16P20 on public) about 0.55, P(above the team best 0.3950094) about 0.35 [I: reading sd 0.00026 = expectation sd 0.00015 and paired public SE 0.00021]. It is an equal-expectation twin. A public reading inside +-0.0004 (2 paired SE) says nothing and must not move a pick.
- Both files have 2,904,806 rows and 28,595 feeds; the builder reproduces the three scored files of the same branch files exactly (section 1). The pre-submission gate was not run here.
- A third file was not built. W8P20 at 0.35 would follow that pool's own lean (+0.00015 at the public mix, 0.5 SE) against the direction of FD16P20 and of both test-time corrections; staging opposite weights for two pools would be fitting noise.

## 7. Web sources (generic searches; no project data left the machine)

- [W] Robust fine-tuning by interpolating the weights of the starting and the fine-tuned model (WiSE-FT): better out-of-distribution accuracy at equal in-distribution accuracy on image benchmarks. https://arxiv.org/abs/2109.01903 . Applies here only as a shrink of the same shift (row G).
- [W] A later analysis of why weight-space blends can beat output-space blends out of distribution, and of their failure through over-confident fine-tuned models. https://arxiv.org/abs/2309.17230
- [W] Fine-tuning can distort pre-trained features; linear probing then fine-tuning. https://arxiv.org/abs/2202.10054 . The project's LP-FT read is -0.00014 (`COVERAGE_MAP.md` line 103).
- [W] Averaging the weights of several fine-tunes ("model soups"). https://arxiv.org/abs/2203.05482 . Here the fine-tunes of one member differ by fold data, not by hyper-parameters, and the logit mean of 15 models is already taken.
- [W] Fine-tuning a subset of layers chosen by the kind of shift. https://arxiv.org/abs/2210.11466 . The low-dimensional fine-tune is inside the closed bound (row G).
- [W] Bagged fold models against a refit on all data: the refit "will typically result in a slight accuracy reduction" and a refit model "is usually weaker if the original was a bagged ensemble". https://auto.gluon.ai/dev/api/autogluon.tabular.TabularPredictor.refit_full.html ; repeated k-fold bagging on different partitions as variance reduction, most useful on small data: https://arxiv.org/abs/2003.06505 . Matches rows B and C: the direction is right, the size here is +0.00002.
- [W] Self-training under shift: gains when consecutive domains are close, error accumulation otherwise. https://arxiv.org/abs/2002.11361 , https://arxiv.org/abs/2310.13852 . Row D.

None of these brings a mechanism that the measurements above leave room for.

## 8. Disclosures

- Looks: 10 weights x 8 pools x 3 views; 3 nested criteria x 8 pools; 25 cells x 2 pools x 3 views; 5 weights x 3 lambda x 12 draws x 3 pools; 4 probe forms. No number is quoted as a gain.
- The control variate of `c_foldavg.py` C3 failed as a method (2.3) and is not used.
- The pmix reweighting is the project's veto lens, not a promoter; here it is used to explain a public reading, and the model allowance of 0.0003 in 3.4 is a judgement.
- The nested members of estimate C are hard-label members with one fine-tune seed; the students use 3 seeds, so C is an upper variant for the students.
- Not checked: the file-level effect of an earlier epoch on a full pool (no per-epoch predictions), anything about the fine-tune on private days beyond the label-free size of the shift, the pre-submission gate of the staged files.
- Side effect: the numba cache of `src/stack_val.feed_rank_z` in `src/__pycache__/` may have been refreshed by these runs (shared by every agent that calls it). No file under `preds/`, `subs/`, `data/`, no script and no status file was written.

## 9. Commands

```
cd <repo>/nn_x/w12/ft_1115
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
PY=<repo>/.venv/bin/python
$PY a_inventory.py   # out/a_inventory.txt   stored files, logged arguments, curves by epoch
$PY b_wft.py         # out/b_wft.txt         w_ft curves, nested picks, bootstrap
$PY c_foldavg.py     # out/c_foldavg.txt     fold noise on test rows, nested bag against single, the two public pairs
$PY c3_simex.py      # out/c3_simex.txt      real-noise extrapolation of the fold-averaging gain
$PY d_recipe.py      # out/d_recipe.txt      four disjoint pools, per-branch popularity-light weight, feed-mix part
$PY e_epoch.py       # out/e_epoch.txt       per-epoch pilot arrays re-read
$PY f_synth.py       # out/f_synth.txt       fold term, test-time curves, public bound, value of a probe
$PY h_seed.py        # out/h_seed.txt        folds, seed-averaging part
$PY g_build.py       # out/g_build.txt       staged files and path checks
```
