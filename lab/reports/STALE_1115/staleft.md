# staleft: STALE-FT = the val fine-tune trained with test-like stale popularity inputs (KEY = staleft)

Written incrementally on 2026-11-15 by the staleft agent. Work directory `nn_x/w12/stale/staleft/`. Conventions: +- = 1 SE paired per feed, feed count given; [M] measured by me, [D] read in a project file (path), [I] inferred. "Stale view" = val inputs replaced by the project's private-like counterfactual files (scenario `both_pre2both` of `nn_x/w5/stale_pred.py`), extended by me to the fine-tune's own extra inputs.

STATUS OF THIS FILE: complete for the pilot, the replication and the pooled W8 read (written 09:0xZ). Two detached queue processes of mine are still running the 1-seed W16 extension (section 5.1; reads are one command each; `nn_x/w12/stale/staleft/PROGRESS.md` has the PIDs).

## 0. Verdict

1. **STALE-FT is built and verified, and it does NOT pass the lead's rule.** 4-member pilot, final file against the same-script control (14,423 feeds): P = 0.5 fresh val -0.00021 +- 0.00020, stale-val +0.00024 +- 0.00019 -> FAIL; P = 1.0 fresh -0.00017 +- 0.00025, stale-val +0.00042 +- 0.00024 -> FAIL, within 1 SE of passing. A/A of the control: 0.00003-0.00005, so the effect is real for these seeds [M].
2. **Replicated on the other four W8 members and pooled over the 8 W8 members (P = 1.0): final fresh +0.00022 +- 0.00026, stale-val +0.00054 +- 0.00028 (z 1.93)**: the +2 SE bar is missed by 0.07 SE. At member level it is a clean, replicated trade: -0.00033 +- 0.00020 on fresh val, +0.00026 +- 0.00023 on the stale view [M].
3. **In the form of the team's best file (W8P20) it adds nothing on fresh val: +0.00012 +- 0.00025 against the matched 1-seed control, -0.00005 +- 0.00025 against the stored 3-seed form.** The stale fine-tune and the popularity-light admixture supply the same kind of diversity [M]. **On stale-val in that form (popularity-light students re-scored on the stale view, read 09:14Z): P = 1.0 +0.00015 +- 0.00024 (z 0.62), P = 0.5 +0.00019 +- 0.00019: the lead's rule FAILS on the form we would send.**
4. What the control shows: private-like stale inputs cost the final file -0.0017 +- 0.0005, and **the standard fine-tune half does not protect against them** (its branch loses -0.0017 as well). A fine-tune trained ENTIRELY on the stale view recovers +0.0004 .. +0.0005 of that, and this is an upper bound (trained and scored on the same synthetic view). The stale penalty is mostly lost information, not something 4 fine-tune epochs can repair [M / I].
5. The "private-like" scenario overshoots: zero share of `it_hist_all_luo` 0.379 (val) -> 0.485 (stale view) against 0.456 on private and 0.428 on public rows; against public rows the stale view is FARTHER than fresh val, and a 50 / 50 mix is the closest view. 7 of 33 column instances fail my label-free gate on magnitude (the construction removes counts; test rows have more) [M].
6. **Recommendation: close STALE-FT for the P20-form files and do not spend the 6-9 GPU-process-hours of a 3-seed replacement run.** Expected value: about zero on the best file's form, at most +0.0002 .. +0.0005 on a synthetic witness for a plain W8 / W16 file, undetectable on public (paired SE 0.0004). One thing is still open and already running on my two slots, low priority (16 jobs left, done about 10:15Z, PIDs 249332 / 254220 if the lead wants the slots): the 1-seed STALE-FT branch for the other 8 W16 members with controls; if the W16 read passes the lead's rule (command in 5.1), that branch is a candidate for the plain W16 final only, with its test predictions on disk; if not, STALE-FT is closed altogether.
7. Where the data disagree with the status files: (a) "val under-reads the fine-tune half because of 3 fold-models against 15": val at the public mix reads the half at +0.0005 .. +0.0007, the public size, with no ensemble argument, and predicts MORE on private (+0.0007 .. +0.0010, SE 0.0006); (b) the fine-tuned half of the "popularity-light" students is not popularity-light (7 of the 8 excluded columns re-enter through the fine-tune extras, fresh); (c) for a READ one fine-tune seed is enough: the 1-seed W8 file equals the stored 3-seed file on val (+0.00001 +- 0.00014); in the W8P20 form the 1-seed branch reads -0.00017 +- 0.00012, so I do not claim three seeds are wasted in a final file.

## 1. Which inputs of the fine-tune depend on staleness [M]

The fine-tune (`nn_x/w5/finetune.py`, recipe n2cpv3) feeds a W16 member two groups of per-candidate inputs on val and test rows:

| group | columns | staleness-dependent | counterfactual (private-like) val file |
|---|---|---|---|
| xt, 34 train-stage inputs (checkpoint `cfg.xt_names`) | `seq_cv_invd20`, `seq_cv_pop`, `seq_cv_norm`, `it_hist_all_luo`, `it_hist_et1..4_luo` (8) | yes (coverage of history snapshots / co-visitation) | exists: `nn_x/r6/stale_cf/both_*.npy` |
| | `pre2_fe_z`, `pre2_fe_s`, `has:pre2_fe` (3) | yes (item-ID part of the pre-trained encoder score) | exists: `solvers/out/stale/cf/pre2both_*.npy` |
| | `pre2_new_z`, `pre2_new_s`, `pre2_hmax32` (3) | yes in principle | NOT in the declared scenario (stay live; a `pre2full_*` sensitivity file exists for the first two) |
| | `ctx_nd_*` (10), `ctx_hn_*` (6), `ctx_hs_*` (4) | no (content / own-history neighbourhood) | - |
| extras, 42 inputs of preset `n2c_pre_v3` (zero-init branch, val / test only) | `it_hist_all_luo`, `it_hist_et1..4_luo`, `seq_cv_invd20`, `seq_cv_pop` (7) | yes | exists (same files as xt) |
| | `it_hist_et5..10_luo` (6) | yes | **built by me** (`build_cf.py` part A; same construction as `s08_cf_feats.py`) |
| | `it_hist_all_luo_rk`, `_zsl`, `_gap`, `seq_cv_invd20_rk` (4) | yes (within-feed transforms of the counts) | **built by me** (part B; `src/features_luo.py` / `src/features.py::_rel_feats` kernels) |
| | `pre_fe`, `pre_fe_z`, `pre_id_contrib_v2`, `pre_idmax32`, `pre_idmax32_has` (5) | yes (item-ID part of the older `pre_` encoder family) | **built by me** (part C; the `build_pre2both` construction on configs v3A -> v3, v4A -> v4) |
| | `pre_new`, `pre_new_z`, `pre_hmax32_c`, `pre_hmax8_c`, `pre_hmax32_sent`, `pre_fe_noid` (6) | `pre_new*` in principle | kept fresh, as the declared scenario keeps `pre2_new*` / `pre2_hmax32` fresh |
| | `centrality`, `seq_max_all`, `seq_top3`, `seq_argmax_hpos`, `ctx_nd_*` (14) | no | - |

So 17 of the 42 extras are popularity / recency columns that no stale-val read of the project had ever switched: every stale-val number on record is a TRAIN-ONLY read. "No fine-tuned predictions exist at stale coverage" [D `reports/STALE_1115/veto.md` item 6] was still true before this run.

Build checks (`logs/build_cf.json`, all bars written in `DECLARATION.txt` before the build) [M]:

| check | bar | result |
|---|---|---|
| A: my rebuild of `both_it_hist_{all,et1..et4}_luo` against the five existing files | max abs diff = 0 | 0.0 on all five (1,389,971 rows) -> `et5..10` written |
| B: fresh `it_hist_all_luo_rk / _zsl / _gap`, `seq_cv_invd20_rk` reproduced from the fresh parent with my copy of the kernels | <= 1e-5 | 0.0 on all four |
| C: fresh `pre_fe`, `pre_fe_noid`, `pre_fe_z` reproduced from the per-config files | <= 1e-4 | 0.0 / 0.0 / 0.0 |
| C: P(`pre_id_contrib_v2` == 0 given `it_hist_all_luo` == 0), val / test rows | >= 0.95 | 1.000 / 1.000 (a cold row never has an item-ID contribution) |
| newly cold rows (count > 0 fresh, 0 in the stale view) | - | 10.56% of val rows; 29.6% of them carry an item-ID contribution today |

## 2. Label-free gate: is the stale val view closer to test rows than the fresh val view? [M]

Criterion (fixed in `DECLARATION.txt` at 04:58:50Z, before any column was built): per column, on the model's input scale, seven listing-age buckets (those of `nn_x/w12/next/poplag/pl_common.py`), weights = val row share; dZ = weighted mean absolute difference of the zero (or missing) share, dM = the same for the mean in test SDs. A column is switched iff dZ and dM do not get worse (tolerance 0.005 on dM) against PRIVATE rows and against ALL test rows. 1,389,971 val rows; 2,017,882 private and 886,924 public test rows. No target file is opened. Full table: `nn_x/w12/stale/staleft/logs/gate.txt`.

| column (xt and extras read the same file) | zero share: fresh val -> stale val / private / public | private dZ fresh -> stale | private dM fresh -> stale | public dZ fresh -> stale | public dM fresh -> stale | result |
|---|---|---|---|---|---|---|
| `it_hist_all_luo` | 0.379 -> **0.485** / 0.456 / 0.428 | 0.080 -> 0.028 | 0.126 -> 0.079 | 0.051 -> **0.055** | 0.085 -> **0.108** | switch; OVERSHOOTS public |
| `it_hist_et1_luo` | 0.411 -> 0.515 / 0.485 / 0.458 | 0.077 -> 0.029 | 0.120 -> 0.080 | 0.049 -> 0.055 | 0.080 -> 0.107 | switch; overshoots public |
| `it_hist_et2_luo` | 0.689 -> 0.749 / 0.728 / 0.715 | 0.041 -> 0.023 | 0.073 -> 0.068 | 0.027 -> 0.032 | 0.048 -> 0.082 | switch; overshoots public |
| `it_hist_et3_luo` | 0.714 -> 0.773 / 0.751 / 0.740 | 0.039 -> 0.024 | 0.077 -> 0.068 | 0.027 -> 0.032 | 0.052 -> 0.081 | switch; overshoots public |
| `it_hist_et4_luo` | 0.749 -> 0.795 / 0.775 / 0.767 | 0.028 -> 0.022 | 0.055 -> **0.061** | 0.019 -> 0.027 | 0.038 -> 0.070 | FAILS dM by 0.001; kept (xt scenario is the declared witness; extras by amendment 1a) |
| `seq_cv_pop` | 0.420 -> 0.474 / 0.476 / 0.464 | 0.063 -> 0.033 | 0.103 -> 0.061 | 0.047 -> 0.020 | 0.074 -> 0.028 | switch |
| `seq_cv_invd20`, `seq_cv_norm` | 0.887 -> 0.890 / 0.902 / 0.898 | 0.017 -> 0.013 | 0.036 -> 0.032 | 0.012 -> 0.008 | 0.024 -> 0.020 | switch (the scenario moves it by one tenth of the gap) |
| `pre2_fe_z` (xt), `pre_fe_z` (extra) | - | - | 0.016 -> 0.012 | - | 0.011 -> 0.009 | switch |
| `pre2_fe_s` (xt), `pre_fe` (extra) | - | - | 0.034 -> 0.037 | - | 0.013 -> 0.014 | switch (inside the tolerance) |
| `it_hist_all_luo_rk` | - | - | 0.118 -> 0.034 | - | 0.081 -> 0.024 | switch |
| `it_hist_all_luo_zsl` | - | - | 0.056 -> 0.024 | - | 0.035 -> 0.010 | switch |
| `it_hist_all_luo_gap` | - | - | 0.061 -> **0.246** | - | 0.008 -> 0.292 | **FAILS: kept fresh** |
| `it_hist_et5 / et6 / et8_luo` | 0.905 -> 0.926 / 0.920 / 0.915 (et5) | 0.016 -> 0.007 | 0.046 -> 0.029 | 0.010 -> 0.011 | 0.027 -> 0.041 | switch; overshoot public |
| `it_hist_et7 / et9 / et10_luo` | 0.875 -> 0.899 / 0.886 / 0.883 (et7) | 0.012 -> 0.013 | 0.032 -> 0.050 | 0.009 -> 0.015 | 0.024 -> 0.053 | **FAIL: kept fresh** |
| `seq_cv_invd20_rk` | - | - | 0.037 -> 0.034 | - | 0.025 -> 0.017 | switch |
| `pre_id_contrib_v2` | 0.548 -> 0.579 / 0.598 / 0.579 | 0.054 -> 0.023 | 0.073 -> 0.057 | 0.038 -> 0.012 | 0.048 -> 0.032 | switch |
| `pre_idmax32_has` | 0.553 -> 0.584 / 0.601 / 0.582 | 0.052 -> 0.021 | 0.106 -> 0.043 | 0.035 -> 0.011 | 0.071 -> 0.022 | switch |
| `pre_idmax32` | same missing share | 0.052 -> 0.021 | 0.037 -> **0.042** | 0.035 -> 0.011 | 0.022 -> 0.033 | fails dM by 0.001; switched with its flag (amendment 1b) |

Zero share of `it_hist_all_luo` by listing age (fresh val / stale val / private rows / public rows): < 1 d 0.923 / 0.981 / 0.975 / 0.944; 1-3 d 0.842 / 0.948 / 0.916 / 0.878; 3-7 d 0.617 / 0.921 / 0.860 / 0.832; 7-10 d 0.409 / 0.836 / 0.787 / 0.606; 10-24 d 0.262 / 0.389 / 0.396 / 0.334; 24-76 d 0.175 / 0.217 / 0.192 / 0.185; > 76 d 0.184 / 0.230 / 0.200 / 0.197.

What the gate says, plainly:

1. **Against PRIVATE rows the stale view is closer than the fresh view on 26 of 33 column instances**; the seven failures are magnitude failures. The `both` construction REMOVES counts (test-split snapshots and train Sep 17-19) to raise the zero share, while real test rows have MORE history than val rows, not less. It emulates coverage, not level. The count gap column (`x - feed max`) shows it most clearly and is kept fresh.
2. **Against PUBLIC rows the scenario overshoots on the `it_hist` family**: zero share 0.485 against 0.428 (fresh val 0.379); dZ and dM both get worse than with fresh inputs. The status files call the scenario "private-like"; that is right, and it follows that P = 1 trains the fine-tune at a staleness BEYOND both test parts in coverage (0.485 > 0.456 > 0.428).
3. **A 50 / 50 feed mix of fresh and stale is the best label-free match of both test parts** (`gate_scen.py`, `logs/gate_scen.txt`; information, computed before any labelled read): `it_hist_all_luo` public dZ / dM: fresh 0.051 / 0.085, stale 0.055 / 0.108, **mix 0.016 / 0.045**; private: fresh 0.080 / 0.126, stale 0.028 / 0.079, **mix 0.034 / 0.069**. The same holds for `it_hist_et1 / et2 / et4`. For `seq_cv_pop` the full stale view is the best on both parts. A-priori, P = 0.5 is the arm to prefer; P = 1 is the over-dose arm.
4. Amendment 1 (written before any GPU run, label-free; `DECLARATION.txt`): two extras that miss the dM bar by 0.001 are switched anyway because the alternative creates an input pattern no real row has (`it_hist_et4_luo` different in the two branches of one model; `pre_idmax32` present with its flag at 0). `it_hist_all_luo_gap`, `it_hist_et7 / et9 / et10_luo` stay fresh as the rule says (`logs/gate_exclude.txt`).

Final stale view: xt = the 11 columns of `both_pre2both` (unchanged: it is the project's stale-val witness); extras = 18 of 42 (`it_hist_all_luo`, `_rk`, `_zsl`, `it_hist_et1..6_luo`, `it_hist_et8_luo`, `seq_cv_invd20`, `seq_cv_invd20_rk`, `seq_cv_pop`, `pre_fe`, `pre_fe_z`, `pre_id_contrib_v2`, `pre_idmax32`, `pre_idmax32_has`). 94.6% of val rows change in at least one xt column (the construction lowers the count of almost every warm item, not only of the newly cold ones).

## 3. Code [M]

`nn_x/w12/stale/staleft/code/`: frozen copies of `nn_x/w5/{common,data,blend,model,finetune}.py` (md5 in `MD5_SOURCE.txt`; `data.py`, `model.py`, `blend.py` byte-identical; `common.py` differs only in the output root: predictions and logs go to `nn_x/w12/stale/staleft/{preds,logs}`, checkpoints are read from `nn_x/w5/ckpt` or `$STALEFT_CKPT` / a per-job `CKPT=` token).

`finetune_stale.py` is GENERATED from the copy of `finetune.py` by `make_ft_stale.py` (exact once-only replacements; `diff finetune.py finetune_stale.py` = 77 changed lines, none inside the loss / optimiser / schedule / EMA code):

- `--stale_train P` (default 0). In every fine-tune step, a TRAINING val feed of the batch reads the stale view with probability P: `b["xt"]` and `b["xe"]` of those feeds are replaced by the rows of a second val view `vs` (a shallow copy of the val `SplitData` whose `xt` / `xe` are attached through the `data.feat_path` switch of `stale_pred.py`; the extras are standardised with the FRESH val statistics, exactly as test rows are). The draw uses its own generator (`777000 + 1000 fold + seed`), so the batch stream, the dropout stream and the fold split are identical in all variants. P = 0 executes no new statement inside the training step.
- Extra outputs: `nn_<out>_val_oof_stale.npy` (final EMA weights, held-out fold, stale view), `nn_<out>_val_e0.npy` / `_val_e0_stale.npy` (the starting = train-only model on the fresh / stale view; the stale one IS the project's GPU stale-val prediction of that member). `_val_oof.npy`, `_test.npy`, `_test_folds.npy` as before; test rows always read their real inputs.
- `--only_folds` (test harness only): run a subset of folds and save raw arrays; used for the identity run.
- `run_queue.py`: several fine-tunes inside ONE GPU process (one launch under the lock, the process keeps its slot); jobs are claimed by atomic `mkdir`, so two queue processes can share one list. `launch_q.sh`: the mechanism of `nn_x/w12/kd/ft_lane.sh` (blocking `flock` on `nn_x/w5/logs/locks/gpu<g>.lock`, check <= 3 compute processes and >= 60 GB free under the lock, launch, hold the lock until the process is visible).
- `test_view_cpu.py` (CPU, label-free): the stale view changes exactly the switched columns; the feed-level mix on a batch equals the stale rows for drawn feeds and the fresh rows for the others.

Leak rules: the stale view only REMOVES information from val inputs (snapshot counts); no label and no later snapshot of the same user enters (the leave-user-later-out correction `ov` of the original construction is kept). Fold discipline is that of `finetune.py` (GroupKFold by user, `--u0_singleton`): every val row, fresh or stale, is scored only by models that did not train on its user.

## 3b. Side finding from stored predictions: where the fine-tune half earns its value [M] (`ft_worth.py`, `out/ft_worth.txt`)

The status files say (CURRENT_STATUS section 1; E_kd_predeclared.txt D'): "the fine-tune half is worth +0.00048 on public although val out-of-fold reads it at +0.00009: a val row is scored by 3 fold-models, a test row by 15". The stored predictions offer a second explanation that needs no ensemble-size argument, and the data do not separate the two:

| pool (stored 3-seed files, 14,423 val feeds) | final - TO pool, plain val | at the public mix (pmix) | at the private mix (pmix) | feeds with cold share >= 45% (1,213) | 30-45% (5,077) | 15-30% (6,763) | < 15% (1,370) |
|---|---|---|---|---|---|---|---|
| W8 | +0.00009 +- 0.00043 | **+0.00066 +- 0.00054** | +0.00104 +- 0.00067 | +0.00257 +- 0.00168 | -0.00007 +- 0.00072 | -0.00031 +- 0.00059 | +0.00044 +- 0.00148 |
| W16 | +0.00014 +- 0.00041 | +0.00053 +- 0.00052 | +0.00071 +- 0.00064 | +0.00119 +- 0.00160 | +0.00041 +- 0.00070 | -0.00022 +- 0.00056 | +0.00005 +- 0.00149 |

Reading [I]: val reweighted to the public mix of (M, feed cold share) reads the fine-tune half at +0.0005 .. +0.0007, the size public showed (+0.00048, G3 +0.00047). The public mix puts 21% of its weight on feeds with cold share >= 45% (val: 8%; private: 33%), and that is where the fine-tune half reads positive. Every cell is within 1.6 SE of zero, so this is a direction, not a result. It matters here for two reasons: (a) the standing explanation "3 fold-models against 15" is not needed to explain the public reading, and should not be used to predict that the half is worth the same on private; the mix column predicts MORE on private (+0.0007 .. +0.0010); (b) it is the motive for STALE-FT in one line: the component that learns from out-of-period labels earns its value on the stale / cold feeds, and it is trained with inputs fresher than those it is applied to.

One more fact the name hides [D `nn_x/abl/logs/abl_e10_pop1kd05t25_s951_n2cpv3s3_ft.log`]: the fine-tuned half of the "popularity-light" students is NOT popularity-light. 7 of the 8 excluded popularity columns re-enter through the extras branch (plus their 10 derived columns), with fresh val values. STALE-FT therefore applies to those members as much as to the W16 members.

## 3c. Identity of the code at P = 0 [M]

| check | result |
|---|---|
| code diff | `diff code/finetune.py code/finetune_stale.py`: 77 changed lines; at P = 0 no statement of the training step runs differently (the mix block is skipped by `if a.stale_train > 0`); the extra predictions are made in eval mode and draw no random number |
| W16 member: `--stale_train 0 --seeds 3 --only_folds 0` on `w5_e10_kd05w192_s951` against the STORED 3-seed file, 2,885 fold-0 feeds (`out/ident_check_w5_e10_kd05w192_s951.txt`) | per-feed corr 0.999999 (bar 0.9995: met); NDCG mine - stored **+0.000303 +- 0.000203** (bar 0.0003: **missed by 0.000003**); top-1 differs on 0.28% of feeds |
| the same rows, starting model WITHOUT any training against the stored train-only file | per-feed corr 1.0000000, top-1 differs on 0.17% of feeds, NDCG -0.000103 +- 0.000066 |
| popularity-light student through the same script with `CKPT=nn_x/abl/ckpt`: `abl_e10_pop1kd05t25_s951` against the stored `nn_x/abl/preds` file (`out/ident_check_abl_e10_pop1kd05t25_s951.txt`) | per-feed corr 0.999999; NDCG -0.00004 +- 0.00006: both bars met |
| A/A: the P = 0 control of the 4 pilot members run twice (p0b - p0), 4-pool, 14,423 feeds | final fresh val -0.00003 +- 0.00003; final stale-val +0.00004 +- 0.00005; test FT branch per-feed corr 1.00000, top-1 changed on 0.07% (public) / 0.15% (private) of feeds |

Reading: the stored fine-tune is not bit-reproducible on the GPU (bf16 inference changes in the 3rd digit of a logit with the batch composition; the starting model alone moves 0.17% of top-1s), so "bit-identical" cannot be shown by a re-run and my NDCG bar on 2,885 feeds was set tighter than the stored file reproduces itself. By the letter the W16 identity bar is MISSED by 0.000003; the correlation clause, the second member and the A/A all say the code path is the stored one. No comparison below uses a stored fine-tune file: the control is the same script at P = 0.

## 4. Pilot: 4 members x 1 fine-tune seed, P = 0 / 0.5 / 1.0 [M] (`out/pilot_read_sft.txt`)

Members `w5_e10_kd05w192_s951`, `_s952`, `w5_e10_kd05t25w192_s951`, `_s952`; recipe n2cpv3 with `--seeds 1`; same fold split, batch stream and dropout stream in all variants. 14,423 val feeds (8,530 with M >= 50). Pools = member-equal mean of per-feed z; final = 0.5 z(FT pool) + 0.5 z(TO pool). TO on the stale view = the starting model's predictions on the stale view from the same GPU runs (equal to the stored train-only file on the fresh view: per-feed corr 1.000000, NDCG within 0.00002; against the veto agent's CPU stale-val files: per-feed corr 0.99993, member NDCG within 0.0004, the known fp32 / bf16 offset).

Levels:

| variant | FT 4-pool fresh | FT 4-pool stale | final fresh | final stale | FT member mean fresh | stale |
|---|---|---|---|---|---|---|
| train-only 4-pool | - | - | 0.40120 | 0.39964 | - | - |
| P = 0 (control) | 0.40052 | 0.39889 | 0.40136 | 0.39961 | 0.39883 | 0.39748 |
| P = 0.5 | 0.40036 | 0.39898 | 0.40115 | 0.39985 | 0.39869 | 0.39773 |
| P = 1.0 | 0.40024 | 0.39923 | 0.40119 | 0.40003 | 0.39851 | 0.39773 |
| P = 0 repeated (A/A) | 0.40050 | 0.39886 | 0.40134 | 0.39966 | 0.39880 | 0.39747 |

Three facts from the control alone:
- Private-like inputs cost the final file -0.00175 +- 0.00050 and the train-only pool -0.00156 +- 0.00050. **The standard fine-tune half does not protect against stale inputs**: its branch loses -0.00164 +- 0.00053.
- The fine-tune half is worth +0.00016 +- 0.00042 on fresh val and -0.00003 +- 0.00041 on the stale view (4 members, 1 seed).
- The stored 3-seed fine-tune of the same members reads 0.40131 as a final file; the 1-seed control 0.40136. One seed is not a handicap at pool level.

Deltas against the control, paired per feed:

| read | P = 0.5 | P = 1.0 | A/A (P = 0 repeated) |
|---|---|---|---|
| **FINAL fresh val** | **-0.00021 +- 0.00020** | **-0.00017 +- 0.00025** | -0.00003 +- 0.00003 |
| FINAL fresh, M >= 50 / M < 50 | -0.00025 +- 0.00023 / -0.00015 +- 0.00034 | -0.00008 +- 0.00031 / -0.00031 +- 0.00042 | |
| FINAL fresh, pmix-public / pmix-private | -0.00004 +- 0.00024 / +0.00001 +- 0.00029 | -0.00017 +- 0.00031 / -0.00022 +- 0.00038 | |
| **FINAL stale-val** | **+0.00024 +- 0.00019** | **+0.00042 +- 0.00024** | +0.00004 +- 0.00005 |
| FINAL stale, M >= 50 / M < 50 | +0.00014 +- 0.00024 / +0.00039 +- 0.00033 | +0.00041 +- 0.00030 / +0.00043 +- 0.00038 | |
| FINAL stale, pmix-public / pmix-private | +0.00020 +- 0.00022 / +0.00018 +- 0.00026 | +0.00047 +- 0.00028 / +0.00048 +- 0.00033 | |
| FT branch alone, fresh / stale | -0.00017 +- 0.00028 / +0.00009 +- 0.00032 | -0.00028 +- 0.00037 / +0.00035 +- 0.00040 | -0.00002 +- 0.00004 / -0.00002 +- 0.00005 |
| FT member mean, fresh / stale | -0.00015 +- 0.00015 / +0.00025 +- 0.00019 | -0.00032 +- 0.00023 / +0.00025 +- 0.00026 | |
| mix read (0.5 fresh + 0.5 stale), FINAL | +0.00002 +- 0.00014 | +0.00012 +- 0.00018 | +0.00001 +- 0.00003 |
| **RULE** (fresh >= 0 AND stale >= +2 SE) | **FAIL** (z -1.07 / +1.24) | **FAIL**; within 1 SE of passing (z -0.69 / +1.77) | - |
| TEST, FT branch vs control: per-feed corr public / private | 0.99972 / 0.99973 | 0.99903 / 0.99908 | 1.00000 / 1.00000 |
| TEST, FT branch: top-1 changed public / private | 4.39% / 4.41% | 8.63% / 8.63% | 0.07% / 0.15% |
| TEST, final file: top-1 changed public / private | 2.39% / 2.37% | 4.46% / 4.41% | 0.07% / 0.09% |

Per member (FT prediction alone, fresh / stale): P = 1.0: s951 w192 +0.00009 / -0.00043; s951 t25w192 -0.00024 / +0.00033; s952 w192 -0.00076 / +0.00082; s952 t25w192 -0.00037 / +0.00027. Three of four members trade in the expected direction; one goes the other way on both views.

Ensemble mix (declared 2d: add stale fine-tunes to the standard ones instead of replacing them; 8 FT predictions against the control doubled): p0 + p10: final fresh -0.00027 +- 0.00018, final stale-val +0.00007 +- 0.00018, mix read -0.00010 +- 0.00013. **Not supported.**

Reading [I]:
1. The effect is real for these seeds (the A/A is 5-10 times smaller) and it is a TRADE: about -0.0002 on fresh val for +0.0002 (P = 0.5) to +0.0004 (P = 1.0) on the stale view. The dose moves the stale side (p10 - p05: +0.00018 +- 0.00014) and not the fresh side (+0.00004 +- 0.00017).
2. The stale-val gain is an UPPER bound: at P = 1 the fine-tune is trained and scored on the same synthetic transformation, and that transformation overshoots both test parts in coverage and moves count magnitudes the wrong way (section 2). Even so it recovers only +0.0004 of the -0.0018 that stale inputs cost: **a 4-epoch fine-tune on 11,500 val feeds cannot teach these members to live with stale counters. The stale penalty is mostly lost information, not a mis-calibration the fine-tune could repair.**
3. On the mix read, which is the best label-free stand-in for the public rows, both doses are null (+0.00002 / +0.00012, SE 0.00014 / 0.00018). The fine-tune changes 2.4% / 4.4% of the final file's top-1s on test, equally on public and private days: a public reading could not tell this from noise (paired SE about 0.0004).

## 5. Replication on the four other W8 members, the pooled W8 read, and the file forms [M]

Declared at 07:5xZ (DECLARATION 3) before any replication job started: the same pilot on `w5_e10_kd05w192_s953`, `_s954`, `w5_e10_kd05t25w192_s953`, `_s954` (P = 0, P = 1.0; P = 0.5 if time allows), and the POOLED read on the 8 W8 members = the pool of the scored file W8. Files `out/pilot_read_sft_rep.txt`, `out/pilot_read_sft_w8.txt`.

| P = 1.0 against the 1-seed control, 14,423 feeds | pilot 4-pool (s951, s952) | replication 4-pool (s953, s954) | **pooled W8 (8 members)** |
|---|---|---|---|
| train-only pool, fresh -> stale | 0.40120 -> 0.39964 | 0.40132 -> 0.39925 | 0.40157 -> 0.39980 (-0.00176 +- 0.00052) |
| control final, fresh / stale | 0.40136 / 0.39961 | 0.40113 / 0.39944 | 0.40166 / 0.39998 |
| **FINAL fresh val** | -0.00017 +- 0.00025 | -0.00005 +- 0.00025 | **+0.00022 +- 0.00026** (z +0.87) |
| **FINAL stale-val** | +0.00042 +- 0.00024 | +0.00024 +- 0.00028 | **+0.00054 +- 0.00028** (z +1.93) |
| FINAL stale-val, pmix-public / pmix-private | +0.00047 +- 0.00028 / +0.00048 +- 0.00033 | +0.00022 +- 0.00038 / +0.00024 +- 0.00047 | +0.00024 +- 0.00036 / **-0.00006 +- 0.00044** |
| FINAL stale-val, M >= 50 / M < 50 | +0.00041 / +0.00043 | +0.00034 / +0.00009 | +0.00035 +- 0.00032 / +0.00081 +- 0.00050 |
| FINAL mix read | +0.00012 +- 0.00018 | +0.00009 +- 0.00020 | +0.00038 +- 0.00020 (z +1.86) |
| FT branch alone, fresh / stale | -0.00028 +- 0.00037 / +0.00035 +- 0.00040 | -0.00082 +- 0.00037 / +0.00016 +- 0.00042 | -0.00032 +- 0.00036 / +0.00015 +- 0.00041 |
| FT member mean, fresh / stale | -0.00032 +- 0.00023 / +0.00025 +- 0.00026 | -0.00033 +- 0.00024 / +0.00027 +- 0.00027 | **-0.00033 +- 0.00020 / +0.00026 +- 0.00023** |
| the lead's rule | FAIL (within 1 SE) | FAIL (stale z +0.85) | FAIL by 0.07 SE (fresh >= 0 holds; stale z 1.93 < 2) |
| TEST, final file: top-1 changed public / private | 4.5% / 4.4% | 4.6% / 4.3% | 4.4% / 4.2% |

Per member, P = 1.0 (fine-tuned prediction alone, fresh / stale): s953 w192 -0.00071 / -0.00023; s953 t25w192 +0.00020 / +0.00036; s954 w192 -0.00049 / +0.00079; s954 t25w192 -0.00033 / +0.00017. Over the 8 members: 6 lose on fresh val, 6 gain on the stale view.

What the three columns say together [I]:
- **Member level is a clean trade and replicates exactly**: -0.00033 on fresh val, +0.00026 on the stale view, in both halves.
- **At file level the fresh-val loss disappears as the pool grows** (-0.00017, -0.00005, pooled +0.00022): a fine-tune branch trained under other inputs is a less correlated partner for the train-only half. That is a diversity effect, the same kind the popularity-light admixture supplies.
- By my recommendation rule (fixed before the replication: pooled stale-val >= +2 SE, fresh >= -1 SE, replication sign): **not met, by 0.07 SE**. Applied as written: the 24-member x 3-seed replacement run is not recommended on this evidence.
- The stale-val gain is NOT where the private mix puts its weight: at pmix-private the pooled stale read is -0.00006 +- 0.00044; the gain sits in short feeds (M < 50: +0.00081 +- 0.00050).

**The form of the team's best file (W8P20)** (`p20_read.py`, `out/p20_read.txt`; fresh val; declared as information before it was computed):

| final file, fresh val, 14,423 feeds | level | against the stored 3-seed form | against the 1-seed control form |
|---|---|---|---|
| W8 form, stored 3-seed FT branch | 0.40165 | - | - |
| W8 form, 1-seed P = 0 branch | 0.40166 | +0.00001 +- 0.00014 | - |
| W8 form, 1-seed P = 1.0 branch | 0.40188 | +0.00023 +- 0.00026 | +0.00022 +- 0.00026 |
| W8P20 form, stored branches | 0.40214 | - (W8P20 - W8 = +0.00049 +- 0.00031: the lead's number, reproduced) | - |
| W8P20 form, W8 branch = 1-seed P = 0 | 0.40196 | -0.00017 +- 0.00012 | - |
| **W8P20 form, W8 branch = 1-seed P = 1.0** | 0.40209 | **-0.00005 +- 0.00025** | +0.00012 +- 0.00025 |

**On fresh val the stale fine-tune adds nothing to a file that already carries the popularity-light admixture** (+0.00012 +- 0.00025 against the matched 1-seed control; -0.00005 against the stored 3-seed form, which also carries the 1-seed handicap of -0.00017 +- 0.00012 that this form shows). **The same form on the stale view** (`p20_stale_read.py`, `out/p20_stale_read.txt`; rule fixed in DECLARATION 4b before the four popularity-light runs existed; W8 branch = 1-seed STALE-FT against 1-seed P = 0, popularity-light branch = its own 1-seed P = 0 run in both arms, its train-only half on the stale view from the same runs):

| W8P20 form, final file, against its control form (fresh 0.40202, stale 0.40053) | P = 1.0 | P = 0.5 |
|---|---|---|
| fresh val | +0.00001 +- 0.00025 | -0.00016 +- 0.00019 |
| **stale-val** | **+0.00015 +- 0.00024** (z +0.62) | +0.00019 +- 0.00019 (z +1.00) |
| stale-val, M >= 50 / M < 50 | +0.00045 +- 0.00029 / -0.00029 +- 0.00040 | +0.00038 +- 0.00024 / -0.00008 +- 0.00032 |
| stale-val, pmix-public / pmix-private | -0.00012 +- 0.00031 / -0.00036 +- 0.00038 | +0.00007 +- 0.00026 / -0.00008 +- 0.00032 |
| mix read | +0.00008 +- 0.00019 | +0.00001 +- 0.00014 |
| **the lead's rule on this form** | **FAIL** | **FAIL** |

This is the read I called decisive in advance: **in the form of the best file the stale fine-tune is worth +0.0000 on fresh val and +0.0002 (0.6-1.0 SE) on the stale view, and negative at the private mix.** By-product, new on record: the popularity-light admixture measured at FINAL-file level under stale inputs (control branches, W8P20 form minus W8 form): fresh val +0.00036 +- 0.00031, stale-val +0.00056 +- 0.00034, M >= 50 on the stale view +0.00111 +- 0.00035. Every earlier stale-val number of that admixture was train-only.

**P = 0.5 in the replication and in the pooled W8 read** (jobs finished 09:01Z; same files):

| P = 0.5 against the 1-seed control | pilot 4-pool | replication 4-pool | pooled W8 |
|---|---|---|---|
| FINAL fresh val | -0.00021 +- 0.00020 | +0.00002 +- 0.00016 | -0.00004 +- 0.00019 |
| FINAL stale-val | +0.00024 +- 0.00019 | +0.00043 +- 0.00022 (z 1.94) | +0.00033 +- 0.00022 (z 1.51) |
| FINAL mix read | +0.00002 +- 0.00014 | +0.00022 +- 0.00014 | +0.00014 +- 0.00015 |
| FT member mean, fresh / stale | -0.00015 / +0.00025 | -0.00018 / +0.00009 | -0.00016 +- 0.00012 / +0.00017 +- 0.00016 |
| the lead's rule | FAIL | FAIL by 0.06 SE (within 1 SE) | FAIL (within 1 SE) |
| W8P20 form, fresh val, against the 1-seed control form | | | -0.00003 +- 0.00018 |

P = 1.0 against P = 0.5 on the pooled W8 final file: fresh +0.00027 +- 0.00017, stale-val +0.00021 +- 0.00019, mix read +0.00024 +- 0.00013. **My label-free a-priori preference for P = 0.5 (section 2, point 3) is not borne out**: the full dose is at least as good on both views at pool level, and by the preference rule I wrote in advance (DECLARATION 2c: P = 0.5 unless P = 1.0 leads the mix read by more than 1 SE of the paired difference) the best variant is P = 1.0. Both doses show the same picture: a member-level trade, a small positive at file level that stays below +2 SE on the synthetic witness, and nothing in the W8P20 form on fresh val.

### 5.1 Running when this report was written: the 1-seed W16 extension (started by me, detached)

Task step 4 applies (a variant is within 1 SE of passing). A 3-seed run of 16-24 members costs 6-9 GPU-process-hours today and could not be read today; a 1-seed pool is as good as the stored 3-seed pool at file level (0.40166 against 0.40165 for W8). So I started, on my two slots and through the locked launcher, after my replication queues had exited:

| job list `jobs_w16ext.txt` (20 jobs, about 8 min each, two queue processes) | purpose | expected |
|---|---|---|
| 4 x P = 0 of `abl_e10_pop1kd05t25_s951-954` (`CKPT=nn_x/abl/ckpt`) | stale-view predictions of the popularity-light students -> stale-val read of the W8P20 form | done 09:13Z |
| 8 x P = 1.0 of `w5_e10_kd05w192_s955-958`, `w5_e10_kd05t25w192_s955-958` | STALE-FT branch for all 16 W16 members (test files on disk) | about 09:45Z |
| 8 x P = 0 of the same members | matched control for the W16 read | about 10:15Z |

Reads for the lead (rules fixed in DECLARATION 4 / 4b before the runs):

```
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python nn_x/w12/stale/staleft/p20_stale_read.py p10 p05          # W8P20 form, fresh + stale-val; lead's rule on that form -> out/p20_stale_read.txt
W16=$(for s in 951 952 953 954 955 956 957 958; do for a in kd05w192 kd05t25w192; do echo -n "w5_e10_${a}_s$s "; done; done)
STALEFT_MEMBERS="$W16" STALEFT_OUTTAG=_w16 .venv/bin/python nn_x/w12/stale/staleft/pilot_read.py p10    # W16 pool: PASS iff final fresh >= 0 AND final stale-val >= +2 SE -> out/pilot_read_sft_w16.txt
```

If the lead prefers the slots back: the two queue PIDs are in `nn_x/w12/stale/staleft/logs/PIDS.txt` (queues G and H; stop by exact PID after reading `/proc/<pid>/cmdline`); finished jobs keep their files.

## 6. Full 3-seed run: exact commands and cost (for the lead; NOT started by me)

The 3-seed run is not started because (a) no variant passes the rule, (b) by my own recommendation rule (DECLARATION 3) the pooled W8 read misses the bar (section 5), (c) at today's load it costs 6-9 GPU-process-hours and could not be read today. What I started instead is the 1-seed extension of section 5.1. If the lead wants the 3-seed run anyway, everything is in place:

```
cd <repo>
bash nn_x/w12/stale/staleft/gen_full_jobs.sh 1.0 p10      # -> nn_x/w12/stale/staleft/jobs_full_p10.txt: 24 lines = 16 W16 members, then the 8 popularity-light students (CKPT=nn_x/abl/ckpt)
                                                          #    (P 0.5: gen_full_jobs.sh 0.5 p05; a control with stale OOF: gen_full_jobs.sh 0 p0)
for g in <ids>; do                                          # one queue process per listed GPU; add GPUs to go faster (the list is shared, jobs are claimed by mkdir)
  setsid nohup bash nn_x/w12/stale/staleft/launch_q.sh F$g nn_x/w12/stale/staleft/jobs_full_p10.txt $g > nn_x/w12/stale/staleft/logs/launch_F$g.log 2>&1 < /dev/null &
done
# launch_q.sh = the mechanism of nn_x/w12/kd/ft_lane.sh: blocking flock on nn_x/w5/logs/locks/gpu<g>.lock, <= 3 compute processes and >= 60 GB free checked under the lock,
# the lock held until the process is visible. ONE process per launcher runs its jobs one after the other and keeps its slot. PIDs -> nn_x/w12/stale/staleft/logs/PIDS.txt.
# outputs: nn_x/w12/stale/staleft/preds/nn_<member>_sft3_p10_{val_oof,val_oof_stale,val_e0,val_e0_stale,test,test_folds}.npy ; logs nn_x/w12/stale/staleft/logs/<member>_sft3_p10_ft.{log,json}
CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/stale/staleft/read_full_fresh.py p10 w16      # fresh-val final-file delta against the STORED standard fine-tunes (w8 | w16 | all24)
```

Expected cost [M / I]: a 1-seed job (5 fold-models, both views scored) took 6.1-8.9 min today (mean 7.7 min over the 16 pilot jobs, 7.7-8.9 min in the replication; GPU shared by 4 processes, one of them a full-data run), i.e. about 1.5 min per fold-model against 0.35-0.65 min in the stored logs of 11-14 (`nn_x/w5/logs/*_n2cpv3s3_ft.json`: 5.2-9.9 min per 3-seed member). A 3-seed member therefore costs about 22 min today (8 min on an idle day): 16 W16 members = about 6 GPU-process-hours today (2.1 h idle), all 24 = about 9 h today (3.2 h idle); wall-clock = that divided by the number of queue processes.

To use a finished run in a file: the FT branch files are `nn_<member>_sft3_p10_{val_oof,test}.npy` in `nn_x/w12/stale/staleft/preds/` (same format and row order as `nn_<member>_n2cpv3s3_{val_oof,test}.npy`); `nn_x/w12/kd/build_r3.py` reads `preds/`, so copy them there under a NEW suffix and point `ftn` at it. `src/presub_gate.py` as for every file.

## 7. Limits, and what I did NOT check

- **One fine-tune seed per member** (the task asked for `--seeds 1` in the pilot; I kept it in the replication and the extension). The A/A shows that GPU non-determinism is negligible (0.00003-0.00005 at pool level); it says nothing about fine-tune SEED noise, because both control runs use the same seeds. All variants share seeds and batch stream, so the deltas are paired on the seed too; whether another seed gives the same sign is covered only by the member-to-member spread (4 + 4 members).
- **Stale-val is a synthetic witness and, for a stale-trained fine-tune, an in-distribution one.** The scenario overshoots the coverage of both test parts on `it_hist_*` and lowers count magnitudes where real test rows have higher ones (section 2). `pre2_new_*`, `pre2_hmax32`, `pre_new*`, `pre_hmax*` and, for the popularity-light students, `xp_imp_c50` stay fresh in it. A better stale view for the fine-tune would be built from the hz2 agent's causal construction applied to VAL rows; I did not build it.
- **No lagged-holdout read.** The fine-tune trains on val; the lagged holdout is a train-period witness for train-only members and cannot score a fold fine-tune out of sample.
- **Not run:** 3 seeds; the 16-member pool; the popularity-light students (only their identity run); P between 0.5 and 1 or above the declared two doses; a stale view of the extras alone or of xt alone (which half of the trade comes from where); a higher fine-tune learning rate or more epochs under the stale view (the present recipe was tuned on fresh val).
- **Gate criterion is mine.** It compares marginal distributions by listing age; it cannot show that the right rows are cold. Amendment 1 (two columns switched against the dM bar for structural consistency) was written before any GPU run and is label-free, but it is an amendment.
- The identity bar on the W16 member is missed by the letter (0.000003 over 0.0003); see section 3c.
- `ft_worth.py` (section 3b) is a read of stored predictions with SEs of 0.0004-0.0017; it is a direction, not a result.
- Pool reads use my own numba copies of `feed_rank_z[:, 1]` and of the pessimistic-tie NDCG of `pc_common.py`; cross-checks: the control's train-only pool on fresh val equals the stored files to 0.00002 in NDCG, and the stored 3-seed final of the 4 pilot members reads 0.40131.

## 8. Commands (all from `<repo>`; CPU commands with `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1`)

```
P=.venv/bin/python; D=nn_x/w12/stale/staleft
$P $D/list_inputs.py                       # xt names of a W16 member + the 42 extras
$P $D/build_cf.py                          # counterfactual extras -> $D/cf/both_*.npy, checks -> $D/logs/build_cf.json (20 s)
$P $D/gate.py                              # label-free gate -> $D/logs/gate.txt / gate.json ; $P $D/gate_scen.py -> $D/logs/gate_scen.txt
(cd $D/code && $P make_ft_stale.py)        # regenerates finetune_stale.py from the frozen finetune.py
$P $D/code/test_view_cpu.py                # CPU check of the stale view and of the feed mix
$P $D/code/test_main_cpu.py 0.5 1.0 0      # CPU smoke test of the whole main() (truncated work)
$P $D/ft_worth.py                          # section 3b
setsid nohup bash $D/launch_q.sh A $D/jobs_pilot.txt 2 > $D/logs/launch_A.log 2>&1 < /dev/null &    # GPU, under the lock (B: GPU N)
setsid nohup bash $D/chain_rep.sh <pid of queue A> 2 C > $D/logs/launch_C.log 2>&1 < /dev/null &    # replication after my own queue has exited (E: GPU N)
$P $D/ident_check.py w5_e10_kd05w192_s951 ; $P $D/ident_check.py abl_e10_pop1kd05t25_s951 <repo>/nn_x/abl/preds
$P $D/pilot_read.py p05 p10 p0b            # -> $D/out/pilot_read_sft.txt
STALEFT_MEMBERS="w5_e10_kd05w192_s953 w5_e10_kd05t25w192_s953 w5_e10_kd05w192_s954 w5_e10_kd05t25w192_s954" STALEFT_OUTTAG=_rep $P $D/pilot_read.py p10 p05
STALEFT_MEMBERS="<the 8 W8 members>" STALEFT_OUTTAG=_w8 $P $D/pilot_read.py p10 p05
```

## 9. Files

- `nn_x/w12/stale/staleft/DECLARATION.txt` (all rules, each time-stamped before the read it judges; 7 blocks), `PROGRESS.md`.
- Inputs: `list_inputs.py`, `build_cf.py`, `stale_map.py` (the one place that says which val file the stale view reads), `gate.py`, `gate_scen.py`; `cf/both_*.npy` (15 counterfactual extras, raw feat_cache scale, val row order); `logs/build_cf.{json,log}`, `logs/gate.{txt,json}`, `logs/gate_scen.txt`, `logs/gate_exclude.txt`.
- Code: `code/{common,data,blend,model,finetune}.py` (frozen copies, `MD5_SOURCE.txt`), `code/make_ft_stale.py`, `code/finetune_stale.py`, `code/run_queue.py`, `code/test_view_cpu.py`, `code/test_main_cpu.py`; `launch_q.sh`, `chain_rep.sh`, `gen_full_jobs.sh`, `jobs_pilot.txt`, `jobs_rep.txt`.
- Predictions (`preds/`, 2.1 GB at 09:15Z, about 2.6 GB when the extension ends): `nn_<member>_sft_{p0,p05,p10,p0b}_{val_oof,val_oof_stale,val_e0,val_e0_stale,test,test_folds}.npy` for the 4 pilot members, `..._sft_{p0,p10,p05}_...` for the 4 replication members, `nn_ident_*_partial_folds0.npz`. `val_e0_stale` of the P = 0 runs = GPU (bf16) train-only stale-val predictions of 8 W8 members, which `nn_x/w5/preds/stale/` does not hold.
- Reads: `pilot_read.py`, `ident_check.py`, `ident_extra.py`, `ft_worth.py`, `read_full_fresh.py`; `out/pilot_read_sft.txt` (+ `_interim_*`), `out/pilot_read_sft_rep.txt`, `out/pilot_read_sft_w8.txt`, `out/ident_check_*.txt`, `out/ft_worth.txt`; fine-tune logs `logs/<out>_ft.{log,json}`, queue logs `logs/queue_{A,B,C,E}.log`, launcher logs `logs/launch_*.log`, `logs/PIDS.txt`.

## 10. Compliance

- GPU use as granted: fine-tune runs only; never more than TWO GPU processes of mine (queue A PID 142825 on GPU N 07:00:11-08:10Z, queue B PID 152016 on GPU N 07:09:47-08:13Z; replication queue C PID 207887 on GPU N 08:10:02-08:56Z, queue E PID 211330 on GPU N 08:13:07-09:01Z; W16-extension queue G PID 249332 on GPU N from 08:56:31Z, queue H PID 254220 on GPU N from 09:01:11Z, STILL RUNNING when this report was closed (20 jobs, expected end about 10:15Z); each launched only after my previous process on that GPU had exited). Every launch under `nn_x/w5/logs/locks/gpu<g>.lock` (blocking flock; the first two launchers waited 6,768 s and 7,343 s behind the lead's lanes), with the check <= 3 compute processes and >= 60 GB free under the lock (3 / 92 GB, 3 / 97 GB, 3 / 97 GB, 3 / 96 GB, 2 / [n] GB, 2 / [n] GB), lock held until the process was visible. No GPU ever had more than 4 processes through me.
- Two launchers of my first, non-blocking design (PIDs 14117, 14150; no GPU process started) were stopped by exact PID after reading `/proc/<pid>/cmdline`. No other process was stopped, signalled or reniced.
- CPU: `CUDA_VISIBLE_DEVICES=""`, 12 threads, at most two heavy processes. Existing files read-only; numba caches redirected to `out/numba_cache` so nothing is written into `src/__pycache__`; outputs only under `nn_x/w12/stale/staleft/` and this report. No install, no network, no submission file.
- Leak rules: section 3. Val labels are read only to score out-of-fold predictions; test labels do not exist here; the gate and every construction are label-free.
