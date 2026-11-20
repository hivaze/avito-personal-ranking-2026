# FULLDATA: which data the finals do not train on, and a verified code path for full-data twins

Written incrementally from 2026-11-15 02:50 UTC. Key = fulldata. CPU only; no GPU run, no submission file, nothing launched on the lanes.
Working directory `nn_x/w12/stale/fulldata/`. Marks: [M] measured here, [D] read in a project file (path given), [I] inferred.
First run 02:18-03:22 UTC (cut by an account limit), resumed 03:31 UTC and finished.

## Verdict

1. **What the finals do not train on [M].** Every final member (16 W16 + 8 popularity-light) leaves out exactly the 30,000 in-period holdout feeds (0.95% of the train feeds, 1.88% of the candidates, 18.3% of the exploration-like feeds) and keeps the holdout-selected checkpoint. Val labels enter only through the fine-tune half, where each of the 15 models sees 80% of val and every label is in 12 of the 15. Nothing else is left out.
2. **The code path is ready and proven.** `train_full.py` / `train_full_abl.py` + `--holdout_in_train`. Flag off = the original bit for bit (syntax-tree proof and three end-to-end CPU pairs: from scratch, re-cool, popularity-light). Flag on: five smoke runs pass every pre-declared item (holdout feeds counted in the batches and in the pool, target = the teacher-file rows, which are the teachers' out-of-sample predictions, 72 -> 73 steps as computed, last-step checkpoint, files named IN-SAMPLE).
3. **"Trained on all data" is the full 10-epoch retrain only** (stages F / P; 1.15 GPU-h per member; the 20 members of today's two finals = 23 GPU-h, then 20 fine-tunes). A re-cool twin (0.25 GPU-h) sees the feeds in 2 of 10 passes and 8% of the learning-rate mass: a cheap probe of the path, not the owner's sentence.
4. **Expected effect: +0.00005 to +0.0002 on val [I], against a 16-pool seed noise of 0.00030 [M]: not measurable.** The price is that a twin has no out-of-sample holdout number (the originals stay on disk). It is done because the owner asks for it, not for score.
5. **Fine-tune half: keep n2cpv3s3.** Row 99 compared ONE model on 80% of val with FIVE on 64% (-0.00025 +- 0.00018, reproduced; pool level -0.00017 +- 0.00025): nothing on record says that more val per model helps, and a full-val model has no out-of-fold witness. `finetune_full.py --full_val` is ready and proven if it is wanted anyway; the 6-part nested test is designed (4 to 8 GPU-h).
6. **Launch and accept.** `lane_full.sh` (the GPU guard of `lane.sh`; DRY output in 6.2; 48 of 48 commands equal the members' logged arguments), then `accept_twins.py check F`: A1 structure, A2 in-sample holdout lift >= +0.0043 (calibrated: an original W16 member reads 0.346 on feeds it trained on and 0.322 on the holdout), A3 |twin 16-pool minus W16 @last pool| <= 0.00090 on val. First step: a 3-minute `FD_SMOKE=1` run on a GPU, because nothing here ran on a GPU.
7. **By-product for the re-cool arms [M].** The A/A of the re-cool harness (6 ref2 / ref pairs, cooldown seed changed) is as noisy at pool level as independent seeds: 0.00032 at 16 members against 0.00030. Sharing the plateau snapshot buys no precision on val; for same-seed arms such as hz this is an upper bound of their noise, not a measurement of it.

## 1. What the current finals train on (facts)

All numbers of this section are [M] from `facts.py` (members' own logs, checkpoints and prediction files) and `teacher_holdout.py`, unless marked.

| fact | value |
|---|---|
| members checked | 16 W16 members (`w5_e10_kd05w192_s951-958`, `w5_e10_kd05t25w192_s951-958`) + 8 popularity-light students (`abl_e10_pop1kd05t25_s951-954` d128, `abl_e10_pop1kd05w192_s951-954` d192) |
| arguments, all 24 | `--holdout 30000 --select_on holdout`, `--keep best`, `--train_feeds 0`, `--val_mix 0`; log line "holdout: 30000 random-like train feeds excluded from training" in all 24 |
| training feeds | 3,121,208 of 3,151,208; steps/epoch 9,835; total 98,350; cooldown from step 78,680; cooldown exploration pool 134,118 feeds |
| the excluded 30,000 feeds | 0.952% of train feeds; 4,630,966 candidate rows = 1.877% of train candidates (mean M 154 vs 78); 33.4% of the 89,794 feeds with p_random > 0.9 and M >= 50 (the draw set); 18.28% of the 164,118 exploration-like feeds (p_random > 0.5) |
| kept checkpoint | the holdout-selected one: steps 92,500 (2 members), 97,500 (14), 98,350 = last (8) |
| selected minus @last, W16 members | holdout +0.00011 (min 0, max +0.00062); val +0.00012 (min -0.00011, max +0.00065) |
| selected minus @last, W16 train-only 16-pool | holdout +0.00011 +- 0.00009 (n 30,000; 0.32460 vs 0.32449); val -0.00013 +- 0.00016 (n 14,423; 0.40144 vs 0.40157) |
| selected minus @last, 8 popularity-light members | holdout +0.00012; val -0.00009 |
| val labels in the train-only half | none (no `--val_mix`, no val file in any argument; the teachers have none either, section 2.4) |
| val labels in the fine-tune half | `nn_x/w5/finetune.py`, recipe n2cpv3s3 (log of `w5_e10_kd05w192_s951_n2cpv3s3`): GroupKFold(5) by user (658 user-0 feeds as singletons) x 3 seeds, lr 5e-5, 4 epochs, cosine, EMA 0.99 from 0.5 epochs, 42 extra features; fit sizes 11,538-11,539 of 14,423 val feeds (80.0%) |
| what a test prediction of the FT half is | the mean of 15 models, each fitted on 80% of val; every val label is in the fit set of 12 of the 15; no single model sees all of val |

The task text quoted "+0.00005..+0.00009" for selected minus @last on the holdout; measured on the 16 W16 members it is +0.00011 at both member and pool level. On val the @last pool is not worse (+0.00013 +- 0.00016 in its favour), so a twin that keeps the last step gives up nothing measurable.

The inventory of labelled data, as one table [M]:

| labelled data | feeds | train-only half of a final member | fine-tune half |
|---|---|---|---|
| train, outside the holdout | 3,121,208 | all of them, 10 passes | only through the starting checkpoint |
| train, the in-period holdout | 30,000 (4,630,966 candidates) | NOT trained on; used to select the checkpoint and as the second witness | not used |
| val | 14,423 (1,389,971 candidates) | NOT trained on | each of the 15 models: 80% (11,538 or 11,539 feeds); every label is in 12 of the 15 |
| test | 28,595 | no labels; inputs at prediction only | no labels (no pseudo-labels in the recipe: `--pseudo ''`) |

Nothing else is left out: every one of the 3,151,208 train feeds and of the 14,423 val feeds has at least one positive, so no feed has an empty listwise target; no member uses `--train_feeds` or `--val_mix`. Two smaller facts of the same kind: in the cooldown 2% of the regular batches are replaced by pool batches (about 393 of 19,670 steps), so in the last two passes a random 2% of the feeds are skipped once; and the 25 teacher members behind `teacher_mean11 / 25` exclude the same 30,000 feeds, which a full-data student twin does not change (section 2.4: on those rows the student gets an out-of-sample teacher). A final file therefore trains its train-only half on 98.60% of the 3,165,631 labelled feeds of the competition; a full-data twin raises that to 99.54%, and val stays with the fine-tune half by design.

Resume check (03:33 UTC, `recheck_sample.py`, an independent re-derivation on a sample before the cached numbers of the first run were trusted): holdout rule -> 30,000 feeds / 4,630,966 rows / 0.952% / 1.877% / 18.28%; teacher files on 500 random holdout feeds: finite, none constant, max abs difference to the mean of the members' own holdout files 2.9e-05 (mean11) and 6.8e-06 (mean25); W16 16-pool val selected 0.40144 / @last 0.40157, -0.00013 +- 0.00016 (n 14,423); three random member logs: "holdout: 30000 ... excluded", 3,121,208 feeds, 9,835 steps per epoch. All equal to the first run's numbers.

On record [D]:

| record | what it says | source |
|---|---|---|
| row 99 "refit on full val: -0.00025 +- 0.00018" | ONE model fine-tuned on 80% of val against the mean of FIVE models on 64% each (nested inside the production folds; 3 hard-label members; single-seed n2cpv3). It is not a test of 15 models on 80% against 15 on 100%. Reproduced exactly in section 5.1 | `reports/ROUND5.md` section 2b, `reports/r5_refit.md`, `reports/COVERAGE_MAP.md` row 99 |
| row 148 "valcd = the fine-tune's value" | val labels replayed in the cooldown (5 user folds, OOF): 4-root pool vs FT pool +0.00014 +- 0.00054; vs the m02 reference pool +0.00094 +- 0.00049 | `nn_x/w5/logs/w5_eval_valcd.txt` |
| row 284 "holdout + val in the cooldown pool" | closed by arithmetic: the number of pool draws is fixed (2% of the cooldown steps), a larger pool adds no draws; about +0.00004 val | `reports/COVERAGE_MAP.md` row 284, `reports/SOLVER_POWER_1113.md` |
| public value of the FT half | +0.00048 +- 0.00055 for students (W8 vs W8TO), +0.00047 for G3 | `nn_x/w5/logs/E_kd_predeclared.txt` block 2026-11-15T02:15:30Z |
| "full-data refit of the finals" | dropped 11-14: "+0.0001 expected, unmeasurable, never-executed code path, removes the second witness". The owner's instruction of 11-15 re-opens it; this report removes the "never-executed code path" part | `reports/RECHECK_1114/SYNTHESIS.md` section 5 |

## 2. The code path: `train_full.py`, `train_full_abl.py`, `cd_full.py`

`nn_x/w12/stale/fulldata/train_full.py` is a copy of `nn_x/w5/train.py` (md5 e4b5f0d7..., frozen as `ref/train_w5.py.txt`), `train_full_abl.py` a copy of `nn_x/abl/train.py` (md5 1317bdae..., `ref/train_abl.py.txt`). Both carry the same patch, line for line: one flag `--holdout_in_train` and the helper module `cd_full.py`. Nothing outside my directory was edited.

### 2.1 What the flag does (code read, and observed in the smoke runs of 2.3)

| step of train.py | flag absent (= original) | `--holdout_in_train` |
|---|---|---|
| holdout draw (`p_random > 0.9`, `M >= 50`, rng 12345), `hold`, `hold_rows`, `p_random`, domain token | as today | identical (the draw happens before the first guarded line) |
| training feed list | `feeds = np.setdiff1d(feeds, hold)` | the list before that removal is put back (`FullData.put_back` asserts that it is exactly "before minus holdout" plus the holdout) |
| feature statistics of a from-scratch run, epoch batches, cooldown pool | built from `feeds` | built from `feeds`, which now contains the holdout: no separate code path |
| cooldown pool | `exploration_pool(feeds, p, 0.5, hold)` with its leak guard | `exploration_pool(feeds, p, 0.5, None)`: the holdout feeds are in the pool (all 30,000 have p > 0.9) |
| distillation target of a holdout feed | not used | `(1 - kd_w) y / sum(y) + kd_w softmax(teacher[ci] / T)`: the same expression as for every feed, `ci` = the feed's own candidate rows |
| K9 aux weight, domain token of a holdout feed | not used | `1 - p_random` (< 0.1) and `[1 - p, p, 0]`, as for any exploration-like training feed |
| holdout NDCG of an eval | `holdout_ndcg x` in the eval line, used for selection | logged on its own line "FULLDATA step N: holdout IN-SAMPLE ndcg x (...)"; the regular field, the history entry and the `--save_steps` snapshot field are nan |
| checkpoint kept | best holdout NDCG among evals at step >= `--select_min_step` | the LAST step: `a.keep = 'last'` and the selection score is -inf at every eval; `--time_limit` refused |
| holdout prediction files | `preds/holdout/nn_<v>_holdout.npy`, `nn_<v>@last_holdout.npy` (`nn_<v>_holdout_hz.npy` with `--hzcd`) | only `nn_<v>_holdoutINSAMPLE.npy`, `nn_<v>@last_holdoutINSAMPLE.npy` (`nn_<v>_holdoutINSAMPLE_hz.npy` with `--hzcd`); extra summary `logs/<v>_fulldata.json` |
| val / test predictions, `--save_last`, `--save_steps`, cliplog | as today | as today |
| refused with the flag | - | `--xfold`, `--xfold_unlab`, `--val_mix`, `--hcap`, `--rec_q`, `--rho_file`, `--trunc_off`, `--prand2`, `--only_random`, `--time_limit`, any `--holdout` other than 30000 (they use the holdout as a held-out set or redefine the feed list; none is in a final recipe) |

The whole patch, without the docstring paragraph (`logs/patch_w5.diff`; the abl patch is the same):

```
> from cd_full import FullData
>     p.add_argument("--holdout_in_train", action="store_true", help=...)
>     fd = None
>             if a.holdout_in_train:
>                 fd = FullData(a, feeds, hold, hold_rows, tr, log)          # before feeds = np.setdiff1d(feeds, hold)
>             if fd is not None:
>                 feeds = fd.put_back(feeds, log)                             # after the "holdout: ... excluded" log line
>     if fd is not None:
>         fd.steps_note(epoch_batches, total, a, tr, log)
<         pool = exploration_pool(feeds, prand_np, a.pool_p, hold)
>         pool = exploration_pool(feeds, prand_np, a.pool_p, hold if fd is None else None)
>         if fd is not None:
>             fd.pool_note(pool, log)
>             if fd is not None:
>                 ndh = fd.insample(step + 1, ndh, log)
>             if fd is not None:
>                 sel = -math.inf
<         last_sd = save_last(a, model, va, tr, hold, hold_rows, ...)
>         last_sd = save_last(a, model, va, tr, hold if fd is None else None, hold_rows, ...)
>         if fd is not None:
>             fd.save_insample(f"{a.variant}@last", model, tr, PREDS, predict, ndcg, log)
<     if a.save_holdout:
>     if a.save_holdout and fd is None:
<     if a.hzcd:
>     if a.hzcd and fd is None:
>     if fd is not None:
>         fd.finish(a, model, tr, tr_hz, best[2], total, PREDS, LOGS, predict, ndcg, log)
```

Two cosmetic facts of a flag-on log: the original line "holdout: 30000 random-like train feeds excluded from training" is still printed (it is an unchanged statement) and is followed at once by "FULLDATA: 30000 holdout feeds (... candidates) put back: training feeds 3121208 -> 3151208"; and the first log line shows the arguments as typed (`keep: 'best'`), while the checkpoint and the json carry `keep: 'last'`, `holdout_in_train: True`.

### 2.2 Flag absent = the original, bit for bit

Static proof, `test_full.py` (40 checks, all PASS, `logs/test_full.log`), for BOTH copies:

| check | result |
|---|---|
| frozen reference == `ref/md5.txt` == the live `nn_x/w5/train.py` / `nn_x/abl/train.py` | PASS (first at 02:56 UTC; again at 03:33 and 04:04 UTC: the four live train / fine-tune files still equal the frozen copies) |
| AST(copy minus the FULLDATA statements) == AST(reference), module docstring dropped | PASS; the stripped set is exact and counted: 1 import, 1 `add_argument`, 1 `fd = None`, 1 `if a.holdout_in_train:`, 7 `if fd is not None:` (none with an else), 2 `hold if fd is None else None` -> `hold`, 2 `X and fd is None` -> `X` |
| textual diff | 35 added lines (12 of them the docstring paragraph), 4 rewritten lines that equal the old line once the `fd` conditional is removed, 0 deleted |
| parser | `--holdout_in_train` defaults to False; the 88 other defaults equal the reference parser's; the flag changes no other argument |
| `fd` | 2 assignments (`None`; `FullData(...)` inside `if a.holdout_in_train`), 17 reads, all inside the stripped statements; no module of nn_x/w5 or nn_x/abl imports `cd_full`; no file of my directory shadows a harness module |
| `FullData` unit behaviour | `put_back` restores the list (all feeds, or a 20,000 subsample); every refused flag raises; `insample()` returns nan |

End-to-end proof, rule R1 of `PREDECLARED.txt` (written 02:34 UTC, before the comparison): the original file and the copy without the flag, run as `__main__` on the CPU by `cpu_run.py` on the lane's smoke configuration (`DRY=1 W5_SMOKE=1 bash nn_x/w5/lane.sh 0 "E:kd05w192:951"` = the W16 member recipe + `--train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25 --cliplog 10`), 72 steps, 3 evals, 12 threads, deterministic kernels:

| R1 item, A_orig_E vs B_copy_E_off | result |
|---|---|
| (a) logs line for line (wall-clock fields removed; the copy's args line has the extra key `holdout_in_train: False`) | identical, 23 lines |
| (b) feed ids of every training batch, flat candidate rows, listwise targets, per-step main loss (float64), train-mode logit checksums | identical (72 steps; loss 1.324003 first, 1.308992 last) |
| (c) pre-clip gradient norm of every step, main / aux split rows | identical |
| (d) `ckpt/<v>.pt`, `ckpt/<v>@last.pt` | all 153 state-dict tensors bit-identical; cfg equal; args equal up to the new key |
| (e) prediction files (val, test, holdout; kept and @last), `holdout_meta`, `holdout_y`, summary json | identical |

The same rule on the popularity-light harness (`nn_x/abl/train.py` against `train_full_abl.py` without the flag; the wide student's command of `nn_x/w12/kd/launch_pop1w.sh` seed 951 + the smoke settings; 72 steps): G_orig_P vs H_copy_P_off: logs (25 lines), training batches, targets, per-step loss (1.325842 first, 1.337553 last), gradient norms, both checkpoints (153 tensors), all prediction files and the json identical: ALL PASS. The re-cool form has its own pair in section 4. Three pairs, three PASS.

Limits of the end-to-end runs, stated once: they are CPU fp32 runs (the GPU lanes use bf16 autocast and TF32), so the numbers are comparable only between CPU runs; and `FD_EVAL_CAP=1000,1500,1500` scores only the first 1,000 val / 1,500 holdout / 1,500 test feeds of each predict call (an uncapped run needs about 66 minutes on 12 threads), so the logged NDCG values of the smoke runs are not real NDCGs and the "identical predictions" statement covers those rows. The checkpoints are identical in full, and predictions are a deterministic function of the checkpoint through unchanged code.

### 2.3 Flag present: what was verified

Rule R2 of `PREDECLARED.txt`. `check_e2e.py flagon`, all items PASS (`logs/check_e2e.txt`):

| R2 item | C_copy_E_on (W16 recipe from scratch, smoke) | D_copy_K_on_mix03 (re-cool form, `--cooldown_mix 0.3`) |
|---|---|---|
| (a) holdout feeds in the batches | 194 of the 20,000-feed subsample are holdout feeds (29,937 candidates); the regular batches hold every one of the 20,000 feeds exactly once; the flag-off run has 0 holdout feeds in any batch; the other 19,806 feeds are the same set | 186 holdout feeds in the subsample (seed 1951); 129 of them in the 50 regular steps (24 regular steps are replaced by pool batches, as in the original stream logic) |
| (b) distillation target | recorded target of all 194 holdout feed-occurrences == `0.5 y/sum(y) + 0.5 softmax(teacher_mean11.npy[rows])` recomputed from the files: max abs diff 3.1e-08 (non-holdout: 6.5e-08) | 401 holdout feed-occurrences (regular + pool): max abs diff 4.9e-08 (non-holdout 8.1e-08) |
| (c) steps | steps/epoch 72 -> 73 = the independent `SplitData.batches` count with the holdout; total 73; t_cool 58 = round(0.8 x 73) | steps/epoch 74 (74 without: no bucket boundary is crossed in this subsample); t_cool 0 |
| (d) logs and files | 3 evals, each "holdout_ndcg nan" + one IN-SAMPLE line; BEST step 73 = total; kept checkpoint bit-identical to @last; only `*_holdoutINSAMPLE.npy` files (4,630,966 rows) in `preds/holdout`, no `holdout_meta` / `holdout_y`; val / test files of both checkpoints; json history all nan; `<v>_fulldata.json` | same (BEST step 74) |
| (e) pool | pool 1,004 feeds = 810 + the 194 holdout feeds (no pool batch drawn: 15 cooldown steps at p 0.02) | pool 998 = 812 + 186; 24 pool batches drawn, 1,528 feed-occurrences of which 272 are holdout feeds (17.8%; pool share 18.6%) |
| (f) schedule | logged lr of every eval equals the unchanged formula at the new total | same |

The same rule on the two remaining forms (runs finished 03:29 UTC, checked 03:32 UTC after the resume; all items PASS):

| R2 item | I_copy_P_on (wide popularity-light student, `train_full_abl.py`, from scratch) | J_copy_K_hz_on (re-cool form + `--hzcd`, `--cooldown_mix 0.3`) |
|---|---|---|
| (a) holdout feeds in the batches | 194 holdout feeds (29,937 candidates) in the regular batches; every one of the 20,000 feeds exactly once; flag-off run H has 0 holdout feeds in any batch and the same other 19,806 feeds | 129 of the 186 holdout feeds of the subsample in the 50 regular steps (24 steps replaced by pool batches) |
| (b) distillation target | 194 holdout feed-occurrences: max abs diff to the recomputed target 3.1e-08 (non-holdout 6.5e-08) | 401 holdout feed-occurrences: 4.9e-08 (non-holdout 8.1e-08) |
| (c) steps | steps/epoch 72 -> 73, total 73, t_cool 58 | steps/epoch 74 (74 without), t_cool 0 |
| (d) logs and files | 3 evals "holdout_ndcg nan" + IN-SAMPLE lines; BEST step 73 = total; kept == @last bit for bit; only `*_holdoutINSAMPLE.npy` files | same, BEST step 74; three in-sample files: kept, @last and `nn_<v>_holdoutINSAMPLE_hz.npy` (the hz view); no `*_holdout_hz.npy` |
| (e) pool | 1,004 = 810 + 194 (no pool batch drawn at p 0.02) | 998 = 812 + 186; 24 pool batches, 272 holdout feeds among 1,528 pool-batch feeds; the hz view is read by regular and pool batches alike (harness log line "wave-5 hzcd: from step 0 ...") |
| (f) schedule | lr of every eval equals the formula | same |

A fifth run covers the stage-FH form: K_copy_E_hz_on = the W16 recipe from scratch + `--hzcd` + the flag (04:04 UTC). All R2 items PASS (194 holdout feeds in the regular batches, target max abs diff 3.1e-08, steps 72 -> 73, t_cool 58, BEST step 73, the three in-sample holdout files incl. the hz one). One more cross-check against run C, which is the same command without `--hzcd`: the training batches of the two runs are identical, the per-step loss is identical on steps 1-58 and differs on every one of the 15 cooldown steps (59-73). The hz view is therefore read from the cooldown start on and nowhere before, with the holdout feeds in the batches.

Five flag-on runs, five ALL PASS: both harness files, the from-scratch and the re-cool form, with and without `--hzcd`.

### 2.4 The distillation target on the holdout rows is an out-of-sample teacher prediction

`teacher_holdout.py` (`logs/teacher_holdout.txt`), all [M]:

| check | teacher_mean11.npy | teacher_mean25.npy |
|---|---|---|
| rows | 246,745,836 (= train candidates), 0 non-finite in the whole file | same |
| the 4,630,966 holdout rows | finite, 0 of 30,000 feeds with a constant teacher, centred sd 1.218 | finite, 0 constant, sd 1.188 |
| equals the mean of the members' OWN saved holdout predictions (per-feed centred) | 11 members: corr 0.99999999999, max abs diff 3.1e-05 | 25 members: corr 1.0000000000, max abs diff 7.6e-06 |
| NDCG on the holdout (out of sample) | 0.32370 (kd_audit: 0.32370) | 0.32286 (kd_audit: 0.32286) |
| NDCG on 30,000 other exploration-like TRAINING feeds drawn by the same rule (in sample for the members) | 0.34388 (gap +0.0202) | 0.34412 (gap +0.0213) |
| provenance | 25 of 25 member checkpoints have `holdout: 30000`, and so has the plateau snapshot of each of the 7 re-cooled members; none has a teacher file or val labels | - |

So a full-data student learns the holdout feeds from their labels and from a teacher that never saw those labels. On every other training feed the teacher is in-sample (0.344 against 0.324): the holdout rows carry the only honest teacher targets of the file. No leak in either direction; val and test are untouched by the change. `reports/RECHECK_1114/kd_audit.md` had checked the component files a / d / e / f / p1 / p2 / h1 / h2 / xf; here the two assembled files that the students read are checked directly against all 25 members.

### 2.5 Steps per epoch and schedule with all data

Exact bucket arithmetic of `SplitData.batches` (cand_budget 32,768, max_feeds 512), `teacher_holdout.py` part E. The held-out line reproduces the members' logs (9,835 / 98,350 / 78,680), which validates the arithmetic.

| form | steps/epoch | total steps | cooldown start | cooldown steps | expected pool batches (2%) |
|---|---|---|---|---|---|
| 10 epochs, holdout excluded (today) | 9,835 | 98,350 | 78,680 | 19,670 | 393 |
| 10 epochs, all data | 9,984 (+149, +1.5%) | 99,840 | 79,872 | 19,968 | 399 |
| re-cool, holdout excluded (lane stages K / L) | 9,835 | 19,670 | 0 | 19,670 | 393 |
| re-cool, all data | 9,984 | 19,968 | 0 | 19,968 | 399 |

The holdout feeds are long (mean M 154), so 0.95% of the feeds are 1.88% of the candidates and 1.5% of the steps. `--save_steps 70000,77500` still fall in the plateau (cooldown from 79,872). The pool grows from 134,118 to 164,118 feeds while the number of pool draws stays about 82,000 feed-draws: 0.61 passes over the pool today, 0.50 with the holdout in it; the holdout feeds take 18% of the draws from the other pool feeds. Everything else is the same recipe: the fractions (cooldown 0.2, pool mix 0.02, warm-up 500 steps) are arguments, not step counts.

## 3. Full retrain or re-cool: which one is "trained on all data"

Both forms run through the same file and flag; the lane commands are in section 6.

| | full retrain twin (stage F / P) | re-cool twin (stage R / PR) |
|---|---|---|
| what it is | the member's own 10-epoch command + `--holdout_in_train` (same `--seed`) | the lane's K / L re-cool command (`--init <member>@77500.pt` + B_BASE, `--seed 1000 + seed`) + `--holdout_in_train` |
| steps that see the holdout feeds | all 99,840 | the 19,968 cooldown steps; the plateau snapshot at step 77,500 (7.88 epochs) was trained without them |
| regular passes over a holdout feed | 10 of 10 | 2 of 10 (20%) |
| share of the learning-rate-weighted updates [I] | 100% | 8.1% (the 1 - sqrt cooldown averages 0.353 of the peak lr: 7,055 of 86,677 peak-lr step equivalents) |
| pool membership (acts only in the cooldown) | yes | yes, the same 0.50 expected passes |
| feature normalisation | recomputed on feeds that include the holdout | loaded from the snapshot (computed without it) |
| cost [D: task text] | 1.15 GPU-h per member; 24 members = 27.6 GPU-h, the 20 members of today's two finals = 23 GPU-h | 0.25 GPU-h per member; 24 members = 6 GPU-h |
| wall clock [M: members' logs, "train time"] | 78 min on a lightly loaded GPU (3 of the 24 originals), 180-209 min (W16) and 180-262 min (popularity-light) at 4 processes per GPU (the other 21); +1.5% for the twin | 56-58 min at 4 processes per GPU (the eight K:ref re-cools of tonight) |
| what the twin can still tell | val only; its holdout numbers are in-sample | val only |
| honest label | "trained on all train data" | "the last 20% of the steps saw all train data" |

Verdict: only the full retrain supports the sentence "trained on all data". The re-cool twin is a different, weaker object; it should not be described with the owner's words.

What the 30,000 feeds are worth [M + I]. `facts.py` part 4: the cross-fit members (`--xfold k`, one user fold of five dropped, 80% of the feeds and of the steps) against the hard-label base on val, @last checkpoints: 20% of the data = +0.00243 per member (0.39638 vs 0.39395; 3 vs 15 members) and +0.00121 +- 0.00060 between 3-member pools (n 14,423). Scaled to 0.952% of the feeds by a log law this is +0.00010 per member and about +0.00005 at pool level; scaled by candidates (1.88%) it doubles. The full retrain therefore buys an expected +0.00005 to +0.0002 on val, the re-cool twin 8-20% of that for the regular batches, i.e. nothing above +0.00004, which is also what `COVERAGE_MAP.md` row 284 found by arithmetic for the pool alone. Against that, the noise of comparing two 16-pools of one recipe on val is 0.0003 (section 6.3). Neither form can be shown to help or to hurt; the full retrain is done because the owner asked for it, not for a measurable gain.

What it costs besides GPU hours: the in-period holdout stops being a witness for the twin members (their holdout numbers are in-sample and are written under names that say so). The originals stay on disk untouched, so every holdout read that the project relies on (two-witness rule for the non-staleness families, kd_audit, poplag) keeps its data. Twins are for the final file only.

One inconsistency to know for re-cool twins with `--hzcd` (stage RH): `feat_new/hz_common.py` forces the popularity lag of the 30,000 holdout feeds to 0, so that the holdout stays an as-built witness. Measured (`hz_holdout.py`, label-free): under hz 3.2% of the holdout rows change (zero rate 0.220 -> 0.245) against 12.6% for other exploration-like train feeds (0.220 -> 0.306). In an RH twin the holdout feeds therefore train with fresher popularity than the rest (they behave like the third of the train feeds whose hash lag is 0). It is small (1.9% of the candidates) and it goes against the purpose of hz, not against validity. A clean fix is to rebuild `it_hist_hz_*` with the holdout feeds lagged like every other feed; not done here.

## 4. Re-cool harness form, verified

The re-cool form (`--init <snapshot>@77500` + B_BASE flags, `lane.sh` stages K / L) takes the same code path as the from-scratch form after the model is built; the flag acts identically (feed list, pool, selection, file names). Verified by runs D and J of section 2.3 (flag on, 24 pool batches with holdout feeds; J with `--hzcd`) and by the identity pair below (flag off, original against copy, with `--cooldown_mix 0.3` so that the pool path and its leak guard are executed in both).

Identity pair of the re-cool form (R1; `lane.sh` K:ref:951 smoke command + `--cooldown_mix 0.3`, 74 steps, 24 of them pool batches):

| R1 item, E_orig_K_mix03 vs F_copy_K_off_mix03 | result |
|---|---|
| (a) logs line for line | identical, 24 lines |
| (b) training batches (15,797 feed-occurrences incl. pool batches), targets, per-step main loss, logit checksums | identical (74 steps; loss 0.212740 first, 1.161535 last) |
| (c) pre-clip gradient norms, pool flags | identical |
| (d) kept and @last checkpoints | 153 tensors bit-identical |
| (e) prediction files, summary json | identical |

What the re-cool twin is, in numbers (sections 2.5 and 3): it starts from the member's stored snapshot of step 77,500 (7.88 epochs without the holdout feeds; the originals' cooldown started at 78,680), resets the optimiser, and runs 19,968 cooldown steps = two passes over all 3,151,208 feeds with the holdout feeds in the regular batches and in the pool. It costs 0.25 GPU-h against 1.15 GPU-h. It is a legitimate, cheap probe of the code path at full scale; it is not "trained on all data" (section 3).

## 5. The fine-tune half: what "all of val" means

### 5.1 Today, and what row 99 measured

Today (n2cpv3s3, section 1): the test prediction of the fine-tune half is the mean of 15 models; each is fitted on 80% of val (11,538-11,539 feeds); every val label is used by 12 of the 15. As an ensemble the half already uses 100% of the val labels, each at weight 0.8. What it does not have is a single model that saw all of val. "All of val" can therefore only mean: every model is fitted on all 14,423 feeds (k seeds, no folds, no out-of-fold prediction).

Row 99 reproduced from the stored nested predictions (`ft_row99.py`, `logs/ft_row99.txt`; [M], n = 14,423 val feeds, every feed scored by models that never saw it; 3 hard-label members v15_cd_all_e10, v14_s93, v17_pre2s_s102; single-seed n2cpv3):

| comparison (per-feed mean over the 3 members) | delta |
|---|---|
| (ii) ONE model on 80% of val minus (i) the mean of FIVE models on 64% each = **row 99** | -0.00025 +- 0.00018 (z -1.37) |
| a second seed of the 80% model (the production OOF) minus (i) | -0.00052 +- 0.00017 (z -3.04) |
| the two 80% seeds averaged minus (i) | -0.00022 +- 0.00016 |
| 0.5 (i) + 0.5 (ii) minus (i) | -0.00002 +- 0.00012 |

So row 99 says "one model with 25% more labels loses to five models with fewer labels". It mixes two effects, model count (5 against 1) and data per model (64% against 80%), and it never compared equal numbers of models. A two-point 1/k fit of the same numbers (1 model -0.00038, 2 models -0.00022) puts five 80% models at -0.00013 and infinitely many at -0.00006 relative to five 64% models: the data-per-model effect is about zero within +-0.0002 [I].

Pool-level re-read, declared in `PREDECLARED.txt` R3 before it was computed (the finals average 16 members, which removes single-model fine-tune noise, so the trade could differ there): members pooled by per-feed z first.

| pool of the 3 members | NDCG | delta vs 15 models x 64% |
|---|---|---|
| (i) 15 models x 64% | 0.39821 | - |
| (ii) 3 models x 80% | 0.39804 | D_pool = -0.00017 +- 0.00025 (z -0.68); M<50 -0.00053 +- 0.00049, M>=50 +0.00008 +- 0.00026 |
| production OOF, 3 models x 80% (other seed) | 0.39768 | -0.00054 +- 0.00023 (z -2.28) |
| 6 models x 80% | 0.39788 | D_pool2 = -0.00034 +- 0.00023 (z -1.47) |

R3 reading as declared: "no evidence either way" (|D_pool| < 2 SE). Informally, every one of the six numbers above is negative or zero: nothing on record points to "more val per model" being worth anything, and fewer models is measurably worse.

### 5.2 The cleanest nested test (design only; not run)

Question: 15 models on 80% each (production) against k models on 100% each, at the production proportions and at pool level.

- Parts: `GroupKFold(6)` by user with the user-0 feeds as singletons (the production rule with 6 instead of 5): parts P0..P5 of about 2,404 feeds.
- For a held-out part h the other five parts R_h (83.3% of val, about 12,019 feeds) play "all of val".
  - Arm F15 (production analogue): the five folds of R_h are the five other parts. Fold model (h, j) is fitted on R_h minus P_j = 4 parts = 80% of R_h, 3 seeds: 15 models; prediction on P_h = their raw-score mean (final EMA weights, as `finetune.py`).
  - Arm R_k (full analogue): k seeds fitted on all of R_h; k = 12 is equal total data (12 x 100% = 15 x 80%), k = 15 equal model count, k = 3 the cheap variant (a subset of the 12).
- Rotation over h scores every val feed once, out of sample, by both arms. The fit set of fold model (h, j) is "val minus P_h and P_j", which is also the fit set of (j, h): 15 distinct fit sets x 3 seeds = 45 fine-tunes serve all six rotations; arm R needs 6 x k.
- Members: the 8 W8 members (the scored pool). Statistics, paired per feed, n = 14,423: (1) file level, the quantity that ships: NDCG(0.5 z(pool R_k) + 0.5 z(TO pool)) - NDCG(0.5 z(pool F15) + 0.5 z(TO pool)); (2) the fine-tune pools alone; (3) per member. Slices reported: M < 50, M >= 50, Tue / Thu, cold tercile.
- Decision rule to be written by the lead before the run; proposed: use full-val models for test iff (1) is >= +2 SE for k = 12 and >= 0 for k = 3; otherwise keep the folds.
- Cost [I]: one fine-tune without per-epoch test predictions took 0.5 min in ROUND5 (30 fine-tunes in 15.1 min, `nn_x/solvers/logs/r5_refit_nested.log`). Per member 45 + 72 = 117 fine-tunes = about 1 hour of a shared GPU; 8 members about 8 hours; with k = 3 only, about 4 hours. A driver exists for the 5-fold form (`nn_x/solvers/refit_ft.py`, modes check / nested / full); the 6-part rotation needs a small new driver.
- Power: the pool-of-3 delta above has SE 0.00025; with 8 members and the 0.5 file weight the file-level SE should be near 0.0001 [I]. The expected effect is within +-0.0002. The test can reject a loss or a gain of 0.0003; it cannot rank the arms if they differ by 0.0001.

### 5.3 Command lines, if a full-val variant is wanted

`<v>` = a member, e.g. `w5_e10_kd05w192_s951` or its twin `w5_fd10_kd05w192_s951`. All run in `nn_x/w5` with `CUDA_VISIBLE_DEVICES=<g>` and the 12-thread exports of `nn_x/w12/kd/ft_w5.sh`.

| variant | each model sees | command | out-of-fold witness |
|---|---|---|---|
| today, n2cpv3s3 | 80% | `python finetune.py --variant <v> --out <v>_n2cpv3s3 --extra n2c_pre_v3 --lr 5e-5 --epochs 6 --decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton --seeds 3` (= `bash nn_x/w12/kd/ft_w5.sh <g> <v>`) | yes, 3 models per val feed |
| FV93, unmodified `finetune.py` | 93.3% | the same with `--out <v>_n2cpv3f15 --folds 15 --seeds 1` (15 models as today, +17% steps) | yes, but 1 model per val feed: its OOF NDCG reads lower than the s3 file's by the seed-averaging effect, not by quality |
| FV100, verified copy | 100% | `PYTHONPATH=<repo>/nn_x/w5 python <repo>/nn_x/w12/stale/fulldata/finetune_full.py --variant <v> --out <v>_n2cpv3fv15 --extra n2c_pre_v3 --lr 5e-5 --epochs 6 --decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton --seeds 15 --full_val` (`--seeds 12` = equal compute; +25% steps per model) | none: val predictions are in-sample and are written as `nn_<out>_valINSAMPLE.npy` |

For the popularity-light students (nn_x/abl): `cd nn_x/abl/popfree` and `finetune_full_abl.py` (copy of `nn_x/abl/popfree/finetune.py`, same patch), no PYTHONPATH needed.

`finetune_full.py` / `finetune_full_abl.py` + `ft_full.py`: one flag `--full_val`; with it one fit set = all val feeds, `--seeds` models (seeds 0..S-1), `a.folds` forced to 1 so that every normalisation of the original holds; the val file is named `..._valINSAMPLE.npy`, the log announces that its "OOF" figures are in-sample, the json gets `insample_*` keys; `--pseudo`, `--add_random`, `--select best` without EMA and the output names of the fold recipes are refused. Flag absent: `test_ft_full.py`, AST(copy minus the FULLVAL statements) == AST(original) for both copies (1 import, 1 add_argument, 1 `fv = None`, 1 `if a.full_val:`, 2 `if fv is not None:`, 5 `X if fv is None else Y` -> X), parser defaults equal, `FullVal` unit checks: ALL PASS (`logs/test_ft_full.log`).

End-to-end CPU smoke runs of the fine-tune files (rule R5; the production n2cpv3s3 command cut to 1 epoch; `logs/check_ft.txt`; all PASS). No NDCG of these runs is a result.

| run pair / run | what | result |
|---|---|---|
| FT_A_orig vs FT_B_copy_off | `nn_x/w5/finetune.py` against `finetune_full.py` without the flag, member `w5_e10_kd05w192_s951`, `--folds 2 --seeds 1` | logs identical (9 lines); 224 training steps: feed ids, targets, per-step loss (5.382063 first, 4.884649 last), logit checksums identical; the three prediction files and the json identical |
| FT_D_orig_abl vs FT_F_copy_abl_off | `nn_x/abl/popfree/finetune.py` against `finetune_full_abl.py` without the flag, member `abl_e10_pop1kd05t25_s951` | the same items identical (11 log lines; loss 5.363435 first, 4.869611 last) |
| FT_C_copy_fv | `finetune_full.py --full_val --seeds 1 --out <v>_n2cpv3fv1` | 218 steps = every one of the 14,423 val feeds exactly once in the epoch, every batch a val batch; outputs `..._valINSAMPLE.npy`, `..._test.npy`, `..._test_folds.npy` (1 row), no `..._val_oof.npy`; json with `insample_*` keys; the in-sample warning precedes the first "OOF" figure of the log |
| FT_E_orig_on_twin, FT_G_orig_abl_on_twin | the ORIGINAL, unmodified fine-tune files on checkpoints written by flag-on runs (smoke runs C and I; checkpoint args `holdout_in_train True`, `keep 'last'`) | both run to the end (224 val steps, three finite prediction files): the production fine-tune (`ft_w5.sh`, `ft_one_s3_r6.sh`) accepts a twin checkpoint as it is |

### 5.4 Recommendation for the fine-tune half

Keep n2cpv3s3 as it is. Numbers: (a) the half already uses every val label (12 of 15 models each); (b) every measured variant with more val per model and fewer models is at or below zero: -0.00025 +- 0.00018 (row 99), -0.00052 +- 0.00017 (second seed), pool level -0.00017 +- 0.00025 and -0.00054 +- 0.00023; (c) the half's whole public value is +0.00048 +- 0.00055, so a change of its recipe can move the file by a small fraction of that; (d) a full-val model has no out-of-fold prediction, so the val side of the pre-submission gate would have to be read on a different set of models than the test side. If the owner still wants every fine-tune model to see more of val, FV93 (`--folds 15 --seeds 1`) is the variant that keeps a witness and needs no new code; FV100 is ready and proven equal to the original when the flag is off, but nothing in the project can measure it except the 6-part test of 5.2 or the public leaderboard.

## 6. Launcher, DRY run, acceptance check

### 6.1 `lane_full.sh`

`nn_x/w12/stale/fulldata/lane_full.sh <gpu> "<stage>:<arm>:<seed> ..."` has the structure and the GPU guard of `nn_x/w5/lane.sh`: only GPUs in `W5_GPUS` (default "<ids>"); before each launch `nvidia-smi` must show <= 3 compute processes and >= `LANE_GB` (60) GB free, checked under the per-GPU flock `nn_x/w5/logs/locks/gpu<g>.lock` (the lane's own lock file, for both harnesses), held until the job logs "steps/epoch"; polled every 120 s for `WAIT_MAX_H` (6) hours; `CUDA_VISIBLE_DEVICES` = that id; 12 CPU threads; atomic `mkdir` claims in `logs/queue`; the PID (verified through `/proc/<pid>/cmdline`) is appended to `<harness>/logs/RUNNING_full.txt` and to `nn_x/w12/stale/fulldata/logs/RUNNING_full.txt`; a finished job is never re-run, an unfinished one is never relaunched over its outputs, a failed one leaves `logs/queue/failed_<v>`. `DRY=1` prints the commands and writes nothing. `FD_SMOKE=1` adds the smoke settings. There is no `holdout_eval.py` fallback, on purpose: that script would write an in-sample file under the out-of-sample name.

| stage | job | variant | command = |
|---|---|---|---|
| F | `F:kd05w192:951` .. `958`, `F:kd05t25w192:951` .. `958` | `w5_fd10_<arm>_s<seed>` | `lane.sh` E:<arm>:<seed> + `--holdout_in_train`, through `train_full.py` |
| P | `P:pop1kd05t25:951` .. `954`, `P:pop1kd05w192:951` .. `954` | `abl_fd10_<arm>_s<seed>` | the member's launch command (`nn_x/abl/pop1/launch_e10.sh` + KD flags) + the flag, through `train_full_abl.py` |
| R | `R:<arm>:<seed>` | `w5_fdcd_<arm>_s<seed>` | `lane.sh` K:ref / L:ref + the flag |
| PR | `PR:<arm>:<seed>` | `abl_fdcd_<arm>_s<seed>` | B_BASE + HYG from `abl_e10_<arm>_s<seed>@77500.pt`, seed 1000 + seed, the member's teacher, + the flag |
| RH | `RH:<arm>:<seed>` | `w5_fdcdhz_<arm>_s<seed>` | `lane.sh` K:hz / L:hz + the flag (only if arm hz passes its rule) |
| FH | `FH:<arm>:<seed>` | `w5_fd10hz_<arm>_s<seed>` | stage F + `--hzcd` (= `lane.sh` E:hzcd form with the member's KD flags) + the flag (only if arm hz passes its rule) |

The twin names share no prefix with an existing family (`w5_fd*`, `abl_fd*`: 0 files of that name in either harness), so no existing glob or read-out picks them up by accident. Outputs land where the member's harness writes them (`nn_x/w5/{ckpt,logs,preds}`, `nn_x/abl/{ckpt,logs,preds}`), so the fine-tune scripts work on a twin unchanged: `bash nn_x/w12/kd/ft_w5.sh <gpu> w5_fd10_kd05w192_s951 ...` and `MODE=abl CUDA_VISIBLE_DEVICES=<g> bash nn_x/abl/ftbag/ft_one_s3_r6.sh abl_fd10_pop1kd05t25_s951`.

Checked (`check_cmds.py`, 48 of 48 PASS, `logs/check_cmds.txt`): each of the 24 F / P commands, parsed with the copy's own parser, equals the args dictionary in the ORIGINAL member's training log in every key except `variant` and `holdout_in_train` (keys newer than a log, such as `xfold`, are at their defaults); each of the 16 R commands equals the `lane.sh` K:ref / L:ref DRY command token for token except script path, variant and the flag; each of the 8 PR commands equals that form with the abl snapshot, seed and teacher; all 24 snapshots exist, are at step 77,500 and belong to the right member.

### 6.2 DRY-run output

Full output (95 lines, all six stages, smoke examples, refusals): `nn_x/w12/stale/fulldata/logs/DRY_lane_full.txt`. One line per stage:

```
$ DRY=1 bash nn_x/w12/stale/fulldata/lane_full.sh 0 "F:kd05w192:951 P:pop1kd05t25:951 R:kd05w192:951 PR:pop1kd05t25:951 RH:kd05t25w192:955"
lane: (cd <repo>/nn_x/w5 && PYTHONPATH=<repo>/nn_x/w5) <repo>/.venv/bin/python <repo>/nn_x/w12/stale/fulldata/train_full.py --variant w5_fd10_kd05w192_s951 --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed 951 --kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --holdout_in_train
lane: (cd <repo>/nn_x/abl && PYTHONPATH=<repo>/nn_x/abl) <repo>/.venv/bin/python <repo>/nn_x/w12/stale/fulldata/train_full_abl.py --variant abl_fd10_pop1kd05t25_s951 --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed 951 --xt_exclude seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo --xt_add xp_imp_c50 --kd_file <repo>/nn_x/w12/kd/out/teacher_mean25.npy --kd_w 0.5 --holdout_in_train
lane: (cd <repo>/nn_x/w5 && PYTHONPATH=<repo>/nn_x/w5) <repo>/.venv/bin/python <repo>/nn_x/w12/stale/fulldata/train_full.py --variant w5_fdcd_kd05w192_s951 --init <repo>/nn_x/w5/ckpt/w5_e10_kd05w192_s951@77500.pt --epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.02 --loss_norm feed --holdout 30000 --select_on holdout --eval_every 1000 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed 1951 --kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --holdout_in_train
lane: (cd <repo>/nn_x/abl && PYTHONPATH=<repo>/nn_x/abl) <repo>/.venv/bin/python <repo>/nn_x/w12/stale/fulldata/train_full_abl.py --variant abl_fdcd_pop1kd05t25_s951 --init <repo>/nn_x/abl/ckpt/abl_e10_pop1kd05t25_s951@77500.pt --epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.02 --loss_norm feed --holdout 30000 --select_on holdout --eval_every 1000 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed 1951 --kd_file <repo>/nn_x/w12/kd/out/teacher_mean25.npy --kd_w 0.5 --holdout_in_train
lane: (cd <repo>/nn_x/w5 && PYTHONPATH=<repo>/nn_x/w5) <repo>/.venv/bin/python <repo>/nn_x/w12/stale/fulldata/train_full.py --variant w5_fdcdhz_kd05t25w192_s955 --init <repo>/nn_x/w5/ckpt/w5_e10_kd05t25w192_s955@77500.pt --epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.02 --loss_norm feed --holdout 30000 --select_on holdout --eval_every 1000 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed 1955 --kd_file <repo>/nn_x/w12/kd/out/teacher_mean25.npy --kd_w 0.5 --hzcd --holdout_in_train
$ DRY=1 bash nn_x/w12/stale/fulldata/lane_full.sh 0 "F:kd05:951"          ->  lane_full.sh: stage F arms are kd05w192 | kd05t25w192 | kd08w192 (got kd05)
$ DRY=1 bash nn_x/w12/stale/fulldata/lane_full.sh 0 "P:pop1kd05t25:955"   ->  lane_full.sh: stage P seeds are 951-954 (got 955)
$ DRY=1 bash nn_x/w12/stale/fulldata/lane_full.sh 9 "F:kd05w192:951"      ->  lane_full.sh: refusing GPU N (allowed: <ids>)
```

Job lists for the lead (one lane per GPU slot; lanes given the same list share it):

```
JF="";  for a in kd05w192 kd05t25w192;   do for s in 951 952 953 954 955 956 957 958; do JF="$JF F:$a:$s"; done; done
JP="";  for a in pop1kd05t25 pop1kd05w192; do for s in 951 952 953 954;               do JP="$JP P:$a:$s"; done; done
W5_GPUS="0 1 2 3 4 5 6 7" setsid nohup bash nn_x/w12/stale/fulldata/lane_full.sh <g> "$JF $JP" > nn_x/w12/stale/fulldata/logs/lane_g<g>.out 2>&1 < /dev/null &
# GPU plumbing smoke first (about 3 minutes on a GPU): FD_SMOKE=1 bash nn_x/w12/stale/fulldata/lane_full.sh <g> "F:kd05w192:951"
```

### 6.3 Acceptance check for the twins (rule R4 of `PREDECLARED.txt`, written before any twin exists)

Val is the only clean witness for a twin. Null band [M] (`facts.py` part 3): 60 random splits of the 16 W16 members (@last) into two disjoint 8-pools (4 + 4 per arm): RMS of the val NDCG difference s8 = 0.00042 (max 0.00100; paired per-feed SE of one split 0.00046), so for two independent 16-pools of one recipe s16 = s8 / sqrt(2) = 0.00030 and the band is 3 s16 = 0.00090. Member level: SD of the 16 members' val NDCG = 0.00048.

| check | reference | accept | defect |
|---|---|---|---|
| A1 structure (each twin's log and files) | - | "training feeds 3121208 -> 3151208"; steps/epoch 9984; total 99840 (re-cool: 19968); "exploration pool 164118 feeds, of which 30000 holdout feeds"; every eval line "holdout_ndcg nan"; BEST step == total; `logs/<v>_fulldata.json` present; val / test files finite with 1,389,971 / 2,904,806 rows | any item missing; any file `nn_<twin>_holdout.npy` |
| A2 positive control | the twin's IN-SAMPLE holdout NDCG at the last eval minus its original's out-of-sample @last holdout NDCG (0.322-0.323 for the W16 members) | R4: >= +0.005 (re-cool twin: >= +0.002); in force after the calibration R7 below: >= +0.0043 (re-cool: >= +0.0017) | below: the holdout feeds did not reach the training batches |
| A3 pool level | val NDCG of the z-mean of the 16 twins minus that of the 16 originals at @last (0.40157), paired per feed | abs(D) <= 0.00090 | D < -0.00090: defect suspected, inspect before any use. D > +0.00090: not explained by +1% data (expected +0.00005 to +0.0002); inspect, do not claim it as a gain |
| A4 member level (information) | each twin minus its original (@last), val | - | flag a member below -4 member SDs = -0.0019 (as declared; the difference of two members has SD 0.00048 x sqrt(2) = 0.00068, so this is 2.8 SDs of the difference) |

I think the band of A3 is wide for a pool-level decision (it is three seed-noise SDs and the expected effect is a tenth of it); it is a defect detector, not an effect detector, and I apply it as written. The sharp checks are A1 and A2: they fail on the first twin if the code path is wrong.

**The check is executable: `accept_twins.py`** (rule R6 of `PREDECLARED.txt`, written 03:37 UTC before the script existed; R4 is applied as written). R6 fixes three points that R4 left open:

| point | R6 |
|---|---|
| (a) pools that are not the 16 W16 twins | band 3 s16 sqrt(16 / k): 8-pool 0.00127 (one W16 arm, or the 8 popularity-light twins), 4-pool 0.00180 (one popularity-light arm; the W8P20 students). Assumes the popularity-light students have the W16 members' seed noise (4 members per arm are too few to measure it). An incomplete set gets no verdict |
| (b) reference checkpoint | the originals' @last (like against like); the difference to the selected checkpoints is printed as information |
| (c) re-cool twins | reference = the K:ref / L:ref re-cool of the same snapshot (RH: the K:hz / L:hz re-cool; PR: the original member, no reference re-cool exists in nn_x/abl); band = 3 x the pool-level A/A noise of the re-cool harness, never wider than (a) |

```
accept_twins.py null             # A/A band of the re-cool harness from the (ref, ref2) pairs that exist; run before the first twin
accept_twins.py selftest         # the script checked without any twin
accept_twins.py check F          # also: P | R | RH | PR | FH, optional arm; exit code 1 on a DEFECT, 2 if no twin exists yet
```

Besides the R4 items, A1 of the script compares the twin's arguments (its summary json) with the reference run's (stages F, P, R, RH): every key equal except `variant`, `holdout_in_train`, `keep`; a flag that was added to `train.py` after the reference ran must be at its parser default. A wrong teacher file, seed or snapshot fails there. It also reads the run's own counts from `<v>_fulldata.json` (30,000 holdout feeds in training, in every epoch's batches, in the pool).

Self-test [M] (`logs/accept_selftest.txt`, all PASS):

| self-test | result |
|---|---|
| S1 the A1 parser on the four flag-on smoke runs (C, I, D, J; smoke expectations 19,806 -> 20,000 / 73 steps / pool 1,004 with 194, and 19,814 -> 20,000 / 74 / 998 with 186) | all items PASS |
| S2 the A1 parser on two flag-OFF smoke runs (B, F) | both rejected, 8 of 10 items DEFECT each |
| S3 A3 / A4 numerics, the 16 originals' selected checkpoints standing in for twins of their @last checkpoints | D = -0.00013 +- 0.00016 (n 14,423) = the `facts.py` number to 1e-6; verdict ACCEPT |
| S4 a degraded pool (per-feed z of every member + N(0, 1) noise) | D = -0.00774 +- 0.00104: flagged |
| S5 names: the twin variants the script looks for against the `--variant` names of the `lane_full.sh` DRY output, all six stages | equal (16 / 16 / 16 / 16 / 8 / 8) |
| S6 the check mode end to end on a mock harness tree (`runs/_mock_accept`: eight "twins" of arm kd05w192 = symlinks to smoke run C, references = the real originals; smoke expectations) | as it must be: A1 PASS for all 8 (arguments compared with the real members' json: equal up to the smoke settings and the seed; flags newer than the members' logs, `aug_all`, `xfold`, `xfold_unlab`, at their defaults), A2 DEFECT for all 8 (a 73-step smoke model: 0.169 against 0.322), A3 DEFECT SUSPECTED (-0.192), exit code 1 |

A/A noise of the re-cool harness [M] (`accept_twins.py null`, `logs/accept_null.txt`, 03:42 UTC, before any twin exists; val, @last checkpoints, n 14,423):

| quantity | value |
|---|---|
| (ref, ref2) pairs finished | 6: `w5_kcd_ref[2]_s951 / 953 / 955 / 957`, `w5_lcd_ref[2]_s951 / 953` |
| member level, ref2 minus ref (same snapshot, cooldown seed 2000 + s against 1000 + s) | +0.00014, -0.00041, -0.00139, -0.00004, -0.00035, -0.00022, each +- 0.00058 paired per feed; RMS 0.00062 |
| pool level: 6-pool against 6-pool over the 32 within-pair label swaps | s_AA(6) = 0.00053 (max 0.00130) |
| scaled to 16 members | 0.00032, against s16 = 0.00030 for independent from-scratch pools |
| band in force for re-cool twins | the R6(a) cap: 16-pool 0.00090, 8-pool 0.00127, 4-pool 0.00180 (3 s_AA sqrt(6 / k) would be 0.00097 / 0.00137 / 0.00193) |
| information: K:ref / L:ref re-cool minus the original cooldown of the same snapshot, 12 pairs | member mean +0.00014, SD 0.00057; 12-pool difference -0.00009 +- 0.00031 |

Three things follow.

1. Two re-cools of one snapshot that differ only in the cooldown seed differ on val, at pool level, as much as pools of independent from-scratch members do (0.00032 against 0.00030 at 16 members). Sharing the plateau buys no precision. The first run of this task wrote in 6.4 that the re-cool comparison would be "2-3 times tighter" [I]; the measurement says it is not, and the sentence is corrected.
2. One A/A pair (kcd s955) is at -0.00139, 2.4 paired SEs. The logs show why single-member val reads are that loose: `w5_kcd_ref2_s955` reads 0.39878 at step 19,000 and 0.39802 at step 19,670. Member-level val NDCG moves by 0.0005 to 0.001 between consecutive evals at the end of a cooldown. For re-cool arms that keep the cooldown seed of their reference (arm hz) the A/A number is an upper bound: the shared batch order may remove part of it, which the ref2 runs cannot tell.
3. Rule history, stated plainly. R6(c) as first written took the pool-level A/A noise as (member-level RMS) / sqrt(k). After the first null read (5 pairs, no twin involved) I saw that this formula under-states the measured pool-level noise of the from-scratch members by a factor 2.2 (0.00019 predicted, s8 = 0.00042 measured), amended the rule to the pool-level label-swap statistic above BEFORE computing it (`PREDECLARED.txt`, block 03:42 UTC), and recomputed; a sixth pair had finished in between. The amended statistic gives nothing tighter than R6(a), so one band holds for every stage. `accept_twins.py null` may be re-run when more ref2 runs have finished; R6 says which value is in force.

**Calibration of the A2 bars (rule R7, written 04:06 UTC before the measurement; no twin involved).** The bars of R4 were inferred from the teacher ensemble. A student is a different object: half of its target is the teacher, and on the holdout rows a twin's teacher half is out of sample, so its lift there comes through the label half only. The in-sample gap G_s of two ORIGINAL final members was therefore measured: the @last checkpoint predicts 20,000 exploration-like TRAINING feeds drawn by the holdout rule from the non-holdout feeds (in sample), against its saved out-of-sample holdout file (`insample_gap.py` through `cpu_run.py`, 9 and 12 CPU minutes; `logs/a2_calib.txt`) [M]:

| member (@last) | NDCG in sample (n 20,000) | NDCG holdout, out of sample (n 30,000) | G_s | path check: this CPU code against the saved GPU file on 2,000 holdout feeds |
|---|---|---|---|---|
| `w5_e10_kd05w192_s951` (W16) | 0.34631 +- 0.00150 | 0.32219 +- 0.00115 | +0.0241 +- 0.0019 | 0.32279 vs 0.32265 (+0.00014 +- 0.00022), centred corr 0.99995 |
| `abl_e10_pop1kd05t25_s951` (popularity-light, W8P20) | 0.33294 +- 0.00145 | 0.31583 +- 0.00111 | +0.0171 +- 0.0018 | 0.31279 vs 0.31296 (-0.00017 +- 0.00042), centred corr 0.99994 |
| their teachers, same feeds | mean11 0.34317 / mean25 0.34332 | 0.32370 / 0.32286 | +0.0195 / +0.0205 | - |

| R7 quantity | value |
|---|---|
| null noise of the A2 statistic: sqrt(2) x SD of the 16 originals' @last holdout NDCG (0.00030) | s_null = 0.00043; floor 3 s_null = 0.00128 |
| G_s used (the smaller of the two) | 0.0171; expected lift of a full-retrain twin about 0.5 G_s = +0.0086 [I] |
| bar in force, full retrain: max(floor, min(0.005, 0.25 G_s)) | +0.0043 (R4: +0.005) |
| bar in force, re-cool: max(floor, min(0.002, 0.10 G_s)) | +0.0017 (R4: +0.002) |

The calibration confirms the order of magnitude of R4 and lowers both bars slightly; they stay 10 and 4 null SDs above zero. Two side facts: the W16 student is BETTER than its teacher on the feeds both trained on (+0.0031 +- 0.0006) and worse on the holdout (-0.0015 +- 0.0004), so the label half of the target does memorise; and a member's holdout NDCG is 0.017 to 0.024 below what it reads on feeds it trained on, which is the size of the number that a twin's holdout column would silently inflate by if it were ever read as out of sample.

A doubt that remains about the re-cool bar: the factor 0.10 is an inference (two passes at 35% of the peak learning rate). I apply the rule as written. If a re-cool twin passes A1 (the run itself counts 30,000 holdout feeds in every epoch's batches, and writes that count to `<v>_fulldata.json`) and fails only A2, the script prints DEFECT with that count next to it: the feeds were in the batches and the bar was too high.

Use of accepted twins: member for member in the SAME file recipe (fine-tune n2cpv3s3 of the twin checkpoints; 0.5 z(FT) + 0.5 z(TO)); presub gate on the rebuilt file as on every candidate (the gate reads val / test specs and the members' logs, no holdout file: `src/presub_gate.py audit_members`; copy `<v>.log` and `<v>_n2cpv3s3_ft.log` to `nn/logs/` as `nn_x/w12/kd/w16_build_gate.sh` does). Any holdout column printed for a file that contains twins is in-sample for them.

### 6.4 Suggested order for the lead

1. GPU plumbing smoke (minutes): `FD_SMOKE=1 bash nn_x/w12/stale/fulldata/lane_full.sh <g> "F:kd05w192:951 R:kd05w192:951 P:pop1kd05t25:951"`, then the A1 items on the three smoke logs (training feeds 19806 -> 20000 for seed 951 of the from-scratch form). This is the first GPU execution of the path: bf16 autocast and TF32 were not exercised on the CPU.
2. Four re-cool twins as an inexpensive probe of the path at full scale (`R:kd05w192:951-954`, about 1 GPU-h): `accept_twins.py check R kd05w192` gives A1 and A2 on real runs, which is what this step is for. Their val comparison with the K:ref re-cools of the same four snapshots is information only: measured A/A noise of the harness puts a 4-pool difference at about 0.0006 (6.3), no tighter than independent seeds.
3. The full retrains (F, P) for the members of the files that will be the final picks, after the hz verdict fixes which recipe that is (the H16 members are hz re-cools of the plateau snapshots; their twins are stage RH for the re-cool form and stage FH = the stage-F command + `--hzcd` for a full retrain whose last 20% of steps read the hz view; FH has no optimiser reset at the cooldown start, so it is a close relative of "snapshot + hz re-cool", not the identical recipe).
4. Fine-tune the twins (n2cpv3s3), rebuild the file with the twins member for member, gate, A3 / A4 on val.

## 7. Method

- Copies, not edits: `train.py`, `finetune.py` (w5 and abl) were copied into my directory and patched by exact-match string replacement (each replacement asserted to occur once). Frozen references with md5 in `ref/`. The four live train / fine-tune files were unchanged at the last check (04:04 UTC; `test_full.py` and `test_ft_full.py` re-check this on every run). `nn_x/w5/lane.sh` did change during the task (03:35 UTC, by its owner: three new arm lines `E:kwxd15`, `E:kwxd30`, `K:th|L:th`; `logs/lane_sh_change_0335.diff`). Nothing that `lane_full.sh` copies is touched: E_BASE, B_BASE, HYG and `check_gpu` are identical in the live file and in the launcher, and `check_cmds.py`, which compares against the live `lane.sh` DRY output, gives 48 of 48 after the change (04:04 UTC).
- Flag-off identity is proven twice: on the syntax tree (the copy minus an exact, counted set of guarded statements equals the reference) and end to end (original file and copy executed unchanged as `__main__` on the same configuration; logs, every training batch, per-step loss and gradient norm, checkpoints and prediction files compared exactly).
- CPU execution of GPU-only files: `cpu_run.py` starts with `CUDA_VISIBLE_DEVICES=""`, pins `torch.cuda.is_available` to False and makes `torch.cuda._lazy_init` raise, maps `torch.device("cuda:0")` to the CPU, satisfies the import guard of `common.py` for the duration of that import only, and redirects `common.CKPT / LOGS / PREDS` into `runs/<tag>/` before the train file imports them. No file of the harness directories was written (`find src nn_x/w5 nn_x/abl -newermt ...` shows only other agents' logs / preds / ckpt). 12 threads per process, never more than two heavy processes of mine.
- Flag-on checks recompute everything independently of the code under test: the holdout from the rule, the subsample from the seed, the batch counts from `feeds.parquet`, the distillation target from the teacher `.npy` and the label `.npy`.
- Rules before reads: `PREDECLARED.txt` (R1 identity, R2 flag-on, R3 pool-level re-read of the ROUND5 files, R4 acceptance of the twins, R5 fine-tune copy, R6 the acceptance script, with one amendment of R6(c) that is explained in 6.3, R7 the calibration of the A2 bars), each block time-stamped before the comparison it judges.
- Numbers: paired per-feed mean +- SE with the feed count; NDCG = `src/metrics.ndcg_per_feed` (pessimistic ties); pools = member mean of per-feed z.
- Resume (03:31 UTC): the first run of this task was cut at 03:22 UTC. Before continuing I checked that no process of mine was alive, that the live sources still equal the frozen references (md5), re-ran the static tests, and re-derived a sample of the cached numbers independently (`recheck_sample.py`, section 1). Runs I and J had finished after the cut and were checked then.

## 8. Limits, and what was NOT checked

| not checked / limit | why it matters | what covers it |
|---|---|---|
| No GPU execution of `train_full.py` / `train_full_abl.py` (rule 1: CPU only) | the lanes run bf16 autocast, TF32 and the non-deterministic GPU kernels; all my end-to-end runs are CPU fp32 | the patch touches no tensor code (feed list, pool list, a score set to -inf, file names); step 1 of 6.4 is a 3-minute `FD_SMOKE=1` run on a GPU, then `accept_twins.py` A1 / A2 on the first real twin |
| No full-scale run of either form | the smoke runs are 72-74 steps on a 20,000-feed subsample; steps per epoch 9,984 and the pool of 164,118 are arithmetic (the same arithmetic reproduces the members' logged 9,835 / 98,350 / 78,680 / 134,118) | A1 of the acceptance check tests exactly these numbers on the twin's log |
| Smoke predictions are eval-capped (first 1,000 val / 1,500 holdout / 1,500 test feeds per predict call) | "identical predictions" covers those rows; logged NDCG values of the smoke runs are not real NDCGs | the checkpoints are compared in full (153 tensors, bit for bit); predictions are a deterministic function of the checkpoint through unchanged code |
| The VALUE of the 30,000 feeds | not measurable on the CPU; +0.00005 to +0.0002 on val is a scaling of the cross-fit members [I], and the share a re-cool twin captures (8-20%) is an inference from pass counts and learning-rate mass | nothing in the project can resolve it: 16-pool seed noise on val is 0.00030 |
| A2 bars | calibrated on the in-sample gap of two original members (R7: +0.0043 / +0.0017); the factors 0.25 and 0.10 that turn the gap into a bar are inferences, most of all for the re-cool form | the run's own batch counts in `<v>_fulldata.json` separate "feeds not in the batches" from "bar too high" (6.3) |
| Band for popularity-light pools (R6a) | assumes the W16 members' seed noise | the script prints the paired per-feed SE next to it |
| `--full_val` fine-tune: value | never run on a GPU, no nested test run; the smoke runs prove mechanics only (rule R5) | the 6-part design of 5.2, if the owner wants the number |
| Stage FH and RH on a GPU | both forms were smoke-run on the CPU (K and J); the hz view is a 2.3 GB tensor per process on the GPU on top of the usual footprint | `FD_SMOKE=1 ... "FH:kd05w192:951"` first; FH is a relative of "snapshot + hz re-cool", not the identical recipe (6.4) |
| hz popularity of the holdout feeds in RH / FH twins | `feat_new/hz_common.py` keeps lag 0 for the 30,000 holdout feeds: they train with fresher popularity than other feeds (3.2% of rows changed against 12.6%) | stated in section 3; fixing it means rebuilding `it_hist_hz_*` (not done) |
| Teachers | the 25 teacher members still exclude the holdout; "all data" here means the student's own training set | by design (2.4) |
| Inputs of the members (features built from val / test impressions, `pre_*` embeddings) | whether any INPUT statistic uses data that a "trained on all data" claim should mention was not audited here | `src/presub_gate.py` audits the input lists of every candidate file |
| Building or gating a twin file | no submission file was built (rule 5); the builder `nn_x/w12/kd/build_r3.py` takes fixed member-name lists | to build twin files the lead changes its two name lists (`W`, `P1n`) to the twin names after `ft_w5.sh` / `ft_one_s3_r6.sh` have copied the twins' files into `preds/` |

One observation outside my task, for the staleness family: the fine-tune of the popularity-light students re-reads 7 of the 8 popularity columns that their train-only stage excludes (harness log line "abl popfree note: 7 of the 8 excluded xt columns re-enter through the fine-tune extra branch", seen in my smoke run of `nn_x/abl/popfree/finetune.py`). The fine-tune half of a "popularity-light" member is therefore not popularity-light. This is the harness's pre-declared recipe, not a defect; it is relevant to how the W8P20 admixture is read.

## 9. Exact commands

All from `<repo>`, with `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12` and `F=nn_x/w12/stale/fulldata`.

```
# static proofs (seconds)
.venv/bin/python $F/test_full.py                 # 40 checks: AST identity flag-off (w5 + abl), parser, diff, FullData unit
.venv/bin/python $F/test_ft_full.py              # the same for finetune_full.py / finetune_full_abl.py
.venv/bin/python $F/check_cmds.py                # 48 checks: lane_full.sh commands == the members' own logged arguments
# facts (minutes)
.venv/bin/python $F/facts.py ; .venv/bin/python $F/teacher_holdout.py ; .venv/bin/python $F/ft_row99.py
.venv/bin/python $F/hz_holdout.py ; .venv/bin/python $F/recheck_sample.py
# CPU smoke runs (10-13 minutes each, detached; at most two at a time) and their checks
bash $F/run_smoke.sh A_orig_E orig E ;            bash $F/run_smoke.sh B_copy_E_off copy E
bash $F/run_smoke.sh C_copy_E_on copy E --holdout_in_train
bash $F/run_smoke.sh D_copy_K_on_mix03 copy K --cooldown_mix 0.3 --holdout_in_train
bash $F/run_smoke.sh E_orig_K_mix03 orig K --cooldown_mix 0.3 ; bash $F/run_smoke.sh F_copy_K_off_mix03 copy K --cooldown_mix 0.3
bash $F/run_smoke.sh G_orig_P orig P ;            bash $F/run_smoke.sh H_copy_P_off copy P
bash $F/run_smoke.sh I_copy_P_on copy P --holdout_in_train
bash $F/run_smoke.sh J_copy_K_hz_on copy K --cooldown_mix 0.3 --hzcd --holdout_in_train
bash $F/run_smoke.sh K_copy_E_hz_on copy E --hzcd --holdout_in_train
.venv/bin/python $F/check_e2e.py identity A_orig_E B_copy_E_off          # also E / F and G / H
.venv/bin/python $F/check_e2e.py flagon C_copy_E_on B_copy_E_off         # also D, I (ref H), J, K (ref B)
# fine-tune smoke runs (6-7 minutes each)
bash $F/run_ft_smoke.sh FT_A_orig orig --folds 2 --seeds 1 ; bash $F/run_ft_smoke.sh FT_B_copy_off copy --folds 2 --seeds 1
bash $F/run_ft_smoke.sh FT_C_copy_fv copy --full_val --seeds 1 --out w5_e10_kd05w192_s951_n2cpv3fv1
bash $F/run_ft_smoke_abl.sh FT_D_orig_abl orig --folds 2 --seeds 1 ; bash $F/run_ft_smoke_abl.sh FT_F_copy_abl_off copy --folds 2 --seeds 1
bash $F/run_ft_on_ckpt.sh FT_E_orig_on_twin w5 $F/runs/C_copy_E_on/ckpt/w5_smoke_e10_kd05w192_s951.pt --folds 2 --seeds 1
bash $F/run_ft_on_ckpt.sh FT_G_orig_abl_on_twin abl $F/runs/I_copy_P_on/ckpt/abl_smoke_e10_pop1kd05w192_s951.pt --folds 2 --seeds 1
.venv/bin/python $F/check_ft.py identity FT_A_orig FT_B_copy_off ; .venv/bin/python $F/check_ft.py flagon FT_C_copy_fv
.venv/bin/python $F/check_ft.py identity FT_D_orig_abl FT_F_copy_abl_off
# A2 calibration (R7; 9-12 minutes each), then the acceptance script
bash $F/run_gap.sh w5_e10_kd05w192_s951 ; bash $F/run_gap.sh abl_e10_pop1kd05t25_s951 ; .venv/bin/python $F/a2_calib.py
.venv/bin/python $F/accept_twins.py null ; .venv/bin/python $F/accept_twins.py selftest
# launcher, DRY
DRY=1 bash $F/lane_full.sh 0 "F:kd05w192:951 P:pop1kd05t25:951 R:kd05w192:951 PR:pop1kd05t25:951 RH:kd05t25w192:955 FH:kd05w192:951"
```

For the lead, the launch itself (not run by me): the job lists of 6.2, then per finished stage `accept_twins.py check F` (P, R, ...). The twins that the two finals of today need are the 16 W16 members and the 4 students of W8P20 (`P:pop1kd05t25:951-954`): 20 full retrains, 23 GPU-h at 1.15 GPU-h each [D: task text], plus their 20 fine-tunes; the 4 wide popularity-light students (`P:pop1kd05w192:951-954`) are in no current file.

## 10. Files

Everything is in `nn_x/w12/stale/fulldata/` (1.5 GB, almost all of it the smoke-run directories `runs/`, which can be deleted once the report is accepted) and this report. Nothing outside was written: no new file under `data/`, `src/`, `feat_new/`, `eda/`, `nn/feat_cache` since the task started, no bytecode file in `nn_x/w5` / `nn_x/abl`, no twin- or smoke-named file in the harness `ckpt / logs / preds` directories, and the production fine-tune files of the two members used in the fine-tune smokes keep their 11-14 time stamps (checked 03:48 and 04:21 UTC).

| file | role |
|---|---|
| `train_full.py`, `train_full_abl.py`, `cd_full.py` | deliverable 1: the copies with `--holdout_in_train` and their helper |
| `finetune_full.py`, `finetune_full_abl.py`, `ft_full.py` | optional `--full_val` fine-tune copies and helper (section 5.3) |
| `lane_full.sh` | deliverable 5: launcher with the GPU guard of `nn_x/w5/lane.sh` (stages F, FH, P, R, RH, PR) |
| `accept_twins.py` | the acceptance check, executable (null / selftest / check) |
| `insample_gap.py`, `run_gap.sh`, `a2_calib.py` | rule R7: the measured in-sample gap of two original members and the A2 bars in force (`logs/gap_<member>.json`, `logs/a2_calib.json`, read by `accept_twins.py`) |
| `PREDECLARED.txt` | rules R1-R7 with time stamps, including the amendment of R6(c) |
| `test_full.py`, `test_ft_full.py`, `check_cmds.py` | static proofs |
| `cpu_run.py`, `run_smoke.sh`, `run_ft_smoke.sh`, `run_ft_smoke_abl.sh`, `run_ft_on_ckpt.sh`, `check_e2e.py`, `check_ft.py` | CPU end-to-end runs and their checks |
| `facts.py`, `teacher_holdout.py`, `ft_row99.py`, `hz_holdout.py`, `recheck_sample.py` | measurements of sections 1, 2.4, 2.5, 3, 5.1 |
| `ref/` | frozen copies of the five source files with md5 |
| `logs/` | every output quoted above: `test_full.log`, `check_e2e.txt`, `check_ft.txt`, `check_cmds.txt`, `facts.txt`, `teacher_holdout.txt`, `ft_row99.txt`, `hz_holdout.txt`, `recheck_sample.txt`, `accept_null.txt`, `accept_selftest.txt`, `accept_selftest_mock.txt`, `a2_calib.txt`, `lane_sh_change_0335.diff`, `DRY_lane_full.txt`, `patch_w5.diff`, `patch_abl.diff`, `RUNNING_smoke.txt`, one `.stdout` per smoke run |
| `runs/<tag>/` | the 11 training and 7 fine-tune smoke runs: `cmd.txt`, `ckpt/`, `logs/`, `preds/`, `record.npz`; `runs/GAP_<member>/` = the two R7 prediction runs; `runs/_mock_accept/` = the symlink tree of the acceptance self-test |
| `PROGRESS.md` | progress file |

md5 of the deliverables at hand-over (`logs/deliverables_md5.txt`; the train copies were last written at 02:26 UTC, before the first smoke run, so every run above executed exactly these files):

| file | md5 |
|---|---|
| `train_full.py` | 1366cb4cbf84689990df7ad99542f164 |
| `train_full_abl.py` | 34ad72020dd41203763df0321d0fc9f1 |
| `cd_full.py` | f0ff47afd0008b96f8412fdc27a223a9 |
| `finetune_full.py` | d34dcc6ed6edec38afd991c3ef2fad50 |
| `finetune_full_abl.py` | 5aa0945d4cfa2b69be19ea7256f83ba2 |
| `ft_full.py` | 6a77bc95da089c6b5c12b5610d96ea95 |
| `lane_full.sh` | 3a27322e37d67b1507958e31805b9a0f |

The copies are derived from `nn_x/w5/train.py` e4b5f0d7... and `nn_x/abl/train.py` 1317bdae... (unchanged at 04:20 UTC). If one of them is edited before the launch, the copy still trains the members' recipe, but the identity proof then refers to the older source: `test_full.py` reports it in its first item, and the copy should be re-derived (the 12 replacements are in `logs/patch_w5.diff`) if the edit touches a code path the finals use.
