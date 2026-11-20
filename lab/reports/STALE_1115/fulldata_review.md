# FULLDATA REVIEW: adversarial check of `reports/STALE_1115/fulldata.md` and `nn_x/w12/stale/fulldata/`

Reviewer run 2026-11-15 04:22 UTC onward. CPU only, at most two heavy processes, 12 threads. Everything written is in
`nn_x/w12/stale/fulldata_review/` and this file. Marks: [M] measured here with my own code, [D] read in a project file, [I] inferred.
"Study" = the work under review.

## Verdict

1. **CONFIRMED.** Every number and identity the study's verdict rests on re-derives with my own code (section 1, 37 rows, all "yes"): the 30,000 feeds (0.952% / 1.877% / 18.28%), selected minus @last (+0.00011 +- 0.00009 holdout, -0.00013 +- 0.00016 val), steps 9,835 -> 9,984, pool 134,118 -> 164,118, row 99 (-0.00025 +- 0.00018), s16 (0.00028 here, 0.00030 there), the A/A numbers, the A2 calibration (0.34631 / 0.32219).
2. **Code path sound for stages F / P / R / PR.** Flag off = the source at hand-over (my AST folding on all four copies + my own end-to-end pair). Flag on, in four runs of mine, two of them forms the study never executed (narrow popularity-light student, abl re-cool): holdout feeds in the regular batches and in the pool, target = the teacher-file rows (3e-08), which are out of sample for each of the ten teacher components, last-step checkpoint, in-sample file names only. No val / test label and no in-sample teacher value is introduced. The study changed no read-only file and started no GPU process.
3. **MAJOR, arose after the hand-over: the hz stages RH / FH are stale.** The owner edited the live `nn_x/w5/train.py` at 04:52 and 04:57 UTC (`--xt_drop_init`; hz module `cd_arms6` -> `cd_arms6b` with the hz2 view through `HZCD_DIR` / `HZ2_KEEP`). With the delivered copy an hz2-family twin would SILENTLY train the old hz view, and an `--xt_drop_init` twin is rejected by the parser. Stages F / P / R / PR execute the same statements as before. A re-derived candidate is in `rederived2/` (AST-equal to the live file; identical end to end on a re-cool).
4. **A2 is safe but blunt.** A student trained with an out-of-sample teacher (`w5_e10_kx05_s951`) memorises its training feeds by +0.0208 +- 0.0008: the expected lift of a full-retrain twin is about +0.02, not the study's +0.0086 [I]. The bar +0.0043 stands; inspect any full-retrain twin below +0.012.
5. **Two hand-over gaps.** A copy of `build_r3.py` with only the two name lists changed stops on two asserts (new tags and separate twin lists are needed); a twin checkpoint still says `holdout: 30000`, so provenance audits must also read `holdout_in_train`.
6. **The launch is already under way** (lead, 04:57-05:05 UTC): two GPU smokes with rc 0, whose logs and files I read (all structural items right; my NDCG equals the logged one), and 20 waiting lanes. The four P lanes are `pop1kd05w192`; W8P20 is built from `pop1kd05t25`: check that this is intended. Nothing has to be stopped because of item 3.

## 1. Claim | study's number | my number | agree

Own code throughout: `rv_facts.py` (holdout rule, steps, pool; NDCG re-implemented and cross-checked against `src/metrics.py` on a
tied random score), `rv_teacher.py` / `rv_teacher2.py`, `rv_row99.py`, `rv_aa.py`, `rv_folds` (inline), `rv_hz.py`, `rv_ast.py`,
`rv_cmds.py`, `rv_cpu.py` + `rv_cmp.py` (own CPU runner and comparison). Nothing of the study's evaluation, builder or test code
is imported. Samples are rebuilt from `data/flat/*`, `eda/labels_position/out/train_prandom.npz`, the members' prediction files
and logs.

### 1.1 Facts about the current finals

| claim | study | mine [M] | agree |
|---|---|---|---|
| holdout = 30,000 feeds; share of train feeds / candidates / exploration-like feeds | 0.952% / 1.877% (4,630,966 rows) / 18.28% of 164,118 | 0.952% / 1.877% (4,630,966 rows) / 18.28% of 164,118; draw set 89,794 (33.4%); mean M 154.4 vs 78.3 | yes |
| all 24 members log "holdout: 30000 ... excluded", 3,121,208 feeds, 9,835 steps/epoch, 98,350 steps, pool 134,118 | 24 of 24 | 24 of 24 (grep of the 24 training logs) | yes |
| kept checkpoint steps | 92,500 (2), 97,500 (14), 98,350 (8) | 92,500 (2), 97,500 (14), 98,350 (8) | yes |
| selected minus @last, W16 16-pool, holdout | +0.00011 +- 0.00009 (n 30,000; 0.32460 vs 0.32449) | +0.00011 +- 0.00009 (n 30,000; 0.32460 vs 0.32449) | yes |
| same, val | -0.00013 +- 0.00016 (n 14,423; 0.40144 vs 0.40157) | -0.00013 +- 0.00016 (n 14,423; 0.40144 vs 0.40157) | yes |
| same, member level (16) | holdout +0.00011 (0 .. +0.00062); val +0.00012 (-0.00011 .. +0.00065) | holdout +0.00011 (0 .. +0.00062); val +0.00012 (-0.00011 .. +0.00065) | yes |
| nothing else dropped (every train / val feed has a positive) | 0 / 0 feeds without a positive | 0 / 0 | yes |
| share of the 3,165,631 labelled feeds in the train-only half: today / twin | 98.60% / 99.54% | 98.597% / 99.544% | yes |
| fine-tune: GroupKFold(5) by user with 658 user-0 singletons x 3 seeds; fit 11,538-11,539 feeds; every val label in 12 of 15 models | as stated | fit sizes 11,538 x3, 11,539 x2 (80.00%); every label in exactly 12 of 15; `folds 5, seeds 3, lr 5e-05, epochs 4, pseudo ''` in the fine-tune logs of all 16 W16 members and the 4 W8P20 students | yes |
| train-only half never sees val | no `--val_mix`, no val file | `val_mix 0`, no val path in the logged arguments of the 24 members; the 25 teacher checkpoints and the 7 v17 snapshots behind the re-cooled teachers: `holdout 30000`, `val_mix 0`, no `kd_file` | yes |

### 1.2 Steps, pool, schedule with all data

| claim | study | mine [M] | agree |
|---|---|---|---|
| steps/epoch without -> with the holdout | 9,835 -> 9,984 | 9,835 -> 9,984 (own bucket arithmetic on `feeds.parquet`) | yes |
| total steps, cooldown start (10 epochs) | 98,350 / 78,680 -> 99,840 / 79,872 | 98,350 / 78,680 -> 99,840 / 79,872 | yes |
| re-cool total | 19,670 -> 19,968 | 19,670 -> 19,968 | yes |
| cooldown pool | 134,118 -> 164,118 | 134,118 -> 164,118; all 30,000 holdout feeds have p_random > 0.5 | yes |
| share of lr-weighted updates a re-cool twin gives the holdout feeds | 8.1% [I] | 8.1% [I] (mean of the 1 - sqrt cooldown = 0.3533; 7,055 of 86,677 peak-lr step equivalents) | yes |

### 1.3 Teacher rows on the holdout

| claim | study | mine [M] | agree |
|---|---|---|---|
| `teacher_mean11.npy` / `teacher_mean25.npy`: 246,745,836 rows, no non-finite value; no constant feed among the 30,000 | as stated | as stated (whole file scanned) | yes |
| the files are the mean of the members' out-of-sample predictions on the holdout rows | max diff 3.1e-05 / 7.6e-06 to the members' own holdout files | holdout rows of the files == per-feed-centred mean of the component dumps a..f (11) / a..p2 (25): max diff 3.1e-05 / 4.8e-07; components e, f, p1 == the members' own saved holdout files: max diff 3.5e-05 / 1.6e-05 / 8.9e-05 (sums of 2 / 1 / 4 members), corr 1.00000000 | yes |
| every teacher component is out of sample on the holdout rows | provenance (25 checkpoints `holdout 30000`) + NDCG of the assembled files | provenance (same) AND, per component, NDCG on the holdout rows against 30,000 other draw-set training feeds: a 0.3218 / 0.3426, b 0.3223 / 0.3435, c 0.3217 / 0.3435, d 0.3215 / 0.3434, e 0.3221 / 0.3419, f 0.3200 / 0.3409, g1 0.3216 / 0.3420, g2 0.3213 / 0.3421, p1 0.3168 / 0.3354, p2 0.3166 / 0.3378: all ten show the +0.019 to +0.022 in-sample gap (SE 0.0017). The cross-fitted file `teacher_oof` as control: 0.3210 / 0.3246 (+0.0036) | yes, and stronger than the study's check |
| teacher NDCG, holdout / in-sample feeds | mean11 0.32370 / 0.34388; mean25 0.32286 / 0.34412 | 0.32370 / 0.34519; 0.32286 / 0.34478 (another in-sample feed sample; SE 0.0012) | yes |

### 1.4 Code path

| claim | study | mine [M] | agree |
|---|---|---|---|
| flag off, static: copy minus the guarded statements == original, both harness files | AST PASS; 1 import, 1 add_argument, 1 `fd = None`, 1 `if a.holdout_in_train`, 7 `if fd is not None`, 2 `hold if fd is None else None`, 2 `and fd is None` | own constant-folding AST check (`rv_ast.py`): equal for both files against the sources frozen at hand-over (md5 e4b5f0d7 / 1317bdae), no residual `fd` reference; same counts 1 / 1 / 1 / 1 / 7 / 2 / 2 | yes, for the frozen sources (see issue 1 for the live file) |
| same for the fine-tune copies | AST PASS; 1 / 1 / 1 / 1 / 2 `if fv` / 5 conditional expressions | equal for both; 1 / 1 / 1 / 1 / 2 / 5 | yes |
| flag off, end to end | 3 pairs identical | own pair A_orig / B_copy_off (`rv_cpu.py`; W16 recipe arm kd05t25w192 seed 953 + smoke settings + `--cooldown_mix 0.3` so that the pool and its leak guard run; eval feeds chosen by hash over the whole split, not the first n): 23 log lines, feed ids of 73 training steps, targets, per-step loss (3.426935 first, 2.229349 last), pre-clip gradient norms, eval calls, 153 tensors of both checkpoints, 8 prediction files: all identical | yes |
| flag on: holdout feeds in the batches, target, steps, pool, files | 5 runs ALL PASS; 194 holdout feeds (seed 951), 72 -> 73 steps, pool 1,004 = 810 + 194, target diff 3.1e-08 | C_copy_on (same configuration as A / B + flag): 179 of the 20,000 feeds are holdout feeds; 161 of them in the 68 regular steps and 61 holdout feed-occurrences in the 5 pool batches (16.0% of 381; pool share 18.0%); pool 995 = 816 + 179; target of all 222 holdout feed-occurrences == 0.5 y/sum(y) + 0.5 softmax(teacher_mean25 rows): max abs diff 3.0e-08; steps 73 == my own count; BEST step 73 == total; kept == @last bit for bit; only `*_holdoutINSAMPLE.npy` in `preds/holdout`. D_lane_P_on (the launcher's own `FD_SMOKE` P line, narrow popularity-light student, `train_full_abl.py`): 194 holdout feeds, all in the regular batches, 72 -> 73 steps, pool 1,004, target diff 3.0e-08, same file and checkpoint items | yes |
| flag off run has no holdout feed in any batch; other feeds unchanged | yes | 0 holdout feed-occurrences in B; feed list of C minus the holdout == feed list of B; every step of B == my recomputed batch or a pool batch | yes |

### 1.5 Fine-tune record, noise bands, calibration

| claim | study | mine [M] | agree |
|---|---|---|---|
| row 99 = ONE model on 80% minus mean of FIVE on 64%, 3 members | -0.00025 +- 0.00018 (z -1.37), n 14,423 | -0.00025 +- 0.00018 (z -1.37), n 14,423; members -0.00028 / -0.00054 / +0.00007 | yes |
| production OOF (another 80% seed) minus the five | -0.00052 +- 0.00017 | -0.00052 +- 0.00017 | yes |
| pool level (z-mean of 3 members) | -0.00017 +- 0.00025; other seed -0.00054 +- 0.00023; 6 x 80% -0.00034 +- 0.00023 | -0.00017 +- 0.00025; -0.00054 +- 0.00023; -0.00034 +- 0.00023 | yes |
| null band: 60 random 8 vs 8 splits of the W16 @last members on val | s8 0.00042 (max 0.00100) -> s16 0.00030 -> band 0.00090 | s8 0.00040 (max 0.00101) -> s16 0.00028 -> 0.00085 (my own 60 splits; the RMS of 60 draws has about 9% error) | yes |
| member SD: val / holdout (@last) | 0.00048 / 0.00030 (0.32199-0.32303) | 0.00048 / 0.00030 (0.32199-0.32303) | yes |
| A/A of the re-cool harness, 6 (ref2, ref) pairs | +0.00014, -0.00041, -0.00139, -0.00004, -0.00035, -0.00022 (each +- 0.00058); RMS 0.00062; pool s_AA(6) 0.00053 (max 0.00130) -> 0.00032 at 16 | identical to the last digit; two more pairs (l 955, l 957) have finished since | yes |
| in-sample gap of an original W16 member (A2 calibration) | 0.34631 in sample vs 0.32219 holdout, +0.0241 +- 0.0019 | re-predicted on the CPU with my own script (`rv_gap_same.py`, the project's teacher-dump path, the same feed rule): 0.34631 +- 0.00150 (n 20,000) vs 0.32219 +- 0.00115 (n 30,000), +0.02412 +- 0.00189; its teacher 0.34317 / 0.32370; student minus teacher +0.00314 +- 0.00061 in sample, -0.00151 +- 0.00044 holdout. CPU against the saved GPU file on MY 2,000 holdout feeds: centred corr 0.99995, NDCG -0.00092 +- 0.00049 (study, other feeds: +0.00014 +- 0.00022). Independent second view: the 4-member student dumps h1 / h2 read 0.3497 / 0.3494 on 30,000 in-sample draw-set feeds vs 0.3241 / 0.3241 on the holdout (+0.0256 / +0.0254 +- 0.0017) | yes |
| hz view on the holdout feeds | 3.2% of rows changed (zero rate 0.220 -> 0.245) vs 12.6% | 3.2% (0.220 -> 0.245) vs 12.8% on another sample of exploration-like feeds; `feat_new/hz_common.py` forces lag 0 for the holdout [D] | yes |
| popularity-light fine-tune re-reads 7 of the 8 excluded columns | harness log line | the line is in the production log `nn_x/abl/logs/abl_e10_pop1kd05t25_s951_n2cpv3s3_ft.log` (line 4) | yes |

### 1.6 Launcher and hygiene

| claim | study | mine | agree |
|---|---|---|---|
| launcher commands == the members' logged arguments | 48 of 48 | 24 F / P commands parsed with the copies' own parsers (extracted by AST, not the study's helper) == the first-line arguments of the 24 members' training logs in all 89 keys except `variant` / `holdout_in_train`; 16 R commands == `lane.sh` K:ref / L:ref DRY output + the flag, token for token (re-run after the 04:52 change of `lane.sh`) | yes (the 8 PR lines were not compared separately; one was executed, see 2.4) |
| GPU guard | copied from `lane.sh` | `check_gpu` body identical line for line to the live `lane.sh` (04:52 version); E_BASE, B_BASE, HYG identical; same lock file `nn_x/w5/logs/locks/gpu<g>.lock`; flock taken before `check_gpu`, released after "steps/epoch"; 12-thread exports; PID written to two RUNNING files; `DRY=1` prints only and creates nothing (verified: no file appeared); refuses unknown stage / arm / seed / 4th field / GPU outside `W5_GPUS` | yes |
| read-only files untouched by the study | none changed | `nn_x/abl/train.py` 11-14 07:25, `nn_x/w5/cd_arms6.py` 11-12 22:05, `nn_x/w5/finetune.py` 11-12 19:33, `nn_x/abl/popfree/finetune.py` 11-13 07:13: md5 == the study's frozen copies; `src/`, `feat_new/`, `eda/`, `subs/`: no file newer than 02:18 UTC; no `__pycache__` / `.pyc` created in `nn_x/w5`, `nn_x/abl`, `src`, `feat_new`; no twin- or smoke-named file in the harness `ckpt / logs / preds`; `nn_x/w5/train.py` and `lane.sh` were edited at 04:52:01 and 04:57:57 UTC, 31 and 37 minutes AFTER the hand-over, by their owner (new flag `--xt_drop_init`, hz import from the new `cd_arms6b.py`, new K / L arms), see Issue 1 | yes |
| no GPU process started by the study | none | at 04:50 UTC (before the lead's first launch at 04:57): 33 compute processes on the the GPUs, all `train.py` / `finetune.py` of the harness lanes (variants `w5_e10_kw*`, `w5_e10_kd08w192_*`, `w5_[kl]cd_{ref2,hz,th,xd30}_*`, one pop1kd05w192 fine-tune); none has `stale/fulldata`, `train_full`, `holdout_in_train` or `full_val` in its command line; no `RUNNING_full.txt` exists anywhere | yes |

## 2. Code audit

### 2.1 Flag off

The copies are the originals plus guarded statements only. My own check (`rv_ast.py`, written without reading the study's test) parses
both files, removes the module docstring, constant-folds `fd := None` and `a.holdout_in_train := False` (drops `if fd is not None:` /
`if a.holdout_in_train:` blocks, which have no `else`; rewrites `X if fd is None else Y` to `X` and `A and fd is None` to `A`; drops the
`cd_full` import, the `fd = None` assignment and the one `add_argument`) and compares the syntax trees:

| pair | result |
|---|---|
| `train_full.py` vs the source frozen at hand-over (`ref/train_w5.py.txt`, md5 e4b5f0d7, == the live file until 04:52 UTC) | equal, no `fd` reference left |
| `train_full_abl.py` vs the live `nn_x/abl/train.py` (md5 1317bdae, unchanged) | equal |
| `finetune_full.py` / `finetune_full_abl.py` vs the live `nn_x/w5/finetune.py` / `nn_x/abl/popfree/finetune.py` | equal |
| `train_full.py` vs the LIVE `nn_x/w5/train.py` after 04:52:01 UTC (md5 26d254d0) and after 04:57:57 UTC (md5 e5391fb8, current at 05:26) | NOT equal: the live file has one more `add_argument` (`--xt_drop_init`, default -1.0), a 3-line block `if a.xt_drop_init >= 0:` inside `if a.init:`, and (second edit) imports the hz functions from `cd_arms6b` instead of `cd_arms6` (Issue 1) |
| `rederived2/train_full.py` (the study's patch on the e5391fb8 file) vs the live file | equal, counts 1 / 1 / 1 / 1 / 7 / 2 / 2 |

What the folding does not cover, checked by reading: `cd_full.py` imports only `json` and `numpy` and executes no statement at import
besides two constants and a class; no file of `nn_x/w12/stale/fulldata/` has the name of a module of `nn_x/w5`, `nn_x/abl` or `src`
(the script directory is first on `sys.path` in the launcher layout), and there is no `encdose` package there; the new argument does
not shadow an existing one (`--holdout` stays an exact match; every launcher command parses to the members' logged values, 1.6).

End to end: my pair A_orig / B_copy_off of 1.4 (run 04:25-04:36 UTC, i.e. on the e4b5f0d7 file; md5 read at 04:23 and 04:26).

### 2.2 Flag on (code read + runs C, D, E of this review)

| question | answer | evidence |
|---|---|---|
| are the 30,000 holdout feeds in the regular batches? | yes | `feeds = fd.put_back(feeds)` returns the list before `np.setdiff1d`, asserted to be "after + holdout"; every later consumer reads `feeds`. Runs: 179 / 194 / 186 holdout feeds of the 20,000-feed subsamples in the batches (C / D / E), recomputed from the seed with my own batching; at full scale the count is arithmetic (9,984 steps), not a run |
| are they in the cooldown pool? | yes | `exploration_pool(feeds, prand_np, a.pool_p, None)`: the only use of the 4th argument is the leak-guard assert (`nn_x/w5/cd_arms.py` lines 47-55). C: 61 holdout feed-occurrences in 5 pool batches; E: 11 in 1 |
| is a val or test label introduced? | no | the patch adds no data source. Val is read by `predict(model, va)` / `eval_ndcg` for the logged `val_ndcg` only; `--val_mix` is refused with the flag; test has no labels |
| is an in-sample teacher value introduced? | no | the target is `kd[b["ci"]]` of the same file as today; on the holdout rows every teacher component is out of sample (1.3). The reverse holds on the other 99% of the feeds, as today |
| is the checkpoint choice free of the in-sample holdout? | yes | `sel = -math.inf` at every eval and `a.keep = "last"`: `best` is assigned only by `(a.keep == "last" and last)`; `--time_limit` (the only early exit) is refused. Runs: BEST step == total, kept == @last in all 153 tensors (C, D, E). Val is never a selection input either |
| is the in-sample holdout number kept out of readers' way? | yes | regular log field, history entry and snapshot field are nan; the number is on its own "FULLDATA step N: holdout IN-SAMPLE" line and in `<v>_fulldata.json` |
| are the file names unmistakable? | yes for the files; see issue 4 for the checkpoint metadata | `preds/holdout` of C, D, E holds only `nn_<v>_holdoutINSAMPLE.npy` and `nn_<v>@last_holdoutINSAMPLE.npy`; `save_last` gets `hold=None`, `save_holdout` / `save_holdout_hz` are skipped. A regular expression `nn_(.+)_holdout\.npy` does not match them; the family discovery of `nn_x/w12/stale/veto/read8.py` requires an out-of-sample holdout file and therefore skips twins |
| other effects of the flag | feature statistics of a from-scratch run are computed on a feed list that includes the holdout (40 random batches, tiny); a re-cool keeps the snapshot's statistics; `--save_steps` snapshots are still written | code read |

### 2.3 Launcher

`DRY=1 bash nn_x/w12/stale/fulldata/lane_full.sh 0 "<all six stages>"` prints one command per job and writes nothing
(`logs/rv_dry_lane_full.txt`); wrong arm / seed / stage / a 4th field / a GPU outside `W5_GPUS` are refused with exit code 1.
The guard is the `lane.sh` guard: same `check_gpu` body (<= 3 compute processes, >= `LANE_GB` = 60 GB free), same lock file for
both harnesses, lock held from before the check until the job logs "steps/epoch", 12-thread exports, `CUDA_VISIBLE_DEVICES` = the
one id, PID appended to two RUNNING files, exact-PID stop note. The non-DRY branch was read, not executed (nothing may be launched
on a GPU by me). Two operating facts for the lead, neither a defect: the default `W5_GPUS` is "<ids>", so GPUs N need
`W5_GPUS="<ids>"` as in the study's command 2; and one lane runs its list one job after the other (the originals took
193-209 minutes each at four processes per GPU), so the 20 retrains need one lane per GPU slot, and a lane whose GPU never drops
to three processes within `WAIT_MAX_H` = 6 hours exits with code 3 and must be started again.

### 2.4 Forms the study never executed, executed here on the CPU

| run | form | result |
|---|---|---|
| D_lane_P_on | the launcher's own `FD_SMOKE` line for `P:pop1kd05t25:951` (narrow popularity-light student, d128, teacher_mean25; the study's popularity-light runs were the wide student) | ALL PASS (1.4) |
| E_lane_PR_on | the launcher's own `FD_SMOKE` line for `PR:pop1kd05t25:951` = re-cool of `abl_e10_pop1kd05t25_s951@77500.pt` with the flag (no abl re-cool had been run by the study, with or without the flag) | ALL PASS: 27 popularity-light inputs come from the snapshot cfg; 186 holdout feeds in the feed list, 185 in the 73 regular steps, 11 in the one pool batch; target max abs diff 3.1e-08; 74 steps; BEST == total; kept == @last; only in-sample-named holdout files |

## 3. Issues, by severity

### Issue 1 (MAJOR for the hz stages RH / FH; no effect on stages F / P / R as written): the live `nn_x/w5/train.py` is no longer the file the copy was made from

The owner of the harness edited `nn_x/w5/train.py` twice after the hand-over (04:21 UTC). [M: md5 and `diff` against the study's frozen
`ref/train_w5.py.txt`.]

| time (UTC) | md5 | change | kind of failure if the delivered copy is used for it |
|---|---|---|---|
| until 04:52:01 | e4b5f0d7 | the source of `train_full.py` | - |
| 04:52:01 | 26d254d0 | new flag `--xt_drop_init` (one `add_argument`, default -1.0; a 3-line block `if a.xt_drop_init >= 0:` inside `if a.init:`); `lane.sh` arms K / L : xd15, xd30, xd50, hzxd, s3 | LOUD: the copy's parser rejects the flag |
| 04:57:57 | e5391fb8 | `from cd_arms6 import save_holdout_hz, setup_hzcd` became `from cd_arms6b import ...`; `cd_arms6b.py` reads the environment variables `HZCD_DIR` / `HZ2_KEEP` (the "hz2" popularity view, more swapped columns, kept rows); `lane.sh` exports them for the arms hz2, h2t, h2x, hz2L | SILENT: the copy imports `cd_arms6`, which ignores both variables; `lane_full.sh` does not export them. An RH / FH twin of an hz2-family member would train the OLD hz view and nothing would stop it |

Consequences, in order of weight:

1. Stages RH and FH of the delivered launcher are valid only for the old `--hzcd` recipe (arm hz, `data/feats_new`). If the staleness
   verdict selects an hz2 arm or an `--xt_drop_init` arm, a twin made with the delivered files is either impossible (flag rejected)
   or, worse, a twin of a different recipe (hz2 silently replaced by hz). `accept_twins.py` would not see the second case: the
   environment variables are not arguments, so A1 compares equal; only the "[6b: ...]" part of the hzcd log line differs.
2. `accept_twins.py` A1 "arguments == the reference run's arguments" gives a FALSE DEFECT for RH twins whose reference re-cool ran
   on the edited file. Shown with real logs (`rv_a1_sim.py`, the script's own lines of logic): `RH:kd05w192:951` against
   `w5_kcd_hz_s951` (started 04:52:43) and `RH:kd05t25w192:953` against `w5_lcd_hz_s953` (started 04:52:30) differ in
   `xt_drop_init: (-1.0, <absent>)`. Stage R against the 16 finished K:ref / L:ref re-cools and stages F / P against the original
   members are not affected (those references ran on the old file).
3. "Flag off = the original bit for bit" is proven against the 04:20 source, not against the live file (my AST check: FAIL against
   the live file, PASS against the frozen one). For stages F, P, R, PR this is a statement about the proof only: no command of
   theirs passes `--hzcd` or `--xt_drop_init`, the two edits execute nothing without them (`cd_arms6b` runs no statement at import
   besides imports, constants, one class and function definitions), and `nn_x/abl/train.py` is unchanged (md5 1317bdae).

State of the launch (read-only observation at 05:09 UTC): the lead has already run `FD_SMOKE` on GPU N (`F:kd05w192:951` 04:57-05:01,
`P:pop1kd05w192:951` 05:01-05:04, both rc 0) and started 20 lanes at 05:01-05:05 (16 stage F, 4 stage P), all of them waiting in
the GPU guard at 05:08 ("GPU N: 4 compute processes ... need <= 3"). These are stages F and P: Issue 1 does not touch them and
nothing has to be stopped. Do NOT edit `lane_full.sh` or `train_full.py` in place while those lanes wait (bash reads a running
script from disk; python reads `train_full.py` when each job starts): put a re-derived copy and its launcher under new file names.

Fix for the hz stages (minutes): re-derive the copy from the live file. The study's recorded patch no longer applies in full: 10 of 11
hunks apply with offsets, the import hunk is rejected because its context line changed, and the import has to be re-inserted by hand
(without it the file still passes a naive "equal after stripping" test and then fails with a NameError when the flag is given; my
`rv_ast.py` now requires the exact counts). Candidate in my directory: `nn_x/w12/stale/fulldata_review/rederived2/train_full.py`
(md5 7af994ea; source e5391fb8; + an unchanged copy of `cd_full.py`): AST-equal to the live file after folding, counts 1 / 1 / 1 / 1 /
7 / 2 / 2; it differs from the delivered copy in exactly the two edits above. The launcher additionally needs the `HZCD_DIR` /
`HZ2_KEEP` exports of `lane.sh` for hz2 arms, and stages for the new arms. The live files moved three times tonight (03:35, 04:52,
04:57): whoever launches an hz twin compares the md5 of the live `train.py` with the copy's source at launch time (the study's
`test_full.py` does it in its first item).

### 3.1 The re-derived candidates, end to end (CPU)

All on the re-cool form, where the 04:52 edit sits: the live `lane.sh` smoke line for `L:ref:953` (snapshot
`w5_e10_kd05t25w192_s953@77500.pt`, seed 1953, teacher_mean25) + `--cooldown_mix 0.3`, 74 steps.

| pair / run | what | result |
|---|---|---|
| G_orig_live_L vs H2_rederived2_L_off | the LIVE `nn_x/w5/train.py` (md5 e5391fb8, started 05:01:04) against `rederived2/train_full.py` (source e5391fb8) without the flag | identical in all 24 items: 24 log lines, feed ids of the 74 training steps (15,501 feed-occurrences, pool batches included), targets, per-step loss (1.415815 first, 0.655864 last), gradient norms, 153 tensors of both checkpoints, 8 prediction files |
| G_orig_live_L vs H_rederived_L_off | the same live run against `rederived/train_full.py` (source 26d254d0: has `--xt_drop_init`, still imports `cd_arms6`) | identical in all 24 items: on a command without `--hzcd` the import change of 04:57 has no effect, as claimed in consequence 3 |
| I_rederived_L_on (reference H) | `rederived/train_full.py` + `--holdout_in_train` = the stage-R form for the L family | ALL PASS: 194 holdout feeds in the feed list (19,806 -> 20,000); 118 of them in the 45 regular steps and 363 holdout feed-occurrences among the 1,886 of the 30 pool batches (19.2%; pool share 19.4%); target of 481 holdout feed-occurrences == 0.5 y/sum(y) + 0.5 softmax(teacher_mean25 rows), max abs diff 6.4e-08; steps 74 -> 75 == my count; BEST step == total; kept == @last; only in-sample-named holdout files |

Not run: the candidates with `--hzcd` (old or hz2 view) and the flag; the hz stages need that run, and a GPU smoke, before use.

### Issue 2 (MEDIUM): the A2 bar is safe but blunt; the study's expected lift is too low by a factor of 2 to 3

- R7 reasons: on the holdout rows a twin's teacher half is out of sample, so "expected lift about 0.5 G_s = +0.0086 [I]"; bar
  0.25 G_s = +0.0043.
- A direct measurement exists in the project and contradicts the factor 0.5 [M, `rv_gap.py`]: `w5_e10_kx05_s951` is the same
  student (d192, kd_w 0.5, 10 epochs, holdout excluded) trained on labels + the CROSS-FITTED teacher `teacher_oof.npy`, i.e. with an
  out-of-sample teacher on every training feed, which is exactly the situation of the holdout feeds in a twin. Its @last checkpoint
  reads 0.34695 +- 0.00152 on 20,000 draw-set training feeds against 0.32189 +- 0.00114 on the holdout (n 30,000). Relative to its
  own teacher on the same feeds: +0.02164 +- 0.00068 in sample, +0.00088 +- 0.00048 on the holdout: difference in differences
  +0.0208 +- 0.0008. The student with the in-sample teacher reads 0.34631 on the study's sample. So the label half alone carries the
  student to the same in-sample level; the expected A2 lift of a full-retrain twin is about +0.02 (W16) and about +0.017
  (popularity-light), not +0.009.
- Effect: the bar of +0.0043 will be cleared by a factor of 4 to 5 by a correct twin (no false DEFECT risk), but a twin that saw the
  holdout feeds in only a part of the passes would also clear it. A1 covers that case through the run's own per-epoch count, so
  this is not a hole, only a weak second lock. I apply R7 as written and add an information line for the lead: a full-retrain twin
  whose A2 lift is below +0.012 should be inspected even though it passes. For re-cool twins (bar +0.0017) I have no measurement;
  the study's own doubt about that bar stands.

### Issue 3 (MEDIUM): hand-over command 5 ("change the two member-name lists in a copy of build_r3.py") stops on two asserts

- `nn_x/w12/kd/build_r3.py` checks that the W8 branches it builds reproduce the stored `preds/ens_W8kd_{ft,to}_{val,test}.npy`
  (`assert c > 0.999999`), and before writing it asserts that neither `subs/cand_<tag>_r3.parquet` nor `preds/ens_<tag>_*` exists.
  With twin names in `W` the first assert fails (the twin W8 is another set of models), and the tags W16kd / W8P20kd exist
  (12 `ens_*` files and both candidate files are on disk). [D: code read; M: the files exist. The script was not run.]
- Both fail loudly; nothing wrong can be written. The lead needs a copy with new tags (for example W16fd / W8P20fd), with the
  reproduction check kept on the ORIGINAL W8 names and the twins loaded under separate name lists, so that the printed deltas are
  "twin file minus the scored W8", which is also the comparison the presub gate needs.

### Issue 4 (MEDIUM for later work, none for the launch): a twin checkpoint still says `holdout: 30000`, `select_on: 'holdout'`

- The project proves "this model never trained on the holdout" from exactly that field (kd_audit row 22, the study's
  `teacher_holdout.py`, my `rv_teacher.py`). A twin passes such an audit; only `holdout_in_train: True` (and `keep: 'last'`) tells
  it apart. `nn_x/w5/holdout_eval.py` and `nn_x/abl/holdout_eval.py` write `preds/holdout/nn_<v>_holdout.npy` for any variant
  name without reading the arguments (the study says so for `holdout_eval.py`; it does not mention the audits).
- Consequences to avoid: a twin as a member of a future teacher file (its rows on the holdout would be in-sample while the audit
  says out-of-sample), or a twin in any in-period holdout read.
- Mitigation: twins are used in final files only; every provenance audit adds `not args.get("holdout_in_train")`; the twin
  prefixes `w5_fd*` / `abl_fd*` are never passed to `holdout_eval.py`.

### Issue 5 (LOW): smaller points

- The value of the 30,000 feeds, +0.00005 to +0.0002 [I], is scaled by their share of ALL feeds. They are 22% more
  exploration-like material (30,000 on top of the 134,118 pool feeds), the kind the recipe itself up-weights in the cooldown.
  The range may be low by a small factor; it stays below the 16-pool noise (0.0003) and cannot be measured on the CPU.
- A file made of twins is a re-draw of the model set: it has no public score, and by seed noise alone it can differ from the
  scored original by a few 1e-4. With 5 submissions a day and the deadline on 12-09 the twin file should be scored on public
  before it is chosen as a final.
- Wall clock of the W16 originals at four processes per GPU: the logs say 192.7-208.7 minutes (the study's table says 180-209).
- Two more (ref, ref2) pairs have finished (l 955, l 957); `accept_twins.py null` can be re-run before the first re-cool twin, as
  R6 allows.
- My CPU predictions against the saved GPU holdout files on 2,000 feeds: -0.0007 +- 0.0004 and -0.0009 +- 0.0005 (two models, the
  same feeds), centred correlation 0.99995; the study saw +0.0001 and -0.0002 on other feeds. Rank flips between fp32 and bf16;
  two orders of magnitude below the A2 lift.

## 4. What the lead must do before using the deliverable on a GPU

1. Issue 1. The 20 lanes already waiting are stages F and P: the delivered copies train exactly the members' recipe there, and
   the only cost is that the identity proof refers to the 04:20 source; leave them as they are. BEFORE any hz twin (RH / FH, or a
   twin of an hz2 / `--xt_drop_init` member): re-derive the copy from the then-live `train.py` (candidate `rederived2/`; under a
   new file name, with a copy of the launcher that exports `HZCD_DIR` / `HZ2_KEEP` as `lane.sh` does), re-freeze the reference,
   re-run the study's `test_full.py` and `check_cmds.py`, smoke the hz form with the flag on the CPU and on a GPU, and read the
   md5 of the live `train.py` at launch time. Without this an hz2 twin is silently a twin of another recipe.
2. `FD_SMOKE=1` on a GPU: done by the lead for stages F and P (section 5). Still open for R / PR and for every hz form.
3. Check the P arm: the queued P lanes are `pop1kd05w192`; the W8P20 students are `pop1kd05t25` (section 5). One lane per GPU
   slot with `W5_GPUS="<ids>"`; a lane exits after 6 hours without a free slot and must be started again.
4. Acceptance: apply `accept_twins.py` as written; additionally inspect any full-retrain twin with an A2 lift below +0.012.
5. Build: a copy of `build_r3.py` with new tags and separate twin name lists (Issue 3), then the presub gate; score the twin
   file on public before choosing it.
6. Never pass a twin to `holdout_eval.py`, never list a twin as a teacher member; audits read `holdout_in_train`.

## 5. Observations about the launch that is already under way (read-only; nothing of it was touched)

| observation [M, 05:09-05:14 UTC] | why it matters |
|---|---|
| GPU smoke by the lead, the first GPU execution of the path: `w5_fdsmoke_10_kd05w192_s951` and `abl_fdsmoke_10_pop1kd05w192_s951`, both rc 0. Read with my code: "training feeds 19806 -> 20000", 194 holdout feeds per epoch, steps 73 (72 without), pool 1,004 with 194, every eval `holdout_ndcg nan` + an IN-SAMPLE line, BEST step 73, kept == @last in all 153 tensors, args `keep 'last'`, `holdout_in_train True`; only `*_holdoutINSAMPLE.npy` in `preds/holdout`; val / test predictions finite; my NDCG of the val file == the logged `val_ndcg` (0.34931 / 0.30837) and of the in-sample holdout file == the logged value (0.27476 / 0.24640) | the bf16 / TF32 execution that neither the study nor I could run works, with the same counts as the CPU runs of that configuration |
| the guard in operation: the smoke lane waited at 04:59:11 ("4 compute processes") and started at 05:01:12 ("3 compute processes, [n] GB free") | the launcher's guard does what `lane.sh` does |
| 20 lanes are waiting: `F:kd05w192:951-958`, `F:kd05t25w192:951-958`, `P:pop1kd05w192:951-954` | the four P jobs are the WIDE popularity-light students. W8P20, the team best, is built from `abl_e10_pop1kd05t25_s951-954` (`build_r3.py`: `P1n`); no `P:pop1kd05t25` job is queued. If the aim is twins of today's two finals, the four `pop1kd05t25` twins are missing and the four wide ones are in no current file. This may be deliberate; it is the opposite of the study's command 2 |
| each lane holds the per-GPU lock while it polls (as `lane.sh` does), and exits after `WAIT_MAX_H` = 6 hours without a slot (about 11:01-11:05 UTC) | twin lanes and the queued staleness-family lanes share one lock queue per GPU; lanes that time out leave no claim and must be started again |
| the study's own order was "full retrains after the hz verdict fixes the recipe" (its section 6.4, step 3) | F twins of the plain W16 recipe are twins of the final members only if the final recipe stays W16 / W8P20 |

## 6. Method

- Independent re-derivation: the holdout from the rule (p_random > 0.9 with M < 50 zeroed, M >= 50, `default_rng(12345)`), rows from
  `feeds.parquet`, labels from `cand_target.npy`; NDCG re-implemented (binary gain, pessimistic ties) and compared with
  `src/metrics.ndcg_per_feed` on a tied random score; pools = member mean of per-feed z; deltas paired per feed, mean +- SE, n given.
- Identity: my own AST folding (`rv_ast.py`) and my own CPU runner (`rv_cpu.py`: `CUDA_VISIBLE_DEVICES=""`, `torch.cuda` blocked,
  `torch.device("cuda:0")` mapped to the CPU, outputs redirected into my `runs/<tag>/`, recorder of feed ids / targets / loss /
  gradient norm; evals restricted to a hash-selected subset of feeds). Each train file is executed unchanged as `__main__`.
  Configurations differ from the study's on purpose: other arm and seed (kd05t25w192, 953), `--cooldown_mix 0.3` from scratch, the
  launcher's own DRY lines for the popularity-light forms, the L re-cool form.
- Flag-on checks recompute the feed list, the batches and the distillation target from the seed and the raw files.
- Read-only discipline: nothing outside `nn_x/w12/stale/fulldata_review/` and this report was written (the study's directory has
  new files since 04:22, all of them the lead's `lead_*.log` / `RUNNING_full.txt`); `PYTHONDONTWRITEBYTECODE=1` everywhere; no GPU
  process; no process signalled. Heavy processes: two at a time, 12 threads each; read-outs with 2-4 threads beside them.
- No new judging rule was needed: the study's R1-R7 are applied as written. The "+0.012 inspect" line of Issue 2 is information for
  the lead, not a rule.

## 7. Limits: what I did NOT check

- No GPU execution by me. The GPU evidence is the lead's two `FD_SMOKE` runs (73 steps each), read after the fact.
- No full-scale run: 9,984 steps per epoch, 99,840 steps and the 164,118-feed pool are arithmetic (it reproduces the logged
  9,835 / 98,350 / 78,680 / 134,118).
- `accept_twins.py`: A1 and A2 code read, the A1 argument logic replayed on real logs, the inputs of A3 re-derived (pool val NDCG
  0.40157, s16). I did not run its `null` / `selftest` modes (they write into the study's directory) and no real twin exists yet.
- The fine-tune copies: AST identity and code read only; the study's flag-on fine-tune run was not repeated. The 6-part nested
  design was read, not simulated.
- Stages RH / FH (`--hzcd` with the flag) were not re-run by me; the study's runs J and K cover the old hz view only.
- The non-DRY branch of `lane_full.sh` was read, not executed by me.
- The value of the 30,000 feeds and the share a re-cool captures remain inferences. The A2 bar for re-cool twins has no measurement.
- Inputs of the members (features built from val / test impressions, `pre_*` embeddings) were not audited, as in the study.
- `build_r3.py` was read, not run (Issue 3).

## 8. Commands (all from `<repo>`, with `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1` and 2-12 threads)

```
V=nn_x/w12/stale/fulldata_review; F=nn_x/w12/stale/fulldata
.venv/bin/python $V/rv_ast.py nn_x/w5/train.py $F/train_full.py nn_x/abl/train.py $F/train_full_abl.py     # at 04:26: ALL PASS; after 04:52: w5 FAIL
.venv/bin/python $V/rv_ast.py $F/ref/train_w5.py.txt $F/train_full.py nn_x/w5/train.py $V/rederived2/train_full.py
.venv/bin/python $V/rv_ast_ft.py nn_x/w5/finetune.py $F/finetune_full.py nn_x/abl/popfree/finetune.py $F/finetune_full_abl.py
.venv/bin/python $V/rv_facts.py ; .venv/bin/python $V/rv_teacher.py ; .venv/bin/python $V/rv_teacher2.py
.venv/bin/python $V/rv_row99.py ; .venv/bin/python $V/rv_aa.py ; .venv/bin/python $V/rv_hz.py ; .venv/bin/python $V/rv_cmds.py ; .venv/bin/python $V/rv_a1_sim.py
EA=$(cat $V/logs/EARGS.txt)                                                # E:kd05t25w192:953 smoke line + --cooldown_mix 0.3
bash $V/run_rv.sh A_orig <repo>/nn_x/w5/train.py <repo>/nn_x/w5 w5_smoke_e10_kd05t25w192_s953 $EA
bash $V/run_rv.sh B_copy_off <repo>/$F/train_full.py <repo>/nn_x/w5 w5_smoke_e10_kd05t25w192_s953 $EA
bash $V/run_rv.sh C_copy_on  <repo>/$F/train_full.py <repo>/nn_x/w5 w5_smoke_e10_kd05t25w192_s953 $EA --holdout_in_train
FD_SMOKE=1 DRY=1 bash $F/lane_full.sh 0 "P:pop1kd05t25:951 PR:pop1kd05t25:951 R:kd05t25w192:953" > $V/logs/rv_dry_smoke.txt
bash $V/run_dryline.sh 1 D_lane_P_on ; bash $V/run_dryline.sh 2 E_lane_PR_on
DRY=1 W5_SMOKE=1 bash nn_x/w5/lane.sh 0 "L:ref:953" | sed 's/^GPU N: //' > $V/logs/rv_lane_Lref953_smoke.txt
bash $V/run_laneline.sh G_orig_live_L <repo>/nn_x/w5/train.py --cooldown_mix 0.3
bash $V/run_laneline.sh H_rederived_L_off $V/rederived/train_full.py --cooldown_mix 0.3
bash $V/run_laneline.sh I_rederived_L_on  $V/rederived/train_full.py --cooldown_mix 0.3 --holdout_in_train
bash $V/run_laneline.sh H2_rederived2_L_off $V/rederived2/train_full.py --cooldown_mix 0.3
.venv/bin/python $V/rv_cmp.py identity A_orig B_copy_off ; .venv/bin/python $V/rv_cmp.py identity G_orig_live_L H_rederived_L_off
.venv/bin/python $V/rv_cmp.py identity G_orig_live_L H2_rederived2_L_off
.venv/bin/python $V/rv_cmp.py flagon C_copy_on B_copy_off 953 nn_x/w12/kd/out/teacher_mean25.npy
.venv/bin/python $V/rv_cmp.py flagon D_lane_P_on - 951 nn_x/w12/kd/out/teacher_mean25.npy
.venv/bin/python $V/rv_cmp.py flagon E_lane_PR_on - 1951 nn_x/w12/kd/out/teacher_mean25.npy recool
.venv/bin/python $V/rv_cmp.py flagon I_rederived_L_on H_rederived_L_off 1953 nn_x/w12/kd/out/teacher_mean25.npy recool
.venv/bin/python $V/rv_gap.py w5_e10_kx05_s951 20000 ; .venv/bin/python $V/rv_gap_same.py w5_e10_kd05w192_s951 20000
```

## 9. Files created (all in `nn_x/w12/stale/fulldata_review/`)

| file | role |
|---|---|
| `rv_cpu.py`, `run_rv.sh`, `run_dryline.sh`, `run_laneline.sh`, `rv_cmp.py` | own CPU runner, launch helpers, comparison (identity / flag on) |
| `rv_ast.py`, `rv_ast_ft.py` | own AST identity checks (train and fine-tune copies) |
| `rv_facts.py`, `rv_teacher.py`, `rv_teacher2.py`, `rv_row99.py`, `rv_aa.py`, `rv_hz.py`, `rv_cmds.py`, `rv_a1_sim.py` | re-derivations of section 1 and the Issue 1 demonstration |
| `rv_gap.py`, `rv_gap_same.py`, `rv_gap2.py` (unused variant) | in-sample gap of an out-of-sample-teacher student; the study's A2 calibration number |
| `rederived/` (source 26d254d0), `rederived2/` (source e5391fb8, current) | the study's patch applied to the live `nn_x/w5/train.py`, with the frozen source next to it; candidates, not deliverables |
| `logs/` | every output quoted here (`rv_*.txt`, one `.stdout` per run, `RUNNING.txt` with PIDs) |
| `runs/<tag>/` | 9 smoke runs (`cmd.txt`, `ckpt/`, `logs/`, `preds/`, `rec.npz`); deletable |
| `PROGRESS.md` | progress file |
