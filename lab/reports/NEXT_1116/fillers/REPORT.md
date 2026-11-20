# Lane B: never-run items as fillers

(Written by the lead on 2026-11-16 from the structured result of workflow a workflow run: the lane could not write report files itself. Text below is the lane's own, unedited.)

One of the five items is launch-ready (the paired fine-tune arms); three of the "never-run" items turned out to be run or closed by later records. No real run was launched; six plumbing smokes ran, all inside the limits. REPORT.md was not written: the harness refuses report files from a subagent, so this result carries the report.

VERDICT PER ITEM
1. Fine-tune settings: READY as a paired design on the 8 all-data wide students of FD8P20, 3 arms: half batch (bs05), double batch (bs2), extra-branch learning rate x3 (xm30). The lr arm is dropped (closed by bound for students, COVERAGE_MAP l.303); the wd arm is dropped as a no-op by arithmetic. About 9 GPU-h, prior of a pass 0.03 for the family.
2. CTXDROP / MODSUP: NOT READY. Neither flag exists in nn_x/w5/train.py or anywhere in the repo. Stopped there, no model code written.
3. P8 aux normalisation: DROPPED. The fix is --aux_norm feed (AUXF); it ran in the cooldown harness on 11-12 and closed (COVERAGE_MAP l.145). A kit for the retry under distillation is prepared and smoke-tested; default NO-GO.
4. Training batch size: DROPPED. It ran on 11-13 in three forms and closed (COVERAGE_MAP l.211, l.212). Same kind of kit (arm cbs05) prepared and smoke-tested; default NO-GO.
5. Deferred list 4.2 and the location-subspace row: no re-entry trigger is open (one line each in item 5).

WHY 3 AND 4 ARE DROPPED RATHER THAN OFFERED
Both were closed under hard labels in the same cooldown-harness form; the only untested form is the retry under distillation. That class has its own stop rule on record (ledger 2026-11-14T22:14Z item 10, read as MET at 2026-11-15T21:27:39Z: "closed under hard labels, retried under distillation ... is CLOSED"). The one retry since (kwwd0) entered as a stated exception. I found no such reason here, so the default is NO-GO; an exception costs the lead one command per kit.

RECORD ROWS THAT ARE STALE (I edited none)
- COVERAGE_MAP l.39 batch size NEVER: closed by l.211 / l.212; nn_x/w5/logs/E_batch_predeclared.txt cites that very row as its motivation.
- l.49 P8 NEVER: closed as AUXF, l.145 and CURRENT_STATUS l.559.
- l.98 fine-tune lr / batch / wd / extra_lr_mult NEVER: lr, wd, extra_lr_mult were in the 11-12 retune on hard-label trunks (l.139; batch was not in its search space); for students the learning-rate / epoch class is closed by bound (l.303).
- l.166: HDROP-ALL (l.184), RDROP-CD and SAM-CD (l.205) are closed; CTXDROP / MODSUP were never implemented.
- l.199 location-subspace OPEN: its CPU pre-gate ran as C03 and failed (l.260).
- CURRENT_STATUS 2026-11-16T14:11:33Z "remaining never-run items" lists batch size, P8 and fine-tune lr / wd among them.

SMOKES (one at a time, 5-minute cap with a watchdog on the exact PID, GPU with 2 compute processes and about [n] GB free at each launch; none hit the cap, nothing was signalled)
- 14:49Z fine-tune bs2, GPU N, pid 1978236, 66 s, exit 0.
- 14:50Z fine-tune bs05, GPU N, pid 1979416, 76 s, exit 0.
- 14:52Z K:cbs05:951 started directly (10,000 feeds, no test), GPU N, pid 1980478, 114 s, exit 0.
- 14:54Z K:cbs05:951 through the lane copy (W5_SMOKE=1), GPU N, pid 1982200, 135 s, DONE.
- 14:56Z K:auxf:951 direct, GPU N, pid 1983910, 114 s, exit 0.
- 14:58Z fine-tune xm30, GPU N, pid 1985569, 76 s, exit 0.
The smokes print NDCG of degenerate configurations (2 folds x 1 epoch; 10,000-20,000 training feeds); they are not reads. Rule texts and the reader were written 14:30-14:35Z, before the first smoke.

WHAT WAS TOUCHED
- The smokes wrote 45 files and one claim directory under nn_x/w5/{logs,ckpt,preds} with my smoke names. I moved the 16 logs to reports/NEXT_1116/fillers/smoke/artifacts/, deleted the 29 checkpoint / prediction files, removed the claim; a search for the three name patterns under nn_x/w5 returns 0.
- Two lines (START / END of pid 1982200) were appended by the lane copy to the shared registry nn_x/w5/logs/RUNNING_w5.txt, as every lane does.
- After smoke 4, one comment line of lane_f1116.sh was reworded (no code change; DRY prints byte-identical before and after).
- Not touched: the filler queue, its lanes, stop file and outputs; no stock script (md5 of nn_x/w5/lane.sh and nn_x/w12/kd/fill.sh equal before and after); no ledger, status, docs or memory file.

NOT RUN
No arm, no fresh control, no read of any arm. The path check of the day needs a 21-minute fine-tune, above the smoke limit: it is step 2 of the launcher and gates the arms. No stale-val scoring, no CPU build for any deferred item.

NOT ASKED, ONE LINE
The stock lane still carries declared, never-run harness arms of the 11-14 21:46Z family (kw03, m10, m0, e4, aux0, dcd0, dcd2); they need no lane copy. I did not price them.

## 1. Fine-tune settings: paired arms bs05 / bs2 / xm30 on the 8 all-data wide students of FD8P20

- ready: True
- GPU-hours: 9
- prior of a pass: 0.03
- dropped because: 
- jobs file: <repo>/reports/NEXT_1116/fillers/out/dry_ftarm_launch.txt (DRY print of the 1 + 24 commands; the job list is built by <repo>/reports/NEXT_1116/fillers/ftarm_launch.sh, this item does not go through fill.sh)
- launch: `cd <repo> && setsid nohup bash reports/NEXT_1116/fillers/ftarm_launch.sh "bs05 bs2 xm30" 4 > reports/NEXT_1116/fillers/logs/ftarm_launch.log 2>&1 < /dev/null &     # writes ftarm_DECLARED.txt (refuses a second run), re-runs one control + path check (anything but PASS stops), then 4 lanes; dry print: DRY=1 bash reports/NEXT_1116/fillers/ftarm_launch.sh "bs05 bs2 xm30" 4`
- read: `cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python reports/NEXT_1116/fillers/ftarm_read.py > reports/NEXT_1116/fillers/out/ftarm_read_out.txt`

Rule text:

PAIRED FINE-TUNE ARMS ftbs05, ftbs2, ftxm30 (8 all-data wide students of FD8P20, w5_fd10_{kd05w192,kd05t25w192}_s951-954; 5 folds x 3 seeds, production flags; script reports/TRICKS_1115/launch/finetune_mixft.py without --feed_w; control = the stored mixctl3 runs of Q4). Written before any run.
PATH CHECK OF THE DAY first: one control is re-run today (member w5_fd10_kd05w192_s951, suffix ftctl3c). With A = mixctl3, A' = mixctl3b (stored), C = today's run, d = 1 - correlation of the raw out-of-fold scores, n = |difference of member val NDCG|: PASS iff max(d(C,A), d(C,A')) <= 1.5 x d(A,A') AND max(n(C,A), n(C,A')) <= max(n(A,A'), 2 x se(A,A')). Stored band: d(A,A') = 6.559e-09, n(A,A') = 0.000131, 2 x se = 0.000129. FAIL -> no arm is launched and none is read against the stored controls.
RULE (on record, unchanged): arm minus control at FILE level: PASS iff val >= +2 SE with Holm over the arms AND private-mix >= 0 AND val M<50 >= -1 SE. A pass only buys a 3-seed replication on the 16 all-data members and then a candidate for PICK 2, never pick 1. Anything else: CLOSED.
Operational form: FILE(x) = 0.5 z(0.8 z(FT pool of the 8 members, x) + 0.2 z(stored FT pool of the 4 popularity-light students)) + 0.5 z(preds/ens_FD8P20kd_to_val.npy), paired per val feed (14,423), SE = paired per-feed SE, private mix = src/pmix.py weights. Holm step-down on the one-sided p of z = delta / SE, family alpha = P(z > 2) = 0.02275 over the k arms declared in reports/NEXT_1116/fillers/ftarm_DECLARED.txt (k = 3: z >= 2.43 for the best arm, then 2.28, then 2.00; the first failure fails every later arm). If several arms pass, the one with the largest val z goes to the replication. INCOMPLETE while any member of any declared arm is missing. One look, after all 24 fine-tunes are complete. Reader: reports/NEXT_1116/fillers/ftarm_read.py.
Forecast: no arm passes; P(pass) about 0.03 for the family. Power: the best arm needs about +0.0005 on val at the SE on record (0.00022 val, 0.00035 private mix); 50% at a true +0.0005, 14% at +0.0003. The bound on record for this class is +0.00005 (95% upper +0.0005), so a pass is about as likely a false pass (about 0.02 under no effect) as a true one: the replication decides, not this read.

Evidence:

STATUS PER KNOB. (a) batch (8192 candidates / 128 feeds): never varied in any recipe; it was not in the search space of the 11-12 retune (nn_x/ftopt/findings.txt). Arms bs05 (--cand_budget 4096 --max_feeds 64) and bs2 (--cand_budget 16384 --max_feeds 256). (b) extra_lr_mult (10): values 3/10/30 were in the 11-12 retune on hard-label trunks, where the extra branch itself was null; never varied on students, where the fine-tuned half is worth +0.0005 on public and that value is attributed to the cold promotion entering through the extra branch (reports/TRICKS_1115/int_ft.md verdicts 2, 6). Arm xm30 (--extra_lr_mult 30). (c) lr: DROPPED. Students: half-lr pilot member +0.0008 +- 0.0005, file 0.0000 +- 0.0003, and 'Stop testing FT recipe changes of the shrink class (learning rate, epochs, EMA, anchors ...)' (reports/RECHECK_1114/ft_redesign.md l.188, l.198); class bounded at +0.00005, 95% upper +0.0005 (COVERAGE_MAP l.303); epoch 1 beats the saved model in 20/20 members but reads -0.00017 +- 0.00034 at file level (int_ft.md verdict 6). (d) wd: DROPPED, no run needed: decoupled shrink over the whole fine-tune = wd x sum of lr = 0.02 x 0.0168 = 3.4e-4 of a weight (1.7e-3 at wd 0.1), computed from the val feed sizes (175 steps/epoch, 700 steps, cosine).
HONEST READING OF THE BATCH ARMS: at fixed lr and epochs the batch mainly sets the number of optimiser steps (345 / 175 / 91 per epoch), i.e. the dose of the same shift, so the bound of l.303 applies by inference; the two arms are the first direct GPU read of the dose curve at file level. They also move the EMA horizon in epoch units (0.29 / 0.57 / 1.10); left alone so each arm differs from control by one setting.
CONTROL REUSE. All 8 stored controls <member>_mixctl3 exist with val and test files (written 2026-11-15T23:35Z..2026-11-16T02:11Z), plus mixctl3b and mixone3 for s951 and the 8 production runs. Nothing the fine-tune reads changed since: checkpoints dated 11-15 09:55-10:19, finetune_mixft.py 11-15 22:10, harness modules older; the only feature files newer than 11-15 23:00 are nx_rp3_{z,rk}.npy, which these members do not read (0 matches in their logs). Stored path check: PASS (reports/TRICKS_1115/followup/logs/pathcheck_out.txt). A fresh control of today was NOT run by me; it is step 2 of the launcher and gates the arms.
KIT CHECKS (files in reports/NEXT_1116/fillers/out/). ftarm_pathcheck.py on stored files: mixone3 PASS with the Q4 numbers (7.926e-09 <= 9.838e-09; 0.000105 <= 0.000131), mixft3 FAIL (1.078e-05) as it must. ftarm_read.py --selftest mixft3 reproduces Q4 to the digit (file val -0.00007 +- 0.00022, private mix +0.00022 +- 0.00035, M<50 +0.00004 +- 0.00040); declared mode prints INCOMPLETE and no number while files are missing, and refuses without a declaration. DRY print dry_ftarm_launch.txt + dry_ftarm_token_diff.txt: for every job the whole logged command of the stored control (same member, --out aside) is the prefix; appended tokens are exactly [--cand_budget 4096 --max_feeds 64], [--cand_budget 16384 --max_feeds 256], [--extra_lr_mult 30]. parse_check_ftarms.txt: control parses to exactly the logged arguments of the stored control; each arm differs only in its own keys. GPU smokes through ftarm_lane.sh (2 folds, 1 seed, 1 epoch): bs2 66 s, bs05 76 s, xm30 76 s, all exit 0, epoch-0 out-of-fold NDCG 0.39951 (stored control logs 0.39952 for the same checkpoint).
COST BASIS: 21.4 minutes per fine-tune (mean of the 8 stored controls on shared GPUs); 24 fine-tunes + 1 control; about 2.5 h elapsed on 4 lanes. GPU guard as in mixft_lane.sh (<= 3 compute processes and >= 60 GB free, per-GPU flock non-blocking; FTARM_MAXP lowers it).
TWO POINTS FOR THE CRITIC, not in the rule on record: (1) the rule has no minimum effect, and the paired SE collapses when an arm barely moves predictions (control vs production: file val +0.00001 +- 0.00001); a tightening such as 'AND val delta >= +0.0003' would cost nothing. (2) The replication on 'the 16 all-data members' includes the 8 members of this screen; the clean confirmation is the read on the other 8 (s955-958 of both teachers).

## 2. CTXDROP and MODSUP (TRICKS_SWEEP_1113 section 3)

- ready: False
- GPU-hours: None
- prior of a pass: None
- dropped because: Flags missing: both need model code. Stopped on this item as instructed; nothing built, nothing run.
- jobs file: 
- launch: ``
- read: ``

Rule text:



Evidence:

nn_x/w5/train.py has no --retr_ctx_drop and no --modsup (argument list l.105-264). A search of every .py and .sh in the project for retr_ctx_drop, modsup, ctx_drop, ctxdrop returns nothing. nn_x/w5/model.py l.211: p = torch.softmax(lg, -1) in HistRetrieval is used as is (l.221), no dropout on the retrieval probabilities; no intermediate supervision head exists. Pricing on record, for the lead only: the sweep designed both as 10-epoch from-scratch pairs (2 x 4.5 GPU-h each, P about 0.05), not as cooldown arms; reports/TRICKS_1115/int_weak.md T19 (11-15) re-priced them at P 0.02 and 9-14 GPU-h each. MODSUP in a re-cool would also need new heads on a snapshot whose architecture comes from the checkpoint. Since the sweep every dropout direction retried under distillation closed (kw_drop0 null, kwd2 pooled z -1.88, kw256d2 -3.64; COVERAGE_MAP l.46).

## 3. P8 pool-batch aux normalisation (= AUXF, --aux_norm feed)

- ready: False
- GPU-hours: 9
- prior of a pass: 0.015
- dropped because: Closed by a later record in the same form: the cooldown arm of the fix ran 2026-11-12 with both signs negative (COVERAGE_MAP l.145, CURRENT_STATUS l.559). The only untested form is the retry under distillation, a class closed by the pre-declared stop rule (ledger 2026-11-14T22:14Z item 10, MET 2026-11-15T21:27:39Z); no reason for an exception found. Default NO-GO; the kit is complete and smoke-tested if the lead decides otherwise.
- jobs file: <repo>/reports/NEXT_1116/fillers/fill_jobs_auxf_1116.txt
- launch: `ONLY ON AN EXPLICIT EXCEPTION: cd <repo> && setsid nohup bash reports/NEXT_1116/fillers/fill_f1116.sh reports/NEXT_1116/fillers/fill_jobs_auxf_1116.txt 2 > reports/NEXT_1116/fillers/logs/fill_auxf_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &`
- read: `cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/kd/arm_read8.py rc:auxf     # val M<50, private mix, stale-val: nn_x/w12/kd/hz_read.py rc:auxf`

Rule text:

Harness rule on record, unchanged. 16 re-cools K:auxf:951-958 + L:auxf:951-958 (flag --aux_norm feed, nothing else). PASS iff 16-pool against the ref re-cools: val >= 0 AND holdout >= 0 AND pooled z >= 2.5; [2.0, 2.5) = candidate only; otherwise CLOSED. A holdout miss ALONE does not close it: then val >= 0 AND stale-val >= +2 SE AND val M<50 >= -1 SE decide. On a PASS: 16 all-data students of the arm, then the FILE read against FD16P20 (final val >= -1 SE AND private mix >= 0) -> candidate for PICK 2 only; no file from a re-cool arm itself. One look, after all 16 members are complete. Forecast: not passed.

Evidence:

The fix exists as a flag: --aux_norm feed (nn_x/w5/train.py l.235, loss line l.532-533; reports/INSIGHTS_1112.md l.245: 'AUXF ... i.e. the never-run REVIEW_SYNTHESIS P8'). It was run in the cooldown harness, roots 101-104 against m02ref, on 2026-11-12: member val -0.00010 (z -0.56), holdout -0.00012 (z -1.25), M<50 -0.00032, private mix +0.00013 -> STOP (nn_x/w5/logs/w5_eval_B_prestale.txt l.80, l.96; CURRENT_STATUS l.559 'AUXF is null in the cooldown form too'; COVERAGE_MAP l.145). COVERAGE_MAP l.49 is stale.
Still applicable to the distilled recipe? Mechanically yes: the aux line is the same under --kd_file, no assertion excludes the pair, and the smoke ran (K:auxf:951 direct, 114 s, exit 0, 'wave-5 AUXF' line logged with the teacher on). Nothing in the mechanism changes under distillation: the teacher term acts on the main target only, the cooldown mix is still 0.02 (pool batches are 2% of cooldown steps), and the aux weight has a flat dose-response there (kwaux1 x4 pooled z +0.34, closed; kwaux0 -0.0013).
KIT (only on an explicit exception by the lead): arm K:auxf|L:auxf in the lane copy reports/NEXT_1116/fillers/lane_f1116.sh (diff against nn_x/w5/lane.sh = 1 header line + 5 inserted lines, out/diff_lane.txt); filler copy fill_f1116.sh (lane path + a second own stop file, out/diff_fill.txt). DRY=1: K:ref / L:ref / K:kw08 / K:m05 from the copy are byte-identical to the stock lane's prints; the auxf command = the whole 49-token reference command + '--aux_norm feed' (out/dry_token_diff.txt); parsed arguments differ only in aux_norm (out/parse_check_harness_arms.txt). 16 snapshots and 16 ref re-cools present, 0 files of the arm exist. Start the filler copy only when no other fill.sh instance is polling (today's stock filler ended 14:48:48Z).

## 4. Training batch size (cooldown half batch cbs05 under distillation)

- ready: False
- GPU-hours: 11
- prior of a pass: 0.02
- dropped because: Closed by later records: 4x batch from scratch FAIL and cooldown batch 0.25x / 0.5x / 2x all STOP on 2026-11-13 (COVERAGE_MAP l.211, l.212). The only untested form is the retry under distillation, a class closed by the pre-declared stop rule (ledger 2026-11-14T22:14Z item 10, MET 2026-11-15T21:27:39Z); no reason for an exception found. Default NO-GO; the kit is complete and smoke-tested if the lead decides otherwise.
- jobs file: <repo>/reports/NEXT_1116/fillers/fill_jobs_cbs05_1116.txt
- launch: `ONLY ON AN EXPLICIT EXCEPTION: cd <repo> && setsid nohup bash reports/NEXT_1116/fillers/fill_f1116.sh reports/NEXT_1116/fillers/fill_jobs_cbs05_1116.txt 2 > reports/NEXT_1116/fillers/logs/fill_cbs05_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &`
- read: `cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/kd/arm_read8.py rc:cbs05     # val M<50, private mix, stale-val: nn_x/w12/kd/hz_read.py rc:cbs05`

Rule text:

Harness rule on record, unchanged. 16 re-cools K:cbs05:951-958 + L:cbs05:951-958 (flags --cand_budget 16384 --max_feeds 256 --eval_every 2000 --select_min_step 24000: half batch, about 39,340 cooldown steps; eval cadence and selection floor scaled by 2). PASS iff 16-pool against the ref re-cools: val >= 0 AND holdout >= 0 AND pooled z >= 2.5; [2.0, 2.5) = candidate only; otherwise CLOSED. A holdout miss ALONE does not close it: then val >= 0 AND stale-val >= +2 SE AND val M<50 >= -1 SE decide. On a PASS: 16 all-data students of the arm, then the FILE read against FD16P20 (final val >= -1 SE AND private mix >= 0) -> candidate for PICK 2 only; no file from a re-cool arm itself. One look, after all 16 members are complete. Forecast: not passed.

Evidence:

Run on 2026-11-13 in three forms, all under hard labels. (a) 4x batch at 10 epochs from scratch, bs4 and bs4lr4: val -0.00262 / holdout -0.00209 and -0.0023 / -0.0016, both FAIL (nn_x/w5/logs/w5_eval_E_bs4.txt, w5_eval_E_bs4lr4.txt; COVERAGE_MAP l.211). (b) Cooldown batch in the harness, roots 101-104 (w5_eval_B_wave8.txt; l.212): half batch val +0.00006 +- 0.00040, holdout +0.00034 +- 0.00023 (z 1.46), 4-root pool val -0.00013 / holdout +0.00021 +- 0.00031, stale-val +0.00053 +- 0.00040 -> STOP; double batch val -0.00029 -> STOP; quarter batch pool -0.0025 / -0.0013 -> STOP; 'the standard batch is the optimum'. The gradient-noise measurement of r6_recap 4.2 was the stated motivation of those runs (E_batch_predeclared.txt cites the map row), so it is spent. COVERAGE_MAP l.39 is stale.
Under distillation the batch was never varied (no cand_budget / max_feeds / cbs entry in the ledger). The stock lane has no K / L batch arm, so one arm needs a lane copy (shell case line, no model code). The one direction with non-negative signs under hard labels is the half batch; its 16-snapshot analogue would have read pooled z about 0.3 there against a bar of 2.5; no new mechanism under distillation is on record.
KIT (only on an explicit exception by the lead): arm K:cbs05|L:cbs05 in reports/NEXT_1116/fillers/lane_f1116.sh, flags identical to stage-B cbs05 of wave 8. DRY=1: the cbs05 command = the whole 49-token reference command + '--cand_budget 16384 --max_feeds 256 --eval_every 2000 --select_min_step 24000' (out/dry_copy_cbs05.txt, out/dry_token_diff.txt); parsed arguments differ only in cand_budget, max_feeds, eval_every, select_min_step (the last two are bookkeeping scaled by 2 as in wave 8, so the selected checkpoint is chosen among the same fraction of the run). The DRY reference parses to the logged arguments of the finished run w5_kcd_ref_s951 with no differing value. Smokes: K:cbs05:951 direct (10,000 feeds) 114 s exit 0, 75 steps, teacher loaded, pool batches drawn; K:cbs05:951 through the lane copy with W5_SMOKE=1 (claim, flock, GPU check, done-detection) 135 s, DONE, 135 steps, test predictions saved. Same caveat as for kw08 / m05: the ref re-cools ran 11-14 and the arm would run with today's train.py (11-15 07:42; added flags off by default).
COST BASIS: today's kw08 re-cools took 33-36 minutes each at 2-3 processes per GPU; the arm has twice the steps at half the batch and the smoke shows a half-batch step costs about half a full one.

## 5. Deferred list of TRICKS_SWEEP_1113 section 4.2 and the embedding location-subspace row (COVERAGE_MAP l.199)

- ready: False
- GPU-hours: None
- prior of a pass: None
- dropped because: No re-entry trigger is open: each is either not met, or was met and then spent by a later run. Nothing to launch; no build was in scope.
- jobs file: 
- launch: ``
- read: ``

Rule text:



Evidence:

No build, one line each.
- ANCHOR-REPLAY (trigger: wave-6 m04 < m02 with a visible late decline): HALF MET. m04 is below m02 at member level (val -0.00077 +- 0.00034, holdout -0.00046 +- 0.00019, w5_eval_B_wave6.txt) but flat at pool level (-0.00020 / -0.00009) and the last step is not worse than the selected one. Under distillation the anchor is already in the loss on pool batches (nn_x/w12/recheck/kd_reprice/out/classification.tsv row 167: 'moot'); the dose test is the m05 filler running now.
- MetaBalance / PCGrad on the aux (trigger: aa0 or ap1 moves the member by > +1 SE): MET 11-13, then spent. ap1 +0.0003 (z 1.6) fired it; the confirmation failed at 10 epochs (7-root pool holdout -0.0002, COVERAGE_MAP l.209) and under distillation (kwaux1 x4 pooled z +0.34, ledger 2026-11-16T02:38Z).
- POP-REL (trigger: pop-free Q1 readout in AND label-free format check): MET 11-13, never run. Its place was taken by the popularity-light students (adopted at 0.2); further views of the popularity inputs closed 11-15 (ledger 16:46Z; 20:49Z not replicated).
- MODDROP-POP (same trigger): MET 11-13, then closed. Its runnable form, popularity-input dropout, ran under distillation: xd30 val -0.00066 +- 0.00048, stale-val +0.00001; 'closes xd15 / xd50 / hzxd / s3 / h2x and the from-scratch kwxd arms' (ledger 2026-11-15T08:57Z).
- HZCOND (same trigger): MET 11-13, never entered. 'Prepared as code first; declared when ready' (ledger l.330) and never declared; its base view failed at file level three times (FDX8P20:hz -0.00041, FDX16P20:hz -0.00047, ENSB -0.00056) and the hz2n family closed (2026-11-16T02:38Z). It needs a CPU build of a lag column; TRICKS_1115 ranks it below G5 at P 0.03.
- FPCD (same trigger): MET and used. FPCD-lite closed 11-13: val -0.0010 (z -2.5), holdout -0.0012 (COVERAGE_MAP l.221).
- PRE2T (same trigger): MET, not used. No definition or run on record beyond the row it shares with FPCD; its sibling closed with the opposite sign.
- Cross-feed negatives (trigger: a new argument that the category block is not information-limited): NOT MET. No such argument on record; int_weak.md T19 (11-15) prices it at P 0.02.
- Score-conditioned second pass, not head-only or an M<50 head arm (trigger: a new design or an M<50 judge): NOT MET. No design on record; the judge it needs was never built.
- M50-HOLDOUT (attach to the next from-scratch family): NOT MET, never built (reports/RECHECK_1114/skeptic_duplicates.md l.149); HOLD-SHORT in arm_read8.py is the substitute, secondary only.
- Embedding location-subspace match, l.199 (NN screen only if the CPU nested pre-gate passes): NOT MET, the gate ran and failed. It is C03 of the closure batch (COVERAGE_MAP l.260): val -0.00008 +- 0.00030, holdout -0.00029 (z -2.08), best column +0.00022 +- 0.00014; re-screened 11-16 with the calibrated lens: 'at most 2.1 / 1.3: no' (reports/INPUTS_1116/residual_rows.md rows 260 and 199, '= C03'). Not a holdout-only rejection: val is null. Row l.199 should read CLOSED.

