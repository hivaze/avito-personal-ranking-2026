# Critic of workflow a workflow run

(Written by the lead on 2026-11-16 from the structured result of workflow a workflow run: the lane could not write report files itself. Text below is the lane's own, unedited.)

CRITIC of workflow a workflow run. Run 2026-11-16 15:06-15:29Z, CPU only, at most two 12-thread processes, no GPU process, nothing signalled. Nothing from this workflow is passed for launch.

CRITIC.md WAS NOT WRITTEN: report .md files are forbidden to subagents in this session, so this text is the report. Scripts and outputs: <repo>/reports/NEXT_1116/critic/.

VERDICTS
- Lane A, repaired click-strength input: NO-GO confirmed.
- Lane B item 1, fine-tune arms bs05 / bs2 / xm30: NO-GO as proposed. The kit is sound; the arms are covered by closures already on record.
- Lane B items 2-5: NO-GO, as the lane itself concluded.

LANE A (one re-read; critic/a01_reread.py, a02_placebo300.py)
The failing gate is real and fails on val as well as on the holdout, so the holdout-only clause is not reached.

| Read | All-feeds form, val twin | Exploration form, val twin | All-feeds form, holdout | Exploration form, holdout |
|---|---|---|---|---|
| Conditional test, lens of record, my call (bar 2.74, then 2.50) | +1.39 | +2.17 | +1.22 | +1.85 |
| Wald test, pool-only null | +1.61 | +2.32 | +1.40 | +1.94 |
| Click part against my placebo, 300 draws (bar 2.5) | +1.39 | +2.19 | +1.28 | +1.78 |
| NDCG at the best in-sample weight (upper bound) | -0.000002 | +0.00008 +- 0.00015 | +0.00028 +- 0.00016 | -0.000004 |

- The conditional-test reads equal the lane's to the digit. The lead's looser kill rule (2.5 on val and on the lag-view holdout) is missed too.
- One number moved: my first 60 placebo draws read the exploration form's click part on val at +2.70. The spread of those draws was 0.87; with 300 draws it is 0.98 and the click part is +2.19 (lane: +2.34 and +1.91).
- Pre-declaration holds: RULE.txt 14:29:06Z, first array 14:32:17Z, same checksum in every script log. The build implements the two declared forms; my recomputation of the entered value matches the stored arrays to 1e-7.
- Parity and leak scans: read in the lane's outputs, not recomputed.

LANE B ITEM 1, the five checks
(i) FAIL: closed by later records.
- bs05 / bs2: the batch sets the number of optimiser steps, i.e. the dose of the same shift. The lane states this and dropped the learning-rate arms for that reason (COVERAGE_MAP l.303; ft_redesign.md section 10 item 2). Three file-level dose reads are flat: half learning rate 0.0000 +- 0.0003; first epoch against saved, val -0.00017 +- 0.00034; fine-tune weight 0.60, val -0.00011 +- 0.00016.
- xm30: a larger cold-item offset through the extra branch. That dose is closed (ft_redesign.md l.146; COVERAGE_MAP l.281, l.355). The cited int_ft.md verdicts do not attribute the fine-tune half's value to it. Multipliers 3 / 10 / 30 were in the closed 11-12 retune.
- The rule leads with val, where the fine-tuned half reads +0.00003 against +0.00048 on public.
(ii) PASS. My dry print from the current scripts is byte-identical to the lane's. Own token diff: 25 of 25 commands are the stored control command plus exactly the intended flags.
(iii) Rule text equals the brief's wording; Holm thresholds correct. Missing: a sentence that no holdout witness exists for these members, and a minimum effect.
(iv) PASS. The reader reproduces the Q4 read on mixft3 to the digit (val -0.00007 +- 0.00022, private mix +0.00022 +- 0.00035). Path-check self-tests: mixone3 PASS, mixft3 FAIL, as they must.
(v) PARTIAL. No file can come from the read, but the replication has no decision rule and 8 of its 16 members are the screened ones.

Minor disclosure gap: ftarm_lane.sh and ftarm_launch.sh were re-saved at 15:04:21Z, after the smokes; the lane names only lane_f1116.sh. Harmless: the dry print is unchanged and ftarm_read.py, which holds the rule, is dated 14:33:25Z, before the first smoke.

IF THE LEAD OVERRIDES (stated exception, as for kwwd0): xm30 alone, about 3.2 GPU-h, with these written into the ledger first.
1. Declare xm30 only (one arm, val clause z >= 2.00). Do not run bs05 / bs2.
2. Add a minimum effect: val delta >= +0.0003. The paired SE collapses when predictions barely move (control minus production +0.00001 +- 0.00001 in my run).
3. Write the replication rule now, on the other 8 members (seeds 955-958 of both teachers) against their stored production fine-tunes. Suggested: val >= +1 SE, private mix >= 0, val M<50 >= -1 SE on those 8 alone; only then a 16-member file, gate PASS, as a pick-2 candidate.
4. State that no holdout clause exists, and that a miss on the val clause is recorded as "not shown on val" with the private-mix number beside it.

STALE RECORD ROWS (confirmed by reading; I edited none): COVERAGE_MAP l.39, l.49, l.98, l.166, l.199, and the "remaining never-run items" list in CURRENT_STATUS 2026-11-16T14:11:33Z. For l.98: learning rate and batch are closed by the bound of l.303, weight decay is a no-op by arithmetic (0.02 x 0.0168 = 3.4e-4 of a weight), the extra-branch multiplier is the cold-offset dose.

NOT RUN: any GPU job, the path check of the day, any arm; own parity, leak-scan and nested-NDCG recomputation for lane A.

TOUCHED: only reports/NEXT_1116/critic/. The dry print created no declaration file.

## Click-strength: NO_GO

The failing gate is real, and it fails on val, not only on the holdout. I recomputed the conditional test from the lane's stored arrays with my own script and got the lane's reads exactly: val with the private-lag column +1.39 (all feeds form) and +2.17 (exploration form); holdout +1.22 and +1.85. The bar is 2.74 for the better form, and nothing reaches a plain 2.5 either, so the lead's looser kill rule (2.5 on val and on the lag-view holdout) is missed too. An independent Wald test under a pool-only null reads +1.61 / +2.32 on val and +1.40 / +1.94 on the holdout. My own exposure-preserving placebo (300 draws, own permutation code) gives a click part of +1.39 / +2.19 on val and +1.28 / +1.78 on the holdout against a bar of 2.5. A first pass with 60 draws read +2.70 for the exploration form on val; that came from a small-sample spread of 0.87, and 300 draws give 0.98. Direct NDCG with the organisers' metric, using the best in-sample weight (an upper bound on a nested fit), is -0.000002 and +0.00008 +- 0.00015 on val; on the lag-view holdout it is +0.00028 +- 0.00016 and -0.000004. The rule file predates every array (14:29:06Z against 14:32:17Z, checksum unchanged), and the build implements the two declared forms and no other. Parity and the leak scans pass in the lane's outputs; I read them and did not recompute them, since a NO-GO does not need them.

Checks re-run: File times: RULE.txt 14:29:06Z (creation time = modification time; md5 5b526db8..., the value every lane script printed); first array 14:32:17Z; first witness output 14:36Z. The aborted first launch of c03 printed no number of the new forms.; Forms: c01_build.py read against RULE.txt section 3 (position weights, expected clicks, shrinkage, centring in 8 exposure bins); exactly two forms are scored.; Entered value recomputed from the stored item statistic and exposure with my own binning and centring code: largest difference 1e-7, both forms, both witnesses.; g1 conditional test from the stored arrays (critic/a01_reread.py): val twin +1.386 / +2.172, holdout +1.220 / +1.853, equal to the lane's reads.; Independent Wald test, pool-only null, user-clustered: val twin +1.61 / +2.32, holdout +1.40 / +1.94.; g2 exposure-preserving placebo, own code, 300 draws (critic/a02_placebo300.py): placebo means -0.08 to +0.03, spreads 0.98 to 1.03; click part val twin +1.39 / +2.19, holdout +1.28 / +1.78 (bar 2.5).; g4 direct NDCG, organisers' metric copy, best in-sample weight: val twin -0.000002 / +0.000078 +- 0.000147; lag-view holdout +0.000284 +- 0.000163 / -0.000004.; Not re-run: own parity recomputation (g3), own leak scan (g5), nested NDCG. The lane's outputs for these were read only.

## Fillers

### 1. Fine-tune arms bs05 / bs2 / xm30 on the 8 all-data wide students of FD8P20: NO_GO

Check (i) fails for all three arms; the kit itself is sound. Batch arms: at fixed learning rate and epochs the batch sets the number of optimiser steps (345 / 175 / 91 per epoch), which is the dose of the same shift. The lane says so itself and dropped the learning-rate arms for exactly that reason (COVERAGE_MAP l.303; ft_redesign.md section 10 item 2: stop testing learning rate, epochs, EMA). The dose is already read flat at file level three ways: half learning rate 0.0000 +- 0.0003; first epoch against the saved model val -0.00017 +- 0.00034, private mix +0.00016 +- 0.00050; fine-tune weight 0.60 val -0.00011 +- 0.00016. xm30: its mechanism is a larger cold-item offset through the extra branch. That dose is closed on record: the best global cold offset on the train-only pool is worth -0.0003 (ft_redesign.md l.146, 'Cold offsets stays CLOSED'; COVERAGE_MAP l.281, l.355), and the offset part of the fine-tune alone reads val +0.00008 +- 0.00021, private mix +0.00025 +- 0.00041. The lane's stated reason cites int_ft.md verdicts 2 and 6, which do not make that attribution: verdict 6 says the first-epoch shift has no cold promotion yet and is level with the saved model at file level. Extra-branch multipliers 3 / 10 / 30 were also in the closed 11-12 retune. The rule leads with val, where the fine-tuned half reads +0.00003 against +0.00048 on public, so it looks where this mechanism does not show. The best arm needs about +0.00053 on val; the class bound's 95% upper limit is +0.0005, so a pass is as likely false as true. Other checks: (ii) PASS: my dry print from the current scripts is byte-identical to the lane's, and my token diff finds 25 of 25 commands equal to the stored control command plus exactly the intended flags. (iii) The rule text equals the brief's wording and the Holm thresholds are right; it has no sentence on the holdout (no holdout witness exists for these members) and no minimum effect. (iv) PASS: the reader reproduces the Q4 read to the digit on the finished arm mixft3; path-check self-tests behave (mixone3 PASS, mixft3 FAIL). (v) PARTIAL: the reader builds no file and a pass only buys a replication, but the replication has no decision rule and 8 of its 16 members are the screened ones. If the lead overrides as a stated exception: xm30 alone, after the four fixes listed in the summary.

### 2. CTXDROP / MODSUP: NO_GO

Neither flag exists. My search of every .py and .sh under nn, nn_x and src for retr_ctx_drop, modsup, ctxdrop, ctx_drop returns 0 files. Both need model code, which is out of scope for a filler.

### 3. P8 aux normalisation (AUXF, --aux_norm feed) under distillation: NO_GO

The fix ran in the cooldown harness on 11-12 and stopped: val -0.00010, holdout -0.00012 (CURRENT_STATUS l.559, COVERAGE_MAP l.145, w5_eval_B_prestale.txt). Val is negative too, so this is not a holdout-only rejection. The only untested form is the retry under distillation, a class closed by the stop rule met at 2026-11-15T21:27:39Z; no reason for an exception is given. The prepared kit is clean (my diff: the lane copy adds 6 lines to the stock lane, nothing else) but should not run.

### 4. Training batch size (cooldown half batch cbs05) under distillation: NO_GO

Closed on 11-13 in three forms (COVERAGE_MAP l.211, l.212; w5_eval_B_wave8.txt): half batch val +0.00006 +- 0.00040, double batch -0.0003, quarter batch -0.0015, 4x batch from scratch -0.0026. Val is null, so this is not a holdout-only rejection. A retry under distillation falls in the closed class with no new mechanism stated. The kit is clean but should not run.

### 5. Deferred list of TRICKS_SWEEP_1113 section 4.2 and the location-subspace row: NO_GO

Nothing to launch: every trigger is either not met or was met and then spent by a later run. I checked the location-subspace row myself: its CPU pre-gate ran as C03 and failed with val -0.00008 +- 0.00030 (COVERAGE_MAP l.260), so that is a val-side closure. The other one-line statuses were read, not re-derived.

