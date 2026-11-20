# RECHECK 11-14, skeptic "feasibility_value": do the flags exist, what does it really cost, is the probability calibrated

Written 2026-11-14 ~21:05 UTC. CPU only. Scripts and outputs: `nn_x/w12/recheck/feasibility_value/` (`s1_costs.py`, `s2_wext.py`, `s3_holdshort_free.py`, `s4_power.py`, `s5_holdout_pool.py`, `s6_w8p20_probe.py`, `out/`). No submission file built, nothing written to `preds/` or `subs/`, no training process touched.
Probability convention (as the lenses used it): lever / long bet / file recipe = P(true pool or file gain >= +0.0005); null result = P(the closed idea is still worth >= +0.0005); risk = P(the statement is true and material).

## 0. Bottom line

1. **GPU is not the scarce resource; resolution is.** Measured cost of a d192 10-epoch run is about 1 GPU-h (1.15 h alone; about 3 h wall with 3-4 per GPU). The cluster does about 250 such runs a day and 4,800 GPU-h remain to 12-09. All GPU arms proposed by the nine lenses together are about 135 GPU-h (3%). Four cost statements were 3-4x too high (§2).
2. **The declared screens throw away most real effects of the size we still hope for.** The 2-seed Qhat screen passes 15% of nulls and 36% of true +0.0005 shifts. The 4-seed asymptote rule passes 3% of nulls and 22% of true +0.0005 shifts (37% of a width-sized +0.0007). Screen plus 8-seed confirmation keeps 11%. A wave of 15 arms at a 4% prior has a 6.5% chance of one confirmed real +0.0005 (§3).
3. **Own-model arms are worth about +1 point of P(top 3) in total; the merge decision is worth about 5-8x that.** This is arithmetic from the base rate, the measured SEs and lb_stats' rank model, not a new measurement.
4. **W16 will read below W8 on val.** With 4 of the 8 extension seeds finished: W12 - W8 = -0.00024 ± 0.00029 val, -0.00005 ± 0.00017 holdout. The extension 4-pool (0.40055 val) is below every one of the 70 4-subsets of W8 (min 0.40065), while on the holdout it sits at the subset mean (0.32410 vs 0.32414). kd_reprice's forecast (not above W8) is the right one; the +0.00015 val forecasts of lb_stats, pool_composition and kd_audit will miss (§4).
5. **The W8P20 public probe cannot do what its rule asks.** Measured with all four fine-tunes finished: final val +0.00049 ± 0.00031 vs W8 (M>=50 +0.00080 ± 0.00035, Tuesday +0.00100 ± 0.00045, Thursday -0.00008 ± 0.00043). The paired public SE is 0.00041, not 0.0002-0.0003, so the ±0.0003 thresholds are 0.74 SE: at a true delta of 0 it is adopted with P 0.23 and closed with P 0.23 (§5).
6. **HOLD-SHORT exists for free.** Restricting the stored holdout predictions to random sub-feeds gives a short-feed witness with SE 0.0003 and per-feed correlation 0.42-0.52 to the full holdout, in 3 CPU-minutes, with no stale-input artefact (§6). The inference version is not worth its code.
7. Of 106 findings: 70 keep (mostly nulls, risks and read-out rules), 23 modify, 13 drop. No finding names a flag or file that does not exist, except where the author already says "needs a patch".

## 1. Feasibility of flags and files (checked in code, not run)

| claim | check | result |
|---|---|---|
| `--init` + `--kd_file` work together (kd_audit#7, kd_literature#1) | train.py 319-354, 422-427 | yes; architecture comes from the checkpoint, the KD block is independent of `--init` |
| @77500 snapshots | `nn_x/w5/ckpt` | d128: kd05 x4, kd08 x4, kd10 x4, base x6. **d192: kd05w192 x8 and kd05t25w192 x8 (seeds 951-958)**, kw_drop0 x2, kd08w192 x1 |
| `--aug_all --aug_hist` without `--aug_cand` (kd_reprice#9) | cd_arms.py 58-59, train.py 375-379 | works for arch tf; needs `--cooldown_mix > 0` (E_BASE has it) |
| `--aux_pos 0` under HYG `--cliplog 50` (queued kwaux0) | cd_arms5.py 240-246 | no crash: the aux part is `loss - main` = zero-gradient tensor, logged as 0 |
| `--hzcd` with `--kd_file` (kd_reprice#8) | cd_arms6.py 84-108 | works: the hz view delegates gather and keeps `b['ci']`; 9 hz feature files present |
| lane arms | lane.sh 122-137 | present: u20a/b/c, kwh30, kwh10, kwaux0, kwaux1, kwwd05, kwwd10, kwm05, kwcf35, kd08w256, kw256d2, kwr32, kwh6, kwsb2. Missing (one line each): kwm10, kwlr14, kd065w192, kd065w256, kd08w320, kd10w192, d192 `--drop 0.2`, `--retr_uniq --retr_tcnt`, `--drop 0 --drop_cd 0.1` |
| `--holdout 0 --keep last` (long_bets#7) | train.py 284-306, cd_arms5.py 346 | path never executed; `save_holdout` asserts holdout 30000, HYG must be changed |
| two-sided temperature (kd_audit#6), sub-feed ctx recompute (kd_reprice#6), M>=100-only augmentation (winner_model#2), logit offset (kd_reprice#12) | train.py 489-503 | none exists; each needs a copy of a file that running jobs use |
| HS1 students (kd_reprice#11) | `nn_x/w9/hs1/train.py` | no `--kd_file`; port needed |
| soft-target FT patch, TO-only probe file, pop1kd fine-tunes | `ft_redesign/finetune_kd.py`, `subs/cand_W8TOkd_r3.parquet`, `nn_x/abl/preds/*pop1kd05t25*_n2cpv3s3_*` | all present |

## 2. Real GPU cost (measured, `out/s1_costs.txt`)

From `RUNNING_w5.txt` and the run logs, today's finished 10-epoch runs. GPU-h per run is an upper bound (other lanes on the same GPU are not counted).

| run type | wall alone | wall with 3-4 per GPU | GPU-h per run |
|---|---|---|---|
| d192, 10 epochs | 1.15 h (18 runs) | 2.9-3.3 h (21 runs) | 0.8-1.15 |
| d128, 10 epochs | - | 1.9-3.2 h | 0.8-1.0 |
| d192, 20 epochs | - | 4.2-6.2 h | 2.3 |
| d256, 10 epochs | - | 2.2 h | 1.2 |
| re-cool from @77500 (19,670 steps) | about 15 min | about 40 min | 0.2-0.25 (inferred from the step share) |

| finding | stated cost | real cost |
|---|---|---|
| winner_model#4 (6 arms x 2 seeds) | 40 GPU-h | about 12 |
| winner_model#2 (2 x 20 epochs) | 13 GPU-h | about 4.6 |
| kd_audit#6 (2 d192 runs) | 6.5 GPU-h | about 2 |
| pool_composition#10 (4 d192 runs, already running) | 13 GPU-h | about 4 |
| kd_reprice arms (1.6 per 2-seed arm), kd_audit#7 (6), kd_literature#1 (9), long_bets#5/#6 (15-20 each) | - | about right |

## 3. Calibration: base rate and what the rules can see (`out/s4_power.txt`)

Priors from the record, Beta(0.5, 19.5) (the 5-of-200 project rate) updated with today's arms: a lever closed under hard labels and re-tested under KD 0.054 (1 of 8: width); a KD target or teacher variant 0.017 (0 of 9); all KD arms 0.064 (2 of 19).

P(pass) under a normal model with the measured SEs (per-feed SE of the asymptote delta 0.00037 holdout / 0.00065 val at 4 seeds, seed sd of a 2-seed Qhat 0.0003 / 0.0006, 8-vs-8 pool delta 0.00028 / 0.00049):

| rule | true 0 | +0.0003 | +0.0005 | +0.0007 | +0.0010 |
|---|---|---|---|---|---|
| 2-seed Qhat screen (hold >= +0.0004, val >= 0) | 0.15 | 0.26 | 0.36 | 0.46 | 0.61 |
| 4-seed asymptote rule (hold >= +2 SE, val >= 0) | 0.03 | 0.11 | 0.22 | 0.37 | 0.62 |
| 8-seed pool confirmation (hold >= +0.0004, val >= 0) | 0.06 | 0.27 | 0.51 | 0.74 | 0.93 |
| 8-vs-8 pools, adopted pooled-z rule (both >= 0, z >= 2) | 0.04 | 0.23 | 0.48 | 0.73 | 0.94 |
| the same with a 3-rotation holdout (§7.2) | 0.06 | 0.38 | 0.68 | 0.87 | 0.96 |

Consequences. A 2-seed screen has a likelihood ratio of 2.4: it is not a filter. Screen (4 seeds) then confirmation (8 seeds) keeps 0.22 x 0.51 = 11% of true +0.0005 effects. One stage at 8 seeds with the pooled-z rule keeps 48% at a 4% false-pass rate and costs 8 GPU-h per arm. Since GPU is free at this scale, every arm worth running should start at 8 seeds. Even so, 15 arms at a 4% prior give P(at least one real +0.0005 found) about 0.25, with about 0.6 false passes next to it; a passing arm is real with P about 1/3 and a public read (SE 0.0004) cannot separate the two.

## 4. W8 extension seeds, read now (`out/s2_wext.txt`; measured)

Finished: kd05w192 s955, s957; kd05t25w192 s955, s957.

| set | W8 | W12 | W12 - W8 | extension 4-pool | W8 4-subsets mean (sd; min..max) |
|---|---|---|---|---|---|
| val | 0.40157 | 0.40132 | -0.00024 ± 0.00029 | 0.40055 | 0.40114 (0.00022; 0.40065..0.40162) |
| holdout | 0.32456 | 0.32450 | -0.00005 ± 0.00017 | 0.32410 | 0.32414 (0.00016; 0.32379..0.32455) |
| holdout @last | 0.32447 | 0.32442 | -0.00005 ± 0.00017 | 0.32394 | 0.32411 (0.00017; 0.32371..0.32442) |

Mean member is equal (val 0.39940 vs 0.39935). Inferred: W8's val level carries about +0.0003 of val-specific seed luck that the holdout does not share; W16 will land near 0.4012-0.4014 val and 0.3245 holdout. Under the adopted evidence rule (val delta >= 0 and holdout delta >= 0) W16 can never replace W8, so its status must be fixed before it is read: a variance-reduction pick by construction, or no slot.

## 5. What the W8P20 probe can resolve (`out/s6_w8p20_probe.txt`; measured, all four pop1kd fine-tunes present)

| weight | final val delta vs W8 | M>=50 | M<50 | test corr to W8 | SE public delta | SE private delta |
|---|---|---|---|---|---|---|
| 0.15 | +0.00039 ± 0.00027 | +0.00072 ± 0.00031 | -0.00010 ± 0.00048 | 0.99967 | 0.00035 | 0.00023 |
| **0.20 (declared)** | +0.00049 ± 0.00031 | +0.00080 ± 0.00035 | +0.00003 ± 0.00058 | 0.99942 | 0.00041 | 0.00027 |
| 0.30 | +0.00012 ± 0.00040 | +0.00086 ± 0.00045 | -0.00094 ± 0.00073 | 0.99868 | 0.00052 | 0.00034 |

Weight 0.2 by weekday: Tuesday +0.00100 ± 0.00045, Thursday -0.00008 ± 0.00043. The build rule (val >= 0) passes. With the declared thresholds at weight 0.2: true 0 gives adopt 0.23 / close 0.23; true +0.0003 gives adopt 0.50 / close 0.07; true +0.0005 gives adopt 0.69. Val and public are independent samples, so the usable rule is their inverse-variance mean: with val at +0.00049 ± 0.00031 the public reading must be >= W8 + 0.0005 for a combined 2 SE. The in-period holdout reads this family at -0.0004..-0.0006 and its penalty sits on weekdays (lb_stats#11), private is mostly weekdays, and private is also the coldest part of test (cold-share >= 0.45 feeds: 8.4% of val, 21.5% of public, 32.7% of private, lb_stats#11). The evidence is two-sided: this is a pick-2 candidate, not a pick-1 replacement.

## 6. HOLD-SHORT without inference (`out/s3_holdshort_free.txt`; measured)

Stored full-feed holdout predictions restricted to a random sub-feed (size from the val M<50 distribution, one positive forced, 4 replicates), z-scored inside the sub-feed, pooled. Level check: W8 0.5251 vs 0.5183 on real val M<50 feeds (positives per feed 1.06 vs 1.22).

| contrast | holdout, full | HOLD-SHORT (free) | val M<50 | val M>=50 | per-feed corr full vs short |
|---|---|---|---|---|---|
| KD at d128 (4 vs 4) | +0.00021 ± 0.00036 | +0.00072 ± 0.00034 | +0.00309 ± 0.00117 | +0.00121 ± 0.00063 | 0.51 |
| width under KD (8 vs 8) | +0.00083 ± 0.00028 | +0.00073 ± 0.00027 | -0.00011 ± 0.00089 | +0.00071 ± 0.00055 | 0.47 |
| W8 vs hard-label 4-pool | +0.00128 ± 0.00037 | +0.00164 ± 0.00035 | +0.00292 ± 0.00122 | +0.00254 ± 0.00067 | 0.52 |
| S16 vs W8 | -0.00028 ± 0.00019 | +0.00002 ± 0.00018 | +0.00056 ± 0.00057 | -0.00013 ± 0.00039 | 0.42 |

It moves toward the val M<50 reading in three of four contrasts (the KD gain is larger on short lists; the narrow students stop losing) and is half independent of the full holdout. It omits only the set-dependence of the scores, which kd_audit#10 and kd_literature#2 measured at KL 0.002-0.005 nats. A re-scored version with stale context inputs would favour students trained on stale sub-feeds (kwsub), and an honest one needs the 1-day context-recompute helper of kd_reprice#6. Verdict: add this column to the read-out; do not build the inference version.

## 7. Cross-cutting

### 7.1 Contradictions between lenses
- **Does a kwsub null close function matching?** kd_literature#2 and long_bets#1: yes. kd_reprice#2, winner_model#1 and the lead's addendum: no. The three function-space measurements (sub-feed vs restricted logits corr 0.9978; student-teacher match 0.9933 on seen and unseen feeds; XPOOL null) all say there is little for a cleaner arm to gain. I side with closing after the by-M read; the repaired arm (kd_reprice#6) is not worth a day of code.
- **Is the holdout the informative witness?** kd_audit#11: yes, it "agreed with public" on W8 > S16. lb_stats#3: that public gap is 0.95 SE. lb_stats#2: the holdout alone is the worst predictor for the newer files. Use the pooled val + holdout rule that is already adopted.
- **W16 val forecast:** +0.00015 (lb_stats#4/#6, pool_composition#1, kd_audit#9) vs not above W8 (kd_reprice#3). Measured W12: -0.00024 ± 0.00029.
- **TO-only probe:** decision value (ft_redesign#8, forensics_skeptic#10, lead item D) vs none (lb_stats#9). Both outcomes leave files within 0.0002; the only decision it changes is whether to run 10-minute fine-tunes.
- **Pick 2:** S16 (lb_stats#1) vs C3 (pool_composition#11). The difference is below 0.0001.
- **Soft-target fine-tune:** priced 0.05-0.07 by three lenses, measured null at file level by two independent CPU pilots (ft_redesign#2, kd_audit#5).

### 7.2 Not proposed by any lens
1. **A KD re-cool harness at d192.** 16 stored snapshots (kd05w192 and kd05t25w192 s951-958 @77500) plus the B_BASE flags and `--kd_file` give a d192 student in 0.2 GPU-h and 40 minutes. Every cooldown-phase lever can then be read at 8-16 seeds against control re-cools: cooldown kd_w (kd_audit#7 at the width we ship), cooldown mix 0.05 / 0.10, `--drop_cd`, aux anneal, kd_T, pool-only sub-feed augmentation. Late dropout is two re-cools of the stored kw_drop0@77500 snapshots with `--drop_cd 0.1`. First run: 16 control re-cools vs the 16 originals, to show the harness reproduces the from-scratch cooldown.
2. **A 3-rotation holdout for confirmations.** 89,794 train feeds are holdout-eligible (p_random > 0.9, M >= 50; `out/s5_holdout_pool.txt`), so three disjoint 30K holdouts fit; rotation 0 is the current one, so existing runs stay valid. Per-feed SE falls by 0.58 and the power of the pooled-z rule for a true +0.0005 rises from 0.48 to 0.68. Cost: a 3-line holdout-rotation flag, per-rotation label / meta files, and 3x the runs of the confirmed arm and of the reference. Use it only at the confirmation step.
3. **One stage at 8 seeds, then one bundle arm.** Drop the 2-seed screens. After the wave, train one arm with every knob whose point estimate is non-negative on both witnesses, 8-16 seeds, against the W16 members. Effects below the detection limit can only be seen summed.
4. **Pick 2 as a structural pick aligned with the known shift.** Private is colder than public, which is colder than val. The popularity-light admixture is the only lever aligned with that direction; the right use is pick 2, with W8 or W16 as pick 1.
5. **Order of operations.** Resolve lb_stats#12 (manual or automatic final selection) before W16, W8P20 or W8TO are submitted. Under automatic best-two-by-public every near-copy probe is a lottery ticket that can displace W8.
6. **Spare GPU should buy power, not arms**: more seeds per arm and rotations, in line with the standing instruction to keep GPUs busy.

### 7.3 Expected value of the whole programme (inferred)
About 15 own-model arms at priors 0.02-0.05, read at 8 seeds: P(a confirmed real +0.0005) about 0.25 at best design, about 0.07 with the declared two-stage screens. lb_stats#7 prices +0.0005 at +0.10..0.15 of P(top 3): about +1 to +3 points. The pop1 admixture as pick 2: P(true gain > 0) about 0.6, P(>= +0.0005) about 0.07, about +1 point. A merge (long_bets#3): P(>= +0.001) about 0.25 given a partner in ranks 2-7, about +6 to +9 points, at 0 GPU-h.

## 8. Five findings with the best value per GPU-hour

| rank | finding | cost | why |
|---|---|---|---|
| 1 | lb_stats#12: confirm how the two finals are taken; fix the stale pick text | 0 GPU-h, 10 owner-minutes | removes a tail risk larger than any modelling delta and gates every probe submission |
| 2 | long_bets#3: team merge with a non-neural partner | 0 GPU-h, 1 slot | the only item with P(>= +0.001) above 0.2 |
| 3 | kd_reprice#5 (= pool_composition#10, built as W8P20) | 0 GPU-h more, 1 slot | the only measured structural diversity; val +0.00049 ± 0.00031; read with the pooled rule of §5, as pick 2 |
| 4 | kd_reprice#1 + pool_composition#2: pool / asymptote read-out, amended to 8 seeds in one stage | 0 GPU-h for the rule | raises the share of real +0.0005 effects that survive from 11% to 48% |
| 5 | winner_model#4, weight-decay pair first (kwwd05 queued, kwwd10 in the lane) | about 8 GPU-h for two 4-seed arms | never varied in 285 runs and the direct counter to the measured width memorisation |

Next: kd_audit#7 run at d192 as the first use of the re-cool harness (3-4 GPU-h), kd_literature#3 reduced to d192 `--drop 0.2` (4-8 GPU-h), long_bets#5 reduced to kd08w192 seeds 952-954 (3 GPU-h; seed 951 has the best single-member val on record, 0.40021).

## 9. Verdicts

Adjusted probability in brackets. "Keep" for a null means the closure stands.

**kd_audit.** #1 keep (0.95), reproduced here (W8 0.40157 / 0.32456). #2 keep (0.5), free read-out split. #3 keep (0.9), free; the read-out tool already prints @last. #4 modify (0.15): gradient shares are measured, but Q is flat in kd_w 0.3-0.8, so little is left to explain; queued kwaux0 settles it, no further arm. #5 keep (0.02). #6 modify (0.02): the code fact is right; the two-sided arm needs a train.py copy for a flat, calibrated teacher; only if kwT07 passes at pool level; cost 2 GPU-h, not 6.5. #7 modify (0.03): feasible; run it at d192 from the stored kd05w192 snapshots instead of d128. #8 keep (0.02). #9 keep (0.02); val band too high, see §4. #10 keep (0.02). #11 modify (0.5): SEs right, "holdout agreed with public" is a 0.95-SE reading. #12 keep (0.9).

**kd_reprice.** #1 keep (0.9). #2 modify (0.25): staleness is a fact; that it hides a gain is contradicted in function space; free by-M read only. #3 keep (0.02); its W16 forecast is the one §4 supports. #4 keep (0.03). #5 modify (0.07): adopted as W8P20; no second weight; pick 2; rule of §5. #6 drop (0.03): one day of code on the training path, XPOOL null, no matching gap. #7 drop (0.015): two CPU pilots null at file level. #8 modify (0.02): feasible, same popularity axis as pop1, unverifiable offline; only if W8P20 is adopted. #9 modify (0.02): covered by queued kwh30; history-derived dense inputs stay stale, so the augmentation is weak. #10 keep (0.03): kwaux0 queued; kwaux1 only if aux 0 is not worse. #11 modify (0.05): flags exist; move late dropout and cooldown levers to the re-cool harness; drop the HS1 port and the retrieval pair. #12 drop (0.015): the binding error is estimation noise from finite labels (long_bets#2), which a residual pass on the same labels cannot remove; it also needs offsets from 15 fold models at inference and in the fine-tune.

**kd_literature.** #1 modify (0.03): duplicate of kd_audit#7; a 36-run grid with Holm over 6 cells has no power; run the two-cell falsification at d192. #2 modify (0.03): write the rule; the 15-epoch follow-up is not worth a slot. #3 modify (0.04): dropout above 0.1 was never run at 10 epochs; start with d192 `--drop 0.2`, 4-8 seeds; d256 only if kd05w224 is not below kd05w192. #4 drop (0.015). #5 keep (0.02). #6 keep (0.02). #7 keep (0.02). #8 keep (0.85). #9 drop (0.01): a cost lever when cost is not binding and the pool is saturated in N. #10 keep (0.01). #11 keep (0.02). #12 drop (0.015): U20 decides it.

**lb_stats.** #1 keep (0.9). #2 keep (0.6): 14 scores on one public draw, split chosen after the fact; the consequence (pooled rule) is sound. #3 keep (0.02). #4 modify (0.8): right about public; its +0.00015 val forecast fails (§4); fix W16's status before reading. #5 modify (0.03): free form done (§6); skip the inference form. #6 keep (0.02). #7 keep (0.85). #8 keep (0.9). #9 keep (0.01). #10 keep (0). #11 keep (0.05); its weekday observation bears on W8P20. #12 keep (0.5), top value.

**ft_redesign.** #1 keep (0.02). #2 keep (0.01). #3 keep (0.01): valid; saves wall-clock only (fine-tunes cost about 80 GPU-min). #4 keep (0.85). #5 keep (0.02). #6 keep (0.95). #7 keep (0.02). #8 modify (0.02): file exists; near-zero decision value; last among probes and only after lb_stats#12. #9 drop (0.01). #10 drop (0.01). #11 keep (0.01). #12 keep (0.03).

**pool_composition.** #1 keep (0.02). #2 keep (0.05), with the power caveat of §3. #3 keep (0.02). #4 keep (0.03). #5 keep (0.02). #6 keep (0.02). #7 keep (0.85). #8 keep (0.8). #9 keep (0.02). #10 modify (0.05): same file as kd_reprice#5; the wide pop1 arm is already running (about 4 GPU-h). #11 modify (0.5): status already updated; the better pick 2 is the pop-light admixture if it holds on val and public. #12 modify (0.03): same as lb_stats#5.

**winner_model.** #1 keep (0.6): the handicaps are measured; the consequence is the free by-M read, not a new arm. #2 drop (0.02): new flag in files running jobs use, kw_e20 negative, no matching gap. #3 keep (0.01). #4 modify (0.06): weight decay first; cooldown mix in the re-cool harness; drop aux 0.25; 12 GPU-h, not 40. #5 keep (0.03). #6 modify (0.6): the M >= 200 production set includes truncated lists; run the M = 300 check first; matters only for sub-feed arms. #7 keep (0.01). #8 keep (0.02); skip the optional `--loss_norm idcg` pair. #9 keep (0.02). #10 keep (0.03).

**forensics_skeptic.** #1 drop (0.015). #2-#5 keep (0.01). #6 keep (0.9). #7 keep (0.01). #8 keep (0.8). #9 drop (0.02): found after about ten contrasts, no second witness. #10 drop (0.02): duplicate of ft_redesign#8; its rule is under 1 SE. #11 modify (0.01): contingent on a kwsub pass; winner_model's c02 / c03 on the kwsub checkpoint answers it in 8 CPU-min. #12 keep (0.8).

**long_bets.** #1 keep (0.02). #2 keep (0.85). #3 keep (0.25): the law is extrapolated from partners built on our own pipeline; owner's decision, deadline Dec 1. #4 keep (0.02). #5 modify (0.03): kd08w192 seeds 952-954 only; d256 / d320 on a pass. #6 modify (0.04): merge with winner_model#4 into one pre-declared wave. #7 drop (0.01): +0.0001 expected, unmeasurable, never-executed path in the freeze week, and it removes the second witness for the final members. #8 keep (0.02): queued; closes a family that the external report priced at 0.10. #9 keep (0.02); skip the co-click NN screen. #10 keep (0.01). #11 keep (0.02). #12 keep (0.5).

## 10. Limits

- The power table is a normal model on quoted SEs with val and holdout taken as independent; it ignores the @last clause and correlation between seeds.
- GPU-h per run are upper bounds; re-cool cost is inferred from the step share, not from a d192 re-cool log.
- §4 uses 4 of 8 extension seeds. §5 is exact for the files on disk at 21:02 UTC. §6 is not a re-scored sub-feed and its sub-feeds come from long-feed users.
- Priors are Beta updates on arm counts that I grouped by hand; the adjusted probabilities are judgements anchored on them.
- I did not read all nine lens reports in full; verdicts on nulls rest on the finding text and on spot checks.
