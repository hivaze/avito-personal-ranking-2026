# RECHECK 11-14, skeptic "duplicates": prior evidence and duplicate check of the 106 findings

Written 2026-11-14 21:10 UTC. Lens: is each finding already closed, running, or bounded by an existing measurement? Sources: `reports/COVERAGE_MAP.md` (read in full, 311 rows, including rows 296-311 that the lead added today from these same reports), `nn_x/w5/logs/E_kd_predeclared.txt` (in full, to 20:21Z), `reports/TRICKS_SWEEP_1113.md` and `reports/EXTERNAL_SCAN_1113.md` (searched for every idea named in a finding, not read end to end), `nn_x/w5/lane.sh`, `reports/CURRENT_STATUS.md` sections 1-5 and the log from 18:24Z, `docs/04_SUBMISSIONS.md`, and the sections of the nine lens reports that a verdict depended on. CPU only; scripts and outputs in `nn_x/w12/recheck/duplicates/`.

Verdict words. For levers, recipes and long bets: **keep** = untested and not running; **modify** = only a narrower version is new, or it must wait for a running arm; **drop** = closed under the same regime, already running / queued / declared, or bounded by a measurement. For null results and risks: **keep** = a new measurement, the reference copy of its cluster; **modify** = partly covered (earlier closure on another base, or overlap with another finding); **drop** = a strict duplicate or a restatement that changes no decision.

## 0. Summary

- **38 keep, 35 modify, 33 drop.** The 106 findings are about 45 distinct statements; 16 clusters are the same measurement reported by 2 to 6 lenses (section 2). The repetition is useful as replication (independent code, same numbers), but it should not be counted as 106 pieces of evidence.
- **Nothing proposed is a silent re-run of a CLOSED row.** Four proposals rest on a premise that an existing row contradicts (kd_reprice#6, kd_literature#9, winner_model#8's optional arm, kd_reprice#8's judge change). Eleven proposals are already queued, running or pre-declared by the lead (aux 0, weight decay, history drop, U20, W8P20, wide pop1 students, W8 train-only probe, W16).
- **After removing duplicates, running arms and bounded items, six GPU items are untested** (section 5), each at P 0.02-0.03 of a pool-level gain. Their sum is consistent with the status line "own-model levers left: about +0.0005 public".
- **Four measurements of my own** (section 1): (a) the 12-wide pool reads below W8 on val and equal on the holdout, a second out-of-sample check of saturation; (b) the pre-declared W8P20 public rule has thresholds of 0.74 SE; (c) the residual gate that closed about 300 feature rows recovers 10% (linear) to 26% (monotone GBDT) of a real, load-bearing input block on the holdout and nothing on val; (d) sub-sampled long feeds do not reproduce the label format of short feeds, while 56,015 natural short exploration-like train feeds do and already have out-of-fold predictions.

## 1. New measurements by this skeptic

All paired per feed unless stated; ± is one standard error.

### 1.1 Twelve wide students against W8 (seeds 955 and 957 of both wide arms finished at 20:52Z)

`pool_composition/pc_pool_vs_w8.py` with the four finished extension members:

| pool | TO val | TO holdout | final val (FT of W8 + TO of pool) |
|---|---|---|---|
| W8 | 0.40157 | 0.32456 | 0.40165 |
| W8 + 4 (12 wide) | 0.40132, **-0.00024 ± 0.00029** | 0.32450, **-0.00005 ± 0.00017** (@last -0.00005 ± 0.00017) | 0.40185, +0.00019 ± 0.00022 |

Measured. The saturation claim holds out of sample a second time (pool_composition reported 11 wide: -0.00030 ± 0.00027 / -0.00007 ± 0.00016). The fits made on the 8 W8 members alone (lb_stats#6: W16 TO val 0.40161-0.40175; pool_composition#1: +0.00016) are optimistic on val; kd_reprice#3's reading (W8 is a favourable val draw of an average 8-pool at 0.40133; W16 about 0.4014 ± 0.0002) is the one the data support. On the holdout all forecasts agree with the read. The four new members have val 0.39866-0.39977 and holdout 0.32234-0.32258: ordinary members.

### 1.2 Paired public SE of the pre-declared W8P20 probe (`d01_w8p20_se.py`)

The 20:16Z block (A) declares: public >= W8 + 0.0003 adopts the pop1 view for student pools, <= W8 - 0.0003 closes it, and quotes a paired SE of "about 0.0002-0.0003 at test corr ~0.998". Measured on val, per branch 0.8 z(W8) + 0.2 z(pop1kd), FT branch from the 3 fine-tuned pop1kd members that exist (s951-953), TO branch from all 4:

| weight, branches | val delta vs W8 final | M>=50 | M<50 | SE of the public delta (8,562 feeds) | 0.0003 in SE | test corr to W8 |
|---|---|---|---|---|---|---|
| 0.15, both | +0.00053 ± 0.00026 | +0.00075 ± 0.00030 | +0.00021 ± 0.00048 | 0.00034 | 0.88 | 0.9997 |
| **0.20, both (declared)** | **+0.00029 ± 0.00031** | +0.00058 ± 0.00035 | -0.00012 ± 0.00058 | **0.00041** | **0.74** | 0.9994 |
| 0.30, both | +0.00011 ± 0.00040 | +0.00080 ± 0.00044 | -0.00089 ± 0.00072 | 0.00051 | 0.58 | 0.9987 |
| 0.20, TO branch only | +0.00024 ± 0.00024 | +0.00068 ± 0.00028 | -0.00038 ± 0.00042 | 0.00031 | 0.97 | 0.9998 |

Measured: SE and val deltas. Inferred: with thresholds at 0.74 SE, a file with zero true effect is "adopted" with probability 0.23 and "closed" with probability 0.23; a file with a true +0.0003 is adopted with probability 0.50. The likelihood ratio of an "adopt" reading is about 2. The val delta falls as the weight rises (0.15, 0.20, 0.30), which is the pattern of noise plus a cost, not of a view that adds. The "public confirmed pop1 pools 3 of 3" premise is one reading, not three: Q2a's 4 pop1 members are a subset of C1's 8, C3 is C1 with 3-seed fine-tunes, and all three are scored against G3 on the same 8,562 feeds (docs/04 section 3: +0.000385 at paired SE 0.00040; +0.000574; +0.000634).

### 1.3 GATE-CAL: the residual gate against real positive controls (`d02_gate_cal.py`)

About 300 feature rows of COVERAGE_MAP were closed by "nested residual on top of the pool" (270 linear reads, CCB-1113, DATA_PIPELINE, AUTOFE_1114, forensics_skeptic). `PRETRAIN_PLAN_1113.md` l.312 says this lens is "not validated. pre2 fails it (-0.00018) yet is worth about +0.001 in the NN". Nobody calibrated it. Here the stored 10-epoch members trained WITHOUT an input block (`abl_e10_xdrop_pop`, `abl_e10_xdrop_pre2`) are the base, their with-input twins (`w5_e10_base`, same seeds) give the true pool-level value of the block as an NN input, and the removed columns are offered to the gate.

| set | block | true NN-input value (pool) | linear nested, best single column (post hoc pick) | monotone LightGBM residual (forensics hx.py) | recovered share |
|---|---|---|---|---|---|
| holdout, 30,000 feeds | 8 popularity / co-visitation inputs, 4 seeds | **+0.00852 ± 0.00066** | +0.00086 ± 0.00029 (seq_cv_invd20) | **+0.00222 ± 0.00033** (control 0.00000) | 0.10 linear; 0.26 GBDT |
| val | same block | +0.00160 ± 0.00102 | +0.00011 ± 0.00045 | -0.00058 ± 0.00037 | 0.07 / negative |
| val | pre2 scalars (3 of 5 columns stored), 2 seeds | +0.00061 ± 0.00083 | +0.00039 ± 0.00043 | -0.00030 ± 0.00020 | not resolved (true value 0.7 SE) |

Measured. Reading: a block worth +0.0085 as a trained input shows as +0.0009 in the linear lens and +0.0022 in the monotone GBDT lens with 30,000 in-period feeds, and as nothing on val (where its true value is itself only 1.6 SE). So a new input worth +0.001 to the ranker would read +0.0001 (linear) to +0.0003 (GBDT) on the holdout: invisible at the gates' SE of 0.0002-0.0004. The planted-signal controls of forensics_skeptic#6 and AUTOFE measure a different thing (a signal that 7,000 feeds can teach a residual model). Inferred consequence: the feature rows closed by residual gates are closed "in residual form"; their value as trained inputs is bounded at about +0.0025 by the best gate (GBDT on the holdout, 2 SE / 0.26) and at about +0.006 by the linear lens; the val gates, which closed most rows, show no recovery at all. This does not reopen them by itself: the five direct NN tests of added inputs are all null or negative (xrel_more -0.0019 val, pop2, TIME, kv64, retr_cat), and GBDT proxies fail in the other direction (TRUNC). It does change what a future feature test must be (section 5).

### 1.4 Short-feed witness: sub-sampled long feeds against natural short feeds (`d03_short_witness.py`)

lb_stats#5 and pool_composition#12 propose a short-feed holdout made by sub-sampling the 30,000 holdout feeds; pool_composition#12 asks for a premise check first. Scorer everywhere: a 3-member hard-label fold pool. NAT = 56,015 train feeds with XPOOL posterior p_random > 0.9 and M < 50, scored by `teacher_oof.npy` (out-of-fold by user); SYN = holdout feeds sub-sampled uniformly to NAT's size distribution, at least one positive; VAL = val feeds with M < 50 scored by the xf0 fold pool.

| M | NAT n | NAT positives / feed | NAT NDCG | SYN positives / feed | SYN NDCG | SYN - NAT | VAL positives / feed | VAL NDCG |
|---|---|---|---|---|---|---|---|---|
| 5-9 | 6,319 | 1.063 | 0.6912 ± 0.0032 | 1.031 | 0.6909 ± 0.0089 | -0.0003 ± 0.0094 | 1.078 | 0.6844 ± 0.0078 |
| 10-19 | 9,628 | 1.140 | 0.5662 ± 0.0027 | 1.062 | 0.5619 ± 0.0050 | -0.0042 ± 0.0057 | 1.157 | 0.5520 ± 0.0066 |
| 20-34 | 20,203 | 1.232 | 0.4729 ± 0.0018 | 1.104 | 0.4655 ± 0.0024 | -0.0074 ± 0.0030 | 1.250 | 0.4578 ± 0.0053 |
| 35-49 | 19,865 | 1.326 | 0.4186 ± 0.0017 | 1.152 | 0.4100 ± 0.0019 | -0.0087 ± 0.0025 | 1.359 | 0.4182 ± 0.0064 |

Measured. Natural short exploration-like train feeds match val short feeds in positives per feed in every size bucket (1.06 / 1.14 / 1.23 / 1.33 against 1.08 / 1.16 / 1.25 / 1.36). Sub-sampled long feeds do not (1.03 / 1.06 / 1.10 / 1.15) and are 0.007-0.009 harder at M 20-49 (about 3 SE). A short feed is not a random subset of an observed long feed: it carries about 2.8 times the click density. The same holds for the sub-feeds that kwsub trains on. Inferred: HOLD-SHORT by sub-sampling would measure a format that test does not contain; a natural short-feed set exists and is the better instrument (section 5, item N2).

## 2. Duplicate clusters

"Ref" is the copy to keep; the others carry the same measurement.

| # | statement | ref | duplicates | earlier row |
|---|---|---|---|---|
| D1 | wide-student pool is at its seed asymptote; W16 worth +0.0001 | pool_composition#1 | kd_audit#9, kd_reprice#3 (part), lb_stats#6, long_bets#4 (part) | l.301 "More same-recipe student seeds (W16, W24, W32) ... CLOSED 11-14 for LEVEL" |
| D2 | teacher identity gives no student diversity; one mean function | kd_audit#8 | kd_literature#6, pool_composition#4, long_bets#4, kd_reprice#3 (part) | l.299 "one common mean function (rho_sys 0.999+); students equal for teachers 0.3210-0.3245 holdout" |
| D3 | FT branch adds 0 ± 0.0004 at pool level (W8 FT 0.40087, TO 0.40157, blend - TO +0.00009 ± 0.00043: the same number in six reports) | ft_redesign#1 | pool_composition#8, winner_model#9, kd_reprice#7 (evidence), kd_literature#4 (evidence), forensics_skeptic#1 (evidence), long_bets#11 | l.303 "FT half 0 ± 0.0004 at pool level in every pool" |
| D4 | soft-target fine-tune on val | ft_redesign#2 (pilot, null) | kd_audit#5 (second pilot, null); proposed as a lever by kd_reprice#7, kd_literature#4, forensics_skeptic#1; retracted by long_bets#11 | l.304 "CPU pilot: gain = shrinkage (half-lr control equal)" |
| D5 | w_ft stays 0.5 for students | ft_redesign#1 | ft_redesign#7, lb_stats#9 (part), pool_composition#8 (part), kd_reprice#7 (part) | l.100 "w_ft != 0.5 \| CLOSED \| nested +0.0002 (z 0.96)" |
| D6 | member-level screen is blind; read arms on the 2-seed asymptote | kd_reprice#1 | pool_composition#2, kd_literature#8, kd_audit#11, long_bets#12 | l.115 protocol; E_kd 20:21Z (3) adopted it |
| D7 | kwsub read-out is confounded | kd_reprice#2 + winner_model#1 (different confounds) | kd_literature#2, kd_audit#10 (teacher side only) | l.309; E_kd 20:21Z (3) "a kwsub FAIL does not close sub-feed distillation" |
| D8 | full-feed teacher logits are valid sub-feed targets | kd_audit#10 | kd_literature#2, winner_model#1 (Spearman 0.986-0.995), long_bets#1 (0.9949) | none |
| D9 | phase-split KD by re-cooling stored plateau snapshots | kd_audit#7 | kd_literature#1 | l.240 "KD / test pseudo-labels from G3 in the cooldown \| RE-OPEN, P 0.02 each" |
| D10 | pop1-view students as a low-weight admixture | (declared: W8P20) | kd_reprice#5, pool_composition#10 | l.302; E_kd 20:16Z (A), (B) |
| D11 | K9 aux / weight decay / cooldown mix under KD | (queued: kwaux0, kwwd05) | kd_reprice#10, winner_model#4, long_bets#6, kd_audit#4 | l.311 "QUEUED 11-14 (kwaux0, kwwd05, kwh30 x 4 seeds)" |
| D12 | history-drop function matching | (queued: kwh30; lane kwh10) | kd_reprice#9, kd_literature#2 (part) | l.311; l.184 HDROP-ALL replay |
| D13 | W8 train-only public probe | (declared (D), file built) | ft_redesign#8, forensics_skeptic#10 | docs/04 row 12 (G3 TO-only -0.00047) |
| D14 | fp32 inference is null | winner_model#7 | kd_audit#12, ft_redesign#11 (part) | r6_tta "-0.00003 ± 0.00016" for d128 |
| D15 | item-CTR privileged teacher is null; user-bagged members are weaker | kd_literature#5, kd_literature#10 | long_bets#9 (part), long_bets#2 | l.250 "User-half data-split members \| UNTESTED-EXTERNAL" (now measured) |
| D16 | short-feed holdout by sub-sampling; W8 > S16 on public is noise; pick 2 | lb_stats#5, lb_stats#3, lb_stats#1 | pool_composition#12, pool_composition#7, pool_composition#11 | TRICKS §4.2 "M50-HOLDOUT \| Attach to the next from-scratch 10-ep family; not standalone"; l.162 PICK-EMAX |

## 3. Verdict per finding

P = adjusted probability in the finding's own sense (gain for levers and nulls, "the risk is real" for risks).

### kd_audit
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | keep | No earlier audit of the KD path; independent recomputation of every headline number. | 0.95 |
| 2 | keep | New and material. E_kd 20:21Z (1): "Read: member deltas @last and selected, paired per feed, pooled over the 2 seeds, on the holdout and val": no fold split. The fold-0-user slice of the holdout is also the channel TRICKS §4.1 names ("TTT / Mean-Teacher on val+test \| Opens the banned same-user later-feed channel"), so GO must be read on the cross-user columns. | 0.6 |
| 3 | keep | l.241 "Checkpoint read @last as primary (protocol) ... RE-OPEN (rule now ...)". The wave-1 rule already has "@last >= 0"; the size of the bias (base +0.00036, students +0.00006) is new and applies to the gru / kdmix / w224 / kwsub / kwT reads. | 0.9 |
| 4 | modify | Its only test is kwaux0, queued (l.311). l.299 already closes the dose at pool level ("Q flat in kd_w 0.3-0.8, falls at 1.0"). The gradient-norm mechanism is new; no arm follows from it. | 0.15 |
| 5 | modify | D4: second, weaker pilot (1 member, 2 folds) of ft_redesign#2; keep as replication and for the asymmetric-recipe table. | 0.02 |
| 6 | modify | Caveat verified (train.py l.500: `qt = softmax(kd / kd_T)`, student at T = 1). Keep it as a read-out note. Drop the 6.5 GPU-h arm unless a kwT arm moves Qhat: the teacher is flat and calibrated (kd_literature#7: entropy 4.41 of 4.90 nats, T* 1.10). | 0.02 |
| 7 | modify | D9. Not closed (l.240 RE-OPEN, never run). Bounded at pool level by l.299: with one mean function a member gain is worth a/N, about +0.0002 at N = 8. Keep a 2-seed minimal version (kd08 plateau -> kd_w 0.5, with the kd05 control; base plateau -> kd_w 0.5), read on Qhat. | 0.02 |
| 8 | keep | D2 reference copy; mechanism (0.2-0.7% of label mass) is new. | 0.02 |
| 9 | drop | D1 duplicate (its own text records the correction to the band of the other lenses). | 0.02 |
| 10 | modify | D8 reference for the teacher side. Its conclusion "kwsub is a fair test" is not kept: the student-side confounds were measured (D7) and adopted at 20:21Z (3). | 0.02 |
| 11 | modify | Superseded by the adopted evidence rule (status §1: val >= 0 AND holdout >= 0, combined z >= 2). "Holdout first" conflicts with lb_stats#2 for popularity-light pools. Keep the SE table. | 0.5 |
| 12 | drop | D14. The +0.00081 on 2 folds of one member is a draw: ft_redesign#11 has two members with opposite signs, winner_model#7 the pool at -0.00012 ± 0.00015. | 0.2 |

### kd_reprice
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | keep | D6 reference. l.115 had the protocol; the proof that width itself fails the member screen is new; adopted at 20:21Z (3). | 0.9 |
| 2 | keep | D7. Earlier closures of the class: TRICKS §4.1 ":D candidate sub-list sampling ... Keeping all positives makes them label-dependent (the wave-2b aug failure mode, STOP)"; r6 `aug` cooldown arm end - selected -0.0021 val / -0.0016 holdout. Stale inputs are new. Section 1.4 adds that sub-feeds miss the short-feed label format. | 0.4 |
| 3 | modify | D1 + D2. Unique: rho_sys, agreement by split. Its W16 forecast (not above W8 on val) is the one section 1.1 supports. | 0.02 |
| 4 | keep | New statistic (mixing gain at fixed pool size). l.67, l.113, l.213-217 closed the same families on another criterion. | 0.03 |
| 5 | modify | D10: no longer a proposal (W8P20 declared at weight 0.2; wide pop1 students running). The evidence is weaker than stated: one correlated public reading, thresholds at 0.74 SE, val gain falling with weight (section 1.2). | 0.03 |
| 6 | drop | Premise contradicted: train does contain short exploration feeds, and adding them to the cooldown pool was null (l.293 "292K feeds, 105K with M<50 ... member val +0.00025 / -0.00016 / -0.00161; in a pool worse than seed duplicates (M<50 -0.0005..-0.0013)"). Sub-feed targets carry no new function (KL 0.002-0.005 nats). Section 1.4: sub-sampled feeds do not have the short-feed label format. Also inside the U20 KILL list. | 0.02 |
| 7 | drop | D4: answered today by two pilots and the bound of ft_redesign#1 (l.303, l.304). | 0.02 |
| 8 | modify | l.144 "hzcd popularity-parity cooldown \| CLOSED 11-12 (B STOP) \| val +0.0003, holdout -0.0005 (z -2.8), stale-val +0.0011 (cold/warm trade)". The judge change was refuted twice (TRICKS §2 HOLD-CLEAN "REFUTED as specified (outcome-chosen judge)"; l.252 PDA/TIDE "circular stale judge"). Own mixing gain +0.0002 ± 0.0002. Only as a follow-up if W8P20 adopts the popularity-light premise. | 0.02 |
| 9 | drop | D12: queued as kwh30 x 4, lane arm kwh10 (lane.sh l.125-126). | 0.02 |
| 10 | modify | D11: aux 0 is queued; aux 1.0 is lane arm kwaux1 and hard-label ap1 was null three times (l.86, l.203, l.209); kd_audit#4 shows kd05 already runs at the ap1 gradient ratio (0.32 vs 0.35). Only kwaux1, and only if kwaux0 is clearly worse. | 0.02 |
| 11 | modify | retr32 closed at 2 and 10 epochs (l.152; TRICKS l.46 "retr_m 32 (10 ep -0.0002 / -0.0003); retr32u (-0.0003 / -0.0003)"); late dropout l.164 / l.207 plus kw_drop0; cooldown length l.25; HS1 zero-sum by its own numbers. Reference class today: 0 of 7 hard-label-closed levers flipped under KD. Keep sim_bias 2 only (l.243 "RE-OPEN (idle lanes), P 0.012-0.02", never completed in any regime; lane arm kwsb2). | 0.03 |
| 12 | modify | TRICKS §4.1 "RESID-TARGET boosting member \| Its pre-check is a holdout stacker (closed class); ~10 GPU-h at P ~0.03"; the re-entry condition of TRICKS l.487 (honest first pass) is now met. Narrow to the 0.3 GPU-h falsification; fix the offset parity (3-model mean on train, 15-model on val). The ranker is noise-limited, not bias-limited. | 0.02 |

### kd_literature
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | modify | D9 duplicate; keep the 4-run falsification, drop the 36-run grid. Its "cost lever" outcome has no value now (l.301). | 0.02 |
| 2 | drop | (a) validity probe = D8; (b) its closing rule contradicts the adopted 20:21Z (3), and the 15-epoch follow-up is bounded by "kw_e20 member -0.0008 val / -0.0004 hold -> CLOSED" and is in the U20 KILL list; (c) the history-drop arms are D12. | 0.02 |
| 3 | modify | The d256 + dropout 0.2 cell is lane arm kw256d2 (defined, not queued), never run under KD; l.292 covers cooldown dropout 0.2 at d192 under hard labels only. Keep 2 seeds of kw256d2 after kd05w224 is read; drop the d192 twin. | 0.03 |
| 4 | drop | D4. | 0.02 |
| 5 | keep | D15 reference; l.140 and l.234 were input / residual forms, not a teacher-only feature. | 0.02 |
| 6 | drop | D2 duplicate. | 0.02 |
| 7 | keep | Measurement; forecasts for kwT; extends the bound to DKD / WSL. | 0.02 |
| 8 | modify | D6; E_kd l.9-10 already states the risk ("students collapse onto the teacher"). Keep the fidelity numbers. | 0.85 |
| 9 | drop | Its stated value (twice the students per GPU-hour) is closed by l.301. l.33 / l.37 stay open only as a cost lever nobody needs. | 0.01 |
| 10 | keep | Closes l.250 with existing models; D15 reference. | 0.01 |
| 11 | keep | New (cold / warm slice of the KD gain). | 0.02 |
| 12 | drop | Gated by U20 (E_kd 20:21Z "KILL closes in one step: val/test feeds as a transfer set ..."); TRICKS §4.1 banned-channel note applies. | 0.02 |

### lb_stats
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | modify | Value of pick 2 bounded before (l.162 "PICK-EMAX \| final-pick tie-break only \| claimed +0.00003..+0.0001"). Status §1 was re-centred on W8 / S16 at 20:30Z. Open: rewrite FREEZE_PROTOCOL before 12-07. | 0.9 |
| 2 | keep | Re-fit on 7 new public scores (l.220 rested on 7 files). Lineage split is post hoc. | 0.6 |
| 3 | keep | D16 reference. | 0.02 |
| 4 | keep | Adopted as (C) at 20:16Z; section 1.1 agrees. | 0.8 |
| 5 | modify | Not closed (TRICKS §4.2 M50-HOLDOUT was never built). The sub-sampling form fails the label-format check of section 1.4; replace it by the natural short-feed set (item N2). | 0.03 |
| 6 | drop | D1 duplicate; its val forecast band (0.40161-0.40175) is above the 12-wide read (0.40132). | 0.02 |
| 7 | keep | New sizing of the noise share of the gap. | 0.85 |
| 8 | keep | Corrects VAL_PUBLIC ± 0.001; adopted in status §1. | 0.9 |
| 9 | modify | Principle is prior (FREEZE §4; l.100; l.163). The SE table is new and contradicts two declared probes (section 4, C5). | 0.01 |
| 10 | keep | l.168 dropped PPI-LENS for another reason; the variance-reduction form is now measured. | 0 |
| 11 | keep | Repeat of VAL_PUBLIC §1.6 on files that did not exist then. | 0.05 |
| 12 | modify | The platform check is an old open action (TRICKS §2 "confirm on ods.ai that the best of 2 selected files counts"; status §5). Status §1 is fixed; docs/04 l.26-28 still names G3 and Pall as the picks. | 0.5 |

### ft_redesign
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | keep | D3 reference; extends l.139 / l.100 / l.275 to students with a bound. | 0.02 |
| 2 | keep | D4 reference (pre-declared pilot, half-lr control). | 0.01 |
| 3 | keep | New, and actionable now: status §4 says wave 1 is "Queued behind the W16 fine-tunes", which this finding (and pool_composition#5, kd_audit#5) shows are not needed for the file. | 0.01 |
| 4 | keep | New control (hard-label twins); corrects the attribution in forensics_skeptic#1, pool_composition#8, kd_audit#5. | 0.85 |
| 5 | keep | New decomposition; l.120 / l.281 cold offsets stay closed. | 0.02 |
| 6 | keep | New parity measurement; no action. | 0.95 |
| 7 | drop | D5: l.100 (closed three times) and the same W8 numbers in three other reports. Fold into #1. | 0.02 |
| 8 | modify | D13: declared and built. Decision value is small (pooled FT-half value below zero needs a W8 reading about one SE against its forecast). | 0.03 |
| 9 | drop | l.103 "LP-FT ... -0.00014"; l.139 "extra branch null on m02 trunks"; l.303 "low-dim FT ... CLOSED by bound". The author does not recommend it. | 0.01 |
| 10 | drop | l.102 "pseudo-labels (test) \| CLOSED \| negative"; l.285 FT-PSEUDO "CLOSED by mechanism"; TRICKS §4.1 banned channel; no val-visible effect. | 0.01 |
| 11 | keep | Clean checks; D14 for the precision part. | 0.01 |
| 12 | modify | l.305 found no input to feed it ("best nested block +0.00006 ± 0.00016"). Section 1.3 shows its pre-gate (nested residual on val) reads a real input block at zero or below (26% of its value on the holdout), so it would reject every real input of ordinary size. Keep only "judge at file level". | 0.02 |

### pool_composition
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | keep | D1 reference; second out-of-sample check in section 1.1. | 0.02 |
| 2 | modify | D6 duplicate. Unique: asymptote table of 8 constructions; its SE for a 2-seed asymptote delta (0.0011 / 0.0006) is twice kd_reprice's run sd (0.0006 / 0.0003), so the +0.0004 extension bar of 20:21Z (3) is at or below 1 SE. | 0.05 |
| 3 | keep | l.219 "second-level combiners ... 0/78 pass val AND holdout" on hard-label pools; repeated on student groups, same answer. | 0.02 |
| 4 | drop | D2 duplicate. | 0.03 |
| 5 | keep | Null: five ties, none to submit; documents the asymmetric-file equivalence. | 0.02 |
| 6 | modify | Val side known (E_kd l.18 "C3 TO + kd05 x4 -0.00004"; l.296 "C3 + students 0"). Holdout side is new but carries the pop1 bias of C3. Action already taken (W8C5 not submitted). | 0.02 |
| 7 | drop | D16 duplicate of lb_stats#3. | 0.85 |
| 8 | drop | D3 duplicate; attribution corrected by ft_redesign#4. | 0.8 |
| 9 | keep | Pool-level re-read of the capacity closures with per-feed SEs; agrees with kd_reprice#1; l.300. | 0.02 |
| 10 | drop | D10: declared as W8P20. Its adverse evidence should travel with the probe (gain on Tuesday only, several weights read). | 0.03 |
| 11 | drop | Superseded by status §1 at 20:30Z (pick 2 = S16, C3 as the alternative); value at most +0.0001 (l.162; lb_stats#1). | 0.7 |
| 12 | drop | D16 duplicate of lb_stats#5; its premise check was run (section 1.4) and fails at label level. | 0.02 |

### winner_model
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | keep | D7 (size-mix shift, false format motive); adopted at 20:21Z (3). | 0.8 |
| 2 | drop | Bounded: sub-feed logits equal full-feed logits (KL 0.002-0.005 nats, a tenth of seed noise), kw_e20 closed under KD, l.210 "training length 15 / 20 / 30 epochs ... CLOSED", and it is in the U20 KILL list. 13 GPU-h. | 0.02 |
| 3 | keep | Verified: r6_tta.md has T1 MC-dropout, T2 history dropout, T3 history crop only; the candidate axis was never measured before. | 0.01 |
| 4 | modify | D11: aux 0 and weight decay 0.05 are queued with a pool-level rule. Cooldown mix is bounded (l.203 "m04 -0.0008 / -0.0005"; kd_reprice §7 "pool-level dose curve flat from 0.01 to 0.10"). Only second doses (kwwd10, aux 0.25) after wave 1. | 0.04 |
| 5 | keep | New null (per-candidate production-membership score). | 0.02 |
| 6 | keep | Refines docs/01 §3 ("set ranker-shaped: f1 alone scores 0.336 vs random 0.290"); no COVERAGE_MAP row. Section 1.4 is a second sign that test-format feeds cannot be made from production or long feeds. | 0.7 |
| 7 | keep | D14 reference. | 0.01 |
| 8 | modify | Audit stands. "--loss_norm idcg never reported" is wrong: TRICKS §4.1 "Training form neutral (v2_idcg 0.38004 vs 0.38003); acts on 26% of feeds; ceiling ±0.0001". Drop the optional 6.4 GPU-h pair. | 0.01 |
| 9 | drop | l.294 "M-split blend (TO-only on M<50) ... CLOSED 11-14 \| +0.00006 ± 0.00029" plus D3. | 0.02 |
| 10 | drop | Restates l.62-64, l.137, l.287, l.292, l.300; kd05w224 already has a rule (E_kd 18:24Z). | 0.02 |

### forensics_skeptic
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | drop | D4; premise corrected by ft_redesign#4. | 0.02 |
| 2 | keep | Corrects a recorded near-miss of TOP1_GAP_FORENSICS; l.305 agrees. Caveat of section 1.3: this closes the residual form only. | 0.01 |
| 3 | modify | l.232 "Per-user prior-error / preference from earlier labelled feeds \| CLOSED 11-13 (nested G3, pre-declared)", l.226, l.227. New: W8 base, non-linear form, holdout. Same answer. | 0.01 |
| 4 | modify | l.120 and l.281 ("CLOSED for good"). New: W8 base and the calibration read. | 0.01 |
| 5 | modify | Instances of closed classes (l.280 routing; l.192 per-cluster FT/TO mix; l.303 group-conditional FT weight). New: the seen-user grouping on W8. | 0.01 |
| 6 | keep | New method result (unconstrained residual damages the base). Section 1.3 limits its "detection floor": it holds for signals that val can teach, not for real inputs. | 0.9 |
| 7 | modify | Repeat of forensics probe 1; value-level equality is the new part. | 0.01 |
| 8 | drop | Same script and slice as #5; remedy closed (l.227). Merge into #5. | 0.8 |
| 9 | drop | Closed class: l.163 "FT/TO confidence mixing ... CLOSED 11-13 \| bounded by infinite-member headroom +0.0002 val / ~0 hold". +0.0003 after about ten contrasts, no second witness. | 0.01 |
| 10 | drop | D13 duplicate, with a decision rule that differs from the declared one; (D) governs. | 0.03 |
| 11 | modify | Contingent on a kwsub pass (interim negative). winner_model#3 measured sub-feed averaging at -0.0007 on students never trained on sub-feeds. Keep only as the zero-GPU diagnostic of winner_model#1. | 0.01 |
| 12 | drop | Restates TRICKS §5 ("not closable by any of the 63 items") and the two TOP1_GAP reports; its numbers live in #6 and ft_redesign#1. | 0.8 |

### long_bets
| id | verdict | prior evidence (quoted) and reason | P |
|---|---|---|---|
| 1 | modify | The seen = unseen measurement (0.9933 / 0.9933) is new. "Closed" is a forecast until U20 reads (queued); it conflicts with the 4-seed kwh30 queue (section 4, C6). | 0.02 |
| 2 | modify | D15: same 15 fold models as kd_literature#10. Unique: the pool-level slope; confounded with 20% fewer steps. | 0.85 |
| 3 | keep | Owner's decision. Prior: GLOBAL_RECAP §6.4, frontier law l.110. Re-priced at W8. The +0.0010..+0.0026 is an extrapolation: no partner at correlation below 0.95 and gap below 0.005 was ever observed (own GBDT: 0.90 and 0.017). | 0.3 |
| 4 | drop | D1 + D2 duplicate. | 0.02 |
| 5 | modify | Untested cell; l.299's "flat in kd_w" was read at d128. Narrow to completing kd08w192 to 4 seeds (seed 951 exists: 0.40021) and 2 seeds of lane arm kd08w256; drop d320 and kd_w 0.65. | 0.03 |
| 6 | modify | D11: aux and weight decay queued; cooldown mix bounded (l.203; kd_reprice §7); lr 1.4e-3 bounded by l.19 and a mixing gain of 0; kd_w 0.65 by l.299. Nothing to launch before wave 1 reads. | 0.02 |
| 7 | drop | Bounded by its own arithmetic: l.284 "ALL-LABELS-CD ... CLOSED by arithmetic ... ~+0.00004 val" for the pool part, plus 1% more plateau feeds, total about +0.0001. Cannot be verified, costs 16-20 GPU-h in the freeze week and puts a never-executed code path into the final pick. | 0.01 |
| 8 | drop | Queued under a pre-declared rule (E_kd 20:21Z (1)). Open part is kd_audit#2. | 0.02 |
| 9 | modify | PFD part is D15. The co-click null is new (l.308). Drop the optional NN screen (proxy negative; label-dependent target). | 0.01 |
| 10 | keep | New null (l.111 was a hard-label GBDT). | 0.01 |
| 11 | drop | Self-retraction; D4. The "nested teacher" it leaves open leaks the held-out fold through the soft target (kd_literature#4). | 0.02 |
| 12 | drop | Already in 20:21Z (3); D6. | 0.5 |

## 4. Contradictions between findings

| # | conflict | resolution |
|---|---|---|
| C1 | kd_audit#10 and kd_literature#2: "a kwsub null closes function matching". kd_reprice#2 and winner_model#1: "it cannot decide". long_bets#1: "closed by mechanism before any read". | Different objects: the teacher targets are valid (4 lenses), the student inputs are not (2 lenses). The adopted rule is 20:21Z (3). Section 1.4 adds a third defect (label format). The clean form (kd_reprice#6) is dropped on other grounds. |
| C2 | Asymptote and W16 forecasts: val Q 0.40154 (kd_reprice), 0.40176 (lb_stats), 0.40181 (pool_composition), 0.40195 (long_bets); falsification bars overlap (kd_reprice: "falsified above 0.40165 val"; lb_stats: forecast 0.40161-0.40175). | The 12-wide read (0.40132 / 0.32450) favours kd_reprice. Use one bar: W16 minus W8 >= +0.0004 on the holdout. |
| C3 | "Fine-tune does not help KD students" (forensics_skeptic#1 z -3.6, pool_composition#8, kd_audit#5) against "it is the w5_e10 recipe family" (ft_redesign#4). | ft_redesign#4 has the controlled comparison (hard-label twins -0.00029 ± 0.00041). |
| C4 | kd_audit#11 and pool_composition#7: the holdout is the better judge. lb_stats#2: for the newer files the holdout alone is the worst predictor (k 0.38). | Both hold: the holdout misreads popularity-light pools only. The adopted two-witness rule covers it. |
| C5 | lb_stats#9: near copies (corr >= 0.998) "never" get a slot, w_ft variants "no". The lead declared W8P20 (test corr 0.9994, holdout negative) and the W8 train-only probe (0.9979). | Slots are not scarce (5 a day), so the cost is over-reading. For W8P20 the thresholds are 0.74 SE (section 1.2). |
| C6 | long_bets#1 predicts kwh30 null and kd_reprice#9 says "run only if a sub-feed arm passes"; the lead queued kwh30 at 4 seeds. l.184: history dropout costs -0.0008 at 10% in one pass. | kwh30 is the weakest use of 4 runs in wave 1; 2 seeds with the Qhat extension rule would do. |
| C7 | Cooldown mix under KD: long_bets#6 "first in the order", winner_model#4 two arms; kd_reprice §7 "moot: pool-level dose curve flat from 0.01 to 0.10". | kd_reprice has the measurement; kd_audit#7 shows the cooldown gain is unchanged under KD at kd_w 0.5 (+0.0032 both). Bounded. |
| C8 | Cost of one d192 run: 0.8 GPU-h (kd_reprice, long_bets) against 3.2 h (kd_audit, winner_model). | Same run: 3.2 h wall at 4 jobs per GPU = 0.8 GPU-h. Budgets quoted in "GPU-h" differ by 4 between reports. |
| C9 | SE of a 2-seed asymptote: 0.0003 holdout (kd_reprice, run sd) against 0.0006 (pool_composition, delta). | Different noise sources (seed against feed sampling). Until combined, treat the +0.0004 bar as 1 SE. |
| C10 | Residual-gate bars: +0.0005 at 2 SE (ft_redesign#12), +0.0013 (forensics_skeptic#6). | Section 1.3: at a recovery of 0.26 (holdout) the bars correspond to a trained-input value of +0.002 and +0.005; on val there is no recovery. |
| C11 | W16: "do not spend a submission" (kd_reprice#3) against "submit once" (lb_stats#4, lead (C)). | One validity reading of a pick candidate is fine; the acceptance rule of lb_stats#4 stands. |
| C12 | Pick 2: S16 (lb_stats#1) against C3 (pool_composition#11). | Either is worth at most +0.0001; owner's choice, already in status §1. |

## 5. What is left, and what nobody proposed

### 5.1 Untested GPU items after de-duplication (all low prior; none is running or queued)

| item | from | narrowest form | cost | P |
|---|---|---|---|---|
| kd_w 0.8 at width 192 / 256 | long_bets#5 | kd08w192 seeds 952-954 (seed 951 exists) and lane arm kd08w256 x 2; pool-level read | about 4 GPU-h | 0.03 |
| d256 with dropout 0.2 | kd_literature#3 | lane arm kw256d2 x 2, after kd05w224 reads | 2 GPU-h | 0.03 |
| typed similarity bias | kd_reprice#11 | lane arm kwsb2 x 2 | 1.6 GPU-h | 0.02 |
| phase-split re-cool | kd_audit#7, kd_literature#1 | kd08 plateau -> kd_w 0.5 with the kd05 control, and base plateau -> kd_w 0.5; 2 seeds | 1-2 GPU-h | 0.02 |
| residual on the cross-fitted teacher | kd_reprice#12 | the 2-epoch falsification only | 0.3 GPU-h | 0.02 |
| two-sided temperature | kd_audit#6 | only if a kwT arm moves Qhat | 1.6 GPU-h | 0.02 |

Reference class for pricing: 8 levers closed under hard labels were re-run under distillation today; width 192 flipped, the other 7 (dropout 0, 20 epochs, ID table, 2 history layers, FFN x4, width 256, 3 blocks) did not. Inferred combined chance that one of the six gives a pool gain of +0.0005: about 0.1.

### 5.2 Zero-GPU actions that follow from this review

1. **U20 read-out**: add the three columns of kd_audit#2 and apply GO / KILL to the cross-user columns only. The fold-0-user column is the banned same-user channel, not evidence for a transfer-set programme.
2. **W8P20 rule**: either widen the thresholds to ± 2 SE (± 0.0008) or state now that the reading is one input next to val and holdout under the adopted evidence rule. Do not start more pop1-view GPU work on a reading inside ± 0.0008. Carry pool_composition#10's adverse evidence in the probe's record.
3. **Do not hold wave 1 behind the W16 fine-tunes**: three lenses show the file is the same with the W8 fine-tunes as the FT branch (ft_redesign#3: -0.00001 ± 0.00023).
4. **kwh30**: 2 seeds, not 4 (C6).
5. **Records**: docs/04 l.26-28 still names G3 and Pall; COVERAGE_MAP rows closed by residual gates need the words "residual form only" (section 1.3); l.46 "dropout (0.1) \| NEVER" is stale (kw_drop0, LATE-DROPOUT and WIDTH-GATE drop02 varied it).

### 5.3 Ideas nobody proposed

**N1. Calibrate every gate against a real positive control before using it to close.** Section 1.3 is the first such calibration. The stored ablation members give more controls at no cost: `abl_xdrop_ctxnd`, `abl_xdrop_ctxhn_hs`, `abl_no_retr`, `abl_e10_xdrop_pre2` (val and holdout predictions exist for all). Rule to adopt: a gate may close an idea only if it recovers at least half of a known block of similar size on the same data. By that rule the residual gates (recovery 0.10 to 0.26 on the holdout, none on val) close nothing about trained inputs. Consequence for any further feature work: the only valid test is a student arm (the KD block is already ported to `nn_x/abl/train.py`), 2 seeds, pool-level read; there is no cheap pre-gate. Given 5 direct nulls so far, I price a further feature arm at P 0.02 and would not start a hunt; the point is that the 300 residual rows are weaker evidence than their count suggests, and that ft_redesign#12 / forensics_skeptic#6 should not be used as closing rules.

**N2. A natural short-feed witness, and a doubled witness for screens, from user-fold-excluded models.** The XPOOL posterior marks 56,015 short (M < 50) exploration-like train feeds (105,374 at p > 0.5) whose label format matches val short feeds (section 1.4). They are in-sample for W8-type members but out-of-sample (a) today for the hard-label fold models: `teacher_oof.npy` and `teacher_xf0..4` hold their logits on every train row; (b) for any model trained with `--xfold 0`: about 11,000 short and 26,000 long exploration-like train feeds of fold-0 users (inferred as 20% of 56,015 and of the 128,293 long ones outside the holdout), next to the 30,000 holdout feeds. Uses, in order of cost:
- now, CPU: read the hard-label recipe by feed size on natural short random-like feeds (the table of section 1.4 is the start); any label-free question about short feeds can be asked on 56K feeds instead of the 5,893 of val;
- when u20a / u20b / u20c finish (already queued, fold 0 excluded, teacher that never saw fold 0): dump their logits on the fold-0 exploration-like feeds and read B - A and C - A there. Inferred SE about 0.0003-0.0004 on the short set and about 0.0002 on long plus holdout, against 0.0005-0.0009 on val M < 50;
- later screens: run arm and reference both with the u20a flags (80% of users, 20% fewer steps). The level cost (about -0.002) cancels in the paired read; the witness is twice the holdout and includes short feeds. This is the M50-HOLDOUT of TRICKS §4.2 without a new exclusion list.
Limits: the short set is chosen by a classifier (selection on set composition and order, not on labels); conclusions at 80% of users may differ slightly for capacity levers.

**N3. Price a "closed under hard labels, retry under KD" arm by its reference class.** 1 flip in 8 so far. Each further arm of this type (kd_reprice#11, long_bets#6, winner_model#4) should be listed with P at most 0.03, and the batch should stop after the first 4 nulls of wave 1 and the lane's remaining arms.

**N4. Merge is the only item above P 0.1 (long_bets#3).** The technical part that can be done before the owner decides: a label-free compatibility check to ask of any partner (within-feed correlation of two test files; the frontier law then prices the blend), prepared as a script, so the decision by 12-01 is made on a number.

## 6. Limits of this review

- I read the nine reports' findings as given in the task and checked their sections where a verdict depended on it; I did not re-run their scripts except `pc_pool_vs_w8.py`.
- Section 1.3: the "true" value on val is 1.6 SE (popularity block) and 0.7 SE (pre2), so only the holdout row is a clean calibration; one block is one control. Two of the five pre2 columns are not in the feature store. The GBDT gate chose its round cap (250) in one holdout half, so 0.26 may be a slight under-read of an uncapped gate.
- Section 1.4: NAT is selected by the XPOOL posterior; the scorer is the 3-member hard-label fold pool, not a student.
- Rows 296-311 of COVERAGE_MAP were written today from these same reports. Where I quote them, they show that the lead has recorded a finding, not that independent earlier evidence exists.

## 7. Files

`nn_x/w12/recheck/duplicates/d01_w8p20_se.py`, `d02_gate_cal.py`, `d03_short_witness.py`; outputs in `out/` (`d00_w12_read.txt`, `d01_w8p20_se.txt`, `d02_val_pre2.txt`, `d02_val_pop.txt`, `d02_hold_pop.txt`, `d03_short_witness.txt`).
