# veto: audit of closures that rested on the in-period holdout, and of adoptions that leaned on it (KEY = veto)

Written 2026-11-15 (02:18Z onward) by the veto agent. CPU only. Work directory `nn_x/w12/stale/veto/`. Conventions: +- = 1 SE paired per feed; val = 14,423 feeds; holdout = 30,000 in-period feeds; [M] = measured by me in this audit, [D] = read in a project file (path given), [I] = inferred. "pmix-priv" = val per-feed deltas reweighted to the private mix (`src/pmix.py`); "stale-val" = val re-predicted with private-like inputs (`nn_x/w5/stale_pred.py` scenario); "coldpos / warmpos" = val feeds whose positives are all cold / none cold. Pools are train-only (TO) member means of per-feed z unless "final" is written (0.5 z(FT) + 0.5 z(TO)). Rules of every re-read: `nn_x/w12/stale/veto/PREDECLARED.txt` (02:23:09Z, before the first re-read; addenda R1 03:39:52Z and R2 for the resume-run reads, each written before its read). [R] marks what the resume run (03:31Z onward) added or corrected.

## 0. Verdict, and the plain list of what to re-open

State of this file: complete. The first run (02:18-03:22Z) was cut by an account limit; the resume run (03:31Z onward) re-checked a sample of the first run's numbers by an independent code path (six pool reads, 22 numbers, all equal to 1e-6: section 3.9), finished the open reads, and added sections 3.8-3.9, 5.1 and the rows marked [R]. It makes ONE CORRECTION to the first run: the basis of the aux-weight arm (item 10) is weaker than written at 03:12Z.

**Verdict (8 lines).**
1. Screen: all 277 idea rows of `reports/COVERAGE_MAP.md` (336 file lines), the 142 registry lines, the component ledger and the read-out logs named in section 1. The list has 38 rows: 31 closures with the holdout as deciding or co-deciding negative while val was >= 0 or unread, 4 closed as a "cold/warm trade" on fresh val, 1 process row, 2 that turn out not to be holdout vetoes. 15 of the 38 belong to the staleness family, where the holdout is a biased judge by mechanism.
2. Re-read under the new policy (12 staleness-family reads from stored predictions): 10 pass "val >= 0 and pmix-private >= +1 SE", 2 fail; the 10 are three ideas: the popularity-light view as pool members, the same view inside the KD teacher, the hz cooldown. Stale-val: hzcd +0.00084 +- 0.00054 at selected and +0.00115 +- 0.00051 at @last (stored GPU files). On CPU re-predictions (information by my own rule; identical to the witness agent's files to 1.5e-5 in logit) the scored W8P20 form reads +0.00088 +- 0.00036 (z 2.5) and the declared 16-student form with 8 pop1 students +0.00065 +- 0.00033 (z 2.0), while same-view control admixtures read -0.0002 .. -0.0008.
3. Nothing outside the staleness family changes status. Row 301 re-read (85 stored families, each as a 0.2 admixture to W16): 5 carry the "val up, private-mix up, holdout down" signature against 6.8 expected by chance, the 4 non-staleness ones all sit at the bar (z 1.0-1.2), no family passes the two-witness rule, and the popularity-light families are the only ones that lift cold-positive feeds by +0.002 or more. CORRECTION: aux weight 1.0 (hard labels, 7 roots) reads +0.00070 +- 0.00028 on val only at holdout-selected checkpoints; at @last it reads +0.00009 +- 0.00025, because the reference's selected 7-pool sits 0.00068 below its own @last pool.
4. Reverse check: no adopted setting is contradicted off the holdout. KD weight 0.5, cooldown mix 0.02, cooldown length and the 0.5 / 0.5 FT weight were chosen by val (or public) and hold on val and pmix. Width 192 rested on the holdout (+0.00083 +- 0.00028); off it: val +0.00038 +- 0.00049, CPU stale-val +0.00075 +- 0.00050, pmix-priv -0.00008 +- 0.00074: it holds. New [R]: lr 2e-3 (on record with holdout z 2.9, val z_run 1.4) reads val +0.00051 +- 0.00067, cold-positive feeds -0.0018, pmix-priv +0.00002 +- 0.00103 against lr 1e-3: confirmed by the letter and worth nothing at the private mix.
5. Checkpoint selection on the holdout loses nothing net at pool level (ten hard-label 4-pools, selected minus @last: val +0.00003, stale-val +0.00012, pmix-priv +0.00013) but shifts quality from short feeds and cold positives to long feeds and warm positives (cold-positive feeds -0.0012, negative in 8 of 9 arms; M<50 -0.0003, negative in 9 of 9). Under distillation there is no shift: 18 student re-cools read cold-positive feeds -0.00003 +- 0.00019, and the 9 W16 members whose checkpoints differ read +0.00025 +- 0.00022 on CPU stale-val as a pool.
6. The full-popularity input preset is not shown to beat the popularity-light preset out of period: 4-pool val -0.0013 (z -1.5) for the light view, pmix-priv +0.0005 +- 0.0014; under private-like inputs the two wide student 4-pools are level (CPU stale-val, light minus full: +0.00047 +- 0.00091; the full view drops 0.0018, the light view 0.0000). The holdout's -0.0064 is the fresh-popularity artefact. On the lagged holdout the full view is still +0.0028 ahead [D], so the light view stays an admixture, not the base. On fresh val its dose saturates at +0.0003 .. +0.0007 for weights 0.1-0.5; on stale-val it keeps growing (0.1 / 0.2 / 0.3 / 0.5: +0.0004 / +0.0007 / +0.0011 / +0.0014); the lagged holdout does not support weights above 0.2 [D row 336]. The weight question is open and no file here answers it.
7. For the hz read: its rule decides on holdout-SELECTED checkpoints, and for the hard-label hzcd arm selection alone moved the stale-val read from z 2.25 (@last) to z 1.54 (selected). The lead adopted the @last column and a re-read with selection on the hz holdout file at 03:35Z. The hz re-cools started at about 03:38Z (9 of 16 started by 03:50Z, about 51 min each, first results expected near 04:30Z): this report contains no hz number.
8. Checks of the arms that the lead declared at 03:35Z from this audit [M]: `--xt_drop` blanks exactly the 8 popularity inputs, and a blanked row has the input pattern of a real cold row (all 8 inputs are zero on 100.0% of the cold rows of val and test); the train zero share of the LUO count becomes 0.26 / 0.39 at q 0.15 / 0.30 (val rows 0.379, test rows 0.447). `teacher_half_p1.npy` equals 0.5 teacher_mean11 + 0.5 mean of the 8 pop1 members up to a per-feed constant that the per-feed softmax removes (within-feed sd 1.01-1.05, against 1.02-1.07 and 1.00-1.03 for the two existing teachers).

**What should be re-opened, in order, and why.**

| # | re-open | why (one line) | cost | status at 03:50Z |
|---|---|---|---|---|
| 1 | **hz under distillation** (`K:hz` / `L:hz`, 16 re-cools), read at @last next to selected | the only closed staleness-family arm with stored stale-val files: +0.0008 (selected) / +0.0012 (@last) at 4-pool level, val >= 0; it was stopped by "holdout < 0" alone, at member level, on a dose that closes about 30% of the train-test coverage gap of 5 of the 8 popularity inputs | 4 GPU-h | RUNNING since 03:38Z; the @last column is in its rule since 03:35Z |
| 2 | **hz2 = full-dose test-like popularity view in the cooldown** (all 8 inputs; the `hz2` agent builds the view) | the model pays -0.0033 +- 0.0008 for stale inputs [D]; hz moves the train zero share of the LUO count from 0.13 to 0.23 while private test rows sit at 0.46, and leaves co-visitation fresh | 4 GPU-h per 16 re-cools | view under construction (`reports/STALE_1115/hz2.md`) |
| 3 | **popularity-light dose on the 16-student base**: fine-tune the 4 wide pop1 students, build the W16 twin of the scored W8P20 | 8 of 9 stored reads of the view as pool members pass the pmix-private bar; public rewarded it twice; CPU stale-val of the declared form +0.00065 +- 0.00033 (W8-based forms +0.0009, z 2.5-2.6; same-view controls negative) | 1 GPU-h | DECLARED 03:35Z as W16P20x8; the 4 fine-tunes wait for a GPU slot since 03:36Z |
| 4 | **popularity-input dropout** (`--xt_drop`; from scratch under KD) | uniform and age-conditioned dropout were both stopped by the holdout (old recipe: "the holdout prefers q = 0"; agedrop holdout -0.0019) with stale-val +0.0009 .. +0.0013; q 0.30 brings the train zero share of the LUO count to 0.39 (val 0.38, test 0.45) | 5-13 GPU-h (16 from-scratch runs at 0.3-0.8 GPU-h) | DECLARED 03:35Z as kwxd15 / kwxd30; flag checked (line 8) |
| 5 | **@last next to selected for staleness-family arms** (protocol; no GPU) | in hard-label re-cools holdout selection moves quality from cold positives and short feeds to warm positives and long feeds and hid a third of the hzcd stale-val read; student re-cools do not show it (18 runs), so this is a second column, not a rule change | 0 | ADOPTED 03:35Z (amendment to "ARM hz") |
| 6 | **fine-tune on val with private-like stale inputs (STALE-FT)** | never run ("no fine-tuned predictions exist at stale coverage" [D GLOBAL_RECAP_1113.md section 5 item 4]); not a holdout rejection, listed because it is the one staleness lever that trains on real out-of-period labels, and stale-val can judge it out of fold | 3 GPU-h + a copy of the fine-tune script | "prepared as code first" (lead, 03:35Z) |
| 7 | **popularity-light teacher dose** (re-cools distilled from a teacher that is half pop1 members) | the 12 students of teacher_mean25 (8 of its 25 members are pop1) against their 12 teacher_mean11 twins: val +0.00056 +- 0.00044, cold-positive feeds +0.00282 +- 0.00084, pmix-priv +0.00123 +- 0.00067 (at @last +0.00032 / +0.00282 / +0.00082); CPU stale-val only +0.00034 +- 0.00044. Same lever as item 3 and it does not add to it (section 3.7) | 4 GPU-h | DECLARED 03:35Z as arm th; teacher file checked (line 8) |
| 8 | **lag conditioning (HZCOND) on the hz2 view** | never run; refuted on 11-13 only through hzcd's holdout veto; it is the form in which the model is told how stale its counters are | CPU build + 4 GPU-h | "prepared as code first" |
| 9 | **PSEUDO-DIRECT** (pseudo-feeds without pre-computed inputs), as a 4-seed pool read | stopped by "holdout > +1 SE" with member val +0.00108 +- 0.00072; pool val +0.0007 .. +0.0013, pmix-priv +0.0013 .. +0.0016; late holdout positive | 10-20 GPU-h | not declared |
| 10 | **aux weight 1.0 under distillation** (`E:kwaux1`; not staleness family, the holdout stays a witness) | CORRECTED [R]: hard-label 7-pool val +0.00070 +- 0.00028 at selected checkpoints but +0.00009 +- 0.00025 at @last; 10-epoch pair +0.00060 +- 0.00081. What remains is the dose direction under KD: aux weight 0 reads -0.00123 +- 0.00058 val / -0.00114 +- 0.00034 holdout / cold-positive feeds -0.0057 at 7 seeds | 4-6 GPU-h | DECLARED 03:35Z as kwaux1; I would run it last and drop it first if GPU time is short |

Not re-opened: the age-conditioned dropout as built (fails the letter; item 4 replaces it), it_hist_hz / hzcv as GBDT-gated inputs (items 1-2 are their NN form), PDA / TIDE re-judging, HOLD-CLEAN as a judge (the lagged holdout replaces it), K/V branch, TIME arm, lr 1e-3 (level at the private mix, not better), 3 blocks (the hard-label cold-positive gain does not carry over to students: section 3.9), and every non-staleness arm of section 1.2. Details and exact commands: section 5; status and checks of the declared arms: section 5.1.

## 1. The list

Screened: 277 idea rows of `reports/COVERAGE_MAP.md` (274 table lines of the 336-line file of 03:40Z, three of which hold two rows; the first run counted 273 in lines 17-331; row numbers below are file line numbers), 142 lines of `reports/EXPERIMENT_REGISTRY.md`, `docs/03_EXPERIMENTS.md`, the summary lines of every stage-B and r6 cooldown read-out (`nn_x/w5/logs/w5_eval_B*.txt`, `nn_x/r6/logs/r6_eval_cd*.txt`), `w5_eval_E_pseudo.txt`, `abl_eval_time.txt`, `abl_eval_kv64.txt`, `nn_x/abl/pop1/x8/RESULT_pop1x8.txt`, `reports/ROUND5.md`, `reports/FEATURES_REVIEW.md` 3.1, `reports/TRICKS_SWEEP_1113.md` 2 and 4, `reports/DATA_PIPELINE_1113.md` 3-4, `reports/GLOBAL_RECAP_1113.md` 4, `reports/RECHECK_1114/SYNTHESIS.md` 4-5, `nn_x/insights/ccb_1113/RESULT.md`, `reports/nn_report.md` R2, `nn_x/w5/logs/E_kd_predeclared.txt`. [R] Second screen of the resume run: two pattern passes over every row that carries a holdout, veto, stale, pmix or cold/warm word (89 of the 114 such rows read again), `nn_x/abl/ledger/ledger.md` section 1 (components A-I, for the reverse check), `docs/05_METHODOLOGY.md` (rule table), `reports/RECHECK_1114/pool_composition.md`, `reports/FEATURES_REVIEW.md`, the `E_kd_predeclared.txt` blocks up to 03:35Z. It added five rows (301, 261, 266, 268, U20) and found no further closure of kind (a), (b) or (c).

Kinds: (a) holdout deciding or co-deciding negative with val >= 0 or unread; (b) val mildly negative while a staleness lens is positive; (c) a stop rule fired on its holdout clause. "Files" = stored predictions / checkpoints exist today.

### 1.1 Staleness family (the holdout is a biased judge; classification in section 2)

| row | idea | kind | witnesses as recorded when it was closed | rule that closed it | files |
|---|---|---|---|---|---|
| 144 | hzcd: popularity-parity cooldown (train batches read `it_hist_hz_*` for the 5 LUO columns) | a, c | member @last: val +0.00026 +- 0.00033 (M>=50 +0.00070, M<50 -0.00037), holdout -0.00052 +- 0.00019 (late 20% -0.00031 +- 0.00038), pmix-priv +0.00048 +- 0.00052, stale-val +0.00107 +- 0.00036, coldpos +0.00531 / warmpos -0.00181 [D `w5_eval_B_hzcd.txt`] | "STOP iff val < 0 or hold < 0 ..." fired on "holdout < 0" alone | preds, stale, holdout, holdout_hz, 8 checkpoints |
| 53 | age-conditioned input dropout (`--agedrop prv_rl13`, 2-epoch pair) | a | val +0.00005 +- 0.00061, stale-val +0.00089 +- 0.00062 (GPU basis +0.00126 +- 0.00063), holdout -0.00189 +- 0.00034, late 20% +0.00031 +- 0.00075; young-positive feeds +0.0247, others -0.0063 [D `ROUND5.md` 2a] | PASS iff stale-val > 2 SE and val >= -1 SE and holdout >= -1 SE: failed on two clauses; "the holdout fails on its own" | preds, holdout, 2 checkpoints (`nn_x/solvers`); no stale file |
| 52 | uniform popularity-input dropout (`--xt_drop` 0.15 / 0.3, AdamW era, 1 seed) | b | val -0.0004 +- 0.0006 / -0.0007 +- 0.0007, holdout -0.0003 / -0.0009, cold-positive feeds +0.0129 / +0.0130, warm -0.0043 / -0.0047 [D `nn_report.md` R2 step 2] | "chosen on the holdout ... the holdout prefers q = 0" | none (deleted) |
| 217, 208 | popularity-view families as pool members: Q1a (pop-free x4), Q2a (pop1 x4), Q2a8 (pop1 x8), Q3 (pop2 x4), Q3b (pop1 + pop2) | a, c | Q1a V +0.00036 +- 0.00030, H -0.00052 +- 0.00021, pmix +0.00112; Q2a V +0.00029, H -0.00002, pmix +0.00062; Q2a8 V +0.00057 +- 0.00040, H -0.00073 +- 0.00029, pmix +0.00107; Q3 V +0.00015, H -0.00023, pmix +0.00084; Q3b V -0.00010, H -0.00062 [D registry 06:40, 08:16, 15:27, 15:57] | decide9 / x8 rule clause (e) "H >= -1 SE": "VETOED" | all preds, FT files, checkpoints |
| 78 | it_hist_hz (age-gated LUO for train rows), hzcv (lagged co-visitation), "causal / lagged popularity" as inputs | a | GBDT gates: it_hist_hz val +0.00111 +- 0.00075 (other harnesses -0.00027, -0.00067), holdout -0.0010 .. -0.0015, pmix +0.0008 n.s.; young-positive feeds +0.041, others -0.0053; hzcv val +0.00032 +- 0.00071, holdout -0.00057 +- 0.00041 [D `FEATURES_REVIEW.md` 3.1] | adversarial verdicts: "cold/warm trade", holdout negative in every run; hzcv also fails coverage parity (co-vis zero share 0.65 vs 0.42-0.48) | feature files in `data/feats_new/train`; no NN run except hzcd |
| 252 | PDA / TIDE re-judging of popularity-robust arms | a | no arm run; "hzcd half dose; circular stale judge; matched-control holdout -0.0004 .. -0.0008" | two skeptics, on hzcd's holdout and a holdout dose proxy | none |
| 167 | MODDROP-POP (per-feed zeroing of the 8 popularity inputs + indicator), HZ-FULL (hz view for the whole run), HZCOND (lag indicator input), POP-REL | a | never run; re-entry triggers were met on 11-13; `GLOBAL_RECAP_1113.md` 4.2 refuted the re-entry: "hzcd fails the holdout veto (z -2.75)" | deferred, then refuted on the holdout record of hzcd | none |
| 168 | COLDMIMIC | b | never run: "cold/warm-trade class; content-only scores add ~0" | dropped by class | none |
| 302 | pop1-view students as member-equal partners of the student pool | a | [kd05t25 x4 + pop1kd x4] vs kd05t25 x4: val 0.40106 vs 0.40095; rule needed +0.0005 val and holdout >= -0.0003 [D `E_kd_predeclared.txt` 10:47Z] | val bar first, holdout clause second; re-opened 11-14 as the 0.2 admixture | all |
| 330 | wide pop1 students (d192 x4) | a | 0.8 z(W16) + 0.2: val +0.00047 +- 0.00031, holdout -0.00046 +- 0.00022 [D 01:06Z] | rule A': not fine-tuned unless public >= W8 + 0.0006 (read: +0.00052) | preds, checkpoints; no FT |
| 299 | KD teacher with popularity-light members (teacher_mean25 = 11 tf + 6 GRU + 8 pop1; arms kd05t25, kd08t25, kd05t25w192) | a | kd05t25 4-pool 0.40095 / 0.32317 against kd05 0.40085 / 0.32349; kd05t25w192 0.40100 / 0.32412 against 0.40085 / 0.32349 [D `E_kd_predeclared.txt` 10:47Z] | "+0.0005 val AND holdout delta >= -0.0003": val bar first; kd05t25 also misses the holdout clause (-0.00032); then "teacher-identity class closed" | all preds; teacher parts `nn_x/w12/kd/out/teacher_p1.npy`, `teacher_p2.npy` (sums of 4 pop1 members each) |
| 326 | hzcd students | a | never run: "substitute judges for a closed arm" [D SYNTHESIS 5] | dropped; re-opened 11-15 as arm hz | - |
| 264 | C07 item trend / lifecycle at a gap-matched as-of (residual on G3) | a | val +0.00031 +- 0.00038, holdout -0.00024, late -0.00047, pmix +0.00031 +- 0.00059, coldpos +0.00432 / warmpos -0.00123 [D `ccb_1113/RESULT.md`] | failed val >= +0.0005 first; holdout and late clauses also | logs only |
| 280 | feed-conditional routing between G3 and popularity-light pools | a | nested val -0.0003 .. -0.0010; val-fitted router on the holdout -0.0025 (z -6.0) | closed on val; the holdout number was the headline | scripts |
| 160, 220, 253 | validation design: HOLD-CLEAN re-judging on Sep 9-19; "holdout kept as veto for popularity-light contrasts"; gap-matched lens | process | hzcd late 20% -0.00031 +- 0.00038; "outcome-chosen judge" | refuted / kept on 11-13; contradicted by two public readings (C3 - G3 +0.00063, W8P20 - W8 +0.00052) | - |

### 1.2 Not staleness family (the holdout veto stands; listed so that nothing is hidden)

| row | idea | kind | recorded witnesses | my pool-level re-read [M] | status |
|---|---|---|---|---|---|
| 209, 203 | aux_pos 1.0 (double K9 aux weight), cooldown form and 10-epoch pair | a, c | member 7 roots val +0.00038 +- 0.00017, holdout +0.00019 +- 0.00009, "7-root pool holdout -0.00023"; 10-epoch pair val +0.00049 +- 0.00071, holdout +0.00022 +- 0.00041 (rule: holdout > +1 SE) | 7-pool vs m02ref 7-pool: val +0.00070 +- 0.00028, pmix-priv +0.00099 +- 0.00042, coldpos +0.00084 / warmpos +0.00052, holdout -0.00023 +- 0.00015 | veto stands (not a cold/warm trade). [R] CORRECTION: at @last the 7-pool reads val +0.00009 +- 0.00025, pmix-priv +0.00007 +- 0.00035, holdout -0.00011 +- 0.00012; the +0.00070 exists only at holdout-selected checkpoints (section 3.9). Item 10 of section 5 |
| 295 | PSEUDO-DIRECT | c | member val +0.00108 +- 0.00072, holdout +0.00024 +- 0.00042, pmix +0.00078 | 2-pool val +0.00072 +- 0.00084 (selected), +0.00133 +- 0.00083 (@last); pmix-priv +0.00130 +- 0.00124 / +0.00159 +- 0.00123; holdout -0.00012 / +0.00029; late holdout +0.00062 / +0.00100 | borderline mechanism (section 2); proposed as item 9 of section 5 |
| 331 | GRU students as diversity ([W8 + 4 GRU] vs W8) | a, c | val +0.00016 +- 0.00033, holdout -0.00024 +- 0.00018 (rule: holdout >= +0.0003) | GRU 4-pool vs wide 4-pool: val 0.00000 +- 0.00059, coldpos -0.00263 +- 0.00114, holdout -0.00018 | stands |
| registry l.140 | 3 blocks, 10-epoch pair | a, c | val +0.0003, holdout -0.0004, pool +0.0012 / -0.0006 | 2-pool val +0.00121 +- 0.00083, pmix-priv +0.00256 +- 0.00122, holdout -0.00055 +- 0.00047 | stands: under KD the 3-block students read 0.40026 vs 0.40075 on val [D] |
| registry l.111 | a4u unique retrieval slots, cset | a | val +0.00036 / +0.00017, holdout +0.00018 / -0.00001, "trade warm for cold" | a4u 6-pool: val +0.00058 +- 0.00061, pmix-priv -0.00023 +- 0.00093, holdout -0.00048 +- 0.00034 | stands (pmix does not support it) |
| R7 | dwa (DenseFormer), 2-epoch screen | a | val +0.00064, holdout -0.00027 | not re-read | stands |
| 62 | d = 256 / 8 heads (AdamW era) | a (val unread) | holdout -0.0004 to 27.5k steps, -0.0008 at 2 epochs; no val number in the row | not re-read (old recipe) | stands: under KD the d256 students read 0.40014 against 0.40075 on val [D `E_kd_predeclared.txt` 10:47Z] |
| 36 | Ce = Cd + F8 (2-epoch seed family) | a | val +0.00020, holdout 0.32208 vs 0.32319 | not re-read | stands (weak members) |
| 198 | embedding projection proj_mu | a | val +0.00045 +- 0.00022, holdout -0.00004 | - | stands |
| 219 | val-fitted family weights (F10 / F7x / F7x70) | a | val +0.00049 (z 1.5), holdout -0.00048 (z -2.85) | - | stands (same-view families) |
| 263 | C06 Bradley-Terry item strength | a | val +0.00028 +- 0.00037, holdout -0.00015, pmix +0.00059 +- 0.00055 | - | stands: failed its val bar first; it adds a train-period item statistic |
| 271 | C14 cross-feed consensus | a | info val +0.00022 +- 0.00018, holdout -0.00050 (z -2.7), pmix +0.00067 +- 0.00035 | - | stopped by its coverage gate (4.5% < 10%), not by the holdout |
| 277 | time-gap features | a | best +0.0003 +- 0.0003, holdout <= 0 | - | stands (val bar first) |
| 293 | XPOOL exploration posterior in the cooldown | a | member val +0.00025, holdout -0.00019 | 4-pool val -0.00011 +- 0.00048, holdout +0.00027 | stands |
| 157 | PRODDIR / DOMAIN-DIAG (inference-time production-direction removal) | b | val +0.00011 +- 0.00017, all-cold-positive feeds +0.0026, others -0.0007; no holdout read | not re-read (side-observation files not located) | closed as "pure cold/warm trade" on fresh val; see section 6 |
| 291 | K/V branch ("stale veto") | - | val -0.00007 +- 0.00053 (needed +0.0015), stale-val veto | 2-pool @last: val -0.00020 +- 0.00054, stale-val -0.00019 +- 0.00056 | stands: the stale lens is against it |
| 276 | TIME arm (weekday / hour inputs) | a | holdout weekend slice +0.00021 +- 0.00051, val +0.00017, stale -0.00080 | 2-pool @last: val -0.00001 +- 0.00058, stale-val +0.00004 +- 0.00058, pmix-priv -0.00083 +- 0.00090 | stands: null on the only split with weekend labels |
| 205, 212 | R-Drop, SAM, cooldown batch size | - | val null, holdout positive | sam 4-pool: val +0.00033 +- 0.00045, stale-val +0.00068 +- 0.00044, holdout +0.00025 +- 0.00023; rdrop val -0.00024 | not holdout vetoes; closed on val and on the in-pool replay |
| 301 [R] | pool-composition screen of 11-14, judged holdout-first: "0 of 52 families adds on the holdout at 2 SE", "C3 admixture hurts the holdout at every weight", forward selection | a | W8 + pop1kd x4 member-equal: val +0.00020 +- 0.00046, holdout -0.00113 +- 0.00030; W8P = 0.7 z(W8) + 0.3 z(hard pop1 x8), final: val +0.00039 +- 0.00041, pmix-priv +0.00077 +- 0.00066, TO holdout -0.00117 +- 0.00029 ("rejected by the holdout-first rule"); selection fitted on val picks pop1 groups and reads -0.00095 on the holdout [D `RECHECK_1114/pool_composition.md`] | section 3.8: 85 stored families as a 0.2 admixture to W16; 5 flags (6.8 expected by chance), no two-witness pass | its popularity-light part is rows 302 / 330 (staleness family, re-opened); for every other family the screen stands |
| 261 [R] | C04 co-visitation variants + NB log-lift (residual on G3) | b | val -0.00020 +- 0.00034, holdout -0.00030 (z -1.70), pmix +0.00033 +- 0.00053, coldpos +0.00595 / warmpos -0.00245 [D `ccb_1113/RESULT.md`] | - | BORDERLINE by mechanism (other co-visitation windows; measured as a cold/warm trade). FAIL: val < 0, pmix 0.6 SE; it failed its val bar (+0.0005) first |
| 266, 268 [R] | C09 RP3beta walk; C11 typed implicit rating (residuals on G3) | a by the letter | val +0.00012 +- 0.00036 / +0.00006 +- 0.00033, holdout -0.00013 / -0.00018, pmix -0.00056 +- 0.00055 / -0.00071 +- 0.00048 (veto), C09 coldpos -0.00400 / warmpos +0.00211 | - | stand: both failed the val bar first and the private-mix lens is against both |
| `E_kd` 03:13Z [R] | U20: fold-0 feeds kept with teacher-only targets (B) against the fold dropped (A), 2 seeds | c | B - A val +0.00158 +- 0.00071, holdout -0.00017 +- 0.00039; the KILL clause "B - A < +0.0004 on the holdout" was met | as a 0.2 admixture to W16 the u20b 2-pool is null (val +0.00005 +- 0.00029) | NOT closed: extended to 4 seeds on the owner's directive (GO iff val >= +2 SE and holdout >= -1 SE). Not staleness family: the holdout stays a guard, which is the right form for it |

Reverse cases (holdout positive, val negative, closed on val) were not counted: m01 (val -0.00035, holdout +0.00046), R-Drop, d20, pp02.

## 2. Classification (from the mechanism)

The definition was written before any re-read (`PREDECLARED.txt` item 2); the table applies it item by item and was filled in afterwards, from what each idea does, not from how it read. Definition: an idea is in the staleness family iff it (a) removes, lags, blanks or re-weights popularity / recency inputs, (b) trains the model on inputs with test-like staleness, (c) is an admixture or pool weight of such members, or (d) exists to make cold items rank better.

Why the holdout is biased for this family [D `docs/01_DATA_AND_FINDINGS.md` 8; `reports/NEXT_1115/poplag.md` 1.1]: on a holdout row the LUO counts include dense snapshots after the row and co-visitation sees every earlier train day; warm-item staleness is 1.75 d against 9.34 d (public) and 11.23 d (private). Any change that makes the model lean less on these counters loses on the holdout by construction and can only be judged where the counters are stale.

| item | class | reason |
|---|---|---|
| hzcd, hz, hz2, HZ-FULL, it_hist_hz, hzcv | STALENESS (b) | train rows get counters with a test-like horizon |
| age-conditioned dropout, uniform `--xt_drop`, MODDROP-POP | STALENESS (a) | blanks the popularity inputs on train rows |
| pop-free, pop1, pop2 members; W8P20; wide pop1 students; pool weights of these | STALENESS (a, c) | 8 popularity inputs removed (one exposure or click-rate count added) |
| teacher_mean25 students | STALENESS (c) | the distillation target is an average that contains popularity-light members: the same admixture, applied to the target instead of the prediction (confounded with the 6 GRU members of that teacher) |
| HZCOND, C07 gap-matched item trend | STALENESS (b) | staleness made explicit or matched between train and test |
| COLDMIMIC, routing to popularity-light pools | STALENESS (d, c) | cold-item handling / feed-wise weight of popularity-light members |
| PDA / TIDE, HOLD-CLEAN, row 220 | STALENESS (process) | re-judging rules for the family |
| PSEUDO-DIRECT | BORDERLINE | main mechanism = more training lists from histories (not staleness); side effect = half of the training rows have all 34 pre-computed inputs blank, which is (a). I apply the non-staleness rule and report the staleness lenses |
| PRODDIR / DOMAIN-DIAG | BORDERLINE | inference-time domain token; its measured effect is a cold/warm trade; no holdout read was involved |
| aux_pos 1.0, 3 blocks, GRU students, a4u / cset, dwa, XPOOL, proj_mu, family weights, K/V, TIME, R-Drop / SAM, batch size, C06, C14, time-gap | NOT | capacity, objective weight, architecture, pool composition among same-view members, calendar inputs, added train-period statistics: none changes how popularity is used under stale inputs |
| cooldown mix (m0 / m01 / m02 / m04) | NOT | share of exploration feeds in the cooldown: a domain lever. It does move cold positives (m02 - m0: +0.0031), see section 4 |

## 3. Re-reads from stored predictions

Tooling check [M]: my pool code reproduces the lead's stored numbers exactly: hzcd 4-pool @last val +0.00050 +- 0.00047 and holdout -0.00067 +- 0.00027 (`w5_eval_B_hzcd.txt` "ENS"); W8P20 TO holdout -0.00063 +- 0.00022; Q2a8 holdout -0.00073 +- 0.00029; W8P20 final pmix-priv +0.00079 +- 0.00048.

Verdict words (`PREDECLARED.txt` item 3): PASS-S = val >= 0 and stale-val >= +2 SE; PASS-P = val >= 0 and pmix-priv >= +1 SE; FAIL otherwise. PASS-P is weak: same feeds as val, 16% false passes under the null.

### 3.1 hzcd (4 hz re-cools against 4 reference re-cools, hard labels) [M]

| checkpoint | val | M>=50 | M<50 | coldpos | warmpos | pmix-pub | pmix-priv | stale-val (stored GPU files) | holdout | holdout >= Sep 13 | Sep 17-19 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| selected (decision) | +0.00079 +- 0.00051 | +0.00013 +- 0.00054 | +0.00175 +- 0.00098 | +0.00348 +- 0.00106 | -0.00028 +- 0.00068 | +0.00089 +- 0.00063 | +0.00105 +- 0.00076 (z 1.39) | +0.00084 +- 0.00054 (z 1.54) | -0.00011 +- 0.00028 | +0.00061 +- 0.00039 | +0.00123 +- 0.00051 |
| @last | +0.00050 +- 0.00047 | -0.00044 +- 0.00051 | +0.00186 +- 0.00090 | +0.00452 +- 0.00102 | -0.00086 +- 0.00062 | +0.00042 +- 0.00059 | +0.00048 +- 0.00072 | +0.00115 +- 0.00051 (z 2.25) | -0.00067 +- 0.00027 | -0.00042 +- 0.00038 | +0.00003 +- 0.00048 |

- **Selected: PASS-P** (val >= 0, pmix-priv +1.39 SE); not PASS-S (1.54 SE). **@last: PASS-S** (2.25 SE). The 11-12 stop used the member-level @last holdout (-0.00052, z -2.75).
- At pool level the selected hz re-cools are not below the reference on the holdout (-0.00011 +- 0.00028) and are above it on the causal days Sep 17-19 (+0.00123 +- 0.00051, 8,279 feeds).
- In a pool with a matched control the effect is diluted as expected: G3 + hzcd x4 against G3 + m02ref x4: val -0.00003 +- 0.00019 (selected), +0.00031 +- 0.00018 (@last), pmix-priv -0.00018 / +0.00041, holdout -0.00010 / -0.00016 (4 of 24 members exchanged).
- Dose [D `nn_x/w5/logs/w5_cd_hzcd_r101.*`]: zero share of `it_hist_all_luo` on train rows 0.130 -> 0.228 under hz; private test rows (M >= 50) 0.455 [D poplag 1.2]. The three co-visitation inputs are untouched. hz closes about 30% of the coverage gap of 5 of the 8 inputs.

### 3.2 Popularity-view families in hard-label pools [M]

Train-only level, member-equal, against G3 (20 members):

| pool | val | M>=50 | M<50 | coldpos | warmpos | pmix-pub | pmix-priv | holdout | holdout >= Sep 13 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1a = G3 + pop-free x4 | +0.00043 +- 0.00033 | +0.00097 | -0.00036 | +0.00471 | -0.00134 | +0.00089 +- 0.00043 | +0.00106 +- 0.00053 (z 2.0) | -0.00052 +- 0.00021 | -0.00090 +- 0.00031 | PASS-P |
| Q2a = G3 + pop1 x4 (951-954) | +0.00037 +- 0.00034 | +0.00116 | -0.00077 | +0.00409 | -0.00126 | +0.00064 +- 0.00044 | +0.00063 +- 0.00054 (z 1.2) | -0.00002 +- 0.00021 | -0.00044 +- 0.00032 | PASS-P |
| Q2a' = G3 + pop1 x4 (955-958) | +0.00088 +- 0.00035 | +0.00141 | +0.00012 | +0.00378 | -0.00046 | +0.00099 +- 0.00045 | +0.00104 +- 0.00055 (z 1.9) | -0.00022 +- 0.00021 | -0.00047 +- 0.00032 | PASS-P |
| Q2a8 = G3 + pop1 x8 | +0.00068 +- 0.00045 | +0.00156 | -0.00061 | +0.00743 | -0.00213 | +0.00101 +- 0.00057 | +0.00117 +- 0.00070 (z 1.7) | -0.00073 +- 0.00029 | -0.00067 +- 0.00043 | PASS-P |
| Q3 = G3 + pop2 x4 | +0.00045 +- 0.00034 | +0.00105 | -0.00042 | +0.00423 | -0.00098 | +0.00092 +- 0.00043 | +0.00121 +- 0.00054 (z 2.3) | -0.00023 +- 0.00022 | -0.00063 +- 0.00032 | PASS-P |
| Q3b = G3 + pop1 x4 + pop2 x4 | +0.00038 +- 0.00045 | +0.00140 | -0.00109 | +0.00700 | -0.00242 | +0.00084 +- 0.00057 | +0.00113 +- 0.00071 (z 1.6) | -0.00062 +- 0.00029 | -0.00077 +- 0.00044 | PASS-P |
| control: G3 + base x4 (same view) | -0.00003 +- 0.00023 | -0.00003 | -0.00003 | +0.00037 | -0.00026 | -0.00006 | -0.00005 +- 0.00036 | -0.00003 +- 0.00013 | -0.00026 +- 0.00018 | - |

Final-form files (0.5 z(FT) + 0.5 z(TO)), with the public reading where one exists:

| pair | val | pmix-pub | pmix-priv | PUBLIC |
|---|---|---|---|---|
| Q1a - G3 | +0.00036 +- 0.00030 | +0.00090 +- 0.00039 | +0.00112 +- 0.00048 | +0.00007 [D] |
| Q2a - G3 | +0.00029 +- 0.00031 | +0.00054 +- 0.00040 | +0.00062 +- 0.00050 | +0.00039 [D] |
| Q2a8 - G3 | +0.00057 +- 0.00040 | +0.00090 +- 0.00050 | +0.00107 +- 0.00062 | - |
| C3 - G3 | +0.00069 +- 0.00040 | +0.00108 +- 0.00050 | +0.00130 +- 0.00062 | +0.000634 |
| C3 - C2 (same 3-seed FT) | +0.00051 +- 0.00040 | +0.00085 +- 0.00051 | +0.00108 +- 0.00063 | - |

- No stale-val files exist for these 10-epoch hard-label members, so PASS-S cannot be read. All six pass PASS-P; they are one view (pop1 and pop2 pools correlate 0.986 [D]).
- The veto of Q2a8 (clause (e), holdout -0.00073) is the clearest holdout-only rejection on record: the same members with a 3-seed fine-tune are C3, which scored +0.00063 over G3 on public.
- The gain is carried by long feeds (M >= 50: +0.0010 .. +0.0016) and cold-positive feeds (+0.004 .. +0.007); short feeds are flat to negative and warm-positive feeds lose 0.0005 .. 0.0024. pmix reweights exactly these two axes (M bin x cold-share bin), which is why it reads higher than plain val.

### 3.3 Popularity-light students as an admixture (train-only level) [M]

| pool against its base | val | M>=50 | M<50 | coldpos | pmix-pub | pmix-priv | holdout | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8 (the W8P20 form) | +0.00055 +- 0.00035 | +0.00081 | +0.00018 | +0.00246 | +0.00095 +- 0.00042 | +0.00124 +- 0.00052 (z 2.4) | -0.00063 +- 0.00022 | PASS-P |
| the built W8P20 file (final form) vs W8 | +0.00049 +- 0.00031 | +0.00080 | +0.00003 | +0.00209 | +0.00063 +- 0.00039 | +0.00079 +- 0.00048 (z 1.6) | - | PASS-P; public +0.000523 |
| 0.8 z(W16) + 0.2 z(pop1kd d128 x4) vs W16 | +0.00008 +- 0.00032 | +0.00061 | -0.00069 | +0.00238 | +0.00012 +- 0.00040 | +0.00026 +- 0.00049 (z 0.5) | -0.00056 +- 0.00022 | FAIL (pmix < 1 SE) |
| 0.8 z(W16) + 0.2 z(pop1kd w192 x4) vs W16 | +0.00047 +- 0.00031 | +0.00087 | -0.00012 | +0.00267 | +0.00041 +- 0.00039 | +0.00050 +- 0.00048 (z 1.04) | -0.00046 +- 0.00022 | PASS-P (at the bar) |
| 0.8 z(W16) + 0.2 z(pop1kd d128 + w192 x8) vs W16 | +0.00030 +- 0.00031 | +0.00057 | -0.00009 | +0.00254 | +0.00037 +- 0.00038 | +0.00051 +- 0.00048 (z 1.07) | -0.00055 +- 0.00021 | PASS-P (at the bar) |
| 0.8 z(W16) + 0.2 z(hard pop1 x8) vs W16 | +0.00031 +- 0.00035 | +0.00076 | -0.00034 | +0.00307 | +0.00024 | +0.00031 +- 0.00053 | -0.00066 +- 0.00023 | FAIL |
| control: 0.8 z(W16) + 0.2 z(hard base x6) | -0.00008 +- 0.00023 | +0.00009 | -0.00033 | -0.00062 | +0.00001 | +0.00005 +- 0.00036 | -0.00023 +- 0.00013 | - |
| control: 0.8 z(W16) + 0.2 z(narrow kd05t25 x4) | +0.00017 +- 0.00021 | +0.00016 | +0.00019 | -0.00003 | +0.00006 | +0.00007 +- 0.00034 | -0.00008 +- 0.00011 | - |
| member-equal [kd05t25 x4 + pop1kd x4] vs kd05t25 x4 (row 302) | +0.00012 +- 0.00057 | +0.00062 | -0.00061 | +0.00841 | +0.00045 | +0.00052 +- 0.00086 | -0.00161 +- 0.00036 | FAIL |

Weight scan on W16 with all 8 pop1 students (information; no weight is chosen here): w 0.1 val +0.00036 +- 0.00019 / pmix-priv +0.00076 +- 0.00030 / holdout -0.00017; w 0.2 +0.00030 / +0.00051 / -0.00055; w 0.3 +0.00032 / +0.00047 / -0.00107; w 0.4 +0.00059 / +0.00106 / -0.00166; w 0.5 +0.00074 +- 0.00054 / +0.00129 +- 0.00082 / -0.00200. Val and pmix are flat in the weight (+0.0003 .. +0.0007); only the holdout has a slope.

Reading [I]: on the 16-student base the admixture is worth about +0.0003 on val (seven admixture reads between -0.00003 and +0.00047), half of what the W8-based file shows. The W8P20 val number contains a W8-specific part; the public reading (+0.00052 +- 0.00041) was also taken against W8.

### 3.4 The other staleness-family items

| item | re-read | verdict |
|---|---|---|
| age-conditioned dropout (2 members, 2 epochs) | 2-pool val -0.00056 +- 0.00070 [M] (member level +0.00005 [D]); pmix-priv -0.00021 +- 0.00107; coldpos +0.01911 / warmpos -0.00856; holdout -0.00195 +- 0.00038, Sep 17-19 -0.00041 +- 0.00070; stale-val +0.00089 +- 0.00062 [D] | FAIL by the letter (pool val < 0, stale-val < 2 SE); member-level, noise; the full-strength dose (drop rate up to 0.96 for the youngest listings) |
| uniform `--xt_drop` | files deleted | UNREADABLE |
| it_hist_hz, hzcv as inputs | GBDT gates only; no NN prediction with these inputs except hzcd | UNREADABLE (hz / hz2 are the NN form) |
| MODDROP-POP, HZ-FULL, HZCOND, POP-REL, COLDMIMIC, PDA / TIDE | never run | UNREADABLE |
| C07 gap-matched item trend | val +0.00031 +- 0.00038, pmix +0.00031 +- 0.00059 [D] | FAIL (pmix 0.5 SE); residual form only |
| routing to popularity-light pools | nested val < 0 [D] | FAIL |
| 2-epoch input screens, stored GPU stale files, 2 seeds, @last [M]: pop-free - base | val -0.00246 +- 0.00100; stale-val -0.00062 +- 0.00100; pmix-priv -0.00106 +- 0.00148; holdout -0.00701 | (see the preset row of section 4) |
| pop1 - base | val -0.00172 +- 0.00097; stale-val +0.00010 +- 0.00100; pmix-priv +0.00025 +- 0.00146; holdout -0.00600 | |
| pop2 - base | val -0.00107 +- 0.00098; stale-val +0.00073 +- 0.00100; pmix-priv +0.00125 +- 0.00147; holdout -0.00569 | |

Count: 12 staleness-family reads with stored predictions, one pre-chosen form per list item (hzcd, agedrop, Q1a, Q2a, Q2a', Q2a8, Q3, Q3b, the 50 / 50 partner, W8P20, the wide-student admixture on W16 in the lead's form, the teacher_mean25 students of section 3.7); the other admixture variants of section 3.3 are information. PASS-P: 10. PASS-S on selected checkpoints: 0 of the 2 that have stale files (hzcd 1.54 SE; agedrop 1.44 SE [D]). Under the null about 1.9 PASS-P would be expected from 12 independent reads; these are not independent (three ideas, one val sample).

### 3.5 CPU stale-val re-predictions of mine (information only)

`stale_cpu.py` is a CPU fp32 copy of `nn_x/w5/stale_pred.py`. Its control on the first member (`w5_e10_kd05w192_s951`, no overrides): within-feed correlation to the stored bf16 file 0.999933 (bar 0.9999: met), NDCG 0.39897 against 0.39859 (bar 0.0003: **missed by 0.00008**). I set that bar before the run and apply it: no verdict of this report uses these files. The second control (`abl_e10_pop1kd05w192_s951`) met both bars: correlation 0.999935, NDCG 0.39963 against 0.39981 (-0.00018). The two offsets have opposite signs, so this is fp32-against-bf16 noise of one member on 14,423 feeds (the poplag control saw -0.00018 .. +0.00016 on 30,000 feeds [D]) and my bar was too tight; it stays applied. Paired CPU-against-CPU deltas do not carry the offset. The `witness` agent computes the same scenario with its own declared controls and is the reference for stale-val of the student pools.

<!-- cpu-block-start -->
State at 2026-11-15T03:48:46Z: 25 CPU stale-val files of mine exist (`nn_x/w12/stale/veto/preds/`); reads that need a file not yet written are absent from the table, not estimated.

| CPU stale-val read (14,423 feeds, CPU against CPU) | stale-val delta | z | private-weighted | M>=50 | M<50 | coldpos | warmpos |
|---|---|---|---|---|---|---|---|
| S - L, pool of 9 members | +0.00025 +- 0.00022 [0.39960 vs 0.39935] | +1.14 | +0.00029 +- 0.00040 | +0.00034 +- 0.00023 | +0.00013 +- 0.00043 | -0.00015 +- 0.00037 | +0.00043 +- 0.00031 |
| S - L, member-level mean over 9 runs | +0.00023 +- 0.00012 [0.39785 vs 0.39762] | +1.95 | +0.00009 +- 0.00018 | +0.00032 +- 0.00013 | +0.00010 +- 0.00022 | +0.00052 +- 0.00024 | +0.00011 +- 0.00015 |
| pop1kd w192 x4 vs kd05w192 x4 (family level) | +0.00047 +- 0.00091 [0.39997 vs 0.39950] | +0.52 | +0.00258 +- 0.00136 | +0.00188 +- 0.00106 | -0.00157 +- 0.00163 | +0.00095 +- 0.00165 | +0.00015 +- 0.00126 |
| pop1kd d128 x4 vs kd05t25 d128 x4 (family level) | -0.00085 +- 0.00088 [0.39908 vs 0.39993] | -0.97 | +0.00009 +- 0.00130 | -0.00045 +- 0.00100 | -0.00142 +- 0.00159 | +0.00160 +- 0.00152 | -0.00209 +- 0.00122 |
| 0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8 | +0.00088 +- 0.00036 [0.40070 vs 0.39982] | +2.48 | +0.00085 +- 0.00052 | +0.00172 +- 0.00039 | -0.00034 +- 0.00066 | -0.00005 +- 0.00064 | +0.00126 +- 0.00049 |
| 0.8 z(W16) + 0.2 z(pop1kd d128 x4) vs W16 | +0.00081 +- 0.00035 [0.40075 vs 0.39994] | +2.34 | +0.00119 +- 0.00051 | +0.00175 +- 0.00037 | -0.00056 +- 0.00065 | +0.00112 +- 0.00065 | +0.00066 +- 0.00047 |
| 0.8 z(W16) + 0.2 z(pop1kd w192 x4) vs W16 | +0.00074 +- 0.00033 [0.40068 vs 0.39994] | +2.25 | +0.00086 +- 0.00047 | +0.00183 +- 0.00035 | -0.00085 +- 0.00062 | +0.00045 +- 0.00061 | +0.00091 +- 0.00045 |
| 0.8 z(W8) + 0.2 z(pop1kd w192 x4) vs W8 | +0.00091 +- 0.00034 [0.40073 vs 0.39982] | +2.64 | +0.00110 +- 0.00050 | +0.00148 +- 0.00037 | +0.00007 +- 0.00064 | +0.00014 +- 0.00059 | +0.00129 +- 0.00048 |
| 0.9 z(W8) + 0.1 z(pop1kd w192 x4) vs W8 | +0.00034 +- 0.00024 [0.40016 vs 0.39982] | +1.41 | +0.00067 +- 0.00036 | +0.00084 +- 0.00025 | -0.00039 +- 0.00046 | -0.00014 +- 0.00038 | +0.00049 +- 0.00034 |
| 0.7 z(W8) + 0.3 z(pop1kd w192 x4) vs W8 | +0.00112 +- 0.00043 [0.40094 vs 0.39982] | +2.62 | +0.00148 +- 0.00063 | +0.00214 +- 0.00047 | -0.00037 +- 0.00079 | +0.00081 +- 0.00077 | +0.00128 +- 0.00059 |
| 0.5 z(W8) + 0.5 z(pop1kd w192 x4) vs W8 | +0.00166 +- 0.00056 [0.40148 vs 0.39982] | +2.94 | +0.00235 +- 0.00085 | +0.00256 +- 0.00063 | +0.00034 +- 0.00104 | +0.00132 +- 0.00104 | +0.00200 +- 0.00077 |
| W8 (8 wide) vs 8 narrow (kd05 + kd05t25) | +0.00075 +- 0.00050 [0.39982 vs 0.39907] | +1.51 | +0.00052 +- 0.00076 | +0.00084 +- 0.00053 | +0.00062 +- 0.00094 | +0.00001 +- 0.00093 | +0.00098 +- 0.00068 |
| kd05w192 x4 vs kd05 x4 | +0.00064 +- 0.00057 [0.39950 vs 0.39886] | +1.11 | +0.00109 +- 0.00087 | +0.00046 +- 0.00058 | +0.00090 +- 0.00112 | +0.00225 +- 0.00121 | -0.00002 +- 0.00076 |
| kd05 x4 vs hard base x4 | +0.00127 +- 0.00062 [0.39886 vs 0.39759] | +2.06 | +0.00135 +- 0.00090 | +0.00128 +- 0.00062 | +0.00124 +- 0.00121 | +0.00263 +- 0.00127 | +0.00079 +- 0.00082 |
| W16 vs W8 | +0.00012 +- 0.00034 [0.39994 vs 0.39982] | +0.36 | +0.00005 +- 0.00051 | -0.00027 +- 0.00034 | +0.00070 +- 0.00067 | -0.00024 +- 0.00070 | +0.00036 +- 0.00045 |

Pool level under private-like inputs (CPU stale against the stored bf16 fresh file, so each drop carries the numerics offset of about +-0.0003; read the difference between rows):

| pool | stale-val | fresh val (stored) | drop |
|---|---|---|---|
| kd05w192 x4 | 0.39950 | 0.40134 | -0.00184 +- 0.00053 |
| pop1kd w192 x4 | 0.39997 | 0.40000 | -0.00003 +- 0.00018 |
| pop1kd d128 x4 | 0.39908 | 0.39943 | -0.00035 +- 0.00020 |
| W8 | 0.39982 | 0.40157 | -0.00174 +- 0.00052 |

<!-- cpu-block-end -->

[R] Sources and cross-check of the table above (`crosscheck.py`, `logs/crosscheck.txt`): 25 CPU files are mine; the selected checkpoints of `w5_e10_kd05w192_s956-958`, `w5_e10_kd05t25w192_s955-958`, the 8 narrow students, `w5_e10_base_s951-954` and `abl_e10_pop1kd05t25_s954` come from the witness agent's cache (read-only), which runs the same scenario with its own copy of the script. The 12 members that exist in both places agree to 1.5e-5 in logit and to less than 1e-6 in NDCG, so the two sources are one tool. The witness agent's control B' for that tool is VALID (effect of the override equal to the GPU script's to 0.00015 per member, 0.00006 at 4-pool level [D `witness.md` section 2]); its control b2 failed by the letter and my own bar was missed by 0.00008: every CPU stale-val number in this report stays information-grade, and the lead decides which control governs.

[R] More CPU stale-val reads (`read12.py`; 14,423 feeds; train-only pools, selected checkpoints; fresh-val columns from the stored files for the same pair):

| pool against its base | fresh val | pmix-priv (fresh) | CPU stale-val | z | stale, private-weighted | stale M>=50 | stale M<50 |
|---|---|---|---|---|---|---|---|
| 0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8 (the scored W8P20 form) | +0.00055 +- 0.00035 | +0.00124 +- 0.00052 | +0.00088 +- 0.00036 | +2.48 | +0.00085 +- 0.00052 | +0.00172 +- 0.00039 | -0.00034 +- 0.00066 |
| 0.9 z(W16) + 0.1 z(pop1 students x8) vs W16 | +0.00036 +- 0.00019 | +0.00076 +- 0.00030 | +0.00035 +- 0.00023 | +1.53 | +0.00036 +- 0.00034 | +0.00087 +- 0.00025 | -0.00039 +- 0.00043 |
| **0.8 z(W16) + 0.2 z(pop1 students x8) vs W16 (the declared W16P20x8 form)** | +0.00030 +- 0.00031 | +0.00051 +- 0.00048 | **+0.00065 +- 0.00033** | +1.96 | +0.00085 +- 0.00048 | +0.00171 +- 0.00035 | -0.00090 +- 0.00063 |
| 0.7 z(W16) + 0.3 z(pop1 students x8) vs W16 | +0.00032 +- 0.00040 | +0.00047 +- 0.00060 | +0.00106 +- 0.00041 | +2.56 | +0.00145 +- 0.00062 | +0.00205 +- 0.00044 | -0.00038 +- 0.00078 |
| 0.5 z(W16) + 0.5 z(pop1 students x8) vs W16 | +0.00074 +- 0.00054 | +0.00129 +- 0.00082 | +0.00139 +- 0.00055 | +2.51 | +0.00244 +- 0.00083 | +0.00249 +- 0.00061 | -0.00020 +- 0.00103 |
| control: 0.8 z(W16) + 0.2 z(narrow kd05t25 x4) vs W16 | +0.00017 +- 0.00021 | +0.00007 +- 0.00034 | -0.00023 +- 0.00022 | -1.07 | +0.00028 +- 0.00033 | +0.00011 +- 0.00021 | -0.00072 +- 0.00043 |
| control: 0.8 z(W16) + 0.2 z(hard base x4) vs W16 | -0.00041 +- 0.00026 | -0.00029 +- 0.00040 | -0.00076 +- 0.00027 | -2.84 | -0.00054 +- 0.00040 | -0.00029 +- 0.00024 | -0.00145 +- 0.00056 |
| teacher_mean25 students x12 vs teacher_mean11 twins x12 | +0.00056 +- 0.00044 | +0.00123 +- 0.00067 | +0.00034 +- 0.00044 | +0.77 | +0.00101 +- 0.00066 | +0.00070 +- 0.00045 | -0.00019 +- 0.00084 |
| kd05t25w192 x8 vs kd05w192 x8 | +0.00044 +- 0.00048 | +0.00044 +- 0.00073 | +0.00072 +- 0.00048 | +1.48 | +0.00071 +- 0.00073 | +0.00132 +- 0.00050 | -0.00016 +- 0.00094 |
| kd05t25 x4 vs kd05 x4 (d128) | +0.00010 +- 0.00054 | +0.00109 +- 0.00081 | +0.00107 +- 0.00054 | +1.98 | +0.00231 +- 0.00086 | +0.00101 +- 0.00058 | +0.00117 +- 0.00103 |
| 0.8 z(T8) + 0.2 z(pop1 x8) vs 0.8 z(W16) + 0.2 z(pop1 x8) (does the teacher dose add to the member dose?) | -0.00021 +- 0.00028 | -0.00013 +- 0.00042 | -0.00000 +- 0.00029 | -0.00 | +0.00024 +- 0.00043 | -0.00023 +- 0.00031 | +0.00033 +- 0.00056 |

Reading [M / I].
- Under private-like inputs the popularity-light admixture reads +0.0007 .. +0.0009 at weight 0.2 on both bases (z 2.0-2.6 in five forms), where fresh val had read the 16-student base at +0.0001 .. +0.0005. A same-view admixture of equal weight reads -0.0002 (narrow students) and -0.0008 (hard-label members), so the gain belongs to the view and not to "any admixture". On the letter of `PREDECLARED.txt` these would be PASS-S for the W8 forms and for W16 with either 4-student pool (z 2.2-2.6) and one hair short for the declared 8-student form (z 1.96); they are not counted as verdicts because the source is information-grade.
- The gain sits in long feeds (M >= 50: +0.0015 .. +0.0018, z 4-5); short feeds read -0.0009 .. +0.0001 (negative in four of the five forms at weight 0.2, z down to -1.4). A rule with an "M<50 >= -1 SE" clause (as "RULE hz" has) would be decided by that clause for an admixture of this kind.
- On stale-val the gain grows with the weight up to 0.5; on fresh val it is flat; the lagged holdout peaks at 0.1-0.2 [D row 336]. Stale-val is an upper bound for this view because the scenario leaves `xp_imp_c50`, the one popularity input of the pop1 members, fresh; the lagged holdout lags it. I read the three together as "0.2 is defensible, 0.3 is not shown to be better".
- The teacher route is positive but does not reach +2 SE under private-like inputs (+0.0003 .. +0.0011 by sub-family), and it does not add to the member route on either lens (last row).

### 3.6 The six scored public pairs with the private-mix column [M]

Final-form files, my computation (`read3.py`); PUBLIC from the owner's readings [D `E_kd_predeclared.txt` 02:15Z].

| pair | val | pmix-pub | pmix-priv | PUBLIC |
|---|---|---|---|---|
| W16 - W8 | -0.00007 +- 0.00034 | +0.00014 +- 0.00042 | +0.00015 +- 0.00051 | -0.000003 |
| W8P20 - W8 | +0.00049 +- 0.00031 | +0.00063 +- 0.00039 | +0.00079 +- 0.00048 | +0.000523 |
| W8TO - W8 | -0.00009 +- 0.00043 | -0.00066 +- 0.00054 | -0.00104 +- 0.00067 | -0.000477 |
| S16 - W8 | +0.00008 +- 0.00034 | -0.00022 +- 0.00041 | -0.00038 +- 0.00049 | -0.000408 |
| C3 - G3 | +0.00069 +- 0.00040 | +0.00108 +- 0.00050 | +0.00130 +- 0.00062 | +0.000634 |
| W8 - C3 | +0.00064 +- 0.00055 | +0.00047 +- 0.00069 | +0.00036 +- 0.00085 | +0.001032 |

pmix-public reproduces the lead's column. Strictly it has the public sign in 5 of 5 non-zero pairs; the sixth public delta is -0.000003. pmix-private has the same signs and is larger in size on the three staleness pairs (W8P20, W8TO, C3 - G3), as it must be: private is colder. Older pairs fit the same picture less well: Q1a - G3 pmix-pub +0.00090 +- 0.00039 against public +0.00007; Cd, Pall, PF7x - G3 have the right sign on val and on pmix alike.

### 3.7 Screen of every stored 10-epoch family for the same signature [M]

`read5.py`: all 47 families in `nn_x/w5/preds` against their same-seed reference (hard-label base, kd05 or kd05w192), flag = pmix-priv >= +1 SE with val >= -1 SE and holdout <= 0. Six flags: PSEUDO-DIRECT (section 1.2), GRU students (pmix up but cold-positive feeds -0.0026: not the signature), and four arms that share one thing, the teacher_mean25 target. Read against twins that differ only in the teacher:

| students of teacher_mean25 against teacher_mean11 twins | n | val | M>=50 | coldpos | warmpos | pmix-priv | holdout |
|---|---|---|---|---|---|---|---|
| kd05t25 - kd05 (d128) | 4 | +0.00010 +- 0.00054 | +0.00093 | +0.00226 +- 0.00106 | -0.00048 | +0.00109 +- 0.00081 | -0.00032 +- 0.00031 |
| kd08t25 - kd08 (d128, kd_w 0.8) | 3 | +0.00053 +- 0.00051 | +0.00056 | +0.00576 +- 0.00098 | -0.00147 | +0.00162 +- 0.00076 | -0.00097 +- 0.00030 |
| kd05t25w192 - kd05w192 (d192) | 8 | +0.00044 +- 0.00048 | +0.00115 | +0.00170 +- 0.00093 | -0.00026 | +0.00044 +- 0.00073 | -0.00021 +- 0.00027 |
| all 12 kd_w 0.5 students of the first and third rows | 12 | +0.00056 +- 0.00044 | +0.00135 +- 0.00047 | +0.00282 +- 0.00084 | -0.00021 +- 0.00059 | +0.00123 +- 0.00067 (z 1.85) | -0.00005 +- 0.00025 |

- The 12-against-12 read is PASS-P. It is the popularity-light signature (cold up, warm down, long feeds up, holdout flat to down) produced by the target alone, with all 34 inputs in place. The teacher axis was closed on 11-14 as "one common mean function"; that holds for the level and not for this split.
- Confound: teacher_mean25 also contains 6 GRU members. The signature is not GRU-like (GRU students lose cold positives), so I attribute it to the 8 pop1 members [I].
- Final form, zero GPU (all 16 are fine-tuned; `read6.py`): the 8 teacher_mean25 students (T8) against the 8 teacher_mean11 students: val +0.00090 +- 0.00047, pmix-priv +0.00161 +- 0.00073; against the built W16: val +0.00039 +- 0.00032, pmix-priv +0.00067 +- 0.00048; train-only against W16: val +0.00010 +- 0.00032, holdout -0.00051 +- 0.00019. This is a split by a design factor, not a val-chosen subset, but it is one draw of 8 against 8.
- **The lever does not stack.** 0.8 z(T8) + 0.2 z(pop1kd x8) against 0.8 z(W16) + 0.2 z(pop1kd x8): val -0.00021 +- 0.00028, pmix-priv -0.00013 +- 0.00042, holdout -0.00053 +- 0.00016. Together with the flat weight scan of section 3.3 this says the popularity-light dose saturates on val at about +0.0003 .. +0.0006, however it is delivered (members at weight 0.1-0.5, or the teacher), while the holdout penalty keeps growing with the dose. W16 with its 8 teacher_mean25 students plus a 0.2 admixture is already on the plateau.
- W16 already has 8 of its 16 students on this teacher. What is open is the dose (item 7 of section 5).

### 3.8 [R] Row 301 re-read: every stored family as a 0.2 admixture to the 16-student pool [M]

Rule: `PREDECLARED.txt` addendum R1 (03:39:52Z, before `read8.py` was written). Statistic per family F: NDCG(0.8 z(W16) + 0.2 z(F pool)) - NDCG(z(W16)), train-only, selected checkpoints. 85 families with at least 2 members and all files on disk, discovered from the file names (49 of them with 4 or more members). FLAG = val >= 0 and pmix-priv >= +1 SE and holdout < 0; about 6.8 flags are expected by chance. Controls of R1.4 hold: the first run's numbers for pop1kd d128, pop1kd w192 and hard pop1 x8 are reproduced exactly, and the same-view families read |val| < 1 SE.

| family (members) | class | val | M>=50 | M<50 | coldpos | warmpos | pmix-priv | holdout | holdout >= Sep 13 | verdict / flag |
|---|---|---|---|---|---|---|---|---|---|---|
| pop1 students, d192 (4) | staleness | +0.00047 +- 0.00031 | +0.00087 | -0.00012 | +0.00267 | -0.00021 | +0.00050 +- 0.00048 (z 1.0) | -0.00046 +- 0.00022 | -0.00060 | PASS-P at the bar; FLAG |
| pop2, hard labels (4) | staleness | +0.00038 +- 0.00036 | +0.00077 | -0.00019 | +0.00313 | -0.00052 | +0.00026 +- 0.00055 (z 0.5) | -0.00068 +- 0.00023 | -0.00078 | FAIL (pmix) |
| pop1, hard labels (8) | staleness | +0.00031 +- 0.00035 | +0.00076 | -0.00034 | +0.00307 | -0.00068 | +0.00031 +- 0.00053 (z 0.6) | -0.00066 +- 0.00023 | -0.00066 | FAIL (pmix) |
| pop1 students, d128 (4) | staleness | +0.00008 +- 0.00032 | +0.00061 | -0.00069 | +0.00238 | -0.00074 | +0.00026 +- 0.00049 (z 0.5) | -0.00056 +- 0.00022 | -0.00062 | FAIL (pmix) |
| pop-free, hard labels (4) | staleness | -0.00003 +- 0.00035 | +0.00063 | -0.00098 | +0.00338 | -0.00128 | +0.00023 +- 0.00053 (z 0.4) | -0.00067 +- 0.00024 | -0.00099 | FAIL (val < 0) |
| hzcd re-cools, hard labels (4) | staleness | -0.00005 +- 0.00026 | -0.00004 | -0.00007 | -0.00070 | +0.00018 | +0.00010 +- 0.00040 | -0.00004 +- 0.00014 | -0.00011 | FAIL as an admixture; not the idea's form (its control m02ref x7 reads -0.00003 / +0.00018 / -0.00006: hard-label re-cools add nothing to a student pool) |
| kwwd05: weight decay 0.05 under KD (4) | not | +0.00018 +- 0.00021 | -0.00006 | +0.00053 | -0.00054 | +0.00059 | +0.00036 +- 0.00031 (z 1.2) | -0.00013 +- 0.00011 | -0.00016 | FLAG; pooled z -0.18 |
| kx05: cross-fitted teacher (4) | not | +0.00018 +- 0.00022 | +0.00027 | +0.00004 | -0.00043 | +0.00048 | +0.00039 +- 0.00034 (z 1.1) | -0.00027 +- 0.00012 | -0.00026 | FLAG; pooled z -1.02 |
| kdmixw192: mixed teacher (4) | not | +0.00033 +- 0.00022 | +0.00024 | +0.00046 | +0.00081 | +0.00026 | +0.00033 +- 0.00032 (z 1.0) | -0.00013 +- 0.00012 | -0.00004 | FLAG; pooled z +0.29 |
| trunc2: per-M-bucket lifts (4) | not | +0.00014 +- 0.00025 | +0.00015 | +0.00011 | -0.00100 | +0.00060 | +0.00040 +- 0.00039 (z 1.0) | -0.00014 +- 0.00014 | -0.00022 | FLAG; pooled z -0.30 |
| control: hard base (6) | same view | -0.00008 +- 0.00023 | +0.00009 | -0.00033 | -0.00062 | +0.00023 | +0.00005 +- 0.00036 | -0.00023 +- 0.00013 | -0.00019 | - |
| control: narrow students kd05t25 (4) | same view | +0.00017 +- 0.00021 | +0.00016 | +0.00019 | -0.00003 | +0.00039 | +0.00007 +- 0.00034 | -0.00008 +- 0.00011 | -0.00006 | - |
| mean of the 77 families outside the staleness family | | -0.00003 | | | -0.00094 | +0.00040 | -0.00009 | -0.00018 | -0.00026 | |
| mean of the 5 popularity-light families | | +0.00024 | | | +0.00293 | -0.00069 | +0.00031 | -0.00061 | -0.00073 | |

- 5 flags of 85 (expected by chance about 6.8). The four flags outside the staleness family are at the bar (z 1.0-1.2), none lifts cold-positive feeds, and none is close to the two-witness rule (largest pooled z over the 77 families: +1.2). The holdout-first screen of row 301 did not hide a non-staleness family.
- The in-period holdout dislikes admixtures in general: 90% of the 77 families read below 0 on it (mean -0.00018), with val at 0. A popularity-light family costs a further 0.0004 on the holdout and is the only kind that moves cold-positive feeds up (+0.0024 .. +0.0034; the largest value among the other 77 families is +0.0008).
- On the 16-student base and at train-only level, only the wide pop1 students pass the pmix bar, and at 1.0 SE. This repeats section 3.3: W16 already carries a popularity-light dose through its 8 teacher_mean25 students. The stale-val lens reads the same admixtures at z 2.0-2.6 (section 3.5).

### 3.9 [R] Resume-run checks of the first run's numbers [M]

**(a) Independent re-check** (`recheck.py`: z by `np.add.reduceat`, NDCG by `src/metrics.ndcg_per_feed`, pmix by `src/pmix.py`; no code shared with `vlib.py`). Six pool pairs, 22 numbers (hzcd selected: val, pmix-priv, coldpos, stale-val, holdout; Q2a8; the W8P20 form; W16 selected minus @last; the 12-against-12 teacher read; width): the largest difference to the first run's value is 7.6e-7 in the mean and 3.8e-9 in the SE.

**(b) Do the passes depend on the checkpoint kind?** Same pool pairs with every member at @last (`read11.py`, `read10.py`):

| read | selected: val / pmix-priv | @last: val / pmix-priv | cold-positive feeds, selected / @last |
|---|---|---|---|
| hzcd x4 vs m02ref x4 (section 3.1) | +0.00079 +- 0.00051 / +0.00105 +- 0.00076 | +0.00050 +- 0.00047 / +0.00048 +- 0.00072 | +0.00348 / +0.00452 |
| 0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8 | +0.00055 +- 0.00035 / +0.00124 +- 0.00052 | +0.00023 +- 0.00035 / +0.00131 +- 0.00055 | +0.00246 / +0.00285 |
| 0.8 z(W16) + 0.2 z(pop1 students x8) vs W16 | +0.00030 +- 0.00031 / +0.00051 +- 0.00048 | +0.00027 +- 0.00031 / +0.00060 +- 0.00048 | +0.00254 / +0.00220 |
| teacher_mean25 students x12 vs teacher_mean11 twins | +0.00056 +- 0.00044 / +0.00123 +- 0.00067 | +0.00032 +- 0.00044 / +0.00082 +- 0.00067 | +0.00282 / +0.00282 |
| width: W8 vs 8 narrow students | +0.00038 +- 0.00049 / -0.00008 +- 0.00074 | +0.00050 +- 0.00050 / -0.00016 +- 0.00074 | +0.00188 / +0.00153 |
| **aux weight 1.0: 7 re-cools vs 7 reference re-cools (hard labels)** | **+0.00070 +- 0.00028 / +0.00099 +- 0.00042** | **+0.00009 +- 0.00025 / +0.00007 +- 0.00035** | +0.00084 / +0.00013 |
| 3 blocks, hard labels: 2 vs 2 (10 epochs) | +0.00121 +- 0.00083 / +0.00256 +- 0.00122 | +0.00138 +- 0.00085 / +0.00352 +- 0.00126 | +0.00385 / +0.00606 |

Every staleness-family read keeps its sign and its cold-positive lift at @last. The three popularity-light reads still meet the pmix-private bar there (2.4, 1.3 and 1.2 SE); hzcd at @last misses it (0.7 SE) and passes on stale-val instead (2.25 SE, section 3.1). One read does not survive the change of checkpoint kind: aux weight 1.0.

**(c) CORRECTION to the first run: aux weight 1.0.** The first run called it "the one non-staleness arm with a pool val read above +2 SE". That holds only at holdout-selected checkpoints. By root group: roots 101-104 selected +0.00030 +- 0.00030, @last +0.00016 +- 0.00029; roots 105-107 +0.00112 +- 0.00039 and +0.00076 +- 0.00028; all 7: +0.00070 +- 0.00028 and +0.00009 +- 0.00025 (pool levels: arm 0.40053 / 0.40059, reference 0.39983 / 0.40050). The whole difference between the two columns is the reference: its selected 7-pool is 0.00068 +- 0.00030 below its own @last 7-pool, although no member moves by more than 0.0005 (mean member change +0.00009; selected steps 14,000-19,670 of 19,670). The 10-epoch pair reads val +0.00060 +- 0.00081 (M>=50 +0.00263, M<50 -0.00235), holdout -0.00010 +- 0.00047. So the hard-label evidence for a larger aux weight is one checkpoint-kind-dependent read. What stands is the direction at the other end of the dose under distillation, now at 7 seeds (`kwaux0`, finished seeds 951-955, 957, 958 against kd05w192 on the same seeds): val -0.00123 +- 0.00058 (M<50 -0.00263 +- 0.00107), cold-positive feeds -0.00573 +- 0.00114, pmix-priv -0.00088 +- 0.00088, holdout -0.00114 +- 0.00034. The first run quoted the 2-seed numbers (-0.00250 / -0.00135); at 7 seeds the effect is half that size and still negative on both witnesses. The aux task is worth about +0.0012 between weight 0 and 0.5; nothing here says that 1.0 is better than 0.5.

**(d) 3 blocks.** The hard-label 2-pool carries the cold-positive signature at both checkpoint kinds (table). Under distillation it does not: `kd05t25w192b3` x2 against `kd05t25w192` on the same seeds reads val -0.00095 +- 0.00071 (selected) / +0.00005 +- 0.00069 (@last), cold-positive feeds +0.00025 / +0.00012, holdout -0.00009 / -0.00019. The closure stands.

**(e) A general caution that follows from (c)** [I]: a pool of holdout-selected hard-label re-cools can sit 0.0007 below the pool of the same runs at @last without any member showing it, because early selected steps (here 14,000-19,000) have a less converged mean function. Reads of hard-label re-cool arms "on the selected checkpoints" inherit that noise on the reference side. Student re-cools do not show it (18 runs: pool -0.00016 +- 0.00022).

## 4. The reverse check: what was adopted with the holdout as a witness

Rule (`PREDECLARED.txt` item 4), A = adopted, B = alternative: CONFIRMED-OFF-HOLDOUT iff val >= 0, stale-val >= 0 where readable, pmix-priv >= -1 SE. IN DOUBT iff val < 0 or a staleness lens <= -1 SE while the holdout is > 0. CONTRADICTED iff val <= -2 SE.

| adopted setting | chosen on | A - B off the holdout [M] | holdout A - B [M] | verdict |
|---|---|---|---|---|
| checkpoint selection on the holdout (`--select_on holdout`), students (W16) | holdout by construction | pool S - L: val -0.00013 +- 0.00016, pmix-priv -0.00031 +- 0.00025 (z -1.2); final file with TO at @last: +0.00003 +- 0.00010 in favour of @last | +0.00011 +- 0.00009 | IN DOUBT by the letter, immaterial in size (9 of 16 runs differ, by the last 850 steps) |
| the same, hard-label re-cools (40 runs, 10 arms x 4 roots) | holdout | member-level S - L: val -0.00013 +- 0.00007, M<50 -0.00031 +- 0.00013, **coldpos -0.00147 +- 0.00014**, warmpos +0.00042 +- 0.00009, pmix-priv -0.00028 +- 0.00011, stale-val -0.00011 +- 0.00007 (coldpos -0.00143 +- 0.00015) | +0.00016 +- 0.00004 | IN DOUBT: selection on fresh counters buys warm positives with cold ones |
| the same, student re-cools of the K / L harness (15 finished runs: ref x12, ref2 x3; selected steps 12,000-19,670 of 19,670) | holdout | member-level S - L: val -0.00016 +- 0.00011, coldpos -0.00002 +- 0.00022, warmpos -0.00026 +- 0.00015, pmix-priv -0.00023 +- 0.00017; pool of 15: val +0.00009 +- 0.00026, pmix-priv +0.00004 +- 0.00041 | +0.00011 +- 0.00006 (member), +0.00013 +- 0.00014 (pool) | no cost visible under distillation |
| the same, hard-label 10-epoch members | holdout | base x6 pool: val +0.00002 +- 0.00029; pop1 x8: -0.00029 +- 0.00033; pop-free x4: -0.00061 +- 0.00048 (pmix-priv -0.00171 +- 0.00071) | +0.00025 / +0.00017 / +0.00013 | IN DOUBT for popularity-light members |
| final = 0.5 z(FT) + 0.5 z(TO) | nested val CV and two public reads (G3 TO-only -0.00047, W8TO -0.00048); the holdout has no FT branch | W16: w_ft 0 -0.00014 +- 0.00041 (pmix-priv -0.00071 +- 0.00064), 0.3 -0.00009, 0.7 -0.00012, 1.0 -0.00088 +- 0.00039 | n/a | CONFIRMED (not a holdout choice) |
| cooldown mix 0.02 | val and holdout (r6 stage 2) | m02 - c0 (mix 0.10), 7 roots: val +0.00108 +- 0.00059, coldpos +0.00539, pmix-priv +0.00089 +- 0.00087; m02 - m0: val +0.00042 +- 0.00042, coldpos +0.00308 +- 0.00087; m02 - m01: val +0.00024 +- 0.00045; m02 - m04: +0.00020 +- 0.00047 | +0.00137 +- 0.00033; +0.00011; **-0.00054 +- 0.00024**; +0.00009 | CONFIRMED-OFF-HOLDOUT (the holdout alone would have picked 0.01) |
| cooldown fraction 0.2 / 2-epoch re-cool length | val (cf4 "val -0.0012") | m02 - cf4 (40% cooldown): val +0.00308 +- 0.00070; m02 - e3: +0.00120 +- 0.00053; base - e20 pool: -0.00055 +- 0.00070 | +0.00049; -0.00009; +0.00023 | CONFIRMED on val. `cooldown_frac` itself was never varied in a full run (arm `E:kwcf35` exists, not run) |
| KD weight 0.5 | val (pool) | kd05 - kd08: val +0.00080 +- 0.00049, pmix-priv +0.00127 +- 0.00075; kd05 - kd03: +0.00108 +- 0.00057; kd05 - kd10: +0.00179 +- 0.00053 | -0.00028 +- 0.00028; +0.00007; +0.00058 | CONFIRMED-OFF-HOLDOUT (the holdout slightly prefers 0.8) |
| KD itself | val and holdout | kd05 x4 - base x4: val +0.00198 +- 0.00061, pmix-priv +0.00156 +- 0.00091 | +0.00021 +- 0.00036 | CONFIRMED |
| **width 192** | post hoc, on the holdout ("+0.00107 +- 0.00031 ... 3.4 se" [D 10:47Z]); the val rule missed by 0.00001 | W8 - 8 narrow students: val +0.00038 +- 0.00049, M>=50 +0.00071, coldpos +0.00188 +- 0.00095, pmix-pub +0.00018, **pmix-priv -0.00008 +- 0.00074**; kd05w192 - kd05: val +0.00049 +- 0.00056; kd05t25w192 - kd05t25: +0.00006 +- 0.00055 | +0.00083 +- 0.00028 (>= Sep 13: +0.00129 +- 0.00042) | CONFIRMED by the letter, weakly: val +0.8 SE, pmix-priv 0; public S16 - W8 = -0.00041 supports it at about 1 SE |
| the 34-input preset (8 popularity inputs) against popularity-lighter presets | val +0.0023 and holdout +0.0074 (2-epoch ablation) | students d192, pop1 view - full view (4-pools): val -0.00134 +- 0.00090 (M>=50 +0.00028 +- 0.00109, M<50 -0.00368 +- 0.00155), pmix-priv +0.00054 +- 0.00136; hard labels: -0.00140 +- 0.00100, pmix-priv +0.00004 +- 0.00151; 2-epoch stale-val: +0.00010 .. +0.00073 +- 0.00100 | -0.00636 +- 0.00061 | UNDECIDED (my rule has no word for it): the full preset is ahead on val by 1.5 SE and level at private mix and on stale-val. Lagged holdout: full view still +0.0028 +- 0.0008 ahead [D poplag_verify] |
| teacher = 11 full-view members (against teacher_mean25, which contains 8 pop1 members) | val bar (+0.0005) | kd05t25w192 x8 - kd05w192 x8: val +0.00044 +- 0.00048, M>=50 +0.00115 +- 0.00053, coldpos +0.00170 +- 0.00093, pmix-priv +0.00044 +- 0.00073 | -0.00021 +- 0.00027 | not a holdout choice; both teachers are in W16; the dose is item 7 of section 5 (section 3.7) |
| [R] **lr 2e-3 at 10 epochs** (against 1e-3) | val z_run 1.4 and holdout z 2.9 [D ledger G3: +0.00078 / +0.00067] | F7 x6 - F9 x6 (hard labels, mix 0.10, different seeds): val +0.00051 +- 0.00067, M>=50 +0.00073, **coldpos -0.00178 +- 0.00134, warmpos +0.00122 +- 0.00090**, pmix-pub -0.00001 +- 0.00084, pmix-priv +0.00002 +- 0.00103 | +0.00049 +- 0.00038 (7 against 6 members: +0.00077 +- 0.00037) | CONFIRMED by the letter (val >= 0, pmix-priv >= -1 SE); its gain sits on warm-positive feeds and is zero at the private mix. No stale-val files; never varied from scratch under KD |
| [R] width under hard labels (information) | - | F12 (d192) x7 - F7 (d128) x7: val -0.00038 +- 0.00063, coldpos +0.00363 +- 0.00115, warmpos -0.00205 +- 0.00086, pmix-priv -0.00074 +- 0.00096 | -0.00040 +- 0.00035 | under hard labels width was a cold-for-warm trade at no gain; under KD the wide students gain on both sides (row "width 192") |
| [R] width 192, CPU stale-val (information-grade source) | | W8 - 8 narrow students: +0.00075 +- 0.00050, private-weighted +0.00052 +- 0.00076; kd05w192 x4 - kd05 x4: +0.00064 +- 0.00057 | | holds under private-like inputs: the doubt of the row "width 192" is removed |
| [R] KD itself, CPU stale-val | | kd05 x4 - hard base x4: +0.00127 +- 0.00062 (z 2.1), coldpos +0.00263 +- 0.00127 | | holds under private-like inputs |
| [R] checkpoint selection at POOL level, ten hard-label 4-pools (the declared form, `PREDECLARED.txt` item 5) | holdout | mean of the ten S - L pool reads: val +0.00003 (6 positive, 3 negative, 1 identical; range -0.00025 .. +0.00039), stale-val +0.00012, pmix-priv +0.00013, **coldpos -0.00115 (8 of 9 negative), warmpos +0.00050 (8 of 9 positive), M<50 -0.00026 (9 of 9 negative), M>=50 +0.00022** | +0.00007 | no net loss on val, stale-val or pmix; a shift from short feeds and cold positives to long feeds and warm positives. The member-level row above ("IN DOUBT") is the same data without pooling. One 7-pool (m02ref, roots 101-107) reads val -0.00068 +- 0.00030 |
| [R] the same, student re-cools, 18 finished runs (ref x12, ref2 x6) | holdout | pool: val -0.00016 +- 0.00022, coldpos +0.00015 +- 0.00048, warmpos -0.00033 +- 0.00030, pmix-priv -0.00010 +- 0.00035; member level: val -0.00010 +- 0.00010, coldpos -0.00003 +- 0.00019, pmix-priv -0.00012 +- 0.00015 | pool +0.00019 +- 0.00013; member +0.00011 +- 0.00006 | no cost and no shift under distillation (replaces the 15-run row above) |
| [R] the same, W16, CPU stale-val of the 9 members whose selected step is not the last step | holdout | pool S - L: +0.00025 +- 0.00022 (coldpos -0.00015 +- 0.00037, warmpos +0.00043 +- 0.00031); member level +0.00023 +- 0.00012 | +0.00021 +- 0.00012 | no cost under private-like inputs |

Notes.
- Checkpoint selection. The 40 hard-label re-cool runs have stored stale files for both checkpoints, so this is a clean read: the selected checkpoint is better on the holdout by construction and worse where the positives are cold, on fresh val and on stale-val alike. For students the selected step is the last or the one before it (97,500 or 98,350 of 98,350 [D run logs]), and the pool does not move.
- hzcd is the arm where selection matters most: selected minus @last within the hz arm is coldpos -0.00223 +- 0.00060, stale-val +0.00004 +- 0.00034, holdout +0.00034 +- 0.00018. The holdout picks the hz checkpoints that look least like hz.
- Under distillation the picture is different (`read7.py`): the 15 finished student re-cools, which select between steps 12,000 and 19,670 like the hard-label ones, show no cold-positive cost of selection. Soft targets seem to remove the late-cooldown drift that selection reacted to [I]. Whether hz students behave like hzcd or like the reference students is unknown until the hz runs exist.
- Width. The mechanism gives no reason to distrust it (wide students are better on cold positives, not worse), but its evidence is the holdout plus one public pair. The `witness` agent's stale-val of the 16 wide and 8 narrow students will decide whether the entry needs a note.

Notes of the resume run [R].
- Selection. The answer to "do holdout-selected checkpoints lose on val / stale-val / pmix against @last, at pool level?" is no on all three for hard-label re-cools (ten 4-pools) and for students (W16, 18 re-cools), with stale-val read on stored GPU files for the hard-label arms and on CPU files for W16. What selection on the holdout does in hard-label re-cools is a change of composition: the holdout has only M >= 50 feeds and fresh counters, and the checkpoints it picks are better on long feeds and warm positives and worse on short feeds and cold positives, by about 0.0003 and 0.0012. The arms share roots 101-104, so "8 of 9" and "9 of 9" are not independent draws.
- Width. Three lenses agree off the holdout: val +0.0004 (0.8 SE), CPU stale-val +0.0008 (1.5 SE), public S16 - W8 = -0.00041 (in favour of wide); pmix-priv is at 0. The entry stands.
- lr. An optimiser setting is not staleness family, and the two-witness rule (val +, holdout +) keeps 2e-3. The footprint is still worth knowing: at the higher lr the pool is better where positives are warm and worse where they are cold. The harness arms `K:lr1` / `L:lr1` (`--lr 1e-3`, in `lane.sh`, listed as "not declared, not run") would test the cooldown lr only; the plateau lr needs from-scratch runs (`E_BASE` carries `--lr 2e-3`; an arm flag `--lr 1e-3` placed after it wins). I do not rank it: level at the private mix is not a reason to change.
- Adopted on both witnesses at more than 2 SE each, not re-read [D ledger section 1]: Muon (+0.0020 val / +0.0030 holdout), the WSD schedule, retrieval of 16 history items (+0.0019 / +0.0033), the listwise auxiliary task (+0.0024 / +0.0027), per-feed loss normalisation (+0.0020 / +0.0006), 10 epochs against 2 (pool +0.0040 / +0.0031). None is a popularity lever and none rests on the holdout alone.
- The in-period holdout reads almost any admixture to the 16-student pool as negative (section 3.8: 90% of 77 families, mean -0.00018), so "it lowers the holdout" was never specific evidence against a family; the family-specific part for the popularity-light view is about -0.0004.

## 5. Ranked re-open list

Expected private gains are my estimates [I] from the lenses of section 3, with the lagged-holdout penalty of the current pool as the ceiling (-0.0033 +- 0.0008 on 13,424 feeds, -0.0042 +- 0.0006 on 30,000 [D]). Paired private SE between near copies is about 0.0003, so only items 1-2 can be large enough to be seen on their own. Nothing below was launched by me.

Current recipe pieces [D `nn_x/w5/lane.sh`]: `E_BASE` = `--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500`; `HYG` = `--save_last --save_holdout --select_min_step 12000 --cliplog 50`; `B_BASE` = `--epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.02 --loss_norm feed --holdout 30000 --select_on holdout --eval_every 1000`. Harness run = `train.py --variant w5_{k|l}cd_<arm>_s<seed> --init nn_x/w5/ckpt/w5_e10_{kd05w192|kd05t25w192}_s<seed>@77500.pt $B_BASE $HYG --seed $((1000+seed)) --kd_file nn_x/w12/kd/out/{teacher_mean11|teacher_mean25}.npy --kd_w 0.5 <arm flag>`, 0.25 GPU-h (57 min wall at four jobs per GPU [D `w5_kcd_ref_s951.json`]). A from-scratch d192 run took 78-195 min wall under the same sharing [D], about 0.3-0.8 GPU-h.

| rank | arm | expected private gain [I] | P(passes its rule) [I] | GPU-h | exists | exact arm |
|---|---|---|---|---|---|---|
| 1 | **hz** | +0.0005 (0 .. +0.0012) | 0.35 | 4 (queued) + 3 for 16 x 3 fine-tunes on a pass | `K:hz` / `L:hz` in `lane.sh` (flag `--hzcd`); rule "ARM hz" 02:15Z | as queued. Add to the read: the @last column next to selected (hard-label precedent: stale-val 1.54 SE selected, 2.25 SE @last), with the reference at the same checkpoint kind |
| 2 | **hz2** (all 8 popularity inputs with test-like coverage on train rows, cooldown only) | +0.0008 (0 .. +0.0020) | 0.3 | 4 | view under construction by the `hz2` agent; `--hzcd` swaps only the 5 LUO columns, `data/feats_new/train/seq_cvlag_*` exists but failed coverage parity (zero share 0.65 vs 0.42-0.48 [D]) | harness arm `hz2` = the `K:` / `L:` command with the hz2 view flag once the view passes a label-free parity gate against test rows (the poplag gate G1-G5); declared before the run, same rule as hz |
| 3 | **popularity-light dose on W16** | +0.0003 (0 .. +0.0007) | 0.5 | 1 | 4 wide pop1 students trained (`nn_x/abl/ckpt/abl_e10_pop1kd05w192_s951-954.pt`); FT script `nn_x/abl/ftbag/ft_one_s3_r6.sh` (`MODE=abl`); d128 pop1 students already fine-tuned | fine-tune the 4 wide pop1 students (n2cpv3s3); file W16P20 = per branch 0.8 z(W16) + 0.2 z(pop1kd x8: d128 + w192), final 0.5 / 0.5, gate PASS; one SE-denominated public read against W16; no weight scan on public |
| 4 | **popularity-input dropout** | +0.0003 (-0.0002 .. +0.0008) | 0.2 | 4-6 | flag `--xt_drop q` in `nn_x/w5/train.py` (blanks the 8 `seq_cv*` / `it_hist*` inputs of a fraction q of train candidates, `model.py` l.115; snapshot cfg `xt_idcols` = columns 0-7 [M]); never run on the current recipe. **Trap [M, code read]:** with `--init` the model is rebuilt from the snapshot cfg (`train.py` l.350-353, cfg `xt_drop` 0.0), so `--xt_drop` on a `K:` / `L:` harness command changes nothing | (i) no code: from scratch, stage E, 8 seeds: `E_BASE + HYG + --kd_file .../teacher_mean11.npy --kd_w 0.5 --d 192 --xt_drop 0.15` (the arm flag comes last and overrides `--xt_drop 0` of `E_BASE`), against kd05w192 on the same seeds; or (ii) harness, 16 re-cools, in a COPY of train.py with `model.xt_drop = a.xt_drop` after the init block, `--xt_drop 0.3` (zero share of the LUO count on train rows about 0.13 -> 0.39 [I]). Rule = "RULE hz" |
| 5 | **@last next to selected for staleness-family reads** | 0 by itself; it protects items 1, 2, 4 from a false stop | - | 0 | `HYG` writes `@last` checkpoints and predictions for every run; `--hzcd` also writes `preds/holdout/nn_<v>_holdout_hz.npy` | for hz, hz2 and the dropout arm: report selected and @last side by side; if they disagree on the rule's verdict, say so and re-read with selection on the hz holdout file. Changes no existing file |
| 6 | **STALE-FT** | +0.0002 (0 .. +0.0006) | 0.2 | 3 | `nn_x/w5/finetune.py` (n2cpv3s3 recipe: `--extra n2c_pre_v3 --lr 5e-5 --decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton` [D RESULT_pop1x8]); counterfactual val inputs `nn_x/r6/stale_cf/both_*.npy`, `solvers/out/stale/cf/pre2both_*.npy` | a copy of the fine-tune script whose val TRAINING folds read the counterfactual columns (the `feat_path` switch of `stale_pred.py`); 16 W16 members x 3 seeds; read: out-of-fold stale-val and fresh val of the stale-FT pool against the standard FT pool, rule val (fresh, final file) >= 0 and stale-val >= +2 SE. Test rows keep their real inputs |
| 7 | **popularity-light teacher dose** | +0.0002 (0 .. +0.0005) | 0.15 | 4 + a CPU minute | teacher parts `nn_x/w12/kd/out/teacher_p1.npy` + `teacher_p2.npy` (logit sums of `abl_e10_pop1_s951-954` / `s955-958`, counts in the `.n` files [M]) and `teacher_mean11.npy`; `--kd_file` takes any such file | teacher_half = 0.5 teacher_mean11 + 0.5 (teacher_p1 + teacher_p2) / 8, written to the builder's own directory; harness arm = the `K:` command with `--kd_file <teacher_half>` on the 16 snapshots (both snapshot families, one teacher); reference = the ref re-cools; rule = "RULE hz" |
| 8 | **lag conditioning (HZCOND) on the hz2 view** | +0.0003 (speculative) | 0.15 | 4 + CPU build | `nn_x/abl/train.py` has `--xt_add`, `--kd_file`, `--kd_w`, `--init`; the lag of a train row under hz is `data/feats_new/train/hz_lag.npy` | one extra xt input = imposed lag in days (train rows: the hz2 lag; val 2 / 4; test 6-9), together with the hz2 view; only if hz2 passes its gate; 16 re-cools through `nn_x/abl/train.py --init ... --xt_add <lag column>` |
| 9 | **PSEUDO-DIRECT as a pool read** | +0.0004 (0 .. +0.0010) | 0.15 | 10-20 | code `nn_x/w11/pseudo/` (no `--kd_file` there: a port is needed); 2 hard-label runs stored | 4 seeds, hard labels first (2 more runs of `E:pseudo`, seeds 953 / 954, about 10 GPU-h) read at pool level under the two-witness rule; a KD port only on pooled z >= 1.6 |
| 10 | **aux weight 1.0 under distillation** (not staleness family: two-witness rule, the holdout stays in). [R] basis corrected, see 5.1 | +0.0001 (0 .. +0.0004), lowered [R] | 0.10 [R] | 4-6 | `E:kwaux1` in `lane.sh` (`--kd_file .../teacher_mean11.npy --kd_w 0.5 --d 192 --aux_pos 1.0`); hard-label records: `w5_cd_ap1_r101-107`, `w5_e10_ap1_s951/952` | `E:kwaux1`, seeds 951-958, against kd05w192 on the same seeds, the one-stage 8-seed rule of 21:26Z (val >= 0, holdout >= 0, pooled z >= 2). Basis: kwaux0 at 2 seeds -0.00250 +- 0.00073 val / -0.00135 +- 0.00043 holdout [M `read5.py`]; hard-label ap1 7-pool val +0.00070 +- 0.00028 |

Not ranked: more popularity-light wide students (pop1 seeds 955-958, or the pop2 / pop-free views: the `nn_x/abl/pop1/launch_e10.sh` command + `--kd_file .../teacher_mean11.npy --kd_w 0.5 --d 192`, with `--xt_add xp_ctr_c` for pop2 or no `--xt_add` for pop-free; 2-3 GPU-h per 4 runs). The three views correlate 0.986 [D], so this is variance reduction and waits for item 3.

What I would not spend on: the age-conditioned dropout as built (`--agedrop` exists only in `nn_x/solvers`, and item 4 gives the same test with a flag that exists), it_hist_hz from scratch (the plateau is overwritten by the cooldown; hz2 covers the cooldown), PDA / TIDE.

### 5.1 [R] Status at 03:50Z and checks of the arms the lead declared from this list

The lead wrote the re-open list into `nn_x/w5/logs/E_kd_predeclared.txt` at 03:35:48Z (hz first, then the fine-tunes of the wide pop1 students, ref2, kwxd15 / kwxd30, th, kwaux1; STALE-FT and HZCOND "prepared as code first") and added the arms to `nn_x/w5/lane.sh`. I checked what could be checked without a GPU. Nothing below was launched by me.

| arm | form in `lane.sh` [D] | check [M] | remark |
|---|---|---|---|
| hz | `K:hz\|L:hz) --hzcd` | 9 of 16 re-cools started by 03:50Z (`w5_kcd_hz_s952, 954, 956, 957, 958`, `w5_lcd_hz_s952, 954, 956, 958`), 2.6 min per 1,000 of 19,670 steps; no result file yet | read by the `witness` agent at about 07:15Z; the amendment of 03:35Z puts @last next to selected and re-reads once with selection on the hz holdout file if the two disagree |
| kwxd15, kwxd30 | `E:kwxd15) --kd_file .../teacher_mean11.npy --kd_w 0.5 --d 192 --xt_drop 0.15` (0.30) | snapshot config of `w5_e10_kd05w192_s951`: `xt_idcols` = columns 0-7 = `seq_cv_invd20, seq_cv_pop, seq_cv_norm, it_hist_{all,et1,et2,et3,et4}_luo`, exactly the 8 popularity inputs; `model.py` l.115-122 multiplies these raw columns by 0 for a random fraction q of the candidates of each training batch (train mode only: evaluation, holdout selection and prediction are untouched), and the within-feed rank and z inputs of `seq_cv_invd20` and `it_hist_all_luo` are computed after the blanking. The 8 inputs enter as log1p(max(x, 0)) with no standardisation at that stage (`data.py` l.194-195), so a value multiplied by 0 is the value of a zero count. Label-free check on stored inputs: among rows whose LUO count is 0, all 8 inputs are 0 on 100.0% of rows, on val (1,389,971 rows) and on test (2,904,806 rows); no NaN. Zero share of the LUO count: val 0.379, test 0.447; train 0.130 [D] becomes 0.26 at q 0.15 and 0.39 at q 0.30 | (i) `E_BASE` carries `--xt_drop 0`; `lane.sh` l.195 builds `$E_BASE $HYG --seed <s> <arm flags>`, so the arm flag is last and argparse keeps it: the first log line of a run should show `xt_drop=0.15`. I did not run it. (ii) The three `pre2_fe*` inputs, which the stale-val scenario also overrides, are not blanked. (iii) On test a stale counter is often too small, not zero; dropout teaches only the zero case, hz / hz2 teach the other. (iv) The trap stands: with `--init` the flag is ignored (model rebuilt from the snapshot config), so these arms cannot be moved to the K / L harness without a code change |
| th | `K:th\|L:th) --kd_file .../teacher_half_p1.npy` (overrides the stage teacher) | on 9,000,000 rows in three blocks the file equals 0.5 `teacher_mean11` + 0.5 (`teacher_p1` + `teacher_p2`) / 8 to 7.6e-6. `teacher_mean11` and `teacher_mean25` are per-feed-centred means of raw member logits (reproduced from the parts to 3.0e-5 and 4.7e-7 on 120,000 feeds); the parts are raw sums with a per-feed offset of about 140, so the half file is not centred. The teacher enters only through a per-feed softmax (`train.py` l.500, fp32, outside the autocast block), which removes a per-feed constant: after centring, the half file equals 0.5 `teacher_mean11` + 0.5 centred pop1 mean to 1.2e-5. Within-feed sd 1.01-1.05 (teacher_mean11 1.02-1.07, teacher_mean25 1.00-1.03, pop1 mean 1.00-1.04); within-feed correlation of teacher_mean11 with the pop1 mean 0.971-0.981; mean top-1 teacher probability 0.096-0.117 (teacher_mean11 0.101-0.122, teacher_mean25 0.096-0.116) | (i) Cosmetic: the log line "teacher logit sd" (`train.py` l.427, a global sd) will print about 5.3 for this file against 1.1 for the others; it is the offset, not a scale error. (ii) The dose differs by stage: K snapshots were trained with a pop1 share of 0 in the teacher, L snapshots with 0.32 (and 0.24 GRU); the arm sets 0.5 (and 0 GRU) for both. Read K and L separately as well as pooled. (iii) Basis: fresh val pmix-priv +0.00123 +- 0.00067 at selected, +0.00082 +- 0.00067 at @last; CPU stale-val +0.00034 +- 0.00044 for the 12 students (+0.0007 .. +0.0011 by sub-family). Positive on every lens and below +2 SE on each |
| W16P20x8 | file: per branch 0.8 z(W16) + 0.2 z(pop1 students x8), final 0.5 / 0.5; BUILD iff final val delta vs W16 >= 0 and gate PASS | train-only level of the declared form: val +0.00030 +- 0.00031 (@last +0.00027), pmix-priv +0.00051 +- 0.00048, CPU stale-val +0.00065 +- 0.00033 (private-weighted +0.00085 +- 0.00048), in-period holdout -0.00055 +- 0.00021; same-view controls on stale-val -0.00023 and -0.00076 (section 3.5). The 4 fine-tunes of the wide pop1 students wait for a GPU slot since 03:36Z (`nn_x/w12/kd/out/ftp1w_s95x.log`) | the build clause is on the final file; the train-only level predicts a val delta near +0.0003, so the clause can fail by noise (SE 0.0003). The W8-based twin scored +0.00052 on public |
| kwaux1 | `E:kwaux1) ... --d 192 --aux_pos 1.0` | basis corrected in section 3.9 (c): +0.00070 +- 0.00028 only at selected checkpoints, +0.00009 +- 0.00025 at @last; kwaux0 at 7 seeds -0.00123 +- 0.00058 val / -0.00114 +- 0.00034 holdout | expected gain +0.0001 (0 .. +0.0004), P(passes the 8-seed two-witness rule) about 0.1: the last arm to run, the first to drop |

What I would add to the read-outs of the staleness arms [I]: (a) the M >= 50 / M<50 split of the stale-val delta next to the val split (the popularity-light admixtures gain +0.0015 .. +0.0018 on long feeds and lose up to 0.0009 on short feeds under private-like inputs); (b) cold-positive and warm-positive feeds; (c) for kwxd, the dose line 0 / 0.15 / 0.30 on stale-val and on fresh val M<50, where a too large dose would show first.

## 6. Method, limits, and what I did not check

Method. Classification and rules were written first (`PREDECLARED.txt`). Reads use `vlib.py`: per-feed z of train-only predictions, member mean, numba NDCG with pessimistic ties (copy of `pc_common._ndcg`), pmix weights from `src/pmix.py`, cold = `it_hist_all_luo == 0`, holdout days from the train timestamps. Stale-val for verdicts = stored GPU files only.

Limits.
- pmix-private is not a second witness: same feeds as val. Every PASS-P here is "val is positive and its cold, long feeds carry the gain". pmix-public has the public sign in the 5 scored pairs whose public delta is not zero (section 3.6); pmix-private has the same signs [M], which says nothing yet about private.
- stale-val exists only for 2-epoch screens and hard-label re-cools. For the popularity-light view it changes only the three pre2 columns of those members (the 8 overridden counters are not inputs, and `xp_imp_c50` has no private-like counterfactual), so it flatters the view by construction.
- The hzcd reads are 4 roots, hard labels, the pre-KD recipe. Distillation may change the sign: the student's targets come from a teacher that saw fresh counters, which can help (the teacher's knowledge of popularity is passed on while the input is stale) or cancel the arm (the student is pulled back to the teacher).
- "10 of 12 pass" is three ideas on one val sample. I did not correct for multiplicity beyond saying so.
- Expected gains in section 5 are judgement, not measurement.
- My validity bar for the CPU stale path was too tight for fp32 against bf16 and I held to it; section 3.5 is information.

Where I think a pre-declared rule is wrong (applied as written anyway).
- "RULE hz" decides on holdout-selected checkpoints. For this family the selection step uses the biased split; the hard-label precedent reads 1.54 SE selected and 2.25 SE @last. The student reference re-cools show no selection effect, so the clause may be harmless here; I would still not let a selected-only FAIL close the arm if @last passes.
- Rule A' kept the wide pop1 students un-fine-tuned at public +0.00052 against a +0.0006 bar; the other written criterion (inverse-variance mean of val and public >= 2 SE) was met. The fine-tune costs 1 GPU-h.
- The task's PASS bar "val >= 0" turns on the sign of a number with SE 0.0003-0.0005.

Not checked.
- The hz and reference re-cools (not finished during this audit): no hz number here.
- Stale-val of 10-epoch hard-label members, of G3, C3, and of the narrow students (the `witness` agent's queue has the students).
- Lagged-holdout reads of anything (the `witness` agent's tool).
- PRODDIR / DOMAIN-DIAG, dwa, Ce, proj_mu, family weights, C06, C14, time-gap: taken from the record, not re-read.
- Fine-tuned (FT) branches of closed arms: only TO-level reads, except the files of section 3.2.
- Test-prediction diagnostics of any re-opened arm; the presub gate.
- Whether `--xt_drop` behaves as documented on the current recipe (read in the code, not run). The `--init` trap of item 4 is from the code and one snapshot cfg; I did not run a harness command to see it.
- Rows closed "in residual form only" (COVERAGE_MAP row 323, about 300 feature rows): their closing gates used val and holdout jointly on G3 residuals; I did not re-screen them one by one. The 16 CCB rows were screened (section 1).

### 6.1 [R] Resume run: limits, and what is still not checked

- No hz number: the re-cools were running, none had finished when this file was closed.
- CPU stale-val is information-grade by both agents' own rules (mine: one member 0.00008 outside a 0.0003 bar; the witness's b2: 1.04 SE against a 1 SE bar, its B' valid). The two tools give identical files. The scenario leaves `xp_imp_c50` fresh, so every stale-val read of a popularity-light member or admixture is an upper bound; the lagged holdout, which lags that input, reads the W8P20 form at +0.00024 +- 0.00031 on 13,424 feeds [D].
- The row-301 screen is one weight (0.2), one base (W16), train-only level; 36 of the 85 families have 2 or 3 members. Its flags are a chance-level count, not five findings.
- lr: F7 and F9 are different seeds and both pre-date the cooldown mix 0.02; the read is a pool comparison with SE 0.0007.
- The correction of the aux read (section 3.9 c) rests on 7 roots; I did not find why the reference's selected pool is that low beyond the early selected steps, and I did not test other 7-root arms for the same effect (no other arm has 7 roots).
- "8 of 9 arms" in the selection row share four plateaus.
- Still not checked: stale-val of G3 / C3 members and of hard-label 10-epoch members other than base x4; any fine-tuned (FT) branch of a closed arm; PRODDIR / DOMAIN-DIAG, dwa, Ce, proj_mu, C06, C14 and time-gap (taken from the record); the about 300 feature rows of row 323 one by one. `reports/FEATURES_REVIEW.md` was screened for rows where the holdout decided against a positive val read: it_hist_hz and hzcv only (both on the list), plus FR-5, a harness finding and not a closed idea: removing train-period count columns from the GBDT gate base reads val +0.0029 +- 0.0010 and holdout -0.0053 +- 0.0006, the same pattern inside the gate that closed feature rows.
- I did not run `--xt_drop`, the th arm or any GPU job; the two checks of section 5.1 are a code read plus label-free data checks.

### 6.2 [R] Rules on the books that still carry a holdout veto for the staleness family

For the lead; I changed none of these files.
- `docs/05_METHODOLOGY.md` l.54 and l.121 ("Popularity-light pools: decision on val; holdout as a -1 SE veto"), l.119 ("select members and checkpoints on the holdout"), l.125 ("m0, hzcd, a4u, cset, blocks3 net about 0": hzcd is re-opened). Lines 41-51 give every rule family a "holdout >= 0" clause, which is right for arms outside the staleness family.
- `reports/COVERAGE_MAP.md` rows 53, 78, 144, 160, 167, 217, 220, 252 still read CLOSED / REFUTED without a pointer to row 333 (the new policy).
- `nn_x/w12/kd/arm_read8.py` prints a pooled z that includes the in-period holdout; for hz, hz2, kwxd, th the declared rule excludes it: read the columns, not the pooled z.
- `src/pmix.py` (docstring: "report-only + veto ... never promotes"): consistent with the new policy, where pmix is information next to stale-val. The "+1 SE on pmix-private" bar of this audit is the task's bar, not a project rule.

## 7. Commands

```
cd <repo>
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
PY=.venv/bin/python; D=nn_x/w12/stale/veto
$PY $D/read1.py > $D/logs/read1.txt    # staleness-family items (hzcd, popularity-view pools, admixtures, family level)
$PY $D/read2.py > $D/logs/read2.txt    # reverse check (selection, FT weight, cooldown, KD weight, width) + information reads
$PY $D/read3.py > $D/logs/read3.txt    # final-form files, the six public pairs with pmix-private, PSEUDO-DIRECT, matched-control hzcd
$PY $D/read4.py > $D/logs/read4.txt    # teacher contrasts, TO@last file, CPU stale-val reads
$PY $D/read5.py > $D/logs/read5.txt    # screen of the 47 stored 10-epoch families
$PY $D/read6.py > $D/logs/read6.txt    # pools by teacher inside W16 (train-only and final form)
$PY $D/read7.py > $D/logs/read7.txt    # selected minus @last for the finished student re-cools of the harness
setsid nohup $PY $D/stale_cpu.py <variant> [<variant> ...] > $D/logs/<name>.log 2>&1 < /dev/null &   # CPU stale-val, 5-7 min per d192 pass
# resume run [R]
$PY $D/recheck.py    > $D/logs/recheck.txt     # independent re-check of six first-run pool reads (src/metrics.ndcg_per_feed, own z)
$PY $D/read8.py      > $D/logs/read8.txt       # row 301: 85 stored families as a 0.2 admixture to W16
$PY $D/read9.py      > $D/logs/read9.txt       # reverse check: lr 2e-3 vs 1e-3, hard-label width, GRU family
$PY $D/read10.py     > $D/logs/read10.txt      # aux weight basis with the seeds finished now; student re-cools selected vs @last; harness control
$PY $D/read11.py     > $D/logs/read11.txt      # checkpoint-kind robustness of the first run's passes
$PY $D/crosscheck.py > $D/logs/crosscheck.txt  # my CPU stale-val files against the witness cache
$PY $D/read4.py > $D/logs/read4.txt && $PY $D/mkblock.py   # CPU stale-val table of section 3.5
$PY $D/read12.py     > $D/logs/read12.txt      # CPU stale-val of the declared W16P20x8 form, teacher axis, controls
$PY $D/edit_report.py                          # applies the resume additions to this file (anchors asserted)
```

## 8. Files created

All under `nn_x/w12/stale/veto/` except this report.
- `PREDECLARED.txt` (rules, 02:23:09Z), `PROGRESS.md`.
- `vlib.py` (loaders, NDCG, pmix, slices), `read1.py` .. `read7.py`, `stale_cpu.py`, `mkblock.py` (writes the table of section 3.5 from `out/read4.json`), `chain_a2.sh`, `chain_b2.sh`, `chain_b3.sh` (waiters that start the next CPU list when the previous job of mine has exited; the comment line of `chain_b2.sh` mentions PSEUDO members that it does not run).
- `logs/read1.txt` .. `logs/read7.txt`, `out/read1.json` .. `out/read7.json` (every number of sections 3-4).
- `preds/cpu_<variant>_staleval.npy`, `preds/cpu_<variant>_val.npy`, `preds/stale_cpu.log`, `logs/stale_cpu_*.log`.
- `reports/STALE_1115/veto.md` (this file).
- [R] resume run: `recheck.py`, `read8.py` .. `read12.py`, `crosscheck.py`, `edit_report.py`; `logs/recheck.txt`, `logs/read8.txt` .. `logs/read12.txt`, `logs/crosscheck.txt`; `out/read8.json` .. `out/read12.json`; `out/sec0_new.md`, `out/sec35_add.md`, `out/sec38_39.md`, `out/sec4_rows.md`, `out/sec4_notes.md`, `out/sec51.md`, `out/sec6_add.md` (the text blocks that `edit_report.py` inserted); `out/veto_report_before_resume.md` (this report as the first run left it); two addenda in `PREDECLARED.txt`.

## 9. Compliance

- CPU only; `CUDA_VISIBLE_DEVICES=""` in every command; at most 12 threads per process; never more than two `stale_cpu.py` inference jobs at a time. The read scripts (`read1.py` .. `read7.py`, 20-60 s each, 6-12 threads) were run while those jobs were running, so for those seconds three processes of mine were active: I count the read scripts as light, and name it here in case the rule is read strictly.
- Processes stopped: one, my own first `stale_cpu.py` job (PID 4002144), by exact PID after reading `/proc/4002144/cmdline`. No pattern kill.
- No existing file edited, moved or overwritten. Imports of `nn_x/w5/data.py`, `nn_x/w5/model.py`, `src/pmix.py`, `src/common.py`, `src/metrics.py` and `eda/errors/segments/segdefs.py` ran with `PYTHONDONTWRITEBYTECODE=1`; `src/stack_val.py` was not imported (its numba cache would be written under `src/`), the z-score is re-implemented in `vlib.py`.
- No install, no network, no submission file. Val labels were used only to score stored predictions; no test label exists here; no later-snapshot input, no quarantined file.
- Rules written before the reads; one bar of mine was missed and is applied (section 3.5).
- [R] Resume run. Found five processes of the first run alive (two bash waiters with one `stale_cpu.py` each, one bash waiter). Stopped two of my own `stale_cpu.py` jobs by exact PID after reading `/proc/<pid>/cmdline` (4062266 at 03:33Z, 4089312 at 03:42Z): the members they still had to compute were already in the witness agent's cache. No pattern kill; no process of anybody else touched. From 03:33Z at most one inference job plus one read script of mine ran at a time; from 03:42Z no job of mine was left running. Read scripts used 4-8 threads. The witness agent's cache (`nn_x/w12/stale/witness/cache/staleval/`) was read, never written. `nn_x/w12/kd/out/teacher_*.npy` and `nn/feat_cache/{val,test}/*.npy` were opened read-only (memory-mapped) for the two label-free checks of section 5.1. One checkpoint was loaded on the CPU to print its config.
