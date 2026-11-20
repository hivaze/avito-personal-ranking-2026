# int_weak: inventory of everything that is NOT soundly closed (KEY = int_weak)

Written 2026-11-15 21:06-22:10 UTC by the internal auditor of workflow a workflow run. CPU only, read-only on every project file. Work directory `nn_x/w12/weak_1115/` (scripts, outputs in `out/`, row classification `rows.txt`, group coverage `groups.txt`).
Tags: [M] measured by the auditor today (command or file in `nn_x/w12/weak_1115/`), [R] record on disk (path given), [W] web source (URL), [I] judgement. "+-" = 1 SE paired per feed. val = 14,423 feeds; holdout = the 30,000 in-period feeds. Costs as given in the task: a 10-epoch student = 3.4 GPU-h, a 2-epoch re-cool = 0.7, a fine-tune = 1. "File gain" = forecast private gain of the final file if the idea worked.

## 0. Verdict in twelve lines

1. `reports/COVERAGE_MAP.md` holds 284 idea rows (lines 17-350; four header lines and two blank lines excluded; line 326 holds two rows) [M `rows.txt`]. Classified: SOUND 43, WEAK 113, NEVER RUN / PARKED / QUEUED 18, UNREAD 1, OPEN or RUNNING 5, adopted / protocol / statistic 55, superseded by a later row 49.
2. By the task's definition most closures are WEAK: 45 were closed in residual form or by a CPU / GBDT / frozen-trunk proxy, 24 in the hard-label era and never retried on distilled width-192 students, 18 by argument without a run, 11 on a 2-epoch screen, 6 on one or two seeds, 5 in the pre-Muon recipe, 3 on an in-sample or out-of-fold-only read, 1 for lack of power. Weak does not mean promising: the auditor's probability is at or below 0.03 for about 110 of the 137 non-sound rows [I].
3. The unread runs were read (section 2). Five stored arms, about 60 printed contrasts, no declared rule passes. kwr32 (retrieval depth 32) is closed by its rule [M]. hzt is closed by its rule without the stale-val column [M]. hz2nF and hznFS still lack stale-val files and stay undecided by their rule B [M].
4. Three records are wrong about what is on disk: the arms hz2n, hz2L (hz2Ln, hz2LnF), hzn, hznF do not exist; of the 11 harness arms only ONE run exists (K:kw08 seed 953); stage M was never run [M `ls nn_x/w5/preds`, `nn_x/w5/logs`].
5. One closed row has a zero-training re-opening reason: row 157 (inference-time domain token). It was closed on 11-13 as a "pure cold/warm trade" under the doctrine that such trades net zero; the public leaderboard contradicted that doctrine four times on 11-14/15 (popularity-light admixture). The veto audit could not re-read it ("files not located"). The files exist. Re-read today on the 12 hard-label members: production token on all feeds, 12-pool val +0.00031 +- 0.00035, pmix-public +0.00045 +- 0.00043, pmix-private +0.00052 +- 0.00052, cold-positive feeds +0.00333 +- 0.00073, warm-positive feeds -0.00099 +- 0.00046 [M, 3 forms looked at]. This is the signature of the popularity-light admixture at about half its size. It is information, not evidence (old members, same val feeds, post hoc slice). It needs inference only on the stored students.
6. One never-run dose is the open end of a monotone curve: weight decay 0. On record the pool falls monotonically with weight decay (0.02 / 0.05 / 0.10). Measured today: the mean pairwise correlation of members rises with it, val 0.9722 / 0.9810 / 0.9848, while the single member does not get worse (0.39918 / 0.39940 / 0.39894) [M `out/dose_diversity.txt`]. So weight decay costs the pool through diversity, and the dose below 0.02 was never run in any recipe (row 47).
7. Final-blend-only levers are bounded and belong to the sibling audits: the pool-size law prices 16 -> 32 members at +0.00008 [R ledger 21:07Z]; the w_ft curve is flat between 0.5 and 0.7 on val and at both mixes (within +-0.0004) [R `reports/MIX_1115/rulesA.md` l.561-622]. No blend-only item in this inventory has an expected gain above +0.0001 [I].
8. Inputs stale on test (special item iii): all three stale input families ALREADY use the unlabelled val and test logs (popularity counts over the history snapshots of all splits; co-visitation with every earlier val / test history; the encoder corpus with the earliest val / test snapshot per user) [R code, section 4.3]. What is left is one later-using co-visitation form with a val-test parity problem (P 0.02).
9. Public contradicted an internal witness in five places (section 4.4). Three are already acted on. Two leave weak closures behind: every "cold/warm trade" closure made before 11-15 (row 157 is the only one never re-read) and every fine-tune-stage closure judged on plain out-of-fold val (rows 97, 99, 103, 139, 303, 304).
10. Worth GPU tonight (list A, section 6): domain-token inference on the stored students (about 1 GPU-h), stale-val scoring of hz2nF / hznFS (0.5 GPU-h), the reads of the running arms (kwsb2, kwaux1, U20 seeds 953-954; no new GPU), then as the next filler arms weight decay 0 (4 seeds, 13.6 GPU-h) and width 224 seeds 955-958 (13.6 GPU-h) in place of the last 8 same-recipe students of rule G (forecast +0.00003 for 27 GPU-h).
11. Sum over everything in list A: probability that at least one item yields a file gain of +0.0002 or more on private about 0.25-0.3; expected total about +0.0001 [I]. Nothing in this inventory closes the 0.0033 gap to the public leader.
12. Not checked: no stale-val number exists for hznFS / hzt / hz2nF (needs model inference); the domain-token lever was not measured on any distilled student; the roughly 300 residual-form feature rows were not re-tested one by one.

## 1. Method and classes

Read in full: `reports/COVERAGE_MAP.md` (350 lines), `nn_x/w5/logs/E_kd_predeclared.txt` blocks 2026-11-14T20:16Z to 2026-11-15T21:07Z, `reports/CURRENT_STATUS.md` sections 1-6 and the log from 2026-11-15T02:19Z, `reports/STALE_1115/veto.md` sections 0-3.5 and 6, `reports/RECHECK_1114/SYNTHESIS.md` sections 4-5, `reports/MIX_1115/rulesA.md` (w_ft parts), `reports/LIT_1115/CRITIC.md` (cold-row parts), `reports/STALE_1115/staleft.md` section 0-1, `reports/EXPERIMENT_REGISTRY.md` "Running" and "Open levers", the feature builders `src/features_luo.py`, `feats_v2/seq_/feat_covis.py`, `docs/02_SOLUTION.md` 4.2-4.5.

Classes (one per row; file `nn_x/w12/weak_1115/rows.txt`):

| class | meaning | rows |
|---|---|---|
| A | adopted setting, protocol, statistic, information row | 55 |
| S | SOUND: powered multi-seed read under the current recipe, or a measured mechanism that does not depend on the recipe (coverage 0.1%, metric no-op, public reading) | 43 |
| W | WEAK, sub-codes below | 113 |
| N | never run, parked, queued or deferred | 18 |
| U | runs on disk, not read (before today) | 1 |
| O | open or running | 5 |
| X | superseded by a later row (status is carried by that row) | 49 |

WEAK sub-codes: px = residual form on a finished pool, or CPU / GBDT / MLP / frozen-trunk proxy (45); hl = hard-label era, multi-seed in its era, never retried on distilled width-192 students (24); arg = argument without a run (18); e2 = 2-epoch screen (11); s1 = one or two seeds (6); old = pre-Muon recipe (5); is = in-sample or out-of-fold-only witness (3); pw = lack of power (1).

Row-by-row classification (map line : class). Superseded rows name their successor.

- Optimisation and schedule: 17 A, 18 A, 19 W-hl, 20 N, 21 S, 22 A, 23 W-hl, 24 W-hl, 25 W-hl, 26 X (22, 23, 203), 27 N, 28 A, 29 W-old, 30 W-hl, 31 W-hl, 32 A, 33 O, 34 W-hl, 35 A, 36 W-hl, 37 N, 38 X (210), 39 X (211, 212), 40 N, 41 X (296).
- Regularisation: 46 S (dropout 0.2: kwd2 x4, kw256d2 x4; dropout 0 at 2 seeds and the cooldown doses dcd0 / dcd2 are in row 300 / 317), 47 N (doses 0.05 and 0.10 are closed at 8 / 4 seeds; dose 0 was never run), 48 X (24, 203), 49 X (145), 50 W-px, 51 X (184; kwh30 x4 closed on both witnesses), 52 X (339), 53 X (339).
- Architecture: 58 A, 59 A, 60 X (152; closed today by kwr32, section 2), 61 W-old, 62 X (300), 63 A, 64 X (300), 65 X (300; 6 heads closed at 4 seeds, ledger 20:46Z), 66 X (300), 67 X (331), 68 W-hl, 69 A.
- Inputs: 74 A, 75 A, 76 A, 77 S, 78 W-px, 79 A.
- Losses: 84 A, 85 A, 86 W-hl, 87 W-px, 88 W-px, 89 S, 90 W-old.
- Fine-tune: 95 A, 96 A, 97 W-is, 98 X (139), 99 W-px, 100 X (303, 343), 101 X (275), 102 W-old, 103 W-s1.
- Ensembling: 108 A, 109 A, 110 A, 111 S, 112 W-old, 113 A, 114 A, 115 A.
- Shift: 120 S, 121 S, 122 A, 123 S. Statistics 128-132: A.
- Do-not list: 137 W-arg, 138 W-hl, 139 W-is, 140 W-px, 141 W-px, 142 W-px, 143 W-hl, 144 X (337), 145 W-hl, 146 W-px, 147 W-hl, 148 W-hl, 149 S, 150 W-hl, 151 A, 152 W-e2.
- Tricks sweep: 157 W-arg, 158 S, 159 W-arg, 160 A, 161 X (185), 162 A, 163 S, 164 X (207), 165 X (183), 166 N, 167 N, 168 W-arg.
- Pre-training block: 173 W-arg, 174 W-arg, 175 W-arg, 176 X (291), 177 W-arg, 178 W-arg, 179-182 X (192-195), 183 S, 184 S, 185 S, 186 A, 187 X (205), 188 S, 189 W-e2, 190 A, 191 W-e2, 192-196 W-px, 197 A, 198 W-px, 199 X (260), 200 S, 201 W-px, 202 X (302), 203 W-hl, 204 W-px, 205 W-hl, 207 W-s1, 208 X (302), 209 W-hl, 210 W-hl, 211 W-s1, 212 W-hl, 213-216 W-e2, 217 X (302, 332), 218 W-px, 219 S, 220 X (333), 221 W-hl.
- External scan: 224 S, 225 W-px, 226 W-px, 227 W-pw, 228-230 W-px, 231 S, 232 W-px, 233 W-px, 234 N, 235 N, 236 X (254), 237 X (255), 238 X (256), 239 N, 240 X (296, 310), 241 A, 242 N, 243 O, 244 N, 245 N, 246 N, 247 N, 248 N, 249 X (271), 250 N, 251 X (258-273), 252 X (337), 253 X (220), 254-256 W-e2.
- CPU closure batch: 258-270 W-px, 271 S, 272 S, 273 W-px. Then 274 W-px, 275 S, 276 W-e2, 277 W-px, 278 S.
- Solver-power block: 280 S, 281 S, 282 W-arg, 283 W-px, 284 W-arg, 285 W-arg, 286 W-arg, 287 X (292), 288 X (295), 289 X (291), 290 W-arg, 291 W-hl, 292 S, 293 W-hl, 294 S, 295 W-s1.
- Distillation era: 296 A, 297 S, 298 A, 299 S, 300 W-s1 (width 256 is sound; 3 blocks, FFN x4, 2 history layers, dropout 0, 20 epochs rest on 2 seeds), 301 O (re-opened 21:07Z as rule G), 302 A, 303 W-is, 304 W-px, 305 W-px, 306 S, 307 W-px, 308 S, 309 X (331), 310 O, 311 X (316), 312 A, 313 A, 314 A, 315 A, 316 S, 317 N, 318 S, 319 W-arg, 320 A, 321 A, 323 A, 324 A, 325 O, 326 W-arg (second row on the line: X 341), 327 A, 328 X (341), 329 A, 330 A, 331 W-s1 (kdmix, GRU, width 224 at 4 seeds; kwT07, kwT15, kwsub at 2 seeds), 332 A, 333 A, 334 X (337), 335 A, 336 S, 337 W-arg (sound on fresh val at file level; the file was never read under stale inputs), 338 U, 339 S, 340 S, 341 S, 342 A, 343 S, 344 S, 345 S, 346 S, 347 S, 348 W-arg, 349 S, 350 X (227).

## 2. The unread runs, read today [M]

Tools: `nn_x/w12/kd/arm_read8.py`, `nn_x/w12/kd/hz_read.py`, `nn_x/w12/kd/hz_refs_read.py`, `nn_x/w12/kd/bucket_h256.py` (CPU, read-only). Outputs: `nn_x/w12/weak_1115/out/*.txt`. Arms looked at: 5 stored arms (hznFS, hzt, hz2nF, kw08, kwr32), the three re-cool arms against 2 declared references (ref, hz) and 4 reference pools, 2 checkpoint kinds, 6 columns each: about 60 contrasts. No correction for multiplicity is applied; no rule passes.

What exists on disk (train-only val / holdout / test predictions, both checkpoint kinds):

| arm | runs on disk | stale-val files | note |
|---|---|---|---|
| hznFS | 16 of 16 (K + L, seeds 951-958) | none | |
| hzt | 16 of 16 | none | |
| hz2nF | 16 of 16 | none | |
| hz2n, hz2Ln, hz2LnF, hzn, hznF, hz2L | 0 | - | lane lines exist (`nn_x/w5/lane.sh` l.154, 158), never launched |
| harness arms kw08, kw03, m05, m10, m0, e4, wd05, wd0, aux0, dcd0, dcd2 | 1 run in total (K:kw08 seed 953) | none | the other 175 runs were parked 11-15 03:35Z / 04:51Z and never re-queued |
| stage M (M:ps, M:kw08) | 0 | - | never run |
| kwr32 | 4 of 4 (finished 20:52-21:00Z) | - | not in the ledger at 21:07Z |

Train-only 16-pools, paired per feed. Decision column of the family = @last.

| contrast | val selected | val @last | val M<50 @last | pmix-public @last | pmix-private @last | in-period holdout @last (reported) |
|---|---|---|---|---|---|---|
| hznFS - ref | +0.00015 +- 0.00041 | +0.00001 +- 0.00039 | +0.00015 +- 0.00073 | +0.00041 +- 0.00052 | +0.00055 +- 0.00065 | -0.00044 +- 0.00022 |
| hznFS - hz | +0.00024 +- 0.00023 | -0.00023 +- 0.00021 (z -1.09) | -0.00026 +- 0.00037 | -0.00045 +- 0.00027 | -0.00073 +- 0.00036 | -0.00000 +- 0.00009 |
| hzt - ref | -0.00019 +- 0.00044 | -0.00011 +- 0.00043 | -0.00029 +- 0.00080 | +0.00048 +- 0.00056 | +0.00090 +- 0.00070 | -0.00101 +- 0.00025 |
| hzt - hz | -0.00011 +- 0.00032 | -0.00035 +- 0.00029 (z -1.21) | -0.00071 +- 0.00051 (z -1.4) | -0.00037 +- 0.00036 | -0.00038 +- 0.00044 | -0.00057 +- 0.00019 |
| hz2nF - ref | +0.00021 +- 0.00057 | +0.00057 +- 0.00057 (z +1.00) | +0.00054 +- 0.00104 | +0.00126 +- 0.00073 | +0.00185 +- 0.00091 (z +2.0) | -0.00275 +- 0.00039 |
| hz2nF - hz | +0.00030 +- 0.00048 | +0.00034 +- 0.00048 | +0.00012 +- 0.00086 | +0.00040 +- 0.00060 | +0.00058 +- 0.00075 | -0.00231 +- 0.00036 |
| hz2nF - ref2 / originals / all 48 | - | +0.00019 / +0.00031 / +0.00039 (SE 0.00058) | - | +0.00092 / +0.00079 / +0.00100 | +0.00153 / +0.00143 / +0.00161 (SE 0.0009) | -0.0026 .. -0.0029 |

Readings against the declared rules:
- hzt (rule of 08:57Z: replaces hz iff val >= +1.5 SE with stale-val >= -1 SE, or stale-val >= +1.5 SE with val >= -1 SE, and val M<50 >= -1 SE; decision at @last against hz): val -1.21 SE and M<50 -1.4 SE. Both branches of the rule need val >= -1 SE and the M<50 clause fails too. CLOSED by its rule without the stale-val column. With it the arm th (popularity-light teacher) is closed.
- hznFS (rule B of 07:42Z, as amended 08:16Z: replaces hz iff stale-val SVL >= +1.5 SE and val >= -1 SE and M<50 >= -1 SE at @last): val -1.09 SE, which misses the val clause by 0.09 SE; no stale-val file. At the selected checkpoints it is level or above hz (val +0.00024, holdout +0.00022 +- 0.00011). Reading: no sign that removing the cold-row tells improves on hz. Undecided by the letter (stale-val missing), no positive signal on any fresh column.
- hz2nF (same rule B): the val and M<50 clauses hold against hz (+0.7 SE, +0.1 SE); the stale-val clause cannot be read. It has the highest @last val level of all re-cool pools on disk (0.40188; ref 0.40130, hz 0.40154, ref2 0.40169, originals 0.40157 [M `out/hz_refs_read_unread.txt`]). The harness's own A/A draw (ref2 - ref) reads +0.00039 +- 0.00031 on the same column [R ledger 05:50Z], so +0.00057 is inside the null. The pmix-private reading of +2.0 SE is the largest of about 60 contrasts and is what one expects from the largest of 60. It keeps the full-dose cost on fresh in-period inputs (-0.0028). UNDECIDED; the missing piece is 32 inference passes for the stale-val files.
- kw08 (cooldown-only teacher weight 0.8), ONE snapshot pair: member val +0.00059 +- 0.00056, holdout +0.00096 +- 0.00032. A single member is not a decision input (ledger 21:46Z). It reproduces the from-scratch result at member level (kd08w192: member +0.00126 / +0.00078, 8-pool -0.00049 / -0.00029 [R ledger 07:49Z]): the member gain of a heavier teacher weight arises in the cooldown. Whether the diversity loss also arises there (pair correlation 0.990 against 0.972 from scratch [M `out/dose_diversity.txt`]) is the open question of stage M and needs the other 15 re-cools.
- kwr32 x4 (rule of 17:03Z: 4-pool vs kd05w192 on the same seeds; pooled z < 1 closes): val -0.00098 +- 0.00061 (z -1.62; M<50 -0.00244 +- 0.00114), holdout -0.00043 +- 0.00034 (z -1.27), pooled z -2.04; matched-N reference -0.00065 / -0.00042, pooled z -2.01; HOLD-SHORT -0.00062 +- 0.00032. Declared secondary, feeds at the history cap: val -0.00148 +- 0.00115 (4,176 feeds), holdout -0.00001 +- 0.00054 (11,601 feeds). CLOSED by its rule; no support from the cap bucket [M `out/arm_read8_kwr32.txt`, `out/bucket_h256_kwr32.txt`].

Descriptive table of the stored dose arms (train-only, selected checkpoints, seeds 951-954; `nn_x/w12/weak_1115/dose_diversity.py`, one look, arms named before the run) [M]:

| arm | val member | val 4-pool | val pair corr | holdout member | holdout 4-pool | holdout pair corr |
|---|---|---|---|---|---|---|
| kd05w192 (weight decay 0.02, reference) | 0.39918 | 0.40134 | 0.9722 | 0.32267 | 0.32408 | 0.9737 |
| kwwd05 (0.05) | 0.39940 | 0.40059 | 0.9810 | 0.32262 | 0.32355 | 0.9820 |
| kwwd10 (0.10) | 0.39894 | 0.39965 | 0.9848 | 0.32233 | 0.32317 | 0.9854 |
| kwd2 (dropout 0.2) | 0.39918 | 0.40066 | 0.9773 | 0.32238 | 0.32362 | 0.9785 |
| kd08w192 (teacher weight 0.8) | 0.40029 | 0.40042 | 0.9904 | 0.32338 | 0.32397 | 0.9907 |
| kwaux0 (no auxiliary loss) | 0.39712 | 0.39919 | 0.9683 | 0.32092 | 0.32248 | 0.9705 |
| kwh6 (6 heads) | 0.39877 | 0.40081 | 0.9710 | 0.32273 | 0.32464 | 0.9726 |
| kwr32 (retrieval 32) | 0.39902 | 0.40036 | 0.9705 | 0.32241 | 0.32365 | 0.9721 |
| kd05w224 (width 224) | 0.39935 | 0.40160 | 0.9693 | 0.32270 | 0.32424 | 0.9711 |

Reading: every closed regulariser above the reference (weight decay 0.05 / 0.10, dropout 0.2, teacher weight 0.8) raised the correlation between members and lowered the pool, while the single member stayed level or improved. The pool gain over the member is 0.00216 / 0.00119 / 0.00071 for weight decay 0.02 / 0.05 / 0.10. Width 224 is the only arm with a member at least as strong and a lower correlation than the reference on both witnesses.

## 3. Every WEAK, NEVER RUN, UNREAD and OPEN row (137 rows in 28 groups)

Each group lists its map lines; coverage of all 137 lines is checked in `nn_x/w12/weak_1115/groups.txt` [M]. "Test" = the decisive test and its cost. P = the auditor's probability that the test ends in a file gain of the stated size [I].

| # | map lines | idea | best number on record | why the closure is weak | decisive test, cost | file gain if it worked | P |
|---|---|---|---|---|---|---|---|
| T01 | 47 | weight decay below 0.02 (dose 0) | 8-pool, 0.05: val -0.00043 +- 0.00049, holdout -0.00063 +- 0.00027; 4-pool, 0.10: -0.00169 / -0.00091 [R ledger 03:48Z, 05:06Z]; pair corr 0.9722 / 0.9810 / 0.9848 [M] | never run in any recipe; the three doses on record are monotone in pool level and in member correlation | from-scratch arm `--wd 0`, 4 seeds 13.6 GPU-h, extension to 8 by the standing rule (27 in all) | +0.0003 | 0.12 |
| T02 | 157 | inference-time domain token (production token instead of the random-like token; soft mixes) | member level +0.00011 +- 0.00017, cold-positive +0.0026 [R `reports/sol_cpu_CPU3_aux_head_member_W6.md` section 5]; 4 older members M<50 +0.0007 .. +0.0013 each [R `reports/review_nnmath.md` NNMATH-8]; 12-pool val +0.00031 +- 0.00035, pmix-private +0.00052 +- 0.00052 [M] | closed as a "pure cold/warm trade" on fresh val before the evidence policy of 11-15; never read at the test mixes; never measured on a distilled student; the veto audit did not re-read it | inference only: 16 + 4 stored students, 2 token settings, val and test; about 1 GPU-h or 4-6 CPU-h; then a file-level read against FD16P20 | +0.00025 | 0.12 |
| T03 | 23, 24, 25, 317 | cooldown-phase doses under distillation: exploration mix 0 / 0.05 / 0.10, anneal target, cooldown twice as long, teacher weight 0.3 / 0.8, weight decay 0 / 0.05, auxiliary 0, cooldown dropout 0 / 0.2 | hard labels: mix 0 vs 0.02 val +0.00002; mix 0.04 -0.0008 / -0.0005; 40% cooldown -0.0012 [R map]; under distillation one run: kw08 member +0.00059 / +0.00096 [M] | hard-label reads only; the 11-arm harness declared 11-14 21:46Z ran 1 of 176 runs | 16 re-cools per arm = 11.2 GPU-h | +0.0003 | kw08 0.06, m05 0.05, wd0 0.04, others 0.02-0.03 |
| T04 | 19, 20, 27, 37, 40 | learning rate, cooldown fraction and floor, 5-7 epochs, warm-up, momentum, betas, clipping | lr 2e-3 vs 1e-3: +0.0008 val (z_run 1.4), +0.0007 holdout (z_run 2.9) [R map]; re-read val +0.00051 +- 0.00067, pmix-private +0.00002 +- 0.00103 [R `reports/STALE_1115/veto.md` 0.4] | hard-label adoption led by the holdout; never varied under distillation; lr above 2e-3 never run | harness arms K:lr1 / K:lr4 exist (11.2 GPU-h each); from scratch 13.6 per 4 seeds | +0.0002 | 0.04 each |
| T05 | 29, 30, 31, 34, 36 | weight averaging, EMA, plateau snapshots, longer plateau, 2-epoch seed families | plateau averaging: single +0.0004, 7-pool -0.0025 [R map l.30] | hard-label or older schedule; the diversity structure changed under distillation | 4-seed student arm with EMA, 13.6 GPU-h | +0.0002 | 0.02 |
| T06 | 33 | earlier cooldown (7-epoch schedule) as a cost lever | follows from two r6 findings; never run | cost lever only; cost is not binding | 4 seeds, 9.5 GPU-h | 0 (cheaper members) | 0.02 for any level gain |
| T07 | 50, 87, 88 | label smoothing, focal, LambdaLoss, RankNet, NeuralNDCG, ApproxNDCG, BCE, ListMLE | all within +-0.001 or worse [R map] | MLP proxy or frozen-trunk zoo; the hard-label term of a distilled student was never varied | 4-seed arm per loss, 13.6 GPU-h | +0.0002 | 0.02 |
| T08 | 61, 68, 90, 112 | FoX, QK-norm, MoE, PLE, TabM, propensity head, semantic IDs, EWMA; DIN; intent head, position tower; encoder term in the blend | all within +-0.0005 or negative [R map] | pre-Muon recipe, short runs | 4-seed arm each, 13.6 GPU-h | +0.0002 | 0.01-0.02 |
| T09 | 78, 242 | production-position prior of the item (ixh), lagged co-visitation (hzcv), ud, src_expo as model inputs | ixh: GBDT gate + one AdamW seed +0.0010 / +0.0002 [R map l.242] | adversarial GBDT verdicts; the neural form exists only for the hz view | trained-input student arm, 8 seeds 27 GPU-h | +0.0002 | 0.03 |
| T10 | 86, 209 | auxiliary weight 1.0; aux temperature 30; top-20 Plackett-Luce aux | hard labels: 7-pool val +0.00070 +- 0.00028 at selected, +0.00009 +- 0.00025 at @last [R veto.md 0.3]; under distillation aux 0 costs -0.00135 / -0.00128 [R ledger 03:48Z] | hard-label; the dose 1.0 under distillation is RUNNING (kwaux1 x4, read due tonight) | running; extension 13.6 GPU-h | +0.0003 | 0.05 |
| T11 | 97, 99, 102, 103, 139, 303, 304 | fine-tune stage: shorter fine-tune, refit on full val, test pseudo-labels, LP-FT / SAM, recipe retune, redesign (anchor, low-dimensional, conditional weight), soft-target fine-tune | retune CONFIRM +0.00055 (z 1.3); soft-target pilots -0.00036 +- 0.00031 and +0.00002 +- 0.00044; refit -0.00025 +- 0.00018 [R map] | the only witness is plain out-of-fold val, which reads the whole fine-tune half at +0.00009 where public reads +0.00048 [R CURRENT_STATUS section 1]; val has 8% cold feeds against 21-33% on test [R rulesA.md l.655] | no offline witness; a public pair per variant (16 fine-tunes = 16 GPU-h, 1 slot, paired SE 0.0005) | +0.0002 | 0.04 each |
| T12 | 137, 286 | language-model backbones as rankers | -0.0038 .. -0.0053 at matched 150K feeds [R map] | no per-size retuning, full-data scaling untested | one-pass read 4-6 GPU-h | +0.0006 as a family | 0.01 |
| T13 | 138, 145, 148, 203, 205, 212, 221, 293 | hard-label cooldown arms: history cap, recency, RHO-loss, feed-normalised aux, val replay, aux dose, mix 0.01 / 0.04, R-Drop, SAM, cooldown batch, format parity, exploration posterior | SAM 4-pool: val +0.00033 +- 0.00045, stale-val +0.00068 +- 0.00044, holdout +0.00025 +- 0.00023 [R veto.md 1.2]; the rest null or negative | 4-root hard-label re-cools; none retried on the 16 distilled snapshots | 16 re-cools per arm, 11.2 GPU-h (SAM 22.4) | +0.0003 | SAM 0.05, others 0.02 |
| T14 | 143, 147 | training-only rank-from-end offset on production rows | member val -0.0005 +- 0.0005, pool -0.0003 / -0.0009 [R map] | hard-label 10-epoch pairs | 4 seeds, 13.6 GPU-h | +0.0002 | 0.01 |
| T15 | 150, 152, 189, 191, 245 | more inputs in the ranker (4 within-feed rank / z channels), same-category retrieval, removal of ctx_hn + ctx_hs, tree-leaf or tree-score inputs | xrel_more -0.0019 val at 10 epochs; retr_cat +0.00046 +- 0.00049 / 0.0 at 2 epochs [R map] | hard labels; "trained inputs under distillation" (bet 5b of the recheck synthesis) was never run and was closed with the capacity class by argument | one student arm, 8 seeds 27 GPU-h | +0.0003 | 0.03 |
| T16 | 140-142, 146, 192-196, 198, 201, 204, 218, 225, 226, 228-230, 232, 233, 258-270, 273, 274, 277, 283, 305, 307 (39 rows) | features and cues tested as a residual on a finished pool or through a CPU proxy | best single reads: it_tr_coec +0.00060 +- 0.00033, cos_u GBM +0.00073 +- 0.00058, proj_mu +0.00045 +- 0.00022 val / -0.00004 holdout, item trend +0.00031 +- 0.00038 [R map]; neighbour-completed co-visitation on cold rows: likelihood ratio p 0.036 / 0.0007 [R `reports/LIT_1115/CRITIC.md` 2.3] | a block worth +0.0085 as a trained input reads +0.0009 / +0.0022 on the holdout and nothing on val in residual form [R map l.323]; none was ever a trained input of a student | a bundle arm: build 1-2 CPU-days with parity checks, then 8 students 27 GPU-h; the teacher has not seen the input, which halves its pull at teacher weight 0.5 | +0.0003 | bundle 0.05; single rows 0.01-0.02 |
| T17 | 159, 168, 282, 284, 285, 290, 319, 326 | dropped by argument: cross-user labelled kNN; 19 tricks (snapshot ensembles, mean teacher, WiSE-FT, last-layer refit, ...); upcycled MoE; all labels in the cooldown; pseudo-feeds in the fine-tune; in-ranker self-supervision; clean sub-feed sampling; the recheck's dropped list | WiSE-FT: "FT branch saturated; pool ceiling about +0.0001" [R `reports/TRICKS_SWEEP_1113.md` l.462]; sub-feed logits equal full-feed logits (KL 0.002-0.005 nats) [R SYNTHESIS section 5] | no run; several reasons rest on measurements, several on analogy | WiSE-FT: 16 fine-tunes with stored weights 16 GPU-h; clean sub-feed arm 27 GPU-h + a day of code | +0.0002 | 0.02-0.04 each |
| T18 | 173, 174, 175, 177, 178, 291 | pre-training programmes; encoder dose | encoder channel +0.00077 val at 10 epochs against a +0.0015 trigger [R map l.291] | closed by ratio arguments and triggers; the encoder dose read is a hard-label pair | 10-20 GPU-h per programme | +0.0003 | 0.01 |
| T19 | 166, 167 | queued or deferred tricks never run: retrieval-probability dropout, deep supervision, anchor replay, gradient balancing of the aux loss, relative popularity, staleness indicator input (HZCOND), encoder variants, cross-feed negatives | priors 0.04-0.07 in the sweep [R map l.166]; the gradient pre-gate reads near-orthogonal (row 272) | never run; HZCOND was "prepared as code first" on 11-15 and not declared | HZCOND: CPU build + 16 re-cools 11.2 GPU-h; the others 9-14 GPU-h each | +0.0003 | HZCOND 0.03, others 0.02 |
| T20 | 300, 331 | under distillation at 2 seeds: 3 blocks, FFN x4, 2 history layers, dropout 0, 20 epochs, one-sided temperature, sub-feed augmentation as built; at 4 seeds: width 224 | width 224: val +0.00026 +- 0.00058, holdout +0.00016 +- 0.00032, pooled z +0.68; matched-N +0.00059 / +0.00017, pooled z +1.38 [R ledger 23:00Z]; the 2-seed arms all below pooled z 1 | 2-seed member screens keep 11-22% of a true +0.0005 [R map l.315]; width 224 fell between "closed below 1" and "extend at 1.6" of the later rule and was read by an older bar | width 224: seeds 955-958, 13.6 GPU-h, the 8-seed rule; each 2-seed arm: 6 more seeds 20 GPU-h | +0.0002 | width 224 0.10; others 0.02 |
| T21 | 207, 210, 211 | late dropout; 15 / 20 / 30 epochs; 4x batch | e20 member -0.0008 / -0.0006; bs4 -0.0026 / -0.0021 [R map] | hard labels; one pair or 2 seeds for two of the three; under distillation only kw_e20 at 2 seeds | 4 seeds, 13.6-27 GPU-h | +0.0002 | 0.02 |
| T22 | 295 | pseudo-feeds built from histories (PSEUDO-DIRECT) | member val +0.00108 +- 0.00072, holdout +0.00024; 2-pool val +0.00072 / +0.00133 (@last), pmix-private +0.0013 .. +0.0016 [R veto.md 1.2] | 2 seeds, hard labels; stopped on a holdout clause; item 9 of the veto re-open list, never declared | after the U20 read: 8 students 27 GPU-h | +0.0004 | 0.05 |
| T23 | 213-216, 254-256, 276 | member families read at 2 epochs: domain re-weighting, category-factorised head, surface-restricted history, early-fusion transformer, examination offset, CE + BCE, cross branch, weekday / hour inputs | all weaker than the base or exact near copies [R map] | 2-epoch hard-label screens | 4 seeds, 13.6 GPU-h each | +0.0002 | 0.01-0.02 |
| T24 | 227 | history extended from the user's earlier snapshots (beyond the 256-event cap) | replay numerator +0.00117 +- 0.00082 (z 1.43) and +0.00151 +- 0.00093 (z 1.62); val same sign; pooled z 2.0-2.4 [R ledger 18:45Z] | closed for lack of power; the mechanism is probably real | model change (384 positions) + data build + 8 students at about 1.5x step cost = 41 GPU-h; train rows cannot be built like test rows (gap 2.8 d against 14.2 d) | +0.00025 | 0.10 |
| T25 | 234, 235, 239, 244, 246, 247, 248, 250 | external-scan items never run: additive item click rate as a full-view input, stage 2 on out-of-fold scores of the exploration feeds, example-age input, truncated CE, down-weighting negatives below the last click, event-field dropout, token MLP, user-half members | priors 0.01-0.03 in the scan [R map] | never run | CPU gates of 1-2 h exist for two of them; arms 13.6 GPU-h | +0.0002 | 0.01-0.03 |
| T26 | 243, 301, 310, 325 | open or running: similarity bias 2 (kwsb2 x4), unlabelled-fold distillation (U20 seeds 953-954), stage M (never run), pool growth to 32 (rule G) | U20 at 2 seeds: B - A val +0.00158 +- 0.00071, holdout -0.00017 +- 0.00039 [R ledger 03:13Z]; growth forecast +0.00008 [R ledger 21:07Z] | running; rules written | none new; reads tonight | +0.0003 (U20, kwsb2); +0.00008 (growth) | U20 GO 0.07; kwsb2 0.03; growth non-inferior 0.9 |
| T27 | 337, 338 | stale-inputs cooldown at file level; the hz2n family | hz: 16-pool stale-val +0.0010 .. +0.0012 (z 2.3-2.7), file val -0.00047 +- 0.00035, pmix-private -0.00072 +- 0.00060 [R ledger 08:56Z, 12:51Z]; hz2nF @last val +0.00057 +- 0.00057, pmix-private +0.00185 +- 0.00091 [M] | the file was judged on fresh val and on re-weighted fresh val only; no fine-tuned prediction exists under stale inputs, so the witness that passed the pool was never applied to the file; hz2nF / hznFS lack the stale-val column | stale-val scoring of hznFS + hz2nF: 0.5 GPU-h inference; file-level stale read of hz: 16 fine-tunes through the stale-FT script = 16 GPU-h | +0.0003 private only | hz file 0.08; hz2nF chain 0.06 |
| T28 | 348 | graph networks; experts inside the network | in-network experts once: val +0.00027 +- 0.00047 (old optimiser, 2 epochs); cold-row neighbour scores detectable but small [R `reports/LIT_1115/CRITIC.md`] | never trained; closed by audit, literature and critic | 12-31 GPU-h + 1-1.5 days of parity work | +0.0002 | 0.05 |

## 4. Special attention

### 4.1 Changes of the final blend only, or members already on disk

| item | what | record | status | P of a private gain >= +0.0001 [I] |
|---|---|---|---|---|
| Union of the 32 wide students on disk (16 held-out + 16 all-data) + 8 wide popularity-light | key U32P20w, declared by the lead at 21:07Z | pool-size law: 16 -> 32 about +0.00008 per branch [R] | OPEN (lead); a non-inferiority file | 0.4 |
| Train-only branch from EVERY same-view distilled student on disk (16 originals, 32 ref / ref2 re-cools, 16 all-data, the 4-seed arms that read level) with the fine-tuned branch of the 16 all-data students | zero training | fine-tuning only 8 members and pooling all train-only members gave -0.00001 +- 0.00023 against fine-tuning every member [R `reports/RECHECK_1114/ft_redesign.md` l.197]; law 16 -> 100 about +0.00014 on that branch | never built; sibling audit `nn_x/w12/ens_1115/` | 0.35 |
| w_ft above 0.5 | constant 0.6 / 0.7 | FD16 + 4: 0.6 val -0.00011 +- 0.00016, pmix-private -0.00007 +- 0.00024; 0.7 val -0.00038 +- 0.00024, pmix-private -0.00002 +- 0.00036; other three compositions within +-0.0004 with both signs [R rulesA.md l.561-622] | closed on val; "val cannot say more" (l.622); the curve is flat, so the stake is small whatever public says | 0.2 |
| Domain token on the train-only branch (T02) | inference only | section 0 item 5 | NOT re-read before today | 0.2 |
| All-data hz file FDX16P20:hz | buildable from disk (16 X:hz members are fine-tuned) | file val -0.00047 +- 0.00035; pool stale-val +0.0010 [R] | closed as a recipe on fresh val; a public slot would be a reading at 9.3 days of staleness with a paired SE of about 0.0005 | 0.2 |
| 12 wide popularity-light students instead of 4 | variance-reduced twin | -0.00016 +- 0.00018 [R ledger 16:16Z] | SOUND as "not better" | 0.1 |
| Hard-label families, GRU / 6-head / width-224 students as an admixture | 85 stored families as a 0.2 admixture to W16 | 5 flags against 6.8 expected by chance; no two-witness pass [R veto.md 3.8] | SOUND | 0.05 |

Statement that goes with every row: between near-copy files the paired public SE is 0.0004-0.0007; a blend-only file can land above 0.3950094 by noise with probability about 0.4 and that would not be progress on private.

### 4.2 Closed under hard labels, interacting with distillation

| lever | hard-label closure | under distillation | status |
|---|---|---|---|
| weight decay | never varied | 0.05 x8 and 0.10 x4 closed on both witnesses; dose 0 never run; diversity mechanism measured [M] | T01, list A |
| dropout | 0.1 never varied at 10 epochs | 0 at 2 seeds (member -0.0006 / -0.0002), 0.2 at 4 seeds (pooled z -1.88): interior optimum | SOUND for 0.2; dose 0 weak, low prior |
| exploration mix in the cooldown | 0.02 adopted on 7 roots; 0.04 negative | never run (harness m0 / m05 / m10) | T03, 11.2 GPU-h per dose |
| teacher weight by phase | - | from scratch 0.3 / 0.8 closed at pool level; cooldown-only 0.8: one run [M] | T03 (kw08), stage M never run |
| auxiliary loss weight | 1.0: selected +0.00070, @last +0.00009 | 0: -0.00135 / -0.00128 (8 seeds); 1.0 running | read tonight |
| R-Drop, SAM | val null, holdout +0.0004 .. +0.0006; SAM stale-val +0.00068 +- 0.00044 | never run | T13, 22.4 GPU-h, P 0.05 |
| weight averaging, EMA | pool -0.0025 through lost diversity | never run; the same diversity mechanism as weight decay predicts a loss | T05, not recommended |
| history-event dropout | TTA negative | kwh30 x4: -0.00098 / -0.00066 | SOUND |
| sub-feed augmentation | - | as built: -0.0003 .. -0.0005 at 2 seeds with a format mismatch (1.88 positives per sub-feed against 1.22) | weak; clean form dropped by a measured argument (KL 0.002-0.005 nats) |
| loss form | zoo on a frozen trunk | teacher temperature one-sided at 2 seeds; pairwise or top-weighted distillation never run | T07 / T20, P 0.02 |
| epochs, batch | 15 / 20 / 30 epochs and 4x batch worse | 20 epochs at 2 seeds: member -0.0008 / -0.0004 | weak, low prior |
| learning rate | 2e-3 adopted on the holdout | never varied | T04 |
| history length | cap 256 = data cap | extension undecided for lack of power | T24 |
| more inputs | xrel_more -0.0019 | never run | T15 / T16 |

The class stop rule of 11-14 22:14Z ("closed under hard labels, retried under distillation" closes if no arm of wave 1, kd08w192, kwd2, kwsb2 reads pooled z >= 1.6) will fire when kwsb2 is read. It was written for from-scratch capacity and regularisation arms. It does not cover the cooldown-phase doses (T03), the dose below the reference for weight decay (T01), or width 224 at 8 seeds (T20), none of which was run.

### 4.3 Inputs that are stale on test: what the unlabelled val and test logs already supply, with the leak analysis

| input family | built from | uses val / test logs today | recomputable without a user's later snapshots? | leak analysis |
|---|---|---|---|---|
| `it_hist_*_luo` (5 train-stage inputs, 14 fine-tune extras) | occurrences of the item in ALL history snapshots of train, val and test [R `src/features_luo.py` l.3-6] | yes, including OTHER users' later snapshots | nothing left to add | same-user later snapshots are subtracted in every split; user 0 is never linked. Staleness is a density effect: 43,018 val + test snapshots against 3.15 million train snapshots |
| `seq_cv_*` (3 inputs) | all train histories + every val / test history with a timestamp BEFORE the row's own feed [R `feats_v2/seq_/feat_covis.py` l.4-7] | yes, causal | one form is left: other users' LATER val / test histories | allowed by the leak rule if same-user sequences stay skipped AND user-0 later histories are excluded (a pooled bucket can hold the same person). Problem: parity. A val row has all 28,595 test histories in its future, a late private row has almost none, so the fine-tune would train on denser inputs than private rows carry. The later-using exposure form read no signal [R `reports/FRESH_1115/`]. NEVER RUN for co-visitation; P 0.02 |
| `pre2_*` encoder scores (6 inputs) | corpus = latest train snapshot + EARLIEST val / test snapshot per user, nothing after a val / test feed of the same user [R `docs/02_SOLUTION.md` 4.3] | yes | adding later snapshots of other users needs encoders cross-fitted by user (several GPU-h each) | safe only with cross-fitting; the whole channel is worth +0.0008 val at 10 epochs, so a few percent more corpus is worth about 0; P 0.01 |
| `xp_imp_c50` (popularity-light students) | train impressions only | no | yes: appearances in other users' val / test feeds | tested as-of and later-using on 11-15: click lift given the pool 1.008 +- 0.021 [R ledger 16:16Z]. SOUND closure |
| `ctx_*` (20 inputs) | own feed and own history | not stale | - | - |
| fine-tune labels | val labels through user-grouped folds | - | - | the only stage with out-of-period labels; STALE-FT closed on val and stale-val |

Conclusion: no stale input remains that the unlabelled logs could refresh at a parity-safe cost. The staleness loss (-0.0017 +- 0.0005 at file level under private-like inputs [R `reports/STALE_1115/staleft.md` 0.4]) is lost information. The remaining levers act on the model side: the popularity-light admixture (adopted), the domain token (T02), the file-level stale read of the hz family (T27).

### 4.4 Where the public leaderboard contradicted the internal witness

| # | contrast | internal witness | public | what follows for closed rows |
|---|---|---|---|---|
| 1 | popularity-light admixture (C3 - G3, W8P20 - W8, FD16P20 - W16) | in-period holdout -0.0005 .. -0.0007 | +0.00063, +0.00052, +0.00041 | acted on: veto audit, evidence policy. Left over: row 157, closed on the same "cold/warm trades net zero" doctrine and never re-read (T02) |
| 2 | fine-tune half (W8 - W8TO, G3 pair) | plain out-of-fold val +0.00009 | +0.00048, +0.00047 | pmix-public reads it right (+0.00066). Every fine-tune-stage closure on plain val is weak (T11). The weight itself is second order (section 4.1) |
| 3 | M-rule | in-sample val +0.0003 .. +0.0007 | -0.00015 | acted on: nested reads only |
| 4 | S16 against W8 | val +0.00008 for S16 | -0.00041 | holdout was right; two-witness rule stands for non-popularity contrasts. kwh6 has the same split (val -0.00053, holdout +0.00057) and stays closed by rule at 4 seeds |
| 5 | all-data files against held-out files on the level line | val: FD16P20 +0.00058 +- 0.00043 above W8P20 | -0.00012; residuals -0.0003 / -0.0007 against +0.0003 | unexplained at 1.25 paired SE. The two member sets differ in the checkpoint kind (last step against holdout-selected) and in a favourable draw of 4 students (+0.0002). A label-free drift table by test day for the two sets was proposed on 11-14 (idea I6) and not built: list B |

## 5. Ranking: top 15 by probability x gain / cost

Score = P x gain (in units of 0.000001 NDCG) / cost in GPU-h, with a floor of 1 GPU-h for zero-training items (their real cost is a public slot and selection risk). All P and gains are [I].

| rank | item | group | cost to the decisive read | P | gain | score |
|---|---|---|---|---|---|---|
| 1 | union / max-N train-only branch from members on disk | 4.1 | 0 (1 slot) | 0.4 | +0.00008 | 32 |
| 2 | domain token on the train-only branch, stored students | T02 | about 1 GPU-h inference | 0.12 | +0.00025 | 30 |
| 3 | reads of the running arms (kwsb2, kwaux1, U20 seeds 953-954) by their rules | T10, T26 | 0 new | 0.10 for any pass | +0.0003 | 30 |
| 4 | harness kw08: the other 15 re-cools | T03 | 10.5 | 0.06 | +0.0003 | 1.7 |
| 5 | width 224, seeds 955-958 | T20 | 13.6 | 0.10 | +0.0002 | 1.5 |
| 6 | hz2nF: stale-val, then 16 fine-tunes and a file read | T27 | 0.5 + 16 | 0.06 | +0.0004 | 1.5 |
| 7 | hz at file level under stale inputs (fine-tunes through the stale-FT script) | T27 | 16 | 0.08 | +0.0003 | 1.5 |
| 8 | weight decay 0 from scratch, 4 then 8 seeds | T01 | 27 (first stage 13.6) | 0.12 | +0.0003 | 1.3 |
| 9 | harness m05 (exploration mix 0.05 in the cooldown) | T03 | 11.2 | 0.05 | +0.0003 | 1.3 |
| 10 | cooldown learning-rate doses (K:lr1, K:lr4) | T04 | 11.2 each | 0.04 | +0.0003 | 1.1 |
| 11 | staleness indicator input on the hz2 view (HZCOND) | T19 | 11.2 + CPU build | 0.03 | +0.0003 | 0.8 |
| 12 | pseudo-feeds under distillation, pool read | T22 | 27 | 0.05 | +0.0004 | 0.7 |
| 13 | SAM in the cooldown, 16 re-cools | T13 | 22.4 | 0.05 | +0.0003 | 0.7 |
| 14 | extended history arm | T24 | 41 | 0.10 | +0.00025 | 0.6 |
| 15 | trained-input bundle (cold-row neighbour scores, item prior, position prior) | T16 | 27 + 1-2 CPU-days | 0.05 | +0.0003 | 0.55 |

Next: WiSE-FT (0.5), clean sub-feed arm (0.3), later-using co-visitation (0.2).

## 6. Three lists

### (A) Worth GPU tonight

1. Domain-token inference (T02). A copy of the train-only prediction path with `model.infer_domain` replaced by a soft vector (`nn_x/w5/model.py` l.439, l.536-540 accept one); two settings declared in advance: production token on every feed, and an even mix. Members: the 16 all-data wide students and the 4 all-data popularity-light students; val and test. Read at FILE level against FD16P20 (fine-tuned branch unchanged): val, M<50 / M>=50, pmix-public, pmix-private, stale-val. Staleness-family rule: val >= 0, pmix-private >= +1 SE, stale-val >= +2 SE. About 1 GPU-h.
2. Stale-val scoring (scenario both_pre2lt3) of the 32 hznFS and 32 hz2nF checkpoints at @last (`nn_x/w12/kd/stale_any_now.sh`). Completes rule B. 0.5 GPU-h. hz2nF goes on to fine-tunes only if rule B passes against hz.
3. Reads of kwsb2, kwaux1 and U20 seeds 953-954 by the rules on record; no new GPU. kwr32 is read (section 2): closed.
4. Next filler arms, in place of seeds 963-966 of rule G: weight decay 0 (new lane line `--wd 0`, seeds 951-954, 13.6 GPU-h) and width 224 seeds 955-958 (lane arm E:kd05w224 exists, 13.6 GPU-h). The pool-size law prices students 25-32 at +0.00003 for the same 27 GPU-h [M from c = 0.0027: 0.0027 x (1/24 - 1/32)].
5. After those: K:kw08 / L:kw08 (15 runs, 10.5 GPU-h) and K:m05 / L:m05 (16 runs, 11.2 GPU-h); lane lines exist.

### (B) Worth a CPU check first

1. Domain token without a GPU: replay of 2-4 stored students on val with the production token on CPU (pattern of `solvers/CPU3_aux_head_member_W6.py`), and the same contrast on the lagged holdout. 4-6 CPU-h.
2. CPU stale-val of hz2nF / hznFS (the witness agent's CPU path, `reports/STALE_1115/witness.md`) if no GPU slot is free. 3-4 CPU-h.
3. Label-free drift table by test day for the all-data members against their held-out twins: correlation to pick 1, cold-minus-warm offset, top-1 agreement (idea I6 of `reports/RECHECK_1114/SYNTHESIS.md` 4.6). Purpose: the level-line residual of section 4.4 row 5. 1 CPU-h.
4. Cold-row neighbour scores, the windowed same-cluster variant, read with the likelihood ratio and a placebo; re-open trigger LR >= 10.83 on all val feeds (`reports/LIT_1115/CRITIC.md` l.175). 1 CPU-h. Only a pass opens the trained-input bundle.
5. Later-using co-visitation: a label-free count of the test rows it would turn from cold to warm, by test day. Build nothing below 5% of rows. 1 CPU-h.
6. Scale-curve gate for stage 2 on the exploration feeds (row 235). 2 CPU-h, P 0.03.

### (C) Weakly closed, not worth the cost (one line each)

- T04 learning rate, warm-up, momentum, clipping from scratch: no mechanism under distillation; P 0.04 for +0.0002 at 13.6 GPU-h each.
- T05 weight averaging and EMA: the diversity table predicts a pool loss.
- T06 seven-epoch schedule: a cost lever while cost is not binding.
- T07, T08, T23 loss zoo, old-recipe blocks, 2-epoch families: 0 of 13 block forms and 0 of 8 family screens moved; P 0.01-0.02.
- T09, T15, T16 single feature rows: residual gates are blind, but no single row carries more than about +0.0006 in any lens, and a new input needs teacher regeneration (37 GPU-h) to count fully.
- T11 fine-tune-stage variants: no offline witness; each costs 16 GPU-h and a public slot that cannot resolve +0.0002.
- T12 language-model backbones, T18 pre-training programmes: P 0.01.
- T13 other hard-label cooldown arms (history cap, recency, RHO-loss, format parity, exploration posterior): null or negative in their era with no stated interaction.
- T14 rank-from-end offsets: negative twice.
- T17 argument-only drops: WiSE-FT is the only one with a stated robustness mechanism; 16 GPU-h for an expected +0.00005.
- T21 epochs, batch, late dropout: negative dose-response in the hard-label era, same sign at 2 seeds under distillation.
- T24 extended history: probably real, +0.00015 .. +0.0003, parity unreachable, 41 GPU-h.
- T25 external-scan items: priors 0.01-0.03, no new reason.
- T28 graph networks, in-network experts: P 0.05 at 12-31 GPU-h and 1-1.5 days of parity work.
- w_ft above 0.5, FDX16P20:hz as a slot, 12 popularity-light students: near-copy questions; public cannot order them.

## 7. Corrections for the records (the auditor edited none of these files)

- `reports/COVERAGE_MAP.md` l.317: the 11 harness arms are not "QUEUED" and not on disk; 1 run exists (K:kw08 s953). `reports/EXPERIMENT_REGISTRY.md` l.316 and the task text speak of "the 11 harness arms" as unread runs.
- l.338: of the hz2n family only hznFS and hz2nF were run (16 each); hz2n, hz2Ln, hz2LnF, hzn, hznF were not.
- l.46, l.47, l.60, l.325: kwd2, kw256d2, kwh6 are read and closed (ledger 20:46Z, 20:50Z); kwr32 is closed by today's read; weight decay 0.05 / 0.10 are closed; the rows still say RUNNING or QUEUED.
- l.339: th is closed by the hzt read of section 2.
- l.157: add the pool-level re-read of section 0 item 5 and the status "not re-read under the 11-15 evidence policy; inference-only test open".
- Web support for T01 (two generic searches, no project data sent): tuning weight decay for the ensemble as a whole, not member by member, "generally matches or improves performance" [W https://arxiv.org/abs/2511.04160, abstract]; a third-party summary of that paper reports a lower best weight decay for the ensemble than for the single model (not verified in the paper itself) [W https://www.themoonlight.io/review/on-joint-regularization-and-calibration-in-deep-ensembles]. Against dose 0 exactly: common Muon practice keeps a non-zero decoupled decay, because weight norms grow without it on long runs [W https://arxiv.org/html/2502.16982v1]. A second dose 0.005-0.01 is the hedge if dose 0 weakens the members.

## 8. Files

- `nn_x/w12/weak_1115/rows.txt` (class of every row), `groups.txt` (coverage of the 137 non-sound rows).
- `nn_x/w12/weak_1115/dose_diversity.py`, `domtok_read.py` (read-only CPU scripts written today).
- `nn_x/w12/weak_1115/out/`: `arm_read8_rc_{hznFS,hzt,hz2nF}_vs_{ref,hz}.txt`, `hz_read_rc_*`, `hz_refs_read_unread.txt`, `arm_read8_rc_kw08_vs_ref.txt`, `arm_read8_kwr32.txt`, `bucket_h256_kwr32.txt`, `dose_diversity.txt`, `domtok_read.txt`.
