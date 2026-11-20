"""e9: assemble reports/TRICKS_1115/int_mega.md = prose below + the tables of out/tables.md (every table number comes from a json written by e0..e8)."""
import re
from pathlib import Path
R = Path('<repo>'); H = R / 'nn_x/w12/ens_1115'
T = {}; cur = None
for line in open(H / 'out/tables.md').read().split('\n'):
    m = re.match(r'^### (T\d+)\. ', line)
    if m: cur = m.group(1); T[cur] = []; continue
    if cur: T[cur].append(line)
T = {k: '\n'.join(v).strip() for k, v in T.items()}
OUTSIDE = open(H / 'out/inventory_outside.md').read().strip()
# T1 is long: keep it whole (the task asks for every arm)
doc = f'''# int_mega: what a dramatically larger or better-composed ensemble gives, and what to train next

Date 2026-11-15 (snapshots 21:09Z and 21:43Z). CPU only; read-only on every prediction file; nothing submitted; two candidate files staged in `nn_x/w12/ens_1115/stage/` (not in `subs/`).
Work directory `nn_x/w12/ens_1115/`: scripts `ens_common.py`, `e0_cache.py` .. `e9_report.py`; logs and json in `out/`; z cache in `cache/`.
Tags: [M] measured in this study (file named), [R] record, [W] web source, [I] judgement.
Conventions: every delta is paired per feed over the 14,423 val feeds; "±" is the paired SE; "pmix" is val reweighted to the public / private mix (`src/pmix.py`); "holdout" is the 30,000-feed in-period holdout (out of sample only for members that exclude it; reported, not a veto for popularity-light or staleness arms).
Member score = per-feed z-score `stack_val.feed_rank_z[:, 1]` ((x - mean) / sd inside the feed), the same code path as `nn_x/w12/kd/build_r4.py`; pool = member-equal mean of z; file = 0.5 z(fine-tuned pool) + 0.5 z(train-only pool); per branch 0.8 z(base pool) + 0.2 z(popularity-light pool).
Check of the code path [M, `out/e3_forms.log`]: FD16P20 and W8P20 rebuilt from members match the stored branch files to 9.5e-07 and give final val 0.40271 and 0.40214, the values on record.

## 0. Answer

1. **A larger same-recipe ensemble is worth at most +0.00015 val (about +0.00014 public) over what is already submitted, and that number is now measured three ways** [M]. The law N(n) = N_inf - c / n fits the 16 held-out students, the 16 full-data students and all 32 together with residuals <= 0.00014 on the train-only branch, the fine-tuned branch and the final 0.5 / 0.5 mix (T2). From 16 members: 32 adds +0.00007 ± 0.00001, 64 adds +0.00011 ± 0.00002, 128 adds +0.00013 ± 0.00002, infinitely many +0.00015 ± 0.00003. At file level with the popularity-light pool in place (T3) 16 + 12 -> 32 + 12 is +0.00006 ± 0.00001 and the whole headroom above a 16 + 4 file is +0.00015 ± 0.00003. Paired public SE between near-copy files is 0.0003-0.0007 (0.00034-0.00057 for the staged files, section 4), so no pool size can be read on the public board.
2. **No second arm on disk adds more than extra seeds** [M, T4]. Among 24 arms whose pool level is within 0.001 of the wide students (kd08w192, GRU students, width 224, nine re-cool recipes, kwh6, kwr32, kwd2, kwsb2, kdmix, two generation-2 arms, kx05, kwwd05, kwh30, the d128 students kd05 and kd05t25) the matched "half the pool replaced by the arm" delta averages -0.00007 on val (range -0.00051 .. +0.00026; z mean -0.24, sd 0.73, max +1.11) and -0.00026 on the holdout; none reaches z = 2 on either witness. Cross-arm correlation equals within-arm correlation (0.963-0.980 against 0.9725; every value below 0.970 comes with a lower member level or a holdout loss): under distillation the seed is the only source of diversity. The only input family that decorrelates beyond seed noise is popularity (T12), and it is already in the file at weight 0.2.
3. **The largest honest ensembles available today do not beat FD16P20** [M, T6, T7]. Structure A (32 wide + 36 popularity-light students, 68 members): final val 0.40247 = -0.00024 ± 0.00028 against FD16P20 and +0.00033 ± 0.00036 against W8P20. Structure B (60 members, all trained on all data: 16 wide + 16 hz re-cools + 28 popularity-light): 0.40216 = -0.00056 ± 0.00027 against FD16P20 and +0.00002 ± 0.00044 against W8P20. The hz re-cools lower the fine-tuned branch again (FD16 + FDXHZ16 against FD16, same four popularity-light students: file -0.00034 ± 0.00025, fine-tuned branch -0.00038 ± 0.00024; the shape of the three member sets on record).
4. **FD16P20's val level is a favourable draw, and the draw does not transfer** [M, T10]. Its four popularity-light students put it above 472 of the 495 possible "4 of the 12" files (mean 0.40249, sd 0.00013). Across random halves of the val users the draw excess has correlation -0.16 (val), -0.03 (public mix), -0.005 (private mix): reliability <= 0. Against the expected FD16P20-type file A reads -0.00002 ± 0.00024 and B -0.00033 ± 0.00022. So A is level with the recipe, B is below it, and neither is a gain.
5. **Learned combiners are worth zero nested** [M, T8]. Ten forms (Caruana greedy with replacement, non-negative least squares, ridge; member or arm columns; all members or all-data members only): nested deltas against B from -0.00083 ± 0.00042 to +0.00020 ± 0.00034, all below FD16P20; in-sample the same forms read up to +0.00177 above B. Twenty scan forms (arm-balanced weights, d128 / hard-label admixture, rank / softmax / trimmed / median pooling) are all inside ±2 SE of their structure except one negative; the best one (rank pooling, +0.00032 on B and +0.00022 on A) was put to a pre-written replication read on four disjoint member sets and did not replicate (pooled -0.00007, z -0.5; T9).
6. **What to train next for the ensemble: nothing pays for level** [I from 2]. Sixteen more full-data wide students cost about 70 GPU-hours for +0.00006 ± 0.00001; cooldown re-runs of existing snapshots correlate 0.990 with their root and cap at +0.00007 on the train-only branch; fine-tunes of the 76 held-out members that lack one would move a held-out file by -0.0003 .. +0.0003 (all inside noise, law value <= +0.0001). A second arm would need a cross-arm correlation near 0.945 at unchanged member level to add +0.0005 (frontier regression, T4); the lowest value on disk among wide-level arms is 0.963, and the popularity-light students sit at 0.961 with a pool level 0.001 lower.
7. **Staged** (section 4): `cand_ENSAkd_stage.parquet` (A) and `cand_ENSBkd_stage.parquet` (B) with branch files, member lists, sha256 and read-outs. No third or fourth file: no form beats B by more than 2 SE nested. Neither file is recommended for a slot as a level bet; A is a lower-variance copy of the recipe (not all-data), B is all-data and expected below FD16P20.

## 1. Inventory [M]

Source: `e0_cache.py` -> `out/inventory.json` (per member: path used, number of identical copies, file time, NDCG), `e1_inventory.py` -> `out/inventory_arms.json` / `.md`, `e1b_outside.py` -> `out/inventory_outside.md`.
On disk at 21:43Z: 1,223 member names over `preds/`, `nn_x/w5/preds/`, `nn_x/abl/preds/` and their `holdout/` folders. 633 are in the cache: every 10-epoch-class member that has a train-only val AND a train-only test file (selected checkpoint), plus the last-step twins of the re-cool arms and of the 8 held-out hz re-cools that were fine-tuned at the last step. 132 members have the fine-tuned pair (n2cpv3s3 val_oof + test). 573 have a holdout file. Every file that exists in two directories was compared: 0 differing copies. Full-data flag = `args.holdout_in_train` in the member's training-log json: 60 members.
kwsb2 x4 finished at 21:25Z and is included (second cache build 21:43Z); kwaux1 and u20 seeds 953-954 were still training and are not on disk.

{T['T1']}

Train-only val pool values with two numbers are the K lineage (kd05w192 snapshots) / the L lineage (kd05t25w192 snapshots). The four files of a fine-tuned member are all present for the 132 members counted in "FT val + test"; no member has a fine-tuned val file without the test file.

Members on disk outside the cache (not used; names and file counts):

{OUTSIDE}

The `w5_cd_valcd` members were cooled on val folds; their val files are not out of sample and must not enter any pool.

## 2. Pool math [M]

### 2.1 Pool-size law for the wide students

`e2_law.py` -> `out/law.json`, `out/e2_law.log`. For every n up to 200 distinct random subsets (all subsets when fewer); the subset mean of the per-feed NDCG estimates E[N(n)] over seeds. Fit by least squares over n = 1..N. SE of c = feed SE (the fit is linear in the per-n means) combined with a member jackknife; the forecast interval is that SE times (1/16 - 1/K).

{T['T2']}

Reading.
- The three sets give the same law: c between 0.0020 and 0.0025, N_inf (final) 0.40176 / 0.40208 / 0.40195. The held-out and the full-data students are the same recipe to within the SE (same-seed pairs across the two sets correlate 0.9732, different-seed pairs 0.9726, pairs inside a set 0.9725 / 0.9737: the full-data twin of a seed is an independent member) [M, `e2d_misc.py` -> `out/misc_corr.json`].
- Out-of-range check: a fit on n <= N/2 predicts the realised pool of N with error -0.00015 .. +0.00017. Direct check: the realised 32-pool minus the mean 16-pool is -0.00010 ± 0.00014 (train-only), +0.00021 ± 0.00014 (fine-tuned), +0.00004 ± 0.00014 (final); the law says +0.00007.
- Under the public / private mix the constant is the same (final: 0.00251 / 0.00239 for W32, `out/law.json` keys `all_public`, `all_private`).
- One realised 16-member file carries val draw noise: sd 0.00020 over 16-of-32 draws at file level (T10). That noise is larger than every gain in this table, and it does not transfer to new feeds (section 2.5).

### 2.2 The same law at file level, both pools

`e2c_file_law.py` -> `out/file_law.json`. All-data members: nb of the 16 full-data wide students as base, np of the 12 full-data wide pop1 students at weight 0.2, 40 random draws per cell, final val.

{T['T3']}

Reading. The popularity-light pool is saturated at once (cp = 0.00014: one student already carries the admixture; 4 -> 12 students is worth +0.00002). The base pool carries cb = 0.00177, lower than the 0.0021-0.0025 of a base-only file because the admixture already removes part of the noise. Total headroom above a 16 + 4 file: +0.00015 ± 0.00003 (val), +0.00014 ± 0.00003 (public mix), +0.00012 ± 0.00004 (private mix).

### 2.3 A second arm at equal size

`e2_arms.py` -> `out/arms.json`, `out/arms_late.json` (kwsb2), `out/e2_arms.log`. For every arm with at least 4 members: correlation of per-feed z inside the arm and against the 16 held-out wide students; pool level of h members against h wide students; "replace" = pool(h arm + h wide) minus pool(2h wide) with disjoint seeds (re-cool arms: disjoint snapshot roots, and the removed wide students are the ORIGINAL runs of the arm's roots, so the contrast is the recipe at equal independence); "add" = all arm members added to W16. 60 random draws per arm. h = 8 for arms with at least 8 members, else 4. Re-cool arms are read at the last step (the decision checkpoint of the lead's rule); the selected-checkpoint rows are in the json.

{T['T4']}

Reading.
- Arms within 0.001 of the wide level, popularity views excluded (24 arms): replace on val mean -0.00007, range -0.00051 .. +0.00026, z mean -0.24 (sd 0.73, max +1.11); on the holdout mean -0.00026, max z +1.95; none at z >= 2 on either witness. Two arms are positive on both (kd05w224 +0.00004 | +0.00021; kdmixw192 +0.00026 | +0.00009), both closed by the lead's rule [R: pooled z < 1; kd05w224 +1.38 at matched N]; with 24 arms two such rows are what a null gives.
- Correlation: cross-arm 0.970-0.980 against 0.9725 inside the wide students for 20 of the 24 arms. The four lower values come with a lower member level or a holdout loss (kd03g2w192 0.963 at member 0.39762; hz2 / hz2nF 0.967-0.968 with holdout pool level -0.0029; xd30 0.969 with level -0.0007 | -0.0011). Higher teacher weight or weight decay raises correlation (kd08w192 0.990 within, 0.980 cross) and loses at pool level.
- Same-root re-cools of the same recipe correlate 0.990 with their original run (0.983-0.990 for the other recipes; 0.989 for the full-data hz re-cools on both branches, `out/misc_corr.json`) against 0.972 for a different root: a cooldown re-run is about one third of a new member.
- Popularity-light students are the one arm that is positive at the public / private mix (pop1kd05w192: replace val +0.00056 ± 0.00048, pmix +0.00116 / +0.00149) and negative on the in-period holdout (-0.00183 ± 0.00034); this is the known two-sided picture of the admixture, already used at weight 0.2.
- Hard-label generations (base, F7x, F7x70, F10, v17, v22, v26): cross correlation 0.945-0.959 but member level 0.0027-0.0056 lower; replace -0.0004 .. -0.0013 on val. They stay out.

Frontier regression across arms (same table, h = 4): replace = alpha x (arm pool level - wide pool level) + beta x (rho_wide - rho_cross), fitted with `e8_tables.py` -> `out/frontier.json`: val alpha 0.46, beta 0.018 (R2 0.81, residual sd 0.00020); holdout alpha 0.32, beta 0.005 (R2 0.98). In words: half of a level gap is paid in full by the mixed pool, and one point (0.01) of extra decorrelation at unchanged level buys +0.00018 on val and +0.00005 on the holdout. A second arm that should add +0.0005 on val needs rho_cross near 0.945 at the wide member level. No student arm on disk is below 0.961.

### 2.4 Cooldown branches of one snapshot (two-level law)

Same script, part C. Branches orig / ref / ref2 are three cooldowns of the same 16 plateau snapshots with the same recipe (harness control and its A/A twin [R, E_kd_predeclared.txt blocks 04:16Z and 05:50Z]).

{T['T5']}

Reading. The branch term is real (c_branch 0.0007-0.0010) but it is divided by R x B: three branches per root add +0.00004 ± 0.00001 on val and +0.00003 on the holdout; infinitely many +0.00007 / +0.00004. The observed 16 x 3 pool is 0.40148 against 0.40152 for 16 x 1 on val and 0.32444 against 0.32438 on the holdout. Doubling the roots is worth twice as much as tripling the branches, and both are below 0.0001.

### 2.5 Draw noise and whether it transfers

`e6_draws.py` -> `out/draws.json`.

{T['T10']}

Reading. The level differences between same-recipe draws on val (sd 0.00013 for the popularity-light draw, 0.00020 for a 16-of-32 base draw, 0.00025 for an 8-of-16 base draw) are not a property of the members: the split-half correlation over users is between -0.16 and +0.11. With 12 or 32 members behind these draws the correlation itself has an SE of roughly 0.1-0.3, so "zero" is the reading, not "negative". Consequence for the picks: the +0.00022 val (+0.00044 public mix, +0.00053 private mix) by which FD16P20 exceeds its expected draw is val-sample noise; the expected level of FD16P20 on new feeds is that of the average draw, and choosing members or member subsets by val has no value (the lead's lesson 2, now with a number).

### 2.6 Which views differ beyond seed noise

`e7_views.py` -> `out/views.json`. Hard-label members against the standard hard-label view (10-epoch arms against `w5_e10_base` x6, base-base correlation 0.9484; 2-epoch ablations against `abl_base`, base-base 0.9778). Excess decorrelation = base-base correlation minus arm-to-base correlation at different seeds.

{T['T12']}

Reading. Only the popularity input family (pop1, pop2, popularity-free) decorrelates by 0.013-0.019 beyond seed noise. The next ones are the no-retrieval model (+0.013 at a member loss of 0.0035 | 0.0029), batch x4 (+0.012 at -0.0025), and the old families with a different learning rate, width or encoder (+0.008..+0.011 at -0.0017..-0.0028); inputs other than popularity are at -0.001..+0.005 (pre-trained scores +0.004, context +0.005); 20 epochs gives +0.007 at -0.0007. By the frontier regression none of these is predicted to pay for its level gap, and no 10-epoch matched mix is positive on both witnesses by more than 1 SE. The mix column of the 2-epoch rows compares against one base pair (seeds 23, 24) and is positive on the holdout for almost every arm, including null arms (time, kv64): it measures that pair, not the arms; only their correlation column is used. This is the same answer as the correlation table of the students: there is no unused view on disk.

## 3. The largest honest ensemble available today [M]

`e3_forms.py` -> `out/forms.json`; `e3c_pairs.py` -> `out/pairs.json`; `e3d_scan_pairs.py` -> `out/scan_pairs.json`; `e3_nested.py` -> `out/nested.json`; `e3b_rank_rep.py` -> `out/rank_rep.json` (rule in `out/e3b_rule.txt`, written before the run).

**Pre-declared structures (member lists fixed before any number was read, nothing fitted):**
- **A** = base W16 + FD16 (32 wide students) | popularity-light pool = every popularity-light student with a fine-tune (4 held-out d128, 4 held-out wide, 12 full-data wide pop1, 4 full-data d128, 4 full-data pop2, 8 full-data popularity-free = 36) | weight 0.2. 44 of 68 members are trained on all data.
- **B** = base FD16 + FDXHZ16 | popularity-light pool = the 28 full-data students | weight 0.2. All 60 members are trained on all data.

**Scans (every form is listed; none was chosen for a file):** c1 / c2 (popularity-light pool arm-equal / view-equal), d1 / d2 / d3 (12 d128 students, 20 hard-label members, or both, at weight 0.1 inside the base pool), f (rank-z mean, softmax-probability mean, trimmed mean 10% and 25% per side, median inside each of the four sub-pools), each on A and on B = 20 forms; e = 10 learned combiners, nested. "step" rows are descriptive sub-structures (11 rows).

{T['T6']}

{T['T7']}

Scan forms read against their own structure (`out/scan_pairs.json`): A: c1 -0.00014 ± 0.00009, c2 -0.00013 ± 0.00010, d1 -0.00010 ± 0.00012, d2 -0.00022 ± 0.00016, d3 -0.00045 ± 0.00019, rank +0.00022 ± 0.00024, softmax -0.00003 ± 0.00012, trim10 -0.00009 ± 0.00011, trim25 -0.00006 ± 0.00015, median -0.00007 ± 0.00018. B: c1 +0.00011 ± 0.00011, c2 +0.00020 ± 0.00011, d1 +0.00005 ± 0.00013, d2 -0.00015 ± 0.00015, d3 -0.00025 ± 0.00020, rank +0.00032 ± 0.00024, softmax -0.00008 ± 0.00012, trim10 -0.00003 ± 0.00011, trim25 -0.00004 ± 0.00017, median +0.00002 ± 0.00020. Largest positive z: +1.75 (view-equal on B; the same form reads -1.28 on A).

Learned combiners (e):

{T['T8']}

Method of (e): two columns per member or per arm pool (z of the fine-tuned and of the train-only prediction). Column sets: FD = the 60 all-data members (6 arms), ALL = the 132 members with a fine-tune (20 arms). Caruana: 64 greedy steps with replacement from the empty pool, target NDCG. NNLS: feed-centred labels, row weight 1 / M. Ridge: lambda chosen on inner folds. Outer folds = the fine-tune's fold rule (GroupKFold(5) by user in flat feed order, user-0 feeds as singletons, `nn/finetune.py --u0_singleton`; fold sizes 2,885 / 2,885 / 2,885 / 2,884 / 2,884); no user is on both sides and nothing is tuned on the outer test fold. A bias that no nesting removes: on val the fine-tuned columns are out-of-fold predictions of models that saw 80% of val, on test they are 15-model averages, so val-fitted weights under-weight the fine-tuned branch (NNLS gives it 0.15-0.18 of the weight).

Replication read of the one scan form that stood out (rank pooling):

{T['T9']}

Reading of section 3.
- Against the realised FD16P20 both structures are lower (A -0.9 SE, B -2.1 SE) and both are below -1 SE at the private mix (A -0.00047 ± 0.00041, B -0.00084 ± 0.00044): by the standing pmix veto neither replaces FD16P20. Against the expected FD16P20-type file A is level (-0.00002 ± 0.00024; pmix -0.00004 / +0.00006) and B is lower (-0.00033 ± 0.00022).
- On the FD16 or W16 + FD16 base, every structure that uses all 12 wide pop1 students sits at 0.40243-0.40258, i.e. at the expected level 0.40249; the two structures whose whole popularity-light pool is the first four of them (FD16P20, W16 + FD16 + FDP4) sit at 0.40270-0.40271. The gap is the draw of those four students, not the size of anything.
- Leave-one-arm-out: A moves by -0.00043 .. +0.00004 (the two full-data wide arms matter most: -0.00028 ± 0.00019 and -0.00043 ± 0.00018; every popularity-light arm is within ±0.00022); B moves by -0.00006 .. +0.00024, and it improves when either hz arm is removed (+0.00014 ± 0.00018, +0.00024 ± 0.00019). No single arm makes or breaks either structure.
- Forms tried: 2 structures, 20 scan forms, 10 nested combiners, 11 descriptive step rows, 1 replication rule. Largest nested reading against B: +0.00020 ± 0.00034 (ridge on all-data arm pools). The bar for an extra file (beats B by more than 2 SE nested) is met by nothing; the nearest fixed structure is "W16 + FD16 + FDP4" at +0.00054 ± 0.00029 (z 1.85), which is the favourable four-student draw again.

## 4. Staged files [M]

`e4_build.py` -> `nn_x/w12/ens_1115/stage/` (`out/e4_build.log`). Format = `src/make_sub.make_sub` (2,904,806 rows, 28,595 feeds, columns feed_id / item_id / score as in `subs/cand_FD16P20kd_r4.parquet`, identical row order). Pools are rebuilt straight from the member files in `preds/` (the build_r4 path) and asserted equal to the cache path. Not gated: the lead gates and moves them.

| file | members | all data | sha256 | final val | FT / TO branch val | vs FD16P20: val | pmix public | pmix private | test corr | paired public SE | vs W8P20: val | pmix public | pmix private | test corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage/cand_ENSAkd_stage.parquet` | 32 base + 36 popularity-light, w 0.2 | 44 of 68 | 0ddea3148bb0fa94250fc938eb2f5b8beb358b09fc17af2ecba209197c9724aa | 0.40247 | 0.40162 / 0.40183 | -0.00024 ± 0.00028 | -0.00048 ± 0.00034 | -0.00047 ± 0.00041 | 0.99951 | 0.00036 | +0.00033 ± 0.00036 | +0.00027 ± 0.00044 | +0.00022 ± 0.00053 | 0.99882 |
| `stage/cand_ENSBkd_stage.parquet` | 32 base + 28 popularity-light, w 0.2 | 60 of 60 | 1840dfb71c5798a0e95e68a84ac2f92d833add4e02cd61feb54ba448b400c221 | 0.40216 | 0.40144 / 0.40203 | -0.00056 ± 0.00027 | -0.00081 ± 0.00035 | -0.00084 ± 0.00044 | 0.99959 | 0.00034 | +0.00002 ± 0.00044 | -0.00006 ± 0.00054 | -0.00015 ± 0.00067 | 0.99759 |

Next to each file: `ens_<TAG>kd_{{ft,to}}_{{val,test}}.npy` (flat branch files), `<TAG>_members.txt`, `<TAG>_manifest.json` (member lists, sha256 of the parquet and of the four branch files, the read-out above).
Forecast [I]: level line public = 0.0074 + 0.963 x val gives 0.39498 for A and 0.39468 for B (residual sd 0.0006). Relative to FD16P20 (public 0.394892): its val edge over A is the non-transferable draw of section 2.5, so the expected public difference of A is about 0.0000 ± 0.0004 and P(A above the team best 0.3950094) is about 0.4; B is expected about 0.0003 lower, P(B above 0.3950094) about 0.1. Neither is a level bet. A is not an all-data file (24 of its 68 members exclude the 30,000 holdout feeds), which conflicts with the owner's instruction for finals; its only merit is a member draw twice as large.

## 5. What to train next [I, from section 2]

Costs as given: a 10-epoch student 3.4 GPU-hours, its fine-tune 1 hour (4.4 per usable member), 24 jobs at once. A cooldown re-run costs about 0.25 GPU-hours [R, COVERAGE_MAP row "KD re-cool harness"] plus the fine-tune. Forecasts are file-level val; public = 0.963 x val; intervals are ±2 SE of the law.

| option | new jobs | GPU-hours | forecast file gain (val) | interval | gain per 100 GPU-hours | source |
|---|---|---|---|---|---|---|
| same recipe: 16 more full-data wide students (base 16 -> 32) | 16 + 16 FT | 70 | +0.00006 | +0.00004 .. +0.00008 | +0.00008 | T3: cb (1/16 - 1/32) |
| same recipe: 48 more (base 64) | 48 + 48 FT | 211 | +0.00008 | +0.00006 .. +0.00011 | +0.00004 | T3 |
| same recipe: 112 more (base 128) | 112 + 112 FT | 493 | +0.00010 | +0.00007 .. +0.00013 | +0.00002 | T3 |
| popularity-light: 12 more wide pop1 students (12 -> 24) | 12 + 12 FT | 53 | +0.00001 | 0.00000 .. +0.00001 | +0.00001 | T3: cp (1/12 - 1/24) |
| same recipe, cheap: 32 cooldown re-runs of the 16 full-data snapshots (3 branches per root) | 32 + 32 FT | 40 | <= +0.00004 on the train-only branch; fine-tuned branch not measured for this recipe (hz-type re-cools: -0.00038 ± 0.00024 and -0.00019 ± 0.00021) | +0.00002 .. +0.00006 at best | <= +0.00010 | T5, T11 controls |
| second arm at equal level | none on disk qualifies | | 0 | | | T4, T12 |

- **Same recipe, more seeds:** the best rate is the first 16 students, +0.00006 for 70 GPU-hours, which is 0.2 of the paired public SE of a near-copy pair. It cannot be confirmed on public and would not change a rank except by a tie-break. Its real effect is on variance: a 32-member base halves the member-draw component of a file (val sd 0.00020 -> about 0.00014), which matters only if the final board is decided by 0.0001.
- **Second arm:** the requirement is rho_cross <= about 0.945 at the wide member level for +0.0005, or equivalently one more input view like popularity. No arm, recipe or view on disk meets it (T4, T12). More popularity-light students add nothing (cp 0.00014); the pop2 and popularity-free views are the same view (closed today on replication).
- **Fine-tunes of existing held-out members without one** (T11): 76 wide-level members could be fine-tuned for 76 GPU-hours. The train-only branch says the file would move by -0.0003 .. +0.0003 with SE 0.0002, law value +0.00003 .. +0.00011. The two controls where fine-tunes exist warn that the fine-tuned branch of re-cools does not follow the train-only branch (W16 + 8 hz re-cools: TO +0.00018 ± 0.00023, FT -0.00019 ± 0.00021; FD16 + FDXHZ16: TO -0.00018 ± 0.00025, FT -0.00038 ± 0.00024). These would also be held-out members, not all-data ones. Value today: none.

{T['T11']}

Caveat on T11 [M, `e5b_addcheck.py` -> `out/addcheck.json`]: its holdout column carries a common offset of +0.0001 .. +0.0003 against the "add" column of T4 (same members on the bare base pool), because every row shares one baseline term (the admixture on W16 alone reads -0.00055 ± 0.00021 on the holdout). Examples: kd05w224 -0.00006 ± 0.00012 on the bare pool against +0.00021 ± 0.00011 inside the branch; the 12-member union of GRU, width 224 and kdmix +0.00000 ± 0.00015 against +0.00024 ± 0.00013. On the bare base pool no addition is positive on the holdout (-0.00020 .. +0.00000). The one row of T11 that is positive on both witnesses (the 12-member union: val +0.00026 ± 0.00024, holdout +0.00024 ± 0.00013) is the best of 17 rows and its holdout half is that offset; it is not evidence for a mixed-recipe pool.

- **Recommendation:** no GPU-hour for pool size or pool composition. If GPUs are idle after the running lanes, 16 more full-data wide students are the only ensemble item with a positive, bounded forecast (+0.00006, variance halved); it should be queued behind anything that changes what a member knows.

## 6. Web sources [W] (generic searches; no project data left the machine)

- Lobacheva, Chirkova, Kodryan, Vetrov, "On Power Laws in Deep Ensembles", NeurIPS 2020, https://arxiv.org/abs/2007.08483 : ensemble quality follows a power law in ensemble size and the fitted law forecasts the gain of larger ensembles from a few trained networks. The 1 / n law of T2 is the same device; here the forecast is +0.00015 in total.
- Fort, Hu, Lakshminarayanan, "Deep Ensembles: A Loss Landscape Perspective", 2019, https://arxiv.org/abs/1912.02757 : independent initialisations reach different modes, while functions along one optimisation trajectory or its subspace cluster in one mode. Matches T4 / T5: a cooldown re-run of the same snapshot correlates 0.990 with its root against 0.972 for a new root.
- Allen-Zhu, Li, "Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning", ICLR 2023, https://arxiv.org/abs/2012.09816 (text read at https://ar5iv.labs.arxiv.org/html/2012.09816): an ensemble over models trained by knowledge distillation "does not exceed the ensemble over the original single models"; distilled models "have learned most of the features from the ensemble" and have less variety. Matches T1: students correlate 0.972 against 0.948 for hard-label members, and a student pool gains 0.0019-0.0021 from 1 to 8 members where a hard-label pool gains 0.0039 from 1 to 7.
- Abe, Buchanan, Pleiss, Cunningham, "Pathologies of Predictive Diversity in Deep Ensembles", TMLR 2024, https://arxiv.org/abs/2302.00704 : interventions that raise diversity at the cost of member accuracy harm ensembles of large networks; more accurate, less diverse members are preferred. Matches the frontier regression (level coefficient 0.46 against 0.018 per unit of decorrelation).
- Gontijo-Lopes, Dauphin, Cubuk, "No One Representation to Rule Them All", ICLR 2022, https://arxiv.org/abs/2110.12899 : error correlation falls in the order re-initialisation, hyper-parameters, architectures, frameworks, datasets; large ensemble gains need a different training methodology or different data. Matches T12: only a different input family (popularity) decorrelates beyond seed noise.
- Wenzel, Snoek, Tran, Jenatton, "Hyperparameter Ensembles for Robustness and Uncertainty Quantification", NeurIPS 2020, https://arxiv.org/abs/2006.13570 : hyper-parameter diversity plus seeds beats seeds alone in image classification. Not reproduced here: under a shared teacher the hyper-parameter arms correlate like seeds (T4).
- Purucker, Beel, "CMA-ES for Post Hoc Ensembling in AutoML", AutoML 2023, https://arxiv.org/abs/2307.00286 , and Purucker et al., https://arxiv.org/abs/2307.08364 : post hoc weight optimisation overfits the validation data for ROC AUC (a ranking metric), and population-based selection beats greedy selection "only statistically significant on validation data". Matches T8 (in-sample up to +0.0018, nested 0).
- Abe et al., "Deep Ensembles Work, But Are They Necessary?", NeurIPS 2022, https://nips.cc/virtual/2022/poster/53960 : ensemble benefits track in-distribution accuracy rather than a separate robustness effect. Consistent with the same law constant under the public and private mix (section 2.1).

## 7. Closed rows this study touches (nothing here re-opens them)

- COVERAGE_MAP "More same-recipe student seeds (W16, W24, W32), students of several teachers, narrow + wide mixes, hard-label families added to the student pool: CLOSED 11-14 for LEVEL": confirmed with 32 students, with the fine-tuned branch and at file level (T2, T3).
- "second-level combiners ... 78 arms: CLOSED 11-13" and "GBDT members, stackers, rank / RRF fusion: CLOSED": confirmed on today's members at final-file level, nested (T8, T9).
- "hz ... CLOSED as a file recipe 11-15": the fourth member set (FD16 + FDXHZ16 as one base) has the same shape (T6 step rows, T11 controls).
- "View diversity inside the popularity-light pool: CLOSED 11-15": confirmed (cp 0.00014; c1 / c2 scans).
- "snapshot / SWAG" and "plateau weight averaging": the prediction-level version (cooldown branches) is measured here for the first time and is worth <= +0.00007 (T5).
- New measurement that bears on a pick: the val edge of FD16P20's four popularity-light students has zero split-half reliability (T10). The row "more wide pop1 students (8 / 12 instead of 4): -0.0002" compared the 12-student file with that favourable draw; against the expected draw the 12-student file is +0.00007 ± 0.00010. This does not make the 12-student file a gain; it says the two are equal in expectation.

## 8. Files

- Scripts: `nn_x/w12/ens_1115/ens_common.py`, `e0_cache.py`, `e1_inventory.py`, `e1b_outside.py`, `e2_law.py`, `e2_arms.py`, `e2c_file_law.py`, `e3_forms.py`, `e3b_rank_rep.py`, `e3c_pairs.py`, `e3d_scan_pairs.py`, `e3_nested.py`, `e4_build.py`, `e2d_misc.py`, `e5_ft_preview.py`, `e5b_addcheck.py`, `e6_draws.py`, `e7_views.py`, `e8_tables.py`, `e9_report.py`.
- Outputs: `nn_x/w12/ens_1115/out/` (`inventory.json`, `inventory_arms.md`, `inventory_outside.md`, `law.json`, `arms.json`, `arms_late.json`, `file_law.json`, `frontier.json`, `forms.json`, `pairs.json`, `scan_pairs.json`, `nested.json`, `rank_rep.json`, `e3b_rule.txt`, `draws.json`, `ft_preview.json`, `addcheck.json`, `misc_corr.json`, `views.json`, `tables.md`, logs `e*.log`).
- Staged: `nn_x/w12/ens_1115/stage/cand_ENSAkd_stage.parquet`, `cand_ENSBkd_stage.parquet`, `ens_ENS{{A,B}}kd_{{ft,to}}_{{val,test}}.npy`, `ENS{{A,B}}_members.txt`, `ENS{{A,B}}_manifest.json`.
- Cache: `nn_x/w12/ens_1115/cache/` (23 GB of z arrays; can be deleted, `e0_cache.py` rebuilds it in 35 s).
'''
out = R / 'reports/TRICKS_1115/int_mega.md'; out.write_text(doc); print(out, len(doc.split('\n')), 'lines', len(doc), 'chars')
