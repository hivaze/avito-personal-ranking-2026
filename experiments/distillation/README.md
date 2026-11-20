# Ensemble distillation

In brief:

- A student ranker, trained toward half clicks and half the averaged scores of 11 earlier click-label rankers (the teacher), beat a click-label member by +0.00257 ± 0.00047 on validation, per student over 4 seeds.
- At width 192 the first file of 8 students, W8, scored 0.394486 on public, +0.001032 over the best click-label file. Every full-view student of the final files is trained this way. Heavier teacher weights, students as the next teacher and a cross-fitted teacher were all closed at pool level, and so were 26 further changes to the width-192 student tried from 14 to 16 November.
- Pseudo-feeds built from user histories, added to the students' training, failed their 8-seed rule (randomization p 0.19). The all-data file whose 16 full-view students train this way, FDX16P20:ps, scored 0.39526061 on public, 0.00037 above FD16P20 at a paired SE of 0.00047, and became pick 2 on 2026-11-19.

Member, pool, student, teacher, cooldown, holdout and the other terms of this page are defined in the [glossary](../../docs/glossary.md).

## Why it was tried

By 2026-11-13 the best click-label pool was about 0.004 above its best member on validation, and wider models only memorised. With about one click among 100 candidates the one-hot target is noisy; a teacher's soft scores are less so.

One form of distillation had already been closed. On 2026-11-11, the first day, a single 2-epoch teacher read +0.00064 on validation (z 1.6). An ensemble teacher on the 10-epoch recipe had never been run (decision ledger, 2026-11-14 03:14 UTC).

## What ran

### Teacher and student

The teacher is the mean logit of 11 train-only members trained on click labels: 7 transformer members whose cooldown was re-run with 2% random-order batches, and 4 plain ones. `dump_teacher.py` writes it for all 246.7 million train candidates, in at most 4.6 minutes per call of one or two checkpoints. It reads NDCG 0.3237 on the unseen holdout and 0.3465 on the feeds it saw.

A student is trained by the usual recipe (Muon, 10 epochs, a cooldown with 2% random-order feeds) on this target, taken over the candidates of one feed:

(1 - w) · clicks / number of clicks + w · softmax(teacher)

Here w is the teacher weight. One width-192 student took 195 minutes of wall-clock on a GPU shared by four jobs.

A feed of three candidates shows the arithmetic. The user clicked the first, so clicks / number of clicks is (1, 0, 0), and the teacher's softmax is (0.5, 0.3, 0.2). At w = 0.5 the target is 0.5 · (1, 0, 0) + 0.5 · (0.5, 0.3, 0.2) = (0.75, 0.15, 0.10). The click keeps most of the mass, and the teacher orders the two unclicked candidates instead of setting both to zero.

On real feeds of about 100 candidates the teacher's softmax is much flatter ([What it taught](#what-it-taught)).

### The cross-fitted teacher

The teacher above scores the train feeds it trained on, so its soft targets partly copy the click. A cross-fitted teacher removes that copy. The train users were split into 5 folds by a hash of the user id, and the anonymous user 0 by feed index. For each fold, 3 click-label members of width 128 (seeds 951 to 953) trained without it: 15 runs (`lab/nn_x/w12/kd/run_oof.sh`).

A train feed's target then comes from the mean logit of the 3 fold teachers that never saw its user (`assemble_oof.py`). The holdout feeds stayed out of all training. A gate written before the students checked that the result is out of sample: 0.32390 on 30,000 random-order train feeds against 0.32100 on the holdout, a gap of 0.0029 under the bar of 0.004. The in-sample teacher reads 0.34419 on the same train feeds.

Each fold teacher saw only 80% of the users. Per member that cost -0.00275 ± 0.00049 on validation and -0.00217 ± 0.00028 on the holdout against the same recipe on all users, and -0.0017 / -0.0019 for a pool of 3 (`lab/nn_x/w12/recheck/kd_literature/bagging.txt`). Students were trained toward it at teacher weight 0.5 and 0.8 (`kx05`, `kx08`).

### How an arm was read

An arm is one variant, run on the same seeds as its reference. On 2026-11-14 each arm was a 4-seed pool of train-only students, set against a reference pool on the same seeds: click-label members at first, then the best student pool of the day. Both were read on validation (14,423 feeds) and on the holdout (30,000 random-order train feeds kept out of training), under a rule written in the ledger before launch.

That evening an audit of nine separate checks recomputed the headline numbers, and three reviews tried to refute its 106 findings. The checks and reviews are listed under [Raw material and records](#raw-material-and-records).

From then on a change to the student recipe was adopted only at 8 seeds, with both differences ≥ 0 and pooled z = (z_val + z_holdout) / √2 ≥ 2 (`arm_read8.py`). Most arms were first screened on seeds 951 to 954 against the same 4 seeds of the reference recipe. They went on to 8 seeds only with both differences ≥ 0 and pooled z ≥ 1.6 ([student recipe arms](#student-recipe-arms-14-to-16-november)).

## Results

### 4-seed pools of 2026-11-14

NDCG of pools of train-only predictions. The member difference is per student, against a click-label member on the same seed. The rows read on 2026-11-14 come from the decision ledger (a working record, not published).

| Arm | Member difference, validation / holdout | Pool, validation / holdout | Verdict |
|---|---|---|---|
| click labels only, 4 seeds | | 0.39887 / 0.32328 | reference |
| teacher weight 0.5, width 128 | +0.00257 ± 0.00047 / +0.00200 ± 0.00028 | 0.40085 / 0.32349 | adopted |
| teacher weight 0.3 | +0.00152 (validation) | 0.39977 / 0.32342 | closed |
| teacher weight 0.8 | +0.00273 (validation) | 0.40004 / 0.32377 | closed |
| teacher weight 1.0 | +0.00228 (validation) | 0.39906 / 0.32291 | closed |
| 25-member teacher | | 0.40095 / 0.32317 | closed |
| width 192, 4 seeds (`kd05w192`) | | 0.40134 / 0.32408 | reference for the rows below |
| width 192, 25-member teacher, 4 seeds (`kd05t25w192`) | | 0.40100 / 0.32412 | in the next row |
| width 192, 8 students (two teachers, post hoc) | | 0.40157 / 0.32456 | adopted |
| students as the next teacher, weight 0.5 | | 0.40080 / 0.32410 | closed |
| students as the next teacher, weight 0.3 | | 0.40093 / 0.32393 | closed |
| cross-fitted teacher, weight 0.5 | | 0.40079 / 0.32347 | closed |
| cross-fitted teacher, weight 0.8 | | 0.40008 / 0.32321 | closed |

Teacher weight 0.5 at width 128 read against click labels only. The rows after it, down to the two 4-seed arms of width 192, read against teacher weight 0.5 at width 128 on the same seeds. The students as the next teacher learned from a teacher made of the 8 first width-192 students.

The width-192 reference is seeds 951 to 954 of `kd05w192` at the checkpoints the holdout selected. Among the 70 possible 4-seed pools of the 8 stored students of that recipe it ranks 3rd on validation, +0.00057 over their mean of 0.40077, and 37th on the holdout (`lab/nn_x/w12/moe/why/verify_where/out/v1_seed.txt`).

Arms with the same teacher and the same seeds share most of that luck. At the last checkpoint, seeds 951 to 954 minus seeds 955 to 958 read +0.00113 on validation for the reference, and +0.00082 and +0.00085 for routed experts and depth 7, two arms of 19 November ([optimiser-and-modern-tricks](../optimiser-and-modern-tricks/); `lab/nn_x/w12/moe/logs/seedluck.txt`). A seed-paired read cancels it, so the same-teacher reads of this page stand as recorded.

Re-reading a same-teacher arm against the mean of the 70 pools counts the luck twice. The two arms of 19 November read +0.00070 and +0.00045 that way, and +0.00006 and -0.00001 at 8 fresh seeds.

The rows with another teacher, students as the next teacher and the cross-fitted teacher, may not share it: on the same comparison the width-192 recipe with the 25-member teacher read -0.00051, one comparison only. Their validation differences against the reference can be up to about 0.0006 too low. Against the mean of the 70 reference pools at the last checkpoint no row reads above the reference recipe by more than 0.6 SE on either sample, so the verdicts stand (`lab/nn_x/w12/moe/logs/fair_all_lead.txt`):

| Row with another teacher | Against the mean of the 70 pools, validation / holdout |
|---|---|
| 25-member teacher, width 192 | +0.00017 ± 0.00048 / +0.00015 ± 0.00028 |
| students as the next teacher, weight 0.5 | -0.00013 ± 0.00046 / +0.00016 ± 0.00027 |
| students as the next teacher, weight 0.3 | -0.00023 ± 0.00053 / -0.00014 ± 0.00030 |
| cross-fitted teacher, weight 0.5 | -0.00015 ± 0.00047 / -0.00059 ± 0.00026 |
| cross-fitted teacher, weight 0.8 | -0.00096 ± 0.00052 / -0.00084 ± 0.00030 |

### Later reads: paired pool differences

Each difference is against a reference on the same seeds. The 8-seed rows compare with width-192 students at teacher weight 0.5, the cooldown rows with plain cooldown re-runs of the same students. The first row comes from the decision ledger (a working record, not published).

| Arm | Member difference, validation / holdout | Pool difference, validation / holdout | Verdict |
|---|---|---|---|
| teacher weight 0.8, width 192, 8 seeds | +0.00126 ± 0.00033 / +0.00078 ± 0.00019 | -0.00049 ± 0.00048 / -0.00029 ± 0.00026 | closed |
| weight 0.8 in the cooldown only, 16 cooldown re-runs | +0.00061 ± 0.00023 / +0.00061 ± 0.00013 | -0.00047 ± 0.00039 / -0.00059 ± 0.00021 (pooled z -2.88) | closed |
| weight 0.3 in the cooldown only, 16 cooldown re-runs | | +0.00011 ± 0.00032 / +0.00009 ± 0.00018 (randomization p 0.31) | closed |
| pseudo-feeds from user histories as extra training lists, width 192, 8 seeds | +0.00046 ± 0.00031 (validation) | +0.00054 ± 0.00047 / +0.00010 ± 0.00027 (randomization p 0.19) | failed its rule; in pick 2 from 2026-11-19 |

### Student recipe arms, 14 to 16 November

From 14 to 16 November the team changed one knob at a time on the width-192 student. The base was the recipe of `kd05w192`: the 11-member teacher at weight 0.5, 2 blocks, 4 heads, one history layer, dropout 0.1, weight decay 0.02, auxiliary-loss weight 0.5 (0.1 by the end of the cooldown), retrieval of 16 history items and 10 epochs. Each arm trained from scratch under a rule written first in the decision ledger. The [glossary](../../docs/glossary.md) maps the `kw*` codes.

| Change to the width-192 student (code) | Read on | Read | Validation / holdout | Pooled z |
|---|---|---|---|---|
| width 224 (`kd05w224`) | 11-14 | pool, 4 seeds | +0.00026 ± 0.00058 / +0.00016 ± 0.00032 | +0.68 |
| width 256, 25-member teacher (`kd05t25w256`) | 11-14 | pool, 2 seeds | 0.40014 against 0.40075 at width 128, validation | |
| width 256 with dropout 0.2 (`kw256d2`) | 11-15 | pool, 4 seeds | -0.00163 ± 0.00058 / -0.00075 ± 0.00032 | -3.64 |
| 3 blocks, 25-member teacher (`kd05t25w192b3`) | 11-14 | pool, 2 seeds | 0.40026 against 0.40075 at width 128, validation | |
| 2 history layers (`kw_hl2`) | 11-14 | member, 2 seeds | +0.0002 / -0.0010 | |
| feed-forward width 4 times the model width, not 2 (`kw_ff4`) | 11-14 | member, 2 seeds | -0.0011 / -0.0004 | |
| 6 heads (`kwh6`) | 11-15 | pool, 4 seeds | -0.00053 ± 0.00056 / +0.00057 ± 0.00031 | +0.61 |
| richer similarity bias in the history attention (`kwsb2`) | 11-15 | pool, 4 seeds | -0.00053 (z -0.93) / -0.00042 (z -1.30) | -1.58 |
| retrieval of 32 similar history items, not 16 (`kwr32`) | 11-15 | pool, 4 seeds | -0.00098 (z -1.62) / -0.00043 (z -1.27) | -2.04 |
| hashed item-ID table (`kw_id`) | 11-14 | member, 2 seeds | -0.0057 / -0.0049 | |
| GRU history encoder (`kd05w192gru`) | 11-14 | pool, 4 seeds | -0.00000 ± 0.00059 / -0.00018 ± 0.00033 | -0.38 |
| 20 epochs (`kw_e20`) | 11-14 | member, 2 seeds | -0.0008 / -0.0004 | |
| dropout 0 (`kw_drop0`) | 11-14 | member, 2 seeds | -0.0006 / -0.0002 | |
| dropout 0.2 (`kwd2`) | 11-15 | pool, 4 seeds | -0.00068 ± 0.00056 / -0.00046 ± 0.00032 | -1.88 |
| history events dropped at rate 0.3 (`kwh30`) | 11-15 | pool, 4 seeds | -0.00098 ± 0.00058 / -0.00066 ± 0.00033 | -2.62 |
| weight decay 0 (`kwwd0`) | 11-16 | pool, 8 seeds | -0.00134 ± 0.00056 / -0.00011 ± 0.00031 | -1.95 |
| weight decay 0.01 (`kwwd01`) | 11-16 | pool, 8 seeds | -0.00032 ± 0.00051 / -0.00021 ± 0.00028 | -0.97 |
| weight decay 0.05 (`kwwd05`) | 11-15 | pool, 8 seeds | -0.00043 ± 0.00049 / -0.00063 ± 0.00027 | -2.31 |
| weight decay 0.10 (`kwwd10`) | 11-15 | pool, 4 seeds | -0.00169 ± 0.00061 / -0.00091 ± 0.00034 | -3.86 |
| auxiliary-loss weight 0 (`kwaux0`) | 11-15 | pool, 8 seeds | -0.00135 ± 0.00058 / -0.00128 ± 0.00033 | -4.38 |
| auxiliary-loss weight 1.0 (`kwaux1`) | 11-16 | pool, 4 seeds | -0.00023 ± 0.00055 / +0.00028 ± 0.00031 | +0.34 |
| teacher temperature 0.7 (`kwT07`) | 11-14 | pool, 2 seeds | -0.00014 ± 0.00067 / -0.00032 ± 0.00038 | -0.74 |
| teacher temperature 1.5 (`kwT15`) | 11-14 | pool, 2 seeds | +0.00001 ± 0.00069 / -0.00035 ± 0.00039 | -0.61 |
| random sub-lists, each negative kept with a probability drawn from 0.5 to 1 (`kwsub5`) | 11-14 | pool, 2 seeds | -0.00035 ± 0.00066 / -0.00031 ± 0.00038 | -0.96 |
| the same, probability from 0.2 to 1 (`kwsub2`) | 11-14 | pool, 2 seeds | -0.00014 ± 0.00068 / -0.00047 ± 0.00039 | -0.99 |
| mixed teacher: half the 25-member teacher, half the 8 first width-192 students (`kdmixw192`) | 11-14 | pool, 4 seeds | -0.00030 ± 0.00057 / +0.00013 ± 0.00032 | -0.08 |

A pool row is the arm's pool minus the reference's pool on the same seeds, at the checkpoints the holdout selected; a member row is the mean difference per student. No arm reached the bar for more seeds, both differences ≥ 0 and pooled z ≥ 1.6, and on 15 November the ledger closed this class of ideas, closed under click labels and retried under distillation. The teacher weights are in the two tables above.

Students made less alike still lost. Weight decay 0 and 0.01 (16 November, a stated exception) lowered the mean member correlation to 0.9563 and 0.9643 from 0.9723, but each student lost -0.00195 ± 0.00037 and -0.00080 ± 0.00033 on validation. The GRU students, as strong as the transformer students, correlated only 0.982 to 0.984 with the W8 pool, yet 4 of them added to W8 read +0.00018 ± 0.00030 / -0.00016 ± 0.00016 against 4 more transformer students.

The 4-seed pool rows compare seeds 951 to 954 with the same seeds of the reference, the pool that ranks 3rd of 70 under the first table. Same-teacher arms share its luck, so these seed-paired reads stand, and a re-read against the mean of the 70 pools would count the luck twice. The mixed teacher is another teacher; against the mean of 200 random 4-student pools of the 16 width-192 students it read +0.00003 / +0.00014 (pooled z +0.44).

Paired by seed at the last checkpoint, width 224 reads +0.00031 ± 0.00057 / +0.00021 ± 0.00032. It got a second try on 18 November as all-data try files, FD16P20 with 8 or 16 students replaced by width-224 students or 16 added: +0.00005 ± 0.00030, +0.00006 ± 0.00037 and +0.00017 ± 0.00025 against FD16P20 on validation, not scored.

### Width 192 and the board

By its own rule the width-192 pool failed: +0.00049 on validation against a bar of +0.0005. The two width-192 arms, one per teacher, were then pooled after the fact, and the ledger said so before any fine-tune ran. Together they read +0.00107 ± 0.00031 on the holdout over the width-128 pool, and the board settled it.

| File | Members | Public |
|---|---|---|
| best click-label file (C3) | 28 click-label rankers | 0.393454 |
| W8 | 8 full-view students of width 192; fine-tuned and train-only halves at 0.5 each | 0.394486 (+0.001032) |
| W16 | 16 such students | 0.39448337 (-0.000003 against W8, paired SE 0.00045) |
| S16 | 8 wide plus 8 narrow students | 0.3940782 |
| FDXH16P20:ps (2026-11-18) | all-data file with the 8 students of the 11-member teacher replaced by pseudo-feed students | 0.3947873 (0.00010 below FD16P20) |
| FDX16P20:ps (2026-11-19) | all-data file with all 16 full-view students replaced by pseudo-feed students; pick 2 | 0.39526061 (0.00037 above FD16P20) |
| FDXU32P20:ps (2026-11-19) | FD16P20 plus the 16 pseudo-feed students | 0.39524613 (0.00035 above FD16P20) |

Every full-view student in the final files was trained this way, at width 192 and teacher weight 0.5.

The pseudo-feed students reached the board in three all-data files. Against FD16P20, replacing the 8 students of the 11-member teacher read +0.00028 ± 0.00029 on validation and -0.00010 on public, replacing all 16 read +0.00015 ± 0.00036 and +0.00037, and adding the 16 read +0.00029 ± 0.00026 and +0.00035. The two replacement files differ by about 0.0005 on public, inside the paired noise. The second file, FDX16P20:ps, became pick 2 ([what worked, change 18](../../docs/06-what-worked.md#18-pseudo-feeds-for-the-full-view-students)).

### The audit

The audit found no bug and no leak; three independent recomputations reproduced every pool.

It also found W8 to be a favourable draw of its recipe. W8 minus the second 8-pool of the same recipe reads pooled z +1.60, a value reached by 2.4% of null splits. Under the null of two 8-pools of one recipe the pooled z has sd 0.80, so the rule "both ≥ 0 and pooled z ≥ 2" passes 0.25% of nulls.

### A teacher that sees later snapshots

A privileged teacher, called the hindsight teacher in the glossary, sees the user's later snapshot. On the 4,815 holdout feeds it covers, it read +0.037 ± 0.0025 NDCG over the teacher.

Events before the first interaction with a candidate added -0.00026 ± 0.00036, events after it +0.03883 ± 0.00248. The information follows from the click, so it was never used ([the later-snapshot leak](../../docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)).

## What it taught

All teachers behaved as one function: students were equal for teachers between 0.3210 and 0.3245 holdout NDCG.

The teacher's softmax is nearly flat, with entropy 4.41 of 4.90 nats and 0.032 of the mass on clicked items. The in-sample copy of the label is 0.2 to 0.7% of the target mass. Cross-fitting therefore had little to remove and only weakened the teacher (holdout 0.32100 against 0.32370).

One reading, not tested: the teacher lowers label noise, and all teachers pull the students toward the same function.

Twice a heavier teacher made better members and a worse pool: students that follow the teacher closely are more alike. Weight decay 0.05 did the same (member +0.00056, pool -0.00043 on validation). Under distillation, diversity came only from seeds and from a different input view ([popularity-views](../popularity-views/)).

The closure of 2026-11-11 covered only the form it tested. The ensemble form became the largest late gain. Pseudo-feeds went a smaller way along the same path: closed by their rule on 2026-11-17, they reached pick 2 two days later on public readings inside the noise.

Not run:

- distillation on the unlabelled validation and test feeds;
- a teacher weight per feed;
- a clean form of training on random sub-lists. The form that ran kept 1.88 positives per sub-list against 1.22 on real short feeds.

## Files

`code/`

- `dump_teacher.py`: step 1, teacher logits on every train candidate.
- `train.py`: step 2, the trainer; the distillation lines are marked 'w12 KD'.
- `build_w.py`: step 3, the first student-only files W8 and S16.
- `arm_read8.py`, `aa_null.py`: the 8-seed read-out and the null of its rule.
- `assemble_oof.py`: the cross-fitted teacher.
- `hindsight_01_build.py`, `hindsight_04_prepost.py`: the privileged teacher and its check.

`results/`

- `teacher_weight_0.8_in_cooldown_readout.txt`: the heavier teachers.
- `recheck_headline_pools_recomputed.txt`, `recheck_pool_size_saturation.txt`, `pooled_z_rule_null.txt`: the audit.
- `hindsight_teacher_eval.txt`, `hindsight_pre_post_click.txt`: the privileged teacher.

### Raw material and records

- Lab folders: `lab/nn_x/w12/kd/`, `lab/nn_x/w12/recheck/`, `lab/nn_x/w12/hindsight/`, `lab/nn_x/w12/pseudokd/` (the pseudo-feed students and their rule; `train_full_pk.py` trains the all-data students of pick 2, which the package trainer cannot), and the student runs under `lab/nn_x/w5/`.
- `lab/nn_x/w5/lane.sh` holds the flags of every student recipe arm.
- The cross-fitted teacher: `lab/nn_x/w12/kd/run_oof.sh` (15 fold teachers, then `assemble_oof.py`) and the fold-teacher strength in `lab/nn_x/w12/recheck/kd_literature/bagging.txt`.
- The rank of the width-192 reference among its 70 4-seed pools: `lab/nn_x/w12/moe/why/verify_where/out/v1_seed.txt`; seed luck by recipe: `lab/nn_x/w12/moe/logs/seedluck.txt`; reads against the mean of the 70 pools: `lab/nn_x/w12/moe/logs/fair_all_lead.txt`; seed-paired reads at the last checkpoint: `lab/nn_x/w12/moe/logs/paired_last.txt`.
- Records: the decision ledger and the experiment registry (working records, not published).
- In context: [the solution](../../docs/03-solution.md), [what worked](../../docs/06-what-worked.md).

The audit reports are in `lab/reports/RECHECK_1114/`, with the synthesis in `SYNTHESIS.md`. The nine checks:

- `kd_audit.md`: correctness of the distillation pipeline.
- `kd_reprice.md`: which closed ideas distillation changes.
- `kd_literature.md`: the distillation literature.
- `ft_redesign.md`: the fine-tune for students.
- `pool_composition.md`: which students belong in the pool.
- `lb_stats.md`: board statistics and private risk.
- `winner_model.md`: structural choices of the ranker.
- `long_bets.md`: multi-day bets.
- `forensics_skeptic.md`: a non-linear re-test of the board forensics.

The three reviews are `skeptic_duplicates.md`, `skeptic_feasibility_value.md` and `skeptic_leak_overfit.md`.
