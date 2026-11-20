# Lessons

In brief:

- Nineteen lessons, each from an incident that cost a day, a submission or a wrong closure.
- Worth repeating unchanged: one board test of a suspected leak with its drop rule written first (lesson 1), checkpoints chosen on held-out feeds (6), the public error computed before each submission (8) and the rule written before the run (11).
- From 17 to 19 November: the label-draw test (16), try files (17), the noise in public scores of near copies (18) and how to read a screen against a reference that shares its seeds (19).

## Notation

- "±" is one standard error of a paired per-feed difference, z is the difference divided by it, and a pair a / b is validation / holdout ([04-validation.md](04-validation.md)).
- Validation is the 14,423 labelled feeds of 22 and 24 September. The holdout is 30,000 random-order train feeds kept out of training. A witness is a labelled sample a change is judged on: validation, the holdout or one of three views built from them. The public board is not a witness.
- A member is one trained ranker, a pool the equal-weight mean of several members, a file one submission.
- A near copy is a pool that correlates at 0.995 or more with another and differs from it by at most 0.002 on validation. A model generation is a step of 0.002 or more.
- FD16P20 is the file of 20 students trained on all train feeds that was pick 1 until 18 November. Pick 1, FD16P20@m0.4_0.2, has the same members with the popularity-light weight raised to 0.4 on feeds of fewer than 50 candidates. Pick 2, FDX16P20:ps, is FD16P20 with its 16 full-view students retrained on all train feeds plus pseudo-feeds built from user histories.

Other terms are in [glossary.md](glossary.md); the experiments are told in [07-what-failed.md](07-what-failed.md), and how the work was run in [13-how-the-work-was-run.md](13-how-the-work-was-run.md).

## Leaks and data

1. Look for the label inside the inputs before the first model, and test a suspected leak on the board once.

   92.1% of a feed's positives are in the same user's next history snapshot, and counts over all snapshots had put about 0.002 of leak into the day-1 validation score. The board test of 12 November, its drop rule written first, read +0.00454 ± 0.00047 on validation and nothing on public, 0.392819 against 0.392820 for its base file ([the leak](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)).

2. Build a train row exactly the way a test row is built, and treat opposite signs on two witnesses as an alarm.

   Scores of the [trajectory encoder](glossary.md) for train rows raised the holdout by 0.006 to 0.0085 and lowered validation by 0.005 to 0.009. The encoder's item table held no train candidates, so on train rows missing values marked the positives. The rebuild was accepted once its zero-popularity train rows matched validation: mean z -0.153 against -0.146, where the leaky build read -1.46 ([failure 2](07-what-failed.md#2-encoder-scores-for-train-rows-first-build)).

3. Check what an oracle or a teacher is made of before believing its number.

   A teacher that saw the user's later snapshots read +0.037 ± 0.0025 on the feeds it covered. All of it came from events after the user's first interaction with the candidate, which is the label itself. An oracle for routing feeds with cold items, allowed to know which feeds hold only cold positives, read +0.025 on validation and was an echo of the label too.

4. Re-implement the metric and feed it ties.

   The metric puts negatives first among equal scores. A bfloat16 output head made ties and cost 0.001 to 0.003 on validation (the first model read 0.3705, then 0.3736 with a float32 head). Organiser feature f3 alone scores 0.2518 with its ties against 0.3175 with random tie-breaks. Submissions are written as tie-free within-feed ranks, and even so four exact float32 ties in 2.9 million rows once made the team's own gate reject a valid file.

   The competition page says IDCG@30 (the ideal ranking's score cut at 30 positions), but a cut at 30 would have predicted 0.322 and 0.331 on public for two files that scored 0.380 and 0.388: the board scores the full list ([the metric](01-competition-and-data.md)).

## Noise and witnesses

5. Run same-recipe pairs before the first comparison, to learn the smallest gain one run can show.

   Fourteen pairs of same-recipe runs with different seeds showed that the per-feed standard error is honest (root-mean-square z 1.04) and that one member against another can show nothing below 0.0029 on validation ([noise](04-validation.md#noise)). Fifteen later seed pairs agreed on the holdout as well (root-mean-square z 0.89).

   On day 1 an input was accepted at validation z 1.44 on one seed, under a rule that passed a change if either of two witnesses reached 2 standard errors. That rule lets through 1.6 times the nominal rate of null changes. A 3-against-3 retest of the input read +0.00119 ± 0.00078 before the fine-tune and -0.00031 ± 0.00075 after it.

6. Validation predicts the level of a model generation but cannot order near copies.

   Steps of +0.003 to +0.010 between generations transferred to public one for one. Seven click-label pools of one generation spanned 0.00198 on validation and 0.00074 on public (rank correlation +0.04). The two picked as best on validation landed 0.0009 to 0.0010 below forecast, the expected excess of the best of 5 to 50 near copies ([noise](04-validation.md#noise)).

   Choose checkpoints and members on a sample that is not read afterwards, here the holdout, and subtract the expected excess of the best of K near copies from any validation maximum. Lesson 19 adds a limit: read an arm at the checkpoint its shipped member would keep.

7. Find out which changes each witness is biased on before giving it a veto.

   The holdout of lesson 6 read the two popularity-light admixtures (members whose 8 popularity inputs are replaced by one exposure count) at -0.00063 and -0.00073 where public read +0.000523 and +0.000634. Its popularity inputs are 1.75 days old, against 9.34 on public. List for each witness the inputs that are fresher or staler on it than on test, and take away its vote on changes to them ([the audit](04-validation.md#rules-written-before-the-read)).

8. Compute the public error before each submission: the board cannot separate files 0.0005 apart.

   Scaled from the validation per-feed differences to the 8,562 public feeds, the paired public error between two late files was 0.0003 to 0.0006. Of the seven files of the final family scored by 16 November (0.39436 to 0.39501 on public), the one with the lowest validation score had the highest public score ([risks](11-final-picks.md#risks)).

   Before the five submissions of 16 November the ledger recorded that five such readings would put at least one score above the team best, then 0.3950094, with probability about 0.7 from noise alone, which would not be progress. None of the five did. Four of the ten near copies scored from 17 to 19 November did (lesson 18).

## Rules and references

9. Judge a change at the level you ship: member, pool or file.

   Averaging plateau checkpoints moved a member by +0.00041 / +0.00072 and the 7-member pool by -0.0025 / -0.0016: it removed the noise that makes seeds differ (seed correlation 0.975 against 0.941). Teacher weight 0.8 moved a student by +0.00126 ± 0.00033 on validation and the 8-student pool by -0.00049 ± 0.00048.

   One level further up, a cooldown on stale inputs passed as a pool (stale validation +0.00120 ± 0.00045) and lost as a file (-0.00047 ± 0.00035 on validation), with the same shape on three member sets ([failure 19](07-what-failed.md#19-a-cooldown-on-stale-popularity-inputs)). On top of the 0.2 popularity-light part already in the file, the stale view keeps 43% of its effect: the two do much of the same job.

10. A reference pool is itself one draw.

    Four popularity-free students put a file above 70 of 70 control files; four fresh seeds of the same recipe were above 16 of 70 ([the replication](04-validation.md#rules-written-before-the-read)). The 16-student file that set the bar for larger pools sat about 0.0003 above its family's level on validation, so pools of 32 and 48 failed the rule for larger pools by diluting a lucky draw ([failure 18](07-what-failed.md#18-growing-the-ensemble-past-16-members)).

    When two pools differ in members, changing one member re-draws 49% to 75% of the variance of their difference. When the pools share no seeds, compare with the mean of same-size subsets; when they share seeds and a teacher, compare seed by seed (lesson 19). Replicate on fresh seeds before adopting.

11. Write the rule before the run, and check that it can return every verdict.

    The distillation rule was written at 03:14 UTC on 14 November and read at 05:18 UTC with nothing to argue about. The replay test of longer histories assumed about 11,600 qualifying feeds and found 3,235, on which even events already in the model read z 1.58: it could only end undecided.

    Count the qualifying sample first. Log a threshold moved after it fired, such as the gate's leak probe from z 2 to z 3, as a change made after the fact ([rules](04-validation.md#rules-written-before-the-read)).

12. An in-sample read of a rule chosen from a scan is not evidence.

    A rule raising the popularity-light weight to 0.4 on feeds of 50 or more candidates read +0.00026 to +0.00071 on validation in 6 of 7 member compositions. Chosen and scored on separate user folds, it read +0.00002 ± 0.00027 on the members of FD16P20, and the paired public test read -0.00015. The same scan on a placebo with nothing to find reads +0.00057 (95th percentile +0.00082): that is the bar.

## Experiments

13. A cheap proxy needs a positive control.

    A residual test on a finished pool closed about 300 feature rows, yet on a block worth +0.00852 as trained inputs it read +0.00086 to +0.00222 on the holdout and nothing on validation. A tree proxy read +0.0022 to +0.0031 for an offset that cost the neural ranker 0.0005 ± 0.0005 on validation. A 2-epoch screen read +0.0012 on validation for a change worth +0.0004 at 10 epochs.

    In reverse, the best re-opened input was used by the trained model (+0.00179 against a permuted copy) and added nothing net: [pooled z](glossary.md) +0.59 and +0.62, public -0.00054 ± 0.00050. It replaced information the ranker already had ([failure 14](07-what-failed.md#14-the-residual-gate-and-the-trained-input-that-passed-every-gate)).

14. A setting tuned at one training length is a dose, and the dose changes with the length.

    A 30% share of random-order feeds in the cooldown (the last 20% of steps) meant about two passes over that pool at 2 epochs, where it was tuned, and about 9 at 10 epochs. All six members fell from 0.391 to 0.364 on validation ([failure 3](07-what-failed.md#3-a-cooldown-dose-tuned-at-2-epochs-and-used-at-10)); the adopted share is 2%. Longer runs failed the same way: 15, 20 and 30 epochs read -0.0006, -0.0008 and -0.0016 on validation against 10.

15. A closure covers only the form that was tested, and the records must say which form.

    Distillation was closed on 11 November after one 2-epoch teacher read +0.00064 on validation (z 1.6). On 13 November a scan of 211 outside techniques still gave it a probability of 0.02 of adding +0.0005 at pool level. With an 11-member teacher it then gave +0.00257 per student on validation, and the first student file scored +0.001032 on public over the best click-label file ([distillation](06-what-worked.md#7-distillation-from-an-ensemble-of-earlier-members)).

    On 16 November a list of ideas "never run" held three that had been run and closed days before, copied from stale rows of the idea table. Record the tested form with each closure, and rebuild any summary from the newest ledger entry.

    On 19 November the [lead session](glossary.md) proposed a cross-fitted teacher as untested. It had been tested and closed on 14 November ([13](13-how-the-work-was-run.md#what-the-setup-lacked)). Five ideas that were closed only on paper, by a proxy or in a flawed form are listed in [12, section 6](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed).

16. Run the label-draw test before aiming a loss at the biggest misses, and stop if re-drawn labels put the same share of the loss there.

    The test re-draws the clicks from the model's own softmax and measures the loss again. On 17 November the hardest 20% of feeds held 27.5% of the loss, and 27.6% under re-drawn clicks: what they lose is mostly the draw of the label.

    Online hard-example mining (OHEM) and group DRO, a loss on the worst group of feeds, stopped at probes. The one loss run anyway weighted each feed by the student's gap to the teacher and down-weighted surprising clicks. It read -0.00058 ± 0.00047 / -0.00074 ± 0.00028 ([error analysis](../experiments/error-analysis/)).

## The last submissions

17. Try files are cheap information on spare submissions, and a single one says nothing about its idea.

    From 17 November, all-data [try files](glossary.md) carried ideas with a positive validation mean, including some that had missed their rule. A per-feed loss weight by feed size scored 0.00007 below its matched control, and students trained with pseudo-feeds from user histories 0.00010 below FD16P20 after +0.00028 on validation.

    The one file that rose by 18 November, 0.00032 above FD16P20, came from the side of a mixing rule that public had not read: popularity-light weight 0.4 on short feeds. On validation 0.4 is a single high point between lower readings at 0.3 and 0.5. The file became pick 1 after its public score, with no rule written for it ([change 17](06-what-worked.md#17-popularity-light-weight-04-on-short-feeds)).

    Both changes to the popularity-light part on long feeds that were sent, before and after it, landed about 0.0004 below their validation reading. Two files cannot tell that apart from noise.

    On 19 November the file with all 16 full-view students replaced by pseudo-feed students scored 0.00037 above FD16P20 and became pick 2, the team's best public score. The file with 8 of the 16 replaced had scored 0.00010 below. The two differ by about 0.0005, inside the paired noise, so neither reading is evidence for or against pseudo-feeds.

18. Read public differences between near copies as noise: their paired error is 0.0002 to 0.0006.

    Ten near copies of FD16P20 scored from 17 to 19 November read -0.00014 to +0.00037 against it, at paired public errors of 0.00028 to 0.00047. Four scored above the earlier team best of 0.3950094: pick 2 at 0.39526061, a file with 32 full-view students at 0.39524613, pick 1 at 0.3952092 and A1, a variant of pick 1, at 0.395036.

    Over the 17 scored files of the final family (sd 0.00025), the best of 17 equal files would sit about 1.8 sd (0.00044) above the mean; pick 2 sits 1.6 sd and pick 1 1.4 sd above it. When pick 1 was chosen it was the best of 14, at 1.9 sd. The team counts on about +0.00005 from pick 1 over FD16P20 on private, not on its public lead ([risks](11-final-picks.md#risks)).

## The last screen

19. Compare arm and reference seed by seed under the same teacher, read the arm at the checkpoint the shipped member keeps, and do not correct a paired read afterwards against an average reference.

    The student screens of 14 to 19 November compared an arm's seeds 951 to 954 with the same 4 seeds of the reference recipe. Of the 70 possible 4-seed pools of the 8 stored references, that one ranks 3rd on validation (+0.00057 over their mean of 0.40077) and 37th on the holdout ([seeds shared](04-validation.md#seeds-shared-by-arm-and-reference)).

    An arm with the same teacher and seeds shares most of that luck. On validation, seeds 951 to 954 read above seeds 955 to 958 by +0.00113 for the reference, +0.00082 for routed experts and +0.00085 for depth 7 (each ± 0.00056 to 0.00059). A seed-paired read cancels it.

    An arm with another teacher may not share it (the 25-member teacher read -0.00051 ± 0.00056 on the same comparison), and its seed-paired read can be up to about 0.0006 too low.

    Re-reading a same-teacher arm against the mean of the 70 pools counts the luck twice. The two arms of 19 November rose that way to +0.00070 ± 0.00046 and +0.00045 ± 0.00047 and then read +0.00006 ± 0.00047 and -0.00001 ± 0.00048 at 8 fresh seeds.

    The real flaw of that screen was the checkpoint. Holdout selection picked steps 85,000 to 90,000 for 4 of its 12 arm runs, while all-data students keep the last step. At the last checkpoint and paired by seed the three arms read +0.00009, -0.00093 and -0.00016 on validation (± 0.00056), against -0.00086, -0.00144 and -0.00069 at the selected checkpoints. Only fresh seeds under a rule written first settled it ([phase 9](05-timeline.md#phase-9-mixture-of-experts-and-depth-19-november-from-0400)).
