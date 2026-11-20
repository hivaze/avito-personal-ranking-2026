# Why the score stopped here

In brief:

- On the public board of 2026-11-19 the best file, FDX16P20:ps (0.39526061), is 0.0030 below the leader and 0.0028 below place 2.
- A judgement written into the records on 18 November gives the largest probability to better modelling or better use of the training data (P 0.64), then to public luck and the selection of the best of several uploads (P 0.12). Label noise and data limits hold for every team and explain the team's stall, not the gap.
- Validation luck sits mainly in the final files. Part of it comes with the seeds: recipes that share seeds and teacher share it, so a comparison paired seed by seed cancels it.
- The 8-seed rule passes 56% of true +0.0005 effects, and distillation and pseudo-feeds were closed before they reopened. Yet a scan of all closures finds positive reads at the rate of noise, and positive first reads that were re-tested kept nothing on average, so the scan finds no sign of lost gains, though its bound is loose (up to 13 of 93 rows).
- Five ideas never had a fair training run. A judgement of 19 November, with no experiment behind it, puts the chance that at least one of them is worth +0.001 or more at about 0.25, and expects none to be the whole gap ([section 6](#6-ideas-the-search-may-have-missed)).

Notation follows [04-validation.md](04-validation.md#notation-and-files): differences are paired per feed, "±" is one standard error (SE), and a / b is validation / holdout. Validation has 14,423 labelled feeds. The holdout is 30,000 train feeds that the first detector marked as random-order, kept out of training; public scores 8,562 test feeds, private 20,033.

P is a subjective probability that the lead session wrote into the records on 18 November ([08-outside-ideas.md](08-outside-ideas.md#hypotheses)). No experiment produced it, and it was not re-estimated after the board of 19 November. In section 6, P is a separate judgement of 19 November, defined there. Other terms are in the [glossary](glossary.md).

## The gap

On 2026-11-19 the leader stood at 0.3982556 on public, unchanged since 12 November, with place 2 at 0.3980582 and place 3 at 0.3968832 ([05-timeline.md](05-timeline.md#board-snapshots)). The paired SE between files of different teams is put at 0.0012 to 0.0018 on public, an estimate, since other teams' files are not available, so the leader's margin of 0.00305 over pick 1 on 18 November was 1.7 to 2.5 such errors.

The team's own score had stopped earlier. In the 18 files scored after W8P20 (15 November) the best public score rose by 0.00025, below the 0.0011 the board needs to separate two near copies at 80% power. A near copy correlates at 0.995 or more with another file within each feed and differs by at most 0.002 on validation.

## 1. Noise in the labels

The label is one bit per candidate: 74.5% of validation feeds have exactly one positive, and 1.58% of validation candidates are positive. The click draw, the chance of which candidate received the click, is 67% of the per-feed variance of NDCG, and on its own it moves a file's mean by ±0.0017 on validation, ±0.0021 on public and ±0.0014 on private ([error-analysis](../experiments/error-analysis/README.md#how-much-of-the-error-is-noise-2026-11-18)).

The label re-draw test re-draws the clicks from the model's own softmax and computes the loss again. On 17 November the hardest 20% of validation feeds held 27.5% of the loss, and 27.6% under re-drawn clicks. Pick 1 lies 0.0017 above its mean under 200 re-draws (0.40124 on validation, sd 0.00176). Its scores are calibrated, and a perfect correction of every detected bias would add +0.00035 on validation ([02-what-the-data-told-us.md](02-what-the-data-told-us.md#where-the-ranking-error-sits)).

Label noise is a weak explanation of the gap. The test checks calibration and sets no ceiling: a boosted-tree model at 0.370 on validation passes it too. The draw is the same for every team, and the leader's margin equals what a model gains whose logits are 2.5% larger and still calibrated (`lab/nn_x/w12/noise/quantify/out/a_ceiling.txt`). The error analysis puts perfect soft labels at +0.0001 to +0.0008 per file and concludes that cleaner labels would not close the 0.003.

The cost of label noise was in measurement. Even paired per feed, one member against another cannot show less than 0.0029 on validation at 80% power, and two near-copy pools need 0.00086 on validation, 0.00112 on public and 0.00073 on private ([04-validation.md](04-validation.md#smallest-detectable-effect)). Label noise reaches the gap only through the noise of the board ([section 3](#3-luck-in-measurement)).

## 2. The data runs out

Less data hurt, and more capacity did not help. Training on 80% of the users cost a member 0.00275 ± 0.00049 / 0.00217 ± 0.00028 and a 3-member pool 0.0017 / 0.0019 ([distillation](../experiments/distillation/README.md#the-cross-fitted-teacher)). Language-model trunks of 0.6 to 9 billion parameters read 0.0038 to 0.0053 below the ranker on validation.

The inputs look exhausted. None of 70 attributes told a missed positive from the negatives within 5 ranks of it (local AUC 0.492 to 0.504 on validation), and of 11,617 generated features 1.02% passed the screen, against 1.01% of label-free noise features.

A channel here is a source of information in the files that the model does not yet use. Apart from a leak that scored nothing on public, every channel inside the files probed on 18 November read at most +0.00003 NDCG on validation on top of pick 1 ([08-outside-ideas.md](08-outside-ideas.md#probe-results-on-the-best-file)).

What the data lacks is large. There is no item content, and 42.8% of public and 45.6% of private candidate rows hold an item that occurs in no history. The 8 popularity inputs are worth 0.0074 NDCG on the holdout at 1.75 days old, 0.0023 on validation at 5.69 days and 0.0009 on validation replayed with inputs as stale as on private rows (11.23 days); public rows are 9.34 days old.

The data limit explains the gap poorly as well, because every team has the same files. It needs a leader with inputs others lack: an entry linked to the organiser, such as a reference run on production inputs (H-C, P 0.08; the board marks no such entry, and nothing in the files can test it), a small channel shared by many teams (H-E, P 0.04) or fresh popularity from outside the files (H-F, P 0.03).

The probes cannot see effects below about +0.0002, so its bound covers only larger channels.

What remains is the narrow claim of the [README](../README.md#why-the-score-stopped-here): nothing more could be found in these inputs as given. Staleness also lowers private for whatever leans on popularity: under private-like lag the 8 full-view students lost 0.0033 ± 0.0008 on 13,424 holdout feeds, the popularity-light pool 0.00049 ± 0.00038 ([02-what-the-data-told-us.md](02-what-the-data-told-us.md#the-popularity-light-view)).

## 3. Luck in measurement

### On validation

Validation reads model generations and not near copies: seven click-label pools of one generation spanned 0.00198 on validation and 0.00074 on public, with a rank correlation of +0.04 ([04-validation.md](04-validation.md#selection-on-validation)). The two files picked as best on validation on 12 November landed 0.0010 ± 0.0005 and 0.0009 ± 0.0004 below their public forecast.

The final files carry favourable draws. FD16P20 reads 0.40271 on validation, where pools of 32 and 48 students of its recipe read 0.40237 and 0.40222, and its four popularity-light students beat 472 of 495 possible draws ([07-what-failed.md](07-what-failed.md#18-growing-the-ensemble-past-16-members)).

Seed luck was measured on 19 November. Two capacity arms were screened on students that day: routed experts (a mixture-of-experts layer) and depth 7 (7 blocks in place of 5). Seeds 951 to 954, the 4-seed reference of the student screens, rank 1st to 3rd of the 70 possible 4-seed pools of the 8 stored seeds, depending on the checkpoint, about +0.0006 above their mean on validation.

At the last checkpoint, seeds 951 to 954 minus seeds 955 to 958 read +0.00113 ± 0.00056 for the reference, +0.00082 ± 0.00056 for routed experts and +0.00085 ± 0.00059 for depth 7.

With the 25-member teacher the same difference is -0.00051 ± 0.00056 ([04-validation.md](04-validation.md#seeds-shared-by-arm-and-reference)). The luck seems tied to the data order or the targets that go with a seed and teacher, so a read paired seed by seed cancels most of it. The paired 4-seed reads of routed experts and depth 7 at the last checkpoint, +0.00009 ± 0.00056 and -0.00016 ± 0.00056 on validation, agreed with 8 seeds: +0.00006 ± 0.00047 and -0.00001 ± 0.00048.

Re-read after the fact against the mean of the 70 pools, the two arms rose to +0.00070 ± 0.00046 and +0.00045 ± 0.00047. The arms were trained on the lucky seeds and the mean reference was not, so the re-read counted the arms' share of the seed luck as a gain.

By the same re-read width 224 and the pseudo-feed students rose to +0.00092 and +0.00104, against +0.00031 and +0.00042 paired by seed ([optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/README.md#mixture-of-experts-re-tested-on-distilled-students-19-november)). The real flaw of the screen was the checkpoint: the holdout picked steps 85,000 to 90,000 for 4 of 12 arm runs, where the final files keep the last step.

Arms with a different teacher may not share the luck, so their seed-paired reads can be biased by up to about 0.0006; the one case on record is the cross-fitted teacher of [section 5](#5-a-better-model-the-search-missed). The rule that follows: pair by seed, do not correct a paired read afterwards against an average reference, and settle with fresh seeds.

As a cause of the gap, validation luck is weak: it changes which files are built and picked. The bar set by FD16P20's 0.40271 failed the 32- and 48-student files. The 24- and 48-student files, scored on 17 November, read +0.000013 and -0.00005 against FD16P20 on public.

### On the public board

Seventeen scored files of the final family span 0.39436 to 0.39526 on public (mean 0.39486, sd 0.00025), inside the paired SE of 0.0002 to 0.0006 between two of them. With one true level the best would sit about 1.8 sd above the mean; pick 2 sits 1.6 sd and pick 1 1.4 sd above it ([11-final-picks.md](11-final-picks.md#both-picks-are-among-the-best-of-17-near-copies)).

This selection works against the team. On 18 November pick 1 was the best of 14 near copies, 0.00041 above their mean; the leading score, that team's first entry, carries about no selection gain. Net of selection on both sides, the leader was about 0.0035 above the team's level, place 2 +0.0012 to +0.0020, place 3 +0.0004 to +0.0013, and places 4 to 7 within one cross-team SE (H-B, P 0.12).

The level line, public = 0.0074 + 0.963 × validation (residual sd 0.0006), shows no public luck for the final family: pick 1 sits 0.00021 and pick 2 0.00009 below it, the 16 all-data files 0.00031 on average ([11-final-picks.md](11-final-picks.md#all-data-files-sit-below-the-level-line)).

Private is less noisy: the click draw moves a file by ±0.0014 there. The feed-size mix alone projects private = public + 0.0026, but private inputs are staler (11.23 days against 9.34), and no labelled sample is as stale ([04-validation.md](04-validation.md#what-no-witness-saw)). About +0.00005 of pick 1's public lead over FD16P20 is expected on private.

Places 2 to 6 rose after that reading. On 19 November place 2 reached 0.3980582, 0.0002 below the leader and about 0.1 cross-team SE. That ends the isolation of the leader that H-B and H-C leaned on: two teams now stand 0.0028 to 0.0030 above the team. Public luck can explain the order of places 1 and 2, and among places 4 to 7 and the team. It cannot explain that margin.

## 4. Ideas closed too strictly

The rules kept false passes rare and paid in power. From 14 November a student change needed, at 8 seeds, both differences at or above zero and a pooled z, (z on validation + z on the holdout) / √2, of at least 2. That rule passes 0.25% of null arms and about 56% of true +0.0005 effects; the 2-seed screen it replaced passed 15% of nulls and 36% of true effects ([04-validation.md](04-validation.md#decision-statistics)).

Cheaper tests closed many rows. Of 211 outside techniques scanned on 13 November, 82 were weakly closed, among them 16 by a tree proxy, 9 with the old optimiser, 4 at a 2-epoch screen and 2 on one seed ([08-outside-ideas.md](08-outside-ideas.md#how-good-the-priors-were)).

Some closures rest on tests known to be flawed. The residual gate tested a candidate feature as one fitted weight on top of stored pool predictions. It closed about 300 candidate features (rows of the feature screens, not of the catalogue of 312) and sees only 10 to 26% of a true effect ([07-what-failed.md](07-what-failed.md#14-the-residual-gate-and-the-trained-input-that-passed-every-gate)).

Function-matching distillation ran in a flawed form, with 1.88 positives per sub-list against 1.22 on real short feeds, and its clean form never ran ([idea 2 of section 6](#idea-2-function-matching-on-random-variants-of-each-feed)).

Seed pairing may also have read arms with a different teacher up to about 0.0006 low, though against the mean of the 70 pools none of them rises above the reference by more than 0.6 SE ([distillation](../experiments/distillation/README.md#how-an-arm-was-read)).

The holdout veto was wrong for one family. With popularity inputs 1.75 days old against 9.34 on public, it read two popularity-light additions at -0.00073 and -0.00063 where public gave +0.000634 and +0.000523. An audit on 15 November found 31 closures in which a negative holdout decided while validation was not negative or not read; three popularity ideas reopened, and nothing outside that family changed status ([04-validation.md](04-validation.md#rejecting-on-the-holdout-alone)).

Closures did reopen. Distillation was closed on 11 November after a student of one frozen 2-epoch teacher read +0.00064 on validation (z 1.6). In ensemble form a student gained +0.00257 ± 0.00047 on validation, and the first student file scored +0.001032 on public over the best click-label file. Pseudo-feeds failed their 8-seed rule (randomization p 0.19) and reached pick 2 at +0.00037 over FD16P20 on public, 0.8 paired SE.

A scan of the whole catalogue on 19 November found no sign that closures were too strict as a rule ([REPORT.md](../lab/nn_x/w12/closures_scan/REPORT.md)). Of 112 closed model changes with a non-zero validation read and no clear loss, 52 read positive (46%, 95% interval 37 to 56%), compatible with the 50% of noise alone. Of the 93 closed rows that quote a standard error, 11 sit at z between 1 and 2, where noise alone puts 12.6.

Ten ideas with a positive first read (mean +0.00053) were re-read later in a stronger form, and their mean became -0.00001 ± 0.00013. A mixture fit puts the number of hidden real gains of +0.0003 at 0, with a 95% upper bound of 13 of the 93 rows. The bound is loose because one closure is weak evidence: at each row's own SE, a test at z 2 passes a true +0.0003 with median probability 0.13.

Two closures of the click-label period are probably real gains for a single member. R-Drop in the cooldown, a loss that makes two dropout passes agree, read +0.00057 ± 0.00030 / +0.00052 ± 0.00017 at the last checkpoint. An auxiliary-loss weight of 1.0 in the cooldown read +0.00038 ± 0.00017 / +0.00019 ± 0.00009 over 7 runs.

Their pool reads were flat, and neither ran on students. Two retrieval variants were closed although their 10-epoch rule said to go on to 6 seeds, and their final reads are not on record.

The case against is what the board said when closed ideas were scored anyway. Try files from 17 November carried closed ideas with a positive validation mean. A per-feed loss weight by feed size scored 0.00007 below its matched control on public, half pseudo-feed students 0.00010 below FD16P20, and a trained input from a two-step co-visitation walk 0.00054 ± 0.00050 below it.

The priors were low as well: on 13 November no technique had a probability above 0.035 of adding +0.0005 at pool level, and the whole re-open queue was worth +0.0003. An effect the size of the gap would not have hidden from a fair run: distillation passed its first fair rule clearly. Behind a weak test it did hide, for three days.

The scan and the try files cover ideas that ran, and the priors are numbers written without a run: distillation sat in the queue of 13 November at 0.02 before it paid ([08-outside-ideas.md](08-outside-ideas.md#how-good-the-priors-were)). [Section 6](#6-ideas-the-search-may-have-missed) takes up five ideas that were closed without a fair run and gives a later judgement that is higher than any of those priors.

## 5. A better model the search missed

The board suggests that room exists. On 18 November the top 8 places formed a continuum with gaps of 0.00008 to 0.00098 on public, and on 19 November place 2 rose from 0.3972749 to 0.3980582. Distillation added 0.0010 on public to the team's file in one day, so steps of that size existed. The records put better modelling or better use of the training data at P 0.64 (H-A), up from 0.55 on 14 November.

Every member of both picks is the same small transformer, and the picks correlate at 0.9949 per feed on test ([11-final-picks.md](11-final-picks.md#both-picks-are-one-model-family)). Boosted trees reached only 0.368 to 0.370 on validation, and students of another teacher, generation or width were as alike as two seeds (0.9718 against 0.9722).

A team merge is the only way left to add another model family: +0.0002 to +0.0004 over the better pool for an unknown partner, up to +0.0008 for one of equal strength ([08-outside-ideas.md](08-outside-ideas.md#options-left-open-on-18-november)). A merge means passing the solution to the partner, which rule 5.5 allows only until 2026-12-02 12:00 MSK ([11-final-picks.md](11-final-picks.md#what-would-change-the-choice)).

Distillation on the 43,018 validation and test feeds, with the teacher's scores as the only target, never ran. Cross-fitted on user halves, its expected value is about +0.0001, with P about 0.02 of +0.001 or more. Its precedent, teacher-only targets on 624,752 held-out train feeds, read +0.00069 ± 0.00057 / -0.00017 ± 0.00032 for the pool; one half here is 3.4% of that fold. The same precedent is the main evidence against [idea 2 of section 6](#idea-2-function-matching-on-random-variants-of-each-feed).

A cross-fitted teacher on train, which scores each feed with the 3 models that never saw its user, closed on 14 November: its student pool read 0.40079 / 0.32347 against 0.40134 / 0.32408 on seeds 951 to 954 ([distillation](../experiments/distillation/README.md#the-cross-fitted-teacher)).

With a different teacher that -0.00055 may carry seed luck. Against the mean pool it reads +0.00002 ± 0.00048 on validation at the same checkpoints and -0.00015 ± 0.00047 at the last, and -0.00059 ± 0.00026 on the holdout.

There was little for cross-fitting to remove: the teacher's softmax is nearly flat (entropy 4.41 of 4.90 nats), and the in-sample copy of the label is 0.2 to 0.7% of the target mass. Cross-fitting only weakened the teacher, whose fold members each saw 80% of the users: 0.32100 on the holdout against 0.32370.

Against a missed model stands the breadth of the search: of 312 catalogued ideas 155 had no effect and 45 hurt ([07-what-failed.md](07-what-failed.md)), and on 15 November no untested change was worth more than +0.0001 for a file. Heavier use of the validation labels (H-D, P 0.06) is worth at most +0.0003. The most probable explanation is the least specific: the records name no recipe of the right size.

That +0.0001 is an expected gain written on paper, with no run behind it. [Section 6](#6-ideas-the-search-may-have-missed) lists the five ideas that such paper closures leave most open. It expects none of them to be worth the whole 0.003, so the conclusion above stands.

## 6. Ideas the search may have missed

This section is a judgement made on 19 November by the lead session, the agent session that ran the work ([13-how-the-work-was-run.md](13-how-the-work-was-run.md#how-the-work-was-run)). No experiment stands behind it. It names five ideas that may be worth more than the records of [section 5](#5-a-better-model-the-search-missed) allow, and gives for each the reason it never had a fair run and the evidence against it.

The five come from the two largest effects measured late: the amount of labelled data and cleaner labels. Training on 80% of the users cost a [member](glossary.md) 0.0028 / 0.0022 ([section 2](#2-the-data-runs-out)), and [distillation](03-solution.md#distillation) gave one +0.00257 ± 0.00047 / +0.00200 ± 0.00028.

Ideas 1 to 3 touch one of the two. Idea 4 adds a side of the data the ranker does not model, and idea 5 re-tests closures made before distillation, the largest late change. Each was closed on paper, by a proxy or in a flawed form, never by a fair training run. The closures scan of [section 4](#4-ideas-closed-too-strictly) covers only ideas that ran.

P in this section is the subjective probability that an idea is worth +0.001 NDCG or more for a final file. The chance that at least one of the five is worth that much is put at about 0.25. None is expected to be the whole 0.003.

| Idea | What it would change | Why it never had a fair run | What speaks against it | P |
|---|---|---|---|---|
| [1. Next-event training of the history encoder](#idea-1-next-event-training-of-the-history-encoder) | a deeper [history](glossary.md) encoder inside the ranker, trained jointly with a next-event loss at every history position: 463.6 million events against about 5.5 million clicks | three catalogue rows closed it without training, one by an argument about corpus size and two by tests on stored predictions | the separate encoder ranks validation at 0.3590, a scorer of about 330 parameters at 0.3584; a second history layer added nothing for [students](glossary.md) | 0.10 |
| [2. Function matching on random variants of each feed](#idea-2-function-matching-on-random-variants-of-each-feed) | each student sees random subsets of a feed's candidates and cropped histories, each with the [teacher](glossary.md)'s target for that variant; the variants act as extra data | only a flawed form ran (pooled z -0.96 and -0.99 at sampling rates 0.5 and 0.2): its sub-lists kept every positive and reused inputs and teacher scores computed on the full feed | teacher-only targets on 624,752 extra real feeds read +0.00069 ± 0.00057 / -0.00017 ± 0.00032 for the [pool](glossary.md) | 0.07 to 0.10 |
| [3. Hindsight labels from later snapshots](#idea-3-hindsight-labels-from-later-snapshots) | unclicked candidates that the user opened later become soft positives in the training target; every input stays as it is | closed on 13 November by an argument and under the team's rule against later snapshots, with no run | validation and test labels count these candidates as negatives; there are about 0.05 of them per train feed; an oracle on later activity adds +0.00275 ± 0.00068 on 15% of holdout feeds | 0.03 |
| [4. A learned model of each item's audience](#idea-4-a-learned-model-of-each-items-audience) | the ranker would also look from a candidate to the users who viewed it and to what else they viewed | no graph model was trained; one was sized on paper at +0.00004 to +0.00035 | 42.8% of public and 45.6% of private candidate rows hold an item that occurs in no history; the counts for the rest are 9.34 and 11.23 days old | 0.05 |
| [5. Click-label closures never re-tested on students](#idea-5-click-label-closures-never-re-tested-on-students) | the position tower, 16 ranking losses, the single-stream transformer and 13 block designs, tried again under the distillation target | each was closed before distillation: on the model of the first day, on a proxy model or at 2-epoch screens | the closures scan puts the hidden gains of +0.0003 at 0, at most 13 of 93 rows (95%); none of 26 recipe arms run on students reached its bar | 0.05 for the group |

### Idea 1. Next-event training of the history encoder

The ranker learns from clicks alone: 2.21% of the 246.7 million train candidates, about 5.5 million positive labels. The histories, the past events of each user that come with the feeds, hold 463.6 million events in 2.59 million sequences, and each event says what a user did next. That is about 85 times as many targets.

That signal reached the model only indirectly, in two forms:

- a separate encoder trained on the histories to predict the next item, whose 6 score columns are inputs of the ranker, worth +0.00077 ± 0.00073 on validation in one 10-epoch pair;
- [pseudo-feeds](glossary.md), extra training lists built from a history, which enter the loss at weight 0.3: the 8-seed pool read +0.00054 ± 0.00047 / +0.00010 ± 0.00027 against the reference pool.

The idea is a deeper history encoder inside the ranker, trained jointly with it, with a next-event loss at every history position. It never ran. Three rows of the catalogue closed it without training:

- "Pre-training programme" ([pre-training theme](07-what-failed.md#pre-training-and-embeddings-12)): the unlabelled corpus of 463.6 million events is about the size of the histories inside the labelled feeds, 448.7 million. That is true of the inputs and not of the supervision.
- "Encoder as the initialisation of the ranker's history encoder" (same theme): a gate on stored predictions failed two of three legs.
- "Self-supervised next-item task on the history encoder inside the ranker" ([losses theme](07-what-failed.md#losses-and-auxiliary-tasks-20)): a trigger test on the pool's residual read a partial correlation of -0.0012 ± 0.0139 against a bar of +0.05.

Tests of this kind on stored predictions were later found to see 10 to 26% of a true effect ([failure 14](07-what-failed.md#14-the-residual-gate-and-the-trained-input-that-passed-every-gate)). The capacity arms beyond width 192 (more width, more depth, routed experts) were all null, and all ran under click or teacher targets on the feed, never with dense targets on the history.

Against it stands how little the history has given beyond the first encoder:

- the separate encoder alone ranks validation at 0.3590 (0.3625 in its gate), close to a similarity scorer of about 330 parameters (0.3584);
- its frozen states as an extra branch of the ranker read -0.00007 ± 0.00053 on validation ([failure 11](07-what-failed.md#11-pre-training-on-the-history-corpus));
- a second history layer read +0.0002 / -0.0010 per student over 2 seeds;
- a history cut to feed events or to target actions hurt: -0.00311 / -0.00266 and -0.00473 / -0.00493.

### Idea 2. Function matching on random variants of each feed

Function matching is distillation in which student and teacher are given the same randomly changed input. A student is a ranker trained toward half clicks and half the scores of the teacher, an average of click-trained rankers. Here each student would see random subsets of a feed's candidates and cropped histories, each with the teacher's target for that same variant. The variants act as extra data, which is what the data-volume slope of [section 2](#2-the-data-runs-out) asks for.

Only a flawed form ran, the row "Students trained on random sub-lists of each feed" of the [distillation theme](07-what-failed.md#distillation-18): pooled z -0.96 and -0.99 over 2 seeds, at sampling rates 0.5 and 0.2. Its sub-lists kept every positive, 1.88 per list against 1.22 on real short feeds, and reused the in-feed context inputs computed on the full feed. The clean form never ran.

History crops ran in the same form. Students trained with 30% of their history events dropped read -0.00098 ± 0.00058 / -0.00066 ± 0.00033 as a 4-seed pool against the reference pool (pooled z -2.62). In both arms the target was the teacher's stored score of the full feed and the full history (`--kd_file` in `lab/nn_x/w5/train.py`), so no teacher re-labelled the changed input.

Against it: teacher-labelled data has been worth much less than click-labelled data. Teacher-only targets on 624,752 extra real feeds read +0.00069 ± 0.00057 / -0.00017 ± 0.00032 for the pool, the precedent of [section 5](#5-a-better-model-the-search-missed).

### Idea 3. Hindsight labels from later snapshots

For a train user with a later snapshot, the later history shows which unclicked candidates the user opened afterwards by another route. They are 0.402% of the negatives in the 18% of train feeds that have a later snapshot: about 0.3 per such feed and 0.05 per average train feed ([TRICKS_SWEEP_1113.md](../lab/reports/TRICKS_SWEEP_1113.md)). Positives are 2.21% of train candidates.

The idea uses them as soft positives in the training target, with every input built as before. The team's rule forbids it as written, and it stands until the person on the team changes it. In the judgement of 19 November the idea is still not [the leak](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak) found on the first day: nothing from a later snapshot enters an input, and no test row needs one.

It never ran. The catalogue row "Loss ideas never trained" ([losses theme](07-what-failed.md#losses-and-auxiliary-tasks-20)) covers masking these negatives. The record of 13 November closed masking and softening by an argument, and a note of the same day put label repair from later snapshots under the team's rule against any use of a later snapshot (`lab/reports/CODES_DECODING_1113.md`).

Against it: the argument of that closure still applies. Validation and test labels count these candidates as negatives too, so pushing them down in training agrees with the metric. The ceiling also looks low. An oracle on the categories of the user's later activity added +0.00275 ± 0.00068 on the 15% of holdout feeds it covers (row "Auxiliary task on the user's future activity window").

### Idea 4. A learned model of each item's audience

The ranker looks from a candidate to the user's history. It never looks from the candidate to the users who viewed it and to what else they viewed. That side enters only as pre-computed co-visitation counts: how often two items occur in the same history. With the popularity counts they were among the largest gains, +0.0032 / +0.0092 as inputs of the train stage in one paired 2-epoch run.

A graph model over users and items was never trained. The row "Graph neural networks over the user-item graph" of the [architecture theme](07-what-failed.md#neural-architecture-and-capacity-38) sized it on paper at +0.00004 to +0.00035.

Against it stand coverage, age and one related run:

- 42.8% of public and 45.6% of private candidate rows hold an item that occurs in no history, so there is no audience to read for them;
- the counts for the rest are on average 9.34 days old on public rows and 11.23 on private rows;
- a two-step co-visitation walk over the whole history, trained as an input, read pooled z +0.62 for students, and its file scored 0.00054 ± 0.00050 below FD16P20 on public ([failure 14](07-what-failed.md#14-the-residual-gate-and-the-trained-input-that-passed-every-gate)).

### Idea 5. Click-label closures never re-tested on students

Width 192 was closed under click labels and adopted under distillation ([03-solution.md](03-solution.md#width-192)), so a change of training target can reverse a closure. Four closures of the click-label period, the days before 14 November when every ranker trained on clicks alone, were never run again with students:

- the position tower, a learned bias for the display position in production feeds: 0.37361 with it against 0.37385 without it on validation, one run on the model of 11 November;
- the comparison of 16 ranking losses, made on a small proxy model;
- the single-stream transformer over user, history and candidates: -0.00073 ± 0.00051 / -0.00027 at a 2-epoch screen;
- the 13 block designs, at 2 epochs and 2 seeds each, none of them promoted.

Against it: the closures scan of [section 4](#4-ideas-closed-too-strictly) puts the number of hidden gains of +0.0003 at 0, with a 95% upper bound of 13 of 93 rows. The reversal of width 192 did not repeat. Of 26 recipe arms run on width-192 students from 14 to 16 November, none reached its bar ([distillation](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)), and routed experts and depth 7 read +0.00006 ± 0.00047 and -0.00001 ± 0.00048 on validation at 8 seeds.

### Ideas that sound missed but ran

These ideas are often proposed for data of this kind. Each of them ran, and none was worth keeping in the form that ran. They are listed to save a repeat of that form. The position tower is the exception: it ran once, on the model of 11 November, and idea 5 asks for it again on students.

The outcome is the catalogue's own, and each link leads to the theme table of [07-what-failed.md](07-what-failed.md#the-catalogue-by-theme) that holds the row. The first five rows are the five forms of position debiasing.

| Idea | Outcome in the catalogue |
|---|---|
| Position tower, a learned bias for the display position | no effect: 0.37361 with it against 0.37385 without it on validation, on the model of 11 November only, see idea 5 ([architecture](07-what-failed.md#neural-architecture-and-capacity-38)) |
| Propensity head | no effect: +0.00007 on validation ([architecture](07-what-failed.md#neural-architecture-and-capacity-38)) |
| Examination offset on random-order train rows | no effect: +0.00002 ± 0.00028 / -0.00008 ± 0.00015 ([losses](07-what-failed.md#losses-and-auxiliary-tasks-20)) |
| Cascade correction: on production feeds the loss ignores negatives more than 5 positions below the last click | no effect: pool of 16, +0.00004 ± 0.00027 / -0.00000 ± 0.00015 ([losses](07-what-failed.md#losses-and-auxiliary-tasks-20)) |
| Training-only offset for the truncation of production feeds after the last interaction | no effect: member -0.0005 ± 0.0005 / -0.0001, pool -0.0003 / -0.0009 ([popularity and display order](07-what-failed.md#popularity-staleness-and-display-order-bias-30)) |
| Items the user was shown earlier and skipped, as negative feedback | no effect: nested on validation -0.0006 to +0.0001; skipped items mark interest, with a within-feed AUC of 0.58 ([inputs](07-what-failed.md#inputs-and-features-27)) |
| Fresh exposure counts from other users' validation and test feeds | no effect: nested on validation -0.00004 ± 0.00020 ([inputs](07-what-failed.md#inputs-and-features-27)) |
| A random-order detector that also covers short feeds, for a larger random-order pool | no effect: member +0.00025 ± 0.00035 on validation at a 2% share of cooldown batches, -0.00161 ± 0.00043 at 10% ([optimiser and schedule](07-what-failed.md#optimiser-schedule-and-regularisation-33)) |
| Signals from the user's earlier labelled train feeds | no effect: 5 signals, -0.00036 to +0.00010 ([forensics and error analysis](07-what-failed.md#leaderboard-forensics-error-analysis-and-user-segments-11)) |
| A cross-fitted teacher | hurt: student pool 0.40079 / 0.32347 against 0.40134 / 0.32408; [section 5](#5-a-better-model-the-search-missed) has the re-read ([distillation](07-what-failed.md#distillation-18)) |
| Pseudo-labels on test feeds inside the fine-tune | no effect: out-of-fold 0.38535 against 0.38580 on validation ([fine-tune](07-what-failed.md#fine-tune-on-validation-18)) |
| A hashed item-ID table | hurt: -0.0022 ± 0.0008 on validation; for students -0.0057 / -0.0049 ([architecture](07-what-failed.md#neural-architecture-and-capacity-38)) |
| Trees on top of the network: a residual stacker on the neural score, a tree residual on the 20-member pool | helped then superseded: the stacker read -0.0001 ± 0.0006 on the next neural generation; hurt: the residual read -0.00031 on validation and -0.00042 (z -2.3) on the holdout ([trees and stackers](07-what-failed.md#baselines-trees-and-stackers-15)) |
| Labels of similar users | no effect: -0.0001 to -0.0003 on validation ([pre-training and embeddings](07-what-failed.md#pre-training-and-embeddings-12)) |

### What a fair test costs

A fair test is one training run of the idea's clean form: 8 seeds, read at the last checkpoint against the reference on the same seeds. Where the idea changes the training target (ideas 2 and 3), arm and reference both use fresh seeds, the rule of [section 3](#3-luck-in-measurement).

One 10-epoch student run took about 1.2 hours alone on a GPU on 19 November, and 4.1 GPU-hours as recorded on shared GPUs for the final files ([03-solution.md](03-solution.md#costs)). Eight seeds are then about 10 to 33 GPU-hours, before any extra cost of the idea itself.

The list runs from the cheapest to the dearest. The records place only its two ends, and the order of the three in the middle is a judgement.

1. Idea 2 is the cheapest. The sub-list code is in the student trainer (`lab/nn_x/w5/train.py`, flags `--aug_all --aug_cand`). It needs uniform sampling, context inputs recomputed per sub-list, and the teacher's score of each variant in place of the stored one.
2. Idea 5: the position tower is a switch of that trainer, which the student recipe turns off with `--no_pos`. The losses, the single-stream model and the block designs exist only in older lab code that has no teacher target.
3. Idea 3 needs a target built from the later snapshots of the 18% of train feeds that have one, and a decision on the team's rule against later snapshots before any run.
4. Idea 4 needs a new model. Its catalogue row cites an estimate on paper and no code.
5. Idea 1 has the highest ceiling and is a build of several days.

These runs were not made: each of the five ideas was closed on paper before a run was planned ([13-how-the-work-was-run.md](13-how-the-work-was-run.md#what-a-stronger-setup-would-do-here)).

## How the explanations compare

The last column gives the records' own probabilities and bounds.

| Explanation | Strength of evidence | Probability or bound on record |
|---|---|---|
| Noise in the labels | strong as a limit on measurement (27.6% against 27.5% of the loss); weak as a difference between teams | none directly; perfect soft labels +0.0001 to +0.0008 per file |
| The data runs out | strong for the inputs as given (channels at most +0.00003); weak for the gap, since the files are shared | channels at most +0.00003, blind below about +0.0002; H-C P 0.08, H-E P 0.04, H-F P 0.03 |
| Luck on validation | measured: FD16P20 0.40271 against 0.40237 and 0.40222 for larger pools; seed luck shared within a teacher; re-reads against the mean counted shared seed luck as a gain | not measured; acts through decisions |
| Luck on the public board | measured: 17 near copies, sd 0.00025; no selection gain in the leader's score; place 2 within 0.0002 of the leader on 19 November | H-B P 0.12, places 4 to 7; the team's excess (0.00041) widens the gap to about 0.0035 |
| Ideas closed too strictly | rule power 56% at +0.0005; 31 holdout-led closures; positive reads among closures at the rate of noise (46%); re-tested positive first reads -0.00001 ± 0.00013; closed ideas mostly level on the board | best estimate 0 hidden gains of +0.0003, at most 13 of 93 rows (95%); re-open queue +0.0003 (13 November) |
| A better model missed | board continuum; distillation's one-day +0.0010 on public; no recipe on record; five ideas closed without a fair run ([section 6](#6-ideas-the-search-may-have-missed)), a judgement with no experiment behind it | H-A P 0.64; H-D P 0.06, at most +0.0003; open options +0.0001 (unlabelled feeds) and +0.0002 to +0.0004, up to +0.0008 (merge); a judgement of about 0.25 that at least one of the five ideas of section 6 is worth +0.001 or more |

## What would settle it

- The private board of 2026-12-09, on 20,033 feeds. If the leader and place 2 keep margins like their public 0.0030 and 0.0028, public luck (H-B) cannot explain them; a reshuffle of places 4 to 7 is what H-B predicts.
- If private scores of earlier files become visible, W8P20 against W8 re-reads the 0.2 popularity-light part: +0.00052 on public, -0.00063 ± 0.00022 on the holdout.
- An account of the top entries: none was found on 2026-11-18, and under rule 8.6.3 the organiser reviews the top 10 solutions.
- Fresh seeds, paired by seed, for the closures with the best member-level evidence: R-Drop and an auxiliary-loss weight of 1.0 in the cooldown, tried on students, and the two retrieval variants whose final reads are missing.
- One fair training run for each of the five ideas of section 6, with 8 seeds read at the last checkpoint, starting with the clean form of function matching ([what a fair test costs](#what-a-fair-test-costs)).
