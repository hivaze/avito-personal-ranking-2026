# What worked

In brief:

- Eighteen changes account for the part of the score that can be attributed. The public score rose from 0.3739 (first file, 11 November) to 0.39526061 (pick 2, FDX16P20:ps), and 0.0182 of that 0.0214 came on the first day.
- [evidence.md](evidence.md) names, for each of the 18 changes, the published file that holds its number.
- The board measured six changes by pairs of files; the largest step was distillation with width 192, +0.001032. The other twelve rest on validation and the holdout, six of them on a single run.

## Reading the numbers

- All scores are NDCG, the competition metric, averaged over feeds. A pair "a / b" is the paired difference on validation / on the holdout.
- Validation is the 14,423 labelled random-order feeds of 22 and 24 September. The holdout is 30,000 random-order train feeds of 50 or more candidates that held-out members never train on. Notation follows [04-validation.md](04-validation.md); terms and file names (W8, C3, FD16P20) are in the [glossary](glossary.md).
- A click-label member is a ranker trained on clicks alone; a student is trained toward clicks and a teacher's scores (change 7). A full-view member reads all 34 pre-computed inputs; a popularity-light member reads 27, with the 8 popularity inputs replaced by one exposure count (change 16).
- A prediction out of fold comes from fine-tune models that did not train on the row's fold of users (change 3).
- ± is one standard error (SE), and z is the difference divided by it. A pooled z is (z on validation + z on the holdout) / √2.
- Public is the score on the 8,562 test feeds of the public board. A paired public SE is the per-feed SE of a file difference on validation, scaled to those 8,562 feeds.
- A z of 1 is ordinary noise. Pick 1 leads FD16P20 by +0.00032 on public at a paired SE of 0.00031, z 1.0, and noise alone gives |z| of 1 or more about one time in three.
- Six changes (1, 2, 4, 8, 9 and 10) were adopted on one run each. One run against another cannot show less than about 0.0029 on validation, 2.8 times its SE, the effect detectable at 80% power ([noise](04-validation.md#noise)). Changes 8, 9 and 10 did not reach that.
- "Removal" is the removal screen of 13 November: each component taken out of the final click-label recipe, 2 epochs, 2 seeds, with SEs of about 0.0006 on validation and 0.0003 on the holdout. It measured changes 4, 8, 9 and 10 again, and change 9 came out inconclusive.
- Holdout numbers overstate changes that feed on popularity, because the holdout's popularity inputs are 1.75 days stale against 9.34 on public ([the witnesses](04-validation.md#the-witnesses)).
- A reference pool is one draw of seeds. Student screens from 14 to 19 November paired an arm's seeds 951 to 954 with the same 4 seeds of the reference recipe, a pool 3rd of the 70 possible 4-seed pools on validation (+0.00057 over their mean) and 37th on the holdout. An arm with the same teacher shares most of that luck, so its seed-paired read is fair ([04](04-validation.md#seeds-shared-by-arm-and-reference)).
- An arm is read at the checkpoint its shipped member keeps, and a seed-paired read is not corrected afterwards. Routed experts (19 November) read -0.00086 at the checkpoints the holdout selected and +0.00009 paired by seed at the last checkpoint. Re-read against the mean of the 70 pools, which counts the luck twice, they rose to +0.00070. Eight fresh seeds then read +0.00006 ± 0.00047.
- Costs are wall-clock minutes of one process on a GPU shared with other processes, so they are upper bounds.

## The changes, ranked

The table ranks the changes by the gain on validation measured when each was adopted (for a range, by its lower end). Each number links to the section on that change, which says what it is, why it was tried and how later checks read it.

| # | Change | Gain when adopted, validation / holdout | How it was measured | Public |
|---|---|---|---|---|
| [1](#1-attention-from-each-candidate-to-the-users-history) | Attention from each candidate to the user's history | +0.0055 / no holdout yet | 1 seed, 2 epochs: 0.37361 with history tokens, 0.36808 without | not isolated |
| [2](#2-ten-epochs-instead-of-two) | Ten epochs of training instead of two | +0.0052 / +0.0050 | 1 seed, first optimiser: 0.38625 to 0.39141 and 0.31168 to 0.31666 | not isolated |
| [3](#3-fine-tune-on-the-validation-days) | Fine-tune on validation with inputs that exist only for validation and test | +0.0040 out of fold / not measurable | 1 early model: 0.38068 train-only, 0.38471 fine-tuned | +0.00047 (best click-label pool, paired SE about 0.0005); +0.00048 ± 0.00055 (8 students) |
| [4](#4-co-visitation-and-popularity-counts-as-inputs-of-the-train-stage) | Co-visitation and popularity counts as inputs of the train stage | +0.0032 / +0.0092 | 1 paired run, same seed, 2 epochs | not isolated |
| [5](#5-muon-with-a-warmup-stable-decay-schedule) | Muon optimiser with a warmup-stable-decay schedule | +0.0031 / +0.0033 | 3 seeds, 2 epochs; validation z 4.6 | not isolated |
| [6](#6-retrieval-of-the-16-most-similar-history-items) | Retrieval of the 16 most similar history items, with gated attention | +0.0028 / +0.0038 | 3 seeds, 2 epochs; validation z 3.8 | not isolated |
| [7](#7-distillation-from-an-ensemble-of-earlier-members) | Distillation from an ensemble of 11 click-label members | +0.00257 ± 0.00047 / +0.00200 ± 0.00028 | 4 seeds, 10 epochs, per student | +0.001032 together with change 15 (paired SE about 0.00043) |
| [8](#8-an-auxiliary-loss-toward-the-production-display-order) | Auxiliary loss toward the production display order | +0.0024 / +0.0027 | 1 paired run, same seed, 2 epochs; listwise form against regression form | not isolated |
| [9](#9-a-cosine-similarity-bias-in-the-history-attention) | Cosine-similarity bias in the history attention | +0.0024 / no holdout yet | 1 seed, 2 epochs: 0.37385 to 0.37624 | not isolated |
| [10](#10-equal-loss-weight-per-feed) | Equal loss weight per feed | +0.0020 / +0.0006 | 1 paired run, same seed, 2 epochs | not isolated |
| [11](#11-scores-of-the-trajectory-encoder-as-inputs) | Scores of the trajectory encoder as inputs | +0.00191 ± 0.00058 / +0.0002 | 2 seeds, 3 epochs | not isolated |
| [12](#12-two-per-cent-random-order-feeds-in-the-cooldown) | 2% random-order feeds in the cooldown | +0.00117 ± 0.00039 / +0.00134 ± 0.00022 | 7 members, cooldown re-run, against 10% | not isolated |
| [13](#13-score-head-in-32-bit-floats) | Score head in 32-bit floats | +0.001 to +0.003 / no holdout yet | same weights predicted again; the first model went from 0.37047 to 0.37361 | not isolated |
| [14](#14-pools-of-members-from-different-recipes) | Pools of members from different recipes | +0.00083 ± 0.00054 / +0.00157 ± 0.00034 | 19 members of three recipes against 6 members of one | not isolated |
| [15](#15-width-192-for-the-students) | Width 192 for the students | +0.00049 ± 0.00056 / +0.00059 ± 0.00032 | pools of 4 students at widths 192 and 128 | inside the +0.001032 of change 7 |
| [16](#16-popularity-light-students-at-weight-02) | Popularity-light students at weight 0.2 | +0.00049 ± 0.00031 / -0.00063 ± 0.00022 | W8 with and without 4 narrow popularity-light students | +0.00052 (paired SE 0.00041) |
| [17](#17-popularity-light-weight-04-on-short-feeds) | Popularity-light weight 0.4 on feeds of fewer than 50 candidates | +0.00022 ± 0.00024 / not measurable | FD16P20 with and without the rule, same 20 members | +0.00032 (paired SE about 0.0003) |
| [18](#18-pseudo-feeds-for-the-full-view-students) | Pseudo-feeds built from user histories as extra training lists for the full-view students | +0.00015 ± 0.00036 / not measurable | FDX16P20:ps against FD16P20, all 16 full-view students replaced; the 8-seed pool of held-out students read +0.00054 ± 0.00047 / +0.00010 ± 0.00027 | +0.00037 (paired SE 0.00047); -0.00010 with 8 of the 16 replaced |

## Where the public score came from

The board saw most of the changes only in bundles. The 13 new team bests group into nine steps, and most steps carried several changes at once. File numbers are the row numbers of the [table of submissions](05-timeline.md#all-40-scored-submissions); the steps are drawn in the [progress chart](figures/public-progress.svg).

| Files | Public before | Public after | Step | Ranked changes that entered |
|---|---|---|---|---|
| 1 to 3 | 0.3739 | 0.3802 | +0.0063 | changes 8 and 10 and the extra inputs of change 3, under a residual tree stacker (a tree model that corrects the rankers' scores) |
| 3 to 5 | 0.3802 | 0.38786 | +0.0077 | changes 2 and 4, with in-feed context inputs; the tree stacker dropped |
| 5 to 6 | 0.38786 | 0.39208 | +0.0042 | changes 5, 6 and 14; change 11 in 7 of the 19 members |
| 6 to 7 | 0.39208 | 0.392820 | +0.0007 | change 12 in 14 of the 20 members; 6 members with a GRU (gated recurrent unit) history encoder; change 11 in every member |
| 7 to 16 | 0.392820 | 0.393454 | +0.0006 | 8 popularity-light click-label members; a fine-tune averaged over 3 seeds |
| 16 to 19 | 0.393454 | 0.394486 | +0.0010 | changes 7 and 15 |
| 19 to 22 | 0.394486 | 0.3950094 | +0.0005 | change 16 |
| 22 to 33 | 0.3950094 | 0.3952092 | +0.0002 | change 17, on the members of file 24, trained on all train feeds |
| 33 to 38 | 0.3952092 | 0.39526061 | +0.00005 | change 18 in all 16 full-view students; change 17 dropped, weight 0.2 on every feed |

The first day, 11 November, took the public score from 0.3739 to 0.39208: 0.0182 of the total 0.0214. The 0.0032 added in the eight days after it came from:

- 0.0010 with distillation (changes 7 and 15);
- 0.0006 plus 0.0005 with popularity-light members (file 16, then change 16);
- 0.0007 with file 7: the 2% cooldown re-runs, the GRU members and encoder scores in every member;
- 0.0002, about one paired SE, with the short-feed weight of change 17;
- 0.00005 with the pseudo-feed students of change 18, which replaced change 17 in the best file; against FD16P20 that file gained 0.00037.

Capacity and tuning under click labels added nothing on validation. Width did not help in three tests (change 15), none of 13 block designs and hyper-parameter settings was promoted, and language-model backbones lost 0.0038 to 0.0053. Less data did hurt: removing 20% of the users from training cost a member 0.0028 / 0.0022.

Under distillation, routed experts and depth 7 for the students (19 November) read +0.00006 ± 0.00047 and -0.00001 ± 0.00048 on validation at 8 seeds against 8 reference students, and both were closed ([experiments/optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/)).

The two largest effects measured late in the work were the amount of labelled data and cleaner labels. Five ideas that touch them never had a fair training run. A judgement of 19 November on each is in [section 6 of 12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed).

## 1. Attention from each candidate to the user's history

Each candidate token attends over a user token and the user's 256 newest history events. An event is its item embedding plus embeddings of the event type, the event source, the category and the recency index.

The data analysis had shown history similarity to be the strongest raw signal, so the first ranker was built around it. The best single similarity feature scored 0.349 on validation, and a 330-parameter attention scorer over the history 0.3584.

With the history tokens the model read 0.37361 on validation, without them 0.36808 (one seed, 2 epochs). Without them it peaked after 2,000 steps and then degraded, although it kept its hand-made history-similarity features. The report of that round gives the component as +0.006 to +0.010; the table holds the one pair with a matching base. The result rests on one seed, and the removal screen of 13 November did not include it.

The model with history tokens has 0.84 million parameters against 0.65 million. The run times of the two runs are not comparable, because the batching changed between them. Code: `avito_ranker/model/`; the round-1 runs are in [neural-ranker-v1](../experiments/neural-ranker-v1/).

## 2. Ten epochs instead of two

Every member trains for 10 epochs over the train feeds; the first rankers trained for 2. On one seed the holdout was still rising at 5 epochs (0.3117 at 2 epochs, 0.3128 at 3, 0.3148 at 5), so the next run went to 10.

With the first optimiser, AdamW with cosine decay, the gain showed in a single model: 0.38625 to 0.39141 on validation and 0.31168 to 0.31666 on the holdout. With the final optimiser (change 5) one member gains only +0.0006 / +0.0007, and for 90 minutes on 11 November two epochs were taken to be enough. A six-seed ensemble showed otherwise: +0.0040 / +0.0031.

Ten-epoch seeds differ more from each other: two-epoch seeds correlate 0.978 with each other and ten-epoch seeds 0.93 to 0.94. The long runs earn their time only as members of a pool. Going further was worse: 15, 20 and 30 epochs cost a member 0.0006, 0.0008 and 0.0016 on validation.

A 10-epoch member at width 128 took 56.8 to 182.1 minutes (117.6 to 182.1 for the click-label members of the teacher). The first two-epoch model, without retrieval and with the first optimiser, had taken 12 to 20. The number of epochs is a field of the recipes in `configs/recipes/`, read by `avito_ranker/training/train.py`. The epoch probes are in [neural-ranker-v1](../experiments/neural-ranker-v1/) and the six-seed comparison in [rounds-5-7](../experiments/rounds-5-7/).

## 3. Fine-tune on the validation days

Each member trained on the train split is then fine-tuned on the validation feeds, in 5 folds grouped by user. The fine-tune adds a branch of 42 inputs that exist only for validation and test rows. The final score is half the fine-tuned and half the train-only prediction.

Validation is the only labelled random-order data from near the test days: train ends on 19 September, validation is 22 and 24 September, and test is 26 to 29 September.

On an early model the fine-tune added +0.0040 out of fold, that is, on predictions by models that did not train on the row's fold (0.38068 train-only, 0.38471 fine-tuned). Most of that came from the extra inputs: the same fine-tune with a smaller set of 27 item-level inputs reached 0.38123.

As the train stage took over those inputs (change 4) the validation gain shrank. In the final recipe the fine-tuned half reads +0.00009 ± 0.00043 on validation for 8 students.

The public board reads it higher. The half-and-half mix scores about 0.0005 above either branch alone: +0.00047 for the best click-label pool (paired SE about 0.0005) and +0.00048 ± 0.00055 for 8 students. The 8-student file W8 scored 0.394009, 0.394486 and 0.393960 at fine-tune weights 0, 0.5 and 1.0.

[03-solution.md](03-solution.md#fine-tune-on-the-validation-days) has the rest of the evidence, the variants that did not help and why validation reads the branch low. A fine-tune costs 15.1 to 22.2 minutes per member for 5 folds and 3 seeds. Code: `avito_ranker/training/finetune.py`, with the 42 inputs in `avito_ranker/features/`; [fine-tune](../experiments/fine-tune/).

## 4. Co-visitation and popularity counts as inputs of the train stage

The final recipe has 8 such inputs, the popularity inputs:

- 3 item-to-item co-visitation scores, which count how often the candidate occurs near the feed's history items in other users' histories. A train row of day d uses only feeds of earlier days.
- 5 leave-user-out counts, from which the same user's later snapshots are removed.

The run that measured the gain added 16 columns of these two kinds.

In the fine-tune branch such signals had carried most of the gain of change 3, and 3.15 million train feeds should learn them better than 14,423 validation feeds. The gain showed from the first 1,000 steps, and one paired run (same seed, 2 epochs) ended at +0.0032 / +0.0092. The removal screen read -0.0023 / -0.0074.

The holdout reads three times the validation gain. The causes were found only later. Co-visitation is non-zero for 21% of train candidates and 11% of validation candidates, and the holdout's counts are 1.75 days old. On stale validation the 8 inputs are worth 0.0009.

They also cost something on cold items, which were never a train candidate before the feed's day. The first ensemble with them gained 0.0046 ± 0.0010 on validation over the last one without. That is +0.0121 ± 0.0013 on feeds whose positives are all warm items (70.9% of validation) and -0.0214 ± 0.0023 on feeds whose positives are all cold (17.4%). Change 16 exists because of that trade.

The 20 in-feed context inputs that entered the model in the same round were measured in single paired runs when added. The 10 near-duplicate inputs read +0.0005 / +0.0017 and the 10 candidate-competition inputs +0.0004 / +0.0009. In the removal screen the near-duplicate inputs were worth 0.0014 / 0.0003, and the competition inputs were inconclusive (0.0005 / 0.0000).

Building the counts takes about 31 minutes of CPU for the co-visitation of train rows and 44 seconds for the leave-user-out counts. Code: `avito_ranker/features/`; the feature screens are in [feature-families](../experiments/feature-families/).

## 5. Muon with a warmup-stable-decay schedule

Muon is an optimiser that orthogonalises the momentum update of each weight matrix. It updates the 46 hidden weight matrices, which hold 2.02 million of the 2.12 million parameters, and AdamW updates the rest. The learning rate is held at 2e-3 after 500 warm-up steps and decays to 3% over the last 20% of training, the cooldown. The pair replaced AdamW with cosine decay followed by a second training stage on random-order feeds.

The idea comes from recent language-model training practice. It was tested on 11 November in a batch of optimiser, attention and feature tricks. A re-check with a fresh seed confirmed this one, retrieval (change 6) and a fine-tune schedule, and rejected 14 others.

The pair gave +0.0031 / +0.0033 (3 seeds, 2 epochs, validation z 4.6). Muon alone gave +0.0020 / +0.0030 and the schedule the rest; a second stage after it lowered the holdout by 0.0015. The gain arrives in the cooldown: a 7-seed ensemble gained 0.0031 ± 0.0006 on validation between the end of the constant phase and the end of the cooldown.

No later check is on record. The first use at 10 epochs collapsed six members ([failure 3](07-what-failed.md#3-a-cooldown-dose-tuned-at-2-epochs-and-used-at-10)). Together with retrieval, the change made a training step about twice as slow. Code: the trainer in `avito_ranker/training/`; [optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/).

## 6. Retrieval of the 16 most similar history items

For each candidate, the 16 history items with the highest cosine similarity are retrieved. Each contributes a value built from the difference of learned keys and from its event type, source and recency, and the weighted sum is added to the candidate token. Attention outputs pass through a learned gate.

The aim was to give each candidate direct access to its nearest history events and to what the user did with them. The pair read +0.0028 / +0.0038 (3 seeds, 2 epochs, validation z 3.8):

- Retrieval alone read +0.0019 / +0.0033 on 2 seeds.
- The gate alone read +0.00042 on validation (z 0.9) and +0.00065 on average over four run pairs. It was kept because it was cheap and never negative.
- After the fine-tune the pair was still worth +0.00195 out of fold (z 2.6).
- Removal read -0.0034 / -0.0029, the largest entry of the removal screen.

A depth of 32 failed: -0.00050 / -0.00048 at 2 epochs, -0.0002 / -0.0003 in 10-epoch pairs and -0.00098 / -0.00043 for distilled students. The cost is retrieval's share of the doubled step time of change 5. Code: `avito_ranker/model/`; [optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/).

## 7. Distillation from an ensemble of earlier members

The teacher is the mean score of 11 click-label members (rankers trained on clicks alone) on every one of the 246.7 million train candidates. A student is the same ranker trained toward a target that is half the click distribution of the feed and half the softmax of the teacher's scores over the feed. Half of the final students use a 25-member teacher, which made no measurable difference. [03-solution.md](03-solution.md#distillation) draws the target and lists the teacher's members.

The records give the motive for trying it on 14 November. A pool scored about 0.004 above any of its members, and a width test in the night before had shown the ranker memorising: with one click among about 100 candidates per feed the label is noisy, and the pool had averaged part of that noise out.

Distillation was on the closed list by then. On 11 November the form from the team's research note on training, one frozen two-epoch teacher of the student's own architecture, read +0.00064 on validation (z 1.6) and was filed as no effect. On 13 November the scan of outside techniques still gave distillation in the cooldown a probability of 0.02 of adding +0.0005 at pool level ([how good the priors were](08-outside-ideas.md#how-good-the-priors-were)).

The ensemble form passed a promotion rule written at 03:14 UTC on 14 November, before its first run, and read at 05:18 ([rules written before the read](04-validation.md#rules-written-before-the-read)):

- per student (4 seeds, 10 epochs): +0.00257 ± 0.00047 / +0.00200 ± 0.00028;
- pool of 4 students: +0.00198 ± 0.00061 / +0.00021;
- W8, the first file of distilled students (8 full-view students), against the best click-label file (C3): +0.00064 ± 0.00055 on validation and +0.001032 on public (paired SE about 0.00043), together with width 192 (change 15).

That public step is the largest after the first day.

Students are more alike than click-label members: their within-feed correlation is 0.978 against 0.949 at width 128, so a pool of students gains less than a single student does. At width 128 a pool of four students matched C3 (28 members) on validation without passing it (0.40085 against 0.40103, train-only), and adding the four to C3 changed it by -0.00004. The public step came with width 192.

The teacher scores are in-sample for the teacher on train feeds: its NDCG is 0.3465 there and 0.3237 on the holdout.

A cross-fitted teacher removed this. Its four students read 0.40079 against 0.40134 for the same seeds of the reference, a read that may be up to 0.0006 too low for a different teacher ([04](04-validation.md#seeds-shared-by-arm-and-reference)). Against the mean of the 70 possible 4-seed reference pools it is level on validation (-0.00015 ± 0.00047) and 0.00059 ± 0.00026 lower on the holdout.

It was closed: the teacher's softmax is nearly flat, and the in-sample copy of the label is only 0.2 to 0.7% of the target mass ([distillation](../experiments/distillation/)). Teacher weights of 0.3, 0.8 and 1.0 and a teacher temperature of 0.7 or 1.5 lost at pool level. Two arms of a second generation, with students as teacher, read -0.00013 and -0.00023 on validation against the mean reference pool, no gain ([what failed](07-what-failed.md)).

The teacher cost no new training: its 11 members existed, at 30.9 GPU-hours of logged time. Writing their scores for all train candidates took at most 4.6 minutes per group of checkpoints and produces a 987 MB file. The first four width-128 students took 105 to 120 minutes each. Code: `avito_ranker/training/teacher.py` for the teacher file and `avito_ranker/training/train.py` for the student; [distillation](../experiments/distillation/).

## 8. An auxiliary loss toward the production display order

Train feeds are logged in the order of Avito's production ranker. A second head is trained with a listwise cross-entropy toward softmax(-position / 10), each feed weighted by the probability that it was shown in production order. The loss weight is 0.5, annealed to 0.1 in the cooldown, and the head is not used at inference.

The production position is informative and absent at test, and the two obvious uses of it came first. An additive position term, dropped at inference, changed nothing (0.37361 with it, 0.37385 without). Regressing the position in a second head added 0.0020 (0.37385 to 0.37581, one seed). The listwise form added +0.0024 / +0.0027 on top of the regression form (one paired run, same seed, 2 epochs).

Removal read -0.0022 / -0.0023. Removal from students, 8 seeds, read -0.00135 ± 0.00058 / -0.00128 ± 0.00033 on the pool.

The head regularises the shared trunk and carries little by itself: its own output scores 0.304 on validation, where the main head of the same members scores 0.398. The training cost of the extra head is not on record. Code: `avito_ranker/model/` and the loss in `avito_ranker/training/`; [neural-ranker-v1](../experiments/neural-ranker-v1/).

## 9. A cosine-similarity bias in the history attention

The raw cosine between a candidate's embedding and each history item's embedding is added to the attention logits, with a learned scale per head. Some heads can then look up the nearest history item directly in the provided embedding space. Those embeddings are already a good similarity: the data analysis judged them to come from Avito's own two-tower recommender ([what the embeddings are](02-what-the-data-told-us.md#what-the-embeddings-are)).

It read +0.0024 on validation (0.37385 to 0.37624, one seed, 2 epochs). Together with the first auxiliary head it was worth 0.0046 in round 1 (0.37385 to 0.37848). The removal screen, run after retrieval (change 6) had been added, read -0.0004 / -0.0005 and could not confirm it. Whether retrieval had taken over its role was not tested.

Doubling the scale for students lost (pooled z -1.58). No cost is on record. Code: `avito_ranker/model/`; [neural-ranker-v1](../experiments/neural-ranker-v1/).

## 10. Equal loss weight per feed

The loss is summed over feeds with every feed at the same weight; the first version took the mean over a batch. A batch holds at most 32,768 candidates and 512 feeds. A batch of 300-candidate feeds therefore holds about 109 feeds, and under the batch mean each of them weighed about 4.7 times a feed of 64 candidates or fewer.

The metric is an unweighted mean over feeds, and the loss now matches it. One paired run (same seed, 2 epochs) read +0.0020 / +0.0006, and removal -0.0015 / -0.0010. In the fine-tune, where batches are small and feed sizes vary less, the same change was neutral (0.38088 against 0.38097).

It costs nothing. Code: the loss in `avito_ranker/training/`, with the batching in `avito_ranker/data/loader.py`; [neural-ranker-v1](../experiments/neural-ranker-v1/).

## 11. Scores of the trajectory encoder as inputs

The trajectory encoder is a separate self-supervised transformer, trained on all user histories without labels. Its scores for a candidate given the history enter the ranker as 5 of the 34 pre-computed inputs, plus a presence flag. Used alone, the encoder scored 0.359 on validation, the best single signal found.

It read +0.00191 ± 0.00058 / +0.0002 (2 seeds, 3 epochs). The gain sits on feeds with fewer than 50 candidates (+0.0040, against +0.0005 on longer feeds), which the holdout does not contain. Removal read -0.00128 ± 0.00061 / -0.00054. A 10-epoch pair read +0.00077 ± 0.00073 / +0.00067, smaller than at 3 epochs.

The adopted build is the fourth ([failure 2](07-what-failed.md#2-encoder-scores-for-train-rows-first-build)). The first gave train rows a missingness pattern that marked positives, a stop-gap was accepted on weak evidence, and the full rebuild left 76 to 92% of the values, depending on column and split, at a clipping bound. A stronger encoder dose and the encoder's hidden states as an extra input were both closed.

The encoder took 5.2 GPU-hours to train as recorded on 13 November: four runs of 21.5 to 43 minutes on the full corpus and six windowed runs of 13.4 to 58.5 minutes. Its corpora, ten checkpoints and window scores take 202 GiB in `artifacts/data/pretrain/`. Code: `avito_ranker/features/`; [pretraining](../experiments/pretraining/).

## 12. Two per cent random-order feeds in the cooldown

During the cooldown a training step is, with probability 0.02, a batch from the pool of train feeds that the random-order detector marks as shown in random order ([how they were found](02-what-the-data-told-us.md#random-order-feeds-inside-train)). The pool holds 164,118 feeds with 50 or more candidates, 134,118 without the holdout.

Validation and test were all shown in random order, train mostly in production order. An analysis of all 285 training runs to that date showed that the cooldown is where this pool gets memorised. From its best step to its last, a run lost 0.0007 on validation with no pool batches, 0.0033 at 10% and 0.027 at 30%, the share that had collapsed six members ([failure 3](07-what-failed.md#3-a-cooldown-dose-tuned-at-2-epochs-and-used-at-10)).

Against the 10% used before, 2% read +0.00117 ± 0.00039 / +0.00134 ± 0.00022 (7 members, cooldown re-run). The other doses:

- against no pool batches, 2% reads -0.00002 on validation and +0.0023 on feeds whose positive is a cold item;
- 1% and 4% read -0.0004 / +0.0005 and -0.0008 / -0.0005 against 2%;
- 5% on students read a pooled z of -0.08 on the last day.

On the board, pools heavy in members with a cooldown re-run transferred poorly, because a cooldown re-run by itself adds 0.0009 on validation and loses 0.0005 on the holdout (phase 3 of the [timeline](05-timeline.md)). The change costs nothing in a run from scratch; as a cooldown re-run it took 23 to 51 minutes per member. Code: the cooldown in `avito_ranker/training/`, with the detector's probability per train feed in `avito_ranker/features/`; [rounds-5-7](../experiments/rounds-5-7/) and [exploration-detection](../experiments/exploration-detection/).

## 13. Score head in 32-bit floats

The model runs in 16-bit floats and its output head in 32-bit. The metric orders tied scores pessimistically, negatives first. 16-bit scores are quantised (in steps of 1/64 between 2 and 4), which creates ties inside a feed: one early model read 0.37218 with pessimistic ties and 0.37408 with a random tie-break.

Predicting again with the same weights and a 32-bit head gained +0.001 to +0.003; the first model went from 0.37047 to 0.37361. No later check is on record. The fix needed no retraining and costs nothing. The submission writer breaks any remaining exact tie with a fixed random number per row. Code: `avito_ranker/model/`, `avito_ranker/metrics.py` and the writer in `avito_ranker/ensemble/`; [neural-ranker-v1](../experiments/neural-ranker-v1/).

## 14. Pools of members from different recipes

A pool is the equal-weight mean of its members' within-feed z-scores. From the evening of 11 November pools were filled with members whose recipes differ in learning rate, cooldown share, inputs or history encoder. A study of the stored predictions that day had shown that a member from another recipe adds two to three times what another seed of the same recipe adds.

Nineteen members of three recipes against 6 members of one read +0.00083 ± 0.00054 / +0.00157 ± 0.00034. That cleared 2 SE on the holdout only (z 4.6; validation z 1.55), and the pool went to the board as file 6.

The effect saturated within a day: by 13 November four more members trained from scratch changed the 20-member pool by +0.00003 / -0.00003. Weights were never fitted. The equal-weight mean was the holdout optimum among 78 second-level alternatives, and the one learned stacker sent to the board gained 0.00056 on validation and lost 0.00035 on public.

Under distillation the differences between recipes disappeared, and the only member type that stayed different was an input view (change 16). Pooling costs nothing beyond the members. Code: `avito_ranker/ensemble/`; [hard-label-pools](../experiments/hard-label-pools/).

## 15. Width 192 for the students

The token width goes from 128 to 192, which roughly doubles the parameters, to 2.12 million. Under click labels width had not helped in three tests:

- width 256 in round 1: 0.37271 against 0.37385 on validation, one seed;
- a 7-seed family at width 192, read on 12 November: its members averaged 0.39371 / 0.31843 against 0.39460 / 0.31911 for seven of width 128, and its 7-member pool tied with theirs (F12 against F7 in the lab records);
- cooldown re-runs at width 192 in the night before distillation, 5 runs against 5 at width 128: -0.00087 / -0.00106, with a lower in-sample loss and no gain held out.

The second distillation round asked whether capacity becomes usable once the targets are less noisy. By the rule written for that round it did not: the pool of four wide students missed its bar of +0.0005 on validation by 0.00001. Pools of 4 students at widths 192 and 128 differed by +0.00049 ± 0.00056 / +0.00059 ± 0.00032 (`lab/nn_x/w12/stale/veto/logs/read2.txt`).

All eight wide students were then pooled, with the step recorded as post hoc. Eight wide against four narrow read +0.00072 / +0.00107 ± 0.00031, a comparison that also doubles the pool and so overstates width.

Widths 224 and 256 added nothing. Width 224 read +0.00026 ± 0.00058 / +0.00016 ± 0.00032 against the same 4 seeds at width 192, and +0.00031 ± 0.00057 on validation at the last checkpoint. As all-data files on 18 November it read +0.00005 to +0.00017 on validation against FD16P20 and was not scored.

On the board the eight wide students scored 0.394486, and the same eight plus eight narrow ones 0.394078, a difference of 0.95 paired SE. The first four wide students took 194 to 195 minutes each, against 105 to 120 for the four width-128 students of the same round; the all-data wide students of the final file took 221 to 259. The width is a field of the recipes in `configs/recipes/`; [distillation](../experiments/distillation/).

## 16. Popularity-light students at weight 0.2

A popularity-light member is trained with the 8 popularity inputs of change 4 replaced by one exposure count clipped at 50; a full-view member reads all 34 pre-computed inputs. Popularity-light members are pooled apart, and each half of a file (train-only and fine-tuned) is 0.8 of the main pool's z-score plus 0.2 of theirs.

The removal screen had shown the popularity inputs to be the channel for warm items, and they are 9 to 11 days stale on test ([staleness](02-what-the-data-told-us.md#cold-items-and-stale-popularity)). After distillation had made all students alike, this view was also the only structural difference left to pool.

W8 with and without 4 narrow popularity-light students read +0.00049 ± 0.00031 on validation, -0.00063 ± 0.00022 on the holdout and +0.00052 on public (paired SE 0.00041). The holdout read the admixture negative in both generations (-0.00063 for the students, -0.00073 for the click-label pool of file 16) where public read +0.00052 and +0.00063. [04-validation.md](04-validation.md#rules-written-before-the-read) explains the bias and how the holdout lost its veto.

The holdout with lagged inputs (13,424 feeds) read +0.00024 ± 0.00031, the most favourable of nine member compositions, so it does not settle the question. The evidence is validation and public together (+0.00050 ± 0.00025), and the click-label generation, where the three scored pools with such members averaged 0.00053 above the three without.

The weight was not tuned on the board. On the 13,424 feeds where the lagged holdout is valid, three weights read as follows against the full-view pool alone ([witness_control_A.txt](../experiments/popularity-staleness/results/witness_control_A.txt)):

| Popularity-light weight | 0.1 | 0.2 | 0.3 |
|---|---|---|---|
| Lagged holdout | +0.00013 ± 0.00021 | +0.00024 ± 0.00031 | -0.00015 ± 0.00038 |

A rule that raised the weight to 0.4 on long feeds scored 0.00015 lower on public; the same weight on short feeds is change 17.

Eight or twelve students in place of four read -0.00021 ± 0.00017 and -0.00016 ± 0.00018 on validation, so the four of the final file are a favourable draw. They cost four more trainings of 248 to 257 minutes each. Code: the exposure input in `avito_ranker/features/`, the 0.8 / 0.2 mix in `avito_ranker/ensemble/` with its weights in `configs/ensembles/`; [popularity-views](../experiments/popularity-views/) and [popularity-staleness](../experiments/popularity-staleness/).

## 17. Popularity-light weight 0.4 on short feeds

On feeds of fewer than 50 candidates each half of the file mixes the popularity-light part at 0.4 instead of 0.2; longer feeds keep 0.2. The members are the 20 of FD16P20 (16 full-view and 4 popularity-light students, all trained on all train feeds), so the change is one number in the mixing step, and the two files correlate 1.0000 on test.

It was built on 17 November as a try file: an idea with a positive validation mean, sent to the board without a rule written for it in advance. The opposite rule, 0.4 on long feeds, had already scored 0.00015 below FD16P20 (change 16), and short feeds were the side the board had not read. They are 40.9% of validation, 38.7% of public and 39.2% of private; the holdout has none.

| Reading | FD16P20@m0.4_0.2 against FD16P20 |
|---|---|
| Validation | +0.00022 ± 0.00024 |
| Validation re-weighted to the public mix (public-mix view) | +0.00027 |
| Validation re-weighted to the private mix (private-mix view) | +0.00033 |
| Holdout | not measurable: no feed below 50 candidates |
| Public | +0.00032 (0.3952092 against 0.394892), paired SE about 0.0003 |

It also scored 0.00020 above W8P20, pick 2 at the time. It was the team's best public score until 19 November and is the first final file.

The weight is not an optimum. With 0.2 on long feeds, the short-feed weight reads as follows on validation:

| Weight on short feeds | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|
| Validation | 0.40271 | 0.40266 | 0.40293 | 0.40251 | 0.40232 | 0.40184 |

Against 0.4, the readings at 0.3 and 0.5 are -0.00027 ± 0.00017 and -0.00042 ± 0.00016. One high point with lower readings on both sides points to a favourable draw of the four popularity-light students. Other member sets agree: with 8 or 12 popularity-light students the rule read 0.00057 and 0.00039 below the file. On the file whose students were partly trained with pseudo-feeds (row 36 of the [table](05-timeline.md#all-40-scored-submissions)), it lowered validation from 0.40298 to 0.40262.

What supports the rule is one paired SE on public, in the direction that validation and both mix views gave. The team made it pick 1 because it changes no member and reads above FD16P20 on validation and on both mix views. Both its validation and its public lead over FD16P20 are the best of several tries, so the expected gain on private is only about +0.00005 ([11-final-picks.md](11-final-picks.md)).

The long-feed side stayed at 0.2. Weights of 0.1 and 0 there read 0.00025 ± 0.00013 and 0.00068 below the file on validation, and a different popularity-light part on long feeds (A1 in the [catalogue](07-what-failed.md)) scored 0.00017 below it on public.

The rule costs nothing, since the mix is computed from stored member predictions. Code: the rule in `avito_ranker/ensemble/build.py`, its weights in `configs/ensembles/fd16p20_m04_02.toml`; [popularity-views](../experiments/popularity-views/).

## 18. Pseudo-feeds for the full-view students

A pseudo-feed is an extra training list built from a user's history. On every training step before the cooldown, the trainer builds up to one per logged feed in the batch, from that feed's own history, and adds its listwise loss at weight 0.3. The teacher's share of the target stays on the logged feeds.

The target of a pseudo-feed is an item-card view from the main-page feed among the 64 newest history events. Its history starts 16 to 64 events further back, and its other candidates are those of the logged feed that were not clicked.

The idea had been closed once. As a family of click-label members it read +0.00108 ± 0.00072 / +0.00024 per member on 14 November and failed its rule: the holdout gain was below one SE, and the members differed too little from the base (diversity margin +0.0024 against a bar of +0.005). On 17 November it was tried inside the distilled students, with a prior of 0.02 for +0.001 on public written in the ledger before the runs.

| Reading | Result |
|---|---|
| Member, 2 seeds of held-out students | +0.00092 ± 0.00060 on validation |
| Member, 8 seeds | +0.00046 ± 0.00031 on validation; 7 of 8 seeds positive |
| Pool of 8 against 8 plain students of the same seeds | +0.00054 ± 0.00047 / +0.00010 ± 0.00027; pooled z +1.08, randomization p 0.19, rule failed |
| FDXH16P20:ps, 8 of the 16 all-data full-view students replaced | validation +0.00028 ± 0.00029; public -0.00010 |
| FDX16P20:ps, all 16 replaced | validation +0.00015 ± 0.00036, public-mix view +0.00001, private-mix view +0.00004; public +0.00037, paired SE 0.00047 |
| FDXU32P20:ps, the 16 added to those of FD16P20 | validation +0.00029 ± 0.00026; public +0.00035, paired SE 0.00034 |

The three file rows are read against FD16P20 (validation 0.40271, public 0.394892).

The public evidence is weak. The two replacement files differ by about 0.0005 on public, inside the paired noise, and in the order opposite to their validation readings. On validation the full replacement is below one SE, and the 8-seed pool, at 1.2 SE, failed the rule written for it.

FDX16P20:ps scored 0.39526061, the team's best public score and 0.00005 above pick 1, a tie. It became pick 2 on 19 November in place of W8P20, for a higher expected level (+0.0007 on validation and +0.00025 on public over W8P20) and 16 full-view students from runs separate from those of pick 1 ([11-final-picks.md](11-final-picks.md)).

A pseudo-feed student trained in 108.8 to 114.5 minutes alone on a GPU. Code: the lab trainer `lab/nn_x/w12/pseudokd/train_full_pk.py`, because `avito_ranker/training/train.py` has no pseudo-feed option; `configs/ensembles/fdx16p20_ps.toml` rebuilds the file from the stored member predictions; [distillation](../experiments/distillation/).

## In the final files without a measured gain

Training on all train feeds, the 30,000 holdout feeds included, was done so that the final files use every labelled feed. Both picks are trained this way. From 18 to 19 November pick 2 was W8P20, for which the rule was waived ([11-final-picks.md](11-final-picks.md)).

The all-data file FD16P20 scored 0.00012 below W8P20, whose members never trained on the holdout feeds, on public at a paired SE of 0.00056, and the pool of 16 all-data members read +0.00004 ± 0.00042 on validation against their held-out twins. It cost 81.8 GPU-hours of training for 20 members, and it cost the holdout as a check on them ([full-data-retrains](../experiments/full-data-retrains/)).

Two other choices tied. Sixteen students in place of eight read -0.000003 on public at a paired SE of 0.00045 ([pool-size-and-growth](../experiments/pool-size-and-growth/)). Averaging each fine-tune over 3 seeds changed a member by 0.00000 ± 0.00004 and the pool's public score by -0.00007.

Self-attention across the candidates of a feed was in the first model, where removing it changed nothing (0.37386 without, 0.37361 with). In the final click-label recipe removing it cost 0.0020 / 0.0014. The records do not say what changed in between.

The domain embedding tells the model whether a feed is in production or random order. It and the two columns that flag the pooled anonymous user (user 0) were neutral when added, and the removal of the domain embedding was inconclusive (-0.0004 / 0.0000).
