# Final picks

In brief:

- Pick 1 is FD16P20@m0.4_0.2: public 0.3952092, validation 0.40293, chosen on 2026-11-18.
- Pick 2 is FDX16P20:ps: public 0.39526061, the team's best score and rank 9 of 541 teams on the board of 2026-11-19 03:34 UTC (leader 0.3982556); validation 0.40286. It replaced W8P20 on 2026-11-19.
- The two picks tie on public (0.00005 apart). Pick 1 leads FD16P20, pick 1 until 2026-11-18, by +0.00032 on public, about one standard error; about +0.00005 of that lead is expected on private (20,033 feeds, unknown until 2026-12-09).

Scores, splits and notation as in [04-validation.md](04-validation.md); terms in [glossary.md](glossary.md). The metric is NDCG (normalised discounted cumulative gain) over each feed's full list, averaged over feeds: 14,423 labelled feeds for validation, 8,562 test feeds for public.

A paired standard error (SE) is the standard error of the mean per-feed difference between two files, and "±" gives one paired SE. A short feed has fewer than 50 candidates, a long feed 50 or more.

## The two files

Pick 1 was fixed on 2026-11-18 at 02:33 UTC and pick 2 on 2026-11-19 at 03:32 UTC. The person on the team, who uploads the files, chose them from options that the lead session set out in the decision ledger. The lead session is the Claude Code session that ran the work ([13-how-the-work-was-run.md](13-how-the-work-was-run.md)).

FD16P20, pick 1 until 2026-11-18, and W8P20, pick 2 until 2026-11-19, stay in the table for comparison. All four files were written to `artifacts/submissions/`, which is not versioned, so they are not in this repository.

Each file z-scores predictions within each feed (subtracts the feed mean and divides by the feed standard deviation) and averages two branches 0.5 / 0.5: the fine-tuned branch, whose members were trained further on the validation days, and the train-only branch. In each branch a pool of full-view students is mixed with a pool of popularity-light students ([03-solution.md](03-solution.md#ensemble)).

A student is a ranker trained toward a target that is half clicks and half a teacher's averaged scores ([03-solution.md](03-solution.md#distillation)). The teacher is an ensemble of 11 or 25 earlier rankers trained on clicks alone. A full-view student reads all 34 pre-computed inputs. A popularity-light student reads 27, with one clipped exposure count in place of the 8 popularity inputs.

A pseudo-feed student is a full-view student that also trains on extra lists built from user histories ([Why pick 2](#why-pick-2)).

| | Pick 1: FD16P20@m0.4_0.2 | Pick 2: FDX16P20:ps | FD16P20, pick 1 until 2026-11-18 | W8P20, pick 2 until 2026-11-19 |
|---|---|---|---|---|
| File name in the records | `cand_FD16P20_m04_02kd_r4.parquet` | `cand_FDX16P20_pskd_r4.parquet` | `cand_FD16P20kd_r4.parquet` | `cand_W8P20kd_r3.parquet` |
| Members | 16 full-view students and 4 popularity-light students, all width 192 | 16 pseudo-feed full-view students and the 4 popularity-light students of pick 1, all width 192 | the 20 members of pick 1 | 8 full-view students (width 192) and 4 narrow popularity-light students (width 128) |
| Weight of the popularity-light pool | 0.4 on feeds of fewer than 50 candidates, 0.2 on the others | 0.2 | 0.2 | 0.2 |
| Train feeds per member | 3,151,208 (all) | 3,151,208 (all), and about 21.7 million pseudo-feeds per full-view student | 3,151,208 (all) | 3,121,208 (the 30,000 holdout feeds left out) |
| Checkpoint kept | last step, 99,840 | last step, 99,840 | last step, 99,840 | chosen on the holdout among steps 92,500, 97,500 and 98,350 |
| Validation NDCG | 0.40293 | 0.40286 | 0.40271 | 0.40214 |
| Public NDCG | 0.3952092 | 0.39526061 (team best) | 0.394892 | 0.3950094 |
| Public minus the level line | -0.00021 | -0.00009 | -0.00032 | +0.00035 |
| Rank correlation with pick 1 on test | | 0.9949 | 0.9984 | 0.9931 |
| Same top item as pick 1 | | 91.4% of test feeds | 98.2% of test feeds | 89.3% of test feeds |

The level line is public = 0.0074 + 0.963 × validation, fitted on 18 scored files, with a residual standard deviation (sd) of 0.0006 ([04-validation.md](04-validation.md#noise)). Rank correlation is Spearman within each test feed, averaged over the 28,595 feeds. The holdout is a fixed set of 30,000 train feeds shown in random order, as validation and test were; members that leave it out use it to choose their checkpoint.

The submission files are not in this repository. The sha256 of each file, as recorded when it was built, lets a rebuild be compared with it:

```
3cf6faccbad1357ae1a1b11999fef8168835d77407684832c7578c832c6e71b3  cand_FD16P20_m04_02kd_r4.parquet
8235c55cb35931129a845ce6825a94913f81bebb51824f84bced6eadfb1571db  cand_FDX16P20_pskd_r4.parquet
14adc19ac794e3977cc0275a0c15d2c76e55c5dcdeec911dc47a8193128626a8  cand_FD16P20kd_r4.parquet
406e5705cef75d264118eeb20102ab963cedf6d00b0b7c6be7c2f1b1361a4d78  cand_W8P20kd_r3.parquet
```

## Why pick 1

Pick 1 has the same 20 members as FD16P20 and changes one weight. The two files give the same order on all 17,434 test feeds of 50 or more candidates and differ on 8,364 of the other 11,161. On 20,231 of the 28,595 test feeds (71%) the order is identical, and no member was retrained, so pick 1 adds no training risk.

Every reading puts pick 1 ahead of FD16P20, none by much more than one SE:

- validation: +0.00022 ± 0.00024, all of it from the 5,893 feeds of fewer than 50 candidates (+0.00053 ± 0.00058 there);
- public-mix and private-mix views (validation re-weighted so that feed size and the share of cold candidates match the public or the private part of test): +0.00027 and +0.00033;
- public: +0.00032 at a paired SE of 0.00031.

The holdout and the lagged holdout (the holdout with its popularity inputs rebuilt as stale as on test) hold only feeds of 50 or more candidates and cannot read the change.

Both leads are the best of several tries:

- The 0.40271 of FD16P20 is a favourable draw of its members. Pools of 32 and 48 full-view students of the same recipe read 0.40237 and 0.40222 on validation ([07-what-failed.md](07-what-failed.md#18-growing-the-ensemble-past-16-members)).
- The weight 0.4 is the high point of a validation curve on which 0.3 and 0.5 read 0.00027 and 0.00042 lower ([03-solution.md](03-solution.md#the-two-input-views) has the curve).
- 0.3952092 was the best public score of the 14 scored files of the family when pick 1 was chosen. Of the 17 scored by 2026-11-19 it is third, 0.00005 below pick 2 ([Risks](#risks)).

After both discounts the expected private difference between pick 1 and FD16P20 is about +0.00005, a tie within noise. The team took pick 1 for that small expected gain, since it changes no member.

### The all-data rule

The members were made under the rule set on 2026-11-15 that every final file is trained on all train feeds. Until then each member left out the 30,000 holdout feeds, which chose its checkpoint and gave a second read next to validation.

Those feeds are 0.95% of the train feeds but 18% of the random-order pool (134,118 feeds without them, 164,118 with them). The cooldown, the last 20% of training steps, draws 2% of its batches from that pool. The expected gain from adding the holdout feeds was +0.00005 to +0.0002 on validation.

An all-data member has no out-of-sample read. It keeps its last step, since nothing is left to choose a checkpoint on, and it was accepted against its twin: the run with the same command and seed that leaves the holdout feeds out. [03-solution.md](03-solution.md#the-final-files) has the check and its numbers, [experiments/full-data-retrains/](../experiments/full-data-retrains/) its code.

The composition of FD16P20 was written down at 04:52 UTC on 2026-11-15, before any of its 20 members was trained: 16 full-view students (8 seeds for each of the two teachers), 4 popularity-light students at weight 0.2, branches mixed 0.5 / 0.5.

### The replacement rule was set aside

Pick 1 did not come through the rule written for replacing FD16P20 (the freeze protocol, a working record that is not published). A replacement had to meet four conditions:

1. Its recipe change passed a rule written before its run.
2. It passes the pre-submission gate ([04-validation.md](04-validation.md#the-pre-submission-gate)).
3. Its validation is at or above FD16P20 minus one paired SE.
4. If scored, its public score is at or above 0.394292 (FD16P20 minus 0.0006).

A public score alone could promote a near copy ([glossary](glossary.md)) only from 0.395992 upward (FD16P20 plus 0.0011), because +0.0011 is the smallest difference the public board can detect between near-copy pools at 80% power.

Pick 1 meets conditions 2 to 4, fails condition 1, and its public score is below 0.395992. The weight 0.4 on short feeds was read off a validation curve and built on 2026-11-17 as a try file, scored for information with no rule written for it. It became pick 1 after its public score, so the choice is post hoc.

## Why pick 2

The rules text does not settle whether one final counts or the better of two. One clause (7.4.2) is in the singular, the FAQ in the plural. If only one counts, pick 2 has no effect.

If the better of two counts, a second file is worth most when it differs from the first and stands at the same level. Pick 2 was W8P20 from 2026-11-18 and is FDX16P20:ps from 2026-11-19, after FDX16P20:ps scored 0.39526061 on public.

### The pseudo-feed students

FDX16P20:ps is FD16P20 with its 16 full-view students replaced by 16 students of the same two recipes, teachers and seeds (951 to 958 for each teacher) that also train on pseudo-feeds.

In the first 79,872 steps each batch adds up to one pseudo-feed per real feed, built from that feed's own history. A recent card the user opened from an earlier feed becomes the one positive, the events older than it by a gap of 16 to 64 become the history, and the feed's non-clicked candidates are the negatives.

The pseudo-feeds enter the loss at weight 0.3, and each student saw about 21.7 million of them ([03-solution.md](03-solution.md#the-pseudo-feed-students-of-the-second-pick) has the construction). The trainer is `lab/nn_x/w12/pseudokd/train_full_pk.py`; the package trainer has no such option.

The idea had failed its rule. As held-out students of the 11-member teacher, an 8-seed pool read +0.00054 ± 0.00047 on validation and +0.00010 ± 0.00027 on the holdout (randomization p 0.19), and the idea was closed on 2026-11-17 at 22:04 UTC. The person on the team asked for try files anyway. The students of the 25-member teacher were never screened.

The composition of FDX16P20:ps was written into the ledger at 22:50 UTC that day, before any all-data pseudo-feed student existed, and the file was built on 2026-11-18 at 03:14 UTC. FDXH16P20:ps, with only the 8 students of the 11-member teacher replaced, read +0.00028 ± 0.00029 on validation against FD16P20 and scored 0.00010 below it on public. FDX16P20:ps read +0.00015 ± 0.00036 and scored 0.00037 above it.

FDX16P20:ps scored 0.00047 above FDXH16P20:ps on public, 1.3 times their paired SE of 0.00036, after reading 0.00012 below it on validation. That is inside noise, so public cannot tell whether the pseudo-feed students helped. FDX16P20:ps became pick 2 after its public score, so the choice is post hoc.

### Choosing between W8P20 and FDX16P20:ps

The table takes the private difference between a second file and pick 1 as normal, with the paired SE of the validation per-feed differences scaled to the 20,033 private feeds. It gives the expected gain of the better of the two files over pick 1 alone.

| | FDX16P20:ps (pick 2) | W8P20 (pick 2 until 2026-11-19) | FD16P20, as a second file |
|---|---|---|---|
| Members shared with pick 1 | 4 of 20 (the popularity-light students); the 16 full-view students are separate runs on all train feeds | 0 of 12 (same recipe for the full-view students, separate runs without the holdout feeds) | 20 of 20 |
| Validation minus pick 1 | -0.00007 ± 0.00037 | -0.00079 ± 0.00045 | -0.00022 ± 0.00024 |
| Public minus pick 1 | +0.00005, paired SE 0.00048 | -0.00020, paired SE 0.00059 | -0.00032, paired SE 0.00031 |
| Paired SE against pick 1 on 20,033 private feeds | 0.00031 | 0.00038 | 0.00020 |
| Gain of the better of two over pick 1 alone, at equal true level | +0.00013 | +0.00015 | +0.00008 |
| The same if the true level of the second file is 0.0002 lower | +0.00005 | +0.00007 | +0.00002 |

The last three rows count feed-sampling noise only.

At equal true level the private difference D between a second file and pick 1 is normal with mean 0 and standard deviation s, the paired SE of the table. The better of the two then gains the mean of the positive part of D, s/√(2π), about 0.4 s: 0.4 × 0.00038 ≈ 0.00015 for W8P20 and 0.4 × 0.00020 = 0.00008 for FD16P20.

A file that differs more from pick 1 has a larger s and so a larger expected gain. The last row moves the mean of D to -0.0002.

At equal level W8P20 is worth a little more, because it differs more from pick 1. The levels differ: FDX16P20:ps reads +0.00072 ± 0.00045 over W8P20 on validation and +0.00025 on public. With W8P20 0.0001 to 0.0002 below pick 1 and FDX16P20:ps level with it, either second file is worth about +0.0001 (+0.00007 to +0.00011 for W8P20, +0.00013 for FDX16P20:ps).

The person on the team took FDX16P20:ps on 2026-11-19 at 03:32 UTC: the two values are about equal, its expected level is higher, and its 16 full-view students come from runs separate from pick 1. Both picks now meet the all-data rule.

Against FDX16P20:ps:

- It is closer to pick 1 than W8P20 was. It shares 4 members, and its full-view students follow the same recipes, teachers and seeds; the rank correlation with pick 1 on test is 0.9949, against 0.9931 for W8P20.
- Its public score is the top of 17 near copies, and the pseudo-feed idea failed its rule on validation ([Risks](#risks)).
- W8P20 sat above the level line, and with it gone no final file is free of a cost of all-data training on test, if there is one ([level line](#all-data-files-sit-below-the-level-line)).
- Its 16 full-view students can be retrained only with the lab trainer ([retraining](#pick-2-cannot-be-retrained-with-the-package)).

## Risks

The risks on record, each told in a subsection below:

- Both picks are among the top three of 17 near copies on public. The best of 17 equal files is expected 0.00044 above their mean and pick 2 sits 0.00040 above it; about +0.00005 of pick 1's lead over FD16P20 is expected on private ([best of 17](#both-picks-are-among-the-best-of-17-near-copies)).
- Two changes to the popularity-light part on long feeds scored about 0.0004 below their validation reading, and the short-feed change of pick 1 may be the same kind of artefact. Pick 2 does not carry it ([long-feed changes](#long-feed-changes-scored-below-validation)).
- The 0.2 popularity-light part of both picks entered after a public read, against a holdout read of -0.00063 ± 0.00022. If the holdout is right, both finals lose about 0.0005 ([the 0.2 part](#the-popularity-light-part-entered-after-a-public-read)).
- Files with all-data members sit 0.00031 below the level line on average, and neither a gain nor a cost of all-data training is shown. Since 2026-11-19 both picks are all-data files ([level line](#all-data-files-sit-below-the-level-line)).
- Both picks are one model family (rank correlation 0.9949 on test, 4 shared members), and private inputs are staler than public ones, so pick 2 does not hedge a shortfall the family shares ([one family](#both-picks-are-one-model-family)).
- The 16 full-view students of pick 2 can be retrained only with the lab trainer ([retraining](#pick-2-cannot-be-retrained-with-the-package)).
- A member's validation path and test path differ by design ([paths](#validation-and-test-paths-differ)).

### Both picks are among the best of 17 near copies

Seventeen scored files belong to the final family, full-view students plus a popularity-light pool. They are near copies ([glossary](glossary.md)): files so close that neither validation nor the board can order them.

| File | What differs from FD16P20 | Members trained on all data | Validation | Public | Public minus level line | Public minus FD16P20, in paired standard errors (that error) |
|---|---|---|---|---|---|---|
| FDX16P20:ps | all 16 full-view students replaced by students of the same recipes that also train on pseudo-feeds (pick 2) | 20 of 20 | 0.40286 | 0.39526061 | -0.00009 | +0.8 (0.00047) |
| FDXU32P20:ps | the 16 pseudo-feed students of pick 2 added to the 16 full-view students | 36 of 36 | 0.40301 | 0.39524613 | -0.00025 | +1.0 (0.00034) |
| FD16P20@m0.4_0.2 | popularity-light weight 0.4 on feeds of fewer than 50 candidates (pick 1) | 20 of 20 | 0.40293 | 0.3952092 | -0.00021 | +1.0 (0.00031) |
| A1 | pick 1, with the popularity-light part of the fine-tuned branch taken from the train-only predictions on feeds of 50 or more candidates (A1) | 20 of 20 | 0.40310 | 0.395036 | -0.00055 | +0.4 (0.00036) |
| W8P20 | W8P20 (pick 2 until 2026-11-19) | 0 of 12 | 0.40214 | 0.3950094 | +0.00035 | +0.2 (0.00056) |
| FDX16P20:hz | the 16 full-view students with their cooldown re-run on staler popularity inputs ([07](07-what-failed.md#19-a-cooldown-on-stale-popularity-inputs)) | 20 of 20 | 0.40224 | 0.3949761 | +0.00022 | +0.2 (0.00045) |
| FD24P20 | 8 more full-view students, new seeds | 28 of 28 | 0.40258 | 0.394905 | -0.00018 | +0.0 (0.00028) |
| FD16P20 | FD16P20 itself | 20 of 20 | 0.40271 | 0.394892 | -0.00032 | reference |
| FD48P20 | 32 more full-view students, new seeds | 52 of 52 | 0.40222 | 0.39484072 | +0.00010 | -0.1 (0.00036) |
| FDX16P20:ref | the 16 full-view students trained again from their step-77,500 checkpoints to the end (a cooldown re-run) | 20 of 20 | 0.40214 | 0.39482063 | +0.00016 | -0.2 (0.00036) |
| FDXH16P20:ps | the 8 full-view students of the 11-member teacher replaced by students that also train on pseudo-feeds built from user histories | 20 of 20 | 0.40298 | 0.3947873 | -0.00068 | -0.3 (0.00038) |
| FDX16P20:fsw50t | the same cooldown re-run with each feed's loss weighted by the square root of its number of candidates | 20 of 20 | 0.40212 | 0.39475033 | +0.00011 | -0.3 (0.00042) |
| FD16P20@m0.2_0.4 | popularity-light weight 0.4 on feeds of 50 or more candidates | 20 of 20 | 0.40298 | 0.3947465 | -0.00072 | -0.5 (0.00027) |
| ENSA68 | 32 full-view and 36 popularity-light students | 44 of 68 | 0.40247 | 0.3946427 | -0.00034 | -0.7 (0.00036) |
| FD16P20vf | 4 popularity-free students, which read no popularity input, added to the popularity-light part (FD16P20vf) | 24 of 24 | 0.40286 | 0.394630 | -0.00072 | -1.0 (0.00026) |
| FDM32P20 | 16 more full-view students, trained with one extra input | 36 of 36 | 0.40264 | 0.394578 | -0.00056 | -0.9 (0.00034) |
| FDN16P20 | all 16 full-view students trained with the extra input | 20 of 20 | 0.40283 | 0.3943559 | -0.00097 | -1.1 (0.00050) |

The 17 public scores have mean 0.39486 and sd 0.00025. The paired SE between two of these files is 0.00018 to 0.00056 (0.00018 between pick 1 and A1, 0.00048 between the two picks, 0.00026 to 0.00056 against FD16P20). Each paired SE in the last column comes from the validation per-feed differences scaled to the 8,562 public feeds ([04-validation.md](04-validation.md#noise)).

If the 17 files had one true level, the best of them would be expected about 1.8 sd (0.00044) above their mean. Pick 2 sits 1.6 sd (0.00040) above it and pick 1 1.4 sd (0.00035). The order at the top is about what choosing the best of 17 produces by itself.

When pick 1 was chosen it was the best of 14, 1.9 sd above their mean against 1.7 expected. The +0.00005 expected on private for its lead over FD16P20 assumes that the lead is selection.

The figure draws all 17 files with bars of one and two paired SE of their difference to FD16P20.

![The 17 scored files of the final family on public, from 0.3943559 to 0.39526061, each with the paired standard error of its difference to FD16P20; every difference is within 1.1 paired standard errors, and the two final picks, FD16P20@m0.4_0.2 and FDX16P20:ps, are marked.](figures/near-copies-noise.svg)

### Long-feed changes scored below validation

Changes to the popularity-light part on feeds of 50 or more candidates have twice scored about 0.0004 below their validation reading on public:

- FD16P20@m0.2_0.4, with weight 0.4 on those feeds, read +0.00026 against FD16P20 on validation and scored 0.00015 below it.
- A1 read +0.00017 against pick 1 on validation and scored 0.00017 below it.

Pick 1 keeps 0.2 on long feeds, and lower long-feed weights read worse on validation (weight 0.1 reads -0.00025 ± 0.00013 and weight 0 reads -0.00068, both against pick 1). Nothing on record explains why public would treat a change on short feeds differently.

If the short-feed gain of pick 1 is the same kind of validation artefact, pick 1 can land below FD16P20 on the 39.2% of private feeds that are short. On the other feeds the two files give the same order. Pick 2 keeps 0.2 on every feed and does not carry this risk.

### The popularity-light part entered after a public read

The rule written before W8P20 was scored asked for +0.0008 over the same pool without the 0.2 part (W8) to put the design into pick 1. W8P20 scored +0.00052 over W8 (1.3 SE), which by that rule left it a candidate for pick 2 at most.

Ledger entries of 2026-11-15 at 02:15 UTC record that score and a new rule for this family of changes: the holdout is still reported but is no longer a veto. The same night the 0.2 part was written into the all-data design. [04-validation.md](04-validation.md#rules-written-before-the-read) tells the holdout's bias and the audit of its vetoes in full.

Both picks carry the 0.2 part against a holdout read of -0.00063 ± 0.00022. The team put that read down to the holdout's fresh popularity inputs (1.75 days old, against 9.34 on public). If that explanation is wrong, it costs about 0.0005 on both finals.

### All-data files sit below the level line

Of the 16 scored files in the table with all-data members, 12 sit 0.00009 to 0.00097 below the level line and 4 sit 0.00010 to 0.00022 above it, with a mean of -0.00031.

Three student files trained without the holdout feeds sit above it: W8P20 at +0.00035, its full-view pool alone at +0.00030 with 8 students and +0.00036 with 16. A fourth, the pool that adds 8 narrow full-view students (width 128) to the 8 wide ones, sits at -0.00019.

Part of the gap may be validation noise. Among the all-data files the residual falls as validation rises (correlation -0.74 over the 16), and the four above the line are the four with the lowest validation (0.40212 to 0.40224). Validation noise that public does not share produces that pattern. At equal validation the gap is smaller: FDX16P20:ref, whose cooldown was re-run, reads 0.40214 on validation like W8P20 and sits 0.00019 below it on the residual.

The direct comparison, FD16P20 minus W8P20, reads +0.00058 ± 0.00043 on validation and -0.00012 ± 0.00056 on public. Neither a gain nor a cost of training on all data is shown. The plot of validation against public with the level line, for the 38 scored files with a clean validation score, is in [04-validation.md](04-validation.md#noise).

W8P20 hedged this gap until 2026-11-19. Both picks are now all-data files, pick 1 at 0.00021 and pick 2 at 0.00009 below the line.

### Both picks are one model family

The per-feed rank correlation between the two picks on test is 0.9949, and they put the same item on top in 91.4% of test feeds. They share the 4 popularity-light students, and their full-view students follow the same recipes, teachers and seeds. The popularity inputs of the private feeds are staler than those of the public feeds (11.23 against 9.34 days), and no member was trained to see them that stale.

If the family falls short on private for a reason its members share, pick 2 does not cover it. W8P20 (0.9931) did not either.

### Pick 2 cannot be retrained with the package

The package trainer (`avito_ranker/training/`) has no pseudo-feed option. The 16 full-view students of pick 2 were trained with `lab/nn_x/w12/pseudokd/train_full_pk.py`, a copy of the lab trainer with the pseudo-feed flags and the all-data patch. Their published read-outs, `lab/nn_x/w5/logs/<member>.json`, record every argument. Their checkpoints were kept in the lab's own checkpoint folder, outside `artifacts/members/`, and were not re-scored with the package.

`configs/ensembles/fdx16p20_ps.toml` rebuilds the file byte for byte from its 80 stored member predictions, and the gate prints the text it printed when the file was built. A retrain with either trainer is a new draw, not the stored weights ([03-solution.md](03-solution.md#known-limits)).

### Validation and test paths differ

The validation path and the test path of a member differ by design: for example, 3 fine-tuned models score a validation row and 15 score a test row ([03-solution.md](03-solution.md#known-limits)). The public scores of one pool at fine-tune weights 0, 0.5 and 1 are the check that the fine-tuned branch works on test ([03-solution.md](03-solution.md#fine-tune-on-the-validation-days)).

## What the audit checked and found

On 2026-11-16 a separate audit agent checked FD16P20, FD16P20vf and W8P20 with code written for that purpose: its own within-feed z-score, pooling and NDCG, with no call into the builder that made the files. It ran on CPU and wrote nothing into the working tree. The verdict was the same for each file: suitable, with notes.

The audit's report is a working record and is not published.

The audit predates pick 1, and its code was not run on it. Pick 1 has the members of FD16P20, so the member-level findings below (inputs, logs, training data, fine-tune folds) hold for pick 1 unchanged.

The audit predates pick 2 as well, and 16 of its 20 members are new. For them the gate's checks stand in: every member's logged inputs pass the banned list and the quarantine check, and the file has the sample submission's keys, no tie and the order of its stored pools (minimum per-feed Spearman 0.999991). The 59-column leak test below was not run on them.

The file-level checks were repeated with the package on 2026-11-18 for pick 1 and on 2026-11-19 for pick 2. Pick 1 and its four pool files (fine-tuned and train-only branch scores on validation and test), rebuilt from the 80 stored member predictions of FD16P20, are byte-identical to the stored ones. The gate passes with 0 failures and 0 warnings and prints the text it printed when the file was built.

Pick 2 and its four pool files, rebuilt from its 80 stored member predictions, are byte-identical to the stored ones as well, and its gate text equals the one stored at build time.

### Each file is a valid submission

For each of the three audited files:

- Its sha256 equals the one recorded when it was built.
- It has 2,904,806 rows in 28,595 feeds and the same (feed_id, item_id) pairs as the sample submission, with no null, NaN or infinite score, no duplicate pair, no constant feed and no tie.
- Rebuilt from the stored member predictions (80, 96 and 48 arrays for FD16P20, FD16P20vf and W8P20), the branch scores agree with the stored ones within 1.2e-6, and validation comes out at 0.40271, 0.40286 and 0.40214.
- A re-run of the gate printed PASS with 0 failures and 0 warnings, byte-identical to the output stored at build time.

The rebuilt order differs from the submitted one in 2 rows of 1 feed, 8 rows of 4 feeds and 6 rows of 3 feeds. These are float32 ties, which the submission writer breaks with a fixed random number per row. The package in `avito_ranker/` keeps the writer's tie-breaking and rebuilds all three files byte for byte from the stored member predictions.

### No leak found

The audit read the training log of every member and compared the logged inputs with the banned names and with the quarantined files, matched by inode because inputs are hard-linked. The full-view students read the 34 standard inputs, the popularity-light students 27 and the popularity-free students 26. None reads a banned or quarantined file.

As a direct test it took the 117 validation feeds whose user has a later feed, where a leak from later snapshots would show. On each of 59 input columns it compared the within-feed percentile of the positives with the other feeds. The largest |z| is 2.45. The positive control, the banned form of the history count that includes the same user's later snapshots, reads z = +3.7.

The gate's own leak probe (gain on those 117 feeds minus gain on the rest, against a reference file) has little power with 117 feeds. It reads +0.00552 ± 0.00463 for FD16P20, +0.00125 ± 0.00562 for FD16P20vf, -0.00052 ± 0.00110 for W8P20, +0.00613 ± 0.00469 for pick 1 and +0.00308 ± 0.00365 for pick 2.

The pseudo-feeds of pick 2 are built from each train feed's own history and candidates. They read no later snapshot of the user and no validation or test row.

### Validation labels enter only through the fine-tune

The fine-tune is split into 5 folds by user. The stored validation prediction of a row comes from the 3 models, one per seed, that never saw its user, at the final averaged weights and with no choice of epoch. No pseudo-labels from test are used.

The fine-tuned branch of FD16P20 scores 0.4019 on validation and the train-only branch 0.4021, so the fine-tuned branch shows no sign of having seen its own validation labels.

### When each design choice was fixed

- The member counts, the weight 0.2 and the 0.5 / 0.5 mix of FD16P20 were on record before its members were trained.
- The 0.2 was set on 2026-11-14 at 20:16 UTC without a scan, before the first file with it was built. It lies between an earlier estimate of 0.15 and the 0.29 share of popularity-light members in C3, the best file of rankers trained on clicks alone (8 of its 28 members).
- The 0.5 / 0.5 mix dates from 2026-11-11. Three later public scores of one pool at fine-tune weights 0, 0.5 and 1 put the optimum at 0.49 ([03-solution.md](03-solution.md#fine-tune-on-the-validation-days)).
- One choice was post hoc and was labelled so when it was made: width 192 for the students, which missed its bar on validation by 0.00001 and went in on its holdout read ([04-validation.md](04-validation.md#rules-written-before-the-read)).

The weight 0.4 on the short feeds of pick 1, chosen after the audit, is a second post hoc choice ([Why pick 1](#why-pick-1)).

Pick 2 is a third. Its composition was written into the ledger on 2026-11-17 at 22:50 UTC, before any of its all-data students was trained, but as a try file after the pseudo-feed idea had failed its rule, and it became pick 2 after its public score ([Why pick 2](#why-pick-2)).

### Training scope in the logs

All 20 members of FD16P20 and all 24 of FD16P20vf report 3,151,208 train feeds and keep step 99,840, the last. All 12 members of W8P20 report 3,121,208 feeds and 98,350 steps, with the checkpoint chosen on the holdout and never on validation.

All 16 pseudo-feed students of pick 2 report 3,151,208 train feeds, keep step 99,840, the last, and log 21,721,131 to 21,723,022 pseudo-feeds each. Its 4 popularity-light students are those of FD16P20.

For the all-data members these numbers are in the published read-outs `<member>_fulldata.json` (keys `n_training_feeds` and `kept_step`), under `lab/nn_x/w5/logs/` for full-view students and `lab/nn_x/abl/logs/` for the others. The raw training logs are not published.

## What would change the choice

- If only one final counts, pick 2 has no effect. If the platform took the two best public scores without a selection by the team, the finals would be pick 2 and FDXU32P20:ps (0.39524613), which share 20 members. The team therefore selects both files itself in the selection window.
- No model-side screen is open. The last one, routed experts and depth 7 for the students, closed on 2026-11-19 at 8 seeds with no try file built ([05, phase 9](05-timeline.md#phase-9-mixture-of-experts-and-depth-19-november-from-0400)). Five ideas that never had a fair training run are listed in [12, section 6](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed); they have no experiment behind them and no part in this choice.
- If a check fails on freeze day, the selection stops. On 2026-12-07, `python scripts/freeze_check.py configs/ensembles/fd16p20_m04_02.toml configs/ensembles/fdx16p20_ps.toml` hashes the two files again and re-runs the gate. A sha256 that differs from the list above means that the file on disk is not the file that was scored. A run on 2026-11-19 passed both files.
- If the team merges with another team, the merged file enters as a new candidate under the same gate, and both picks are chosen again. The decision is due by 2026-12-01.
- Rule 7.4.1 lets teams form until 2026-12-08 23:59 MSK, but rule 5.5 bars any disclosure of the solution, including passing it to another participant, later than 7 calendar days before the end of the period in rule 7.4 (2026-12-09 12:00 MSK). Disclosure after 2026-12-02 12:00 MSK (09:00 UTC) leads to disqualification, so a merge that exchanges predictions has to be settled before then.
- Otherwise a pick changes before the freeze only by a new decision, written into the decision ledger with its reason.

| Date (UTC) | Event |
|---|---|
| 2026-12-01 | decision on a team merge, as set in the team's records |
| 2026-12-02 09:00 | 12:00 MSK: from here on, any disclosure of the solution, passing it to another participant included, leads to disqualification (rule 5.5) |
| 2026-12-07 00:00 | freeze: no new candidate after this time; freeze check of both picks |
| 2026-12-08 20:59 | uploads close |
| 2026-12-08 21:00 to 2026-12-09 09:00 | the final files are selected on the platform |

## Earlier picks and candidates

FD16P20 was pick 1 until 2026-11-18 and W8P20 pick 2 from 2026-11-18 to 2026-11-19; both stay in the table of [the two files](#the-two-files).

### W8P20, pick 2 from 2026-11-18 to 2026-11-19

W8P20 shares none of its 12 members with pick 1 and was the strongest file from other runs when the picks were first fixed. For it the team waived the rule that every final file is trained on all data.

It also covered a doubt about pick 1. Files with all-data members sit below the level line on average, and W8P20 sits above it ([Risks](#risks)). If part of that gap is a cost on test of training on all data, W8P20 does not carry it.

Against W8P20, as recorded on 2026-11-18:

- It has 12 members where pick 1 has 20, trained without 18% of the random-order pool.
- Its popularity-light students are narrow (width 128).
- Its 8 full-view students are a favourable validation draw of their recipe: a pool of 12 students of that recipe reads 0.00024 lower.
- The trainer source was edited on 2026-11-15, the day after those 8 students were trained. W8P20 can be rebuilt from its stored predictions but not retrained from identical source.

### The plan before 2026-11-18

Until 2026-11-18 FD16P20 was pick 1. Pick 2 was to be chosen at the freeze between W8P20 and FD16P20vf (validation 0.40286, public 0.394630). FD16P20vf is FD16P20 with 4 popularity-free students, which read none of the popularity inputs, added to its 0.2 part.

FD16P20vf kept the all-data rule, but its validation edge came from one draw of 4 popularity-free students that fresh seeds did not repeat. The first draw put the file above 70 of 70 control files, the replication above 16 of 70 ([04-validation.md](04-validation.md#rules-written-before-the-read)). It puts the same item on top as FD16P20 in 97.8% of test feeds.

The audit of 2026-11-16 recommended W8P20 for pick 2 if the all-data rule was dropped for the second file, and called FD16P20vf admissible otherwise.

A second final from members disjoint from FD16P20 (FDB16P20: 16 new full-view and 4 new popularity-light students) read -0.00052 ± 0.00039 on validation against FD16P20. It missed its admission bar of one SE and was never scored.
