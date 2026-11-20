# Timeline

In brief:

- The work ran for nine days, 11 to 19 November 2026, and 40 files were scored on the public board. Public NDCG rose from 0.3739 to 0.39526061, and 0.018 of the 0.021 gain came on the first day.
- On the board of 19 November, 03:34 UTC, 0.39526061 is at rank 9 of 541 teams.
- The final picks are FD16P20@m0.4_0.2, chosen on 18 November at 02:33 UTC, and FDX16P20:ps, which replaced W8P20 as pick 2 on 19 November at 03:32 UTC.

The research, code, experiments, analysis and documentation were done by one lead session of Claude Code (versions 2.1.303 to 2.1.325) running Claude Opus 5.5 at the xhigh reasoning effort in "ultracode" mode, which lets the session write and launch multi-agent workflows. It was not an unattended search: when a request was finished, the session stopped and waited for the next message. [13-how-the-work-was-run.md](13-how-the-work-was-run.md) tells it in full.

The person on the team sent about 110 short requests and questions in the nine days, wrote no code, ran no experiment and analysed no result. The person uploaded the files to the platform, which takes uploads only from a person, and passed the scores back. The same person answered a handful of decision questions (the final picks, the license, the repository's visibility).

The phases below say where a request from that person started a line of work or settled a choice.

![Two stacked scatter plots of the public NDCG of the 40 scored submissions in the order their scores were logged, 11 to 19 November 2026; the lower plot is a detail of 12 to 19 November. The team-best step line climbs from 0.3739 to 0.39526061, the final picks FD16P20@m0.4_0.2 and FDX16P20:ps are ringed, and horizontal lines mark places 1 and 2, 0.398256 and 0.398058, on the board of 19 November, where the team ranks 9 of 541.](figures/public-progress.svg)

## Phases at a glance

| Phase | When (UTC) | What changed | Rows scored | Team best after it (public) |
|---|---|---|---|---|
| 1 | 11 Nov, 02:19 to about 07:30 | First files from neural rankers and tree models; a leak audit | 1 to 4 | 0.3802 |
| 2 | 11 Nov, 07:30 to 23:20 | Ranker trained for 10 epochs with a new optimiser, retrieval and new inputs; first pools | 5, 6 | 0.39208 |
| 3 | 12 Nov | Pools with re-run cooldowns; a new leader; the limit of choosing by validation | 7 to 13 | 0.392820 |
| 4 | 13 Nov | A sweep closes almost every idea; popularity-light members help | 14 to 18 | 0.393454 |
| 5 | 14 Nov | Distillation from a pool of click-label members | 19, 20 | 0.394486 |
| 6 | 15 Nov | Holdout staleness explained; W8P20; retrain on all train feeds | 21 to 25 | 0.3950094 |
| 7 | 15 Nov, 21:00, to 16 Nov, 15:31 | Larger pools and a trained input, none better | 26 to 30 | 0.3950094 |
| 8 | 17 Nov, 01:10, to 19 Nov, 03:32 | Losses aimed at the misses, try files, the final picks | 31 to 40 | 0.39526061 |
| 9 | 19 Nov, from 04:00 | Mixture of experts and depth for students, null at 8 seeds; checkpoint choice and shared seed luck behind the screen reads | none | 0.39526061 |

## Reading notes

- All times are UTC. "(row N)" points to the [table of submissions](#all-40-scored-submissions) at the end.
- A feed is one user's list of candidate items. The score is NDCG (normalised discounted cumulative gain) over the full list, averaged over feeds ([01-competition-and-data.md](01-competition-and-data.md)).
- Validation is the 14,423 labelled feeds of 22 and 24 September. The holdout is 30,000 train feeds kept out of training. Public is the 8,562 of the 28,595 test feeds that the board scores; the other 20,033 are private.
- A pair a / b is validation / holdout. "±" is one standard error of a difference paired per feed, and z is the difference divided by it ([04-validation.md](04-validation.md)).
- A member is one trained ranker; a pool is the equal-weight mean of its members' within-feed z-scores. A member gives a train-only branch and a branch fine-tuned on the validation days, and the final files mix the two 0.5 / 0.5.
- Click-label members learn from clicks alone. Students learn from clicks and a teacher, the averaged scores of earlier click-label members. Popularity-light members read one clipped exposure count in place of the 8 popularity inputs; full-view members read all inputs.
- A random-order feed is a feed that was displayed in random order: every validation and test feed, and about 13% of the train feeds with 50 or more candidates.
- In file names, FD marks members trained on all train feeds, the number counts full-view students, P20 means a popularity-light part at weight 0.2, and @m0.4_0.2 means weight 0.4 on feeds of fewer than 50 candidates and 0.2 on the rest. W8 is 8 full-view students of width 192 that never trained on the holdout.
- An X after FD marks a file whose full-view students were trained another way, named after the colon: `ps` for students that also train on pseudo-feeds built from user histories, `hz` for a cooldown on stale popularity inputs, `ref` for a plain cooldown re-run and `fsw50t` for a per-feed loss weight. H means that half of the 16 were replaced, and U32 is the union of the old and the new 16.
- A near copy is a file with a within-feed correlation of at least 0.995 to another and a validation difference of at most 0.002. Neither validation nor the board can order near copies.
- Clock times, rules and quoted log lines come from the status log and the decision ledger, two working records that are not published. Each phase links the experiment folder that holds what is published.
- Other terms are in [glossary.md](glossary.md). The closed experiments named here are told in [07-what-failed.md](07-what-failed.md).

## Phase 1: trees, a first neural ranker and a leak that did not pay (11 November, 02:19 to about 07:30)

The plan written that morning from the data analysis had two layers. A neural ranker with attention from each candidate to the user's history would be the main model, and a gradient-boosted tree model would be stacked on its scores and on a bank of features. A boost for candidates found in the same user's later history snapshot, projected at +0.0041 on public, would be tested as a separate file ([the leak](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)).

The morning produced the features, tree models at 0.368 to 0.370 on validation, a first neural ensemble at 0.3806 without any validation label (0.3819 after a fine-tune on validation), and tree stackers that added 0.0006 to 0.0010 (± 0.0005) on top of it.

The board gave 0.3739, 0.3772 and 0.3802 for the first three files (rows 1 to 3). The forecast, validation re-weighted by feed size alone, had been within 0.0003 to 0.0014 of each. The fourth file was the third plus the later-history boost. It scored 0.3802, the same as without it (row 4).

An audit of every feature on the 117 validation feeds whose user has a later feed then found the same channel inside the team's own inputs. The contaminated features went to a quarantine folder, and the counts were rebuilt without the same user's later snapshots. About 0.002 of the third file's validation score had been leak (note (a) to the table; [04-validation.md](04-validation.md#leak-incidents)).

Two decisions ended the phase:

- Nothing may use a user's later snapshots, whether as a feature, a label or post-processing.
- The stacker line stopped at about 07:30. The next neural ensemble (0.3879 on validation after the fine-tune) left the tree stacker with -0.0001 ± 0.0006 to add.

Corrected later: the leak came back on 12 November as an explanation of a new leader and was tested a second time (phase 3). A note of that morning said that validation gains transfer about 1:1 to public; that held for large steps only (phase 3).

## Phase 2: making the ranker itself strong (11 November, 07:30 to 23:20)

With the stacker dropped, the rest of the gap had to come from the ranker. Places 2 to 10 of the board sat at 0.3884 to 0.3900 and the leader at 0.3935, read as 0.008 to 0.010 of validation still to find. Training length and the optimiser looked like the source. On one seed the holdout was still rising at 5 epochs (0.3117 at 2 epochs, 0.3128 at 3, 0.3148 at 5).

The changes, in the order they were built:

- Co-visitation and popularity counts as inputs of the train stage: +0.0032 / +0.0092 (one paired run).
- Four members trained for 10 epochs. As an equal mix of their fine-tuned and train-only versions they read 0.39401 on validation. The file sent (row 5) added a small share of encoder scores and read 0.39399.
- The Muon optimiser with a warmup-stable-decay schedule: +0.0031 / +0.0033 (3 seeds at 2 epochs).
- Retrieval of the 16 history items most similar to each candidate: +0.0028 / +0.0038 (3 seeds).
- The scores of the trajectory encoder, a self-supervised model of user histories, as inputs.

In the afternoon a review covered the metric code, the leak paths, the selection procedures and the model maths. The same afternoon produced the pre-submission gate that every later file had to pass ([04-validation.md](04-validation.md#the-pre-submission-gate)).

Three of the day's steps failed and were repaired or dropped.

First, the encoder scores for train rows raised the holdout and lowered validation, and the opposite signs were a leak. The encoder's item table held no train candidates, so on train rows a missing value marked likely positives ([failure 2](07-what-failed.md#2-encoder-scores-for-train-rows-first-build)). The scores were quarantined and rebuilt with every candidate in the table. A feature review then found most rebuilt values at a clipping bound, and only the rescaled version was adopted.

Second, all six members first trained with the new optimiser collapsed in the cooldown, the last 20% of training, from 0.391 to 0.364 on validation. In the cooldown 30% of the batches came from a pool of 134,118 random-order train feeds. That share had been tuned at 2 epochs, and at 10 epochs it meant about 9 passes over the pool ([failure 3](07-what-failed.md#3-a-cooldown-dose-tuned-at-2-epochs-and-used-at-10)).

The checkpoints from before the collapse were kept and the cooldown was re-run at 10%, which gave each member +0.0018 ± 0.0005 / +0.0015 ± 0.0003.

Third, a note at 21:20 said that with the new optimiser 2 epochs were as good as 10 (0.39375 against 0.39389 on validation, one seed), and 12 two-epoch members were started. At 22:50 the pool with them read 0.32208 on the holdout against 0.32319 without. Two-epoch seeds correlate 0.978 with each other and ten-epoch seeds 0.93 to 0.94, so six short seeds add 0.0010 to a pool where six long ones add 0.0040. Training of two-epoch members stopped.

The board gave 0.38786 at 14:10 for the four 10-epoch members (row 5; forecast 0.3860) and 0.39208 at 23:20 for a 19-member pool of three recipes (row 6; forecast 0.3909, validation 0.39901).

The rule that closed the day was written at 22:25. A pool's composition is chosen on the holdout among candidates named in advance, and a pool replaces the first pick only if it is more than 2 standard errors better on validation.

The holdout chose the three-recipe pool (+0.00157 ± 0.00034 over the single-recipe file), but its validation gain of +0.00083 ± 0.00054 did not clear the bar. The single-recipe file (validation 0.39818, never sent) stayed first pick. The pool became second pick and went to the board as a sanity check.

On that pool the gate's leak probe fired at z 2.0. Its failure threshold was raised to z above 3 after it had fired ([04-validation.md](04-validation.md#leak-incidents) has the controlled pair behind that change).

Corrected later: after the 14:10 score a note said the gap between validation and public was shrinking, and the review of the same afternoon put the change at +0.0007 ± 0.0018. The forecasts of this phase come from validation re-weighted by feed size alone. That forecast read 0.0003 to 0.0019 below the board on all five files of the day with a validation score, and from 13 November the fitted line of phase 3 replaced it.

## Phase 3: pools, a new leader and the limit of choosing by validation (12 November)

The working belief on 12 November was that the pool with the higher validation score would win on public. At 11:31 the rule of the night before made a 38-member pool of six recipes the first pick, on a validation gain of +0.00156 (z 2.79).

The new member type of the morning was a cooldown re-run: the last 20% of training repeated from a stored checkpoint, with 2% of its batches drawn from the random-order pool. Against the same re-run at 10% it passed on 7 members at +0.00117 ± 0.00039 / +0.00134 ± 0.00022.

Three pools built with these members went to the board:

- G3 (row 7), 20 members: 6 with a GRU (gated recurrent unit) history encoder and 14 re-run members. It is the best click-label pool.
- The 38-member pool with the re-run members in place of their originals (row 8).
- The 14 re-run members alone (row 9), the best validation score of the day at 0.40094.

In the same hours several lines closed with nothing promoted. They were language-model backbones of 0.6, 4 and 9 billion parameters (-0.0038, -0.0048 and -0.0053 on validation against the ranker trained on the same 150,000 feeds; [experiments/llm-rankers/](../experiments/llm-rankers/)), tree models in every role, and 13 block designs and hyper-parameter settings.

At 14:04 the board gave 0.392820 for G3, 0.392723 for the 38-member pool and 0.392434 for the pool with the best validation score. Against the pool of row 6, G3 had gained 0.00131 on validation and 0.00074 on public. The pool of re-run members alone had gained 0.00193 and 0.00035.

A diagnosis finished at 15:22 with three findings:

- There was no pipeline bug. A replay of 40 members reproduced their predictions.
- The public noise of such pairs is a paired standard error of 0.00047 to 0.00069, in place of the 0.0012 used before. The shortfall itself was therefore not established (joint test of 1:1 transfer, p 0.19).
- A plain cooldown re-run gains 0.0009 on validation while losing 0.0005 on the holdout, so pools made of re-run members carry a part that only validation sees.

G3 became first pick through the tie-break of a rule that was written after the scores, and the records flag it as such. At 18:53 a stacker that re-weights the pool's members per feed scored 0.392473 (row 10). Against G3 it had gained 0.00056 on validation and lost 0.00035 on public.

At 20:20 the board showed a new leader at 0.398256, with a single entry 26 minutes old and 0.0054 above the team. Second place was at 0.394695, and the team was fourth. The jump was first taken for the later-snapshot leak of phase 1. A kill rule was written before submitting, and at 21:52 an overlay of the leaked items on G3 scored 0.392819 against 0.392820 for its base (row 11), after +0.00454 ± 0.00047 on validation.

The overlay was dropped and later snapshots were banned from every input for good. The data page tells the story in full ([the later-snapshot leak](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)). From then on the leader was read as a better model, about 0.0056 above the team on the validation scale.

The same batch scored the train-only branch of G3 at 0.392346 (row 12), 0.00047 below the full file. It was the first public measurement of the fine-tuned branch. At 23:46 the 38-member pool without re-run members scored 0.392436 (row 13).

The phase ended at 00:39 on 13 November with a study of how validation and public relate over the files scored so far ([04-validation.md](04-validation.md#noise)). Levels transfer: public = 0.0071 + 0.964 × validation over 11 files, refitted on 15 November as 0.0074 + 0.963 × validation. Differences between near copies do not. The two files chosen for their validation score fell 0.0010 and 0.0009 short of their forecast.

Three rules followed:

- G3 stays first pick unless a candidate beats it by 2 standard errors on the holdout and is not below it on validation.
- Public checks the level of a file and never the order of near copies.
- No more submissions or GPU time go to near-copy questions. Only a new model generation (+0.002 or more on validation, at a within-feed correlation below 0.99 to the pool) could still move the private score.

Corrected later: a note of 11:31 that the holdout understates the re-run members was withdrawn by the 15:22 diagnosis. The trust this phase placed in the holdout was revised on 15 November (phase 6).

## Phase 4: closing everything (13 November)

By 13 November the recipe looked saturated, and the holdout looked like the witness to trust for whatever came next. The day was a sweep, and almost everything in it closed:

- A screen of 26 two-epoch runs removed each component of the recipe in turn and found nothing to remove. Validation fell by 0.0034 without retrieval, by 0.0023 without the popularity inputs and by 0.0022 without the auxiliary loss toward the production order.
- All four scale-ups in the same screen failed. Training for 15, 20 or 30 epochs cost 0.0006, 0.0008 and 0.0016 on validation against 10, and a four times larger batch cost 0.0026.
- A faithful transfer of the winning recipe of the 2025 Avito ML Cup added 0.00067 ± 0.00045 to a tree model, less than the 0.00086 by which a second seed of the same model differed.
- A sweep of 63 post-processing, label and training tricks adopted none.
- A scan of 211 techniques from other competitions' solutions found none with a prior above 0.035.
- Eight new member families, four built to see the data differently and four taken from outside techniques, were near copies of the pool or 0.001 to 0.003 too weak. None advanced.

The one gain of the day came from members with reduced popularity inputs ([experiments/popularity-views/](../experiments/popularity-views/)). Each file below adds members to G3:

- Four members trained without any popularity input scored 0.392886 (row 14), +0.000066.
- Four popularity-light members scored 0.393205 (row 15), +0.000385, which the estimates of that day put at 0.6 to 0.8 of a paired public standard error.
- The pool with eight popularity-light members read -0.00073 on the holdout (z -2.5), which under the rule written for it vetoed it as a pick. Sent as a probe, it scored 0.393394 (row 17).
- With every fine-tune averaged over 3 seeds, the same pool scored 0.393454 (row 16, the best click-label file, C3), 0.00063 above G3, the first pick.
- Seed averaging alone did nothing (row 18; -0.00007).

The three scored pools with popularity-light members averaged 0.393351 on public (standard deviation 0.00011), and the three without averaged 0.392820 (0.00005).

At 19:43 the team was fourth of 453 teams with 0.393206, and places 3 to 7 lay within 0.0010 of each other.

At 20:43 the first freeze protocol named G3 as first pick, the best popularity-light pool as its alternate and the 38-member pool without re-run members as second pick. The log entry of that hour reads "no experiment with a stated mechanism and a prior above the noise floor remains", and research was declared closed.

Corrected later: the closure lasted until 03:14 the next morning. The holdout veto on the popularity-light pools was the holdout's error, explained on 15 November (phase 6).

## Phase 5: distillation re-opens the work (14 November)

Four tests committed to before the closure ran through the night, at a recorded expected gain of 0.00005 on private. They were ranker width with clicks as the only target, a larger dose of the trajectory encoder, self-supervised training as a way to make width usable, and pseudo-feeds built from histories. All four were closed by 04:09.

The width test left a usable finding. At the checkpoints of step 77,500, before the cooldown, width 192 against width 128 (5 runs against 12) lowered the in-sample loss (z -5.5), raised the held-out loss (+0.0019, z +1.5) and left held-out NDCG where it was (+0.0002, z 0.6). After cooldown re-runs it read -0.00087 / -0.00106 against width 128.

The wider model memorised, so the ranker was limited by its data, with one click among about 100 candidates per feed. A pool, meanwhile, scored about 0.004 above any of its members. The working belief became that the click label was the noisy part, and that one model could be trained toward what the pool had averaged out.

Distillation was on the closed list. On day 1 a single two-epoch teacher had given +0.00064 on validation (z 1.6), and it had been filed as no effect.

At 03:14 the rule was written before the first run, in the ledger of time-stamped rules kept from that night on (the decision ledger; [04-validation.md](04-validation.md#rules-written-before-the-read) gives the rule and its read). The teacher was the mean score of 11 click-label members on all 246.7 million train candidates. A student's target was half the clicks and half the teacher's distribution over the feed.

The read at 05:18, on four seeds, was +0.00257 ± 0.00047 / +0.00200 ± 0.00028 per student and +0.00198 ± 0.00061 / +0.00021 for the pool. Distillation was promoted.

A second round crossed student width with teacher size. By its written rule nothing was promoted, because the pool of four wide students read +0.00049 against a bar of +0.0005. All eight wide students were then pooled, a step recorded as post hoc before any fine-tune number existed. That pool read +0.00107 ± 0.00031 on the holdout over the pool of four narrow students ([experiments/distillation/](../experiments/distillation/) has the runs and the teacher files).

The board answered at 17:32: 0.394486 for the 8 full-view students (W8, row 19), 0.001032 above the best click-label file, and 0.394078 for the same eight plus eight narrow ones (row 20).

Nothing varied on top of W8 that day did better: teacher weights of 0.3, 0.8 and 1.0, a 25-member teacher, students as the next teacher, a cross-fitted teacher, widths 224 and 256, a third block, an item-ID table (-0.0057 / -0.0049), 20 epochs, and students with a GRU history encoder.

A jump of 0.001 this late deserved suspicion. That evening nine separate reviews of the result were each read by three reviews written to argue against them ([lab/reports/RECHECK_1114/SYNTHESIS.md](../lab/reports/RECHECK_1114/SYNTHESIS.md)). The nine included a code and leak audit of the distillation path, board statistics, pool composition and the fine-tune of students.

They found no bug and no leak, and three independent recomputations reproduced every headline number. They also showed that W8 was a favourable draw of its recipe: a 12-student pool read -0.00024 / -0.00005 against it.

The decision of 21:24 made W8 first pick and fixed three files with reading rules written before any of them had a score:

- 16 full-view students (W16), expected to tie;
- W8 blended 80/20 with 4 narrow popularity-light students (W8P20), which read +0.00049 ± 0.00031 on validation and -0.00063 ± 0.00022 on the holdout;
- the train-only branch of W8, for information.

## Phase 6: staleness explained and a retrain on all data (15 November)

The day began with the holdout still treated as the stricter witness, and it had voted against the blend. At 02:15 the three files came back:

- W8P20 scored 0.3950094 (row 22), 0.00052 above W8 at a paired standard error of 0.00041. It stayed the team best for 69 hours, until 23:46 on 17 November (phase 8).
- W16 scored 0.39448337 (row 21), 0.000003 below W8.
- The train-only branch of W8 scored 0.394009 (row 23), 0.00048 below the full file.

Under the rule written the evening before, +0.00052 was short of the +0.0008 needed to change pick 1, which left the blend a candidate for pick 2 at most. Combined with the validation reading by inverse variance, it gave +0.00050 ± 0.00025.

The scores contradicted the holdout. It had read the blend at -0.00063 and the popularity-light pool of 13 November at -0.00073, where public gave +0.00052 and +0.00063. By that morning the cause was known: popularity inputs 1.75 days old on the holdout against 9.34 on public ([04-validation.md](04-validation.md#the-witnesses) tells it in full).

Two decisions followed that morning, both on requests from the person on the team:

- The holdout lost its vote on changes to how the model uses popularity, after the person asked that no idea be rejected on the holdout alone. An audit of the idea records re-opened three ideas ([04-validation.md](04-validation.md#rules-written-before-the-read)). One of them, a cooldown on train rows made to look as stale as test rows, passed at pool level, failed as a file and was never sent ([failure 19](07-what-failed.md#19-a-cooldown-on-stale-popularity-inputs)).
- Every file sent from then on is trained on all train feeds, the 30,000 holdout feeds included.

Twenty members were retrained on all data, 16 full-view students and 4 popularity-light students, all of width 192. The runs and their checks are in [experiments/full-data-retrains/](../experiments/full-data-retrains/). Such a member has no out-of-sample holdout score, so each was accepted against its held-out twin. The pool of the 16 all-data full-view students read +0.00004 ± 0.00042 on validation against the pool of their twins.

At 14:34 the all-data file, FD16P20, scored 0.394892 (row 24), 0.00012 below the team best at a paired standard error of 0.00056. The same 20 members with the popularity-light weight raised to 0.4 on feeds of 50 or more candidates scored 0.3947465 (row 25), 0.00015 lower. On validation that rule had read +0.00026 ± 0.00021 in sample for this pair and +0.00002 ± 0.00027 when nested by user. It was not adopted.

A plan written at 04:52, at the person's request, had set 12 hours to beat the best public score, and the ledger entry of 14:34 records that it had not. The day produced an all-data file level with the best one.

The day also closed the stale-view cooldown as a file, popularity-input dropout, a fine-tune with stale inputs, a teacher weight of 0.8 (each student +0.00126 on validation, the pool of eight -0.00049), a second round of CatBoost, finer mixing rules, the feed-size rule, and narrow or additional popularity-light students.

At 20:30 FD16P20 (row 24) became first pick, and the freeze protocol was rewritten around the final pair. The second pick had two options. The default was an all-data file, under the rule of that morning, and the alternative was W8P20 (row 22), whose members hold out the 30,000 feeds. The choice was left to the team's freeze on 2026-12-07 ([11-final-picks.md](11-final-picks.md)).

The same entry corrected the deadline in the records. The "12:00" of the rules page is Moscow time, so uploads close on 2026-12-08 at 20:59 UTC.

Corrected later:

- FD16P20vf (row 26), a file of 16:46 with four popularity-free students added to the popularity-light part, read above 70 of 70 control files on validation, and one table recorded the view as adopted. The replication read at 20:49 was above 16 of 70, and the view was not adopted ([the rule that failed](04-validation.md#rules-written-before-the-read)). The file stayed the all-data option for the second pick.
- The validation score of row 24, 0.40271, turned out on 16 November to be a favourable draw (phase 7).
- On 18 November the same 20 members with a higher popularity-light weight on short feeds replaced FD16P20 as first pick, and W8P20 became the second. On 19 November FDX16P20:ps replaced W8P20 (phase 8).

## Phase 7: probing the ceiling (15 November, 21:00, to 16 November, 15:31)

At 21:07 the person asked for experimental files with the best chance of progress and named a much larger ensemble as one option. The open question was whether a step remained in a much larger ensemble or in a new input. A scan of 82 outside sources and three audits of the team's own records, read at 23:08, estimated every untested change to the model at below +0.0001 at file level.

The remaining submissions were spent as measurements, each with a band written in advance inside which nothing would change. At 01:31 three files came back, all inside their bands:

- FD16P20vf scored 0.394630 (row 26), one paired standard error below the first pick.
- A union of 68 members (ENSA68) scored 0.3946427 (row 27).
- The fine-tuned branch of W8 alone scored 0.393960 (row 28). A parabola through it and rows 23 and 19 (fine-tune weights 1, 0 and 0.5) peaks at 0.49, and the weight stayed at 0.5 ([fine-tune](03-solution.md#fine-tune-on-the-validation-days)).

Growing the pool did not help. All-data pools of 32 and 48 full-view students read 0.40237 and 0.40222 on validation against 0.40271 for the 16, and failed the rule for larger pools. The first 16 had been a favourable draw of a family whose level is about 0.4024 ([failure 18](07-what-failed.md#18-growing-the-ensemble-past-16-members)).

The 32-student file was never sent. On 17 November the 48-student file scored 0.39484072 (row 35) and a 24-student file built on the way 0.394905 (row 34), -0.00005 and +0.000013 against FD16P20.

The last new idea was a trained input, a two-step co-visitation walk over whole histories given to the ranker as a real input ([experiments/trained-input-search/](../experiments/trained-input-search/)). About 300 feature ideas had been closed before, but only as additive corrections to a finished pool. Eight seeds under click labels read a pooled z of +0.59 and eight under distillation +0.62, against a bar of 2.0.

The members did use the input: shuffling its values cost them 0.00179 ± 0.00034 on validation. It replaced information the ranker already extracted. At 11:55 sixteen all-data full-view students with the input scored 0.3943559 (row 29) and a 32-student mix 0.394578 (row 30), both inside the band declared as carrying no information.

The remaining reads closed weight decay at 0 and at 0.01, a heavier teacher in the cooldown (each member +0.0006, pooled z -2.88) and 5% random-order feeds in the cooldown. At 15:31 a click-strength input repaired for exposure failed its gate ([failure 15](07-what-failed.md#15-the-rebuilt-click-strength-input)), and the ledger recorded that no open model-side item remained.

By then seven files of the final design, full-view students with popularity-light students at weight 0.2, had public scores from 0.39436 to 0.39501. That is closer together than the board can separate ([11-final-picks.md](11-final-picks.md#risks) has the files and the figure).

FD16P20 had been chosen as the first final file by design, for its all-data training and member count. The second was open between W8P20 (row 22) and FD16P20vf (row 26). What remained on 16 November were decisions outside the model: whether to merge with another team, which file to take as second pick, and how to use the remaining submissions.

Corrected later:

- At about 14:20 on 16 November a list named the training batch size, a fix to the normalisation of the auxiliary loss and the fine-tune settings as never run. All three had been run and closed on 12 and 13 November. The list came from rows of the coverage table that had not been updated.
- An idea to offer the file of row 29 as a second pick with upside was withdrawn at 11:55, when its score came in as the lowest of the family.
- The model side was opened again at 01:10 on 17 November. Pick 1 was chosen on 18 November and the present pick 2 on 19 November (phase 8).

## Phase 8: aiming at the misses, try files and the final picks (17 November, 01:10, to 19 November, 03:32)

On 17 November the model looked finished, and the error left in FD16P20 seemed to have no handle not yet tried. At the person's request the first work went into the errors of the final file. Two checks came first:

- Language-model reviewers asked to correct its order feed by feed found no edit that helped. The one rule they allowed read -0.00011 ± 0.00007 on 13,701 validation feeds.
- An error analysis on one half of the validation users proposed one correction rule, which read -0.00038 ± 0.00028 on the sealed other half.

The analysis also drew a failure map ([experiments/error-analysis/](../experiments/error-analysis/)). Feeds of 101 to 300 candidates were 37% of the feeds and 43% of the loss. Feeds whose best positive lay outside the category of the top-ranked item were half the feeds and 59% of the loss.

A round written at 03:17 and read at 09:32 aimed the loss at that map. Each arm was a 2-epoch cooldown re-run of the 16 held-out full-view students of W16 (row 21). It was read against reference re-runs from the same checkpoints by a randomization test with a Holm correction over 8 arms:

- more weight on the order inside categories (two doses);
- more weight on the choice of category;
- a per-feed weight that grows with feed size (two doses);
- three cooldown settings prepared on 14 November and never run (teacher weight 0.3, learning rate 4e-3, 4 epochs).

A ninth arm gave feeds at the 256-event history cap up to 128 older events from the same user's earlier snapshot. None of the nine passed. The smallest p was 0.16, for the history arm on the feeds it reaches. No arm moved the part of the error it aimed at beyond the spread of two reference runs.

The best arm on validation, a per-feed weight proportional to the square root of the feed size, read +0.00036 ± 0.00034 on validation and -0.00030 ± 0.00019 on the holdout. After the person asked for files to try, it was rebuilt as a probe on the 16 all-data full-view students of FD16P20, with a matched control.

On validation the two files read -0.00002 ± 0.00027 apart. On public the control scored 0.39482063 and the arm 0.39475033 (rows 31 and 32). The re-run itself cost the control 0.00056 ± 0.00028 on validation and 0.00007 on public against FD16P20.

In the evening, on another request from the person, the loss itself was aimed at the hardest feeds. A composite loss (a per-feed weight from the student's gap to the teacher, times a lower weight on clicks the model finds surprising) read -0.00058 ± 0.00047 / -0.00074 ± 0.00028. A cascade correction read +0.00004 / -0.00000.

A probe run before that round had shown why. The hardest 20% of feeds carry 27.5% of the loss, and 27.6% when the labels are redrawn from the model's own softmax. Most of what the biggest misses lose is the draw of the label.

Two more ideas closed that evening. Like the two losses above, neither was adopted ([07-what-failed.md](07-what-failed.md)):

- History vectors filled at prediction time from earlier splits' item tables read +0.00021 ± 0.00024 on the train-only branch and -0.00012 on the file with the stored fine-tuned branch.
- Pseudo-feeds built from user histories, closed in a first form on 14 November, came back inside the distilled students. They read +0.00092 ± 0.00060 per member on a 2-seed screen, then +0.00054 ± 0.00047 / +0.00010 ± 0.00027 for the 8-seed pool at a randomization p of 0.19, and were closed at 22:04. Try files built from them followed (below).

From 22:50 the remaining submissions went to try files, after the person asked to send ideas even when the written rules held them back. A try file is an idea with a positive mean on validation, built as an all-data file and sent even when it had missed the rule written for it or had none.

The first try file, built at 22:54, was the side of the feed-size rule of 15 November that public had not read. It gives the popularity-light students weight 0.4 on feeds of fewer than 50 candidates and 0.2 on the rest, on the 20 members of FD16P20. It read +0.00022 ± 0.00024 on validation, +0.00027 on the public-mix view and +0.00033 on the private-mix view (validation re-weighted to match the public or the private part of test).

At 23:46 it scored 0.3952092 (row 33). That is 0.00032 above FD16P20, about one paired standard error, and 0.00020 above W8P20. The same batch scored the 24- and 48-student files.

Read after the score, the validation curve of the short-feed weight has a single high point at 0.4. With 8 or 12 popularity-light students under the same rule the file read 0.00057 and 0.00039 below it, so nothing more was built in that direction ([06-what-worked.md](06-what-worked.md#17-popularity-light-weight-04-on-short-feeds)).

Two try files came back at 02:30 on 18 November:

- FDXH16P20:ps is FD16P20 with its 8 students of the 11-member teacher replaced by all-data students trained with pseudo-feeds. It read +0.00028 ± 0.00029 on validation and scored 0.3947873 (row 36), 0.00010 below FD16P20.
- A1 is FD16P20@m0.4_0.2 with the popularity-light part of its fine-tuned branch taken from the popularity-light students' train-only predictions on feeds of 50 or more candidates. It read +0.00017 ± 0.00014 on validation and scored 0.395036 (row 37), 0.00017 below the best. It was the second change to the popularity-light part on long feeds to land about 0.0004 below its validation reading, after row 25.

A stale-view all-data file was also built on the night of 17 November (validation 0.40224, 0.00047 below FD16P20). It was sent a day later (row 40). An error analysis of the new best file found nothing left that a correction could fix. Its score is a calibrated log-odds, and corrections estimated on half of the users cost more in placebo runs (0.0004 to 0.0009) than a perfect correction of every detected bias would gain (+0.00035).

At 02:33 the person on the team chose a first final pair from the options the lead session had laid out:

- Pick 1 is FD16P20@m0.4_0.2 (row 33), trained on all train feeds, with the same members as FD16P20.
- Pick 2 was W8P20 (row 22), whose members leave out the 30,000 holdout feeds. The all-data rule was waived for the second file.

The reasoning on record treats FD16P20@m0.4_0.2 and FD16P20 as tied within noise for private. Both leads of the new file over FD16P20, +0.00022 on validation and +0.00032 on public, are the best of several tries, and the expected private difference is about +0.00005.

W8P20 was the strongest file built from other runs. If the better of the two finals counts, it adds about +0.00015 at equal level. It also covers the chance that the all-data files fall short of their validation level on private, as the files scored up to 16 November did on public.

The choice set aside the rule for replacing pick 1, which asked a near copy to pass a rule written before its run or to score at least 0.395992 on public. FD16P20 stays on record as the earlier pick 1.

At 03:14 two more pseudo-feed try files were built. FDX16P20:ps has all 16 full-view students of FD16P20 replaced (validation 0.40286, +0.00015 ± 0.00036 against FD16P20). FDXU32P20:ps adds the 16 pseudo-feed students to those of FD16P20 (validation 0.40301). After the score of row 36 both were expected at about the level of FD16P20.

At 05:18 three files with width-224 students followed: 8 or all 16 full-view students replaced, or the 16 added, at validation 0.40276, 0.40276 and 0.40287. That is level with FD16P20 (+0.00005 to +0.00017, standard error 0.0003 to 0.0004). None of the width-224 files had been sent by 19 November.

Three files sent after 21:00 UTC on 18 November, when a new submission day began, had their scores logged at 03:31 and 03:32 on 19 November:

- FDX16P20:ps scored 0.39526061 (row 38), the team's best public score. It is 0.00037 above FD16P20 at a paired standard error of 0.00047, and 0.00005 above pick 1, a tie.
- FDXU32P20:ps scored 0.39524613 (row 39), 0.00035 above FD16P20.
- The stale-view file, FDX16P20:hz, scored 0.3949761 (row 40), 0.00008 above FD16P20 at a paired standard error of 0.00045, after -0.00047 on validation.

The two replacement files disagree. Replacing the 8 students of the 11-member teacher scored 0.00010 below FD16P20 (row 36), and replacing all 16 scored 0.00037 above it. The gap of about 0.0005 between them is inside the paired noise. The stale-view file read about 0.0005 better on public than on validation, a weak sign, also inside the noise, that public inputs behave stale.

At 03:32 the person on the team changed pick 2 ([11-final-picks.md](11-final-picks.md)):

- Pick 1 stays FD16P20@m0.4_0.2 (row 33).
- Pick 2 is FDX16P20:ps (row 38): 16 full-view students trained on all train feeds plus pseudo-feeds, and the 4 popularity-light students of FD16P20 at weight 0.2. Both picks now meet the all-data rule.

The reasoning on record gives W8P20 and FDX16P20:ps about equal value if the better of the two finals counts, about +0.0001 each. FDX16P20:ps has the higher expected level, +0.0007 on validation and +0.00025 on public over W8P20, and its 16 full-view students come from runs separate from those of pick 1. W8P20 stays on record as the earlier pick 2.

A ledger rule written at 22:50 on 17 November, before any pseudo-feed file existed, said that a probe within 2 paired standard errors of FD16P20 on public changes no pick. FDX16P20:ps is 0.8 paired standard errors above FD16P20, and the change of pick 2 set that rule aside.

Only the lab trainer `lab/nn_x/w12/pseudokd/train_full_pk.py` trains the students of pick 2; the package trainer has no pseudo-feed option. `configs/ensembles/fdx16p20_ps.toml` rebuilds the file byte for byte from the stored member predictions.

The picks can still change until the team's freeze on 2026-12-07. The platform's selection window runs from 2026-12-08 21:00 to 2026-12-09 09:00 UTC, and a merge with another team is possible until 2026-12-01.

## Phase 9: mixture of experts and depth (19 November, from 04:00)

An earlier mixture-of-experts test (11 November, +0.00027 ± 0.00047 on validation) had collapsed: the router of the last block sent almost every token to one expert. The re-test used the final student recipe and current designs. In DeepSeek-V4 and Kimi K3, many small routed experts sit beside a shared expert that every token passes through, and a per-expert bias balances the load instead of an extra loss term. SwitchHead routes the value and output projections of attention.

Three arms were fixed in a rule before any run: routed feed-forward experts (A), routed value and output projections (B), and depth 7 at about equal size (C). Each ran with 4 seeds against the stored students, 12 runs in two waves. The early-stop rules were set to report only, and every run trained to its end.

All three failed the rule. On validation the 4-pools read -0.00086 (A), -0.00144 (B) and -0.00069 (C), at a standard error of about 0.00057. On the holdout all three read slightly positive, +0.00020 to +0.00040.

Router entropy stayed at 0.78 or higher, so no expert took the whole load as on 11 November, although in the last block 2 of 8 experts of A went unused on 3 of 4 seeds. Mixed into the 16-student pool, the new members changed nothing (-0.00004 to -0.00042). No try file was built.

An analysis of why, with each finding checked by a second reviewer, found two things:

- Checkpoint selection on the holdout picked steps 85,000 to 90,000 for 4 of the 12 arm runs. Students trained on all data keep the last step. At the last checkpoint, paired by seed, the arms read +0.00009 (A), -0.00093 (B) and -0.00016 (C) on validation.
- The 4 reference students of the screen form the 1st to 3rd best of the 70 possible 4-seed pools of the 8 stored references, about 0.0006 above their mean on validation.

The analysis took the second finding for a bias against the arms. Re-read against the mean reference pool, A rose to +0.00070 ± 0.00046 and C to +0.00045 ± 0.00047, and the same re-read lifted width 224 from +0.00026 to +0.00092. Across about 37 arms of student screens with stored predictions, such top values are also what noise alone gives.

At 15:15 seeds 955 to 958 of A and C started. A rule written first asked the 8-pool at the last checkpoint for +0.0003 on validation over the 8 references before an all-data try file would be built.

Both arms closed at 16:40. Against the 8 references, A read +0.00006 ± 0.00047 on validation and C -0.00001 ± 0.00048; on the holdout -0.00004 and +0.00027. Both arms pooled across seeds as well as the reference did.

The fresh seeds explained the overshoot. On validation at the last checkpoint, seeds 951 to 954 read above seeds 955 to 958 by +0.00113 for the reference and by +0.00082 and +0.00085 for A and C. Arms that share the teacher and the seeds share the luck, so the seed-paired screen had cancelled it, and the re-read against the mean counted it twice.

A recipe with the 25-member teacher did not share the luck (-0.00051 on the same comparison), so arms with another teacher may read up to 0.0006 too low when paired by seed. Of the two findings, only the checkpoint choice was a flaw of the screen. Paired by seed at the last checkpoint, width 224 and the pseudo-feed students read +0.00031 and +0.00042 on validation, not the +0.00092 and +0.00104 of the re-read.

Ideas that never had a fair training run are named, with a judgement of 19 November on each, in [section 6 of 12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed).

## All 40 scored submissions

The files were written to `artifacts/submissions/` with the extension `.parquet`; that folder is not versioned, so they are not in this repository. "Name" is the handle the working records and this page use for a file; [glossary.md](glossary.md) lists the others. Validation is the file's own NDCG on the 14,423 validation feeds, out-of-fold for the fine-tuned branch. Scores keep the digits the records hold.

"Best" marks a file that scored strictly above every earlier file of the team; when several scores were reported together, only the top one is marked. Dates are the dates the score was logged. The send times of rows 1 to 4 were not recorded.

| # | Date | Name | File | What it was | Validation | Public | Best |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-11 | v3 | `sub_v3_nnens_stk_gb` | First file: a blend of the first neural rankers, a tree-model stacker on top of them, and gradient-boosted tree models | 0.3816 | 0.3739 | yes |
| 2 | 2026-11-11 | v4 | `sub_v4_resstk_nnft` | A tree model that corrects the neural ranker's scores, mixed with a neural ranker fine-tuned on validation | 0.3839 | 0.3772 | yes |
| 3 | 2026-11-11 | v5 | `sub_v5_nnfinal2_res2` | Equal mix of the second neural ensemble, fine-tuned on validation, and a tree model correcting its train-only version | 0.3869 (a) | 0.3802 | yes |
| 4 | 2026-11-11 | v5_lh | `sub_v5_nnfinal2_res2_lh` | Row 3 plus a boost for items found in the same user's later history | not recorded | 0.3802 | no |
| 5 | 2026-11-11 | v10 | `sub_v10_nnfinal4mix_pre` | Four neural rankers trained for 10 epochs, each used half fine-tuned on validation and half train-only, plus a small share of encoder scores | 0.39399 | 0.38786 | yes |
| 6 | 2026-11-11 | v13 (Cd) | `sub_v13_Cd_memeq_F5cdF6F7` | Equal-weight pool of 19 rankers from three training recipes with the new optimiser and retrieval, chosen on the holdout | 0.39901 | 0.39208 | yes |
| 7 | 2026-11-12 | G3 | `cand_G_F10F7xF7x70_memeq` | The best click-label pool: 20 click-label rankers, 6 with a GRU history encoder and 14 whose cooldown was re-run with 2% random-order feeds | 0.40032 | 0.392820 | yes |
| 8 | 2026-11-12 | PF7x | `cand_PF7x_memeq_6fam` | Pool of 38 rankers from six recipes, one of them the re-run members | 0.39974 | 0.392723 | no |
| 9 | 2026-11-12 | M2 | `cand_M_F7xF7x70_memeq` | Pool of the 14 re-run members alone; best validation score of the day | 0.40094 | 0.392434 | no |
| 10 | 2026-11-12 | S2 | `cand_S2_nnstk_g3gate3r` | Row 7 with a small neural model that re-weights its members per feed | 0.40088 | 0.392473 | no |
| 11 | 2026-11-12 | G3 overlay | `cand_G3_later_overlay` | Row 7 with items from the same user's later feed pushed to the top; the leak, tested on purpose | 0.40486 (b) | 0.392819 (c) | no |
| 12 | 2026-11-12 | G3 TO-only | `cand_G3_TOonly` | Row 7 without its fine-tuned branch | 0.40035 | 0.392346 | no |
| 13 | 2026-11-12 | Pall | `cand_Pall_memeq_6fam` | Pool of 38 rankers from six recipes with no re-run member; second pick at the time | 0.39896 | 0.392436 | no |
| 14 | 2026-11-13 | Q1a | `cand_Q1a_G3popfree_memeq` | Row 7 plus 4 rankers trained without any popularity input | 0.40068 | 0.392886 | yes (d) |
| 15 | 2026-11-13 | Q2a | `cand_Q2a_G3pop1_memeq` | Row 7 plus 4 popularity-light rankers | 0.40061 | 0.393205 | yes |
| 16 | 2026-11-13 | C3 | `cand_Q2a8_ftbag3_memeq` | The best click-label file: row 7 plus 8 popularity-light rankers, every member fine-tuned with 3 seeds | 0.40101 | 0.393454 (e) | yes |
| 17 | 2026-11-13 | C1 | `cand_Q2a8_G3pop1x8_memeq` | Row 7 plus 8 popularity-light rankers, fine-tuned with one seed | 0.40089 | 0.393394 | no |
| 18 | 2026-11-13 | C2 | `cand_G3_ftbag3_memeq` | Row 7 with the fine-tune averaged over 3 seeds and no new member | 0.40051 | 0.392754 | no |
| 19 | 2026-11-14 | W8 | `cand_W8kd_r2` | 8 full-view students of width 192 | 0.40165 | 0.394486 | yes |
| 20 | 2026-11-14 | S16 | `cand_S16kd_r2` | Row 19 plus 8 full-view students of width 128 | 0.40173 | 0.394078 | no |
| 21 | 2026-11-15 | W16 | `cand_W16kd_r3` | 16 full-view students of width 192 | 0.40158 | 0.39448337 | no |
| 22 | 2026-11-15 | W8P20 | `cand_W8P20kd_r3` | Row 19 blended 80/20 with 4 popularity-light students of width 128; members trained without the 30,000 holdout feeds; pick 2 from 2026-11-18 to 2026-11-19 | 0.40214 | 0.3950094 | yes |
| 23 | 2026-11-15 | W8TO | `cand_W8TOkd_r3` | Row 19 without its fine-tuned branch | 0.40157 | 0.394009 | no |
| 24 | 2026-11-15 | FD16P20 | `cand_FD16P20kd_r4` | The design of row 22 retrained on all train feeds: 16 full-view students blended 80/20 with 4 popularity-light students, all of width 192; pick 1 until 2026-11-18 | 0.40271 (f) | 0.394892 | no |
| 25 | 2026-11-15 | FD16P20@m0.2_0.4 | `cand_FD16P20_m02_04kd_r4` | The 20 members of row 24 with the popularity-light weight raised to 0.4 on feeds of 50 or more candidates | 0.40298 | 0.3947465 | no |
| 26 | 2026-11-16 | FD16P20vf | `cand_FD16P20vfkd_r4` | Row 24 with 4 popularity-free students added to its popularity-light part, 24 members; a candidate for pick 2 until 2026-11-18 | 0.40286 | 0.394630 | no |
| 27 | 2026-11-16 | ENSA68 | `exp_ENSA68kd_1115` | 68 members: 32 full-view students and 36 popularity-light students | 0.40247 | 0.3946427 | no |
| 28 | 2026-11-16 | W8 FT-only | `probe_W8_FTonly_1115` | The fine-tuned branch of row 19 alone | 0.40087 | 0.393960 | no |
| 29 | 2026-11-16 | FDN16P20 | `cand_FDN16P20kd_r4` | 16 all-data full-view students trained with one extra input, a two-step co-visitation walk, and the popularity-light part of row 24 | 0.40283 | 0.3943559 | no |
| 30 | 2026-11-16 | FDM32P20 | `cand_FDM32P20kd_r4` | The 16 full-view students of row 24 plus the 16 of row 29, and the same popularity-light part | 0.40264 | 0.394578 | no |
| 31 | 2026-11-17 | FDX16P20:ref | `cand_FDX16P20_refkd_r4` | The 16 full-view students of row 24 with their cooldown re-run on all data from the stored checkpoint, and the popularity-light part of row 24; control for row 32 | 0.40214 | 0.39482063 | no |
| 32 | 2026-11-17 | FDX16P20:fsw50t | `cand_FDX16P20_fsw50tkd_r4` | Row 31 with a per-feed loss weight proportional to the square root of the feed size in the re-run cooldown | 0.40212 | 0.39475033 | no |
| 33 | 2026-11-17 | FD16P20@m0.4_0.2 | `cand_FD16P20_m04_02kd_r4` | The 20 members of row 24 with the popularity-light weight raised to 0.4 on feeds of fewer than 50 candidates; pick 1 | 0.40293 (g) | 0.3952092 | yes |
| 34 | 2026-11-17 | FD24P20 | `cand_FD24P20kd_r4` | Row 24 with 8 more all-data full-view students, 24 in all | 0.40258 | 0.394905 | no |
| 35 | 2026-11-17 | FD48P20 | `cand_FD48P20kd_r4` | Row 24 with 32 more all-data full-view students, 48 in all | 0.40222 | 0.39484072 | no |
| 36 | 2026-11-18 | FDXH16P20:ps | `cand_FDXH16P20_pskd_r4` | Row 24 with its 8 students of the 11-member teacher replaced by all-data students trained with pseudo-feeds built from user histories | 0.40298 | 0.3947873 | no |
| 37 | 2026-11-18 | A1 | `cand_FD16P20_m04_02_A1kd_r4` | Row 33 with the popularity-light part of its fine-tuned branch, on feeds of 50 or more candidates, taken from the popularity-light students' train-only predictions | 0.40310 | 0.395036 | no |
| 38 | 2026-11-19 | FDX16P20:ps | `cand_FDX16P20_pskd_r4` | Row 24 with all 16 full-view students replaced by all-data students that also train on pseudo-feeds built from user histories; pick 2 from 2026-11-19 | 0.40286 | 0.39526061 (h) | yes |
| 39 | 2026-11-19 | FDXU32P20:ps | `cand_FDXU32P20_pskd_r4` | Row 24 plus the 16 pseudo-feed students of row 38, 32 full-view students in all | 0.40301 | 0.39524613 | no |
| 40 | 2026-11-19 | FDX16P20:hz | `cand_FDX16P20_hzkd_r4` | The 16 full-view students of row 24 with their cooldown re-run on all train feeds on a stale view of the popularity inputs, and the popularity-light part of row 24 | 0.40224 | 0.3949761 | no |

Notes to the table:

- (a) About 0.002 of this validation score came from the two feature groups that used the same user's later snapshots. The clean value of that pipeline was about 0.385.
- (b) Inflated by the leak. The base file, row 7, reads 0.40032.
- (c) The log records the overlay as 0.392819, "equal to the base file to 6 decimals". The board value of the base file is 0.3928199333, which the records carry as 0.392819 or 0.392820.
- (d) 0.000066 above row 7, far inside the noise. It was the team best for two minutes, until row 15.
- (e) Attributed by inference. The submission note named the file of row 15, which had already scored 0.393205, and a file scores the same every time, so the new score was read as row 16. It was never confirmed on the platform.
- (f) A favourable draw of members. The unbiased validation level of this family is about 0.4024.
- (g) On validation the short-feed weight 0.4 is a single high point: 0.3 and 0.5 read 0.40266 and 0.40251. The public lead over row 24, +0.00032, is about one paired standard error.
- (h) The public lead over row 24 is +0.00037 at a paired standard error of 0.00047, and the lead over row 33 is +0.00005, a tie. Row 36, with only 8 of the 16 students replaced, scored 0.00010 below row 24.

## Board snapshots

The public board was fetched six times between 11 and 13 November, once on 18 November at 03:33, after the first choice of final picks, and once on 19 November at 03:34, two minutes after pick 2 changed. The first snapshot predates the team's first score. There was no fetch from 14 to 17 November, so the team's place on those days is not on record.

| Time (UTC) | Leader | Second | Third | Team place | Team best | Teams |
|---|---|---|---|---|---|---|
| 2026-11-11, hour not recorded | 0.39354 | 0.38995 | 0.38953 | none yet | none yet | not recorded |
| 2026-11-12 20:20 | 0.398256 | 0.394695 | 0.393767 | 4 | 0.392820 | 439 |
| 2026-11-12, about 22:00 | 0.3982555809 | 0.3946950126 | 0.3937667627 | 5 | 0.3928199333 | 440 |
| 2026-11-13 03:50 | 0.398256 | 0.395115 | 0.393767 | 5 | 0.392820 | 441 |
| 2026-11-13 09:22 | 0.398256 | 0.395115 | 0.393767 | 5 | 0.392820 | 444 |
| 2026-11-13 19:43 | 0.398256 | 0.395115 | 0.393778 | 4 | 0.393206 | 453 |
| 2026-11-18 03:33 | 0.3982556 | 0.3972749 | 0.3964761 | 8 | 0.3952092 | 532 |
| 2026-11-19 03:34 | 0.3982556 | 0.3980582 | 0.3968832 | 9 | 0.3952606 | 541 |

The 0.393206 of the 13 November 19:43 row is the file of row 15, which the team's own log holds as 0.393205. The leader's score has not changed since it appeared, at about 19:54 on 12 November.

On 19 November at 03:34 the board listed 541 teams, 1,438 participants and 7,785 submissions. It had listed 532 teams, 1,397 participants and 7,384 submissions on 18 November, and 453 teams, 1,247 participants and 5,722 submissions at 19:43 on 13 November.

The top of the board on 19 November at 03:34, from the [public leaderboard](https://ods.ai/competitions/avitotechrecsys2026/leaderboard):

| Place | Public NDCG | Difference to the team |
|---|---|---|
| 1 | 0.3982556 | +0.00299 |
| 2 | 0.3980582 | +0.00280 |
| 3 | 0.3968832 | +0.00162 |
| 4 | 0.3966275 | +0.00137 |
| 5 | 0.3963550 | +0.00109 |
| 6 | 0.3959215 | +0.00066 |
| 7 | 0.3954529 | +0.00019 |
| 8 | 0.3953650 | +0.00010 |
| 9 (the team, pick 2) | 0.3952606 | 0 |
| 10 | 0.3949725 | -0.00029 |

Against the snapshot of 13 November at 19:43, the score of pick 1, 0.3952092, would have been second, 0.0001 above the 0.395115 then in second place. Six teams posted scores above it between the two fetches, so on 18 November it was eighth, 0.0030 below the leader.

On 19 November the team's best is 0.00005 higher and ninth. One team passed it, from ninth place on 18 November (0.39501) to sixth (0.39592). Five of the seven teams above it on 18 November raised their scores, and place 2 went from 0.39727 to 0.39806. The private ranking, on the other 20,033 test feeds, is unknown until 2026-12-09.
