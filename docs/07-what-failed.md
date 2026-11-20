# What failed

In brief:

- 312 ideas were catalogued: 58 adopted, 22 helped then superseded, 155 no effect, 45 hurt, 10 inconclusive and 22 not run. The adopted ones that made the score are in [06-what-worked.md](06-what-worked.md).
- Twenty failures changed how the work was done. Each is told below as a short story, followed by the patterns they share and the full catalogue by theme.
- A recurring lesson: a change that makes one model better can make the ensemble worse, often because the members become more alike.
- [evidence.md](evidence.md) names the published file behind the headline number of each of the twenty failures. [catalogue.csv](catalogue.csv) holds the catalogue below as a table, one row per idea.

## Conventions

- Scores, splits and notation follow [04-validation.md](04-validation.md); other terms are in [glossary.md](glossary.md). A difference is paired per feed against the reference of its day, "±" is one standard error (SE) of that difference, z is the difference divided by its SE, and a pair "a / b" is validation / holdout. Members trained on all data have no holdout number.
- The witnesses are the five samples a change is judged on: validation, the holdout, and three samples derived from them, the lagged holdout, the private-mix view and stale validation ([04-validation.md](04-validation.md#the-witnesses)). Pooled z is (z on validation + z on the holdout) / sqrt(2).
- A member is one trained ranker, and a pool is the equal-weight mean of its members' within-feed z-scores. A student is a ranker trained toward half clicks and half a teacher ensemble's scores ([distillation](03-solution.md#distillation)). Full-view students read all 34 pre-computed inputs; popularity-light students have the 8 popularity inputs replaced by one exposure count. Wide means width 192, narrow width 128.
- A random-order feed was displayed in random order, not in the order of the site's ranker. Every validation and test feed is one, and so are about 13% of the train feeds with 50 or more candidates. The cooldown is the last 20% of training steps, in which the learning rate falls.
- FD16P20 is 16 full-view and 4 popularity-light students, all of width 192 and retrained on every train feed (validation 0.40271, public 0.394892). Pick 1, FD16P20@m0.4_0.2, is the same 20 members with the popularity-light weight raised to 0.4 on feeds of fewer than 50 candidates (validation 0.40293, public 0.3952092).
- Pick 2, FDX16P20:ps, is FD16P20 with its 16 full-view students retrained on all train feeds plus pseudo-feeds built from user histories (validation 0.40286, public 0.39526061). It replaced W8P20 (public 0.3950094), whose members held out the 30,000 feeds, on 19 November.
- From the evening of 14 November a change to the student recipe had to pass at pool level with 8 seeds: validation and holdout both at or above zero, and a pooled z of at least 2. Most arms were screened first with 4 seeds, paired seed by seed with the reference recipe, and went on to 8 seeds at a pooled z of 1.6 or more. The rules and their dated reads are in the decision ledger, a working record that is not published.
- Costs are GPU-hours as recorded: hours of one process, with several processes sharing a GPU.

## Twenty failures that changed the method

### 1. A boost for items seen in the user's later history

A feed's clicks reappear in the same user's next history snapshot (92.1% of positives), so moving such candidates to the top of test feeds looked like a free +0.004 on public. Laid over the best click-label pool, the boost read +0.00454 ± 0.00047 on validation and scored 0.392819 on public, against 0.392820 for the base file.

Moving 108 items to the top of 57 public feeds left the score unchanged to six decimals, so those feeds are either not scored or have no positives left. The full account is in [02-what-the-data-told-us.md](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak), and [04-validation.md](04-validation.md#leak-incidents) lists every leak incident.

### 2. Encoder scores for train rows, first build

The trajectory encoder (failure 11) scored validation and test rows, worth +0.0037 ± 0.0008 on validation in the tree harness. The 3.15 million train feeds had no scores, so only the fine-tune could use them. Giving train rows the same inputs took four builds.

The first, from encoders trained on time windows, raised the holdout by 0.006 to 0.0085 and lowered train-only validation by 0.005 to 0.009. Two witnesses moving that far in opposite directions pointed at the build.

The encoder's item table held the validation and test candidates and every item seen in any history, later snapshots included, but no train candidates. A train candidate with content in the table and zero popularity was therefore often the positive of its own feed. The holdout is made of train rows and shared the artefact; validation did not.

- The second build, a stop-gap, blanked the scores wherever an item had no popularity: +0.0011 ± 0.0007 / +0.0008 ± 0.0004 at 3 epochs. It passed the rule of that day, 2 SE on either witness, through the holdout alone, which a later review counted as a process error. In the final members it was worth -0.0003 ± 0.0008 on validation after the fine-tune.
- The third put every candidate into the item table and passed its test with a broken input: a clip at ±10 in the loader left 76 to 92% of rows, depending on column and split, at the clip value, so those columns were nearly constant.
- The fourth, rescaled copies of the third, read +0.00191 ± 0.00058 / +0.0002 at 3 epochs (2 seeds) and +0.00077 ± 0.00073 on validation in a 10-epoch removal pair. It is in the final input set.

The fourth build was accepted only after its train rows matched: on zero-popularity train rows its mean z is -0.153, against -0.146 on validation and -1.46 for the leaky build. The rule kept: a train row is built exactly like a validation or test row, and a new statistic must show a population stability index (PSI) below 0.5 across splits before any training.

### 3. A cooldown dose tuned at 2 epochs and used at 10

The first day's optimiser study produced a schedule that holds the learning rate constant and then cools it down over the last 20% of steps, drawing a share of the cooldown batches from the random-order train feeds (134,118 without the holdout). The share was tuned at 2 epochs, where 30% won.

At 10 epochs the same 30% collapsed all six members after step 85,000: validation fell from about 0.391 to 0.364 and the holdout from about 0.317 to 0.289. A same-seed run at 10% was identical at step 77,500 (0.39270 on validation) and 0.0157 ahead at step 87,500.

A share is a fraction of the run, so its dose grows with the run. 30% of a 2-epoch cooldown is about 1.9 passes over the random-order pool, 30% of a 10-epoch cooldown about 9 passes, enough to memorise it. The design note of the schedule had this arithmetic in it and called the 10-epoch regime untested. The fix took three steps:

- That night the cooldown was re-run at 10% from checkpoints kept from before the collapse: +0.0018 ± 0.0005 / +0.0015 ± 0.0003 per member.
- 10% still over-fitted late in 19 of 21 runs: in cooldown re-runs the best step fell between 5,000 and 10,000 of 19,670, and the runs then lost 0.0026 / 0.0026. Over all 285 training runs to that date, the validation NDCG lost between the holdout-selected step and the last step grew with the share: 0.0007 at 0%, 0.0033 at 10%, 0.027 at 30%.
- The final share is 2%, run as a separate 2-epoch cooldown re-run from a stored plateau checkpoint: +0.00117 ± 0.00039 / +0.00134 ± 0.00022 per member over 7 runs against 10%, with the best step at or near the last. Against 2%, shares of 1% and 4% read -0.00035 and -0.00077 on validation, and a share of 0% left validation level (+0.00002 ± 0.00025) but lost 0.0023 ± 0.0005 (z -4.5) on validation feeds with cold positives.

### 4. More epochs and larger batches

Ten epochs had beaten two: under the final optimiser a 6-seed pool gained +0.0040 / +0.0031, and the holdout of a 10-epoch run was still rising through its constant-rate phase (0.3158 at step 20,000, 0.3173 at step 75,000). A gradient-noise scale of 4,900 to 9,400 feeds, against a batch of about 317, argued for larger batches. Longer runs and larger batches were both measured against 10 epochs at the normal batch:

| Change | Validation / holdout |
|---|---|
| 15, 20 and 30 epochs | -0.0006 / -0.0002, -0.0008 / -0.0006, -0.0016 / -0.0007 |
| 20 epochs, students | -0.0008 / -0.0004 |
| 4× batch | -0.00262 / -0.00209 |
| 4× batch, doubled learning rate | -0.00232 / -0.00163 |

The constant-rate phase is saturated by about 5 epochs, and the gain comes from the cooldown, which needs its number of steps. On the holdout it adds about +0.0035 at the normal batch, nothing at four times the batch with the same learning rate and +0.0012 with twice the rate. A longer run stretches the cooldown until it memorises the random-order pool (failure 3); a larger batch shortens it.

The longer-run and larger-batch members were more different from the base members than seeds are from each other, but too weak for that difference to pay in a pool.

### 5. Averaging plateau checkpoints

Averaging checkpoints from the constant-rate plateau is the textbook way to a better model at no training cost. For 7 seeds of one family, the average of the checkpoints at steps 60,000, 70,000 and 77,500 was compared with the same run's cooled checkpoint.

The averaged member read +0.00041 ± 0.00040 / +0.00072, with 6 of 7 seeds positive (the first seed tried read +0.0024). Against the pool of cooled members, the 7-member pool of averaged members lost 0.00247 ± 0.00055 on validation and 0.00164 ± 0.00032 on the holdout, and the within-feed correlation between seeds rose from 0.941 to 0.975.

Averaging removes the run-to-run noise that makes seeds differ, and a pool gains over a member through that difference. From this failure and the next, every recipe change was read at pool level.

### 6. Cheap two-epoch seeds

A 2-epoch member scored 0.39322 / 0.31775, close to the 0.39460 / 0.31940 of the best 10-epoch family, in a fifth of the training time, and passed the single-member strength gate. Its seeds correlated 0.978 with each other, against 0.930 to 0.941 within every 10-epoch family. Adding twelve of them lowered the holdout from 0.32319 to 0.32208 (-0.00112 ± 0.00024), and the holdout rule written before the read rejected the pool.

Going from 1 to 6 members gained +0.0010 on validation, where the 10-epoch families gained +0.0040 to +0.0047, so six 2-epoch seeds were worth about 1.5 to 2 ten-epoch members in a pool. The likely reason, never isolated in a test, is that the long constant-rate phase is what drives seeds apart, so the cost of a member cannot be cut there.

### 7. Regularisers in the cooldown

Failure 3 showed that members memorise the random-order pool late in the cooldown, even at a 10% share. Regularisers that punish memorisation were the direct remedy, each tried as a cooldown re-run from stored plateau checkpoints:

| Regulariser | Result |
|---|---|
| R-Drop (two forward passes tied by a KL term) | member -0.00012 ± 0.00032 / +0.00059 (z 3.3); pool holdout +0.00036 (z 1.5) |
| Sharpness-aware minimisation | member +0.00004 / +0.00041 (z 2.4); pool holdout +0.00025 (z 1.1) |
| Auxiliary-loss weight 1.0 in the cooldown | members +0.00032 ± 0.00020 on validation over four runs; a 7-run pool 0.00023 lower on the holdout |
| No dropout on the plateau, 0.1 in the cooldown | +0.00006 ± 0.00071 / +0.00006 ± 0.00041 |
| Weight decay 0.1 in the cooldown | -0.00055 on validation |
| Label smoothing, on a small proxy model | within ±0.001; did not stop the proxy's collapse |
| Students: dropout 0.2; 30% of history events dropped | pooled z -1.88; pooled z -2.62 |
| Students: dropout of the popularity inputs in the cooldown | -0.00066 ± 0.00048 on validation |

R-Drop and sharpness-aware members correlated 0.990 with the plain ones: a regulariser that leaves the data schedule alone produces a near copy of the base member. What worked against memorisation was the dose of failure 3, fewer passes over the pool. Two more regularisers of the same class, for which no code existed, were not built.

### 8. Weight decay in both directions

Weight decay had stood at 0.02 since the first day. More of it might slow the late memorisation, and when that failed, less of it promised members that differ more. Each dose was read on 8 seeds of students (4 at 0.10):

| Weight decay | Mean correlation between members | Pool, validation / holdout | Pooled z |
|---|---|---|---|
| 0 | 0.9563 | -0.00134 ± 0.00056 / -0.00011 ± 0.00031 | -1.95 |
| 0.01 | 0.9643 | -0.00032 ± 0.00051 / -0.00021 ± 0.00028 | -0.97 |
| 0.02 | 0.9722 (seeds 951-954), 0.9723 (seeds 951-958) | reference | |
| 0.05 (`kwwd05`) | 0.9810 | -0.00043 ± 0.00049 / -0.00063 ± 0.00027 | -2.31 |
| 0.10 | 0.9848 | -0.00169 ± 0.00061 / -0.00091 ± 0.00034 | -3.86 |

At 0.05 the single member was slightly better (+0.00056 / +0.00008); at 0, an 8-seed arm planned at 27 GPU-hours, it lost 0.00195 ± 0.00037 / 0.00135 ± 0.00022. Both directions lose, for opposite reasons: above 0.02 the members converge, below it they weaken faster than they diverge.

### 9. A heavier teacher, twice

At a teacher weight of 0.8 a single wide student gained +0.00126 ± 0.00033 / +0.00078 ± 0.00019 over one at 0.5. At width 128 the 4-pools had already been level or lower at every other weight: 0.39977 at 0.3, 0.40004 at 0.8 and 0.39906 at 1.0 on validation, against 0.40085 at 0.5. The pools of wide students lost both times:

- 8 seeds trained at 0.8 (`kd08w192`): the 8-pool read -0.00049 ± 0.00048 / -0.00029 ± 0.00026 against the 8-pool at 0.5 (pooled z -1.51), and pooled z -2.36 against the average of same-size pools drawn from the 16 reference students.
- A day later, with the GPUs idle, the cheap form ran: teacher weight 0.8 in the cooldown only (`kw08`), as cooldown re-runs of about 0.25 GPU-hours per member. Members gained +0.00061 on both witnesses; the 16-pool read -0.00047 (z -1.21) / -0.00059 (z -2.86), pooled z -2.88. The forecast written before the run had put the chance of a pass at 4%.

Students that follow one teacher more closely are more alike, and a pool averages over what is left of their differences.

### 10. An item-ID embedding table

Item identity is the classic recommender input, and 77% of validation positives have an item id that occurs in train. A hashed table of 2^22 rows by 32 dimensions was added to the candidate and history tokens, initialised at zero and trained with 50% ID dropout, so that an unseen item looks like a dropped one.

At its holdout-selected step the model was 0.0022 ± 0.0008 worse on validation, and in the second epoch it collapsed to 0.372 as it memorised train ids. Feeds whose positives all have a seen id were neutral (-0.0002 ± 0.0010), and feeds with no covered positive lost 0.0075 ± 0.0021.

The table teaches the model to prefer items it knows, and test would have lost more: 61% of its candidates have an id seen in train, against 71% on validation. Students read -0.0057 / -0.0049 over two seeds. Collaborative signal reached the model in two other forms: co-visitation counts, and the trajectory encoder, whose own ID table is trained without click labels.

### 11. Pre-training on the history corpus

The histories are the largest unlabelled resource in the data: 2.59 million sequences with 463.6 million events. Following papers such as ARGUS and PinnerFormer, a causal transformer (width 384, 4 layers, learned item-ID table) was trained to predict the next item a user opens from the feed; two epochs took 22 minutes on one GPU. Zero-shot it was the best single signal, and every later use shrank:

| Use of the encoder | Result |
|---|---|
| Zero-shot ranking | 0.3590 on validation; the best similarity feature scored 0.349 |
| Blend term | +0.00082 ± 0.00038 on the pool of that day; -0.00078 ± 0.00042 on a stronger one |
| Fine-tuned on validation labels | out-of-fold 0.3536 fell to between 0.350 and 0.352 |
| Score columns as ranker inputs | +0.00128 ± 0.00061 / +0.00054 in a 2-epoch removal test; +0.00077 ± 0.00073 on validation at 10 epochs |
| Frozen hidden states as an extra branch | -0.00007 ± 0.00053 |
| Initialising the ranker's history encoder | failed its gate on stored predictions (residual on the pool at or below zero); never trained |

The 10-epoch worth was below the +0.0015 set as the trigger for a bigger encoder. A study of about 5.6 GPU-hours agreed: across 19 stored encoder variants the worth of the channel did not grow with encoder quality (+0.00003 per +0.001 of zero-shot score).

A second programme of six designs, among them a masked-event task, an in-category encoder and contrastive user encoders, was refuted on paper and closed with four CPU diagnostics that took about 30 minutes and were all null.

The reason is in the corpus. The unlabelled histories hold 463.6 million events and the histories inside the labelled feeds 448.7 million. Pre-training pays when unlabelled data dwarfs labelled data or when an ID table would overfit on labels, and here the ranker already reads nearly the same sequences with a label attached. What stayed in the final model are the six encoder columns of failure 2.

The size argument compares inputs. The supervision differs: each of the 463.6 million events says what the user did next, against about 5.5 million positive labels in the feeds. Training the history encoder inside the ranker on those next-event targets never ran, and [section 6 of 12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) treats it as an idea the search may have missed.

### 12. Language-model backbones

The data has no text: ids, categories and features are anonymised numbers. The feasibility note expected a language model to add only a computation prior, worth +0.00002 to +0.00008, and the test ran anyway as a matched comparison with its acceptance rules written first.

Three Qwen trunks read the ranker's inputs as soft tokens (LoRA rank 32, the ranker's loss, 150,000 train feeds) against two ranker seeds on the same data (mean 0.38003 / 0.30386):

| Trunk | Epochs | Against the ranker, validation / holdout |
|---|---|---|
| 0.6B | 4 | -0.00378 ± 0.00077 / -0.00398 ± 0.00046 |
| 4B | 2 | -0.00480 ± 0.00081 / -0.00479 ± 0.00049 |
| 9B | 2 | -0.00532 ± 0.00080 / -0.00477 ± 0.00050 |

The curve is flat over a 16-fold range of size, and the 0.6B architecture trained from random initialisation was within one SE of the pretrained one (-0.00007 ± 0.00060 on validation at 2 epochs), so language pre-training added nothing measurable. The trunks correlated 0.932 to 0.943 with the ranker, against 0.946 between two ranker seeds, which ruled them out as pool members. The study used 46.74 of its 60 capped GPU-hours.

Untrained, a language model ranking 100 feeds from a text rendering scored 0.3957 against 0.4433 for the pool (-0.048 ± 0.023), the level of a one-column heuristic. The limit of this task is information, and a larger trunk reads the same inputs. Not tested: tuning per model size, more than 150,000 feeds, more than 2 epochs at 4B and 9B.

### 13. Decoding the category and event codes

Every code in the data is anonymised, and knowing what the codes mean promised features that a model cannot guess.

The decoding worked: event types and sources are frequency ranks, categories follow the site's public category-id order (52 leaf categories in use, codes 2 to 53, in 11 verticals), f4 is listing age, f2 a region key and f5 an unsmoothed rate. The target is an engagement that starts from the recommendation feed: 83 to 87% of positives come back in a later history as a feed-sourced view.

The features built on it did not help. Of 47 cues tested as residuals on the best click-label pool (vertical shares, delivery classes, region match, funnel depth, category-relative age and others), none cleared +0.0003 at 2 SE on validation.

The one positive sign, an item's share of opens that come from the feed, read +0.00033 ± 0.00042 on validation and turned out to echo other users' validation clicks through their later snapshots. Rebuilt from strictly earlier days it read -0.00001 ± 0.00030, and -0.00073 ± 0.00024 on the holdout.

The idea that a contact means a satisfied need was refuted: near-duplicates of contacted items are clicked more (odds ratio within category 1.16 to 1.18). The home-region effect is real (odds ratio 0.82 to 0.89), and the pool was already calibrated on it (actual over expected 0.981).

The residual gate is nearly blind (failure 14), but the calibration supports the closure: the pool's actual over expected click rate is 0.97 to 1.03 on every decoded class, because the ranker embeds every code and sees the groupings in 3 million feeds. The decoding paid once, indirectly: it showed that members without popularity inputs lacked exposure information, which led to the popularity-light view.

### 14. The residual gate, and the trained input that passed every gate

A trained input costs GPU-hours per seed, so for most of the project features were screened as a residual on the pool: one weight fitted in user folds on validation and checked on the holdout. About 300 feature rows were closed that way.

On 14 November a positive control showed that the gate sees only 10 to 26% of a true effect. A block of inputs worth +0.00852 ± 0.00066 as trained inputs read +0.00086 with a linear residual and +0.00222 with monotone trees on the holdout, and nothing on validation. Two single columns worth +0.0015 and +0.0020 as trained inputs had been rejected by it. The 300 closures hold for the residual form only.

A sharper screen, a calibrated conditional score test, found 25 of 284 stored columns on both witnesses, which reduce to six independent pieces of information. One survived the parity and lag checks: a two-step co-visitation walk over the whole history (`nx3`). Trained as a real input, it failed at every level:

- click-label members (8 seeds): pooled z +0.59 against a bar of 2.0 (+0.00037 ± 0.00056 on validation, +0.00006 ± 0.00033 on the 15,225 feeds of the lagged holdout where the input can be built at the test lag);
- students (8 seeds): +0.00005 ± 0.00029 and +0.00016 ± 0.00024, pooled z +0.62;
- the file of 16 all-data students: +0.00011 ± 0.00038 on validation and 0.3943559 on public, 0.00054 ± 0.00050 below FD16P20.

The members do use the column: with its values permuted they lose 0.00179 ± 0.00034 on validation. The likely reading is that the model learns the new column and stops extracting the same information from inputs it already had; a screen that detects information measures the first effect, while the score moves by the difference.

The forecast before the run was +0.0003 per member, which neither witness can show at any number of seeds, because the standard error is feed noise. Members with and without the walk were no more different than two plain members (correlation 0.9487 across groups, 0.9490 within).

### 15. The rebuilt click-strength input

The second round of the same search found a statistic of how often an item is clicked relative to the clicks expected at the positions where it was shown. On the conditional test, whose statistic is standard normal when a column adds nothing, it read +4.17 on the lagged holdout and +4.18 on stale validation, far above every other candidate.

Two checks failed: a placebo built from exposure alone read +2.56 and +2.34, and on train rows built to look like test, 37.0% of items aged 7 to 10 days had been seen before, against 14.3% on private rows.

The rebuild gave every row of every split three consecutive source days, with the lag to the newest of them matched to private (8, 9 or 10 days). Three days is the deepest window for which a 10-day lag exists on train rows, because the data starts on 6 September. Exposure was removed by centring the value within the feed and the exposure bin.

Both checks then passed: the placebo was centred at zero, and the share of 7 to 10 day old items seen before was 0.155 against 0.143. What was left read +1.22 on the lagged holdout and +1.39 on stale validation for the form built on all feeds, and +1.85 and +2.17 for the form built on random-order feeds, against a bar of 2.74. Its direct NDCG on validation was at or below zero. Nothing was trained.

Built fresh, from the three days up to the row's own day, the statistic reads +6.94 on the holdout. The click part of an item's appeal decays within days, and the first version looked strong because it was partly exposure and partly fresher than a test row can be. The check ran on CPU between 14:26 and 14:50 UTC on 16 November, with the rule written before any array existed.

### 16. Trees in every role

Gradient-boosted trees were in the first submissions, and a second model family is the usual source of diversity in an ensemble:

| Tried as | Result |
|---|---|
| Model trained on train rows, and its blend with the network | about 0.368 to 0.370 on validation, against 0.38186 for the first neural ensemble, at a correlation of 0.85 to 0.875 with it; best blend 0.3820 |
| Stacker on top of the network | part of the first three submissions; the residual stacker, which adds features to the neural score, read -0.0001 ± 0.0006 once the network took co-visitation and popularity as inputs |
| Residual on the 20-member pool of 12 November | -0.00031 on validation, -0.00042 (z -2.3) on the holdout |
| Pool member (CatBoost YetiRank on random-order feeds) | 0.3820 at a correlation of 0.856 to the pool; +0.00015 ± 0.00024 at weight 0.05; break-even needs about 0.393 |
| Pipeline of the 2025 Avito ML Cup winners | +0.00067 ± 0.00045 on validation over the team's CatBoost baseline, inside the one-seed spread of 0.00086; as a member 0.3827 against a bar of 0.3933 |
| Trees given label statistics from all train feeds | best information gain +0.00147 ± 0.00043 / +0.00080 ± 0.00021 nats per click (gain in click log-likelihood, [08](08-outside-ideas.md#probe-results-on-the-best-file)), against a bar of 0.0016 |
| Student of the network's teacher | best 0.38461 / 0.30797, 0.017 below the neural pool |
| Members with CatBoost's native embedding and text features | pooled z between -1.42 and +0.67 |

A tree fitted directly to the pool's own score from 177 features reproduced it at a correlation of 0.909 and stopped at 0.3865 on validation: trees cannot rebuild what the network computes from histories and embeddings. A what-if on stored predictions showed that a tree which absorbs what the network knows never passes the member rule at any level up to 0.4006, while one that adds label signal the pool lacks would pass at about 0.3867.

As a proxy, trees misled once. A production feed is cut after the user's last interaction, so its last shown item is positive 18% of the time against a base rate of 2.2%. A training-only offset for this gained +0.0022 to +0.0031 in a tree model and read -0.0005 ± 0.0005 / -0.0001 per member and -0.0003 / -0.0009 for the pool in the ranker. After that a tree gain no longer justified a GPU run.

### 17. Validation-chosen pools and fitted stackers

With six families of members on disk, choosing the best pool on validation, or fitting the weights there, was the obvious step.

A design study had already warned against it. Against the mean of within-feed z-scores, rank-based fusion lost 0.0007 to 0.0020 on validation. Tree stackers over member scores read between -0.0020 ± 0.0006 and +0.0004 ± 0.0007, a listwise linear stacker +0.00095 ± 0.00064 and greedy member selection +0.0010 ± 0.0005 with 0.0014 of optimism, all against +0.00117 ± 0.00048 for an equal mix declared in advance.

The two files that looked best on validation were still submitted:

- The pool of the 14 members whose cooldown had been re-run with 2% random-order batches, the best pool of that day on validation (0.40094 against 0.40032 for the 20-member pool chosen by the holdout rule), scored 0.392434 on public against 0.392820.
- A small neural gate that re-weights the 20 members per feed read +0.00056 on validation and scored 0.392473, 0.00035 below the plain pool.

Both landed 0.0009 to 0.0010 below their public forecast, about the excess (0.00045 to 0.00087) that choosing the best of 5 to 50 near copies produces at the validation standard error ([04-validation.md](04-validation.md#noise)). Since then the combiner has no fitted parameter, and near copies are never ranked by validation or by public alone. A later sweep of 78 second-level combiners passed none on both witnesses, and family weights fitted on validation read z -2.85 on the holdout.

### 18. Growing the ensemble past 16 members

More seeds is the cheapest use of spare GPUs, and a fitted pool-size law, NDCG(n) = a - b / n, said how much to expect. 8 full-view students already sat 0.0002 below the asymptote a, and the law fitted on files (0.40254 - 0.00177 / n_full - 0.00014 / n_light) predicted +0.000055 on validation from 16 to 32 all-data students and +0.000018 from 32 to 48. Neither validation nor the board could see any of it:

| Pool | Difference on validation (/ holdout) | Public |
|---|---|---|
| 16 held-out full-view students, against 8 | -0.00007 ± 0.00034 / +0.00004 ± 0.00019 | 0.39448337 against 0.394486 |
| 68 members on disk (32 full-view and 36 popularity-light students), against FD16P20 | -0.00024 ± 0.00028 | 0.3946427, 0.00025 or 0.7 SE below FD16P20 |
| 24 all-data full-view students, against FD16P20 | -0.00013 ± 0.00022 | 0.394905, +0.000013 |
| 32 all-data full-view students | -0.00034 ± 0.00025 (0.40237 against 0.40271) | not scored |
| 48 all-data full-view students | -0.00049 ± 0.00028 (0.40222) | 0.39484072, -0.00005 |
| 8 or 12 popularity-light students in place of 4 | -0.00021 ± 0.00017 and -0.00016 ± 0.00018 | not scored |

The 32- and 48-student files failed their bar, which allowed a validation score no more than one SE below the reference. The 24- and 48-student files were scored on 17 November. Each of the 32 extra students took 220.8 to 258.8 minutes on a GPU.

Nothing was wrong with the new members. Their single-model scores (mean train-only 0.39931 and 0.39971 for the two teachers, against 0.39940 and 0.39951 for the first 16), pair correlations (0.9736 among the new, 0.9737 among the old) and logged arguments matched the old ones. Yet the train-only pool of the second 16 read 0.00086 ± 0.00039 below that of the first 16, and the third 16 read 0.00027 ± 0.00038 below.

FD16P20 was a favourable draw, a pool that sits above the mean of same-size pools of its recipe by seed luck. It was above both of its 8-member halves, and its 0.40271 sits about 0.0003 above the family's unbiased level of 0.4023 to 0.4024, which the level line ([04-validation.md](04-validation.md#noise)) maps to about 0.3949 on public.

Its four popularity-light students were a favourable draw too, at 0.3990 to 0.4004 per member against 0.3971 to 0.3990 for the next eight. Adding members diluted the luck.

It had happened before. The first 8 full-view students were a favourable draw of their recipe (a 12-student pool read 0.00024 lower on validation), and the 8-seed reference declared for later arm tests sat 0.0003 below the mean of same-size subsets. A non-inferiority bar against one realised file fails by noise more often than its nominal rate.

When arm and reference share no seeds, the reference to use is the average of same-size pools; [04-validation.md](04-validation.md#seeds-shared-by-arm-and-reference) covers the case of shared seeds.

### 19. A cooldown on stale popularity inputs

The 8 popularity inputs are fresh on train rows and stale on test rows. On average a warm item was last a train candidate 1.75 days before a holdout feed and 11.23 days before a private one.

With private-like stale inputs a pool of 8 full-view students loses 0.0033 ± 0.0008 on the lagged holdout and 0.0045 ± 0.0007 on the same holdout days under the causal stale view of arm `hz2` ([the measurements](02-what-the-data-told-us.md#cold-items-and-stale-popularity), [the staleness study](../experiments/popularity-staleness/)). Stale inputs during the cooldown should teach the model what test looks like.

The cooldown of the 16 full-view students was re-run on a view (`hz`) that closes 33% of the train-test coverage gap on 5 of the 8 inputs. It was read on stale validation, which re-predicts validation with private-like stale popularity inputs, and the verdict depended on the checkpoint:

- at the last checkpoint the 16-pool read +0.00024 ± 0.00038 on validation and +0.00120 ± 0.00045 (z 2.66) on stale validation, a pass;
- at the holdout-selected checkpoint it read -0.00009 on validation and +0.00077 ± 0.00044 on stale validation, no pass, although the two checkpoints differ by only 0.0003 on validation.

A full-dose view read +0.00141 ± 0.00060 on stale validation, only +0.00017 ± 0.00046 (z 0.4) above the half dose, and cost 0.0028 on the holdout with fresh inputs.

The file lost on validation. With 16 all-data members whose cooldown was re-run this way it read -0.00047 ± 0.00035 on validation against its plain twin (-0.00041 ± 0.00038 with 8 members), and -0.00072 ± 0.00060 on the private-mix view, which re-weights validation to the private mix of feed size and cold share.

The train-only branch was level with the plain one (-0.00001 ± 0.00036); the loss appeared once the fine-tuned branch was mixed in, whichever branch carried the stale-view members. Three member sets showed the same shape, and the recipe was closed.

Two earlier measurements explain most of it. The reference pool was a low draw under stale inputs: against three other references the gain on stale validation was +0.0006 to +0.0007, or 1.2 to 1.5 SE. And the stale view does the job of the popularity-light students already in the file: on top of their 0.2 weight it keeps 43% of its effect.

On 19 November the file went to the board as a probe. It scored 0.3949761, 0.00008 above FD16P20 at a paired SE of 0.00045, so public read it about 0.0005 better than validation did. That is a weak sign that public inputs behave stale, inside the noise, and no pick changed.

### 20. Post-processing rules and test-time tricks

Once the members were fixed, the remaining freedom was in how scores are combined per feed and how inference runs. Such tests cost CPU minutes, so hundreds ran, and a scan of that size produces winners by chance.

The rule that came closest set the popularity-light weight to 0.4 on feeds of 50 or more candidates and kept 0.2 on short ones. In-sample on validation it read +0.00026 to +0.00071 in six of seven member compositions and +0.00051 ± 0.00021 (z 2.43) on the first all-data file (8 full-view and 4 popularity-light students). Nested by user, with the choice of rule inside the folds, it read +0.00002 ± 0.00027 on the members of FD16P20.

On public it scored 0.3947465 against 0.394892 for FD16P20 (-0.00015, paired SE 0.00027). A greedy scan of the same rules on a placebo with nothing to find reads +0.00057 in-sample.

A wider search over 95 rule families (131,547 rules) and 7 families of learned gates found nothing that beat the fixed weight when nested on more than one member set. A router between pools built on 31 label-free feed descriptors had an AUC of 0.494 to 0.511, against 0.496 to 0.503 for a permuted null.

Since then an in-sample gain below about +0.0008 for a rule chosen from a scan counts as no evidence. The mirror rule, 0.4 on short feeds and 0.2 on long ones, scored 0.00032 above FD16P20 on 17 November, about one paired SE, and became pick 1 ([06-what-worked.md](06-what-worked.md#17-popularity-light-weight-04-on-short-feeds)).

At inference, on validation:

- Dropout sampling over 8 passes: -0.00009 ± 0.00024 for a 6-member pool.
- Dropping 10% of history events over 8 passes: +0.00036 per member (5 of 6 positive) and -0.00008 ± 0.00034 for the pool, because averaging over members already removes that variance.
- Averaging over shortened histories: -0.00100 ± 0.00045. 29% of validation feeds sit at the 256-event cap, and every dropped event costs information.
- A second pass over the head of the list: closed before training, since a perfect re-order of the top 8 recovers 29% of the validation headroom (the NDCG a perfect order would still add), and 66 to 77% of the headroom is in feeds with no positive in the top 8.
- An offset for warm items whose counts are deflated by the lag fixed a real calibration error (coefficient +0.20 ± 0.03) and read -0.00012 ± 0.00031: an error shared by a whole feed does not change the order inside it.

## Patterns that recur

- A gain for one member is not a gain for the pool. Plateau weight averaging, 2-epoch seeds, teacher weight 0.8 in two forms, weight decay 0.05, an auxiliary-loss weight of 1.0 and teacher-only labels on an unlabelled user fold all passed at member level and made the pool worse or no better. The last, over 4 seeds, read +0.00103 / +0.00052 per member and +0.00069 ± 0.00057 / -0.00017 ± 0.00032 for the pool.
- The pool-level rule has known error rates. With 8 seeds it passes 0.25% of null arms and 56% of true +0.0005 effects; the 2-seed screen it replaced passed 15% of nulls and 36% of true effects.
- Several changes traded accuracy on warm items for accuracy on cold ones and netted zero. Input dropout by listing age read +0.0247 on feeds whose positives are all young and -0.0063 elsewhere; unique retrieval slots +0.009 on cold positives and -0.0034 on warm ones. Error analysis found that every alternative model tried moves along this one axis.
- Different members were hard to manufacture. Under click labels a member from a different recipe was worth 2 to 3 seeds of the same recipe, but none of the eight arms of 13 November built to differ from the base advanced: the arms that differed were 0.001 to 0.003 too weak, and the strong ones were near copies. Only an input view, the popularity-light one, stayed different at nearly equal strength.
- A pass was often a draw. A 2-epoch screen read +0.00117 (z 2.48) for unique retrieval slots and +0.00036 ± 0.00068 at 10 epochs. The first four popularity-free students put their file above 70 of 70 control pools, four fresh ones above 16 of 70 ([the replication](04-validation.md#rules-written-before-the-read)).
- Several kept results rest on such draws. The first 8-student file (public 0.394486), the first four popularity-light students and FD16P20 were each a favourable draw of their recipe (failure 18), and the short-feed weight of pick 1 is a single high point of its validation curve.
- Files of one design cannot be ordered on the board. The seventeen scored files of the final family span 0.39436 to 0.39526 on public (mean 0.39486, standard deviation 0.00025), inside the paired SE of 0.0002 to 0.0006 between them ([11-final-picks.md](11-final-picks.md#risks)).
- Oracles and privileged teachers turned out to be the label seen from another side. The oracle for routing cold-item feeds (+0.0257) was a label echo, a score that reads the feed's own clicks, and a teacher allowed to see the user's later snapshots gained +0.037 ± 0.0025, all of it from events after the first interaction with a candidate. Before sizing an idea by an oracle, look at what the oracle is made of.
- Capacity and outside information did not move the model. Width and depth under click labels, routed experts and depth 7 for students, 13 block designs, three language-model trunks, 11,617 generated features (1.02% passed, against 1.01% of label-free noise features) and every external prior were null, while removing 20% of the users cost a member 0.0028 / 0.0022. The ranker is limited by the amount of labelled data and by what the inputs contain.
- A closure covers only the form that was tested. Self-distillation from one 2-epoch model read +0.00064 (z 1.6) on the first day and was closed; three days later an 11-member teacher gave members +0.00257 ± 0.00047 / +0.00200 ± 0.00028, and the first distilled file scored 0.001032 above the best click-label file on public. Width 192 was closed under click labels and adopted under distillation.
- A closed idea can still reach a final file. Pseudo-feeds failed their 8-seed rule on 17 November (randomization p 0.19). Built anyway as all-data try files, the file with 8 of the 16 full-view students replaced scored 0.00010 below FD16P20 and the file with all 16 replaced 0.00037 above it, the team's best public score. That file is pick 2, and neither reading is outside the paired noise.
- Some closures rest on a test known to be flawed. The 300 residual-form closures of failure 14 are one case. Students trained on random sub-lists of each feed (function matching, distillation on randomly changed inputs) are another: the sub-lists kept 1.88 positives against 1.22 on real short feeds, and the clean form was never run.
- Some closures never had a fair run. Next-event training of the history encoder, function matching, hindsight labels, a model of each item's audience and the click-label closures such as the position tower rest on rows below that closed them on paper, by a proxy or in a flawed form. [Section 6 of 12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) gives each with its evidence against it and a judgement, backed by no experiment, of 0.25 that at least one is worth +0.001 NDCG.
- Closures can also be wrong the other way. The holdout's veto closed popularity-light changes that public then rewarded (popularity-light students at weight 0.2: -0.00063 on the holdout, +0.000523 on public); [04-validation.md](04-validation.md#rules-written-before-the-read) has the audit that re-opened three ideas. [13-how-the-work-was-run.md](13-how-the-work-was-run.md#what-the-setup-lacked) sets this slip beside the other slips of the process.
- A scan of the closures (`lab/nn_x/w12/closures_scan/REPORT.md`) found what noise gives: 52 of 112 non-zero validation reads positive (46%) among model changes closed as no effect or inconclusive, and 11 of 93 reads with an SE at 1 ≤ z < 2, against 12.6 expected. The best estimate of hidden real +0.0003 effects is 0 (95% upper bound 13 of 93). Ten ideas with a positive first validation read (mean +0.00053) read -0.00001 ± 0.00013 in a stronger re-test.

## The catalogue by theme

![Bar chart of the 312 catalogued ideas by theme, each bar split by outcome: 58 adopted, 22 helped then superseded, 155 no effect, 45 hurt, 10 inconclusive and 22 not run in total. Architecture (38 ideas) and optimiser, schedule and regularisation (33) are the largest themes, and no effect is the most frequent outcome in 13 of the 17 themes.](figures/experiments-outcomes.svg)

One row is one catalogue entry. Several rows bundle many arms (13 block designs, a batch of 16 ideas from other competitions, 34 decoding cues, 63 tricks, 95 rule families), so the number of distinct things tried is well above 312.

Outcomes are: adopted (in the final pipeline or the team's method), helped then superseded, no effect (inside the noise of its test), hurt, inconclusive, and not run (closed by a bound, a probe or a low prior before any training).

How to read the cells:

- Effects follow the [conventions](#conventions) at the top of the page, and the five witnesses are defined in [04-validation.md](04-validation.md#the-witnesses). The lagged holdout is valid on the 13,424 feeds from 13 September on, and its build for the walk input of failure 14 covers 15,225 feeds.
- The "public-mix view" is validation re-weighted to public's mix of feed size and cold share, as the private-mix view is to private's.
- "Nested" means read on validation with the fitted weight or the chosen rule inside user folds.
- Fine-tune rows are read out-of-fold on validation, because the fine-tune has no holdout witness.
- Where a cell gives a lone number and names no witness, the record names none and the number is copied as recorded.
- The last column gives, as paths from the repository root, the research folder under [experiments/](../experiments/) that covers the idea and then the script or record behind the numbers: a file of the working tree under [lab/](../lab/), or a working record cited by name (the status log, the decision ledger, the experiment registry, the coverage map or the freeze protocol). Those records are not published, so a row that cites only a record has no file behind its number in this repository.

### Baselines, trees and stackers (15)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Reference levels of the metric: constant score, random order, raw feature f1 alone, best single similarity feature | helped then superseded | validation NDCG 0.1986, 0.2890, 0.3279 and 0.349 | `experiments/data-understanding/`; `lab/src/metrics.py` |
| LightGBM lambdarank fitted on validation only, 5-fold | helped then superseded | out-of-fold validation 0.365-0.367 | `experiments/gbdt-and-stacking/`; `lab/src/cv_val.py` |
| LightGBM, XGBoost and CatBoost trained on train rows | helped then superseded | about 0.368-0.370 on validation alone; nothing in blends with the neural ranker | `experiments/gbdt-and-stacking/`; `lab/src/train_xgb.py` |
| First three submissions: neural ensemble with a stacker and trees, then a residual stacker with a fine-tuned network | helped then superseded | validation 0.3816, 0.3839, 0.3869; public 0.3739, 0.3772, 0.3802; the third carried about 0.002 of leak on validation | `experiments/gbdt-and-stacking/`; `lab/src/stack2.py` |
| Residual XGBoost stacker on the neural score plus compact features | helped then superseded | clean stacker 0.3849 on validation; on the next neural generation -0.0001 ± 0.0006 | `experiments/gbdt-and-stacking/`; `lab/src/stack2.py` |
| Blends of tree predictions with the neural ensemble | no effect | best blend 0.3820 against 0.38186 for the network alone on validation; correlation 0.85-0.875 | `experiments/gbdt-and-stacking/`; `lab/src/train_lgb.py` |
| Neural ranker trained on validation only, from scratch | hurt | folds 0 and 1 about 0.352 and 0.366, against 0.365 and 0.372 when fine-tuned from the train-fitted model | `experiments/fine-tune/`; `lab/nn/finetune.py`, `lab/reports/nn_report.md` |
| Tree residual on the 20-member neural pool | hurt | trained on holdout feeds: validation -0.00031, holdout out-of-fold -0.00042 (z -2.3); nested on validation -0.00009 | `experiments/gbdt-and-stacking/`; `lab/nn_x/cb26/cb_resid.py` |
| Best tree member (CatBoost YetiRank on random-order feeds) added to the pool | no effect | member 0.3820 on validation at correlation 0.856 to the pool; +0.00015 ± 0.00024 at weight 0.05, negative from 0.1; break-even needs about 0.393 | `experiments/gbdt-and-stacking/`; `lab/nn_x/cb26/cb_member.py` |
| Recipe of the 2025 Avito ML Cup winners (linear item-item scorers, contact-ratio statistics, CatBoost YetiRank) | no effect | +0.00067 ± 0.00045 on validation over the team's CatBoost baseline, inside the one-seed tree spread of 0.00086; member 0.3827 against a bar of 0.3933 | `experiments/external-solutions/`; `lab/nn_x/w25/train_w25.py` |
| Tree score and tree-leaf embedding as a residual on the pool | hurt | -0.00264 ± 0.00064 (z -4.11) / -0.00136 (z -3.82); the input form was never trained | `experiments/error-analysis/`; `lab/nn_x/insights/ccb_1113/RESULT.md` |
| Audit of why trees lose to the network | no effect | a tree fitted to the pool's own score from 177 features stops at 0.3865 on validation (correlation 0.909); 13 tree models read pooled z -1.40 to +0.64 as members | `experiments/catboost-round-2/`; `lab/nn_x/w12/cb/nats.py` |
| CatBoost round 2: label statistics from all train feeds and native target encodings | no effect | best information gain +0.00147 ± 0.00043 / +0.00080 ± 0.00021 nats against a bar of 0.0016; pooled z at most +0.65 as a member | `experiments/catboost-round-2/`; `lab/nn_x/w12/cb/PREDECLARED.txt` |
| CatBoost round 2: tree students of the neural teacher, with raw embedding vectors | no effect | best student 0.38461 / 0.30797 against a gate of 0.3900; pooled z -1.05 to +0.06; 0.017 below the neural pool | `experiments/catboost-round-2/`; `lab/nn_x/w12/cb/PREDECLARED.txt` |
| CatBoost round 2: native embedding features, text bag-of-words, quantisation options | no effect | pooled z -1.42 to +0.67 as members | `experiments/catboost-round-2/`; `lab/nn_x/w12/cb/PREDECLARED.txt` |

### Neural architecture and capacity (38)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Set ranker with attention from each candidate to the user's history events | adopted | +0.0055 on validation over the same model without history tokens (0.37361 against 0.36808, one seed; the round report gives +0.006 to +0.010) | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Cosine-similarity bias inside the history attention | adopted | +0.0024 on validation; a later 2-epoch removal test read +0.0004 / +0.0005 | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Position tower (learned bias for the production display position) | no effect | 0.37361 with the tower against 0.37385 without it on validation, one 2-epoch run of round 1 on 11 November; loss proxy -0.00061 (z -2.1) | `experiments/neural-ranker-v1/`; `lab/nn/model.py`, `experiments/rounds-5-7/results/review_loss_zoo_report.txt` |
| Self-attention across the candidates of a feed | adopted | neutral in round 1 (0.37361 against 0.37386 without); removal from the final recipe costs -0.0020 / -0.0014 | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Score head in 32-bit floats (removes tied scores) | adopted | +0.001 to +0.003 on validation | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Wider or deeper ranker under click labels: width 256, 3 blocks, 8 heads | no effect | width 256: 0.37271 against 0.37385 on validation; 3 blocks at 10 epochs +0.0003 / -0.0004 | `experiments/neural-ranker-v1/`; `lab/nn/train.py`, the status log |
| Retrieval of the 16 history items most similar to each candidate | adopted | with the attention gate, 3 seeds: +0.00278 / +0.00382; removal from the final recipe -0.0034 / -0.0029 | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/attn/model.py` |
| Gated attention output (sigmoid gate per head) | adopted | mean +0.00065 on validation over 4 run pairs; alone +0.00042 (z 0.9) | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/attn/model.py` |
| Attention variants: forgetting bias; normalised query-key attention with scalable softmax | no effect | -0.00019 / -0.00014 and -0.00006 / -0.00010 | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/attn/model.py` |
| Mixture of experts in the feed-forward layers (8 experts, top 2) | no effect | +0.00027 ± 0.00047 on validation alone, with the last block's router sending almost every token to one expert; -0.00034 after the fine-tune on top of retrieval; +84% parameters | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/attn/model.py` |
| Piecewise-linear embeddings of numeric inputs | no effect | -0.00043 / +0.00078; after the fine-tune -0.00010 | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/feat` |
| Propensity head, semantic-ID codes for items, exponentially weighted history summaries | no effect | validation +0.00007, +0.00038 (z 0.8) and -0.00002 | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/feat` |
| Weight-sharing mini-ensemble inside one network (TabM, k = 4) | no effect | +0.00009 on validation over 3 seeds at 1.45× cost; sub-models 0.999 correlated | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/feat` |
| Attention bias with more terms (similarity scale per event type, same-category and recency terms) | no effect | within seed noise of the base at equal steps; stopped at 11k steps | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Domain embedding (production or random-order feed) and flags for the anonymous user bucket | adopted | neutral when introduced; removal of the domain embedding later read +0.0004 / 0.0000 | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Bidirectional GRU history encoder as a second model family | helped then superseded | member 0.39481 / 0.31885 against 0.39460 / 0.31911 for the transformer family; as students +0.00018 ± 0.00030 / -0.00016 ± 0.00016 | `experiments/hard-label-pools/`; `lab/nn/run_final10.sh` |
| DIN-style history encoder (MLP attention) as a model family | hurt | member 0.39097 / 0.31613 against 0.39667 / 0.31999 for the best family | `experiments/neural-ranker-v1/`; `lab/nn/run_final10.sh` |
| Width 192 with 6 heads under click labels, and its re-test with corrected learning rate and cooldown dropout | no effect | member 0.39371 / 0.31843; best re-test arm +0.00021 ± 0.00048 / -0.00010 against a bar of +0.0008 / +0.0004 | `experiments/training-waves/`; `lab/nn_x/w11/width/gap_probe.py` |
| Width 192 for students | adopted | 8 wide students: holdout +0.00107 ± 0.00031 over the narrow 4-pool; public 0.394486 against 0.394078 for 8 wide plus 8 narrow | `experiments/distillation/`; `lab/nn_x/w5/lane.sh` |
| Width 224 and width 256 for students | no effect | 224: +0.00026 ± 0.00058 / +0.00016 ± 0.00032, pooled z +0.68; 256 with dropout 0.2: -0.00163 ± 0.00058 / -0.00075 ± 0.00032, pooled z -3.64. Width 224 again on all train feeds (18 Nov), as files with 8 or 16 students replaced or added: validation +0.00005 to +0.00017 against FD16P20 (standard error 0.0003 to 0.0004), not scored. Paired by seed at the last checkpoint: +0.00031 ± 0.00057 / +0.00021; a post hoc re-read against the mean reference pool (+0.00092) counts the seeds' shared luck twice ([04](04-validation.md#seeds-shared-by-arm-and-reference)) | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/lane.sh`; the all-data width-224 runs were launched from a lane script that is not published |
| Other capacity knobs for students: three blocks, a second history layer, 4× feed-forward width, 6 heads, the richer similarity bias (`--sim_bias 2`: scale and prior per event type, same-category and recency terms) | no effect | 3 blocks: pool 0.40026 against 0.40075 on validation; 2 history layers +0.0002 / -0.0010; feed-forward ×4 -0.0011 / -0.0004; 6 heads pooled z +0.61; richer similarity bias pooled z -1.58 | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/lane.sh` |
| Routed feed-forward experts for students in the style of DeepSeek-V4 and Kimi K3: 8 experts of width 48, top 2, the dense layer kept as a shared expert, bias balancing by quantile (19 Nov) | no effect | 4-pool -0.00086 ± 0.00057 on validation (z -1.51), +0.00020 ± 0.00031 on the holdout, randomization p 0.875. The routed output was 0.07 to 0.21 of the dense layer's, and switching the experts off at inference cost -0.00010 ± 0.00014; in the last block 2 of 8 experts died on 3 of 4 seeds. The holdout had chosen a checkpoint of step 87,500 for one seed; at the last checkpoint, paired by seed: +0.00009 ± 0.00056 / +0.00026. Post hoc against the mean of the 70 reference 4-pools: +0.00070 ± 0.00046, which counts the seeds' shared luck twice. Extended to 8 seeds under a rule written first, last checkpoint against the 8 references: +0.00006 ± 0.00047 on validation, -0.00004 ± 0.00026 on the holdout | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/w12/moe/` |
| Routed value and output projections in the two cross-attention blocks, SwitchHead style: 4 experts, top 2 (19 Nov) | hurt | 4-pool -0.00144 ± 0.00057 on validation (z -2.51), +0.00029 ± 0.00032 on the holdout. At the last checkpoint, paired by seed: -0.00093 ± 0.00056 / +0.00028. Fit its own training feeds better (+0.0025 NDCG in sample) and moved away from the teacher, so the extra capacity went into the training clicks | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/w12/moe/` |
| Depth 7 blocks in place of 5 at about equal size, feed-forward ×1 (19 Nov) | no effect | 4-pool -0.00069 ± 0.00058 on validation (z -1.20), +0.00040 ± 0.00034 on the holdout. Swapped for 4 of the 16 wide students: -0.00004 ± 0.00028 on validation. At the last checkpoint, paired by seed: -0.00016 ± 0.00056 / +0.00042; post hoc against the 70-pool mean +0.00045 ± 0.00047, which counts the seeds' luck twice. At 8 seeds against the 8 references: -0.00001 ± 0.00048 on validation, +0.00027 ± 0.00026 on the holdout | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/w12/moe/` |
| Hashed item-ID embedding table | hurt | -0.0022 ± 0.0008 on validation; -0.0075 ± 0.0021 on feeds whose positives have unseen ids; for students -0.0057 / -0.0049 | `experiments/neural-ranker-v1/`, `experiments/distillation/` for students; `lab/nn/model.py` |
| Retrieval depth 32 in place of 16 | hurt | 10-epoch pairs -0.0002 / -0.0003; for students -0.00098 / -0.00043, pooled z -2.04 | `experiments/popularity-views/`, `experiments/distillation/` for students; `lab/nn_x/abl/abl_retr.py`, the experiment registry |
| Unique retrieval slots with counts per event type | no effect | 2-epoch screen +0.00117 on validation (z 2.48); 10-epoch pair +0.00036 ± 0.00068 / +0.00018, cold positives +0.0095 (z 7.1), warm -0.0034 (z -3.7), private-mix view -0.00108 and -0.00032 for the two seeds. The 10-epoch rule said proceed to a 6-seed family; the family's final read is not on record | `experiments/rounds-5-7/`; `experiments/rounds-5-7/results/round6_retrieval_slots_10epoch_pair.txt`, the experiment registry |
| Candidate-set variant of the self-attention over the feed's candidates (set attention); a second retrieval inside the candidate's category | no effect | set variant: 2-epoch holdout +0.00090 (z 2.7), at 10 epochs +0.00017 / -0.00001; the 10-epoch rule said proceed to a 6-seed family, and the family's final read is not on record. Category retrieval +0.00046 ± 0.00049 / 0.0 at 2 epochs, failed its rule | `experiments/popularity-views/`; `lab/nn_x/abl/abl_retr.py`, the experiment registry |
| Thirteen modern block designs at 2 epochs, the worst three: normalisation-free blocks, Peri-LN, feature-group tokens | hurt | -0.00124 / -0.00096; -0.00111 / -0.00064; -0.00114 on validation; 16.7 GPU-hours for the 29 runs of the round | `experiments/rounds-5-7/`; `lab/nn_x/r7/model.py` |
| The other block designs: DenseFormer weighted averaging, hyper-connections, differential attention, SwiGLU, RealMLP input encoder, post-LN | no effect | 0 of 13 promoted; nearest miss +0.00064 on validation with -0.00027 on the holdout; the others -0.00022 to -0.00107 on validation | `experiments/rounds-5-7/`; `lab/nn_x/r7/model.py` |
| Single-stream transformer over user, history and candidates together | no effect | 2-epoch screen: -0.00073 ± 0.00051 / -0.00027; diversity margin +0.0011 against a bar of +0.008; 1.84× step time | `experiments/training-waves/`; `lab/nn_x/w9/joint/model.py` |
| Category-factorised output with a separate category head | hurt | -0.00097 ± 0.00061 / -0.00120 ± 0.00034 (z -3.5) | `experiments/training-waves/`; `lab/nn_x/w9/catgrad/model.py` |
| History restricted to feed-surface events, or to target actions only | hurt | -0.00311 / -0.00266 and -0.00473 / -0.00493 | `experiments/training-waves/`; `lab/nn_x/w9/hs1/train.py` |
| Parallel feature-cross branch next to the transformer | no effect | +0.00001 ± 0.00050 / +0.00023 ± 0.00028; diversity margin +0.0001 | `experiments/training-waves/`; `lab/nn_x/w10/r4/model.py` |
| Expert networks or adapters per vertical | not run | CPU pre-gate: per-vertical function minus the global one -0.00095 ± 0.00039 (z -2.4); 99.2% of feeds span several verticals | `experiments/mixing-and-segments/`; the coverage map |
| Graph neural networks over the user-item graph | not run | a transductive graph has no node for 42.8% / 45.6% of public / private candidates; the inductive form was sized at +0.00004 to +0.00035 | `experiments/external-solutions/`; `lab/reports/LIT_1115/CRITIC.md` |
| Removing each component of the final click-label recipe in turn (2-epoch screen) | adopted | worth: retrieval and gate +0.0034 / +0.0029; 8 popularity inputs +0.0023 / +0.0074; auxiliary loss +0.0022 / +0.0023; set attention +0.0020 / +0.0014; per-feed loss weight +0.0015 / +0.0010 | `experiments/popularity-views/`; `lab/nn_x/abl/abl_arms.py` |
| Ideas from other competitions that were never trained: non-linear token MLP, event-field dropout, non-additive position embedding, cross-feed negatives | not run | priors of 0.01-0.02 each from the scan of outside solutions | `experiments/external-solutions/`; `lab/reports/EXTERNAL_SCAN_1113.md` |

### Optimiser, schedule and regularisation (33)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Muon optimiser on hidden weight matrices with a warmup-stable-decay schedule, in place of AdamW with cosine decay | adopted | 3 seeds: +0.00306 (z 4.6) / +0.00331 | `experiments/optimiser-and-modern-tricks/`; `lab/archive/nnx/optim/train.py` |
| Second training stage on random-order train feeds | helped then superseded | +0.0012 mean train-only validation in round 1; after the new schedule -0.00121 / -0.00147 | `experiments/neural-ranker-v1/`; `lab/nn/train.py` |
| Up-weighting random-order feeds during the main run | hurt | 0.37704 against 0.3780-0.3786 for four unweighted seeds on validation; a later plateau variant -0.00071 ± 0.00047 / +0.00002 | `experiments/training-waves/`; `lab/nn_x/w9/dw3/train.py` |
| Longer training: 2, 3, 5, 10 epochs | adopted | same seed: validation 0.3863, 0.3873, 0.3897, 0.3914; holdout 0.3117, 0.3128, 0.3148, 0.3167; with the final optimiser a 6-seed pool gains +0.0040 / +0.0031 | `experiments/neural-ranker-v1/`; `lab/nn/train.py` |
| Many 2-epoch seeds of one recipe as pool members | hurt | seed correlation 0.978; adding 12 of them lowered the holdout by 0.00112 ± 0.00024 | `experiments/rounds-5-7/`; `lab/nn/train.py` |
| 15, 20 and 30 epochs under click labels; 20 epochs for students | hurt | -0.0006 / -0.0002, -0.0008 / -0.0006, -0.0016 / -0.0007; students -0.0008 / -0.0004 | `experiments/training-waves/`, `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/logs/E_epochs_predeclared.txt` |
| Larger batches: 4× at 10 epochs, 2× at 2 epochs | hurt | 4×: -0.00262 / -0.00209, with doubled learning rate -0.00232 / -0.00163; 2×: -0.00145 / -0.00109 | `experiments/training-waves/`; `lab/nn_x/w5/logs/E_batch_predeclared.txt` |
| Cooldown with 30% random-order feeds at 10 epochs | hurt | all six members: validation about 0.391 to 0.364, holdout 0.317 to 0.289 | `experiments/neural-ranker-v1/`; `lab/nn/train.py` |
| Cooldown with 10% random-order feeds | helped then superseded | same seed at step 87.5k: +0.0157 on validation over the 30% run; still over-fits late in 19 of 21 runs (last step -0.0033 on validation) | `experiments/neural-ranker-v1/`; `lab/nn/train.py` |
| Cooldown with 2% random-order feeds, as a separate 2-epoch cooldown re-run from a plateau checkpoint | adopted | 7 runs: member +0.00117 ± 0.00039 (z 3.0) / +0.00134 ± 0.00022 (z 6.0); about 0.25 GPU-h per cooldown re-run | `experiments/rounds-5-7/`; `lab/nn_x/r6/cd_lane.sh` |
| Other shares: 0%, 1%, 4% under click labels, 5% for students | no effect | 0%: validation +0.00002, feeds with cold positives -0.0023 (z -4.5); 1%: -0.00035 / +0.00046; 4%: -0.00077 / -0.00046; 5%: 16-pool -0.00015 / +0.00006, pooled z -0.08 | `experiments/rounds-5-7/`; `lab/nn_x/r6/cd_lane.sh` |
| One-epoch cooldown continuation of members kept at a plateau checkpoint | helped then superseded | member +0.0018 (z 3.5) / +0.0015 | `experiments/rounds-5-7/`; `lab/nn_x/r6/cd_lane.sh` |
| Cooldown length: 40% of the run, a 3-epoch cooldown, 12 epochs in total | hurt | 40%: validation -0.0012 (z -2.3), cold positives z -5.4; the other two -0.0001 | `experiments/rounds-5-7/`; `lab/nn_x/r6/cd_lane.sh`, the status log |
| Cooldown batch size 0.25×, 0.5×, 2× | no effect | validation -0.0015, +0.0001, -0.0003 | `experiments/training-waves/`; `lab/nn_x/w5/cd_arms.py` |
| Auxiliary-loss weight in the cooldown (annealed to 0 or to 0.5, or 1.0), and 1.0 from scratch | no effect | validation: annealed to 0 -0.00049 ± 0.00018, to 0.5 -0.00007, 1.0 in the cooldown +0.00032 ± 0.00020 (z 1.6) per member over 4 runs; over 7 runs +0.00038 ± 0.00017 (z 2.2) / +0.00019 ± 0.00009 (z 2.0) per member and -0.00023 (z -1.5) on the holdout for the 7-member pool; from scratch +0.00049 ± 0.00071 / +0.00022 | `experiments/training-waves/`; `lab/nn_x/w5/cd_arms.py` |
| More weight decay: 0.1 in the cooldown; 0.05 and 0.10 for students | hurt | cooldown 0.1, a null arm: -0.00055 ± 0.00053 (z -1.04) / +0.00029; students at 0.05: member +0.00056 / +0.00008, 8-pool -0.00043 ± 0.00049 / -0.00063 ± 0.00027, pooled z -2.31; at 0.10: -0.00169 / -0.00091, pooled z -3.86 | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/lane.sh`, `experiments/rounds-5-7/results/round6_cooldown_stage1_2pct_and_weight_decay.txt` |
| Averaging the weights of plateau checkpoints | hurt | member +0.00041 ± 0.00040 / +0.00072; 7-member pool -0.0025 / -0.0016; seed correlation 0.975 against 0.941 | `experiments/rounds-5-7/`; `lab/nn_x/r6/recap_avg7.py` |
| Exponential moving average of weights during training | no effect | validation +0.00003 at decay 0.999, -0.00057 at 0.9998 | `experiments/optimiser-and-modern-tricks/`; `lab/archive/nnx/optim/train.py` |
| Plateau checkpoints (before the cooldown) as extra pool members | hurt | member 0.39218 / 0.31741, against 0.39460 on validation for the cooled members of the same runs | `experiments/neural-ranker-v1/`; `lab/nn/holdout_eval.py` |
| Learning rate 1e-3 or 2e-3 | adopted | 2e-3 is the final setting. At 10 epochs (7 against 6 seeds) it is ahead by +0.0008 (z 1.4) / +0.00067 (z 2.9, run-to-run SE); at 2 epochs it is behind by 0.0010 on validation | `experiments/optimiser-and-modern-tricks/`; `lab/nn/train.py`, `lab/reports/r6_recap.md` |
| The 5 input projections moved from Muon to AdamW | hurt | -0.00101 (z -1.79) / -0.00031, 2 seeds at 2 epochs; Muon stays on every 2-D matrix | `experiments/optimiser-and-modern-tricks/`; `lab/nn_x/r7/model.py`, `lab/reports/R7_ARCH_HPARAM.md` |
| More from-scratch members of the same recipe added to the pool | no effect | pool plus 4 new runs: +0.00003 ± 0.00020 / -0.00003 ± 0.00013 | `experiments/hard-label-pools/`; `lab/nn_x/picks/decide5.py` |
| Dropout: none on the plateau and 0.1 in the cooldown; 0 or 0.2 for students | no effect | late dropout +0.00006 ± 0.00071 / +0.00006; students with 0: -0.0006 / -0.0002; with 0.2: -0.00068 ± 0.00056 / -0.00046 ± 0.00032, pooled z -1.88 | `experiments/tricks-sweeps/`; `lab/nn_x/tricks/late_dropout/readout_late_dropout.py` |
| Weight decay below the default for students: 0 and 0.01 | hurt | 0: 8-pool -0.00134 ± 0.00056 / -0.00011, pooled z -1.95, member correlation 0.9563 against 0.9723 for the reference's seeds 951-958; 0.01: -0.00032 / -0.00021, pooled z -0.97 | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/lane.sh` |
| R-Drop and sharpness-aware minimisation in the cooldown | no effect | at the selected checkpoint R-Drop -0.00012 ± 0.00032 / +0.00059, SAM +0.00004 / +0.00041, pool holdout below 2 SE; at the last checkpoint R-Drop +0.00057 ± 0.00030 / +0.00052 ± 0.00017, SAM +0.00039 / +0.00038 and stale validation +0.00065 (z 2.04); correlation to the base 0.990 | `experiments/training-waves/`; `lab/nn_x/w5/cd_arms.py`, `experiments/training-waves/results/wave7_readout.txt` |
| Dropping history events at random during training | hurt | inference replay on two trained 10-epoch members, one pass at 10%: -0.0008 / -0.0006 (no full run); students trained at rate 0.3: -0.00098 ± 0.00058 / -0.00066 ± 0.00033, pooled z -2.62 | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/lane.sh`, `lab/reports/TRICKS_SWEEP_1113.md` |
| Uniform dropout of the extra numeric inputs | hurt | negative under the old recipe; the number is not on record | `experiments/neural-ranker-v1/`; `lab/nn/train.py` |
| Label smoothing or a calibrated anchor against memorising the random-order pool | no effect | within ±0.001 on an MLP proxy; smoothing 0.2 gave +0.0007 on one seed and did not prevent the collapse | `experiments/rounds-5-7/`; `lab/solvers/solvers/loss_zoo.py` |
| Cooldown on selected data: equal history length, recent days only, examples chosen by learnability, or without the first three train days | hurt | equal history length, a null arm: +0.00007 / +0.00002; recent days -0.0021 / -0.0012; learnability -0.0035 / -0.0020; without the first three days -0.00101 ± 0.00041 (z -2.5) / -0.00122 (z -4.8) | `experiments/training-waves/`; `lab/nn_x/w10/fpcd/launch_fpcd.sh` |
| Larger random-order pool from a detector that also covers short feeds | no effect | member validation +0.00025 ± 0.00035 at a 2% share, -0.00161 ± 0.00043 at 10% | `experiments/exploration-detection/`; `lab/nn_x/w12/xdet/setfeat.py` |
| Feed-normalised auxiliary loss on batches from the random-order pool | no effect | -0.00010 / -0.00012 | `experiments/training-waves/`; `lab/nn_x/w5/cd_arms.py` |
| Learning rate 4e-3 or 4 epochs in the students' 2-epoch cooldown re-run | no effect | 16-pools: 4e-3 -0.00062 ± 0.00032 / -0.00030 ± 0.00018 (randomization p 0.98); 4 epochs -0.00046 ± 0.00033 / +0.00043 ± 0.00018 (p 0.26) | `experiments/training-waves/`; `lab/nn_x/w12/gap1117/perm_read.py` |
| Never varied or never run: warmup, optimiser momentum, a learning rate above 2e-3 at the normal batch outside the students' cooldown, the cooldown floor, a 7-epoch schedule, two regularisers without code, 7 of 11 cooldown arms prepared for students | not run | no numbers; gradient clipping was checked as a diagnostic only: 0% of steps clipped after warmup in 26 runs. The cooldown fraction was varied once (40%, the cooldown-length row) and 4e-3 ran at 4× batch (the larger-batch row) | `experiments/training-waves/`; `lab/nn_x/w5/lane.sh` |

### Losses and auxiliary tasks (20)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Listwise softmax cross-entropy over the feed in place of pointwise binary cross-entropy | adopted | binary loss -0.001 on the real network (0.37272 against 0.37361 on validation); -0.00396 on the MLP proxy | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Equal loss weight per feed | adopted | +0.0020 / +0.0006 on the same seed; removal from the final recipe -0.0015 / -0.0010 | `experiments/neural-ranker-v1/`; `lab/nn/train.py` |
| Auxiliary head that regresses the production display position | helped then superseded | +0.002 on validation (0.37581 against 0.37385) | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Listwise auxiliary loss toward the production display order (temperature 10, weight 0.5 annealed to 0.1) | adopted | +0.0024 / +0.0027; removal from the final recipe -0.0022 / -0.0023; for students removal costs -0.00135 ± 0.00058 / -0.00128 ± 0.00033 | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Variants of that loss: temperature 30, top-20 Plackett-Luce target, weight 1.0 | hurt | 2 seeds at 2 epochs. Temperature 30: -0.00091 ± 0.00049 / -0.00109 ± 0.00028 (z -3.86); Plackett-Luce +0.00059 ± 0.00045 / +0.00021 and with weight 1.0 +0.00051 ± 0.00051 / +0.00026, both null | `experiments/rounds-5-7/`; `lab/nn_x/solvers`, `lab/reports/sol_gpu.md` |
| Sixteen ranking losses on a small proxy model, the ones that tied: label smoothing, focal, LambdaLoss, cross-entropy plus Lambda, Plackett-Luce over positives | no effect | validation, all within 1 SE of softmax cross-entropy: +0.00007 ± 0.00012, +0.00011, +0.00006, +0.00015, -0.00006 | `experiments/rounds-5-7/`; `lab/solvers/solvers/loss_zoo.py` |
| The same comparison, the losers: RankNet, LambdaRank, NeuralNDCG, ApproxNDCG, binary cross-entropy, ListMLE | hurt | validation: -0.00033, -0.00057, -0.00165 (z -2.5), -0.00295, -0.00396, -0.00516 (z -6.5) | `experiments/rounds-5-7/`; `lab/solvers/solvers/loss_zoo.py` |
| Pairing members trained with different losses for pool diversity | no effect | no loss paired with cross-entropy beats two cross-entropy seeds; best hint +0.00056 ± 0.00042 on validation with a weaker single model (holdout -0.00119) | `experiments/rounds-5-7/`; `lab/solvers/solvers/loss_zoo.py` |
| Member family trained with listwise plus binary cross-entropy | hurt | -0.00114 ± 0.00051 / -0.00047; diversity margin +0.0006 against a bar of +0.008 | `experiments/training-waves/`; `lab/nn_x/w10/r3` |
| The auxiliary head's own output as a pool member | no effect | its NDCG is 0.30355 against 0.39770 for the main head at correlation 0.03; the gap of 0.094 is twice the break-even of 0.047, so no blend was built | `experiments/rounds-5-7/`; `lab/solvers/CPU3_aux_head_member_W6.py` |
| Auxiliary head predicting the user's category intent | hurt | -0.0014 / +0.0003 | `experiments/neural-ranker-v1/`; `lab/nn/model.py` |
| Auxiliary task on the user's future activity window | not run | oracle probe: perfect knowledge gives +0.00275 ± 0.00068 on 15% of feeds and is 0.093 of a label echo; using the past share costs -0.0012 | `experiments/tricks-sweeps/`; `lab/nn_x/tricks/FUTWIN` |
| Examination offset on random-order train rows | no effect | +0.00002 ± 0.00028 / -0.00008 ± 0.00015 | `experiments/training-waves/`; `lab/nn_x/w10/r2/train.py` |
| Self-supervised next-item task on the history encoder inside the ranker | not run | trigger test at 0 GPU-h: partial correlation of the pool's residual with hidden-event recall -0.0012 ± 0.0139 against a bar of +0.05 | `experiments/training-waves/`; `lab/nn_x/w11/ssl/trigger` |
| Gradient-conflict surgery between the main and the auxiliary loss | not run | the two gradients are near-orthogonal (median cosine +0.007 to +0.014) | `experiments/error-analysis/`; `lab/nn_x/insights/ccb_1113/RESULT.md` |
| Auxiliary-loss weight 0 or 1.0 for students | hurt | weight 0, the removal of the adopted loss: -0.00135 ± 0.00058 / -0.00128 ± 0.00033, pooled z -4.38; weight 1.0, a null arm: -0.00023 ± 0.00055 / +0.00028 ± 0.00031, pooled z +0.34 | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/lane.sh` |
| Loss re-weighting aimed at the final model's failure map: extra weight 0.5 or 1.0 on the order inside categories, 0.5 on the choice of category, per-feed weight by feed size (M^0.25, M^0.5), as students' cooldown re-runs | no effect | 16-pools: +0.00005 / -0.00000, +0.00012 / -0.00007, -0.00036 / -0.00007, +0.00012 / -0.00008, +0.00036 ± 0.00034 / -0.00030 ± 0.00019; randomization p 0.48-0.94; none moved its targeted part of the error beyond the spread of two reference runs; M^0.5, the only arm level or ahead on every validation view, re-run on the 16 all-data students of FD16P20 against a matched control: -0.00002 ± 0.00027 on validation and -0.00057 ± 0.00040 on the private-mix view; public 0.39475033 against 0.39482063 for the control | `experiments/error-analysis/`; `lab/nn_x/w12/gap1117/mechanism_check.py` |
| Composite loss aimed at the hardest feeds: per-feed weight from the student's gap to the teacher, times a lower weight on clicks the model finds surprising, as students' cooldown re-runs | hurt | 16-pool -0.00058 ± 0.00047 / -0.00074 ± 0.00028, randomization p 0.98; a probe showed the hardest 20% of feeds hold 27.5% of the loss against 27.6% when labels are redrawn from the model's own softmax, so the biggest misses are the label draw | `experiments/error-analysis/`; `lab/nn_x/w5/cd_arms10.py` |
| Cascade correction: on production feeds the loss ignores negatives more than 5 positions below the last click | no effect | 16-pool +0.00004 ± 0.00027 / -0.00000 ± 0.00015, randomization p 0.45 | `experiments/error-analysis/`; `lab/nn_x/w5/cd_arms10.py` |
| Loss ideas never trained: truncated cross-entropy, masking later-interacted negatives | not run | later-interacted negatives are 0.402% of negatives and count as negatives in the validation and test labels too, so masking them was closed by an argument on 13 November; truncated cross-entropy had a prior of 0.02. Two ideas once on this list ran on 17 Nov: negatives below the last click ignored (the cascade correction above) and more weight on the order inside categories (the loss re-weighting row) | `experiments/tricks-sweeps/`; `lab/reports/TRICKS_SWEEP_1113.md`, `lab/reports/EXTERNAL_SCAN_1113.md` |

### Inputs and features (27)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Sequence-similarity and co-visitation feature family | adopted | +0.006 to +0.007 on validation in the tree harness (5 seeds and a time split) | `experiments/feature-families/`; `lab/feats_v2/seq_/build_keep.sh` |
| In-feed context family (near-duplicate candidates, neighbour statistics) | adopted | +0.002 on validation in the tree harness; as ranker inputs, removing the 10 near-duplicate inputs costs 0.0014 / 0.0003 (2-epoch screen); adding them with the 10 competition inputs read 0.0000 / +0.0014 and +0.0005 / +0.0017 in two paired runs | `experiments/feature-families/`; `lab/feats_v2/ctx_/build_ctx.py` |
| Cluster, calendar, trend and listing-lifecycle features | no effect | validation, tree harness: 6 cluster columns +0.00125 ± 0.00045 (8 seeds) and -0.00006 ± 0.00054 in another column order, kept by the harness and never a ranker input; the whole 122-column cluster family +0.0007 ± 0.0007; calendar -0.00023 ± 0.00067 while a noise control read +0.00071 | `experiments/feature-families/`; `lab/feats_v2/clu_` |
| Co-visitation and leave-user-out popularity as inputs of the train stage | adopted | +0.0032 / +0.0092 on the same seed | `experiments/feature-families/`; `lab/src/features_luo.py` |
| Candidate-competition inputs (reciprocal nearest neighbour, attention mass over recent history) | adopted | 10 of the final 34 inputs (`ctx_hn_*`, `ctx_hs_*`); +0.0004 / +0.0009 when added; a later removal test +0.0005 / 0.0, inconclusive | `experiments/feature-families/`; `lab/feats_v2/ctx_` |
| Item context across feeds (feed sizes it appears in, its usual production position) | no effect | first form +0.0010 / +0.0002; leak-safe rebuild +0.00012 ± 0.00072 / +0.00007 ± 0.00042; stale validation -0.00109 | `experiments/feature-families/`; `lab/feat_new/ixh.py` |
| Time-aware target encodings as fine-tune inputs | no effect | +0.0003 in the fine-tune branch on validation | `experiments/feature-families/`; `lab/src/features_tte.py` |
| Raw item features f1 to f5 | adopted | zeroing the family at inference costs -0.0177 on validation; f4 (listing age) alone -0.0138 | `experiments/popularity-views/`; `lab/nn_x/abl/f_replay.py` |
| Co-visitation for train rows built from a lagged source | no effect | +0.00032 ± 0.00071 / -0.00057 ± 0.00041; 43% of train feeds lose the source | `experiments/feature-families/`; `lab/feat_new/hzcv.py` |
| Drift of the user vector since the user's earlier snapshot | no effect | residual on the pool, validation: +0.00027 ± 0.00028 with trees, -0.00008 ± 0.00020 linear; train-fitted gate -0.00056 ± 0.00069 / +0.00034 ± 0.00042 | `experiments/feature-families/`; `lab/feat_new/ud.py` |
| Four more within-feed relative inputs | hurt | 10-epoch pairs -0.0019 / -0.0007; private-mix view -0.0032 | `experiments/popularity-views/`; `lab/nn_x/abl/abl_arms.py` |
| Weekday and hour of the feed | no effect | holdout weekend slice +0.00021 ± 0.00051; all feeds 0.0000; stale validation -0.00080 | `experiments/popularity-views/`; `lab/nn_x/abl/time/build_tm.py` |
| Time-gap features; item id as a posting clock; item-level weekend affinity | no effect | 0 of 6 pass, best +0.00030 ± 0.00027; Spearman of item id and listing age inside a feed 0.000 / 0.003 | `experiments/error-analysis/`; `lab/nn_x/insights/timegap_1113` |
| Audit of the input pipeline: scaling, clipping, re-encoding, aggregates, non-linear residual models (94 constructions) | no effect | 6 of 94 at z > 2 with 2.1 expected by chance, none on both halves; tree residual +0.00005 | `experiments/validation-and-leaks/`; `lab/nn_x/audit/dpa_1113` |
| Batch of 16 ideas from other competitions (decayed-recency counts, co-visitation weightings, item strength models, user aggregates and others) | no effect | 0 of 16 pass; best +0.00031 ± 0.00038 on validation with -0.00024 on the holdout | `experiments/error-analysis/`; `lab/nn_x/insights/ccb_1113/RESULT.md` |
| Negative feedback from items the user was shown earlier and skipped | no effect | skipped items have a within-feed AUC of 0.58, so they mark interest; nested -0.0006 to +0.0001 | `experiments/external-solutions/`; `lab/reports/EXTERNAL_SCAN_1113.md` |
| Further probes from the scan of outside solutions (IDF damping, typed co-visitation, category sequences, seller slots, earlier errors, user-relative profile) | no effect | at most +0.00024 ± 0.00025; the user-relative profile reads +0.00325 on a features-only base and +0.00003 once popularity is present | `experiments/external-solutions/`; `lab/reports/EXTERNAL_SCAN_1113.md` |
| Item-level prior from the labels of random-order feeds | no effect | nested -0.00011 from validation, +0.00018 from train, union +0.00042 (z 1.4); 16% row coverage | `experiments/validation-and-leaks/`; `lab/nn_x/audit/itemprior` |
| Automatic feature engineering on top of the student pool (11,617 generated features, 19 nested blocks) | no effect | 1.02% of features pass against 1.01% of label-free noise features; best nested block +0.00006 ± 0.00016; a planted +0.0009 interaction was recovered | `experiments/trained-input-search/`; `lab/nn_x/w12/autofe/openfe_run.py` |
| Standardising inputs at test time with validation and test statistics | hurt | -0.0024 to -0.0057 on pool members | `experiments/trained-input-search/`; `lab/nn_x/w12/autofe` |
| Two-step co-visitation walk over the whole history as a trained input | no effect | pooled z +0.59 under click labels and +0.62 for students (bar 2.0); file validation +0.00011 ± 0.00038; public 0.3943559, which is -0.00054 ± 0.00050; natural minus permuted +0.00179 | `experiments/trained-input-search/`; `lab/nn_x/w12/nx3/nx3_build.py` |
| Second input search: neighbour density, counts by search surface, click strength, six new walk constructions | no effect | the trio reads +0.0004 ± 0.0004 on validation and -0.0002 ± 0.0003 on the lagged holdout; the search-surface count is worth about +0.0001 | `experiments/trained-input-search/`; `lab/reports/INPUTS_1116/round2` |
| Click-strength input repaired for exposure and train/test parity | no effect | conditional test +1.22 to +2.17 against bars of 2.50-2.74; NDCG at or below 0 on validation | `experiments/trained-input-search/`; `lab/reports/NEXT_1116/clickstrength/c01_build.py` |
| Fresh exposure counts from other users' validation and test feeds | no effect | nested -0.00004 ± 0.00020; in-sample ceiling +0.00017; click lift given the pool 1.008 ± 0.021 where about 1.35 would be needed | `experiments/mixing-and-segments/`; `lab/nn_x/w12/fresh/pregate` |
| Longer user history rebuilt from the same user's earlier snapshots | no effect | replay without training: +0.00117 ± 0.00082 (z 1.43) and +0.00151 ± 0.00093 (z 1.62) on the holdout; trained as up to 128 older events for feeds at the 256-event cap, 16 cooldown re-runs: +0.00150 ± 0.00117 / +0.00001 ± 0.00058 on the 1,132 / 2,159 feeds it reaches (randomization p 0.16), whole set -0.00013 / -0.00005 | `experiments/mixing-and-segments/`; `lab/nn_x/w12/gap1117/lifelong/read_hx.py` |
| History events whose item vector is missing, filled at prediction time from earlier splits' item tables | inconclusive | 14% of history events are missing a vector and a quarter of those are recovered; replay of the 20 final members: train-only half +0.00021 ± 0.00024 (z 0.84), final file with the stored fine-tuned half -0.00012; train rows cannot be filled the same way | `experiments/mixing-and-segments/`; `lab/nn_x/w12/hfill/r_read.py` |
| Never built: tree-leaf embeddings as a ranker input, content-completed co-visitation for cold items, members trained on user halves | not run | priors of 0.015, or an expected +0.00004 to +0.00035 | `experiments/external-solutions/`; the coverage map |

### Leak-safe statistics and leaks found (12)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Boosting candidates that appear in the same user's later history | no effect | public 0.3802 with and without the boost; as an overlay on the best pool: validation +0.00454 ± 0.00047, public 0.392819 against 0.392820 | `experiments/validation-and-leaks/`; `lab/src/later_overlay.py` |
| Popularity counts without the same user's later snapshots | adopted | trees on the uncorrected counts gain +0.025 to +0.052 on the 117 validation feeds whose user has a later feed; the corrected version shows no jump | `experiments/validation-and-leaks/`; `lab/src/features_luo.py` |
| ALS item factors trained on all histories | helped then superseded | single-feature NDCG on validation: leaky version 0.583 on later-user feeds against 0.329 elsewhere; clean rebuild 0.318 against 0.326 | `experiments/validation-and-leaks/`; `lab/src/als_clean_pairs.py` |
| Leak probe on the feeds whose user has a later feed, run on every feature and every file | adopted | the other 589 features show at most about 0.03 excess; threshold set to z > 3 after a false alarm at z 2.0 on 117 feeds | `experiments/validation-and-leaks/`; `lab/src/presub_gate.py` |
| Encoder scores for train rows from time-windowed encoders, first build | hurt | holdout +0.006 to +0.0085, validation -0.005 to -0.009 | `experiments/pretraining/`; `lab/pretrain/assemble_train.py` |
| Rule: every train row is built like a validation or test row, with a stability check (PSI < 0.5) for new statistics | adopted | no number | `experiments/validation-and-leaks/`; the status log |
| Each feed of the anonymous user bucket as its own group in cross-validation | adopted | part of the adopted fine-tune, which passed non-inferiority at +0.00024 ± 0.00029 | `experiments/neural-ranker-v1/`; `lab/nn/finetune.py` |
| Validation labels as an item prior for test | no effect | 12.7% of test candidates were validation candidates; the lift exists only for validation-positive items, 0.2% of rows (3.1% against 1.6% positive) | `experiments/validation-and-leaks/`; the status log of 11 November |
| Forward echo of feature f5 (other users' later value minus the current one) | no effect | leak class; nested on the best click-label pool +0.00038 ± 0.00024 (z 1.6), and +0.00022 ± 0.00027 within validation in `p09_f5echo.py`; usable test coverage 4.5-5.1%. Re-read on pick 1 on 18 Nov: +0.00457 ± 0.00070 nats per click but -0.00015 ± 0.00038 NDCG, because the information is that an item whose f5 stayed flat was not clicked, at the bottom of the lists | `experiments/external-solutions/`; `lab/reports/EXTERNAL_SCAN_1113.md`, `lab/nn_x/w12/forensics/p09_f5echo.py`, `lab/nn_x/w12/leader/data_channels/p2_read.py` |
| Privileged teacher that sees the user's later snapshots | no effect | the teacher gains +0.037 ± 0.0025 on covered feeds, all from events after the first candidate interaction; earlier events -0.0003 ± 0.0004 | `experiments/distillation/`; `lab/nn_x/w12/hindsight/01_build.py` |
| Item statistic "share of opens that come from the feed", built from all histories | no effect | +0.00033 ± 0.00042 on validation; -0.00001 ± 0.00030 when rebuilt from strictly earlier days; -0.00073 ± 0.00024 on the holdout | `experiments/codes-and-embeddings/`; `lab/nn_x/codes/exploit` |
| Encoder input "id contribution", built differently on validation and test | hurt | banned from every input set; no number | `experiments/validation-and-leaks/`; `lab/reports/REVIEW_SYNTHESIS.md` |

### Popularity, staleness and display-order bias (30)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Detector of random-order feeds inside train, and a pool of 164K such feeds | adopted | about 206K of 1.55M train feeds with 50+ candidates are random-order; the target rate at rank 0 is 2.5% there against 9.1% in production feeds | `experiments/exploration-detection/`; `lab/eda/labels_position/04b_explore_detect.py` |
| The 8 popularity inputs | adopted | removal costs -0.0023 / -0.0074; with stale inputs they are worth only +0.0009 | `experiments/popularity-views/`; `lab/src/features_luo.py` |
| Measurement of popularity staleness across splits | adopted | warm items: 1.75 d on the holdout, 5.69 d on validation, 9.34 d on public, 11.23 d on private; with private-like stale inputs 8 full-view students lose 0.0033 ± 0.0008 on the lagged holdout and 0.0045 ± 0.0007 under the causal stale view of arm `hz2` | `experiments/popularity-staleness/`; `lab/nn_x/w12/next/poplag` |
| Pool members with all 8 popularity inputs removed | no effect | member -0.00251 ± 0.00081 / -0.00807; pool plus 4 such members +0.00036 ± 0.00030 / -0.00052 ± 0.00021; public +0.000066 | `experiments/popularity-views/`; `lab/nn_x/abl/popfree/build_pool.py` |
| Popularity-light members under click labels (one clipped exposure count in place of the 8 inputs) | helped then superseded | pool plus 4: +0.00029 ± 0.00031 / -0.00002, public +0.000385; pool plus 8 with 3-seed fine-tune: public 0.393454 against 0.392820 | `experiments/popularity-views/`; `lab/nn_x/abl/pop1/build_xp_imp_c50.py` |
| Popularity inputs replaced by one click-rate ratio | no effect | pool plus 4: +0.00015 / -0.00023; correlation 0.9863 with the exposure-count view | `experiments/popularity-views/`; `lab/nn_x/abl/pop2` |
| Admixture of popularity-light students at weight 0.2 | adopted | validation +0.00049 ± 0.00031; public +0.000523 at a paired SE of 0.00041; holdout -0.00063 ± 0.00022 | `experiments/distillation/`; `lab/nn_x/w12/kd/build_r3.py` |
| Other weights for the popularity-light part: 0.5, or 0.3 and more | no effect | 0.5: 0.40106 against 0.40095 on validation, bar +0.0005; lagged holdout on its 13,424 valid feeds at 0.1, 0.2, 0.3: +0.00013 ± 0.00021, +0.00024 ± 0.00031, -0.00015 ± 0.00038 (on all 30,000 feeds, where its check fails: +0.00017, +0.00020, +0.00007) | `experiments/popularity-staleness/`; `lab/nn_x/w12/next/poplag` |
| Feed-size rule: popularity-light weight 0.4 on feeds with 50+ candidates, 0.2 on short feeds | no effect | in-sample on validation +0.00026 to +0.00071; nested +0.00002 ± 0.00027; public -0.00015 (-0.5 SE) | `experiments/mixing-and-segments/`; `lab/nn_x/w12/mix/rulesA` |
| The other side: popularity-light weight 0.4 on feeds with fewer than 50 candidates, 0.2 on longer feeds, on the members of FD16P20 | adopted | validation +0.00022 ± 0.00024, public-mix view +0.00027, private-mix view +0.00033; public 0.3952092 against 0.394892, +0.00032 (about 1 paired SE) | `experiments/popularity-views/`; `lab/nn_x/w12/kd/build_r4.py` |
| Other short-feed weights under that rule: 0.3, 0.5, 0.6, 0.8 | no effect | validation 0.40266, 0.40251, 0.40232, 0.40184 against 0.40293 at 0.4 and 0.40271 at 0.2; against 0.4: -0.00027 ± 0.00017, -0.00042 ± 0.00016, -0.00061, -0.00109 (z -3.2); 0.4 is a single high point | `experiments/popularity-views/`; the decision ledger |
| Lower long-feed weights under that rule: 0.1 or 0 | hurt | validation against pick 1 (FD16P20@m0.4_0.2) -0.00025 ± 0.00013 (z -1.94) and -0.00068 (z -3.45); not built | `experiments/popularity-views/`; the decision ledger |
| Popularity-light part of the fine-tuned branch taken from the popularity-light students' train-only predictions on feeds of 50+ candidates (lab name A1) | no effect | against pick 1: validation +0.00017 ± 0.00014, public-mix view +0.00029, private-mix view +0.00040; public 0.395036, -0.00017 at a paired SE of about 0.00018; the second change on long feeds to land about 0.0004 below its validation reading | `experiments/popularity-views/`; `lab/nn_x/w12/fixmiss/build_a1.py` |
| Search over gated mixing rules and learned gates (95 rule families, 131,547 rules, 7 gate families) | no effect | 0 of 95 families and no learned gate beat the fixed weight nested on more than one member set | `experiments/mixing-and-segments/`; `lab/nn_x/w12/mix/gateB` |
| More popularity-light students (8 or 12 where the file has 4), or narrow ones added | no effect | validation: 8 students -0.00021 ± 0.00017; 12 students -0.00016 ± 0.00018; narrow ones added +0.00001 ± 0.00018; under the short-feed rule of the first final file, 8 and 12 students -0.00057 (z -2.75) and -0.00039 (z -1.86) against it | `experiments/full-data-retrains/`; `lab/nn_x/w12/stale/fulldata/lane_full_p2.sh` |
| A second view inside the popularity-light part (popularity-free or click-rate students) | no effect | first draw +0.00036 ± 0.00019 on validation against its matched control file, above 70 of 70 control pools; against FD16P20 the first draw reads +0.00015 ± 0.00020 and the replication -0.00024 ± 0.00018 (above 16 of 70); public 0.394630, which is -0.00026 (-1.0 SE) | `experiments/full-data-retrains/`; `lab/nn_x/w12/stale/fulldata/lane_full_v.sh` |
| Input dropout conditioned on listing age | inconclusive | stale validation +0.00089 ± 0.00062, validation +0.00005, holdout -0.00189 ± 0.00034; +0.0247 on feeds whose positives are all young, -0.0063 elsewhere | `experiments/rounds-5-7/`; `lab/nn_x/solvers` |
| Dropout of the popularity inputs in the cooldown (rate 0.30), students | no effect | validation -0.00066 ± 0.00048; stale validation +0.00001 ± 0.00050 | `experiments/popularity-staleness/`; `lab/nn_x/w12/kd/hz_read.py` |
| Cooldown on a stale view of the popularity inputs | inconclusive | 16-pool: validation +0.00024 ± 0.00038, stale validation +0.00120 ± 0.00045; as an all-data file of 16 members (validation 0.40224, built and gated on 2026-11-17) -0.00047 ± 0.00035 on validation and, scored on 2026-11-19, public 0.3949761, +0.00008 against FD16P20 at a paired SE of 0.00045 | `experiments/popularity-staleness/`; `lab/nn_x/w12/kd/hz_read.py` |
| Full-dose stale view of all 8 inputs, and variants that blank the encoder's ID score on rows made cold | no effect | full dose is not better than half dose (+0.00017, z 0.4) at a fresh-input cost of -0.0028; variants +0.00046 ± 0.00048 and -0.00017 ± 0.00023 | `experiments/popularity-staleness/`; `lab/nn_x/w12/stale/hz2n` |
| Popularity-light members inside the teacher, alone or with the stale view | no effect | validation +0.00042 ± 0.00032, stale validation +0.00008 ± 0.00030; with the stale view -0.00035 ± 0.00029 on validation | `experiments/popularity-staleness/`; `lab/nn_x/w12/kd/hz_read.py` |
| Fine-tune on validation with test-like stale inputs | no effect | 8-member pool: validation +0.00022 ± 0.00026, stale validation +0.00054 ± 0.00028 (z 1.93), and -0.00006 ± 0.00044 for stale validation at the private mix; averaged over 70 four-member subsets, stale validation +0.00017 ± 0.00014 | `experiments/fine-tune/`; `lab/nn_x/w12/stale/staleft` |
| Equalising item-occurrence inputs to the test lag at inference | no effect | file validation -0.00024 ± 0.00017, private-mix view -0.00013 ± 0.00027; +0.00120 on feeds without a cold positive, -0.00254 on feeds whose positives are all cold | `experiments/popularity-staleness/`; `lab/nn_x/w12/eqb_1115/eqb_pred.py` |
| One-parameter output offset for warm items whose counts are deflated by the lag | no effect | nested -0.00012 ± 0.00031, although the pools do under-score such rows (coefficient +0.20 ± 0.03) | `experiments/tricks-sweeps/`; `lab/reports/TRICKS_1115/web_shift.md` |
| Running the model with the production-order vector of the domain embedding at inference | no effect | click-label members +0.00011 ± 0.00017 (cold positives +0.0026, the rest -0.0007); student pool -0.00049 ± 0.00039 | `experiments/tricks-sweeps/`; `lab/nn_x/w12/weak_1115/domtok_read.py` |
| Offsets, gates and routers that treat cold-item feeds differently | no effect | the oracle of +0.0257 is a label echo; gate AUC 0.65-0.67 but value-weighted 0.52-0.53; nested -0.00005 ± 0.00004 | `experiments/error-analysis/`; `lab/solvers/errors` |
| Age-gated, density-matched popularity for train rows | no effect | +0.041 on feeds whose positives are all young; holdout -0.0010 to -0.0015 in every run | `experiments/feature-families/`; `lab/feat_new/it_hist_hz.py` |
| Training-only offset for the truncation of production feeds after the last interaction | no effect | member -0.0005 ± 0.0005 / -0.0001; pool -0.0003 / -0.0009; the tree proxy had shown +0.0022 to +0.0031 | `experiments/training-waves/`; `lab/nn_x/w5/train.py` |
| Training-only listing-age margin on production rows | hurt | -0.0046 / -0.0034 on the CPU proxy | `experiments/validation-and-leaks/`; `lab/nn_x/audit/ageprod` |
| Within-category production order as a cleaner auxiliary teacher | not run | probe: the same-category excess is equal in random-order feeds (+0.0145 against +0.0130) | `experiments/external-solutions/`; the coverage map |

### Fine-tune on validation (18)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Fine-tune of the train-fitted ranker on validation, 5 folds by user, with a branch of inputs that exist only for validation and test | adopted | round 1: 0.37848 to 0.38068 on validation; a larger set of extras +0.0037; at the end about 0 on validation and +0.0005 on public at pool level | `experiments/neural-ranker-v1/`; `lab/nn/finetune.py` |
| Averaging the predictions of all fine-tune epochs | helped then superseded | +0.0015 to +0.002 over the best single epoch | `experiments/neural-ranker-v1/`; `lab/nn/finetune.py` |
| Cosine decay with an exponential weight average (4 epochs) | adopted | +0.00112 out-of-fold (SE 0.00038) on members of the new optimiser; +0.00025 on older ones | `experiments/neural-ranker-v1/`; `lab/nn/finetune.py` |
| Cleaned set of 42 extra fine-tune inputs, anonymous-user feeds as singleton folds | adopted | non-inferiority: ensemble -0.00022 ± 0.00042, family mix +0.00024 ± 0.00029 | `experiments/neural-ranker-v1/`; `lab/nn/build_w3b_feats.py` |
| Final score = 0.5 fine-tuned + 0.5 train-only | adopted | public, one pool: train-only alone 0.394009, half and half 0.394486, fine-tuned alone 0.393960 | `experiments/distillation/`; `lab/nn_x/w12/kd/build_r4.py` |
| Other fine-tune weights: 0.67, 0.60, dependent on feed size | no effect | on one 6-member family a nested choice picks 0.5 in 5 of 5 folds and 0.67 reads -0.0003 ± 0.0002; on 19 members all 5 folds chose 0.67 and the nested gain was +0.00024 ± 0.00025; 0.60 file -0.00011 ± 0.00016 | `experiments/rounds-5-7/`; `lab/solvers/r5_wft.py` |
| Refit of the fine-tune on all of validation for the test predictions | no effect | -0.00025 ± 0.00018 in a nested simulation | `experiments/rounds-5-7/`; `lab/solvers/r5_refit_eval.py` |
| Three fine-tune seeds per fold | adopted | in the final recipe; member +0.00000 ± 0.00004; pool +0.00018 ± 0.00012; public -0.00007 for the paired files | `experiments/fine-tune/`; `lab/nn_x/abl/ftbag/build_pool_s3.py` |
| Automatic re-tuning of the fine-tune hyper-parameters (35 trials, tune and confirm halves by user) | no effect | confirm half +0.00055 ± 0.00041 (z 1.3); second member set -0.00009 | `experiments/fine-tune/`; `lab/nn_x/ftopt/driver.py` |
| Linear probe before the fine-tune, sharpness-aware fine-tune, per-feed loss normalisation | no effect | -0.00014; three normalisation variants within ±0.0002 (0.38097, 0.38088, 0.38113) | `experiments/neural-ranker-v1/`; `lab/reports/nn_report.md`, `lab/reports/nnx_VERDICT.md` |
| Pseudo-labels on test feeds inside the fine-tune | no effect | out-of-fold 0.38535 against 0.38580 (-0.0004) | `experiments/fine-tune/`; `lab/reports/nn_report.md` |
| 12,000 random-order train feeds mixed into every fine-tune epoch | no effect | first generation +0.0005 (7 of 8 pairs); second generation +0.0003, +0.0001, -0.0009 | `experiments/exploration-detection/`; `lab/reports/nn_report.md` |
| Shorter fine-tune or nested epoch selection | no effect | the first epoch is the peak in 37 of 52 runs, but the mix moves by ±0.0004 only | `experiments/fine-tune/`; `lab/reports/REVIEW_SYNTHESIS.md`, `lab/reports/r6_recap.md` |
| Validation labels replayed inside the cooldown, out-of-fold by user | no effect | +0.00014 against the fine-tuned pool; +0.0009 against the plain cooldown, which is the fine-tune's own value | `experiments/training-waves/`; `lab/nn_x/w5/train.py`, `lab/nn_x/w5/logs/w5_eval_valcd.txt` |
| Fine-tune on the teacher's soft targets | no effect | two CPU pilots: -0.00036 ± 0.00031 and +0.00002 ± 0.00044 in the final file | `experiments/distillation/`; `lab/reports/RECHECK_1114/ft_redesign.md` |
| Redesigned fine-tune for students (shrinkage, anchor, low-dimensional update, group-dependent weight) | not run | bound: best case +0.00005 (95% upper +0.0005); the fine-tuned half is one shared shift, 82% common mode | `experiments/distillation/`; `lab/reports/RECHECK_1114/ft_redesign.md` |
| Fine-tune with feed weights matched to the private mix | no effect | file: private-mix view +0.00022 ± 0.00035, validation -0.00007 ± 0.00022 | `experiments/tricks-sweeps/`; `lab/reports/TRICKS_1115/launch/finetune_mixft.py` |
| Fine-tune batch size 0.5× or 2×, and 3× learning rate on the extra branch | not run | launch kit built; closed in review because the dose these knobs vary had been read flat three ways | `experiments/trained-input-search/`; `lab/reports/NEXT_1116/fillers/ftarm_launch.sh` |

### Distillation (18)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Self-distillation from one earlier model, day 1 | no effect | validation +0.00064 (z 1.6); +0.00018 inside the new-optimiser combination | `experiments/optimiser-and-modern-tricks/`; `lab/archive/nnx/optim/train.py` |
| Ensemble distillation: target = 0.5 clicks + 0.5 softmax of the mean score of 11 train-only members | adopted | member +0.00257 ± 0.00047 / +0.00200 ± 0.00028 over 4 seeds; public +0.001032 for the first distilled file over the best click-label file | `experiments/distillation/`; `lab/nn_x/w12/kd/dump_teacher.py` |
| Teacher weight 0.3, 0.8 or 1.0 at width 128 | no effect | 4-pools 0.39977 / 0.32342, 0.40004 / 0.32377, 0.39906 / 0.32291 against 0.40085 / 0.32349 at 0.5 | `experiments/distillation/`; `lab/nn_x/w5/lane.sh` |
| Teacher weight 0.8 for wide students, 8 seeds | hurt | member +0.00126 ± 0.00033 / +0.00078 ± 0.00019; 8-pool -0.00049 ± 0.00048 / -0.00029 ± 0.00026, pooled z -1.51 (-2.36 against same-size reference pools) | `experiments/distillation/`; `lab/nn_x/w5/lane.sh` |
| Teacher weight 0.3 in the cooldown only | no effect | 16-pool +0.00011 ± 0.00032 / +0.00009 ± 0.00018, randomization p 0.31 | `experiments/distillation/`; `lab/nn_x/w12/gap1117/perm_read.py` |
| Teacher weight 0.8 in the cooldown only | hurt | member +0.00061 on both witnesses; 16-pool -0.00047 (z -1.21) / -0.00059 (z -2.86), pooled z -2.88 | `experiments/distillation/`; `lab/nn_x/w12/kd/fill.sh` |
| Teacher form: 25 members, a mixed teacher, temperature 0.7 or 1.5 | no effect | 25 members: 4-pool 0.40095 / 0.32317 against 0.40085 / 0.32349 at width 128, and 0.40100 / 0.32412 against 0.40134 / 0.32408 at width 192; its students are 8 of the 16 final full-view students. Mixed: -0.00030 ± 0.00057 / +0.00013 ± 0.00032. The width-192 and mixed reads are paired by seed with a reference whose seeds' luck another teacher does not share, so they may read up to about 0.0006 too low ([04](04-validation.md#seeds-shared-by-arm-and-reference)). Temperature pooled z -0.74 and -0.61 | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w12/kd/dump_teacher.py` |
| Students of students: a teacher made of distilled students | no effect | 4-pools 0.40080 / 0.32410 and 0.40093 / 0.32393 against 0.40134 / 0.32408, the seed-paired reference, whose seeds' luck an arm with another teacher may not share; against the mean of the 70 reference pools at the last checkpoint -0.00013 and -0.00023 on validation | `experiments/distillation/`; `lab/nn_x/w12/kd/dump_teacher.py` |
| Cross-fitted teacher (each train feed scored by teachers that did not train on it; 5 folds by user, 3 fold teachers each) | hurt | pools 0.40079 / 0.32347 (weight 0.5) and 0.40008 / 0.32321 (weight 0.8) against the seed-paired 0.40134 / 0.32408. Against the mean of the 70 reference pools at the last checkpoint: -0.00015 ± 0.00047 / -0.00059 ± 0.00026 and -0.00096 / -0.00084. Little to gain: the teacher's softmax is nearly flat, and the in-sample copy of the label is 0.2 to 0.7% of the target mass | `experiments/distillation/`; `lab/nn_x/w12/kd/run_oof.sh` |
| Students trained on random sub-lists of each feed (function matching) | no effect | pooled z -0.96 and -0.99 over 2 seeds; the sub-lists kept every positive, 1.88 per list against 1.22 on real short feeds, and reused the in-feed context inputs computed on the full feed; the clean form was never run | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/train.py` |
| More students of the same recipe (16, 24, 32), and mixes of narrow and wide students | no effect | 16 against 8: -0.00007 ± 0.00034 / +0.00004 ± 0.00019, public -0.000003; wide plus narrow: public -0.00041 (0.95 SE) | `experiments/distillation/`; `lab/nn_x/w12/kd/build_r3.py` |
| Click-label members added to the student pool | no effect | the old click-label pool plus 4 students at teacher weight 0.5 or 0.8: -0.00004 and -0.00007 on validation; 0 of 52 families adds at 2 SE on the holdout | `experiments/distillation/`; `lab/nn_x/w12/kd/build_k.py`, the decision ledger |
| Students with a GRU history encoder | no effect | +0.00018 ± 0.00030 / -0.00016 ± 0.00016 against the same number of extra plain students | `experiments/distillation/` ([student recipe arms](../experiments/distillation/README.md#student-recipe-arms-14-to-16-november)); `lab/nn_x/w5/lane.sh` |
| Teacher-labelled feeds of a held-out user fold (a step toward using unlabelled test feeds) | inconclusive | 4 seeds, fold kept with teacher targets minus fold dropped: pool +0.00069 ± 0.00057 / -0.00017 ± 0.00032; member +0.00103 / +0.00052 | `experiments/popularity-staleness/`; `lab/nn_x/w12/next/u20wit` |
| Pseudo-feeds built from user histories as extra training lists | adopted | as click-label members (11-14): member +0.00108 ± 0.00072 / +0.00024; diversity margin +0.0024 against +0.005. Inside the distilled students (11-17): 2-seed member screen +0.00092 ± 0.00060; 8-seed pool +0.00054 ± 0.00047 / +0.00010 ± 0.00027, randomization p 0.19, rule failed. All-data files against FD16P20: 8 of 16 full-view students replaced, validation +0.00028 ± 0.00029, public -0.00010; all 16 replaced (FDX16P20:ps, pick 2 from 11-19), validation +0.00015 ± 0.00036, public +0.00037 at a paired SE of 0.00047 | `experiments/distillation/`; `lab/nn_x/w11/pseudo/train.py`, `lab/nn_x/w12/pseudokd/train_full_pk.py` |
| Teacher with a privileged item click-rate input; label-conditional co-click targets; members trained on user bags | no effect | privileged click rate +0.0001; bagging -0.0011 / -0.0018; no number on record for the co-click targets | `experiments/distillation/`; the coverage map |
| Teacher-only targets on train feeds without a click; distillation on later unlabelled feeds | not run | 0 of 3,151,208 train feeds lack a positive. Distillation on later unlabelled feeds was declared on 18 Nov (a screen of about 16 GPU-h, rule: 8-pool validation at least +2 SE) and not run. A teacher weight per feed ran as the composite loss of the losses theme | `experiments/distillation/`; `lab/reports/TOP1_GAP_EXTERNAL_1114.md`, the decision ledger |
| Audit of the distillation path | adopted | no bug and no leak; every headline number reproduced in three independent recomputations; the first 8-student file is a favourable draw (pooled z +1.60 against the second 8-pool) | `experiments/distillation/`; `lab/nn_x/w12/recheck/synthesis/aa_null.py` |

### Ensembling and pool composition (22)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Member-equal mean of within-feed z-scores as the only combiner | adopted | ties the raw-logit mean and the softmax-probability mean on validation: +0.00004 ± 0.00015 and +0.00005 ± 0.00021 | `experiments/hard-label-pools/`; `lab/src/stack_val.py` |
| Rank-based fusion (rank mean, reciprocal rank fusion, Gaussianised ranks) | hurt | validation -0.0020 ± 0.00035, -0.0007 ± 0.0003, -0.0008 ± 0.0003 | `experiments/hard-label-pools/`; `lab/solvers/ensemble/ens_common.py` |
| Stackers over member scores (three LightGBM forms, listwise linear, greedy selection) | no effect | validation: -0.0020 ± 0.0006, +0.0004 ± 0.0007, -0.0008 ± 0.0004, +0.00095 ± 0.00064, +0.0010 ± 0.0005 with +0.0014 optimism, against +0.00117 ± 0.00048 for a pre-declared equal mix | `experiments/hard-label-pools/`; `lab/solvers/ensemble` |
| Neural gate stacker on the 20-member pool, submitted | no effect | validation +0.00056; public -0.00035 against the plain pool, about 1 paired SE (0.00035 to 0.00062) | `experiments/gbdt-and-stacking/`; `lab/nn_x/stack/nnstack` |
| Pooling same-strength families trained with different recipes | helped then superseded | first such pool +0.00083 ± 0.00054 / +0.00157 ± 0.00034 over the one-family file; public 0.39208 against 0.38786 for the previous generation | `experiments/rounds-5-7/`; `lab/solvers/CPU1_xrecipe_member_equal_pool_v3.py` |
| Strength gates for families (member and six-seed mix within 0.002 of the best family) | helped then superseded | rejected the 2-epoch family (holdout 0.31903 against about 0.322) and the DIN family | `experiments/hard-label-pools/`; `lab/nn_x/picks/decide3.py` |
| Best click-label pool (G3): the GRU family and two transformer families whose cooldown was re-run, 20 members | helped then superseded | 0.40032 / 0.32379; public 0.392820 | `experiments/hard-label-pools/`; `lab/nn_x/picks/build_pool.py` |
| Pool of only the 14 members with a re-run cooldown, the best pool on validation | no effect | validation 0.40094; public 0.392434, which is 0.00039 below the 20-member pool, about 1 paired SE (0.00035 to 0.00062) | `experiments/hard-label-pools/`; `lab/nn_x/picks/build_pool.py` |
| Six-family pools of 38 members | no effect | public 0.392436 and 0.392723 against 0.392820 | `experiments/hard-label-pools/`; `lab/nn_x/picks/build_pall.py` |
| Second-level combiners in bulk (78 arms: pooling functions, family weights, feed-size weights, stackers) | no effect | 0 of 78 pass validation and holdout; family weights fitted on validation read z -2.85 on the holdout | `experiments/validation-and-leaks/`; `lab/nn_x/audit/dpa_1113` |
| Routing between pools per feed (31 label-free descriptors, model agreement, confidence) | no effect | router AUC 0.494-0.511 against a permuted null of 0.496-0.503; nested routed gain -0.0003 to -0.0010 | `experiments/mixing-and-segments/`; the coverage map |
| Post-hoc pooling variants (risk-averse pooling, agreement weights, head-weighted z, confidence mixing) | no effect | bounded by the infinite-member headroom: +0.0002 on validation, about 0 on the holdout | `experiments/tricks-sweeps/`; `lab/reports/TRICKS_SWEEP_1113.md` |
| Softmax-probability, raw-logit or rank pooling for the distilled files | no effect | holdout +0.00007 ± 0.00008, -0.00001 ± 0.00002, -0.00033 ± 0.00018; validation +0.00005 ± 0.00012 | `experiments/tricks-sweeps/`; `lab/reports/TRICKS_1115/web_competitions_checks/pool_forms.py` |
| Blend of the two best scored files | no effect | six disjoint file pairs 0.0000 ± 0.00012; blend minus the mean of its parts -0.00009 ± 0.00020 on validation | `experiments/tricks-sweeps/`; `lab/reports/TRICKS_1115/synth_checks` |
| Very large pools: 68 members, 60 all-data members of which 16 had a cooldown re-run on the stale view, 32 with held-out members | no effect | validation: 68 members -0.00024 ± 0.00028, 60 members -0.00056 ± 0.00027, 32 members -0.00036 ± 0.00027; public for the 68-member file 0.3946427 (-0.00025, -0.7 SE) | `experiments/pool-size-and-growth/`; `lab/nn_x/w12/ens_1115` |
| Growing the all-data student pool from 16 to 24, 32 and 48 | no effect | validation -0.00013 ± 0.00022, -0.00034 ± 0.00025 and -0.00049 ± 0.00028 against the 16-student file; the file law predicted +0.000055 from 16 to 32 and +0.000018 from 32 to 48; public 0.394905 (24) and 0.39484072 (48) against 0.394892 | `experiments/full-data-retrains/`; `lab/nn_x/w12/kd/fd/watch_growth2.sh`, `experiments/pool-size-and-growth/results/ens_e2c_file_law.txt` |
| Pool-size law NDCG(n) = a - b / n for planning | adopted | wide students: a 0.4015-0.4021 on validation, 0.3246-0.3250 on the holdout | `experiments/pool-size-and-growth/`; `lab/nn_x/w12/kd/pool_multi.py` |
| Second final file built from members disjoint from the first | inconclusive | test correlation between two disjoint all-data files 0.99726, against 0.99753 between the held-out best file and the all-data one; the candidate read -0.00052 ± 0.00039 on validation and missed its bar | `experiments/pool-size-and-growth/`; `lab/nn_x/w12/kd/build_r4.py`, the decision ledger |
| Mixed pool of plain students and students with the walk input | no effect | validation -0.00007 ± 0.00026; public 0.394578 (-0.00031, -0.9 SE); click-label members with and without the walk: cross-group correlation 0.9487 against 0.9490 within a group | `experiments/trained-input-search/`; `lab/nn_x/w12/nx3/mix_read.py` |
| Members retrained on all train feeds, the 30,000 holdout feeds included, last step kept | adopted | public 0.394892 against 0.3950094 for the held-out file: -0.00012 at a paired SE of 0.00056 | `experiments/full-data-retrains/`; `lab/nn_x/w12/stale/fulldata/lane_full.sh` |
| Merge with another team whose model is of a different family | not run | expected +0.0002 to +0.0004 before knowing the partner; protocol and checking script written | `experiments/tricks-sweeps/`; `lab/src/merge_check2.py` |
| Feed-size split blend (train-only branch alone on short feeds) | no effect | found at +0.00054 on the best click-label pool; its pre-declared confirmation on 6 unused members read -0.00017 ± 0.00031 on validation; on a pool with 3-seed fine-tune +0.00006 ± 0.00029 | `experiments/hard-label-pools/`; `lab/nn_x/w12/msplit_c3.py`, the status log |

### Post-processing and test-time tricks (10)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Test-time augmentation by dropout sampling, 8 passes | no effect | 6-member pool -0.00009 ± 0.00024 on validation | `experiments/rounds-5-7/`; `lab/solvers/r6/tta.py` |
| Test-time augmentation by dropping 10% of the history events, 8 passes | no effect | validation: member +0.00036 (5 of 6 positive); 6-member pool -0.00008 ± 0.00034 | `experiments/rounds-5-7/`; `lab/solvers/r6/tta.py` |
| Averaging predictions over shortened histories (256, 192, 128, 64 events) | hurt | validation: -0.00100 ± 0.00045 with 0 of 6 members positive; single crops cost -0.0007, -0.0015, -0.0041 | `experiments/rounds-5-7/`; `lab/solvers/r6/tta.py` |
| Re-shuffling candidates, scoring random sub-lists, 32-bit inference | no effect | the model is permutation-equivariant to 1e-4; sub-list averaging -0.0007 at pool level; 32-bit inference -0.0001 | `experiments/rounds-5-7/`; `lab/reports/r6_tta.md`, `lab/reports/RECHECK_1114/winner_model.md` |
| Re-ranking the head of the list with a second pass | not run | a perfect re-order of the top 8 recovers 29% of the validation headroom and 16% on the holdout, against a bar of 40%; 66-77% of the headroom is in feeds with no positive in the top 8 | `experiments/tricks-sweeps/`; `lab/nn_x/tricks/recycle_step0` |
| Metric-optimal post-processing key | no effect | -0.00006 ± 0.00032 / +0.00003 | `experiments/error-analysis/`; `lab/nn_x/insights/ccb_1113/RESULT.md` |
| Sweep of 63 post-processing, label and training tricks | not run | 12 ranked, 24 of 24 review verdicts negative; six follow-up probes closed 4 of them | `experiments/tricks-sweeps/`; `lab/nn_x/tricks` |
| Nearest-neighbour lookup of other users' labels at inference, additive key | not run | closed by mechanism: the key reduces to item similarity plus an item prior. The re-opened form, neighbours by trajectory with a label bank at least 3 days older, ran on CPU and read -0.0001 to -0.0003 on validation (labels from similar users, pre-training theme) | `experiments/tricks-sweeps/`; `lab/reports/TRICKS_SWEEP_1113.md` |
| Consensus of the model's own predictions for the same item in other users' earlier feeds | no effect | 4.54% of validation rows are covered against a gate of 10%; +0.00022 ± 0.00018 / -0.00050 | `experiments/error-analysis/`; `lab/nn_x/insights/ccb_1113/RESULT.md` |
| Code rules from a second error analysis of the final model (eight lenses, about 9,900 variants on half of validation) | no effect | the one rule that cleared the discovery bar read -0.00038 ± 0.00028 on the sealed half; a learned residual model reaches at most +0.0004 out of fold | `experiments/error-analysis/`; `lab/nn_x/errfix_1117/RESULT.txt` |

### Pre-training and embeddings (12)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Self-supervised trajectory encoder on all 2.59M histories (463.6M events) | helped then superseded | zero-shot 0.3590 on validation; +0.0037 ± 0.0008 in the tree harness; as a blend term +0.00082 ± 0.00038, later -0.00078 ± 0.00042 on a stronger pool | `experiments/pretraining/`; `lab/pretrain/train.py` |
| Encoder scores for train rows, quick leak-free version (blank where the item has no popularity) | helped then superseded | +0.0011 ± 0.0007 / +0.0008 ± 0.0004 at 3 epochs; in final members -0.0003 ± 0.0008 after the fine-tune | `experiments/pretraining/`; `lab/pretrain/make_prew.py` |
| Encoder scores for train rows, rebuilt with all candidates in the item table and the clip fixed | adopted | +0.00191 ± 0.00058 (z 3.3) on validation at 3 epochs, +0.00077 ± 0.00073 in a 10-epoch removal pair; removal costs -0.00128 ± 0.00061 / -0.00054 | `experiments/feature-families/`; `lab/feat_new/pre2_rescale.py` |
| Fine-tuning the encoder itself on validation labels | hurt | 0.3536 to 0.350-0.352 on validation | `experiments/pretraining/`; `lab/pretrain/finetune_val.py` |
| Encoder variants (larger ID learning rate, ID bias, ID dimension 128) | no effect | paired differences 0.0000 ± 0.00025 and +0.0002 ± 0.0003 | `experiments/pretraining/`; `lab/pretrain/train.py`, `lab/reports/pretrain_report.md` |
| Encoder as the initialisation of the ranker's history encoder, or as a frozen item-ID table | not run | gate on stored predictions failed two of three legs: zero-shot passed (0.3625 against a bar of 0.36), the nested residual on the pool was at or below 0 and the pair AUC on the task cell at most 0.476 against 0.52 | `experiments/pretraining/`; `lab/nn_x/r3/stage0_gate.py`, `experiments/pretraining/results/encoder_as_init_stage0_gate.txt` |
| Pre-training programme (in-category encoder, masked-event task, larger encoder, pseudo-feed pre-training, contrastive user encoders, foundation models) | not run | 0 GPU-h: the unlabelled corpus (463.6M events) is about 1:1 with the histories inside the labelled feeds (448.7M), a match of inputs and not of supervision; all six designs refuted on paper | `experiments/pretraining/`; `lab/nn_x/pretrain2` |
| The four CPU diagnostics of that programme (trajectory clusters, user-vector residual, whitened similarity, labels from similar users) | no effect | validation: 0 of 8 arms, best +0.00022 ± 0.00025; -0.00034 ± 0.00033; cell AUC +0.02 against +0.05 needed; -0.0001 to -0.0003 | `experiments/pretraining/`; `lab/nn_x/pretrain2` |
| Scaling the encoder channel: worth at 10 epochs, ID-free encoder swap, frozen hidden-state branch | no effect | channel worth +0.00077 ± 0.00073 on validation at 10 epochs against a trigger of +0.0015; hidden-state branch -0.00007 ± 0.00053; about 5.6 GPU-h | `experiments/pretraining/`; `lab/nn_x/abl/encdose/build_pre2noid.py` |
| Identification of the provided embeddings | adopted | 128-d unit vectors × 256 rounded to int8; they encode category (probe 0.983) and location, and no click rate (R2 0.003) | `experiments/codes-and-embeddings/`; `lab/nn_x/emb/origin` |
| Processing of the embeddings, five pre-declared items: change of a vector between tables, centred or whitened similarity (top directions removed), mean-direction projection, near-duplicates of history items, the user-item two-tower score | no effect | 0 of 5 pass; best, the mean-direction projection, +0.00045 ± 0.00022 nested on validation, and on the holdout -0.00004 ± 0.00003 nested or -0.00017 ± 0.00012 at the validation weight; whitening -0.00002 | `experiments/codes-and-embeddings/`; `lab/nn_x/emb/usability` |
| Location sub-space match between user and item vectors | no effect | -0.00008 ± 0.00030 / -0.00029 (z -2.08) | `experiments/codes-and-embeddings/`; `lab/nn_x/emb/structure` |

### Large language models (8)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Feasibility study of small language models for data without text | no effect | expected gain +0.00002 to +0.00008: a language model can add a computation prior and no new information; verdict a conditional go for stage A only, which ran (the rows below) | `experiments/llm-rankers/`; `lab/nn_x/llm/model_llm.py` |
| Qwen3-0.6B as a soft-token ranker backbone (LoRA rank 32, 150K feeds, 4 epochs) | hurt | -0.00378 ± 0.00077 / -0.00398 ± 0.00046 against the ranker on the same data; 17.09 GPU-hours, the 2.41 of the random-initialisation control included | `experiments/llm-rankers/`; `lab/nn_x/llm/train_llm.py` |
| Qwen3.5-4B and Qwen3.5-9B as backbones (2 epochs) | hurt | validation -0.00480 ± 0.00081 and -0.00532 ± 0.00080; 11.46 + 17.89 GPU-hours | `experiments/llm-rankers/`; `lab/nn_x/llm/train_llm.py` |
| The same architecture trained from random initialisation | no effect | pretrained minus random -0.00007 ± 0.00060 on validation at 0.6B; 2.41 GPU-hours, counted in the 0.6B row | `experiments/llm-rankers/`; `lab/nn_x/llm/train_llm.py` |
| Full-data training of a language-model backbone, or a one-pass slope read | not run | no size passed its rules, so the escalation never fired; by its correlation the family is worth at most +0.0006 | `experiments/llm-rankers/`; `lab/nn_x/llm/run_matched.sh` |
| A language model ranking 100 feeds from a text rendering, with no trained model | hurt | 0.3957 against 0.4433 for the pool on the same feeds (-0.048 ± 0.023) | `experiments/llm-rankers/`; `lab/nn_x/llm_manual/RESULT.txt` |
| Language-model reviewers correcting the final order feed by feed, after studying 100 labelled feeds | no effect | the reviewers found no edit that beats the model; their one rule read -0.00011 ± 0.00007 on 13,701 validation feeds and three other rules -0.00034 to -0.00070 | `experiments/llm-rankers/`; `lab/nn_x/llm_review_1117/RESULT.txt` |
| Eight ideas for new model families from a fresh review of the data | not run | 5 kept after review; the 4 trained as screens (single-stream transformer, category-factorised output, restricted history, up-weighted random-order feeds) have their own rows and none advanced; the fifth survivor (prior 0.013) and the 3 refuted ideas were never trained | `experiments/training-waves/`; `lab/nn_x/w9/fresh_eye_ideas_1113.json` |

### Decoding the anonymised codes (8)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Decoding what the codes are: event types, event sources, categories, features f1 to f5, the target | adopted | event types and sources are frequency ranks; categories follow the public category-id order (11 verticals); 83-87% of positives return as a feed-sourced view | `experiments/codes-and-embeddings/`; `lab/nn_x/codes/synth/x02_decoded_residuals.py` |
| Six pre-declared features built from the decoding (vertical shares, delivery block, region match) | no effect | 0 of 6 pass; -0.00039 to 0.00000, best nested -0.00000 ± 0.00011 | `experiments/codes-and-embeddings/`; `lab/nn_x/codes/synth/x02_decoded_residuals.py` |
| Hypothesis that a contact or order means a satisfied need | no effect | refuted: odds ratio within category 1.16 / 1.18; near-duplicates of contacted items are clicked more (actual over expected up to 1.03) | `experiments/codes-and-embeddings/`; `lab/nn_x/codes/exploit` |
| Home-region deficit (items outside the user's region are clicked less) | no effect | real (odds ratio 0.82 / 0.89), and the pool is already calibrated on it (0.981) | `experiments/codes-and-embeddings/`; `lab/nn_x/codes/synth` |
| 34 further constructions enabled by the decoding (surfaces, funnel depth, visits, verticals, category-relative age and others) | no effect | none clears +0.0003 at 2 SE as a nested residual; 47 cues tested in total | `experiments/codes-and-embeddings/`; `lab/nn_x/codes/exploit` |
| Item-to-item graph from the "similar listings" surface | no effect | covers 0.1% of candidates; nested -0.0001 to -0.0003 | `experiments/codes-and-embeddings/`; `lab/nn_x/codes/followup` |
| Popularity normalised by region | no effect | nested 0; matched win rate 0.50 against plain popularity | `experiments/codes-and-embeddings/`; `lab/nn_x/codes/followup` |
| Feature f1 as a key for a global sub-category grouping | no effect | refuted: 1.02M of 1.11M (feed, category, f1) groups are singletons | `experiments/external-solutions/`; `lab/nn_x/w12/forensics/p10_f1key.py` |

### External data and outside research (6)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Calendar, paydays, weather, public events, site promotions and market statistics as category × time priors | no effect | every label-free prior within ±0.0003 on validation and holdout; extrapolated trends -0.00035; joint -0.00034 | `experiments/external-solutions/`; `lab/nn_x/audit/external/e02_nested_priors.py`, `lab/nn_x/audit/external/e05_joint.py` |
| Ceiling for any time prior: validation's own same-day positive rate by category, day and hour | no effect | leaky in-sample oracle by category, day and hour +0.00062 ± 0.00036; out-of-fold by category and day +0.00002 ± 0.00018 | `experiments/external-solutions/`; `lab/nn_x/audit/external/e02_nested_priors.py` |
| Outside knowledge per category (seasonality, purchase cycle, listing lifetime, complement categories, local hour) | no effect | 0 of 9; best +0.00019 ± 0.00019; sourced seasonality is uncorrelated with the real category drift (Spearman -0.06) | `experiments/error-analysis/`; `lab/nn_x/insights/enrich_1113` |
| Scan of 211 techniques from winning solutions of related competitions | not run | classified by prior against the team's records: 82 weakly closed, 7 new, none with a probability above 0.035; most CPU probes run during the scan have their own rows | `experiments/external-solutions/`; `lab/reports/EXTERNAL_SCAN_1113.md` |
| Web search for write-ups or hints about this competition | no effect | nothing public beyond the organiser page | `experiments/external-solutions/`; `lab/reports/TOP1_GAP_EXTERNAL_1114.md` |
| Final research round on every remaining trick (6 scans, 82 web sources) | no effect | no untested change to the team's own model has an expected file gain above +0.0001 | `experiments/tricks-sweeps/`; `lab/reports/TRICKS_1115/synth_checks` |

### Leaderboard forensics, error analysis and user segments (11)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Test of whether the leader's margin is the same-user later-feed leak (overlay on the best pool, submitted) | no effect | validation +0.00454 ± 0.00047; public 0.392819 against 0.392820. Re-read on pick 1 on 18 Nov, nested over user halves: +0.00452 ± 0.00047, and +0.0248 ± 0.0030 nats per click | `experiments/validation-and-leaks/`; `lab/src/later_overlay.py`, `lab/nn_x/w12/leader/data_channels/p4_battery.py` |
| Five audits of test-time information channels (cross-user counts, co-candidacy, id orders, earlier feeds) | no effect | all other channels within ±0.0005 on validation. Re-read on pick 1 on 18 Nov, nested over user halves: other users' activity as of the feed or later and the backward change of f5 -0.00011 to -0.00038, against placebos of -0.00013 to -0.00024; history events lost from the user's current snapshot +0.00115 ± 0.00040 nats per click and +0.00000 NDCG, and all of them put first -0.00059 ± 0.00022 | `experiments/validation-and-leaks/`; `lab/nn_x/audit/testinfo`, `lab/nn_x/w12/leader/data_channels/` |
| Raw-data forensics: schema, row order, part files, ids, embedding table order, sample-submission order | no effect | all 19 raw columns are used; row order or item id as a ranking key scores 0.288-0.289, which is random; best of 270+ nested reads +0.00044 ± 0.00020, the expected maximum of 248 nulls. Re-read on pick 1 on 18 Nov: stored order, embedding index, item id and item-id rank minus f4 rank -0.00004 to +0.00003 | `experiments/external-solutions/`; `lab/nn_x/w12/forensics/p05_sweep.py`, `lab/nn_x/w12/leader/data_channels/p4_battery.py`, the status log |
| Signals from the same user's earlier labelled train feeds | no effect | 5 signals, -0.00036 to +0.00010 | `experiments/external-solutions/`; `lab/nn_x/w12/forensics/p08_transfer.py` |
| Heavier use of validation labels (trees trained jointly on train and validation rows) | inconclusive | trees +0.00206 ± 0.00101 with validation rows at weight 1; the network analogue equals the existing fine-tune. Other users' validation clicks on near-duplicates of the candidate, re-read on pick 1 on 18 Nov: -0.00033 ± 0.00012 as of the feed, against a placebo of -0.00004 | `experiments/external-solutions/`; `lab/nn_x/audit/top1`, `lab/nn_x/w12/leader/data_channels/p3_read.py` |
| Error analysis of the pool | no effect | 83% of the gap comes from positives at ranks 3-99; 70 attributes have a local AUC of 0.492-0.504; nested residual models -0.0006 to -0.011 | `experiments/error-analysis/`; `lab/solvers/errors` |
| Search of pick 1 for systematic misses a correction could fix (659 calibration slices, cross-fitted corrections) | no effect | the score is a calibrated log-odds (temperature 1.004 and 1.011 on two halves of the users); of 659 slices, 19 deviate with the same sign on both halves (actual over expected 0.80 to 1.08), none systematic; a perfect correction of every detected bias is worth +0.00035, corrections estimated from half of the users cost 0.0004-0.0009 in placebo runs; about 50 cross-fitted corrections read 0 to -0.0013; positives in a category new to the user (32% of positives) are calibrated, model mass 0.329 against 0.322 | `experiments/error-analysis/`; `lab/nn_x/w12/fixmiss/systematic` |
| Cold-start, anonymous and sparse users as a segment to repair | no effect | no history: 0.86% of test feeds, NDCG 0.378 ± 0.018; users never seen in train score 0.4065 against 0.3940 for seen ones; ceiling about +0.0002 | `experiments/mixing-and-segments/`; `lab/nn_x/w12/seg/diag` |
| Heavy users at the 256-event history cap as a segment to repair | inconclusive | 28.8% of test feeds; validation 0.3906 ± 0.0037 against 0.4077 for the rest; removing 64 events costs -0.0023 ± 0.0008 | `experiments/mixing-and-segments/`; `lab/nn_x/w12/seg/diag` |
| Data-volume slope: training on 80% of the users | adopted | member -0.0028 / -0.0022; 3-pool -0.0017 / -0.0019 | `experiments/distillation/`; `lab/nn_x/w12/recheck` |
| Reading experimental files of one family on the public board | no effect | seventeen files of the final family read 0.3943559-0.39526061, mean 0.39486, sd 0.00025, below the paired noise between them (0.0002-0.0006) | `experiments/full-data-retrains/`; `lab/nn_x/w12/kd/build_r4.py` |

### Validation protocol and decision rules (24)

| Idea | Outcome | Measured effect | Where |
|---|---|---|---|
| Paired per-feed differences with a standard error as the unit of every comparison | adopted | a single validation score has an SE of about 0.0019; the paired SEs on this page are 0.0002-0.0008 | `experiments/validation-and-leaks/`; `lab/src/metrics.py` |
| Holdout of 30,000 random-order train feeds for checkpoint selection and as a second witness | adopted | later found blind to short feeds (39% of private) and biased for changes to popularity inputs | `experiments/neural-ranker-v1/`; `lab/nn/holdout_eval.py` |
| Adversarial validation between splits | adopted | train against validation AUC 0.925; validation against test 0.524; feed-level cold-item share by the classifier's own definition 0.29 on validation, 0.37 on public, 0.41 on private (the cold share of the glossary: 0.348, 0.439, 0.477) | `experiments/data-understanding/`; `lab/solvers/advval/feed_adv.py` |
| Validation or holdout re-weighted with adversarial weights | no effect | invalid: effective sample size 2,880 | `experiments/data-understanding/`; `lab/solvers/advval/weights.py` |
| Prediction of the public score from validation re-weighted by feed size | helped then superseded | within 0.0003-0.0019 of the actual score on the first five files; replaced by public = 0.0074 + 0.963 × validation, residual sd 0.0006 | `experiments/validation-and-leaks/`; `lab/src/calib.py` |
| Public-mix and private-mix views: validation re-weighted to the public or private mix of feed size and cold share | adopted | differences overshoot about 5× and the level by about +0.0045; as a sign witness it agreed with public on the 5 scored pairs where public moved (the sixth was a tie), plain validation on 5 of 6 | `experiments/popularity-staleness/`; `lab/src/pmix.py`, the status log |
| Other validation designs: public-matched weights, 40 pre-specified re-weightings, a temporal holdout, an as-of stale holdout | no effect | public-matched weights transfer at 0.10 against 0.23 unweighted; none of the 40 explains the shortfall of the validation-best pools; 63.6% of the as-of holdout feeds have no history | `experiments/validation-and-leaks/`; `lab/nn_x/audit/valpub` |
| Study of how validation, holdout and public relate | adopted | between generations public = 0.0071 + 0.964 × validation over 11 files; within a generation 7 pools span 0.00198 on validation and 0.00074 on public, rank correlation +0.04 | `experiments/validation-and-leaks/`; `lab/nn_x/audit/valpub` |
| Stale validation: validation replayed with counterfactual stale inputs | adopted | reads the stale-view cooldown at +0.00103 to +0.00120 where plain validation reads +0.00024 | `experiments/popularity-staleness/`; `lab/nn_x/w5/stale_pred.py` |
| Lagged holdout: holdout re-scored with popularity inputs rebuilt at the test lag | adopted | valid on 13,424 feeds; there the 0.2 admixture reads +0.00024 ± 0.00031 with lagged inputs against -0.00050 ± 0.00032 with stored ones | `experiments/popularity-staleness/`; `lab/nn_x/w12/next/poplag` |
| The holdout as a veto for every idea | helped then superseded | wrong sign on both scored popularity-light contrasts: -0.00063 against public +0.000523, and -0.00073 against +0.000634; 38 decisions re-audited, 31 of them holdout-only closures | `experiments/popularity-staleness/`; `lab/nn_x/w12/stale/veto` |
| Rules written and time-stamped before the first run they apply to (the ledger) | adopted | no number | `experiments/training-waves/`; the decision ledger |
| Mandatory pre-submission gate on every file (leak probe, overfit and validity checks) | adopted | two false alarms: the leak probe at z 2.0 on 117 feeds, and check B on a valid file whose 4 feeds each held one exact float32 tie that the writer and the gate broke differently (a tie-aligned copy passed) | `experiments/validation-and-leaks/`; `lab/src/presub_gate.py` |
| Two-epoch paired screen as a futility filter | helped then superseded | detects about +0.0008 on validation; one pass at +0.00117 (z 2.48) shrank to +0.00036 at 10 epochs | `experiments/popularity-views/`; `lab/nn_x/abl/abl_lane.sh` |
| Decision rule: 8 seeds in one stage, pooled z of validation and holdout, against a same-size reference | adopted | passes 0.25% of nulls and 56% of true +0.0005 effects; the earlier 2-seed screen passed 15% of nulls and 36% of true effects | `experiments/distillation/`; `lab/nn_x/w12/kd/arm_read8.py` |
| Judging recipe changes at pool level | adopted | weight averaging: member +0.0004, pool -0.0025; heavier teacher: member +0.00126, pool -0.00049 (validation) | `experiments/distillation/`; `lab/nn_x/w12/kd/arm_read8.py` |
| Nested residual gate for candidate features | helped then superseded | a block worth +0.00852 ± 0.00066 as a trained input reads only +0.00086 and +0.00222 on the holdout in two residual forms, and nothing on validation; about 300 feature rows were closed by it | `experiments/validation-and-leaks/`; `lab/nn_x/audit/features` |
| Calibrated conditional score test for stored feature columns | inconclusive | finds 25 of 284 stored columns on both witnesses; one survives the parity and lag checks and then fails as a trained input | `experiments/trained-input-search/`; `lab/reports/INPUTS_1116/residual_rows/condlib.py` |
| The public board as a judge of near copies | no effect | paired public SE between near-copy files 0.0004-0.0007; smallest detectable effect 0.00112 on public | `docs/04-validation.md`; the freeze protocol |
| Acceptance check for members trained without a holdout (twin check) | adopted | in-sample holdout lift +0.0158 to +0.0179 against a bar of +0.0043; 16-pool of twins against the held-out originals +0.00004 ± 0.00042 on validation | `experiments/full-data-retrains/`; `lab/nn_x/w12/stale/fulldata/accept_twins.py` |
| Non-inferiority rule against a single realised file | inconclusive | the 32- and 48-student files failed it although their members are exchangeable with the old ones; the reference's 0.40271 on validation is a favourable draw | `experiments/full-data-retrains/`; `lab/nn_x/w12/kd/fd/watch_growth2.sh` |
| Re-judging closed ideas on a "clean" part of the holdout | no effect | refuted as a judge chosen after the outcome; kept as a label-free information column | `experiments/tricks-sweeps/`; `lab/nn_x/tricks/hold_clean` |
| Calibration of the holdout standard error | inconclusive | first study, 3 pairs: run-to-run holdout z about 2.3 times larger than the per-feed SE implies; second study on 15 same-recipe seed pairs: rms z 0.83 on validation, 0.89 on the holdout; an A/A test on 14 pairs: rms z 1.04 on validation | `experiments/optimiser-and-modern-tricks/`; `lab/reports/nnx_VERDICT.md`, `lab/reports/REVIEW_SYNTHESIS.md`, `lab/reports/review_metric.md` |
| Freeze protocol for the two final files, written in advance | adopted | no score; public is read only outside 2 paired SE; on 2026-11-18 the team set aside its rule for replacing pick 1 (a near copy needed a rule written before its run or 0.395992 on public) and took FD16P20@m0.4_0.2 at about +1 paired SE; for pick 2 the team took its option of W8P20, which waives the all-data rule; on 2026-11-19 FDX16P20:ps (public 0.39526061, 0.8 paired SE above FD16P20) replaced W8P20, which set aside the 2-SE rule for probes, and both picks meet the all-data rule | the freeze protocol; `lab/nn_x/w12/freeze` |
