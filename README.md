# Avito ML Cup 2026: personal ranking

This is an experiment and a statement. I entered a Kaggle-style competition, the "Personal ranking" track of the Avito ML Cup 2026 on ods.ai, without writing a line of code. The task is to order the listings of a recommendation feed so that the ones the user opens come first.

Claude (Claude Code running Claude Opus 5.5) did the research, the code, the experiments and this write-up in nine days, steered by about 110 short requests from me. On 2026-11-19 the entry stood at rank 9 of 541 teams on the public leaderboard, under the name "Baseline".

The name is deliberate. In my view a result like this is the baseline now: it is what comes out when one person asks an agent, with no automated search behind it. Such competitions have become contests of harness and compute more than of ideas, and I see little reason to enter them by hand ([my view](#my-view-on-competitions-like-this)).

The repository is meant to be useful beyond that argument. It documents what the data is, how to measure small effects on a noisy score, 312 ideas with their numbers, most of which did nothing or hurt, and why the score stopped about 0.003 below the leader. The final pipeline is the Python package `avito_ranker/`; the 25 research lines and a trimmed copy of the working tree sit beside it.

The work covers 2026-11-11 to 2026-11-19 and was written up by 2026-11-20. Submissions stay open until 2026-12-08, and the private result comes on 2026-12-09.

## In brief

- A feed is one user's request to the Avito main-page recommendations, with 5 to 300 candidate listings to put in order. The score is NDCG (normalised discounted cumulative gain) over the full list with binary gain, averaged over feeds.
- The best file scores 0.39526061 on the public leaderboard: rank 9 of 541 teams on 2026-11-19 at 03:34 UTC. The private result is unknown until 2026-12-09.
- The two final picks are ensembles of 20 small transformer rankers each, with 4 members in common: public 0.3952092 and 0.39526061. Their lab names, FD16P20@m0.4_0.2 and FDX16P20:ps, are decoded under [Notation and names](#notation-and-names).
- The catalogue holds 312 ideas with their numbers, 22 of them closed without a run ([what was tried](#what-was-tried)). [docs/12](docs/12-why-the-score-stopped.md) weighs why the score stopped about 0.003 below the leader and lists five ideas the search may have missed.
- Claude Code running Claude Opus 5.5 did all the research, code, experiments, analysis and documentation, steered by about 110 short requests and questions from me. I also uploaded the files and chose the picks. No automated search loop ran ([how this was built](#how-this-was-built), [docs/13](docs/13-how-the-work-was-run.md)).
- In this README "I" is the person who entered the competition and "Claude" is the agent session that did the work. The pages under `docs/` say "the team" for the two together.

![Public NDCG of 40 scored files in submission order, 11 to 19 November. The best score climbs from 0.3739 to 0.3921 on 11 November, then steps up to 0.39282 and 0.39345 with pools of rankers trained on clicks, 0.39449 with distilled students, 0.39501 with W8P20, 0.39521 with FD16P20@m0.4_0.2 (pick 1) and 0.39526 with FDX16P20:ps (pick 2). On the board of 19 November the leader is at 0.398256, place 2 at 0.398058 and this entry at rank 9 of 541.](docs/figures/public-progress.svg)

The figure shows all 40 scored files. The first file scored 0.3739 on 11 November, and 0.018 of the 0.021 total gain came that day.

### Where to start

The first five paths are reading only, and the sixth runs the code on synthetic data. The competition data, the predictions and the submission files are not in this repository.

- What the data is and what was found in it: [the data section](#the-data-and-what-claude-found-in-it), then [docs/02](docs/02-what-the-data-told-us.md).
- How to measure small effects on a noisy score: [docs/04](docs/04-validation.md) and the lessons of [docs/09](docs/09-lessons.md).
- What did not work and why the score stopped: the patterns of [docs/07](docs/07-what-failed.md#patterns-that-recur), then [docs/12](docs/12-why-the-score-stopped.md).
- How one agent session ran the work and what automation could change: [docs/13](docs/13-how-the-work-was-run.md).
- Where a number comes from: [docs/evidence.md](docs/evidence.md) traces each headline number to a published file, and [docs/catalogue.csv](docs/catalogue.csv) holds the 312 ideas as a table.
- How the code runs, in half a minute and without the data: [Try it without the data](#try-it-without-the-data).
- How to rebuild the files, which needs the competition data and about 150 GPU-hours for the first pick: [Reproducing](#reproducing).

### Notation and names

Scores are NDCG. "±" is one standard error (SE) of a difference paired per feed, and a pair a / b is validation / holdout. Other terms are in the [glossary](docs/glossary.md).

File names such as FD16P20@m0.4_0.2 are lab handles. FD means members trained on all train feeds, and an X after it means the full-view students come from other runs of the same recipes. 16 is the number of full-view students, and P20 a popularity-light pool at weight 0.2. @m0.4_0.2 sets that weight to 0.4 on feeds of fewer than 50 candidates and 0.2 on the others, and :ps marks students that also trained on pseudo-feeds.

## Results

The public leaderboard scores 8,562 of the 28,595 test feeds; the private part is the other 20,033. On the [public leaderboard](https://ods.ai/competitions/avitotechrecsys2026/leaderboard) as fetched on 2026-11-19 at 03:34 UTC (541 teams, 1,438 participants, 7,785 submissions), the top of the table read:

| Place | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9, this entry | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Public NDCG | 0.3982556 | 0.3980582 | 0.3968832 | 0.3966275 | 0.3963550 | 0.3959215 | 0.3954529 | 0.3953650 | 0.3952606 | 0.3949725 |

The leader's score has not changed since 12 November. On 18 November at 03:33 UTC this entry was at rank 8 of 532 with 0.3952092. It gained 0.00005 by the next day and still lost a place, because the team then at rank 9 rose to rank 6. The board snapshots are in [leaderboard.csv](docs/figures/data/leaderboard.csv).

The milestone files below run from the first file to the two picks. Their descriptions use the terms below; the [glossary](docs/glossary.md) has these and the other lab names.

- A click-label ranker is trained on clicks alone. A student is trained toward half clicks and half the averaged scores of click-label rankers, which are its teacher; this is distillation.
- A pseudo-feed is an extra training list built from a train feed's own history: a recent card the user opened from an earlier feed becomes the one positive, the events older than it by a gap of 16 to 64 events become the history, and the feed's non-clicked candidates are the negatives.
- A full-view model reads all 34 pre-computed inputs. A popularity-light model reads one clipped exposure count in place of the 8 popularity inputs, which go stale on test.
- The cooldown is the last 20% of the training steps. The fine-tune trains each model further on the two validation days.
- Most train feeds are in the display order of Avito's production ranker. Random-order train feeds were shuffled before display, like validation and test. The holdout is 30,000 feeds that the first detector marked as random-order, kept out of training; a later detector found 2,294 of them in production order. It chose checkpoints and gave a second read next to validation.

Validation in the table is the mean NDCG over the 14,423 labelled validation feeds (two days). The fine-tuned branch of a file enters this score with out-of-fold predictions. The first column is the row in the [list of all 40 scored files](docs/05-timeline.md#all-40-scored-submissions), which says what each file changed.

| # | Date (2026) | What changed | Validation NDCG | Public NDCG |
|---|---|---|---|---|
| 1 | 11-11 | First file: early neural rankers, a tree-model stacker on top of them and boosted trees, blended | 0.3816 | 0.3739 |
| 5 | 11-11 | Four rankers trained for 10 epochs (before: 2 to 3) | 0.39399 | 0.38786 |
| 6 | 11-11 | 19 rankers from three recipes with the Muon optimiser, equal weights (Cd) | 0.39901 | 0.39208 |
| 7 | 11-12 | 20 click-label rankers, 6 with a recurrent (GRU) history encoder and 14 with a cooldown re-run in which 2% of the batches are random-order train feeds (G3) | 0.40032 | 0.392820 |
| 16 | 11-13 | G3 plus 8 popularity-light click-label rankers, fine-tuned over 3 seeds: the best click-label file (C3) | 0.40101 | 0.393454 |
| 19 | 11-14 | 8 full-view students of width 192 in place of the click-label rankers (W8) | 0.40165 | 0.394486 |
| 22 | 11-15 | The same 8 at weight 0.8 plus 4 narrow popularity-light students (width 128) at 0.2: W8P20, pick 2 until 19 November | 0.40214 | 0.3950094 |
| 24 | 11-15 | 16 full-view and 4 popularity-light students, all width 192, all trained on all train feeds (FD16P20) | 0.40271 | 0.394892 |
| 33 | 11-17 | The members of FD16P20 with the popularity-light weight raised to 0.4 on feeds of fewer than 50 candidates: FD16P20@m0.4_0.2, pick 1 | 0.40293 | 0.3952092 |
| 38 | 11-19 | FD16P20 with its 16 full-view students replaced by students that also train on pseudo-feeds (FDX16P20:ps): pick 2 and the best public file | 0.40286 | 0.39526061 |

### How far validation forecasts the board

A file reads about 0.0075 lower on public than on validation, the mean over the 38 scored files with a clean validation score. Public feeds are longer (103.6 candidates on average against 96.4), and longer feeds score lower. The level line public = 0.0074 + 0.963 × validation, fitted on 18 scored files with a residual standard deviation (sd) of 0.0006, forecasts the public score of a file.

For pick 1, validation 0.40293 forecasts 0.39542, and it scored 0.39521, 0.00021 below the line.

Neither validation nor the board can order near copies, pools whose scores correlate at 0.995 or more within each feed and differ by at most 0.002 on validation. Seven click-label pools of one model generation spanned 0.00198 on validation and 0.00074 on public, with a rank correlation of +0.04 ([docs/04](docs/04-validation.md#noise)).

Choosing among near copies by validation failed twice on 12 November:

- the pool with the best validation score (0.40094) landed 0.00039 below the plain pool on public;
- a small model that re-weights the members per feed gained 0.00056 on validation and scored 0.00035 below the plain pool.

Seventeen scored files of the final family lie between 0.39436 and 0.39526 on public (mean 0.39486, sd 0.00025). The paired standard error, the standard error of the per-feed difference between two files, is 0.0002 to 0.0006 among them ([docs/11](docs/11-final-picks.md#risks)).

The 0.40271 of FD16P20 is a favourable draw of its members: pools of 32 and 48 full-view students of the same recipe read 0.40237 and 0.40222 on validation ([docs/07](docs/07-what-failed.md#18-growing-the-ensemble-past-16-members)).

### Final picks

The picks are rows 33 and 38 of the table above. Both began as try files and became picks after their public scores, so both choices are post hoc. The submission files are not in this repository: a pick is defined by its ensemble config and can be rebuilt with the pipeline ([Reproducing](#reproducing)). Each has 16 full-view and 4 popularity-light students, all trained on all train feeds, and the 4 are the same in both.

Pick 1, FD16P20@m0.4_0.2, has the 20 members of FD16P20 and gives the same order as FD16P20 on every test feed of 50 or more candidates. Its public lead over FD16P20, +0.00032, is one paired standard error. Pick 1 was the best of 14 near copies on public when it was chosen, so only about +0.00005 of that lead, a tie, is expected on private (estimate in [docs/11](docs/11-final-picks.md)).

Pick 2, FDX16P20:ps, takes its 16 full-view students from separate runs of the same recipes and seeds, with pseudo-feeds added. It replaced W8P20 (public 0.3950094, members trained without the 30,000 holdout feeds) on 19 November. If the better of two final files counts, a second file adds about +0.0001 to the expected private score over pick 1 alone, with W8P20 or with FDX16P20:ps. FDX16P20:ps reads 0.0007 higher on validation and 0.00025 higher on public than W8P20.

The package trainer has no pseudo-feed option, so those 16 students can be retrained only with the lab trainer. [docs/11-final-picks.md](docs/11-final-picks.md) has the reasoning, the audit and the risks.

## The data and what Claude found in it

The data covers September 2026 in three splits cut by time ([docs/01](docs/01-competition-and-data.md#the-splits)):

- train: 3,151,208 labelled feeds (6 to 14 and 17 to 19 September), shown in the order of Avito's production ranker;
- validation: 14,423 labelled feeds (22 and 24 September), shown in random order;
- test: 28,595 unlabelled feeds (26 to 29 September), shown in random order.

A classifier on feed-level statistics tells train from validation at AUC 0.925, validation from test at 0.524.

![Calendar of the data, 6 to 29 September 2026, Moscow time: train on 6 to 14 and 17 to 19 September, validation on Tuesday 22 and Thursday 24, test on 26 to 29 with the public / private cut at 11:32 on Sunday 27. Below, the age of the popularity inputs (mean days since a candidate item was last seen in train), one line per set: holdout (30,000 random-order train feeds) 1.75, validation 5.69, public 9.34, private 11.23.](docs/figures/data-timeline.svg)

### What one feed holds

The evidence for the last column is in [docs/01](docs/01-competition-and-data.md#fields).

| Field | Raw content | Claude's reading |
|---|---|---|
| `user_id` | anonymous id, one numbering for all splits | the id 0, documented as missing, pools visitors who were not logged in (123,904 train feeds) and is never linked across feeds |
| `user_embedding` | 128 int8 values | the user side of Avito's own recommender, recomputed at each request |
| `item_id`, `item_id__embedding_idx` | 5 to 300 candidate items, each with 128 int8 values in its split's table | the item side of the same recommender |
| `category_id` | code 0 to 53, per candidate and per event | 1 plus the dense rank of the leaf category id in Avito's public category tree |
| `f1` .. `f5` | "normalised real features", per candidate | f3 the user's affinity to the leaf category, f1 a finer one inside it, f2 the affinity to the listing's region, f4 the listing's age, f5 an unsmoothed engagement rate |
| `history__*` | up to 256 events, newest first: item, category, event type, event source | a snapshot at request time, with no timestamps; 42.3% of train feeds and 29.0% of validation feeds reach the cap |
| `history__event_type` | code 0 to 10 | a frequency rank: 1 is a card view (43.9% of validation events), 4 a favourite, 5, 6 and 8 a delivery block |
| `history__event_source` | code 0 to 30, set only for types 1, 4 and 7 | a frequency rank: 1 is the main-page feed (64.5% of validation views), 2 search or catalogue |
| `pos` | train only | display position in the production order, from 0 |
| `target` | train and validation | the label, one bit for any of several unnamed actions |

A positive is a candidate with one of those actions after this feed showed it; by Claude's reading, 83 to 87% of positives are a card opened from the feed. Positives are 2.21% of train and 1.58% of validation candidates, and 66.6% of train and 74.5% of validation feeds have exactly one.

f1 to f5 are quantile-bin encodings fitted on train. None is the production ranker's score: each correlates with `pos` within ±0.08 (per-feed Spearman). The ranker of this solution leans on f4 most: zeroing it at inference cost 0.0138 validation NDCG, averaged over two 2-epoch click-label rankers.

### The embeddings

Users and items come as 128-dimensional int8 embeddings, missing for 14.2% of validation history events. The values are unit vectors times 256, rounded, so a dot product is about 65,536 times the cosine. Each split has its own table, and 77 to 83% of the items shared between splits have bit-identical embeddings.

Claude judged them, at a subjective probability of 0.94, to be the two towers of Avito's own recommender: two networks that embed users and items into one space. The user embedding is recomputed at every request, and only 0.08% of consecutive same-user train feeds have identical ones ([docs/02](docs/02-what-the-data-told-us.md#whose-vectors-they-are)).

Linear probes on 1.5 million train items read the category (98.3% accuracy) and a region component (partial R² 0.346 for f2 given the category), and almost no click rate (R² 0.0034). The click signal is in how a candidate relates to the user. On validation, in NDCG:

- the user-item dot product ranks at 0.3432;
- the mean cosine to the 5 newest history items at 0.3474;
- an attention scorer of about 330 parameters at 0.3584.

The ranker reads each embedding as a unit vector: each candidate attends to the history with the cosine added to the attention logits and retrieves its 16 most similar history items ([docs/03](docs/03-solution.md#the-ranker), gains in [How the solution works](#how-the-solution-works)). Whitening the embeddings cost 0.0440 ± 0.0018 validation NDCG when ranking by the user embedding.

What was built on the embeddings or in their place, in NDCG:

| Idea | Result |
|---|---|
| 10 inputs that describe near-duplicates among the feed's candidates | in the final model; removing them cost 0.0014 on validation and 0.0003 on the holdout (2-epoch runs) |
| A trajectory encoder pre-trained on 2.59 million user histories | 0.3590 on validation alone; its 6 score columns are in the final model, worth +0.00077 ± 0.00073 on validation in a 10-epoch pair |
| Five processings: drift flag, whitened or centred similarity, mean-direction projection, near-duplicates of acted items, user-item cosine interactions | none passed its bar of +0.0005 and 2 SE; best +0.00045 ± 0.00022 on validation and -0.00004 on the holdout |
| A hashed item-ID embedding table of 2^22 rows by 32, with 50% ID dropout | -0.0022 ± 0.0008 on validation for a click-label ranker, -0.0075 ± 0.0021 on feeds whose positives have unseen ids; -0.0057 / -0.0049 (validation / holdout) for students |
| Re-ranking near-duplicates after the model, 64 set-level rules | best +0.00023 ± 0.00015 on validation; a 2-fold estimate of that choice read -0.00076 |

The item-ID table taught the model to prefer items it knew, and test has fewer: 61% of its candidates have an item id that occurs in train at all, against 71% on validation (a wider count than the train-candidate share below). Pre-training had little to add as input: its corpus of 463.6 million events is about the size of the 448.7 million distinct events in the labelled feeds' histories ([docs/07](docs/07-what-failed.md#11-pre-training-on-the-history-corpus)). As supervision it is [an open idea](#what-to-try-next).

### Categories, event types and sources

Event types and sources are frequency ranks: by share of train events, none of the 45 pairs of types or 435 pairs of sources is out of order. Category codes follow Avito's public leaf category ids, in 11 verticals. Other meanings came from behaviour: types 5, 6 and 8 are absent from the 15 category codes where nothing can be shipped, so they mark delivery.

The decoding explained the label and added no score. Six features that need it read -0.00039 to 0.00000 validation NDCG over the best click-label pool, and none of 47 cues cleared +0.0003 at 2 SE. The ranker embeds every code, and that pool was already calibrated on every decoded class: actual over expected clicks 0.97 to 1.03 ([docs/02](docs/02-what-the-data-told-us.md#the-decoding-added-no-score), [experiments/codes-and-embeddings](experiments/codes-and-embeddings/)).

One reading reached the final files. f5 is an unsmoothed rate, so a ranker without popularity inputs cannot tell a reliable high f5 from a noisy one; the popularity-light view therefore keeps one exposure count, log1p(min(n, 50)) for an item shown n times as a train candidate. On public ([docs/02](docs/02-what-the-data-told-us.md#one-reading-that-reached-the-final-files)) the best click-label pool scored:

- 0.392820 alone;
- 0.392886 with 4 popularity-free rankers;
- 0.393205 with 4 popularity-light ones.

The count did not repair the validation cell it was built for, high f5 with fewer than 5 exposures. The view stayed for its public scores.

### Findings that shaped the model

- [Train labels come from a ranked list](docs/02-what-the-data-told-us.md#train-labels-come-from-a-ranked-list): in train, 11.81% of candidates at display position 0 are positive and 0.92% at position 200. A boosted-tree model fitted on train scored 0.3301 validation NDCG, below f1 alone (0.3365). The freshest tenth of candidates by f4 has 1.22 times the mean positive rate in train and 0.46 times in validation.
- [Random-order feeds hide inside train](docs/02-what-the-data-told-us.md#random-order-feeds-inside-train): about 13% of the train feeds with 50 or more candidates, unflagged. A detector built on the logged positions found them; they became the holdout and the random-order cooldown batches. Without those batches a ranker ties on validation and loses 0.0023 NDCG on the validation feeds whose positives are all cold items (never train candidates).
- [Item statistics go stale](docs/02-what-the-data-told-us.md#cold-items-and-stale-popularity). The share of candidate rows whose item was a train candidate falls from 65.4% on validation to 56.4% on public and 52.7% on private, and the counts age (calendar above). Blended with f1 on validation, an item's train click rate read -0.0016 NDCG and its count in user histories +0.0032, so the 8 popularity inputs are co-visitation scores and history counts, never a click rate.
- [A user's next snapshot holds the clicks of the earlier feed](docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak), so counts over all histories carry train labels. Tree models trained on such counts gained 0.025 to 0.052 NDCG on the 117 validation feeds whose user has a later feed. Every input since skips the user's own later snapshots, never links user-0 feeds and is built alike for train and test rows.
- [Many listings are near-duplicates of each other](docs/02-what-the-data-told-us.md#near-copies-among-the-listings): 33.2% of validation items have a neighbour at cosine above 0.98. A candidate whose largest cosine to the history is 0.99 or more (1.2% of candidates) is positive 2.8 times as often as the base rate. The first neural ensemble, of 11 November, ordered a positive above its near-duplicate negative 61.1% of the time, and f1, f4 and f5 were at chance.

## How the solution works

Each final file is an ensemble of one small transformer ranker, trained many times with different seeds, teachers and input views. A full-view member of width 192 has 4 heads, 5 attention blocks and 2,118,848 parameters. [docs/03-solution.md](docs/03-solution.md) describes every part with its measured effect. The steps, as run for the first pick:

1. Per feed the ranker reads up to 300 candidates and the user's 256 newest history events, with 128-dimensional int8 embeddings of listings and user. Each candidate also carries 34 inputs computed beforehand: co-visitation (how often the candidate appears near the user's history items in other users' histories), popularity counted without the same user's later snapshots, the structure of the feed, and scores of a self-supervised model of history sequences.
2. Twenty-five rankers trained on clicks alone score every train candidate, and their logits, averaged over 11 of them (over all 25 for half of the full-view students), are the teacher.
3. Each student trains for 10 epochs (99,840 steps) on all 3,151,208 train feeds toward 0.5 × clicks + 0.5 × softmax of the teacher, with Muon (an optimiser that orthogonalises each update of a weight matrix) on the hidden weight matrices and AdamW on the rest.
4. Each student is then fine-tuned on the 14,423 validation feeds, in 5 folds by user and 3 seeds for 4 epochs, with 42 extra inputs that exist only for validation and test.
5. The ensemble standardises the predictions within each feed (z-scores) and averages each pool. The popularity-light pool enters at 0.4 on feeds of fewer than 50 candidates and 0.2 on the others, the fine-tuned and train-only results are mixed 0.5 / 0.5, and the file holds within-feed ranks.

The second pick differs in steps 3 and 5. During the first 79,872 steps its 16 full-view students also train on up to one pseudo-feed per real feed of each batch, about 21.7 million per student, at loss weight 0.3. Its popularity-light pool enters at 0.2 on every feed.

Two parts of the training exist because most train feeds are in production order and validation and test feeds are not. An auxiliary loss uses the production order as a second target (weight 0.5, annealed to 0.1). The cooldown draws 2% of its batches from the 164,118 train feeds of 50 or more candidates that a detector marks as shown in random order.

![The ranker in boxes and arrows. A user token and tokens of the 256 newest history events pass through one self-attention block. Each candidate token adds values from its 16 most similar history items, then passes twice through cross-attention to the user and history and self-attention across the candidates of the feed. A score head gives one score per candidate, and an auxiliary head is used in training only. Width 192 throughout.](docs/figures/architecture.svg)

The seven largest gains are below, each measured against the recipe of its day. A pair is the paired difference on validation / on the holdout. Several of the early gains rest on one run, and [docs/06-what-worked.md](docs/06-what-worked.md) gives all 18 gains with their later checks.

| Change | Validation / holdout | Runs |
|---|---|---|
| Attention from every candidate to the user's history | +0.0055 / no holdout yet | 1 seed, 2 epochs |
| 10 epochs in place of 2 | 0.3863 to 0.3914 / 0.3117 to 0.3167 | 1 seed, first optimiser |
| Fine-tune on the validation days | +0.0040 out of fold / not measurable | 1 early model |
| Co-visitation and popularity counts as inputs of the train stage | +0.0032 / +0.0092 | 1 paired run, 2 epochs |
| Muon with a constant learning rate and a cooldown | +0.0031 / +0.0033 | 3 seeds, 2 epochs |
| Retrieval of the 16 most similar history items, with gated attention | +0.0028 / +0.0038 | 3 seeds, 2 epochs |
| Distillation at teacher weight 0.5, per member | +0.00257 ± 0.00047 / +0.00200 ± 0.00028 | 4 seeds, 10 epochs |

On public, the first 8 students scored 0.394486, +0.001032 over the best click-label file (C3, 0.393454). In the final recipe the fine-tuned branch adds about zero on validation (removing it from W8 cost 0.00009 ± 0.00043), but on public the half-and-half mix is about 0.0005 above either branch alone. One pool of 8 students scored 0.394009 with the fine-tuned branch at weight 0, 0.394486 at 0.5 and 0.393960 at 1 ([docs/03](docs/03-solution.md#fine-tune-on-the-validation-days)).

## Try it without the data

The tests build small synthetic copies of the data, so the code runs without the competition archive. After `uv sync`, five of them go from a teacher file to the pre-submission gate, in pipeline order:

```
export OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12   # thread caps as in docs/10; they matter on a machine with many cores
python -m pytest -q \
  tests/test_training.py::test_dump_and_average \
  tests/test_training.py::test_a_run_writes_its_outputs_and_the_gate_reads_its_inputs \
  tests/test_finetune.py::test_finetune_scores_every_row_and_reports_each_fold \
  tests/test_ensemble.py::test_write_ensemble_stores_five_files_and_never_overwrites \
  tests/test_validation.py::test_gate_passes_a_clean_build
# -s prints the training logs ("loss nan" in them is a gap: the loss is sampled every 50th step)
# --basetemp=DIR keeps every file the tests write
```

It ends with `5 passed` after 10 to 30 seconds on CPU.

What the five tests do:

- The first two write 64 train, 24 validation and 24 test feeds. One trains a member and averages its train scores into a teacher file. The other trains a width-32 student for 39 steps toward half clicks and half a teacher, and reads back its predictions.
- The third fine-tunes a member in 3 folds by user on 60 feeds.
- The last two build a submission file from five members' predictions on 240 validation and 160 test feeds and run the gate's six checks.

The scores mean nothing: inputs and labels are random, and each test builds its own tree, so no stage reads the output of the one before. The run shows that the code works and which files each stage reads and writes.

`python -m pytest` runs the whole suite. Without the data 179 tests pass in about seven minutes and 34 skip: 28 need the stored outputs and 6 a GPU.

A reading order for the code:

- `avito_ranker/metrics.py`: NDCG as the evaluator computes it, with ties scored at their worst.
- `avito_ranker/model/ranker.py`: how the candidates attend to the user's history and to each other, with no position input.
- `avito_ranker/training/loss.py`: the listwise loss and the target that mixes clicks with the teacher.
- `avito_ranker/training/train.py`: one training step, the cooldown and the two ways to use the train split.
- `avito_ranker/training/finetune.py`: folds that hold whole users, so that validation is scored out of fold.
- `avito_ranker/ensemble/build.py`: z-score per feed, pool, mix, with no fitted weight.
- `avito_ranker/validation/gate.py`: the six checks and what makes each one fail or warn.

## Reproducing

The environment is Python 3.12 with torch 2.11.0 (CUDA 13.0 wheels), pinned in `uv.lock`; `uv sync` builds it. The competition archive (17.9 GiB) and the sample submission go into the folder that `configs/paths.toml` names under `data.raw`, the extracted parquet and embedding tables into `data.competition`. Every other location is read from the same file.

### The best file as a recipe

The two picks are one recipe family and tie on the public board (0.3952092 and 0.39526061). This section gives the recipe of the first, FD16P20@m0.4_0.2, which the package covers apart from 13 of the 25 teacher members (below). The second differs in two places. Its 16 full-view students also train on pseudo-feeds, which only the lab trainer `lab/nn_x/w12/pseudokd/train_full_pk.py` does, and its popularity-light pool enters at 0.2 on every feed ([docs/10](docs/10-reproduce.md#gaps-in-the-lineage)).

| Part | What it is | Recipe in `configs/` |
|---|---|---|
| Teacher | 25 rankers trained on clicks; their scores on every train candidate, averaged over 11 members and over all 25 | `recipes/teacher_hard_label.toml`, `recipes/teacher_hard_label_popularity_light.toml`, `recipes/teacher_mean11.toml`, `recipes/teacher_mean25.toml` |
| 8 full-view students | width 192, trained toward half clicks and half the 11-member teacher, seeds 951 to 958 | `recipes/student_wide.toml` |
| 8 full-view students | the same with the 25-member teacher, seeds 951 to 958 | `recipes/student_wide_t25.toml` |
| 4 popularity-light students | one exposure count in place of the 8 popularity inputs, seeds 951 to 954 | `recipes/student_popularity_light.toml` |
| Fine-tune | each student on the two validation days, 5 folds by user and 3 seeds | `recipes/finetune.toml` |
| Ensemble | z-scores per feed; the popularity-light pool at 0.4 on feeds of fewer than 50 candidates and 0.2 on the others; fine-tuned and train-only branches mixed 0.5 / 0.5 | `ensembles/fd16p20_m04_02.toml` |

The commands, in the order they run:

```
python scripts/01_flatten.py
python scripts/02_build_features.py counts covisitation context exposure exploration
python scripts/03_train_encoder.py corpus        # then "train" and "score" for each of the ten encoders
python scripts/02_build_features.py encoder extras
python scripts/04_train_member.py configs/recipes/teacher_hard_label.toml --seed S --member NAME   # teacher members
python scripts/05_build_teacher.py average configs/recipes/teacher_mean11.toml                     # after "dump"; then teacher_mean25.toml
python scripts/04_train_member.py configs/recipes/student_wide.toml --seed S --member NAME         # the 20 students, three recipes
python scripts/07_finetune.py --checkpoint FILE --member NAME
python scripts/08_build_submission.py configs/ensembles/fd16p20_m04_02.toml --write --members-dir artifacts/runs/predictions
python scripts/09_check_submission.py configs/ensembles/fd16p20_m04_02.toml --logs-dir artifacts/runs/logs
```

The run needs the competition data and about 150 GPU-hours. Each run must carry the member name that the ensemble config lists, for example `w5_fd10_kd05w192_s951` for `student_wide.toml` with seed 951, and the commands run inside the environment that `uv sync` builds. [docs/10-reproduce.md](docs/10-reproduce.md) has every step with its command, hardware and run time, and the expected output of the last three.

A fresh run gives a new file of the same recipe, not the submitted file. Thirteen of the 25 teacher members have no package recipe: seven were cooldown re-runs of an older ranker and six use a GRU block. The original teacher files hold their stored train scores, and no stored output is in this repository. Their lab code is under `lab/`. How far a teacher of only the 12 package members moves the score was not measured.

The commands above were run on a machine that holds the outputs of the original runs. From an empty tree three of them need a change: `--refit-proxy` in the first feature step, teacher recipes cut down to the 12 members that can be trained, and a reference pool for the gate. [docs/10](docs/10-reproduce.md#from-a-fresh-checkout) gives each change.

### Checking a built file

The original check rebuilds the first pick from its stored member predictions and runs the pre-submission gate on the rebuild, in about 15 s on CPU. It needs those predictions, which are not in this repository. For a reader it applies after the steps above, with `--members-dir` and `--logs-dir` as in the last two commands there and a reference pool for the gate ([docs/10](docs/10-reproduce.md#from-a-fresh-checkout)).

```
python scripts/08_build_submission.py configs/ensembles/fd16p20_m04_02.toml --write --out-dir rebuild/
python scripts/09_check_submission.py configs/ensembles/fd16p20_m04_02.toml --in-dir rebuild/
```

Among other lines, the first command prints, for the stored members:

```
val NDCG: final 0.40293 | fine-tuned branch 0.40139 | train-only branch 0.40208
sha256 3cf6faccbad1357ae1a1b11999fef8168835d77407684832c7578c832c6e71b3  cand_FD16P20_m04_02kd_r4.parquet
```

The sha256 is that of the submitted file. From the evening of the first day the gate ran on every file offered for submission. The one file built on the data leak, on 12 November, was sent on purpose as a test ([docs/02](docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)).

The gate checks the format, rebuilds the file from its member predictions, compares each member's logged inputs with the list of leaking features, and probes the 117 validation feeds whose user has a later feed. It ends with `GATE: PASS  (FAIL 0, WARN 0)` and exits with code 1 on any failure.

Four more configs in `configs/ensembles/` check other files the same way, with these validation scores:

- `fdx16p20_ps.toml`, pick 2: 0.40286 (sha256 starting `8235c55c`);
- `w8p20.toml`, W8P20: 0.40214;
- `fd16p20.toml`, FD16P20: 0.40271;
- `fd16p20vf.toml`, FD16P20vf (FD16P20 plus 4 popularity-free students, which read no popularity input; public 0.394630): 0.40286.

One more command hashes both stored picks again and re-runs the gate on each. It takes about 8 s, ends with `freeze check passed at <UTC time>` and needs the stored files as well ([docs/11](docs/11-final-picks.md#what-would-change-the-choice) has the dates it is for):

```
python scripts/freeze_check.py configs/ensembles/fd16p20_m04_02.toml configs/ensembles/fdx16p20_ps.toml
```

![A vertical chain of nine stages between the competition data (17.9 GiB zipped) and the submission file: flatten, exploration probability, candidate inputs, teacher rankers, teacher files, students, fine-tune, ensemble and file, pre-submission gate. A column on the right gives hardware and run time in the original run: the CPU stages take seconds to about half an hour, encoder pre-training 5.2 GPU-hours, the 25 teacher rankers 58.0, the 20 students 81.8 and their fine-tunes 6.6.](docs/figures/pipeline.svg)

A training run needs one GPU with at least 55 GiB free: the train split and its inputs stay on the GPU (36.8 GiB for the full view), and the peak while they load is 52.5 GiB.

In the original runs several processes shared a GPU, so their per-run times are upper bounds. They ran on a few NVIDIA data-centre GPUs; the model and the number are not given here.

After the work was done the pipeline was rewritten as the package `avito_ranker/`. The five files checked above rebuild byte for byte from their stored member predictions, and every stage was compared with the stored outputs of the original runs; the full-length trainings were not repeated. [docs/10](docs/10-reproduce.md#how-the-package-was-checked-against-the-original-runs) gives the checks and the gaps in the lineage.

## Repository map

| Path | What it holds |
|---|---|
| `avito_ranker/` | The final pipeline only. One trainer replaces the 26 distinct `train.py` files of the lab. The lab trainer has 89 flags; the ones the final members used survive as values in `configs/recipes/`. |
| `scripts/` | The nine numbered entry points, `freeze_check.py` and `figures/`, the code that draws `docs/figures/`. |
| `configs/` | `paths.toml` is the only place where a data location is written; the variable `AVITO_PATHS` points the package at another copy. `ensembles/` holds the member lists and weights of the two picks and the three earlier candidates; in the lab they were hard-coded in the builder scripts. |
| `tests/` | Runs without the data: 179 tests need nothing, and 34 more need the stored artefacts or a GPU and skip without them ([Try it without the data](#try-it-without-the-data)). |
| `docs/` | The write-up (pages below), with the figures and their data in `docs/figures/`. |
| `docs/records/` | The stored text of the pre-submission gate for five files (`gate-outputs/`). The working records themselves (the decision ledger, the status log, the experiment registry, the coverage map, the freeze protocol and the picks audit of 16 November) are cited by name in the docs and are not published. |
| `experiments/` | One folder per research line, each with a README, the code that produced its numbers and small result tables. [experiments/README.md](experiments/README.md) is the index. |
| `lab/` | A trimmed copy of the working tree: code, written reports and per-run read-outs. Left out of the publication: run logs, launch and queue scripts, arrays and predictions, the working records, samples of the competition data, old model dumps, and other teams' names, personal data and secrets. Device numbers, memory figures and paths of the original machine are redacted. It is an archive to look things up in, not code to run ([lab/README.md](lab/README.md)). |
| `artifacts/` | Not in the repository. It is where the pipeline reads and writes data, predictions, submission files, checkpoints, logs and teacher files (`configs/paths.toml`). |

| Page | What it covers |
|---|---|
| [01 Competition and data](docs/01-competition-and-data.md) | Task, metric, files, calendar, the public / private cut, and every field with Claude's reading of f1 to f5, event types, event sources and category codes |
| [02 What the data showed](docs/02-what-the-data-told-us.md) | What the embeddings and the codes encode and what that bought, and the findings that shaped the model, such as the position-confounded train labels and the popularity inputs that go stale |
| [03 The final solution](docs/03-solution.md) | Inputs, ranker, loss, schedule, distillation, fine-tune, ensemble, costs, known limits |
| [04 Validation](docs/04-validation.md) | The witnesses (validation, the holdout and three samples derived from them), noise, leak incidents, the pre-submission gate, rules written before the read |
| [05 Timeline](docs/05-timeline.md) | The nine days in nine phases, all 40 scored submissions, the board snapshots |
| [06 What worked](docs/06-what-worked.md) | 18 gains, each with its later check |
| [07 What failed](docs/07-what-failed.md) | Twenty failures told in full and the catalogue of 312 ideas |
| [08 State of the art](docs/08-outside-ideas.md) | What wins in comparable competitions, the research state of the art, which outside ideas transferred here, and what the top of this board may use |
| [09 Lessons](docs/09-lessons.md) | Nineteen lessons, each from an incident that cost a day, a submission or a wrong closure |
| [10 Reproducing](docs/10-reproduce.md) | Commands, expected outputs, how the package was checked against the original runs |
| [11 Final picks](docs/11-final-picks.md) | Why these two files, the audit, the risks |
| [12 Why the score stopped here](docs/12-why-the-score-stopped.md) | Noise in the labels, limits of the data, luck in the measurements, ideas closed too strictly and the model the search may have missed, each weighed against the evidence; section 6 lists five ideas that never had a fair run |
| [13 How the work was run](docs/13-how-the-work-was-run.md) | One agent session steered by short messages, what that setup lacked, the harnesses published today and the limits a stronger setup would still meet |
| [Evidence index](docs/evidence.md) | For the 18 changes that worked and the 20 failures: the published file that holds each headline number, and what rests only on unpublished records |
| [Catalogue table](docs/catalogue.csv) | The 312 ideas of page 07 as a CSV, written by `scripts/export_catalogue.py` |
| [Glossary](docs/glossary.md) | The terms and the code names used in `lab/` and the records, such as `G3`, `C3` or `FD16P20`, with their plain meaning |

## What was tried

The catalogue in [docs/07](docs/07-what-failed.md#the-catalogue-by-theme) holds 312 ideas, one row each, with the numbers behind every outcome. Several rows bundle many arms, such as 13 block designs or 95 rule families, so the number of distinct things tried is well above 312.

| Theme | Ideas | Adopted | Superseded | No effect | Hurt | Inconclusive | Not run |
|---|---|---|---|---|---|---|---|
| Baselines, trees and stackers | 15 | 0 | 5 | 7 | 3 | 0 | 0 |
| Neural architecture and capacity | 38 | 9 | 1 | 18 | 7 | 0 | 3 |
| Optimiser, schedule and regularisation | 33 | 4 | 3 | 11 | 14 | 0 | 1 |
| Losses and auxiliary tasks | 20 | 3 | 1 | 6 | 6 | 0 | 4 |
| Inputs and features | 27 | 5 | 0 | 18 | 2 | 1 | 1 |
| Leak-safe statistics and leaks found | 12 | 4 | 1 | 5 | 2 | 0 | 0 |
| Popularity, staleness, display order | 30 | 5 | 1 | 19 | 2 | 2 | 1 |
| Fine-tune on validation | 18 | 5 | 1 | 10 | 0 | 0 | 2 |
| Distillation | 18 | 3 | 0 | 10 | 3 | 1 | 1 |
| Ensembling and pool composition | 22 | 3 | 3 | 13 | 1 | 1 | 1 |
| Post-processing and test-time tricks | 10 | 0 | 0 | 6 | 1 | 0 | 3 |
| Pre-training and embeddings | 12 | 2 | 2 | 5 | 1 | 0 | 2 |
| Large language models | 8 | 0 | 0 | 3 | 3 | 0 | 2 |
| Decoding the anonymised codes | 8 | 1 | 0 | 7 | 0 | 0 | 0 |
| External data and outside research | 6 | 0 | 0 | 5 | 0 | 0 | 1 |
| Leaderboard forensics, errors, segments | 11 | 1 | 0 | 8 | 0 | 2 | 0 |
| Validation protocol and decision rules | 24 | 13 | 4 | 4 | 0 | 3 | 0 |
| All | 312 | 58 | 22 | 155 | 45 | 10 | 22 |

"No effect" means inside the noise of the test that was run, and "not run" means closed by a bound, a probe or a low prior before any training. "Superseded" means that it helped and was later replaced. Half of all ideas did nothing measurable. The largest groups of adopted rows are the validation method and decision rules (13) and the ranker's architecture (9).

Behind the table, [docs/06](docs/06-what-worked.md) tells the gains and [docs/07](docs/07-what-failed.md) twenty failures that changed the method. The next two sections ask which closures may have been wrong.

## Why the score stopped here

The best file is 0.0030 below the leader on public and 0.0028 below place 2. [docs/12](docs/12-why-the-score-stopped.md) weighs five explanations against the records. The six points below follow its sections; the second and third both belong to its section 2. Numbers are on validation unless marked:

- Noise in the labels limits measurement more than the score. The remaining loss looks like the random draw of one click: the hardest 20% of feeds hold 27.5% of the loss, and 27.6% when the labels are re-drawn from the model's own softmax. The scores are calibrated (temperature 1.004 and 1.011 on two halves of the users), and a perfect correction of every detected bias would add +0.00035 ([docs/02](docs/02-what-the-data-told-us.md#where-the-ranking-error-sits)).
- The inputs as given look exhausted. None of 70 attributes told a missed positive from the negatives near it (local AUC 0.492 to 0.504), 1.02% of 11,617 generated features passed the screen against 1.01% of noise features, and apart from the leak, every channel inside the files probed on 18 November reads at most +0.00003 on top of pick 1.
- Scale did not help, and less data hurt. Language-model trunks of 0.6 to 9 billion parameters, 13 block designs, routed experts and depth 7 added nothing, and 15 to 30 epochs cost a member 0.0006 to 0.0016. A pool of 8 students sits 0.00024 below the fitted limit of an infinitely large pool, and removing 20% of the users cost a member 0.0028 (0.0022 on the holdout).
- Luck in measurement changed which files were built and picked. The final files are favourable draws of their recipes, the 17 files of the final family cannot be ordered on public, and on 19 November a re-read that counted shared seed luck as a gain made two closed arms look positive for a few hours ([docs/04](docs/04-validation.md#seeds-shared-by-arm-and-reference)).
- A scan of the closed ideas finds no sign of lost gains, though its bound is loose (up to 13 of 93 rows). The rules had little power for small effects (56% for a true +0.0005 at 8 seeds), but among closed ideas positive reads come at the rate of noise (46%), and ten positive first reads re-tested in a stronger form averaged -0.00001 ± 0.00013 ([closures scan](lab/nn_x/w12/closures_scan/REPORT.md)).
- A better model that the search missed stays the most likely reading of the gap (P 0.64, a judgement of 18 November that was not re-estimated). On 18 November the top 8 places formed a continuum (gaps of at most 0.00098), on 19 November place 2 rose to within 0.0002 of the leader, and distillation showed that a step of 0.001 could come in one day. The records name no recipe of that size.

The label re-draw and calibration tests of the first point show that the model is calibrated. They do not show that 0.40 is a ceiling: a boosted-tree model at 0.370 passes the same label-draw test. The leader's margin equals what a model gains whose logits are 2.5% larger and still calibrated, so information that was not found would show as sharper probabilities.

The leak in the data does not explain the gap: an overlay built on the same-user later snapshots scored 0.392819 on public against 0.392820 for its base file ([docs/02](docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)). Fresh popularity from outside the files, such as crawled view counters, is the only input gap of the right size (P 0.03, the same judgement). I read it as grey under the rules. It also contains the user's own later view, and it was not pursued.

[docs/08](docs/08-outside-ideas.md#what-the-top-of-this-board-may-use) lists every hypothesis about the top of the board with its probe numbers. Two options with a positive expected value were left open: a merge with a team of another model family, to be settled by 2026-12-01, and teacher-only distillation on the validation and test feeds, worth about +0.0001 on paper. Five ideas that never had a fair training run follow.

## What to try next

This section lists five ideas that never had a [fair training run](docs/12-why-the-score-stopped.md#what-a-fair-test-costs). It is a judgement Claude made on 19 November, and no experiment stands behind it.

P is the subjective probability that an idea is worth +0.001 NDCG or more for a final file.

| Idea | Why it could matter | P |
|---|---|---|
| Next-event training of the history encoder inside the ranker | The ranker learns from about 5.5 million clicks; the histories hold 463.6 million events, and each says what a user did next. | 0.10 |
| Function matching: random variants of each feed, re-labelled by the teacher | Variants act as extra data, and training on 80% of the users cost a member 0.0028 NDCG on validation. Only a flawed form ran. | 0.07 to 0.10 |
| Hindsight labels: candidates the user opened later, as soft positives in training | Cleaner labels paid before, in distillation. Inputs stay as they are, so by this judgement it is not the [leak](docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak), but my rule against later snapshots forbids it as written. | 0.03 |
| A learned model of each item's audience | The ranker never looks from a candidate to the users who viewed it; that side enters only as co-visitation counts. | 0.05 |
| Click-label closures re-tested on students | Width 192 was closed under click labels and adopted under distillation; a learned position bias, 16 ranking losses, a single-stream transformer and 13 block designs never ran with students. | 0.05 for the group |

The chance that at least one of the five is worth +0.001 or more is put at about 0.25 (the ideas overlap), and none is expected to be the whole 0.003 to the leader.

Other ideas sound missed but ran and read no effect, such as a propensity head for the display position, earlier skipped items as negative feedback, and fresh exposure counts from other users' validation and test feeds. [docs/12](docs/12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) has the evidence for and against each, and the list of ideas that ran.

## What limited everyone and how the data could be better

These limits held for every team. Claude measured each of them in the data, the rules or the board, and the links give the evidence.

- The label is one unnamed bit per candidate, and most labelled feeds have a single positive, so each feed carries little signal ([what one feed holds](#what-one-feed-holds)).
- Display position drives clicks and is hidden where it counts: validation and test carry no positions, and the random-order train feeds carry no flag. In those feeds the first slot is clicked at 2.51% against a base rate of 1.15%, and ordering by position gains +0.0046 NDCG over a random order. Claude's first detector is blind below 50 candidates, and about 10% of its confident feeds were in production order, 2,294 of them in the holdout ([docs/02](docs/02-what-the-data-told-us.md#random-order-feeds-inside-train)).
- Inputs built from train are staler on test than on any labelled data: the calendar in [the data section](#the-data-and-what-claude-found-in-it) gives the mean age of the popularity inputs per set. In input age validation and public do not overlap: the 90th percentile over validation feeds is 7.42 days, the 10th percentile over public feeds 8.17 ([docs/01](docs/01-competition-and-data.md#time-lag)).
- The fields are anonymous and the codes undocumented: f1 to f5, the event types and the sources come without a meaning, and the history filter is not given. Claude's decoding is in [the data section](#categories-event-types-and-sources).
- History events have no timestamps, so recency can be counted only in events, and popularity cannot be cut exactly at the request time. On holdout rows 55.6% of the popularity count mass comes from other users' snapshots taken after the feed, against 1.9% on validation ([docs/04](docs/04-validation.md#the-witnesses)).
- The data holds an undocumented label leak. Over 571,923 pairs of consecutive train feeds of one user, 92.1% of the earlier feed's positives appear in the later feed's history. Pushing such items up gained +0.00454 on validation and nothing on public. The inference is that the affected test feeds are not scored or have no positives; the organisers have not said ([docs/02](docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)).
- The labelled samples are small. Validation is 14,423 feeds on two weekdays, where one trained ranker against another cannot show an effect below 0.0029 at 80% power. Public is the earliest 30% of test by time, 8,562 feeds all on a weekend ([docs/04](docs/04-validation.md#noise)).
- There is no item content: no titles, text, images, prices or seller fields. 42.8% of public and 45.6% of private candidate rows hold an item that occurs in no history ([docs/02](docs/02-what-the-data-told-us.md#whose-vectors-they-are), [docs/08](docs/08-outside-ideas.md#signals-this-data-does-not-carry)).
- The rules leave gaps. The competition page words the normaliser of the metric as IDCG@30 while the evaluator uses the full list, and NDCG@30 on validation would have predicted public about 0.06 too low. The calendar has days with almost no feeds that the organisers' readme does not mention, the baselines came without scores, and the rules do not settle whether one final file counts or the better of two ([metric](docs/01-competition-and-data.md#no-cut-off-at-position-30), [limits and dates](docs/01-competition-and-data.md#submission-limits-and-dates), [calendar](docs/01-competition-and-data.md#calendar-of-the-data), [two finals](docs/11-final-picks.md#why-pick-2)).

### Proposals for organisers of such competitions

These proposals are my view of what would have helped.

1. Publish a data card that defines every field: f1 to f5 with their windows, smoothing and denominators, dictionaries of event types, sources and categories, the history filter and its cap, how a candidate list was cut, what user 0 means and how the embeddings were made.
2. Give every history event a timestamp, or its age at request time.
3. Flag the random-order feeds in train, and publish display positions for the shuffled feeds of the labelled splits.
4. Build test inputs at the same lag as training inputs. Ship unlabelled exposure logs for the validation and test days, or item statistics computed at request time, and cut public and private with one lag distribution. The organisers' own f1 to f5 did not drift between validation and test (population stability index at most 0.007).
5. Name the target actions and give graded labels (card open, favourite, contact, purchase), scored with graded gains or as several targets.
6. Use a clean user-level design. Publish no snapshot that contains the outcome of a scored feed, or keep one scored feed per user, and document how affected feeds are scored.
7. Draw the public part at random by user and day, make it larger, and show an interval next to each score. On 2026-11-19, places 6 to 10 lay within 0.00095 of each other, less than one paired SE between files of different teams (put at 0.0012 to 0.0018).
8. Provide a larger labelled validation set in test format that follows the test calendar, weekends included. The two files chosen as best on the 14,423 weekday feeds on 2026-11-12 landed 0.0010 and 0.0009 below their public forecast.
9. Ship item content or documented content embeddings: text vectors from a named model, price, region, seller type.
10. Publish baseline scores, word the metric as the evaluator computes it, document the real calendar and its outages, and state how final files count.

## How this was built

The research, code, experiments, analysis and documentation in this repository, the package `avito_ranker/` and this README included, were done by Claude Code (versions 2.1.303 on 11 November to 2.1.325 on 19 November) running Claude Opus 5.5 in ultracode mode, which lets a session write and launch multi-agent workflows, at the xhigh reasoning effort. [docs/13](docs/13-how-the-work-was-run.md) has the full account.

One lead session ran for the nine days, chose the experiments and kept the records. It launched workflows in which parallel agents searched, built, ran, reviewed and tried to refute results, and after each request it waited for the next message.

I wrote no code, ran no experiment and analysed no result. I steered the session by chat, decided a few questions and made the uploads, which the platform accepts only from a person. I:

- sent about 110 chat messages with a direction or a question, with a median length of 22 words, and about 50 more that only asked for status or passed a public score back;
- uploaded the files, at most 5 per platform day, which rolls over at 21:00 UTC (a UTC date in the figure at the top can therefore show 6 or 7), and passed the public scores back;
- decided a handful of questions: which GPUs the work could use, the two final picks, the license and the visibility of this repository;
- gave short directions, for example to look at tuned CatBoost, larger language models, stacking and pre-training on histories, to train the final files on all data, not to reject an idea on the holdout alone, and to build [try files](docs/glossary.md) from ideas that had failed the strict rule, or that had no rule written for them; both picks began as try files.

What this was not:

- an unattended loop that proposes, runs and selects experiments on its own schedule;
- a population of candidate programs evolved against an evaluator, as in OpenEvolve, an open-source implementation of AlphaEvolve ([next section](#my-view-on-competitions-like-this));
- a group of models: no model of another vendor took part or reviewed;
- a process that reviews itself: agents checked results inside a workflow, and the two looks back over days of decisions (15 and 19 November) each followed a message from me.

![Two box-and-arrow diagrams, one above the other. Top, how this entry was run: a loop from the person, who sends a short request (about 110 in nine days), to the lead session (one session, its context summarised 21 times, a status log and a decision ledger as memory), to about 50 multi-agent workflows, to experiments on GPUs, to results and options written down, and back to the person, who uploads up to 5 files a day and passes the public score back. A marker in the person's box says that the loop waits there for the next message. Bottom, what an automated harness would add: a closed loop of five steps inside a frame, propose (checked against the catalogue of closed ideas), run (with a noise probe), evaluate (on validation and the holdout, under a rule written first), keep or drop, and archive, from which the next round starts without a message. Outside the frame stand the person, who sets the goal and the budget and uploads the files, a second model as critic, and a scheduled look back over the ledger.](docs/figures/work-loop.svg)

The upper panel is this entry; the lower panel is a design from [docs/13](docs/13-how-the-work-was-run.md#what-a-stronger-setup-would-do-here) that did not run.

Comparisons are paired per-feed differences with a standard error; the single-run gains of the first day were confirmed only by the removal runs of 13 November. The rule for reading an experiment was written with a UTC time stamp before its first run, and a choice made after the fact was labelled as such. From 2026-11-14 the rules and their outcomes were kept in one decision ledger, a working record that is not published.

One rule was wrong for a whole family of changes. The holdout, whose popularity inputs are 1.75 days old against 9.34 on public, read two popularity-light changes at -0.00063 and -0.00073 where public gave +0.00052 and +0.00063. From 2026-11-15, after my message, the holdout kept no vote on such changes ([docs/04](docs/04-validation.md#rules-written-before-the-read)).

The scale, from the records:

- nine calendar days, 11 to 19 November, from the start of work (2026-11-11, about 02:19 UTC) to the final picks (pick 1 on 2026-11-18 at 02:33 UTC, pick 2 on 2026-11-19 at 03:32 UTC), with 40 scored files ([docs/05](docs/05-timeline.md));
- 312 catalogued ideas ([what was tried](#what-was-tried)) and a scan of 211 technique rows from about 120 public sources ([docs/08](docs/08-outside-ideas.md#how-good-the-priors-were));
- about 50 multi-agent workflows named in the status log, the largest with 20 to about 40 agents;
- 151.6 GPU-hours for the pipeline of pick 1 from raw data and 29.7 more for the 16 pseudo-feed students of pick 2 ([docs/03](docs/03-solution.md#costs)). The work ran on a few NVIDIA data-centre GPUs, whose model and number are not given here, and its total GPU time was not recorded.

## My view on competitions like this

This section is my opinion, not a finding.

Such competitions come with debatable data and built-in limits: anonymous, undescribed fields, a label that stands in for what the business wants, splits that differ in ways each team must discover, and a public board too small to rank the top places. Once the known tricks are applied, the rest is a race of fixes at the level of noise.

This entry was not an automated search. It was one agent session, steered by short messages, that stopped after each request. A stronger setup would likely have scored higher, and its parts exist (sources checked on 19 and 20 November 2026):

- an evolutionary search over recipes, as in [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve), an open-source implementation of [AlphaEvolve](https://cloud.google.com/blog/products/ai-machine-learning/alphaevolve-is-available-for-everyone);
- a loop that trains, keeps or discards a change and repeats without a person, as in [autoresearch](https://github.com/karpathy/autoresearch/blob/master/README.md);
- a second agent that takes part or reviews, such as Codex through the [Codex plugin for Claude Code](https://github.com/openai/codex-plugin-cc);
- a more capable model, and a process that studies its own mistakes.

Such a setup can research every day until the deadline and climb the board step by step, with a person only uploading. So the outcome will more and more show who runs the better harness, on more compute, for longer, and less who had the better idea. That leaves me little reason to enter without such a harness, and it makes the final table a weak measure of skill.

The records and the published reports add two cautions. Public cannot separate near copies below 0.0011 NDCG, so a loop that keeps each day's best public score would mostly climb noise. And in the reports checked, the one gold medal in a live competition was on a language-model task and is reported in a social-media thread, and the one first place came from a workflow guided by a person ([docs/13](docs/13-how-the-work-was-run.md#what-these-reports-do-not-show)).

What I think is worth keeping is the part a reader can reuse: the reading of the data, the validation method and the negative results, which this repository documents.

## License

The code and the documents in this repository are released under the MIT license ([LICENSE](LICENSE)). The competition data is not included: it belongs to the organiser, and its use is subject to the competition's terms.
