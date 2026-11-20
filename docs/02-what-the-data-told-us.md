# What the data showed

In brief:

- Train feeds were shown in ranked order, validation and test in random order: a tree model fitted on train scored 0.3301 NDCG on validation, below f1 alone (0.3365).
- About 13% of train feeds with 50 or more candidates were also in random order, unflagged; they became the holdout and a training pool.
- Popularity inputs are 9 to 11 days old on test against 1.75 on the holdout; both final files mix in a popularity-light view that keeps one exposure count in their place.
- A same-user leak gained 0.00454 on validation and nothing on public; most remaining error sits inside categories.

Scores are NDCG (normalised discounted cumulative gain), the competition metric, defined in [competition and data](01-competition-and-data.md). Splits, notation and uncertainty conventions follow [validation](04-validation.md), and terms are in the [glossary](glossary.md). The scripts and reports behind each finding are listed at the [end of the page](#scripts-and-records).

## Where the analysis started

The analysis began in the early hours of 11 November, the first day of work (day 1 below). Five studies reported that morning, in `lab/reports/`: labels and display position (`eda_labels_position.md`), the shift between splits (`eda_shift_overlap.md`), item and user statistics (`eda_collab_stats.md`), the embeddings (`eda_embeddings.md`) and the history snapshots (`eda_history_leak.md`). The plan for the first model was written from them ([timeline](05-timeline.md)).

The first question was how train differs from the two scored splits. A classifier on feed-level statistics told train from validation with an area under the ROC curve (AUC) of 0.925, and validation from test with only 0.524 ([competition and data](01-competition-and-data.md)). Validation and test are one kind of traffic, and train is another.

The first surprise was a baseline. A LightGBM ranker fitted on 60,000 train feeds, with the organisers' anonymous features f1 to f5, their within-feed ranks and the category, scored 0.3301 NDCG on validation. That is 0.0063 ± 0.0013 below f1 alone (0.3365, ties broken at random). Fitted to train labels, the features lost value.

## Train labels come from a ranked list

In train, where a listing was shown decides most of whether it was clicked ([the position curve](01-competition-and-data.md#candidate-order)). Validation and test were shown in random order. The clearest damage is to f4, the age of the listing.

Cut by f4 decile, the freshest tenth of candidates has 1.22 times the mean positive rate in train and 0.46 times in validation. The production ranker shows fresh listings higher (mean position 65 for that decile, about 90 for the middle ones), and a top slot raises the click rate five to ten times. From position 50 on, the same decile converts at 0.53 times the mean in train, which is the validation picture.

A model fitted on train learns that fresh is good, and f4 is the organiser feature the neural ranker depends on most. Zeroing it at inference cost 0.0138 validation NDCG, averaged over two 2-epoch rankers of the final click-label recipe (rankers trained on clicks alone). Zeroing f5 cost 0.005, and f1 and f3 0.003 each.

### Correcting for the display order

The first fixes were tried on tree models. XGBoost was given `pos` (the display position) and the rank from the end of the feed as training-only features, fixed at mid-list values at inference (position about 20, rank from the end about 50). That added 0.0042 ± 0.0012 validation NDCG (0.3652 against 0.3611, 200,000 train feeds). Fixing the position at 0 cost 0.005 to 0.016, because slot 0 is where the relation is inverted.

In the neural ranker the same idea, an additive position tower, did nothing (0.3736 validation with the tower, 0.3739 without). What worked there was a second target: an auxiliary head trained with a listwise loss toward softmax(-pos / 10), at weight 0.5 annealed to 0.1 at the end of training and dropped at inference. Removing it cost 0.0022 NDCG on validation and 0.0023 on the holdout (next section) in a 2-epoch ablation. `pos` is never an input.

Train labels were still worth more than validation labels alone. The first neural ensemble trained only on train scored 0.3806 on validation, and the best tree model fitted on train 0.3700. No single LightGBM fitted on validation alone passed 0.367, and a z-score mix of 12 tree models fitted on it reached 0.3705. The validation labels enter at the end, in a 5-fold fine-tune by user ([solution](03-solution.md#fine-tune-on-the-validation-days)).

## Random-order feeds inside train

About 13% of the train feeds with 50 or more candidates were themselves shown in random order. The organisers do not say so. The closest precedent is the Expedia hotel-search competition of 2013, whose data carried a random-order flag. Here the flag had to be inferred.

### How the detector works

A tree model learns to predict the within-feed rank of `pos` from the features, on 200,000 feeds. It is a weak proxy of the production ranker (mean per-feed Spearman 0.195), which is enough. For each other feed, z = Spearman(prediction, pos) × sqrt(M - 1), where M is the number of candidates. Under a random order z is standard normal.

For feeds of 50 or more candidates, z fits a mixture of two components, one of them exactly N(0, 1), with weight 12 to 14% up to 250 candidates and 18.6% above. A second statistic that uses no feature separates the same feeds. Random-order feeds have 1.4 to 2.4% of their positions missing (listings removed because already viewed), production feeds 5.6 to 8.7%.

The two together give each feed a probability of having been shown in random order, its exploration probability. The estimate is 206,000 of the 1.55M train feeds with 50 or more candidates, of which 89,794 have a probability above 0.9.

### What these feeds look like

| | Train, probability above 0.9 | Train, probability below 0.05 | Validation |
|---|---|---|---|
| feeds compared (all with 50 or more candidates) | 89,794 | 89,794 | 8,530 |
| positive rate per candidate | 1.15% | 1.60% | 1.18% |
| positive rate at rank 0 / 10 / 100 by `pos` | 2.51 / 1.43 / 1.02% | 9.16 / 2.62 / 1.17% | no positions |
| NDCG gained by ordering on `pos` over a random order | +0.0046 | +0.0651 | |
| positive rate of the freshest f4 decile, relative to the mean | 0.46 | 1.03 | 0.46 |

The f4 row is partly circular, since the proxy model uses f4. The position curve and the base rate are independent of it. The share of random-order feeds is flat across days (0.134 to 0.153) and hours, and train feeds of validation and test users are no more often in random order than others, so the randomisation is per request.

Per feed, these are better training data. At equal size, 50,000 of them trained a better tree model than 50,000 production feeds: +0.0055 to +0.0066 validation NDCG, with a standard error (SE) of 0.0015.

### How the ranker uses them

- 30,000 feeds with probability above 0.9 and 50 or more candidates became the holdout: a second labelled set in the test order, used to select checkpoints without touching validation labels ([validation](04-validation.md#the-witnesses)).
- Feeds with probability above 0.5 and 50 or more candidates form the random-order pool, 164,118 feeds (134,118 when the holdout is kept out). In the cooldown, the last 20% of training steps, each step is a batch from this pool with probability 0.02. Without these batches a ranker ties on validation as a whole and loses 0.0023 (z 4.5) on the validation feeds whose positives are all cold items, never seen as train candidates. [Solution](03-solution.md#optimiser-and-schedule) has the dose.
- The auxiliary loss toward the production order is weighted by one minus the probability. Each feed also carries a domain embedding mixed by it, and validation and test rows get its random-order row. The embedding's own worth is within noise ([solution](03-solution.md#the-ranker)).

Two other uses of the pool were dropped. A separate second training stage on it gave +0.0012 on validation on day 1 and -0.0012 once the schedule had a cooldown. Up-weighting random-order feeds through the whole run cost about 0.001.

### A second detector for short feeds

The first detector does not work below 50 candidates, which is half of train: the two components are at most 1.6 standard deviations apart there. On 2026-11-14 a second detector was built for all sizes. It uses no positions: it is a classifier of set summaries trained to tell validation and test feeds from train feeds. Its confident feeds passed the position test in all 20 size bins (z mean +0.03 to +0.12, standard deviation 1.0 to 1.09).

It found 292,000 feeds with probability above 0.5, 105,000 of them with fewer than 50 candidates. It also showed that about 10% of the first detector's feeds above 0.9 were in production order, 2,294 of them in the holdout. As a cooldown pool it gave +0.00025 ± 0.00035 on validation per member at a 2% share and -0.00161 ± 0.00043 at 10%, so the pool stayed as it was.

## Cold items and stale popularity

The share of candidate rows whose item was a train candidate falls by about 2 points per day after train ends ([table by day](01-competition-and-data.md#time-lag)). Cold items are clicked less: 1.37% positive on validation against 1.69% for warm ones. Every statistic keyed on an item id thins out with the lag.

### Which item statistics survive the lag

This was measured on day 1 by blending the within-feed rank of each statistic with the rank of f1 (`lab/eda/shift_overlap/s03c_signals_fair.py`). Each change is against f1 alone with ties broken at random, at blend weight 0.25, with a paired SE of 0.0005 to 0.0010.

| Item statistic from train | Change of validation NDCG |
|---|---|
| click-through rate as a candidate, smoothed | -0.0016 |
| number of times shown as a candidate | +0.0005 |
| number of appearances in user histories | +0.0032 |
| the same, views only | +0.0035 |
| appearances in histories, counted only up to 7 September (a 15-day gap) | +0.0030 |

Train click-through rates carry the position bias of production and are worth nothing on shuffled traffic. History counts hold their value when the gap grows from 3 to 15 days and their coverage falls from 58.7% to 32.6% of validation rows.

So the ranker's 8 popularity inputs are 3 co-visitation scores (how often the candidate occurs near the feed's history items in other users' histories) and 5 history counts. All are leave-user-out, which means they skip the later snapshots of the row's own user ([the leak below](#the-same-user-later-snapshot-leak)), and none is a click rate.

For a user with feeds on 8 and 12 September, the counts of the 8 September row leave out that user's 12 September snapshot, which holds the 8 September clicks. Other users' later snapshots stay in.

### Staleness and the holdout

The counts still go stale. What that did to the samples used to judge changes was found only on 2026-11-15. The events they count are almost all train data (3,151,208 of the 3,194,226 feeds of the three splits), and train ends on 19 September. Staleness counts the days since a warm item was last a train candidate. The worth of the 8 popularity inputs falls with it (2-epoch rankers with and without them):

| Set | Mean staleness of warm items | NDCG lost without the popularity inputs |
|---|---|---|
| holdout | 1.75 days | 0.0074 |
| validation | 5.69 days | 0.0023 |
| stale validation: validation replayed with inputs as stale as on private rows | 11.23 days on private | 0.0009 |

The public part sits at 9.34 days. The calendar shows where each set lies and how old its inputs are:

![Calendar of the data from 6 to 29 September 2026: train on 6 to 14 and 17 to 19 September, validation on Tuesday 22 and Thursday 24, test on 26 to 29 with the public/private cut at 11:32 Moscow time on Sunday 27. Below it, the mean age of the popularity inputs in each set: 1.75 days on the holdout, 5.69 on validation, 9.34 on the public part and 11.23 on the private part.](figures/data-timeline.svg)

The rankers had never seen these inputs that stale in training. The sample with the freshest inputs, the holdout, was the one used to veto changes.

### The popularity-light view

The popularity-light view had been built two days earlier, on 2026-11-13, because popularity was the input family most likely to look different on test. It replaces the 8 inputs with one clipped exposure count, log1p(min(n, 50)), where n is the number of times the item was shown as a train candidate ([solution](03-solution.md#the-two-input-views)).

To measure what staleness costs each view, the holdout's popularity inputs were rebuilt at the lag of private rows. The rebuilt inputs pass the check against test rows on the 13,424 feeds from 13 September on and fail it on all 30,000 ([validation](04-validation.md#the-witnesses)). Full-view students are distilled rankers of width 192 that read all 34 inputs ([solution](03-solution.md#distillation)). Change of holdout NDCG under the rebuilt inputs:

| Pool | 13,424 feeds from 13 September | All 30,000 feeds |
|---|---|---|
| 8 full-view students | -0.0033 ± 0.0008 | -0.0042 ± 0.0006 |
| 4 popularity-light students | -0.00049 ± 0.00038 | +0.00015 ± 0.00033 |

With its fresh inputs, the holdout gave the wrong sign for both popularity-light additions scored on the board: -0.00073 and -0.00063, where public gave +0.00063 and +0.00052. [Validation](04-validation.md#rules-written-before-the-read) tells how that ended its veto on such changes.

Both final files give popularity-light students a 0.2 share of the pool, set on 2026-11-14 before any file with it was scored ([solution](03-solution.md#the-two-input-views)). Validation and public together put its gain at +0.00050 ± 0.00025. Pick 1 raises the share to 0.4 on feeds of fewer than 50 candidates. That change was scored on 2026-11-17 without a rule written for it. It read +0.00022 ± 0.00024 on validation and +0.00032 on public, about one paired standard error ([validation](04-validation.md#the-witnesses)).

Two other repairs, a cooldown on a stale view of the inputs and a popularity-free view, are in [solution](03-solution.md#the-two-input-views). Neither was adopted into the pool. Boosting cold items by a fixed offset won 0.008 to 0.018 on feeds whose positives are all cold and lost more on the others.

## The same-user later-snapshot leak

A history is a snapshot at the time of the request, so what a user clicks in one feed is in the history of that user's next feed. This was found on day 1 by lining up consecutive feeds of the same user (`02_snapshot.py` in [validation-and-leaks](../experiments/validation-and-leaks/)). Over 571,923 train pairs, 92.1% of the earlier feed's positives appear in the later feed's history, against 0.40% of its negatives. An item found there is a positive with probability 0.847.

The later user vector leaks too. Scoring a feed's candidates by cosine to the user vector of the next feed gives 0.3939 NDCG on 40,000 train triples, against 0.3630 with the feed's own vector.

### The leak in the training inputs

Of the train feeds, 18% have a later feed of the same user, so any count over "all histories" contains the labels of train rows. Tree models trained on such counts gained 0.025 to 0.052 NDCG on the 117 validation feeds whose user has a later feed in validation or test. They had learned from 1.27M leaky train rows that a count increment means a click.

A single-feature audit on those 117 feeds found two more contaminated families. One was flags built from later histories. The other was ALS item factors (a matrix factorisation of user-item pairs) trained on pairs that included the user's own later snapshots, which scored 0.583 NDCG on the 117 feeds and 0.329 elsewhere. With these removed, the day-1 best read about 0.385 on validation in place of 0.3869.

### A boost on test that scored nothing

The leak also looked like a free gain on test. On validation, 184 candidates in 110 feeds are in a later history of the same user, and 172 of them are positive (93.5%).

Test offers less, because only later test feeds can be used. Only 108 test feeds have a later feed of the same user (59 on the public part, 49 on the private part). In 103 of them (57 public, 46 private) at least one candidate occurs in that later history: 183 candidates, 108 of them public.

At 0.61 NDCG gained per affected feed, the expected gain was +0.0041 on the public board and +0.0014 on the private one. The day-1 file pushed these 183 items to the top of their feeds, from a mean rank of 31 to 1.8. It scored 0.3802 on public, the same as the file without the boost.

### The second test, 12 November

The leak came back on 2026-11-12, when a new leader appeared at 0.398256, 0.0054 above the team's best file. A model at the level of the board's second place at the time, plus this leak, would have explained the jump. The day-1 reading was also in doubt, since the 0.3802 on record could have been the board's best-score display.

So the test was built on purpose. The base was the best click-label pool (20 click-label rankers, G3 in the lab records, public 0.392820), with the flagged items placed first. On validation this overlay added 0.00454 ± 0.00047 (0.40032 to 0.40486), and the 117 feeds whose user has a later feed went from 0.383 to 0.943. That predicted about 0.3967 on public.

On test the overlay lifted the same 108 public items in 57 feeds to the top of their feeds, from a median rank of 21. The signal had been checked in the test inputs beforehand: the flagged items sit in the next history with the same event mix as on validation, 86% as a view opened from the feed against 74%.

The rule written before the submission was to drop the overlay if public came within ±0.0005 of the base file. The overlay scored 0.392819 on public. The base file shows 0.3928199 on the board (0.392820 rounded), so the two agree to six decimals.

Moving 108 items to the top of 57 feeds would not leave the mean unchanged to that precision if those feeds were scored with their labels. So they are either not scored or have no positives left, and a feed without positives scores 1 in any order. Which of the two the organisers do was never learned. The leak is in the data and worth nothing on the board, so the leader's margin had to come from elsewhere.

### Rules that followed

From then on, every input had to meet these rules:

- Nothing may come from a later snapshot of the same user: no later-history flag, no later user vector, no label repair from it.
- Counts over histories are leave-user-out. Occurrences that come from later snapshots of the row's own user are removed, in the same way for train, validation and test rows.
- Validation features never read test histories of the same user (92 users, 94 validation feeds).
- User 0, the pooled bucket of visitors who were not logged in, is never linked across feeds.
- Contaminated feature and prediction files went to quarantine folders and are never read.
- Every candidate file passes a gate ([validation](04-validation.md#the-pre-submission-gate)) whose leak probe compares the gain over a reference on the 117 feeds with the gain on all other validation feeds. A z above 3 fails the file and 2 to 3 warns. For the all-data files the reference is the W8P20 pool: pick 1 reads +0.00613 ± 0.00469 and pick 2 +0.00308 ± 0.00365, both below the warning level.

No student was trained on later-snapshot targets; the idea was considered twice. Negatives the user later interacted with are 0.40% of negatives and count as true negatives under the metric. A probe teacher given later-snapshot events as extra inputs was built and gained 0.037 ± 0.0025 NDCG on the holdout feeds it covers. All of it came from events after the first interaction with a candidate, which makes it a copy of the label.

[Section 6 of 12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) lists one use of those later events that never ran: soft positives in the training target alone (idea 3). It needs a decision on the first rule above before any run, and the page gives it as a judgement with no experiment behind it.

## What the embeddings are

That the vectors are unit vectors multiplied by 256 and rounded to int8 is an inference from the norms. The mean squared norm is 65,546.48 in train, where 65,536 + 128/12 = 65,546.67 is predicted for that scheme. The norm carries no information (0.2895 validation NDCG as a score, random 0.2890), and rounding moves a cosine by about 1.6e-4, so nothing is gained by undoing it.

### Whose vectors they are

The team judged them to be the two towers of Avito's own recommender, with a subjective probability of 0.94. A two-tower model embeds users and items with separate networks into one space. Here the user vector lives in the item space with the same quantisation and is recomputed at every request: 0.08% of consecutive same-user train feeds have identical vectors. Probes are in [codes-and-embeddings](../experiments/codes-and-embeddings/), and the full report is `lab/reports/EMBEDDINGS_1113.md`.

The vectors are not a Matryoshka prefix of a public text model, the kind of model whose first dimensions form a usable shorter vector. The Spearman correlation of per-dimension spread with dimension index is -0.06, against -0.52 and -0.83 for two such models cut to 128.

The data has no titles, descriptions or images, so no public embedding model can be applied to it. A frozen public text encoder inside Avito's tower cannot be excluded, and it would change nothing that can be done with the data.

### What an item vector encodes

Probes on 1.5M train items find the category (linear probe 98.3%) and a region component that adds to it (f2 partial R² 0.346 given the category). They find almost nothing about how the item performs (R² 0.0034 for its smoothed train click rate). The click signal is in how a candidate relates to this user. Single scores on validation, ties broken at random (one draw; the exact expectation for f1 is 0.3365):

| Score | Validation NDCG |
|---|---|
| random order | 0.2890 |
| mean cosine to the whole history | 0.3294 |
| f1 | 0.3366 |
| dot product of user and item vectors | 0.3432 |
| largest cosine to a history item | 0.3436 |
| mean cosine to the last 5 history items | 0.3474 |
| recency-weighted mean cosine to the history, half-life 3 to 5 events | 0.3493 |
| attention scorer with about 330 parameters, fitted on train | 0.3584 |

The newest history events carry most of the signal. This is why the ranker is built around attention from each candidate to the history events, with a cosine bias and a retrieval of the 16 most similar history items ([solution](03-solution.md#the-ranker)). Taking the history attention out of the day-1 ranker cost 0.0055 validation NDCG (0.3681 against 0.3736).

### What the vectors meant for the code

- Raw cosine is the native metric. Whitening the vectors cost 0.0440 ± 0.0018 NDCG in scoring by the user vector, and centring cost 0.0046 ± 0.0007.
- History events without a vector (14.2% on validation, 33% for event type 9) get a learned "no vector" embedding and are left out of the similarity retrieval. They are never pooled as zero vectors.
- The user vector stays a separate input. It is predictable from the history (R² 0.594) and still adds 0.0022 ± 0.0008 to history similarity in an unfitted sum.
- Between splits, 77 to 83% of the vectors of shared items are bit-identical, depending on the weighting, with a mean cosine of 0.999. The changed ones are clicked as often as the unchanged (lift 1.05, interval 0.94 to 1.17), so item statistics can be joined across splits by `item_id`.

No processing of the vectors added score. On 2026-11-13 five treatments were declared in advance, each read as an addition to the scores of the best click-label pool with a bar of +0.0005 and 2 SE. They were a drift flag, whitened or centred similarity, projection on the mean direction, near-duplicates of acted items, and interactions of the user-item cosine.

None passed. The best, the projection, read +0.00045 ± 0.00022 on validation and -0.00004 on the holdout. A sixth, a match in the region subspace, read -0.00008 ± 0.00030.

## What the codes mean and what that bought

The meanings tabulated in [competition and data](01-competition-and-data.md) were worked out on 2026-11-13 from behaviour in the data. Scripts and mapping tables are in [codes-and-embeddings](../experiments/codes-and-embeddings/), and the full report is `lab/reports/CODES_DECODING_1113.md`.

Event types and sources are numbered by frequency: sorted by their share of train events, none of the 45 pairs of types and none of the 435 pairs of sources is out of order. Their meaning came from four kinds of evidence:

- The next snapshot. A clicked item's oldest new event there is a view with source 1 in 83% of well-aligned pairs. When the first new event is such a view, the item was a positive in 88.7% of cases. The rate is 89.7% for a favourite with source 1, and 1.6% and 0.7% for a view from source 2 or 4. That defines the target as an engagement started from this feed, and source 1 as the feed.
- Which types carry a source at all (1, 4 and 7). That marks them as actions on a surface.
- Category gating. Types 5, 6 and 8 are absent from 15 category codes over about 7.9M events (3 stray events in services): all of real estate, cars, trucks, jobs, services, ready business, hotels and shift gigs. In goods categories they are 8 to 29% of events. A block absent wherever nothing can be shipped is delivery. Messaging, the earlier guess, does not fit: chat on Avito exists in every vertical (a product fact not checked in the data).
- The guest bucket. Types 2, 5, 8 and 9 never occur for user 0, and type 10 is in 86 to 88% of user-0 feeds that have a history, against 3.7% of logged-in feeds.

Category codes are not frequency ranks. They follow the order of Avito's public leaf category ids. Neighbouring codes fall in the same embedding cluster in 20 of 51 pairs, where chance gives 7.5 ± 2.3, and the grouping into verticals agrees with the transitions in histories (adjusted Rand index 0.55).

### The decoding added no score

Six features that need the decoding were tested under a rule written in advance, as additions to the scores of the best click-label pool. They were the user's history share in the candidate's vertical, the same outside the leaf, a match with the vertical of the newest event, vertical recency, delivery activity crossed with deliverable categories, and a mask of the hard zeros. None passed. The six deltas lie between -0.00039 and 0.00000 validation NDCG.

With the follow-up constructions, 47 cues were tested and none cleared +0.0003 at 2 SE. The one positive sign, +0.00033 ± 0.00042, came from a count that included days after the validation feeds. Rebuilt from earlier days only, it fell to -0.00001 ± 0.00030. The ranker embeds every code and reads f1 to f5 directly, so a name adds nothing it cannot learn.

### One reading that reached the final files

Feature f5 is an engagement rate without smoothing, so a ranker stripped of every popularity input cannot tell a reliable high f5 from a noisy one. The popularity-light view keeps one exposure count for that reason. On the public board, the best click-label pool plus 4 popularity-light rankers scored 0.393205, the same pool plus 4 popularity-free rankers 0.392886, and the pool alone 0.392820.

The count did not do the job it was built for. In the validation cell of high f5 with fewer than 5 exposures, actual over expected clicks was 0.938 for both the popularity-free and the popularity-light screen pools, against 0.969 for the full view ([popularity-views](../experiments/popularity-views/)). The view stayed because of its public scores and its small loss under stale inputs.

## Near-copies among the listings

Many listings are near-copies of each other. In the validation embedding table, 33.2% of items have a neighbour with cosine above 0.98 and 15.0% above 0.99 (exact nearest-neighbour search). Pairs above 0.98 share a category 99.86% of the time and never share an `item_id`, so they look like similar listings or re-posts. Inside feeds they are common too: 55.2% of validation feeds hold a candidate pair above 0.98, covering 7.2% of candidate rows.

A near-copy of a history item is a strong positive. A candidate whose largest cosine to the history is 0.99 or more (1.2% of candidates) is positive 2.8 times as often as the base rate, and 1.9 times at 0.95 to 0.99. Near-copies inside one feed share labels more often than chance, yet only 6 to 9% of a positive's near-copies (cosine above 0.9, 0.95 or 0.98) are positive.

The hard part is telling near-copies apart. Among near-duplicate candidates (cosine above 0.95), the day-1 neural ensemble put the positive above the negative 61.1% of the time. The organiser features f1, f4 and f5 did no better than chance there (46.3%, 49.0% and 50.8%). An oracle that lifts each positive to the best score in its near-copy group would add 0.017 ± 0.001.

The ranker reads a family of 10 near-duplicate inputs that describe this structure among the feed's candidates ([solution](03-solution.md#inputs)), worth 0.0014 on validation and 0.0003 on the holdout in the 2-epoch removal screen. Re-ranking after the model did not help. Of 64 set-level rules tried on the day-1 ensemble (similarity propagation, score smoothing over near-copies, diversity), the best gave +0.00023 ± 0.00015, and a 2-fold estimate of that choice read -0.00076.

## Where the ranking error sits

The first measurement, on the first neural ensemble (validation 0.382, day 1), pointed at categories. An oracle that puts the categories of the positives first and keeps the model's order inside them would score 0.552. Of the model's wrongly ordered positive-negative pairs, 70% are pairs from different categories. The conclusion at the time was that knowing which category the user wants now was the largest gain available, and a category-intent model was planned.

### The first reading was wrong

Three measurements from the same day showed why ([error-analysis](../experiments/error-analysis/)):

- The 70% is a count effect: 74% of all positive-negative pairs are cross-category, and the error rate per pair is higher inside a category (0.334) than across (0.279).
- The ranker already orders categories better than the dedicated models built for the task. Its category-level NDCG is 0.7797 against 0.7477 for f3, and four category models trained on 464,000 train feeds came out 0.0008 to 0.0025 below it. Adding their output to the day-1 best file changed its validation NDCG by -0.0004 to 0.0000 (± 0.0002).
- The mirror oracle, with the model's category order kept and the order inside each category made perfect, scores 0.679, above the category oracle (0.551 on the same base).

A category-intent auxiliary head, already trained by then, read -0.0014 on validation and +0.0003 on the holdout, and the direction was dropped.

### Where the loss sits on the best click-label pool

On the best click-label pool (validation 0.40032, measured 2026-11-13), exact Shapley attribution over the two oracles splits the lost NDCG into 58% inside categories, 39% across categories and 3% interleaving. Each oracle's gain is averaged over the two orders in which the oracles can be applied, so the shares add to 100% without double counting. This split is measured on the half of the feeds that holds two thirds of the loss.

That half is defined by size: 7,379 validation feeds (51.2%) carry 66.0% of the loss, their median size is 109 candidates against 32 for the rest, and size alone predicts membership with AUC 0.766.

The share across categories depends on the history. When the category of the positive is among the 5 newest events (54% of those feeds), 76 to 80% of the loss is inside the category: the user is in the right category and about 11 items of it outrank the click. When the category is older or absent, 57 to 75% is across.

### Across categories the loss is mostly out of reach

On the day-1 best file, 20.8% of validation positives are ranked in the lower half of their feed, and 42% of those are in a category that does not occur among the user's 32 newest events (5.8% for positives ranked in the top 3). The model is calibrated on them (actual over expected positives 0.95 to 1.01 by novelty class). Such categories are clicked half as often, and ranking them low is right given the inputs.

An oracle that knows which feeds have positives only in categories new to the user (15.4% of feeds), and lifts such candidates there, would add 0.032. The best detector of those feeds, AUC 0.71 to 0.75, changed the score by -0.0005 ± 0.00015.

### Inside categories the inputs run out

In the heavier half of the feeds, on the same pool, the clicked item loses to the items ranked above it on every available cue. It wins 37% of comparisons on cosine to the history, 38% on f1, 41% on f3, 45% on popularity and 46% on f4. [Near-copies](#near-copies-among-the-listings) are the extreme case.

More than twenty signal families were aimed at this gap. Each stayed under +0.001 once the ranker had co-visitation and leave-user-out counts, and an item-id embedding table cost 0.0022 ± 0.0008. The larger gains that came later were from training (ten epochs in place of two, a new optimiser, distillation) and from retrieval over the history. Among new inputs only the scores of the trajectory encoder, a self-supervised model of history sequences, added more than 0.001 ([solution](03-solution.md#inputs)).

### No systematic miss left in the final model

No systematic miss was found in the final model. On 2026-11-17 nine training changes were aimed at its failure map ([error-analysis](../experiments/error-analysis/)). Some put more loss weight on the order inside categories, on the choice of category or on long feeds. Others lowered the weight on the teacher, made the cooldown longer or ran it at a higher learning rate, or added 128 older history events.

They were run as re-runs of only the cooldown of the 16 held-out full-view students, from stored checkpoints. None passed its randomization test (smallest p 0.16). The five that targeted one part of the loss did not move that part beyond the spread between two reference seeds.

A composite distillation loss weighted each feed by the gap between teacher and student and down-weighted the clicks the model found surprising. It lowered both validation and the holdout (-0.00058 ± 0.00047 and -0.00074 ± 0.00028). The concentration of the loss is what the draw of the click alone produces: the hardest 20% of feeds hold 27.5% of the loss, and 27.6% when the labels are re-drawn from the model's own softmax ([07-what-failed.md](07-what-failed.md) has the runs).

On 2026-11-18 the scores of FD16P20@m0.4_0.2 on validation were searched for biases (`lab/nn_x/w12/fixmiss/`). The score behaves as a calibrated log-odds. The fitted temperature, the factor that best rescales the scores to match the observed clicks, is 1.004 and 1.011 on two halves of the users; at 1 the softmax already matches them, and a monotone correction has nothing to fix.

Of the 659 slices scanned, 19 deviate with the same sign (|z| of 3 or more) on both halves of the users, with actual over expected clicks of 0.80 to 1.08, and none qualifies as a systematic miss (`lab/nn_x/w12/fixmiss/systematic/logs_s2_final.txt`). Positives in a category new to the user, 32% of all positives, are calibrated as well (model mass 0.329, actual share 0.322).

A perfect correction of every bias detected would be worth +0.00035 on validation. Corrections estimated from half of the users cost 0.0004 to 0.0009 in placebo runs, and about 50 cross-fitted corrections read 0 to -0.0013. In the team's reading, a further gain needs new information or a new input view.

Five ideas that touch the amount of labelled data or the cleanliness of the labels were closed on paper, by a proxy or in a flawed form, and never had a fair run. [Section 6 of 12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) lists them as a judgement, with no experiment behind it.

## Other readings that were corrected

| First reading | Corrected reading | What exposed it |
|---|---|---|
| The leak covers 354 validation and 334 test feeds, and a flagged item is positive 27.6% of the time | 110 validation and 103 test feeds, 93.5% positive | The count had linked all feeds of user 0 as one user; rows of user 0 alone are 3.2% positive (`lab/eda/synthesis/v2_leak.py`) |
| Inside validation only 4.5% of positives reappear in a later feed, so the leak is negligible | Same error: 681 validation feeds have a "later feed" with user 0 included, 25 without | The same cross-check |
| Co-visitation adds 0.0153 validation NDCG to f1 | 0.0041 ± 0.0008 | The f1 baseline was scored with its ties (0.3279) and the blend had none. Against f1 with ties broken at random, about 0.008 higher, the gain is 0.0041, and a history count's +0.0097 becomes +0.0038 (`v5_fair_blend.py` in [data-understanding](../experiments/data-understanding/)) |
| f4 is a static item attribute, f5 is popularity, f3 is an affinity to a top-level category | Listing age, an unsmoothed engagement rate, an affinity to the leaf category | The decoding of 2026-11-13: f4 never decreases for an item; f5 follows the click rate and not the impressions; f3 never coincides for two categories |
| Event types 5, 6 and 8 are messaging; source 3 is the item card | A delivery block; the favourites list | The 15 categories without such events; actions on a card inherit the source of the view that opened it |
| Candidates similar to items with a type-7 or type-9 event are clicked less (odds ratio 0.83), a satisfied need | Within (feed, category) the odds ratio is 1.16 | 28% of the flagged candidates are services, the large category with the lowest click rate |
| The public part is 8,578 feeds (30% by count) | 8,562 by the timestamp cut, or 8,580 with the boundary tick | Equal timestamps stay together |
| The hour-of-day mix differs between validation and test | A logging artefact | Several single hours with 0 to 9 feeds in both splits; with those hours filled, every day has the same profile |

## Scripts and records

The experiment folders hold the scripts and their outputs:

- [data-understanding](../experiments/data-understanding/): train against validation and test, single signals, the tie correction. `s07_train_vs_val_relation.py` there has the f4 inversion, and `lab/eda/shift_overlap/s10_train_transfer_debias.py` the train-fitted baseline.
- [exploration-detection](../experiments/exploration-detection/): both detectors of random-order feeds. The XGBoost runs with `pos` as a training-only feature are section 6 of `lab/reports/eda_labels_position.md`.
- [popularity-staleness](../experiments/popularity-staleness/) and [popularity-views](../experiments/popularity-views/): staleness, the lagged holdout, the popularity-light and popularity-free rankers.
- [validation-and-leaks](../experiments/validation-and-leaks/): the later-snapshot leak and the overlay.
- [codes-and-embeddings](../experiments/codes-and-embeddings/): what the vectors and the codes mean.
- [error-analysis](../experiments/error-analysis/): where the loss sits, from day 1 to the final model.

The reports written at the time are under `lab/reports/`. The near-copy numbers come from `eda_embeddings.md`, `EMBEDDINGS_1113.md` and `err_feed_structure.md` there.
