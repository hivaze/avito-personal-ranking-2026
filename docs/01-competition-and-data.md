# Competition and data

In brief:

- The task: for each request to Avito's main-page recommendation feed, order its 5 to 300 candidate listings so that the ones the user acted on come first.
- The score is NDCG (normalised discounted cumulative gain) over the whole list with binary labels, averaged over feeds. Tied scores count in their worst order.
- The data covers September 2026 in three splits cut by time: 3,151,208 train feeds, 14,423 validation feeds and 28,595 test feeds. Train feeds were shown in the order chosen by Avito's production ranker (its live recommendation system), validation and test feeds in random order.

Terms used here and on later pages are in the [glossary](glossary.md). On these pages "the team" is the entrant and the Claude Code session that did the work, taken together; the README says "I" and "Claude".

## The task

The Avito ML Cup 2026 on ods.ai had a track called "Personal ranking". Avito is a classifieds marketplace, and its main page shows a recommendation feed of listings. Each request to that feed is called a feed here, and each feed is one row of the data. The task is to order the candidate listings of each feed so that the ones the user acted on come first.

A feed holds:

- M candidate listings, between 5 and 300. Each comes with a category code, five numeric features (f1 to f5) and a 128-dimensional item vector.
- A 128-dimensional vector of the user.
- A snapshot of the user's history at the moment of the request: H events, at most 256, newest first. Each event has a listing, its category, an event type and an event source.

Train and validation rows also carry a binary target per candidate: 1 when at least one target action on that listing was recorded after it was shown in this feed. All actions weigh the same, and the organisers do not name them. The data suggests that the target is an engagement started from this feed, in 83 to 87% of cases an item card opened from it ([the evidence](02-what-the-data-told-us.md#what-the-codes-mean-and-what-that-bought)).

A candidate with target 1 is a positive. Every train and validation feed has at least one positive, and 74.5% of validation feeds have exactly one. A submission gives one score per test candidate, and only the order of the scores inside a feed counts.

## The splits

| Split | Days with traffic, Moscow time, September 2026 | Feeds | Labels | Order in which the candidates were shown |
|---|---|---|---|---|
| train | 6 to 14 and 17 to 19 | 3,151,208 | yes | the production ranker's order, given in `pos` |
| validation | 22 and 24 | 14,423 | yes | random |
| test | 26 to 29 | 28,595 | hidden | random |

The public leaderboard scores the earliest 30% of test feeds, 8,562 feeds. The private leaderboard scores the other 20,033 and stays unknown until 2026-12-09 ([the cut](#the-public--private-cut)).

The feeds per day are in [Calendar of the data](#calendar-of-the-data). The ways train differs from the other two splits are in [How train differs from validation and test](#how-train-differs-from-validation-and-test).

## The metric

The NDCG of one feed is computed in three steps:

1. Sort the candidates by score from high to low.
2. A positive at position p, counted from 1, earns 1 / log2(p + 1). The sum over the positives is the DCG.
3. Divide the DCG by the DCG of the ideal order, with every positive on top. The result is the NDCG of the feed.

The leaderboard score is the unweighted mean over feeds. The list is not cut off: a positive at position 300 still earns 0.12. A feed without a positive would score 1; there is none in train or validation.

Equal scores are ordered pessimistically: inside a group of tied candidates the negatives go first. A group of tied candidates therefore scores as the worst order of that group.

### A worked example

A feed of five candidates, of which B and E were clicked:

| Candidate | A | B | C | D | E |
|---|---|---|---|---|---|
| Target | 0 | 1 | 0 | 0 | 1 |
| Score | 0.9 | 0.7 | 0.7 | 0.2 | 0.1 |

B and C tie at 0.7, so the negative C goes ahead of B and the order is A, C, B, D, E.

| Order | Positions of B and E | DCG | NDCG |
|---|---|---|---|
| the scores above: A, C, B, D, E | 3 and 5 | 1/log2(4) + 1/log2(6) = 0.500 + 0.387 = 0.887 | 0.887 / 1.631 = 0.544 |
| B scored 0.71 instead | 2 and 5 | 0.631 + 0.387 = 1.018 | 1.018 / 1.631 = 0.624 |
| one constant score for all five | 4 and 5 | 0.431 + 0.387 = 0.818 | 0.818 / 1.631 = 0.501 |
| ideal | 1 and 2 | 1.000 + 0.631 = 1.631 | 1 |

The one tie between B and C costs this feed 0.080.

### Reference levels on validation

| Score used for ranking | Validation NDCG (14,423 feeds) |
|---|---|
| one constant for every candidate | 0.1986 |
| random order | 0.2890 |
| feature f3 alone, ties ordered negatives first | 0.2518 |
| feature f3 alone, expected value with ties broken at random | 0.3175 |
| feature f1 alone, ties ordered negatives first | 0.3279 |
| feature f1 alone, expected value with ties broken at random | 0.3365 |
| final pick 1, FD16P20@m0.4_0.2 ([final picks](11-final-picks.md)): 20 rankers trained on all train feeds (public 0.3952092) | 0.40293 |
| final pick 2, FDX16P20:ps: 16 pseudo-feed students and the 4 popularity-light students of pick 1 (public 0.39526061) | 0.40286 |

The tie rule puts a constant score 0.09 below a random order. Feature f3, which has one value per category in a feed, loses 0.066 to ties.

### Ties in the team's own models

The team ran into the tie rule once. The first rankers computed their output in bfloat16, a 16-bit float format that rounds scores between 2 and 4 to steps of 1/64. The ties cost 0.001 to 0.003 validation NDCG: one model read 0.3722 with the ties and 0.3741 with them broken at random.

From then on:

- The output layer runs in float32.
- Rankers are averaged as within-feed z-scores (score minus the feed mean, divided by the feed standard deviation). Averaging the ranks of two tree models had created ties and cost 0.0085 validation NDCG.
- The submission writer stores the within-feed rank of the blend and breaks exact ties with a fixed random key, so no file has two equal scores in one feed.

### No cut-off at position 30

The competition page writes the normaliser as "IDCG@30", the DCG of the ideal order cut at position 30. The board is not truncated. NDCG@30 computed on validation would have predicted 0.322 and 0.331 on the public board for two files that scored 0.380 and 0.388. The organisers' evaluator (`calculate_metrics.py` in their baseline archive) computes the full-list value.

The team's implementation, `lab/src/metrics.py`, agreed with two independent brute-force implementations to 3e-16, on random data with heavy ties and on real validation predictions. `avito_ranker/metrics.py` is its port.

## Submission, limits and dates

A submission is one parquet file with a row for every test candidate, 2,904,806 rows:

| Column | Type | Content |
|---|---|---|
| `feed_id` | int32 | test feed |
| `item_id` | int32 | candidate of that feed |
| `score` | float32 | higher means earlier in the list |

The evaluator joins on (`feed_id`, `item_id`), so row order is free. It rejects duplicate keys, missing keys, nulls and non-finite scores. The key set is the one in `sample_submission.parquet`.

Rules and dates:

- A team has 5 submissions per day, and the day rolls over at 21:00 UTC.
- Teams have up to 4 members and can merge until 2026-12-01.
- Uploads close on 2026-12-08 at 20:59 UTC. The final files are chosen between 2026-12-08 21:00 and 2026-12-09 09:00 UTC.
- The ten best teams hand in their solution for a check by the organisers. The prizes are 750,000, 450,000 and 300,000 roubles.
- The rules do not forbid reading the unlabelled test rows. The team used the test histories for item statistics.

The team's 40 scored submissions were made between 2026-11-11 and 2026-11-19 ([timeline](05-timeline.md)). Its places on the [public board](https://ods.ai/competitions/avitotechrecsys2026/leaderboard) are in [board snapshots](05-timeline.md#board-snapshots). The date the competition opened is not in the team's records. The platform takes uploads only from a person, so the person on the team uploaded every file by hand ([13-how-the-work-was-run.md](13-how-the-work-was-run.md#how-the-work-was-run)).

The organisers published two baselines without scores. One is a CatBoost model with the YetiRank ranking loss on nine inputs: f1 to f5, the category, and the dot product, cosine and a missing flag between the user and item vectors. The other is a Transformer of about 85,000 parameters over the user vector and the candidates. Neither reads the history.

## Data files and sizes

```
competition/
  readme.md
  sample_submission.parquet
  train/  val/  test/
    impressions/part-*.parquet     feeds, ZSTD-compressed; read all parts of a split
    impressions/_SUCCESS           empty marker
    item_embeddings/data.memmap    raw int8 array of shape (N, 128), C order
    item_embeddings/data.meta      JSON with dtype, shape and order of that array
```

The archive `competition.zip` is 17.9 GiB and is not part of this repository. The team kept it, with the organisers' baseline archive, in `artifacts/data/raw/` and extracted it to `artifacts/data/competition/`. `configs/paths.toml` holds both locations.

| File | train | validation | test |
|---|---|---|---|
| `impressions/part-*.parquet` | 512 files, 9.1 GiB | 64 files, 44 MiB | 64 files, 92 MiB |
| `item_embeddings/data.memmap` | 81,557,944 rows, 9.7 GiB | 2,516,711 rows, 307 MiB | 4,710,948 rows, 575 MiB |
| `sample_submission.parquet` (one file, top level) | | | 2,904,806 rows, 16 MiB |

What the splits contain, measured on the extracted data:

| | train | validation | test |
|---|---|---|---|
| feeds | 3,151,208 | 14,423 | 28,595 |
| candidate rows | 246,745,836 | 1,389,971 | 2,904,806 |
| history events | 583,586,378 | 2,499,547 | 4,973,106 |
| candidates per feed, mean / median | 78.3 / 48 | 96.4 / 67 | 101.6 / 72 |
| candidates per feed, 10% / 90% quantile | 9 / 213 | 12 / 247 | 13 / 256 |
| history events per feed, mean | 185 | 173 | 174 |
| feeds with the full 256 events / with none | 42.3% / 1.1% | 29.0% / 1.3% | 28.8% / 0.9% |
| distinct user ids | 2,455,382 | 13,741 | 27,309 |
| positive rate per candidate | 2.21% | 1.58% | hidden |
| feeds with exactly one positive | 66.6% | 74.5% | hidden |

## Fields

Types and the first sentence of each description follow the organisers' readme. The rest of the last column is what the team measured.

| Field | Type | What it holds |
|---|---|---|
| `feed_id` | int32 | Feed identifier, unique across the three splits. |
| `timestamp` | int32 | Unix time of the request in seconds. Every value is a multiple of 128 s, and day boundaries in the data follow Moscow time (UTC+3). |
| `user_id` | int32 | Anonymised user, one numbering for all splits. 0 is documented as a missing id. In the data it is a pooled bucket of visitors who were not logged in (123,904 train, 658 validation and 1,179 test feeds) and must never be treated as one user. |
| `user_embedding` | list of 128 int8 | User vector at the time of the request, in the same vector space as the item vectors. All zeros in 2.53% of train, 1.42% of validation and 0.28% of test feeds. |
| `item_id` | list of int32, one per candidate | Anonymised listing, one numbering for candidates and history in all splits. |
| `category_id` | list of int8, per candidate | Category code 1 to 53, 0 for unknown. 52 codes occur among candidates (2 to 53). |
| `f1` .. `f5` | lists of float32, per candidate | "Normalised real features" with no further description. Each is a quantile-bin encoder fitted on train and clipped to the train range. Meanings below. |
| `item_id__embedding_idx` | list of int32, per candidate | Row of the candidate in the embedding table of its own split. -1 means no vector: 0.003% of train, 0.015% of validation and 0.016% of test candidates. |
| `pos` | list of int16, per candidate, train only | Position at which the production ranker showed the candidate, from 0. |
| `target` | list of int8, per candidate, train and validation | The label. |
| `history__item_id` | list of int32, one per history event | Listing of the event. |
| `history__category_id` | list of int8, per event | Category of that listing, same codes as for candidates. |
| `history__event_type` | list of int8, per event | Code 1 to 10, 0 for unknown. |
| `history__event_source` | list of int8, per event | Code 1 to 30, 0 for unknown or missing. |
| `history__item_id__embedding_idx` | list of int32, per event | Row in the embedding table, -1 for no vector: 15.5% of train, 14.2% of validation and 14.0% of test events. |

All per-candidate lists of a feed are aligned, and so are all per-event lists. In validation and test the candidates of a feed were shuffled with one permutation for all fields, so the stored order carries no information.

Listings the user had already viewed were removed from the candidates of every split. A candidate's own item is therefore almost never in the history of its feed: 11 of 246,745,836 train candidates, none in validation or test. In train the removal leaves gaps in `pos`: 85.6% of train feeds have at least one, with 7.9 missing positions on average.

The history has no timestamps, only the order. The position of an event in it (`hpos` in the flattened arrays, 0 for the newest) is the only recency information. The history is a snapshot: a later feed of the same user shows the earlier history with the new events at the front.

The history is also thinned by event type before the cut at 256. After a gap of more than 7 days, 54% of a feed's view events are still in the same user's next snapshot, against 77% to 94% for event types 2 to 9. The old end of a long history is therefore weighted toward non-view events, such as contacts, by construction.

The vectors are unit vectors multiplied by 256 and rounded to int8, which makes a dot product about 65,536 times the cosine. Each split has its own table holding exactly the items that split refers to. An index is therefore local to its split, and items are matched across splits by `item_id`.

The meanings in the next four tables were worked out from behaviour in the data on 2026-11-13. The evidence is in [What the codes mean and what that bought](02-what-the-data-told-us.md#what-the-codes-mean-and-what-that-bought), the scripts and mapping tables in [codes-and-embeddings](../experiments/codes-and-embeddings/).

### f1 to f5

| Field | Meaning | Main evidence | Confidence |
|---|---|---|---|
| f3 | Affinity of the user to the candidate's leaf category | One value per (feed, category) in 100% of groups; R² (share of variance explained) 0.78 from the visible history on held-out feeds | high |
| f1 | A finer affinity inside the category, nested in f3 | When f3 is at its floor, f1 is at its floor in 99.6% of cases; it varies inside 70% of (feed, category) groups | high for the nesting, medium for what the finer grouping is |
| f2 | Affinity of the user to the region of the listing. 0 means no activity in that region; the most frequent value in a feed marks the home region | 99% of listing pairs from different categories are either always or never equal in f2, which exposes a region key on the item | high for the structure, medium-high for geography |
| f4 | Age of the listing | For the same item it never decreases; a 64-knot map from f4 to age reproduces the time between two appearances with a median error of 0.4 s | high |
| f5 | Engagement rate of the item without smoothing; 0 means no data | Spearman rank correlation +0.37 with the train click rate and -0.16 with train impressions; the top decile is small-denominator noise | high that it is a rate, medium that the numerator is clicks |

None of the five is the score of the production ranker: the per-feed Spearman correlation of each with `pos` stays within ±0.08.

### Event types

The codes are frequency ranks over train history events and carry no other meaning.

| Code | Share of validation events | Meaning | Confidence |
|---|---|---|---|
| 1 | 43.9% | View of an item card, with the surface it was opened from | high |
| 2 | 13.9% | First contact with the seller about the item (once per user and item, login needed) | low |
| 3 | 12.0% | Repeatable contact action such as a phone reveal, a call or a message | low |
| 4 | 10.7% | Add to or remove from favourites | medium-high |
| 5, 6 | 5.3%, 4.7% | Delivery or purchase flow; the two are logged at the same moment | medium-high for the block, low for which code is which |
| 7 | 4.2% | Lead-type action that can start from a tile, a card or the favourites list | low |
| 8 | 3.2% | Later step of the same order flow as 5 and 6 | as for 5 and 6 |
| 9 | 1.5% | Completion of 7 or 2, logged-in users only | low |
| 10 | 0.5% | Marker of the guest state attached to a view | medium |

### Event sources

A source is filled only for event types 1, 4 and 7, and types 4 and 7 inherit it from the view that opened the card. It names the surface where the listing was opened. The codes are frequency ranks again.

| Code | Share of validation views | Meaning | Confidence |
|---|---|---|---|
| 1 | 64.5% | Main-page recommendation feed, the surface of this competition | high |
| 2 | 23.7% | Search or catalogue listing | medium-high |
| 0 | 4.3% | Unknown or missing surface | medium |
| 4 | 3.2% | "Similar listings" block on an item card | medium |
| 5 | 1.5% | A goods-only, login-only surface that behaves like search | low |
| 6 | 1.2% | A second personalised recommendation surface | low to medium |
| 7 | 0.75% | Real-estate surface (85% of its events are apartments, houses and land) | medium |
| 3 | under 1% | Favourites list: 61% of its events are type 4 and half of its views are repeat views | medium |
| 8 to 30 | rare | Not identified | |

### Categories

A category code is 1 plus the dense rank of the leaf category id in Avito's public category tree. Code 1 is unused, and code 0 marks an unknown category on a few type-2 history events.

The eleven verticals are firm. When the codes are clustered by their item vectors, or by how users move between them in their histories, the vertical labels reach an adjusted Rand index of 0.32 to 0.56 against 0.00 ± 0.03 for shuffled labels (`lab/nn_x/codes/categories/out/05_mapping.txt`). The adjusted Rand index measures agreement between two groupings: 0 is chance, 1 is identical.

Most leaf names are of medium confidence. Only ids 9 and 84 were checked against Avito's site. The other ids come from a list written down without a source and not checked one by one.

| Code | Avito id | Name | Vertical | Share of validation candidates | Positive rate on validation | Confidence in the name |
|---|---|---|---|---|---|---|
| 2 | 9 | Cars | Transport | 11.18% | 1.53% | high |
| 3 | 10 | Parts and accessories | Transport | 7.58% | 1.61% | high |
| 4 | 11 | Water transport | Transport | 0.46% | 1.76% | medium |
| 5 | 14 | Motorcycles | Transport | 1.47% | 1.79% | medium |
| 6 | 19 | Repair and construction | Home | 5.74% | 1.49% | medium-high |
| 7 | 20 | Furniture and interior | Home | 4.82% | 1.62% | medium |
| 8 | 21 | Household appliances | Home | 2.04% | 1.39% | medium |
| 9 | 23 | Rooms | Real estate | 0.46% | 1.32% | medium |
| 10 | 24 | Apartments | Real estate | 4.08% | 1.58% | high |
| 11 | 25 | Houses and dachas | Real estate | 3.09% | 1.44% | medium |
| 12 | 26 | Land plots | Real estate | 0.98% | 1.17% | medium |
| 13 | 27 | Clothing and shoes | Personal | 12.43% | 1.97% | high |
| 14 | 28 | Watches and jewellery | Personal | 1.25% | 1.95% | medium |
| 15 | 29 | Kids clothing | Personal | 3.57% | 1.75% | medium-high |
| 16 | 30 | Kids goods and toys | Personal | 2.44% | 1.61% | medium |
| 17 | 31 | Desktop PCs | Electronics | 0.23% | 1.78% | medium |
| 18 | 32 | Audio and video | Electronics | 1.53% | 1.47% | medium |
| 19 | 33 | Tickets and travel | Hobby | 0.17% | 1.67% | medium |
| 20 | 34 | Bicycles | Hobby | 0.95% | 1.13% | medium |
| 21 | 36 | Collectibles | Hobby | 2.27% | 1.58% | medium |
| 22 | 38 | Musical instruments | Hobby | 0.30% | 2.05% | medium |
| 23 | 39 | Sports and leisure | Hobby | 2.46% | 1.40% | medium |
| 24 | 40 | Business equipment | Business | 0.47% | 1.51% | medium |
| 25 | 42 | Commercial property | Real estate | 0.59% | 0.95% | medium |
| 26 | 81 | Trucks and machinery | Transport | 1.14% | 1.62% | medium |
| 27 | 82 | Food | Home | 1.49% | 1.62% | medium |
| 28 | 83 | Books and magazines | Hobby | 0.83% | 1.99% | medium |
| 29 | 84 | Phones | Electronics | 1.60% | 1.83% | high |
| 30 | 85 | Garages and parking | Real estate | 0.62% | 0.98% | medium |
| 31 | 86 | Property abroad | Real estate | 0.07% | 2.42% | high |
| 32 | 87 | Tableware and kitchen | Home | 1.53% | 1.53% | medium |
| 33 | 88 | Beauty and health | Personal | 1.73% | 1.31% | medium |
| 34 to 38 | 89 to 93 | Dogs, cats, birds, aquarium, other animals | Animals | 2.29% together | 1.2% to 2.3% | medium |
| 39 | 94 | Pet supplies | Animals | 0.96% | 1.08% | high |
| 40 | 96 | Tablets and e-readers | Electronics | 0.26% | 1.79% | medium |
| 41 | 97 | Games and consoles | Electronics | 0.59% | 2.08% | medium |
| 42 | 98 | Laptops | Electronics | 0.28% | 1.71% | medium |
| 43 | 99 | Office equipment | Electronics | 0.35% | 1.27% | medium |
| 44 | 101 | PC components | Electronics | 0.91% | 1.72% | medium |
| 45 | 102 | Hunting and fishing | Hobby | 0.73% | 1.46% | medium |
| 46 | 105 | Photo equipment | Electronics | 0.23% | 1.94% | medium |
| 47 | 106 | Plants | Home | 1.45% | 2.11% | medium |
| 48 | 111 | Vacancies | Jobs | 2.69% | 1.62% | high |
| 49 | 112 | CVs | Jobs | 0.28% | 1.98% | medium-low |
| 50 | 114 | Services | Services | 8.95% | 1.17% | high |
| 51 | 116 | Ready business | Business | 0.14% | 1.48% | medium |
| 52 | above 116 | Hotels | Travel | 0.07% | 0.65% | medium-high |
| 53 | above 116 | Shift gigs | Jobs | 0.27% | 0.11% | low to medium |

Services appear in 81% of validation feeds although the affinity of their users is low (median f3 0.24), and they have the lowest positive rate among the large categories. The team reads that, and code 53 at 0.11%, as categories the production system inserts on purpose. A low score there is then not a model error.

## Calendar of the data

![Calendar of the data, 6 to 29 September 2026, Moscow time: train on 6 to 14 and 17 to 19 September, validation on Tuesday 22 and Thursday 24, test on 26 to 29 with the public / private cut at 11:32 on Sunday 27. Below, the age of the popularity inputs (mean days since a candidate item was last seen in train), one line per set: holdout (30,000 random-order train feeds) 1.75, validation 5.69, public 9.34, private 11.23.](figures/data-timeline.svg)

Dates and times of the data on this page are Moscow time (MSK, UTC+3), the clock by which the days of the data are cut. The first train feed is stamped 2026-09-05 21:00:48 UTC, which is 00:00:48 MSK on 6 September.

| Split | Days with traffic (2026) | Feeds per day |
|---|---|---|
| train | Sun 6 to Mon 14 Sep and Thu 17 to Sat 19 Sep | 203,548 to 291,760. Also 42 feeds on 15 Sep, none on 16 Sep and 34 at midnight of 20 Sep |
| validation | Tue 22 Sep and Thu 24 Sep | 7,542 and 6,878. Also 1 feed on 23 Sep and 2 at midnight of 25 Sep |
| test | Sat 26 to Tue 29 Sep | 6,050, 7,367, 7,578 and 7,600 |

The stray feeds on 15, 20, 23 and 25 September are all stamped 00:00:48 MSK, the first point of the 128-second time grid after midnight. They most likely hold the last seconds of the previous day.

The readme states full ranges: 6 to 19 September for train, 22 to 24 for validation and 26 to 29 for test, with 25 September left out as a gap. The two missing train days and the missing Wednesday of validation show only in the timestamps.

Inside validation and test, several single hours hold 0 to 9 feeds where about 400 is normal. These outages distort any hour-of-day statistic, and the final model has no hour-of-day input.

## How train differs from validation and test

Validation and test are alike. For every marginal distribution compared (f1 to f5, category, feed size, history length, event types and sources), the Jensen-Shannon divergence between them is at most 0.0013; it is 0 for identical distributions.

A classifier on feed-level statistics separates validation from test with AUC 0.524 (area under the ROC curve; 0.5 means it cannot tell them apart). The same classifier separates train from validation with AUC 0.925.

### Candidate order

Train feeds are stored in the order in which the production ranker showed them, with the position in `pos`. Validation and test contain only feeds whose candidates were shuffled uniformly at random before the user saw them, and the positions are not published. A click in train was made on a ranked list; a click in validation or test on a random one.

Feeds of the first kind are called production feeds here, those of the second kind random-order feeds. The lab code and the working records call the second kind exploration feeds. About 13% of the train feeds with 50 or more candidates were also shown in random order ([Random-order feeds inside train](02-what-the-data-told-us.md#random-order-feeds-inside-train)).

In train the positive rate falls steeply with the position (all 246.7 million train candidates):

| Position in the train feed | 0 | 10 | 100 | 200 |
|---|---|---|---|---|
| Positive rate | 11.81% | 4.05% | 1.27% | 0.92% |

The last item of a train feed is positive in 17.95% of feeds, against a base rate of 2.2%. The logged feed ends at the last listing the user interacted with, or one grid row after it, so the length of a train feed depends on what the user did.

Ordering train candidates "last item first, then by position" scores 0.4250 NDCG on train against 0.3240 for a random order. Neither signal exists in validation or test.

The feature values shift with the order. The mean of f1 is 0.443 in train and 0.400 in validation. That fits a production feed being the ranker's choice from a pool, and a random-order feed a random part of it. The positive rate per candidate is 2.21% in train and 1.58% in validation, and it is lower in validation in every bin of feed size.

### Feed length

Validation and test feeds are longer than train feeds: 96.4 and 101.6 candidates on average against 78.3, with medians of 67, 72 and 48. Two causes fit the data:

- In a random order, users scroll further before they interact.
- Fewer already-viewed listings are removed from a random-order feed: 1.4 to 2.4% of positions against 5.6 to 8.7% in a production feed, as estimated for train feeds of 50 or more candidates.

Length sets the level of the metric. A random order scores 0.539 on validation feeds of 5 to 9 candidates and 0.194 on feeds of 250 to 300. From the size mix alone, a random order scores 0.2890 on validation and is expected to score 0.2825 on the public part of test and 0.2846 on the private part.

The scored files show the same step. With the slope fixed at 1, the public score of a file is its validation score minus about 0.0075, the mean over the 38 scored files with a recorded validation score, the leak-inflated row 11 left out ([timeline](05-timeline.md#all-40-scored-submissions); the fitted line is in [Validation](04-validation.md#noise)). A public score below the validation score is therefore no sign of overfitting.

### Time lag

Train ends on 19 September. Validation is 3 and 5 days later, test 7 to 10 days later, and listings turn over. An item is cold when it was never a train candidate and warm otherwise.

Staleness is the number of days since a warm item was last a train candidate. An item last shown as a train candidate on 19 September is 3 days stale for a validation feed of 22 September and 7 days stale for a public feed of 26 September.

| Day (2026) | Part | Days after the last train day | Candidate rows whose item was a train candidate |
|---|---|---|---|
| Tue 22 Sep | validation | 3 | 69.1% |
| Thu 24 Sep | validation | 5 | 61.4% |
| Sat 26 Sep | test, public | 7 | 56.9% |
| Sun 27 Sep | test, public until 11:32, then private | 8 | 54.7% |
| Mon 28 Sep | test, private | 9 | 52.9% |
| Tue 29 Sep | test, private | 10 | 51.4% |

The warm share falls by about 2 percentage points per day: 65.4% of validation rows, 56.4% of public rows and 52.7% of private rows are warm.

Mean staleness is 5.69 days on validation (4.64 on Tuesday, 6.85 on Thursday), 9.34 on the public part and 11.23 on the private part. The ranges do not overlap: the 90th percentile of validation feeds is 7.42 days and the 10th percentile of public feeds 8.17. No re-weighting of validation feeds can reproduce the staleness of test.

With "share of candidates seen in train" added to its inputs, the classifier that separated validation from test at AUC 0.524 reaches 0.909.

What the organisers encoded themselves does not drift. The population stability index (PSI, a measure of how far a distribution moved between two samples) of each of f1 to f5 between validation and test is at most 0.007.

Few users and items repeat. 31.8% of validation users and 30.3% of test users appear in train. Leaving user 0 aside, validation and test have 1.00 feeds per user (25 validation users and 105 test users have more than one). 91% of the candidate items of validation and 86% of those of test occur in a single feed of their split.

## The public / private cut

The public board scores the earliest 30% of test feeds by timestamp, with feeds of equal timestamp kept together, and each feed belongs wholly to one part. The matching cut is timestamp 1790497920, which is 11:32 MSK on Sunday 27 September. The cut leaves 8,562 feeds before it and 20,033 from it on.

Exactly 18 feeds carry that timestamp, and which side they fall on is unknown. With them the public part would be 8,580 feeds, closer to 30%. No predicted level moves by more than 0.00003 with them.

| | public | private |
|---|---|---|
| period | Sat 26 Sep 00:00 to Sun 27 Sep 11:32 | Sun 27 Sep 11:32 to Tue 29 Sep 24:00 |
| feeds | 8,562 (6,050 + 2,512) | 20,033 (4,855 + 7,578 + 7,600) |
| feeds on a weekend day | 100% | 24.2% |
| candidates per feed, mean / median | 103.6 / 73 | 100.7 / 71 |
| feeds with fewer than 50 candidates | 38.7% | 39.2% |
| candidate rows whose item was a train candidate | 56.4% | 52.7% |
| mean staleness of warm items | 9.34 days | 11.23 days |
| feeds whose logged-in user has a later test feed | 59 | 49 |

The private part resembles validation in weekday mix and feed size, and lies further from train than any labelled data. From the size mix, a file was expected to score about 0.0026 higher on private than on public. The last row counts the test feeds on which the [same-user later-snapshot leak](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak) could act.
