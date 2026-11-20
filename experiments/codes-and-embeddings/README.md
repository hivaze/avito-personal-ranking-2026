# Decoding the anonymised codes and the item vectors

In brief:

- Every event type, event source, category and candidate feature in the data is an anonymous number, and every item is a 128-dimensional int8 vector. On 2026-11-13 the team asked what these are and whether knowing it gives the ranker anything new.
- The decoding held. The vectors most likely come from the marketplace's own two-tower recommender (subjective probability 0.94), stored as round(256 × unit vector).
- No feature built on the decoding passed its bar, and nothing from this line was submitted. One finding, that f5 is an unsmoothed rate, led to the popularity-light view that is in every final file.

Scores are NDCG on validation or on the holdout (30,000 random-order train feeds kept out of training), and ± is one paired standard error (SE). Other terms are in the [glossary](../../docs/glossary.md).

## Why it was tried

The best pool of the day, G3 (20 rankers trained on click labels; validation NDCG 0.40032, public 0.392820), had stopped responding to model changes. Each decoding had a possible use:

- If a code meant "contact", items whose need was already met could be pushed down.
- If the vectors came from a public text model, they could be extended.
- If the categories had a hierarchy, rare ones could share statistics.

## What ran

Everything ran on CPU with 12 threads, apart from exact nearest neighbours on the validation item table and the public-model controls (one GPU). Time is on record only for the leak audit: 3 CPU minutes.

- Four decodings: event types, event sources, category codes, and the features f1 to f5 together with the target. Cross-checks settled their conflicts, and a permutation test checked the category numbering.
- Three embedding studies: structure (probes on 1.5M train candidate items), origin (quantisation, duplicates, seven public embedding models as controls) and usability (five processing ideas).
- One test for every candidate feature built on the decoding, a nested residual on G3: z(G3) + w × z(x), where z is the within-feed z-score. The weight w was chosen on some users and scored on the others, then carried to the holdout.

The bars were written before each run. A decoding cue had to gain more than +0.0003 and more than 2 SE. An embedding processing had to gain at least +0.0005 and at least 2 SE.

## Results

### The codes

| Code | Decoded meaning | Evidence |
|---|---|---|
| Event types, event sources | frequency ranks | 0 inversions in 45 and 435 pairs |
| Category codes | the marketplace's public category-id order, which gives 11 verticals | 20 of 51 adjacent code pairs share a data-derived cluster, against 7.5 ± 2.3 under random relabelling |
| Target | an engagement started from the recommendation feed | 83 to 87% of positives return later as a card view from the feed |
| Event codes 5, 6 and 8 | a delivery block | zero in 15 categories over about 7.9M events, apart from 3 events in services |

Of the five candidate features, f3 is a user-by-leaf-category affinity and f1 a finer one. Feature f2 is a user-by-region affinity, f4 the listing age and f5 an unsmoothed engagement rate. None is a leaked production score.

One claim of the first decoding fell. "Items like a contacted one get fewer clicks" was a category confound: the odds ratio within feed and category is 1.16 on validation.

### The item vectors

The team judged, with a subjective probability of 0.94, that the vectors come from the marketplace's own two-tower recommender. A two-tower model embeds users and items with separate networks into one space.

They are stored as round(256 × unit vector). The mean squared norm is 65546.5 to 65546.7 against 65546.67 predicted.

They are not a Matryoshka prefix (an embedding trained so that its leading dimensions work on their own). The per-dimension spread against the dimension index has Spearman -0.06, against -0.52 and -0.83 for two Matryoshka controls.

A linear probe reads the category at 0.983 accuracy and the item click rate at R² 0.003. Between splits, 77 to 83% of the vectors of shared items are bit-identical.

### Tests of features built on the decoding

| Test | Witness | Result | Verdict |
|---|---|---|---|
| 6 features that need the decoding (vertical shares, delivery class) | validation, nested | -0.00039 to -0.00000 (SE 0.00011 to 0.00025) | 0 of 6 pass |
| 34 further decoding cues | validation, nested | largest z 0.95 | 0 of 34 pass |
| item feed-pull share, first build | validation, nested | +0.00033 ± 0.00042 | leak |
| the same from strictly earlier days | validation, nested | -0.00001 ± 0.00030 | fail |
| the same, validation weight carried over | holdout | -0.00073 ± 0.00024 (z -3.1) | fail |
| "similar listings" item graph | validation, nested | -0.00012 to -0.00025; 0.05 to 0.13% coverage | fail |
| 5 embedding processings (drift, whitening, centring, mean direction, near-duplicates) | validation, nested | best +0.00045 ± 0.00022 | 0 of 5 pass |
| mean-direction projection | holdout, nested | -0.00004 ± 0.00003 | not replicated |
| location sub-space match | validation / holdout | -0.00008 ± 0.00030 / -0.00029 (z -2.08) | fail |

Nothing from this line was submitted, so there is no public reading.

## What it taught

The decoding went into the documentation and did nothing for the score. The ranker embeds every code and reads f1 to f5, and G3 was already calibrated on every decoded class (actual over expected clicks 0.97 to 1.03).

The only ideas that needed the decoding were cross-user aggregates. Built without leaks, they reduced to popularity (popularity-matched win rate 0.500).

The one positive validation sign came from a forward-in-time leak. Counts that included the test split's histories carried other users' later clicks, and their forward-only part alone separated positives at AUC 0.521.

A holdout clause "nested gain at or above 0" passes by default when the weight grid picks w = 0. Carrying the validation weight to the holdout is the real check.

One finding reached the score. Since f5 is an unsmoothed rate, a model without popularity inputs cannot tell a reliable high f5 from a noisy one. That motivated the popularity-light view, in which the popularity inputs are replaced by one clipped exposure count ([popularity-views](../popularity-views/)).

The view did not repair the validation cell of high f5 with fewer than 5 exposures, and it still ended up in every final file.

## Files

`code/`

- `06_order_test.py`: permutation test of the category numbering.
- `x01_skeptic_checks.py`: cross-checks of the four decodings.
- `x02_decoded_residuals.py`: residual screen of the 6 decoding features.
- `k1_k2.py`: leak audit of the feed-pull cue.
- `o01_quant_hash.py`: quantisation fingerprint and duplicate census.
- `o06_public.py`: public embedding models as controls.
- `s02_probes.py`: probes of what the vector encodes.
- `s1_eval.py`: label tests of the five embedding processings.

`results/`

- `category_mapping.csv`: categories with ids, Russian and English names, verticals.
- `event_type_mapping.csv`: meaning per event code.
- `event_source_mapping.csv`: meaning per source code.
- `06_order_test.txt`: permutation test output.
- `x01_skeptic_checks.txt`: cross-check output.
- `c1_delivery_share.csv`: delivery-event share per category.
- `x02_decoded_residuals.txt`: residual screen, 0 of 6.
- `k1_k2_predeclared.txt`, `k1_k2.txt`: rules and output of the leak audit.
- `followup_result_cut.txt`: item graph, regional popularity, popularity-free readout (cut).
- `emb_origin_predeclared.txt`: origin gates.
- `emb_o01_quantisation.json`: quantisation and duplicate numbers.
- `emb_o06_public_controls_1.json`, `emb_o06_public_controls_2.json`: statistics of the seven public models (four, then three).
- `emb_s02_probes.json`: probe accuracies and R².
- `emb_usability_predeclared.txt`, `emb_usability_deviations.txt`: processing rules and the one recorded deviation.
- `emb_usability_eval.txt`, `emb_usability_gbm.txt`, `emb_usability_holdout.txt`: validation reads, corrected tree-model reads, holdout reads.

### Raw material and records

- Scripts and read-outs: `lab/nn_x/codes/` and `lab/nn_x/emb/`. The raw logs and the arrays are not in this repository.
- Reports: `lab/reports/CODES_DECODING_1113.md`, `lab/reports/CODES_EXPLOIT_1113.md`, `lab/reports/EMBEDDINGS_1113.md`.
- Records: the coverage map (a working record, not published).
- In context: [the competition and the data](../../docs/01-competition-and-data.md), [what the data showed](../../docs/02-what-the-data-told-us.md), with its sections on [the embeddings](../../docs/02-what-the-data-told-us.md#what-the-embeddings-are) and [the codes](../../docs/02-what-the-data-told-us.md#what-the-codes-mean-and-what-that-bought).
