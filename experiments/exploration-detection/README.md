# Finding the random-order feeds inside train

In brief:

- About 13% of the train feeds with 50 or more candidates were shown in random order, as validation and test feeds were, with no flag in the data. A detector built on the logged positions found them on 2026-11-11.
- They became the 30,000-feed holdout and the random-order pool of the cooldown. A second detector for all feed sizes (2026-11-14) was valid, but its larger pool did not help in the cooldown and hurt at higher doses.

Holdout, cooldown, cooldown re-run and the other terms of this page are defined in the [glossary](../../docs/glossary.md).

## Why it was tried

Validation and test feeds were shown in random order, train feeds in the order of Avito's production ranker. Display order was the largest difference between train and the scored splits: a classifier told train from validation at AUC 0.925.

The question was whether some train feeds also came from random-order traffic, and whether the logged positions could reveal them.

Expedia's hotel-search competition of 2013 flagged such feeds in its data. The team's survey of position-bias methods (`lab/reports/deep_position_bias_random.md`) ranked finding them first.

Train keeps every candidate's logged position, so a feed ordered independently of the ranker's inputs should be detectable.

## What ran

### First detector (2026-11-11, CPU)

1. A LightGBM proxy of the production ranker was fitted on 200,000 train feeds. It predicts each candidate's position rank from its features.
2. On other feeds, z = Spearman(prediction, position) × sqrt(M - 1), where M is the number of candidates. Under random order z follows N(0, 1).
3. A two-component mixture with that component fixed was fitted per feed-size bin, together with a second signal: the share of gaps in the logged positions, left by items the user had already seen. It gave each feed a probability p_random of random order. It is identifiable only for feeds with 50 or more candidates.

Two sets came from p_random. The holdout is 30,000 feeds drawn from those with p_random > 0.9. The exploration pool, also called the random-order pool, holds the feeds with p_random > 0.5: 164,118 feeds, 134,118 without the holdout.

### Second detector (2026-11-14)

A check on 2026-11-14 found that the trainer set p_random to 0 below 50 candidates. That covers 51% of train feeds and 39 to 41% of validation and test feeds.

The second detector (`lab/nn_x/w12/xdet/`) used no positions. A cross-fitted LightGBM classifier told validation and test feeds from train feeds by 51 summaries of the candidate set: a production feed holds the top of a ranked list, a random-order feed a random subset. Its density ratio was multiplied by an order likelihood ratio from the ranker's auxiliary head, which gave each feed a probability p_set.

The larger pool was then tested in 12 cooldown re-runs: 4 roots at three doses (2%, 5% and 10% of cooldown batches), under a rule written beforehand. A cooldown re-run is a 2-epoch run of only the cooldown from a stored checkpoint. Each took 24 to 37 minutes on one GPU.

## Results

### First detector

Train feeds with M ≥ 50 (`eda_labels_position.md` section 2.5):

| Quantity | Random-like feeds | Production-like feeds |
|---|---|---|
| mixture weight per size bin | 0.119 to 0.186 | the rest |
| estimated count among 1.55M feeds | about 206,000 (13.3%) | |
| gap share in logged positions | 1.4 to 2.4% | 5.6 to 8.7% |
| positive rate at rank 0 / base | 2.51% / 1.15% (89,794 feeds, p > 0.9) | 9.15% / 1.58% (1.01M feeds, p < 0.05) |
| NDCG of position order / random order | 0.2288 / 0.2235 | 0.3067 / 0.2420 |

The share was the same for train feeds of validation and test users: P(z < 0) was 0.075, against 0.082 in a control. It was flat by day (0.134 to 0.153).

Per feed, the random-like feeds were better training data. An XGBoost ranker trained on 50,000 of them beat one trained on 50,000 size-matched production-like feeds by +0.0066 ± 0.0015 validation NDCG with basic inputs, and by +0.0055 ± 0.0015 with the larger input set.

### Second detector

The classifier reached AUC 0.93 to 0.95 in every size bin. Its feeds with p_set > 0.9 had z mean +0.03 to +0.12 with sd 1.0 to 1.09 in all 20 bins, as random order requires.

It found 291,763 feeds with p > 0.5, 105,374 of them with M < 50.

It also showed that about 10% of the old p > 0.9 set was production-ordered, so part of the holdout was too. The holdout holds 2,294 contaminated feeds, on which members scored 0.332, against 26,655 clean ones at 0.318.

### Cooldown re-runs with the larger pool

Member differences against the reference cooldown re-run on the old pool at 2%, pooled over 4 roots (`results/xpool_cooldown_readout.txt`):

| Cooldown share | Validation | Holdout |
|---|---|---|
| 2% | +0.00025 ± 0.00035 | -0.00019 ± 0.00019 |
| 5% | -0.00016 ± 0.00038 | -0.00053 ± 0.00021 |
| 10% | -0.00161 ± 0.00043 | -0.00138 ± 0.00024 |

In an ensemble the four new members added less than four other-seed reference members: validation -0.00056, -0.00033 and -0.00032 at 2%, 5% and 10%. None passed the rule.

## What it taught

The first detector was adopted, and much rests on it:

- the holdout, which selects checkpoints without validation labels and is the second witness ([validation](../../docs/04-validation.md#the-witnesses));
- the pool replayed at 2% in the cooldown ([rounds-5-7](../rounds-5-7/));
- the domain embedding.

The second detector was valid and its lever null. More random-order feeds in the cooldown did not help, and a larger share hurt, as with the old pool. Test-like train feeds help in a small dose late in training; a bigger dose gets memorised.

Earlier runs pointed the same way. Up-weighting random-order feeds in the main run had cost about 0.001 on validation (0.37704 against 0.3780 to 0.3786). Mixing 12,000 of them into each fine-tune epoch was neutral in the second model generation (`lab/reports/nn_report.md`).

The holdout's contamination stayed a known limit.

## Files

`code/`

- `04b_explore_detect.py`: ranker proxy and first mixture fit.
- `04e_gaps.py`: gaps in logged positions.
- `04g_full_posterior.py`, `04h_refit.py`: the mixture on all train feeds; the refit wrote `train_prandom.npz` (not in this repository).
- `04i_exam_curve.py`: positive rate by rank for both groups.
- `xdet_setfeat.py`, `xdet_adv.py`, `xdet_post.py`: second detector (set summaries, classifier, posterior).

`results/`

- `04b_detect.json`, `04c_user_bucket.json`, `04e_gaps.json`: first fit per size bin, the user check, gap statistics.
- `04h_refit.json`, `04h_refit.txt`: final mixture, estimated counts, validation and test shares.
- `04i_exam_curve.json`: rate by rank, NDCG of position order.
- `xpool_rule_predeclared.txt`, `xpool_cooldown_readout.txt`: rule and read-out of the cooldown re-runs.

### Raw material and records

- `lab/eda/labels_position/`: scripts and JSON read-outs.
- `lab/nn_x/w12/xdet/`, with `check2.py` (the contamination check) and `mixcheck.py` (the ensemble check).
- The raw run logs and the arrays `train_prandom.npz` and `train_prandom_v2.npz` are not in this repository.
- Reports: `lab/reports/eda_labels_position.md` (sections 2.5 and 6) and `lab/reports/deep_position_bias_random.md`.
- Records: the status log and the coverage map (working records, not published).
- In context: [what the data showed](../../docs/02-what-the-data-told-us.md#random-order-feeds-inside-train), [validation](../../docs/04-validation.md).
