# Understanding the data

In brief:

- On the first night of the work (early hours of 2026-11-11), four studies asked how train, validation and test differ, what the organisers' embeddings encode, which collaborative signals carry information, and whether the way of measuring was sound.
- Validation and test are the same kind of traffic and train is not (adversarial AUC 0.524 between validation and test, 0.925 between train and validation). The metric breaks ties pessimistically, and blend gains measured against tied f1 were three times too large.
- The answers set the validation scheme and the shape of the first neural ranker.

NDCG values are on validation (14,423 feeds) unless a row says otherwise. Other terms are in the [glossary](../../docs/glossary.md).

## Why it was tried

Train feeds were shown in the production ranker's order, validation and test feeds in random order ([competition and data](../../docs/01-competition-and-data.md)). A model fitted on train is therefore judged on another distribution.

With 14,423 validation feeds, a paired per-feed difference has a standard error of 0.0005 to 0.0012 NDCG. The team needed to know which differences stand out from that noise.

## What ran

Four studies ran in the early hours of 2026-11-11, on stored arrays, mostly on CPU:

- `lab/eda/shift_overlap/`: field distributions per split, adversarial LightGBM classifiers between splits, coverage of train statistics by day, the public/private boundary of test, and the feature-label relation in train and validation.
- `lab/eda/embeddings/`: vector statistics, about forty similarity signals scored one at a time, scorers of about 330 learned parameters, and the lift of embedding features in a LightGBM ranker fitted on 200,000 train feeds.
- `lab/eda/collab_stats/`: target encodings, popularity, item2vec, implicit ALS in torch on one GPU (6 to 8 minutes per fit), and LightGBM group ablations fitted 5-fold on validation.
- `lab/eda/synthesis/`: cross-checks of the other three.

An adversarial classifier is trained to tell two splits apart: an AUC near 0.5 means they look alike. ALS (alternating least squares) and item2vec learn item vectors from user histories.

A single signal was scored with a random tie-break. For scale, a random order scores 0.2890 on validation.

## Results

| Finding | Split | Value |
|---|---|---|
| adversarial AUC, feed level: validation vs test / train vs validation | validation, test, train | 0.524 / 0.925 |
| validation vs test with "share of candidates seen in train" added | validation, test | 0.909 |
| candidates whose item was a train candidate: first validation day / last test day | validation, test | 69.1% / 51.4% |
| the same, public part (first 8,562 feeds, all weekend) / private part | test | 56.4% / 52.7% |
| f1 alone, pessimistic ties / random tie-break (one draw; expected value 0.3365) | validation | 0.3279 / 0.3364 |
| positive rate of the freshest f4 decile relative to the mean | train / validation | 1.22 / 0.46 |
| recency-weighted cosine to the history, over f1 | validation | 0.3492, +0.0126 ± 0.0017 |
| small attention scorer fitted on train, over the user-item dot product | validation | 0.3584, +0.0152 ± 0.0014 |
| LightGBM on train: f1 to f5 and category / plus 39 embedding features | validation | 0.3303 / 0.3651 |
| first ALS: feeds whose user has a later feed / other feeds (leak-free rebuild) | validation | 0.583 / 0.329 (0.318 / 0.326) |
| f1 + co-visitation rank blend, against f1 with pessimistic / random ties | validation | +0.0125 / +0.0041 ± 0.0008 |

Uncertainties are paired per-feed standard errors; rows without one are single readings.

## What it taught

### Which split to judge on

The adversarial AUCs put validation and test on one side and train on the other. Models were judged on validation and later also on 30,000 random-order train feeds held out of training, never on train rows.

The f4 inversion (the freshest listings are clicked more than average in train and less in validation) is a display-position artefact. From position 50 on, train shows the validation picture. The ranker never takes the position as an input; the production order entered only as an auxiliary training target ([solution](../../docs/03-solution.md)).

### Staleness from validation to test

The shift from validation to test acts through age. Train-derived item statistics lose about two points of coverage per day, and private is staler than public. That started the work on stale popularity inputs ([popularity-staleness](../popularity-staleness/)).

The boundary and the feed-size re-weighting of `s05_public_private.py` became the public-score calibration (`lab/src/calib.py`, now `avito_ranker/validation/calibration.py`). With the cold share added, they became the private-mix view (`lab/src/pmix.py`, now `avito_ranker/validation/private_mix.py`), which later rules used as a veto.

### Ties

The metric breaks ties pessimistically: tied candidates are ordered with the negatives first. Ties cost f1 0.0085, and blend gains measured against tied f1 were three times too large.

Every submitted score was made tie-free. For the same reason the first ranker's output head moved from bf16 (bfloat16, a 16-bit float format) to 32-bit floats.

### The embeddings

The embeddings are unit vectors scaled by 256, so a dot product is a cosine. The signal lies in the similarity of a candidate to the user's recent history. The mean over the whole history scored 0.3294, below f1.

The first neural ranker was built around attention from each candidate to the history, with the raw cosine as a bias ([neural-ranker-v1](../neural-ranker-v1/)).

### The ALS leak and what failed

The first ALS was trained on pairs that included the users' later snapshots, and it leaked. It was rebuilt without them (`04g_als_clean_gpu.py`). The leak is told in [validation-and-leaks](../validation-and-leaks/).

What failed:

- Train candidate counts and click rates were worth nothing on validation, because they carry the production position.
- A LightGBM fitted on train with f1 to f5 scored below f1 alone with random ties (0.3303 against 0.3364).

Each number is one run, and the trained ranker superseded most of them within a day.

## Files

`code/`

- `s04_adversarial.py`: adversarial classifiers between splits.
- `s05_public_private.py`: public/private boundary, feed-size re-weighting.
- `s03b_coverage_decay.py`: coverage of train statistics by day.
- `s07_train_vs_val_relation.py`: feature-label relation in train and validation.
- `02_val_signals.py`: single embedding signals.
- `06c_history_attention.py`: the small history-attention scorer.
- `04g_als_clean_gpu.py`: leak-free implicit ALS.
- `v5_fair_blend.py`: blend gains under both tie conventions.

`results/`

- `summary.csv`: these numbers and about twenty more, each with its source.
- `shift_*.txt`: outputs of the four shift scripts.
- `emb_*`: single signals, the scorer, the LightGBM lift.
- `collab_*.txt`: ALS runs and the group ablation.
- `synthesis_*.txt`: the tie correction (`synthesis_v5_fair_blend.txt`) and the test feeds the later-history channel touches (`synthesis_v2_leak.txt`).

### Raw material and records

- Scripts and result files (JSON, text and CSV): `lab/eda/shift_overlap/`, `lab/eda/embeddings/`, `lab/eda/collab_stats/`, `lab/eda/synthesis/`, and `lab/src/als_clean_pairs.py`. The raw logs and the arrays are not in this repository; the copies in `results/` carry the numbers.
- Reports: `lab/reports/eda_shift_overlap.md`, `lab/reports/eda_embeddings.md`, `lab/reports/eda_collab_stats.md`.
- The other two first-night studies (`eda_labels_position.md`, `eda_history_leak.md`) are in [exploration-detection](../exploration-detection/) and [validation-and-leaks](../validation-and-leaks/).
- Records: the status log (a working record, not published).
- In context: [what the data showed](../../docs/02-what-the-data-told-us.md).
