# Pre-training a sequence encoder on user histories

In brief:

- A causal transformer was pre-trained on the unlabelled user histories and tried as extra input scores, as starting weights for the ranker's history encoder and as a ranker of its own.
- Only the input scores were kept, worth +0.00077 ± 0.00073 on validation in a 10-epoch pair.
- Their first build for train rows leaked, and the diagnosis of that leak is the main lesson of this line.

## Why it was tried

The histories are the largest unlabelled resource in the data: 2,588,072 leak-safe sequences with 463.6M events.

A click in a feed leaves a trace in the user's next history as an "open the item, from the feed" event (85% of positives). "Predict the next new feed-originated open" is therefore a label-free stand-in for the competition target. The encoder followed the usual next-item recipe for a causal sequence model.

## What ran

### The encoder and its scores

On 2026-11-11 the encoder was trained with `seqenc.py` and `train.py`. It is a causal transformer of width 384 with 4 layers, 8.0M dense parameters plus learned ID vectors for the 40.3M items seen at least 3 times, trained with a sampled softmax. The [glossary](../../docs/glossary.md) calls it the trajectory encoder, a separate model from the ranker's history encoder.

The corpus (`build_corpus.py`) holds each user's latest train snapshot but only the earliest validation or test snapshot. No sequence therefore contains what a user did after a labelled feed.

Seven variants took 9.3 to 43 min each on one GPU, and the adopted configuration takes 21.5 min for 2 epochs. Validation and test candidates got 9 score features, each the mean over two configurations.

### Scores for train rows

Train feeds need the same scores, computed the way a test feed is scored. Three time windows were built with `build_corpus_window.py` (Sep 9-11, 12-14 and 17-19), with two encoders each at 12.6 to 25.0 min per run. Of the 246.7M train candidate rows, 72.1% got scores.

That first build failed (see [The train-row leak](#the-train-row-leak)). Four steps followed:

1. a CPU diagnosis (`diag_trainstage.py`);
2. a fix of the content table (`build_item_tab2.py`);
3. a retrain of the six window encoders, 13.4 to 58.5 min each on a shared GPU;
4. a gate the rebuilt scores had to pass (`gate_pre2.py`).

The 19 timed encoder runs add up to about 8 hours of single-GPU wall time, part of it on shared GPUs.

### Follow-ups

- 2026-11-12: a CPU gate for the encoder as starting weights (`stage0_gate.py`, 55 s).
- 2026-11-13: a written programme for larger pre-training. Two critiques cut it to four CPU diagnostics, about 30 CPU-minutes and no GPU.
- 2026-11-14: three dose tests of whether the ranker could use more of the encoder, about 5.6 GPU-hours. They swapped in scores from an encoder without item IDs, ran a 10-epoch removal pair, and added frozen encoder states as an extra branch of the ranker.

## Results

Signed values are paired per-feed differences, and ± is one standard error (SE) of the difference. A witness is a sample a change is judged on. Where a row names two witnesses, its values are given in the same order.

The witness "validation (train-only)" scores a member's predictions from before the fine-tune on the validation days.

| Reading | Witness | Value |
|---|---|---|
| Encoder alone, zero-shot (best similarity feature) | validation | 0.3590 (0.349) |
| 9 encoder features in a LightGBM harness, 3 seeds | validation, paired | +0.0037 ± 0.0008 |
| Encoder score at weight 0.1: day-one blend; stronger ensemble of that evening | validation | +0.00082 ± 0.00038; -0.00078 ± 0.00042, dropped |
| Fine-tuning the encoder on validation labels, 5 folds | validation | 0.3536 fell to 0.350-0.352 |
| First train-row scores as ranker inputs | validation (train-only); holdout | -0.005 to -0.009; +0.006 to +0.0085 |
| The same in the LightGBM diagnosis | validation; held-out train feeds | -0.0090 ± 0.0013; +0.0106 ± 0.0011 |
| Rebuilt train-row scores, LightGBM gate | validation; held-out train feeds | +0.00535 ± 0.00086; +0.00522 ± 0.00082 |
| Rebuilt scores as ranker inputs, 3 epochs, 2 seeds | validation (train-only); holdout | +0.00191 ± 0.00058; +0.0002 |
| Their removal, 2-epoch pair, 2 seeds | validation; holdout | -0.00128 ± 0.00061; -0.00054 ± 0.00034 |
| Their worth in a 10-epoch pair (newest read) | validation; holdout | +0.00077 ± 0.00073; +0.00067 |
| Encoder score as a residual on the best click-label pool, weight chosen in user folds | validation | -0.00018 ± 0.00020 |
| CPU diagnostic 1: trajectory clusters | validation | 0 of 8 arms passed |
| CPU diagnostic 2: user vector | validation | -0.00034 ± 0.00033 |
| CPU diagnostic 3: whitened similarity | validation | -0.00031 ± 0.00017 |
| CPU diagnostic 4: neighbour labels | validation | 0 of 4 arms passed |
| Frozen encoder states as an extra branch of the ranker | validation | -0.00007 ± 0.00053 |

## What it taught

### What was kept

Five encoder scores and their presence flag are among the 34 pre-computed inputs of every final member. The 9-feature family is among the extra inputs of the fine-tune.

Their worth is small, and smaller in the full recipe: +0.0019 on validation at 3 epochs and +0.0013 at 2 epochs, against +0.0008 ± 0.0007 in a 10-epoch pair.

### The train-row leak

The first train-row build is the instructive failure. The holdout rose and validation fell by similar amounts, which is what a leak present only on train rows looks like. The holdout is made of train feeds, so it shared the leak.

The cause was the encoder's content table. It had been built from histories and from the validation and test candidate files, without the train candidates. A train candidate therefore had a content vector only if it appeared in some history, often the same user's later one.

Leave-user-out popularity counts an item without the same user's later snapshots. Among candidates with zero leave-user-out popularity, those with content were positive at 3.2% to 10.1%, against 1.0% to 1.7% for those without.

The leak checks missed it because they split feeds only by later feeds inside the same window, while the leaking snapshots sat in later windows and in validation and test. Adding the 14.2M train candidates to the table removed the leak. [docs/07-what-failed.md](../../docs/07-what-failed.md#2-encoder-scores-for-train-rows-first-build) follows every build of these scores.

A second defect survived the gate. The loader clipped the raw scores at ±10, which put 83.8% of validation rows and 92.5% of train rows of the main score on the clip value. The feature review found it, and rescaled columns were in place for the 3-epoch read.

### What was closed

Every larger use of the encoder was closed. Its extra signal is its item-ID part, and the content part adds about nothing to a ranker trained on 3M labelled feeds.

The literature review in `lab/reports/PRETRAIN_PLAN_1113.md` (GPSD, LazFormer, PinFM, LUM and others) found three conditions under which pre-training beat a strong supervised ranker. None holds here:

- An unlabelled corpus much larger than the labelled data. Here the two are about the same size: 463.6M corpus events against 448.7M history events inside the labelled feeds.
- An item-ID table that overfits and can be frozen. The ranker has no ID table.
- Inputs the ranker cannot see. The ranker reads the same history, and the one outside source, the user vector supplied with each feed, is already an input.

The encoder as starting weights was never trained. Its gate failed, with a pair AUC of 0.316 on the target cell against a bar of 0.52. That cell holds the positives in the category the best click-label pool ranks first, with at least 3 candidates of that category above them.

Never run, refuted on paper only: a scale-up to width 512-768 with 6-8 layers, and a masked-event auxiliary task inside the ranker.

## Files

Scripts in `code/`:

- `build_corpus.py`: the leak-safe corpus, the two next-item targets and the content table.
- `seqenc.py`: the encoder.
- `train.py`: pre-training, zero-shot evaluation and scoring.
- `build_corpus_window.py`: the three time-windowed corpora for train rows.
- `diag_trainstage.py`: the diagnosis of the train-row leak.
- `build_item_tab2.py`: the fix of the content table.
- `gate_pre2.py`: the gate the rebuilt scores had to pass.
- `stage0_gate.py`: the gate for the encoder as starting weights.

Read-outs in `results/`:

- `pretrain_v3_training_log.txt`: loss and zero-shot validation NDCG of the adopted configuration.
- `zero_shot_and_blend_evals.jsonl`: zero-shot, leak-audit and blend reads of every variant.
- `train_window_checks_first_build.json`: the leak checks that the first build passed.
- `train_rows_leak_diagnosis.txt`, `rebuild_gate.json`: the diagnosis and the gate.
- `encoder_as_init_stage0_gate.txt`: the failed starting-weights gate.
- `diag_P1_trajectory_clusters.txt`, `diag_P2_user_vector_residual.txt`, `diag_P3_whitened_similarity.txt`, `diag_P4_neighbour_label_transfer.txt`: the four CPU diagnostics.

### Raw material and records

- `lab/pretrain/`: all scripts.
- `lab/logs/pretrain_*` and `lab/logs/pre2_*`: the JSON read-outs of the training runs and checks (the raw logs are not in this repository).
- `lab/nn_x/r3/`: the starting-weights gate and its plan.
- `lab/nn_x/pretrain2/`: the four diagnostics with their pre-declared rules.
- `lab/nn_x/abl/logs/`: the dose tests, in `abl_eval_encdose.txt`, `readout_pre2_10ep.txt` and `abl_eval_kv64.txt`.
- `lab/reports/pretrain_report.md`: sections 8b and 8c cover the failure and the fix.
- `lab/reports/PRETRAIN_PLAN_1113.md`: the programme for larger pre-training and its literature review.
- `lab/reports/FEATURES_REVIEW.md`: item FR-1 is the clipping defect.
- Records: the status log (a working record, not published).
