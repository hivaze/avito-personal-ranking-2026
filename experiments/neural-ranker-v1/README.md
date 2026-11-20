# The first neural ranker

In brief:

- On 2026-11-11 a neural set ranker trained on all 3.15M train feeds read 0.37361 NDCG on validation after 2 epochs, above the 0.368 to 0.370 of gradient-boosted trees.
- Attention to the user's history carried the gain (-0.0055 without it). Width, heads and blocks added nothing. Inputs, exact fp32 scores and training length moved more: 2 to 10 epochs gained +0.0052 on validation.
- `sub_v10`, four 10-epoch members, scored 0.38786 on public, the better of the two submitted files on this page.

## Why it was tried

Trees trained on train rows had stopped at validation NDCG 0.368 to 0.370, and LightGBM trained on validation alone at 0.365 to 0.367 (5-fold out-of-fold). NDCG, the competition metric, and the other terms on this page are defined in the [glossary](../../docs/glossary.md).

A feed is one request: the candidate listings shown to one user. Each feed holds up to 256 history events of that user and a 128-dimensional vector per item. A tree sees these only through hand-made similarity features, while attention from each candidate to the history reads them directly.

## What ran

### The model

The model (`model.py`) has width 128, 4 heads and 0.86M parameters:

- one transformer layer over the user token and the history
- two blocks, each with target attention (every candidate attends to the history) and set attention (the feed's candidates attend to one another)
- a listwise softmax loss (a softmax over the candidates of each feed)

Training used AdamW at 1e-3 with cosine decay.

Most train feeds were shown in the production order, the order of the site's own ranker. Validation and test feeds were shown in random order, so the display position never reaches inference. It enters training only through a train-only position term or an auxiliary loss.

Two epochs are about 20K steps and took 12 to 20 minutes on a GPU shared with 2 to 3 other jobs. Training is that fast because the whole train split (about 21 GB) lives in GPU memory, in batches bucketed by feed size (`data.py`).

### After training

Members then went through two more steps:

- a short second stage on 164K random-order train feeds
- a fine-tune on validation with 5 folds by user (`finetune.py`), with a zero-initialised branch of the network for inputs that exist only for validation and test

Ensembles average the per-feed z-scores of their members (`ensemble.py`). Each member's scores are standardised within the feed before the mean is taken.

From the second version on, checkpoints were chosen on a holdout of 30,000 random-order train feeds. Validation scores of train-only members (members not fine-tuned on validation) therefore use no validation label.

### Rounds and noise

Four rounds ran that day. Most comparisons are one seed, paired per feed on the 14,423 validation feeds. A single score has a standard error (SE) of about 0.0019, a paired difference 0.0005 to 0.0010. A "±" below is one paired SE.

The 48 training runs in `results/run_summary.csv` logged 1,799 minutes on shared GPUs, fine-tunes and second stages excluded.

Round 4 ported into this code the winners of the parallel round on modern training techniques ([optimiser-and-modern-tricks](../optimiser-and-modern-tricks/)):

- Muon, an optimiser for the hidden weight matrices, with a warmup-stable-decay schedule (warmup, a constant rate, then a cooldown in which the rate falls)
- retrieval of the 16 most similar history items
- gated attention
- a weight-averaged fine-tune

## Results

### Round 1: one component at a time

Round 1 tested components at 2 epochs on validation, with an fp32 score head. The first three changes are read against the base, the last four against the reference, which is the base without its position term.

| Model | Validation | Difference | Read against |
|---|---|---|---|
| base: history attention, set attention, position term | 0.37361 | | |
| without history tokens | 0.36808 | -0.0055 | base |
| without set attention | 0.37386 | +0.0003 | base |
| pointwise loss in place of listwise | 0.37272 | -0.0009 | base |
| without position term (reference) | 0.37385 | | |
| width 256, 8 heads | 0.37271 | -0.0011 | reference |
| auxiliary head toward the production position | 0.37581 | +0.0020 | reference |
| raw-cosine bias in the history attention | 0.37624 | +0.0024 | reference |
| auxiliary head and cosine bias together | 0.37848 | +0.0046 | reference |

The first runs scored in bf16 (bfloat16, a 16-bit float format), whose coarse steps tie scores within a feed. The metric breaks ties pessimistically, with the negatives first, and an fp32 head was worth +0.001 to +0.003 on validation.

### Later the same day

Paired runs with the same seed:

| Change | Validation | Holdout |
|---|---|---|
| equal loss weight per feed | +0.0020 | +0.0006 |
| listwise distillation of the production order (auxiliary loss) | +0.0024 | +0.0027 |
| co-visitation and leave-user-out popularity as inputs | +0.0032 | +0.0092 |
| hashed item-ID table | -0.0022 ± 0.0008 | -0.0036 |
| first train-row encoder scores (equal steps) | -0.0046 to -0.0089 | +0.0060 to +0.0085 |

Co-visitation counts how often the candidate occurs near the feed's history items in other users' histories. Leave-user-out counts leave out the same user's occurrences in later history snapshots.

The item-ID table cost -0.0075 ± 0.0021 on validation feeds whose positives had never been seen in train. The encoder scores came from the trajectory encoder; their first build for train rows is [failure 2](../../docs/07-what-failed.md#2-encoder-scores-for-train-rows-first-build) in the docs.

Training length, seed 23:

| Epochs | Validation | Holdout |
|---|---|---|
| 2 | 0.38625 | 0.31168 |
| 3 | 0.38733 | 0.31281 |
| 5 | 0.38970 | 0.31483 |
| 10 | 0.39141 | 0.31666 |

Round 4 then trained six 10-epoch members with 30% random-order feeds in the cooldown. All six collapsed: validation fell from about 0.391 to 0.364.

### Submitted files

| Submitted ensemble | Validation | Public |
|---|---|---|
| round-1 ensemble (8 fine-tunes of 5 members) with a residual tree stacker (`sub_v5`) | 0.3869 | 0.3802 |
| four 10-epoch members, half fine-tuned, half train-only, encoder score at weight 0.1 (`sub_v10`) | 0.39399 | 0.38786 |

The 0.3869 of the first file carried about 0.002 of a leak through inputs that were later quarantined, that is, withdrawn from every pipeline. The clean pipeline of that morning read about 0.385 ([validation-and-leaks](../validation-and-leaks/)).

### Re-measured two days later

The 2-epoch removal screen of 2026-11-13 ([popularity-views](../popularity-views/)) re-measured the round-1 components on the later recipe. Each value is what the component was worth, the score lost without it:

| Component | Validation | Holdout |
|---|---|---|
| loss toward the production order | +0.00219 | +0.00229 |
| set attention | +0.00198 | +0.00142 |
| equal feed weight | +0.00152 | +0.00096 |
| cosine bias | +0.00044 | +0.00045 |

## What it taught

Attention to the history carried the model. Removing the history tokens cost 0.0055, the largest change in round 1, while width, heads and blocks added nothing at this data size.

Exact fp32 scores were worth more than any block change. Inputs, in turn, moved more than architecture: the co-visitation inputs gave +0.0032 on validation and +0.0092 on the holdout.

Training length was the day's main lever. From 2 to 10 epochs the same seed gained +0.0052 on validation and +0.0050 on the holdout.

The encoder scores showed what a leak looks like. The train-row version passed a single-feature check: the NDCG of the feature alone over the ranker's NDCG was 0.88 to 0.89 on holdout feeds and 0.90 on validation. It failed the paired model check, with the holdout up and validation down at every step.

Opposite signs on two witnesses, the samples a change is judged on, became the team's leak signature. The rebuilt scores were adopted later ([pretraining](../pretraining/)).

Most round-1 numbers rest on one seed. In the re-measure the cosine bias (+0.00044 / +0.00045) sat inside noise. Set attention was neutral in round 1 and worth +0.00198 on validation in the later screen, and the records do not say what changed.

A cooldown share that worked at 2 epochs collapsed the 10-epoch members of round 4. That story continues in [optimiser-and-modern-tricks](../optimiser-and-modern-tricks/) and in [failure 3](../../docs/07-what-failed.md#3-a-cooldown-dose-tuned-at-2-epochs-and-used-at-10) of the docs.

## Files

`code/`:

- `model.py`: the ranker, its dense features and losses.
- `data.py`: GPU-resident data and size-bucketed batches.
- `common.py`: prediction, exact NDCG, normalisation statistics.
- `train.py`: trainer with holdout selection and the second stage.
- `finetune.py`: 5-fold-by-user fine-tune on validation.
- `holdout_eval.py`: member-set selection on the holdout.
- `ensemble.py`: per-feed z-score averaging.
- `run_final3.sh`: driver that records the round-2 recipe.

`results/` holds tables extracted from `lab/nn/logs/<run>.json`. Scores are as logged during training, so round-1 rows carry the bf16 head and sit below the round-1 table above.

- `run_summary.csv`: one row per run, with inputs, encoder, chosen step, validation and holdout NDCG, and minutes.
- `eval_curves.csv`: evaluation curves of 11 runs.
- `run_args.json`: arguments of the same 11 runs.

### Raw material and records

- `lab/nn/`: code and `logs/`, with one JSON read-out per run. The raw logs and the checkpoints are not in this repository.
- `lab/reports/nn_report.md`: all four rounds.
- `lab/reports/nnx_design.md` and `lab/reports/nnx_research_{llm_arch,recsys_arch,tabular_dl,training_gen}.md`: the design inputs of round 4.
- Records: the status log (a working record, not published).
