# Language models as rankers

In brief:

- Three tests, on 2026-11-12, 13 and 17, asked whether a language model helps on data that holds no text. None of them found a gain.
- As the trunk of a ranker, Qwen3-0.6B, Qwen3.5-4B and Qwen3.5-9B read -0.00378 to -0.00532 NDCG on validation against the team's ranker trained on the same 150,000 feeds. A randomly initialised trunk tied the pretrained one.
- Ranking 100 validation feeds directly, a language model scored 0.3957 against 0.4433 for G3, the best click-label pool. The one rule it proposed for editing the order of FD16P20 read -0.00011 ± 0.00007 on 13,701 validation feeds.

Terms such as NDCG, holdout, G3 and FD16P20 are defined in the [glossary](../../docs/glossary.md).

## Why it was tried

The data holds no text. Item ids, categories, event codes and features are anonymised numbers. A language model can only read them as projected vectors placed where word tokens would go.

Three questions followed from that:

- On 2026-11-12: can a pretrained language model be the trunk of a ranker, the transformer body under the ranker's own input and output layers?
- On 2026-11-13: can a general-purpose language model rank feeds directly from a text rendering?
- On 2026-11-17: can one correct the order of a final file?

The feasibility study (`lab/reports/LLM_FEASIBILITY.md`) rated the first idea a long shot. A language model brings a computation prior and no information the ranker lacks. Published comparisons on non-language inputs found no reliable gain over a matched model trained from scratch, except at small data.

It ran because a matched test was cheap, 1.8 to 2 GPU-hours estimated for the first stage, and would close a question that kept coming back.

## What ran

### The soft-token ranker

The soft-token ranker (`model_llm.py`) feeds the user vector, up to 256 history events and the candidates into a Qwen trunk through `inputs_embeds`, the argument that takes vectors in place of token ids. A soft token is one such projected vector. One frozen sink token is added.

Each candidate attends to the user, the history and itself. All candidates share one position id, so scores do not depend on candidate order.

LoRA (low-rank adapters trained on top of frozen weights) of rank 32 sits on every large linear layer. The 104 dense candidate features and the loss are the ranker's own.

Qwen3.5 has linear-attention layers that ignore custom masks. It ran through an exact "branch" trunk (`qwen35_branch.py`). Its relative error against the library's forward pass is 4.5e-6 for 4B and 1.6e-6 for 9B.

### Training and the rules

Training (`train_llm.py`, launched by `run_matched.sh`) used 150,000 sampled train feeds minus the holdout: 148,600 for seed 23 and 148,557 for seed 24. Checkpoints were chosen on the 30,000-feed holdout of random-order train feeds, and training never scored validation. The reference was the team's ranker trained on the same subsets and seeds.

| Arm | Seeds | Epochs |
|---|---|---|
| Qwen3-0.6B | 2 | 2, then 4 by a rule fixed in advance |
| Qwen3.5-4B | 2 | 2 |
| Qwen3.5-9B | 2 | 2 |
| control: the 0.6B architecture from random initialisation | 1 | 2 |

The acceptance rules (`compare.py`) predate every arm:

- A1: beat the ranker by more than 2 SE (standard errors), with both seeds positive, the holdout not negative and no slice below -2 SE. The slices are feed size, day of the week and the latest 20% of the holdout.
- A2: stay within 0.002 of the ranker, and pair with it better than two ranker seeds pair with each other.

The study used 46.74 of a cap of 60 GPU-hours, counted as hours of one job queue. The GPUs were shared, and the arms got 20 to 50% of the compute.

### Direct ranking

A general-purpose language model ranked 100 validation feeds, in five batches of 20, from a label-free text rendering. It was scored against G3, the best pool of rankers trained on click labels, on the same feeds.

### Review of a final order

The language model started from the order of FD16P20, pick 1 at the time, and could only move items. It studied 100 labelled validation feeds, shown as text cards of what FD16P20's members read, and wrote a guide. Its proposed rules were then tested in code on 13,701 other labelled validation feeds.

## Results

### Trunks

Scores are validation and holdout NDCG. Differences are paired per feed and pooled over seeds. On the same data the ranker read 0.38003 / 0.30386, and the randomly initialised 0.6B read 0.37599 / 0.29978.

| Trunk (body parameters) | Epochs | Validation / holdout | Minus ranker, validation | Minus ranker, holdout | Minus random init, validation (seed 23) | A1 / A2 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (0.44B) | 2 | 0.37552 / 0.29933 | -0.00451 ± 0.00079 | -0.00454 ± 0.00047 | -0.00007 ± 0.00060 | fail / fail |
| Qwen3-0.6B (0.44B) | 4 | 0.37624 / 0.29988 | -0.00378 ± 0.00077 | -0.00398 ± 0.00046 | +0.00046 ± 0.00074 | fail / fail |
| Qwen3.5-4B (3.57B) | 2 | 0.37523 / 0.29907 | -0.00480 ± 0.00081 | -0.00479 ± 0.00049 | +0.00038 ± 0.00067 | fail / fail |
| Qwen3.5-9B (6.92B) | 2 | 0.37470 / 0.29909 | -0.00532 ± 0.00080 | -0.00477 ± 0.00050 | -0.00019 ± 0.00065 | fail / fail |

- The size curve is flat or slightly down: 9B minus 0.6B read -0.00081 ± 0.00047 on validation.
- The arms correlate 0.932 to 0.943 with the ranker, against 0.946 between two ranker seeds, and 0.968 to 0.994 with each other.
- On feeds whose positives are all cold (never a train candidate before the feed's day), every arm gained (+0.009 to +0.013 on validation, pooled over seeds). The randomly initialised control gained there too (+0.0127, one seed). All of them lost on the other feeds.
- A separate re-check reproduced every number from the labels and found no leak.

### Direct ranking

| Direct ranking, 100 validation feeds | NDCG | Paired against G3 |
|---|---|---|
| language model | 0.3957 ± 0.0256 | -0.0476 ± 0.0232 |
| G3 | 0.4433 ± 0.0256 | |
| highest cosine to the history (one column) | 0.3888 ± 0.0243 | -0.0545 ± 0.0208 |
| random order | 0.3336 ± 0.0116 | -0.1097 ± 0.0214 |

A rank blend with 0.25 weight on the language model read -0.0214 ± 0.0116 against G3. Ranking the test set this way would have cost about 112M tokens.

### Review of FD16P20's order

The language model concluded that no edit made without labels beats FD16P20's order. Its rules were tested in code on the 13,701 feeds:

- The one rule it proposed demotes an item at ranks 2 to 10 that nearly duplicates an old history item. It read -0.00011 ± 0.00007.
- Ten variants of that rule read -0.00008 to -0.00089, and three other demotion rules -0.00034 to -0.00070.

Of 49 planned blind batches on a held-back 5% of validation, two ran as a dry run (one edit in 30 feeds, +0.00014 ± 0.00014). The rest were not run.

## What it taught

The route stopped at every size. Neither a 16-fold range of trunk size nor language pretraining changed anything measurable.

After one pass over the data, all arms, the ranker included, sat at the same holdout level (0.297 to 0.299). The ranker's lead came from its later passes, about +0.004 at 150,000 feeds.

The cold-feed gain shows up in the randomly initialised control too, so it belongs to this architecture at this data scale. It is paid for on warm feeds. Several weaker models showed the same trade of cold feeds for warm ones.

Ranking directly, the language model did as well as a one-column heuristic. Reviewing FD16P20's order, it found nothing a rule could correct. It saw only what FD16P20's members read, and FD16P20 is calibrated on the items the review would have moved.

Not tested: tuning per size, more than 150,000 feeds, and more than 2 epochs at 4B and 9B. Three of the four large arms were still improving at their last holdout evaluation.

Doubling the epochs at 0.6B read +0.00072 ± 0.00047 on validation. At that rate, closing a gap of about 0.005 looked implausible, and the rule written in advance said stop.

## Files

`code/`

- `model_llm.py`: soft-token ranker on a Qwen3 trunk.
- `qwen35_branch.py`: branch trunk for Qwen3.5.
- `train_llm.py`: trainer on the matched subset.
- `run_matched.sh`: launcher, rules in its header.
- `compare.py`: paired read-out, A1/A2, size curve.
- `estimate.py`: cost estimate made before any GPU use.

`results/`

- `compare_stageA.txt`, `compare_stageA.json`: Qwen3-0.6B, 4 epochs (the judged arm), text and numbers.
- `compare_q06_2ep.txt`, `compare_q06_2ep.json`: Qwen3-0.6B, 2 epochs.
- `compare_q4.txt`, `compare_q4.json`: Qwen3.5-4B.
- `compare_q9.txt`, `compare_q9.json`: Qwen3.5-9B.
- `size_curve.txt`, `size_curve.json`: the size curve.
- `size_extra.txt`: differences between sizes.
- `all_cold_positive_slice.txt`: the all-cold-positive slice.
- `e1_decision.json`: the 4-epoch extension rule.
- `probe_qwen3_06b_precision.json`: the fp32 prediction decision.
- `probe_qwen35_4b.json`, `probe_qwen35_9b.json`: speed and memory probes of 4B and 9B.
- `manual_ranking_pilot_cut.txt`: the direct-ranking tables (cut).

### Raw material and records

Scripts and read-outs:

- `lab/nn_x/llm/`: code, `logs/` (JSON read-outs), `results/`, `probe*/`, `baseline/`, `review_large/`.
- `lab/nn_x/llm_manual/`: `RESULT.txt`. The 100 rendered feeds are competition data and are not in this repository.
- `lab/nn_x/llm_review_1117/`: the review of FD16P20's order, 2026-11-17, with `RESULT.txt`.

The raw logs and the trained checkpoints are not in this repository. The Qwen weights are public checkpoints and are not part of it either.

Reports are `lab/reports/LLM_FEASIBILITY.md`, `lab/reports/llm_research.md`, `lab/reports/llm_build.md` and `lab/reports/llm_large.md`, whose section R8 gives the verdicts and GPU hours.

Records: the coverage map (a working record, not published).

In context: [what failed](../../docs/07-what-failed.md), [outside ideas](../../docs/08-outside-ideas.md).
