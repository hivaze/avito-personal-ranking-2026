# Rounds 5 to 7: screens of the training recipe

In brief:

- Once the neural ranker had its recipe, rounds 5 to 7 (2026-11-11 and 12) screened the loss, the schedule, inference and the blocks for a stronger member or a more different one.
- One change reached the board: a cooldown with 2% random-order feeds instead of 10%.
- Weight averaging showed a failure that came back often, members that improve while the pool gets worse.

## Why it was tried

The recipe then was the Muon optimiser, a warmup-stable-decay schedule (WSD), retrieval of similar history items and 10 epochs. Much of it had never been varied. Terms are defined in the [glossary](../../docs/glossary.md).

On the evening of 2026-11-11 the best pool was Cd: 19 members of three recipes, public 0.39208. It read +0.00083 ± 0.00054 on validation over the previous best candidate.

Families (members of one recipe that differ only by seed) correlated at 0.92 to 0.94 within a feed. That is about the level of two seeds of one recipe, and it held even across encoder types.

## What ran

Every comparison is paired per feed on validation (14,423 feeds) and on the holdout of 30,000 random-order train feeds, under a rule written before the run. A random-order feed was shown to the user in random order, not ranked by the site. Every validation and test feed is one.

- Solutions review, 2026-11-11, 17:50 to 22:45 UTC:
  - 16 ranking losses on a CPU proxy (an MLP on 48 features, 300,000 train feeds, 3 to 6 seeds);
  - three variants of the auxiliary loss toward the production display order, on the real ranker: 8 two-epoch runs of about 49 min.
- Round 5, 23:05 to 00:40:
  - input dropout conditioned on listing age (2 seeds, 2 epochs);
  - a refit of the fine-tune on all of validation;
  - a nested check of the fine-tune weight, the weight of the fine-tuned branch in the final score. The weight 0.67 read +0.00024 ± 0.00025 on validation, so 0.5 stayed.
- Round 6, 2026-11-12, 01:00 to 13:00:
  - weight averaging of plateau checkpoints (7 seeds, CPU);
  - test-time augmentation (TTA), averaging predictions over dropout passes (MC dropout), random drops of history events or crops of the history (4.7 CPU process-hours);
  - a cooldown harness, `cd_lane.sh`, with arms for the random-order share, dropout, weight decay and the size of the random-order pool;
  - a structure screen whose winner, unique retrieval slots, got a 10-epoch pair (219 to 244 min per run).
- Round 7, 05:24 to 07:46: 13 block designs and never-varied settings at 2 epochs and 2 seeds, 29 runs, 16.7 hours of training summed over the runs.

The cooldown is the last 20% of training steps, in which the learning rate falls. The harness re-runs only the 2-epoch cooldown of 7 stored 10-epoch members, the roots, from step 77,500, about 24 min per run.

## Results

± is one standard error of the paired difference, and z is the difference divided by it. A blank cell has no read or no verdict of its own.

| Round | Change | Validation | Holdout | Verdict |
|---|---|---|---|---|
| review | ListMLE, Avito's production loss, against softmax cross-entropy (proxy) | -0.00516 (z -6.5) | | softmax kept |
| review | auxiliary loss at temperature 30 | -0.00091 ± 0.00049 | -0.00109 ± 0.00028 | worse |
| review | 12 two-epoch seeds added to the pool | -0.00063 ± 0.00042 | -0.00112 ± 0.00024 | rejected |
| 5 | age-conditioned input dropout | +0.00005 ± 0.00061 | -0.00189 ± 0.00034 | fail |
| 5 | fine-tune refit on all of validation | -0.00025 ± 0.00018 | | fail |
| 6 | weight averaging, single member | +0.00041 ± 0.00040 | +0.00072 | |
| 6 | weight averaging, 7-member ensemble | -0.00247 ± 0.00055 | -0.00164 ± 0.00032 | closed |
| 6 | averaging over 4 history lengths, 6-member ensemble | -0.00100 ± 0.00045 | | harmful |
| 6 | cooldown with 2% random-order feeds instead of 10%, 7 roots | +0.00117 ± 0.00039 | +0.00134 ± 0.00022 | adopted |
| 6 | cooldown with 0% against 2% | +0.00002 ± 0.00025 | +0.00057 ± 0.00014 | fail |
| 6 | unique retrieval slots, 2 epochs | +0.00117 (z 2.48) | -0.00004 | pass |
| 6 | unique retrieval slots, 10-epoch pair | +0.00036 ± 0.00068 | +0.00018 ± 0.00041 | not adopted |
| 7 | nearest miss of 13 designs (DenseFormer depth-weighted averaging of earlier block outputs) | +0.00064 (z 1.32) | -0.00027 | 0 of 13 promoted |

### The 2% cooldown

This is the change of these rounds that reached the board. Its 7-member ensemble also gained: +0.00108 ± 0.00059 on validation and +0.00137 ± 0.00033 on the holdout.

Swapping the family with the re-run cooldown into the 38-member six-family pool moved public from 0.392436 to 0.392723. Members with a re-run cooldown made two of the three families of the next best pool, G3 (20 members, public 0.392820). Every later member, distilled students included, was cooled this way.

A cold item was never a train candidate before the feed's day. With 0% the model lost on the validation feeds whose positives are cold items: -0.00230 ± 0.00051 on them.

### The unique retrieval slots

The unique retrieval slots came out of the test-time study. Of the history events, 30% repeat an item, so tie-breaking decided which copy filled the top-16 retrieval. With unique slots each distinct item takes one slot, with counts per event type.

## What it taught

Weight averaging was the first case of a pattern that came back often: members improved (6 of 7 seeds) and the pool got worse. Averaged seeds became near copies, with a within-feed correlation of 0.975 against 0.941. The 2-epoch seeds failed the same way (correlation 0.978). From then on the cooldown rule also required a 7-member ensemble to be no worse (`r6_eval_cdw2.py`).

A 2-epoch screen is good for stopping ideas and poor for promoting them. It cannot see cooldown effects, since 29 of 29 two-epoch runs selected one of their last two checkpoints. The retrieval-slot gain shrank from +0.00117 at 2 epochs to +0.00036 at 10.

In round 7 every block change cost the feeds with 50 or more candidates: negative in 13 of 13 arms, mean -0.0013 on validation. Norm-free blocks and batch ×2 were worse on both validation and the holdout.

Several arms traded cold items for warm ones and netted zero:

- age-conditioned dropout: +0.0247 on feeds whose positives are all young, -0.0063 on the 72% with none;
- the retrieval slots: +0.009 cold, -0.0034 warm.

Age-conditioned dropout was closed on the holdout alone. On 2026-11-15 the holdout was found biased against this kind of change ([popularity-staleness](../popularity-staleness/)).

Stale validation is validation re-predicted with private-like stale inputs. A later popularity dropout in the cooldown read -0.00066 ± 0.00048 on validation and +0.00001 on stale validation, so the idea stayed closed (decision ledger, 2026-11-15T08:57Z).

Never tested: the linear learning-rate rule for a larger batch, a 7-epoch schedule, momentum and Adam betas.

## Files

Scripts in `code/`:

- `loss_zoo.py`, `agedrop.py`, `recap_avg7.py`: the loss proxy, age-conditioned dropout, weight averaging.
- `cd_lane.sh`: the cooldown harness with every cooldown arm.
- `r6_eval.py`, `r6_eval_cdw2.py`: paired read-outs of round 6.
- `screen.sh`, `eval_screen.py`: round 7 launcher and read-out.

Read-outs in `results/`:

- `review_*`, `round5_*`: read-outs of the review and of round 5.
- `round6_cooldown_*.txt`, `round6_retrieval_slots_*.txt`: the cooldown arms and the retrieval slots.
- `round6_weight_averaging_readout.txt`, `round6_tta_summary.csv`, `round6_family_table.csv`: averaging, test-time augmentation, all families.
- `round7_screen_readout.txt`: all 13 arms of round 7.

### Raw material and records

- Lab folders: `lab/nn_x/solvers/`, `lab/nn_x/r6/`, `lab/nn_x/r7/`, `lab/solvers/`.
- Solutions review, in `lab/reports/`: `SOLUTIONS_REVIEW.md` and the eight `sol_*.md` behind it.
- Round 5: `ROUND5.md` with `r5_agedrop.md`, `r5_refit.md`, `r5_wft.md`.
- Round 6: `R6_PLAN.md` with `r6_recap.md`, `r6_tta.md`, `r6_design.md`, `r6_external.md`.
- Round 7: `R7_ARCH_HPARAM.md` with `r7_research.md`, `r7_build.md`, `r7_screen.md`.
- Records: the experiment registry (a working record, not published).
