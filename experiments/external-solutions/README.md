# Outside solutions, outside data and the leader's margin

In brief:

- On 2026-11-12 a new leader reached 0.398256 on the public board, 0.0054 above the team's best pool of that day (G3, 0.392820). This line asked whether other competitions' techniques, outside data or a channel hidden in the data could explain it, and whether any of them held signal the ranker lacked.
- Nothing from it became an input or a member. On 2026-11-14 the records judged the margin most likely a better model (P 0.55), and P 0.07 for a missed channel. The probes of 2026-11-18 put better modelling or better use of the training data at P 0.64 ([the gap to the top](#the-gap-to-the-top-18-and-19-november)).

Terms such as pool, member and holdout are defined in the [glossary](../../docs/glossary.md).

## Why it was tried

A gap of that size has few sources: a better model, unused data, heavier use of the validation labels, or a missed channel. Each of them can be measured on the competition data.

Two kinds of measurement were used. A ceiling is the gain of an oracle built from validation's own labels, an upper bound for a signal. A nested read fits the weight of a candidate signal on one half of the validation users and scores it on the other half, then the other way round.

Outside solutions had shaped the model once already. On 2026-11-11 write-ups of related competitions (OTTO, H&M, Amazon KDD Cup 2023, RecSys Challenges 2020 to 2024, Russian ODS cups, Expedia hotel search) and surveys of learning-to-rank and position bias were read. `lab/reports/deep_PLAYBOOK.md` merged them into a plan.

The leader's score has not changed since it appeared. On the board of 2026-11-18 03:33 UTC it stood at 0.3982556, and the team was at rank 8 of 532 teams with pick 1 (FD16P20@m0.4_0.2, public 0.3952092, scored on 2026-11-17). On the board of 2026-11-19 03:34 UTC the team was at rank 9 of 541 with pick 2 (FDX16P20:ps, public 0.3952606). On the board snapshot of 2026-11-13 19:43 UTC, place 2 was at 0.395115.

## What ran

Nearly all of it ran on CPU.

- Hypotheses about the leader (2026-11-12): eight hypotheses, each with a measurement where one was possible.
- The known leak (2026-11-12): the same-user later-snapshot boost on G3, submitted once more on purpose ([the leak](../../docs/02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)).
- Heavier use of the validation labels (2026-11-12): a LightGBM proxy with 53 inputs, fitted on 150,000 train feeds, on validation alone (5 folds), or on train and validation rows together with validation rows at weight 1 or 10. A fifth arm added 100 boosting rounds on validation to the train fit.
- Outside data (2026-11-12 and 13): calendar, paydays, weather, events and Avito news for 6 to 29 September as category × time priors. A ceiling came from validation's own same-day category × day × hour click rate.
- The 2025 winning solution of the previous Avito cup (2026-11-13): its click-ratio statistics and item-item scorers rebuilt, then CatBoost with the YetiRank ranking loss on random-order train feeds. Each arm took 10 to 88 minutes on one GPU.
- A scan of winning solutions (2026-11-13): 211 technique rows from about 120 sources, each checked against the runs on record.
- Raw-data forensics (2026-11-14): 12 probes on the raw files and on W8, the pool of 8 full-view distilled students (validation 0.40165).
- A web search (2026-11-14) for write-ups, repositories and posts on this competition.
- Mixture of experts and graph networks (2026-11-15): literature and probes.

The 2025 winners' published code was studied in `lab/nn_x/w25/study/`. None of it is copied into this folder.

## Results

Nested reads and differences are paired per feed, ± one standard error. Validation is 14,423 feeds. The holdout is 30,000 random-order train feeds kept out of training.

| Study | Sample | Result |
|---|---|---|
| hypotheses about the leader | judgement on record | a better model most likely: P 0.50 on 2026-11-12, 0.55 on 2026-11-14, 0.64 on 2026-11-18 (better modelling or better use of the training data) |
| the known leak | public | 0.392819 against 0.392820 for the base |
| heavier use of validation labels | validation, out of fold | best arm +0.00206 ± 0.00101 (train and validation rows at weight 1); validation alone -0.00622 ± 0.00145 |
| outside data | validation and holdout, nested | every prior within ±0.0003; ceiling from validation's own click rate +0.00062 ± 0.00036 in-sample, +0.00002 ± 0.00018 out of fold |
| the 2025 winning solution | validation / holdout | +0.00067 ± 0.00045 over the existing tree member `cb26`, holdout -0.00012 ± 0.00025; best arm 0.38268 on validation, where a member at correlation 0.858 needed 0.3933 |
| scan of winning solutions | prior per technique | 7 new, none with a prior above 0.035; expected gain of all that was left about +0.0003 |
| raw-data forensics | validation, nested | best of 270+ nested reads +0.00044 ± 0.00020, about the maximum of 248 null reads |
| web search | none | nothing beyond the organiser page |
| graph networks | coverage of test rows | not tried; a graph network has no node for 42.8% / 45.6% of public / private candidate rows |
| mixture of experts | validation; validation and holdout | tried on 11 November (+0.00027 ± 0.00047 on validation) and on 19 November inside the distilled students at 4 and 8 seeds; closed ([optimiser-and-modern-tricks](../optimiser-and-modern-tricks/)) |

The five arms of the validation-label test were all read on the 14,423 validation feeds. The arms that fit validation rows predict each user fold from the other four.

| Arm of the LightGBM proxy | Validation NDCG | Difference to the train-only arm |
|---|---|---|
| train only, 150,000 feeds | 0.37064 | reference |
| validation alone | 0.36442 | -0.00622 ± 0.00145 |
| train fit plus 100 rounds on validation | 0.36599 | -0.00465 ± 0.00069 |
| train and validation, validation rows at weight 1 | 0.37271 | +0.00206 ± 0.00101 |
| train and validation, validation rows at weight 10 | 0.37067 | +0.00002 ± 0.00127 |

The forensics found no unused column. Every raw column was already in use, and row order, part files, ids and embedding-table order were unrelated to the positives (relative rank by embedding index 0.4996 ± 0.0019).

A user's earlier labelled train feeds gave -0.00036 to +0.00010 in nested reads on validation.

Feature f5 behaves like a slowly updated item counter, so its value in a later feed tells whether the item was clicked in between. That reads the future. Within validation it was worth +0.00022 ± 0.00027.

Feeds whose best positive W8 puts at rank 8 or below (the top is rank 0) are 50.8% of validation and carry 65.6% of the loss. No simple cue beat the pool inside any slice (`results/forensics_where_the_pool_loses.txt`).

A second seed of one tree arm of the 2025 transfer moved validation by -0.00086 and the holdout by -0.00204. Every ablation difference of the transfer fell inside that one-seed spread and followed the early-stopping iteration.

### The gap to the top, 18 and 19 November

- On the board of 2026-11-18 03:33 UTC the leader still stood at 0.3982556, 0.00305 above pick 1, and places 2 and 3 were 0.00207 and 0.00127 above pick 1. The top 8 places form a continuum, with gaps of 0.00008 to 0.00098.
- On the board of 2026-11-19 03:34 UTC place 2 had risen to 0.3980582, 0.0002 below the leader, and place 3 stood at 0.3968832.
- Five CPU probes re-read the information channels of the files on top of pick 1, as nested reads over user halves. Apart from the known same-user leak (+0.00452 on validation, nothing on public when submitted on 12 November), no channel read above +0.00003 NDCG. The forward f5 echo and history events lost from the snapshot add likelihood (+0.00457 and +0.00115 nats per click) and no NDCG.
- The judgement of 18 November moved to better modelling or better use of the training data, at P 0.64. Net of the selection of each team's best file, place 2 was then about +0.0012 to +0.0020 above the team's level and place 3 +0.0004 to +0.0013. Fresh popularity from outside the files is the only input gap of the right size, and it was not pursued. These probabilities were not re-estimated after place 2 rose on 19 November.
- [docs/08](../../docs/08-outside-ideas.md#what-the-top-of-this-board-may-use) has the hypotheses, the probe table and the options left.
- Code and read-outs are in `lab/nn_x/w12/leader/`. `data_channels/` holds the scripts and the final numbers in `out/p2_read.json`, `out/p3_read.json`, `out/p4_battery.json` and `out/p5_read.json`. `past_audits/` and `outside_practice/` hold one script each, `critic/` holds `c01_checks.py` and `answer_corrected.txt`, and `synthesis/` holds `answer.txt`. The raw logs are not in this repository.

## What it taught

Nothing from this line became an input or a member.

Outside data has no room to work here. Every candidate in a feed shares its timestamp, so a feed-level fact can matter only through an interaction with a candidate attribute. An oracle of that interaction, built from validation's own labels, added +0.0006.

The 2025 recipe was built for retrieval scored by Recall@40, the share of a user's target items found in the top 40. On this data it brought no independent signal.

On 2026-11-14 the records judged the leader's margin most likely due to a better model (P 0.55), with a missed channel at P 0.07. The expected gains of every outside lever still open added up to about +0.0008 on public.

The outside ideas that changed the model came through other lines, and [outside ideas](../../docs/08-outside-ideas.md) lists them:

- random-order feeds inside train ([exploration-detection](../exploration-detection/));
- Muon, the warmup-stable-decay schedule and TabR-style retrieval ([optimiser-and-modern-tricks](../optimiser-and-modern-tricks/));
- ensemble distillation ([distillation](../distillation/)).

Later reads kept two rules from this line. A one-seed tree ablation says nothing below the one-seed spread. A sweep over many columns must be read against the maximum of its nulls.

## Files

`code/`

- `w25_build_target_statistics.py`: click-ratio statistics in the 2025 style, with a time rule against leaks.
- `w25_train_ranker.py`: CatBoost YetiRank arms of the transfer.
- `leader_hypothesis_value_of_val_labels.py`: the validation-label test.
- `forensics_harness.py`: nested read shared by the probes.
- `forensics_raw_files.py`: schema and ordering checks of the raw files.
- `forensics_column_sweep.py`: every stored column re-read on the pool.
- `forensics_same_user_earlier_feeds.py`: the user's earlier labelled feeds.
- `forensics_f5_forward_echo.py`: the leak-class f5 probe, sized to rule it out.

`results/`

- `w25_results_final.txt`: verdict, summary and noise check of the 2025 transfer.
- `leader_hypothesis_value_of_val_labels.json`: the five arms of the validation-label test.
- `forensics_248_column_sweep.tsv`: the sweep sorted by z; the first line names the columns.
- `forensics_where_the_pool_loses.txt`: loss by slice.
- `forensics_same_user_earlier_feeds.txt`, `forensics_f5_forward_echo.txt`, `forensics_f5_floor_persistence.txt`, `forensics_embedding_drift.txt`: the other probes.

### Raw material and records

- `lab/nn_x/w25/`: arms, `eval_w25.py` and the JSON read-outs in `out/`. `study/` holds only `study.md`, the notes of the study. The third-party code and the raw logs are not in this repository.
- `lab/nn_x/audit/top1/` and `lab/nn_x/audit/external/`.
- `lab/nn_x/w12/forensics/` (probes p01 to p12) and `lab/nn_x/w12/lit/`.
- `lab/nn_x/w12/leader/`: the channel probes, board arithmetic, posting-time key and critic checks of 2026-11-18.
- Reports under `lab/reports/`: `EXTERNAL_TOP1_1113.md`, `EXTERNAL_SCAN_1113.md`, `TOP1_GAP_FORENSICS_1114.md`, `TOP1_GAP_EXTERNAL_1114.md`, `LIT_1115/`, `AUDIT_1112.md`, and the `deep_*.md` and `research_*.md` surveys.

The probe scripts of the scan were not kept in the tree.
