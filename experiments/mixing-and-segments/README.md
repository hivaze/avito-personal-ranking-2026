# Mixing rules, user segments and fresh exposure

In brief:

- On 2026-11-15 three studies on stored predictions looked for gains without training: mixing weights that depend on the feed, a user segment where the model is weak, and counts of how often other users' validation and test feeds showed an item. None of them changed the final files.
- On 2026-11-17 and 18 try files came back to the mixing weight. One of them, FD16P20@m0.4_0.2, scored 0.3952092 on public, +0.00032 over FD16P20 at a paired standard error (SE) of about 0.0003, and became pick 1.

A try file is a file scored on the public board for information. Other terms are in the [glossary](../../docs/glossary.md).

## Why it was tried

A final file mixes four pools with two fixed weights. The popularity-light pool gets 0.2 against the full-view pool, and the fine-tuned branch (predictions after a fine-tune on the validation days) gets 0.5 against the train-only branch. Popularity-light students read one exposure count in place of the 8 popularity inputs; full-view students read all inputs.

A feed-size rule lets the popularity-light weight depend on whether a feed has fewer than 50 candidates. It is written `@m<short>_<long>`, with `<short>` the weight on feeds of fewer than 50 candidates and `<long>` the weight on the others.

The form `@m0.2_0.4`, weight 0.4 on feeds of 50 or more candidates, had read +0.00026 to +0.00071 on validation, in sample, on four member sets. In sample means the weights were chosen and scored on the same feeds.

Users at the 256-event history cap score lower than the rest. The segment study looked for the kind of user on which the model is weak: users with no history, anonymous users, sparse users and users at the cap.

Popularity counted on train days covers fewer test candidates than validation candidates. Counts taken inside the validation and test feeds might make up part of that loss.

## What ran

Each study of 2026-11-15 wrote its criteria to a time-stamped declaration before its first labelled number. A second analysis with its own code (the check) re-ran each study.

### Mixing

In `lab/nn_x/w12/mix/`:

- A search over 95 rule families (131,547 rules) on label-free feed descriptors. Weights were chosen on four user-grouped folds and scored on the fifth, a read called nested.
- A second search over seven learned gate families.
- Both searches ran on four member sets: 16 or 8 full-view distilled students, each with 4 popularity-light students. Each set was either all-data (trained on every train feed) or held-out (trained without the 30,000 holdout feeds).
- The check re-ran both with its own folds against a null of 400 user-level sign flips.
- The `@m0.2_0.4` file went to the public board against its plain twin FD16P20 (16 all-data full-view students plus 4 popularity-light students at weight 0.2, public 0.394892).

### Segments

In `lab/nn_x/w12/seg/`:

- NDCG of FD16P20 by kind of user.
- A replay of stored members with the history cut to its newest 192, 128 or 64 events.
- A refill replay that adds events from the user's earlier snapshot (inference only, 9.8 GPU-minutes).

### Fresh exposure

In `lab/nn_x/w12/fresh/`, each candidate got the number of other users' validation and test feeds that showed the item, in six time windows. The count was read as a nested residual on the final score, a correction whose weight is fitted in user folds (5 folds × 3 seeds, 579 s on CPU).

### Try files

The try files of 2026-11-17 and 18 were built by `lab/nn_x/w12/kd/build_r4.py` from stored member predictions:

- FD16P20 under the reverse form `@m0.4_0.2`, for which no rule had been written.
- The validation curve of the short-feed weight from 0.2 to 0.8, with the long-feed weight at 0.2.
- The long-feed weight at 0.1 and 0, with the short-feed weight at 0.4.
- The same rule with 8 or 12 popularity-light students.

Then came A1 (`lab/nn_x/w12/fixmiss/build_a1.py`), the one correction kept by the error analysis of 2026-11-18. On feeds of 50 or more candidates, the popularity-light part of the fine-tuned branch is taken from the popularity-light students' train-only predictions. The threshold of 50 was fixed before the analysis scanned for corrections.

## Results

± is one standard error. On public, the paired SE is the validation per-feed SE of the file difference scaled to the 8,562 public feeds. The public-mix and private-mix views re-weight validation so that feed size and cold share match the public or private part of test.

### Studies of 2026-11-15

| Study | Measurement | Sample | Result |
|---|---|---|---|
| mixing | `@m0.2_0.4` minus plain weights, 16 + 4 all-data students: in sample / nested (check) | validation | +0.00026 ± 0.00021 / +0.00002 ± 0.00027 |
| mixing | `@m0.2_0.4` file against FD16P20 | public | 0.3947465 against 0.394892 (-0.00015, paired SE about 0.00027) |
| mixing | rule families passing the declared criteria | validation, 4 member sets | 0 of 95 |
| segments | no history; anonymous user id 0 (share of test feeds) | validation | 0.3784 ± 0.0176 (0.86%); 0.4093 ± 0.0101 (4.12%) |
| segments | feeds at the cap against the rest (28.81% of test feeds) | validation | 0.3906 against 0.4077, -0.01711 ± 0.00443 |
| segments | refill: earlier-snapshot events in slots 129-256 minus empty slots | holdout (2,358 feeds with events to add) / validation | +0.00117 ± 0.00082 / +0.00269 ± 0.00157 |
| fresh exposure | within-feed AUC of the strictly-earlier count | validation | 0.4970 ± 0.0012 |
| fresh exposure | nested gain over the final score: conditional logit / direct NDCG (check) | validation | -0.00044 ± 0.00015 / -0.00004 ± 0.00020 |

### Try files

| Measurement | Sample | Result |
|---|---|---|
| `@m0.4_0.2` minus FD16P20, in sample | validation; public-mix view; private-mix view | +0.00022 ± 0.00024; +0.00027; +0.00033 |
| FD16P20@m0.4_0.2 against FD16P20 | public | 0.3952092 against 0.394892 (+0.00032, paired SE about 0.0003) |
| short-feed weight 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.8, long-feed weight 0.2 | validation | 0.40271 / 0.40266 / 0.40293 / 0.40251 / 0.40232 / 0.40184 |
| long-feed weight 0.1 / 0 against 0.2, short-feed weight 0.4 | validation | -0.00025 ± 0.00013 / -0.00068 ± 0.00020 |
| `@m0.4_0.2` with 8 / 12 popularity-light students against 4 | validation | -0.00057 ± 0.00021 / -0.00039 ± 0.00021 |
| A1 against FD16P20@m0.4_0.2 | validation; public | +0.00017 ± 0.00014; 0.395036 (-0.00017, paired SE about 0.00018) |

### The `hx` training run, 2026-11-17

On 2026-11-17 the longer history got a training run after all. Arm `hx` was 16 cooldown re-runs of the held-out full-view students that read up to 128 older events from the user's most recent earlier snapshot. A cooldown re-run is a 2-epoch run of only the cooldown, started from a stored plateau checkpoint.

| `hx` against its reference | Result |
|---|---|
| the 1,132 validation feeds the extension reaches | +0.00150 ± 0.00117 |
| the 2,159 holdout feeds it reaches | +0.00001 ± 0.00058 |
| randomization p (bar 0.05) | 0.16 |
| all validation feeds | -0.00013 ± 0.00028 |

### Two ideas closed on 2026-11-13

Two related ideas closed at CPU gates on 2026-11-13:

- Routing each feed to one of several pools: router AUC 0.494 to 0.511 on 31 label-free feed descriptors, against a permuted null of 0.496 to 0.503.
- Per-vertical expert heads, where a vertical is one of the eleven groups of category codes: 99.2% of validation feeds hold more than one vertical, and a per-vertical correction on the best click-label pool read -0.00095 ± 0.00039 on validation against one global correction.

Both are in section 4 of `lab/reports/SOLVER_POWER_1113.md` and in the coverage map. Their scripts are not in the tree.

## What it taught

None of the three studies of 2026-11-15 changed the final files. The reverse form of the feed-size rule did: FD16P20@m0.4_0.2 is pick 1.

### In-sample reads of mixing rules

In-sample reads of `@m0.2_0.4` were 2.4 to 13 times its nested value. A greedy scan of all rules on a placebo with nothing to find read +0.00057 in sample (95th percentile +0.00082). In-sample reads of scanned rules below about +0.0008 on validation are therefore not evidence.

The check judged `@m0.2_0.4` "not shown" on the all-data members. No learned gate beat it nested either (-0.00022 ± 0.00021 on the all-data students), and the public reading of -0.00015 closed it.

### The short-feed weight in pick 1

By the same measure, the +0.00022 of `@m0.4_0.2` on validation was no evidence either. The public board is its only read outside the sample it was chosen on. There it scored 0.3952092, then the team's best public score, +0.00032 over FD16P20 at a paired standard error of about 0.0003. On 2026-11-19 pick 2, FDX16P20:ps, scored 0.00005 higher.

The rule leaves every feed of 50 or more candidates as it was. The whole difference sits on the 38.7% of public feeds with fewer candidates, the side that `@m0.2_0.4` had not touched.

The weight 0.4 is not read as an optimum. On validation it is a single high point of the short-feed curve, which is flat at 0.2 and 0.3 and falls from 0.5 on. It is also a high point of this draw of 4 popularity-light students.

With 8 or 12 popularity-light students the same rule read 0.00057 and 0.00039 lower. On FDXH16P20:ps, the file with 8 of its 16 full-view students replaced by pseudo-feed students (students that also train on feeds built from user histories), the same rule lowered validation from 0.40298 to 0.40262.

### The long-feed weight stays at 0.2

On long feeds, both changes to the popularity-light part landed about 0.0004 below their validation reading:

- `@m0.2_0.4`: +0.00026 on validation, -0.00015 on public.
- A1: +0.00017 ± 0.00014 on validation, -0.00017 on public.

Lower long-feed weights read worse on validation, so 0.2 stays there.

FD16P20@m0.4_0.2 has the same 20 members as FD16P20. The two are expected to be level on private, with a difference of about +0.00005. [docs/11](../../docs/11-final-picks.md) gives the choice of the final files.

### User segments

Cold, anonymous and sparse users are small segments, and the model is not weak there. The gap at the cap goes mostly with the kind of user: rankers that never read the history lose there too.

The diagnosis first saw no sign that the model is starved of history at the cap. Its review, written about 40 minutes later, found that removing the oldest 64 stored events at inference costs those feeds -0.0023 ± 0.0008 on validation (six members).

The refill replay lacked the data to decide. On the holdout 3,235 feeds qualified, where the rule's power estimate had assumed about 11,600. Even the stored events placed in the same slots read only z 1.58. The line closed without a training run, for lack of power.

The mechanism is probably real. It would be worth +0.00015 to +0.0003 at pool level if a retrained model captured all of it. Train rows could not be built like test rows either: the median gap to the earlier snapshot is 2.8 days on holdout feeds against 14.2 on test feeds.

The `hx` training run of 2026-11-17 did not pass either: +0.00150 ± 0.00117 on the validation feeds it reaches, +0.00001 ± 0.00058 on the holdout feeds, randomization p 0.16.

### Fresh exposure

An appearance in another random-order feed only says that the item is in the candidate pool. Clicks on candidates shown elsewhere were 1.008 ± 0.021 times what the final score expected; a usable input would have needed about 1.35.

### Routing

A feed router had nothing to route on. Its AUC on label-free feed descriptors stayed inside the permuted null, and almost every feed mixes verticals, so per-vertical heads have no clean feeds to learn from.

## Files

`code/`

- `mix_rulesA_lib.py`, `mix_rulesA_search_real.py`: nested rule search.
- `mix_gateB_gate.py`: learned gate families.
- `mix_review_placebo.py`: the sign-flip null.
- `seg_trunc_replay.py`: truncation replay.
- `seg_refill_s2_replay.py`: refill replay.
- `fresh_s1_build.py`: the exposure counts.
- `fresh_nested.py`: the nested residual read.

`results/`

- `summary.csv`: these numbers and more, with sources.
- `mix_*`: rule search (first member set), check, null, learned gates.
- `seg_*`: segment table, truncation replay, refill tables.
- `fresh_*`: build, AUC, nested reads.

### Raw material and records

The three studies:

- Folders: `lab/nn_x/w12/mix/`, `lab/nn_x/w12/seg/`, `lab/nn_x/w12/fresh/`. Each sub-study folder holds its `DECLARATION.txt`.
- Reports: `lab/reports/MIX_1115/`, `lab/reports/SEG_1115/` and `lab/reports/FRESH_1115/`, each with the study reports and their reviews.

The try files and A1 have no copy in this folder:

- Builders: `lab/nn_x/w12/kd/build_r4.py` (the `@m` keys) and `lab/nn_x/w12/fixmiss/build_a1.py`.
- Gate outputs: `gate_FD16P20_m04_02kd.txt` and `gate_FD16P20_m04_02_A1kd.txt` in `lab/nn_x/w12/kd/out/`.
- The recipe of pick 1: `configs/ensembles/fd16p20_m04_02.toml`.

The `hx` arm: `lab/nn_x/w12/gap1117/` (`lifelong/`, `reads/hx_vs_ref_ext.json`).

Records: the decision ledger (a working record, not published).
