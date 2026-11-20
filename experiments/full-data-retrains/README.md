# Retraining the final members on all train feeds

In brief:

- The 20 members of FD16P20 were retrained on all 3,151,208 train feeds, the 30,000 holdout feeds included, and each was accepted against its held-out twin by checks written before the runs.
- FD16P20 read +0.00058 ± 0.00043 against W8P20 on validation and 0.394892 against 0.3950094 on public, level within noise, as forecast. Its 20 members are also those of pick 1, FD16P20@m0.4_0.2. Pick 2, FDX16P20:ps, keeps its 4 popularity-light students and replaces the 16 full-view students with all-data students that also train on pseudo-feeds.
- Pools of 24, 32 and 48 all-data full-view students read below the pool of 16 on validation.

Scores are NDCG, the competition metric. Terms such as holdout, twin, student, popularity-light and random-order pool are defined in the [glossary](../../docs/glossary.md).

## Why it was tried

W8P20 was the team's best public file until 2026-11-17: 8 full-view distilled students of width 192 plus 4 narrow popularity-light students at weight 0.2, public 0.3950094. Every one of its members left the holdout, 30,000 random-order train feeds, out of training and used it to choose its checkpoint.

The team wanted the final files trained on all labelled train data. The line asked whether the final members could be retrained on all 3,151,208 train feeds, and how to accept a member that then has no out-of-sample holdout score.

What the held-out members missed was small: 0.952% of train feeds and 1.877% of train candidates. The holdout feeds made up a larger share, 18.3%, of the random-order pool, the 164,118 train feeds that look most like test (134,118 without the holdout feeds).

The expected gain, +0.00005 to +0.0002 on validation, was below the 0.00030 seed noise of a 16-member pool (`lab/reports/STALE_1115/fulldata.md`). The retrain was done on principle, so it needed proof that it did no harm.

## What ran

### The trainer flag and the teacher check

A copy of the trainer got one flag, `--holdout_in_train`. With the flag on:

- the 30,000 holdout feeds stay in the training batches and in the random-order pool of the cooldown;
- the last step is kept;
- every holdout number is written under an IN-SAMPLE name.

With the flag off, the copy runs the original code path. A syntax-tree comparison and three end-to-end CPU pairs checked this.

A student trained on holdout feeds takes its soft targets for them from the teacher file, so those teacher rows had to be out-of-sample. A check before the runs confirmed it: the 11-member teacher scores 0.3237 on the holdout, against 0.3439 on 30,000 random-order feeds it trained on.

### The acceptance check

`accept_twins.py` compares each all-data member with its held-out twin, the run with the same command and seed. Its rules were written before the runs.

| Check | Passes when |
|---|---|
| A1 | the run's structure is as expected; for example, the log shows 3,151,208 training feeds and the last step kept |
| A2 | the in-sample holdout NDCG is at least +0.00428 above the twin's out-of-sample one |
| A3 | the pool of all-data members is level with the pool of twins on validation, within three times the pool seed noise (0.00090 for 16 members) |

### The runs

On 2026-11-15 these members were retrained and then fine-tuned:

- 16 full-view distilled students: width 192, 10 epochs, two teacher arms;
- 8 popularity-light students of two widths, of which the file uses the 4 wide ones.

| Time | Value |
|---|---|
| Wall-clock per process, full-view student | 220.8 to 258.8 minutes |
| Wall-clock per process, wide popularity-light student | 248.2 to 256.7 minutes |
| Logged minutes of the file's 20 members and their fine-tunes, added up | 88.4 hours |

The plan written that morning had estimated 1.15 GPU-hours per member. The logged times are the later, measured record. They are upper bounds on GPU time, because the GPUs were shared with other processes.

Two more blocks of 16 full-view students (seeds 959 to 974) then tested larger pools. They had no twins, so `accept_without_twin.py` accepted them against the band of the first 16.

## Results

FD16P20 is the file built from these members: 16 all-data full-view students plus 4 all-data popularity-light students at weight 0.2, all of width 192. Differences are paired per feed, and ± is one standard error.

| Check or file | Sample | Result |
|---|---|---|
| A2, 16 full-view members | holdout, in-sample | +0.01581 to +0.01791, bar +0.00428 |
| A3, 16 full-view members against 16 twins | validation | +0.00004 ± 0.00042, band 0.00090 |
| A3, popularity-light, per width (4 each) | validation | +0.00002 ± 0.00053; +0.00001 ± 0.00059 |
| FD16P20 against W8P20 | validation | +0.00058 ± 0.00043 |
| the same pair | public | 0.394892 against 0.3950094: -0.00012, paired SE 0.00056 |
| 24 full-view students against 16 | validation / public | -0.00013 ± 0.00022 / 0.394905 against 0.394892 |
| 32 full-view students against 16 | validation | -0.00034 ± 0.00025 |
| 48 full-view students against 16 | validation / public | -0.00049 ± 0.00028 / 0.39484072 against 0.394892 |
| the 16 full-view students with the cooldown re-run on all data from their stored checkpoints, against FD16P20 | validation / public | -0.00056 ± 0.00028 / 0.39482063 against 0.394892 |
| 16 all-data full-view students that also train on pseudo-feeds (FDX16P20:ps, pick 2), against FD16P20 | validation / public | +0.00015 ± 0.00036 / 0.39526061 against 0.394892 |
| gate leak probe, four all-data files | validation, 117 feeds | +0.0050 to +0.0077 at 1.1 to 1.6 SE |

The leak probe compares the gain on validation feeds whose user has a later history snapshot with the gain on the rest. Decomposed, the all-data change alone read +0.00145 ± 0.00345, and the two seed halves had opposite signs. The probe is a tail statistic on 117 feeds, and it showed no leak signature.

## What it taught

The retrain was adopted. FD16P20 became the best file trained on all data and was pick 1 until 2026-11-18. On public it is level with W8P20, the held-out file, within noise, as forecast.

On 2026-11-18 FD16P20@m0.4_0.2 took its place as pick 1. It has the same 20 members, with the popularity-light weight raised to 0.4 on feeds of fewer than 50 candidates (public 0.3952092; [popularity-views](../popularity-views/)).

For pick 2 the team first waived the all-data rule and chose W8P20, the strongest file built from other runs. On 2026-11-19 FDX16P20:ps took its place: the design of FD16P20 with its 16 full-view students retrained on all train feeds plus pseudo-feeds ([distillation](../distillation/)). Both final files are now trained on all train feeds ([final picks](../../docs/11-final-picks.md)).

The twin check found no defect, and that is all it can do. An all-data member has no out-of-sample holdout score, so its acceptance rests on structure, the in-sample lift and parity with its twin on validation. No twin check is on record for the 16 pseudo-feed students of pick 2; the file passed the gate.

The growth to 32 and 48 students failed its non-inferiority rule, although the new members were exchangeable with the old. Their mean train-only member scores on validation were 0.39931 and 0.39971 for the two teachers, against 0.39940 and 0.39951 for the first 16.

The 16-member reference was a favourable draw: a pool above the mean of same-size pools of its recipe by seed luck. The nested files of 16, 32 and 48 students read 0.40271, 0.40237 and 0.40222 on validation, and the family's unbiased level is about 0.4023 to 0.4024. A bar set against a lucky file fails by noise more often than its nominal rate ([pool-size-and-growth](../pool-size-and-growth/)).

Seventeen scored files of this family span 0.39436 to 0.39526 on public (mean 0.39486, standard deviation 0.00025). The board cannot separate them.

## Files

`code/`

- `trainer_patch.diff`: the whole change to the trainer. A header is prepended, so it may not apply cleanly with `patch`.
- `holdout_in_train_helper.py`: what the flag does.
- `teacher_rows_of_holdout_feeds.py`: the teacher check.
- `lane_full.sh`: runner that rebuilt each member's command with the flag.
- `accept_twins.py`, `accept_without_twin.py`: acceptance with and without twins.
- `build_and_gate.sh`: builds one all-data file and gates it.
- `watch_pool_growth.sh`: accepts a block of new students, builds and reads the larger file.

`results/`

- `acceptance_rules_predeclared.txt`: the rules.
- `members_training_data_facts.txt`: what the held-out finals trained on.
- `teacher_rows_of_holdout_feeds.txt`: the teacher check.
- `acceptance_16_wide_students.txt`, `acceptance_8_popularity_light_students.txt`: acceptance, cut to the final run.
- `gate_all_data_16_plus_4.txt`: the pre-submission gate on FD16P20.

### Raw material and records

- Lab folders: `lab/nn_x/w12/stale/fulldata/` (trainer copies, smoke tests, logs), `lab/nn_x/w12/stale/fulldata_review/` (an independent re-derivation of the facts and the code path), `lab/nn_x/w12/kd/fd/` (runners, watchers).
- Reports: `lab/reports/STALE_1115/fulldata.md`.
- Records: the decision ledger and the freeze protocol (working records, not published).
- In context: [the solution, the final files](../../docs/03-solution.md), [final picks](../../docs/11-final-picks.md).
