# Popularity staleness and the biased holdout

In brief:

- On 2026-11-15 the public board and the in-period holdout disagreed about the popularity-light admixture. The holdout's popularity inputs are much fresher than those of test (mean staleness 1.75 days on the holdout, 11.23 on private), which biases it on such changes.
- From 2026-11-15 02:15 UTC the holdout lost its veto on changes to how a model uses popularity.
- No repair in training or at inference was adopted, and the admixture stayed the only lever in the final files.

Terms are defined in the [glossary](../../docs/glossary.md); [docs/04](../../docs/04-validation.md#the-witnesses) describes the witnesses, the samples a change is judged on. A difference is paired per feed, and "±" is one standard error (SE) of it. The admixture is the weight of the popularity-light students in the pool: 0.2 in both final files, and 0.4 in pick 1 on short feeds, those of fewer than 50 candidates.

## Why it was tried

The public board rewarded the admixture a second time. W8P20 scored 0.3950094 against 0.394486 for W8, +0.000523 at a paired SE of 0.00041. W8 is 8 full-view distilled students of width 192; W8P20 adds 4 narrow popularity-light students at weight 0.2.

The holdout of 30,000 random-order train feeds read the same contrast at -0.00063 ± 0.00022.

The suspect was the age of the popularity inputs, which are counts over train days. Warm-item staleness is the mean number of days since the item's last train-candidate day:

| Sample | Warm-item staleness (days) |
|---|---|
| Holdout | 1.75 |
| Validation | 5.69 |
| Public | 9.34 |
| Private | 11.23 |

Members trained on fresh counts may lean on them more than test supports. If so, the holdout is biased for every change in how a model uses popularity, and a model could be taught the staleness it meets on test.

## What ran

From 2026-11-14 22:17 UTC to 2026-11-16 the work ran on CPU, except the cooldown re-runs and one inference job on GPUs. Each rule was time-stamped before the numbers it judges.

- The holdout's popularity inputs were rebuilt at the lag of test rows, giving the lagged holdout (`build_lag2.py`). The lag is 6 to 9 days counted from the end of the train data at 00:00 MSK on 20 September, which is 7 to 10 days after the last train day as [docs/01](../../docs/01-competition-and-data.md#time-lag) counts it.
- A label-free gate of five criteria (`gate.py`) decides where the lagged holdout may be read. A CPU replay reproduced 12 members' stored holdout scores within 0.00018.
- Two witnesses got scorers. The lagged holdout is scored by `lagged_score.py` (3.2 CPU minutes per member and view). Stale validation (stale-val in the records) is validation re-predicted with private-like stale inputs. `c_calib2.py` calibrated both on scored public pairs.
- An audit of past decisions that leaned on the holdout (reverse check: `read2.py`).
- Cooldown re-runs of the 16 held-out full-view students on stale views of the popularity inputs, 16 re-runs per arm, about 51 minutes each. The cooldown is the last 20% of training steps.
- A pre-mortem of the candidate files for the private board, on stored predictions only.
- EQ-B: inputs equalised to the test lag at inference only (`eqb_pred.py`).

The cooldown arms:

| Arm | Stale view |
|---|---|
| hz | half dose on 5 inputs |
| hz2 | a causal test-like view of all 8 inputs and the exposure count for 246,745,836 train rows (`views.py`, swapped in by `cd_arms6b.py`) |
| hz2n | variants of hz2 and of hz that also blank the encoder's item-ID score on rows made cold |

## Results

### Which witness predicted public

The ledger holds six scored pairs; three are shown here, and all six are in [docs/04](../../docs/04-validation.md#which-witness-predicted-public).

| Pair | Validation | Public-mix view | Holdout | Public |
|---|---|---|---|---|
| W8P20 - W8 | +0.00049 | +0.00063 | -0.00063 | +0.000523 |
| C3 - G3 (8 popularity-light click-label members added to the best click-label pool, with a 3-seed fine-tune) | +0.00069 | +0.00108 | -0.00073 | +0.000634 |
| W8TO - W8 (W8 without its fine-tuned half) | -0.00009 | -0.00066 | 0 by construction | -0.000477 |

The public-mix view had the public sign on the 5 pairs where public moved; the sixth was a tie. Plain validation had it on 5 of 6. The holdout was wrong on both popularity-light pairs.

### The lagged holdout

The gate failed on all 30,000 feeds. The failing criterion was the exposure count on listings 10-24 days old. Its zero share on the lagged holdout was 0.246, against 0.130 on private rows: a gap of 0.116, over the bar of 0.10 (`poplag_gate_v3.txt`). The gate held on the 13,424 feeds of days from Sep 13.

On those feeds, private-like inputs in place of fresh ones changed the two pools:

| Pool | Change under private-like inputs |
|---|---|
| W8 | -0.0033 ± 0.0008 |
| Popularity-light | -0.00049 ± 0.00038 |

The 0.2 admixture over W8 moved from -0.00050 ± 0.00032 with fresh inputs to +0.00024 ± 0.00031 with private-like ones.

The pre-mortem showed +0.00024 to be the best of nine compositions, whose mean is +0.00003 ± 0.00023. The lagged holdout gave neither a veto nor a confirmation.

### Repairs

| Repair | Witness | Result | Verdict |
|---|---|---|---|
| hz, 16-pool, last step | validation / stale-val | +0.00024 ± 0.00038 / +0.00120 ± 0.00045 | pool pass |
| hz, holdout-selected checkpoints | validation | -0.00009 ± 0.00040 | no pass |
| hz2, causal view | validation / stale-val / holdout | +0.00032 ± 0.00057 / +0.00141 ± 0.00060 / -0.00282 ± 0.00038 | pass; against hz +0.00017 ± 0.00046 |
| hz2n variants against hz | stale-val | +0.00046 ± 0.00048; -0.00017 ± 0.00023 | fail |
| hz as a file, 16 students | final validation / private mix / public | -0.00047 ± 0.00035 / -0.00072 ± 0.00060 / +0.00008 against FD16P20 (paired SE 0.00045) | closed; built on 2026-11-17, scored on 2026-11-19 |
| popularity dropout 0.3 in the cooldown | validation / stale-val | -0.00066 ± 0.00048 / +0.00001 ± 0.00050 | closed |
| EQ-B, file | validation / private mix | -0.00024 ± 0.00017 / -0.00013 ± 0.00027 | closed |
| popularity-light weight 0.4 on long feeds (`@m0.2_0.4`) | public | -0.00015 (-0.5 SE) | long-feed weight stays 0.2 |
| popularity-light weight 0.4 on short feeds (`@m0.4_0.2`), 2026-11-17 | public | +0.00032 (about +1 SE) | pick 1 |
| popularity-light part of the fine-tuned branch from train-only predictions on long feeds (A1), 2026-11-18 | public | -0.00017 against `@m0.4_0.2` (about -0.9 SE) | not adopted |

The two hz rows are two verdicts on record. The witness report judged the holdout-selected checkpoints and gave no pass. Stale-val read +0.00113 ± 0.00043 there, and +0.00077 ± 0.00044 in a newer read.

A ledger note of 07:21 UTC says the last-step read governs, because the amendment that chose the last step predates every hz number. The file failed either way. The full story of hz is failure 19 in [docs/07](../../docs/07-what-failed.md#19-a-cooldown-on-stale-popularity-inputs).

## What it taught

The holdout was biased for a whole family of ideas: changes to how a model uses popularity. Its popularity inputs are fresher than those of test. From 2026-11-15 02:15 UTC it was no longer a veto in that family.

From then on, such decisions rested on validation, stale-val, the private-mix view, the lagged holdout where its gate holds, and public. [docs/04](../../docs/04-validation.md#rules-written-before-the-read) tells the policy and the audit.

The audit screened the 277 idea rows that the coverage map held at 03:40 UTC that day; the catalogue in [docs/07](../../docs/07-what-failed.md), built later from more records, has 312 entries. The audit listed 38 decisions that leaned on the holdout, 15 of them in this family, and re-read 12 from stored predictions.

Ten of the 12 passed the new bar: validation not negative and the private-mix view at least +1 SE. The ten reads cover three ideas, and these three were re-opened: popularity-light pool members, the same view inside the teacher, and hz. Nothing outside the family changed status, and no adopted setting was contradicted.

Staleness is real. On holdout days from Sep 13 the W8 pool lost 0.0045 ± 0.0007 under the causal view of arm hz2 (0.0025 ± 0.0009 on Sep 17 to 19 alone), and 0.0033 ± 0.0008 under the private-like inputs of the lagged holdout. Every repair, though, substituted for the 0.2 popularity-light admixture. On top of it, hz kept 43% of its stale-val effect and nothing on validation. A pool-level pass did not make a better file.

The one public read of a stale-view file came on 2026-11-19. FDX16P20:hz scored 0.3949761, 0.00008 above FD16P20, where validation had read -0.00047 and the public-mix view -0.00072. Public read it about 0.0005 better than validation, a weak sign that public inputs behave stale. The public difference is inside its paired SE of 0.00045, and no pick changed.

The admixture stayed the only adopted lever: at 0.2, and in pick 1 at 0.4 on feeds of fewer than 50 candidates. That last change was scored on 2026-11-17 for information ([mixing-and-segments](../mixing-and-segments/)). The two changes of the popularity-light part on long feeds that reached the board both landed about 0.0004 below their validation reading.

A model told how stale its counters are (lag conditioning) was planned and has no run on record.

## Files

`code/`

- `build_lag2.py`: holdout inputs at test lag.
- `gate.py`: label-free parity gate.
- `lagged_score.py`: lagged-holdout scorer.
- `c_calib2.py`: witness calibration.
- `read2.py`: audit reverse check.
- `views.py`: causal stale view for train rows.
- `cd_arms6b.py`: cooldown hook for stale views.
- `eqb_pred.py`: lag-equalised inference.

`results/`

- `poplag_gate_v3.txt`: gate, all feeds.
- `poplag_gate_subset.txt`: gate by day group.
- `poplag_control.txt`: CPU replay control.
- `witness_control_A.txt`: scorer control.
- `witness_calibration_cut.txt`: calibration summary (cut).
- `witness_dose.txt`: admixture weight per witness.
- `witness_read_first_stale_cooldown.txt`: click-label stale cooldown.
- `witness_hz_tables.txt`: hz reads.
- `veto_predeclared.txt`: audit rules.
- `veto_read1.txt`: first audit re-read.
- `hz2_holdout_read_late_days.txt`: causal-view read.
- `hz2n_gate_table.txt`: hz2n view checks.
- `premortem_t1_witness_table.txt`: every witness, every file.
- `premortem_t3_weight.txt`: admixture weights.
- `premortem_t6_lagged_holdout.txt`: lagged-holdout reads.
- `premortem_t8_same_view_control.txt`: same-view control.
- `eqb_predeclared.txt`: EQ-B rule.
- `eqb_read_val.txt`: EQ-B read.
- `eqb_read_val.json`: its numbers.

### Raw material and records

- Lab folders: `lab/nn_x/w12/stale/` (veto, witness, hz2, hz2n, premortem), `lab/nn_x/w12/next/poplag/` and `lab/nn_x/w12/eqb_1115/`.
- Reports: `lab/reports/STALE_1115/` (witness, veto, hz2, hz2_review, hz2n, hz2n_review, premortem), `lab/reports/NEXT_1115/poplag.md` and `poplag_verify.md`.
- Records: the decision ledger (a working record, not published).
- The fine-tune with stale inputs is in [fine-tune](../fine-tune/).
