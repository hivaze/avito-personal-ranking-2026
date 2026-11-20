# Popularity views: rankers that see little or no popularity

In brief:

- On 2026-11-13 a 2-epoch removal screen measured what the 8 popularity inputs were worth. Members were then trained on a popularity-free view (none of them) and a popularity-light view (one exposure count in their place).
- Added to the best click-label pool, the popularity-light view gained on public while the holdout read it negative. As distilled students it is in both final files.
- The popularity-free view never helped on public and was not adopted.

Terms are defined in the [glossary](../../docs/glossary.md); [docs/04](../../docs/04-validation.md#the-witnesses) describes the witnesses, the samples a change is judged on. A difference is paired per feed, and "±" is one standard error (SE) of it.

## Why it was tried

The ranker reads eight popularity inputs: leave-user-out counts of the item in all histories, and co-visitation scores. On 2026-11-13 the question was what they were worth on the current recipe. A second question was whether members trained without them, or with one exposure count in their place, would add to the best pool.

That pool is G3: 20 click-label rankers, validation NDCG 0.40032, holdout 0.32379, public 0.392820.

Popularity was the input family most likely to look different on test, where rows sit 6-9 days after the end of the dense train data. A member that leans less on it should make different errors, which a pool of near copies needs.

The decoding of the anonymised features ([codes-and-embeddings](../codes-and-embeddings/)) added a mechanism. The feature f5 is an unsmoothed engagement rate, so a model without counts cannot tell a reliable high f5 from a noisy one. One capped exposure count should fix that.

## What ran

A 2-epoch removal screen of the recipe came first: 2 base runs and 12 arms on seeds 23 and 24, 26 runs of 11-55 minutes on two GPUs (`abl_lane.sh`, flags in `abl_arms.py`).

Its read-out rules (`abl_eval.py`) were frozen at 00:08 UTC on 2026-11-13, before any run: paired per feed, pooled over seeds, last checkpoint, SE calibrated with run-to-run noise.

One arm dropped the 8 popularity inputs (the popularity-free view). Two follow-ups put a single input back:

- popularity-light view: log1p(min(n, 50)), n being the number of times the item was shown in train feeds before the row's day (`build_xp_imp_c50.py`);
- click-rate view: clip(log(rate / 0.0159), -2, 2) of the item's smoothed train click rate (`build_xp_ctr.py`).

The popularity-free view and each light view that passed its screen were trained as 10-epoch families of 4 seeds (`launch_e10.sh`; 56.8-59.7 minutes per popularity-light member). The members were fine-tuned on validation and added to G3.

A rule written before the members existed (`decide9.py`) set four conditions on each pool:

- the holdout rises by more than 1 SE;
- validation does not fall;
- no slice is below -2 SE;
- the private-mix reweighting of validation is not below -1 SE.

## Results

### The screen and the 10-epoch members

Screen, 2 epochs, difference ± calibrated SE:

| Arm | Against | Validation | Holdout |
|---|---|---|---|
| popularity-free | base | -0.00234 ± 0.00094 | -0.00743 ± 0.00061 |
| popularity-light | popularity-free | +0.00116 ± 0.00065 | +0.00144 ± 0.00038 |
| click-rate view | popularity-free | +0.00159 ± 0.00063 | +0.00133 |
| popularity-light | base | -0.00118 ± 0.00088 | -0.00599 ± 0.00058 |

Without popularity, validation feeds whose positives were all cold gained +0.02273, and feeds with no cold positive read -0.01283.

The screen also measured the rest of the recipe, and nothing could be removed (`results/abl_eval_summary_cut.txt`):

| Component dropped | Validation | Holdout |
|---|---|---|
| retrieval of similar history items | -0.00337 | -0.00294 |
| production-order auxiliary loss | -0.00219 | -0.00229 |
| set attention | -0.00198 | -0.00142 |

The 10-epoch families confirmed the member-level loss. Mean member against base members:

| Family | Validation | Holdout |
|---|---|---|
| popularity-free | -0.00251 ± 0.00081 | -0.00807 ± 0.00055 |
| popularity-light | -0.00236 ± 0.00080 | -0.00623 ± 0.00053 |

### What the exposure count did

The exposure count did not do what it was built for. On the validation cell of high f5 with fewer than 5 exposures, actual over expected clicks was 0.938 for both the popularity-free and the popularity-light screen pools, against 0.969 for the base.

Its gain over the popularity-free arm sat on feeds with no cold positive (+0.00236 ± 0.00081).

The click-rate view correlated 0.9863 with the popularity-light view on validation, about as much as two seed groups of the popularity-light view correlate with each other (0.9864, `pop1_8_members_result.txt`). It was closed as the same view.

### Pools against G3

Paired per feed; the last column is the public difference against 0.392820.

| File | Members | Validation | Holdout | Private mix | Public | Difference |
|---|---|---|---|---|---|---|
| Q1a: G3 + 4 popularity-free | 24 | +0.00036 ± 0.00030 | -0.00052 ± 0.00021 | +0.00112 ± 0.00048 | 0.392886 | +0.000066 |
| Q2a: G3 + 4 popularity-light | 24 | +0.00029 ± 0.00031 | -0.00002 ± 0.00021 | +0.00062 ± 0.00050 | 0.393205 | +0.000385 |
| Q3: G3 + 4 click-rate | 24 | +0.00015 ± 0.00031 | -0.00023 ± 0.00022 | +0.00084 ± 0.00049 | not sent | |
| C1: G3 + 8 popularity-light | 28 | +0.00057 ± 0.00040 | -0.00073 ± 0.00029 | +0.00107 ± 0.00062 | 0.393394 | +0.000574 |
| C3: C1 with 3-seed fine-tune | 28 | +0.00069 ± 0.00040 | -0.00073 | +0.00130 ± 0.00062 | 0.393454 | +0.000634 |

Q1a, Q2a and C1 failed their written rules on the holdout clause, and Q3 failed on validation as well. C3 passed a separate non-inferiority rule against C1.

Q1a, Q2a, C1 and C3 were built and gated anyway, and sent as declared probes of the public level.

Paired public SEs for these pairs were only estimated that day: the +0.000385 of Q2a was put at 0.6 to 0.8 SE. Comparable pairs two days later had SEs of 0.0004 to 0.0006.

C3's score was assigned by elimination. The submission message named the Q2a file, which had already scored.

### The popularity-free view as distilled students

On 2026-11-15 the popularity-free view came back as 4 all-data distilled students added to the 0.2 part of FD16P20. FD16P20 is 16 all-data full-view students plus 4 popularity-light students at weight 0.2 (public 0.394892). A control file has 4 more popularity-light students in place of the popularity-free ones.

| Draw | Validation | Control files it is above (of 70, same size) | Against FD16P20 |
|---|---|---|---|
| FD16P20vf, first draw | 0.40286 | 70 | +0.00015 ± 0.00020 |
| fresh seeds 955-958 | | 16 | -0.00024 ± 0.00018 |

The rule asked for a replication with fresh seeds, above at least 63 of the 70 controls and no more than one SE below FD16P20 ([docs/04](../../docs/04-validation.md#a-rule-that-failed-the-popularity-free-view)). The replication failed, so the view was not adopted. Public read 0.394630 for FD16P20vf, -0.00026 against FD16P20 at a paired SE of 0.00026.

### The short-feed weight

On 2026-11-17 the other side of the feed-size rule of 2026-11-15 (`@m0.2_0.4`, weight 0.4 on long feeds) was tried on the 20 members of FD16P20. FD16P20@m0.4_0.2 puts popularity-light weight 0.4 on feeds of fewer than 50 candidates and 0.2 on the rest.

Against FD16P20 it read +0.00022 ± 0.00024 on validation, +0.00027 on the public-mix view and +0.00033 on the private-mix view. It scored 0.3952092 on public, +0.00032 at about one paired SE. That was then the team's best public score, and the file is pick 1. On 2026-11-19 pick 2, FDX16P20:ps, scored 0.00005 higher (0.39526061).

The short-feed weight, with 0.2 on long feeds:

| Weight on short feeds | Validation | Public |
|---|---|---|
| 0.2 (FD16P20) | 0.40271 | 0.394892 |
| 0.3 | 0.40266 | |
| 0.4 (FD16P20@m0.4_0.2) | 0.40293 | 0.3952092 |
| 0.5 | 0.40251 | |
| 0.6 | 0.40232 | |
| 0.8 | 0.40184 | |

0.4 is a single high point, read as a favourable draw of the four popularity-light students. With 8 or 12 of them under the same rule, validation read -0.00057 and -0.00039 against the file.

On long feeds the weight stayed at 0.2:

- raising it to 0.4 had scored 0.3947465 on public on 2026-11-15;
- lowering it to 0.1 or 0 read -0.00025 ± 0.00013 and -0.00068 ± 0.00020 against pick 1 on validation;
- A1 took the popularity-light part of the fine-tuned branch from the train-only predictions on long feeds. It read +0.00017 ± 0.00014 on validation and scored 0.395036 on public, 0.00017 below pick 1.

Both changes on long feeds that reached the board landed about 0.0004 below their validation reading.

## What it taught

Popularity was worth a lot on fresh data: -0.00743 on the holdout without it. Under test-like stale inputs it was worth little, -0.00089 ± 0.00091 on stale validation (validation re-predicted with those inputs).

For the popularity-light pools the holdout veto pointed the wrong way: -0.00002 to -0.00073 on the holdout, +0.000385 to +0.000634 on public. Two days later the cause was traced to the holdout, whose popularity inputs are fresher than those of test ([popularity-staleness](../popularity-staleness/)).

The popularity-light view returned as distilled students and is in both final files, as the same 4 all-data popularity-light students of width 192:

- FD16P20@m0.4_0.2, pick 1: weight 0.4 on feeds of fewer than 50 candidates and 0.2 on the rest, public 0.3952092;
- FDX16P20:ps, pick 2: weight 0.2 next to 16 pseudo-feed full-view students, public 0.39526061 ([distillation](../distillation/)).

From 18 to 19 November pick 2 was W8P20: 8 held-out full-view distilled students plus 4 narrow popularity-light students at weight 0.2, public 0.3950094.

The popularity-free view never helped on public, and its one favourable validation draw did not replicate.

FD16P20vf was a candidate for the second final until 2026-11-18, as an all-data file of equal expected level that differs from FD16P20 in 4 of its 24 members. W8P20 was chosen instead, and the view itself showed no gain. On 2026-11-19 FDX16P20:ps replaced W8P20 ([final picks](../../docs/11-final-picks.md)).

## Files

`code/`

- `abl_arms.py`: input flags of the screen.
- `abl_lane.sh`: GPU queue of the screen.
- `abl_eval.py`: paired read-out, rules in its docstring.
- `build_xp_imp_c50.py`: the exposure count.
- `build_xp_ctr.py`: the click-rate input.
- `pop1_actexp.py`: calibration by f5 and exposure cell.
- `launch_e10.sh`: launches one 10-epoch member.
- `decide9.py`: pool decision against G3.

`results/`

- `abl_eval_summary_cut.txt`: screen runs, popularity-free block, summary tables (cut).
- `abl_eval_pop1_cut.txt`: screen read of the popularity-light view.
- `abl_eval_pop2_cut.txt`: screen read of the click-rate view.
- `xp_imp_c50_parity.json`: exposure count per split.
- `pop1_actexp.txt`: calibration cells.
- `pop1_diversity.txt`: correlation of the screen pools.
- `popfree_readout.json`: 10-epoch popularity-free members.
- `pop1_readout.json`: 10-epoch popularity-light members.
- `popfree_decide9_pick.json`: Q1a decision.
- `pop1_decide9_pick.json`: Q2a decision.
- `pop1_8_members_result.txt`: the 8-seed pool and its veto.
- `pop1_8_members_decide.json`: its numbers.
- `pop2_decide.json`: click-rate pools.
- `gate_Q1a_popfree.txt`: pre-submission gate of Q1a.
- `gate_Q2a_pop1.txt`: pre-submission gate of Q2a.

### Raw material and records

- Lab folder: `lab/nn_x/abl/` (harness, `pop1/`, `pop2/`, `popfree/`, `ledger/ledger.md`, `logs/`).
- Reports: `lab/reports/ABLATION_PLAN_1113.md` (component ledger, screen design, decision tree) and `lab/reports/CODES_DECODING_1113.md` section 8, item D1.
- The files of the short-feed weight were built by `lab/nn_x/w12/kd/build_r4.py` and `lab/nn_x/w12/fixmiss/build_a1.py`.
- Records: the experiment registry and the decision ledger (working records, not published).
