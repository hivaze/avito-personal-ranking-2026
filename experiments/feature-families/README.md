# Feature families

In brief:

- On 2026-11-11 six feature families (groups of pre-computed per-candidate columns that share a name prefix) were judged with a tree ranker on validation. Columns kept from sequence similarity with co-visitation (`seq_`) and from in-feed context (`ctx_`) became inputs of the neural ranker.
- A second round tried four more families, three of them rebuilding train rows to look as stale as test rows. All four were dropped. The one change adopted was a scaling fix for two encoder scores that the loader clipped.

Terms are defined in the [glossary](../../docs/glossary.md).

## Why it was tried

The organisers give five quantile-coded item fields (f1 to f5), two embeddings and up to 256 history events per feed. Co-occurrence, the rest of the feed and time had to be built.

The first question was which built columns add information to the organisers' fields. On the first day (2026-11-11) the families were built and judged with a tree ranker, and the columns kept then became inputs of the neural ranker.

The second question was whether train rows can be built to mean what they mean on test rows, whose popularity sources are stale ([staleness](../../docs/02-what-the-data-told-us.md#cold-items-and-stale-popularity)).

## What ran

Round 1 built six families:

- `seq_`: sequence similarity and co-visitation;
- `ctx_`: in-feed context and competition among candidates;
- `clu_`: embedding clusters;
- `cal_`: calendar;
- `trd_`: population trends;
- `life_`: listing lifecycle.

One harness judged all of them. `eval_family.py` fits LightGBM lambdarank (gradient-boosted trees trained to rank) on the validation days only, in 5 folds grouped by user. It fits once on a frozen base (118 or 122 columns, validation NDCG 0.3634) and once on base plus the family.

The harness reports the paired per-feed difference over 1 to 8 fold-shuffle seeds (3 in most runs). The log holds 120 runs, 17.0 hours of wall time at 16 CPU threads.

Round 2, the feature review of the same afternoon, attacked one mismatch. The popularity inputs of train rows come from sources 0 to 1 day old, those of test rows from sources 2 to 10 days old. Three new families rebuilt train rows to look as stale as test rows, and a fourth measured drift:

- lagged co-visitation;
- exposure context from other users' earlier candidate lists;
- age-gated popularity;
- how far the user vector had drifted since the user's earlier snapshot.

Each was screened by a gate, a LightGBM fitted on 50,000 random-order train feeds. The gate was scored on validation and on the 30,000-feed holdout of random-order train feeds.

## Results

Round 1, validation NDCG, paired per feed (`results/harness_runs.csv`):

| Family | Columns | Harness difference (validation) | Verdict |
|---|---|---|---|
| `seq_` | 26 | +0.00802 ± 0.00091 (3 seeds); +0.00693 ± 0.00079 (5-seed re-run) | kept |
| `ctx_` | 24 | +0.00321 ± 0.00080 and +0.00200 ± 0.00080 (same columns, two column orders); +0.00201 ± 0.00067 (5 seeds) | kept |
| `clu_` | 6 | +0.00125 ± 0.00045 (8 seeds); -0.00006 ± 0.00054 in another column order | kept, never a ranker input |
| `cal_` | 16 | -0.00023 ± 0.00067; two noise columns +0.00071 ± 0.00068 | dropped |
| `trd_` | 107 | -0.00024 ± 0.00063; four noise columns -0.00098 ± 0.00052 | dropped |
| `life_` | 16 / 27 | +0.00034 ± 0.00068 / -0.00173 ± 0.00071 | dropped |

`ctx_hs_t10` is the attention mass a candidate wins when the feed's candidates compete for each recent history item. Alone it read 0.3544 validation NDCG, against 0.349 for the maximum cosine to the last 32 history items.

The `life_` builder found that f4 is the exact listing age: a 64-knot map reproduces observed age gaps to a median 0.4 s. So no freshness column could add anything.

In the neural ranker, three groups of inputs were read in single paired runs (`lab/reports/nn_report.md`):

| Inputs added to the neural ranker | Validation | Holdout |
|---|---|---|
| co-visitation and leave-user-out popularity | +0.0032 | +0.0092 |
| competition among candidates | +0.0004 | +0.0009 |
| neighbour inputs | +0.0005 | +0.0017 |

The holdout overstates popularity inputs 2.5 to 5 times (`FEATURES_REVIEW.md` section 2.1). The final 34 inputs are 3 co-visitation, 5 popularity, 20 in-feed context and 6 encoder columns.

Round 2 (`results/second_round_verdicts.csv`), gate difference with its standard error in brackets:

| Family | Gate, validation | Gate, holdout |
|---|---|---|
| lagged co-visitation for train rows | +0.00032 (0.00071) | -0.00057 (0.00041) |
| item exposure context | +0.00012 (0.00072) | +0.00007 (0.00042) |
| age-gated popularity for train rows | +0.00111 (0.00075) | -0.0010 to -0.0015 in every run |
| user-vector drift | -0.00056 (0.00069) | +0.00034 (0.00042) |

All four were dropped. Counting a causal rebuild (-0.0018) and a 4-day lag (-0.0027) tried before, five causal or horizon-matched rebuilds failed to beat the columns as built.

The data start on Sep 6, so a lagged source leaves 43% of train feeds without one.

The age-gated counts restored parity and became a cold-for-warm trade. On validation they read +0.041 on the 1,487 feeds whose positives are all young listings and -0.0053 on the 11,771 feeds with no young positive.

The one change adopted was a scaling fix. The loader clipped raw columns to ±10 before ranking them within the feed. Two encoder scores have a median near 15, so between 75.9% and 92.5% of present rows, depending on column and split, sat tied at the clip.

With the rescaled copies of `pre2_rescale.py` the encoder inputs read +0.00191 ± 0.00058 on validation (z 3.3, 2 seeds, 3 epochs) and +0.0002 on the holdout (status log of 2026-11-11, 18:30 UTC).

## What it taught

Harness noise was as large as the effects. Column order moved the same 24 `ctx_` columns from +0.0032 to +0.0020 and the same 6 cluster columns from +0.00136 (5 seeds; +0.00125 at 8 seeds) to -0.00006, because LightGBM samples features by column index.

One fold seed had a low base that inflated 3-seed means by about 0.0008. From then on a family was read against a noise control and averaged over seeds (`eval_ens.py`).

Round 2 showed that a 14-day train window starting on Sep 6 cannot hold train rows as stale as test rows. That mismatch was left to the fine-tune on validation ([fine-tune](../fine-tune/)).

The cluster columns and 20 borderline trend columns were never tested inside the ranker.

## Files

`code/`

- `eval_family.py`, `eval_ens.py`: the harness, single-seed and seed-averaged.
- `feat_covis.py`, `build_ctx3.py`: builders of the co-visitation and candidate-competition columns.
- `hz_common.py`, `hzcv.py`: lag assignment and the lagged co-visitation of round 2.
- `hz_gate.py`: the round-2 screen (imports a scratch folder; does not run as copied).
- `pre2_rescale.py`: the clip fix.

`results/`

- `harness_runs.csv`: all 120 harness runs.
- `context_seed_ensemble_runs.csv`: seed-averaged runs of `ctx_`.
- `sequence_family_single_feature_ndcg.txt`: single-column NDCG of the 26 `seq_` columns.
- `second_round_verdicts.csv`: round 2.

### Raw material and records

- `lab/feats_v2/` (`seq_/`, `ctx_/`, `clu_/`, `cal_/`, `trd/`, `life_/`) and `lab/feat_new/`.
- The harness log `lab/reports/feature_evals.jsonl`.
- Round 1 reports under `lab/reports/`: `feat_seq_.md`, `feat_ctx_.md`, `feat_clu_.md`, `feat_cal_.md`, `feat_trd_.md`, `feat_life_.md`.
- Round 2 reports under `lab/reports/`: `featrev_prep.md`, `featrev_inventory.md`, `featrev_research.md`, `featnew_hzcv.md`, `featnew_ixh.md`, `featnew_it_hist_hz.md`, `featnew_ud.md`, `FEATURES_REVIEW.md`.
- Records: the status log (a working record, not published).

In context: [the solution, inputs](../../docs/03-solution.md), [what failed](../../docs/07-what-failed.md).
