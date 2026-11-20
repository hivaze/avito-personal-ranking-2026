# Sweeps of small tricks

In brief:

- Two sweeps near the end, on 2026-11-13 and 2026-11-15, screened the remaining small tricks: post-processing, pooling forms, regularisers, auxiliary tasks and test-time augmentation.
- The first ranked 63 candidates, kept 12 and ran six closure probes. The second read 82 web sources and the internal records, then ran three proposals on GPU.
- Nothing was adopted. The second sweep found no untested lever of the model with an expected file gain above +0.0001.

## Why it was tried

By 2026-11-13 the best click-label pool, G3 (public 0.392820), no longer moved when members were added. By 2026-11-15 the distilled family stood at 0.3950094 on public, with files the board could not tell apart.

Ideas from papers and other competitions are cheap to screen on stored predictions, and a closed idea needs a written reason. Each sweep asked whether any trick had an expected gain worth a GPU run or a submission.

## What ran

### First sweep, 2026-11-13

The first sweep collected 63 candidates from six angles.

| Angle | Candidates |
|---|---|
| post-processing | 9 |
| labels | 10 |
| preprocessing | 13 |
| competition write-ups | 8 |
| network training tricks | 13 |
| metric facts | 10 |

A combined ranking kept 12 of them. Two independent sceptical reviews gave 24 verdicts on those 12.

Six closure probes followed. Each had its rule written to `lab/nn_x/tricks/<probe>/PREDECLARED.txt` minutes before the run (04:44 to 04:51 UTC). Five ran on CPU or on one GPU within minutes:

- the share of the remaining error a perfect re-order of the top 8 could recover (`recycle_step0.py`);
- an oracle for an auxiliary task on the user's future activity, with the label echo as a control (`futwin_stepA.py`);
- test-time averaging over passes with history events dropped, on two 10-epoch members (`hdrop_replay.py`);
- whether dropout still acts at the end of the cooldown, as a gate for consistency regularisers (`probe_kl.py`);
- a label-free format check of the early holdout days (script not copied; `lab/nn_x/tricks/hold_clean/`).

The label echo is a control score that marks the candidates in the same category as the feed's own positives, so it reads the label directly.

The sixth probe, late dropout, was the only trained arm. It used dropout 0 in the constant-rate phase and 0.1 in the cooldown: two 10-epoch seeds of 78 min each against two base seeds (`readout_late_dropout.py`).

### Second sweep, 2026-11-15

The second sweep had six scans:

- three of web sources, 82 sources in all, on ranking competitions, cheap ensemble growth and ranking under temporal shift;
- three of the internal records, on larger ensembles, the fine-tune stage and everything not soundly closed.

A synthesis followed, with four CPU checks (`anchored_forecast.py`, `pick2_decorr.py`, `pool_forms.py`, a blend of the two best files) and an adversarial critique. Two more CPU reads ran under rules written before them: an output offset for warm items whose counts the lag deflates (`lab/reports/TRICKS_1115/web_shift.md`), and the production-order vector of the domain embedding at inference on the 16 all-data students of FD16P20 (`lab/reports/TRICKS_1115/synth_checks/domtok/`). Three of the proposals ran on GPU on 2026-11-16:

- weight decay 0 on the full-view distilled student, 8 seeds;
- weight decay 0.01 on the same student, 8 seeds;
- a fine-tune with feed weights matched to the private test mix.

## Results

"±" is one paired standard error (SE) over feeds. The bar is the pass threshold written before the run. The holdout is 30,000 random-order train feeds kept out of training ([glossary](../../docs/glossary.md)).

| Probe | Read on | Reading | Bar | Verdict |
|---|---|---|---|---|
| Perfect re-order of G3's top 8: share of headroom | validation; holdout | 29%; 16% | 40% on both | closed |
| Oracle future-category share added to G3 at weight 0.1 | holdout, 4,476 feeds | +0.00275 ± 0.00068, 0.093 of the label echo | +0.005 | closed |
| Mean of 4 passes with 10% of history dropped | validation | -0.00013 ± 0.00036, correlation 0.9957 | | closed |
| Dropout KL between two passes, end of cooldown | train batches | 0.011 nats | close below 0.001 | left open |
| Late dropout, member | validation; holdout | +0.00006 ± 0.00071; +0.00006 ± 0.00041 | 5 clauses | met 3 of 5, closed |
| Softmax-probability pooling against z-mean, held-out students | holdout; validation | +0.00007 ± 0.00008; +0.00005 ± 0.00012 | | not adopted |
| z-blend of the two best files against the mean of its parts | validation | -0.00009 ± 0.00020 | | not built |
| Weight decay 0, 8-student pool | validation; holdout | -0.00134 ± 0.00056; -0.00011 ± 0.00031 | pooled z 2.2 | closed |
| Weight decay 0.01, 8-student pool | validation; holdout | -0.00032 ± 0.00051; -0.00021 ± 0.00028 | pooled z 2.2 | closed |
| Private-mix fine-tune, file | private-mix view; validation | +0.00022 ± 0.00035; -0.00007 ± 0.00022 | +1.5 SE | closed |
| One-parameter offset for warm items whose counts the lag deflates, nested, FD16P20 | validation | -0.00012 ± 0.00031 | | closed |
| Production-order vector of the domain embedding at inference, 16 all-data students of FD16P20 | validation | file +0.00015 ± 0.00024; their pool alone -0.00049 ± 0.00039 | private-mix view +1 SE | closed |

Headroom is the NDCG still missing from a perfect order. The z-mean is the standard pool: the mean of the members' within-feed z-scores. The private-mix view is validation re-weighted so that feed size and cold share match the private part of test. The weight-decay and private-mix fine-tune rows come from the decision ledger (a working record, not published).

The weight-decay bar was pooled z 2.2, above the standing 2, because two doses were read under one rule. Each dose also had to lower the mean member correlation on validation to 0.9714 or less, 2 SE below the reference's 0.9723 ([distillation](../distillation/#student-recipe-arms-14-to-16-november) has the other doses).

The warm-item offset aimed at a real miscalibration: the pools under-score warm rows whose counts the lag deflates (conditional-logit coefficient +0.20 ± 0.03, z 6.1, train-only branch), yet the nested offset was worth nothing.

The domain embedding adds a learned vector for the kind of feed, such as production order or random order, to the candidate tokens, and the stored predictions use the random-order vector. On 12 click-label members the production-order vector had read +0.0033 on feeds with a cold positive and -0.0010 on the rest. On the distilled students that sign reversed.

### The two regularisers the KL probe left open

R-Drop adds a KL penalty between two dropout passes. Sharpness-aware minimisation takes its gradient at weights nudged uphill.

Both were trained in the cooldown wave of the same morning, wave 7 in [training-waves](../training-waves/), 4 roots each. A root is a stored 10-epoch run whose cooldown is re-run.

| Arm | Validation | Holdout | Pool, holdout |
|---|---|---|---|
| R-Drop | -0.00012 ± 0.00032 | +0.00059 | +0.00036 (z 1.5) |
| Sharpness-aware minimisation | +0.00004 | +0.00041 | +0.00025 (z 1.1) |

Both stopped by their rule.

## What it taught

Nothing in either sweep was adopted.

The first sweep's reviewers rejected all 12 ranked items. Each was a near copy of the pool or traded cold-item feeds for warm ones, with a prior below the smallest effect the sweep report took a screen of that day to detect (0.0012 on validation, 0.0006 on the holdout, `lab/reports/TRICKS_SWEEP_1113.md`). For near-copy pools at 80% power [docs/04](../../docs/04-validation.md#smallest-detectable-effect) gives 0.00086 and 0.00060.

The second sweep found no untested lever of the team's own model with an expected file gain above +0.0001. The paired public SE between near-copy files is 0.0003 to 0.0006. The standing 8-seed rule passes a true +0.0005 half of the time and a true +0.0003 one time in five.

The one item above +0.0003 was a merge with another team's model of a different family. The final files contain no other team's model.

### Where the error sits

51% of validation feeds, holding 66% of the validation headroom, have no positive in the top 8. A re-ranker of the head cannot reach them.

Even perfect knowledge of the user's next days adds +0.00275 on the holdout, and most of it is an echo of this feed's own click. After removing events on the feed's own positives, +0.0007 (z 1.1) is left.

Dropping history at test time only removes information. One pass at 10% costs -0.00077 on validation, and averaging four passes only wins it back.

### Regularisers

No regulariser improved a pool. Late dropout left the members as strong and as alike as before (diversity margin +0.00073 against the +0.005 required). On the full-view distilled student, dropout 0 and 0.2 in place of 0.1 also lost on validation ([student recipe arms](../distillation/#student-recipe-arms-14-to-16-november)).

Weight decay 0 lowered the mean member correlation from 0.9723 to 0.9563, the intended effect. The members lost more than the pool regained.

### Method

A holdout subset chosen after seeing which arms it flips is not a judge. The early holdout days therefore became only a label-free information column.

A forecast needs a fair baseline. The anchored forecast reproduced the nine scored files with RMS error 0.00022 on public, and the synthesis compared that with 0.0006 for the level line (the straight-line fit of public to validation, [docs/04](../../docs/04-validation.md#noise)). The critique measured the level line at 0.00034 on the same nine files.

## Files

`code/`

- `recycle_step0.py`: headroom of a head re-ranker (`lab/nn_x/tricks/recycle_step0/step0.py`).
- `futwin_stepA.py`: the future-activity oracle and its controls.
- `hdrop_replay.py`: test-time history dropout on two members.
- `probe_kl.py`: dropout KL at the end of the cooldown.
- `readout_late_dropout.py`: the five-clause read of the late-dropout pair.
- `anchored_forecast.py`: public forecast of an unscored file from the scored ones.
- `pick2_decorr.py`: how far a candidate second final is from the first.
- `pool_forms.py`: z-mean, rank, softmax-probability and logit pooling of the distilled students.

`results/`

- `recycle_step0_RESULT.txt`, `futwin_RESULT.txt`, `hdrop_RESULT.txt`, `rdrop_kl_RESULT.txt`, `hold_clean_RESULT.txt`: the five probe results.
- `late_dropout_PREDECLARED.txt`, `late_dropout_RESULT.txt`: the trained arm's rule and read.
- `anchored_forecast.txt`, `pick2_decorr.txt`, `pool_forms.txt`: the synthesis checks.

### Raw material and records

- Probes, rules and read-outs: `lab/nn_x/tricks/`, one folder per probe. The raw logs are not in this repository.
- First sweep report: `lab/reports/TRICKS_SWEEP_1113.md` (section 2b for the probes).
- Second sweep: `lab/reports/TRICKS_1115/`, with `SYNTHESIS.md`, `CRITIC.md`, the six scans, `synth_checks/`, `critic_checks/` and `launch/`.
- Records: the experiment registry and the decision ledger (working records, not published).
