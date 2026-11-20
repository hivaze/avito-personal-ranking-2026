# Hindsight (privileged-information) features on train feeds: measurement only

2026-11-14. Nothing here is a model input, nothing is submitted, no existing file was changed. The features use the same user's LATER
snapshot, which rule 3 forbids as a model input; they are measured only as a possible teacher-side signal.

Scripts and logs (all in `nn_x/w12/hindsight/`): `01_build.py` (pairs, alignment, features; `out/01_build.log`), `03_stats.py`
(`out/03_stats.log`), `02_eval.py` (`out/02_eval.log`, per-feed NDCG in `out/perfeed_HOLD.parquet`, `out/perfeed_INS.parquet`),
`04_prepost.py` (`out/04_prepost.log`). CPU only, 12 threads; no GPU used.

## Verdict

- **The hindsight signal beyond the past control is LARGE: +0.037 +- 0.0025 NDCG on the covered holdout feeds** (teacher out of sample,
  0.3194 -> 0.3551, LightGBM combiner; +0.025 +- 0.002 with a plain logistic combiner). In-sample sample: +0.032 +- 0.0005.
  The past control itself adds nothing (-0.001): the teacher already uses the history.
- **Without source-1 events the gain is moderate: +0.012 +- 0.0016** (holdout; +0.011 in-sample; logistic +0.005). About 2/3 of the
  signal sits in recommendation-surface events on items that are not candidates of A.
- **Caveat that decides how to use it: the signal is entirely POST-click.** 98.6% of holdout pair feeds have an interaction with one of
  A's candidates among the new events, and 89% of the non-candidate new events are newer than that first interaction. Events before
  the first candidate interaction add -0.0003 +- 0.0004; events after it add +0.039 +- 0.0025. So the features are most likely a
  causal consequence of the click (browsing items similar to the clicked one), i.e. a soft copy of the label rather than an independent
  view of the user's interest. The item-level exclusion is exact, but the label still enters through similarity. This was not
  separated further (it would need event timestamps, which the history does not have).
- **Coverage: 16.0% of train feeds (503,863) and 17.3% of train positives** have an aligned later feed with >= 1 new event; 14.7% of
  feeds have >= 1 non-candidate new event, 13.1% >= 1 non-candidate non-source-1 event. 16.05% of the 30K holdout (4,815 feeds).
  Val/test coverage of a later feed is 0.4%, so this can only ever act through training targets.
- Averaged over all train feeds the covered gain is about +0.037 x 0.147 = +0.005 of teacher NDCG on train, on feeds where the true
  label is available anyway.

## 1. Coverage and distributions

| Quantity | Value |
|---|---|
| Train feeds / positives | 3,151,208 / 5,446,872 |
| Feeds with a later same-user feed (user_id != 0) | 571,923 = 18.15% of feeds, 20.3% of positives |
| Aligned pairs (rule below, gap > 0) | 505,238 = 88.3% of pairs |
| Eligible = aligned and >= 1 raw new event | 503,863 = 15.99% of feeds, 939,940 positives = 17.26% |
| Eligible in the 30K holdout | 4,815 = 16.05% |
| Eligible by traffic type | 14,051 exploration (p_random > 0.9, M >= 50), 477,698 production (p_random <= 0.5) |
| Raw new events k (eligible) | mean 31.6; q05 2, q25 7, median 19, q75 43, q90 77, q99 169 |
| Non-candidate new events n_FUT (120K sample) | mean 28.8; q25 5, median 17, q75 40, q90 73; zero in 8.2% of feeds |
| Non-candidate, non-source-1 n_FUTNS1 | mean 15.7; q25 2, median 7, q75 21, q90 42; zero in 18.3% |
| Chain over up to 8 later feeds n_FUTC | mean 40.3, median 23; 1.32 hops on average |
| Share of new events removed by the candidate exclusion | 8.8% (holdout 6.8%) |
| Share of non-candidate new events with source 1 | 45.5% |
| New events on items already in A's history | 5.8% |
| Gap A -> B (eligible) | median 61 h; q25 22 h, q75 125 h, q95 241 h |
| Gap buckets | < 1 d 27.1% (median k 7), 1-3 d 28.0% (17), > 3 d 44.9% (35) |

## 2. NDCG: teacher, past control, hindsight

Teacher = `nn_x/w12/kd/out/teacher_mean11.npy`. Paired per-feed deltas, user-clustered SE. Combiner fitted on one user-grouped half and
read on the other (2-fold cross-fit; holdout: 3 repeats of the split averaged).

| Arm (LightGBM combiner) | Holdout pair feeds (4,815; teacher out of sample) | Non-holdout sample (120,000; teacher IN-SAMPLE) |
|---|---|---|
| Teacher alone | 0.31940 | 0.44095 |
| Teacher + context only (harness control) | 0.31922 (-0.00018 +- 0.00010) | 0.44065 (-0.00031 +- 0.00011) |
| Teacher + past control | 0.31786 (-0.00154 +- 0.00058) | 0.44039 (-0.00057 +- 0.00016) |
| Teacher + hindsight, no past | 0.34704 (+0.02764 +- 0.00226) | 0.46294 (+0.02198 +- 0.00043) |
| Teacher + past + hindsight (all non-candidate new events) | 0.35506 (+0.03566 +- 0.00254) | 0.47199 (+0.03104 +- 0.00050) |
| Teacher + past + hindsight excluding source 1 | 0.32951 (+0.01011 +- 0.00166) | 0.45132 (+0.01036 +- 0.00034) |
| Teacher + past + chain hindsight (up to 8 later feeds) | 0.35559 (+0.03619 +- 0.00253) | 0.47248 (+0.03153 +- 0.00051) |
| Sanity: teacher + past + new events WITHOUT the exclusion | 0.97429 (+0.65489 +- 0.00315) | 0.96270 (+0.52174 +- 0.00077) |

Future beyond the matched past control (the number that matters):

| Delta | Holdout GBM | Holdout logistic | In-sample GBM | In-sample logistic |
|---|---|---|---|---|
| Hindsight beyond past (PF - P) | +0.03721 +- 0.00248 | +0.02496 +- 0.00207 | +0.03160 +- 0.00050 | +0.02148 +- 0.00045 |
| Excluding source 1 (PN - PNc) | +0.01218 +- 0.00159 | +0.00531 +- 0.00133 | +0.01064 +- 0.00035 | +0.00572 +- 0.00029 |
| Chain (PCF - PC) | +0.03866 +- 0.00247 | +0.02538 +- 0.00213 | +0.03198 +- 0.00051 | +0.02169 +- 0.00045 |
| Past control - context | -0.00136 +- 0.00057 | -0.00087 +- 0.00045 | -0.00026 +- 0.00016 | -0.00011 +- 0.00009 |

Hindsight beyond past (PF - P), GBM, by slice:

| Slice | Holdout n | Holdout delta | In-sample n | In-sample delta |
|---|---|---|---|---|
| gap < 1 d | 1,366 | +0.0285 +- 0.0041 | 32,517 | +0.0267 +- 0.0009 |
| gap 1-3 d | 1,319 | +0.0448 +- 0.0052 | 33,703 | +0.0330 +- 0.0010 |
| gap > 3 d | 2,130 | +0.0381 +- 0.0038 | 53,780 | +0.0337 +- 0.0008 |
| n_FUT = 0 | 382 | +0.0001 +- 0.0001 | 9,890 | +0.0005 +- 0.0003 |
| n_FUT 1-5 | 786 | +0.0383 +- 0.0060 | 21,749 | +0.0295 +- 0.0012 |
| n_FUT 6-20 | 1,407 | +0.0387 +- 0.0050 | 35,329 | +0.0363 +- 0.0010 |
| n_FUT 21-50 | 1,224 | +0.0432 +- 0.0052 | 30,858 | +0.0362 +- 0.0011 |
| n_FUT > 50 | 1,016 | +0.0410 +- 0.0054 | 22,174 | +0.0338 +- 0.0012 |
| exploration (p_random > 0.9, M >= 50) | 4,815 | +0.0372 +- 0.0025 | 2,249 | +0.0366 +- 0.0037 |
| production (p_random <= 0.5) | - | - | 114,885 | +0.0314 +- 0.0005 |

Excluding source 1 (PN - PNc) by gap, GBM: holdout +0.0096 / +0.0156 / +0.0117 (SE 0.003); in-sample +0.0098 / +0.0108 / +0.0111.
The gain is already present with 1-5 new events and does not grow much with more, and it does not depend on the gap. Full slice
tables for both combiners: `out/02_eval.log`.

Standalone NDCG of single features on all 124,815 evaluation feeds (teacher 0.4363): max cosine to the non-candidate new events 0.407,
to the matched past 0.349, to the non-source-1 new events 0.358, to the matched non-source-1 past 0.328; category share 0.268 vs 0.261.

Pre/post diagnostic (holdout, GBM, `04_prepost.py`): past + events older than the first candidate interaction -0.00026 +- 0.00036
(only 12.4% of feeds have such events, 3.8 per feed on average); past + events newer than it +0.03883 +- 0.00248 (29.8 per feed).

## 3. Sanity of the exclusion

- Exact item match between a candidate and the event set: 0 rows of 10,454,034 for FUT, FUTNS1 and the chain set (asserted in
  `01_build.py`); 0 for the past sets as well.
- Without the exclusion the new events contain the candidate's own item for 98.07% of positives and 0.32% of negatives, and NDCG
  jumps to 0.974 (holdout) / 0.963 (in-sample); the max-cosine feature alone scores 0.944. The leak is visible and the exclusion removes it.

## 4. Definitions and assumptions

- Pair: A and B = the next feed of the same user after sorting by (user_id, timestamp, fidx); user_id 0 excluded; gap = 0 pairs
  (0.06%) dropped.
- Alignment: histories are not clean append-only logs (views expire, same-timestamp events reorder), so k = the modal shift q - t over
  A's 24 newest events t and every B position q with the same (item_id, event_type); ties go to the smaller shift. A pair is aligned
  when >= 50% of those events vote for k (88.3%; 63.7% unanimous), or A's history is empty (0.3% of aligned pairs; then all of B's
  history is new). New events = B's positions 0..k-1. Unaligned pairs (11.7%) are not covered. Recall check: 98.1% of A's positives
  are found among the new events.
- Past control: the newest K events of A's own history, K = number of events in the matching future set (non-source-1 control: the
  newest K non-source-1 events).
- Features per set (15): event counts (all, with embedding, strong = types 2/3/4, views = type 1), cosine max / mean / top-3 mean over
  all events, max / mean over strong and over view events (L2-normalised int8 128-d vectors, events or candidates without an embedding
  skipped), category count and share, category count over strong and over view events.
- Combiner inputs: teacher logit z-scored within the feed, centred teacher logit, M, H(A), log gap, k, plus the feature sets. LightGBM
  binary with init_score = logistic fit on the teacher (holdout: 200 rounds, lr 0.03, 15 leaves; in-sample: 300 rounds, lr 0.05,
  63 leaves; fixed in advance, no early stopping). Logistic combiner: 8 within-feed z-scored features per set.
- Holdout reproduced as in `nn_x/w5/holdout_eval.py` (30,000 feeds; teacher 0.31940 on the 4,815 pair feeds). Feeds of holdout users
  that are NOT in the holdout (including B) were in the teacher's training data.
- Evaluation feeds: every eligible holdout pair feed + 120,000 random eligible non-holdout pair feeds (seed 20261114). All have >= 1
  positive. Feeds with n_FUT = 0 are kept in the totals (8% of feeds, delta ~0).
- Not done: a wider feature set (sequence model over the new events), later feeds' user embeddings, tuning of the combiner. The numbers
  are a lower bound on what a model could extract.
