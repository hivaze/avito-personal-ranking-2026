# INVENTORY D: raw fields and relations that no trained input reads

Written 2026-11-16 (UTC) by the raw-fields agent. CPU only; nothing submitted; nothing was written except this file and the directory `reports/INPUTS_1116/raw_fields/`. Markers: [M] measured here (script and output file named; all under `reports/INPUTS_1116/raw_fields/`), [R] record on disk (path), [I] judgement. `reports/COVERAGE_MAP.md` was opened only after the 30 pre-declared reads of section 3 were on disk (the post hoc block of section 3.3 was running at that moment and its design was already fixed in `d05_followup.py`).

## 0. Answer

1. **The raw data hold no unread column.** The competition supplies 19 columns and three embedding tables (section 1). The identifiers carry no hidden order: `feed_id`, `user_id`, `item_id`, the embedding row index, the parquet part and row order and the candidate order of val are all uncorrelated with time, feed size, listing age and labels (|Spearman| at most 0.03) [M `d02_relations.txt`, `d02b_fileorder.txt`]. There is no external table.
2. **Three things are read coarsely or not at all** (section 2): (a) `user_id` as a link between a user's feeds: no input reads what the same user was shown and did in EARLIER feeds (30% of val and test feeds have one); (b) the mass points of the quantile-encoded f columns (f1 at its floor, f2 equal to the feed's modal value): inputs carry value, z and percentile, never an equality flag; (c) `target` and `pos` of train feeds as statistics of the item, and the event source as a statistic of the item.
3. **The likelihood-ratio lens flags two families that replicate on val and on the holdout, and two that are popularity-type and die at test-like staleness** (section 3). By the rule written first, 4 of 12 forms are "interesting"; none of them moves NDCG in residual form (the cross-fitted residual deltas of the flagged forms are inside ±0.0003 and within 1.3 SE of zero), as expected from a lens with 10-25% sensitivity.
4. **Size.** A positive control of the same lens (inputs the model is known to need) reads LR 28-67 per column and 124 for the 8-column popularity block; the two new families together read 33 (four pre-declared forms) to 50 (six best forms) on val and 35 to 44 on the holdout. My judgement of their worth as trained inputs is +0.0001 to +0.0010 on val, centre +0.0003 [I]. **Nothing found here is of the size of the 0.0032 gap to #1.**
5. **One statement of the brief does not hold as written**: the histories of val and test feeds do not end at 2026-09-20. In val feeds of Sep 22 the newest history item was posted a median 3.6 days and, in 5% of feeds, less than 0.50 days before the feed, and the newest posting date reaches the feed day itself; test is the same [M `d07_histfresh.txt`]. What ends at the dense cutoff is the stock of OTHER feeds from which cross-feed counters are built.

Ranked list (detail in section 5):

| # | Raw signal | Raw source | Lens (LR; rule 10.83) val-TO / val-FIN / holdout | Stale on test but fresh on train? | Leak | Build cost | Verdict [I] |
|---|---|---|---|---|---|---|---|
| 1 | Same user's earlier feeds: user × category response (clicks against position-expected clicks), exposure, click similarity, re-shown flag | `user_id` × `target`, `pos`, `item_id` of the user's earlier train feeds | pre-declared `uc_ctr` 16.3 / 16.8 / 4.9; post hoc `uc_coec` 23.3 / 23.8 / 6.1; block of 3: 32.0 / 32.4 / 14.3 (3 dof) | Yes unless lagged: source feed is 0.9-9.1 d old on train rows, 4-16 d on val, 7-21 d on test. At a 7.5-9.5 d lag on val the LR is still 14.5 | Clean if causal: earlier feeds only, user 0 excluded, no own-feed label | CPU about 1 h (unlagged files exist: `tte_uc_*`); as fine-tune extras 5-7 GPU-h; as a train-stage input 27 GPU-h for 8 students | BUILD (first as fine-tune extras) |
| 2 | Mass points and ties of f1 / f2 inside a feed: f1 at floor, f2 = modal f2 of the feed, tie-group size | `f1`, `f2` of the row and of its feed | `f2_home` 5.0 / 5.8 / 12.9; `f1_tie` 5.7 / 5.3 / 9.9; post hoc `f1_floor` 14.2 / 13.0 / 19.2; block of 3: 18.7 / 18.2 / 30.5 (3 dof) | No: row's own values, built identically everywhere | None | 0 CPU (3-6 columns inside `dense_features`); train-stage arm 27 GPU-h, or free inside arm 1 | BUILD (bundle with 1) |
| 3 | Surface mix of the item across users: share of main-feed opens among its views | `history__event_source` × `history__item_id` of other users | 13.0 / 11.7 / not admissible; at test-like staleness 3.0 / 3.1 | Yes; the signal is gone at the test lag | Clean with leave-user-out | CPU 15 min | Do not build as a train-stage input; at most a lagged fine-tune extra |
| 4 | Train `target` as an item statistic (clicks over expected clicks) | `target`, `pos`, `item_id` of train feeds | 18.7 / 17.6 / not admissible; at test-like staleness 9.0 / 8.4 | Yes; halves at the test lag; 44% (public) and 47% (private) of test rows have no train impression, against 35% on val | Clean if causal | Files exist (`tte_it_*`) | Low: lagged form only |
| 5 | History events without an embedding (14-16% of events; 16-33% of the events of types 2 to 9 against 5.8% of views): content recovered from the train table by `item_id` | `history__item_id__embedding_idx` = -1, train `item_embeddings` | 3.2 / 3.5 / 1.2 (null) | Reverse defect: recoverable for 21.6% (val) / 26.3% (test) of such events, 0.14% on train rows | None | CPU 2 min; usable only at fine-tune or inference time | Low; a zero-training replay is the only sensible test |
| - | Feed hour × category or × listing age | `timestamp` | 0.0 / 0.0 / 2.8 and 3.1 / 2.2 / 0.2 | No | None | - | NOT worth building (and already a trained-input screen, map l.276) |
| - | Production-placement prior of the item | `pos`, `item_id` | 0.0 / 0.2 | Yes | None | - | NOT worth building |
| - | Fresh co-exposure of the item inside val | `item_id` across val feeds | 6.0 / 3.3 (placebo 3.1) | No (density parity needed) | None | - | NOT worth building (agrees with map l.345) |

## 1. Raw tables and columns

Source: `data/competition/readme.md` (organisers' dictionary) and the flat arrays `data/flat/<split>/`, a bit-exact flattening of the parquet by `src/prep.py`. All numbers [M `d01_inventory.txt`] unless marked.

Tables: `impressions` (one row per feed; 512 / 64 / 64 parquet parts for train / val / test) and one int8 item-embedding table per split (81,557,944 / 2,516,711 / 4,710,948 rows × 128). Every table row is referenced by a candidate or a history event (unreferenced share 0.00000 in each split); item ↔ row is a bijection inside a split.

| Column | Type | Level | Cardinality and range | Missing / special values | Time coverage, relation |
|---|---|---|---|---|---|
| `feed_id` | int32 | feed | 3,151,208 + 14,423 + 28,595 unique values in 1..3,194,226 | none | key; corr with timestamp -0.0012 / -0.0002 / +0.0085 |
| `timestamp` | int32 | feed | 128-s grid in all splits; 7,613 / 1,240 / 2,524 distinct slots | none | train 09-05 21:00 to 09-19 21:00 UTC; val 09-21 21:00 to 09-24 21:00; test 09-25 21:00 to 09-29 20:58 |
| `user_id` | int32 | feed | 2,455,382 / 13,741 / 27,309 distinct; max 2,483,761 | 0 = pooled bucket: 3.93% / 4.56% / 4.12% of feeds | 421,339 train users have more than one feed (993,262 feeds); 25 val and 105 test users do |
| `category_id` | list int8 | candidate | 52 codes (2..53) | 0 never | shared dictionary with history |
| `f1`..`f5` | list float32 | candidate | quantile codes in about [0.016, 0.984]; distinct values on val: 1.01M / 215K / 132K / 333K / 1.13M | no NaN; f2 = 0 in 11.9% / 15.4% / 15.0% of rows; f5 = 0 in 1.4% / 1.3% / 1.1%; f1 at its floor 15.4% / 17.6% / 17.3-17.5% [M `d08_coverage.txt`] | f4 = listing age, f3 constant per (feed, category) [R docs/01 §5.3] |
| `pos` | list int16 | candidate, train only | 0..299 | absent in val / test | production order |
| `target` | list int8 | candidate, train and val | positive 2.21% / 1.58% | absent in test | label |
| `item_id__embedding_idx` | list int32 | candidate | row of the split's table | -1 in 0.003% / 0.015% / 0.016% | split-local |
| `item_id` | list int32 | candidate | 38,950,205 / 1,239,391 / 2,381,673 distinct; 42.5% / 91.0% / 85.8% of items occur in one feed of the split | 0 never; no item twice in a feed [M `d02_relations.txt` D] | shared across splits and history |
| `user_embedding` | list int8 ×128 | feed | - | none | - |
| `history__item_id` | list int32 | event, H ≤ 256 | 94,705,754 distinct in train | 0 never | newest first; no event timestamps |
| `history__category_id` | list int8 | event | 53-54 codes | 0 in 0.003% | - |
| `history__event_type` | list int8 | event | 1..10; share of type 1: 51.9% / 43.9% / 44.1% | none | - |
| `history__event_source` | list int8 | event | 0..30 (val to 26, test to 29); source 0: 39.3% / 45.4% / 45.2%; source 1: 37.6% / 33.8% / 34.3% | 0 = unknown | - |
| `history__item_id__embedding_idx` | list int32 | event | row of the split's table | **-1 in 15.5% / 14.2% / 14.0% of events**; by type on val: e1 5.8%, e4 24.2%, e7 25.3%, e8 25.7%, e9 32.9% [M `d02_relations.txt` F] | split-local |

Feed sizes: M 5..300 (mean 78.3 / 96.4 / 101.6); H 0..256 (mean 185.2 / 173.3 / 173.9; H = 256 in 42.3% / 29.0% / 28.8%).

Hidden order in the identifiers and in the files, all null [M `d02_relations.txt` A-C, H; `d02b_fileorder.txt`]:

| Relation | Read |
|---|---|
| `item_id` against posting time (feed time minus age decoded from f4) | Spearman -0.0004 (val), +0.0008 (test); median age by `item_id` decile flat (670-688 h) |
| embedding row index against `item_id`, first use, posting time | -0.002, -0.0005, +0.003 (val) |
| `user_id` against H, M, time, first train feed | +0.012, -0.005, +0.009, -0.0002 |
| `feed_id` against M, user, feed positive rate | +0.002, +0.009, -0.004 |
| candidate order in val | mean relative index of positives 0.5005 |
| parquet part index and row rank inside a part (val) | against time, user, feed_id |ρ| ≤ 0.013; row rank against M +0.025 (feed-level only) |

Freshness of the row's own history [M `d07_histfresh.txt`; the dates in that file carry a correction note]. Lower bound of the newest event = newest posting time among the history items whose posting time is decodable (44% of history items on train rows, 52% on val and test rows):

| Split | Days from that posting time to the feed: q05 / median | Share under 1 day | Type-1 share in the newest 32 events / in the whole history |
|---|---|---|---|
| train (200,000 feeds) | 0.19 / 2.51 | 0.253 | 0.567 / 0.515 |
| val | 0.50 / 3.94 | 0.129 | 0.582 / 0.460 |
| test | 0.57 / 4.88 | 0.110 | 0.584 / 0.463 |

For val feeds of Sep 22 the newest posting date among history items has q95 = Sep 21.98 and maximum Sep 22.95; for test feeds of Sep 26 to 29 the maxima are Sep 26.91 to 29.88. So the histories run up to the feed. The larger gap on val and test is at least partly a measurement artefact: posting times of listings created after Sep 20 are known only from the sparse val and test feeds [I]. The newest 32 events have the same type mix in all splits; the whole-history mix differs (fewer old views and shorter histories on val and test), which I did not explain.

## 2. Which raw column does a trained input derive from

Trained inputs = the tokens and the 63 in-model columns of `nn_x/w5/model.py` (`dense_features`, read here), the 34 pre-computed train-stage inputs `cvnd_pre2s`, the pop1 view's `xp_imp_c50` and the 42 fine-tune extras [R docs/02_SOLUTION.md §4.1-4.6]. 273 feature files exist under `data/feats/val/`, but only these are read by a model; a file on disk is not a trained input.

| Raw column or relation | Read by trained inputs as | Not read, or read coarsely |
|---|---|---|
| `feed_id` | key | nothing to read (section 1) |
| `timestamp` | only to order sources: causal windows of co-visitation and of the trajectory encoder, later-snapshot removal | hour, weekday, absolute time are no input (a 2-epoch trained screen was closed, map l.276); posting time = timestamp minus age(f4) exists only as GBDT-screened features (`feats_v2/life_/`) |
| `user_id` | user-0 flag (2 columns), leave-user-later-out removal, fold grouping | **never links a user's feeds**: the candidates, labels and positions of the same user's earlier feeds are unread (check V2) |
| `category_id` | embedding; counts by event type; recency; share in the feed | fully read |
| `f1`..`f5` | value, within-feed z, percentile (15 columns) | **mass points and ties**: no flag for "f1 at floor", "f2 = modal value of the feed", "same value as k other candidates" (check V8) |
| `pos` (train) | auxiliary target (production order), exploration detector | as an item statistic (check V4, null); as the exposure denominator of user or item response |
| `target` | label, teacher, fine-tune | as an item statistic (check V3) and as a user × category statistic (check V2b) |
| candidate `emb_idx` | lookup, missing flag, norm | cross-split differences (drift: closed in residual form, map l.198) |
| candidate `item_id` | co-visitation with the history (3 columns), counts over all history snapshots by event type 1-4 (5 columns; types 5-10 only in the fine-tune), ID table inside the trajectory encoder (scores only), train impressions (pop1 view) | no ID embedding in the ranker (student arm closed, map row "kw_id"); label statistics; source mix; co-exposure inside the split (check V7) |
| `user_embedding` | user token, cosine and dot with the candidate, product in the candidate token | fully read |
| history `item_id`, category, event type, order | tokens, per-type similarities and counts, recency embedding, retrieval | fully read |
| `history__event_source` | per-event embedding in the history token and in the retrieval value; target definition of the encoder | no dense column by source (types have 20); **no source statistic of the candidate item across users** (check V5) |
| history `emb_idx` = -1 | token kept with a learned "missing" vector; excluded from every similarity | **content of 14-16% of events is absent**; 21.6% (val) / 26.3% (test) of them have a vector in the train table, 0.14% of the holdout's in the val or test table (check V6) |
| candidates of one feed | set attention, `ctx_nd` / `ctx_hn` / `ctx_hs`, category share | equality structure of f values (check V8) |
| a user's feeds | later-snapshot removal only | earlier feeds (check V2) |
| the same item in other users' feeds | counts from histories | labels (V3), positions (V4), co-exposure in the split (V7) |

## 3. Cheap checks: conditional-logit likelihood ratio with a within-feed-shuffle placebo

Rule and list were written at 2026-11-16T01:53:50Z, before the first run: `d03_rule.txt`. Lens: `d03_lib.py`. Utility u = C·a + β·x with C = [s, s², s³, s·log(M/50)], s = within-feed z of the pool score; LR = 2·(ll[C, x] − ll[C]), 1 dof; placebo = x shuffled within feed. **Interesting iff LR ≥ 10.83 and placebo LR < 3.84.** Controls: val-TO = `preds/ens_FD16P20kd_to_val.npy` (val NDCG 0.40213); val-FIN = 0.5 z(fine-tuned out-of-fold) + 0.5 z(train-only) of FD16P20 (0.40271), to see whether the fine-tune stage already reads the signal; holdout = 0.8 z(16 held-out wide students) + 0.2 z(8 held-out pop1 students) on the 30,000 holdout feeds (0.32405), used only where the signal is not popularity-type [M `d04a_results.txt`].

### 3.1 The 30 pre-declared reads (12 forms, 8 signals)

[M `d04a_results.txt`, `d04b_results.txt`]. Entries: LR (placebo LR). β in the order val-TO / val-FIN / holdout.

| Check | Form x | val-TO | val-FIN | holdout | β | Verdict by the rule |
|---|---|---|---|---|---|---|
| V1a | hour-bin × category lift | 0.00 (0.82) | 0.00 (0.82) | 2.82 (0.76) | +0.00 / +0.02 / +0.45 | null |
| V1b | hour-bin × age-decile lift | 3.07 (0.14) | 2.22 (0.14) | 0.23 (1.22) | -0.91 / -0.77 / +0.16 | null |
| V2a | same (user, item) shown in an earlier train feed (0.18% of rows) | 6.30 (3.16) | 6.64 (3.17) | 7.43 (0.06) | -0.38 / -0.39 / -0.27 | null; same sign three times |
| **V2b** | log(user × category train CTR / category CTR) | **16.34 (0.33)** | **16.75 (0.33)** | 4.94 (0.04) | +0.092 / +0.094 / +0.044 | **interesting on val**; holdout same sign |
| **V3** | item clicks over expected clicks in train | **18.72 (0.84)** | **17.61 (0.83)** | not admissible | +0.099 / +0.097 | **interesting** (popularity-type) |
| V4 | item placement prior from `pos` | 0.00 (0.17) | 0.21 (0.20) | not admissible | -0.00 / -0.02 | null |
| **V5a** | share of source 1 among the item's views, other users | **13.03 (0.01)** | **11.68 (0.01)** | not admissible | +0.030 / +0.028 | **interesting** (popularity-type) |
| V5b | log1p count of such views | 1.01 (0.26) | 0.77 (0.28) | not admissible | +0.005 / +0.004 | null |
| V6 | similarity gain from recovered missing-embedding events (3.05% of val rows, 0.03% of holdout rows) | 3.23 (0.13) | 3.47 (0.12) | 1.18 (0.10) | +1.27 / +1.31 / -7.86 | null |
| V7 | log number of val feeds holding the item (18.9% of rows in ≥ 2) | 6.01 (3.09) | 3.29 (3.11) | not admissible | +0.047 / +0.034 | null |
| **V8a** | f2 equals the feed's modal non-zero f2 | 5.01 (0.10) | 5.79 (0.10) | **12.92 (0.47)** | -0.036 / -0.038 / -0.037 | **interesting on the holdout only**; same β on val |
| V8b | log size of the candidate's f1 tie group | 5.71 (0.02) | 5.26 (0.03) | 9.93 (0.82) | -0.020 / -0.019 / -0.015 | null; same sign three times |

Cross-fitted residual NDCG deltas of these single forms (5 user-grouped folds) lie between -0.00030 and +0.00025 with SE 0.00001 to 0.00033; none is 2 SE above zero [M same files]. Implementation note on V2b: the column is log((clicks + 20·c) / (impressions + 20) / c) with c the category CTR, rebuilt from `tte_uc_ctr`, `tte_uc_imp` and `tte_c_ctr`, so that an unknown pair reads exactly 0; the file's own smoothing prior is 0.0159. This was fixed in `d04a_signals.py` before its first run.

Multiplicity: 12 forms and 6 holdout reads are 18 roughly independent looks (the two val controls are near copies of each other). The best of 18 pure-noise chi-square draws has median 4.3 and exceeds 10.83 with probability 1.8%; four forms pass.

### 3.2 Calibration of the lens (post hoc, not candidates)

| Read | Value | Source |
|---|---|---|
| Placebo over 160 shuffles (8 columns × 20) | means 0.56-1.39 per column, 3 of 160 above 3.84 (8 expected): the placebo behaves as chi-square or slightly below | [M `d05_followup.txt` A] |
| Richer score control (11 columns: cubic, hinges, percentile, interactions with M) | flagged forms unchanged: `uc_ctr` 16.0, `uc_coec` 23.0, `f1_floor` 13.3 / 17.7 (val / holdout), `f2_home` 4.6 / 12.2, `it_coec` 18.7, `it_s1share` 13.0 | [M `d10_robust.txt`] |
| **Positive control**: one removed popularity input against the pool of the 4 all-data popularity-free students | `it_hist_all_luo` 39.5, `it_hist_et1_luo` 43.3, `seq_cv_invd20` 67.4, `seq_cv_pop` 28.4; the 8 inputs jointly 124.2 (8 dof) | [M `d06_poscontrol.txt`] |
| The same columns against a pool that reads them (4 full-view students) | 2.8, 2.4, 3.5, 2.9; jointly 14.1 (8 dof). Against FD16P20-TO: 0.0, 0.0, 8.5, 0.2 | [M `d06_poscontrol.txt`] |
| Worth of that block as trained inputs on val | +0.00043 ± 0.00095 between the two 4-student distilled pools (the teacher of both saw the inputs); +0.00251 between hard-label pools | [M `d06_poscontrol.txt`]; [R docs/01_DATA_AND_FINDINGS.md §8.3] |

Reading: an input the pool already reads shows LR 0 to 8.5; an input it lacks shows 28 to 67; the four hits sit at 12 to 19, between the two.

### 3.3 Post hoc reads (27 further reads; they describe the hits and cannot promote anything by the rule)

[M `d05_followup.txt`]. With the 30 above, 57 candidate reads in total; the best of 57 noise draws has median 6.3 and exceeds 10.83 with probability 5.5%.

| Block | Form | val-TO | val-FIN | holdout | Reading |
|---|---|---|---|---|---|
| B: hits rebuilt from train feeds of MSK date ≤ Sep 14 (lag to val 7.5-9.5 d, like test rows) | `it_coec` stale | 9.01 | 8.35 | - | halves (18.7 → 9.0), below the rule |
| | `uc_ctr` stale | 14.51 | 14.57 | - | **keeps 89%** (16.3 → 14.5); coverage 26.4% → 20.9% of rows |
| | `it_s1share` stale | 2.95 | 3.10 | - | gone (13.0 → 3.0) |
| C: same user's earlier feeds, richer forms | `skip_sim`: max cosine to items shown earlier and not clicked | 7.21 | 4.11 | 2.62 | sign flips on the holdout: nothing |
| | `click_sim`: max cosine to items clicked in earlier feeds | 9.05 | 8.27 | 3.41 | same sign three times, below the rule |
| | `uc_coec`: user × category clicks over position-expected clicks | **23.33** | **23.82** | 6.05 | strongest form; holdout same sign, β half (+0.38 / +0.39 / +0.19) |
| D: more equality forms | f2 = 0 | 4.29 | 4.81 | 1.79 | null |
| | log size of the f2 tie group | 1.96 | 2.56 | 4.01 | null |
| | log size of the f5 tie group | 4.14 | 1.11 | 4.68 | sign flips: nothing |
| | **f1 at its floor** | **14.15** | **12.97** | **19.18** | passes on all three; β -0.110 / -0.106 / -0.084 |

Joint reads of the two replicating families [M `d09_bundle.txt`]:

| Bundle | val-TO | val-FIN | holdout | Cross-fitted residual NDCG delta (val-TO / val-FIN / holdout) |
|---|---|---|---|---|
| P: the four pre-declared forms of V2 and V8 (4 dof; chi-square p = 0.001 at 18.5) | 33.47 | 34.47 | 34.70 | +0.00005 ± 0.00027 / 0.00000 ± 0.00027 / +0.00002 ± 0.00012 |
| B: six best forms (6 dof; columns chosen after their LRs were seen) | 49.67 | 49.61 | 44.47 | -0.00021 ± 0.00029 / +0.00019 ± 0.00028 / +0.00011 ± 0.00011 |
| U: earlier-feed block (`uc_coec`, `click_sim`, re-shown; 3 dof) | 31.96 | 32.39 | 14.31 | -0.00004 ± 0.00020 / +0.00019 ± 0.00020 / +0.00004 ± 0.00005 |
| E: equality block (`f1_floor`, `f2_home`, `f1_tie`; 3 dof) | 18.66 | 18.17 | 30.46 | -0.00031 ± 0.00023 / -0.00013 ± 0.00022 / +0.00010 ± 0.00011 |

Bundle P has p below 1e-6 on val and, independently, on the holdout, with the same coefficient signs. Reweighted to the private feed mix, bundle B on val-FIN reads +0.00033 ± 0.00043 and on the feeds of linked users +0.00070 ± 0.00069: no NDCG evidence in residual form. The signal exists; its size in NDCG is not shown by this lens.

## 4. Comparison with `reports/COVERAGE_MAP.md` (read after section 3.1)

| Signal | Row found | How it was closed | Ever a trained input? | Agreement with this scan |
|---|---|---|---|---|
| V1 timestamp | l.276 "feed weekday + hour-of-day as ranker inputs (TIME arm)" | trained 2-epoch hard-label screen, failed on the holdout weekend slice (+0.0002 ± 0.0005) | yes, 2 epochs | agrees (null) |
| Hidden order of ids | l.278 "item ids as a posting-time clock" | lead measurement, Spearman 0.000 | - | agrees |
| V2 same user's earlier feeds | l.226 "embedding-level negative feedback from the user's earlier impressions" (nested -0.0006..+0.0001); l.267 "C10 user aggregates" (residual, val -0.00155); `reports/featrev_inventory.md` l.625 `tte_uc_ctr` ("GB+gate"); l.227 lifelong history (no arm, lack of power) | nested residual gates and a GBDT screen | **never** | the residual closures are reproduced (residual NDCG 0); the lens shows the likelihood signal the gates could not see |
| V3 item clicks over expected | l.234 "additive item CTR/COEC as a train-stage input (F17)": RE-OPEN (low), P ≤ 0.02, "AUC 0.5225 → 0.501 from 1- to 8-day gap"; l.263 `it_tr_coec` alone +0.00060 ± 0.00033 | residual | never | agrees: present on fresh val, halves at the test lag |
| V4 item placement prior | l.242 "ixh as NN input": RE-OPEN, P 0.02 (GBDT gate + one AdamW seed +0.0010 / +0.0002); l.78 closed by review verdicts | GBDT and one seed of an old recipe | one seed | this lens reads nothing (0.0 / 0.2) |
| V5 source mix of the item | l.204 "decoding-enabled constructions (34 cues: surfaces, ...)"; l.274 "same-category counts by source" | nested residual on the G3 pool | never | the family was covered; I did not find this exact form in the records; it fails at the test lag anyway (LR 3.0) |
| V6 missing-embedding events | no row. `reports/eda_embeddings.md` §1.2 has the rates and says "mask them" | - | never | the cross-split recoverability (21.6% / 26.3%) is new; the lens is null |
| V7 fresh co-exposure | l.345 "fresh exposure counts from the val + test feeds": closed at the CPU pre-gate (nested -0.00004 ± 0.00020; click lift given the pool 1.008 ± 0.021) | residual | never | agrees (null) |
| V8 f mass points and ties | l.196 "home-region deficit real but calibrated (0.981)"; l.201; `reports/CODES_EXPLOIT_1113.md` l.31 ("null ... G3 calibrated"); `reports/CODES_DECODING_1113.md` l.213 documents "f1 floor = no fine affinity"; no row for a floor or equality flag | residual on the G3 pool | never | differs in degree: against the current pool the home-region and floor terms are not fully calibrated (β -0.037 and -0.11 to -0.08, same on val and holdout) |

None of the four hits was ever a trained input of a student, in line with `reports/TRICKS_1115/int_weak.md` row T16.

## 5. Ranked list with leak, parity and cost

Coverage and lag numbers: [M `d08_coverage.txt`, `d08b_lagcoverage.txt`].

**1. Same user's earlier feeds (block U).** Columns: user × category clicks over position-expected clicks, the matching exposure, maximum cosine to items clicked in earlier feeds, re-shown flag.
- Leak: admissible. It uses only feeds of the same user that are strictly earlier, never a later snapshot and never the row's own feed; user 0 gets zeros. For train rows the build must be causal (earlier feeds only); the val form here reads train feeds only, so no val label enters it.
- Parity: this is the weak point. Rows with a user × category record: val 26.4%, public 25.8%, private 24.2%, but 15.0% of train rows (causal, earlier day). Age of the user's last source feed, q10 / median / q90 in days: train rows 0.9 / 3.5 / 9.1, val 4.0 / 10.0 / 15.8, public 7.4 / 13.2 / 19.0, private 9.5 / 15.5 / 21.0. So on train rows it is fresher and rarer than on test rows. A lag of at least 7 days on train rows leaves 4.6% of train feeds with a source (12.0% of the logged-in feeds of Sep 13-19). The signal itself tolerates the lag (LR 14.5 at 7.5-9.5 days against 16.3).
- Consequence [I]: the natural first home is the fine-tune stage, whose 42 extras are val/test-only by design: no train-row parity is needed there, val has the test-like coverage, and the val lag (4-16 d) is only moderately shorter than the test lag (7-21 d).
- Cost: val/test columns in minutes (`src/features_tte.py` wrote `tte_uc_*`; the position-debiased and similarity forms are in `d05_followup.py`, 77 s). Fine-tune of the 20 FD16P20 members with added extras: 20 × 15-22 min = 5-7 GPU-h [R docs/02_SOLUTION.md §12]. Train-stage version: lagged builder with parity checks about 1 CPU-hour of compute, then 8 students, 27 GPU-h [R `reports/TRICKS_1115/int_weak.md` T16].
- Caution: on the holdout the coefficient is half the val one although its sources are fresher; I have no explanation.

**2. Mass points and ties of f1 and f2 (block E).** Columns: f1 at floor, f2 equal to the feed's modal non-zero f2, log size of the f1 tie group; the other reserved codes (f3 floor, f5 = 1/62, f5 = 0, f2 = 0) can ride along although f2 = 0 and the f5 ties read null here.
- Leak: none. Parity: exact, the row's own values and its feed; share of rows with f1 at floor 17.6% val, 17.3% public, 17.5% private, 15.4% train, 18.6% holdout. No staleness: the f columns are computed at feed time.
- Cost: no CPU build; 3 to 6 columns inside `dense_features` of a code copy (n_dense 104 → 107-110). As a train-stage input it needs students (27 GPU-h for 8); as pre-computed fine-tune extras it is free inside arm 1.
- Why the model misses it [I]: the raw value passes through a standardised MLP that cannot represent a step at one exact value; the fine-tuned branch does not fix it (val-FIN reads the same LR).

**3. Surface mix of the item across users** and **4. item clicks over expected clicks.** Both are popularity-type, both are stale on test in a way they are not on train rows, and both lose most or half of their LR when rebuilt at the test lag (13.0 → 3.0; 18.7 → 9.0). They are admissible only in lagged, leave-user-out, causal form; then little is left. They would enter a bundle only as fine-tune extras and I would not spend student seeds on them.

**5. Content of history events without an embedding.** Not buildable as a train-stage input (0.14% recoverable on train rows against 21.6% / 26.3% on val / test). It changes only the fine-tune or the inference of existing members; 72% of val feeds have at least one recovered event but only 3.05% of candidates gain similarity. Null here.

**Not worth building:** feed hour (null, and already trained once), item placement prior (null), fresh co-exposure (null, agrees with the record), any use of the numeric value of an id or of file order (no information).

**Expected size [I].** The 8-column popularity block reads LR 124 in this lens and is worth +0.0004 (distilled students, ± 0.0010) to +0.0025 (hard labels) on val as trained inputs. Blocks U and E together read 33 to 50, about 0.3 to 0.4 of that. A proportional guess is +0.0001 to +0.0010 on val, centre +0.0003. The mapping from LR to NDCG is not linear and the lens sees only additive terms, so the range is wide; but nothing here suggests a missing input of the order of 0.003.

## 6. Limits of this scan

- The lens is additive and linear in one column. An input that matters only through an interaction (a feed-level field such as hour with another input) is invisible to it; hour was tested through two fixed interactions only.
- Weekday against weekend cannot be tested on val (Tuesday and Thursday only).
- Holdout reads of block U use sources that are fresher than on val or test; holdout reads of the popularity-type forms were not made (not admissible).
- The post hoc forms (`uc_coec`, `f1_floor`, `click_sim`) were chosen after the pre-declared reads; `f1_floor` and `uc_coec` are in-sample choices among 10 post hoc forms, although `f1_floor` passes on val and on the holdout separately.
- The whole-history type mix differs between train and val/test rows (type-1 share 0.515 against 0.460) while the newest 32 events agree; this was observed, not explained, and not tested as a signal.
- I did not rebuild any existing feature and did not check the builders of the `tte_*` files beyond reading `src/features_tte.py`; the file-based `uc_ctr` and my rebuild from raw arrays correlate at 1.0000 [M `d05_followup.txt` B].

## 7. Files (all in `reports/INPUTS_1116/raw_fields/`)

| File | Content |
|---|---|
| `d01_inventory.py`, `.txt` | column inventory per split |
| `d02_relations.py`, `.txt`; `d02b_fileorder.txt` | hidden-order checks, tie structure, missing-embedding recoverability, same-user overlap, file order |
| `d03_rule.txt` | rule and list, written before the first LR run |
| `d03_lib.py` | conditional-logit lens, loaders of the val and holdout controls |
| `d04a_signals.py`, `d04a_results.txt`; `d04b_signals.py`, `d04b_results.txt` | the 30 pre-declared reads |
| `d05_followup.py`, `.txt` | 20-shuffle placebo, stale rebuilds, earlier-feed forms, equality forms |
| `d06_poscontrol.py`, `.txt` | positive control of the lens |
| `d07_histfresh.py`, `.txt` | freshness of the row's own history per split (with a correction note on the date labels) |
| `d08_coverage.py`, `.txt`; `d08b_lagcoverage.txt` | coverage and lag of the candidate inputs on val, public, private, train, holdout |
| `d09_bundle.py`, `.txt` | joint reads of the two families |
| `d10_robust.py`, `.txt` | flagged forms under a richer score control |
| `x_<form>_<split>.npy` | the signal columns, aligned with `data/flat/val/cand_*` and with the holdout rows |
