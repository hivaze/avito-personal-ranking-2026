# EDA: user-history semantics and cross-feed (leak-like) signals

Scripts: `<repo>/eda/history_leak/` (01_..15_*.py, with a `.log` for each). All NDCG values are the exact competition metric (`src/metrics.py`: full list, pessimistic ties) on **val (14,423 feeds)**. "paired SE" is the standard error of the per-feed difference against the stated baseline. With f1 as the baseline, val NDCG is **0.3279** (pessimistic ties; random order gives 0.2902).

---

## TL;DR

1. **History is a strict snapshot at feed time.** A later feed's history is the earlier history plus new events added at the front. **What a user clicks in feed A later shows up in the history of that user's next feed B.** Train has 571,923 pairs of consecutive feeds from the same user. Across them, **92.1 % of A's positives appear in B's history, against 0.40 % of A's negatives**, so P(target | item in later history) = **0.847**. When B comes within 1 day of A and the two histories line up well, the recall is **98.5 %**.
2. **Target action ≈ an interaction that starts from this feed surface.** 84.7 % of positives show up in the later history as **event_type 1 (open the item / view)**, and 99.7 % of those type-1 events have **event_source 1**, which looks like the recommendation feed. The other ~15 % show up only as types 2/3 (a contact-like pair) or type 4 with source 1.
3. **The leak on val** (feature "the item appears in the history of a later feed of the same user"):
   - Over f1: **+0.0051** (t = 10).
   - Over our best legitimate LightGBM (0.3707 → 0.3753): **+0.0046** (t = 9.7).
   - The 113 affected val feeds (0.78 %) go from 0.366 to **0.960**.
   - "The item is a candidate again in a later feed" has a target rate of 0 / 123, but it adds nothing on top of the first flag.
4. **Coverage on test is small.** A feed can only use this if the same user has a *later test* feed, because nothing comes after test. That holds for **106 to 108 test feeds (0.38 %)**: **58 of 8,578 Public (0.68 %)** and **48 of 20,017 Private (0.24 %)**. Expected gain: about **+0.0040 Public, +0.0014 Private, +0.0022 overall**. This is a cheap post-processing step worth applying, but it does not change the ranking much on Private.
5. **No wider cross-feed linking is possible.**
   - `user_id = 0` is an anonymous bucket (123,904 train feeds, 658 val, 1,179 test). Its feeds share almost no history with each other (only 6 pairs of user-0 feeds share at least 20 rare items).
   - Linking feeds by history overlap finds only about 17 extra pairs of different user_ids in val+test.
   - `emb_idx` and `item_id` order carry no signal (NDCG 0.288 to 0.289, the same as random).
   - Re-showing an item that was shown earlier (in train) without a click is neutral (target rate 1.61 % vs 1.58 % base).
   - An item found in an *earlier* feed's history is mildly positive (3.8 % target rate on 635 candidates) but adds nothing in practice.
6. **The user embedding is also a snapshot.** A *later* feed's embedding leaks the earlier feed's clicks: cos(ue_next, cand) gives NDCG 0.394 on A's candidates, against 0.363 for A's own embedding (train, 40k feed triples). A's own embedding shows no sign of containing A's clicks. Coverage is the same as point 3.
7. **Legitimate history features**, measured on val:
   - Category affinity, single features: about 0.317. Blended with f1: +0.0018.
   - **Co-visitation from train histories**: 0.327 alone. Rank-blended with f1: **+0.0153** (t = 12).
   - **History-embedding similarity** is the strongest single family: max cosine to the last 50 history items scores **0.348**, against 0.343 for the user embedding and 0.328 for f1.
   - LightGBM ablation (200k train feeds → val):

     | Model | val NDCG |
     |---|---:|
     | base (f1..f5, category) | 0.3322 |
     | + embedding-history | 0.3608 |
     | + category affinity | 0.3620 |
     | **+ co-visitation** | **0.3707** |

     Co-visitation adds **+0.0082 ± 0.0014** over base+emb. Affinity adds +0.0012 ± 0.0012, or +0.0017 ± 0.0009 once co-visitation is in.

---

## 1. Semantics of `history__event_type` and `history__event_source`

### 1.1 Frequencies, position in history, sources

**Val, all 2.50M events** (test is almost identical; the train numbers come from a sample of 200k feeds):

| type | share val | share train | mean hpos | % feeds with it | distinct sources | missing-emb share | repeat share* | after the same item it most often follows (older → newer) |
|---|---|---|---|---|---|---|---|---|
| 1 | 43.9 % | 51.9 % | **78** | 97.1 % | 0,1,2,...,22 | 5.8 % | 14 % | 1→1, 1→2/3/4, 1→10 |
| 2 | 13.9 % | 11.6 % | 143 | 85.6 % | only 0 | 20.0 % | 52 % | 2→3 (63K), 3→2 (73K) |
| 3 | 12.0 % | 10.8 % | 91 | 91.0 % | only 0 | 15.9 % | 88 % | 3→3 (85K), 3↔2 |
| 4 | 10.7 % | 9.3 % | 132 | 78.6 % | 0..25 | 24.2 % | 68 % | 4→4 (86K) |
| 5 | 5.3 % | 4.4 % | 126 | 64.8 % | only 0 | 19.4 % | 94 % (6.0 per item) | 5↔6, 5/6→8 |
| 6 | 4.7 % | 4.1 % | 129 | 60.8 % | only 0 | 20.2 % | 95 % (6.0 per item) | 6↔5, 6→8 |
| 7 | 4.2 % | 3.7 % | 135 | 67.2 % | 0..26 | 25.3 % | 61 % | 7→7, 7→9 |
| 8 | 3.2 % | 2.6 % | 151 | 53.7 % | only 0 | 25.7 % | 91 % (7.0 per item) | 8↔5/6 |
| 9 | 1.5 % | 1.2 % | 145 | 43.2 % | only 0 | 32.9 % | 75 % | 9→9, after 7 |
| 10 | 0.5 % | 0.4 % | **44** | **6.8 %** | only 0 | 1.3 % | 96 % | 1→10 (10.7K of 13.1K) |

\*Repeat share is the share of that type's events whose item has more than one event in the same history.

**Event sources:**
- Only types **1, 4 and 7** have a non-zero `event_source`. Types 2, 3, 5, 6, 8, 9 and 10 always have source 0. So a source is a UI surface (where the item was opened from), and it only exists for "surface" actions.
- In val, source 1 = 33.8 % of all events and source 2 = 13.8 %. Sources 3 to 29 form a long tail.
- **Type 1 by source:** source 1 = 707K (64.5 %), source 2 = 260K (23.7 %), source 0 = 47K, source 4 = 35K, sources 5 to 7 = 38K.
- **Type 4 by source:** source 1 = 112K (42 %), source 2 = 56K, source 3 = 39K, source 0 = 38K.
- **Type 7 by source:** fairly even across sources 0 to 3.

**Event type by position in history:**
- Type 1 dominates the newest positions: 258K type-1 events in hpos 0 to 31, against 13K in hpos 224 to 255.
- Types 2, 4, 8 and 9 dominate the oldest positions.

This is explained by **per-type retention windows** (section 2.2): view events expire much sooner than contact or message events.

**Category and embedding coverage:**
- `category_id = 0` occurs only for type 2 (0.025 %).
- The share of events without an embedding ranges from 5.8 % (type 1) to 33 % (type 9).
- In 23 % of (feed, item) pairs the item has more than one event, and 48 % of events fall on such repeated items.

**Most plausible reading** (codes are anonymised, so this is inference):

| Code | Likely meaning |
|---|---|
| 1 | Item-card view / open, with the surface it came from |
| 4 | A second surface-bearing action: favourite / add to cart style, often repeated |
| 7, 9 | A surface-bearing action (7) and its follow-up (9) |
| 2 and 3 | A pair that almost always comes together: a contact intent (e.g. show phone / start chat) |
| 5, 6, 8 | Messaging events (sent / received / read): many per item, alternating 5↔6, ending in 8 |
| 10 | Rare and short-lived; almost always follows a type-1 view of the same item |

### 1.2 History length

- Mean history length H: train 185, val 173, test 174.
- H = 256 (truncated) for 29 % of val feeds. H = 0 for 1.3 %.
- Type-1 share is higher in train (51.9 %) than in val/test (44 %), a train-to-eval distribution shift.
- Single-feature NDCG of H alone is not meaningful, because H is constant within a feed.

---

## 2. Is the history a snapshot at feed time?

### 2.1 Consecutive feeds of the same user (train; `02_snapshot.py`, `11_target_semantics.py`, `13_dropped_events.py`)

The data: 571,923 pairs (A earlier, B later) from the same non-zero user, covering 46.8M candidates of A. Examples (`user 27`, `user 74`, ...) show B's history = a few new events + A's history, and **the new events are exactly A's positives, as type-1 events**.

**Alignment.** For each pair we looked for a shift k such that B[k:] matches A's history on (item, type):
- 8.6 % of pairs match perfectly, 24.9 % match at least 90 %, and 5.9 % have no anchor at all.
- The number of new events grows with the time gap. Median: 1 (gap ≤ 128 s), 3 (≤ 1 h), 9 (≤ 1 d), 34 (≤ 7 d), 51 (> 7 d).
- When the gap is 0 s, k = 0 in 42 % of pairs.

**Why most alignments are imperfect:**
- Events that share a timestamp come in unstable order, e.g. 5/6 swapped.
- **View events expire.** Take A's events that should still fit inside B's window of 256. Among these, B has dropped 18 % of type-1 events, and still drops 31 % even when B's history is not full. That compares with 6 to 12 % for types 2, 4, 7, 8 and 9.
- The type-1 drop rate rises with the event's age in A: 1.3 % at hpos 0 to 31, 47 to 49 % at hpos 192 and above.
- So history = the newest 256 events, after per-type time windows have been applied, and views have the shortest window.

**Where A's candidates end up in B:**

| | A positives (1.107M) | A negatives (45.7M) |
|---|---|---|
| found anywhere in B's history | **92.1 %** | 0.40 % |
| found in B's *new* part (k first events) | 88.7 % | 0.33 % |
| appear again as a candidate of B | 0.06 % | 2.78 % |
| target rate given "found in B's history" | **0.847** | |
| target rate given "candidate in B again" | 0.054 % (vs 2.36 % base) | |

**Recall of A's positives in B's history, by time gap:**

| gap | ≤ 128 s | ≤ 10 min | ≤ 1 h | ≤ 6 h | ≤ 1 d | ≤ 3 d | ≤ 7 d | > 7 d |
|---|---|---|---|---|---|---|---|---|
| recall | 68.9 % | 94.4 % | 96.7 % | 97.4 % | 96.7 % | 94.7 % | 90.1 % | 84.1 % |
| precision P(t \| in B's history) | 0.978 | 0.970 | 0.953 | 0.925 | 0.902 | 0.858 | 0.814 | 0.786 |

Recall drops for very short gaps because some clicks happen after B's snapshot. It drops for long gaps because events expire or get pushed past 256.

### 2.2 Which event types are the target action?

This uses 48,351 well-aligned pairs (gap ≤ 1 d, match ≥ 0.9).

- **98.5 % of A's positives appear in B's new part.**
  - 84.7 % of them have a type-1 event, and **83.0 % have type 1 with source 1**.
  - Shares of positives with other types: type 2 10.8 %, type 3 12.1 %, type 4 5.6 %, types 5/6 2.0 / 1.6 %, type 7 1.5 %.
  - Only type 1: 78.5 %. Only type 4: 3.4 %. Only type 3: 2.1 %. Only type 2: 1.3 %.
- The 15 % of found positives with *no* type-1 event mostly have types 2/3 (53 % / 58 %) or type 4 (24 %).
- Among the oldest occurrences of A's candidates in B (all train pairs), target rates by type: type 1 0.853, type 4 0.854, types 5/6 0.87, type 2 0.79, type 3 0.81, types 7/9 0.74, type 10 0.20.
- **Of B's new events with type 1 / source 1, 42 % are A's positives**, and similarly 41 % for type 4 / source 1 and 38 % for type 7 / source 1. Type 1 / source 2 is only 0.1 %.

**Conclusion:** the target is "≥1 interaction with the item that started from this feed". Mostly that is a card open (type 1, source 1). Sometimes it is a feed-surface action without a recorded view (type 4 / source 1), or a contact action (2/3). The other type-1 / source-1 events in B come from other, unobserved feed impressions.

### 2.3 Candidates never appear in their own history

**Verified on all of train: 11 of 246,745,836 candidates (4e-8). Zero in val and test.**

This is the README rule "items the user already viewed are excluded". It was applied to **candidates**, not to the history:
- 85.6 % of train feeds have gaps in `pos`. On average 7.9 positions are missing (median 4), about 9.9 % of the original list.
- In 15 % of feeds the top position (`pos = 0`) was removed.
- Even feeds with H = 0 lose 2.6 positions on average, so the filter uses a wider "viewed" log than the 256-event history.
- Items viewed long ago (absent from the current history but present in an *earlier* feed's history) can come back as candidates: 3.4 % of val feeds have one. This matches view events expiring.

The filter does **not** remove an item from earlier feeds' candidate lists when it is clicked later: clicked items stay as positives. **So the filter itself leaks nothing usable in val/test.** The `pos` gaps exist only in train.

### 2.4 User and item embeddings (`07_emb_consistency.py`, `09_ue_leak.py`)

**User embeddings are snapshots:**
- Only 0.08 % of consecutive same-user train feeds have identical vectors.
- Median cosine between consecutive feeds falls with the time gap: 0.974 (< 1 h), 0.935 (< 1 d), 0.864 (< 7 d), 0.823 (> 7 d). Different users have a median cosine of 0.405.

**Leak test** on 40,000 train triples P → A → B, scoring A's candidates by cos(user_emb, item_emb):

| user embedding taken from | NDCG on A |
|---|---:|
| P (previous feed) | 0.3536 |
| A (own) | 0.3630 |
| B (next feed) | **0.3939** (+0.031 ± 0.001 over own) |

The later snapshot contains A's clicks. The own snapshot does not look contaminated: it is weaker than B's.

**Item embedding tables:** 76 to 78 % of shared items have identical vectors across split tables, and the rest are near-identical (median cos 0.9993 to 0.9996, 1st percentile 0.98). They were recomputed per split with little drift.

---

## 3. Cross-feed (transductive / leak-like) signals (`03_leak.py`, `08_item_transductive.py`, `14_final_eval.py`)

**Linking rule.** For every val/test candidate we checked every other feed of the same non-zero user in train+val+test: 20,374 train feeds, 13,765 val feeds and 27,416 test feeds, holding 12.1M history rows and 5.8M candidate rows. Exactly 0 history matches share the candidate's own timestamp.

**What linking is possible:**

| | val | test |
|---|---|---|
| feeds whose user has an earlier train feed | 34.9 % | 33.2 % |
| non-zero users with ≥2 feeds in the split | 25 | 105 (102 with 2 feeds, 3 with 3) |
| users shared val ↔ test | 93 users (0.68 % of val users) | |
| `user_id = 0` feeds (anonymous; cannot be linked) | 658 (4.6 %) | 1,179 (4.1 %) |
| feeds with a **later** feed of the same user | 117 (37 in first 30 % / 80 in last 70 %) | **108** (59 Public / 49 Private) |
| median gap to the next feed | 94 h | 23 h |

### 3.1 Val target rates (base 1.58 %)

| signal | candidates | feeds | target rate | positives |
|---|---|---|---|---|
| **later_hist**: item in a later feed's history | 184 | 110 (0.76 %) | **0.935** | 172 |
| ... first-occurrence type 1 / 2 / 3 / 4 | 147 / 11 / 7 / 14 | | 0.966 / 0.55 / 0.86 / 0.93 | |
| later_cand: item is a candidate again later | 123 | 46 | **0.000** | 0 |
| earlier_hist: item in an earlier feed's history | 635 | 484 (3.4 %) | 0.038 | 24 |
| earlier_cand: shown earlier (train) and not clicked | 2,526 | 1,216 (8.4 %) | 0.016 (neutral) | 41 |
| earlier_cand: shown earlier and clicked | 16 | 16 | 0.000 | 0 |

In val feeds that have a later feed, 172 of 224 positives (77 %) are flagged. Recall is lower when the next feed comes more than 3 days later (102 of 145) than when it comes within 3 days (70 of 79), consistent with view events expiring.

### 3.2 NDCG impact on val (paired, per feed)

| score | val NDCG | delta | t |
|---|---|---|---|
| f1 | 0.3279 | | |
| f1 + 10 · later_hist | 0.3330 | +0.0051 | 10.1 |
| f1 + 10 · later_hist − 10 · later_cand | 0.3330 | +0.0051 | 10.1 |
| f1 ± earlier_cand | 0.3278 / 0.3272 | −0.0001 / −0.0007 | n.s. / harmful |
| **LGB (base+emb+aff+covis)** | **0.3707** | | |
| + leak_pos − leak_neg | **0.3753** | **+0.0046** | 9.7 |
| + 0.5 · earlier_hist (types other than 1) | 0.3755 | +0.0048 | 9.9 |

On the 113 affected val feeds, NDCG goes from 0.366 to **0.960**, a gain of 0.594 per affected feed. earlier_hist adds no more than +0.0002, within noise.

### 3.3 Test coverage and expected gain

| | Public (first 30 %: 8,578 feeds) | Private (20,017 feeds) | all |
|---|---|---|---|
| feeds with ≥1 leak_pos candidate | 57 | 46 | 103 (183 candidates) |
| feeds with leak_pos or leak_neg | 58 | 48 | 106 |
| expected NDCG gain (0.59 per affected feed) | **+0.0040** | **+0.0014** | +0.0022 |

Test gaps are shorter than val gaps (median 23 h), so recall should be at least as high as on val (train: about 97 % for gaps ≤ 1 d). The leak is skewed toward Public, because an earlier feed needs a later feed inside the same 4-day window. **It explains part of any Public-vs-Private gap on the leaderboard.**

### 3.4 Item-level transductive signals (other users)

- Number of *other* users' val+test feeds whose history contains the candidate:
  - Later feeds, any event: 0.306 alone, **+0.0049** with f1.
  - Later feeds, type 1 / source 1: +0.0053.
  - Earlier feeds: +0.0014.
- Candidate exposure count in other val/test feeds: +0.0033 with weight 0.25, neutral at weight 0.5.
- Coverage is asymmetric. On val, "later" includes all of test (17 % of candidates nonzero). On test, only later test feeds count (8.6 %), so the val gains overstate test.
- Items rarely repeat: val has 1.24M unique candidate items in 1.39M rows, and only 220K items are shared between val and test.

### 3.5 Checked and found negative

- `item_id__embedding_idx` and `item_id` order: NDCG 0.2876 to 0.2890, the same as random.
- Linking by history overlap: same-user pairs share at least 42 rare items (5th percentile). Only 11 pairs of different non-zero users and 6 pairs of `user_id = 0` feeds share at least 20 rare items. This is negligible.
- The candidate-filter rule (section 2.3) gives no usable signal in val/test.

---

## 4. Legitimate history-affinity features (`04_affinity.py`, `hist_feats.py`)

These are per-candidate statistics of the feed's own history restricted to the candidate's category: counts, recency-weighted sums, the most recent position, counts by event type, "feed clicks" (type 1 / source 1), and the category's rank.

**Single features on val** (random tie-break; pessimistic ties score about 0.06 lower because the features are count-like):

| feature | NDCG | t vs random |
|---|---|---|
| cat_rec_feedclick (Σ 0.95^hpos over same-category type-1/source-1 events) | **0.3173** | 15.9 |
| cat_minpos (most recent position of the same category) | 0.3168 | 15.5 |
| cat_rec80 / cat_rec95 | 0.3166 / 0.3157 | 15.5 / 14.9 |
| cat_frac20 (same-category share of the last 20 events) | 0.3149 | 14.7 |
| cat_cnt_feedclick | 0.3144 | 14.1 |
| cat_cnt_t1 | 0.3130 | 13.5 |
| cat_cnt / cat_frac / cat_rank | 0.3105 / 0.3091 / 0.3101 | 12 |
| cat_cnt_t2 / t3 / t4 / t5 | 0.2997 / 0.2991 / 0.2970 / 0.2962 | 4 to 6 |
| cat_cnt_t7 / t9 / t10 | 0.291 / 0.289 / 0.288 | none |

**Target rate by same-category count** rises steadily: 0.66 % at 0, 1.05 % at 1, 1.42 % at 4 to 10, 2.01 % at 31 to 100, and 2.14 % above 100.

**Combinations:**
- Blended with f1, best is rank(f1) + 0.5 · rank(cat_rec_feedclick) = 0.3297 (+0.0018, t = 3.4).
- In LightGBM: f1..f5 + category alone = 0.3305, and adding the affinity block gives 0.3341 (+0.0036 ± 0.0015).
- Over base + embedding-history, affinity adds +0.0012 ± 0.0012. On top of co-visitation it adds +0.0017 ± 0.0009.
- The category id itself is the top LightGBM gain feature (15 to 23 % of gain).

---

## 5. Co-visitation and history-embedding similarity

### 5.1 Item–item co-visitation from train histories (`05_covis.py`)

**How it is computed:**
- **Source:** the latest train feed per user, plus every `user_id = 0` feed: 2.58M sequences, 456M events, 91.4M unique items.
- **Score:** for a candidate c, scan every source occurrence of c and its neighbours within ±W positions in that sequence. Sequences of the same user are excluded. Add up the neighbours that are in the feed's own history, each weighted by its recency 0.95^hpos.
- Variants: W = 5 vs 20, 1/distance weighting, forward vs backward direction, normalised by popularity, plus `pop_hist` (the number of source sequences containing c).
- It runs in seconds with numba.

**Coverage:**
- 58 % of val candidate rows appear in a train history at all (test: 51 %). 42 % are items unseen in train histories.
- The co-visitation score is nonzero for 11.3 % of val candidates with W = 20, and 6.4 % with W = 5.
- Target rate is 3.0 to 3.6 % when the score is nonzero, against 1.4 % when it is zero.

| score | alone | rank(f1) + w · rank(score), best w |
|---|---|---|
| cv_invd20 (Σ recency / distance, W = 20) | **0.3266** | **0.3432** (w = 1, +0.0153, t = 12.1) |
| cv_raw20 | 0.3243 | |
| cv_rec5 (W = 5, recency-weighted) | 0.3204 | 0.3402 (+0.0123) |
| cv_fwd5 (c came after the history item) | 0.3121 | 0.3373 |
| pop_hist (popularity in train histories) | 0.3098 | 0.3376 (w = 0.25, +0.0097) |

**Other observations:**
- Adding val histories, or val+test histories, of other users as sources: +0.0003 / +0.0007 on top (0.3435 / 0.3439).
- **Coverage decays over time.** Nonzero share of cv_invd20: val 11.3 %, test 9.7 % (test day 0 10.1 %, day 3 9.4 %; Private 9.5 %). For test, the val+test histories should be included as sources.
- In LightGBM, co-visitation adds **+0.0082 ± 0.0014** over base+emb (0.3608 → 0.3690). `pop_hist` and `cv_invd20` are among the top 6 gain features.

### 5.2 History-embedding similarity (`06_emb_hist.py`)

Cosine between the candidate's item embedding and summaries of the user's history embeddings. Single features on val:

| feature | NDCG |
|---|---|
| **he_max50**: max cosine to the last 50 history items | **0.3478** |
| he_cos_last10: cosine to the mean of the last 10 items | 0.3457 |
| he_cos_rec95: recency-weighted mean | 0.3440 |
| ue_cos: user embedding | 0.3432 |
| he_cos_last1: the newest item | 0.3395 |
| he_cos_feedclick: mean over type-1/source-1 items | 0.3336 |
| he_cos_all | 0.3292 |
| he_cos_t2t3 | 0.3120 |
| candidate embedding norm | 0.2896 |

### 5.3 LightGBM ablation

Lambdarank, 200k random train feeds (15.6M candidates), 500 rounds, evaluated on all of val:

| features | val NDCG | delta vs previous row |
|---|---|---|
| f1..f5 + category | 0.3322 | |
| + embedding-history (9 features) | 0.3608 | +0.0286 |
| + category affinity (25 features) | 0.3620 | +0.0012 ± 0.0012 |
| + co-visitation instead of affinity | 0.3690 | +0.0082 ± 0.0014 vs base+emb |
| + affinity + co-visitation | **0.3707** | +0.0099 ± 0.0014 vs base+emb |
| + cross-feed leak post-processing | **0.3753** | +0.0046 ± 0.0005 |

---

## 6. Recommendations

1. **Apply the leak as post-processing on test, not as a model feature.** In train, 18 % of feeds have a later feed, against 0.4 % in test, so a model feature would be badly miscalibrated.
   - Formula: `score += BIG · leak_pos − BIG · leak_neg`.
   - The flags for val and test are in `<repo>/eda/history_leak/leak_flags_{val,test}.parquet`, keyed by `feed_id, item_id`, with columns `leak_pos`, `leak_pos_event_type`, `leak_pos_dt_sec`, `leak_neg`, `earlier_hist`, `user_has_later_feed`.
   - Expected gain: +0.0040 Public, +0.0014 Private.
   - Optionally, for these ~108 feeds only, use the later feed's user embedding (+0.03 NDCG per feed on train) to re-rank the unflagged candidates.
2. **Add co-visitation features.** Use cv_invd20, cv_rec5, pop_hist and cv_fwd5, computed from train histories plus the val+test histories of *other* users, and always leave out the same user's own sequences. This is the largest legitimate history gain found here (+0.008 over a model with embeddings).
3. **Add history-embedding similarity features** (max or recent-mean cosine), plus a few affinity features (cat_rec_feedclick, cat_minpos, cat_rec80, cat_frac20).
4. **Use event semantics in sequence models.** Type 1 / source 1 events are past "target actions", so separate them. Treat 5/6/8 as messaging and 2/3 as contact. View events expire faster, so for long-dormant users recent type-1 events may simply be missing.

---

## 7. Risks and pitfalls

- **Val overstates the leak.** For val, later feeds include test feeds 1 to 7 days later. For test, only feeds within test count (106 feeds). Use the coverage table in 3.3, not the val delta.
- **Private gets little from the leak** (48 of 20,017 feeds). Do not let Public gains from it drive model selection.
- **Leakage in co-visitation evaluation.** Using test histories as a source when scoring val would include the val user's own later clicks. The script excludes same-user sequences; any re-implementation must do the same. `user_id = 0` sequences cannot be excluded this way, but they share almost no history with each other.
- **Training on train with history features.** Train is production-ranked traffic: target rate falls from 11.8 % at pos 0 to 1.0 % at pos ≥ 200. It also has more type-1 events (51.9 % vs 44 %). Train co-visitation features computed from other users' train histories include somewhat "future" data within the train period.
- **Pessimistic ties.** Count-like history features have many ties, and single-feature NDCG drops by about 0.06 under pessimistic ties. Always break ties with a continuous score.
- **Co-visitation decays with time.** Coverage falls from 11.3 % on val to 9.5 % on test Private, and 42 to 49 % of candidates are items never seen in train histories.
- The history is not a clean append-only log. Same-timestamp events get reordered and view events expire. Sequence-diff logic must tolerate that (only 8.6 % of consecutive histories align perfectly).

---

## Appendix: scripts and outputs

| script | purpose |
|---|---|
| `01_semantics.py` | type/source frequencies, hpos, repeats, transitions (val, test, 200k train feeds) |
| `02_snapshot.py` | consecutive-pair alignment; where A's candidates appear in B (`out_pairs_train*.{npz,parquet}`) |
| `03_leak.py` | cross-feed features for val/test (`out_crossfeed_{val,test}.parquet`), val NDCG, test coverage |
| `04_affinity.py`, `hist_feats.py` | category-affinity features and LightGBM increment |
| `05_covis.py` | co-visitation kernel (`out_covis_*.npy`); argument `val` or `val,test` adds sources |
| `06_emb_hist.py` | history-embedding features and the final LightGBM ablation (`out_lgb_val_*.npy`) |
| `07_emb_consistency.py`, `09_ue_leak.py` | snapshot nature of user/item embeddings; later-user-embedding leak |
| `08_item_transductive.py` | other-user item popularity in val/test histories |
| `10_pos_gaps.py`, `15_own_hist_check.py` | candidate filtering (pos gaps), own-history check on all of train |
| `11_target_semantics.py`, `13_dropped_events.py` | which event types are the target; per-type expiry of history |
| `12_export_leak_flags.py` | `leak_flags_{val,test}.parquet` for downstream use |
| `14_final_eval.py` | leak on top of LightGBM; expected test gain |
