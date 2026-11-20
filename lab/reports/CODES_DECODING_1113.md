# CODES DECODING 11-13: what the anonymised codes are (synthesis and skeptic pass)

Inputs: four analyst decodings (event types, event sources, category codes, features f1..f5 and the target). Their scripts and logs are under `nn_x/codes/{events,sources,categories,features}/`.

This pass did three things:
1. Cross-checked the four decodings against each other.
2. Re-graded every claim as either **IS** (what the code is: an identity or encoding claim) or **CORR** (what the code is correlated with).
3. Ran six new checks (C1-C4 here, R1-R6 in §6). Scripts and logs: `nn_x/codes/synth/x01_skeptic_checks.py` → `logs/x01.log`, `logs/c1_delivery_share.csv`; `nn_x/codes/synth/x02_decoded_residuals.py` → `logs/x02.log`.

Compute was CPU only with 12 threads. No submissions. No GPU.

Confidence words:
- **high**: several independent tests agree, and no live alternative remains.
- **medium**: consistent with all the data, but one plausible alternative remains.
- **low**: a best guess.

Each grade below is mine. Where it differs from the analyst's grade, the reason is given.

---

## 0. Executive summary

1. **IS (high):** event_type 1..10 and event_source 1..30 are frequency ranks over train history events (0 rank inversions in 45 and 435 pairs). The numbers carry no meaning beyond that rank.
2. **IS (high):** category codes are not frequency-ranked. They follow Avito's public category-ID order: code = 1 + dense rank of the leaf category id, so c2 = Cars (id 9) ... c50 = Services (114), c51 = Ready business (116), c52/c53 = two newer categories, c1 unused.
   - The 11 **verticals** (Transport, Real estate, Personal, Home, Electronics, Hobby, Animals, Jobs, Services, Business, Travel) are **high**.
   - Most **leaf names** are **medium**, because the id list comes from memory and only ids 9 and 84 are web-verified.
3. **IS (high):** the target is an engagement **started from the recommendation feed (source s1)**.
   - 83-87% of positives come back in a later history as a card view (e1) with source s1.
   - The rest come back as source-1 favourites (e4) or source-less contact/delivery events.
   - Views from other surfaces (search s2, similar-items s4) almost never count (target rate 0.7-1.6%).
4. **CORR → near-IS (high for the block, low for which code is which):** e5/e6/e8 are a **category-gated transaction block (Avito Delivery)**.
   - In 15 codes the share is **exactly 0 over ~7.9M events**: all of real estate, cars, trucks, jobs, services, ready business, hotels and shift gigs. Check C1 settles the three analysts' conflicting lists.
   - Live animals and food are near 0 (≤0.2%). Goods categories run 8-29%.
5. **CORR (high), IS (medium):**
   - user_id 0 = a not-logged-in device session. Login-only types e2/e5/e8/e9 never occur, but f1..f3 stay personalised.
   - e10 marks the guest state: it is present in 86-88% of user-0 feeds that have history, vs 3.7% of logged-in feeds.
6. **IS (high):** f3 = user × **leaf category** affinity, not "top-level category" as the features analyst wrote. f1 = a finer (sub-leaf) affinity. f2 = user × **item-region** affinity: items share a cross-category region key, and f2 = 0 means no activity in that region. f4 = listing age. f5 = an unsmoothed item engagement rate. **None of f1..f5 is a leaked production score.**
7. **Two analyst claims are refuted by my checks:**
   - **"Similarity to e7/e9 items is negative (completed intent)" is a category confound.** Within (feed, category) the odds ratio is 1.16 (val) / 1.18 (train-explore), not 0.83 / 0.94. 28% of the flagged candidates are Services, the lowest-CTR big category (check C3).
   - **"s3 = item card" (events analyst) is wrong.** Card actions inherit the source of the view that opened the card. s3 is the favourites list (sources analyst).
8. **One claim survives a confound check:** the home-region (f2 = feed mode) CTR deficit holds within (feed, category): OR 0.82 val / 0.89 train-explore (check C2). G3 is already calibrated on it (act/exp 0.981).
9. **Six decoding-enabled features were tested as G3 residuals with a pre-declared rule (§6). 0 of 6 pass.** They were:
   - user × vertical share, the share outside the leaf, last-event vertical match and vertical recency;
   - the delivery-activity × deliverable-category cross;
   - the structural-zero mask.

   G3 shows a small calibration gradient on vertical recency (act/exp 0.88 to 1.04), but a nested blend recovers nothing (-0.00012 ± 0.00024).
10. **The leak-refinement idea (events analyst §5) is moot.** The later-feed leak is neutralised on test (public 0.392819 = G3) and is banned by the project leak rule. Graded relevance by event type is not supported (target rates per type 0.74-0.90).
11. **Expected LB value of the decoding is ~0.** The ranker already embeds every code and sees f1..f5. The remaining decoding-only levers are in §8, each with prior ≤ 0.1 and a pre-declared CPU test. The most useful immediate use is **interpretability for the running pop-free family**: f5 is an unsmoothed rate, so a model without exposure counts should mis-rank "noisy-high f5" items. This is a free readout (§8, item D1).
12. **Documentation value is real.** The named category table (§3), the event funnel (§1), the source families (§2) and the f-semantics (§4) should replace guesswork in future error analyses.

---

## 1. Event types e1..e10 (history__event_type)

**Encoding (IS, high).** Full-train shares are 51.88 / 11.64 / 10.82 / 9.29 / 4.41 / 4.11 / 3.67 / 2.55 / 1.19 / 0.44%, with 0 of 45 pairs inverted (`events/e01_code_ranks.log`). The readme says the codes "are not the original event numbers".

| code | meaning | IS / CORR | confidence (analyst → mine) | key evidence |
|---|---|---|---|---|
| e1 | item-card view (open). Carries the surface it was opened from; s1 = feed | IS | high → high | 43.9% of val events. A source in 95.7% (s1 64.5%, s2 23.7%). Deduplicated per item (1.009 per item in fresh flow). Oldest event on an item 67-87% of the time. Conditional-logit similarity weight 0.53-0.66 per SD vs ≤ 0.10 for other types. 2025 analog: code 17 (88.8%, non-contact, precedes everything). |
| e4 | add to / remove from favourites (toggle) | IS | medium → **medium-high** | Carries a source and inherits it from the opening view (s1 94.7%). 61% of s3 (favourites-list) events are e4. 2.2 events per item, and the next same-item event is e4 79% of the time (toggles). Allowed for guests (4% of anonymous events). Highest share in realty, cars and hotels (0.29-0.31). The upgrade comes from the s3 cross-check (§2). |
| e5, e6, e8 | Avito Delivery / purchase-flow block. e5+e6 are one action logged twice (same moment). e8 is a later order step | CORR (block) → near-IS | medium → **medium-high** (block); **low** (which code is which) | Exactly 0 in 15 codes over ~7.9M events (C1). Category shares correlate 0.99-1.00 across the three. Login-only (e5, e8 = 0 for guests). e8 follows e5/e6 (0.80). The same structure appears in 2025 (delivery-only codes are zero in 18 of 53 categories; 3+15 fire in the same second). Avito chat exists in every vertical, so a category-gated block cannot be chat (external fact, unverified). |
| e2 | first contact with the seller about this item (unique per user × item; login required) | CORR | low → low | No source. Once per item (1.013 fresh). 0 for user 0. Fires at the same moment as e3 and precedes e9 and e5/e6/e8. Present in every category. 2025 analog: code 10 (contact, once per item). |
| e3 | repeatable contact action (phone reveal / call / message) | CORR | low → low | No source. 2.36 per item, in bursts. Possible as a guest, concentrated in jobs (et3/et2 = 6.45 in c48) and services. |
| e7 | lead-type action launchable from tiles, the card or the favourites list (services / realty / jobs) | CORR | low → low | Has a source (s0..s3 about equal; 34% of s3 events). Lifts: services ×7.3 (guests), rooms ×5.6, commercial ×5.0, jobs ×2.7. |
| e9 | login-only completion of e7 / e2 | CORR | low → low | No source. 0 for guests. Follows e7 (0.67) and e2 (0.89). Longest retention. 48% of old e9 items have no embedding (closed listings). |
| e10 | guest-state marker attached to a view (login prompt or guest-view duplicate) | CORR high / IS low | medium → medium | In 86.1% / 87.8% of user-0 feeds with H > 0 (val / test) vs 3.7% / 3.6% of logged-in feeds (C4). 97% of e10 items also have an e1, and the e1 comes first 87-89% of the time. Flat category profile (lift ≈ 1). |
| TARGET | ≥ 1 engagement started from this feed impression: mostly a card open from s1, plus s1 favourites and source-less contact / delivery actions | IS | high → **high** (surface); **medium** (composition) | 571,923 consecutive train pairs. 92.1% of positives reappear in B vs 0.40% of negatives. First-event target rate: 1s1 0.887, 4s1 0.897, 2 0.787, 3 0.808, 5 0.873, against 1s2 0.016 and 1s4 0.007. Every train and val feed has ≥ 1 positive. Downgraded composition: positives with a contact event but no view (68% of e2-positives) could be contacts started from a tile, or views lost to history compaction (§1.2). Not separated. |

### 1.1 Statistics that matter
- **Retention depends on type**, so hpos is a type-dependent clock. Survival of A's events into an untruncated B at gaps > 7 d: e1 0.54, e10 0.38, e3 0.77, e5 0.83, e4 0.87, e2 0.89, e7/e8 0.91, e9 0.94. Even at gaps ≤ 1 d, an e1 at hpos ≥ 192 survives only 11%.
  - Consequence: old history is contact-heavy by construction, and "no views in the tail" is censoring, not disinterest.
  - The sources analyst independently finds hpos within type 1 is 73-91 for every source: retention follows the type, not the source.
- Item-level type sets: {1} 56.2%, {2} 10.0%, {4} 7.7%, {2,3} 4.6%, {7} 2.9%, {3} 2.8%, {1,4} 1.5%, {5,6} 1.5%.
- Anonymous type mix (val): e1 0.61, e10 0.256, e7 0.068, e4 0.034, e3 0.026, e6 0.002. e2, e5, e8 and e9 are exactly 0.

### 1.2 Open question (not resolvable from the analysts' outputs)
Why do 13-17% of found positives have no e1 at all, even at gaps ≤ 10 min?
- (a) The contact started directly from the feed tile (Avito has call/write buttons on some tiles).
- (b) History compaction drops a view once a stronger event exists on the same item.

Evidence for (b): views are deduplicated (1.009 per item), and e1 survival within 1 h is only 0.77 vs 0.92-0.98 for other types. A test: in A → B pairs, compare survival of A's fresh e1 on items that gained a contact in B with items that gained nothing. Value for ranking ≈ 0; this matters only for wording.

---

## 2. Event sources s0..s30 (history__event_source)

**Encoding (IS, high).** Frequency ranks: 0 of 435 pairs inverted on full train.

**Scope (IS, high).** A source is filled only for e1, e4 and e7. Every other type is always s0. e4 and e7 inherit the source of the item's nearest older view: s1 94.7%, s2 87.8%, s4 79.5%. So a source is the UI surface where the item was opened, not where the action button sat.

**Platform (CORR, medium).** Platform is NOT encoded: no source pair is mutually exclusive across users, and dayparts are flat within ±13%. A 2025-style "last platform" feature cannot be rebuilt.

| code | meaning | family | IS / CORR | confidence (analyst → mine) | key evidence |
|---|---|---|---|---|---|
| s1 | main-page recommendation feed: the competition surface | recommendation (target) | IS | high → high | A's positives return with oldest event (1,s1) in 83.0%. New (1,s1) events in B were A-candidates 13.2% of the time, 94.5% of them positive. 33.8% of events; lowest consecutive-view cosine (0.64), as expected for a diversified feed. |
| s2 | search / catalogue listing (SERP) | search | CORR | high → **medium-high** (listing family); **medium** (search vs category browse) | Consecutive s2 → s2 cosine 0.79, same category 70%. Contact follow-up 7.2% vs 4.1%. P(target \| first event 1s2) = 0.016. The evidence is behavioural, so search cannot be told apart from browse. |
| s3 | favourites (saved-items) list | saved | CORR | medium → medium | Type mix 5/61/34 (views/e4/e7). 50% of its views are re-views. The nearest older view is never s3. 25% of items lack an embedding (closed listings kept in the list). Guest O/E 0.27. **Resolves the conflict:** the events analyst's "s3 = item card" is impossible, because card actions inherit the opening view's source (94.7% for s1). |
| s4 | "similar listings" block on an item card (item-to-item recs) | recommendation (other) | CORR | medium → medium | Cosine with the previous view 0.76 (median 0.83), same category 74% whatever the previous source. 1.8% of events. Feed overlap 0.9%, of which 3.5% are positive. |
| s5 | goods-only, login-only search-like surface | search-like | CORR | low → low | 91% of its views fall in delivery categories. Guest O/E 0.09. |
| s6 | second personalised recommendation surface (not the logged feed) | recommendation (other) | CORR | medium → **low-medium** | Category mix closest to s1 (JS 0.017). Candidate-pool share near s1's. Click-backs: 1.8% of new s6 views were A-candidates, but only 1.9% of those were positive. |
| s7 | real-estate surface (map / housing-complex page) | vertical | CORR | medium → medium (vertical); low (which page) | 85% of its events are in c10/c11/c12 (57/21/7%) (Apartments / Houses / Land, consistent with §3). Consecutive cosine median 0.89. Favourite follow-up 9%. |
| s8 | unidentified (old / closed items) | ? | - | low | 0.1% of events |
| s9 | vertical recommendations for jobs / cars / realty, guest-heavy | recommendation (vertical) | CORR | low | Categories 48/2/10; guest O/E 1.12 |
| s10, s13 | saved-items-adjacent lists | saved | CORR | low | Transitions with s3 lift 3.2-3.5; s13 has an s3-like type mix |
| s11, s12 | goods-only surfaces (c13/c29) / login-only, s4-like | ? | CORR | low | - |
| s14, s17, s21 | real-estate (s14, s17) and jobs (s21) vertical surfaces | vertical | CORR | low | s14: 50-61% c10; s17: 96% e4 in c10/c11; s21: c48 only |
| s0 | unknown / missing surface (deep link, push, external landing) | unknown | IS (readme) | medium | Readme says 0 = unknown. 4.3% of views. Some feed clicks lose their source: 28% of new (1,s0) events that were A-candidates were A-positives. |
| s15..s30 | rare tail (< 0.006% each) | - | - | - | Not identifiable |

### 2.1 Statistics that matter
- P(source | view), val: s1 64.5, s2 23.7, s0 4.3, s4 3.2, s5 1.5, s6 1.2, s7 0.75%. This mix is the same in train production, train exploration, val and test, so the train-vs-eval view-share gap is **not** a source shift.
- Only 0.42% of viewed items are viewed from two or more sources.
- No viewed item, from any source, reappears as a candidate in a later feed of the same split: exclusion covers types 1, 4 and 7 from all sources.

---

## 3. Category codes c1..c53

**Numbering rule (IS).** Code = 1 + dense rank of Avito's public leaf category id. The rule fits the data:
- Adjacent codes share a data cluster far more often than chance: 20/51 adjacent pairs vs a null of 7.5 ± 2.3 (embedding kNN), p < 1e-5.
- Vertical ARI against history transitions is 0.55 (p < 1e-4).

The vertical split is **high**. The leaf names are **medium**, with three caveats:
- Only Avito ids 9 (Cars) and 84 (Phones) were web-verified. The rest of the id list is recalled from memory, and my own recall matches the analyst's list. That is the same model family, so it is not independent evidence.
- The analyst's "delivery AUC = 1.00" depends on a tiny margin. In the pooled C1 data, c4 Water transport is at 0.048 and c24 Business equipment at 0.061, so the boundary between the two is post hoc. The **hard zeros** carry the weight: 15 codes, each with tens of thousands to millions of events, at exactly 0.
- Within a vertical, codes could be permuted (c4/c5/c26, c34/c35, c40/c42) and the data would barely change.

**Conflict resolution (C1, val + test + 100K train feeds, 33.5M events).** The analysts gave three different "no delivery" lists.

| class | e5/e6/e8 share | codes |
|---|---|---|
| hard zero | 0 (c50: 3 events of 1.55M) | c2, c9, c10, c11, c12, c25, c26, c30, c31, c48, c49, c50, c51, c52, c53 |
| near zero | 0.07-0.2% | c27, c34, c35, c36, c38 |
| very low | 0.65-1.9% | c19, c37, c5, c47 |
| low | 4.8-6.1% | c4, c24 |
| goods | 8.1-29.4% | the other 27 codes |

- The sources analyst's 15-code list is right.
- The events analyst's "exactly 0" list wrongly includes c5, c19, c27 and c34-c38.
- The category analyst's binary split is right in direction but fragile at c4/c24.

### 3.1 Mapping table

Columns: val share (% of val candidates), val CTR, the C1 delivery share, and the share of candidates with f2 below the feed's mode (`categories/out/mapping.csv`).

| code | Avito id | name | vertical | val % | val CTR | deliv | f2 < mode | name conf. |
|---|---|---|---|---|---|---|---|---|
| c2 | 9 | Cars | Transport | 11.18 | .0153 | .000 | .48 | high (web-verified id) |
| c3 | 10 | Parts & accessories | Transport | 7.58 | .0161 | .171 | .56 | high |
| c4 | 11 | Water transport | Transport | 0.46 | .0176 | .048 | .58 | medium |
| c5 | 14 | Motorcycles | Transport | 1.47 | .0179 | .010 | .54 | medium |
| c6 | 19 | Repair & construction | Home | 5.74 | .0149 | .081 | .36 | medium-high |
| c7 | 20 | Furniture & interior | Home | 4.82 | .0162 | .093 | .33 | medium |
| c8 | 21 | Household appliances | Home | 2.04 | .0139 | .147 | .43 | medium |
| c9 | 23 | Rooms | Real estate | 0.46 | .0132 | .000 | .30 | medium |
| c10 | 24 | Apartments | Real estate | 4.08 | .0158 | .000 | .22 | high |
| c11 | 25 | Houses / dachas | Real estate | 3.09 | .0144 | .000 | .45 | medium |
| c12 | 26 | Land plots | Real estate | 0.98 | .0117 | .000 | .45 | medium |
| c13 | 27 | Clothing & shoes | Personal | 12.43 | .0197 | .219 | .49 | high |
| c14 | 28 | Watches & jewellery | Personal | 1.25 | .0195 | .161 | .50 | medium |
| c15 | 29 | Kids clothing | Personal | 3.57 | .0175 | .262 | .52 | medium-high |
| c16 | 30 | Kids goods & toys | Personal | 2.44 | .0161 | .197 | .45 | medium |
| c17 | 31 | Desktop PCs | Electronics | 0.23 | .0178 | .089 | .47 | medium |
| c18 | 32 | Audio & video | Electronics | 1.53 | .0147 | .180 | .47 | medium |
| c19 | 33 | Tickets & travel | Hobby | 0.17 | .0167 | .007 | .31 | medium |
| c20 | 34 | Bicycles | Hobby | 0.95 | .0113 | .142 | .45 | medium |
| c21 | 36 | Collectibles | Hobby | 2.27 | .0158 | .191 | .49 | medium |
| c22 | 38 | Musical instruments | Hobby | 0.30 | .0205 | .150 | .55 | medium |
| c23 | 39 | Sports & leisure | Hobby | 2.46 | .0140 | .204 | .48 | medium |
| c24 | 40 | Business equipment | Business | 0.47 | .0151 | .061 | .61 | medium |
| c25 | 42 | Commercial property | Real estate | 0.59 | .0095 | .000 | .30 | medium |
| c26 | 81 | Trucks & machinery | Transport | 1.14 | .0162 | .000 | .58 | medium |
| c27 | 82 | Food | Home | 1.49 | .0162 | .001 | .36 | medium |
| c28 | 83 | Books & magazines | Hobby | 0.83 | .0199 | .294 | .49 | medium |
| c29 | 84 | Phones | Electronics | 1.60 | .0183 | .131 | .46 | high (web-verified id) |
| c30 | 85 | Garages & parking | Real estate | 0.62 | .0098 | .000 | .23 | medium |
| c31 | 86 | Property abroad | Real estate | 0.07 | .0242 | .000 | .86 | high (f2 anchor) |
| c32 | 87 | Tableware & kitchen | Home | 1.53 | .0153 | .231 | .49 | medium |
| c33 | 88 | Beauty & health | Personal | 1.73 | .0131 | .284 | .44 | medium |
| c34-c38 | 89-93 | Dogs / Cats / Birds / Aquarium / Other animals | Animals | 2.29 | .012-.023 | ≤ .008 | .32-.49 | medium (vertical high) |
| c39 | 94 | Pet supplies | Animals | 0.96 | .0108 | .174 | .45 | high |
| c40 | 96 | Tablets & e-readers | Electronics | 0.26 | .0179 | .213 | .48 | medium |
| c41 | 97 | Games & consoles | Electronics | 0.59 | .0208 | .168 | .43 | medium |
| c42 | 98 | Laptops | Electronics | 0.28 | .0171 | .119 | .45 | medium |
| c43 | 99 | Office equipment | Electronics | 0.35 | .0127 | .226 | .49 | medium |
| c44 | 101 | PC components | Electronics | 0.91 | .0172 | .196 | .47 | medium |
| c45 | 102 | Hunting & fishing | Hobby | 0.73 | .0146 | .212 | .52 | medium |
| c46 | 105 | Photo equipment | Electronics | 0.23 | .0194 | .192 | .42 | medium |
| c47 | 106 | Plants | Home | 1.45 | .0211 | .019 | .35 | medium |
| c48 | 111 | Vacancies | Jobs | 2.69 | .0162 | .000 | .22 | high |
| c49 | 112 | CVs | Jobs | 0.28 | .0198 | .000 | .31 | medium-low |
| c50 | 114 | Services | Services | 8.95 | .0117 | .000 | .19 | high |
| c51 | 116 | Ready business | Business | 0.14 | .0148 | .000 | .45 | medium |
| c52 | > 116 | Hotels (Avito Travel) | Travel | 0.07 | .0065 | .000 | .62 | **medium-high** (analyst: high) |
| c53 | > 116 | Shift gigs (Avito Podrabotka) | Jobs | 0.27 | .0011 | .000 | .24 | **low-medium** (analyst: medium) |
| c1 / c0 | - | c1 unused; c0 = unknown (history type-2 only) | - | 0 | - | - | - | high |

Why two names are downgraded:
- **c52 Hotels.** The evidence is a single-day bulk posting date (2024-09-02, 55% of datable items) about a month before the public hotel-booking launch, plus ~14K items vs ~13K announced hotels. That is a strong coincidence but not an identity.
- **c53.** Only "a jobs-like category created in 2025 with no contact events" is robust. The product name is a guess.

**Production injection (CORR, high).**
- c50 Services appears in 81% of val feeds despite low affinity (f3 median 0.24) and has the lowest CTR among large codes.
- c53 is injected with CTR 0.0011.
- The model already learns both through the category embedding.

---

## 4. Features f1..f5 and the target

**Encoding (IS, high).** All five are quantile encoders fitted on train candidate rows, clipped to their outer knots. f2 = 0 and f5 = 0 are reserved codes. The floor/ceiling mass points carry meaning:
- f1 floor = no fine affinity;
- f3 floor = no category affinity;
- f5 at 1/62 = zero-rate item;
- f5 = 0 = no data.

PSI between val and test public/private is ≤ 0.005. Production train sits at 0.014-0.046.

| feature | meaning | IS / CORR | confidence (analyst → mine) | key evidence |
|---|---|---|---|---|
| f3 | user × **leaf category** (category_id code) affinity, real-time, reaching beyond the 256-event history | IS | high → high (with a correction) | Constant in 100% of (feed, category) groups. Pairs of different categories are never equal, so it is per leaf code, **not per vertical or "top-level category"** as the features analyst wrote. Changes between feeds minutes apart: +0.010 when same-category history grew, −0.0066 otherwise. Held-out R² 0.78 from visible history. |
| f1 | user × finer (sub-leaf) affinity nested in f3, with back-off | IS (nesting) / CORR (key) | high → **high** (nesting); **medium** (key = Avito microcategory) | P(f1 floor \| f3 floor) = 0.996. Varies within 70% of (feed, category) groups. Closer f1 values go with closer embeddings (cosine 0.745 → 0.629). |
| f2 | user × **item-region** affinity. Items carry a fixed cross-category region key K2; f2 is graded by the user's activity in that region; 0 = no activity there; the feed mode = home region | IS (key structure) / CORR (geography) | high → **high** (structure); **medium-high** (geography) | Cross-category pair equality is bimodal (99% of pairs always or never equal). Components have 0.2% purity violations. Home-component activity curves are shifted −1 to −5 h in Moscow time (eastern time zones). Property abroad sits below the mode 86% of the time. |
| f4 | listing age (posting time recoverable on a 128-s grid) | IS | high → high | Verified earlier (feat_life_, featrev_prep). Item-level eta² 0.98-0.995. |
| f5 | item-level, near-real-time, **unsmoothed** engagement rate (CTR-like). 0 = no data, 1/62 = zero events | CORR | high → **high** (item-level rate, not a count); **medium** (numerator = clicks) | Spearman with train CTR +0.37 (partial on impressions +0.39) vs impressions −0.16. Top decile = small-denominator noise. Value stays equal for the same item 86% of the time within 10 min and 2% at 6-24 h. |
| none of f1..f5 is a leaked score | - | IS (negative) | high → high | Per-feed Spearman with production position \|ρ\| ≤ 0.08, never < −0.9. Single-feature AUC matches between val and random-like train within 0.006. |
| user_id 0 | not-logged-in device session; f1..f3 stay personalised | CORR | high → high | eta² of f3 per (feed, category) = 1.000 for user 0. Floors are higher (f3 floor 17.7% vs 5.1%). Matches the event evidence (§1). |

**Home-region deficit (CORR, survives a confound check).** Within a feed, items in the home region (f2 = feed mode) are clicked less. Check C2, Mantel-Haenszel OR for home vs non-home:

| split | within feed | within feed × vertical | within feed × category |
|---|---|---|---|
| val | 0.757 | 0.804 | 0.823 (z −10.4) |
| train-explore (40K feeds) | 0.843 | 0.874 | 0.893 (z −7.5) |

- About a quarter of the raw effect is category composition: Services and Jobs are 70-73% home-region and have low CTR.
- The rest is real. A plausible mechanism (low confidence): non-local candidates pass a stricter retrieval bar.
- G3 is already calibrated on this flag (act/exp 0.981, `features/09_residual_g3.log`).

**Target design facts (IS, high).**
- Every train and val feed has ≥ 1 positive.
- Train impression lists end at the session end: the last shown item is positive in 22-29% of feeds, vs ~1.1% for its neighbours.
- These are logging properties. `pos` is already excluded at inference.

---

## 5. Cross-topic consistency checks

| # | Check | Verdict |
|---|---|---|
| X1 | Encodings: types and sources are frequency ranks; categories are not, and follow the Avito id order | Consistent across all three analysts |
| X2 | Target vs the event funnel: (1,s1) is the first event on 83-87% of positives in all three of the events analyst, the sources analyst and `eda_history_leak` (84.7%); the remainder are source-1 e4/e7 or source-less e2/e3/e5/e6 | Consistent. The surface-defined target is high confidence; the "contact-only" composition is medium (§1.2) |
| X3 | s3: "item card" (events) vs "favourites list" (sources) | **Conflict, resolved for favourites list.** Card-launched e4/e7 inherit the opening view's source (94.7% for s1); s3 is 61% e4; 50% of its views are re-views |
| X4 | Delivery-free category lists: events (~20 codes), sources (15), categories (23, AUC 1.00) | **Conflict, resolved by C1:** 15 hard zeros plus a graded near-zero tail; see §3 |
| X5 | e10 in user-0 feeds: 73.7% (sources) vs 86-89% (events) | **Denominators differ.** All user-0 feeds: 72-75%. Those with H > 0: 86-88% (C4). 15-16% of user-0 feeds have H = 0 |
| X6 | f3 "top-level category" (features) vs codes being Avito **leaf** categories (categories) | **Corrected:** f3 is per leaf code. No organiser feature carries a vertical-level (parent) affinity, so I tested one (§6). Null |
| X7 | f2 locality (categories: Services 0.73 / Jobs 0.70 at mode; Abroad 0.86 below) vs f2 = region (features) | Consistent and mutually reinforcing. The home deficit survives category control (C2) |
| X8 | "Near e7/e9 items → fewer clicks" (events) vs category CTRs (categories: Services lowest) | **Refuted as an intent effect.** Within (feed, category) the OR flips to 1.16 / 1.18 (C3) |
| X9 | user 0: login-only types absent (events), personalised f1..f3 (features), s3/s5/s12 rare or absent (sources) | Consistent: a device-level guest session, not a missing id. The project's "pooled bucket" rule for id-keyed statistics stays right |
| X10 | Source verticals vs categories: s7 = c10/c11/c12, s9 = c48/c2/c10, s14/s17 = c10/c11, s21 = c48 | Consistent with the realty / jobs mapping; supports both decodings |
| X11 | e4 = favourite ↔ s3 = favourites list ↔ highest favourite share in realty, cars and hotels (c10/c11/c52 0.29-0.31) | A coherent triangle. Both e4 and s3 are raised to medium-high / medium |
| X12 | Retention is set by type (events) and not by source (sources: hpos within type 1 is 73-91 for every source) | Consistent |
| X13 | Events §5, the leak refinement (source-aware later-history flag) | **Moot.** The later-feed leak is neutralised by the organisers (public 0.392819 = G3) and banned by the leak rule in CURRENT_STATUS. FN-PU was already closed by mechanism |

---

## 6. Residual screen of decoding-enabled features (pre-declared, run here)

**Rule, declared before running.** A feature counts as information G3 lacks only if the nested 2-fold-by-user delta of z(G3) + w·z_feed(x) is > +2 SE AND > +0.0003. This is the same harness as `features/09_residual_g3.py`. Data: val, own-feed history only, label-free. Log: `nn_x/codes/synth/logs/x02.log`. G3 = 0.40032.

| feature (needs the decoding) | nested Δ ± SE | G3 act/exp across quintiles | verdict |
|---|---|---|---|
| R1 v_share: history share in the candidate's vertical | −0.00009 ± 0.00025 | 0.92 → 1.03 | FAIL |
| R2 v_share_xleaf: same vertical, other leaf (invisible to f3) | −0.00003 ± 0.00020 | 0.98 → 1.00 | FAIL |
| R3 v_last: candidate vertical = vertical of the newest event | −0.00039 ± 0.00019 | 0.96 / 1.04 | FAIL |
| R4 v_rec: −log1p(min hpos in the candidate's vertical) | −0.00012 ± 0.00024 | **0.88** → 1.04 | FAIL |
| R5 deliv_x: user's delivery count × deliverable category | −0.00007 ± 0.00015 | 1.00 | FAIL |
| R6 deliv_mask: delivery-active user × hard-zero category | −0.00000 ± 0.00011 | 1.00 | FAIL |

Reading:
- G3 has a mild calibration gradient on vertical recency: candidates from a vertical the user has not touched recently are over-scored by ~12%.
- A per-feed linear blend cannot use that gradient without hurting other rows. It is the same information f3 / f1 / history attention already carry, at a finer grain.
- This repeats err_category_intent ("category intent saturated").

---

## 7. External sources
- Avito ML Cup 2025 public release: <public repository, link removed> (clickstream.pq, 68.8M rows; events.pq, 19 codes with is_contact). Used as a structural reference only.
- Avito category ids 9 = Автомобили and 84 = Телефоны are web-confirmed through scraper samples (https://apify.com/server0/avito-scraper.md, https://apify.com/stealth_mode/avito-product-search-scraper.md). The other ids (9 … 116) come from recall of Avito's public tree and are not verified on the web.
- Avito Travel hotels: about 13K hotels announced (snob.ru, 19 Apr 2024); booking launched 8 Oct 2024 (comnews.ru).
- Avito Podrabotka: test launch 13 Mar 2025 (eanews.ru); 2026 activity (retail.ru).
- Avito platform facts reused from `nn_x/audit/avito/findings.txt` (habr 1090584, 1086210, 974682, 1004694):
  - recommendations bring about half of views;
  - the main-page ranker's target covers post-impression actions, including favourites;
  - an exploration feed exists.
- Not found publicly: surface/source names, a list of delivery-excluded categories, Avito analytics event names. The family labels in §2 are inferred from data.
- No project data or user details were sent to any service.

---

## 8. Implications for the model

The ranker embeds every code as a categorical input and sees f1..f5 directly. Knowing a code's name therefore adds nothing on its own. Below are only the things that need the decoding. Each has an honest prior and a test that can be declared in advance. Everything else the analysts proposed is listed under "not decoding-enabled" and dropped.

### A. External hierarchy / grouping
- **A1. User × vertical affinity features** (vertical share, outside-leaf share, recency, last-event match): **tested here, 0/4 pass (§6). CLOSED.**
- **A2. Vertical-level parameter sharing for rare codes** (c31, c52, c53, c17, c19 each < 0.3% of rows), e.g. category embedding = vertical embedding + leaf residual. The same goes for source families (recommendation s1/s4/s6/s9; search s2/s5/s11/s12; saved s3/s10/s13; vertical s7/s14/s17/s21; unknown s0).
  - Prior P(≥ +0.0003) ≈ 0.03: rare codes are < 2% of rows, and the main families are well estimated.
  - Pre-declared test, only if a free 2-epoch slot appears: one ablation pair (seeds 23/24) under the standard 2-epoch rule (val > 2 SE pooled, holdout ≥ 0). Not worth GPU on its own. **Recommend: do not run.**

### B. Structural zeros
- **B1. Delivery-class masks and the guest flag** (is_guest = user 0 or has e10; hard-zero delivery category × user delivery activity): **tested as R5/R6, null.**
  - The NN already sees user 0 (`--user0`) and the event types. **CLOSED.**

### C. Cross-split / cross-feed knowledge
- **C1. Region key K2 propagated to all items.** f2-equality components are in `features/out/k2_item_comp.npy` (~60K frequent items). Labels can be propagated through (feed, f2-level) groups. Uses: region-normalised popularity, region-level co-visitation, local time.
  - Prior ≈ 0.05. G3 is calibrated on f2 classes (act/exp 0.98-1.03), and the home deficit is already absorbed. Calendar × category priors were measured null (+0.00022 ± 0.00012).
  - Pre-declared test (CPU): build `it_pop_region` = the item's train impressions-weighted CTR within its K2 component, taken as a share of the component's CTR (train only, leave-user-out). Run it through the x02 harness. PASS = nested > 2 SE and > +0.0003. Also check the buried-cell pair AUC > 0.52.
- **C2. s4 "similar-listings" item-to-item graph.** Edges = (previous view → s4 view) pairs in train histories, i.e. the clicks that Avito's own item-to-item recommender received. It is distinct from generic co-visitation, which mixes feed, search and favourites.
  - Score a candidate by its s4-edge weight from the user's last 32 views.
  - Prior ≈ 0.05: co-visitation is in the recipe, and source-specific affinities were null before. It is the one cross-feed object no current input encodes.
  - Pre-declared test (CPU): x02 harness on val, plus pair AUC in the "right category, buried" cell (7,624 positives). PASS = nested > 2 SE and > +0.0003, or buried-cell AUC ≥ 0.55. The PRETRAIN P8 encoder sided with beaters (0.31-0.33), so AUC > 0.5 in that cell would be new.

### D. Interpretability of known error cells and running experiments
- **D1. Pop-free family readout (free, do it).** Because f5 is an unsmoothed rate, a model without the 8 popularity inputs (abl_e10_xdrop_pop_s951-954, running) cannot tell a reliable high f5 from a noisy one.
  - Prediction: act/exp on "noisy-high f5" (top f5 decile with fewer than 5 train impressions plus history appearances) is < 0.95 for pop-free vs 0.974 for G3.
  - If confirmed, that explains part of the pop-free gap (−0.0023 val / −0.0074 holdout in the ablation screen).
  - It suggests the cheapest fix for that family: add a single exposure-count input (not the 8 popularity features). That keeps diversity while removing the failure. Decide with the existing decide9 rule; no new GPU unless the readout confirms.
- **D2. "Right category, buried" cell (37% of the loss).** The decoding says the beaters are older (f4), have higher engagement rates (f5) and are more often non-home-region (f2 below the mode). These are **production-ranker inputs**, so the cell is plausibly where exploration feeds contradict the production ranker's priors.
  - Pre-declared diagnostic (no LB claim): compute the buried-cell share by delivery class (§3), f2 class (home / other / zero) and vertical.
  - If any class holds > 1.5× its row share, that class is the target of any future item-discrimination work. Otherwise the cell is uniformly information-limited, which confirms the existing CLOSED status.
- **D3. Category CTR facts for error reading.** Services (c50) is injected into 81% of feeds at the lowest CTR; c53 at 0.0011; c52 at 0.0065. Read per-category errors as production injection, not model failure.

### E. Target design / sample weighting
- **E1. Graded relevance by event type:** not supported. Target rates per type run 0.74-0.90 and are flat; an earlier multi-task-by-type run was negative. **CLOSED.**
- **E2. Label repair or pseudo-labels from later snapshots** (source-aware): forbidden by the leak rule, and FN-PU is already closed by mechanism. **CLOSED.**
- **E3. End-of-session positives in train** (the last shown item is positive in 22-29% of feeds). The K9 aux uses production order. A candidate idea is to mask the last-position item from the K9 aux target only.
  - Prior ≈ 0.02: the main listwise loss is order-free, and val/test are shuffled.
  - Test only as a free rider on an existing 2-epoch pair. **Recommend: do not run.**

### Not decoding-enabled (dropped)
- Type-dependent recency, within-type hpos ranks, and channel-collapsed counts: the attention encoder sees (type, source, hpos) per event and can learn these.
- Signed "completed intent" inputs (e7/e9 similarity): refuted (C3).
- Same-category e3 count negative in the conditional logit: a likely jobs/services composition effect, untested within category. Not worth a slot.
- Source reweighting of the train-vs-eval view-share gap: the gap is not a source shift.
- Platform features: platform is not in the data.

---

## 9. Files
- New: `nn_x/codes/synth/x01_skeptic_checks.py` (C1-C4), `nn_x/codes/synth/x02_decoded_residuals.py` (R1-R6), `nn_x/codes/synth/logs/{x01.log, x02.log, c1_delivery_share.csv}`.
- Analyst mappings: `nn_x/codes/events/out/event_type_mapping.csv`, `nn_x/codes/sources/out/s11_source_mapping.csv`, `nn_x/codes/categories/out/{mapping.csv, code_vertical.json}`, `nn_x/codes/features/out/k2_item_comp.npy`.
