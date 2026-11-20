# EMBEDDINGS 11-13: what the item and user embeddings are, what they encode, and whether processing them helps

Synthesis and skeptic pass over three analyst studies plus the lead's 05:40 UTC measurement. Written 2026-11-13 05:45 UTC.

- `nn_x/emb/structure/`: what the vector encodes. Samples: 1.5M train candidate items (uniform, seed 0), 500K uniform train rows, the full val table (2,516,711 items, exact kNN on GPU N), all 14,423 val feeds, and 293K train feeds.
- `nn_x/emb/origin/`: where the vector comes from. Gates T1-T7 were pre-declared in `PREDECLARE.txt` at 05:22:18 UTC. It used public-model controls on GPU N.
- `nn_x/emb/usability/`: does any pre- or post-processing help G3? Five tests were pre-declared in `PREDECLARED.txt` at 05:20:55 UTC, with Addendum A at 05:27:46. CPU only. The method deviation is recorded in `DEVIATIONS.txt`.

Nothing was written under data/, preds/, subs/ or nn/, and no submissions were made. I spot-checked the key gate numbers (proj_mu, cos_u and ZCA, nested and holdout) against `usability/out/eval.log` and `holdout.log`, and they match.

Confidence words: **high** means several independent measurements agree. **medium** means a single line of evidence, or inference from a proxy. **low** means suggestive only.

---

## 0. TL;DR

1. **The vectors are not from a public text or vision model, and they are not a Matryoshka (MRL) prefix.** They are native 128-d float unit vectors, multiplied by 256, rounded and stored as int8. They come from Avito's own two-tower recommender (item tower), and `user_emb` is the matching user tower in the same space. Subjective probabilities: in-house two-tower 0.94, appended attribute block 0.03, public text model as the direct source 0.02, vision 0.01.
2. **No public embedding model can be used in pre- or post-processing.** The data has no titles, descriptions, images or search queries. The only text we have is 52 decoded category names, and the vector already encodes category at 98.3% (linear probe). Category geometry follows co-browsing behaviour, not the meaning of the names.
3. **What the vector encodes:** category (linear probe 0.983); a strong additive location/region component, which is new as a vector-level finding but overlaps f2; a large share of f2; little popularity or age; and **no item CTR** (R^2 0.003).
4. **Items are effectively static across splits.** 77-83% of vectors are bit-identical, depending on how you weight. Mean cos is 0.999 and only about 1.5-1.8% drift below cos 0.99. Drift carries no click signal (lift 1.05 [0.94, 1.17]).
5. **Every processing idea is null on G3:** vector change, centring, whitening, ABTT, mean-direction projection, acted vs viewed near-duplicates, and cos_u interactions. 0 of 5 primaries passed the pre-declared bar. The one near-miss (proj_mu, +0.00045 at 2.05 SE) gave -0.00004 on the train holdout.
6. **No embedding-derived feature deserves a 2-epoch NN screen on current evidence.** The only untested idea with any rationale is a location-subspace match feature. Its prior is about 0.05, and it should get a CPU nested gate first, not an NN slot (§5.2).

---

## 1. What the item and user embeddings are

### 1.1 Origin verdict

| Hypothesis | P (subjective) | Why |
|---|---|---|
| (a) Avito in-house two-tower item tower (structured content + location + seller-level inputs, learned 128-d output) | **0.94** | The quantisation and spectrum fit a learned native-128 output. Location is additive and spread over non-contiguous dims. Bit-identical twins come from one seller's batch uploads. `user_emb` lives in the same space with the same quantisation. Category geometry follows behaviour, not name meaning. The vectors are re-inferred over time. All of this matches Avito's published design (habr 1004694, via `nn_x/audit/avito/findings.txt`). |
| (d) public or other vector with an appended attribute block | 0.03 | T4 FAIL. No aligned or contiguous block is separately normalised or attribute-specific. Adjacent-dim \|r\| is 0.052 against 0.051 for all pairs. |
| (b) public text model as the direct source (MRL or not) | 0.02 | Not MRL (§1.2). A title-only model would give many bit-identical vectors, but only 0.21% of train rows have one. **Not excludable:** a frozen public text encoder could sit *inside* Avito's tower. Nothing in the data could identify it, and it would not change anything we can do. |
| (c) vision or multimodal | 0.01 | No missing-modality zeros. The published item tower has no image input. |

**Skeptic downgrades.**
- The probabilities are subjective weights, not a fitted posterior.
- The analysts said the drift and twins show "price-like dynamic inputs" and "f5 is not an input". Neither was measured directly. Identical twins with different current f5 or age only show that the cached vector does not track *current* f5 or age. A vector computed once at posting and cached explains the same pattern. Kept as: "no item-ID input; the vector is a cache refreshed occasionally, not a function of current f4/f5".
- Location is a **post-hoc** test, and its region proxy (K2 components from f2 ties, with time-zone evidence) is itself medium confidence. The analyst rated this claim high; the region-is-geography part is downgraded to **medium**. That location is an additive, category-independent, user-stable component is still **high**: it was reproduced three ways (K2 probe, feed-level subspace, cross-split same-user stability).
- The guess that the dim-6 and dim-96 modes (23.3% and 9.0% of items) are Moscow and St Petersburg is speculation and should not be relied on.

### 1.2 The Matryoshka question: answered "no" with measurements

| Measurement | Dataset | MRL controls (Qwen3-Emb-0.6B / deepvk USER2-base, cut to 128) | Non-MRL controls (e5-small, USER-base, LaBSE, arctic-l-v2, mxbai) |
|---|---|---|---|
| Spearman(per-dim sd, dim index) | **-0.06** (val table, all rows: -0.07; lead: -0.04) | -0.52 / -0.83 | -0.12 .. +0.04 |
| sd ratio, first 16 / last 16 dims | **1.005** | 1.26 / 1.64 | about 1 |
| top-10 neighbour overlap, prefix minus random dims at k=32 | **-0.002** (sd of random draws 0.015) | +0.041 / +0.042 | -0.002 .. +0.021 |
| prefix-64 minus suffix-64 overlap | **+0.002** (0.639 vs 0.637) | +0.047 / +0.058 | n/a |
| category probe, first k vs random k dims (k = 16/32/64) | 0.576/0.819/0.949 vs 0.576/0.784/0.948 | n/a | n/a |

- The sd-by-index profile is the decisive evidence. Pre-declared power check T6 was **not met**: even true MRL models gain only about 0.04 from the prefix on this corpus, so the neighbour-overlap test is weak.
- Caveat (no effect on the verdict for practical use): an MRL vector followed by a random rotation would also look flat.
- Either way, there is no larger parent vector to recover, and re-truncating is pointless. 128 dims hold a participation-ratio rank of about 64–66, and the top 64 PCs carry 77% of the variance.

### 1.3 Quantisation fingerprint (T1 PASS, high)

- Each vector is `round(256 * unit_vector)`, with one global scale. This is neither sentence-transformers' per-dim calibrated int8 nor per-vector absmax.
- Mean ||v||^2: 65546.48 (train) / 65546.70 (val) / 65546.61 (test). The prediction is 65536 + 128/12 = 65546.67; a scale of 255 would give 65035.7.
- Norm sd is 0.289, against sqrt(1/12) = 0.2887. Only 0.012-0.014% of rows hit 127. Values run from -116 to 127, with 2.10% exact zeros.
- The zero excess is 1.003, so no dims are zeroed out and no missing-modality blocks exist.
- Rounding moves cos by only about 1.6e-4, so **de-quantising gains nothing**.

### 1.4 user_emb

- Same space and same quantisation (norm about 252-256). cos(mean user, mean item) is 0.73-0.74, and the user mean is the same in train and val (0.9998).
- User variance concentrates in the item principal subspace: 25% of it lies in the top-16 item PCs, against 12.5% for a random direction.
- 62% of feeds with no history still have a valid user vector. This fits the published design, where the user tower also reads search queries; the queries themselves are not in our data.

---

## 2. What they encode

### 2.1 Probes (item level, 5-fold, 1.5M train candidate items; high)

| Target | ridge(emb) R^2 | category only | partial R^2 (emb given cat) | best nonlinear (MLP) |
|---|---|---|---|---|
| category (52 classes, majority 17.6%) | logistic accuracy **0.983**, top-3 0.999 | n/a | n/a | kNN-1 0.970 (val table 0.9825) |
| f2 (user × item-region affinity, mean per item) | 0.387 | 0.066 | **0.346** | 0.473 |
| f1 | 0.318 | 0.249 | 0.107 | 0.342 |
| f3 | 0.337 | 0.287 | 0.099 | 0.374 |
| f5 | 0.112 | 0.045 | 0.078 | 0.137 |
| f4 (age) / log age | 0.082 / 0.071 | 0.043 / 0.036 | 0.049 / 0.045 | 0.141 / 0.128 |
| log train impressions | 0.140 | 0.084 | 0.072 | 0.174 |
| smoothed train CTR | **0.0034** | 0.0022 | about 0 | 0.0041 |

- **Click propensity on unbiased val traffic** (user-grouped 5-fold logistic, within-feed AUC): category 0.535; emb 0.548; emb+cat 0.553; f1..f5+cat 0.624; f1..f5+cat+emb 0.630. cos(user, item) gives 0.643. The item vector carries a small item-level click prior beyond category, far below the user-relative signals.
- **k-means clusters** (cl1024/cl4096) are argmax-cosine to their centroids, reproduced on 99.9999% of items. They add **no information** beyond the raw vector. This agrees with the P1 cluster closure.
- Category is spread over many dims (per-dim eta^2 median 0.19-0.21; each 32-dim block alone scores 0.78-0.82). A few dims act as flags: dim 4 for category 48 (r^2 0.40), dim 42 for categories 15/16. They carry nothing beyond category_id.

### 2.2 Location-like component (new as a vector-level fact; structure high, geography medium)

- After removing category means, 24.3% of the variance sits between feeds. Leading dims: 96, 38, 127, 103, 71, 0 (between-feed share 0.47-0.63).
- The same user's val feed and last train feed agree at cos 0.957 in the top-16 feed-level subspace, against 0.014 for a random user.
- The K2 region proxy can be read off the vector within a category at accuracy 0.993-0.9995 (majority 0.35-0.65; permuted labels 0.33-0.61). Transfer to another category scores 0.973 (majority 0.288), and the same region's offset in different categories agrees at cos 0.83.
- Region explains 28-47% of within-category variance. Per-dim region eta^2 reaches 0.91 and is not contiguous (Spearman with dim index -0.08).
- Mode-match candidates (dim 6, dim 96) have f2 higher by +0.31..+0.43 and click rate 1.55-1.62% against 1.32-1.38%.
- **Ranking value is small and mostly already carried by f2.** Within-feed AUC: feed-level subspace cos 0.536-0.551; f2 0.494; subspace within (feed, f2-level) 0.514.
- CODES_DECODING_1113 already decoded f2 as user × region affinity and found the home-region deficit G3-calibrated (actual/expected 0.981).

### 2.3 Special dims and anisotropy (high)

- **Special dims.** Dims 0, 6, 43, 77 and 83 are right-skewed (skew 1.1-1.7), category-free (eta^2 ≤ 0.012) and the least predictable from the other dims (R^2 0.19-0.32). Only dim 6 is truly bimodal; dim 96 has a minor bump. These are the location dims of §2.2, not attribute slots. No attribute sits on a single dim: the best single-dim |rho| is 0.33 (f3 → dim 33).
- **Common direction.** The mean unit vector has norm 0.552 (users 0.646), and random-pair cos is 0.305 (−0.00006 after centring).
  - The projection on the mean direction is category-modulated (eta^2 0.40).
  - It is unrelated to popularity, CTR or age (|rho| ≤ 0.04; ≤ 0.13 within category for f3).
  - It is not PC1 (|cos| 0.12).
- **Spectrum.** PC1 6.1-6.5% of variance; top-32 PCs 54%; participation ratio 64-66. The global geometry does **not** separate the origin hypotheses: public models give similar numbers.
- **Metric.** Raw cosine is the native metric.
  - Centring keeps 92% of top-10 neighbours; whitening keeps only 70%.
  - Whitening hurts user scoring by **-0.0440 ± 0.0018** NDCG and rec5 history scoring by -0.0050 ± 0.0011.
  - Centring hurts user scoring by -0.0046 ± 0.0007 and is neutral for history.

### 2.4 Local geometry and near-duplicates (high)

- Same-category pairs average cos 0.439; cross-category pairs 0.295. Within-category tightness ranges from 0.36 to 0.76.
- Category-centroid geometry follows co-browsing behaviour. Spearman with history transition lift is 0.69-0.71. Spearman with public-model cosines of the category names is only 0.13-0.48, and the partial correlation given behaviour is about 0.
- **Near-duplicates are common** (val table, exact kNN):
  - NN-1 cos median is 0.968.
  - 33.2% of items have a neighbour above 0.98, 15.0% above 0.99 and 0.43% above 0.999.
  - Exact twins: 0.035% of val items and 0.210% of train rows (max group 52).
  - Pairs above 0.98 share a category 99.86% of the time and never share an item_id. The median posting gap is 1,287 h. They look like similar listings or re-posts.
  - Exact twins are one seller's batch uploads: 25.4% share a 128-s posting slot, against 0.014% for random same-category pairs.
- **Within one feed:**
  - 55.2% of val feeds hold a candidate pair above 0.98, covering 7.2% of candidate rows, which click at 1.40% against a 1.58% base.
  - The co-click rate is 4.9% for near-duplicate pairs against 1.6% for same-category pairs.
  - **Skeptic note:** this is *already exploited*. `ctx_nd_max`, the `ctx_nd_knn_*` features (kNN-smoothed scores over in-feed candidate neighbours) and `ctx_nd_top5*` are G3/ranker inputs, and `ctx_nd` is load-bearing in the ablation (+0.0014 / +0.0003). The near-duplicate *count* features (`ctx_nd_c80/c90/c95`, `dupb_*`) were tested in feat_ctx_ and gave nothing. The NN also has set attention across candidates (+0.0020 worth). The analyst's suggestion of "in-feed duplicate-cluster features / score propagation" is therefore not new.

### 2.5 User tower vs history (high)

| Score (14,095 val feeds with user and history) | pooled AUC | within-feed AUC | NDCG (random 0.2888) |
|---|---|---|---|
| cos(user_emb, cand) | 0.637 | 0.643 | 0.3441 |
| mean of all history | 0.614 | 0.607 | 0.3302 |
| last-20 history mean | 0.629 | 0.627 | 0.3455 |
| rec5 (recency-weighted, half-life 5) | 0.631 | 0.633 | **0.3503** |
| transformer prediction of user_emb from history (val cos 0.880) | 0.634 | 0.636 | 0.3476 |
| user residual (u minus its history prediction) | 0.474 | 0.493 | 0.2904 |
| z(user) + z(rec5) | n/a | n/a | 0.3533 (+0.0022 ± 0.0008 over z(rec5)+z(transformer prediction)) |

- user_emb is mostly a function of history: a ridge map gives R^2 0.594, and a 2-layer transformer reaches cos 0.880.
- It adds a small, significant amount on top of history in an *unfitted* two-score fusion, and it is the strongest single cue at H=0 (within-feed AUC 0.693).
- Its residual alone is useless (0.493). This matches the P2 closure (||r|| × cos_u, val -0.00034).
- Candidate sets are history-driven. Candidate-centroid R^2 is 0.772 from history, 0.788 from history+user and 0.661 from the user vector alone, so the user tower is not the sole retrieval query.
- **G3 already absorbs cos_u** (§4, item 5). The "+0.0022" is not an available gain.

---

## 3. Cross-period changes of the same item

| Study (weighting) | Bit-identical | Other numbers |
|---|---|---|
| lead (200K train-sample items present in val) | 78.6% | cos > 0.99 98.9%, min 0.63 |
| structure (60,459 train-sample items in val) | 80.7% | mean cos 0.99906; 1.54% < 0.99; 0.44% < 0.95 |
| origin (all 2,030,667 val items present in train) | 82.7% | changed median cos 0.99964; 2.3% of changed < 0.95; min 0.275 |
| usability (879,569 distinct val-candidate items) | 77.9% | 10.0% at quantisation level (≤ 1 step); 10.4% small (cos ≥ 0.99); 1.73% < 0.99 |
| usability (val candidate *rows*) | 77.1% | 1.79% < 0.99; mean cos 0.99885 |

The spread from 77% to 83% comes from the weighting (rows vs items, candidates vs all), not from disagreement between studies.

**Structure of the change** (medium):
- Re-inference with the same or near-same inputs dominates: ±1 int8 flips across about 13-47 dims.
- About 1.7% are real edits, and category changes in 2.78% of those, against 0.15% of unchanged items.
- Young listings change more often (26-30% vs 13-18%), and items shown more in train are re-encoded more often.
- Same-seller twins co-change far above independence: both changed 6.2% vs 0.94% expected.
- T3 FAIL: drift is not low-rank (top-5 PCA share 0.218 against 0.178 for neighbour differences). The mean drift direction is weak (norm 0.076).

**Is the change a usable signal? No (high).**
- Click lift of changed vs unchanged is 1.048 [0.937, 1.172]; for cos < 0.9 it is 1.161 [0.873, 1.544].
- chg_d AUC is 0.508, and its nested residual on G3 is -0.00024 ± 0.00012.
- Drifted *rows* click more (0.018 vs 0.017), but that follows popularity (it_hist_all_luo 2.33 vs 1.99), not the drift itself.
- Public and private test look the same: 62.6% / 59.2% have a train vector, and 2.95% / 2.92% of those changed (cos < 0.99). There is no shift risk.

**Practical consequence (high):** joining item-level statistics across splits by item_id, and pre-training on train-table vectors, are safe.

---

## 4. Pre- and post-processing tests on G3

- Base zG3 is the within-feed z of `preds/ens_G3_final_val.npy`: NDCG 0.40032 on all val, 0.40046 on the 14,306 leak-free feeds.
- **Bar (pre-declared):** nested residual ≥ +0.0005 **and** ≥ 2 SE (secondaries need 2.64 SE).
- Nested means w is picked from {0, ±.02, ±.05, ±.1, ±.15, ±.2, ±.3, ±.5} on 4/5 of user groups (user 0 kept as singletons), using 2 GroupKFold seeds.
- The holdout is the 30K-feed train holdout (G3 = z-mean of 20 members, NDCG 0.32379).
- "GBM" is the corrected version, boosted from k·zG3. Its control itself scores **-0.00240 ± 0.00050 vs base**, so a positive GBM delta never means a gain over G3. The original from-scratch GBM was void because of tie artefacts (control -0.014).

| # | Idea | AUC | Nested Δ ± SE (val) | Cell-wise (M×H) | GBM (with − control) | Holdout nested | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | vector changed train→val, chg_d = 1 − cos (P) | 0.508 | -0.00024 ± 0.00012 | -0.00020 ± 0.00029 | +0.00013 ± 0.00062 | n/c | **CLOSED** |
| 1 | chg_flag (S) | 0.500 | -0.00002 ± 0.00009 | -0.00055 ± 0.00016 | +0.00122 ± 0.00049 (2.49 SE < 2.64; still about -0.0012 vs G3) | n/c | CLOSED |
| 1 | in_train (S) | 0.529 (0.508 within category × popularity bin) | +0.00024 ± 0.00028 | +0.00023 ± 0.00033 | n/c | n/c | CLOSED (popularity proxy) |
| 2 | ZCA-whitened max cos to history (P) | 0.651 (raw 0.653) | -0.00002 ± 0.00021 | -0.00022 ± 0.00032 | +0.00064 ± 0.00059 | 0.00000 (w=0 in 10/10 folds) | **CLOSED** |
| 2 | centred / ABTT3 max cos (S) | 0.650 / 0.645 | -0.00019 ± 0.00017 / -0.00034 ± 0.00020 | n/a | n/a | n/a | CLOSED |
| 2 | centred / ABTT3 / ZCA cos to newest 5 events (S) | 0.634 / 0.628 / 0.630 | -0.00013 / -0.00006 / -0.00028 ± 0.00013 | n/a | n/a | n/a | CLOSED |
| 2 | within-category whitened max (P3, read only) | n/a | -0.00031 ± 0.00017 | n/a | n/a | n/a | CLOSED (already) |
| 3 | projection on the mean (anisotropy) direction, proj_mu (P) | 0.500 | **+0.00045 ± 0.00022** (w=+0.02 in 10/10 folds) | +0.00007 ± 0.00037 | +0.00114 ± 0.00062 | **-0.00004 ± 0.00003**; fixed w=+0.02: -0.00017 ± 0.00012 | **CLOSED** (below bar; not replicated) |
| 3 | projection on PC1 (S) | 0.509 | -0.00037 ± 0.00021 | n/a | n/a | -0.00005 ± 0.00004 | CLOSED |
| 4 | candidate is a near-duplicate (cos > 0.98) of an *acted* history item, nd_act (P) | 0.509 | +0.00003 ± 0.00016 | -0.00023 ± 0.00024 | +0.00047 ± 0.00040 | 0.00000 | **CLOSED** |
| 4 | nd_view_only (S); max cos to acted items (S) | n/a; 0.608 | -0.00009 ± 0.00009; -0.00004 ± 0.00018 | -0.00097 ± 0.00026; n/c | n/c; +0.00085 ± 0.00059 | n/c | CLOSED |
| 5 | cos_u (two-tower score) with M/H/user-0 interactions (P = GBM) | 0.648 | -0.00015 ± 0.00022 (linear) | -0.00043 ± 0.00044 | +0.00073 ± 0.00058 (1.26 SE) | -0.00007 ± 0.00006 | **CLOSED** (G3 has absorbed it) |

Key descriptive facts behind the nulls:
- **Near-duplicates.** They click 2.4× the base rate, but acted vs viewed-only does not matter (ratio 0.96 [0.87, 1.06]).
  - Dose response of max cos to acted items: (0.9, 0.95] 1.54×, above 0.995 3.43×. This is the `seq_max_all` / `maxsim32` signal already in G3.
  - The usability study's raw max cos equals `seq_max_all` exactly (Spearman 1.0).
- **Processing as a lever.** Every processing idea is a fixed linear transform or lookup of vectors the NN already reads through retrieval keys, sim_bias, the history encoder and candidate tokens, so its first layer can learn any of them.

---

## 5. Implications

### 5.1 Can an external embedding model be used anywhere in pre- or post-processing?

**No, not usefully.**

- **Nothing to encode.** The competition data contains no item titles, descriptions, attributes, images or search queries. A public text or vision model can only embed text we actually have. That means the 52 decoded category names, medium confidence, from CODES_DECODING_1113 (plus event and source codes if they ever get names).
  - Test T7 shows the item vector already encodes category at 98.3%.
  - Category geometry follows behaviour (Spearman 0.69-0.71), not name meaning (partial correlation about 0 given behaviour).
  - A name-embedding prior therefore has an expected gain of about 0.
- **Nothing to download or match.** The vectors come from Avito's unpublished in-house two-tower model. Even an exact model match would leave nothing to feed it.
- **No MRL parent.** MRL is ruled out, so there is no parent vector to recover and no reason to re-truncate.
- **Rounding noise is negligible** (cos noise about 1.6e-4), so de-quantising gains nothing.
- **Fixed transforms are all null on G3:** centring, whitening, ABTT, mean-direction projection and drift flags (§4).
- **Related, already closed:** an LLM reading a text rendering of feeds scored 0.3957 vs G3 0.4433 (nn_x/llm_manual) and was closed.

The only embedding *models* that make sense here are ones we **train** on these vectors: sequence or two-tower models over history, as the ranker already does. The pre-training track (P1-P4, E1/E4/F8) is closed.

### 5.2 Does any embedding-derived feature deserve a pre-declared 2-epoch NN screen?

**Not on current evidence.** These are the candidates, re-graded by the skeptic:

| Candidate | Why it might help | Why it probably won't | Honest prior of passing a 2-epoch screen (≥ +0.0005, ≥ 2 SE, holdout same sign) | Recommendation |
|---|---|---|---|---|
| **Location-subspace match**: cos between candidate and history in the category-residualised, feed-level 16-d subspace (refit label-free per split), plus dim-6 and dim-96 mode-match flags and feed "location purity" | New as a vector-level fact. It is user-stable across splits and carries a little signal beyond f2 level (within-feed AUC 0.514) | f2 already encodes user × region affinity. The home-region effect is G3-calibrated (0.981). The CODES R1-R6 region and vertical residuals were all null. The NN sees the raw vectors and can learn a 16-d projection. The 112-d complement (0.620) does not beat raw cos (0.625) | **~0.05** | **CPU nested G3 residual first** (the usability harness: bar ≥ +0.0005 and ≥ 2 SE, holdout same sign; pre-declare with `date -u`). Give it an NN slot only if it PASSES. |
| In-feed near-duplicate cluster features or score propagation | Co-click 4.9% vs 1.6% | Already covered by `ctx_nd_max`, `ctx_nd_knn_*` (score smoothing over in-feed neighbours), set attention, and the count features that were null in feat_ctx_ | ~0.02 | Do not screen. At most a free CPU check of zG3 + w·mean(zG3 of in-feed cos > 0.98 neighbours) as post-processing, prior ~0.03. |
| proj_mu (mean-direction projection) | +0.00045 val | Holdout -0.00004; AUC 0.500 | ~0.02 | Closed |
| user_emb beyond history | +0.0022 in an unfitted fusion; H=0 AUC 0.693 | Already in G3 (cos_u residual null on val and holdout); the P2 residual was null | ~0.02 | Closed |
| Drift, whitening, near-duplicate act/view, cluster ids | n/a | All null (§4, P1, P3) | ≤ 0.02 | Closed |

The **embeddings direction is CLOSED as a lever.** It stays open only through the single CPU pre-gate on the location-subspace match. This is consistent with the RECAP_1113 prior of ~0.02-0.03 for the embeddings study.

### 5.3 Facts worth keeping (documentation, ADOPTED)

- Use raw cosine and dot for user and history scoring. Never whiten. Do not centre the user dot.
- Item vectors are static enough to join across splits by item_id.
- Cluster ids are redundant with the vector (they are only a convenience for the ranker).
- Popularity, CTR and age are barely encoded, so popularity and age features stay complementary.
- `user_emb` must stay a separate input (no distillation from history: that loses about 0.002-0.003 in the unfitted fusion).

---

## 6. Registry one-liners

**reports/COVERAGE_MAP.md** (table rows):

```
| embedding origin / fingerprint (public text or vision model? Matryoshka? appended block?) | ANSWERED 11-13: Avito in-house two-tower item tower (P 0.94); not MRL (sd-vs-index rho -0.06 vs MRL -0.52/-0.83); round(256·unit) int8; no item text/images in data -> no external embedding model usable | T1/T2 PASS, T3/T4 FAIL, T6 power not met (sd profile carries it); location is an additive learned component (K2 probe 0.993-0.9995 within category, 0.973 cross-category) | reports/EMBEDDINGS_1113.md; nn_x/emb/origin |
| embedding pre/post-processing (drift flag, centred/whitened/ABTT cos, mean-direction proj, acted-vs-viewed near-duplicate, cos_u interactions) | CLOSED 11-13 | 0/5 primaries pass; best proj_mu val +0.00045 ± 0.00022 -> holdout -0.00004; ZCA -0.00002; drift -0.00024; nd_act +0.00003; cos_u GBM +0.00073 ± 0.00058 | reports/EMBEDDINGS_1113.md §4; nn_x/emb/usability/out/eval.log, holdout.log |
| embedding location-subspace match (category-residualised feed-level 16-d cos; dim-6/96 mode match) | OPEN, prior ~0.05: CPU nested G3 pre-gate only, NN screen only if it passes | within-feed AUC 0.536-0.551, 0.514 within f2 level; overlaps f2 (region affinity) | reports/EMBEDDINGS_1113.md §5.2; nn_x/emb/structure/s08-s10 |
```

**reports/EXPERIMENT_REGISTRY.md** (log line):

```
- 2026-11-13 | EMBEDDINGS_1113 (3 analysts + skeptic synthesis; GPU N for exact val kNN and public-model controls only) | item emb = Avito two-tower item tower (P 0.94), round(256·unit) int8, not MRL (sd-vs-index rho -0.06 vs MRL -0.52/-0.83), no appended block; user_emb = matching user tower; encodes category (probe 0.983), additive location (K2 0.993+; overlaps f2), f2 partial R^2 0.35, CTR ~0; 77-83% bit-identical across splits, drift not predictive (lift 1.05); near-duplicates common but already in G3 (seq_max_all, ctx_nd). Pre-declared G3 residuals 0/5 PASS (proj_mu +0.00045 val -> -0.00004 holdout). External embedding model: not usable (no item text/images). Open: location-subspace CPU pre-gate, prior ~0.05 | reports/EMBEDDINGS_1113.md, nn_x/emb/{structure,origin,usability}/
```
