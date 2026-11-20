# Feature family `seq_`: history sequence structure and intent

Code: `<repo>/feats_v2/seq_/`. Final features: `<repo>/data/feats/{train,val,test}/seq_*.npy`
(26 files per split, float32, aligned with `cand_*.npy`). Work files (transition matrices, ridge weights) are in
`<repo>/data/seq_work/`. The KEEP list is also in `feats_v2/seq_/KEEP.txt`.

## TL;DR

* **KEEP: 26 features. Harness `--family seq_` on the final files (3 seeds, val 5-fold GroupKFold, base ~118 + tte_*):
  +0.0080 ± 0.0009** (base 0.3634 -> 0.3714; per seed +0.0072 / +0.0075 / +0.0093). An identical earlier run of the same
  26 features (`seq_k1_s3`) gave +0.0075 ± 0.0009, so run-to-run LightGBM nondeterminism is about ±0.0005. The whole explored
  family (88 features) gives only +0.0063 ± 0.0009, so the pruning is worth about +0.001.
* **Where the gain comes from:**
  1. **Embedding x sequence structure (group E): +0.0034 alone (1 seed).** The useful signals are max-type similarities over
     different parts of the history: the whole history (`seq_max_all`; base `maxsim32` only sees the last 32 events), the
     last 8 events (`seq_max_new8`, the **best single feature found: 0.3511** vs maxsim32 0.349 and dot_u 0.3435),
     old events (hpos >= 32), type-1 clicks from event_source 1, non-type-1 events, and source-specific decayed profiles.
  2. **Item co-visitation from de-duplicated train histories (group CV): +0.0016 alone; +0.0017 on top of the rest.**
     It is the only collaborative signal in the family. Its coverage falls with the train-to-feed gap. A gap simulation on val
     (train source cut 3 days earlier, a harsher gap than private) halves its gain to +0.0009, so expect about +0.001 on private.
* **Did not help on top of the base (dropped):** category-sequence structure (runs, momentum, entropy, decayed shares;
  +0.0002); learned category / embedding-cluster Markov transition models (-0.0004); dense item->cluster co-visitation
  (-0.0004); a ridge "next-click embedding" predictor (0.3492 single, no LGB gain); explicit contrast features
  (recent vs long-term match; -0.001 vs K1). Most are strong single features (0.31-0.35) but redundant with the base
  embedding and category features.
* **Drift:** every KEEP feature has PSI < 0.006 (val vs test-public / test-private), except the within-feed ranks of the
  co-vis score: PSI 0.019 / 0.037, from co-vis coverage 11.3% (val) -> 10.2% (public) -> 9.8% (private).

## 1. Feature definitions (KEEP set)

History events are ordered newest first (`hpos` 0 = newest). Decay weights are `0.5^(hpos/hl)`. Cosines use L2-normalised
int8 embeddings, and history events without an embedding are skipped. `type 1` (t1) = event_type 1, the type a feed click turns
into in the next history (eda/history_leak: of positives found in the next history, most are type 1). "deep" = event types 5, 6, 8, 9.
NaN means undefined (no history or no candidate embedding).

### E. Embedding x sequence (`feat_hist.py::_emb_kernel`; per feed, no learning, identical for all splits)
| feature | definition | single NDCG (+dir / -dir) |
|---|---|---|
| seq_max_new8 | max cos over the newest 8 events | **0.3511** / 0.2539 |
| seq_dcos_hl2 | cos(candidate, decayed mean of history vectors, hl 2) | 0.3493 / 0.2550 |
| seq_dmax_hl10 | max_j max(cos_j,0) * 0.5^(hpos_j/10) (recency-decayed max) | 0.3482 / 0.2542 |
| seq_dcos_t1_hl5 | as dcos, type-1 events only, hl 5 | 0.3468 / 0.2551 |
| seq_dcos_s1 | as dcos, type-1 events from event_source 1, hl 5 | 0.3444 / 0.2566 |
| seq_max_all | max cos over the WHOLE history (up to 256 events) | 0.3436 / 0.2449 |
| seq_top3 | mean of the 3 largest cosines (whole history) | 0.3425 / 0.2450 |
| seq_max_t1 | max cos over type-1 events | 0.3425 / 0.2468 |
| seq_max_s1 | max cos over type-1 events from source 1 | 0.3411 / 0.2506 |
| seq_cos_last1 | cos to the newest history item | 0.3398 / 0.2617 |
| seq_max_nt | max cos over non-type-1 events | 0.3273 / 0.2580 |
| seq_dcos_s2 | as dcos, events from source 2, hl 5 | 0.3269 / 0.2672 |
| seq_dcos_nt_hl10 | as dcos, non-type-1 events, hl 10 | 0.3266 / 0.2637 |
| seq_max_old | max cos over events with hpos >= 32 (long-term interest) | 0.3262 / 0.2558 |
| seq_dcos_rep | as dcos over items with >= 2 events in the history, weighted by event count, hl 30 (strong-interest items) | 0.3221 / 0.2638 |
| seq_dcos_deep | as dcos over deep events (5,6,8,9), hl 30 | 0.3071 / 0.2730 |
| seq_argmax_hpos | hpos of the most similar history event (how recent the best match is; lower = better) | 0.2746 / **0.3123** |
| seq_max2_gap | largest minus 2nd-largest cosine (is the match unique) | 0.2885 / 0.2877 |

### CV. Item co-visitation (`feat_covis.py`, OTTO style)
Source sequences are the latest train feed per non-zero user plus every user-0 feed (2.58M sequences, 456M events). For
**val/test rows** the source also includes every val/test feed history with timestamp < the own feed's timestamp. This is
causal, so the definition is identical for val and test and there is no future window to censor. Sequences of the same
non-zero user are skipped. For each occurrence of the candidate in the source (latest 3000), every neighbour within ±20
positions that is in the feed's history (weight 0.95^hpos of its newest occurrence) adds to the scores.
**Train rows of day d** use a source built only from train feeds of days < d (one index per day, no val/test).
On the first train day no source exists, so all CV values are 0.
| feature | definition | single NDCG |
|---|---|---|
| seq_cv_invd20 | sum of w/d over matches (d = distance in the source sequence) | 0.3289 / 0.2807 |
| seq_cv_norm | invd20 / sqrt(pop + 1) | 0.3287 / 0.2807 |
| seq_cv_pop | # scanned source occurrences of the candidate (0 = cold item: lets the model read 0-scores as "unknown") | 0.3095 / 0.2758 |

### R. Within-feed relative versions (`feat_rel.py`, using src/features.py::_rel_feats)
| feature | single NDCG |
|---|---|
| seq_max_all_rk, seq_max_all_z (within-feed percentile rank / z-score) | 0.3438 / 0.3436 |
| seq_dcos_hl2_rk | 0.3493 |
| seq_dcos_t1_hl5_z | 0.3468 |
| seq_cv_invd20_rk | 0.3288 |

Single NDCG = full-list NDCG on val, the feature as the score, random tie-break, NaN at the bottom of the feed (random 0.290, f1 0.336).
Global rank correlation with the closest base feature: `seq_max_new8` / `seq_dcos_hl2` vs cos_h10 0.86 / 0.94;
`seq_max_all` vs maxsim32 0.76; `seq_max_old` 0.53; `seq_cv_*` vs it_hist_* 0.71-0.90 (the co-vis score is 0 for 89% of rows);
`seq_argmax_hpos` vs H 0.48. Full tables: `feats_v2/seq_/singles_*.txt`.

## 2. Explored and dropped (files deleted; code kept)

| group | features | single NDCG range | harness add-on to base (1 seed) | note |
|---|---|---|---|---|
| C: category sequence (`feat_hist.py::_cat_kernel`) | decayed category share (hl 3/20, t1, deep), momentum (last-10 share / all), # separate visits (runs) and run share, head-run length, rank, # items / repeated items in category, newest t1 / oldest event in category, entropy, last event type, # t1 in last 10 | 0.29-0.32 | +0.0002 ± 0.0012 | redundant with base cat_cnt_* / cat_min_hpos (rank corr 0.8-0.98) |
| T: Markov transition models (`feat_trans.py`) | P(next category / cluster \| previous) from train sequences (pairs within 3 positions, same item skipped, any-type and type-1 targets), K = 54 / 1024 / 4096; scores lift / mixture / last / cross-cluster lift (`xlift`); expanding-window matrices for train rows | 0.28-0.33 | -0.0004 ± 0.0012 | self-transitions duplicate embedding similarity; the cross-cluster part (xlift, 0.31) has a high gain share but no NDCG gain |
| ICV: item -> cluster co-vis (`feat_icv.py`) | histogram of the K-means clusters of the source neighbours of the 40 newest history items, scored at the candidate's cluster (share, lift 1024/4096, per-feed mass); dense, works for cold candidates | 0.32-0.33 | -0.0004 ± 0.0011 | redundant with embedding similarity |
| N: next-click embedding (`feat_nxt.py`) | ridge from 8 pooled history vectors to the next type-1 event's embedding (train-sequence cos 0.773 vs 0.755 for a plain pool) | 0.3492 | +0.0000 (E+CV+R+N vs E+CV+R) | equivalent to a decayed mean |
| other E variants | dcos hl5/hl15, recency-attention softmax (0.3501), near-duplicate counts (cos >= 0.8/0.9/0.95), segment centroid, cos to newest t1 item, max over source 0/2, deep, repeated items, type 4, types 2/3, dcos source 0 | 0.31-0.35 | adding 8 of them to E1+CV+R: 0.0074 -> 0.0059 (3 seeds) | correlated duplicates dilute the val-only model |
| contrasts (`feat_contrast.py`) | dcos_hl2 - cos_hall, max_new8 - max_old, dcos_s1 - dcos_s2, cos_u - dcos_hl5 | 0.30-0.325 | K1 + 4: 0.0065 vs K1 0.0075 (3 seeds) | — |
| CV extras | cv_w5 (d <= 5), cv_fwd5 (directional), cv_t1, cv_seqs, rk/z | 0.31-0.32 | not needed (E1+CV vs K1) | — |

## 3. Harness results (all on val; `reports/feature_evals.jsonl`)

Delta is meaningful when > 2·SE. 3-seed runs use base 0.3634 (seed mean); 1-seed runs use base 0.3642 (seed 0).

| tag | # feats | delta | SE | per-seed |
|---|---|---|---|---|
| **seq_final_keep_s3 (`--family seq_`, final files = K1)** | 26 | **+0.0080** | 0.0009 | 0.0072 / 0.0075 / 0.0093 |
| seq_k1_s3 (= KEEP) | 26 | **+0.0075** | 0.0009 | 0.0063 / 0.0072 / 0.0088 |
| seq_e1cvr_s3 (E 25 + CV 7 + R 12) | 44 | +0.0074 | 0.0009 | 0.0057 / 0.0065 / 0.0101 |
| seq_e1cv_s3 (E 25 + CV 7) | 32 | +0.0069 | 0.0009 | 0.0056 / 0.0064 / 0.0085 |
| seq_k2_s3 (14-feature core) | 14 | +0.0067 | 0.0009 | 0.0057 / 0.0056 / 0.0087 |
| seq_k1c_s3 (K1 + 4 contrasts) | 30 | +0.0065 | 0.0009 | |
| seq_all_v3_s3 (everything explored) | 88 | +0.0063 | 0.0009 | 0.0054 / 0.0067 / 0.0068 |
| seq_e2cvr_s3 (E 33 + CV + R) | 52 | +0.0059 | 0.0009 | |
| seq_k1_plus_trans_icv_s3 (K1 + tr_c4096_xlift, tr_cat_xlift, icv_lift4096, entropy) | 30 | +0.0058 | 0.0009 | |
| gap test: K1 without CV | 22 | +0.0048 | 0.0008 | |
| gap test: + 3 CV (normal source) | 25 | +0.0065 | 0.0009 | |
| gap test: + 3 CV (train source cut at Sep 17 -> val gap 8-10 days, coverage 8.8% vs private 9.8%) | 25 | +0.0057 | 0.0009 | |
| 1 seed: E / C / T / CV / ICV alone | 25/16/15/7/4 | +0.0034 / +0.0002 / -0.0004 / +0.0016 / -0.0004 | ~0.0012 | |
| 1 seed: all 80 (v2) / E+CV / E+CV+R / E+CV+R+N | | +0.0056 / +0.0056 / +0.0061 / +0.0056 | 0.0013 | |

Seed-to-seed spread of a single configuration is 0.002-0.004. Differences of 0.001 or less between subsets are not significant, and
choosing K1 as the best of about 8 similar sets makes its +0.0075 slightly optimistic (estimated by about 0.0005). The robust claims
are: **E + CV gives +0.006 to +0.008**, and C / T / ICV / N add nothing on top of the base.

## 4. Drift check (KEEP; PSI with 10 val-quantile bins + a NaN bin)

| feature | PSI val-pub | val-priv | pub-priv | median val / pub / priv |
|---|---|---|---|---|
| seq_max_new8 | 0.0037 | 0.0024 | 0.0004 | 0.661 / 0.660 / 0.660 |
| seq_max_all | 0.0037 | 0.0023 | 0.0003 | 0.829 / 0.829 / 0.830 |
| seq_dcos_hl2 | 0.0039 | 0.0024 | 0.0003 | 0.614 / 0.612 / 0.612 |
| seq_max_s1 | 0.0037 | 0.0013 | 0.0007 | 0.783 / 0.783 / 0.783 |
| seq_max_old | 0.0010 | 0.0006 | 0.0002 | 0.799 / 0.799 / 0.798 |
| seq_argmax_hpos | 0.0038 | 0.0025 | 0.0004 | 46 / 46 / 46 |
| seq_cv_invd20 (share of zeros 0.887 / 0.898 / 0.902) | 0.0014 | 0.0028 | 0.0002 | 0 |
| seq_cv_pop (share of zeros 0.420 / 0.464 / 0.476) | 0.0082 | 0.0135 | 0.0007 | 1 |
| seq_cv_invd20_rk | 0.0185 | 0.0369 | 0.0052 | 0.459 / 0.463 / 0.463 |
| all other KEEP features | <= 0.0051 | <= 0.0024 | <= 0.0007 | |

Full table: `feats_v2/seq_/drift_keep.txt`. The E features depend only on the feed's own history and the embeddings, so they are
stable (val is weekday-only; public is 100% weekend; val-public PSI is still <= 0.005). The co-vis features lose coverage
over time, as the gap test above quantifies. The relative (rank) version moves most, because ties at zero change the ranks.

## 5. Train-split notes (for whoever trains on train)

* E/R features are computed identically for train. Train candidates sit closer to the history (production ranking): median
  seq_max_all is 0.876 on train vs 0.829 on val, the same shift as maxsim32.
* CV on train uses expanding per-day sources (train days < d only). The non-zero share therefore grows over train days:
  0% (Sep 6, no source), then 8% -> 21% by Sep 19 (val: 11%). Drop the first train day, or treat it with care, if CV matters.
* No train labels are used anywhere in the family. The co-vis / transition sources use only histories (which implicitly contain
  clicks), and for train rows only from strictly earlier days.

## 6. Final KEEP list (26)

seq_max_all, seq_max_all_z, seq_max_all_rk, seq_max_s1, seq_max_old, seq_dcos_hl2, seq_dcos_hl2_rk, seq_dcos_s1,
seq_dcos_s2, seq_cos_last1, seq_top3, seq_max_nt, seq_dcos_rep, seq_dcos_t1_hl5, seq_dcos_t1_hl5_z, seq_dcos_nt_hl10,
seq_dmax_hl10, seq_cv_invd20, seq_cv_invd20_rk, seq_cv_norm, seq_max_t1, seq_argmax_hpos, seq_dcos_deep, seq_max2_gap,
seq_cv_pop, seq_max_new8

Reproduce (each script takes `<splits> [threads]`): `feat_hist.py` (E; it also writes the C group, so delete those files
afterwards or run with group arg `E`), `run_cv.py` (CV, plus ICV unless `cv` is passed as the 3rd argument), `feat_rel.py`.
Runtime with 16 threads: E train 2.5 min; CV train 32 min (12 per-day indexes); val/test 4 min.


## Verification

An independent adversarial verifier ran this check on 2026-11-11. Scripts and outputs are in
`/tmp/scratchpad/ver/` (scratch). No seq_ feature file and no
builder code was modified. **Verdict: the gain is real and leak-free, but smaller than claimed: about +0.006 to +0.007, not +0.008.
KEEP the 21 raw features. The 5 within-feed rank/z features are redundant (neutral), and `seq_cv_invd20_rk` drifts the most.**

### V1. Leakage and asymmetry audit (code + empirical)

* **No labels are used.** The only `cand_target` read in `feats_v2/seq_` is `seq_lib.single_ndcg` (evaluation only). `cand_pos` is never
  read. The rank features' arbitrary order on NaN rows (see below) does not correlate with `pos` on train (corr 0.003).
* **E group (18 raw):** each value is a pure per-feed function of the feed's own history, its embeddings and the split's own embedding
  table, and the same kernel runs on every split. **Reproduced bit-exactly from the current code:** val and test in full, and train on a random subset of
  20k feeds (0 mismatches for all 18).
* **CV group (3 raw):** for val/test rows the source is all-train (latest feed per non-zero user + all user-0 feeds, histories only)
  plus val/test feed histories with `timestamp < own timestamp`. The comparison is strict, so feeds in the same 128-s slot are excluded. Sequences of the same user are skipped.
  Val rows cannot see test, because all test feeds are later, and test rows see all of val plus earlier test. The definition is the same for both splits, causal, and has
  no future window, so nothing needs censoring at the edge of the data. **Reproduced bit-exactly** on val and test.
  For train rows of day d, the source is train feeds of days < d only (stragglers on Sep 15/20 map to an earlier cutoff), so on Sep 6 every value is 0.
  - Transductive share: with a train-only source, non-zero coverage is 11.30% on val (unchanged) and 9.70% on test (vs about 10.0% with
    the val/test histories). The time-split check below shows that the train-only variant performs the same (+0.0063 / +0.0069 vs +0.0060 / +0.0063).
    So the val/test part is not what drives the gain. Either variant is acceptable.
  - The CAP = 3000 window counts later (skipped) val/test occurrences for val rows. It binds for only 0.003% of rows, so the effect is negligible.
* **R group (5 rel):** all 3 splits were recomputed from the current raw files and match exactly (0 mismatches). This matters because the train and test
  rel files are older than the latest E reruns. Quirk of the shared `src/features.py::_rel_feats`: a NaN value gets an
  arbitrary rank. That rank comes from the array position in all-NaN feeds (1.4% of val rows, 197 feeds), and it is about 1.0 in mixed feeds (0.015% of rows). Array order carries no label
  signal (NDCG of the within-feed array index: 0.288 / 0.289 vs random 0.290). The quirk is harmless and was not changed, because it lives in shared code outside seq_.
* **Single-feature NDCG claims reproduce** (e.g. seq_max_new8 0.3511, seq_dcos_hl2 0.3493, seq_argmax_hpos −dir 0.3123).
  seq_max2_gap is at random level as a single feature (0.2885 / 0.2877); it can only help through interactions.

### V2. Harness re-runs (val, paired, base 0.3636)

| run | # feats | delta | SE | per seed |
|---|---|---|---|---|
| `--family seq_ --seeds 5` (tag `seq_VERIFY_keep_s5`) | 26 | **+0.0069** | 0.0008 | 0.0067 / 0.0075 / 0.0093 / 0.0060 / 0.0052 |
| KEEP without the 5 rk/z features, 5 seeds (`seq_VERIFY_keep_noRel_s5`) | 21 | **+0.0070** | 0.0008 | 0.0057 / 0.0074 / 0.0098 / 0.0060 / 0.0062 |
| own replication of harness folds, seeds 0-2 (base+KEEP), two runs | 26 | +0.0071 / +0.0074 | 0.0009 | 0.0061 / 0.0065 (0.0072) / 0.0088 |
| same folds, E-only (KEEP minus the 4 CV features) | 22 | +0.0049 | 0.0008 | 0.0049 / 0.0043 / 0.0054 |

* Multithreaded LightGBM is not deterministic: the same seed differs by up to 0.0007 between runs.
* The builder's +0.0080 used seeds 0-2. Those are the seeds the subset choice was made on, and they are the three highest of the five.
  On the fresh seeds 3-4 the delta is +0.0060 / +0.0052. **Honest estimate: +0.006 to +0.007.**
* The rk/z features add nothing: 21 raw features = 26 features (per-seed differences −0.0010 to +0.0010, mean +0.0001), even though
  `seq_max_all_z` ranks 2nd of 148 by gain. They substitute for the raw columns.
* CV on top of E: +0.0025 ± 0.0008 (paired, same folds).

### V3. Time-split check (fit on one val day, evaluate on the other; users of the fit day removed; same params, 350 rounds, 4 seeds)

| fit -> eval | base | +seq (26) | delta ± SE | +seq E-only | +seq, CV from train-only source |
|---|---|---|---|---|---|
| Tue Sep 22 -> Thu Sep 24 (6,870 feeds) | 0.3619 | 0.3679 | **+0.0060 ± 0.0012** | +0.0051 ± 0.0011 | +0.0063 ± 0.0012 |
| Thu Sep 24 -> Tue Sep 22 (7,533 feeds) | 0.3590 | 0.3652 | **+0.0063 ± 0.0012** | +0.0046 ± 0.0011 | +0.0069 ± 0.0012 |

The gain holds forward in time, at about 5 SE.

### V4. Co-vis coverage mismatch (simulates training on val and applying to test-private)

Harness folds, 3 seeds. The models were fit with the normal CV features, and the held-out fold was predicted with CV recomputed from a train source cut 3 days
earlier (val coverage 8.8%, harsher than private's 9.8%):
normal +0.0074, **mismatched +0.0060**, E-only +0.0049. Under mismatch the CV group still adds +0.0012 ± 0.0007, and the
coverage shift costs 0.0013 ± 0.0003. The builder's claim ("co-vis halves, about +0.001 on private") is confirmed, and there is no sign of harm.

### V5. Drift (PSI, 10 val-quantile bins + NaN bin; public/private boundary ts 1790497920)

* E and R features: PSI ≤ 0.0051 for val vs public and ≤ 0.0024 for val vs private. **Weekend vs weekday within test: ≤ 0.0004.** Tue vs Thu within val: ≤ 0.0023.
  The per-day medians are flat across Tue, Thu, Sat, Sun, Mon and Tue (e.g. seq_max_new8 0.661 / 0.661 / 0.659 / 0.658 / 0.660 / 0.662).
* CV: the non-zero share falls steadily with the gap to train: val Tue 12.0%, Thu 10.7%; test Sat 10.3%, Sun 10.0%, Mon 9.7%, Tue 9.7%.
  PSI for seq_cv_invd20 and seq_cv_norm is ≤ 0.003. seq_cv_pop: 0.008 (pub) / 0.014 (priv). **seq_cv_invd20_rk: 0.019 / 0.037**, the largest of all
  seq_ features (ties at zero move its ranks).
* Weekend proxy on train, where labels are position-biased: the single-feature lift over random is identical on weekend and weekday days
  (seq_max_new8 +0.0474 vs +0.0475, seq_dcos_hl2 +0.0493 vs +0.0484). The ratio to maxsim32 is also unchanged. CV is lower on weekends only because
  Sep 6 (a Sunday) has no source.
* Segment reweighting: per-feed val deltas reweighted to the test-public and test-private mix of hour × history length × cold share give
  +0.0076 and +0.0073 (plain val +0.0071). The gain is positive in every hour bucket (00-06 +0.012, 07-11 +0.004, 12-17 +0.009, 18-23 +0.006),
  it does not shrink for feeds with more cold candidates, and it is about zero for short histories (H 1-31: +0.0003, n = 846).

### V6. Train split

All 26 train files exist with length 247M (= `cand_fidx`). None is all-NaN or constant. NaN shares match val (e.g. seq_max_all 1.2% vs 1.4%,
seq_dcos_deep 24.8% vs 24.0%). Each feature's direction on train (position-biased labels, 20k feeds) matches val (all E features above random 0.320;
seq_argmax_hpos −dir best). Train candidates sit closer to the history (median seq_max_all 0.876 vs 0.829), the same shift maxsim32 shows.
**CV semantics differ on train.** The non-zero share by day is 0% (Sep 6), 8.4%, 12.2%, 14.4%, ... 21.0% (Sep 19), while val is 11.3% and private 9.8%.
A model trained on train would see much higher CV coverage than test. Drop Sep 6, or treat the CV columns with care, in train-trained models.

### V7. Overlap with families outside the harness base

Rank correlations on val: seq_max_new8 vs `pre_v2x_hmax8` 0.94; seq_max_all / seq_top3 / seq_max_t1 vs `ctx_hs_k256` / `ctx_hs_e1k64`
0.83-0.85; seq_dmax_hl10 vs `ctx_hs_d7` 0.87; seq_dcos_hl2 vs base cos_h10 0.94.
Whether seq_ still adds on top of base + ctx_ / pre_ was still being measured when this section was written. Script: scratch ver/overlap.py. Expect a smaller increment when those families are in the model, and re-measure before the final ensemble.

### V8. Verdict per feature

| feature(s) | verdict | reason |
|---|---|---|
| seq_max_new8, seq_dcos_hl2, seq_max_all, seq_max_s1, seq_top3, seq_max_old, seq_dcos_s1, seq_dcos_s2, seq_cos_last1, seq_dcos_t1_hl5, seq_dmax_hl10, seq_max_t1, seq_max_nt, seq_dcos_nt_hl10, seq_dcos_rep | **KEEP** | clean, reproducible, PSI < 0.006 (weekend PSI < 0.0005), same direction on train; together they make up the E group's +0.005 |
| seq_argmax_hpos, seq_dcos_deep, seq_max2_gap | **KEEP (weak)** | clean and stable, but not individually confirmable (gain rank 47-68 of 148; max2_gap is at random level alone) |
| seq_cv_invd20, seq_cv_norm | **KEEP (with caveat)** | +0.0025 on val and +0.0012 under a harsher-than-private coverage gap; coverage decays about 0.5 pp per day; train-side coverage semantics differ |
| seq_cv_pop | **KEEP (weak)** | lowest gain (rank 87); about a popularity count (overlaps it_hist_*), but it marks "cold, 0 = unknown" for the CV scores |
| seq_max_all_z, seq_max_all_rk, seq_dcos_hl2_rk, seq_dcos_t1_hl5_z | **NEUTRAL (redundant)** | removing all 5 rel features changes nothing (+0.0070 vs +0.0069, 5 seeds); harmless to keep |
| seq_cv_invd20_rk | **DROP (or neutral)** | redundant, and the highest drift in the family (PSI 0.037 val vs private) |

**Final KEEP (21):** seq_max_new8, seq_dcos_hl2, seq_max_all, seq_max_s1, seq_top3, seq_max_old, seq_dcos_s1, seq_dcos_s2,
seq_cos_last1, seq_dcos_t1_hl5, seq_dmax_hl10, seq_max_t1, seq_max_nt, seq_dcos_nt_hl10, seq_dcos_rep, seq_argmax_hpos, seq_dcos_deep,
seq_max2_gap, seq_cv_invd20, seq_cv_norm, seq_cv_pop. Expected private gain on top of base: about +0.006 (E about +0.005 stable, CV about +0.001 to +0.002).
The 5 rel files are still on disk. `--family seq_` picks them up, which is harmless (equal score), but use `--names` with the 21 for a clean set.
