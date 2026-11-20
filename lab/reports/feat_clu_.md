# Feature family `clu_`: embedding clusters and dense collaborative target encoding

Code: `<repo>/feats_v2/clu_/`. Intermediate tables and cluster assignments: `<repo>/data/clu_work/`.
All NDCG numbers are full-list val NDCG (`src/metrics.py`). Harness = `src/eval_family.py` (paired 5-fold GroupKFold-by-user LightGBM on val, base = frozen 122-feature set, fixed 350 rounds). "s5" means 5 fold-shuffle seeds and "s8" means 8.
SE is the standard error of the paired per-feed delta. A delta counts only if it is larger than 2·SE.

---

## 0. TL;DR

1. **The family mostly duplicates what the base already has.** Whole family (122 features), 5 seeds: **+0.0007 ± 0.0007**, not significant.
   On top of the base, history-relative cluster signals add little, because maxsim/attn/cos_h*/cat_cnt* already carry that information. Label-based cluster statistics add nothing.
2. **Label-based cluster target encodings (item-cluster CTR/COEC, user-cluster × item-cluster COEC, history-cluster → candidate-cluster CF lift, Rocchio preference directions, daypart lifts) do not help and tend to hurt.**
   Results: 36 features, 3 seeds: −0.0001 ± 0.0007. The first 1-seed screens were −0.0025, −0.0018 and −0.0017.
   They do carry signal on their own (CF lift 0.3226, PMI 0.327, Rocchio-user 0.314 single NDCG vs random 0.289). The problem is that every per-cluster constant acts as a cluster ID, so the val-trained LightGBM fits fold-specific cluster effects.
   They also add nothing to a LightGBM trained on 300K **train** feeds (paired delta on val **−0.0005 ± 0.0012**, and no label-based clu_ feature makes the top-30 gain list). The "dense collaborative TE helps cold items" hypothesis is **not supported**.
3. **KEEP: 6 label-free history-relative cluster features.** 8 seeds: **+0.00125 ± 0.00045** (2.8 SE). Per-seed deltas: +0.0009 +0.0011 +0.0045 −0.0003 +0.0006 +0.0018 +0.0007 +0.0006.
   The features: `clu_hcnt1024, clu_hmin1024, clu_hrw4096, clu_het14096, clu_pmi1024_mean, clu_pmi11024_mean`.
   *Caveat:* I picked this subset after seeing 3-seed results. On the 5 fold-seeds not used for the choice (s3..s7), the gain is smaller: mean **+0.0007**, 4 of 5 positive. Seed 2's base OOF is weak (base 0.3619; the other 7 seeds give 0.3631-0.3642), so every family looks better on s2.
   Expect roughly +0.0005..0.001 on top of the base.
   As a control, the full history-histogram block was defined before any results (18 features, `clu_{hcnt,hshare,hrw,het1,hmin,fshare}{256,1024,4096}`).
   It gets **+0.00104 ± 0.00048 at 8 seeds, with all 8 seeds positive**, which supports a real effect of about +0.001. compact6 matches it with a third of the features.
4. **Drift is negligible for the KEEP set.** PSI val-vs-public ≤ 0.0044, val-vs-private ≤ 0.0025, public-vs-private ≤ 0.0006. All 6 features are label-free, defined identically for every split, and equally discriminative for unseen items (§6), so the train-coverage drop on test does not affect them.
5. **Calendar and time-of-day ideas (per the user's request)** fail on val:
   * hour-of-day cluster demand lift (label-free, all splits): 0.282 single, both directions;
   * daypart × cluster COEC lift (train labels);
   * weekday/weekend × cluster COEC lift.
   Time context, 4 features: −0.0017 ± 0.0011 (1 seed). The weekend lift cannot be validated on val because val has only weekdays, and its PSI val-vs-public is 0.78-1.2 by construction. **Dropped.**
6. **PCA-16 of the item embedding as raw features: −0.0007 ± 0.0012. Dropped.** The components are pure item coordinates, and the val-trained model can only memorise with them. Distance to the centroid (`clu_ccos*`, 0.295-0.298 single) is also not useful.

---

## 1. Clusters

Script: `kmeans.py`. Spherical k-means in torch on GPU N (`CUDA_VISIBLE_DEVICES=N`): cosine on L2-normalised int8 vectors, random init, 25-30 Lloyd iterations; empty clusters are re-seeded (none occurred).

| object | sample | K | mean cos to own centroid | cluster size min/max (sample) |
|---|---|---|---|---|
| item | 2.80M **occurrence-weighted** rows: 1.0M train + 0.5M val + 0.7M test candidate rows, 0.3M + 0.15M + 0.25M history rows | 256 / 1024 / 4096 | 0.789 / 0.836 / 0.876 | 2165-25097 / 587-7548 / 114-2417 |
| user | 1.0M train feeds (non-zero vectors) + all val + all test | 64 / 256 / 1024 | 0.741 / 0.771 / 0.795 | — |

* Every item-table row of every split gets a cluster id for each K from the same centroids (train has 81.6M rows, val 2.5M, test 4.7M). Assigning all of them takes 10 s on the GPU.
  Candidates and history rows map through `emb_idx`. A missing embedding gives cluster −1, and the features are then NaN. That is 0.015% of candidates and 14% of history rows.
* All clusters are used in every split. The largest test share is 0.93% / 0.31% / 0.10% for K = 256 / 1024 / 4096.
* PCA on the same sample: the top-16 components explain 26.9% of the variance (top-1 6.0%). The embedding is high-rank (EDA: 90 dimensions for 90% of the variance).

## 2. Position-debiased expected clicks (for all label-based features)

`tables.py`: `ex = CTR(cell)`, where cell = pos bucket (0..9, 10-14, 15-19, 20-29, 30-49, 50-99, 100-199, 200+) × rank-from-end bucket (0, 1, 2, 3, 4, 5-9, 10-19, 20+) × M bucket (<10, 10-19, 20-39, 40-79, 80-149, 150+). That is 816 cells over all 247M train rows.
Rank-from-end is `M-1-rank(pos)` within the feed, because pos has gaps. corr(ex, y) = 0.27.
COEC = Σclick / Σex. The cell table is a nuisance model fitted on 247M rows, so its own-day leakage is negligible.

## 3. Feature definitions (all written for train, val and test; float32, aligned with `cand_*.npy`)

**Time-awareness.** Every label-based table is stored per MSK train day (15 day slots). For train rows on day d, the table sums days < d (day 0, Sep 6, gets NaN). For val/test rows, it sums all train days.
No val labels are used anywhere. Label-free transductive tables use history and candidate rows of all splits and are identical for val and test. None of them looks forward in time, so no censoring issue arises.

| group | features | definition |
|---|---|---|
| hist (label-free) | `clu_hcnt{K}, clu_hshare{K}, clu_hrw{K}, clu_het1{K}, clu_hmin{K}, clu_fshare{K}`, K = 256/1024/4096 | Over the user's ≤256 history events with the candidate's cluster: count, share of valid history, recency-weighted count Σ 0.5^(hpos/5), count of event_type 1, newest hpos (999 if none). Also the share of the feed's candidates in the same cluster. |
| hist (label-free) | `clu_het{e}_1024`, e = 2..10 | History count of event type e in the candidate's K=1024 cluster |
| pmi (label-free) | `clu_pmi{K}_{mean,max}, clu_pmi1{K}_{mean,max}`, K = 256/1024 | Cluster transition matrix T[a→b] over consecutive valid history events, from all splits. Train feeds are weighted 1/(#train feeds of the user), so repeated histories of heavy users count once. T1 counts only transitions whose newer event has event_type 1. PMI(a→b) = log(P(b\|a)/P(b)) with P(b\|a) smoothed by 10·P(b). Feature = recency-weighted mean (weight 0.5^(hpos/5)) / max over the last 16 valid history clusters a of PMI(a → candidate cluster). |
| emb | `clu_ccos{K}`, `clu_pca0..15` | Cosine of the candidate to its centroid; PCA-16 projection of the normalised item vector |
| te (labels) | `clu_ic{K}_ctr, clu_ic{K}_coec` | Item-cluster CTR (prior 0.0221, strength 100) and COEC ((pos+20)/(ex+20)), time-aware |
| uxi (labels) | `clu_u{Ku}i{Ki}_{coec,lift}` for (64,256), (64,1024), (256,256), (256,1024), (256,4096), (1024,256), (1024,1024), (1024,4096) | User-embedding cluster × item cluster COEC, shrunk to the item-cluster COEC with 10 expected clicks; lift = log(coec / item-cluster coec) |
| cf (labels) | `clu_cf{K}_{ll,ratio}`, K = 256/1024 | Debiased history-cluster → candidate-cluster table. Every train candidate row adds y and ex to cells (h, c) for its feed's last 16 history clusters h, with normalised recency weights. Feature = recency-weighted mean log-lift of (P+5R)/(E+5)/R over the user's last 16 history clusters (R = the candidate cluster's COEC), plus the pooled ratio. |
| roc (labels) | `clu_roc_{g,u64,u256,u1024,i256,i1024}` | Rocchio preference direction per group: d = (Σy·v)/Σy − (Σex·v)/Σex (shrunk to the global direction with 20 clicks). Feature = cos(v_item, d_group). Groups: global, user-emb cluster, or the candidate's own item cluster. |
| time (labels) | `clu_dp{K}_lift, clu_wk{K}_lift`, K = 256/1024 | log COEC lift of the cluster in the request's MSK 6-h daypart / weekday-vs-weekend vs the cluster overall |
| hod (label-free) | `clu_hod{K}_lift` | log lift of the cluster's share among recent history events (hpos < 5) of feeds requested at the same MSK hour, vs all hours (all splits) |
| etr (label-free) | `clu_etr1024_e1..e10, clu_etr{256,4096}_e1` | Cluster's event-type profile over all histories: log lift of the share of event type e |
| dsr (label-free) | `clu_dsr{K}, clu_dem{K}` | Same-day cluster demand share (history events with hpos < 10 of that day's feeds) over supply share (that day's candidates); log demand share |
| frel (label-free) | `clu_frel{K}_f{1..5}` | z-score of f1..f5 against the candidate's cluster mean/std (val+test candidate rows) |
| dup (label-free) | `clu_dup95, clu_maxcos_feed, clu_grp_rk_dot, clu_grp_rk_f1` | Number of near-duplicates (cos ≥ 0.95) in the feed; max cos to another candidate; pct rank of dot(user,item) / f1 among same-K=4096-cluster candidates of the feed |

## 4. Single-feature NDCG on val (random tie-break averaged over 3 noise draws, NaN to the bottom) and drift

Random = 0.2894, f1 = 0.336, maxsim32 = 0.349 with the same scorer. Both directions are shown; the better one is in bold for the KEEP features.
Several cluster-level constants score **below random in both directions**, for example PCA, dp/hod lifts and etr. Positives sit in the middle of their distribution: they belong to large clusters, whose lift is near 0. This shows the features act as cluster identifiers rather than ordinal scores.
PSI uses val deciles plus a NaN bin. pub = test before 2026-09-27 11:32 MSK, priv = after.

| feature | coverage | ndcg(+) | ndcg(−) | PSI val-pub | PSI val-priv | PSI pub-priv |
|---|---|---|---|---|---|---|
| clu_ccos1024 | 1.000 | 0.2947 | 0.2794 | 0.0000 | 0.0000 | 0.0000 |
| clu_ccos256 | 1.000 | 0.2959 | 0.2773 | 0.0000 | 0.0000 | 0.0000 |
| clu_ccos4096 | 1.000 | 0.2977 | 0.2772 | 0.0000 | 0.0000 | 0.0000 |
| clu_cf1024_ll | 0.986 | 0.3197 | 0.2617 | 0.0042 | 0.0025 | 0.0004 |
| clu_cf1024_ratio | 0.986 | 0.3226 | 0.2586 | 0.0048 | 0.0030 | 0.0008 |
| clu_cf256_ll | 0.986 | 0.3191 | 0.2599 | 0.0045 | 0.0028 | 0.0005 |
| clu_cf256_ratio | 0.986 | 0.3196 | 0.2579 | 0.0043 | 0.0028 | 0.0009 |
| clu_dem1024 | 1.000 | 0.2894 | 0.2785 | 0.0108 | 0.0121 | 0.0093 |
| clu_dem256 | 1.000 | 0.2887 | 0.2797 | 0.0375 | 0.0121 | 0.0350 |
| clu_dp1024_lift | 1.000 | 0.2833 | 0.2828 | 0.0262 | 0.0060 | 0.0571 |
| clu_dp256_lift | 1.000 | 0.2826 | 0.2837 | 0.0451 | 0.0106 | 0.0991 |
| clu_dsr1024 | 1.000 | 0.2949 | 0.2736 | 0.0095 | 0.0072 | 0.0055 |
| clu_dsr256 | 1.000 | 0.2933 | 0.2748 | 0.0566 | 0.0128 | 0.0306 |
| clu_dup95 | 1.000 | 0.2991 | 0.2865 | 0.0001 | 0.0001 | 0.0000 |
| clu_etr1024_e1 | 1.000 | 0.2918 | 0.2754 | 0.0002 | 0.0004 | 0.0003 |
| clu_etr1024_e10 | 1.000 | 0.2833 | 0.2800 | 0.0005 | 0.0007 | 0.0010 |
| clu_etr1024_e2 | 1.000 | 0.2777 | 0.2857 | 0.0003 | 0.0002 | 0.0005 |
| clu_etr1024_e3 | 1.000 | 0.2746 | 0.2883 | 0.0003 | 0.0002 | 0.0004 |
| clu_etr1024_e4 | 1.000 | 0.2786 | 0.2825 | 0.0004 | 0.0002 | 0.0008 |
| clu_etr1024_e5 | 1.000 | 0.2810 | 0.2788 | 0.0003 | 0.0003 | 0.0007 |
| clu_etr1024_e6 | 1.000 | 0.2811 | 0.2788 | 0.0003 | 0.0003 | 0.0008 |
| clu_etr1024_e7 | 1.000 | 0.2753 | 0.2876 | 0.0005 | 0.0003 | 0.0013 |
| clu_etr1024_e8 | 1.000 | 0.2822 | 0.2791 | 0.0006 | 0.0003 | 0.0007 |
| clu_etr1024_e9 | 1.000 | 0.2745 | 0.2895 | 0.0004 | 0.0005 | 0.0012 |
| clu_etr256_e1 | 1.000 | 0.2897 | 0.2763 | 0.0003 | 0.0003 | 0.0002 |
| clu_etr4096_e1 | 1.000 | 0.2935 | 0.2738 | 0.0002 | 0.0001 | 0.0002 |
| clu_frel1024_f1 | 1.000 | 0.3265 | 0.2574 | 0.0007 | 0.0005 | 0.0002 |
| clu_frel1024_f2 | 1.000 | 0.2879 | 0.2884 | 0.0005 | 0.0005 | 0.0001 |
| clu_frel1024_f3 | 1.000 | 0.3125 | 0.2579 | 0.0020 | 0.0013 | 0.0006 |
| clu_frel1024_f4 | 1.000 | 0.2991 | 0.2696 | 0.0016 | 0.0001 | 0.0013 |
| clu_frel1024_f5 | 1.000 | 0.2993 | 0.2671 | 0.0006 | 0.0002 | 0.0002 |
| clu_frel256_f1 | 1.000 | 0.3269 | 0.2576 | 0.0009 | 0.0006 | 0.0003 |
| clu_frel256_f2 | 1.000 | 0.2872 | 0.2892 | 0.0006 | 0.0005 | 0.0002 |
| clu_frel256_f3 | 1.000 | 0.3152 | 0.2584 | 0.0021 | 0.0013 | 0.0005 |
| clu_frel256_f4 | 1.000 | 0.2999 | 0.2686 | 0.0016 | 0.0001 | 0.0012 |
| clu_frel256_f5 | 1.000 | 0.2993 | 0.2665 | 0.0006 | 0.0003 | 0.0002 |
| clu_fshare1024 | 1.000 | 0.3075 | 0.2704 | 0.0007 | 0.0003 | 0.0002 |
| clu_fshare256 | 1.000 | 0.3047 | 0.2687 | 0.0015 | 0.0004 | 0.0006 |
| clu_fshare4096 | 1.000 | 0.3083 | 0.2731 | 0.0014 | 0.0005 | 0.0005 |
| clu_grp_rk_dot | 0.709 | 0.3107 | 0.2872 | 0.0008 | 0.0006 | 0.0000 |
| clu_grp_rk_f1 | 0.709 | 0.3074 | 0.2942 | 0.0003 | 0.0002 | 0.0001 |
| **clu_hcnt1024** (KEEP) | 1.000 | **0.3171** | 0.2608 | 0.0004 | 0.0001 | 0.0004 |
| clu_hcnt256 | 1.000 | 0.3103 | 0.2605 | 0.0003 | 0.0001 | 0.0002 |
| clu_hcnt4096 | 1.000 | 0.3206 | 0.2630 | 0.0004 | 0.0001 | 0.0004 |
| clu_het10_1024 | 1.000 | 0.2912 | 0.2886 | 0.0000 | 0.0000 | 0.0000 |
| clu_het11024 | 1.000 | 0.3191 | 0.2624 | 0.0005 | 0.0002 | 0.0005 |
| clu_het1256 | 1.000 | 0.3133 | 0.2618 | 0.0004 | 0.0002 | 0.0002 |
| **clu_het14096** (KEEP) | 1.000 | **0.3224** | 0.2649 | 0.0005 | 0.0001 | 0.0003 |
| clu_het2_1024 | 1.000 | 0.3013 | 0.2794 | 0.0000 | 0.0001 | 0.0001 |
| clu_het3_1024 | 1.000 | 0.3031 | 0.2785 | 0.0001 | 0.0001 | 0.0001 |
| clu_het4_1024 | 1.000 | 0.2983 | 0.2834 | 0.0000 | 0.0000 | 0.0000 |
| clu_het5_1024 | 1.000 | 0.2983 | 0.2840 | 0.0000 | 0.0001 | 0.0000 |
| clu_het6_1024 | 1.000 | 0.2970 | 0.2848 | 0.0000 | 0.0001 | 0.0000 |
| clu_het7_1024 | 1.000 | 0.2935 | 0.2862 | 0.0000 | 0.0000 | 0.0000 |
| clu_het8_1024 | 1.000 | 0.2959 | 0.2861 | 0.0000 | 0.0000 | 0.0000 |
| clu_het9_1024 | 1.000 | 0.2906 | 0.2886 | 0.0000 | 0.0000 | 0.0000 |
| **clu_hmin1024** (KEEP) | 1.000 | 0.2591 | **0.3298** | 0.0002 | 0.0001 | 0.0001 |
| clu_hmin256 | 1.000 | 0.2584 | 0.3219 | 0.0005 | 0.0001 | 0.0004 |
| clu_hmin4096 | 1.000 | 0.2619 | 0.3310 | 0.0002 | 0.0001 | 0.0002 |
| clu_hod1024_lift | 1.000 | 0.2823 | 0.2831 | 0.0189 | 0.0033 | 0.0206 |
| clu_hod256_lift | 1.000 | 0.2817 | 0.2838 | 0.0251 | 0.0065 | 0.0259 |
| clu_hrw1024 | 1.000 | 0.3269 | 0.2606 | 0.0004 | 0.0002 | 0.0001 |
| clu_hrw256 | 1.000 | 0.3212 | 0.2593 | 0.0007 | 0.0002 | 0.0007 |
| **clu_hrw4096** (KEEP) | 1.000 | **0.3292** | 0.2634 | 0.0003 | 0.0001 | 0.0002 |
| clu_hshare1024 | 0.986 | 0.3171 | 0.2608 | 0.0040 | 0.0025 | 0.0003 |
| clu_hshare256 | 0.986 | 0.3103 | 0.2605 | 0.0040 | 0.0025 | 0.0006 |
| clu_hshare4096 | 0.986 | 0.3206 | 0.2630 | 0.0039 | 0.0023 | 0.0004 |
| clu_ic1024_coec | 1.000 | 0.2955 | 0.2695 | 0.0012 | 0.0003 | 0.0008 |
| clu_ic1024_ctr | 1.000 | 0.2947 | 0.2709 | 0.0005 | 0.0002 | 0.0003 |
| clu_ic256_coec | 1.000 | 0.2923 | 0.2722 | 0.0008 | 0.0002 | 0.0009 |
| clu_ic256_ctr | 1.000 | 0.2924 | 0.2724 | 0.0010 | 0.0005 | 0.0008 |
| clu_ic4096_coec | 1.000 | 0.2964 | 0.2695 | 0.0007 | 0.0002 | 0.0004 |
| clu_ic4096_ctr | 1.000 | 0.2966 | 0.2699 | 0.0006 | 0.0002 | 0.0003 |
| clu_maxcos_feed | 1.000 | 0.3070 | 0.2599 | 0.0004 | 0.0002 | 0.0001 |
| clu_pca0 | 1.000 | 0.2826 | 0.2794 | 0.0003 | 0.0002 | 0.0009 |
| clu_pca1 | 1.000 | 0.2721 | 0.2912 | 0.0002 | 0.0001 | 0.0003 |
| clu_pca10 | 1.000 | 0.2832 | 0.2865 | 0.0002 | 0.0001 | 0.0005 |
| clu_pca11 | 1.000 | 0.2875 | 0.2825 | 0.0002 | 0.0000 | 0.0001 |
| clu_pca12 | 1.000 | 0.2849 | 0.2802 | 0.0001 | 0.0001 | 0.0002 |
| clu_pca13 | 1.000 | 0.2829 | 0.2866 | 0.0003 | 0.0000 | 0.0004 |
| clu_pca14 | 1.000 | 0.2878 | 0.2822 | 0.0001 | 0.0000 | 0.0001 |
| clu_pca15 | 1.000 | 0.2797 | 0.2869 | 0.0003 | 0.0000 | 0.0002 |
| clu_pca2 | 1.000 | 0.2825 | 0.2836 | 0.0002 | 0.0001 | 0.0002 |
| clu_pca3 | 1.000 | 0.2755 | 0.2868 | 0.0001 | 0.0001 | 0.0001 |
| clu_pca4 | 1.000 | 0.2857 | 0.2828 | 0.0000 | 0.0002 | 0.0002 |
| clu_pca5 | 1.000 | 0.2830 | 0.2852 | 0.0002 | 0.0007 | 0.0013 |
| clu_pca6 | 1.000 | 0.2823 | 0.2833 | 0.0002 | 0.0003 | 0.0008 |
| clu_pca7 | 1.000 | 0.2886 | 0.2820 | 0.0001 | 0.0000 | 0.0002 |
| clu_pca8 | 1.000 | 0.2791 | 0.2894 | 0.0007 | 0.0001 | 0.0003 |
| clu_pca9 | 1.000 | 0.2842 | 0.2879 | 0.0003 | 0.0002 | 0.0003 |
| clu_pmi1024_max | 0.986 | 0.3200 | 0.2574 | 0.0040 | 0.0023 | 0.0004 |
| **clu_pmi1024_mean** (KEEP) | 0.986 | **0.3270** | 0.2562 | 0.0044 | 0.0025 | 0.0006 |
| clu_pmi11024_max | 0.986 | 0.3188 | 0.2574 | 0.0041 | 0.0023 | 0.0005 |
| **clu_pmi11024_mean** (KEEP) | 0.986 | **0.3265** | 0.2565 | 0.0042 | 0.0025 | 0.0005 |
| clu_pmi1256_max | 0.986 | 0.3117 | 0.2606 | 0.0039 | 0.0025 | 0.0006 |
| clu_pmi1256_mean | 0.986 | 0.3200 | 0.2602 | 0.0045 | 0.0028 | 0.0006 |
| clu_pmi256_max | 0.986 | 0.3125 | 0.2602 | 0.0040 | 0.0024 | 0.0006 |
| clu_pmi256_mean | 0.986 | 0.3202 | 0.2602 | 0.0044 | 0.0027 | 0.0006 |
| clu_roc_g | 1.000 | 0.2910 | 0.2706 | 0.0006 | 0.0002 | 0.0006 |
| clu_roc_i1024 | 1.000 | 0.2945 | 0.2732 | 0.0001 | 0.0000 | 0.0001 |
| clu_roc_i256 | 1.000 | 0.2944 | 0.2724 | 0.0002 | 0.0001 | 0.0002 |
| clu_roc_u1024 | 0.986 | 0.3144 | 0.2619 | 0.0283 | 0.0175 | 0.0013 |
| clu_roc_u256 | 0.986 | 0.3082 | 0.2642 | 0.0281 | 0.0175 | 0.0015 |
| clu_roc_u64 | 0.986 | 0.3045 | 0.2633 | 0.0286 | 0.0170 | 0.0015 |
| clu_u1024i1024_coec | 0.986 | 0.3056 | 0.2665 | 0.0282 | 0.0169 | 0.0015 |
| clu_u1024i1024_lift | 0.986 | 0.3009 | 0.2721 | 0.0283 | 0.0170 | 0.0012 |
| clu_u1024i256_coec | 0.986 | 0.3039 | 0.2678 | 0.0283 | 0.0173 | 0.0015 |
| clu_u1024i256_lift | 0.986 | 0.2996 | 0.2722 | 0.0288 | 0.0174 | 0.0015 |
| clu_u1024i4096_coec | 0.986 | 0.3060 | 0.2659 | 0.0282 | 0.0169 | 0.0013 |
| clu_u1024i4096_lift | 0.986 | 0.3011 | 0.2754 | 0.0281 | 0.0169 | 0.0012 |
| clu_u256i1024_coec | 0.986 | 0.3020 | 0.2673 | 0.0281 | 0.0169 | 0.0012 |
| clu_u256i1024_lift | 0.986 | 0.2959 | 0.2725 | 0.0285 | 0.0170 | 0.0014 |
| clu_u256i256_coec | 0.986 | 0.3012 | 0.2684 | 0.0285 | 0.0171 | 0.0014 |
| clu_u256i256_lift | 0.986 | 0.2955 | 0.2726 | 0.0291 | 0.0174 | 0.0014 |
| clu_u256i4096_coec | 0.986 | 0.3037 | 0.2667 | 0.0282 | 0.0170 | 0.0013 |
| clu_u256i4096_lift | 0.986 | 0.2967 | 0.2737 | 0.0284 | 0.0169 | 0.0013 |
| clu_u64i1024_coec | 0.986 | 0.3012 | 0.2674 | 0.0284 | 0.0170 | 0.0013 |
| clu_u64i1024_lift | 0.986 | 0.2938 | 0.2722 | 0.0283 | 0.0169 | 0.0016 |
| clu_u64i256_coec | 0.986 | 0.3000 | 0.2678 | 0.0286 | 0.0169 | 0.0016 |
| clu_u64i256_lift | 0.986 | 0.2952 | 0.2712 | 0.0286 | 0.0171 | 0.0017 |
| clu_wk1024_lift | 1.000 | 0.2800 | 0.2838 | 0.7805 | 0.0579 | 0.4105 |
| clu_wk256_lift | 1.000 | 0.2798 | 0.2859 | 1.2118 | 0.1092 | 0.5856 |

## 5. Harness results (`reports/feature_evals.jsonl`, tags `clu_*`)

| tag | #feat | seeds | base | cand | delta | SE | per-seed delta |
|---|---|---|---|---|---|---|---|
| clu_lf_hist_pmi_s1 | 26 | 1 | 0.3642 | 0.3652 | +0.0011 | 0.0012 | +0.0011 |
| clu_all_v1_s1 | 79 | 1 | 0.3642 | 0.3650 | +0.0008 | 0.0013 | +0.0008 |
| clu_te_uxi_cf_s1 | 26 | 1 | 0.3642 | 0.3617 | -0.0025 | 0.0012 | -0.0025 |
| clu_compact_lb_s1 | 5 | 1 | 0.3642 | 0.3624 | -0.0018 | 0.0011 | -0.0018 |
| clu_compact_lf_s1 | 9 | 1 | 0.3642 | 0.3640 | -0.0001 | 0.0012 | -0.0001 |
| clu_timectx_s1 | 4 | 1 | 0.3642 | 0.3625 | -0.0017 | 0.0011 | -0.0017 |
| clu_dup_s1 | 4 | 1 | 0.3642 | 0.3634 | -0.0007 | 0.0012 | -0.0007 |
| clu_lf_hist_pmi_s3 | 26 | 3 | 0.3634 | 0.3645 | +0.0011 | 0.0007 | +0.0006 -0.0001 +0.0028 |
| clu_pca16_s1 | 16 | 1 | 0.3642 | 0.3635 | -0.0007 | 0.0012 | -0.0007 |
| clu_all_v2_s3 | 113 | 3 | 0.3634 | 0.3646 | +0.0012 | 0.0008 | +0.0011 +0.0002 +0.0024 |
| clu_labelbased_s3 | 36 | 3 | 0.3634 | 0.3633 | -0.0001 | 0.0007 | -0.0007 -0.0005 +0.0008 |
| clu_ctx_lf_s3 | 51 | 3 | 0.3634 | 0.3638 | +0.0004 | 0.0007 | +0.0001 -0.0011 +0.0022 |
| clu_pmi_s3 | 8 | 3 | 0.3634 | 0.3644 | +0.0010 | 0.0007 | +0.0006 -0.0002 +0.0026 |
| clu_hist18_s3 | 18 | 3 | 0.3634 | 0.3645 | +0.0011 | 0.0007 | +0.0003 +0.0008 +0.0021 |
| clu_compact6_s3 | 6 | 3 | 0.3634 | 0.3656 | +0.0022 | 0.0007 | +0.0009 +0.0011 +0.0045 |
| clu_lf16_s3 | 16 | 3 | 0.3634 | 0.3646 | +0.0012 | 0.0007 | +0.0016 -0.0005 +0.0024 |
| clu_pmi2_s5 | 2 | 5 | 0.3636 | 0.3631 | -0.0004 | 0.0005 | -0.0007 -0.0006 +0.0008 -0.0009 -0.0007 |
| clu_compact6_s5 | 6 | 5 | 0.3636 | 0.3649 | +0.0014 | 0.0005 | +0.0009 +0.0011 +0.0045 -0.0003 +0.0006 |
| clu_hist4_s5 | 4 | 5 | 0.3636 | 0.3642 | +0.0006 | 0.0005 | +0.0007 +0.0004 +0.0031 +0.0006 -0.0016 |
| clu_hist18_s5 | 18 | 5 | 0.3636 | 0.3646 | +0.0010 | 0.0006 | +0.0003 +0.0008 +0.0021 +0.0019 +0.0001 |
| clu_het9_s5 | 9 | 5 | 0.3636 | 0.3637 | +0.0001 | 0.0005 | -0.0007 -0.0014 +0.0025 +0.0000 +0.0001 |
| clu_lf26_s5 | 27 | 5 | 0.3636 | 0.3640 | +0.0005 | 0.0006 | +0.0007 -0.0007 +0.0029 -0.0001 -0.0004 |
| clu_hist12_s5 | 12 | 5 | 0.3636 | 0.3644 | +0.0008 | 0.0006 | -0.0004 +0.0008 +0.0026 +0.0009 +0.0004 |
| clu_all_v3_s5 | 122 | 5 | 0.3636 | 0.3642 | +0.0007 | 0.0007 | +0.0014 -0.0002 +0.0027 -0.0008 +0.0003 |
| clu_compact6_s8 | 6 | 8 | 0.3635 | 0.3647 | +0.0012 | 0.0005 | +0.0009 +0.0011 +0.0045 -0.0003 +0.0006 +0.0018 +0.0007 +0.0006 |
| clu_hist18_s8 | 18 | 8 | 0.3635 | 0.3645 | +0.0010 | 0.0005 | +0.0003 +0.0008 +0.0021 +0.0021 +0.0001 +0.0017 +0.0010 +0.0002 |

Reading:
* **Whole family:** 79 features (v1, 1 seed) +0.0008; 113 features (v2, 3 seeds) +0.0012 ± 0.0008; 122 features (v3, 5 seeds) **+0.0007 ± 0.0007**.
* **Label-free history-relative features** (hist + pmi) are the only consistently non-negative group: hist18 s5 +0.0010 ± 0.0006 with all 5 seeds positive; hist18 s8 +0.00104 ± 0.00048 (2.2 SE, all 8 seeds positive).
  PMI alone is not useful (pmi2 s5 −0.0004), but it helps next to the hist counts (compact6 vs hist4: +0.0014 vs +0.0006 at s5).
* **Label-based groups** are neutral or negative at every granularity: te_uxi_cf, compact_lb (roc + cf + uxi lift), timectx, labelbased_s3.
* **Context groups** (etr, dsr, hod, frel, ccos, dup, pca): +0.0004 ± 0.0007. None of them is worth its cost.

### 5.1 Train-trained LightGBM check (are label-based clu_ features useful when 3M train feeds are available?)

`src/train_lgb.py --n_feeds 300000 --pos_mode none` (same seed, so the same 300K train feeds as the existing `lgb_tr300k_none` run), base + 97 clu_ features (all except pca/wk):
val **0.3677 vs 0.3682** (paired −0.0005 ± 0.0012). The run stopped at iteration 52 instead of 251.
The only clu_ features in the top-30 gain are `clu_frel1024_f5`, `clu_grp_rk_dot` and `clu_frel256_f5`. With plenty of data the label-based cluster TE still adds nothing.
The likely reason: the train-label signal at cluster level reflects production-ranker exposure, which COEC with a position-only model cannot remove, and the base `tte_*` / `it_hist_*` features already cover the item-level part.

## 6. Cold items

Mean within-feed percentile rank of val positives (0.5 = random), split by whether the item appeared in train (as a candidate or in history). 22.8% of val positives are unseen.

| feature | seen positives | unseen positives |
|---|---|---|
| maxsim32 | 0.642 | 0.645 |
| dot_u | 0.635 | 0.657 |
| clu_hrw4096 | 0.603 | 0.610 |
| clu_pmi1024_mean | 0.598 | 0.620 |
| clu_cf1024_ratio | 0.597 | 0.604 |
| clu_u1024i4096_coec | 0.554 | 0.569 |
| clu_ic4096_coec | 0.544 | 0.560 |

All cluster features work at least as well for unseen items as for seen ones, as designed. Their value is limited by redundancy with the base, not by coverage.

## 7. Train-split sanity

Train feature distributions are shifted toward history (median `clu_hcnt1024`: 5 on train vs 2 on val/test; `clu_pmi1024_mean`: 0.85 vs 0.40/0.38). This matches the known production-vs-exploration shift (EDA: train candidates sit closer to history).
For a model trained on train rows, this is the same shift the base similarity features have.
Label-based train features are strictly earlier-day. The PMI tables include later train histories, but only as cluster-level aggregates over about 0.6B transitions, so the leakage is negligible.

## 8. KEEP list and deliverables

**KEEP (6, all label-free):** `clu_hcnt1024, clu_hmin1024, clu_hrw4096, clu_het14096, clu_pmi1024_mean, clu_pmi11024_mean`.
8-seed harness: +0.00125 ± 0.00045, with an expected real gain of about +0.0005..0.001 (see TL;DR 3).
Optional alternative if the final model prefers a fuller history block: the 18 `clu_{hcnt,hshare,hrw,het1,hmin,fshare}{256,1024,4096}` features (s8 +0.00104 ± 0.00048 (2.2 SE, all 8 seeds positive)). Those files were deleted, but they can be regenerated (below).

All other clu_ feature files were deleted from `data/feats/{train,val,test}/`. Regenerate everything from the code if needed:
```
cd <repo>
CUDA_VISIBLE_DEVICES=N .venv/bin/python feats_v2/clu_/kmeans.py          # centroids + assignments (~40 s)
.venv/bin/python feats_v2/clu_/pmi_table.py                              # transition + demand/supply tables
CLU_KEEP_ONLY=1 .venv/bin/python feats_v2/clu_/build.py train,val,test hist,pmi   # writes only the KEEP files (~5 min)
# label-based / other groups: feats_v2/clu_/tables.py te cf=256,1024 ; tables_roc.py ; etr_table.py ; hod_table.py ;
# build.py <splits> te,uxi,time,cf,roc,etr,hod,dsr,frel,dup,het  (without CLU_KEEP_ONLY)
```
Tools: `single.py` (single-feature NDCG), `drift.py` (PSI), `cold.py` (seen/unseen check).

## 9. Lessons and risks

* **Per-cluster constants work as IDs** for a val-trained GBDT with 14K feeds. They overfit even at K=256, and smoothing does not prevent it, because the model splits on the exact value.
  If cluster-level label statistics are wanted, they belong in a model trained on train (where they did not help either) or in a neural model as embeddings, not in the val-CV LightGBM.
* **Weekend features cannot be validated on val** (val is weekdays only). Do not ship them on faith: the weekend lift moves the distribution by PSI ~1 between val and public.
* **Seed 2 of the harness base is weak** (base 0.3619 vs 0.3631-0.3642 for the other seeds): every family gets about +0.002 on s2. Use ≥ 5 seeds before believing deltas around 0.001.

---

## Verification (adversarial verifier, 2026-11-11)

Scope: the 6 KEEP features. Verifier scripts are in `<repo>/feats_v2/clu_/verify_*.py`, with logs in `data/clu_work/verify_*.log`. No clu_ code was changed, because no bug was found.

### V0. Verdict

* **KEEP all 6.** The gain is real but small: **about +0.001 NDCG** on top of the frozen base.
* **No leakage.** Drift is negligible, including weekend vs weekday. The train files are correct.
* **Correction to §0 / §5:** the 8-seed figure of **+0.00125 ± 0.00045 holds for one column order only.** Here is why:
  * The harness feeds LightGBM `feature_fraction = 0.7`, and its column sampling depends on column order.
  * In the builder's runs, the family columns were appended in the order pmi1024, pmi11024, hrw, het1, hmin, hcnt.
  * Re-running the same 5 seeds with the features in another order (hcnt, hmin, hrw, het1, pmi, pmi1) gives **−0.00006 ± 0.00054**.
  * The base OOF is identical in both runs; only the candidate model differs.
* **Honest estimates** use pre-registered seeds and random column orders:
  * permutation-robust CV: **+0.0012 ± 0.0004**;
  * Tue→Thu time split: **+0.0016 ± 0.0007**, and Thu→Tue: **+0.0016 ± 0.0006**;
  * LightGBM trained on train: **+0.0017 ± 0.0007**.
  
  All of these are positive and agree with the builder's "expected +0.0005..0.001" (if anything, slightly above it).

### V1. Leakage and asymmetry audit

| check | result |
|---|---|
| Val labels used anywhere in KEEP code (`kmeans.py`, `pmi_table.py`, `build.py` hist/pmi, `clu_common.py`) | **none**. KEEP features are label-free. |
| Train labels | not used by KEEP features. Label-based tables (`tables.py`, dropped) are per-day with `cut_tables` = cumsum over days < d for train rows and all train days for val/test. Correct. The only same-day component is the 816-cell position nuisance model. |
| Val vs test definition | identical: per-feed history kernels, plus k-means centroids and PMI tables shared by all splits. No time windows, so there is no split-edge or censoring effect. |
| Future information for val but not test (the PMI table includes test histories, which can contain val clicks, see `eda_history_leak` §1) | **negligible**: val+test histories make up **1.7%** of the transition mass. Rebuilt from **train histories only**, `clu_pmi1024_mean` / `clu_pmi11024_mean` correlate **0.99997 / 0.99996** with the stored files (val and test), and single NDCG is unchanged (0.3269 vs 0.3270; 0.3265 vs 0.3265). `verify_pmi_leak.py` |
| Correctness of the kernels | the 4 history-count features were recomputed independently with a polars group-by on val and test. NaN pattern identical; max \|diff\| = 0 (hcnt, hmin, het1), 2.6e-6 (hrw, float rounding). History row j == hpos holds for every feed. `verify_recompute.py` |
| Transition direction | T[older, newer], and the query is S[history cluster → candidate cluster]. T1 conditions on the newer event being type 1. Consistent. |
| Train rows | the PMI table includes later train histories. They carry no labels, but they hold earlier feeds' clicks as type-1 events. The cluster-level aggregate has per-user weight 1/#feeds, so the effect is negligible, and the train-trained check (V3) is positive on val. Only relevant if the final model trains on train. |

### V2. Drift (`verify_drift.py`; PSI on val deciles or integer bins plus a NaN bin; public/private boundary ts 1790497920)

| feature | val-pub | val-priv | pub-priv | val-test weekend (Sat+Sun) | val-test weekday (Mon+Tue) | weekend-weekday (test) | valTue-valThu | mean val / pub / priv |
|---|---|---|---|---|---|---|---|---|
| clu_hcnt1024 | 0.0004 | 0.0001 | 0.0004 | 0.0003 | 0.0001 | 0.0003 | 0.0002 | 14.31 / 13.88 / 14.25 |
| clu_hmin1024 | 0.0002 | 0.0001 | 0.0001 | 0.0002 | 0.0001 | 0.0003 | 0.0002 | 394.6 / 392.8 / 392.0 |
| clu_hrw4096 | 0.0003 | 0.0001 | 0.0002 | 0.0003 | 0.0002 | 0.0004 | 0.0001 | 0.588 / 0.564 / 0.578 |
| clu_het14096 | 0.0005 | 0.0001 | 0.0003 | 0.0004 | 0.0001 | 0.0004 | 0.0001 | 4.14 / 3.95 / 4.12 |
| clu_pmi1024_mean | 0.0044 | 0.0025 | 0.0006 | 0.0028 | 0.0032 | 0.0003 | 0.0004 | 0.861 / 0.815 / 0.844 |
| clu_pmi11024_mean | 0.0042 | 0.0025 | 0.0005 | 0.0027 | 0.0032 | 0.0004 | 0.0003 | 1.044 / 1.000 / 1.026 |

* Per-day means are flat from val Tue through test Tue (for example, `clu_hcnt1024`: 14.18, 14.45, 13.71, 14.04, 14.37, 14.34).
* The PMI PSI is mostly the NaN bin (feeds with no valid history): 1.44% in val vs 0.89% in test.
* The builder's drift claims are confirmed.

### V3. Gain re-measurement

| protocol | delta | SE | per-replicate |
|---|---|---|---|
| harness `--seeds 5`, features in the order hcnt,hmin,hrw,het1,pmi,pmi1 (tag `clu_verify_keep6_s5`) | **−0.00006** | 0.00054 | −0.0008 +0.0007 +0.0004 −0.0006 −0.0000 |
| harness `--seeds 5`, builder's order (tag `clu_verify_keep6_builderorder_s5`) | +0.00136 | 0.00055 | reproduces the builder's numbers exactly (+0.0009 +0.0011 +0.0045 −0.0003 +0.0006) |
| permutation-robust CV, 9 replicates (`verify_eval.py cv`: GroupKFold by user with fold seed 1000+r, LightGBM seed 1000+r, random column permutation per replicate, fresh base each time, harness params/rounds) | **+0.00120** | 0.00041 (between replicates) | −0.0006 +0.0016 +0.0010 +0.0010 +0.0006 +0.0020 +0.0033 −0.0003 +0.0022 (7 of 9 > 0) |
| time split: train on val Tue Sep 22 (7,542 feeds), test on Thu Sep 24 (6,878); 8 seeds and column permutations | **+0.00162** | 0.00068 (feed level) | 8 of 8 ≥ 0 |
| time split, reverse (Thu → Tue) | **+0.00160** | 0.00062 | 6 of 8 > 0 |
| time split, hist4 only (hcnt, hmin, hrw, het1): Tue→Thu / Thu→Tue | +0.00076 / +0.00144 | 0.00067 / 0.00062 | |
| time split, pmi2 only: Tue→Thu / Thu→Tue | +0.00086 / +0.00060 | 0.00061 / 0.00059 | |
| time split, keep5 (keep6 minus `clu_pmi11024_mean`): Tue→Thu / Thu→Tue | +0.00054 / +0.00177 | 0.00068 / 0.00063 | no better than keep6 |
| LightGBM trained on 100K **train** feeds, fixed 250 rounds, 4 feed-sample seeds, scored on val (`verify_trainlgb.py`) | **+0.00169** | 0.00072 (between seeds) | +0.0023 −0.0005 +0.0025 +0.0024. KEEP gain ranks 36-85 of 128 features |

How to read this:
* The noise from model randomness and column order in a single 5-fold run is about **0.0011-0.0012**. That is as large as the effect itself, so any one fixed-order harness number around 0.001 can land anywhere from −0.0005 to +0.0045 per seed.
* The builder chose compact6 after seeing 3-seed results in one ordering, so the +0.0022 (s3) and +0.00125 (s8) figures carry winner's-curse inflation.
* The pre-registered estimates (permutation CV, both time-split directions, train-trained) independently land at **+0.0012 to +0.0017**. The unlucky alphabetical-order harness run lands at −0.0001.
* A reasonable point estimate is **+0.001 ± 0.0005**.
* Both blocks contribute (hist4 and pmi2 are each about +0.0007..0.001 alone in the time split). The combination is the best set measured.
* **Harness recommendation (applies to every family, not only clu_):** for deltas below about 0.002, average over random column permutations (or LightGBM seeds separate from fold seeds). The fixed order inside `eval_family.py` is otherwise part of the "seed".

### V4. Train-split files and semantics

* All 6 files exist for train, val and test. They are float32 with length equal to the candidate rows: train 246,745,836, val 1,389,971, test 2,904,806.
* No all-NaN or constant columns.
* NaN rate: train 0.0000 / 0.012 for pmi; val 0.0002 / 0.014; test 0.0002 / 0.009.

Distinct values on a 2M sample:

| feature | distinct values |
|---|---|
| hcnt | 257 |
| hmin | 257 |
| hrw | 0.67M |
| het1 | 208 |
| pmi | 1.6M |

Single-feature NDCG (`verify_single.py`, random tie-break):

| feature | val (+ / −) | val Tue / Thu (best direction) | train-100K (+ / −), train labels |
|---|---|---|---|
| clu_hcnt1024 | **0.3164** / 0.2601 | 0.3172 / 0.3155 | **0.3393** / 0.2999 |
| clu_hmin1024 | 0.2596 / **0.3290** | 0.3302 / 0.3278 | 0.2956 / **0.3547** |
| clu_hrw4096 | **0.3293** / 0.2634 | 0.3306 / 0.3279 | **0.3528** / 0.2996 |
| clu_het14096 | **0.3219** / 0.2643 | 0.3232 / 0.3205 | **0.3434** / 0.3023 |
| clu_pmi1024_mean | **0.3266** / 0.2563 | 0.3259 / 0.3274 | **0.3515** / 0.2929 |
| clu_pmi11024_mean | **0.3253** / 0.2563 | 0.3243 / 0.3265 | **0.3512** / 0.2932 |
| reference: cat_cnt_all / maxsim32 / f1 | 0.3089 / 0.3488 / 0.3368 | | 0.3419 / 0.3657 / 0.3632 |
| random | 0.2889 | | 0.3241 |

* Direction is the same on train and val. The lift over random is +0.015..+0.031 on train vs +0.028..+0.040 on val, and the features are stable from Tue to Thu.
* As the builder notes, train levels are shifted (median hcnt 5 vs 2) by the production-vs-exploration traffic difference. The train-trained check in V3 shows this does not stop the features from transferring.

Redundancy:
* Max \|Spearman\| with any base feature is 0.585 (hcnt vs attn32), 0.666 (hmin vs maxsim32), 0.661 (hrw), 0.574 (het1) and 0.795 (pmi vs cos_h10). These are not duplicates of the base.
* Inside the set, `clu_pmi1024_mean` and `clu_pmi11024_mean` have Spearman **0.997**. `clu_hcnt1024` vs `clu_hmin1024` is −0.877, and `clu_hrw4096` vs `clu_het14096` is 0.877.

### V5. Per-feature verdict

| feature | verdict | reason |
|---|---|---|
| clu_hcnt1024 | **KEEP** | correct (exact recompute), label-free, PSI ≤ 0.0004, works on train and val; part of the hist4 block (+0.0008..0.0014 in the time split) |
| clu_hmin1024 | **KEEP** | same; strongest single feature in the set (0.329), highest gain in the train-trained model (rank 36-40) |
| clu_hrw4096 | **KEEP** | same; single 0.329 |
| clu_het14096 | **KEEP** | same; single 0.322 |
| clu_pmi1024_mean | **KEEP** | transductive table verified leak-free (train-only rebuild has corr 0.99997); PSI ≤ 0.0044; pmi2 adds +0.0006..0.0009 alone in the time split |
| clu_pmi11024_mean | **KEEP (redundant)** | near-duplicate of clu_pmi1024_mean (Spearman 0.997). Dropping it (keep5) measured no better than keep6 (+0.0012 vs +0.0016 mean over both time-split directions, within noise). It costs one column and carries no risk. |

**Final KEEP list:** `clu_hcnt1024, clu_hmin1024, clu_hrw4096, clu_het14096, clu_pmi1024_mean, clu_pmi11024_mean`. **Expected gain: about +0.001** (not +0.00125 or more).
