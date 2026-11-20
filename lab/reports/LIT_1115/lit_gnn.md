# LIT_1115 / lit_gnn: graph neural networks and graph-derived signals (literature scout)

Written 2026-11-15 by the graph literature scout (workflow a workflow run). CPU only, no GPU process, no install.
Work directory `nn_x/w12/lit/lit_gnn/` (two label-free scripts and their outputs). Nothing else was written or edited.

Marks: **[M]** measured by me today; **[D]** read in a project file (path given); **[W]** read today on the fetched source page;
**[S]** search-engine summary of the source only (page not opened, or not parseable); **[K]** my own knowledge, not re-verified today;
**[I]** inferred. Web queries were generic (method names, titles, venues); no project detail left the machine.

Reference points [D, `reports/CURRENT_STATUS.md` section 1]: team best public 0.3950094 (W8P20), full-data twin FD16P20 0.394892,
leaderboard #1 0.398256; paired val SE 0.0003-0.0005; the pool loses 0.0033-0.0042 under test-like stale inputs.

## 0. Verdict

1. **A GNN model is not worth a GPU hour here.** No family in the scan (LightGCN-type collaborative GNNs, graph contrastive
   variants, graph transformers, session-graph encoders, temporal graph networks, PinSage / TwHIN / OmniSage embedding
   pre-training) has a mechanism for the bottleneck. All are transductive on item nodes or need a daily graph refresh:
   at least 42.8% (public) / 45.6% (private) of test candidate rows have no interaction, hence no node [M], and the rest carry edges that are 9-11 days old.
2. **The reproducibility literature says the same thing independently of our data:** tuned EASE / RP3beta / kNN / sequential
   rules match or beat the published GNNs (RecSys 2023, ECIR 2024, TOIS 2025), and the one clean ablation of a graph embedding
   next to a sequence encoder: removing it changes the three scores by -0.0001 / -0.0001 / +0.0002 (RecSys Challenge 2025, 4th place).
3. **Graph-derived features are already covered in at least 14 forms** (section 1): co-visitation is adopted and load-bearing;
   ALS, item2vec, group-level EASE, RP3beta, Swing / IDF, EMDE, cluster transition graphs, item-to-cluster co-visitation,
   neighbour click statistics, in-feed neighbour borrowing, cross-user kNN and cross-feed consensus are all measured nulls.
4. **What the literature reports as working for cold or low-degree nodes is one mechanism only: a one-hop aggregation over
   content-similarity edges to warm nodes** (LinkedIn graph densification, Taobao GIFT, TAG-CF test-time aggregation, TwHIN
   neighbour averaging, Spotify inductive HGNN, Tubi content-only item tower). Here that reduces to features, not to a GNN.
5. **Shortlist = 2 items, both low prior (about 0.05 each), neither a GNN**: (S1) neighbour-completed pairwise co-visitation,
   (S2) lag-invariant neighbour densities as trained inputs of the popularity-light students. S1 has a CPU-only stage 0.
   Nothing ranks 3-5. If S1's stage 0 fails, the graph class is closed for this competition.
6. Measured today, label-free: pairwise co-visitation is non-zero on 11.4% / 10.2% / 9.8% of val / public / private rows;
   63-65% of cold rows have a much-shown train item at cosine >= 0.95 and that share does not fall with the lag [M].

## 1. What the project already has, by graph family

| literature family | project equivalent | status and evidence [D] |
|---|---|---|
| link-prediction heuristics (common neighbours, Adamic-Adar) | `seq_cv_invd20 / pop / norm`: distance-weighted co-occurrence count between history items and the candidate | ADOPTED, load-bearing (`reports/feat_seq_.md`; zeroing `seq_cv_pop` -0.0055 / -0.0037, `reports/EXTERNAL_SCAN_1113.md` F1) |
| transductive node embeddings (MF, LightGCN, TwHIN) | ALS f64 / f128, item2vec, group-level EASE / RDLAE, hashed item-ID table, pre-trained ID-aware encoder | nested residual 0 to -0.0005 (`reports/PRETRAIN_PLAN_1113.md` l.123-126); ID table -0.0022, -0.0057 under distillation; R3/P8 stage-0 FAIL (`reports/COVERAGE_MAP.md` rows 142, 300) |
| random walks, multi-hop item-item | RP3beta (C09), Swing / IDF / BM25 counts (C05), typed and backward co-visitation (C04, C11) | all null; RP3beta is a cold/warm trade (coldpos -0.0040, warmpos +0.0021) (`COVERAGE_MAP.md` rows 261-268) |
| graph sketch embeddings (Cleora / EMDE) | C08 EMDE multi-partition sketch | val -0.00063, holdout -0.00055 (row 265) |
| macro / cluster graph (MacGNN) | `clu_` PMI and transition tables, ICV item-to-cluster co-visitation, W25 group-level scorers, CatBoost family B block b | `clu_` label statistics -0.0001 +- 0.0007 (`reports/feat_clu_.md`); ICV -0.0004 +- 0.0011, "redundant with embedding similarity" (`reports/feat_seq_.md` l.91); block b +0.00004 nats (`reports/CATBOOST_1115/famB.md` l.109) |
| content-kNN densification for cold nodes | block c: click statistics of the 20 nearest much-shown train items; `cset`: in-feed neighbour borrowing inside the ranker; `ctx_nd`: in-feed kNN smoothing | click statistics +0.00004 +- 0.00007 nats; label-free density columns +0.00018 val / +0.00032 holdout nats (`famB.md` l.99-104); `cset` val +0.00017, holdout -0.00001 at 10 epochs (`reports/EXPERIMENT_REGISTRY.md` l.111); `ctx_nd` ADOPTED |
| inductive content-to-collaborative mapping (PinSage, CLCRec, ALDI) | organiser's item vectors (two-tower item tower, P 0.94, `reports/EMBEDDINGS_1113.md`); pre2 trajectory encoder with a no-ID head; retrieval over the user's own history | in use; ID part of pre2 +0.0008 val / 0.0000 holdout (`COVERAGE_MAP.md` row 291) |
| temporal popularity | LUO counters over train + val + test snapshots; `trd_` fresh demand / supply densities; C07 trends; hz / hz2 lag views | counters ADOPTED; `trd_` +0.0009 +- 0.0006 in a val-fit LightGBM (noise control -0.0010 +- 0.0005), -0.00035 as a residual on G3 (`reports/feat_trd_.md`, `nn_x/audit/features/findings.txt` l.254); hz closed at file level (`CURRENT_STATUS.md` log 13:10Z) |
| user-user graph, user kNN | P4 trajectory-kNN label transfer, XUSER-RIM | -0.0001 to -0.0003; closed by mechanism (`COVERAGE_MAP.md` rows 159, 195) |
| session-graph encoders | GRU, DIN, typed and surface-restricted history views against the transformer history encoder | GRU about equal, DIN -0.0018, restricted views -0.003 to -0.005 (rows 67-68, 215) |
| candidate co-exposure graph across feeds | C14 consensus, s4 similar-listings graph, seller-slot co-membership | coverage 4.5% / 0.1% / 0.1% of rows (rows 200, 231, 271) |
| graph pre-training (LightGCN / PinSage / TwHIN) | never run | closed by argument, "same information as ALS / co-visitation" (`PRETRAIN_PLAN_1113.md` l.290, l.465) |

Scope note that applies to every "residual" entry above [D, `COVERAGE_MAP.md` row 323]: a block worth +0.0085 as a trained input read
only +0.0009 to +0.0022 in residual form, so residual nulls bound an effect loosely; only a student arm read at pool level closes an input.

## 2. Measurements made for this report (label-free)

Scripts `nn_x/w12/lit/lit_gnn/m1_cold_neighbour_coverage.py`, `m2_fresh_density_coverage.py`; outputs `m1_out.txt`, `m2_out.txt`.
Inputs are existing per-row files only (`data/feats/*/{it_hist_all_luo,seq_cv_pop,seq_cv_invd20,trd_*}.npy`,
`nn_x/w12/cb/famB/feat/{c,d}_{val,test}.npy`); no label was read. "Cold" = `it_hist_all_luo == 0` and `seq_cv_pop == 0`.
Public / private split at ts 1790497920 [D, `reports/eda_shift_overlap.md`]: 8,562 public feeds of 28,595 [M].

**M1. Cold rows and their nearest warm content neighbour** [M]

| quantity | val (1,389,971 rows) | public (886,924) | private (2,017,882) |
|---|---|---|---|
| cold rows | 37.9% | 42.8% | 45.6% |
| `seq_cv_pop == 0` | 42.0% | 46.4% | 47.6% |
| pairwise co-visitation score > 0, all rows | 11.35% | 10.23% | 9.76% |
| same, among rows whose item is in the co-visitation source | 19.6% | 19.1% | 18.7% |
| cold rows: cosine to the nearest index item, p10 / p50 / p90 | 0.897 / 0.964 / 0.992 | 0.899 / 0.965 / 0.993 | 0.900 / 0.966 / 0.993 |
| cold rows with a neighbour at cosine >= 0.95 / >= 0.90 | 62.8% / 89.0% | 64.0% / 89.7% | 65.2% / 90.1% |
| cold rows with >= 5 index items above 0.95 | 39.8% | 41.1% | 42.4% |
| warm rows with a neighbour at cosine >= 0.95 | 76.5% | 76.5% | 76.8% |
| rows without pairwise co-visitation that have a neighbour >= 0.95 | 69.2% | 69.3% | 69.7% |

Index = family B's 5,862,592 much-shown train items (at most 1,500 per 4096-cluster, >= 5 train impressions), search inside the
candidate's own cluster, own item skipped [D, `famB.md` l.52]. Reading: the content edge from a cold candidate to a warm item exists
for about two thirds of cold rows at near-duplicate strength, and its availability is flat to rising with the lag, while the ID
counters go from 38% to 46% zeros. **Not measured:** whether those neighbours have pairwise co-visitation with the user's history
(the index is built from impressions, not from interactions); that is S1's stage 0.

**M2. Fresh neighbour-pooled demand (`trd_*`, same-day pool of other users' newest 10 history events)** [M]

| share of rows with value > 0 | val all / cold | public all / cold | private all / cold | train 4M-row sample all / cold |
|---|---|---|---|---|
| exact item (`trd_ex_sday`) | 0.9% / 0.0% | 0.9% / 0.0% | 0.9% / 0.0% | 1.9% / 0.0% |
| cosine >= 0.95 (`trd_n95_sday`) | 13.2% / 8.8% | 12.3% / 8.6% | 14.0% / 10.4% | 16.0% / 6.2% |
| cosine >= 0.85 (`trd_n85_sday`) | 67.1% / 58.9% | 65.3% / 58.5% | 68.4% / 62.4% | 69.8% / 51.4% |
| supply, cosine >= 0.95 (`trd_sn95_sday`) | 30.7% / 24.6% | 30.0% / 24.7% | 32.6% / 27.6% | 31.5% / 15.4% |

All 20 `trd_*` files exist for train (246.7M rows), val and test [M]. `nn/train.py`, `nn_x/abl/train.py` and `nn_x/w5/train.py` do not
reference them [M]; a 60-second recursive search of `nn`, `nn_x/{abl,w5,w12,audit,insights}` and `src` found them only in residual
scans (the search may be incomplete). Reading: a lag-0 collaborative signal exists for cold rows, dense at cosine 0.85 and sparse
(9-10%) at 0.95; the limit is the 1/37 density of test-period snapshots [D, `reports/NEXT_1115/poplag.md` l.33]. The train "cold" column
uses as-built counters, which are denser than test [D, `CURRENT_STATUS.md` log 05:02Z], so the train cold cells are not comparable.

## 3. Literature scan

Column "bottleneck" answers: does the work address stale popularity / co-visitation (9-11 days) and 43-49% interaction-free candidates?

### 3.1 Collaborative-filtering GNNs and the "are GNNs needed" literature

| work | setting | reported result | bottleneck |
|---|---|---|---|
| Anelli et al., "Challenging the Myth of Graph Collaborative Filtering", RecSys 2023 (reproducibility), arXiv 2308.00404 | 6 graph models re-run against tuned classic baselines, 5 datasets, static random splits | Recall@20 [W]: Gowalla RP3beta 0.1829, LightGCN 0.1826, UltraGCN 0.1863; Yelp-2018 RP3beta 0.0671, LightGCN 0.0629, GFCF 0.0697; Amazon-Book EASE 0.0710 = GFCF 0.0710, LightGCN 0.0419; BookCrossing RP3beta 0.0941, EASE 0.0925, LightGCN 0.0803; Allrecipes LightGCN 0.0459, EASE 0.0351 | No. Transductive, static. Shows that the item-item linear / walk baselines we already ran carry the GNN's signal |
| Ferrari Dacrema, Benigni, Ferro, "Reproducibility and Artifact Consistency of the SIGIR 2022 Recommender Systems Papers Based on Message Passing", ACM TOIS 2025, arXiv 2503.07823 | 10 message-passing papers | claims of most papers not confirmed; faulty splits and train-test leakage; baselines weaker than simple ones; on Amazon-Book "the state-of-the-art has significantly worsened" [W, abstract] | No |
| Ju et al., "How Does Message Passing Improve Collaborative Filtering?", NeurIPS 2024, arXiv 2404.08660 | LightGCN dissected; TAG-CF = one message-passing step at test time on top of plain MF | removing neighbour information at test time costs LightGCN 14.5-29.7% NDCG@20, removing the extra gradients 0.1-4.1%; MF + TAG-CF vs LightGCN NDCG@20: 8.13 vs 8.06, 6.36 vs 6.33, 9.88 vs 9.96; gain larger on the lowest-degree 30% (9.1% vs 7.1% overall) [W] | Partly: says the whole benefit is a one-hop neighbour average for low-degree nodes, obtainable without training a GNN. Not inductive: unseen items have no embedding [W] |
| Rendle et al., "Revisiting the Performance of iALS on Item Recommendation Benchmarks", RecSys 2022, arXiv 2110.14037 | tuned iALS on 4 benchmarks | "outperforms any method on at least half of the comparisons" [S] | No. Our ALS residual is 0 to -0.0005 [D] |
| Milogradskii et al., "Revisiting BPR: A Replicability Study", RecSys 2024, arXiv 2409.14217 | tuned BPR-MF | implementation defects cost up to 50%; tuned BPR beats Mult-VAE by 10% NDCG@100 on MSD [S] | No |
| Turbo-CF, SIGIR 2024, arXiv 2404.14243 | training-free polynomial low-pass filter on the item-item graph | accuracy comparable to the best GCN competitors in under 1 second [S] | No. It is a smoothed item-item co-occurrence matrix, the object we already have |
| LightGCN (SIGIR 2020), SGL, SimGCL / XSimGCL, LightGCL, graph transformers (GFormer, SIGformer) [K] | transductive ID embeddings propagated over the user-item graph | gains reported over BPR-MF baselines of the same papers | No. Same graph, same staleness, no node for a cold item |

### 3.2 Session-graph models against sequence models

| work | setting | reported result | bottleneck |
|---|---|---|---|
| Shehzad and Jannach, "Performance Comparison of Session-based Recommendation Algorithms based on GNNs", ECIR 2024, arXiv 2312.16695 | 8 GNN models (SR-GNN, GCE-GNN, TAGNN, COTREC, ...) against tuned kNN and sequential rules, same protocol | MRR@20 [W]: Diginetica STAN / SFSKNN 0.351, best GNN (COTREC) 0.335; RSC15 1/64 STAN 0.296, best GNN 0.285; Retailrocket VSTAN 0.631 = SR-GNN 0.631; GCE-GNN 0.429 / 0.285 / 0.225 / 0.236. Seeds alone move a model by up to 0.09 MRR | No |
| Zhang et al., Atten-Mixer, WSDM 2023, arXiv 2206.12781 | SR-GNN-type models dissected | graph propagation found redundant; an attention read-out without it beats the GNN baselines [S] | No |
| Betello et al., "A Reproducible Analysis of Sequential Recommender Systems", IEEE Access 2025, arXiv 2408.03873 | standardised re-runs | SASRec dominates once the model is large enough [S] | No |

Project-side [D]: the history has no timestamps and at most 256 events; GRU about equals the transformer, DIN is worse; a session graph
adds repeat-transition edges that attention with recency embeddings already reads. No mechanism for staleness.

### 3.3 Cold-start items and temporal drift

| work | setting | reported result | how it avoids the transductive problem | bottleneck |
|---|---|---|---|---|
| Borisyuk et al., "LiGNN: Graph Neural Networks at LinkedIn", KDD 2024, arXiv 2402.11139 | production GraphSAGE; embeddings fed to rankers | **graph densification**: nodes at or below the 30th out-degree percentile get artificial edges to their top-k (k about 50) content-similar nodes at or above the 90th; +0.5% val AUC (Follow Feed), +0.28% AUC (Ads). Embeddings as ranker features: Ads AUC +0.17% / +0.39%, Jobs +1.1% [W] | content-similarity edges for low-degree nodes; a nearline pipeline "to avoid stale recommendations from outdated GNN embeddings" [W] | Yes for cold in principle; staleness solved by refresh, which we cannot do. Feature-level equivalent = block c (null) and S1 |
| Hu et al., GIFT, arXiv 2202.11525 (CIKM 2022 [K]) | cold-start video CTR, Taobao | transfers ID representations **and historical statistics** from warmed-up to cold videos over physical and semantic (content-similarity) links; online CTR +6.82% [W, abstract] | inductive transfer through content edges | Yes for cold; statistics still stale. Closest published analogue of S1 |
| De Nadai et al., "Personalized Audiobook Recommendations at Spotify Through GNNs", WWW 2024, arXiv 2403.05185 | co-listening item graph + two-tower | HR@10 0.353 vs two-tower 0.231 vs content-kNN 0.164; online +46.83% new-audiobook starts [W] | inductive: "the embedding for an audiobook that has never been streamed can be generated with just the LLM features"; trained daily on a 90-day graph [W] | Cold yes, by content fallback. Our item vectors already are that fallback |
| Truong et al., "Bridging the Semantic-Collaborative Gap: An Asymmetric Graph Architecture for Cold-Start Item Recommendation", arXiv 2606.06225 (2026) | Tubi retrieval | item tower uses intrinsic features only (no IDs, no item-side neighbours); cold items retrieve "warm surrogate neighbors"; online gains stated without numbers [W, abstract only] | content-only item side removes the warm / cold asymmetry | Supports the popularity-light view we already run at weight 0.2 |
| Pembek et al., "Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization", RecSys 2025, arXiv 2507.19473 | SASRec, item = frozen content vector + norm-bounded trainable delta | cold-item HR@10 on Amazon-M2 0.509 vs 0.454 content kNN vs 0.435 content-initialised SASRec; warm 0.617 vs 0.610; no gain on Beauty [W]. No graph | cold items stay in the content space; warm items move at most cosine about 0.87 away | Would remove the "has an ID" tell of the pre2 score; gated out by the encoder-scaling closure (section 6, R8) |
| Lin et al., TDRO, AAAI 2024, arXiv 2312.09901 | cold-start under temporal feature shift | group-robust loss with a shifting factor; numbers not read [S] | reweights training groups by the shift trend | Reweighting class is closed here (ESS 2,880, `reports/SOLUTIONS_REVIEW.md`) |
| Daniluk and Dąbrowski, "Temporal graph models fail to capture global temporal dynamics", arXiv 2309.15730 | Temporal Graph Benchmark link prediction | a decayed recent-popularity counter beats the neural temporal graph models on the medium and large datasets [W, abstract; S] | no model: recent global popularity | Says temporal GNNs do not learn popularity drift; the remedy is a fresh counter, which is `trd_` (S2) |
| El-Kishky et al., TwHIN, KDD 2022, arXiv 2202.05387 | heterogeneous graph embeddings as ranker features | Ads RCE 21.23 -> 21.48, 13.53 -> 13.63, 17.11 -> 17.27; search MAP 55.7 -> 57.0 [W] | embeddings "must be updated regularly"; new vertices initialised by averaging graph neighbours [W] | Stale by construction; cold needs neighbours |
| Wang et al., FeSAIL, arXiv 2505.02844 (IJCAI 2023 version exists) | incremental CTR training | replays samples whose features went stale; numbers not read [W, abstract] | training-side staleness replay | Same family as hz / hz2 (closed at file level) |
| "Long-History User Transformers for Real-Time Ad Ranking", arXiv 2607.14331 | ad ranking | 5-day embedding lag costs 0.38 of the gain, a 6-month-old checkpoint 3.40 [S, not opened] | n/a | Indicates learned embeddings degrade slowly; counters are the fragile part |
| CLCRec (MM 2021), GAR (SIGIR 2022), ALDI (2023), Cold Brew (ICLR 2022), ColdRec benchmark [S / K] | map content to the collaborative embedding space | gains on cold items against content-only or dropout baselines | inductive through a learned content mapper | Target space has no residual value here even on warm items (ALS / item2vec 0 to -0.0005 [D]) |

### 3.4 Graph-derived features and embeddings fed to a ranker

| work | setting | reported result | bottleneck |
|---|---|---|---|
| Makeev et al., "Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025", arXiv 2508.06970 | TwHIN on a user-item graph next to a causal transformer encoder, DCN and engineered features | ablation: without TwHIN 0.7530 / 0.8146 / 0.7901 against 0.7531 / 0.8147 / 0.7899 full; without the sequence encoder -0.0029 / -0.0120 / -0.0016. "The impact of the TwHIN is relatively low due to overlapping collaborative signals with the sequential embedding" [W] | The cleanest published answer to "graph embedding next to a sequence encoder": zero |
| Badrinath et al., OmniSage, KDD 2025, arXiv 2504.17811 | graph + sequence + content contrastive embeddings as ranking features | Homefeed ranking +0.92% sitewide repins; graph rebuilt and inferred daily; entities without edges get zero-imputed neighbour features; no cold evaluation; cold-node graph augmentation listed as future work [W] | No: daily refresh, cold falls back to content |
| Chen et al., MacGNN, WWW 2024, arXiv 2401.14939 | macro nodes = clusters of users / items; macro edges = summed interaction counts | AUC +0.0033 to +0.0115, GAUC +0.0002 (Kuaishou) to +0.0125 over the best baseline; online PCTR +3.13% vs SIM; no time-aware weighting, no cold evaluation [W] | Cluster level would survive item churn, but this is `clu_` / ICV / block b, closed three times |
| Min et al., NI-CTR / Graph-masked Transformer, SIGIR 2022, arXiv 2201.13311 | attention over the local neighbourhood of the user-item pair | online CTR +21.9% against the previous baselines [S] | The ranker is already a pair-wise model over the user's local neighbourhood (retrieval over own history + co-visitation) |
| Yuan et al., ContextGNN, ICLR 2025, arXiv 2411.19513 | pair-wise scores for items in the user's subgraph + two-tower for the rest | +20% over the best pair-wise baseline on RelBench; fewer than half of the ground-truth items lie in the 3-hop subgraph; shallow item embeddings: "transductive", new items unhandled [W] | No for cold. Confirms pair-wise > two-tower, which the ranker already is; our pair-wise ID coverage is 10% of rows [M] |
| Wang, Yang, Zhang, Neural Common Neighbor with Completion, ICLR 2024, arXiv 2302.00890; Chamberlain et al., subgraph sketching (BUDDY), ICLR 2023, arXiv 2209.15486 | link prediction | message passing cannot count common neighbours; explicit pair-wise structural features are needed; **graph incompleteness reduces common neighbours and shifts their distribution between train and test; completing the missing links first repairs it** [S] | The only work that names our failure mode (the pair-wise count is missing at test time). Basis of S1 |

RelBench on the organiser's older public dataset: relational GNN 65.90 against LightGBM 53.60 AUROC on user-clicks
[D, `reports/research_1112/tools.md` l.288]; the baseline has no engineered features [K], and the task is entity classification, so it does not transfer.

### 3.5 What winning solutions of session / feed ranking competitions used

| competition | what the top solutions used | graph part |
|---|---|---|
| OTTO (Kaggle 2023) | co-visitation matrices + GBDT re-rankers, neural candidate generators [D, `reports/deep_otto_hm.md`] | 7th place: GNN 0.59874, LightGBM 0.60008, blend 0.60313 private [D, l.182]: an equal-strength second member on warm IDs in a same-period test set |
| KDD Cup 2020 Debiasing | 1st: multi-hop item-item graph for candidates + LightGBM [D, `reports/research_kaggle_solutions.md` l.54] | random-walk retrieval, not a GNN |
| RecSys Challenge 2022 (fashion sessions) | transformer / GRU ensembles + GBDT; 3rd place used 17 first-stage models [S] | 4th place: heterogeneous session GNN with side information, MRR 0.20762 [S] |
| KDD Cup 2023 (multilingual sessions) | 1st in all tasks: transformers, co-visitation matrices, product embeddings, two-stage re-ranking [S]; 10th: co-visitation candidates + LightGBM [S] | none reported at the top |
| H&M (Kaggle 2022) | 1st: recent-popularity candidates + feature engineering + GBDT; cold items "never ranked in the top 12" [S] | graph centrality features in some solutions [D, `deep_otto_hm.md` l.325] |
| RecSys Challenge 2024 (news) | transformer over the impression + LightGBM / CatBoost stacking [S] | none |
| RecSys Challenge 2025 (behaviour modelling) | 2nd: GRU autoencoder > iALS > LightFM [D, `reports/deep_seq_pretrain_cf.md` l.58]; 4th: sequence encoder + DCN + features | removing TwHIN changes the three scores by -0.0001 / -0.0001 / +0.0002 [W] |
| RecSys Challenge 2026 (conversational music) | LLM encoders + two-tower retrieval + LightGBM LambdaRank re-rankers [S] | none |
| Organiser's 2025 cup | linear autoencoders + SASRec scorers + CatBoost YetiRank [D, `reports/deep_ru_competitions.md`] | none; transfer to this task measured and closed (W25) |

Pattern: co-visitation counts plus a strong re-ranker win; a GNN appears only as a diversity member where items are warm and the test
period overlaps the graph. Neither condition holds here, and a member must be within about 0.002 of the best family to enter the pool
[D, strength gate (a), `CURRENT_STATUS.md` log 11:31], while ID-graph scorers reach 0.33-0.36 against 0.40 (EASE single 0.331, pre2 zero-shot 0.359 [D]).

## 4. Which works avoid the transductive problem, and how

A GNN trained on the Sep 6-19 interaction graph has (a) no node for 43-46% of test candidates, (b) edges 9-11 days old for the rest,
(c) a train / test asymmetry in node degree that the model learns to rely on (the same defect as the popularity inputs).

| route | works | what it needs | available here? |
|---|---|---|---|
| 1. refresh the graph daily or nearline | LiGNN, OmniSage, Spotify, TwHIN | interactions up to the request time | No. Dense snapshots end 6-9 days before the test feeds; later snapshots are at 1/37 density [D] |
| 2. content-only item side (inductive) | Spotify HGNN, Tubi Shallow-RHS, GraphSAGE / PinSage [K], content-initialised SASRec | content features for every item | Already the case: content vectors for 99.985% of candidates [D, `eda_shift_overlap.md` l.161]; popularity-light students are this view |
| 3. content-similarity edges from cold to warm nodes | LiGNN densification, GIFT, TAG-CF, TwHIN neighbour averaging, NCN completion | a warm near-neighbour per cold item | Yes: 63-65% of cold rows at cosine >= 0.95 [M]. Tested for click statistics (null); untested for pair-wise co-visitation (S1) |
| 4. cluster-level nodes | MacGNN | stable clusters | Tested three times, null |
| 5. recent-popularity counter instead of a model | PopTrack | fresh events | Only at 1/37 density, pooled over content neighbours: `trd_` (S2) |

Routes 1 and 4 are unavailable or closed; route 2 is in the pool. Routes 3 and 5 are the only places where a graph idea is not already
answered, and both are features.

## 5. Shortlist (ranked; 2 items; nothing ranks 3-5)

Priors are mine [I]. "File level" = the submitted mix; a gain g on a pool mixed at weight 0.2 moves the file by roughly 0.2-0.3 g [I].

### S1. NBR-CV: neighbour-completed pair-wise co-visitation (CPU stage 0 first)

- **Mechanism.** The pair-wise collaborative input exists on 9.8% of private rows and on no cold row [M]. For a candidate c, take its
  k nearest items that are in the co-visitation source (same 4096-cluster, cosine >= 0.90, kernel weight exp(20(cos - 1))) and
  read the existing score through them: nbr_cv(H, c) = sum_n w_n cv_invd20(H, n) / sum_n w_n, plus the max, the number of neighbours
  with a non-zero score and the kernel-weighted log count. Defined identically for cold and warm candidates (own item excluded), so
  its zero share does not move with the lag: a collaborative input without the staleness tell.
- **Literature.** NCN with completion (the distribution shift of common-neighbour counts under missing links, repaired by
  completing links first); GIFT (statistics transferred over content links, +6.82% CTR on cold videos); LiGNN densification
  (+0.28% to +0.5% AUC); TAG-CF (the gain of message passing is the one-hop neighbour read, largest on low-degree nodes).
- **For** [M]: 63-65% of cold rows and 69-70% of rows without pair-wise co-visitation have a near-duplicate warm item.
- **Against** [D]: ICV (the same idea at 1024 / 4096-cluster resolution) -0.0004 +- 0.0011 and judged redundant with embedding
  similarity; neighbour click statistics +0.00004 nats; `cset` null at 10 epochs; ALS / item2vec residual 0 to -0.0005. The item
  vectors come from a two-tower model, so cosine to history is already a smoothed co-engagement score. The untested part is narrow:
  item-level near-duplicates (about 1000 times tighter than a 4096-cluster of about 9,400 items [I]), pair-wise rather than item-level, trained input.
- **Stage 0 (CPU only, about 4-6 h with coding [I]; val, the 30K holdout and its lagged view).** Written before any number:
  (a) label-free: nbr_cv > 0 on at least 25% of rows with `seq_cv_invd20 == 0`, zero-share gap val to private-like below 0.05, else STOP;
  (b) within-feed AUC on cold rows above 0.52 after rank-residualising on `seq_max_new8` and `pre2_fe_noid`, on val AND on the lagged holdout, else STOP;
  (c) nested monotone residual on the popularity-light pool and on W16 as information only (residuals under-read trained inputs).
- **Stage 1 (lead's GPU queue, only after a PASS).** Popularity-light wide students with the nbr_* inputs, 8 seeds, same seeds as the
  reference; staleness-family witnesses (val, stale-val, pmix, lagged holdout; in-period holdout reported, not a veto): val >= 0,
  stale-val >= 0, pooled z >= 2; then the file 0.8 z(W16) + 0.2 z(pool) against FD16P20 / W8P20. Train build about 10 times the 31-minute
  co-visitation build [D, `feats_v2/seq_/log_cv_train.log`], so about 5 CPU-h [I].
- **Prior.** P(stage 0 PASS) 0.25; P(file-level gain >= +0.0003 on private) 0.05.

### S2. NBR-DENS: lag-invariant neighbour densities as trained inputs of the popularity-light students

- **Mechanism.** Two label-free families that are equally defined on cold and warm rows and do not age: (i) fresh demand and supply
  around the candidate in embedding space from the same-day pool (`trd_{kde,n85,n95,skde,sn95}_{sday,s6h}`, on disk for all splits;
  drop `trd_dsr_*`, train PSI 0.19 / 0.33 [D]); (ii) catalogue density (`kn_cos1`, `kn_cos20`; exists for val / test / 120K train feeds,
  full-train build about 30 CPU-min [I], scaled from the 191 s of block c [D, `famB.md` l.52] over the 28.7M rows of its four files [M, file sizes]). Family (i) is the PopTrack counter made inductive by a content
  kernel; family (ii) is a degree statistic of the content kNN graph (a listing with many near-identical substitutes is clicked less [D]).
- **For** [D]: `trd_` +0.0009 +- 0.0006 in a val-fit LightGBM against a noise control of -0.0010 +- 0.0005; the six `trd_kde* / trd_skde*`
  columns are the only unused columns with residual correlation above 0.03 in the CatBoost audit (`reports/CATBOOST_1115/audit.md` l.230);
  density columns +0.00018 +- 0.00015 val / +0.00032 +- 0.00013 holdout nats over the pool, and next to a tree that has them
  +0.00007 +- 0.00009 val / +0.00022 +- 0.00011 holdout (an in-period-holdout signal; val is inside 1.2 SE); distributions equal across splits [D, M].
- **Against** [D]: `trd_` -0.00035 as a residual on G3; a stage-2 fit on the 20 columns R^2 -0.007 val / 0.000 holdout; the density
  signal is "diffuse" (no compact subset keeps it); four extra inputs once cost -0.0019 val at 10 epochs (`xrel_more`); at cosine 0.95
  the fresh signal covers 9-10% of cold rows [M].
- **Test.** No informative CPU gate exists (three residual reads are on record). One 8-seed popularity-light student arm, same rule as
  S1 stage 1. Recommended only folded into an arm that runs anyway (S1 stage 1, or the next popularity-light student batch), not as its own batch.
- **Prior.** P(file-level gain >= +0.0003 on private) 0.04.

If S1 passes stage 0, run S1 + S2 as ONE student arm; a null then closes route 3 and route 5 together. If S1 fails stage 0, S2 alone
does not justify a batch.

## 6. Rejected, with the reason

| # | idea | reason |
|---|---|---|
| R1 | LightGCN-family GNN on the train interaction graph (LightGCN, NGCF, UltraGCN, SGL, SimGCL, LightGCL, graph transformers) | Transductive: no node for 43-46% of test candidates [M], stale edges for the rest. Its signal is a smoothed item-item matrix; tuned EASE / RP3beta match it [W] and here ALS, item2vec, group-level EASE and RP3beta are all 0 to -0.0005 [D]. An ID table inside the ranker costs -0.0022 / -0.0057 [D]. Graph of 88.2M items and 428M events [D, `log_cv_train.log`] |
| R2 | Session-graph encoders (SR-GNN, GCE-GNN, TAGNN, COTREC) for the history | Tuned kNN and sequential rules beat all eight on MRR [W]; propagation is redundant given an attention read-out [S]; here GRU equals the transformer and DIN loses [D]. No link to staleness |
| R3 | Temporal graph networks (TGN, TGAT, DyGFormer [K]) | Node memory is transductive; they fail to track global popularity against a decayed counter [W, abstract]; no dense event stream exists after Sep 19 to update a memory |
| R4 | PinSage / GraphSAGE / OmniSage / TwHIN pre-training for extra item features | Their gains are measured against systems without such an embedding and with daily refresh [W]. The organiser's vectors already are a production two-tower output [D]; cold nodes fall back to content. Next to a sequence encoder, removing the graph embedding costs 0.0001 or nothing [W]. Closed in `PRETRAIN_PLAN_1113.md` section 5 |
| R5 | Macro / cluster graph (MacGNN) | `clu_` label statistics -0.0001 +- 0.0007, ICV -0.0004, W25 group-level scorers +0.0002 nested, family B block b +0.00004 nats [D] |
| R6 | Pair-wise link predictors (ContextGNN, NBFNet, NI-CTR) | The ranker is already pair-wise over the user's local neighbourhood. ContextGNN is transductive on items [W]. Capacity and block changes are closed (0 of 13, JOINT, 3 blocks) [D] |
| R7 | Content-to-collaborative alignment (CLCRec, GAR, ALDI, Cold Brew, Heater) | The target space has no residual value even on warm rows [D]; the pre2 no-ID head and the two-tower vectors are content-to-behaviour maps already |
| R8 | Content-anchored bounded ID delta for the pre2 encoder (Pembek et al.) | Sound fix for the "has an ID" tell, but the channel is worth +0.00077 val at 10 epochs and its ID part +0.0008 val / 0.0000 holdout, below the +0.0015 re-open trigger [D, `COVERAGE_MAP.md` row 291]; needs re-pre-training and a re-export of 246.7M rows |
| R9 | Label propagation of click statistics over content neighbours (densification as a CTR feature, GIFT statistics) | Measured: +0.00004 +- 0.00007 nats; item exploration prior +0.0004 (z 1.4); cluster target encodings hurt [D] |
| R10 | Random-walk and multi-hop item-item scores, node2vec / EGES, Cleora / EMDE | RP3beta, EMDE, item2vec nulls; every popularity-type score re-trades cold against warm [D] |
| R11 | Cross-feed co-exposure graph, prediction consensus | Coverage 4.5% of val rows; similar-listings graph and seller slots 0.1% [D] |
| R12 | User-user graph, user kNN | 30% of test users appear in train; trajectory-kNN label transfer -0.0001 to -0.0003; "shown but ignored" items click above base rate [D] |
| R13 | A GNN as a diversity member (OTTO 7th: +0.003 blend) | Needs a member near pool strength; ID-graph scorers sit 0.04-0.07 below the pool, and tree members at 0.3865 already fail the member rule [D] |
| R14 | Graph centrality on the co-occurrence graph (PageRank, degree, Katz) | Equals popularity, with the same staleness; the content-graph version is S2 (ii) |
| R15 | Refreshing the graph with test-period events | Already done to the extent the data allow: LUO counters and the co-visitation source include val / test snapshots at 1/37 density [D] |

## 7. Not checked

- No GNN was trained or run; every statement about GNN value on this data is inferred from the project's feature-form nulls and from the literature.
- S1's key quantity (do the warm neighbours carry pair-wise co-visitation with the user's history?) is not measured; M1 measures only
  that near-duplicate warm neighbours exist. No labelled quantity was computed today.
- Opened and read: sources 1 (tables via ar5iv), 3, 7, 10, 11, 14, 19, 20, 21 (ar5iv), 22, 24. Abstract page only: 2, 12, 13, 18, 27.
  Search summary only [S]: 4, 5, 6, 8, 9, 15, 16, 17, 23, 25, 26, 28-35. Two arXiv PDFs could not be parsed by the fetch tool and were read through ar5iv instead.
- First-author names of sources 10, 11, 13 (venue), 22 are from memory [K].
- Kaggle forum write-ups, the RecSys / KDD / CIKM 2026 proceedings as a whole, and Russian-language write-ups were not searched
  systematically; competition rows marked [D] rest on the project's earlier reports.
- The 60-second recursive search for `trd_` usage may be incomplete; the three ranker `train.py` files were checked directly.
- Costs in CPU hours and all priors are estimates [I].

## 8. Sources

1. Anelli et al., "Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis", RecSys 2023. https://arxiv.org/abs/2308.00404
2. Ferrari Dacrema, Benigni, Ferro, "Reproducibility and Artifact Consistency of the SIGIR 2022 Recommender Systems Papers Based on Message Passing", ACM TOIS 2025. https://arxiv.org/abs/2503.07823
3. Ju et al., "How Does Message Passing Improve Collaborative Filtering?", NeurIPS 2024. https://arxiv.org/abs/2404.08660
4. Rendle et al., "Revisiting the Performance of iALS on Item Recommendation Benchmarks", RecSys 2022. https://arxiv.org/abs/2110.14037
5. Milogradskii et al., "Revisiting BPR: A Replicability Study of a Common Recommender System Baseline", RecSys 2024. https://arxiv.org/abs/2409.14217
6. "Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation", SIGIR 2024. https://arxiv.org/abs/2404.14243
7. Shehzad, Jannach, "Performance Comparison of Session-based Recommendation Algorithms based on GNNs", ECIR 2024. https://arxiv.org/abs/2312.16695
8. Zhang et al., "Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network", WSDM 2023. https://arxiv.org/abs/2206.12781
9. Betello et al., "A Reproducible Analysis of Sequential Recommender Systems", IEEE Access 2025. https://arxiv.org/abs/2408.03873
10. "LiGNN: Graph Neural Networks at LinkedIn", KDD 2024. https://arxiv.org/abs/2402.11139
11. "Personalized Audiobook Recommendations at Spotify Through Graph Neural Networks", WWW 2024. https://arxiv.org/abs/2403.05185
12. Truong et al., "Bridging the Semantic-Collaborative Gap: An Asymmetric Graph Architecture for Cold-Start Item Recommendation", arXiv 2026. https://arxiv.org/abs/2606.06225
13. Hu et al., "GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction", 2022. https://arxiv.org/abs/2202.11525
14. Pembek et al., "Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization", RecSys 2025. https://arxiv.org/abs/2507.19473
15. Lin et al., "Temporally and Distributionally Robust Optimization for Cold-Start Recommendation", AAAI 2024. https://arxiv.org/abs/2312.09901
16. ColdRec benchmark toolkit. https://github.com/YuanchenBei/ColdRec
17. Wei et al., "Contrastive Learning for Cold-Start Recommendation", ACM MM 2021. https://arxiv.org/abs/2107.05315
18. Daniluk, Dąbrowski, "Temporal graph models fail to capture global temporal dynamics", 2023. https://arxiv.org/abs/2309.15730
19. Yuan et al., "ContextGNN: Beyond Two-Tower Recommendation Systems", ICLR 2025. https://arxiv.org/abs/2411.19513
20. Chen et al., "Macro Graph Neural Networks for Online Billion-Scale Recommender Systems", WWW 2024. https://arxiv.org/abs/2401.14939
21. El-Kishky et al., "TwHIN: Embedding the Twitter Heterogeneous Information Network for Personalized Recommendation", KDD 2022. https://arxiv.org/abs/2202.05387
22. "OmniSage: Large Scale, Multi-Entity Heterogeneous Graph Representation Learning", KDD 2025. https://arxiv.org/abs/2504.17811
23. Min et al., "Neighbour Interaction based Click-Through Rate Prediction via Graph-masked Transformer", SIGIR 2022. https://arxiv.org/abs/2201.13311
24. Makeev et al., "Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025". https://arxiv.org/abs/2508.06970
25. Wang, Yang, Zhang, "Neural Common Neighbor with Completion for Link Prediction", ICLR 2024. https://arxiv.org/abs/2302.00890
26. Chamberlain et al., "Graph Neural Networks for Link Prediction with Subgraph Sketching", ICLR 2023. https://arxiv.org/abs/2209.15486
27. Wang et al., "Feature Staleness Aware Incremental Learning for CTR Prediction". https://arxiv.org/abs/2505.02844
28. "Long-History User Transformers for Real-Time Ad Ranking", arXiv 2026. https://arxiv.org/abs/2607.14331
29. Intel, SIHG4SR, RecSys Challenge 2022 (4th place). https://github.com/Intel-bigdata/recsys2022_intel_submission
30. NVIDIA, RecSys Challenge 2022 write-up. https://medium.com/nvidia-merlin/building-a-diverse-models-ensemble-for-fashion-session-based-recommendation-for-recsys2022-2419d2182c4c
31. NVIDIA, KDD Cup 2023 win. https://blogs.nvidia.com/blog/recommendation-systems-win ; workshop page https://kddcup23.github.io
32. Tokutake et al., "A Completely Locale-independent Session-based Recommender System by Leveraging Trained Model", KDD Cup 2023 (10th). https://arxiv.org/abs/2310.07281
33. H&M 2022 first-place summary (third-party blog). https://goldandrabbit.github.io/2022/05/15/H&M%20Recommendation%20Competition%20Top1%20%E6%96%B9%E6%A1%88
34. RecSys Challenge 2024 top-team deck. https://speakerdeck.com/unonao/jsai2025-recsyschallenge2024-you-sheng-bao-gao
35. RecSys Challenge winners page. https://recsys.acm.org/challenges/
36. KDD Cup 2020 Debiasing, 1st place code. https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place
