# LIT_1115 / int_gnn: "have we checked graph networks?" (internal audit of the record + data feasibility)

Written 2026-11-15 15:28 UTC by the internal auditor for graph methods. CPU only, no GPU process, no install, no web search.
Work directory `nn_x/w12/lit/int_gnn/` (6 scripts, a shared builder, their logs, 10 result files). Nothing else was written or edited.
The external literature is in `reports/LIT_1115/lit_gnn.md` (another agent; I read its sections 0, 1, 5, 7 and did not verify its sources).

Marks: **[M]** measured by me today (log named); **[D]** read in a project file (path given); **[I]** inferred.
Row counts: val 14,423 feeds / 1,389,971 candidate rows; test 28,595 / 2,904,806; public = test feeds with timestamp < 1790497920
(8,562 feeds / 886,924 rows), private 20,033 / 2,017,882 [M; split rule D `src/calib.py`].

## 0. Verdict

1. **Graph information: checked thoroughly. A graph neural network: never trained.** Section 1 has 27 rows of graph-type ideas.
   5 are in the model (pairwise co-visitation, item degree, an in-feed similarity graph, attention over the candidate set and over
   the user's history, item vectors that come from a two-tower model). 16 were built and closed. 6 were never run; among them a
   trained message-passing model on the interaction graph (LightGCN / PinSage / TwHIN), closed on 11-13 **by argument, with no run**
   (`reports/PRETRAIN_PLAN_1113.md` l.290 and l.465).
2. **No closure of a graph idea rests on the in-period holdout alone.** Three rows have a partial holdout dependence and are flagged
   (cset, C04 co-visitation variants, hzcv). The reverse bias is larger: the ADOPTION numbers of co-visitation and popularity are
   holdout-heavy (+0.0092 holdout against +0.0032 val).
3. **A transductive graph model can reach at most 58.7% / 53.1% / 50.8% of val / public / private candidate rows** (item present in
   any train history), 62.1% / 57.2% / 54.4% with every history on disk [M]. The other 37.9% / 42.8% / 45.6% are isolated nodes.
4. **Where it can reach, two hops already saturate:** 97% of the candidates that have a node are within two item-user-item hops of
   the user's history [M]. The territory a graph model adds over the stored pairwise co-visitation input is 46% / 42% / 40% of rows.
   In that territory a 5-step walk score has within-feed AUC 0.58-0.59, below the content-only encoder score that is already an
   input (0.64), and its residual on the W16 pool is not separable from zero (best of 42 arms +0.00046 +- 0.00032) [M].
5. **Only a content-based (inductive) construction reaches the cold rows, and the reach is there:** 96-97% of cold rows have a warm
   train item at cosine >= 0.90; for 47-48% such a neighbour is co-visited with the feed's own history [M].
   On cold val rows the neighbour-imputed walk score has AUC 0.61 (content-only encoder 0.68), partial AUC 0.545 +- 0.010 after two
   content inputs are removed, and a residual on the pool of -0.0002 to -0.0007 +- 0.0005-0.0008 (3,000 feeds) [M].
6. **Judgement [I]: no GNN run.** Expected file-level gain of a transductive GNN about 0, P(>= +0.0003 private) about 0.03, and it
   leans on the inputs that cost 0.0033-0.0042 when stale. The one unpriced form is a feature block, not a network:
   content-completed co-visitation for cold rows as trained inputs of a student arm, P about 0.05.

## 1. Record: one row per graph-type idea

Columns: what it was (graph, model, where the output entered) | built or argued | numbers | closing rule and whether it rests on
the in-period holdout alone. "Residual" = nested weight on a finished pool, which under-reads a trained input (a block worth +0.0085
as a trained input read +0.0009 to +0.0022 in residual form [D `reports/COVERAGE_MAP.md` l.323]).

### 1.1 In the model (ADOPTED)

| # | idea | graph definition, where it enters | numbers [D] | rule / holdout flag |
|---|---|---|---|---|
| 1 | Pairwise co-visitation (COVERAGE_MAP l.74) | Item-item co-occurrence within +-20 positions of de-duplicated source histories (latest train feed per non-zero user + every user-0 feed: 2.58M sequences, 456M events; val / test rows add other users' earlier val / test histories; train rows of day d use days < d). 3 of the 34 dense inputs: `seq_cv_invd20`, `seq_cv_pop`, `seq_cv_norm`, plus in-model rank and z of `invd20` (`reports/feat_seq_.md`; `nn_x/abl/train.py` l.91) | LightGBM harness +0.0016; first ranker run with 16 co-visitation + popularity inputs +0.0032 val / +0.0092 holdout (`reports/nn_report.md` l.195); 2-epoch ablation of the 8 popularity + co-visitation columns: worth +0.00234 val / +0.00743 holdout (`nn_x/abl/logs/abl_eval.txt` l.162) | val AND holdout. Not holdout-only, but the holdout part is inflated: non-zero on 21% of train rows against 11% val / 10% test (`nn_report.md` l.197); a 3-day gap simulation halves the gain (`feat_seq_.md`) |
| 2 | Item degree (popularity) | Degree of the item in the all-splits user-item history graph minus the same user's later snapshots: `it_hist_{all,et1..et4}_luo`, 5 inputs + rank and z (`src/features_luo.py`) | same ablation as row 1; the pool loses 0.0033-0.0042 under test-like stale inputs (`reports/CURRENT_STATUS.md` section 1) | as row 1 |
| 3 | In-feed similarity graph `ctx_nd` (10 inputs) | Nodes = the feed's candidates, edge weight exp(20 (cos - 1)); ONE fixed propagation step of three per-candidate scalars (cos to the user vector, max cos to the last 32 history items, f1): `knn_u`, `knn_h`, `knn_f1`, own value minus neighbourhood mean (`knn_u_dev`, `knn_h_dev`), mean cos to the 5 best candidates (`top5u`, `top5h`), highest cos to another candidate (`max`), two centroid offsets (`reports/feat_ctx_.md` section 1.2; `feats_v2/ctx_/build_ctx.py` l.240) | worth +0.00135 +- 0.00061 val / +0.00030 holdout (`abl_eval.txt` l.192) | val > 2 SE; holdout >= 0 |
| 4 | Attention over the candidate set (cosine bias) and TabR retrieval over the user's own history | Learned message passing on the complete in-feed candidate graph and on the bipartite candidate-to-history graph, keyed by content similarity | set attention +0.0020 val; retrieval +0.0034 val / +0.0029 holdout (`reports/PRETRAIN_PLAN_1113.md` section 1.3) | val AND holdout |
| 5 | Inductive content-to-collaborative vectors | Organiser's item vectors = item tower of the production two-tower (P 0.94, `reports/EMBEDDINGS_1113.md`); our pre-trained trajectory encoder with an item-ID table, used as scalar scores | encoder channel +0.00128 val / +0.00054 holdout at 2 epochs, +0.00077 val at 10; its ID part +0.0008 val / 0.0000 holdout (COVERAGE_MAP l.190, l.291) | val AND holdout |

### 1.2 Built and closed

| # | idea | graph definition, model, entry point | numbers [D] | rule / holdout flag |
|---|---|---|---|---|
| 6 | `cset`: the only learned neighbour-aggregation layer over an explicit kNN graph ever trained in the ranker | TabR copy over each candidate's 8 most similar same-feed candidates + cosine prior in the set attention; "a cold candidate can borrow its content neighbours' popularity and co-visitation inputs" (`reports/r6_design.md` l.145-154) | 2 epochs: val +0.00028, holdout +0.00090 (z 2.7); 10-epoch pair: val +0.00017, holdout -0.00001; cold positives +0.009, warm -0.0034; mixed 2-member pools no better than 2x base (`reports/EXPERIMENT_REGISTRY.md` l.107, l.111) | 10-epoch pair + family rule. **FLAG (partial):** a cold/warm-trading arm judged on val + in-period holdout, never re-read on stale-val / pmix / lagged holdout; `reports/STALE_1115/veto.md` l.70 re-read a4u only (pmix -0.00023 +- 0.00093) and classed cset as architecture (l.108). Val alone was inside noise, so not holdout-only |
| 7 | ALS (f64 / f128) and item2vec (SGNS d128) | Bipartite user-item graph on de-duplicated train + val + test histories; factor models; candidate scored against a decayed profile of the history (`reports/eda_collab_stats.md`) | single feature 0.3310 / 0.3250; LightGBM group +0.0107 / +0.0078 over an f1-f5 base; clean ALS in the val fine-tune branch about 0 (removing it reads +0.0001; `nn_report.md` l.247, l.263); nested residual on G3 0 to -0.0005, cell AUC 0.43-0.46 (`nn_x/audit/altapproach/findings.txt` 2.1 C) | val nested residual (+ pmix). No holdout used. Residual and fine-tune-branch forms only: **no train-row version exists**, so never a train-stage input |
| 8 | Group-level EASE-DAN / RDLAE / EASE (W25) | Item-group graph (4096 clusters, categories); linear autoencoders; CatBoost features | single 0.331; nested +0.0002; member rho 0.858, bar 0.3933 not met (COVERAGE_MAP l.149). Item-level EASE impossible: top-50K items cover 1.6% of val rows (`nn_x/w25/study/study.md` l.32) | val fails first |
| 9 | RP3beta whole-sequence walk (C09, l.251 / l.266) | 3-step walk item-user-item from the history, beta 0.5 / 0; non-zero on 19.8% of val rows (`nn_x/insights/ccb_1113/scratch/tester3/logs/c09_build.log`) | residual on G3: val +0.00012 +- 0.00036, holdout -0.00013, pmix -0.00056 (veto); coldpos -0.0040 / warmpos +0.0021 | CCB rule, val >= +0.0005 failed first; residual form only |
| 10 | Swing / IDF / IUF / BM25 co-visitation counts (C05, l.262) | Re-weighted pairwise counts; Swing non-zero on 1.4% of val rows (`tester2/logs/eval_covis.log`) | val -0.00034 +- 0.00031, holdout -0.00013 | val fails; residual form only |
| 11 | Co-visitation variants w5 / fwd5 / t1 / seqs + naive-Bayes log-lift (C04, l.261) | Other windows and directions of row 1 | val -0.00020 +- 0.00034, holdout -0.00030, pmix +0.00033 +- 0.00053; coldpos +0.0060 / warmpos -0.0025 | val fails first. **FLAG (weak):** a measured cold/warm trade with a positive private-mix sign; `veto.md` l.85 keeps it closed (pmix 0.6 SE) |
| 12 | Typed / backward / reference co-visitation | Edge types by event type and direction | nested on G3 all <= +0.00024 (COVERAGE_MAP l.229) | val |
| 13 | EMDE multi-partition sketch (C08, l.265) | Density sketch of the history in content space (not an interaction graph) | val -0.00063 +- 0.00036, holdout -0.00055 (z -3.0), pmix veto | val fails |
| 14 | s4 similar-listings item-to-item graph (l.200) | Directed edge A -> B from "similar listings" views: 7.76M edges, 0.8% seen by >= 2 users | coverage 0.05-0.13% of candidates; nested <= 0 (`nn_x/codes/followup/RESULT.txt`) | coverage |
| 15 | Cluster-level graphs: transition matrices, item-to-cluster co-visitation (ICV, "dense, works for cold candidates"), cluster CF lifts | Markov transitions between categories / 1024 / 4096 content clusters; history-cluster -> candidate-cluster lifts | LightGBM harness: transitions -0.0004 +- 0.0012, ICV -0.0004 +- 0.0011, label-based cluster statistics -0.0001 +- 0.0007; as a residual on G3 -0.00042 (`feat_seq_.md` l.89-90, `reports/feat_clu_.md`, `reports/EXTERNAL_SCAN_1113.md` F3) | val only |
| 16 | Nearest-train-item label statistics (CatBoost round 2, blocks c / d) | Content-kNN label propagation: 20 nearest of 5.86M much-shown train items inside the candidate's 4096-cluster | click statistics +0.00004 +- 0.00007 / +0.00006 nats (val / holdout); label-free density +0.00018 +- 0.00015 / +0.00032 +- 0.00013 nats (`reports/CATBOOST_1115/famB.md` section 2.3) | tree-track STOP rule, val and holdout both fail |
| 17 | Item-ID embedding table in the ranker | Hashed 2^22 x 32 table in candidate and history tokens; history-only variant | val -0.0022 +- 0.0008, holdout -0.0036, collapse in epoch 2 (`nn_report.md` l.196-200, l.232); under distillation at d192: pool quality dQ -0.0051 val / -0.0046 holdout (`reports/RECHECK_1114/kd_reprice.md` l.92) | both witnesses negative; trained form |
| 18 | Pre-trained ID-aware encoder as init / frozen ID table (R3 / P8) | Sequence-trained item-ID embeddings moved into the ranker | stage-0 gate FAIL: nested residual <= 0, cell AUC 0.31-0.33 (COVERAGE_MAP l.142); stage 1 never built | val gate |
| 19 | User-user graph | P4 trajectory-kNN label transfer (built); XUSER-RIM labelled kNN (closed by mechanism, no run) | P4 val -0.0001 to -0.0003 (l.195); XUSER-RIM l.159 | val; argument |
| 20 | Candidate co-exposure across feeds | C14 consensus; seller-slot co-membership | coverage 4.54% / 0.1% of rows (l.271, l.231) | coverage |
| 21 | hzcv: lagged co-visitation source for train rows | Row 1 built from a horizon-matched source | GBDT gates val +0.00032 +- 0.00071, holdout -0.00057 +- 0.00041 (`reports/FEATURES_REVIEW.md` l.120) | **FLAG:** the holdout was the only significant negative. Covered since: the family was re-opened (hz / hz2) and closed at file level on val (-0.00041 +- 0.00038, `CURRENT_STATUS.md` section 1) |

### 1.3 Argued only, never run

| # | idea | status in the record |
|---|---|---|
| 22 | **Trained GNN / graph pre-training on the interaction graph (LightGCN, PinSage, TwHIN)**, COVERAGE_MAP l.178 | **ARGUMENT, no run.** `PRETRAIN_PLAN_1113.md` l.290: "the same information as ALS / co-visitation (residual 0)"; l.465: "about 0 next to a sequence encoder (arXiv 2508.06970). High cost on a 95.6M-node graph", reopen trigger "none". The plan states "No training was run" (l.3). Its basis is row 7, which is a residual-form result |
| 23 | GNN as a pool member (OTTO 7th place: blend +0.003, `reports/deep_otto_hm.md` l.182) | noted in research, never run |
| 24 | Graph centrality on the co-occurrence graph (PageRank, Katz; `deep_otto_hm.md` section 7.7) | never built; degree is an input (row 2) |
| 25 | Session-graph history encoder (SR-GNN type) | not in the record before today; nearest runs: GRU about equal to the transformer, DIN -0.0018 (COVERAGE_MAP l.67-68) |
| 26 | Multi-hop walk or node-embedding score as a TRAIN-STAGE input | never run: rows 7 and 9 exist only as residuals / fine-tune extras |
| 27 | Item-level content-kNN completion of co-visitation for cold rows | never built before today (section 2.5-2.6 is the first measurement); the cluster-level form is row 15, the label form row 16 |

## 2. Data feasibility (measured today)

Scripts in `nn_x/w12/lit/int_gnn/`: `s1_basic.py`, `s2_graph.py`, `s3_walk.py`, `s3b_overlap.py`, `s4_content.py`, `s5_coldsignal.py`
(shared builder `g_common.py`). Logs: `s1_basic.log`, `s2_train.log`, `s2_all.log`, `s3_train.log`, `s3_all.log`, `s3b_train.log`,
`s3b_all.log`, `s4.log`, `s5.log`.

### 2.1 The interaction graph

Bipartite graph (user node, item). A user node is one non-zero user_id (union of all its history snapshots) or one user-0 feed
(user 0 is a pooled bucket). An edge = the item occurs in a history of that node, any event type, de-duplicated.
For a query feed its own user node is skipped everywhere.

| graph | user nodes | history events | distinct edges | items with degree >= 1 |
|---|---|---|---|---|
| train histories | 2,579,285 (2,455,381 users + 123,904 user-0 feeds) | 583,586,378 | 338,934,571 | 94,705,754 |
| train + val + test histories (most generous transductive graph) | 2,609,501 | 591,059,031 | 343,023,355 | 95,575,147 |

Train graph [M, `s2_train.log`]: 51.2% of items have degree 1, 31.4% >= 3, 6.7% >= 10, 0.13% >= 100, maximum 3,522;
a user node holds 133 items on average. The catalogue is huge and short-lived: half of the item nodes are leaves.

### 2.2 Candidates with >= 1, >= 3, >= 10 train interactions (share of candidate rows, all feeds)

"Interaction" = a distinct user node holds the item in a history. Skipping the feed's own node changes no figure by more than 0.0001.

| | val | public | private |
|---|---|---|---|
| train graph: >= 1 / >= 3 / >= 10 users | 0.587 / 0.423 / 0.228 | 0.531 / 0.384 / 0.209 | 0.508 / 0.369 / 0.202 |
| all-splits graph: >= 1 / >= 3 / >= 10 | 0.621 / 0.433 / 0.232 | 0.572 / 0.395 / 0.213 | 0.544 / 0.378 / 0.205 |
| cold = stored LUO count is 0 (`s1_basic.log`) | 0.379 | 0.428 | 0.456 |
| shown in a train feed (impression): >= 1 / >= 3 / >= 10 | 0.654 / 0.521 / 0.329 | 0.564 / 0.441 / 0.274 | 0.527 / 0.410 / 0.254 |
| clicked in a train feed: >= 1 / >= 3 / >= 10 | 0.197 / 0.060 / 0.013 | 0.163 / 0.049 / 0.011 | 0.149 / 0.045 / 0.010 |
| history OR impression >= 1 | 0.711 | 0.631 | 0.597 |

Cross-checks: the all-splits ">= 1" equals 1 minus the stored cold share to four digits; "seen in train histories" 0.587 and
"history or impression" 0.711 on val reproduce `reports/eda_collab_stats.md` section 2 (58.7%, 71.1%).
The "43-49% cold" of the task text is the upper range; over all rows it is 42.8% (public) and 45.6% (private) [M].

### 2.3 Hop reach through the user's own history (share of candidate rows, all feeds)

hop 1 = some other user node holds the candidate AND an item of the feed's history (whole-sequence co-visitation).
hop 2 = the candidate shares a co-visited neighbour item with a history item (path candidate - user - item - user - history item).
Counts are numbers of distinct linking user nodes.

| | val | public | private |
|---|---|---|---|
| stored pairwise co-visitation input > 0 (+-20 window) | 0.114 | 0.102 | 0.098 |
| train graph, hop 1: >= 1 / >= 3 / >= 10 | 0.212 / 0.079 / 0.018 | 0.190 / 0.071 / 0.016 | 0.180 / 0.067 / 0.015 |
| train graph, hop 2: >= 1 / >= 3 / >= 10 | 0.570 / 0.407 / 0.218 | 0.518 / 0.371 / 0.201 | 0.494 / 0.356 / 0.194 |
| train graph, hop 2 through a non-hub bridge item (degree <= 100) | 0.565 | 0.513 | 0.489 |
| all-splits graph, hop 1 >= 1 | 0.219 | 0.198 | 0.187 |
| all-splits graph, hop 2 >= 1 / >= 3 / >= 10 | 0.602 / 0.417 / 0.222 | 0.558 / 0.382 / 0.205 | 0.530 / 0.365 / 0.197 |
| of candidates with a node: hop 1 / hop 2 (train graph) | 0.361 / 0.971 | 0.358 / 0.975 | 0.354 / 0.974 |
| cold rows: hop 1 / hop 2 (both graphs) | 0 / 0 | 0 / 0 | 0 / 0 |

Feed-clustered SE of every share <= 0.0016 [M, `s2_train.log`, `s2_all.log`]. The hop-1 figure on val (0.212) agrees with the
RP3beta build of 11-13 (0.198 with a 500-occurrence cap, row 9).

Readings:
- Two hops reach 97% of the candidates that have a node. A third hop cannot add more than 3% of them. The structural bound of ANY
  transductive model on this graph is the ">= 1" row of 2.2.
- The hop-2 counts follow the degree distribution (>= 3: 0.41 against 0.42; >= 10: 0.22 against 0.23). Past hop 1, reach measures
  popularity. Popularity is already an input.
- Territory a graph model adds over the stored co-visitation input: hop 2 minus stored co-visitation = 0.457 / 0.416 / 0.397 of rows.
- A median feed's history touches 2,200 other user nodes (0.09% of the graph) and 336,000 items in one hop. 1.1-1.5% of feeds touch none.
- Lag: every reach figure falls from val to private (>= 1: 0.587 -> 0.508; hop 1: 0.212 -> 0.180).

### 2.4 Users

| share of feeds | val | test | public | private |
|---|---|---|---|---|
| non-zero user WITH a train feed | 0.304 | 0.291 | 0.303 | 0.286 |
| non-zero user without a train feed (new user) | 0.650 | 0.668 | 0.657 | 0.672 |
| user 0 (pooled anonymous bucket) | 0.046 | 0.041 | 0.041 | 0.041 |
| distinct non-zero users present in train | 0.318 | 0.303 | 0.314 | 0.298 |

[M, `s1_basic.log`; agrees with 30.4% / 29.1% in `PRETRAIN_PLAN_1113.md` section 2.] 70% of val / test feeds have no user node in the
train graph. A user representation has to be computed from the feed's own history, which is what the ranker does.
Of that history, 84% / 81% / 79% of events (val / public / private) point at an item that exists in the train graph [M].

### 2.5 Cold candidates and content-near warm neighbours

**What `ctx_nd` does and does not do [D, row 3].** It is a similarity graph among the candidates of ONE feed. It smooths three
user-affinity scalars over content neighbours and reports near-duplicate density. It reads no interaction count, no co-visitation
and no other feed. So it does not carry collaborative information from a warm neighbour to a cold candidate.
The only route for that today is the learned candidate-set attention (row 4), which sees each candidate's popularity inputs.
The explicit version of that route was `cset` (row 6): about 0 at 10 epochs.

**Within the feed (all feeds) [M, `s4.log`]:** share of cold rows that have a WARM candidate in the same feed at cosine

| | >= 0.80 | >= 0.90 | >= 0.95 | median of the max cosine |
|---|---|---|---|---|
| val | 0.599 | 0.275 | 0.120 | 0.832 |
| public | 0.605 | 0.276 | 0.120 | 0.833 |
| private | 0.595 | 0.272 | 0.118 | 0.831 |
| control: warm row to another warm row (val) | 0.753 | 0.426 | 0.211 | 0.881 |

Inside a feed only a quarter of the cold candidates have a close warm neighbour. An in-feed graph cannot serve the rest.

**In the catalogue (3,000 random rows per cell; exact search; SE about 0.004-0.008) [M, `s4.log`]:** nearest warm TRAIN item.
Bank = 24,788,629 train-history items with >= 3 users and an embedding (5,757,821 of them with >= 10 users).

| nearest warm train item at cosine | >= 0.90 | >= 0.95 | >= 0.99 | median | bank >= 10 users: >= 0.90 / >= 0.95 |
|---|---|---|---|---|---|
| cold rows, val | 0.963 | 0.775 | 0.230 | 0.977 | 0.861 / 0.591 |
| cold rows, public | 0.968 | 0.797 | 0.230 | 0.977 | 0.875 / 0.615 |
| cold rows, private | 0.971 | 0.790 | 0.229 | 0.976 | 0.891 / 0.610 |
| control: warm rows, item itself removed (val) | 0.991 | 0.905 | 0.346 | 0.985 | 0.971 / 0.817 |

**Content -> collaborative path [M, `s4.log`]:** share of cold rows for which one of the (up to 5) nearest warm train items at
cosine >= 0.90 has a hop-1 link with the feed's own history: **0.471 / 0.479 / 0.480** (val / public / private); at >= 0.95:
0.384 / 0.392 / 0.398. Today that share is 0 by construction.
Fidelity on warm control rows, where the item's own link is known: P(imputed link | own link) 0.88,
P(own link | imputed link) 0.60-0.63, P(own link | no imputed link) 0.13-0.14.

As a share of ALL test candidate rows (cold share x the figures above): cold with a warm train item at cosine >= 0.90 =
0.414 public / 0.443 private (>= 0.95: 0.341 / 0.360); cold with a warm candidate at >= 0.90 in the same feed = 0.118 / 0.124;
cold with a near neighbour that is co-visited with the feed's history = 0.205 / 0.219 [M, products of measured shares].

Reading: an inductive, content-based construction can structurally reach nearly every cold row, and half of them with a pairwise
collaborative signal. A transductive one reaches none. The reach does not fall with the lag.

### 2.6 Is there signal, beyond reach? (val, labels used for evaluation only)

These are residual-lens and AUC reads on val. They are weak witnesses (COVERAGE_MAP l.323) and no holdout view was computed.

**Warm rows: multi-hop walk on the interaction graph [M, `s3_train.log`, `s3_all.log`, `s3b_*.log`].**
s1 = 3-step walk (item-user-item) from the recency-weighted history; s2 = 5-step walk; divided by degree^beta.

| read (train graph unless stated) | s1 | s2 / deg^0.5 | ALS profile | stored co-visitation | content-only encoder score (already an input) | W16 pool |
|---|---|---|---|---|---|---|
| NDCG as the only score | 0.3388 | 0.3408 (all-splits graph 0.3444) | 0.331 [D] | 0.3289 [D] | - | 0.4016 |
| AUC, warm rows | 0.596 | 0.623 | 0.580 | 0.566 | 0.639 | 0.710 |
| AUC, warm rows WITHOUT a hop-1 link (37.6% of rows, 31.3% of positives) | 0.500 | 0.583 (s2 / deg 0.588) | 0.559 | 0.500 | 0.636 | 0.706 |
| same, after rank-residualising on `seq_max_new8` + `pre2_fe_noid` | - | 0.535 | 0.520 | - | - | - |
| Spearman with s2 / deg^0.5, warm rows | 0.68 | 1 | 0.46 | 0.48 | 0.52 | 0.46 |

Nested residual on the W16 pool (val 0.40158; z(pool) + w z(x), w chosen on 4/5 of the feeds by user-grouped folds):
7 columns x 3 transforms x 2 graphs = 42 arms. Range -0.00062 to +0.00046. Best: RP3beta-normalised s1, all-splits graph,
+0.00046 +- 0.00032 (z 1.45); s2 with a warm-rows-only z, +0.00042 +- 0.00026 (z 1.6). No arm reaches 2 SE; the maximum of 42
correlated nulls is expected near that size [I]. In every arm with a non-zero weight, feeds whose positives are all cold lose
0.002-0.016 and feeds whose positives are all warm gain 0.001-0.008: the cold/warm trade that the record shows for every collaborative residual.
Calibration in the same lens: ALS reads exactly 0 (weight 0 in every fold), stored co-visitation 0, `pre2_fe` -0.00003.

Reading: the walk is a strong stand-alone score, stronger than the adopted pairwise co-visitation and than ALS.
In the territory only a graph model could add, it is weaker than a content-only score the ranker already has, half-correlated with
it, and not visible as a residual on the pool.

**Cold rows: neighbour-imputed walk [M, `s5.log`].** 3,000 random val feeds, 107,992 cold rows, 1,358 cold positives.
For each cold row: up to 5 nearest warm train items (bank >= 10 users, cosine >= 0.90), cosine-weighted mean of THEIR walk scores
against the feed's history.

| read among cold rows (1,076 feeds with a cold positive and a cold negative) | value |
|---|---|
| coverage: imputed s1 > 0 / imputed s2 > 0 | 0.551 / 0.856 |
| AUC: imputed s1 / imputed s2 | 0.609-0.612 +- 0.009 / 0.613 +- 0.009 |
| AUC: imputed log-degree (neighbour popularity) / density (top-1 cosine) | 0.516 / 0.502 |
| AUC: content-only encoder score / max cosine to history / W16 pool | 0.677 / 0.662 / 0.751 |
| Spearman of imputed s1, s2 with the content-only encoder score | 0.51 / 0.57 |
| AUC after rank-residualising on `seq_max_new8` + `pre2_fe_noid`: imputed s1 / s2 / log-degree | 0.545-0.548 +- 0.010 / 0.544-0.545 +- 0.010 / 0.505 |
| nested residual on W16 (7 columns, z among cold rows, warm rows 0) | -0.00019 to -0.00067, SE 0.0005-0.0008; 0 of 7 positive |

Reading: the imputed pairwise score carries information on cold rows that two content inputs do not (partial AUC 0.545, 4.5 SE
above 0.5). Neighbour popularity carries none (0.505), which agrees with row 16. Against the whole pool nothing is visible, at a
power that cannot exclude +0.001. This is the one graph construction with any positive evidence today.
It is a 2-4 column feature block. It is not a network.
It corresponds to shortlist item S1 of `reports/LIT_1115/lit_gnn.md` and fills the quantity that report lists as not measured.
It is NOT that report's pre-declared stage 0: I used the whole-sequence walk instead of the windowed count, a catalogue-wide bank
instead of the candidate's own cluster, val only, and no lagged holdout.

## 3. Judgement

### 3.1 Graph forms never run

| form | why it was not run | headroom [I] |
|---|---|---|
| A. Trained transductive GNN on the interaction graph (LightGCN, TwHIN, PinSage on IDs), embeddings or scores into the ranker | closed by argument on 11-13 (row 22) | **about 0; P(file-level >= +0.0003 private) about 0.03.** Reasons, all measured: (i) no node for 42.8% / 45.6% of public / private rows; (ii) two hops already reach 97% of the rest and reach beyond hop 1 tracks degree; (iii) in its exclusive territory the walk score is weaker than, and half-correlated with, a content-only input; residual on W16 at most +0.0005 +- 0.0003 over 42 arms; (iv) 70% of test users have no node; (v) the trained forms of the same information that exist are negative or flat: ID table -0.0022 / -0.0051, encoder ID part +0.0008 val / 0.0000 holdout, four more collaborative rank inputs -0.0019 at 10 epochs (COVERAGE_MAP l.150); (vi) direction: every collaborative residual trades cold positives for warm ones, and the private set is colder and staler. Cost: per-day causal graph cuts for 3.15M train feeds, a 95M-node model, an 8-seed student arm: 2-3 GPU-days plus build |
| B. Multi-hop walk score (s2) as a train-stage input | rows 7 and 9 were only ever residuals | P about 0.03-0.05. It is the cheapest decisive closure of A: the walk kernel does 14,423 feeds in 30 s on 8 threads [M], so about 2 CPU-h for the train rows with per-day sources [I], then one 8-seed student arm. Not recommended ahead of the current queue |
| C. Content-completed co-visitation for cold rows (section 2.6, second block), as trained inputs | never built | **P(file-level >= +0.0003 private) about 0.05.** For: reach 47-48% of cold rows with a pairwise link, no decay with lag, partial AUC 0.545 +- 0.010 on cold rows. Against: residual on the pool -0.0002 to -0.0007; cluster-level form -0.0004 +- 0.0011 (row 15); neighbour click statistics null (row 16); `cset` null at 10 epochs (row 6); the item vectors already are a content-to-collaborative map (row 5). Needs a cluster-restricted neighbour search for the 38.95M train candidate items [M count]; exact search would cost about 45 CPU-h [I] |
| D. Session-graph history encoder | never proposed | P about 0.02. No measurement today. Histories are <= 256 events without timestamps, about 30% repeat an item [D `PRETRAIN_PLAN_1113.md` section 2]; transformer, GRU and DIN encoders are already compared |
| E. GNN-only ranker as a pool member | never run | none: stand-alone collaborative scores reach 0.33-0.34 against members at 0.40; trees at 0.3865 already fail the member rule [D `CURRENT_STATUS.md` section 4] |
| F. Graph centrality (PageRank, Katz) | never built | none: hop-2 counts equal degree (section 2.3); degree is an input |

### 3.2 Answer to the owner

- "Have we checked graph networks?" The graph INFORMATION, yes: 5 forms adopted, 16 built and closed. A graph NETWORK was never
  trained; it was closed by argument. Today's data support that argument on reach and on signal.
- "Do we need to try them?" No GNN. If one graph test is wanted, it is C, as input columns of a popularity-light student arm, read
  on val, stale-val, pmix and the lagged holdout (the in-period holdout is the wrong witness for it). Prior about 0.05.
- Two open points in the record, neither urgent: `cset` was never read under stale inputs (row 6); ALS / walk scores were never
  train-stage inputs (rows 7, 9, 26).

## 4. Not checked, and limits

- No GNN was trained. No GPU was used. Every statement about a GNN's value is inferred from reach, walk scores and the record.
- All signal reads are on val only. No in-period holdout, lagged holdout, stale-val or pmix read of my columns.
- The residual lens under-reads trained inputs (COVERAGE_MAP l.323). The nulls in 2.6 bound a residual, not a trained input.
- 49 nested arms were read in total (42 + 7), none pre-declared. I treat the best ones as noise.
- Edges ignore event type and order. A windowed 2-hop graph was not built; the whole-sequence figures are upper bounds for any window.
- Catalogue search: neighbours are the best item of each 100K-item bank chunk (exact top-1, approximate top-5); 16.6% of train items
  with >= 3 users have no embedding and are not in the bank; 3,000 rows per cell.
- Section 2.6 cold-row read: one random sample of 3,000 val feeds, 7 columns.
- I did not re-run `cset` or any stored model, and did not verify the sources of `lit_gnn.md`. No web search was made.
- Memory: `s4_content.py` peaked at 23.5 GB resident for one 6-minute run, above the 20 GB cap. Every other run stayed at or below
  13.4 GB. One process at a time, 12 threads.

## 5. Files

- Scripts and logs: `nn_x/w12/lit/int_gnn/` (`s1_basic`, `s2_graph`, `s3_walk`, `s3b_overlap`, `s4_content`, `s5_coldsignal`, `g_common`).
- Arrays: `reach_{train,all}_{val,public,private}.npz` (per-row degree and hop counts), `walk_{train,all}_val.npy` (s1, s2, degree),
  `s4_knn.npz` (nearest warm train items of the sampled rows), `s5_cold_val.npz` (neighbours and their walk scores for cold val rows).
