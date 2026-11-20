# EXTERNAL SCAN 11-13: winning-solution techniques vs what we ran

Date 2026-11-13, workflow a workflow run (Opus 5.5). Six researchers covered: Kaggle recsys/CTR cups; RecSys / KDD / WSDM / CIKM challenges 2015-2025; Russian cups and habr/ods write-ups; industry and literature; validation, ensembling and data usage under temporal shift; feature catalogues. That produced 211 technique rows. 18 cross-checkers then classified every row against COVERAGE_MAP, EXPERIMENT_REGISTRY, CURRENT_STATUS and the earlier scans (TRICKS_SWEEP_1113, deep_otto_hm, deep_ru_competitions, EXTERNAL_TOP1_1113, nnx_research_recsys_arch, nn_x/w25). The 8 highest-prior candidates each went to two cited-reason skeptics, and several skeptics ran CPU probes. CPU only, generic web queries, no project data left the machine.

Where things are (scratchpad, session-local; copy `ext/` to `nn_x/audit/ext_1113/` to keep it):
- cluster notes: `/tmp/scratchpad/ext/{kaggle,challenges,russian,industry,validation,features}/NOTES.md`
- probes: `.../ext/russian/probe/`, `.../ext/features/probe_*.{py,log}`, `.../ext/xcheck*/`, `.../ext/skeptic*/`
- the joined cross-check table (211 rows, each with `our_rows` evidence and a full `spec`): `.../ext/synth/checks.json`
- skeptic verdicts: `.../ext/synth/skeptics.json`

Row ids `x<check>.<item>` below index `checks.json`.

**P** (prior) = probability of >= +0.0005 val on G3 at pool level with holdout >= 0 (or the stated E/B rule). These are the cross-checkers' numbers, cut where the skeptics or fresh probes cut them. No number here was raised by the synthesis.

---

## 0. TL;DR

1. 211 technique rows from about 120 public sources (about 105 distinct mechanisms, §1): 99 PROPERLY_CLOSED or ADOPTED, 82 WEAKLY_CLOSED, 23 N/A, 7 NEW.
2. Why the 82 are weak: 26 tested a different mechanism, 19 were rejected by a skeptic without a run, 16 by a GBDT proxy, 9 in the AdamW era, 6 by the MLP loss-zoo proxy, 4 at a 2-epoch screen, 2 on one seed.
3. No external technique has an honest P above 0.035 on our data. All 8 top candidates (P 0.03-0.07 before the skeptics) failed at least one of their two skeptics.
4. Fresh CPU measurements closed 3 outright: per-user "prior error" (nested G3 −0.00005..−0.00027); user-relative f4/f5 profile (+0.00003 once popularity or pre2 is in); additive item CTR (residual over base members about −0.0002; its pop2 trigger fired at 15:57).
5. Every probe run during the scan is null or leak-class: f5 forward echo (SNA'19 "next counter": +0.00038 ± 0.00024, leak-class), earlier-impression negative feedback, history beyond 256, IDF damping, typed co-vis, category-sequence family, seller slots, Avito's within-category blender order.
6. "Rejected too early": yes for about 15 mechanisms procedurally, but none has a documented effect large enough here. Queue (§2.2): D2/EXPLO-STACK via a zero-GPU E1 scale-curve gate (0.03); XOFF exploration-row position offset (0.02-0.03); CE + BCE objective-diverse family (0.025); parallel DCN (0.02-0.025); example-age, CPU gate first (0.02); KD / pseudo-labels from G3 in the cooldown (0.02); a 16-item CPU closure batch at <= 0.02 (§2.3).
7. E[pool gain of everything in §2-§3] is about +0.0003. The catalogue does not explain #1's +0.0054: H2 0.50, H7 (Avito/organiser reference with data we lack, above all history-event timestamps) 0.12, H8 0.10, H3 0.10 (§5).
8. Code finding: `--loss both|bce` (nn_x/w5/train.py:462-467) drop fw, `--loss_norm feed` and denom; any BCE-family arm needs a new `--bce_w` flag (§2.2 R3).

---

## 1. Catalogue by kind

Each row gives: what the winners did and what they gained → how it maps to our data → our status and evidence.

Status words:
- **ADOPTED**: in G3.
- **CLOSED**: a proper NN test, a nested G3 residual or a mechanism argument.
- **WEAK**: closed only by a proxy, an argument, the AdamW era or a 2-epoch screen.
- **NEW**: never run.
- **N/A**: needs data we do not have.

Source tags: [V] primary source read this scan; [V-m] Kaggle post read via the HF mirror `Abirate/all_write_ups`; [P] verified earlier in a project report; [S] snippet only. URLs are in §7.

### 1.1 Features: history → candidate

| # | technique | source → reported gain | mapping → our status (evidence) | xref |
|---|---|---|---|---|
| F1 | Co-visitation item-item scores weighted by recency, type and distance; sum/max aggregation | OTTO'23 1st [V-m]: removing co-vis −0.0052 R@20 (same as removing the NN). OTTO 3rd Deotte [P]: 20 matrices, rules 0.575→0.590, XGB 0.601. YC'22 2nd [P]: 1/(rank+1) weighting "+10% recall". KDD Cup'23 UEC 10th [V]: directional counts | seq_cv_invd20/pop/norm, causal source → **ADOPTED** (zeroing seq_cv_pop −0.0055/−0.0037). Variants (w5/fwd5/t1/seqs) were pruned on the GBDT harness. Typed fwd5_t1, bwd5, t1share and ref_invd20 nested on G3 this scan all <= +0.00024, mostly negative (xcheck169/probe_typedcv.log) | x1.9, x1.10, x6.1, x15.2-4, x9.5 |
| F2 | Inverse-user-activity / Swing / IDF-damped co-vis | XMRec WSDM'22 2nd [V] (Swing); QDU CIKM'19 1st [V] w_u = 1/log(\|I_u\|+1); Wildberries habr 861466 [V] TF-IDF +20-30% offline; OTTO 3rd Theo `w_recsys` 1/log(1+cumcount) [V] | IDF on embedding-cos aggregation probed this scan: +(idfmax − smax) +0.00024 ± 0.00025 (z 0.96). IDF/IUF on co-vis COUNTS never built → **WEAK** | x6.2, x10.0, x16.1 |
| F3 | Back-off co-vis at cluster/category level, Markov category transitions | Avito Data Fest 2024 [P]: +0.9% users with contact online; Avito'25 1st [P]: second SASRec over categories | clu_ PMI, T, ICV → **CLOSED** (T −0.0004, ICV −0.0004; clu_ GBM over G3 −0.00042) | x15.5 |
| F4 | Learned CF similarity (BPR/ALS u2i, item2vec, EASE / EASE-DAN / RDLAE) | H&M'22 3rd [P]: BPR 0.03363→0.03510 LB; OTTO 17th [P]: w2v +0.003; Avito'25 1st [P]: RDLAE R@40 0.1994 | ID coverage falls from 74% (train) to 40% (private). ALS/i2v/EASE/W25 nested on G3 0 to −0.0005 → **CLOSED** | x2.6, x12.0, x9.7, x15.6 |
| F5 | Graph walks (RP3beta, PPR, multi-hop i2i with time/activity penalties) | Dacrema RecSys'19 [K]; aister KDD Cup'20 1st [V code, S write-up]: multi-hop i2i graph, NDCG@50_half +6% over 2nd; KDD Cup'23 5th [P] | Never built; same sparse ID graph as F4 → **WEAK** (P 0.01) | x15.7, x6.0 |
| F6 | Embedding similarity of candidate to history: max/mean/top-k/last-k/per type | OTTO 3rd [P]; VK'24 3rd [P]; Avito two-tower dot product is the top-1 SHAP feature, +2.3% target actions (habr 1004694) [P] | smax/lse/top3/per-type max, cos_u, TabR retrieval → **ADOPTED** (retrieval +0.0034 val / +0.0029 hold) | x15.8, x15.9, x4.6, x8.4 |
| F7 | Per-engagement-type target attention | Layer 6 RecSys'20 2nd [P] | Type embeddings and sim_bias 1 adopted. `--sim_bias 2` (per head x event-type bias) ran once: AdamW, single seed, stopped at 11k steps. It is a zero-code screen today (the base uses const 1: nn_x/w5/lane.sh:62) → **WEAK** (P 0.012) | x5.4 |
| F8 | Category affinity by type/recency; streak Σ2^-k; "orders since last" | Instacart'17 54th [V-m] (UP_order_strike "my strongest feature", J. Peeters blog); ONODERA 2nd [V]; Avito interest model, decayed counts [V] | In-model counts / first-recency → ADOPTED. EWMA B4 got a 2-epoch AdamW screen only (−0.00002). The category-sequence C family nested on G3 this scan: every column <= +0.0001 (xcheck169/probe_catC.log) → **WEAK** only for per-type recency | x2.8, x15.10 |
| F9 | Implicit negative feedback / skip-above / attention over seen-and-ignored items | WeChat DFN IJCAI'20 [V]: AUC 0.7824→0.7898; SNA'19 OK single NN [V]: attention over viewed posts incl. removals, logloss 0.4119→0.3873; XDM [V]; KDD12 T1 [V] | Item level covers 0.18% of rows. Embedding level probed this scan (features §3.1): skipped items behave like interest (wAUC 0.58 vs clicked 0.584); nested on G3 −0.0006..+0.0001 → **CLOSED** | x2.4, x15.11, x8.7 |
| F10 | Lifelong history beyond the window | Alibaba SIM CIKM'20 [V]: +7.1% CTR, +4.4% RPM | The cap binds on 29% of feeds; earlier snapshots add a median of 93 items. Probed: nested −0.0002..0 → **CLOSED** | x16.0 |
| F11 | Concatenated "competing items" / impression-list context | Outbrain'17 2nd [V-m]: competing ads hashed into FFM; trivago L6 [V]: "contrast target vs impression summary" | Set attention + within-feed z/rank → **ADOPTED** (set attention +0.0020/+0.0014) | x2.0, x17.8 |

### 1.2 Features: item, user, list context, transductive

| # | technique | source → reported gain | mapping → our status (evidence) | xref |
|---|---|---|---|---|
| F12 | Item popularity by event type over windows, ranked within the list | OTTO 1st [V-m]: "popularity worked well when ranked"; H&M 1st-5th [P] | it_hist_*_luo + XT_REL → **ADOPTED** (zeroing it_hist_all_luo −0.0114). Causal/lagged/hz variants and the pop-free/pop1/pop2 families CLOSED (16:01) | x16.4, x2.2 |
| F13 | One-way transductive counts over train+test | Expedia ICDM'13 5th [V]: 0.52477 vs 0.50099 NDCG@38 without | LUO counts include val/test histories → **ADOPTED** (+0.0023 val / +0.0074 hold) | x2.2, x13.7 |
| F14 | Test-period impression counts | RecSys'24 :D 1st [V]: 0.8864 → 0.7699 AUC without test impressions | xs_ and cross-user test-window counts nested <= +0.0003 → **CLOSED** | x5.6, x17.2 |
| F15 | Forward-looking / future-view / next-counter leak | TalkingData'18 1st [V-m] (next-click deltas); Outbrain 2nd [V-m]; SNA'19 1st [V] ("two nearest impressions of an object reveal the reaction in the first") | Same-user later-feed overlay: public 0.392819 = G3 (neutralised). **f5 forward echo probed this scan** (russian §3.1): train within-feed AUC 0.579 (< 6 h) vs backward control 0.498; val 0.560 ± 0.012; nested on G3 +0.00038 ± 0.00024 (z 1.6); leak-class (own-feed label echo) → **CLOSED** | x2.3, x7.4, x16.3 |
| F16 | Item CTR / COEC, Bayesian smoothing, lagged or two-step TE | Facebook ADKDD'14 [V]: historical NE 96.32% vs contextual 100%; KDD12 T2 NTU [V]: (c+αβ)/(i+β), α 0.05, β 75; Avito'24 2nd [P]; Elo 5th two-step TE [V-m]; TalkingData 5th lag-1 TE [V-m] | tte_/it_tr_ nested on G3, best it_tr_coec +0.00060 (z 1.8). pop2 (CTR replacing popularity) CLOSED at 10 epochs. The additive train-stage input was never run. Skeptic probes: CTR within-feed AUC 0.5225 at a 1-day gap → 0.501 at 8 days; residual over w5_e10_base members about −0.0002 → **WEAK, P <= 0.02** | x16.6, x2.7, x11.10, x5.3, x9.6, x8.8 |
| F17 | Pairwise-aware TE / Bradley-Terry strength from co-candidate wins | Avito'24 2nd `pos_neg_stat` [P] | Never built; parent CTR null; item exploration prior +0.0004 (z 1.4) → **WEAK** (P 0.015) | x16.7 |
| F18 | Funnel rates (contact/view, fav/view) | Avito'25 1st [P]; VK'24 3rd [P]; AIJ'23 1st [P] | Per-type counts are inputs; W25 contact ratio CLOSED | x16.8 |
| F19 | Trend slope / lifecycle | MTS Kion'22 4th [V]; Avito'24 2nd [P]; DF'24 churn 1st pace ratios [V] | trd_/life_ GBDT −0.00013; trends nested −0.00035 → **WEAK** (GBDT proxy; P 0.015) | x9.0, x16.5, x5.0 |
| F20 | Item age and its within-list rank | RecSys'24 :D [V] | f4 + z/rank → **ADOPTED** | x16.9 |
| F21 | User activity / CTR / repeat rate / diversity | Instacart [V-m]; Avito'25 [P]; VK'24 3rd [P] | Feed constants in G3; ud_ GBM +0.00056 (z 1.8) → **WEAK** (P 0.015 / 0.008) | x17.3, x17.4 |
| F22 | User-relative item attributes | H&M'22 1st [S]; NVIDIA KDD Cup'23 P(brand\|user) [P]; trivago L6 price-rank diff to last item, 2.2% gain [V] | GBDT −0.0012 (noise level); R4 −0.00173. **Skeptic leak-free pre-test this scan**: S5 +0.00325 ± 0.00051 on an f-only base, +0.00003 once popularity or pre2 is added → **CLOSED** (P 0.01) | x2.10, x17.5 |
| F23 | Per-user model residual ("prior error") | Instacart 8th [V-m]; Riiid 8th [V-m]; KKBox user offset +0.001 [V-m]; Yandex PWS [S] | **Two skeptic runs on G3 this scan** (pre-declared): 4 cues jointly −0.00005..−0.00012, summed −0.00018..−0.00027, known-user slice −0.0002..−0.0009; split-half preference stability 0.03-0.18 → **CLOSED** | x1.2 |
| F24 | Conditional ranks within a subgroup ("rank within same star rating") | trivago L6 top-20 [V]; FeatureSalad RecSys'24 [P] | rc_ family nested +0.00001; best single +0.00052 (z 1.4). The NN-input form was never run → **WEAK** (P 0.02) | x4.1, x17.9 |
| F25 | Item cross-list context / estimated production position | Expedia'13 (Owen Zhang) [K]; BlackPearl RecSys'24 2nd [P]; trivago global avg position [V] | Per-feed form = K9 → **ADOPTED** (+0.0024). Item-level ixh nested +0.00022; v5_cvix NN +0.0010 val / +0.0002 hold (1 seed, AdamW) → **WEAK** as an NN input (P 0.02) | x2.1, x4.5, x18.1 |
| F26 | List composition, candidate-candidate similarity | trivago L6 [V]; Avito L3 candidates attending each other +0.6% [V] | fcs, ctx_nd/hn/hs → **ADOPTED** (ctx_nd zeroed −0.0110) | x17.10, x17.11 |
| F27 | Conditional-probability / source-mix features | KKBox 1st [V-m] | CODES_EXPLOIT 0/34 → **CLOSED** | x2.5 |
| F28 | MNAR presence SVD, topic models | KKBox [V-m]; TalkingData'18 LDA/NMF 0.9808→0.9821 [V-m] | als_ −0.00030, clu_ −0.00042 nested → **CLOSED** | x2.6 |
| F29 | Substitution / satisfied need | Instacart 6th [V-m]; QDU CIKM'19 "previous BUY less preferred" [V] | Wrong sign: near-duplicates of contacted items are clicked MORE → **CLOSED** (the typed-rating split x7.0 is WEAK, P 0.015) | x2.9, x7.0, x8.11 |
| F30 | Latent geometry from a pairwise measure | Expedia'16 1st [V-m] (sphere fit to orig_destination_distance); hotel distance-matrix completion, arXiv 1908.07498 [V] | Location subspace within-feed AUC 0.536. The EMBEDDINGS §5.2 gate was pre-declared but never run → **WEAK** (P 0.02-0.03, CPU only) | x2.11, x17.1 |
| F31 | GBDT leaf indices → SVD → NN input | trivago L6 [V]: LSTM 0.6721→0.6750, Transformer 0.6734→0.6765 MRR; Outbrain 3rd [V-m] (leaves into FFM: single 0.69586→0.69588, blend 0.6971); Criteo 3 Idiots [V-m]; Facebook'14 | Never run in any form → **NEW / WEAK** (P 0.015) | x1.1, x4.0, x18.4 |
| F32 | EMDE multi-partition density sketch | Synerise RecSys'21 2nd [V]: −2.8% AP / −2.3% RCE when removed | clu_ (single partition per resolution) is the run relative, nested −0.00042 → **WEAK** (P 0.015) | x5.9 |
| F33 | Sequence-NN scores as ranker inputs | OTTO 1st [V-m]: −0.0053 without the NN; ARGUS [P] | pre2 → **ADOPTED** (+0.00128 / +0.00054) | x18.3, x9.1, x12.1 |
| F34 | History filtered to strong-intent events | Avito L3 habr 1090584 [V] | HS1 −0.0031, HS1b −0.0047 → **CLOSED** | x7.7, x18.6 |
| F35 | Price, ratings, seller, time-since-last-event, per-event time features | trivago [V]; H&M [P]; TalkingData [V-m]; :D `elapsed_ts` [V]; Riiid time lag [V-m]; Hawkes RecSys'16 [S]; Time2Vec (VK) [V] | No price, rating or seller fields, and no history-event timestamps → **N/A** | x16.2, x16.10, x16.11, x17.0, x3.5, x7.2, x8.3 |
| F36 | Retrieval-source one-hots | H&M 3rd [P] | Avito's generators are not exposed → **N/A** | x18.0 |

### 1.3 Labels and targets

| # | technique | source → gain | mapping → status | xref |
|---|---|---|---|---|
| L1 | Graded gain by action depth | OTTO 6th [P]: +0.0005 (graded metric) | Our metric is binary; the aux-head half (DEPTHAUX) is untested → **WEAK** (P 0.01) | x3.0 |
| L2 | Contact-weighted / fractional deal target | Avito L3 model 4 [V] ("not a game changer"); Avito search YetiRank [V] | Binary labels → **N/A** | x7.9 |
| L3 | Multi-positive 1/n weighting | Santander'16 1st [V-m] | y/Σy in the listwise CE → **ADOPTED** | x3.2 |
| L4 | Label smoothing / soft labels | Müller et al. NeurIPS'19 [K] | MLP proxy LS 0.1 +0.00007; cooldown-only arm untested → **WEAK** (P 0.015) | x11.0 |
| L5 | Soft label adjustment with an unbiased teacher | WSDM Cup'23 ULTR 3rd [V] | Our unbiased teacher (pool-only 0.382) is weaker than the student → **WEAK** (P 0.01) | x6.6 |
| L6 | Noisy-positive denoising (ADT truncated / reweighted CE) | Wang et al. WSDM'21 [V] | R-CE form ran in proxies only (ce_rw holdout −0.00119, z −3.3); T-CE never → **WEAK** (P 0.02) | x10.4 |
| L7 | Drop negatives from unattended sessions | KDD12 T1 [V]: 67.96M → 7.59M negatives | Every feed has K >= 1 → **WEAK** (P 0.007) | x3.1 |
| L8 | Post-last-click items as unexamined (down-weight / replace) | WSDM Cup'23 3rd [V] | **NEW.** CPU check this scan: 41.2% of production M>=50 negatives sit below the last positive (median 38 per feed; 23.5% of feeds have none) (xcheck121/below_last_result.txt). This contradicts "feeds end at the last click" | x6.4 |
| L9 | Category-share intent target | Avito interest model [V] | FUTWIN → **CLOSED** | x8.2 |

### 1.4 Losses

| # | technique | source → gain | mapping → status | xref |
|---|---|---|---|---|
| O1 | Softmax CE for binary relevance | Bruch ICTIR'19 [V]; DASALC ICLR'21 [P]: Web30K softmax 48.22 > every neural loss | **ADOPTED** | x10.6, x9.9 |
| O2 | Pointwise BCE / LogLoss beats ranking losses | trivago L6 [V]; RosettaAI 4th [S]; SNA'19 1st [V] (LogLoss 0.745 vs 0.725 per-user AUC, and the classification + ranking ensemble added) | As the main loss: round-1 −0.0009, zoo −0.004 → CLOSED. As a diversity member: MLP proxy plus one AdamW point → **WEAK** (P 0.025) | x4.3, x8.5 |
| O3 | LambdaLoss / NDCG-Loss2++ | Wang et al. CIKM'18 [V] | Zoo ±0.0002 → **WEAK** (P 0.01 as replacement, 0.025 as a diversity family) | x10.8, x10.5 |
| O4 | ApproxNDCG / NeuralNDCG / PiRank / ListMLE / Gumbel / xENDCG | NeurIPS'21 PiRank [V]; NeuralNDCG [V]; Avito L3 ListMLE [V] | Zoo −0.003 / −0.0017 / ListMLE −0.0052 → **CLOSED in effect** (P <= 0.005) | x10.7, x10.9-11 |
| O5 | Popularity-weighted loss + rare-item boost | aister KDD Cup'20 1st [V code]: weights 6.5:1, x(1+1/0.7) | Metric uniform over items → **N/A** | x5.11 |

### 1.5 Architecture

| # | technique | source → gain | mapping → status | xref |
|---|---|---|---|---|
| A1 | Self-attention over the candidate set | Allegro 2020 [V]; Avito L3 +0.6% [V] | **ADOPTED** | x12.9 |
| A2 | Joint transformer over [history; candidates] | Avito L3 model 5 [V]: +7.2% NDCG offline over CatBoost, +4.3% users with contacts online | JOINT wave 9: −0.00073, diversity +0.0011 → **CLOSED** | x7.5 |
| A3 | Parallel DCN / GDCN / DCNv2 cross branch | Avito L3 [V]; Yandex RecSys'25 4th [V]; LiRank +2.15% [P]; BARS CIKM'21 [V]: Criteo DNN 81.40 vs DCN 81.44 | Only ever inside DIN (F11, weaker). Never on the transformer → **WEAK** (P 0.02-0.025) | x7.6, x11.8 |
| A4 | Session-MF head (tied item table) / dot-product head | Booking WSDM'21 1st [V]: +2.1% | ID tying is N/A. An explicit MaxSim head was never run → **WEAK** (P 0.012) | x3.4 |
| A5 | Concatenate instead of sum the token components | Riiid'20 2nd/3rd [V-m] (no isolated number) | Linear concat = our sum; a nonlinear token MLP was never run → **NEW** (P 0.01) | x1.3 |
| A6 | Post-fusion multiplicative context h⊙(1+c) | NVIDIA SIGIR eCom'21 [V] | FiLM ran only as post-hoc cluster weights → **WEAK** (P 0.015) | x7.1 |
| A7 | Task-gated multi-task (PLE/MMoE) for main vs K9 | Tang RecSys'20 [K] | "PLE closed" was numeric PLE (a naming collision). Task gating never ran → **WEAK** (P 0.02, gradient-cosine pre-gate) | x12.3 |
| A8 | Masked-session-modelling aux | Layer 6 RecSys'22 2nd [V]: MSM 0.2109 vs XGB 0.1993 | E2 closed by critics without a run → **WEAK** (P 0.02) | x4.9 |
| A9 | Action-conditioned pretraining as init; multi-task encoder (next type / time delta / item, logQ) | VK'24 2nd [P]; Sber RecSys'25 2nd [V]; Yandex RecSys'25 4th [V] (churn/category −0.0025..−0.0035 without NETP or TDP; SKU flat or up) | pre2 scores ADOPTED; init transfer never → **WEAK** (P 0.02 / 0.01) | x9.8, x5.7, x5.8 |
| A10 | Contrastive SSL with feature corruption, then FT | Layer 6 RecSys'23 1st [P]: logloss 0.3297 vs 0.3471 supervised | R-Drop, ftok and dropout closed; corruption + InfoNCE never → **WEAK** (P 0.015) | x5.1 |
| A11 | Numeric encodings (Yeo-Johnson, quantile-interp bins, PLE, Fourier) | trivago L6 [V]: "+3 MRR points" from all transforms; Avito L3 128 quantile bins [V] | Only 2-epoch screens, with signature val −, holdout + (production step functions). 10 of 33 xt channels are still skewed and have no rank channel (xcheck205/xt_skew_audit.txt) → **WEAK** (P 0.01) | x4.2, x5.10, x7.8, x11.9, x18.5 |
| A12 | Event-field dropout inside history tokens | Avito interest model habr 974682 [V] | **NEW** (P 0.015) | x8.1 |
| A13 | Position / context tower detached at inference, observation dropout, gradient reversal | YouTube RecSys'19 [P]; Airbnb KDD'20 [P]; Google KDD'23 [V]: oracle logging PAL 0.3333 vs naive 0.4191 | Production rows: TRUNC −0.00048 and trunc2 −0.0004 at 10 epochs; old tower −0.0002. A non-additive position embedding with observation dropout was never run → **WEAK** (P 0.02) | x9.10, x11.3, x6.8 |
| A14 | FFM / field towers; last-query attention; reversed sequences; rare-ID hashing | Criteo/Avazu/KKBox [V-m]; Riiid 1st [V]; Booking [V] | No sparse crosses, H <= 256, order is semantic → **N/A** | x3.6, x3.7 |

### 1.6 Training and data usage

| # | technique | source → gain | mapping → status | xref |
|---|---|---|---|---|
| T1 | Random-order (exploration) data as gold + is_random flag; retrain on exploration traffic | Expedia'13 random_bool [V]; Yandex Lavka [V] GMV +0.6%; VK full logging x15 data [V] | Domain token + m02 → **ADOPTED** (+0.00117 / +0.00134). DW3, m04, pp02 CLOSED | x3.3, x9.11 |
| T2 | Examination offset on RANDOM-logged rows (XOFF) | Google KDD'23 [V]: random logging PAL 0.4103 vs naive 0.3860 | Proxies only (+0.00003; half strength +0.0004, z 1.3). Holdout CTR 0.021 at pos < 5 vs 0.0095 at pos >= 100 → **WEAK** (P 0.02-0.03) | x10.3 |
| T3 | ULTR (IPS / DLA / Regression-EM / fixed propensities) | WSDM Cup'23 1st/3rd [V]; Hager et al. SIGIR'24 [V]: "struggle to consistently improve ranking" | IPW −0.0022 → **WEAK in name, closed by the literature** (P 0.005) | x11.4, x6.9, x11.5, x6.10 |
| T4 | Meta-learned debiasing with small uniform data | AutoDebias SIGIR'21 [V]: Yahoo!R3 NDCG@5 0.645 vs 0.550 | Cheap forms adopted; meta-weights never → **WEAK** (P 0.01) | x11.6 |
| T5 | Big biased train → fine-tune on the in-distribution labelled set | NVIDIA RecSys'21 1st [P]; Synerise [V]; :D retrain +0.0097-0.0123 AUC [P] | 5-fold val FT → **ADOPTED**; full-val refit −0.00025; valcd +0.00014 | x5.5, x13.3, x3.11 |
| T6 | Recency weights / recency window / TIF | Outbrain 3rd [V-m]: future model +0.001; TIF arXiv 2311.16878 [V]: +0.0017-0.0055 AUC on Avazu, +0.0001-0.0007 on an industrial CVR set | rec33 −0.0021 / −0.0012 (z −4); the newest production regime transfers negatively to val (−0.00034, z −2.0) → **CLOSED in direction** (gentle RECW never; P 0.007) | x3.8, x13.9, x13.10 |
| T7 | Example-age input (0 at serving) | YouTube RecSys'16 [P] | Critique-only rejection (P 0.02) by subsumption into calendar context, which is a different mechanism → **WEAK** (P 0.02) | x13.11, x12.10 |
| T8 | Season / weekday-matched window; "score-as" date | Santander'16 1st/2nd [V] | Weekend specialisation DiD −0.0009 (GBDT); domain token = "score as exploration" → **WEAK** (P 0.01); the Sep-12 regime token (REGIME4) never ran (P 0.01) | x13.8, x3.9 |
| T9 | Negative down-sampling + bagging; per-user caps | TalkingData 1st [V-m]; :D <= 3 per user [P]; WSDM'23 query-frequency weights [V] | Set model / 1.5% dose → **N/A** | x1.8, x6.7 |
| T10 | Item-balanced sampling | Layer 6 RecSys'17 1st [V]: "relative gains of over 10%" | Never; hzcd is input parity, not re-weighting → **WEAK** (P 0.02) | x4.7 |
| T11 | Session augmentation (truncate / shuffle / swap target / reverse) | NVIDIA RecSys'22 3rd [V]; Booking reversed trips +0.8% [V] | HDROP TTA closed; training-time history dropout never → **WEAK** (P 0.01) | x4.10, x12.4 |
| T12 | Pseudo-labels of test in training | :D RecSys'24 1st [P]; KKBox [V-m]: "helps single, ensemble not at all" | AdamW FT-level −0.0004 → **WEAK** (P 0.02) | x1.7, x14.9 |
| T13 | Distillation from an ensemble teacher | Furlanello ICML'18 [K] | Born-again from ONE model, AdamW, 2 epochs, +0.0006 n.s. → **WEAK** (P 0.02) | x12.8 |
| T14 | Fold the selection holdout into the final fit | Jane Street 8th [V]; NVIDIA playbook [V] | pp02 elasticity about 0 → **WEAK** (P 0.03 nominal; expected about 0) | x13.4 |
| T15 | Hyper-active / anomaly user filtering | DF'24 churn 1st [V] | Users with >= 5 feeds hold 1.97% of feeds; no event rates → **N/A** | x9.3 |
| T16 | Test-time adaptation (TENT, TTT4Rec, T²ARec) | arXiv 2504.01489 [V] | SCTTA −0.00008 on the ensemble → **WEAK** (P 0.01) | x12.5 |
| T17 | Cross-feed / mixed negatives with logQ | WSDM'23 3rd [V]; MNS WWW'20 [V]; PinnerFormer [P] | Deferred; f1-f3 are missing for foreign items → **WEAK** (P 0.01) | x6.5, x12.2 |
| T18 | Drop rare items from ranker training | Rekko'19 2nd [V]: 0.0395 → 0.0429 | 41% of private candidates have no series → **N/A** | x8.10 |

### 1.7 Validation and selection

| # | technique | source → gain | mapping → status | xref |
|---|---|---|---|---|
| V1 | Gap-matched validation (drop the gap days; simulate the private gap) | Avito Demand'18 3rd [V] (days 6-9 dropped); Jane Street 8th [V] (200-day gap); IEEE-CIS 1st [V] | Late-holdout column ADOPTED 16:10 (amendment b); as-of stale holdout refuted (63.6% of feeds lose history) → **done** | x12.11, x3.8 |
| V2 | True time-backtest fold (retrain at an earlier cutoff) | M5 3rd [S] (3 windows); purged CV (López de Prado 2018) [S] | DPA T5 refuted: it cannot change any decision → **NEW, not worth it** | x13.2 |
| V3 | Per-feature time-consistency check | IEEE-CIS 1st [V]: 19 of 242 features removed, AUC 0.9363 → 0.9472 | Done for popularity and pre2 → **ADOPTED** | x13.0 |
| V4 | Adversarial validation | Upstage RecSys'23 9th [V]; Elo 10th/16th [V-m] | **ADOPTED** | x5.2, x3.10, x14.2 |
| V5 | Checkpoint-selection protocol under shift | DomainBed ICLR'21 [K]; Wild-Time NeurIPS'22 [V] | **WEAK**: @last vs [selected] varies by wave and diverges by up to 0.0007 val (wave 7: rdrop selected −0.00012 vs @last +0.00057, z 1.9). See §4 | x11.7 |
| V6 | Cold/tail slices as a veto; winner's-curse control; trust local val; hedge at the midpoint | KDD Cup'20 1st [P]; Ladder 1502.04585 [V]; Thresholdout [S]; Santander 2nd [V]; Roelofs NeurIPS'19 [S]; Home Credit'24 1st [P] | **ADOPTED** | x14.3, x14.11, x15.0, x15.1 |

### 1.8 Ensembling and stacking

| # | technique | source → gain | mapping → status | xref |
|---|---|---|---|---|
| E1 | Big-data stage 2 on OOF scores of the strong model | OTTO 1st [V-m]: NN alone 0.5151 vs GBDT over NN + features 0.5884; ARGUS arXiv 2507.15994 [V]: +2.32% as a feature vs −5.33% standalone; Booking [V]: GBDT re-rank +0.5% CV (not submitted); NVIDIA RecSys'21 [P] | E1 (30K, pre-declared): val −0.00031, holdout OOF −0.00042 (z −2.3). The 134K exploration form never ran → **WEAK** (P 0.03) | x1.0, x14.8 |
| E2 | Val-side stacking | :D [P]; Avito Demand 2nd 6-level [V] | S2: val +0.00056 → public −0.00035 → **CLOSED** | x14.7, x4.11 |
| E3 | Architecture-diverse members | Booking corr ~0.8 [V]; KKBox GBDT/NN corr 0.92, +0.004 AUC [V-m] | GRU + transformer → **ADOPTED**; DIN, GBDT, LLM, JOINT, HS1, CATGRAD too weak | x1.5 |
| E4 | Objective-diverse / data-split-diverse members | SNA'19 1st [V]: "ensemble with different data splits gave a big gain"; VK'24 5th [P]: split-seed 0.6732→0.6773; Gontijo-Lopes ICLR'22 [V] | Loss: MLP proxy only. Data split: never; a half-data member is expected −0.002..−0.004 val → **WEAK / NEW** (P 0.025 / 0.015) | x8.5, x10.5, x8.6, x14.10 |
| E5 | Group-by aggregates of first-level predictions | TalkingData 3rd [V-m]: 0.9835 → 0.9840 private; Instacart 8th/9th [V-m] | Within-feed form ADOPTED; cross-feed item consensus never → **NEW** (P 0.015) | x1.4 |
| E6 | Rank / RRF / LB-weighted blends; hill-climbing | TalkingData 1st/2nd [V-m]; Santander 2nd [V]; Caruana ICML'04 [V] | RRF −0.00069; Caruana optimism +0.0014; LB weights conflict with the frozen rule → **CLOSED** | x1.6, x14.5, x14.6, x8.9, x7.10 |
| E7 | Seed bagging, unweighted mean | KKBox "never submit a single NN" [V-m]; Jane Street [V] | **ADOPTED** (20 members) | x14.4 |
| E8 | Segment-specific level-2 models | DF'24 geo 2nd [V] | Routers closed → **CLOSED** | x9.4 |
| E9 | Joint / negative-correlation ensembles | Jeffares NeurIPS'23 [V]: learner collusion | Do not run | x12.7 |

### 1.9 Post-processing

| # | technique | source → gain | mapping → status | xref |
|---|---|---|---|---|
| P1 | Expected-metric-optimal re-ordering for correlated positives | Santander 2nd [V]: +0.2% local, about 0 LB; Instacart 4th [V-m]: F1 optimiser +0.001 | Key B −0.00000 ± 0.00032 on AdamW-era models → **WEAK** (P 0.01; free CPU rerun on G3) | x3.2 |
| P2 | Per-feed calibration, LB probing, threshold optimisation | Criteo/Avazu [V-m]; Yandex RecSys'15 1st [S] | Monotone no-op under per-feed NDCG → **N/A** | x1.6, x7.3 |

Count: 99 catalogue rows (about 105 mechanisms, since some rows bundle several): 18 ADOPTED, about 23 CLOSED (proper test, fresh measurement or mechanism), 9 N/A, 6 NEW, about 43 WEAK, almost all at P <= 0.02.

---

## 2. Weak closures: what to re-open and how

### 2.1 The skeptic round (8 highest-prior candidates, 2 skeptics each)

| candidate | original closure / why weak | skeptic outcome (new evidence) | P now |
|---|---|---|---|
| **EXPLO-STACK / D2**: stage 2 on OOF NN scores from 134K exploration train feeds (x1.0, x14.8) | Only the 30K form ran: cb26 E1, pre-declared, val −0.00031, holdout OOF −0.00042 (z −2.3). The 4.5x scale was dropped by argument (TRICKS_SWEEP:451, CURRENT_STATUS:553) | 1 ok / 1 refuted (both sides of both items). Against it: E1's early-stopping fits used 7-276 trees, with in-fold gains of only +0.0001..+0.0009. E3 already ran at 134K without the NN score: w 0.1 −0.00013 / −0.00041. In stage-2 rows, unseen items are 12% (val 29%, test 39%; skeptic-explostack/overlap.py). 55.6% of LUO mass on in-period rows comes from later snapshots (review_leakage). In ARGUS the GBDT is the stronger model with features the NN lacks; here the reverse. | 0.03 |
| **Gap-matched lens** (x12.11) | H was made the primary near-copy lens without a gap view | 1 ok / 1 refuted. Superseded by the 16:10 DPA audit: late-holdout column adopted; as-of stale H refuted. 86.9% of H rows are items trained on as candidates, 62.6% on a later day (skeptic_gapval_b/warm_check.log), so input overrides cannot make H private-like. | done |
| **F17 additive item CTR / COEC** (x16.6) | Only the replacement form (pop2) ran at train stage | 1 ok / 1 refuted. The pop2 trigger fired (Q3 V +0.00015, H −0.00023). CTR AUC decays with the gap: 0.5225 at 1 day, 0.501 at 8 days. Fresh-fit weights applied to the gap-8/10 version halve the G3 residual (+0.00051 → +0.00022). Residual over base members: −0.00085..+0.00021 (skeptic_f17/cross.log). | <= 0.02 |
| **Position/context tower detached + distillation** (x9.10) | Production-row tower −0.0002 (old recipe); PAL proxy −0.00061 | 1 ok / 1 refuted. The rkend terms ARE TRUNC/trunc2 (10-epoch FAIL). The additive pos term was always correctly indexed (model.py:630). Additive-tower distillation is a ranking no-op (a constant per feed at a fixed slot). Open only in a non-additive form with observation dropout. | 0.02 |
| **Stale-aware re-judging, PDA/TIDE** (x10.2) | Arms vetoed by a holdout now known to be biased | Refuted twice. Step 2 is hzcd at half dose (pre-declared hzcd holdout −0.00052, z −2.75). The stale-val judge is circular (it overrides the same 5 LUO columns). The dose proxy against matched-diversity controls loses holdout −0.0004..−0.0008 (skeptic-pda-tide/dose_proxy.py). TIDE's per-item quality q_i is not learnable without stable IDs. | 0.015 |
| **Per-user residual** (x1.2) | Only raw-rate / label-free forms ran | Refuted twice **by measurement** (§1.2 F23; skeptic-uresid/{probe,uresid_check}.py). 92% of an earlier feed's positives are already in the next history (eda_history_leak:9). | 0.01, CLOSED |
| **User-relative item attributes** (x2.10) | GBDT eval_family only | 1 ok / 1 refuted **by a leak-free pre-test** (skeptic_userrel_attr/mech_check{,2,3}.log): absorbed by popularity or pre2 | 0.01, CLOSED |
| **Train-side large-OOF stacking** (x14.8) | Same family as EXPLO-STACK | 1 ok / 1 refuted. The kill rule as written cannot fire: a 5-fold mean beats a single member by about +0.0035 without any GBDT. | 0.03 (merged into D2) |

The skeptic verdicts in full: `ext/synth/skeptics.json`.

### 2.2 Re-open queue (proper tests), ordered by value per cost

All GPU arms: the GPUs are allowed and have been idle since 16:01; thread caps as in lane.sh.
- Pre-declare the rule with `date -u` before any number.
- Use the E/B rules with the 16:10 amendments: pmix is a veto at −1 SE only; the late and early holdout columns are information; for popularity-type contrasts the holdout is a veto at −1 SE.
- Any submission candidate passes `src/presub_gate.py`.

**R1. D2 / EXPLO-STACK, zero-GPU gate first** (x1.0, x14.8; P 0.03)
- *Original closure:* E1 at 30K, negative. The 134K form was dropped by argument.
- *Why weak:* the scale was never tested. The external wins (OTTO, ARGUS) are of the right order but in the opposite strength regime.
- *Gate (CPU, ~2 h, reuses the nn_x/cb26 X_hr caches):*
  - (a) rerun E1 on 25/50/100% of trH feeds with FIXED trees (50/200/500), 3 seeds, no early stopping;
  - read the paired per-feed listwise NLL delta vs β·z(pool) on val and on trH-OOF (lower variance than NDCG; Spearman(NDCG, −CE) 0.95);
  - (b) the discriminator: E1 with the popularity family removed from the 174 columns.
- *GO iff* NLL at 30K >= 0 AND the gain per doubling > 2 SE AND (b) > 0 on val and trH. Otherwise record "E1 scale curve flat" and CLOSE D2 for good.
- *On GO, the EXPLO-STACK spec with the skeptics' fixes:*
  - residual form: CatBoost baseline β·z(OOF half-member), learn Δ only;
  - apply stage 2 to each half member's own score, then average (never feed a member mean into a single-model-trained stage 2);
  - M>=50 only, with G3 kept on M<50;
  - re-weight stage-2 feeds to val's unseen-item share;
  - comparator = the pass-through half-member mean;
  - free kill K0b on trH: Δ applied to G3 >= +1 SE before reading val;
  - fixed iterations from internal user-grouped CV.
- *Cost of the full arm:* ~3 GPU-h NN (4 half members) + 0.3 GPU-h scoring + 2.5 h CatBoost + 0.5 day plumbing.
- The half members double as the data-split diversity read (x8.6, §3 N6) and as arm (b) of the per-user residual (now CLOSED; skip it).

**R2. XOFF, examination offset on exploration rows only** (x10.3; P 0.02-0.03)
- *Original closure:* LightGBM/MLP proxy (+0.00003; half strength +0.0004, z 1.3). It was gated on DOMAIN-DIAG, which asks a production-row question.
- *Why weak:* proxy only. Random logging is exactly the regime where Google KDD'23 finds PAL helps (+6.3%).
- *Test:*
  - Step 0: add kind `xoff` to solvers/loss_zoo.py, 6 seeds, futility only (val >= 0, holdout >= 0, M>=50 >= 0).
  - Step 1: CD arm B:xoff, offset 0.5·1[p_f > 0.5]·g_noend[clamp(pos, 0, 299)] on the MAIN logits of training batches only. Never on K9 or eval. NO end-of-feed term: A3 cost −0.0011 holdout, z −3.7.
  - Roots 101-104 vs m02ref, B rule.
- *Cost:* 30 min CPU + ~2.7 GPU-h.
- *Caveat:* exploration rows are about 4% of plateau batches and 2% of the cooldown mix, so the dose is small.

**R3. Objective-diverse member family: CE + BCE term** (x8.5, x10.5; P 0.025)
- *Original closure:* MLP zoo ce_bce −0.00128 (about −0.0003 real at the measured ~4x proxy exaggeration). Round-1 AdamW 2-epoch base+bce pair −0.00033 ± 0.00059 at corr 0.974.
- *Why weak:* no Muon/WSD 10-epoch member with a pointwise term exists. External: SNA'19 1st (classification + ranking ensemble added); Gontijo-Lopes ICLR'22 (objective divergence → less correlated errors).
- *Test:*
  - New flag `--bce_w 5`: loss = the recipe's listwise loss (with fw, `--loss_norm feed`, denom) + 5·mean-per-feed BCE.
  - **Do not use `--loss both`** (nn_x/w5/train.py:467 drops fw / loss_norm / denom).
  - 10-epoch pair s951/952 vs w5_e10_base. PASS on the E rule OR on the diversity clause (d1 member val and holdout >= −0.0005; d2 ρ_bb − ρ_x >= +0.005; d3 2-member pool >= −0.0005). Then 953/954 + n2cpv3 FT + decide7.
- *Cost:* ~9 GPU-h.
- NDCG-Loss2++ (x10.5) is the second objective candidate, as a 2-epoch screen with a diversity margin >= +0.003. Run it only if BCE fails d2.

**R4. Parallel DCN / GDCN branch** (x7.6, x11.8; P 0.02-0.025)
- *Original closure:* DCNv2 only inside the DIN family F11, whose failure is DIN's. GDCN was "parked" (nnx_design:330).
- *Test:* `--dcn_par 2` (two rank-32 GDCN layers on [LN(c_final); X_std], zero-init output added in parallel to the MLP head, Muon on U/V/G). 2-epoch s23/24 vs abl_base under SCALE-UP, then a 10-epoch E-rule pair with the diversity clause.
- *Cost:* ~1 h code + 1.6 GPU-h screen + 9 GPU-h pair.
- BARS CIKM'21 says explicit crosses roughly equal a tuned DNN, hence the low P.

**R5. Example-age input, CPU gate first** (x13.11, x12.10; P 0.02)
- *Original closure:* critique-only (P 0.02) by subsumption into calendar context.
- *Why weak:* the mechanism is not calendar. Train-row input semantics drift over the 14 days (it_hist zero rate 17.5% → 29.7%), and a model with no time handle learns the window average.
- *Stage 0 (CPU ~1 h):* per-day within-feed sAUC slope of it_hist_all_luo, it_hist_et1_luo, seq_cv_invd20 and pre2_fe_s on random-like M>=50 train feeds, plus act/exp of the w5_e10_base TO holdout predictions on zero rows by day. **GO iff** a slope >= 3 SE AND the act/exp trend >= 2 SE (i.e. set attention has NOT already inferred the day).
- *Stage S:* the non-extrapolating `ex_depth` variant (days of source before the feed / 12; 1.0 for val/test), 2-epoch screen, then the E rule.
- *Cost:* 1 h CPU (+1.6 GPU-h on GO).

**R6. KD and pseudo-labels from G3 in the cooldown** (x12.8, x14.9, x1.7; P 0.02 each)
- *Original closure:* born-again KD from ONE AdamW model at 2 epochs (+0.0006 n.s.); test pseudo-labels at FT level in the AdamW era (−0.0004).
- *Test:* CD harness, roots 101-104 from @77500 vs m02ref, two arms:
  - (A) KD w 0.5, τ 1 to z(G3-TO) on train feeds, cooldown only;
  - (B) the 28.6K test feeds at 2% of cooldown batches with soft targets softmax(z(G3-TO)), K9 off on them, exploration domain token.
- **POOL clause:** G3 with the 4 students replacing their m02 counterparts must reach val >= +0.0005 and holdout >= 0, and the cross-seed corr must not rise by more than 0.005 (KKBox: "the lost diversity just makes the results after ensemble not improving at all").
- Presub has-later probe on (B).
- *Cost:* ~2.7 GPU-h per arm.
- *Expected outcome:* member up, pool flat.

**R7. Checkpoint-read protocol + one replication** (x11.7; P 0.02)
- *Zero-cost rule (adopt now):* see §4.1.
- *Replication:* the only legitimate check of whether the protocol hid a gain is fresh roots 105-108 for rdrop_cd 1.0 and sam_cd 0.05 (cd_arms8.py exists) vs m02ref.
  - Wave 7 @last: rdrop +0.00057 (z 1.9; M>=50 z 3.1), sam +0.00039.
  - PASS iff @last val > +2 SE (Holm over 2) AND [selected] val >= 0 AND holdout >= 0 AND pool holdout >= +1 SE.
  - Closed arms are NOT re-judged on their old roots (the HOLD-CLEAN rule).
- *Cost:* ~9 GPU-h.

**R8. ixh as an NN input** (x4.5, x18.1; P 0.02)
- *Original closure:* GBDT gate R3 +0.00012 and stale −0.00109; one AdamW single-seed NN read v5_cvix (+0.0010 val / +0.0002 hold).
- *CPU pre-gate:*
  - NaN-share and within-feed AUC parity between p_random>0.9 train rows and val within 0.01 (train rows are 63.9% NaN vs 35-47% on val/test);
  - Spearman(ixh_good, K9 head) < 0.8.
- *Then:* a 2-epoch screen with stale-val (ixh_stale.py cutoff Sep 18) >= −1 SE, then the E rule.
- *Cost:* 30 min CPU + 1.6 + 9 GPU-h.

**R9. Cheap screens, 1.6 GPU-h each, run only if lanes would otherwise idle:**
- `--sim_bias 2` (x5.4; zero code; P 0.012);
- within-(feed, category) relative inputs `--xrel_cat` (x4.1; 29 GPU-side channels; P 0.02; the xrel_more −0.0019 precedent is the kill prior);
- non-additive position embedding with observation dropout (x9.10 skeptic fix). Its 0-GPU gate comes first: does G3's residual on the 30K random-like holdout correlate with shown rank / rkend beyond the known Spearman +0.005 ± 0.001? Skip the arm if not. P 0.02; 4 pairs are needed for power (~18 GPU-h).

**R10. ADT truncated CE** (x10.4; P 0.02)
- CPU loss zoo first (rce_b10/b25, tce10, tcepos; 6 seeds; continue iff val > +1 SE AND holdout >= 0 AND M<50 >= −1 SE). ce_rw failed exactly the holdout clause.
- Then cooldown arm B:tce, ~2.7 GPU-h.

### 2.3 CPU closure batch (pre-declare as ONE batch; Holm over the batch or a 2.5 SE bar)

About 15 looks at nominal 2 SE give about 0.3 expected false passes. Harness: s04_resid_probe / x02 / altapproach nested user-grouped on G3, plus the fixed-weight holdout transfer, the pmix veto, and the s04 noise band (−0.00047..+0.00022). Report each next to the band.

| item | xref | why weak | build / test | CPU | P |
|---|---|---|---|---|---|
| GBDT-OOF score and leaf-SVD64 (Stage 0) | x1.1, x4.0 | skeptic, no run | LightGBM lambdarank 500 trees on cb26 XM rows; pred_leaf → TruncatedSVD(64); nested [z(G3), leafSVD] vs control [z(G3), score]. Caveat PRETRAIN_PLAN:312: the nested lens under-reads NN-input value (pre2 fails it but is worth +0.0012), so a Stage-0 fail kills only the residual form | 2-3 h | 0.015 |
| Streak / EWMA + per-type same-category recency (e1, e4, e2/e3) | x2.8 | AdamW 2-epoch | B4 set + Σ_{k<16} 2^-k·1[same cat] + per-type first-recency | 45 min | 0.01 |
| Latent geometry (pre-declared EMBEDDINGS §5.2 gate + f2-supervised 16-d location) | x2.11 | skeptic, no run | s08-s10 through nn_x/emb/usability; plus cos(W e_c, p_u^last8) − cos(·, p_u^all), the f2 residual, and the contact-location share | 2 h | 0.02 |
| Co-vis variants w5/fwd5/t1/seqs + OTTO-7th NB log-lift | x1.10 | GBDT pruning, AdamW | feat_covis.py rebuild, same as-of source | 2-3 h | 0.01 |
| IDF / IUF / Swing / BM25 co-vis counts | x6.2, x10.0, x16.1 | never on counts | copy of feat_covis.py; weight 1/log(2+pop_x) or 1/log(2+L_s) | 1.5-2 h | 0.015 |
| Bradley-Terry / first-choice item strength | x16.7 | skeptic, no run | sparse L-BFGS on train p_random>0.5 + val OOF folds, leave-user-out | 1 h | 0.015 |
| Item trend / lifecycle at a gap-matched as-of | x9.0 | GBDT proxy | ts_train_stream.py: it_n7, it_trend, it_slope7, it_q95gap, exhaustion; train as-of = day start − 5 d | 1 h | 0.015 |
| EMDE multi-partition sketch | x5.9 | relative only | 32 partitions x 6 projections, recency-weighted per event-type group | 2 h | 0.015 |
| RP3beta whole-sequence walk | x15.7 | never built | causal Index, latest 500 source occurrences | 1-2 h | 0.01 |
| User aggregates close-out (F26 + F27) | x17.3, x17.4 | GBDT proxy | rerun inter_resid.py on val AND holdout (as-of tte_*) | 1 h | 0.015 |
| Typed implicit rating cvx_g1/g2/g3 | x7.0 | GBDT proxy | co-vis accumulator split by the user's history event type | 1 h | 0.015 |
| Metric-optimal post-processing key B on G3 | x3.2 | AdamW-era models | 05_kappa_oracles.py key B on ens_G3_final_val + holdout; deterministic post-processing if it passes | 1 h | 0.01 |
| Pending-favourite (saved, not contacted) | x8.11 | different mechanism | odd n_e4, no contact/delivery on the same item, within 256 | 15 min | 0.01 |
| Cross-feed prediction consensus | x1.4 | NEW | mean TO-branch G3 z of the item over other users' val+test feeds; STOP if coverage < 10% (only 220K items are shared between val and test) | < 1 h | 0.01 |
| PLE / PCGrad pre-gate | x12.3 | naming collision | gradient cosine main vs K9 on w5_e10_base at 60k and @last; PROCEED iff median cos < 0 on >= 1/3 of matrices | 0.2 GPU-h | 0.02 |
| REGIME4 Step 0 (kill-only GBDT arms on the Sep-12 production step) | x3.9 | never run | 07_gbdt_arms.py; close if no arm moves proxy val > 1 SE | 1 h | 0.01 |

### 2.4 Weak closures to leave closed (reason)

- **Gap lens, true backtest fold** (x12.11, x13.2): superseded by the DPA audit (16:10). The late-holdout column is adopted; T4 and T5 were refuted there.
- **PDA/TIDE re-judging** (x10.2): refuted twice (§2.1).
  - The only faithful PDA test is a 0.6 GPU-h 2-epoch abl screen: pop-free trunk + an additive training-only γ·log1p(luo) term, read as PD (γ=0) and as PDA, vs pop1.
  - Optional; P 0.01-0.02.
- **Per-user residual, user-relative attributes, F17 additive CTR**: closed or cut by measurement (§2.1).
  - If F17 is ever run: a CPU gate first (residual over w5_e10_base members >= +0.0005; today about −0.0002) and a d−8 gap build, not d−3.
- **Recency re-weighting RECW, weekday windows** (x3.8, x13.8): measured direction negative (rec33; newest regime −0.00034, z −2.0).
- **Numeric encodings** (x4.2, x5.10, x7.8, x11.9, x18.5): both screens show the production-fitting signature (val −, holdout +).
  - Diversity-family use only. Yeo-Johnson on the 9 skewed ctx_*/seq_cv_norm channels can ride along in any run that changes the preset.
- **MSM aux, contrastive SSL in FT, encoder-as-init, pre3** (x4.9, x5.1, x9.8, x5.7): the corpus-to-labelled ratio is 1:1 (PRETRAIN_PLAN §3.3); idle-lane only, P <= 0.02.
- **Others:** label smoothing in the cooldown, AutoDebias meta-weights, PFD with position, TENT, NCL, cross-feed negatives, ULTR teacher labels, DEPTHAUX, EXAMOFF on production rows, IPW/ULTR, item-balanced sampling, training-time session augmentation, hpool, hzmix, replacement losses, DOTHEAD, post-fusion FiLM (x11.0, x11.6, x11.2, x12.5, x12.7, x6.5/x12.2, x6.6, x3.0, x3.1, x6.9/x11.4/x11.5, x4.7, x4.10, x13.4, x14.0, x10.7-11, x3.4, x7.1).
  - Each is at P <= 0.02, with its mechanism argument and a ready spec in `checks.json`.
  - Run only to fill otherwise idle lanes, and only with the spec's own pre-declared rule.

---

## 3. NEW techniques (never run in any form)

**N1. GBDT leaf-index SVD as an NN input** (x1.1, x4.0, x18.4; trivago'19 L6 2nd: +0.003 MRR on both NN types)
- *Spec:*
  - Stage 0 is the CPU gate in §2.3.
  - Stage A (T01 form) puts the 2-fold cross-fit GBDT OOF score in as one xt column 'gbdt_oof' (raw + XT_REL z/rank, N(0, 0.1) train noise), with an is_exploration flag set to 1 at val/test.
  - Stage B (only if A passes) adds pred_leaf → randomized TruncatedSVD(32) on a 2M-row one-hot sample, projected chunk-wise as Σ_trees V[tree, leaf] (numba; ~13.5 GB fp16).
  - The GBDT is trained on a 10% user-hash slice that is excluded from NN training in both arms (in-fold leaves memorise labels).
- *Rule:* 2-epoch screen CONTINUE iff pooled val >= +0.0005 AND holdout >= 0 AND stale-val >= −1 SE AND no M-slice < −2 SE; then the E rule + diversity clause.
- *Cost:* 4-6 CPU-h + 1.6 GPU-h per screen + ~9 GPU-h pair.
- *P* 0.015. In trivago the GBM was the strongest single model; ours is 0.015-0.018 weaker than the NN, and a linear readout of the leaves (= the GBDT score) adds <= +0.00002 to G3.

**N2. Post-last-click negatives treated as unexamined** (x6.4; WSDM Cup'23 3rd)
- *Mechanism:* fact found this scan: 41% of production M>=50 negatives sit below the last positive.
  - In production feeds they are both deep in the production order and partly unexamined, which pushes the main CE toward the production order on top of K9.
  - In random-like feeds the effect is item-independent noise.
- *Spec:* `--tail_w 0.3` adds log(0.3) to the logits of negatives with cand_pos > the feed's last positive, in the main CE only (not K9, not eval), on train feeds only.
- *Rule:* a 10-epoch pair directly (offsets act through training; the TRUNC lesson), E rule.
- *Cost:* ~1 h + 9 GPU-h.
- *P* 0.02. The anti-production-echo class has hurt before: IPS proxy −0.0058, rho70 −0.0035, TRUNC −0.0005.

**N3. Event-field dropout in history tokens** (x8.1; Avito interest model, habr 974682)
- *Spec:* `--hfield_drop 0.1`, training only, at model.py:549. Independent Bernoulli masks on the et, es and cat terms, replaced by learned zero-init mask vectors; h_lin(he) is dropped at P/2 to h_miss (imitating the real 14% missing vectors). Flag-off bit-identity test.
- *Rule:* a 10-epoch pair directly (a regulariser builds over training; 2-epoch screens mis-ranked resdrop0 / LATE-DROPOUT), E rule + diversity clause.
- *Cost:* 1 h + 9 GPU-h.
- *P* 0.015 (blocks3 shows no overfitting to regularise).

**N4. Nonlinear token MLP instead of summed token components** (x1.3; Riiid 2nd/3rd)
- *Spec:* `--htok_mlp`: delta = MLP(concat(h_lin(he), et16, es16, cat16, pos16, miss)) with a zero-init last layer, added before h_ln. Linear concat equals our sum, so only the nonlinear form is new.
- *Rule:* 2-epoch SCALE-UP, then the E rule.
- *Cost:* 1 h + 1.6 GPU-h.
- *P* 0.01 (capacity screens 0/13).

**N5. Cross-feed prediction consensus** (x1.4; TalkingData 3rd groupby predictions, +0.0005 private)
- CPU, §2.3. *P* 0.01. Coverage is the likely kill.

**N6. Data-split (user-half) members** (x8.6, x14.10; SNA'19 1st, VK'24 5th 0.6732 → 0.6773)
- *Pre-check (free):* the scaling slope from B23/B24 (about +0.0037 val per data doubling) predicts −0.002..−0.004 val per half member, below the strength bar.
- *Spec:* read it for free on R1's half members if R1 goes ahead.
  - Adopt the axis only if the mean pairwise within-feed corr of the half pool is <= the full-data pool's − 0.01 AND G3 + halves >= G3 + full seeds on val with H >= −1 SE.
- *P* 0.015. Never as a standalone 9 GPU-h arm.

**N7. True time-backtest fold** (x13.2)
- Refuted by the DPA audit (T5: it cannot change any decision; M<50 exploration is unidentifiable in train).
- Not recommended.

---

## 4. Validation and ensembling practices worth adopting

### 4.1 New rules (zero or near-zero cost)

1. **Pre-declare @last as the primary read for every future WSD arm; [selected] is information** (x11.7).
   - Waves 5 and 6-8 used different primaries, and the two reads differ by up to 0.0007 val on the same runs.
   - @last is the oracle-free endpoint (lr reaches 0). DomainBed's lesson is that the selection protocol changes the method ranking.
2. **Matched-diversity control for every "base + arm" pool claim** (from the PDA skeptic's dose proxy).
   - z-mean(m02ref, hzcd) read +0.00033 val against m02ref alone.
   - Against m02ref + a matched non-hz second member (rdrop, cbs2, an m02 seed or cbs05) the same blend lost holdout −0.00038..−0.00083 (z −2.6..−5.1).
   - The apparent gain was the ensemble bonus. Any family-level claim must beat "base + one more seed of base".
3. **Price train-period statistics at the private gap before any GPU spend** (from the F17 skeptic).
   - Rebuild the column at about an 8-10 day gap (not val's 3-5).
   - Fit on the fresh version, apply to the stale one, and require the residual to survive.
   - The CTR's within-feed AUC falls from 0.5225 (1-day gap) to 0.501 (8 days).
4. **Report the item-regime mix of any in-period training or evaluation population**: the seen-later / unseen share.
   - EXPLO-STACK stage-2 rows: 12% unseen items vs 29% val and 39% test.
   - H: 62.6% of rows are items trained on at a later date.
   - These populations over-weight the warm regime where G3 already wins.
5. **One-batch multiplicity for CPU probe sweeps:** pre-declare the batch, apply Holm or a 2.5 SE bar, and print the s04 noise-column band next to every row (§2.3).
6. **Code hygiene:** `--loss bce|both` in nn_x/w5/train.py:462-467 ignore fw / loss_norm / denom. Either fix them to call `listwise_loss(..., fw, norm=a.loss_norm, denom=a.max_feeds)`, or forbid them in lanes.
7. **Test-info audit addendum:** record the f5 forward echo as an organiser-counter channel. It is leak-class (own-feed label through Avito's live counter), worth <= +0.0004 on val, with 4.5-5.1% of public/private rows in a < 6 h window. Banned under "no own-feed labels".

### 4.2 Already adopted (do not redo)

External practice that matches what we already do:
- time-ordered val;
- late-holdout gap column (Avito Demand'18 3rd);
- per-feature time-consistency (IEEE-CIS 1st);
- adversarial validation;
- member-equal mean (Caruana ICML'04: an unweighted average "cannot overfit the validation set");
- trust local validation over a small public board (Santander 2nd, TalkingData 2nd, Roelofs NeurIPS'19);
- hedging shift-dependent knobs at the midpoint (Home Credit'24 1st);
- winner's-curse shrinkage (Ladder, Thresholdout, Cawley & Talbot);
- seed bagging ("never submit a single NN", KKBox).

Nothing in this cluster beats the frozen selection protocol as amended on 16:10.

---

## 5. What the #1 team most plausibly does, given this catalogue

Leader 0.398256 vs G3 0.392820: +0.0054 public. The cross-team paired SE is 0.0012-0.0018, so this is real but of uncertain size. Base rates are from EXTERNAL_TOP1_1113 §2.3; the adjustments below are from this scan.

| rank | hypothesis | P | what this scan adds |
|---|---|---|---|
| 1 | **H2: a structurally different or better model, or several levers each below our screen MDE** | 0.50 (=) | The winners' recipes are all ADOPTED or CLOSED here: GBDT stacks on target statistics and CF scores (OTTO, H&M, Avito'24/'25, Kion), sequence transformers with listwise loss (Avito L3, VK'24), set attention, co-visitation, transductive popularity. No transferable external mechanism has a documented effect above about +0.001 on a comparable task. Our single-member MDE is 0.0029 and the near-copy pool MDE 0.0009, so 5-10 real +0.0005 levers could hide in our "closed at screen power" rows. That reading is unchanged. |
| 2 | **H7: an organiser- or Avito-affiliated reference with information we do not have** | 0.12 (+0.02) | Avito's production L3 is our architecture class (habr 1090584) but uses inputs that are N/A to us: **event age / timestamps** of history events (the interest model encodes age in seconds, habr 974682), real-time counters, seller ids, price. The N/A rows of §1 (F35, A14, T15) are exactly where a data-privileged entry gains. The f5 forward echo also shows that live counters echo labels. A single upload on a Thursday evening fits a reference check. |
| 3 | H8: an information channel our audits missed | 0.10 (−0.02) | This scan closed more channels: f5 echo <= +0.0006 even with val→test occurrences; earlier-impression negative feedback; history beyond 256; IDF; seller-slot batches (coverage 0.1%); blender order. The DPA audit found no residual structure in 92 inputs or 94 constructions. |
| 4 | H3: heavy val-label use under a protocol we did not try | 0.10 (=) | External val/train+val retrain gains (:D +0.0097-0.0123 AUC) are already captured by our FT. valcd (+0.00014) and full-val refit (−0.00025) are closed. No new protocol emerged from the scan. |
| 5 | H1 merged team / H5 LB probing / H4 external data / H6 format / residual | 0.05 / 0.05 / 0.03 / 0.01 / 0.04 | Nothing new. TalkingData 2nd and Santander 2nd confirm that LB-fitted weights lose on private. |

Bottom line: the catalogue gives no concrete "they probably did X and we did not" with P > 0.05. If H2 holds, the most plausible ingredients among those visible to us are:
- a production-feed debiasing that works where TRUNC did not;
- a stronger stage 2 trained on in-domain data (D2's question, R1);
- objective / architecture diversity at full strength (R3, R4).

Together, at our priors, that is about +0.0003 expected, not +0.005. Nothing here justifies chasing the public gap with model changes.

---

## 6. COVERAGE_MAP rows and registry one-liners

### 6.1 COVERAGE_MAP rows (new section "External scan 11-13")

| idea | status | best evidence | file |
|---|---|---|---|
| f5 forward echo (SNA'19 "next counter"): other-user later f5 minus f5 now | CLOSED 11-13 (leak-class) | train AUC 0.579 (< 6 h) vs backward 0.498; val 0.560; nested on G3 +0.00038 ± 0.00024 (z 1.6); test coverage < 6 h 4.5-5.1% | EXTERNAL_SCAN_1113 §1.2 F15; ext/russian/probe/f5_* |
| Avito blender: within-category production order as a cleaner K9 teacher | CLOSED 11-13 | same-category excess equal in random-order feeds (+0.0145 vs +0.0130 at d 4-7) | ext/russian/probe/blender_probe{,2}.py |
| Embedding-level negative feedback from the user's earlier impressions (DFN / XDM / skip-above) | CLOSED 11-13 | skipped items wAUC 0.58 (= interest); nested −0.0006..+0.0001 | ext/features/probe_negfb.log |
| Lifelong history from earlier snapshots (SIM/TWIN) | CLOSED 11-13 | +93 items median; nested −0.0002..0 | ext/features/probe_histext.log |
| IDF damping of history items (OTTO 3rd w_recsys) on the embedding-sim aggregation | CLOSED 11-13 | +0.00024 ± 0.00025 (z 0.96) | ext/features/probe_histidf.log |
| Typed / backward co-vis, t1 share, ref co-vis | CLOSED 11-13 | nested on G3 all <= +0.00024, mostly negative | ext/xcheck169/probe_typedcv.log |
| Category-sequence family (runs, momentum, decayed shares, head-run) | CLOSED 11-13 | nested per column <= +0.0001 | ext/xcheck169/probe_catC.log |
| Seller-slot batch co-membership | CLOSED 11-13 | coverage 0.1% of candidates | ext/xcheck193/s3.log |
| Per-user prior-error / preference from earlier labelled feeds | CLOSED 11-13 (nested G3, pre-declared) | −0.00005..−0.00027; split-half stability 0.03-0.18 | ext/skeptic-uresid/ |
| User-relative f4/f5 profile (user preferred age / engagement level) | CLOSED 11-13 (leak-free pre-test) | +0.00325 on f-only base → +0.00003 with popularity or pre2 | ext/skeptic_userrel_attr/ |
| Additive item CTR/COEC as a train-stage input (F17) | RE-OPEN (low), P <= 0.02 | pop2 trigger fired; residual over base members about −0.0002; AUC 0.5225 → 0.501 from 1- to 8-day gap | ext/skeptic_f17/ |
| D2 / EXPLO-STACK, stage 2 on OOF NN scores of 134K exploration feeds | RE-OPEN via zero-GPU E1 scale-curve gate, P 0.03 | E1 30K −0.00031 / −0.00042 (z −2.3) | §2.2 R1 |
| XOFF: examination offset on exploration rows only | RE-OPEN, P 0.02-0.03 | proxies only; Google KDD'23 random-logging PAL +6.3% | §2.2 R2 |
| Objective-diverse member family (CE + BCE term, `--bce_w`) | RE-OPEN, P 0.025 | MLP proxy + one AdamW point; `--loss both` code path drops fw/loss_norm | §2.2 R3 |
| Parallel DCN / GDCN branch on the transformer ranker | RE-OPEN, P 0.02-0.025 | only inside DIN (F11) | §2.2 R4 |
| Example-age / source-depth input | RE-OPEN (CPU gate first), P 0.02 | critique-only rejection | §2.2 R5 |
| KD / test pseudo-labels from G3 in the cooldown | RE-OPEN, P 0.02 each | AdamW single-teacher / FT-level only | §2.2 R6 |
| Checkpoint read @last as primary (protocol) + rdrop/sam fresh-root replication | RE-OPEN (rule now; replication P 0.02) | wave-7 @last rdrop +0.00057 (z 1.9) vs [selected] −0.00012 | §2.2 R7, §4.1 |
| ixh (item production-position prior) as NN input | RE-OPEN, P 0.02 | GBDT gate + 1 AdamW seed (+0.0010 / +0.0002) | §2.2 R8 |
| sim_bias 2; within-(feed, category) xrel_cat; non-additive position embedding with observation dropout | RE-OPEN (idle lanes), P 0.012-0.02 | §2.2 R9 | checks.json x5.4, x4.1, x9.10 |
| ADT truncated / reweighted CE | RE-OPEN (CPU zoo first), P 0.02 | ce_rw holdout −0.00119 (z −3.3) | §2.2 R10 |
| GBDT leaf-SVD / GBDT-OOF score as NN input | UNTESTED-EXTERNAL, P 0.015 | trivago L6 +0.003 MRR | §3 N1 |
| Post-last-click negatives down-weighted as unexamined | UNTESTED-EXTERNAL, P 0.02 | 41% of production M>=50 negatives below the last positive | §3 N2; ext/xcheck121/below_last_result.txt |
| Event-field dropout in history tokens | UNTESTED-EXTERNAL, P 0.015 | Avito interest model | §3 N3 |
| Nonlinear token MLP (concat vs sum) | UNTESTED-EXTERNAL, P 0.01 | Riiid 2nd/3rd | §3 N4 |
| Cross-feed prediction consensus | UNTESTED-EXTERNAL, P 0.01 | TalkingData 3rd | §3 N5 |
| User-half data-split members | UNTESTED-EXTERNAL, P 0.015 (only as an R1 by-product) | SNA'19 1st, VK'24 5th | §3 N6 |
| CPU closure batch (streak, latent geometry, co-vis variants, IDF/IUF/Swing counts, Bradley-Terry, trends, EMDE, RP3beta, user aggregates, typed rating, key-B post-processing, pending favourite, PLE pre-gate, REGIME4 step 0) | UNTESTED-EXTERNAL / RE-OPEN (one pre-declared batch, Holm) | each P 0.01-0.02 | §2.3 |
| PDA/TIDE re-judging of popularity-robust arms | CLOSED 11-13 (two skeptics) | hzcd half dose; circular stale judge; matched-control holdout −0.0004..−0.0008 | ext/skeptic-pda-tide/ |
| Gap-matched lens / true backtest fold | CLOSED 11-13 (superseded by the DPA audit) | late-holdout column adopted; T4/T5 refuted | DATA_PIPELINE_1113 |

### 6.2 Registry / CURRENT_STATUS one-liners

- `16:32 11-13 EXTERNAL SCAN DONE (a workflow run; reports/EXTERNAL_SCAN_1113.md; scratch ext/ -> nn_x/audit/ext_1113/)`: 211 rows / ~105 mechanisms from ~120 public sources; 99 closed or adopted, 82 weak, 23 N/A, 7 new; 8 top candidates x 2 skeptics: 0 survive both; no item above P 0.035; E[pool gain of the whole re-open list] ~ +0.0003.
- `CLOSED by fresh CPU probes`: f5 forward echo (leak-class, +0.00038 z 1.6); earlier-impression negative feedback; lifelong history; IDF damping; typed co-vis; category-sequence family; seller slots; within-category blender teacher; per-user prior error (−0.00005..−0.00027); user-relative f4/f5 profile (+0.00003 with popularity or pre2).
- `CODE`: `--loss bce|both` in nn_x/w5/train.py:462-467 ignore fw / loss_norm / denom; use a new `--bce_w` for any BCE-family arm.
- `RULES proposed`: @last primary for WSD arms; matched-diversity control for pool claims; gap-priced train-period statistics; one-batch Holm for CPU sweeps.
- `QUEUE (idle the GPUs)`:
  - R1 D2 zero-GPU gate (CPU 2 h);
  - R2 XOFF (CPU 30 min + 2.7 GPU-h);
  - R3 `--bce_w 5` 10-epoch pair (9 GPU-h);
  - R4 GDCN screen (1.6 GPU-h);
  - R5 example-age Stage 0 (CPU 1 h);
  - R6 KD / PL cooldown arms (2 x 2.7 GPU-h);
  - §2.3 CPU batch.

---

## 7. Sources (all public; see the cluster notes for page-level quotes)

Kaggle (write-up text via the HF mirror https://huggingface.co/datasets/Abirate/all_write_ups unless a URL is given):
- OTTO 2023 1st / 13th / 16th / 20th; 3rd: https://github.com/TheoViel/kaggle_otto_rs, https://github.com/bschifferer/Kaggle-Otto-Comp, https://github.com/cdeotte/Kaggle-OTTO-Comp
- H&M 2022 1st / 3rd / 10th
- Instacart 2017: 2nd https://github.com/KazukiOnodera/Instacart; 3rd https://github.com/sjvasquez/instacart-basket-prediction; 4th, 6th, 8th, 9th, 11th, 54th; J. Peeters blog, jacquespeeters.github.io, 2017-08-28
- Expedia ICDM 2013, 5th: https://arxiv.org/abs/1311.7679
- Expedia Hotel Recommendations 2016, 1st
- Outbrain 2017, 2nd / 3rd / 4th
- Booking WSDM WebTour 2021, 1st (NVIDIA): https://ceur-ws.org/Vol-2855/challenge_short_2.pdf and https://developer.nvidia.com/blog/how-to-build-a-winning-deep-learning-powered-recommender-system-part-3/
- Santander 2016: 1st; 2nd https://ttvand.github.io/Second-place-in-the-Santander-product-Recommendation-Kaggle-competition
- Elo 2019, 5th / 10th / 11th / 16th
- Riiid 2020: 1st https://arxiv.org/abs/2102.05038; 2nd, 3rd, 8th
- KDD Cup 2012: T2 1st https://jyunyu.csie.org/docs/awds/kddcup2012paper.pdf; T1 https://arxiv.org/abs/2105.00991
- Criteo 2014, 3 Idiots: https://github.com/zzbased/kaggle-2014-criteo
- Avazu 2015: https://techblog.youdao.com/wp-content/uploads/2015/03/Avazu-CTR-Prediction.pdf
- TalkingData 2018, 1st / 2nd / 3rd / 5th / 6th / 8th
- KKBox WSDM 2018, 1st
- Yandex PWS 2014: https://blog.dataiku.com/2014/01/14/winning-kaggle
- Recruit Coupon 2015, 2nd: https://cnblogs.com/yymn/p/4902547.html
- Avito Context Ad Clicks 2015: https://adamdrake.com/a-quick-overview-on-the-kaggle-competition-for-avito.html

Challenges:
- Layer 6 RecSys'17 https://www.cs.toronto.edu/~guangweiyu/pdfs/recsys2017.pdf; '18 https://www.cs.toronto.edu/~mvolkovs/recsys2018_challenge.pdf; '19 https://www.cs.toronto.edu/~mvolkovs/recsys2019_challenge.pdf; '22 https://www.cs.toronto.edu/~mvolkovs/Recsys_2022.pdf
- NVIDIA-Merlin/competitions (RecSys 2019-2022)
- Synerise RecSys'21: arXiv 2109.12985; EMDE arXiv 2006.01894
- Upstage RecSys'23: arXiv 2310.03778
- RecSys'24 :D: https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning and https://github.com/k-fujikawa/recsys-challenge-2024-1st-place; organisers arXiv 2409.20483
- RecSys'25: 4th arXiv 2508.06970; 2nd arXiv 2508.07748
- KDD Cup'20 1st: https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place
- KDD Cup'23: https://kddcup23.github.io; UEC arXiv 2310.07281
- WSDM Cup'22 XMRec: arXiv 2203.00897, 2204.12682
- WSDM Cup'23 ULTR: arXiv 2302.13756, 2304.12650, 2302.07530; Hager et al. SIGIR'24 arXiv 2404.02543
- CIKM Cup'16: arXiv 1708.04479
- CIKM AnalytiCup'19 QDU: github ChuanyuXue/CIKM-2019-AnalytiCup
- SIGIR eCom'21 NVIDIA: arXiv 2107.05124
- RecSys'15 1st: Romov & Sokolov slides (slideshare)
- RecSys'16 1st: doi 10.1145/2987538.2987543

Russian:
- Avito: habr.com/ru/companies/avito/articles/1090584 (2026), habr.com/ru/articles/974682 (2025), habr.com/ru/articles/1004694 (2026), habr.com/ru/articles/846832 (2024)
- SNA Hackathon 2019: habr.com/ru/post/447376 (1st), /448140 (organisers), /445348; habr.com/ru/companies/vk/articles/438392
- Rekko 2nd: habr.com/ru/companies/tbank/articles/454818
- MTS Kion: Doklady RAN 514(2), 2023
- Data Fusion: habr.com/ru/companies/ods/articles/670572 (DF'22), habr.com/ru/companies/T1Holding/articles/856228 (DF'24)
- VK: habr.com/ru/post/720956, habr.com/ru/companies/vk/articles/947456, /1088034, /1068048
- Lamoda: habr.com/ru/companies/lamoda/articles/943272
- Yandex Lavka: habr.com/ru/articles/1051044
- Wildberries: habr.com/ru/companies/rwb/articles/972082, /861466
- Positional bias how-to: habr.com/ru/articles/887426
- Klenitskiy & Vasilev: arXiv 2309.07602
- ARGUS: arXiv 2507.15994
- Avito ML Cup repos: <public repository, link removed> (2024 2nd), <public repository, link removed> (2025 1st), <public repository, link removed> (spring 2026, 30th)

Industry and literature:
- Losses: Bruch ICTIR'19 (research.google pubs); DASALC ICLR'21; LambdaLoss CIKM'18 (research.google/pubs/pub47258); stochastic treatment WSDM'20 (pub48689); xENDCG arXiv 1911.09798; NeuralNDCG arXiv 2102.07831; PiRank arXiv 2012.06731
- Crosses and debiasing: BARS arXiv 2009.05794; Google KDD'23 arXiv 2212.13937; AutoDebias arXiv 2105.04170
- Distillation: PFD arXiv 1907.05171 and 2209.08754; CLID arXiv 2312.08727; Ranking Distillation arXiv 1809.07428; RankDistil (pub50695); RankFlow github qinjr/RankFlow; LCRON arXiv 2503.09492
- Negatives and sequences: MNS (pub50257); gSASRec arXiv 2308.07192; ADT arXiv 2006.04153
- Popularity: PDA arXiv 2105.06067; TIDE arXiv 2109.07946
- Shift and ensembles: Wild-Time arXiv 2211.14238; Gontijo-Lopes arXiv 2110.12899; Fort et al. arXiv 1912.02757; Jeffares et al. arXiv 2301.11323; Allegro arXiv 2005.10084
- Other: T²ARec arXiv 2504.01489; Joachims SIGIR'05

Validation:
- Competition write-ups: IEEE-CIS 1st https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution; Avito Demand 2018 https://sijunhe.github.io/2018/09/04/kaggle-predict-demand-for-classified-ads.html; Jane Street 8th https://raw.githubusercontent.com/evgeniavolkova/kagglejanestreet/master/solution.md; H&M 1st https://speakerdeck.com/kfujikawa/h-and-m-personalized-fashion-recommendation; M5 3rd arXiv 2310.13029; Favorita 5th (forums.fast.ai); Home Credit 2024 1st (via research_1112/shift.md)
- Practice and theory: TIF arXiv 2311.16878; YouTube RecSys'16; NVIDIA Grandmasters Playbook 2025; Ladder arXiv 1502.04585; Thresholdout arXiv 1506.02629; Roelofs et al. NeurIPS'19; Caruana ICML'04 https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf; TabReD arXiv 2406.19380; Cawley & Talbot JMLR'10; purged CV (López de Prado 2018); Bojer & Meldgaard arXiv 2009.07701

Features:
- WeChat DFN IJCAI'20: https://ruobingxie.github.io/publications/IJCAI-20_feedback.pdf
- Alibaba SIM: arXiv 2006.05639
- XDM: arXiv 2010.12837
- TransAct: arXiv 2306.00248
- Facebook ADKDD'14, "Practical Lessons from Predicting Clicks on Ads at Facebook"
- Dacrema et al. RecSys'19
- Hotel distance-matrix completion: arXiv 1908.07498
