# Solvers, ranking losses, training paradigms and overall solution design (sol_solvers)

Author: solvers research agent (workflow "solutions review"), 2026-11-11, ~17:50-19:45 UTC. CPU only (`CUDA_VISIBLE_DEVICES=""`, <= 16 threads in total).
Code: `<repo>/solvers/solvers/` (`loss_zoo.py` + `zoo_report.py` loss zoo; `cd_pool.py` + `pool_report.py` data/cooldown-pool proxies; `run_B.sh` ... `run_G.sh` batch drivers; `nn_lossdiv.py`, `gbdt_member.py`; `proxy_lib.py` copied from the ideas review).
Reports of the analyses: `solvers/out/solvers/zoo_report_A.txt`, `zoo_report_A6.txt`, `pool_report_1.txt`, `pool_report_F.txt`, `pool_report_CD.txt`, `pool_report_CDlo.txt`, `zoo_MEMW_s*.log`.
Outputs: `<repo>/solvers/out/solvers/` (`zoo_<tag>_<loss>_s<seed>.npz` hold per-feed val/holdout NDCG and the raw val/holdout scores; `zoo_*.log`).
Nothing under nn/, src/, pretrain/, data/feats/, preds/ or subs/ was touched. No GPU was used.

This report extends `nnx_research_*.md`, `research_sota_ltr.md`, `deep_neural_rankers.md` §8, `deep_gbdt_ranking.md`,
`deep_position_bias_random.md` and `review_ideas.md`. It does not repeat their literature summaries; it adds (a) the first
side-by-side measurement of 16 ranking losses / paradigms on our data, (b) loss-diversity ensemble tests, (c) a
GBDT-member re-test on the current best v11, and (d) a mapping of the remaining literature onto our pipeline with exact
experiment specs. Tags: **[E]** own experiment (numbers reproducible from the files above), **[V]** source read this
session, **[R]** earlier project report (file cited), **[K]** background knowledge.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

**Bottom line: the solver choices of the current pipeline are right. Softmax CE beats or ties all 14 alternative
surrogate losses we measured. Loss diversity buys no ensemble gain. GBDT rankers and the sequence encoder hurt v11 at every
pre-declared weight. ULTR corrections are closed. What is left are small, cheap refinements of components that already
showed large effects.**

Key findings (all CPU, paired per feed on the 14,423 val feeds, 3-6 seeds; details in §2):

1. **Loss zoo (§2.1).** 16 losses/paradigms on one proxy ranker (MLP on 48 features, 300K train feeds, 3 seeds; the
   harness reproduces the ideas-review CE runs bit-exactly). Nothing beats softmax CE (`nn/model.py:513-534`). Ties
   (|Δ| < 1 SE): label smoothing +0.00007 +- 0.00012 (6 seeds), focal +0.00011, LambdaLoss NDCG-Loss2++ +0.00006,
   CE+LambdaLoss +0.00015, PL over positives -0.00006. Worse: RankNet -0.00033, calibrated anchor -0.00045,
   LambdaRank@max -0.00057, PAL position tower -0.00061 (z -2.1; holdout +0.00016, the confounding signature),
   CE+BCE -0.00128, NeuralNDCG -0.00165 (z -2.5), ApproxNDCG -0.00295, BCE -0.00396, **full ListMLE (Avito's
   production loss) -0.00516 (z -6.5)**. The losers lose mostly on M<50 feeds (41% of val, holdout-blind).
   Calibration against the real NN: same sign for BCE (proxy exaggerates ~4x) and the position tower; the proxy cannot
   see aux-task effects (K9: NN +0.0024, proxy +0.00003) but does see data-mixture effects (cooldown mix: NN +0.0018,
   proxy +0.0010).
2. **Loss diversity (§2.2).** No loss paired with CE beats a CE seed pair, except a hint for an NDCG-sensitivity-weighted
   CE (`ce_rw`, +0.00056 +- 0.00042, z 1.3) whose single-model holdout is worse (-0.00119 +- 0.00036). Real-NN check
   (round-1 models): base+bce -0.00033 +- 0.00059 vs base+nopos.
3. **Data paradigm (§2.3).** Per feed, a random-order (exploration) train feed is worth +0.0031 val over a production
   feed. Adding 300K production feeds to 300K exploration feeds adds nothing in the proxy (the real NN still needs
   them for representation and K9; up-weighting exploration in the main run cost the NN -0.001).
4. **Memorisation (§2.4).** A wide-MLP proxy reproduces the cooldown-style collapse (val 0.370 → 0.354). Label
   smoothing (ε 0.1/0.2) and the calibrated anchor do not prevent it (ε 0.2: +0.0007 at the end, 1 seed).
5. **Cooldown pool (§2.6).** Adding the short (M<50) "random-like" feeds to the pool hurts (-0.00053 +- 0.00016, also
   on M<50); a looser threshold p_random > 0.2 on M>=50 feeds costs nothing (+0.00000 +- 0.00014) and gives +47% distinct
   pool feeds (~197,500 vs 134,118 after the holdout).
6. **Blend members on v11 (§2.5).** CatBoost YetiRank / YetiRankPairwise and XGBoost (LUO) rankers and pre_final all
   hurt v11 = 0.39793 at pre-declared w 0.05/0.1/0.2 (e.g. cb_tr300k_luo -0.00042 / -0.00062 / -0.00124). The
   winners' multi-family stacks need members within ~1% of each other; ours are 0.03 NDCG apart.

Ranked proposals (§4; exact specs there):

| rank | proposal | expected gain | cost | risk |
|---|---|---|---|---|
| P1 | Tune the K9 production-order distillation (τ 30; Plackett-Luce top-20 target): never tuned (τ 10, w 0.5 in every logged run), largest loss-side win so far (+0.0024) | +0.0003 to +0.0012 / member | ~3 GPU-h (2-epoch pairs) + 1 confirmation pair | low |
| P2 | SAM (ρ 0.05) in the fta fine-tune | 0 to +0.0008 on the FT half | ~4 GPU-h | low |
| P3 | Cooldown pool threshold p_random > 0.2 (M>=50): +47% distinct feeds against memorisation | 0 to +0.0008 on cooled members | ~1.5 GPU-h | low |
| P4 | MC-dropout test-time averaging | -0.0005 to +0.0005 | GPU minutes | none |
| P5 | ce_rw member pair, judged only at ensemble level | 0 to +0.0005 | ~8 GPU-h | medium |
| P6 | Anneal-mix (val fold feeds inside a cooldown continuation) | 0 to +0.001, uncertain | ~3 GPU-h / member | medium |

Do not spend GPU on: surrogate-loss swaps, loss-diverse members, label smoothing as a memorisation guard, short feeds
in the cooldown pool, GBDT ranker members, ULTR corrections (IPS/DLA/DR/PAL), generative/sequence-encoder members,
test pseudo-labels, mixup (§4 do-not list).


---------------------------------------------------------------------------------------------------------------------

## 1. What is already known (inputs to this report; not re-proposed)

| question | what the project already measured | where |
|---|---|---|
| listwise vs pointwise in the NN | softmax CE 0.37361 vs BCE 0.37272 (round 1, same seed) | [R] nn_report.md:10, round-1 table |
| per-feed loss normalisation | equal per-feed weight +0.0020 val in the train stage; K/IDCG weighting neutral; neutral in fine-tune | [R] nn_report.md:134-150, err_SYNTHESIS.md:50 |
| multi-positive target | per-positive softmax vs ListNet y/K: -0.00002 +- 0.00007 (MLP proxy, 5 seeds) | [R] review_ideas.md §1.5 |
| production-order use | MSE aux +0.002, listwise K9 aux a further +0.0024; position tower neutral in the NN | [R] nn_report.md:10, 141-142 |
| exploration feeds | stage 2 +0.001; up-weighting them inside the main run -0.001; WSD cooldown mix 0.1 works, 0.3 collapses; pool memorised even at 0.1 | [R] nn_report.md:10, 500; REVIEW_SYNTHESIS P1 |
| debiasing in GBDT | LightGBM GAM `position=` +0.0022 (1 seed); XGBoost IPW -0.0022 (full) / -0.0002 (sqrt) | [R] deep_gbdt_ranking.md §2.3 |
| GBDT objectives | binary ≈ rank:ndcg (topk 300) ≈ YetiRank ≈ YetiRankPairwise ≈ QuerySoftMax, all within +-0.001 | [R] deep_gbdt_ranking.md §2.1 |
| KD / label smoothing / mixup / SAM / TTA | born-again KD n.s.; LS and mixup "do not use" (argument only, never run); SAM and R-Drop in the fine-tune "medium" (never run); history-length TTA worse; MC-dropout "low" (never run) | [R] nnx_VERDICT §5; nnx_research_training_gen.md §4.3-4.6, §9.6, §10 |
| test pseudo-labels | fine-tune with 12K soft-labelled test feeds per epoch: -0.0004 | [R] nn_report.md round 2 (N6) |
| stackers / GBDTs in the blend | residual stacker on nn_final3 -0.0001 +- 0.0006; train-fit GBDTs add nothing (v5 era) | [R] err_SYNTHESIS.md TL;DR 1; CURRENT_STATUS 3 |
| pretrained sequence encoder | pre_final zero-shot 0.359; +0.1 z(pre_final) hurt v11 (-0.00078 +- 0.00042) | [R] pretrain_report.md; CURRENT_STATUS 3j |

The literature on loss choice was summarised in `research_sota_ltr.md` §4 and `deep_neural_rankers.md` §8 (softmax CE
bounds NDCG for binary relevance; DASALC App. B.2: softmax beats LambdaRank in neural nets). What was missing is a
measurement on our data. Section 2 provides it.

---------------------------------------------------------------------------------------------------------------------

## 2. New experiments (CPU)

### 2.1 Loss-zoo proxy: 16 losses / paradigms on the same model, data and seeds [E]

**Setup** (`loss_zoo.py`, a superset of the ideas-review harness, which it reproduces bit-exactly: CE val 0.37274 /
0.37231 / 0.37200 for seeds 1-3, identical to `review/ideas/exp_loss_300k.log`):
- 300K train feeds drawn from Sep 9-19 (production + random-like, as in the NN train stage); 20K random-like M>=50
  feeds held out (proxy holdout); evaluation on all 14,423 val feeds (no val label is used for anything but scoring).
- 48 precomputed per-candidate features (content, within-feed z/rank, history similarity, LUO popularity,
  co-visitation) + NaN flags = 68 inputs. MLP 68-128-64-2 (second output = aux head, used only by `k9`), AdamW 2e-3
  cosine, batches of 256 feeds bucketed by M, 3 epochs, 3 seeds (same init and batch order across losses for a seed),
  every feed weighted equally (as `--loss_norm feed`).
- Paired per-feed deltas vs CE, averaged over the 3 seeds, SE = sd/sqrt(14,423). Slices: M<50 / M>=50, Tue / Thu,
  K=1 / K>=2 positives.

**Loss definitions** (s = scores of one feed, y binary, K = #positives, r = current rank, D(r) = log2(1+r)):

| kind | loss per feed | source |
|---|---|---|
| ce | -Σ_i (y_i/K) log softmax(s)_i (current `nn/model.py:513-534`) | ListNet / Bruch 2019 |
| ce_ls | target (1-ε) y/K + ε/M, ε = 0.1 (ce_ls2: ε = 0.2) | label smoothing |
| focal | -(1/K) Σ_pos (1-p_i)^2 log p_i | focal loss on the listwise softmax |
| anchor | softmax over [s, 0]: a virtual candidate with fixed logit 0 in every denominator | calibrated listwise loss, Yan et al. KDD'22 |
| plk | Plackett-Luce over the positives only (random order), negatives never ranked; = ce when K = 1 | PL top-K |
| listmle | full ListMLE: PL likelihood of one label-consistent permutation (positives first, random within ties) | Xia et al. 2008; Avito production loss [V habr 1090584] |
| bce | per-feed mean BCE | pointwise |
| ce_bce | ce + 5 · per-feed mean BCE (BCE is ~0.1 per feed, so this puts both on a similar gradient scale) | "Sigmoid+Softmax" (arXiv 2211.01494 via deep_neural_rankers §8) |
| ranknet | mean over (pos, neg) pairs of log(1+exp(-(s_p - s_n))) | RankNet |
| lambdarank | Σ_pairs \|ΔNDCG_pn\| log2(1+exp(-(s_p - s_n))), full list, /IDCG | LambdaRank@max |
| l2pp | Σ_pairs (ρ_pn + 5 δ_pn) log2(1+exp(-(s_p - s_n))), ρ = \|1/D(r_p) - 1/D(r_n)\|, δ = \|1/D(\|r_p - r_n\|) - 1/D(\|r_p - r_n\|+1)\| | LambdaLoss NDCG-Loss2++ (Wang et al. CIKM'18) |
| ce_l2pp | ce + l2pp | mixture |
| ce_rw | ce weighted per feed by dNDCG/dlog r at the positives' current mean rank (detached, batch mean 1) | NDCG-sensitivity-weighted softmax (new) |
| approx | 1 - Σ_pos 1/D(r̂_p) / IDCG, r̂_p = 1 + Σ_j σ(s_j - s_p) | ApproxNDCG (Qin 2010; Bruch SIGIR'19), α = 1 |
| neuralndcg | 1 - NDCG with NeuralSort P̂ (τ = 1) + 10 Sinkhorn iterations, full list | NeuralNDCG (Pobrotyn & Bialobrzeski 2021) |
| k9 | ce + 0.5 (1-p_random) · CE(aux, softmax(-pos/10)) | our K9 aux |
| pos | ce with an additive train-only position logit g(pos) + g'(M-1-pos) (×(1-p_random)), dropped at inference | PAL / shallow tower |

**Results** (val = mean of 3 seeds; Δ vs ce; SE of the seed-averaged paired per-feed delta):

| loss | val | Δ val (SE) | z | Δ holdout (SE) | M<50 | M>=50 | Tue | Thu | K=1 | K>=2 |
|---|---|---|---|---|---|---|---|---|---|---|
| **ce (baseline = current NN loss)** | **0.37235** | 0 | | 0 | | | | | | |
| ce_rw (NDCG-sensitivity-weighted CE; 6 seeds: +0.00016 (0.00052), holdout -0.00119 (0.00036)) | 0.37264 | +0.00028 (0.00056) | +0.5 | **-0.00111 (0.00038)** | +0.0000 | +0.0005 | +0.0001 | +0.0004 | +0.0004 | +0.0000 |
| ce_ls (label smoothing 0.1; 6 seeds: +0.00007 (0.00012)) | 0.37250 | +0.00015 (0.00016) | +0.9 | +0.00018 (0.00011) | +0.0003 | +0.0000 | +0.0003 | +0.0000 | +0.0001 | +0.0002 |
| ce_l2pp (CE + LambdaLoss) | 0.37250 | +0.00015 (0.00026) | +0.6 | -0.00019 (0.00018) | -0.0000 | +0.0002 | -0.0001 | +0.0004 | +0.0002 | -0.0001 |
| focal (γ 2) | 0.37246 | +0.00011 (0.00016) | +0.7 | +0.00001 (0.00009) | +0.0001 | +0.0001 | -0.0002 | +0.0004 | +0.0002 | -0.0001 |
| l2pp (LambdaLoss NDCG-Loss2++) | 0.37241 | +0.00006 (0.00028) | +0.2 | -0.00023 (0.00019) | -0.0001 | +0.0002 | -0.0001 | +0.0002 | +0.0002 | -0.0003 |
| k9 (CE + production-order aux) | 0.37239 | +0.00003 (0.00035) | +0.1 | -0.00036 (0.00025) | -0.0003 | +0.0003 | -0.0002 | +0.0003 | -0.0001 | +0.0004 |
| plk (PL over positives) | 0.37230 | -0.00006 (0.00006) | -0.9 | +0.00003 (0.00004) | -0.0001 | +0.0000 | -0.0000 | -0.0001 | -0.0001 | -0.0000 |
| ranknet | 0.37202 | -0.00033 (0.00027) | -1.2 | -0.00055 (0.00018) | -0.0008 | -0.0000 | -0.0003 | -0.0004 | -0.0003 | -0.0005 |
| anchor (calibrated softmax) | 0.37190 | -0.00045 (0.00037) | -1.2 | -0.00001 (0.00025) | -0.0015 | +0.0003 | -0.0005 | -0.0004 | -0.0005 | -0.0004 |
| lambdarank (@max) | 0.37179 | -0.00057 (0.00040) | -1.4 | -0.00076 (0.00026) | -0.0014 | -0.0000 | -0.0008 | -0.0003 | -0.0006 | -0.0005 |
| pos (additive PAL position tower) | 0.37175 | -0.00061 (0.00028) | -2.1 | +0.00016 (0.00024) | -0.0006 | -0.0006 | -0.0008 | -0.0004 | -0.0007 | -0.0003 |
| ce_bce (CE + BCE) | 0.37108 | -0.00128 (0.00052) | -2.5 | -0.00052 (0.00034) | **-0.0034** | +0.0002 | -0.0014 | -0.0011 | -0.0014 | -0.0010 |
| neuralndcg (τ 1, Sinkhorn) | 0.37071 | -0.00165 (0.00066) | -2.5 | -0.00236 (0.00044) | -0.0029 | -0.0008 | -0.0021 | -0.0012 | -0.0019 | -0.0010 |
| approx (ApproxNDCG) | 0.36940 | -0.00295 (0.00088) | -3.3 | -0.00350 (0.00062) | -0.0052 | -0.0014 | -0.0040 | -0.0017 | -0.0038 | -0.0005 |
| bce | 0.36839 | -0.00396 (0.00076) | -5.2 | -0.00332 (0.00055) | -0.0052 | -0.0031 | -0.0050 | -0.0029 | -0.0049 | -0.0013 |
| listmle (full ListMLE) | 0.36720 | -0.00516 (0.00079) | -6.5 | -0.00647 (0.00056) | -0.0054 | -0.0050 | -0.0063 | -0.0039 | -0.0054 | -0.0045 |

Full output: `solvers/out/solvers/zoo_report_A.txt`. Run time per model ~1 min (NeuralNDCG ~40 min: the n×n Sinkhorn on CPU).

Per-seed val deltas (seed 1 / 2 / 3): ce_rw +0.00001 / +0.00024 / +0.00060; ce_ls +0.00009 / -0.00015 / +0.00051;
ce_l2pp +0.00002 / +0.00036 / +0.00006; focal +0.00021 / +0.00009 / +0.00002; l2pp +0.00012 / +0.00007 / -0.00001;
lambdarank -0.00017 / -0.00084 / -0.00069; neuralndcg -0.00201 / -0.00162 / -0.00131; listmle -0.00571 / -0.00425 / -0.00550.

**Calibration of the proxy against the real NN** (same comparisons, real NN, round 1/2):

| change | real NN (val) | proxy (val) | reading |
|---|---|---|---|
| BCE instead of softmax CE | -0.0009 (0.37272 vs 0.37361, nn_report round 1) | -0.0040 +- 0.0008 | same sign, the proxy exaggerates loss-family effects ~4x |
| position tower | -0.0002 (base 0.37361 vs nopos 0.37385) | -0.0006 +- 0.0003 | same sign |
| K9 production-order aux | +0.0024 (v2_k9 vs v2_batch) | +0.00003 +- 0.00035 | the proxy cannot see representation-shaping aux tasks |

So the proxy resolves *loss-family* differences well (if anything it inflates them), but it is blind to auxiliary
tasks whose benefit comes through a shared deep representation. I therefore use it to rank surrogate losses, not aux
heads.

**Reading.**
1. **Nothing beats softmax CE.** The best alternatives are statistically indistinguishable from it (label smoothing,
   focal, LambdaLoss NDCG-Loss2++, PL over positives). Every pairwise loss (RankNet, LambdaRank) and every
   differentiable-sort / metric-approximation loss (ApproxNDCG, NeuralNDCG) is worse, and pointwise losses (BCE, CE+BCE)
   are clearly worse. This is what Bruch et al. (ICTIR'19) predict for binary relevance: softmax CE is a bound on
   -log NDCG and on MRR, and with 1-2 positives per list the metric-specific weighting of LambdaLoss/NeuralNDCG has
   little to add. On graded benchmarks the gaps are also small (allRank table [V]: Web30K NDCG@10 ListNet 52.80 vs
   NeuralNDCG@max 53.46 vs LambdaRank@max 53.47; Istella ListNet 70.70 vs 70.55 / 71.09; their τ was "tuned for optimal
   performance on the test set").
2. **The losers lose mostly on short feeds (M<50).** CE+BCE -0.0034 on M<50 vs +0.0002 on M>=50; anchor -0.0015 vs
   +0.0003; LambdaRank -0.0014 vs 0.0000; ApproxNDCG -0.0052 vs -0.0014. Short feeds are 41% of val and 39% of private,
   and invisible to the holdout (REVIEW_SYNTHESIS P3). Any loss change must be judged on the M<50 slice.
3. **Multi-positive variants are a dead end.** PL-top (positives do not compete with earlier positives) = CE
   (-0.00006 +- 0.00006, K>=2 slice -0.0000); per-positive softmax was already null. Full ListMLE (the Avito production loss) is the worst loss tested (-0.00516 +- 0.00079): with binary labels it spends most of its likelihood on the random order *among negatives*, which is pure noise (allRank reports the same ranking: ListMLE 51.82 vs ListNet 52.80 NDCG@10 on Web30K).
4. **The additive position tower (PAL) hurts val (-0.0006 +- 0.0003) while helping the train-distribution holdout
   (+0.0002)**: the confounding pattern of Google KDD'23 (`deep_position_bias_random.md` TL;DR 5). It confirms the
   NN-side choice of no tower and K9 distillation instead.
5. **Re-weighting feeds by the metric's sensitivity** (`ce_rw`, a listwise analogue of LambdaRank weights) leaves
val unchanged (+0.00016 +- 0.00052, 6 seeds) and hurts the holdout (-0.00119 +- 0.00036). Equal per-feed weight (as
now) stays the right weighting; this matches the neutral K/IDCG weighting already measured in the NN.

### 2.2 Loss diversity does not buy ensemble gain [E]

If an alternative loss gave equal single-model quality but lower correlation, it would be a cheap diversity source.
Test: z-mean(CE seed i, X seed j) against z-mean(CE seed i, CE seed j), i != j cyclic (3 pairs), paired per feed.

| X (paired with CE of another seed) | ensemble val | Δ vs CE seed pair (SE) | within-feed corr(CE, X), same seed |
|---|---|---|---|
| CE (seed pair, reference) | 0.37266 (single 0.37235, ensembling gain +0.00030) | 0 | (seed vs seed) |
| ce_rw (6 seeds) | 0.37324 | **+0.00056 (0.00042)**, z 1.3 | 0.957 |
| focal | 0.37272 | +0.00006 (0.00011) | 1.000 |
| plk | 0.37271 | +0.00005 (0.00004) | 1.000 |
| l2pp | 0.37270 | +0.00004 (0.00019) | 0.998 |
| k9 | 0.37268 | +0.00002 (0.00023) | 0.996 |
| ce_ls (6 seeds) | 0.37268 | -0.00000 (0.00009) | 0.994 |
| ce_l2pp | 0.37265 | -0.00001 (0.00019) | 0.998 |
| neuralndcg | 0.37259 | -0.00007 (0.00053) | 0.881 |
| pos | 0.37257 | -0.00008 (0.00018) | 0.999 |
| ranknet | 0.37249 | -0.00017 (0.00018) | 0.999 |
| ce_bce | 0.37245 | -0.00020 (0.00035) | 0.982 |
| anchor | 0.37243 | -0.00023 (0.00025) | 0.994 |
| lambdarank | 0.37236 | -0.00029 (0.00028) | 0.992 |
| approx | 0.37216 | -0.00050 (0.00072) | 0.827 |
| bce | 0.37172 | -0.00093 (0.00049) | 0.964 |
| listmle | 0.37063 | -0.00203 (0.00065) | 0.846 |

(6-seed numbers from `solvers/out/solvers/zoo_report_A6.txt`: seeds 4-6 of ce / ce_rw / ce_ls added in batch E.)

Real-NN data point (`nn_lossdiv.py`, round-1 train-only models, all seed 0, 2 epochs): base (CE) 0.37361, bce 0.37272.
z-mean(base, bce) 0.37389 vs z-mean(base, nopos) 0.37422: -0.00033 +- 0.00059 (within-feed correlation 0.974 vs
0.994). The BCE member is less correlated, but it is weaker by more than the diversity it adds.

**Reading:** most surrogate losses produce rankings that are 0.98-1.00 correlated with CE within feed and add nothing
over a second CE seed. The low-correlation ones (ApproxNDCG 0.83, ListMLE 0.85, NeuralNDCG 0.88) are too weak to help.
The one exception is the NDCG-sensitivity-weighted CE (`ce_rw`): equal single-model val (+0.00016 +- 0.00052 over 6
seeds, positive in 4 of 6), but **consistently worse on the holdout (-0.00119 +- 0.00036, z -3.3)**, and its pairing
with CE gains +0.00056 +- 0.00042 over a CE seed pair (z 1.3, corr 0.957). This is a hint of useful diversity, not a
result; it fails the single-model holdout criterion, so it can only ever be an ensemble member (proposal P5, low).
Overall, loss diversity is not a diversity source worth main GPU time; the ensemble's diversity has to keep coming
from seeds / recipes / inputs.

### 2.3 Data paradigm: exploration-only vs production feeds (proxy) [E]

Same CE proxy, same seeds, only the training-feed pool changes (`pool_report.py`; `pool_report_1.txt`,
`pool_report_F.txt`). "random-like" = `p_random > 0.5` from `eda/labels_position/out/train_prandom.npz`; the NN zeroes
it for M<50 because the detector is not identifiable there (`nn/train.py:160`, eda_labels_position.md:209-211).

| pool (Sep 9-19, proxy holdout excluded) | feeds / rows | val | Δ val vs production-only (SE) | Δ holdout | M<50 | M>=50 |
|---|---|---|---|---|---|---|
| production only (p <= 0.1 or unknown) | 300K / 25.2M | 0.37079 | 0 | 0 | | |
| mixed, as the NN samples (16% random-like) = tag A | 300K / 23.2M | 0.37235 | +0.00157 (0.00048) | +0.00147 | +0.0028 | +0.0007 |
| random-like only (all M) | 300K / 16.2M | 0.37387 | **+0.00308 (0.00077)** | +0.00371 | +0.0034 | +0.0029 |
| random-like 300K + production 300K | 600K / 41.4M | 0.37373 | +0.00294 (0.00061) | +0.00267 | +0.0031 | +0.0029 |
| same + K9 aux | 600K | 0.37418 | +0.00340 (0.00063) | +0.00231 | +0.0032 | +0.0035 |
| production only | 110K / 9.2M | 0.36968 | 0 (reference for the next two rows) | 0 | | |
| random-like, M>=50 (= the NN cooldown pool definition) | 108K / 16.0M | 0.37287 | **+0.00318 (0.00084)** | +0.00431 | +0.0023 | +0.0038 |
| "random-like", M<50 only (detector unreliable) | 110K / 1.7M | 0.37077 | +0.00109 (0.00090) | +0.00225 | **+0.0038** | -0.0008 |

**Reading.**
1. Per feed, an exploration (random-order) feed is worth far more than a production feed for ranking val's random-order
   feeds (+0.0031 at equal feed count, with fewer candidate rows). Adding 300K production feeds to 300K random-like feeds
   adds nothing in this proxy (-0.00014 val, -0.0010 holdout). The real NN disagrees in one respect: up-weighting
   exploration feeds inside the main run cost -0.001 (pw_aux_sim, nn_report round 1), because production feeds drive
   representation learning and the K9 distillation that this fixed-feature proxy cannot express. The consistent part of
   both is what the project already does: production feeds for the bulk of training, exploration feeds at the end
   (stage 2 +0.001; cooldown continuation +0.0018 per member).
2. **The short "random-like" feeds that the NN discards carry signal for exactly the short-feed population**: 110K of
   them (only 1.7M rows) beat 110K production feeds by +0.0038 on val M<50, while being useless for M>=50 (-0.0008).
   The NN's cooldown pool is M>=50 only (134,118 feeds, `nn/logs/v15_cd_all_e10.log`), the holdout is M>=50 only, and
   M<50 is 41% of val / 39% of private (REVIEW_SYNTHESIS P3). Train holds ~351K short feeds with p_random > 0.5
   (257,553 with M<20, 93,352 with M 20-49; counted from `train_prandom.npz`). Caveat: for short feeds the detector's
   evidence is the click position itself, so this subset is partly label-selected (feeds whose click is at a
   position atypical for production order). That does not leak val labels, but it means "random-like" is a misnomer
   there. Whether they help as a *cooldown* pool is tested in §2.6: they do not.

### 2.4 Memorisation regime: does label smoothing protect a small, repeatedly seen pool? [E]

**Why:** the WSD cooldown memorises the exploration pool (mix 0.3 collapses val 0.391 → 0.364; at mix 0.1 the pool
loss still falls far below the regular loss, nn_report:500). With softmax CE the optimum for a memorised feed is an
unbounded logit gap; label smoothing ε bounds it (K = 1: log((1-ε+ε/M)/(ε/M)) ≈ 6.8 nats at ε = 0.1, M = 96), so it is
the textbook guard. Two proxies:

(a) Small MLP (68-128-64), 30K random-like feeds, 15 epochs (tag MEM, 3 seeds): the small MLP barely memorises
(val 0.3708), and LS / focal / anchor change nothing (ce_ls -0.00004 +- 0.00017, ce_ls2 -0.00009 +- 0.00022, anchor
+0.00026 +- 0.00045, focal -0.00016 +- 0.00018).

(b) Wide MLP (68-512-512-2, ~300K parameters), 30K random-like feeds, 24 epochs, val/holdout at 6 points (tag MEMW).
This one reproduces a collapse of the same shape as the NN cooldown:

| step (of 2832) | ce s1 | ce_ls 0.1 s1 | ce_ls 0.2 s1 | anchor s1 | ce s2 | ce_ls 0.1 s2 |
|---|---|---|---|---|---|---|
| 472 | 0.37017 | 0.37008 | 0.37007 | 0.36950 | 0.36920 | 0.36940 |
| 944 | 0.36903 | 0.36844 | 0.36872 | 0.36878 | 0.36572 | 0.36705 |
| 1416 | 0.36520 | 0.36504 | 0.36657 | 0.36687 | 0.36421 | 0.36524 |
| 1888 | 0.35762 | 0.35761 | 0.35989 | 0.35871 | 0.35881 | 0.35973 |
| 2360 | 0.35580 | 0.35520 | 0.35604 | 0.35549 | 0.35681 | 0.35757 |
| end | 0.35374 | 0.35321 | 0.35446 | 0.35376 | 0.35506 | 0.35555 |

**Reading:** label smoothing does **not** stop memorisation. ε = 0.1 is within +-0.001 of CE at every point; ε = 0.2
delays the collapse a little (+0.0014 / +0.0023 at steps 1416 / 1888, 1 seed) and ends +0.0007. The calibrated anchor
does nothing. The lever against pool memorisation is *more distinct feeds per pass* (§2.6), not a
loss-side regulariser. Label smoothing on pool batches is therefore dropped (do-not list).

### 2.5 GBDT rankers and the sequence encoder as members of the current best (v11) [E]

`gbdt_member.py`: v11 = 0.5 z(nn_final5_val_oof) + 0.5 z(nn_final5_trainonly_val) = 0.39793. Add w · z(member) with
w pre-declared in {0.05, 0.1, 0.2} (no fitting); paired per-feed delta (SE):

| member | single val | within-feed corr with v11 | w 0.05 | w 0.10 | w 0.20 |
|---|---|---|---|---|---|
| cb_tr300k_luo (CatBoost YetiRank, 300K train feeds, LUO) | 0.36577 | 0.767 | -0.00042 (0.00027) | -0.00062 (0.00038) | -0.00124 (0.00053) |
| xgb_x300_luo (XGBoost rank:ndcg, topk 300 pairs, 300K train feeds, LUO) | 0.36723 | 0.811 | -0.00032 (0.00024) | -0.00050 (0.00033) | -0.00096 (0.00044) |
| cb_tr500k_pw (CatBoost YetiRankPairwise, 500K) | 0.36855 | 0.803 | -0.00050 (0.00027) | -0.00065 (0.00037) | -0.00133 (0.00052) |
| pre_final (pretrained sequence encoder) | 0.35900 | 0.689 | -0.00021 (0.00030) | -0.00078 (0.00042) | -0.00109 (0.00059) |
| pre2_final (val near-identical to pre_final) | 0.35900 | 0.689 | -0.00021 | -0.00078 | -0.00109 |

Every GBDT ranker objective we have (YetiRank, YetiRankPairwise, XGB rank:ndcg) **hurts v11 at every weight**, monotonically
in w. They are 0.03 NDCG weaker than v11 and only 0.77-0.81 correlated: the weakness dominates the diversity. In the
winning designs that blend GBDTs with NNs, the members are close in quality (RecSys'24 1st place: transformer AUC 0.8864
vs LightGBM 0.8817 vs CatBoost 0.8805 [R research_kaggle_solutions.md]); ours are not. A new GBDT member would need
to be within ~0.005 of the NN to have a chance; the best GBDT trained so far is at ~0.369-0.370 val (CURRENT_STATUS 3), 0.028 below v11.

### 2.6 Cooldown pool composition (proxy of the WSD exploration mix) [E]

`cd_pool.py`: the 300K-feed CE base of §2.1, but in the last 20% of the steps a batch is replaced by an exploration-pool
batch with probability 0.3 (~200 pool batches ≈ 0.5 pass over the pool, so no memorisation here). The arm without a
pool reproduces the §2.1 CE run bit-exactly (0.37274 / 0.37231).

| pool | feeds in pool (Sep 9-19, proxy holdout excluded) | val | Δ val (SE) | Δ holdout (SE) | M<50 | M>=50 |
|---|---|---|---|---|---|---|
| none | 0 | 0.37235 | -0.00100 (0.00023) vs rand50 | -0.00113 (0.00015) | -0.0010 | -0.0010 |
| **rand50: p > 0.5, M >= 50 (current NN definition)** | 108,040 | **0.37335** | 0 | 0 | | |
| rand50lo: p > 0.2, M >= 50 | 156,204 | 0.37335 | +0.00000 (0.00014) | -0.00015 (0.00009) | +0.0003 | -0.0002 |
| randall: rand50 + short p > 0.5 feeds | 367,036 | 0.37282 | **-0.00053 (0.00016)** | -0.00042 (0.00010) | -0.0008 | -0.0003 |

**Reading.**
1. The proxy reproduces the NN's cooldown-mix gain in direction and size (+0.0010 here; N2 continuation +0.0018 per NN
   member), a fourth calibration point: the proxy does see data-mixture effects, it does not see aux-task effects.
2. **Short "random-like" feeds do not belong in the cooldown pool** (-0.00053 +- 0.00016, and -0.0008 on the M<50 slice
   they were supposed to help). From scratch they beat production feeds (§2.3), but as a late-phase pool they dilute
   the M>=50 exploration feeds. The NN's M>=50 restriction is right.
3. **A looser threshold (p > 0.2) costs nothing** (+0.00000 +- 0.00014) and gives 45% more distinct pool feeds in this
   window (all days: 227,477 vs 164,118; after the 30K holdout ~197,500 vs 134,118 = +47%). That is the cheap lever against the memorisation that limits the
   cooldown (proposal P3); its benefit can only be measured on the NN.

---------------------------------------------------------------------------------------------------------------------

## 3. Literature mapped onto our setup (extensions only)

### 3.1 Ranking losses (task item 1)

| loss family | evidence that it beats softmax CE on NDCG | our data | verdict |
|---|---|---|---|
| Softmax CE variants (temperature, LS, focal, anchor) | Bruch et al. ICTIR'19: CE bounds NDCG/MRR for binary relevance [R]; temperature is absorbed by the logit scale; LS / focal untested in LTR literature that I found | §2.1: LS +0.00015 +- 0.00016, focal +0.00011 +- 0.00016, anchor -0.00045 (worse on M<50) | keep CE; LS does not even work as a memorisation guard (§2.4) |
| LambdaLoss / NDCG-Loss2++ / LambdaLoss@K | Wang et al. CIKM'18: small gains on graded LETOR; Jagerman et al. SIGIR'22 LambdaLoss@K better than other losses for NDCG@K on LETOR [V abstract]; DASALC App. B.2: LambdaRank worst in neural nets [R] | §2.1: l2pp +0.00006 +- 0.00028 (= CE), lambdarank -0.00057; our metric is full-list (no K), so @K variants do not apply | no |
| NeuralNDCG / ApproxNDCG / PiRank (differentiable sorting) | allRank: NeuralNDCG ≈ LambdaRank@max ≈ ListNet within 0.7 NDCG@10, τ tuned on test [V]; PiRank beats LambdaRank on 13/16 metrics, aggregator numbers only [V secondary]; Bruch SIGIR'19 GumbelApproxNDCG ≈ softmax [R] | §2.1: approx -0.00295 +- 0.00088 (M<50 -0.0052), neuralndcg -0.00165 +- 0.00066 (holdout -0.00236), 40x the compute | no |
| ListMLE / Plackett-Luce | allRank: ListMLE worst listwise loss (Web30K 51.82 vs ListNet 52.80) [V]; Avito's production L3 transformer uses ListMLE, but their comparison is against BCE only [V habr 1090584] | §2.1: listmle -0.00516 +- 0.00079 (worst of all); plk = ce | no |
| Pairwise logistic (RankNet) | worse than softmax in neural LTR (allRank, DASALC) | §2.1: -0.00033 +- 0.00027 | no |
| Mixtures (CE + BCE, CE + pairwise) | binarised Web30K/Yahoo/Istella: Sigmoid+Softmax > Softmax by 0.4-0.9 NDCG@10 points [R deep_neural_rankers §8] | §2.1: ce_bce -0.00128 +- 0.00052 (M<50 -0.0034); ce_l2pp +0.00015 +- 0.00026 (= CE) | no |
| Top-K softmax losses for recsys (SL@K, KDD'25; Talos 2026) | +6% avg on sampled top-K CF benchmarks [V abstract] | they target truncated top-K retrieval metrics over full catalogues; our metric is full-list over ~96 candidates | no |

**Conclusion for losses:** the current loss is the right one. None of the 14 alternative surrogate losses gives a measurable gain, and
the theory says none should for binary relevance with 1-2 positives. Do not spend GPU on surrogate-loss swaps.

### 3.2 Training paradigms (task item 2)

| paradigm | mechanism on our data | evidence | status / verdict |
|---|---|---|---|
| Unbiased LTR (IPS, DLA, doubly robust, PAL) | correct the production position bias of the 3.15M train feeds | IPW: XGBoost -0.0022 [R]; PAL tower: NN neutral, proxy -0.0006 val / +0.0002 holdout (§2.1); Baidu-ULTR: loss choice matters more than any correction; pointwise IPS worse than naive [R deep_position_bias_random §3.9]; DR (Oosterhuis TOIS'23) needs a click model of examination and only reduces variance relative to IPS [V abstract] | **closed.** We already have the better remedy: an unbiased yardstick plus exploration feeds in the cooldown, and production order used as a *teacher* (K9), not as a nuisance |
| Multi-task (contact + production order + exploration domain) | shared representation shaped by extra targets | K9 +0.0024 [R]; domain token neutral; category-intent head -0.0014 [R]; propensity head null [R nnx_VERDICT] | K9 is the only multi-task win; more heads have no signal source left (no other labels exist) |
| Contrastive / SSL pretraining of the history encoder | content and ID co-occurrence from 463M history events | pre_* features exist and are leak-checked; their value is the ID part only; train-row use needs windowed encoders (pre2s, live test W2) [R pretrain_report] | **in progress** (cvnd_pre2s). Initialising the NN history encoder from the pretrained encoder is leaky unless the windowed encoders are used, and the NN already learns content from 3M labelled feeds |
| Generative recommenders (HSTU, OneRec, TIGER) as members | next-item generation over the catalogue | OneRec/TIGER solve retrieval+generation, not re-ranking of a fixed list; the HSTU-like scorer we have (pre_final, 0.359) hurts v11 at every weight (§2.5) | no |
| Cross-encoder vs bi-encoder | candidate attends to history | the NN is already a cross-encoder (target attention + retrieval top-16 + set attention); bi-encoder scores (cos_u, pre_final) are 0.03-0.04 weaker | done |
| Set-to-set listwise rerankers (PRM, SetRank, RankFormer) | candidates attend to each other | set attention neutral in our NN [R nn_report round 1]; Avito production reports +0.6% from candidate self-attention [V] | done |
| DIN / DIEN / SIM / TWIN long-history attention | target attention over long histories | H <= 256, full attention is already affordable; TabR/TWIN-like top-16 retrieval gave +0.0028 [R nnx_VERDICT] | done |
| Denoising / mixup for ranking | regularisation | no LTR evidence; breaks feed semantics [R training_gen §4.6] | no |
| SAM / ASAM | flatter minima, robustness to shift | SAMRec (SASRec), Helen (CTR) [V abstracts]; never run here | **fine-tune only, P2** |
| SWA / EMA / model soups | weight averaging along one trajectory or across cooldown branches | stage-1 EMA null [R]; fta (cos + EMA fine-tune) +0.0011 [R]; plateau snapshots (W1) and cooldown continuation (N2, cd12 +0.0008 n.s.) in progress | in progress elsewhere |
| Test-time ensembling | MC-dropout passes, history-length TTA | history-length TTA worse [R]; permutation TTA meaningless (equivariant model) | MC-dropout: P4 (GPU minutes) |
| Test pseudo-labels / self-training | adapt to the stale test period | -0.0004 at fine-tune level [R]; self-training on 1-2 noisy positives per list has no published ranking win I could find | no |
| Training on val labels at the end | more exploration-labelled data | 5-fold fine-tune (in place), full-val refit (in progress), anneal-mix (P6 below) | see P6 |

### 3.3 Overall solution design (task item 3)

What the closest winning designs do, and what transfers:

| design element | winners | our evidence | transfer |
|---|---|---|---|
| Stage-1 NN + GBDTs, stage-2 stacker, stage-3 average | RecSys'24 1st (DeNA): transformer + LightGBM + CatBoost, LightGBM/Optuna stacker, simple average; members within 0.006 AUC of each other [R] | GBDTs 0.03 below the NN, every pre-declared blend weight hurts (§2.5); residual stacker on the NN adds -0.0001 [R] | no: our members are not close in quality |
| NN + CatBoost rank blend 50/50, no tuning | Avito production (+0.7% contacts) [V habr 1090584] | their CatBoost has ~200 production features we do not have; ours is 0.03 weaker | no |
| Many GBDTs on heavy features | H&M 1st (5 LightGBM + 7 CatBoost), OTTO (XGB/LGBM rankers), Avito ML Cup 2025 1st (fold-averaged CatBoost YetiRank) [R] | candidate generation is given here; the feature space is saturated relative to the NN (FEATURES_REVIEW) | no |
| Retrain on the latest labelled period with the chosen hyper-parameters | RecSys'24 1st (+recency), Kaggle practice | val fine-tune (5-fold) + full-val refit in progress | yes (in progress) |
| Equal-weight blends of several seeds/recipes of the best family | DASALC-ens (+0.8-1.3 NDCG@5), Riiid, our nn_final4/5 | plateau members gain +0.0047 from 6-way ensembling [R REVIEW_SYNTHESIS P1] | yes (current design) |
| Pseudo-labelling of test | common in classification competitions, rare in ranking | -0.0004 [R] | no |

**Design verdict.** The current design (one strong NN family, many seeds/recipes, a train-only half and a val
fine-tuned half, equal-weight z-means, holdout-selected member sets) is what the evidence supports. The winners'
multi-family stacks work when families are within ~1% of each other; here the best non-NN family captures only ~72%
of the NN's lift over random (0.368 vs 0.398, random 0.290). Spend remaining effort on NN-internal levers
(cooldown, inputs, fine-tune hygiene), not on new solver families.

---------------------------------------------------------------------------------------------------------------------

## 4. Ranked proposals (exact experiment specs)

Common rules for every GPU arm below: code copy `nn_x/solvers/` (copy only `nn/*.py` and `nn/*.sh`; symlink
`nn_x/solvers/feat_cache -> <repo>/nn/feat_cache`; audit every output path so checkpoints, logs, preds
and holdout preds go to `nn_x/solvers/{ckpt,logs,preds}`; `nn/common.py:13` accepts only 4/5/6/7, keep that guard);
`CUDA_VISIBLE_DEVICES=N`, at most 3 workflow processes on GPU N, `OMP_NUM_THREADS=16`. Judge with `nn/pcmp.py`-style
slices (all, M<50, M>=50, Tue, Thu; holdout all + late) and the unified acceptance rule (train-only paired val > 2 SE
pooled over >= 2 seeds AND holdout >= 0; fine-tune claims: >= 2 FT seeds or z >= 2.5).

The honest summary first: the solver space is close to exhausted. The loss is right, the paradigm is right, and the
design is right. What remains are small, cheap refinements of components that have already shown large effects.

| rank | proposal | expected gain (val / private) | cost (GPU N, shared) | overfit risk |
|---|---|---|---|---|
| P1 | Tune the production-order distillation (K9 τ and target form): the only loss-side component with a large measured effect (+0.0024), never tuned | +0.0003 to +0.0012 per member | 4 x 2-epoch runs (~3 GPU-h) + 1 x 10-epoch confirmation pair | low (2 pre-declared arms) |
| P2 | SAM (ρ 0.05) in the fta fine-tune (rated "medium" in training_gen §9.6, never run) | 0 to +0.0008 on the fine-tuned half (~+0.0004 on the mix) | ~2x fine-tune cost: 6 members x ~20 min x 2 FT seeds (~4 GPU-h) | low |
| P3 | Cooldown exploration pool threshold p_random > 0.2 instead of > 0.5 (M >= 50): +47% distinct feeds against pool memorisation | 0 to +0.0008 on cooled members (only if memorisation is the binding limit) | 2 cooldown continuations (~1.5 GPU-h) | low |
| P4 | MC-dropout test-time averaging (8 passes) on existing checkpoints | -0.0005 to +0.0005 | GPU minutes | none (one pre-declared setting) |
| P5 | One NDCG-sensitivity-weighted-CE (`ce_rw`) member pair, judged only at ensemble level | 0 to +0.0005 (proxy hint z 1.3) | 2 x 10-epoch members (~8 GPU-h) | medium (single-model holdout worse) |
| P6 | Anneal-mix: val fold-train feeds mixed into a cooldown continuation, then the usual fta fine-tune | 0 to +0.001 (uncertain; `--add_random` evidence is neutral) | 5 continuations per member (~3 GPU-h per member) | medium |

### P1. Production-order distillation (K9) tuning

**Mechanism.** The K9 aux head (CE toward softmax(-pos/τ) on production feeds, weight 0.5·(1-p_random), annealed to
0.1 in the cooldown) transfers the production ranker's ordering. Avito's production L2/L3 stack (CatBoost on ~200
features + a ListMLE transformer [V habr 1090584]) sees item/seller statistics we do not have, so its order is the
richest extra "label" in the data. K9 was worth +0.0024 val / +0.0027 holdout over MSE distillation
(nn_report.md:141-142), the largest loss-side gain in the project. Yet `aux_tau` = 10 and `aux_pos` = 0.5 in all 92 /
114 logged runs (`grep -h '"aux_tau"\|"aux_pos"' nn/logs/*.json | sort | uniq -c`; only `nopos_aux1`, weight 1.0, was
started and stopped at 6k steps). With τ = 10 the target puts ~63% of its mass on the top-10 production slots
((1-e^-1) for contiguous slots) and almost none on the order of slots 20-80. A flatter target (τ 30) transfers more
pairwise order; a Plackett-Luce target transfers the exact order of the head.

**Why not decided on the proxy:** the proxy cannot see this kind of effect (K9 proxy +0.00003 vs NN +0.0024, §2.1);
in the 600K-feed proxy K9 gave +0.00046 (§2.3), same direction, too small to rank variants.

**Spec.**
1. `nn_x/solvers/train.py`: add `--aux_mode pl` = Plackett-Luce log-likelihood of the production permutation truncated
   to the first 20 slots (sort candidates by `pos`; for k = 1..min(20, M): logit_k - logsumexp(logits of slots >= k)),
   per-feed weight (1-p_random), same `aux_w(step)` schedule; next to the existing `listwise` mode (`nn/train.py:303-307`).
2. Base = the 2-epoch recipe used for nnx arms, updated to the live recipe: `--no_pos --aux_pos 0.5 --aux_mode listwise
   --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd --muon
   --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.1 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2`,
   seeds 23 and 24 (reuse an identical-flag base if it exists).
3. Pre-declared arms (2 seeds each): (a) `--aux_tau 30`; (b) `--aux_mode pl`. Only if one of them is positive: (c) the
   winner with `--aux_pos 1.0`.
4. Accept with the unified rule; then one 10-epoch confirmation pair (seed 23, recipe of nn_final6) before seeding
   new members with it.

**Expected:** +0.0003 to +0.0012 per member (a refinement of a channel that gave +0.0024). **Risk:** the holdout is
production-distribution and may reward stronger production-order fit; the train-only-val-primary rule protects
against that.

### P2. SAM in the fta fine-tune

**Mechanism.** The fine-tune fits 11.5K feeds per fold for 4 epochs; fine-tune A/A spread is 1.4x the per-feed SE
(REVIEW_SYNTHESIS P4), so it is noisy and overfits fast. SAM (Foret et al. 2021) seeks flat minima; in recsys, SAMRec
reports that SASRec-type transformers converge to sharp minima and that SAM matches contrastive regularisers
[V abstract]. Rated "medium, never run" in nnx_research_training_gen.md §9.6.

**Spec.** `nn_x/solvers/finetune.py --sam 0.05`: per step, g = ∇L(w); e = ρ·g/||g|| over all trainable parameters
(the zero-init extra branch included); w ← w + e; second forward/backward; w ← w - e; AdamW step with the second
gradient; EMA and cosine schedule unchanged. Run on the six nn_final5 members (`--extra n2c_pre --lr 5e-5 --epochs 4
--decay cos --ema 0.99 --ema_start 0.5 --sam 0.05`) with 2 FT seeds, outputs in nn_x/solvers/preds. Compare the
6-member fta ensemble OOF and the 0.5/0.5 mix with v11, paired, all slices. One pre-declared ρ; ρ = 0.02 only as a
follow-up if 0.05 is positive but < 2 SE.

**Expected:** 0 to +0.0008 on the fine-tuned half. **Risk:** low.

### P3. Larger M>=50 cooldown pool (p_random > 0.2)

**Mechanism.** The cooldown's exploration mix is a proven training lever (N2 continuation +0.0018 per member), and its
limit is memorisation of a 134K-feed pool (P1 of REVIEW_SYNTHESIS; pool loss 2.43 vs regular 2.79,
nn_report:500). The wide-MLP proxy (§2.4) shows that a loss regulariser does not stop memorisation; more distinct feeds
per pass is the remaining lever. Lowering the threshold from 0.5 to 0.2 grows the M>=50 pool from 164,118 to 227,477
train feeds (counted from `train_prandom.npz`); after removing the 30K holdout (all of it inside both pools) the
cooldown pool grows from 134,118 to ~197,500 feeds (+47%). The proxy (§2.6) measures what this
costs in pool quality: with p > 0.2 the cooldown proxy is unchanged (val +0.00000 +- 0.00014, holdout -0.00015 +- 0.00009 vs the p > 0.5 pool, 3 seeds), i.e. the extra lower-confidence feeds cost nothing in quality. The memorisation benefit itself cannot be measured on the proxy and needs the GPU pair.

**Spec.** `nn_x/solvers/train.py --pool_thr 0.2` (replace the constant 0.5 at `nn/train.py:260`; keep the holdout
exclusion and the M<50 zeroing at :160). N2-style continuations (`--init <ckpt> --muon --lr 2e-3 --sched wsd
--cooldown_frac 1.0 --warmup 0 --cooldown_mix 0.1 --aux_anneal 0.1 --epochs 1`, same seed as the existing v15_cd run)
from two nn_final5 plateau checkpoints (v12_all_e10, v12_all_s81), paired against the existing v15_cd_* continuations
(train-only val, holdout, late holdout, M<50 slice). Two checkpoints count as two seeds.

**Do not** add the short (M<50) "random-like" feeds to the pool: §2.6 shows that this hurts (-0.00053 +- 0.00016,
including on M<50).

### P4. MC-dropout test-time averaging

**Mechanism.** Averaging K stochastic passes approximates an ensemble of thinned networks; our model has dropout 0.1.
**Spec.** `nn_x/solvers/predict.py --mc 8`: `model.eval()`, then switch only `nn.Dropout` modules to train mode,
8 passes with fixed seeds, per-feed z-mean; apply to the six nn_final5 train-only checkpoints; compare the 6-member
train-only ensemble with the deterministic one on val and the holdout. **Expected:** -0.0005 to +0.0005
(training_gen §10). **Risk:** none (one setting). Only if GPU minutes are free.

### P5. NDCG-sensitivity-weighted CE as an ensemble member (low)

**Mechanism.** `ce_rw` weights each feed's CE by dNDCG/dlog r at its positives' current rank (detached, batch mean 1):
feeds whose positive is already near the top get more weight. In the proxy it is the only loss whose pairing with CE
beats a CE seed pair (+0.00056 +- 0.00042, z 1.3, within-feed corr 0.957, 6 seeds), with equal single-model val but a
clearly worse holdout (-0.00119 +- 0.00036). **Spec.** `listwise_loss(..., norm="rw")` in nn_x/solvers/model.py
(weights from `torch.argsort` ranks of the detached logits; mean over valid feeds = 1); two 10-epoch members (seeds
23/24, nn_final6 recipe); judged only as an addition to the nn_final6 ensemble with equal member counts (6 CE vs
4 CE + 2 ce_rw), member set chosen on the holdout as in the frozen final-pick protocol. **Risk:** medium; spend GPU on
it only after P1-P3.

### P6. Anneal-mix (val labels inside a cooldown continuation) (low)

**Mechanism.** LLM practice up-samples small high-quality target-domain data during the final LR anneal rather than
fine-tuning afterwards [K: Llama 3 / OLMo 2 reports]. Here: an N2-style 1-epoch continuation whose pool is the
exploration pool plus the val fold-train feeds (val share 3% of batches ≈ 2-3 passes), then the usual fta fine-tune
on the same fold; fold k's model never sees fold k; test = fold mean. **Evidence against:** mixing 12K train
exploration feeds into each fine-tune epoch (`finetune.py --add_random 12000`) was neutral on v2 members (+0.0003 /
+0.0001 / -0.0009, nn_report round 1), which is the same mixture seen from the other side. **Cost:** a second
SplitData (val) inside train.py, 5 continuations per member. **Risk:** medium (strict fold hygiene needed).

### Do-not list (evidence in this report)

| idea | evidence |
|---|---|
| Swap the main loss (LambdaLoss / NDCG-Loss2++, LambdaRank, RankNet, ApproxNDCG, NeuralNDCG, ListMLE, CE+BCE, focal, calibrated anchor, PL multi-positive, LS) | §2.1: no alternative beats CE; most are worse, especially on M<50 |
| Loss-diverse ensemble members (except the P5 probe) | §2.2: no pairing beats a CE seed pair; real NN base+bce -0.00033 +- 0.00059 vs base+nopos |
| Label smoothing as a memorisation guard | §2.4: does not stop the collapse in the wide-MLP proxy |
| Short (M<50) "random-like" feeds in the cooldown pool | §2.6: -0.00053 +- 0.00016 vs the M>=50 pool, -0.0008 on M<50 |
| Up-weighting exploration feeds in the main run | NN: -0.001 (pw_aux_sim, nn_report round 1) |
| GBDT ranker members (YetiRank, YetiRankPairwise, rank:ndcg, LambdaMART) | §2.5: all hurt v11 at w 0.05-0.2 |
| ULTR corrections (IPS, DLA, DR, PAL tower) | §2.1 position tower -0.0006 val / +0.0002 holdout; XGB IPW -0.0022 [R]; Baidu-ULTR [R] |
| Generative / sequence-encoder members (HSTU-like pre_final) | §2.5: -0.0008 at w 0.1 |
| Test pseudo-labels, mixup, history-length TTA | nn_report N6 -0.0004; training_gen §4.6; nn_report K10 |


---------------------------------------------------------------------------------------------------------------------

## 5. Checked and correct

- **Proxy harness reproduces the ideas review bit-exactly** (CE val 0.37274 / 0.37231 / 0.37200 for seeds 1-3 vs
  `review/ideas/exp_loss_300k.log`), so the earlier per-positive null and these results share one baseline.
- **Loss implementations sanity-checked:** `plk`'s per-feed loss is identical to `ce` on K=1 feeds (the K=1 slice differs by -0.0001 only through the shared weights);
  `l2pp`'s δ term had a 0/0 at the self-pair, fixed by clamping |r_p - r_n| >= 1 before use (the self-pair is masked);
  `p_random` is NaN for 400,000 train feeds in `train_prandom.npz` (the feeds used to fit the detector's proxy,
  eda_labels_position.md:211) and is treated as 0 (production), exactly as `nn/train.py:159` does, which is the conservative choice for the aux and position-tower weights. NeuralNDCG uses the
  NeuralSort formula of Pobrotyn & Bialobrzeski eq. (3)/(5) with Sinkhorn scaling, verified against the paper text.
- **The current NN loss is exactly the CE baseline that no alternative beat:** `nn/model.py:513-534` (target y/K, other
  positives in the denominator, `--loss_norm feed` = equal per-feed weight), and `--loss both` in `nn/train.py:292`
  uses the batch-mean normalisation (not feed), so any past "both" run is not an equal-feed-weight CE+BCE test.
- **GBDT member test is clean:** all inputs are LUO-trained GBDTs (CURRENT_STATUS 2b), weights pre-declared, val-only
  scoring; the v11 base reproduces 0.39793 (CURRENT_STATUS 3j).
- **Cooldown-pool proxy reproduces the CE base bit-exactly** with the pool switched off (`cd_pool.py --pool none`:
  0.37274 / 0.37231 / 0.37200), so §2.6's deltas isolate the pool.
- **pre2_final_val vs pre_final_val:** different files (md5 bab8cd05 vs 072aaae3) with the same val NDCG 0.3590, as
  documented (pretrain_report.md:257); not a copy error.

---------------------------------------------------------------------------------------------------------------------

## 6. Sources

Read this session [V]:
- Pobrotyn & Bialobrzeski 2021, *NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting*, https://arxiv.org/pdf/2102.07831v2.pdf (Table 3; τ = 1 tuned on the test set; Sinkhorn scaling).
- Jagerman et al. SIGIR'22, *On Optimizing Top-K Metrics for Neural Ranking Models*, https://research.google/pubs/on-optimizing-top-k-metrics-for-neural-ranking-models/ (abstract).
- Yan et al. KDD'22, *Scale Calibration of Deep Ranking Models*, https://research.google/pubs/scale-calibration-of-deep-ranking-models/ (abstract; calibrated ranking losses keep ranking quality).
- Yang et al. KDD'25, *Breaking the Top-K Barrier* (SL@K), https://arxiv.org/pdf/2508.05673 (abstract).
- Swezey et al. NeurIPS'21, *PiRank*, https://arxiv.org/abs/2012.06731 (abstract; numbers via secondary aggregators only).
- Bruch et al. SIGIR'19, *Revisiting Approximate Metric Optimization in the Age of Deep Neural Networks*, https://research.google/pubs/pub48168/ (abstract).
- Oosterhuis TOIS'23, *Doubly-Robust Estimation for Correcting Position-Bias in Click Feedback for ULTR*, https://arxiv.org/abs/2203.17118 (abstract).
- Avito (Habr, Oct 2026), main-page ranking stack, https://habr.com/ru/companies/avito/articles/1090584/ (ListMLE vs BCE only; candidate self-attention +0.6%; 50/50 untuned rank blend with CatBoost).
- RecSys Challenge 2024 overview, https://arxiv.org/pdf/2409.20483, and 1st-place slides, https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning (3-stage design).
- Trivago RecSys Challenge 2019: 2nd place https://www.cs.toronto.edu/~cheng/projects/recsys_challenge_2019/ (GBDT + deep rerankers).
- Yandex Personalized Web Search: Dataiku https://blog.dataiku.com/2014/01/14/winning-kaggle, Volkovs https://www.arxiv.org/pdf/1502.00527.
- SAM in recommenders (abstracts): Helen https://arxiv.org/html/2403.00798v1, SAMRec https://arxiv.org/pdf/2308.10347v1.pdf.

Cited from earlier project reports [R]: Bruch et al. ICTIR'19 (softmax CE bound), DASALC (Qin et al. ICLR'21) App. B.2,
LambdaLoss (Wang et al. CIKM'18), Google KDD'23 confounding, Baidu-ULTR SIGIR'24, RecSys'24 / OTTO / H&M / Avito ML Cup
2025 solutions - see `research_sota_ltr.md`, `deep_neural_rankers.md`, `deep_position_bias_random.md`,
`research_kaggle_solutions.md`, `deep_ru_competitions.md`.

