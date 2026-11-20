# LIT_1115 / lit_moe: mixture of experts and gating in ranking models (literature scout)

Written 2026-11-15T14:55Z. CPU only, no training, no GPU process. Nothing was computed by me; the single [M] item is a grep of the code tree (section 4, N1).
Marks: [M] measured by me; [D] read in a project file (path given); [W] read on the web in this session (through a page-to-text fetch tool that summarises the page; I did not open the PDFs, so a [W] number is "as extracted", cross-checked only where stated); [K] my own knowledge, not re-read today; [I] inferred.
Search log (generic queries only, no project detail): `nn_x/w12/lit/lit_moe/search_log.txt`.

## 0. Verdict

1. **The shortlist is empty.** No MoE-type idea published 2022-2026 passes all three filters (not multi-task-only, cheap to bolt on, replicated or cleanly ablated) AND survives the project's own measurements. Nothing found reaches a prior of 0.03 for a pool-level gain of +0.0003 [I].
2. **MoE has been checked here, in 12 forms** (section 1): in-ranker sparse MoE, implicit ensemble (TabM form), cold gates, feed-level routers, per-vertical experts, trajectory-cluster experts, second-level conditional weights, a learned gate over the four pools, the feed-size rule (public -0.00015), a task-conflict pre-gate, event-type heads, category / domain heads. One gate is in the recipe (attention output gate).
3. **What the 2024-2026 literature says, in three sentences.**
   - Sparse / per-token MoE (RankMixer, TokenMixer-Large, SORT, Zenith, MTmixAtt) is a way to add parameters at constant FLOPs at 9e9 .. 1e12+ training rows; by the authors' own tables the sparse model *matches* the dense model of the same size, it does not beat it. Here the dense scale-ups are measured null or negative under distillation (FFN x4: -0.00068 val / -0.00088 holdout at the asymptote [D]).
   - Multi-gate MoE (MMoE, PLE, HoME, SMES, HA-MoE) repairs conflicts between 8-20+ tasks; this ranker has one task plus one auxiliary whose gradient is orthogonal to the main one [D].
   - Multi-scenario gating (STAR, PEPNet, AdaSparse, M3oE) needs a scenario variable that varies at inference; in an independent 6-dataset benchmark the best gain over a shared-bottom net on the three datasets above 10M interactions is +0.0028 / +0.0025 / +0.0003 AUC, no model wins everywhere, and PPNet is below shared-bottom on 4 of 6 datasets [W].
4. **Competitions:** no winner write-up of a feed / session ranking task that I could read reports MoE as a source of gain (section 3.7). The organiser's own engineering post reports a DCNv2 + MoE ranker at -3% NDCG against their tree model [D].
5. **Recommendation: spend no GPU on MoE.** If a direct closure of "sparse MoE FFN under the current distilled recipe" is wanted for the record, it costs 8 from-scratch students and I expect a null (P(pass) about 0.02-0.03 [I]); section 4 states the rule.

## 1. What the project has already measured (the bar every paper has to clear)

| # | MoE / gating form | result | file |
|---|---|---|---|
| 1 | Sparse MoE FFN inside the ranker (A5: 8 routed SwiGLU experts of width 64 + a shared expert, top-2 sigmoid router, 4 candidate blocks, +0.72M parameters) | alone val +0.00027 +- 0.00047, holdout +0.00053; on top of gate + retrieval +0.00015; after fine-tune -0.00034; router learned vertical specialisation (0.33-0.50 nats with category) and it was worth 0; one router collapsed [D] | reports/nnx_VERDICT.md l.91, l.149; reports/SOLVER_POWER_1113.md 4.3 |
| 2 | Upcycled MoE in the cooldown | closed without a run: 32 stored cooldown-only arms have diversity margin median +0.0016 against a bar of 0.10 [D] | reports/SOLVER_POWER_1113.md 4.3 |
| 3 | Implicit ensemble in one model (TabM-mini, k = 4) | val +0.00009 (3 seeds), members 0.999-correlated, 1.45x cost [D] | reports/nnx_VERDICT.md l.154 |
| 4 | Cold / warm gates and routers | oracle +0.0257 is a label echo; best label-free gate AUC 0.65-0.67 but value-weighted AUC 0.52-0.53; nested -0.00005 +- 0.00004 [D] | reports/COVERAGE_MAP.md l.281 |
| 5 | Feed-conditional routing between pools (31 label-free covariates) | router AUC 0.494-0.511 vs permuted null 0.496-0.503; val-fitted router on the holdout -0.0025 (z -6.0) [D] | reports/COVERAGE_MAP.md l.280 |
| 6 | Per-vertical experts / adapters | 99.2% of feeds multi-vertical; per-vertical minus global -0.00095 +- 0.00039 [D] | reports/COVERAGE_MAP.md l.283 |
| 7 | Trajectory-cluster conditioning (per-cluster weights, FiLM, MoE) | 0 of 8 arms; best +0.00022 +- 0.00025, below the random-label control [D] | reports/COVERAGE_MAP.md l.192; reports/PRETRAIN_PLAN_1113.md l.325 |
| 8 | Second-level combiners with feed-conditional weights | 0 of 78; val-fitted weights anti-transfer (z -2.85) [D] | reports/COVERAGE_MAP.md l.219 |
| 9 | Learned gate over the four pools (7 families, nested by user) | nested minus the feed-size rule -0.00022 +- 0.00021 and -0.00038 +- 0.00016; optimism of every family with more than one free weight 0.0004-0.0010 [D] | reports/MIX_1115/gateB.md section 0, 3 |
| 10 | Feed-size rule (popularity-light weight 0.4 on M >= 50) | public -0.00015 (-0.5 SE): closed [D] | reports/CURRENT_STATUS.md log 14:34Z |
| 11 | Task-conflict pre-gate for Progressive Layered Extraction / PCGrad (C15) | main vs auxiliary gradient cosine: median of medians +0.007..+0.014, PROCEED = False [D] | reports/COVERAGE_MAP.md l.272 |
| 12 | Event-type heads; category-factorised head; domain heads | no type label at train time; category head -0.0010 val / -0.0012 holdout; domain constant at inference [D] | reports/SOLVER_POWER_1113.md 4.5; reports/COVERAGE_MAP.md l.214, l.168 |
| 13 | Attention output gate (A1) | ADOPTED: +0.00065 mean over 4 pairs (SE about 0.0003), mostly absorbed by the fine-tune [D] | reports/nnx_VERDICT.md l.93 |
| 14 | Dense capacity under distillation (upper bound for any sparse MoE of the same size) | FFN x4 (2 seeds, val / holdout): member -0.00107 / -0.00040, asymptote -0.00068 / -0.00088 +- 0.00058; d256 asymptote -0.00045 / -0.00106 +- 0.00061; only width 192 moved the asymptote (+0.0002 val / +0.0007 holdout) [D] | reports/RECHECK_1114/kd_reprice.md l.56-58; reports/COVERAGE_MAP.md l.300 |

Naming trap: "PLE" in reports/COVERAGE_MAP.md l.61 and nnx_VERDICT B1 is *piecewise-linear encoding* of numeric inputs, not Progressive Layered Extraction. The multi-task PLE was never run in the network; it was closed at the pre-gate of row 11.
Scope note on row 1: A5 ran once (seeds 23 / 24) in the 2-epoch AdamW era. A sparse MoE FFN has never been run under the current recipe (Muon, 10 epochs, distillation, d192). Row 14 bounds it by inference, not by a direct run.

## 2. Scale of this task against the literature

| | training rows | tasks | model size |
|---|---|---|---|
| this project | 3.15M feeds / 246.7M candidate rows / about 5.45M positives [D reports/SOLVER_POWER_1113.md 5.1] | 1 (+ 1 auxiliary) | 1.0M (d128), 2.1M (d192) [D] |
| RankMixer | "trillions of records per day", 2-week window [W] | multi (finish, skip, ...) | 100M-1.1B |
| TokenMixer-Large | 300M-17B records per day, up to 2 years [W] | CTR + CVR | 500M-15B |
| HoME | 50B logs per day [W] | 8 reported, "20+" in production | not read |
| OneTrans | 29.1B impressions [W] | CTR + CVR | 24-91M+ |
| SORT | 9B impressions [W] | click, cart, purchase | 14M-685M |
| MTmixAtt | 786M users, 2 months [W] | CTR + CTCVR, multi-scenario | 15M-1B |
| KDD Cup 2026 (UniRec) | 35M samples [D reports/research_1112/recsys.md] | 1 (pCVR) | small |
| D-MoE (Criteo / Avazu) | 45M / 40M rows [W] | 1 (CTR) | small |

Only the last two rows are within 10x of our data. In both, an architecture mechanism is worth 1e-4..1e-3 AUC, and neither reports seeds for the MoE part.

## 3. Scan by family

### 3.1 Multi-gate lineage: MMoE, PLE, HoME and successors

| work | setting | experts and gate | isolated gain vs their noise | transfer to a single-task listwise ranker |
|---|---|---|---|---|
| **HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou**, KDD 2025, https://arxiv.org/abs/2408.05430 | multi-task (8 tasks in the table), 50B logs / day, offline + online | experts = MLPs shared / per task group / per task; gates = per-task softmax; fixes: BatchNorm + Swish on every expert output (against expert collapse: one expert 90%+ zero activations yet gate weight > 0.98), hierarchy mask, feature-gate, self-gate | expert norm + Swish (MMoE -> MMoE*) +0.02..+0.23 pt AUC; feature-gate +0.02..+0.09; self-gate -0.01..+0.06; they call 0.03-0.05 pt significant; online play time +0.17..+0.47% [W]. Headline average +0.52% GAUC [D reports/nnx_research_recsys_arch.md l.199] | NO. Every component is defined across tasks. The one portable piece (normalise expert outputs, Swish) repairs a collapse that needs a softmax gate over ReLU experts; A5 used SwiGLU experts with a sigmoid router and still gained 0 [D]. |
| **SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity**, arXiv 2602.09386 (2026), https://arxiv.org/abs/2602.09386 | multi-task, Kuaishou + KuaiRand-1K | sparse experts; instance-level shared subset + task-adaptive subset; load-balance regulariser | numbers not read (search summary only) [W] | NO: the problem it solves (union of per-task top-k grows) does not exist with one task. |
| **Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study (HA-MoE)**, arXiv 2607.27577 (2026), https://arxiv.org/abs/2607.27577 | multi-task (K = 11 per a search summary), mixed-content feed (articles, videos, user posts) | multi-gate MoE with content-type context fed to gates and experts | abstract only: "consistent improvements", no magnitudes [W] | NO evidence to transfer. The closest thing to our case (a feed mixing heterogeneous verticals) but multi-task, unquantified, and our per-vertical test is negative (row 6). |
| MMoE (KDD 2018), PLE (RecSys 2020), AdaTT (KDD 2023) [K] | multi-task | task gates over shared (+ task-specific) experts | PLE's claim is the removal of the seesaw between conflicting tasks [K] | NO: row 11 shows no conflict to remove. |
| Independent check: **Scenario-Wise Rec** (below) includes MMoE and PLE | 6 public sets, 3-10 seeds | | MMoE and PLE are *below* SharedBottom on MovieLens, Amazon, Douban, Mind [W] | confirms that the multi-gate structure alone is not a free gain at 1e6-1e8 rows. |

### 3.2 Multi-scenario / multi-domain gating: STAR, PEPNet, AdaSparse, M3oE

Independent replication exists and it is unfavourable: **Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark**, CIKM 2025, https://arxiv.org/abs/2412.17374 (6 public datasets of 0.8M-85M interactions, 3-5 scenarios each; 10 seeds per the main text, 3 per the appendix; +- about 0.001-0.002 per cell). AUC as extracted; identical in two fetches of two versions for every row except AdaSparse, which was extracted once [W]:

| model | MovieLens | KuaiRand | Ali-CCP | Amazon | Douban | Mind |
|---|---|---|---|---|---|---|
| SharedBottom | 0.8095 | 0.7793 | 0.6232 | 0.6792 | 0.7993 | 0.7509 |
| MMoE | 0.8086 | 0.7794 | 0.6242 | 0.6744 | 0.7978 | 0.7508 |
| PLE | 0.8091 | 0.7796 | 0.6250 | 0.6721 | 0.7977 | 0.7503 |
| STAR | 0.8096 | 0.7806 | 0.6253 | 0.6738 | 0.7957 | 0.7512 |
| EPNet | 0.8097 | 0.7801 | 0.6236 | 0.7101 | 0.7997 | 0.7418 |
| PPNet | 0.8063 | 0.7800 | 0.6144 | 0.6791 | 0.7994 | 0.7494 |
| AdaSparse | 0.8108 | 0.7816 | 0.6239 | 0.6888 | 0.7963 | 0.7497 |
| M3oE | 0.8116 | 0.7812 | 0.6249 | 0.7010 | 0.8036 | 0.7451 |

- No gating model wins everywhere. PPNet is below SharedBottom on 4 of 6, STAR on 2 of 6, M3oE on 1 of 6 (Mind, by 0.0058), EPNet on 1 of 6 (Mind, by 0.0091).
- Size against data: on the three datasets above 10M interactions the best model of the whole table beats SharedBottom by +0.0028 (KuaiRand), +0.0025 (Ali-CCP) and +0.0003 (Mind), i.e. 0-2 of their own +- values. Larger gains appear only on small sets: +0.004 on MovieLens and Douban (1.0M / 1.7M interactions), +0.03 on Amazon (0.8M).
- Original papers [K unless marked]: STAR (CIKM 2021, arXiv 2101.11427): per-domain weights multiplied onto shared weights, partitioned BatchNorm. **PEPNet** (KDD 2023, https://arxiv.org/abs/2302.01115): EPNet = sigmoid gate (scaled by 2) on the embedding layer driven by the domain id; PPNet = the same gate on every DNN layer driven by user / item / author ID embeddings with stop-gradient; multi-domain AND multi-task, Kuaishou, online "> 1%" [W abstract]. AdaSparse (CIKM 2022): domain-conditioned neuron pruning. M3oE (SIGIR 2024): searched mixing of domain and task experts.
- Transfer: NO.
  - The scenario variable must vary at inference. Here the production / exploration domain is constant on val and test (all random-order), and 99.2% of feeds contain several verticals inside one softmax (rows 6, 12).
  - PPNet's gate input is an ID embedding (a personal prior). This ranker has no useful ID table: the hashed item-ID table costs -0.005 under distillation [D reports/COVERAGE_MAP.md l.300].
  - STAR's partitioned BatchNorm has no counterpart in a LayerNorm model [D reports/SOLVER_POWER_1113.md 4.4].
  - The domain token (adopted) already is the minimal scenario conditioning.

### 3.3 Per-token and sparse MoE feed-forward blocks in large ranking transformers

| work | setting | experts and gate | isolated effect vs their noise | transfer |
|---|---|---|---|---|
| **RankMixer: Scaling Up Ranking Models in Industrial Recommenders**, arXiv 2507.15551 (ByteDance, 2025), https://arxiv.org/abs/2507.15551 | multi-task, 300+ features, trillions of records / day; they state 0.0001 AUC is significant | (a) per-token FFN: every feature-group token has its own FFN = hard routing by token identity, no learned gate; (b) sparse MoE inside each per-token FFN with a ReLU router, dense training / sparse inference | per-token FFN -> shared FFN **-0.31%** AUC; no token mixing -0.50%; sparse MoE "nearly no loss" against the dense 1B model at 1/8 activation, i.e. a cost device; online +0.3% active days at 1B parameters [W] | NO. (a) needs heterogeneous feature-group tokens. The one in-project test of such tokens (ftok: 8 group tokens + 1 block) is harmful: val -0.00114 (z -2.12) [D reports/R7_ARCH_HPARAM.md l.97]. The tokens of this ranker are M homogeneous candidates and history events. (b) is bounded by row 14. |
| **TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders**, arXiv 2602.06563 (ByteDance, 2026), https://arxiv.org/abs/2602.06563 | CTR + CVR, 0.3-17B records / day, 500M-15B parameters | per-token SwiGLU; sparse per-token MoE with a shared expert and gate scaling alpha about 1 / sparsity, sparse in training too | per-token SwiGLU -> shared -0.21%; sparse per-token MoE -> standard sparse MoE -0.10%; no shared expert -0.02%; 4B sparse = 4B dense (both +1.14%) [W, agrees with reports/nnx_research_recsys_arch.md l.92-98] | NO, same reasons. Their own statement of why per-token beats standard MoE: it gives the MoE "a routing prior" [D ibid. l.72]; our known priors (vertical, feed size, coldness) are explicit inputs and rows 4-7 show they carry no routing value. |
| **OneTrans**, arXiv 2510.26104 (ByteDance + NTU, 2025), https://arxiv.org/abs/2510.26104 | CTR + CVR, 29.1B impressions; offline > 0.1% "meaningful" | mixed parameterisation: non-sequence feature tokens get token-specific Q/K/V and FFN; **all sequence tokens share one set** | "shared parameters" row: -0.15% CTR AUC, -0.29% UAUC (91M -> 24M parameters, so capacity is confounded) [W] | NO. The paper itself shares parameters across sequence tokens; only feature tokens are specialised. This ranker already has separate modules per token class (history encoder, candidate blocks, user and domain tokens). |
| **SORT: A Systematically Optimized Ranking Transformer**, arXiv 2603.03988 (Alibaba, 2026), https://arxiv.org/abs/2603.03988 | 3 objectives, 9B impressions | FFN replaced by a DeepSeek-style top-k MoE, best sparsity 1/8 | Transformer -> + MoE: CTR AUC 0.7456 -> 0.7475 (+0.19 pt) with 18M -> 83M parameters at equal FLOPs; no seeds, no threshold [W] | NO. This is exactly A5's design (row 1), run at 36x our candidate rows with 4.6x the parameters. Here: null alone, negative after fine-tune, and extra dense capacity is negative under distillation (row 14). |
| **MTmixAtt**, arXiv 2510.15286 (Meituan, 2025), https://arxiv.org/abs/2510.15286 | multi-task, multi-scenario | learned token grouping, shared dense sigmoid-gated experts + scenario sparse experts | MoE ablation only in figures; dense sigmoid MoE beats ReLU sparse MoE; vs RankMixer +0.11 pt CTR GAUC (the text claims +0.39%, the table gives about +0.16%: internal inconsistency) [W] | NO: multi-scenario, unquantified. |
| **Zenith**, arXiv 2601.21285 (2026), https://arxiv.org/abs/2601.21285 | livestream ranking | "prime tokens", token fusion / boost; a search summary mentions a sparse MoE block | abstract only: CTR AUC +1.05% online model vs baseline [W] | not assessable; scale argument as above. |
| HyFormer, arXiv 2601.12681 (ByteDance, 2026), https://arxiv.org/abs/2601.12681 | baseline architecture of KDD Cup 2026 | query decoding + token mixing; no MoE reported [W summary] | not read in full | no MoE content. |
| Wukong (arXiv 2403.02545), HiFormer (arXiv 2311.05884) | 146B samples / Google Play | stacked FM blocks / heterogeneous attention | about 0.1% LogLoss per 4x compute [D reports/nnx_research_recsys_arch.md l.22, l.56] | already rejected there; nothing new. |

Common pattern [I]: in every paper of this family the MoE or per-token block buys *parameters per FLOP* under a serving-latency budget. This competition has no latency budget, so the honest comparison is the dense model of the same size, and that comparison is a tie in the papers and negative here.

### 3.4 MoE in a single-task model

| work | setting | experts and gate | isolated effect | transfer |
|---|---|---|---|---|
| **Enhancing CTR Prediction with De-correlated Expert Networks (D-MoE)**, arXiv 2505.17925 (2025), https://arxiv.org/abs/2505.17925 | **single-task** CTR, Avazu 40M / Criteo 45M rows; online GMV +1.19% vs multi-embedding MoE | 2-3 parallel experts (same or different interaction modules), each with its own embedding table; per-sample softmax gate (MLP 64); loss = BCE + cross-expert correlation penalty | plain shared-embedding MoE vs one expert: DCNv2 **-0.0002** (Avazu) / **-0.0005** (Criteo), others +0.0000..+0.006; multi-embedding -> + de-correlation loss: +0.00005..+0.0010 in 11 of 12 cells (one cell +0.0027); **no seeds, no gate ablation, no comparison with averaging independently trained models** [W] | NO. The mechanism is diversity between experts, i.e. an ensemble inside one model. This pool is 0.0002 below its infinite-member asymptote and 0 of 52 families adds at 2 SE [D reports/COVERAGE_MAP.md l.301]; the in-model form was null (row 3). The experts' separate ID embedding tables are the main diversity source and we have no ID tables. |
| **On the Embedding Collapse when Scaling up Recommendation Models** (multi-embedding), ICML 2024, https://arxiv.org/abs/2310.04400 | single-task CTR, public sets | several embedding sets, each with its own interaction module | gains appear when the embedding size is scaled up [W abstract; numbers not read] | NO: needs large ID embedding tables. |
| **(GG) MoE vs. MLP on Tabular Data**, arXiv 2502.03608 (2025), https://arxiv.org/abs/2502.03608 | 38 small tabular sets | Gumbel-softmax gated MLP experts | claims parity or better vs MLP with fewer parameters; not compared with TabM in what I read [W summary] | NO: no comparison with an ensemble; our TabM-form test is null (row 3). |
| **TabM**, ICLR 2025 [K; W summary] | tabular | k implicit members with shared weights, outputs averaged, no gate | strong and independently replicated on tabular benchmarks [K] | already tested, null (row 3). |
| **Towards an empirical understanding of MoE design choices**, arXiv 2402.13089 (2024), https://arxiv.org/abs/2402.13089 | language models | learned vs frozen random router | "comparable performance between a learned router and a frozen, randomly initialized router" [W abstract] | supports reading row 1 as "extra parameters, not routing": a learned gate is not where MoE gains come from. |
| Sparse Mixture-of-Experts are Domain Generalizable Learners (GMoE), ICLR 2023, arXiv 2206.04046 | vision, DomainBed | sparse MoE ViT | authors' numbers only; no independent replication found [W] | NO: vision, covariate shift across image domains, not input staleness. |

### 3.5 MoE in sequential recommenders

| work | setting | experts and gate | isolated effect | transfer |
|---|---|---|---|---|
| **Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation (FAME)**, WSDM 2025, https://arxiv.org/abs/2411.01457 | next-item, Amazon Beauty / Sports / Toys (168K-296K interactions), ML-20M subset (1.9M); full ranking | each attention head of the last layer predicts separately; MoE replaces the query matrix inside each head; a gate mixes the heads | NDCG@10 vs SASRec +17..+75% relative; **MoE ablation only in figures, no numbers; no seeds; baselines at their papers' hyperparameters** [W] | NO. Three failures of filter (c). Candidate-conditioned retrieval over the history (adopted) is already the limiting case of multi-facet user heads. |
| **Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation (HM4SR)**, WWW 2025, https://arxiv.org/abs/2501.14269 | 4 public sets, multi-modal items | MoE over item modalities, MoE with timestamp embeddings | ablation not read [W abstract] | NO: needs text / image modalities and timestamps; we have neither. |
| Replicability background: **A Systematic Review and Replicability Study of BERT4Rec**, RecSys 2022, https://arxiv.org/abs/2207.07483 | | | later "improvements" were measured against under-trained baselines [W] | the reason to distrust unreplicated +20..75% claims on 1e5-1e6 interactions. |

### 3.6 Routing by user activity or by cold / warm item

| work | setting | experts and gate | isolated effect | transfer |
|---|---|---|---|---|
| **POSO: Personalized Cold Start Modules**, arXiv 2108.04690 (Kuaishou, 2021), https://arxiv.org/abs/2108.04690 | multi-task (MMoE base), new users < 5% of samples, model has user-ID embeddings | gate = 2-layer MLP with sigmoid on a "personalisation code" only (bucketed impression count of the user), multiplying each MLP layer, attention K / V, or MMoE expert | new users +0.19..+0.44 pt GAUC on the Long-View task (by where the gate is applied), regular users +0.17..+0.34 for the combined form; a gate fed with *all* features is **negative** for new users (-0.27 / -0.50); online new-user watch time +7.75% [W] | NO. The stated mechanism is that a rare-group indicator is drowned among ID-dominated activations. Here the anonymous bucket has its own token, there is no user-ID table, and routers on history length / feed size / cold share have AUC equal to the permuted null (row 5). Never run in-network here [I: no row found in COVERAGE_MAP]. |
| **Cold & Warm Net**, arXiv 2309.15646 (2023), https://arxiv.org/abs/2309.15646 | retrieval stage, cold users | cold expert + warm expert + gate, distillation between them, bias net | abstract only [W] | NO: retrieval, user cold start. |
| **AdaF^2M^2**, DASFAA 2025, https://arxiv.org/abs/2501.15816 | Douyin retrieval + ranking | feature-mask multi-forward training + a "state-aware adapter" that re-weights features by user activity / item interaction count | ablation numbers not read; online +1.37% active days [W summary] | The closest published relative of our staleness lever, and its two halves are already measured: popularity-input dropout re-cools (xd30) val -0.00066 +- 0.00048, stale-val +0.00001, CLOSED; stale-view training (hz) fails at file level three times [D reports/CURRENT_STATUS.md log 08:58Z, 13:10Z]. NO. |
| **MAMEX: Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation**, CIKM 2025, https://arxiv.org/abs/2508.08042 | Amazon sets, item cold start | per-modality MoE + gate | offline only [W summary] | NO: needs modalities. |
| MESH (Pinterest, arXiv 2607.12392, 2026) | retrieval, fresh content tiers | content-tier separation | fresh-item repins +5.5% online [W summary] | NO: retrieval. |

Why "cold routing" keeps failing here, in one line [D reports/SOLVER_POWER_1113.md 4.1]: the covariate that predicts a cold click (the feed's cold share) also predicts a small payoff from boosting cold items, so the gate's posterior is below the local break-even in every quintile. No paper above addresses that structure.

### 3.7 Competitions with a feed / session ranking task: is MoE used, and where?

| competition | what the top write-ups use | MoE? | source |
|---|---|---|---|
| KDD Cup 2026 UniRec (pCVR, 35M samples, single task) | 1st QueryFormer: cross-attention query generation, H parallel tokenisation views (H 1 -> 8: +0.00075 AUC at 1.89x latency), distillation; 2nd UniDot: dot-product highway, two-path mutual learning; 9th FA-RankMixer: RankMixer blocks + shallow MLP stream; 10th Z Lab: dense-feature representation stack | no gated MoE in any abstract; token mixing comes from the baseline. Each architecture mechanism is worth 0.0002-0.0011 AUC | [D reports/research_1112/recsys.md 2.1]; [W] https://arxiv.org/abs/2609.16548, https://arxiv.org/abs/2607.15590 |
| RecSys Challenge 2024 (news feed) | 1st: set transformer with an impression token inside an ensemble | none reported | [D reports/deep_recsys2024.md section 3, l.352] |
| RecSys Challenge 2025 (universal behavioural profiles) | 2nd: GRU autoencoder; 4th: causal transformer + graph net + cross network + features, multi-task heads | none reported; 1st-place write-up not found | [W] https://arxiv.org/abs/2508.07748, https://arxiv.org/abs/2508.06970 |
| RecSys Challenge 2026 (conversational music) | retrieval fusion + LightGBM LambdaRank rerankers | none | [W search]; [D reports/TOP1_GAP_EXTERNAL_1114.md] |
| H&M 2022 | 4th: two-tower MMoE *retrieval* with gated experts for active vs inactive users, used as one feature of a LightGBM ranker | yes, in retrieval, as a feature | [D reports/deep_otto_hm.md l.266] |
| OTTO 2023, KDD Cup 2023 | co-visitation + GBDT rerankers | none | [D reports/deep_otto_hm.md] |
| Organiser's production ranker (engineering post, 2026) | iteration 1 "DCNv2 + MoE, BCE": -3% NDCG vs the tree model; the gains came later from embedding projections, a listwise loss and a history transformer with candidate self-attention | MoE tried first and dropped (not an isolated ablation) | [D reports/deep_ru_competitions.md l.59-72] |

Answer: winners of ranking competitions do not use gated MoE in the ranker. Where MMoE appears it is for several targets (multi-engagement tasks) or in retrieval.
Not read: any Kaggle write-up page directly; the RecSys Challenge 2025 winner; VK-LSVD 2025 and TAAC 2025 winners (no write-up found in earlier scans [D reports/TOP1_GAP_EXTERNAL_1114.md]).

## 4. Shortlist

**Empty.**

Three nearest misses, in the order of how close they came, with the reason each fails. None is recommended.

| rank | idea | filters (a) single-task / (b) cheap / (c) replicated or isolated | why it is not shortlisted | what would re-open it |
|---|---|---|---|---|
| N1 | Sparse MoE FFN under the *current* recipe (SORT / DeepSeek form; = A5 re-run as distilled d192 students) | a yes; b mostly: the block and its flags (`--moe 8 --moe_topk 2`) exist only in nn_x/attn/{model,train}.py; nn/, nn_x/w5/ and nn_x/abl/ train.py and nn/model.py contain no "moe" line [M, grep], so it needs a port and its tests; c yes (SORT ablation; A5 here) | A5 null and negative after fine-tune; the papers show sparse = dense at equal size; dense FFN x4 and d256 are negative under distillation (rows 1, 14). Expected result 0 +- 0.0003; P(pass the 8-seed rule) about 0.02-0.03 [I] | only as a record-keeping closure: 8 from-scratch students (seeds 951-958) vs kd05w192 on the same seeds, the standing rule (val >= 0, holdout >= 0, pooled z >= 2). Do not run it for expected value. |
| N2 | In-network context gate (POSO / PPNet / LHUC form): sigmoid gate (scale 2, zero-init) on FFN hidden units, gate input = stop-gradient of {anonymous flag, log history length, log M, feed cold share} only | a partly (POSO and PEPNet were shown on multi-task bases); b yes (about 30 lines, identity at init); c mixed: POSO's own ablation isolates it, the independent benchmark puts PPNet below shared-bottom on 4 of 6 sets | the published mechanism (rare group drowned by ID embeddings) is absent; the adopted attention gate is its near-equivalent and is worth +0.00065, mostly absorbed by the fine-tune; A5's router had the same inputs for 0; label-free routers on these covariates equal the permuted null (rows 1, 5, 13). P about 0.02 [I] | a label-free covariate with router AUC clearly above the permuted null on val AND the holdout (none exists today). |
| N3 | Cross-expert de-correlation loss between 2-3 parallel heads (D-MoE) | a yes; b yes; c no (no seeds, no gate ablation, no ensemble baseline) | it is in-model diversity; the pool is 0.0002 below its asymptote, TabM-form heads were 0.999-correlated and null, and induced diversity lowered strength in every family tried (rows 3 and the frontier law) | a pool whose asymptote, not its member count, moves: nothing in sight. |

## 5. Rejected popular ideas

| idea | reason |
|---|---|
| MMoE / PLE / AdaTT task gates | multi-task-only; main vs auxiliary gradients are orthogonal here (row 11); MMoE and PLE are below shared-bottom on 4 of 6 public sets in the independent benchmark [W]. |
| HoME (expert normalisation + Swish, hierarchy mask, feature-gate, self-gate) | every component is defined across tasks; gains +0.02..+0.23 pt AUC at 50B logs / day with 8-20+ tasks [W]. |
| SMES, HA-MoE | multi-task; numbers not available. |
| STAR, AdaSparse, M3oE, HAMUR | need a scenario that varies at inference; ours is constant (domain) or mixed inside each feed (vertical); benchmark gains over shared-bottom at most +0.003 AUC on the sets above 10M interactions, and no model wins on all six [W]. |
| PEPNet (EPNet / PPNet) | multi-domain and multi-task; PPNet needs ID embeddings as the gate input; PPNet below shared-bottom on 4 of 6 sets [W]. |
| RankMixer per-token FFN, TokenMixer-Large per-token SwiGLU, OneTrans mixed parameterisation | hard routing by feature-token identity; needs heterogeneous feature tokens (ftok is harmful here); shown at 3e10..1e12+ rows; OneTrans itself shares parameters across sequence tokens. |
| Sparse MoE as capacity (RankMixer DTSI, TokenMixer-Large sparse per-token MoE, SORT, Zenith) | a parameters-per-FLOP device; equals the dense model of the same size by the authors' tables; dense scale-ups are negative here. |
| MoE upcycling in the cooldown | closed on stored arms (row 2). |
| Multi-embedding MoE (embedding collapse, ICML 2024) | needs large ID embedding tables; the hashed item-ID table costs -0.005 here. |
| TabM / implicit ensembles / multi-view heads (QueryFormer H) | not gated MoE; tested (row 3); students of an 11-member teacher already carry the ensemble function. |
| Sequential-recommender MoE (FAME, HM4SR) | 1e5-1e6 interactions, no numeric MoE ablation, no seeds, untuned baselines; HM4SR needs modalities and timestamps. |
| POSO / Cold & Warm Net / MAMEX / MESH (activity or cold-warm experts) | user cold start, retrieval stage or modalities; mechanism absent here; cold gates closed with the value-weighted AUC argument (row 4). |
| AdaF^2M^2 (feature mask + state-aware adapter) | both halves measured here in the staleness lane: input dropout buys no stale robustness, stale-view training fails at file level. |
| Learned gate over ensemble pools / feed-conditional weights | rows 5, 8, 9, 10: nested negative, optimism 0.0004-0.0010, public -0.00015 for the simplest rule. |
| GMoE for distribution shift | vision only, unreplicated, different kind of shift. |
| AdaMoE (concept drift, arXiv 2204.05101) [W title only] | weights experts by recent streaming performance; needs labels arriving at test time. |

## 6. What I did not check

- No experiment and no model code run: every project number is copied from the cited file; I did not recompute any. The only thing I ran on the project is a grep for "moe" in six source files.
- Web numbers come from an automated page reader, not from my own reading of PDFs. Cross-checked: RankMixer -0.31% (two extractions), TokenMixer-Large -0.21% / -0.10% (extraction + the project's earlier independent note), Scenario-Wise table (two versions, identical). Single extraction only: HoME ablation deltas, SORT, OneTrans, D-MoE table, POSO tables, FAME.
- Abstract only: Zenith, HA-MoE, SMES, QueryFormer, FA-RankMixer, HM4SR, Cold & Warm Net, multi-embedding, GG MoE, AdaF^2M^2 (their ablation numbers are not in this report).
- Not opened at all: the original STAR, AdaSparse, M3oE, HAMUR, MMoE, PLE, AdaTT, TabM papers (cited from memory [K] and from the benchmark); MSN (arXiv 2602.07526); any Kaggle discussion page.
- AUC effects were not converted to NDCG; no conversion is known to me. The comparison with our noise (paired val SE 0.0003-0.0005) is therefore qualitative.
- I did not verify that no in-network POSO-type gate was ever run here; I found no such row in COVERAGE_MAP (336 lines read in full).
- Graph networks are outside this scout's brief.

## 7. Sources

Web (read this session unless marked abstract-only above):
- RankMixer: Scaling Up Ranking Models in Industrial Recommenders, https://arxiv.org/abs/2507.15551
- TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders, https://arxiv.org/abs/2602.06563
- OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender, https://arxiv.org/abs/2510.26104
- SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders, https://arxiv.org/abs/2603.03988
- MTmixAtt: Integrating Mixture-of-Experts with Multi-Mix Attention for Large-Scale Recommendation, https://arxiv.org/abs/2510.15286
- Zenith: Scaling up Ranking Models for Billion-scale Livestreaming Recommendation, https://arxiv.org/abs/2601.21285
- HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction, https://arxiv.org/abs/2601.12681
- HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou, https://arxiv.org/abs/2408.05430
- SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity, https://arxiv.org/abs/2602.09386
- Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study (HA-MoE), https://arxiv.org/abs/2607.27577
- Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark, https://arxiv.org/abs/2412.17374
- PEPNet: Parameter and Embedding Personalized Network, https://arxiv.org/abs/2302.01115
- Enhancing CTR Prediction with De-correlated Expert Networks, https://arxiv.org/abs/2505.17925
- On the Embedding Collapse when Scaling up Recommendation Models, https://arxiv.org/abs/2310.04400
- (GG) MoE vs. MLP on Tabular Data, https://arxiv.org/abs/2502.03608
- Towards an empirical understanding of MoE design choices, https://arxiv.org/abs/2402.13089
- Sparse Mixture-of-Experts are Domain Generalizable Learners, https://arxiv.org/abs/2206.04046
- Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation, https://arxiv.org/abs/2411.01457
- Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation, https://arxiv.org/abs/2501.14269
- A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation, https://arxiv.org/abs/2207.07483
- POSO: Personalized Cold Start Modules for Large-scale Recommender Systems, https://arxiv.org/abs/2108.04690
- Cold & Warm Net: Addressing Cold-Start Users in Recommender Systems, https://arxiv.org/abs/2309.15646
- AdaF^2M^2: Comprehensive Learning and Responsive Leveraging Features in Recommendation System, https://arxiv.org/abs/2501.15816
- Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation, https://arxiv.org/abs/2508.08042
- QueryFormer (KDD Cup 2026 industrial track, 1st), https://arxiv.org/abs/2609.16548
- FA-RankMixer (KDD Cup 2026, 9th), https://arxiv.org/abs/2607.15590
- Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025, https://arxiv.org/abs/2508.06970
- Encode Me If You Can: Learning Universal User Representations via Event Sequence Autoencoding, https://arxiv.org/abs/2508.07748

Project files used: reports/CURRENT_STATUS.md, reports/COVERAGE_MAP.md, reports/INDEX.md, reports/SOLVER_POWER_1113.md (section 4, 5.1), reports/nnx_VERDICT.md, reports/nnx_exp_attn.md, reports/nnx_research_recsys_arch.md, reports/R7_ARCH_HPARAM.md, reports/MIX_1115/gateB.md, reports/RECHECK_1114/kd_reprice.md, reports/RECHECK_1114/pool_composition.md, reports/research_1112/recsys.md, reports/deep_ru_competitions.md, reports/deep_otto_hm.md, reports/deep_recsys2024.md, reports/TOP1_GAP_EXTERNAL_1114.md, reports/PRETRAIN_PLAN_1113.md.
