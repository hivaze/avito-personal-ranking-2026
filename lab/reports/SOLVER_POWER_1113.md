# SOLVER_POWER_1113: more powerful solvers on this data (pretrained models, next-item training, mixture of experts, scale)

Written 2026-11-13 ~23:40 UTC by the synthesis step of the solver-power study. CPU only. The project was read and not modified; this report is the one file written to it. Nothing was trained or submitted.

**Inputs**
- Four lens analyses, each with its own CPU probes:
  - pretrained and foundation models;
  - next-item and self-supervised training at scale;
  - mixture of experts;
  - scale and generative recommenders.
- 17 proposals from those lenses. The 12 with the highest priors each received two independent reviews. Most reviews re-ran numbers on CPU from stored predictions, training logs or the flat data.
- The project record: GLOBAL_RECAP_1113 (read in full), COVERAGE_MAP, PRETRAIN_PLAN_1113, the research reports (nnx_research_*, deep_seq_pretrain_cf, deep_neural_rankers), the LLM track (LLM_FEASIBILITY, llm_research, llm_build, llm_large, nn_x/llm_manual/RESULT.txt), R7_ARCH_HPARAM, r7_research, DATA_PIPELINE_1113 §3, docs/02_SOLUTION and docs/03_EXPERIMENTS.

The harness blocked the lenses' report files. Their full texts and every review are copied to the scratchpad (Appendix A). "[lens]" marks a number first measured by a lens today, "[review]" one first measured by a reviewer today. Every other number cites a project file.

**Conventions**

| term | meaning |
|---|---|
| P | honest probability that an item raises the final **private** score by **>= +0.002** over its comparator (G3 unless stated). GLOBAL_RECAP_1113 uses +0.001; that threshold is quoted separately where it matters. |
| generational | val >= +0.002 at within-feed corr < 0.99 (GLOBAL_RECAP glossary) |
| frontier law | equal-mix gain = 0.0186(1 − ρ) − 0.378·\|gap\| (docs/03_EXPERIMENTS §13). An equal-strength family needs margin m >= ~0.10 for +0.002; the largest full-strength m on record is 0.019 (GLOBAL_RECAP §3.1). |
| m / g | m = ρ(base, base) − ρ(arm, base) across seeds; g = member val deficit |
| GPU-h | GPU hours. A 10-epoch d128 member logs 117.6-156.7 min (transformer) or 179.7-208.5 min (GRU) wall-clock on shared GPUs (docs/02_SOLUTION §12). |
| CD harness | re-running only the cooldown from stored plateau snapshots (@77500) |

**A correction to the brief.** The figure "82M history events in train" does not appear in the project files. The labelled train feeds hold 448.7M history events, and the de-duplicated pre-training corpus has 463.6M tokens, 98.4% of them the same events (PRETRAIN_PLAN §2).

---

## 1. Executive summary

1. **The binding constraint is information, not compute or model class.** A 10-epoch member needs about 1.3e16 FLOPs, a few seconds of one GPU at peak against 2-3 hours of wall-clock (scale lens, from LLM_FEASIBILITY §3). Every size, pre-training, expert and generative probe on record learns the same function from the same inputs. 66% of the lost NDCG sits in feeds where the click loses to its beaters on every cue we have (GLOBAL_RECAP §1).
2. **No pretrained model attaches beyond what is already used.** The only models that share our item and user space are Avito's own towers, and they already build every token; there is no text, image, timestamp or shared ID vocabulary. The LLM track lost 0.0038-0.0053 val at matched 150K feeds, flat over a 16x size range, with random init equal to pretrained. The deficit opens after the first pass, so more data is not expected to close it [review]. A text rendering carries strictly less information, and hand ranking by an LLM scored −0.048.
3. **Next-item training reuses the ranker's own inputs.** Inside a usable window the histories hold 44-56M distinct next-feed-open targets, 8-10x the labelled positives rather than 26x [review]. pre2, trained on exactly this target, is worth +0.00128 val, about 0.006 real feeds per target; +0.002 needs 0.03-0.08 [review]. Across 19-28 stored encoder variants, zero-shot quality does not predict value on G3 [review].
4. **MoE has nothing to route on.** The +0.025 cold oracle is an echo of the label. Label-free routers on 31 covariates reach AUC 0.494-0.511 against a permuted null of 0.496-0.503, and a val-fitted router loses −0.0025 (z −6.0) on the holdout [lens]. Sparse experts learn vertical routing but add no strength, and only attention-gate-level diversity; 32 cooldown-only arms sit 10-60x below the diversity bar [review].
5. **Width does not raise the plateau.** d192 equals d128 at matched plateau steps (pooled holdout +0.00005 ± 0.00009) and loses only late in the cooldown, by memorisation [review]. The one untested mechanism, Muon's per-step update growing as √width, leaves a capacity verdict unlikely (P(CAP) ~0.04-0.08), but it is cheap to test.
6. **Review outcome.** Of 17 proposals, 12 were reviewed twice: 1 survived (BIGPOOL, a build that spends nothing unless a width gate passes), 10 split (refuted as specified, each with a repaired form), and 1 was refuted by both reviewers (the upcycled-MoE cooldown). Of the 5 unreviewed proposals, one (MoE routing) was executed on CPU and closed.
7. **Recommendation.** Pre-declare four cheap gates, about 30-45 GPU-h in total: WIDTH-GATE (d192 re-cools with a per-matrix μP arm), ENCODER-DOSE (in-ranker dose read, PRE2-10EP and a frozen hidden-state probe), PSEUDO-DIRECT (a 10-epoch pseudo-feed co-training pair) and SSL-WIDTH (next-item aux × width, behind a CPU trigger). Spend beyond that only on a GO: at most about 300-400 GPU-h, under 10% of the idle budget.
8. **Combined P(>= +0.002 private) ≈ 0.01** (P(>= +0.001) ≈ 0.03; expected private gain ≈ +0.00005). The leader's +0.0048 is out of reach of any model-side bet here. The zero-GPU pair decision and a merge (GLOBAL_RECAP §6) remain the larger levers.
9. **Close at 0 GPU-h:**
   - feed- and pool-level MoE routing, upcycled experts, per-vertical experts and event-type heads;
   - ALL-LABELS-CD and FT-PSEUDO;
   - the text LLM and TFMs in every role;
   - the 20-30 GPU-h LLM-SLOPE gate of GLOBAL_RECAP §4.2, replaced by nothing or by a 4-6 GPU-h one-pass read.
10. **What would change the verdict** is new information (event timestamps, text, sellers, prices) or a partner model, not parameters or GPU-hours.

---

## 2. Pretrained and foundation models

### 2.1 How a pretrained model can attach, and which route exists here

| route by which pre-training beats a supervised ranker | evidence in the literature | here | source |
|---|---|---|---|
| shared tokens: text, images or a shared ID vocabulary | LLM gains in recommendation come from text representations (RecBench, arXiv 2503.05493) | none: anonymised IDs, int8 vectors, no text or images | llm_research §0; PRETRAIN_PLAN §3.4 |
| a shared embedding space | a public model trained in the same space | only Avito's own towers: the item vectors (in-house two-tower, P 0.94) and user_emb (InfoNCE, ~10B samples, reads search queries). Both are already inputs. | EMBEDDINGS_1113; PRETRAIN_PLAN §1.1 |
| unlabelled data much larger than labelled | GPSD, LazFormer, PinFM, ARGUS, LUM ran at 10-1000:1 | about 1:1: 463.6M corpus tokens against 448.7M history events in the labelled feeds | PRETRAIN_PLAN §2, §3.3 |
| an overfitting ID table to transfer and freeze | GPSD: sparse transfer + freeze 0.6838 against no transfer 0.6375; dense transfer alone 0.6376 (arXiv 2506.03699) | the ranker has no ID table; hashed IDs −0.0022; candidate ID coverage 74% (train) → 40% (private) | PRETRAIN_PLAN §1.3, §2, §3.1 |
| inputs the downstream model cannot see | SUM, Zalando LTE, PinFM's 2-year histories | H <= 256, all visible to the ranker; only user_emb sees other surfaces, and it is an input | PRETRAIN_PLAN §3.3 |
| label scarcity or overfitting | Layer 6, RecSys Challenge 2022 | late train loss 2.9448 (3 blocks) vs 2.9487 (base); val rises to the last step | PRETRAIN_PLAN §1.3 |

### 2.2 Model classes checked

| class | examples | what it would need | status here | P |
|---|---|---|---|---|
| Avito towers | item tower, user_emb | — | attached: the backbone of every member (docs/02_SOLUTION §4.1, §5) | — |
| our trajectory encoder | pre2: d384, 4 layers, 8.0M dense parameters plus an ID table | — | enters as 6+3 scalar columns worth +0.00128 val; its hidden states have never entered the ranker (PRETRAIN_PLAN §1.1) | §3 |
| public sequential and generative recsys checkpoints | SASRec/BERT4Rec, HSTU (2402.17152), TIGER (2305.05065), OneRec/OpenOneRec (2506.13695, 2512.24762) | a shared ID vocabulary or item text | weights unusable. The architecture is matched already: GRU ≈ transformer; JOINT early fusion −0.0007 (COVERAGE_MAP l.216). RQ semantic-ID codes over our vectors gave +0.00038 (z 0.8) (PRETRAIN_PLAN §1.3). | 0 |
| LLMs over soft tokens | Qwen3-0.6B, Qwen3.5-4B / 9B | — | −0.0038 to −0.0053 at matched data (§2.3) | ~0.001 |
| LLMs over text | 7-32B LoRA fine-tune | text | strictly less information than the soft tokens (§2.5) | ~0.001 |
| embedding models | Qwen3-Embedding-4B, MiniLM (local) | text or images | nothing to embed | 0 |
| tabular foundation models | TabPFN v2/2.5/3, TabICL / TabICLv2 (2602.11139), TabDPT (2410.18164) | pointwise rows, contexts of up to ~1M rows; no published ranking objective [lens] | as a stacker, capped by the measured second-level headroom (~+0.0002 val; GBDT[G3 + 92 inputs] − GBDT[G3] +0.00005 ± 0.00021, DATA_PIPELINE §4); as a member GBDT-level (frontier ≈ −0.003); per-feed in-context over the user's history is TabR retrieval, already in the ranker | < 0.001 |
| time-series foundation models | Chronos, TimesFM, Moirai | series | history events have no timestamps; item series have at most 12 mostly-zero points (PRETRAIN_PLAN §3.4) | 0 |
| graph pre-training | LightGCN, PinSage, TwHIN | — | the same information as ALS and co-visitation (residual 0 to −0.0005) | < 0.001 |
| MoE grown from checkpoints | sparse upcycling (2212.05055), Branch-Train-MiX (2403.07816) | a capacity-bound regime | §4.3 | ≤ 0.001 |

### 2.3 The LLM track: what was built, what it scored, why it stopped

| stage | what | result | source |
|---|---|---|---|
| feasibility (11-12) | text, embedding and number-as-text routes | dead: about 287 tokens per item, about 76K per feed. A matched 150K-feed test was pre-declared, with gates A1 (beat the ranker) and A2 (a useful member). | LLM_FEASIBILITY; llm_research |
| build (11-12) | `nn_x/llm/` soft-token ranker. Layout L1 is [sink, user, history, candidates], with candidates attending only to the user, the history and themselves, so the model is permutation-equivariant. LoRA r32; fp32 score and K9 heads; the ranker's own 104 inputs and listwise loss; amendments S1, E1, I3, V0. | CPU smoke: equivariance 8e-6, leakage exactly 0 | llm_build |
| matched test (11-12/13) | the same 148.6K-feed subsets for every arm; holdout selection | **L − B val:** 0.6B (4 ep, judged) −0.00378 ± 0.00077; 4B −0.00480 ± 0.00081; 9B −0.00532 ± 0.00080. **Holdout** −0.0040 to −0.0048. A1 and A2 fail at every size. 9B − 0.6B −0.00081 ± 0.00047. L − R (random init) within about 1 SE. ρ(L, B) 0.932-0.943 against ρ(B23, B24) 0.946. 46.74 lane-hours. An independent review reproduced every number. | llm_large R3-R8 |
| manual pilot (11-13) | Opus 5.5 with tools ranks 100 rendered val feeds | 0.3957 vs G3 0.4433 (paired −0.048 ± 0.023), tied with one-column heuristics; blending hurts at every weight; scoring the test set would take about 112M tokens | nn_x/llm_manual/RESULT.txt |

**Why it stopped:**
- the pre-declared STOP fired at every size;
- size was flat over 0.44B-6.92B body parameters;
- language pretraining added nothing measurable;
- LLM arms are near copies of each other (corr 0.968-0.994).

The record names what it left untested: per-size retuning, more than 150K feeds, and more than 2 epochs at 4B/9B (llm_large R8.3 finding 4).

### 2.4 Three new readings of the LLM record (this study)

**(a) The deficit is a multi-pass deficit, not a data-slope deficit** [review; skeptic_llmslope/passes.log, fulldata_passes.log].
- After about one pass, every 150K arm is level on the holdout:

  | arm | holdout at ~1 epoch |
  |---|---|
  | ranker B (s23 / s24, 1.05 ep) | 0.2990 / 0.2968 |
  | Qwen3-0.6B | 0.2981 / 0.2976 |
  | random-init twin | 0.2981 |
  | Qwen3.5-4B | 0.2972 / 0.2972 |
  | Qwen3.5-9B | 0.2979 / 0.2969 |

- The gap opens afterwards. The ranker gains about +0.004 over epochs 1.6-3.7 at constant lr; the LLM arms gain +0.0014..+0.0025 by 2-4 epochs.
- At full data the ranker's multi-pass gain does not shrink. w5_e10_base s951 / s952 holdout reads 0.3126 / 0.3118 at 1 epoch, 0.3173 / 0.3179 at 7.6 epochs, and 0.3205 / 0.3199 after the cooldown.
- So more unique data should leave L − B near −0.003..−0.006. The untested "full-data slope" cell points at the wrong axis.

**(b) The diversity route is closed** [review; skeptic_softslope/struct_corr.log].
- With seed noise removed, the structural LLM-to-ranker correlation is 0.968-0.974 at every size, and 0.974 at both 2 and 4 epochs.
- The full-data pop-free and pop1 families sit at 0.983 / 0.986.
- By the frontier law, whole LLM families at g = 0 are worth +0.0005-0.0006 val. +0.002 needs ≤ ~0.89.

**(c) The LLM arms' cold-feed advantage is a popularity-light tilt** [lens; pretrained/llm_axis.log, llm_blend_fine.log].
- The deviation correlates −0.24..−0.27 with z(log popularity).
- Only 1.2% of its variance is shared with the pop-free / pop1 deviations, an axis C3 already carries.
- It has no gradient in label-free coldness: Q4 − B is −0.0025 / −0.0069 / −0.0065 / −0.0080 across luo0-share bins.
- In-sample, G3 + w·z(LLM) peaks at +0.00014 ± 0.00011 (w 0.01) and is ≤ 0 nested.

### 2.5 A fine-tuned 7-32B LLM over text renderings

- **Information.** The pilot rendering carries about 10 cues per candidate plus the history listing. It drops the raw vectors, cos_u, co-visitation, the ctx block, pre2 and the per-type LUO counts. Every column one could add is already a ranker input, and vertical "world knowledge" is closed (ENRICH 0/9, COVERAGE_MAP l.274). So the text holds a subset of what the soft-token arms saw, and they lost.
- **Cost** [lens]. About 4-6K tokens per feed against 265 soft tokens (15-23x). That is 300-600 GPU-h per epoch at 7B and 1,400-2,800 at 32B. 5,000 GPU-h buys 8-16 epochs at 7B, so compute is not what binds.
- **Public evidence.**
  - RecBench (2503.05493): ID-based fine-tuned LLMs trail ID-based CTR models (best P5-Llama-7B AUC 0.6576 vs GDCN 0.6825) and trail SASRec on NDCG@10 (0.026-0.029 vs 0.067-0.070). Their gains come from text, at about 1,000x slower inference.
  - Xu et al. (2408.14238): CE-trained SASRec beats fine-tuned LLM recommenders.
  - Hou et al. (2305.08845) and *Lost in Sequence* (2502.13909): LLM rankers are insensitive to history order.
  - Kang et al. (2305.06474): fine-tuned LLMs are data-efficient only with item titles.
- **P ≈ 0.001.** Never run it ahead of a soft-token pass, and the soft-token route is itself closed (§2.4).

### 2.6 Verdict

"Building on top of a pretrained model" is already what the solver does: Avito's item and user towers supply every token. No public model shares that space. The one large-model family that can read our tokens, LLM trunks, learns the same function as the 1M-parameter ranker, and learns less per pass.

The one attachment never tried is our own encoder's hidden states. It is tested in ENCODER-DOSE (§6.3, item 2) as a 1-3 GPU-h probe on a single frozen checkpoint.

---

## 3. Next-item and self-supervised training at scale

### 3.1 The pre2 baseline, exactly

| item | value | source |
|---|---|---|
| model | causal pre-LN transformer with RoPE, d384, 4 layers, 8.0M dense parameters, plus an ID table of 40.3M rows (dim 64/128, ID-dropout 0.3). v1 was d256, 3 layers, 2.8M. | pretrain_report §2-3; PRETRAIN_PLAN §1.1 |
| objective | next-item prediction, not contrastive or masked. Head `fe` targets the next new feed-originated open (type 1, source 1); head `new` targets the next new item (weight 0.5); from v2, dense next-4 targets. | same |
| loss | sampled softmax over 6,144 in-batch fe targets, 2,048 in-batch new targets and 8,192 uniform items, with logQ correction | nextitem lens §1 |
| corpus and cost | 2.59M sequences, 463.6M tokens; 2 epochs take 21.5-43 min on one GPU | PRETRAIN_PLAN §1.1; docs/02_SOLUTION §12 |
| zero-shot val NDCG | v1 0.3537 → v2 0.3578 → pre_final 0.3590 (G3 0.40032; maxsim32 0.3478) | PRETRAIN_PLAN §1.2 |
| worth inside the ranker | +0.00128 ± 0.00061 val / +0.00054 ± 0.00034 holdout at 2 ep; +0.0009 val at 10 ep (M<50 +0.0020, M>=50 +0.0001) | COVERAGE_MAP l.190; PRETRAIN_PLAN §1.2 |
| distance to G3 | within-feed corr 0.684. Its oracle headroom (+0.041) is within 0.009 of a random scorer's (+0.032). On feeds where G3 buries the first positive, pre2 ranks it higher in 47.6% and lower in 49.9%. | nextitem/diag1.log, diag1b.log [lens] |
| buried-cell pair AUC | 0.31-0.33, against anti-popularity controls 0.570 / 0.632: every label-free score sides with the popular beaters | PRETRAIN_PLAN §1.3, §7.3 |
| closures | F8 (scale-up) and E1 closed for good by the +0.0015 trigger; E2 (in-ranker aux) and pseudo-feed pre-training closed by argument and never run | PRETRAIN_PLAN §5 |

### 3.2 Volume: how many next-item targets exist, and what they are worth

| quantity | value | source |
|---|---|---|
| fe&new events in the train corpus, all positions | 141.4M = 26.0x the 5,446,872 labelled positives | nextitem/diag2.log [lens] |
| the same in the last 64 positions | 51.1M (9.4x) | skeptic_genrec/last64.py [review] |
| usable window: target at t <= 63 with >= 1 event older than t+16 / t+40 / t+64 (per user sequence) | 49.7M / 47.3M / 44.5M (9.1x / 8.7x / 8.2x) | skeptic_p1_pseudofeed/k1_pool_t63.log [review] |
| same rule over train feeds, distinct (user, item), g >= 16 / g >= 64 | 54.3-56.1M / 50.0-50.9M (~10x) | skeptic_p1/k1.log; skeptic_p4/k1_pseudo_count.log [review] |
| distinct targets visited by a 10-epoch member (8 plateau epochs) at pseudo:real 1:1 / 4:1 | 16.0-17.5M / ~42M | skeptic_p1/k1.log [review] |
| labelled positives that reappear in the user's next snapshot | 92.1% | CODES_DECODING_1113 l.68 |

**Exchange-rate arithmetic.**
- At the measured +0.0037 val per doubling of labelled feeds, +0.002 val needs ×1.45 the data, i.e. about +1.42M real-feed equivalents (nextitem lens §3.3).
- **Rate needed:** about 0.025-0.03 real feeds per pseudo-target if each distinct target is used once; about 0.08 at 1:1 over 8 plateau epochs; about 0.034 at 4:1 [review].
- **Rate implied by pre2:** its worth (+0.00128 val ≈ 0.35 doublings ≈ 0.85M feed equivalents) spread over the 141M targets it was trained on gives about 0.006 [review].
- **Stand-alone bound:** pre2 zero-shot (0.359) is below a ranker trained on 150K labelled feeds (0.380), i.e. under 0.001 per target [lens].
- **Caveat:** the +0.0037 slope is a two-point average over 4.4 doublings (150K vs full data). Part of it is the cooldown the 150K baseline never got (skeptic_softslope/member_level.log [review]). The local slope at 3.15M feeds is unmeasured.

So co-training would have to extract 5-13x what the encoder channel already extracts from the same targets.

### 3.3 Pseudo-feeds as a training format: what the CPU probes show

| property | finding | source |
|---|---|---|
| difficulty | Without a gap, history ends are near-duplicates of the target (maxsim NDCG 0.438 vs 0.349 on real feeds). A gap of 16-64 events brackets real difficulty: similarity lift +0.103 at g 16 and +0.064 at g 64 vs +0.061 real; category match 0.819 / 0.755 vs 0.827. | nextitem/diag3.log [lens] |
| content-only shortcut | Within-feed AUC 0.593 on pseudo-feeds vs 0.555 on real. Adding 16 in-batch pseudo-positives with logQ leaves it at 0.590: logQ corrects sampled negatives, not impression negatives. | skeptic_p1/k2.log [review] |
| request-time set leak | The L2 candidate set points at the target: set-centroid AUC 0.608 vs 0.582 on real feeds. Foreign in-batch items are trivially separable (AUC 0.963), so they are easy negatives. | skeptic_p1_pseudofeed/k2_set_shortcut.log [review] |
| required leak handling | user_emb is request-time and contains the target event, so it must be zeroed. f1-f5 and the 34 pre-computed columns do not exist for history items. Holdout users can never be a source. Later same-user snapshots hold 54.4% of train positives. | PRETRAIN_PLAN §2 pitfall 1; nextitem lens §5 |
| staleness | Pseudo-rows carry no freshness or popularity inputs, so they cannot teach the private-specific shift (popularity staleness, 11.2 d) | skeptic P5 reviews; DATA_PIPELINE §3.1 |

### 3.4 Forms, and what more scale would change

| form | status | what more GPU changes | P |
|---|---|---|---|
| next-item score as a member family | Closed. At within-feed corr 0.684, the frontier needs zero-shot >= 0.385 to break even and >= 0.390 for +0.002; pre2 is 0.359. The law reproduces the measured equal mix (predicted −0.0097, observed −0.0109). | A larger encoder is not expected to gain +0.03 zero-shot | < 0.001 |
| encoder as init of the ranker's history encoder | Closed: dense transfer ≈ 0 (GPSD 0.6376 vs 0.6375; LazFormer 0.7701 vs 0.7699); random-init LLM = pretrained (PRETRAIN_PLAN §5) | nothing | < 0.001 |
| larger encoder feeding refreshed scalars (PRE2-XL / F8) | Closed by trigger. Across 19 distinct stored variants, the nested residual on G3 has slope +0.00003 per +0.001 zero-shot (r +0.28) [review, skeptic_p3/dose.log]. Across 28 variant-keys the blend gain has corr −0.04 with zero-shot [review, scale_transfer.log]. | The data-constrained optimum is about 90M parameters × 4 epochs, 3-5 GPU-h per copy (Muennighoff et al., 2305.16264); beyond about 4 epochs, returns on repeated data vanish | ~0.001 |
| frozen hidden states as an extra K/V branch (PinFM / LazFormer) | never run; hidden states have never entered the ranker. PinFM frozen +0.10% vs fine-tuned +3.76% (2507.12704). | — | ~0.002-0.003 |
| next-item aux head on the ranker's own history encoder (JOINT-SSL / E2) | closed by argument, never run. The 1M trunk does not overfit; K9's +0.0022 comes from an external teacher. | trivially cheap | ~0.002 |
| pseudo-feed co-training (the full ranker trained on history-built lists) | closed only as pre-train-then-finetune, never run as co-training | the only form that turns GPU-hours into training examples | ~0.003 |
| contrastive or masked user encoders (CoLES, CL4SRec, DuoRec, BERT4Rec-style) | user_emb is already a contrastive user tower from a ~20x larger corpus with search inputs; SASRec with CE beats BERT4Rec (2309.07602) | nothing | < 0.001 |
| val/test-period pseudo-feeds in the fine-tune (FT-PSEUDO) | The information is already in G3: LUO counts span all snapshots, and pre2 was trained on the val/test snapshots with the same target. Transductive increment <= +0.0004 (PRETRAIN_PLAN §1.3). | nothing | ~0.001 |

### 3.5 Intermediate metrics that are on the ranker's path

1. **In-ranker dose.** How much ranker value is lost when the pre2 scalars are swapped for a weaker (ID-free) version. This is the only direct link between encoder quality and ranker value; the existing variant spread says the link is flat (§3.4).
2. **Nested residual on G3, on val and on the holdout** (through a windowed copy), >= +0.0005 at 2 SE.
3. **Buried-cell pair AUC against the anti-popularity controls** 0.570 / 0.632, not against pre2's 0.31-0.33.
4. **For aux heads:** recall@k of hidden events against a windowed w3r encoder on post-window holdout feeds over a fixed universe (the E2 reopen trigger, PRETRAIN_PLAN §5).
5. **For co-training:** effective real-feed equivalents, (2^(Δval/0.0037) − 1) × 3.12M, per visited pseudo-target. Also a substitution check: does the gain survive when the pre2 channel is removed?

Not on the path:
- zero-shot NDCG of a stand-alone encoder;
- "within-feed corr to G3 <= 0.75" (every existing score passes it, at 0.61-0.68).

### 3.6 Verdict

Next-item supervision is dense only as a token count. As information it is the ranker's own input, recorded positive-only and without the slate. pre2 already transports it at about 0.006 feed-equivalents per target.

Two cheap reads remain, and both are in §6.3:
- whether co-training extracts more than an input channel does (PSEUDO-DIRECT);
- whether a frozen representation adds anything beyond 6 scalars (ENCODER-DOSE).

No form justifies 500-1,000 GPU-h. Encoders are data-constrained at a few GPU-h per copy, and GLOBAL_RECAP §7.4 keeps this on the do-not list.

---

## 4. Mixture of experts

### 4.1 Why the cold oracle earns +0.025 and a learned router earns 0

The study on record is sol_errors §2.3, cited at COVERAGE_MAP l.120. Re-run on G3 today [lens; moe/cold_mech.log]:
- **The oracle.** Boost every cold candidate by 1.0 z only in feeds whose positives are all cold (17.4% of feeds): +0.0257.
- **The best label-free gate.** LightGBM on 10-31 covariates reaches AUC 0.654-0.671 and earns −0.00005 ± 0.00004 nested.

**Mechanism.** The oracle knows one bit about the label: "the clicked item is cold". No feed-level covariate carries it. The covariate that predicts a cold click best, the feed's cold share, also predicts that boosting cold items will pay little:

| cold-share quintile | P(all-cold) | gain per all-cold feed | loss per other feed | local break-even posterior | mean gate p |
|---|---|---|---|---|---|
| q0 (0.139) | 0.083 | +0.203 | −0.029 | 0.127 | 0.089 |
| q2 (0.282) | 0.164 | +0.158 | −0.053 | 0.250 | 0.164 |
| q4 (0.462) | 0.280 | +0.119 | −0.084 | 0.414 | 0.272 |

- In every quintile the gate's posterior is below the local break-even.
- The value-weighted AUC (weights = |payoff|) is therefore only 0.517-0.528, against a plain AUC of 0.67.
- An empirical gain-vs-AUC curve that keeps the real covariance gives: AUC 0.671 → +0.00015; 0.846 → +0.0058; 0.944 → +0.0139. Recovering a quarter of the oracle needs AUC ≈ 0.85.
- Intent is undetectable on this data: missed-positive feeds AUC 0.566 (INSIGHTS_1112 l.113); the future-window intent oracle is 0.093x the echo of the feed's own label (COVERAGE_MAP l.185).

### 4.2 The routing-signal check (executed today, CPU)

**Setup** [lens; moe/route_check.py, route_check.log]:
- 14,306 leak-free val feeds.
- Nested 5-fold × 2-seed user-grouped routers: LightGBM regression and classification, |Δ|-weighted, ridge, logistic, plus a soft feed-conditional blend weight.
- 31 label-free covariates, including two the 78 second-level arms never used: model agreement and G3 confidence.
- Controls: a permuted-covariate null, and an A/A pair (G3 half vs G3 half).

| alternative to G3 | Δ vs G3 (val) | router AUC (4 routers) | permuted null | nested routed gain vs nested global choice | soft conditional weight vs global |
|---|---|---|---|---|---|
| pop1x8 | −0.00080 | 0.498-0.506 | 0.496 | −0.0003..−0.0007 | −0.00087 ± 0.00049 |
| popfree4 | −0.00242 | 0.506-0.511 | 0.503 | +0.00014..−0.00104 | +0.00024 ± 0.00048 |
| C3 | +0.00074 | 0.494-0.498 | 0.497 | −0.0001..−0.0005 (ridge z −3.0) | −0.00030 ± 0.00024 |
| A/A control | −0.00017 | 0.481-0.500 | 0.500 | −0.0003..+0.0001 | −0.00027 ± 0.00027 |

- The top features were the same for every pair, A/A included: top-1 agreement, G3 top-1 margin, log M, maxsim. These say how much two pools disagree, never which one is right.
- **Holdout transfer.** The val-fitted router applied to the 30K holdout has AUC 0.504 and loses −0.00247 ± 0.00042 (z −6.0). Holdout-internal routers reach AUC 0.506-0.516 against a null of 0.503.
- **Staleness.** The slope of Δ(pop1 − G3) against cold share is +0.0033 ± 0.0074, implying +0.0004 ± 0.0009 on the colder private split: null.
- **Domain.** The production/exploration domain cannot route: val and test are entirely random-order, so the variable is constant at inference (docs/02_SOLUTION §5).

This extends the existing closures from a new angle: second-level combiners 0/78 with val-fitted weights anti-transferring at z −2.85 (DATA_PIPELINE §4); segment routing −0.0004..−0.0012; trajectory-cluster MoE / FiLM 0/8 (COVERAGE_MAP l.192).

### 4.3 Sparse MoE inside the ranker, and sparse upcycling

**A5, measured once** (nnx_exp_attn A5; 2-epoch AdamW era):
- **Design:** 8 routed SwiGLU experts of width 64 plus a shared expert in the 4 candidate blocks; top-2 sigmoid router.
- **Strength:** +0.00027 val / +0.00053 holdout alone; about 0 on top of retrieval; −0.00034 after the fine-tune.
- **Routing:** the router learned vertical specialisation (0.33-0.50 nats of mutual information with category). Specialisation was available and was worth nothing. Set block 1 collapsed.
- **Diversity** [review; skeptic_m2upcycle/a5_margin.log]: margin m vs the dense base +0.0032. That equals the trivial attention gate A1 (+0.0033) and is a sixth of the information change A4 (+0.0195). The quoted "MoE pair beats dense pair" result holds at the FT stage only (+0.00052 ± 0.00060) and reverses at TO and stage 2 (skeptic_m2_upcycle/a5_div.log).

**Sparse upcycling of the F7 plateau snapshots** (experts diverge only in the cooldown) is bounded by data already on disk [review; skeptic_m2_upcycle/cd_margin.log, skeptic_m2upcycle/cd_envelope.log, replay_envelope.log]:
- Across all 32 cooldown-only arms stored for CD roots 101-104, the frontier margin is median +0.0016 and max +0.0104 (rho70, a −0.0035 member). The bar is m >= ~0.10.
- A 4-of-20 swap inside G3 keeps pool corr at 0.9993-0.9998, so the diversity ceiling of the proposed in-pool replay is about +0.00001.
- Measured replays: val −0.00048..+0.00032. 0 of 14 arms pass that proposal's own gate.

**Externals.**
- Sparse upcycling's gains (Komatsuzaki et al., 2212.05055) are capacity gains in capacity-bound T5/ViT models given ~50% of the dense sunk cost; our cooldown is ~20% and capacity is not binding (§5).
- MoE beats one expert when the data has cluster structure that a single expert cannot fit (Chen et al., 2208.02813). Our cluster variables (vertical, leaf, M, coldness) are explicit inputs.
- Even large LLM MoEs often do not route by domain (Mixtral, 2401.04088).

### 4.4 Hard routing by a known prior: vertical and feed length

**Per-vertical experts** (STAR-style adapters):
- **Feeds are not single-domain.** 99.2% of val feeds are multi-vertical (mean 6.0 verticals), so adapters change cross-vertical comparisons inside one listwise softmax [review; skeptic_m3vert/vert_check.log].
- **No per-vertical function is left at G3's output** [review; skeptic_m3vert/vert_resid.log, vert_check.log]:

  | listwise correction on z(G3), nested by user | val vs G3 |
  |---|---|
  | one global function of f1-f5 | +0.00027 ± 0.00032 |
  | per-vertical function | −0.00068 ± 0.00042 |
  | per-vertical minus global | −0.00095 ± 0.00039 (z −2.4) |
  | per-vertical recalibration minus global (two λ) | −0.0006 / −0.0003 |

  The GBM category residual on G3 is −0.00040 val / −0.00071 holdout (z −3.07) (DATA_PIPELINE §1.4).
- **The mechanism does not port.** STAR (2101.11427) partitions BatchNorm statistics across separately served domains. Our LayerNorm keeps no statistics, so only a per-vertical FiLM remains, and its shift half duplicates the 54-code leaf embedding.

**Feed-length experts** (cooldown specialists for M >= 100; unreviewed):
- every cooldown data-selection arm hurt: rec33 −0.0021, rho70 −0.0035 (COVERAGE_MAP l.138);
- M-conditional weighting is a winner's curse (DATA_PIPELINE §4);
- long-feed errors are information-limited (win rates 0.37-0.46).

### 4.5 Multi-head experts by event type

- **No type label exists at train time.** The label is binary, and 83-87% of positives first return as an e1 card view from the feed surface. The type of a positive is visible only in the same user's later snapshot (CODES_DECODING_1113 l.68), which is the FUTWIN channel, closed at 0.093x the label echo (COVERAGE_MAP l.185).
- **No seesaw to fix.** PLE / MMoE address conflicting task gradients (Tang et al., RecSys 2020, DOI 10.1145/3383313.3412236). Main and K9 gradients are near-orthogonal: median of medians +0.007..+0.014; C15 PROCEED = False (COVERAGE_MAP l.272).

### 4.6 Verdict

MoE is closed in every form, for one structural reason: the variables that look predictive describe how feeds differ, not which model is right on a given feed. The one strongly predictive quantity, an attribute of the clicked item, comes from the label. Conditional computation inside the ranker specialises and gains nothing, because the dense net already conditions on every routing variable we have.

Nothing in this lens reaches P 0.002. The only remaining read is the CPU stage 0 of event-type heads (non-view share of positives), expected to FAIL on its first clause.

---

## 5. Scale and generative recommenders

### 5.1 The regime in numbers

| quantity | value | source |
|---|---|---|
| training feeds / candidates / positives | 3,121,208 / 247M / ~5.45M (2.2%) | nn/logs header; nnx_research_recsys_arch §0 |
| ranker | 1.01M parameters, 0.944M of them in 46 Muon matrices | r7_research l.80; CPU count [lens] |
| CPU parameter counts at other widths | d192 2.12M; d384 (2 blocks) 7.91M; d512 13.8M; d768 (4/2 blocks) 57.1M | scale lens §1 |
| compute per 10-epoch member | ~1.3e16 FLOPs (0.42 GFLOP per feed × 3.12M × 10): ~6 s at the quoted the vendor's peak rate peak, against 130-182 min of wall-clock, i.e. far below 1% utilisation | LLM_FEASIBILITY §3; w5_e10_base logs [lens] |
| data slope | +0.0037 val per doubling of labelled feeds (150K → 3.12M) | GLOBAL_RECAP §3.1 |
| Chinchilla framing (~20 tokens per parameter) | 5.45M click positives suggest ~0.3M parameters; the 247M K9-supervised candidates suggest ~12M | scale lens §1 |

Compute does not bind. A dense model 10-50x larger costs the same wall-clock order, and sparse MoE buys capacity at constant FLOPs, a problem we do not have.

### 5.2 Data-limited, information-limited or recipe-limited?

| # | measurement | result | reads as | source |
|---|---|---|---|---|
| 1 | d192 − d128 under m02, 5 vs 6 seeds, matched plateau steps 67.5-77.5K | val −0.00019 ± 0.00022, holdout −0.00005 ± 0.00015 | width does not raise the plateau | skeptic_scalediag/phases.log [review] |
| 2 | the same at mix 0.1, 7 vs 7 seeds | val −0.00008 ± 0.00021, holdout +0.00012 ± 0.00012; pooled holdout +0.00005 ± 0.00009 | same | matched_step.log [review] |
| 3 | d192 − d128 at the end | selected val −0.00085 ± 0.00050, holdout −0.00056 ± 0.00025; @last holdout −0.00157 ± 0.00025; d192 peaks at step 88.0K vs 97.65K; holdout drop from the selected step to @last −0.00105 vs −0.00002 | the wider model memorises the exploration pool late in the cooldown | phases.log [review]; GLOBAL_RECAP App. B |
| 4 | lr 1e-3 vs 2e-3 at d128 (7 vs 6 seeds) | plateau +0.0009 val / +0.0008 holdout, but selected −0.00104 ± 0.00036 and @last holdout −0.0018 | the plateau is a gradient-noise floor; a cooler Muon loses at the end | matched_step.log [review]; a1_phases A1.6 |
| 5 | 3 blocks at 10 ep; d256 (AdamW era) | +0.0003 val / −0.0004 holdout; −0.0008 | depth and width flat | COVERAGE_MAP l.62, l.152 |
| 6 | LLM trunks 0.44-6.92B at 150K | flat; random init = pretrained | larger models learn the same function | llm_large R8.1 |
| 7 | late train loss, 3 blocks vs base | 2.9448 vs 2.9487 | slightly more fit, no generalisation | PRETRAIN_PLAN §1.3 |

**The recipe mechanism nobody tested.**
- With `match_rms_adamw`, Muon scales each hidden update to spectral norm 0.2·lr·√max(A, B) (torch `_muon.py` l.80-81; nn/train.py l.229-230; Moonlight, 2502.16982).
- Spectral μP wants ‖ΔW‖ ∝ √(fan_out/fan_in), constant for square hidden maps (Yang, Simon and Bernstein, 2310.17813).
- So at the same lr, d192 trains 1.22x and d384 1.73x "hotter" per step on hidden matrices [lens].

**Its measured size is small.**
- Calibrated on row 4, d192's extra heat (0.29 octave) predicts plateau −0.0003, cooldown gain +0.0005 and selected val +0.0002 [review].
- Observed: cooldown gain −0.00042 ± 0.00038 (z −2.4 against the prediction) and selected val −0.00085 (z −2.2).
- Net of heat, extra width is worth about +0.0001 ± 0.0002 at the plateau. Extrapolated in log-width to d384, that is about +0.0003 ± 0.0006 (at most ~+0.001).
- A perfect fix for the late decline brings d192 level with d128, not ahead.

**External priors.**
- Ardalani et al. (2208.08489): for DLRM-class models, "parameter scaling is out of steam … data scaling is the path forward".
- Zhang et al. (2311.11351) scaled ID-based sequential recommenders to 0.8B on ~20M interactions, but only with layer-wise adaptive dropout and an Adam → SGD switch.
- Muennighoff et al. (2305.16264): under a data constraint, spend extra compute on epochs before parameters. This argues against capacity at fixed data.

**Classification.** Information- or data-limited, with P(capacity-helps under a corrected recipe) ≈ 0.04-0.08. This combines the scale lens's 0.07 with the reviewers' 0.04-0.08.

### 5.3 The one discriminating experiment

**WIDTH-GATE** (§6.3, item 1) is the repaired form of SCALE-DIAG.

| step | what | cost |
|---|---|---|
| 0 | generalisation-gap probe on stored @77500 snapshots, inference only | < 1 GPU-h |
| 1 | d192 re-cools with three arms: m02 control, cooldown dropout 0.2, and a per-matrix μP Muon multiplier | 8-12 GPU-h |
| 2 | d384, only on a GO | ~25 GPU-h |

- It respects GLOBAL_RECAP §7.4 ("no d384+ before a positive width slope at d192").
- It fixes four defects of the original spec:
  - a uniform lr cut would have cooled the 7 fixed-fan-in input projections and the retrieval keys;
  - the multiplier would have been overwritten by the per-step lr reset;
  - the gap probe was contaminated by the cooldown-pool replay;
  - an AND kill rule left a grey zone.
- **Outcome.** A FAIL closes ranker capacity at this data size for good, including BIGPOOL and any width arm under pseudo volume.

### 5.4 Larger ensembles of a bigger model

| quantity | value | source |
|---|---|---|
| extra roots of the same recipe | +0.00003 ± 0.00020 | docs/03_EXPERIMENTS §2 |
| infinite-member headroom | about +0.0002 | GLOBAL_RECAP §2.2 |
| member → pool transfer, noise-type gain (m02) | +0.0012 → +0.0003 (~0.25) | GLOBAL_RECAP §3.1 |
| member → pool transfer, bias-type gap (F10 vs F7x) | −0.00187 → −0.00213 ± 0.00068 (~1.1) | skeptic_p4_bigpool/pool_ladder.log [review] |
| member → pool transfer, d192 vs d128 | val 0.87, holdout 0.57 | skeptic_bigpool [review] |
| continuous-cooldown TO pools vs G3's TO branch on test | corr 0.991-0.995, so a "corr < 0.99" clause would be a coin flip for a same-family pool | skeptic BIGPOOL reviews |
| d384 root cost | 3-4.5 GPU-h (d192 took 1.29x the wall time at 2.1x the parameters) | skeptic_bigpool/plateau_width.log; skeptic_scalediag [review] |

- A bigger-model pool pays only through member strength of the bias type.
- **External:** the width gains of over-parameterised nets are largely initialisation variance that ensembling removes (Geiger et al., 1901.01608). At equal parameters, an ensemble of medium nets beats one large net (Lobacheva et al., 2007.08483).
- **BIGPOOL** (§6.2) is therefore a conditional build behind WIDTH-GATE. Its P is the same event as the gate's, not additive.

### 5.5 HSTU-style generative recommenders: replication cost and fit

**What HSTU is** (2402.17152):
- one causal sequence of interleaved items and actions, so every history position is a target;
- pointwise SiLU attention, U-gating, and a relative attention bias over position and time;
- M-FALCON batching of candidates;
- a compute power law at ~100B examples per job.

**What drives its scaling.** A re-examination (2412.00714) [review]:
- **Time.** Table 3 (ML-20M, HR@10): no bias 0.3083; position-only 0.3122; full bias 0.3376; RoPE 0.3149. From these numbers about 87% of the bias gain (0.3083 → 0.3376) comes from the time-difference buckets. We have no event timestamps (PRETRAIN_PLAN §2).
- **Depth.** Ranking AUC is not monotone in depth: ML-1M 0.7559 / 0.7530 / 0.7591 / 0.7943 / 0.7947 at 2/4/8/16/32 blocks; ML-20M 0.7813 at 2, 0.7992 at 24, 0.7914 at 32.
- **End-to-end.** In LUM, end-to-end HSTU tied DLRM (0.7334 vs 0.7338) (PRETRAIN_PLAN §3.1).

**What we already have of it.**

| HSTU ingredient | nearest built thing | result |
|---|---|---|
| next-event generative targets | pre2 | §3.1 |
| single causal stream over [user; history; candidates] | JOINT; LLM soft-token rankers | JOINT −0.0007, diversity +0.0011, 1.84x step time (COVERAGE_MAP l.216); LLM −0.0038..−0.0053 |
| next-event aux inside the ranker | E2 / JOINT-SSL | closed by argument, never run |
| time-aware relative bias | impossible | history has order only |

**Supervision is already dense.**
- Dense targets: 141.4M fe&new (26:1 against labelled positives), 51.1M in the last 64 positions (9.4:1), about 149M distinct feed-open runs [review].
- They are past clicks without their slates.
- K9 already supervises all 246.7M candidate rows (docs/02_SOLUTION §11).

**Replication cost.**

| item | estimate |
|---|---|
| ladder of 8M-400M at 3 epochs | 35-60 GPU-h [lens] |
| engineering | 3-5 days from Meta's Apache-2.0 `generative-recommenders` [lens]; 1-2 weeks to adapt it to content-only items with candidate sets, retrieval, K9 and the FT pipeline [review] |
| whole program with a family | 95-200 GPU-h |

**Calendar.** The training stop is Dec 2 and families must start by ~Nov 24 (GLOBAL_RECAP §7.1). A full HSTU-joint member therefore cannot be built, screened and pooled in time. Only its cheapest mechanism, a next-item aux interacting with width, can be read before then (SSL-WIDTH, §6.3, item 4).

### 5.6 More labels: train+val, holdout in the cooldown, pseudo-labels

- **Val in the plateau.** Val is 0.46% of feeds, worth +0.00002 by the slope. Val in the cooldown (valcd) reproduced exactly the FT's value (+0.00014; COVERAGE_MAP l.148).
- **Holdout and val in the m02 pool** [review; skeptic_alllabels/dose.py]:
  - Under m02 the number of pool draws is fixed: about 416 pool steps, ~87K feeds, 0.65 passes over 134,118 feeds. Enlarging the pool adds no draws.
  - The new information is about 15K holdout + 5.7K val presentations, ~0.0095 doublings, about +0.00004 val.
  - pp02 (+47% pool) gave val −0.00074 (r6_eval_cd_w2).
- **Test pseudo-labels lost −0.0004** (docs/03_EXPERIMENTS §8):
  - the student reads the teacher's inputs;
  - the unlabelled pool is 1/109 of the labelled one, against Noisy Student's 300M unlabelled per 1.28M labelled (1911.04252);
  - the test shift is popularity staleness, which pseudo-labels cannot correct.
- **A larger noised student** is the one form with a mechanism, but G3's FT soft targets for val feeds carry cross-fold val labels into the student's train-only half, and the form raises correlation to G3 by construction [review]. Drop it.

### 5.7 Verdict

The ranker is information- or data-limited, not capacity-limited. The one recipe mechanism that could make width pay is cheap to test and unlikely to fire. Generative recommenders scale on timestamps and ~100B examples; we have neither, and a full build does not fit the calendar.

---

## 6. What survived review

### 6.1 Review outcomes (all 17 proposals)

Two independent reviewers per proposal ("A" / "B"). "split" means one reviewer refuted the proposal as specified. In every split case both reviewers proposed a corrected test, and the corrections converge.

| proposal | lens | A / B | P: proposer → A / B | disposition |
|---|---|---|---|---|
| BIGPOOL: a 14-20 member family of a bigger ranker, behind a capacity verdict | scale | ok / ok | 0.012 → 0.005 / 0.004 | **survives**, corrected (§6.2) |
| SCALE-DIAG: d384 width ladder with recipe arms | scale | ok / refuted | 0.012 → 0.006 / 0.004 | repaired as WIDTH-GATE |
| GENREC-JOINT: HSTU-style joint generative + ranking model | scale | ok / refuted | 0.007 → 0.003 / 0.003 | repaired as SSL-WIDTH |
| ALL-LABELS-CD: holdout + val in the m02 pool | scale | ok / refuted | 0.004 → 0.002 / 0.001 | close by arithmetic (both reviewers) |
| P1 PSEUDO-FEED: co-train on history-built lists | next-item | ok / refuted | 0.006 → 0.003 / 0.003 | repaired as PSEUDO-DIRECT |
| P2 JOINT-SSL with impression negatives | next-item | ok / refuted | 0.004 → 0.003 / 0.002 | merged into SSL-WIDTH |
| P3 NEXTITEM-XL: content-only encoder scale-up + frozen K/V branch | next-item | ok (arm b) / refuted | 0.005 → 0.003 / 0.003 | repaired as ENCODER-DOSE |
| P4 PSEUDO-XL: wider ranker at pseudo volume | next-item | ok / refuted | 0.003 → 0.0015 / 0.0015 | folded into PSEUDO-DIRECT |
| P5 FT-PSEUDO: test-period pseudo-feeds in the FT | next-item | ok / refuted | 0.002 → 0.001 / 0.001 | close |
| SOFT-TOKEN-SLOPE: LLM full-data slope gate | pretrained | ok / refuted | 0.002 → 0.001 / 0.001 | close; optional 4-6 GPU-h one-pass read |
| M2 UPCYCLE-CD: sparse-upcycled MoE re-cools | MoE | refuted / refuted | 0.003 → 0.001 / 0.0008 | close |
| M3 VERT-HARD: per-vertical adapters | MoE | ok / refuted | 0.002 → 0.0015 / 0.001 | close (CPU pre-gate fails) |
| M1 ROUTE-CLOSE: feed-conditional routing between pools | MoE | executed on CPU | 0.001 | CLOSED (§4.2) |
| M4 SEG-RECOOL: feed-length cooldown experts | MoE | not reviewed | 0.002 (lens) | not recommended (§4.4) |
| M5 TYPE-HEADS: PLE heads by event type | MoE | not reviewed | 0.002 (lens) | CPU stage 0 only, optional |
| TEXT-LLM: 7-32B LoRA on rendered feeds | pretrained | not reviewed | 0.001 (lens) | close (§2.5) |
| TFM-STACK: TabICLv2 / TabPFN-3 stacker | pretrained | not reviewed | 0.0005 (lens) | close; optional 2 GPU-h closure read |

The median reviewer shrink of the proposers' priors is about ×0.5 (range ×0.3-0.6), milder than the ×0.3 measured in the global recap.

### 6.2 The surviving item: BIGPOOL, corrected by both reviews

| field | content |
|---|---|
| mechanism | If a width-corrected ranker is a stronger member of the bias type, build a family the way G3 was built (from-scratch roots with plateau snapshots, m02 re-cools from two snapshots, n2cpv3 FT of every member) and pool member-equal. Bias-type member gaps carry to the pool at ~0.6-1.1; noise-type gaps at ~0.25 (§5.4). |
| gate | WIDTH-GATE Step 2 PASS (§6.3, item 1), plus a BIGPOOL-specific GO read in Step 2 on an equal-count basis. The d384 TO pool must beat a pool of the same number of from-scratch d128 roots (w5_e10_base s951-956) by >= +0.0012 val and >= +0.0004 holdout. Report the change in seed correlation and the variance-adjusted gain, member Δ − 0.075·Δρ, so that a variance-type gain does not open the build. |
| roots | the 4 Step-2 primary seeds (launched with `--save_steps` at 70% and 79% of steps) + 3 new; never the Step-1 selection runs |
| interim kill (before any re-cool or FT) | 7-root TO pool >= the matched from-scratch d128 base (average over 7-member subsets of V30 s901-906 + w5_e10_base s951-956) + 0.0010 val AND + 0.0004 holdout, AND >= G3 TO + 0.0015 val, which projects to about G3 + 0.002 after re-cools and FT |
| build | 14 m02 re-cools (2 epochs, seeds analogous to 811/812); n2cpv3 FT for all 21 members; G3big = 0.5 FT + 0.5 TO member-equal; also C3big = G3big + the 8 pop1 members |
| final read | Compare against G3, against M2 (`cand_M_F7xF7x70_memeq`, matched re-cool share) and against C3. Apply the re-cool discount S = min(V, V − D) (CURRENT_STATUS l.328). Require holdout >= +0.0014. Report correlation to G3 as information only: continuous pools sit at 0.991-0.995, so a corr clause would reject real level gains. |
| promotion | one public slot only if val >= +0.002 AND holdout >= +2 SE; then FREEZE §3 rule 2b as written (public >= G3 + 0.0011, holdout not more than 1 SE below G3); presub_gate on every file |
| dropped | the Noisy-Student add-on. Val-feed soft targets leak cross-fold val labels into the TO half; test-only TO targets are a near copy by construction. |
| GPU-h | 30-45 with the Step-2 reuse (45-65 without) |
| calendar | start by Nov 24-26 so that FT, pool and gate fit before Dec 2 |
| P | ≈ 0.004 (P(CAP) 0.04-0.08 × P(pool >= +0.002 val given CAP) 0.10-0.15 × private transfer ~0.5). The same event as WIDTH-GATE, not additive. |

### 6.3 Repaired forms that both reviewers converge on (ranked)

These are the proposals refuted as specified, rewritten with the corrections both reviewers asked for. They have not been re-reviewed as rewritten. Ranking is by P per GPU-hour, adjusted for information value: WIDTH-GATE ranks first because it also settles the "a bigger model would win" hypothesis.

#### 1. WIDTH-GATE (from SCALE-DIAG; gates BIGPOOL)

| field | content |
|---|---|
| question | Is ranker capacity recipe-limited (Muon's √width over-step, cooldown memorisation) or not? |
| Step 0 (inference, < 1 GPU-h, information only) | GAP probe on the @77500 snapshots of v29 d192 (5), v30 d128 (6) and w5_e10_base d128 (6). Main-loss listwise CE in eval mode with no aux, on: the 30K holdout; 30K in-sample production feeds; 30K in-sample random-like feeds (p_random > 0.9, M >= 50, fixed seed). Read at the plateau end only, before step 78,680, where the cooldown-pool replay begins. Gap = CE(holdout) − CE(random-like in-sample). This classifies DATA vs INFO at d192 for free. |
| Step 1 (8-12 GPU-h) | CD-harness re-cools from @77.5K. d192: 5 v29 roots × 3 arms: (a) m02 control; (b) m02 + `--drop_cd 0.2` (flag exists in nn_x/r6); (c) m02 + a per-matrix μP Muon multiplier. d128: 5 v30 roots × arm (a). |
| μP implementation | Multiplier = √(fan_in at d128 / fan_in at d) for matrices whose fan-in grows with d (0.816 at d192, 0.577 at d384). 1.0 for the 7 fixed-fan-in input projections (c_lin, c_mlp.0, retr.wk, retr.t1, retr.t2, u_lin, h_lin; skeptic_scalediag/muon_shapes.log). Apply it inside the per-step loop, `g["lr"] = lr_at(step) * g["lr_mult"]`, because nn_x/w5/train.py l.438-439 resets every group's lr each step. Unit-test flag-off bit-identity. Log the effective lr of one hidden, one input and one fixed matrix. State that Muon's decoupled decay also scales. |
| GO to Step 2 (pre-declared) | best d192 arm − matched d128 re-cool >= +0.0008 val AND >= +0.0004 holdout, with no rise in member corr to the base; the 12 full-run d128 seeds (v30 + w5_e10_base) are reported as information. Otherwise close ranker width for good. |
| Step 2 (on GO, ~25 GPU-h) | 500-step smoke on an unshared GPU first. Then d384 (7.91M parameters) with the Step-1 winner's recipe × 4 fresh seeds, 10 epochs, `--save_steps` at 70% / 79%; optionally a second arm × 2 seeds as information. |
| Step 2 kill | member val < +0.0012 OR holdout < +0.0006 against the 12-seed d128 base (about 3% false passes, about 0.95 power at +0.002) |
| GPU-h | ~10-13 to the Step-1 verdict; +25 on GO; then BIGPOOL |
| P | 0.004 for the whole chain to a private gain |

The data-fraction ladder from SCALE-DIAG is dropped. No further labelled data exists, so it decides nothing, and it is underpowered (slope-difference SE ≈ 0.0005-0.0007).

#### 2. ENCODER-DOSE (from NEXTITEM-XL; also gates PRE2-XL and GENREC Stage 0)

| field | content |
|---|---|
| question | Does encoder quality drive ranker value, and does a pretrained *representation* (not 6 scalars) add anything? |
| (A) in-ranker dose (1.5-3 GPU-h) | 2-epoch pair s23/s24 with the 6 pre2s columns rebuilt from `pre2_fe_noid` / `new_noid` (these exist in data/feats and are 0.004-0.005 weaker zero-shot), against the existing abl_base_s23/24 and abl_xdrop_pre2_s23/24. **Kill encoder scaling** if the swap loses < 40% of the channel's worth (< 0.0005 of +0.00128): zero-shot quality would then not drive value. The read is ~1 SE at the bar; it can close, not confirm. |
| (B) PRE2-10EP (3-5 GPU-h) | abl_e10_xdrop_pre2 s951/952 vs w5_e10_base s951/952. Sequential gate, not a parallel read (GLOBAL_RECAP §5 row 7): >= +0.0015 val, holdout > 0, M>=50 > 0. Its runs are reused as the pre2-free reference by PSEUDO-DIRECT. |
| (C) frozen hidden-state K/V probe (~1 GPU-h export + ~2 GPU-h pair; ~1 day of code) | One frozen checkpoint for every row (`ckpt_w1r_v3`, corpus ends Sep 11), run ID-zeroed (trained with ID-dropout 0.3). Export 64-d last-layer states with one fixed PCA for train rows Sep 12-19, the holdout, val and test; earlier rows get a learned null token. One latent space, so there is no train/serve misalignment and no later-snapshot leak. 2-epoch pair: abl_base vs abl_base + a zero-init gated K/V adapter in the 2 cross-attention blocks. Stream states from pinned host RAM, at most 2 processes per GPU. **GO** only if val >= +0.0015 with holdout > 0 and stale-val / pmix vetoes >= −1 SE. |
| escalation (only if (C) passes AND either (A) shows the swap losing >= 40% of the channel's worth or (B) passes) | Encoder sweep 30M / 90M, 2 encoder seeds per size, in one latent space: a warm-start chain with a Procrustes R² >= 0.95 anchor check, or a single encoder on a corpus that precedes the scored rows. Gate on the nested residual on val and the holdout, and on cell AUC above the 0.570 / 0.632 controls. Then a 10-epoch pair and a family: 70-90 GPU-h in total. |
| GPU-h | 5-11 to the verdicts |
| P | 0.003 |

#### 3. PSEUDO-DIRECT (from P1 PSEUDO-FEED, with P4 folded in)

| field | content |
|---|---|
| question | Does co-training the full ranker on lists built from history events extract more than the pre2 input channel already does from the same targets? This is the only route that turns GPU-hours into training examples. |
| construction (1.5-2 days of code, flags default off and bit-identical) | Pseudo-feeds built on the GPU from the batch's own histories. Target: a new type-1/source-1 event at t <= 63. History: the events older than t + g, g ~ U[16, 64], recency re-indexed. Negatives: the source feed's non-clicked candidates. user_emb zeroed (a learned no-user vector); a fourth domain embedding; f1-f5 and the 34 pre-computed columns on their missing path; K9 weight 0; listwise CE weight 0.3; plateau only. The cooldown stays on real feeds for an absolute 19,670 steps (never cooldown_frac × total). Holdout users and later same-user snapshots are never a source. The pseudo domain gets its own context-free linear head on [content, category], so the shared pathway is not paid to learn the past-click marginal (shortcut AUC 0.590 is not removed by logQ). Report a set-attention-masked variant on pseudo rows as information (set leak 0.608 vs 0.582). |
| first test (10-20 GPU-h) | 10-epoch 2-seed pair at pseudo:real 1:1 against w5_e10_base s951/952, @last primary, E rule + diversity clause. Information arm: pseudo + xdrop_pre2 vs abl_e10_xdrop_pre2 (from ENCODER-DOSE (B)), to test substitution. No 2-epoch GO: gains shrank from 2 to 10 epochs in every recorded case. |
| kill | member Δval < +0.001, OR holdout < 0, OR the gain appears only without pre2 |
| escalation | 4:1 dose read (12-14 GPU-h), then a 7-root family with FT (60-70 GPU-h). A width arm only if the pseudo d128 member beats real-only by >= +0.0037 val (one effective doubling) AND WIDTH-GATE is GO. In that case one d192 pseudo pair (20-26 GPU-h) must reach d192 − d128 >= +0.0015 val, holdout >= +1 SE and an in-pool replay >= +0.0005. |
| hard stops | no pass by Nov 21 → no family; no width gate by Nov 22 → no width arm |
| P | 0.003 (the width increment adds ~0.001 that overlaps WIDTH-GATE) |

#### 4. SSL-WIDTH (from GENREC-JOINT and P2 JOINT-SSL)

| field | content |
|---|---|
| question | Does dense next-item supervision on the history pathway let a wider ranker generalise (the HSTU/ARGUS mechanism), or improve the d128 member? |
| Step −1 (CPU, 0 GPU-h) | the E2 reopen trigger of PRETRAIN_PLAN §5: partial corr > 0.05 at 2 SE between the per-feed G3 NDCG residual and the hidden-event recall@10 of the windowed w3r encoder on post-window holdout feeds, controlling for M, H and cold share. **Fail → close GENREC-JOINT, JOINT-SSL and SSL-as-capacity-enabler together at 0 GPU-h.** |
| Step 0 (on pass, 6-8 GPU-h) | 2×2 of 2-epoch screens, {d128, d192} × {aux off, aux on}, seeds 23/24; d128-off reuses abl_base_s23/24. Aux: an ID-free next-fe sampled softmax with logQ over in-batch history items only. No current-feed candidates as negatives: they are a shortcut and a label channel. Targets at a gap of 16-64 events from the state, over all fe&new positions. Weight 0.1 (0.01 as information; TransAct V2's optimum is 0.01, 2506.02267). Causality: F10 uses only the reverse GRU half (the forward half reads the target; perturbation test 0.297 vs 0.0, skeptic_p2/gru_dir_check.py); the transformer uses a newest-first key mask key_idx > query_idx with no user token. Unit test: perturbing event j leaves the aux state at j+1 bit-identical. |
| kill | d128+aux < −1 SE val, OR interaction (d192+aux − d128+aux) − (d192 − d128) < +0.001 val at 2 SE, OR holdout sign disagrees; read at the last step, the matched plateau step as information |
| escalation | a 10-epoch 4-seed pair at a matched epoch count (20-30 GPU-h); level rule member >= +0.002 val, >= +0.0014 holdout, in-pool replay >= +0.0005; then a 20-member rebuild (+30-70 GPU-h). The full HSTU joint model is out of calendar (§5.5). |
| P | 0.002 |

#### 5. LLM ONE-PASS (from SOFT-TOKEN-SLOPE; information only, not recommended)

| field | content |
|---|---|
| question | Does the LLM's one-pass level beat the ranker's at 4x more data? It must, by about +0.002, to offset the ranker's multi-pass gain (§2.4 a). |
| spec (4-6 GPU-h) | 600K feeds. L = Qwen3-0.6B L1, 1 epoch, cosine, holdout every 200 steps. B = the exact 150K B_FLAGS with only `--train_feeds 600000` changed, seeds 23 AND 24 (compare.py needs two B seeds). Select on even-index holdout feeds, read odd-index. |
| kill | L(1 pass) − B(1-pass step) < +0.002 on the read half, OR B's multi-pass gain at 600K >= +0.004. Either closes the LLM-backbone route at every size. |
| P | 0.001; run only on an otherwise idle lane |

### 6.4 Refuted or closed items

| item | why it does not run | evidence |
|---|---|---|
| M2 UPCYCLE-CD (refuted by both) | A cooldown-only change cannot create diversity. 32 cooldown arms have margin median +0.0016, max +0.0104, against >= 0.10 needed. A from-scratch MoE (A5) reached only +0.0032, equal to an attention gate. A 4-of-20 swap keeps pool corr >= 0.9993. Spec defects: GELU FFN, γ 1e-2 contrary to DeepSeek-V3's practice (2412.19437), mismatched control. | skeptic_m2_upcycle/cd_margin.log; skeptic_m2upcycle/*.log |
| ALL-LABELS-CD (both reviewers: close) | m02 pool draws are fixed (~416 steps, ~87K feeds, 0.65 passes), so pool size adds no draws: ~+0.00004 val. pp02 (+47% pool) −0.00074; valcd = FT. No FREEZE path for holdout-trained members. | §5.6 |
| P5 FT-PSEUDO (both reviewers: information at most) | The information is already in G3 (LUO counts over all snapshots; pre2 trained on val/test snapshots with the same target; transductive ≤ +0.0004). Pseudo rows carry no freshness inputs, so they cannot reach staleness. Volume 14x, not 90x. The OOF lens is blind to the FT effect. The leak rule as written lets later same-user snapshots in. | skeptic_p5_ftpseudo/*.log |
| M3 VERT-HARD (both reviewers: close or 1 GPU-h information) | 99.2% of feeds are multi-vertical. The per-vertical function residual on G3 is −0.00095 ± 0.00039 (z −2.4). A5's router already learned vertical routing for 0. LayerNorm has no statistics to partition. | §4.4 |
| SCALE-DIAG as specified | uniform ×0.577 is not μP (it cools input maps and retrieval keys); the flag is lost to the per-step lr reset; skips the d192 gate; AND kill rule; CAP read on holdout-selected seeds (winner's curse ~+0.0003); gap probe contaminated by the pool replay; best-of-4 2-seed screen passes 19% under the null | skeptic_scalediag/*.log → WIDTH-GATE |
| GENREC-JOINT as specified | Stage 0 reopens F8 without its trigger and reads zero-shot quality, which is off the ranker's path. The frontier arithmetic gives +0.0003, not +0.0013. No timestamps for the HSTU time bias. Supervision is 26:1 (9.4:1 in the last 64 positions), not 80:1. True cost is 95-200 GPU-h plus 1-2 weeks. | → SSL-WIDTH |
| P1 PSEUDO-FEED as specified | the pool is 8-10x, not 26x; logQ in-batch negatives leave the shortcut at 0.590; set-centroid leak; XRATE draws pseudo sources from unseen users and runs in the overfit regime, so it is biased to pass | → PSEUDO-DIRECT |
| P2 JOINT-SSL as specified | the forward GRU half leaks the target; current-feed negatives are a shortcut and a label channel; an 8:1 label-scarce gate cannot discriminate the 1:1 deployment | → SSL-WIDTH |
| P3 NEXTITEM-XL as specified | arm (a) is F8 under the same ceiling; zero-shot quality is unrelated to value; two of three gate clauses are vacuous; a K/V branch across separately trained encoders is misaligned | → ENCODER-DOSE |
| P4 PSEUDO-XL as a separate line | volume overstated ~2.5x (17.8 real-epoch equivalents, not 45); pseudo volume does not touch the cooldown phase where width lost; cost 100-130 GPU-h, not 400 | folded into PSEUDO-DIRECT |
| SOFT-TOKEN-SLOPE as specified | GO unreachable with one B seed; B's recipe changes between scales; the deficit is multi-pass; structural correlation 0.968-0.974 | → LLM ONE-PASS (information only) |
| M1 ROUTE-CLOSE (executed) | router AUC = null on val and holdout; holdout transfer −0.0025 (z −6.0) | moe/route_check.log |
| M4 SEG-RECOOL (unreviewed) | cooldown data selection hurt every time; M-conditional weighting is a winner's curse | §4.4 |
| M5 TYPE-HEADS (unreviewed) | no train-time type label; no gradient conflict for PLE to fix. Optional CPU stage 0 (non-view share of positives >= 20%), expected FAIL | §4.5 |
| TEXT-LLM (unreviewed) | strictly less information than the soft-token arms; 300-600 GPU-h per epoch at 7B; negative public evidence on ID-only data | §2.5 |
| TFM-STACK (unreviewed) | ceiling = measured second-level headroom (~+0.0002 val); pointwise; no ranking objective | §2.2 |

---

## 7. Recommendation

### 7.1 What to pre-declare and run, in order

1. **Day 0 (Nov 14, 0 GPU-h).**
   - Pre-declare WIDTH-GATE, ENCODER-DOSE, PSEUDO-DIRECT and SSL-WIDTH with `date -u`, copying the kill rules of §6.3 verbatim.
   - Write a dated amendment of GLOBAL_RECAP §7.4: the d192 gate now includes the per-matrix μP arm, and d384+ still runs only after a positive d192 slope.
   - The lead records the closures of Appendix B in COVERAGE_MAP.
   - Code: the per-group `lr_mult` and its bit-identity test; the GAP-probe script; the fe_noid column rebuild; the E2 trigger on CPU; start the pseudo-feed builder (1.5-2 days).
2. **Nov 15-17.**
   - WIDTH-GATE Step 0 (inference) and Step 1 (20 re-cools).
   - ENCODER-DOSE (A) and (B).
   - The E2 trigger, then SSL-WIDTH Step 0 if it passes.
   - The K/V state export and probe (C).
3. **Nov 18-20.** PSEUDO-DIRECT 10-epoch pair and its pre2-free information arm. On a WIDTH GO, d384 Step 2.
4. **Nov 21-22: verdicts, with hard stops.**
   - no WIDTH GO by Nov 19 → no d384;
   - no Step-2 pass by Nov 22 → no BIGPOOL;
   - no PSEUDO pass by Nov 21 → no family;
   - no SSL 2×2 pass by Nov 20 → no 10-epoch pair.
5. **Nov 22-26: families only on GO.** BIGPOOL starts by Nov 24-26; a pseudo or encoder family by Nov 22-24.
6. **Dec 2:** last day to start training. **Dec 2-6:** presub_gate and the FREEZE §3 decisions (GLOBAL_RECAP §7.1).

None of this displaces the zero-GPU items of GLOBAL_RECAP §6-7: the pick-2 amendment, the counting-rule check, merge outreach and the compliance dry run. Those come first.

### 7.2 Budget

| block | GPU-h | condition |
|---|---|---|
| WIDTH-GATE Steps 0-1 | ~10-13 | committed |
| ENCODER-DOSE (A) + (B) + (C) | ~5-11 | committed |
| SSL-WIDTH Step 0 | 0 or 6-8 | only if the CPU trigger passes |
| PSEUDO-DIRECT pair + information arm | ~10-20 | committed (after 1.5-2 days of code) |
| **committed, no GO** | **~30-45** | |
| d384 Step 2 | ~25 | WIDTH GO |
| BIGPOOL | 30-45 | Step-2 pass + BIGPOOL GO |
| pseudo 4:1 dose + 7-root family + FT | 70-85 | PSEUDO pass |
| pseudo d192 width pair | 20-26 | PSEUDO >= +0.0037 and WIDTH GO |
| encoder sweep + family | 70-90 | ENCODER GO |
| SSL 10-epoch pair + rebuild | 50-100 | SSL pass |
| **maximum, every gate passing** | **~300-400** | under 10% of the ~5,000 idle GPU-h |
| **expected** | **~40-60** | P(any gate passes) ≈ 0.2-0.3 |

The other ~4,900 GPU-h have no positive-expected-value use on this evidence. More seeds, roots, epochs and batch are 0 ± 0.0002 at pool level (GLOBAL_RECAP §7.4), and no proposal in this study converts GPU-hours into information. Leaving GPUs idle is the correct outcome, not a failure of planning.

### 7.3 Combined probability

| chain | P(>= +0.002 private) | P(>= +0.001 private), approximate |
|---|---|---|
| WIDTH-GATE → d384 → BIGPOOL | 0.004 | 0.008 |
| ENCODER-DOSE → K/V branch / encoder family | 0.003 | 0.007 |
| PSEUDO-DIRECT → family (+ width arm) | 0.003 | 0.008 |
| SSL-WIDTH → aux member / rebuild | 0.002 | 0.005 |
| LLM ONE-PASS (if run) | 0.001 | 0.002 |
| **union** (positively correlated: all need "capacity or supervision is the limit") | **≈ 0.01 (0.007-0.012)** | **≈ 0.025-0.03** |

- Expected private gain of the program: about +0.00005.
- This sits below the zero-GPU pair decision ({C3, G3}: +0.0003..+0.0007 E[max] over G3) and below a merge (+0.0004..+0.0008 given a near-equal partner; GLOBAL_RECAP §6.3-6.4).
- The leader's +0.0048 would need about three generational steps; P < 0.01 for any route in this study.

### 7.4 House rules for every run

- the GPUs are allowed. At most 4 training processes per GPU, >= 60 GB free before a launch, and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=12` per process.
- Launch detached, with real PIDs recorded, and stop only by exact PID.
- Every new flag defaults off and is bit-identical when off, with a unit test before launch.
- Pre-declare with `date -u` before the first number. Read @last, or on a pre-declared selection set; never select on val.
- Information lanes are firewalled from the frozen set: nothing enters except through FREEZE §3.
- `src/presub_gate.py` runs on every candidate file, and a FAIL blocks it.
- Public slots go only to a generational pass (expected 0).

### 7.5 Do not

- Spend 500-1,000 GPU-h on encoder pre-training or an HSTU-scale build. Encoders are data-constrained at a few GPU-h per copy, and a joint build does not fit the calendar.
- Run d384+ before the d192 gate passes, or cut Muon's lr uniformly: that cools the input projections and retrieval keys below the tuned setting.
- Run LLM backbones at any size or over text; use TFMs in any role; route between pools by feed covariates; upcycle experts in the cooldown; add per-vertical adapters.
- Distil G3 soft targets on val feeds into any student: this leaks cross-fold val labels into the TO half.
- Use label-scarcity gates that draw pseudo sources from unseen users: they are biased to pass.
- Treat zero-shot encoder NDCG, or a within-feed corr cap that every existing score passes, as a gate.

---

## Appendix A. Sources

**Project files**
- reports/GLOBAL_RECAP_1113.md §1-8, App. B
- reports/COVERAGE_MAP.md (cited as l.N)
- reports/PRETRAIN_PLAN_1113.md §1-7
- reports/DATA_PIPELINE_1113.md §1.4, §3, §4
- docs/02_SOLUTION.md §4-12; docs/03_EXPERIMENTS.md §1-14
- reports/llm_large.md R3-R8; reports/LLM_FEASIBILITY.md; reports/llm_research.md; reports/llm_build.md; nn_x/llm_manual/RESULT.txt
- reports/nnx_research_recsys_arch.md §0-1; reports/nnx_research_llm_arch.md; reports/nnx_research_tabular_dl.md §2.7; reports/nnx_research_training_gen.md; reports/deep_seq_pretrain_cf.md; reports/deep_neural_rankers.md
- reports/R7_ARCH_HPARAM.md; reports/r7_research.md
- reports/pretrain_report.md; reports/nnx_exp_attn.md (A5); reports/CODES_DECODING_1113.md; reports/EMBEDDINGS_1113.md
- reports/CURRENT_STATUS.md (l.298, l.328, log entries cited)

**Lens reports and reviews (full text).** Workflow journal: (private path) workflow run/journal.jsonl. Copies:
- /tmp/scratchpad/power/synth/lens_pretrained.md
- /tmp/scratchpad/power/synth/lens_nextitem.md
- /tmp/scratchpad/power/synth/lens_moe.md
- /tmp/scratchpad/power/synth/lens_scale.md
- /tmp/scratchpad/power/synth/reviews_all.md

**CPU probes** (session scratchpad, which may be cleaned; the key numbers are reproduced above). All paths are under /tmp/scratchpad/power/:

| area | files |
|---|---|
| pretrained | pretrained/llm_axis.log; pretrained/llm_blend_fine.log |
| next-item | nextitem/diag1.log; diag1b.log; diag2.log; diag3.log |
| MoE | moe/route_check.log; moe/cold_mech.log; moe/out/route_check.json |
| width | skeptic_scalediag/phases.log; matched_step.log; screen_sim.log; muon_shapes.log |
| bigger-model pools | skeptic_bigpool/plateau_width.log; comparator.log; skeptic_p4_bigpool/pool_ladder.log; ens_gain.log |
| LLM | skeptic_llmslope/passes.log; fulldata_passes.log; skeptic_softslope/struct_corr.log; member_level.log |
| pseudo-feeds | skeptic_p1/k1.log; k2.log; skeptic_p1_pseudofeed/k1_pool_t63.log; k2_set_shortcut.log; skeptic_p4/k1_pseudo_count.log |
| encoder | skeptic_p3/dose.log; scale_transfer.log |
| SSL and generative | skeptic_p2/gru_dir_check.py; skeptic_genrec/last64.py |
| FT pseudo-feeds | skeptic_p5_ftpseudo/k1_pool_leak.log; k2_format.log; k2_ft_lens.log |
| upcycled MoE | skeptic_m2_upcycle/cd_margin.log; a5_div.log; skeptic_m2upcycle/cd_envelope.log; replay_envelope.log; a5_margin.log |
| per-vertical experts | skeptic_m3vert/vert_check.log; vert_resid.log |
| all-labels cooldown | skeptic_alllabels/dose.py |

**External sources.** [V] = read this session by a lens or reviewer; [P] = cited through a project report; [K] = background, not re-checked.

| area | sources |
|---|---|
| generative recommenders | HSTU, arXiv 2402.17152 [V]; HSTU re-examination, 2412.00714 [V]; github.com/meta-recsys/generative-recommenders (Apache-2.0) [V]; TIGER 2305.05065 [V]; OneRec 2506.13695 [P]; OpenOneRec 2512.24762 [V] |
| pre-training conditions | GPSD 2506.03699 [V]; LazFormer 2609.14978 [P]; PinFM 2507.12704 [V]; ARGUS 2507.15994 [P]; LUM 2502.08309 [P]; PinnerFormer 2205.04507 [P]; Layer 6 RecSys Challenge 2022 (cs.toronto.edu/~mvolkovs/Recsys_2022.pdf) [P] |
| self-supervision and scale | Zoph et al. 2006.06882 [V]; Yao et al. 2007.12865 [V]; Muennighoff et al. 2305.16264 [V]; Zhang et al. 2311.11351 [V]; Ardalani et al. 2208.08489 [V]; Kaplan et al. 2001.08361 [K]; Hoffmann et al. 2203.15556 [K]; Nakkiran et al. 2003.01897 [K]; SASRec vs BERT4Rec 2309.07602 [P]; TransAct V2 2506.02267 [V] |
| optimiser and width | Moonlight 2502.16982 [V]; spectral μP, Yang, Simon and Bernstein 2310.17813 [V]; 2601.01306 and 2512.05620 [S: search snippets only]; Geiger et al. 1901.01608 [V]; Lobacheva et al. 2007.08483 [V] |
| pseudo-labels | Noisy Student 1911.04252 [V]; test-time training, Sun et al. 1909.13231 [V] |
| MoE | sparse upcycling 2212.05055 [V]; Branch-Train-MiX 2403.07816 [P]; Mixtral 2401.04088 [V]; Chen et al. 2208.02813 [V]; STAR 2101.11427 [V]; MMoE (Ma et al., KDD 2018) [V]; PLE (Tang et al., RecSys 2020, DOI 10.1145/3383313.3412236) [V]; aux-loss-free balancing 2408.15664 [V]; DeepSeek-V3 2412.19437 [V] |
| LLMs for recommendation | RecBench 2503.05493 [V]; Xu et al. 2408.14238 [S]; Hou et al. 2305.08845 [S]; Lost in Sequence 2502.13909 [P]; Kang et al. 2305.06474 [S]; Tan et al. 2406.16964 [P]; Yuan et al. 2303.13835 [P] |
| tabular foundation models | TabICLv2 2602.11139 (github soda-inria/tabicl) [V]; RTFM 2512.03307, TabPFN-MT 2605.20234, TabDPT 2410.18164 [S] |
| tabular deep learning | FT-Transformer (Gorishniy et al., NeurIPS 2021, 2106.11959) [K]; Grinsztajn et al. (NeurIPS 2022, 2207.08815) [K]; ViT (Dosovitskiy et al., ICLR 2021, 2010.11929) [K] |

## Appendix B. Rows to add to COVERAGE_MAP (for the lead; this report does not edit it)

| row | text |
|---|---|
| new: feed-conditional routing G3 vs popularity-light / same-recipe pools (31 label-free covariates incl. model agreement and confidence) | CLOSED 11-13 (power/moe/route_check.py). Router AUC 0.494-0.511 vs permuted null 0.496-0.503 and A/A 0.48-0.50. Nested routed gain vs global −0.0003..−0.0010. Soft conditional weight −0.00087 ± 0.00049 (pop1x8). Val→holdout AUC 0.504, −0.0025 (z −6.0). Domain constant at inference. |
| l.120 cold gates / routers | Re-run on G3 11-13: the oracle +0.0257 is a label echo. Gate AUC 0.65-0.67 but value-weighted AUC 0.52-0.53 (cold share predicts the label and anti-predicts the payoff in every quintile). Empirical AUC 0.85 → +0.006. Nested −0.00005 ± 0.00004. CLOSED for good. |
| l.61 MoE (sparse, upcycled) | A5 diversity margin +0.0032 = attention gate (+0.0033). 32 cooldown-only arms: margin median +0.0016, max +0.0104 (bar 0.10); 4/20 swap pool corr >= 0.9993. Upcycled-MoE cooldown CLOSED without a run. |
| new: per-vertical experts / adapters | CLOSED at the CPU pre-gate 11-13: 99.2% of feeds multi-vertical (mean 6.0); per-vertical function minus global residual on G3 −0.00095 ± 0.00039 (z −2.4); recalibration −0.0006 / −0.0003. |
| new: ALL-LABELS-CD (holdout + val in the m02 pool) | CLOSED by arithmetic 11-13: m02 draws fixed (~416 pool steps, ~87K feeds, 0.65 passes), so pool size adds no draws; ~+0.00004 val. pp02 −0.00074; valcd = FT. |
| new: FT-PSEUDO (val/test-period pseudo-feeds in the FT) | CLOSED by mechanism 11-13: the information is already in LUO counts and pre2; pseudo rows carry no freshness inputs; eligible volume 14x val positives; the OOF lens is blind to FT effects. |
| l.137 LLM backbones, full-data slope | 11-13: all 150K arms are level after one pass (holdout 0.297-0.299); the deficit is the ranker's multi-pass gain (+0.004 at 150K; +0.0047/+0.0062 then +0.003 cooldown at full data). Structural corr 0.968-0.974 → family worth ≤ +0.0006. The 20-30 GPU-h LLM-SLOPE gate is replaced by an optional 4-6 GPU-h one-pass read (kill unless L − B >= +0.002 after one pass). |
| l.152 capacity (width) | 11-13 matched steps: d192 − d128 plateau 0 (pooled holdout +0.00005 ± 0.00009); d192 peaks 9.65K steps earlier; @last holdout drop −0.00105 vs −0.00002: cooldown memorisation, not Muon heat. Open only as WIDTH-GATE (d192 re-cools: m02 / drop_cd 0.2 / per-matrix μP), 8-12 GPU-h. |
| l.177 pseudo-feed ranker training | 11-13 volumes: usable-window targets 44-56M distinct (8-10x labelled positives, not 26x). Break-even 0.03-0.08 feed per target vs ~0.006 implied by pre2's worth. Shortcut AUC 0.590 survives logQ in-batch negatives; set-centroid leak 0.608 vs 0.582. Open only as PSEUDO-DIRECT (10-epoch pair, ~10-20 GPU-h). |
| l.176 / l.190 pre2 scale | 11-13: across 19 stored variants (zero-shot 0.3496-0.3595) the nested residual on G3 has slope +0.00003 per +0.001 zero-shot; zero-shot is not a gate. Open only via ENCODER-DOSE (fe_noid dose pair + PRE2-10EP + frozen K/V probe). |
