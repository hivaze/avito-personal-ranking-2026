# Small LLMs (Qwen3 / Qwen3.5, 0.6B-9B) as rankers for the anonymised Avito feeds: evidence, compute math, decisive experiment

Status: research and design only. No GPU was used, no project code was edited, and nothing was trained. Date 2026-11-12. Author: LLM-research agent.
Files: this report; `nn_x/llm/research_compute_math.py` (pure-python FLOP/time/memory calculator) and its output `nn_x/llm/research_compute_math.out`.
A separate CPU-only workflow (CURRENT_STATUS 01:00) is building `.venv_llm` and running the smoke test for `reports/LLM_FEASIBILITY.md`. This report does not touch either. Section 4.8 lists the checks that smoke test should cover.

Evidence tags:
* **[V]** = primary source read this session (URL in Sources).
* **[S]** = search snippet or secondary summary only.
* **[K]** = background knowledge, not re-verified this session.
* **[P]** = this project's own files or measurements.
* **[inf]** = my own arithmetic or inference.

---

## TL;DR, verdict: **LONG SHOT**. Run one cheap, pre-declared, matched-data test (about 1.5-3 GPU-h) when a GPU lane frees up. Never displace NN-family training for it.

1. **The data contains no text at all, so an LLM's knowledge has nothing to attach to** [P].
   - Item IDs, user IDs, categories (1..53), event codes and the int8 128-d content embedding are all anonymised.
   - An LLM can contribute only (a) a pretrained *computation prior* (its attention and MLP circuits) applied to soft tokens we project into its hidden space, or (c) distillation of that same model.
   - Option (b), embedding or serialising numbers as text, is dead. There is no text to embed, and Qwen3-Embedding-4B's weights are not even in the local cache. Text serialisation would cost about 287 tokens per item, i.e. about 76K tokens per feed instead of about 265 soft tokens [inf].
2. **When the inputs are non-language and someone actually ran the ablation, the pretrained-LLM advantage over a same-compute model trained from scratch is about zero.**
   - Tan et al. (NeurIPS 2024) [V]: ablations without the LLM beat Time-LLM in 26/26 dataset-metric cases, CALF in 22/26 and OneFitsAll in 19/26. They train in 1.9 min vs 3,003 min (Time-LLM, Weather).
   - Chronos [S]: random initialisation is at least as good as T5 language initialisation.
   - Positive results come with caveats: toy tasks or a comparison against *frozen random* transformers (FPT [V]), low-data regimes (Wikipedia-RL [V], ORCA [V], HLLM's Pixel200K ablation [V]), text-bearing items (HLLM, LLaRA, CoLLM, OpenOneRec), or weak, unmatched baselines (E4SRec vs BCE-SASRec [V]).
   - LLM4Rec models are measurably insensitive to item order ("Lost in Sequence", KDD'25 [S]).
   - Large ID-only industrial recommenders (HSTU, OneRec-V1) are trained from scratch [V/K].
   - In the competitions on anonymised logs I could find (OTTO, RecSys Challenge 2025), the top solutions did not use pretrained LLM backbones [S/K].
3. **Our own evidence says the bottleneck is information, not trunk capacity** [P].
   - d=256 (3M params), 3 blocks, 8 heads, MoE, TabM, GRU/DIN/deep variants: all null or "similar alone".
   - Error analysis: "no candidate signal has residual information beyond the model", "missing-information frontier".
   - An LLM trunk adds capacity and a prior, but no new information.
4. **The bar for being useful is high** [P].
   - The strength-diversity law (sol_ensemble 2.5) puts the break-even gap at 0.049(1-ρ) on val.
   - So a full-data LLM member must reach train-only val ≈ 0.393+ at ρ ≈ 0.96, or ≈ 0.390 at ρ ≈ 0.90. The best family's single members are at 0.3946 (F7 mean).
5. **It is expensive per feed but cheap to test** [inf].
   - Per training feed, an LLM costs 1.7K× (Qwen3-0.6B) to 27K× (Qwen3.5-9B) the ranker's FLOPs.
   - One full-data epoch on one exclusive GPU takes 2.2 h (0.6B), 4.2 h (Qwen3.5-2B), 11 h (4B) or 21 h (9B), with pessimistic times 2× that. A 10-epoch ranker member is about 4-5 h wall-clock on a shared GPU (2 epochs: 12-20 min).
   - The decisive matched-data test at 150K feeds with Qwen3-0.6B (2 seeds plus controls) is about 1.5 GPU-h exclusive, 3 GPU-h pessimistic, plus about one agent-day of engineering.
6. **Recommended design** (section 4):
   - **Qwen3-0.6B** (pure softmax attention, local).
   - Soft tokens [user, history old→new, candidates]. Each candidate attends to user + history + itself only, and **all candidates share position id H+1**, so each one is scored exactly as "the next token" and the model is permutation-invariant in one pass.
   - LoRA r=32 on all linear layers, plus trainable projector, RMSNorm and head.
   - The same inputs (cvnd_pre2s), the same listwise loss + K9 aux, and the same holdout/val as the ranker.
   - The ranker baseline is run with `nn/train.py --train_feeds 150000` on the identical subset.
   - Pre-declared acceptance A1 (beats the ranker) and A2 (useful second member), with explicit stop rules.
   - Qwen3.5 (2B/4B/9B) is **not** the first choice. 18 of its 24 (2B) or 24 of 32 (4B/9B) layers are Gated DeltaNet, which ignores any custom attention mask. I verified this in transformers 5.16.1: `create_recurrent_attention_mask` returns None for a 4D mask. It needs a custom "branch" implementation (section 4.3).
7. **Subjective priors (speculation, for planning only):**
   - P(A1: 0.6B beats the ranker at matched data by > 2 SE) ≈ 10-20%.
   - P(A2: useful second member at matched data) ≈ 25-35%.
   - P(a full-data LLM member passes gates (a)+(b) and the holdout selector adds it to the final pool) ≈ 5-10%.
   - Gain if it does: +0.0003..+0.0008 val, about what one more same-strength NN family brings.

---

## 0. Facts about our setting that decide the question

| fact | value | source |
|---|---|---|
| Text anywhere in the data | none; all IDs anonymised, categories 1..53, event_type 1..10, event_source 1..30, `f1..f5` normalised floats, int8[128] item and user embeddings | data/competition/readme.md [P] |
| History timestamps | none; history is ordered newest-first only | readme [P] |
| Data volume | 3.15M train feeds, 246.7M candidate rows (78.3/feed), 583.6M history rows (185.2/feed), ~1.73 positives per train feed; val 14,423 feeds (96.4 cand, 173.3 hist); test 28,595 (101.6 / 173.9) | data/flat [P], measured this session |
| Tokens per feed (1 user + H + M) | train mean 264.5 (p50 271, p90 415, max 557); val 270.7; test 276.5; holdout (M≥50) 331.3 | measured [P] |
| Train candidate storage order | random w.r.t. `pos` and target (mean Spearman −0.005 and +0.0002 over 5K feeds), so storage order leaks nothing | measured [P] |
| Current ranker | 0.86M params (d=128; 1 history layer, 2 target-attention and 2 set blocks); 2 epochs ≈ 3e15 FLOPs, 12-20 min shared; recipe Muon+WSD, retr16+gate, K9 aux, cvnd_pre2s inputs. Single members train-only val 0.392-0.3957; pools 0.399 | nn_report.md, nnx_VERDICT.md, CURRENT_STATUS 3n-3p [P] |
| Capacity probes | d=256 (3.0M) 0.37271 vs 0.37385 base; 3 blocks / 8 heads neutral; MoE, TabM null; GRU/DIN/deep "similar alone" | nn_report.md, nnx_VERDICT.md 5, SOLUTIONS_REVIEW 2 [P] |
| Where the loss is | mid-list ordering of familiar candidates; no residual signal in 70 attributes; cold-vs-warm trade-offs only | sol_errors.md, SOLUTIONS_REVIEW 2 [P] |
| Ensemble law | gain over the better member ≈ 0.0186(1-ρ) − 0.378·gap (val, R² 0.95); break-even gap 0.049(1-ρ) | sol_ensemble.md 2.5 [P] |
| Pretrained history encoder | 8M dense params + 40M-row ID table on 463.6M events; zero-shot 0.359. Scaling 2.8M → 8M gave +0.004, v2 → v4 flat. Its incremental value sits in the ID part | pretrain_report.md [P] |
| Local LLM weights (safetensors present) | Qwen3-0.6B (body 440.5M), Qwen3.5-2B (body 1,373.3M + 508.6M emb + 331M vision), Qwen3.5-4B (body 3,570.1M), Qwen3-4B-Instruct-2507 (body 3,633.5M), gemma-3-4b-it | safetensors headers, this session [P] |
| **Not local** (config/tokenizer only) | **Qwen3.5-9B, Qwen3-8B, Qwen3-Embedding-4B, Qwen2.5-0.5B-Instruct**: they would need a download (~18 GB for 9B) | broken safetensors links, this session [P] |
| Qwen3.5 architecture | hybrid: every 4th layer is gated softmax attention (`attn_output_gate`), the rest are Gated DeltaNet (linear attention, conv kernel 4). 2B: 24 layers, d=2048, 6 attention + 18 DeltaNet. 4B: 32 layers, d=2560. 9B: 32 layers, d=4096 | config.json [P] |
| DeltaNet masking | `Qwen3_5GatedDeltaNet.forward` only zeroes padded positions (`apply_mask_to_padding_states`); `create_recurrent_attention_mask` returns None when given a custom 4D mask, so **custom masks are silently ignored in 3/4 of the layers** | transformers 5.16.1 `models/qwen3_5/modeling_qwen3_5.py:187-195, 433-540`, `masking_utils.py:1474-1499` [V] |
| Qwen3 (0.6B, 4B-Instruct-2507, 8B) | pure softmax attention; HF returns a custom 4D mask "as-is" (masking_utils.py:810), so arbitrary masks and explicit `position_ids` work | [V] |
| Compute | a few GPUs; some GPUs never; currently NO GPU for this work | memory file, spec sheets [P][S] |

---

## 1. What can an LLM contribute when there is no text?

| option | mechanism | what it needs here | evidence (section 2) | verdict |
|---|---|---|---|---|
| **(a) LLM as a generic sequence backbone over soft tokens** | Project the item embedding, category, event codes, user embedding and per-candidate dense features into d_model. Run the pretrained trunk with LoRA or a partial fine-tune. Score the candidate tokens with a head | a projector (learned from labels, like the ranker's first Linear), an attention layout that hides other candidates and labels, and the same loss | FPT, Tan et al., Chronos, ORCA, Wikipedia-RL, HLLM user-LLM ablation, E4SRec | **The only live option. Long shot**: the expected effect is a different inductive bias (diversity), not more information |
| **(b) Embedding models (Qwen3-Embedding) / numbers as text** | embed or serialise the features as text | text. **None exists.** Serialising the int8 embedding: ~287 tokens per item → ~76K tokens per feed (287× the soft-token length); one forward pass of Qwen3.5-2B over the train set ≈ 0.7 ZFLOP [inf] | TabLLM (wins only in the very-few-shot regime [V]), Hua et al. (random numeric IDs as text tokens are a poor indexing [S]), Tan et al. | **Dead.** Also, the Qwen3-Embedding-4B weights are not local |
| **(c) LLM as teacher / distillation** | train the large model, distil it into the ranker | (a) first: without semantics, the teacher knows only what it learns from our data | our born-again KD: val +0.00064 (n.s.); KD inside the Muon+WSD combo +0.0002 (nnx_VERDICT 5, CURRENT_STATUS 3g) | **Redundant.** If (a) produces a strong model, blend it directly: test inference is under a minute (section 3). KD only shrinks the ensemble diversity (nnx_VERDICT C5) |
| (d1) LLM backbone for the self-supervised history encoder (pretrain/) | next-feed-open prediction over 463.6M events with a LoRA'd LLM trunk | 1.2e18 FLOPs per epoch for 0.6B (~1.2 h exclusive) [inf] | encoder scaling already saturated (v2 → v4 flat); its incremental value is the ID table, which an LLM does not improve [P] | not worth it |
| (d2) Frozen LLM as a feature extractor (FPT-style: only projector, norms and head train) | cheapest variant of (a) | 4NT per training feed (no recompute) | FPT ≈ a tuned small transformer on toy tasks, and underfits (FPT Table 7) [V] | only as an optional sub-arm of (a) |
| (d3) LLM *architecture* pieces trained from scratch | gated attention, Muon, WSD, MTP-style aux annealing | already done | gate+retr16 +0.0028, Muon+WSD +0.0031 (nnx_VERDICT) [P] | done; this is where "LLM tech" has already paid off |
| (d4) Initialise the small ranker from LLM weights (weight selection) | copy a subset of LLM weights into d=128 blocks | arch/width mismatch (d=128 vs 1024+) | weight-selection gains are reported on small vision datasets [K] | speculative, not recommended |
| (d5) LLMs as coding/research agents | what this project already does | n/a | n/a | the realistic contribution |

---

## 2. Evidence review: what was gained, and does it survive a same-compute baseline trained from scratch?

### 2.1 Non-language inputs fed into pretrained language transformers

**Frozen Pretrained Transformers (Lu, Grover, Abbeel, Mordatch 2021, arXiv 2103.05247)** [V, full text read]
- **Setup.** GPT-2 with self-attention and FFN frozen. Only the input layer, output layer, LayerNorm and positional parameters are trained (≈0.086% of the parameters).
- **Table 1, FPT vs a fully trained transformer vs an LSTM:**

  | model | Bit Memory | XOR | ListOps | MNIST | CIFAR-10 | C10-LRA | Homology |
  |---|---|---|---|---|---|---|---|
  | FPT | 100% | 100% | 38.4% | 98.0% | 72.1% | 38.6% | 12.7% |
  | Full transformer | 100% | 100% | 38% | 99.1% | 70.3% | 42% | 9% |
  | LSTM | 60.9% | 50.1% | 17.1% | 99.5% | 73.6% | 11.7% | 12% |

  So FPT reaches **parity, not superiority**. C10-LRA is worse (38.6 vs 42). The CIFAR-10 FPT number is the 36-layer, 774M model.
- **Table 2, against a frozen *random* transformer:**

  | model | ListOps | MNIST | C10 | C10-LRA | Homology |
  |---|---|---|---|---|---|
  | FPT | 38.4% | 98.0% | 68.2% | 38.6% | 12.7% |
  | Random | 34.3% | 91.7% | 61.7% | 36.1% | 9.3% |

- **Convergence (Table 6):** FPT needs 1-40× fewer steps than the frozen random model.
- **Fine-tuning more of the network (Table 13):** fine-tuning the attention layers as well *diverged* (C10 47.7%, MNIST 89.0%). Fine-tuning the FFNs helped CIFAR-10 (76.6%).
- **Reading:**
  - The language prior beats a frozen random network.
  - Against a properly trained small transformer the result is parity.
  - FPT underfits: on C10-LRA train accuracy 38.5% vs test 38.6%.

**"Are Language Models Actually Useful for Time Series Forecasting?" (Tan, Merrill, Gupta, Althoff, Hartvigsen; NeurIPS 2024 spotlight, arXiv 2406.16964)** [V]
- **Ablations replace the LLM in three LLM forecasters** (none, one attention layer, one transformer block). The ablations win:
  - Time-LLM: 26/26 dataset × metric cases;
  - CALF: 22/26;
  - OneFitsAll: 19/26.
  - Over 40 MAE cases: 35/40, 31/40, 29/40.
- **Cost:**
  - Time-LLM (LLaMA, ~6.65B params) trains for 3,003 min on Weather vs 1.91 min for the LLM-free ablation (181 vs 0.99 min on ETTh1).
  - Inference is 28.2× slower.
- **Pretrained vs random initialisation (CALF, 8 datasets):**

  | configuration | Pre+FT | woPre+FT | Pre+woFT | woPre+woFT |
  |---|---|---|---|---|
  | wins | 3 | **8** | 5 | 0 |

- **Few-shot (10% of data):** no LLM advantage. CALF's GPT-2 won 2/14 cases.
- **Input shuffling:** the LLM methods are no more sensitive to shuffling than the ablations, i.e. no special sequence modelling.
- **This is the closest analogue to our option (a):** numeric tokens projected into an LLM. **The LLM gain does not survive the matched ablation.**

**Chronos (Ansari et al. 2024, arXiv 2403.07815)** [S]
- T5 trained on tokenised series. Random initialisation reached lower training loss. LM initialisation "either overlaps with or slightly underperforms" random initialisation, at every model size.
- A 2025 follow-up, "Random Initialization Can't Catch Up" (arXiv 2506.21570) [S], reports the opposite for univariate T5 forecasting. Its authors flag that the benefit may come from data volume rather than linguistic structure.
- **Net: mixed, near zero.**

**Can Wikipedia help offline RL? (Reid, Yamada, Gu 2022)** [V abstract]
- GPT-2/Wikipedia-pretrained Decision Transformers: "accelerating training by 3-6x" and "consistent performance gains" on Gym/Atari (D4RL-scale, small data).
- Takagi (NeurIPS 2022) [S]: pretrained transformers change their representations a lot; the gains are partly explained by optimisation effects.
- **Gains are mainly convergence speed in small-data settings.**

**ORCA, cross-modal fine-tuning "align then refine" (Shen et al. ICML 2023)** [V abstract]
- State of the art on 3 benchmarks (60+ datasets, 12 modalities), but only after an explicit **embedder-alignment** step. Useful "in data-limited regimes".
- **Transferable lesson:** match the projector's output distribution to the LLM's embedding statistics.

### 2.2 LLMs with ID or embedding tokens in recommendation

**HLLM (ByteDance 2024, arXiv 2409.12740)** [V]
- **Setup.**
  - An item LLM encodes item *text* into an embedding. A user LLM runs over the sequence of item embeddings: no text, "discard[s] the word embeddings … retain[s] all other pre-trained weights".
  - Base models: TinyLlama-1.1B (HLLM-1B) and Baichuan2-7B.
- **Pretrained vs scratch, Pixel200K** (200K users, 3.97M interactions; R@10 / N@10):

  | item LLM | user LLM | R@10 | N@10 |
  |---|---|---|---|
  | scratch | scratch | 5.063 | 2.755 |
  | scratch | pretrained | 5.416 | 2.969 |
  | pretrained | scratch | 5.331 | 2.940 |
  | pretrained | pretrained | 5.581 | 3.100 |

  - Pretrained user-LLM weights on embedding (non-text) inputs: **+5-7% R@10**.
  - This is the single most relevant positive data point for option (a). But it is at a small data scale (4M interactions vs our ~250M candidate rows), and the inputs come from a jointly trained text model.
- **Frozen user LLM:** R@10 2.470, much worse.
- **Main results (R@10):**
  - Pixel8M: HLLM-1B 6.129 vs SASRec 5.083, SASRec-1B 5.142, HSTU 4.848.
  - Amazon Books: 6.97 vs HSTU-large 4.78.
  - Not compute-matched: HLLM trained 5 epochs vs 50-200 for the baselines; the † rows use a different negative count.
  - Scaling SASRec from scratch to 1B gave almost nothing (5.083 → 5.142). The **HLLM gain comes with text content**.
- **Online:** +0.705%.

**E4SRec (arXiv 2312.02443)** [V]
- **Setup.** Frozen LLaMA2-13B + LoRA r=16 (gate/up/down). ID embeddings from a pretrained SASRec go through a linear projection, inside an Alpaca text prompt. Scored over all items.
- **Reported gain:** "about 12% on HR@k and 21% on nDCG@k" over the best baseline on Beauty/Sports/Toys/Yelp (each 0.17-0.32M actions).
- **But:**
  - The SASRec baseline is the BCE version (Beauty nDCG@10 0.0258). Klenitskiy & Vasilev (RecSys'23) show that CE-trained SASRec reverses such rankings [S].
  - The "LLM-free" ablation is a linear-only model ("extremely poor", given only as a figure), not a same-compute transformer.
  - BERT-base in place of LLaMA is worse, with no numbers in the text.
- **Not a matched-baseline result.**

**LLaRA (SIGIR'24, arXiv 2312.02445) and CoLLM (arXiv 2310.19488)** [V abstracts; details K]
- Hybrid prompts: item *text* + a projected SASRec/MF embedding. LLaRA uses a curriculum from text-only to hybrid prompts.
- Evaluation: candidate selection (LLaRA: HitRatio@1 among ~20 candidates on MovieLens/Steam/LastFM [K]) or CTR AUC (CoLLM [K]) on small datasets.
- **They depend on text; they are not transferable to us.**

**"Lost in Sequence: Do LLMs Understand Sequential Recommendation?" (Kim et al., KDD'25, arXiv 2502.13909)** [S]
- LLM4Rec models trained or evaluated on *shuffled* sequences perform about as well as on ordered ones; SASRec degrades significantly. "Existing LLM4Rec models do not fully capture sequential information."
- **Relevance:** the order and recency structure that our target attention exploits is exactly what LLM4Rec models were found not to use.

**Scale trained from scratch in industrial recommenders**
- **HSTU (Meta, ICML'24, arXiv 2402.17152)** [V abstract]: a new architecture with pointwise SiLU attention, so no LLM weights can initialise it [K]. Quality "scales as a power-law of training compute across three orders of magnitude" up to 1.5T parameters; +12.4% online.
- **OneRec-V1** (Kuaishou) [K, see nnx_research_recsys_arch.md]: trained from scratch.
- **OpenOneRec (arXiv 2512.24762)** [S]: Qwen3-1.7B/8B *initialised* models, but with 16-130B tokens of itemic-*text* alignment and text co-training. It needs item text.

### 2.3 Competitions on anonymised data
- **RecSys Challenge 2025** (Synerise; 170M anonymised events with quantised text embeddings) [S].
  - Reported top solutions: 2nd an event-sequence autoencoder + GRU ensemble; 4th (Yandex) a heterogeneous causal transformer trained from scratch, plus a GNN, DCN and engineered features.
  - No reported top solution used a pretrained LLM backbone. Absence of evidence, not proof.
- **OTTO** (Kaggle 2023, anonymised sessions) [K]: co-visitation candidates + GBDT and from-scratch NN rankers.
- **RecSys 2023 (ShareChat)** [K]: anonymised features; GBDT/DNN.
- **Our own reviews** (deep_recsys_engagement.md) [P]: even with real text, BERT features gave no gain in NVIDIA's RecSys 2020 XGBoost.

### 2.4 Synthesis

| study | inputs | claimed LLM gain | same-compute from-scratch baseline? | survives? |
|---|---|---|---|---|
| FPT | toy bits, pixels, protein tokens | ≈ parity with a tuned transformer; +2.5-6.5 pts vs a frozen random one (+24 on Bit Memory) | yes (tuned depth) | **parity only** |
| Tan et al. | time-series patches | papers claim SOTA | yes (3 ablations) | **no**: ablations win 19-26/26 |
| Chronos (+ 2025 follow-up) | tokenised series | none / small | yes (random init) | **no / mixed** |
| Wikipedia-RL | RL trajectories, small data | 3-6× faster convergence | partly | speed only, small data |
| ORCA | 12 modalities, mostly small | SOTA with alignment | vs task-specific | partly; data-limited regimes |
| HLLM | item **text** → embeddings → user LLM | +5-7% from a pretrained user LLM (Pixel200K) | no (5 vs 50-200 epochs) | small-data, text-driven |
| E4SRec | IDs via SASRec embeddings + text prompt | +12% HR / +21% nDCG | no (BCE-SASRec, linear ablation) | **not established** |
| LLaRA / CoLLM | text + CF embeddings | various | no | text-dependent |
| Lost in Sequence | — | — | — | LLM4Rec ignores sequence order |
| HSTU / OneRec-V1 | IDs and actions, from scratch | scaling with compute | n/a | from scratch wins at scale |

**Conclusion [inf].**
- No published result shows a pretrained LLM beating a tuned, same-compute model trained from scratch on purely numeric or ID tokens at our data scale (hundreds of millions of rows).
- The positive regime is small data, text-bearing items, or weak baselines.
- At 3.15M feeds we are outside the low-data regime, but the matched 150K test below sits closer to it. That is why it needs a second data point before any extrapolation (rule A-B in 4.6).

---

## 3. Compute math (exclusive GPU; `nn_x/llm/research_compute_math.py`)

**Assumptions.**
- Training FLOPs per token:
  - LoRA + full activation checkpointing ≈ 6N: 2N forward, 2N backward to activations (frozen weights get no weight gradient) and 2N recompute;
  - 4N without checkpointing;
  - 8N for a full fine-tune with checkpointing.
- N = body (non-embedding) parameters. Embedding tables, the LM head, the vision tower and the MTP head are dropped, because we use `inputs_embeds` and our own head.
- +5% for attention, the gated delta rule and LoRA matmuls [inf]. For example, Qwen3-0.6B masked attention is ~3% of 2NT, and Qwen3.5-2B DeltaNet ~2.6%.
- Padding efficiency 0.85.
- MFU 15% for 0.6B (d=1024, launch-bound) and 25% for 2B-9B, on a dense BF16 peak of the vendor's peak rate. Pessimistic case = half the MFU.
- **These MFUs are planning assumptions, not measurements.**

| model | body params | TFLOP / train feed (T=264.5) | × ranker FLOPs/feed | 100K feeds | 1M feeds | 3.15M (1 epoch) | pessimistic 1 epoch |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.44B | 0.73 | 1.7K | 4.3 min | 43 min | **2.2 h** | 4.5 h |
| Qwen3.5-2B | 1.37B | 2.29 | 5.4K | 8.0 min | 80 min | **4.2 h** | 8.4 h |
| Qwen3.5-4B | 3.57B | 5.95 | 14K | 21 min | 3.5 h | **11 h** | 22 h |
| Qwen3-4B-Instruct-2507 | 3.63B | 6.05 | 14K | 21 min | 3.5 h | 11 h | 22 h |
| Qwen3.5-9B (not local) | 6.92B | 11.5 | 27K | 40 min | 6.7 h | **21 h** | 42 h |

The ranker reference is ~1.8 MFLOP forward per candidate × 3 × 78 candidates ≈ 0.42 GFLOP per train feed. A 10-epoch ranker member is ≈1.3e16 FLOPs in total. One epoch of Qwen3-0.6B is ≈2.3e18 (~180×), and one epoch of 9B ≈3.6e19 (~2,800×).

**Inference** (2NT, fp32 head):

| model | val (14.4K) | test (28.6K) | holdout (30K, T=331) |
|---|---|---|---|
| 0.6B | 13 s | 25 s | 32 s |
| 2B | 24 s | 48 s | 60 s |
| 4B | 61 s | 2.1 min | 2.6 min |
| 9B | 2.0 min | 4.0 min | 5.0 min |

- Holdout evaluation every ~250 steps therefore costs 5-15% extra for 0.6B-2B.
- Test-time augmentation over K candidate permutations, needed only for order-dependent layouts, multiplies inference by K.

**Memory with checkpointing** (bf16 weights + stored layer inputs + one-layer recompute working set, ×1.3 slack):

| model | weights | per feed | 128-feed batch total |
|---|---|---|---|
| 0.6B | 0.9 GB | 27 MB | ~6 GB |
| 2B | 2.7 GB | 49 MB | ~12 GB |
| 4B | 7.1 GB | 74 MB | ~22 GB |
| 9B | 13.8 GB | 115 MB | ~37 GB |

- Feed length goes up to 557 tokens (2.1× the mean), so batches should be capped by a **token budget** (e.g. 48-64K tokens), not by feed count.
- If `nn/data.SplitData` is reused, it adds ~21 GB of GPU memory for the train split.
- Without checkpointing (4NT, ~1.5× faster), 0.6B needs ~0.3 GB per feed, i.e. ~40 GB per 128 feeds. That is fine on a free GPU but not on a shared one.
- LoRA r=32 on all 7 linear layers of Qwen3-0.6B ≈ 20M trainable parameters (+ ~3M projector/head). Adam state < 0.5 GB.

**CPU** (smoke tests only):
- At an estimated 0.2-0.5 TFLOP/s with 12 threads [inf], Qwen3-0.6B takes ~0.5-1 s per feed forward and ~2-4 s forward+backward; Qwen3.5-2B ~2-4 s and ~6-12 s.
- So a few dozen feeds are feasible within the CPU rules.

---

## 4. Minimal decisive experiment: does an LLM backbone beat our ranker at MATCHED data?

### 4.1 Arms (all train-only: no val labels; checkpoints selected on the standard 30K random-like holdout)

| arm | what | seeds | why |
|---|---|---|---|
| **B (baseline)** | current ranker recipe via the unmodified `nn/train.py` with `--train_feeds 150000` | 23, 24 | matched-data reference |
| **L (LLM)** | Qwen3-0.6B pretrained, soft tokens, layout L1, LoRA r=32 + projector/norm/head | 23, 24 (same subsets as B) | the question |
| **R (control)** | the same Qwen3-0.6B architecture, **random init**, all weights trained | 23 | Tan et al.-style attribution: is any gain language pretraining or just capacity? |
| F (optional) | the L arm with LoRA off (frozen trunk, FPT-style) | 23 | information only |

### 4.2 Data (identical across arms; no code change to nn/)
- `nn/train.py` builds its subset as `feeds = np.sort(np.random.default_rng(seed).choice(3151208, 150000, replace=False))`. It then removes the fixed holdout (`p_random>0.9 & M>=50`, `default_rng(12345)`, 30K feeds); see `nn/train.py:150-167`.
- The LLM loader must reproduce exactly this list per seed.
- Measured subset sizes:
  - seed 23: 148,600 feeds (11.51M candidate rows, 6,507 with p_random>0.5, 2,950 holdout-eligible random-like);
  - seed 24: 148,557 feeds.
- Each seed is a different subset, and both arms share it per seed.
- "Including random-like feeds": a uniform subset holds them at the natural ~4.4% rate. A 120K production + 30K exploration stratified subset would need a *copied* trainer with a `--feed_list` flag under `nn_x/llm/` (nn/ must not be edited). Keep it uniform for the decisive test.
- **Inputs: identical to the baseline.**
  - Import `nn/data.py` (`SplitData`, `attach_xt`) and `nn/model.py` (`dense_features`) **read-only** from `nn_x/llm/`. Neither imports `nn/common.py`, whose GPU guard raises when `CUDA_VISIBLE_DEVICES=""`.
  - Use the same `--xt cvnd_pre2s` preset, `--user0 --domain`.
  - Never use train-row `pre_*`, `*_id_contrib`, `x_lh_*`, non-LUO `it_hist_*` or the quarantined hard links (CURRENT_STATUS 2, 3f, 3k-P6).

### 4.3 LLM arm architecture

**Tokens.** All inputs are soft tokens fed via `inputs_embeds`; no vocabulary tokens are used.
- **User token** = MLP(z-scored user_emb int8[128] ‖ is_user0 ‖ domain one-hot).
- **History token j** = MLP(L2-normed item emb ‖ missing flag) + E_cat + E_event_type + E_event_source + E_recency(j). History is ordered **old → new**, so the newest events sit next to the candidates.
- **Candidate token c** is the ranker's own construction with output dimension d_model: Linear([e_c, e_c ⊙ u]) + E_cat + MLP(the 61 dense features + cvnd_pre2s extras) + missing vector.
- **Projector initialisation (ORCA's "align" lesson):** zero-mean, with the output RMS matched to the RMS of Qwen's `embed_tokens` rows. Measure it once from the checkpoint.

**Layout L1 (recommended).** Sequence = [user, h_1 … h_H, c_1 … c_M].
- **Mask:** user + history are causal. **Candidate c attends to user + all history + itself only.**
- **position_ids:** user 0, history 1..H, and **every candidate H+1**. Each candidate is then computed exactly as if it were the single next token after the history. The model is permutation-invariant over candidates by construction, and the M "continuations" are scored in one pass.
- Candidate tokens see no labels and no other candidate. Targets are never inputs.
- **Implementation:** HF Qwen3 accepts a custom 4D float or bool mask as-is. For batches, flex-attention with a block mask (document × candidate rule) avoids T² padding waste.
- **Variant L2 (only if L1 passes):** the candidate block is additionally bidirectional over candidates. With shared position ids it stays permutation-invariant. It is the set-attention analogue, which was neutral in the ranker.

**Qwen3.5 (2B/4B/9B), if escalated.** A custom mask is ignored in the DeltaNet layers, which would see candidates causally in storage order. Two options:
- **(i) "Branch" implementation, exact and permutation-invariant.**
  - Run [user, history] once and take each DeltaNet layer's final state S_H plus the last 3 conv inputs.
  - Each candidate is one recurrent step from S_H. With g, β, k, q, v from the candidate token and its conv window [x_{H-2}, x_{H-1}, x_H, x_c]:
    `o_c = α_c (S_H q_c − β_c (S_H k_c)(k_cᵀ q_c)) + β_c v_c (k_cᵀ q_c)`
  - That is two batched matvecs per head per layer. The full-attention layers use the L1 mask.
  - About 60-100 lines of pure torch. Gradients flow through S_H. Whether the fla kernel supports a backward pass through the final state must be checked; the torch fallback does.
- **(ii) Zero-code fallback:** candidates appended causally in a fresh random order every step, with test-time augmentation over K=4 permutations. This is order-dependent and noisier, and costs 4× inference.
- **Prefer the Qwen3 family** (Qwen3-4B-Instruct-2507 is local and pure attention) for the size step.

**Head.**
- score_c = MLP(RMSNorm(h_c^last)) + a linear "wide" term on the standardised dense features (as in the ranker).
- K9 aux head = a second MLP on h_c (listwise production-order distillation, τ 10).
- **Both computed in fp32.** bf16 scores create within-feed ties that the pessimistic-tie metric punishes; see nn_report's precision note: 0.37218 vs 0.37408.

**Trainable parameters.**
- LoRA r=32, α=32, dropout 0.05 on q, k, v, o, gate, up and down in all layers (B=0 init, so the trunk starts exactly pretrained).
- The projectors, the head and all RMSNorm gains are trained too; FPT finds LayerNorm the most important parameter to fine-tune [V].

**Loss** (exactly the ranker's):
- listwise softmax CE per feed against y/Σy, with equal per-feed weight (`--loss_norm feed`);
- plus K9 aux with weight 0.5·(1-p_random) (`--aux_pos 0.5 --aux_mode listwise --aux_tau 10`).

**Optimiser.**
- AdamW: LoRA lr 2e-4 (wd 0), projectors/head lr 1e-3 (wd 0.02).
- Warmup 100 steps, cosine to 10%, grad clip 1.0, bf16 autocast.
- Token-budget batches of ~48-64K tokens (~180-240 feeds).
- **2 epochs** (≈1.3-1.7K steps). Evaluate the holdout every 200 steps and keep the holdout-best checkpoint.
- **Control R:** full AdamW, lr 5e-4, warmup 200, otherwise identical.

### 4.4 Baseline arm (exact command; no code change)
```
cd <repo>/nn && CUDA_VISIBLE_DEVICES=<4..7 as allocated> OMP_NUM_THREADS=12 ../.venv/bin/python train.py \
  --variant llmcmp_base_s{23,24} --train_feeds 150000 --seed {23,24} \
  --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout \
  --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 \
  --attn_gate --lr 2e-3 --cooldown_mix 0.1 --epochs 20 --eval_every 250
```
- These are the F7 stage-1 flags copied from `nn/run_final7b.sh`. F7 is the strongest family, at lr 2e-3. Only `--train_feeds`, `--epochs`, `--eval_every` and the variant name change, and `--save_steps` is dropped.
- At ~400 steps per epoch this is ~8K steps; the warmup of 500 steps is ≈1.25 epochs.
- **Matched data, not matched epochs:** the 1M-parameter ranker is allowed many passes. Each arm keeps its holdout-best checkpoint, and both arms are selected with a similar number of evaluations (20-35).
- `--cooldown_mix 0.1` draws only from exploration feeds inside the subset (~6.5K).

### 4.5 Evaluation (all on train-only predictions; `src/metrics.py`; `nn/pcmp.py` slices)
- Val NDCG, holdout NDCG, and paired per-feed deltas L−B pooled over the two seeds (mean over seeds of the per-feed delta, SE = sd/√n).
- Per-seed deltas.
- Slices: M<50 / M≥50, Tue / Thu, all-cold-positive feeds, late holdout.
- Within-feed z correlation ρ(L, B).
- The equal z-mix eq(L_s, B_s) and the reference eq(B_23, B_24).
- Control: R vs L and R vs B.

### 4.6 Pre-declared acceptance and stop rules (declare in CURRENT_STATUS before the first look)

- **A1 (LLM beats the ranker at matched data):** pooled paired train-only val delta L−B **> 2 SE**, **both seeds > 0**, **and holdout delta ≥ 0**. No slice may be below −2 SE.

- **A2 (useful ensemble member at matched data):** both parts must hold.
  - (i) **Strength:** L ≥ B − 0.002 on val **and** on holdout. This is the matched-data analogue of gate (a), "within 0.002 of our best single member".
  - (ii) **Diversity pays:** the mean over seeds of eq(L_s, B_s) − eq(B_23, B_24) > 0 on val with z ≥ 1.5, and holdout ≥ 0.
    - So the LLM must be a better second member than another ranker seed.
    - This is conservative: the B-B pair sees two different subsets, i.e. more data diversity.
  - Also report the law-predicted gain 0.0186(1-ρ) − 0.378·|gap|. If (ii) fails while the law predicts a gain, believe the measurement.

- **Attribution (information, not a gate):** if L passes A1 or A2 but L − R ≤ 1 SE, the gain comes from capacity or architecture, not language pretraining. Then a small from-scratch family is the cheaper way to get it.

- **Decision tree:**
  - **A1 and A2 both fail → STOP.** Add "LLM backbones (soft-token, Qwen3-0.6B, matched 150K)" to the do-not list.
    - Exception: if L ≥ B − 1 SE (not clearly worse) **and** L − R > 2 SE (pretraining clearly helps), allow **one** size probe: Qwen3-4B-Instruct-2507, layout L1, 150K, seed 23, ~1.1 h exclusive. FPT Table 8 shows the frozen-LM benefit growing with size (68.2 → 72.1 from 117M to 774M).
    - Apply A1 and A2 to the probe; stop if it fails.
  - **A1 passes → Stage B (data slope) before any full-data run.**
    - Train L and B on 600K feeds (seed 23; L 1 epoch ≈ 26 min for 0.6B, B the same recipe with fewer epochs).
    - Continue only if the L−B advantage at 600K is ≥ 50% of the advantage at 150K. Pretrained priors typically shrink with data.
  - **A2 passes (A1 may fail), or Stage B passes → Stage C (full-data member).**
    - Qwen3-0.6B L1, 1 epoch over all train feeds (≈2.2 h, pessimistic 4.5 h), 2 seeds.
    - Gate (a): the single-member train-only mean is within 0.002 of the best family on val **and** holdout.
    - Then the remaining 4 seeds (≈9-18 h) for gate (b), the family 6-seed z-mean within 0.002.
    - Then the frozen protocol: candidate pools with/without the LLM family, chosen on the holdout; replace a pick only with val > 2 SE, no slice < −2 SE, and `src/presub_gate.py` PASS.
  - **Integration cost before Stage C:**
    - The pool uses 0.5 FT + 0.5 TO branches.
    - An LLM member needs its own 5-fold GroupKFold val fine-tune with the n2cpv3 extras branch. With LoRA this is ~11.5K × 4 epochs × 5 folds ≈ 230K feed-passes ≈ 10-15 min for 0.6B.
    - Or the member enters the TO branch only. Decide this before Stage C.

### 4.7 Cost (exclusive-GPU estimates; shared GPUs are slower)

| item | estimate |
|---|---|
| L arm, 0.6B: 2 seeds × 2 epochs × 148.6K feeds (≈594K feed-passes × 0.73 TFLOP) | ≈25 min (pessimistic 50) |
| L holdout evaluations (~8 per run × 32 s) + val/test inference | ≈10 min |
| R control (8NT with checkpointing, or 6NT without; 1 seed × 2 epochs) | ≈15 min |
| B baseline: 2 seeds × ~3M feed-passes of a 1M-param ranker | ≈15-25 min |
| **Stage A total** | **≈1.1-1.5 GPU-h; pessimistic ≈3 GPU-h** |
| Optional 4B probe (1 seed, 150K × 2 epochs) | ≈1.1 h (pessimistic 2.2) |
| Stage B (600K, 1 seed) | ≈0.7 h |
| Stage C (2 seeds, full epoch), then 4 more seeds | 4.4 h (9 h), then +9 h (18 h) |
| Engineering (loader reuse, layout, LoRA, head, evaluation, smoke tests) | ~1 agent-day |

### 4.8 Implementation plan (new code only under `nn_x/llm/`; scratch under `…/scratchpad/llm/<label>/`)

**Environment.**
- `.venv_llm`, as specified by the lead (uv venv with the same Python, plus a `.pth` pointing at `.venv`'s site-packages placed after its own, plus `uv pip install transformers peft accelerate` into `.venv_llm` only).
- Verify `torch.__version__` = 2.11 and an empty before/after `uv pip freeze` diff of `.venv`.
- transformers ≥ 5.x is needed for `qwen3_5`; Qwen3 works with older versions too.
- LoRA can be hand-rolled (~30 lines) if peft causes trouble.

**Files.**
- `nn_x/llm/data_llm.py`: subset reproduction plus a wrapper over `nn/data.SplitData` / `dense_features`.
- `nn_x/llm/model_llm.py`: projectors, the L1 mask and position ids (flex block mask on GPU, dense bool mask on CPU), LoRA, heads.
- `nn_x/llm/train_llm.py`: loss, optimiser, holdout selection, prediction writer to `nn_x/llm/preds/`, aligned with `data/flat/<split>/cand_*`.
- `nn_x/llm/compare.py`: the A1/A2 computation.

**CPU smoke tests** (CUDA_VISIBLE_DEVICES="", ≤12 threads, ~32 train feeds):
1. **Permutation invariance:** shuffling the candidates changes each candidate's score by < 1e-4 (fp32).
2. **Equivalence:** a candidate's score in the batched L1 pass equals its score from a separate forward of [user, history, c] alone. This validates the mask and the shared position id.
3. **No leakage:** the input tensors contain no target, no `pos`, and no banned feature names.
4. **One optimiser step lowers the loss on the tiny batch**; LoRA B starts at 0; head outputs are fp32.
5. Measured CPU seconds per feed (sanity check of section 3).
6. For Qwen3.5 only: confirm that a custom 4D mask is ignored by the DeltaNet layers. The forward pass gives different candidate scores under candidate permutation, so the branch implementation is mandatory.

**GPU rules for the later run:**
- Only after the lead allocates it; some GPUs only, never 0-3.
- At most 4 training processes per GPU.
- Plan ~35-45 GB for a 0.6B job with SplitData on the GPU.
- Detached jobs with PIDs in `nn/logs/RUNNING.txt` conventions, but log under `nn_x/llm/logs/`.

### 4.9 Pitfalls to pre-empt
- **Ties:** compute the head in fp32 and write fp32 predictions.
- **Order leaks:**
  - Train storage order is random (measured), but any order-dependent layout (Qwen3.5 option ii) must still reshuffle every step. `pos` must never be an input.
  - The shared-position L1 layout is order-free by construction.
- **Feature parity:** reuse the exact presets. Anything new is hard-linked under a *new* name (FEATURES_REVIEW rule). Never patch `attach_xt` in place.
- **Selection optimism:** the holdout picks checkpoints in both arms. Count val z, not holdout z, as evidence (SOLUTIONS_REVIEW 1.2).
- **LoRA rank and learning rate are not tuned.** Allow at most one pre-declared fallback (r=64, lr 1e-4) if the first run diverges. No search on val.
- **Embedding statistics:** a projector that outputs vectors 10-100× larger than Qwen token embeddings will saturate RMSNorm and wash out the prior. Match the RMS (4.3).

---

## 5. Verdict and reasons

**LONG SHOT.** Run Stage A once when a GPU lane is free. Do not pre-empt family training, and stop at the first failed gate.

Reasons, in order of weight:
1. **No new information** [P].
   - The LLM cannot decode anonymised IDs or embeddings. Its only asset is a generic computation prior.
   - Our losses are information-limited: no residual signal, capacity probes null, and every alternative trunk lands at the same strength.
2. **Well-controlled evidence on non-language inputs is null at matched compute** (Tan et al., Chronos), or positive only at small data, with text, or against weak baselines (FPT, HLLM, E4SRec, ORCA). LLM4Rec models ignore sequence order (Lost in Sequence).
3. **The usefulness bar is high** [P]. A member must get within ~0.002 of 0.3946 train-only val at ρ ≈ 0.96, or ~0.005 at ρ ≈ 0.90. The stronger the LLM gets, the more it tends to correlate with the ranker on identical inputs.
4. **Cost** [inf]. Per feed, 1.7K-27K× the ranker's FLOPs. Full-data members cost 2.2-21 GPU-h each (pessimistic 2×) against ~4-5 h for a ranker family member, and GPUs are fully committed.
5. **Why not "not worth it":**
   - Stage A is cheap (≈1.5-3 GPU-h + 1 agent-day). Its result is decisive and reusable: it settles the user's question with project data.
   - A pretrained, differently biased trunk is one of the few remaining sources of *diversity at equal information*, and diversity is the one lever that still pays (sol_ensemble, the SOLUTIONS_REVIEW bottom line).
   - The deadline (2026-12-09) leaves time for Stage B/C if Stage A surprises.

**What would change the verdict:**
- **Up:** A1 passes and Stage B shows the advantage holding at 600K → promising, go to full data. A2 passes with ρ ≤ 0.93 → a full-data diversity member is worth about one family's GPU budget.
- **Down to "not worth it":** L < B − 2 SE and L − R ≤ 0 (pretraining contributes nothing at this scale) → close the line, including the 4B/9B variants.

---

## Sources
- Lu, Grover, Abbeel, Mordatch. Pretrained Transformers as Universal Computation Engines, arXiv 2103.05247 [V full text]. https://arxiv.org/abs/2103.05247
- Tan, Merrill, Gupta, Althoff, Hartvigsen. Are Language Models Actually Useful for Time Series Forecasting? NeurIPS 2024, arXiv 2406.16964 [V]. https://arxiv.org/html/2406.16964v2
- Ansari et al. Chronos: Learning the Language of Time Series, arXiv 2403.07815 [S]. https://arxiv.org/abs/2403.07815 ; follow-up "Random Initialization Can't Catch Up", arXiv 2506.21570 [S]. https://arxiv.org/abs/2506.21570
- Reid, Yamada, Gu. Can Wikipedia Help Offline Reinforcement Learning? arXiv 2201.12122 [V abstract]. https://arxiv.org/abs/2201.12122 ; Takagi, NeurIPS 2022, arXiv 2211.09817 [S]. https://arxiv.org/abs/2211.09817
- Shen et al. Cross-Modal Fine-Tuning: Align then Refine (ORCA), arXiv 2302.05738 [V abstract]. https://arxiv.org/abs/2302.05738
- HLLM (ByteDance), arXiv 2409.12740 [V]. https://arxiv.org/html/2409.12740
- E4SRec, arXiv 2312.02443 [V]. https://arxiv.org/html/2312.02443v1
- LLaRA, arXiv 2312.02445 [V abstract]. https://arxiv.org/abs/2312.02445 ; CoLLM, arXiv 2310.19488 [V abstract]. https://arxiv.org/abs/2310.19488
- Kim et al. Lost in Sequence: Do LLMs Understand Sequential Recommendation? KDD'25, arXiv 2502.13909 [S]. https://arxiv.org/abs/2502.13909
- Klenitskiy & Vasilev. Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec? arXiv 2309.07602 [S]. https://arxiv.org/abs/2309.07602
- Zhai et al. HSTU: Actions Speak Louder than Words, arXiv 2402.17152 [V abstract]. https://arxiv.org/abs/2402.17152
- OpenOneRec Technical Report, arXiv 2512.24762 [S]. https://arxiv.org/abs/2512.24762
- Hegselmann et al. TabLLM, arXiv 2210.10723 [V abstract]. https://arxiv.org/abs/2210.10723 ; Hua et al. How to Index Item IDs for Recommendation Foundation Models, arXiv 2305.06569 [S]. https://arxiv.org/abs/2305.06569
- RecSys Challenge 2025: 4th place (arXiv 2508.06970), 2nd place autoencoder (arXiv 2508.07748) [S]. https://arxiv.org/abs/2508.06970 , https://arxiv.org/abs/2508.07748
- GPU BF16 dense ~the vendor's peak rate: SemiAnalysis InferenceX chip page, flopper spec sheet [S]. https://inferencex.semianalysis.com/chips/b300 , https://flopper.io/system/nvidia-hgx-b300-8-gpu
- transformers 5.16.1 source (Qwen3.5 GatedDeltaNet masking, custom 4D masks), read locally [V]: `models/qwen3_5/modeling_qwen3_5.py`, `masking_utils.py`.
- Project: CURRENT_STATUS.md, SOLUTIONS_REVIEW.md, nnx_VERDICT.md, pretrain_report.md, sol_errors.md, sol_ensemble.md, nn_report.md, nnx_research_llm_arch.md, nnx_research_recsys_arch.md, deep_recsys_engagement.md [P].
