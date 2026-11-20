# PRETRAIN_PLAN_1113: pre-training on trajectories / embeddings / time series

2026-11-13. Finaliser report of the pre-training study, built from one audit (assets, tests, data facts, pitfalls), three research lenses (seq-ssl, emb-ts, competitions), one plan and two independent critiques (information lens, protocol/cost lens). No training was run. Audit scripts and logs: `nn_x/pretrain2/audit/{a1_items,a2_users_time,a3_pos_corpus,a4_extension,a5_item_ts}.{py,log}`.

The user's question: can we pre-train on trajectories, embeddings or time series, use existing pre-trained models, and get models that separate trajectories on their own, without the NDCG task at first?

**Bottom line.**
- Two label-free trajectory models already feed every G3 member:
  - Avito's production user tower, as `user_emb` (InfoNCE, about 10B samples, also reads search queries);
  - our own pre-trained sequence encoder, as the 6 `pre2s` columns plus 3 relational copies.
- The current measurement (interim `xdrop_pre2`) says the whole pre2 channel is worth about **+0.0012 val / +0.0006 holdout** inside the ranker.
- We are outside every regime where pre-training beat a supervised ranker in the literature. Our unlabelled corpus is about 1:1 with the labelled feeds' own histories, the ranker has no ID table that could overfit, and the ranker sees the same inputs.
- Both critics refuted all six planned experiments on the generational bar (>= +0.002 val, within-feed corr < 0.99).
- What survives is **4 CPU-only closure diagnostics** with tightened gates, plus **2 GPU items behind explicit reopen triggers**.
- Planned GPU spend: 0 GPU-h. A full reopen chain would cost at most about 10 GPU-h.
- **Status 11-13 (§7): the track is closed.** All four CPU closures ran (about 30 CPU-min in total, 12 threads, no scale-down) and all four are CLOSED by their pre-declared gates: P1 0/8 arms, P2 val -0.00034 ± 0.00033, P3 val -0.00031 ± 0.00017, P4 0/4 arms, with none of them positive on val. E1 and F8 are CLOSED for good: the final xdrop_pre2 is +0.00128 ± 0.00061 val, below the +0.0015 trigger, and the P3 cell proxy failed. No reopen trigger fired. GPU spent: 0 h.

---

## 1. What already exists and was measured

### 1.1 Assets on disk

| asset | content | how it reaches the ranker |
|---|---|---|
| Corpus `data/pretrain/` | 2,588,072 sequences / 463,552,406 tokens:<br>- train 456.1M (latest train snapshot per non-zero user + every user-0 feed);<br>- val 2.49M; test 4.93M (earliest val/test snapshot per user).<br>Targets: `tgt_fe` (next new feed-open, et1&es1, valid 95.2%), `tgt_new` (99.0%), `tgt_fe_multi` (next 4). | - |
| Content tables | Use `item_tab2` / `item_has2` only (83.3M items). `item_tab` carries the content-membership leak. | - |
| Checkpoints `data/pretrain/ckpt_*.pt` (~[n] GB, fp32, no optimiser state) | All share the SeqEncoder: d=384, 4 pre-LN RoPE blocks, 8.0M dense params, plus an ID table.<br>- v3: id 40.25M x 64<br>- v4: x 128 + id_bias<br>- v3A / v4A: val side, no test snapshots, 39.92M rows<br>- w{1,2,3}r_{v3,v4}: train windows, e.g. w3r_v4 has 41.73M x 128 | only as scalars |
| Scores `data/feats/*/pre2_*.npy` | train-stage inputs:<br>- pre2_fe_z, pre2_new_z, pre2_fe_s, pre2_new_s, pre2_hmax32, has:pre2_fe;<br>- XT_REL copies of fe_s, new_s, hmax32.<br>Train NaN outside windows W1-W3 (Sep 6-8 uncovered; 72.1% of train candidate rows covered). | every G3 member (`cvnd_pre2s`) |
| FT-only extras | the `pre_*` 9-column family (n2c_pre_v3: fe_noid, idmax32, id_contrib_v2, ...) | val fine-tune half only |
| Organisers' `user_emb` | int8 128, **one vector per feed** (train 3.15M / val 14,423 / test 28,595). This is Avito's real-time two-tower user tower (https://habr.com/ru/articles/1004694). | user token + cos_u / dot_u + Linear([c, c*u]) |
| CF / cluster / Markov | ALS f64/f128, item2vec d128 (`eda/collab_stats/out/*_emb_sub.npz`); k-means codes 256/1024/4096 (`data/clu_work`); daily transition matrices (`data/seq_work/trans_*`). | tested as residuals (§1.3) |
| External models in the local cache | LLMs only, plus Qwen3-Embedding-4B and MiniLM.<br>No time-series foundation model, no recsys foundation checkpoint. | - |

The encoder's weights, ID table and hidden states have never entered the ranker. Only scalars have.

### 1.2 Zero-shot and feature-level results (val, 14,423 feeds)

Sources: `reports/pretrain_report.md`, `nn_x/r3/stage0_gate.log`.

**Zero-shot NDCG**

| score | NDCG |
|---|---|
| random | 0.2882 |
| f1 | 0.3279 |
| ALS | 0.331 |
| cos_u | 0.3417 |
| maxsim32 | 0.3478 |
| pre2 fe (v1 → v4s1) | 0.3537 → 0.3585 |
| pre2 new | 0.3547-0.3588 |
| fe_noid | 0.3496-0.3543 (the ID part is worth 0.004-0.005) |
| idmax32 | 0.294-0.297 |
| pre_final (clean v3A+v4A) | 0.3590 |
| ENS_all+h | 0.3625 |
| G3 | 0.40032 |

- A label-free encoder recovers about 64% of G3's lift over random.
- Every variant (ID lr, ID bias, ID dim 128, dense next-4) lies within the 0.00025 noise band.
- A 330-parameter DIN-style similarity reaches 0.3584 (`reports/eda_embeddings.md`).

**Encoder scores as features**

| route | result |
|---|---|
| LGB harness | +0.0037 ± 0.0008 |
| residual stacker | +0.00075 ± 0.00044 |
| z-blend | +0.00082 ± 0.00038 |
| encoder blend term in later pools | CLOSED (-0.0008) |

**Fine-tuning the encoder on val labels (5-fold listwise)**
- OOF 0.3536 → 0.350-0.352, i.e. worse.

**Train-stage features**

| version | result |
|---|---|
| leaky content table | train-fit val -0.0090 ± 0.0013, holdout +0.0106 (§8b leak) |
| prew | +0.0031..+0.0037 |
| pre2, LGB gate | val +0.00535 ± 0.00086, holdout +0.00522 |
| pre2 in the NN, 3-epoch pair | +0.0019 val (z 3.3), +0.0002 holdout |
| pre2 in the NN, 10 epochs (F9 vs F6) | +0.0009 val (M<50 +0.0020, M>=50 +0.0001), +0.0005 holdout |
| stale replay | the pre2 ID part is inert |

**xdrop_pre2 interim.** This run drops the 6+3 pre2s columns, at 2 epochs. Logs: `nn_x/abl/logs/abl_xdrop_pre2_s{23,24}.log` vs `abl_base_s{23,24}.log` at the same step, still in the plateau, before the cooldown.

| step | s23 Δval | s23 Δholdout | s24 Δval | s24 Δholdout |
|---|---|---|---|---|
| 12K | -0.00108 | -0.00016 | -0.00120 | -0.00019 |
| 13K | -0.00250 | -0.00074 | -0.00146 | -0.00063 |
| 14K | -0.00136 | -0.00129 | -0.00125 | +0.00023 |
| 15K | -0.00135 | -0.00051 | -0.00099 | -0.00076 |
| 16K | -0.00075 | -0.00064 | - | - |

- Mean of the last 3 logged steps: val -0.0012, holdout -0.0006. So the channel is worth about +0.0012 val and +0.0006 holdout.
- The final value (abl_eval line) is still pending. It is the calibration point for every kill rule below.
- **Final (11-13 05:12, `nn_x/abl/logs/abl_eval_wave2.txt`, last checkpoints, s23/s24 pooled):** the channel is worth **+0.00128 ± 0.00061 val** (SE_cal; seeds +0.00150 / +0.00106) and **+0.00054 ± 0.00034 holdout**. Without it, stale-val loses -0.00188 ± 0.00056. Verdict LOAD-BEARING, but below the +0.0015 E1/F8 trigger (§6).

### 1.3 Gate and residual tests on G3

**R3 / P8 stage-0 gate** (`nn_x/r3/stage0_gate.log`, leak-free 14,306 feeds). Nested residual on G3:

| score | residual |
|---|---|
| pre_final | -0.00018 ± 0.00020 |
| ENS_clean+h | -0.00011 ± 0.00005 |
| ENS_all+h | -0.00026 ± 0.00012 |
| ID-only | exactly 0 (w=0 in every fold) |
| equal z-mix | -0.0079..-0.0111 |

- The 'right category, buried' cell holds 7,624 positives and 151,881 beater pairs: **37% of G3's loss, 43% under the private mix.**
- Pair AUC in that cell:

| score | pair AUC |
|---|---|
| encoder scores | 0.308-0.328 |
| ID-only | 0.471 / 0.476 |
| content-only | 0.329 |

- Every score sides with the popular, older, more history-similar beaters (+0.37 log-LUO).
- Verdict STOP. Stage 1 (encoder as init, frozen ID table) was never built.

**Collaborative filtering and other residuals** (`nn_x/audit/altapproach/findings.txt`)
- ALS f64/f128/clean, item2vec, exact co-visitation (with val+test sources): nested 0 to -0.0005; cell AUC 0.39-0.46.
- pre2_hmax32: -0.00032. pre2_fe: -0.00023.
- EASE / RDLAE (`nn_x/w25/study/study.md`): single feature 0.331, nested +0.0002.

**IDs inside the ranker**
- Hashed item IDs: -0.0022 ± 0.0008 val; -0.0075 on feeds with no ID-covered positive; collapses in epoch 2.
- Semantic-ID codes: +0.00038 (z 0.8), holdout +0.00003.

**LLM backbones** (`nn_x/llm_large.md`)
- Qwen3-0.6B / Qwen3.5-4B / 9B soft-token rankers: -0.0038 / -0.0048 / -0.0053 vs the ranker.
- Random-init 0.6B ties the pretrained one (+0.0005 ± 0.0007).
- The all-cold-positive slice gains +0.012-0.013 for every transformer arm, random init included. That is an architecture effect, not a pre-training effect.

**Architectures and test-time adaptation**
- F10 GRU ≈ tf; DIN -0.0012..-0.004.
- TTA (`r6_tta.md`): MC-dropout -0.00009 ± 0.00024; history-event dropout -0.00008 ± 0.00034; crop -0.0010 (history length is binding: -0.0041 at 64 events).
- Transductive val/test snapshots in pre-training (v3 vs v3A): <= +0.0004 ± 0.0003.

**The ranker is not overfitting.** At 10 epochs, late train loss is 2.9448 for blocks3 vs 2.9487 for base, and val rises monotonically to the last step (0.3905 → 0.3960). `nn_x/abl/logs/abl_e10_blocks3_s951.log` vs `nn_x/w5/logs/w5_e10_base_s951.log`.

**Ablation screen** (`nn_x/abl/logs/abl_eval.txt`): what each component is worth.

| component | val | holdout |
|---|---|---|
| retrieval | +0.0034 | +0.0029 |
| popularity | +0.00234 | +0.00743 |
| K9 aux | +0.0022 | +0.0023 |
| set attention | +0.0020 | - |
| ctx_nd | +0.00135 | +0.00030 |
| retr32 | FAIL: -0.00050 | -0.00048 |
| retr32u | FAIL: -0.00011 | +0.00005 |
| no_sim | inconclusive: +0.00044 | - |

## 2. Data facts that bound pre-training

Measured this audit; logs in `nn_x/pretrain2/audit/`.

**Overlap with the labelled data: the core fact**
- The 463.6M corpus tokens are 98.4% train-format and are the same events as the 448.7M train-history events inside the 3.15M labelled feeds.
- Only 7.35M val/test snapshot tokens (1.6%) are pre-training-only.
- So the unlabelled : labelled-context ratio is about **1:1**. The success cases ran at 10-1000:1.

**Time**
- Train: Sep 6-19 MSK, 12 effective days (no data Sep 16; 42 feeds on Sep 15).
- Val: Sep 22 + 24, 2-5 days after train. Test: Sep 26-29, 6-10 days after.
- Public = 8,562 feeds before Sep 27 11:32 MSK; private = 20,033 feeds.
- Timestamps have 128 s granularity.
- **History events have no timestamps, only order (hpos).** So per-user time series do not exist. The only time series are feed timestamps and item/category daily counts.

**History length H (cap 256)**
- Median: train 225 / val 191 / test 190.
- H=256: 42.3% / 29.0% / 28.6-29.4%.
- H=0: 1.1% train, 1.35% val.
- About 30% of events repeat an item already in the snapshot.
- A candidate is never in its own feed's history. Its category is there 84% of the time.

**Format mismatch**
- Event-type-1 share at hpos 192-255: train 0.405 vs val 0.114 / test 0.112.
- Overall et1 share: 0.519 vs 0.439 / 0.441.
- A corpus that is 98.4% train-format learns train-format statistics.

**Users**
- user_id 0: 3.9-4.6% of feeds.
- 17.2% of train users have 2+ feeds.
- Only 30.4% (val) / 29.1% (test; private 28.6%) of feeds belong to a user who has a train feed. **About 70% of val/test feeds are new users.**
- Returning users' train snapshot is median 10 days stale (val) / 15.5 days (private). Their val/test snapshot holds a median of 48 / 61 newer events.

**Coverage**
- 77.5% (val) / 82.2% (test) of history events are missing from every train snapshot.
- Item-ID row coverage of history events: train 84.6%, val 85.3%, test 82.7%. With the no-test table, test drops to 65.8%.
- 14% of history events have no content vector; 76% of those have an ID row.

**Candidate coverage**

| candidates with | train | val | test | private |
|---|---|---|---|---|
| a pre2 ID row | 74.1% (62% windowed) | 46.2% | 40.8% | 40.2% |
| any train exposure | - | 65.4% | 53.8% | 52.7% |

- 17.4% of val feeds have **only** cold positives; 29.1% have at least one.

**Item time series**
- Exposed val candidates: median 4 of 12 days, 9-10 exposures.
- Only 18-22% have 7 or more days.
- Items with 5 or more train clicks: 3.3% (val) / 2.4% (private).

**Categories**
- 52/53 codes, the same in every split.
- Jensen-Shannon divergence of candidate category shares train vs val/test 0.0022-0.0025. The largest share shift is about 1 point.

**user_emb dynamics**
- Identical across a user's snapshots for only 0.05-0.08%.
- Earliest vs latest train feed cos: q50 0.86. Val vs latest train: q50 0.82.
- A linear map from pooled history reconstructs it with held-out R² **0.56** (`eda/embeddings/03_user_vs_history.log`).

**Snapshot union**
- 18.1% of train feeds have an earlier same-user snapshot (median +50 older events).
- For val/test, the union adds 32 or more events to 25-26% of feeds, vs 11% of train feeds. The two sides are asymmetric.

**Leak and parity pitfalls** that any new corpus must respect:
1. Later same-user snapshots contain 54.4% of train positives. Train-row features must come only from time-windowed corpora (`build_corpus_window` rule). Leaky vs windowed: +0.017 / +0.009 / +0.004 on W1-W3.
2. Use `item_tab2` only. Never mix emb_idx across splits: 22% of items differ slightly between tables.
3. The val-side model must exclude test snapshots (v3A rule). Test-side transductive optimism is <= +0.0004.
4. Candidate ID coverage drifts from 74% (train) to 40% (private). id_contrib is banned for parity.
5. Candidate lists are L2 top-300 minus viewed items, truncated at the last interaction. Self-supervision on candidate sets absorbs that structure.
6. Never select on val. The val fine-tune of the encoder hurt.
7. Gains do not transfer between lenses: GBDT +0.0037 became <= 0 on G3, and the NN gain halved from 3 to 10 epochs.

## 3. Field research: what pre-training bought elsewhere, and under which conditions

### 3.1 Where pre-training beat a strong supervised ranker

| system | numbers | condition that held |
|---|---|---|
| GPSD, arXiv 2506.03699, Table 3 (L4H128) | - no transfer 0.6375<br>- full transfer 0.6376<br>- sparse transfer 0.6395<br>- full transfer + sparse freeze **0.6838**<br>- sparse transfer + freeze 0.6834; 0.7018 at 0.3B dense params | (b): the ID table overfits; the gain is sparse transfer + freeze, and dense transfer ≈ 0 |
| LazFormer, arXiv 2609.14978 | - from scratch 0.7577<br>- sparse transfer + freeze 0.7675<br>- sparse reset each epoch 0.7731<br>- + dense transfer 0.7761<br>- direct dense transfer ≈ 0 (0.7701 vs 0.7699); a zero-init adapter is required<br>Pre-training corpus: 11B tokens over 1 year, vs 370M impressions over 25 days | (a) and (b) |
| PinFM, arXiv 2507.12704 | - frozen Save +0.10% vs fine-tuned +3.76%<br>- 2-year sequences of up to 16K events<br>- fresh-item fixes (candidate-ID randomisation 10% + item-age dropout): 7-day Save from -23.98% to +8.16% | (a) and (c) |
| ARGUS, arXiv 2507.15994 | - as a feature: +1.17% without pre-training vs +2.32% with it<br>- by size: +1.35% at 3.2M params → +2.66% at 1B<br>- standalone: -4.8..-7.5% below production | (a): years of history |
| LUM, arXiv 2502.08309, Table 5 | - condition token +0.0098 AUC<br>- as features: incremental DLRM 0.7541 → 0.7659<br>- end-to-end HSTU 0.7334 vs DLRM 0.7338 | (a) and (c): 4B interactions, length 4096 |
| MAP, arXiv 2308.01737 | Criteo 0.8118 → 0.8165 with the same data, 60 pre-training epochs | (b): ID-heavy CTR overfitting |
| SUM, arXiv 2311.09544; Zalando LTE, arXiv 2604.08181 | - SUM: downstream NE -0.04..-0.35%<br>- LTE prefix token: +1.31% NDCG@500 | (c): the upstream sees features or windows the downstream lacks |
| Layer 6, RecSys Challenge 2022 (https://www.cs.toronto.edu/~mvolkovs/Recsys_2022.pdf) | MRR 0.2030 → 0.2121 with an MSM aux; it allowed a 10x larger model | label scarcity (1.1M sessions) and overfitting |
| WSDM Cup 2023, arXiv 2302.13498 | click-log pre-training, then a small expert-annotated set | huge unlabelled set, tiny labelled set |

### 3.2 Where it did not win

- End-to-end generative recommenders tie or lose: HSTU tied DLRM in LUM; ARGUS standalone was below production.
- GPSD full dense transfer = no transfer.
- PinFM frozen ≈ 0.
- EBES (arXiv 2410.03399): no SSL gain on the 1.5M-client MBD dataset.
- MLEM (arXiv 2401.15935): contrastive (CoLES-type) 0.729 < supervised 0.768.
- Our LLMs: random init = pretrained.
- RecSys 2025 4th place (arXiv 2508.06970): the graph model added about 0 next to a sequence encoder.
- In label-rich slate competitions the winners were GBDT or NN rankers with history and co-visitation features, with no trajectory pre-training:
  - OTTO (https://github.com/bschifferer/Kaggle-Otto-Comp);
  - KDD Cup 2023;
  - EB-NeRD (arXiv 2409.20483);
  - Avito 2026 public material (https://ods.ai/competitions/avitotechrecsys2026, a participant repository: CatBoost, public 0.34614).

### 3.3 Which conditions hold here: none

| condition | us |
|---|---|
| (a) unlabelled corpus >> labelled | about 1:1. Only val/test snapshots (1.6%) are new, and transductive pre-training on them gave <= +0.0004. |
| (b) an overfitting ID table to freeze | The ranker has no ID table. The generative ID channel has nested residual exactly 0 and cell AUC 0.47. Hashed IDs collapse. |
| (c) inputs the ranker cannot see | H <= 256, all of it visible to the ranker. The only exception is Avito's user tower (search queries, other surfaces), and it is already an input. |
| label scarcity / overfitting (Layer 6) | Not limited: blocks3 = base late train loss, and val rises to the end. |

Our recipe already has the pre-train → fine-tune structure:
- production feeds with the K9 production-order aux (which distils an external teacher);
- the exploration cooldown;
- the val fine-tune.

That is the TAAC-2025 2nd-place pattern (exposure pre-training, then click fine-tuning; arXiv 2604.04976).

### 3.4 Existing pre-trained models checked (the user's explicit question)

- **Time-series foundation models.** Chronos (2403.07815), Chronos-2 (2510.15821), TimesFM (2310.10688), Moirai (2402.02592), Lag-Llama (2310.08278), MOMENT (2402.03885), TimeGPT (2310.03589). There is nothing to forecast:
  - histories have no timestamps;
  - item series have at most 12 mostly-zero points;
  - 41% of private candidates have no series at all.
  - The target quantity is observed transductively (it_hist_*_luo, worth +0.0023 val / +0.0074 holdout), and every staler estimate lost: causal -0.0018, 4-day lag -0.0027, trends -0.00035.
  - Zero-shot TSFMs lose on intermittent series (arXiv 2502.12944).
- **Recsys foundation and generative models.** HSTU (2402.17152), OneRec / V2 (2502.18965, 2508.20900), OpenOneRec (2512.24762), TIGER (2305.05065), HLLM (2409.12740), RecGPT (2506.06270), UniSRec / VQ-Rec. Their weights need shared ID vocabularies or item text; ours are anonymised IDs and int8 vectors.
- **Event-sequence and user encoders.** CoLES / pytorch-lifestream (2002.08232), CLUE (2111.11294), LURM (2110.11337), LLM4ES (2508.05688). They are trained on other vocabularies or on bank transactions, and they win on user-level classification with weak downstream models.
- **Tabular foundation models.** TabPFN v2/3, TabICL, CARTE: closed (`research_1112/tools.md` T5).
- **Graph models.** LightGCN (2002.02126), PinSage (1806.01973), TwHIN (2202.05387): the same information as ALS / co-visitation (residual 0).
- **The only applicable pre-trained trajectory models are the two already in use: `user_emb` and pre2.**

## 4. The plan: what survives the critique (ranked)

Both critics refuted all six planned experiments (E1-E6) on the generational bar. The fixes they agree on:
1. No GPU spend now.
2. Keep only CPU closures, with stricter gates on val **and** the holdout.
3. Use rule R3 for any generational claim: >= 3 SE_v, holdout sign agreement, corr < 0.99. The earlier "holdout >= 2 SE, val >= 0" is the near-copy rule R4.
4. No OR branches on G3-error-selected cells.
5. Bonferroni correction across arms.
6. Read the two running results first: final `xdrop_pre2` and `abl_retr_cat`.
   - **Read 11-13:** xdrop_pre2 final +0.00128 ± 0.00061 val / +0.00054 ± 0.00034 holdout (LOAD-BEARING, below the +0.0015 trigger). abl_retr_cat FAIL: val +0.00046 ± 0.00049 (z_cal 0.93), holdout +0.00000. Source: `nn_x/abl/logs/abl_eval_wave2.txt`.

E1 and E2 move to §5, with reopen triggers. E3-E6 survive only in reduced, CPU-only form.

**Status 11-13: P1, P2, P3 and P4 all ran and are all CLOSED** by their pre-declared gates (§7). No stage 1 or stage 2 runs. 0 GPU-h.

**Calibration**
- A 2-epoch pair is 1.3-1.8 GPU-h (37-55 min per seed).
- A 10-epoch pair is about 4.7 GPU-h (142 min per seed).
- Generational steps had within-feed corr 0.86-0.93. For comparison, F10only is at 0.9946 and Cd at 0.9957 (`reports/VAL_PUBLIC_1113.md`).
- The stage-0 nested-residual lens is **not validated**. pre2 fails it (-0.00018) yet is worth about +0.001 in the NN. So a pass is weak evidence, and a fail is not a clean closure of NN value.

**Common harness for all items**
- New code only under `nn_x/pretrain2/`, CPU only.
- `OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 POLARS_MAX_THREADS=12`.
- Gates evaluate the nested residual on G3:
  - leak-free 14,306 val feeds, user-grouped 5-fold, 2 fold seeds;
  - a holdout replica on the 30K holdout feeds, using the G3 holdout preds;
  - pmix as report/veto (< -1 SE vetoes).
- The cell AUC is always reported against the anti-popularity controls -log1p(it_hist_all_luo) and -maxsim32. Never against a bare 0.5.

### P1. E6-reduced: label-free trajectory separation diagnostic (direct answer to the user)

> **Status 11-13: CLOSED (§7.1).** 0 of 8 arms pass. The best val result is ue32 shrunk: per-cluster minus global +0.00022 ± 0.00025 (z 0.89), which is below the random-label control rnd32 shrunk at +0.00028 ± 0.00024. The one holdout signal, ue32 shrunk +0.00056 ± 0.00015, does not transfer in either direction. Clusters predict G3 difficulty (ANOVA p down to 1e-8; buried-cell share 0.22-0.61 by cluster) but not a better scoring function. Public and private have the same cluster mix (JS <= 0.0009). Not run: the plan's per-cluster z(G3) + w_k z(x) arm for x ∈ {pre2_final, cos_u, maxsim32, it_hist_all_luo}. The pre-declared design swapped it for per-cluster family and FT/TO weights (§7.1).

- **Hypothesis.** Trajectories cluster. The question is whether cluster membership changes the *optimal scoring function*, i.e. whether per-cluster weights beat global ones. Earlier segment work used hand-defined covariates only (M, H, cold share, user0, category intent).
- **Objective / corpus.**
  - k-means with K ∈ {8, 32} (K = 128 dropped as underpowered: about 110 feeds per cluster);
  - fitted label-free on train latest snapshots;
  - two keys: (a) user_emb; (c) a history profile = last-32 category histogram + event-type mix + recency-weighted mean content;
  - the pre2 h_last key is dropped: it needs a GPU dump, and a closure does not need it;
  - assignment to val / holdout / test is transductive.
  - Script: `nn_x/pretrain2/cluster_diag.py`.
- **Entry into the ranker.** None unless it passes. On a pass: a zero-init cluster FiLM / prefix token.
- **Stage-0 gate.**
  1. Per-cluster G3 NDCG residual after regressing on M, H, cold share and user0, with power stated.
  2. Buried-cell concentration by cluster.
  3. A hierarchical (shrunk) per-cluster weight vs a global weight on z(G3) + w_k z(x), for x ∈ {pre2_final, cos_u, maxsim32, it_hist_all_luo}.
  - Bonferroni over 2 keys x 2 K x 4 signals.
  - **PASS: >= +0.001 at 3 SE on val AND the holdout agrees in sign at >= +0.0005, AND pmix >= -1 SE.**
  - Extra output: test cluster shares public vs private (label-free), as a diagnostic of the private shift. This is the most useful deliverable.
- **Stage 1/2.** Only on PASS: a 2-epoch pair (seeds 23/24) vs abl_base (val > +1 SE_cal, holdout >= 0), then a 10-epoch pair vs w5_e10_base_s951/952, then rule R3 vs G3.
- **Cost.** 0 GPU-h; about 3 CPU-h; 0.5 day.
- **Prior.** P(PASS) about 0.03. P(member >= +0.001) about 0.01. Expected Δval about 0.
- **Value: a clean, cited closure** of "separate trajectories first, then rank".
- **Leak and parity.**
  - Clusters are label-free and fitted on train only.
  - Val labels are used only inside nested folds; no cluster is selected on val NDCG.
  - Holdout feeds use their own user_emb only.
- **Dependencies.** None. It can run now on CPU.

### P2. E3-reduced: the non-history part of Avito's user tower, as the ||r|| x cos_u interaction

> **Status 11-13: CLOSED (§7.2).** The cross-fitted ridge reaches R² 0.590 (the plan quoted 0.56). The decile diagnostic holds: G3 is worse on high-||r|| feeds, residual contrast -0.0086 ± 0.0048 val and -0.0094 ± 0.0030 holdout. But the nested ||r|| x cos_u residual is val -0.00034 ± 0.00033 and holdout -0.00006 ± 0.00016, with pmix -0.00024 ± 0.00049. cos(c,r) has cell AUC 0.493, below the controls. Parity is fine (median ||r|| val / holdout 0.991), and public vs private shows no shift.

- **Hypothesis.** user_emb is the only pre-trained trajectory model whose inputs exceed ours: listing interactions + search queries, about 10B samples (https://habr.com/ru/articles/1004694; https://habr.com/ru/articles/974682). Pooled history explains only R² 0.56 of it.
- **Critic correction: the planned columns are redundant.** With r = u - f(h):
  - cos(c,r) = (|u| cos_u(c) - |f(h)| cos(c,f(h))) / |r|;
  - after within-feed z this is an exact per-feed linear combination of cos_u (already an input; residual on G3 +0.00003 ± 0.00028) and a pooled-history similarity (already covered).
  - **The only non-redundant quantity is the per-feed scalar ||r||**, i.e. how much Avito knows beyond the history, interacting with cos_u.
- **Objective / corpus.**
  - Ridge from format-robust pooled inputs (last1, means of the last 5 / 20 / 32, per-event-type means over the last 64, category histogram of the last 64, counts) to L2-normalised u.
  - **Cross-fitted 5-fold over train users**, so train rows get out-of-sample residuals. Applied to val/test with the full fit.
  - The MLP arm is dropped unless the ridge passes.
  - Script: `nn_x/pretrain2/uresid.py`.
- **Entry into the ranker.** On a pass: z(cos_u) x ||r||-decile columns.
- **Stage-0 gate.**
  1. Diagnostic: G3 per-feed NDCG residual by ||r|| decile. If G3 is not worse in high-||r|| deciles, stop.
  2. Nested residual of z(cos_u) x ||r|| on G3, val + holdout.
  3. Parity: median ||r|| on val within 10% of train random-like feeds; otherwise it is a format artefact.
  4. Report the within-feed Spearman of cos(c,r) with cos_u and with abl_base_s23 logits.
  - **PASS: residual >= +0.001 at 3 SE on val AND holdout >= 0 AND pmix >= 0.**
- **Stage 1/2.** Only on PASS: a 2-epoch pair (1.5 GPU-h), then a 10-epoch pair (4.7 GPU-h), then R3.
- **Cost.** 0 GPU-h; about 2 CPU-h; 0.5 day.
- **Prior.** P(PASS) about 0.04. P(member) about 0.02. Expected Δval about +0.00005.
- **Argument against.** Val/test candidate lists are L2 top-300 lists from Avito's own stack, which plausibly includes this tower, so the within-feed variance of cos_u is compressed.
- **Leak and parity.**
  - f is label-free and cross-fitted.
  - Use only the feed's own u. Never use another snapshot of the same user (the 'ud' drift family was rejected).
  - u is an own-time snapshot: cos_u AUC is flat across days.
- **Dependencies.** None.

### P3. E4-reduced: within-category similarity re-projection, as a label-free proxy (shared go/no-go for E1)

> **Status 11-13: CLOSED (§7.3)**; 4 of 5 gate clauses fail. S1cat cell AUC is 0.3924 vs raw 0.3730, a gain of +0.0195 ± 0.0030 (needed +0.05) and still below the controls (0.570 / 0.632). The nested residual is val -0.00031 ± 0.00017 and holdout -0.00012 ± 0.00008. Top-16 Jaccard is 0.57 for S1cat, so whitening does change retrieval. CSLS is stopped by the Jaccard rule (0.82 > 0.8). Stage 1 was already closed by the abl_retr_cat FAIL. The E1 cell proxy fails.

- **Hypothesis.** HistRetrieval (`nn_x/abl/model.py`) selects the top-16 history items by raw content cosine. The raw space is anisotropic (random-pair cos 0.31) and category-dominated (98.5% linear probe). A within-category-centred, whitened space might separate the buried positive from its beaters.
- **Critic counter-evidence.**
  - retr32 / retr32u FAILED, so the selection set is probably not binding.
  - `abl_retr_cat` (a learned second retrieval over same-category history, `nn_x/abl/abl_retr.py`) is running now and tests the same hypothesis. At step 4-5K it reads s23 0.38666 / s24 0.38551 val.
- **Objective / corpus.** No training.
  - Fit mu_cat (53 x 128) and a shrinkage-ZCA W on 5M train-table items.
  - S1 = cos(W(c - mu_cat(c)), W(h - mu_cat(c))).
  - S2 = CSLS hubness-corrected raw cosine.
  - The pre2 content-tower arm S3 is dropped (hmax32 0.345 < raw maxsim32 0.348).
  - Script: `nn_x/pretrain2/whiten_gate.py`.
- **Stage-0 gate (CPU only; the out-of-distribution inference-swap probe is dropped).**
  1. Label-free Jaccard of the val top-16 sets, raw vs S1 / S2. If > 0.8, stop that arm.
  2. Single-feature NDCG (rec_hl3, maxsim, last5) in each space, paired vs raw.
  3. **S1 cell pair AUC.**
  - **PASS: S1 cell AUC >= raw + 0.05 AND above the anti-popularity controls, AND nested residual on G3 > +0.0003 at 2 SE on val with holdout >= 0.**
- **Stage 1/2.** Only if PASS **and** abl_retr_cat PASSes (val > +1 SE_cal, holdout >= 0) **and** Jaccard < 0.6:
  - a flag in a `nn_x/pretrain2` copy of model.py that swaps only the top-m selection, bit-identical when off;
  - 2-epoch pair, then 10-epoch pair;
  - classify it as a **near-copy** and rank it by holdout (rule R4).
- **Cost.** 0 GPU-h; about 1.5 CPU-h; 0.5 day.
- **Prior.** P(PASS) about 0.08. P(member) about 0.03. P(generational) < 0.01.
- **Use as a proxy.** If even a hand-made within-category metric cannot move the cell above the controls, an in-category-negative encoder (E1) trained on the same content is unlikely to.
- **Leak and parity.**
  - Label-free, fitted on train only.
  - Each split's own emb table; emb_idx never mixed.
  - Missing-content masks unchanged.
- **Dependencies.** Read `abl_retr_cat_s23/s24` first for stage 1. Stage 0 can run now.

### P4. E5-reduced: cross-user trajectory-neighbour label transfer, val-only CPU stage-0

> **Status 11-13: CLOSED (§7.4).** 0 of 4 primary arms pass. Val runs from -0.00011 ± 0.00005 (s_neg_item, contrast_item) to -0.00031 ± 0.00012 (s_neg_cl). Holdout on the eligible feeds runs from -0.00009 to -0.00021. No cell AUC beats the controls (best is contrast_cl at 0.545). The neighbour signal is real on its own (s_pos_cl beats random by +0.021 NDCG), but G3 already contains it. 'Shown but ignored' items are clicked more than the base rate (0.0191 vs 0.0158), so exposure works as relevance and s_neg points the wrong way.

- **Hypothesis.** Use the pre-trained trajectory space (user_emb) as a retrieval key over *other* users' labelled train feeds, and transfer what they clicked (s_pos) and ignored (s_neg).
- **Critic corrections.**
  - Ignored impressions are the ranker's own listwise negatives, and s_pos is close to cos_u. Only s_neg / contrast has a distinct mechanism.
  - The train-stage build is underestimated by about 10x (about 1.6e10 gathers, about 2e14 MACs). It would take many CPU-hours to a day.
  - The neighbour pool size drifts with row date.
  - user_id 0 feeds cannot be self-excluded.
- **Objective / corpus.**
  - Keys: user_emb, cosine, K = 50, weights softmax(10 cos).
  - Value pool: random-like train feeds (p_random > 0.9, M >= 8, about 0.2M).
  - **Fixed-length trailing pool on every split**, e.g. the 4 days before T-2.
  - Same user excluded; holdout feeds never serve as neighbours.
  - Script: `nn_x/pretrain2/uknn_gate.py`.
- **Stage-0 gate.**
  - Pre-declared primary: z(s_neg) and z(contrast).
  - Within-feed corr with cos_u must be < 0.8.
  - **PASS: nested residual >= +0.001 at 3 SE on val AND holdout replica >= 0 (neighbours >= 2 days older), AND cell AUC above the controls.**
  - Report the query-to-neighbour time-gap distribution, train-stage vs val vs private.
- **Stage 1/2.** Only on PASS:
  - re-estimate the train-stage build on measured CPU throughput;
  - then a 2-epoch pair, then a 10-epoch pair.
- **Cost.** 0 GPU-h; about 3 CPU-h for val + holdout; 1 day.
- **Prior.** P(PASS) about 0.04. P(member) about 0.02. Expected Δval about +0.00005.
- **Counter-evidence.** Every collaborative-filtering / label-transfer channel has a residual of 0 to -0.0005 on G3. TabReD (arXiv 2406.19380) finds retrieval models losing their edge under a 6-10-day temporal shift.
- **Dependencies.** None. It is lowest priority.

**Totals for §4.** About 9-10 CPU-h, 0 GPU-h. A full reopen chain for any single passing item costs about 6-8 GPU-h.
- **Spent 11-13:** about 30 CPU-min of gated runs in total (P1 4 min, P2 1 min, P3 16 min, P4 7.5 min, all at 12 threads, full designs with no scale-down); 0 GPU-h. No item passes, so no reopen chain runs.

## 5. Rejected directions (do not re-propose without a new mechanism)

| direction | reason | reopen trigger |
|---|---|---|
| **E1 / P8-v2: re-targeted encoder** (in-category negatives, LUM category condition token, non-feed / contact targets) | Refuted by both critics.<br>- The fe target is the ranker's own click signal on the same events.<br>- Sampled in-category corpus negatives were never shown to the user; the ranker's in-feed negatives are stronger (mostly same-category, 84%).<br>- Scalar channel ceiling about +0.0012 val (xdrop_pre2).<br>- The stage-0 lens is unvalidated; the cell-feed OR branch is biased upward.<br>- Own prior: generational 0.02. | **All of:**<br>(i) final xdrop_pre2 >= +0.0015 val with holdout > 0;<br>(ii) P3's S1 cell AUC >= raw + 0.05 above the controls.<br>Then run arm B (nonfeed) alone, with:<br>- a holdout replica from retrained windowed encoders at stage 0;<br>- Bonferroni;<br>- residual >= +0.001 at 3 SE;<br>- Spearman to pre2_fe and to maxsim32 < 0.85;<br>- rule R3.<br>About 0.7 GPU-h gate + 3 windows + 1.5 + 4.7.<br>**Status 11-13: both clauses dead, CLOSED for good.** (i) The final xdrop_pre2 is +0.00128 ± 0.00061 val, below +0.0015. (ii) P3 S1 cell AUC is raw + 0.0195, below the controls (§7.3). |
| **E1 stage-2b: frozen ID table (GPSD ST&SF / PinFM)** | The existing ID channel is exactly 0 (cell 0.47). Hashed IDs collapse (-0.0075 on cold-positive feeds). Covers 40% of private candidates. | Only if a reopened E1 encoder has ID-only cell AUC > 0.55. **Status 11-13: dead (E1 CLOSED for good).** |
| **E2: in-ranker masked-newest-event / MSM aux** | - K9's +0.0022 comes from an external teacher (production order), not same-data self-supervision.<br>- The ranker does not overfit, so there is no regulariser path.<br>- Its probe always says GO (the comparator, the pre2 fe head, was trained in-sample on the same val-snapshot targets).<br>- Near-copy by construction; generational about 0.01. | A non-trivial probe passes:<br>- per-feed G3 NDCG residual vs pre2 recall@10 of hidden events, partial corr > 0.05 at 2 SE, controlling for M, H and cold share;<br>- the comparator is a windowed w3r encoder on post-window holdout feeds.<br>Then a pure 2-epoch screen (1.6 GPU-h), with rule R3 for any pool claim. |
| F8: scale-up of the existing pre2 (d 512-768, 6-8 layers, 4-6 epochs, bag-of-future head) | Pre-declared eligibility (xdrop_pre2 >= +0.0015 val, holdout >= 0) is not met on the interim (+0.0012 / +0.0006). Scaling sharpens the same popularity-aligned direction (cell 0.31-0.33). | The final xdrop_pre2 meets the eligibility rule. **Status 11-13: not met** (final +0.00128 ± 0.00061 val, holdout +0.00054). **CLOSED for good.** |
| Encoder as init of the ranker's history encoder (R3 stage 1; HSTU-style deep pre-training) | Dense transfer ≈ 0 at our ratio: GPSD 0.6376 vs 0.6375; LazFormer 0.7701 vs 0.7699; LUM HSTU = DLRM. Random-init LLM = pretrained. No overfitting; 10 epochs wash out convergence speed. | none |
| Fine-tuned parallel branch with a zero-init adapter (PinFM / LazFormer re-scope) | None of PinFM's or LazFormer's data conditions hold. The val fine-tune of the encoder hurt (0.3536 → 0.350-0.352). | A reopened E1 encoder passes stage 1. **Status 11-13: dead (E1 CLOSED for good).** |
| Pseudo-feed pre-training of the whole ranker (LazFormer-style) | Same dense-transfer argument. Its one real argument (10-25x denser supervision) is E2's mechanism, which is closed. 1.5-2 days of engineering. | E2 passes stage 2. |
| In-ranker two-view contrastive SSL (CL4SRec 2010.14395, DuoRec 2110.05730), the blocks3 "capacity enabler" | Regularisers with nothing to regularise (identical late train loss). DA4Rec (2403.11136): augmentation matches CL. | none |
| Contrastive trajectory / user encoders (CoLES, CLUE, LURM, LLM4ES, ICLRec 2202.02519, autoencoder "universal profiles") | user_emb already is one, from a corpus about 20x larger with search inputs, and it is already the user token. A per-feed constant only acts through interactions. EBES: no gain at 1.5M clients; MLEM: contrastive < supervised. | P1 passes (then as a neighbour space for P4). **Status 11-13: did not fire** (P1 CLOSED, 0/8 arms; P4 CLOSED on user_emb neighbours too). |
| Exposure-IPS pre-training target (fe_ips) | Label join plus pre-snapshot time masks; the weakest mechanism. | A reopened E1 arm moves the cell above the controls. **Status 11-13: dead (E1 CLOSED for good).** |
| Distilling the encoder's ranking into the ranker | Dominated: the teacher is already an input. Its order sides with the beaters (AUC 0.31). Born-again KD closed (+0.0006 n.s.). | none |
| Time-series foundation models (Chronos / Chronos-2, TimesFM, Moirai, Lag-Llama, MOMENT, TimeGPT) | No series (§3.4). A forecast is staler than the transductive nowcast; all staler popularity variants lost. | none |
| Generative recommenders (HSTU, OneRec / V2, TIGER) and public recsys foundation checkpoints | They replace the discriminative ranker and lose standalone. Candidates are fixed. Weights need text or a shared vocabulary. Semantic-ID inputs closed (+0.0004 n.s.). | none |
| Graph pre-training (LightGCN, PinSage, TwHIN, hyperbolic / PQ codes) | The same information as ALS / co-visitation (residual 0 to -0.0005, cell 0.39-0.46); about 0 next to a sequence encoder (2508.06970). High cost on a 95.6M-node graph. | none |
| Tabular foundation models (TabPFN v2/2.5/3, TabICL, Mitra, CARTE) | Pointwise; row ceilings; no listwise variant; CARTE needs strings. Closed as member, stacker and per-feed model. | none |
| Test-time adaptation (T2ARec 2504.01489, AdaBN, MC-dropout, crop) | Measured <= 0 (r6_tta). user_emb is already per-request and test-time adapted. | none |
| ID-channel variants of the existing encoder (idmax32 / fe_noid in the train stage, rare-ID hygiene, norm-bounded deltas) | A variant of a zero-information channel. id_contrib is banned for parity. | none |
| Snapshot-union corpus (sequences beyond 256 events) | Asymmetric: it extends val/test returning users far more than train rows (25% vs 11% of feeds gain 32+ events), a parity hazard. Long-history value is already captured by the H effect. | none |
| **P1 / E6: trajectory-type conditioning** (k-means clusters on user_emb or history profile; cluster FiLM, prefix / intent tokens, per-cluster MoE, per-cluster ensemble or FT/TO weights) | CLOSED 11-13 (§7.1). Per-cluster family weights: 0/8 arms; best val +0.00022 ± 0.00025, below a random-label control. Holdout-only ue32 +0.00056 ± 0.00015 does not transfer either way. Clusters predict G3 difficulty, not the scoring function. | A new key that gives per-cluster gain >= +0.001 at 3 SE on val with holdout agreement. |
| **P2 / E3: user_emb decomposition** (r = u - f(h), \|\|r\|\| x cos_u, cos(c,r)) | CLOSED 11-13 (§7.2). Nested val -0.00034 ± 0.00033, holdout -0.00006 ± 0.00016. cos(c,r) zero-shot 0.291 (about random), cell AUC 0.493. | none |
| **P3 / E4: within-category whitened or CSLS similarity space** for retrieval or features | CLOSED 11-13 (§7.3). Cell AUC +0.0195 over raw, below the controls. Nested val -0.00031 ± 0.00017, holdout -0.00012 ± 0.00008. abl_retr_cat FAIL. | none |
| **P4 / E5: cross-user trajectory-kNN label transfer** (s_pos / s_neg / contrast at item and cl4096 level) | CLOSED 11-13 (§7.4). 0/4 primary arms pass. Val -0.00011 to -0.00031, holdout negative. 'Shown but ignored' is a positive signal (exposure = relevance), so s_neg points the wrong way. | none |

## 6. Recommendation and kill criteria

**Start first** (CPU, any time, does not compete with the GPU queue):
1. **P1 (E6-reduced)**, 0.5 day. It answers the user's question directly and gives the public/private cluster-share diagnostic.
2. **P2 (E3-reduced)**, the ||r|| decile diagnostic first.
3. **P3 stage 0 (E4-reduced)**: Jaccard + S1 cell AUC. This is also the proxy gate for E1.
4. **P4 (E5-reduced)** only if CPU is idle.

**Status 11-13:** all four ran on CPU between 05:14 and 05:31 UTC, each pre-declared with `date -u` before any result. All four are CLOSED (§7). Nothing is left to start on this track.

**When GPUs free up: no pre-training GPU work by default.**
- GPU time goes to the main queue (generational candidates).
- Pre-training GPU work opens only through a reopen trigger in §5:
  - E1 arm B (needs the final xdrop_pre2 >= +0.0015 val with holdout > 0, plus the P3 cell proxy);
  - or a §4 item passing at 3 SE on both lenses.
- First GPU spend in that case: the E1-B val-side encoder (about 0.7 GPU-h on GPU N, after `nvidia-smi` shows < 4 compute processes and >= 60 GB free).
- **Status 11-13: neither route is open.** E1 arm B is dead: xdrop_pre2 +0.00128 < +0.0015, and the P3 cell proxy failed. No §4 item passed on either lens. Pre-training GPU work: none.

**Kill criteria (pre-declared)**

| trigger | action | status 11-13 |
|---|---|---|
| Final xdrop_pre2 < +0.0012 val | close E1 and F8 for good | not this row: the final value is +0.00128 ± 0.00061 |
| Final xdrop_pre2 in +0.0012..+0.0015 | E1 stays closed unless P3 passes | **FIRED** (+0.00128). P3 failed (§7.3), so **E1 is CLOSED for good** |
| Final xdrop_pre2 >= +0.0015 val with holdout > 0 | F8 and E1-B become eligible behind their gates | not met (+0.00128 val, +0.00054 holdout). **F8 is CLOSED for good** |
| abl_retr_cat FAIL | close the E4 stage-1 path; the P3 cell AUC stays as the E1 proxy only | **FIRED** (val +0.00046 ± 0.00049, holdout +0.00000). The proxy then failed too (§7.3) |
| P1 per-cluster gain < +0.001 at 3 SE, or the holdout disagrees | close trajectory-type conditioning (clusters, FiLM, MoE, intent tokens) for good | **FIRED**: best val +0.00022 ± 0.00025, 0/8 arms. CLOSED for good (§7.1) |
| P2: G3 not worse in the high-\|\|r\|\| deciles, or residual < +0.001 at 3 SE | close user_emb decomposition | **FIRED** on the residual clause (val -0.00034 ± 0.00033); G3 *is* worse in the high deciles. CLOSED (§7.2) |
| P4: s_neg / contrast residual < +0.001 at 3 SE, or holdout < 0 | close cross-user label transfer with the other collaborative-filtering channels | **FIRED** on both clauses (val -0.00011..-0.00031, holdout -0.00009..-0.00021). CLOSED (§7.4) |
| Any stage-1 2-epoch pair: val <= +1 SE_cal or holdout < 0 | stop | not reached (no stage 1) |
| Any 10-epoch member: Δval < 3 SE_v, holdout sign disagrees, or corr >= 0.99 | no generational claim; pool only under rule R4 | not reached |

**Budget cap for the whole track: 10 GPU-h** (one reopen chain). **Expected: 0 GPU-h.** Combined P(generational) across the track is about 0.03. P(any >= +0.001 member) is about 0.08.
- **Spent 11-13: 0 GPU-h, about 30 CPU-min.** The track is closed. Any further pre-training proposal has to name a new mechanism that clears §5; the closures in §7 do not give one.

---

## Registry one-liners

> Superseded 11-13 for E1, F8 and P1-P4 (the 'pending' and 'OPEN' lines below): the current lines are in §7.6. The other lines still stand.

For `reports/COVERAGE_MAP.md`:
```
| Pre-training on trajectories (PRETRAIN_PLAN_1113) | CLOSED for GPU: unlabelled:labelled-context ratio ~1:1 (463.6M corpus vs 448.7M train-history events in labelled feeds), no ID table, inputs fully visible; none of the 3 literature win conditions hold (GPSD 2506.03699, LazFormer 2609.14978, PinFM 2507.12704, LUM 2502.08309). pre2 channel worth ~+0.0012 val / +0.0006 holdout (xdrop_pre2 interim). Open only via reopen triggers in reports/PRETRAIN_PLAN_1113.md §5 |
| E1 P8-v2 in-category / condition-token / nonfeed / contact encoder | CLOSED pending triggers: final xdrop_pre2 >= +0.0015 val & holdout>0 AND within-cat whitened S1 cell AUC >= raw+0.05 (P3) |
| E2 in-ranker masked-newest-event / MSM SSL aux | CLOSED: K9 gain = external teacher, not same-data SSL; no overfitting (blocks3 vs base late loss 2.9448/2.9487); near-copy by construction |
| F8 pre2 scale-up (d512+, 4-6 ep, bag head) | CLOSED pending final xdrop_pre2 >= +0.0015 val, holdout >= 0 |
| Encoder-as-init / parallel fine-tuned branch / pseudo-feed ranker pre-training | CLOSED: dense transfer ~0 at same-data ratio (GPSD FT 0.6376 vs NT 0.6375; LazFormer 0.7701 vs 0.7699); random-init LLM = pretrained |
| Contrastive user encoders (CoLES/CLUE/LURM/ICLRec), TS foundation models (Chronos/TimesFM/Moirai), graph PT (LightGCN/TwHIN), generative recsys (HSTU/OneRec/TIGER), TFMs, distillation of pre2 | CLOSED (PRETRAIN_PLAN_1113 §5; no series, no text/shared vocab, CF-equivalent, teacher already an input) |
| P1 trajectory clustering diagnostic (user_emb / history profile, K 8/32) | OPEN CPU closure; PASS >= +0.001 at 3 SE val + holdout sign |
| P2 user_emb non-history residual (||r|| x cos_u, cross-fitted ridge) | OPEN CPU closure; cos(c,r) itself redundant (exact within-feed lin. comb. of cos_u and cos(c,f(h))) |
| P3 within-category whitening / CSLS of retrieval similarity | OPEN CPU stage-0 (Jaccard + S1 cell AUC); stage-1 only if abl_retr_cat PASS |
| P4 cross-user trajectory-kNN label transfer (s_neg/contrast) | OPEN CPU val+holdout stage-0 only; fixed-length trailing pool; train build ~10x costlier than first estimated |
```

For `reports/EXPERIMENT_REGISTRY.md`:
```
2026-11-13 | PRETRAIN_PLAN_1113 | analysis only, no training | audit + 3 research lenses + 2 critiques; all 6 planned pre-training experiments refuted on generational bar; 4 CPU closures kept (P1 clusters, P2 ||r||xcos_u, P3 within-cat whitening proxy, P4 trajectory-kNN), GPU 0 h planned, cap 10 h | reports/PRETRAIN_PLAN_1113.md, nn_x/pretrain2/audit/
2026-11-13 | xdrop_pre2 interim (2-ep, s23/s24, steps 12-16K vs abl_base same step) | pre2s channel worth ~+0.0012 val / ~+0.0006 holdout; final readout pending, sets E1/F8 triggers (>= +0.0015 val & holdout>0) | nn_x/abl/logs/abl_xdrop_pre2_s{23,24}.log
```

---

## 7. CPU closures run (11-13)

All four §4 items ran on CPU on 11-13 between 05:14 and 05:31 UTC.
- Setup: 12 threads, no GPU, full designs with no scale-down, about 30 CPU-min of gated runs in total.
- Each gate was written with `date -u` to `nn_x/pretrain2/<P>/PREDECLARED.txt` before any result was computed.
- New files exist only under `nn_x/pretrain2/P{1,2,3,4}/`. Nothing in data/, preds/, subs/, nn/ or nn_x/{r6,w5,abl} was modified, no submissions were made and no processes were killed.
- Every base reproduces the registry exactly: G3 val 0.40032 (leak-free 0.40046) and G3 holdout 0.32379 (equal-weight z-mean of the 20 train-only members on the 30K holdout feeds).
- Sign convention: Δ = arm minus base, mean ± SE over feeds. On val the lens is the 14,306 leak-free feeds.

| item | pre-declared (UTC) | gated primary: val Δ | holdout Δ | verdict |
|---|---|---|---|---|
| P1 trajectory clusters | 05:14:50 | best of 8 arms +0.00022 ± 0.00025 (z 0.89; needs +0.001 at z 3.58) | same arm +0.00056 ± 0.00015; does not transfer | **CLOSED** (0/8) |
| P2 \|\|r\|\| x cos_u | 05:15:18 | -0.00034 ± 0.00033 | -0.00006 ± 0.00016 | **CLOSED** |
| P3 within-category whitening | 05:15:05 | -0.00031 ± 0.00017; cell AUC +0.0195 over raw | -0.00012 ± 0.00008 | **CLOSED** (4/5 clauses fail) |
| P4 trajectory-kNN transfer | 05:15:18 | -0.00011 to -0.00031 (4 arms) | -0.00009 to -0.00021 | **CLOSED** (0/4) |

Shared conclusion: each of the three pre-trained label-free spaces (user_emb, its non-history residual and the content space) can sort *feeds* or *items* by difficulty or relevance on its own. None of them changes G3's ranking inside a feed in a way that replicates. In the 'right category, buried' cell, every new score stays below both anti-popularity controls (0.570 / 0.632):
- S1cat 0.392;
- cos(c,r) 0.493;
- kNN contrast_cl 0.545, the best of them.

### 7.1 P1: label-free trajectory clustering diagnostic (E6-reduced)

**Pre-declared.** 2026-11-13T05:14:50Z (`nn_x/pretrain2/P1/PREDECLARED.txt`). The post-hoc checks were declared at 05:20:49Z, after the gated run (`POSTHOC.txt`). They have no gate role.

**Design.**
- Clusters:
  - k-means (n_init 4, seed 0), K ∈ {8, 32}, on 150K train latest snapshots (non-zero users, holdout excluded), assigned to val, holdout and test by nearest centroid.
  - Key ue = L2-normalised user_emb.
  - Key hp = four equal-variance blocks:
    - sqrt category shares over the last 32 events;
    - event-type shares over the last 32 events;
    - log1p H;
    - exp-decay (half-life 8 events) recency-weighted mean content over the last 64 events, PCA 16.
- Random-label controls at K 8 and 32.
- Test 1 (the only gate): per-cluster minus global nested weighting of the three G3 families F10 / F7x / F7x70.
  - Simplex grid step 0.1 plus the G3 point (67 candidates).
  - Raw argmax and shrunk (tau = 1000 feeds) variants.
  - User-grouped 5-fold with user-0 singletons, 2 fold seeds.
  - Selection on val leak-free feeds; holdout nested within its own 30K feeds.
- Test 2: the same contrast for the FT/TO mix. Val only, because the FT models have no holdout predictions. Information only.
- Test 3: label-free shift diagnostic.

**Gate.** PASS iff some arm of the 8 (key {ue, hp} x K {8, 32} x {raw, shrunk}) has:
- Δval >= +0.001;
- z_val >= 3.58 (3 SE one-sided, Bonferroni over 8 arms);
- Δholdout > 0;
- private-mix Δ >= -1 SE (veto).

The plan's holdout clause (>= +0.0005) was reported alongside. Power: SE_val is 0.00020-0.00032, so a true +0.001 effect would have read at z of about 3-5.

**Numbers. Test 1** (per-cluster minus global):

| arm | val | holdout | private mix |
|---|---|---|---|
| ue8 raw | -0.00044 ± 0.00024 (z -1.85) | +0.00005 ± 0.00014 | -0.00031 ± 0.00036 |
| ue8 shrunk | -0.00025 ± 0.00020 | +0.00003 ± 0.00013 | -0.00022 ± 0.00031 |
| ue32 raw | -0.00015 ± 0.00032 | +0.00046 ± 0.00018 (z 2.57) | -0.00006 ± 0.00048 |
| **ue32 shrunk** | **+0.00022 ± 0.00025 (z 0.89)** | **+0.00056 ± 0.00015 (z 3.72)** | +0.00031 ± 0.00037 |
| hp8 raw | -0.00009 ± 0.00031 | +0.00015 ± 0.00016 | -0.00030 ± 0.00051 |
| hp8 shrunk | -0.00015 ± 0.00025 | +0.00019 ± 0.00015 | -0.00009 ± 0.00041 |
| hp32 raw | -0.00004 ± 0.00031 | -0.00034 ± 0.00017 (z -2.03) | +0.00004 ± 0.00050 |
| hp32 shrunk | -0.00025 ± 0.00022 | -0.00017 ± 0.00015 | -0.00023 ± 0.00034 |
| control rnd8 raw / shrunk | +0.00006 ± 0.00027 / +0.00008 ± 0.00022 | -0.00038 ± 0.00014 / -0.00022 ± 0.00013 | |
| control rnd32 raw / shrunk | +0.00001 ± 0.00032 / **+0.00028 ± 0.00024** | -0.00048 ± 0.00016 / -0.00039 ± 0.00014 | |

- The best real val arm (+0.00022) is below the random-label control rnd32 shrunk (+0.00028).
- Against fixed G3, ue32 shrunk reads val +0.00052 ± 0.00032 and holdout +0.00038 ± 0.00016. The global nested arm alone reads val +0.00030 and holdout -0.00018 against G3.

**Test 2** (FT/TO mix, val only; information):
- Per-cluster minus global spans -0.00051 ± 0.00027 (ue32 raw) to +0.00048 ± 0.00028 (hp32 raw).
- The largest z is hp32 shrunk, +0.00047 ± 0.00022 (z 2.15; private mix +0.00029 ± 0.00030). Random controls reach +0.00027.
- The global nested mix equals G3 (a = 0.5) to 0.00000.
- The val FT/TO curve is flat: a = 0 0.40047, 0.3 0.40091, 0.5 0.40046, 1 0.39997.

**Information: clusters do measure difficulty.**
- G3 residual on log M, log1p H, user0 and cold share differs by cluster on both lenses:
  - ue8 val F 2.62 (p 0.01) / holdout F 3.53 (p 9e-4);
  - ue32 holdout p 1e-8;
  - hp8 holdout p 2e-6;
  - hp32 holdout p 2e-7;
  - random p 0.47-0.91.
- Cluster residuals correlate between val and holdout at +0.35 to +0.53, against -0.46 / -0.19 for the random controls.
- Buried-cell share of positives is 0.347 overall.
  - By cluster: ue32 0.219-0.608 (p 7e-168), hp32 0.219-0.560 (p 4e-151), ue8 0.274-0.467, hp8 0.297-0.437.
  - Random controls: rnd8 p 0.77.
- Val-fitted per-cluster weights moved to the holdout (pre-declared transfer) give -0.00029 ± 0.00020 to +0.00017 ± 0.00018 over the 8 arms; ue32 shrunk gives -0.00018 ± 0.00017.

**Post-hoc (no gate role).**
- (a) Null from 50 random labelings (K32 shrunk): val -0.00006 (sd 0.00028), holdout -0.00011 (sd 0.00018). ue32 shrunk is at z_null +0.97 on val and +3.85 on holdout (0 of 50 nulls reach it). ue32 raw holdout is at z_null +3.18. All other arms are within ±1.6 sd of the null.
- (b) Holdout-fitted per-cluster weights applied to val, minus holdout-global: from -0.00012 ± 0.00026 (ue8 shrunk) to -0.00038 ± 0.00030 (hp8 raw); ue32 shrunk -0.00032 ± 0.00029.
  - Editor's note: the item summary gave this range as "-0.00024 to -0.00038"; the log has ue8 shrunk at -0.00012.
- (c) Full-fit per-cluster weights correlate across lenses at -0.54 to +0.15 (F10) and -0.14 to +0.64 (F7x); ue32 at -0.16 / +0.06.

**Test 3: shift diagnostic** (label-free; public 8,562 / private 20,033 test feeds).
- Public vs private JS: 0.0004 (ue8), 0.0009 (ue32), 0.0001 (hp8), 0.0005 (hp32); chi² p 0.02 / 0.13 / 0.89 / 0.83.
- Val vs private JS: 0.0019 / 0.0021 / 0.0003 / 0.0005. On ue this is almost entirely the zero-user_emb cluster: its share drops from 0.014 to 0.003 (-0.011 ± 0.001).
  - Zero-user_emb feeds number 205 in val, 80 in test and 1,187 in the holdout.
  - Their val G3 is 0.3817 ± 0.0166 (residual -0.028), but the holdout residual is +0.0003.
- Every other share shift is <= 0.008 ± 0.003.
- Val G3 reweighted to each split's cluster mix moves by at most -0.00053 ± 0.00027 on public (ue32) and +0.00021 ± 0.00019 on private (ue8). For comparison, the M x cold-share pmix gives 0.39711 (public) and 0.40261 (private) from 0.40046.

**Verdict: CLOSED by the pre-declared gate (0/8 arms; best val z 0.89).**
- Trajectory clusters are real and predict *how hard* a feed is for G3: ANOVA up to p 1e-8, residuals replicate across lenses at +0.35 to +0.53, and the buried cell is concentrated at 0.22-0.61 of a cluster's positives.
- They do not change *how* a feed should be ranked: no val gain, below a random control. The holdout-internal ue32 signal fails both transfer directions, and the per-cluster optimal weights are uncorrelated across lenses. So that signal is a property of the holdout lens (train-only members on train-format feeds), not a trajectory-type ranking rule.
- Public and private share the same trajectory composition. The private shift is item-side (staleness / M), not trajectory-side.

**Closes / opens.**
- Closes trajectory-type conditioning for good, per the §6 kill row: cluster FiLM, prefix or intent tokens, per-cluster MoE, and per-cluster ensemble or FT/TO weights.
- The §5 reopen trigger for contrastive user encoders ('P1 passes') does not fire. Nothing opens.
- Kept as information: the zero-user_emb feeds (1.4% of val vs 0.3% of test) score worse on val. That understates val vs test by about 0.011 x 0.028 ≈ +0.0003, the same size as the +0.00021 ± 0.00019 reweighted private shift. It is negligible, and the holdout does not show the penalty.

**Deviation from the plan, and its scope.**
- The plan's P1 gate item 3 was a per-cluster weight on z(G3) + w_k z(x), for x ∈ {pre2_final, cos_u, maxsim32, it_hist_all_luo}, with Bonferroni over 2 x 2 x 4.
- The pre-declared design replaced it with per-cluster *family* weights (Test 1) and FT/TO mixes (Test 2). The per-cluster x-signal arm was not run.
- Only the *global* versions of those signals are measured: cos_u +0.00003 ± 0.00028, pre_final -0.00018 ± 0.00020, maxsim -0.00011 ± 0.00023 (P3 raw), plus P2's continuous ||r|| x cos_u interaction at -0.00034 ± 0.00033.
- So the closure is complete for re-weighting what G3 already is. For per-cluster single-signal residuals it holds only by implication (no cluster carries a different scoring function in Test 1, and every global x residual is <= 0).
- If anyone needs the plan's exact gate, it is a CPU-minutes rerun on `out/clusters.npz`. The editor does not recommend it (prior P(PASS) about 0.03 before these results).

**Files.**
- `nn_x/pretrain2/P1/{PREDECLARED.txt, POSTHOC.txt, cluster_build.py, cluster_diag.py, p1_posthoc.py}`
- `out/{cluster_build.log, cluster_diag.log, cluster_diag.json, p1_posthoc.log, p1_posthoc.json, clusters.npz, cluster_info.json, nd_grid.npz}`

### 7.2 P2: the non-history part of user_emb, ||r|| x cos_u (E3-reduced)

**Pre-declared.** Fri Nov 13 05:15:18 UTC 2026 (`nn_x/pretrain2/P2/PREDECLARED.txt`). Before writing it, only shapes, dtypes, the hpos convention and the existence of user_emb were looked at.

**Design.**
- f(history) is a ridge from 1,539 format-robust pooled history features to L2-normalised u:
  - last1;
  - means of the last 5 / 20 / 32 events;
  - per-event-type means ({1}, {2}, {3}, {4}, {5..10}) over the last 64;
  - category histogram;
  - counts.
- f is cross-fitted 5-fold by user on 200K random train feeds plus the holdout feeds. The holdout gets out-of-fold predictions; val and test get the mean of the 5 fold models.
- r = u/||u|| - f(h).
- Primary: a nested 3-parameter ridge on within-feed-demeaned y over [zG3 (unpenalised), z(cos_u), nr_s·z(cos_u), nr_s·z(cos(c,r))], with penalty 0.01·N_rows. Weights are w = β_x/β_G3.
- Folds: user-grouped 5-fold, user-0 singletons, fold seeds 0/1.
- Feeds with user 0, |u| = 0 or H = 0 keep G3 (Δ 0).

**Gate.** PASS iff all of:
- val Δ >= +0.001;
- val Δ at least 3 SE;
- holdout Δ > 0.

Plan conditions, also required: pmix >= 0, and a G3 residual contrast (top three ||r|| deciles minus bottom three) < 0. A secondary NDCG-grid fit is information only.

**Numbers.**
- f(history): λ = 1000, chosen label-free by OOF R². OOF R² by fold is 0.5893-0.5899 (mean 0.5896). Holdout OOF 0.5801, val 0.5909, test 0.5885 (the plan quoted 0.56 for a simpler map). So about 41% of u's variance is beyond the history.
- ||r|| median: val 0.477, holdout 0.481, train sample 0.478, test public 0.480, test private 0.478. Parity val / holdout is 0.991, inside the 10% band, and there is no public/private shift.
- Checks: recomputed cos_u matches data/feats to 1.1e-7. G3 reproduces 0.40032 / 0.32379.
- G3 by ||r|| decile on val, D1 to D10: 0.4154 → 0.3841 (SE about 0.0067 each). Residual contrast D8-10 minus D1-3 (after log M, log1p H):
  - val -0.00861 ± 0.00482 (z -1.79; raw -0.01969 ± 0.00547);
  - holdout -0.00936 ± 0.00303 (z -3.09).
  - So the plan's decile condition is met: G3 is worse where Avito knows more than the history.
- **Primary nested ridge: val -0.00034 ± 0.00033 (z -1.04); holdout -0.00006 ± 0.00016 (z -0.36); pmix -0.00024 ± 0.00049.**
  - Val slices: M < 50 -0.00093 ± 0.00061; M >= 50 +0.00006 ± 0.00036.
  - Fold weights have stable signs on both lenses (val +0.030 / -0.053 / -0.021, holdout +0.018 / -0.050 / -0.010). High-||r|| feeds get less cos_u weight, but there is no NDCG gain.
  - Full-val weights applied to the holdout give -0.00025 ± 0.00019.
- Secondary NDCG grid: val +0.00025 ± 0.00028, holdout -0.00003 ± 0.00012. The best grid points differ between the lenses.
- Within-feed Spearman of cos(c,r): with cos_u 0.242, with abl_base_s23 logits 0.030, with G3 0.036. For reference, cos_u with abl_base_s23 is 0.592.
- Zero-shot NDCG: cos_u 0.3443, cos(c,r) 0.2910 (about random), G3 0.4006.
- Cell pair AUC (fittable feeds, 139,204 pairs): z(cos_u) 0.364, z(cos(c,r)) 0.493, controls 0.563 / 0.630.

**Verdict: CLOSED.**
- Task gate FAIL: val is negative, and the holdout sign does not agree.
- pmix condition fails (-0.00024, needs >= 0).
- The decile condition holds but does not rescue it.

**Closes / opens.**
- Closes user_emb decomposition (§6 P2 row): ||r|| x cos_u, cos(c,r), and the z(cos_u) x ||r||-decile input columns. No stage-1 pair.
- Avito's beyond-history information marks *harder* feeds, about 0.009 lower residual NDCG in the top three deciles on both lenses, but gives no usable within-feed re-weighting.
- Nothing opens.

**Notes.**
- User-0 feeds carry a per-feed, non-constant user_emb (norm about 256, per-dimension sd about 16.6 across feeds). They were excluded as declared.
- Runtime: 23 s for the residual build plus 19 s for the gate.

**Files.**
- `nn_x/pretrain2/P2/{PREDECLARED.txt, uresid.py, gate.py}`
- `out/{uresid.log, uresid.json, gate.log, gate.json, delta_val.npy, delta_holdout.npy, val_nr.npy, test_nr.npy, hold_nr.npy, {val,hold}_{cosu,coscr}.npy, hold_fidx.npy}`

### 7.3 P3: within-category whitening (S1) and CSLS (S2) of candidate-to-history similarity (E4-reduced stage 0)

**Pre-declared.** 2026-11-13 05:15:05 UTC (`nn_x/pretrain2/P3/PREDECLARED.txt`); results came in at 05:21-05:31. When it was written, abl_retr_cat had already FAILED, so stage 1 was closed in every case. Stage 0 decides only whether the direction and the E1 proxy stay alive.

**Design.**
- S1cat (gated): per-category centring plus per-category shrinkage-ZCA W_k, shrunk towards the pooled W (n0 = 5000, λ = 0.1 × mean eigenvalue). Fitted label-free on 4.73M distinct train-table items.
- S1pool: the plan's single shrinkage-ZCA W (a declared deviation; reported only).
- S2: CSLS hubness-corrected raw cosine.
- Primary signal: max similarity over the de-duplicated history (maxall). Other signals are reported only.
- No OR branches.

**Gate.** The direction stays OPEN only if all five hold for S1cat:
- (i) val cell AUC(S1) - AUC(raw) >= +0.05;
- (ii) S1 cell AUC above both controls;
- (iii) val nested residual >= +0.0003 and >= 2 SE, with the ridge variant > 0;
- (iv) holdout nested residual >= 0;
- (v) pmix >= -1 SE (veto).

Plan stop rule: top-16 Jaccard > 0.8 stops an arm.

**Numbers.**
- Checks:
  - raw last-32 max reproduces data/feats maxsim32 (6e-8);
  - random-pair raw cos 0.307;
  - category means explain 23.6% of the variance;
  - pooled within-category covariance has effective rank 79.8.
- Top-16 Jaccard vs raw, val [holdout]:
  - S1cat 0.572 [0.578]; 0.528 on feeds with > 32 distinct history items.
  - S1pool 0.663 [0.669].
  - **CSLS 0.822 [0.826]: above 0.8, so the plan stop rule ends the CSLS arm.**
- Cell pair AUC (7,543 positives, 151,881 pairs, SE clustered by positive):

| score | val | holdout (23,325 pos., 562,545 pairs) |
|---|---|---|
| control -log1p(it_hist_all_luo) | 0.5699 ± 0.0050 | 0.5947 ± 0.0028 |
| control -maxsim32 | 0.6315 ± 0.0050 | 0.6015 ± 0.0027 |
| raw maxall | 0.3730 ± 0.0050 | 0.4015 |
| **S1cat maxall** | **0.3924 ± 0.0052** (paired vs raw +0.0195 ± 0.0030) | 0.4200 (+0.0185) |
| S1pool maxall | 0.3693 ± 0.0051 (-0.0037 ± 0.0019) | 0.3978 |
| CSLS maxall | 0.3767 ± 0.0052 (+0.0037 ± 0.0029) | 0.4028 |

- Single-feature NDCG, paired vs raw (val): S1cat is worse on every signal (maxall -0.0032 ± 0.0009 to last5 -0.0048 ± 0.0010). S1pool is between +0.0006 and -0.0036, CSLS between -0.0021 and -0.0091.
- Nested residual on G3 (grid | ridge | pmix):

| signal | val | holdout |
|---|---|---|
| **S1cat maxall (gated)** | **-0.00031 ± 0.00017 (z -1.86)** \| -0.00048 ± 0.00045 \| -0.00024 ± 0.00026 | **-0.00012 ± 0.00008** \| -0.00040 ± 0.00020 |
| raw maxall | -0.00011 ± 0.00023 \| -0.00002 ± 0.00005 \| -0.00004 ± 0.00034 | -0.00003 ± 0.00008 \| -0.00034 ± 0.00012 |
| S1pool maxall | -0.00027 ± 0.00018 \| -0.00017 ± 0.00029 \| -0.00042 ± 0.00027 | -0.00004 ± 0.00008 \| +0.00004 ± 0.00009 |
| CSLS maxall | -0.00031 ± 0.00016 \| +0.00000 ± 0.00005 \| -0.00005 ± 0.00023 | -0.00005 ± 0.00007 \| -0.00019 ± 0.00011 |
| S1cat top16 (not gated) | +0.00009 ± 0.00023 \| -0.00131 ± 0.00062 \| +0.00042 ± 0.00032 | +0.00024 ± 0.00012 (z 1.96) \| -0.00085 ± 0.00031 |
| S1cat rec_hl3 (not gated) | +0.00019 ± 0.00024 \| -0.00297 ± 0.00078 \| +0.00061 ± 0.00038 | -0.00012 ± 0.00015 \| -0.00317 ± 0.00047 |

**Verdict: CLOSED; 4 of 5 clauses fail.**
- (i) +0.0195 ± 0.0030, short of +0.05.
- (ii) 0.392, below 0.570 and 0.632.
- (iii) -0.00031 ± 0.00017, ridge -0.00048.
- (iv) -0.00012 ± 0.00008.
- (v) passes the veto only.

Whitening changes retrieval a lot (about half of the top-16 set). It moves max-similarity slightly towards chance in the buried cell (+0.02, replicated on the holdout). But it still sides with the beaters (0.39 vs 0.5), and its value on G3 is negative.

**Closes / opens.**
- Closes P3 and the within-category re-projection direction (E4).
- Stage 1 was already closed by abl_retr_cat FAIL (val +0.00046 ± 0.00049, holdout +0.00000).
- **Fails the E1 cell proxy** (needed S1 >= raw + 0.05 and above the controls). With the final xdrop_pre2 at +0.00128 (in the +0.0012..+0.0015 band, where E1 'stays closed unless P3 passes'), **E1 is closed for good**.
- S1cat top16 on the holdout (+0.00024 ± 0.00012, z 1.96) is not a lead: it was not gated, val is +0.00009 ± 0.00023, and the ridge variants are negative on both lenses.
- Nothing opens.

**Notes.**
- For the whitened signals (S1cat, S1pool), the pointwise ridge picks a large positive weight (0.05-0.19) and loses NDCG, while the grid picks about ±0.02. For raw and CSLS the ridge weight is about 0 on val and about -0.03 on the holdout.
  - Editor's note: the item summary said 'always'; that holds for the whitened signals only.
- The cell is about popularity and age, not the geometry of the similarity space. Scores rank as follows:
  - pre2 encoder scores 0.31-0.33;
  - raw content 0.37;
  - whitened 0.39;
  - only the anti-popularity controls (0.57-0.63) side with the buried positive.
- Runtime about 16 min: fit 13 s, val 140 s, holdout 361 s (of which CSLS kNN 290 s), gate 225 s.

**Files.**
- `nn_x/pretrain2/P3/{PREDECLARED.txt, whiten_gate.py}`
- `out/{fit.log, fit.npz, sims_val.log, sims_val.npz, sims_holdout.log, sims_holdout.npz, gate.log, gate.json}`

### 7.4 P4: cross-user trajectory-neighbour label transfer (E5-reduced stage 0)

**Pre-declared.** 2026-11-13T05:15:18Z (`nn_x/pretrain2/P4/PREDECLARED.txt`), before any signal, coverage or residual was computed.

**Design.** The bank uses train labels only:
- train feeds with p_random > 0.5 and M >= 50;
- excluding the holdout feeds, user 0, and every user who appears in val, test or the holdout;
- a fixed-length trailing window: the 5 most recent data days at least 3 days before the query;
- one entry per bank user, with key = mean L2-normalised user_emb and value = all shown candidates with labels.

Neighbours: the query's own user_emb, exact top-32 by cosine, weights softmax(10 cos).

Signals at item and cl4096 level:
- s_pos (a neighbour clicked);
- s_neg (shown, none clicked);
- contrast = s_pos - s_neg.

Primary arms: s_neg_item, contrast_item, s_neg_cl, contrast_cl. Lens: zG3 + w·z(x), with w from a ±{0..1.0} grid, nested user-grouped 5-fold, 2 fold seeds.

Declared deviations from the plan text:
- k = 32 (plan 50);
- bank p_random > 0.5 and M >= 50 (plan > 0.9 and M >= 8);
- 5-day window with a >= 3-day gap (plan 4 days before T-2);
- all val, test and holdout users excluded (stricter).

**Gate.** An arm PASSES iff val Δ >= +0.0010 and >= 3 SE (one-sided over 4 arms, family-wise α <= 0.0054), and holdout Δ on the eligible feeds > 0. P4 passes iff any arm passes. Plan extras (pmix >= -1 SE, |corr with cos_u| < 0.8, cell AUC above both controls) would have downgraded a pass to PASS-CONDITIONAL.

**Numbers.** Full build, no subsampling, 451 s.
- Banks:
  - val (Sep 13, 14, 17, 18, 19): 57,067 feeds, 56,660 users, 8.48M shown rows, CTR 0.0124.
  - holdout: 48-52K feeds per window.
  - Holdout eligible: 13,424 of 30,000 feeds. Feeds from Sep 6-12 have no full window.
  - Query-to-neighbour gap q50: 6.17 days (val) vs 5.51 days (holdout).
  - Top-32 mean cos: 0.710 vs 0.697.
- Coverage (val / holdout-eligible):
  - item-level evidence 0.029 / 0.027 of candidates; item click evidence 0.0005;
  - cl4096 evidence 0.854 / 0.843; cl4096 click evidence 0.336 / 0.320.
- Label rate by evidence (val; base 0.0158):

| evidence | label rate |
|---|---|
| item clicked by a neighbour | 0.0329 |
| **item shown to a neighbour but ignored** | **0.0191** |
| cluster clicked | 0.0205 |
| cluster ignored only | 0.0145 |
| no cluster evidence | 0.0097 |

- Within-feed corr with cos_u: s_neg_item +0.08, contrast_item -0.08, s_neg_cl +0.52, contrast_cl -0.47. All are under the 0.8 limit.
- Single feature vs random tie-break (post-hoc, descriptive): s_pos_cl +0.0208 ± 0.0015, s_neg_cl +0.0168 ± 0.0016, s_neg_item +0.0049 ± 0.0009, contrast_cl -0.0202 ± 0.0014.
- Cell pair AUC (7,543 positives, 151,885 pairs):
  - controls 0.5700 ± 0.0051 / 0.6315 ± 0.0056;
  - item-level 0.495-0.505;
  - s_pos_cl 0.468;
  - s_neg_cl 0.444;
  - contrast_cl 0.545 ± 0.0056.
- Nested residual on G3:

| arm | val leak-free | pmix | holdout eligible | holdout all 30K |
|---|---|---|---|---|
| s_neg_item | -0.00011 ± 0.00005 (z -2.16) | -0.00018 ± 0.00007 | -0.00009 ± 0.00005 | -0.00004 ± 0.00002 |
| contrast_item | -0.00011 ± 0.00005 (z -2.10) | -0.00017 ± 0.00007 | -0.00009 ± 0.00005 | -0.00004 ± 0.00002 |
| s_neg_cl | -0.00031 ± 0.00012 (z -2.59) | -0.00025 ± 0.00018 | -0.00021 ± 0.00010 | -0.00009 ± 0.00005 |
| contrast_cl | -0.00022 ± 0.00021 (z -1.07) | -0.00003 ± 0.00033 | -0.00016 ± 0.00007 | -0.00007 ± 0.00003 |
| s_pos_item (secondary) | -0.00003 ± 0.00003 | -0.00004 ± 0.00005 | +0.00001 ± 0.00002 | +0.00000 ± 0.00001 |
| s_pos_cl (secondary) | -0.00066 ± 0.00028 | -0.00073 ± 0.00046 | 0 (w = 0 in every fold) | 0 |
| JOINT ridge (secondary) | -0.00052 ± 0.00024 | -0.00031 ± 0.00039 | 0 | n/a |

- In-sample best weight (an upper bound with no SE): val <= +0.00009 (contrast_cl at w = -0.02); holdout <= +0.00002.

**Verdict: CLOSED (0/4 primary arms).**
- Val is negative for every arm, nowhere near +0.0010 at 3 SE.
- The holdout is negative too.
- pmix is below -1 SE for 3 of 4 arms.
- No cell AUC beats the controls.

**Closes / opens.**
- Closes cross-user label transfer together with the other collaborative-filtering channels: ALS, item2vec, co-visitation, EASE, and now trajectory-kNN s_pos / s_neg / contrast.
- No stage 1 and no train-stage build, so the roughly 10x build cost never needs re-estimating.
- Nothing opens. P4 was never the route to reopen contrastive user encoders; that depended on P1, which failed.

**Reading.**
- The neighbour signal is informative on its own: s_pos_cl beats random by +0.021, and clicked clusters have about 2x the label rate of clusters with no evidence. G3 already contains all of it.
- The plan's 'shown but ignored' mechanism points the wrong way. Exposure to similar users is itself a relevance signal from Avito's candidate generation (0.0191 > 0.0158 base rate). So a low contrast_cl marks the popular beaters, which is why contrast_cl reads 0.545 in the cell, below both controls.
- Item-level evidence is too sparse to matter (2.9% of candidates; 0.05% with a neighbour click).

**Notes.**
- The first run crashed in the nested step (a stale variable). It was rerun with an unchanged design; the crash output is kept.
- A brute-force check of the kernel on 40 val feeds matches within 2.4e-4, which is float16 storage error.
- `uknn_gate.log` was truncated once by `check_transfer.py`. It was restored byte-for-byte from `uknn_gate.stdout`, and the script is now guarded.

**Files.**
- `nn_x/pretrain2/P4/{PREDECLARED.txt, uknn_gate.py, uknn_gate.log, uknn_gate.stdout, uknn_gate.json, uknn_gate.stdout.crash1, signals_val.npz, signals_holdout.npz, deltas_val.npz, deltas_holdout.npz, check_transfer.py, check_transfer.log}`

### 7.5 Track status after the closures

| item | status 11-13 | deciding numbers |
|---|---|---|
| E1 re-targeted encoder (arm B nonfeed included) | **CLOSED for good** | final xdrop_pre2 +0.00128 ± 0.00061 val (the +0.0012..+0.0015 band: 'closed unless P3 passes'); P3 cell proxy +0.0195 < +0.05, below the controls |
| E1 stage-2b frozen ID table; parallel fine-tuned branch; fe_ips target | **CLOSED for good** | their triggers needed a reopened E1 |
| F8 pre2 scale-up | **CLOSED for good** | eligibility needed final xdrop_pre2 >= +0.0015 val; got +0.00128 (holdout +0.00054) |
| E2 masked-event aux; pseudo-feed pre-training | CLOSED (unchanged) | trigger probe not run; nothing here bears on it |
| E4 stage 1 (retrieval-key swap) | **CLOSED** | abl_retr_cat FAIL (val +0.00046 ± 0.00049, holdout 0.0) plus P3 FAIL |
| Contrastive user encoders | CLOSED (unchanged) | reopen trigger 'P1 passes' did not fire |
| P1-P4 | **CLOSED** | §7.1-7.4 |

- Pre-training GPU work: none, and none is eligible. 0 GPU-h spent of the 10 GPU-h cap.
- The answer to the user's question stands and is now measured. Label-free trajectory models *do* separate trajectories: clusters predict feed difficulty, ||r|| predicts feed difficulty, and neighbours carry real click information. But every such separation is either already inside G3 or does not change the within-feed ranking on two lenses.

### 7.6 Registry one-liners (11-13 closures; these replace the E1, F8 and P1-P4 lines above)

For `reports/COVERAGE_MAP.md`:
```
| E1 P8-v2 in-category / condition-token / nonfeed / contact encoder | CLOSED for good 11-13: final xdrop_pre2 +0.00128 ± 0.00061 val (< +0.0015 trigger) AND P3 cell proxy failed (S1 +0.0195 over raw, below controls 0.570/0.632) | reports/PRETRAIN_PLAN_1113.md §7.5 |
| F8 pre2 scale-up (d512+, 4-6 ep, bag head) | CLOSED for good 11-13: final xdrop_pre2 +0.00128 val / +0.00054 holdout < +0.0015 eligibility | nn_x/abl/logs/abl_eval_wave2.txt |
| P1 trajectory clustering (user_emb / history profile, K 8/32; per-cluster family + FT/TO weights) | CLOSED 11-13: 0/8 arms; best val per-cluster minus global +0.00022 ± 0.00025 (ue32 shrunk, z 0.89; random-label control +0.00028); holdout-only ue32 +0.00056 ± 0.00015 does not transfer (val->hold -0.00018 ± 0.00017, hold->val -0.00032 ± 0.00029); clusters predict G3 difficulty (holdout ANOVA p 1e-8, buried-cell share 0.22-0.61) not the scoring function; public vs private cluster JS <= 0.0009 (private shift is item-side); per-cluster z(x) arm of the plan not run | nn_x/pretrain2/P1/ |
| P2 user_emb non-history residual (r = u - f(h), ||r|| x cos_u, cos(c,r)) | CLOSED 11-13: f(h) R² 0.590; G3 worse on high-||r|| feeds (resid contrast val -0.0086 ± 0.0048, hold -0.0094 ± 0.0030) but nested val -0.00034 ± 0.00033, hold -0.00006 ± 0.00016, pmix -0.00024; cos(c,r) zero-shot 0.291, cell AUC 0.493 | nn_x/pretrain2/P2/ |
| P3 within-category whitening / CSLS of retrieval similarity (E4; E1 cell proxy) | CLOSED 11-13: S1cat cell AUC 0.392 vs raw 0.373 (+0.0195 ± 0.0030, need +0.05; controls 0.570/0.632); nested val -0.00031 ± 0.00017, hold -0.00012 ± 0.00008; Jaccard S1 0.57, CSLS 0.82 (stop rule); stage 1 already closed by abl_retr_cat FAIL | nn_x/pretrain2/P3/ |
| P4 cross-user trajectory-kNN label transfer (s_pos/s_neg/contrast, item + cl4096) | CLOSED 11-13: 0/4 primary arms; val -0.00011..-0.00031, hold(elig) -0.00009..-0.00021; cell best contrast_cl 0.545 < controls; shown-but-ignored items clicked above base rate (0.0191 vs 0.0158) -> s_neg points the wrong way; closed with ALS/item2vec/co-vis/EASE | nn_x/pretrain2/P4/ |
```

For `reports/EXPERIMENT_REGISTRY.md`:
```
2026-11-13 | P1 cluster_diag (CPU, 4 min) | label-free k-means on user_emb / history profile (K 8/32), nested per-cluster re-weighting of G3 families + FT/TO mix vs global, val + holdout | CLOSED (0/8 arms; best val +0.00022 ± 0.00025 < random control; holdout-only ue32 +0.00056 does not transfer); trajectory-type conditioning closed for good | nn_x/pretrain2/P1/out/{cluster_diag,p1_posthoc}.log
2026-11-13 | P2 uresid + gate (CPU, 1 min) | cross-fitted ridge f(h) -> u (R² 0.590), nested ||r|| x cos_u / cos(c,r) residual on G3 | CLOSED (val -0.00034 ± 0.00033, hold -0.00006 ± 0.00016, pmix -0.00024); G3 worse on high-||r|| feeds (-0.009 resid both lenses) but no usable re-weighting | nn_x/pretrain2/P2/out/gate.log
2026-11-13 | P3 whiten_gate (CPU, 16 min) | per-category shrinkage-ZCA (S1cat), pooled ZCA, CSLS similarity; Jaccard, buried-cell AUC, nested residual | CLOSED (cell +0.0195 vs +0.05 needed, below controls; val -0.00031 ± 0.00017, hold -0.00012 ± 0.00008); E1 cell proxy FAIL -> E1 closed for good | nn_x/pretrain2/P3/out/gate.log
2026-11-13 | P4 uknn_gate (CPU, 7.5 min) | user_emb top-32 neighbours in a fixed 5-day trailing train bank, s_pos/s_neg/contrast label transfer at item + cl4096 | CLOSED (0/4; val -0.00011..-0.00031, hold negative; exposure = relevance, s_neg wrong-signed) | nn_x/pretrain2/P4/uknn_gate.log
2026-11-13 | PRETRAIN track status | final xdrop_pre2 +0.00128 ± 0.00061 val / +0.00054 holdout + P3 FAIL -> E1, F8 closed for good; abl_retr_cat FAIL -> E4 stage 1 closed; no reopen trigger fired; 0 GPU-h spent | reports/PRETRAIN_PLAN_1113.md §7
```
