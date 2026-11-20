# r6_external: external tricks and model structures, with a strict gap analysis against COVERAGE

Written 2026-11-12, about 01:30-03:00 UTC, by the R6 external-research agent. CPU only (`CUDA_VISIBLE_DEVICES=""`, 8 threads, 1 process). Nothing under nn/, src/, pretrain/, data/feats/, preds/, subs/ or nn_x/solvers/ was touched. The only files written are this report and two information-only scripts with their logs in `/tmp/scratchpad/r6/external/` (`branch_corr.py`, `branch_corr.log`, `branch_corr_hold.py`, `branch_corr_hold.log`).

**Evidence tags**
- [V]: primary source read this session (URL in §7).
- [S]: search snippet or abstract only.
- [K]: background knowledge.
- [R]: earlier project report (file:line).
- [E]: my own CPU measurement this session.

**Pre-declaration note.** The two [E] scripts are information only: they read existing prediction files and no decision depended on them. Every GPU pilot below has its acceptance rule written here, before it runs.

**What already covers this ground.** The parallel R6 workflow covers three things, so I cite it and do not repeat it:
- TTA: `reports/r6_tta.md`, pre-declared;
- the batch-size / gradient-noise-scale analysis: `nn_x/r6/recap_gns.py`;
- plateau weight averaging: `nn_x/r6/recap_swa.py`.

The run meta-analysis is in `nn_x/r6/recap/{curves,paired}_report.txt`. Earlier external reviews are `deep_*`, `research_*` and `nnx_research_*`, all of which I read. They cover most of the requested sources, so this report adds only what they lack.

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **Coverage is nearly complete.** Of the 34 technique families on the task list (§2):
   - **20 are covered** (tested, or closed with evidence);
   - **10 are partial** (proposed or parked, never run, or run only in an old recipe);
   - **4 are new**: cooldown-branch members, augmentation of the exploration pool, candidate-side retrieval values, and a typed target-action view.
   - No external source points to a lever that is both large and untested. Single members are saturated on capacity, losses and features. The ensemble is saturating too: pool gains plateau at about +0.0009 val (CURRENT_STATUS 01:20).
2. **New, measured [E]: a second cooldown started from the same trajectory decorrelates as much as an independent seed** (§1.1).
   - v15_cd_all_e10 and v13_lr2_mix1 share the seed-23 trajectory up to about 78.7k steps and were cooled separately.
   - They correlate **0.944 on val / 0.948 on the holdout**, with pair gain **+0.0019 / +0.0020**.
   - Independent-seed pairs: 0.937-0.947 / 0.938-0.953, with gain +0.0018-0.0025 / +0.0019-0.0023.
   - So most of a 10-epoch member's seed-specific variance is created in the noisy early-cooldown phase where the holdout selects. It does not come from the 8 plateau epochs.
   - Implication: a member can cost about 30 GPU-min (1-epoch cooldown branch plus fine-tune) instead of about 2.3 h.
   - 21 F7 plateau checkpoints already exist (`nn/ckpt/v17_pre2s_s10x@{60000,70000,77500}.pt`).
   - This contradicts SOLUTIONS_REVIEW §4.2 #5 ("F7 already cooled, so no F7cd: near-duplicates"). That statement holds for re-cooling a cooled checkpoint. It does not hold for branching from a plateau snapshot.
3. **Honest size of (2):** the direct ensemble gain is small. Adding seed-like members to the 25-member P(F6\*) pool gives about +0.0001 val (a − b/k with b ≈ 0.0048, sol_ensemble §2.6). Its value is as a **cost multiplier**: it makes architecture families (F10-F12, R4) and hyperparameter families (R3, R7) large enough to pass gate (b).
4. **The exploration pool is still memorised at mix 0.1 in every 10-epoch run.** End-of-cooldown minus selected step:

   | family | val | holdout |
   |---|---|---|
   | F6 | −0.0042 | −0.0044 |
   | F9 | −0.0044 | −0.0053 |
   | F7 | −0.0020 | −0.0031 |
   | mix-0 run | −0.0007 | −0.0009 |

   Sources: `nn_x/r6/recap/curves_report.txt` and the `v13_lr1_mix0` log.
   - At 10 epochs the pool therefore adds nothing at the selected step: mix 0 vs mix 0.1 is +0.00045 val in favour of mix 0 (paired_report).
   - At 2 epochs the pool mix was worth about +0.0018 (sol_solvers TL;DR 1).
   - Label smoothing did not stop the memorisation (sol_solvers §2.4). Input augmentation is the untested, literature-standard remedy (R2).
5. **Hyperparameter audit [E]** (logs of 224 stage-1 runs, `nn/logs` + `nn_x/*/logs`):
   - never varied: dropout 0.1 (224/224, including attention-probability dropout), wd 0.02 (444/444, including fine-tunes), ff 2, batch 32768 candidates / 512 feeds, max_hist 256, retr_m 16 (62/62 retrieval runs), cooldown_frac 0.2;
   - wd matters most in principle. The decay timescale is τ = 1/(ηλ) ≈ 50k steps = 5 epochs at lr 1e-3. Modern weight-decay scaling results (Power Lines; Wang & Aitchison; Moonlight for Muon) say long runs of small models want a shorter τ (R3).
6. **Ranked proposals** (detail in §3):

   | rank | proposal | status | expected gain (val) | GPU | risk |
   |---|---|---|---|---|---|
   | R1 | Cooldown-branch members from saved plateau snapshots (flags only) | new mechanism, measured | direct +0.0001..+0.0003; cost per member ÷4 | pilot 1.5 h; 12 members 6 h | low |
   | R2 | Augmented exploration-pool cooldown (candidate drop + history-event drop on pool batches) | new | +0.0005..+0.0015 per member if it unlocks the pool (p ≈ 0.3) | pilot 2-2.5 h | medium |
   | R3 | wd 0.05/0.1 and dropout (attention 0 / 0.2) screens | partial (never run) | ±0.001 single; hyper-diverse families | 4 h | low |
   | R4 | Candidate-side TabR retrieval values + similarity-biased set attention | new | 0..+0.0008; diversity | 2 h | low-medium |
   | R5 | A4 enrichment: product term, retr_m 32 | partial | 0..+0.0008 | 2 h | low |
   | R6 | Target-action-only key view in cross block 2 | new in the tf arch | 0..+0.0005; diversity | 1 h | low |
   | R7 | Hyper-branches: cooldown branches with varied mix / dropout / wd / aux | new; depends on R1 | +0.0002..+0.0005 over seed-only branches | 30 min per member | low |
   | R8 | Structure variants as diversity families (hist_layers 2, SwiGLU/RMSNorm, GDCN head, dual-softmax head) | partial | ~0 single; diversity only | 1 h per screen | low |

7. **Proposed morning order:**
   1. R1 pilot together with the R2 pilot. They share the A0 arm, all 5-6 runs use the continuation harness, and nothing needs code except the 20-line R2 flag.
   2. R3 screens (8 × 2-epoch).
   3. R4 / R5 screens.
   4. Then grow whatever passes, using branches.
   - Also: add `--save_steps 50000,60000,70000,77500` (the flag exists, train.py:131,332-334) to every future 10-epoch launch. F10-F12 do not save snapshots (`nn/run_final10.sh:12-15`), so they cannot be branched later.

---------------------------------------------------------------------------------------------------------------------

## 1. New CPU evidence (information only)

### 1.1 Cooldown branches vs seeds [E]

Script: `.../scratchpad/r6/external/branch_corr{,_hold}.py`. Method: within-feed z (`stack_val.feed_rank_z`), Pearson correlation over all candidates, and NDCG (`src/metrics.ndcg_per_feed`) of the equal z-sum of the pair. "Gain" = pair NDCG − mean single NDCG, ± per-feed SE. Train-only predictions.

| pair | relation | val corr | val pair gain | holdout corr | holdout pair gain |
|---|---|---|---|---|---|
| B: v15_cd_all_e10 vs v13_lr2_mix1 | same seed-23 trajectory to ~78.7k; two separate cooldowns (a 1-epoch continuation from @80k with optimizer reset, selected at +7k; the run's own 20% cooldown, selected at 82.5k) | **0.9437** | **+0.00188 ± 0.00044** | **0.9476** | **+0.00203 ± 0.00024** |
| C: v12_all_e10 vs v15_cd_all_e10 | plateau @80k vs its own cooled continuation | 0.9632 | +0.00146 | 0.9650 | +0.00158 |
| C2: v12_all_e10 vs v13_lr2_mix1 | plateau @80k vs the same trajectory's cooled end | 0.9465 | +0.00159 | - | - |
| A: v17_pre2s_s101@77500 vs v17_pre2s_s101 | F7 plateau snapshot vs cooled selected | 0.9587 | +0.00169 | 0.9606 | +0.00142 |
| A: same for s102 | | 0.9484 | +0.00197 | - | - |
| D: v15_cd_all_e10 vs v15_cd_all_s81 | independent seeds, cooled continuations | 0.9471 | +0.00175 | 0.9526 | +0.00187 |
| D: v12_all_e10 vs v12_all_s81 | independent seeds, plateau checkpoints | 0.9400 | +0.00184 | - | - |
| D: v17_pre2s_s101 vs s102 | independent seeds, F7 cooled | 0.9365 | +0.00249 | 0.9380 | +0.00225 |
| E: v13_lr1_mix1 vs v13_lr2_mix1 | same seed, lr 1e-3 vs 2e-3 from step 0 | 0.9483 | +0.00177 | 0.9513 | +0.00172 |

**Reading:**
- Two cooldowns from one trajectory (B) are as different as two seeds (D), on both label sets.
- Even a plateau checkpoint and its own cooled child (A, C) correlate only 0.948-0.965.
- Contrast with F8 (2-epoch, fully annealed to 3% lr, last step selected): its seeds correlate 0.978 (SOLUTIONS_REVIEW §3).
- Suspected mechanism: diversity is set by the gradient noise around the selected checkpoint. 10-epoch members are selected at moderately high LR, 1-9k steps into the cooldown (curves_report: "sel in cooldown 7/7" for F7). It is not set by trajectory length.

**Caveat:** pair B differs in cooldown schedule and optimizer-state reset as well as noise. Two branches with an identical schedule may correlate more. The R1 pilot measures exactly this.

### 1.2 Late-cooldown memorisation (cited, not re-derived)

`nn_x/r6/recap/curves_report.txt` gives family means for "end − selected":

| family | val | holdout |
|---|---|---|
| F6 (lr 1e-3, mix 0.1) | −0.00424 | −0.00439 |
| F9 (lr 1e-3, mix 0.1, pre2s) | −0.00441 | −0.00533 |
| F7 (lr 2e-3, mix 0.1) | −0.00200 | −0.00308 |
| F5 (mix 0.3) | −0.0269 | −0.0282 |

My read of `nn/logs/v13_lr1_mix0.log` (mix 0, same seed as v13_lr1_mix1):
- val 0.39433 at 85k → 0.39361 at 98.35k; holdout 0.31845 → 0.31751.
- v13_lr1_mix1 drops 0.39389 → 0.38965 and 0.31828 → 0.31338.
- The AdamW cosine run v9_e10 selects its last step.

So the late decline is driven by the pool, even at mix 0.1. The project knows this qualitatively (CURRENT_STATUS 17:20: "Pool memorisation appears even at mix 0.1 late in the cooldown"). No input-side remedy has been tried.

### 1.3 Hyperparameter tally [E]

Values found in the `args` of 444 JSON logs (224 stage-1 runs, 220 fine-tunes):

| knob | values seen in stage-1 runs |
|---|---|
| d | 128 ×222, 192 ×1, 256 ×1 |
| heads | 4 ×221, 8 ×2, 6 ×1 |
| blocks | 2 ×221, 3 ×3 |
| hist_layers | 1 ×222, 2 ×2 |
| **drop** | **0.1 ×224** |
| **wd** | **0.02 ×444** |
| **ff** | **2 ×222** |
| **cand_budget / max_feeds** | **32768 / 512 ×224** |
| **max_hist** | **256 ×224** |
| **retr_m** | **16 ×62** |
| **cooldown_frac** | **0.2 ×83** (1.0 ×9 for continuations) |
| cooldown_mix | 0 / 0.1 / 0.3 |
| lr | 1e-3 / 2e-3 |
| aux_tau | 10 (30 ×2) |
| aux_pos | 0.5 (1.0 ×2) |
| arch | tf / gru / din |

---------------------------------------------------------------------------------------------------------------------

## 2. Gap analysis: every requested topic vs COVERAGE

Status: **C** = covered (tested or closed with evidence); **P** = partial (proposed/parked or old recipe only); **N** = new.

| # | technique / source | status | where it is covered / what is missing | verdict |
|---|---|---|---|---|
| 1 | OTTO, H&M, KDD Cup 2023 re-rankers | C | deep_otto_hm.md §1-7: co-vis, w2v/CF scores, generator ranks, negatives, objective ensembles. All transferred, as co-vis/LUO/ctx features and objective tests. | nothing left. One reusable fact: OTTO 1st averaged 9 LGBMRankers with *different hyperparameters* (deep_otto_hm §1.8), which supports R3/R7 |
| 2 | RecSys Challenge 2020-2024 | C | deep_recsys_engagement.md, deep_recsys2024.md (RecSys'24 1st transformer, 2nd BlackPearl) | nothing left |
| 3 | RecSys Challenge 2025 | C/P | 4th place (Yandex) in deep_ru §8.3. 1st (rec2) has no public write-up; 2nd (ai_lab_recsys) is a GRU sequence autoencoder for universal user representations [S] | a user-representation task. Our pre2 encoders already play that role. Nothing to transfer |
| 4 | Avito ML Cups 2024-2026, Avito tech blog | C | deep_ru §2-5. The production L3: one transformer over [history; candidates], candidates attend to each other (+0.6%), DCN branch, ListMLE, **history filtered to target actions, 50 events, clicks removed** | ListMLE is closed (−0.0052, sol_solvers). The candidate self-attention exists. The filtered-history idea is untested in our tf arch → R6. The DCN branch exists in F11 (DIN arch) → R8 |
| 5 | Yandex Cup / VK RecSys 2024 / ARGUS | C | deep_ru §6, §8: pretrain-then-finetune, SwiGLU/ALiBi/RMSNorm, ALS-initialised IDs, seed averaging over splits | pretraining = pre2s (adopted). ID tables are closed (N4 −0.0022). SwiGLU/RMSNorm → R8 |
| 6 | MTS / ODS / Sber cups | C | deep_ru §9: no ranking-NN technique beyond co-occurrence + GBDT | nothing |
| 7 | PRM / SetRank / DASALC / RankFormer | C | deep_neural_rankers §2. Our model *is* a set transformer; set attention measured neutral (nn_report:68) | see 11 |
| 8 | MIR (SIGIR'22) set-to-list interaction (history attends to the candidate set) [S] | P | not in prior reports. The hand-built `ctx_hs_*` / `ctx_hn_*` "candidates competing for history items" features are the fixed-parameter version and the best single features in the project (feat_ctx_.md:14). Adding them in-model gave holdout +0.0014, val 0.0000 (nn_report:266) | a learned dual-softmax head is low value (R8) |
| 9 | HSTU pointwise SiLU attention, relative attention bias | C/P | SiLU pooling channel is in the DIN family (F11, running). rab ≈ learned recency embedding; `sim_r` table neutral (nn_report:61); FoX null (nnx_VERDICT §5). Stochastic Length is efficiency-motivated; ULTRA-HSTU reports throughput, not quality [S] | nothing new |
| 10 | DIN / DIEN / SIM / TWIN / ETA / TWIN-V2 | C | deep_neural_rankers §3, nnx_research_recsys_arch §2.2. GSU stages are unnecessary at ≤256 events; GRU (≈DIEN encoder) is F10. TWIN-V2 `ln n` duplicate bias parked (nnx_design §8) | low; not proposed |
| 11 | Candidate-candidate interaction in sets | P | set attention without any similarity prior (neutral); ctx_nd_* kernel features (cvnd adopted); twin disambiguation has "no residual signal" (sol_errors:152) | **R4**: an A4-style difference-vector retrieval over the feed's own candidates plus a similarity bias in set attention. Skeptical prior because of the twin result |
| 12 | Multi-scale history (recent-window branch) | P | in-model smax8 / smax32 / recency embedding; FoX null; EWMA features null (nnx_VERDICT §5) | a separate recent-window branch duplicates these. Not proposed. The typed view (R6) is the more distinct variant |
| 13 | FT-Transformer / feature tokenisation | P | PLE numeric embeddings rejected (nnx_VERDICT B1); RankMixer-style group tokenizer parked (nnx_research_recsys_arch P7) | low; not proposed (PLE failure mode: fits production step functions) |
| 14 | SwiGLU / RMSNorm / pre-norm | P | already pre-LN; SwiGLU/RMSNorm/small-init bundle parked (nnx_design §7). Gains at our scale are ≈0 (LLM ablations: T5 log-ppl 1.997 → 1.944) | R8, as a diversity variant only |
| 15 | Attention dropout / stochastic depth | P | attention-probability dropout 0 parked (nnx_design §8); stochastic depth rejected (5 sublayers). Dropout never varied (§1.3) | **R3** |
| 16 | Gated attention, QK-norm, SSMax, FoX, MoE, differential attention | C | nnx_VERDICT (gate adopted; others null / rejected) | nothing |
| 17 | Retrieval / TabR values | C/P | A4 adopted (+0.0028 with gate); 90% of the gain is in `T(k_c − k_j)` (nnx_exp_attn:99); m never varied | **R5** (enrichment), **R4** (candidate side) |
| 18 | Mixture-of-denoisers / masked-history auxiliary | C/P | pre2 sequence encoders (adopted as pre2s inputs); TransAct-V2 next-action loss parked (nnx_design §8); propensity head null (B2) | low; not proposed |
| 19 | Embedding-table tricks for cold items (DropoutNet, hashed IDs, semantic IDs, meta-warm-up) | C | N4 hashed IDs −0.0022; semantic IDs null (B3); uniform / age-conditioned ID-input dropout = DropoutNet-style, closed as a warm/cold trade (ROUND5 §2a) | closed |
| 20 | Sequence augmentation (crop / mask / reorder / slide-window) | **N** (never run) | listed "low" in nnx_research_training_gen §4.2; history-length TTA rejected; TTA variants in flight (r6_tta) | **R2** (pool batches, where memorisation is measured). The DA4Rec ranking of operators: slide-window ≫ crop > subset > delete; mask/replace hurt on short sequences. Slide-window's gain is an artefact of next-item training and does not apply here |
| 21 | List / candidate-subset augmentation for listwise rankers | **N** | none in the project | **R2** |
| 22 | Large-batch training with Muon | P | batch never varied; GNS analysis in flight (recap_gns). Muon keeps data efficiency past the critical batch (Essential AI 2025) [S] | throughput only. Caution: less gradient noise may raise seed correlation (the F8 lesson), so check member correlation before adopting |
| 23 | Weight decay scaling (τ = 1/(ηλ)) | **N** for us | wd never varied | **R3** |
| 24 | Muon details (wd for long runs, update RMS) | C/P | Muon adopted with match_rms_adamw; wd 0.02 untested | in R3 |
| 25 | Checkpoint soups / WiSE-FT | C | stage-1 EMA null (C1); fta EMA adopted; plateau SWA in flight (recap_swa). WiSE-FT itself: weight-space and output-space ensembles both give the favourable trade-off; output-space costs 2x compute (Wortsman et al., Fig. 10 / §F) [V]. Our 0.5 z(FT) + 0.5 z(TO) is the output-space version | closed. **Do not soup branches**: that removes the diversity R1 exploits |
| 26 | WSM / checkpoint merging instead of the decay | C/P | C1 null at 2 epochs; recap_swa in flight | wait for recap_swa |
| 27 | Snapshot / cyclic ensembles (Huang 2017; q0 "hyper-epoch" 2026) | **N** | F5cd / F6cd are single cooldown continuations; never branched; F7p (uncooled snapshots) fails the strength gate | **R1**. q0 [V]: at a 4-epoch budget, 4 cooled snapshots of one run reach loss 3.360, vs 3.501 for two independent 2-epoch runs |
| 28 | Multi-seed best practice (count, diversity) | C | sol_ensemble §2.4-2.6 (a − b/k, strength gate, family law); F8 lesson | add §1.1 (branches ≈ seeds) |
| 29 | Hyperparameter ensembles (Wenzel 2020); divergent-training diversity (Gontijo-Lopes 2022) [S] | P | used only as lr / mix / recipe families (F6, F7, F9) | **R7** (cheap through R1), R3 |
| 30 | Seed averaging | C | every pick | - |
| 31 | Rank averaging | C | z-mean beats rank mean (−0.0020) and RRF (−0.0007) (sol_ensemble §2.1) | closed |
| 32 | TTA (MC-dropout, history dropout / crop, permutation) | C/in flight | r6_tta.md (pre-declared); permutation TTA is useless (the model is equivariant) | wait |
| 33 | Distillation (self / ensemble teacher) | C | C5 KD n.s., neutral inside the combo; it raises correlation with the teacher (nnx_VERDICT §5) | closed for an ensemble-final pipeline |
| 34 | Losses (LambdaLoss, NeuralNDCG, ListMLE, PL, focal, ...) | C | sol_solvers §2.1 (16 losses); K9 variants (sol_gpu) | closed |

---------------------------------------------------------------------------------------------------------------------

## 3. Ranked proposals (NEW / PARTIAL only)

GPU cost units (task brief): 2-epoch screen ≈ 30 min/run on a free GPU; 10-epoch ≈ 2 h. Measured cost of a 1-epoch cooldown continuation: 19-28 min (`v15_cd_*`, `v21_cd_*` logs). Measured cost of an n2cpv3 fta fine-tune: about 8.5 min (`v21_cd_v14_s92_n2cpv3_ft.log`).

All code changes go in a copy (nn_x/r6 or a new nn_x/<label>), with outputs redirected, per the house rules.

### R1. Cooldown-branch members from saved plateau snapshots (flags only)

**Mechanism.** §1.1: the seed-specific component of a 10-epoch member is mostly created in the early, noisy part of the cooldown. Branching K cooldowns from one plateau snapshot, with different data-order seeds, yields K seed-like members for about 1/4 of the cost. External support:
- snapshot ensembles (Huang et al. 2017) [K];
- q0 (2026): in a data-constrained multi-epoch regime, cooled snapshots of cyclic schedules beat independently trained ensembles at equal compute [V];
- F5cd and F6cd already show that a continuation member is +0.0013..+0.0018 stronger than its plateau parent (paired_report).

**Spec.** No code: `nn/run_f6cd.sh` is the template (train.py `--init` loads cfg and weights, train.py:171-199; seed sets data order, pool order and dropout, train.py:142-150, 233, 261).
```
train.py --variant v27_br_s101_77k_s811 --init v17_pre2s_s101@77500 --epochs 1 --lr 2e-3 --warmup 0 --sched wsd \
  --cooldown_frac 1.0 --cooldown_mix 0.1 --aux_anneal 0.1 --muon --aux_pos 0.5 --aux_mode listwise --aux_tau 10 \
  --holdout 30000 --select_on holdout --eval_every 1000 --seed 811
# then: EXTRA=n2c_pre_v3 SUF=n2cpv3 FTARGS='--decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton' ft_list.sh; holdout_eval.py
```
- Roots available now: `v17_pre2s_s10{1..7}@{60000,70000,77500}.pt` (21 checkpoints).
- Use lr 2e-3, the F7 plateau lr, so there is no re-warm.
- For future families, add `--save_steps 50000,60000,70000,77500`.

**Pilot** (pre-declared; 3 runs, can share A0 with R2):
- b1 = root s101@77500, seed 811;
- b2 = root s101@77500, seed 812;
- b3 = root s101@60000, seed 811.

**PASS** iff all of:
- corr(b1, b2) ≤ 0.955 on val train-only **and** on the holdout;
- pair gain (b1 + b2) ≥ +0.0015 val;
- each branch's single train-only val ≥ 0.3926 and holdout ≥ 0.3171 (F7 means − 0.002).

**If PASS:**
- F7b = 12 branches: s101-s106 × {@60000, @70000}, one seed each.
- It enters as a family through gates (a)+(b).
- Pre-declared candidate = current holdout-selected pool + F7b, member-equal, chosen on the holdout.
- It replaces v12 only under the frozen rule.
- Afterwards, the same recipe grows any architecture family that passes, provided its launches saved snapshots.

**Expected gain:**
- direct, on the current 25-member pool: +0.0001..+0.0003 val (a − b/k);
- indirect: about 4x cheaper members for R3, R4 and R7 families.

**GPU:** pilot about 1.5 h (3 × 30 min); F7b about 6 h.

**Risk:**
- (i) identical-schedule branches may correlate more than pair B. The pilot decides.
- (ii) Branches share their root family's structured component, so they add seed-like diversity, not recipe diversity.
- (iii) Each branch is holdout-selected (about 10 evaluations), the same exposure as F5cd and F6cd.

### R2. Augmented exploration-pool cooldown (anti-memorisation)

**Mechanism:**
- The 134,118-feed exploration pool is the most valuable training data per feed: +0.0031 val per feed vs production feeds (sol_solvers TL;DR 3).
- It is memorised in the late cooldown even at mix 0.1 (§1.2), so at 10 epochs it adds about 0 (mix 0 ≥ mix 0.1 at the selected step).
- At 2 epochs the mix was worth about +0.0018 (sol_solvers TL;DR 1). Label smoothing did not help (sol_solvers §2.4).
- Input augmentation is the standard remedy for memorising a small set seen repeatedly:
  - DA4Rec [V]: for SASRec, crop gives +19..+60% R@20. "The less training data available, the more significant the relative improvement." Mask and replace hurt on short sequences.
  - Per-epoch candidate sampling per query helps listwise neural rankers (Rahimi et al. 2019) [S]. CoRNStack and REARANK sample candidate subsets per query [S].

**Spec** (copy of train.py, about 20 lines, inserted after `b = tr.gather(fb, with_pos=True)` and before `X = model.dense(b)`, train.py:281-282). Flags `--aug_cand q_min` (default 1.0 = off), `--aug_hist p` (default 0), `--aug_all`:
```python
if (is_pool or a.aug_all) and a.aug_cand < 1.0:     # keep all positives; keep each negative w.p. q ~ U(q_min, 1) per feed
    q = a.aug_cand + (1 - a.aug_cand) * torch.rand(b["y"].shape[0], 1, device=dev)
    keep = (torch.rand_like(b["y"]) < q) | (b["y"] > 0)
    b["cmask"] = b["cmask"] & keep; b["y"] = b["y"] * b["cmask"]
if (is_pool or a.aug_all) and a.aug_hist > 0:       # drop each history event w.p. p (recency positions kept)
    hk = torch.rand(b["hmask"].shape, device=dev) >= a.aug_hist
    b["hmask"] = b["hmask"] & hk; b["hemb_ok"] = b["hemb_ok"] & hk
```
- **Consistency:** every in-model input is recomputed from the masks: within-feed z/pct, cnt / log M, feed category share, history-similarity maxima, category counts, retrieval, set and target attention (model.py:55-132, 416-489), and the loss masks (train.py:285-307).
- K9 has ≈0 weight on pool feeds (1 − p_random, train.py:306).
- Still stale: precomputed xt columns (`ctx_nd_*`, `ctx_hn_*`, `ctx_hs_*`, `seq_cv_*`, `pre2_hmax32`) still describe the full feed and history. This is a mild input-noise component.
- `b["H"]` (the log H feature) is left as is. Use with arch tf only.

**Pilot** (continuation harness as R1; roots s101@77500 and s102@77500, seed 811):
- A0 = mix 0.1, no aug (shared with R1 b1);
- A1 = mix 0.3, no aug (memorisation stress);
- A2 = mix 0.3, aug_cand 0.6, aug_hist 0.15.

**PASS** iff:
- A2 − A0 paired train-only val at the selected checkpoints is > 0 for both roots, and > 2 SE pooled;
- holdout ≥ 0.

Diagnostic (reported, not decisive): A2's end-of-run drop (last − best) ≤ half of A1's.

**If PASS:** one 10-epoch pair, cvnd_pre2s, recipe (ii) + `--aug_cand 0.6 --aug_hist 0.15 --cooldown_mix 0.2`, seeds 101/102 paired against F7 s101/s102. Then a family.

**Expected:** +0.0005..+0.0015 per member if it restores the 2-epoch pool value (p ≈ 0.3), plus a diversity bonus from the different data path.

**GPU:** pilot +4 runs (A1, A2 × 2 roots) ≈ 2-2.5 h; 10-epoch pair 2 × 2.2 h.

**Risk:**
- medium: the stale xt features;
- the pool's M distribution shifts down in augmented batches, and M<50 is invisible to the holdout, so report M slices;
- a cold/warm trade is possible, so report the cold slices and the stale-val replay before adoption.

### R3. Regularisation hyperparameters never varied: weight decay and dropout

**Mechanism:**
- (a) wd. The AdamW/Muon EMA timescale is τ = 1/(ηλ): 50k steps = 5.1 epochs = 0.51 of a 10-epoch run at lr 1e-3; 2.5 epochs (0.25) at lr 2e-3.
  - Wang & Aitchison (ICML 2025): the optimal τ measured in epochs is roughly constant and should not be much below 1 epoch [S].
  - Power Lines (NeurIPS 2025): the optimal τ as a fraction of training falls as TPP^−0.5, ≈0.21 at 20 tokens/param, toward ~0.01 at 1000 [V]. Over-trained small models want stronger decay.
  - Moonlight: weight decay is "crucial" for Muon in long runs, because weights grow too large [S].
  - Our model trains a 0.86M-parameter network for 10 passes over 247M candidates, i.e. deep in the over-trained regime. LLM numbers do not transfer quantitatively, so this is a direction, not a value.
- (b) Dropout. Attention-probability dropout zeroes the single matched history event 10% of the time in the retrieval-like sim-bias heads (nnx_research_training_gen §4.2). Liu et al. (ICML 2023): dropout timing matters, with late dropout for over-fitting regimes [S].

**Spec.** 2-epoch harness, paired with the existing `sx_base_s23/s24` controls in nn_x/solvers (cvnd_pre2s, lr 1e-3, mix 0.1). 4 arms × 2 seeds:
- (i) `--wd 0.1`;
- (ii) `--wd 0.05`;
- (iii) attention-probability dropout 0 with residual dropout 0.1. Code in a copy: `Attn.__init__(..., attn_drop=None)` and `dropout_p = (self.drop if self.attn_drop is None else self.attn_drop) if self.training else 0.0` at model.py:254, plumbed through the Ranker cfg and `--attn_drop`;
- (iv) `--drop 0.2`.
- Note: the wd name filter also decays the `cat/et/es/rec/dom` tables (train.py:223). Keep it unchanged for pairing and report it.

**Acceptance:**
- As a recipe change: the unified rule (train-only paired val > 2 SE over 2 seeds, holdout ≥ 0, slices reported).
- As a family candidate (hyper-deep ensembles): |Δ| < 0.002 and member correlation vs control ≤ 0.96.
- Because the wd effect should grow with τ/T, a 2-epoch null does not close wd at 10 epochs. Run a 10-epoch confirmation pair only if the 2-epoch Δ ≥ 0.

**Expected:** ±0.001 single; P(Δ > +0.0008) ≈ 0.25.

**GPU:** 8 × 30 min = 4 h (2 per GPU).

**Risk:** low.

### R4. Candidate-side retrieval values + similarity-biased set attention

**Mechanism:**
- A4's gain comes from the difference vector between a candidate and its top history matches (nnx_exp_attn:99).
- The same question asked against the feed's *own* competitors (same-topic alternatives, near-duplicates) currently reaches the model only through:
  - fixed kernel means (`ctx_nd_knn_*`);
  - a set attention that has no similarity prior. By comparison, the history-side sim bias alone was worth +0.0024 (nn_report:68).
- Skeptical note: twins (19.8% of positives have a same-feed negative at cos ≥ 0.95) show "no signal can break twin ties beyond v11" (sol_errors:152). The upside is therefore in non-twin competitors, and the prior is modest.

**Spec** (copy of model.py, about 50 lines; flags `--cretr_m 8 --set_sim_bias`):
- `CandRetrieval` mirrors `HistRetrieval` (model.py:151-197):
  - `Scc = einsum('bmd,bnd->bmn', ce, ce)`, with diagonal, padded and missing-embedding candidates set to −2; top-m;
  - logits `−‖k_i − k_j‖² + a·S_ij` plus a sink;
  - values `LN(E_rank[bucket8(pct_f1_j)] + T(k_i − k_j))`, where pct_f1 = raw `X[..., 2]` captured before standardisation (model.py:419);
  - output `Z(Σ p v)`, with Z zero-initialised under `extra_init()`;
  - added to `c` right after the history retrieval (model.py:474-475).
- Set-attention bias: build `self.sets` with `sim_bias=1`. `bias_scale` is a deterministic linspace, so the other parameters' init RNG stream is unchanged. Call `self.sets[i](c, cmask, bias=Scc)` at model.py:489. `Block.forward` already forwards `bias`, and the `Attn` sim_bias=1 path masks padded keys (model.py:242-244).
- Leak-free: same-feed candidates only, all known at request time (as in ctx_nd).

**Screen:** 2-epoch pairs vs sx_base for {cretr} and {cretr + set_sim_bias}, 4 runs. Acceptance: the unified rule, or the family-candidate rule from R3.

**Expected:** 0..+0.0008 single, plus architecture diversity.

**GPU:** 2 h.

**Risk:** low-medium (memory B×M×M is ≤ 300×32768 entries per batch, fine).

### R5. A4 enrichment: product term and m

**Spec:**
- (a) `--retr_m 32`: flag only, never varied.
- (b) `--retr_prod`: add `self.t3(kc[:, :, None, :] * khg)` to the HistRetrieval values, with `t3 = Linear(dk, d)` zero-initialised under `extra_init()`. This is DIN's activation-unit input [e_h, e_c, e_h − e_c, e_h ⊙ e_c] [K]; A4 has only the difference.

**Screen:** 4 runs, 2-epoch pairs.

**Expected:** 0..+0.0008. **GPU:** 2 h. **Risk:** low.

### R6. Target-action-only key view in one cross block

**Mechanism:**
- Avito's production L3 filters history to target actions and drops clicks "so they would not crowd out intent signals" (deep_ru §2.1).
- ULTRA-HSTU separates engagement and consumption modules to avoid "signal competition" (nnx_research_recsys_arch §2.2).
- Later-history matches of positives are 87% event_type 1 / source 1 (CURRENT_STATUS:59).
- Partial coverage today: per-type similarity maxima, `ctx_hs_e1`, and the per-type sim scale of sim_bias=2 (neutral at 10k steps, old recipe).

**Spec:** in `Ranker.forward`, `kt = kmask.clone(); kt[:, 1:] &= (b["het"] == 1)`. Use `kt` in place of `kmask` for `crosses[1]` only (model.py:486-487). The user token always stays valid as the sink. Flag `--typed_cross`.

**Screen:** 2 runs.

**Expected:** 0..+0.0005 single, plus diversity. **GPU:** 1 h. **Risk:** low.

### R7. Hyper-branches (depends on R1 passing)

**Mechanism.** Error correlation falls as training methodology diverges (Gontijo-Lopes et al., ICLR 2022 [S]). Ensembles over hyperparameters beat deep ensembles (Wenzel et al., NeurIPS 2020 [S]). Through R1, each such member costs about 30 min.

**Spec.** Branches from the same F7 roots with different cooldown recipes:
- `--cooldown_mix 0` vs 0.1;
- `--aux_anneal 0.5` (constant K9 weight: K9 acts as trunk regularisation, CPU3) vs 0.1;
- `--wd 0.1`;
- cooldown dropout {0, 0.2}. This needs `--drop_override`: `ck["cfg"]["drop"] = a.drop_override` before `Ranker(**ck["cfg"])` at train.py:197.

Each variant enters only through gates (a)+(b) as a sub-family. Report its correlation with seed-only branches.

**Expected:** +0.0002..+0.0005 over seed-only branches if correlation falls to about 0.93. **Risk:** low.

### R8. Structure variants as diversity families (only after R1 makes members cheap)

Each is a 2-run screen and enters only if it passes the family gates:
- hist_layers 2 under Muon (ULTRA-HSTU: self-attention depth keeps paying; the old AdamW test was neutral);
- SwiGLU + RMSNorm + small-init hygiene bundle;
- GDCN / DCNv2 parallel head on the tf arch (Avito production; already inside F11);
- dual-softmax "competition" head, i.e. a learned `ctx_hs` (MIR-style set-to-list).

Expected single-model effect is ≈0 for each, so their only value is lower correlation.

---------------------------------------------------------------------------------------------------------------------

## 4. Hyperparameter and model-structure audit (user request)

### 4.1 Hyperparameters

| knob | current | varied? (§1.3) | evidence | recommendation |
|---|---|---|---|---|
| d / heads | 128 / 4 | 256/8 once (AdamW, worse); 192/6 = F12 running | capacity flat under AdamW (nn_report:53,68); Muon raises c_lin effective rank 30 → 79 (review_ideas #12) | wait for F12; if it passes the gates, grow it by branching. This needs snapshots, so add `--save_steps` to the remaining F12 launches |
| blocks / hist_layers | 2 / 1 | 3 blocks ×3, 2 hist layers ×2 (old recipe) | neutral | R8 (hist_layers 2), low |
| ff | 2 | never | FFN is a small share of compute and parameters | keep |
| **drop** (residual + attention probabilities) | 0.1 | **never** | §3 R3 | **R3** |
| **wd** | 0.02 | **never** | τ = 5 epochs at lr 1e-3, 2.5 at 2e-3 | **R3** |
| lr | 1e-3 / 2e-3 | yes | F7 (2e-3) vs F9 (1e-3) singles: val +0.00078 (z_run 1.4), holdout +0.00067 (z_run 2.9) (paired_report); F7 also gets the larger cooldown gain (+0.0031 vs +0.0005) | prefer lr 2e-3 for new pre2s families |
| warmup | 500 | never | irrelevant at 98k steps | keep |
| batch (cand_budget / max_feeds) | 32768 / 512 | never | GNS in flight (recap_gns) | wait. Larger batches lower the noise that creates member diversity (§1.1); check member correlation before adopting |
| max_hist | 256 | never | train histories are deeper than val/test (INTEGRITY-1, REVIEW_SYNTHESIS:85); crop TTA in flight | keep; R2's `aug_hist` / `--aug_all` covers the augmentation side |
| retr_m | 16 | never | A4 core | R5 |
| cooldown_frac | 0.2 | never | selection lands 1-9k steps into a 19.7k cooldown, and the rest declines (§1.2) | low: cooldown_frac 0.1 would end near the peak and save about 10% of a run. Only if R2 fails (with R2, a longer cooldown may become useful) |
| cooldown_mix | 0.1 | 0 / 0.1 / 0.3 | 0.3 collapses; 0.1 declines late; 0 is flat | R2 |
| aux (pos / tau / anneal) | 0.5 / 10 / → 0.1 | tau 30 worse, weight 1.0 null; anneal never isolated | K9 = trunk regularisation (CPU3) | constant-aux branch in R7 |
| eval granularity | 2500 steps | - | cooldown peaks are about 2.5-5k steps wide (curves_report) | eval every 1000 steps in the cooldown is possible but adds holdout-selection optimism; not recommended |
| optimizer internals (Muon momentum 0.95 / Nesterov / 5 NS steps; AdamW betas 0.9 / 0.98; clip 1.0) | fixed | never | standard values | keep |

### 4.2 Model structure: verdict per component

| component | measured value | external view | action |
|---|---|---|---|
| history target attention + sim bias | largest single component (−0.006..−0.010 without it; sim bias +0.0024) | the industry consensus: DIN / TWIN / TransAct / HSTU | keep |
| A4 retrieval values (top-16, difference vectors) | +0.0019..+0.0028 | TabR | enrich (R5) |
| gated attention | small, never negative | Qwen G1 / HSTU U-gating | keep |
| candidate set attention (2 blocks) | neutral in the old recipe (nn_report:68) | Avito production +0.6%; PRM / SetRank | give it a prior (R4) or treat it as a diversity knob |
| 1-layer history encoder | old-recipe depth neutral | ULTRA-HSTU: depth pays at 6B samples | R8 |
| head (MLP + wide) | - | DCN branch in Avito production; F11 has DCNv2 | R8 |
| inputs (cvnd_pre2s, 95 dense + xt) | saturated (FEATURES_REVIEW) | - | none |
| K9 production-order distillation | +0.0024, locally optimal | - | none |
| Muon + WSD + early-cooldown selection | adopted | the noise at the selected point = the member diversity (§1.1) | use it: R1 / R7 |

---------------------------------------------------------------------------------------------------------------------

## 5. Do-not list added by this research

- **Do not weight-soup cooldown branches or seeds.**
  - Soups remove the noise diversity that §1.1 shows is the ensemble's raw material.
  - WiSE-FT reports weight-space ≈ output-space ensembling (output-space costs 2x compute), and we already pay for output-space.
- **Do not adopt a larger batch on throughput grounds alone.**
  - Lower gradient noise can raise member correlation, like the fully annealed 2-epoch F8 seeds (0.978).
  - Check correlation first.
- **Do not port slide-window augmentation.**
  - Its DA4Rec gains (+79..+277% R@20) come from adding next-item training targets, which our feed task does not have.
- **Do not re-cool already-cooled checkpoints as "new" members.** That is the near-duplicate case SOLUTIONS_REVIEW warned about. Branch from pre-cooldown snapshots instead.
- **Do not judge R2 on val alone.**
  - It changes the effective M distribution and the cold handling.
  - Report M<50 / M≥50, cold slices, and the stale-val train-only replay (CPU2 harness) before any adoption.

---------------------------------------------------------------------------------------------------------------------

## 6. Expected outcome if executed

**Single-model side:**
- R2 and R3 are the only proposals with a credible single-model upside of ≥ +0.001.
- My combined probability that at least one passes the unified rule is about 0.4.

**Ensemble side:**
- R1 makes everything else cheaper, but adds only about +0.0001..+0.0003 directly.
- R4-R8 add +0.0002..+0.0005 each only if they pass gate (b).

**Realistic total:** +0.0005..+0.0015 val over the current best pool (P(F6\*) +0.00087 vs v12). That is at or below the MDE for one model generation (≈0.0025, REVIEW_SYNTHESIS §4), so decisions must keep using the pre-declared pooled rules.

---------------------------------------------------------------------------------------------------------------------

## 7. Sources

Read this session [V] or seen as abstract/snippet [S]:
- Bergsma et al., *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training*, NeurIPS 2025 [V]: https://arxiv.org/html/2505.13738v2
- Wang & Aitchison, *How to set AdamW's weight decay as you scale model and dataset size*, ICML 2025 [S]: https://arxiv.org/abs/2405.13698
- Liu et al. (Moonshot), *Muon is Scalable for LLM Training* [S]: https://arxiv.org/abs/2502.16982
- Essential AI, *Practical Efficiency of Muon for Pretraining* [S]: https://arxiv.org/abs/2505.02222
- Wortsman et al., *Robust fine-tuning of zero-shot models* (WiSE-FT), CVPR 2022 [V, Fig. 10 / §F text]: https://arxiv.org/abs/2109.01903
- *q0: Primitives for Hyper-Epoch Pretraining* (2026) [V]: https://arxiv.org/html/2606.03938v2
- Tian et al., *WSM: decay-free LR schedule via checkpoint merging*, ICLR 2026 [S]: https://arxiv.org/abs/2507.17634
- Huang et al., *Snapshot Ensembles*, ICLR 2017 [K]
- Wenzel, Snoek, Tran, Jenatton, *Hyperparameter Ensembles for Robustness and Uncertainty Quantification*, NeurIPS 2020 [S]: https://arxiv.org/abs/2006.13570
- Gontijo-Lopes, Dauphin, Cubuk, *No One Representation to Rule Them All*, ICLR 2022 [S]: https://arxiv.org/abs/2110.12899
- *Is Contrastive Learning Necessary? A Study of Data Augmentation vs Contrastive Learning in Sequential Recommendation* (DA4Rec, 2024) [V]: https://arxiv.org/html/2403.11136v1
- Dang et al., *Data Augmentation for Sequential Recommendation: A Survey* (2024) [S]: https://arxiv.org/abs/2409.13545
- Rahimi, Montazeralghaem, Allan, *Listwise Neural Ranking Models* (ICTIR 2019 poster) [V, poster text]: https://people.cs.umass.edu/~rahimi/posters/LNRM.pdf
- CoRNStack (2024) [S]: https://arxiv.org/abs/2412.01007 ; REARANK (2025) [S]: https://arxiv.org/abs/2505.20046
- Xi et al., *Multi-Level Interaction Reranking with User Behavior History* (MIR), SIGIR 2022 [S]: https://arxiv.org/abs/2204.09370
- Liu et al., *Dropout Reduces Underfitting*, ICML 2023 [S]: https://arxiv.org/abs/2303.01500
- Muennighoff et al., *Scaling Data-Constrained Language Models*, NeurIPS 2023 [S]: https://arxiv.org/abs/2305.16264
- ULTRA-HSTU, *Bending the Scaling Law Curve in Large-Scale Recommendation Systems* (2026) [S]: https://arxiv.org/abs/2602.16986
- RecSys Challenge 2025: 4th place (Yandex) [S]: https://arxiv.org/abs/2508.06970 ; 2nd place sequence autoencoder [S]: https://arxiv.org/abs/2508.07748
- Earlier project reviews: deep_otto_hm.md, deep_recsys2024.md, deep_recsys_engagement.md, deep_ru_competitions.md (Avito Habr 1090584 production L3), deep_neural_rankers.md, nnx_research_{llm_arch,recsys_arch,tabular_dl,training_gen}.md, sol_*.md, nnx_VERDICT.md, REVIEW_SYNTHESIS.md, ROUND5.md.
