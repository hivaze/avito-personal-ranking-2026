# research_1112 / shift: competition winners under distribution shift and public/private shake-ups

Angle: how winners of time-shifted competitions (2023-2026) validated, picked final submissions, ensembled, adapted at test time and survived shake-ups. Each lesson is mapped to our val (weekday) → public (weekend) → private (later, colder) setting.

Author: research agent (research only). No GPU, no training, nothing running was touched, no existing file was edited. Two light CPU checks were run read-only (§1). Their scripts are in `/tmp/scratchpad/shift/`.

Evidence tags:
- **[E]** = measured in the cited writeup or paper.
- **[E-x]** = the number comes from a search-engine extract or a secondary report of the writeup. Kaggle writeup pages render client-side, and automated fetching of them is blocked, so I could not read them directly.
- **[D]** = my own CPU check on project files (§1).
- **[O]** = opinion or inference.

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **The m02 public under-transfer looks like selection inflation plus noise, not weekend shift or staleness [D].** Three read-only checks:
   - On the 30K train holdout, the paired m02 − c0 gain (7 roots) is the same on weekend days (Sep 6/12/13/19) and weekdays. Weekend minus weekday is −0.00012 ± 0.00048. For p70k it is −0.00021 ± 0.00048.
   - On val, M2 − Cd is flat from Tue (3 days after train end) to Thu (5 days): Thu − Tue = +0.00028 ± 0.00105. So no staleness gradient appears inside val.
   - The gain is larger in the morning, not smaller (h<12 +0.0024 vs h≥18 +0.0015). Public is morning-heavy.
   - **The ordering of the public results matches the holdout, not val.** Deltas vs PF7x:

     | candidate | holdout | public | val |
     |---|---|---|---|
     | G | +0.0002 | +0.0001 | +0.0006 |
     | M | +0.00002 | −0.0003 | +0.0012 |

   - The val-only part of M's edge, which was val-selected under a rule declared non-blind, is the part that failed to transfer.
   - Winners' practice and Cawley & Talbot (2010) say the same thing: a small advantage that wins on the selection set should be shrunk.
2. **Under shift, prefer breadth to concentration.** Deep ensembles are the most shift-robust method in Ovadia et al. (2019). Top Kaggle shake-up survivors used plain averages or votes of several models (AES 2.0 1st, CMI-PIU 1st). Our own public transfer ratios agree: 6 families 0.9x, G 0.6x, M (m02 only) 0.2x. **For pick 1, use a broad pool that includes the m02 families (G-like), not M.**
3. **The new training lever is "fresh-data annealing", and nothing in `nn/train.py` does it today.** Every time-series winner gained from training on the latest data:
   - Optiver 1st: GRU MAE 5.469 → 5.426 with 5 online updates [E-x].
   - Enefit 1st: XGB MAE 55.62 → 54.13 [E-x].
   - Jane Street 2nd-place reproduction: R² 0.0143 → 0.0208 with online updates [E].

   LLM practice shows that the data seen in the last 10-20% of a decaying-LR run steers the final model (domain upsampling: +6.9 MMLU, +8.3 GSM8K [E]). We already do the traffic-type version of this (m02). The time-axis version is new:
   - **upsample the last train days (Sep 17-19) in the WSD cooldown;**
   - or run a short low-LR "online" pass on them.

   Internal support: the train-row popularity coverage moves toward test with time. The zero rate of `it_hist` is 17.5% on Sep 6, 29.7% on Sep 19, 37.9% on val and 45.6% on private (review_shift SHIFT-1).
4. **Label-free transfer diagnostics exist, and we have not used them.** Agreement-on-the-line (Baek et al., ICML 2022) estimates OOD performance from how members agree on unlabeled target data. It could answer "noise vs shift" on public and private without spending submissions. Our candidate-to-candidate check costs about 30 CPU minutes.
5. **Things the shift literature says not to do, which are already closed here:**
   - importance or adversarial reweighting of training (Byrd & Lipton 2019);
   - DRO, IRM or CORAL (Wild-Time 2022: none beats ERM under temporal shift);
   - post-hoc cold-item boosts;
   - any LB-tuned constant (Home Credit's "betting" hack, AES 2.0 thresholds).

   This is consistent with COVERAGE_MAP; do not reopen them.

---------------------------------------------------------------------------------------------------------------------

## 1. New CPU evidence [D] (information only; read-only on project files)

### 1.1 Does the m02 gain depend on weekend vs weekday? (`shift/dow_holdout.py`)

- **Holdout reconstruction.** It reproduces `train.py --holdout 30000` exactly (seed 12345, p_random > 0.9, M ≥ 50), and the M vector matches `preds/holdout/holdout_meta.npy`.
- **Days.** The holdout covers all 12 train days, 1.8K-2.9K feeds per day. So **checkpoint selection uses an interleaved in-period holdout, not a time-forward one** (see idea T3).
- **Slices.** Weekend days (MSK) are Sep 6 (Sun), 12 (Sat), 13 (Sun) and 19 (Sat): 9,632 feeds. Weekdays: 20,368 feeds.

Paired deltas, mean over roots 101-107, per-feed SE:

| arm − c0 | all | weekend | weekday | weekend & h<12 | weekday & h≥18 | weekend − weekday |
|---|---|---|---|---|---|---|
| m02 | +0.00134 ± 0.00022 | +0.00126 ± 0.00039 | +0.00138 ± 0.00027 | +0.00058 ± 0.00065 | +0.00134 ± 0.00049 | **−0.00012 ± 0.00048** |
| m02_p70k | +0.00135 ± 0.00022 | +0.00121 ± 0.00040 | +0.00142 ± 0.00027 | +0.00108 ± 0.00064 | +0.00157 ± 0.00046 | **−0.00021 ± 0.00048** |

Reading:
- The in-period m02 gain does not depend on weekend vs weekday.
- The weekend-morning cell (the public-like mix) has a smaller point estimate for m02 (+0.0006), but its SE is 0.00065, and p70k shows +0.0011 in the same cell. This is not evidence.

Caveat: this is in-period (no staleness). It tests the DOW/hour mechanism only.

### 1.2 Does the pool gain shrink with staleness inside val? (`shift/val_day_slices.py`)

Pools are built exactly as in `build_pool.py`: 0.5 z(ft) + 0.5 z(to), per-feed z. Tue Sep 22 = 7,542 feeds, Thu Sep 24 = 6,878 feeds.

| Δ vs Cd (pool) | all | Tue | Thu | h<12 | h≥18 | Thu − Tue |
|---|---|---|---|---|---|---|
| Pall | −0.00005 ± 0.00033 | +0.00001 | −0.00011 | −0.00024 | +0.00036 | −0.00011 ± 0.00066 |
| PF7x | +0.00073 ± 0.00036 | +0.00076 | +0.00070 | +0.00091 | +0.00058 | −0.00005 ± 0.00073 |
| G3 | +0.00131 ± 0.00049 | +0.00160 | +0.00099 | +0.00222 | +0.00078 | −0.00061 ± 0.00098 |
| M2 | +0.00192 ± 0.00053 | +0.00179 | +0.00207 | +0.00238 | +0.00145 | **+0.00028 ± 0.00105** |

The train-only halves show the same pattern. M2 − Cd (TO): Tue +0.00224, Thu +0.00191, h≥18 +0.00345.

Reading:
- Within val, no staleness gradient explains M's public shortfall.
- No hour gradient explains it either. If anything, public's morning-heavy mix should favour M.
- What remains:
  - **public noise.** The paired SE is about 0.0012. M's shortfall vs val is about 0.0016, so z ≈ 1.3.
  - **selection inflation.** M was the val-best of about 5-8 correlated candidates, and decide4 discloses that its rule is not blind.

  The holdout, which was not used to pick M, already called M a tie with PF7x.
- Caveat: Sat/Sun staleness (7-8 days) lies outside val's 3-5-day range, so an effect that starts after day 5 cannot be excluded.

---------------------------------------------------------------------------------------------------------------------

## 2. Source digest (what winners did under shift)

| # | competition / paper (year) | shift type | what they did | evidence | lesson for us |
|---|---|---|---|---|---|
| 1 | **Home Credit – Credit Risk Model Stability**, 1st "My Betting Strategy" (yuuniee, 2024) | later weeks; metric = mean weekly Gini + 88·min(0, slope) − 0.5·std | Found a proxy for WEEK_NUM (date_decision − min_refreshdate, corr ≥ 0.9). Lowered scores of earlier weeks by a constant REDUCE. The public-optimal value was 0.03. **Chose a "neutral" middle value because the private optimum was uncertain (0.02-0.04).** | [E-x] | (a) A metric exploit is not available to us: per-feed NDCG ignores per-feed constants. (b) **For a knob whose best value depends on unseen private conditions, take the middle of the plausible range, not the LB- or val-optimal extreme.** |
| 2 | Home Credit CRMS 9th (team gold, GitHub Dxfory, 2024) | same | A time-aware grouped CV; a 50/50 blend of two pipelines; −0.025 post-hoc on a subset. | [E] (README) | Simple equal blends of independent pipelines were the norm in the gold zone. |
| 3 | **Optiver – Trading at the Close**, 1st (hyd, 2023-24) | private = future market months | Online retraining every 12 days (5 passes); CatBoost + GRU + Transformer; post-processing. GRU 5.469 → 5.4259 with 5 online updates. | [E-x] | The freshest data is worth retraining for. Others found online learning useless on their own backtests, so mixed results are possible. |
| 4 | **Enefit – Prosumers**, 1st (hyd, 2024) | private = 8 later months | Online retraining. XGB 55.62 → 54.13 (1 update) → 54.12 (3 updates); blend 53.66 → 52.31. "Online learning didn't help much in public LB, I think it is crucial in private." Chose low-frequency retraining. | [E-x] | Private gains from fresh data can exceed public ones. The winner trusted the mechanism, not the public delta. |
| 5 | **Jane Street Real-Time Market Data** (2024-25): 2nd Patrick Yam (reproduction by cweill), 8th Evgeniia Grigoreva, 162nd max-abhi | non-stationary; private = future | 2nd: GRU ensembles; a separate Adam optimizer with online LR 1e-4 (offline 5e-4), 3 steps per day on the previous day's labels. Frozen R² 0.01433 → online 0.02078 (17 seeds). 8th: 1 pass per new day at lr 3e-4, "+0.008 CV". 162nd: rolling 3-4-day fine-tune at low LR, **mixing old data against forgetting**. | [E] (2nd repro); [E-x] (8th, 162nd) | Recipe for a fresh-data pass: low LR (0.1-0.2x), few steps, replay of old data. |
| 6 | **Learning Agency Lab AES 2.0**, 1st "Trust CV (and LB a bit)" (flg, 2024) | train = Persuade + "Kaggle-only"; test = Kaggle-only with different grading | **Two-stage: pretrain on old data, fine-tune on test-like data: +0.015.** **Pseudo-label the old data with a model trained on the new data: +0.005.** Simple average of 4 DeBERTa models. 3-seed averaging of everything. Moved from rank 619 to 1; 4th place moved from 1277 to 4. | [E-x] (hippocampus-garden report) | Our TO → val-FT is the same design, which is a confirmation. Re-labelling train with the val-style model is the untested half (idea T5). Huge shake-ups punish LB-chasing. |
| 7 | **LMSYS Chatbot Arena**, 1st "Distill is all you need" (sayoulala, 2024) | pairwise preference | 70B teachers distilled into gemma2-9b; 5-fold training; **LoRA weights averaged across folds**; swap-TTA. | [E-x] | A fold soup is a cheap inference trick (idea L3). The LLM side is closed for us. |
| 8 | **WSDM Cup 2025 Multilingual Arena**, 1st (whitefebruary, 2025) | language/domain mix | Filtered the external data to match the test language mix (50% non-English) and domain categories. | [E-x] | Matching the training data to test descriptors. For us the main mismatch is staleness, which the cooldown data choice can target (idea T1). |
| 9 | **Leash BELKA**, 1st (Victor Shlepov, 2024) + JCIM/bioRxiv generalisability study (2026) | private dominated by unseen building blocks (OOD chemistry) | MLM pretraining plus external data aimed at OOD. The follow-up study: **"increased model size improved performance on the public test set but failed to generalize consistently with the private and more out-of-distribution set."** | [E-x] | Gains measured on the warmer public/val can vanish on the colder private. Our private has the most cold items (0.41). Give a private-coverage emulation veto power (idea D3). |
| 10 | **UM – MCTS game strength**, 1st (jday96314, 2024) | test = unseen games | GroupKFold by game; synthetic and LLM-generated games to widen coverage; GBDT + TabM ensemble. | [E] (repo structure); [E-x] (CV scheme) | Validate on held-out *entities* when the test has new entities. Our analogue is item coldness. |
| 11 | **Stanford Ribonanza**, 1st/2nd (2023-24) | test Lmax 457 vs train 206 | Dynamic positional bias or ALiBi, built for length extrapolation. Absolute positions failed. **Zeroed predictions for the 13% of public sequences that were in train, to avoid rewarding memorisation.** | [E-x] (solutions mirror) | Check that inputs extrapolate (not a problem here: same caps). Do not let memorised items drive the selection metric. |
| 12 | **CMI – Problematic Internet Use**, 1st "How I won the lottery" (Lennart Haupt, 2024) | tiny noisy test; huge shake-up | 10-20x repeated stratified K-fold for tuning; a voted GBDT ensemble; "focus on robustness"; the winner calls it luck. | [E-x] | Repeat the evaluation to cut selection variance, and vote across models. |
| 13 | Roelofs et al., *A meta-analysis of overfitting in ML* (NeurIPS 2019; 120 Kaggle comps) | public vs private | Public − private gaps are "approximately symmetric and centered at zero"; outliers come from non-i.i.d. splits or tiny test sets. "≥ 10,000 examples is a reasonable minimum." "Distribution shift may be a more pressing problem than test-set re-use." | [E] | Our public has 8,562 feeds, below 10K, and is non-i.i.d. (weekend). Public deltas under 0.003 are not decision-grade (already a rule). |
| 14 | rsakata, "shake-upを科学する" (Kansai Kaggler meetup, Jul 2025) | taxonomy of shake-ups | The SD of a model-difference shrinks as 1/√n. The more submissions you make, the larger the upward luck on public. A "Trust CV ②" competition has a known public ≠ private shift, so make CV mimic private and do not over-trust public. Submissions on an unstable public are "bug checks". | [E] (simulation) | We are a "Trust CV ②" competition: val resembles private in weekday mix, public is weekend. |
| 15 | The Kaggle Book, validation chapter (summary) | final picks | Pick one LB-best and one CV-best. If the LB is not trusted, pick two strong-CV but **very different** models. | [O, practitioner] | Our 2-pick rule already does this. Make pick 2 maximally different. |
| 16 | Cawley & Talbot, *Over-fitting in model selection and selection bias* (JMLR 2010) | selection bias | The loss from overfitting the selection criterion "can be surprisingly large… comparable to differences between learning algorithms". **Low variance of the criterion matters as much as unbiasedness.** | [E] | The val-best of correlated pools carries an upward bias. Pick on the lower-variance holdout and val jointly (idea S1). |
| 17 | Baek et al., *Agreement-on-the-Line* (ICML 2022) | label-free OOD estimate | Pairwise ID agreement vs OOD agreement is linear, with the same slope and bias as accuracy. This predicts OOD accuracy from unlabeled data and beats prior estimators. It holds for NN classifiers when accuracy-on-the-line holds. | [E] (abstract) | A label-free check of candidate transfer to public and private (idea D1). |
| 18 | Yao et al., *Wild-Time* (NeurIPS 2022 D&B) | temporal shift, 5 datasets | **No invariant-learning, continual-learning, SSL or ensemble method is consistently more robust than ERM.** Average ID → OOD drop of 20%. | [E] | Do not invest in DRO/IRM/CORAL. Spend on data recency and validation instead. |
| 19 | Ovadia et al., *Can you trust your model's uncertainty?* (NeurIPS 2019) | dataset shift | Deep ensembles were the most robust under shift. Gains on in-distribution data often do not carry over to shifted data. 5 members are enough. | [E] | Breadth over concentration (idea S2). |
| 20 | Wortsman et al., *WiSE-FT* (CVPR 2022) | fine-tune under shift | Weight-space interpolation of the original and fine-tuned weights: +4-6 pp under shift over prior work, +1.6 pp in-distribution. Output ensembling is equivalent only for linear models. | [E] | A weight-space TO↔FT member is untested (idea T4). |
| 21 | Blakeney et al., *Does your data spark joy? Domain upsampling at the end of training* (ICML-W 2024); Llama 3 (2024) | target-domain annealing | Upsampling target-domain data in the final 10-20% of training: MMLU +6.90, GSM8K +8.26, HumanEval +6.17 (7B, 1T tokens). Llama 3 upsamples top-quality data while annealing the LR to 0. | [E] | Fresh-day cooldown (idea T1). m02 already works the same way on the traffic axis. |
| 22 | Byrd & Lipton, *Effect of importance weighting in deep learning* (ICML 2019) | IW under shift | IW affects deep nets early, but **its effect vanishes over epochs** once the net separates the data, unless strong L2 is used. | [E] | Explains the closed "coverage-weighted val / reweighting" items. Do not train with adversarial weights. Data *selection* in the cooldown (T1) is not the same thing as weighting. |
| 23 | Ji et al., *A critical study on data leakage in recommender offline evaluation* (TOIS) | timeline leakage | Without a global timeline split, "relative performance orders become unpredictable". | [E] | Selection on an interleaved in-period holdout can mis-rank recipes whose value is temporal (idea T3). |
| 24 | Schneider et al., *Covariate shift adaptation* (NeurIPS 2020) | test-time normalisation | Re-estimating BN statistics on the target data: ResNet-50 ImageNet-C mCE 76.7 → 62.2. | [E] | Test-time re-standardisation of drift-prone inputs. Low prior here (idea L1). |
| 25 | Akyürek et al., *TTT for abstract reasoning* (2024); Zhang et al., *T2ARec* (2025) | test-time training | ARC: up to 6x over the fine-tuned base, 53% with 8B. T2ARec: self-supervised TTT on test batches for sequential recommendation. | [E] | Covered in spirit by our transductive pretrained encoder (`pre_*`, small gain). Low prior (idea L2). |
| 26 | KDD Cup 2020 Debiasing, 1st (aister) | half the metric on rare items | i2i graph walks + LightGBM + post-boost of low-popularity items. | [E-x] | Our learned cold offsets gave 0. Closed; do not reopen. |

---------------------------------------------------------------------------------------------------------------------

## 3. Map to our setting (val → public → private)

| axis | val | public | private | what winners did on this axis | implication |
|---|---|---|---|---|---|
| day type | Tue + Thu (weekday) | 100% weekend (Sat + Sun morning) | 24% weekend (Sun pm + Mon + Tue) | Rohlik/Enefit-style competitions validate on matching calendar slices | [D] The m02 gain is DOW-invariant in-period. **Val resembles private more than public does, so for private decisions trust val (and the holdout) over public** (rsakata "Trust CV ②"). |
| staleness (days after train end Sep 19) | 3 / 5 | 7 / 8 | 8 / 9 / 10 | online retraining (Optiver, Enefit, Jane Street); fresh-data annealing (LLMs) | [D] No Tue → Thu gradient in pool gains. Fresh-data cooldown (T1/T2) is the untested lever. |
| item coverage (cand rows seen in train) | 65% | 56% | 53% | entity-held-out validation (MCTS, BELKA) | Keep the private-coverage emulation (stale-val replay) as a veto (D3). |
| hour mix | 25% evening | 56% morning | 37% evening | – | [D] Pool gains do not depend on hour (if anything they are larger at h<12). |
| feed size M | 96.4 | 103.6 | 100.7 | – | Already in calib.py. |
| size of set | 14.4K | 8.6K (paired SE ~0.0012) | 20.0K | Roelofs: ≥ 10K needed; rsakata: fewer submissions | Public is a bug check plus a weak tie-breaker. Never select on deltas < 0.003. |

---------------------------------------------------------------------------------------------------------------------

## 4. Ideas (ranked). Status is vs COVERAGE_MAP / EXPERIMENT_REGISTRY.

### P0: zero GPU, decision-relevant today

**S1. Pick on "val ∧ holdout", and shrink val-only advantages (winner's-curse correction).**
- **Mechanism.** The val-best candidate among K correlated pools carries an upward selection bias of about SE·E[max z]. A delta that shows on val but not on the independent, lower-variance holdout is the part most likely to be selection noise or val-specific.
- **Rule.** Rank candidates by min(z_val, z_hold) of the paired delta vs the reference, or by the inverse-variance pooled delta with a shrinkage prior. Admit a val-only advantage only if it is replicated by a split that was not used for selection.
- **Source.** Cawley & Talbot 2010 [E]; rsakata 2025 (luck grows with the number of submissions) [E]; AES 2.0 / CMI-PIU winners ("trust CV", repeated CV) [E-x].
- **Evidence [D].** The public order G > PF7x > M matches the holdout order (G +0.0002, M +0.00002 vs PF7x), not the val order (M +0.0012 > G +0.0006). M's public shortfall cannot be explained by weekend, hour or Tue → Thu staleness (§1).
- **Relevance.** Direct: it replaces decide4's "val-best contender" (disclosed as not blind) with a joint criterion.
- **Cost.** None. Status: **NEW (rule).**

**S2. Breadth over concentration under shift (equal-weight, multi-family pools).**
- **Mechanism.** Errors from different recipes decorrelate more under shift than in-distribution, so the averaging gain grows OOD. A single-family pool spends its whole edge on one recipe's in-distribution fit.
- **Source.** Ovadia et al. 2019 [E]; AES 2.0 1st (plain average of 4), CMI-PIU 1st (vote), Home Credit gold (50/50 independent pipelines) [E-x].
- **Evidence [E, ours].** Public transfer ratio vs val: PF7x (6 families) 0.9x, G (3) 0.6x, M (m02 only) 0.2x.
- **Relevance.** Pick 1 = a broad pool that includes the m02 families (G-like, or PF7x + F7x70), not M. Pick 2 = the most *different* strong pool (Kaggle Book rule), e.g. Pall (no m02).
- **Cost.** None. Status: consistent with the member-equal policy. **NEW as a pick constraint.**

**S3. Hedge uncertain shift knobs at the midpoint, not the val optimum (Home Credit 1st).**
- **Mechanism.** When a knob's best value depends on unobserved private conditions, pick a middle value or mix members from adjacent settings. Examples: the cooldown mix (c0 0.1 / m02 0.02 / m0), and lr 1e-3 vs 2e-3.
- **Source.** yuuniee's "neutral REDUCE" choice [E-x].
- **Relevance.** Do not build m02-only pools. Keep F7 (c0) and F7x (m02) in the same pool.
- **Cost.** None. Status: PARTIAL (Pall / PF7x already mix).

### P1: new training levers (GPU, for the lead to schedule)

**T1. Fresh-day annealing: upsample the last train days in the WSD cooldown.**
- **Mechanism.** The cooldown fixes the final function (r6: the cooldown improves the mean function, +0.0031 on a 7-seed ensemble). Annealing on the most test-like data steers it toward the target, as m02 already shows on the traffic axis.
- **Why the last days are the test-like data:**
  - item coverage decays about 2 pp per day;
  - train-row `it_hist` zero rates rise from 17.5% (Sep 6) to 29.7% (Sep 19), toward val's 37.9% and private's 45.6%;
  - the non-causal popularity leak of SHIFT-1 is smallest on the last days, because they have the least within-train future.
- **Arms** (from the @77500 snapshots via cd_lane, m02 kept):
  - **fd50**: 50% of cooldown batches from Sep 17-19 (about 750K feeds);
  - **fdexp**: day weights ∝ exp(−(Sep19 − day)/3);
  - optional **fdpool**: the m02 pool restricted to Sep 17-19 random-like feeds.
- **Gates.** The existing PASS-CD rules + the pmix veto + stale-val replay + two new report columns: the "late holdout" (holdout feeds on Sep 17-19: 8,279 feeds) and the DOW slice.
- **Source.** Domain upsampling (Blakeney 2024: +6.9 MMLU, 10-20% of training is the sweet spot) [E]; Llama 3 annealing [E]; online retraining by the Optiver, Enefit and Jane Street winners [E/E-x].
- **Risk.**
  - The holdout is uniform over days, so it may show a small loss. Judge on val and the late holdout.
  - It may trade warm accuracy for cold accuracy like the closed input-dropout tricks did. Here the data is real, not corrupted, which is the difference.
- **Cost.** One wave: 7 roots × ~1 GPU-hour (the same as m02/m0). Status: **NEW.** `train.py` has no day or recency weighting, and the m02 pool is drawn from all days.

**T2. Low-LR "online" continuation on the last train days (Jane Street recipe).**
- **Mechanism.** Start from each selected F7x/F9 checkpoint with a fresh AdamW/Muon state. Use LR at about 0.1-0.2x of the cooldown start, 1-3 passes over Sep 18-19 (production + exploration), and 50% replay of older days against forgetting (the 162nd-place lesson). Then the usual FT.
- **Source.** Jane Street 2nd (online 1e-4 vs 5e-4 offline, 3 steps/day: R² 0.0143 → 0.0208) [E]; 8th (+0.008 CV) [E-x]; 162nd (rolling window + old-data mixing) [E-x].
- **Relevance.** A cheaper post-hoc version of T1 that needs no re-cool. It may be partly absorbed by the val FT, which is fresher still, so judge it on the TO half first and then the FT mix.
- **Cost.** About 10-20 GPU-min per member. Status: **NEW.**

**T3. Time-forward holdout for checkpoint selection on new launches.**
- **Mechanism.** Select checkpoints on random-like feeds from the last 2 train days only, excluded from training, instead of the 30K holdout interleaved over all 12 days [D: 1.8-2.9K per day].
- **Rationale.** Under temporal shift, selection on an in-period interleaved set can prefer more memorisation of the train-period item pool. A forward set is closer to val and test in item pool and coverage.
- **Source.** Ji et al. (timeline leakage changes model orderings) [E]; purged and forward CV in every finance winner (Jane Street repro uses date-ordered splits with warm-up gaps) [E]; Wild-Time Eval-Fix [E].
- **Cost.** A code flag. It applies only to new runs and needs no extra compute. Status: **NEW.**
- **Expected effect.** Small. Its main value is that selection agrees better with val/test.

**T4. WiSE-FT: weight-space interpolation between the train-only checkpoint and each fold's fine-tuned weights.**
- **Mechanism.** For fold k, θ_α = (1 − α)·θ_TO + α·θ_FT,k with α ∈ {0.3, 0.5, 0.7}. Predict fold k's val and use the average of the fold models for test. Use θ_0.5 either as a replacement for the 0.5/0.5 output blend or as a third component.
- **Source.** Wortsman et al. CVPR 2022: weight-space ensembling beats output ensembling under shift, +4-6 pp over prior work [E].
- **Relevance.** Our FT starts from the TO checkpoint, which is the WiSE-FT precondition. Output-space w_ft is closed (nested +0.0002), but weight space was never tried.
- **Cost.** Inference only (the CPU replay harness from r6_tta, or minutes of GPU). Status: **NEW.** Prior: low-moderate.

**T5. Re-label old data with the new-distribution model (AES 2.0 1st, +0.005).**
- **Mechanism.** In the cooldown, add a KD term toward a val-fine-tuned teacher on train exploration-like feeds, so the trunk itself learns the cold repair the FT now does. On all-cold-positive feeds that repair is worth +0.014 (review_shift).
- **Leakage control.** The teacher must not have seen the val fold being evaluated: use a 2-fold val split (teacher A → trunk → evaluate on half B, and the reverse).
- **Source.** AES 2.0 1st (pseudo-labelling Persuade with the Kaggle-only model) [E-x].
- **Relevance.** This is a different mechanism from the closed born-again KD (TO teacher) and from test pseudo-labels.
- **Cost.** High: 2 re-cools per root plus the FT. Status: **NEW, P3 (speculative).**

### P2: diagnostics (CPU)

**D1. Agreement-on-the-line check of public/private transfer, without labels.**
- **Method.**
  1. For the TO halves of about 40 members (all families), compute pairwise per-feed agreement (Spearman, or top-10 overlap) on val, public and private feeds.
  2. Fit the probit-scale line agreement_target = a·agreement_val + b over member pairs and check R².
  3. Map member val NDCG to predicted public/private NDCG with the same (a, b) (ALine-S).
  4. Validate on the 5 public-scored submissions (v10, v13/Cd, PF7x, G, M).
- **Use.** If the m02 members' agreement with non-m02 members drops more on public than the line predicts, their val edge is shift-sensitive. Otherwise it is noise or selection (S1).
- **Source.** Baek et al. 2022 [E].
- **Caveat.** The theory is for classifier accuracy, so NDCG is a heuristic extension.
- **Cost.** About 30 CPU minutes. Status: **NEW.**

**D2. Add DOW-sliced and "late-days" holdout columns to the decide scripts.**
- Done here for m02 [D]. Cheap, and it makes any future weekend claim testable in-period.
- Cost: zero. Status: **NEW (report column).**

**D3. Give the private-coverage emulation (stale-val replay) veto power in pool picks.**
- **Why.** pmix is level-miscalibrated by its own validation (+0.0040 over public). The mechanism-faithful emulation is the stale-val replay (train Sep 17-19 removed, no future snapshots).
- **Rule.** Veto a pool whose TO-half delta under replay is < −1 SE.
- **Source.** BELKA (public-favouring scaling failed on the more OOD private) [E-x]; MCTS and Eedi validation on unseen entities [E-x].
- **Cost.** CPU (the CPU2 harness exists, for TO halves). Status: PARTIAL (report-only today).

### P3: low prior (listed for completeness)

- **L1. Test-time re-standardisation of drift-prone inputs (AdaBN-style).**
  - Re-estimate the model's input-standardisation statistics for the popularity and co-vis columns on the target split.
  - Source: Schneider et al. 2020 [E].
  - Low prior: the shift is semantic (coverage), not a nuisance corruption, and most inputs are already within-feed normalised.
  - Cost: a ~30-min CPU replay test. NEW.
- **L2. Test-time training on test-period histories** (self-supervised next-event prediction).
  - Source: T2ARec 2025, ARC TTT 2024 [E].
  - Our transductive pretrained encoder already uses test snapshots and added only +0.0004-0.0009, so the prior is low. PARTIAL.
- **L3. Fold soup for FT test inference.** Average the weights of the 5 fold FT models (same init) instead of their outputs.
  - Source: LMSYS 1st [E-x]; model soups.
  - Expected gain ≈ 0 (±0.0002). Cheap. NEW.

---------------------------------------------------------------------------------------------------------------------

## 5. Do-not list confirmed by this sweep (already closed here; external support added)

| item | closed in | external evidence |
|---|---|---|
| Importance / adversarial reweighting of training or val | sol_advval (ESS 2,880), COVERAGE "coverage-weighted val" | Byrd & Lipton 2019: the IW effect vanishes in deep nets trained to separation [E] |
| DRO / IRM / CORAL / invariance penalties | not tried | Wild-Time 2022: none beats ERM under temporal shift [E] |
| Post-hoc cold-item offsets | COVERAGE "cold offsets / gates" (learned 0) | KDD Cup 2020 debiasing used post-boosts, but for a rare-item *metric*; ours is not one [E-x] |
| LB-tuned constants or thresholds | rule "never choose on < 0.003 public" | Home Credit hack (only for a time-slope metric); AES 2.0 threshold overfitting caused the >1000-rank drops [E-x] |
| Val-fitted ensemble weights (Nelder-Mead / hill climbing) | COVERAGE (closed, optimistic) | AES 2.0 1st/2nd used a plain average or a hard vote; only 3rd used Nelder-Mead [E-x] |
| Many public submissions to pick among near-ties | CURRENT_STATUS rule | rsakata 2025: upward luck grows with the number of submissions; Roelofs: public < 10K is unreliable [E] |

---------------------------------------------------------------------------------------------------------------------

## 6. Sources (URL, competition or paper, place/author, year)

1. Home Credit CRMS 1st, "My Betting Strategy" (yuuniee), 2024: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/yuuniee-1st-place-solution-my-betting-strategy (content via search extract [E-x]); competition overview with the metric: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview
2. Home Credit CRMS 9th / team gold (Dxfory), 2024: https://github.com/Dxfory/home-credit
3. Optiver Trading at the Close 1st (hyd), 2023-24: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution ; summary deck: https://speakerdeck.com/richwomanbtc/optiver-trading-at-the-close-shang-wei-jie-fa-matome
4. Enefit Prosumers 1st (hyd), 2024: https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/writeups/hyd-1st-place-solution ; 5th (loh-maa): https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/writeups/loh-maa-5th-place-solution
5. Jane Street RTMDF 2nd (Patrick Yam), reproduction by cweill, 2025: https://github.com/cweill/jane-street-causal-forecasting ; 8th (Evgeniia Grigoreva): https://www.kaggle.com/c/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution ; 162nd (max-abhi): https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/max-abhi-private-lb-162nd-solution
6. AES 2.0 1st (flg), "Trust CV (and LB a bit)", 2024: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/writeups/flg-1st-place-solution-trust-cv-and-lb-a-bit ; report: https://hippocampus-garden.com/kaggle_aes2/
7. LMSYS Chatbot Arena 1st (sayoulala), 2024: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629
8. WSDM Cup 2025 Multilingual Arena 1st (whitefebruary), 2025: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/whitefebruary-1st-place-solution
9. Leash BELKA 1st (Victor Shlepov), 2024: https://www.kaggle.com/competitions/leash-BELKA/discussion/519020 ; NeurIPS talk: https://nips.cc/virtual/2024/109287 ; generalisability study: https://pmc.ncbi.nlm.nih.gov/articles/PMC13663858
10. UM MCTS 1st (jday96314), 2024: https://github.com/jday96314/MCTS ; deck: https://speakerdeck.com/rist/solution-um-game-playing-strength-of-mcts-variants
11. Stanford Ribonanza top-3 (solutions mirror), 2023-24: https://kaggle-solutions-quartz.pages.dev/competitions/Stanford-Ribonanza-RNA-Folding
12. CMI Problematic Internet Use 1st (Lennart Haupt), 2024: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/writeups/lennart-haupts-first-place-write-up-or-how-i-won-t
13. Eedi 1st (MTH 101 / rbiswasfc), 2024: https://github.com/rbiswasfc/eedi-mining-misconceptions
14. KDD Cup 2020 Debiasing 1st (aister): https://awesome.ecosyste.ms/projects/github.com%2Faister2020%2FKDDCUP_2020_Debiasing_1st_Place
15. Roelofs et al., NeurIPS 2019: https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning
16. rsakata, "shake-upを科学する", 2025: https://speakerdeck.com/rsakata/shake-upwoke-xue-suru
17. The Kaggle Book, validation chapter (summary): https://alibakh62.github.io/pages/content/book_summary/the_kaggle_book/designing_good_validation.html
18. Cawley & Talbot, JMLR 2010: https://www.jmlr.org/papers/v11/cawley10a.html
19. Baek et al., Agreement-on-the-Line, ICML 2022: https://arxiv.org/abs/2206.13089
20. Yao et al., Wild-Time, NeurIPS 2022: https://arxiv.org/abs/2211.14238
21. Ovadia et al., NeurIPS 2019: https://research.google/blog/can-you-trust-your-models-uncertainty/
22. Wortsman et al., WiSE-FT, CVPR 2022: https://arxiv.org/abs/2109.01903
23. Blakeney et al., domain upsampling, 2024: https://arxiv.org/abs/2406.03476 ; Llama 3 Herd: https://arxiv.org/abs/2407.21783
24. Byrd & Lipton, ICML 2019: https://proceedings.mlr.press/v97/byrd19a.html
25. Ji et al., data leakage in recommender offline evaluation: https://arxiv.org/abs/2010.11060
26. Schneider et al., NeurIPS 2020: https://arxiv.org/abs/2006.16971
27. Akyürek et al., TTT for ARC, 2024: https://arxiv.org/abs/2411.07279 ; Zhang et al., T2ARec, 2025: https://arxiv.org/abs/2504.01489

Access note: Kaggle writeup and discussion pages could not be fetched directly (client-rendered, and automated access is blocked). Their numbers come from search-engine extracts or secondary reports, tagged [E-x]. Verify any number before quoting it as decisive.
