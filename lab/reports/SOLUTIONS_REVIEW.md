# SOLUTIONS_REVIEW: solvers, adversarial validation, error analysis, experiments -> final design and roadmap

Synthesis lead, 2026-11-11 ~22:45 UTC. This document merges six inputs:
- `sol_solvers.md`, `sol_advval.md`, `sol_errors.md` and `sol_ensemble.md` (research);
- `sol_gpu.md` (3 K9 arms, 8 runs on GPU N);
- `sol_cpu_CPU{1,2,3}_*.md` (CPU experiments), with independent verifier re-computations for CPU1 and CPU2 A/B.

It also uses CURRENT_STATUS 3k-3o and three new information-only checks of mine. The checks are CPU-only, read-only on preds/ and need no new training:
- `ce_check`: the pre-declared Ce = Cd + F8 decision.
- `corr_check`: within-family seed correlation and ensemble curves.
- `cd_stale` / `cd_weighted`: Cd on stale-val and on weighted val.

Scripts and logs are in `/tmp/scratchpad/solvers/synth/`. Nothing was written under nn/, src/, preds/ or subs/.

Notation:
- v12 = R = 0.5 z(nn_final5v3) + 0.5 z(nn_final5_trainonly), val 0.39818.
- Cd = member-equal F5cd+F6+F7, all n2cpv3: subs/cand_Cd_memeq_F5cdF6F7.parquet.
- TO = train-only, FT = fine-tuned.
- Deltas are paired per feed, mean ± SE.

## 0. Bottom line

1. **Picks:** v12 stays **pick 1** under the frozen protocol, and **Cd is pick 2**.
   - Cd vs v12 on val is +0.00083 ± 0.00054 (z 1.55). The frozen rule needs > 2 SE.
   - Cd is the holdout-selected candidate (holdout 0.32319, +0.00157 ± 0.00034).
   - New check: Cd holds up on every valid private-like yardstick:
     - its TO half on stale-val is **+0.00171 ± 0.00059** vs v12's TO half;
     - M-private-weighted val +0.00087, intrinsic-adversarial +0.00075.
2. **New, verified: Ce (= Cd + F8) is rejected by the pre-declared holdout selector.**
   - Ce holdout is 0.32208, vs Cd 0.32319 (−0.00112 ± 0.00024). Val is +0.00020 vs v12.
   - F8 passes the single-member strength gate, but **2-epoch seeds are near-duplicates**:
     - seed-to-seed within-feed correlation is 0.978, vs 0.930-0.941 for every 10-epoch family;
     - so 6 members add only +0.0010 val / +0.0014 holdout, vs +0.0040-0.0047 / +0.0035-0.0039 for 10-epoch families.
   - "2 epochs ≈ 10 epochs" (CURRENT_STATUS 21:20) holds for single-member strength only, **not for ensemble value**.
3. **Adversarial validation:** val, public and private are one population on intrinsic covariates. The whole val→test shift is item staleness (coverage).
   - Public ranks models like private does. The public LB cannot resolve differences below ~0.0025.
   - Expected val→private gap is −0.006 (range −0.003 to −0.009).
   - No new feature drops: the fine-tune drifters were already removed by W3b.
4. **Error analysis:** v11's loss sits in **mid-list ordering of familiar candidates** (ranks 3-99 hold 83% of the gap).
   - No candidate signal has residual information beyond the model.
   - Every alternative solver only trades **cold vs warm** feeds.
   - The one measured lever is **variance reduction across same-strength, different-recipe families**.
5. **Experiments:** none of the 8 GPU runs or 3 CPU experiments yields a new recipe change or a new blend member.
   - K9 is locally optimal: tau 30 is worse, PL-top-20 and weight 1.0 are null.
   - The aux-head export (W6) fails its gate by 2x.
   - Cross-recipe pooling (Q3) is "likely" but below 2 SE on val, and is superseded by the pre-declared Cd.
   - pre2s (F7's input family) is **not** staleness-fragile.

---------------------------------------------------------------------------------------------------------------------

## 1. Adversarial validation: what it says and what it changes

### 1.1 Findings

All verified in `sol_advval.md`. LightGBM classifiers, GroupKFold by user, OOF AUC.

| comparison | intrinsic AUC (no time, no coverage) | coverage-inclusive AUC | main drivers |
|---|---|---|---|
| val vs private | **0.541** | 0.950 | cold_item_share PSI 1.07 (0.290→0.406); days_since_prior 0.75 (9.8→15.2 d); it_hist_all_luo zero share 0.46 (0.381→0.459) |
| val vs public | 0.546 | 0.897 | same, about half the size (cold share 0.371) |
| public vs private | 0.535 | 0.779 | ~1 day more staleness, plus the hour mix |
| val Tue vs Thu (reference) | 0.526 | 0.855 | the same staleness gradient inside val |
| train random-like vs val (M≥50) | 0.756 | 0.9987 | coverage (LUO zero share 0.22 vs 0.38); intrinsic part from H and hist et1 share (0.49-0.50 vs 0.46): random-like production feeds are a different population from exploration feeds |

**Val, public and private: same population.**
- The intrinsic val→private shift is no larger than Tue→Thu, so val's two days already span the composition variation that matters.
- Model agreement on test is identical on public and private (within-feed Spearman FT~TO 0.9847 on both; v11~v10 0.9551 vs 0.9549).
- Deltas agree under public and private weights (v11−v10 +0.00386 vs +0.00388). **Public is a valid proxy for private for ranking.**
- Its paired SE is ~0.0012, so it resolves only differences ≥ ~0.0025.

**Candidate-level drift of live inputs.**
- With the `pre_id_contrib` parity defect fixed, the NN train-stage inputs separate val from private at AUC 0.595 (coverage-driven).
- The fine-tune extras separate them at 0.737. The worst is `it_tr_coec_rk`: PSI 0.26 on cold rows, because ranks of tied defaults move with the feed's cold share.
- `pre_id_contrib` itself was the #1 separator: AUC 0.956 → 0.793 when replaced by `_v2`.
- All of these are removed or replaced by W3b (n2c_pre_v3), which is adopted for every family (CURRENT_STATUS 18:55).

**Train → val/private, by NN input family.**
- Popularity / co-vis: 0.80.
- prew: 0.62.
- **pre2s: 0.61**.
- f1-5 / category, M / H, ctx: 0.56-0.57.
- Embedding proxies: 0.53.

**Weighting.**
- Intrinsic adversarial weights reproduce the M-only calibration within noise: v11 private 0.3931 vs 0.3924; null sd of the level 0.0022, of a delta 0.0002.
- Coverage-inclusive weights are invalid: ESS 2,880 and a v11 "private" level of 0.442. Small feeds look "private-like".
- Reweighting the holdout lowers its agreement with val across 59 models (Spearman 0.975 → 0.961).

**Trap (ADV-8).** Per-feed means of within-feed z columns fingerprint the split through float rounding: AUC 0.997 from 1e-7 differences. Any future adversarial validation must drop aggregates that are constant by construction.

### 1.2 What it changes

| item | decision |
|---|---|
| **Primary yardstick** | Val (paired per feed, standard slices) stays primary; levels come from M-only calib.py plus an explicit hour term (M × hour-4: v11 private 0.39186 vs M-only 0.39244). |
| **Robustness columns** (report, never select on) | (a) Intrinsic-adversarial weighted delta. (b) **Stale-val replay** (CPU2 harness `solvers/CPU2_stale_replay_pre2s_and_arms.py`, review_shift `both` scenario, TO halves): the only yardstick that sees private-like coverage. Mandatory for any change that trades cold vs warm or adds an input family. |
| **Holdout** | A selector, not independent evidence. It is M≥50 only, its LUO coverage is train-like, and every member's checkpoint was selected on it (~40 evaluations per run; twice for F5cd; verifier CPU1 caveat 1). Count val TO-half z, not holdout z, as evidence. |
| **Never use** | Coverage-weighted val; reweighted holdout; public LB to choose between candidates < 0.003 apart. |
| **Features to drop** | Nothing new. The fine-tune drifters (pre_id_contrib, it_tr_*, u_tr_*, it_cand_cnt_train, hour) are already out via W3b. Nothing to drop in the train stage: the drift is known popularity coverage (FR-3/4; 5 causal rebuilds failed). pre2s was flagged as the most train-shifted non-popularity family and was tested directly (CPU2, §3): not fragile. |
| **Expected gap** | val → private ≈ −0.006 (range −0.003..−0.009): M-mix −0.0055, hour −0.0006, extra staleness vs public −0.0002..−0.0006, mean public residual +0.0012. Predicted private: v12 0.3927, Cd 0.3935, each ± 0.003 (prediction-error SD 0.0027). |

---------------------------------------------------------------------------------------------------------------------

## 2. Where v11 fails, and which solver types address it

Source: `sol_errors.md` (verified). Gap = Σ(1 − NDCG).

| failure | evidence | solver types that could address it | measured outcome |
|---|---|---|---|
| **Mid-list order of familiar candidates** | Ranks 3-99 hold 83% of the gap (ranks 10-29: 32%). Bottom-half positives moved to the median: only +0.0069. Positives with in-history category and maxsim ≥ 0.7 are 79% of positives; v11 pair AUC 0.681 there. | Better representations / more independent models: ensembling across recipes, stronger train-stage inputs (pre2s). | **The only working lever.** Cross-recipe pools +0.0006..+0.0012 val, +0.0013..+0.0016 holdout. pre2s +0.0019 per model. |
| **No residual signal** | Symmetric local AUC of 70 attributes 0.492-0.504. Nested conditional-logit residual −0.0006 ± 0.0005. LightGBM residuals −0.0015..−0.011. Twins: v11 AUC 0.629 vs best single signal 0.564. | Stackers, GBDT residuals, feature additions. | Closed: no input carries information the NN lacks. |
| **Cold vs warm axis** | All-cold-positive feeds (17.4%, 22% of test) have lift 0.088 vs 0.116 for all-warm. Every residual or solver gains on cold and loses more on warm: FT +0.0077 cold, TO +0.0013 warm; LightGBM residual +0.011 cold, net −0.0015. Gates reach AUC 0.66-0.72 and gain 0 (oracle +0.025). Cold offset flat (+0.00013). | Routers / gates / MoE by cold or exploration, cold offsets, w_ft routing. | Closed: a missing-information frontier, not a mis-weighted blend. Only new information helps. pre2s content partly substitutes for popularity under staleness: stale cost pre2s −0.0009/−0.0014 vs cvnd −0.0021/−0.0014 (CPU2). |
| **Large feeds, long histories, returning users** | M ≥ 200 gainR 0.096 vs 0.319 at M < 10. H = 256 lift 0.097. Users with train feeds −0.0034 lift after H × M stratification. | Capacity / attention variants. | Already rejected (d=256, deep stacks, GRU/DIN: similar alone). No new mechanism found. |
| **Staleness on private** | Private cold share 0.406 vs 0.290. Stale-val cost per member −0.0004..−0.0015. | Coverage-robust inputs; staleness-robust ensembles. | Pools and pre2s keep or grow their gain under stale-val (§3). Causal popularity rebuilds failed (FR-3/4). |

**Frontier law for new solvers** (sol_ensemble, R² 0.95 val / 0.96 holdout):
- A member helps only if its gap to the ensemble is below 0.049(1 − ρ) on val.
- Every non-NN or different-paradigm solver fails this:
  - GBDT rankers: −0.03 gap at ρ 0.77-0.81;
  - sequence encoder: ρ 0.69, −0.039;
  - 14 surrogate losses: none beats softmax CE in a 3-6-seed proxy;
  - val stackers;
  - the K9 aux head: gap 0.094 at ρ 0.03 (CPU3).
- The winners' multi-family stacks (RecSys'24, Avito production) need members within ~1% of each other. Ours are 3% apart.

---------------------------------------------------------------------------------------------------------------------

## 3. Experiment results, adversarial verdicts, recommended use

"Verified" means an independent re-computation reproduced the numbers (verifier entries). For the GPU arms, the independent check is CPU2's fp32 re-inference of the same checkpoints.

| experiment | key numbers | verdict (after adversarial check) | files | recommended use |
|---|---|---|---|---|
| **GPU A**: K9 tau 30 (2-epoch cvnd_pre2s base, seeds 23/24) | TO val −0.00091 ± 0.00049 (both seeds < 0); holdout **−0.00109 ± 0.00028** (z −3.9); Thu −2.45 SE. Replay: −0.00111 base, −0.00077 stale. | **WORSE**. Direction confirmed by independent replay. | `nn_x/solvers/{ckpt,preds,preds/holdout}/…sx_tau30_s{23,24}…`; `nn_x/solvers/logs/sx_eval_r12.txt` | **Drop.** Keep tau 10: the useful production order is at the top. |
| **GPU B**: K9 as Plackett-Luce top-20 | TO val +0.00059 ± 0.00045 (z 1.31; s24 −0.00006); holdout +0.00021 ± 0.00025. M<50 +0.00235, M≥50 −0.00063. Stale DiD **−0.00064 ± 0.00039** (M<50 −0.00148). Ensemble value +0.00003. | **NULL**. The M<50 gain shrinks at private-like coverage. bf16 vs fp32 moves the per-seed delta by up to 0.0005. | `nn_x/solvers/train.py:61,103,318`; `test_pl_aux.py`; `logs/test_pl_aux.txt`; `…sx_pl20_s{23,24}…` | **Drop.** No 10-epoch pair. The post-hoc M<20 idea (full-permutation PL) is "later" only (§4.3). |
| **GPU C**: B + aux_pos 1.0 (pre-declared branch) | vs base +0.00051 ± 0.00051; vs B −0.00008 ± 0.00048; holdout +0.00026. Late20 z 2.38 is 1 of ~20 slices. | **NULL**. | `…sx_pl20_w1_s{23,24}…`; `logs/sx_eval_r3_vs_{base,pl20}.txt` | **Drop.** Keep listwise / tau 10 / aux_pos 0.5 in all families. Keep the 2-epoch harness for paired A/B: 8/8 runs selected the last step, so selection noise is 0. |
| **CPU1**: cross-recipe member-equal pools; holdout picked Q3 = F5+F6+F5cd+H (21) | Holdout Q3 0.32300 (+0.00138 ± 0.00028). Val TO half +0.00102 ± 0.00050 (z 2.04). Full mix vs v12 **+0.00061 ± 0.00046** (z 1.33); all slices > 0. Test Spearman 0.9914 vs v12. | **Confirmed, "likely"; fails the frozen rule.** Verifier caveats: holdout used in member selection; D4 val-selection optimism ~+0.0005; Q2 vs Q3 a near tie (72/28 bootstrap). F5cd n2cpv3 now exists and adds +0.00010 ± 0.00014. H still n2cpf. | `solvers/CPU1_xrecipe_member_equal_pool_v3.py`; `solvers/out/CPU1_xrecipe_member_equal_pool_v3/` | **Information only.** Superseded by the lead's pre-declared Cd: holdout 0.32319 > Q3's 0.32300, all n2cpv3, H excluded. Do not run H n2cpv3 FTs for this. The script now silently picks n2cpv3 for F5cd; keep cand_A files as n2cpf versions. |
| **CPU2 A**: stale replay of pre2s (cvnd → cvnd_pre2s, 3 epochs, 2 seeds) | Base **+0.00189 ± 0.00058**; stale **+0.00246 ± 0.00057**; DiD +0.00057 ± 0.00044 (carried by s23). Pre-checks (a)/(b)/(d) pass. Base-defined cold-positive DiD −0.00219 ± 0.00075 (non-acceptance slice). | **Confirmed. pre2s is not staleness-fragile.** The emulation is mainly popularity/co-vis staleness; the pre2-ID part is nearly inert (43K rows). | `solvers/CPU2_stale_replay_pre2s_and_arms.py`; `solvers/stale/`; `solvers/out/CPU2_stale_replay_pre2s_and_arms/phaseA*.json` | **Keep cvnd_pre2s as the input base for every new family.** No raw-channel ablation. Do not cite the DiD as a bonus. |
| **CPU2 B**: F7 vs F6 family TO, stale | 6 members: base +0.00147 ± 0.00074, stale +0.00170, DiD +0.00023. **7 members (pre-declared, verifier): base +0.00159 ± 0.00072, stale +0.00169 ± 0.00071, DiD +0.00009.** Cold-positive base −0.0026..−0.0035 (z −1.85..−2.6; depends on the member set and on replay vs stored preds). Thu stale with s107 +0.00018 ± 0.00100. | **Confirmed. No flag: F7 is pick-1 eligible.** The cold-positive loss is suspected to come from lr 2e-3 (one matched cvnd pair: lr2 − lr1 cold-positive −0.00486 ± 0.00197). | `…/phaseB.{json,log}` | **Use F7 in the final pools.** Isolate the lr cause with F9 (§4.2, item 4). |
| **CPU2 B-supp**: pools, TO halves | F5+F6 → +F7 (7 members): base +0.00074 ± 0.00034, stale +0.00094 ± 0.00034. | Confirmed; information only. | `phaseB.json` (extra.pools_supplementary) | Supporting evidence only. |
| **CPU2 C**: stale veto for GPU arms | tau30 stale −0.00077 ± 0.00049; pl20 −0.00037 ± 0.00045. | No veto (moot). | `phaseC_{tau30,pl20}.json` | None. |
| **CPU3 / W6**: K9 aux head as a member | aux_prod 12-member NDCG **0.30355** vs main 0.39770; ρ 0.034; gap 0.094 > break-even 0.047. Production order's own click NDCG on production feeds is 0.335 vs the main head's 0.38 (the main head is scored in-sample there). | **NULL; W6 closed** (stop rule; no blend computed). | `solvers/CPU3_aux_head_member_W6.py`; `solvers/out/CPU3_aux_head_member_W6/` | **Never blend the aux output.** K9's +0.0024 is trunk regularisation, so judge K9 variants on the main head only. The main_prod domain trick is a net-zero cold/warm trade (+0.00011 ± 0.00017); drop it. |
| **New (lead synthesis): Ce = Cd + F8 (12 two-epoch pre2s members), pre-declared rule** | Strength gate: F8 single TO 0.39322 / holdout 0.31775 vs best 0.39460 / 0.31940 → **passes**. Ce holdout **0.32208 vs Cd 0.32319** (−0.00112 ± 0.00024). Ce val vs v12 +0.00020 ± 0.00054; vs Cd −0.00063 ± 0.00042. F8 family mix 0.39491 (−0.00327 vs v12). | **Rejected by the pre-declared holdout selector.** The protocol worked. | `…/scratchpad/solvers/synth/ce_check.{py,log,json}` | **Drop F8 as ensemble members.** Cd stays pick 2. |
| **New: seed correlation by family** | Within-feed TO seed correlation: F5 0.938, F6 0.930, F7 0.941, **F8 0.978** (sx_base s23/s24: 0.978). k=1→6 TO gain: val +0.0047 / +0.0040 / +0.0040 / **+0.0010**; holdout +0.0037 / +0.0039 / +0.0035 / **+0.0014**. F8 FT adds nothing (FT single 0.39308 vs TO 0.39322). | **Verified** (same files, same metric). Mechanism suspected, not isolated: fewer high-LR steps → less seed divergence. The seed does change both init and data order (nn/train.py:142-150). | `…/synth/corr_check.{py,log}` | Six 2-epoch seeds are worth about 1.5-2 ten-epoch members in ensemble value (val: F8 k6 0.39418 vs F5 k1 0.39247 / k2 0.39495). **No more 2-epoch same-recipe seeds.** The 2-epoch harness stays useful for paired A/B (high seed correlation = low paired noise). |
| **New: Cd on private-like yardsticks** | TO half vs v12 TO (replay vs replay, 19 vs 6 members): base +0.00154 ± 0.00059 (stored +0.00165), **stale +0.00171 ± 0.00059**, DiD +0.00018 ± 0.00052; Thu stale +0.00185, Tue +0.00159. Full mix vs v12: M-private +0.00087 ± 0.00053; adv-intrinsic private +0.00075 ± 0.00054; adv-all +0.00101; **Thu-only +0.00011 ± 0.00078**. | **Verified.** Replay fidelity ≥ 0.9999987 for the 7 replays not covered by CPU2. | `…/synth/cd_stale.{py,log}`, `cd_weighted.log` | Cd is robust to staleness and to the weightings. The weak spot is the full mix on Thu, carried by the FT branch: the TO half on Thu is +0.0008 base / +0.0019 stale. |

**Observation (information only): the FT branch adds less as members improve.**
- Ensemble FT minus TO (first 6 members): F5 +0.0005, F6 +0.0010, F7 −0.0004, F8 +0.0001.
- Mix minus TO half: v12 +0.00106, Cd +0.00024.
- w_ft = 0.5 stays (pre-declared; nested 5/5 folds on F5). See roadmap item 9.

---------------------------------------------------------------------------------------------------------------------

## 4. Recommended final solution design and roadmap

### 4.1 Final design (pre-declarable; the current candidates already satisfy it)

1. **Members.**
   - Holdout-selected TO checkpoints of 10-epoch NN families on the live leak-free inputs: cvnd / cvnd_prew / cvnd_pre2s, K9 listwise tau 10 / aux_pos 0.5, Muon + WSD.
   - Each member has its **n2cpv3** fta fine-tune (GroupKFold(5) by user, u0 singleton).
2. **Family admission (amended).** A family must pass both gates:
   - (a) the existing single-member gate: mean TO within 0.002 of the best family on val **and** holdout;
   - (b) **new: a family-ensemble gate.** The z-mean of the family's first 6 seeds (by seed order) must have TO within 0.002 of the best family's 6-seed TO ensemble, on val **and** holdout.
   - Reason: (a) is blind to within-family correlation (F8).
   - Checked: (b) leaves Cd unchanged. 6-seed TO holdout: F5cd 0.32279, F7 0.32263, F6 0.32183, F5 0.32162, F8 0.31903 (fails). Val: F7 0.39871, F5 0.39712, F5cd 0.39702, F6 0.39697, F8 0.39418 (fails).
   - Declare (b) now, before any further family lands. It is not applied retroactively; the holdout selector already rejected Ce.
3. **Combiner.**
   - Per-feed z of each member; member-equal mean per branch; final = 0.5 z(FT) + 0.5 z(TO).
   - No fitted weights, segment weights, rank fusion, stackers, GBDT, encoder or aux terms (all measured negative or null).
4. **Selection.**
   - Pre-declared candidate sets; choose on the holdout.
   - Replace v12 only if val > 2 SE, no slice < −2 SE, holdout ≥ 0.32162, and presub_gate PASS.
   - Report the stale-val TO delta and the intrinsic-weighted delta as columns.
5. **Picks (best of two assumed counted).**
   - **Pick 1 = v12** (subs/sub_v12_nnfinal5v3mix.parquet; val 0.39818; predicted public 0.3900 / private 0.3927).
   - **Pick 2 = Cd** (subs/cand_Cd_memeq_F5cdF6F7.parquet; val 0.39901; predicted public 0.3909 / private 0.3935; test Spearman vs v12 0.986; presub_gate PASS with 7 WARN, all FR-2 or the justified leak-probe WARN).
   - If a new family (F9) produces a holdout-selected candidate that beats Cd on the holdout, it replaces Cd as pick 2, or v12 as pick 1 if it clears 2 SE.
6. **If only one submission counts:**
   - The frozen rule says v12.
   - The evidence favours Cd: positive on every slice, stale TO +0.0017 (z 2.9), weighted deltas +0.0008..+0.0010. Its only soft spot is the Thu full mix (+0.0001).
   - Overriding the frozen rule is the user's call. Do not decide it from a public score: the expected public difference is ~+0.0009, below the public resolution of ~0.0025.

### 4.2 Roadmap (ranked; merges the in-progress items of CURRENT_STATUS 3m-3o)

**Next 24 h**

| # | action | owner / cost | expected | acceptance |
|---|---|---|---|---|
| 1 | Record Ce as rejected by the pre-declared holdout selector (0.32208 < 0.32319). Keep v12 / Cd. Declare gate (b) (§4.1). | lead, CPU minutes | protects against member-count dilution | n/a |
| 2 | Public sanity submission of **Cd only** (not Ce). Alarm only if public < v12 − 0.003. | lead, 1 submission | file validity; not a selector | frozen rule: never choose on < 0.003 |
| 3 | Stop farming 2-epoch same-recipe seeds. Free GPUs N. | lead | saves ~6 GPU-h per family | n/a |
| 4 | **F9 = cvnd_pre2s + recipe (iii) (lr 1e-3, mix 0.1), 10 epochs, 6 seeds**, n2cpv3 fta + holdout_eval. This is F7's input with F6's recipe. It (a) adds a same-strength, different-recipe family and (b) isolates the suspected lr 2e-3 cold-positive loss: F9 vs F7 differ only in lr. | NN agent, GPUs N, ~4-5 h wall (2 per GPU) | +0.0004..+0.0007 to the pool (sol_ensemble per-family estimate) | gates (a)+(b); then the pre-declared candidate {Cd families + F9}, chosen on the holdout vs Cd, confirmed on val, presub_gate |
| 5 | **F6cd**: N2 cooldown continuation (1 epoch, mix 0.1, holdout-selected step) **only** for F6 members whose selected checkpoint is a plateau checkpoint (selected step before the cooldown start; e.g. v14_s96 at 47.5k). Skip members that are already cooled, because they would be near-duplicates. F7 is already cooled (selected 80-87.5k), so no F7cd. | NN agent, ~30 min + fta per member | +0.0003..+0.0008 if ≥ 3 members qualify (F5cd gave +0.0008 val / +0.0011 holdout) | same as #4 |

**This week**

| # | action | cost | expected | acceptance |
|---|---|---|---|---|
| 6 | Age-conditioned ID-input dropout (review_ideas #6): blank co-vis/LUO inputs mainly for young listings, matching where test zeros occur. A/B in the 2-epoch harness (2 seeds), then a 10-epoch pair. Judged primarily on **stale-val** (P3 rule: base val ≥ −1 SE and stale > 2 SE, or the unified rule). | code + 2 + 2 runs | 0..+0.0015 private, ~0 val | pre-declared stale-val primary; never on val alone |
| 7 | Full-val fine-tune refit for the test predictions (ideas #8), with the nested simulation. | ~1 GPU-h per family | +0.0002..+0.0005 test only | nested simulation > 0; otherwise skip |
| 8 | Optional, only if it keeps rising on its own: a cvnd_pre2s + recipe (iv) (mix 0) 10-epoch family, i.e. the original nn_final8 plan before it was replaced. | ~4-5 h wall | +0.0003..+0.0005 | gates (a)+(b), holdout selector |
| 9 | CPU, information only: a pre-declared nested-CV check of w_ft ∈ {0.33, 0.5, 0.67} on the Cd pool, prompted by the shrinking FT value (§3). | CPU minutes | ±0.0003 | adopt only if 5/5 folds agree and z ≥ 2.5. Otherwise keep 0.5 |

**Later / only with spare GPU**
- **M-adaptive K9 target** (full-permutation PL for M ≤ 20, K9 elsewhere). It needs its own pre-declared pair with M<50 and stale-val as primary.
  - Evidence so far is post-hoc: M<20 +0.0046 ± 0.0015.
  - It is holdout-blind, and it shrinks under staleness.
- **SAM (ρ 0.05) in fta**: below the fine-tune MDE.
- **Cooldown pool p_random > 0.2**: only as a paired half of a future continuation (#5).
- **MC-dropout TTA, ce_rw member pair, anneal-mix**: expected ≤ +0.0005, low value per cost.

**Closed by this workflow** (add to the do-not list):
- Training and members:
  - K9 tau 30, PL-top-20 aux and aux_pos 1.0 (§3);
  - aux-head export / W6, and the main_prod domain trick;
  - 2-epoch same-recipe seed families as ensemble members (F8);
  - plateau snapshots as members (W1 / F7p: single TO 0.39218, fails the gate).
- Ensembling:
  - H-containing pools and n2cpv3 fine-tunes of H (Q3 ≈ Q3h; H is not in the pre-declared Cd);
  - GBDT ranker members and stackers of any kind;
  - surrogate-loss swaps and loss-diverse members;
  - rank / RRF fusion and segment / routing weights.
- Shift handling:
  - coverage-weighted val and reweighted holdout;
  - ULTR corrections (IPS / DLA / DR / PAL);
  - cold offsets and intent / cold gates.

### 4.3 Caveats that travel with these numbers
- Holdout z-scores overstate independent evidence, because checkpoints were selected on the same 30K feeds. Use val TO-half z and stale-val.
- GPU bf16 vs CPU fp32 inference moves per-seed deltas by up to 0.0005. Do not read sub-0.001 single-seed differences.
- Stale-val covers popularity / co-vis coverage only (not pre2_hmax32, ctx_* or history-side co-vis), and TO halves only. It overshoots private's LUO-zero share (0.485 vs 0.456), so it is a pessimistic stress test.
- The leak-probe threshold in presub_gate was changed after a trigger (CURRENT_STATUS 3o). This is logged there. The controlled pre2s pair shows no has-later gap (−0.0009 ± 0.0039).
