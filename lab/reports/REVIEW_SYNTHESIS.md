# Critical review: synthesis (lead)

Date: 2026-11-11, 16:10 UTC. Inputs: reports/review_{metric,leakage,overfit,nnmath,shift,integrity,ideas}.md, the adversarial verification of 24 findings, and my own CPU spot-checks. My scripts and logs are in `/tmp/scratchpad/review/synth/` (s1_v11.py ... s5_ensgain.py, *.log). Everything ran CPU-only with 12 threads, and nothing in the project was modified.

**Timing.** The auditors wrote between 13:55 and 14:25. Since then nn_final5 has finished (CURRENT_STATUS 3j, 15:45): sub_v11_nnfinal5mix is the new best, with val 0.39793. The WSD cooldown collapse was found after the audits, and the v13 hedges plus the pre2_* train assembly (`pretrain/assemble_train.py`, PID 3880241) are running now. I re-checked every recommendation against this newer state.

SE always means the SE of the paired per-feed delta, n = 14,423 val feeds. Clustering by user changes nothing (section 4).

---

## 0. Bottom line

1. **No evaluation bug and no new leak in the shipped pipeline.**
   - The metric, submission writer, z-blends, holdout mechanics, fine-tune folds and calibration math were all independently re-derived and are correct (section 3).
   - The quarantined train-row pre_* leak is the only material leak found today. Nothing in v10 or v11 uses it.
2. **sub_v11 is a genuine improvement.**
   - +0.00394 ± 0.00090 over v10 (z 4.4).
   - Positive on Tue and on Thu, on M<50 and on M>=50, on all-cold-positive and on all-warm-positive feeds, and under private M-weights.
   - The submitted file reproduces 0.5 z(nn_final5) + 0.5 z(nn_final5_trainonly): rank match 0.9999993 (2 of 2.9M rows differ, consistent with a float near-tie); tie-free; no nulls.
3. **The largest open problem is the recipe-(c) cooldown collapse** (problem P1).
   - Same-seed hedge evidence pins it on cooldown_mix 0.3, not on lr 2e-3.
   - Caution: cooled members ensemble much worse than plateau members. On an equal-size 4-member ensemble, plateau members gain +0.0038, cosine-cooled members +0.0015. So a working cooldown is not automatically a better ensemble. Judge nn_final6 at ensemble level.
4. **The remaining confirmed issues are process and guardrail issues worth at most ~0.001 each.**
   - prew was accepted below the 2-SE rule, and it still fails in the 3-vs-3 test.
   - The holdout has blind spots: M>=50 only, plus popularity inflation.
   - The acceptance rules are inconsistent with each other.
   - id_contrib is built differently on val and test. pre2 is being assembled with this bug now.
   - The quarantine hard-link gap is still open.
   - User-0 grouping.
5. **Several prominent auditor claims were refuted**: the causal it_hist rebuild, the horizon asymmetry, w_ft = 2/3, M×cold calibration, the duplicated fta launch, and "holdout SE 2.3x too small" in the nn/ regime. Section 2 lists them so nobody chases them.

---

## 1. Confirmed problems, ranked by impact on private LB / decision quality

"Live" marks issues that touch work running now: the v13 hedges, nn_final6 and the pre2 assembly. Fixes to nn/*.py are instructions for the lead, not applied.

### P1. Recipe-(c) WSD cooldown collapses; the cooldown lever is unrealised. **[live: v13 hedges / nn_final6]**
- **Evidence.** In all six v12 members, val drops from about 0.391 to 0.364 and holdout from about 0.317 to 0.289 after step about 85k (nn_report.md:447-468).
- **Same-seed paired comparison from the logs.** v13_lr2_mix1 (lr 2e-3, mix 0.1) vs v12_all_e10 (lr 2e-3, mix 0.3), both seed 23. The runs are near-identical up to the cooldown: at 77.5k val is 0.39270 vs 0.39270. After that:

| step | val v13_lr2_mix1 | val v12_all_e10 | val delta | holdout v13_lr2_mix1 | holdout v12_all_e10 | holdout delta |
|---|---|---|---|---|---|---|
| 82.5k | 0.39403 | 0.39121 | +0.0028 | 0.31899 | 0.31685 | +0.0021 |
| 85k | 0.39310 | 0.38849 | +0.0046 | 0.31821 | 0.31345 | +0.0048 |
| 87.5k | 0.39455 | 0.37882 | +0.0157 | 0.31836 | 0.30356 | +0.0148 |

  So **the mix fraction causes the collapse, at fixed lr.** The mix-0.1 run at 87.5k is already +0.0032 val / +0.0011 holdout over v12_all_e10's selected checkpoint.
- **The logged training "loss" cannot diagnose memorisation.** train.py:302-304 adds aux_w(step)·aux_CE, and aux_w anneals from 0.5 to 0.1 (train.py:247-251). aux_CE is about log M − 0.1..0.3 ≈ 4.4 nats (NNMATH-2 verifier). The anneal alone removes about 0.4 × 4.4 ≈ 1.8 nats, which matches the logged 4.2 → 2.4. The mix-0.1 and mix-0.3 runs show the same loss at 87.5k (3.22 vs 3.17).
- **Mechanism (open).** Either the main loss memorises the 134K-feed pool, seen about 9 times during the cooldown, or the K9 aux term enters pool batches at full weight (NNMATH-2: the (1−p) weight cancels in a weighted mean, train.py:304) and so fits random display orders repeatedly. v13_lr1_mix3 will show whether the collapse depends on lr.
- **Verdict.** Confirmed; the mechanism is unresolved.
- **Impact.** nn_final5 members are uncooled plateau checkpoints. A working cooldown gives about +0.003 val / +0.001 holdout per single member so far.
  - The ensemble gain is uncertain. s5_ensgain.py: plateau members gain +0.00466 val / +0.00373 holdout from 6-way ensembling (4-member subsets: +0.0038), against +0.00153 / +0.00159 for nn_final4's cosine-cooled members.
  - If cooled members lose that diversity, an ensemble of cooled members (about 0.394 + 0.0015) could fail to beat nn_final5's 0.3971 train-only.
- **Minimal fix.**
  1. Pick the cooldown arm on the holdout among non-collapsing v13 arms. Prefer lr 1e-3 unless lr 2e-3 is better by > 2 SE on train-only val (nnx_exp_optim.md:273 recommendation).
  2. Evaluate nn_final6 as an ensemble, with equal member counts and pre-declared sets: {cooled only}, {cooled + nn_final5 plateau}.
  3. Do not raise mix above 0.1 without the aux-on-pool fix (P8).

### P2. prew_* accepted below the 2-SE / 2-seed rule; still unproven 3-vs-3. (METRIC-1, LEAKAGE-1, OVERFIT-2, SHIFT-8) **[affects nn_final6 member choice]**
- **Evidence.** Single-seed acceptance: val +0.00106 ± 0.00074 (z 1.44), holdout +0.00084 ± 0.00042 (z 2.00). Labelled "PASSED" at CURRENT_STATUS:127 and nn_report.md:441; that passes only the weaker rule in 3d, not the section-4 rule.
- **My 3-vs-3 test on the final round-4 members (s1_v11.py, s3_hold.py).** Comparison: z-mean of v12_allprew_s83/84/85 minus z-mean of v12_all_e10/s81/s82.

| measure | delta | z |
|---|---|---|
| train-only val | +0.00119 ± 0.00078 | 1.52 |
| after fta fine-tune | −0.00031 ± 0.00075 | |
| holdout | +0.00052 ± 0.00046 | |
| late holdout slice (Sep 17-19, n = 8,279) | −0.00015 ± 0.00084 | |

- **prew is a real warm-prior trade** (train-only, 3v3): all-cold-positive feeds −0.0053 ± 0.0016, all-warm-positive +0.0040 ± 0.0011. After fine-tuning: −0.0044 ± 0.0016 / +0.0014 ± 0.0010. This supersedes the OVERFIT-2 verifier's single-baseline "seed artefact" reading of the cold harm. The net effect stays about 0, since label-defined reweighting cannot be extrapolated (SHIFT-4).
- **Verdict.** Confirmed as a protocol violation. Impact about 0 ± 0.001.
- **Fix.**
  1. Relabel prew as "not proven" in 3g.
  2. Keep S1 for nn_final5 (done). There the prew members act as extra seeds; S1 is also the holdout-best set.
  3. For nn_final6, prew is optional, so do not spend GPU on prew A/B runs. pre2_* must pass the full rule (P4).

### P3. Holdout yardstick blind spots: holdout-only gains can be population artefacts. (LEAKAGE-1, OVERFIT-1 corrected, INTEGRITY-1, INTEGRITY-8) **[live: v13 arm and nn_final6 set selection]**
- **Feed size.** The holdout is M>=50 only (nn/train.py:157-161). Val is 40.9% M<50 and private 39.2%.
  - cvnd stage 2 on M<50 is −0.0036 ± 0.0015, invisible to the holdout.
  - prew's whole single-seed val gain sits in M<50 (+0.0031).
  - The epochs 5 → 10 gain is +0.0039 on M<50 vs +0.0002 on M>=50.
- **Popularity inflation.** On the holdout, 55.6% of it_hist_all_luo mass comes from other users' future snapshots (val 1.9%). The LightGBM leave-group-out effect of the LUO counts is +0.0047 on the holdout vs +0.0007 on val M>=50. Almost all of it sits in Sep 6-8 holdout feeds; the late slice matches val.
- **Category-affinity inputs are under-rated** (0.0003 holdout vs 0.0020 val M>=50).
- **Deep train histories.** Val/test cap views at about 100-109; train goes to about 210. Removing them costs a member about −0.0006 on the holdout, about 0 on val (INTEGRITY-1 verifier).
- **No decision has been shown wrong so far.** The nn_final4 and nn_final5 sets win on val too; holdout checkpoint selection costs ≤ 0.0003.
- **Fix (post-hoc, no retraining).** Every holdout decision also reports:
  - the late-slice holdout (Sep 17-19 MSK, day index ≥ 11, 8,279 feeds) from preds/holdout/*.npy;
  - val deltas split M<50 / M>=50 and Tue / Thu.

  Popularity, CF or encoder inputs are judged on train-only val (with the late slice as support), never on the full holdout. Do NOT reweight the holdout to test cold share: ESS falls to 1,465 and warm-prior gains inflate (OVERFIT-1 verifier).

### P4. Two inconsistent acceptance rules, and a lenient OR-rule in nnx. (METRIC-2, METRIC-1) **[decision quality for every remaining change]**
- **Evidence.**
  - CURRENT_STATUS:113: "> 2 SE, ≥ 2 seeds".
  - 3d: "train-only val improves, holdout not worse".
  - nnx rule 1.6, max(zv, zh) ≥ 2 with both deltas > 0: null false-WIN rate 3.7% at the measured val/holdout correlation of 0.52, i.e. 1.6x nominal, not 2x.
- **The one live consequence.** lr 2e-3 under WSD traces to an OR-rule WIN: C2 had val z 1.56, holdout z 2.47, and under WSD val was −0.00097 ± 0.00058. All six v12 members use lr 2e-3, against nnx_exp_optim.md:273. The v13 hedges now test it.
- **Fix: one rule, written into CURRENT_STATUS section 4.**
  - **Primary statistic: paired train-only val delta > 2 SE, pooled over ≥ 2 seeds.**
  - Holdout delta ≥ 0, plus the late slice for popularity-type inputs.
  - Report the M<50/M>=50 and Tue/Thu splits.
  - Fine-tune recipe changes: ≥ 2 fine-tune seeds, or z ≥ 2.5, because fine-tune A/A spread is 1.4x the per-feed SE (METRIC-8).

  I prefer val-primary over the METRIC-2 verifier's Stouffer variant, (zv + zh)/√3 ≥ 2. Stouffer gives the holdout half the vote, and P3 shows the holdout over-rewards popularity inputs by 2-7x.

### P5. id_contrib built differently on val and test (separate affine maps on val). (LEAKAGE-2, SHIFT-5, INTEGRITY-3) **[live: pre2 assembly running now]**
- **Evidence.** pretrain/assemble.py:50-54 and assemble_train.py:57-61 both still compute `id_contrib = fe - noid` after separate moment matching (checked 16:05, unchanged). No-ID rows: val −0.0031 ± 0.0017, test exactly 0 (59%). PSI 3.5.
- **Impact.** About 0 today (LightGBM ±0.0002). A trap if pre2_id_contrib enters any model, especially the train stage: train and val would agree with each other but not with test, so val could not detect the problem.
- **Fix.** Compute id_contrib = a_fe·(fe_raw − fe_noid_raw), or 0 where raw fe == fe_noid. Until that is done, keep `*_id_contrib` out of every XT preset and fine-tune extra. cvnd_prew already excludes it; nn/finetune.py:148 PRE still includes pre_id_contrib, measured ±0.0002.

### P6. Quarantine does not cover the NN loader. (LEAKAGE-4) **[guardrail]**
- **Evidence.** Re-checked 16:05: nn/feat_cache/train/pre_fe.npy and data/feats_quarantine/train/pre_fe.npy share inode 2650183 (link count 2). nn/data.py:31-46 reads FEAT_CACHE first. The `cvnd_pre` / `cvnd_prez` presets are still at nn/train.py:38-46.
- **Impact.** None today. A single typo would load the section-8b leak.
- **Fix (lead, when no job imports nn/).** Delete nn/feat_cache/train/pre_*.npy, remove both presets, and make feat_path refuse names present in data/feats_quarantine/<split>/.

### P7. User-0 handling deviates from the stated rule in three places. (METRIC-6, NNMATH-6, OVERFIT-9/10/11, LEAKAGE-3, SHIFT-6) **[next fine-tune / feature rebuild]**
- All 658 user-0 val feeds fall in fine-tune fold 0 (nn/finetune.py:183). Fine-tuned minus train-only on them: +0.0028 ± 0.0033.
- LUO keeps later user-0 snapshots (src/features_luo.py:23). On the holdout, 0.59% of rows have lift 2.5-3x; val 0.036%.
- u_tr_* gives user-0 rows the pooled CTR (src/features.py:265-278).
- **Impact.** Each ≤ 0.0005, mostly in the conservative direction.
- **Fix.** Groups = np.where(u > 0, u, −arange − 1) at the next fine-tune. LUO: treat user 0 as one user. u_tr_*: NaN for user 0 at the next rebuild.

### P8. Aux-loss normalisation quirks. (NNMATH-2, NNMATH-3) **[nn_final6 only if mix > 0.1 is used]**
- **Evidence.** In pure exploration batches the (1−p) aux down-weighting cancels: applied factor 1.0 vs intended 0.159. The aux:main ratio scales with 512/B, i.e. 1.6x on average and up to 4.5x on long feeds.
- **Impact.** Previously judged ≤ 0.0005. It is now a candidate contributor to P1.
- **Fix.** In an nn_x copy, `aux = aux_w * (ce * fw_aux).mean()` (denominator numel, not sum(w), and not max_feeds), or aux_w = 0 on pool batches. Run one paired arm only if v13 shows mix 0.3 also collapses at lr 1e-3 and a higher mix is still wanted.

### P9. Documentation overstates evidence. (METRIC-3, METRIC-9, OVERFIT-3, METRIC-2, INTEGRITY-2) **[decision hygiene]**
Correct these in CURRENT_STATUS / nn_report:

| where | current text | correction |
|---|---|---|
| 3g / nn_report:441 | prew "PASSED" | not proven (val z 1.44; 3v3 z 1.52; −0.0003 after fine-tune) |
| 3h | "gap is SHRINKING" | the residual change v5 → v10 is +0.0007 ± 0.0018, i.e. noise |
| 3h | "treat predictions as lower bounds within ~0.002" | ± 0.003 bands (section 4) |
| 3i | "holdout per-feed SE 2.3x too small" | not reproduced in nn/ (section 4) |
| v5 | predicted 0.3788 | 0.37896 |
| v5_lh | 57 public feeds changed, expected +0.004 | 54 changed, +0.0037 |
| nn_report:430 | gate-only +0.0015 / +0.0013 | matched-step +0.0010 / +0.0009 |
| 3d | nn_final3 Tue 0.3931 / Thu 0.3868 | not reproducible (0.39047 / 0.38860) |

Use an explicit day definition: MSK date, (ts + 10800) // 86400.

### P10. Minor code hygiene (no measurable effect; fix when convenient)
- calib.py public boundary should be `<=` BOUND: 8,580 public feeds, not 8,562; changes predictions by 0.00003 (METRIC-4).
- src/stack_val.py:92 and src/cv_val.py:26-29 early-stop on the evaluated fold. Only the historical stk_* line is affected (METRIC-7).
- src/eval_family.py's frozen base contains non-LUO it_hist_* and it_cand_cnt_split*: ≈ +0.0002 leak in harness base numbers. Family verdicts are unaffected (LEAKAGE-5).
- metrics.py lacks finiteness and offsets == cumsum(M) asserts; NaN ranks last silently. make_sub asserts this already.
- Dense features run in bf16 at inference and fp32 in training: ±0.0002 per model, about 0 for the ensemble (NNMATH-4).
- Retrieval top-16 ties on bf16 cosines (NNMATH-5). LUO uses strict th > tc within one 128-s tick (INTEGRITY-5).
- err_SYNTHESIS / err_segments "test-mix" (M×cold) levels are inflated by +0.007 (SHIFT-4). Retire them; nothing live depends on them.

### Known properties, confirmed, keep as is (not bugs)
- **it_hist_*_luo is non-causal on train rows.** 56-58% of its mass is other users' future snapshots. It is popularity, not labels, and the causal version loses −0.0018 to −0.0023 val in two independent LightGBM proxies (SHIFT-1, LEAKAGE-7, INTEGRITY-4).
- **Staleness.** Private coverage costs the train-only half −0.0011 ± 0.0006 (coverage-matched emulation, SHIFT-3 verifier). This is a level note, not a model-ranking issue.
- **Train histories are deeper than val/test** (organiser data property, INTEGRITY-1).

---

## 2. Refuted or unverified claims (do not chase)

| claim | source | verdict | why |
|---|---|---|---|
| v12 fine-tunes run without fta | NNMATH-1 | **refuted** | Separate fta chains (SUF=n2cpf) existed. All 6 n2cpf fine-tunes exist (15:04-15:43), and fta beats the snapshot fine-tune on every member (ensemble +0.00151 ± 0.00039). Do not relaunch. |
| Causal it_hist rebuild worth +0.001-0.003 private; cold items mis-calibrated | SHIFT-1, INTEGRITY-4 idea | **refuted** | Nested cold-bonus test −0.0001 ± 0.0003; causal counts −0.0023 ± 0.0008 in LightGBM. Counterfactual "−0.0017" measures lost information, not a fixable bias. |
| Val it_hist should drop test snapshots (val overstates private by 0.0007) | SHIFT-2 | **refuted** | Private gets equal sparse-corpus information from earlier snapshots (log1p lift 0.0482 vs 0.0487). The fix would create the opposite asymmetry. |
| Fine-tune value grows with staleness; use w_ft = 2/3 | SHIFT-3 | **refuted** | s16 reweights on label-defined coldness. A constant cold-boost placebo scores +0.009 on it. w_ft 2/3 is −0.0004 on val and unchanged under emulation. |
| calib.py biased low; M×cold removes the residual; or private = M-only − 0.002 | OVERFIT-3, SHIFT-4 | **not supported** | 4 LB residuals ≈ 1 effective draw (SD 0.0033). Covariate picked post hoc (never-train overshoots, covis0 undershoots). Cold share is confounded with feed type in val and drifts with time on test. |
| Holdout/val acceptance systematically favours warm priors; reweight holdout to test cold share | OVERFIT-1 | **refuted (mechanism)** | cvnd is not a warm prior (cold-positive +0.0029 on val). The holdout under-rates prew. The real blind spot is M<50 (P3). |
| prew gain ≈ 0 under private mix; hurts private | OVERFIT-2 | **overstated** | Single-baseline-seed artefact for the private-mix part. The cold/warm trade itself is real (3v3, P2), but the net is about 0. |
| Fine-tune over-training leaves +0.0005-0.001 | OVERFIT-4 | **refuted for shipped blends** | Holds for the fine-tuned component alone. In the 0.5/0.5 mix all early-stop variants are within ±0.0004 (≤ 0.6 SE). nn_final5's fta-vs-snapshot gives mix +0.00025 ± 0.00024. Do not add `--epochs 2`. |
| OR-rule doubles the false-WIN rate; gate is a no-op | METRIC-2 | **overstated** | Rate is 1.6x nominal. Gate evidence is consistently small-positive (stage-2 z 2.64); keep it. |
| Holdout over-rewards CF/encoder inputs generally | LEAKAGE-1 | **narrowed** | Inflation is specific to transductive popularity counts. Co-vis is 1.6x, within noise. |
| Holdout per-feed SE is 2.3x too small | nnx_VERDICT 2.2 (3i) | **not reproduced in nn/** | 15 same-config seed pairs (v9_e10 ×4, v12_all ×3, v12_allprew ×3, v8_nd_e3 ×3): rms z 0.83 val, 0.89 holdout (s4_aa.py). The nnx estimate rests on 3 pairs (2 df). |
| v5_lh: exact 0.0000 public change implies broken boost | INTEGRITY-2 framing | **no bug** | Boost applied correctly (97 feeds, rank 30.0 → 0.79). Scores are recorded to 4 decimals. The later-snapshot rule stays. |
| Domain token for short feeds; it_hist level drift | NNMATH-8, NNMATH-9 | **unverified, < 2 SE** | +0.00027 ± 0.00027 overall; do not select a token on val. |
| Per-feed cold-share splits as a staleness proxy (prew3 −0.0040 on cold ≥ 0.45) | ideas §0 | **confounded** | The ideas report itself shows the day split reverses the sign. Do not decide on cold-share arguments. |

---

## 3. Verified correct (safe foundations)

- **Metric.**
  - src/metrics.py matches two brute-force references on 9,000 synthetic feeds with heavy ties and on real preds (max diff 3e-16).
  - Full-list IDCG is consistent with public LB levels (NDCG@30 would be off by 0.057).
  - Pessimistic vs random tie-break difference = 0.00000 for all main preds.
  - No val/test/holdout feed has zero positives; M ≥ 5 everywhere.
- **Submissions.**
  - make_sub.py: rank M−1 = best, tie-free, key set equals sample_submission, Int32/Int32/Float32.
  - v5, v10 and v11 reproduce from their formulas (v11 checked here: s2_sub.py, rank match 0.9999993, 0 nulls, 100% feeds tie-free).
  - No NaN/inf in any of 639 pred files.
- **Alignment.** All 741 feature files and 637 preds are length- and order-consistent. The flat test arrays equal the raw parquet. No history contains post-request events. Candidates never appear in their own history.
- **Fine-tune.**
  - GroupKFold by user; OOF only from held-out folds; test = mean of 5 fold models.
  - Label-free standardisation.
  - Saved preds come from the snapshot mean or the final EMA weights, never from an OOF epoch choice.
  - The 51 loaded extras contain no val labels, ALS, x_lh_* or non-LUO it_hist_*.
- **Holdout.**
  - Deterministic (rng 12345); identical in train.py and holdout_eval.py.
  - Excluded from training, stage 2, feature_stats and the cooldown pool.
  - Holdout-selected 10-epoch checkpoints sit at or near the end of training (nn_final4) or on the plateau (nn_final5). In both cases val is not used for selection.
- **Train-stage inputs.**
  - No labels. Item-ID path off (id_buckets 0) in all nn_final4/5 members.
  - seq_cv_* is causal. ctx_* are feed-internal. LUO arithmetic matches a raw recomputation exactly.
  - pre2_* availability gate passes (train −0.153 vs val −0.146 mean z on LUO-cold; the leaky pre_* had −1.46).
- **NN math.**
  - No cross-feed attention. Padding and permutation invariance hold (≤ 2e-6).
  - Masks, listwise loss, the aux target direction, the WSD schedule, the Muon setup, the domain flag and the normalisation buffers all check out.
  - A CPU re-implementation reproduces the GPU val within 0.0003.
- **Statistics.**
  - User-clustered SE = iid SE (ratio 1.00; 25 users have 2 val feeds; 0.98-1.12 with user 0 as one cluster).
  - Per-feed SE captures seed noise for train-only models (rms z 0.83-1.04 across A/A sets).
  - calib.py post-stratification math is correct (all buckets populated, ESS 14.3K).
- **Selection optimism.**
  - v10's val is optimistic by ≤ 0.001 (blend-choice bootstrap +0.00012).
  - v11 adds: the S1 set chosen on the holdout; fta pre-declared (nn_report:444); one val-driven choice (dropping +0.1 pre, z −1.87). Expected optimism ≤ 0.0005.

---

## 4. Corrected statistics

**Paired val SEs (iid = user-clustered).**

| pair type | sd(d) | SE val | MDE (80% power, z > 2) | SE private (paired) | P(order holds on private \| val Δ = +0.0005) |
|---|---|---|---|---|---|
| blend tweak | 0.050 | 0.00042 | 0.0012 | 0.00036 | 0.82 |
| seed / same-arch variant | 0.089 | 0.00074 | 0.0021 | 0.00063 | 0.70 |
| model generation | 0.096-0.108 | 0.00080-0.00090 | 0.0023-0.0026 | 0.00068-0.00076 | 0.68 |

- Single-seed tests detect only ≥ 0.002 effects reliably. Power for a true +0.001 is 0.23-0.37.
- Below a ~0.0002 true private difference, the private order is a coin flip.

**Seed / fine-tune noise.**
- Train-only per-feed SE is calibrated: 15 nn/ seed pairs give rms z 0.83 val, 0.89 holdout.
- Single-member val seed SD: 0.0003-0.0004 (cosine, nn_final4); 0.0003-0.0011 (plateau-selected v12 sets).
- Fine-tune A/A spread is 1.4x the per-feed SE: use ≥ 2 fine-tune seeds or z ≥ 2.5.

**Key deltas, recomputed (s1_v11.py).**

| comparison | delta | z |
|---|---|---|
| v11 − v10 | +0.00394 ± 0.00090 | 4.38 |
| nn_final5_to − nn_final4_to | +0.00431 ± 0.00091 | |
| v11 mix − nn_final5 train-only | +0.00081 ± 0.00044 | 1.85 |
| v11 mix − nn_final5 fine-tuned | +0.00009 ± 0.00043 | |
| + 0.1 pre_final on v11 | −0.00078 ± 0.00042 | |
| eq(v10, v11) − v11 | −0.00064 ± 0.00060 | |

- v11 − v10 splits: Tue +0.0045 / Thu +0.0033; M<50 +0.0043 / M>=50 +0.0037; private M-weights +0.0039.
- The fine-tune's value has shrunk as the train stage improved. It now mostly trades cold (+0.0082) against warm (−0.0026).

**LB prediction (calib.py, M-only).**

| submission | public predicted | private predicted | public actual |
|---|---|---|---|
| v10 | 0.38592 | 0.38856 | 0.38786 |
| v11 | 0.38978 | 0.39244 | |

- **Total error SD of one prediction: public 0.0033, private 0.0027.** Components: val sampling 0.0020 (user bootstrap); LB sampling 0.0026 public / 0.0017 private.
- The 4 public residuals (+0.0003 / +0.0013 / +0.0014 / +0.0019) share both error sources, so they are about one draw. "Pessimistic by 0.0012" is not established.
- Unconfirmed directional terms that roughly cancel: staleness about −0.001 private (SHIFT-3 verifier), evening-heavy hour mix about −0.001 private, and the positive public residual +0.0012.
- **Best estimate: v11 private 0.392 ± 0.003 (1 SD), public 0.390 ± 0.0033.**
- Expected public gain of v11 over v10: +0.0039 ± 0.0015 (paired public SE 0.0012). P(v11 public < v10 public) ≈ 0.5% under this model. P(v11 private > v10 private) > 0.99.
- Steer by paired val deltas, never by levels. Never use M×cold or logistic cold reweighting: it is biased +0.003 to +0.035 out of sample and drops ESS to about 3-6K.

---

## 5. Ranked improvement plan

Merged from the auditors and the ideas report. Done, rejected and refuted items are dropped (list at the end). Gains are expected private NDCG, and all are estimates.

### Now (next 24 h)
| # | action | exp. gain | cost | overfit risk | validation |
|---|---|---|---|---|---|
| N1 | **Fix the cooldown (P1).** Choose the arm from the 4 v13 hedges on the holdout among non-collapsing arms; prefer lr 1e-3 unless 2e-3 wins by > 2 SE on train-only val. Train nn_final6 members with it and fta fine-tune. | 0 to +0.002 (ensemble; single members +0.003 val so far, but diversity loss, see P1) | running + 2-3 h/member | low | Pre-declared holdout ensemble sets at **equal member count**: {cooled}, {cooled + nn_final5}. Confirm on train-only val > 2 SE vs nn_final5_trainonly; report M and day splits. |
| N2 | **Cooldown-only continuation from the saved nn_final5 plateau checkpoints.** Likely feasible with existing flags: `--init v12_* --epochs 2 --sched wsd --cooldown_frac 1.0 --warmup 0 --cooldown_mix 0.1 --aux_anneal 0.1`, same Muon/retr/gate/xt flags. The init path is train.py:168-196, the schedule lr_at:238-245. The NN agent must confirm. Muon/AdamW state resets. Gives 6 cooled twins at about 0.2 run each. | +0.001 to +0.002 (plateau + cooled ensemble) | about 30 GPU-min per member | low | Holdout-selected inside each run. Pre-declared set {6 plateau + 6 cooled}, equal weights; val > 2 SE vs nn_final5. |
| N3 | **Guardrails before pre2 is used anywhere:** keep `*_id_contrib` out of presets/extras, or fix assemble*.py (P5). Delete the feat_cache pre_* links and the cvnd_pre/cvnd_prez presets (P6). | protects | minutes | none | grep the presets; stat the inodes |
| N4 | **Adopt the single acceptance rule (P4)** and post-hoc reporting (late holdout slice; M<50/M>=50; Tue/Thu) for every remaining decision. | decision quality | CPU minutes | none | — |
| N5 | **Freeze the final-pick protocol now (section 6)** and correct the docs (P9). | protects ±0.001 | minutes | none | — |
| N6 | **One public sanity submission of v11** (if submissions are available). Expected 0.390 ± 0.0015 relative to v10. Alarm if below 0.3879. | gross-error check | 1 submission | none | not for choosing between candidates within 0.003 |

### This week
| # | action | exp. gain | cost | overfit risk | validation |
|---|---|---|---|---|---|
| W1 | **Plateau snapshot ensembles.** Save several holdout-evaluated plateau checkpoints per run (e.g. every 5k steps over the last 30% of the plateau) plus the cooled end, and predict val/test/holdout for each. Plateau members gain 2.5x more from ensembling than cooled ones (s5_ensgain.py). Code change goes in a copy, or in nn/train.py once no live job imports it. | +0.0005 to +0.002 | inference only | low | Pre-declared "all saved snapshots, equal weight" vs best single, on the holdout; then val > 2 SE |
| W2 | **pre2_* train-stage test** (2 seeds, paired, `cvnd_pre2` built like cvnd_prew without id_contrib). Ideas #4. | 0 to +0.001 vs prew | 2 runs | low; re-check the 8b availability gate | P4 rule plus cold/warm and M splits; never decide on the full holdout alone |
| W3 | **Fine-tune-branch hygiene re-FT.** Drop hour, it_tr_*, u_tr_*, it_cand_cnt_train (ideas #2; SHIFT-7, INTEGRITY-7); user-0 singleton groups (P7). Pre-declared removal. | 0 on val, 0 to +0.0005 private | < 1 GPU-h per member set | none | non-inferiority on OOF (Δ > −2 SE, 2 fine-tune seeds) |
| W4 | **Aux normalisation fix (P8)** as one paired arm, only if v13 implicates the pool at mix 0.3 and a larger mix is wanted. | 0 to +0.0005 | 1 run | low | P4 rule |
| W5 | **Stale-val diagnostic** (ideas #1). Use the coverage-matched, internally consistent emulation (SHIFT-3 verifier: it_hist minus train snapshots of days ≥ Sep 17, seq_cv rebuilt with source < Sep 18; scripts v03/v04 in verify_SHIFT-3). | indirect | CPU 1-2 h + inference | none if diagnostic only | report next to val; **never** an acceptance or tuning target |
| W6 | **Export the aux (production-order) head as a signal** (ideas #7). | 0 to +0.0005 | GPU minutes | low with a pre-declared weight | fixed +0.1 z weight, val > 2 SE and holdout ≥ 0 |
| W7 | **Density-matched popularity counts** (val/test snapshots × ρ ≈ 37; ideas #5). | +0.0003 to +0.001 | CPU 1 h + 2 runs | low | P4 rule; young-item calibration ratio should move toward 1 |

### Later / low prior
- Full-val fine-tune refit for test (ideas #8): +0.0002 to +0.0005, not measurable on val; ship only with the nested simulation.
- Age-conditioned ID dropout (ideas #6): medium risk.
- Recency-weighted cooldown (#10).
- Capacity re-probe under Muon (#12).
- Optimizer grouping ablations (NNMATH-7).
- PAL position tower.
- Retrieval dedupe / fp32 S (NNMATH-5).
- Short-feed domain token (NNMATH-8).
- Val-style history caps for train (INTEGRITY-1; at most 0.0005).
- Per-positive softmax (#11): proxy null.
- Exploration-propensity features (SHIFT-9).

### Dropped (refuted, done, or rejected; do not re-propose)
- Causal or "no-test-snapshot" it_hist rebuild (SHIFT-1/2).
- w_ft = 2/3 (SHIFT-3).
- M×cold calibration or holdout cold-reweighting (OVERFIT-1/3, SHIFT-4).
- fta `--epochs 2` / nested fine-tune epoch selection for the shipped mix (OVERFIT-4).
- Relaunching fta (NNMATH-1).
- More prew A/B runs.
- 0.1 pre_final term (now −0.0008).
- Plus everything rejected in nnx_VERDICT and nn_report: KD, ID residual, uniform ID dropout, stage 2 after WSD, FoX, QK-norm, MoE, PLE, TabM, LP-FT, stage-1 EMA, train-row pre_*, later-history boost, stacker line.

---

## 6. Protocol for the 2 final private submissions

This assumes the best of the two selected submissions counts. If only one counts, use pick 1 alone.

1. **Freeze the candidate list now, before seeing nn_final6 numbers.**
   - C1 = v11 = 0.5 z(nn_final5) + 0.5 z(nn_final5_trainonly).
   - C2 = nn_final6 mix: same 0.5/0.5 recipe, fta fine-tune, member set chosen on the holdout among the pre-declared sets in N1/N2.
   - C3 = equal z-mean of C1 and C2.
   - Reference only: v10, and C4 = eq(v10, v11).
   - No pre_final term. No other weights. At most these 3 coarse choices.
2. **Gate for replacing v11 as pick 1.** A candidate must pass all of:
   - Paired val Δ > 2 SE over v11 (MDE ≈ 0.0012-0.0021).
   - No split (Tue, Thu, M<50, M>=50) below −2 SE.
   - Holdout ensemble ≥ nn_final5's 0.32162.
   - The file passes the s2_sub.py-style checks: formula reproduction, key set, tie-free, finite.

   Otherwise v11 stays pick 1.
3. **Pick 2** = the frozen candidate with the lowest test rank-correlation to pick 1 among those within 1 SE of pick 1 on val. Preference order C3, then C2 or C1 (whichever is not pick 1).
   - If nn_final6 does not exist, pick 2 = C4: −0.00064 ± 0.00060 vs v11, roughly a 20% chance of beating v11 on private.
   - Never use v10 alone: it is 4.4 SE worse, P(beats v11) < 1%.
4. **Public LB.**
   - At most one sanity submission per candidate family, with alarms set from section 4 (± 0.0015 paired).
   - Never choose between candidates whose val difference is < 0.003 by public score (public paired SE 0.0010-0.0014).
5. **Report the final picks with honest numbers**: val, paired Δ ± SE, and predicted private ± 0.003. Record every val-driven choice made after this freeze.
