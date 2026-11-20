# web_shift: ranking under temporal shift, stale counters and cold items (literature scout + CPU probes)

Written 2026-11-15 21:00-21:45 UTC by the web_shift scout. CPU only; no training; nothing submitted. Work directory of the probes: `reports/TRICKS_1115/web_shift_probe/` (scripts `s1`..`s10`, logs next to them, rules in `PREDECLARED.txt`, each block written with `date -u` before its read). Markers: [M] measured for this report by a command that is on disk, [R] record (path given), [W] web source (URL in section 6), [I] judgement. `+-` = 1 SE paired per feed, 14,423 val feeds unless stated. Web searches used generic technical terms only; no project data, file name or score left the machine.

## 0. Verdict

1. **No technique in this literature clears a prior of 0.15 for +0.0005 at file level.** 34 sources were opened (section 6). Most of what they recommend is already in the pool (a second input view without the stale inputs), already closed here by measurement (training on stale-looking inputs, age-conditioned blanking, offsets, gates, test-time standardisation, pseudo-labels), or addresses a model this project does not have (item-ID embeddings).
2. **One idea was new to the project and testable on CPU; it was run today and reads INCONCLUSIVE with a positive lean.** "Lag-equalised" item-occurrence inputs fed to unchanged held-out students at inference (no retraining): 4-pool, train-only branch, val +0.00030 +- 0.00025 (z 1.2) at half dose, +0.00008 +- 0.00041 at full dose; reweighted to the private mix +0.00054 +- 0.00041 and +0.00091 +- 0.00067 [M, section 2.5]. Declared rule: SUPPORT needs +2.3 SE; not met. It is a cold / warm trade (+0.0017 on feeds without a cold positive, -0.0033 on feeds whose positives are all cold). Proposal P1 gives the two cheap reads that would decide it (lagged holdout, final-file form), 0 GPU-h.
3. **The output-offset form of the same idea is closed by its own rule.** The pools under-score warm rows whose count is deflated by the lag (conditional-logit coefficient +0.20 +- 0.03, z 6.1, train-only branch; +0.08 +- 0.03, z 2.3-2.7, fine-tuned branch), but a nested one-parameter offset on the final file reads -0.00012 +- 0.00031 (FD16P20) and -0.00008 +- 0.00030 (W8P20) [M, sections 2.3-2.4]. A 6-SE calibration error that is worth nothing in NDCG: the same pattern as the closed cold offset.
4. **New mechanism facts for the lead [M, sections 2.1-2.2]:** (a) given a listing's exposure length E before the dense cutoff, the count distribution is the same on val, public rows, private rows and the in-period holdout; the staleness shift is a shift of E at fixed listing age; (b) from val to test the share of deflated warm rows does NOT grow (24.6% of val rows, 22.6% public, 20.6% private); what grows is the share of rows with no count at all (37.9% -> 42.8% -> 45.6%) and of listings created after the cutoff (20.1% -> 27.9% -> 31.3%); (c) the pools are unbiased between cold and warm rows on val (cold dummy within 1.6 SE of zero in all 8 pool branches read). Consequence [I]: the part of the staleness cost that any normalisation, calibration or test-time adaptation can address is small (order +0.0003 on the train-only branch); the increment from val to test is lost information, which no method in this literature restores.
5. **Ranked proposals (section 4):** P1 lag-equalised inputs at inference, two CPU reads then at most one public slot (expected file gain +0.0002 public / +0.0003 private, P 0.20); P2 fine-tune with test-mix feed weights (+0.0001..+0.0002 private, P 0.15, about 1 GPU-h for the read); P3 from-scratch blanked students at the test cold share with an occlusion flag (parked arm `kwxd`, literature gives the dose; +0.0003 private, P 0.10, 12-31 GPU-h); P4 the equalised view as a trained student view (P 0.07; only if P1 passes its lagged-holdout clause).
6. **Do not try (section 5), each with a literature number and a project number:** entropy / batch-statistic test-time adaptation; label-shift correction; domain-adversarial, IRM-type or list-level invariance against time; recency weighting of train rows; popularity forecasting at inference (PDA); content-to-collaborative generators for cold items (DropoutNet-as-generator, CLCRec, Heater, GAR, meta-embeddings, MWUF, CB2CF); cold / warm score-distribution alignment (ALDI, score regulariser); inference-time embedding or magnitude normalisation; weight-space extrapolation of the fine-tune direction.

## 1. What constrained the search (project records read first)

Read before searching: `reports/CURRENT_STATUS.md` sections 1-6, `docs/02_SOLUTION.md` sections 1-6, `reports/COVERAGE_MAP.md` (all 351 lines), `reports/STALE_1115/{hz2,veto,staleft}.md` (verdicts and re-open list), `reports/LIT_1115/{SYNTHESIS,CRITIC,int_moe}.md`, `reports/FEATURES_REVIEW.md` FR-3 / FR-4, `reports/featnew_it_hist_hz.md`, `reports/review_ideas.md` finding 3, `reports/eda_shift_overlap.md` section 0, `reports/AUTOFE_1114.md` section 6.2, and by search the earlier scans (`EXTERNAL_SCAN_1113`, `TRICKS_SWEEP_1113`, `RESEARCH_SWEEP_1112`).

Closed rows that cover this topic [R, `reports/COVERAGE_MAP.md` line numbers]:

| family from the task text | closed row(s) | what was measured |
|---|---|---|
| train with stale-looking inputs | l.144, l.334-338 (hz, hz2, hz2n) | pool level pass (stale-val +0.0010..+0.0012), file level -0.00041 / -0.00047 / -0.00045 on three member sets |
| age-conditioned or uniform blanking of the stale inputs | l.52, l.53, l.339 (xd30 in the cooldown) | stale-val +0.0009 (z 1.4) with holdout -0.0019; xd30 val -0.00066, stale-val 0; from-scratch `kwxd15 / kwxd30` are PARKED, not closed (`CURRENT_STATUS.md` section 4) |
| fine-tune on stale inputs | l.340 (STALE-FT) | +0.0002 on a synthetic witness, 0 at the private mix |
| time / lag / example-age conditioning | l.239, l.276, LIT_1115 M4 (prior 0.02) | weekday + hour null; lag input never run, needs stale views and extrapolates from simulated lags |
| counter transforms | l.78 (hzcv, it_hist_hz, causal, lagged), l.277 (time gaps made split-stationary), l.201 (region-normalised), l.264 (trend / lifecycle) | all null or a cold / warm trade; l.323: closed in residual or GBDT form only |
| test-time adaptation | l.102 (test pseudo-labels), l.168 (TTT / Mean-Teacher, SCTTA), l.306 (test-time standardisation -0.0024..-0.0057), EXTERNAL_SCAN T16 (TENT, P 0.01) | negative or closed by mechanism |
| importance weighting, invariance | l.121 (ESS 2,880), RESEARCH_SWEEP 2.5 (DRO / IRM / CORAL), l.138 (recency rec33 -0.0021) | closed |
| cold rows: offsets, gates, experts, alignment | l.120, l.280-281, l.343, LIT_1115 M5-M14, CRITIC section 5 | oracle is a label echo; learned 0; row-level gate -0.00011 +- 0.00003; popularity-light weight on cold rows only -0.0006; the admixture pays through the order among WARM rows (`reports/LIT_1115/int_moe.md` l.37) |
| cold rows: content-completed collaborative signal | LIT_1115 G1, CRITIC 2.3 | detectable (LR 11.4) but expected worth +0.00004..+0.00035; no build |
| fresh counters from val / test feeds | l.345 | no signal over the pool |
| popularity debiasing across time | l.252 (PDA / TIDE), l.157 | closed |

What the records left open and this report used: the direction "make TEST rows look like TRAIN rows at inference" was never run. Every horizon-parity construction on record goes the other way (thin or lag the train rows, then retrain); the one idea in the test-to-train direction (`reports/review_ideas.md` idea 5, "val/test snapshot occurrences x 37") was replaced by `it_hist_hz` before it was tried and judged with a GBDT (`reports/featnew_it_hist_hz.md` l.30) [R].

## 2. Measurements made for this report

All scripts and logs: `reports/TRICKS_1115/web_shift_probe/`. Labels were read by `s4_read.py`, `s5_resid.py`, `s6_nested.py`, `s9_pmix.py` only; each under a block of `PREDECLARED.txt` written first (21:23Z, 21:25Z, 21:26Z), except s9 (information, after s4, said so in its header). Things tried on labels in total: 2 input arms (A, B), 1 residual statistic, 1 nested offset.

### 2.1 The count is a function of exposure length E, the same in every split [M, label-free, `s1_growth.log`]

E = dense cutoff (2026-09-20 00:00 MSK) minus the listing's posting time (`feat_new/hz_common.expo_hours`; posting time from f4). Mean log1p(`it_hist_all_luo`) and zero share by E:

| E before the cutoff | val | test public | test private | in-period holdout (train rows) |
|---|---|---|---|---|
| E < 0 (created after the cutoff) | 0.113 / zero 0.890 / 20.1% of rows | 0.113 / 0.888 / 27.9% | 0.088 / 0.911 / 31.3% | no such rows |
| 0-1 d | 0.285 / 0.750 | 0.275 / 0.757 | 0.263 / 0.767 | 0.163 / 0.854 |
| 2-4 d | 0.856 / 0.474 | 0.790 / 0.504 | 0.757 / 0.514 | 0.749 / 0.547 |
| 7-10 d | 1.407 / 0.314 | 1.333 / 0.333 | 1.316 / 0.341 | 1.307 / 0.368 |
| 10-14 d | 1.700 / 0.257 | 1.642 / 0.273 | 1.604 / 0.278 | 1.636 / 0.298 |
| 14-20 d | 1.974 / 0.217 | 1.910 / 0.232 | 1.878 / 0.239 | 2.165 / 0.185 |
| > 20 d or capped | 2.250 / 0.181 / 53.9% | 2.205 / 0.192 / 52.0% | 2.189 / 0.195 / 51.0% | 2.443 / 0.149 / 62.6% |

Reading: at equal E the four populations read the same count to within 0.1 log unit up to 14 d. A train row of a listing of age a has E = a + u (u = time from the row to the end of the dense data; mean 183.6 h, `s2_build.log`), because the as-built count is not causal [R, `reports/STALE_1115/hz2.md` section 1]; a val / test row has E = a - lag. The whole train-to-test difference of this input is that displacement of E.

### 2.2 Size of the deflation, by listing age and by day [M, label-free, `s2_build.log`, `s7_test_delta.log`]

Warm rows only (count > 0). Mean log1p(count) at equal listing age:

| listing age | in-period holdout (what the ranker learned) | val rows as stored | val rows after equalisation (arm A) |
|---|---|---|---|
| 0-1 d | 1.885 | 0.995 | 1.931 |
| 2-4 d | 2.324 | 1.138 | 2.168 |
| 4-7 d | 2.471 | 1.537 | 2.334 |
| 7-10 d | 2.570 | 1.762 | 2.431 |
| 10-14 d | 2.655 | 2.051 | 2.538 |
| 14-20 d | 2.725 | 2.375 | 2.618 |
| 30-60 d | 2.888 | 2.714 | 2.762 |

Definition of the correction (no label, no fitted parameter): `log1p(c') = log1p(c) + kappa * clip(m_ref(age) - m(E), 0, 2.5)` on warm rows; `m(E)` = mean log1p(count | warm) at exposure E on val + test rows pooled (monotone, 22 bins); `m_ref(age)` = mean of `m(age + u)` over the train row days. Zero rows and capped-age rows are untouched. delta = the clipped difference.

| rows | lag (d) | cold share | share of all rows with delta > 0.05 | mean delta on those | E < 0 and warm |
|---|---|---|---|---|---|
| val Tue | 2.58 | 0.358 | 0.253 | 0.411 | 0.018 |
| val Thu | 4.55 | 0.402 | 0.238 | 0.441 | 0.027 |
| test Sat | 6.58 | 0.424 | 0.228 | 0.463 | 0.031 |
| test Sun | 7.60 | 0.439 | 0.218 | 0.472 | 0.031 |
| test Mon | 8.58 | 0.454 | 0.207 | 0.471 | 0.028 |
| test Tue | 9.61 | 0.467 | 0.199 | 0.469 | 0.025 |

Reading: the deflated-warm population and the size of its deflation are the same on val and on test. Val is therefore a faithful witness for any correction of this kind, and nothing of this kind can be "larger on private". The val-to-test increment of staleness is rows turning cold.

### 2.3 The pools under-score lag-deflated warm rows; they are unbiased between cold and warm [M, `s5_resid.log`]

Listwise conditional logit on val, target y / n_pos, logit = b1 z(pool) + b2 1[cold row] + g delta; feed-clustered SE.

| pool branch (`preds/ens_<tag>_{to,ft}_val.npy`) | g (delta) | cold dummy without the delta term |
|---|---|---|
| W16kd train-only | +0.2002 +- 0.0322 (z 6.22) | +0.0083 +- 0.0180 |
| W8kd train-only | +0.1993 +- 0.0322 (z 6.19) | +0.0055 +- 0.0180 |
| FD16P20kd train-only | +0.1950 +- 0.0322 (z 6.05) | -0.0119 +- 0.0180 |
| W8P20kd train-only | +0.1953 +- 0.0323 (z 6.05) | -0.0083 +- 0.0180 |
| W16kd / W8kd fine-tuned (out of fold) | +0.0735 / +0.0736 +- 0.0321 (z 2.29) | -0.0223 +- 0.0179 |
| FD16P20kd / W8P20kd fine-tuned | +0.0824 / +0.0856 +- 0.032 (z 2.56 / 2.66) | -0.0277 / -0.0261 +- 0.0179 |

Reading: (a) the closed cold offset is confirmed from a new angle (no branch has a cold dummy beyond 1.6 SE); (b) the val fine-tune removes about 60% of the lag mis-scoring, which is one concrete thing the fine-tune half does; (c) the popularity-light admixture does not change it (0.195 against 0.200).

### 2.4 Output-offset form: NOT A CANDIDATE [M, `s6_nested.log`; rule in `PREDECLARED.txt` addendum 2]

score' = z(pool) + c delta, c fitted by the conditional logit on 4 user-grouped folds, applied to the fifth.

| file | nested delta | M < 50 | M >= 50 | in-sample g |
|---|---|---|---|---|
| FD16P20kd final (0.5 z(FT) + 0.5 z(TO)) | -0.00012 +- 0.00031 | -0.00067 +- 0.00057 | +0.00026 +- 0.00035 | +0.139 +- 0.032 (z 4.33) |
| W8P20kd final | -0.00008 +- 0.00030 | -0.00018 +- 0.00055 | -0.00001 +- 0.00034 | +0.141 +- 0.032 (z 4.38) |
| FD16P20kd train-only / fine-tuned branch | +0.00021 +- 0.00036 / -0.00011 +- 0.00024 | | | |

Dose bins (in-sample, FD16P20 final): delta 0.05-0.3 -> +0.044 +- 0.025; 0.3-0.8 -> +0.057 +- 0.028; > 0.8 -> +0.174 +- 0.038 (monotone). Rule: nested <= +1 SE -> not a candidate. The mis-scoring is real and worth nothing as an additive offset.

### 2.5 Input form at inference: INCONCLUSIVE, positive lean [M, `s4_read.log`, `s9_pmix.log`; rule in `PREDECLARED.txt` first block]

The 5 `it_hist_*_luo` train-stage inputs of val rows were replaced by their equalised copies (`cf/A_*.npy` kappa 1.0, `cf/B_*.npy` kappa 0.5) and the held-out students `w5_e10_kd05w192_s951..954` were re-predicted on CPU (fp32; `s3_predict.py`, 78-109 s per member pass). The CPU baseline equals the stored GPU val files (within-array correlation 1.000000 for each of the 4 members). 4-pool baseline 0.40122.

| read | arm A (full dose) | arm B (half dose) |
|---|---|---|
| 4-pool val delta | +0.00008 +- 0.00041 (z 0.19) | +0.00030 +- 0.00025 (z 1.19) |
| M < 50 / M >= 50 | +0.00021 +- 0.00073 / -0.00002 +- 0.00047 | +0.00017 +- 0.00044 / +0.00039 +- 0.00030 |
| Tue (lag 2.6 d) / Thu (lag 4.6 d) | -0.00005 / +0.00021 | +0.00013 +- 0.00034 / +0.00049 +- 0.00038 |
| feeds whose positives are all cold (3,422) | -0.00756 +- 0.00065 | -0.00330 +- 0.00040 |
| feeds with no cold positive (9,118) | +0.00323 +- 0.00056 | +0.00174 +- 0.00036 |
| feeds with >= 30% rows changed (3,244) | +0.00083 +- 0.00101 | +0.00083 +- 0.00064 |
| top-1 changed | 11.0% of feeds | 5.0% of feeds |
| members (s951 / 952 / 953 / 954) | -0.00018 / -0.00027 / +0.00012 / +0.00006 | -0.00014 / -0.00031 / +0.00004 / +0.00026 |
| reweighted to the public mix (information) | +0.00055 +- 0.00053 | +0.00036 +- 0.00033 |
| reweighted to the private mix (information) | +0.00091 +- 0.00067 | +0.00054 +- 0.00041 |

Rule read: best arm +1.19 SE < 2.3 SE -> no SUPPORT; both arms > 0 -> not REFUTED; INCONCLUSIVE. By the screen the veto audit used for this family ("val >= 0 and pmix-private >= +1 SE", `reports/STALE_1115/veto.md` verdict 2) arm B would pass at train-only 4-pool level; that screen let hz through and hz failed at file level, so it is quoted as a lean, not as evidence. Two looks, one sample, in-sample val below +0.001: lesson 1 applies in full.

### 2.6 Effective sample size of the test-mix weights on val [M, `s8_pmix_ess.log`]

`src/pmix.py` weights over the 14,423 val feeds: public ESS 9,321 (64.6%), max weight 3.6; private ESS 6,389 (44.3%), max 6.0, q10 0.24, q90 1.87. Used by proposal P2.

### 2.7 What the probes add up to [I]

- The staleness cost has a bias part (the ranker reads deflated counts with fresh-count semantics) and an information part (rows with no count; old, short observation windows). The bias part is now measured directly: it is a 6-SE calibration error and a cold / warm trade whose net is +0.0001..+0.0003 on the train-only branch at val's mix. The information part is what grows from val to test.
- This agrees with three records: the fine-tune on stale inputs recovers +0.0004..+0.0005 of a -0.0017 cost on a synthetic witness ("mostly lost information", `reports/STALE_1115/staleft.md` verdict 4); hz gains +0.0010..+0.0012 on stale-val at pool level and nothing at file level; the cold offset is null.
- It bounds the families "time-aware or age-conditioned normalisation of counters", "score calibration between warm and cold candidates" and "label-free adaptation at test time" from above by about +0.0003 at branch level, +0.00015 at file level, unless they bring fresh information.

## 3. Literature by topic, mapped to this pipeline

| topic | what the sources say [W] | what this project has that the technique needs | status here | verdict |
|---|---|---|---|---|
| Feature staleness, training-serving skew | Rules of ML #29 / #31 / #37: log the features used at serving and train on them; a table joined at training and at serving time may have changed. FeSAIL defines staleness for ID embeddings only (replay + update regulariser, +0.8..1.7% relative AUC); nothing on numeric counters. IEFF (Meta, 2026): zeroing a feature's coverage at once costs 0.83% vs 0.37% when faded over days WITH the faded values logged into training; a model is never served on a coverage it was not trained on. RecSys Challenge 2023 9th: per-feature adversarial validation, features with AUC >= 0.75 dropped; counts over the previous week chosen over the previous day. WNS 2019 winner (search summary only): train features rebuilt from a disjoint log slice so that they match how test features arise | timestamps, posting time from f4, the hz2 builder that replays the test observation process on train rows | "train like you serve" = hz / hz2 (pool pass, file fail); "drop the drifting feature" = the popularity-light view (in both picks) | no new arm; confirms the two existing levers are the standard ones |
| Missing or occluded features at serving | DropoutNet: blanking the preference input from the START of training costs under 1% warm recall up to rate 0.7 and lifts cold recall by over 60% at rate 0.1. Opendoor: with naive imputation it is better to drop the feature above 30% occlusion; one model trained on stacked blanked + unblanked copies with an occlusion flag equals two separate models. Marginalized Corrupted Features: train on the expected loss under the corruption. Le Morvan et al.: impute-then-predict is Bayes-optimal only when the predictor is trained on the imputed data | test zero share 0.43-0.46 [M 2.2] against 0.13 on train rows [R, `reports/STALE_1115/veto.md` 5.1]: above the 30% break-even | cooldown-only blanking closed (xd30); from-scratch blanking parked; popularity-light = rate 1.0 as a separate model | P3 (parked arm, dose and form from the literature) |
| Delayed feedback | ES-DFM (read); Chapelle 2014 and DEFER (search summaries): a label observed for a short elapsed time is a censored version of the final label; correct by importance weights that depend on the probability of observation within the elapsed time (ES-DFM +0.3..0.4% AUC) | exposure length E is known for every row; the count is a censored quantity in exactly this sense | new as an INPUT correction: sections 2.1-2.5 | P1 (run today, inconclusive); output-offset form closed |
| Time-aware normalisation, temporal modulation | Feature-aware temporal modulation (2512.03678): per-feature scale / shift / skew conditioned on a time embedding, +2.09% over a static MLP on TabReD, input-level modulation carries 87% of it; extrapolation beyond the training period is claimed, not tested. YouTube example age (search summary only): train with the example's age, serve at zero | train days Sep 6-19, test Sep 26-29: the conditioning variable is out of support at test | lag / time inputs closed or prior 0.02 (M4) | not proposed; P1 is the label-free, in-support form of the same goal |
| Temporal-shift benchmarks | Wild-Time: 13 methods (invariance, continual, self-supervised, ensembles) do not close a 20% average drop. TabReD: CORAL and DFR rank 1.4 against 1.0 for the plain MLP; retrieval models lose their edge under time splits (TabR rank 6.0 vs MLP 5.0); ensembles help (MLP-PLR rank 3.8 -> 2.4). ELBO_TDS (Shopee, 2025): IRM / V-REx / RVP are 0.015-0.020 AUC BELOW ERM; direct augmentation of statistical features = ERM (0.7982 vs 0.7977); the gain (+0.0065 click AUC) needs a self-supervised objective and one pass over each day | - | DRO / IRM / CORAL closed; same-data SSL closed (E2); pool is an ensemble | do not try (section 5) |
| Test-time adaptation | AdapTable: on tabular shift TENT 51.2, SAR 50.0 against the unadapted 53.2 (HELOC); gains come only from re-estimating the label marginal. FTAT (abstract read; the finding is quoted from a search summary of the paper): existing adaptation methods "fail to surpass the non-adaptation baseline" as shift grows. DT3OR, T2ARec: test-time objectives for ID-based sequential recommenders; abstracts give no number. BBSE: label-shift correction needs p(x given y) fixed. Agreement-on-the-line: label-free accuracy estimates with 0.3-4% absolute error | unlabelled val and test feeds | pseudo-labels, TTT, standardisation closed; a listwise ranker has no label marginal within a feed; the one cross-row marginal (cold vs warm) is unbiased [M 2.3] | do not try |
| Importance weighting, continual schedules | Byrd & Lipton: the effect of importance weights fades over epochs in deep networks, it survives in short training. SML: a transfer between the old model and the new-data model trained for NEXT-period performance beats fine-tuning (Yelp recall@10 0.338 vs 0.239) and full retraining; plain fine-tuning wins only on a news set. Time vectors: scaling a fine-tune direction improved future periods in some tasks and hurt in others. Facebook 2014 (search summary only): accuracy decays with the train-test delay | val = the only in-distribution labelled period; its mix differs from test by feed size x cold share | train-row weighting closed (ESS 2,880; rec33); `w_ft` closed; FT loss weights never varied | P2 (weights at the fine-tune stage, where the shift is covariate-only) |
| Cold items from content | DropoutNet, CLCRec, Heater, GAR, meta-embeddings, MWUF, CB2CF: produce a collaborative EMBEDDING for an item that has none. RecSys 2025 (Meehan & Pauwels): a content kNN beats Heater / GAR / GoRec on cold items (NDCG@20 0.0738 vs 0.0640) and those generators inherit popularity bias. YouTube fresh content (KDD 2023): fresh items get their own slot and a content-only tower; no ranking-calibration recipe | content embeddings for every item, fresh f4 / f5 | the ranker has no item-ID embedding (ID table -0.005, l.300); it is already the content model; "cold" = 8 of 104 dense inputs at zero | do not try |
| Warm and cold in one list | Pinterest (2512.17277): cold scores 8-14% lower; MMD score regulariser + residual path for non-historical inputs + mixup: fresh-item Hit@3 +3.2%, ALL items -0.04 / -0.13. ALDI: align cold and warm rating distributions | - | cold dummy 0 +- 0.018 on 8 branches [M]; cold offset closed; row gates closed | do not try: the literature's own all-item number is zero |
| Popularity debiasing across time | PDA: inject a linear popularity forecast at inference: +1.7..3.9% over last-stage popularity on two sets, 0 on the third. TTEN: normalise item-embedding norms at test time | - | PDA / TIDE closed; trends closed (C07); no embedding norm to normalise | do not try |
| Horizon-matched features in forecasting | M5 winner (DRFAM): simple average of "direct" models (features lagged by the horizon) and "recursive" models (fresh features); the two are complementary | hz re-cools = direct members | plain + hz pooled: -0.00015 +- 0.00017 [R, ledger 14:47Z] | closed |

## 4. Ranked proposals

Ranking = expected gain on the final file x probability. Costs use the records' units: an 8-seed student arm 12-31 GPU-h (`reports/LIT_1115/SYNTHESIS.md` l.25), 16 x 3 fine-tunes 3 GPU-h (`reports/STALE_1115/veto.md` section 5) [R]; CPU re-prediction 78-109 s per member on val [M].

### P1. Lag-equalised item-occurrence inputs at inference (EQ-B), train-only branch. Expected +0.0002 public / +0.0003 private [I]; P(true file gain >= +0.0002 on private) 0.20 [I]

- What: at inference only, replace the 5 `it_hist_*_luo` train-stage inputs of val and test rows by `expm1(log1p(c) + 0.5 * delta)` on warm rows (section 2.2). Members, checkpoints and the fine-tuned branch stay as they are. File form: 0.5 z(FT branch, unchanged) + 0.5 z(0.8 z(16 wide members re-predicted on equalised inputs) + 0.2 z(popularity-light members, unchanged; they do not read these inputs)).
- Why it could pay: sections 2.1-2.3. Why it may not: section 2.4, and the trade in section 2.5 (-0.0033 on feeds whose positives are all cold). The half-dose arm was the better of two looks.
- Attachment: test columns are built (`web_shift_probe/cf_test/B_it_hist_*.npy`, label-free, `s10_build_test.py`; the val curve reproduces to 6e-8). Re-prediction = the `feat_path` switch of `nn_x/w5/stale_pred.py` pointed at `cf/B_*` (val) and `cf_test/B_*` (test); `s3_predict.py` is the CPU form for val. Then `nn_x/w12/kd/build_r4.py` with the re-predicted train-only member files under new names (a copy of the key FD16P20; nothing existing is overwritten).
- Cost: 0 GPU-h. CPU: 16 members x (val + test) about 16 x 5 min; lagged-holdout read about 1 h more (below).
- Rule, written here before any further run. Step 1 (lagged holdout, the only read with more feeds than val): on the 13,424 holdout feeds of days >= Sep 13, held-out W8 members, inputs = the hz2 causal test-like view (`nn_x/w12/stale/hz2/`, reader `lagged_pair_read.py`), arm = the same view with the equalisation applied with E taken against each feed's own cutoff; CONTINUE iff 8-pool arm minus view >= +2 SE; STOP and close if <= +1 SE. Step 2 (final file on val, 16 full-data members): BUILD iff final val delta >= 0 AND pmix-private >= +1 SE AND M < 50 >= -1 SE. Step 3: ONE public slot; the file is kept only as an experimental pick-2 option if public >= FD16P20 - 0.5 paired SE; no dose scan on public.
- Closest closed rows: l.78 / `it_hist_hz` (density matching of TRAIN rows, GBDT judge), l.306 (global test-time standardisation with val statistics, -0.0024..-0.0057), l.120 (cold offset), l.337 (hz). Difference: test-to-train direction, conditional on listing exposure, warm rows only, no retraining, the reference is the label-free growth curve; and it has been run through the actual members (section 2.5), not through a residual model.

### P2. Fine-tune with test-mix feed weights (MIXFT). Expected +0.0001..+0.0002 private [I]; P(true gain >= +0.0002 private) 0.15 [I]

- What: in the val fine-tune, multiply each feed's listwise loss by a tempered density ratio w^0.5, w = the `src/pmix.py` private-mix weight of the feed (feed size x cold share), normalised to mean 1 within the training folds. Test inference unchanged.
- Mechanism: the fine-tune is the only stage whose data differ from test by covariates alone (val and test match on every intrinsic covariate, PSI <= 0.01; the drift is the feed mix [R, `GLOBAL_RECAP_1113.md` l.71]); the fine-tune half is worth more on colder feeds (+0.00066 at the public mix, +0.00104 at the private mix [R, STATUS section 4]); importance weights act in short training (Byrd & Lipton) and the fine-tune is 4 epochs at lr 5e-5. A parallel study reaches the same reading of where the fine-tune pays: the val-to-public gap of the fine-tune half is reproduced by the feed mix [R, `reports/TRICKS_1115/int_ft.md` verdict 3]. Tempering because the full private weights leave ESS 44% [M 2.6].
- Attachment: a copy of `nn_x/w5/finetune.py` with one argument `--feed_w <npy>` applied where `listwise_loss(lg, y, m, norm=a.loss_norm, ...)` is called (l.249), recipe n2cpv3 otherwise; lane `nn_x/w12/kd/ft_lane.sh`. One fine-tune seed is enough for a read (`staleft.md` verdict 7c) [R].
- Cost: 8 W8 members x 1 seed, arm + same-script control: about 1 GPU-h; 30 lines of code.
- Rule: final file (W8P20 form, 1-seed branches on both sides): pmix-private (arm minus control) >= +1.5 SE AND plain val >= -1 SE AND M < 50 >= -1 SE -> fine-tune the 20 full-data members with 3 seeds (3 GPU-h) and build; otherwise closed. Honest limit: the pmix-private SE of such a contrast is about 0.0003-0.0005, so only an effect of +0.0006 or more passes; a true +0.0002 passes in roughly one run of five [I]. Public cannot see it either (it targets the private mix).
- Closest closed rows: l.121 (reweighted val / holdout as a JUDGE), RESEARCH_SWEEP 2.5 (importance weighting of TRAIN rows, ESS 2,880), l.303 (group-conditional FT blend weight at inference). Difference: weights inside the fine-tune loss; ESS 44-65% instead of 2,880 of 3.1M.

### P3. From-scratch blanked students at the test cold share, stacked form with an occlusion flag (kwxd45i). Expected +0.0003 private, 0 public [I]; P 0.10 [I]

- What: the parked `E:kwxd15 / kwxd30` arm with three changes taken from the literature: rate 0.45 (the test zero share 0.43-0.46, not 0.15-0.30); blanking drawn per LISTING within a batch rather than per row; one extra input = 1 where the 8 inputs were blanked or are all zero (the stacked form, in which the unblanked half of the rows keeps full-strength counters).
- Evidence: DropoutNet (from-scratch blanking nearly free for warm items up to 0.7); Opendoor (above 30% occlusion a naive model is worse than dropping the feature; stacked = two models); here the cooldown-only form failed (xd30: val -0.00066, stale-val 0, l.339) and the full-dose separate model works only at weight 0.2.
- Attachment: `nn_x/abl/train.py` carries every flag needed (`--xt_drop` l.183, which blanks `xt_idcols` 0-7; `--xt_add` l.282; `--kd_file` l.215) [M, grep]; the flag column is one `--xt_add` input (1 where the stored 8 inputs are all zero) plus a 5-line change in a COPY of `model.py` so that the indicator is also set on blanked rows; per-listing blanking is a change of the mask draw in the same copy. Command = the parked `E:kwxd` line with `--kd_file teacher_mean11.npy --kd_w 0.5 --d 192 --xt_drop 0.45`.
- Cost: 8 seeds, 12-31 GPU-h, then 3 GPU-h of fine-tunes on a pass.
- Rule: the staleness-family rule on record ("RULE hz"): 8-pool against kd05w192 on the same seeds, val >= 0, stale-val (scenario SVL) >= +2 SE, val M < 50 >= -1 SE; then the FILE rule that hz failed (final val >= twin - 1 SE, pmix-private >= 0). Reason for the low prior: hz passed the first rule and failed the second on three member sets; this arm is of the same family and substitutes for the popularity-light admixture.
- Closest closed rows: l.52 (uniform dropout, old recipe), l.53 (age-conditioned, hard labels, holdout veto), l.339 (xd30 cooldown-only). Difference: from scratch under distillation, test-matched rate, per-listing, with the occlusion flag.

### P4. The equalised view as a trained student view. Expected +0.0000..+0.0003 [I]; P 0.07 [I]; run only if P1 passes step 1

- What: 4-8 wide students whose 5 `it_hist` inputs are equalised to a common reference exposure on ALL splits (train rows too: `log1p(c) + m(top) - m(E)`), so that the input has the same meaning at every lag by construction; fine-tune with the same view.
- Cost: CPU build about 1 h (E for 246.7M train rows by `hz_common.expo_hours`), 12-31 GPU-h for 8 seeds.
- Rule: as P3. Closest rows: hz2 (l.338; truncates train rows and loses information on them), l.277 (within-feed z / rank, residual form). Difference: keeps every train count, changes only its level.

## 5. Do not try, with reasons

| technique | literature number [W] | project number | reason |
|---|---|---|---|
| Entropy / batch-statistic test-time adaptation (TENT, EATA, SAR, test-time training) | AdapTable: TENT 51.2, SAR 50.0 vs source 53.2 on HELOC; FTAT: no gain as shift grows | test-time standardisation -0.0024..-0.0057 (l.306); test pseudo-labels negative (l.102) | the shifted inputs are information; LayerNorm model, no batch statistics; same-user later feeds are banned |
| Label-shift correction (BBSE, label-distribution handler) | needs p(x given y) fixed | cold dummy 0 +- 0.018 in 8 branches [M 2.3] | a per-feed listwise ranker has no class marginal to re-estimate; the only cross-row marginal is already unbiased |
| Domain-adversarial / IRM / CORAL / list-level invariance (ListDA) against time | Shopee: IRM 0.7775, V-REx 0.7827 vs ERM 0.7977 AUC; TabReD: CORAL, DFR rank 1.4 vs MLP 1.0; Wild-Time: no method closes the gap | l.121, RESEARCH_SWEEP 2.5 | the val-to-test difference (cold share) predicts the label; making the representation blind to it removes signal |
| Recency weighting of train rows, recency windows | SML: fine-tuning on new data alone loses on a long-interest set (0.239 vs 0.288) | rec33 -0.0021 / -0.0012 (l.138) | the newest production regime transfers negatively to val [R] |
| Popularity forecast at inference (PDA) | +1.7..3.9% over last-stage popularity on two sets, 0 on the third | C07 trends closed; extrapolated category trends hurt -0.00035 | 14 days of dense data, 6-9 day gap |
| Content-to-collaborative generators (CLCRec, Heater, GAR, meta-embeddings, MWUF, CB2CF) | a content kNN beats three of them on cold items (0.0738 vs 0.0640) | ID table -0.005 (l.300); neighbour-completed scores worth +0.00004..+0.00035 (CRITIC 2.3) | they create an ID embedding for a model that has none; the ranker is already the content model |
| Cold / warm score alignment (ScoreReg, ALDI), residual path for non-historical inputs | Pinterest: all-item Hit@3 -0.04 / -0.13 for the combination | cold offset 0; popularity-light weight on cold rows only -0.0006 (int_moe) | the metric here is over all items; the literature's own all-item effect is zero |
| Output offset by lag deflation | - | nested -0.00012 +- 0.00031 [M 2.4] | closed today by its rule |
| Inference-time norm / magnitude scaling (TTEN, magnitude scaling) | user-oriented accuracy unchanged or down to -10% | - | acts on ID-embedding norms; none here |
| Weight-space extrapolation of the fine-tune direction (time vectors, SML) | time vectors: gains in some tasks, losses in others; uniform "time soups" far below joint training (0.61 vs 0.79 F1) | `w_ft` != 0.5 nested +0.0002 (z 0.96), l.100 | no labelled period after val to set the scale; public paired SE 0.0003-0.0006 cannot read a +0.0002 step |
| Label-free choice of the popularity-light weight for private days (agreement-on-the-line) | estimation error 0.3-4% absolute accuracy where the line holds | weight 0.3 not supported on the lagged holdout (l.336) | the estimator's error is two orders above the +0.0003 at stake |
| Direct + recursive averaging (M5) | complementary in forecasting | plain + hz pooled -0.00015 +- 0.00017 | run today in the ledger |

## 6. Sources (opened and read unless marked "search result only")

| # | source | URL | what it contributed |
|---|---|---|---|
| 1 | FeSAIL, IJCAI 2023 | https://arxiv.org/html/2505.02844v1 | staleness defined for ID embeddings; nothing on numeric counters |
| 2 | Warmer for Less, Pinterest 2025 | https://arxiv.org/html/2512.17277v1 | cold-score gap 8-14%; three fixes worth +3.2% on fresh items and -0.04 / -0.13 on all items |
| 3 | ELBO_TDS, Shopee 2025 | https://arxiv.org/pdf/2511.21032v1 | statistical features drift most; augmentation alone = ERM; IRM-type below ERM |
| 4 | DT3OR | https://arxiv.org/abs/2407.15620 | test-time training for ID recommenders (abstract only, no number) |
| 5 | T2ARec | https://arxiv.org/abs/2504.01489 | test-time alignment for sequential models (abstract only) |
| 6 | Marginalized Corrupted Features | https://arxiv.org/abs/1402.7001 | training on expected loss under feature blanking (abstract) |
| 7 | Le Morvan et al., NeurIPS 2021 | https://arxiv.org/abs/2106.00311 | impute-then-predict is optimal only for a predictor trained on imputed data (abstract) |
| 8 | Test-Time Embedding Normalization, CIKM 2023 | https://arxiv.org/pdf/2308.11288 | score = cos x norm^(1-p); acts on embedding norms only |
| 9 | Inherited popularity bias, RecSys 2025 | https://arxiv.org/html/2510.11402v1 | content kNN beats cold-start generators; magnitude scaling formula |
| 10 | ALDI, SIGIR 2023 | https://web.comp.polyu.edu.hk/xiaohuang/docs/2023SIGIR_Feiran.pdf | warm-teacher / cold-student distribution, ranking and identification alignment |
| 11 | Feature-aware temporal modulation | https://arxiv.org/html/2512.03678v1 | time-conditioned scale / shift / skew of inputs; +2.09% on TabReD; extrapolation untested |
| 12 | TabReD, ICLR 2025 | https://arxiv.org/html/2406.19380v4 | CORAL / DFR below MLP; retrieval loses under time splits; ensembles help |
| 13 | ListDA, NeurIPS 2023 | https://arxiv.org/abs/2212.10764 | list-level domain-invariant ranking (abstract) |
| 14 | FTAT, AAAI 2025 | https://arxiv.org/abs/2412.10871 | tabular test-time adaptation; existing methods fail under stronger shift |
| 15 | AdapTable | https://arxiv.org/html/2407.10784v4 | TENT / SAR below the unadapted model; label-marginal re-estimation is the only gain |
| 16 | PDA, SIGIR 2021 | https://ar5iv.labs.arxiv.org/html/2105.06067 | popularity forecast injected at inference; small and dataset-dependent |
| 17 | Time vectors, ACL 2024 | https://arxiv.org/html/2312.13401v2 | fine-tune directions interpolate; extrapolation inconsistent |
| 18 | BBSE | https://arxiv.org/abs/1802.03916 | label-shift estimation from a confusion matrix (abstract) |
| 19 | Agreement-on-the-line | https://ar5iv.labs.arxiv.org/html/2206.13089 | label-free OOD accuracy, error 0.3-4% where the line holds, about 5% where it does not |
| 20 | SML, SIGIR 2020 | https://ar5iv.labs.arxiv.org/html/2005.13258 | transfer trained on next-period data; fine-tune vs retrain depends on the data |
| 21 | Fresh content needs more attention, KDD 2023 | https://ar5iv.labs.arxiv.org/html/2306.01720 | dedicated slot and content-only tower for fresh items |
| 22 | DropoutNet, NeurIPS 2017 | https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2017/dropoutnet_addressing_cold_start_in_recommender_systems__e121d7ba.txt (text mirror of the NeurIPS paper) | from-scratch input blanking: warm cost under 1% to rate 0.7 |
| 23 | MWUF | https://arxiv.org/abs/2105.04790 | scaling / shifting networks for cold ID embeddings (abstract) |
| 24 | Meta-embedding, SIGIR 2019 | https://arxiv.org/abs/1904.11547 | generated initial ID embedding from content (abstract) |
| 25 | CLCRec | https://arxiv.org/abs/2107.05315 | contrastive content-collaborative alignment (abstract) |
| 26 | RecSys Challenge 2023, 9th place | https://arxiv.org/html/2310.03778v1 | per-feature adversarial validation; previous-week counts |
| 27 | IEFF, Meta 2026 | https://arxiv.org/html/2605.00324 | coverage zero-out 0.83% vs faded 0.37%; faded values logged into training |
| 28 | Taming missing features at serving time (Opendoor) | https://www.opendoor.com/articles/taming-missing-features-at-serving-time | 30% break-even; stacked training equals two models |
| 29 | ES-DFM, AAAI 2021 | https://ar5iv.labs.arxiv.org/html/2012.03245 | elapsed-time censoring and its importance weights |
| 30 | Rules of Machine Learning | https://developers.google.com/machine-learning/guides/rules-of-ml/ | rules 29, 31, 37 on training-serving skew |
| 31 | Wild-Time, NeurIPS 2022 | https://arxiv.org/abs/2211.14238 | 13 methods do not close the temporal gap |
| 32 | Feature shift detection and correction, NeurIPS 2023 | https://arxiv.org/abs/2312.04546 | correcting shifted features toward a reference set (abstract) |
| 33 | M5 accuracy winner (DRFAM) | https://econpapers.repec.org/RePEc:eee:intfor:v:38:y:2022:i:4:p:1386-1399 | average of horizon-matched and fresh-feature models (abstract) |
| 34 | Byrd & Lipton, ICML 2019 | https://arxiv.org/abs/1812.03372 | importance weights fade over epochs (abstract) |

Search result only (not opened; quoted nowhere above as a number except where stated): Chapelle KDD 2014 (delayed feedback), DEFER, Facebook "Practical lessons" 2014, YouTube example age (RecSys 2016), CB2CF, Heater, GAR, Courty et al. optimal transport, WNS 2019 winners' write-up, tempered importance weights (Shimodaira). Their one-line content comes from search summaries and carries less weight.

## 7. Files, commands, compliance

- Report: `reports/TRICKS_1115/web_shift.md`. Probe directory `reports/TRICKS_1115/web_shift_probe/` (189 MB): `PREDECLARED.txt`; `s1_growth.py` (.log); `s2_build.py` (.log) -> `cf/{A,B}_it_hist_*_luo.npy`, `cf/delta_val.npy`; `s3_predict.py` (`s3_p1.log`, `s3_p2.log`) -> `preds/cpu_w5_e10_kd05w192_s95{1..4}_{base,A,B}.npy`; `s4_read.py` (.log); `s5_resid.py` (.log); `s6_nested.py` (.log); `s7_test_delta.py` (.log); `s8_pmix_ess.log`; `s9_pmix.py` (.log); `s10_build_test.py` (.log) -> `cf_test/B_it_hist_*_luo.npy`, `cf_test/delta_test.npy`.
- Every probe command: `cd <repo> && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 .venv/bin/python reports/TRICKS_1115/web_shift_probe/<script> [args]` (`s3_predict.py <variant> ...`; `s5_resid.py W16kd W8kd FD16P20kd W8P20kd`; `s6_nested.py FD16P20kd W8P20kd`).
- Compliance: CPU only (two 12-thread inference processes for 10 minutes, PIDs 971839 / 971840, both ended on their own); no GPU job, no training; no process signalled; nothing written outside `reports/TRICKS_1115/`; every project file opened read-only (`PYTHONDONTWRITEBYTECODE=1`); nothing submitted.
- Not checked: the lagged-holdout read of P1 (step 1); the equalisation of the co-visitation inputs (`seq_cv_*` are causal on train rows and were left as stored); the fine-tuned branch under equalised inputs; a per-event-type curve (one delta was used for all five columns); the growth curve was fitted on val + test rows pooled and applied to val, which uses no label but is transductive. The priors and expected gains in section 4 are judgements.
