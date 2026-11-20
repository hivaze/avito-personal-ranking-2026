# RECHECK 1114: forensics_skeptic

Task: try to refute the nulls of `reports/TOP1_GAP_FORENSICS_1114.md` with a non-linear, interaction-capable but honest protocol.
CPU only, 12 threads, new files only. Scripts, logs and arrays: `nn_x/w12/recheck/forensics_skeptic/` (`hx.py` harness, `s00`..`s09`).

## 0. Verdict

**No null turns into a signal.** Nothing reaches +0.0005 at 2 SE in both directions of the user split, on val or on the holdout. The forensics report stands on all four re-tested points and on the raw-schema claim.

- (a) earlier labelled feeds of the same user: +0.00004 ± 0.00009 val (mean of five user splits +0.00004, largest +0.00027 ± 0.00013), +0.00004 ± 0.00004 holdout.
- (b) existing feature columns as a block: +0.00016 ± 0.00029 (12 columns picked inside the fit half), −0.00077 ± 0.00039 (all 246). The report's "best of 248, +0.00044 ± 0.00020" is selection noise: picking the column inside the fit half gives +0.00012 ± 0.00030 (mean over 5 splits −0.00030), and on the holdout the same column reads −0.00008 ± 0.00008.
- (c) "positive has zero popularity" pocket: defined by the label. W8 is calibrated on cold candidates (observed / expected positives 1.005 ± 0.011 val, 0.983 ± 0.010 holdout). No label-free indicator opens a better blend (best +0.00045 ± 0.00024, holdout ≤ 0).
- (d) "users seen in train are worse": −0.0051 ± 0.0040 after adjustment, not significant, and C3 and content-only scorers show the same sign. No group-conditional blend helps (−0.00003 ± 0.00030).
- Raw schema: all 19 raw columns are in `data/flat` bit-exactly (val and test in full, train on 6 random parts). No side file in the archive.

What the protocol can and cannot see (measured with planted signals, §1.3): an effect worth +0.003 is recovered at 75-83% (z 5); +0.0013 gives +0.0009 (z 2.8); +0.0005 is invisible. So "null" here means "below about +0.001", which is also the level below which near-copy files do not transfer to public.

Two side findings, both measured, neither a confirmed lever (§4):
1. The hard-label val fine-tune no longer helps KD students at member level: FT − TO = −0.00015 ± 0.00038 for students vs +0.00071 ± 0.00032 for hard-label members (difference −0.00086 ± 0.00024, z −3.6). At pool level FT − TO is about 0 for both.
2. The fine-tuned branch is worse than the train-only branch on candidates whose category the user clicked in an earlier labelled feed: trusting TO there gives +0.00032 ± 0.00016 (z 2.0, same in two splits). Below the bar; recorded as a hint only.

## 1. Protocol

### 1.1 Base and data
- Val base = W8 = z(0.5 z(FT) + 0.5 z(TO)), reproduced at 0.40165 (FT 0.40087, TO 0.40157). 14,423 feeds, 1,389,971 rows.
- Holdout base = W8 train-only pool (8 wide students), reproduced at 0.32456. 30,000 feeds, 4,630,966 rows. There is no FT branch on the holdout.
- Split: users in two halves (guest feeds split by feed), same seed as the forensics report. Fit on half A, score half B, and the reverse. ± is the paired per-feed SE.

### 1.2 Residual model
LightGBM lambdarank, `init_score = a·z(base)`, final score = init + trees. Inside the fit half only: scale `a`, 3 tree configs, number of rounds from {0, 10, 20, 40, 80, 150, 250} by an inner user-grouped 3-fold. 0 rounds means "leave the base alone" and is always a candidate. The other half is touched once.

Features: base z and base rank, the signal block, M, H, guest flag, hour.

**Monotone constraint (important).** The first version had no constraints. With base-only features it lost 0.002-0.006 at every round count in the inner folds (`v1_unconstrained_s02_val_s0.log`): free trees re-order the base by themselves. Version 2 puts a +1 monotone constraint on the two base columns. A model without a signal then cannot change any within-feed order (control arm: exactly 0.00000 on val and holdout), so every order change must come from the signal columns. All numbers below are version 2.

### 1.3 Power: planted signals on val (30% covered feeds, the real coverage mask of (a))

| planted signal | oracle value (in-sample) | linear nested (report's protocol) | LightGBM residual |
|---|---:|---:|---:|
| additive, large | +0.00297 | +0.00297 ± 0.00063 | +0.00223 ± 0.00041 (z 5.4) |
| interaction only (sign flips with M), large | +0.00325 | −0.00006 ± 0.00015 | +0.00253 ± 0.00050 (z 5.1) |
| additive, threshold | +0.00134 | +0.00134 ± 0.00051 | +0.00094 ± 0.00034 (z 2.8) |
| interaction only, threshold | +0.00203 / +0.00184 | −0.00016 / +0.00022 | +0.00063 ± 0.00036 / +0.00124 ± 0.00038 |
| additive, small | +0.00046 / +0.00032 / +0.00020 | +0.00046 / −0.00055 / +0.00020 | +0.00027 ± 0.00033 / +0.00017 ± 0.00032 / −0.00012 ± 0.00025 |
| interaction only, small | +0.00113 / +0.00111 | −0.00011 / +0.00023 | +0.00023 ± 0.00038 / +0.00067 ± 0.00027 |

- The linear test is blind to a pure interaction, as suspected. The residual model is not.
- Detection limit of the residual model on val: about +0.0013 for an additive signal (z about 2.8), nearer +0.002 for a pure interaction. Effects of +0.0005 or less cannot be told from zero by either protocol.
- Under the null the residual arm is slightly negative, not zero: the inner fold picks the best of 21 noisy cells (winner's curse), and the picked model is then applied out of half.

## 2. Results

Val base 0.40165, holdout base 0.32456. "halves" = delta on each scored half.

| # | test | val | val halves | holdout |
|---|---|---:|---:|---:|
| a | linear nested, 8 signals (replication) | −0.00036 .. +0.00012 | - | −0.00010 .. +0.00006 |
| a | residual, all feeds (29 bank columns + context) | +0.00004 ± 0.00009 | −0.00003 / +0.00012 | +0.00004 ± 0.00004 (any earlier feed); −0.00008 ± 0.00005 (gap ≥ 2 days) |
| a | same, 4 more user splits | +0.00027 ± 0.00013, −0.00000 ± 0.00016, +0.00008 ± 0.00029, −0.00017 ± 0.00020 (5-split mean +0.00004) | | - |
| a | residual, covered feeds only (as all-feed delta) | −0.00018 ± 0.00016 (4 more splits: +0.00014, +0.00002, +0.00004, +0.00009) | +0.00006 / −0.00041 | −0.00006 ± 0.00011; −0.00005 ± 0.00007 |
| b | cos_het1, linear nested (the report's best column) | +0.00045 ± 0.00020 | +0.00049 ± 0.00028 / +0.00040 ± 0.00029 | −0.00008 ± 0.00008 |
| b | best column + weight picked inside the fit half (linear) | +0.00012 ± 0.00030 | +0.00018 / +0.00006 | - |
| b | same, 4 more user splits | −0.00054, −0.00005, −0.00083, −0.00018 | | - |
| b | cos_het1, residual | −0.00072 ± 0.00026 | −0.00034 / −0.00110 | 0 (0 rounds chosen in both halves) |
| b | 12 columns picked inside the fit half, residual | +0.00016 ± 0.00029 | +0.00038 / −0.00007 | - |
| b | 4 columns picked inside the fit half, residual | +0.00028 ± 0.00019 | +0.00057 / 0 (0 rounds) | - |
| b | all 246 leak-safe columns, residual | −0.00077 ± 0.00039 | −0.00043 / −0.00111 | - |
| b | 133 own-feed label-free columns, residual | 0 (0 rounds chosen in both halves) | 0 / 0 | −0.00007 ± 0.00008 |
| c | best alternative signal with one weight per cold-share quartile | +0.00045 ± 0.00024 (cos_het1), +0.00044 ± 0.00026 (cos_hall) | - | all 9 negative (−0.00058 .. −0.00005); cos_het1 −0.00010 ± 0.00011 |
| c | signal × cold-candidate terms (6) | −0.00087 .. −0.00002 per quartile | - | −0.00003 .. +0.00008 |
| c | residual: cold indicators × alternatives (18 signal columns on val, 13 on holdout) | −0.00021 ± 0.00028 | +0.00000 / −0.00043 | 0 (0 rounds in both halves) |
| d | group-conditional FT weight (new / seen / guest) | −0.00003 ± 0.00030 | +0.00010 / −0.00017 | no FT on holdout |
| d | same, 3 more user splits | −0.00017, −0.00043, −0.00021 | | |
| d | group-conditional weight on 9 alternative signals | −0.00072 .. +0.00029, all \|z\| < 1.7 | - | - |
| - | planted control, same harness | see §1.3 | | |

Two positive reads in the table exceed 2 SE overall, and neither is a signal:
- cos_het1 linear, +0.00045 ± 0.00020: the post-hoc maximum of 248 columns, under 2 SE in each half, negative on the holdout (§2.2).
- (a) on the second user split, +0.00027 ± 0.00013 (halves +0.00019 ± 0.00012 / +0.00035 ± 0.00023): the largest of five splits, below +0.0005, under 2 SE in each half, and the holdout bounds the same construction at +0.00004 ± 0.00004.

### 2.1 (a) Earlier labelled feeds
- Coverage reproduced: 30.4% of val feeds have an earlier labelled feed, 8.4% contain a re-shown (user, item) pair, 0.18% of rows are re-shown, 16 rows follow an earlier click. Test: public 30.3%, private 28.6% (train bank), +0.1 point with the val bank.
- The bank is thin: 50,574 train feeds for the 42,055 val and holdout users. Of the 4,386 covered val feeds, 2,793 have one earlier feed, 1,325 have two or three, 268 have more.
- Signals built (`s01_bank.py`): item level (shown before, count, clicks, examination-weighted skips, best position, days since), user × category (exposure, clicks, shrunk click rate, click share, skip rate, days since), user level (feeds, clicks, rate, days since), plus within-feed z views. Strictly earlier timestamps only; user 0 excluded.
- Holdout bank: same user's earlier train feeds, as-of. Two variants: any earlier feed (18.2% of feeds covered) and at least 2 days earlier (12.8%), which mimics the train → val distance.
- Re-shown rows click slightly more than other rows of the same feeds (val 1.64% vs 1.23%; holdout 1.19% vs 1.12%), and W8 already ranks them high (z +0.50 val, +0.41 holdout).
- Result: null in every arm. The inner folds find small positive rounds (10-40) and the gain does not carry to the other half.

### 2.2 (b) Feature block
- The linear number for cos_het1 reproduces exactly (+0.00045 ± 0.00020). It is 2.2 SE overall, 1.4-1.7 SE in each half, and it was the maximum of 248 reads.
- The identity of the best column is not stable: the fit halves pick prew_fe_z, cos_u_rk, trd_dsr_s6h, f2, it_hist_all_luo_rk, it_hist_et10_luo, prew_fe_noid, dot_u, clu_het14096. cos_het1 is never picked. 71% of columns have a positive in-sample best-weight mean, as expected when a weight is fitted to noise.
- The holdout is the independent witness for the pre-named columns: cos_het1 −0.00008 ± 0.00008, cos_h50 −0.00007 ± 0.00009, meansim32 −0.00012 ± 0.00008, seq_max_old −0.00031 ± 0.00013, cat_min_hpos −0.00003 ± 0.00003, seq_dcos_deep −0.00015 ± 0.00007.
- As a block the columns hurt: the inner folds select 10-40 rounds on a noise maximum and the other half loses.

### 2.3 (c) Zero-popularity pocket
- Reproduced: 25.3% of val feeds, NDCG 0.3673 vs 0.4133. Holdout: 10.8% of feeds, 0.2474 vs 0.3339.
- The pocket is a label statement ("the click landed on a cold item"). Label-free indicators find it weakly: cold share of the feed AUC 0.657 val / 0.685 holdout; mean log popularity 0.643 / 0.672; M and H 0.50-0.52.
- Cold share does not mark hard feeds. NDCG by cold-share quartile: val 0.424 / 0.369 / 0.382 / 0.431; holdout 0.325 / 0.321 / 0.317 / 0.335.
- W8 does not under-score cold candidates as a group: with a softmax over the base, observed / expected positives on cold candidates = 1.005 ± 0.011 val (quartiles 0.97-1.03), 0.983 ± 0.010 holdout. There is no offset to add.
- One weight per cold-share quartile for 12 alternative signals on val (popularity flag and z, cos_u, cos_hall, maxsim32, seq_max_all, f4, f5, cos_het1, two pre2 columns, item COEC) and 9 on the holdout: nothing at 2 SE on val, nothing positive on the holdout. "More weight on embedding similarity in cold feeds" is not supported: cos_u per quartile +0.00024 ± 0.00031 val, −0.00058 ± 0.00017 holdout.

### 2.4 (d) Users seen in train
- Adjusted gap vs new users (fixed effects: M bucket × positives, H bucket), W8: 1 train feed −0.0033 ± 0.0046, 2-3 feeds −0.0109 ± 0.0062, 4+ feeds +0.0032 ± 0.0134, pooled −0.0051 ± 0.0040. The report's numbers (−0.005, −0.015) are of this size; with H controlled they are 0.7-1.8 SE.
- It is not specific to W8 or to KD: C3 −0.0048 / −0.0131; popularity-only −0.0021 / −0.0030; pre2-only −0.0032 / −0.0057; cos_u-only −0.0025 / +0.0003.
- Holdout analogue (earlier train feeds as-of): 1 earlier feed +0.0044 ± 0.0032, 2+ earlier feeds −0.0121 ± 0.0046. So users with several earlier feeds are harder in both sets (val −0.011, holdout −0.012; together about 3 SE). They are heavy users: 61-74% have a full 256-event history vs 21% of new users.
- Reading (inference, not measured): already-viewed items are removed from every feed, so heavy users are left with the less predictable part of their interest. No test here turned this into a gain: the (a) residual has their feed count and click history as inputs, and the group-conditional blends are null.
- KD does not act through these users: kd05 − base member delta is +0.0030 ± 0.0008 on seen and +0.0024 ± 0.0006 on new users; the in-sample vs cross-fitted teacher contrast leans to seen users on val (+0.0020 ± 0.0012) and the other way on the holdout (−0.0007 ± 0.0007).

## 3. Raw schema re-read (`s00_raw_schema.py`)

- `competition.zip`: 651 files, all extracted; the only non-part files are the readme, the sample submission, three `_SUCCESS` markers and three embedding tables with their meta files.
- All 512 + 64 + 64 part files have one schema per split: 19 columns in train, 18 in val (no `pos`), 17 in test (no `pos`, no `target`). Parquet key-value metadata holds only the Spark schema and version. No field-level metadata.
- Every raw column equals its flat array exactly, list order included: val and test in full, train on 6 random parts (36,863 feeds, 2.9M candidate rows, 6.8M history rows). No nulls inside lists, no NaN, no duplicate feed_id.
- `sample_submission.parquet`: 2,904,806 rows, all scores 0, identical to the copy in `data/raw`.
- The organisers' `calculate_metrics.py` (in `baseline.zip`) matches `src/metrics.py`: binary gain, log2 discount, pessimistic ties, feeds without positives = 1, unweighted mean over feeds. It also reports NDCG@10 and NDCG@30. That the leaderboard uses the full-list value is taken from the level match in VAL_PUBLIC_1113 §1.1, not re-measured here.
- Not re-checked here: whether every flattened column is used by the ranker (feed_id is not, by design).

## 4. Side findings

### 4.1 The val fine-tune under KD (`s07_ft_vs_to.py`)

| family | member TO | member FT − TO | pool FT − TO | 0.5/0.5 blend − pool TO |
|---|---:|---:|---:|---:|
| m02 (hard, 7) | 0.39667 | +0.00046 ± 0.00040 | −0.00008 ± 0.00059 | +0.00006 ± 0.00040 |
| m02_p70k (hard, 7) | 0.39621 | +0.00046 ± 0.00038 | −0.00039 ± 0.00060 | +0.00003 ± 0.00040 |
| gru (hard, 6) | 0.39481 | +0.00075 ± 0.00039 | +0.00058 ± 0.00059 | +0.00042 ± 0.00040 |
| pop1 (hard, 8) | 0.39467 | +0.00118 ± 0.00047 | +0.00023 ± 0.00068 | +0.00110 ± 0.00046 |
| kd05 d128 (KD, 4) | 0.39911 | +0.00001 ± 0.00044 | −0.00036 ± 0.00060 | +0.00039 ± 0.00041 |
| kd08 d128 (KD, 4) | 0.39928 | +0.00013 ± 0.00048 | −0.00008 ± 0.00060 | +0.00004 ± 0.00039 |
| kd05t25 d128 (KD, 4) | 0.39957 | −0.00003 ± 0.00042 | −0.00014 ± 0.00057 | +0.00032 ± 0.00037 |
| kd05w192 (KD, 4) | 0.39918 | −0.00012 ± 0.00044 | −0.00055 ± 0.00062 | −0.00015 ± 0.00042 |
| kd05t25w192 (KD, 4) | 0.39952 | −0.00073 ± 0.00045 | −0.00013 ± 0.00061 | +0.00022 ± 0.00040 |
| mean, hard-label families | | +0.00071 ± 0.00032 | +0.00008 ± 0.00044 | +0.00040 ± 0.00026 |
| mean, KD families | | −0.00015 ± 0.00038 | −0.00025 ± 0.00047 | +0.00016 ± 0.00027 |
| KD minus hard | | **−0.00086 ± 0.00024** | −0.00034 ± 0.00036 | |

- Measured: fine-tuning on one-hot val labels adds nothing to a KD student. For the two wide arms that make up W8 the member delta is −0.0001 and −0.0007.
- Measured: at pool level the FT branch is not reliably better than the TO branch in any family (largest +0.0006 ± 0.0006). The blend adds +0.0004 (hard) and +0.0002 (KD) over the TO pool on average, within 1.5 SE; pop1 is the one family above 2 SE (+0.0011 ± 0.0005).
- Inference: for hard-label members the fine-tune mostly removes variance (EMA, 3 seeds), which a pool or a teacher removes anyway. The val labels carry little that survives pooling.

### 4.2 Train-only vs fine-tuned branch on returning users (`s06`, `s08`, `s09`)
- W8, paired TO − FT: seen users +0.00279 ± 0.00110, new users +0.00000 ± 0.00075 (difference +0.0028 ± 0.0013). For the C3 pool the same contrast is +0.0006 vs +0.0008, so this is not a general property.
- Label-conditioned diagnostic (not a feature): on seen users TO − FT is +0.0059 ± 0.0018 when the positive is in a category the user clicked in an earlier labelled feed (n 1,938), +0.0003 ± 0.0014 otherwise (n 2,448).
- Label-free form: add w·(z_TO − z_FT) on candidates whose category the user clicked before (11.5% of rows), w picked on the fit half: +0.00032 ± 0.00016 in two user splits (halves +0.00049 / +0.00015 and +0.00030 / +0.00034), 5-split mean +0.00028. The weight sits at the grid edge 0.5, i.e. pure TO on those candidates.
- The same information fed to the residual model: +0.00005 ± 0.00020 and +0.00006 ± 0.00012. A plain boost of previously clicked categories is negative (−0.00032 ± 0.00017).
- Status: below +0.0005, about 2 SE, found after several looks. Not a lever. It fits the reading that the hard-label fine-tune partly overwrites what the train-only model learned about a returning user.

## 5. Future ideas (ranked; probabilities are judgement)

P = probability of at least +0.001 public at pool level.

| # | idea | why now | cheapest falsification | cost | P |
|---|---|---|---|---|---:|
| 1 | **Soft-target val fine-tune (KD-FT)** for students: per-feed target = 0.5·y/Σy + 0.5·softmax(z of the students' own TO pool on that val feed). | §4.1: the hard-label FT is worth 0 for students (−0.00086 ± 0.00024 vs hard-label members); §4.2: it overwrites returning-user knowledge. The TO pool never saw val labels, so no cross-fitting is needed. `finetune.py` already has a soft-target path for test pseudo-labels (line 244). | Copy `finetune.py`, add the val soft target, run the 8 W8 members. Pass: FT-branch pool ≥ TO pool (0.40157) and blend ≥ 0.40165 + 0.0005. | about 1.3 GPU-h (10 min per member measured) | 0.07 |
| 2 | **TO-only student file as a public probe** (W8 or W16 train-only pool, no FT branch). | Val says the FT branch adds +0.00008 to W8. If public agrees within noise, new student families need no fine-tune and the blend weight question is closed for the KD era. | One submission; no training. Decision value only. | 0 GPU, 1 submission | 0.03 |
| 3 | **Sub-feed test-time averaging** for students trained with sub-feed augmentation (kwsub arms, running). Score K random sub-feeds of each val/test feed, average each candidate's per-feed z. | A set-attention model's score depends on the candidate set; a model trained on sub-feeds is in-distribution on them, so this is a second, weight-free ensemble axis. Only meaningful if kwsub passes its own screen. | Inference only on val + holdout for the 2 kwsub seeds, K = 8. Pass: member ≥ +0.0005 val and holdout ≥ 0; then the 4-pool rule. | GPU minutes | 0.05 |
| 4 | **Use the monotone residual gate for every new signal** (`hx.resid_cv`, plus one planted control). | The linear test misses interactions (§1.3) and an unconstrained GBDT residual damages the base by itself (§1.2). The gate runs in under a minute on val. | - | CPU seconds | process |
| 5 | **Stop feature hunts on top of the pool.** | 270 linear reads, 16 CCB items and now 12 non-linear arms are all inside noise; the gate cannot see below +0.001 and neither can public. New inputs belong in the ranker with a member / pool test, or nowhere. | Apply to AUTOFE output: only a column block that passes the gate at ≥ +0.0013 val with a positive holdout earns a GPU arm. | - | process |

What this lens says about the gap to #1 (inference from the numbers above and from `E_kd_predeclared.txt`, not a measurement): every wide-student pool sits at val 0.4015-0.4017, the residual gate finds nothing above its +0.001 floor on top of that pool, and the val labels add about +0.0002 at pool level. The public line puts #1 near val 0.406. No post-hoc signal, blend rule or val-label protocol tested here can supply 0.004; idea 1 is worth a few 0.0001 at best. A gain of that size has to come from the training side (the running sub-feed and teacher arms) or from outside the pipeline (a merge).

Ideas considered and dropped:
- Explicit "earlier labelled feed" tokens in the ranker: the residual on those signals is null on val and holdout, and coverage is 30% with one to three feeds per user.
- A returning-user blend rule (§4.2 as a file change): +0.0003 at 2 SE is below the near-copy transfer floor.
- Any cold-feed gate: the pocket is label-defined and the model is calibrated there (§2.3). This agrees with the earlier closures on G3 (COVERAGE_MAP "cold offsets / gates / routers", "feed-conditional routing").
- A remedy for heavy returning users (§2.4): history extension from earlier snapshots is already closed (nested −0.0002..0), and the deficit looks like feed difficulty.

## 6. Limits

- One user split is the primary read; extra splits were run for (a), the best-column pick, the FT-weight rule and §4.2 only. The residual arm is conservative under the null (§1.3).
- Detection limit about +0.001 on val. A real effect of +0.0005 would be missed here and would also not be resolvable on public.
- The holdout has no FT branch, only M ≥ 50 feeds and train-period feature formats. Columns with known train-row format problems (pre*, it_tr_*, tte_*, candidate counts) were left out of the holdout arms; LUO popularity on train rows is inflated relative to val.
- (d) FT-weight tests have no holdout witness.
- §4.2 is a post-hoc contrast among about ten looked at.
- Not re-tested: f5 forward echo and floor persistence (leak-class), embedding drift (linear 0.00000), same-user later-feed neutralisation on test.

## 7. Files

- `nn_x/w12/recheck/forensics_skeptic/hx.py`: loaders, `resid_cv` (monotone residual), `nested_lin`, `nested_bucket`.
- `s00_raw_schema`, `s01_bank`, `s02_resid_a` (val, hold_g0, hold_g2), `s03_power`, `s03b_power_small`, `s04_resid_b` (val, hold), `s04b_best_of`, `s05_pocket_c` (val, hold), `s06_pocket_d`, `s07_ft_vs_to`, `s08_memory`, `s09_branch_trust`: one `.py` and one `.log` each.
- `v1_unconstrained_s02_val_s0.log`: the unconstrained harness, kept as evidence for §1.2.
