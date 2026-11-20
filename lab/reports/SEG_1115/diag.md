# SEG_1115 / diag: user-information segments (cold start, unknown users, sparse users, heavy users)

Assembled 2026-11-15T16:59:18Z by `nn_x/w12/seg/diag/assemble_report.py`. Analyst directory: `nn_x/w12/seg/diag/` (scripts `s0`..`s10`, raw outputs `out_s*.txt`, `DECLARATION.txt` with five stamped blocks, `PROGRESS.md`). CPU only. File analysed: FD16P20 (`subs/cand_FD16P20kd_r4.parquet`, public 0.394892), rebuilt on val exactly as `nn_x/w12/kd/build_r4.py`: 0.40271 [M]. Marks: [M] measured here, [D] arithmetic from [M], [I] judgement. All SE are cluster-robust by user (user_id 0 is a pooled bucket, each of its feeds is its own cluster). H = stored history length (the data caps it at 256).

## 0. Answer to the owner in twelve lines

1. Cold-start and unknown users are a small part of the test and the model is not weak on them. H = 0 is 0.86% of test feeds (1.35% of val), H <= 9 is 2.8%, the anonymous bucket user_id 0 is 4.1% [M]. FINAL reads 0.378 +- 0.018 at H = 0 (per-feed AUC 0.729 against 0.733 overall), 0.443 +- 0.017 at H 1-9 (the best segment of all) and 0.409 +- 0.010 on user_id 0 [M].
2. How the setup handles them: the ranker has no user-ID embedding, so an unseen user is processed exactly like a seen one (69.6% of val feeds come from users with no train feed and they score 0.4065 against 0.3940 for seen users [M]); it has a flag for user_id 0 and flag x log(1 + H), log H as an input, the organisers' user embedding (non-zero for 62% of the H = 0 feeds), and item-side inputs that need no history. Those inputs are MORE informative on cold users than on heavy ones: popularity alone has per-feed AUC 0.62 at H = 0 and 0.54 at H = 256 [M].
3. The one weak cold cell is H = 0 with an all-zero user embedding: 74 val feeds, 0.352 +- 0.025; in test it is 57 feeds = 0.20% [M]. Lifting it by 0.05 would be worth +0.0001 [D].
4. Ceiling of the small segments: no stored model family beats FINAL on H = 0, H 1-9 or user_id 0 when the family is chosen nested by user (nested gains -0.003 .. -0.008; in sample the best is +0.0006 +- 0.0054). Ceiling by "nested gain + 2 SE" x segment share: at most 0.00003 val NDCG per segment [D]. A generous bound (H = 0 lifted to the level of H 100-255) is +0.0004 on val and +0.00024 at the test share [D].
5. The weakest large segment is heavy users at the cap: H = 256, 28.9% of val and 28.8% of test, 0.3906 +- 0.0037 against 0.4077 for the rest (D -0.0171 +- 0.0044) [M].
6. That gap is, as far as val can tell, shared difficulty and not a deficit of our model. Every ranker loses the same share of its skill at the cap: FINAL keeps 0.872 of its skill at H 100-255, the hard-label teacher 0.885, the popularity-light pool 0.871, the self-supervised encoder score 0.872, max cosine to the history 0.859, the f1 column 0.793, popularity alone 0.805 [M]. Rankers that never look at the history get worse too.
7. The model itself predicts the difficulty: the teacher's own probability for its top-1 falls by 0.0098 +- 0.0020 at the cap, the realised hit rate by 0.0143 +- 0.0055 [M].
8. There is no sign that the model is starved of history at the cap. The cosine ranker on the newest 64 events equals the one on all 256 (+0.0008 +- 0.0021); the 17th-32nd most similar events carry no signal beyond the top 16 (AUC -0.0045 +- 0.0012); dropping 20% of the history at inference costs the same at the cap (-0.0020) as below it (-0.0020) [M].
9. About half of the gap goes with the KIND of user, measurable without labels (event-type mix of the history, repeats, activity in train): matching on that leaves -0.0063 +- 0.0039 (37%) [M]. So the lead's upper bound of +0.005 is not available; what val cannot exclude is a model-specific part of up to about +0.003 at pool level (2 SE of the ratio test) [D].
10. Audit: 30 stored arms on history depth, retrieval depth, pooling, recency, capacity and sparse users (plus 5 component-removal information rows) were re-read by H bucket: 0 of 90 cells pass the multiplicity threshold; 6 cells sit at |z| >= 2 where 4.1 are expected by chance [M].
11. Proposals: NO GPU arm follows from A-C. One optional arm for idle GPUs only: unique retrieval slots under distillation (prior 0.05 for a pool gain >= +0.0005) [I].
12. Caveat on method: my first decision rule for "intrinsic or deficit" compared absolute NDCG gaps of rankers whose range differs eightfold and reads "not established" as declared; the ratio rule of line 6 was written after the point ratios were seen and is reported as post hoc (DECLARATION block 1).

## 1. Reproduction and the segment table

FINAL = 0.5 z(`preds/ens_FD16P20kd_ft_val.npy`) + 0.5 z(`preds/ens_FD16P20kd_to_val.npy`): 0.40271 [M] (target 0.40271); rebuilt from its 20 members the scores are bit-identical (max abs diff 0.0). 14,423 feeds, 13,741 users, no feed without a positive. The lead's table is reproduced; the only difference is that H = 0 has 194 feeds, not 195.

| segment | val feeds | val % | test % | FINAL NDCG +- SE | FINAL AUC | FINAL skill | teacher (11 hard-label) | popularity-light pool | popularity only NDCG | popularity only AUC | mean M | n_pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H = 0 | 194 | 1.35 | 0.86 | 0.3784 +- 0.0176 | 0.7286 | 0.1518 | 0.3755 | 0.3728 | 0.3037 | 0.6195 | 100.2 | 1.18 |
| H 1-4 | 150 | 1.04 | 1.02 | 0.4396 +- 0.0246 | 0.7492 | 0.2310 | 0.4306 | 0.4301 | 0.3169 | 0.5972 | 91.2 | 1.17 |
| H 5-9 | 135 | 0.94 | 0.88 | 0.4470 +- 0.0245 | 0.7587 | 0.2407 | 0.4437 | 0.4394 | 0.3288 | 0.6266 | 98.5 | 1.36 |
| H 10-19 | 266 | 1.84 | 1.77 | 0.4149 +- 0.0159 | 0.7601 | 0.1969 | 0.4131 | 0.4201 | 0.3291 | 0.6159 | 103.2 | 1.37 |
| H 20-49 | 785 | 5.44 | 5.52 | 0.4114 +- 0.0093 | 0.7414 | 0.1853 | 0.4079 | 0.4118 | 0.3042 | 0.5839 | 91.9 | 1.33 |
| H 50-99 | 1552 | 10.76 | 10.66 | 0.4091 +- 0.0065 | 0.7556 | 0.1881 | 0.4055 | 0.4109 | 0.3088 | 0.5742 | 97.1 | 1.37 |
| H 100-255 ("NEAR") | 7165 | 49.68 | 50.48 | 0.4060 +- 0.0029 | 0.7339 | 0.1718 | 0.4036 | 0.4048 | 0.3039 | 0.5526 | 96.3 | 1.58 |
| H = 256 (cap) | 4176 | 28.95 | 28.81 | 0.3906 +- 0.0037 | 0.7201 | 0.1498 | 0.3900 | 0.3888 | 0.2997 | 0.5398 | 96.5 | 1.57 |
| all H < 256 ("REST") | 10247 | 71.05 | 71.19 | 0.4077 +- 0.0025 | | | | | | | 96.3 | 1.51 |
| user_id 0 | 658 | 4.56 | 4.12 | 0.4093 +- 0.0101 | 0.7548 | 0.1916 | 0.4060 | 0.4048 | 0.3054 | 0.6055 | 106.1 | 1.56 |
| user seen in train (uid != 0) | 4386 | 30.41 | | 0.3940 +- 0.0036 | | | | | | | 98.2 | 1.64 |
| user never in train (uid != 0) | 9379 | 65.03 | | 0.4063 +- 0.0026 | | | | | | | 94.8 | 1.47 |
| ALL | 14423 | 100 | 100 | 0.4027 | 0.7334 | 0.1694 | 0.4005 | 0.4016 | 0.3041 | 0.5561 | 96.4 | 1.52 |

All [M]. "AUC" = per-feed mean share of negatives ranked below a positive (0.5 for a random order whatever M and n_pos). "skill" = (NDCG - E_random) / (1 - E_random) with the exact expectation of NDCG under a random order given (M, n_pos); E_random is 0.2696 at H = 0, 0.2908 at NEAR and 0.2902 at the cap, so M and n_pos do not make the cap look worse. Segment shares of the public and private parts of test are the same (H = 256: 29.4% / 28.6%; H = 0: 0.85% / 0.86%; user_id 0: 4.10% / 4.13%) [M], so nothing here shifts between public and private.

D_raw: H = 256 minus REST -0.01711 +- 0.00443; minus NEAR -0.01549 +- 0.00470 [M]. Share x gap = +0.00495 [D] = the lead's "+0.005 if lifted".

Two readings that the table gives directly:
- Predictability falls with user activity. Popularity alone goes from AUC 0.62 (H = 0) to 0.54 (cap). FINAL's lift over popularity in AUC is 0.109 / 0.152 / 0.132 / 0.144 / 0.158 / 0.181 / 0.181 / 0.180 over the eight buckets [D]: the history adds a constant +0.18 from H = 50 upward while the history-free base keeps falling.
- Users with little information (H 1-9) are the EASIEST segment (0.44-0.45, AUC 0.75-0.76), not a weak one.

## 2. A. Is the heavy-user gap intrinsic or a model deficit?

### 2.1 Matching on label-free feed composition (A1)

Cross-fitted LightGBM of per-feed NDCG on label-free covariates, fitted on the reference feeds only, 5 folds by user; D = mean residual at H = 256 minus mean residual on the reference [M]:

| matching set | D vs REST | share left | D vs NEAR | share left |
|---|---|---|---|---|
| none | -0.0171 +- 0.0044 | 100% | -0.0155 +- 0.0047 | 100% |
| M only | -0.0161 +- 0.0039 | 94% | -0.0143 +- 0.0042 | 92% |
| feed only (M, day, hour, uid0, cold share, candidate categories) | -0.0133 +- 0.0040 | 78% | -0.0135 +- 0.0042 | 87% |
| + event-type mix of the newest 64 events | -0.0081 +- 0.0039 | 48% | -0.0073 +- 0.0042 | 47% |
| X1 = declared label-free set (+ repeats, category structure) | -0.0089 +- 0.0039 | 52% | -0.0093 +- 0.0042 | 60% |
| the same set re-fitted in s9 (other column order: fit noise of the procedure) | -0.0092 +- 0.0039 | 54% | -0.0097 +- 0.0042 | 63% |
| X1 + user-activity markers (seen in train, number of train feeds) | -0.0063 +- 0.0039 | 37% | -0.0063 +- 0.0042 | 41% |
| X1 + n_pos (NOT label-free) | -0.0137 +- 0.0038 | 80% | -0.0140 +- 0.0041 | 90% |
| exact cells, M decile x cold-share tercile | -0.0161 | 94% | | |

- Feed composition in the usual sense (M, cold share, day) explains almost nothing: the candidate sets of heavy users are as long and as cold as everybody's (M 96.5 against 96.3, cold share 0.375 against 0.369) [M].
- What does explain 40-60% is the kind of user: at the cap the newest 64 events hold fewer plain views (event type 1: 0.52 against 0.63), more of types 2, 3, 5, 6, 8, and fewer distinct items (0.658 against 0.738); 52% of the cap feeds belong to users with train feeds against 28% at NEAR [M]. Feeds of that kind of user are harder below the cap as well (NEAR, user seen in train: 0.3990 against 0.4088 for unseen) [M].
- Adding the number of positives (label-derived, handled separately as required) makes the residual LARGER, because the fit then knows that the cap feeds do not have fewer positives.
- By the declared A1 rule (residual with n_pos <= -2/3 of raw and <= -2 SE): "deficit candidate", i.e. NOT a composition artefact. Whether it is the model is decided by 2.2-2.4. Two expectations I wrote before the block decomposition were wrong and are recorded: feed-only explains 13-22% (I wrote < 10%), and X1 + activity leaves 37-41% (I wrote that < 1/3 would support "different kind of user").

### 2.2 The same table for other rankers (A2)

| ranker | NDCG NEAR | NDCG cap | D(cap - NEAR) | AUC NEAR | AUC cap | skill ratio cap / NEAR (bootstrap SE) | AUC-edge ratio |
|---|---|---|---|---|---|---|---|
| FINAL | 0.4060 | 0.3906 | -0.0155 +- 0.0047 | 0.7339 | 0.7201 | 0.872 (0.034) | 0.941 |
| FD16 base alone | 0.4053 | 0.3899 | -0.0154 +- 0.0047 | 0.7336 | 0.7195 | 0.871 | 0.940 |
| popularity-light pool | 0.4048 | 0.3888 | -0.0160 +- 0.0047 | 0.7328 | 0.7189 | 0.871 | 0.940 |
| teacher, 11 hard-label members | 0.4036 | 0.3900 | -0.0136 +- 0.0047 | 0.7307 | 0.7174 | 0.885 | 0.943 |
| one hard-label member | 0.4001 | 0.3857 | -0.0144 +- 0.0047 | 0.7274 | 0.7109 | 0.873 | 0.928 |
| G3 file (hard labels, FT + TO) | 0.4034 | 0.3893 | -0.0141 +- 0.0047 | 0.7311 | 0.7180 | 0.881 | 0.943 |
| pre-trained encoder score (pre2_fe) | 0.3592 | 0.3501 | -0.0091 +- 0.0042 | 0.6630 | 0.6549 | 0.872 (0.051) | 0.950 |
| max cosine to the history | 0.3450 | 0.3383 | -0.0066 +- 0.0041 | 0.6458 | 0.6360 | 0.859 (0.060) | 0.933 |
| cosine to the user embedding | 0.3418 | 0.3392 | -0.0026 +- 0.0041 | 0.6386 | 0.6336 | 0.965 (0.068) | 0.963 |
| f1 column | 0.3316 | 0.3220 | -0.0096 +- 0.0039 | 0.6226 | 0.6131 | 0.793 (0.076) | 0.923 |
| popularity only | 0.3039 | 0.2997 | -0.0042 +- 0.0037 | 0.5526 | 0.5398 | 0.805 (0.258) | 0.758 |
| exposure only | 0.2740 | 0.2702 | -0.0038 +- 0.0029 | 0.5311 | 0.5167 | (skill negative in both buckets) | 0.538 |

All [M]. Reading by the rule AS DECLARED in block 0: (a) fails (popularity loses 36%, exposure 27% of FINAL's absolute NDCG gap; the rule asked for 50%); (b) fails (FINAL's AUC at the cap is 3.1 SE below NEAR); (c) passes in AUC (lift over popularity -0.0010 +- 0.0064) and fails in NDCG (-0.0113 +- 0.0048). So "intrinsic" is not established by the rule as written, and the model-deficit signature I declared (history-free rankers do not get worse) is not met either: they get worse by the same AUC amount (-0.013 .. -0.014).

Why the declared rule was mis-specified: (a) compared absolute NDCG gaps of rankers whose NDCG range above the random level differs by 8x, and (b) assumed AUC is free of difficulty. The task text names the right criterion (the same gap in the same RATIO). Rule A2' (post hoc, DECLARATION block 1, written before its bootstrap was run): FINAL's skill ratio 0.872 is not below any valid comparator by more than 2 SE: pre2_fe +0.000 +- 0.047, max cosine +0.012 +- 0.059, f1 +0.079 +- 0.072, popularity +0.067 +- 0.256; exposure-only has negative skill in both buckets, so its ratio is not defined, and in AUC edge it keeps 0.54 where FINAL keeps 0.94 [M]. Verdict of A2': shared difficulty.

Power: the tightest comparator (pre2_fe) has a bootstrap SE of 0.047 on the ratio difference = 0.0054 NDCG at the cap = 0.0016 at pool level [D]. A model-specific part of up to about +0.003 at pool level (2 SE) cannot be excluded by this test; nothing points to it.

### 2.3 Where the positives of heavy users sit (A3)

| | H 50-99 | H 100-255 | H = 256 |
|---|---|---|---|
| positive ranked top-1 / top-5 / top-10 by FINAL | 0.098 / 0.323 / 0.463 | 0.085 / 0.275 / 0.407 | 0.075 / 0.255 / 0.398 |
| median rank of a positive | 11 | 14 | 15 |
| cold share of positives / of candidates | 0.285 / 0.384 | 0.293 / 0.368 | 0.327 / 0.375 |
| click propensity of a cold candidate relative to average | 0.74 | 0.80 | 0.87 |
| positives with a history event at cosine >= 0.9 | 0.458 | 0.506 | 0.469 |
| most similar event of a POSITIVE at index < 16 / 16-63 / >= 64 / >= 128 | 0.454 / 0.478 / 0.068 / 0 | 0.258 / 0.371 / 0.371 / 0.104 | 0.200 / 0.281 / 0.519 / 0.281 |
| the same for a NEGATIVE, index >= 64 | 0.097 | 0.438 | 0.594 |
| candidates with more than 16 events at cosine >= 0.8 (positives) | 0.087 (0.177) | 0.150 (0.274) | 0.169 (0.292) |
| distinct items among the 16 retrieval slots | 11.6 | 12.1 | 11.2 |

All [M]; index 0 is the newest event. Code fact that matters for the question as posed: `HistRetrieval` takes the 16 most similar events BY COSINE over the whole stored history, not the newest 16, and the cross-attention with the cosine bias sees all stored events. An old event is therefore reachable; what depth 16 cannot reach is events ranked 17+ by cosine, and what nothing can reach is history older than the 256-event data cap.

- The best-matching event of a heavy user's positive is old more often (52% at index >= 64, 28% at index >= 128), but for negatives it is older still (59%). That is what a longer history does to every candidate, not a property of the relevant event. Positives are, as everywhere, matched by more RECENT events than negatives.
- Declared rule "relevant history is old" (>= 40% of positives at index >= 64 AND the cosine ranker on the newest 64 loses >= 0.005 NDCG against all 256): first half true, second half false: newest-64 minus all-256 = +0.0008 +- 0.0021 NDCG at the cap; the newest 16 alone are +0.0072 +- 0.0029 BETTER in NDCG (and equal in AUC) [M]. NOT met.
- Declared rule "retrieval depth binds" (mean of the top-32 cosines beats the top-16 by >= +0.005 AUC at the cap): -0.0045 +- 0.0012, sign reversed; the 17th-32nd most similar events, residualised on the top-16 mean, have AUC 0.477 +- 0.004 (below 0.5: breadth dilutes) [M]. NOT met.
- How FINAL treats a positive depends on the strength and recency of its best match in the same way at the cap and below: best match at index < 16: top-1 12.8% (cap) / 12.6% (NEAR); at index >= 128: 5.9% / 6.1% [M]. The cap has more positives of the weak kind (28% against 10% at index >= 128), fewer with a strong match (46.9% against 50.6% at cosine >= 0.9) and more cold ones (32.7% against 29.3%). Those are properties of what heavy users click.
- Model-based counterpart from stored predictions (hard-label members scored with history events dropped at random, `nn_x/tricks/HDROP-ALL`): one pass with 20% of events dropped costs -0.0020 +- 0.0007 at the cap and -0.0020 +- 0.0005 at NEAR (contrast -0.00001 +- 0.0009); per removed event the cost is 0.39e-4 at the cap against 0.55e-4 at NEAR [M]. The marginal event is worth less at the cap, not more.
- Does the model know? Softmax of the teacher's mean logit: predicted probability of its top-1 0.1112 (NEAR) -> 0.1014 (cap), -0.0098 +- 0.0020; realised 0.1078 -> 0.0935, -0.0143 +- 0.0055; realised minus predicted -0.0045 +- 0.0052; normalised entropy +0.0102 +- 0.0010 [M]. Two thirds of the drop is predicted by the model; no over-confidence signature.

### 2.4 Verdict on A

Predominantly INTRINSIC (shared difficulty of what heavy users click), with the honest qualifications: (i) the declared A2 rule did not establish it, the ratio rule that does is post hoc; (ii) A1 says the gap is not a feed-composition artefact; 40-63% goes with label-free user kind, the residual -0.0063 +- 0.0039 is neither significant nor shown to be zero; (iii) the tests cannot exclude a model-specific part below about +0.003 at pool level. What is excluded with numbers: that the relevant history is too old, that depth 16 binds, that the model is starved of history at the cap.

## 3. B. The small segments

What the model still has to go on (within-feed AUC of single inputs, sign fixed on all feeds) [M]:

| input | ALL | H = 0 | H 1-9 | user_id 0 | varies inside the feed at H = 0 |
|---|---|---|---|---|---|
| f5 | 0.561 | 0.650 +- 0.018 | 0.643 | 0.629 | 100% |
| f4 | 0.545 | 0.620 +- 0.018 | 0.612 | 0.607 | 100% |
| popularity (it_hist_all_luo) | 0.556 | 0.620 +- 0.020 | 0.611 | 0.606 | 100% |
| exposure (xp_imp_c50) | 0.533 | 0.611 +- 0.020 | 0.596 | 0.583 | 100% |
| feed-context kNN (ctx_nd_max) | 0.573 | 0.617 +- 0.019 | 0.616 | 0.611 | 100% |
| candidate-category share in the feed | 0.579 | 0.595 +- 0.017 | 0.609 | 0.615 | 99% |
| cosine to the user embedding | 0.641 | 0.619 +- 0.015 | 0.667 | 0.679 | 62% |
| f3 / f1 | 0.601 / 0.623 | 0.613 / 0.596 | 0.651 / 0.663 | 0.658 / 0.664 | 62% / 50% |
| pre-trained encoder score (pre2_fe) | 0.663 | 0.586 +- 0.021 | 0.669 | 0.673 | 100% |
| history cosine, co-visitation, ctx_hn / ctx_hs, pre2_hmax32, ALS | 0.55-0.65 | 0.500 (constant) | 0.53-0.68 | 0.55-0.66 | 0% |

At H = 0 every history-derived input is constant; the item-side inputs carry more than they do on average, and FINAL combines them to AUC 0.729.

| segment | feeds | FINAL | popularity-light pool - FINAL | FD16 alone - FINAL | popularity only - FINAL | best stored family IN SAMPLE (not a witness) | NESTED gain of the chosen family | ceiling = max(0, nested + 2 SE) x val share / x test share |
|---|---|---|---|---|---|---|---|---|
| H = 0 | 194 | 0.3784 +- 0.0176 | -0.0055 +- 0.0082 | -0.0007 +- 0.0031 | -0.0747 +- 0.0172 | Q1a popularity-free file +0.0006 +- 0.0054 | -0.0079 +- 0.0042 | 0.00001 / 0.00000 |
| H 1-9 | 285 | 0.4431 +- 0.0173 | -0.0086 +- 0.0052 | -0.0054 +- 0.0021 | -0.1206 +- 0.0148 | S16 file +0.0001 +- 0.0016 | -0.0048 +- 0.0018 | 0 / 0 |
| H <= 9 | 479 | 0.4169 [D] | -0.0073 +- 0.0045 | -0.0035 +- 0.0018 | | none positive | -0.0043 +- 0.0015 | 0 / 0 |
| user_id 0 | 658 | 0.4093 +- 0.0101 | -0.0044 +- 0.0035 | +0.0000 +- 0.0012 | -0.1039 +- 0.0092 | FT branch only +0.0005 +- 0.0013 | -0.0027 +- 0.0015 | 0.00002 / 0.00001 |
| user_id 0 and H >= 10 | 423 | 0.3968 +- 0.0119 | -0.0024 +- 0.0038 | +0.0011 +- 0.0016 | | Q2a pop1 file +0.0019 +- 0.0029 | -0.0031 +- 0.0021 | 0.00003 / 0.00003 |

All [M] except the ceilings [D]. Candidate set (fixed in block 0): FD16 alone, the popularity-light pool, popularity-light weights 0 .. 1, FT branch only, TO branch only, the W8 / W16 / S16 / G3 / Q1a / Q2a files; 15 candidates x 5 segments = 75 looks, 3 cells at |z| >= 2 (all negative) where 3.4 are expected by chance; the largest positive z is +0.80. Nested = 20 repeats of 5 folds by user, family chosen on 4 folds and read on the 5th (FINAL itself is a candidate).

- FINAL is the best stored ranker on every small segment; choosing another family for a segment loses when the choice is nested (it is selection noise on 200-650 feeds).
- The popularity-light admixture is not harmful to sparse users: without it (FD16 alone) H 1-9 reads -0.0054 +- 0.0021 (z -2.5; inside the multiplicity of the 75 looks).
- A popularity-only ranker is 0.07-0.12 NDCG below FINAL on these segments: the model is far from "falling back to popularity".
- Generous ceilings that assume the gap to NEAR is not intrinsic: H = 0: (0.4060 - 0.3784) x share = +0.00037 on val, +0.00024 at the test share [D] (and the AUC there already equals the overall AUC, the NDCG difference is 1.6 SE). H 1-9 and user_id 0 are at or above NEAR: 0.
- Splits inside H = 0 [M]: with a user embedding 120 feeds 0.3949 +- 0.0238 (AUC 0.7525); all-zero embedding 74 feeds 0.3515 +- 0.0253 (AUC 0.690); user_id 0 105 feeds 0.4180; known user id 89 feeds 0.3317 +- 0.0225. In test, 245 feeds have H = 0 and 23.3% of them an all-zero embedding (57 feeds, 0.20%); all-zero embeddings are 0.28% of test feeds against 1.42% of val.

## 4. C. Audit: what was tried for long histories and for sparse users

"H = 256" and "H <= 9" columns are read HERE from the stored train-only val predictions: pool of the arm's members minus the same-seed control pool, cluster SE [M]. "Earlier verdict" is from COVERAGE_MAP / EXPERIMENT_REGISTRY / r6_recap.

| lever | run or only argued | earlier verdict (overall) | stored predictions | read here: ALL | read here: H = 256 | read here: H <= 9 |
|---|---|---|---|---|---|---|
| second history layer, 2 epochs (r7 hl2) | run, 2 seeds | val -0.00059, null (registry l.103) | yes | -0.0004 +- 0.0006 | -0.0001 +- 0.0010 | +0.0011 +- 0.0035 |
| second history layer under distillation (kw_hl2, d192) | run, 2 seeds | closed, <= kd05w192 (COVERAGE l.300) | yes | -0.0003 +- 0.0007 | -0.0001 +- 0.0013 | +0.0060 +- 0.0038 (z 1.6) |
| deep = 3 blocks + 2 history layers (AdamW era) | run, 14-30 pairs | +-0.0002 at 3 / 5 / 10 ep (r6_recap 3.5) | no | | | |
| retrieval top-16 (adopted); read = retrieval REMOVED, 2 ep | run | +0.0019 val / +0.0033 hold when adopted | yes | -0.0030 +- 0.0008 | -0.0038 +- 0.0015 | -0.0020 +- 0.0043 |
| retrieval depth 32, 2 epochs | run, 2 seeds | member -0.0005 / -0.0005, FAIL (l.152) | yes | +0.0002 +- 0.0006 | +0.0016 +- 0.0011 (z 1.5) | -0.0066 +- 0.0037 |
| retrieval depth 32, 10 epochs, hard labels | run, 2 seeds | member -0.0002 / -0.0003, FAIL (registry l.140) | yes | +0.0008 +- 0.0009 | +0.0005 +- 0.0016 | -0.0002 +- 0.0047 |
| retrieval 32 + unique slots + type counts, 2 / 10 ep | run, 2 + 2 seeds | -0.0001 / -0.0003, FAIL | yes | +0.0004 / -0.0001 | +0.0012 +- 0.0012 / +0.0008 +- 0.0016 | -0.0011 / -0.0025 |
| retrieval depth 8 | never run | | | | | |
| retrieval depth 32 under distillation (kwr32) | declared, never run | | | | | |
| second same-category retrieval (retr_cat) | run, 2 seeds | val +0.00046 +- 0.00049, hold 0, FAIL (l.189) | yes | +0.0002 +- 0.0005 | +0.0012 +- 0.0009 | -0.0015 +- 0.0034 |
| unique retrieval slots a4u, 2 epochs | run, 2 seeds (seed 23 readable) | PASS 2 ep: val +0.00117 (z 2.48) | yes | +0.0012 +- 0.0006 | +0.0030 +- 0.0011 (z 2.7) | +0.0011 +- 0.0036 |
| a4u, 10 epochs, 6 seeds vs F7 | run | closed: +0.00036 +- 0.00068, warm -> cold trade | yes | +0.0006 +- 0.0006 | +0.0027 +- 0.0012 (z 2.3) | -0.0007 +- 0.0032 |
| a4u snapshots re-cooled with m02, 6 roots | run | "does not stack with m02" on 2 roots | yes | +0.0002 +- 0.0007 | +0.0012 +- 0.0012 (z 0.95) | |
| DIN product term in retrieval values (a4w) | run, 2 ep | FAIL | yes (seed 23) | +0.0002 +- 0.0007 | +0.0011 +- 0.0013 | +0.0004 +- 0.0044 |
| GRU history encoder, family F10 | run, 6 seeds | ~ transformer (+0.0003..+0.0008 member), in the old pool | yes | -0.0002 +- 0.0007 | +0.0009 +- 0.0013 | -0.0013 +- 0.0039 |
| GRU history encoder under distillation | run, 4 seeds | closed, pooled z < 1 (l.331) | yes | -0.0000 +- 0.0006 | +0.0022 +- 0.0011 (z 2.0) | -0.0020 +- 0.0029 |
| DIN history encoder, family F11 | run, 4 seeds | -0.0018 / -0.0013, family fail | no | | | |
| pooling of history similarity as dense inputs (max, max of newest 8 / 32, mean, soft-max, top-3, per event type) | adopted since v8 | | | | | |
| target attention = cross-attention with cosine bias (adopted); read = sim_bias REMOVED | run, 2 seeds | worth +0.0004 / +0.0005, inconclusive (l.151) | yes | +0.0004 +- 0.0005 | +0.0012 +- 0.0010 | -0.0003 +- 0.0034 |
| typed cross-attention (target actions only in the last block) | run, 2 ep | FAIL (registry l.77) | yes (seed 23) | +0.0007 +- 0.0006 | +0.0009 +- 0.0009 | +0.0047 +- 0.0036 |
| rich cosine bias with a recency prior per head (sim_bias 2, kwsb2) | declared, parked, never run | | | | | |
| multi-interest readout | never run, never argued in a report; nearest runs: 4 attention heads, MoE / FiLM by trajectory type (0 of 8, l.192), per-vertical experts (CPU pre-gate FAIL, l.283), category-intent head (closed, old recipe) | | no | | | |
| positional encoding of history events (index embedding, log2 recency bucket in retrieval) | adopted | | | | | |
| time encoding of history events | impossible: the data has no event timestamps (CURRENT_STATUS section 6) | | | | | |
| recency-decayed category features (streak / EWMA, tau 4 / 16 / 64) | run on CPU (C02) | val -0.00084 +- 0.00041, closed (l.259) | residual form only | | | |
| IDF damping of history items in the similarity aggregation | run on CPU | +0.00024 +- 0.00025, closed (l.228) | | | | |
| lifelong history from the user's earlier snapshots | run on CPU | +93 items median; nested -0.0002..0, closed (l.227) | | | | |
| per-event-type history caps in the cooldown (hcap) | run, 4 roots | member +0.00007, null (l.138) | yes | -0.0005 +- 0.0004 | -0.0001 +- 0.0006 | +0.0011 +- 0.0017 |
| history view restricted to feed-surface events / target actions (HS1 / HS1b) | run, 2 seeds | -0.0031 / -0.0047 (l.215) | yes | -0.0022 / -0.0045 | -0.0023 +- 0.0015 / -0.0050 +- 0.0017 | -0.0065 / -0.0058 |
| history-event dropout at inference (HDROP-ALL, 10% / 20%) | run | TTA -0.00013 / -0.00098 (l.184) | yes | single pass -0.0008 / -0.0017 | -0.0012 +- 0.0006 / -0.0020 +- 0.0007 | |
| history-drop 0.3 function matching under distillation (kwh30) | run, 4 seeds | closed 11-15 | yes | -0.0010 +- 0.0006 | -0.0018 +- 0.0011 | -0.0017 +- 0.0032 |
| candidate subset + history drop 0.15 on cooldown pool batches (aug) | run, 4 roots | closed | yes | -0.0012 +- 0.0006 | -0.0021 +- 0.0012 | +0.0004 +- 0.0036 |
| frozen encoder hidden states as extra keys / values (kv64) | run, 2 seeds | -0.0001, stale veto (l.291) | yes | +0.0003 +- 0.0006 | +0.0020 +- 0.0010 (z 2.0) | +0.0004 +- 0.0032 |
| hashed item-ID table under distillation | run, 2 seeds | -0.005 (l.300) | yes | -0.0051 +- 0.0009 | -0.0036 +- 0.0017 | -0.0021 +- 0.0054 |
| capacity under distillation: 3 blocks, FFN x4, width 224, width 256 | run, 2-4 seeds | all <= kd05w192 (l.300, l.331) | yes | -0.0010 / -0.0009 / +0.0003 / -0.0011 | -0.0010 / -0.0022 / +0.0004 / -0.0001 | -0.0003 / +0.0030 / -0.0025 / -0.0044 |
| width 256 with dropout 0.2 (kw256d2); 6 heads at d192 (kwh6 = `--heads 6`) | declared, never run | | | | | |
| SPARSE USERS: flag for user_id 0 and flag x log(1 + H) | adopted (K5) | | | | | |
| SPARSE USERS: user_id 0 feeds as singleton groups in the fine-tune | adopted (non-inferiority) | | | | | |
| SPARSE USERS: routers / gates / MoE by coldness | run | oracle +0.025 is a label echo, learned 0, closed for good (l.120, l.281) | | | | |
| SPARSE USERS: user aggregates; per-user prior error; user-relative attribute profile | run on CPU | -0.00155; -0.00005..-0.00027; +0.00003 (l.267, l.232, l.233) | | | | |
| SPARSE USERS: cross-user kNN label transfer; trajectory clusters | run on CPU | -0.0001..-0.0003; 0 of 8 (l.195, l.192) | | | | |
| SPARSE USERS: popularity-light admixture 0.2 (adopted) | run, public +0.00052 | val +0.00049 +- 0.00031 | yes | +0.0009 (FD16P20 vs FD16) | | +0.0035 +- 0.0018 |
| gated mixing rules incl. gates on H or M | run, 95 families | nested about 0, public -0.00015 (l.343) | | | | |

Reading rule (block 0): an arm is flagged for a segment only if segment z >= 3.3 (Bonferroni over about 90 looks) AND interaction z >= 2 AND the complement is not hurt. Result [M]: 35 pools = 30 arms + 5 component-removal information rows; 90 non-information cells (30 arms x H = 256, H <= 9, user_id 0): 0 flagged; |z| >= 2 in 6 cells against 4.1 expected by chance; interaction |z| >= 2 in 4 against 4.1 expected. The five largest positive cells are all at H = 256: a4u 2 ep (z +2.66, one seed), cset 2 ep (z +2.37, one seed; at 10 ep with 6 seeds +0.0004 +- 0.0012, gone), a4u 10 ep (z +2.30), kv64 (z +1.99), GRU under distillation (z +1.99). A caution on the 2-epoch rows: with the abl_base control even REMOVING the cosine bias (+0.0012) or the pre2 inputs (+0.0010) reads positive at H = 256, so that control pool is a low draw there by about +0.001.

a4u was followed up because it is the one arm with a label-free mechanism (duplicates crowd the 16 retrieval slots: 11.2 distinct items in the slots at the cap against 12.1 at NEAR; repeat share of the stored history 0.358 against 0.290 [M]) and it was never run under distillation. Three reads declared in block 2 before they were computed:
- F1 replication on the m02 re-cools (6 roots): H = 256 +0.00115 +- 0.00121 (z 0.95), fine-tuned form +0.00146 +- 0.00134 (z 1.09): UNDECIDED (needed z >= 1.64).
- F2 dose-response by the repeat share of the history (H >= 100): most-repeats tercile minus fewest-repeats tercile +0.0011 +- 0.0017 (10-epoch pair) and -0.0017 +- 0.0018 (re-cool pair): NOT SUPPORTED; the gain does not sit where the repeats are.
- F3 cold / warm: on the 10-epoch pair the H = 256 gain is in all-cold and all-warm feeds alike (+0.0041 / +0.0023); on the re-cool pair the signs split (-0.0035 / +0.0035): no stable pattern. Member level, 10-epoch pair at H = 256: 6 of 6 seeds positive, mean +0.0013.

## 5. D. Proposals

Recommendation: NO GPU arm. A says the heavy-user gap is shared difficulty with no depth or reach signature; B puts the ceiling of the small segments at or below 0.00003 val NDCG by the nested read (0.0004 by the generous one); C flags nothing.

One optional arm, for idle GPUs only:

| | kwa4u: unique retrieval slots under distillation |
|---|---|
| mechanism and the number it is tied to | At the cap 30% of the 16 retrieval slots repeat an item already in another slot (11.2 distinct items against 12.1 at NEAR) and 16.9% of candidates (29.2% of positives) have more than 16 events at cosine >= 0.8 [M]: one slot per distinct item with its repeat and per-type counts gives the retrieval more distinct evidence. Stored hard-label evidence at H = 256: +0.0027 +- 0.0012 (10 ep, 6 seeds), +0.0012 +- 0.0012 after the m02 cooldown; the dose-response by repeats does not support the mechanism. |
| flags | `--retr_uniq --retr_tcnt` exist in `nn_x/w5/train.py` (parser lines 186-188) and `model.py` (`HistRetrieval(uniq, tcnt)`, `history_dedup`). Arm line for `lane.sh` (by write-new + mv, as for every script that lanes execute): `E:kwa4u) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --retr_uniq --retr_tcnt" ;;` No model code change. |
| cost | 8 seeds (951-958), 10-epoch distilled students: 8 x 1.2-3.5 GPU-hours = 10-28 GPU-hours; the dedup adds a [B, 256, 256] comparison per batch, about +5-10% step time [I]. A cooldown-only form (switching the flag on at the @77500 snapshots, 16 x 0.25 GPU-hours) would need a change of about 15 lines in train.py (`--init` takes the architecture from the checkpoint) and a FAIL would not close the from-scratch form; not recommended. |
| kill rule (to be copied into E_kd_predeclared.txt before launch) | 8-pool vs kd05w192 at the same seeds (`nn_x/w12/kd/arm_read8.py`), train-only: PASS needs val >= 0, holdout >= 0 and pooled z >= 2 (REPLICATE band 1.6-2). Futility stop after the first 4 seeds if the 4-pool val delta is below -0.0005. Declared secondary: H = 256 bucket of the 8-pool positive at z >= 2 (cluster SE about 0.0011, so about +0.0022). If the pool is null and only the secondary passes: CLOSED (gates on H are closed, MIX_1115). Because a4u is a known warm -> cold trade (coldpos +0.0095, warmpos -0.0034 at 10 epochs), stale-val must be >= 0 as well (section 3). |
| prior for a pool-level gain >= +0.0005 | 0.05 [I]. Reasons: under the m02 cooldown the overall effect is +0.0002 +- 0.0007 and the H = 256 effect halves; every architecture / capacity lever tried under distillation except width 192 was null or negative (width 256, width 224, 3 blocks, FFN x4, 2 history layers, GRU encoder, ID table); the cell that motivates it is z 2.3 among 90 looks. Expected value about +0.00003. |

On the three arms declared but never run under distillation: kwr32 (retrieval depth 32) is not supported for heavy users (depth does not bind: top-32 below top-16 by 0.0045 AUC at the cap; the 10-epoch hard-label pair reads +0.0005 +- 0.0016 there), prior 0.03 [I]; kwh6 is `--heads 6` at d192, nothing in A-C speaks to it; kw256d2 (width 256 with dropout 0.2): the stored width-256 students read -0.0001 +- 0.0013 at the cap and -0.0011 overall, no segment signal.

Not proposed, with the reason: a second history layer (two runs, -0.0001 at the cap); a multi-interest readout (never run, but the three nearest forms are null and A3 shows that breadth beyond the 16 most similar events dilutes); anything for H = 0 (0.86% of test, ceiling <= +0.0004 even by the generous bound); longer history (the data caps it at 256; the earlier-snapshot extension is closed and the model's marginal use of events is lower at the cap, not higher).

## 6. Method notes, deviations, files

- Declaration: `DECLARATION.txt` block 0 (16:40:12Z) was written before any number of this study; blocks 1-4 are stamped amendments written between runs. Deviations from "criterion before number": (1) rule A2' (ratio of skill) was written after the point ratios of s2 were seen; its bootstrap SE was computed afterwards; it is labelled post hoc everywhere. (2) Descriptive rows that block 0 did not list were printed and are used only as description: the co-visitation-only ranker; splits inside H = 256 by seen-in-train / repeat share / category count; the slot-saturation table and the residualised AUC of events 17-32; the per-bucket table of named arms; s10 (distinct items in the retrieval slots). (3) Two expectations of block 4 were wrong (section 2.1).
- Pairing limits in C: the r6 2-epoch arms (typed, a4u, a4w, cset) have a stored control for seed 23 only; GRU family v23 against F7 v17 is not seed-paired; the a4u m02 re-cools are cross-root against the F7 re-cools, as the registry notes. DIN (F11) and the AdamW-era "deep" runs have no stored val predictions and could not be read by bucket.
- Leak rules: no later snapshots, no labels in features. Val labels are used descriptively, or inside 5-fold GroupKFold-by-user procedures where something is fitted (A1) or chosen (B). In-sample "best family" numbers are printed next to the nested ones and labelled as not a witness.
- Hard rules: CPU only (CUDA_VISIBLE_DEVICES=""), one process at a time with 12 threads, memory far below 30 GB, no process signalled, nothing written outside `nn_x/w12/seg/diag/` and `reports/SEG_1115/`, no network, no install, no submission file.

## Appendix: raw outputs

### s0 reproduction
```
FINAL val 0.40271 (target 0.40271) | feeds 14423 | feeds without positive 0 | users 13741 | clusters 14398
| segment                        | feeds | val % | test % | NDCG   | SE(cluster) | mean M | mean n_pos |
|--------------------------------|-------|-------|--------|--------|-------------|--------|------------|
| H=0                            | 194   | 1.35  | 0.86   | 0.3784 | 0.0176      | 100.2  | 1.18       |
| 1-4                            | 150   | 1.04  | 1.02   | 0.4396 | 0.0246      | 91.2   | 1.17       |
| 5-9                            | 135   | 0.94  | 0.88   | 0.4470 | 0.0245      | 98.5   | 1.36       |
| 10-19                          | 266   | 1.84  | 1.77   | 0.4149 | 0.0159      | 103.2  | 1.37       |
| 20-49                          | 785   | 5.44  | 5.52   | 0.4114 | 0.0093      | 91.9   | 1.33       |
| 50-99                          | 1552  | 10.76 | 10.66  | 0.4091 | 0.0065      | 97.1   | 1.37       |
| 100-255                        | 7165  | 49.68 | 50.48  | 0.4060 | 0.0029      | 96.3   | 1.58       |
| H=256                          | 4176  | 28.95 | 28.81  | 0.3906 | 0.0037      | 96.5   | 1.57       |
| REST                           | 10247 | 71.05 |        | 0.4077 | 0.0025      | 96.3   | 1.51       |
| uid0                           | 658   | 4.56  |        | 0.4093 | 0.0101      | 106.1  | 1.56       |
| user seen in train (uid != 0)  | 4386  | 30.41 |        | 0.3940 | 0.0036      | 98.2   | 1.64       |
| user never in train (uid != 0) | 9379  | 65.03 |        | 0.4063 | 0.0026      | 94.8   | 1.47       |
| never in train incl. uid0      | 10037 | 69.59 |        | 0.4065 | 0.0025      | 95.6   | 1.47       |
D_raw(REST) = -0.01711 +- 0.00443  -> x share 0.2895 = +0.00495 (lead: +0.005)
D_raw(NEAR) = -0.01549 +- 0.00470
test H=256 share public / private: 0.2939 / 0.2857 | H=0: 0.0085 / 0.0086 | H<=9: 0.0278 / 0.0275 | uid0: 0.0410 / 0.0413
val: H<=9 0.0332 uid0 0.0456
```

### s2 rankers by H bucket
```
rebuild from members vs stored FINAL: max abs diff 0.0 ndcg 0.40271355928380465

### NDCG by H bucket
| ranker                                  | ALL    | H=0    | 1-4    | 5-9    | 10-19  | 20-49  | 50-99  | 100-255 | H=256  | D(256-NEAR)     | D(256-REST)     | ratio 256/NEAR |
|-----------------------------------------|--------|--------|--------|--------|--------|--------|--------|---------|--------|-----------------|-----------------|----------------|
| FINAL                                   | 0.4027 | 0.3784 | 0.4396 | 0.4470 | 0.4149 | 0.4114 | 0.4091 | 0.4060  | 0.3906 | -0.0155+-0.0047 | -0.0171+-0.0044 | 0.962          |
| FD16 (16 students, no pop-light)        | 0.4019 | 0.3777 | 0.4363 | 0.4394 | 0.4131 | 0.4097 | 0.4088 | 0.4053  | 0.3899 | -0.0154+-0.0047 | -0.0168+-0.0044 | 0.962          |
| FD16 TO branch                          | 0.4016 | 0.3808 | 0.4303 | 0.4385 | 0.4127 | 0.4081 | 0.4069 | 0.4048  | 0.3909 | -0.0139+-0.0047 | -0.0150+-0.0044 | 0.966          |
| FD16 FT branch                          | 0.4012 | 0.3733 | 0.4386 | 0.4471 | 0.4123 | 0.4105 | 0.4093 | 0.4046  | 0.3882 | -0.0164+-0.0047 | -0.0182+-0.0044 | 0.959          |
| POPLIGHT pool (4)                       | 0.4016 | 0.3728 | 0.4301 | 0.4394 | 0.4201 | 0.4118 | 0.4109 | 0.4048  | 0.3888 | -0.0160+-0.0047 | -0.0180+-0.0044 | 0.961          |
| TEACHER 11 hard-label (TO)              | 0.4005 | 0.3755 | 0.4306 | 0.4437 | 0.4131 | 0.4079 | 0.4055 | 0.4036  | 0.3900 | -0.0136+-0.0047 | -0.0148+-0.0044 | 0.966          |
| hard-label single w5_e10_base_s951 (TO) | 0.3963 | 0.3644 | 0.4278 | 0.4400 | 0.4029 | 0.3987 | 0.4024 | 0.4001  | 0.3857 | -0.0144+-0.0047 | -0.0150+-0.0044 | 0.964          |
| student single fd10_kd05w192_s951 (TO)  | 0.3995 | 0.3718 | 0.4276 | 0.4432 | 0.4072 | 0.4080 | 0.4079 | 0.4033  | 0.3866 | -0.0167+-0.0047 | -0.0181+-0.0044 | 0.959          |
| G3 file (hard labels, FT+TO)            | 0.4003 | 0.3775 | 0.4316 | 0.4486 | 0.4150 | 0.4069 | 0.4058 | 0.4034  | 0.3893 | -0.0141+-0.0047 | -0.0155+-0.0044 | 0.965          |
| popularity only (it_hist_all_luo)       | 0.3041 | 0.3037 | 0.3169 | 0.3288 | 0.3291 | 0.3042 | 0.3088 | 0.3039  | 0.2997 | -0.0042+-0.0037 | -0.0061+-0.0035 | 0.986          |
| exposure only (xp_imp_c50)              | 0.2736 | 0.2823 | 0.2976 | 0.2955 | 0.2868 | 0.2725 | 0.2731 | 0.2740  | 0.2702 | -0.0038+-0.0029 | -0.0047+-0.0027 | 0.986          |
| co-visitation only (seq_cv_invd20)      | 0.2811 | 0.1856 | 0.2157 | 0.2644 | 0.2463 | 0.2529 | 0.2827 | 0.2901  | 0.2799 | -0.0102+-0.0039 | -0.0017+-0.0037 | 0.965          |
| pre-trained encoder score (pre2_fe)     | 0.3578 | 0.3180 | 0.3631 | 0.3870 | 0.3763 | 0.3631 | 0.3683 | 0.3592  | 0.3501 | -0.0091+-0.0042 | -0.0108+-0.0040 | 0.975          |
| max cosine to history (all 256)         | 0.3423 | 0.1856 | 0.3597 | 0.3750 | 0.3414 | 0.3494 | 0.3520 | 0.3450  | 0.3383 | -0.0066+-0.0041 | -0.0055+-0.0038 | 0.981          |
| cosine to user embedding                | 0.3417 | 0.2799 | 0.3476 | 0.3493 | 0.3477 | 0.3569 | 0.3456 | 0.3418  | 0.3392 | -0.0026+-0.0041 | -0.0035+-0.0039 | 0.992          |
| f1 only (+ sign, chosen by overall AUC) | 0.3279 | 0.2579 | 0.3337 | 0.3346 | 0.3293 | 0.3323 | 0.3314 | 0.3316  | 0.3220 | -0.0096+-0.0039 | -0.0082+-0.0037 | 0.971          |
| f3 only (+ sign, chosen by overall AUC) | 0.2518 | 0.2193 | 0.2496 | 0.2376 | 0.2437 | 0.2457 | 0.2464 | 0.2553  | 0.2518 | -0.0035+-0.0023 | -0.0001+-0.0021 | 0.986          |

### per-feed AUC by H bucket
| ranker                                  | ALL    | H=0    | 1-4    | 5-9    | 10-19  | 20-49  | 50-99  | 100-255 | H=256  | D(256-NEAR)     | D(256-REST)     | ratio 256/NEAR |
|-----------------------------------------|--------|--------|--------|--------|--------|--------|--------|---------|--------|-----------------|-----------------|----------------|
| FINAL                                   | 0.7334 | 0.7286 | 0.7492 | 0.7587 | 0.7601 | 0.7414 | 0.7556 | 0.7339  | 0.7201 | -0.0138+-0.0044 | -0.0188+-0.0041 | 0.981          |
| FD16 (16 students, no pop-light)        | 0.7331 | 0.7278 | 0.7479 | 0.7575 | 0.7600 | 0.7411 | 0.7561 | 0.7336  | 0.7195 | -0.0140+-0.0044 | -0.0191+-0.0041 | 0.981          |
| FD16 TO branch                          | 0.7325 | 0.7300 | 0.7450 | 0.7579 | 0.7592 | 0.7404 | 0.7541 | 0.7331  | 0.7192 | -0.0138+-0.0044 | -0.0187+-0.0042 | 0.981          |
| FD16 FT branch                          | 0.7329 | 0.7256 | 0.7552 | 0.7572 | 0.7600 | 0.7412 | 0.7558 | 0.7335  | 0.7187 | -0.0148+-0.0044 | -0.0200+-0.0041 | 0.980          |
| POPLIGHT pool (4)                       | 0.7324 | 0.7277 | 0.7482 | 0.7564 | 0.7623 | 0.7412 | 0.7542 | 0.7328  | 0.7189 | -0.0139+-0.0044 | -0.0190+-0.0042 | 0.981          |
| TEACHER 11 hard-label (TO)              | 0.7306 | 0.7250 | 0.7448 | 0.7569 | 0.7613 | 0.7420 | 0.7517 | 0.7307  | 0.7174 | -0.0132+-0.0045 | -0.0185+-0.0042 | 0.982          |
| hard-label single w5_e10_base_s951 (TO) | 0.7252 | 0.7083 | 0.7399 | 0.7517 | 0.7511 | 0.7331 | 0.7429 | 0.7274  | 0.7109 | -0.0165+-0.0045 | -0.0200+-0.0042 | 0.977          |
| student single fd10_kd05w192_s951 (TO)  | 0.7293 | 0.7242 | 0.7422 | 0.7575 | 0.7565 | 0.7375 | 0.7509 | 0.7302  | 0.7154 | -0.0148+-0.0044 | -0.0196+-0.0042 | 0.980          |
| G3 file (hard labels, FT+TO)            | 0.7311 | 0.7236 | 0.7477 | 0.7576 | 0.7597 | 0.7417 | 0.7527 | 0.7311  | 0.7180 | -0.0131+-0.0044 | -0.0184+-0.0042 | 0.982          |
| popularity only (it_hist_all_luo)       | 0.5561 | 0.6195 | 0.5972 | 0.6266 | 0.6159 | 0.5839 | 0.5742 | 0.5526  | 0.5398 | -0.0127+-0.0051 | -0.0230+-0.0049 | 0.977          |
| exposure only (xp_imp_c50)              | 0.5335 | 0.6114 | 0.5879 | 0.6045 | 0.5816 | 0.5515 | 0.5512 | 0.5311  | 0.5167 | -0.0144+-0.0052 | -0.0236+-0.0049 | 0.973          |
| co-visitation only (seq_cv_invd20)      | 0.5515 | 0.5000 | 0.5152 | 0.5438 | 0.5336 | 0.5350 | 0.5582 | 0.5571  | 0.5476 | -0.0095+-0.0037 | -0.0055+-0.0034 | 0.983          |
| pre-trained encoder score (pre2_fe)     | 0.6632 | 0.5857 | 0.6721 | 0.6665 | 0.7009 | 0.6816 | 0.6797 | 0.6630  | 0.6549 | -0.0081+-0.0049 | -0.0118+-0.0047 | 0.988          |
| max cosine to history (all 256)         | 0.6447 | 0.5000 | 0.6668 | 0.6791 | 0.6715 | 0.6606 | 0.6640 | 0.6458  | 0.6360 | -0.0098+-0.0048 | -0.0123+-0.0045 | 0.985          |
| cosine to user embedding                | 0.6413 | 0.6191 | 0.6764 | 0.6575 | 0.6723 | 0.6671 | 0.6537 | 0.6386  | 0.6336 | -0.0051+-0.0050 | -0.0108+-0.0047 | 0.992          |
| f1 only (+ sign, chosen by overall AUC) | 0.6226 | 0.5962 | 0.6837 | 0.6408 | 0.6510 | 0.6421 | 0.6289 | 0.6226  | 0.6131 | -0.0095+-0.0051 | -0.0133+-0.0048 | 0.985          |
| f3 only (+ sign, chosen by overall AUC) | 0.6007 | 0.6133 | 0.6662 | 0.6336 | 0.6400 | 0.6202 | 0.6011 | 0.5999  | 0.5917 | -0.0082+-0.0047 | -0.0127+-0.0044 | 0.986          |

### skill = (NDCG - E_random)/(1 - E_random) by H bucket
| ranker                                  | ALL     | H=0     | 1-4     | 5-9     | 10-19   | 20-49   | 50-99   | 100-255 | H=256   | D(256-NEAR)     | D(256-REST)     | ratio 256/NEAR |
|-----------------------------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-----------------|-----------------|----------------|
| FINAL                                   | 0.1694  | 0.1518  | 0.2310  | 0.2407  | 0.1969  | 0.1853  | 0.1881  | 0.1718  | 0.1498  | -0.0221+-0.0062 | -0.0276+-0.0059 | 0.872          |
| FD16 (16 students, no pop-light)        | 0.1683  | 0.1511  | 0.2265  | 0.2304  | 0.1947  | 0.1828  | 0.1879  | 0.1708  | 0.1489  | -0.0220+-0.0062 | -0.0273+-0.0059 | 0.871          |
| FD16 TO branch                          | 0.1680  | 0.1575  | 0.2195  | 0.2267  | 0.1936  | 0.1796  | 0.1852  | 0.1703  | 0.1505  | -0.0198+-0.0062 | -0.0246+-0.0059 | 0.884          |
| FD16 FT branch                          | 0.1673  | 0.1454  | 0.2297  | 0.2413  | 0.1927  | 0.1843  | 0.1886  | 0.1701  | 0.1463  | -0.0238+-0.0062 | -0.0296+-0.0058 | 0.860          |
| POPLIGHT pool (4)                       | 0.1678  | 0.1448  | 0.2170  | 0.2273  | 0.2043  | 0.1853  | 0.1911  | 0.1696  | 0.1477  | -0.0219+-0.0062 | -0.0282+-0.0058 | 0.871          |
| TEACHER 11 hard-label (TO)              | 0.1665  | 0.1507  | 0.2199  | 0.2396  | 0.1959  | 0.1797  | 0.1835  | 0.1683  | 0.1490  | -0.0194+-0.0062 | -0.0246+-0.0059 | 0.885          |
| hard-label single w5_e10_base_s951 (TO) | 0.1605  | 0.1330  | 0.2158  | 0.2325  | 0.1785  | 0.1660  | 0.1796  | 0.1636  | 0.1428  | -0.0208+-0.0062 | -0.0249+-0.0058 | 0.873          |
| student single fd10_kd05w192_s951 (TO)  | 0.1650  | 0.1436  | 0.2150  | 0.2331  | 0.1879  | 0.1801  | 0.1861  | 0.1685  | 0.1437  | -0.0247+-0.0062 | -0.0299+-0.0058 | 0.853          |
| G3 file (hard labels, FT+TO)            | 0.1660  | 0.1528  | 0.2201  | 0.2471  | 0.1976  | 0.1781  | 0.1837  | 0.1678  | 0.1479  | -0.0199+-0.0062 | -0.0254+-0.0058 | 0.881          |
| popularity only (it_hist_all_luo)       | 0.0221  | 0.0497  | 0.0527  | 0.0684  | 0.0724  | 0.0254  | 0.0358  | 0.0187  | 0.0150  | -0.0036+-0.0048 | -0.0099+-0.0046 | 0.805          |
| exposure only (xp_imp_c50)              | -0.0227 | 0.0242  | 0.0260  | 0.0251  | 0.0111  | -0.0212 | -0.0169 | -0.0247 | -0.0293 | -0.0046+-0.0036 | -0.0093+-0.0034 | 1.185          |
| co-visitation only (seq_cv_invd20)      | -0.0182 | -0.1288 | -0.1076 | -0.0330 | -0.0584 | -0.0560 | -0.0074 | -0.0076 | -0.0219 | -0.0144+-0.0056 | -0.0053+-0.0052 | 2.892          |
| pre-trained encoder score (pre2_fe)     | 0.1043  | 0.0714  | 0.1202  | 0.1560  | 0.1398  | 0.1156  | 0.1278  | 0.1041  | 0.0907  | -0.0134+-0.0055 | -0.0191+-0.0052 | 0.872          |
| max cosine to history (all 256)         | 0.0807  | -0.1288 | 0.1140  | 0.1407  | 0.0868  | 0.0965  | 0.1049  | 0.0828  | 0.0712  | -0.0117+-0.0053 | -0.0134+-0.0050 | 0.859          |
| cosine to user embedding                | 0.0798  | 0.0150  | 0.0973  | 0.1036  | 0.0996  | 0.1070  | 0.0920  | 0.0774  | 0.0747  | -0.0027+-0.0053 | -0.0072+-0.0050 | 0.965          |
| f1 only (+ sign, chosen by overall AUC) | 0.0592  | -0.0201 | 0.0802  | 0.0764  | 0.0719  | 0.0684  | 0.0708  | 0.0623  | 0.0494  | -0.0129+-0.0051 | -0.0138+-0.0048 | 0.793          |
| f3 only (+ sign, chosen by overall AUC) | -0.0563 | -0.0749 | -0.0496 | -0.0682 | -0.0560 | -0.0627 | -0.0564 | -0.0539 | -0.0581 | -0.0042+-0.0024 | -0.0025+-0.0023 | 1.078          |
| ranker                           | ALL    | H=0    | 1-4    | 5-9    | 10-19  | 20-49  | 50-99  | 100-255 | H=256  | D(256-NEAR)     | D(256-REST)     |  |
|----------------------------------|--------|--------|--------|--------|--------|--------|--------|---------|--------|-----------------|-----------------|--|
| random order (exact expectation) | 0.2890 | 0.2696 | 0.2839 | 0.2809 | 0.2795 | 0.2860 | 0.2838 | 0.2908  | 0.2902 | -0.0005+-0.0021 | +0.0018+-0.0020 |  |

### lift of each model over history-free rankers (paired per feed): value at H=256, at NEAR, contrast (256 - NEAR), contrast (256 - REST)
| model                               | minus                             | metric | lift@256 | lift@NEAR | contrast 256-NEAR | contrast 256-REST |
|-------------------------------------|-----------------------------------|--------|----------|-----------|-------------------|-------------------|
| FINAL                               | popularity only (it_hist_all_luo) | NDCG   | 0.0908   | 0.1021    | -0.0113+-0.0048   | -0.0110+-0.0045   |
| FINAL                               | popularity only (it_hist_all_luo) | AUC    | 0.1803   | 0.1813    | -0.0010+-0.0064   | +0.0042+-0.0060   |
| TEACHER 11 hard-label (TO)          | popularity only (it_hist_all_luo) | NDCG   | 0.0903   | 0.0997    | -0.0094+-0.0048   | -0.0087+-0.0045   |
| TEACHER 11 hard-label (TO)          | popularity only (it_hist_all_luo) | AUC    | 0.1776   | 0.1781    | -0.0005+-0.0064   | +0.0044+-0.0060   |
| POPLIGHT pool (4)                   | popularity only (it_hist_all_luo) | NDCG   | 0.0891   | 0.1009    | -0.0118+-0.0048   | -0.0119+-0.0045   |
| POPLIGHT pool (4)                   | popularity only (it_hist_all_luo) | AUC    | 0.1791   | 0.1802    | -0.0011+-0.0065   | +0.0040+-0.0061   |
| max cosine to history (all 256)     | popularity only (it_hist_all_luo) | NDCG   | 0.0386   | 0.0411    | -0.0025+-0.0046   | +0.0006+-0.0043   |
| max cosine to history (all 256)     | popularity only (it_hist_all_luo) | AUC    | 0.0961   | 0.0932    | +0.0030+-0.0071   | +0.0106+-0.0067   |
| pre-trained encoder score (pre2_fe) | popularity only (it_hist_all_luo) | NDCG   | 0.0504   | 0.0553    | -0.0049+-0.0043   | -0.0047+-0.0040   |
| pre-trained encoder score (pre2_fe) | popularity only (it_hist_all_luo) | AUC    | 0.1151   | 0.1104    | +0.0046+-0.0069   | +0.0112+-0.0065   |
| FINAL                               | exposure only (xp_imp_c50)        | NDCG   | 0.1203   | 0.1320    | -0.0117+-0.0044   | -0.0124+-0.0042   |
| FINAL                               | exposure only (xp_imp_c50)        | AUC    | 0.2034   | 0.2028    | +0.0006+-0.0066   | +0.0048+-0.0062   |
| TEACHER 11 hard-label (TO)          | exposure only (xp_imp_c50)        | NDCG   | 0.1198   | 0.1295    | -0.0098+-0.0044   | -0.0102+-0.0042   |
| TEACHER 11 hard-label (TO)          | exposure only (xp_imp_c50)        | AUC    | 0.2007   | 0.1996    | +0.0011+-0.0066   | +0.0051+-0.0062   |
| POPLIGHT pool (4)                   | exposure only (xp_imp_c50)        | NDCG   | 0.1186   | 0.1307    | -0.0121+-0.0044   | -0.0134+-0.0041   |
| POPLIGHT pool (4)                   | exposure only (xp_imp_c50)        | AUC    | 0.2022   | 0.2017    | +0.0005+-0.0067   | +0.0046+-0.0063   |
| max cosine to history (all 256)     | exposure only (xp_imp_c50)        | NDCG   | 0.0681   | 0.0709    | -0.0028+-0.0040   | -0.0009+-0.0038   |
| max cosine to history (all 256)     | exposure only (xp_imp_c50)        | AUC    | 0.1193   | 0.1147    | +0.0046+-0.0072   | +0.0113+-0.0068   |
| pre-trained encoder score (pre2_fe) | exposure only (xp_imp_c50)        | NDCG   | 0.0799   | 0.0851    | -0.0053+-0.0038   | -0.0062+-0.0036   |
| pre-trained encoder score (pre2_fe) | exposure only (xp_imp_c50)        | AUC    | 0.1382   | 0.1319    | +0.0063+-0.0069   | +0.0118+-0.0065   |
| FINAL                               | cosine to user embedding          | NDCG   | 0.0514   | 0.0642    | -0.0129+-0.0043   | -0.0136+-0.0041   |
| FINAL                               | cosine to user embedding          | AUC    | 0.0865   | 0.0952    | -0.0087+-0.0049   | -0.0079+-0.0046   |
| TEACHER 11 hard-label (TO)          | cosine to user embedding          | NDCG   | 0.0508   | 0.0618    | -0.0109+-0.0043   | -0.0113+-0.0041   |
| TEACHER 11 hard-label (TO)          | cosine to user embedding          | AUC    | 0.0838   | 0.0920    | -0.0082+-0.0049   | -0.0077+-0.0046   |
| POPLIGHT pool (4)                   | cosine to user embedding          | NDCG   | 0.0496   | 0.0629    | -0.0133+-0.0043   | -0.0145+-0.0041   |
| POPLIGHT pool (4)                   | cosine to user embedding          | AUC    | 0.0853   | 0.0941    | -0.0088+-0.0049   | -0.0081+-0.0046   |
| max cosine to history (all 256)     | cosine to user embedding          | NDCG   | -0.0008  | 0.0032    | -0.0040+-0.0041   | -0.0020+-0.0039   |
| max cosine to history (all 256)     | cosine to user embedding          | AUC    | 0.0024   | 0.0071    | -0.0047+-0.0053   | -0.0015+-0.0050   |
| pre-trained encoder score (pre2_fe) | cosine to user embedding          | NDCG   | 0.0109   | 0.0174    | -0.0064+-0.0037   | -0.0073+-0.0035   |
| pre-trained encoder score (pre2_fe) | cosine to user embedding          | AUC    | 0.0213   | 0.0243    | -0.0030+-0.0041   | -0.0009+-0.0039   |
```

### s2b ratio bootstrap
```

### R = skill(H=256) / skill(NEAR); cluster bootstrap 2000 draws
| ranker                                  | R     | boot SE | 95% CI         | R_FINAL - R_ranker     |
|-----------------------------------------|-------|---------|----------------|------------------------|
| FINAL                                   | 0.872 | 0.034   | [0.804, 0.940] |                        |
| FD16 (16 students, no pop-light)        | 0.871 | 0.035   | [0.804, 0.941] | +0.000+-0.006 (z +0.0) |
| FD16 TO branch                          | 0.884 | 0.035   | [0.818, 0.953] | -0.012+-0.008 (z -1.5) |
| FD16 FT branch                          | 0.860 | 0.034   | [0.794, 0.930] | +0.011+-0.009 (z +1.3) |
| POPLIGHT pool (4)                       | 0.871 | 0.035   | [0.805, 0.940] | +0.001+-0.014 (z +0.0) |
| TEACHER 11 hard-label (TO)              | 0.885 | 0.035   | [0.819, 0.957] | -0.013+-0.012 (z -1.1) |
| hard-label single w5_e10_base_s951 (TO) | 0.873 | 0.036   | [0.803, 0.947] | -0.001+-0.018 (z -0.1) |
| student single fd10_kd05w192_s951 (TO)  | 0.853 | 0.034   | [0.788, 0.921] | +0.018+-0.015 (z +1.2) |
| G3 file (hard labels, FT+TO)            | 0.881 | 0.035   | [0.813, 0.952] | -0.010+-0.011 (z -0.9) |
| popularity only (it_hist_all_luo)       | 0.805 | 0.258   | [0.417, 1.428] | +0.067+-0.256 (z +0.3) |
| exposure only (xp_imp_c50)              | 1.185 | 0.163   | [0.907, 1.534] | -0.313+-0.168 (z -1.9) |
| pre-trained encoder score (pre2_fe)     | 0.872 | 0.051   | [0.778, 0.974] | -0.000+-0.047 (z -0.0) |
| max cosine to history (all 256)         | 0.859 | 0.060   | [0.745, 0.984] | +0.012+-0.059 (z +0.2) |
| cosine to user embedding                | 0.965 | 0.068   | [0.839, 1.104] | -0.093+-0.063 (z -1.5) |
| f1 only (+ sign, chosen by overall AUC) | 0.793 | 0.076   | [0.653, 0.949] | +0.079+-0.072 (z +1.1) |

### R = NDCG - E_random(H=256) / NDCG - E_random(NEAR); cluster bootstrap 2000 draws
| ranker                                  | R     | boot SE | 95% CI         | R_FINAL - R_ranker     |
|-----------------------------------------|-------|---------|----------------|------------------------|
| FINAL                                   | 0.870 | 0.033   | [0.806, 0.934] |                        |
| FD16 (16 students, no pop-light)        | 0.870 | 0.033   | [0.803, 0.935] | -0.000+-0.006 (z -0.0) |
| FD16 TO branch                          | 0.883 | 0.034   | [0.815, 0.949] | -0.013+-0.008 (z -1.5) |
| FD16 FT branch                          | 0.861 | 0.032   | [0.798, 0.923] | +0.010+-0.008 (z +1.2) |
| POPLIGHT pool (4)                       | 0.865 | 0.033   | [0.800, 0.926] | +0.006+-0.013 (z +0.4) |
| TEACHER 11 hard-label (TO)              | 0.885 | 0.034   | [0.816, 0.951] | -0.014+-0.012 (z -1.2) |
| hard-label single w5_e10_base_s951 (TO) | 0.873 | 0.034   | [0.809, 0.939] | -0.003+-0.018 (z -0.2) |
| student single fd10_kd05w192_s951 (TO)  | 0.857 | 0.033   | [0.791, 0.921] | +0.014+-0.014 (z +1.0) |
| G3 file (hard labels, FT+TO)            | 0.880 | 0.034   | [0.812, 0.944] | -0.009+-0.011 (z -0.8) |
| popularity only (it_hist_all_luo)       | 0.722 | 0.219   | [0.354, 1.198] | +0.149+-0.218 (z +0.7) |
| exposure only (xp_imp_c50)              | 1.195 | 0.127   | [0.960, 1.462] | -0.325+-0.132 (z -2.5) |
| pre-trained encoder score (pre2_fe)     | 0.875 | 0.049   | [0.781, 0.972] | -0.005+-0.045 (z -0.1) |
| max cosine to history (all 256)         | 0.887 | 0.060   | [0.777, 1.011] | -0.017+-0.058 (z -0.3) |
| cosine to user embedding                | 0.959 | 0.064   | [0.840, 1.093] | -0.089+-0.062 (z -1.4) |
| f1 only (+ sign, chosen by overall AUC) | 0.779 | 0.073   | [0.647, 0.932] | +0.091+-0.071 (z +1.3) |

### R = AUC - 0.5(H=256) / AUC - 0.5(NEAR); cluster bootstrap 2000 draws
| ranker                                  | R     | boot SE | 95% CI         | R_FINAL - R_ranker     |
|-----------------------------------------|-------|---------|----------------|------------------------|
| FINAL                                   | 0.941 | 0.018   | [0.905, 0.975] |                        |
| FD16 (16 students, no pop-light)        | 0.940 | 0.018   | [0.903, 0.974] | +0.001+-0.002 (z +0.8) |
| FD16 TO branch                          | 0.941 | 0.018   | [0.905, 0.975] | +0.000+-0.003 (z +0.2) |
| FD16 FT branch                          | 0.936 | 0.018   | [0.901, 0.971] | +0.005+-0.003 (z +1.7) |
| POPLIGHT pool (4)                       | 0.940 | 0.018   | [0.905, 0.975] | +0.001+-0.004 (z +0.2) |
| TEACHER 11 hard-label (TO)              | 0.943 | 0.019   | [0.906, 0.978] | -0.001+-0.004 (z -0.4) |
| hard-label single w5_e10_base_s951 (TO) | 0.928 | 0.019   | [0.889, 0.965] | +0.013+-0.007 (z +1.9) |
| student single fd10_kd05w192_s951 (TO)  | 0.936 | 0.019   | [0.899, 0.972] | +0.005+-0.005 (z +1.0) |
| G3 file (hard labels, FT+TO)            | 0.943 | 0.018   | [0.907, 0.979] | -0.002+-0.004 (z -0.6) |
| popularity only (it_hist_all_luo)       | 0.758 | 0.090   | [0.586, 0.939] | +0.183+-0.090 (z +2.0) |
| exposure only (xp_imp_c50)              | 0.538 | 0.143   | [0.276, 0.838] | +0.403+-0.143 (z +2.8) |
| pre-trained encoder score (pre2_fe)     | 0.950 | 0.029   | [0.894, 1.009] | -0.009+-0.025 (z -0.4) |
| max cosine to history (all 256)         | 0.933 | 0.032   | [0.871, 0.995] | +0.008+-0.029 (z +0.3) |
| cosine to user embedding                | 0.963 | 0.036   | [0.897, 1.038] | -0.022+-0.033 (z -0.7) |
| f1 only (+ sign, chosen by overall AUC) | 0.923 | 0.040   | [0.848, 1.006] | +0.018+-0.036 (z +0.5) |

### R = skill(H=256) / skill(REST); cluster bootstrap 2000 draws
| ranker                                  | R     | boot SE | 95% CI         | R_FINAL - R_ranker     |
|-----------------------------------------|-------|---------|----------------|------------------------|
| FINAL                                   | 0.844 | 0.032   | [0.781, 0.909] |                        |
| FD16 (16 students, no pop-light)        | 0.845 | 0.032   | [0.782, 0.910] | -0.001+-0.005 (z -0.1) |
| FD16 TO branch                          | 0.859 | 0.033   | [0.794, 0.924] | -0.015+-0.008 (z -2.0) |
| FD16 FT branch                          | 0.832 | 0.032   | [0.769, 0.894] | +0.013+-0.008 (z +1.6) |
| POPLIGHT pool (4)                       | 0.840 | 0.031   | [0.778, 0.904] | +0.005+-0.013 (z +0.4) |
| TEACHER 11 hard-label (TO)              | 0.858 | 0.033   | [0.794, 0.925] | -0.014+-0.011 (z -1.3) |
| hard-label single w5_e10_base_s951 (TO) | 0.852 | 0.034   | [0.786, 0.918] | -0.007+-0.017 (z -0.4) |
| student single fd10_kd05w192_s951 (TO)  | 0.828 | 0.032   | [0.763, 0.892] | +0.016+-0.014 (z +1.2) |
| G3 file (hard labels, FT+TO)            | 0.853 | 0.033   | [0.790, 0.919] | -0.009+-0.011 (z -0.8) |
| popularity only (it_hist_all_luo)       | 0.603 | 0.170   | [0.304, 0.967] | +0.241+-0.169 (z +1.4) |
| exposure only (xp_imp_c50)              | 1.464 | 0.204   | [1.124, 1.895] | -0.619+-0.209 (z -3.0) |
| pre-trained encoder score (pre2_fe)     | 0.826 | 0.045   | [0.742, 0.919] | +0.018+-0.041 (z +0.4) |
| max cosine to history (all 256)         | 0.842 | 0.056   | [0.740, 0.957] | +0.003+-0.055 (z +0.0) |
| cosine to user embedding                | 0.912 | 0.061   | [0.802, 1.043] | -0.067+-0.057 (z -1.2) |
| f1 only (+ sign, chosen by overall AUC) | 0.782 | 0.071   | [0.645, 0.926] | +0.063+-0.068 (z +0.9) |

### R = NDCG - E_random(H=256) / NDCG - E_random(REST); cluster bootstrap 2000 draws
| ranker                                  | R     | boot SE | 95% CI         | R_FINAL - R_ranker     |
|-----------------------------------------|-------|---------|----------------|------------------------|
| FINAL                                   | 0.841 | 0.031   | [0.782, 0.904] |                        |
| FD16 (16 students, no pop-light)        | 0.842 | 0.031   | [0.784, 0.905] | -0.001+-0.005 (z -0.2) |
| FD16 TO branch                          | 0.857 | 0.031   | [0.798, 0.921] | -0.015+-0.008 (z -2.1) |
| FD16 FT branch                          | 0.830 | 0.031   | [0.773, 0.891] | +0.011+-0.007 (z +1.5) |
| POPLIGHT pool (4)                       | 0.832 | 0.031   | [0.773, 0.894] | +0.009+-0.012 (z +0.7) |
| TEACHER 11 hard-label (TO)              | 0.857 | 0.032   | [0.797, 0.920] | -0.016+-0.011 (z -1.5) |
| hard-label single w5_e10_base_s951 (TO) | 0.850 | 0.032   | [0.789, 0.913] | -0.009+-0.016 (z -0.6) |
| student single fd10_kd05w192_s951 (TO)  | 0.829 | 0.031   | [0.769, 0.891] | +0.013+-0.013 (z +1.0) |
| G3 file (hard labels, FT+TO)            | 0.851 | 0.032   | [0.791, 0.914] | -0.010+-0.010 (z -0.9) |
| popularity only (it_hist_all_luo)       | 0.544 | 0.148   | [0.276, 0.856] | +0.297+-0.149 (z +2.0) |
| exposure only (xp_imp_c50)              | 1.478 | 0.166   | [1.186, 1.834] | -0.636+-0.169 (z -3.8) |
| pre-trained encoder score (pre2_fe)     | 0.826 | 0.043   | [0.742, 0.911] | +0.016+-0.041 (z +0.4) |
| max cosine to history (all 256)         | 0.868 | 0.056   | [0.759, 0.979] | -0.026+-0.053 (z -0.5) |
| cosine to user embedding                | 0.902 | 0.055   | [0.795, 1.009] | -0.060+-0.054 (z -1.1) |
| f1 only (+ sign, chosen by overall AUC) | 0.760 | 0.066   | [0.639, 0.893] | +0.081+-0.063 (z +1.3) |

### R = AUC - 0.5(H=256) / AUC - 0.5(REST); cluster bootstrap 2000 draws
| ranker                                  | R     | boot SE | 95% CI         | R_FINAL - R_ranker     |
|-----------------------------------------|-------|---------|----------------|------------------------|
| FINAL                                   | 0.921 | 0.017   | [0.887, 0.955] |                        |
| FD16 (16 students, no pop-light)        | 0.920 | 0.017   | [0.886, 0.953] | +0.002+-0.001 (z +1.1) |
| FD16 TO branch                          | 0.921 | 0.017   | [0.887, 0.956] | +0.000+-0.002 (z +0.0) |
| FD16 FT branch                          | 0.916 | 0.017   | [0.882, 0.950] | +0.005+-0.002 (z +2.0) |
| POPLIGHT pool (4)                       | 0.920 | 0.017   | [0.885, 0.954] | +0.001+-0.004 (z +0.3) |
| TEACHER 11 hard-label (TO)              | 0.921 | 0.018   | [0.887, 0.957] | +0.000+-0.004 (z +0.0) |
| hard-label single w5_e10_base_s951 (TO) | 0.913 | 0.018   | [0.877, 0.951] | +0.008+-0.007 (z +1.2) |
| student single fd10_kd05w192_s951 (TO)  | 0.917 | 0.018   | [0.882, 0.951] | +0.005+-0.005 (z +1.0) |
| G3 file (hard labels, FT+TO)            | 0.922 | 0.018   | [0.888, 0.956] | -0.001+-0.003 (z -0.2) |
| popularity only (it_hist_all_luo)       | 0.634 | 0.069   | [0.499, 0.774] | +0.287+-0.070 (z +4.1) |
| exposure only (xp_imp_c50)              | 0.415 | 0.105   | [0.207, 0.624] | +0.507+-0.106 (z +4.8) |
| pre-trained encoder score (pre2_fe)     | 0.929 | 0.028   | [0.880, 0.985] | -0.008+-0.022 (z -0.4) |
| max cosine to history (all 256)         | 0.917 | 0.031   | [0.859, 0.978] | +0.005+-0.027 (z +0.2) |
| cosine to user embedding                | 0.925 | 0.033   | [0.861, 0.990] | -0.004+-0.028 (z -0.1) |
| f1 only (+ sign, chosen by overall AUC) | 0.895 | 0.038   | [0.820, 0.971] | +0.027+-0.034 (z +0.8) |
```

### s3 matching
```
### covariate shift, H=256 vs NEAR and vs REST (standardised mean difference, sd of ALL feeds)
| covariate          | mean H=256 | mean NEAR | mean REST | SMD vs NEAR | SMD vs REST |
|--------------------|------------|-----------|-----------|-------------|-------------|
| M                  | 96.541     | 96.321    | 96.303    | +0.00       | +0.00       |
| logM               | 4.084      | 4.077     | 4.078     | +0.01       | +0.01       |
| day                | 20718.961  | 20718.947 | 20718.951 | +0.01       | +0.01       |
| hour               | 13.138     | 12.902    | 12.995    | +0.04       | +0.02       |
| uid0               | 0.002      | 0.023     | 0.064     | -0.10       | -0.30       |
| cold_share         | 0.375      | 0.369     | 0.383     | +0.05       | -0.06       |
| emb_miss_share     | 0.000      | 0.000     | 0.000     | -0.02       | -0.03       |
| n_ccat             | 12.366     | 11.806    | 11.863    | +0.10       | +0.09       |
| top_ccat_share     | 0.376      | 0.370     | 0.381     | +0.03       | -0.03       |
| ccat_in_hist_share | 0.878      | 0.884     | 0.832     | -0.04       | +0.30       |
| h64_ncat           | 5.968      | 6.361     | 5.917     | -0.13       | +0.02       |
| h64_topcat         | 0.590      | 0.556     | 0.555     | +0.15       | +0.15       |
| h64_noemb          | 0.001      | 0.020     | 0.060     | -0.17       | -0.52       |
| h64_distinct       | 0.658      | 0.738     | 0.713     | -0.40       | -0.27       |
| h64_src_top        | 0.601      | 0.601     | 0.591     | +0.00       | +0.06       |
| h64_et1            | 0.520      | 0.634     | 0.584     | -0.41       | -0.23       |
| h64_et2            | 0.088      | 0.060     | 0.062     | +0.31       | +0.30       |
| h64_et3            | 0.170      | 0.129     | 0.135     | +0.27       | +0.23       |
| h64_et4            | 0.067      | 0.065     | 0.073     | +0.02       | -0.05       |
| h64_et5            | 0.058      | 0.031     | 0.028     | +0.40       | +0.45       |
| h64_et6            | 0.051      | 0.026     | 0.024     | +0.38       | +0.42       |
| h64_et7            | 0.019      | 0.024     | 0.031     | -0.07       | -0.16       |
| h64_et8            | 0.020      | 0.013     | 0.012     | +0.20       | +0.20       |
| h64_et9            | 0.006      | 0.007     | 0.009     | -0.04       | -0.09       |
| h64_et10           | 0.002      | 0.012     | 0.023     | -0.12       | -0.26       |
| smax64_mean        | 0.759      | 0.781     | 0.759     | -0.21       | -0.00       |
| smax64_sd          | 0.138      | 0.132     | 0.132     | +0.16       | +0.18       |
| cos_u_mean         | 0.466      | 0.479     | 0.479     | -0.20       | -0.20       |
| cos_u_sd           | 0.086      | 0.082     | 0.081     | +0.19       | +0.24       |
| logpop_mean        | 1.563      | 1.590     | 1.538     | -0.06       | +0.06       |
| logpop_sd          | 1.560      | 1.551     | 1.527     | +0.04       | +0.12       |
| logexp_mean        | 0.758      | 0.775     | 0.758     | -0.09       | +0.00       |
| logexp_sd          | 0.595      | 0.596     | 0.595     | -0.01       | +0.01       |
| cand_pair_cos      | 0.473      | 0.493     | 0.502     | -0.32       | -0.46       |
| cand_in_hist_share | 0.000      | 0.000     | 0.000     | +0.00       | +0.00       |
| npos               | 1.569      | 1.580     | 1.506     | -0.01       | +0.04       |

### outcome NDCG; reference REST: D = mean(H=256) - mean(REST), cross-fitted LightGBM on REST feeds (5-fold by user)
| matching set                    | D        | SE      | z    | share of raw left | OOF R2 on ref |
|---------------------------------|----------|---------|------|-------------------|---------------|
| raw (no matching)               | -0.01711 | 0.00443 | -3.9 | 100%              |               |
| X1 (declared label-free set)    | -0.00894 | 0.00393 | -2.3 | 52%               | 0.2151        |
| X2 = X1 + range restriction     | -0.01033 | 0.00393 | -2.6 | 60%               | 0.2203        |
| X1 + n_pos (NOT label-free)     | -0.01365 | 0.00384 | -3.6 | 80%               | 0.2432        |
| X2 + n_pos (NOT label-free)     | -0.01511 | 0.00384 | -3.9 | 88%               | 0.2482        |
| M + n_pos only (NOT label-free) | -0.01982 | 0.00384 | -5.2 | 116%              | 0.2481        |
| M only                          | -0.01610 | 0.00394 | -4.1 | 94%               | 0.2183        |

### outcome NDCG; reference NEAR: D = mean(H=256) - mean(NEAR), cross-fitted LightGBM on NEAR feeds (5-fold by user)
| matching set                    | D        | SE      | z    | share of raw left | OOF R2 on ref |
|---------------------------------|----------|---------|------|-------------------|---------------|
| raw (no matching)               | -0.01549 | 0.00470 | -3.3 | 100%              |               |
| X1 (declared label-free set)    | -0.00932 | 0.00418 | -2.2 | 60%               | 0.2162        |
| X2 = X1 + range restriction     | -0.00899 | 0.00418 | -2.1 | 58%               | 0.2142        |
| X1 + n_pos (NOT label-free)     | -0.01398 | 0.00408 | -3.4 | 90%               | 0.2447        |
| X2 + n_pos (NOT label-free)     | -0.01381 | 0.00409 | -3.4 | 89%               | 0.2426        |
| M + n_pos only (NOT label-free) | -0.01582 | 0.00407 | -3.9 | 102%              | 0.2508        |
| M only                          | -0.01429 | 0.00418 | -3.4 | 92%               | 0.2168        |

### outcome per-feed AUC; reference REST: D = mean(H=256) - mean(REST), cross-fitted LightGBM on REST feeds (5-fold by user)
| matching set                    | D        | SE      | z    | share of raw left | OOF R2 on ref |
|---------------------------------|----------|---------|------|-------------------|---------------|
| raw (no matching)               | -0.01877 | 0.00415 | -4.5 | 100%              |               |
| X1 (declared label-free set)    | -0.01392 | 0.00416 | -3.3 | 74%               | -0.0118       |
| X2 = X1 + range restriction     | -0.01702 | 0.00416 | -4.1 | 91%               | -0.0090       |
| X1 + n_pos (NOT label-free)     | -0.01377 | 0.00416 | -3.3 | 73%               | -0.0121       |
| X2 + n_pos (NOT label-free)     | -0.01654 | 0.00415 | -4.0 | 88%               | -0.0089       |
| M + n_pos only (NOT label-free) | -0.01808 | 0.00417 | -4.3 | 96%               | -0.0119       |
| M only                          | -0.01835 | 0.00417 | -4.4 | 98%               | -0.0111       |

### outcome per-feed AUC; reference NEAR: D = mean(H=256) - mean(NEAR), cross-fitted LightGBM on NEAR feeds (5-fold by user)
| matching set                    | D        | SE      | z    | share of raw left | OOF R2 on ref |
|---------------------------------|----------|---------|------|-------------------|---------------|
| raw (no matching)               | -0.01377 | 0.00441 | -3.1 | 100%              |               |
| X1 (declared label-free set)    | -0.01472 | 0.00444 | -3.3 | 107%              | -0.0151       |
| X2 = X1 + range restriction     | -0.01509 | 0.00444 | -3.4 | 110%              | -0.0122       |
| X1 + n_pos (NOT label-free)     | -0.01415 | 0.00443 | -3.2 | 103%              | -0.0131       |
| X2 + n_pos (NOT label-free)     | -0.01511 | 0.00443 | -3.4 | 110%              | -0.0109       |
| M + n_pos only (NOT label-free) | -0.01337 | 0.00443 | -3.0 | 97%               | -0.0108       |
| M only                          | -0.01351 | 0.00444 | -3.0 | 98%               | -0.0103       |

### outcome skill; reference REST: D = mean(H=256) - mean(REST), cross-fitted LightGBM on REST feeds (5-fold by user)
| matching set                    | D        | SE      | z    | share of raw left | OOF R2 on ref |
|---------------------------------|----------|---------|------|-------------------|---------------|
| raw (no matching)               | -0.02762 | 0.00586 | -4.7 | 100%              |               |
| X1 (declared label-free set)    | -0.01831 | 0.00581 | -3.2 | 66%               | 0.0167        |
| X2 = X1 + range restriction     | -0.02116 | 0.00581 | -3.6 | 77%               | 0.0227        |
| X1 + n_pos (NOT label-free)     | -0.01947 | 0.00580 | -3.4 | 70%               | 0.0162        |
| X2 + n_pos (NOT label-free)     | -0.02181 | 0.00581 | -3.8 | 79%               | 0.0237        |
| M + n_pos only (NOT label-free) | -0.02830 | 0.00581 | -4.9 | 102%              | 0.0280        |
| M only                          | -0.02690 | 0.00581 | -4.6 | 97%               | 0.0279        |

### outcome skill; reference NEAR: D = mean(H=256) - mean(NEAR), cross-fitted LightGBM on NEAR feeds (5-fold by user)
| matching set                    | D        | SE      | z    | share of raw left | OOF R2 on ref |
|---------------------------------|----------|---------|------|-------------------|---------------|
| raw (no matching)               | -0.02208 | 0.00623 | -3.5 | 100%              |               |
| X1 (declared label-free set)    | -0.01805 | 0.00620 | -2.9 | 82%               | 0.0126        |
| X2 = X1 + range restriction     | -0.01872 | 0.00621 | -3.0 | 85%               | 0.0107        |
| X1 + n_pos (NOT label-free)     | -0.01947 | 0.00619 | -3.1 | 88%               | 0.0131        |
| X2 + n_pos (NOT label-free)     | -0.01960 | 0.00620 | -3.2 | 89%               | 0.0123        |
| M + n_pos only (NOT label-free) | -0.02180 | 0.00617 | -3.5 | 99%               | 0.0241        |
| M only                          | -0.02117 | 0.00618 | -3.4 | 96%               | 0.0236        |

### OLS cross-check: coefficient of 1[H=256] on NDCG (all feeds, cluster-robust SE)
| covariates                      | coef 1[H=256] | SE      |
|---------------------------------|---------------|---------|
| X1 (declared label-free set)    | -0.01299      | 0.00420 |
| X2 = X1 + range restriction     | -0.01316      | 0.00425 |
| X1 + n_pos (NOT label-free)     | -0.01544      | 0.00415 |
| X2 + n_pos (NOT label-free)     | -0.01584      | 0.00421 |
| M + n_pos only (NOT label-free) | -0.01822      | 0.00407 |
| M only                          | -0.01641      | 0.00391 |
exact-cell matching [M decile x cold tercile]: NDCG H=256 0.39056 vs re-weighted REST 0.40665 -> D -0.01609 (raw -0.01711); ESS of REST 10121
exact-cell matching [M decile x cold tercile x n_pos(1,2,3+) (NOT label-free)]: NDCG H=256 0.39056 vs re-weighted REST 0.41038 -> D -0.01982 (raw -0.01711); ESS of REST 9813

### inside H=256: NDCG / AUC by label-free splits
| split                                     | feeds | NDCG           | AUC            | mean M | n_pos |
|-------------------------------------------|-------|----------------|----------------|--------|-------|
| user seen in train                        | 2158  | 0.3894+-0.0050 | 0.7210+-0.0048 | 98     | 1.64  |
| user not in train                         | 2018  | 0.3918+-0.0053 | 0.7191+-0.0050 | 95     | 1.50  |
| distinct-item share of newest 64 < median | 2034  | 0.3916+-0.0054 | 0.7239+-0.0051 | 94     | 1.40  |
| distinct-item share >= median             | 2142  | 0.3895+-0.0050 | 0.7165+-0.0048 | 99     | 1.73  |
| history categories (all 256) <= 5         | 445   | 0.3868+-0.0110 | 0.7245+-0.0106 | 92     | 1.50  |
| history categories 6-10                   | 1419  | 0.3962+-0.0063 | 0.7400+-0.0058 | 98     | 1.55  |
| history categories > 10                   | 2312  | 0.3878+-0.0049 | 0.7070+-0.0048 | 96     | 1.59  |
| NEAR: user seen in train                  | 2029  | 0.3990+-0.0054 | 0.7273+-0.0051 | 97     | 1.68  |
| NEAR: user not in train                   | 5136  | 0.4088+-0.0035 | 0.7365+-0.0032 | 96     | 1.54  |
```

### s9 covariate blocks
```

### NDCG, reference REST
| matching block                  | D after matching | SE      | share of raw left |
|---------------------------------|------------------|---------|-------------------|
| raw                             | -0.01711         | 0.00443 | 100%              |
| F (feed only)                   | -0.01332         | 0.00395 | 78%               |
| F + T (event-type mix)          | -0.00814         | 0.00394 | 48%               |
| F + T + R (= X1)                | -0.00917         | 0.00393 | 54%               |
| X1 + A (user activity markers)  | -0.00630         | 0.00393 | 37%               |
| F + A                           | -0.01133         | 0.00395 | 66%               |
| A only                          | -0.01370         | 0.00443 | 80%               |
| X1 + A + n_pos (NOT label-free) | -0.01083         | 0.00384 | 63%               |

### NDCG, reference NEAR
| matching block                  | D after matching | SE      | share of raw left |
|---------------------------------|------------------|---------|-------------------|
| raw                             | -0.01549         | 0.00470 | 100%              |
| F (feed only)                   | -0.01353         | 0.00420 | 87%               |
| F + T (event-type mix)          | -0.00732         | 0.00419 | 47%               |
| F + T + R (= X1)                | -0.00969         | 0.00418 | 63%               |
| X1 + A (user activity markers)  | -0.00631         | 0.00418 | 41%               |
| F + A                           | -0.01140         | 0.00420 | 74%               |
| A only                          | -0.01310         | 0.00470 | 85%               |
| X1 + A + n_pos (NOT label-free) | -0.01096         | 0.00409 | 71%               |

user-activity markers: share seen in train H=256 0.517, NEAR 0.283, REST 0.217; mean train feeds per user (uid != 0) H=256 0.98, NEAR 0.40
| cell                   | feeds | FINAL NDCG     | n_pos | M   |
|------------------------|-------|----------------|-------|-----|
| H=256, not in train    | 2018  | 0.3918+-0.0053 | 1.50  | 95  |
| H=256, 1-3 train feeds | 1959  | 0.3901+-0.0053 | 1.59  | 98  |
| H=256, 4+ train feeds  | 199   | 0.3821+-0.0156 | 2.10  | 105 |
| NEAR, not in train     | 5136  | 0.4088+-0.0035 | 1.54  | 96  |
| NEAR, 1-3 train feeds  | 1961  | 0.3975+-0.0054 | 1.67  | 97  |
| NEAR, 4+ train feeds   | 68    | 0.4425+-0.0366 | 1.94  | 101 |
```

### s4 positives
```
### rank of the positives under FINAL, and cold positives, by H bucket
| H       | positives | top-1 | top-5 | top-10 | rank>=20 | median rank | median pct | mean pct | cold share of positives | cold share of candidates | ratio |
|---------|-----------|-------|-------|--------|----------|-------------|------------|----------|-------------------------|--------------------------|-------|
| H=0     | 229       | 0.087 | 0.301 | 0.437  | 0.402    | 12          | 0.207      | 0.275    | 0.332                   | 0.513                    | 0.65  |
| 1-4     | 175       | 0.177 | 0.343 | 0.520  | 0.280    | 9           | 0.158      | 0.247    | 0.337                   | 0.509                    | 0.66  |
| 5-9     | 184       | 0.136 | 0.353 | 0.473  | 0.359    | 10          | 0.172      | 0.244    | 0.304                   | 0.466                    | 0.65  |
| 10-19   | 364       | 0.121 | 0.319 | 0.484  | 0.376    | 10          | 0.171      | 0.247    | 0.297                   | 0.437                    | 0.68  |
| 20-49   | 1042      | 0.107 | 0.329 | 0.487  | 0.372    | 10          | 0.200      | 0.262    | 0.287                   | 0.407                    | 0.70  |
| 50-99   | 2119      | 0.098 | 0.323 | 0.463  | 0.387    | 11          | 0.182      | 0.248    | 0.285                   | 0.384                    | 0.74  |
| 100-255 | 11319     | 0.085 | 0.275 | 0.407  | 0.427    | 14          | 0.211      | 0.274    | 0.293                   | 0.368                    | 0.80  |
| H=256   | 6551      | 0.075 | 0.255 | 0.398  | 0.428    | 15          | 0.222      | 0.283    | 0.327                   | 0.375                    | 0.87  |

| H       | kind          | n    | top-1 | top-5 | top-10 | mean pct |
|---------|---------------|------|-------|-------|--------|----------|
| 100-255 | cold positive | 3319 | 0.072 | 0.239 | 0.366  | 0.320    |
| 100-255 | warm positive | 8000 | 0.090 | 0.290 | 0.424  | 0.255    |
| H=256   | cold positive | 2139 | 0.061 | 0.216 | 0.345  | 0.311    |
| H=256   | warm positive | 4412 | 0.082 | 0.274 | 0.424  | 0.270    |

### max cosine to the history and the index of the most similar event (0 = newest)
| H       | set       | n      | mean smax | median smax | share smax>=0.9 | share smax<0.7 | idx<16 | idx 16-63 | idx>=64 | idx>=128 | median idx |
|---------|-----------|--------|-----------|-------------|-----------------|----------------|--------|-----------|---------|----------|------------|
| 1-4     | positives | 174    | 0.709     | 0.742       | 0.184           | 0.420          | 1.000  | 0.000     | 0.000   | 0.000    | 0          |
| 1-4     | negatives | 13455  | 0.607     | 0.608       | 0.058           | 0.656          | 1.000  | 0.000     | 0.000   | 0.000    | 0          |
| 5-9     | positives | 183    | 0.780     | 0.815       | 0.273           | 0.273          | 1.000  | 0.000     | 0.000   | 0.000    | 1          |
| 5-9     | negatives | 13081  | 0.668     | 0.692       | 0.101           | 0.514          | 1.000  | 0.000     | 0.000   | 0.000    | 2          |
| 10-19   | positives | 363    | 0.810     | 0.850       | 0.317           | 0.207          | 0.983  | 0.017     | 0.000   | 0.000    | 4          |
| 10-19   | negatives | 27080  | 0.714     | 0.737       | 0.151           | 0.425          | 0.983  | 0.017     | 0.000   | 0.000    | 5          |
| 20-49   | positives | 1040   | 0.840     | 0.869       | 0.385           | 0.143          | 0.660  | 0.340     | 0.000   | 0.000    | 10         |
| 20-49   | negatives | 70884  | 0.754     | 0.774       | 0.201           | 0.340          | 0.622  | 0.378     | 0.000   | 0.000    | 11         |
| 50-99   | positives | 2119   | 0.862     | 0.887       | 0.458           | 0.092          | 0.454  | 0.478     | 0.068   | 0.000    | 19         |
| 50-99   | negatives | 148485 | 0.784     | 0.807       | 0.246           | 0.264          | 0.368  | 0.535     | 0.097   | 0.000    | 24         |
| 100-255 | positives | 11318  | 0.879     | 0.901       | 0.506           | 0.055          | 0.258  | 0.371     | 0.371   | 0.104    | 43         |
| 100-255 | negatives | 678707 | 0.815     | 0.838       | 0.304           | 0.185          | 0.205  | 0.357     | 0.438   | 0.138    | 54         |
| H=256   | positives | 6548   | 0.876     | 0.894       | 0.469           | 0.046          | 0.200  | 0.281     | 0.519   | 0.281    | 68         |
| H=256   | negatives | 396559 | 0.815     | 0.838       | 0.293           | 0.180          | 0.141  | 0.265     | 0.594   | 0.342    | 86         |

### FINAL's rank of the positive by position / strength of its best-matching history event
| H       | positives with      | n    | share | mean smax | top-1 | top-5 | mean pct (0 = best) |
|---------|---------------------|------|-------|-----------|-------|-------|---------------------|
| 100-255 | best match idx < 16 | 2915 | 0.258 | 0.886     | 0.126 | 0.366 | 0.210               |
| 100-255 | idx 16-63           | 4201 | 0.371 | 0.882     | 0.074 | 0.249 | 0.280               |
| 100-255 | idx 64-127          | 3025 | 0.267 | 0.877     | 0.069 | 0.243 | 0.297               |
| 100-255 | idx >= 128          | 1177 | 0.104 | 0.859     | 0.061 | 0.223 | 0.348               |
| 100-255 | smax >= 0.9         | 5724 | 0.506 | 0.951     | 0.121 | 0.347 | 0.193               |
| 100-255 | 0.8-0.9             | 3554 | 0.314 | 0.856     | 0.061 | 0.232 | 0.292               |
| 100-255 | 0.7-0.8             | 1415 | 0.125 | 0.758     | 0.028 | 0.160 | 0.422               |
| 100-255 | < 0.7               | 625  | 0.055 | 0.628     | 0.014 | 0.117 | 0.575               |
| H=256   | best match idx < 16 | 1309 | 0.200 | 0.885     | 0.128 | 0.355 | 0.204               |
| H=256   | idx 16-63           | 1843 | 0.281 | 0.880     | 0.071 | 0.245 | 0.277               |
| H=256   | idx 64-127          | 1553 | 0.237 | 0.877     | 0.055 | 0.220 | 0.305               |
| H=256   | idx >= 128          | 1843 | 0.281 | 0.864     | 0.059 | 0.222 | 0.328               |
| H=256   | smax >= 0.9         | 3070 | 0.469 | 0.948     | 0.111 | 0.329 | 0.199               |
| H=256   | 0.8-0.9             | 2271 | 0.347 | 0.856     | 0.049 | 0.213 | 0.293               |
| H=256   | 0.7-0.8             | 908  | 0.139 | 0.762     | 0.043 | 0.171 | 0.426               |
| H=256   | < 0.7               | 299  | 0.046 | 0.624     | 0.010 | 0.064 | 0.645               |

### trivial history rankers, within-feed AUC by H bucket
| ranker                   | 1-4    | 5-9    | 10-19  | 20-49  | 50-99  | 100-255 | H=256  |
|--------------------------|--------|--------|--------|--------|--------|---------|--------|
| smax newest 8            | 0.6668 | 0.6794 | 0.6812 | 0.6523 | 0.6485 | 0.6362  | 0.6271 |
| smax newest 16           | 0.6668 | 0.6791 | 0.6735 | 0.6583 | 0.6594 | 0.6455  | 0.6331 |
| smax newest 32           | 0.6668 | 0.6791 | 0.6715 | 0.6634 | 0.6627 | 0.6508  | 0.6379 |
| smax newest 64           | 0.6668 | 0.6791 | 0.6715 | 0.6606 | 0.6653 | 0.6510  | 0.6346 |
| smax newest 128          | 0.6668 | 0.6791 | 0.6715 | 0.6606 | 0.6640 | 0.6457  | 0.6338 |
| smax newest 256          | 0.6668 | 0.6791 | 0.6715 | 0.6606 | 0.6640 | 0.6458  | 0.6360 |
| mean top-1 cosine        | 0.6669 | 0.6788 | 0.6719 | 0.6607 | 0.6640 | 0.6458  | 0.6360 |
| mean top-3 cosine        | 0.6708 | 0.6550 | 0.6785 | 0.6564 | 0.6594 | 0.6465  | 0.6379 |
| mean top-8 cosine        | 0.6720 | 0.6449 | 0.6677 | 0.6522 | 0.6448 | 0.6408  | 0.6352 |
| mean top-16 cosine       | 0.6720 | 0.6437 | 0.6647 | 0.6430 | 0.6315 | 0.6355  | 0.6318 |
| mean top-32 cosine       | 0.6720 | 0.6437 | 0.6641 | 0.6385 | 0.6191 | 0.6243  | 0.6273 |
| mean top-64 cosine       | 0.6720 | 0.6437 | 0.6641 | 0.6376 | 0.6122 | 0.6122  | 0.6204 |
| mean cosine (all events) | 0.6720 | 0.6437 | 0.6641 | 0.6373 | 0.6112 | 0.6005  | 0.6034 |

### trivial history rankers, NDCG by H bucket
| ranker                   | 1-4    | 5-9    | 10-19  | 20-49  | 50-99  | 100-255 | H=256  |
|--------------------------|--------|--------|--------|--------|--------|---------|--------|
| smax newest 8            | 0.3597 | 0.3750 | 0.3505 | 0.3561 | 0.3605 | 0.3526  | 0.3460 |
| smax newest 16           | 0.3597 | 0.3750 | 0.3424 | 0.3557 | 0.3617 | 0.3533  | 0.3455 |
| smax newest 32           | 0.3597 | 0.3750 | 0.3414 | 0.3547 | 0.3583 | 0.3519  | 0.3430 |
| smax newest 64           | 0.3597 | 0.3750 | 0.3414 | 0.3494 | 0.3534 | 0.3504  | 0.3391 |
| smax newest 128          | 0.3597 | 0.3750 | 0.3414 | 0.3494 | 0.3520 | 0.3457  | 0.3378 |
| smax newest 256          | 0.3597 | 0.3750 | 0.3414 | 0.3494 | 0.3520 | 0.3450  | 0.3383 |
| mean top-1 cosine        | 0.3596 | 0.3742 | 0.3411 | 0.3489 | 0.3506 | 0.3438  | 0.3374 |
| mean top-3 cosine        | 0.3599 | 0.3640 | 0.3562 | 0.3408 | 0.3497 | 0.3448  | 0.3357 |
| mean top-8 cosine        | 0.3610 | 0.3643 | 0.3483 | 0.3419 | 0.3439 | 0.3446  | 0.3354 |
| mean top-16 cosine       | 0.3610 | 0.3638 | 0.3505 | 0.3374 | 0.3401 | 0.3435  | 0.3350 |
| mean top-32 cosine       | 0.3610 | 0.3638 | 0.3501 | 0.3385 | 0.3368 | 0.3394  | 0.3343 |
| mean top-64 cosine       | 0.3610 | 0.3638 | 0.3501 | 0.3385 | 0.3310 | 0.3348  | 0.3333 |
| mean cosine (all events) | 0.3613 | 0.3638 | 0.3501 | 0.3383 | 0.3301 | 0.3291  | 0.3264 |

### declared contrasts at H=256 (paired per feed, cluster SE), with the same contrast at H 20-99 and NEAR
| contrast                                | metric | H=256           | NEAR (100-255)  | H 20-99         |
|-----------------------------------------|--------|-----------------|-----------------|-----------------|
| smax newest 64 - smax newest 256        | NDCG   | +0.0008+-0.0021 | +0.0054+-0.0013 | +0.0009+-0.0007 |
| smax newest 64 - smax newest 256        | AUC    | -0.0013+-0.0029 | +0.0052+-0.0017 | +0.0008+-0.0010 |
| smax newest 128 - smax newest 256       | NDCG   | -0.0005+-0.0015 | +0.0007+-0.0007 | +0.0000+-0.0000 |
| smax newest 128 - smax newest 256       | AUC    | -0.0022+-0.0020 | -0.0000+-0.0009 | +0.0000+-0.0000 |
| smax newest 16 - smax newest 256        | NDCG   | +0.0072+-0.0029 | +0.0083+-0.0020 | +0.0085+-0.0029 |
| smax newest 16 - smax newest 256        | AUC    | -0.0029+-0.0041 | -0.0002+-0.0028 | -0.0039+-0.0039 |
| mean top-32 cosine - mean top-16 cosine | NDCG   | -0.0007+-0.0014 | -0.0041+-0.0011 | -0.0019+-0.0020 |
| mean top-32 cosine - mean top-16 cosine | AUC    | -0.0045+-0.0012 | -0.0112+-0.0011 | -0.0097+-0.0018 |
| mean top-64 cosine - mean top-16 cosine | NDCG   | -0.0017+-0.0019 | -0.0087+-0.0016 | -0.0056+-0.0024 |
| mean top-64 cosine - mean top-16 cosine | AUC    | -0.0114+-0.0021 | -0.0233+-0.0018 | -0.0146+-0.0025 |
| mean top-16 cosine - mean top-1 cosine  | NDCG   | -0.0024+-0.0026 | -0.0003+-0.0019 | -0.0109+-0.0037 |
| mean top-16 cosine - mean top-1 cosine  | AUC    | -0.0042+-0.0030 | -0.0103+-0.0023 | -0.0276+-0.0043 |
| mean top-3 cosine - mean top-1 cosine   | NDCG   | -0.0017+-0.0017 | +0.0010+-0.0014 | -0.0033+-0.0026 |
| mean top-3 cosine - mean top-1 cosine   | AUC    | +0.0019+-0.0017 | +0.0007+-0.0012 | -0.0045+-0.0022 |

### saturation of retrieval depth 16: share of candidates with MORE than 16 history events at cosine >= 0.9 / 0.8 / 0.7; cosine of the 16th and 32nd most similar event
| H       | cand >16 ev at 0.9 | at 0.8 | at 0.7 | POS >16 ev at 0.9 | at 0.8 | at 0.7 | mean cos of 16th | mean cos of 32nd | mean cos of 1st |
|---------|--------------------|--------|--------|-------------------|--------|--------|------------------|------------------|-----------------|
| 1-4     | 0.000              | 0.000  | 0.000  | 0.000             | 0.000  | 0.000  | nan              | nan              | 0.608           |
| 5-9     | 0.000              | 0.000  | 0.000  | 0.000             | 0.000  | 0.000  | nan              | nan              | 0.669           |
| 10-19   | 0.001              | 0.002  | 0.014  | 0.003             | 0.006  | 0.028  | 0.450            | nan              | 0.716           |
| 20-49   | 0.008              | 0.047  | 0.137  | 0.022             | 0.104  | 0.238  | 0.518            | 0.424            | 0.755           |
| 50-99   | 0.017              | 0.087  | 0.247  | 0.052             | 0.177  | 0.382  | 0.593            | 0.501            | 0.785           |
| 100-255 | 0.028              | 0.150  | 0.386  | 0.069             | 0.274  | 0.572  | 0.657            | 0.592            | 0.816           |
| H=256   | 0.029              | 0.169  | 0.429  | 0.068             | 0.292  | 0.613  | 0.672            | 0.617            | 0.816           |

within-feed AUC of (mean cosine of events ranked 17-32) residualised on (mean top-16 cosine), H=256 feeds: 0.4772 +- 0.0043 (0.5 = nothing left beyond depth 16)

### share of positives / of all candidates whose category occurs in the stored history
| H       | positives | candidates |
|---------|-----------|------------|
| 1-4     | 0.651     | 0.443      |
| 5-9     | 0.842     | 0.565      |
| 10-19   | 0.827     | 0.648      |
| 20-49   | 0.881     | 0.737      |
| 50-99   | 0.930     | 0.823      |
| 100-255 | 0.955     | 0.881      |
| H=256   | 0.955     | 0.877      |
```

### s8 history-drop, calibration, cold-start splits
```
### G1 cost of dropping history events at inference (hard-label members s951/952), by H bucket
| mode                               | ALL               | 1-4             | 5-9             | 10-19           | 20-49           | 50-99           | 100-255         | H=256           | contrast H=256 minus NEAR |
|------------------------------------|-------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|---------------------------|
| single pass, 10% of events dropped | -0.00077+-0.00032 | -0.0003+-0.0011 | -0.0051+-0.0018 | +0.0013+-0.0017 | +0.0002+-0.0011 | -0.0006+-0.0009 | -0.0007+-0.0005 | -0.0012+-0.0006 | -0.00052+-0.00078         |
| 4-mask average, 10%                | -0.00013+-0.00036 | -0.0003+-0.0019 | -0.0046+-0.0024 | +0.0015+-0.0017 | +0.0001+-0.0012 | -0.0007+-0.0010 | +0.0002+-0.0005 | -0.0005+-0.0007 | -0.00075+-0.00088         |
| single pass, 20% of events dropped | -0.00170+-0.00037 | -0.0004+-0.0010 | -0.0067+-0.0034 | +0.0020+-0.0019 | +0.0003+-0.0014 | -0.0012+-0.0010 | -0.0020+-0.0005 | -0.0020+-0.0007 | -0.00001+-0.00091         |
| 4-mask average, 20%                | -0.00098+-0.00041 | +0.0004+-0.0018 | -0.0041+-0.0029 | +0.0030+-0.0020 | +0.0015+-0.0016 | -0.0004+-0.0011 | -0.0010+-0.0006 | -0.0018+-0.0008 | -0.00085+-0.00100         |
mean number of events removed at 20%: {'1-4': np.float64(0.5), '5-9': np.float64(1.4), '10-19': np.float64(2.9), '20-49': np.float64(7.0), '50-99': np.float64(15.0), '100-255': np.float64(35.8), 'H=256': np.float64(51.2)}
cost per removed event at 20% (NDCG x 1e-4 per event): {'1-4': np.float64(-8.832), '5-9': np.float64(-48.529), '10-19': np.float64(6.929), '20-49': np.float64(0.433), '50-99': np.float64(-0.794), '100-255': np.float64(-0.553), 'H=256': np.float64(-0.388)}

### G2 teacher (11 hard-label members, mean logit) calibration by H bucket
| H       | predicted p(top-1) | realised y_top1/n_pos | realised - predicted | normalised entropy | softmax mass on positives | base rate n_pos/M |
|---------|--------------------|-----------------------|----------------------|--------------------|---------------------------|-------------------|
| H=0     | 0.1140             | 0.0901+-0.0204        | -0.0239+-0.0165      | 0.8851             | 0.0768                    | 0.0320            |
| 1-4     | 0.1678             | 0.1833+-0.0315        | +0.0155+-0.0234      | 0.8353             | 0.1199                    | 0.0366            |
| 5-9     | 0.1469             | 0.1735+-0.0320        | +0.0266+-0.0280      | 0.8510             | 0.0969                    | 0.0354            |
| 10-19   | 0.1261             | 0.1187+-0.0187        | -0.0074+-0.0169      | 0.8669             | 0.0780                    | 0.0370            |
| 20-49   | 0.1326             | 0.1123+-0.0108        | -0.0203+-0.0104      | 0.8717             | 0.0737                    | 0.0387            |
| 50-99   | 0.1264             | 0.1119+-0.0077        | -0.0145+-0.0071      | 0.8747             | 0.0757                    | 0.0375            |
| 100-255 | 0.1112             | 0.1078+-0.0035        | -0.0034+-0.0033      | 0.8926             | 0.0701                    | 0.0397            |
| H=256   | 0.1014             | 0.0935+-0.0043        | -0.0079+-0.0040      | 0.9028             | 0.0644                    | 0.0389            |
| REST    | 0.1169             | 0.1107+-0.0030        | -0.0062+-0.0028      | 0.8861             | 0.0726                    | 0.0390            |
contrast H=256 minus NEAR: predicted p(top-1) -0.0098+-0.0020 | realised -0.0143+-0.0055 | (realised - predicted) -0.0045+-0.0052 | entropy +0.0102+-0.0010

### G3 cold-start splits
| segment                      | feeds | M   | n_pos | FINAL NDCG     | FINAL AUC | popularity NDCG | popularity AUC | cos_u AUC | pop-light pool minus FINAL |
|------------------------------|-------|-----|-------|----------------|-----------|-----------------|----------------|-----------|----------------------------|
| H=0, user embedding present  | 120   | 102 | 1.13  | 0.3949+-0.0238 | 0.7525    | 0.3080          | 0.6312         | 0.6925    | -0.0036                    |
| H=0, user embedding all-zero | 74    | 97  | 1.26  | 0.3515+-0.0253 | 0.6898    | 0.2967          | 0.6007         | 0.5000    | -0.0086                    |
| H=0, uid0                    | 105   | 101 | 1.13  | 0.4180+-0.0258 | 0.7730    | 0.3221          | 0.6525         | 0.6947    | -0.0037                    |
| H=0, known user id           | 89    | 99  | 1.24  | 0.3317+-0.0225 | 0.6762    | 0.2820          | 0.5807         | 0.5299    | -0.0077                    |
| H 1-9, uid0                  | 130   | 93  | 1.28  | 0.4428+-0.0258 | 0.7421    | 0.3259          | 0.6221         | 0.6874    | -0.0117                    |
| H 1-9, known user id         | 155   | 96  | 1.24  | 0.4433+-0.0235 | 0.7635    | 0.3197          | 0.6020         | 0.6507    | -0.0059                    |
| uid0, H>=10                  | 423   | 111 | 1.75  | 0.3968+-0.0119 | 0.7541    | 0.2950          | 0.5888         | 0.6723    | -0.0024                    |
| known user, H>=10            | 13521 | 96  | 1.53  | 0.4024+-0.0021 | 0.7324    | 0.3040          | 0.5530         | 0.6401    | -0.0008                    |
test: H=0 feeds 245 (0.86%), of them user embedding all-zero 0.233; all feeds with all-zero user embedding 0.0028 (val 0.0142)
```

### s5 small segments
```
### segment make-up (label-free)
| segment      | feeds | val %  | mean M | median H | share uid0 | cold share of cands | share with non-zero user_emb | n_pos |
|--------------|-------|--------|--------|----------|------------|---------------------|------------------------------|-------|
| H=0          | 194   | 1.35   | 100.2  | 0        | 0.541      | 0.503               | 0.619                        | 1.18  |
| H 1-9        | 285   | 1.98   | 94.7   | 4        | 0.456      | 0.492               | 0.989                        | 1.26  |
| H<=9         | 479   | 3.32   | 96.9   | 2        | 0.491      | 0.496               | 0.839                        | 1.23  |
| uid0         | 658   | 4.56   | 106.1  | 25       | 1.000      | 0.468               | 0.974                        | 1.56  |
| uid0 & H>=10 | 423   | 2.93   | 111.2  | 74       | 1.000      | 0.431               | 0.991                        | 1.75  |
| ALL          | 14423 | 100.00 | 96.4   | 191      | 0.046      | 0.381               | 0.986                        | 1.52  |
uid0 feeds by H bucket: {'H=0': 105, '1-4': 86, '5-9': 44, '10-19': 70, '20-49': 91, '50-99': 87, '100-255': 168, 'H=256': 7}

### segment H=0 (194 feeds): rankers
| ranker                                  | NDCG           | AUC    | NDCG - FINAL (paired) | AUC - FINAL (paired) |
|-----------------------------------------|----------------|--------|-----------------------|----------------------|
| FINAL                                   | 0.3784+-0.0176 | 0.7286 |                       |                      |
| FD16 (16 students, no pop-light)        | 0.3777+-0.0175 | 0.7278 | -0.0007+-0.0031       | -0.0008+-0.0015      |
| FD16 TO branch                          | 0.3808+-0.0176 | 0.7300 | +0.0024+-0.0054       | +0.0014+-0.0028      |
| FD16 FT branch                          | 0.3733+-0.0175 | 0.7256 | -0.0051+-0.0036       | -0.0030+-0.0024      |
| POPLIGHT pool (4)                       | 0.3728+-0.0176 | 0.7277 | -0.0055+-0.0082       | -0.0009+-0.0046      |
| TEACHER 11 hard-label (TO)              | 0.3755+-0.0173 | 0.7250 | -0.0029+-0.0070       | -0.0036+-0.0044      |
| G3 file (hard labels, FT+TO)            | 0.3775+-0.0175 | 0.7236 | -0.0009+-0.0060       | -0.0050+-0.0039      |
| popularity only (it_hist_all_luo)       | 0.3037+-0.0144 | 0.6195 | -0.0747+-0.0172       | -0.1091+-0.0190      |
| exposure only (xp_imp_c50)              | 0.2823+-0.0133 | 0.6114 | -0.0960+-0.0163       | -0.1173+-0.0219      |
| cosine to user embedding                | 0.2799+-0.0136 | 0.6191 | -0.0985+-0.0158       | -0.1095+-0.0177      |
| pre-trained encoder score (pre2_fe)     | 0.3180+-0.0156 | 0.5857 | -0.0604+-0.0183       | -0.1429+-0.0231      |
| max cosine to history (all 256)         | 0.1856+-0.0043 | 0.5000 | -0.1928+-0.0164       | -0.2286+-0.0172      |
| f1 only (+ sign, chosen by overall AUC) | 0.2579+-0.0128 | 0.5962 | -0.1205+-0.0170       | -0.1324+-0.0182      |

### segment H 1-9 (285 feeds): rankers
| ranker                                  | NDCG           | AUC    | NDCG - FINAL (paired) | AUC - FINAL (paired) |
|-----------------------------------------|----------------|--------|-----------------------|----------------------|
| FINAL                                   | 0.4431+-0.0173 | 0.7537 |                       |                      |
| FD16 (16 students, no pop-light)        | 0.4378+-0.0171 | 0.7525 | -0.0054+-0.0021       | -0.0013+-0.0009      |
| FD16 TO branch                          | 0.4342+-0.0170 | 0.7511 | -0.0089+-0.0041       | -0.0026+-0.0021      |
| FD16 FT branch                          | 0.4426+-0.0172 | 0.7562 | -0.0005+-0.0021       | +0.0025+-0.0015      |
| POPLIGHT pool (4)                       | 0.4345+-0.0168 | 0.7521 | -0.0086+-0.0052       | -0.0016+-0.0028      |
| TEACHER 11 hard-label (TO)              | 0.4368+-0.0172 | 0.7505 | -0.0063+-0.0043       | -0.0032+-0.0030      |
| G3 file (hard labels, FT+TO)            | 0.4397+-0.0173 | 0.7524 | -0.0034+-0.0042       | -0.0013+-0.0026      |
| popularity only (it_hist_all_luo)       | 0.3225+-0.0122 | 0.6111 | -0.1206+-0.0148       | -0.1426+-0.0166      |
| exposure only (xp_imp_c50)              | 0.2966+-0.0102 | 0.5958 | -0.1465+-0.0147       | -0.1579+-0.0185      |
| cosine to user embedding                | 0.3484+-0.0131 | 0.6674 | -0.0947+-0.0144       | -0.0863+-0.0134      |
| pre-trained encoder score (pre2_fe)     | 0.3744+-0.0149 | 0.6694 | -0.0687+-0.0148       | -0.0843+-0.0148      |
| max cosine to history (all 256)         | 0.3669+-0.0147 | 0.6726 | -0.0762+-0.0160       | -0.0811+-0.0153      |
| f1 only (+ sign, chosen by overall AUC) | 0.3341+-0.0129 | 0.6634 | -0.1090+-0.0149       | -0.0904+-0.0149      |

### segment uid0 (658 feeds): rankers
| ranker                                  | NDCG           | AUC    | NDCG - FINAL (paired) | AUC - FINAL (paired) |
|-----------------------------------------|----------------|--------|-----------------------|----------------------|
| FINAL                                   | 0.4093+-0.0101 | 0.7548 |                       |                      |
| FD16 (16 students, no pop-light)        | 0.4093+-0.0101 | 0.7552 | +0.0000+-0.0012       | +0.0004+-0.0007      |
| FD16 TO branch                          | 0.4054+-0.0099 | 0.7542 | -0.0039+-0.0024       | -0.0006+-0.0011      |
| FD16 FT branch                          | 0.4084+-0.0100 | 0.7555 | -0.0009+-0.0012       | +0.0008+-0.0010      |
| POPLIGHT pool (4)                       | 0.4048+-0.0099 | 0.7548 | -0.0044+-0.0035       | +0.0001+-0.0020      |
| TEACHER 11 hard-label (TO)              | 0.4060+-0.0099 | 0.7524 | -0.0033+-0.0027       | -0.0023+-0.0019      |
| G3 file (hard labels, FT+TO)            | 0.4097+-0.0100 | 0.7543 | +0.0004+-0.0028       | -0.0005+-0.0017      |
| popularity only (it_hist_all_luo)       | 0.3054+-0.0072 | 0.6055 | -0.1039+-0.0092       | -0.1493+-0.0102      |
| exposure only (xp_imp_c50)              | 0.2800+-0.0062 | 0.5826 | -0.1293+-0.0092       | -0.1722+-0.0115      |
| cosine to user embedding                | 0.3371+-0.0082 | 0.6788 | -0.0721+-0.0094       | -0.0759+-0.0090      |
| pre-trained encoder score (pre2_fe)     | 0.3449+-0.0084 | 0.6735 | -0.0644+-0.0088       | -0.0813+-0.0087      |
| max cosine to history (all 256)         | 0.3171+-0.0079 | 0.6514 | -0.0921+-0.0094       | -0.1033+-0.0095      |
| f1 only (+ sign, chosen by overall AUC) | 0.3189+-0.0078 | 0.6641 | -0.0904+-0.0090       | -0.0906+-0.0092      |

### raw inputs: within-feed AUC (sign fixed on ALL feeds) and share of feeds where the input varies
| input                                | sign | AUC ALL | AUC H=0      | AUC H 1-9    | AUC uid0     | varies H=0 | varies H1-9 | varies uid0 |
|--------------------------------------|------|---------|--------------|--------------|--------------|------------|-------------|-------------|
| f1                                   | +    | 0.623   | 0.596+-0.015 | 0.663+-0.015 | 0.664+-0.009 | 0.50       | 0.94        | 0.92        |
| f2                                   | -    | 0.506   | 0.493+-0.014 | 0.493+-0.014 | 0.495+-0.009 | 0.58       | 0.94        | 0.94        |
| f3                                   | +    | 0.601   | 0.613+-0.013 | 0.651+-0.013 | 0.658+-0.008 | 0.62       | 0.99        | 0.98        |
| f4                                   | +    | 0.545   | 0.620+-0.018 | 0.612+-0.016 | 0.607+-0.010 | 1.00       | 1.00        | 1.00        |
| f5                                   | +    | 0.561   | 0.650+-0.018 | 0.643+-0.015 | 0.629+-0.009 | 1.00       | 1.00        | 1.00        |
| it_hist_all_luo                      | +    | 0.556   | 0.620+-0.020 | 0.611+-0.016 | 0.606+-0.010 | 1.00       | 1.00        | 1.00        |
| it_hist_et1_luo                      | +    | 0.557   | 0.624+-0.020 | 0.618+-0.016 | 0.612+-0.010 | 1.00       | 1.00        | 1.00        |
| xp_imp_c50                           | +    | 0.533   | 0.611+-0.020 | 0.596+-0.017 | 0.583+-0.011 | 1.00       | 1.00        | 1.00        |
| it_tr_ctr                            | +    | 0.510   | 0.476+-0.023 | 0.509+-0.018 | 0.508+-0.012 | 1.00       | 1.00        | 1.00        |
| it_tr_coec                           | +    | 0.507   | 0.472+-0.022 | 0.487+-0.018 | 0.497+-0.012 | 1.00       | 1.00        | 1.00        |
| seq_cv_invd20                        | +    | 0.552   | 0.500+-0.000 | 0.529+-0.007 | 0.545+-0.006 | 0.00       | 0.40        | 0.64        |
| seq_cv_pop                           | +    | 0.551   | 0.500+-0.000 | 0.610+-0.016 | 0.578+-0.009 | 0.00       | 1.00        | 0.84        |
| ctx_nd_hcent                         | +    | 0.606   | 0.500+-0.000 | 0.635+-0.017 | 0.623+-0.010 | 0.00       | 0.99        | 0.84        |
| ctx_nd_knn_f1                        | +    | 0.607   | 0.597+-0.014 | 0.639+-0.015 | 0.655+-0.009 | 0.50       | 0.94        | 0.92        |
| ctx_nd_knn_h                         | +    | 0.612   | 0.500+-0.000 | 0.636+-0.016 | 0.634+-0.009 | 0.00       | 0.99        | 0.84        |
| ctx_nd_knn_u                         | +    | 0.600   | 0.617+-0.016 | 0.618+-0.016 | 0.644+-0.010 | 0.62       | 0.99        | 0.97        |
| ctx_nd_max                           | +    | 0.573   | 0.617+-0.019 | 0.616+-0.016 | 0.611+-0.010 | 1.00       | 1.00        | 1.00        |
| ctx_nd_top5u                         | +    | 0.594   | 0.593+-0.017 | 0.621+-0.016 | 0.632+-0.010 | 0.62       | 0.99        | 0.97        |
| ctx_nd_ucent                         | +    | 0.565   | 0.518+-0.017 | 0.540+-0.016 | 0.563+-0.010 | 0.62       | 0.99        | 0.97        |
| ctx_hn_rrw                           | +    | 0.643   | 0.500+-0.000 | 0.680+-0.016 | 0.653+-0.009 | 0.00       | 0.99        | 0.84        |
| ctx_hs_k128                          | +    | 0.653   | 0.500+-0.000 | 0.676+-0.016 | 0.660+-0.009 | 0.00       | 0.99        | 0.84        |
| pre2_fe                              | +    | 0.663   | 0.586+-0.021 | 0.669+-0.017 | 0.673+-0.010 | 1.00       | 1.00        | 1.00        |
| pre2_new                             | +    | 0.661   | 0.522+-0.021 | 0.669+-0.017 | 0.659+-0.010 | 1.00       | 1.00        | 1.00        |
| pre2_hmax32                          | +    | 0.649   | 0.500+-0.000 | 0.669+-0.016 | 0.642+-0.009 | 0.00       | 1.00        | 0.84        |
| als_exp5                             | +    | 0.581   | 0.500+-0.000 | 0.583+-0.018 | 0.565+-0.010 | 0.00       | 0.96        | 0.83        |
| centrality                           | +    | 0.582   | 0.589+-0.022 | 0.628+-0.016 | 0.616+-0.010 | 1.00       | 1.00        | 1.00        |
| cos_u (user embedding)               | +    | 0.641   | 0.619+-0.015 | 0.667+-0.015 | 0.679+-0.009 | 0.62       | 0.99        | 0.97        |
| smax (history cosine)                | +    | 0.645   | 0.500+-0.000 | 0.673+-0.016 | 0.651+-0.009 | 0.00       | 0.99        | 0.84        |
| candidate-category share in the feed | +    | 0.579   | 0.595+-0.017 | 0.609+-0.012 | 0.615+-0.008 | 0.99       | 1.00        | 1.00        |

### candidates vs FINAL on ALL val: pop-light weight 0 -0.00085 | pop-light weight 0.1 -0.00007 | pop-light weight 0.3 +0.00015 | pop-light weight 0.4 +0.00048 | pop-light weight 0.6 -0.00010 | pop-light weight 0.8 -0.00058 | pop-light weight 1.0 -0.00108 | FT branch only -0.00081 | TO branch only -0.00058 | W8 file -0.00106 | W16 file -0.00113 | S16 file (d128 students) -0.00099 | G3 file (hard labels) -0.00239 | Q1a popularity-free file -0.00204 | Q2a pop1 file (hard labels) -0.00210

### largest gain of any stored family on a segment (vs FINAL): in-sample best and NESTED (20 x 5-fold by user; the family is chosen on 4 folds, read on the 5th)
| segment      | feeds | val % | test % | IN-SAMPLE best (not a witness)               | NESTED gain     | most frequent nested picks (of 100)                                                  | segment ceiling = max(0, nested + 2 SE) | x val share | x test share |
|--------------|-------|-------|--------|----------------------------------------------|-----------------|--------------------------------------------------------------------------------------|-----------------------------------------|-------------|--------------|
| H=0          | 194   | 1.35  | 0.86   | Q1a popularity-free file: +0.0006+-0.0054    | -0.0079+-0.0042 | Q1a popularity-free file x41; FINAL x29; G3 file (hard labels) x17                   | 0.0006                                  | 0.00001     | 0.00000      |
| H 1-9        | 285   | 1.98  | 1.90   | S16 file (d128 students): +0.0001+-0.0016    | -0.0048+-0.0018 | S16 file (d128 students) x42; pop-light weight 0.4 x18; FT branch only x17           | 0.0000                                  | 0.00000     | 0.00000      |
| H<=9         | 479   | 3.32  | 2.76   | TO branch only: -0.0010+-0.0025              | -0.0043+-0.0015 | FINAL x63; TO branch only x15; S16 file (d128 students) x9                           | 0.0000                                  | 0.00000     | 0.00000      |
| uid0         | 658   | 4.56  | 4.12   | FT branch only: +0.0005+-0.0013              | -0.0027+-0.0015 | G3 file (hard labels) x39; FT branch only x34; FINAL x12                             | 0.0003                                  | 0.00002     | 0.00001      |
| uid0 & H>=10 | 423   | 2.93  | 2.57   | Q2a pop1 file (hard labels): +0.0019+-0.0029 | -0.0031+-0.0021 | Q2a pop1 file (hard labels) x35; G3 file (hard labels) x27; pop-light weight 0.6 x16 | 0.0011                                  | 0.00003     | 0.00003      |

### every candidate on every segment (paired vs FINAL, cluster SE): 5 segments x 15 candidates = 75 looks
| segment      | candidate                   | delta vs FINAL  | z    |
|--------------|-----------------------------|-----------------|------|
| H=0          | pop-light weight 0          | -0.0007+-0.0031 | -0.2 |
| H=0          | pop-light weight 0.1        | -0.0024+-0.0021 | -1.1 |
| H=0          | pop-light weight 0.3        | -0.0023+-0.0022 | -1.0 |
| H=0          | pop-light weight 0.4        | -0.0047+-0.0032 | -1.5 |
| H=0          | pop-light weight 0.6        | -0.0067+-0.0035 | -1.9 |
| H=0          | pop-light weight 0.8        | -0.0090+-0.0052 | -1.7 |
| H=0          | pop-light weight 1.0        | -0.0055+-0.0082 | -0.7 |
| H=0          | FT branch only              | -0.0046+-0.0035 | -1.3 |
| H=0          | TO branch only              | -0.0008+-0.0041 | -0.2 |
| H=0          | W8 file                     | -0.0067+-0.0041 | -1.7 |
| H=0          | W16 file                    | -0.0024+-0.0028 | -0.9 |
| H=0          | S16 file (d128 students)    | -0.0028+-0.0050 | -0.6 |
| H=0          | G3 file (hard labels)       | -0.0009+-0.0060 | -0.1 |
| H=0          | Q1a popularity-free file    | +0.0006+-0.0054 | +0.1 |
| H=0          | Q2a pop1 file (hard labels) | -0.0039+-0.0057 | -0.7 |
| H 1-9        | pop-light weight 0          | -0.0054+-0.0021 | -2.5 |
| H 1-9        | pop-light weight 0.1        | -0.0017+-0.0009 | -1.8 |
| H 1-9        | pop-light weight 0.3        | -0.0017+-0.0014 | -1.2 |
| H 1-9        | pop-light weight 0.4        | -0.0007+-0.0024 | -0.3 |
| H 1-9        | pop-light weight 0.6        | -0.0045+-0.0035 | -1.3 |
| H 1-9        | pop-light weight 0.8        | -0.0066+-0.0048 | -1.4 |
| H 1-9        | pop-light weight 1.0        | -0.0086+-0.0052 | -1.6 |
| H 1-9        | FT branch only              | -0.0003+-0.0018 | -0.2 |
| H 1-9        | TO branch only              | -0.0011+-0.0032 | -0.4 |
| H 1-9        | W8 file                     | -0.0024+-0.0023 | -1.1 |
| H 1-9        | W16 file                    | -0.0017+-0.0019 | -0.9 |
| H 1-9        | S16 file (d128 students)    | +0.0001+-0.0016 | +0.1 |
| H 1-9        | G3 file (hard labels)       | -0.0034+-0.0042 | -0.8 |
| H 1-9        | Q1a popularity-free file    | -0.0043+-0.0046 | -0.9 |
| H 1-9        | Q2a pop1 file (hard labels) | -0.0036+-0.0042 | -0.9 |
| H<=9         | pop-light weight 0          | -0.0035+-0.0018 | -1.9 |
| H<=9         | pop-light weight 0.1        | -0.0020+-0.0010 | -1.9 |
| H<=9         | pop-light weight 0.3        | -0.0020+-0.0012 | -1.6 |
| H<=9         | pop-light weight 0.4        | -0.0023+-0.0019 | -1.2 |
| H<=9         | pop-light weight 0.6        | -0.0054+-0.0025 | -2.2 |
| H<=9         | pop-light weight 0.8        | -0.0076+-0.0036 | -2.1 |
| H<=9         | pop-light weight 1.0        | -0.0073+-0.0045 | -1.6 |
| H<=9         | FT branch only              | -0.0021+-0.0018 | -1.2 |
| H<=9         | TO branch only              | -0.0010+-0.0025 | -0.4 |
| H<=9         | W8 file                     | -0.0041+-0.0021 | -1.9 |
| H<=9         | W16 file                    | -0.0020+-0.0016 | -1.3 |
| H<=9         | S16 file (d128 students)    | -0.0011+-0.0022 | -0.5 |
| H<=9         | G3 file (hard labels)       | -0.0024+-0.0035 | -0.7 |
| H<=9         | Q1a popularity-free file    | -0.0023+-0.0035 | -0.7 |
| H<=9         | Q2a pop1 file (hard labels) | -0.0038+-0.0034 | -1.1 |
| uid0         | pop-light weight 0          | +0.0000+-0.0012 | +0.0 |
| uid0         | pop-light weight 0.1        | -0.0003+-0.0007 | -0.5 |
| uid0         | pop-light weight 0.3        | -0.0005+-0.0011 | -0.5 |
| uid0         | pop-light weight 0.4        | -0.0010+-0.0016 | -0.6 |
| uid0         | pop-light weight 0.6        | -0.0023+-0.0022 | -1.0 |
| uid0         | pop-light weight 0.8        | -0.0039+-0.0028 | -1.4 |
| uid0         | pop-light weight 1.0        | -0.0044+-0.0035 | -1.3 |
| uid0         | FT branch only              | +0.0005+-0.0013 | +0.4 |
| uid0         | TO branch only              | -0.0021+-0.0020 | -1.1 |
| uid0         | W8 file                     | -0.0018+-0.0020 | -0.9 |
| uid0         | W16 file                    | -0.0010+-0.0015 | -0.6 |
| uid0         | S16 file (d128 students)    | -0.0009+-0.0021 | -0.4 |
| uid0         | G3 file (hard labels)       | +0.0004+-0.0028 | +0.1 |
| uid0         | Q1a popularity-free file    | -0.0004+-0.0025 | -0.2 |
| uid0         | Q2a pop1 file (hard labels) | -0.0004+-0.0025 | -0.2 |
| uid0 & H>=10 | pop-light weight 0          | +0.0011+-0.0016 | +0.7 |
| uid0 & H>=10 | pop-light weight 0.1        | +0.0000+-0.0006 | +0.1 |
| uid0 & H>=10 | pop-light weight 0.3        | +0.0008+-0.0010 | +0.8 |
| uid0 & H>=10 | pop-light weight 0.4        | +0.0011+-0.0014 | +0.8 |
| uid0 & H>=10 | pop-light weight 0.6        | +0.0010+-0.0023 | +0.4 |
| uid0 & H>=10 | pop-light weight 0.8        | -0.0019+-0.0030 | -0.6 |
| uid0 & H>=10 | pop-light weight 1.0        | -0.0024+-0.0038 | -0.6 |
| uid0 & H>=10 | FT branch only              | +0.0006+-0.0013 | +0.5 |
| uid0 & H>=10 | TO branch only              | +0.0000+-0.0019 | +0.0 |
| uid0 & H>=10 | W8 file                     | -0.0003+-0.0026 | -0.1 |
| uid0 & H>=10 | W16 file                    | -0.0003+-0.0021 | -0.2 |
| uid0 & H>=10 | S16 file (d128 students)    | +0.0002+-0.0026 | +0.1 |
| uid0 & H>=10 | G3 file (hard labels)       | +0.0017+-0.0032 | +0.6 |
| uid0 & H>=10 | Q1a popularity-free file    | +0.0002+-0.0029 | +0.1 |
| uid0 & H>=10 | Q2a pop1 file (hard labels) | +0.0019+-0.0029 | +0.7 |
cells with |z| >= 2: 3 of 75 (expected by chance 3.4); max z +0.80, min z -2.50
generous ceiling H=0: NDCG 0.3784 vs NEAR 0.4060 (D -0.0277+-0.0178); if lifted to NEAR: val share x gap = 0.00037, test share x gap = 0.00024; skill 0.1518
generous ceiling H 1-9: NDCG 0.4431 vs NEAR 0.4060 (D +0.0371+-0.0176); if lifted to NEAR: val share x gap = 0.00000, test share x gap = 0.00000; skill 0.2356
generous ceiling uid0: NDCG 0.4093 vs NEAR 0.4060 (D +0.0032+-0.0103); if lifted to NEAR: val share x gap = 0.00000, test share x gap = 0.00000; skill 0.1916
```

### s6 audit by H bucket
```
| arm (pool of stored train-only val predictions vs same-seed control pool)                                 | members | ALL                        | H=256                      | H=256 minus rest  | flag | H<=9                       | H<=9 minus rest   | flag | uid0                       | uid0 minus rest   | flag |
|-----------------------------------------------------------------------------------------------------------|---------|----------------------------|----------------------------|-------------------|------|----------------------------|-------------------|------|----------------------------|-------------------|------|
| retrieval depth 32 (2 ep)                                                                                 | 2       | +0.00020+-0.00058 (z +0.3) | +0.00163+-0.00105 (z +1.5) | +0.00202 (z +1.6) | -    | -0.00655+-0.00369 (z -1.8) | -0.00698 (z -1.9) | -    | -0.00234+-0.00273 (z -0.9) | -0.00266 (z -1.0) | -    |
| retrieval 32 + unique slots + type counts (2 ep)                                                          | 2       | +0.00036+-0.00064 (z +0.6) | +0.00119+-0.00117 (z +1.0) | +0.00117 (z +0.8) | -    | -0.00112+-0.00334 (z -0.3) | -0.00153 (z -0.5) | -    | +0.00083+-0.00267 (z +0.3) | +0.00049 (z +0.2) | -    |
| second same-category retrieval (2 ep)                                                                     | 2       | +0.00017+-0.00051 (z +0.3) | +0.00115+-0.00085 (z +1.4) | +0.00138 (z +1.3) | -    | -0.00145+-0.00342 (z -0.4) | -0.00167 (z -0.5) | -    | -0.00076+-0.00236 (z -0.3) | -0.00097 (z -0.4) | -    |
| INFO: retrieval REMOVED (2 ep)                                                                            | 2       | -0.00295+-0.00080 (z -3.7) | -0.00375+-0.00150 (z -2.5) | -0.00113 (z -0.6) | -    | -0.00200+-0.00434 (z -0.5) | +0.00098 (z +0.2) | -    | +0.00096+-0.00349 (z +0.3) | +0.00410 (z +1.1) | -    |
| INFO: cosine sim_bias REMOVED (2 ep)                                                                      | 2       | +0.00037+-0.00054 (z +0.7) | +0.00116+-0.00097 (z +1.2) | +0.00111 (z +0.9) | -    | -0.00029+-0.00341 (z -0.1) | -0.00068 (z -0.2) | -    | -0.00130+-0.00273 (z -0.5) | -0.00175 (z -0.6) | -    |
| 3 blocks (2 ep)                                                                                           | 2       | +0.00021+-0.00057 (z +0.4) | +0.00028+-0.00097 (z +0.3) | +0.00010 (z +0.1) | -    | -0.00348+-0.00386 (z -0.9) | -0.00382 (z -1.0) | -    | +0.00060+-0.00253 (z +0.2) | +0.00040 (z +0.2) | -    |
| frozen encoder hidden-state K/V branch (2 ep)                                                             | 2       | +0.00027+-0.00056 (z +0.5) | +0.00201+-0.00101 (z +2.0) | +0.00244 (z +2.0) | -    | +0.00039+-0.00320 (z +0.1) | +0.00013 (z +0.0) | -    | -0.00044+-0.00267 (z -0.2) | -0.00074 (z -0.3) | -    |
| INFO: pre2 inputs REMOVED (2 ep)                                                                          | 2       | -0.00151+-0.00065 (z -2.3) | +0.00097+-0.00115 (z +0.8) | +0.00349 (z +2.5) | -    | -0.00366+-0.00434 (z -0.8) | -0.00222 (z -0.5) | -    | -0.00371+-0.00286 (z -1.3) | -0.00230 (z -0.8) | -    |
| INFO: popularity inputs REMOVED (2 ep)                                                                    | 2       | -0.00256+-0.00101 (z -2.5) | -0.00185+-0.00184 (z -1.0) | +0.00100 (z +0.5) | -    | -0.01123+-0.00628 (z -1.8) | -0.00897 (z -1.4) | -    | -0.01063+-0.00479 (z -2.2) | -0.00846 (z -1.7) | -    |
| INFO: pop1 view (2 ep)                                                                                    | 2       | -0.00125+-0.00099 (z -1.3) | -0.00193+-0.00180 (z -1.1) | -0.00095 (z -0.4) | -    | +0.00313+-0.00645 (z +0.5) | +0.00454 (z +0.7) | -    | -0.00073+-0.00447 (z -0.2) | +0.00054 (z +0.1) | -    |
| retrieval depth 32 (10 ep, hard labels)                                                                   | 2       | +0.00081+-0.00085 (z +0.9) | +0.00052+-0.00157 (z +0.3) | -0.00040 (z -0.2) | -    | -0.00022+-0.00474 (z -0.0) | -0.00106 (z -0.2) | -    | +0.00065+-0.00348 (z +0.2) | -0.00017 (z -0.0) | -    |
| retrieval 32 + unique + type counts (10 ep)                                                               | 2       | -0.00005+-0.00083 (z -0.1) | +0.00075+-0.00160 (z +0.5) | +0.00113 (z +0.6) | -    | -0.00246+-0.00439 (z -0.6) | -0.00249 (z -0.6) | -    | -0.00250+-0.00355 (z -0.7) | -0.00257 (z -0.7) | -    |
| 3 blocks (10 ep)                                                                                          | 2       | +0.00121+-0.00083 (z +1.5) | -0.00021+-0.00149 (z -0.1) | -0.00200 (z -1.1) | -    | -0.00021+-0.00476 (z -0.0) | -0.00147 (z -0.3) | -    | -0.00280+-0.00345 (z -0.8) | -0.00420 (z -1.2) | -    |
| 2 history layers under KD (d192)                                                                          | 2       | -0.00031+-0.00068 (z -0.5) | -0.00013+-0.00127 (z -0.1) | +0.00026 (z +0.2) | -    | +0.00602+-0.00377 (z +1.6) | +0.00655 (z +1.7) | -    | +0.00019+-0.00238 (z +0.1) | +0.00053 (z +0.2) | -    |
| hashed item-ID table under KD                                                                             | 2       | -0.00509+-0.00094 (z -5.4) | -0.00361+-0.00174 (z -2.1) | +0.00209 (z +1.0) | -    | -0.00212+-0.00543 (z -0.4) | +0.00307 (z +0.6) | -    | -0.00105+-0.00340 (z -0.3) | +0.00423 (z +1.2) | -    |
| FFN x4 under KD                                                                                           | 2       | -0.00088+-0.00071 (z -1.2) | -0.00217+-0.00128 (z -1.7) | -0.00181 (z -1.2) | -    | +0.00303+-0.00402 (z +0.8) | +0.00404 (z +1.0) | -    | +0.00529+-0.00317 (z +1.7) | +0.00646 (z +2.0) | -    |
| history-drop 0.3 function matching under KD                                                               | 4       | -0.00098+-0.00058 (z -1.7) | -0.00178+-0.00110 (z -1.6) | -0.00112 (z -0.9) | -    | -0.00169+-0.00320 (z -0.5) | -0.00073 (z -0.2) | -    | +0.00078+-0.00192 (z +0.4) | +0.00185 (z +0.9) | -    |
| GRU history encoder under KD                                                                              | 4       | -0.00000+-0.00059 (z -0.0) | +0.00216+-0.00109 (z +2.0) | +0.00305 (z +2.3) | -    | -0.00196+-0.00290 (z -0.7) | -0.00202 (z -0.7) | -    | -0.00004+-0.00188 (z -0.0) | -0.00003 (z -0.0) | -    |
| width 224 under KD                                                                                        | 4       | +0.00026+-0.00058 (z +0.5) | +0.00041+-0.00109 (z +0.4) | +0.00020 (z +0.2) | -    | -0.00249+-0.00316 (z -0.8) | -0.00285 (z -0.9) | -    | +0.00315+-0.00213 (z +1.5) | +0.00303 (z +1.4) | -    |
| 3 blocks under KD (teacher 25)                                                                            | 2       | -0.00095+-0.00071 (z -1.3) | -0.00104+-0.00128 (z -0.8) | -0.00012 (z -0.1) | -    | -0.00032+-0.00416 (z -0.1) | +0.00066 (z +0.2) | -    | +0.00021+-0.00330 (z +0.1) | +0.00122 (z +0.4) | -    |
| width 256 under KD (teacher 25)                                                                           | 2       | -0.00108+-0.00072 (z -1.5) | -0.00012+-0.00129 (z -0.1) | +0.00135 (z +0.9) | -    | -0.00442+-0.00408 (z -1.1) | -0.00346 (z -0.8) | -    | -0.00096+-0.00309 (z -0.3) | +0.00012 (z +0.0) | -    |
| r7 hl2: second history layer (2 ep)                                                                       | 2       | -0.00038+-0.00055 (z -0.7) | -0.00009+-0.00095 (z -0.1) | +0.00040 (z +0.3) | -    | +0.00111+-0.00349 (z +0.3) | +0.00154 (z +0.4) | -    | +0.00166+-0.00237 (z +0.7) | +0.00213 (z +0.9) | -    |
| r7 hc4: 4 hyper-connection streams (2 ep)                                                                 | 2       | +0.00000+-0.00054 (z +0.0) | +0.00017+-0.00089 (z +0.2) | +0.00024 (z +0.2) | -    | +0.00556+-0.00350 (z +1.6) | +0.00575 (z +1.6) | -    | +0.00098+-0.00248 (z +0.4) | +0.00103 (z +0.4) | -    |
| r6 typed cross-attention (target actions only in the last block) [seed 23 only (no stored s24 control)]   | 1       | +0.00069+-0.00058 (z +1.2) | +0.00091+-0.00094 (z +1.0) | +0.00031 (z +0.3) | -    | +0.00470+-0.00364 (z +1.3) | +0.00415 (z +1.1) | -    | +0.00196+-0.00237 (z +0.8) | +0.00133 (z +0.5) | -    |
| r6 a4u: unique retrieval slots + repeat count (2 ep) [seed 23 only (no stored s24 control)]               | 1       | +0.00120+-0.00061 (z +2.0) | +0.00297+-0.00112 (z +2.7) | +0.00249 (z +1.9) | -    | +0.00107+-0.00364 (z +0.3) | -0.00014 (z -0.0) | -    | -0.00159+-0.00231 (z -0.7) | -0.00293 (z -1.2) | -    |
| r6 a4w: DIN product term in retrieval values (2 ep) [seed 23 only (no stored s24 control)]                | 1       | +0.00021+-0.00073 (z +0.3) | +0.00109+-0.00129 (z +0.8) | +0.00123 (z +0.8) | -    | +0.00039+-0.00440 (z +0.1) | +0.00018 (z +0.0) | -    | -0.00217+-0.00312 (z -0.7) | -0.00250 (z -0.8) | -    |
| r6 cset: candidate-set retrieval (2 ep) [seed 23 only (no stored s24 control)]                            | 1       | +0.00023+-0.00077 (z +0.3) | +0.00306+-0.00129 (z +2.4) | +0.00399 (z +2.5) | -    | +0.00609+-0.00525 (z +1.2) | +0.00606 (z +1.1) | -    | -0.00202+-0.00346 (z -0.6) | -0.00235 (z -0.7) | -    |
| r6 a4u at 10 ep (6 seeds vs F7 same seeds)                                                                | 6       | +0.00058+-0.00061 (z +0.9) | +0.00274+-0.00119 (z +2.3) | +0.00304 (z +2.2) | -    | -0.00067+-0.00322 (z -0.2) | -0.00129 (z -0.4) | -    | +0.00041+-0.00251 (z +0.2) | -0.00018 (z -0.1) | -    |
| r6 cset at 10 ep (6 seeds vs F7 same seeds)                                                               | 6       | -0.00010+-0.00066 (z -0.2) | +0.00036+-0.00117 (z +0.3) | +0.00066 (z +0.5) | -    | -0.00020+-0.00392 (z -0.1) | -0.00010 (z -0.0) | -    | +0.00037+-0.00295 (z +0.1) | +0.00050 (z +0.2) | -    |
| hcap: per-type history caps in the cooldown (history parity)                                              | 4       | -0.00048+-0.00035 (z -1.4) | -0.00013+-0.00062 (z -0.2) | +0.00049 (z +0.7) | -    | +0.00106+-0.00166 (z +0.6) | +0.00160 (z +0.9) | -    | +0.00115+-0.00144 (z +0.8) | +0.00170 (z +1.1) | -    |
| rec33: cooldown on the last 3 train days                                                                  | 4       | -0.00187+-0.00069 (z -2.7) | -0.00358+-0.00132 (z -2.7) | -0.00241 (z -1.6) | -    | +0.00100+-0.00395 (z +0.3) | +0.00296 (z +0.7) | -    | +0.00078+-0.00305 (z +0.3) | +0.00278 (z +0.9) | -    |
| aug: candidate subset + history drop 0.15 on cooldown pool batches                                        | 4       | -0.00117+-0.00062 (z -1.9) | -0.00214+-0.00116 (z -1.9) | -0.00136 (z -1.0) | -    | +0.00040+-0.00360 (z +0.1) | +0.00162 (z +0.4) | -    | +0.00180+-0.00283 (z +0.6) | +0.00312 (z +1.1) | -    |
| HS1: history restricted to feed-surface events (2 ep) [control = abl_base same seeds]                     | 2       | -0.00217+-0.00084 (z -2.6) | -0.00229+-0.00153 (z -1.5) | -0.00016 (z -0.1) | -    | -0.00648+-0.00464 (z -1.4) | -0.00445 (z -0.9) | -    | -0.00616+-0.00360 (z -1.7) | -0.00417 (z -1.1) | -    |
| HS1b: history restricted to target actions (2 ep) [control = abl_base same seeds]                         | 2       | -0.00449+-0.00093 (z -4.9) | -0.00504+-0.00166 (z -3.0) | -0.00077 (z -0.4) | -    | -0.00582+-0.00455 (z -1.3) | -0.00138 (z -0.3) | -    | -0.00362+-0.00429 (z -0.8) | +0.00091 (z +0.2) | -    |
| GRU history encoder family v23 (6 seeds) vs F7 v17 (6 seeds; different seeds) [families, not seed-paired] | 6       | -0.00023+-0.00067 (z -0.3) | +0.00093+-0.00125 (z +0.7) | +0.00163 (z +1.1) | -    | -0.00130+-0.00385 (z -0.3) | -0.00111 (z -0.3) | -    | -0.00302+-0.00298 (z -1.0) | -0.00293 (z -1.0) | -    |

non-INFO looks: 90 segment cells. |segment z| >= 2: 6 (expected by chance 4.1); segment z >= +2: 3; z >= 3.3: 0; |interaction z| >= 2: 4 (expected 4.1)
   largest positive segment z: +2.66 (interaction z +1.87)  r6 a4u: unique retrieval slots + repeat count (2 ep) @ H=256
   largest positive segment z: +2.37 (interaction z +2.49)  r6 cset: candidate-set retrieval (2 ep) @ H=256
   largest positive segment z: +2.30 (interaction z +2.19)  r6 a4u at 10 ep (6 seeds vs F7 same seeds) @ H=256
   largest positive segment z: +1.99 (interaction z +2.02)  frozen encoder hidden-state K/V branch (2 ep) @ H=256
   largest positive segment z: +1.99 (interaction z +2.35)  GRU history encoder under KD @ H=256
   largest positive segment z: +1.67 (interaction z +1.99)  FFN x4 under KD @ uid0
   largest negative segment z: -3.04 (interaction z -0.39)  HS1b: history restricted to target actions (2 ep) @ H=256
   largest negative segment z: -2.72 (interaction z -1.56)  rec33: cooldown on the last 3 train days @ H=256
   largest negative segment z: -2.07 (interaction z +1.01)  hashed item-ID table under KD @ H=256
   largest negative segment z: -1.85 (interaction z -0.99)  aug: candidate subset + history drop 0.15 on cooldown pool batches @ H=256

### delta by H bucket for the arms named in the brief (and the worth of components, INFO rows = component REMOVED, so a negative number is the component's worth)
| arm                                                   | H=0             | 1-4             | 5-9             | 10-19           | 20-49           | 50-99           | 100-255         | H=256           |
|-------------------------------------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| retrieval depth 32 (2 ep)                             | -0.0062+-0.0060 | -0.0026+-0.0040 | -0.0114+-0.0088 | -0.0005+-0.0031 | +0.0016+-0.0023 | -0.0037+-0.0020 | +0.0005+-0.0008 | +0.0016+-0.0011 |
| INFO: retrieval REMOVED (2 ep)                        | -0.0044+-0.0070 | +0.0021+-0.0056 | -0.0032+-0.0099 | +0.0022+-0.0059 | +0.0004+-0.0030 | -0.0029+-0.0025 | -0.0031+-0.0011 | -0.0038+-0.0015 |
| INFO: cosine sim_bias REMOVED (2 ep)                  | +0.0029+-0.0059 | +0.0032+-0.0051 | -0.0088+-0.0065 | +0.0023+-0.0044 | +0.0042+-0.0020 | -0.0025+-0.0016 | +0.0001+-0.0008 | +0.0012+-0.0010 |
| INFO: pre2 inputs REMOVED (2 ep)                      | +0.0007+-0.0061 | +0.0010+-0.0068 | -0.0151+-0.0102 | -0.0060+-0.0031 | -0.0005+-0.0028 | -0.0032+-0.0020 | -0.0024+-0.0009 | +0.0010+-0.0011 |
| INFO: popularity inputs REMOVED (2 ep)                | -0.0131+-0.0108 | -0.0080+-0.0088 | -0.0121+-0.0126 | +0.0014+-0.0084 | +0.0045+-0.0039 | -0.0010+-0.0032 | -0.0036+-0.0014 | -0.0018+-0.0018 |
| retrieval depth 32 (10 ep, hard labels)               | +0.0082+-0.0064 | -0.0104+-0.0083 | -0.0010+-0.0106 | -0.0042+-0.0061 | -0.0038+-0.0034 | +0.0060+-0.0026 | +0.0006+-0.0012 | +0.0005+-0.0016 |
| 2 history layers under KD (d192)                      | +0.0004+-0.0056 | +0.0042+-0.0048 | +0.0160+-0.0092 | -0.0076+-0.0050 | +0.0036+-0.0028 | -0.0021+-0.0021 | -0.0006+-0.0010 | -0.0001+-0.0013 |
| history-drop 0.3 function matching under KD           | +0.0013+-0.0058 | -0.0033+-0.0047 | -0.0042+-0.0057 | -0.0083+-0.0036 | +0.0014+-0.0025 | -0.0012+-0.0017 | -0.0004+-0.0008 | -0.0018+-0.0011 |
| GRU history encoder under KD                          | +0.0012+-0.0050 | -0.0043+-0.0036 | -0.0039+-0.0062 | -0.0026+-0.0030 | +0.0011+-0.0022 | -0.0001+-0.0018 | -0.0011+-0.0009 | +0.0022+-0.0011 |
| r7 hl2: second history layer (2 ep)                   | +0.0110+-0.0062 | -0.0064+-0.0053 | -0.0047+-0.0061 | -0.0011+-0.0048 | -0.0012+-0.0017 | -0.0011+-0.0017 | -0.0004+-0.0008 | -0.0001+-0.0009 |
| HS1: history restricted to feed-surface events (2 ep) | -0.0038+-0.0059 | -0.0021+-0.0090 | -0.0151+-0.0099 | -0.0024+-0.0054 | -0.0034+-0.0033 | -0.0083+-0.0025 | -0.0003+-0.0012 | -0.0023+-0.0015 |
```

### s7 a4u follow-up
```
| read                                                                   | ALL               | H=256                       | NEAR (100-255)              | H<100                       | H=256 minus rest   |
|------------------------------------------------------------------------|-------------------|-----------------------------|-----------------------------|-----------------------------|--------------------|
| F1 a4u + m02 re-cool vs m02 re-cool, train-only, roots 101-106 (6)     | +0.00017+-0.00065 | +0.00115+-0.00121 (z +0.95) | -0.00005+-0.00093 (z -0.06) | -0.00066+-0.00138 (z -0.47) | +0.00139 (z +0.97) |
| F1 a4u + m02 re-cool vs m02 re-cool, fine-tuned OOF, roots 101-104 (4) | +0.00068+-0.00069 | +0.00146+-0.00134 (z +1.09) | +0.00057+-0.00099 (z +0.57) | -0.00010+-0.00132 (z -0.07) | +0.00109 (z +0.70) |
| (seen) a4u 10 ep vs F7, 6 seeds, train-only                            | +0.00058+-0.00061 | +0.00274+-0.00119 (z +2.30) | -0.00039+-0.00086 (z -0.46) | -0.00009+-0.00128 (z -0.07) | +0.00304 (z +2.19) |
a4u 10 ep member-level deltas at H=256 by seed: +0.0010 +0.0015 +0.0019 +0.0015 +0.0013 +0.0005 | mean +0.0013, seeds > 0: 6/6

### F2 dose-response by the distinct-item share of the stored history (H >= 100)
| pair                           | feeds                         | n    | mean distinct share | delta                       |
|--------------------------------|-------------------------------|------|---------------------|-----------------------------|
| a4u 10 ep (6 seeds)            | bottom tercile (most repeats) | 3831 | 0.563               | +0.00176+-0.00124           |
| a4u 10 ep (6 seeds)            | middle                        | 3752 | 0.682               | -0.00014+-0.00117           |
| a4u 10 ep (6 seeds)            | top tercile (fewest repeats)  | 3758 | 0.812               | +0.00065+-0.00122           |
| a4u 10 ep (6 seeds)            | bottom minus top              |      |                     | +0.00111+-0.00174 (z +0.64) |
| a4u 10 ep (6 seeds)            | H=256 & bottom tercile        | 2033 | 0.570               | +0.00277+-0.00165           |
| a4u 10 ep (6 seeds)            | H=256 & top tercile           | 644  | 0.786               | +0.00596+-0.00337           |
| a4u + m02 re-cool (train-only) | bottom tercile (most repeats) | 3831 | 0.563               | -0.00078+-0.00136           |
| a4u + m02 re-cool (train-only) | middle                        | 3752 | 0.682               | +0.00106+-0.00130           |
| a4u + m02 re-cool (train-only) | top tercile (fewest repeats)  | 3758 | 0.812               | +0.00092+-0.00117           |
| a4u + m02 re-cool (train-only) | bottom minus top              |      |                     | -0.00170+-0.00179 (z -0.95) |
| a4u + m02 re-cool (train-only) | H=256 & bottom tercile        | 2033 | 0.570               | -0.00016+-0.00172           |
| a4u + m02 re-cool (train-only) | H=256 & top tercile           | 644  | 0.786               | +0.00397+-0.00252           |

### F3 the same deltas split by cold / warm positives
| pair                           | segment | all positives cold         | all positives warm         | mixed                      |
|--------------------------------|---------|----------------------------|----------------------------|----------------------------|
| a4u 10 ep (6 seeds)            | H=256   | +0.00410+-0.00251 (n 1023) | +0.00228+-0.00162 (n 2513) | +0.00238+-0.00200 (n 640)  |
| a4u 10 ep (6 seeds)            | NEAR    | +0.00217+-0.00172 (n 1649) | -0.00122+-0.00115 (n 4548) | -0.00090+-0.00158 (n 968)  |
| a4u 10 ep (6 seeds)            | ALL     | +0.00271+-0.00122 (n 3422) | -0.00016+-0.00082 (n 9118) | +0.00027+-0.00114 (n 1883) |
| a4u + m02 re-cool (train-only) | H=256   | -0.00348+-0.00233 (n 1023) | +0.00354+-0.00169 (n 2513) | -0.00083+-0.00214 (n 640)  |
| a4u + m02 re-cool (train-only) | NEAR    | -0.00349+-0.00191 (n 1649) | +0.00150+-0.00124 (n 4548) | -0.00147+-0.00171 (n 968)  |
| a4u + m02 re-cool (train-only) | ALL     | -0.00292+-0.00128 (n 3422) | +0.00148+-0.00087 (n 9118) | -0.00055+-0.00124 (n 1883) |
share of feeds with all-cold positives: H=256 0.245, NEAR 0.230, REST 0.234
```

### s10 distinct items in the retrieval slots
```
| H       | valid slots of 16 | distinct items in the 16 slots | distinct share (candidates) | distinct share (positives) | share of candidates with <= 8 distinct | distinct items in top-32 | repeat share of the whole stored history |
|---------|-------------------|--------------------------------|-----------------------------|----------------------------|----------------------------------------|--------------------------|------------------------------------------|
| 10-19   | 13.8              | 9.37                           | 0.682                       | 0.690                      | 0.447                                  | 9.77                     | 0.317                                    |
| 20-49   | 16.0              | 11.39                          | 0.712                       | 0.712                      | 0.253                                  | 20.98                    | 0.299                                    |
| 50-99   | 16.0              | 11.60                          | 0.725                       | 0.732                      | 0.195                                  | 22.79                    | 0.303                                    |
| 100-255 | 16.0              | 12.12                          | 0.757                       | 0.790                      | 0.138                                  | 23.71                    | 0.290                                    |
| H=256   | 16.0              | 11.15                          | 0.697                       | 0.716                      | 0.203                                  | 21.60                    | 0.358                                    |
```

### DECLARATION.txt
```
DECLARATION  nn_x/w12/seg/diag  (analyst of user-information segments; CPU only; no GPU, no network, no submission)
Every block is stamped with 'date -u' BEFORE the numbers it judges are computed. Nothing in this file is edited after
the fact; amendments are appended with their own stamp.

=== BLOCK 0  2026-11-15T16:40:12Z  (written before ANY number of this study was computed; the only numbers seen so far are the
    lead's first table in the task text and the status / coverage files) ===
Conventions
 - Score analysed: FINAL = 0.5 z(preds/ens_FD16P20kd_ft_val.npy) + 0.5 z(preds/ens_FD16P20kd_to_val.npy), per-feed z
   (src/stack_val.py feed_rank_z[:,1]); metric src/metrics.py ndcg_per_feed (feeds without a positive count 1).
 - SE: cluster-robust by user_id (sum of per-user residual sums; user_id 0 = pooled bucket, each of its feeds is its
   own cluster). Paired deltas are per feed, then the same cluster SE.
 - H buckets: 0 | 1-4 | 5-9 | 10-19 | 20-49 | 50-99 | 100-255 | 256 (H = stored history length, capped at 256 by the data).
 - "REST" = all feeds with H < 256; "NEAR" = H in 100..255 (adjacent bucket).
 - Marks: [M] measured here, [D] derived arithmetic from [M], [I] judgement.
 - Val labels are used (a) descriptively with no choice made from them, or (b) inside 5-fold GroupKFold-by-user
   procedures wherever something is FITTED or CHOSEN. No later snapshots of a user, no labels as features.

D0 reproduction: rebuilt FINAL must read 0.40271 +- 0.00001; bucket shares and NDCG must match the lead's table to
   +-0.001 NDCG / +-0.1 pp. If not, the mismatch is reported and my numbers are used.

A1 (is the H = 256 deficit explained by label-free feed composition?)
 D_raw(ref) = mean NDCG(H=256) - mean NDCG(ref), ref in {REST, NEAR}.
 Covariates, label-free: M (and log M), day, hour, user_id 0 flag, cold share of the candidates (it_hist_all_luo == 0,
   the project's 'cold' definition), share of candidates without an embedding, number of candidate categories, top
   candidate-category share, share of candidates whose category occurs in the history; history composition computed
   on the NEWEST min(H, 64) events so that it has common support across H (event-type shares, event-source top share,
   number of distinct categories, top-category share, share of events without embedding, share of distinct items);
   random-order posterior of the feed if a stored label-free one exists for val. NOT available in the data: event
   timestamps (so "recency of the last event" in time cannot be computed; stated, not approximated) and a traffic
   domain for val feeds (val is shuffled / one inference domain) -> reported as not identifiable.
 Method: LightGBM regression of per-feed NDCG on the covariates, fitted on REST feeds only, 5-fold GroupKFold by user
   over ALL feeds (a feed is always predicted by a model that saw no feed of its user); D_adj = mean over H=256 feeds
   of (NDCG - prediction) minus the same mean over REST feeds (out-of-fold; should be about 0). Cross-check: OLS with
   the same covariates + H=256 indicator; exact-cell matching on (M decile x cold-share tercile) as a third read.
 The number of positives is NOT label-free. It is handled separately: (i) D_adj+ = the same procedure with n_pos added;
   (ii) per-feed AUC (mean percentile rank of the positives; 0.5 for a random order whatever M and n_pos);
   (iii) NDCG minus the exact expectation of NDCG under a random order given (M, n_pos) ("skill").
 Reading rule: "composition" if D_adj+ >= -1/3 |D_raw(REST)| or D_adj+ >= -2 SE; "deficit candidate" if
   D_adj+ <= -2/3 |D_raw(REST)| and <= -2 SE; otherwise "partly". A residual is NOT yet a model deficit: A2 decides.

A2 (difficulty proxies; same bucket table for other rankers)
 Rankers: FINAL; FD16 base pool alone; popularity-light pool alone (4 wide full-data pop1 students); the distillation
   teacher's hard-label members on val (the 11 teacher members if their val predictions are stored, else the stored
   hard-label pools G3 / w5_e10_base); one single hard-label member; trivial rankers: popularity only (it_hist_all_luo),
   exposure only (xp_imp_c50), max cosine to the history (computed here), cosine to the user embedding, the
   pre-trained encoder score (pre2_fe), f-columns if any is a known one-column heuristic.
 For each: NDCG, per-feed AUC and skill by H bucket; D(H=256 - NEAR) and D(H=256 - REST).
 Reading rule: the gap is TASK DIFFICULTY (intrinsic) if all three hold:
   (a) history-free rankers (popularity, exposure) show D(H=256 - REST) in NDCG of the same sign and >= 50% of FINAL's;
   (b) FINAL's per-feed AUC at H=256 is not below NEAR by more than 2 SE;
   (c) FINAL's lift over the popularity-only ranker (paired per feed, in AUC and in skill) at H=256 is not smaller
       than at NEAR by more than 2 SE.
 MODEL-DEFICIT signature: (c) fails, i.e. the model's gain over history-free ranking shrinks at the cap while the
   history-free rankers do not get worse.

A3 (where the positives of heavy users sit; descriptive, no choice)
 For positives (negatives as contrast) by H bucket: rank / percentile of the positive under FINAL (top-1, top-5,
   top-10 shares; median percentile); share of cold positives; max cosine to the history, index (newest-first) of the
   most similar event: share within the newest 16 / 64 / 256; within-feed AUC of smax restricted to the newest
   8 / 16 / 32 / 64 / 128 / 256 events; within-feed AUC of mean-of-top-k cosines for k = 1 / 3 / 16 / 32 / 64 / all.
 NOTE (code fact, nn_x/w5/model.py HistRetrieval): retrieval takes the top-16 events BY COSINE over the whole stored
   history, not the newest 16, and the cross-attention with the cosine bias sees all stored events. So "position
   beyond 16" is not out of reach; what depth 16 cannot reach is events ranked 17+ by cosine, and what nothing can
   reach is history older than the 256-event data cap.
 Reading rules: "relevant history is old" if at H=256 >= 40% of positives have their most similar event at index >= 64
   AND the smax-only ranker restricted to the newest 64 events loses >= 0.005 NDCG at H=256 against smax over all.
   "retrieval depth binds" if at H=256 the within-feed AUC of mean-top-32 exceeds mean-top-16 by >= +0.005 (cluster
   2 SE clear of 0) while it does not at H 20-99.

B (small segments: H = 0; H 1-9; user_id 0)
 Table: FINAL vs FD16 alone, popularity-light pool alone, popularity only, exposure only, cos_u, pre2_fe; paired
   deltas with cluster SE; within-feed AUC of each raw input on the segment vs all feeds (what is still informative).
 "Largest gain any stored family shows": candidate set fixed NOW = {FD16 base, popularity-light pool, FT branch
   only, TO branch only, W8 file, W16 file, S16 file (d128 students), G3 file (hard labels), Q1a popularity-free
   file, Q2a pop1 file, P-weight w in {0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0} between FD16 and the popularity-light
   pool}. Per segment, the best candidate is chosen on 4 user-folds and read on the 5th (nested); in-sample best is
   printed next to it and labelled as such.
 Ceiling of a segment in val NDCG points [D] = segment share x max(0, nested gain + 2 SE); a second, generous
   ceiling = share x (NDCG of NEAR-level performance gap) is printed only as an upper bound that assumes the gap is
   not intrinsic.

C (audit of long-history / sparse-user arms with stored predictions, read by H bucket)
 Arms = every stored train-only val prediction family of an architecture / history arm with a same-seed control:
   retr32, retr32u, retr_cat, no_retr, no_sim, blocks3, kv64 (abl 2-epoch, seeds 23/24, control abl_base);
   e10 retr32, retr32u, blocks3 (seeds 951/952, control w5_e10_base same seeds); kw_hl2, kw_id, kw_ff4, kwh30,
   kd05w192gru, kd05w224 (control kd05w192 same seeds); kd05t25w192b3, kd05t25w256 (control kd05t25w192 same seeds);
   r7 hl2, hc4 (control r7_base); r6 typed, a4u, a4w, cset (control r6_base / the stated control of the report);
   r6_cd hcap, rec33, aug (control r6_cd_m02 same roots); w9 hs1, hs1b (control abl_base same seeds if matched);
   v23_gru vs v17 (families); xdrop_pre2 / xdrop_pop / pop1 as information rows.
 Read: pool = member-equal per-feed z-mean of the arm's stored members; delta vs the control pool on ALL, H=256,
   H<=9 (incl. 0), user_id 0; cluster SE; plus the interaction (segment delta - complement delta).
 Multiplicity: about 30 arms x 3 segments = about 90 looks. An arm is FLAGGED for a segment only if segment z >= 3.3
   (Bonferroni 0.05 one-sided over 90) AND the interaction z >= 2 AND the complement delta >= -1 SE. Anything below is
   reported as "inside the multiplicity noise"; the count of |z| >= 2 cells is compared with the 4.6% expected by chance.

D (proposals): at most two GPU arms, only if A says "deficit candidate / partly" with a mechanism located by A3 or a
   FLAGGED cell of C; otherwise "no arm". Prior for pool-level >= +0.0005 stated as [I].

=== BLOCK 1  2026-11-15T16:43:13Z  (AMENDMENT, written AFTER the s0 / s2 tables were seen and BEFORE s2b / s3 / s4 are run) ===
What was seen: D0 reproduced (0.40271; H = 0 is 194 feeds, not 195). s2: FINAL D(256-NEAR) NDCG -0.0155, AUC -0.0138;
 popularity-only NDCG -0.0042, AUC -0.0127; ratio of skill 256/NEAR 0.872 FINAL, 0.805 popularity, 0.872 pre2_fe.
Reading of the A2 rule AS DECLARED (not changed): (a) FAILS in absolute NDCG (popularity 36%, exposure 27% of FINAL's
 D(256-REST)); (b) FAILS (FINAL AUC at H=256 is 3.1 SE below NEAR); (c) passes in AUC, to be computed in skill.
 So "intrinsic by rule A2" is NOT established as declared. The model-deficit signature (history-free rankers do not get
 worse) is not met either: they get worse by the same AUC amount.
POST-HOC DIAGNOSIS OF THE RULE (labelled post hoc): (a) compared absolute NDCG gaps of rankers whose NDCG range above
 the random level differs by 8x, and (b) assumed AUC is free of difficulty; both are scale errors of mine. The task text
 itself names the right criterion ("the same gap in the same RATIO"). The ratio form is therefore added as rule A2':
 A2' (declared now, before its SE is computed): R_X = skill_X(H=256) / skill_X(NEAR) with a 2,000-draw cluster bootstrap
   by user. The gap is "shared difficulty" if FINAL's R is not below the R of the history-free rankers (popularity,
   exposure, f1) and of the label-free history rankers (max cosine, pre2_fe) by more than the bootstrap 2 SE of the
   paired difference in EVERY one of these five comparisons; "model-specific deficit" if FINAL's R is below at least
   three of the five by more than 2 SE. The same for AUC - 0.5 in place of skill is printed as a second read.
   Because A2' was written after seeing the point ratios, its outcome is reported as POST HOC next to the declared rule.
A1 extension (declared now, not yet computed): extra label-free covariates describing RANGE RESTRICTION of the
   candidate set (hypothesis: the production system already personalises the candidate set of heavy users, so the
   candidates are more alike): per-feed mean and sd over candidates of smax64 (max cosine to the newest 64 events), of
   cos_u, of log1p popularity, of log1p exposure; share of candidates whose own item is in the history; mean pairwise
   cosine between the candidates of the feed. Reported as model "X2" next to the declared covariate set "X1".

=== BLOCK 2  2026-11-15T16:50:20Z  (written AFTER s3, s2b, s4, s5, s6 were read and BEFORE s7 is run) ===
Outcomes by the block-0 / block-1 rules (numbers in out_s*.txt):
 A1: D_adj (X1) -0.0089 +- 0.0039 (52% of raw), D_adj+ (X1 + n_pos) -0.0137 +- 0.0038 (80% of raw, z -3.6) -> by the
     declared rule "DEFICIT CANDIDATE" (= not explained by feed composition); A2 decides whether it is the model.
 A2 (declared): not established as written (block 1). A2' (post hoc): FINAL's skill ratio 256/NEAR 0.872 is not below
     any valid comparator by > 2 SE (pre2_fe 0.872, max cosine 0.859, f1 0.793, popularity 0.805; exposure-only has
     NEGATIVE skill in both buckets, so its ratio is meaningless and is excluded, in AUC - 0.5 it keeps 0.54 vs FINAL 0.94)
     -> "shared difficulty".
 A3: "relevant history is old": first half true (51.9% of positives at H=256 have the best match at index >= 64), second
     half false (smax newest 64 minus all: +0.0008 +- 0.0021 NDCG) -> NOT met. "retrieval depth binds": top-32 minus
     top-16 AUC -0.0045 +- 0.0012 -> NOT met (sign reversed).
 B: no stored family beats FINAL on any small segment (nested gains all negative; 3 of 75 cells at |z| >= 2 vs 3.4 expected).
 C: 0 of 90 cells flagged (z >= 3.3); |z| >= 2 in 6 cells vs 4.1 expected. Largest positive cells are all at H=256:
     a4u (unique retrieval slots) 2 ep seed 23 z +2.66; a4u 10 ep, 6 seeds +0.00274 +- 0.00119 (z +2.30, interaction
     z +2.2); cset 2 ep z +2.37 (10 ep: +0.0004, gone); GRU under KD +0.00216 +- 0.00109 (z +2.0); kv64 z +2.0.
     By the declared rule all of this is INSIDE the multiplicity noise.
FOLLOW-UP on a4u only (it is the one arm with a label-free mechanism from A1: the distinct-item share of the newest 64
 events is 0.658 at H=256 vs 0.738 at NEAR, SMD -0.40, so duplicates can crowd the 16 retrieval slots; and it was
 never run under distillation). Three reads declared NOW, before any is computed:
 F1 independent replication: m02 re-cools of the a4u snapshots (nn_r6_cd_m02_a4u_r101..106, train-only) vs the m02
    re-cools of the F7 snapshots with the same root numbers (cross-root, as the registry notes). One-sided at H=256.
    REPLICATED if delta(H=256) > 0 with z >= 1.64 and interaction (H=256 minus rest) >= 0; REFUTED if delta(H=256) <= 0;
    otherwise UNDECIDED. The fine-tuned form (n2cpv3 OOF, roots with both files) is printed as a second line.
 F2 mechanism dose-response on the 10-epoch 6-seed pair (the z +2.30 cell), feeds with H >= 100: terciles of the
    label-free distinct-item share of the whole stored history; SUPPORTED if delta(bottom tercile = most repeats) minus
    delta(top tercile) > 0 at z >= 2; else NOT SUPPORTED.
 F3 cold/warm trade (section 3 rule: "stale-val for any cold/warm trade, they net about 0"): the H=256 delta of the
    10-epoch pair split by feeds whose positives are all cold / all warm / mixed. If the H=256 gain sits in the all-cold
    feeds and the all-warm feeds lose, it is the known trade and NOT a history effect.
 Proposal rule for D: an a4u arm under distillation is proposed only if F1 is not REFUTED and (F2 SUPPORTED or F3 shows
    the gain is not the cold/warm trade). Otherwise "no arm".
 Also declared now (information, no decision): the worth of the candidate-set / history levers by H for FINAL's own
    family cannot be read (no ablation of FD students exists); stated as a gap.

=== BLOCK 3  2026-11-15T16:51:57Z  (written AFTER s7 was read and BEFORE s8 is run) ===
s7 outcomes by the block-2 rules: F1 UNDECIDED (H=256 +0.00115 +- 0.00121, z +0.95; fine-tuned form +0.00146 +- 0.00134);
 F2 NOT SUPPORTED (bottom minus top tercile of distinct share +0.0011 +- 0.0017 on the 10-epoch pair, -0.0017 +- 0.0018
 on the re-cool pair: the gain does not sit where the repeats are); F3: on the 10-epoch pair the H=256 gain is in cold
 AND warm feeds (+0.0041 / +0.0023), on the re-cool pair the signs split (-0.0035 / +0.0035): no stable pattern.
 By the block-2 proposal rule an a4u arm MAY be proposed (F1 not refuted, F3 not the trade on the pair that carried the
 cell); it is carried as a LOW-PRIOR optional arm, not as a finding.
Three descriptive reads declared now (no decision attached except where stated):
 G1 model-based counterpart of A3 from stored predictions (nn_x/tricks/HDROP-ALL/preds/val: hard-label members
    w5_e10_base_s951/952 scored with 10% / 20% of the history events dropped at random, single pass, 4 masks): cost of
    one dropped-history pass by H bucket (mean over masks of per-feed NDCG minus the full-history NDCG, 2-member mean).
    Reading: if the cost at H=256 is not larger than at NEAR (contrast >= -2 SE), the model's marginal use of history
    is not higher at the cap, i.e. no sign that it is starved of history there; if the cost at H=256 is larger by
    > 2 SE, the cap binds for the model.
 G2 does the model "know" that heavy users are harder: softmax of the mean raw logit of the 11 teacher members per
    feed; predicted probability mass on the top-1 candidate vs the realised y_top1 / n_pos, by H bucket (calibration
    of the listwise target). If realised and predicted fall together at H=256, the lower score is predicted
    difficulty; if realised falls and predicted does not, the model is over-confident there (a deficit signature).
 G3 H = 0 split by whether the organisers' user embedding is non-zero (label-free split): NDCG / AUC of FINAL,
    popularity-only and cos_u; and uid0 vs known users inside H = 0.

=== BLOCK 4  2026-11-15T16:53:16Z  (written AFTER s8 was read and BEFORE s9 is run) ===
s8 outcomes: G1 contrast (cost of a 20% history drop, H=256 minus NEAR) -0.00001 +- 0.00091 -> by the declared reading
 the cap does NOT bind for the model. G2: predicted p(top-1) falls by 0.0098 +- 0.0020 at H=256 vs NEAR, realised by
 0.0143 +- 0.0055, realised minus predicted -0.0045 +- 0.0052 (n.s.) -> predicted difficulty, no over-confidence
 signature. G3: the weak cold cell is H=0 with an all-zero user embedding (74 val feeds, NDCG 0.3515 +- 0.025).
s9 (descriptive decomposition of the A1 result; same cross-fitted procedure, no new decision rule): which covariate
 block carries the 40-48% that matching removed. Blocks: F = feed only (M, log M, day, hour, uid0, cold share, missing
 embeddings, candidate categories); F + T = plus event-type shares of the newest 64 events; F + T + R = plus repeat /
 category structure of the newest 64 (= X1); X1 + A = plus label-free user-activity markers (user seen in train,
 log1p number of train feeds of the user). Expectation written now: feed-only explains < 10%; if X1 + A leaves < 1/3
 of the raw gap the reading is "heavy users are a different KIND of user, and feeds of the same kind of user are as
 hard below the cap".
```

