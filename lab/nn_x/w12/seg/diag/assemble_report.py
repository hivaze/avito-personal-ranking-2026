"""Assembles reports/SEG_1115/diag.md from the prose below and the raw outputs out_s*.txt of this directory."""
from pathlib import Path
import subprocess
HERE = Path("<repo>/nn_x/w12/seg/diag")
OUT = Path("<repo>/reports/SEG_1115/diag.md")
stamp = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip()
raw = lambda n: "```\n" + (HERE / n).read_text().rstrip() + "\n```\n"

BODY = r"""# SEG_1115 / diag: user-information segments (cold start, unknown users, sparse users, heavy users)

Assembled __STAMP__ by `nn_x/w12/seg/diag/assemble_report.py`. Analyst directory: `nn_x/w12/seg/diag/` (scripts `s0`..`s10`, raw outputs `out_s*.txt`, `DECLARATION.txt` with five stamped blocks, `PROGRESS.md`). CPU only. File analysed: FD16P20 (`subs/cand_FD16P20kd_r4.parquet`, public 0.394892), rebuilt on val exactly as `nn_x/w12/kd/build_r4.py`: 0.40271 [M]. Marks: [M] measured here, [D] arithmetic from [M], [I] judgement. All SE are cluster-robust by user (user_id 0 is a pooled bucket, each of its feeds is its own cluster). H = stored history length (the data caps it at 256).

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
__S0__
### s2 rankers by H bucket
__S2__
### s2b ratio bootstrap
__S2B__
### s3 matching
__S3__
### s9 covariate blocks
__S9__
### s4 positives
__S4__
### s8 history-drop, calibration, cold-start splits
__S8__
### s5 small segments
__S5__
### s6 audit by H bucket
__S6__
### s7 a4u follow-up
__S7__
### s10 distinct items in the retrieval slots
__S10__
### DECLARATION.txt
__DECL__
"""
rep = BODY.replace("__STAMP__", stamp)
for key, fn in (("__S0__", "out_s0.txt"), ("__S2B__", "out_s2b.txt"), ("__S2__", "out_s2.txt"), ("__S3__", "out_s3.txt"), ("__S9__", "out_s9.txt"), ("__S4__", "out_s4.txt"), ("__S8__", "out_s8.txt"),
                ("__S5__", "out_s5.txt"), ("__S6__", "out_s6.txt"), ("__S7__", "out_s7.txt"), ("__S10__", "out_s10.txt"), ("__DECL__", "DECLARATION.txt")):
    rep = rep.replace(key, raw(fn))
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(rep)
print("written", OUT, len(rep.splitlines()), "lines")
