# Adversarial review: fresh exposure counts from the val / test feeds (CPU pre-gate)

Reviewer's directory `nn_x/w12/fresh/review/` (own code only: `rlib.py`, `r1_rebuild.py`, `r1b_sameuser.py`, `r2_build_full.py`, `r3_nested.py`, `r5_existing.py`, `r6_parity.py`, `r7_lgb.py`, `r8_controls.py`, `r9_stale.py`, `r10_d3.py`; logs in `logs/`, numbers in `out/`). Criteria and time stamps: `nn_x/w12/fresh/review/DECLARATION.txt` (block 1 at 15:54Z before any number of mine; amendments 1-5 each before their numbers). Reviewed: `nn_x/w12/fresh/pregate/`, `reports/FRESH_1115/pregate.md`. Marks: [M] measured here, [D] documented elsewhere, [I] inference. CPU only, at most 2 processes x 12 threads, nothing written outside my directory and this file, no process of anyone else touched.

## 0. Result of the review

**The builder's verdict NO stands (confirmed).** Fresh exposure counted from other users' val / test feeds adds nothing to the final pool score, on plain val, at the public / private mix, and on a pool that reads private-like stale inputs.

| check | my number [M] | builder's number | status |
|---|---|---|---|
| 1 rebuild from the raw parquet | all 4,294,777 rows x 5 windows identical (500-feed sample: 49,563 rows identical) | brute force on 300 rows | confirmed |
| 1 leak rules | same-user exclusion verified on the 678 rows where it matters; counts rebuilt without reading any label | code | confirmed |
| 2 nested read, as-of forms, against the RAW pool score | one weight -0.00004 ± 0.00020 (z -0.18); nine weights -0.00014 ± 0.00020 (z -0.71) | -0.00044 ± 0.00015 (z -2.85) against its own control | NO confirmed; size and sign of the builder's number are family-specific |
| 2 in-sample ceiling (direct NDCG, best single form and weight) | +0.00017 ± 0.00025 | -0.00004 ± 0.00017 (likelihood fit) | below the +0.0005 bar even in sample |
| 3 placebo (within feed x cold, 20 draws) | real above 15 of 20 (one weight; band -0.00037 .. +0.00006), 11 of 20 (nine weights; band -0.00053 .. +0.00007) | real above 2 of 20 | "no better than shuffled" confirmed; "worse than shuffled" not replicated |
| 4 position in the period | no gain on either val day (-0.00018 ± 0.00025, +0.00012 ± 0.00033; difference z +0.71) | -0.00030 / -0.00059 | no gain to concentrate |
| 5 pool score = the final file | val 0.402714; test order identical to `subs/cand_FD16P20kd_r4.parquet` on all but 2 of 2,904,806 rows | 0.40271 | confirmed |
| 5 existing inputs carry the count? | R^2 0.066 (within feed 0.060) | "not in the 34 inputs" (code) | confirmed: not redundant, simply uninformative |
| 6 parity | PSI 0.001 for sampled feeds; 0.002-0.008 for production-ordered and for all train feeds | PSI <= 0.003, "random-like query feeds only" [I] | attainable; the builder's limit is too pessimistic |
| extra: stale-input witness | stale pool loss -0.00143 ± 0.00049; counts give +0.00013 ± 0.00026 / +0.00005 ± 0.00025, inside the placebo band | not run | staleness route closed as well |
| extra: LightGBM disagreement | +0.00042 ± 0.00043 over its control at 200 rounds (above 8 of 10 within-feed placebo draws), never above the raw pool at any of 6 round counts (best -0.00017 ± 0.00010) | +0.00079 ± 0.00045, 10 of 10 placebo draws | replicates weakly inside the family, not usable (section 8) |

## 1. Check 1: the counts and the leak rules [M]
- `r1_rebuild.py` reads only `feed_id, timestamp, user_id, item_id` from `data/competition/{val,test}/impressions/*.parquet` (never the target) and counts with plain python dictionaries. 500 random val feeds (seed 20261115, 49,563 rows): as-of, 24 h, same day, whole period, later-only and the no-uid-0 variant all equal `out/cnt_main.npy` / `cnt_nzsrc.npy` on 100% of the rows.
- `r2_build_full.py` rebuilds every row of val + test by a polars self-join on the item: all 4,294,777 rows equal the builder's array for the five windows (max abs difference 0).
- Same user: only 219 non-zero users have more than one feed in val + test (444 feeds, 92 users in both splits). On their 47,284 rows the same user's other feeds would change the as-of count on 339 rows and the whole-period count on 678 rows; the builder's array equals the excluding version on every one of them and the including version on none (`r1b_sameuser.py`).
- Labels: the arrays are reproduced by a script that reads no label; the forms are functions of the counts; the cold flag is `tte_it_imp == 0` = train impression count (`src/features_tte.py`, label-free column). No (feed, item) duplicate, no candidate with item_id 0, timestamp grid 128 s.
- uid 0: 658 val and 1,179 test feeds, each its own user in the count; my folds keep them as singletons, the variant with all uid-0 feeds in one fold group gives D1 -0.00014 ± 0.00020, D2 -0.00014 ± 0.00021 (section 2 defines D1 / D2).

## 2. Check 2: nested residual read with my folds and my model [M]
The builder's family (conditional logit, 16 columns, L2 10) has a control that is 0.00040 below the raw pool score, so "arm minus control" is measured inside a model that already damages the ranking. My model has no likelihood: it optimises NDCG directly and its control is the raw pool score.
- D1: s = b + w x, one of 7 as-of columns (within-feed z, rank, indicator, cold & shown, log count, the 24 h z and indicator), w on a 21-point grid; the pair (x, w) is chosen by NDCG on the training folds (w = 0 if nothing beats it).
- D2: coordinate ascent over the 7 columns plus b x shown and b x cold & shown (9 weights, 2 passes).
- Folds: 5 folds grouped by user, my seeds 11-13, my split code. Pool score recomputed with my code: 0.402714; my NDCG equals `src/metrics.py` to 4e-16 per feed, also on a tied score.

| read | plain val | pmix public | pmix private |
|---|---|---|---|
| D1 as-of | -0.00004 ± 0.00020 (z -0.18) | -0.00000 ± 0.00026 | +0.00004 ± 0.00031 |
| D2 as-of | -0.00014 ± 0.00020 (z -0.71) | -0.00010 ± 0.00025 | -0.00007 ± 0.00030 |
| builder, arm A | -0.00044 ± 0.00015 (z -2.85) | -0.00064 ± 0.00021 | -0.00067 ± 0.00026 |
| D1 later-using forms (same day, whole period, later-only) | -0.00069 ± 0.00017 | -0.00075 ± 0.00021 | -0.00073 ± 0.00025 |
| D2 later-using forms | -0.00059 ± 0.00023 | -0.00075 ± 0.00029 | -0.00078 ± 0.00035 |
| builder, arm B | -0.00032 ± 0.00033 | -0.00027 ± 0.00041 | -0.00025 ± 0.00050 |

- In-sample ceiling of D1 (fit = score on all 14,423 feeds, the most optimistic read): as-of +0.00017 ± 0.00025 (within-feed z at w 0.03); later-using forms +0.00006 ± 0.00017. No single weight on "cold & shown" improves NDCG at all (best w = 0).
- 13 of the 15 D1 fits choose a weight of 0.02-0.03 on the within-feed z: a consistent, tiny, positive direction that is worth nothing out of fold.
- Coldest feeds (no-train-impression share >= 45%, 2,724 feeds): D1 +0.00038 ± 0.00042, D2 +0.00009 ± 0.00038. The builder's -0.00097 ± 0.00043 there is not replicated; neither read shows a gain where staleness hurts most.
- Descriptive, my code: within-feed AUC of the as-of count 0.4970 ± 0.0012 (cold 0.4913 ± 0.0018, warm 0.4990 ± 0.0014) = the builder's values. Clicks observed / expected from the pool score on rows shown elsewhere: 2,201 / 2,183 = 1.008 ± 0.021 (cold count 1: 1.073 ± 0.052; cold count >= 2: 1.03 ± 0.10; warm 0.994-0.995).

How much the null is worth (instrument checks, `r8_controls.py`):
- Planted flag with the real coverage (10.8% of rows) and a click lift rho given the pool: rho 1.1 reads 0; rho 1.2 reads -0.0003 .. +0.0006 (not detectable); rho 1.35 reads +0.0008 .. +0.0015 (z 1.4-2.6); rho 1.5 reads +0.003 .. +0.004 (z about 4). The measured lift is 1.008 ± 0.021. A flag would need about 1.35.
- Real positive control (declared: base = fine-tuned branch alone, column = train-only branch): reference gain of the final blend +0.00081 ± 0.00040, nested D1 +0.00017 ± 0.00027, in-sample best inside my grid +0.00041. **This fails my own sensitivity criterion** (half of the reference at z >= 2). The control was badly built: the reference blend is w = 1 and my grid stopped at 0.5. With the grid extended to 2.0 (AMENDMENT 5, declared before the run) the nested read is +0.00058 ± 0.00038 (z +1.53) = 72% of the reference, with w = 0.75 chosen in 11 of 15 fits. So the instrument recovers most of a real effect; what it cannot do on 14,423 feeds is certify +0.0005..+0.0008 at z >= 2 when the effect is spread over all rows (the reference itself is only z 2.0).
- Consequence for the reading: the z of a null read carries little. What closes the idea is the size: the in-sample ceiling (+0.00017), the click lift (1.008 where about 1.35 is needed) and the train-scale AUC (section 6) all sit far below the bar.

## 3. Check 3: my placebo [M]
Within each feed, separately for cold and warm candidates, the count vector is permuted across candidates. Every feed keeps its own counts exactly (level, position in the period, cold-conditional distribution); only "which item has the count" is destroyed (correlation with the real as-of count 0.07). 20 draws, seeds 500-519, identical nested read. Second placebo: the builder's strata (day x cold x 3-hour block), my code, 10 draws.

| read | real | placebo mean | sd | min .. max | real above |
|---|---|---|---|---|---|
| D1 as-of, within-feed placebo | -0.00004 | -0.00014 | 0.00013 | -0.00037 .. +0.00006 | 15 of 20 |
| D2 as-of, within-feed placebo | -0.00014 | -0.00019 | 0.00015 | -0.00053 .. +0.00007 | 11 of 20 |
| D1 as-of, builder's strata | -0.00004 | -0.00011 | 0.00015 | -0.00044 .. +0.00016 | 8 of 10 |
| D2 as-of, builder's strata | -0.00014 | -0.00017 | 0.00014 | -0.00036 .. +0.00002 | 5 of 10 |
| D1 later-using, within-feed placebo | -0.00069 | -0.00028 | 0.00010 | -0.00047 .. -0.00002 | 0 of 20 |
| D2 later-using, within-feed placebo | -0.00059 | -0.00031 | 0.00018 | -0.00061 .. +0.00011 | 1 of 20 |
| builder, arm A (his placebo) | -0.00044 | -0.00015 | 0.00017 | -0.00051 .. +0.00011 | 2 of 20 |

- As-of forms: the real columns sit in the middle of the shuffled ones. A column with no item link costs 0.0001-0.0002 by selection alone; the real one costs the same.
- Later-using forms are BELOW every shuffled draw in the one-weight read (0 of 20). They are not a hidden gain; the fold-to-fold choice is unstable (weights from -0.05 to +0.2 on five different columns), which is what a correlated but useless column does. No form that uses later feeds should be revisited.

## 4. Check 4: position in the period [M]
Label-free coverage (count >= 1, percent of candidates), my arrays:

| position | as-of, all earlier feeds | as-of, last 24 h | through val feeds | through test feeds |
|---|---|---|---|---|
| val day 20718, first / middle / last third (calendar) | 1.9 / 6.1 / 10.8 | 1.9 / 6.1 / 10.8 | same | 0 |
| val day 20720 | 11.5 / 14.2 / 16.8 | 2.2 / 6.3 / 10.3 | same | 0 |
| first test day, hour 0-1 | 16.6 | 0.6 | 16.2 | 0.6 |
| first test day, hours 1-3 / 3-6 / 6-12 / 12-24 | 15.8 / 15.0 / 17.5 / 20.1 | 1.6 / 1.7 / 4.2 / 8.9 | 14.7 / 13.8 / 14.8 / 14.4 | 1.6 / 1.7 / 4.2 / 8.9 |
| test day 2 / 3 / 4 | 24.1 / 27.4 / 30.5 | 12.6 / 13.4 / 13.1 | 13.3 / 12.0 / 11.3 | 15.9 / 21.5 / 25.8 |

- "Would the first test day get almost nothing?" From FRESH feeds, yes: 0.6% in its first hour, under 2% for six hours, 6.8% on average over the day. The 18.9% that the all-earlier form shows on that day comes from val feeds that are 2-4 days old: 63.9% of its covered rows have no source younger than 24 h (public-like rows 58.0%, private-like 52.9%, val day 20718 0%). The as-of count on test is mostly a 2-7 day old count: the same kind of staleness the idea wanted to remove, in a mix that val day 20718 never shows.
- Raw-count shift (PSI), confirming the builder: as-of val -> public-like 0.076, val -> private-like 0.238, first val day -> last private day 0.490; 24 h form 0.003 / 0.044; same day 0.000 / 0.001.
- Gain by position (per-feed gains of D1 | D2): day 20718 -0.00018 ± 0.00025 | -0.00020 ± 0.00021; day 20720 +0.00012 ± 0.00033 | -0.00007 ± 0.00034; difference +0.00030 ± 0.00042 (z +0.71) | +0.00013 ± 0.00040. Thirds of day 20720: +0.00107 ± 0.00057, -0.00028, -0.00043 (one cell of six at 1.9 SE; not a claim). By quintile of the feed's as-of coverage (0.8% .. 23.6%): +0.00051, -0.00063, +0.00053, +0.00031, -0.00091 (D1): no trend, and the best-covered fifth is the worst. Declared statement "gain concentrated late in val": NOT true; there is no gain to concentrate.

## 5. Check 5: step-0 facts and the pool score
- Code lines opened: `src/features.py:241-248` (history counts over train + val + test; per-split candidate bincount), `:336-337` (`it_cand_cnt_split`), `src/features_luo.py:45-51` (subtracts only the same user's later snapshots, `th > tc`), `nn_x/w5/finetune.py:67, 87-89` (ITEM_PRESET; ITEM_CLEAN drops `it_cand_cnt_split*`), `nn_x/abl/train.py:91-111` (cvraw / cvnd / cvnd_pre2s), the 42 extras printed in `nn_x/w5/logs/w5_e10_base_s951_n2cpv3_ft.log`, `nn_x/abl/pop1/build_xp_imp_c50.py`, `src/features_tte.py`. All of the builder's statements hold. `data/feats/val/` holds `it_cand_cnt_split{,_gap,_rk,_z}.npy` and no `ctx_ix_*` file; the "+-48 h exploration appearances" of `feats_v2/ctx_/build_ctx.py:15` exist only in the docstring.
- Does an existing input already carry the count? No [M]: least-squares R^2 of log1p(as-of count) on the 33 stored train-stage inputs + xp_imp_c50 is 0.066 pooled, 0.060 within feed (0.063 with the pool score added); cold rows 0.038, warm 0.102. Mean within-feed Spearman of the count with xp_imp_c50 +0.149, it_hist_all_luo +0.117, the pool score -0.038. The pool ranks shown-elsewhere candidates lower (mean pool z, cold: 0 / 1 / 2-4 / 5+ other feeds = -0.24 / -0.52 / -0.67 / -1.01) and is right to: their clicks match its expectation (1.008).
- So the null is not redundancy. The count is new information about the candidate pool that carries no information about the click.
- Pool score: `preds/ens_FD16P20kd_{ft,to}_val.npy` (written 12:51:55Z together with `subs/cand_FD16P20kd_r4.parquet`); my recomputation 0.402714; on test the same formula gives the within-feed order of the submitted file on all but 2 of 2,904,806 rows (the file stores ranks).

## 6. Check 6: could a train-row version be built? [M]
My construction (`r6_parity.py`): real calendar pattern on train days 20702, 20704 | 20706-20709 (the builder used 20703 ...), sources = random-like train feeds (p_random > 0.9) at the real daily feed numbers, re-weighted to the val + test feed-size deciles (43,015 feeds, mean M 91.6).

| query feeds | as-of coverage per day, train / real | PSI val / public-like / private-like |
|---|---|---|
| the sampled feeds themselves | 8.0/7.3, 15.2/14.6, 18.0/18.9, 22.9/24.1, 26.4/27.4, 29.2/30.5 | 0.001 / 0.001 / 0.001 |
| 20,000 production-ordered feeds (p_random < 0.1) | 9.6/7.3, 17.4/14.6, 19.8/18.9, 25.0/24.1, 28.8/27.4, 31.2/30.5 | 0.008 / 0.002 / 0.003 |
| 20,000 feeds from all train feeds | 9.4/7.3, 16.7/14.6, 19.6/18.9, 25.1/24.1, 28.7/27.4, 30.9/30.5 | 0.006 / 0.001 / 0.004 |
| sampled feeds, sources without the size match | 5.8/7.3 .. 21.8/30.5 | 0.009 / 0.030 / 0.040 |

- Yes: with size-matched random-like sources the distribution is reproduced for ANY train feed as the query, not only for the 11% random-like ones (the builder's stated limit was an inference; measured PSI 0.002-0.008 against a rule of 0.25 / 0.5). A member would not have to be fitted on val alone.
- Real limits: (a) every train feed needs an assigned position in a pseudo-period, because the input's level depends on it (7% -> 30%); (b) train has a hole on days 20711-20712 (42 and 0 feeds), so the real calendar pattern fits only at two start days (20702, 20703); (c) each pseudo-period uses 1.4% of the train feeds as sources, so the count of a train row is one noisy draw.
- What a model would find there: within-feed AUC of the as-of count for the TRAIN click on the 43,015 sampled feeds 0.4979 ± 0.0009 (builder, other days: 0.4991 ± 0.0009; train exposure on his rows 0.5190). Nothing, at 3 times the val sample.
- If it had to be val alone: 21,983 clicks, of which 2,201 sit on rows shown elsewhere and 477 on cold & shown rows. A weight learned from that is the case the project already measured (rule picked on val: +0.0003..+0.0007 in sample, about 0 nested, -0.00015 on public [D]); my positive control recovers +0.00058 ± 0.00038 of a real +0.00081 only when the right weight is inside the grid, +0.00017 when it is not. The fine-tune would also meet the input outside its val range on test (PSI 0.24 to private-like, 0.49 from the first val day to the last private day).

## 7. Extra: the witness the builder did not run (stale inputs) [M]
The bottleneck is private-like staleness (11.2 d); the builder's read is on plain val (5.7 d) and pmix only re-weights feeds, it does not make the inputs stale. `r9_stale.py` (declared in AMENDMENT 3 before its numbers): stale pool = the 16 held-out twins of the FD16 base, train-only branch, scored on val with private-like stale inputs (`nn_x/w5/preds/stale/nn_w5_e10_kd05{w192,t25w192}_s951..958_staleval.npy`, scenario both_pre2both).
- Fresh train-only pool 0.40144, stale pool 0.40001: loss -0.00143 ± 0.00049. The scenario zeroes the history count of 10.6% of val rows.
- As-of counts on the stale pool: D1 +0.00013 ± 0.00026 (z +0.48), D2 +0.00005 ± 0.00025; pmix private +0.00013 ± 0.00038 / +0.00012 ± 0.00036; in-sample ceiling +0.00035. Placebo (10 draws): -0.00039 .. +0.00039, real above 7 of 10 (D1) and 6 of 10 (D2).
- Stale minus fresh pool, paired: D1 +0.00035 ± 0.00022 (z +1.56), D2 +0.00011 ± 0.00021. In the third of feeds with most rows made cold (loss -0.00260): D1 +0.00039 ± 0.00047, D2 +0.00023 ± 0.00046.
- Reading by the declared rule: no repair. This read is biased in favour of the idea (the scenario hides the true train popularity of really warm items, of which the count is a weak proxy), and it still recovers at most a tenth of the loss, inside the placebo band.

## 8. Extra: the LightGBM disagreement [M]
`r7_lgb.py`: the builder's parameters and columns, my folds, init score = b (the builder scales b by a fitted beta of about 1), read at 6 round counts. The feed-level arm holds only columns that are constant inside a feed (coverage and mean log count of the feed).

| rounds | control - raw pool | arm - control | arm - raw pool | feed-level arm - control |
|---|---|---|---|---|
| 5 | +0.00005 ± 0.00009 | -0.00022 ± 0.00010 (z -2.11) | -0.00017 ± 0.00010 | -0.00007 ± 0.00006 |
| 10 | +0.00006 ± 0.00012 | -0.00027 ± 0.00014 (z -1.87) | -0.00021 ± 0.00016 | -0.00021 ± 0.00009 |
| 20 | -0.00031 ± 0.00018 | +0.00008 ± 0.00021 | -0.00023 ± 0.00023 | -0.00020 ± 0.00014 |
| 50 | -0.00097 ± 0.00028 | +0.00021 ± 0.00028 | -0.00077 ± 0.00032 | -0.00023 ± 0.00023 |
| 100 | -0.00191 ± 0.00035 | +0.00064 ± 0.00034 (z +1.86) | -0.00127 ± 0.00037 | +0.00032 ± 0.00030 |
| 200 | -0.00282 ± 0.00043 | +0.00042 ± 0.00043 (z +0.97) | -0.00240 ± 0.00044 | +0.00005 ± 0.00039 |

- With other folds the 200-round difference is +0.00042 ± 0.00043 (builder +0.00079 ± 0.00045).
- The arm never beats the raw pool score: best -0.00017 ± 0.00010 at 5 rounds, in sample over the round count. While the tree model is still at the pool's level (5-10 rounds) the counts make it WORSE (z -2.1, -1.9). The positive difference grows only as the control falls below the pool (by 0.0003 at 20 rounds, 0.0028 at 200), i.e. the extra columns slow down a model that is destroying the ranking.
- Within-feed placebo in this family (10 draws, arm - control): 5 rounds real -0.00022 against a placebo mean of -0.00011 (real above 2 of 10); 10 rounds -0.00027 / -0.00023 (3 of 10); 20 rounds +0.00008 / -0.00019 (10 of 10, +1.7 sd); 50 rounds +0.00021 / -0.00021 (10 of 10, +2.0 sd); 100 rounds +0.00064 / +0.00006 (9 of 10, +2.0 sd, placebo max +0.00066); 200 rounds +0.00042 / +0.00009 (8 of 10, +1.3 sd, placebo max +0.00045). The builder's strata placebo with my folds (5 draws): 200 rounds placebo mean +0.00015, sd 0.00036, max +0.00057, real above 4 of 5 (+0.8 sd); 100 rounds +0.00028 / max +0.00062, real above 5 of 5 (+1.1 sd); 50 rounds 5 of 5 (+1.6 sd); 5 rounds real below all 5.
- The feed-level arm does not reproduce the difference (200 rounds +0.00005 ± 0.00039; 10 rounds -0.00021 ± 0.00009): what the trees use is tied to the item, not to the feed's count level.
- So the builder's observation replicates in part: from 20 rounds on, the real counts lose less than shuffled ones inside this tree family (about +0.0003..+0.0006 against the within-feed shuffle, +1.3 to +2.0 placebo sd; +0.8 to +1.6 sd against the strata shuffle; not above every draw at 100 and 200 rounds). It is a difference between two damaged models: the shuffled arm is 0.0005-0.0027 below the raw pool at those round counts, the real arm 0.0002-0.0024 below.
- Is there an interaction of "shown elsewhere" with the pool score that a direct read can use? No. D2 already holds b x shown and b x cold & shown. D3 (`r10_d3.py`, declared in AMENDMENT 4 before its numbers; 8 threshold interactions such as shown x [b > 1], two or more other feeds x [b > 0]): one weight -0.00026 ± 0.00009, eight weights -0.00029 ± 0.00014, real above 3 of 10 placebo draws in both; in-sample ceiling +0.00010 ± 0.00010, which is below the in-sample ceilings of the shuffled counts (mean +0.00016, max +0.00026).
- [I] The tree family gains nothing from the counts over the pool; the counts only change how fast its own over-fitting of b hurts. By the declared reading (arm - raw pool >= +0.0005 at z >= 2 at some round) it is not usable, and it is not evidence for a neural member either (project rule: GBDT-proxy gains never justify NN GPU; here the proxy is negative against the pool).

## 9. Issues
- MEDIUM. The decision number is family-specific and its sign should not be quoted. "-0.00044 (z -2.85), below 18 of 20 placebo draws" reads as "the counts hurt". Against the raw pool score a direct-NDCG read gives -0.00004 ± 0.00020 / -0.00014 ± 0.00020, inside the placebo band; the builder's -0.00097 in the coldest feeds is not replicated (+0.00038 ± 0.00042). The correct entry is "no signal". His "in-sample ceiling" is a likelihood fit, not an NDCG ceiling; the NDCG ceiling is +0.00017 ± 0.00025, still under the bar.
- MEDIUM. The pre-gate has no stale-input witness, although the evidence policy names stale-val for the staleness family and the idea targets private-like staleness. Closed by this review (section 7: null); the verdict does not change.
- LOW. Nested reads on 14,423 feeds recover a real effect (72% of a real +0.00081 in my positive control) but cannot certify +0.0005 at z >= 2 unless it sits on few rows. The NO is safe because of the effect size (in-sample ceiling +0.00017, click lift 1.008 ± 0.021 against about 1.35 needed, train AUC 0.498), not because of z values. The builder's report leans on z -2.85; the registry line should lean on the size.
- LOW. The LightGBM "disagreement" is real inside that family but not usable (section 8): half the size with other folds (+0.00042 ± 0.00043), above 8 of 10 within-feed and 4 of 5 strata placebo draws at 200 rounds (his 10 of 10 does not hold with other folds), negative at 5-10 rounds, never above the raw pool, and no explicit interaction form reproduces it (D3 -0.00026 ± 0.00009). The builder's wording "reported, not promoted" is right.
- LOW. Parity limit overstated: the builder writes that parity holds for random-like query feeds only (about 11% of train) [I]; measured PSI 0.002-0.008 for production-ordered and all train feeds. Not mentioned by him: the 20711-20712 hole in train, which leaves two real-calendar start days.
- LOW. Source age not reported: 53-64% of the as-of covered test rows have no source younger than 24 h; on the first test day fresh coverage is 0.6% in the first hour. The builder reports the level shift (PSI) but not that the "fresh" count on test is mostly a 2-7 day old count from val feeds.
- LOW. The later-using forms are below all 20 shuffled draws in my one-weight read (-0.00069 against a band of -0.00047 .. -0.00002); the builder reads them as "-0.00032, z -0.96, 9 of 20". Same decision (not WEAK), stronger reason not to revisit them.
- LOW (process, no effect on numbers). The builder's brute-force check re-uses the arrays and the user key of the kernel it checks; it would not catch a wrong flat table or user key. The raw-parquet rebuild closes this.
- No BLOCKER, no MAJOR. Declarations precede the numbers; the deviation from the literal placebo is argued before running and is correct (a within-day permutation of item ids keeps each item's multiplicity).

## 10. Next step for the lead
1. Enter in COVERAGE_MAP / registry: "fresh exposure from val + test feeds (other users, as-of and later-using forms): NO at the CPU pre-gate, confirmed by review; no signal over the pool on val (-0.00004 ± 0.00020), at the private mix and on stale-val (+0.00013 ± 0.00026); click lift given the pool 1.008 ± 0.021". No GPU, no slot.
2. Do not re-open it as a member input or a fine-tune extra (train-scale AUC 0.498; parity was never the obstacle); the staleness lever that remains is on the model side (popularity-light weight / members), not a new exposure input.
