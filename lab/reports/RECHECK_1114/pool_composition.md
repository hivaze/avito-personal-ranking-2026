# RECHECK 11-14, lens "pool_composition": which students belong in the pool

CPU only, read-only on every prediction file, no submission file built. Scripts and logs: `nn_x/w12/recheck/pool_composition/` (`pc_*.py`, `out/*.log`, `out/*.json`).
All deltas are paired per feed vs the **W8 train-only (TO) pool** unless stated; ± = paired SE (val 14,423 feeds, holdout 30,000 feeds). "val | hold" = val | in-period holdout. Pool = member mean of per-feed z.

## 0. Summary

1. **No composition of the existing students beats W8 on the holdout, on val, or on both.** 32 member-equal unions that contain W8, 8 weight grids and 6 cross-witness forward selections: the best holdout delta is +0.00010 ± 0.00015 among unions (W8 + 2 three-block students; −0.00003 at @last) and +0.00017 ± 0.00011 among weightings (0.8 W8 + 0.2 generation-2; +0.00006 at @last). Member-equal W8 stays.
2. **The wide-student pool is saturated.** All 255 subsets of W8 follow NDCG(n) = a − b/n to 1e-4. Asymptote a = 0.40181 val | 0.32472 hold; W8 is 0.00024 ± 0.00012 | 0.00017 ± 0.00007 below it. Expected W16 − W8 = +0.00016 val | +0.00013 hold (law); the three extension seeds already finished give −0.00030 ± 0.00027 | −0.00007 ± 0.00016. In public units the whole family tops out near 0.3947 (#2 is 0.39512, #1 0.39826).
3. **Every d192 construction has the same infinite-seed value** (teacher 11 / 25 / students-as-teacher, kd weight 0.3 or 0.5, no dropout, 20 epochs, FFN x4, 3 blocks; within-arm correlation from 0.955 to 0.972): asymptote 0.4014-0.4021 val | 0.3246-0.3249 hold. Lower correlation obtained from noise buys nothing at the asymptote. The one measured ceiling shift is width under KD: wide − narrow +0.00043 ± 0.00053 val | **+0.00090 ± 0.00031 hold**. Narrow students have the ceiling of their teacher pool (+0.00052 ± 0.00054 | −0.00014 ± 0.00031).
4. **The seed is the only diversity source among students.** Different-teacher, different-generation and different-width pairs with different seeds correlate exactly like same-arm pairs (0.9718 vs 0.9722); same-seed cross-teacher pairs correlate more (0.9762). This closes lever 4 of TOP1_GAP_EXTERNAL ("disjoint-teacher students") by its own CPU rule, and predicts that kdmixw192 cannot add diversity.
5. **Adding C3 hurts the holdout at every weight** (10% C3: −0.00030 ± 0.00009; 50%: −0.00044 ± 0.00023; member-equal 36: −0.00095 ± 0.00029) and is flat on val (−0.0001..+0.0002). Narrow students: S16 −0.00028 ± 0.00019 hold, +0.00015 ± 0.00033 val.
6. **Forward selection fitted on the holdout never picks a narrow or a hard-label group** and reads −0.0005..0.0000 on val; fitted on val it picks pop1 and narrow groups and reads **−0.00095 ± 0.00028 on the holdout**. Nothing transfers in either direction.
7. Public agrees in sign with the holdout on both KD-era contrasts (S16 and C3 below W8). The S16 contrast is weak on every set: holdout −0.00028 ± 0.00019, public −0.00041 ± ~0.00044, val +0.00008 ± 0.00034; pooled −0.00022 ± 0.00016.
8. Recipes for tomorrow (§8): five pools tie with W8 by the pre-declared rule, all with final-file test correlation ≥ 0.9996 to W8. Expected public change is inside ±0.0002 against a paired public SE of about 0.00026. **None deserves a slot for level.** W16 is a variance reduction, not a level step.

## 1. Snapshot, method, rule

- Snapshot 2026-11-14T19:04:24Z: 112 completed `w5_e10_*` runs (`out/snapshot_w5_e10_complete.txt`) plus `abl_e10_pop1kd05t25_s951-954`. New since the task text: kd05w192_s955, kd05t25w192_s955, kd05t25w192_s957. No further run completed by 19:28Z (the other five extension seeds were at step 62-65K of 98K).
- 208 member x checkpoint arrays cached (selected and @last). The fast NDCG reproduces `src/metrics.ndcg_per_feed` exactly (max per-feed difference 0.0).
- C3 on the holdout: all 28 members have holdout files (`preds/holdout/`). The rebuilt C3 TO val pool matches `preds/ens_Q2a8s3_to_val.npy` at corr 1.000000. C3 TO holdout = **0.32306** (G3 TO 0.32379).
- Label-free test check: each member's correlation to the W8 pool on test equals its val value within 0.0017 (students within 0.0010); all files finite.
- **Ordering rule, written 19:05:48Z before any pool number of this recheck** (`out/PREDECLARED_rule.txt`): order by holdout delta vs W8; "ahead" needs ≥ +2 SE; val is a veto only (TO or final delta below −1 SE); the holdout is also read at @last because checkpoints are selected on it; ties (within 1 SE) prefer more d192 members, then lower test correlation; fitted weights are judged only on the set they were not fitted on.
- Checkpoint-selection bias on the holdout is small: W8 selected 0.32456 vs @last 0.32447; 32 of 86 members have selected = last.

## 2. Members, groups, correlation structure

Asymptote A = (n · pool − mean member) / (n − 1), the infinite-seed pool value under NDCG(n) = a − b/n. Selected checkpoints, val | hold.

| group | n | member | corr within (val) | pool | asymptote A |
|---|---|---|---|---|---|
| base (hard labels, d128) | 6 | 0.39648 \| 0.32022 | 0.9484 | 0.39993 \| 0.32347 | 0.40062 \| 0.32412 |
| F7x (hard, teacher family) | 7 | 0.39667 \| 0.31999 | 0.9497 | 0.40061 \| 0.32332 | 0.40127 \| 0.32388 |
| teacher tf11 (F7x + base x4) | 11 | | | 0.40054 \| 0.32371 | 0.40093 \| 0.32407 |
| kd05 | 4 | 0.39911 \| 0.32229 | 0.9776 | 0.40085 \| 0.32349 | 0.40143 \| 0.32389 |
| kd05t25 | 4 | 0.39957 \| 0.32221 | 0.9780 | 0.40095 \| 0.32317 | 0.40141 \| 0.32349 |
| kd03 | 4 | 0.39806 \| 0.32168 | 0.9657 | 0.39977 \| 0.32342 | 0.40034 \| 0.32400 |
| kd08 | 4 | 0.39928 \| 0.32302 | 0.9904 | 0.40004 \| 0.32377 | 0.40030 \| 0.32402 |
| kd10 | 4 | 0.39883 \| 0.32256 | 0.9950 | 0.39906 \| 0.32291 | 0.39913 \| 0.32303 |
| **kd05w192** | 4 | 0.39918 \| 0.32267 | 0.9722 | 0.40134 \| 0.32408 | 0.40206 \| 0.32455 |
| **kd05t25w192** | 4 | 0.39952 \| 0.32259 | 0.9723 | 0.40100 \| 0.32412 | 0.40150 \| 0.32463 |
| **W8** | 8 | 0.39935 \| 0.32263 | 0.9726 | **0.40157 \| 0.32456** | 0.40188 \| 0.32483 |
| W11 (W8 + s955, s955, s957) | 11 | new: 0.39965 \| 0.32246 | 0.9724 | 0.40127 \| 0.32448 | 0.40145 \| 0.32467 |
| kd05g2w192 | 4 | 0.39914 \| 0.32252 | 0.9702 | 0.40080 \| 0.32410 | 0.40136 \| 0.32463 |
| kd03g2w192 | 4 | 0.39762 \| 0.32136 | 0.9548 | 0.40093 \| 0.32393 | 0.40203 \| 0.32479 |
| kw4 = kw_drop0, e20, hl2, ff4 (x2 each) | 8 | 0.39857 \| 0.32229 | | 0.40142 \| 0.32461 | 0.40183 \| 0.32494 |
| kx05 (cross-fitted teacher) | 4 | 0.39916 \| 0.32237 | 0.9736 | 0.40079 \| 0.32347 | 0.40133 \| 0.32383 |
| kx08 | 4 | 0.39938 \| 0.32273 | 0.9907 | 0.40008 \| 0.32321 | 0.40032 \| 0.32337 |
| pop1kd (pop1 view, d128) | 4 | 0.39821 \| 0.31625 | 0.9759 | 0.39943 \| 0.31715 | 0.39984 \| 0.31744 |
| C3 TO (F10, F7x, F7x70, pop1 x8) | 28 | | 0.9399 | 0.40103 \| 0.32306 | |

Paired asymptote contrasts (val | hold):

| contrast | val | hold |
|---|---|---|
| W8 − N8 (narrow kd05 + kd05t25) | +0.00043 ± 0.00053 | **+0.00090 ± 0.00031** |
| W8 − base x6 | +0.00126 ± 0.00065 | +0.00071 ± 0.00038 |
| W8 − F7x x7 | +0.00062 ± 0.00063 | +0.00096 ± 0.00037 |
| N8 − teacher tf11 | +0.00052 ± 0.00054 | −0.00014 ± 0.00031 |
| N8 − base x6 | +0.00083 ± 0.00063 | −0.00019 ± 0.00036 |
| gen-2 x8 − W8 | +0.00001 ± 0.00051 | −0.00021 ± 0.00029 |
| kw4 x8 − W8 | −0.00005 ± 0.00058 | +0.00010 ± 0.00032 |
| kx05 − kd05w192 | −0.00073 ± 0.00067 | −0.00072 ± 0.00036 |
| kd08 − kd05 | −0.00113 ± 0.00059 | +0.00012 ± 0.00034 |
| kd10 − kd05 | −0.00230 ± 0.00062 | −0.00087 ± 0.00036 |

Reading: a member gain that comes with a higher within-arm correlation is a move along one frontier, not a better pool (kd08 vs kd05: member +0.00017 | +0.00073, asymptote −0.00113 | +0.00012).

Mean member-member correlation on val (holdout values are about +0.001 higher for same-view pairs):

| pair type | same seed | different seed | within arm |
|---|---|---|---|
| kd05w192 x kd05t25w192 (teacher 11 vs 25) | 0.9762 | 0.9718 | 0.9722 / 0.9723 |
| kd05 x kd05t25 | 0.9819 | 0.9773 | 0.9776 / 0.9780 |
| kd05w192 x kd05g2w192 (generation) | 0.9760 | 0.9709 | 0.9722 / 0.9702 |
| kd05w192 x kx05 (cross-fitted teacher) | 0.9761 | 0.9719 | 0.9722 / 0.9736 |
| kd05 x kd05w192 (width) | 0.9751 | 0.9740 | 0.9776 / 0.9722 |
| wide students x base / F7x / F10 / pop1 | | 0.959 / 0.959 / 0.949 / 0.946 | |
| pop1kd x wide students | | 0.960-0.964 | 0.9759 |

Pair-to-pair sd of the within-arm correlation is 0.0002-0.0006, so one pair estimates it well.

## 3. Saturation with the number of wide students

All subsets of the 8 wide students (8, 28, 56, 70, 56, 28, 8, 1 subsets per size); the subset mean is unbiased for E[NDCG(n)] over seeds. Fit residuals ≤ 0.0001.

| n | TO val | TO hold | FT val | final val (0.5 z FT + 0.5 z TO) |
|---|---|---|---|---|
| 1 | 0.39935 | 0.32263 | 0.39892 | 0.39957 |
| 2 | 0.40050 | 0.32355 | 0.39997 | 0.40063 |
| 3 | 0.40095 | 0.32396 | 0.40035 | 0.40104 |
| 4 | 0.40114 | 0.32414 | 0.40055 | 0.40126 |
| 5 | 0.40132 | 0.32426 | 0.40071 | 0.40139 |
| 6 | 0.40138 | 0.32438 | 0.40082 | 0.40148 |
| 7 | 0.40149 | 0.32445 | 0.40099 | 0.40147 |
| 8 (actual W8) | 0.40157 | 0.32456 | 0.40087 | 0.40165 |
| fit a − b/n | a 0.40181, b 0.00250 | a 0.32472, b 0.00215 | a 0.40118, b 0.00230 | a 0.40185, b 0.00232 |
| **16** (extrapolated) | 0.40165 | 0.32459 | 0.40104 | 0.40170 |
| **32** | 0.40173 | 0.32466 | 0.40111 | 0.40178 |
| infinity | 0.40181 | 0.32472 | 0.40118 | 0.40185 |
| 16 − fitted 8 | +0.00016 ± 0.00002 | +0.00013 ± 0.00001 | +0.00014 | +0.00014 ± 0.00002 |
| 32 − fitted 8 | +0.00023 ± 0.00003 | +0.00020 ± 0.00002 | +0.00022 | +0.00022 ± 0.00003 |
| 16 − actual W8 | +0.00009 ± 0.00013 | +0.00003 ± 0.00007 | +0.00016 ± 0.00012 | +0.00005 ± 0.00013 |
| infinity − actual W8 | +0.00024 ± 0.00012 | +0.00017 ± 0.00007 | +0.00031 ± 0.00012 | +0.00020 ± 0.00012 |

- The realised W8 sits 0.00007 | 0.00010 above its own fitted curve, so part of the remaining headroom is already banked. @last checkpoints give the same curve (a 0.40189 | 0.32469).
- Out-of-range check: a fit on n = 1..4 predicts n = 8 with error −0.00013 ± 0.00021 val | −0.00019 ± 0.00011 hold.
- Out-of-sample check with the 3 finished extension seeds: W11 − W8 = −0.00030 ± 0.00027 val | −0.00007 ± 0.00016 hold (law: +0.00009 | +0.00007). The mean of the 56 eight-pools that contain all three new seeds is −0.00035 ± 0.00028 | −0.00020 ± 0.00017 vs W8. A curve refitted on all 11 gives a = 0.40155 | 0.32463.
- Leave-one-out of W8: every drop is within −0.00030..+0.00005; no member is harmful.
- Size of a seed effect in public units: a W16-type file differs from W8 by a paired SE of about 0.00026 on 8,562 public feeds (from the val paired sd). The expected +0.0001 is 0.4 SE. **A public move of ±0.0005 between W8 and W16 is not evidence about pool size.**

## 4. Member-equal unions and the diversity-adjusted expectation

Law fitted on 18 homogeneous groups: pool − mean member = k (1 − rho)(1 − 1/N), k = 0.084 val (R² 0.95, residual sd 0.00028), 0.076 hold (R² 0.97, residual sd 0.00019). "Expect" = mean member + k (1 − mean pairwise rho)(1 − 1/N).

| pool | N | val | hold | d val | d hold | d hold @last | expect val \| hold | test corr to W8 |
|---|---|---|---|---|---|---|---|---|
| W8 | 8 | 0.40157 | 0.32456 | 0 | 0 | 0 | 0.40136 \| 0.32436 | 1 |
| W8 + kd05t25w192b3 x2 | 10 | 0.40199 | 0.32466 | +0.00043 ± 0.00027 | +0.00010 ± 0.00015 | −0.00003 ± 0.00015 | 0.40152 \| 0.32439 | 0.9996 |
| W8 + kw4 x8 | 16 | 0.40140 | 0.32452 | −0.00017 ± 0.00035 | −0.00004 ± 0.00019 | +0.00004 ± 0.00019 | 0.40143 \| 0.32458 | 0.9989 |
| all 38 d192 students | 38 | 0.40128 | 0.32450 | −0.00029 ± 0.00038 | −0.00006 ± 0.00021 | −0.00021 ± 0.00020 | 0.40147 \| 0.32448 | 0.9986 |
| W11 | 11 | 0.40127 | 0.32448 | −0.00030 ± 0.00027 | −0.00007 ± 0.00016 | −0.00007 ± 0.00016 | 0.40154 \| 0.32440 | 0.9994 |
| W8 + kd03g2w192 | 12 | 0.40146 | 0.32448 | −0.00011 ± 0.00030 | −0.00008 ± 0.00017 | −0.00004 ± 0.00017 | 0.40134 \| 0.32440 | 0.9992 |
| W8 + kd05g2w192 | 12 | 0.40165 | 0.32445 | +0.00009 ± 0.00028 | −0.00010 ± 0.00016 | −0.00009 ± 0.00015 | 0.40143 \| 0.32444 | 0.9994 |
| W8 + kx05 | 12 | 0.40167 | 0.32444 | +0.00010 ± 0.00030 | −0.00012 ± 0.00016 | −0.00024 ± 0.00016 | 0.40138 \| 0.32434 | 0.9994 |
| W8 + kd03 | 12 | 0.40126 | 0.32445 | −0.00030 ± 0.00031 | −0.00011 ± 0.00018 | −0.00019 ± 0.00018 | 0.40124 \| 0.32431 | 0.9992 |
| W8 + kd05 | 12 | 0.40155 | 0.32434 | −0.00001 ± 0.00028 | −0.00022 ± 0.00017 | −0.00027 ± 0.00017 | 0.40129 \| 0.32426 | 0.9994 |
| W8 + kd05t25 | 12 | 0.40150 | 0.32431 | −0.00007 ± 0.00030 | −0.00025 ± 0.00016 | −0.00019 ± 0.00017 | 0.40144 \| 0.32423 | 0.9993 |
| W8 + kd08 | 12 | 0.40141 | 0.32432 | −0.00015 ± 0.00031 | −0.00023 ± 0.00016 | −0.00016 ± 0.00017 | 0.40108 \| 0.32428 | 0.9993 |
| **S16** (W8 + kd05 + kd05t25) | 16 | 0.40172 | 0.32428 | +0.00015 ± 0.00033 | −0.00028 ± 0.00019 | −0.00022 ± 0.00019 | 0.40134 \| 0.32417 | 0.9989 |
| W8 + gen-2 x8 | 16 | 0.40168 | 0.32426 | +0.00011 ± 0.00032 | −0.00029 ± 0.00018 | −0.00008 ± 0.00018 | 0.40140 \| 0.32446 | 0.9990 |
| S20 (S16 + kd08) | 20 | 0.40132 | 0.32416 | −0.00025 ± 0.00037 | −0.00040 ± 0.00021 | −0.00028 ± 0.00021 | 0.40114 \| 0.32413 | 0.9985 |
| W8 + gen-2 x8 + kx x8 | 24 | 0.40151 | 0.32414 | −0.00006 ± 0.00035 | −0.00042 ± 0.00020 | −0.00039 ± 0.00019 | 0.40133 \| 0.32437 | 0.9988 |
| W8 + kd10 | 12 | 0.40120 | 0.32407 | −0.00037 ± 0.00033 | −0.00048 ± 0.00018 | −0.00039 ± 0.00019 | 0.40091 \| 0.32410 | 0.9991 |
| W8 + base x6 | 14 | 0.40081 | 0.32406 | −0.00075 ± 0.00037 | −0.00049 ± 0.00021 | −0.00044 ± 0.00021 | 0.40113 \| 0.32418 | 0.9986 |
| all 63 tf students | 63 | 0.40107 | 0.32428 | −0.00049 ± 0.00041 | −0.00028 ± 0.00023 | −0.00032 ± 0.00022 | 0.40127 \| 0.32427 | 0.9981 |
| N8 (narrow only) | 8 | 0.40119 | 0.32372 | −0.00038 ± 0.00049 | −0.00083 ± 0.00028 | −0.00073 ± 0.00028 | 0.40094 \| 0.32365 | 0.9955 |
| W8 + pop1kd x4 | 12 | 0.40177 | 0.32342 | +0.00020 ± 0.00046 | −0.00113 ± 0.00030 | −0.00115 ± 0.00030 | 0.40145 \| 0.32289 | 0.9980 |
| W8 + C3 members (member-equal) | 36 | 0.40145 | 0.32361 | −0.00011 ± 0.00048 | −0.00095 ± 0.00029 | | 0.40084 \| 0.32305 | 0.9961 |
| C3 TO | 28 | 0.40103 | 0.32306 | −0.00054 ± 0.00056 | −0.00149 ± 0.00034 | | 0.40046 \| 0.32247 | 0.9936 |

- No union is ≥ +2 SE on either set. On the holdout only one row is positive, and it is zero at @last.
- The expectation is a first-order law: W8 itself is +0.0002 above it on both sets, the other student unions are between −0.0003 and +0.0005 val | −0.0002 and +0.0003 hold, and mixed-family pools exceed it by up to 0.0006 (C3). Relative to it no student union stands out.
- The eight capacity-lever students closed at member level (kw4) form a pool equal to W8 (−0.00015 ± 0.00053 | +0.00005 ± 0.00029). At 2-seed pool level every kw arm is ≤ kd05w192 on the same seeds; none was mis-closed (kw_drop0 2-pool −0.00026 ± 0.00075 | 0.00000 ± 0.00043; kw_hl2 −0.00031 | −0.00102 ± 0.00039; kd05t25w256 −0.00023 | −0.00084 ± 0.00042).
- Pooling form: mean raw logit = mean z (+0.00000 ± 0.00009 | −0.00003 ± 0.00004); mean within-feed rank is worse (−0.00094 ± 0.00030 | −0.00056 ± 0.00017).

## 5. Weights and selection across the two witnesses

lambda · z(W8 pool) + (1 − lambda) · z(other pool):

| other pool | lambda | d val | d hold | d hold @last |
|---|---|---|---|---|
| N8 (narrow) | 0.5 (= S16) | +0.00016 ± 0.00033 | −0.00028 ± 0.00019 | −0.00023 ± 0.00019 |
| N8 | 0.7 | +0.00006 ± 0.00025 | −0.00022 ± 0.00015 | −0.00018 ± 0.00015 |
| N8 | 0.9 | +0.00021 ± 0.00014 | −0.00008 ± 0.00008 | −0.00001 ± 0.00008 |
| gen-2 x8 | 0.5 | +0.00011 ± 0.00032 | −0.00029 ± 0.00018 | −0.00007 ± 0.00018 |
| gen-2 x8 | 0.8 | +0.00024 ± 0.00019 | +0.00017 ± 0.00011 | +0.00006 ± 0.00011 |
| kx05 | 0.7 | +0.00013 ± 0.00029 | −0.00012 ± 0.00015 | −0.00021 ± 0.00015 |
| base x6 | 0.8 | +0.00011 ± 0.00025 | −0.00030 ± 0.00014 | −0.00026 ± 0.00014 |
| C3 TO | 0.5 | −0.00010 ± 0.00038 | −0.00044 ± 0.00023 | |
| C3 TO | 0.7 | +0.00016 ± 0.00029 | −0.00027 ± 0.00017 | |
| C3 TO | 0.9 | −0.00000 ± 0.00015 | **−0.00030 ± 0.00009** | |
| F7x + F7x70 | 0.9 | +0.00004 ± 0.00017 | −0.00023 ± 0.00010 | |

No weight is ≥ +2 SE on either set. Weighting wide above narrow moves monotonically toward W8 on the holdout; lambda = 1 (drop the narrow students) is the holdout optimum.

Group-level greedy forward selection with replacement (candidate = a group's pool):

| candidate set | fitted on | steps | in-sample d | other witness d | weights chosen |
|---|---|---|---|---|---|
| d192 groups | holdout | 12 | +0.00032 ± 0.00025 | val −0.00024 ± 0.00044 | kw4 5, kd03g2 4, b3 2, kd05g2 1 |
| d192 groups | holdout | 24 | +0.00028 ± 0.00024 | val −0.00008 ± 0.00043 | kw4 9, kd03g2 7, b3 4, kd05g2 3, t25w192 1 |
| d192 groups | val | 12 | +0.00053 ± 0.00034 | hold −0.00004 ± 0.00019 | kd05g2 3, t25w192 3, kx05 2, four others 1 |
| d192 groups | val | 24 | +0.00051 ± 0.00033 | hold −0.00009 ± 0.00019 | spread over 8 groups |
| all tf students | holdout | 12, 24 | identical path to the d192 run | | no d128 group is ever chosen |
| all tf students | val | 24 | +0.00087 ± 0.00043 | hold −0.00043 ± 0.00024 | w256 5, b3 5, narrow groups enter |
| students + hard-label groups | holdout | 12, 24 | identical path to the d192 run | | no hard-label group is ever chosen |
| students + hard-label groups | val | 12 | +0.00085 ± 0.00046 | **hold −0.00109 ± 0.00029** | pop1 2, w256 2, kw4 2, ... |
| students + hard-label groups | val | 24 | +0.00085 ± 0.00044 | **hold −0.00095 ± 0.00028** | pop1 4, w256 4, kd05t25 3, ... |

Stability (10 random halves of the fitting set, 12 steps): the selection is negative even on the other half of its own witness (holdout-fitted −0.00017, sd 0.00024; val-fitted −0.00050, sd 0.00031). Member-equal vs group-equal weights over fixed sets differ by ≤ 0.00024 on both sets. There is nothing to select.

## 6. C3 train-only pool with the students

C3 TO is 0.40103 | 0.32306. W8 − C3 = +0.00054 ± 0.00056 val (M≥50 +0.00013, M<50 +0.00113) | **+0.00149 ± 0.00034 hold**. About half of the holdout gap is the pop1 members, which the in-period holdout penalises (G3 TO 0.32379, W8 − G3 = +0.00076 ± 0.00033). Any admixture lowers the holdout (§5) and none raises val by more than +0.00016 ± 0.00029. At final-file level, 0.7 W8 + 0.3 C3 per branch reads −0.00014 ± 0.00029 and 0.5/0.5 (the built W8C5) −0.00027 ± 0.00039. **C3 does not help the student pool on either witness.**

## 7. Final-file level and public evidence

Val only (the holdout cannot see the FT branch). W8 final 0.40165 (FT branch 0.40087, TO branch 0.40157, branch correlation 0.9907).

| final file | val | d vs W8 final | test corr to W8 final |
|---|---|---|---|
| S16 (as submitted) | 0.40173 | +0.00008 ± 0.00034 | 0.9989 |
| S20 | 0.40114 | −0.00051 ± 0.00037 | 0.9986 |
| C3 | 0.40101 | −0.00064 ± 0.00055 | 0.9938 |
| FT W8 + TO W11 | 0.40170 | +0.00005 ± 0.00020 | 0.9999 |
| FT W8 + TO (W8 + b3 x2) | 0.40200 | +0.00035 ± 0.00019 | 0.9999 |
| FT W8 + TO (W8 + kw4 x8) | 0.40146 | −0.00019 ± 0.00026 | 0.9997 |
| FT W8 + TO all 38 d192 | 0.40148 | −0.00017 ± 0.00028 | 0.9996 |
| FT S16 + TO W8 | 0.40174 | +0.00009 ± 0.00024 | 0.9997 |
| FT C3 + TO W8 | 0.40136 | −0.00029 ± 0.00039 | 0.9983 |
| W8, w_ft 0.3 | 0.40193 | +0.00028 ± 0.00026 | 0.9997 |
| W8 TO only (w_ft 0) | 0.40157 | −0.00009 ± 0.00043 | 0.9979 |

- An asymmetric file (FT branch = the existing W8 fine-tunes, TO branch = a larger pool) reads the same as a symmetric one. **New extension seeds do not need a fine-tune to enter a file.**
- Inputs for the FT lens. Member-level FT − TO on val: 8 wide students −0.00043 ± 0.00041 (M<50 −0.00114 ± 0.00067), 8 narrow −0.00001 ± 0.00039, 28 hard-label members +0.00073 ± 0.00032; students − hard-label **−0.00095 ± 0.00024**. At pool level the FT branch is 0.0006-0.0009 below the TO branch for W8, S16 and C3 alike, and the w_ft curve is flat (best point +0.00028 ± 0.00026). The private-mix reweighting still favours keeping FT (TO-only −0.00104 ± 0.00067), so nothing here argues for dropping it.

Public vs the two witnesses (delta vs W8):

| file | final val | TO holdout | public | paired public SE (from the val paired sd) |
|---|---|---|---|---|
| S16 | +0.00008 ± 0.00034 | −0.00028 ± 0.00019 | −0.00041 | ~0.00044 |
| C3 | −0.00064 ± 0.00055 | −0.00149 ± 0.00034 | −0.00103 | ~0.00071 |

The holdout has the public sign in both contrasts; val has it in one. Inverse-variance pooling of the three independent sets: S16 − W8 = −0.00022 ± 0.00016 (1.4 SE), C3 − W8 = −0.00122 ± 0.00027. The generation step is solid. "W8 > S16" is a weak preference; it is enough to keep the narrow students out, not enough to claim they hurt.

## 8. Recipes for tomorrow (ordered by the pre-declared rule)

No recipe is "ahead" (holdout ≥ +2 SE). All five below tie with W8 (holdout within 1 SE) and pass the val veto; the tie-break (more d192 members, then lower test correlation) gives the order. FT branch = the existing W8 fine-tunes in every row, weight 0.5 / 0.5, member-equal TO pool.

| # | TO-branch members (all `w5_e10_<arm>_s<seed>`) | N | TO hold (d; @last) | TO val (d) | final val (d) | test corr TO / final |
|---|---|---|---|---|---|---|
| 1 | every d192 student: W8 + kd05w192 s955 + kd05t25w192 s955, s957 + kd05g2w192, kd03g2w192, kx05, kx08 (s951-954) + kw_drop0, kw_e20, kw_hl2, kw_ff4, kd05t25w192b3 (s951-952) + kd08w192 s951 | 38 | 0.32450 (−0.00006 ± 0.00021; −0.00021 ± 0.00020) | 0.40128 (−0.00029 ± 0.00038) | 0.40148 (−0.00017 ± 0.00028) | 0.9986 / 0.9996 |
| 2 | W8 + kw_drop0, kw_e20, kw_hl2, kw_ff4 (s951-952) | 16 | 0.32452 (−0.00004 ± 0.00019; +0.00004 ± 0.00019) | 0.40140 (−0.00017 ± 0.00035) | 0.40146 (−0.00019 ± 0.00026) | 0.9989 / 0.9997 |
| 3 | W8 + kd03g2w192 s951-954 | 12 | 0.32448 (−0.00008 ± 0.00017; −0.00004 ± 0.00017) | 0.40146 (−0.00011 ± 0.00030) | 0.40162 (−0.00003 ± 0.00024) | 0.9992 / 0.9998 |
| 4 | W8 + kd05g2w192 s951-954 | 12 | 0.32445 (−0.00010 ± 0.00016; −0.00009 ± 0.00015) | 0.40165 (+0.00009 ± 0.00028) | 0.40159 (−0.00006 ± 0.00021) | 0.9994 / 0.9999 |
| 5 | W8 + kd05t25w192b3 s951-952 | 10 | 0.32466 (+0.00010 ± 0.00015; −0.00003 ± 0.00015) | 0.40199 (+0.00043 ± 0.00027) | 0.40200 (+0.00035 ± 0.00019) | 0.9996 / 0.9999 |
| ref | **W8** = kd05w192 + kd05t25w192, s951-954 | 8 | 0.32456 | 0.40157 | 0.40165 | 1 / 1 |

- W11 (the partial W16) is dropped by the letter of the val veto (TO val −0.00030 ± 0.00027); its final reads +0.00005 ± 0.00020. This is noise in both directions.
- Recipe 5 has the best val and holdout, both inside noise (val-only gain below +0.0005, holdout sign flips at @last).
- **Recommendation: keep W8 (or W16 when complete) as the level file and submit none of the five for level.** With test correlation ≥ 0.9996 they cannot move public by a resolvable amount.

Outside the ranking, one holdout-blind probe exists with files on disk: **W8P = 0.7 z(W8) + 0.3 z(pop1 x8) per branch** (pop1 share as in C3). Final val +0.00039 ± 0.00041 (M≥50 +0.00096 ± 0.00047, M<50 −0.00043 ± 0.00075; Tuesday +0.00127 ± 0.00060, Thursday −0.00056 ± 0.00057), private-mix +0.00077 ± 0.00066, TO holdout −0.00117 ± 0.00029, test corr 0.9985. By the holdout-first rule it is rejected. The only argument for it is the pop1 precedent (same val / private-mix / holdout pattern, then +0.0005 on public in three C3-type builds). The Tuesday-only gain contradicts the staleness mechanism, and one public reading has an SE of about 0.0005. I would not spend a slot on it ahead of any candidate with a ceiling-shift reading.

## 9. Future ideas from this lens

1. **Read new arms at the asymptote, not at member level.** Tool: `pc_arm_readout.py <arm> <seeds> [ref arm]` prints member, n-pool and asymptote deltas vs the reference arm on the same seeds (val, holdout, holdout @last), what the arm adds to W8, and the seed / arm correlation split. `pc_pool_vs_w8.py <variants>` reads [W8 + any members] vs W8 at TO and final level (use it for W16 when the five remaining seeds finish). Proposed reading: a ceiling shift needs asymptote ≥ +2 SE on the holdout and ≥ 0 on val. Power: the 4-seed asymptote SE is 0.00065 val | 0.00037 hold, the 2-seed one 0.0011 | 0.0006. A 2-seed screen can only see shifts ≥ +0.0012 on the holdout; the current member-level screen (SE 0.0006 | 0.00035) is equally blind to +0.0005.
2. **Predictions for the running arms** (to be checked with the tool):
   - kdmixw192: teacher identity adds no diversity and teacher quality between 0.3229 and 0.3246 holdout does not move the student asymptote. Expect a tie with kd05w192.
   - kd05w224: d256 read −0.00045 ± 0.00108 | −0.00106 ± 0.00061 at the asymptote vs kd05w192. Expect ≤ 0.
   - kd05w192gru: the rule asks [W8 + gru4] ≥ +0.0003 on the holdout. Four more plain seeds are worth +0.00009, so GRU students must correlate with the transformer students at about ≤ 0.9675 on the holdout (within-family 0.9735) at equal member strength. Hard-label GRU vs transformer families differ by 0.005-0.009, so this is borderline.
   - W16: +0.0001 expected. After it, stop adding seeds: infinity − W8 is +0.0002.
3. **Only information-diverse members can lift the pool.** Eight wide constructions with correlation 0.955-0.972 share one asymptote (4-seed arms individually, the 2-seed kw arms as a group: kw4 x8 − W8 = −0.00005 ± 0.00058 | +0.00010 ± 0.00032), so decorrelation from noise (kd weight, dropout, generation, teacher) is worth nothing. The only student group decorrelated from W8 by its inputs is the pop1 view (0.960-0.964 vs 0.972). It is weaker by 0.0011 val per member and the holdout cannot judge it. A wide pop1 arm (pop1 view, d192, kd 0.5, 4 seeds) is the one composition bet left; its closure today used the val bar and the in-period holdout that the freeze protocol calls biased for this family, on narrow students only. Honest prior for ≥ +0.001 public: about 0.05.
4. **A small-feed second witness.** The holdout has no M<50 feed; 39% of test feeds are M<50 and val's M<50 slice has an SE of 0.0005-0.0009 for these contrasts. Sub-sampling the candidates of the 30K holdout feeds to the test M<50 size distribution (positives kept with probability m/M, `ctx_nd` features recomputed on the sub-set, inference only) would give about 30K small-feed evaluations for every existing checkpoint. It assumes that small val/test feeds are random subsets of longer lists, the same premise as the kwsub arms.
5. **The hedge pick.** Pick 2 in CURRENT_STATUS §1 is still Pall (public 0.392436, 0.0020 below W8). C3 is 0.0010 below W8 on public at test correlation 0.9938, a different generation with no distillation. If pick 1 becomes W8 or W16, C3 is the stronger structural hedge. This is the owner's decision.

## 10. Limits

- The holdout is in-period, M≥50, TO only: blind to the FT branch, to small feeds and biased against popularity-light members. Every "holdout says no" above is subject to that; the pop1 rows are the only place where it matters.
- The asymptote estimator assumes NDCG(n) = a − b/n. It is verified on W8 subsets; for 2-seed arms it has a large SE.
- The extrapolation to 16 and 32 assumes exchangeable seeds. The three finished extension seeds are individually normal (member 0.39965 | 0.32246) but pull the pool slightly down; with 11 seeds the asymptote estimate is 0.40145-0.40155 | 0.32463-0.32467.
- Not checked: the five extension seeds still training, the GRU / kdmix / w224 / kwsub / kwT arms (no prediction files yet), FT of any member beyond the 20 fine-tuned students, per-day holdout slices.
- Multiple looks: about 40 pools and 60 weightings were read. The single best val row (+0.00043 ± 0.00027) and the best holdout row (+0.00010 ± 0.00015) are what the maximum of that many null draws looks like.

## 11. Files

| file | content |
|---|---|
| `pc_common.py`, `pc_cache.py` | loaders, fast NDCG, z cache (`cache/`, 208 arrays x 3 splits, 7 GB; safe to delete, rebuilt in 20 s) |
| `pc_members.py` → `out/members.log` | member / group table, correlation matrices |
| `pc_unions.py` → `out/unions.log` | law fit, unions, weight grids, C3 mixes |
| `pc_saturation.py` → `out/saturation.log` | subset curves, extrapolation, W11 check, leave-one-out |
| `pc_forward.py` → `out/forward.log` | cross-witness forward selection, stability |
| `pc_final.py` → `out/final.log` | FT vs TO, finals, w_ft curve, 2-seed arm reads |
| `pc_extra.py` → `out/extra.log` | asymptote contrasts, pooling form, public table |
| `pc_hedge.py`, `pc_probe.py` → `out/hedge.log`, `out/probe.log` | pop1 / TO-only candidates with private-mix |
| `pc_recipes.py` → `out/recipes.log`, `out/recipes.json` | the recipe table with member lists |
| `pc_arm_readout.py` | read-out tool for new arms (member, n-pool, asymptote vs a reference arm; no cache needed) |
| `pc_pool_vs_w8.py` | [W8 + listed members] vs W8: TO val / holdout / @last, final with the W8 FT branch, test correlations (reproduces the W11 row) |
| `out/PREDECLARED_rule.txt` | the ordering rule and its timestamp |
