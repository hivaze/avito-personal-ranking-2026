# RECHECK 2026-11-14, lens `lb_stats`: leaderboard statistics and private-leaderboard risk

CPU only, read-only on every prediction file. Scripts and logs: `nn_x/w12/recheck/lb_stats/` (`lib.py`, `s1_build.py` .. `s7_eiv_check.py`, `out/*.log`).
All 14 modern scored files + v5/v10 were rebuilt per feed (0.5 z(FT) + 0.5 z(TO)); every rebuilt val mean equals the recorded one to 1e-5 (`out/s1_build.log`).
"Measured" = computed here from per-feed arrays; "inferred" = model-based. SE of a public / private delta = per-feed sd of the paired val delta under the public / private M mix, divided by sqrt(8,562) / sqrt(20,033).
Check of that noise model: label-free disagreement of every pair is the same on val, public and private (ratios 1.01-1.02), and over 120 file pairs the rms z of (public delta - val delta) is 1.09 (near copies), 0.91 (corr 0.99-0.995), 0.78 (old generations).

## 0. Summary

1. **W8 is the best-supported file; W8 > S16 on public is noise.** SE of the public delta W8 - S16 is 0.00043; the observed +0.00041 is 0.95 SE (val predicted -0.00008 ± 0.00034, holdout +0.00028 ± 0.00019). W8 > G3 is real (three independent readings of about 2.3 SE each: val, holdout, public). W8 > C3 is likely, not proven (val + public: +0.00079 ± 0.00043).
2. **Val deltas among modern files transfer at k = 0.60 (1 SE 0.35-0.90, 95% 0.10-1.33), not at 1 and not at 0.** The shortfall sits in the 11-12 files (m02 re-cools, val-chosen pools: k = 0.18); for the 11-13/14 files, whose checkpoints were chosen on the holdout, k = 0.88 (0.38-1.62). Lineage, not the 0.995 correlation threshold, separates the two.
3. **No reweighting or slice of val beats plain val** as a predictor of public deltas (public feed-size mix, M<50, M>=50, Tuesday, Thursday). The holdout alone is now the worst single predictor (it mis-read the pop1 pools); val and holdout together are best (public delta ≈ 0.4 val delta + 0.4 holdout delta).
4. **Private forecast:** W8 0.3971, S16 0.3967-0.3970, C3 0.3961-0.3966, each ± 0.004 common to all files and all teams. P(W8 > C3 on private) 0.81-0.91, P(W8 > S16) 0.61-0.69, P(W8 > G3) 0.93-0.99.
5. **Gap to #1 (0.00377) is 2.1-3.6 SE of cross-team public noise**; expected true lead 0.0014-0.0034; P(no true lead) below 2% under a flat prior, up to 10% if the contenders' true spread is only 0.001. Ranks 2-5 are a lottery: P(we finish top 3 on private) 0.56-0.67; every +0.0005 of true gain adds 0.10-0.15.
6. **Picks:** pick 1 = W8. The second pick is worth at most +0.0001 whatever is chosen; the old structural hedges are worth less (C3 +0.00003..+0.00007, G3 +0.00000..+0.00002) than S16 (+0.00007..+0.00011). The freeze protocol applied by the letter today gives W8 + S16.
7. **Probes:** one slot per new pick candidate (validity and level). No slot can order near copies; a pool-size extension such as W16 will score below W8 on public with probability about 0.34 even if it is truly better.

## 1. Val -> public with all submissions

### 1.1 Level line (measured)

| fit | intercept | slope | residual sd | LOO rmse |
|---|---|---|---|---|
| all 18 files | +0.0074 | 0.963 ± 0.023 | 0.00060 | 0.00067 |
| 16 files, without W8 / S16 | +0.0081 | 0.961 ± 0.025 | 0.00063 | 0.00072 |
| modern 14 only | +0.1407 | 0.630 ± 0.158 | 0.00047 | 0.00052 |

Out of sample (line fitted without them): W8 +0.00041, S16 -0.00008. The line is unchanged by today's files.
M-reweighted val (`src/calib.py`) minus public over the modern 14: mean +0.00054, sd 0.00054.
The sampling SE of an absolute level is 0.0021 (val), 0.0026 (public), 0.0017 (private): per-feed NDCG sd is 0.247. The SE of "public level minus M-reweighted val level" is therefore 0.0033; the observed +0.0005 offset is a 0.16-SE draw, not a measured property. Level statements such as "private ≈ 0.3949 ± 0.001" (VAL_PUBLIC_1113 §1.1) understate the interval by a factor of 3-4.

### 1.2 Per-file table (measured; deltas vs G3, paired)

| file | val | holdout (TO) | public | val Δ ± SE | holdout Δ ± SE | public Δ | SE public Δ | corr to G3 |
|---|---|---|---|---|---|---|---|---|
| Cd | 0.39901 | 0.32319 | 0.392080 | -0.00131 ± 0.00049 | -0.00060 ± 0.00027 | -0.00074 | 0.00062 | 0.9960 |
| Pall | 0.39896 | 0.32335 | 0.392436 | -0.00136 ± 0.00043 | -0.00045 ± 0.00025 | -0.00038 | 0.00055 | 0.9972 |
| PF7x | 0.39974 | 0.32360 | 0.392723 | -0.00058 ± 0.00039 | -0.00020 ± 0.00022 | -0.00010 | 0.00050 | 0.9980 |
| G3 | 0.40032 | 0.32379 | 0.392820 | 0 | 0 | 0 | - | 1 |
| M2 | 0.40094 | 0.32361 | 0.392434 | +0.00062 ± 0.00034 | -0.00018 ± 0.00019 | -0.00039 | 0.00043 | 0.9991 |
| S2 | 0.40088 | n/a | 0.392473 | +0.00056 ± 0.00028 | n/a | -0.00035 | 0.00035 | rank-valued |
| TOonly | 0.40035 | = G3 | 0.392346 | +0.00003 ± 0.00039 | 0 | -0.00047 | 0.00050 | 0.9980 |
| Q1a | 0.40068 | 0.32327 | 0.392886 | +0.00036 ± 0.00030 | -0.00052 ± 0.00021 | +0.00007 | 0.00039 | 0.9995 |
| Q2a | 0.40061 | 0.32378 | 0.393205 | +0.00029 ± 0.00031 | -0.00002 ± 0.00021 | +0.00039 | 0.00039 | 0.9995 |
| C1 | 0.40089 | 0.32306 | 0.393394 | +0.00057 ± 0.00040 | -0.00073 ± 0.00029 | +0.00057 | 0.00050 | 0.9987 |
| C2 | 0.40051 | = G3 | 0.392754 | +0.00018 ± 0.00012 | 0 | -0.00007 | 0.00015 | 1.0000 |
| C3 | 0.40101 | 0.32306 | 0.393454 | +0.00069 ± 0.00040 | -0.00073 ± 0.00029 | +0.00063 | 0.00051 | 0.9986 |
| W8 | 0.40165 | 0.32456 | 0.394486 | +0.00133 ± 0.00058 | +0.00076 ± 0.00033 | +0.00167 | 0.00074 | 0.9935 |
| S16 | 0.40173 | 0.32428 | 0.394078 | +0.00141 ± 0.00053 | +0.00049 ± 0.00030 | +0.00126 | 0.00067 | 0.9954 |

Omnibus tests on the 14 public scores (13 df, full noise covariance): "all files equal" chi2 14.8 (p 0.32); "public Δ = val Δ" chi2 11.0 (p 0.61). Fourteen files still cannot reject either extreme as a whole; the one-parameter slope below is the sharper statement.

### 1.3 Transfer factor k in public Δ = k × true offline Δ (inferred; errors in both variables)

Two estimators: "Deming" corrects for the sampling noise of the offline delta (the factor a noise-free val delta would transfer at); "predictive" is the slope to apply to an observed delta (it includes regression to the mean). p(k=1) is a parametric bootstrap of the Deming estimate under k = 1 with a true spread of 0.0004 / 0.0008 between files (`out/s7_eiv_check.log`).

| block | predictor | Deming k (1 SE) | 95% | predictive k | p(k=1) |
|---|---|---|---|---|---|
| modern 14 | val | 0.60 (0.35, 0.90) | 0.10, 1.33 | 0.40 | 0.020 / 0.003 |
| 11 without M2, S2, TOonly | val | 0.58 (0.35, 0.88) | 0.12, 1.25 | 0.43 | 0.035 / 0.003 |
| 11-12 files: Cd, Pall, PF7x, G3, M2, S2, TOonly | val | 0.18 (-0.12, 0.48) | -0.42, 0.85 | 0.12 | 0.052 / 0.000 |
| 11-13/14 files: G3, C2, Q1a, Q2a, C1, C3, W8, S16 | val | 0.88 (0.38, 1.62) | -0.05, 3.8 | 0.45 | 0.32 / 0.18 |
| G3, C3, W8, S16 | val | 1.03 (0.53, 1.90) | 0.08, >4 | 0.73 | 0.57 |
| modern 13 (no S2) | holdout | 0.73 (0.38, 1.15) | 0.02, 1.68 | 0.58 | 0.045 / 0.005 |
| 11-12 files (no S2) | holdout | predictive 0.79 (0.12, 1.54) | | 0.79 | 0.76 (LR) |
| 11-13/14 files | holdout | 0.38 (0.05, 0.78) | -0.30, 1.25 | 0.33 | 0.015 / 0.000 |

Reading:
- The 11-13 conclusion "near-copy val deltas do not transfer; the holdout is the lens that transfers" described the 11-12 files. For the 11-13/14 files it reverses: val transfers at about 0.9 and the holdout at about 0.4, because the holdout read the pop1 pools at -0.0007 where public read +0.0006.
- Each lens failed exactly where it had a known bias: val for components chosen on val, the in-period holdout for popularity-light members. Files where both agree (W8, S16 vs G3) transferred fully: val +0.0013 / +0.0014, holdout +0.0008 / +0.0005, public +0.0017 / +0.0013.
- For planning, an observed val gain of +0.001 in a pool predicts about +0.0004..+0.0005 on public; half of the discount is val's own sampling noise.

### 1.4 Which offline number predicts public deltas best (measured; leave-one-file-out rmse of the public delta, slope refitted)

| predictor | modern 14: corr / LOO rmse | 11 unselected: corr / LOO rmse |
|---|---|---|
| no information | - / 0.00071 | - / 0.00074 |
| val, uniform | +0.76 / 0.00052 | +0.92 / 0.00034 |
| val, public M mix (`src/calib.py`) | +0.76 / 0.00052 | +0.91 / 0.00035 |
| val, M<50 only | +0.60 / 0.00062 | +0.90 / 0.00037 |
| val, M>=50 only | +0.73 / 0.00052 | +0.79 / 0.00049 |
| val, Tuesday only | +0.67 / 0.00057 | +0.81 / 0.00047 |
| val, Thursday only | +0.76 / 0.00050 | +0.93 / 0.00031 |
| val FT half / TO half (11 files) | +0.92 / 0.00034 each | same |
| holdout (TO pool) | +0.55 / 0.00072 (13 files) | +0.64 / 0.00067 |

- Reweighting val to the public feed-size mix changes nothing (the M mixes differ by 2 points).
- Two-component fits (public Δ = k1 × share × val Δ on part 1 + k2 × share × val Δ on part 2), modern 14: M<50 k1 = 0.10 (-0.22, 0.43) vs M>=50 k2 = 0.65 (0.35, 1.00), difference p 0.26; Tuesday 0.50 vs Thursday 0.30, p 0.78. On the 11 unselected files both differences vanish (0.35 vs 0.52; 0.45 vs 0.43). No slice is a better lens; M<50 is the weakest hint of a worse one.
- Val + holdout jointly (public noise only, 13 files): k_val 0.39 ± 0.23, k_holdout 0.41 ± 0.31, chi2 7.9 / 10. This is the best descriptive predictor: public Δ ≈ 0.4 val Δ + 0.4 holdout Δ. It gives W8 - C3 +0.00085 (actual +0.00103), S16 - C3 +0.00077 (+0.00062), W8 - S16 +0.00008 (+0.00041).

### 1.5 Pairs (measured; pairs share files, so this is descriptive)

| pair class (test score corr) | pairs | slope through origin | sign agreement | sign agreement where val Δ > 2 SE | mean SE public Δ |
|---|---|---|---|---|---|
| near copies, >= 0.995 | 71 | 0.46 | 0.73 | 0.86 (28 pairs) | 0.00052 |
| 0.99 - 0.995 (all involve W8, S16 or Cd) | 18 | 0.81 | 0.94 | 0.92 (12 pairs) | 0.00074 |
| < 0.99 (v5, v10 vs modern) | 31 | 0.91 | 1.00 | 1.00 (29 pairs) | 0.00136 |

## 2. W8 and S16 on public (measured unless stated)

| contrast | val Δ ± SE | M<50 | M>=50 | Tue | Thu | holdout (TO) | SE public Δ | public Δ | z vs 0 | z vs val |
|---|---|---|---|---|---|---|---|---|---|---|
| W8 - C3 | +0.00064 ± 0.00055 | +0.00144 ± 0.00101 | +0.00009 ± 0.00061 | -0.00016 | +0.00152 | +0.00149 ± 0.00034 | 0.00070 | +0.00103 | 1.47 | +0.44 |
| S16 - C3 | +0.00072 ± 0.00051 | +0.00196 ± 0.00095 | -0.00014 ± 0.00056 | -0.00011 | +0.00163 | +0.00121 ± 0.00031 | 0.00065 | +0.00062 | 0.96 | -0.11 |
| W8 - S16 | -0.00008 ± 0.00034 | -0.00052 ± 0.00062 | +0.00023 ± 0.00038 | -0.00005 | -0.00011 | +0.00028 ± 0.00019 | 0.00043 | +0.00041 | 0.95 | +0.89 |
| W8 - G3 | +0.00133 ± 0.00058 | +0.00201 ± 0.00108 | +0.00086 ± 0.00063 | +0.00059 | +0.00214 | +0.00076 ± 0.00033 | 0.00074 | +0.00167 | 2.26 | +0.36 |

- **W8 > S16:** per-feed sd of the paired delta is 0.040 and 48% of feeds have identical NDCG, so the SE of the public delta is 0.00043. From val alone P(public W8 > S16) = 0.44 and P(gap >= +0.00041) = 0.19. The three-file test of the public deltas against the val deltas has chi2 0.80 on 2 df. Noise is enough.
- Combined true delta W8 - S16: +0.00011 ± 0.00027 (val + public), +0.00022 ± 0.00015 if the holdout is also taken at face value. A slight edge for W8, the pool with fewer members.
- **W8 vs C3:** public +0.00103 is 1.5 SE; val + public give +0.00079 ± 0.00043. The holdout's +0.00149 is inflated by its bias against C3's pop1 members; against G3 the holdout (+0.00076) and val M>=50 (+0.00086) agree.
- The KD val gain over C3 sits in M<50 and on Thursday; M>=50 is flat. Neither concentration is significant (1.2 and 1.5 SE).
- W8's public score is +0.0004 above its level line and S16's -0.0001 below: part of W8's lead over everything is a favourable draw. New files should be expected about 0.0002-0.0004 below W8's public even when equal.

## 3. Private-leaderboard forecast (inferred)

Composition (measured): public = Saturday 6,050 + Sunday before 11:32 MSK 2,512; private = Sunday 4,855 + Monday 7,578 + Tuesday 7,600. M<50 share 0.387 / 0.392, mean M 103.6 / 100.7. Cold share >= 0.45: val 0.08, public 0.22, private 0.33.

| file | public | private level forecast | private Δ vs W8 (mean ± sd) | P(beats W8) | P(best of the 7 built files) |
|---|---|---|---|---|---|
| W8 | 0.394486 | 0.3971 | 0 | - | 0.42-0.54 |
| S16 | 0.394078 | 0.3967-0.3970 | -0.00011..-0.00018 ± 0.00038 | 0.31-0.39 | 0.17-0.24 |
| W8C5 (unscored) | - | 0.3967 | -0.00030..-0.00039 ± 0.00046 | 0.19-0.26 | 0.09-0.10 |
| C3 | 0.393454 | 0.3961-0.3966 | -0.00055..-0.00079 ± 0.00062 | 0.09-0.19 | 0.03-0.07 |
| G3 | 0.392820 | 0.3954-0.3962 | -0.00092..-0.00146 ± 0.00065 | 0.01-0.07 | 0.00-0.02 |

- Level = public + 0.0026 (feed-size mix, `src/calib.py`). Interval ± 0.004 (1 SE): 0.0032 sampling of two disjoint feed sets, plus an unmeasured weekday effect (holdout weekend minus weekday level +0.0019 ± 0.0025, M-standardised). This shift is common to all files and teams and does not move ranks.
- Ranges = three evidence models (`out/s5_picks.log`): val at face value + public; val shrunk by 0.4 + public; the latter + holdout with its variance inflated 4-fold (it sees only the TO branch, M>=50, in-period).
- Weekday: W8 - C3 on the holdout is +0.00143 ± 0.00040 on weekdays and +0.00161 ± 0.00062 on weekends; reweighted to private days +0.00132 ± 0.00057, to public days +0.00119 ± 0.00068. No weekday risk for the KD step.
- Cold tail: on val feeds with cold share >= 0.45 (1,211 feeds) W8 - C3 is -0.0007 ± 0.0023 and S16 - C3 -0.0017 ± 0.0021. M × cold post-stratified to private: W8 - C3 +0.00038 ± 0.00085, S16 - C3 +0.00001 ± 0.00076, W8 - S16 +0.00037 ± 0.00046, W8 - G3 +0.00165 ± 0.00088. No veto; the slice is too small to say more.
- Side observation on C3 as a hedge: C3 - G3 on the holdout is -0.00130 ± 0.00049 on private-like days and -0.00010 ± 0.00055 on public-like days (Monday alone -0.0026 ± 0.0007). Val, which is all weekdays, reads +0.00069 ± 0.00040, so this is the holdout's in-period popularity bias rather than a weekday effect, but the public confirmation of pop1 came from weekend feeds only. W8 and S16 contain no pop1 members.

## 4. Gap to #1 and rank risk (inferred)

Per-feed NDCG delta sd between two models as a function of their score correlation, measured on our own diverse models against W8: sd = 0.43 × sqrt(1 - corr) (C3 0.065, single member 0.103, v10 0.119, v5 0.153, CatBoost 0.193).

| score corr with a rival | SE of a public difference | SE of a private difference | #1 gap 0.00377 in SE | P(#1 has no true lead), flat prior | P(we beat #2 on private) |
|---|---|---|---|---|---|
| 0.85 | 0.0018 | 0.0012 | 2.1 | 0.019 | 0.39 |
| 0.90 | 0.0015 | 0.0010 | 2.6 | 0.005 | 0.36 |
| 0.95 | 0.0010 | 0.0007 | 3.6 | 0.0002 | 0.31 |

- With a winner's-curse simulation (5-20 contenders, true spread 0.001-0.002, score corr 0.85-0.95) the expected true lead of #1 is 0.0014-0.0034, P(true lead <= 0.001) is 0.01-0.35 and P(true lead <= 0) is 0.0004-0.10. So 0.0004-0.0023 of the gap may be noise; most likely about 0.001.
- #1 has three entries and has not moved since 11-12: fitting to public by selection is worth at most about 0.0003.
- Rank lottery against the recorded top six (snapshot 11-13 19:43 plus W8; rivals may have moved):

| our true gain from here | P(rank 1) | P(top 2) | P(top 3) | P(rank 5 or lower) |
|---|---|---|---|---|
| 0 | 0.00-0.03 | 0.27-0.28 | 0.56-0.67 | 0.11-0.22 |
| +0.0005 | 0.00-0.05 | 0.38-0.42 | 0.67-0.82 | 0.05-0.15 |
| +0.0010 | 0.01-0.08 | 0.48-0.59 | 0.76-0.92 | 0.02-0.09 |
| +0.0020 | 0.08-0.19 | 0.69-0.86 | 0.90-0.99 | 0.00-0.03 |

## 5. Final two picks and probes

### 5.1 Picks (assuming the better of two picks counts on private)

Value of the second pick = E[max of the pair] minus E[W8 alone], private noise and posterior uncertainty included:

| pair | value of the second pick | P(pair holds the best of the 7) |
|---|---|---|
| W8 + S16 | +0.00007..+0.00011 | 0.59-0.74 |
| W8 + W8C5 / K5 / KD4 | +0.00003..+0.00011 | 0.51-0.65 |
| W8 + C3 | +0.00003..+0.00007 | 0.49-0.57 |
| W8 + G3 | +0.00000..+0.00002 | 0.43-0.56 |

Recommendation:
1. **Pick 1 = W8** (`subs/cand_W8kd_r2.parquet`). It qualifies under FREEZE rule 2a (holdout +0.00076 = 2.3 SE over G3) and 2b (public +0.00167 >= +0.0011); S16 qualifies under 2b only; the rule's tie-break (within 0.0002 val, fewer members) selects W8.
2. **Pick 2 = the file with the largest expected excess over pick 1**, not the least-correlated file. Today that is S16. The freeze protocol's pick-2 rule gives the same answer by accident: S16 is the only file within 0.0008 of W8 on public; G3, C3, Pall and PF7x all fail that clause. A "different failure mode" hedge costs nothing measurable either (C3: -0.00004 of expected max), so the owner may keep C3 for that reason; the numbers do not favour it.
3. When a better generation arrives (W16, a GRU-student pool): new file = pick 1, W8 = pick 2. For W16 with an expected +0.00015 and a realised-difference sd of 0.0003, the pair {W16, W8} is worth about +0.0002 over W8 alone.
4. Replace the G3-centred text of FREEZE_PROTOCOL_1207 before 12-07: default pick 1 = W8; replacement rule = val Δ >= 0 and holdout Δ >= 0 with (val z + holdout z) / sqrt(2) >= 2, public only as a validity check (not below pick 1 by more than 2 SE of the public delta); pick 2 by expected excess (`s5_picks.py` computes it from per-feed arrays).
5. Confirm on the platform whether the two finals are chosen manually. If the two best public scores were taken automatically, a truly better file that draws a low public score would be excluded.

### 5.2 Which probes deserve a slot

| probe | SE of its public Δ | expected public z | verdict |
|---|---|---|---|
| a new pick candidate (W16; any pool that passed its val + holdout rule) | 0.00036 (W16 vs W8) | 0.4 | **yes, one slot each**: file validity and level. Do not rank it against W8 by public: P(W16 public > W8 public) = 0.66 if it is truly +0.00015 better |
| a new family with corr < 0.995 to W8 that passed val + holdout | 0.0005-0.0007 | 1-1.5 at +0.0008 | **yes**: public is a third independent sample of 8,562 feeds; val + public cut the SE of a delta by 21% |
| W8C5, K5, KD4 (blends with C3, narrow students) | 0.00047-0.00070 | 0.1-0.8 | no decision value: all sit 0.0003-0.0007 below W8 in posterior mean |
| w_ft variants of W8 (0.75 vs 0.25) | 0.00053 | - | no: val curve is flat for 0.25-0.5 (+0.00036 ± 0.00029 at 0.25) and negative above (-0.00076 ± 0.00027 at 0.75); FT - TO shows no Tuesday-to-Thursday trend (-0.0004 ± 0.0012) |
| ordering of near copies (corr >= 0.998) | 0.0004-0.0005 | < 0.5 | never |

## 6. Forward ideas from this lens

1. **Short-feed holdout (HOLD-SHORT), no retraining.** Re-score the 30K holdout feeds after sub-sampling each to a size drawn from the val M<50 distribution (random subset, positives kept; the `cmask` path of `augment_pool_batch` already does this) and report it next to the holdout. Reasons that are new today: the KD val gain sits in M<50 (+0.00144 ± 0.00101 vs +0.00009 ± 0.00061) where val's SE is worst and the holdout is blind; the running sub-feed arms (kwsub5 / kwsub2) target exactly the short-feed format and are screened on a holdout that contains no short feed; M<50 val deltas have the least certain transfer (k 0.10, -0.22..0.43). Cost: inference of 8-16 checkpoints on 30K feeds. Caveat: synthetic short feeds come from long-feed users, so it tests set-size effects, not the short-feed population.
2. **Stop same-recipe seeds at 16.** Pool-size law of the 8 wide students (a - b / k, max residual 0.00015): W16 is worth +0.00015 val / +0.00013 holdout over W8, W32 +0.00023, infinitely many +0.0003. The two wide arms are one recipe (score corr 0.9729 across arms, 0.9722 within). Forecast to check when W16 lands: TO val 0.40161-0.40175, holdout 0.32456-0.32470, final val 0.40174-0.40180.
3. **One evidence rule for every future candidate:** expected test gain ≈ 0.4 val Δ + 0.4 holdout Δ; adopt on combined z >= 2 with both signs positive; treat a val-only or holdout-only gain as unproven. This is what the data of §1.3 supports and what the KD promotion rules already approximate.
4. **Budget view:** the realistic prize is P(top 3) (0.56-0.67 now). It needs +0.0005..+0.001 of true gain, i.e. about +0.001..+0.002 of observed val gain at the predictive slope. Pool-size and blend work cannot deliver that (ceiling +0.0003); only a new member family of equal strength and lower correlation can.

## 7. Nulls and limits

- **Control variate for paired deltas (closed here):** a label-free model-expected NDCG delta has correlation -0.06..+0.01 with the realised per-feed delta; variance reduction <= 0.4%. Offline SEs cannot be reduced this way.
- Val by weekday, by M and with public M weights: no better than plain val (§1.4).
- Weekday and cold-tail composition: no private-specific risk found for W8 or S16 (§3).
- Not checked: the current leaderboard (last snapshot 11-13 19:43); the platform's final-selection mechanics; the rivals' real correlation with us (0.85-0.95 assumed); the FT branch out of period (no holdout form exists); HOLD-SHORT itself (needs model inference).
- The transfer factors rest on 14 public scores sharing one public draw; the lineage split (7 vs 8 files, G3 in both) was made after seeing the data and is a description, not a pre-registered test.
- Housekeeping: `CURRENT_STATUS.md` §1 and `docs/04_SUBMISSIONS.md` still name G3 as pick 1 and do not list W8 / S16; today's log lists C3 again at 0.393454, which, if it was a fresh submission of the C3 file, closes open item §9.1 of docs/04.

## 8. Files

- Scripts: `nn_x/w12/recheck/lb_stats/{lib,s1_build,s2_transfer,s3_kd_private,s4_crossteam_pool,s5_picks,s6_cold_cv,s7_eiv_check}.py`
- Logs: `nn_x/w12/recheck/lb_stats/out/{s1_build,s2_transfer,s3_kd_private,s4_crossteam_pool,s5_picks,s6_cold_cv,s7_eiv_check,s8_holdout_level_dow}.log`, `s2_pairs.csv`
- Arrays: `out/nd_val.npz` (per-feed val NDCG, 20 files), `out/nd_hold.npz` (holdout, 14 pools), `out/corr_test.npz`
