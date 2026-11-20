# VAL <-> PUBLIC <-> PRIVATE (2026-11-13): what val measures, how our solutions differ, and how to pick

Written 2026-11-13 by the synthesis step of the val/public workflow, from three independent CPU analyses (no GPU, no process touched, new files only):

- **calib**: `nn_x/audit/valpub/calib/findings.txt` (c1..c5, logs/JSON in `out/`): calibration, 40 pre-specified val reweightings, transfer-ratio anatomy, private projection, a pre-registered rule.
- **landscape**: `nn_x/audit/valpub/landscape/findings.txt` (s0..s5): verified rebuild of every pool, the val/holdout/public/private matrix, slices, transfer models, label-free test-side diagnostics, predictions for unscored pools.
- **semantics**: `nn_x/audit/valpub/semantics/findings.txt` (s01..s06): val vs test construction, public/private boundary and composition, exploration traffic, dispersion tests (half-splits, winner's curse), a pre-registered rule.

Prior work cited: `nn_x/pubdiag/noise/noise_out.txt` (noise model; calibration `public = 0.0022 + 0.976 val`, 8 points), `nn_x/audit/critique2/critique2.txt` (re-cool share vs val-selection collinear, corr 0.95), `nn_x/audit/publicprobe/findings.txt`, `reports/eda_shift_overlap.md`, `reports/review_shift.md`, `reports/sol_advval.md`, `reports/INSIGHTS_1112.md`, `src/pmix.py`, `src/calib.py`, `reports/CURRENT_STATUS.md` (21:52, 23:39, 23:46 entries).

**Conventions.** G3 = `cand_G_F10F7xF7x70_memeq` (val 0.40032, holdout 0.32379, public 0.392820) is the reference for every delta. "d" = paired per-feed NDCG delta vs G3; ± = paired SE (val: over 14,423 feeds; public: expected paired SE under the M x cold noise model of pubdiag/noise, 0.00035-0.00065 per contrast; holdout: over 30K train-period M>=50 feeds, train-only members only). TO = train-only half, FT = val-fine-tuned half; the final of every pool is 0.5 z(ft) + 0.5 z(to). "Modern 7" = Cd, Pall, PF7x, G3, M2, S2, TOonly (exact public scores on record). Public scores are used as data because the user asked for this analysis; nothing in this report was selected on them, and the one rule proposed (section 3) is stated for future use and was not re-fitted per candidate. The G3_later overlay (public = G3 to 6 digits) is a control only.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **Val's level is read correctly; val's between-pool differences were over-read.** M-only reweighted val predicts the public level of all 7 modern files to +0.0005 ± 0.0008 (the -0.0075 raw offset is feed-size composition, nothing else; no staleness penalty beyond M is visible on public). Between generations val transfers 1:1 (public = 0.0071 + 0.964 ± 0.033 val over 11 files, resid sd 0.0008). But *within* the modern generation the 7 pools span 0.00198 on val and 0.00074 on public, the rank correlation of the two orderings is +0.04 (Kendall +0.05), and the fitted val->public transfer factor of the 6 contrasts vs G3 is 0.05-0.16 ± 0.32-0.34 (95% CI reaches +0.67-0.83): 1:1 transfer is excluded at 2.4 SE / LR p 0.002-0.005, and "val says nothing about near-copy order" has the best leave-one-out error. Random half-splits of val give P(slope <= 0.10) = 0.015, so this is not sampling noise.
2. **The misreading was selection, not covariate shift.** None of 40 pre-specified val reweightings (M, M x cold, hour, user, H, category, Tue/Thu, cold cuts, 23 subsets) reproduces the public shortfall of the val-best pools: every composition-matched weighting still puts M2/S2 at +0.0005..+0.0009 where public read -0.0004 (z -1.5..-2.4), and for 5 of 6 contrasts post-stratification moves the expectation *away* from the realised value. What fits is a val-selection excess: the val-chosen pools M2 and S2 came in -0.0010 ± 0.0005 and -0.0009 ± 0.0004 below their val prediction, the un-selected hedges Cd/Pall/PF7x +0.0005..+0.0010 above; the excess sits in the TO half (members whose checkpoints were picked on val: val-TO minus holdout +0.0007 ± 0.0004 for M2, -0.0010 ± 0.0006 for Cd), not in the FT half (ft - to within 1 SE in every pool); and the winner's-curse arithmetic for the val-max of K = 5..50 near-copies (+0.00045..+0.00087 at SE_val 0.0004) is exactly the observed size. "Re-cool/m02 share" and "val selection" are collinear (corr 0.96) and cannot be separated on these pools; the practical consequence is identical.
3. **The train-period holdout is the lens that transfers.** With no fitted parameter, public delta = holdout delta: chi2(4) 0.27 (p 0.99) on the 4 contrasts with a train-only form, 1.19/6 with S2 and TOonly; absolute level LOO rmse 0.00017 vs 0.00044 for val on the same pools; the mixing profile public = lam*val + (1-lam)*holdout has its minimum at lam = -0.1 (1-sigma [-0.5, +0.3]). Caveat: the null "all 7 pools equal on public" fits as well (chi2 2.9/6), the agreement is tighter than chance (P 0.05), and the holdout is blind to w_ft, to M<50 feeds (39% of private) and to the cold>=0.45 tail (33% of private).
4. **Val is the right population; the "hour" and "Thu-as-staleness" stories are unnecessary.** Every intrinsic descriptor of val and test matches to within the val Tue-vs-Thu yardstick (JS category 0.0002 vs 0.0006); the only monotone-in-time variable is the train-derived cold share (0.26 Tue -> 0.42 test Tue). The raw hour gap (14-24h: val 0.48, public 0.36, private 0.60) is val's evening outages plus the public/private boundary cut; with outage hours filled every day is 0.52-0.55, hour has no effect on positives-per-feed or NDCG given M, and hour reweighting moves no pool delta by > 0.0004. The 117 later-history leak feeds affect no pool comparison (< 0.00003) and the organisers neutralise them on test. The staler Thu half is not a better public proxy than Tue (chi2 10.3 vs 7.3).
5. **Private vs public.** Level: private = public + ~0.0026 (M mix), i.e. ~0.3949 ± 0.001 for a G3-class pool. Composition differs in hour (by construction), weekday share (0.76 vs 0) and the cold tail (cold>=0.45: 0.33 vs 0.21); projected pool-delta shifts from any of these are +0.0003..+0.0007 ± 0.0003 (all non-G3 pools lose slightly less to G3 on colder feeds), the holdout's private-like subset leans the other way for the m02-heavy challengers (M2 -0.0006 ± 0.0005), and label-free every pool's disagreement with G3 is identical on val/public/private (ratios 0.99-1.01 overall, 0.9-1.15 inside every feed type) and flat across the four test days. Nothing in composition can re-order the modern pools by more than ~0.0005.
6. **Best supported for private: G3** (holdout-best, public-best, within 0.0006 of val-best; combined private estimate puts every other pool at -0.00001..-0.00059 with P(beats G3) <= 0.47). Pick 2: PF7x dominates Pall on val (+0.00078 ± 0.00023), holdout (+0.00025 ± 0.00013) and public (+0.00029) with the same hedge value (corr with G3 0.9978 vs 0.9971); Pall vs Cd is a toss-up (0.00017 ± 0.00037). No re-reading of val yields a better *candidate*: every pool we can build is within 0.0007 of G3 on val, 0.0003 on the holdout, 0.0007 on public. What changes is the decision rule.
7. **Pre-registered rule (section 3, numbers fixed today, not to be re-fitted):** val is a gate (presub PASS, V >= -1 SE_v, no label-free slice < -2 SE, pmix >= -1 SE, winner's-curse term logged); near-copies are ranked by the holdout, E_C = H_C + 0.15 V'_C, switch only at E_C > +2 SE and H_C > 0; after a public readout, S_C = 0.5 H_C + 0.5 P_C, switch only at S_C > +2 SE_S, and |P_C - H_C| > 2.5 sqrt(SE_pub^2 + SE_h^2) declares the lens broken (revert to G3); generational changes (val d >= 0.002, corr < 0.99) still transfer 1:1 and are adopted on a >= 3 SE val readout with holdout sign agreement; public never ranks near-copies. Applied retroactively it keeps G3 at every decision point and spends no public slot on M2/S2/TOonly.
8. **What public rewarded, and the modelling direction:** nothing resolvable among near-copies (7 files inside 0.00074 at paired SEs 0.00035-0.00062; H0 "all equal", shrinkage and holdout models are indistinguishable). The differences val rewarded (more re-cool/m02 members, val-trained stacker gates, val-picked checkpoints, +0.0006 val) bought nothing; the differences public did reward are the generational steps (v3 -> v5 -> v10 -> Cd, +0.003..+0.010 each, 1:1). The one within-generation component with a forward hint is the FT half (val 0, public +0.00047 for keeping it, z 0.9, composition says worth more on private): keep w_ft = 0.5. Implication: stop pool recombination work; spend effort on generational changes read on val at >= 3 SE, and select members/checkpoints on the holdout (or a cross-fitted val half) rather than on the val that is then read.
9. **Limits:** 7 modern points, 6 contrasts sharing G3's own public draw, SE_tot 0.0004-0.0008 each. They support exactly one inferential statement (near-copy val deltas do not transfer 1:1). They cannot give the transfer factor to better than ± 0.3, separate m02-share from val-selection from generic shrinkage, rank composition weightings against each other (expectations differ by < 0.0005), say anything about w_ft beyond one -0.00047 ± 0.0006 reading, or promise the holdout ordering holds on private's cold tail and M<50 feeds. The realised private paired SE between near-copies is 0.0003-0.0005: the realised private order within ± 0.0005 is a coin flip.

---------------------------------------------------------------------------------------------------------------------

## 1. How val relates to public and private

### 1.1 Level: calibrated, and the M mix is the whole story

| quantity | value | source |
|---|---|---|
| public = a + c*val, 11 files (v3, v4, v5, v10 + modern 7) | a +0.0071, c 0.964 ± 0.033, resid sd 0.00079, LOO rmse 0.00089 | calib 1a |
| same with slope fixed 1 | offset -0.00728, LOO rmse 0.00083 (equivalent) | calib 1a |
| earlier 8-point calibration (pubdiag) | 0.0022 + 0.976 ± 0.036 val, resid sd 0.00077 | pubdiag/noise part 3 |
| M-only reweighted val (src/calib) vs public, modern 7, slope 1 | offset +0.00053, sd over files 0.00076 | landscape 1d; semantics 4B |
| M x cold reweighted val (src/pmix) vs public | overshoots by -0.0042 (known level miscalibration) | semantics 4B |
| hour / M x hour / user / age reweightings | change nothing at the level | semantics 4B |
| M-only prediction of G3: public / private | 0.39235 / 0.39492 (realised public 0.39282) | semantics 1D |
| expected private level, G3-class pool | public + 0.0026 ~ 0.3949 ± 0.001 | semantics R1 |

The feared extra staleness penalty of -0.001..-0.002 (review_shift) has not appeared on public, which already has 0.735 of feeds at cold share > 0.3 (private 0.836). The cross-generation residuals are structured (v3 -0.0009, v10 +0.0011, Pall +0.0009, M2 -0.0010), which is the first sign that the 11-point slope is the wrong tool for within-generation differences: fitted on the 4 old files + Cd (c 1.04), it misses the 6 later modern files by -0.00125 on average (M2 -0.0021, S2 -0.0020, TOonly -0.0016) (calib 1a).

### 1.2 Between-pool differences inside the modern generation: they did not transfer

Exact public deltas vs G3 and their uniform-val expectations (calib 2; landscape 1):

| contrast | val d ± SE | public d (exact) | SE_pub | resid (public - val) | z |
|---|---|---|---|---|---|
| Cd | -0.00131 ± 0.00049 | -0.00074 | 0.00062 | +0.00057 | +0.7 |
| Pall | -0.00136 ± 0.00043 | -0.00038 | 0.00055 | +0.00097 | +1.4 |
| PF7x | -0.00058 ± 0.00039 | -0.00010 | 0.00050 | +0.00048 | +0.8 |
| M2 | +0.00062 ± 0.00034 | -0.00039 | 0.00043 | -0.00100 | -1.8 |
| S2 | +0.00056 ± 0.00028 | -0.00035 | 0.00035 | -0.00090 | -2.0 |
| TOonly | +0.00003 ± 0.00039 | -0.00047 | 0.00050 | -0.00051 | -0.8 |

- Spread: 0.00198 on val vs 0.00074 on public; Spearman of the 7-pool orders +0.04, Kendall +0.05 (p 0.5). On public the 7 scores deviate from their mean with sd 0.00024 against an expected sampling SE 0.00036 (chi2 3.1/6): public does not separate them at all; on val they differ at chi2 49/6 (semantics 4A).
- Transfer factor, three estimators: OLS of the 7 public scores on val, c = 0.10 ± 0.13 (calib 1b); GLS of the 6 contrasts with full covariance, k_v = 0.16 ± 0.34, 95% CI [-0.51, 0.83] (calib 3); with a free offset, b = 0.05 ± 0.32 (landscape 3d) and 0.10 ± 0.32 raw / 0.05-0.22 under every reweighting (semantics 4C). 1:1 is 2.2-3.0 SE away under every predictor; the likelihood-ratio test of k = 1 vs free k gives p 0.005, of b = 1 vs free (a, b) p 0.003, with PF7x instead of G3 as reference p 0.002 (landscape s5 b). The slope-0 model has the best LOO rmse (0.00026 vs 0.00083 for slope 1; calib 1b).
- Not noise: random half-splits of val (n_B = 7,211 ~ public) give slope 0.79 ± 0.42 and P(b <= 0.10) = 0.015; the Tue -> Thu split inside val transfers at 0.67 ± 0.35 (semantics 4E).

### 1.3 Why: val selection, not composition

**Composition is excluded as the explanation.** calib 2 ran 40 pre-specified weightings/subsets of val (uniform; public M; M x cold; hour5/hour24; M x hour; cold x hour; M x cold x hour; M x user; user; H; category-in-history; n_cat; top category; luo-cold; M x luo-cold; M x cold x H; subsets M<50 / M>=50 / Tue / Thu / cold cuts / user0 / tr0 / tr1+ / H bins / cih / hour bands / luo bands). None is rejected (public SEs are 0.0005-0.0008, so chi2(6) can only reject mis-predictions > ~0.0015), none beats the zero model, and every composition-matched weighting leaves M2/S2 at z -1.5..-2.4. The three "best" subsets (cold>=.45 only, cih<=.8, H<64; chi2 2.1-2.6) have n_eff 1.2-3.7K and imply an effective val (user0 13-22%, H 29-135, 100% cold > 0.3) nothing like public (4%, 175, 73%); the one-parameter tilts that lower chi2 (f1_mean theta +4.1, tr0 theta -2.1) push the val mean of the tilted variable outside the public range (f1 0.55 vs 0.39; tr0 0.18 vs 0.66): label fits, not composition. landscape 3a shows the direction problem: Cd/Pall lose most at h00-05 on val (-0.0043 / -0.0035 ± 0.0015) and public is 43% h00-09 (val 28%), so hour post-stratification predicts a *larger* deficit (-0.00159 / -0.00163) than realised (-0.00074 / -0.00038); M2/S2's val edges sit in cold30-45% (+0.0015 ± 0.0006) and all-cold-positive feeds (+0.0029 / +0.0025 ± 0.0009), which are *more* common on public (52%) and private, so M x cold raises their expectation (+0.00076 / +0.00061) while public read -0.00039 / -0.00035. Only TOonly's composition pushes the right way (privMC -0.00060, public -0.00047).

**Selection fits, in three independent forms.**
- *Residual structure:* resid = -0.00037 - 0.95 (± 0.32) x d_val (chi2 0.4/4): pools above G3 on val (M2, S2, both val-chosen) came in 0.0009-0.0010 below their prediction, pools below G3 (Cd, Pall, PF7x, never val-chosen) 0.0005-0.0010 above (semantics 4D; landscape 3d H3).
- *Where the excess lives:* val TO-half delta minus holdout delta = -0.00098 ± 0.00058 (Cd), -0.00073 ± 0.00052 (Pall), -0.00031 ± 0.00046 (PF7x), +0.00068 ± 0.00037 (M2): the m02 late-cooldown members, whose checkpoints were picked on val, look better on val than on the holdout and on public, while the FT half tracks the TO half within 1 SE in every pool (ft - to: +0.00063 ± 0.00055, +0.00006, -0.00011, -0.00024). The leak is val *selection* of members and pools, not the OOF fine-tune (calib B3, c5 ii). The selection-free holdout shrinks val's TO ordering the same way (slope 0.36 ± 0.31; semantics 4E) and sees ~1/3 of the val spread (M2 - Cd: val +0.0019, holdout +0.0004, public +0.0004; landscape B4).
- *Size:* the expected val-only excess of the best of K near-copies is SE_val x E[max_K] = +0.00045 (K 5), +0.00059 (K 10), +0.00072 (K 20), +0.00087 (K 50) at SE_val 0.00039; the observed shortfall of M2/S2 is -0.0009..-0.0010 (semantics 4E; calib 3).

**What cannot be separated:** re-cool/m02 share, val-selected indicator, val-holdout discrepancy and generic shrinkage are collinear across the 6 contrasts (corr share~d_val 0.96, share~discrepancy 0.97, selected~d_val 0.81; critique2 found 0.95 on 5 pools). All one-parameter models are accepted (chi2 0.7-7.8); by leave-one-contrast-out the val-holdout-discrepancy model is best (beta -1.07 ± 0.45: the whole val excess over the holdout vanishes on public), then generic shrinkage, val-selected indicator (-0.00088 ± 0.00042), re-cool share (-0.00140 ± 0.00078), FT-half excess, FT weight; differences < ~2 in LOO sum z^2 are not evidence (calib B5). The re-cool model (H4) fits worse than plain shrinkage (chi2 5.6/5 vs 0.4/4; landscape 3d). The m02 mechanism (members selected on val inside the cooldown) is itself a form of val selection, so the rule is the same either way.

**Weak forward evidence inside val (direction, not result):** G3's edge over every family-different pool is larger on Tue (3 days after the last train day) than on Thu (5 days): Thu - Tue of d = +0.0005..+0.0017 for Cd, Pall, PF7x, M2, TOonly, N1, NEW12, F10only, each <= 1.7 SE; the G3-plus-more-of-the-same pools (U1, U3, PoP3, S1) show no day effect (landscape 3e).

### 1.4 The holdout transfers

| test | result | source |
|---|---|---|
| public d = holdout d, no parameter, Cd/Pall/PF7x/M2 | chi2(4) 0.27, p 0.99, per-contrast abs(z) <= 0.35 | calib B2 |
| same with TOonly (holdout = G3's) | chi2(5) 0.72, p 0.98 | calib 3 |
| same, 6 contrasts incl. S2 (recorded H -0.00006) | chi2 1.19/6, p 0.98; with common offset -0.00019 ± 0.00024, LOO rmse 0.00023 | landscape 3d H5 |
| fitted k_h | 1.13 ± 1.16 | calib 3 |
| absolute level, public = a + c*holdout, 5 pools | c 1.10 ± 0.31, LOO rmse 0.00017; val on the same 5 pools: c 0.16 ± 0.17, LOO rmse 0.00044 | calib 1c |
| public = lam*val + (1-lam)*holdout | minimum at lam = -0.1, 1-sigma [-0.5, +0.3]; lam = 1 p 0.18; LOO rmse rises monotonically with lam | calib 1d |
| tau(holdout, public) / tau(val, public) / tau(val, holdout) | +0.28 / +0.05 / +0.55 | landscape B2 |

Caveats, all three analyses: the null "all pools equal on public" also fits (chi2 2.88/6, p 0.82), so public cannot tell "holdout is right" from "nothing differs"; the holdout-public agreement is tighter than chance (P(chi2_4 <= 0.72) = 0.05), so part of it may be luck; the holdout is TO-only, M>=50, train-period (Sep 6-19, 32% weekend), its by-day contrasts swing ± 0.0015, and it is blind to w_ft (TOonly), to M<50 feeds and to the test-period cold tail.

### 1.5 Val is the right population

- **Construction identical to test** (semantics 1A): M (96.5 vs 103.6 public / 100.7 private; weekend days +4), M<50 share 0.41 / 0.39 / 0.39, H 173 / 175 / 174, user-0 0.045 / 0.041 / 0.041, train-user share 0.30-0.32, n_cat 12.0 / 12.6 / 12.5, f1 mean 0.39 everywhere, listing age (f4) < 1 d 0.11-0.12 everywhere, history composition equal; JS(category) val-vs-test 0.0002 (Tue-vs-Thu 0.0006). The only monotone-in-time descriptor is the train-derived cold share (0.258 Tue, 0.324 Thu, 0.367 Sat ... 0.418 Tue; public 0.371, private 0.406). Listing age does not shift: "later = staler listings" is wrong for the candidates; only our train-derived statistics go stale.
- **Hour is an artefact** (semantics 2C, 1D): val lost Tue 20h and Thu 20-21h to outages; with outage hours filled the 14-24h share is val 0.517, Sat 0.538, Sun 0.554, Mon 0.516, Tue 0.533 (raw: val 0.482, public 0.360, private 0.597). Positives per feed do not depend on hour or day given M (npos ~ logM +0.29 ± 0.01; hour terms -0.02..-0.07 ± 0.06); G3 NDCG given logM has hour terms within 1 SE; hour reweighting does not improve val->public calibration (chi2 9.99 vs 10.85) and no hour term is consistently non-zero in the pool-delta regressions (h06-09 for M2/S2 at -0.0008..-0.0012 among 105 tests). The hour-reprojection programme (publicprobe 5.3 slice twins) is dropped.
- **Positives-per-feed selection** affects only the level through P(npos | M) and the random-ranking baseline tracks the hour pattern exactly; it does not touch within-feed ranking or pool choice (semantics 1D).
- **The later-history leak channel** exists in test as in val (103/108 same-user pairs; 183 rows = 1.48% of earlier candidates vs 1.30% on val) and the overlay that re-ranked exactly those rows moved public by -0.000001: those feeds are unscored or positive-free in the organisers' evaluation. Its contribution to any val pool comparison is < 0.00003; the FT half did not learn it (FT - TO on the 117 leak feeds -0.0024 ± 0.0050) (semantics 1E, 6B; CURRENT_STATUS 21:52).
- **Boundary**: the organisers' wording selects ts <= 1790497920 (8,580 feeds) rather than ts < (8,562); the 18 boundary feeds change every number by < 0.00003 (semantics 2A).
- **Exploration traffic**: the display order is random but the candidate *set* is ranker-shaped (positives at within-feed f1 percentile 0.38; f1 alone 0.336 vs random 0.290). Production-order information lives in the position tower / K9 aux, not in f1 (within-feed Spearman of production order with f1 only 0.09), and its value does not drift (f1 lift flat over the 12 train days, -0.00015 ± 0.00023/day; Tue = Thu) (semantics 3).

### 1.6 Private vs public

Composition (val | holdout | public | private; calib 4, semantics 2B, landscape 2b):

| descriptor | val | holdout | public | private |
|---|---|---|---|---|
| mean M / M<50 share | 96 / 0.41 | 154 / 0 | 104 / 0.39 | 101 / 0.39 |
| H / user0 / tr0 / cand-cat-in-hist | 173 / 0.046 / 0.65 / 0.845 | - / 0.051 / - / - | 175 / 0.041 / 0.66 / 0.847 | 174 / 0.041 / 0.67 / 0.847 |
| cold > 0.3 / cold >= 0.45 / cold >= 0.6 | 0.43 / 0.08 / 0.01 | - | 0.74 / 0.21 / 0.03 | 0.84 / 0.33 / 0.05 |
| luo-cold > 0.5 | 0.14 | - | 0.24 | 0.33 |
| 14-24h / 18-24h (raw) | 0.48 / 0.25 | 0.52 / 0.32 | 0.36 / 0.18 | 0.60 / 0.37 |
| weekend share | 0 | 0.32 | 1.00 | 0.24 |

Public -> private effect sizes (SMD): hour +0.53 (construction), ccnt_zero_share +0.34, cold_share +0.31 (half the val Tue -> Thu step of +0.59), luo_cold_share +0.25; 30 of 33 descriptors |SMD| < 0.10. Private by day: Sun 4,855 feeds (cold 0.39), Mon 7,578 (0.40), Tue 7,600 (0.42).

What that does to the pools:
- **Level**: carried by logM (t -42), H, f1_sd, which do not shift; projected public -> private +0.0020 ± 0.0013, M-only 0.39235 -> 0.39492 (semantics 2D).
- **Ranking, three lenses that all land inside ± 0.0005**: (a) val regressions projected through the descriptor shifts: Cd +0.00055 ± 0.00035, Pall +0.00054 ± 0.00033, PF7x +0.00026 ± 0.00029, M2 +0.00033 ± 0.00024, S2 +0.00037 ± 0.00020, TOonly +0.00067 ± 0.00028 (every non-G3 pool loses slightly less to G3 on colder/evening feeds; 1-2 SE among 105 tests; semantics 2D); (b) private-oriented val reweightings (M, M x cold, hour5, M x hour, cold x hour, M x cold x hour, M x luo-cold) move every pool's delta by < 0.0004 (calib 4b); (c) the holdout reweighted to private hour x weekend moves < 0.0002, and its private-like subset (Sun >= 12h, Mon, Tue; n 11,169) minus public-like puts M2 -0.00063 ± 0.00049, U3 -0.00053 ± 0.00042, U1 -0.00043 ± 0.00034, N1 -0.00031 ± 0.00047 vs the no-m02 pools +0.0003..+0.0008 ± 0.0007 (landscape 3e; calib 4: M2's holdout weekday -0.00043 ± 0.00023 vs weekend +0.00035 ± 0.00030, against M2 on private's 76% weekdays). (a) and (c) point in opposite directions for M2 at 1-2 SE each: the honest reading is that composition cannot re-order the modern pools by more than ~0.0005.
- **Label-free test-side diagnostics** (landscape 4): every pool's disagreement with G3 (1 - within-feed corr; top-1 disagreement) has private/public ratio 0.987-1.007, private/val ratio 0.9-1.15 inside every cold-share, LUO, hour, M, user, H and weekday slice with n >= 300, and is flat across Sat/Sun/Mon/Tue; every pool raises the cold share of its top-5 by the same +0.077 from val to private (label-free expectation +0.064); pool-agreement statistics show no within-test trend after cold/M adjustment (residual <= 0.0006; semantics 3C). No pool is an outlier on cold-heavy, late-hour, Mon/Tue or weekend feeds. TOonly is the only pool whose disagreement with G3 is systematically lower on test than on val (0.83-0.93): the FT half contributes slightly less to the ranking on test feeds.
- **Weekend**: public is 100% weekend, private 24%, val 0%; the holdout (32% weekend) is the only labelled weekend sample. Re-cooled F7x/M2 members gain on holdout weekend feeds (+0.0010..+0.0014, 1.7-2.0 SE), which should have favoured M2 on the all-weekend public, yet public put M2 0.0004 below G3: the weekend pattern is not a private risk in the feared direction and did not even show on public (semantics 2E).

### 1.7 Were we misreading val? A ledger

| reading in use until 11-12 | verdict | evidence |
|---|---|---|
| val level -> public level via the cross-generation calibration (0.0022 + 0.976 val; critique 0.0078 + 0.962 val) | **right for levels and generational steps**; wrong tool for near-copy differences (misses the 6 later modern files by -0.00125 on average when fitted on old + Cd) | calib 1a; semantics 4B |
| val is a different population from test (hour, staleness) | **population is the same**; only train-derived cold share drifts; hour is outages + boundary | semantics 1, 2C |
| a +0.0006 val edge of a near-copy (M2, S2, U3) is worth ~+0.0006 on test | **over-read by ~0.0005-0.001**: transfer 0.05-0.16 ± 0.3, winner's curse of a val max 0.0005-0.0009 | calib B1, semantics 4E |
| "the m02-heavy pool's val edge transfers at ~0.3x" (CURRENT_STATUS 23:46) | **consistent but not sharper than 0.1-0.3 ± 0.3**; 0 is inside the CI; the mechanism is val selection, of which m02 checkpoint picking is one instance | calib B5; semantics 4D |
| M x cold (pmix) reweighting as the private proxy | **level-miscalibrated (-0.0042)** and no better than uniform val for deltas; keep as a veto only | semantics 4B; calib 1e |
| Thu (staler) half as the better test proxy | **not supported** (chi2 10.3 vs 7.3 for Tue) | semantics 4C |
| hour reprojection / slice twins for private | **unnecessary**: hour has no effect on NDCG or pool deltas given M | semantics 1D, 2D |
| public readouts as a way to rank near-copies | **wrong**: public SE 0.00045-0.0008 incl. val; the 7 modern files are inside noise of each other | semantics 4A, 5 |
| the holdout as a secondary tie-breaker | **should be the primary near-copy lens**: predicts public with no free parameter | calib B2; landscape B2 |
| the FT half adds nothing (val 0 ± 0.0004) | **val prices it at zero because it trades warm positives (-0.0017) for cold positives (+0.0051)**; public read +0.00047 for keeping it; it is not the part of G3 at risk on private | landscape 5c |

---------------------------------------------------------------------------------------------------------------------

## 2. The landscape of our solutions

### 2.1 The matrix (landscape s1, verified builders: G3 rebuild corr 1.0000000, all recorded numbers reproduced to 1e-5)

| pool | n members | re-cool / m02 share | val | ft-half | to-half | pubM | privM | pubMC | privMC | holdout | public | families |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cd | 19 | 0.00 / 0.00 | 0.39901 | 0.39889 | 0.39877 | 0.39094 | 0.39355 | 0.39607 | 0.40173 | 0.32319 | 0.39208 | F5cd(6)+F6(6)+F7(7) |
| Pall | 38 | 0.00 / 0.00 | 0.39896 | 0.39871 | 0.39917 | 0.39088 | 0.39350 | 0.39605 | 0.40165 | 0.32335 | 0.392436 | Cd + F9(6)+F10(6)+F12(7) |
| PF7x | 38 | 0.18 / 0.18 | 0.39974 | 0.39921 | 0.39984 | 0.39167 | 0.39429 | 0.39657 | 0.40209 | 0.32360 | 0.392723 | Pall with F7x for F7 |
| **G3** | 20 | 0.70 / 0.70 | 0.40032 | 0.39984 | 0.40035 | 0.39227 | 0.39487 | 0.39705 | 0.40249 | **0.32379** | **0.392820** | F10(6)+F7x(7)+F7x70(7) |
| M2 | 14 | 1.00 / 1.00 | **0.40094** | 0.40009 | 0.40085 | 0.39286 | 0.39547 | 0.39781 | 0.40322 | 0.32361 | 0.392434 | F7x(7)+F7x70(7) |
| S2 | 20 | 0.70 / 0.70 | 0.40088 | - | - | 0.39278 | 0.39540 | 0.39766 | 0.40310 | (-0.00006 rec.) | 0.392473 | neural gate stacker on G3 |
| TOonly | 20 | 0.70 / 0.70 | 0.40035 | - | 0.40035 | 0.39220 | 0.39484 | 0.39675 | 0.40189 | 0.32379 | 0.392346 | G3 train-only half |
| N1 | 32 | 0.44 / 0.81 | 0.40007 | 0.39964 | 0.40013 | 0.39199 | 0.39462 | 0.39700 | 0.40248 | 0.32363 | - | G3 + NEW12 |
| U1 | 24 | 0.58 / 0.75 | 0.40035 | 0.39994 | 0.40032 | 0.39229 | 0.39491 | 0.39704 | 0.40238 | 0.32376 | - | G3 + W5base(4) |
| U3 | 30 | 0.47 / 0.80 | 0.40066 | 0.39995 | 0.40033 | 0.39258 | 0.39520 | 0.39754 | 0.40307 | 0.32375 | - | U1 + v30(6) |
| S1 | 38 | 0.18 / 0.18 | 0.40046 | 0.39956 | 0.40022 | 0.39238 | 0.39500 | 0.39718 | 0.40254 | - | - | hfit P11 W1to (PF7x families) |
| PoP3 | 52 | 0.27 / 0.27 | 0.40059 | - | - | 0.39251 | 0.39513 | 0.39738 | 0.40279 | - | - | pool of pools |
| NEW12 | 12 | 0.00 / 1.00 | 0.39955 | 0.39961 | 0.39945 | 0.39150 | 0.39412 | 0.39646 | 0.40194 | 0.32323 | - | from-scratch m02 (4 archs) |
| F10only | 6 | 0.00 / 0.00 | 0.39881 | 0.39874 | 0.39848 | 0.39067 | 0.39333 | 0.39574 | 0.40147 | 0.32254 | - | F10(6) |
| v10 | - | - | 0.39401 | - | - | 0.38598 | 0.38861 | 0.39054 | 0.39596 | - | 0.38786 | nn_final4 mix (old gen) |
| v5 | - | - | 0.38692 | - | - | 0.37896 | 0.38153 | 0.38354 | 0.38931 | - | 0.38020 | nn_final2 + residual stacker |

Deltas vs G3 (paired; landscape s1_deltas_vs_G3.csv):

| pool | val d ± SE | pubMC d | privMC d | holdout d ± SE | public d (SE_pub) | resid vs pubM (z) |
|---|---|---|---|---|---|---|
| Cd | -0.00131 ± 0.00049 | -0.00098 | -0.00076 | -0.00060 ± 0.00027 | -0.00074 (0.00062) | +0.00059 (+0.7) |
| Pall | -0.00136 ± 0.00043 | -0.00100 | -0.00083 | -0.00045 ± 0.00025 | -0.00038 (0.00055) | +0.00100 (+1.4) |
| PF7x | -0.00058 ± 0.00039 | -0.00048 | -0.00040 | -0.00020 ± 0.00022 | -0.00010 (0.00050) | +0.00050 (+0.8) |
| M2 | +0.00062 ± 0.00034 | +0.00076 | +0.00073 | -0.00018 ± 0.00019 | -0.00039 (0.00043) | -0.00098 (-1.8) |
| S2 | +0.00056 ± 0.00028 | +0.00061 | +0.00062 | n/a | -0.00035 (0.00035) | -0.00086 (-1.9) |
| TOonly | +0.00003 ± 0.00039 | -0.00030 | -0.00060 | 0 (same TO) | -0.00047 (0.00050) | -0.00040 (-0.6) |
| N1 | -0.00025 ± 0.00028 | -0.00005 | -0.00000 | -0.00017 ± 0.00016 | - | |
| U1 | +0.00003 ± 0.00020 | -0.00001 | -0.00010 | -0.00003 ± 0.00013 | - | |
| U3 | +0.00034 ± 0.00028 | +0.00049 | +0.00058 | -0.00005 ± 0.00015 | - | |
| S1 | +0.00013 ± 0.00034 | +0.00013 | +0.00006 | n/a | - | |
| PoP3 | +0.00027 ± 0.00022 | +0.00033 | +0.00030 | n/a | - | |
| NEW12 | -0.00077 ± 0.00049 | -0.00058 | -0.00055 | -0.00057 ± 0.00028 | - | |
| F10only | -0.00151 ± 0.00049 | -0.00131 | -0.00101 | -0.00125 ± 0.00029 | - | |

Readings (landscape 1): every pool is at or below G3 on the holdout (the ties U1/U3 are G3 plus more m02 members: saturation, INSIGHTS_1112); every scored pool is below G3 on public; the FT half adds ~0 on val inside every pool (final - to-half: G3 -0.00003 ± 0.00039, M2 +0.00009, Cd +0.00024, Pall -0.00021, U3 +0.00033), so the val differences between pools are carried entirely by the TO halves; re-cool share and val delta are collinear (corr 0.96), the holdout delta is collinear with the val delta too (0.83) but at 1/3 the scale.

### 2.2 How the pools differ

- **Correlation structure** (landscape 2a; identical on val / public / private to 3 decimals): within-feed Pearson with G3: U1 0.9998, PoP3 0.9997, U3/N1 0.9993, M2 0.9990, S1 0.9988, TOonly 0.998, PF7x 0.9978, Pall 0.9971, Cd 0.9957, NEW12 0.9953, F10only 0.9946; S2 Spearman 0.997 (rank-valued scores). Old generations: v10 0.930, v5 0.880, v4 0.864, v3 0.855. Top-1 agreement with G3 on private: U1/PoP3/S2 0.97, M2 0.94, PF7x 0.91, Pall 0.89, Cd 0.87, v10 0.63, v3 0.39. Everything modern is a near-copy (corr >= 0.995, |d| <= 0.002); the generational steps are at corr 0.86-0.93.
- **Where the val deltas sit** (landscape 2c): Cd/Pall (no m02) lose most at night (h00-05 -0.0043 / -0.0035 ± 0.0015), on cold15-30% (-0.0021 / -0.0019 ± 0.0007), new users (tr0 -0.0019 / -0.0015 ± 0.0006), H256, cat2, 3-positive feeds; at or above G3 on cold >= 45% and 6+ positives. PF7x is a diluted Cd. M2's edge: cold30-45% (+0.0015 ± 0.0006), H64-191, tr0, all-cold-positive feeds (+0.0029 ± 0.0009), single-positive feeds, Thu (+0.0011 ± 0.0005) vs Tue (+0.0002 ± 0.0004); nothing on cold15-30% or warm positives. S2: cat13, h14-23, all-cold-positive (+0.0025 ± 0.0007). TOonly (FT removed): -0.0051 ± 0.0012 on all-cold-positive, -0.0022 on mixed, +0.0017 ± 0.0005 on warm-positive feeds; the FT half is a cold-positive promoter whose val value nets to zero. By the rank of the positive under G3: every alternative is below G3 on the 12.8% of feeds where G3 has the positive at rank 1 (Cd -0.031 ... U1 -0.005) and above it everywhere else: what "near-copy with noise" looks like, not a mechanism.
- **The FT half** (landscape 5c): val 0 ± 0.0004 inside every pool, public +0.00047 (z 0.9) for keeping it, private cold-positive mix makes it worth more (privMC -0.00089 for removing it vs -0.00045 on val), label-free it is G3's cold promoter (TOonly top-5 cold share 0.289 on private vs 0.305 for G3) and it does not switch off on private. This does not license a different w_ft (val curve flat ± 0.0005, no holdout form, closed 3x; critique2 R6); it says the FT half is not the part of G3 at risk.

### 2.3 Which is best supported for private

**Pick 1: G3**, on every lens (landscape 6a; calib B6):
- holdout (30K, labelled, no val labels, no public): G3 first; every other pool -0.00003..-0.00125;
- public (8.5K, exact): G3 first by +0.0001 (PF7x) .. +0.0007 (Cd), inside noise;
- val: within 0.0006 of the top (M2, S2, U3 are above it on val, at or below it on holdout and public);
- combined private estimate (holdout + composition-corrected public, inverse-variance; calib 4e): TOonly -0.00001 ± 0.00010 (holdout blind to w_ft; public alone -0.00038 ± 0.00065), PF7x -0.00017 ± 0.00022, M2 -0.00025 ± 0.00019, S2 -0.00031 ± 0.00046, Pall -0.00042 ± 0.00025, Cd -0.00059 ± 0.00027; P(beats G3 on private) 0.47 / 0.22 / 0.09 / 0.25 / 0.04 / 0.02; order unchanged under every leave-one-contrast-out k_v (0.04-0.22);
- label-free private: no pool separates from G3 beyond what it does on val/public; the holdout private-like subset leans against M2/U1/U3 (<= 1.3 SE) and toward nothing;
- the only pools that could plausibly beat G3 on private are U1 / PoP3 (G3 + more of the same; holdout tie; corr 0.9998 / 0.9997): a coin-flip of < 0.0003 either way, not worth a switch.

**Pick 2 (hedge):** PF7x dominates the current pick 2 Pall on all three labelled samples: val +0.00078 ± 0.00023, holdout +0.00025 ± 0.00013, public +0.00029 (paired SE ~0.0003), with the same hedge value (corr with G3 0.9978 vs 0.9971; landscape 6b). Pall's rationale ("no re-cool in case re-cool is val-only") was tested by two non-val samples and did not hold; keeping Pall costs ~0.0003 on every sample. calib's own pick-2 comparison among zero-m02 hedges puts Pall above Cd by 0.00017 ± 0.00037 (nothing). The landscape rule (section 3) selects PF7x; semantics R8 says "PF7x or Pall". The decision is the lead's; the evidence favours PF7x.

**Unscored pools, pre-registered public predictions vs G3** (landscape 6c; for future validation only; H1 = 1:1 | H3 = shrinkage -0.00037 + 0.05 pubM | H5o = holdout - 0.00019; SE_pub): N1 -0.00028 | -0.00039 | -0.00035 (0.00036); U1 +0.00003 | -0.00037 | -0.00022 (0.00025); U3 +0.00031 | -0.00036 | -0.00023 (0.00035); PoP3 +0.00024 | -0.00036 | n/a (0.00028); S1 +0.00011 | -0.00037 | n/a (0.00043); NEW12 -0.00077 | -0.00041 | -0.00075 (0.00062); F10only -0.00160 | -0.00045 | -0.00144 (0.00063). Readable probes of 1:1 vs shrinkage: PoP3 (2.2 SE), F10only (1.8), U3 (1.9); none is a pick candidate, so they are diagnostic submissions only, and calib notes that for separating the *collinear* stories (re-cool vs selection) the predicted readings differ by < 0.9 SE, i.e. no probe settles that one.

---------------------------------------------------------------------------------------------------------------------

## 3. Pre-registered selection rule (from 2026-11-13 on) and the modelling direction it implies

The three analyses each stated a rule (calib section 5, landscape 6c, semantics section 6). They agree on the structure and differ only in the form of the score; the unified rule below fixes one form. Its numbers were set today from the same 6-7 public points it is consistent with, so it is validated only by readings made after this file and must not be re-fitted per candidate. Log K (the number of pools compared on val for the decision) with every decision.

**R0 Classify the candidate C against the incumbent (G3).**
- *Generational*: val d >= 0.002 and within-feed corr < 0.99 (our generational steps: v3 -> v4 -> v5 -> v10 -> Cd, corr 0.86-0.93).
- *Near-copy*: everything else (all 13 current pools: corr >= 0.995, |d| <= 0.002).

**R1 Val is a gate, never the ranking criterion among near-copies.** Eligibility: presub PASS (src/presub_gate.py); V_C = uniform val d >= -1 SE_v; no label-free val slice (M bins, hour, cold share, user type) below -2 SE; pmix (M x cold) delta >= -1 SE (veto only, never a promoter). Winner's curse: if C is the val-max of K pools, subtract SE_v x E[max_K] (0.00045 at K 5, 0.0006 at K 10, 0.0007 at K 20, 0.0009 at K 50) from V_C before any use (semantics R4). Ignore hour slices and reprojections, Tue-vs-Thu as a staleness proxy (use the two days only as a replication check), listing-age / user / category reweightings, the 117 leak feeds, and weekend reasoning from public (semantics R5).

**R2 Levels** are predicted with the M-only reweighting (src/calib) + 0.0005 for the modern generation; private = public + 0.0026. Not pmix, not hour (semantics R1).

**R3 Generational candidates** transfer ~1:1: adopt on a single val readout >= 3 SE_v, the holdout sign agreeing where a train-only form exists, and R1 passed (semantics R2).

**R4 Near-copy candidates, before any public look (holdout-led; calib section 5).** H_C = paired holdout delta vs the incumbent (train-only form, SE_h); V'_C = M x cold x hour5 private-weighted val delta. Score E_C = H_C + 0.15 V'_C, SE_E = sqrt(SE_h^2 + 0.0225 SE_v'^2) (~ SE_h). Switch pick 1 only if H_C > 0 AND E_C > +2 SE_E AND R1 passed; among qualifiers the highest E_C. Equivalently, in semantics' units, a near-copy val delta is worth 0.3x (0.6x if holdout-confirmed at >= 1 SE_h and positive on both val days) and must still clear 2 SE_v after shrinkage, i.e. in practice a near-copy is never adopted on val alone (semantics R3). Candidates with no train-only form (stackers, FT-only files, w_ft variants, overlays) have no H_C and are not eligible for pick 1.

**R5 After a public readout of C (verification, not ranking).** P_C = exact public delta vs the incumbent. (a) Verification: |P_C - H_C| > 2.5 sqrt(SE_pub^2 + SE_h^2) declares the holdout lens broken for that candidate class; revert to "G3 stays" and re-examine before any further candidate of that class. Also reject C if it lands > 2 SE_pub + 0.001 below its M-mix level prediction (semantics R7). (b) Score: S_C = 0.5 H_C + 0.5 P_C, SE_S = 0.5 sqrt(SE_h^2 + SE_pub^2) (typically 0.00015-0.00035); switch pick 1 only if S_C > +2 SE_S (landscape R2/R3). Val does not enter S. Public never ranks near-copies (it can only reject differences >= 0.0015), no hour slice twins, no re-cool-share discount (not identifiable; agrees with critique2 R7).

**R6 Pick 2** = the eligible, non-dominated pool (not below pick 1's alternatives on all of val/holdout/public) with the lowest test correlation to pick 1; among pools inside one band, prefer the higher holdout TO score and the lower share of val-selected components (landscape R3; semantics R8). A pool with no H_C may be pick 2 only if P_C >= -1 SE_pub.

**R7 Retroactive check** (what the rule would have done on the 7 scored pools; landscape 6d, calib 5): R1 passes all; no near-copy has H_C > 0, so no pre-public switch; S vs G3: PF7x -0.00015, S2 -0.00021, TOonly -0.00024, M2 -0.00029, Pall -0.00042, Cd -0.00067: no switch; pick 2 = PF7x. It keeps G3 at every decision point of 11-12 and would not have spent public slots on the M2 / S2 / TOonly readouts. This is the decide6/decide8 holdout-led rule already in use, minus the val tie-break, plus public as verification and as the second half of S.

**What modelling direction this implies (which kind of model difference public rewarded):**
1. Public rewarded the *generational* steps (val +0.003..+0.010 at corr 0.86-0.93; transfer 1:1) and nothing within the near-copy generation (7 files inside 0.00074; the differences val rewarded, i.e. more re-cool/m02 members, val-trained stacker gates, val-picked checkpoints, a higher val score, bought nothing). The expected realised private value of a +0.001 near-copy val gain is ~+0.0003. Stop spending submissions and GPU on near-copy pool questions (weights, stackers, aggregation, M-split, w_ft, re-cool share); the holdout already answers what the planned share-0 probes were meant to ask.
2. The one within-generation component with a forward hint is the FT half (val 0, public +0.00047 for keeping it, z 0.9, worth more under the private cold-positive mix): keep w_ft = 0.5; it is not the part of G3 at risk on private.
3. All three collinear explanations of the val excess (m02 checkpoints picked on val, val-chosen pools, val-holdout discrepancy) share one operational cause, selection on the val that is then read. For future members and pools: select checkpoints and members on the holdout (or on a val half not used for the final readout, cross-fitted), and read the result on the untouched half. This is the direction the evidence points, at the 1-2 SE level; it is not a measured gain.
4. A better *solution* requires a generational change (new families/architectures/inputs with corr < 0.99 to G3 and val d >= 0.002, read at >= 3 SE and confirmed by the holdout sign), not a re-reading of val among the pools we have: every buildable pool is within 0.0007 of G3 on val, 0.0003 on the holdout, 0.0007 on public, and label-free identical to G3 on private. The external top-1 gap (~+0.0056 val-equivalent over G3; CURRENT_STATUS 21:52) is of generational size.
5. What would change this picture: a public reading of a pool that is >= 0.001 above G3 on val AND >= +1 SE_h on the holdout (none exists; U1/U3 tie G3 on the holdout), or two of the pre-registered probes (PoP3, F10only) landing on the 1:1 predictions.

---------------------------------------------------------------------------------------------------------------------

## 4. Honest limits (read before quoting any number above)

- **Seven modern public points, six contrasts, all sharing G3's own public draw, SE_tot 0.0004-0.0008 each.** They support exactly one inferential statement, that near-copy val deltas do not transfer 1:1 (2.2-3.0 SE under every predictor; half-split null puts the realised slope at its 1.5th percentile; the old-era calibration misses the 6 later files by -0.0013 on average). The transfer factor itself is known to ± 0.3 (0 and 0.4 are both inside the CI); H0 "all equal", shrinkage and holdout are statistically indistinguishable (chi2 0.4-2.9 on 4-6 df); the shrinkage fit (chi2 0.41/4) and the holdout fit (P(chi2_4 <= 0.72) = 0.05) are both tighter than chance, so the public SEs may be slightly over-estimated or part of the agreement is luck.
- **Collinear mechanisms.** m02/re-cool share, val-selected indicator, val-holdout discrepancy and generic shrinkage have corr 0.8-0.98 across the contrasts; a share-0 pool that is val-better than G3, or a val-selected low-share pool, would be needed to break it, none is scored, and the predicted readings of the planned probes differ by < 0.9 SE between the models. The practical rule does not depend on which story is true.
- **Composition weightings cannot be ranked against each other**: their expectations differ by < 0.0005, below public resolution; the reweighting analyses assume the within-type val delta is the within-type test delta, which is exactly what a val-period-specific mechanism violates. They can say composition does not explain M2/S2; they cannot say which weighting is best.
- **The holdout's blind spots**: it is train-period (Sep 6-19), TO-only, M>=50, production-traffic-like; its by-day contrasts swing ± 0.0015 (SE 0.0009); it cannot see w_ft (TOonly's -0.00047 ± 0.0006 is the only reading on that), M<50 feeds (39% of private; the val M<50 subset transfers to public at c 0.02 ± 0.10), or the test-period cold tail (cold >= 0.45: 33% of private, 21% of public, 8% of val; val cold>=.45 slices put the modern contrasts at -0.0007..+0.0001, i.e. the m02-heavy pools gain nothing there either). Its agreement with public is one sample's worth.
- **Private-specific unknowns** are unobserved by every labelled lens: 1-2 extra days of staleness and the weekday/evening mix. Projected re-orderings from composition are 0.0003-0.0007 ± 0.0003 and point in opposite directions across lenses (val regressions favour the non-G3 pools slightly on colder feeds; the holdout private-like subset leans against the m02-heavy ones); 4-10 more twin readouts would be needed to detect either. Weekend effects are unidentifiable (public 100%, private 24%, val 0%).
- **Realised private noise**: the paired SE between near-copies on 20,033 private feeds is 0.0003-0.0005; the realised private order of pools within ± 0.0005 of each other is a coin flip regardless of what any lens says. G3 is the best-supported pick, not a guaranteed winner by any margin.
- **Reconstruction caveats** (landscape 7): S1 was rebuilt as 0.5 z(ft) + 0.5 z(to) of the stored arrays (its recorded val is not reproduced exactly); PoP3's member list is the union (52) with a nominal re-cool share; old-generation val arrays were rebuilt from recorded recipes for v10/v5 only (match to 2e-5), v3/v4 use recorded numbers; public is "ts < 1790497920" (8,562) throughout although the organisers' wording selects <= (8,580), immaterial at < 0.00003.

---------------------------------------------------------------------------------------------------------------------

## 5. Files

- This report: `<repo>/reports/VAL_PUBLIC_1113.md`
- calib: `<repo>/nn_x/audit/valpub/calib/{findings.txt, vplib.py, c1_calib.py, c2_reweight.py, c3_transfer.py, c4_private.py, c5_extra.py}`, logs/JSON `out/c1_calib.{log,json}`, `out/c2_reweight.{log,json}`, `out/c3_transfer.{log,json}`, `out/c4_private.{log,json}`, `out/c5_extra.log`
- landscape: `<repo>/nn_x/audit/valpub/landscape/{findings.txt, lib.py, s0_check.py, s1_matrix.py, s2_pairs_slices.py, s3_transfer.py, s4_testside.py, s5_extras.py}`, `out/s0_check.log`, `out/s1_matrix.{log,csv}`, `out/s1_deltas_vs_G3.csv`, `out/s2_pairs_slices.log`, `out/s2_{corr.npz,composition.csv,slices.csv}`, `out/s3_transfer.log`, `out/s4_testside.log`, `out/s4_disagreement.csv`, `out/s5_extras.log`
- semantics: `<repo>/nn_x/audit/valpub/semantics/{findings.txt, lib.py, s01_construction.py, s02_pubpriv.py, s03_exploration.py, s04_calibration.py, s05_dispersion.py, s06_holdout_days.py}`, `out/s01_construction.log`, `out/s02_pubpriv.log`, `out/s03_exploration.log`, `out/s04_calibration.log`, `out/s05_dispersion.log`, `out/s06_holdout_days.log`, `out/desc_{val,test}.parquet`
- Prior work cited: `nn_x/pubdiag/noise/noise_out.txt`, `nn_x/pubdiag/shift/p3_slices.log.txt`, `nn_x/audit/critique2/critique2.txt`, `nn_x/audit/publicprobe/findings.txt`, `reports/{eda_shift_overlap,review_shift,sol_advval,INSIGHTS_1112,review_metric,CURRENT_STATUS}.md`, `src/{calib,pmix,presub_gate,metrics,common,stack_val}.py`
