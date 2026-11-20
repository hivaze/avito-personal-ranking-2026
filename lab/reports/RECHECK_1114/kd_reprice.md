# RECHECK 11-14, lens "kd_reprice": which closed ideas does distillation re-price?

Written 2026-11-14 ~20:10 UTC. CPU only, read-only on preds and logs. Scripts and raw outputs: `nn_x/w12/recheck/kd_reprice/` (`m1`..`m16`, `classify.py`, `out/`).
All deltas are paired per feed (val 14,423 feeds; holdout 30,000 feeds, M >= 50, in-period); "a / b" = val / holdout; ± = per-feed SE.
"Measured" = computed here from stored predictions or logs. "Inferred" = follows from a fitted law or an analogy and is marked as such.

## 0. Bottom line

1. **Distillation re-priced exactly one closed lever (width 192) and the reason is now measurable.** All transformer students compute one common mean function: correlation of family means (rho_sys) is 0.9990-0.9997 across teacher (11 / 25 members / generation 2 / cross-fitted), teacher weight 0.3-0.8, width and seed. Pool quality follows NDCG(N) = Q - a/N (fit error <= 0.0001 for N = 1..11). Width 192 raised Q by +0.0002 val / +0.0007 holdout and did nothing for the single member (+0.00007 ± 0.00041 / +0.00038 ± 0.00024). Under hard labels the same width was a null at Q level (-0.0002 / +0.0002), so the re-pricing is real.
2. **The wide-student pool is at its asymptote.** Q = 0.40154 / 0.32463; the mean 8-pool is 0.40133 / 0.32436, the realised W8 is a favourable val draw (0.40157 / 0.32456). Forecast for W16 (inferred): 0.4014 / 0.3245, i.e. not above W8.
3. **The member-level screen used for today's capacity re-tests cannot see the kind of gain width gave** (d192 itself fails it). Re-read at pool level, none of the five capacity levers and none of 55 older hard-label closures is rescued on both witnesses. The closures stand; the screen statistic should change for the arms still running.
4. **Structural diversity exists only between information views, and it is unchanged by distillation.** At fixed pool size, mixing gain G_mix is 0 (|G| <= 0.0004, never positive at 2 SE on both witnesses) for teacher, width, seed, lr, epochs, GRU, retrieval variants; it is +0.0006..+0.0008 val / +0.0010..+0.0013 holdout for the popularity view (pop1) as students, the same as under hard labels. Every view pays about twice its G_mix in strength on val and holdout, so the net is ~0 there; the popularity view is the one whose cost is known to shrink on public (C3).
5. **The running sub-feed arms (kwsub5 / kwsub2) are confounded**: 6 of the 20 feed-context inputs differ by 0.8-3.6 SD between short and long feeds and are not recomputed for the sub-feed. A null there does not close sub-feed distillation.
6. Ranked re-tests are in §6. No closed row reaches P = 0.10 for +0.001 at pool level; the best expected value is a file recipe (pop1-view students at pool weight 0.15), not a new training arm.

## 1. One mean function and the pool-size law (measured)

Member table (`out/m1_members.txt`), train-only level, 4 seeds unless noted.

| arm | member val / hold | member Δ vs reference | within-arm corr | 4-pool val / hold |
|---|---|---|---|---|
| base (hard labels) | 0.39654 / 0.32029 | reference | 0.9485 | 0.39887 / 0.32328 |
| kd05 | 0.39911 / 0.32229 | +0.00257 ± 0.00047 / +0.00200 ± 0.00028 vs base | 0.9776 | 0.40085 / 0.32349 |
| kd08 | 0.39928 / 0.32302 | +0.00017 / +0.00073 ± 0.00022 vs kd05 | 0.9904 | 0.40004 / 0.32377 |
| kd10 | 0.39883 / 0.32256 | -0.00028 / +0.00027 vs kd05 | 0.9950 | 0.39906 / 0.32291 |
| **kd05w192** | 0.39918 / 0.32267 | **+0.00007 ± 0.00041 / +0.00038 ± 0.00024 vs kd05** | 0.9722 | 0.40134 / 0.32408 |
| kd05t25w192 | 0.39952 / 0.32259 | +0.00034 / -0.00007 vs kd05w192 | 0.9723 | 0.40100 / 0.32412 |
| kx05 (cross-fitted teacher) | 0.39916 / 0.32237 | -0.00002 / -0.00030 vs kd05w192 | 0.9736 | 0.40079 / 0.32347 |
| kd05g2w192 | 0.39914 / 0.32252 | -0.00004 / -0.00014 vs kd05w192 | 0.9702 | 0.40080 / 0.32410 |

Pool-size law (`out/m3_ncurve.txt`; random N-subsets, fit NDCG(N) = Q - a/N):

| family (members available) | Q val / hold | a val / hold | N=8 predicted | N=16 predicted |
|---|---|---|---|---|
| wide students (kd05w192 + kd05t25w192, 11) | 0.40154 / 0.32463 | 0.0021 / 0.0021 | 0.40128 / 0.32437 | 0.40141 / 0.32450 |
| narrow students (kd05 + kd05t25, 8) | 0.40133 / 0.32392 | 0.0020 / 0.0017 | 0.40107 / 0.32370 | 0.40120 / 0.32381 |
| hard-label base (6) | 0.40037 / 0.32404 | 0.0040 / 0.0039 | 0.39987 / 0.32355 | 0.40012 / 0.32379 |

- Distillation at width 128 moved Q by +0.0010 val and -0.0001 holdout against the hard-label asymptote: on the in-period holdout it is ensemble compression, not a better function. Width under distillation added +0.0002 / +0.0007 (N-matched 4-pools: +0.00020 ± 0.00034 / +0.00063 ± 0.00020, `out/m5_slices.txt`; no feed-size or history-length slice carries it alone).
- Q does not depend on the teacher weight between 0.3 and 0.8 (two-point extrapolations 0.3240 / 0.3239 / 0.3240 holdout) and falls at 1.0 (0.3230). A weight schedule cannot raise Q.
- Realised W8 = 0.40157 / 0.32456; random 8-subsets of the 11 wide students: 0.40133 (sd 0.00018) / 0.32436 (sd 0.00012); all 11: 0.40127 / 0.32448 (`out/m15_w8luck.txt`). **Forecast (inferred): W16 train-only pool 0.4014 ± 0.0002 / 0.3245 ± 0.0001, i.e. -0.0002 / -0.0001 against the realised W8.** A W16 submission is a near-copy of W8 and tests nothing.
- Student agreement does not degrade under the shift: within-family correlation 0.9737 holdout, 0.9722 val, 0.9718 public, 0.9719 private (label-free, `out/m9_shift_agree.txt`). There is no fidelity gap for transductive distillation on test feeds to close.
- Students need no checkpoint selection: W8 at the last step vs the holdout-selected step +0.00014 ± 0.00020 / -0.00009 ± 0.00011 (`out/m12_w8.txt`). With `@last` predictions the holdout is selection-free for student pools.

## 2. The member screen is blind; pool-level re-read of the closures (measured)

The screen for kw_drop0 / kw_e20 / kw_id / kw_hl2 / kw_ff4 (and, by the 18:48 pre-declaration, for kwsub and kwT) is "member Δ val >= +0.0008 and holdout >= +0.0005". Width 192, the lever that worked, scores +0.00007 / +0.00038 on it and fails.

Mean-function estimate for a 2-seed arm: Qhat = 2·NDCG(2-pool) - mean member NDCG (the N-law at N = 2). Run-to-run sd of one 2-seed Qhat, from the six seed pairs of seven 4-seed arms: 0.0006 val / 0.0003 holdout; a paired arm-vs-reference difference has run sd ~0.0009 / 0.0004 on top of the per-feed SE below.

| arm vs kd05w192 (seeds 951, 952) | member Δ | 2-pool Δ | Qhat Δ (per-feed SE) | reading |
|---|---|---|---|---|
| kw_drop0 | -0.00059 / -0.00019 | -0.00026 / 0.00000 | +0.00007 ± 0.00112 / +0.00018 ± 0.00063 | null (more member noise, same function) |
| kw_e20 | -0.00079 / -0.00041 | 0.00000 / -0.00021 | +0.00078 ± 0.00108 / 0.00000 ± 0.00060 | null |
| kw_hl2 | +0.00019 / -0.00095 | -0.00031 / -0.00102 | -0.00081 / -0.00109 ± 0.00057 | negative |
| kw_ff4 | -0.00107 / -0.00040 | -0.00088 / -0.00064 | -0.00068 / -0.00088 ± 0.00058 | negative |
| kw_id | -0.00571 / -0.00493 | -0.00509 / -0.00456 | -0.00448 / -0.00420 | negative |
| kd05t25w256 | -0.00001 / -0.00062 | -0.00023 / -0.00084 | -0.00045 / -0.00106 ± 0.00061 | negative |
| kd05t25w192b3 | +0.00023 / -0.00074 | -0.00011 / -0.00047 | -0.00044 / -0.00020 | null |
| kd05 (width 128, same statistic) | +0.00009 / -0.00043 | +0.00038 / -0.00077 | +0.00067 / **-0.00111 ± 0.00057** | Qhat sees the width effect the member Δ misses |

All five capacity closures stand. Pool-level re-read of 55 hard-label arms (`out/m8_qlevel_hard.txt`: 17 ten-epoch arms, 32 cooldown arms on roots 101-104, six old 6-seed families): none is positive at 2 SE on both witnesses. Nearest: hzcd 4-pool +0.00079 ± 0.00051 / -0.00011 ± 0.00028; cbs2 +0.00043 / +0.00040 ± 0.00028; m01 -0.00024 / +0.00054 ± 0.00024. Two facts that change the reading of old rows:
- hard-label d192 (F12 vs F7, 6 seeds): member -0.00112 / -0.00063, 6-pool -0.00031 / +0.00004. It was never worse as a function; distillation made it better (+0.0007 holdout).
- cooldown exploration-mix dose 0.01 / 0.04 / 0.10 (the last on the XPOOL pool): member val -0.0004 / -0.0008 / -0.0016, 4-pool -0.0002 / +0.0005, -0.0002 / -0.0001, -0.0002 / -0.0004. The dose curve is flat at pool level, so "pool memorisation" has nothing for soft targets to unlock.

**Recommendation (process):** before kwsub / kwT are read, add by dated addendum a pool-level column (2-pool Δ and Qhat Δ vs kd05w192 on the same seeds) and an extra branch "extend to 4 seeds if Qhat holdout >= +0.0004 and val >= 0". The 4-pool rule already used for kdmixw192 / kd05w224 is the right statistic.

## 3. Which families can serve as students: structural diversity at fixed pool size (measured)

G_mix = NDCG(pool of n/2 members of A + n/2 of X) - mean of the two pure n-pools. Seed noise cancels because every pool has n members. dQ = pure X - pure A. Net at 50/50 = G_mix + dQ/2; at pool weight w the net is about 4w(1-w)·G_mix + w·dQ. Sources: `out/m10_gmix.txt` (n = 4 or 6), `out/m11_gmix2.txt` (n = 2).

| pair (A x X) | n | G_mix val / hold | dQ val / hold | rho_sys | as students? |
|---|---|---|---|---|---|
| kd05w192 x kd05t25w192 (teacher 11 vs 25) | 4 | -0.00004 ± 0.00017 / +0.00009 ± 0.00010 | ~0 | 0.9995 | no value: same function |
| kd05w192 x kx05 (cross-fitted teacher) | 4 | -0.00020 / +0.00014 ± 0.00010 | -0.0006 / -0.0006 | 0.9990 | no |
| kd05w192 x kd05g2w192 (generation 2) | 4 | -0.00003 / -0.00001 | ~0 | 0.9997 | no |
| kd05w192 x kd05 (width) | 4 | -0.00017 / +0.00002 | -0.0005 / -0.0006 | 0.9991 | no (narrow students only dilute) |
| F7 x GRU F10 (hard labels) | 6 | 0.00000 ± 0.00021 / +0.00002 ± 0.00011 | -0.0002 / -0.0001 | 0.997 | **no structural diversity; forecast for the running GRU students: [W8 + 4 GRU] - W8 = +0.0001 holdout (N-law only), below the +0.0003 bar** |
| F7 x lr 1e-3 F9 | 6 | -0.00001 / +0.00009 | -0.0005 / -0.0005 | 0.996 | no |
| F7 x d192 F12, x a4u, x cset | 6 | -0.0002 / -0.0001; -0.0004 / +0.0001; +0.0001 / +0.0001 | within ±0.0006 | 0.995-0.999 | no |
| base x e20, x bs4, x pseudo, x trunc | 2-4 | all within ±0.0004, none at 2 SE on both | | | no |
| **kd05t25 x pop1kd (pop1 view, students)** | 4 | **+0.00057 ± 0.00029 / +0.00127 ± 0.00019** | -0.0015 / -0.0060 | 0.990 / 0.984 | **yes, at low weight (T1)** |
| kd05t25w192 x pop1kd | 4 | +0.00077 ± 0.00029 / +0.00101 ± 0.00019 | -0.0016 / -0.0070 | 0.989 / 0.984 | same |
| base x pop1 (hard labels, 10 ep) | 4 | +0.00069 ± 0.00032 / +0.00143 ± 0.00020 | -0.0014 / -0.0065 | 0.986 / 0.978 | view diversity is the same with and without distillation |
| base x pop2, x pop-free (hard, 10 ep) | 2 | +0.0012 / +0.0018; +0.0005 / +0.0016 | -0.0026 / -0.0069; -0.0014 / -0.0081 | 0.984 / 0.975 | same view as pop1 |
| abl_base x HS1 (feed-surface history view, 2 ep) | 2 | +0.00100 ± 0.00032 / +0.00110 ± 0.00017 | -0.0022 / -0.0022 | 0.983 | zero-sum on both witnesses (T12, low) |
| abl_base x HS1b, x no_retr, x no_set (2 ep); m02 x no-dropout cooldown (d0) | 2 | +0.0015 / +0.0011; +0.0010 / +0.0007; +0.0003 / +0.0006; +0.0006 / +0.0005 | -0.0045; -0.0030 / -0.0024; -0.0025 / -0.0014; -0.0023 | 0.973-0.995 | no: cost exceeds 2 x G (no_set also prices a DIN student) |
| abl_base x ctx_nd removed (2 ep) | 2 | +0.00051 ± 0.00025 / +0.00045 ± 0.00014 | -0.0013 / 0.0000 | 0.994 | marginal |
| w9 base x CATGRAD (2 ep) | 2 | +0.00042 ± 0.00027 / +0.00022 ± 0.00014 | -0.0017 / -0.0011 | 0.990 | no: its "+0.011 diversity margin" is not useful diversity |
| abl_base x JOINT (2 ep) | 2 | +0.00017 / +0.00036 ± 0.00013 | -0.0009 / -0.0001 | 0.999 | no (1.84x cost) |
| m02ref x hzcd (cooldown, 2 roots) | 2 | +0.00019 ± 0.00024 / +0.00020 ± 0.00013 | 0.0000 / -0.0004 | 0.999 | cheap view, small G (T4) |
| kd05w192 x kw_id (item-ID table) | 2 | +0.00004 / +0.00036 ± 0.00018 | -0.0051 / -0.0046 | 0.982 | no |
| R-Drop, SAM, d20, DW3, m04, retr32, blocks3, xrel_more, ap1 | 2-4 | all \|G\| <= 0.0005, none at 2 SE on both | | 0.997-1.000 | no |

Reading:
- The old "diversity margin" (a correlation gap) mixed seed noise with structure. Under hard labels seeds correlated 0.94-0.95, so a family at 0.93 looked diverse; G_mix shows GRU, lr 1e-3, a4u, cset, 20 epochs and the snapshot families carried no structure at all. As students they add pool size only, and pool size is exhausted (§1).
- Structural diversity appears only when an information channel is removed or replaced (popularity inputs, history subset, retrieval, in-feed context). Its strength cost is about 2 x G_mix on val and holdout, and distillation changes neither number (pop1: G +0.0007 vs +0.0006 val, dQ -0.0014 vs -0.0015). **Distillation does not re-price view families.**
- No existing family, at weight 0.15, 0.30 or 0.50, adds to the 11-wide pool on the holdout at 2 SE (52 families, `out/m4_mix.txt`; largest +0.00014 ± 0.00027 for kw_drop0 at 0.50 and +0.00009 ± 0.00010 for generation-2 students at 0.15). On val about half add +0.0002..+0.0006 at weight 0.15 whatever they are (hard base6 +0.00033, kx05 +0.00027, retr32 +0.00064), which I read as val-specific noise of the reference pool rather than a signal (inferred: the same mixes are -0.0001..-0.0004 on the holdout, and G_mix at fixed N is 0 for them).

## 4. Sub-feed distillation arms are confounded (measured)

`--aug_all --aug_cand a` keeps each negative with probability q ~ U(a, 1) and leaves the pre-computed inputs untouched (`cd_arms.py`: "the precomputed xt columns still describe the full feed"). Twenty of the 34 dense inputs are computed over the feed's full candidate set (`ctx_nd_*`, `ctx_hn_*`, `ctx_hs_*`). On val (`out/m14_ctx_format.txt`):

| input | mean at M 5-16 | 17-49 | 50-99 | 100+ | short-vs-long effect (SD) |
|---|---|---|---|---|---|
| ctx_hn_rrw | 0.289 | 0.129 | 0.067 | 0.030 | +3.4 |
| ctx_hn_e1_rrw | 0.289 | 0.129 | 0.067 | 0.030 | +3.6 |
| ctx_hn_top1w | 0.101 | 0.032 | 0.014 | 0.005 | +1.9 |
| ctx_nd_max | 0.716 | 0.795 | 0.841 | 0.881 | -1.6 |
| ctx_hn_margin | -0.040 | -0.121 | -0.158 | -0.190 | +0.9 |
| ctx_hs_mutual | 0.018 | 0.010 | 0.006 | 0.003 | +0.8 |

The other 14 are within 0.5 SD. A 20-candidate sub-feed cut from a 100-candidate feed therefore carries long-feed values on six inputs, 1-3.5 SD away from a real 20-candidate feed, on every training batch. The 11-13 closure of sub-list sampling (row 168) and the r6 `aug` arm had the same defect. If kwsub fails its screen, sub-feed distillation is not closed; the clean form is T2. Interim values of all running arms are inside the plateau jitter of the reference (no reading).

## 5. Closure mechanisms of every COVERAGE_MAP row

Full row-by-row file (235 rows, keyed by line number): `nn_x/w12/recheck/kd_reprice/out/classification.tsv`.

| class | rows | COVERAGE_MAP lines | meaning | what distillation changes |
|---|---|---|---|---|
| A | 29 | 17-18, 22-23, 32, 35, 58-59, 69, 74, 76, 84, 95-96, 108, 114, 122, 128-132, 186, 190-191, 197, 222, 257, 279 | adopted recipe, protocol, statistics, headers | - |
| 1 | 106 | 24, 39-40, 48, 61, 75, 90, 111-112, 120, 137-138, 140-143, 145-150, 157-159, 161, 165, 173-183, 185, 188-189, 192-196, 198-201, 204, 211-212, 218, 221, 225-234, 236, 239, 242, 245-246, 249, 251, 254, 258-274, 276-278, 280-286, 288-291, 293, 295 | no information in the signal: nested residual nulls, label echoes, mechanism closures, arithmetic nulls | nothing |
| 2 | 46 | 19-21, 25-27, 31, 38, 41, 46-47, 50-51, 60, 62-66, 85-88, 97-99, 103, 139, 151-152, 164, 167-168, 184, 203, 207, 209-210, 235, 243-244, 247-248, 287, 292, 296 | noise-limited, memorisation or overfit | 14 rows answered under distillation today (width works; 256, depth, FFN x4, ID table, dropout 0, 20 epochs fail); 4 moot by this recheck (cooldown mix dose is flat at pool level); 18 feed T2, T3, T5-T11; 9 stay closed with no distillation-specific mechanism |
| 3n | 22 | 28-30, 34, 36, 67-68, 101, 113, 163, 166, 187, 205, 213, 216, 219, 237-238, 250, 255-256, 275 | diversity of the noise type: seeds, snapshots, weight averaging, R-Drop / SAM, GRU, lr family, CE+BCE, GDCN, DW3, JOINT, combiners, bagging | worth less than before: the pool is at its asymptote and G_mix = 0 |
| 3v | 11 | 52-53, 78, 89, 110, 144, 202, 208, 214-215, 217 | diversity of the information-view type: pop1 / pop2 / pop-free, hz popularity, input dropout, HS1, CATGRAD, aux head | G_mix and cost both unchanged; only the popularity view has public evidence (T1, T4) |
| 4 | 21 | 33, 37, 49, 77, 79, 100, 102, 109, 115, 121, 123, 160, 162, 220, 224, 240-241, 252-253, 294, 297 | process, leak, judge | two process findings (§2 screen statistic; §1 `@last` makes the holdout selection-free); the test pseudo-label rows 102 / 240 are moot (no agreement gap, §1) |

## 6. Ranked re-tests under distillation

Base for arms: `E_BASE + HYG + --kd_file nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192` (= E:kd05w192; one new case line in `lane.sh`, added by write-new + mv). Cost: one d192 run is about 0.8 GPU-h (195 min at 4 jobs per GPU). P = my probability of >= +0.001 public at pool level. Every read-out should use the 4-pool rule or the Qhat column of §2, never the member Δ alone.

| # | re-test | exact flags or patch | mechanism | why today's failures do not kill it | cost | P |
|---|---|---|---|---|---|---|
| T1 | pop1-view students as a 0.15 admixture to W8 / W16, judged on public | no training for the first file: per branch z = 0.85 z(wide pool) + 0.15 z(mean of `abl_e10_pop1kd05t25_s951-954`); these four have no fine-tune yet (run the n2cpv3s3 fine-tune, or use their train-only pool in both branches). Optional stronger members: the `nn_x/abl/pop1/launch_e10.sh` command (`nn_x/abl/train.py`, E_BASE + HYG + `--xt_exclude $POP --xt_add xp_imp_c50`; the KD block is already ported there) + `--kd_file .../teacher_mean11.npy --kd_w 0.5 --d 192`, 4 seeds | the only measured structural diversity that survives distillation (G_mix +0.0008 ± 0.0003 / +0.0010 ± 0.0002) | the 10:47 closure tested 50/50 on val, where G + dQ/2 = 0 by arithmetic. At weight 0.15: train-only +0.00053 ± 0.00031 val / -0.00043 ± 0.00019 holdout; final level +0.00032 ± 0.00020 val, +0.00046 ± 0.00026 with the hard pop1 fine-tune pool in the FT branch (`out/m12_w8.txt`, `out/m16_t1_final.txt`). Same signature as Q2a / C1 / C3 (val +0.0003..+0.0007, holdout 0..-0.0007, public +0.0004 / +0.0006 / +0.0006). By the project's own rule popularity-light pools are judged on public | ~2 GPU-h fine-tune or 0; 1 submission | 0.06 (P(public > W8) 0.65; expected +0.0003) |
| T2 | sub-feed distillation with sub-feed-consistent context inputs | new helper (about 30 lines, e.g. `cd_arms12.py`) that recomputes the six M-dependent inputs of §4 for the kept candidates from the batch embeddings, plus a 3-line hook after `augment_pool_batch`; then `--aug_all --aug_cand 0.2` | manufactures exploration-format short feeds with teacher targets; train has no exploration feed with M < 50, val has 41% | this is the repair of the confound in §4; it is only worth building once kwsub is read. If kwsub passes as is, extend it instead | 1 day code + 1.6 GPU-h screen | 0.08 |
| T3 | soft-target fine-tune on val | copy `finetune.py` to a new file; target = (1-w) y/sum(y) + w softmax(z of the train-only wide pool on the same val feeds / T), w 0.5; about 10 lines; the teacher uses no val label, so the 5-fold OOF read stays clean | the hard-label fine-tune is the last noise-limited stage: it does not help students (member FT - TO between +0.0001 and -0.0007, wide arms -0.0001 and -0.0007 ± 0.0004; W8 FT branch 0.40087 vs TO branch 0.40157) and lowers M < 50 in 8 of 9 families (-0.0003..-0.0016), `out/m7_ft.txt` | not tested today. Limits: the 0.5 / 0.5 blend already hides the damage (blend - TO-only +0.00009 ± 0.00043 for W8, within ±0.00003 for S16, C3, G3) and the fine-tune's forward-in-time value is invisible on val (G3 TO-only public -0.00047 ± 0.0005) | ~2 GPU-h + 1 submission | 0.05 |
| T4 | hzcd students (stale-popularity cooldown view) | `--hzcd` | under distillation this is privileged-input distillation: the stored teacher saw fresh popularity, the student sees horizon-limited popularity in the cooldown; under hard labels the click still depended on the unseen fresh popularity | closed on a member holdout veto (-0.0005) although the holdout is in-period; at pool level +0.00079 ± 0.00051 / -0.00011 ± 0.00028, G_mix +0.0002 / +0.0002 at zero val cost. Judge with the `holdout_hz` file, stale-val and public | 1.6 GPU-h screen | 0.04 |
| T5 | history-side function matching | `--aug_all --aug_hist 0.1` (no `--aug_cand`) | teacher saw the full history, student a thinned one; an augmentation that hard labels punish (the dropped event may explain the click) | history dropout was closed by an inference-time replay on models never trained with it; never trained at 10 epochs. Caveat: `ctx_hn_*` / `ctx_hs_*` stay stale (smaller than §4). Run only if a sub-feed arm passes | 1.6 GPU-h | 0.04 |
| T6 | K9 auxiliary dose under distillation | two arms: `--aux_pos 0` and `--aux_pos 1.0` | K9 was adopted as a dense de-noising signal; the teacher now supplies one that is aligned with the metric, so K9 may be redundant or still additive | every K9 dose test is hard-label (CD ap1 4-pool +0.00030 ± 0.00030 / +0.00026 ± 0.00016; no_aux at 2 epochs -0.0016 / -0.0018) | 3.2 GPU-h | 0.04 |
| T7 | retrieval capacity | `--retr_m 32`; second arm `--retr_uniq --retr_tcnt` | the capacity that worked sits on the information path (c_lin 256 -> d); retrieval is the other information path | under hard labels null with wide errors (retr32 Q +0.0018 ± 0.0013 / -0.0009 ± 0.0007; a4u 6-pool +0.0006 / -0.0005); today's failed levers (depth, FFN, 256) are generic capacity | 3.2 GPU-h | 0.03 |
| T8 | longer cooldown | `--cooldown_frac 0.35` | the late-cooldown decline is gone under distillation (last - best -0.0001 vs -0.0003 hard, `out/m6_curves.txt`) | kw_e20 doubles passes per feed and its decline returns (-0.0007); this keeps 10 passes. Weak: kw_e20's twice-longer cooldown gained less (+0.0022 vs +0.0028) | 1.6 GPU-h | 0.03 |
| T9 | late dropout | `--drop 0 --drop_cd 0.1` | kw_drop0 leads the reference through the plateau (+0.00067 / +0.00037) and to step 85,000 (+0.0008 / +0.0010), then memorises (-0.0010 holdout by the end, `out/m13_drop0_curve.txt`) | restores the brake where it is needed. Weak: plateau noise reductions are not kept through the cooldown (kd10 over kd05: +0.0019 at the plateau, +0.0003 at the end) | 1.6 GPU-h | 0.02 |
| T10 | heads 6 at width 192 | `--heads 6` | the wide students run 4 heads of 48 dims; F12 used 6 | never varied under distillation | 1.6 GPU-h | 0.02 |
| T11 | typed similarity bias | `--sim_bias 2` | per-event-type scale, same-category and recency biases in target attention; never run at 10 epochs | untested in any regime | 1.6 GPU-h | 0.02 |
| T12 | feed-surface history view (HS1) students | port the 5-line `--kd_file` block to `nn_x/w9/hs1/train.py`, then `--d 192` + kd flags | second view with G_mix +0.0010 / +0.0011 | zero-sum at 2 epochs (dQ -0.0022); only worth it if a 10-epoch student pair shows dQ > -0.0015 | 2 x 0.5 GPU-h screen | 0.02 |

Combined probability that at least one of T1-T12 yields +0.001 public: about 0.3 if all were run (the failures are positively dependent: they share the saturated mean function); the expected total is about +0.0005, most of it from T1 and T2.

What the structure implies for ideas outside the map (inferred):
- Three things moved the mean function in this project: the exploration-format cooldown (F7x m02 vs F7 at pool level +0.00195 ± 0.00064 / +0.00075 ± 0.00035, both witnesses), distillation (+0.0010 val, -0.0001 holdout) and width under distillation (+0.0002 val, +0.0007 holdout). Each changes what the model is trained on or how much function it can hold. No architecture variant and no added feature did. T2 is the direct continuation of the first: exploration-format supervision for M < 50, which train does not contain.
- A plateau-phase gain is mostly lost in the cooldown when it is a noise reduction (kd10 over kd05: +0.0019 at the plateau, +0.0003 at the end; kw_drop0: +0.0004, then -0.0002) and mostly kept when it is a better target (kd05 over base: +0.0026 at the plateau, +0.0020 at the end). Price future plateau-only levers (lr, batch, dropout, noise) at about one sixth of their plateau effect.
- Not ranked, long bet (P 0.04): residual boosting on the cross-fitted teacher. `teacher_oof.npy` now holds honest first-pass logits for all 246.7M train rows, which was the missing piece of the closed RESID-TARGET / RECYCLE ideas (their first-pass scores were in-sample). Patch: load the file and add `off[b["ci"]]` to `lg` in training (3 lines), score offline as (mean logit of the 15 fold models) + residual, use the `@last` checkpoint. It aims at the bias of the mean function, the only thing pooling cannot remove; the fold teacher is weaker than the student pool, so the likely outcome is one more near-copy member.

## 7. Moot by measurement: do not spend GPU or submissions on these

| idea | measured reason |
|---|---|
| W16, W24, more seeds of the wide arms | N-law: +0.00013 from 8 to 16, +0.00006 from 16 to 32; realised W8 already sits above the mean 8-pool on val |
| students of several teachers, teacher-subset bagging, teacher mixes as a diversity source | G_mix 0 across teacher 11 / 25 / generation 2 / cross-fitted (rho_sys 0.9990-0.9997) |
| teacher-weight or temperature schedules aimed at member strength | Q is flat in the weight from 0.3 to 0.8; plateau leads are not kept through the cooldown |
| GRU, CATGRAD, JOINT, lr-1e-3, a4u / cset, 20-epoch students for diversity; DIN and user-fold (xf) students | G_mix 0 or cost > 2 x G_mix (§3). DIN and user-fold are inferred, not measured: DIN from the no_set view, user-fold from the data-volume slope (80% of users costs about -0.0012) with noise-type diversity only |
| larger cooldown exploration mix under distillation | pool-level dose curve flat from 0.01 to 0.10 |
| transductive distillation on val / test feeds | no agreement loss under the shift (0.9737 / 0.9722 / 0.9718 / 0.9719); also leak-grey |
| adding hard-label families to the student pool | 0 of 52 families adds on the holdout at 2 SE at weight 0.15 / 0.30 / 0.50; the lead's W8C5 file is the C3 case (val -0.0003 vs W8) |

## 8. Not checked, and limits

- Everything is train-only level except §6 T1 / T3; the holdout has no fine-tuned branch and no feed with M < 50.
- G_mix for the 2-epoch families (HS1, CATGRAD, JOINT, ablations) uses two seeds; per-feed SEs are given, run noise is not separately estimated there.
- The W16 and GRU forecasts are inferences from the N-law and from hard-label G_mix; both become checkable within hours.
- The pop1 public expectation rests on an analogy with three hard-label builds; the val gain alone (+0.0003..+0.0005, 1.6-1.8 SE) is noise by the project's own threshold.
- I did not test any training arm (no GPU), and I did not look at the automatic feature-engineering work.
