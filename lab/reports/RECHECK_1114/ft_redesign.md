# RECHECK 2026-11-14, lens `ft_redesign`: the fine-tune-on-val stage for distilled students

Reviewer key: `ft_redesign`. CPU only. Scripts and outputs: `nn_x/w12/recheck/ft_redesign/` (`a_member.py` … `n_ftsubset.py`, `out_*.txt`, `ft_cpu.py`, `finetune_kd.py`, `PILOT_PREDECLARED.txt`). Nothing was written to `preds/`, `subs/` or any file a running job uses; the CPU pilot processes were stopped by exact PID after their third fold.

Conventions: FT = `n2cpv3s3` out-of-fold val prediction (5-fold GroupKFold by user, 3 FT seeds); TO = train-only prediction; z = per-feed z-score; pool = mean of member z; file = 0.5 z(FT pool) + 0.5 z(TO pool). Every delta is paired per feed over the 14,423 val feeds; ± is the paired standard error. "Measured" = computed here from stored predictions or logs; "inferred" is marked.

## 1. Verdict

1. **The FT half is worth 0 ± 0.0004 on val at pool level in every pool, students and hard-label alike** (file − TO pool: W8 +0.00009 ± 0.00043, S16 +0.00001 ± 0.00041, C3 −0.00002 ± 0.00039, G3 +0.00015 ± 0.00039). FT alone is below TO alone in every strong pool (W8 −0.00069 ± 0.00061, S16 −0.00087, C3 −0.00060, G3 −0.00044). Measured.
2. **The premise "FT is worse for distilled students" is a member-level effect of the w5_e10 recipe family, not of distillation.** The hard-label twins of the students (w5_e10_base, same seeds) have FT gain −0.00029 ± 0.00041; students minus twin is +0.0003 (kd05), +0.0002 (kd05w192), −0.0005 (kd05t25w192), all within 1.1 SE. Measured.
3. **What the FT does is one shared shift.** 82% of the FT-minus-TO shift variance is common to all W8 members, the W8 shift and the C3 shift correlate 0.90, and the shift of one family can be transplanted onto another family's TO pool with the same result. Pooling cannot average it out: the FT branch stops improving at 4 members (0.4008-0.4009 for 4, 8 and 16 students) while the TO branch keeps rising (0.40117 → 0.40136 → 0.40172). Measured.
4. **The shift has one systematic part and it is small**: a promotion of cold candidates (4% of the shift variance) that carries the whole cold-positive gain (+0.0031 ± 0.0006) and costs warm positives (−0.0011 ± 0.0002), net +0.0001 ± 0.0002. The other 96% (re-ranking inside cold and inside warm candidates) is net −0.0006 ± 0.0006 at full weight. Measured.
5. **Upper bound for any FT redesign that shrinks, anchors or re-weights the same shift**: quadratic fit of NDCG(TO + λ·shift) over four disjoint pools gives a best-case gain of +0.00005 (bootstrap 95% upper bound +0.0005) at λ ≈ 0.2. Measured fit; the bound for new variants is inferred.
6. **CPU pilot of the two cheapest redesigns** (§8; 2 W8 members, 3 of 5 folds = 8,655 feeds, rule written before the first read): soft targets inside the FT give member +0.00053 ± 0.00038 over the plain FT, the half-learning-rate control gives +0.00075 ± 0.00046, and at file level both give nothing (soft targets −0.00036 ± 0.00031, half learning rate −0.00001 ± 0.00032). The pre-declared rule fails: no 8-member GPU run. Measured (file level through a 2-member proxy).
7. **Val/test parity holds**: same function of the inputs (coefficient cosine 0.98), 0.95x the size on test (5-fold averaging), flat over test days and public/private, 94 of 28,595 test feeds share a user with val. One deviation: the cold promotion grows with the calendar day and is 2x larger on the last test day than on val Tuesday; bounded effect < 0.0002 at w_ft 0.5. Measured.

Recommendation: no GPU for FT redesign; keep w_ft = 0.5 as a hedge (one public reading and the private-mix reweighting both favour keeping the FT half, each at about 1 SE); new members do not need their own fine-tune (the FT pool of the 8 existing wide students with the TO pool of all members gives the same file, §4); expect the FT half of W16 to add 0.0000 ± 0.0004 on val.

## 2. What `nn_x/w5/finetune.py` does (read in full)

- Loads the train-only checkpoint, adds a zero-initialised extra branch for 42 val/test-only features (`n2c_pre_v3`: leave-user-out popularity counts, co-visitation, neighbour-density and pre-trained-encoder scores), standardised with val statistics on both val and test.
- 5-fold GroupKFold by user (user-0 feeds are singleton groups), AdamW lr 5e-5 (extra branch x10), weight decay 0.02, warmup 30 steps, cosine to 0 over 4 epochs (about 178 steps per epoch at d192), EMA 0.99 from epoch 0.5, gradient clip 1.0, listwise softmax on the one-hot click target with per-batch normalisation; feeds without a positive carry no loss.
- Saved predictions = the final EMA weights (EMA horizon about 100 steps, so they equal the epoch-4 model). Test = mean raw logit over 5 fold models x 3 seeds.
- Soft-target support: only `--pseudo <name>` (soft targets softmax(teacher z / `--pseudo_T`) on sampled **test** feeds, weight `--w_test`, teacher read from `preds/nn_<name>_test_folds.npy`). There is no soft-target option for val rows.
- Not in the script: any anchor to the starting weights, any parameter freezing.

## 3. Member level: FT minus TO

Mean over members of the per-feed delta (`a_member.py`, `e_extra.py`).

| group | n | TO | all | M<50 | M>=50 | Tue | Thu | user 0 (658) | H 21-100 | H 101-255 | H 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W8 (d192 students) | 8 | 0.39935 | −0.00043 ± 0.00041 | −0.00114 ± 0.00067 | +0.00007 ± 0.00051 | −0.00098 ± 0.00056 | +0.00018 ± 0.00059 | +0.0013 ± 0.0020 | +0.0012 ± 0.0010 | −0.0006 ± 0.0006 | −0.0012 ± 0.0007 |
| N8 (d128 students) | 8 | 0.39934 | −0.00001 ± 0.00039 | −0.00041 ± 0.00066 | +0.00027 ± 0.00048 | −0.00003 ± 0.00053 | +0.00001 ± 0.00057 | +0.0023 ± 0.0021 | +0.0022 ± 0.0010 | −0.0002 ± 0.0006 | −0.0007 ± 0.0007 |
| kd08 | 4 | 0.39928 | +0.00013 ± 0.00048 | −0.00071 ± 0.00080 | +0.00071 ± 0.00059 | −0.00006 | +0.00034 | +0.0013 ± 0.0025 | +0.0026 ± 0.0012 | +0.0004 ± 0.0007 | −0.0013 ± 0.0008 |
| w5_e10_base (hard-label twin, 1 FT seed) | 4 | 0.39654 | −0.00029 ± 0.00041 | −0.00050 ± 0.00071 | −0.00014 ± 0.00049 | | | | | | |
| m02 re-cools (hard) | 7 | 0.39667 | +0.00046 ± 0.00040 | −0.00034 ± 0.00066 | +0.00101 ± 0.00050 | +0.00016 | +0.00079 | +0.0023 ± 0.0019 | +0.0017 ± 0.0010 | +0.0005 ± 0.0006 | −0.0004 ± 0.0007 |
| C3 members (hard) | 28 | 0.39558 | +0.00073 ± 0.00032 | −0.00022 ± 0.00053 | +0.00138 ± 0.00040 | +0.00036 ± 0.00044 | +0.00113 ± 0.00046 | +0.0018 ± 0.0016 | +0.0014 ± 0.0008 | +0.0008 ± 0.0005 | +0.0003 ± 0.0006 |

Paired contrasts of the FT gain: W8 − C3 members −0.00115 ± 0.00028; W8 − m02 −0.00088 ± 0.00032; N8 − m02 −0.00047 ± 0.00031; W8 − N8 −0.00042 ± 0.00026; **base twin − m02 −0.00075 ± 0.00035**; kd05 − base twin +0.00030 ± 0.00039; kd05w192 − twin +0.00017 ± 0.00040; kd05t25w192 − twin −0.00045 ± 0.00040.

Reading: the missing FT gain is shared by all 24 w5_e10 members, hard-label twins included; distillation does not change it (three of four student arms are above their twin); width 192 costs a further −0.0004 (1.6 SE). No slice (M, weekday, user 0, history length) isolates the loss; long histories (H 256) are the weakest slice in every group, students and hard-label.

Per-epoch OOF from the FT logs (`ft_logs.py`; raw weights, epoch 0 = TO):

| group | n | epoch 1 | epoch 2 | epoch 3 | epoch 4 | saved (final EMA) |
|---|---|---|---|---|---|---|
| W8 | 8 | +0.00047 | −0.00013 | −0.00038 | −0.00054 | −0.00044 |
| N8 | 8 | +0.00066 | +0.00027 | +0.00001 | −0.00010 | +0.00001 |
| w5_e10_base | 4 | +0.00038 | −0.00001 | −0.00028 | −0.00035 | −0.00029 |
| m02 re-cools | 7 | +0.00106 | +0.00063 | +0.00047 | +0.00041 | +0.00046 |
| GRU | 6 | +0.00115 | +0.00088 | +0.00083 | +0.00077 | +0.00074 |
| pop1 | 8 | +0.00152 | +0.00135 | +0.00118 | +0.00108 | +0.00117 |

Epoch 1 beats the saved prediction in 20 of 20 student fine-tunes (mean +0.0007). At member level this equals a linear shrink of the saved shift: z_TO + 0.2·shift gives +0.00033 ± 0.00011 for W8 members, the logged epoch 1 gives +0.00047. It does not survive pooling (§4).

## 4. Pool level and the blend weight

`b_pool.py`, `l_transplant.py`.

| pool | n | FT | TO | file | FT − TO | file − TO |
|---|---|---|---|---|---|---|
| W8 | 8 | 0.40087 | 0.40157 | 0.40165 | −0.00069 ± 0.00061 | +0.00009 ± 0.00043 |
| N8 | 8 | 0.40107 | 0.40119 | 0.40104 | −0.00012 ± 0.00059 | −0.00015 ± 0.00039 |
| S16 | 16 | 0.40085 | 0.40172 | 0.40173 | −0.00087 ± 0.00060 | +0.00001 ± 0.00041 |
| C3 | 28 | 0.40043 | 0.40103 | 0.40101 | −0.00060 ± 0.00057 | −0.00002 ± 0.00039 |
| G3 (3-seed FT) | 20 | 0.39991 | 0.40035 | 0.40051 | −0.00044 ± 0.00059 | +0.00015 ± 0.00039 |
| m02 x7 | 7 | 0.40053 | 0.40061 | 0.40067 | −0.00008 ± 0.00059 | +0.00006 ± 0.00040 |
| pop1 x8 | 8 | 0.39863 | 0.39841 | 0.39950 | +0.00023 ± 0.00068 | +0.00110 ± 0.00046 |

Only the weak pop1 pool gains from the FT half. Students by pool size (random subsets of the 16 students): the FT half is "ensembling for small pools".

| members | FT | TO | file | file − TO |
|---|---|---|---|---|
| 1 | 0.39918 | 0.39926 | 0.39966 | +0.00040 |
| 2 | 0.40006 | 0.40022 | 0.40064 | +0.00042 |
| 4 | 0.40080 | 0.40117 | 0.40135 | +0.00018 |
| 8 | 0.40094 | 0.40136 | 0.40147 | +0.00011 |
| 16 | 0.40085 | 0.40172 | 0.40173 | +0.00001 |

The FT branch does not need every member (`n_ftsubset.py`): for the 16 students, file with the FT pool of W8 only and the TO pool of all 16 = 0.40172, −0.00001 ± 0.00023 vs fine-tuning all 16 (test correlation of the two files 0.9997); for W8, the FT pool of either 4-member arm gives −0.00006 ± 0.00028 / −0.00004 ± 0.00026. Smaller or narrow FT pools are 0.0002-0.0005 lower (with the 16-student TO pool: FT of N8 −0.00040 ± 0.00026, FT of one 4-member wide arm −0.00023 ± 0.00030 / −0.00051 ± 0.00029; with the W8 TO pool: FT of 2 members −0.00039 ± 0.00037).

Slices of file − TO (W8): M<50 +0.00018 ± 0.00076, M≥50 +0.00002 ± 0.00049, Tue +0.00005 ± 0.00057, Thu +0.00012 ± 0.00063, user 0 +0.0011 ± 0.0021, H 21-100 +0.0016 ± 0.0011, H 101-255 −0.0003 ± 0.0006, H 256 −0.0002 ± 0.0008. FT pool − TO pool is negative on full-length histories in every pool (H 256: W8 −0.0021 ± 0.0011, S16 −0.0020 ± 0.0011, C3 −0.0018 ± 0.0010, G3 −0.0014 ± 0.0010) and positive on H 21-100 (+0.0013..+0.0028, SE 0.0015); a history-dependent w_ft belongs to the closed routing class and would be worth at most +0.0001..+0.0003.

Blend weight w_ft (file = w·z(FT) + (1−w)·z(TO)), W8, delta vs TO pool: 0.1 +0.00040, 0.2 +0.00025, 0.3 +0.00036, 0.4 +0.00039, 0.5 +0.00009, 0.6 +0.00002, 0.7 −0.00060, 1.0 −0.00069. Paired vs 0.5: w 0.3 +0.00028 ± 0.00026, w 0.0 −0.00009 ± 0.00043, w 0.7 −0.00068 ± 0.00025.

Nested choice of w_ft (grid 0..1 step 0.1 chosen on one user half, applied to the other; 20 random user-half splits, both directions):

| pool | full-val argmax | nested pick, mean (sd) | nested gain vs 0.5 (sd over splits) |
|---|---|---|---|
| W8 | 0.1 | 0.23 (0.14) | +0.00009 (0.00026) |
| N8 | 0.2 | 0.42 (0.31) | +0.00005 (0.00033) |
| S16 | 0.5 | 0.28 (0.26) | −0.00031 (0.00032) |
| C3 | 0.6 | 0.35 (0.28) | −0.00022 (0.00020) |
| G3 | 0.3 | 0.33 (0.17) | −0.00004 (0.00021) |

Verdict: w_ft ≠ 0.5 stays CLOSED for students too. The nested optimum sits near 0.25-0.35 in every pool and is worth nothing.

Quadratic model d(λ) = a·λ − b·λ² of NDCG(z(TO pool) + λ·pooled shift) − NDCG(TO pool), least squares on λ = 0.1..1.0: W8 a +0.0014, b 0.0023 (λ* 0.30, max +0.0002); N8, C3, kd08 a ≤ +0.0002. Mean of the four disjoint pools: a +0.0004 (bootstrap sd 0.0007), b 0.0009 (sd 0.0006), best gain +0.00005, bootstrap 95% upper bound +0.0005. A variant that only shrinks or anchors the shift moves along this curve.

## 5. How far the FT moves predictions, and what moves

`c_corr.py`, `e_extra.py`.

| group | corr z(FT), z(TO): val OOF | test | mean per-feed Spearman: val | test | RMS shift: val | test |
|---|---|---|---|---|---|---|
| W8 members | 0.9889 | 0.9902 | 0.9818 | 0.9840 | 0.149 | 0.140 |
| N8 members | 0.9911 | 0.9920 | 0.9847 | 0.9864 | 0.134 | 0.126 |
| m02 members | 0.9884 | 0.9895 | 0.9811 | 0.9829 | 0.152 | 0.145 |
| GRU members | 0.9864 | 0.9873 | 0.9785 | 0.9801 | 0.165 | 0.160 |
| pop1 members | 0.9890 | 0.9902 | 0.9805 | 0.9827 | 0.149 | 0.140 |
| W8 pool | 0.9907 | 0.9916 | | | 0.137 | 0.129 |
| C3 pool | 0.9924 | 0.9930 | | | 0.124 | 0.118 |

- A student's FT moves its predictions as much as a hard-label member's FT (0.9889 vs 0.9884); narrow students move slightly less.
- Common-mode share of the shift (variance of the member-mean shift / mean member shift variance): W8 0.82 (pairwise shift correlation 0.79), N8 0.86, m02 0.78, all 28 C3 members 0.61. Same on test (0.83 / 0.87 / 0.79 / 0.61).
- Common shift W8 vs C3: correlation 0.904 (val and test); W8 vs N8 0.96. Putting the N8, C3 or 4-family-mean shift on the W8 TO pool gives −0.0003 .. −0.0005 at full weight, the own shift −0.0007: interchangeable within ± 0.0004, so cross-family averaging does not de-noise it.
- FT makes members more alike: member-member correlation W8 0.9726 (TO) → 0.9736 (FT); m02 0.950 → 0.959; GRU 0.941 → 0.958.
- A single plain fine-tune already moves a member 13% of the way to its pool (pilot, §8), which is the member-level FT gain that pooling makes redundant.

## 6. Val/test parity of the shift

`d_parity.py`, `f_overlap.py`, `j_misc.py`, `c_corr.py`.

- **Size.** RMS shift test / val = 0.94-0.95 for every family, every M bucket (0.938-0.949) and flat over test days (Sat 0.1294, Sun-am 0.1310, Sun-pm 0.1286, Mon 0.1290, Tue 0.1298; public 0.1298, private 0.1292). The 5% is fold averaging: a single fold model on test has the val-OOF size (0.1548 vs 0.1539), fold-to-fold shift correlation is 0.85, and the fold-specific part is common to members that share the fold (correlation 0.73 same fold, −0.18 different folds). On val OOF this fold noise is in every prediction (11% of the pooled shift variance), on test it is averaged over 5 folds. Inferred size of the resulting OOF pessimism: ≤ +0.0002 for FT alone, ≤ +0.00005 for the file.
- **Direction.** Ridge of the pooled W8 shift on 86 input columns (42 extra features as the model sees them, their per-feed z, z_TO, z_TO²): fit on val R² 0.180 (0.218 applied to test), fit on test R² 0.223 (0.175 applied to val), coefficient cosine 0.983. The FT applies the same function to test.
- **Inputs.** Under the val standardisation the test means of the 42 features stay within 0.11 sd (largest: `it_hist_all_luo` −0.109, public −0.08, private −0.12; `it_hist_et1_luo` −0.103; `seq_cv_pop` −0.089); NaN share of `pre_idmax32` 0.553 val vs 0.595 test. Rank and z channels are exact.
- **Users.** 94 of 28,595 test feeds (0.3%) belong to a user with a val feed; no test candidate was a val positive of the same user. GroupKFold OOF does not hide a user-memorisation channel.
- **Deviation.** Cold-minus-warm mean shift per feed (z units, SE 0.0007): val Tue +0.021, Thu +0.029; test Sat +0.034, Sun +0.037, Mon +0.040, Tue +0.044. It is not composition (at matched feed cold share test is still 1.3-2.6x val). Val labels do not ask for a growing offset: the best global cold offset on the TO pool is 0 on Tue and 0 on Thu (`k_offset_day.py`). Size of the effect: a global cold offset of 0.03 changes the W8 TO pool by −0.00008 ± 0.00015, so at w_ft 0.5 the extrapolated part is below 0.0002 in either direction.

## 7. What the shift is made of

`h_cold.py`, `i_decomp.py`. Cold = `it_hist_all_luo == 0` (38% of val candidates, 43% public, 46% private). coldpos / warmpos = feeds whose positives are all cold / all warm (3,422 / 9,118 feeds).

| W8, added to z(TO pool) | all | coldpos | warmpos | public mix | private mix |
|---|---|---|---|---|---|
| full shift (= FT pool) | −0.00069 ± 0.00061 | +0.00283 ± 0.00120 | −0.00228 ± 0.00081 | −0.00007 ± 0.00075 | +0.00051 ± 0.00095 |
| (i) cold/warm offset only (4.1% of the variance) | +0.00008 ± 0.00021 | +0.00313 ± 0.00059 | −0.00106 ± 0.00023 | +0.00017 ± 0.00031 | +0.00025 ± 0.00041 |
| (ii) within-group re-ranking only (95.9%) | −0.00063 ± 0.00060 | +0.00018 ± 0.00119 | −0.00105 ± 0.00081 | +0.00015 ± 0.00075 | +0.00086 ± 0.00093 |
| 0.5 full (= the file) | +0.00009 ± 0.00043 | +0.00167 ± 0.00075 | −0.00066 ± 0.00058 | +0.00066 ± 0.00054 | +0.00104 ± 0.00070 |

Same split for the other pools, part (i) / part (ii), all feeds: S16 +0.00023 ± 0.00020 / −0.00081 ± 0.00060; C3 +0.00041 ± 0.00021 / −0.00080 ± 0.00057; G3 +0.00029 ± 0.00020 / −0.00048 ± 0.00058.

- The FT half is a cold/warm trade in every pool (file − TO: G3 coldpos +0.0029 ± 0.0008, warmpos −0.0012 ± 0.0005; N8 +0.0024 / −0.0012; C3 +0.0007 / −0.0004), as VAL_PUBLIC_1113 found for G3. Students are the same.
- Under the `src/pmix.py` public / private feed mix (M x cold-share reweighting of val, label-free weights; a veto lens by project rule, not a promoter) file − TO is +0.0007 ± 0.0005 / +0.0010 ± 0.0007 for W8, +0.0005 ± 0.0005 / +0.0008 ± 0.0006 for G3 (the one public reading of that contrast is +0.00047), +0.0004 / +0.0007 for C3, +0.0001 / +0.0002 for S16. This is the whole case for keeping the FT half: about 1 SE, twice.
- A global cold offset on the TO pool (one parameter, nested by user halves) is worth −0.0003 (W8), −0.0002 (S16, C3), 0.0000 (G3): the TO pools have no first-order cold miscalibration. "Cold offsets" stays CLOSED.

## 8. CPU pilot: soft targets inside the FT, and half learning rate

Purpose: the cheapest falsification of "KD inside the fine-tune" (lever 6 of TOP1_GAP_EXTERNAL_1114) and of "smaller step", before any GPU time. `ft_cpu.py` is a float32 CPU port of the n2cpv3 recipe (1 FT seed, same folds, same seeds 1000k, same batch order); it imports the unchanged `nn_x/w5/model.py` and `data.py`. Reading rule: `PILOT_PREDECLARED.txt`, written 19:36 UTC before any fold was read. Runs were stopped after 3 of 5 folds (8,655 held-out feeds; 2 s per step on 2 threads).

Arms, members w5_e10_kd05w192_s951 and w5_e10_kd05t25w192_s951 (first seed of each W8 arm):
- A = n2cpv3 baseline.
- B = A with soft targets, y ← 0.5·y/Σy + 0.5·softmax(teacher), teacher = mean raw logit of the 8 W8 train-only members on val (label-free; the `train.py` KD formula).
- C = A at lr 2.5e-5 (control for B, whose hard-label gradient is halved); first member only (thread cap).

Sanity (passes): the CPU baseline against the stored GPU `n2cpv3s3` OOF on the same feeds has per-feed-z correlation 0.99955 / 0.99953 and NDCG delta +0.00007 ± 0.00039 / +0.00055 ± 0.00035. (Side reading: the same train-only weights scored in float32 instead of bf16 differ by per-feed-z RMS 0.011 and NDCG +0.00062 ± 0.00027 / −0.00028 ± 0.00023 at member level on the 8,655 feeds: opposite signs for the two members, no ties problem, `m_bf16.py`.)

Final-EMA predictions (what the recipe saves), paired per feed. File proxy = 0.5 z(z(TO pool W8) + member shift) + 0.5 z(TO pool W8), minus the TO pool; "de-projected" removes the member-to-pool component of the shift, which a soft-target arm adds by construction.

| quantity (3 folds, 8,655 feeds) | A baseline | B soft targets | B − A | C half lr (member 1) | C − A (member 1) |
|---|---|---|---|---|---|
| member − TO, mean of 2 members | −0.00058 ± 0.00067 | −0.00004 ± 0.00052 | **+0.00053 ± 0.00038** | | |
| member − TO, member 1 | −0.00002 ± 0.00079 | +0.00045 ± 0.00063 | +0.00047 ± 0.00048 | +0.00072 ± 0.00067 | **+0.00075 ± 0.00046** |
| RMS shift (z), member 1 | 0.155 | 0.114 | | 0.120 | |
| share of the shift that points at the pool | 0.133 | 0.165 | | | |
| file proxy − TO pool, mean of 2 | +0.00054 ± 0.00053 | +0.00018 ± 0.00042 | **−0.00036 ± 0.00031** | | |
| file proxy, de-projected, mean of 2 | +0.00038 ± 0.00053 | +0.00019 ± 0.00042 | −0.00019 ± 0.00029 | | |
| file proxy − TO pool, member 1 | +0.00033 ± 0.00059 | −0.00003 ± 0.00048 | −0.00036 ± 0.00038 | +0.00032 ± 0.00051 | −0.00001 ± 0.00032 |

Rule read-out:
- (i) B − A at member level ≥ +0.0005 **and** ≥ 2 SE: +0.00053 at 1.4 SE. FAIL.
- (ii) file proxy of B minus that of A ≥ +0.0003: −0.00036 ± 0.00031. FAIL.
- B − C on member 1: −0.00028 ± 0.00044, so the soft target adds nothing over a smaller step.
- Epoch 1 (raw weights) instead of the final EMA, baseline arm, mean of 2: member +0.00045 better, file proxy −0.00017 (+0.00037 ± 0.00045 vs +0.00054 ± 0.00053). Early stopping does not help the file either.

Verdict: soft targets, a halved learning rate and early stopping all cut the shift by about a quarter and lift the member by +0.0005..+0.0008, and none of them moves the file. This is the shrink curve of §4 seen in a real fine-tune. KD inside the FT is CLOSED at the pilot gate (2 members, 3 folds, proxy error ± 0.0004: a file-level effect above +0.0007 is excluded at about 2 SE; smaller ones are not).

Proxy accuracy, for reference (`e_extra.py`): with the stored GPU predictions, z(TO pool) + one member's shift lands within −0.0009..+0.0003 of the true FT pool across the 8 members (mean −0.0002).

## 9. FT variants (at most 5), with flags and test plans

`finetune_kd.py` in the recheck folder is a copy of `nn_x/w5/finetune.py` with four opt-in flags (`--kd_val`, `--kd_w`/`--kd_T`/`--kd_all`, `--l2sp`, `--train_only`); defaults reproduce the original code path; it compiles; its GPU path was not executed here (the CPU port `ft_cpu.py` runs the same logic). P = my probability of ≥ +0.001 public at pool level.

| # | variant | exact command (on top of `--extra n2c_pre_v3 --lr 5e-5 --epochs 4 --decay cos --ema 0.99 --u0_singleton`) | mechanism | evidence here | test | P |
|---|---|---|---|---|---|---|
| V1 | Soft targets on val rows (KD inside the FT) | `finetune_kd.py --kd_val nn_x/w12/recheck/ft_redesign/out/teacher_w8to_val.npy --kd_w 0.5 --kd_T 1.0` (teacher = mean raw logit of the 8 W8 train-only members on val: label-free, so no cross-fitting is needed) | output-space anchor to the TO pool plus a pull of each member towards its pool | member +0.0005 ± 0.0004, file proxy −0.0004 ± 0.0003 (§8) | done on CPU; rule failed, no GPU run | 0.01 |
| V2 | Smaller step | `finetune.py --lr 2.5e-5` or `--epochs 1` (flags exist) | stop before the label fit turns negative (epoch 1 > saved in 20/20 students) | member +0.0008 ± 0.0005, file proxy 0.0000 ± 0.0003 (§8, one member); pool level bounded by §4 | if run at all: 8 W8 members x 3.3 GPU-min (1 seed); read file − current file, pass at ≥ +0.0005 and 2 SE | 0.01 |
| V3 | Low-dimensional FT | `finetune_kd.py --train_only extra.,wide.` | keeps the only systematic part (the cold offset, expressible through the extra branch) and removes trunk re-fitting, the source of the 96% net-noise part | part (i) alone +0.0001 ± 0.0002 on val, +0.0003 ± 0.0004 private mix | same 8 x 3.3 GPU-min; pass rule as V2 plus the cold/warm table | 0.02 |
| V4 | Anchor on test feeds (existing `--pseudo` path, no patch) | copy `out/nn_W8TOpool_test_folds.npy` to `nn_x/w5/preds/`, then `finetune.py --pseudo W8TOpool --w_test 0.3 --pseudo_T 1.0 --pseudo_n 12000` | anchors the test predictions to the TO pool where they are scored; stops the day-growing cold promotion of §6 | none; the old closure (row "pseudo-labels (test)") used a fine-tuned single-model teacher in the pre-Muon recipe; here the teacher is a label-free ensemble | OOF is blind to it; label-free read (test shift RMS and cold offset by day) then one public probe | 0.02 |
| V5 | No training: file-level | TO-only W8 file (= z of `preds/ens_W8kd_to_test.npy`) as a public probe | measures the FT half on public for students; test correlation to the W8 file 0.9979, the same as G3 TO-only vs G3 (paired public SE 0.0005) | val −0.00009 ± 0.00043; G3 analogue read −0.00047 on public | 1 submission; pooled with the G3 pair the FT-half public value gets SE 0.00035; change nothing unless the pooled value is below 0 | 0.03 (that TO-only beats W8 by ≥ 0.001) |

All five move along, or stay under, the curve of §4. None is expected to reach +0.001. V1 failed its pre-declared pilot rule; V2 was the pilot's control arm (one member) and shows the same pattern (member up, file flat); V3 and V4 are listed for completeness and I do not recommend running them; V5 costs one submission slot and no compute.

## 10. Future work seen from this lens (ranked)

1. **Take the FT stage off the critical path.** For W16 and any later student pool, build the file as 0.5 z(FT pool of the 8 existing wide students) + 0.5 z(TO pool of all members). Measured on the 16-student analogue: −0.00001 ± 0.00023 vs fine-tuning every member. No new fine-tunes, no waiting for them. Expected public effect 0.
2. **Stop testing FT recipe changes of the shrink class** (learning rate, epochs, EMA, anchors, soft targets from the same pool, blend weight). §4 bounds them at +0.0005 (95%) and §8 shows the mechanism directly: they move the member, not the file.
3. **Gate for any future FT idea: new information, not a new fit.** Label re-fitting on 11.5K feeds has first-order value a = +0.0004 ± 0.0007 at pool level, and the 42 extra inputs contribute one cold offset. The FT branch can only pay if a val/test-only input passes a nested residual test on the W8 TO pool (≥ +0.0005 at 2 SE, user halves) first; the extra branch is then the place to inject it. Without such an input, do not reopen the FT.
4. **One optional public probe** (V5): W8 TO-only. With the G3 pair it gives the public value of the FT half to ± 0.00035. It can show that the FT half is useless on public, not that more FT is better.
5. **Judge FT-side ideas at file level on CPU first.** `ft_cpu.py` reproduces the GPU fine-tune (per-feed-z correlation 0.9995) at 25 minutes per fold on 2 threads; two members and the pool proxy separate a +0.0007 file effect from zero. Member-level FT gains are not evidence: a plain fine-tune moves a member 13% of the way to its own pool.

## 11. What I could not check

- Pool-level effect of a redesigned FT with all 8 members (needs GPU; the pilot uses 2 members, 3 of 5 folds and a pool proxy whose own error is ± 0.0004, §8). Folds 3-4 of the pilot were not run.
- Whether the FT half is worth more on test than on val OOF. The private-mix reweighting (+0.0010 ± 0.0007 for W8) and one public pair (+0.00047 ± 0.00050 for G3) point the same way at about 1 SE each.
- Why w5_e10 members respond worse to FT than the m02 re-cools (−0.00075 ± 0.00035, 2.1 SE; same FT arguments and the same 42 feature files, checked from the logs). Not resolved; could be noise.
