# NEXT_1115 / merge_verify: adversarial review of the study "merge" (merge-compat check v2 and per-day drift)

Written 2026-11-14 23:1x UTC. CPU only (12 threads, one process at a time), no GPU, nothing written outside `nn_x/w12/next/merge_verify/` and this file.
[M] = measured by me with my own code (`v_common.py`, no import of the study's scripts), [I] = inferred. ± = 1 SE paired per feed; val 14,423 feeds, test 28,595 (public 8,562, private 20,033).

## 0. Verdict

1. **Verdict lines 1, 2, 3 and 5 of the study are CONFIRMED** [M]: every number I re-derived agrees to the last printed digit, bootstrap bounds within 0.0001 with another seed (correlation definitions, both known answers, all 10 calibration rows, refit, bootstrap, 55 pairs, the full drift table).
2. **Verdict line 4: the advice stands (ask for model scores), its numbers and mechanism do not** [M]. The study's "v2 way" column is an equal mix in which 2-15% of the val rows are EXACTLY tied (items with swapped ranks), and the ties are resolved in the PARTNER's order by a floating-point residue of 2e-12 (repeated z-scoring) that nobody chose. The tie-break alone moves the 10-partner mean from -0.0011 (partner's order) to -0.0008 (random) to -0.00005 (our order; 7 neural partners +0.0003) and to -0.0057 if left to the metric. Tie-free reads (w 0.4 / 0.6) show a real rank-only cost that grows as the correlation falls: within ± 0.0003 at rho 0.999, -0.0003..-0.0010 at rho 0.983 (GRU + base), -0.0003..-0.0018 at rho 0.90 (LLM + XGBoost), each ± 0.0003-0.0006.
3. **Two defects in `merge_check_v2.py` that can change a decision** [M]: (a) with a rank-file partner the val check at w = 0.5 reads whatever the residue gives: `--ours subs/cand_W8TOkd_r3.parquet` reads the equal mix with C3 at -0.00557 ± 0.00040 (z -13.8), truth +0.00004 ± 0.00040; multiplying our raw scores by 3 moves the Pall read from -0.00200 to -0.00055. (b) The reader tests the mean correlation only: a file with 10% of its feeds inverted, or 12% of its feeds shuffled, gets no FAIL, no WARN and "PROCEED" (law +0.0033 / +0.0019).
4. **Line 3 is right and, if anything, too soft**: the 55-pair interval with a positive lower bound rests on 3 pairs of two weak non-neural models; without them it is [-0.0000, +0.0033] at (0.90, 0) and [-0.0005, +0.0024] at (0.90, 0.002) [M].
5. Rules: no read-only file changed, no GPU process of the study, no file in `subs/`, no label leak, thresholds unchanged [M]. Gate 3 is read on the study's own weight w*, an interpretation the lead has to confirm.
6. **What to conclude instead of line 4:** exchange model scores (exact rho, no ties, and a rank-only cost of the order of 0.0005-0.001 for a partner at rho 0.90-0.98 [M on weaker pairs, I for a strong partner]). Do not use the tool's equal-mix val line or `--blend --w eq` with a rank-file partner until the tie-break is explicit, and add a per-feed distribution check to the reader before any untrusted file is read.

## 1. Claim by claim

| # | claim of the study | study's number | my number [M] | agree |
|---|---|---|---|---|
| 1 | W8 file vs C3 file, test, pooled Spearman of the rank files (v1) | 0.9908 (M>=50 0.9914, M<50 0.9850) | 0.9908 (0.9914, 0.9850) | yes |
| 2 | same pair, pooled Pearson of per-feed z of the model scores (lens) | 0.9938 | 0.9938 (0.9938, 0.9934) | yes |
| 3 | same pair, feed-mean Pearson (the law on record) | test 0.9935 (0.9938 / 0.9929), val 0.9935 | test 0.9935 (0.9938 / 0.9929), val 0.9935 | yes |
| 4 | the law on record was fitted with the feed-mean Pearson on val | `exp7_pairs.py pf_corr` | code read: `(bincount(fid, a*b)/M).mean()` on per-feed z; `solvers/out/ensemble/exp7.json` "val_fit_gap<0.01" = 0.018577, 0.37766 | yes |
| 5 | both files are the rank transform of the `preds/ens_*` blend | 99.9998% / 99.9999% | 99.9999% / 100.0000% | yes |
| 6 | v2 rho from two rank files (shape-restored) | 0.9948 feed-mean / 0.9943 pooled | 0.9948 / 0.9943 | yes |
| 7 | error of the shape restoration <= 0.0020 in rho | neural mean +0.0014 (max 0.0020), non-neural +0.0009 (max 0.0015) | C3 +0.0013, S16 +0.0004, GRU +0.0020, CatBoost +0.0010, XGBoost +0.0002, LLM +0.0015 (6 of 10 checked) | yes |
| 8 | v1's Spearman is 0.003-0.024 too low | C3 -0.003 ... XGBoost -0.024 | C3 -0.0027, GRU -0.0044, CatBoost -0.0125, LLM -0.0127, XGBoost -0.0238 | yes |
| 9 | W8 + C3: gap, rho, law on record, refit, measured equal z-mix | +0.00064 ± 0.00055, 0.9935, -0.00012, -0.00013, -0.00027 ± 0.00039 | identical | yes |
| 10 | W8 + CatBoost | +0.01964 ± 0.00144, 0.8397, -0.00444 (test rho 0.8375: -0.00440), -0.00434, -0.00458 ± 0.00096 | identical | yes |
| 11 | the other 8 calibration rows (G3, Pall, Cd, S16kd, GRU x6, base x4, XGBoost, LLM x2) | table 1.2 | identical to 5 decimals in val, gap ± SE, both rho, both laws, measured ± SE | yes |
| 12 | law error (law - measured), 10 partners | on record mean -0.00009, rms 0.00035, max 0.00098; refit +0.00007, 0.00023, 0.00043 | identical; largest \|error\| / paired SE = 0.89 (XGBoost) | yes |
| 13 | refit of a, b | pooled rho 0.0103 / 0.304; feed-mean 0.0089 / 0.296; corr(1 - rho, gap) 0.992 | identical | yes |
| 14 | feed bootstrap of a, 5-95% | [-0.012, +0.035], sd 0.0142 | [-0.0126, +0.0351], sd 0.0145 (own seed, 2,000 resamples); better file re-decided in each resample: [-0.0128, +0.0353] | yes |
| 15 | equal-mix gain at (rho 0.90, gap 0) / (0.90, 0.002) / (0.95, 0) / (0.86, 0), 10 partners, 5-95% | [-0.00118, +0.00350] / [-0.00147, +0.00258] / [-0.00059, +0.00175] / [-0.00165, +0.00490] | [-0.00126, +0.00351] / [-0.00157, +0.00256] / [-0.00063, +0.00176] / [-0.00177, +0.00492]; 2.5-97.5% at (0.90, 0): [-0.0019, +0.0039] | yes |
| 16 | 55 pairs | a 0.0192, b 0.364, rms 0.00054; [+0.00062, +0.00344] and [+0.00008, +0.00250] | a 0.0192, b 0.364, rms 0.00054; [+0.00065, +0.00348] and [+0.00010, +0.00254] | number yes; not robust, see 2.4 |
| 17 | no pair with rho < 0.95 and gap < 0.005 | 0 | 0 of 55 | yes |
| 18 | mix from rank files the v1 way (z of ranks, ties at random) minus the z-mix | mean -0.0009; C3 +0.00023, S16 -0.00046, CatBoost -0.00179, XGBoost -0.00323, LLM -0.00270 | mean -0.00102 with another tie key; C3 +0.00007 ± 0.00034, S16 -0.00047 ± 0.00021, CatBoost -0.00187 ± 0.00061, XGBoost -0.00298 ± 0.00070, LLM -0.00268 ± 0.00064 (paired) | yes (tie-key noise 0.0002) |
| 19 | mix from a rank file the v2 way minus the z-mix | mean -0.0011, range 0.0000 .. -0.0031 | reproduced ONLY with ties resolved in the partner's order: mean -0.00110, +0.00006 .. -0.00305. Other tie-breaks in 2.1 | number yes, reading NO |
| 20 | "a 0.999 copy loses -0.0004 ± 0.0002 through rank ties" (S16, v1 way) | -0.00041 ± 0.00023 vs the better file | -0.00042 ± 0.00024 vs the better file; -0.00047 ± 0.00021 vs the z-mix (paired, 2.2 SE) | yes, at 2 SE |
| 21 | drift, cold-minus-warm offset vs the W8 file (score-z), 5 candidates x (val Tue, val Thu, Sat, Sun am, Sun pm, Mon, Tue, PUBLIC, PRIVATE) | table 1.6 | identical to 4 decimals in all 45 cells, e.g. W8TO -0.0094 ± 0.0004 / -0.0136 / -0.0162 / -0.0184 / -0.0174 / -0.0193 / -0.0210 ± 0.0003; W16 -0.0002 / -0.0001 / +0.0004 / +0.0009 / -0.0001 / +0.0002 / +0.0006 | yes |
| 22 | drift, Spearman / same top-1 / top-3 move, public and private | W16 0.9967 / 0.9966, 93.3% / 93.2%; W8TO 0.9953 / 0.9950, 90.7% / 90.6%; C3 0.9874 / 0.9873, 82.4% / 82.6% | identical (W16 top-3 move 0.184 / 0.182, C3 0.538 / 0.533) | yes |
| 23 | contrasts | W8P20 private - public -0.0009 ± 0.0002 (z -4.2); W8TO last day - val Tue -0.0116 ± 0.0005 (z -22.0) | identical | yes |
| 24 | known answer of the drift tool: cold offset of z(FT pool) - z(TO pool) of W8 | ft_redesign: +0.021 (val Tue) to +0.044 (last day) | +0.0212, +0.0294 (val Thu), +0.0344 (Sat), +0.0404 (Mon), +0.0438 (Tue 09-29) | yes |
| 25 | feed counts, public bound | 7,542 / 6,878 / 6,050 / 2,512 / 4,855 / 7,578 / 7,600; public 8,562 | identical; cold share 0.361 / 0.403 / 0.427 / 0.435 / 0.447 / 0.456 / 0.471 | yes |
| 26 | reader: 28 of 28 synthetic checks pass | `out/t1_reader_tests.txt` | 2 of theirs re-run (inverted: REFUSED, exit 3; NaN / inf: FAIL, DECISION NONE); 4 new files of mine: 3 partly corrupted files get PROCEED (2.3) | partly |

## 2. Issues by severity

### 2.1 HIGH: the numbers of verdict line 4 are a tie-break; the tie-free rank-only cost is smaller for our neural partners and real for less correlated pairs [M]

Mechanism: ours has rank a for item i and b for item j, the partner has them swapped. Restored on our shape, 0.5 z[a] + 0.5 z[b] equals 0.5 z[b] + 0.5 z[a] exactly. Share of val rows in such ties: 7-15% for the 7 neural partners, 2-3% for the 3 non-neural models. `mc_core.zfeed` divides by sd + 1e-9 and the shape is z-scored once more than our scores, which leaves a relative residue z2 / z1 - 1 = 1e-9 (1/s - 1) = +2.15e-12 for the W8 blend (s = per-feed sd of the raw blend, 0.998). That residue resolves 99.8% of the ties, in the partner's order (the resulting order equals "ties to the partner" in 99.98-99.99% of the rows; `out/v5_emulate.txt`).

Equal mix of W8 scores and a partner rank file restored on the W8 shape, minus the z-mix of the model scores (paired, 14,423 val feeds):

| partner | val gap | tied rows | ties -> OUR order | ties at random | ties -> PARTNER order (= the study's "v2 way") | ties left to the metric |
|---|---|---|---|---|---|---|
| C3 | +0.00064 | 9.4% | +0.00040 ± 0.00033 | -0.00000 ± 0.00033 | +0.00006 ± 0.00032 | -0.00490 ± 0.00031 |
| G3 | +0.00133 | 9.2% | -0.00002 ± 0.00035 | -0.00049 ± 0.00034 | -0.00032 ± 0.00032 | -0.00562 ± 0.00033 |
| Pall | +0.00269 | 8.5% | +0.00041 ± 0.00036 | -0.00072 ± 0.00035 | -0.00105 ± 0.00033 | -0.00586 ± 0.00034 |
| Cd | +0.00264 | 8.2% | +0.00049 ± 0.00037 | -0.00058 ± 0.00036 | -0.00106 ± 0.00036 | -0.00604 ± 0.00036 |
| S16kd | -0.00008 | 15.2% | -0.00016 ± 0.00023 | -0.00035 ± 0.00021 | -0.00020 ± 0.00020 | -0.00329 ± 0.00022 |
| GRU x6 | +0.00317 | 6.7% | +0.00017 ± 0.00039 | -0.00069 ± 0.00039 | -0.00098 ± 0.00039 | -0.00646 ± 0.00040 |
| base x4 | +0.00278 | 7.6% | +0.00060 ± 0.00037 | -0.00006 ± 0.00038 | -0.00067 ± 0.00039 | -0.00622 ± 0.00037 |
| CatBoost | +0.01964 | 2.6% | -0.00036 ± 0.00056 | -0.00121 ± 0.00056 | -0.00181 ± 0.00056 | -0.00586 ± 0.00055 |
| XGBoost | +0.03395 | 2.1% | -0.00057 ± 0.00062 | -0.00145 ± 0.00062 | -0.00193 ± 0.00062 | -0.00559 ± 0.00062 |
| LLM x2 | +0.02489 | 2.6% | -0.00148 ± 0.00058 | -0.00247 ± 0.00059 | -0.00305 ± 0.00060 | -0.00728 ± 0.00059 |
| mean of 10 (7 neural) | | | -0.00005 (+0.00027) | -0.00080 (-0.00041) | -0.00110 (-0.00060) | -0.00571 |

Control 1, roles swapped (ours = the weaker model, partner = W8 as a rank file): ties to the weaker file -0.00114 ± 0.00035 (Cd), -0.00087 ± 0.00039 (GRU), -0.00184 ± 0.00056 (CatBoost); ties to W8 +0.00037 ± 0.00036, +0.00029 ± 0.00039, -0.00033 ± 0.00056. The sign follows which file wins the ties, not which file is the rank file.

Control 2, the tie-free rank-only cost (`out/v7_noties.txt`): partner as a rank file restored on our shape minus the z-mix of the model scores at the SAME weight; w = weight of the partner; 0.00% tied rows at 0.4 and 0.6:

| ours + partner (ranks) | val rho | val gap | w 0.4 | w 0.6 | w 0.5, ties -> ours / random / -> partner |
|---|---|---|---|---|---|
| W8 + S16kd | 0.9989 | -0.00008 | -0.00025 ± 0.00020 | -0.00009 ± 0.00019 | -0.00016 / -0.00035 / -0.00020 (± 0.0002) |
| Pall + Cd | 0.9989 | -0.00005 | -0.00003 ± 0.00018 | -0.00002 ± 0.00020 | +0.00003 / -0.00018 / -0.00010 (± 0.0002) |
| C3 + G3 | 0.9986 | +0.00069 | +0.00022 ± 0.00024 | -0.00018 ± 0.00024 | +0.00037 / +0.00018 / -0.00001 (± 0.00025) |
| W8 + C3 | 0.9935 | +0.00064 | +0.00027 ± 0.00029 | - | +0.00040 / -0.00000 / +0.00006 (± 0.0003) |
| W8 + Pall | 0.9911 | +0.00269 | -0.00020 ± 0.00032 | - | +0.00041 / -0.00072 / -0.00105 (± 0.00035) |
| W8 + GRU x6 | 0.9828 | +0.00317 | -0.00022 ± 0.00036 | -0.00105 ± 0.00038 | +0.00017 / -0.00069 / -0.00098 (± 0.0004) |
| GRU x6 + base x4 | 0.9827 | -0.00039 | -0.00095 ± 0.00033 | -0.00050 ± 0.00038 | -0.00084 / -0.00097 / -0.00076 (± 0.00037) |
| base x4 + GRU x6 | 0.9827 | +0.00039 | -0.00027 ± 0.00035 | -0.00082 ± 0.00036 | -0.00052 / -0.00076 / -0.00062 (± 0.00037) |
| CatBoost + LLM x2 | 0.9296 | +0.00525 | -0.00007 ± 0.00040 | -0.00119 ± 0.00049 | +0.00092 / +0.00077 / +0.00028 (± 0.00046) |
| LLM x2 + CatBoost | 0.9296 | -0.00525 | -0.00058 ± 0.00044 | +0.00027 ± 0.00046 | +0.00024 / +0.00079 / +0.00094 (± 0.00047) |
| LLM x2 + XGBoost | 0.8972 | +0.00906 | -0.00029 ± 0.00049 | -0.00183 ± 0.00055 | -0.00017 / -0.00035 / -0.00050 (± 0.00055) |
| XGBoost + LLM x2 | 0.8972 | -0.00906 | -0.00118 ± 0.00049 | -0.00136 ± 0.00056 | -0.00109 / -0.00089 / -0.00074 (± 0.00055) |
| W8 + CatBoost | 0.8397 | +0.01964 | +0.00015 ± 0.00052 | -0.00122 ± 0.00056 | -0.00036 / -0.00121 / -0.00181 (± 0.00056) |

Reading:
- For the partners of W8 at the weight a weaker partner would get (0.4), a rank file costs nothing measurable: +0.0003, -0.0002, -0.0003 (S16), -0.0002, +0.0002, each within 1.3 SE. The study's -0.0011 mean for these partners is the tie-break. With more weight on the rank file (0.6) a cost appears: W8 + GRU -0.00105 ± 0.00038, W8 + CatBoost -0.00122 ± 0.00056.
- A tie-free cost does exist for less correlated pairs of similar strength: GRU + base -0.0003..-0.0010 in all four reads (0.8-2.9 SE each), LLM + XGBoost -0.0003..-0.0018 (0.6-3.3 SE); CatBoost + LLM is mixed (-0.0012..+0.0003 tie-free, +0.0002..+0.0009 at w 0.5). The reads of one pair share the feeds, they are not independent.
- [I] So for the partner a merge wants (rho about 0.90, equal strength) a rank file probably costs 0.0005-0.001 of the blend, which supports the study's advice; the evidence is from pairs at level 0.37-0.40, not from a strong partner, and not from the numbers the study quotes.

Consequence: the NOTE the tool prints for a rank-file partner ("0.0000..-0.0031 below the z-mix, mean -0.0011") describes the tie-break of that particular arithmetic. The study's sentence on the 0.999 copy (v1 way, ties named as the cause) is consistent with my reads: -0.00047 ± 0.00021 with random ties, -0.00025 ± 0.00020 and -0.00009 ± 0.00019 without ties. The lead's v1 `--blend` has the same structure with 16-17% tied rows broken by `make_sub`'s random key (row 18 of the table: 7 neural partners -0.0004 on average against the z-mix).

### 2.2 HIGH: the val check of the tool for a rank-file partner depends on the scale of our own scores [M]

- `merge_check_v2.py --ours subs/cand_W8TOkd_r3.parquet --ours_public 0.3940 --partner cache/NOTSUB_c3_test_ranks.parquet --partner_public 0.3940 --partner_val cache/NOTSUB_c3_val_ranks.parquet` (`out/tool_W8TO_vs_C3ranks_gap0.txt`): "equal mix 0.39600, vs the better file -0.00557 ± 0.00040 (z -13.78)", and gate 3 is read on it because w* = 0.5 at gap 0. My own computation: z-mix of the model scores +0.00004 ± 0.00040; rank partner with ties left tied -0.00573 ± 0.00040; ties to our order -0.00019 ± 0.00027. Cause: the W8TO scores have unit variance, the residue is 0, the ties stay exact and the metric scores them pessimistically.
- Same file, same partner (Pall ranks), our raw W8 scores multiplied by 1 / 3 / 0.3 (`out/v6_scale_flip.txt`): residue +2.15e-12 / -6.66e-10 / +2.34e-09, ties go to the partner / to us / to the partner, the equal-mix read is -0.00200 ± 0.00052 / -0.00055 ± 0.00030 / -0.00200 ± 0.00052. A 0.0015 swing from a change that carries no information.
- Scope: exact ties need w = 0.5 exactly, so gate 3 (read at w*) is hit only at a public gap of exactly 0 or with `--w eq`; the printed "equal mix" line and a file written by `--blend --w eq` from a rank-file partner are always hit [I for the written file: same arithmetic, `make_sub`'s random key decides only the 0.01-0.05% of rows the residue leaves tied].
- Fix (not applied, the study's files are read-only for me): for a rank-file partner resolve ties explicitly towards the file with the better public score (`np.lexsort((z_better, mix, fid))`), or refuse w = 0.5 exactly; never let `ndcg_per_feed` see the exact ties.

### 2.3 HIGH: the reader passes a partly corrupted file as the ideal partner [M]

Files in `nn_x/w12/next/merge_verify/cache/` (C3 file, partner public 0.393454):

| synthetic partner file | reader | rho the law gets | law on record | decision printed |
|---|---|---|---|---|
| 10% of the feeds negated | no FAIL, no WARN | 0.8035 | +0.00326 | PROCEED |
| 12% of the feeds with scores shuffled inside the feed | no FAIL, no WARN | 0.8745 | +0.00194 | PROCEED |
| 30% of the feeds negated | WARN only (mean Spearman 0.395) | 0.4201 | +0.01040 | PROCEED |
| all feeds negated (the study's T_12) | FAIL "INVERTED" | - | - | REFUSED |

The law rewards low correlation, so any damage the reader does not name looks like diversity at rho 0.80-0.87, exactly the profile a merge is looking for. The tool checks the mean per-feed Spearman only. A distribution check separates the cases cleanly (`out/v4_extra.txt`, test, feeds with M >= 20, share with per-feed Spearman below 0.3 against the W8 file): C3 0.000%, GRU 0.000%, LLM 0.033%, CatBoost 0.084%, XGBoost 0.447%; the three corrupted files 9.8%, 11.8%, 30.0%. Proposed line for the reader: FAIL when more than 1% of the feeds with M >= 20 have per-feed Spearman below 0.3 [threshold is my proposal from 5 genuine partners, not a rule on record].

### 2.4 MEDIUM

- **The 55-pair interval is carried by 3 pairs** [M]. Highest-leverage pairs of the fit: base x4 + CatBoost, XGBoost + LLM, GRU + CatBoost, CatBoost + LLM. Without the 3 pairs of two non-neural models (level 0.37-0.38): a 0.0140, b 0.330; bootstrap 5-95% of a [-0.0003, +0.0328]; gain at (0.90, 0) [-0.00003, +0.00328], at (0.90, 0.002) [-0.00049, +0.00237]. The 28 pairs among the 8 neural pools give a 0.0498, b 0.541; the 19 pairs with W8 or S16 give 0.0076 / 0.290; `reports/sol_ensemble.md` 2.5 itself lists 0.0130 (all 36 pairs) and 0.0615 (holdout). The diversity coefficient is known to a factor of about 6. The clause "+0.0001..+0.0034 if the 55-pair fit is accepted" and the same sentence in the tool's UNCERTAINTY line should not be read as "a gain is likely".
- **"Prediction interval" is a confidence band** of the fitted line under feed resampling. It contains neither the partner-to-partner residual (the study's leave-one-out rms 0.0005) nor the val-to-public transfer. The study says so in its limits; the tool's output does not.
- **Gate 3 is read on w\*, the study's own definition** of "the blend at the law's weight". For every on-disk partner w* = 0, so the gate line reads +0.00000 ± 0.00000 "no pass" and was never exercised at a non-trivial weight; the study's own check finds the weight model 0.0003-0.0004 too pessimistic at small weights. The threshold (2 paired SE) is unchanged. The lead has to say whether the equal mix (the object the law prices) or w* decides.
- **Gate 1 of bet 1** (partner public within 0.002 of ours) is not printed or checked by the tool.
- **`--blend` writes when the rule says "do NOT blend"** (WARN only) and does not ask for the val gate; v1 behaves the same.

### 2.5 LOW

- "17 pairs with gap < 0.01" is `sol_ensemble.md`'s number; `solvers/out/ensemble/exp7.json` holds 19 such pairs of 36 with the same coefficients. An inconsistency on record, not of the study.
- The drift flag has no A/A null (the study says so). My partial check: the W16-vs-W8 cold offset over the 5 test-day groups (+0.00037 ± 0.00029, +0.00085 ± 0.00041, -0.00009 ± 0.00030, +0.00021 ± 0.00024, +0.00055 ± 0.00024) has chi-square 4.75 on 4 df against a common mean, so the per-group SEs are not visibly too small [M, coarse].
- `cache/T_00_baseline.parquet` of the study is byte-identical to `subs/cand_Q2a8_ftbag3_memeq.parquet`: a valid submission file outside `subs/`. Harmless; delete it with the cache.
- A float64 score file with three values of 1e200 passes with overflow warnings and an unchanged result (0.9935); no FAIL line.

## 3. Audit

- **Rule applied as written**: `go = law on record - 0.0005 >= 0 and no reader FAIL`, coefficients 0.0186 / 0.378, rho = feed-mean Pearson (the definition of the fit; for a rank file the shape-restored value, biased +0.0002..+0.0020, at most 0.00004 in the law, towards not blending). Thresholds unchanged [code read].
- **Leak**: val labels enter only `ndcg_per_feed` (scoring of blends) and the two-coefficient refit, as in the lens. Cold = `it_hist_all_luo == 0`, a leave-user-out history count. No later snapshots of a user, no file from `data/feats_quarantine` or `archive/preds_quarantine` (the XGBoost file is `archive/preds/xgb_tr1m_ndcg_*`, the lens's own input) [code read].
- **SEs**: blend deltas and gaps are paired per feed; correlations are SE over feeds; drift contrasts are two independent feed groups. Correct for what they compare.
- **Read-only files** [M]: `nn_x/w5/lane.sh` 22:14:15, `nn_x/w5/train.py` 20:20:00, `src/merge_check.py` 21:48:36 (before the study started), nothing else under `src/` changed on 11-14; `data.py`, `finetune.py`, `cd_arms*.py` unchanged since 11-12 / 11-13; no new file in `subs/`, no `cand_MERGE*` anywhere. Files changed after 22:19 outside the study's folder: the lead's `CURRENT_STATUS.md` (22:20), training logs under `nn_x/abl/logs/`, one `feat_new/__pycache__` file at 22:22 (a compiled cache of `hz_common`; no script of the study imports it, `nn_x/w12/next/poplag/` does).
- **GPU** [M]: `nvidia-smi --query-compute-apps` lists 32 processes, all `train.py --variant ...` lanes started 17:32-22:44 UTC; none belongs to the study. The study's scripts contain no torch, no subprocess, no network call. Its five snippets without `CUDA_VISIBLE_DEVICES=""` (self-reported) cannot be checked after the fact, nor can its thread counts.
- **Cached arrays of the study**: I rebuilt `NOTSUB_c3_val_ranks.parquet` and `NOTSUB_c3_test_ranks.parquet` from `preds/ens_Q2a8s3_*`: identical in 100.0000% of rows; `NOTSUB_catboost_test_scores.parquet` equals `nn_x/cb26/data/e3_xm_yeti_test.npy` exactly. I used them only as inputs of the tool runs in 2.2.

## 4. Method

- `v_common.py`: own per-feed z (float64, no epsilon), own within-feed rank, own parquet-to-flat join, the partner list of `p4_partner.py`; the metric is `src/metrics.ndcg_per_feed`.
- `v1_calib.py`: correlation definitions for W8 vs C3, calibration rows, refit, two feed bootstraps (2,000 multinomial resamples), 55 pairs with leverage and subsets. Its section C prints a "v2 way" column in which the exact ties are left to the metric (about -0.005 for every partner): that is how the tie problem was found; the resolved numbers are in `v3_rank_ties.txt` and `v7_noties.txt`, and section C's swapped-role lines are superseded by them.
- `v2_drift.py`: per-feed Spearman, top-1, top-3 move from the files; cold offset from `preds/ens_*` (score-z) and from the files (rank-z); val days from `preds/ens_*_val.npy`.
- `v3_rank_ties.py`: the equal mix with exact ties, resolved four ways (`np.lexsort` with our z, the partner's z, a random key, or left to the metric), swapped roles, equal-strength pairs, w = 0.4.
- `v5_emulate.py`, `v6_scale_flip.py`: the study's arithmetic (z with sd + 1e-9, repeated) re-implemented to show which way its residue resolves the ties; `v4_extra.py`: truth for W8TO + C3 and the per-feed Spearman distributions; `v7_noties.py`: the rank-only cost at weights without ties and the chi-square of 2.5.
- The study's tool was run 7 times read-only (`-B -I`, no `--blend`) on its cache files and on my synthetic files.

## 5. Not checked

- `c1_calib.py` and `t1_reader_tests.py` were not re-run; 26 of the 28 reader cases were not repeated.
- Leave-one-partner-out ranges, the weight-model grid (T6), the interpolated public gap SE, the per-day rows of the tool for pairs other than W8 vs C3.
- `--blend` to disk (no submission file may be written); the statement about the written file in 2.2 is inferred from the arithmetic.
- Anything on the private side or about val-to-public transfer of the law (no labels).
- Thread counts and environment of the study's past processes.
- Whether a tie-break towards the better file is better than a random one on public: measured on val only, for partners weaker than W8.
- The rank-only cost for a STRONG partner at rho 0.90: no such model on disk; the tie-free reads at rho 0.90-0.93 are pairs of weak models.

## 6. Commands

```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
D=nn_x/w12/next/merge_verify; O=$D/out; M=nn_x/w12/next/merge
.venv/bin/python $D/v1_calib.py > $O/v1_calib.txt 2>&1          # 76 s
.venv/bin/python $D/v2_drift.py > $O/v2_drift.txt 2>&1          # 18 s
.venv/bin/python $D/v3_rank_ties.py > $O/v3_rank_ties.txt 2>&1  # 79 s
.venv/bin/python -B $D/v0_make_synth.py                           # the 4 synthetic partner files in $D/cache (seed 5)
.venv/bin/python -B $D/v4_extra.py > $O/v4_extra.txt 2>&1
.venv/bin/python -B $D/v5_emulate.py > $O/v5_emulate.txt 2>&1
.venv/bin/python -B $D/v6_scale_flip.py > $O/v6_scale_flip.txt 2>&1
.venv/bin/python -B $D/v7_noties.py > $O/v7_noties.txt 2>&1          # 60 s
.venv/bin/python -B -I $M/merge_check_v2.py --ours subs/cand_W8TOkd_r3.parquet --ours_public 0.3940 --partner $M/cache/NOTSUB_c3_test_ranks.parquet --partner_public 0.3940 --partner_val $M/cache/NOTSUB_c3_val_ranks.parquet > $O/tool_W8TO_vs_C3ranks_gap0.txt
for t in V_partial_inversion_10pct V_partial_inversion_30pct V_shuffled_12pct_feeds V_huge_values_3rows; do .venv/bin/python -B -I $M/merge_check_v2.py --partner $D/cache/$t.parquet --partner_public 0.393454 > $O/tool_$t.txt; done
for t in T_12_inverted_sign T_09_nan_inf; do .venv/bin/python -B -I $M/merge_check_v2.py --partner $M/cache/$t.parquet --partner_public 0.393454 > $O/tool_$t.txt; done
```

## 7. Files created

- Scripts: `<repo>/nn_x/w12/next/merge_verify/v_common.py`, `v0_make_synth.py`, `v1_calib.py`, `v2_drift.py`, `v3_rank_ties.py`, `v4_extra.py`, `v5_emulate.py`, `v6_scale_flip.py`, `v7_noties.py` in the same folder.
- Outputs: `<repo>/nn_x/w12/next/merge_verify/out/v1_calib.txt`, `v2_drift.txt`, `v3_rank_ties.txt`, `v4_extra.txt`, `v5_emulate.txt`, `v6_scale_flip.txt`, `v7_noties.txt`, `tool_*.txt` (7 runs of the study's tool).
- Cache (4 synthetic partner files, 74 MB, can be deleted; none is a candidate): `<repo>/nn_x/w12/next/merge_verify/cache/V_partial_inversion_10pct.parquet`, `V_partial_inversion_30pct.parquet`, `V_shuffled_12pct_feeds.parquet`, `V_huge_values_3rows.parquet`.
- This report: `<repo>/reports/NEXT_1115/merge_verify.md`.
